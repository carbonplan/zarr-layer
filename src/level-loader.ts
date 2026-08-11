import type * as zarr from 'zarrita'
import type { LevelRuntime } from './region-state'
import type { NormalizedSelector } from './types'

type ResolvedLevel = Pick<
  LevelRuntime,
  'zarrArray' | 'width' | 'height' | 'regionSize' | 'xyLimits'
> & { reusedArray: boolean }

/**
 * Why a load stopped.
 *
 * `superseded` is not a failure: a newer load (a zoom change, a selector
 * rebuild) took over, and its result is the one that counts. Callers waiting
 * on a level must retry rather than conclude there is none — which is the
 * whole reason this is reported instead of inferred from `activeLevel`.
 */
export type LevelLoadOutcome =
  | 'committed'
  | 'superseded'
  | 'failed'
  /** Index out of range, or non-zero on a single-level store. */
  | 'ignored'

export type LevelLoaderContext = {
  isMultiscale: () => boolean
  getLevelCount: () => number
  resolveArray: (levelIndex: number, reuse: boolean) => Promise<ResolvedLevel>
  buildSliceArgs: (
    selector: NormalizedSelector,
    array: zarr.Array<zarr.DataType>,
    coordLevelIndex: number
  ) => Promise<Pick<LevelRuntime, 'baseSliceArgs' | 'baseMultiValueDims'>>
  getSelector: () => NormalizedSelector
  isRemoved: () => boolean
  onCancelInflight: () => void
  onNewArrayCommitted: () => void
  invalidate: () => void
  getAssetLabel: (levelIndex: number) => string
}

export class LevelLoader {
  private loadToken = 0
  private desiredLevelIndex = 0
  private loadingLevelIndex: number | null = null
  private activeLevel: LevelRuntime | null = null
  private inflight: Promise<LevelLoadOutcome> | null = null

  constructor(private context: LevelLoaderContext) {}

  get active(): LevelRuntime | null {
    return this.activeLevel
  }
  get desiredIndex(): number {
    return this.desiredLevelIndex
  }
  set desiredIndex(levelIndex: number) {
    this.desiredLevelIndex = levelIndex
  }
  get loadingIndex(): number | null {
    return this.loadingLevelIndex
  }

  /**
   * Unified level load: handles initial load, zoom-driven switch, and
   * selector-driven slice-args rebuild. Builds a `LevelRuntime` off to
   * the side and swaps it into `activeLevel` atomically, so readers
   * never see a half-committed level.
   *
   * `reuseArray` reuses the current committed array/dims — used by
   * `setSelector` to rebuild slice args without refetching. `loadToken`
   * acts as a cancellation token: any load whose token is stale at
   * commit time drops its result.
   *
   * The returned promise settles when the load does, so callers outside the
   * render loop (`ensureActive`) can await a commit. The synchronous guards
   * live here rather than in `runLoad` so a deduped call hands back the
   * in-flight promise instead of an already-resolved one.
   */
  loadLevel(
    levelIndex: number,
    { reuseArray = false }: { reuseArray?: boolean } = {}
  ): Promise<LevelLoadOutcome> {
    if (this.context.isMultiscale() && this.context.getLevelCount() > 0) {
      if (levelIndex < 0 || levelIndex >= this.context.getLevelCount()) {
        return Promise.resolve('ignored')
      }
    } else if (levelIndex !== 0) {
      return Promise.resolve('ignored')
    }

    // Dedupe: an in-flight load for the same target is already on it.
    // A selector rebuild (`reuseArray`) intentionally supersedes.
    if (this.loadingLevelIndex === levelIndex && !reuseArray) {
      return this.inflight ?? Promise.resolve('ignored')
    }

    const pending = this.runLoad(levelIndex, reuseArray)
    this.inflight = pending
    const clear = () => {
      if (this.inflight === pending) this.inflight = null
    }
    // Settled both ways, and handled here so the bookkeeping chain can't
    // surface as an unhandled rejection alongside the caller's own.
    pending.then(clear, clear)
    return pending
  }

  /**
   * Await a committed level, loading the desired one if the render loop
   * hasn't. Returns null only when no level can be had: the load failed, or
   * the loader was disposed.
   *
   * Targets the render loop's own desired index so this can't fight it: the
   * level `update()` wants is the level `update()` would load, and the dedupe
   * in `loadLevel` folds the two onto one request.
   *
   * A zoom change mid-load displaces the load we are waiting on, which
   * finishes without committing. Retrying picks up whichever load replaced
   * it, so a camera move during startup doesn't read as "this store has no
   * levels".
   *
   * The retry is deliberately uncapped. A cap would turn a burst of camera
   * movement into a spurious "no level" for whoever is waiting, which is the
   * failure this exists to prevent. It cannot spin: every pass awaits a real
   * load, so it advances only as fast as loads settle, and it ends as soon as
   * one commits, one genuinely fails, or the renderer is disposed. Callers
   * that need to stop waiting sooner pass an AbortSignal.
   */
  async ensureActive(): Promise<LevelRuntime | null> {
    for (;;) {
      if (this.activeLevel) return this.activeLevel
      if (this.context.isRemoved()) return null
      const outcome = await this.loadLevel(this.desiredLevelIndex)
      if (outcome !== 'superseded') return this.activeLevel
    }
  }

  private async runLoad(
    levelIndex: number,
    reuseArray: boolean
  ): Promise<LevelLoadOutcome> {
    const token = ++this.loadToken
    // Snapshot the selector so we can detect a concurrent `setSelector`
    // that arrived after `buildSliceArgsForSelector` resolved; committing
    // old slice args with a new selector would leak stale data.
    const selectorSnapshot = this.context.getSelector()
    this.loadingLevelIndex = levelIndex

    // Cancel any in-flight region fetches — they were tied to the old
    // level's array/dims (or old selector) and can't be reused.
    this.context.onCancelInflight()

    try {
      const resolved = await this.context.resolveArray(levelIndex, reuseArray)
      const coordLevelIndex =
        this.activeLevel?.index ??
        this.loadingLevelIndex ??
        this.desiredLevelIndex ??
        0
      const { baseSliceArgs, baseMultiValueDims } =
        await this.context.buildSliceArgs(
          selectorSnapshot,
          resolved.zarrArray,
          coordLevelIndex
        )
      const targetStillDesired =
        reuseArray ||
        !this.context.isMultiscale() ||
        levelIndex === this.desiredLevelIndex

      // Drop on the floor if anything raced past us: a newer load (or
      // dispose) bumped the token, the zoom target moved on, or
      // `setSelector` replaced the selector we built slice args against.
      if (
        token !== this.loadToken ||
        this.context.isRemoved() ||
        this.context.getSelector() !== selectorSnapshot ||
        !targetStillDesired
      ) {
        this.context.invalidate()
        return 'superseded'
      }

      // Atomic commit — one reference swap replaces all per-level state.
      this.activeLevel = {
        index: levelIndex,
        zarrArray: resolved.zarrArray,
        width: resolved.width,
        height: resolved.height,
        regionSize: resolved.regionSize,
        xyLimits: resolved.xyLimits,
        baseSliceArgs,
        baseMultiValueDims,
      }

      // Don't clear the region cache on level changes — older-level
      // regions serve as fallback rendering while the new level's
      // regions load, and the LRU disposes them properly once they're no
      // longer protected. Bare `.clear()` here would leak WebGL resources.
      if (!resolved.reusedArray) this.context.onNewArrayCommitted()
      this.context.invalidate()
      return 'committed'
    } catch (err) {
      // A load that a newer one raced past is reported as superseded, not as
      // a failure: its error is about an attempt nobody is waiting on, and
      // calling it a failure would stop `ensureActive` retrying.
      if (token !== this.loadToken) return 'superseded'
      console.error(
        `Failed to load level ${this.context.getAssetLabel(levelIndex)}:`,
        err
      )
      return 'failed'
    } finally {
      if (token === this.loadToken) this.loadingLevelIndex = null
    }
  }

  dispose(): void {
    // Bump so any pending `loadLevel` drops its result on commit.
    this.loadToken++
    this.loadingLevelIndex = null
    this.activeLevel = null
    this.inflight = null
  }
}
