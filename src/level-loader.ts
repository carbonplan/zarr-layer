import type * as zarr from 'zarrita'
import type { LevelRuntime } from './region-state'
import type { NormalizedSelector } from './types'

type ResolvedLevel = Pick<
  LevelRuntime,
  'zarrArray' | 'width' | 'height' | 'regionSize' | 'xyLimits'
> & { reusedArray: boolean }

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
   */
  async loadLevel(
    levelIndex: number,
    { reuseArray = false }: { reuseArray?: boolean } = {}
  ): Promise<void> {
    if (this.context.isMultiscale() && this.context.getLevelCount() > 0) {
      if (levelIndex < 0 || levelIndex >= this.context.getLevelCount()) return
    } else if (levelIndex !== 0) {
      return
    }

    // Dedupe: an in-flight load for the same target is already on it.
    // A selector rebuild (`reuseArray`) intentionally supersedes.
    if (this.loadingLevelIndex === levelIndex && !reuseArray) return

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
        return
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
    } catch (err) {
      if (token === this.loadToken) {
        console.error(
          `Failed to load level ${this.context.getAssetLabel(levelIndex)}:`,
          err
        )
      }
    } finally {
      if (token === this.loadToken) this.loadingLevelIndex = null
    }
  }

  dispose(): void {
    // Bump so any pending `loadLevel` drops its result on commit.
    this.loadToken++
    this.loadingLevelIndex = null
    this.activeLevel = null
  }
}
