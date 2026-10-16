/**
 * @module region-renderer
 *
 * The unified renderer for every Zarr dataset. Reads the visible region of a
 * resolution level as chunk-sized sub-rectangles, reprojects each onto an
 * adaptive source→WGS84 mesh, and lets the GPU project to Mercator or ECEF.
 * Handles single arrays and resolution pyramids alike. Legacy slippy-map
 * pyramids reach this same path as arrays chunked at the tile size (see
 * ZarrStore). Automatic level selection is driven by map zoom.
 */

import * as zarr from 'zarrita'
import { MIN_SUBDIVISIONS, MAX_SUBDIVISIONS } from './constants'
import type {
  RenderContext,
  TileId,
  RegionRenderState,
  CustomShaderConfig,
} from './renderer-types'
import type {
  QueryGeometry,
  QueryLevel,
  QueryOptions,
  QueryResult,
} from './query/types'
import type {
  LoadingStateCallback,
  MapLike,
  NormalizedSelector,
  Selector,
  Bounds,
  DimIndicesProps,
  ResolutionLevel,
} from './types'
import { ZarrStore } from './zarr-store'
import { type MercatorBounds, type XYLimits } from './map-utils'
import { getBands } from './zarr-utils'
import type { ZarrRenderer, ShaderProgram } from './zarr-renderer'
import { renderTile, type TileRenderOptions } from './tile-renderer'
import {
  createProjectionContext,
  type ProjectionContext,
} from './projection-utils'
import type {
  LevelMeta,
  LevelRuntime,
  QueryLevelSnapshot,
  RegionState,
} from './region-state'
import {
  buildSliceArgsForSelector,
  type DimensionValuesCache,
} from './selector-resolution'
import { queryData as queryDataWithContext } from './query/data-query'
import {
  computeMercatorBoundsFromProjection,
  computeRegionMercatorBounds,
  getRegionBounds,
  getRegionSize,
  getRegionsInBounds,
  getVisibleRegions,
  selectLevelForZoom,
  type RegionCoordinate,
} from './region-math'
import {
  boundsIntersect,
  getTileBounds,
  getTileLonLatBounds,
} from './tile-utils'
import { MAX_CLIP_BOXES } from './shaders'
import {
  MAX_CACHED_REGIONS,
  RegionCache,
  hasRegionData,
  isRegionCpuReady,
  isRegionFailed,
  isRegionGpuReady,
  makeRegionKey,
  releaseRegionTextures,
} from './region-cache'
import { bandFormatForDtype, type BandFormat } from './band-format'
import { RegionFetcher } from './region-fetcher'
import { LevelLoader, type LevelLoadOutcome } from './level-loader'
import { wrapError } from './errors'
import { createHybridMesh } from './mesh-reprojector'
import {
  type RequestCanceller,
  type LoadingManager,
  type ChunkLoadingDebouncer,
  createRequestCanceller,
  MAX_ACTIVE_REGION_FETCHES,
  createLoadingManager,
  createChunkLoadingDebouncer,
  cancelAllRequests,
  dropQueuedFetches,
  hasActiveRequests,
  prioritizeQueuedFetches,
  setLoadingCallback as setLoadingCallbackUtil,
  emitLoadingState as emitLoadingStateUtil,
} from './region-utils'
import {
  ensureRegionGpuResources,
  setupBandTextureUniforms,
} from './render-helpers'
import { renderRegion, type RenderableRegion } from './renderable-region'

/** A tile the map draws, with the map zoom its pixel density matches. */
export interface TileRequest {
  tileId: TileId
  zoom: number
  /** Extend a polar edge tile to the pole: the globe draws there. */
  poles?: boolean
}

/**
 * Regions one update may want in total. A dataset with no coarser level maps
 * a far tile onto many regions of its only level; wanting more than the
 * cache can hold would only make it thrash, so selection stops there,
 * nearest tiles first.
 */
const MAX_WANTED_REGIONS = MAX_CACHED_REGIONS

const tileKey = (tile: TileId) => `${tile.z}/${tile.x}/${tile.y}`

/**
 * Merge boxes that share a row (same y extent) and touch along x into one,
 * so a block of n by m finer regions costs m clip boxes rather than n by m.
 */
function mergeRows(boxes: MercatorBounds[]): number[][] {
  const rows = new Map<string, MercatorBounds[]>()
  for (const box of boxes) {
    const row = `${box.y0}:${box.y1}`
    const list = rows.get(row)
    if (list) list.push(box)
    else rows.set(row, [box])
  }
  const merged: number[][] = []
  for (const list of rows.values()) {
    list.sort((a, b) => a.x0 - b.x0)
    let current = [list[0].x0, list[0].y0, list[0].x1, list[0].y1]
    for (let i = 1; i < list.length; i++) {
      const box = list[i]
      if (box.x0 <= current[2] + 1e-9) {
        current[2] = Math.max(current[2], box.x1)
      } else {
        merged.push(current)
        current = [box.x0, box.y0, box.x1, box.y1]
      }
    }
    merged.push(current)
  }
  return merged
}

export class RegionRenderer {
  isMultiscale: boolean = false

  private levelLoader: LevelLoader

  private get activeLevel(): LevelRuntime | null {
    return this.levelLoader.active
  }
  private get desiredLevelIndex(): number {
    return this.levelLoader.desiredIndex
  }
  private set desiredLevelIndex(levelIndex: number) {
    this.levelLoader.desiredIndex = levelIndex
  }
  private get loadingLevelIndex(): number | null {
    return this.levelLoader.loadingIndex
  }

  // Bounds
  private mercatorBounds: MercatorBounds | null = null

  // Store and metadata
  private zarrStore: ZarrStore
  private variable: string
  private selector: NormalizedSelector
  private bandNames: string[] = []
  private invalidate: () => void
  private dimIndices: DimIndicesProps = {}
  private xyLimits: XYLimits | null = null
  private latIsAscending: boolean = true

  // Multi-level support
  private levels: ResolutionLevel[] = []
  private levelMetadataFetched: Set<number> = new Set() // Tracks which levels have had metadata fetched
  private projection: ProjectionContext = createProjectionContext({
    crs: 'EPSG:4326',
    proj4def: null,
    xyLimits: null,
  })

  // Loading state
  private isRemoved: boolean = false
  private _antimeridianWarnings: Set<string> = new Set()

  // Shared state managers
  private requestCanceller: RequestCanceller
  private loadingManager: LoadingManager = createLoadingManager()
  private loadingDebouncer: ChunkLoadingDebouncer = createChunkLoadingDebouncer(
    this.loadingManager
  )

  // Dimension values cache (supports numeric and string coordinate arrays)
  private dimensionValues: DimensionValuesCache = {}
  // Deduplicate coordinate-read warnings across loads.
  private warnedCoordDimensions: Set<string> = new Set()

  // Region-based loading (for multi-level datasets with chunking/sharding)
  // Single unified cache with LRU eviction - keys include level index (e.g., "2:0,0")
  private regionCache = new RegionCache()
  // Regions wanted by the last update, per level: fetched, protected and
  // drawn. Each tile the map draws reads the level matching its zoom, so
  // a pitched view wants coarser levels in the distance than near the camera.
  private wantedRegions: Map<number, RegionCoordinate[]> = new Map()
  private wantedKeys: Set<string> = new Set()
  // Levels tiles asked for that are still opening, and those tiles: part
  // of the view has no regions to want yet, so it is not covered.
  private pendingLevels: Set<number> = new Set()
  private pendingTiles: TileId[] = []
  // The map's own tile cover for the frame, when the map exposes it.
  private coveringTiles: TileRequest[] | null = null
  // Tiles the draped render hooks were asked to draw, which on terrain reach
  // past the sea-level cover. Kept in two generations keyed by the layer's
  // terrain revision: a revision change redraws every visible tile, so the
  // current generation refills with the whole visible set while the previous
  // one still covers tiles drawn since. Older tiles drop out.
  private drawnTiles: Map<string, TileRequest> = new Map()
  private previousDrawnTiles: Map<string, TileRequest> = new Map()
  private drawnTilesRevision: number | null = null
  private tileRegionsMemo: Map<string, RegionCoordinate[]> = new Map()
  private lastViewportHash: string = ''
  private selectorVersion: number = 0 // Incremented on selector change to track stale regions

  // Cached WebGL context for use in setSelector
  private cachedGl: WebGL2RenderingContext | null = null
  // Whether the active shader samples per-band textures instead of the main
  // texture. Owned by ZarrLayer, which decides the shader variant.
  private rendersFromBandTextures: boolean = false
  // Track current projection for Mapbox's direct globe render path.
  private isGlobeProjection: boolean = false
  // Fixed data scale for normalization (set at initialization, passed from ZarrLayer)
  private fixedDataScale: number = 1
  private minRegionSize: number

  constructor(
    store: ZarrStore,
    variable: string,
    selector: NormalizedSelector,
    invalidate: () => void,
    fixedDataScale: number = 1,
    minRegionSize: number = 0,
    maxRegionFetches: number = MAX_ACTIVE_REGION_FETCHES
  ) {
    this.minRegionSize = minRegionSize
    this.requestCanceller = createRequestCanceller(maxRegionFetches)
    this.zarrStore = store
    this.variable = variable
    this.selector = selector
    this.bandNames = getBands(variable, selector)
    this.invalidate = invalidate
    this.fixedDataScale = fixedDataScale
    this.levelLoader = new LevelLoader({
      isMultiscale: () => this.isMultiscale,
      getLevelCount: () => this.levels.length,
      resolveArray: async (levelIndex, reuse) => {
        const existing = this.activeLevel
        const canReuseArray =
          reuse && existing !== null && existing.index === levelIndex
        if (canReuseArray) {
          return {
            zarrArray: existing.zarrArray,
            width: existing.width,
            height: existing.height,
            regionSize: existing.regionSize,
            xyLimits: existing.xyLimits,
            reusedArray: true,
          }
        }
        return { ...(await this.openLevel(levelIndex)), reusedArray: false }
      },
      buildSliceArgs: async (selectorSnapshot, array, coordLevelIndex) => {
        const { sliceArgs, multiValueDims } =
          await this.buildSliceArgsForSelector(
            selectorSnapshot,
            {
              includeSpatialSlices: false,
              trackMultiValue: true,
              array,
            },
            coordLevelIndex
          )
        return {
          baseSliceArgs: sliceArgs,
          baseMultiValueDims: multiValueDims,
        }
      },
      getSelector: () => this.selector,
      isRemoved: () => this.isRemoved,
      onCancelInflight: () => {
        // Queued fetches have no controller yet but would still start
        // against the level being replaced.
        if (hasActiveRequests(this.requestCanceller)) {
          cancelAllRequests(this.requestCanceller)
          this.loadingDebouncer.hide()
        }
      },
      onNewArrayCommitted: () => {
        this.resetVisibleRegions()
        // A freshly opened level is a new attempt at its regions.
        this.regionCache.clearFailures()
      },
      onLoadErrorChange: (error) => {
        this.loadingManager.error = error
        this.emitLoadingState()
      },
      invalidate: this.invalidate,
      getAssetLabel: (levelIndex) =>
        this.isMultiscale
          ? this.levels[levelIndex]?.asset ?? String(levelIndex)
          : 'single-level',
    })
  }

  async initialize(): Promise<void> {
    this.loadingManager.metadataLoading = true
    this.emitLoadingState()

    try {
      const desc = this.zarrStore.describe()
      this.dimIndices = desc.dimIndices
      this.xyLimits = desc.xyLimits
      this.latIsAscending = desc.latIsAscending
      // Cache transformers once for reuse (major performance optimization)
      this.projection = createProjectionContext({
        crs: desc.crs,
        proj4def: desc.proj4,
        xyLimits: this.xyLimits,
      })

      // Check if this is a multi-level dataset
      if (desc.resolutionLevels && desc.resolutionLevels.length > 0) {
        this.levels = desc.resolutionLevels
        this.isMultiscale = true
        // Ensure all levels have shape (required for level selection)
        // This only fetches levels where consolidated metadata was incomplete
        await this.ensureAllLevelShapes()
        // Don't load level data yet — `update()` will call `loadLevel`
        // once we know the actual zoom level. Avoids loading low-res then
        // immediately switching to high-res.
      } else {
        this.isMultiscale = false
        // Single-level dataset — commit the level eagerly so first render
        // doesn't wait for another tick of `update()`.
        await this.loadLevel(0)
      }

      if (this.xyLimits) {
        if (this.projection.def) {
          this.mercatorBounds = this.computeMercatorBoundsFromProjection()
        }
      } else {
        console.warn('RegionRenderer: No XY limits found')
      }
    } finally {
      this.loadingManager.metadataLoading = false
      this.emitLoadingState()
    }
  }

  /**
   * Open a level's array and derive its pixel dimensions. Shared by the
   * render loop's level commit and by query-local level reads; the store
   * caches array handles, so opening the same level twice is cheap.
   */
  private async openLevel(levelIndex: number) {
    const zarrArray = this.isMultiscale
      ? await (async () => {
          await this.ensureLevelMetadata(levelIndex)
          return this.zarrStore.getLevelArray(this.levels[levelIndex].asset)
        })()
      : await this.zarrStore.getArray()
    const width = zarrArray.shape[this.dimIndices.lon.index]
    const height = zarrArray.shape[this.dimIndices.lat.index]
    return {
      zarrArray,
      width,
      height,
      regionSize: this.getRegionSize(zarrArray) ?? [height, width],
      xyLimits: this.isMultiscale
        ? this.levels[levelIndex]?.xyLimits
        : undefined,
    }
  }

  /**
   * Lazily ensure metadata for a specific level is loaded.
   * Fetch per-level zarr.json if:
   * - We haven't already attempted a fetch for this level, AND
   * - Any of dtype/scaleFactor/addOffset are missing (consolidated metadata incomplete)
   */
  private async ensureLevelMetadata(levelIndex: number): Promise<void> {
    const level = this.levels[levelIndex]
    if (!level) {
      return
    }

    // Skip if we've already attempted a fetch for this level
    if (this.levelMetadataFetched.has(levelIndex)) {
      return
    }

    // Skip if we have dtype, scaleFactor, AND addOffset from consolidated metadata
    // (indicates complete metadata - no need to fetch)
    if (
      level.dtype !== undefined &&
      level.scaleFactor !== undefined &&
      level.addOffset !== undefined
    ) {
      return
    }

    // Mark as fetched before async operation to prevent duplicate fetches
    this.levelMetadataFetched.add(levelIndex)

    try {
      const meta = await this.zarrStore.getResolutionLevelMetadata(level.asset)
      level.shape = meta.shape
      level.chunks = meta.chunks
      // Only set scaleFactor/addOffset if defined - leave undefined for dataset-level fallback
      if (meta.scaleFactor !== undefined) {
        level.scaleFactor = meta.scaleFactor
      }
      if (meta.addOffset !== undefined) {
        level.addOffset = meta.addOffset
      }
      level.fillValue = meta.fillValue
      level.dtype = meta.dtype
    } catch (err) {
      console.warn(`Failed to load metadata for level ${level.asset}:`, err)
      // Already marked as fetched - won't retry
    }
  }

  /**
   * Ensure all levels have shape data (required for level selection).
   * Only fetches metadata for levels where consolidated metadata was incomplete.
   * This runs during initialization to enable proper zoom-based level selection.
   */
  private async ensureAllLevelShapes(): Promise<void> {
    const levelsNeedingShape = this.levels
      .map((level, index) => ({ level, index }))
      .filter(({ level }) => !level.shape)

    if (levelsNeedingShape.length === 0) {
      return // All shapes available from consolidated metadata
    }

    // Fetch metadata for levels missing shape (in parallel)
    await Promise.all(
      levelsNeedingShape.map(async ({ level, index }) => {
        // Skip if already fetched by another path
        if (this.levelMetadataFetched.has(index)) {
          return
        }
        this.levelMetadataFetched.add(index)

        try {
          const meta = await this.zarrStore.getResolutionLevelMetadata(
            level.asset
          )
          level.shape = meta.shape
          level.chunks = meta.chunks
          if (meta.scaleFactor !== undefined) {
            level.scaleFactor = meta.scaleFactor
          }
          if (meta.addOffset !== undefined) {
            level.addOffset = meta.addOffset
          }
          level.fillValue = meta.fillValue
          level.dtype = meta.dtype
        } catch (err) {
          console.warn(`Failed to load shape for level ${level.asset}:`, err)
        }
      })
    )
  }

  private getRegionSize(
    array: zarr.Array<zarr.DataType>
  ): [number, number] | null {
    return getRegionSize(array, this.dimIndices, this.minRegionSize)
  }

  /**
   * Clear region cache and dispose WebGL resources.
   */
  private clearRegionCache(
    gl: WebGL2RenderingContext | WebGLRenderingContext
  ): void {
    this.regionCache.clear(gl)
    this.lastViewportHash = ''
  }

  /**
   * Evict oldest regions when cache exceeds limit (LRU eviction).
   * Uses Map iteration order (oldest first).
   * Never evicts currently visible regions.
   */
  private evictOldRegions(gl: WebGL2RenderingContext): void {
    this.regionCache.evict(gl)
  }

  private getVisibleRegions(
    map: MapLike
  ): Array<{ regionX: number; regionY: number }> {
    return getVisibleRegions({
      map,
      xyLimits: this.xyLimits,
      levelMeta: this.activeLevel,
      projection: this.projection,
      latIsAscending: this.latIsAscending,
    })
  }

  /**
   * Create a region key that includes level index for unified caching.
   */
  private makeRegionKey(
    levelIndex: number,
    regionX: number,
    regionY: number
  ): string {
    return makeRegionKey(levelIndex, regionX, regionY)
  }

  /**
   * Get uniforms for rendering with scale/offset disabled.
   * Per-level scale/offset is applied in JS while a region is fetched, so the
   * shader is told to skip its own scale/offset application.
   */
  private getUniformsForRender(contextUniforms: RenderContext['uniforms']) {
    return {
      ...contextUniforms,
      scaleFactor: 1.0,
      offset: 0.0,
    }
  }

  /**
   * Whether every wanted region is drawable right now. The decision releases
   * the fallbacks, both from the render list and from eviction protection,
   * so CPU-side readiness isn't enough.
   */
  private wantedCovered(): boolean {
    if (this.wantedKeys.size === 0 || this.pendingLevels.size > 0) return false
    const requiredBands = this.requiredBands()
    const bandFormat = this.bandFormat()
    for (const key of this.wantedKeys) {
      const region = this.regionCache.get(key)
      // A failed region is settled: nothing more will arrive for it, so it
      // must not hold the previous level's fallbacks on screen forever.
      if (region && isRegionFailed(region, this.selectorVersion)) continue
      if (!region || !isRegionGpuReady(region, requiredBands, bandFormat)) {
        return false
      }
    }
    return true
  }

  /** The map's tile cover for this frame, or null when it has none to give. */
  setCoveringTiles(tiles: TileRequest[] | null): void {
    this.coveringTiles = tiles
  }

  /**
   * Record a tile a draped render hook was asked to draw, so the next update
   * wants the regions under it at the level matching its zoom. `revision` is
   * the layer's terrain revision at draw time; a new revision starts a new
   * generation. Returns true when the tile needs regions that are neither
   * loaded nor queued, so the caller can ask for a frame: the map runs the
   * update before it draws tiles, and schedules nothing after a tile redraw.
   */
  noteDrawnTile(tile: TileRequest, revision: number): boolean {
    if (revision !== this.drawnTilesRevision) {
      this.previousDrawnTiles = this.drawnTiles
      this.drawnTiles = new Map()
      this.drawnTilesRevision = revision
    }
    const key = tileKey(tile.tileId)
    this.previousDrawnTiles.delete(key)
    this.drawnTiles.set(key, tile)

    const levelIndex = this.selectLevelForZoom(tile.zoom)
    const runtime = this.levelLoader.runtime(levelIndex)
    if (!runtime) return true
    return this.tileRegions(levelIndex, runtime, tile.tileId).some((region) => {
      const cached = this.regionCache.get(
        this.makeRegionKey(levelIndex, region.regionX, region.regionY)
      )
      return !cached || (!hasRegionData(cached) && !cached.loading)
    })
  }

  clearDrawnTiles(): void {
    this.drawnTiles = new Map()
    this.previousDrawnTiles = new Map()
    this.drawnTilesRevision = null
  }

  /** Every tile the map is drawing, most detailed (nearest) first. */
  private requestedTiles(): TileRequest[] {
    const byKey = new Map<string, TileRequest>()
    const add = (tile: TileRequest) => {
      const key = tileKey(tile.tileId)
      if (!byKey.has(key)) byKey.set(key, tile)
    }
    for (const tile of this.coveringTiles ?? []) add(tile)
    for (const tile of this.drawnTiles.values()) add(tile)
    for (const tile of this.previousDrawnTiles.values()) add(tile)
    return [...byKey.values()].sort((a, b) => b.zoom - a.zoom)
  }

  /** Regions of a level under a tile; the tile grid and levels never move. */
  private tileRegions(
    levelIndex: number,
    runtime: LevelRuntime,
    tileId: TileId,
    toPoles = false
  ): RegionCoordinate[] {
    const key = `${levelIndex}:${toPoles ? 'p' : 'm'}:${tileKey(tileId)}`
    const memo = this.tileRegionsMemo.get(key)
    if (memo) return memo
    if (this.tileRegionsMemo.size > 4096) this.tileRegionsMemo.clear()
    const regions = getRegionsInBounds({
      bounds: getTileLonLatBounds(tileId, { toPoles }),
      xyLimits: this.xyLimits,
      levelMeta: runtime,
      projection: this.projection,
      latIsAscending: this.latIsAscending,
    })
    this.tileRegionsMemo.set(key, regions)
    return regions
  }

  /**
   * The regions to want this update, per level. With a tile cover, each tile
   * reads the level matching its zoom; a level not yet open is opened for
   * the next update. Without one (Mapbox without terrain), the viewport is
   * read at the active level.
   */
  private selectWantedRegions(map: MapLike): Map<number, RegionCoordinate[]> {
    const wanted = new Map<number, RegionCoordinate[]>()
    this.pendingLevels = new Set()
    this.pendingTiles = []
    const tiles = this.requestedTiles()
    if (tiles.length === 0) {
      const active = this.activeLevel
      if (active) wanted.set(active.index, this.getVisibleRegions(map))
      return wanted
    }
    const perTile: Array<{
      levelIndex: number
      runtime: LevelRuntime
      regions: RegionCoordinate[]
    }> = []
    let total = 0
    for (const tile of tiles) {
      const levelIndex = this.selectLevelForZoom(tile.zoom)
      const runtime = this.levelLoader.runtime(levelIndex)
      if (!runtime) {
        void this.levelLoader.ensureRuntime(levelIndex)
        this.pendingLevels.add(levelIndex)
        this.pendingTiles.push(tile.tileId)
        continue
      }
      const regions = this.tileRegions(
        levelIndex,
        runtime,
        tile.tileId,
        tile.poles
      )
      perTile.push({ levelIndex, runtime, regions })
      total += regions.length
    }
    // Nearest tiles come first already. When the cap will bite, order each
    // tile's regions by distance from the map center too, so a single tile
    // spanning many regions cannot spend the budget on its far corner.
    const truncating = total > MAX_WANTED_REGIONS
    const seen = new Set<string>()
    for (const { levelIndex, runtime, regions } of perTile) {
      const ordered = truncating
        ? this.nearestCenterFirst(regions, runtime, map)
        : regions
      for (const region of ordered) {
        const key = this.makeRegionKey(
          levelIndex,
          region.regionX,
          region.regionY
        )
        if (seen.has(key)) continue
        if (seen.size >= MAX_WANTED_REGIONS) return wanted
        seen.add(key)
        let list = wanted.get(levelIndex)
        if (!list) {
          list = []
          wanted.set(levelIndex, list)
        }
        list.push(region)
      }
    }
    return wanted
  }

  /** Regions ordered by their distance from the map center, in degrees. */
  private nearestCenterFirst(
    regions: RegionCoordinate[],
    runtime: LevelRuntime,
    map: MapLike
  ): RegionCoordinate[] {
    const center = map.getCenter?.()
    const bounds = map.getBounds?.()?.toArray?.()
    const lng = center?.lng ?? (bounds ? (bounds[0][0] + bounds[1][0]) / 2 : 0)
    const lat = center?.lat ?? (bounds ? (bounds[0][1] + bounds[1][1]) / 2 : 0)
    const inverse = this.projection.toWGS84?.inverse
    const distance = (region: RegionCoordinate) => {
      const b = this.getRegionBounds(region.regionX, region.regionY, runtime)
      const xMid = (b.xMin + b.xMax) / 2
      const yMid = (b.yMin + b.yMax) / 2
      const [lon, latitude] = inverse ? inverse(xMid, yMid) : [xMid, yMid]
      let dLon = ((lon - lng + 540) % 360) - 180
      if (!Number.isFinite(dLon)) dLon = 180
      const dLat = Number.isFinite(latitude) ? latitude - lat : 180
      return dLon * dLon + dLat * dLat
    }
    return regions
      .map((region) => ({ region, d: distance(region) }))
      .sort((a, b) => a.d - b.d)
      .map(({ region }) => region)
  }

  /**
   * Areas (normalized Mercator) still waiting for wanted data: wanted regions
   * not yet drawable, and tiles whose level is still opening. Fallbacks
   * overlapping them are kept until the replacement can be drawn.
   */
  private pendingAreas(): Array<{
    x0: number
    y0: number
    x1: number
    y1: number
  }> {
    const areas: Array<{ x0: number; y0: number; x1: number; y1: number }> = []
    const requiredBands = this.requiredBands()
    const bandFormat = this.bandFormat()
    for (const [levelIndex, regions] of this.wantedRegions) {
      const runtime = this.levelLoader.runtime(levelIndex)
      if (!runtime) continue
      for (const region of regions) {
        const cached = this.regionCache.get(
          this.makeRegionKey(levelIndex, region.regionX, region.regionY)
        )
        if (cached && isRegionGpuReady(cached, requiredBands, bandFormat)) {
          continue
        }
        areas.push(
          cached?.mercatorBounds ??
            this.computeRegionMercatorBounds(
              this.getRegionBounds(region.regionX, region.regionY, runtime)
            )
        )
      }
    }
    for (const tileId of this.pendingTiles) areas.push(getTileBounds(tileId))
    return areas
  }

  /**
   * For each drawn region, the bounds of finer regions drawn after it that
   * overlap it, as fragment clip boxes: each pixel then shows exactly one
   * level, the finest drawn there, whatever the layer opacity. More than
   * MAX_CLIP_BOXES overlaps keep the largest.
   */
  private clipBoxesFor(regions: RegionState[]): Map<string, Float32Array> {
    const clips = new Map<string, Float32Array>()
    for (let i = 0; i < regions.length; i++) {
      const region = regions[i]
      const bounds = region.mercatorBounds
      if (!bounds) continue
      const width = this.levelWidth(region.levelIndex)
      const finer: MercatorBounds[] = []
      for (let j = i + 1; j < regions.length; j++) {
        const fb = regions[j].mercatorBounds
        if (!fb || this.levelWidth(regions[j].levelIndex) <= width) continue
        if (boundsIntersect(bounds, fb)) finer.push(fb)
      }
      if (finer.length === 0) continue
      const boxes = mergeRows(finer).map((box) => ({
        box,
        area:
          (Math.min(bounds.x1, box[2]) - Math.max(bounds.x0, box[0])) *
          (Math.min(bounds.y1, box[3]) - Math.max(bounds.y0, box[1])),
      }))
      if (boxes.length > MAX_CLIP_BOXES) {
        boxes.sort((a, b) => b.area - a.area)
        boxes.length = MAX_CLIP_BOXES
      }
      clips.set(region.key, new Float32Array(boxes.flatMap((b) => b.box)))
    }
    return clips
  }

  /** Pixel width of a level, the order coarse-to-fine drawing sorts by. */
  private levelWidth(levelIndex: number): number {
    const runtime = this.levelLoader.runtime(levelIndex)
    if (runtime) return runtime.width
    const shape = this.levels[levelIndex]?.shape
    return shape?.[this.dimIndices.lon?.index ?? shape.length - 1] ?? 0
  }

  /**
   * The band textures the active shader samples, or undefined when it reads
   * the main texture. Readiness and uploads must agree on this.
   */
  private requiredBands(): string[] | undefined {
    return this.rendersFromBandTextures ? this.bandNames : undefined
  }

  /** Storage format the band sampler reads, set by the active level's dtype. */
  private bandFormat(): BandFormat {
    return bandFormatForDtype(this.activeLevel?.zarrArray.dtype)
  }

  /** The layer's shader config, told which band sampler type to declare. */
  private shaderConfigForLevel(
    config: CustomShaderConfig | undefined
  ): CustomShaderConfig | undefined {
    if (!config) return config
    const bandFormat = this.bandFormat()
    if (config.bandFormat === bandFormat) return config
    return { ...config, bandFormat }
  }

  /**
   * Get drawable regions: the wanted regions plus fallbacks if needed, which
   * are regions no longer wanted but still protected, kept from before a
   * level or tile change until what is wanted can be drawn in full. Uploads
   * the wanted regions first so coverage is judged on what can actually be
   * drawn — regions that displace their fallbacks and then fail to upload
   * would leave the viewport blank. Every returned region is GPU-ready, and
   * coarser levels come first so finer data draws over them where both exist.
   */
  private getLoadedRegions(gl: WebGL2RenderingContext): RegionState[] {
    const requiredBands = this.requiredBands()
    const bandFormat = this.bandFormat()
    const drawable = (region: RegionState | undefined): region is RegionState =>
      !!region &&
      isRegionCpuReady(region) &&
      ensureRegionGpuResources(gl, region, requiredBands, bandFormat)

    const wanted: RegionState[] = []
    for (const key of this.wantedKeys) {
      const region = this.regionCache.get(key)
      if (drawable(region)) wanted.push(region)
    }

    const regions = wanted
    if (!this.wantedCovered()) {
      const fallbacks: RegionState[] = []
      for (const region of this.regionCache.values()) {
        if (this.wantedKeys.has(region.key)) continue
        if (!this.regionCache.isProtected(region.key)) continue
        if (drawable(region)) fallbacks.push(region)
      }
      regions.unshift(...fallbacks)
    }
    return regions.sort(
      (a, b) => this.levelWidth(a.levelIndex) - this.levelWidth(b.levelIndex)
    )
  }

  private getRegionBounds(
    regionX: number,
    regionY: number,
    levelMeta: LevelMeta
  ): { xMin: number; xMax: number; yMin: number; yMax: number } {
    return getRegionBounds({
      regionX,
      regionY,
      levelMeta,
      xyLimits: this.xyLimits,
      latIsAscending: this.latIsAscending,
    })
  }

  /**
   * Create geometry (vertex positions and tex coords) for a region.
   * Uses the source-projected adaptive mesh path for every supported CRS.
   */
  private createRegionGeometry(
    regionX: number,
    regionY: number,
    region: RegionState
  ): void {
    // Guard: can't create geometry without dimension info
    if (!region.levelMeta) return

    // Defensive reset: meshBounds is only set by the geometry producer below.
    // Ensures it's not stale if geometry is recreated.
    region.meshBounds = null
    region.indexArr = null
    // Whatever this produces is newer than the buffers already on the GPU.
    region.geometryUploaded = false

    if (
      !this.projection.def ||
      !this.projection.to4326 ||
      !this.projection.toMercator
    ) {
      return
    }

    const geoBounds = this.getRegionBounds(regionX, regionY, region.levelMeta)

    // Recomputed alongside the mesh: both derive from geoBounds, so keeping a
    // previously computed value would pair a new mesh with old shader bounds.
    region.mercatorBounds = this.computeRegionMercatorBounds(geoBounds)

    // Generate a source-projected mesh via proj4. CPU transforms source CRS to
    // WGS84, then encodes region-local Mercator deltas for the shader.
    const centerX = (geoBounds.xMin + geoBounds.xMax) / 2
    const centerY = (geoBounds.yMin + geoBounds.yMax) / 2
    const samplePoints = [
      this.projection.to4326.forward(geoBounds.xMin, geoBounds.yMin),
      this.projection.to4326.forward(geoBounds.xMax, geoBounds.yMin),
      this.projection.to4326.forward(geoBounds.xMin, geoBounds.yMax),
      this.projection.to4326.forward(geoBounds.xMax, geoBounds.yMax),
      this.projection.to4326.forward(centerX, centerY),
    ]
    const validLats = samplePoints
      .map((p) => p[1])
      .filter((lat) => isFinite(lat))
    const latSpan =
      validLats.length > 0 ? Math.max(...validLats) - Math.min(...validLats) : 0

    // WGS84 longitude span of this chunk, paired with latSpan to size the mesh
    // below. EPSG:4326's source X is longitude in degrees and a 0-360 store
    // keeps its unwrapped span there, so use the source span; forward-projecting
    // would fold it via adjust_lon and understate it. Other projections have no
    // such direct measure, so use the sampled WGS84 longitudes.
    let lonSpan: number
    if (this.projection.kind === 'epsg4326') {
      lonSpan = Math.min(360, Math.abs(geoBounds.xMax - geoBounds.xMin))
    } else {
      const validLons = samplePoints
        .map((p) => p[0])
        .filter((lon) => isFinite(lon))
      lonSpan =
        validLons.length > 0
          ? Math.min(360, Math.max(...validLons) - Math.min(...validLons))
          : 0
    }

    // Tessellate each axis from its span, addressing two distinct concerns:
    //  - Wide axis: ceil(span) gives ~1 vertex/degree so the mesh bends around
    //    the globe. Too-wide cells can't follow the curve and can exceed the
    //    mesh long-edge cull, dropping triangles entirely (issue #58).
    //  - Thin axis: the MIN_SUBDIVISIONS floor. A chunk that is a single data
    //    row spanning tens of degrees has a near-zero short span; without the
    //    floor, too few vertices there let hard data/nodata edges smear across
    //    globe-projected triangles.
    // Handles row and column strip chunks alike (whichever axis is thin).
    const subdivisionsForSpan = (span: number) =>
      Math.max(MIN_SUBDIVISIONS, Math.min(MAX_SUBDIVISIONS, Math.ceil(span)))
    const lonSubdivisions = subdivisionsForSpan(lonSpan)
    const latSubdivisions = subdivisionsForSpan(latSpan)

    const meshResult = createHybridMesh({
      geoBounds,
      width: region.width,
      height: region.height,
      lonSubdivisions,
      latSubdivisions,
      transformer: this.projection.to4326,
      latIsAscending: this.latIsAscending,
      allowUnwrappedLongitudes: this.projection.kind === 'epsg4326',
      // The mesh encodes vertices as deltas from a per-region mercator origin
      // (deriveLocalMercAnchor); renderRegion uploads a matching per-region
      // anchor_clip, keeping the eye origin near the on-screen region for
      // high-zoom precision (no pan/zoom jitter, no seams). See
      // VERTEX_TO_WGS84_TO_MERCATOR.
    })
    region.vertexArr = meshResult.positions
    region.pixCoordArr = meshResult.texCoords
    region.indexArr = meshResult.indices
    region.meshBounds = meshResult.meshBounds
    region.indexCount = region.indexArr.length
  }

  private async buildSliceArgsForSelector(
    selector: NormalizedSelector,
    options: {
      /** If true, set spatial dims to full slices; if false, set to 0 placeholder */
      includeSpatialSlices: boolean
      /** If true, track multi-value dimensions for channel packing */
      trackMultiValue: boolean
      /** Spatial bounds for fetch - bbox for region subset */
      spatialBounds?: {
        minX: number
        maxX: number
        minY: number
        maxY: number
      }
      /**
       * Array to derive shape from. Caller pins this so we don't read
       * `this.activeLevel?.zarrArray` mid-flight (which can swap during
       * `loadLevel` or zoom). Pass the new array during `loadLevel`, or
       * a snapshot of the active array for query/render paths.
       */
      array: zarr.Array<zarr.DataType>
    },
    explicitCoordLevelIndex?: number
  ): Promise<{
    sliceArgs: (number | zarr.Slice)[]
    multiValueDims: Array<{
      dimIndex: number
      dimName: string
      values: number[]
      labels: (number | string)[]
    }>
  }> {
    const coordLevelIndex =
      explicitCoordLevelIndex ??
      this.activeLevel?.index ??
      this.loadingLevelIndex ??
      this.desiredLevelIndex ??
      0
    return buildSliceArgsForSelector(
      {
        zarrStore: this.zarrStore,
        dimIndices: this.dimIndices,
        levels: this.levels,
        isMultiscale: this.isMultiscale,
        dimensionValues: this.dimensionValues,
        coordLevelIndex,
        warnedDimensions: this.warnedCoordDimensions,
      },
      selector,
      options
    )
  }

  /**
   * Reset visible region state after a level switch.
   * This clears stale coordinates from the previous level and forces
   * a fresh viewport calculation on the next update.
   * Note: We intentionally do NOT clear visibleRegionKeys here - old regions
   * need eviction protection until new level's regions are computed.
   */
  private resetVisibleRegions(): void {
    this.wantedRegions = new Map()
    this.wantedKeys = new Set()
    this.pendingLevels = new Set()
    this.pendingTiles = []
    this.lastViewportHash = ''
  }

  /**
   * Update the wanted regions from the map's tile cover (or its viewport)
   * and fetch, protect and release accordingly.
   */
  private updateVisibleRegions(map: MapLike, gl: WebGL2RenderingContext): void {
    const wanted = this.selectWantedRegions(map)
    const wantedKeys = new Set<string>()
    for (const [levelIndex, regions] of wanted) {
      for (const { regionX, regionY } of regions) {
        wantedKeys.add(this.makeRegionKey(levelIndex, regionX, regionY))
      }
    }
    this.wantedRegions = wanted
    this.wantedKeys = wantedKeys

    // Rebuild eviction protection from what is wanted, plus regions kept as
    // fallbacks: ones no longer wanted whose area is still waiting for its
    // replacement to become drawable. A region whose area is already
    // replaced, or wanted elsewhere, is released at once. An empty result
    // with nothing pending is a transient state (map bounds or transformer
    // unavailable), so keep the previous set rather than unprotect regions
    // that are still rendered.
    if (wantedKeys.size > 0 || this.pendingTiles.length > 0) {
      const pending = this.wantedCovered() ? [] : this.pendingAreas()
      this.regionCache.rebuildProtection(wantedKeys, {
        retain: (key) => {
          const bounds = this.regionCache.get(key)?.mercatorBounds
          return (
            !!bounds && pending.some((area) => boundsIntersect(area, bounds))
          )
        },
      })
    }

    // Abort in-flight fetches for regions no longer wanted.
    // Only signal abort — the fetch's own catch/finally handles state cleanup.
    for (const [key, region] of this.regionCache) {
      if (region.loading && region.requestId !== null && !wantedKeys.has(key)) {
        this.requestCanceller.controllers.get(region.requestId)?.abort()
      }
    }
    dropQueuedFetches(
      this.requestCanceller,
      (fetch) => !wantedKeys.has(fetch.key)
    )

    // Per level, separate regions into two categories:
    // 1. New regions (no data) - viewport change
    // 2. Stale regions (have data, wrong selector) - selector change
    const hashParts: string[] = []
    let dispatched = false
    for (const [levelIndex, regions] of wanted) {
      const newRegions: RegionCoordinate[] = []
      const staleRegions: RegionCoordinate[] = []
      for (const { regionX, regionY } of regions) {
        const key = this.makeRegionKey(levelIndex, regionX, regionY)
        const cached = this.regionCache.get(key)

        // Skip if already loading - when the load completes, invalidate()
        // triggers another updateVisibleRegions() check to see if refetch is needed
        if (cached?.loading) {
          continue
        }

        if (!cached || !hasRegionData(cached)) {
          if (cached && isRegionFailed(cached, this.selectorVersion)) continue
          // No data yet - this is a new region (viewport change)
          newRegions.push({ regionX, regionY })
        } else if (cached.selectorVersion !== this.selectorVersion) {
          // Has data but stale selector - this is a selector change
          staleRegions.push({ regionX, regionY })
        }
      }
      hashParts.push(
        `${levelIndex}:${regions
          .map((r) => `${r.regionX},${r.regionY}`)
          .join('|')}`
      )
      if (newRegions.length > 0) {
        this.fetchRegions(newRegions, levelIndex)
        dispatched = true
      }
      if (staleRegions.length > 0) {
        this.fetchRegions(staleRegions, levelIndex)
        dispatched = true
      }
    }

    // Check if the wanted set changed (include selectorVersion in hash)
    const viewportHash = `${this.selectorVersion}:${hashParts.join(';')}`
    const viewportChanged = viewportHash !== this.lastViewportHash
    this.lastViewportHash = viewportHash

    // Queued regions start nearest the viewport center first, judged on the
    // active level's regions.
    const center = this.wantedRegions.get(this.activeLevel?.index ?? -1)
    if ((dispatched || viewportChanged) && center && center.length > 1) {
      const cx = center.reduce((sum, r) => sum + r.regionX, 0) / center.length
      const cy = center.reduce((sum, r) => sum + r.regionY, 0) / center.length
      prioritizeQueuedFetches(this.requestCanceller, cx, cy)
    }
    // Always evict: fetches started on an earlier update have since landed,
    // and their bytes count towards the budget.
    this.evictOldRegions(gl)
  }

  private async fetchRegions(
    regions: Array<{ regionX: number; regionY: number }>,
    levelIndex: number
  ): Promise<void> {
    const fetcher = new RegionFetcher({
      zarrStore: this.zarrStore,
      dimIndices: this.dimIndices,
      levels: this.levels,
      projection: this.projection,
      xyLimits: this.xyLimits,
      latIsAscending: this.latIsAscending,
      fixedDataScale: this.fixedDataScale,
      regionCache: this.regionCache,
      requestCanceller: this.requestCanceller,
      loadingDebouncer: this.loadingDebouncer,
      getLevel: (index) => this.levelLoader.runtime(index),
      getSelectorVersion: () => this.selectorVersion,
      getBandNames: () => this.bandNames,
      usesBandTextures: () => this.rendersFromBandTextures,
      isRemoved: () => this.isRemoved,
      getRegionBounds: (regionX, regionY, levelMeta) =>
        this.getRegionBounds(regionX, regionY, levelMeta),
      computeRegionMercatorBounds: (bounds) =>
        this.computeRegionMercatorBounds(bounds),
      createRegionGeometry: (regionX, regionY, region) =>
        this.createRegionGeometry(regionX, regionY, region),
      // Textures exist only once a frame has run, and every frame caches gl.
      releaseRegionTextures: (region) => {
        if (this.cachedGl) releaseRegionTextures(this.cachedGl, region)
      },
      invalidate: this.invalidate,
    })
    await fetcher.fetchRegions(regions, levelIndex)
  }

  /**
   * Track the map without fetching: pick the level the zoom asks for, which
   * queries and `ready` load, and keep the gl context. `update` does this
   * too; a layer outside its zoom range calls only this.
   */
  followZoom(map: MapLike, gl: WebGL2RenderingContext): void {
    this.cachedGl = gl
    if (this.loadingManager.metadataLoading) return

    // Pick target: zoom-selected for multiscale, single level otherwise.
    if (this.isMultiscale && this.levels.length > 0) {
      const mapZoom = map.getZoom?.() ?? 0
      this.desiredLevelIndex = this.selectLevelForZoom(mapZoom)
    } else {
      this.desiredLevelIndex = 0
    }
  }

  update(map: MapLike, gl: WebGL2RenderingContext): void {
    this.followZoom(map, gl)

    // Don't proceed if metadata is still loading
    if (this.loadingManager.metadataLoading) {
      return
    }

    // Kick off a load only when the committed level doesn't match the
    // target AND we're not already loading that target. `prerender()`
    // calls `update()` every frame, so without the `loadingLevelIndex`
    // dedupe we'd bump `loadToken` on every frame and starve the load.
    if (this.activeLevel?.index !== this.desiredLevelIndex) {
      if (this.loadingLevelIndex !== this.desiredLevelIndex) {
        this.loadLevel(this.desiredLevelIndex)
      }
      return
    }

    // Committed level matches target — render from it. If a selector
    // rebuild is in flight *for this level* (`loadingLevelIndex ===
    // activeLevel.index`), skip the fetch loop: `updateVisibleRegions`
    // would dispatch fetches against the old `baseSliceArgs` and stamp
    // them with the pending selectorVersion, leaving stale data marked
    // fresh in the cache. A pending load for a *different* level (e.g.
    // user zoomed in then zoomed back out while the deeper-level load
    // was still outstanding) shouldn't block rendering the current one.
    if (this.loadingLevelIndex === this.activeLevel.index) return
    this.updateVisibleRegions(map, gl)
  }

  private loadLevel(
    levelIndex: number,
    options: { reuseArray?: boolean } = {}
  ): Promise<LevelLoadOutcome> {
    return this.levelLoader.loadLevel(levelIndex, options)
  }

  private selectLevelForZoom(mapZoom: number): number {
    return selectLevelForZoom({
      mapZoom,
      xyLimits: this.xyLimits,
      levels: this.levels,
      projection: this.projection,
      lonIndex: this.dimIndices.lon?.index,
    })
  }

  render(renderer: ZarrRenderer, context: RenderContext): void {
    context = {
      ...context,
      customShaderConfig: this.shaderConfigForLevel(context.customShaderConfig),
    }
    const useMapbox = !!context.mapbox
    // Use the source-projected mesh path when the CRS is resolved via proj4.
    const useWgs84 = !!this.projection.def && !!this.projection.to4326

    // MapLibre globe exposes a projectionTransition value in the shader prelude.
    // Keep ECEF active while that transition is nonzero.
    const hasMaplibreGlobeTransition =
      context.projectionData?.projectionTransition != null &&
      context.projectionData.projectionTransition > 0
    // Mapbox's globe→mercator zoom morph uses internal globe/mercator matrices
    // that the public custom-layer callback does not expose. Keep direct ECEF
    // only for the fully-globe endpoint; during the morph, fall back to the
    // regular direct Mapbox path so the zoom transition stays stable.
    const hasMapboxGlobe = useMapbox && this.isGlobeProjection
    // Match shader selection to the active render path. The layer-level
    // draped/direct switch is decided in ZarrLayer before this render call;
    // using that same flag here avoids one-frame depth mismatches near the
    // zoom-morph threshold where the public transition value and the layer path
    // can momentarily diverge.
    const hasMapboxDirectGlobePath =
      hasMapboxGlobe && context.mapbox?.directGlobePathActive === true
    const useDirectEcef = useMapbox
      ? hasMapboxDirectGlobePath && useWgs84
      : hasMaplibreGlobeTransition && useWgs84

    const shaderProgram = renderer.getProgram(
      context.shaderData,
      context.customShaderConfig,
      useMapbox,
      useWgs84 || useDirectEcef,
      useDirectEcef
    )

    renderer.gl.useProgram(shaderProgram.program)

    renderer.applyCommonUniforms(
      shaderProgram,
      context.colormapTexture,
      this.getUniformsForRender(context.uniforms),
      context.customShaderConfig,
      context.projectionData,
      context.mapbox,
      context.matrix,
      false
    )

    // When ECEF is active, force worldOffsets to [0]. During globe→flat transitions,
    // isGlobeProjection may flip to false before projectionTransition reaches 0,
    // causing computeWorldOffsets to return multiple offsets (e.g. [-1, 0, 1]).
    // Rendering at shifted world offsets on the globe produces duplicate renders.
    const worldOffsets = useDirectEcef ? [0] : context.worldOffsets

    // Resolve the flat projection matrix for the source-projected eye-coords
    // path. MapLibre flat uses mainMatrix; Mapbox flat passes its matrix
    // directly. ECEF/globe paths ignore these uniforms.
    // Pass it through UN-CAST (number[] | Float32Array | Float64Array) so
    // renderRegion computes anchor_clip from the highest-precision matrix
    // representation available. When the flat matrix is Float64 (as it is in
    // practice) that full-precision anchor is what keeps high-zoom pan/zoom
    // jitter-free; do NOT pre-cast to Float32 here (it re-quantizes the
    // translation and the jitter returns). renderRegion casts to Float32 only
    // for the GPU upload (which drives the small deltaClip).
    let eyeMatrix: number[] | Float32Array | Float64Array | null = null
    if (useWgs84 && !useDirectEcef) {
      const raw = useMapbox
        ? context.matrix
        : context.projectionData?.mainMatrix
      if (raw && raw.length >= 16) {
        eyeMatrix = raw
      }
    }

    this.renderRegions(
      renderer,
      shaderProgram,
      worldOffsets,
      context.customShaderConfig,
      eyeMatrix
    )
  }

  /**
   * Convert a RegionState to a RenderableRegion for unified rendering.
   * The shader program determines whether the mesh is projected flat or to
   * ECEF; both variants consume the same region-local Mercator encoding.
   */
  private regionToRenderable(region: RegionState): RenderableRegion {
    return {
      mercatorBounds: region.mercatorBounds!,
      vertexBuffer: region.vertexBuffer!,
      pixCoordBuffer: region.pixCoordBuffer!,
      indexCount: region.indexCount,
      indexBuffer: region.indexBuffer!,
      meshBounds: region.meshBounds!,
      latIsAscending: region.latIsAscending,
      texture: region.texture,
      bandTexture: region.bandTexture,
      bandTextureKey: region.bandTextureKey,
      bandTransform: region.bandTransform,
    }
  }

  /**
   * Render all loaded regions using the unified render path.
   * Note: Regions have geometry already positioned in mercator space,
   * so we disable the equirectangular shader correction to avoid double transformation.
   */
  private renderRegions(
    renderer: ZarrRenderer,
    shaderProgram: ShaderProgram,
    worldOffsets: number[],
    customShaderConfig?: CustomShaderConfig,
    eyeMatrix: number[] | Float32Array | Float64Array | null = null
  ): void {
    const gl = renderer.gl

    // Set up band texture uniforms once per frame
    setupBandTextureUniforms(gl, shaderProgram)

    // Render each loaded region using unified path
    const regions = this.getLoadedRegions(gl)
    const clips = this.clipBoxesFor(regions)
    for (const region of regions) {
      renderRegion(
        gl,
        shaderProgram,
        {
          ...this.regionToRenderable(region),
          clipBoxes: clips.get(region.key),
        },
        worldOffsets,
        customShaderConfig,
        eyeMatrix
      )
    }
  }

  renderToTile(
    renderer: ZarrRenderer,
    tileId: TileId,
    context: RenderContext,
    options: TileRenderOptions
  ): boolean {
    // Draped rendering only. The direct Mapbox ECEF path disables
    // renderToTile at the layer level.
    return renderTile({
      ...options,
      renderer,
      tileId,
      context: {
        ...context,
        uniforms: this.getUniformsForRender(context.uniforms),
        customShaderConfig: this.shaderConfigForLevel(
          context.customShaderConfig
        ),
      },
      regions: this.getRegionStates(renderer.gl),
    })
  }

  onProjectionChange(isGlobe: boolean): void {
    if (this.isGlobeProjection === isGlobe) return
    this.isGlobeProjection = isGlobe
  }

  /**
   * Get render states for all loaded regions (for multi-region rendering).
   * Includes previous level regions as fallback during level transitions.
   */
  private getRegionStates(gl: WebGL2RenderingContext): RegionRenderState[] {
    if (!(this.activeLevel?.regionSize ?? null)) {
      return []
    }

    const regions = this.getLoadedRegions(gl)
    const clips = this.clipBoxesFor(regions)
    return regions.map((region) => ({
      texture: region.texture,
      vertexBuffer: region.vertexBuffer!,
      pixCoordBuffer: region.pixCoordBuffer!,
      mercatorBounds: region.mercatorBounds!,
      width: region.width,
      height: region.height,
      bandTexture: region.bandTexture,
      bandTextureKey: region.bandTextureKey,
      bandTransform: region.bandTransform,
      indexBuffer: region.indexBuffer!,
      indexCount: region.indexCount,
      meshBounds: region.meshBounds!,
      latIsAscending: region.latIsAscending,
      clipBoxes: clips.get(region.key),
    }))
  }

  dispose(gl: WebGL2RenderingContext | WebGLRenderingContext): void {
    this.isRemoved = true
    this.levelLoader.dispose()
    cancelAllRequests(this.requestCanceller)
    // Clean up region caches
    this.clearRegionCache(gl)
    this.projection = createProjectionContext({
      crs: 'EPSG:4326',
      proj4def: null,
      xyLimits: null,
    })
    this.loadingDebouncer.hide()
  }

  setLoadingCallback(callback: LoadingStateCallback | undefined): void {
    setLoadingCallbackUtil(this.loadingManager, callback)
  }

  /**
   * Tell the renderer which texture the shader will sample. Band-sampling
   * shaders skip the interleaved channel copy that feeds the main texture.
   * Regions fetched under the previous setting keep their old layout until
   * they are refetched, so bump the selector version to refresh them.
   */
  setRendersFromBandTextures(active: boolean): void {
    if (this.rendersFromBandTextures === active) return
    this.rendersFromBandTextures = active
    this.selectorVersion++
    this.lastViewportHash = ''
    this.invalidate()
  }

  private computeMercatorBoundsFromProjection(): MercatorBounds {
    return computeMercatorBoundsFromProjection(this.xyLimits, this.projection)
  }

  private computeRegionMercatorBounds(bounds: {
    xMin: number
    xMax: number
    yMin: number
    yMax: number
  }): MercatorBounds {
    return computeRegionMercatorBounds(bounds, this.projection)
  }

  async setSelector(selector: NormalizedSelector): Promise<void> {
    this.selector = selector
    this.bandNames = getBands(this.variable, selector)
    // Retry levels latched by the previous selector.
    this.levelLoader.clearSelectorFailures()

    // Abort in-flight region fetches still running with the old selector.
    // Their catch/finally handles state cleanup and re-invalidation.
    for (const [, region] of this.regionCache) {
      if (region.loading && region.requestId !== null) {
        this.requestCanceller.controllers.get(region.requestId)?.abort()
      }
    }
    // Other levels were opened against the old selector; they reopen on
    // demand with the new one.
    this.levelLoader.clearSecondaryRuntimes()

    // Rebuild the runtime with the new selector and atomic-swap. Going
    // through `loadLevel` bumps `loadToken`, so any in-flight level load
    // (zoom-driven or initial) drops and restarts against the new
    // selector. `reuseArray` keeps the array/dims so no refetch is done
    // when the level itself isn't changing.
    if (this.activeLevel) {
      const outcome = await this.loadLevel(this.activeLevel.index, {
        reuseArray: true,
      })
      // Cached regions also belong to the previous selector. Without a gl
      // context no update has run yet, so there are none to clear.
      if (outcome === 'failed' && this.cachedGl) {
        this.clearRegionCache(this.cachedGl)
      }
    } else if (this.loadingLevelIndex !== null) {
      // A level load is already in flight; let it pick up the new
      // selector via its pre-commit `this.selector !== selectorSnapshot`
      // check, which drops it and our fresh call takes over.
      await this.loadLevel(this.loadingLevelIndex, { reuseArray: false })
    }

    // Bump only after the runtime commit — before this point, readers
    // that observe the new `selectorVersion` would still see the old
    // `baseSliceArgs` and mis-tag fetches.
    this.selectorVersion++
    this.lastViewportHash = ''
    this.invalidate()
  }

  /**
   * Adopt a new texture-normalization scale. Bumping selectorVersion marks
   * every cached region stale (updateVisibleRegions refetches regions whose
   * selectorVersion doesn't match), so texture data is re-normalized.
   */
  setDataScale(scale: number): void {
    if (scale === this.fixedDataScale) return
    this.fixedDataScale = scale
    this.selectorVersion++
    this.lastViewportHash = ''
    this.invalidate()
  }

  private emitLoadingState(): void {
    emitLoadingStateUtil(this.loadingManager)
  }

  /**
   * Await a committed level. Multiscale init deliberately leaves the level
   * to `update()`, which only runs from the render path, so a query issued
   * against a map that hasn't painted has to commit one itself.
   */
  ensureQueryableLevel(): Promise<LevelRuntime | null> {
    return this.levelLoader.ensureActive()
  }

  get levelLoadError(): Error | null {
    return this.levelLoader.lastError
  }

  /**
   * The highest-resolution level, by pixel width rather than by position:
   * level order follows whatever the store declared, which is conventionally
   * coarsest-first but is not guaranteed to be.
   */
  private finestLevelIndex(): number {
    if (!this.isMultiscale || this.levels.length === 0) return 0
    let finest = 0
    let widest = -1
    for (let i = 0; i < this.levels.length; i++) {
      const shape = this.levels[i].shape
      if (!shape) continue
      const width = shape[this.dimIndices.lon?.index ?? shape.length - 1]
      if (width > widest) {
        widest = width
        finest = i
      }
    }
    return finest
  }

  /**
   * Resolve the level a query reads from.
   *
   * `'finest'` deliberately bypasses `LevelLoader` when the renderer is on a
   * different level: committing the finest level would drag the render loop
   * onto it and fight the zoom-driven target over `loadToken`. A query-local
   * snapshot leaves render state untouched.
   */
  private async resolveQueryLevel(
    requested: QueryLevel = 'current'
  ): Promise<QueryLevelSnapshot | null> {
    const snapshot = (level: LevelRuntime): QueryLevelSnapshot => ({
      index: level.index,
      zarrArray: level.zarrArray,
      width: level.width,
      height: level.height,
      xyLimits: level.xyLimits,
    })

    if (requested !== 'finest') {
      const active = await this.ensureQueryableLevel()
      return active && snapshot(active)
    }

    const finest = this.finestLevelIndex()
    if (this.activeLevel?.index === finest) return snapshot(this.activeLevel)
    try {
      const opened = await this.openLevel(finest)
      return {
        index: finest,
        zarrArray: opened.zarrArray,
        width: opened.width,
        height: opened.height,
        xyLimits: opened.xyLimits,
      }
    } catch (err) {
      // Propagated, not swallowed: an unreachable or malformed finest asset
      // must not come back looking like a geometry that found no data.
      throw wrapError(
        `[ZarrLayer] failed to open level ${
          this.levels[finest]?.asset ?? finest
        } for query`,
        err
      )
    }
  }

  /** Query data for point or region geometries. */
  async queryData(
    geometry: QueryGeometry,
    selector?: Selector,
    options?: QueryOptions
  ): Promise<QueryResult> {
    const level = await this.resolveQueryLevel(options?.level)
    return queryDataWithContext(
      {
        zarrStore: this.zarrStore,
        variable: this.variable,
        selector: this.selector,
        xyLimits: this.xyLimits,
        mercatorBounds: this.mercatorBounds,
        latIsAscending: this.latIsAscending,
        levels: this.levels,
        level,
        projection: this.projection,
        antimeridianWarnings: this._antimeridianWarnings,
        dimensionValues: this.dimensionValues,
        isMultiscale: this.isMultiscale,
        coordLevelIndex: level?.index ?? 0,
        warnedDimensions: this.warnedCoordDimensions,
      },
      geometry,
      selector,
      options
    )
  }
}
