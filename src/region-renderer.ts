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
import {
  latToMercatorNorm,
  type MercatorBounds,
  type XYLimits,
} from './map-utils'
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
  selectLevelForZoom,
  type LonLatBounds,
  type RegionCoordinate,
} from './region-math'
import { boundsIntersect, getTileLonLatBounds } from './tile-utils'
import { tileClipBounds, tileClipPasses, type ClipTile } from './clip-boxes'
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
  /**
   * The world copy the tile is drawn in. The map may cover copies at
   * different zooms, so ownership is per copy; undefined means every copy.
   */
  wrap?: number
  /**
   * The parts of the tile the map shows, for a tile synthesised from the
   * viewport: regions elsewhere in the tile are not its concern. Longitudes
   * follow the viewport's, which may run past ±180; a viewport nearly a
   * turn wide can meet one tile at both ends, giving two parts.
   */
  extents?: LonLatBounds[]
}

/**
 * Regions one update may want in total: wanting more than the cache can hold
 * would only make it thrash. A cover needing more reads coarser levels until
 * it fits; with no coarser level left, selection stops there, nearest tiles
 * first.
 */
const MAX_WANTED_REGIONS = MAX_CACHED_REGIONS

/**
 * Fewest tiles a cover synthesised for the viewport splits it into; the
 * coarsest zoom reaching that gives at most four times as many.
 */
const MIN_VIEWPORT_TILES = 16
const MAX_VIEWPORT_TILE_ZOOM = 24

const wrapLon = (lon: number) => ((((lon + 180) % 360) + 360) % 360) - 180

/**
 * The parts of a tile's longitude range inside a viewport's, which may run
 * past ±180, west to east: one per world copy of the tile the viewport meets.
 */
function intersectLon(
  tileWest: number,
  tileEast: number,
  west: number,
  east: number
): Array<[number, number]> {
  const parts: Array<[number, number]> = []
  for (const shift of [-360, 0, 360]) {
    const a = Math.max(tileWest + shift, west)
    const b = Math.min(tileEast + shift, east)
    if (a < b) parts.push([a, b])
  }
  return parts
}

const tileKey = (tile: TileId) => `${tile.z}/${tile.x}/${tile.y}`
const requestKey = (tile: TileRequest) =>
  `${tileKey(tile.tileId)}@${tile.wrap ?? 'all'}`
/** Copies of one tile show the same thing, so they share a plan and level. */
const planKey = (tile: TileRequest) => `${tileKey(tile.tileId)}:${tile.zoom}`

/** What one map tile shows. */
interface TileDraw {
  tile: TileRequest
  /** The regions, all of one level and all drawable. */
  regions: RegionState[]
  /** Whether they are the tile's own level, complete. */
  ready: boolean
}

/** The level a tile targets and its regions there, none until it opens. */
interface TileTarget {
  levelIndex: number
  runtime: LevelRuntime | null
  regions: RegionCoordinate[]
}

/** A region to draw on the direct path, clipped to the tiles that show it. */
interface RegionDraw {
  region: RegionState
  /** One draw per pass; null draws the region whole at every offset. */
  clipPasses: Array<{ worldOffset: number; boxes: Float32Array }> | null
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
  // The map's own tile cover for the frame, when the map exposes it.
  private coveringTiles: TileRequest[] | null = null
  // A cover made from the viewport when the map gives none and no draped
  // hook is drawing tiles, so that path plans per tile as well.
  private syntheticCover: TileRequest[] | null = null
  // Tiles the draped render hooks were asked to draw, which on terrain reach
  // past the sea-level cover. Kept in two generations keyed by the layer's
  // terrain revision: a revision change redraws every visible tile, so the
  // current generation refills with the whole visible set while the previous
  // one still covers tiles drawn since. Older tiles drop out.
  private drawnTiles: Map<string, TileRequest> = new Map()
  private previousDrawnTiles: Map<string, TileRequest> = new Map()
  private drawnTilesRevision: number | null = null
  private tileRegionsMemo: Map<string, RegionCoordinate[]> = new Map()
  // The level each tile of the current cover targets, by plan key, fitted
  // by the last selection so the whole cover fits the region budget;
  // planning reads it too.
  private tileLevels: Map<string, number> = new Map()
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

    const levelIndex = this.tileLevel(tile)
    if (levelIndex === null) return false
    const runtime = this.levelLoader.runtime(levelIndex)
    if (!runtime) return true
    return this.tileRegions(levelIndex, runtime, tile).some((region) => {
      const cached = this.regionCache.get(
        this.makeRegionKey(levelIndex, region.regionX, region.regionY)
      )
      return !cached || (!hasRegionData(cached) && !cached.loading)
    })
  }

  /**
   * Forget the tiles the draped hooks drew, once the map no longer drapes
   * the layer. Returns whether there were any; the levels fitted to them
   * go too.
   */
  clearDrawnTiles(): boolean {
    if (this.drawnTiles.size === 0 && this.previousDrawnTiles.size === 0) {
      this.drawnTilesRevision = null
      return false
    }
    this.drawnTiles = new Map()
    this.previousDrawnTiles = new Map()
    this.drawnTilesRevision = null
    this.tileLevels = new Map()
    return true
  }

  /**
   * Every tile the map is drawing: the map's cover, then tiles the draped
   * hooks drew beyond it, each most detailed (nearest) first. Hook tiles
   * outlive the view that drew them until the terrain is redrawn, so they
   * come after the cover and only get what budget it leaves.
   */
  private requestedTiles(): TileRequest[] {
    const byZoom = (a: TileRequest, b: TileRequest) => b.zoom - a.zoom
    const keys = new Set<string>()
    const cover: TileRequest[] = []
    const hooked: TileRequest[] = []
    const add = (tile: TileRequest, list: TileRequest[]) => {
      const key = requestKey(tile)
      if (keys.has(key)) return
      keys.add(key)
      list.push(tile)
    }
    for (const tile of this.coveringTiles ?? []) add(tile, cover)
    for (const tile of this.drawnTiles.values()) add(tile, hooked)
    for (const tile of this.previousDrawnTiles.values()) add(tile, hooked)
    return [...cover.sort(byZoom), ...hooked.sort(byZoom)]
  }

  /** Regions of a level under a tile; the tile grid and levels never move. */
  private tileRegions(
    levelIndex: number,
    levelMeta: LevelMeta,
    { tileId, poles, extents }: TileRequest
  ): RegionCoordinate[] {
    const extentKey = (extents ?? [])
      .map((e) => `:${e.west},${e.south},${e.east},${e.north}`)
      .join('')
    const key = `${levelIndex}:${poles ? 'p' : 'm'}:${tileKey(
      tileId
    )}${extentKey}`
    const memo = this.tileRegionsMemo.get(key)
    if (memo) return memo
    if (this.tileRegionsMemo.size > 4096) this.tileRegionsMemo.clear()
    const regions: RegionCoordinate[] = []
    const seen = new Set<string>()
    for (const bounds of extents ?? [
      getTileLonLatBounds(tileId, { toPoles: poles }),
    ]) {
      // Strictly: a region whose edge lies on the tile's edge shows nothing
      // in it, and owning it would make the tile fetch it and wait on it.
      for (const region of getRegionsInBounds({
        bounds,
        xyLimits: this.xyLimits,
        levelMeta,
        projection: this.projection,
        latIsAscending: this.latIsAscending,
        strict: true,
      })) {
        const regionKey = `${region.regionX},${region.regionY}`
        if (seen.has(regionKey)) continue
        seen.add(regionKey)
        regions.push(region)
      }
    }
    this.tileRegionsMemo.set(key, regions)
    return regions
  }

  /**
   * A tile cover for the viewport when the map gives none (Mapbox without
   * terrain): the coarsest tiles that split the viewport into at least
   * MIN_VIEWPORT_TILES, each cut down to the parts of it the viewport shows
   * so selection stays exactly the viewport's regions, each read at the map
   * zoom, nearest the centre first. Empty without map bounds.
   */
  private viewportTiles(map: MapLike): TileRequest[] {
    const bounds = map.getBounds?.()?.toArray?.()
    if (!bounds) return []
    const [[west, south], [rawEast, north]] = bounds
    const east = rawEast < west ? rawEast + 360 : rawEast
    const span = east - west
    const zoom = map.getZoom?.() ?? 0
    const center = map.getCenter?.() ?? {
      lng: (west + east) / 2,
      lat: (south + north) / 2,
    }
    const poles = this.isGlobeProjection
    for (let z = 0; z <= MAX_VIEWPORT_TILE_ZOOM; z++) {
      const n = 2 ** z
      const clamp = (v: number) => Math.min(n - 1, Math.max(0, Math.floor(v)))
      const y0 = clamp(latToMercatorNorm(north) * n)
      const y1 = clamp(latToMercatorNorm(south) * n)
      // Columns follow the unwrapped span: a nearly global viewport whose
      // ends wrap into one column still covers every column between.
      const xw = ((wrapLon(west) + 180) / 360) * n
      const columnCount =
        span >= 360
          ? n
          : Math.min(
              n,
              Math.floor(xw + (span / 360) * n - 1e-9) - Math.floor(xw) + 1
            )
      if (
        columnCount * (y1 - y0 + 1) < MIN_VIEWPORT_TILES &&
        z < MAX_VIEWPORT_TILE_ZOOM
      ) {
        continue
      }
      const tiles: Array<{ tile: TileRequest; d: number }> = []
      for (let i = 0; i < columnCount; i++) {
        const x = (Math.floor(xw) + i) % n
        for (let y = y0; y <= y1; y++) {
          const tileId = { z, x, y }
          const b = getTileLonLatBounds(tileId, { toPoles: poles })
          const lons: Array<[number, number]> =
            span >= 360
              ? [[b.west, b.east]]
              : intersectLon(b.west, b.east, west, east)
          const tileSouth = Math.max(b.south, south)
          const tileNorth = Math.min(b.north, north)
          if (lons.length === 0 || tileSouth >= tileNorth) continue
          const extents = lons.map(([lonWest, lonEast]) => ({
            west: lonWest,
            east: lonEast,
            south: tileSouth,
            north: tileNorth,
          }))
          const dLon = wrapLon((b.west + b.east) / 2 - center.lng)
          const dLat = (b.south + b.north) / 2 - center.lat
          tiles.push({
            tile: { tileId, zoom, poles, extents },
            d: dLon * dLon + dLat * dLat,
          })
        }
      }
      return tiles.sort((a, b) => a.d - b.d).map(({ tile }) => tile)
    }
    return []
  }

  /**
   * The regions to want this update, per level. Each tile of the cover
   * reads the level matching its zoom; a level not yet open is opened for
   * the next update. Without a cover from the map or a draped hook (Mapbox
   * without terrain), the viewport is tiled here instead.
   */
  private selectWantedRegions(map: MapLike): Map<number, RegionCoordinate[]> {
    const wanted = new Map<number, RegionCoordinate[]>()
    this.pendingLevels = new Set()
    this.syntheticCover = null
    let tiles = this.requestedTiles()
    if (tiles.length === 0) {
      tiles = this.viewportTiles(map)
      if (tiles.length === 0) {
        this.tileLevels = new Map()
        return wanted
      }
      this.syntheticCover = tiles
    }
    // The fit follows the current view: hook tiles from an earlier one must
    // not hold it coarse. Hook tiles alone are fitted only without a cover.
    this.tileLevels = this.fitTileLevels(
      this.syntheticCover ?? this.coveringTiles ?? tiles,
      map
    )
    const perTile = this.tileTargets(tiles)
    let total = 0
    for (const target of perTile) {
      if (target.runtime) {
        total += target.regions.length
      } else {
        void this.levelLoader.ensureRuntime(target.levelIndex)
        this.pendingLevels.add(target.levelIndex)
      }
    }
    // Tiles come in order already: the cover's, then the hooks', nearest
    // first. When the budget bites, order each tile's regions by distance
    // from the map center too, so a single tile spanning many regions cannot
    // spend the budget on its far corner.
    const truncating = total > MAX_WANTED_REGIONS
    const seen = new Set<string>()
    for (const { levelIndex, runtime, regions } of perTile) {
      if (!runtime) continue
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

  /**
   * Each tile's target level, with its regions there; a level not yet open
   * has none to give. Tiles no level can serve are left out.
   */
  private tileTargets(tiles: TileRequest[]): TileTarget[] {
    const targets: TileTarget[] = []
    for (const tile of tiles) {
      const levelIndex = this.tileLevel(tile)
      if (levelIndex === null) continue
      const runtime = this.levelLoader.runtime(levelIndex)
      const regions = runtime ? this.tileRegions(levelIndex, runtime, tile) : []
      targets.push({ levelIndex, runtime, regions })
    }
    return targets
  }

  /**
   * The level each tile targets so that the whole cover can be wanted in
   * full. Each starts at its own; while their regions together exceed the
   * budget, tiles drop one level at a time, farthest first (coarser zooms,
   * then farther from the map center), working inward and round again.
   * Tiles at the coarsest usable level are passed over; when none can drop,
   * selection keeps the nearest regions. Levels not yet open count no
   * regions and are opened, so the fit depends only on the tiles and the
   * levels open, which only ever grow, and it settles.
   */
  private fitTileLevels(
    tiles: TileRequest[],
    map: MapLike
  ): Map<string, number> {
    const levels = new Map<string, number>()
    const counts = new Map<string, number>()
    let unique = 0
    const regionKeys = (tile: TileRequest, levelIndex: number) => {
      const runtime = this.levelLoader.runtime(levelIndex)
      if (!runtime) {
        void this.levelLoader.ensureRuntime(levelIndex)
        return []
      }
      return this.tileRegions(levelIndex, runtime, tile).map((region) =>
        this.makeRegionKey(levelIndex, region.regionX, region.regionY)
      )
    }
    const add = (keys: string[]) => {
      for (const key of keys) {
        const count = counts.get(key) ?? 0
        if (count === 0) unique++
        counts.set(key, count + 1)
      }
    }
    const remove = (keys: string[]) => {
      for (const key of keys) {
        const count = counts.get(key)!
        if (count === 1) {
          counts.delete(key)
          unique--
        } else {
          counts.set(key, count - 1)
        }
      }
    }

    const { lng, lat } = this.mapCenter(map)
    const cx = (lng + 180) / 360
    const cy = latToMercatorNorm(lat)
    const entries: Array<{
      tile: TileRequest
      key: string
      levelIndex: number
      keys: string[]
      distance: number
    }> = []
    for (const tile of tiles) {
      const key = planKey(tile)
      if (levels.has(key)) continue
      const levelIndex = this.ownTileLevel(tile)
      if (levelIndex === null) continue
      const keys = regionKeys(tile, levelIndex)
      add(keys)
      levels.set(key, levelIndex)
      const { z, x, y } = tile.tileId
      let dx = (x + 0.5) / 2 ** z - cx
      dx -= Math.round(dx)
      const dy = (y + 0.5) / 2 ** z - cy
      entries.push({ tile, key, levelIndex, keys, distance: dx * dx + dy * dy })
    }
    if (unique <= MAX_WANTED_REGIONS) return levels

    entries.sort((a, b) => a.tile.zoom - b.tile.zoom || b.distance - a.distance)
    let dropped = true
    while (unique > MAX_WANTED_REGIONS && dropped) {
      dropped = false
      for (const entry of entries) {
        if (unique <= MAX_WANTED_REGIONS) break
        const coarser = this.usableLevel(this.coarserLevel(entry.levelIndex))
        if (coarser === null) continue
        remove(entry.keys)
        entry.levelIndex = coarser
        entry.keys = regionKeys(entry.tile, coarser)
        add(entry.keys)
        levels.set(entry.key, coarser)
        dropped = true
      }
    }
    return levels
  }

  /** The map center, or the middle of its bounds without one. */
  private mapCenter(map: MapLike): { lng: number; lat: number } {
    const center = map.getCenter?.()
    if (center) return center
    const bounds = map.getBounds?.()?.toArray?.()
    return {
      lng: bounds ? (bounds[0][0] + bounds[1][0]) / 2 : 0,
      lat: bounds ? (bounds[0][1] + bounds[1][1]) / 2 : 0,
    }
  }

  /** Regions ordered by their distance from the map center, in degrees. */
  private nearestCenterFirst(
    regions: RegionCoordinate[],
    runtime: LevelRuntime,
    map: MapLike
  ): RegionCoordinate[] {
    const { lng, lat } = this.mapCenter(map)
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
   * Areas (normalized Mercator) of wanted regions not yet drawable. Without
   * a tile cover, fallbacks overlapping them are kept until the replacement
   * can be drawn.
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
    return areas
  }

  /** The next coarser level, or null from the coarsest. */
  private coarserLevel(levelIndex: number): number | null {
    const width = this.levelWidth(levelIndex)
    let coarser: number | null = null
    for (let i = 0; i < this.levels.length; i++) {
      const w = this.levelWidth(i)
      if (w < width && (coarser === null || w > this.levelWidth(coarser))) {
        coarser = i
      }
    }
    return coarser
  }

  /**
   * The level itself or, when its selector failed to resolve, the nearest
   * coarser level that did not. Null when none did.
   */
  private usableLevel(levelIndex: number | null): number | null {
    while (levelIndex !== null && this.levelLoader.levelFailed(levelIndex)) {
      levelIndex = this.coarserLevel(levelIndex)
    }
    return levelIndex
  }

  /**
   * The level a zoom reads: the one it selects or, when that level's
   * selector failed to resolve, the nearest coarser level that did not.
   * Null when no level can serve it.
   */
  private resolveLevel(zoom: number): number | null {
    return this.usableLevel(this.selectLevelForZoom(zoom))
  }

  /**
   * The level a tile targets: the one the last selection fitted it to, or
   * its own for a tile outside the fitted cover. Null when no level can
   * serve the tile, which then draws nothing.
   */
  private tileLevel(tile: TileRequest): number | null {
    const fitted = this.tileLevels.get(planKey(tile))
    return fitted === undefined
      ? this.ownTileLevel(tile)
      : this.usableLevel(fitted)
  }

  /**
   * A tile's own level: the one its zoom resolves to or, when that has more
   * regions under the tile than one update may want, the finest coarser
   * level that fits, so the tile can be wanted in full and shown from its
   * target. A level not yet open is trusted until it opens. With no coarser
   * level the tile keeps its own and shows what it has of it.
   */
  private ownTileLevel(tile: TileRequest): number | null {
    let levelIndex = this.resolveLevel(tile.zoom)
    while (levelIndex !== null) {
      const runtime = this.levelLoader.runtime(levelIndex)
      if (!runtime) return levelIndex
      const count = this.tileRegions(levelIndex, runtime, tile).length
      if (count <= MAX_WANTED_REGIONS) return levelIndex
      const coarser = this.usableLevel(this.coarserLevel(levelIndex))
      if (coarser === null) return levelIndex
      levelIndex = coarser
    }
    return null
  }

  /**
   * What one tile shows: its own level once every region under it is
   * drawable, else the one cached level that serves it best meanwhile.
   */
  private planTile(
    tile: TileRequest,
    drawable: (region: RegionState) => boolean
  ): TileDraw {
    const levelIndex = this.tileLevel(tile)
    if (levelIndex === null) return { tile, regions: [], ready: true }
    const runtime = this.levelLoader.runtime(levelIndex)
    if (runtime) {
      const regions: RegionState[] = []
      let ready = true
      const own = this.tileRegions(levelIndex, runtime, tile)
      for (const { regionX, regionY } of own) {
        const cached = this.regionCache.get(
          this.makeRegionKey(levelIndex, regionX, regionY)
        )
        if (cached && drawable(cached)) regions.push(cached)
        else ready = false
      }
      if (ready) return { tile, regions, ready }
    }
    return {
      tile,
      regions: this.fallbackRegions(tile, levelIndex, drawable),
      ready: false,
    }
  }

  /**
   * Regions of the one cached level to show in a tile whose own level is
   * not ready: the finest level with drawable data under the whole tile,
   * or failing that the level covering most of it. Its own level counts
   * too, through the regions being promoted (wanted) rather than whatever
   * of it the cache still holds. Polar tiles on the globe reach the pole,
   * where region bounds clamp to the Mercator limit.
   */
  private fallbackRegions(
    tile: TileRequest,
    ownLevel: number,
    drawable: (region: RegionState) => boolean
  ): RegionState[] {
    const tileBounds = tileClipBounds(tile)
    const byLevel = new Map<number, RegionState[]>()
    for (const region of this.regionCache.values()) {
      const bounds = region.mercatorBounds
      if (!bounds) continue
      if (region.levelIndex === ownLevel && !this.wantedKeys.has(region.key)) {
        continue
      }
      if (!boundsIntersect(bounds, tileBounds) || !drawable(region)) continue
      const list = byLevel.get(region.levelIndex)
      if (list) list.push(region)
      else byLevel.set(region.levelIndex, [region])
    }
    let best: RegionState[] = []
    let bestComplete = false
    let bestCovered = -1
    let bestWidth = -1
    for (const [levelIndex, regions] of byLevel) {
      const keys = new Set(regions.map((region) => region.key))
      const meta = regions[0].levelMeta
      const complete =
        !!meta &&
        this.tileRegions(levelIndex, meta, tile).every(({ regionX, regionY }) =>
          keys.has(this.makeRegionKey(levelIndex, regionX, regionY))
        )
      let covered = 0
      for (const { mercatorBounds: b } of regions) {
        covered +=
          Math.max(
            0,
            Math.min(b!.x1, tileBounds.x1) - Math.max(b!.x0, tileBounds.x0)
          ) *
          Math.max(
            0,
            Math.min(b!.y1, tileBounds.y1) - Math.max(b!.y0, tileBounds.y0)
          )
      }
      const width = this.levelWidth(levelIndex)
      const better = complete
        ? !bestComplete || width > bestWidth
        : !bestComplete &&
          (covered > bestCovered ||
            (covered === bestCovered && width > bestWidth))
      if (better) {
        best = regions
        bestComplete = complete
        bestCovered = covered
        bestWidth = width
      }
    }
    return best
  }

  /** Regions drawable now, uploading what is CPU-ready on demand. */
  private drawableWith(
    gl: WebGL2RenderingContext
  ): (region: RegionState) => boolean {
    const requiredBands = this.requiredBands()
    const bandFormat = this.bandFormat()
    return (region) =>
      isRegionCpuReady(region) &&
      ensureRegionGpuResources(gl, region, requiredBands, bandFormat)
  }

  /**
   * The regions to draw on the direct path, for the world offsets drawn.
   * With a tile cover, each tile shows one level and each region is
   * clipped to the tiles that show it, per world copy, so every pixel shows
   * exactly one level whatever the layer opacity. Without one, the wanted
   * regions are drawn whole over any fallbacks.
   */
  private plannedDraws(
    gl: WebGL2RenderingContext,
    worldOffsets: number[] = [0]
  ): RegionDraw[] {
    const tiles = this.coveringTiles?.length
      ? this.coveringTiles
      : this.syntheticCover
    if (!tiles || tiles.length === 0) {
      return this.getLoadedRegions(gl).map((region) => ({
        region,
        clipPasses: null,
      }))
    }
    const drawable = this.drawableWith(gl)
    const plans = new Map<string, TileDraw>()
    const owners = new Map<string, { region: RegionState; tiles: ClipTile[] }>()
    for (const tile of tiles) {
      const key = planKey(tile)
      let plan = plans.get(key)
      if (!plan) {
        plan = this.planTile(tile, drawable)
        plans.set(key, plan)
      }
      for (const region of plan.regions) {
        let entry = owners.get(region.key)
        if (!entry) {
          entry = { region, tiles: [] }
          owners.set(region.key, entry)
        }
        entry.tiles.push(tile)
      }
    }
    const draws: RegionDraw[] = []
    for (const { region, tiles } of owners.values()) {
      // A tile that only touches the region's edge owns it without showing
      // any of it; a region no tile shows in a copy is not drawn there.
      const clipPasses: Array<{ worldOffset: number; boxes: Float32Array }> = []
      for (const worldOffset of worldOffsets) {
        for (const boxes of tileClipPasses(
          region.meshBounds!,
          tiles,
          worldOffset
        )) {
          clipPasses.push({ worldOffset, boxes })
        }
      }
      if (clipPasses.length > 0) draws.push({ region, clipPasses })
    }
    return draws.sort(
      (a, b) =>
        this.levelWidth(a.region.levelIndex) -
        this.levelWidth(b.region.levelIndex)
    )
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
   * Drawable regions without a tile cover: the wanted regions plus fallbacks
   * if needed, which are regions no longer wanted but still protected, kept
   * from before a level change until what is wanted can be drawn in full.
   * Uploads the wanted regions first so coverage is judged on what can
   * actually be drawn — regions that displace their fallbacks and then fail
   * to upload would leave the viewport blank. Every returned region is
   * GPU-ready, and coarser levels come first so finer data draws over them
   * where both exist.
   */
  private getLoadedRegions(gl: WebGL2RenderingContext): RegionState[] {
    const drawable = this.drawableWith(gl)

    const wanted: RegionState[] = []
    for (const key of this.wantedKeys) {
      const region = this.regionCache.get(key)
      if (region && drawable(region)) wanted.push(region)
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
    this.syntheticCover = null
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
    // fallbacks: with a tile cover, the regions each tile shows while its
    // own level is not ready, judged exactly as the render plan judges
    // them, so what is drawn is what is kept — fine data that has landed
    // but cannot upload must not displace the coarse data on screen;
    // without one, regions no longer wanted whose area is still waiting for
    // its replacement to become drawable. A fallback no tile shows any more
    // is released at once. An empty result without a cover is a transient
    // state (map bounds or transformer unavailable), so keep the previous
    // set rather than unprotect regions that are still rendered.
    const tiles = this.syntheticCover ?? this.requestedTiles()
    if (tiles.length > 0) {
      const drawable = this.drawableWith(gl)
      const retained = new Set<string>()
      for (const tile of tiles) {
        const draw = this.planTile(tile, drawable)
        if (draw.ready) continue
        for (const region of draw.regions) retained.add(region.key)
      }
      this.regionCache.rebuildProtection(wantedKeys, {
        retain: (key) => retained.has(key),
      })
    } else if (wantedKeys.size > 0) {
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
      // A level whose selector failed cannot open, and `update` fetches
      // nothing until the active level matches the desired one, so the
      // centre reads the same coarser level its tiles fall back to.
      this.desiredLevelIndex =
        this.resolveLevel(mapZoom) ?? this.selectLevelForZoom(mapZoom)
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

    for (const { region, clipPasses } of this.plannedDraws(gl, worldOffsets)) {
      const renderable = this.regionToRenderable(region)
      if (!clipPasses) {
        renderRegion(
          gl,
          shaderProgram,
          renderable,
          worldOffsets,
          customShaderConfig,
          eyeMatrix
        )
        continue
      }
      for (const { worldOffset, boxes } of clipPasses) {
        renderRegion(
          gl,
          shaderProgram,
          { ...renderable, clipBoxes: boxes },
          [worldOffset],
          customShaderConfig,
          eyeMatrix
        )
      }
    }
  }

  /**
   * Draw one draped tile: the regions of its own level once complete, a
   * fallback level until then. Returns true when the tile should be drawn
   * again, because its level is not ready or a region failed to draw.
   */
  renderToTile(
    renderer: ZarrRenderer,
    tile: TileRequest,
    context: RenderContext,
    options: TileRenderOptions
  ): boolean {
    if (!(this.activeLevel?.regionSize ?? null)) return true
    const draw = this.planTile(tile, this.drawableWith(renderer.gl))
    const needsRedraw = renderTile({
      ...options,
      renderer,
      tileId: tile.tileId,
      context: {
        ...context,
        uniforms: this.getUniformsForRender(context.uniforms),
        customShaderConfig: this.shaderConfigForLevel(
          context.customShaderConfig
        ),
      },
      regions: draw.regions.map((region) => this.toRenderState(region)),
    })
    return needsRedraw || !draw.ready
  }

  onProjectionChange(isGlobe: boolean): void {
    if (this.isGlobeProjection === isGlobe) return
    this.isGlobeProjection = isGlobe
  }

  /** A drawable region's render state for the tile render path. */
  private toRenderState(region: RegionState): RegionRenderState {
    return {
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
    }
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
