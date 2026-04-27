/**
 * @module untiled-mode
 *
 * Unified mode for non-tiled Zarr datasets.
 * Handles both single-level datasets and multi-level datasets following
 * the zarr-conventions/multiscales standard. Loads full images at each
 * resolution level (not slippy map tiles) with automatic level selection
 * based on map zoom.
 */

import * as zarr from 'zarrita'
import {
  WEB_MERCATOR_EXTENT,
  MIN_SUBDIVISIONS,
  MAX_SUBDIVISIONS,
  MERCATOR_SUBDIVISIONS,
} from './constants'
import type {
  ZarrMode,
  RenderContext,
  TileId,
  RegionRenderState,
} from './zarr-mode'
import type {
  QueryGeometry,
  QueryOptions,
  QueryResult,
  QueryDataValues,
  NestedValues,
} from './query/types'
import type {
  LoadingStateCallback,
  MapLike,
  NormalizedSelector,
  Selector,
  CRS,
  DimIndicesProps,
  UntiledLevel,
} from './types'
import { ZarrStore } from './zarr-store'
import {
  boundsToMercatorNorm,
  flipTexCoordV,
  latToMercatorNorm,
  latToWgs84Norm,
  lonToMercatorNorm,
  mercatorNormToLat,
  type MercatorBounds,
  type XYLimits,
  type Wgs84Bounds,
} from './map-utils'
import { loadDimensionValues, normalizeSelector, getBands } from './zarr-utils'
import {
  createSubdividedQuad,
  interleaveBands,
  normalizeDataForTexture,
} from './webgl-utils'
import type { ZarrRenderer, ShaderProgram } from './zarr-renderer'
import type { CustomShaderConfig } from './renderer-types'
import { renderMapboxTile } from './mapbox-tile-renderer'
import { queryRegionUntiled, findSpatialDimNames } from './query/region-query'
import {
  computePixelBoundsFromGeometry,
  preprocessQueryGeometry,
  wrappedBboxToPixelSpans,
  rasterExtentCrossesAntimeridian,
  type PixelRect,
} from './query/query-utils'
import {
  createTransformer,
  createTransformerTo4326,
  createWGS84ToSourceTransformer,
  pixelToSourceCRS,
  sampleEdgesToMercatorBounds,
} from './projection-utils'
import { createHybridMesh } from './mesh-reprojector'
import { geoToArrayIndex } from './map-utils'
import {
  type RequestCanceller,
  type LoadingManager,
  type ChunkLoadingDebouncer,
  createRequestCanceller,
  createLoadingManager,
  createChunkLoadingDebouncer,
  cancelAllRequests,
  hasActiveRequests,
  setLoadingCallback as setLoadingCallbackUtil,
  emitLoadingState as emitLoadingStateUtil,
} from './mode-utils'
import { setupBandTextureUniforms, uploadDataTexture } from './render-helpers'
import { renderRegion, type RenderableRegion } from './renderable-region'

/** State for a single region (chunk/shard) in region-based loading */
interface RegionState {
  key: string
  levelIndex: number
  regionX: number
  regionY: number
  // Data
  data: Float32Array | null
  width: number
  height: number
  loading: boolean
  requestId: number | null
  channels: number
  // WebGL resources
  texture: WebGLTexture | null
  textureUploaded: boolean
  vertexBuffer: WebGLBuffer | null
  pixCoordBuffer: WebGLBuffer | null
  indexBuffer: WebGLBuffer | null // For adaptive mesh indexed triangles
  // Geometry arrays for this region's quad
  vertexArr: Float32Array | null
  pixCoordArr: Float32Array | null // Texture coordinates for sampling resampled data
  indexArr: Uint32Array | null // Triangle indices for adaptive mesh
  vertexCount: number // Number of vertices (for triangle strip) or indices (for indexed)
  useIndexedMesh: boolean // Whether to use indexed triangles (adaptive mesh)
  // Mercator bounds for this region (for shader uniforms)
  mercatorBounds: MercatorBounds | null
  // WGS84 bounds for vertex shader positioning (proj4 datasets, ECEF globe)
  wgs84Bounds: Wgs84Bounds | null
  // Data orientation: true = row 0 is south
  latIsAscending: boolean
  // Version tracking for selector changes
  selectorVersion: number
  // Multi-band support
  bandData: Map<string, Float32Array>
  bandTextures: Map<string, WebGLTexture>
  bandTexturesUploaded: Set<string>
  bandTexturesConfigured: Set<string>
  // Level-specific dimensions for correct geometry rebuild across projection changes.
  // Set from LevelSnapshot during fetch to avoid races with level switching.
  levelMeta: LevelMeta | null
}

/** Level-specific dimensions for geometry bounds calculation */
type LevelMeta = {
  width: number
  height: number
  regionSize: [number, number]
  // xyLimits omitted - assumed constant across levels for now.
  // TODO: If heterogeneous pyramids with per-level bounds are needed,
  // add xyLimits here and store per-region.
}

/** Maximum number of regions to keep in cache (LRU eviction) */
const MAX_CACHED_REGIONS = 128
/** Snapshot of level state captured at fetch start to prevent race conditions */
interface LevelSnapshot {
  index: number
  zarrArray: zarr.Array<zarr.DataType>
  baseSliceArgs: (number | zarr.Slice)[]
  baseMultiValueDims: Array<{
    dimIndex: number
    dimName: string
    values: number[]
    labels: (number | string)[]
  }>
  width: number
  height: number
  regionSize: [number, number]
  selectorVersion: number
  bandNames: string[]
}

/**
 * Fully-committed per-level state. Replaces six top-level fields that were
 * previously mutated independently across `initializeLevel`, `switchToLevel`,
 * `_initialize`, and `setSelector` — any of which was one `await` away from
 * a half-commit race.
 *
 * `UntiledMode.activeLevel` is either `null` (nothing loaded) or a fully-
 * formed runtime; readers never see a partial level.
 */
interface LevelRuntime {
  index: number
  zarrArray: zarr.Array<zarr.DataType>
  width: number
  height: number
  regionSize: [number, number]
  baseSliceArgs: (number | zarr.Slice)[]
  baseMultiValueDims: Array<{
    dimIndex: number
    dimName: string
    values: number[]
    labels: (number | string)[]
  }>
}

type QueryLevelSnapshot = Pick<
  LevelRuntime,
  'index' | 'zarrArray' | 'width' | 'height'
>

export class UntiledMode implements ZarrMode {
  isMultiscale: boolean = false

  private channels: number = 1

  // The single committed snapshot. All per-level state (array, dims, slice
  // args) is swapped atomically through `loadLevel()`; nothing else mutates
  // these fields.
  private activeLevel: LevelRuntime | null = null
  // Monotonic id stamped by each `loadLevel` call. Async loads check this
  // before committing — a bump invalidates older pending work.
  private loadToken: number = 0
  // Target level requested by zoom/init. `update()` writes this; `loadLevel`
  // reads it to re-target if a zoom change happened mid-load.
  private desiredLevelIndex: number = 0
  // Target of the currently-running `loadLevel`, or null when idle. Used
  // by `update()` to dedupe: if we're already loading the target level,
  // don't restart the fetch every frame (ZarrLayer.prerender calls
  // update() once per frame, so without this we'd never commit).
  private loadingLevelIndex: number | null = null

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
  private crs: CRS = 'EPSG:4326'
  private latIsAscending: boolean = true

  // Multi-level support
  private levels: UntiledLevel[] = []
  private levelMetadataFetched: Set<number> = new Set() // Tracks which levels have had metadata fetched
  private proj4def: string | null = null

  // Cached transformers for proj4 reprojection (created once, reused everywhere)
  private cachedMercatorTransformer: ReturnType<
    typeof createTransformer
  > | null = null
  private cachedWGS84Transformer: ReturnType<
    typeof createWGS84ToSourceTransformer
  > | null = null
  // Transformer: source CRS → EPSG:4326 (for WGS84 vertex positions and ECEF projection)
  private cached4326Transformer: ReturnType<
    typeof createTransformerTo4326
  > | null = null

  // Loading state
  private isRemoved: boolean = false
  private _antimeridianWarnings: Set<string> = new Set()

  // Shared state managers
  private requestCanceller: RequestCanceller = createRequestCanceller()
  private loadingManager: LoadingManager = createLoadingManager()
  private loadingDebouncer: ChunkLoadingDebouncer = createChunkLoadingDebouncer(
    this.loadingManager
  )

  // Dimension values cache (supports numeric and string coordinate arrays)
  private dimensionValues: {
    [key: string]: Float64Array | number[] | string[]
  } = {}

  // Region-based loading (for multi-level datasets with chunking/sharding)
  // Single unified cache with LRU eviction - keys include level index (e.g., "2:0,0")
  private regionCache: Map<string, RegionState> = new Map()
  // Keys of regions protected from eviction. Lifecycle:
  // - Added: in updateVisibleRegions() for current level's visible regions
  // - Retained: across level switches to protect fallback regions during transitions
  // - Cleared: in updateVisibleRegions() when currentLevelCoversViewport() returns true,
  //   at which point non-current-level keys are removed (fallbacks no longer needed)
  private visibleRegionKeys: Set<string> = new Set()
  private lastVisibleRegions: Array<{ regionX: number; regionY: number }> = [] // Last computed visible regions
  private lastVisibleRegionsLevel: number = -1 // Level index that lastVisibleRegions corresponds to
  private lastViewportHash: string = ''
  private selectorVersion: number = 0 // Incremented on selector change to track stale regions

  // Cached WebGL context for use in setSelector
  private cachedGl: WebGL2RenderingContext | null = null
  // Track current projection for subdivision optimization
  private isGlobeProjection: boolean = false
  // Deferred geometry rebuild: when globe→flat transition starts, onProjectionChange(false)
  // fires before projectionTransition reaches 0. Rebuilding geometry immediately would drop
  // subdivisions to 1 while the ECEF shader is still rendering on the globe.
  // This flag defers the rebuild until projectionTransition reaches 0.
  private pendingGeometryRebuild: boolean = false
  // Fixed data scale for normalization (set at initialization, passed from ZarrLayer)
  private fixedDataScale: number = 1

  constructor(
    store: ZarrStore,
    variable: string,
    selector: NormalizedSelector,
    invalidate: () => void,
    fixedDataScale: number = 1
  ) {
    this.zarrStore = store
    this.variable = variable
    this.selector = selector
    this.bandNames = getBands(variable, selector)
    this.invalidate = invalidate
    this.fixedDataScale = fixedDataScale
  }

  async initialize(): Promise<void> {
    this.loadingManager.metadataLoading = true
    this.emitLoadingState()

    try {
      const desc = this.zarrStore.describe()
      this.dimIndices = desc.dimIndices
      this.crs = desc.crs
      this.xyLimits = desc.xyLimits
      this.latIsAscending = desc.latIsAscending
      this.proj4def = desc.proj4 ?? null

      // Cache transformers once for reuse (major performance optimization)
      if (this.proj4def && this.xyLimits) {
        const bounds: [number, number, number, number] = [
          this.xyLimits.xMin,
          this.xyLimits.yMin,
          this.xyLimits.xMax,
          this.xyLimits.yMax,
        ]
        this.cachedMercatorTransformer = createTransformer(
          this.proj4def,
          bounds
        )
        this.cachedWGS84Transformer = createWGS84ToSourceTransformer(
          this.proj4def
        )
        // Source CRS → EPSG:4326 (for WGS84 mesh vertices and ECEF projection)
        this.cached4326Transformer = createTransformerTo4326(
          this.proj4def,
          bounds
        )
      }

      if (this.crs !== 'EPSG:4326' && this.crs !== 'EPSG:3857') {
        console.warn(
          `Unsupported CRS "${this.crs}" - rendering may be incorrect. Supported: EPSG:4326, EPSG:3857`
        )
      }

      // Check if this is a multi-level dataset
      if (desc.untiledLevels && desc.untiledLevels.length > 0) {
        this.levels = desc.untiledLevels
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
        // For proj4, compute mercator bounds by transforming corners
        if (this.proj4def) {
          this.mercatorBounds = this.computeMercatorBoundsFromProjection()
        } else {
          this.mercatorBounds = boundsToMercatorNorm(this.xyLimits, this.crs)
        }
      } else {
        console.warn('UntiledMode: No XY limits found')
      }
    } finally {
      this.loadingManager.metadataLoading = false
      this.emitLoadingState()
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
      const meta = await this.zarrStore.getUntiledLevelMetadata(level.asset)
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
          const meta = await this.zarrStore.getUntiledLevelMetadata(level.asset)
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

  /**
   * Detect optimal region size from array metadata.
   * For sharded arrays: use shard chunk_shape
   * For standard chunked arrays: use array chunks
   */
  private getRegionSize(
    array: zarr.Array<zarr.DataType>
  ): [number, number] | null {
    const latIdx = this.dimIndices.lat?.index
    const lonIdx = this.dimIndices.lon?.index
    if (latIdx === undefined || lonIdx === undefined) return null

    // Check for sharding codec
    const codecs = (array as any).codecs || []
    for (const codec of codecs) {
      if (
        codec.name === 'sharding_indexed' &&
        codec.configuration?.chunk_shape
      ) {
        const shardShape = codec.configuration.chunk_shape as number[]
        return [shardShape[latIdx], shardShape[lonIdx]]
      }
    }

    // Fall back to standard chunks
    const chunks = array.chunks as number[] | undefined
    if (chunks && chunks.length > Math.max(latIdx, lonIdx)) {
      const chunkH = chunks[latIdx]
      const chunkW = chunks[lonIdx]
      // Only use region-based loading if chunks are smaller than the array
      const shape = array.shape as number[]
      if (chunkH < shape[latIdx] || chunkW < shape[lonIdx]) {
        return [chunkH, chunkW]
      }
    }

    return null // No chunking or single chunk
  }

  /**
   * Clear region cache and dispose WebGL resources.
   */
  private clearRegionCache(
    gl: WebGL2RenderingContext | WebGLRenderingContext
  ): void {
    for (const region of this.regionCache.values()) {
      this.disposeRegion(region, gl)
    }
    this.regionCache.clear()
    this.lastViewportHash = ''
  }

  /**
   * Dispose WebGL resources for a single region.
   */
  private disposeRegion(
    region: RegionState,
    gl: WebGL2RenderingContext | WebGLRenderingContext
  ): void {
    if (region.texture) gl.deleteTexture(region.texture)
    if (region.vertexBuffer) gl.deleteBuffer(region.vertexBuffer)
    if (region.pixCoordBuffer) gl.deleteBuffer(region.pixCoordBuffer)
    if (region.indexBuffer) gl.deleteBuffer(region.indexBuffer)
    for (const tex of region.bandTextures.values()) {
      gl.deleteTexture(tex)
    }
  }

  /**
   * Evict oldest regions when cache exceeds limit (LRU eviction).
   * Uses Map iteration order (oldest first).
   * Never evicts currently visible regions.
   */
  private evictOldRegions(gl: WebGL2RenderingContext): void {
    while (this.regionCache.size > MAX_CACHED_REGIONS) {
      let evictedKey: string | null = null
      for (const key of this.regionCache.keys()) {
        if (!this.visibleRegionKeys.has(key)) {
          evictedKey = key
          break
        }
      }
      if (!evictedKey) break // All regions are visible, stop
      const region = this.regionCache.get(evictedKey)
      if (region) this.disposeRegion(region, gl)
      this.regionCache.delete(evictedKey)
    }
  }

  /**
   * Calculate which regions are visible in the current viewport.
   */
  private getVisibleRegions(
    map: MapLike
  ): Array<{ regionX: number; regionY: number }> {
    const bounds = map.getBounds?.()?.toArray?.()
    if (!bounds || !this.xyLimits || !this.activeLevel) return []

    const { width, height, regionSize } = this.activeLevel
    const [[west, south], [east, north]] = bounds
    const { xMin, xMax, yMin, yMax } = this.xyLimits
    const [regionH, regionW] = regionSize

    if (this.proj4def && this.cachedWGS84Transformer) {
      // For projected data, use a two-pass approach:
      // 1. Forward-transform viewport edges to source CRS to find candidate regions
      //    via index math (O(1) proj4 cost, may include false positives for non-
      //    bijective projections like UTM outside their zone)
      // 2. Inverse-transform candidate region bounds to WGS84 for precise overlap
      const transformer = this.cachedWGS84Transformer
      const numRegionsX = Math.ceil(width / regionW)
      const numRegionsY = Math.ceil(height / regionH)

      const candidates = this.getCandidateRegions(
        west,
        south,
        east,
        north,
        transformer,
        numRegionsX,
        numRegionsY,
        regionW,
        regionH,
        width,
        height
      )

      // Verify candidates via inverse transform to WGS84 for precise overlap.
      // This handles non-bijective projections where forward transforms can
      // produce false positives.
      const regions: Array<{ regionX: number; regionY: number }> = []
      for (const { regionX, regionY } of candidates) {
        const regBounds = this.getRegionBounds(regionX, regionY, {
          width,
          height,
          regionSize,
        })
        const xMid = (regBounds.xMin + regBounds.xMax) / 2
        const yMid = (regBounds.yMin + regBounds.yMax) / 2

        const samplePoints = [
          transformer.inverse(regBounds.xMin, regBounds.yMin),
          transformer.inverse(regBounds.xMax, regBounds.yMin),
          transformer.inverse(regBounds.xMax, regBounds.yMax),
          transformer.inverse(regBounds.xMin, regBounds.yMax),
          transformer.inverse(xMid, regBounds.yMin),
          transformer.inverse(xMid, regBounds.yMax),
          transformer.inverse(regBounds.xMin, yMid),
          transformer.inverse(regBounds.xMax, yMid),
        ]

        let regWest = Infinity
        let regEast = -Infinity
        let regSouth = Infinity
        let regNorth = -Infinity
        let hasValid = false
        for (const [lon, lat] of samplePoints) {
          if (!isFinite(lon) || !isFinite(lat)) continue
          hasValid = true
          if (lon < regWest) regWest = lon
          if (lon > regEast) regEast = lon
          if (lat < regSouth) regSouth = lat
          if (lat > regNorth) regNorth = lat
        }
        if (!hasValid) continue

        if (
          regEast >= west &&
          regWest <= east &&
          regNorth >= south &&
          regSouth <= north
        ) {
          regions.push({ regionX, regionY })
        }
      }

      return regions
    }

    // Standard case: viewport bounds are in same CRS as xyLimits
    const xMinIdx = geoToArrayIndex(west, xMin, xMax, width)
    const xMaxIdx = geoToArrayIndex(east, xMin, xMax, width)

    // For Y axis, geoToArrayIndex assumes yMin maps to row 0.
    // But if latIsAscending=false (row 0 = north = yMax), we need to invert.
    let ySouthIdx = geoToArrayIndex(south, yMin, yMax, height)
    let yNorthIdx = geoToArrayIndex(north, yMin, yMax, height)

    // Only invert if we explicitly know latIsAscending is false
    // If null/undefined, assume ascending (yMin at row 0) as default
    if (this.latIsAscending === false) {
      // Invert Y indices: row 0 = north (yMax), row height-1 = south (yMin)
      ySouthIdx = height - 1 - ySouthIdx
      yNorthIdx = height - 1 - yNorthIdx
    }

    // Convert pixel indices to region indices
    const regionXMin = Math.floor(Math.min(xMinIdx, xMaxIdx) / regionW)
    const regionXMax = Math.floor(Math.max(xMinIdx, xMaxIdx) / regionW)
    const regionYMin = Math.floor(Math.min(ySouthIdx, yNorthIdx) / regionH)
    const regionYMax = Math.floor(Math.max(ySouthIdx, yNorthIdx) / regionH)

    // Clamp to valid range
    const numRegionsX = Math.ceil(width / regionW)
    const numRegionsY = Math.ceil(height / regionH)
    const clampedXMin = Math.max(0, regionXMin)
    const clampedXMax = Math.min(numRegionsX - 1, regionXMax)
    const clampedYMin = Math.max(0, regionYMin)
    const clampedYMax = Math.min(numRegionsY - 1, regionYMax)

    // Build list of visible region coordinates
    const regions: Array<{ regionX: number; regionY: number }> = []
    for (let ry = clampedYMin; ry <= clampedYMax; ry++) {
      for (let rx = clampedXMin; rx <= clampedXMax; rx++) {
        regions.push({ regionX: rx, regionY: ry })
      }
    }

    return regions
  }

  /**
   * Create a region key that includes level index for unified caching.
   */
  private makeRegionKey(
    levelIndex: number,
    regionX: number,
    regionY: number
  ): string {
    return `${levelIndex}:${regionX},${regionY}`
  }

  /**
   * Create a new region state entry.
   */
  private createRegionState(
    levelIndex: number,
    regionX: number,
    regionY: number
  ): RegionState {
    return {
      key: this.makeRegionKey(levelIndex, regionX, regionY),
      levelIndex,
      regionX,
      regionY,
      data: null,
      width: 0,
      height: 0,
      loading: false,
      requestId: null,
      channels: 1,
      texture: null,
      textureUploaded: false,
      vertexBuffer: null,
      pixCoordBuffer: null,
      indexBuffer: null,
      vertexArr: null,
      pixCoordArr: null,
      indexArr: null,
      vertexCount: 0,
      useIndexedMesh: false,
      mercatorBounds: null,
      wgs84Bounds: null,
      latIsAscending: this.latIsAscending,
      selectorVersion: this.selectorVersion,
      bandData: new Map(),
      bandTextures: new Map(),
      bandTexturesUploaded: new Set(),
      bandTexturesConfigured: new Set(),
      levelMeta: null, // Set from snapshot in fetchRegion
    }
  }

  /**
   * Check if a region has all required data for rendering.
   */
  private isRegionValid(region: RegionState): boolean {
    return !!(
      region.data &&
      region.textureUploaded &&
      region.texture &&
      region.vertexBuffer &&
      region.pixCoordBuffer &&
      region.vertexArr &&
      region.mercatorBounds &&
      region.levelMeta
    )
  }

  /**
   * Clear loading flags for queued-but-not-started regions in a batch.
   * Only touches regions where requestId is null (pre-marked as loading
   * but no fetch was started). In-flight regions (requestId set) are
   * cleaned up by their own finally block.
   */
  private clearBatchLoadingFlags(
    regions: Array<{ regionX: number; regionY: number }>,
    levelIndex: number
  ): void {
    for (const { regionX, regionY } of regions) {
      const key = this.makeRegionKey(levelIndex, regionX, regionY)
      const region = this.regionCache.get(key)
      if (region && region.requestId === null) {
        region.loading = false
      }
    }
  }

  /**
   * Get uniforms for rendering with scale/offset disabled.
   * Untiled mode applies per-level scale/offset in JS (in fetchRegion),
   * so we tell the shader to skip its scale/offset application.
   */
  private getUniformsForRender(contextUniforms: RenderContext['uniforms']) {
    return {
      ...contextUniforms,
      scaleFactor: 1.0,
      offset: 0.0,
    }
  }

  /**
   * Check if current level fully covers the visible viewport.
   * Returns true if all visible regions have valid loaded data.
   */
  private currentLevelCoversViewport(): boolean {
    // If visible regions are stale (from different level), we can't know coverage
    if (this.lastVisibleRegionsLevel !== (this.activeLevel?.index ?? -1)) {
      return false
    }
    const levelIndex = this.activeLevel?.index ?? -1
    for (const { regionX, regionY } of this.lastVisibleRegions) {
      const key = this.makeRegionKey(levelIndex, regionX, regionY)
      const region = this.regionCache.get(key)
      if (!region || !this.isRegionValid(region)) {
        return false
      }
    }
    return this.lastVisibleRegions.length > 0
  }

  /**
   * Get fallback regions from other levels that are protected from eviction.
   * These were visible before or during level transitions and provide
   * coverage while the current level loads.
   */
  private getProtectedFallbackRegions(): RegionState[] {
    const fallbacks: RegionState[] = []
    for (const region of this.regionCache.values()) {
      if (region.levelIndex === (this.activeLevel?.index ?? -1)) continue
      if (!this.isRegionValid(region)) continue
      // Only include regions that are protected (were visible)
      if (!this.visibleRegionKeys.has(region.key)) continue
      fallbacks.push(region)
    }
    return fallbacks
  }

  /**
   * Get regions to render: current level regions plus fallbacks if needed.
   * When current level fully covers viewport, returns only current level.
   * Otherwise, includes protected fallback regions from other levels.
   */
  private getLoadedRegions(): RegionState[] {
    const currentLevel = this.activeLevel?.index ?? -1
    const currentLevelRegions: RegionState[] = []

    // Collect all valid regions at current level
    for (const region of this.regionCache.values()) {
      if (!this.isRegionValid(region)) continue
      if (region.levelIndex === currentLevel) {
        currentLevelRegions.push(region)
      }
    }

    // If current level fully covers viewport, no fallback needed
    if (this.currentLevelCoversViewport()) {
      return currentLevelRegions
    }

    // Include protected fallback regions from other levels
    const fallbackRegions = this.getProtectedFallbackRegions()

    // Render order: fallbacks first (beneath), current level on top
    return [...fallbackRegions, ...currentLevelRegions]
  }

  /**
   * Build all index combinations from multi-value dimensions.
   * Returns cartesian product of all dimension value arrays.
   */
  private buildChannelCombinations(
    multiValueDims: Array<{ values: number[]; labels: (number | string)[] }>
  ): { combinations: number[][]; labelCombinations: (number | string)[][] } {
    let combinations: number[][] = [[]]
    let labelCombinations: (number | string)[][] = [[]]

    for (const { values, labels } of multiValueDims) {
      const nextCombos: number[][] = []
      const nextLabels: (number | string)[][] = []
      for (let idx = 0; idx < values.length; idx++) {
        for (let c = 0; c < combinations.length; c++) {
          nextCombos.push([...combinations[c], values[idx]])
          nextLabels.push([...labelCombinations[c], labels[idx]])
        }
      }
      combinations = nextCombos
      labelCombinations = nextLabels
    }

    return { combinations, labelCombinations }
  }

  /**
   * Find candidate regions by forward-transforming viewport edges to source CRS
   * and using grid index math to find overlapping region indices.
   *
   * This is a fast prefilter that may include false positives (e.g., for non-
   * bijective projections like UTM outside their zone). Callers must verify
   * candidates with inverse-transform overlap checks.
   *
   * On partial transform failures (projection boundary), uses valid points
   * with a wider margin. Falls back to all regions only if no points are valid.
   */
  private getCandidateRegions(
    west: number,
    south: number,
    east: number,
    north: number,
    transformer: { forward: (lon: number, lat: number) => [number, number] },
    numRegionsX: number,
    numRegionsY: number,
    regionW: number,
    regionH: number,
    width: number,
    height: number
  ): Array<{ regionX: number; regionY: number }> {
    if (!this.xyLimits) return []
    const { xMin, xMax, yMin, yMax } = this.xyLimits

    // Densely sample viewport edges and interior to capture projection curvature
    // and extrema that may fall inside the viewport (e.g., pole in polar stereo).
    const edgeSamples = 16
    let srcXMin = Infinity
    let srcXMax = -Infinity
    let srcYMin = Infinity
    let srcYMax = -Infinity
    let validCount = 0
    let totalCount = 0
    for (let i = 0; i <= edgeSamples; i++) {
      const t = i / edgeSamples
      const lon = west + t * (east - west)
      const lat = south + t * (north - south)
      const points = [
        transformer.forward(lon, south),
        transformer.forward(lon, north),
        transformer.forward(west, lat),
        transformer.forward(east, lat),
      ]
      for (const [x, y] of points) {
        totalCount++
        if (!isFinite(x) || !isFinite(y)) continue
        validCount++
        if (x < srcXMin) srcXMin = x
        if (x > srcXMax) srcXMax = x
        if (y < srcYMin) srcYMin = y
        if (y > srcYMax) srcYMax = y
      }
    }

    // Sample interior grid to catch extrema inside viewport (e.g., pole in polar stereo)
    const interiorSamples = 4
    for (let iy = 1; iy <= interiorSamples; iy++) {
      for (let ix = 1; ix <= interiorSamples; ix++) {
        const lon = west + (ix / (interiorSamples + 1)) * (east - west)
        const lat = south + (iy / (interiorSamples + 1)) * (north - south)
        const [x, y] = transformer.forward(lon, lat)
        totalCount++
        if (!isFinite(x) || !isFinite(y)) continue
        validCount++
        if (x < srcXMin) srcXMin = x
        if (x > srcXMax) srcXMax = x
        if (y < srcYMin) srcYMin = y
        if (y > srcYMax) srcYMax = y
      }
    }

    // No valid points — fall back to all regions
    if (validCount === 0) {
      const all: Array<{ regionX: number; regionY: number }> = []
      for (let ry = 0; ry < numRegionsY; ry++) {
        for (let rx = 0; rx < numRegionsX; rx++) {
          all.push({ regionX: rx, regionY: ry })
        }
      }
      return all
    }

    // Widen margin when some samples failed (projection boundary)
    const margin = validCount < totalCount ? 8 : 2

    const pxXMin = ((srcXMin - xMin) / (xMax - xMin)) * width
    const pxXMax = ((srcXMax - xMin) / (xMax - xMin)) * width
    const pxYMin = ((srcYMin - yMin) / (yMax - yMin)) * height
    const pxYMax = ((srcYMax - yMin) / (yMax - yMin)) * height
    let rXMin: number, rXMax: number, rYMin: number, rYMax: number
    if (this.latIsAscending === false) {
      const invYMin = height - pxYMax
      const invYMax = height - pxYMin
      rYMin = Math.floor(invYMin / regionH) - margin
      rYMax = Math.floor(invYMax / regionH) + margin
    } else {
      rYMin = Math.floor(pxYMin / regionH) - margin
      rYMax = Math.floor(pxYMax / regionH) + margin
    }
    rXMin = Math.floor(pxXMin / regionW) - margin
    rXMax = Math.floor(pxXMax / regionW) + margin

    // Clamp to valid range
    rXMin = Math.max(0, rXMin)
    rXMax = Math.min(numRegionsX - 1, rXMax)
    rYMin = Math.max(0, rYMin)
    rYMax = Math.min(numRegionsY - 1, rYMax)

    const candidates: Array<{ regionX: number; regionY: number }> = []
    for (let ry = rYMin; ry <= rYMax; ry++) {
      for (let rx = rXMin; rx <= rXMax; rx++) {
        candidates.push({ regionX: rx, regionY: ry })
      }
    }
    return candidates
  }

  /**
   * Get geographic bounds for a region.
   * Accounts for data orientation (latIsAscending).
   * Requires level-specific dimensions so async work never reaches back into
   * `activeLevel`, which may have changed since the caller captured a region.
   */
  private getRegionBounds(
    regionX: number,
    regionY: number,
    levelMeta: LevelMeta
  ): { xMin: number; xMax: number; yMin: number; yMax: number } {
    const { width, height, regionSize } = levelMeta

    // xyLimits is assumed constant across all multiscale levels (same geographic extent).
    // If per-level bounds are ever needed, add xyLimits to LevelMeta type.
    if (!this.xyLimits) {
      return { xMin: 0, xMax: 1, yMin: 0, yMax: 1 }
    }

    const [regionH, regionW] = regionSize
    const { xMin, xMax, yMin, yMax } = this.xyLimits

    // Calculate pixel bounds for this region
    const pxXStart = regionX * regionW
    const pxXEnd = Math.min(pxXStart + regionW, width)
    const pxYStart = regionY * regionH
    const pxYEnd = Math.min(pxYStart + regionH, height)

    // Convert pixel bounds to geographic bounds using pixel edges.
    const geoXMin = xMin + (pxXStart / width) * (xMax - xMin)
    const geoXMax = xMin + (pxXEnd / width) * (xMax - xMin)

    // Y mapping depends on data orientation
    // Default (null/undefined) assumes ascending (row 0 = south = yMin)
    let geoYMin: number
    let geoYMax: number
    if (this.latIsAscending === false) {
      // Data has lat decreasing with array index: pixel 0 = north (yMax)
      geoYMax = yMax - (pxYStart / height) * (yMax - yMin)
      geoYMin = yMax - (pxYEnd / height) * (yMax - yMin)
    } else {
      // Data has lat increasing with array index: pixel 0 = south (yMin)
      geoYMin = yMin + (pxYStart / height) * (yMax - yMin)
      geoYMax = yMin + (pxYEnd / height) * (yMax - yMin)
    }

    return { xMin: geoXMin, xMax: geoXMax, yMin: geoYMin, yMax: geoYMax }
  }

  /**
   * Create geometry (vertex positions and tex coords) for a region.
   * Uses subdivided geometry for smooth globe rendering.
   *
   * Three geometry paths based on CRS (GPU projection chosen at render time):
   * - Proj4 datasets: Adaptive mesh with WGS84 vertices (GPU: → Mercator or → ECEF)
   * - EPSG:4326: Subdivided quad in Mercator space (GPU: fragment reprojection or ECEF direct)
   * - EPSG:3857: Subdivided quad with linear texture coords (data already in Mercator)
   */
  private createRegionGeometry(
    regionX: number,
    regionY: number,
    gl: WebGL2RenderingContext,
    region: RegionState
  ): void {
    // Guard: can't create geometry without dimension info
    if (!region.levelMeta) return

    // Defensive reset: wgs84Bounds is only set by the proj4 branch below.
    // Ensures it's not stale after projection toggle or geometry rebuild.
    region.wgs84Bounds = null
    region.indexArr = null
    region.useIndexedMesh = false

    const geoBounds = this.getRegionBounds(regionX, regionY, region.levelMeta)

    // Use cached mercatorBounds if set (from fetchRegion's resampling path),
    // otherwise compute from geoBounds (for non-resampling cases like EPSG:3857)
    const mercBounds =
      region.mercatorBounds ?? boundsToMercatorNorm(geoBounds, this.crs)
    region.mercatorBounds = mercBounds

    if (this.proj4def && this.cached4326Transformer) {
      // Proj4 datasets: compute WGS84 vertex positions via proj4.
      // CPU: transform vertices from source CRS to WGS84.
      // GPU: transform WGS84 → Mercator (flat) or WGS84 → ECEF (globe).

      // Calculate subdivisions based on latSpan.
      // For polar projections, the pole is at the CENTER, not corners, so sample
      // the center point to get the true latitude span.
      const centerX = (geoBounds.xMin + geoBounds.xMax) / 2
      const centerY = (geoBounds.yMin + geoBounds.yMax) / 2
      const samplePoints = [
        this.cached4326Transformer.forward(geoBounds.xMin, geoBounds.yMin),
        this.cached4326Transformer.forward(geoBounds.xMax, geoBounds.yMin),
        this.cached4326Transformer.forward(geoBounds.xMin, geoBounds.yMax),
        this.cached4326Transformer.forward(geoBounds.xMax, geoBounds.yMax),
        this.cached4326Transformer.forward(centerX, centerY), // Center point (pole for polar projections)
      ]
      const validLats = samplePoints
        .map((p) => p[1])
        .filter((lat) => isFinite(lat))
      const latSpan =
        validLats.length > 0
          ? Math.max(...validLats) - Math.min(...validLats)
          : 0
      const meshSubdivisions = Math.max(
        MIN_SUBDIVISIONS,
        Math.min(MAX_SUBDIVISIONS, Math.ceil(latSpan))
      )

      // Always use hybrid mesh (adaptive + uniform grid) for proj4 data.
      const meshResult = createHybridMesh({
        geoBounds,
        width: region.width,
        height: region.height,
        subdivisions: meshSubdivisions,
        transformer: this.cached4326Transformer!,
        latIsAscending: this.latIsAscending,
      })
      region.vertexArr = meshResult.positions
      region.pixCoordArr = meshResult.texCoords
      region.indexArr = meshResult.indices
      region.wgs84Bounds = meshResult.wgs84Bounds
      region.useIndexedMesh = true
      region.vertexCount = region.indexArr!.length
    } else {
      // Non-proj4 paths: EPSG:4326 and EPSG:3857
      // Compute subdivisions once based on CRS

      // Subdivisions: high for globe (smooth curvature), minimal for mercator (flat)
      // Use latitude span in degrees - lat is the primary driver of globe curvature
      let latSpanDegrees: number
      if (this.crs === 'EPSG:3857') {
        // 3857 bounds are in meters - convert to degrees
        const yMinNorm = 0.5 - geoBounds.yMin / (2 * WEB_MERCATOR_EXTENT)
        const yMaxNorm = 0.5 - geoBounds.yMax / (2 * WEB_MERCATOR_EXTENT)
        latSpanDegrees = Math.abs(
          mercatorNormToLat(yMaxNorm) - mercatorNormToLat(yMinNorm)
        )
      } else {
        // 4326 bounds are already in degrees
        latSpanDegrees = Math.abs(geoBounds.yMax - geoBounds.yMin)
      }
      const subdivisions = this.isGlobeProjection
        ? Math.max(
            MIN_SUBDIVISIONS,
            Math.min(MAX_SUBDIVISIONS, Math.ceil(latSpanDegrees))
          )
        : MERCATOR_SUBDIVISIONS

      if (this.crs === 'EPSG:4326') {
        // EPSG:4326 datasets: use fragment shader reprojection
        // Mesh is in Mercator space, fragment shader inverts Mercator → lat for texture lookup
        const subdivided = createSubdividedQuad(subdivisions)
        region.vertexArr = subdivided.vertexArr
        region.pixCoordArr = subdivided.texCoordArr
        region.vertexCount = subdivided.vertexArr.length / 2

        // Compute Mercator bounds for vertex positioning
        // geoBounds for 4326 data: xMin/xMax = lon, yMin/yMax = lat
        const latMin = geoBounds.yMin
        const latMax = geoBounds.yMax
        region.mercatorBounds = {
          x0: lonToMercatorNorm(geoBounds.xMin),
          x1: lonToMercatorNorm(geoBounds.xMax),
          y0: latToMercatorNorm(latMax),
          y1: latToMercatorNorm(latMin),
          // Include lat bounds for fragment shader reprojection
          latMin,
          latMax,
        }

        // Store data orientation for fragment shader
        region.latIsAscending = this.latIsAscending
      } else {
        const subdivided = createSubdividedQuad(subdivisions)
        region.vertexArr = subdivided.vertexArr
        region.vertexCount = subdivided.vertexArr.length / 2

        // EPSG:3857 and other Mercator-compatible CRS
        // Texture coords need V-flip if latitude is ascending
        region.pixCoordArr = this.latIsAscending
          ? flipTexCoordV(subdivided.texCoordArr)
          : subdivided.texCoordArr
      }
    }

    // Create/update buffers
    if (!region.vertexBuffer) {
      region.vertexBuffer = gl.createBuffer()
    }
    if (!region.pixCoordBuffer) {
      region.pixCoordBuffer = gl.createBuffer()
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, region.vertexBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, region.vertexArr, gl.STATIC_DRAW)
    gl.bindBuffer(gl.ARRAY_BUFFER, region.pixCoordBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, region.pixCoordArr, gl.STATIC_DRAW)

    // Upload index buffer for adaptive mesh
    if (region.useIndexedMesh && region.indexArr) {
      if (!region.indexBuffer) {
        region.indexBuffer = gl.createBuffer()
      }
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, region.indexBuffer)
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, region.indexArr, gl.STATIC_DRAW)
    }
  }

  /**
   * Classify a dimension by its name.
   * Used to identify spatial (lat/lon) vs non-spatial dimensions.
   */
  private classifyDimension(dimKey: string): 'lon' | 'lat' | 'time' | 'other' {
    const key = dimKey.toLowerCase()
    if (key === 'lon' || key === 'x' || key === 'lng' || key.includes('lon')) {
      return 'lon'
    }
    if (key === 'lat' || key === 'y' || key.includes('lat')) {
      return 'lat'
    }
    if (key.includes('time')) {
      return 'time'
    }
    return 'other'
  }

  /**
   * Build slice arguments from a selector for all dimensions.
   * Shared logic used by both display (buildBaseSliceArgs) and queries (fetchDataForSelector).
   */
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
    }
  ): Promise<{
    sliceArgs: (number | zarr.Slice)[]
    multiValueDims: Array<{
      dimIndex: number
      dimName: string
      values: number[]
      labels: (number | string)[]
    }>
  }> {
    const { array } = options

    const sliceArgs: (number | zarr.Slice)[] = new Array(
      array.shape.length
    ).fill(0)

    const multiValueDims: Array<{
      dimIndex: number
      dimName: string
      values: number[]
      labels: (number | string)[]
    }> = []

    const dimNames = Object.keys(this.dimIndices)

    for (const dimName of dimNames) {
      const dimInfo = this.dimIndices[dimName]
      const dimType = this.classifyDimension(dimName)

      if (dimType === 'lon') {
        if (options.spatialBounds) {
          sliceArgs[dimInfo.index] = zarr.slice(
            options.spatialBounds.minX,
            options.spatialBounds.maxX
          )
        } else {
          sliceArgs[dimInfo.index] = options.includeSpatialSlices
            ? zarr.slice(0, array.shape[dimInfo.index] ?? 0)
            : 0
        }
      } else if (dimType === 'lat') {
        if (options.spatialBounds) {
          sliceArgs[dimInfo.index] = zarr.slice(
            options.spatialBounds.minY,
            options.spatialBounds.maxY
          )
        } else {
          sliceArgs[dimInfo.index] = options.includeSpatialSlices
            ? zarr.slice(0, array.shape[dimInfo.index] ?? 0)
            : 0
        }
      } else {
        // Non-spatial dimension: resolve selector value
        const selectionSpec =
          selector[dimName] ||
          (dimType === 'time' ? selector['time'] : undefined)

        if (selectionSpec !== undefined) {
          const selectionValue = selectionSpec.selected
          const selectionType = selectionSpec.type

          // Check for multi-value selector
          if (
            options.trackMultiValue &&
            Array.isArray(selectionValue) &&
            selectionValue.length > 1
          ) {
            const resolvedIndices: number[] = []
            const labelValues: (number | string)[] = []
            for (const val of selectionValue) {
              const idx = await this.resolveSelectionIndex(
                dimName,
                dimInfo,
                val,
                selectionType
              )
              resolvedIndices.push(idx)
              labelValues.push(val)
            }
            multiValueDims.push({
              dimIndex: dimInfo.index,
              dimName,
              values: resolvedIndices,
              labels: labelValues,
            })
            sliceArgs[dimInfo.index] = resolvedIndices[0]
          } else {
            // Single value (or first value if array)
            const primaryValue = Array.isArray(selectionValue)
              ? selectionValue[0]
              : selectionValue

            sliceArgs[dimInfo.index] = await this.resolveSelectionIndex(
              dimName,
              dimInfo,
              primaryValue,
              selectionType
            )
          }
        } else {
          sliceArgs[dimInfo.index] = 0
        }
      }
    }

    return { sliceArgs, multiValueDims }
  }

  /**
   * Reset visible region state after a level switch.
   * This clears stale coordinates from the previous level and forces
   * a fresh viewport calculation on the next update.
   * Note: We intentionally do NOT clear visibleRegionKeys here - old regions
   * need eviction protection until new level's regions are computed.
   */
  private resetVisibleRegions(): void {
    this.lastVisibleRegions = []
    this.lastVisibleRegionsLevel = -1
    this.lastViewportHash = ''
  }

  /**
   * Update visible regions based on current viewport.
   */
  private updateVisibleRegions(map: MapLike, gl: WebGL2RenderingContext): void {
    const visible = this.getVisibleRegions(map)
    this.lastVisibleRegions = visible
    this.lastVisibleRegionsLevel = this.activeLevel?.index ?? -1
    const levelIndex = this.activeLevel?.index ?? -1

    // Add new level's visible region keys (protected from eviction)
    // Don't clear old keys yet - they protect fallback regions during transitions
    for (const { regionX, regionY } of visible) {
      this.visibleRegionKeys.add(
        this.makeRegionKey(levelIndex, regionX, regionY)
      )
    }

    // Only clear old keys when current level fully covers viewport
    if (this.currentLevelCoversViewport()) {
      // Safe to remove protection for non-current-level regions
      const currentLevelPrefix = `${levelIndex}:`
      for (const key of this.visibleRegionKeys) {
        if (!key.startsWith(currentLevelPrefix)) {
          this.visibleRegionKeys.delete(key)
        }
      }
    }

    // Abort in-flight fetches for regions that left the viewport.
    // Only signal abort — the fetch's own catch/finally handles state cleanup.
    const visibleKeys = new Set(
      visible.map(({ regionX, regionY }) =>
        this.makeRegionKey(levelIndex, regionX, regionY)
      )
    )
    for (const [key, region] of this.regionCache) {
      if (
        region.loading &&
        region.levelIndex === levelIndex &&
        region.requestId !== null &&
        !visibleKeys.has(key)
      ) {
        this.requestCanceller.controllers.get(region.requestId)?.abort()
      }
    }

    // Separate regions into two categories:
    // 1. New regions (no data) - viewport change
    // 2. Stale regions (have data, wrong selector) - selector change
    const newRegions: Array<{ regionX: number; regionY: number }> = []
    const staleRegions: Array<{ regionX: number; regionY: number }> = []

    for (const { regionX, regionY } of visible) {
      const key = this.makeRegionKey(levelIndex, regionX, regionY)
      const cached = this.regionCache.get(key)

      // Skip if already loading - when the load completes, invalidate() triggers
      // another updateVisibleRegions() check to see if refetch is needed
      if (cached?.loading) {
        continue
      }

      if (!cached?.data) {
        // No data yet - this is a new region (viewport change)
        newRegions.push({ regionX, regionY })
      } else if (cached.selectorVersion !== this.selectorVersion) {
        // Has data but stale selector - this is a selector change
        staleRegions.push({ regionX, regionY })
      }
    }

    // Check if viewport changed (include selectorVersion and level in hash)
    const viewportHash = `${levelIndex}:${this.selectorVersion}:${visible
      .map((r) => `${r.regionX},${r.regionY}`)
      .join('|')}`
    const viewportChanged = viewportHash !== this.lastViewportHash
    this.lastViewportHash = viewportHash

    // Skip if nothing to fetch
    if (
      newRegions.length === 0 &&
      staleRegions.length === 0 &&
      !viewportChanged
    ) {
      return
    }

    if (newRegions.length > 0) {
      this.fetchRegions(newRegions, gl)
    }
    if (staleRegions.length > 0) {
      this.fetchRegions(staleRegions, gl)
    }
  }

  /**
   * Fetch multiple regions with limited concurrency to avoid overwhelming the browser.
   */
  private async fetchRegions(
    regions: Array<{ regionX: number; regionY: number }>,
    gl: WebGL2RenderingContext
  ): Promise<void> {
    // Can't fetch without a committed level.
    if (!this.activeLevel) return
    const level = this.activeLevel

    // Capture ALL level-dependent state at start to pass to fetchRegion.
    // This prevents races where a later `loadLevel` (zoom switch or
    // selector rebuild) swaps `this.activeLevel` mid-batch.
    const snapshot: LevelSnapshot = {
      index: level.index,
      zarrArray: level.zarrArray,
      baseSliceArgs: [...level.baseSliceArgs],
      width: level.width,
      height: level.height,
      regionSize: level.regionSize,
      selectorVersion: this.selectorVersion,
      bandNames: [...this.bandNames],
      baseMultiValueDims: level.baseMultiValueDims.map((dim) => ({
        dimIndex: dim.dimIndex,
        dimName: dim.dimName,
        values: [...dim.values],
        labels: [...dim.labels],
      })),
    }

    this.loadingDebouncer.show()

    // Mark ALL regions as loading upfront to prevent duplicate fetches
    // from subsequent update() calls before we've processed them all
    for (const { regionX, regionY } of regions) {
      const key = this.makeRegionKey(snapshot.index, regionX, regionY)
      let region = this.regionCache.get(key)
      if (!region) {
        region = this.createRegionState(snapshot.index, regionX, regionY)
        this.regionCache.set(key, region)
      }
      region.loading = true
    }

    // Pre-flight staleness check. Mid-flight changes are handled by the
    // `cancelAllRequests` in `loadLevel`/`setSelector`, which aborts the
    // signals that fetchRegion threads through every await.
    if (
      (this.activeLevel?.index ?? -1) !== snapshot.index ||
      this.selectorVersion !== snapshot.selectorVersion
    ) {
      cancelAllRequests(this.requestCanceller)
      this.clearBatchLoadingFlags(regions, snapshot.index)
    } else {
      // Kick off every region synchronously so their underlying chunk reads
      // land in one microtask drain — that's what lets the range coalescer
      // (icechunk-js + zarrita) merge them into a handful of HTTP fetches
      // instead of one per region. Browser HTTP queueing handles back-
      // pressure on the connection pool; throttling fetchRegion calls here
      // just fragments the coalescer's same-tick batch window.
      const fetches = regions.map(({ regionX, regionY }) =>
        this.fetchRegion(regionX, regionY, gl, snapshot)
      )
      await Promise.allSettled(fetches)
    }

    // Only update loading state if we're still on the same level
    if (!hasActiveRequests(this.requestCanceller)) {
      this.loadingDebouncer.hide()

      // Evict old regions if cache is full (LRU via Map insertion order)
      this.evictOldRegions(gl)
      this.invalidate()
    }
  }

  /**
   * Fetch data for a single region.
   * Handles multi-band extraction when selector has multi-value dimensions.
   * @param snapshot - Captured level state from when fetch batch started (prevents race conditions)
   */
  private async fetchRegion(
    regionX: number,
    regionY: number,
    gl: WebGL2RenderingContext,
    snapshot: LevelSnapshot
  ): Promise<void> {
    if ((this.activeLevel?.index ?? -1) !== snapshot.index) {
      return
    }

    if (this.isRemoved) {
      return
    }

    const key = this.makeRegionKey(snapshot.index, regionX, regionY)
    const requestId = ++this.requestCanceller.currentVersion
    const fetchSelectorVersion = snapshot.selectorVersion

    const controller = new AbortController()
    this.requestCanceller.controllers.set(requestId, controller)

    let region = this.regionCache.get(key)
    if (!region) {
      region = this.createRegionState(snapshot.index, regionX, regionY)
      this.regionCache.set(key, region)
    }
    region.loading = true
    region.requestId = requestId

    const [regionH, regionW] = snapshot.regionSize

    // Calculate pixel bounds for this region
    const yStart = regionY * regionH
    const yEnd = Math.min(yStart + regionH, snapshot.height)
    const xStart = regionX * regionW
    const xEnd = Math.min(xStart + regionW, snapshot.width)
    const actualW = xEnd - xStart
    const actualH = yEnd - yStart

    try {
      // Build base slice args with spatial region bounds
      const baseSliceArgs = [...snapshot.baseSliceArgs]
      const latIdx = this.dimIndices.lat.index
      const lonIdx = this.dimIndices.lon.index
      baseSliceArgs[latIdx] = zarr.slice(yStart, yEnd)
      baseSliceArgs[lonIdx] = zarr.slice(xStart, xEnd)

      const desc = this.zarrStore.describe()
      // Use per-level metadata if available (for heterogeneous pyramids)
      const currentLevel = this.levels[snapshot.index]
      const fillValue = currentLevel?.fillValue ?? desc.fill_value

      const { combinations: channelCombinations } =
        this.buildChannelCombinations(snapshot.baseMultiValueDims)
      const numChannels = channelCombinations.length || 1

      // Fetch data for all channels
      const bandArrays: Float32Array[] = []

      const isStale = () =>
        controller.signal.aborted ||
        this.isRemoved ||
        (this.activeLevel?.index ?? -1) !== snapshot.index

      if (numChannels === 1) {
        // Single channel - simple fetch
        if (isStale()) return

        const result = (await zarr.get(snapshot.zarrArray, baseSliceArgs, {
          signal: controller.signal,
        })) as { data: ArrayLike<number> }

        if (isStale()) return

        const rawData = new Float32Array(result.data as ArrayLike<number>)
        bandArrays.push(rawData)
      } else {
        // Multi-channel - fetch all channels in parallel
        if (isStale()) return

        // Build slice args for all channels upfront
        const allSliceArgs: (number | zarr.Slice)[][] = []
        for (let c = 0; c < numChannels; c++) {
          const sliceArgs = [...baseSliceArgs]
          const combo = channelCombinations[c]

          // Apply channel-specific indices to multi-value dimensions
          for (let i = 0; i < snapshot.baseMultiValueDims.length; i++) {
            sliceArgs[snapshot.baseMultiValueDims[i].dimIndex] = combo[i]
          }
          allSliceArgs.push(sliceArgs)
        }

        // Fetch all bands in parallel
        const results = await Promise.all(
          allSliceArgs.map((sliceArgs) =>
            zarr.get(snapshot.zarrArray, sliceArgs, {
              signal: controller.signal,
            })
          )
        )

        if (isStale()) return

        // Process results in order
        for (let c = 0; c < numChannels; c++) {
          const result = results[c] as { data: ArrayLike<number> }
          const bandData = new Float32Array(result.data as ArrayLike<number>)
          bandArrays.push(bandData)
        }
      }

      // Only render if this is newer than what's already rendered for this region
      if (fetchSelectorVersion < region.selectorVersion) return

      // Update region's selector version
      region.selectorVersion = fetchSelectorVersion

      // Resample bands to Mercator space if needed (EPSG:4326 or custom projection)
      // For non-EPSG:3857 datasets: GPU handles reprojection
      // - Proj4 datasets: adaptive mesh (CRS→4326), GPU projects to Mercator or ECEF
      // - EPSG:4326 datasets: subdivided quad, GPU reprojects via fragment or ECEF
      // No CPU resampling needed - mercatorBounds computed in createRegionGeometry
      const needsProj4MercBounds =
        this.proj4def && this.cachedMercatorTransformer

      if (needsProj4MercBounds && this.xyLimits && !region.mercatorBounds) {
        const levelMeta: LevelMeta = {
          width: snapshot.width,
          height: snapshot.height,
          regionSize: snapshot.regionSize,
        }
        const geoBounds = this.getRegionBounds(regionX, regionY, levelMeta)
        region.mercatorBounds = this.computeRegionMercatorBounds(geoBounds)
      }

      // Apply per-level scale/offset to convert raw values to physical units
      // Fall back to dataset-level scale/offset for pyramids that only define them at the root
      const scaleFactor = currentLevel?.scaleFactor ?? desc.scaleFactor
      const addOffset = currentLevel?.addOffset ?? desc.addOffset

      // Normalize bands (single pass) and collect for interleaving
      region.bandData.clear()
      region.bandTexturesUploaded.clear()
      const normalizedBands: Float32Array[] = []

      for (let c = 0; c < bandArrays.length; c++) {
        const bandName = snapshot.bandNames[c] || `band_${c}`
        let bandData = bandArrays[c]

        // Apply scale/offset if needed (converts raw to physical values)
        if (scaleFactor !== 1 || addOffset !== 0) {
          const scaled = new Float32Array(bandData.length)
          for (let i = 0; i < bandData.length; i++) {
            const raw = bandData[i]
            // Scale all values including fill - normalizeDataForTexture will filter by scaled fill
            if (!Number.isFinite(raw)) {
              scaled[i] = raw // Keep NaN/Inf as-is
            } else {
              scaled[i] = raw * scaleFactor + addOffset
            }
          }
          bandData = scaled
        }

        // Compute the fill value in the same space as the data
        const effectiveFillValue =
          fillValue !== null && (scaleFactor !== 1 || addOffset !== 0)
            ? fillValue * scaleFactor + addOffset
            : fillValue

        const { normalized: bandNormalized } = normalizeDataForTexture(
          bandData,
          effectiveFillValue,
          this.fixedDataScale
        )
        region.bandData.set(bandName, bandNormalized)
        normalizedBands.push(bandNormalized)
      }

      // Construct interleaved data from normalized bands
      region.data = interleaveBands(normalizedBands, numChannels)

      // Check if geometry needs to be (re)created before updating dimensions
      // The adaptive mesh only depends on spatial bounds and dimensions, not the selector
      const needsGeometry =
        !region.vertexBuffer ||
        region.width !== actualW ||
        region.height !== actualH

      region.width = actualW
      region.height = actualH
      region.channels = numChannels
      region.loading = false

      // Store level-specific dimensions from snapshot for geometry rebuild.
      // Must use snapshot (not this.*) to avoid races with level switching.
      // Set before createRegionGeometry is called below.
      region.levelMeta = {
        width: snapshot.width,
        height: snapshot.height,
        regionSize: [...snapshot.regionSize] as [number, number],
      }

      // Create/update main texture for this region
      if (!region.texture) {
        region.texture = gl.createTexture()
      }

      // Upload texture using shared helper
      const result = uploadDataTexture(gl, {
        texture: region.texture!,
        data: region.data!,
        width: actualW,
        height: actualH,
        channels: numChannels,
        configured: false,
      })
      region.textureUploaded = result.uploaded

      // Create geometry only if needed (new region or dimensions changed)
      if (needsGeometry) {
        this.createRegionGeometry(regionX, regionY, gl, region)
      }

      this.invalidate()
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        console.error(`[fetchRegion] Error fetching region ${key}:`, err)
      }
    } finally {
      region.loading = false
      region.requestId = null
      this.requestCanceller.controllers.delete(requestId)
      // Re-evaluate visible regions after abort so panned-back regions get re-fetched.
      if (controller.signal.aborted && !this.isRemoved) {
        this.invalidate()
      }
    }
  }

  update(map: MapLike, gl: WebGL2RenderingContext): void {
    // Cache gl context for use in setSelector
    this.cachedGl = gl

    // Don't proceed if metadata is still loading
    if (this.loadingManager.metadataLoading) {
      return
    }

    // Pick target: zoom-selected for multiscale, single level otherwise.
    if (this.isMultiscale && this.levels.length > 0) {
      const mapZoom = map.getZoom?.() ?? 0
      this.desiredLevelIndex = this.selectLevelForZoom(mapZoom)
    } else {
      this.desiredLevelIndex = 0
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

  /**
   * Unified level load: handles initial load, zoom-driven switch, and
   * selector-driven slice-args rebuild. Builds a `LevelRuntime` off to
   * the side and swaps it into `this.activeLevel` atomically, so readers
   * never see a half-committed level.
   *
   * `reuseArray` reuses the current committed array/dims — used by
   * `setSelector` to rebuild slice args without refetching. `loadToken`
   * acts as a cancellation token: any load whose token is stale at
   * commit time drops its result.
   */
  private async loadLevel(
    levelIndex: number,
    { reuseArray = false }: { reuseArray?: boolean } = {}
  ): Promise<void> {
    if (this.isMultiscale && this.levels.length > 0) {
      if (levelIndex < 0 || levelIndex >= this.levels.length) return
    } else if (levelIndex !== 0) {
      return
    }

    // Dedupe: an in-flight load for the same target is already on it.
    // A selector rebuild (`reuseArray`) intentionally supersedes.
    if (this.loadingLevelIndex === levelIndex && !reuseArray) {
      return
    }

    const token = ++this.loadToken
    // Snapshot `this.selector` so we can detect a concurrent `setSelector`
    // that arrived after `buildSliceArgsForSelector` resolved; committing
    // old slice args with a new selector would leak stale data.
    const selectorSnapshot = this.selector
    this.loadingLevelIndex = levelIndex

    // Cancel any in-flight region fetches — they were tied to the old
    // level's array/dims (or old selector) and can't be reused.
    if (this.requestCanceller.controllers.size > 0) {
      cancelAllRequests(this.requestCanceller)
      this.loadingDebouncer.hide()
    }

    try {
      const existing = this.activeLevel
      const canReuseArray =
        reuseArray && existing !== null && existing.index === levelIndex

      let newArray: zarr.Array<zarr.DataType>
      let newWidth: number
      let newHeight: number
      let newRegionSize: [number, number]

      if (canReuseArray) {
        newArray = existing!.zarrArray
        newWidth = existing!.width
        newHeight = existing!.height
        newRegionSize = existing!.regionSize
      } else {
        if (this.isMultiscale) {
          await this.ensureLevelMetadata(levelIndex)
          const level = this.levels[levelIndex]
          newArray = await this.zarrStore.getLevelArray(level.asset)
        } else {
          newArray = await this.zarrStore.getArray()
        }
        newWidth = newArray.shape[this.dimIndices.lon.index]
        newHeight = newArray.shape[this.dimIndices.lat.index]
        const detected = this.getRegionSize(newArray)
        newRegionSize = detected ?? [newHeight, newWidth]
      }

      const { sliceArgs, multiValueDims } =
        await this.buildSliceArgsForSelector(selectorSnapshot, {
          includeSpatialSlices: false,
          trackMultiValue: true,
          array: newArray,
        })

      const targetStillDesired =
        reuseArray ||
        !this.isMultiscale ||
        levelIndex === this.desiredLevelIndex

      // Drop on the floor if anything raced past us: a newer load (or
      // dispose) bumped the token, the zoom target moved on, or
      // `setSelector` replaced the selector we built slice args against.
      if (
        token !== this.loadToken ||
        this.isRemoved ||
        this.selector !== selectorSnapshot ||
        !targetStillDesired
      ) {
        this.invalidate()
        return
      }

      // Atomic commit — one reference swap replaces all per-level state.
      this.activeLevel = {
        index: levelIndex,
        zarrArray: newArray,
        width: newWidth,
        height: newHeight,
        regionSize: newRegionSize,
        baseSliceArgs: sliceArgs,
        baseMultiValueDims: multiValueDims,
      }

      // Don't clear the region cache on level changes — older-level
      // regions serve as fallback rendering while the new level's
      // regions load, and the LRU (`evictOldRegions`) disposes them
      // properly once they're no longer protected by `visibleRegionKeys`.
      // Bare `.clear()` here would leak WebGL textures/buffers.
      if (!canReuseArray) {
        this.resetVisibleRegions()
      }

      this.invalidate()
    } catch (err) {
      if (token === this.loadToken) {
        const assetLabel = this.isMultiscale
          ? this.levels[levelIndex]?.asset ?? String(levelIndex)
          : 'single-level'
        console.error(`Failed to load level ${assetLabel}:`, err)
      }
    } finally {
      if (token === this.loadToken) {
        this.loadingLevelIndex = null
      }
    }
  }

  private selectLevelForZoom(mapZoom: number): number {
    if (!this.xyLimits || this.levels.length === 0) return 0

    // Calculate map resolution: at zoom Z, full world is 256 * 2^Z pixels
    const mapPixelsPerWorld = 256 * Math.pow(2, mapZoom)

    // Calculate what fraction of the world the data covers, accounting for CRS
    let worldFraction: number
    if (this.proj4def && this.cachedMercatorTransformer) {
      // Custom projection: transform bounds corners to mercator
      const [minMercX] = this.cachedMercatorTransformer.forward(
        this.xyLimits.xMin,
        this.xyLimits.yMin
      )
      const [maxMercX] = this.cachedMercatorTransformer.forward(
        this.xyLimits.xMax,
        this.xyLimits.yMax
      )
      const dataWidthMeters = Math.abs(maxMercX - minMercX)
      const fullWorldMeters = 2 * WEB_MERCATOR_EXTENT
      worldFraction = dataWidthMeters / fullWorldMeters
    } else if (this.crs === 'EPSG:3857') {
      // Web Mercator: full world is ~40,075,016 meters
      const dataWidth = this.xyLimits.xMax - this.xyLimits.xMin
      const fullWorldMeters = 2 * WEB_MERCATOR_EXTENT
      worldFraction = dataWidth / fullWorldMeters
    } else {
      // EPSG:4326: full world is 360 degrees
      const dataWidth = this.xyLimits.xMax - this.xyLimits.xMin
      worldFraction = dataWidth / 360
    }

    // Build list of levels with their effective resolution (pixels per full world)
    const levelResolutions: Array<{ index: number; effectivePixels: number }> =
      []
    for (let i = 0; i < this.levels.length; i++) {
      const level = this.levels[i]
      if (!level.shape) continue
      const lonIndex = this.dimIndices.lon?.index ?? level.shape.length - 1
      // Scale up to what resolution would be if data covered full world
      const effectivePixels = level.shape[lonIndex] / worldFraction
      levelResolutions.push({ index: i, effectivePixels })
    }

    // If no levels have shape data yet, return last index (lowest res for untiled)
    if (levelResolutions.length === 0) return this.levels.length - 1

    // Sort by resolution ascending (lowest res first)
    levelResolutions.sort((a, b) => a.effectivePixels - b.effectivePixels)

    // Find the lowest resolution level that still provides sufficient detail
    for (const { index, effectivePixels } of levelResolutions) {
      if (effectivePixels >= mapPixelsPerWorld) {
        return index
      }
    }

    // If no level is sufficient, use the highest resolution available
    return levelResolutions[levelResolutions.length - 1].index
  }

  render(renderer: ZarrRenderer, context: RenderContext): void {
    const useMapbox = !!context.mapbox
    // Use wgs84 shader only for proj4 datasets (vertex shader reprojection)
    // EPSG:4326 uses fragment shader reprojection instead
    const useWgs84 = !!this.proj4def && !!this.cached4326Transformer

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
    const ecefEligible = useWgs84 || this.crs === 'EPSG:4326'
    const useDirectEcef = useMapbox
      ? hasMapboxDirectGlobePath && ecefEligible
      : hasMaplibreGlobeTransition && ecefEligible

    // Flush deferred geometry rebuild once transition completes.
    // proj4 uses hybrid mesh which is projection-independent, so skip rebuilds.
    if (this.pendingGeometryRebuild) {
      if (useWgs84) {
        this.pendingGeometryRebuild = false
      } else if (!hasMaplibreGlobeTransition && !hasMapboxGlobe) {
        this.pendingGeometryRebuild = false
        this.rebuildAllGeometry()
      }
    }

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

    this.renderRegions(
      renderer,
      shaderProgram,
      worldOffsets,
      context.customShaderConfig,
      useDirectEcef
    )
  }

  /**
   * Convert a RegionState to a RenderableRegion for unified rendering.
   * When useDirectEcef is true, computes WGS84 bounds and sets positionSpace/sampleMode
   * for the ECEF vertex shader path. These fields are computed at render time,
   * never cached on RegionState, so projection toggles have no stale state.
   */
  private regionToRenderable(
    region: RegionState,
    useDirectEcef: boolean = false
  ): RenderableRegion {
    const base: RenderableRegion = {
      mercatorBounds: region.mercatorBounds!,
      vertexBuffer: region.vertexBuffer!,
      pixCoordBuffer: region.pixCoordBuffer!,
      vertexCount: region.useIndexedMesh
        ? region.vertexCount
        : region.vertexArr!.length / 2,
      indexBuffer: region.indexBuffer,
      useIndexedMesh: region.useIndexedMesh,
      wgs84Bounds: region.wgs84Bounds ?? undefined,
      latIsAscending: region.latIsAscending,
      texture: region.texture!,
      bandData: region.bandData,
      bandTextures: region.bandTextures,
      bandTexturesUploaded: region.bandTexturesUploaded,
      bandTexturesConfigured: region.bandTexturesConfigured,
      width: region.width,
      height: region.height,
    }

    // Globe ECEF for EPSG:4326: compute WGS84 bounds on the fly from unclamped lat bounds
    if (
      useDirectEcef &&
      this.crs === 'EPSG:4326' &&
      region.mercatorBounds?.latMin != null &&
      region.mercatorBounds?.latMax != null
    ) {
      const mb = region.mercatorBounds!
      base.wgs84Bounds = {
        lon0: mb.x0, // lon mapping is linear, same as Mercator X
        lat0: latToWgs84Norm(mb.latMin!),
        lon1: mb.x1,
        lat1: latToWgs84Norm(mb.latMax!),
      }
      base.positionSpace = 'wgs84-ecef'
      base.sampleMode = 'wgs84-lookup'
      return base
    }

    // Globe ECEF for proj4: wgs84Bounds already set by createHybridMesh
    if (useDirectEcef && region.wgs84Bounds) {
      base.positionSpace = 'wgs84-ecef'
      base.sampleMode = 'linear'
      return base
    }

    return base // defaults handle all other cases
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
    useDirectEcef: boolean = false
  ): void {
    const gl = renderer.gl

    // Set up band texture uniforms once per frame
    setupBandTextureUniforms(gl, shaderProgram, customShaderConfig)

    // Render each loaded region using unified path
    for (const region of this.getLoadedRegions()) {
      renderRegion(
        gl,
        shaderProgram,
        this.regionToRenderable(region, useDirectEcef),
        worldOffsets,
        customShaderConfig
      )
    }
  }

  renderToTile(
    renderer: ZarrRenderer,
    tileId: TileId,
    context: RenderContext
  ): boolean {
    // This method is only used for draped Mapbox rendering. The direct
    // untiled ECEF path disables renderToTile at the layer level.
    return renderMapboxTile({
      renderer,
      mode: this,
      tileId,
      context: {
        ...context,
        uniforms: this.getUniformsForRender(context.uniforms),
      },
      regions: this.getRegionStates(),
    })
  }

  onProjectionChange(isGlobe: boolean): void {
    if (this.isGlobeProjection === isGlobe) return
    this.isGlobeProjection = isGlobe

    // proj4 uses hybrid mesh which is projection-independent — no rebuild needed.
    if (this.proj4def) return

    if (!isGlobe) {
      // Globe→flat: defer geometry rebuild until projectionTransition reaches 0.
      // Rebuilding now would drop subdivisions to 1 while the ECEF shader is still
      // rendering on the globe, causing severe faceting during the transition.
      this.pendingGeometryRebuild = true
      return
    }

    // Flat→globe: rebuild immediately with high subdivisions for smooth globe rendering
    this.rebuildAllGeometry()
  }

  private rebuildAllGeometry(): void {
    const gl = this.cachedGl
    if (!gl) return
    for (const region of this.regionCache.values()) {
      if (!region.data) continue
      this.createRegionGeometry(region.regionX, region.regionY, gl, region)
    }
    this.invalidate()
  }

  getTiledState() {
    return null
  }

  /**
   * Get render states for all loaded regions (for multi-region rendering).
   * Includes previous level regions as fallback during level transitions.
   */
  private getRegionStates(): RegionRenderState[] {
    if (!(this.activeLevel?.regionSize ?? null)) {
      return []
    }

    return this.getLoadedRegions().map((region) => ({
      texture: region.texture!,
      vertexBuffer: region.vertexBuffer!,
      pixCoordBuffer: region.pixCoordBuffer!,
      vertexArr: region.vertexArr!,
      mercatorBounds: region.mercatorBounds!,
      width: region.width,
      height: region.height,
      channels: this.channels,
      bandData: region.bandData,
      bandTextures: region.bandTextures,
      bandTexturesUploaded: region.bandTexturesUploaded,
      bandTexturesConfigured: region.bandTexturesConfigured,
      // Indexed mesh fields for proj4 adaptive mesh
      indexBuffer: region.indexBuffer ?? undefined,
      vertexCount: region.vertexCount,
      useIndexedMesh: region.useIndexedMesh,
      wgs84Bounds: region.wgs84Bounds ?? undefined,
      latIsAscending: region.latIsAscending,
    }))
  }

  dispose(gl: WebGL2RenderingContext | WebGLRenderingContext): void {
    this.isRemoved = true
    // Bump so any pending `loadLevel` drops its result on commit.
    this.loadToken++
    this.loadingLevelIndex = null
    cancelAllRequests(this.requestCanceller)
    // Clean up region caches
    this.clearRegionCache(gl)
    this.activeLevel = null
    this.cachedMercatorTransformer = null
    this.cachedWGS84Transformer = null
    this.cached4326Transformer = null
    this.loadingDebouncer.hide()
  }

  setLoadingCallback(callback: LoadingStateCallback | undefined): void {
    setLoadingCallbackUtil(this.loadingManager, callback)
  }

  getCRS(): CRS {
    return this.crs
  }

  getXYLimits(): XYLimits | null {
    return this.xyLimits
  }

  /**
   * Compute mercator bounds from proj4 by sampling edge points.
   */
  private computeMercatorBoundsFromProjection(): MercatorBounds {
    if (!this.proj4def || !this.xyLimits || !this.cachedMercatorTransformer) {
      return { x0: 0, y0: 0, x1: 1, y1: 1 }
    }
    const result = sampleEdgesToMercatorBounds(
      this.xyLimits,
      this.cachedMercatorTransformer,
      20
    )
    if (!result) {
      console.warn(
        'computeMercatorBoundsFromProjection: No valid samples found'
      )
      return { x0: 0, y0: 0, x1: 1, y1: 1 }
    }
    return result
  }

  /**
   * Compute mercator bounds for a specific region from source CRS bounds.
   */
  private computeRegionMercatorBounds(bounds: {
    xMin: number
    xMax: number
    yMin: number
    yMax: number
  }): MercatorBounds {
    if (!this.proj4def || !this.cachedMercatorTransformer) {
      return { x0: 0, y0: 0, x1: 1, y1: 1 }
    }
    const result = sampleEdgesToMercatorBounds(
      bounds,
      this.cachedMercatorTransformer,
      5
    )
    if (!result) {
      console.warn('computeRegionMercatorBounds: No valid samples found')
      return { x0: 0, y0: 0, x1: 1, y1: 1 }
    }
    return result
  }

  getMaxLevelIndex(): number {
    return this.levels.length > 0 ? this.levels.length - 1 : 0
  }

  getLevels(): string[] {
    return this.levels.map((l) => l.asset)
  }

  async setSelector(selector: NormalizedSelector): Promise<void> {
    this.selector = selector
    this.bandNames = getBands(this.variable, selector)

    if (!this.cachedGl) {
      // No gl context yet — selector is stored, update() will handle loading.
      this.invalidate()
      return
    }

    // Abort in-flight region fetches still running with the old selector.
    // Their catch/finally handles state cleanup and re-invalidation.
    for (const [, region] of this.regionCache) {
      if (region.loading && region.requestId !== null) {
        this.requestCanceller.controllers.get(region.requestId)?.abort()
      }
    }

    // Rebuild the runtime with the new selector and atomic-swap. Going
    // through `loadLevel` bumps `loadToken`, so any in-flight level load
    // (zoom-driven or initial) drops and restarts against the new
    // selector. `reuseArray` keeps the array/dims so no refetch is done
    // when the level itself isn't changing.
    if (this.activeLevel) {
      await this.loadLevel(this.activeLevel.index, { reuseArray: true })
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

  private emitLoadingState(): void {
    emitLoadingStateUtil(this.loadingManager)
  }

  private async resolveSelectionIndex(
    dimName: string,
    dimInfo: {
      index: number
      name: string
      array: zarr.Array<zarr.DataType> | null
    },
    value: number | string | [number, number] | undefined,
    type?: 'index' | 'value'
  ): Promise<number> {
    if (type === 'index') {
      return typeof value === 'number' ? value : 0
    }

    if (!this.zarrStore.root) {
      return typeof value === 'number' ? value : 0
    }

    try {
      const coords = await loadDimensionValues(
        this.dimensionValues,
        null,
        dimInfo,
        this.zarrStore.root,
        this.zarrStore.version
      )
      this.dimensionValues[dimName] = coords

      if (typeof value === 'number' || typeof value === 'string') {
        const coordIdx = (coords as (number | string)[]).indexOf(value)
        if (coordIdx >= 0) return coordIdx
        throw new Error(
          `[ZarrLayer] Selector value '${value}' not found in coordinate array for dimension '${dimName}'. ` +
            `Available values: [${(coords as (number | string)[])
              .slice(0, 10)
              .join(', ')}${coords.length > 10 ? ', ...' : ''}]. ` +
            `Use { selected: <index>, type: 'index' } to select by array index instead.`
        )
      }
    } catch (err) {
      console.debug(`Could not resolve coordinate for '${dimName}':`, err)
    }

    return typeof value === 'number' ? value : 0
  }

  /**
   * Unified method to fetch query data for either point or region queries.
   * Handles multi-value dimensions and channel combinations.
   */
  private async fetchQueryData(
    level: QueryLevelSnapshot,
    selector: NormalizedSelector,
    spatialQuery: {
      minX: number
      maxX: number
      minY: number
      maxY: number
    },
    signal?: AbortSignal
  ): Promise<{
    data: Float32Array
    width: number
    height: number
    channels: number
    channelLabels: (string | number)[][]
    multiValueDimNames: string[]
  } | null> {
    try {
      const { sliceArgs: baseSliceArgs, multiValueDims } =
        await this.buildSliceArgsForSelector(selector, {
          includeSpatialSlices: false,
          trackMultiValue: true,
          spatialBounds: spatialQuery,
          array: level.zarrArray,
        })

      const {
        combinations: channelCombinations,
        labelCombinations: channelLabelCombinations,
      } = this.buildChannelCombinations(multiValueDims)
      const numChannels = channelCombinations.length || 1
      const multiValueDimNames = multiValueDims.map((d) => d.dimName)
      const getOpts = signal ? { signal } : undefined

      const fetchWidth = spatialQuery.maxX - spatialQuery.minX
      const fetchHeight = spatialQuery.maxY - spatialQuery.minY

      if (numChannels === 1) {
        const result = (await zarr.get(
          level.zarrArray,
          baseSliceArgs,
          getOpts
        )) as { data: ArrayLike<number> }
        return {
          data: new Float32Array(result.data),
          width: fetchWidth,
          height: fetchHeight,
          channels: 1,
          channelLabels: channelLabelCombinations,
          multiValueDimNames,
        }
      }

      const packedData = new Float32Array(
        fetchWidth * fetchHeight * numChannels
      )
      for (let c = 0; c < numChannels; c++) {
        const sliceArgs = [...baseSliceArgs]
        const combo = channelCombinations[c]
        for (let i = 0; i < multiValueDims.length; i++) {
          sliceArgs[multiValueDims[i].dimIndex] = combo[i]
        }

        const bandData = (await zarr.get(
          level.zarrArray,
          sliceArgs,
          getOpts
        )) as { data: ArrayLike<number> }
        for (let pixIdx = 0; pixIdx < fetchWidth * fetchHeight; pixIdx++) {
          packedData[pixIdx * numChannels + c] = bandData.data[pixIdx]
        }
      }

      return {
        data: packedData,
        width: fetchWidth,
        height: fetchHeight,
        channels: numChannels,
        channelLabels: channelLabelCombinations,
        multiValueDimNames,
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') throw err
      console.error('Error fetching query data:', err)
      return null
    }
  }

  /**
   * Compute subset bounds from pixel bounds against the full mercator bounds.
   */
  private computeSubsetBounds(
    pixelBounds: PixelRect,
    level: QueryLevelSnapshot
  ): MercatorBounds {
    const { minX, maxX, minY, maxY } = pixelBounds
    const xRange = this.mercatorBounds!.x1 - this.mercatorBounds!.x0
    const yRange = this.mercatorBounds!.y1 - this.mercatorBounds!.y0
    const subsetBounds: MercatorBounds = {
      x0: this.mercatorBounds!.x0 + (minX / level.width) * xRange,
      x1: this.mercatorBounds!.x0 + (maxX / level.width) * xRange,
      y0: this.mercatorBounds!.y0 + (minY / level.height) * yRange,
      y1: this.mercatorBounds!.y0 + (maxY / level.height) * yRange,
    }
    if (
      this.mercatorBounds!.latMin !== undefined &&
      this.mercatorBounds!.latMax !== undefined
    ) {
      const latRange = this.mercatorBounds!.latMax - this.mercatorBounds!.latMin
      if (this.latIsAscending) {
        subsetBounds.latMin =
          this.mercatorBounds!.latMin + (minY / level.height) * latRange
        subsetBounds.latMax =
          this.mercatorBounds!.latMin + (maxY / level.height) * latRange
      } else {
        subsetBounds.latMax =
          this.mercatorBounds!.latMax - (minY / level.height) * latRange
        subsetBounds.latMin =
          this.mercatorBounds!.latMax - (maxY / level.height) * latRange
      }
    }
    return subsetBounds
  }

  /**
   * Query data for point or region geometries.
   */
  async queryData(
    geometry: QueryGeometry,
    selector?: Selector,
    options?: QueryOptions
  ): Promise<QueryResult> {
    const emptyResult = (): QueryResult => ({
      [this.variable]: [],
      dimensions: [],
      coordinates: { lat: [], lon: [] },
    })

    const activeLevel = this.activeLevel
    if (!this.mercatorBounds || !activeLevel) return emptyResult()

    const level: QueryLevelSnapshot = {
      index: activeLevel.index,
      zarrArray: activeLevel.zarrArray,
      width: activeLevel.width,
      height: activeLevel.height,
    }

    const normalizedSelector = selector
      ? normalizeSelector(selector)
      : this.selector

    const desc = this.zarrStore.describe()
    const currentLevel = this.levels[level.index]
    const transforms = {
      scaleFactor: currentLevel?.scaleFactor ?? desc.scaleFactor,
      addOffset: currentLevel?.addOffset ?? desc.addOffset,
      fillValue: currentLevel?.fillValue ?? desc.fill_value,
    }
    const sourceBounds: [number, number, number, number] | null = this.xyLimits
      ? [
          this.xyLimits.xMin,
          this.xyLimits.yMin,
          this.xyLimits.xMax,
          this.xyLimits.yMax,
        ]
      : null

    // Closure for running a single pixel-bounds strip query.
    // Captures request-scoped locals (not instance state) to avoid races
    // when queryData is called concurrently on the same instance.
    const runStrip = async (
      geom: QueryGeometry,
      pixelBounds: PixelRect,
      opts?: QueryOptions
    ): Promise<QueryResult | null> => {
      const fetched = await this.fetchQueryData(
        level,
        normalizedSelector,
        pixelBounds,
        opts?.signal
      )
      if (!fetched) return null

      const subsetBounds = this.computeSubsetBounds(pixelBounds, level)

      let subsetSourceBounds: [number, number, number, number] | null = null
      if (this.proj4def && sourceBounds) {
        const { minX, minY, maxX, maxY } = pixelBounds
        const [xMin, yMin] = pixelToSourceCRS(
          minX,
          minY,
          sourceBounds,
          level.width,
          level.height,
          this.latIsAscending
        )
        const [xMax, yMax] = pixelToSourceCRS(
          maxX,
          maxY,
          sourceBounds,
          level.width,
          level.height,
          this.latIsAscending
        )
        subsetSourceBounds = [
          Math.min(xMin, xMax),
          Math.min(yMin, yMax),
          Math.max(xMin, xMax),
          Math.max(yMin, yMax),
        ]
      }

      return queryRegionUntiled(
        this.variable,
        geom,
        normalizedSelector,
        fetched.data,
        fetched.width,
        fetched.height,
        subsetBounds,
        this.crs ?? 'EPSG:4326',
        desc.dimensions,
        desc.coordinates,
        fetched.channels,
        fetched.channelLabels,
        fetched.multiValueDimNames,
        this.latIsAscending,
        transforms,
        this.proj4def,
        subsetSourceBounds,
        opts,
        desc.dimIndices
      )
    }

    // Helper for the single-fetch path shared by proj4 and non-crossing cases
    const singleFetch = async (geom: QueryGeometry): Promise<QueryResult> => {
      const pixelBounds = computePixelBoundsFromGeometry(
        geom,
        this.mercatorBounds!,
        level.width,
        level.height,
        this.crs ?? 'EPSG:4326',
        this.latIsAscending,
        this.proj4def,
        sourceBounds,
        this.cachedWGS84Transformer ?? undefined
      )
      if (!pixelBounds) return emptyResult()
      const result = await runStrip(geom, pixelBounds, options)
      return result ?? emptyResult()
    }

    // Proj4: no antimeridian preprocessing, just warn and use original geometry.
    // Run preprocessQueryGeometry only for the bbox check — it correctly
    // distinguishes true crossings from explicit-but-non-crossing coords
    // (e.g., rect(200, 210) canonicalizes to [-160, -150], no crossing).
    if (this.proj4def) {
      const { bbox } = preprocessQueryGeometry(geometry)
      if (bbox.crossesAntimeridian) {
        if (!this._antimeridianWarnings.has('proj4-crossing')) {
          this._antimeridianWarnings.add('proj4-crossing')
          console.warn(
            'Antimeridian-crossing polygon queries are not supported for proj4 projections; results may be incorrect'
          )
        }
      }
      return singleFetch(geometry)
    }

    // Standard CRS: preprocess (normalize, canonicalize, maybe clip)
    const { geometry: processedGeometry, bbox: wrappedBbox } =
      preprocessQueryGeometry(geometry)

    // Non-crossing: use processedGeometry (may be canonicalized, e.g. [200,210] → [-160,-150])
    if (!wrappedBbox.crossesAntimeridian) {
      return singleFetch(processedGeometry)
    }

    // Crossing: raster extent guard (EPSG:4326 only — 3857 xyLimits are in meters)
    if (
      rasterExtentCrossesAntimeridian(this.crs ?? 'EPSG:4326', this.xyLimits)
    ) {
      if (!this._antimeridianWarnings.has('raster-extent-crossing')) {
        this._antimeridianWarnings.add('raster-extent-crossing')
        console.warn(
          'Antimeridian-crossing polygon queries are not supported for rasters whose own extent crosses the antimeridian; results may be incorrect'
        )
      }
      return singleFetch(geometry)
    }

    // Crossing: two-strip fetch
    const spans = wrappedBboxToPixelSpans(
      wrappedBbox,
      this.mercatorBounds,
      level.width,
      level.height,
      this.crs ?? 'EPSG:4326',
      this.latIsAscending
    )

    const westResult = spans.west
      ? await runStrip(processedGeometry, spans.west, options)
      : null
    const eastResult = spans.east
      ? await runStrip(processedGeometry, spans.east, options)
      : null

    // If either requested strip failed, return empty rather than partial data
    if ((spans.west && !westResult) || (spans.east && !eastResult)) {
      return emptyResult()
    }
    if (!westResult && !eastResult) return emptyResult()
    if (!westResult || !eastResult) return (westResult ?? eastResult)!

    const { yDim, xDim } = findSpatialDimNames(
      desc.dimensions,
      false,
      desc.dimIndices
    )
    return mergeQueryResults(westResult, eastResult, this.variable, yDim, xDim)
  }
}

/**
 * Merge two QueryResult objects from west and east strips.
 *
 * Ordering: west-strip pixels first, then east-strip pixels. This does NOT
 * preserve row-major scan order. The QueryResult contract provides parallel
 * coordinate arrays so consumers index by position, not implicit grid layout.
 *
 * Spatial coordinate arrays (yDim, xDim) are concatenated.
 * Non-spatial coordinate arrays are taken from the first result unchanged.
 */
function mergeQueryResults(
  a: QueryResult,
  b: QueryResult,
  variable: string,
  yDim: string,
  xDim: string
): QueryResult {
  const spatialKeys = new Set([yDim, xDim])

  // Merge coordinates: concatenate spatial, take first for non-spatial
  const coordinates: Record<string, (number | string)[]> = {}
  for (const key of Object.keys(a.coordinates)) {
    if (spatialKeys.has(key)) {
      coordinates[key] = [...a.coordinates[key], ...b.coordinates[key]]
    } else {
      coordinates[key] = a.coordinates[key]
    }
  }

  // Merge variable values
  const aVals = a[variable] as QueryDataValues
  const bVals = b[variable] as QueryDataValues

  let merged: QueryDataValues
  if (Array.isArray(aVals) && Array.isArray(bVals)) {
    merged = [...aVals, ...bVals]
  } else if (!Array.isArray(aVals) && !Array.isArray(bVals)) {
    merged = mergeNestedValues(aVals as NestedValues, bVals as NestedValues)
  } else {
    merged = aVals // Mismatched types: take first
  }

  return {
    [variable]: merged,
    dimensions: a.dimensions,
    coordinates,
  }
}

/**
 * Recursively merge two NestedValues objects by concatenating leaf arrays.
 */
function mergeNestedValues(a: NestedValues, b: NestedValues): NestedValues {
  const result: NestedValues = {}
  for (const key of Object.keys(a)) {
    const aVal = a[key]
    const bVal = b[key]
    if (Array.isArray(aVal) && Array.isArray(bVal)) {
      result[key] = [...aVal, ...bVal]
    } else if (
      aVal &&
      bVal &&
      !Array.isArray(aVal) &&
      !Array.isArray(bVal) &&
      typeof aVal === 'object' &&
      typeof bVal === 'object'
    ) {
      result[key] = mergeNestedValues(
        aVal as NestedValues,
        bVal as NestedValues
      )
    } else {
      result[key] = aVal
    }
  }
  // Include keys only in b
  for (const key of Object.keys(b)) {
    if (!(key in result)) {
      result[key] = b[key]
    }
  }
  return result
}
