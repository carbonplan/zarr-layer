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
import { WEB_MERCATOR_EXTENT } from './constants'
import type {
  ZarrMode,
  RenderContext,
  TileId,
  RegionRenderState,
} from './zarr-mode'
import type { QueryGeometry, QueryResult } from './query/types'
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
  type MercatorBounds,
  type XYLimits,
} from './map-utils'
import { loadDimensionValues, normalizeSelector, getBands } from './zarr-utils'
import {
  createSubdividedQuad,
  interleaveBands,
  normalizeDataForTexture,
} from './webgl-utils'
import type { ZarrRenderer, ShaderProgram } from './zarr-renderer'
import type { CustomShaderConfig } from './renderer-types'
import { renderMapboxTile } from './mapbox-globe-tile-renderer'
import { queryRegionSingleImage } from './query/region-query'
import {
  mercatorBoundsToPixel,
  computePixelBoundsFromGeometry,
} from './query/query-utils'
import {
  createTransformer,
  createWGS84ToSourceTransformer,
  pixelToSourceCRS,
  sampleEdgesToMercatorBounds,
} from './projection-utils'
import { setObjectValues } from './query/selector-utils'
import { geoToArrayIndex } from './map-utils'
import { resampleToMercator, needsResampling } from './resampler'
import {
  type ThrottleState,
  type RequestCanceller,
  type LoadingManager,
  createThrottleState,
  createRequestCanceller,
  createLoadingManager,
  getThrottleWaitTime,
  scheduleThrottledUpdate,
  markFetchStart,
  clearThrottle,
  cancelOlderRequests,
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
  channels: number
  // WebGL resources
  texture: WebGLTexture | null
  textureUploaded: boolean
  vertexBuffer: WebGLBuffer | null
  pixCoordBuffer: WebGLBuffer | null
  // Geometry arrays for this region's quad
  vertexArr: Float32Array | null
  pixCoordArr: Float32Array | null // Texture coordinates for sampling resampled data
  // Mercator bounds for this region (for shader uniforms)
  mercatorBounds: MercatorBounds | null
  // Version tracking for selector changes
  selectorVersion: number
  // Multi-band support
  bandData: Map<string, Float32Array>
  bandTextures: Map<string, WebGLTexture>
  bandTexturesUploaded: Set<string>
  bandTexturesConfigured: Set<string>
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
}

export class UntiledMode implements ZarrMode {
  isMultiscale: boolean = false

  // Data state (single-level mode)
  private width: number = 0
  private height: number = 0
  private channels: number = 1

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
  private zarrArray: zarr.Array<zarr.DataType> | null = null
  private latIsAscending: boolean | null = null

  // Multi-level support
  private levels: UntiledLevel[] = []
  private currentLevelIndex: number = 0
  private pendingLevelIndex: number | null = null // Guards against concurrent switchToLevel calls
  private proj4def: string | null = null

  // Cached transformers for proj4 reprojection (created once, reused everywhere)
  private cachedMercatorTransformer: ReturnType<
    typeof createTransformer
  > | null = null
  private cachedWGS84Transformer: ReturnType<
    typeof createWGS84ToSourceTransformer
  > | null = null

  // Global mercator grid for seamless chunk stitching
  // All chunks resample to this shared grid to avoid seams at boundaries
  private globalMercatorGrid: {
    x0: number // Left edge in normalized mercator [0,1]
    y0: number // Top edge in normalized mercator [0,1]
    cellW: number // Cell width in normalized mercator
    cellH: number // Cell height in normalized mercator
    gridW: number // Total grid width in pixels
    gridH: number // Total grid height in pixels
  } | null = null

  // Loading state
  private isRemoved: boolean = false
  private throttleMs: number

  // Shared state managers
  private throttleState: ThrottleState = createThrottleState()
  private requestCanceller: RequestCanceller = createRequestCanceller()
  private loadingManager: LoadingManager = createLoadingManager()

  // Dimension values cache
  private dimensionValues: { [key: string]: Float64Array | number[] } = {}

  // Data processing
  private clim: [number, number] = [0, 1]

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
  private regionSize: [number, number] | null = null // [height, width] of each region
  private lastViewportHash: string = ''
  private baseSliceArgs: (number | zarr.Slice)[] = [] // Cached slice args for non-spatial dims
  private selectorVersion: number = 0 // Incremented on selector change to track stale regions
  // Multi-band support: track which dimensions have multiple selected values
  private baseMultiValueDims: Array<{
    dimIndex: number
    dimName: string
    values: number[]
    labels: (number | string)[]
  }> = []

  // Cached WebGL context for use in setSelector
  private cachedGl: WebGL2RenderingContext | null = null
  // Track if base slice args have been built (ready for region fetching)
  private baseSliceArgsReady: boolean = false

  constructor(
    store: ZarrStore,
    variable: string,
    selector: NormalizedSelector,
    invalidate: () => void,
    throttleMs: number = 100
  ) {
    this.zarrStore = store
    this.variable = variable
    this.selector = selector
    this.bandNames = getBands(variable, selector)
    this.invalidate = invalidate
    this.throttleMs = throttleMs
  }

  async initialize(): Promise<void> {
    this.loadingManager.metadataLoading = true
    this.emitLoadingState()

    try {
      const desc = this.zarrStore.describe()
      this.dimIndices = desc.dimIndices
      this.crs = desc.crs
      this.xyLimits = desc.xyLimits
      this.latIsAscending = desc.latIsAscending ?? null
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
        await this.loadLevelMetadata()
        // Don't load data yet - defer to update() where we have the actual zoom level
        // This avoids loading low-res then immediately switching to high-res
        this.currentLevelIndex = -1 // Mark as not yet selected
      } else {
        // Single-level dataset - load immediately
        this.isMultiscale = false
        this.zarrArray = await this.zarrStore.getArray()
        this.width = this.zarrArray.shape[this.dimIndices.lon.index]
        this.height = this.zarrArray.shape[this.dimIndices.lat.index]
      }

      if (this.xyLimits) {
        // For proj4, compute mercator bounds by transforming corners
        if (this.proj4def) {
          this.mercatorBounds = this.computeMercatorBoundsFromProjection()
        } else {
          this.mercatorBounds = boundsToMercatorNorm(this.xyLimits, this.crs)
        }
        // Compute global grid for seamless chunk stitching
        this.computeGlobalMercatorGrid()
      } else {
        console.warn('UntiledMode: No XY limits found')
      }
    } finally {
      this.loadingManager.metadataLoading = false
      this.emitLoadingState()
    }
  }

  private async loadLevelMetadata(): Promise<void> {
    // Fetch metadata for levels that need shape OR per-level scale/offset
    // (consolidated metadata has shape but not always scale_factor/add_offset)
    const levelsNeedingFetch = this.levels.filter(
      (level) => !level.shape || level.scaleFactor === undefined
    )

    if (levelsNeedingFetch.length === 0) {
      return
    }

    await Promise.all(
      levelsNeedingFetch.map(async (level) => {
        try {
          const meta = await this.zarrStore.getUntiledLevelMetadata(level.asset)
          level.shape = meta.shape
          level.chunks = meta.chunks
          level.scaleFactor = meta.scaleFactor
          level.addOffset = meta.addOffset
          level.fillValue = meta.fillValue
          level.dtype = meta.dtype
        } catch (err) {
          console.warn(`Failed to load metadata for level ${level.asset}:`, err)
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
    if (!bounds || !this.xyLimits || !this.regionSize) return []

    const [[west, south], [east, north]] = bounds
    const { xMin, xMax, yMin, yMax } = this.xyLimits
    const [regionH, regionW] = this.regionSize

    if (this.proj4def && this.cachedWGS84Transformer) {
      // For projected data, check each region's geographic footprint against viewport
      const transformer = this.cachedWGS84Transformer
      const numRegionsX = Math.ceil(this.width / regionW)
      const numRegionsY = Math.ceil(this.height / regionH)

      const regions: Array<{ regionX: number; regionY: number }> = []

      for (let ry = 0; ry < numRegionsY; ry++) {
        for (let rx = 0; rx < numRegionsX; rx++) {
          // Get region bounds in source CRS
          const regBounds = this.getRegionBounds(rx, ry)
          const xMid = (regBounds.xMin + regBounds.xMax) / 2
          const yMid = (regBounds.yMin + regBounds.yMax) / 2

          // Transform region corners and edge midpoints to WGS84
          // Edge midpoints are needed for curved projections where extrema may not be at corners
          const samplePoints = [
            // Corners
            transformer.inverse(regBounds.xMin, regBounds.yMin),
            transformer.inverse(regBounds.xMax, regBounds.yMin),
            transformer.inverse(regBounds.xMax, regBounds.yMax),
            transformer.inverse(regBounds.xMin, regBounds.yMax),
            // Edge midpoints
            transformer.inverse(xMid, regBounds.yMin),
            transformer.inverse(xMid, regBounds.yMax),
            transformer.inverse(regBounds.xMin, yMid),
            transformer.inverse(regBounds.xMax, yMid),
          ]

          // Filter out invalid points but continue if we have at least one valid point
          const validPoints = samplePoints.filter(
            (c) => isFinite(c[0]) && isFinite(c[1])
          )
          if (validPoints.length === 0) {
            continue
          }

          // Get geographic bounds of this region
          const regWest = Math.min(...validPoints.map((c) => c[0]))
          const regEast = Math.max(...validPoints.map((c) => c[0]))
          const regSouth = Math.min(...validPoints.map((c) => c[1]))
          const regNorth = Math.max(...validPoints.map((c) => c[1]))

          // Check if region overlaps with viewport
          if (
            regEast >= west &&
            regWest <= east &&
            regNorth >= south &&
            regSouth <= north
          ) {
            regions.push({ regionX: rx, regionY: ry })
          }
        }
      }

      return regions
    }

    // Standard case: viewport bounds are in same CRS as xyLimits
    const xMinIdx = geoToArrayIndex(west, xMin, xMax, this.width)
    const xMaxIdx = geoToArrayIndex(east, xMin, xMax, this.width)

    // For Y axis, geoToArrayIndex assumes yMin maps to row 0.
    // But if latIsAscending=false (row 0 = north = yMax), we need to invert.
    let ySouthIdx = geoToArrayIndex(south, yMin, yMax, this.height)
    let yNorthIdx = geoToArrayIndex(north, yMin, yMax, this.height)

    // Only invert if we explicitly know latIsAscending is false
    // If null/undefined, assume ascending (yMin at row 0) as default
    if (this.latIsAscending === false) {
      // Invert Y indices: row 0 = north (yMax), row height-1 = south (yMin)
      ySouthIdx = this.height - 1 - ySouthIdx
      yNorthIdx = this.height - 1 - yNorthIdx
    }

    // Convert pixel indices to region indices
    const regionXMin = Math.floor(Math.min(xMinIdx, xMaxIdx) / regionW)
    const regionXMax = Math.floor(Math.max(xMinIdx, xMaxIdx) / regionW)
    const regionYMin = Math.floor(Math.min(ySouthIdx, yNorthIdx) / regionH)
    const regionYMax = Math.floor(Math.max(ySouthIdx, yNorthIdx) / regionH)

    // Clamp to valid range
    const numRegionsX = Math.ceil(this.width / regionW)
    const numRegionsY = Math.ceil(this.height / regionH)
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
      channels: 1,
      texture: null,
      textureUploaded: false,
      vertexBuffer: null,
      pixCoordBuffer: null,
      vertexArr: null,
      pixCoordArr: null,
      mercatorBounds: null,
      selectorVersion: this.selectorVersion,
      bandData: new Map(),
      bandTextures: new Map(),
      bandTexturesUploaded: new Set(),
      bandTexturesConfigured: new Set(),
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
      region.mercatorBounds
    )
  }

  /**
   * Clear loading flags for a batch of regions (used when aborting stale fetches).
   */
  private clearBatchLoadingFlags(
    regions: Array<{ regionX: number; regionY: number }>,
    levelIndex: number
  ): void {
    for (const { regionX, regionY } of regions) {
      const key = this.makeRegionKey(levelIndex, regionX, regionY)
      const region = this.regionCache.get(key)
      if (region) region.loading = false
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
    if (this.lastVisibleRegionsLevel !== this.currentLevelIndex) {
      return false
    }
    const levelIndex = this.currentLevelIndex
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
      if (region.levelIndex === this.currentLevelIndex) continue
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
    const currentLevel = this.currentLevelIndex
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
   * Get geographic bounds for a region.
   * Accounts for data orientation (latIsAscending).
   */
  private getRegionBounds(
    regionX: number,
    regionY: number
  ): { xMin: number; xMax: number; yMin: number; yMax: number } {
    if (!this.xyLimits || !this.regionSize) {
      return { xMin: 0, xMax: 1, yMin: 0, yMax: 1 }
    }

    const [regionH, regionW] = this.regionSize
    const { xMin, xMax, yMin, yMax } = this.xyLimits

    // Calculate pixel bounds for this region
    const pxXStart = regionX * regionW
    const pxXEnd = Math.min(pxXStart + regionW, this.width)
    const pxYStart = regionY * regionH
    const pxYEnd = Math.min(pxYStart + regionH, this.height)

    // Convert pixel bounds to geographic bounds (X is always left-to-right)
    const geoXMin = xMin + (pxXStart / this.width) * (xMax - xMin)
    const geoXMax = xMin + (pxXEnd / this.width) * (xMax - xMin)

    // Y mapping depends on data orientation
    // Default (null/undefined) assumes ascending (row 0 = south = yMin)
    let geoYMin: number
    let geoYMax: number
    if (this.latIsAscending === false) {
      // Data has lat decreasing with array index: pixel 0 = north (yMax)
      geoYMax = yMax - (pxYStart / this.height) * (yMax - yMin)
      geoYMin = yMax - (pxYEnd / this.height) * (yMax - yMin)
    } else {
      // Data has lat increasing with array index: pixel 0 = south (yMin)
      // This is also the default when latIsAscending is null/unknown
      geoYMin = yMin + (pxYStart / this.height) * (yMax - yMin)
      geoYMax = yMin + (pxYEnd / this.height) * (yMax - yMin)
    }

    return { xMin: geoXMin, xMax: geoXMax, yMin: geoYMin, yMax: geoYMax }
  }

  /**
   * Create geometry (vertex positions and tex coords) for a region.
   * Uses subdivided geometry for smooth globe rendering.
   * Data is resampled to Mercator on CPU, so linear texture coords are used.
   */
  private createRegionGeometry(
    regionX: number,
    regionY: number,
    gl: WebGL2RenderingContext,
    region: RegionState
  ): void {
    const geoBounds = this.getRegionBounds(regionX, regionY)

    // Use cached mercatorBounds if set (from fetchRegion's resampling path),
    // otherwise compute from geoBounds (for non-resampling cases like EPSG:3857)
    const mercBounds =
      region.mercatorBounds ?? boundsToMercatorNorm(geoBounds, this.crs)
    region.mercatorBounds = mercBounds

    // Subdivisions for smooth globe tessellation - more for larger regions
    const latSpan = Math.abs(geoBounds.yMax - geoBounds.yMin)
    const subdivisions = Math.max(16, Math.min(128, Math.ceil(latSpan)))

    if (this.proj4def && this.cachedMercatorTransformer) {
      const warped = this.createWarpedGridGeometry(
        geoBounds,
        mercBounds,
        subdivisions
      )
      region.vertexArr = warped.vertexArr
      region.pixCoordArr = warped.texCoordArr
    } else {
      const subdivided = createSubdividedQuad(subdivisions)
      region.vertexArr = subdivided.vertexArr

      // Determine texture coordinates based on CRS and data orientation
      // EPSG:4326 is resampled to Mercator on CPU, so use linear coords
      // Other CRS (3857, etc.) need V-flip if latitude is ascending
      if (this.crs === 'EPSG:4326') {
        region.pixCoordArr = subdivided.texCoordArr
      } else {
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
   * Create warped mesh geometry for proj4 regions.
   * Builds a shared grid in source CRS so adjacent regions share edge vertices.
   */
  private createWarpedGridGeometry(
    geoBounds: { xMin: number; xMax: number; yMin: number; yMax: number },
    mercBounds: MercatorBounds,
    subdivisions: number
  ): { vertexArr: Float32Array; texCoordArr: Float32Array } {
    const transformer = this.cachedMercatorTransformer!
    const { xMin, xMax, yMin, yMax } = geoBounds

    const denomX = mercBounds.x1 - mercBounds.x0 || 1
    const denomY = mercBounds.y1 - mercBounds.y0 || 1
    const invExtent = 1 / (2 * WEB_MERCATOR_EXTENT)

    const vertices: number[] = []
    const texCoords: number[] = []

    const pushVertex = (u: number, v: number) => {
      const srcX = xMin + u * (xMax - xMin)
      const srcY = yMin + v * (yMax - yMin)
      const [mercX, mercY] = transformer.forward(srcX, srcY)

      const normX = (mercX + WEB_MERCATOR_EXTENT) * invExtent
      const normY = (WEB_MERCATOR_EXTENT - mercY) * invExtent

      const texU = (normX - mercBounds.x0) / denomX
      const texV = (normY - mercBounds.y0) / denomY

      const clipX = -1 + 2 * texU
      const clipY = 1 - 2 * texV

      vertices.push(clipX, clipY)
      texCoords.push(texU, texV)
    }

    for (let row = 0; row < subdivisions; row++) {
      const v0 = row / subdivisions
      const v1 = (row + 1) / subdivisions
      for (let col = 0; col <= subdivisions; col++) {
        const u = col / subdivisions
        pushVertex(u, v0)
        pushVertex(u, v1)
      }
      if (row < subdivisions - 1) {
        pushVertex(1, v1)
        pushVertex(0, v1)
      }
    }

    return {
      vertexArr: new Float32Array(vertices),
      texCoordArr: new Float32Array(texCoords),
    }
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
      /** Spatial bounds for fetch - point for single pixel, bbox for region subset */
      spatialBounds?:
        | { type: 'point'; x: number; y: number }
        | {
            type: 'bbox'
            minX: number
            maxX: number
            minY: number
            maxY: number
          }
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
    if (!this.zarrArray) {
      return { sliceArgs: [], multiValueDims: [] }
    }

    const sliceArgs: (number | zarr.Slice)[] = new Array(
      this.zarrArray.shape.length
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
        if (options.spatialBounds?.type === 'point') {
          sliceArgs[dimInfo.index] = options.spatialBounds.x
        } else if (options.spatialBounds?.type === 'bbox') {
          sliceArgs[dimInfo.index] = zarr.slice(
            options.spatialBounds.minX,
            options.spatialBounds.maxX
          )
        } else {
          sliceArgs[dimInfo.index] = options.includeSpatialSlices
            ? zarr.slice(0, this.width)
            : 0
        }
      } else if (dimType === 'lat') {
        if (options.spatialBounds?.type === 'point') {
          sliceArgs[dimInfo.index] = options.spatialBounds.y
        } else if (options.spatialBounds?.type === 'bbox') {
          sliceArgs[dimInfo.index] = zarr.slice(
            options.spatialBounds.minY,
            options.spatialBounds.maxY
          )
        } else {
          sliceArgs[dimInfo.index] = options.includeSpatialSlices
            ? zarr.slice(0, this.height)
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
    this.lastVisibleRegionsLevel = this.currentLevelIndex
    const levelIndex = this.currentLevelIndex

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

    // Separate regions into two categories:
    // 1. New regions (no data) - viewport change, fetch immediately
    // 2. Stale regions (have data, wrong selector) - selector change, throttle
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

    // Fetch new regions immediately (viewport changes - no throttle)
    if (newRegions.length > 0) {
      this.fetchRegions(newRegions, gl)
    }

    // Fetch stale regions with throttle (selector changes)
    if (staleRegions.length > 0) {
      this.fetchRegionsThrottled(staleRegions, gl)
    }
  }

  /**
   * Fetch regions with throttling (for selector changes).
   */
  private fetchRegionsThrottled(
    regions: Array<{ regionX: number; regionY: number }>,
    gl: WebGL2RenderingContext
  ): void {
    const waitTime = getThrottleWaitTime(this.throttleState, this.throttleMs)
    if (waitTime > 0) {
      // Set loading state even when throttled so callers know data is pending
      if (!this.throttleState.throttledPending) {
        this.throttleState.throttledPending = true
        this.emitLoadingState()
      }
      scheduleThrottledUpdate(this.throttleState, waitTime, this.invalidate)
      return
    }
    markFetchStart(this.throttleState)

    // Actually fetch the regions
    this.fetchRegions(regions, gl)
  }

  /**
   * Fetch multiple regions with limited concurrency to avoid overwhelming the browser.
   */
  private async fetchRegions(
    regions: Array<{ regionX: number; regionY: number }>,
    gl: WebGL2RenderingContext
  ): Promise<void> {
    // Can't fetch without required state
    if (!this.zarrArray || !this.regionSize) {
      return
    }

    // Capture ALL level-dependent state at start to pass to fetchRegion
    // This prevents races where switchToLevel or selector changes update state mid-batch
    const snapshot: LevelSnapshot = {
      index: this.currentLevelIndex,
      zarrArray: this.zarrArray,
      baseSliceArgs: [...this.baseSliceArgs],
      width: this.width,
      height: this.height,
      regionSize: this.regionSize,
      selectorVersion: this.selectorVersion,
      baseMultiValueDims: this.baseMultiValueDims.map((dim) => ({
        dimIndex: dim.dimIndex,
        dimName: dim.dimName,
        values: [...dim.values],
        labels: [...dim.labels],
      })),
    }

    // Emit loading state
    this.loadingManager.chunksLoading = true
    this.emitLoadingState()

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

    const MAX_CONCURRENT = 32
    const executing: Promise<void>[] = []

    for (const { regionX, regionY } of regions) {
      // Check if level switched - bail out to avoid stale fetches
      if (this.currentLevelIndex !== snapshot.index) {
        cancelAllRequests(this.requestCanceller)
        this.clearBatchLoadingFlags(regions, snapshot.index)
        break
      }

      const promise = this.fetchRegion(regionX, regionY, gl, snapshot)
        .then(() => {
          executing.splice(executing.indexOf(promise), 1)
        })
        .catch(() => {
          executing.splice(executing.indexOf(promise), 1)
        })

      executing.push(promise)

      if (executing.length >= MAX_CONCURRENT) {
        await Promise.race(executing)
      }
    }

    // Wait for remaining requests (if any are still in flight)
    if (executing.length > 0) {
      await Promise.allSettled(executing)
    }

    // Only update loading state if we're still on the same level
    if (!hasActiveRequests(this.requestCanceller)) {
      this.loadingManager.chunksLoading = false
      this.emitLoadingState()

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
    if (this.currentLevelIndex !== snapshot.index) {
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
      const pixelCount = actualW * actualH

      // Fetch data for all channels
      const bandArrays: Float32Array[] = []
      const packedData = new Float32Array(pixelCount * numChannels)
      packedData.fill(fillValue ?? 0)

      if (numChannels === 1) {
        // Single channel - simple fetch
        // Check if already aborted or level changed before starting fetch
        if (
          controller.signal.aborted ||
          this.currentLevelIndex !== snapshot.index
        ) {
          region.loading = false
          this.requestCanceller.controllers.delete(requestId)
          return
        }

        const result = (await zarr.get(snapshot.zarrArray, baseSliceArgs, {
          opts: { signal: controller.signal },
        })) as { data: ArrayLike<number> }

        if (controller.signal.aborted || this.isRemoved) {
          region.loading = false
          return
        }

        if (this.currentLevelIndex !== snapshot.index) {
          region.loading = false
          this.requestCanceller.controllers.delete(requestId)
          return
        }

        const rawData = new Float32Array(result.data as ArrayLike<number>)
        bandArrays.push(rawData)
        packedData.set(rawData)
      } else {
        // Multi-channel - fetch each channel's data
        for (let c = 0; c < numChannels; c++) {
          // Check if already aborted or level changed before starting fetch
          if (
            controller.signal.aborted ||
            this.currentLevelIndex !== snapshot.index
          ) {
            region.loading = false
            this.requestCanceller.controllers.delete(requestId)
            return
          }

          const sliceArgs = [...baseSliceArgs]
          const combo = channelCombinations[c]

          // Apply channel-specific indices to multi-value dimensions
          for (let i = 0; i < snapshot.baseMultiValueDims.length; i++) {
            sliceArgs[snapshot.baseMultiValueDims[i].dimIndex] = combo[i]
          }

          const result = (await zarr.get(snapshot.zarrArray, sliceArgs, {
            opts: { signal: controller.signal },
          })) as { data: ArrayLike<number> }

          if (controller.signal.aborted || this.isRemoved) {
            region.loading = false
            return
          }

          if (this.currentLevelIndex !== snapshot.index) {
            region.loading = false
            this.requestCanceller.controllers.delete(requestId)
            return
          }

          const bandData = new Float32Array(result.data as ArrayLike<number>)
          bandArrays.push(bandData)

          // Pack into interleaved format for main texture
          for (let pixIdx = 0; pixIdx < pixelCount; pixIdx++) {
            packedData[pixIdx * numChannels + c] = bandData[pixIdx]
          }
        }
      }

      // Only render if this is newer than what's already rendered for this region
      if (fetchSelectorVersion < region.selectorVersion) {
        region.loading = false
        return
      }

      // Update region's selector version and cancel any older pending requests
      region.selectorVersion = fetchSelectorVersion
      cancelOlderRequests(this.requestCanceller, requestId)

      // Resample bands to Mercator space if needed (EPSG:4326 or custom projection)
      let bandDataToProcess = bandArrays
      let outputW = actualW
      let outputH = actualH

      if (needsResampling(this.crs, this.proj4def) && this.xyLimits) {
        const geoBounds = this.getRegionBounds(regionX, regionY)

        // Reuse cached mercatorBounds to avoid redundant computation on re-fetches
        let mercBounds: MercatorBounds
        if (region.mercatorBounds) {
          mercBounds = region.mercatorBounds
        } else if (this.proj4def) {
          const gridAligned = this.getGridAlignedRegionBounds(regionX, regionY)
          if (gridAligned) {
            mercBounds = gridAligned.mercBounds
            outputW = gridAligned.outputW
            outputH = gridAligned.outputH
          } else {
            mercBounds = this.computeRegionMercatorBounds(geoBounds)
          }
        } else {
          mercBounds = boundsToMercatorNorm(geoBounds, this.crs)
        }

        region.mercatorBounds = mercBounds

        const resampleOpts = {
          sourceSize: [actualW, actualH] as [number, number],
          sourceBounds: [
            geoBounds.xMin,
            geoBounds.yMin,
            geoBounds.xMax,
            geoBounds.yMax,
          ] as [number, number, number, number],
          targetSize: [outputW, outputH] as [number, number],
          targetMercatorBounds: [
            mercBounds.x0,
            mercBounds.y0,
            mercBounds.x1,
            mercBounds.y1,
          ] as [number, number, number, number],
          fillValue: fillValue ?? 0,
          latIsAscending: this.latIsAscending,
          proj4: this.proj4def,
        }

        // Resample each band once (no duplicate resampling)
        bandDataToProcess = bandArrays.map((bandData) =>
          resampleToMercator({ sourceData: bandData, ...resampleOpts })
        )
      }

      // Apply per-level scale/offset to convert raw values to physical units
      // Fall back to dataset-level scale/offset for pyramids that only define them at the root
      const scaleFactor = currentLevel?.scaleFactor ?? desc.scaleFactor
      const addOffset = currentLevel?.addOffset ?? desc.addOffset

      // Normalize bands (single pass) and collect for interleaving
      region.bandData.clear()
      region.bandTexturesUploaded.clear()
      const normalizedBands: Float32Array[] = []

      for (let c = 0; c < bandDataToProcess.length; c++) {
        const bandName = this.bandNames[c] || `band_${c}`
        let bandData = bandDataToProcess[c]

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
          this.clim
        )
        region.bandData.set(bandName, bandNormalized)
        normalizedBands.push(bandNormalized)
      }

      // Construct interleaved data from normalized bands
      region.data = interleaveBands(normalizedBands, numChannels)
      region.width = outputW
      region.height = outputH
      region.channels = numChannels
      region.loading = false

      // Create/update main texture for this region
      if (!region.texture) {
        region.texture = gl.createTexture()
      }

      // Upload texture using shared helper
      const result = uploadDataTexture(gl, {
        texture: region.texture!,
        data: region.data!,
        width: outputW,
        height: outputH,
        channels: numChannels,
        configured: false,
      })
      region.textureUploaded = result.uploaded

      // Create geometry for this region
      this.createRegionGeometry(regionX, regionY, gl, region)

      this.invalidate()
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        console.error(`[fetchRegion] Error fetching region ${key}:`, err)
      }
      region.loading = false
    } finally {
      this.requestCanceller.controllers.delete(requestId)
    }
  }

  update(map: MapLike, gl: WebGL2RenderingContext): void {
    // Cache gl context for use in setSelector
    this.cachedGl = gl

    // Don't proceed if metadata is still loading
    if (this.loadingManager.metadataLoading) {
      return
    }

    // For multi-level datasets, select/switch levels based on zoom
    if (this.isMultiscale && this.levels.length > 0) {
      const mapZoom = map.getZoom?.() ?? 0
      const bestLevelIndex = this.selectLevelForZoom(mapZoom)

      // Initial load or level switch needed
      if (this.currentLevelIndex === -1) {
        // First time - load the appropriate level for current zoom
        this.initializeLevel(bestLevelIndex)
        return
      } else if (
        bestLevelIndex !== this.currentLevelIndex &&
        this.pendingLevelIndex === null
      ) {
        // Zoom changed enough to warrant level switch (and no switch in progress)
        this.switchToLevel(bestLevelIndex)
        return
      }

      // Update visible regions on viewport change (only if ready)
      if (this.regionSize && this.baseSliceArgsReady) {
        this.updateVisibleRegions(map, gl)
      }
    } else {
      // Single-level dataset - set up region-based loading if not already done
      if (
        !this.regionSize &&
        this.zarrArray &&
        !this.loadingManager.chunksLoading
      ) {
        const detectedRegionSize = this.getRegionSize(this.zarrArray)
        this.regionSize = detectedRegionSize ?? [this.height, this.width]

        // Build base slice args and let updateVisibleRegions handle loading
        this.buildBaseSliceArgs().then(() => {
          this.updateVisibleRegions(map, gl)
        })
        return
      }

      // Update visible regions for single-level dataset (only if ready)
      if (this.regionSize && this.baseSliceArgsReady) {
        this.updateVisibleRegions(map, gl)
      }
    }
  }

  private async initializeLevel(levelIndex: number): Promise<void> {
    if (levelIndex < 0 || levelIndex >= this.levels.length) {
      return
    }
    if (this.loadingManager.chunksLoading) {
      return
    }

    const level = this.levels[levelIndex]
    this.currentLevelIndex = levelIndex

    try {
      this.zarrArray = await this.zarrStore.getLevelArray(level.asset)
      this.width = this.zarrArray.shape[this.dimIndices.lon.index]
      this.height = this.zarrArray.shape[this.dimIndices.lat.index]

      // Always use region-based loading for unified rendering path
      // If no chunk/shard boundaries, treat whole level as one region
      const detectedRegionSize = this.getRegionSize(this.zarrArray)
      this.regionSize = detectedRegionSize ?? [this.height, this.width]
      this.regionCache.clear()

      // Recompute global grid for new level dimensions
      this.computeGlobalMercatorGrid()

      // Build base slice args for non-spatial dimensions
      await this.buildBaseSliceArgs()

      // Let update() trigger viewport-aware loading
      this.invalidate()
    } catch (err) {
      console.error(`Failed to initialize level ${level.asset}:`, err)
    }
  }

  /**
   * Build base slice args for non-spatial dimensions.
   * This caches the selector values for use in region fetching.
   * Also tracks multi-value dimensions for band extraction.
   */
  private async buildBaseSliceArgs(): Promise<void> {
    if (!this.zarrArray) return

    this.baseSliceArgsReady = false

    const { sliceArgs, multiValueDims } = await this.buildSliceArgsForSelector(
      this.selector,
      {
        includeSpatialSlices: false, // placeholders for region fetching
        trackMultiValue: true, // track multi-value dims for band extraction
      }
    )

    this.baseSliceArgs = sliceArgs
    this.baseMultiValueDims = multiValueDims
    this.baseSliceArgsReady = true
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
    // (at least 50% of map's pixel density to avoid obvious pixelation)
    const minRequired = mapPixelsPerWorld * 0.5
    for (const { index, effectivePixels } of levelResolutions) {
      if (effectivePixels >= minRequired) {
        return index
      }
    }

    // If no level is sufficient, use the highest resolution available
    return levelResolutions[levelResolutions.length - 1].index
  }

  private async switchToLevel(newLevelIndex: number): Promise<void> {
    if (newLevelIndex === this.currentLevelIndex) return
    if (newLevelIndex < 0 || newLevelIndex >= this.levels.length) return

    // Mark as pending to prevent concurrent calls for same level
    this.pendingLevelIndex = newLevelIndex

    // Cancel in-flight requests for the old level - data not reusable across resolutions
    const controllersToCancel = this.requestCanceller.controllers.size
    if (controllersToCancel > 0) {
      for (const controller of this.requestCanceller.controllers.values()) {
        controller.abort()
      }
      this.requestCanceller.controllers.clear()
      this.loadingManager.chunksLoading = false
      this.emitLoadingState()
    }

    const level = this.levels[newLevelIndex]

    try {
      const newArray = await this.zarrStore.getLevelArray(level.asset)
      const newWidth = newArray.shape[this.dimIndices.lon.index]
      const newHeight = newArray.shape[this.dimIndices.lat.index]

      // Always use region-based loading for unified rendering path
      // If no chunk/shard boundaries, treat whole level as one region
      const detectedRegionSize = this.getRegionSize(newArray)
      const newRegionSize: [number, number] = detectedRegionSize ?? [
        newHeight,
        newWidth,
      ]

      // Update all level state atomically AFTER async work completes
      // This prevents race conditions where render sees new level index but old dimensions
      this.currentLevelIndex = newLevelIndex
      this.pendingLevelIndex = null
      this.zarrArray = newArray
      this.width = newWidth
      this.height = newHeight
      this.regionSize = newRegionSize
      this.computeGlobalMercatorGrid()
      this.resetVisibleRegions()

      // Build base slice args for non-spatial dimensions
      await this.buildBaseSliceArgs()

      // Let update() trigger viewport-aware loading
      this.invalidate()
    } catch (err) {
      this.pendingLevelIndex = null
      console.error(`Failed to switch to level ${level.asset}:`, err)
    }
  }

  render(renderer: ZarrRenderer, context: RenderContext): void {
    const useMapboxGlobe = !!context.mapboxGlobe
    const shaderProgram = renderer.getProgram(
      context.shaderData,
      context.customShaderConfig,
      useMapboxGlobe
    )

    renderer.gl.useProgram(shaderProgram.program)

    renderer.applyCommonUniforms(
      shaderProgram,
      context.colormapTexture,
      this.getUniformsForRender(context.uniforms),
      context.customShaderConfig,
      context.projectionData,
      context.mapboxGlobe,
      context.matrix,
      false
    )

    // Always use region-based rendering (unified path)
    this.renderRegions(
      renderer,
      shaderProgram,
      context.worldOffsets,
      context.customShaderConfig
    )
  }

  /**
   * Convert a RegionState to a RenderableRegion for unified rendering.
   */
  private regionToRenderable(region: RegionState): RenderableRegion {
    return {
      mercatorBounds: region.mercatorBounds!,
      vertexBuffer: region.vertexBuffer!,
      pixCoordBuffer: region.pixCoordBuffer!,
      vertexCount: region.vertexArr!.length / 2,
      texture: region.texture!,
      bandData: region.bandData,
      bandTextures: region.bandTextures,
      bandTexturesUploaded: region.bandTexturesUploaded,
      bandTexturesConfigured: region.bandTexturesConfigured,
      width: region.width,
      height: region.height,
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
    customShaderConfig?: CustomShaderConfig
  ): void {
    const gl = renderer.gl

    // Set up band texture uniforms once per frame
    setupBandTextureUniforms(gl, shaderProgram, customShaderConfig)

    // Render each loaded region using unified path
    for (const region of this.getLoadedRegions()) {
      renderRegion(
        gl,
        shaderProgram,
        this.regionToRenderable(region),
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

  onProjectionChange(_isGlobe: boolean): void {
    // No-op: regions handle their own geometry
  }

  getTiledState() {
    return null
  }

  /**
   * Get render states for all loaded regions (for multi-region rendering).
   * Includes previous level regions as fallback during level transitions.
   */
  private getRegionStates(): RegionRenderState[] {
    if (!this.regionSize) {
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
      latIsAscending: this.latIsAscending ?? undefined,
      bandData: region.bandData,
      bandTextures: region.bandTextures,
      bandTexturesUploaded: region.bandTexturesUploaded,
      bandTexturesConfigured: region.bandTexturesConfigured,
    }))
  }

  dispose(gl: WebGL2RenderingContext | WebGLRenderingContext): void {
    this.isRemoved = true
    clearThrottle(this.throttleState)
    cancelAllRequests(this.requestCanceller)
    // Clean up region caches
    this.clearRegionCache(gl)
    this.regionSize = null
    this.cachedMercatorTransformer = null
    this.cachedWGS84Transformer = null
    this.loadingManager.chunksLoading = false
    this.emitLoadingState()
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

  /**
   * Compute global mercator grid based on full data extent.
   * This grid ensures all chunks resample to aligned pixel positions,
   * eliminating seams at chunk boundaries.
   */
  private computeGlobalMercatorGrid(): void {
    if (!this.proj4def || !this.mercatorBounds || !this.width || !this.height) {
      this.globalMercatorGrid = null
      return
    }

    const { x0, y0, x1, y1 } = this.mercatorBounds

    // Compute cell size based on source resolution
    // The grid should have roughly the same number of cells as source pixels
    const mercWidth = x1 - x0
    const mercHeight = y1 - y0

    this.globalMercatorGrid = {
      x0,
      y0,
      cellW: mercWidth / this.width,
      cellH: mercHeight / this.height,
      gridW: this.width,
      gridH: this.height,
    }
  }

  /**
   * Get grid-aligned mercator bounds and output size for a region.
   * Returns bounds that snap to the global grid, ensuring seamless stitching.
   */
  private getGridAlignedRegionBounds(
    regionX: number,
    regionY: number
  ): { mercBounds: MercatorBounds; outputW: number; outputH: number } | null {
    if (!this.globalMercatorGrid || !this.regionSize || !this.proj4def) {
      return null
    }

    const grid = this.globalMercatorGrid

    // Get the region's actual mercator footprint by transforming its corners
    const geoBounds = this.getRegionBounds(regionX, regionY)
    const rawMercBounds = this.computeRegionMercatorBounds(geoBounds)

    // Snap to global grid boundaries
    // Find which grid cells this region covers
    const cellStartX = Math.floor((rawMercBounds.x0 - grid.x0) / grid.cellW)
    const cellEndX = Math.ceil((rawMercBounds.x1 - grid.x0) / grid.cellW)
    const cellStartY = Math.floor((rawMercBounds.y0 - grid.y0) / grid.cellH)
    const cellEndY = Math.ceil((rawMercBounds.y1 - grid.y0) / grid.cellH)

    // Clamp to grid bounds
    const clampedStartX = Math.max(0, cellStartX)
    const clampedEndX = Math.min(grid.gridW, cellEndX)
    const clampedStartY = Math.max(0, cellStartY)
    const clampedEndY = Math.min(grid.gridH, cellEndY)

    const outputW = clampedEndX - clampedStartX
    const outputH = clampedEndY - clampedStartY

    if (outputW <= 0 || outputH <= 0) {
      return null
    }

    // Compute snapped mercator bounds
    const mercBounds: MercatorBounds = {
      x0: grid.x0 + clampedStartX * grid.cellW,
      y0: grid.y0 + clampedStartY * grid.cellH,
      x1: grid.x0 + clampedEndX * grid.cellW,
      y1: grid.y0 + clampedEndY * grid.cellH,
    }

    return { mercBounds, outputW, outputH }
  }

  getMaxLevelIndex(): number {
    return this.levels.length > 0 ? this.levels.length - 1 : 0
  }

  getLevels(): string[] {
    return this.levels.map((l) => l.asset)
  }

  updateClim(clim: [number, number]): void {
    this.clim = clim
  }

  async setSelector(selector: NormalizedSelector): Promise<void> {
    this.selector = selector
    this.bandNames = getBands(this.variable, selector)

    const gl = this.cachedGl
    if (!gl) {
      // No gl context yet - selector is stored, update() will handle loading
      this.invalidate()
      return
    }

    // Initialize region size if needed
    if (!this.regionSize && this.zarrArray) {
      this.regionSize = this.getRegionSize(this.zarrArray) ?? [
        this.height,
        this.width,
      ]
    }

    this.selectorVersion++
    await this.buildBaseSliceArgs()
    this.lastViewportHash = ''
    this.invalidate()
  }

  private emitLoadingState(): void {
    // Update chunksLoading to include throttle state
    if (
      this.throttleState.throttledPending &&
      !this.loadingManager.chunksLoading
    ) {
      this.loadingManager.chunksLoading = true
    }
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
    selector: NormalizedSelector,
    spatialQuery:
      | { type: 'point'; x: number; y: number }
      | { type: 'bbox'; minX: number; maxX: number; minY: number; maxY: number }
  ): Promise<{
    type: 'point' | 'bbox'
    values: number[] // For point queries
    data: Float32Array // For bbox queries
    width: number
    height: number
    channels: number
    channelLabels: (string | number)[][]
    multiValueDimNames: string[]
  } | null> {
    if (!this.zarrArray) return null

    try {
      const { sliceArgs: baseSliceArgs, multiValueDims } =
        await this.buildSliceArgsForSelector(selector, {
          includeSpatialSlices: spatialQuery.type !== 'bbox',
          trackMultiValue: true,
          spatialBounds: spatialQuery,
        })

      const {
        combinations: channelCombinations,
        labelCombinations: channelLabelCombinations,
      } = this.buildChannelCombinations(multiValueDims)
      const numChannels = channelCombinations.length || 1
      const multiValueDimNames = multiValueDims.map((d) => d.dimName)

      // Point query: fetch individual values
      if (spatialQuery.type === 'point') {
        const values: number[] = []

        if (numChannels === 1) {
          const result = await zarr.get(this.zarrArray, baseSliceArgs)
          const value = this.extractScalarValue(result)
          if (value === null) return null
          values.push(value)
        } else {
          for (let c = 0; c < numChannels; c++) {
            const sliceArgs = [...baseSliceArgs]
            const combo = channelCombinations[c]
            for (let i = 0; i < multiValueDims.length; i++) {
              sliceArgs[multiValueDims[i].dimIndex] = combo[i]
            }
            const result = await zarr.get(this.zarrArray, sliceArgs)
            const value = this.extractScalarValue(result)
            if (value !== null) values.push(value)
          }
        }

        return {
          type: 'point',
          values,
          data: new Float32Array(0),
          width: 1,
          height: 1,
          channels: numChannels,
          channelLabels: channelLabelCombinations,
          multiValueDimNames,
        }
      }

      // Bbox query: fetch region data
      const fetchWidth = spatialQuery.maxX - spatialQuery.minX
      const fetchHeight = spatialQuery.maxY - spatialQuery.minY

      if (numChannels === 1) {
        const result = (await zarr.get(this.zarrArray, baseSliceArgs)) as {
          data: ArrayLike<number>
        }
        return {
          type: 'bbox',
          values: [],
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

        const bandData = (await zarr.get(this.zarrArray, sliceArgs)) as {
          data: ArrayLike<number>
        }
        for (let pixIdx = 0; pixIdx < fetchWidth * fetchHeight; pixIdx++) {
          packedData[pixIdx * numChannels + c] = bandData.data[pixIdx]
        }
      }

      return {
        type: 'bbox',
        values: [],
        data: packedData,
        width: fetchWidth,
        height: fetchHeight,
        channels: numChannels,
        channelLabels: channelLabelCombinations,
        multiValueDimNames,
      }
    } catch (err) {
      console.error('Error fetching query data:', err)
      return null
    }
  }

  /**
   * Extract a scalar value from zarr.get result (handles various return formats).
   */
  private extractScalarValue(result: unknown): number | null {
    if (result && typeof result === 'object' && 'data' in result) {
      const data = (result as { data: ArrayLike<number> }).data
      return data[0] ?? (data as unknown as number)
    } else if (typeof result === 'number') {
      return result
    } else if (ArrayBuffer.isView(result)) {
      return (result as Float32Array)[0]
    }
    console.warn('Unexpected zarr.get result format:', result)
    return null
  }

  /**
   * Query data for point or region geometries.
   */
  async queryData(
    geometry: QueryGeometry,
    selector?: Selector
  ): Promise<QueryResult> {
    if (!this.mercatorBounds) {
      return {
        [this.variable]: [],
        dimensions: [],
        coordinates: { lat: [], lon: [] },
      }
    }

    const normalizedSelector = selector
      ? normalizeSelector(selector)
      : this.selector

    const desc = this.zarrStore.describe()
    // Use per-level metadata if available (for heterogeneous pyramids)
    const currentLevel = this.levels[this.currentLevelIndex]
    const scaleFactor = currentLevel?.scaleFactor ?? desc.scaleFactor
    const addOffset = currentLevel?.addOffset ?? desc.addOffset
    const fill_value = currentLevel?.fillValue ?? desc.fill_value

    // Point geometries: use optimized single-chunk fetch
    if (geometry.type === 'Point') {
      const [lon, lat] = geometry.coordinates
      const coords = { lat: [lat], lon: [lon] }

      const sourceBounds = this.xyLimits
        ? ([
            this.xyLimits.xMin,
            this.xyLimits.yMin,
            this.xyLimits.xMax,
            this.xyLimits.yMax,
          ] as [number, number, number, number])
        : null
      const pixel = mercatorBoundsToPixel(
        lon,
        lat,
        this.mercatorBounds,
        this.width,
        this.height,
        this.crs ?? 'EPSG:4326',
        this.latIsAscending ?? undefined,
        this.proj4def,
        sourceBounds,
        this.cachedWGS84Transformer ?? undefined
      )

      if (!pixel) {
        return {
          [this.variable]: [],
          dimensions: ['lat', 'lon'],
          coordinates: coords,
        }
      }

      // Fetch only the chunk(s) containing this point
      const pointData = await this.fetchQueryData(normalizedSelector, {
        type: 'point',
        x: pixel.x,
        y: pixel.y,
      })

      if (!pointData) {
        return {
          [this.variable]: [],
          dimensions: ['lat', 'lon'],
          coordinates: coords,
        }
      }

      const { values: rawValues, channelLabels, multiValueDimNames } = pointData
      const valuesNested = multiValueDimNames.length > 0
      let values: number[] | Record<string | number, any> = valuesNested
        ? {}
        : []

      for (let c = 0; c < rawValues.length; c++) {
        let value = rawValues[c]

        // Filter invalid values
        if (value === undefined || value === null || !Number.isFinite(value)) {
          continue
        }
        if (fill_value !== null && value === fill_value) {
          continue
        }

        // Apply transforms
        if (scaleFactor !== 1) value *= scaleFactor
        if (addOffset !== 0) value += addOffset

        if (valuesNested) {
          const labels = channelLabels?.[c]
          if (
            labels &&
            multiValueDimNames.length > 0 &&
            labels.length === multiValueDimNames.length
          ) {
            values = setObjectValues(values as any, labels, value) as any
          } else if (Array.isArray(values)) {
            values.push(value)
          }
        } else if (Array.isArray(values)) {
          values.push(value)
        }
      }

      const dimensions = desc.dimensions
      const mappedDimensions = dimensions.map((d) => {
        const dimLower = d.toLowerCase()
        if (['x', 'lon', 'longitude'].includes(dimLower)) return 'lon'
        if (['y', 'lat', 'latitude'].includes(dimLower)) return 'lat'
        return d
      })

      const outputDimensions = valuesNested ? mappedDimensions : ['lat', 'lon']
      const resultCoordinates: {
        lat: number[]
        lon: number[]
        [key: string]: (number | string)[]
      } = {
        lat: coords.lat,
        lon: coords.lon,
      }

      if (valuesNested) {
        for (const dim of dimensions) {
          const dimLower = dim.toLowerCase()
          if (
            ['x', 'lon', 'longitude', 'y', 'lat', 'latitude'].includes(dimLower)
          ) {
            continue
          }
          const selSpec = normalizedSelector[dim]
          if (selSpec && 'selected' in selSpec) {
            const selected = selSpec.selected
            const vals = Array.isArray(selected) ? selected : [selected]
            resultCoordinates[dim] = vals as (number | string)[]
          } else if (desc.coordinates[dim]) {
            resultCoordinates[dim] = desc.coordinates[dim]
          }
        }
      }

      return {
        [this.variable]: values as any,
        dimensions: outputDimensions,
        coordinates: resultCoordinates,
      }
    }

    // Region queries: calculate pixel bounds and fetch only that subset
    const sourceBounds: [number, number, number, number] | null = this.xyLimits
      ? [
          this.xyLimits.xMin,
          this.xyLimits.yMin,
          this.xyLimits.xMax,
          this.xyLimits.yMax,
        ]
      : null
    const pixelBounds = computePixelBoundsFromGeometry(
      geometry,
      this.mercatorBounds,
      this.width,
      this.height,
      this.crs ?? 'EPSG:4326',
      this.latIsAscending ?? undefined,
      this.proj4def,
      sourceBounds,
      this.cachedWGS84Transformer ?? undefined
    )

    if (!pixelBounds) {
      return {
        [this.variable]: [],
        dimensions: [],
        coordinates: { lat: [], lon: [] },
      }
    }

    const fetched = await this.fetchQueryData(normalizedSelector, {
      type: 'bbox',
      ...pixelBounds,
    })
    if (!fetched) {
      return {
        [this.variable]: [],
        dimensions: [],
        coordinates: { lat: [], lon: [] },
      }
    }

    // Compute adjusted bounds for the subset
    const { minX, maxX, minY, maxY } = pixelBounds
    const xRange = this.mercatorBounds.x1 - this.mercatorBounds.x0
    const yRange = this.mercatorBounds.y1 - this.mercatorBounds.y0
    const subsetBounds: MercatorBounds = {
      x0: this.mercatorBounds.x0 + (minX / this.width) * xRange,
      x1: this.mercatorBounds.x0 + (maxX / this.width) * xRange,
      y0: this.mercatorBounds.y0 + (minY / this.height) * yRange,
      y1: this.mercatorBounds.y0 + (maxY / this.height) * yRange,
    }
    // Preserve lat bounds if present (for EPSG:4326)
    if (
      this.mercatorBounds.latMin !== undefined &&
      this.mercatorBounds.latMax !== undefined
    ) {
      const latRange = this.mercatorBounds.latMax - this.mercatorBounds.latMin
      if (this.latIsAscending) {
        subsetBounds.latMin =
          this.mercatorBounds.latMin + (minY / this.height) * latRange
        subsetBounds.latMax =
          this.mercatorBounds.latMin + (maxY / this.height) * latRange
      } else {
        subsetBounds.latMax =
          this.mercatorBounds.latMax - (minY / this.height) * latRange
        subsetBounds.latMin =
          this.mercatorBounds.latMax - (maxY / this.height) * latRange
      }
    }

    // For proj4 reprojection, compute subset bounds in source CRS
    let subsetSourceBounds: [number, number, number, number] | null = null
    if (this.proj4def && sourceBounds) {
      // Convert subset pixel corners to source CRS coordinates
      const [xMin, yMin] = pixelToSourceCRS(
        minX,
        minY,
        sourceBounds,
        this.width,
        this.height,
        this.latIsAscending ?? null
      )
      const [xMax, yMax] = pixelToSourceCRS(
        maxX - 1, // maxX is exclusive, so use maxX-1 for the last pixel
        maxY - 1,
        sourceBounds,
        this.width,
        this.height,
        this.latIsAscending ?? null
      )
      // Create subset bounds in source CRS
      subsetSourceBounds = [
        Math.min(xMin, xMax),
        Math.min(yMin, yMax),
        Math.max(xMin, xMax),
        Math.max(yMin, yMax),
      ]
    }

    return queryRegionSingleImage(
      this.variable,
      geometry,
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
      this.latIsAscending ?? undefined,
      {
        scaleFactor,
        addOffset,
        fillValue: fill_value,
      },
      this.proj4def,
      subsetSourceBounds
    )
  }
}
