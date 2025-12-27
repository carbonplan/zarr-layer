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
  private regionCache: Map<string, RegionState> = new Map()
  private previousRegionCache: Map<string, RegionState> = new Map() // Fallback during level transitions
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
    // Filter to only levels that don't already have shapes from consolidated metadata
    const levelsNeedingFetch = this.levels.filter((level) => !level.shape)

    if (levelsNeedingFetch.length === 0) {
      // All shapes pre-populated from consolidated metadata - no fetches needed
      return
    }

    await Promise.all(
      levelsNeedingFetch.map(async (level) => {
        try {
          const meta = await this.zarrStore.getUntiledLevelMetadata(level.asset)
          level.shape = meta.shape
          level.chunks = meta.chunks
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
  private clearRegionCache(gl: WebGL2RenderingContext): void {
    this.disposeRegionCache(this.regionCache, gl)
    this.regionCache.clear()
    this.lastViewportHash = ''
  }

  /**
   * Clear previous region cache (fallback during level transitions).
   */
  private clearPreviousRegionCache(gl: WebGL2RenderingContext): void {
    this.disposeRegionCache(this.previousRegionCache, gl)
    this.previousRegionCache.clear()
  }

  /**
   * Dispose WebGL resources for a region cache.
   */
  private disposeRegionCache(
    cache: Map<string, RegionState>,
    gl: WebGL2RenderingContext
  ): void {
    for (const region of cache.values()) {
      if (region.texture) gl.deleteTexture(region.texture)
      if (region.vertexBuffer) gl.deleteBuffer(region.vertexBuffer)
      if (region.pixCoordBuffer) gl.deleteBuffer(region.pixCoordBuffer)
      // Clean up band textures
      for (const tex of region.bandTextures.values()) {
        gl.deleteTexture(tex)
      }
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
   * Create a new region state entry.
   */
  private createRegionState(regionX: number, regionY: number): RegionState {
    return {
      key: `${regionX},${regionY}`,
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
   * Get all loaded regions from both caches (previous first for fallback, then current).
   */
  private getLoadedRegions(): RegionState[] {
    const regions: RegionState[] = []
    for (const region of this.previousRegionCache.values()) {
      if (this.isRegionValid(region)) regions.push(region)
    }
    for (const region of this.regionCache.values()) {
      if (this.isRegionValid(region)) regions.push(region)
    }
    return regions
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

    // Compute mercator bounds - use grid-aligned bounds for proj4 reprojection
    let mercBounds: MercatorBounds
    if (this.proj4def) {
      const gridAligned = this.getGridAlignedRegionBounds(regionX, regionY)
      mercBounds =
        gridAligned?.mercBounds ?? this.computeRegionMercatorBounds(geoBounds)
    } else {
      mercBounds = boundsToMercatorNorm(geoBounds, this.crs)
    }

    // Store mercator bounds for shader uniforms
    region.mercatorBounds = mercBounds

    // Subdivisions for smooth globe tessellation - more for larger regions
    const latSpan = Math.abs(geoBounds.yMax - geoBounds.yMin)
    const subdivisions = Math.max(16, Math.min(128, Math.ceil(latSpan)))
    const subdivided = createSubdividedQuad(subdivisions)

    region.vertexArr = subdivided.vertexArr

    // Determine texture coordinates based on CRS and data orientation
    if (this.crs === 'EPSG:4326') {
      // Resampled to Mercator on CPU - resampler handles latIsAscending internally
      region.pixCoordArr = subdivided.texCoordArr
    } else if (this.crs === 'EPSG:3857') {
      // Already in Mercator space, no resampling - handle latIsAscending manually
      // If latIsAscending (row 0 = south), flip V so V=0 samples north
      region.pixCoordArr = this.latIsAscending
        ? flipTexCoordV(subdivided.texCoordArr)
        : subdivided.texCoordArr
    } else {
      // Fallback for other CRS - use linear coords with latIsAscending handling
      region.pixCoordArr = this.latIsAscending
        ? flipTexCoordV(subdivided.texCoordArr)
        : subdivided.texCoordArr
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
   * Update visible regions based on current viewport.
   */
  private updateVisibleRegions(map: MapLike, gl: WebGL2RenderingContext): void {
    const visible = this.getVisibleRegions(map)

    // Separate regions into two categories:
    // 1. New regions (no data) - viewport change, fetch immediately
    // 2. Stale regions (have data, wrong selector) - selector change, throttle
    const newRegions: Array<{ regionX: number; regionY: number }> = []
    const staleRegions: Array<{ regionX: number; regionY: number }> = []

    for (const { regionX, regionY } of visible) {
      const key = `${regionX},${regionY}`
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

    // Check if viewport changed (include selectorVersion in hash to detect selector changes)
    const viewportHash = `${this.selectorVersion}:${visible
      .map((r) => `${r.regionX},${r.regionY}`)
      .join('|')}`
    const viewportChanged = viewportHash !== this.lastViewportHash
    this.lastViewportHash = viewportHash

    // Skip if nothing to do
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

    // Clear previous level cache once all visible regions are fully loaded with current selector
    // This ensures previous (potentially higher-res) data stays visible as fallback during zoom-out
    const allVisibleLoaded = visible.every(({ regionX, regionY }) => {
      const key = `${regionX},${regionY}`
      const region = this.regionCache.get(key)
      return region && region.data && !region.loading
    })

    if (
      this.previousRegionCache.size > 0 &&
      newRegions.length === 0 &&
      staleRegions.length === 0 &&
      allVisibleLoaded
    ) {
      this.clearPreviousRegionCache(gl)
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
    // Emit loading state
    this.loadingManager.chunksLoading = true
    this.emitLoadingState()

    // Mark ALL regions as loading upfront to prevent duplicate fetches
    // from subsequent update() calls before we've processed them all
    for (const { regionX, regionY } of regions) {
      const key = `${regionX},${regionY}`
      let region = this.regionCache.get(key)
      if (!region) {
        region = this.createRegionState(regionX, regionY)
        this.regionCache.set(key, region)
      }
      region.loading = true
    }

    // Limit concurrent fetches to avoid ERR_INSUFFICIENT_RESOURCES
    const MAX_CONCURRENT = 6
    const executing: Promise<void>[] = []

    for (const region of regions) {
      const promise = this.fetchRegion(region.regionX, region.regionY, gl)
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

    // Wait for remaining requests
    await Promise.allSettled(executing)

    // Check if any regions are still loading
    if (!hasActiveRequests(this.requestCanceller)) {
      this.loadingManager.chunksLoading = false
      this.emitLoadingState()
    }
  }

  /**
   * Fetch data for a single region.
   * Handles multi-band extraction when selector has multi-value dimensions.
   */
  private async fetchRegion(
    regionX: number,
    regionY: number,
    gl: WebGL2RenderingContext
  ): Promise<void> {
    if (!this.zarrArray || !this.regionSize || this.isRemoved) {
      return
    }

    const key = `${regionX},${regionY}`
    const requestId = ++this.requestCanceller.currentVersion
    const fetchSelectorVersion = this.selectorVersion // Capture current version

    const controller = new AbortController()
    this.requestCanceller.controllers.set(requestId, controller)

    let region = this.regionCache.get(key)
    if (!region) {
      region = this.createRegionState(regionX, regionY)
      this.regionCache.set(key, region)
    }
    region.loading = true

    const [regionH, regionW] = this.regionSize

    // Calculate pixel bounds for this region
    const yStart = regionY * regionH
    const yEnd = Math.min(yStart + regionH, this.height)
    const xStart = regionX * regionW
    const xEnd = Math.min(xStart + regionW, this.width)
    const actualW = xEnd - xStart
    const actualH = yEnd - yStart

    try {
      // Build base slice args with spatial region bounds
      const baseSliceArgs = [...this.baseSliceArgs]
      const latIdx = this.dimIndices.lat.index
      const lonIdx = this.dimIndices.lon.index
      baseSliceArgs[latIdx] = zarr.slice(yStart, yEnd)
      baseSliceArgs[lonIdx] = zarr.slice(xStart, xEnd)

      const desc = this.zarrStore.describe()
      const fillValue = desc.fill_value

      const { combinations: channelCombinations } =
        this.buildChannelCombinations(this.baseMultiValueDims)
      const numChannels = channelCombinations.length || 1
      const pixelCount = actualW * actualH

      // Fetch data for all channels
      const bandArrays: Float32Array[] = []
      const packedData = new Float32Array(pixelCount * numChannels)
      packedData.fill(fillValue ?? 0)

      if (numChannels === 1) {
        // Single channel - simple fetch
        const result = (await zarr.get(this.zarrArray, baseSliceArgs, {
          opts: { signal: controller.signal },
        })) as { data: ArrayLike<number> }

        if (controller.signal.aborted || this.isRemoved) {
          region.loading = false
          return
        }

        const rawData = new Float32Array(result.data as ArrayLike<number>)
        bandArrays.push(rawData)
        packedData.set(rawData)
      } else {
        // Multi-channel - fetch each channel's data
        for (let c = 0; c < numChannels; c++) {
          const sliceArgs = [...baseSliceArgs]
          const combo = channelCombinations[c]

          // Apply channel-specific indices to multi-value dimensions
          for (let i = 0; i < this.baseMultiValueDims.length; i++) {
            sliceArgs[this.baseMultiValueDims[i].dimIndex] = combo[i]
          }

          const result = (await zarr.get(this.zarrArray, sliceArgs, {
            opts: { signal: controller.signal },
          })) as { data: ArrayLike<number> }

          if (controller.signal.aborted || this.isRemoved) {
            region.loading = false
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

        // For proj4 reprojection, use grid-aligned bounds to eliminate seams
        let mercBounds: MercatorBounds
        if (this.proj4def) {
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

      // Normalize bands (single pass) and collect for interleaving
      region.bandData.clear()
      region.bandTexturesUploaded.clear()
      const normalizedBands: Float32Array[] = []

      for (let c = 0; c < bandDataToProcess.length; c++) {
        const bandName = this.bandNames[c] || `band_${c}`
        const { normalized: bandNormalized } = normalizeDataForTexture(
          bandDataToProcess[c],
          fillValue,
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
      } else if (bestLevelIndex !== this.currentLevelIndex) {
        // Zoom changed enough to warrant level switch
        this.switchToLevel(bestLevelIndex, gl)
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

    // Always compute worldFraction in Web Mercator meters
    let mercXMin = this.xyLimits.xMin
    let mercXMax = this.xyLimits.xMax
    const dataWidth = Math.abs(this.xyLimits.xMax - this.xyLimits.xMin)
    // If CRS is not EPSG:3857, transform bounds to Web Mercator
    if (this.crs !== 'EPSG:3857') {
      try {
        // Use proj4 if available
        if (typeof window.proj4 === 'function' && this.crs) {
          const toMerc = window.proj4(this.crs, 'EPSG:3857')
          const min = toMerc.forward([this.xyLimits.xMin, this.xyLimits.yMin])
          const max = toMerc.forward([this.xyLimits.xMax, this.xyLimits.yMax])
          mercXMin = Math.min(min[0], max[0])
          mercXMax = Math.max(min[0], max[0])
        } else if (this.crs === 'EPSG:4326' && dataWidth > 10000) {
          // Heuristic: bounds are already in meters, do not convert
          mercXMin = this.xyLimits.xMin
          mercXMax = this.xyLimits.xMax
        } else {
          // Fallback: treat as degrees, approximate conversion
          const DEG2RAD = Math.PI / 180
          const R = 6378137
          const xToMerc = (x: number) => R * x * DEG2RAD
          mercXMin = xToMerc(this.xyLimits.xMin)
          mercXMax = xToMerc(this.xyLimits.xMax)
        }
      } catch (e) {
        console.warn('[UntiledMode] Failed to transform bounds to EPSG:3857:', e)
      }
    }
    const dataWidthMeters = Math.abs(mercXMax - mercXMin)
    const fullWorldMeters = 2 * WEB_MERCATOR_EXTENT
    const worldFraction = dataWidthMeters / fullWorldMeters

    // Build list of levels with their effective resolution (pixels per full world)
    const levelResolutions: Array<{ index: number; effectivePixels: number }> = []
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

    // Find the coarsest (lowest-res) level that still provides sufficient detail
    // (at least 50% of map's pixel density to avoid obvious pixelation)
    const minRequired = mapPixelsPerWorld * 0.5
    for (let i = 0; i < levelResolutions.length; i++) {
      if (levelResolutions[i].effectivePixels >= minRequired) {
        return levelResolutions[i].index;
      }
    }
    // If none meet the requirement, return the coarsest (last) level
    return levelResolutions[levelResolutions.length - 1].index;
  }

  private async switchToLevel(
    newLevelIndex: number,
    gl: WebGL2RenderingContext
  ): Promise<void> {
    if (newLevelIndex === this.currentLevelIndex) return
    if (newLevelIndex < 0 || newLevelIndex >= this.levels.length) return
    if (this.loadingManager.chunksLoading) return // Don't interrupt ongoing load

    const level = this.levels[newLevelIndex]

    // Cancel any pending region requests
    cancelAllRequests(this.requestCanceller)

    this.currentLevelIndex = newLevelIndex

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

      // Move current regions to previous cache (for fallback during transition)
      // Clear any existing previous cache first
      this.clearPreviousRegionCache(gl)
      this.previousRegionCache = this.regionCache
      this.regionCache = new Map()

      this.zarrArray = newArray
      this.width = newWidth
      this.height = newHeight
      this.regionSize = newRegionSize
      this.computeGlobalMercatorGrid()
      this.lastViewportHash = '' // Force viewport recalculation

      // Build base slice args for non-spatial dimensions
      await this.buildBaseSliceArgs()

      // Let update() trigger viewport-aware loading
      this.invalidate()
    } catch (err) {
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
      context.uniforms,
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
      context,
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

  dispose(gl: WebGL2RenderingContext): void {
    this.isRemoved = true
    clearThrottle(this.throttleState)
    cancelAllRequests(this.requestCanceller)
    // Clean up region caches
    this.clearRegionCache(gl)
    this.clearPreviousRegionCache(gl)
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
          data: new Float32Array((result.data as Float32Array).buffer),
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
        const bandArray = new Float32Array(
          (bandData.data as Float32Array).buffer
        )
        for (let pixIdx = 0; pixIdx < fetchWidth * fetchHeight; pixIdx++) {
          packedData[pixIdx * numChannels + c] = bandArray[pixIdx]
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
    const { scaleFactor, addOffset, fill_value } = desc

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
        scaleFactor: desc.scaleFactor,
        addOffset: desc.addOffset,
        fillValue: desc.fill_value,
      },
      this.proj4def,
      subsetSourceBounds
    )
  }
}
