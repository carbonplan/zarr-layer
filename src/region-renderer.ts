/**
 * @module region-renderer
 *
 * The unified renderer for every Zarr dataset. Reads the visible region of a
 * resolution level as chunk-sized sub-rectangles, reprojects each onto an
 * adaptive source→WGS84 mesh, and lets the GPU project to Mercator or ECEF.
 * Handles single-level datasets, untiled multiscale pyramids, and tiled
 * (slippy-map) pyramids alike — a tiled pyramid is just a multiscale whose
 * levels are arrays chunked at the tile size (see ZarrStore). Automatic level
 * selection is driven by map zoom.
 */

import * as zarr from 'zarrita'
import { MIN_SUBDIVISIONS, MAX_SUBDIVISIONS } from './constants'
import type {
  RenderContext,
  TileId,
  RegionRenderState,
  CustomShaderConfig,
} from './renderer-types'
import type { QueryGeometry, QueryOptions, QueryResult } from './query/types'
import type {
  LoadingStateCallback,
  MapLike,
  NormalizedSelector,
  Selector,
  Bounds,
  DimIndicesProps,
  UntiledLevel,
} from './types'
import { ZarrStore } from './zarr-store'
import { type MercatorBounds, type XYLimits } from './map-utils'
import { getBands } from './zarr-utils'
import { interleaveBands, normalizeDataForTexture } from './webgl-utils'
import type { ZarrRenderer, ShaderProgram } from './zarr-renderer'
import { renderMapboxTile } from './mapbox-tile-renderer'
import {
  createProjectionContext,
  type ProjectionContext,
} from './projection-utils'
import type {
  LevelMeta,
  LevelRuntime,
  LevelSnapshot,
  RegionState,
} from './region-state'
import {
  buildChannelCombinations,
  buildSliceArgsForSelector,
  type DimensionValuesCache,
} from './selector-resolution'
import { queryData as queryDataWithContext } from './query/data-query'
import {
  computeMercatorBoundsFromProjection,
  computeRegionMercatorBounds,
  getRegionBounds,
  getRegionSize,
  getVisibleRegions,
  selectLevelForZoom,
} from './region-math'
import { createHybridMesh } from './mesh-reprojector'
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
} from './region-utils'
import { setupBandTextureUniforms, uploadDataTexture } from './render-helpers'
import { renderRegion, type RenderableRegion } from './renderable-region'

/** Maximum number of regions to keep in cache (LRU eviction) */
const MAX_CACHED_REGIONS = 128

export class RegionRenderer {
  isMultiscale: boolean = false

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
  private latIsAscending: boolean = true

  // Multi-level support
  private levels: UntiledLevel[] = []
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
  private requestCanceller: RequestCanceller = createRequestCanceller()
  private loadingManager: LoadingManager = createLoadingManager()
  private loadingDebouncer: ChunkLoadingDebouncer = createChunkLoadingDebouncer(
    this.loadingManager
  )

  // Dimension values cache (supports numeric and string coordinate arrays)
  private dimensionValues: DimensionValuesCache = {}

  // Region-based loading (for multi-level datasets with chunking/sharding)
  // Single unified cache with LRU eviction - keys include level index (e.g., "2:0,0")
  private regionCache: Map<string, RegionState> = new Map()
  // Keys of regions protected from eviction. Rebuilt in updateVisibleRegions()
  // from the current viewport: the current level's visible regions, plus
  // other-level regions retained as fallbacks until the current level covers
  // the viewport.
  private visibleRegionKeys: Set<string> = new Set()
  private lastVisibleRegions: Array<{ regionX: number; regionY: number }> = [] // Last computed visible regions
  private lastVisibleRegionsLevel: number = -1 // Level index that lastVisibleRegions corresponds to
  private lastViewportHash: string = ''
  private selectorVersion: number = 0 // Incremented on selector change to track stale regions

  // Cached WebGL context for use in setSelector
  private cachedGl: WebGL2RenderingContext | null = null
  // Track current projection for Mapbox's direct globe render path.
  private isGlobeProjection: boolean = false
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
      this.xyLimits = desc.xyLimits
      this.latIsAscending = desc.latIsAscending
      // Cache transformers once for reuse (major performance optimization)
      this.projection = createProjectionContext({
        crs: desc.crs,
        proj4def: desc.proj4,
        xyLimits: this.xyLimits,
      })

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

  private getRegionSize(
    array: zarr.Array<zarr.DataType>
  ): [number, number] | null {
    return getRegionSize(array, this.dimIndices)
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
   * Uses the source-projected adaptive mesh path for all supported untiled CRSes.
   */
  private createRegionGeometry(
    regionX: number,
    regionY: number,
    gl: WebGL2RenderingContext,
    region: RegionState
  ): void {
    // Guard: can't create geometry without dimension info
    if (!region.levelMeta) return

    // Defensive reset: wgs84Bounds is only set by the source-projected branch.
    // Ensures it's not stale if geometry is recreated.
    region.wgs84Bounds = null
    region.indexArr = null
    region.useIndexedMesh = false

    if (
      !this.projection.def ||
      !this.projection.to4326 ||
      !this.projection.toMercator
    ) {
      return
    }

    const geoBounds = this.getRegionBounds(regionX, regionY, region.levelMeta)

    if (!region.mercatorBounds) {
      region.mercatorBounds = this.computeRegionMercatorBounds(geoBounds)
    }

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
    region.wgs84Bounds = meshResult.wgs84Bounds
    region.useIndexedMesh = true
    region.vertexCount = region.indexArr.length

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
    const coordLevelIndex =
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

    const visibleKeys = new Set(
      visible.map(({ regionX, regionY }) =>
        this.makeRegionKey(levelIndex, regionX, regionY)
      )
    )

    // Rebuild eviction protection from the current viewport: this level's
    // visible regions, plus other-level regions kept as fallbacks until the
    // current level covers the viewport. An empty result can be a transient
    // state (map bounds or transformer unavailable), so keep the previous
    // set rather than unprotect regions that are still rendered.
    if (visible.length > 0) {
      const nextProtected = new Set(visibleKeys)
      if (!this.currentLevelCoversViewport()) {
        const currentLevelPrefix = `${levelIndex}:`
        for (const key of this.visibleRegionKeys) {
          if (!key.startsWith(currentLevelPrefix)) {
            nextProtected.add(key)
          }
        }
      }
      this.visibleRegionKeys = nextProtected
    }

    // Abort in-flight fetches for regions that left the viewport.
    // Only signal abort — the fetch's own catch/finally handles state cleanup.
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

    // The browser drains requests roughly in issue order, so the viewport
    // center loads first.
    if (visible.length > 1) {
      let cx = 0
      let cy = 0
      for (const { regionX, regionY } of visible) {
        cx += regionX
        cy += regionY
      }
      cx /= visible.length
      cy /= visible.length
      const byCenterDistance = (
        a: { regionX: number; regionY: number },
        b: { regionX: number; regionY: number }
      ) =>
        (a.regionX - cx) ** 2 +
        (a.regionY - cy) ** 2 -
        (b.regionX - cx) ** 2 -
        (b.regionY - cy) ** 2
      newRegions.sort(byCenterDistance)
      staleRegions.sort(byCenterDistance)
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

      const { combinations: channelCombinations } = buildChannelCombinations(
        snapshot.baseMultiValueDims
      )
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

      // GPU handles reprojection. Source-projected data uses an adaptive mesh
      // (source CRS → WGS84), then the GPU projects to Mercator or ECEF.
      const needsProj4MercBounds =
        this.projection.def && this.projection.toMercator

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

      // Store level-specific dimensions from snapshot for geometry creation.
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
    return selectLevelForZoom({
      mapZoom,
      xyLimits: this.xyLimits,
      levels: this.levels,
      projection: this.projection,
      lonIndex: this.dimIndices.lon?.index,
    })
  }

  render(renderer: ZarrRenderer, context: RenderContext): void {
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
      useDirectEcef,
      eyeMatrix
    )
  }

  /**
   * Convert a RegionState to a RenderableRegion for unified rendering.
   * When useDirectEcef is true, uses the region's precomputed WGS84 mesh bounds
   * for the ECEF vertex shader path. Render-only fields are set here instead of
   * cached on RegionState, so projection toggles have no stale state.
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
    useDirectEcef: boolean = false,
    eyeMatrix: number[] | Float32Array | Float64Array | null = null
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
        customShaderConfig,
        eyeMatrix
      )
    }
  }

  renderToTile(
    renderer: ZarrRenderer,
    tileId: TileId,
    context: RenderContext
  ): boolean {
    // This method is only used for draped Mapbox rendering. The direct
    // ECEF path disables renderToTile at the layer level.
    return renderMapboxTile({
      renderer,
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

  /** Query data for point or region geometries. */
  async queryData(
    geometry: QueryGeometry,
    selector?: Selector,
    options?: QueryOptions
  ): Promise<QueryResult> {
    const activeLevel = this.activeLevel
    return queryDataWithContext(
      {
        zarrStore: this.zarrStore,
        variable: this.variable,
        selector: this.selector,
        xyLimits: this.xyLimits,
        mercatorBounds: this.mercatorBounds,
        latIsAscending: this.latIsAscending,
        levels: this.levels,
        level: activeLevel
          ? {
              index: activeLevel.index,
              zarrArray: activeLevel.zarrArray,
              width: activeLevel.width,
              height: activeLevel.height,
            }
          : null,
        projection: this.projection,
        antimeridianWarnings: this._antimeridianWarnings,
        dimensionValues: this.dimensionValues,
        isMultiscale: this.isMultiscale,
        coordLevelIndex: activeLevel?.index ?? 0,
      },
      geometry,
      selector,
      options
    )
  }
}
