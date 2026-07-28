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
import type { ZarrRenderer, ShaderProgram } from './zarr-renderer'
import { renderMapboxTile } from './mapbox-tile-renderer'
import {
  createProjectionContext,
  type ProjectionContext,
} from './projection-utils'
import type { LevelMeta, LevelRuntime, RegionState } from './region-state'
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
  getVisibleRegions,
  selectLevelForZoom,
} from './region-math'
import { RegionCache, isRegionValid, makeRegionKey } from './region-cache'
import { RegionFetcher } from './region-fetcher'
import { LevelLoader } from './level-loader'
import { createHybridMesh } from './mesh-reprojector'
import {
  type RequestCanceller,
  type LoadingManager,
  type ChunkLoadingDebouncer,
  createRequestCanceller,
  createLoadingManager,
  createChunkLoadingDebouncer,
  cancelAllRequests,
  setLoadingCallback as setLoadingCallbackUtil,
  emitLoadingState as emitLoadingStateUtil,
} from './region-utils'
import {
  ensureRegionGpuResources,
  setupBandTextureUniforms,
} from './render-helpers'
import { renderRegion, type RenderableRegion } from './renderable-region'

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
  private regionCache = new RegionCache()
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
            reusedArray: true,
          }
        }
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
          reusedArray: false,
        }
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
        if (this.requestCanceller.controllers.size > 0) {
          cancelAllRequests(this.requestCanceller)
          this.loadingDebouncer.hide()
        }
      },
      onNewArrayCommitted: () => this.resetVisibleRegions(),
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
   * Check if a region has all required data for rendering.
   */
  private isRegionValid(region: RegionState): boolean {
    return isRegionValid(region)
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
      if (!this.regionCache.isProtected(region.key)) continue
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
    region: RegionState
  ): void {
    // Guard: can't create geometry without dimension info
    if (!region.levelMeta) return

    // Defensive reset: wgs84Bounds is only set by the source-projected branch.
    // Ensures it's not stale if geometry is recreated.
    region.wgs84Bounds = null
    region.indexArr = null
    region.useIndexedMesh = false
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
    region.wgs84Bounds = meshResult.wgs84Bounds
    region.useIndexedMesh = true
    region.vertexCount = region.indexArr.length
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
      this.regionCache.rebuildProtection(visibleKeys, {
        retainKeysNotMatching: this.currentLevelCoversViewport()
          ? ''
          : `${levelIndex}:`,
      })
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
      this.fetchRegions(newRegions)
    }
    if (staleRegions.length > 0) {
      this.fetchRegions(staleRegions)
    }
    this.evictOldRegions(gl)
  }

  private async fetchRegions(
    regions: Array<{ regionX: number; regionY: number }>
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
      getActiveLevel: () => this.activeLevel,
      getSelectorVersion: () => this.selectorVersion,
      getBandNames: () => this.bandNames,
      isRemoved: () => this.isRemoved,
      getRegionBounds: (regionX, regionY, levelMeta) =>
        this.getRegionBounds(regionX, regionY, levelMeta),
      computeRegionMercatorBounds: (bounds) =>
        this.computeRegionMercatorBounds(bounds),
      createRegionGeometry: (regionX, regionY, region) =>
        this.createRegionGeometry(regionX, regionY, region),
      invalidate: this.invalidate,
    })
    await fetcher.fetchRegions(regions)
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

  private loadLevel(
    levelIndex: number,
    options: { reuseArray?: boolean } = {}
  ): Promise<void> {
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
      if (!ensureRegionGpuResources(gl, region)) continue
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

    return this.getLoadedRegions()
      .filter((region) => ensureRegionGpuResources(gl, region))
      .map((region) => ({
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
