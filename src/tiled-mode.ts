import type {
  ZarrMode,
  RenderContext,
  TileId,
  TiledRenderState,
} from './zarr-mode'
import type { QueryGeometry, QueryResult } from './query/types'
import { queryRegionTiled } from './query/region-query'
import type {
  LoadingStateCallback,
  MapLike,
  NormalizedSelector,
  Selector,
  CRS,
} from './types'
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
  setLoadingCallback as setLoadingCallbackUtil,
  emitLoadingState as emitLoadingStateUtil,
} from './mode-utils'
import { ZarrStore } from './zarr-store'
import { Tiles } from './tiles'
import {
  getTilesAtZoom,
  getTilesAtZoomEquirect,
  isGlobeProjection,
  latToMercatorNorm,
  lonToMercatorNorm,
  normalizeGlobalExtent,
  parseLevelZoom,
  type MercatorBounds,
  tileToKey,
  TileTuple,
  zoomToLevel,
  type XYLimits,
} from './map-utils'

/** Full tile bounds with all required fields (lat/lon + mercator) */
interface FullTileBounds {
  latMin: number
  latMax: number
  lonMin: number
  lonMax: number
  x0: number
  y0: number
  x1: number
  y1: number
}

/**
 * Convert MercatorBounds to FullTileBounds if all lat/lon fields are present.
 * Returns undefined if any required field is missing.
 */
function toFullBounds(
  bounds: MercatorBounds | undefined
): FullTileBounds | undefined {
  if (
    !bounds ||
    bounds.latMin === undefined ||
    bounds.latMax === undefined ||
    bounds.lonMin === undefined ||
    bounds.lonMax === undefined
  ) {
    return undefined
  }
  return {
    latMin: bounds.latMin,
    latMax: bounds.latMax,
    lonMin: bounds.lonMin,
    lonMax: bounds.lonMax,
    x0: bounds.x0,
    y0: bounds.y0,
    x1: bounds.x1,
    y1: bounds.y1,
  }
}
import { getBands, normalizeSelector } from './zarr-utils'
import { createSubdividedQuad } from './webgl-utils'
import {
  DEFAULT_TILE_SIZE,
  MAX_CACHED_TILES,
  TILE_SUBDIVISIONS,
} from './constants'
import type { ZarrRenderer } from './zarr-renderer'
import { renderMapboxTile } from './mapbox-globe-tile-renderer'

export class TiledMode implements ZarrMode {
  isMultiscale: true = true
  private tileCache: Tiles | null = null
  private vertexArr: Float32Array = new Float32Array()
  private pixCoordArr: Float32Array = new Float32Array()
  private currentSubdivisions: number = 0
  private maxLevelIndex: number = 0
  private tileSize: number = DEFAULT_TILE_SIZE
  private variable: string
  private selector: NormalizedSelector
  private invalidate: () => void
  private zarrStore: ZarrStore
  private visibleTiles: TileTuple[] = []
  private crs: CRS = 'EPSG:4326'
  private xyLimits: XYLimits | null = null
  private tileBounds: Record<string, MercatorBounds> = {}
  private pendingChunks: Set<string> = new Set()
  private currentLevel: number | null = null
  private selectorVersion: number = 0
  private throttleMs: number

  // Shared state managers
  private throttleState: ThrottleState = createThrottleState()
  private requestCanceller: RequestCanceller = createRequestCanceller()
  private loadingManager: LoadingManager = createLoadingManager()

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
    this.invalidate = invalidate
    this.throttleMs = throttleMs
  }

  async initialize(): Promise<void> {
    this.loadingManager.metadataLoading = true
    this.emitLoadingState()

    try {
      const desc = this.zarrStore.describe()
      this.maxLevelIndex = desc.levels.length - 1
      this.tileSize = desc.tileSize || DEFAULT_TILE_SIZE
      this.crs = desc.crs
      this.xyLimits = desc.xyLimits

      const bandNames = getBands(this.variable, this.selector)

      // Create unified tile cache (handles both data and WebGL resources)
      this.tileCache = new Tiles({
        store: this.zarrStore,
        selector: this.selector,
        fillValue: desc.fill_value ?? 0,
        dimIndices: desc.dimIndices,
        coordinates: desc.coordinates,
        maxCachedTiles: MAX_CACHED_TILES,
        bandNames,
        crs: this.crs,
      })

      this.updateGeometryForProjection(false)
    } finally {
      this.loadingManager.metadataLoading = false
      this.emitLoadingState()
    }
  }

  update(map: MapLike, gl: WebGL2RenderingContext): void {
    if (!this.tileCache) {
      return
    }

    // Initialize WebGL context for the unified cache
    this.tileCache.setGL(gl)

    const projection = map.getProjection ? map.getProjection() : null
    const isGlobe = isGlobeProjection(projection)
    this.updateGeometryForProjection(isGlobe)

    const visibleInfo = this.getVisibleTilesWithContext(map)
    this.visibleTiles = visibleInfo.tiles
    this.tileBounds = this.computeTileBounds(this.visibleTiles)
    if (visibleInfo.pyramidLevel !== null) {
      this.currentLevel = visibleInfo.pyramidLevel
    }

    // Pass bounds to tile cache for CPU resampling (EPSG:4326 only)
    for (const [tileKey, mercBounds] of Object.entries(this.tileBounds)) {
      const fullBounds = toFullBounds(mercBounds)
      if (fullBounds) {
        this.tileCache.setTileBounds(tileKey, fullBounds)
      }
    }

    const currentHash = JSON.stringify(this.selector)
    const tilesToFetch: TileTuple[] = []

    for (const tileTuple of this.visibleTiles) {
      const tileKey = tileToKey(tileTuple)
      if (this.pendingChunks.has(tileKey)) {
        continue
      }
      const tile = this.tileCache.upsert(tileKey)
      if (!tile.data || tile.selectorHash !== currentHash) {
        tilesToFetch.push(tileTuple)
      }
    }

    if (tilesToFetch.length > 0) {
      // Throttle: if too soon since last fetch, schedule a trailing update
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

      const wasEmpty = this.pendingChunks.size === 0
      for (const tileTuple of tilesToFetch) {
        this.pendingChunks.add(tileToKey(tileTuple))
      }
      if (wasEmpty) {
        this.emitLoadingState()
      }
      const version = this.selectorVersion
      this.prefetchTileData(tilesToFetch, currentHash, version).catch((err) => {
        console.error('Error prefetching tile data:', err)
        for (const tileTuple of tilesToFetch) {
          this.pendingChunks.delete(tileToKey(tileTuple))
        }
        this.emitLoadingState()
      })
    }
  }

  render(renderer: ZarrRenderer, context: RenderContext): void {
    if (!this.tileCache) {
      return
    }

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

    // Calculate datasetMaxZoom from the highest resolution level
    const maxLevelPath = this.zarrStore.levels[this.maxLevelIndex]
    const datasetMaxZoom = parseLevelZoom(maxLevelPath, this.maxLevelIndex)

    renderer.renderTiles(
      shaderProgram,
      this.visibleTiles,
      context.worldOffsets,
      this.tileCache,
      this.tileSize,
      this.vertexArr,
      this.pixCoordArr,
      Object.keys(this.tileBounds).length > 0 ? this.tileBounds : undefined,
      context.customShaderConfig,
      false,
      datasetMaxZoom
    )
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
    })
  }

  onProjectionChange(isGlobe: boolean) {
    this.updateGeometryForProjection(isGlobe)
  }

  getTiledState(): TiledRenderState | null {
    if (!this.tileCache) return null
    return {
      tileCache: this.tileCache,
      visibleTiles: this.visibleTiles,
      tileSize: this.tileSize,
      vertexArr: this.vertexArr,
      pixCoordArr: this.pixCoordArr,
      tileBounds:
        Object.keys(this.tileBounds).length > 0 ? this.tileBounds : undefined,
      latIsAscending: this.zarrStore.latIsAscending,
    }
  }

  getSingleImageState() {
    return null
  }

  dispose(_gl: WebGL2RenderingContext | WebGLRenderingContext): void {
    clearThrottle(this.throttleState)
    cancelAllRequests(this.requestCanceller)
    this.tileCache?.clear()
    this.tileCache = null
    this.pendingChunks.clear()
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

  getMaxLevelIndex(): number {
    return this.maxLevelIndex
  }

  getLevels(): string[] {
    return this.zarrStore.levels
  }

  updateClim(clim: [number, number]): void {
    this.tileCache?.updateClim(clim)
  }

  private emitLoadingState(): void {
    // Update chunksLoading state based on pending chunks and throttle state
    this.loadingManager.chunksLoading =
      this.pendingChunks.size > 0 || this.throttleState.throttledPending
    emitLoadingStateUtil(this.loadingManager)
  }

  async setSelector(selector: NormalizedSelector): Promise<void> {
    this.selector = selector
    this.selectorVersion++
    const bandNames = getBands(this.variable, selector)

    this.tileCache?.updateSelector(this.selector)
    this.tileCache?.updateBandNames(bandNames)

    if (this.tileCache && this.visibleTiles.length > 0) {
      const currentHash = JSON.stringify(this.selector)
      // Unified cache handles both data and texture state
      await this.tileCache.reextractTileSlices(
        this.visibleTiles,
        currentHash,
        this.selectorVersion
      )
    }

    this.invalidate()
  }

  private updateGeometryForProjection(isGlobe: boolean) {
    // Globe projections need subdivisions for the sphere curvature.
    const targetSubdivisions = isGlobe ? TILE_SUBDIVISIONS : 1

    if (this.currentSubdivisions === targetSubdivisions) return

    const subdivided = createSubdividedQuad(targetSubdivisions)
    this.vertexArr = subdivided.vertexArr
    this.pixCoordArr = subdivided.texCoordArr
    this.currentSubdivisions = targetSubdivisions
    this.tileCache?.markGeometryDirty()
  }

  private getVisibleTilesWithContext(map: MapLike): {
    tiles: TileTuple[]
    pyramidLevel: number | null
    mapZoom: number | null
    bounds: [[number, number], [number, number]] | null
  } {
    if (!map.getZoom || !map.getBounds) {
      return { tiles: [], pyramidLevel: null, mapZoom: null, bounds: null }
    }

    const mapZoom = map.getZoom()
    const levelIndex = zoomToLevel(mapZoom, this.maxLevelIndex)
    const bounds = map.getBounds()?.toArray()
    if (!bounds) {
      return { tiles: [], pyramidLevel: levelIndex, mapZoom, bounds: null }
    }

    // Parse actual zoom from level path to handle pyramids that don't start at 0
    const levelPath = this.zarrStore.levels[levelIndex]
    const actualZoom = parseLevelZoom(levelPath, levelIndex)

    if (this.crs === 'EPSG:4326' && this.xyLimits) {
      return {
        tiles: getTilesAtZoomEquirect(actualZoom, bounds, this.xyLimits),
        pyramidLevel: levelIndex,
        mapZoom,
        bounds,
      }
    }
    return {
      tiles: getTilesAtZoom(actualZoom, bounds),
      pyramidLevel: levelIndex,
      mapZoom,
      bounds,
    }
  }

  private computeTileBounds(
    tiles: TileTuple[]
  ): Record<string, MercatorBounds> {
    if (this.crs !== 'EPSG:4326' || !this.xyLimits) return {}

    const { xMin, xMax, yMin, yMax } = normalizeGlobalExtent(this.xyLimits)
    const lonExtent = xMax - xMin
    const latExtent = yMax - yMin

    const bounds: Record<string, MercatorBounds> = {}
    for (const tile of tiles) {
      const [z, x, y] = tile
      const tilesPerSide = Math.pow(2, z)
      const lonSpan = lonExtent / tilesPerSide
      const latSpan = latExtent / tilesPerSide

      const lonMin = xMin + x * lonSpan
      const lonMax = lonMin + lonSpan
      const latNorth = yMax - y * latSpan
      const latSouth = latNorth - latSpan

      const x0 = lonToMercatorNorm(lonMin)
      const x1 = lonToMercatorNorm(lonMax)
      const y0 = latToMercatorNorm(latNorth)
      const y1 = latToMercatorNorm(latSouth)

      bounds[tileToKey(tile)] = {
        x0,
        y0,
        x1,
        y1,
        latMin: latSouth,
        latMax: latNorth,
        lonMin,
        lonMax,
      }
    }

    return bounds
  }

  private async prefetchTileData(
    tiles: TileTuple[],
    selectorHash: string,
    version: number
  ) {
    // Create AbortController for this version's requests
    const controller = new AbortController()
    this.requestCanceller.controllers.set(version, controller)

    try {
      const fetchPromises = tiles.map((tiletuple) =>
        this.fetchTileData(tiletuple, selectorHash, version, controller.signal)
      )
      await Promise.all(fetchPromises)
    } finally {
      this.requestCanceller.controllers.delete(version)
    }
  }

  private async fetchTileData(
    tileTuple: TileTuple,
    selectorHash: string,
    version: number,
    signal?: AbortSignal
  ): Promise<Float32Array | null> {
    if (!this.tileCache) {
      const tileKey = tileToKey(tileTuple)
      this.pendingChunks.delete(tileKey)
      this.emitLoadingState()
      return null
    }

    const tileKey = tileToKey(tileTuple)

    // Get bounds for this tile (for EPSG:4326 resampling)
    const bounds = toFullBounds(this.tileBounds[tileKey])

    try {
      // Unified cache handles both data fetching and WebGL resources
      const tile = await this.tileCache.fetchTile(
        tileTuple,
        selectorHash,
        version,
        signal,
        bounds
      )

      this.pendingChunks.delete(tileKey)

      if (!tile) {
        this.emitLoadingState()
        return null
      }

      // Cancel all older pending requests since a newer version has completed
      cancelOlderRequests(this.requestCanceller, version)

      this.emitLoadingState()

      // Always invalidate to show data as it arrives - this allows
      // intermediate frames to render when scrubbing through time
      this.invalidate()

      return tile.data
    } catch (err) {
      this.pendingChunks.delete(tileKey)
      this.emitLoadingState()
      // AbortError is expected when requests are cancelled
      if (err instanceof DOMException && err.name === 'AbortError') {
        return null
      }
      throw err
    }
  }

  /**
   * Query data for point or region geometries.
   */
  async queryData(
    geometry: QueryGeometry,
    selector?: Selector
  ): Promise<QueryResult> {
    if (!this.tileCache || !this.xyLimits) {
      return {
        [this.variable]: [],
        dimensions: [],
        coordinates: { lat: [], lon: [] },
      }
    }

    const querySelector = selector ? normalizeSelector(selector) : this.selector
    const level = this.currentLevel ?? this.maxLevelIndex
    const desc = this.zarrStore.describe()

    return queryRegionTiled(
      this.variable,
      geometry,
      querySelector,
      this.zarrStore,
      this.crs,
      this.xyLimits,
      level,
      this.tileSize,
      {
        scaleFactor: desc.scaleFactor,
        addOffset: desc.addOffset,
        fillValue: desc.fill_value,
      }
    )
  }
}
