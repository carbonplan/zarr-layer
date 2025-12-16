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
import { ZarrStore } from './zarr-store'
import { Tiles } from './tiles'
import {
  getTilesAtZoom,
  getTilesAtZoomEquirect,
  latToMercatorNorm,
  lonToMercatorNorm,
  normalizeGlobalExtent,
  type MercatorBounds,
  tileToKey,
  TileTuple,
  zoomToLevel,
  type XYLimits,
} from './map-utils'
import { getBands, normalizeSelector } from './zarr-utils'
import { createSubdividedQuad } from './webgl-utils'
import {
  DEFAULT_TILE_SIZE,
  MAX_CACHED_TILES,
  TILE_SUBDIVISIONS,
} from './constants'
import type { ZarrRenderer } from './zarr-renderer'
import { renderMapboxTile } from './mapbox-globe-tile-renderer'
import { isGlobeProjection } from './render-utils'

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
  private loadingCallback: LoadingStateCallback | undefined
  private pendingChunks: Set<string> = new Set()
  private metadataLoading: boolean = false
  private currentLevel: number | null = null

  constructor(
    store: ZarrStore,
    variable: string,
    selector: NormalizedSelector,
    invalidate: () => void
  ) {
    this.zarrStore = store
    this.variable = variable
    this.selector = selector
    this.invalidate = invalidate
  }

  async initialize(): Promise<void> {
    this.metadataLoading = true
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
      })

      this.updateGeometryForProjection(false)
    } finally {
      this.metadataLoading = false
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
      const wasEmpty = this.pendingChunks.size === 0
      for (const tileTuple of tilesToFetch) {
        this.pendingChunks.add(tileToKey(tileTuple))
      }
      if (wasEmpty) {
        this.emitLoadingState()
      }
      this.prefetchTileData(tilesToFetch, currentHash).catch((err) => {
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
      false
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
    }
  }

  getSingleImageState() {
    return null
  }

  dispose(_gl: WebGL2RenderingContext): void {
    this.tileCache?.clear()
    this.tileCache = null
    this.pendingChunks.clear()
    this.emitLoadingState()
  }

  setLoadingCallback(callback: LoadingStateCallback | undefined): void {
    this.loadingCallback = callback
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

  updateClim(clim: [number, number]): void {
    this.tileCache?.updateClim(clim)
  }

  private emitLoadingState(): void {
    if (!this.loadingCallback) return
    const chunksLoading = this.pendingChunks.size > 0
    this.loadingCallback({
      loading: this.metadataLoading || chunksLoading,
      metadata: this.metadataLoading,
      chunks: chunksLoading,
    })
  }

  async setSelector(selector: NormalizedSelector): Promise<void> {
    this.selector = selector
    const bandNames = getBands(this.variable, selector)

    this.tileCache?.updateSelector(this.selector)
    this.tileCache?.updateBandNames(bandNames)

    if (this.tileCache && this.visibleTiles.length > 0) {
      const currentHash = JSON.stringify(this.selector)
      // Unified cache handles both data and texture state
      await this.tileCache.reextractTileSlices(
        this.visibleTiles,
        currentHash
      )
    }

    this.invalidate()
  }

  private updateGeometryForProjection(isGlobe: boolean) {
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
    const pyramidLevel = zoomToLevel(mapZoom, this.maxLevelIndex)
    const bounds = map.getBounds()?.toArray()
    if (!bounds) {
      return { tiles: [], pyramidLevel, mapZoom, bounds: null }
    }
    if (this.crs === 'EPSG:4326' && this.xyLimits) {
      return {
        tiles: getTilesAtZoomEquirect(pyramidLevel, bounds, this.xyLimits),
        pyramidLevel,
        mapZoom,
        bounds,
      }
    }
    return {
      tiles: getTilesAtZoom(pyramidLevel, bounds),
      pyramidLevel,
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
      }
    }

    return bounds
  }

  private async prefetchTileData(tiles: TileTuple[], selectorHash: string) {
    const fetchPromises = tiles.map((tiletuple) =>
      this.fetchTileData(tiletuple, selectorHash)
    )
    await Promise.all(fetchPromises)
  }

  private async fetchTileData(
    tileTuple: TileTuple,
    selectorHash: string
  ): Promise<Float32Array | null> {
    if (!this.tileCache) {
      const tileKey = tileToKey(tileTuple)
      this.pendingChunks.delete(tileKey)
      this.emitLoadingState()
      return null
    }

    const tileKey = tileToKey(tileTuple)

    try {
      // Unified cache handles both data fetching and WebGL resources
      const tile = await this.tileCache.fetchTile(tileTuple, selectorHash)

      this.pendingChunks.delete(tileKey)

      if (!tile) {
        this.emitLoadingState()
        return null
      }

      this.emitLoadingState()
      this.invalidate()

      return tile.data
    } catch (err) {
      this.pendingChunks.delete(tileKey)
      this.emitLoadingState()
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
