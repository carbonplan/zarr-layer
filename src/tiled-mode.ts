import type {
  ZarrMode,
  RenderContext,
  TileId,
  TiledRenderState,
} from './zarr-mode'
import type {
  QuerySelector,
  QueryDataGeometry,
  QueryDataResult,
} from './query/types'
import { queryRegionTiled } from './query/region-query'
import type {
  LoadingStateCallback,
  MapLike,
  SelectorMap,
  XYLimitsProps,
  CRS,
  ZarrSelectorsProps,
} from './types'
import { ZarrStore } from './zarr-store'
import { TileRenderCache } from './zarr-tile-cache'
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
import { getBands, toSelectorProps } from './zarr-utils'
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
  private tileCache: TileRenderCache | null = null
  private tilesManager: Tiles | null = null
  private vertexArr: Float32Array = new Float32Array()
  private pixCoordArr: Float32Array = new Float32Array()
  private currentSubdivisions: number = 0
  private maxZoom: number = 4
  private minRenderZoom: number = 3
  private tileSize: number = DEFAULT_TILE_SIZE
  private variable: string
  private selector: Record<
    string,
    number | number[] | string | string[] | ZarrSelectorsProps
  >
  private invalidate: () => void
  private zarrStore: ZarrStore
  private selectors: SelectorMap = {}
  private visibleTiles: TileTuple[] = []
  private crs: CRS = 'EPSG:4326'
  private xyLimits: XYLimitsProps | null = null
  private tileBounds: Record<string, MercatorBounds> = {}
  private loadingCallback: LoadingStateCallback | undefined
  private pendingChunks: Set<string> = new Set()
  private metadataLoading: boolean = false
  private currentLevel: number | null = null

  constructor(
    store: ZarrStore,
    variable: string,
    selector: Record<
      string,
      number | number[] | string | string[] | ZarrSelectorsProps
    >,
    minRenderZoom: number,
    invalidate: () => void
  ) {
    this.zarrStore = store
    this.variable = variable
    this.selector = selector
    this.minRenderZoom = minRenderZoom
    this.invalidate = invalidate

    for (const [dimName, value] of Object.entries(selector)) {
      this.selectors[dimName] = toSelectorProps(value)
    }
  }

  async initialize(): Promise<void> {
    this.metadataLoading = true
    this.emitLoadingState()

    try {
      const desc = this.zarrStore.describe()
      this.maxZoom = desc.levels.length - 1
      this.tileSize = desc.tileSize || DEFAULT_TILE_SIZE
      this.crs = desc.crs
      this.xyLimits = desc.xyLimits

      const bandNames = getBands(this.variable, this.selector)

      this.tilesManager = new Tiles({
        store: this.zarrStore,
        selectors: this.selectors,
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
      this.tileCache = new TileRenderCache(gl, MAX_CACHED_TILES)
    }

    const projection = map.getProjection ? map.getProjection() : null
    const isGlobe = projection?.type === 'globe' || projection?.name === 'globe'
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

    const isMapboxTile = !!context.mapboxGlobe
    const shaderProgram = renderer.getProgram(
      context.shaderData,
      context.customShaderConfig,
      isMapboxTile
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

  getMaxZoom(): number {
    return this.maxZoom
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

  async setSelector(
    selector: Record<
      string,
      number | number[] | string | string[] | ZarrSelectorsProps
    >
  ): Promise<void> {
    this.selector = selector
    for (const [dimName, value] of Object.entries(selector)) {
      this.selectors[dimName] = toSelectorProps(value)
    }
    const bandNames = getBands(this.variable, selector)

    this.tilesManager?.updateSelector(this.selectors)
    this.tilesManager?.updateBandNames(bandNames)

    if (this.tilesManager && this.visibleTiles.length > 0) {
      const currentHash = JSON.stringify(this.selector)
      await this.tilesManager.reextractTileSlices(
        this.visibleTiles,
        currentHash
      )

      for (const tileTuple of this.visibleTiles) {
        const tileKey = tileToKey(tileTuple)
        const tile = this.tileCache?.get(tileKey)
        const cache = this.tilesManager.getTile(tileTuple)
        if (!tile || !cache) continue
        tile.data = cache.data
        tile.textureUploaded = false
        tile.bandTexturesUploaded.clear()
        tile.channels = cache.channels
        tile.selectorHash = cache.selectorHash
        tile.bandData = cache.bandData
      }
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
    if (mapZoom < this.minRenderZoom) {
      return { tiles: [], pyramidLevel: null, mapZoom, bounds: null }
    }
    const pyramidLevel = zoomToLevel(mapZoom, this.maxZoom)
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
    if (!this.tilesManager || !this.tileCache) {
      const tileKey = tileToKey(tileTuple)
      this.pendingChunks.delete(tileKey)
      this.emitLoadingState()
      return null
    }

    const tileKey = tileToKey(tileTuple)
    const tile = this.tileCache.upsert(tileKey)

    try {
      const cache = await this.tilesManager.fetchTile(tileTuple, selectorHash)

      this.pendingChunks.delete(tileKey)

      if (!cache) {
        this.emitLoadingState()
        return null
      }

      tile.data = cache.data
      tile.textureUploaded = false
      tile.bandTexturesUploaded.clear()
      tile.selectorHash = cache.selectorHash
      tile.channels = cache.channels
      tile.bandData = cache.bandData

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
    geometry: QueryDataGeometry,
    selector?: QuerySelector
  ): Promise<QueryDataResult> {
    if (!this.tilesManager || !this.xyLimits) {
      return {
        [this.variable]: [],
        dimensions: [],
        coordinates: { lat: [], lon: [] },
      }
    }

    // Use provided selector or fall back to layer's selector
    const querySelector = selector || (this.selector as QuerySelector)
    const level = this.currentLevel ?? this.maxZoom

    return queryRegionTiled(
      this.variable,
      geometry,
      querySelector,
      this.zarrStore,
      this.crs,
      this.xyLimits,
      level,
      this.tileSize
    )
  }
}
