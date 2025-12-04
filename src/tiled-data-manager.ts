import { DataManager, RenderData } from './data-manager'
import { ZarrStore } from './zarr-store'
import { TileRenderCache } from './zarr-tile-cache'
import { Tiles } from './tiles'
import {
  getTilesAtZoom,
  tileToKey,
  TileTuple,
  zoomToLevel,
} from './maplibre-utils'
import { getBands } from './zarr-utils'

const DEFAULT_TILE_SIZE = 128
const MAX_CACHED_TILES = 64
const TILE_SUBDIVISIONS = 32

export class TiledDataManager implements DataManager {
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
  private selector: Record<string, number | number[]>
  private invalidate: () => void
  private zarrStore: ZarrStore
  private selectors: Record<string, any> = {}
  private visibleTiles: TileTuple[] = []

  constructor(
    store: ZarrStore,
    variable: string,
    selector: Record<string, number | number[]>,
    minRenderZoom: number,
    invalidate: () => void
  ) {
    this.zarrStore = store
    this.variable = variable
    this.selector = selector
    this.minRenderZoom = minRenderZoom
    this.invalidate = invalidate

    // Initialize selectors
    for (const [dimName, value] of Object.entries(selector)) {
      this.selectors[dimName] = { selected: value, type: 'index' }
    }
  }

  async initialize(): Promise<void> {
    const desc = this.zarrStore.describe()
    this.maxZoom = desc.levels.length - 1
    this.tileSize = desc.tileSize || DEFAULT_TILE_SIZE

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

    // Initialize geometry
    this.updateGeometryForProjection(false)
  }

  update(map: any, gl: WebGL2RenderingContext): void {
    if (!this.tileCache) {
      this.tileCache = new TileRenderCache(gl, MAX_CACHED_TILES)
    }

    const projection = map.getProjection ? map.getProjection() : null
    const isGlobe = projection?.type === 'globe' || projection?.name === 'globe'
    this.updateGeometryForProjection(isGlobe)

    this.visibleTiles = this.getVisibleTiles(map)
    this.prefetchTileData(this.visibleTiles)
  }

  onProjectionChange(isGlobe: boolean) {
    this.updateGeometryForProjection(isGlobe)
  }

  getRenderData(): RenderData {
    return {
      isMultiscale: true,
      tileCache: this.tileCache,
      visibleTiles: this.visibleTiles,
      tileSize: this.tileSize,
      vertexArr: this.vertexArr,
      pixCoordArr: this.pixCoordArr,
      singleImage: undefined,
    }
  }

  dispose(gl: WebGL2RenderingContext): void {
    this.tileCache?.clear()
    this.tileCache = null
  }

  async setSelector(
    selector: Record<string, number | number[]>
  ): Promise<void> {
    this.selector = selector
    for (const [dimName, value] of Object.entries(selector)) {
      this.selectors[dimName] = { selected: value, type: 'index' }
    }
    const bandNames = getBands(this.variable, selector)

    this.tilesManager?.updateSelector(this.selectors)
    this.tilesManager?.updateBandNames(bandNames)

    // Optimistically update existing visible tiles if possible
    if (this.tilesManager && this.visibleTiles.length > 0) {
      const currentHash = JSON.stringify(this.selector)
      await this.tilesManager.reextractTileSlices(
        this.visibleTiles,
        currentHash
      )

      // Update cache
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

  // Helpers
  private updateGeometryForProjection(isGlobe: boolean) {
    const targetSubdivisions = isGlobe ? TILE_SUBDIVISIONS : 1
    if (this.currentSubdivisions === targetSubdivisions) return

    const subdivided = TiledDataManager.createSubdividedQuad(targetSubdivisions)
    this.vertexArr = subdivided.vertexArr
    this.pixCoordArr = subdivided.texCoordArr
    this.currentSubdivisions = targetSubdivisions
    this.tileCache?.markGeometryDirty()
  }

  private static createSubdividedQuad(subdivisions: number): {
    vertexArr: Float32Array
    texCoordArr: Float32Array
  } {
    const vertices: number[] = []
    const texCoords: number[] = []
    const step = 2 / subdivisions
    const texStep = 1 / subdivisions

    const pushVertex = (col: number, row: number) => {
      const x = -1 + col * step
      const y = 1 - row * step
      const u = col * texStep
      const v = row * texStep
      vertices.push(x, y)
      texCoords.push(u, v)
    }

    for (let row = 0; row < subdivisions; row++) {
      for (let col = 0; col <= subdivisions; col++) {
        pushVertex(col, row)
        pushVertex(col, row + 1)
      }
      if (row < subdivisions - 1) {
        pushVertex(subdivisions, row + 1)
        pushVertex(0, row + 1)
      }
    }

    return {
      vertexArr: new Float32Array(vertices),
      texCoordArr: new Float32Array(texCoords),
    }
  }

  private getVisibleTiles(map: any): TileTuple[] {
    const mapZoom = map.getZoom()
    if (mapZoom < this.minRenderZoom) {
      return []
    }
    const pyramidLevel = zoomToLevel(mapZoom, this.maxZoom)
    const bounds = map.getBounds()?.toArray()
    if (!bounds) {
      return []
    }
    return getTilesAtZoom(pyramidLevel, bounds)
  }

  private async prefetchTileData(tiles: TileTuple[]) {
    const fetchPromises = tiles.map((tiletuple) =>
      this.fetchTileData(tiletuple)
    )
    await Promise.all(fetchPromises)
  }

  private async fetchTileData(
    tileTuple: TileTuple
  ): Promise<Float32Array | null> {
    if (!this.tilesManager || !this.tileCache) return null

    const tileKey = tileToKey(tileTuple)
    const tile = this.tileCache.upsert(tileKey)
    const currentHash = JSON.stringify(this.selector)

    if (tile.data && tile.selectorHash === currentHash) {
      return tile.data
    }

    const cache = await this.tilesManager.fetchTile(tileTuple, currentHash)
    if (!cache) {
      return null
    }

    tile.data = cache.data
    tile.textureUploaded = false
    tile.bandTexturesUploaded.clear()
    tile.selectorHash = cache.selectorHash
    tile.channels = cache.channels
    tile.bandData = cache.bandData
    this.invalidate()

    return tile.data
  }
}
