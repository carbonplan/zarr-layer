import {
  findBestParentTile,
  mercatorTileToGeoBounds,
  getOverlapping4326Tiles,
  get4326TileGeoBounds,
  tileToKey,
  latToMercatorNorm,
  lonToMercatorNorm,
  mercatorNormToLon,
  zoomToLevel,
  type TileTuple,
} from './map-utils'
import type { DataManager, RenderData } from './data-manager'
import type { ZarrRenderer } from './zarr-renderer'
import type { CustomShaderConfig, RendererUniforms } from './renderer-types'

const IDENTITY_MATRIX = new Float32Array([
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
])

interface MapboxTileRenderParams {
  renderer: ZarrRenderer
  renderData: RenderData
  tileId: { z: number; x: number; y: number }
  colormapTexture: WebGLTexture
  uniforms: RendererUniforms
  tileSize: number
  customShaderConfig?: CustomShaderConfig
  dataManager: DataManager
}

export function renderMapboxTile({
  renderer,
  renderData,
  tileId,
  colormapTexture,
  uniforms,
  tileSize,
  customShaderConfig,
  dataManager,
}: MapboxTileRenderParams): boolean {
  // Single image path: render the portion of the image that overlaps with this tile
  if (!renderData.isMultiscale && renderData.singleImage) {
    const singleImage = renderData.singleImage
    const bounds = singleImage.bounds
    if (!bounds) return false

    const tileSizeNorm = 1 / 2 ** tileId.z
    const tileX0 = tileId.x * tileSizeNorm
    const tileX1 = (tileId.x + 1) * tileSizeNorm
    const tileY0 = tileId.y * tileSizeNorm
    const tileY1 = (tileId.y + 1) * tileSizeNorm

    const intersects =
      bounds.x0 < tileX1 &&
      bounds.x1 > tileX0 &&
      bounds.y0 < tileY1 &&
      bounds.y1 > tileY0

    if (!intersects) return false

    const overlapX0 = Math.max(bounds.x0, tileX0)
    const overlapX1 = Math.min(bounds.x1, tileX1)
    const overlapY0 = Math.max(bounds.y0, tileY0)
    const overlapY1 = Math.min(bounds.y1, tileY1)

    const localX0 = (overlapX0 - tileX0) / tileSizeNorm
    const localX1 = (overlapX1 - tileX0) / tileSizeNorm
    const localY0 = (overlapY0 - tileY0) / tileSizeNorm
    const localY1 = (overlapY1 - tileY0) / tileSizeNorm

    const clipX0 = localX0 * 2 - 1
    const clipX1 = localX1 * 2 - 1
    const clipY0 = localY0 * 2 - 1
    const clipY1 = localY1 * 2 - 1

    const scaleX = (clipX1 - clipX0) / 2
    const scaleY = (clipY1 - clipY0) / 2
    const shiftX = (clipX0 + clipX1) / 2
    const shiftY = (clipY0 + clipY1) / 2

    const imgWidth = bounds.x1 - bounds.x0
    const imgHeight = bounds.y1 - bounds.y0
    const texScaleX = imgWidth > 0 ? (overlapX1 - overlapX0) / imgWidth : 1
    const texScaleY = imgHeight > 0 ? (overlapY1 - overlapY0) / imgHeight : 1
    const texOffsetX = imgWidth > 0 ? (overlapX0 - bounds.x0) / imgWidth : 0
    const texOffsetY = imgHeight > 0 ? (overlapY0 - bounds.y0) / imgHeight : 0

    renderer.render({
      matrix: IDENTITY_MATRIX,
      colormapTexture,
      uniforms,
      worldOffsets: [0],
      isMultiscale: false,
      visibleTiles: [],
      tileSize,
      vertexArr: renderData.vertexArr || new Float32Array(),
      pixCoordArr: renderData.pixCoordArr || new Float32Array(),
      singleImage: singleImage,
      customShaderConfig,
      mapboxGlobe: {
        projection: { name: 'globe' },
        globeToMercatorMatrix: IDENTITY_MATRIX,
        transition: 0,
      },
      mode: {
        type: 'mapboxTile',
        tileOverride: {
          scaleX,
          scaleY,
          shiftX,
          shiftY,
          texScale: [texScaleX, texScaleY],
          texOffset: [texOffsetX, texOffsetY],
        },
      },
    })

    return false
  }

  if (!renderData.tileCache) {
    return true
  }

  const tilesPerSide = 2 ** tileId.z
  const mapboxMercX0 = tileId.x / tilesPerSide
  const mapboxMercX1 = (tileId.x + 1) / tilesPerSide
  const mapboxMercY0 = tileId.y / tilesPerSide
  const mapboxMercY1 = (tileId.y + 1) / tilesPerSide
  const EPS = 1e-7
  const x0 = Math.max(0, mapboxMercX0 + EPS)
  const x1 = Math.min(1, mapboxMercX1 - EPS)
  const y0 = Math.max(0, mapboxMercY0 + EPS)
  const y1 = Math.min(1, mapboxMercY1 - EPS)
  const width = x1 - x0
  const height = y1 - y0

  const tileMatrix = new Float32Array([
    2 / width,
    0,
    0,
    0,
    0,
    2 / height,
    0,
    0,
    0,
    0,
    1,
    0,
    -(x0 + x1) / width,
    -(y0 + y1) / height,
    0,
    1,
  ])

  const crs = dataManager.getCRS()
  const xyLimits = dataManager.getXYLimits()
  const maxZoom = dataManager.getMaxZoom()

  if (crs === 'EPSG:4326' && xyLimits) {
    const mapboxGeoBounds = mercatorTileToGeoBounds(
      tileId.z,
      tileId.x,
      tileId.y
    )
    const pyramidLevel = zoomToLevel(tileId.z, maxZoom)
    const overlappingZarrTiles = getOverlapping4326Tiles(
      mapboxGeoBounds,
      xyLimits,
      pyramidLevel
    )

    if (overlappingZarrTiles.length === 0) {
      return false
    }

    let anyTileRendered = false
    let anyMissing = false
    for (const zarrTile of overlappingZarrTiles) {
      const zarrTileKey = tileToKey(zarrTile)
      let tileData = renderData.tileCache.get(zarrTileKey)
      let renderTileKey = zarrTileKey
      let renderTileTuple: TileTuple = zarrTile
      if (!tileData?.data) {
        anyMissing = true
        const parent = findBestParentTile(
          renderData.tileCache,
          zarrTile[0],
          zarrTile[1],
          zarrTile[2]
        )
        if (!parent) continue
        tileData = parent.tile
        renderTileTuple = [parent.ancestorZ, parent.ancestorX, parent.ancestorY]
        renderTileKey = tileToKey(renderTileTuple)
      }

      const [z, tx, ty] = renderTileTuple
      const zarrGeoBounds = get4326TileGeoBounds(z, tx, ty, xyLimits)

      const zarrMercX0 = lonToMercatorNorm(zarrGeoBounds.west)
      const zarrMercX1 = lonToMercatorNorm(zarrGeoBounds.east)
      const zarrMercY0 = latToMercatorNorm(zarrGeoBounds.north)
      const zarrMercY1 = latToMercatorNorm(zarrGeoBounds.south)

      const overlapX0 = Math.max(zarrMercX0, mapboxMercX0)
      const overlapX1 = Math.min(zarrMercX1, mapboxMercX1)
      const overlapY0 = Math.max(zarrMercY0, mapboxMercY0)
      const overlapY1 = Math.min(zarrMercY1, mapboxMercY1)

      if (overlapX1 <= overlapX0 || overlapY1 <= overlapY0) continue

      const zarrLonWidth = zarrGeoBounds.east - zarrGeoBounds.west
      const overlapWest = mercatorNormToLon(overlapX0)
      const overlapEast = mercatorNormToLon(overlapX1)
      const texScaleX =
        zarrLonWidth > 0 ? (overlapEast - overlapWest) / zarrLonWidth : 1
      const texOffsetX =
        zarrLonWidth > 0 ? (overlapWest - zarrGeoBounds.west) / zarrLonWidth : 0
      const texScaleY = 1.0
      const texOffsetY = 0.0

      const tileBoundsForRender = {
        [renderTileKey]: {
          x0: overlapX0,
          y0: overlapY0,
          x1: overlapX1,
          y1: overlapY1,
          latMin: zarrGeoBounds.south,
          latMax: zarrGeoBounds.north,
        },
      }

      renderer.render({
        matrix: tileMatrix,
        colormapTexture,
        uniforms,
        worldOffsets: [0],
        isMultiscale: true,
        visibleTiles: [renderTileTuple],
        tileCache: renderData.tileCache,
        tileSize,
        vertexArr: renderData.vertexArr || new Float32Array(),
        pixCoordArr: renderData.pixCoordArr || new Float32Array(),
        tileBounds: tileBoundsForRender,
        singleImage: renderData.singleImage,
        customShaderConfig,
        mapboxGlobe: {
          projection: { name: 'globe' },
          globeToMercatorMatrix: IDENTITY_MATRIX,
          transition: 0,
        },
        mode: {
          type: 'mapboxTile',
          tileTexOverrides: {
            [renderTileKey]: {
              texScale: [texScaleX, texScaleY],
              texOffset: [texOffsetX, texOffsetY],
            },
          },
        },
      })

      anyTileRendered = true
    }

    return anyMissing || !anyTileRendered
  }

  const tileTuple: TileTuple = [tileId.z, tileId.x, tileId.y]
  const tileKey = tileTuple.join(',')

  const boundsForTile = renderData.tileBounds?.[tileKey]
  const tileBoundsOverride = {
    [tileKey]: {
      x0: mapboxMercX0,
      y0: mapboxMercY0,
      x1: mapboxMercX1,
      y1: mapboxMercY1,
      latMin: boundsForTile?.latMin,
      latMax: boundsForTile?.latMax,
    },
  }

  renderer.render({
    matrix: tileMatrix,
    colormapTexture,
    uniforms,
    worldOffsets: [0],
    isMultiscale: renderData.isMultiscale,
    visibleTiles: [tileTuple],
    tileCache: renderData.tileCache,
    tileSize,
    vertexArr: renderData.vertexArr || new Float32Array(),
    pixCoordArr: renderData.pixCoordArr || new Float32Array(),
    tileBounds: tileBoundsOverride,
    singleImage: renderData.singleImage,
    customShaderConfig,
    mapboxGlobe: {
      projection: { name: 'globe' },
      globeToMercatorMatrix: IDENTITY_MATRIX,
      transition: 0,
    },
    mode: { type: 'mapboxTile' },
  })

  const tileHasData = renderData.tileCache.get(tileKey)?.data
  return !tileHasData
}
