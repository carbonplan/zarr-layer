import {
  findBestParentTile,
  mercatorTileToGeoBounds,
  getOverlapping4326Tiles,
  get4326TileGeoBounds,
  tileToKey,
  latToMercatorNorm,
  lonToMercatorNorm,
  mercatorNormToLat,
  mercatorNormToLon,
  zoomToLevel,
  type MercatorBounds,
  type TileTuple,
} from './map-utils'
import type { ZarrRenderer } from './zarr-renderer'
import type { ZarrMode, RenderContext, TileId } from './zarr-mode'
import type { SingleImageParams } from './renderer-types'
import { computeTexOverride } from './webgl-utils'

/** Identity matrix for globe rendering (no additional transformation) */
const IDENTITY_MATRIX = new Float32Array([
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
])

interface MapboxTileRenderParams {
  renderer: ZarrRenderer
  mode: ZarrMode
  tileId: TileId
  context: RenderContext
}

/**
 * Creates a 4x4 transformation matrix for rendering to a specific tile region.
 *
 * @param tileX0 - Left edge of tile in normalized Mercator X [0,1]
 * @param tileY0 - Top edge of tile in normalized Mercator Y [0,1]
 * @param tileX1 - Right edge of tile in normalized Mercator X [0,1]
 * @param tileY1 - Bottom edge of tile in normalized Mercator Y [0,1]
 * @returns 4x4 column-major transformation matrix
 */
function createTileMatrix(
  tileX0: number,
  tileY0: number,
  tileX1: number,
  tileY1: number
): Float32Array {
  const x0 = Math.max(0, tileX0)
  const x1 = Math.min(1, tileX1)
  const y0 = Math.max(0, tileY0)
  const y1 = Math.min(1, tileY1)
  const width = x1 - x0
  const height = y1 - y0

  return new Float32Array([
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
}

/**
 * Renders a single image (non-tiled zarr data) to a Mapbox globe tile.
 * Handles both EPSG:4326 (equirectangular) and EPSG:3857 (Web Mercator) data.
 *
 * For EPSG:4326 data: Uses the shader's equirectangular reprojection to correctly
 * transform lat/lon linear data to Mercator space for globe rendering.
 *
 * For EPSG:3857 data: Direct texture sampling since the data is already in
 * Mercator projection.
 */
function renderSingleImageToTile(
  renderer: ZarrRenderer,
  context: RenderContext,
  tileId: TileId,
  singleImage: SingleImageParams,
  vertexArr: Float32Array,
  bounds: MercatorBounds
): void {
  const { colormapTexture, uniforms, customShaderConfig } = context

  const tilesPerSide = 2 ** tileId.z
  const tileX0 = tileId.x / tilesPerSide
  const tileX1 = (tileId.x + 1) / tilesPerSide
  const tileY0 = tileId.y / tilesPerSide
  const tileY1 = (tileId.y + 1) / tilesPerSide

  // Calculate overlap between image and tile in Mercator normalized space
  const overlapX0 = Math.max(bounds.x0, tileX0)
  const overlapX1 = Math.min(bounds.x1, tileX1)
  const overlapY0 = Math.max(bounds.y0, tileY0)
  const overlapY1 = Math.min(bounds.y1, tileY1)

  const tileMatrix = createTileMatrix(tileX0, tileY0, tileX1, tileY1)

  const shaderProgram = renderer.getProgram(
    context.shaderData,
    customShaderConfig,
    true
  )
  renderer.gl.useProgram(shaderProgram.program)
  renderer.applyCommonUniforms(
    shaderProgram,
    colormapTexture,
    uniforms,
    customShaderConfig,
    context.projectionData,
    {
      projection: { name: 'globe' },
      globeToMercatorMatrix: IDENTITY_MATRIX,
      transition: 0,
    },
    tileMatrix,
    true
  )

  const baseTexScale: [number, number] = singleImage.texScale ?? [1, 1]
  const baseTexOffset: [number, number] = singleImage.texOffset ?? [0, 0]

  // Check if this is equirectangular (EPSG:4326) data that needs reprojection
  const isEquirectangular =
    bounds.latMin !== undefined && bounds.latMax !== undefined

  if (isEquirectangular) {
    // EPSG:4326 path: Enable shader's equirectangular reprojection
    // The shader converts Mercator Y to latitude, then maps to texture V coordinate
    const cropLatNorth = mercatorNormToLat(overlapY0)
    const cropLatSouth = mercatorNormToLat(overlapY1)

    // Set scale/shift in Mercator space (required for shader's mercY calculation)
    const scaleX = (overlapX1 - overlapX0) / 2
    const scaleY = (overlapY1 - overlapY0) / 2
    const shiftX = (overlapX0 + overlapX1) / 2
    const shiftY = (overlapY0 + overlapY1) / 2

    // Enable equirectangular mode with crop region's lat bounds
    if (shaderProgram.isEquirectangularLoc) {
      renderer.gl.uniform1i(shaderProgram.isEquirectangularLoc, 1)
    }
    if (shaderProgram.latMinLoc) {
      renderer.gl.uniform1f(shaderProgram.latMinLoc, cropLatSouth)
    }
    if (shaderProgram.latMaxLoc) {
      renderer.gl.uniform1f(shaderProgram.latMaxLoc, cropLatNorth)
    }

    // Map crop region's lat range to full image texture coordinates
    // Shader outputs v in [0,1] for crop region; we map to full image space
    const fullLatRange = bounds.latMax! - bounds.latMin!
    const vNorth = (bounds.latMax! - cropLatNorth) / fullLatRange
    const vSouth = (bounds.latMax! - cropLatSouth) / fullLatRange

    // X mapping is linear in lon/Mercator space
    const imgWidth = bounds.x1 - bounds.x0
    const texScaleX = imgWidth > 0 ? (overlapX1 - overlapX0) / imgWidth : 1
    const texOffsetX = imgWidth > 0 ? (overlapX0 - bounds.x0) / imgWidth : 0

    const texOverride = computeTexOverride(
      [texScaleX, vSouth - vNorth],
      [texOffsetX, vNorth],
      baseTexScale,
      baseTexOffset
    )

    renderer.renderSingleImage(shaderProgram, [0], singleImage, vertexArr, {
      scaleX,
      scaleY,
      shiftX,
      shiftY,
      ...texOverride,
    })
  } else {
    // EPSG:3857 path: Direct Mercator mapping, no reprojection needed
    const tileSizeNorm = 1 / tilesPerSide
    const localX0 = (overlapX0 - tileX0) / tileSizeNorm
    const localX1 = (overlapX1 - tileX0) / tileSizeNorm
    const localY0 = (overlapY0 - tileY0) / tileSizeNorm
    const localY1 = (overlapY1 - tileY0) / tileSizeNorm

    // Position geometry in clip space
    const clipX0 = localX0 * 2 - 1
    const clipX1 = localX1 * 2 - 1
    const clipY0 = localY0 * 2 - 1
    const clipY1 = localY1 * 2 - 1

    const scaleX = (clipX1 - clipX0) / 2
    const scaleY = (clipY1 - clipY0) / 2
    const shiftX = (clipX0 + clipX1) / 2
    const shiftY = (clipY0 + clipY1) / 2

    // Texture crop mapping
    const imgWidth = bounds.x1 - bounds.x0
    const imgHeight = bounds.y1 - bounds.y0
    const texScaleX = imgWidth > 0 ? (overlapX1 - overlapX0) / imgWidth : 1
    const texScaleY = imgHeight > 0 ? (overlapY1 - overlapY0) / imgHeight : 1
    const texOffsetX = imgWidth > 0 ? (overlapX0 - bounds.x0) / imgWidth : 0
    const texOffsetY = imgHeight > 0 ? (overlapY0 - bounds.y0) / imgHeight : 0

    const texOverride = computeTexOverride(
      [texScaleX, texScaleY],
      [texOffsetX, texOffsetY],
      baseTexScale,
      baseTexOffset
    )

    renderer.renderSingleImage(shaderProgram, [0], singleImage, vertexArr, {
      scaleX,
      scaleY,
      shiftX,
      shiftY,
      ...texOverride,
    })
  }
}

/**
 * Renders Zarr data to a Mapbox globe tile.
 *
 * This is the main entry point for globe tile rendering. It handles:
 * - Single image (non-tiled) data: Renders the portion of the image that
 *   intersects the requested tile
 * - Tiled pyramid data: Finds and renders the appropriate Zarr tiles that
 *   overlap the Mapbox tile
 *
 * For EPSG:4326 data, performs coordinate reprojection to correctly display
 * equirectangular data on the Mercator-based globe tiles.
 *
 * @param params - Render parameters including renderer, mode, tile ID, and context
 * @returns true if more data is needed (tile data not yet loaded), false otherwise
 */
export function renderMapboxTile({
  renderer,
  mode,
  tileId,
  context,
}: MapboxTileRenderParams): boolean {
  const { colormapTexture, uniforms, customShaderConfig } = context

  // Handle single image (non-tiled) data
  const singleImageState = mode.getSingleImageState?.()
  if (!mode.isMultiscale && singleImageState) {
    const { singleImage, vertexArr } = singleImageState
    const bounds = singleImage.bounds
    if (!bounds) return false

    const tilesPerSide = 2 ** tileId.z
    const tileX0 = tileId.x / tilesPerSide
    const tileX1 = (tileId.x + 1) / tilesPerSide
    const tileY0 = tileId.y / tilesPerSide
    const tileY1 = (tileId.y + 1) / tilesPerSide

    // Check if image intersects this tile
    const intersects =
      bounds.x0 < tileX1 &&
      bounds.x1 > tileX0 &&
      bounds.y0 < tileY1 &&
      bounds.y1 > tileY0

    if (!intersects) return false

    renderSingleImageToTile(
      renderer,
      context,
      tileId,
      singleImage,
      vertexArr,
      bounds
    )
    return false
  }

  // Handle tiled (pyramid) data
  const tiledState = mode.getTiledState?.()
  if (!tiledState?.tileCache) {
    return true
  }

  const { tileCache, vertexArr, pixCoordArr, tileSize, tileBounds } = tiledState

  const tilesPerSide = 2 ** tileId.z
  const mapboxMercX0 = tileId.x / tilesPerSide
  const mapboxMercX1 = (tileId.x + 1) / tilesPerSide
  const mapboxMercY0 = tileId.y / tilesPerSide
  const mapboxMercY1 = (tileId.y + 1) / tilesPerSide

  const tileMatrix = createTileMatrix(
    mapboxMercX0,
    mapboxMercY0,
    mapboxMercX1,
    mapboxMercY1
  )

  const crs = mode.getCRS()
  const xyLimits = mode.getXYLimits()
  const maxZoom = mode.getMaxZoom()

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

    const shaderProgram = renderer.getProgram(
      context.shaderData,
      customShaderConfig,
      true
    )
    renderer.gl.useProgram(shaderProgram.program)

    let anyTileRendered = false
    let anyMissing = false
    for (const zarrTile of overlappingZarrTiles) {
      const zarrTileKey = tileToKey(zarrTile)
      let tileData = tileCache.get(zarrTileKey)
      let renderTileKey = zarrTileKey
      let renderTileTuple: TileTuple = zarrTile
      if (!tileData?.data) {
        anyMissing = true
        const parent = findBestParentTile(
          tileCache,
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

      renderer.applyCommonUniforms(
        shaderProgram,
        colormapTexture,
        uniforms,
        customShaderConfig,
        context.projectionData,
        {
          projection: { name: 'globe' },
          globeToMercatorMatrix: IDENTITY_MATRIX,
          transition: 0,
        },
        tileMatrix,
        true
      )

      renderer.renderTiles(
        shaderProgram,
        [renderTileTuple],
        [0],
        tileCache,
        tileSize,
        vertexArr,
        pixCoordArr,
        tileBoundsForRender,
        customShaderConfig,
        true,
        {
          [renderTileKey]: {
            texScale: [texScaleX, texScaleY],
            texOffset: [texOffsetX, texOffsetY],
          },
        }
      )

      anyTileRendered = true
    }

    return anyMissing || !anyTileRendered
  }

  const tileTuple: TileTuple = [tileId.z, tileId.x, tileId.y]
  const tileKey = tileTuple.join(',')

  const boundsForTile = tileBounds?.[tileKey]
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

  const shaderProgram = renderer.getProgram(
    context.shaderData,
    customShaderConfig,
    true
  )
  renderer.gl.useProgram(shaderProgram.program)

  renderer.applyCommonUniforms(
    shaderProgram,
    colormapTexture,
    uniforms,
    customShaderConfig,
    context.projectionData,
    {
      projection: { name: 'globe' },
      globeToMercatorMatrix: IDENTITY_MATRIX,
      transition: 0,
    },
    tileMatrix,
    true
  )

  renderer.renderTiles(
    shaderProgram,
    [tileTuple],
    [0],
    tileCache,
    tileSize,
    vertexArr,
    pixCoordArr,
    tileBoundsOverride,
    customShaderConfig,
    true
  )

  const tileHasData = tileCache.get(tileKey)?.data
  return !tileHasData
}
