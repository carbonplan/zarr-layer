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
  parseLevelZoom,
  zoomToLevel,
  type MercatorBounds,
  type TileTuple,
} from './map-utils'
import type { ZarrRenderer, ShaderProgram } from './zarr-renderer'
import type {
  ZarrMode,
  RenderContext,
  TileId,
  RegionRenderState,
} from './zarr-mode'
import { computeTexOverride, configureDataTexture } from './webgl-utils'
import { bindBandTextures, setupBandTextureUniforms } from './render-helpers'

/**
 * Cache for linear texture coordinate buffers used in globe rendering.
 * Globe rendering needs linear (non-warped) texture coords - the shader handles reprojection.
 * We compute these from vertex positions and cache the buffer per region.
 */
const linearBufferCache = new WeakMap<RegionRenderState, WebGLBuffer>()

/**
 * Get or create a linear texture coordinate buffer for globe rendering.
 * Computes linear coords from vertex positions: u = (x+1)/2, v = (1-y)/2
 */
function getLinearPixCoordBuffer(
  gl: WebGL2RenderingContext,
  region: RegionRenderState
): WebGLBuffer {
  let buffer = linearBufferCache.get(region)
  if (buffer) return buffer

  // Compute linear tex coords from vertex positions
  const vertexArr = region.vertexArr
  const linearCoords = new Float32Array(vertexArr.length)
  for (let i = 0; i < vertexArr.length; i += 2) {
    const x = vertexArr[i]
    const y = vertexArr[i + 1]
    linearCoords[i] = (x + 1) / 2 // u
    linearCoords[i + 1] = (1 - y) / 2 // v
  }

  buffer = gl.createBuffer()!
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
  gl.bufferData(gl.ARRAY_BUFFER, linearCoords, gl.STATIC_DRAW)

  linearBufferCache.set(region, buffer)
  return buffer
}

/** Identity matrix for globe rendering (no additional transformation) */
const IDENTITY_MATRIX = new Float32Array([
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
])

/**
 * Params for direct WebGL drawing of a region/image to a globe tile.
 */
interface DrawRegionParams {
  gl: WebGL2RenderingContext
  shaderProgram: ShaderProgram
  vertexBuffer: WebGLBuffer
  pixCoordBuffer: WebGLBuffer
  texture: WebGLTexture
  vertexArr: Float32Array
  scaleX: number
  scaleY: number
  shiftX: number
  shiftY: number
  texScale: [number, number]
  texOffset: [number, number]
  /** If true, skip base texture binding (band textures already bound) */
  skipBaseTexture?: boolean
}

/**
 * Draw a region/image directly with WebGL.
 * Used for globe tile rendering where buffers and textures are pre-uploaded.
 */
function drawRegion(params: DrawRegionParams): void {
  const {
    gl,
    shaderProgram,
    vertexBuffer,
    pixCoordBuffer,
    texture,
    vertexArr,
    scaleX,
    scaleY,
    shiftX,
    shiftY,
    texScale,
    texOffset,
    skipBaseTexture,
  } = params

  // Set scale/shift uniforms
  gl.uniform1f(shaderProgram.scaleLoc, 0)
  gl.uniform1f(shaderProgram.scaleXLoc, scaleX)
  gl.uniform1f(shaderProgram.scaleYLoc, scaleY)
  gl.uniform1f(shaderProgram.shiftXLoc, shiftX)
  gl.uniform1f(shaderProgram.shiftYLoc, shiftY)
  gl.uniform2f(shaderProgram.texScaleLoc, texScale[0], texScale[1])
  gl.uniform2f(shaderProgram.texOffsetLoc, texOffset[0], texOffset[1])

  // Bind vertex buffer
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer)
  gl.enableVertexAttribArray(shaderProgram.vertexLoc)
  gl.vertexAttribPointer(shaderProgram.vertexLoc, 2, gl.FLOAT, false, 0, 0)

  // Bind texture coordinate buffer
  gl.bindBuffer(gl.ARRAY_BUFFER, pixCoordBuffer)
  gl.enableVertexAttribArray(shaderProgram.pixCoordLoc)
  gl.vertexAttribPointer(shaderProgram.pixCoordLoc, 2, gl.FLOAT, false, 0, 0)

  // Bind base texture (unless using custom shader with band textures)
  if (!skipBaseTexture) {
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, texture)
    if (shaderProgram.texLoc !== null) {
      gl.uniform1i(shaderProgram.texLoc, 0)
    }
    configureDataTexture(gl)
  }

  // Draw (worldOffset=0 for globe tiles)
  gl.uniform1f(shaderProgram.worldXOffsetLoc, 0)
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, vertexArr.length / 2)
}

interface MapboxTileRenderParams {
  renderer: ZarrRenderer
  mode: ZarrMode
  tileId: TileId
  context: RenderContext
  /** Regions to render (for untiled/region-based modes) */
  regions?: RegionRenderState[]
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

/** Params for rendering a region to a globe tile */
interface RegionTileParams {
  vertexBuffer: WebGLBuffer
  pixCoordBuffer: WebGLBuffer
  texture: WebGLTexture
  vertexArr: Float32Array
  bounds: MercatorBounds
  texScale?: [number, number]
  texOffset?: [number, number]
  latIsAscending?: boolean
  width: number
  height: number
  bandData?: Map<string, Float32Array>
  bandTextures?: Map<string, WebGLTexture>
  bandTexturesUploaded?: Set<string>
  bandTexturesConfigured?: Set<string>
}

/**
 * Renders a region to a Mapbox globe tile.
 * Handles both EPSG:4326 (equirectangular) and EPSG:3857 (Web Mercator) data.
 *
 * For EPSG:4326 data: Uses the shader's equirectangular reprojection to correctly
 * transform lat/lon linear data to Mercator space for globe rendering.
 *
 * For EPSG:3857 data: Direct texture sampling since the data is already in
 * Mercator projection.
 */
function renderRegionToTile(
  renderer: ZarrRenderer,
  context: RenderContext,
  tileId: TileId,
  region: RegionTileParams
): boolean {
  const { colormapTexture, uniforms, customShaderConfig } = context
  const { bounds, vertexBuffer, pixCoordBuffer, texture, vertexArr } = region

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

  const baseTexScale: [number, number] = region.texScale ?? [1, 1]
  const baseTexOffset: [number, number] = region.texOffset ?? [0, 0]
  const gl = renderer.gl

  // Bind band textures for custom shaders with multiple bands
  let skipBaseTexture = false
  if (shaderProgram.useCustomShader && customShaderConfig) {
    setupBandTextureUniforms(gl, shaderProgram, customShaderConfig)
    if (
      region.bandData &&
      region.bandTextures &&
      region.bandTexturesUploaded &&
      region.bandTexturesConfigured
    ) {
      const bandsBound = bindBandTextures(gl, {
        bandData: region.bandData,
        bandTextures: region.bandTextures,
        bandTexturesUploaded: region.bandTexturesUploaded,
        bandTexturesConfigured: region.bandTexturesConfigured,
        customShaderConfig,
        width: region.width,
        height: region.height,
      })
      if (!bandsBound) {
        return false // Missing band data
      }
      skipBaseTexture = true
    }
  }

  // Check if this is equirectangular (EPSG:4326) data that needs reprojection
  const isEquirectangular =
    bounds.latMin !== undefined && bounds.latMax !== undefined

  if (isEquirectangular) {
    // EPSG:4326 path: Enable shader's equirectangular reprojection
    const cropLatNorth = mercatorNormToLat(overlapY0)
    const cropLatSouth = mercatorNormToLat(overlapY1)

    const scaleX = (overlapX1 - overlapX0) / 2
    const scaleY = (overlapY1 - overlapY0) / 2
    const shiftX = (overlapX0 + overlapX1) / 2
    const shiftY = (overlapY0 + overlapY1) / 2

    // Enable equirectangular mode
    if (shaderProgram.isEquirectangularLoc) {
      gl.uniform1i(shaderProgram.isEquirectangularLoc, 1)
    }
    if (shaderProgram.latMinLoc) {
      gl.uniform1f(shaderProgram.latMinLoc, cropLatSouth)
    }
    if (shaderProgram.latMaxLoc) {
      gl.uniform1f(shaderProgram.latMaxLoc, cropLatNorth)
    }

    // Map crop region's lat range to texture coordinates
    // Default assumes ascending (row 0 = south), only flip if explicitly false
    const fullLatRange = bounds.latMax! - bounds.latMin!
    let vNorth: number
    let vSouth: number
    if (region.latIsAscending === false) {
      // Data goes north to south: V=0 at latMax (north), V=1 at latMin (south)
      vNorth = (bounds.latMax! - cropLatNorth) / fullLatRange
      vSouth = (bounds.latMax! - cropLatSouth) / fullLatRange
    } else {
      // Data goes south to north (default): V=0 at latMin (south), V=1 at latMax (north)
      vNorth = (cropLatNorth - bounds.latMin!) / fullLatRange
      vSouth = (cropLatSouth - bounds.latMin!) / fullLatRange
    }

    const imgWidth = bounds.x1 - bounds.x0
    const texScaleX = imgWidth > 0 ? (overlapX1 - overlapX0) / imgWidth : 1
    const texOffsetX = imgWidth > 0 ? (overlapX0 - bounds.x0) / imgWidth : 0

    const texOverride = computeTexOverride(
      [texScaleX, vSouth - vNorth],
      [texOffsetX, vNorth],
      baseTexScale,
      baseTexOffset
    )

    drawRegion({
      gl,
      shaderProgram,
      vertexBuffer,
      pixCoordBuffer,
      texture,
      vertexArr,
      scaleX,
      scaleY,
      shiftX,
      shiftY,
      ...texOverride,
      skipBaseTexture,
    })
  } else {
    // EPSG:3857 path: Direct Mercator mapping
    if (shaderProgram.isEquirectangularLoc) {
      gl.uniform1i(shaderProgram.isEquirectangularLoc, 0)
    }

    const scaleX = (overlapX1 - overlapX0) / 2
    const scaleY = (overlapY1 - overlapY0) / 2
    const shiftX = (overlapX0 + overlapX1) / 2
    const shiftY = (overlapY0 + overlapY1) / 2

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

    drawRegion({
      gl,
      shaderProgram,
      vertexBuffer,
      pixCoordBuffer,
      texture,
      vertexArr,
      scaleX,
      scaleY,
      shiftX,
      shiftY,
      ...texOverride,
      skipBaseTexture,
    })
  }

  return true
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
  regions,
}: MapboxTileRenderParams): boolean {
  const { colormapTexture, uniforms, customShaderConfig } = context

  // Handle region-based loading (UntiledMode - both single-level and multi-level)
  if (regions) {
    if (regions.length === 0) return true // Still loading

    const tilesPerSide = 2 ** tileId.z
    const tileX0 = tileId.x / tilesPerSide
    const tileX1 = (tileId.x + 1) / tilesPerSide
    const tileY0 = tileId.y / tilesPerSide
    const tileY1 = (tileId.y + 1) / tilesPerSide

    let anyRendered = false
    for (const region of regions) {
      const bounds = region.mercatorBounds
      // Check if region intersects this tile
      const intersects =
        bounds.x0 < tileX1 &&
        bounds.x1 > tileX0 &&
        bounds.y0 < tileY1 &&
        bounds.y1 > tileY0

      if (!intersects) continue

      // Globe rendering uses linear (non-warped) tex coords - shader handles reprojection
      // Get or create linear tex coord buffer (computed from vertex positions)
      const linearBuffer = getLinearPixCoordBuffer(context.gl, region)

      const rendered = renderRegionToTile(renderer, context, tileId, {
        vertexBuffer: region.vertexBuffer,
        pixCoordBuffer: linearBuffer,
        texture: region.texture,
        vertexArr: region.vertexArr,
        bounds,
        latIsAscending: region.latIsAscending,
        width: region.width,
        height: region.height,
        bandData: region.bandData,
        bandTextures: region.bandTextures,
        bandTexturesUploaded: region.bandTexturesUploaded,
        bandTexturesConfigured: region.bandTexturesConfigured,
      })
      if (rendered) anyRendered = true
    }
    return !anyRendered // Return true if nothing rendered (still loading)
  }

  // Handle tiled (pyramid) data
  const tiledState = mode.getTiledState?.()
  if (!tiledState?.tileCache) {
    return true
  }

  const {
    tileCache,
    vertexArr,
    pixCoordArr,
    tileSize,
    tileBounds,
    latIsAscending,
  } = tiledState

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
  const maxLevelIndex = mode.getMaxLevelIndex()
  const levels = mode.getLevels()

  if (crs === 'EPSG:4326' && xyLimits) {
    const mapboxGeoBounds = mercatorTileToGeoBounds(
      tileId.z,
      tileId.x,
      tileId.y
    )
    const levelIndex = zoomToLevel(tileId.z, maxLevelIndex)
    // Parse actual zoom from level path to handle pyramids that don't start at 0
    const actualZoom = parseLevelZoom(levels[levelIndex] ?? '', levelIndex)
    const overlappingZarrTiles = getOverlapping4326Tiles(
      mapboxGeoBounds,
      xyLimits,
      actualZoom
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
        latIsAscending,
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
    latIsAscending,
    tileBoundsOverride,
    customShaderConfig,
    true
  )

  const tileHasData = tileCache.get(tileKey)?.data
  return !tileHasData
}
