import {
  createWarpedTexCoords,
  findBestParentTile,
  tileToKey,
  tileToScale,
  type MercatorBounds,
  type TileTuple,
} from './map-utils'
import type { CustomShaderConfig } from './renderer-types'
import type { ShaderProgram } from './shader-program'
import type { Tiles, TileData } from './tiles'
import { setupBandTextureUniforms } from './render-helpers'
import { renderRegion, type RenderableRegion } from './renderable-region'

/**
 * Prepare tile geometry by uploading vertex and texture coordinate buffers.
 * Handles both regular geometry and pre-warped coords for EPSG:4326.
 */
function prepareTileGeometry(
  gl: WebGL2RenderingContext,
  tile: TileData,
  vertexArr: Float32Array,
  pixCoordArr: Float32Array,
  bounds: MercatorBounds | undefined,
  latIsAscending: boolean | null,
  isGlobeTileRender: boolean
): WebGLBuffer {
  // Upload base geometry (vertex positions and linear tex coords)
  gl.bindBuffer(gl.ARRAY_BUFFER, tile.vertexBuffer!)
  if (!tile.geometryUploaded) {
    gl.bufferData(gl.ARRAY_BUFFER, vertexArr, gl.STATIC_DRAW)
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, tile.pixCoordBuffer!)
  if (!tile.geometryUploaded) {
    gl.bufferData(gl.ARRAY_BUFFER, pixCoordArr, gl.STATIC_DRAW)
    tile.geometryUploaded = true
  }

  // For EPSG:4326 data on flat map, compute and upload pre-warped tex coords
  const useWarpedCoords =
    !isGlobeTileRender &&
    tile.latBounds &&
    bounds?.latMin !== undefined &&
    tile.warpedPixCoordBuffer

  if (useWarpedCoords && !tile.warpedGeometryUploaded) {
    const warpedCoords = createWarpedTexCoords(
      vertexArr,
      pixCoordArr,
      {
        x0: bounds!.x0,
        x1: bounds!.x1,
        y0: bounds!.y0,
        y1: bounds!.y1,
      },
      {
        latMin: tile.latBounds!.min,
        latMax: tile.latBounds!.max,
      },
      latIsAscending
    )
    gl.bindBuffer(gl.ARRAY_BUFFER, tile.warpedPixCoordBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, warpedCoords, gl.STATIC_DRAW)
    tile.warpedGeometryUploaded = true
  }

  // Return the appropriate pixCoord buffer
  return useWarpedCoords && tile.warpedGeometryUploaded
    ? tile.warpedPixCoordBuffer!
    : tile.pixCoordBuffer!
}

/**
 * Convert a TileData to a RenderableRegion for unified rendering.
 */
function tileToRenderable(
  tile: TileData,
  bounds: MercatorBounds,
  pixCoordBuffer: WebGLBuffer,
  vertexCount: number,
  tileSize: number,
  texScale: [number, number],
  texOffset: [number, number],
  tileCache: Tiles,
  renderTileKey: string
): RenderableRegion {
  return {
    mercatorBounds: bounds,
    vertexBuffer: tile.vertexBuffer!,
    pixCoordBuffer,
    vertexCount,
    texture: tile.tileTexture!,
    bandData: tile.bandData,
    bandTextures: tile.bandTextures,
    bandTexturesUploaded: tile.bandTexturesUploaded,
    bandTexturesConfigured: tile.bandTexturesConfigured,
    width: tileSize,
    height: tileSize,
    texScale,
    texOffset,
    ensureBandTexture: (bandName) =>
      tileCache.ensureBandTexture(renderTileKey, bandName),
  }
}

export function renderTiles(
  gl: WebGL2RenderingContext,
  shaderProgram: ShaderProgram,
  visibleTiles: TileTuple[],
  worldOffsets: number[],
  tileCache: Tiles,
  tileSize: number,
  vertexArr: Float32Array,
  pixCoordArr: Float32Array,
  latIsAscending: boolean | null,
  tileBounds?: Record<string, MercatorBounds>,
  customShaderConfig?: CustomShaderConfig,
  isGlobeTileRender: boolean = false,
  tileTexOverrides?: Record<
    string,
    { texScale: [number, number]; texOffset: [number, number] }
  >
) {
  // Set up band texture uniforms once per frame
  setupBandTextureUniforms(gl, shaderProgram, customShaderConfig)

  const vertexCount = vertexArr.length / 2

  for (const tileTuple of visibleTiles) {
    const [z, x, y] = tileTuple
    const tileKey = tileToKey(tileTuple)
    const tile = tileCache.get(tileKey)
    const bounds = tileBounds?.[tileKey]

    // Find tile to render (current or parent fallback)
    let tileToRender: TileData | null = null
    let renderTileKey = tileKey
    let texScale: [number, number] = [1, 1]
    let texOffset: [number, number] = [0, 0]

    if (tile && tile.data) {
      tileToRender = tile
    } else {
      const parent = findBestParentTile(tileCache, z, x, y)
      if (parent) {
        tileToRender = parent.tile
        renderTileKey = tileToKey([
          parent.ancestorZ,
          parent.ancestorX,
          parent.ancestorY,
        ])
        const levelDiff = z - parent.ancestorZ
        const divisor = Math.pow(2, levelDiff)
        const localX = x % divisor
        const localY = y % divisor
        texScale = [1 / divisor, 1 / divisor]
        texOffset = [localX / divisor, localY / divisor]
      }
    }

    // Skip tiles without data or WebGL resources
    if (
      !tileToRender ||
      !tileToRender.data ||
      !tileToRender.vertexBuffer ||
      !tileToRender.pixCoordBuffer ||
      !tileToRender.tileTexture
    ) {
      continue
    }

    // Set equirectangular mode per-tile:
    // - Flat map with pre-warped coords: disabled (coords handle reprojection)
    // - Globe with EPSG:4326 data (has latMin): enabled (shader does reprojection)
    // - Globe with EPSG:3857 data (no latMin): disabled (already in Mercator)
    if (shaderProgram.isEquirectangularLoc) {
      const useShaderReproject =
        isGlobeTileRender && bounds?.latMin !== undefined
      gl.uniform1i(
        shaderProgram.isEquirectangularLoc,
        useShaderReproject ? 1 : 0
      )
    }

    // For globe tile rendering, handle texture overrides and lat uniforms
    if (isGlobeTileRender) {
      if (tileTexOverrides?.[tileKey]) {
        const override = tileTexOverrides[tileKey]
        texScale = override.texScale
        texOffset = override.texOffset
      }
      if (bounds?.latMin !== undefined && shaderProgram.latMinLoc) {
        gl.uniform1f(shaderProgram.latMinLoc, bounds.latMin)
      }
      if (bounds?.latMax !== undefined && shaderProgram.latMaxLoc) {
        gl.uniform1f(shaderProgram.latMaxLoc, bounds.latMax)
      }
    }

    // Prepare geometry (upload buffers, compute warped coords)
    const pixCoordBuffer = prepareTileGeometry(
      gl,
      tileToRender,
      vertexArr,
      pixCoordArr,
      bounds,
      latIsAscending,
      isGlobeTileRender
    )

    // Compute mercator bounds for this tile
    // For tiles without explicit bounds, derive from tileToScale
    let mercatorBounds: MercatorBounds
    if (bounds) {
      mercatorBounds = bounds
    } else {
      const [scale, shiftX, shiftY] = tileToScale(tileTuple)
      mercatorBounds = {
        x0: shiftX - scale,
        x1: shiftX + scale,
        y0: shiftY - scale,
        y1: shiftY + scale,
      }
    }

    // Convert tile to RenderableRegion and use unified render path
    const renderable = tileToRenderable(
      tileToRender,
      mercatorBounds,
      pixCoordBuffer,
      vertexCount,
      tileSize,
      texScale,
      texOffset,
      tileCache,
      renderTileKey
    )

    renderRegion(
      gl,
      shaderProgram,
      renderable,
      isGlobeTileRender ? [0] : worldOffsets,
      customShaderConfig
    )
  }
}
