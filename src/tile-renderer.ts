import {
  createWarpedTexCoords,
  findBestParentTile,
  findBestChildTiles,
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
 * Cache for Mapbox tile-specific warped buffers.
 * Key format: `${tileKey}_${boundsKey}` where boundsKey is the Mapbox tile bounds.
 */
const mapboxWarpedBufferCache = new Map<string, WebGLBuffer>()

/**
 * Create a cache key for Mapbox warped buffers.
 */
function mapboxWarpedKey(
  tileKey: string,
  bounds: { x0: number; y0: number; x1: number; y1: number }
): string {
  return `${tileKey}_${bounds.x0},${bounds.y0},${bounds.x1},${bounds.y1}`
}

/**
 * Prepare tile geometry by uploading vertex and texture coordinate buffers.
 * Handles both regular geometry and pre-warped coords for EPSG:4326.
 *
 * For flat map: uses tile's warpedPixCoordBuffer (warped for tile's own bounds)
 * For Mapbox tiles: computes warped coords for the specific Mapbox tile bounds
 */
function prepareTileGeometry(
  gl: WebGL2RenderingContext,
  tile: TileData,
  tileKey: string,
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

  // For EPSG:4326 data, compute and use pre-warped tex coords
  const hasLatBounds = tile.latBounds && bounds?.latMin !== undefined

  if (hasLatBounds) {
    if (isGlobeTileRender) {
      // Mapbox tile rendering: warp for the specific Mapbox tile bounds
      const cacheKey = mapboxWarpedKey(tileKey, bounds!)
      let warpedBuffer = mapboxWarpedBufferCache.get(cacheKey)

      if (!warpedBuffer) {
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
        warpedBuffer = gl.createBuffer()!
        gl.bindBuffer(gl.ARRAY_BUFFER, warpedBuffer)
        gl.bufferData(gl.ARRAY_BUFFER, warpedCoords, gl.STATIC_DRAW)
        mapboxWarpedBufferCache.set(cacheKey, warpedBuffer)
      }

      return warpedBuffer
    } else if (tile.warpedPixCoordBuffer) {
      // Flat map rendering: use tile's own warped buffer
      if (!tile.warpedGeometryUploaded) {
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
      return tile.warpedPixCoordBuffer
    }
  }

  // No warping needed (EPSG:3857 or no lat bounds)
  return tile.pixCoordBuffer!
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
  datasetMaxZoom?: number,
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
      // Try parent first (zoom-in case)
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
      } else if (datasetMaxZoom !== undefined) {
        // Try children (zoom-out case)
        const children = findBestChildTiles(tileCache, z, x, y, datasetMaxZoom)
        if (children && children.length > 0) {
          // Compute target tile's mercator bounds
          let targetBounds: MercatorBounds
          if (bounds) {
            targetBounds = bounds
          } else {
            const [scale, shiftX, shiftY] = tileToScale(tileTuple)
            targetBounds = {
              x0: shiftX - scale,
              x1: shiftX + scale,
              y0: shiftY - scale,
              y1: shiftY + scale,
            }
          }

          // Render each child tile at its sub-position within target bounds
          for (const child of children) {
            if (
              !child.tile.data ||
              !child.tile.vertexBuffer ||
              !child.tile.pixCoordBuffer ||
              !child.tile.tileTexture
            ) {
              continue
            }

            const levelDiff = child.childZ - z
            const divisor = Math.pow(2, levelDiff)

            // Calculate which sub-position this child covers
            const localX = child.childX % divisor
            const localY = child.childY % divisor

            // Compute child's mercator bounds as a sub-region of target bounds
            const xSpan = targetBounds.x1 - targetBounds.x0
            const ySpan = targetBounds.y1 - targetBounds.y0
            const childBounds: MercatorBounds = {
              x0: targetBounds.x0 + (localX / divisor) * xSpan,
              x1: targetBounds.x0 + ((localX + 1) / divisor) * xSpan,
              y0: targetBounds.y0 + (localY / divisor) * ySpan,
              y1: targetBounds.y0 + ((localY + 1) / divisor) * ySpan,
            }

            // Preserve lat bounds if present (scale them for the child's portion)
            if (bounds?.latMin !== undefined && bounds?.latMax !== undefined) {
              const latSpan = bounds.latMax - bounds.latMin
              childBounds.latMin = bounds.latMin + (localY / divisor) * latSpan
              childBounds.latMax =
                bounds.latMin + ((localY + 1) / divisor) * latSpan
            }

            const childTileKey = tileToKey([
              child.childZ,
              child.childX,
              child.childY,
            ])

            // Prepare geometry for this child tile (CPU warp baked into coords)
            const childPixCoordBuffer = prepareTileGeometry(
              gl,
              child.tile,
              childTileKey,
              vertexArr,
              pixCoordArr,
              childBounds,
              latIsAscending,
              isGlobeTileRender
            )

            // Child uses full texture (texScale=1, texOffset=0)
            const childRenderable = tileToRenderable(
              child.tile,
              childBounds,
              childPixCoordBuffer,
              vertexCount,
              tileSize,
              [1, 1],
              [0, 0],
              tileCache,
              childTileKey
            )

            renderRegion(
              gl,
              shaderProgram,
              childRenderable,
              isGlobeTileRender ? [0] : worldOffsets,
              customShaderConfig
            )
          }
          continue // Skip normal tile render - we've rendered children
        }
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

    // For globe tile rendering, handle texture overrides
    if (isGlobeTileRender && tileTexOverrides?.[tileKey]) {
      const override = tileTexOverrides[tileKey]
      texScale = override.texScale
      texOffset = override.texOffset
    }

    // Prepare geometry (upload buffers, compute warped coords for EPSG:4326)
    // CPU warp is baked into the coords, no shader reprojection needed
    const pixCoordBuffer = prepareTileGeometry(
      gl,
      tileToRender,
      renderTileKey,
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
