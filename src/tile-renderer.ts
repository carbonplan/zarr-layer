import {
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
 * Prepare tile geometry by uploading vertex and texture coordinate buffers.
 */
function prepareTileGeometry(
  gl: WebGL2RenderingContext,
  tile: TileData,
  vertexArr: Float32Array,
  pixCoordArr: Float32Array
): WebGLBuffer {
  // Upload geometry (vertex positions and linear tex coords)
  gl.bindBuffer(gl.ARRAY_BUFFER, tile.vertexBuffer!)
  if (!tile.geometryUploaded) {
    gl.bufferData(gl.ARRAY_BUFFER, vertexArr, gl.STATIC_DRAW)
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, tile.pixCoordBuffer!)
  if (!tile.geometryUploaded) {
    gl.bufferData(gl.ARRAY_BUFFER, pixCoordArr, gl.STATIC_DRAW)
    tile.geometryUploaded = true
  }

  return tile.pixCoordBuffer!
}

/**
 * Convert a TileData to a RenderableRegion for unified rendering.
 * Note: For EPSG:4326 tiles, fragment shader reprojection is used.
 * The mercatorBounds should include latMin/latMax for fragment shader to use.
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
  renderTileKey: string,
  latIsAscending: boolean
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
    latIsAscending,
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
  tileBounds?: Record<string, MercatorBounds>,
  customShaderConfig?: CustomShaderConfig,
  isGlobeTileRender: boolean = false,
  datasetMaxZoom?: number,
  tileTexOverrides?: Record<
    string,
    { texScale: [number, number]; texOffset: [number, number] }
  >,
  latIsAscending: boolean = true
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

            // Preserve lat/lon bounds if present (scale them for the child's portion)
            if (bounds?.latMin !== undefined && bounds?.latMax !== undefined) {
              const latSpan = bounds.latMax - bounds.latMin
              childBounds.latMin = bounds.latMin + (localY / divisor) * latSpan
              childBounds.latMax =
                bounds.latMin + ((localY + 1) / divisor) * latSpan
            }
            if (bounds?.lonMin !== undefined && bounds?.lonMax !== undefined) {
              const lonSpan = bounds.lonMax - bounds.lonMin
              childBounds.lonMin = bounds.lonMin + (localX / divisor) * lonSpan
              childBounds.lonMax =
                bounds.lonMin + ((localX + 1) / divisor) * lonSpan
            }

            // Prepare geometry for this child tile
            const childPixCoordBuffer = prepareTileGeometry(
              gl,
              child.tile,
              vertexArr,
              pixCoordArr
            )

            const childTileKey = tileToKey([
              child.childZ,
              child.childX,
              child.childY,
            ])

            // Child uses full texture (texScale=1, texOffset=0)
            // Note: childBounds includes latMin/latMax for fragment shader reprojection
            const childRenderable = tileToRenderable(
              child.tile,
              childBounds,
              childPixCoordBuffer,
              vertexCount,
              tileSize,
              [1, 1],
              [0, 0],
              tileCache,
              childTileKey,
              latIsAscending
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

    // Prepare geometry (upload buffers)
    const pixCoordBuffer = prepareTileGeometry(
      gl,
      tileToRender,
      vertexArr,
      pixCoordArr
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
    // Note: mercatorBounds includes latMin/latMax for fragment shader reprojection (EPSG:4326)
    const renderable = tileToRenderable(
      tileToRender,
      mercatorBounds,
      pixCoordBuffer,
      vertexCount,
      tileSize,
      texScale,
      texOffset,
      tileCache,
      renderTileKey,
      latIsAscending
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
