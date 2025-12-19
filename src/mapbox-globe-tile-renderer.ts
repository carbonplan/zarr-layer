/**
 * @module mapbox-globe-tile-renderer
 *
 * Specialized rendering for Mapbox GL JS globe tile API (renderToTile).
 * This is fundamentally different from MapLibre's globe which uses projectTile().
 *
 * Mapbox's renderToTile() asks the custom layer to render individual tiles
 * to offscreen textures, requiring:
 * - Tile-specific transformation matrix (not camera matrix)
 * - Pre-warped texture coordinates (CPU-based reprojection, same as untiled mode)
 */

import {
  findBestParentTile,
  findBestChildTiles,
  mercatorTileToGeoBounds,
  getOverlapping4326Tiles,
  get4326TileGeoBounds,
  tileToKey,
  latToMercatorNorm,
  lonToMercatorNorm,
  mercatorNormToLon,
  parseLevelZoom,
  zoomToLevel,
  createWarpedTexCoords,
  type TileTuple,
  type XYLimits,
} from './map-utils'
import type { ZarrRenderer, ShaderProgram } from './zarr-renderer'
import type {
  ZarrMode,
  RenderContext,
  TileId,
  RegionRenderState,
} from './zarr-mode'
import { createLinearTexCoordsFromVertices } from './webgl-utils'
import { setupBandTextureUniforms } from './render-helpers'
import { renderRegion, type RenderableRegion } from './renderable-region'
import {
  MAPBOX_GLOBE_IDENTITY_MATRIX,
  createMapboxTileMatrix,
  getMapboxTileBounds,
  boundsIntersect,
} from './mapbox-globe-utils'

/**
 * Cache for warped texture coordinate buffers used in Mapbox globe rendering.
 * Outer WeakMap ensures buffers are garbage-collected when RegionRenderState is removed.
 * Inner Map caches per-tile warped buffers (keyed by tile bounds string).
 */
const warpedBufferCache = new WeakMap<
  RegionRenderState,
  Map<string, WebGLBuffer>
>()

/**
 * Create a cache key from tile bounds.
 */
function tileBoundsKey(bounds: {
  x0: number
  y0: number
  x1: number
  y1: number
}): string {
  return `${bounds.x0},${bounds.y0},${bounds.x1},${bounds.y1}`
}

interface TexCoordResult {
  buffer: WebGLBuffer
  texScale: [number, number]
  texOffset: [number, number]
}

/**
 * Cache for linear tex coord buffers (one per region, reused across tiles).
 */
const linearBufferCache = new WeakMap<RegionRenderState, WebGLBuffer>()

/**
 * Get or create a linear texture coordinate buffer for a region.
 */
function getOrCreateLinearBuffer(
  gl: WebGL2RenderingContext,
  region: RegionRenderState
): WebGLBuffer {
  let buffer = linearBufferCache.get(region)
  if (buffer) return buffer

  const linearCoords = createLinearTexCoordsFromVertices(region.vertexArr)
  buffer = gl.createBuffer()!
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
  gl.bufferData(gl.ARRAY_BUFFER, linearCoords, gl.STATIC_DRAW)

  linearBufferCache.set(region, buffer)
  return buffer
}

/**
 * Get texture coordinates and transform for rendering a region to a specific tile.
 *
 * For EPSG:4326: Returns warped buffer (CPU warp needed for non-linear lat→mercY)
 * For EPSG:3857: Returns linear buffer + texScale/texOffset (shader does linear crop)
 */
function getTexCoordsForTile(
  gl: WebGL2RenderingContext,
  region: RegionRenderState,
  overlapBounds: { x0: number; y0: number; x1: number; y1: number }
): TexCoordResult {
  const bounds = region.mercatorBounds
  const isEquirectangular =
    bounds.latMin !== undefined && bounds.latMax !== undefined

  if (!isEquirectangular) {
    // EPSG:3857: Use linear buffer + shader transform (no CPU warp needed)
    const buffer = getOrCreateLinearBuffer(gl, region)

    const regionWidth = bounds.x1 - bounds.x0
    const regionHeight = bounds.y1 - bounds.y0
    const texScale: [number, number] = [
      regionWidth > 0 ? (overlapBounds.x1 - overlapBounds.x0) / regionWidth : 1,
      regionHeight > 0
        ? (overlapBounds.y1 - overlapBounds.y0) / regionHeight
        : 1,
    ]
    const texOffset: [number, number] = [
      regionWidth > 0 ? (overlapBounds.x0 - bounds.x0) / regionWidth : 0,
      regionHeight > 0 ? (overlapBounds.y0 - bounds.y0) / regionHeight : 0,
    ]

    return { buffer, texScale, texOffset }
  }

  // EPSG:4326: Need CPU warp for non-linear latitude reprojection
  const boundsKey = tileBoundsKey(overlapBounds)

  let regionCache = warpedBufferCache.get(region)
  if (!regionCache) {
    regionCache = new Map()
    warpedBufferCache.set(region, regionCache)
  }

  let buffer = regionCache.get(boundsKey)
  if (!buffer) {
    const linearCoords = createLinearTexCoordsFromVertices(region.vertexArr)

    // Warp for latitude reprojection
    const warpedCoords = createWarpedTexCoords(
      region.vertexArr,
      linearCoords,
      overlapBounds,
      { latMin: bounds.latMin!, latMax: bounds.latMax! },
      region.latIsAscending ?? null
    )

    // Apply X cropping (bake into coords since we're already creating a buffer)
    const regionWidth = bounds.x1 - bounds.x0
    if (regionWidth > 0) {
      const texScaleX = (overlapBounds.x1 - overlapBounds.x0) / regionWidth
      const texOffsetX = (overlapBounds.x0 - bounds.x0) / regionWidth
      for (let i = 0; i < warpedCoords.length; i += 2) {
        warpedCoords[i] = warpedCoords[i] * texScaleX + texOffsetX
      }
    }

    buffer = gl.createBuffer()!
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
    gl.bufferData(gl.ARRAY_BUFFER, warpedCoords, gl.STATIC_DRAW)
    regionCache.set(boundsKey, buffer)
  }

  // Warp is baked in, use identity transform
  return { buffer, texScale: [1, 1], texOffset: [0, 0] }
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
 * Renders Zarr data to a Mapbox globe tile.
 *
 * This is the main entry point for globe tile rendering. It handles:
 * - Single image (non-tiled) data: Renders the portion of the image that
 *   intersects the requested tile
 * - Tiled pyramid data: Finds and renders the appropriate Zarr tiles that
 *   overlap the Mapbox tile
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
  // Handle region-based loading (UntiledMode - both single-level and multi-level)
  if (regions) {
    return renderRegionsToTile(renderer, tileId, context, regions)
  }

  // Handle tiled (pyramid) data
  return renderTiledToTile(renderer, mode, tileId, context)
}

/**
 * Render untiled regions to a globe tile.
 */
function renderRegionsToTile(
  renderer: ZarrRenderer,
  tileId: TileId,
  context: RenderContext,
  regions: RegionRenderState[]
): boolean {
  if (regions.length === 0) return true // Still loading

  const tileBounds = getMapboxTileBounds(tileId)
  const tileMatrix = createMapboxTileMatrix(
    tileBounds.x0,
    tileBounds.y0,
    tileBounds.x1,
    tileBounds.y1
  )

  const { colormapTexture, uniforms, customShaderConfig } = context
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
      globeToMercatorMatrix: MAPBOX_GLOBE_IDENTITY_MATRIX,
      transition: 0,
    },
    tileMatrix,
    true
  )

  setupBandTextureUniforms(renderer.gl, shaderProgram, customShaderConfig)

  let needsMoreData = false
  for (const region of regions) {
    const bounds = region.mercatorBounds
    if (!boundsIntersect(bounds, tileBounds)) continue

    // Calculate overlap bounds
    const overlapBounds = {
      x0: Math.max(bounds.x0, tileBounds.x0),
      x1: Math.min(bounds.x1, tileBounds.x1),
      y0: Math.max(bounds.y0, tileBounds.y0),
      y1: Math.min(bounds.y1, tileBounds.y1),
    }

    // Get tex coords (CPU warp for 4326, shader transform for 3857)
    const texCoords = getTexCoordsForTile(renderer.gl, region, overlapBounds)

    // Create renderable region
    const renderable = createRenderableFromRegion(
      region,
      texCoords,
      overlapBounds
    )

    const rendered = renderRegion(
      renderer.gl,
      shaderProgram,
      renderable,
      [0], // Globe tiles don't need world wrapping
      customShaderConfig
    )
    if (!rendered) {
      // renderRegion returns false when band data is missing
      needsMoreData = true
    }
  }

  // Return true if any region still needs data (triggers re-render when loaded)
  return needsMoreData
}

/**
 * Create a RenderableRegion from a RegionRenderState for globe tile rendering.
 */
function createRenderableFromRegion(
  region: RegionRenderState,
  texCoords: TexCoordResult,
  overlapBounds: { x0: number; y0: number; x1: number; y1: number }
): RenderableRegion {
  return {
    mercatorBounds: overlapBounds,
    vertexBuffer: region.vertexBuffer,
    pixCoordBuffer: texCoords.buffer,
    vertexCount: region.vertexArr.length / 2,
    texture: region.texture,
    bandData: region.bandData ?? new Map(),
    bandTextures: region.bandTextures ?? new Map(),
    bandTexturesUploaded: region.bandTexturesUploaded ?? new Set(),
    bandTexturesConfigured: region.bandTexturesConfigured ?? new Set(),
    width: region.width,
    height: region.height,
    texScale: texCoords.texScale,
    texOffset: texCoords.texOffset,
  }
}

/**
 * Render tiled pyramid data to a globe tile.
 */
function renderTiledToTile(
  renderer: ZarrRenderer,
  mode: ZarrMode,
  tileId: TileId,
  context: RenderContext
): boolean {
  const tiledState = mode.getTiledState?.()
  if (!tiledState?.tileCache) {
    return true
  }

  const crs = mode.getCRS()
  if (crs === 'EPSG:4326') {
    return render4326TiledToTile(renderer, mode, tileId, context, tiledState)
  }

  return render3857TiledToTile(renderer, mode, tileId, context, tiledState)
}

/**
 * Render EPSG:4326 tiled data to a globe tile.
 * Handles the complex case of mapping 4326 tiles to Mapbox Mercator tiles.
 */
function render4326TiledToTile(
  renderer: ZarrRenderer,
  mode: ZarrMode,
  tileId: TileId,
  context: RenderContext,
  tiledState: NonNullable<ReturnType<NonNullable<ZarrMode['getTiledState']>>>
): boolean {
  const { customShaderConfig } = context
  const xyLimits = mode.getXYLimits()
  const maxLevelIndex = mode.getMaxLevelIndex()
  const levels = mode.getLevels()

  if (!xyLimits) return true

  const tileBounds = getMapboxTileBounds(tileId)
  const tileMatrix = createMapboxTileMatrix(
    tileBounds.x0,
    tileBounds.y0,
    tileBounds.x1,
    tileBounds.y1
  )

  const mapboxGeoBounds = mercatorTileToGeoBounds(tileId.z, tileId.x, tileId.y)
  const levelIndex = zoomToLevel(tileId.z, maxLevelIndex)
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

  const maxZoomLevelPath = levels[maxLevelIndex] ?? ''
  const datasetMaxZoom = parseLevelZoom(maxZoomLevelPath, maxLevelIndex)

  let anyTileRendered = false
  let anyMissing = false

  for (const zarrTile of overlappingZarrTiles) {
    const result = renderSingle4326Tile(
      renderer,
      shaderProgram,
      zarrTile,
      tileId,
      tileBounds,
      tileMatrix,
      tiledState,
      context,
      xyLimits,
      datasetMaxZoom
    )

    if (result.rendered) anyTileRendered = true
    if (result.missing) anyMissing = true
  }

  return anyMissing || !anyTileRendered
}

/**
 * Render a single EPSG:4326 Zarr tile to a globe tile.
 * Handles parent/child tile fallback.
 */
function renderSingle4326Tile(
  renderer: ZarrRenderer,
  shaderProgram: ShaderProgram,
  zarrTile: TileTuple,
  _tileId: TileId,
  tileBounds: { x0: number; y0: number; x1: number; y1: number },
  tileMatrix: Float32Array,
  tiledState: NonNullable<ReturnType<NonNullable<ZarrMode['getTiledState']>>>,
  context: RenderContext,
  xyLimits: XYLimits,
  datasetMaxZoom: number
): { rendered: boolean; missing: boolean } {
  const { tileCache, vertexArr, pixCoordArr, tileSize, latIsAscending } =
    tiledState
  const { colormapTexture, uniforms, customShaderConfig } = context

  const zarrTileKey = tileToKey(zarrTile)
  let tileData = tileCache.get(zarrTileKey)
  let renderTileTuple: TileTuple = zarrTile
  let missing = false

  if (!tileData?.data) {
    missing = true

    // Try parent first (zoom-in case)
    const parent = findBestParentTile(
      tileCache,
      zarrTile[0],
      zarrTile[1],
      zarrTile[2]
    )
    if (parent) {
      tileData = parent.tile
      renderTileTuple = [parent.ancestorZ, parent.ancestorX, parent.ancestorY]
    } else {
      // Try children (zoom-out case)
      const children = findBestChildTiles(
        tileCache,
        zarrTile[0],
        zarrTile[1],
        zarrTile[2],
        datasetMaxZoom
      )
      if (children && children.length > 0) {
        let anyChildRendered = false
        for (const child of children) {
          if (!child.tile.data) continue

          const rendered = renderChild4326Tile(
            renderer,
            shaderProgram,
            child,
            tileBounds,
            tileMatrix,
            tiledState,
            context,
            xyLimits
          )
          if (rendered) anyChildRendered = true
        }
        return { rendered: anyChildRendered, missing: true }
      }
      return { rendered: false, missing: true }
    }
  }

  // Render the tile (either original or parent fallback)
  const [z, tx, ty] = renderTileTuple
  const renderTileKey = tileToKey(renderTileTuple)
  const zarrGeoBounds = get4326TileGeoBounds(z, tx, ty, xyLimits)

  const zarrMercX0 = lonToMercatorNorm(zarrGeoBounds.west)
  const zarrMercX1 = lonToMercatorNorm(zarrGeoBounds.east)
  const zarrMercY0 = latToMercatorNorm(zarrGeoBounds.north)
  const zarrMercY1 = latToMercatorNorm(zarrGeoBounds.south)

  const overlapX0 = Math.max(zarrMercX0, tileBounds.x0)
  const overlapX1 = Math.min(zarrMercX1, tileBounds.x1)
  const overlapY0 = Math.max(zarrMercY0, tileBounds.y0)
  const overlapY1 = Math.min(zarrMercY1, tileBounds.y1)

  if (overlapX1 <= overlapX0 || overlapY1 <= overlapY0) {
    return { rendered: false, missing }
  }

  const zarrLonWidth = zarrGeoBounds.east - zarrGeoBounds.west
  const overlapWest = mercatorNormToLon(overlapX0)
  const overlapEast = mercatorNormToLon(overlapX1)
  const texScaleX =
    zarrLonWidth > 0 ? (overlapEast - overlapWest) / zarrLonWidth : 1
  const texOffsetX =
    zarrLonWidth > 0 ? (overlapWest - zarrGeoBounds.west) / zarrLonWidth : 0

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
      globeToMercatorMatrix: MAPBOX_GLOBE_IDENTITY_MATRIX,
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
    undefined,
    {
      [renderTileKey]: {
        texScale: [texScaleX, 1.0],
        texOffset: [texOffsetX, 0.0],
      },
    }
  )

  return { rendered: true, missing }
}

/**
 * Render a child tile for zoom-out fallback in EPSG:4326 mode.
 */
function renderChild4326Tile(
  renderer: ZarrRenderer,
  shaderProgram: ShaderProgram,
  child: {
    childZ: number
    childX: number
    childY: number
    tile: ReturnType<
      NonNullable<
        ReturnType<NonNullable<ZarrMode['getTiledState']>>
      >['tileCache']['get']
    >
  },
  tileBounds: { x0: number; y0: number; x1: number; y1: number },
  tileMatrix: Float32Array,
  tiledState: NonNullable<ReturnType<NonNullable<ZarrMode['getTiledState']>>>,
  context: RenderContext,
  xyLimits: XYLimits
): boolean {
  const { tileCache, vertexArr, pixCoordArr, tileSize, latIsAscending } =
    tiledState
  const { colormapTexture, uniforms, customShaderConfig } = context

  const childTileTuple: TileTuple = [child.childZ, child.childX, child.childY]
  const childTileKey = tileToKey(childTileTuple)
  const childGeoBounds = get4326TileGeoBounds(
    child.childZ,
    child.childX,
    child.childY,
    xyLimits
  )

  const childMercX0 = lonToMercatorNorm(childGeoBounds.west)
  const childMercX1 = lonToMercatorNorm(childGeoBounds.east)
  const childMercY0 = latToMercatorNorm(childGeoBounds.north)
  const childMercY1 = latToMercatorNorm(childGeoBounds.south)

  const childOverlapX0 = Math.max(childMercX0, tileBounds.x0)
  const childOverlapX1 = Math.min(childMercX1, tileBounds.x1)
  const childOverlapY0 = Math.max(childMercY0, tileBounds.y0)
  const childOverlapY1 = Math.min(childMercY1, tileBounds.y1)

  if (childOverlapX1 <= childOverlapX0 || childOverlapY1 <= childOverlapY0) {
    return false
  }

  const childLonWidth = childGeoBounds.east - childGeoBounds.west
  const childOverlapWest = mercatorNormToLon(childOverlapX0)
  const childOverlapEast = mercatorNormToLon(childOverlapX1)
  const childTexScaleX =
    childLonWidth > 0
      ? (childOverlapEast - childOverlapWest) / childLonWidth
      : 1
  const childTexOffsetX =
    childLonWidth > 0
      ? (childOverlapWest - childGeoBounds.west) / childLonWidth
      : 0

  const childTileBoundsForRender = {
    [childTileKey]: {
      x0: childOverlapX0,
      y0: childOverlapY0,
      x1: childOverlapX1,
      y1: childOverlapY1,
      latMin: childGeoBounds.south,
      latMax: childGeoBounds.north,
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
      globeToMercatorMatrix: MAPBOX_GLOBE_IDENTITY_MATRIX,
      transition: 0,
    },
    tileMatrix,
    true
  )

  renderer.renderTiles(
    shaderProgram,
    [childTileTuple],
    [0],
    tileCache,
    tileSize,
    vertexArr,
    pixCoordArr,
    latIsAscending,
    childTileBoundsForRender,
    customShaderConfig,
    true,
    undefined,
    {
      [childTileKey]: {
        texScale: [childTexScaleX, 1.0],
        texOffset: [childTexOffsetX, 0.0],
      },
    }
  )

  return true
}

/**
 * Render EPSG:3857 tiled data to a globe tile.
 * This is simpler since Mapbox tiles are already in Mercator.
 */
function render3857TiledToTile(
  renderer: ZarrRenderer,
  mode: ZarrMode,
  tileId: TileId,
  context: RenderContext,
  tiledState: NonNullable<ReturnType<NonNullable<ZarrMode['getTiledState']>>>
): boolean {
  const {
    tileCache,
    vertexArr,
    pixCoordArr,
    tileSize,
    tileBounds,
    latIsAscending,
  } = tiledState
  const { colormapTexture, uniforms, customShaderConfig } = context
  const levels = mode.getLevels()
  const maxLevelIndex = mode.getMaxLevelIndex()

  const mapboxTileBounds = getMapboxTileBounds(tileId)
  const tileMatrix = createMapboxTileMatrix(
    mapboxTileBounds.x0,
    mapboxTileBounds.y0,
    mapboxTileBounds.x1,
    mapboxTileBounds.y1
  )

  const tileTuple: TileTuple = [tileId.z, tileId.x, tileId.y]
  const tileKey = tileTuple.join(',')

  const boundsForTile = tileBounds?.[tileKey]
  const tileBoundsOverride = {
    [tileKey]: {
      x0: mapboxTileBounds.x0,
      y0: mapboxTileBounds.y0,
      x1: mapboxTileBounds.x1,
      y1: mapboxTileBounds.y1,
      latMin: boundsForTile?.latMin,
      latMax: boundsForTile?.latMax,
    },
  }

  const maxZoomLevelPath = levels[maxLevelIndex] ?? ''
  const datasetMaxZoom = parseLevelZoom(maxZoomLevelPath, maxLevelIndex)

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
      globeToMercatorMatrix: MAPBOX_GLOBE_IDENTITY_MATRIX,
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
    true,
    datasetMaxZoom
  )

  const tileHasData = tileCache.get(tileKey)?.data
  return !tileHasData
}
