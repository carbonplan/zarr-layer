/**
 * @module mapbox-globe-tile-renderer
 *
 * Specialized rendering for Mapbox GL JS globe tile API (renderToTile).
 * This is fundamentally different from MapLibre's globe which uses projectTile().
 *
 * Mapbox's renderToTile() asks the custom layer to render individual tiles
 * to offscreen textures, requiring:
 * - Tile-specific transformation matrix (not camera matrix)
 * - For EPSG:4326 data: fragment shader reprojection (Mercator → latitude for texture lookup)
 */

import {
  findBestParentTile,
  findBestChildTiles,
  get4326TileGeoBounds,
  tileToKey,
  latToMercatorNorm,
  lonToMercatorNorm,
  mercatorNormToLon,
  mercatorNormToLat,
  parseLevelZoom,
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
import { setupBandTextureUniforms } from './render-helpers'
import { renderRegion, type RenderableRegion } from './renderable-region'
import {
  MAPBOX_GLOBE_IDENTITY_MATRIX,
  createMapboxTileMatrix,
  getMapboxTileBounds,
  boundsIntersect,
} from './mapbox-globe-utils'

// ============================================================================
// Helper Functions
// ============================================================================

/** Extended MercatorBounds with lat/lon for EPSG:4326 tiles */
interface TileMercatorBounds {
  x0: number
  y0: number
  x1: number
  y1: number
  latMin?: number
  latMax?: number
  lonMin?: number
  lonMax?: number
}

/** Result of computing tile bounds and texture coordinates */
interface TileBoundsAndTexCoords {
  overlap: { x0: number; y0: number; x1: number; y1: number }
  latBounds: { min: number; max: number }
  texScale: [number, number]
  texOffset: [number, number]
}

/**
 * Get or compute mercator bounds for a 4326 tile.
 * Uses pre-computed bounds if available, otherwise computes from tile coordinates.
 */
function getTileMercatorBounds(
  z: number,
  x: number,
  y: number,
  tileBoundsMap: Record<string, TileMercatorBounds> | undefined,
  xyLimits: XYLimits
): TileMercatorBounds {
  const tileKey = tileToKey([z, x, y])
  const precomputed = tileBoundsMap?.[tileKey]
  if (precomputed) return precomputed

  const geoBounds = get4326TileGeoBounds(z, x, y, xyLimits)
  return {
    x0: lonToMercatorNorm(geoBounds.west),
    x1: lonToMercatorNorm(geoBounds.east),
    y0: latToMercatorNorm(geoBounds.north),
    y1: latToMercatorNorm(geoBounds.south),
    latMin: geoBounds.south,
    latMax: geoBounds.north,
    lonMin: geoBounds.west,
    lonMax: geoBounds.east,
  }
}

/**
 * Compute tile overlap and texture coordinates for EPSG:4326 rendering.
 * Returns null if the tile doesn't overlap with the mapbox tile bounds.
 */
function computeTileBoundsAndTexCoords(
  zarrBounds: TileMercatorBounds,
  mapboxTileBounds: { x0: number; y0: number; x1: number; y1: number }
): TileBoundsAndTexCoords | null {
  // Compute overlap in Mercator space
  const overlapX0 = Math.max(zarrBounds.x0, mapboxTileBounds.x0)
  const overlapX1 = Math.min(zarrBounds.x1, mapboxTileBounds.x1)
  const overlapY0 = Math.max(zarrBounds.y0, mapboxTileBounds.y0)
  const overlapY1 = Math.min(zarrBounds.y1, mapboxTileBounds.y1)

  if (overlapX1 <= overlapX0 || overlapY1 <= overlapY0) {
    return null
  }

  // Get lat/lon bounds (required for fragment shader reprojection)
  const lonMin = zarrBounds.lonMin ?? mercatorNormToLon(zarrBounds.x0)
  const lonMax = zarrBounds.lonMax ?? mercatorNormToLon(zarrBounds.x1)
  const latMin = zarrBounds.latMin ?? mercatorNormToLat(zarrBounds.y1)
  const latMax = zarrBounds.latMax ?? mercatorNormToLat(zarrBounds.y0)

  // Compute texture coordinates for X
  const lonWidth = lonMax - lonMin
  const overlapWest = mercatorNormToLon(overlapX0)
  const overlapEast = mercatorNormToLon(overlapX1)
  const texScaleX = lonWidth > 0 ? (overlapEast - overlapWest) / lonWidth : 1
  const texOffsetX = lonWidth > 0 ? (overlapWest - lonMin) / lonWidth : 0

  // Y texture coords: shader handles lat→texV mapping via u_latBounds
  const texScaleY = 1
  const texOffsetY = 0

  return {
    overlap: { x0: overlapX0, y0: overlapY0, x1: overlapX1, y1: overlapY1 },
    latBounds: { min: latMin, max: latMax },
    texScale: [texScaleX, texScaleY],
    texOffset: [texOffsetX, texOffsetY],
  }
}

// ============================================================================
// Main Rendering Functions
// ============================================================================

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

  // Convert tile bounds to WGS84 for intersection with wgs84Bounds regions
  // Tile bounds are in normalized Mercator [0,1], convert to normalized WGS84 [0,1]
  const tileWgs84Bounds = {
    lon0: tileBounds.x0, // lon is same in both spaces (just x)
    lon1: tileBounds.x1,
    // Mercator Y needs conversion: y0 is north (smaller lat), y1 is south (larger lat)
    // In normalized WGS84: lat0 is south, lat1 is north
    lat0: (mercatorNormToLat(tileBounds.y1) + 90) / 180, // south edge
    lat1: (mercatorNormToLat(tileBounds.y0) + 90) / 180, // north edge
  }

  const tileMatrix = createMapboxTileMatrix(
    tileBounds.x0,
    tileBounds.y0,
    tileBounds.x1,
    tileBounds.y1
  )

  const { colormapTexture, uniforms, customShaderConfig } = context

  // Check if any region uses WGS84 (proj4 datasets or EPSG:4326)
  const useWgs84 = regions.some((r) => !!r.wgs84Bounds)

  const shaderProgram = renderer.getProgram(
    context.shaderData,
    customShaderConfig,
    true, // useMapboxGlobe
    useWgs84
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
    // For regions with wgs84Bounds, do intersection in WGS84 space
    // This avoids the Mercator/WGS84 coordinate mismatch at low zoom
    let intersects: boolean
    if (region.wgs84Bounds) {
      const w = region.wgs84Bounds
      intersects =
        w.lon0 < tileWgs84Bounds.lon1 &&
        w.lon1 > tileWgs84Bounds.lon0 &&
        w.lat0 < tileWgs84Bounds.lat1 &&
        w.lat1 > tileWgs84Bounds.lat0
    } else {
      intersects = boundsIntersect(region.mercatorBounds, tileBounds)
    }
    if (!intersects) continue

    // For proj4 datasets: use indexed mesh with wgs84Bounds
    // For EPSG:4326: use subdivided quad (not indexed) with wgs84Bounds
    // Both use the mapbox-globe-wgs84 shader to convert WGS84 → Mercator
    const useIndexedMesh = !!region.useIndexedMesh && !!region.indexBuffer

    const renderable: RenderableRegion = {
      mercatorBounds: region.mercatorBounds,
      vertexBuffer: region.vertexBuffer,
      pixCoordBuffer: region.pixCoordBuffer,
      vertexCount: useIndexedMesh
        ? region.vertexCount ?? region.vertexArr.length / 2
        : region.vertexArr.length / 2,
      texture: region.texture,
      bandData: region.bandData ?? new Map(),
      bandTextures: region.bandTextures ?? new Map(),
      bandTexturesUploaded: region.bandTexturesUploaded ?? new Set(),
      bandTexturesConfigured: region.bandTexturesConfigured ?? new Set(),
      width: region.width,
      height: region.height,
      // Include indexed mesh fields for proj4 datasets
      indexBuffer: useIndexedMesh ? region.indexBuffer : undefined,
      useIndexedMesh: useIndexedMesh,
      // Include wgs84Bounds for both proj4 and EPSG:4326 datasets
      wgs84Bounds: region.wgs84Bounds,
      latIsAscending: region.latIsAscending,
    }

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
 *
 * IMPORTANT: We use tiledState.visibleTiles (computed by TiledMode from map.getZoom())
 * rather than computing tiles from tileId.z. Mapbox's tileId.z may be overscaled
 * and doesn't match the dataset level that TiledMode fetched.
 */
function render4326TiledToTile(
  renderer: ZarrRenderer,
  mode: ZarrMode,
  tileId: TileId,
  context: RenderContext,
  tiledState: NonNullable<ReturnType<NonNullable<ZarrMode['getTiledState']>>>
): boolean {
  const { customShaderConfig } = context
  const { visibleTiles, tileBounds: zarrTileBounds } = tiledState
  const xyLimits = mode.getXYLimits()
  const maxLevelIndex = mode.getMaxLevelIndex()
  const levels = mode.getLevels()

  if (!xyLimits || visibleTiles.length === 0) return true

  // Get the Mapbox tile bounds we need to render into
  const mapboxTileBounds = getMapboxTileBounds(tileId)
  const tileMatrix = createMapboxTileMatrix(
    mapboxTileBounds.x0,
    mapboxTileBounds.y0,
    mapboxTileBounds.x1,
    mapboxTileBounds.y1
  )

  // Filter visibleTiles to those that overlap with the Mapbox tile
  // Use the pre-computed tileBounds from TiledMode (in Mercator space)
  const overlappingZarrTiles: TileTuple[] = []
  for (const zarrTile of visibleTiles) {
    const zarrKey = tileToKey(zarrTile)
    const zarrBounds = zarrTileBounds?.[zarrKey]
    if (!zarrBounds) continue

    // Check overlap in Mercator space
    if (
      zarrBounds.x0 < mapboxTileBounds.x1 &&
      zarrBounds.x1 > mapboxTileBounds.x0 &&
      zarrBounds.y0 < mapboxTileBounds.y1 &&
      zarrBounds.y1 > mapboxTileBounds.y0
    ) {
      overlappingZarrTiles.push(zarrTile)
    }
  }

  if (overlappingZarrTiles.length === 0) {
    return false
  }

  // EPSG:4326 tiles use fragment shader reprojection (not wgs84 vertex shader)
  // The fragment shader inverts Mercator Y to get latitude for texture lookup
  const shaderProgram = renderer.getProgram(
    context.shaderData,
    customShaderConfig,
    true, // useMapboxGlobe
    false // useWgs84 - fragment shader reprojection for EPSG:4326
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
      mapboxTileBounds,
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
  mapboxTileBounds: { x0: number; y0: number; x1: number; y1: number },
  tileMatrix: Float32Array,
  tiledState: NonNullable<ReturnType<NonNullable<ZarrMode['getTiledState']>>>,
  context: RenderContext,
  xyLimits: XYLimits,
  datasetMaxZoom: number
): { rendered: boolean; missing: boolean } {
  const {
    tileCache,
    vertexArr,
    pixCoordArr,
    tileSize,
    tileBounds: zarrTileBoundsMap,
  } = tiledState
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
            mapboxTileBounds,
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

  // Get or compute bounds for this tile
  const zarrBounds = getTileMercatorBounds(
    z,
    tx,
    ty,
    zarrTileBoundsMap,
    xyLimits
  )

  // Compute overlap and texture coordinates
  const boundsAndTex = computeTileBoundsAndTexCoords(
    zarrBounds,
    mapboxTileBounds
  )
  if (!boundsAndTex) {
    return { rendered: false, missing }
  }

  const { overlap, latBounds, texScale, texOffset } = boundsAndTex

  const tileBoundsForRender = {
    [renderTileKey]: {
      x0: overlap.x0,
      y0: overlap.y0,
      x1: overlap.x1,
      y1: overlap.y1,
      latMin: latBounds.min,
      latMax: latBounds.max,
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
    tileBoundsForRender,
    customShaderConfig,
    true,
    undefined,
    {
      [renderTileKey]: {
        texScale,
        texOffset,
      },
    },
    tiledState.latIsAscending
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
  mapboxTileBounds: { x0: number; y0: number; x1: number; y1: number },
  tileMatrix: Float32Array,
  tiledState: NonNullable<ReturnType<NonNullable<ZarrMode['getTiledState']>>>,
  context: RenderContext,
  xyLimits: XYLimits
): boolean {
  const {
    tileCache,
    vertexArr,
    pixCoordArr,
    tileSize,
    tileBounds: zarrTileBoundsMap,
  } = tiledState
  const { colormapTexture, uniforms, customShaderConfig } = context

  const childTileTuple: TileTuple = [child.childZ, child.childX, child.childY]
  const childTileKey = tileToKey(childTileTuple)

  // Get or compute bounds for this child tile
  const childBounds = getTileMercatorBounds(
    child.childZ,
    child.childX,
    child.childY,
    zarrTileBoundsMap,
    xyLimits
  )

  // Compute overlap and texture coordinates
  const boundsAndTex = computeTileBoundsAndTexCoords(
    childBounds,
    mapboxTileBounds
  )
  if (!boundsAndTex) {
    return false
  }

  const { overlap, latBounds, texScale, texOffset } = boundsAndTex

  const childTileBoundsForRender = {
    [childTileKey]: {
      x0: overlap.x0,
      y0: overlap.y0,
      x1: overlap.x1,
      y1: overlap.y1,
      latMin: latBounds.min,
      latMax: latBounds.max,
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
    childTileBoundsForRender,
    customShaderConfig,
    true,
    undefined,
    {
      [childTileKey]: {
        texScale,
        texOffset,
      },
    },
    tiledState.latIsAscending
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
  const { tileCache, vertexArr, pixCoordArr, tileSize, tileBounds } = tiledState
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
    tileBoundsOverride,
    customShaderConfig,
    true,
    datasetMaxZoom
  )

  const tileHasData = tileCache.get(tileKey)?.data
  return !tileHasData
}
