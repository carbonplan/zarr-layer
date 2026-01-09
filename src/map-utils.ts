/**
 * @module map-utils
 *
 * Utility functions for custom layer integration.
 * Provides tile management, zoom level conversion,
 * coordinate transformations, and projection handling.
 * adapted from zarr-cesium/src/map-utils.ts
 */

import { MERCATOR_LAT_LIMIT, WEB_MERCATOR_EXTENT } from './constants'
import type { ProjectionData, ShaderData } from './shaders'
import type { MapLike } from './types'

export type TileTuple = [number, number, number]

export interface NormalizedExtent {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
}

export function normalizeGlobalExtent(
  xyLimits: {
    xMin: number
    xMax: number
    yMin: number
    yMax: number
  } | null
): NormalizedExtent {
  if (!xyLimits) {
    return { xMin: -180, xMax: 180, yMin: -90, yMax: 90 }
  }
  const extentX = xyLimits.xMax - xyLimits.xMin
  const extentY = xyLimits.yMax - xyLimits.yMin
  const isGlobal = extentX >= 350 && extentY >= 170
  return {
    xMin: isGlobal ? -180 : xyLimits.xMin,
    xMax: isGlobal ? 180 : xyLimits.xMax,
    yMin: isGlobal ? -90 : xyLimits.yMin,
    yMax: isGlobal ? 90 : xyLimits.yMax,
  }
}

export interface MercatorBounds {
  x0: number
  y0: number
  x1: number
  y1: number
  latMin?: number
  latMax?: number
  lonMin?: number
  lonMax?: number
}

/**
 * Converts longitude to tile X coordinate at a given zoom level.
 */
export function lonToTile(lon: number, zoom: number): number {
  return Math.floor(((lon + 180) / 360) * Math.pow(2, zoom))
}

/**
 * Converts latitude to tile Y coordinate at a given zoom level (Mercator).
 */
export function latToTileMercator(lat: number, zoom: number): number {
  const clamped = Math.max(
    -MERCATOR_LAT_LIMIT,
    Math.min(MERCATOR_LAT_LIMIT, lat)
  )
  const z2 = Math.pow(2, zoom)
  return Math.floor(
    ((1 -
      Math.log(
        Math.tan((clamped * Math.PI) / 180) +
          1 / Math.cos((clamped * Math.PI) / 180)
      ) /
        Math.PI) /
      2) *
      z2
  )
}

/**
 * Gets all tiles visible at a given zoom level within geographic bounds.
 * Handles world wrap-around by normalizing tile indices to valid range.
 * Handles antimeridian crossing when west > east.
 * @param zoom - Zoom level.
 * @param bounds - Geographic bounds [[west, south], [east, north]].
 * @returns Array of tile tuples [zoom, x, y].
 */
export function getTilesAtZoom(
  zoom: number,
  bounds: [[number, number], [number, number]]
): TileTuple[] {
  const [[west, south], [east, north]] = bounds
  const clampedSouth = Math.max(-MERCATOR_LAT_LIMIT, south)
  const clampedNorth = Math.min(MERCATOR_LAT_LIMIT, north)
  let nwX = lonToTile(west, zoom)
  let seX = lonToTile(east, zoom)
  const nwY = latToTileMercator(clampedNorth, zoom)
  const seY = latToTileMercator(clampedSouth, zoom)

  const maxTiles = Math.pow(2, zoom)
  const tiles: TileTuple[] = []
  const seenTiles = new Set<string>()

  if (nwX > seX) {
    seX += maxTiles
  }

  for (let x = nwX; x <= seX; x++) {
    const wrappedX = ((x % maxTiles) + maxTiles) % maxTiles
    for (let y = nwY; y <= seY; y++) {
      const clampedY = Math.max(0, Math.min(maxTiles - 1, y))
      const key = `${zoom},${wrappedX},${clampedY}`
      if (!seenTiles.has(key)) {
        seenTiles.add(key)
        tiles.push([zoom, wrappedX, clampedY])
      }
    }
  }

  return tiles
}

/**
 * Gets visible tiles for an equirectangular (EPSG:4326) pyramid.
 * Uses linear latitude spacing instead of Web Mercator's nonlinear spacing.
 * @param zoom - Zoom level.
 * @param bounds - Geographic bounds [[west, south], [east, north]].
 * @param xyLimits - Extent of the dataset in lon/lat (defaults to world).
 */
export function getTilesAtZoomEquirect(
  zoom: number,
  bounds: [[number, number], [number, number]],
  xyLimits: { xMin: number; xMax: number; yMin: number; yMax: number }
): TileTuple[] {
  const [[west, south], [east, north]] = bounds
  const { xMin, xMax, yMin, yMax } = normalizeGlobalExtent(xyLimits)
  const xSpan = xMax - xMin
  const ySpan = yMax - yMin
  const maxTiles = Math.pow(2, zoom)

  const lonToTile = (lon: number) =>
    Math.floor(((lon - xMin) / xSpan) * maxTiles)
  const latToTile = (lat: number) => {
    const clamped = Math.max(Math.min(lat, yMax), yMin)
    const norm = (yMax - clamped) / ySpan
    return Math.floor(norm * maxTiles)
  }

  let nwX = lonToTile(west)
  let seX = lonToTile(east)
  const nwY = latToTile(north)
  const seY = latToTile(south)

  const tiles: TileTuple[] = []
  const seenTiles = new Set<string>()

  if (nwX > seX) {
    seX += maxTiles
  }

  for (let x = nwX; x <= seX; x++) {
    const wrappedX = ((x % maxTiles) + maxTiles) % maxTiles
    for (let y = nwY; y <= seY; y++) {
      const clampedY = Math.max(0, Math.min(maxTiles - 1, y))
      const key = `${zoom},${wrappedX},${clampedY}`
      if (!seenTiles.has(key)) {
        seenTiles.add(key)
        tiles.push([zoom, wrappedX, clampedY])
      }
    }
  }

  return tiles
}

/**
 * Converts tile tuple to cache key string.
 * @param tile - Tile tuple [zoom, x, y].
 * @returns String key "z,x,y".
 */
export function tileToKey(tile: TileTuple): string {
  return tile.join(',')
}

/**
 * Parses a tile key back into a TileTuple [z, x, y].
 */
export function keyToTile(key: string): TileTuple {
  const parts = key.split(',').map(Number)
  return [parts[0], parts[1], parts[2]] as TileTuple
}

/**
 * Computes scale and shift parameters for positioning a tile in mercator coordinates.
 * Used in vertex shader to position tiles correctly on the map.
 *
 * Maps vertices from [-1, 1] clip space to [0, 1] mercator coordinate space.
 * For tile (z, x, y):
 *   - Left edge (vertex.x=-1) maps to x / 2^z
 *   - Right edge (vertex.x=1) maps to (x+1) / 2^z
 *   - Top edge (vertex.y=1) maps to y / 2^z (note: Y increases downward in web mercator)
 *   - Bottom edge (vertex.y=-1) maps to (y+1) / 2^z
 *
 * @param tile - Tile tuple [zoom, x, y].
 * @returns [scale, shiftX, shiftY] for vertex shader uniforms.
 */
export function tileToScale(tile: TileTuple): [number, number, number] {
  const [z, x, y] = tile
  const scale = 1 / 2 ** (z + 1)
  const shiftX = (2 * x + 1) * scale
  const shiftY = (2 * y + 1) * scale
  return [scale, shiftX, shiftY]
}

/**
 * Converts map zoom level to pyramid/multiscale level index.
 * Clamps zoom to valid range for the dataset.
 * @param zoom - Map zoom level.
 * @param maxLevelIndex - Maximum level index available in dataset.
 * @returns Pyramid level index (integer).
 */
export function zoomToLevel(zoom: number, maxLevelIndex: number): number {
  if (maxLevelIndex)
    return Math.min(Math.max(0, Math.floor(zoom)), maxLevelIndex)
  return Math.max(0, Math.floor(zoom))
}

/**
 * Parses the actual zoom number from a pyramid level path.
 * Handles pyramids that don't start at level 0 (e.g., levels ["2", "3", "4"]).
 * @param levelPath - The path string for the level (e.g., "0", "2", "data/level_3").
 * @param fallback - Fallback value if path can't be parsed as a number.
 * @returns The parsed zoom number.
 */
export function parseLevelZoom(
  levelPath: string,
  fallback: number = 0
): number {
  // Try to parse the path directly as a number
  const parsed = parseInt(levelPath, 10)
  if (!isNaN(parsed)) return parsed

  // Try to extract a number from the end of the path (e.g., "data/level_3" -> 3)
  const match = levelPath.match(/(\d+)$/)
  if (match) return parseInt(match[1], 10)

  return fallback
}

/**
 * Converts longitude in degrees to normalized Web Mercator X coordinate [0, 1].
 * Handles wraparound for longitudes outside -180 to 180 range.
 * @param lon - Longitude in degrees.
 * @returns Normalized mercator X coordinate.
 */
export function lonToMercatorNorm(lon: number): number {
  let normalizedLon = lon
  if (lon > 180) {
    normalizedLon = lon - 360
  } else if (lon < -180) {
    normalizedLon = lon + 360
  }
  return (normalizedLon + 180) / 360
}

/**
 * Converts latitude in degrees to normalized Web Mercator Y coordinate [0, 1].
 * Clamps latitude to valid Web Mercator range (±85.05112878°).
 * Note: Y=0 is at the north pole, Y=1 is at the south pole.
 * @param lat - Latitude in degrees.
 * @returns Normalized mercator Y coordinate.
 */
export function latToMercatorNorm(lat: number): number {
  const clamped = Math.max(
    -MERCATOR_LAT_LIMIT,
    Math.min(MERCATOR_LAT_LIMIT, lat)
  )
  return (
    (1 -
      Math.log(
        Math.tan((clamped * Math.PI) / 180) +
          1 / Math.cos((clamped * Math.PI) / 180)
      ) /
        Math.PI) /
    2
  )
}

export function mercatorNormToLat(mercY: number): number {
  const t = Math.PI * (1 - 2 * mercY)
  return (180 / Math.PI) * Math.atan(Math.sinh(t))
}

export function mercatorNormToLon(mercX: number): number {
  return mercX * 360 - 180
}

export interface GeoBounds {
  west: number
  east: number
  south: number
  north: number
}

export function mercatorTileToGeoBounds(
  z: number,
  x: number,
  y: number
): GeoBounds {
  const tilesPerSide = Math.pow(2, z)
  const mercX0 = x / tilesPerSide
  const mercX1 = (x + 1) / tilesPerSide
  const mercY0 = y / tilesPerSide
  const mercY1 = (y + 1) / tilesPerSide

  return {
    west: mercatorNormToLon(mercX0),
    east: mercatorNormToLon(mercX1),
    north: mercatorNormToLat(mercY0),
    south: mercatorNormToLat(mercY1),
  }
}

export interface XYLimits {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
}

export function getOverlapping4326Tiles(
  geoBounds: GeoBounds,
  xyLimits: XYLimits,
  pyramidLevel: number
): TileTuple[] {
  const tilesPerSide = Math.pow(2, pyramidLevel)
  const { xMin, xMax, yMin, yMax } = normalizeGlobalExtent(xyLimits)
  const xSpan = xMax - xMin
  const ySpan = yMax - yMin

  const lonToTileFloat = (lon: number) => ((lon - xMin) / xSpan) * tilesPerSide
  const latToTileFloat = (lat: number) => {
    const clamped = Math.max(Math.min(lat, yMax), yMin)
    return ((yMax - clamped) / ySpan) * tilesPerSide
  }

  const xTileMin = Math.floor(lonToTileFloat(geoBounds.west))
  const xTileMax = Math.floor(lonToTileFloat(geoBounds.east))
  const yTileMin = Math.floor(latToTileFloat(geoBounds.north))
  const yTileMax = Math.floor(latToTileFloat(geoBounds.south))

  const tiles: TileTuple[] = []
  for (let tx = xTileMin; tx <= xTileMax; tx++) {
    const wrappedX = ((tx % tilesPerSide) + tilesPerSide) % tilesPerSide
    for (let ty = yTileMin; ty <= yTileMax; ty++) {
      const clampedY = Math.max(0, Math.min(tilesPerSide - 1, ty))
      tiles.push([pyramidLevel, wrappedX, clampedY])
    }
  }
  return tiles
}

export function get4326TileGeoBounds(
  z: number,
  x: number,
  y: number,
  xyLimits: XYLimits
): GeoBounds {
  const tilesPerSide = Math.pow(2, z)
  const { xMin, xMax, yMin, yMax } = normalizeGlobalExtent(xyLimits)
  const xSpan = xMax - xMin
  const ySpan = yMax - yMin

  const west = xMin + (x / tilesPerSide) * xSpan
  const east = xMin + ((x + 1) / tilesPerSide) * xSpan
  const north = yMax - (y / tilesPerSide) * ySpan
  const south = yMax - ((y + 1) / tilesPerSide) * ySpan

  return { west, east, south, north }
}

interface TileCacheLike<T> {
  get(key: string): T | undefined
}

interface TileDataLike {
  data: Float32Array | null
}

export function findBestParentTile<T extends TileDataLike>(
  tileCache: TileCacheLike<T>,
  z: number,
  x: number,
  y: number
): {
  tile: T
  ancestorZ: number
  ancestorX: number
  ancestorY: number
} | null {
  let ancestorZ = z - 1
  let ancestorX = Math.floor(x / 2)
  let ancestorY = Math.floor(y / 2)

  while (ancestorZ >= 0) {
    const parentKey = tileToKey([ancestorZ, ancestorX, ancestorY])
    const parentTile = tileCache.get(parentKey)
    if (parentTile && parentTile.data) {
      return { tile: parentTile, ancestorZ, ancestorX, ancestorY }
    }
    ancestorZ--
    ancestorX = Math.floor(ancestorX / 2)
    ancestorY = Math.floor(ancestorY / 2)
  }
  return null
}

/**
 * Find cached child tiles that cover a target tile (for zoom-out fallback).
 * Walks DOWN the pyramid to find higher-resolution tiles that cover the target.
 * Limited to maxDepth levels to avoid excessive searching.
 *
 * @param tileCache - Cache to look up tiles
 * @param z - Target tile zoom level
 * @param x - Target tile x coordinate
 * @param y - Target tile y coordinate
 * @param datasetMaxZoom - Maximum zoom level in the dataset pyramid
 * @param maxDepth - Maximum levels to descend (default: 2)
 * @returns Array of child tiles with their coordinates, or null if no children found
 */
export function findBestChildTiles<T extends TileDataLike>(
  tileCache: TileCacheLike<T>,
  z: number,
  x: number,
  y: number,
  datasetMaxZoom: number,
  maxDepth: number = 2
): Array<{
  tile: T
  childZ: number
  childX: number
  childY: number
}> | null {
  let bestCoverage = 0
  let bestChildren: Array<{
    tile: T
    childZ: number
    childX: number
    childY: number
  }> = []

  // Search up to maxDepth levels down
  for (let depth = 1; depth <= maxDepth; depth++) {
    const childZ = z + depth
    if (childZ > datasetMaxZoom) break

    const scale = Math.pow(2, depth)
    const baseX = x * scale
    const baseY = y * scale
    const totalChildren = scale * scale

    const foundChildren: Array<{
      tile: T
      childZ: number
      childX: number
      childY: number
    }> = []

    // Check all child tiles at this level
    for (let dy = 0; dy < scale; dy++) {
      for (let dx = 0; dx < scale; dx++) {
        const childX = baseX + dx
        const childY = baseY + dy
        const childKey = tileToKey([childZ, childX, childY])
        const childTile = tileCache.get(childKey)

        if (childTile && childTile.data) {
          foundChildren.push({
            tile: childTile,
            childZ,
            childX,
            childY,
          })
        }
      }
    }

    const coverage = foundChildren.length / totalChildren

    // Full coverage at this level - return immediately
    if (coverage === 1) {
      return foundChildren
    }

    // Better coverage than previous best - update
    if (coverage > bestCoverage) {
      bestCoverage = coverage
      bestChildren = foundChildren
    }
  }

  return bestChildren.length > 0 ? bestChildren : null
}

/**
 * Check if a tile key has an overlapping ancestor in the rendered keys list.
 * Used to prevent rendering both a parent tile AND its children at the same location.
 *
 * @param childKey - The child tile key to check
 * @param renderedKeys - Array of tile keys currently being rendered
 * @returns The overlapping ancestor key if found, null otherwise
 */
export function getOverlappingAncestor(
  childKey: string,
  renderedKeys: string[]
): string | null {
  const [childX, childY, childZ] = keyToTile(childKey)

  for (const parentKey of renderedKeys) {
    const [parentX, parentY, parentZ] = keyToTile(parentKey)

    // Can't be an ancestor if at same or higher zoom level
    if (childZ <= parentZ) continue

    // Check if parent covers the child
    const factor = Math.pow(2, childZ - parentZ)
    const coversX = Math.floor(childX / factor) === parentX
    const coversY = Math.floor(childY / factor) === parentY

    if (coversX && coversY) {
      return parentKey
    }
  }

  return null
}

/**
 * Converts geographic bounds to normalized Web Mercator bounds [0, 1].
 * Handles both EPSG:4326 (lat/lon) and EPSG:3857 (already mercator) coordinate systems.
 * @param xyLimits - Geographic bounds { xMin, xMax, yMin, yMax }.
 * @param crs - Coordinate reference system ('EPSG:4326' or 'EPSG:3857').
 * @returns Normalized mercator bounds { x0, y0, x1, y1 }.
 */
export function boundsToMercatorNorm(
  xyLimits: { xMin: number; xMax: number; yMin: number; yMax: number },
  crs: 'EPSG:4326' | 'EPSG:3857' | null
): MercatorBounds {
  if (crs === 'EPSG:3857') {
    return {
      x0: (xyLimits.xMin + WEB_MERCATOR_EXTENT) / (2 * WEB_MERCATOR_EXTENT),
      y0: (WEB_MERCATOR_EXTENT - xyLimits.yMax) / (2 * WEB_MERCATOR_EXTENT),
      x1: (xyLimits.xMax + WEB_MERCATOR_EXTENT) / (2 * WEB_MERCATOR_EXTENT),
      y1: (WEB_MERCATOR_EXTENT - xyLimits.yMin) / (2 * WEB_MERCATOR_EXTENT),
    }
  }

  let yMin = xyLimits.yMin
  let yMax = xyLimits.yMax
  if (yMin > yMax) {
    ;[yMin, yMax] = [yMax, yMin]
  }

  return {
    x0: lonToMercatorNorm(xyLimits.xMin),
    y0: latToMercatorNorm(yMax),
    x1: lonToMercatorNorm(xyLimits.xMax),
    y1: latToMercatorNorm(yMin),
    // Preserve original latitude bounds for equirectangular data so callers
    // can perform linear-latitude calculations when needed (e.g. queries).
    latMin: yMin,
    latMax: yMax,
  }
}

// === Untiled mode utilities ===

/**
 * Convert a geographic coordinate to an array index.
 * Used for mapping viewport bounds to array pixel coordinates.
 * @param geo - Geographic coordinate value (lon or lat).
 * @param geoMin - Minimum geographic extent.
 * @param geoMax - Maximum geographic extent.
 * @param arraySize - Size of the array in this dimension.
 * @returns Array index (integer).
 */
export function geoToArrayIndex(
  geo: number,
  geoMin: number,
  geoMax: number,
  arraySize: number
): number {
  const normalized = (geo - geoMin) / (geoMax - geoMin)
  return Math.floor(
    Math.max(0, Math.min(arraySize - 1, normalized * arraySize))
  )
}

/**
 * Convert an array index to a geographic coordinate.
 * Used for computing chunk bounds in geographic space.
 * @param index - Array index.
 * @param geoMin - Minimum geographic extent.
 * @param geoMax - Maximum geographic extent.
 * @param arraySize - Size of the array in this dimension.
 * @returns Geographic coordinate value.
 */
export function arrayIndexToGeo(
  index: number,
  geoMin: number,
  geoMax: number,
  arraySize: number
): number {
  return geoMin + (index / arraySize) * (geoMax - geoMin)
}

/**
 * Convert an array index to a chunk index.
 * @param arrayIndex - Array index.
 * @param chunkSize - Size of each chunk.
 * @returns Chunk index.
 */
export function arrayIndexToChunkIndex(
  arrayIndex: number,
  chunkSize: number
): number {
  return Math.floor(arrayIndex / chunkSize)
}

/**
 * Get the array index range covered by a chunk.
 * @param chunkIndex - Chunk index.
 * @param chunkSize - Size of each chunk.
 * @param arraySize - Total size of the array.
 * @returns [startIndex, endIndex] (exclusive end).
 */
export function chunkIndexToArrayRange(
  chunkIndex: number,
  chunkSize: number,
  arraySize: number
): [number, number] {
  const startIndex = chunkIndex * chunkSize
  const endIndex = Math.min((chunkIndex + 1) * chunkSize, arraySize)
  return [startIndex, endIndex]
}

// === Texture coordinate utilities ===

/** Latitude bounds in degrees */
export interface LatBounds {
  latMin: number
  latMax: number
  lonMin?: number
  lonMax?: number
}

/**
 * Flip texture V coordinates (for EPSG:3857 data with latIsAscending=true).
 */
export function flipTexCoordV(texCoords: Float32Array): Float32Array {
  const flipped = new Float32Array(texCoords.length)
  for (let i = 0; i < texCoords.length; i += 2) {
    flipped[i] = texCoords[i]
    flipped[i + 1] = 1 - texCoords[i + 1]
  }
  return flipped
}

// === Projection utilities ===

/**
 * Detects if the given projection is a globe projection.
 * Works with both Mapbox (projection.name) and MapLibre (projection.type).
 */
export function isGlobeProjection(
  projection:
    | { type?: string | readonly unknown[]; name?: string }
    | null
    | undefined
): boolean {
  return projection?.type === 'globe' || projection?.name === 'globe'
}

interface ProjectionResolution {
  matrix: number[] | Float32Array | Float64Array | null
  shaderData?: ShaderData
  projectionData?: ProjectionData
  mapboxGlobe?:
    | {
        projection: { name: string }
        globeToMercatorMatrix: number[] | Float32Array | Float64Array
        transition: number
      }
    | undefined
}

export function resolveProjectionParams(
  params: unknown,
  projection?: { name: string },
  projectionToMercatorMatrix?: number[] | Float32Array | Float64Array,
  projectionToMercatorTransition?: number
): ProjectionResolution {
  type MatrixLike = number[] | Float32Array | Float64Array
  type ProjectionParams = {
    shaderData?: ShaderData
    defaultProjectionData?: {
      mainMatrix?: MatrixLike
      fallbackMatrix?: MatrixLike
      tileMercatorCoords?: number[]
      clippingPlane?: number[]
      projectionTransition?: number
    }
    modelViewProjectionMatrix?: MatrixLike
    projectionMatrix?: MatrixLike
  }

  const paramsObj =
    params &&
    typeof params === 'object' &&
    !Array.isArray(params) &&
    !ArrayBuffer.isView(params)
      ? (params as ProjectionParams)
      : null

  const shaderData = paramsObj?.shaderData
  let projectionData: ProjectionData | undefined
  const defaultProj = paramsObj?.defaultProjectionData
  if (
    defaultProj &&
    defaultProj.mainMatrix &&
    defaultProj.fallbackMatrix &&
    defaultProj.tileMercatorCoords &&
    defaultProj.clippingPlane &&
    typeof defaultProj.projectionTransition === 'number'
  ) {
    projectionData = {
      mainMatrix: defaultProj.mainMatrix,
      fallbackMatrix: defaultProj.fallbackMatrix,
      tileMercatorCoords: defaultProj.tileMercatorCoords as [
        number,
        number,
        number,
        number
      ],
      clippingPlane: defaultProj.clippingPlane as [
        number,
        number,
        number,
        number
      ],
      projectionTransition: defaultProj.projectionTransition,
    }
  }
  let matrix: number[] | Float32Array | Float64Array | null = null
  if (projectionData?.mainMatrix && projectionData.mainMatrix.length) {
    matrix = projectionData.mainMatrix
  } else if (
    Array.isArray(params) ||
    params instanceof Float32Array ||
    params instanceof Float64Array
  ) {
    matrix = params as number[] | Float32Array | Float64Array
  } else if (paramsObj?.modelViewProjectionMatrix) {
    matrix = paramsObj.modelViewProjectionMatrix
  } else if (paramsObj?.projectionMatrix) {
    matrix = paramsObj.projectionMatrix
  }

  const mapboxGlobe =
    projection && projectionToMercatorMatrix !== undefined
      ? {
          projection,
          globeToMercatorMatrix: projectionToMercatorMatrix,
          transition:
            typeof projectionToMercatorTransition === 'number'
              ? projectionToMercatorTransition
              : 0,
        }
      : undefined

  return { matrix, shaderData, projectionData, mapboxGlobe }
}

export function computeWorldOffsets(
  map: MapLike | null,
  isGlobe: boolean
): number[] {
  if (!map) return [0]

  const bounds = map.getBounds ? map.getBounds() : null
  if (!bounds) return [0]

  const renderWorldCopies =
    typeof map.getRenderWorldCopies === 'function'
      ? map.getRenderWorldCopies()
      : true
  if (isGlobe || !renderWorldCopies) return [0]

  const west = bounds.getWest()
  const east = bounds.getEast()

  let effectiveEast = east
  if (west > east) {
    effectiveEast = east + 360
  }

  const minWorld = Math.floor((west + 180) / 360)
  const maxWorld = Math.floor((effectiveEast + 180) / 360)

  const worldOffsets: number[] = []
  for (let i = minWorld; i <= maxWorld; i++) {
    worldOffsets.push(i)
  }
  return worldOffsets.length > 0 ? worldOffsets : [0]
}
