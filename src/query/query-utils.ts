/**
 * @module query-utils
 *
 * Utility functions for query coordinate transformations,
 * mercator corrections, and point-in-polygon tests.
 */

import {
  getTilesAtZoom,
  getTilesAtZoomEquirect,
  latToMercatorNorm,
  lonToMercatorNorm,
  mercatorNormToLat,
  mercatorNormToLon,
  lonToTile,
  latToTileMercator,
  type TileTuple,
  type XYLimits,
  type MercatorBounds,
} from '../map-utils'
import type { Bounds, CRS } from '../types'
import type { BoundingBox, QueryGeometry } from './types'
import {
  createWGS84ToSourceTransformer,
  sourceCRSToPixel,
  pixelToSourceCRS,
} from '../projection-utils'

/** Cached transformer type for reuse across multiple pixelToLatLon calls */
export type CachedTransformer = ReturnType<
  typeof createWGS84ToSourceTransformer
>

/**
 * Converts pixel coordinates to lat/lon.
 * Handles all CRS types including proj4 reprojection.
 * This is the canonical function for pixel → geographic conversion in queries.
 *
 * @param cachedTransformer - Optional pre-created transformer for performance.
 *   When processing many pixels, create once and reuse to avoid repeated proj4 init.
 */
export function pixelToLatLon(
  x: number,
  y: number,
  bounds: MercatorBounds,
  width: number,
  height: number,
  crs: CRS,
  latIsAscending?: boolean,
  proj4def?: string | null,
  sourceBounds?: Bounds | null,
  cachedTransformer?: CachedTransformer,
  centerPixel: boolean = true
): { lat: number; lon: number } {
  // For proj4, convert pixel → source CRS → WGS84
  if (proj4def && sourceBounds) {
    const transformer =
      cachedTransformer ?? createWGS84ToSourceTransformer(proj4def)

    // pixelToSourceCRS uses edge-based model: pixel 0 → xMin, pixel width → xMax
    // For pixel centers, pass pixel + 0.5; for edges, pass pixel directly
    const px = centerPixel ? x + 0.5 : x
    const py = centerPixel ? y + 0.5 : y
    const [srcX, srcY] = pixelToSourceCRS(
      px,
      py,
      sourceBounds,
      width,
      height,
      latIsAscending
    )

    const [lon, lat] = transformer.inverse(srcX, srcY)
    return { lat, lon }
  }

  // Standard CRS handling
  // Guard against zero-dimension cases
  // centerPixel=true: return center of pixel (x+0.5), centerPixel=false: return corner (x)
  const xFrac = width <= 1 ? 0.5 : centerPixel ? (x + 0.5) / width : x / width
  const yFrac =
    height <= 1 ? 0.5 : centerPixel ? (y + 0.5) / height : y / height
  const mercX = bounds.x0 + xFrac * (bounds.x1 - bounds.x0)
  const mercY = bounds.y0 + yFrac * (bounds.y1 - bounds.y0)

  const lon = mercatorNormToLon(mercX)

  // Guard against zero-range bounds
  const yRange = bounds.y1 - bounds.y0
  const yNorm = yRange === 0 ? 0.5 : (mercY - bounds.y0) / yRange

  const lat =
    crs === 'EPSG:4326' &&
    bounds.latMin !== undefined &&
    bounds.latMax !== undefined
      ? latIsAscending
        ? bounds.latMin + yNorm * (bounds.latMax - bounds.latMin)
        : bounds.latMax - yNorm * (bounds.latMax - bounds.latMin)
      : mercatorNormToLat(mercY)

  return { lat, lon }
}

/**
 * Converts latitude to tile Y coordinate at a given zoom level (Equirectangular/EPSG:4326).
 */
export function latToTileEquirect(
  lat: number,
  zoom: number,
  xyLimits: XYLimits
): number {
  const { yMin, yMax } = xyLimits
  const z2 = Math.pow(2, zoom)
  const clamped = Math.max(Math.min(lat, yMax), yMin)
  const norm = (yMax - clamped) / (yMax - yMin)
  return Math.floor(norm * z2)
}

/**
 * Converts longitude to tile X coordinate at a given zoom level (Equirectangular/EPSG:4326).
 */
export function lonToTileEquirect(
  lon: number,
  zoom: number,
  xyLimits: XYLimits
): number {
  const { xMin, xMax } = xyLimits
  const z2 = Math.pow(2, zoom)
  const clamped = Math.max(Math.min(lon, xMax), xMin)
  const norm = (clamped - xMin) / (xMax - xMin)
  return Math.floor(norm * z2)
}

/**
 * Gets the tile coordinates for a geographic point.
 */
export function geoToTile(
  lng: number,
  lat: number,
  zoom: number,
  crs: CRS,
  xyLimits: XYLimits
): TileTuple {
  if (crs === 'EPSG:4326') {
    return [
      zoom,
      lonToTileEquirect(lng, zoom, xyLimits),
      latToTileEquirect(lat, zoom, xyLimits),
    ]
  }
  return [zoom, lonToTile(lng, zoom), latToTileMercator(lat, zoom)]
}

/**
 * Computes fractional position within a tile for a geographic point.
 * Returns values in [0, 1] representing position within the tile.
 */
export function geoToTileFraction(
  lng: number,
  lat: number,
  tile: TileTuple,
  crs: CRS,
  xyLimits: XYLimits
): { fracX: number; fracY: number } {
  const [z, x, y] = tile
  const z2 = Math.pow(2, z)

  if (crs === 'EPSG:4326') {
    const { xMin, xMax, yMin, yMax } = xyLimits
    const xSpan = xMax - xMin
    const ySpan = yMax - yMin

    const globalFracX = (lng - xMin) / xSpan
    const globalFracY = (yMax - lat) / ySpan

    const fracX = globalFracX * z2 - x
    const fracY = globalFracY * z2 - y

    return { fracX, fracY }
  }

  const globalFracX = lonToMercatorNorm(lng)
  const sin = Math.sin((lat * Math.PI) / 180)
  const globalFracY = 0.5 - (0.25 * Math.log((1 + sin) / (1 - sin))) / Math.PI

  const fracX = globalFracX * z2 - x
  const fracY = globalFracY * z2 - y

  return { fracX, fracY }
}

/**
 * Converts tile pixel position to geographic coordinates.
 */
export function tilePixelToLatLon(
  tile: TileTuple,
  pixelX: number,
  pixelY: number,
  tileSize: number,
  crs: CRS,
  xyLimits: XYLimits
): { lat: number; lon: number } {
  const [z, x, y] = tile
  const z2 = Math.pow(2, z)

  const fracX = (x + pixelX / tileSize) / z2
  const fracY = (y + pixelY / tileSize) / z2

  if (crs === 'EPSG:4326') {
    const { xMin, xMax, yMin, yMax } = xyLimits
    const lon = xMin + fracX * (xMax - xMin)
    const lat = yMax - fracY * (yMax - yMin)
    return { lat, lon }
  }

  // EPSG:3857 - invert mercator projection
  const lon = fracX * 360 - 180
  const y2 = 180 - fracY * 360
  const lat = (360 / Math.PI) * Math.atan(Math.exp((y2 * Math.PI) / 180)) - 90

  return { lat, lon }
}

/**
 * Computes bounding box from GeoJSON geometry.
 */
export function computeBoundingBox(geometry: QueryGeometry): BoundingBox {
  let west = Infinity
  let east = -Infinity
  let south = Infinity
  let north = -Infinity

  if (geometry.type === 'Point') {
    const [lon, lat] = geometry.coordinates
    return { west: lon, east: lon, south: lat, north: lat }
  }

  const processRing = (ring: number[][]) => {
    for (const [lon, lat] of ring) {
      if (lon < west) west = lon
      if (lon > east) east = lon
      if (lat < south) south = lat
      if (lat > north) north = lat
    }
  }

  if (geometry.type === 'Polygon') {
    geometry.coordinates.forEach(processRing)
  } else {
    geometry.coordinates.forEach((polygon) => polygon.forEach(processRing))
  }

  return { west, east, south, north }
}

/**
 * Computes pixel bounds from a geometry's bounding box.
 * Returns the pixel range [minX, maxX, minY, maxY] that covers the geometry.
 * Supports custom projections via proj4.
 */
export function computePixelBoundsFromGeometry(
  geometry: QueryGeometry,
  bounds: MercatorBounds,
  width: number,
  height: number,
  crs: CRS,
  latIsAscending?: boolean,
  proj4def?: string | null,
  sourceBounds?: Bounds | null,
  cachedTransformer?: CachedTransformer
): { minX: number; maxX: number; minY: number; maxY: number } | null {
  const bbox = computeBoundingBox(geometry)

  // If proj4 is provided, use proj4 to transform bbox
  if (proj4def && sourceBounds) {
    const transformer =
      cachedTransformer ?? createWGS84ToSourceTransformer(proj4def)

    // Sample points along bbox edges to capture curved projections
    // (corners alone can miss extrema for conic/polar projections)
    const numSamples = 5
    const samplePoints: [number, number][] = []

    for (let i = 0; i <= numSamples; i++) {
      const t = i / numSamples
      const lon = bbox.west + t * (bbox.east - bbox.west)
      const lat = bbox.south + t * (bbox.north - bbox.south)
      samplePoints.push([lon, bbox.south]) // Bottom edge
      samplePoints.push([lon, bbox.north]) // Top edge
      samplePoints.push([bbox.west, lat]) // Left edge
      samplePoints.push([bbox.east, lat]) // Right edge
    }

    let minX = Infinity
    let maxX = -Infinity
    let minY = Infinity
    let maxY = -Infinity

    for (const [lon, lat] of samplePoints) {
      const [srcX, srcY] = transformer.forward(lon, lat)
      if (!isFinite(srcX) || !isFinite(srcY)) continue

      const [xPixel, yPixel] = sourceCRSToPixel(
        srcX,
        srcY,
        sourceBounds,
        width,
        height,
        latIsAscending
      )
      minX = Math.min(minX, xPixel)
      maxX = Math.max(maxX, xPixel)
      minY = Math.min(minY, yPixel)
      maxY = Math.max(maxY, yPixel)
    }

    // Check if any valid samples were found
    if (
      !isFinite(minX) ||
      !isFinite(maxX) ||
      !isFinite(minY) ||
      !isFinite(maxY)
    ) {
      return null
    }

    // Clamp to valid range
    // Use floor + 1 to ensure integer maxX/maxY values include that pixel
    const xStart = Math.max(0, Math.floor(minX))
    const xEnd = Math.min(width, Math.max(Math.floor(maxX) + 1, xStart + 1))
    const yStart = Math.max(0, Math.floor(minY))
    const yEnd = Math.min(height, Math.max(Math.floor(maxY) + 1, yStart + 1))

    if (xEnd <= xStart || yEnd <= yStart) return null

    return { minX: xStart, maxX: xEnd, minY: yStart, maxY: yEnd }
  }

  // Convert bbox corners to mercator normalized coords
  const polyX0 = lonToMercatorNorm(bbox.west)
  const polyX1 = lonToMercatorNorm(bbox.east)
  const polyY0 = latToMercatorNorm(bbox.north)
  const polyY1 = latToMercatorNorm(bbox.south)

  // Compute overlap with image bounds
  const overlapX0 = Math.max(bounds.x0, Math.min(polyX0, polyX1))
  const overlapX1 = Math.min(bounds.x1, Math.max(polyX0, polyX1))

  let xStart: number
  let xEnd: number
  let yStart: number
  let yEnd: number

  if (
    crs === 'EPSG:4326' &&
    bounds.latMin !== undefined &&
    bounds.latMax !== undefined
  ) {
    // For equirectangular data, compute Y overlap in linear latitude space
    const latMax = bounds.latMax
    const latMin = bounds.latMin
    const clampedNorth = Math.min(Math.max(bbox.north, latMin), latMax)
    const clampedSouth = Math.min(Math.max(bbox.south, latMin), latMax)

    const latRange = latMax - latMin
    if (latRange === 0) return null

    const toFrac = (latVal: number) =>
      latIsAscending
        ? (latVal - latMin) / latRange
        : (latMax - latVal) / latRange
    const yStartFracRaw = toFrac(clampedNorth)
    const yEndFracRaw = toFrac(clampedSouth)
    const yFracMin = Math.min(yStartFracRaw, yEndFracRaw)
    const yFracMax = Math.max(yStartFracRaw, yEndFracRaw)

    if (overlapX1 < overlapX0 || yFracMax < yFracMin) return null

    const minX = ((overlapX0 - bounds.x0) / (bounds.x1 - bounds.x0)) * width
    const maxX = ((overlapX1 - bounds.x0) / (bounds.x1 - bounds.x0)) * width

    xStart = Math.max(0, Math.floor(minX))
    xEnd = Math.min(width, Math.max(Math.ceil(maxX), xStart + 1))
    yStart = Math.max(0, Math.floor(yFracMin * height))
    yEnd = Math.min(height, Math.max(Math.ceil(yFracMax * height), yStart + 1))
  } else {
    const overlapY0 = Math.max(bounds.y0, Math.min(polyY0, polyY1))
    const overlapY1 = Math.min(bounds.y1, Math.max(polyY0, polyY1))

    if (overlapX1 < overlapX0 || overlapY1 < overlapY0) return null

    const minX = ((overlapX0 - bounds.x0) / (bounds.x1 - bounds.x0)) * width
    const maxX = ((overlapX1 - bounds.x0) / (bounds.x1 - bounds.x0)) * width
    const minY = ((overlapY0 - bounds.y0) / (bounds.y1 - bounds.y0)) * height
    const maxY = ((overlapY1 - bounds.y0) / (bounds.y1 - bounds.y0)) * height

    xStart = Math.max(0, Math.floor(minX))
    xEnd = Math.min(width, Math.max(Math.ceil(maxX), xStart + 1))
    yStart = Math.max(0, Math.floor(minY))
    yEnd = Math.min(height, Math.max(Math.ceil(maxY), yStart + 1))
  }

  if (xEnd <= xStart || yEnd <= yStart) return null

  return { minX: xStart, maxX: xEnd, minY: yStart, maxY: yEnd }
}

import {
  DEFAULT_QUERY_DENSIFY_MAX_ERROR,
  MERCATOR_LAT_LIMIT,
} from '../constants'

/** Max recursion depth for adaptive subdivision */
const DENSIFY_MAX_DEPTH = 10

/**
 * Densify a ring by adaptively subdividing edges until the pixel-space error
 * is below DEFAULT_QUERY_DENSIFY_MAX_ERROR. For each edge, the midpoint is interpolated in
 * source coordinates (lon/lat), transformed to pixel space, and compared to
 * the straight-line midpoint in pixel space. If the deviation exceeds the
 * threshold, the edge is recursively split.
 *
 * This matches the adaptive mesh reprojection precision (0.125px) so query
 * polygon boundaries align with rendered pixel boundaries.
 */
function densifyAndTransformRing(
  ring: number[][],
  transformVertex: (lon: number, lat: number) => [number, number]
): number[][] {
  const result: number[][] = []

  function subdivide(
    lon0: number,
    lat0: number,
    px0: [number, number],
    lon1: number,
    lat1: number,
    px1: [number, number],
    depth: number
  ) {
    if (depth >= DENSIFY_MAX_DEPTH) return

    // Midpoint in source space
    const lonM = (lon0 + lon1) * 0.5
    const latM = (lat0 + lat1) * 0.5
    const pxM = transformVertex(lonM, latM)
    if (!isFinite(pxM[0]) || !isFinite(pxM[1])) return

    // Straight-line midpoint in pixel space
    const expectedX = (px0[0] + px1[0]) * 0.5
    const expectedY = (px0[1] + px1[1]) * 0.5

    // Error: distance from true projected midpoint to straight-line midpoint
    const dx = pxM[0] - expectedX
    const dy = pxM[1] - expectedY
    const error = dx * dx + dy * dy // compare squared to avoid sqrt

    if (
      error >
      DEFAULT_QUERY_DENSIFY_MAX_ERROR * DEFAULT_QUERY_DENSIFY_MAX_ERROR
    ) {
      subdivide(lon0, lat0, px0, lonM, latM, pxM, depth + 1)
      result.push(pxM as number[])
      subdivide(lonM, latM, pxM, lon1, lat1, px1, depth + 1)
    }
  }

  for (let i = 0; i < ring.length - 1; i++) {
    const [lon0, lat0] = ring[i]
    const [lon1, lat1] = ring[i + 1]
    const px0 = transformVertex(lon0, lat0)
    const px1 = transformVertex(lon1, lat1)

    if (isFinite(px0[0]) && isFinite(px0[1])) {
      result.push(px0 as number[])
    }
    if (
      isFinite(px0[0]) &&
      isFinite(px0[1]) &&
      isFinite(px1[0]) &&
      isFinite(px1[1])
    ) {
      subdivide(lon0, lat0, px0, lon1, lat1, px1, 0)
    }
  }

  // Close ring (only if first vertex was valid)
  if (result.length > 0 && isFinite(result[0][0]) && isFinite(result[0][1])) {
    result.push([result[0][0], result[0][1]])
  }
  return result
}

/**
 * Transform a query geometry from WGS84 lon/lat into pixel-space coordinates.
 * For proj4 projections: forward-transforms vertices, then converts source CRS → pixel.
 * For standard CRS: uses mercator/equirect math → pixel.
 * Densifies edges to preserve curvature under nonlinear projections.
 *
 * Returns a geometry with the same GeoJSON ring structure but in pixel coordinates,
 * suitable for use with scanline rasterization.
 */
export function transformGeometryToPixelSpace(
  geometry: QueryGeometry,
  bounds: MercatorBounds,
  width: number,
  height: number,
  crs: CRS,
  latIsAscending?: boolean,
  proj4def?: string | null,
  sourceBounds?: Bounds | null,
  cachedTransformer?: CachedTransformer
): QueryGeometry | null {
  if (geometry.type === 'Point') {
    const [lon, lat] = geometry.coordinates
    const px = lonLatToPixel(
      lon,
      lat,
      bounds,
      width,
      height,
      crs,
      latIsAscending,
      proj4def,
      sourceBounds,
      cachedTransformer
    )
    if (!px) return null
    return { type: 'Point', coordinates: [px[0], px[1]] }
  }

  // Build the vertex transform function
  const transformVertex = (lon: number, lat: number): [number, number] => {
    const px = lonLatToPixel(
      lon,
      lat,
      bounds,
      width,
      height,
      crs,
      latIsAscending,
      proj4def,
      sourceBounds,
      cachedTransformer
    )
    return px ?? [NaN, NaN]
  }

  // Densify for any nonlinear CRS. EPSG:3857 uses latToMercatorNorm (nonlinear in Y).
  // EPSG:4326 with lat bounds is linear and doesn't need densification.
  const isLinear4326 =
    crs === 'EPSG:4326' &&
    bounds.latMin !== undefined &&
    bounds.latMax !== undefined
  const needsDensification =
    !!proj4def || (!isLinear4326 && crs !== 'EPSG:4326')

  const transformRing = (ring: number[][]): number[][] => {
    if (needsDensification) {
      return densifyAndTransformRing(ring, transformVertex)
    }
    // For linear projections, just transform vertices directly
    const result: number[][] = []
    for (const [lon, lat] of ring) {
      const pt = transformVertex(lon, lat)
      if (isFinite(pt[0]) && isFinite(pt[1])) {
        result.push(pt as number[])
      }
    }
    // Ensure ring is closed
    if (
      result.length > 1 &&
      (result[0][0] !== result[result.length - 1][0] ||
        result[0][1] !== result[result.length - 1][1])
    ) {
      result.push([result[0][0], result[0][1]])
    }
    return result
  }

  if (geometry.type === 'Polygon') {
    const coords = geometry.coordinates.map(transformRing)
    if (coords[0].length < 4) return null // Need at least a triangle
    return { type: 'Polygon', coordinates: coords }
  }

  // MultiPolygon
  const coords = geometry.coordinates.map((polygon) =>
    polygon.map(transformRing)
  )
  // Filter out degenerate polygons
  const valid = coords.filter((poly) => poly[0].length >= 4)
  if (valid.length === 0) return null
  return { type: 'MultiPolygon', coordinates: valid }
}

/**
 * Transform a query geometry from WGS84 lon/lat into tile-pixel coordinates.
 * For EPSG:3857 the lat→Y mapping is nonlinear, so edges are densified.
 * For EPSG:4326 the mapping is linear and vertices are transformed directly.
 */
export function transformGeometryToTilePixelSpace(
  geometry: QueryGeometry,
  tile: TileTuple,
  tileSize: number,
  crs: CRS,
  xyLimits: XYLimits
): QueryGeometry | null {
  const transformVertex = (lon: number, lat: number): [number, number] => {
    // Clamp latitude to Mercator limits to avoid infinities at the poles
    const clampedLat =
      crs !== 'EPSG:4326'
        ? Math.max(-MERCATOR_LAT_LIMIT, Math.min(MERCATOR_LAT_LIMIT, lat))
        : lat
    const { fracX, fracY } = geoToTileFraction(
      lon,
      clampedLat,
      tile,
      crs,
      xyLimits
    )
    return [fracX * tileSize, fracY * tileSize]
  }

  if (geometry.type === 'Point') {
    const [lon, lat] = geometry.coordinates
    const pt = transformVertex(lon, lat)
    if (!isFinite(pt[0]) || !isFinite(pt[1])) return null
    return { type: 'Point', coordinates: [pt[0], pt[1]] }
  }

  // EPSG:3857 is nonlinear in Y; EPSG:4326 is linear
  const needsDensification = crs !== 'EPSG:4326'

  const transformRing = (ring: number[][]): number[][] => {
    if (needsDensification) {
      return densifyAndTransformRing(ring, transformVertex)
    }
    const result: number[][] = []
    for (const [lon, lat] of ring) {
      const pt = transformVertex(lon, lat)
      if (isFinite(pt[0]) && isFinite(pt[1])) {
        result.push(pt as number[])
      }
    }
    if (
      result.length > 1 &&
      (result[0][0] !== result[result.length - 1][0] ||
        result[0][1] !== result[result.length - 1][1])
    ) {
      result.push([result[0][0], result[0][1]])
    }
    return result
  }

  if (geometry.type === 'Polygon') {
    const coords = geometry.coordinates.map(transformRing)
    if (coords[0].length < 4) return null
    return { type: 'Polygon', coordinates: coords }
  }

  const coords = geometry.coordinates.map((polygon) =>
    polygon.map(transformRing)
  )
  const valid = coords.filter((poly) => poly[0].length >= 4)
  if (valid.length === 0) return null
  return { type: 'MultiPolygon', coordinates: valid }
}

/**
 * Convert a single lon/lat point to pixel coordinates.
 * Handles proj4, EPSG:4326, and EPSG:3857.
 */
function lonLatToPixel(
  lon: number,
  lat: number,
  bounds: MercatorBounds,
  width: number,
  height: number,
  crs: CRS,
  latIsAscending?: boolean,
  proj4def?: string | null,
  sourceBounds?: Bounds | null,
  cachedTransformer?: CachedTransformer
): [number, number] | null {
  if (proj4def && sourceBounds) {
    const transformer =
      cachedTransformer ?? createWGS84ToSourceTransformer(proj4def)
    const [srcX, srcY] = transformer.forward(lon, lat)
    if (!isFinite(srcX) || !isFinite(srcY)) return null
    return sourceCRSToPixel(
      srcX,
      srcY,
      sourceBounds,
      width,
      height,
      latIsAscending
    )
  }

  // Standard CRS: convert to mercator normalized, then to pixel.
  // No bounds clamping — polygon vertices can legitimately lie far outside
  // the raster extent (e.g. when the polygon fully contains a small raster).
  const normX = lonToMercatorNorm(lon)
  const xFrac = (normX - bounds.x0) / (bounds.x1 - bounds.x0)

  let yFrac: number
  if (
    crs === 'EPSG:4326' &&
    bounds.latMin !== undefined &&
    bounds.latMax !== undefined
  ) {
    const latRange = bounds.latMax - bounds.latMin
    if (latRange === 0) return null
    yFrac = latIsAscending
      ? (lat - bounds.latMin) / latRange
      : (bounds.latMax - lat) / latRange
  } else {
    const normY = latToMercatorNorm(lat)
    yFrac = (normY - bounds.y0) / (bounds.y1 - bounds.y0)
  }

  return [xFrac * width, yFrac * height]
}

/**
 * Build a scanline table for a single polygon (outer ring + holes).
 * Returns a Map from integer Y to sorted array of X-intersection values.
 */
function buildScanlineTableForRings(
  rings: number[][][],
  yStart: number,
  yEnd: number
): Map<number, number[]> {
  const table = new Map<number, number[]>()

  for (const ring of rings) {
    for (let i = 0; i < ring.length - 1; i++) {
      const x0 = ring[i][0]
      const y0 = ring[i][1]
      const x1 = ring[i + 1][0]
      const y1 = ring[i + 1][1]

      if (y0 === y1) continue

      const edgeYMin = Math.min(y0, y1)
      const edgeYMax = Math.max(y0, y1)
      const scanYMin = Math.max(yStart, Math.floor(edgeYMin + 0.5))
      const scanYMax = Math.min(yEnd - 1, Math.ceil(edgeYMax - 0.5) - 1)
      const slope = (x1 - x0) / (y1 - y0)

      for (let row = scanYMin; row <= scanYMax; row++) {
        const scanY = row + 0.5
        const xIntersect = x0 + (scanY - y0) * slope
        let arr = table.get(row)
        if (!arr) {
          arr = []
          table.set(row, arr)
        }
        arr.push(xIntersect)
      }
    }
  }

  for (const [, arr] of table) {
    arr.sort((a, b) => a - b)
  }

  return table
}

/**
 * Union two sets of scanline crossing pairs.
 * Each input is a sorted array where consecutive pairs (0-1, 2-3, ...) define filled intervals.
 * Returns a new sorted crossing array whose pairs represent the union of both interval sets.
 */
function unionScanlineIntervals(a: number[], b: number[]): number[] {
  // Convert crossing pairs to [start, end] intervals
  const intervals: [number, number][] = []
  for (let i = 0; i < a.length - 1; i += 2) intervals.push([a[i], a[i + 1]])
  for (let i = 0; i < b.length - 1; i += 2) intervals.push([b[i], b[i + 1]])

  // Sort by start
  intervals.sort((x, y) => x[0] - y[0])

  // Merge overlapping intervals
  const result: number[] = []
  let [curStart, curEnd] = intervals[0]
  for (let i = 1; i < intervals.length; i++) {
    if (intervals[i][0] <= curEnd) {
      curEnd = Math.max(curEnd, intervals[i][1])
    } else {
      result.push(curStart, curEnd)
      curStart = intervals[i][0]
      curEnd = intervals[i][1]
    }
  }
  result.push(curStart, curEnd)
  return result
}

/**
 * Scanline fill: precompute sorted X-intersections for each row in the pixel-space polygon.
 * Returns a Map from integer Y to sorted array of X-intersection values.
 * For each row, pixels between pairs of intersections (0-1, 2-3, ...) are inside.
 *
 * Uses center-based sampling (row + 0.5) matching standard rasterization tools
 * (GDAL/rasterio default). A pixel is included if its center is inside the polygon.
 *
 * For MultiPolygon, each polygon member is rasterized independently and intervals
 * are merged with union semantics, so overlapping members are included (not cancelled
 * by even-odd parity across members).
 *
 * Complexity: O(H*E + H*E*logE) vs O(W*H*V) for per-pixel point-in-polygon.
 */
export function buildScanlineTable(
  geometry: QueryGeometry,
  yStart: number,
  yEnd: number
): Map<number, number[]> {
  if (geometry.type === 'Polygon') {
    return buildScanlineTableForRings(geometry.coordinates, yStart, yEnd)
  }

  if (geometry.type === 'Point') {
    return new Map()
  }

  // MultiPolygon: rasterize each polygon independently, then merge intervals
  const perPolygon = geometry.coordinates.map((polygon) =>
    buildScanlineTableForRings(polygon, yStart, yEnd)
  )
  if (perPolygon.length === 1) return perPolygon[0]

  // Merge: collect all rows, union their interval sets
  const merged = new Map<number, number[]>()
  for (const table of perPolygon) {
    for (const [row, crossings] of table) {
      // Convert crossing pairs to intervals, then merge with existing
      const existing = merged.get(row)
      if (!existing) {
        merged.set(row, crossings)
      } else {
        merged.set(row, unionScanlineIntervals(existing, crossings))
      }
    }
  }
  return merged
}

/**
 * Gets tiles that intersect a bounding box at a given zoom level.
 */
export function getTilesForBoundingBox(
  bbox: BoundingBox,
  zoom: number,
  crs: CRS,
  xyLimits: XYLimits
): TileTuple[] {
  const bounds: [[number, number], [number, number]] = [
    [bbox.west, bbox.south],
    [bbox.east, bbox.north],
  ]

  if (crs === 'EPSG:4326') {
    return getTilesAtZoomEquirect(zoom, bounds, xyLimits)
  }

  return getTilesAtZoom(zoom, bounds)
}

/**
 * Gets tiles that intersect a GeoJSON geometry at a given zoom level.
 */
export function getTilesForPolygon(
  geometry: QueryGeometry,
  zoom: number,
  crs: CRS,
  xyLimits: XYLimits
): TileTuple[] {
  const bbox = computeBoundingBox(geometry)
  return getTilesForBoundingBox(bbox, zoom, crs, xyLimits)
}
