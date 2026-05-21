/**
 * @module query-utils
 *
 * Utility functions for query coordinate transformations,
 * mercator corrections, and point-in-polygon tests.
 */

import {
  getTilesAtZoom,
  getTilesAtZoomEquirect,
  lonToMercatorNorm,
  type TileTuple,
  type XYLimits,
} from '../map-utils'
import { WEB_MERCATOR_EXTENT } from '../constants'
import type { Bounds, CRS } from '../types'
import type { BoundingBox, QueryGeometry, GeoJSONMultiPolygon } from './types'
import {
  clampLatLonToProj4def,
  createWGS84ToSourceTransformer,
  sourceCRSToPixel,
} from '../projection-utils'

/** Pixel rectangle with exclusive max bounds. */
export interface PixelRect {
  minX: number
  maxX: number
  minY: number
  maxY: number
}

/**
 * Check if a raster's own extent crosses the antimeridian (EPSG:4326 only).
 * Returns true when the two-strip antimeridian query path is unsupported.
 */
export function rasterExtentCrossesAntimeridian(
  crs: CRS,
  xyLimits?: XYLimits | null
): boolean {
  if (crs !== 'EPSG:4326' || !xyLimits) return false
  return (
    xyLimits.xMin > xyLimits.xMax || xyLimits.xMax > 180 || xyLimits.xMin < -180
  )
}

/** Cached WGS84↔source-CRS transformer used by query input geometry transforms. */
export type CachedTransformer = ReturnType<
  typeof createWGS84ToSourceTransformer
>

/**
 * Computes fractional position within a tile for a geographic point.
 * Returns values in [0, 1] representing position within the tile.
 */
function geoToTileFraction(
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
 * Converts a tile-pixel position to source-CRS coordinates [x, y].
 *
 * EPSG:4326 tile pyramids: returns [lon, lat] in degrees, against xyLimits.
 * EPSG:3857 tile pyramids: returns [x, y] in Web Mercator meters.
 */
export function tilePixelToSourceCRS(
  tile: TileTuple,
  pixelX: number,
  pixelY: number,
  tileSize: number,
  crs: CRS,
  xyLimits: XYLimits
): [number, number] {
  const [z, x, y] = tile
  const z2 = Math.pow(2, z)
  const fracX = (x + pixelX / tileSize) / z2
  const fracY = (y + pixelY / tileSize) / z2

  if (crs === 'EPSG:4326') {
    const { xMin, xMax, yMin, yMax } = xyLimits
    return [xMin + fracX * (xMax - xMin), yMax - fracY * (yMax - yMin)]
  }

  // EPSG:3857: tile fractions map linearly to Web Mercator meters.
  return [
    (fracX * 2 - 1) * WEB_MERCATOR_EXTENT,
    (1 - fracY * 2) * WEB_MERCATOR_EXTENT,
  ]
}

/**
 * Computes bounding box from GeoJSON geometry.
 */
function computeBoundingBox(geometry: QueryGeometry): BoundingBox {
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
 * Compute pixel bounds covering a GeoJSON geometry, in the raster's pixel grid.
 *
 * All math goes through the source CRS via the supplied proj4 transformer, so
 * the mapping is linear in the raster's own coordinate space regardless of CRS.
 */
export function computePixelBoundsFromGeometry(
  geometry: QueryGeometry,
  sourceBounds: Bounds,
  width: number,
  height: number,
  proj4def: string,
  latIsAscending?: boolean,
  cachedTransformer?: CachedTransformer
): PixelRect | null {
  const transformer =
    cachedTransformer ?? createWGS84ToSourceTransformer(proj4def)

  if (geometry.type === 'Point') {
    const [lon, lat] = geometry.coordinates
    const point = lonLatToPixel(
      lon,
      lat,
      sourceBounds,
      width,
      height,
      proj4def,
      latIsAscending,
      transformer
    )
    if (!point) return null

    const [xPixel, yPixel] = point
    if (
      !isFinite(xPixel) ||
      !isFinite(yPixel) ||
      xPixel < 0 ||
      xPixel > width ||
      yPixel < 0 ||
      yPixel > height
    ) {
      return null
    }

    const xStart = Math.min(Math.max(0, Math.floor(xPixel)), width - 1)
    const yStart = Math.min(Math.max(0, Math.floor(yPixel)), height - 1)
    return { minX: xStart, maxX: xStart + 1, minY: yStart, maxY: yStart + 1 }
  }

  const bbox = computeBoundingBox(geometry)

  // Sample points along bbox edges to capture curved projections
  // (corners alone can miss extrema for conic/polar projections).
  const numSamples = 5
  const samplePoints: [number, number][] = []
  for (let i = 0; i <= numSamples; i++) {
    const t = i / numSamples
    const lon = bbox.west + t * (bbox.east - bbox.west)
    const lat = bbox.south + t * (bbox.north - bbox.south)
    samplePoints.push([lon, bbox.south])
    samplePoints.push([lon, bbox.north])
    samplePoints.push([bbox.west, lat])
    samplePoints.push([bbox.east, lat])
  }

  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity

  for (const [lon, lat] of samplePoints) {
    const [clampedLon, clampedLat] = clampLatLonToProj4def(lon, lat, proj4def)
    const [srcX, srcY] = transformer.forward(clampedLon, clampedLat)
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

  if (
    !isFinite(minX) ||
    !isFinite(maxX) ||
    !isFinite(minY) ||
    !isFinite(maxY)
  ) {
    return null
  }

  const xStart = Math.min(Math.max(0, Math.floor(minX)), width - 1)
  const xEnd = Math.min(width, Math.max(Math.floor(maxX) + 1, xStart + 1))
  const yStart = Math.min(Math.max(0, Math.floor(minY)), height - 1)
  const yEnd = Math.min(height, Math.max(Math.floor(maxY) + 1, yStart + 1))

  if (xEnd <= xStart || yEnd <= yStart) return null

  return { minX: xStart, maxX: xEnd, minY: yStart, maxY: yEnd }
}

import { DEFAULT_QUERY_DENSIFY_MAX_ERROR } from '../constants'

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
 * For source-projected data: forward-transforms vertices, then converts source
 * CRS → pixel. Otherwise uses mercator/equirect math → pixel.
 * Densifies edges to preserve curvature under nonlinear projections.
 *
 * Returns a geometry with the same GeoJSON ring structure but in pixel coordinates,
 * suitable for use with scanline rasterization.
 */
export function transformGeometryToPixelSpace(
  geometry: QueryGeometry,
  sourceBounds: Bounds,
  width: number,
  height: number,
  proj4def: string,
  latIsAscending?: boolean,
  cachedTransformer?: CachedTransformer
): QueryGeometry | null {
  const transformer =
    cachedTransformer ?? createWGS84ToSourceTransformer(proj4def)

  if (geometry.type === 'Point') {
    const [lon, lat] = geometry.coordinates
    const px = lonLatToPixel(
      lon,
      lat,
      sourceBounds,
      width,
      height,
      proj4def,
      latIsAscending,
      transformer
    )
    if (!px) return null
    return { type: 'Point', coordinates: [px[0], px[1]] }
  }

  const transformVertex = (lon: number, lat: number): [number, number] => {
    const px = lonLatToPixel(
      lon,
      lat,
      sourceBounds,
      width,
      height,
      proj4def,
      latIsAscending,
      transformer
    )
    return px ?? [NaN, NaN]
  }

  // Source-CRS forward transforms can be nonlinear, so always densify edges.
  const transformRing = (ring: number[][]): number[][] =>
    densifyAndTransformRing(ring, transformVertex)

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
    const [clampedLon, clampedLat] = clampLatLonToProj4def(lon, lat, crs)
    const { fracX, fracY } = geoToTileFraction(
      clampedLon,
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
 * Convert a single WGS84 lon/lat to source-CRS pixel coordinates.
 */
function lonLatToPixel(
  lon: number,
  lat: number,
  sourceBounds: Bounds,
  width: number,
  height: number,
  proj4def: string,
  latIsAscending?: boolean,
  cachedTransformer?: CachedTransformer
): [number, number] | null {
  const transformer =
    cachedTransformer ?? createWGS84ToSourceTransformer(proj4def)
  const [clampedLon, clampedLat] = clampLatLonToProj4def(lon, lat, proj4def)
  const [srcX, srcY] = transformer.forward(clampedLon, clampedLat)
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

// ---------------------------------------------------------------------------
// Antimeridian preprocessing
// ---------------------------------------------------------------------------

/**
 * Bounding box that correctly represents antimeridian-crossing regions.
 * When crossesAntimeridian is true, west > east (e.g. west=170, east=-170).
 */
export interface WrappedBoundingBox {
  west: number // in [-180, 180]
  east: number // in [-180, 180]
  south: number
  north: number
  crossesAntimeridian: boolean
}

/**
 * Wrap a longitude into (-180, 180], preserving 180 (not folding to -180).
 */
function wrapLon(lon: number): number {
  let w = (lon + 180) % 360
  if (w < 0) w += 360
  w -= 180
  // Preserve 180: the modulo maps 180 to -180, but we want 180
  if (w === -180 && lon > 0) w = 180
  return w
}

/**
 * Ensure a ring is closed (last vertex equals first vertex).
 */
function closeRing(ring: number[][]): number[][] {
  if (ring.length < 2) return ring
  const first = ring[0]
  const last = ring[ring.length - 1]
  if (first[0] !== last[0] || first[1] !== last[1]) {
    return [...ring, [first[0], first[1]]]
  }
  return ring
}

/**
 * Compute the centroid longitude of a closed ring (excludes closing vertex).
 */
function ringCentroidLon(ring: number[][]): number {
  const count = Math.max(1, ring.length - 1)
  let sum = 0
  for (let i = 0; i < count; i++) {
    sum += ring[i][0]
  }
  return sum / count
}

/**
 * Find the single ±360° shift that places centroidLon closest to targetCenter.
 */
function nearestLonShift(centroidLon: number, targetCenter: number): number {
  return Math.round((targetCenter - centroidLon) / 360) * 360
}

/**
 * Shift all ring longitudes by a fixed offset.
 */
function shiftRingsLon(rings: number[][][], shift: number): number[][][] {
  return rings.map((ring) => ring.map(([lon, lat]) => [lon + shift, lat]))
}

/**
 * Scan closed rings for longitude range (excludes closing vertices).
 */
function lonRangeOfRings(rings: number[][][]): {
  min: number
  allAbove180: boolean
} {
  let min = Infinity
  let allAbove180 = true
  for (const ring of rings) {
    for (let i = 0; i < ring.length - 1; i++) {
      const lon = ring[i][0]
      if (lon < min) min = lon
      if (lon <= 180) allAbove180 = false
    }
  }
  return { min, allAbove180 }
}

/**
 * Compute the shift needed to canonicalize a longitude range:
 * +360 if any lon < -180, -360 if all lons > 180, else 0.
 */
function canonicalLonShift(rings: number[][][]): number {
  const { min, allAbove180 } = lonRangeOfRings(rings)
  if (min < -180) return 360
  if (allAbove180) return -360
  return 0
}

/**
 * Apply canonical longitude shift to a ring array.
 */
function canonicalizeLonRange(rings: number[][][]): number[][][] {
  const shift = canonicalLonShift(rings)
  return shift !== 0 ? shiftRingsLon(rings, shift) : rings
}

/**
 * Normalize a polygon's ring longitudes for continuity.
 *
 * The outer ring (index 0) determines the unwrap direction. Holes (index 1+)
 * are shifted by a single per-ring ±360 offset to match the outer ring's
 * coordinate frame. If the result extends below -180, all rings are shifted
 * +360 to canonicalize to eastward extension.
 *
 * Output rings are always closed.
 */
function normalizePolygonRings(rings: number[][][]): number[][][] {
  if (rings.length === 0) return rings

  // Check if any vertex in any ring is outside [-180, 180] (explicit crossing)
  let hasExplicit = false
  for (const ring of rings) {
    for (const [lon] of ring) {
      if (lon > 180 || lon < -180) {
        hasExplicit = true
        break
      }
    }
    if (hasExplicit) break
  }

  // All vertices in [-180, 180]: literal interpretation, just ensure rings are closed
  if (!hasExplicit) {
    return rings.map((ring) => closeRing(ring.map(([lon, lat]) => [lon, lat])))
  }

  // Explicit crossing: normalize outer ring for longitude continuity
  const outer = normalizeRingLongitudes(rings[0])

  // Align hole rings to outer ring's frame with a single per-ring ±360 shift
  const outerCenter = ringCentroidLon(outer)
  const result: number[][][] = [outer]
  for (let r = 1; r < rings.length; r++) {
    const hole = normalizeRingLongitudes(rings[r])
    const shift = nearestLonShift(ringCentroidLon(hole), outerCenter)
    result.push(
      shift !== 0 ? hole.map(([lon, lat]) => [lon + shift, lat]) : hole
    )
  }

  return canonicalizeLonRange(result)
}

/**
 * Apply per-edge longitude unwrapping for continuity.
 *
 * Only called when the caller has already verified explicit out-of-range
 * coordinates (|lon| > 180). Returns a new closed ring.
 */
function normalizeRingLongitudes(ring: number[][]): number[][] {
  if (ring.length < 2) return ring.map(([lon, lat]) => [lon, lat])

  const result: number[][] = [[ring[0][0], ring[0][1]]]
  let prevLon = ring[0][0]

  for (let i = 1; i < ring.length; i++) {
    let lon = ring[i][0]
    const lat = ring[i][1]
    const delta = lon - prevLon
    if (delta > 180) {
      lon -= 360
    } else if (delta < -180) {
      lon += 360
    }
    result.push([lon, lat])
    prevLon = lon
  }

  return closeRing(result)
}

/**
 * Compute a WrappedBoundingBox from normalized polygon rings.
 * Input rings must already be processed by normalizePolygonRings.
 * Uses wrap-normalization (not clamping) for crossings.
 */
function computeWrappedBboxFromNormalized(
  normalizedRings: number[][][]
): WrappedBoundingBox {
  let rawMin = Infinity
  let rawMax = -Infinity
  let south = Infinity
  let north = -Infinity

  for (const ring of normalizedRings) {
    for (let i = 0; i < ring.length - 1; i++) {
      const lon = ring[i][0]
      const lat = ring[i][1]
      if (lon < rawMin) rawMin = lon
      if (lon > rawMax) rawMax = lon
      if (lat < south) south = lat
      if (lat > north) north = lat
    }
  }

  if (rawMin >= -180 && rawMax <= 180) {
    return {
      west: rawMin,
      east: rawMax,
      south,
      north,
      crossesAntimeridian: false,
    }
  }

  // Crossing: rawMax > 180 (westward was canonicalized to eastward)
  const west = wrapLon(rawMin)
  const east = wrapLon(rawMax)
  // Guard: if wrapping collapses the crossing (e.g. rawMin = -180 exactly),
  // treat as non-crossing. This can't happen for valid simple polygons after
  // canonicalization, but defends the west > east invariant.
  if (west <= east) {
    return { west, east, south, north, crossesAntimeridian: false }
  }
  return { west, east, south, north, crossesAntimeridian: true }
}

/**
 * Sutherland-Hodgman clip of a closed ring against a half-plane.
 *
 * keepBelow=true:  keep lon <= clipLon
 * keepBelow=false: keep lon > clipLon (output lons shifted by -360)
 *
 * Returns a closed ring (possibly empty or degenerate).
 */
function clipRingToHalfPlane(
  ring: number[][],
  clipLon: number,
  keepBelow: boolean
): number[][] {
  if (ring.length < 4) return [] // need at least a triangle (3 vertices + close)

  const output: number[][] = []
  const isInside = keepBelow
    ? (lon: number) => lon <= clipLon
    : (lon: number) => lon > clipLon

  // Walk edges of the closed ring (ring[i] -> ring[i+1])
  for (let i = 0; i < ring.length - 1; i++) {
    const [lon0, lat0] = ring[i]
    const [lon1, lat1] = ring[i + 1]
    const in0 = isInside(lon0)
    const in1 = isInside(lon1)

    if (in0 && in1) {
      // Both inside: output p1
      output.push([lon1, lat1])
    } else if (in0 && !in1) {
      // Inside -> outside: output intersection
      const t = (clipLon - lon0) / (lon1 - lon0)
      const latI = lat0 + t * (lat1 - lat0)
      output.push([clipLon, latI])
    } else if (!in0 && in1) {
      // Outside -> inside: output intersection, then p1
      const t = (clipLon - lon0) / (lon1 - lon0)
      const latI = lat0 + t * (lat1 - lat0)
      output.push([clipLon, latI])
      output.push([lon1, lat1])
    }
    // Both outside: output nothing
  }

  if (output.length < 3) return []

  // Shift longitudes for the "outside" half
  const shifted = !keepBelow
    ? output.map(([lon, lat]) => [lon - 360, lat])
    : output

  return closeRing(shifted)
}

/**
 * Clip a normalized polygon (outer + holes) at the antimeridian.
 *
 * Input rings must already be processed by normalizePolygonRings
 * (crossing is always > 180, never < -180).
 *
 * Returns west (lon <= 180) and east (lon > 180, shifted -360) ring sets.
 * Degenerate rings (< 3 unique vertices) are dropped.
 */
function clipNormalizedPolygonAtAntimeridian(normalizedRings: number[][][]): {
  west: number[][][]
  east: number[][][]
} {
  // Check if any vertex lon > 180; if not, no crossing
  let crosses = false
  for (const ring of normalizedRings) {
    for (let i = 0; i < ring.length - 1; i++) {
      if (ring[i][0] > 180) {
        crosses = true
        break
      }
    }
    if (crosses) break
  }

  if (!crosses) {
    return { west: normalizedRings, east: [] }
  }

  const west: number[][][] = []
  const east: number[][][] = []

  for (const ring of normalizedRings) {
    const w = clipRingToHalfPlane(ring, 180, true)
    const e = clipRingToHalfPlane(ring, 180, false)
    if (w.length >= 4) west.push(w)
    if (e.length >= 4) east.push(e)
  }

  return { west, east }
}

/**
 * Align independently-normalized MultiPolygon members into one shared unwrap frame.
 *
 * Uses gap detection across all outer rings to find the optimal unwrap center,
 * then shifts each member by a single ±360 offset (preserving internal continuity).
 * Applies eastward canonicalization if the shared range extends below -180.
 */
function alignMultiPolygonMembers(members: number[][][][]): number[][][][] {
  if (members.length <= 1) return members

  // Collect all outer ring lons, wrap-normalized to [-180, 180]
  const wrappedLons: number[] = []
  for (const member of members) {
    const outer = member[0]
    for (let i = 0; i < outer.length - 1; i++) {
      wrappedLons.push(wrapLon(outer[i][0]))
    }
  }
  wrappedLons.sort((a, b) => a - b)

  // Find the largest gap between consecutive sorted lons (including wrap-around)
  let maxGap = 0
  let gapEndIndex = 0
  for (let i = 1; i < wrappedLons.length; i++) {
    const gap = wrappedLons[i] - wrappedLons[i - 1]
    if (gap > maxGap) {
      maxGap = gap
      gapEndIndex = i
    }
  }
  // Check the wrap-around gap
  const wrapGap = wrappedLons[0] + 360 - wrappedLons[wrappedLons.length - 1]
  if (wrapGap > maxGap) {
    maxGap = wrapGap
    gapEndIndex = 0
  }

  // The unwrap center is opposite the largest gap (180° from the gap's midpoint)
  const gapStart =
    gapEndIndex === 0
      ? wrappedLons[wrappedLons.length - 1]
      : wrappedLons[gapEndIndex - 1]
  const gapEnd = wrappedLons[gapEndIndex]
  const gapMidpoint =
    gapEndIndex === 0 ? (gapStart + gapEnd + 360) / 2 : (gapStart + gapEnd) / 2
  const center = wrapLon(gapMidpoint + 180)

  // Shift each member by a single ±360 offset based on its outer ring centroid
  const aligned: number[][][][] = members.map((member) => {
    const shift = nearestLonShift(ringCentroidLon(member[0]), center)
    return shift !== 0
      ? member.map((ring) => ring.map(([lon, lat]) => [lon + shift, lat]))
      : member
  })

  // Canonicalize: shift all if any lon < -180, or all > 180
  const shift = canonicalLonShift(aligned.flatMap((m) => m))
  if (shift !== 0) {
    return aligned.map((member) => shiftRingsLon(member, shift))
  }

  return aligned
}

/**
 * Preprocess a query geometry for antimeridian handling.
 *
 * For Polygon/MultiPolygon: normalizes ring longitudes, computes a
 * WrappedBoundingBox, and clips at ±180 if crossing. For Point: returns as-is.
 *
 * The normalized geometry is only valid for CRSes that share WGS84 wrapped
 * longitude semantics (EPSG:3857, EPSG:4326). Generic proj4 callers may still
 * use the returned bbox to detect unsupported crossings.
 */
export function preprocessQueryGeometry(geometry: QueryGeometry): {
  geometry: QueryGeometry
  bbox: WrappedBoundingBox
} {
  if (geometry.type === 'Point') {
    const [lon, lat] = geometry.coordinates
    return {
      geometry,
      bbox: {
        west: lon,
        east: lon,
        south: lat,
        north: lat,
        crossesAntimeridian: false,
      },
    }
  }

  if (geometry.type === 'Polygon') {
    const normalized = normalizePolygonRings(geometry.coordinates)
    const bbox = computeWrappedBboxFromNormalized(normalized)

    if (!bbox.crossesAntimeridian) {
      return { geometry: { type: 'Polygon', coordinates: normalized }, bbox }
    }

    const { west, east } = clipNormalizedPolygonAtAntimeridian(normalized)
    const polygons: number[][][][] = []
    if (west.length > 0) polygons.push(west)
    if (east.length > 0) polygons.push(east)

    if (polygons.length === 0) {
      return { geometry: { type: 'Polygon', coordinates: normalized }, bbox }
    }

    return {
      geometry: {
        type: 'MultiPolygon',
        coordinates: polygons,
      } as GeoJSONMultiPolygon,
      bbox,
    }
  }

  // MultiPolygon: normalize each member, align, compute combined bbox, clip
  const normalizedMembers = geometry.coordinates.map((memberRings) =>
    normalizePolygonRings(memberRings)
  )
  const aligned = alignMultiPolygonMembers(normalizedMembers)

  // Compute combined bbox from all aligned rings
  const allRings: number[][][] = []
  for (const member of aligned) {
    for (const ring of member) {
      allRings.push(ring)
    }
  }
  const bbox = computeWrappedBboxFromNormalized(allRings)

  if (!bbox.crossesAntimeridian) {
    return {
      geometry: {
        type: 'MultiPolygon',
        coordinates: aligned,
      } as GeoJSONMultiPolygon,
      bbox,
    }
  }

  // Clip each member
  const resultPolygons: number[][][][] = []
  for (const member of aligned) {
    const { west, east } = clipNormalizedPolygonAtAntimeridian(member)
    if (west.length > 0) resultPolygons.push(west)
    if (east.length > 0) resultPolygons.push(east)
  }

  if (resultPolygons.length === 0) {
    return {
      geometry: {
        type: 'MultiPolygon',
        coordinates: aligned,
      } as GeoJSONMultiPolygon,
      bbox,
    }
  }

  return {
    geometry: {
      type: 'MultiPolygon',
      coordinates: resultPolygons,
    } as GeoJSONMultiPolygon,
    bbox,
  }
}

/**
 * Derive two pixel rectangles from an antimeridian-crossing WrappedBoundingBox.
 *
 * West strip: covers lon [bbox.west, 180].
 * East strip: covers lon [-180, bbox.east].
 *
 * Both strips share the same pixel-Y range. All conversions go through the
 * source CRS via the supplied proj4 transformer, so the math is linear in the
 * raster's own coordinate space regardless of CRS.
 */
export function wrappedBboxToPixelSpans(
  bbox: WrappedBoundingBox,
  sourceBounds: Bounds,
  width: number,
  height: number,
  proj4def: string,
  latIsAscending?: boolean,
  cachedTransformer?: CachedTransformer
): { west?: PixelRect; east?: PixelRect } {
  const transformer =
    cachedTransformer ?? createWGS84ToSourceTransformer(proj4def)

  const toPixel = (lon: number, lat: number): [number, number] | null => {
    const [clampedLon, clampedLat] = clampLatLonToProj4def(lon, lat, proj4def)
    const [srcX, srcY] = transformer.forward(clampedLon, clampedLat)
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

  // Y range shared by both strips. Sample both lat extremes at lon=0 — for the
  // CRSes that take this path (EPSG:4326/3857 without a consumer proj4 prop)
  // the source-CRS Y is independent of lon.
  const yPixels: number[] = []
  for (const lat of [bbox.south, bbox.north]) {
    const px = toPixel(0, lat)
    if (px) yPixels.push(px[1])
  }
  if (yPixels.length === 0) return {}
  const yMinPx = Math.min(...yPixels)
  const yMaxPx = Math.max(...yPixels)
  const yStart = Math.min(Math.max(0, Math.floor(yMinPx)), height - 1)
  const yEnd = Math.min(height, Math.max(Math.ceil(yMaxPx), yStart + 1))
  if (yEnd <= yStart) return {}

  const computeStrip = (
    lonMin: number,
    lonMax: number
  ): PixelRect | undefined => {
    const xPixels: number[] = []
    for (const lon of [lonMin, lonMax]) {
      const px = toPixel(lon, 0)
      if (px) xPixels.push(px[0])
    }
    if (xPixels.length === 0) return undefined
    const rawMinX = Math.floor(Math.min(...xPixels))
    const rawMaxX = Math.ceil(Math.max(...xPixels))
    if (rawMaxX <= 0 || rawMinX >= width) return undefined
    const minX = Math.max(0, rawMinX)
    const maxX = Math.min(width, rawMaxX)
    if (maxX <= minX) return undefined
    return { minX, maxX, minY: yStart, maxY: yEnd }
  }

  const result: { west?: PixelRect; east?: PixelRect } = {}
  const westStrip = computeStrip(bbox.west, 180)
  if (westStrip) result.west = westStrip
  const eastStrip = computeStrip(-180, bbox.east)
  if (eastStrip) result.east = eastStrip

  return result
}
