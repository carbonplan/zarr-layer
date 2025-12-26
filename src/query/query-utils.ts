/**
 * @module query-utils
 *
 * Utility functions for query coordinate transformations,
 * mercator corrections, and point-in-polygon tests.
 */

import type { TileTuple, XYLimits, MercatorBounds } from '../map-utils'
import {
  getTilesAtZoom,
  getTilesAtZoomEquirect,
  latToMercatorNorm,
  lonToMercatorNorm,
  mercatorNormToLat,
  mercatorNormToLon,
  lonToTile,
  latToTileMercator,
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
    const [xMin, yMin, xMax, yMax] = sourceBounds

    let srcX: number, srcY: number
    if (centerPixel) {
      // pixelToSourceCRS uses x/(width-1) normalization: pixel 0 → xMin, pixel width-1 → xMax
      // This is correct for pixel centers
      ;[srcX, srcY] = pixelToSourceCRS(
        x,
        y,
        sourceBounds,
        width,
        height,
        latIsAscending ?? null
      )
    } else {
      // For corners, extend beyond pixel centers by half a pixel on each edge.
      // sourceBounds represents center-to-center (pixel 0 center → pixel width-1 center),
      // but corners need edge-to-edge coverage.
      if (width <= 1 || height <= 1) {
        srcX = (xMin + xMax) / 2
        srcY = (yMin + yMax) / 2
      } else {
        const xPixelWidth = (xMax - xMin) / (width - 1)
        const yPixelWidth = (yMax - yMin) / (height - 1)
        const xHalfPixel = xPixelWidth / 2
        const yHalfPixel = yPixelWidth / 2
        // Map x in [0, width] to [xMin - halfPixel, xMax + halfPixel]
        const xLeftEdge = xMin - xHalfPixel
        const xFullExtent = xMax - xMin + xPixelWidth
        srcX = xLeftEdge + (x / width) * xFullExtent
        // Y handling depends on latIsAscending
        const yFullExtent = yMax - yMin + yPixelWidth
        if (latIsAscending === false) {
          // row 0 = yMax (north), so y=0 → yMax + halfPixel, y=height → yMin - halfPixel
          const yTopEdge = yMax + yHalfPixel
          srcY = yTopEdge - (y / height) * yFullExtent
        } else {
          // row 0 = yMin (south), so y=0 → yMin - halfPixel, y=height → yMax + halfPixel
          const yBottomEdge = yMin - yHalfPixel
          srcY = yBottomEdge + (y / height) * yFullExtent
        }
      }
    }

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

  // EPSG:3857 - Mercator
  const globalFracX = (lng + 180) / 360
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
        latIsAscending ?? null
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
    const xEnd = Math.min(width, Math.floor(maxX) + 1)
    const yStart = Math.max(0, Math.floor(minY))
    const yEnd = Math.min(height, Math.floor(maxY) + 1)

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

    if (overlapX1 <= overlapX0 || yFracMax <= yFracMin) return null

    const minX = ((overlapX0 - bounds.x0) / (bounds.x1 - bounds.x0)) * width
    const maxX = ((overlapX1 - bounds.x0) / (bounds.x1 - bounds.x0)) * width

    xStart = Math.max(0, Math.floor(minX))
    xEnd = Math.min(width, Math.ceil(maxX))
    yStart = Math.max(0, Math.floor(yFracMin * height))
    yEnd = Math.min(height, Math.ceil(yFracMax * height))
  } else {
    const overlapY0 = Math.max(bounds.y0, Math.min(polyY0, polyY1))
    const overlapY1 = Math.min(bounds.y1, Math.max(polyY0, polyY1))

    if (overlapX1 <= overlapX0 || overlapY1 <= overlapY0) return null

    const minX = ((overlapX0 - bounds.x0) / (bounds.x1 - bounds.x0)) * width
    const maxX = ((overlapX1 - bounds.x0) / (bounds.x1 - bounds.x0)) * width
    const minY = ((overlapY0 - bounds.y0) / (bounds.y1 - bounds.y0)) * height
    const maxY = ((overlapY1 - bounds.y0) / (bounds.y1 - bounds.y0)) * height

    xStart = Math.max(0, Math.floor(minX))
    xEnd = Math.min(width, Math.ceil(maxX))
    yStart = Math.max(0, Math.floor(minY))
    yEnd = Math.min(height, Math.ceil(maxY))
  }

  if (xEnd <= xStart || yEnd <= yStart) return null

  return { minX: xStart, maxX: xEnd, minY: yStart, maxY: yEnd }
}

/**
 * Ray-casting point-in-polygon test.
 * Tests if a point is inside a single polygon ring.
 */
export function pointInPolygon(
  point: [number, number],
  polygon: number[][]
): boolean {
  let inside = false
  const [x, y] = point

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0]
    const yi = polygon[i][1]
    const xj = polygon[j][0]
    const yj = polygon[j][1]

    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside
    }
  }

  return inside
}

/**
 * Tests if a point is inside a GeoJSON geometry (Polygon or MultiPolygon).
 * Correctly handles holes in polygons.
 */
export function pointInGeoJSON(
  point: [number, number],
  geometry: QueryGeometry
): boolean {
  if (geometry.type === 'Point') {
    return (
      point[0] === geometry.coordinates[0] &&
      point[1] === geometry.coordinates[1]
    )
  }

  if (geometry.type === 'Polygon') {
    // Test outer ring
    if (!pointInPolygon(point, geometry.coordinates[0])) return false
    // Test holes (if inside any hole, point is outside polygon)
    for (let i = 1; i < geometry.coordinates.length; i++) {
      if (pointInPolygon(point, geometry.coordinates[i])) return false
    }
    return true
  }

  // MultiPolygon - check each polygon
  for (const polygon of geometry.coordinates) {
    if (pointInPolygon(point, polygon[0])) {
      let inHole = false
      for (let i = 1; i < polygon.length; i++) {
        if (pointInPolygon(point, polygon[i])) {
          inHole = true
          break
        }
      }
      if (!inHole) return true
    }
  }

  return false
}

/**
 * Lightweight polygon-rectangle intersection test.
 * Returns true if any rectangle corner is inside geometry,
 * any geometry vertex is inside rectangle, or if any edges intersect.
 */
function rectIntersectsGeometry(
  rect: [number, number][],
  geometry: QueryGeometry
): boolean {
  const rectMinX = Math.min(...rect.map((p) => p[0]))
  const rectMaxX = Math.max(...rect.map((p) => p[0]))
  const rectMinY = Math.min(...rect.map((p) => p[1]))
  const rectMaxY = Math.max(...rect.map((p) => p[1]))

  const pointInRect = (p: [number, number]) =>
    p[0] >= rectMinX && p[0] <= rectMaxX && p[1] >= rectMinY && p[1] <= rectMaxY

  // Any rect corner inside geometry (supports point or polygon)
  for (const corner of rect) {
    if (pointInGeoJSON(corner, geometry)) return true
  }

  // Point geometry inside rectangle
  if (
    geometry.type === 'Point' &&
    pointInRect([geometry.coordinates[0], geometry.coordinates[1]])
  ) {
    return true
  }

  // Any polygon vertex inside rect
  if (geometry.type !== 'Point') {
    const rings =
      geometry.type === 'Polygon'
        ? geometry.coordinates
        : geometry.coordinates.flatMap((poly) => poly)
    for (const ring of rings) {
      for (const [lon, lat] of ring) {
        if (pointInRect([lon, lat])) return true
      }
    }
  }

  // Edge intersection: rectangle edges vs polygon edges (outer rings only)
  const rectEdges: [[number, number], [number, number]][] = [
    [rect[0], rect[1]],
    [rect[1], rect[2]],
    [rect[2], rect[3]],
    [rect[3], rect[0]],
  ]

  const edgesFromRing = (ring: number[][]) =>
    ring
      .slice(0, -1)
      .map(
        (_, i) => [ring[i], ring[i + 1]] as [[number, number], [number, number]]
      )

  const segments =
    geometry.type === 'Polygon'
      ? edgesFromRing(geometry.coordinates[0])
      : geometry.type === 'MultiPolygon'
      ? geometry.coordinates.flatMap((poly) => edgesFromRing(poly[0]))
      : []

  const intersects = (
    a1: [number, number],
    a2: [number, number],
    b1: [number, number],
    b2: [number, number]
  ): boolean => {
    const cross = (v1: [number, number], v2: [number, number]) =>
      v1[0] * v2[1] - v1[1] * v2[0]
    const sub = (p1: [number, number], p2: [number, number]) =>
      [p1[0] - p2[0], p1[1] - p2[1]] as [number, number]

    const d1 = sub(a2, a1)
    const d2 = sub(b2, b1)
    const denom = cross(d1, d2)
    if (denom === 0) return false

    const s = cross(sub(b1, a1), d2) / denom
    const t = cross(sub(b1, a1), d1) / denom
    return s >= 0 && s <= 1 && t >= 0 && t <= 1
  }

  for (const edge of rectEdges) {
    for (const seg of segments) {
      if (intersects(edge[0], edge[1], seg[0], seg[1])) {
        return true
      }
    }
  }

  return false
}

/**
 * Rectangle (pixel) corners in lon/lat for tiled mode.
 */
function pixelRectLonLat(
  tile: TileTuple,
  pixelX: number,
  pixelY: number,
  tileSize: number,
  crs: CRS,
  xyLimits: XYLimits
): [number, number][] {
  const corners: [number, number][] = []
  const offsets = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ]
  for (const [dx, dy] of offsets) {
    const p = tilePixelToLatLon(
      tile,
      pixelX + dx,
      pixelY + dy,
      tileSize,
      crs,
      xyLimits
    )
    corners.push([p.lon, p.lat])
  }
  return corners
}

/**
 * Rectangle (pixel) corners in lon/lat for single-image mode.
 * Uses pixelToLatLon for consistent handling of all CRS types including proj4 reprojection.
 */
function pixelRectLonLatSingle(
  bounds: MercatorBounds,
  width: number,
  height: number,
  x: number,
  y: number,
  crs: CRS,
  latIsAscending?: boolean,
  proj4def?: string | null,
  sourceBounds?: Bounds | null,
  cachedTransformer?: CachedTransformer
): [number, number][] {
  const offsets = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ]
  const corners: [number, number][] = []
  for (const [dx, dy] of offsets) {
    const { lon, lat } = pixelToLatLon(
      x + dx,
      y + dy,
      bounds,
      width,
      height,
      crs,
      latIsAscending,
      proj4def,
      sourceBounds,
      cachedTransformer,
      false // Use pixel corners, not centers
    )
    corners.push([lon, lat])
  }
  return corners
}

export function pixelIntersectsGeometryTiled(
  tile: TileTuple,
  pixelX: number,
  pixelY: number,
  tileSize: number,
  crs: CRS,
  xyLimits: XYLimits,
  geometry: QueryGeometry
): boolean {
  const rect = pixelRectLonLat(tile, pixelX, pixelY, tileSize, crs, xyLimits)
  return rectIntersectsGeometry(rect, geometry)
}

export function pixelIntersectsGeometrySingle(
  bounds: MercatorBounds,
  width: number,
  height: number,
  x: number,
  y: number,
  crs: CRS,
  geometry: QueryGeometry,
  latIsAscending?: boolean,
  proj4def?: string | null,
  sourceBounds?: Bounds | null,
  cachedTransformer?: CachedTransformer
): boolean {
  const rect = pixelRectLonLatSingle(
    bounds,
    width,
    height,
    x,
    y,
    crs,
    latIsAscending,
    proj4def,
    sourceBounds,
    cachedTransformer
  )
  return rectIntersectsGeometry(rect, geometry)
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

/**
 * Converts mercator bounds to pixel coordinates within a data array.
 * Used for single-image mode queries.
 * Supports custom projections via proj4.
 */
export function mercatorBoundsToPixel(
  lng: number,
  lat: number,
  bounds: MercatorBounds,
  width: number,
  height: number,
  crs: CRS,
  latIsAscending?: boolean,
  proj4def?: string | null,
  sourceBounds?: Bounds | null,
  cachedTransformer?: CachedTransformer
): { x: number; y: number } | null {
  // If proj4 is provided, use proj4 to transform lat/lon → source CRS → pixel
  if (proj4def && sourceBounds) {
    const transformer =
      cachedTransformer ?? createWGS84ToSourceTransformer(proj4def)
    const [srcX, srcY] = transformer.forward(lng, lat)

    // Check if within source bounds
    const [xMin, yMin, xMax, yMax] = sourceBounds
    if (srcX < xMin || srcX > xMax || srcY < yMin || srcY > yMax) {
      return null
    }

    // Convert source CRS coords to pixel indices
    const [xPixel, yPixel] = sourceCRSToPixel(
      srcX,
      srcY,
      sourceBounds,
      width,
      height,
      latIsAscending ?? null
    )

    const x = Math.floor(xPixel)
    const y = Math.floor(yPixel)

    if (x < 0 || x >= width || y < 0 || y >= height) {
      return null
    }

    return { x, y }
  }

  let normX: number
  let normY: number

  if (
    crs === 'EPSG:4326' &&
    bounds.latMin !== undefined &&
    bounds.latMax !== undefined
  ) {
    // For equirectangular data, use linear lat mapping
    normX = (lonToMercatorNorm(lng) - bounds.x0) / (bounds.x1 - bounds.x0)
    // Convert lat to mercator for display, but sample linearly in source data
    const latRange = bounds.latMax - bounds.latMin
    if (latRange === 0) return null
    const latNorm = latIsAscending
      ? (lat - bounds.latMin) / latRange
      : (bounds.latMax - lat) / latRange
    normY = latNorm
  } else {
    normX = (lonToMercatorNorm(lng) - bounds.x0) / (bounds.x1 - bounds.x0)
    normY = (latToMercatorNorm(lat) - bounds.y0) / (bounds.y1 - bounds.y0)
  }

  if (normX < 0 || normX > 1 || normY < 0 || normY > 1) {
    return null
  }

  const x = Math.floor(normX * width)
  const y = Math.floor(normY * height)

  return {
    x: Math.min(x, width - 1),
    y: Math.min(y, height - 1),
  }
}
