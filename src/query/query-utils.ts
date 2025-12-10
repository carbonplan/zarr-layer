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
} from '../map-utils'
import type { CRS } from '../types'
import type { BoundingBox, QueryDataGeometry } from './types'
import { MERCATOR_LAT_LIMIT } from '../constants'

/**
 * Converts latitude to normalized mercator Y coordinate [0, 1].
 * This is the carbonplan/maps formula for latitude correction.
 *
 * From carbonplan/maps src/utils.js:81-88
 */
export function mercatorYFromLat(lat: number): number {
  return (
    (180 -
      (180 / Math.PI) *
        Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360))) /
    360
  )
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
export function computeBoundingBox(geometry: QueryDataGeometry): BoundingBox {
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
  geometry: QueryDataGeometry
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
  geometry: QueryDataGeometry
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
 */
function pixelRectLonLatSingle(
  bounds: MercatorBounds,
  width: number,
  height: number,
  x: number,
  y: number,
  crs: CRS
): [number, number][] {
  const offsets = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ]
  const corners: [number, number][] = []
  for (const [dx, dy] of offsets) {
    const mercX = bounds.x0 + ((x + dx) / width) * (bounds.x1 - bounds.x0)
    const mercY = bounds.y0 + ((y + dy) / height) * (bounds.y1 - bounds.y0)
    const lon = mercatorNormToLon(mercX)
    const lat =
      crs === 'EPSG:4326' &&
      bounds.latMin !== undefined &&
      bounds.latMax !== undefined
        ? bounds.latMax -
          ((mercY - bounds.y0) / (bounds.y1 - bounds.y0)) *
            (bounds.latMax - bounds.latMin)
        : mercatorNormToLat(mercY)
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
  geometry: QueryDataGeometry
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
  geometry: QueryDataGeometry
): boolean {
  const rect = pixelRectLonLatSingle(bounds, width, height, x, y, crs)
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
  geometry: QueryDataGeometry,
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
 */
export function mercatorBoundsToPixel(
  lng: number,
  lat: number,
  bounds: MercatorBounds,
  width: number,
  height: number,
  crs: CRS
): { x: number; y: number } | null {
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
    const latNorm = (bounds.latMax - lat) / (bounds.latMax - bounds.latMin)
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
