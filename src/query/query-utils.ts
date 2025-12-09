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
} from '../map-utils'
import type { CRS } from '../types'
import type { BoundingBox, QueryGeometry } from './types'
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
export function computeBoundingBox(geometry: QueryGeometry): BoundingBox {
  let west = Infinity
  let east = -Infinity
  let south = Infinity
  let north = -Infinity

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
  geometry: QueryGeometry
): boolean {
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
