import type { ZarrSelectorsProps } from '../types'

/**
 * Result from a point query at a specific geographic location.
 */
export interface PointQueryResult {
  /** The queried longitude */
  lng: number
  /** The queried latitude */
  lat: number
  /** Primary value at the point (or null if no data) */
  value: number | null
  /** Values per band if multi-band data */
  bandValues?: Record<string, number | null>
  /** Tile coordinates where the data was found */
  tile?: { z: number; x: number; y: number }
  /** Pixel coordinates within the tile */
  pixel?: { x: number; y: number }
}

/**
 * Nested values structure for multi-dimensional region queries.
 */
export interface NestedValues {
  [key: string]: number[] | NestedValues
  [key: number]: number[] | NestedValues
}

/**
 * Values from a region query. Can be flat array or nested when selector has array values.
 *
 * Flat: `number[]` when selector = `{ month: 1 }`
 * Nested: `{ 1: number[], 2: number[] }` when selector = `{ month: [1, 2] }`
 */
export type RegionValues = number[] | NestedValues

/**
 * Result from a region query within a geographic polygon.
 */
export interface RegionQueryResult {
  /** Values within the region (flat or nested based on selector) */
  values: RegionValues
  /** Values per band if multi-band data */
  bandValues?: Record<string, RegionValues>
  /** Dimension names in order (e.g., ['month', 'lat', 'lon']) */
  dimensions: string[]
  /** Coordinate arrays for each dimension */
  coordinates: {
    lat: number[]
    lon: number[]
    [key: string]: (number | string)[]
  }
}

/**
 * Selector for region queries - controls which tiles/slices to load.
 * Can override the layer's render selector.
 */
export type QuerySelector = Record<
  string,
  number | number[] | string | string[] | ZarrSelectorsProps
>

/**
 * Internal structure for point values during region iteration.
 * Used to build nested result structures.
 */
export interface PointValueEntry {
  /** Dimension keys for nested placement (empty for flat results) */
  keys: (string | number)[]
  /** The data value */
  value: number
}

/**
 * Bounding box for a geographic region.
 */
export interface BoundingBox {
  west: number
  east: number
  south: number
  north: number
}

/**
 * GeoJSON Polygon geometry.
 */
export interface GeoJSONPolygon {
  type: 'Polygon'
  coordinates: number[][][]
}

/**
 * GeoJSON MultiPolygon geometry.
 */
export interface GeoJSONMultiPolygon {
  type: 'MultiPolygon'
  coordinates: number[][][][]
}

/**
 * Supported GeoJSON geometry types for region queries.
 */
export type QueryGeometry = GeoJSONPolygon | GeoJSONMultiPolygon
