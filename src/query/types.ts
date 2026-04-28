/**
 * Nested values structure for multi-dimensional data queries.
 */
export interface NestedValues {
  [key: string]: number[] | NestedValues
  [key: number]: number[] | NestedValues
}

/**
 * Values from a data query. Can be flat array or nested when selector has array values.
 *
 * Flat: `number[]` when selector = `{ month: 1 }`
 * Nested: `{ 1: number[], 2: number[] }` when selector = `{ month: [1, 2] }`
 */
export type QueryDataValues = number[] | NestedValues

/**
 * Result from a query (point or region).
 * Matches carbonplan/maps structure: { [variable]: values, dimensions, coordinates }
 *
 * Spatial result dimensions and coordinate keys follow the emitted coordinate space:
 * - WGS84 fallback: `lat`/`lon`
 * - Source CRS: the store's spatial axis names (for example `y`/`x`,
 *   `lat`/`lon`, or `latitude`/`longitude`)
 */
export interface QueryResult {
  /** Variable name mapped to its values (flat array or nested based on selector) */
  [variable: string]:
    | QueryDataValues
    | string[]
    | { [key: string]: (number | string)[] }
  /** Dimension names in order, using the same spatial keys as coordinates */
  dimensions: string[]
  /** Coordinate arrays for each dimension */
  coordinates: {
    [key: string]: (number | string)[]
  }
}

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
 * GeoJSON Point geometry.
 */
export interface GeoJSONPoint {
  type: 'Point'
  coordinates: [number, number]
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
 * Supported GeoJSON geometry types for queries.
 */
export type QueryGeometry = GeoJSONPoint | GeoJSONPolygon | GeoJSONMultiPolygon

/**
 * Transform options for query results to match rendered values.
 */
export interface QueryTransformOptions {
  /** Scale factor to apply: value * scaleFactor */
  scaleFactor?: number
  /** Offset to add: value + addOffset (applied after scaleFactor) */
  addOffset?: number
  /** Fill value to filter out (along with NaN/Infinity) */
  fillValue?: number | null
}

/**
 * Options for queryData calls.
 */
export interface QueryOptions {
  /** AbortSignal to cancel the query. */
  signal?: AbortSignal
  /** Include per-pixel coordinates in the result. Defaults to true. */
  includeSpatialCoordinates?: boolean
}
