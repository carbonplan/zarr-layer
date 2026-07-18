/**
 * @module __fixtures__/geometry
 *
 * Reusable GeoJSON query-geometry fixtures. Coordinates are WGS84 lon/lat,
 * matching the public query API.
 */

import type { GeoJSONPolygon, GeoJSONMultiPolygon } from '../query/types'

/** Axis-aligned rectangle as a closed polygon ring (CCW). */
export function rect(
  west: number,
  south: number,
  east: number,
  north: number
): GeoJSONPolygon {
  return {
    type: 'Polygon',
    coordinates: [
      [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
      ],
    ],
  }
}

/** A 0..20 square with a centered 5..15 square hole. */
export const squareWithHole: GeoJSONPolygon = {
  type: 'Polygon',
  coordinates: [
    [
      [0, 0],
      [20, 0],
      [20, 20],
      [0, 20],
      [0, 0],
    ],
    [
      [5, 5],
      [15, 5],
      [15, 15],
      [5, 15],
      [5, 5],
    ],
  ],
}

/** Two disjoint rectangles. */
export const twoBoxes: GeoJSONMultiPolygon = {
  type: 'MultiPolygon',
  coordinates: [
    rect(-150, -10, -140, 10).coordinates,
    rect(140, -10, 150, 10).coordinates,
  ],
}

/**
 * Polygon that crosses the antimeridian, expressed with explicit out-of-range
 * longitudes (170 → 190). After preprocessing this should be detected as
 * crossing and clipped into a west strip (170..180) and east strip (-180..-170).
 */
export const antimeridianPolygon: GeoJSONPolygon = {
  type: 'Polygon',
  coordinates: [
    [
      [170, -10],
      [190, -10],
      [190, 10],
      [170, 10],
      [170, -10],
    ],
  ],
}
