import { describe, it, expect } from 'vitest'
import {
  buildScanlineTable,
  preprocessQueryGeometry,
  transformGeometryToPixelSpace,
  computePixelBoundsFromGeometry,
} from './query-utils'
import type { QueryGeometry } from './types'
import type { Bounds } from '../types'
import { WEB_MERCATOR_EXTENT } from '../constants'
import {
  rect,
  squareWithHole,
  antimeridianPolygon,
} from '../__fixtures__/geometry'

describe('buildScanlineTable', () => {
  it('returns the two edge crossings for a simple square', () => {
    const square = rect(2, 2, 8, 8) as QueryGeometry
    const row5 = buildScanlineTable(square, 0, 10).get(5)
    expect(row5).toBeDefined()
    expect(row5!.map((x) => Math.round(x))).toEqual([2, 8])
  })

  it('uses center-based sampling (row + 0.5)', () => {
    // Triangle whose tip is at y=2.6: row 2 center (2.5) is above the tip and
    // excluded; row 3 center (3.5) is inside.
    const triangle: QueryGeometry = {
      type: 'Polygon',
      coordinates: [
        [
          [5, 2.6],
          [10, 8],
          [0, 8],
          [5, 2.6],
        ],
      ],
    }
    const table = buildScanlineTable(triangle, 0, 10)
    expect(table.has(2)).toBe(false)
    expect(table.has(3)).toBe(true)
  })

  it('produces four crossings across a polygon with a hole', () => {
    const row10 = buildScanlineTable(
      squareWithHole as QueryGeometry,
      0,
      20
    ).get(10)
    expect(row10).toBeDefined()
    expect(row10!.map((x) => Math.round(x))).toEqual([0, 5, 15, 20])
  })

  it('unions disjoint multipolygon members into separate intervals', () => {
    const disjoint: QueryGeometry = {
      type: 'MultiPolygon',
      coordinates: [
        rect(0, 0, 2, 10).coordinates,
        rect(5, 0, 7, 10).coordinates,
      ],
    }
    const row5 = buildScanlineTable(disjoint, 0, 10).get(5)
    expect(row5!.map((x) => Math.round(x))).toEqual([0, 2, 5, 7])
  })

  it('merges overlapping multipolygon members into a single interval', () => {
    const overlapping: QueryGeometry = {
      type: 'MultiPolygon',
      coordinates: [
        rect(0, 0, 5, 10).coordinates,
        rect(3, 0, 8, 10).coordinates,
      ],
    }
    const row5 = buildScanlineTable(overlapping, 0, 10).get(5)
    expect(row5!.map((x) => Math.round(x))).toEqual([0, 8])
  })
})

describe('preprocessQueryGeometry', () => {
  it('passes a non-crossing polygon through and closes its ring', () => {
    const { geometry, bbox } = preprocessQueryGeometry(rect(-10, -10, 10, 10))
    expect(bbox.crossesAntimeridian).toBe(false)
    expect(geometry.type).toBe('Polygon')
  })

  it('detects an antimeridian crossing and splits into west/east strips', () => {
    const { geometry, bbox } = preprocessQueryGeometry(antimeridianPolygon)
    expect(bbox.crossesAntimeridian).toBe(true)
    // west (170) > east (-170) is the crossing invariant.
    expect(bbox.west).toBeCloseTo(170, 6)
    expect(bbox.east).toBeCloseTo(-170, 6)
    expect(geometry.type).toBe('MultiPolygon')
    if (geometry.type === 'MultiPolygon') {
      expect(geometry.coordinates).toHaveLength(2)
    }
  })

  it('returns a point unchanged', () => {
    const point: QueryGeometry = { type: 'Point', coordinates: [5, 6] }
    const { geometry, bbox } = preprocessQueryGeometry(point)
    expect(geometry).toEqual(point)
    expect(bbox).toMatchObject({
      west: 5,
      east: 5,
      south: 6,
      north: 6,
      crossesAntimeridian: false,
    })
  })
})

describe('transformGeometryToPixelSpace', () => {
  const worldBounds: Bounds = [-180, -90, 180, 90]

  it('maps a polygon linearly under EPSG:4326', () => {
    const result = transformGeometryToPixelSpace(
      rect(-90, -45, 90, 45),
      worldBounds,
      360,
      180,
      'EPSG:4326',
      false // north at top
    )
    expect(result).not.toBeNull()
    expect(result!.type).toBe('Polygon')
    // lon -90 -> x 90, lon 90 -> x 270; lat 45 -> y 45 (north), lat -45 -> y 135.
    const ring = (result as { coordinates: number[][][] }).coordinates[0]
    const xs = ring.map((p) => p[0])
    const ys = ring.map((p) => p[1])
    expect(Math.min(...xs)).toBeCloseTo(90, 6)
    expect(Math.max(...xs)).toBeCloseTo(270, 6)
    expect(Math.min(...ys)).toBeCloseTo(45, 6)
    expect(Math.max(...ys)).toBeCloseTo(135, 6)
  })

  it('keeps a polygon past the mercator pole limit finite and non-collapsed', () => {
    // lat 86 is past the mercator limit (~85.05°); the projection must clamp
    // rather than blow up to infinity. EPSG:3857 bounds are in meters, so the
    // source-CRS forward output and the bounds share units (a realistic call).
    const E = WEB_MERCATOR_EXTENT
    const nearPole: QueryGeometry = {
      type: 'Polygon',
      coordinates: [
        [
          [0, 85],
          [10, 85],
          [10, 86],
          [0, 86],
          [0, 85],
        ],
      ],
    }
    const result = transformGeometryToPixelSpace(
      nearPole,
      [-E, -E, E, E],
      100,
      100,
      'EPSG:3857'
    )
    // transformGeometryToPixelSpace *drops* non-finite vertices, so a bare
    // "no NaN in output" check is vacuous — a singularity would silently
    // collapse the ring (or null the result) instead of emitting NaN. Assert
    // the ring actually survived, then that every surviving vertex is finite.
    expect(result).not.toBeNull()
    expect(result!.type).toBe('Polygon')
    const rings = (result as { coordinates: number[][][] }).coordinates
    expect(rings[0].length).toBeGreaterThanOrEqual(4)
    for (const ring of rings) {
      for (const [x, y] of ring) {
        expect(Number.isFinite(x)).toBe(true)
        expect(Number.isFinite(y)).toBe(true)
      }
    }
  })
})

describe('computePixelBoundsFromGeometry', () => {
  const worldBounds: Bounds = [-180, -90, 180, 90]

  it('returns a single-pixel rect for a point', () => {
    const point: QueryGeometry = { type: 'Point', coordinates: [0, 0] }
    const r = computePixelBoundsFromGeometry(
      point,
      worldBounds,
      360,
      180,
      'EPSG:4326',
      false
    )
    expect(r).toEqual({ minX: 180, maxX: 181, minY: 90, maxY: 91 })
  })

  it('computes a tight bbox for a polygon', () => {
    const r = computePixelBoundsFromGeometry(
      rect(-90, -45, 90, 45),
      worldBounds,
      360,
      180,
      'EPSG:4326',
      false
    )
    expect(r).not.toBeNull()
    expect(r!.minX).toBe(90)
    expect(r!.minY).toBe(45)
    expect(r!.maxX).toBeGreaterThanOrEqual(270)
    expect(r!.maxY).toBeGreaterThanOrEqual(135)
  })
})
