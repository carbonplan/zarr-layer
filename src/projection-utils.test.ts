import { describe, it, expect } from 'vitest'
import {
  sourceCRSToPixel,
  pixelToSourceCRS,
  clampLatLonToProj4def,
  createWGS84ToSourceTransformer,
  createTransformer,
  sampleEdgesToMercatorBounds,
} from './projection-utils'
import { MERCATOR_LAT_LIMIT, WEB_MERCATOR_EXTENT } from './constants'
import type { Bounds } from './types'

const bounds10: Bounds = [0, 0, 10, 10]

describe('sourceCRSToPixel', () => {
  it('uses an edge-to-edge model (xMin -> 0, xMax -> width)', () => {
    expect(sourceCRSToPixel(0, 0, bounds10, 10, 10)).toEqual([0, 0])
    expect(sourceCRSToPixel(5, 5, bounds10, 10, 10)).toEqual([5, 5])
    expect(sourceCRSToPixel(10, 10, bounds10, 10, 10)).toEqual([10, 10])
  })

  it('flips Y when latIsAscending is false (row 0 = north)', () => {
    // y=yMin (south) -> bottom row (pixel = height); y=yMax (north) -> row 0.
    expect(sourceCRSToPixel(0, 0, bounds10, 10, 10, false)).toEqual([0, 10])
    expect(sourceCRSToPixel(0, 10, bounds10, 10, 10, false)).toEqual([0, 0])
  })

  it('falls back to the grid center for degenerate bounds', () => {
    const bad: Bounds = [0, 0, 0, 10] // xMax <= xMin
    expect(sourceCRSToPixel(5, 5, bad, 10, 10)).toEqual([5, 5])
  })
})

describe('pixelToSourceCRS', () => {
  it('round-trips with sourceCRSToPixel (ascending)', () => {
    const [px, py] = sourceCRSToPixel(3, 7, bounds10, 10, 10)
    const [x, y] = pixelToSourceCRS(px, py, bounds10, 10, 10)
    expect(x).toBeCloseTo(3, 12)
    expect(y).toBeCloseTo(7, 12)
  })

  it('round-trips with sourceCRSToPixel (descending)', () => {
    const [px, py] = sourceCRSToPixel(3, 7, bounds10, 10, 10, false)
    const [x, y] = pixelToSourceCRS(px, py, bounds10, 10, 10, false)
    expect(x).toBeCloseTo(3, 12)
    expect(y).toBeCloseTo(7, 12)
  })

  it('falls back to the bounds center for degenerate bounds', () => {
    const bad: Bounds = [0, 0, 0, 10]
    expect(pixelToSourceCRS(5, 5, bad, 10, 10)).toEqual([0, 5])
  })
})

describe('clampLatLonToProj4def', () => {
  it('clamps latitude to the mercator limit for EPSG:3857', () => {
    expect(clampLatLonToProj4def(200, 89, 'EPSG:3857')).toEqual([
      200,
      MERCATOR_LAT_LIMIT,
    ])
    expect(clampLatLonToProj4def(0, -89, 'EPSG:3857')).toEqual([
      0,
      -MERCATOR_LAT_LIMIT,
    ])
  })

  it('leaves coordinates untouched for other projections', () => {
    expect(clampLatLonToProj4def(200, 89, 'EPSG:4326')).toEqual([200, 89])
  })
})

describe('createWGS84ToSourceTransformer (EPSG:3857)', () => {
  const t = createWGS84ToSourceTransformer('EPSG:3857')

  it('maps the origin to the mercator origin', () => {
    const [x, y] = t.forward(0, 0)
    expect(x).toBeCloseTo(0, 6)
    expect(y).toBeCloseTo(0, 6)
  })

  it('maps ±180° lon to the mercator world edges', () => {
    expect(t.forward(180, 0)[0]).toBeCloseTo(WEB_MERCATOR_EXTENT, 2)
    expect(t.forward(-180, 0)[0]).toBeCloseTo(-WEB_MERCATOR_EXTENT, 2)
  })

  it('round-trips forward/inverse', () => {
    const [x, y] = t.forward(-122.4, 37.8)
    const [lon, lat] = t.inverse(x, y)
    expect(lon).toBeCloseTo(-122.4, 9)
    expect(lat).toBeCloseTo(37.8, 9)
  })

  it('throws a helpful error for an invalid proj4 string', () => {
    expect(() => createWGS84ToSourceTransformer('not-a-projection')).toThrow(
      /Invalid proj4 string/
    )
  })
})

describe('sampleEdgesToMercatorBounds', () => {
  it('maps a full-mercator extent to the unit square via an identity transformer', () => {
    const identity = {
      forward: (x: number, y: number): [number, number] => [x, y],
    }
    const result = sampleEdgesToMercatorBounds(
      {
        xMin: -WEB_MERCATOR_EXTENT,
        xMax: WEB_MERCATOR_EXTENT,
        yMin: -WEB_MERCATOR_EXTENT,
        yMax: WEB_MERCATOR_EXTENT,
      },
      identity,
      4
    )
    expect(result).not.toBeNull()
    expect(result!.x0).toBeCloseTo(0, 12)
    expect(result!.y0).toBeCloseTo(0, 12)
    expect(result!.x1).toBeCloseTo(1, 12)
    expect(result!.y1).toBeCloseTo(1, 12)
  })

  it('returns null when no sample projects to a finite point', () => {
    const broken = { forward: (): [number, number] => [NaN, NaN] }
    expect(
      sampleEdgesToMercatorBounds(
        { xMin: 0, xMax: 1, yMin: 0, yMax: 1 },
        broken,
        4
      )
    ).toBeNull()
  })
})

describe('createTransformer (source -> EPSG:3857)', () => {
  it('projects EPSG:4326 source coordinates to Web Mercator meters', () => {
    const t = createTransformer('EPSG:4326', [-180, -90, 180, 90])
    // Origin maps to origin...
    const [x, y] = t.forward(0, 0)
    expect(x).toBeCloseTo(0, 6)
    expect(y).toBeCloseTo(0, 6)
    // ...and ±180° lon reaches the mercator world edges. An origin-only check
    // can't tell a real projection from an identity/no-op; this pins the scale.
    expect(t.forward(180, 0)[0]).toBeCloseTo(WEB_MERCATOR_EXTENT, 2)
    expect(t.forward(-180, 0)[0]).toBeCloseTo(-WEB_MERCATOR_EXTENT, 2)
    expect(t.bounds).toEqual([-180, -90, 180, 90])
  })
})
