import { describe, it, expect } from 'vitest'
import { latToMercatorNorm, boundsToMercatorNorm } from './map-utils'
import { MERCATOR_LAT_LIMIT } from './constants'

describe('latToMercatorNorm', () => {
  it('puts the equator at Y=0.5 with north at the top', () => {
    expect(latToMercatorNorm(0)).toBeCloseTo(0.5, 12)
    // The mercator latitude limit is defined as the lat where Y reaches the edge.
    expect(latToMercatorNorm(MERCATOR_LAT_LIMIT)).toBeCloseTo(0, 6)
    expect(latToMercatorNorm(-MERCATOR_LAT_LIMIT)).toBeCloseTo(1, 6)
    // Northern latitudes sit above the equator (smaller Y).
    expect(latToMercatorNorm(45)).toBeLessThan(0.5)
  })

  it('clamps latitude to the mercator limit', () => {
    expect(latToMercatorNorm(89)).toBeCloseTo(
      latToMercatorNorm(MERCATOR_LAT_LIMIT),
      12
    )
    expect(latToMercatorNorm(-89)).toBeCloseTo(
      latToMercatorNorm(-MERCATOR_LAT_LIMIT),
      12
    )
  })
})

describe('boundsToMercatorNorm', () => {
  it('maps full-world EPSG:3857 bounds to the unit square', () => {
    const E = 20037508.342789244
    expect(
      boundsToMercatorNorm(
        { xMin: -E, xMax: E, yMin: -E, yMax: E },
        'EPSG:3857'
      )
    ).toEqual({ x0: 0, y0: 0, x1: 1, y1: 1 })
  })

  it('maps full-world EPSG:4326 bounds to the unit square', () => {
    const b = boundsToMercatorNorm(
      { xMin: -180, xMax: 180, yMin: -90, yMax: 90 },
      'EPSG:4326'
    )
    expect(b.x0).toBe(0)
    expect(b.x1).toBe(1)
    expect(b.y0).toBeCloseTo(0, 6)
    expect(b.y1).toBeCloseTo(1, 6)
  })

  it('maps a regional EPSG:4326 box', () => {
    const b = boundsToMercatorNorm(
      { xMin: 0, xMax: 90, yMin: 0, yMax: 45 },
      'EPSG:4326'
    )
    expect(b.x0).toBeCloseTo(0.5, 12)
    expect(b.x1).toBeCloseTo(0.75, 12)
    expect(b.y1).toBeCloseTo(0.5, 12) // yMin=0 -> equator
    expect(b.y0).toBeLessThan(0.5) // yMax=45 -> north of equator
  })
})
