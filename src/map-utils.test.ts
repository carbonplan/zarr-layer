import { describe, it, expect } from 'vitest'
import {
  latToMercatorNorm,
  boundsToMercatorNorm,
  normalizeLongitudeExtent,
  wrapLongitudeIntoExtent,
} from './map-utils'
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

describe('normalizeLongitudeExtent', () => {
  it('leaves an extent already in -180-180 alone', () => {
    expect(normalizeLongitudeExtent(-120, -60, 1)).toEqual({
      xMin: -120,
      xMax: -60,
    })
  })

  it('folds a 0-360 extent', () => {
    expect(normalizeLongitudeExtent(200, 300, 1)).toEqual({
      xMin: -160,
      xMax: -60,
    })
  })

  it('leaves a global 0-360 extent unfolded', () => {
    // The fold needs both edges past 180 to tell 0-360 degrees from projected
    // meters, and a global grid's western edge sits at 0. Such a store still
    // needs an explicit `bounds` to render across the antimeridian.
    expect(normalizeLongitudeExtent(0, 360, 45)).toEqual({ xMin: 0, xMax: 360 })
  })

  it('snaps a global extent that misses the antimeridian by a hair', () => {
    const { xMin, xMax } = normalizeLongitudeExtent(-179.9999, 180.0001, 0.25)
    expect(xMin).toBe(-180)
    expect(xMax).toBe(180)
  })

  it('leaves an extent one cell short of global unsnapped', () => {
    // 359.75 deg of span is a real gap, not float drift.
    expect(normalizeLongitudeExtent(-180, 179.75, 0.25)).toEqual({
      xMin: -180,
      xMax: 179.75,
    })
  })

  it('leaves an extent half a cell short of global unsnapped', () => {
    // This is typical of a global dataset with a node spatial registration, e.g.
    // ClimateSET's Zarr version of CMIP6.
    expect(normalizeLongitudeExtent(-180.5, 179.5, 1)).toEqual({
      xMin: -180.5,
      xMax: 179.5,
    })
  })

  it('leaves a regional extent near 180 unsnapped', () => {
    expect(normalizeLongitudeExtent(170, 179.9, 0.25)).toEqual({
      xMin: 170,
      xMax: 179.9,
    })
  })

  it('leaves an extent alone when the cell width is unknown', () => {
    expect(normalizeLongitudeExtent(-179.9999, 180.0001, NaN)).toEqual({
      xMin: -179.9999,
      xMax: 180.0001,
    })
  })
})

describe('wrapLongitudeIntoExtent', () => {
  it('leaves a longitude already inside the extent alone', () => {
    expect(wrapLongitudeIntoExtent(10, -180.5, 179.5)).toBe(10)
  })

  it('shifts a longitude in the wrapped strip onto the extent', () => {
    expect(wrapLongitudeIntoExtent(179.75, -180.5, 179.5)).toBeCloseTo(-180.25)
    expect(wrapLongitudeIntoExtent(-179.75, -170, 190)).toBeCloseTo(180.25)
  })

  it('shifts by as many turns as it takes', () => {
    expect(wrapLongitudeIntoExtent(539.75, -180.5, 179.5)).toBeCloseTo(-180.25)
  })

  it('leaves a longitude alone when no shift fits', () => {
    expect(wrapLongitudeIntoExtent(200, -100, 100)).toBe(200)
  })
})
