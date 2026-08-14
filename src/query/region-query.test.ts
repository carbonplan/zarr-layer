import { describe, it, expect } from 'vitest'
import { queryRegion, findSpatialDimNames } from './region-query'
import type { QueryGeometry } from './types'
import type { Bounds, DimIndicesProps } from '../types'
import { indexRamp, indexToXY } from '../__fixtures__/grids'
import { rect } from '../__fixtures__/geometry'

const WORLD: Bounds = [-180, -90, 180, 90]

/**
 * Query an `indexRamp` grid and return the selected pixels as [x, y] pairs.
 * Because each pixel's value is its own flat index, the returned values map
 * back to exactly which pixels the query selected.
 */
function selectedPixels(
  geometry: QueryGeometry,
  width: number,
  height: number
): Array<[number, number]> {
  const { data } = indexRamp(width, height)
  const result = queryRegion(
    'v',
    geometry,
    {},
    data,
    width,
    height,
    ['lat', 'lon'],
    {},
    WORLD,
    'EPSG:4326',
    1,
    undefined,
    undefined,
    false // latIsAscending = false (north at top)
  )
  return (result.v as number[]).map((value) =>
    indexToXY(Math.round(value), width)
  )
}

describe('queryRegion — coverage', () => {
  it('selects every pixel for a whole-world polygon', () => {
    const pixels = selectedPixels(rect(-180, -90, 180, 90), 10, 10)
    expect(pixels).toHaveLength(100)
  })

  it('selects the interior (center-based) for an inset polygon', () => {
    // rect maps to the pixel box [1,1]..[9,9]; centers inside cover x,y in 1..8.
    const pixels = selectedPixels(rect(-144, -72, 144, 72), 10, 10)
    expect(pixels).toHaveLength(64)
    expect(pixels).toContainEqual([5, 5])
    expect(pixels).not.toContainEqual([0, 0])
    expect(pixels).not.toContainEqual([9, 9])
  })

  it('selects nothing for a polygon smaller than one pixel', () => {
    // ~0.01° box near the origin — no pixel center falls inside.
    const pixels = selectedPixels(rect(-0.005, -0.005, 0.005, 0.005), 10, 10)
    expect(pixels).toHaveLength(0)
  })

  it('reads the single pixel under a point', () => {
    const { data } = indexRamp(10, 10)
    const point: QueryGeometry = { type: 'Point', coordinates: [0, 0] }
    const result = queryRegion(
      'v',
      point,
      {},
      data,
      10,
      10,
      ['lat', 'lon'],
      {},
      WORLD,
      'EPSG:4326',
      1,
      undefined,
      undefined,
      false
    )
    // Origin maps to pixel (5, 5) -> flat index 55.
    expect(result.v).toEqual([55])
  })
})

describe('queryRegion — value handling', () => {
  it('filters fill values / NaN and applies scale + offset', () => {
    const data = new Float32Array([10, -9999, 20, NaN]) // 2x2, row-major
    const result = queryRegion(
      'v',
      rect(-180, -90, 180, 90),
      {},
      data,
      2,
      2,
      ['lat', 'lon'],
      {},
      WORLD,
      'EPSG:4326',
      1,
      undefined,
      undefined,
      false,
      { scaleFactor: 2, addOffset: 1, fillValue: -9999 }
    )
    // 10*2+1 = 21, 20*2+1 = 41; -9999 (fill) and NaN dropped.
    expect(result.v).toEqual([21, 41])
  })

  it('reports spatial dimension names and coordinate arrays', () => {
    const { data } = indexRamp(4, 4)
    const result = queryRegion(
      'v',
      rect(-180, -90, 180, 90),
      {},
      data,
      4,
      4,
      ['lat', 'lon'],
      {},
      WORLD,
      'EPSG:4326',
      1,
      undefined,
      undefined,
      false
    )
    expect(result.dimensions).toEqual(['lat', 'lon'])
    expect(result.coordinates).toHaveProperty('lat')
    expect(result.coordinates).toHaveProperty('lon')
    expect((result.coordinates.lat as number[]).length).toBe(
      (result.v as number[]).length
    )
  })

  it('returns an empty result for null data', () => {
    const result = queryRegion(
      'v',
      rect(-10, -10, 10, 10),
      {},
      null,
      10,
      10,
      ['lat', 'lon'],
      {},
      WORLD,
      'EPSG:4326'
    )
    expect(result.v).toEqual([])
  })
})

describe('findSpatialDimNames', () => {
  it('matches common spatial aliases', () => {
    expect(findSpatialDimNames(['lat', 'lon'])).toEqual({
      yDim: 'lat',
      xDim: 'lon',
    })
    expect(findSpatialDimNames(['latitude', 'longitude'])).toEqual({
      yDim: 'latitude',
      xDim: 'longitude',
    })
    expect(findSpatialDimNames(['y', 'x'])).toEqual({ yDim: 'y', xDim: 'x' })
  })

  it('falls back to lat/lon when no alias matches', () => {
    expect(findSpatialDimNames(['time', 'band'])).toEqual({
      yDim: 'lat',
      xDim: 'lon',
    })
  })

  it('prefers dimIndices overrides', () => {
    const dimIndices: DimIndicesProps = {
      lat: { name: 'south_north', index: 0, array: null },
      lon: { name: 'west_east', index: 1, array: null },
    }
    expect(
      findSpatialDimNames(['south_north', 'west_east'], dimIndices)
    ).toEqual({
      yDim: 'south_north',
      xDim: 'west_east',
    })
  })
})
