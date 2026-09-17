import { describe, it, expect, vi } from 'vitest'
import { queryData, type QueryContext } from './data-query'
import type { QueryGeometry } from './types'
import { ZarrStore } from '../zarr-store'
import { createProjectionContext } from '../projection-utils'
import { buildMemoryZarrStore, ramp } from '../__fixtures__/memory-zarr'

/**
 * LineString queries end to end against the in-memory Zarr fixture.
 *
 * Fixture: 2-time x 4-lat x 8-lon index ramp (value = t*32 + y*8 + x) on a
 * global 45-degree grid, lat north-first. Pixel (x, y) has center
 * lon = (x + 0.5) / 8 * 360 - 180 and lat = 90 - (y + 0.5) / 4 * 180.
 *
 * The fetch run size is pinned to 3 pixels so an 8-wide line reads several
 * windows and the runs have to be stitched back in order.
 */

vi.mock('../constants', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../constants')>()),
  QUERY_LINE_RUN_MAX_PX: 3,
}))

const WORLD = { xMin: -180, xMax: 180, yMin: -90, yMax: 90 }

async function makeContext(): Promise<QueryContext> {
  const memory = buildMemoryZarrStore({
    arrays: [
      {
        name: 'temp',
        shape: [2, 4, 8],
        chunkShape: [2, 4, 8],
        dimensionNames: ['time', 'lat', 'lon'],
        chunks: { '0/0/0': ramp(2 * 4 * 8) },
      },
      {
        name: 'time',
        shape: [2],
        chunkShape: [2],
        dimensionNames: ['time'],
        chunks: { '0': [10, 20] },
      },
    ],
  })
  const store = new ZarrStore({
    customStore: memory,
    variable: 'temp',
    version: 3,
    bounds: [-180, -90, 180, 90],
    latIsAscending: false,
  })
  await store.initialized
  const zarrArray = await store.getArray()

  return {
    zarrStore: store,
    variable: 'temp',
    selector: {},
    xyLimits: WORLD,
    mercatorBounds: { x0: 0, y0: 0, x1: 1, y1: 1 },
    latIsAscending: false,
    levels: [],
    level: { index: 0, zarrArray, width: 8, height: 4 },
    projection: createProjectionContext({
      crs: 'EPSG:4326',
      proj4def: null,
      xyLimits: WORLD,
    }),
    antimeridianWarnings: new Set(),
    dimensionValues: {},
    isMultiscale: false,
    coordLevelIndex: 0,
    warnedDimensions: new Set(),
  }
}

const line = (...coords: [number, number][]): QueryGeometry => ({
  type: 'LineString',
  coordinates: coords,
})

describe('queryData LineString', () => {
  it('samples every cell along the line in path order across fetch runs', async () => {
    const context = await makeContext()
    const result = await queryData(
      context,
      line([-157.5, 67.5], [157.5, 67.5]),
      { time: 10 }
    )
    expect(result.temp).toEqual([0, 1, 2, 3, 4, 5, 6, 7])
    expect(result.dimensions).toEqual(['lat', 'lon'])
    expect(result.coordinates.lat).toEqual(Array(8).fill(67.5))
    expect(result.coordinates.lon).toEqual([
      -157.5, -112.5, -67.5, -22.5, 22.5, 67.5, 112.5, 157.5,
    ])
  })

  it('reports a monotone distance in meters starting at zero', async () => {
    const context = await makeContext()
    const result = await queryData(
      context,
      line([-157.5, 67.5], [157.5, 67.5]),
      { time: 10 }
    )
    const distance = result.coordinates.distance as number[]
    expect(distance).toHaveLength(8)
    expect(distance[0]).toBe(0)
    for (let i = 1; i < distance.length; i++) {
      expect(distance[i]).toBeGreaterThan(distance[i - 1])
    }
    // The line starts at a cell center, so it reaches the first boundary
    // after 22.5 degrees of longitude at 67.5N (about 0.95e6 m) and then
    // crosses a full 45-degree cell (about 1.9e6 m) between later samples.
    expect(distance[1]).toBeGreaterThan(0.9e6)
    expect(distance[1]).toBeLessThan(1.0e6)
    expect(distance[2] - distance[1]).toBeGreaterThan(1.8e6)
    expect(distance[2] - distance[1]).toBeLessThan(2.0e6)
  })

  it('follows the path direction, not scan order', async () => {
    const context = await makeContext()
    const result = await queryData(
      context,
      line([157.5, -67.5], [-157.5, 67.5]),
      { time: 10 }
    )
    const values = result.temp as number[]
    expect(values[0]).toBe(31)
    expect(values[values.length - 1]).toBe(0)
    expect(values).toHaveLength(new Set(values).size)
  })

  it('samples a vertical line down one column', async () => {
    const context = await makeContext()
    const result = await queryData(context, line([22.5, 67.5], [22.5, -67.5]), {
      time: 10,
    })
    expect(result.temp).toEqual([4, 12, 20, 28])
  })

  it('walks an explicit antimeridian crossing in order', async () => {
    const context = await makeContext()
    const eastward = await queryData(
      context,
      line([112.5, 67.5], [202.5, 67.5]),
      { time: 10 }
    )
    expect(eastward.temp).toEqual([6, 7, 0])
    expect(eastward.coordinates.lon).toEqual([112.5, 157.5, -157.5])

    const westward = await queryData(
      context,
      line([202.5, 67.5], [112.5, 67.5]),
      { time: 10 }
    )
    expect(westward.temp).toEqual([0, 7, 6])
    const distance = westward.coordinates.distance as number[]
    expect(distance[0]).toBe(0)
    expect(distance[1]).toBeGreaterThan(distance[0])
    expect(distance[2]).toBeGreaterThan(distance[1])
  })

  it('reads an in-range line literally across the prime meridian', async () => {
    const context = await makeContext()
    const result = await queryData(
      context,
      line([112.5, 67.5], [-112.5, 67.5]),
      { time: 10 }
    )
    expect(result.temp).toEqual([6, 5, 4, 3, 2, 1])
  })

  it('nests per-channel profiles for a multi-value selector', async () => {
    const context = await makeContext()
    const result = await queryData(
      context,
      line([-157.5, 67.5], [-22.5, 67.5]),
      { time: [10, 20] }
    )
    expect(result.temp).toEqual({
      10: [0, 1, 2, 3],
      20: [32, 33, 34, 35],
    })
    expect(result.coordinates.distance).toHaveLength(4)
  })

  it('omits distance along with the spatial coordinates when asked', async () => {
    const context = await makeContext()
    const result = await queryData(
      context,
      line([-157.5, 67.5], [-22.5, 67.5]),
      { time: 10 },
      { includeSpatialCoordinates: false }
    )
    expect(result.temp).toEqual([0, 1, 2, 3])
    expect(result.coordinates.lon).toEqual([])
    expect(result.coordinates.distance).toBeUndefined()
  })

  it('returns an empty result for a line entirely off the raster', async () => {
    const context = await makeContext()
    const clipped = {
      ...context,
      level: {
        ...context.level!,
        xyLimits: { xMin: -180, xMax: 0, yMin: -90, yMax: 90 },
        width: 4,
      },
    }
    const result = await queryData(clipped, line([45, 10], [135, 10]), {
      time: 10,
    })
    expect(result.temp).toEqual([])
    expect(result.coordinates).toEqual({ lat: [], lon: [], distance: [] })
  })

  it('propagates an abort', async () => {
    const context = await makeContext()
    const controller = new AbortController()
    controller.abort()
    await expect(
      queryData(
        context,
        line([-157.5, 67.5], [157.5, 67.5]),
        { time: 10 },
        { signal: controller.signal }
      )
    ).rejects.toMatchObject({ name: 'AbortError' })
  })
})
