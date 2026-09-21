import { describe, it, expect, vi } from 'vitest'
import { queryData, type QueryContext } from './data-query'
import type { QueryGeometry } from './types'
import { ZarrStore } from '../zarr-store'
import { createProjectionContext } from '../projection-utils'
import { buildMemoryZarrStore, ramp } from '../__fixtures__/memory-zarr'
import * as zarr from 'zarrita'

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

vi.mock('zarrita', async (importOriginal) => {
  const actual = await importOriginal<typeof import('zarrita')>()
  return { ...actual, get: vi.fn(actual.get) }
})

vi.mock('../constants', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../constants')>()),
  QUERY_LINE_RUN_MAX_PX: 3,
}))

const WORLD = { xMin: -180, xMax: 180, yMin: -90, yMax: 90 }

async function makeContext(
  values: number[] = ramp(2 * 4 * 8),
  dims: { extra: string; lat: string; lon: string } = {
    extra: 'time',
    lat: 'lat',
    lon: 'lon',
  }
): Promise<QueryContext> {
  const memory = buildMemoryZarrStore({
    arrays: [
      {
        name: 'temp',
        shape: [2, 4, 8],
        chunkShape: [2, 4, 8],
        dimensionNames: [dims.extra, dims.lat, dims.lon],
        chunks: { '0/0/0': values },
      },
      {
        name: dims.extra,
        shape: [2],
        chunkShape: [2],
        dimensionNames: [dims.extra],
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
    spatialDimensions: { lat: dims.lat, lon: dims.lon },
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

  it('reads a long line as several bounded windows', async () => {
    const context = await makeContext()
    const get = vi.mocked(zarr.get)
    get.mockClear()
    await queryData(context, line([-157.5, 67.5], [157.5, 67.5]), { time: 10 })

    const windows = get.mock.calls
      .map(([, slices]) => slices as unknown[])
      .filter((slices) => slices?.length === 3)
      .map((slices) => slices[2] as { start: number; stop: number })
    expect(windows.map((w) => [w.start, w.stop])).toEqual([
      [0, 3],
      [3, 6],
      [6, 8],
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

  it('returns a flat profile with no selector and a labelled one for a one-element array', async () => {
    const context = await makeContext()
    const geometry = line([-157.5, 67.5], [-67.5, 67.5])
    const unselected = await queryData(context, geometry)
    expect(unselected.temp).toEqual([0, 1, 2])

    const single = await queryData(context, geometry, { time: [20] })
    expect(single.temp).toEqual({ 20: [32, 33, 34] })
    expect(single.coordinates.distance).toHaveLength(3)
  })

  it('returns empty per-sample coordinates, distance included, when asked to omit them', async () => {
    const context = await makeContext()
    const result = await queryData(
      context,
      line([-157.5, 67.5], [-22.5, 67.5]),
      { time: 10 },
      { includeSpatialCoordinates: false }
    )
    expect(result.temp).toEqual([0, 1, 2, 3])
    expect(result.coordinates).toEqual({ lat: [], lon: [], distance: [] })
  })

  it('keeps every series aligned with distance when a cell is fill in only some', async () => {
    const values = ramp(2 * 4 * 8)
    values[1] = NaN // time 10, pixel (1, 0)
    const context = await makeContext(values)
    const result = await queryData(
      context,
      line([-157.5, 67.5], [-67.5, 67.5]),
      { time: [10, 20] }
    )
    expect(result.temp).toEqual({ 10: [0, NaN, 2], 20: [32, 33, 34] })
    expect(result.coordinates.lon).toEqual([-157.5, -112.5, -67.5])
    expect(result.coordinates.distance).toHaveLength(3)
  })

  it('stays aligned when one series is fill across a whole fetch run', async () => {
    const values = ramp(2 * 4 * 8)
    // Runs are 3 pixels wide: x 0-2, 3-5, 6-7. Blank time 10 across the
    // middle run, and both series at x = 6 so that cell drops out entirely.
    for (const x of [3, 4, 5, 6]) values[x] = NaN
    values[32 + 6] = NaN
    const context = await makeContext(values)
    const result = await queryData(
      context,
      line([-157.5, 67.5], [157.5, 67.5]),
      { time: [10, 20] }
    )
    expect(result.temp).toEqual({
      10: [0, 1, 2, NaN, NaN, NaN, 7],
      20: [32, 33, 34, 35, 36, 37, 39],
    })
    expect(result.coordinates.lon).toEqual([
      -157.5, -112.5, -67.5, -22.5, 22.5, 67.5, 157.5,
    ])
    expect(result.coordinates.distance).toHaveLength(7)
  })

  it('drops a fill cell from a single-series profile and leaves a gap in distance', async () => {
    const values = ramp(2 * 4 * 8)
    values[1] = NaN
    const context = await makeContext(values)
    const result = await queryData(
      context,
      line([-157.5, 67.5], [-67.5, 67.5]),
      { time: 10 }
    )
    expect(result.temp).toEqual([0, 2])
    expect(result.coordinates.lon).toEqual([-157.5, -67.5])
  })

  it('reports distance under a caller-chosen key', async () => {
    const context = await makeContext()
    const result = await queryData(
      context,
      line([-157.5, 67.5], [-67.5, 67.5]),
      { time: 10 },
      { distanceKey: 'along' }
    )
    expect(result.coordinates.along).toHaveLength(3)
    expect(result.coordinates.distance).toBeUndefined()
  })

  it('rejects a distance key that names a store dimension, spatial or not', async () => {
    const geometry = line([-157.5, 67.5], [-67.5, 67.5])

    const nonSpatial = await makeContext(undefined, {
      extra: 'distance',
      lat: 'lat',
      lon: 'lon',
    })
    await expect(queryData(nonSpatial, geometry)).rejects.toThrow(/distanceKey/)

    const spatial = await makeContext(undefined, {
      extra: 'time',
      lat: 'lat',
      lon: 'distance',
    })
    await expect(queryData(spatial, geometry, { time: 10 })).rejects.toThrow(
      /distanceKey/
    )
    // Rejected before any result is built, including the no-level empty one.
    await expect(
      queryData({ ...spatial, level: null }, geometry, { time: 10 })
    ).rejects.toThrow(/distanceKey/)

    const context = await makeContext()
    await expect(
      queryData(context, geometry, { time: 10 }, { distanceKey: 'time' })
    ).rejects.toThrow(/distanceKey/)
  })

  it('queries a store that has a distance dimension once given another key', async () => {
    const context = await makeContext(undefined, {
      extra: 'time',
      lat: 'lat',
      lon: 'distance',
    })
    const result = await queryData(
      context,
      line([-157.5, 67.5], [-67.5, 67.5]),
      { time: 10 },
      { distanceKey: 'along' }
    )
    expect(result.temp).toEqual([0, 1, 2])
    expect(result.coordinates.distance).toEqual([-157.5, -112.5, -67.5])
    expect(result.coordinates.along).toHaveLength(3)
  })

  it('leaves point and polygon queries alone on a store with a distance dimension', async () => {
    const context = await makeContext(undefined, {
      extra: 'time',
      lat: 'lat',
      lon: 'distance',
    })
    const result = await queryData(
      context,
      { type: 'Point', coordinates: [-157.5, 67.5] },
      { time: 10 }
    )
    expect(result.temp).toEqual([0])
  })

  it('rejects coordinates that cannot be traced instead of hanging', async () => {
    const context = await makeContext()
    for (const bad of [NaN, Infinity, 1e20]) {
      await expect(
        queryData(context, line([0, 0], [bad, 0]), { time: 10 })
      ).rejects.toThrow(RangeError)
    }
  })

  it('samples a cell once when the line only touches the antimeridian', async () => {
    const context = await makeContext()
    const result = await queryData(
      context,
      line([190, 67.5], [180, 67.5], [190, 67.5]),
      { time: 10 }
    )
    expect(result.temp).toEqual([0])
  })

  it('follows a raster extent that reaches past the antimeridian', async () => {
    // Cell centers at -180..135 put the edges at -202.5..157.5, so the first
    // column also covers 157.5..180 on the far side.
    const base = await makeContext()
    const xyLimits = { ...WORLD, xMin: -202.5, xMax: 157.5 }
    const context: QueryContext = {
      ...base,
      xyLimits,
      projection: createProjectionContext({
        crs: 'EPSG:4326',
        proj4def: null,
        xyLimits,
      }),
    }
    const result = await queryData(context, line([150, 67.5], [170, 67.5]), {
      time: 10,
    })
    expect(result.temp).toEqual([7, 0])
  })

  it('rejects selector keys that are not store dimensions', async () => {
    const context = await makeContext()
    await expect(
      queryData(context, line([-157.5, 67.5], [-67.5, 67.5]), {
        time: 10,
        junk: [1],
      })
    ).rejects.toThrow(/'junk'.*\[time\]/)
  })

  it('selects a dimension by its own name, with no time alias', async () => {
    const context = await makeContext(undefined, {
      extra: 'forecast_time',
      lat: 'lat',
      lon: 'lon',
    })
    const geometry = line([-157.5, 67.5], [-67.5, 67.5])

    const named = await queryData(context, geometry, { forecast_time: [20] })
    expect(named.temp).toEqual({ 20: [32, 33, 34] })
    expect(named.coordinates.forecast_time).toEqual([20])

    await expect(queryData(context, geometry, { time: [20] })).rejects.toThrow(
      /'time'.*\[forecast_time\]/
    )
  })

  it('collapses repeated selector values into one series', async () => {
    const context = await makeContext()
    const result = await queryData(
      context,
      line([-157.5, 67.5], [-67.5, 67.5]),
      { time: [10, 10] }
    )
    expect(result.temp).toEqual({ 10: [0, 1, 2] })
    expect(result.coordinates.time).toEqual([10])
    expect(result.coordinates.distance).toHaveLength(3)
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
