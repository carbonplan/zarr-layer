import { describe, it, expect, vi } from 'vitest'
import {
  queryData,
  mergeQueryResults,
  mergeNestedValues,
  type QueryContext,
} from './data-query'
import type { NestedValues, QueryGeometry, QueryResult } from './types'
import { ZarrStore } from '../zarr-store'
import { createProjectionContext } from '../projection-utils'
import { buildMemoryZarrStore, ramp } from '../__fixtures__/memory-zarr'

/**
 * End-to-end orchestration tests against the in-memory Zarr fixture (real
 * ZarrStore + zarrita reads): pixel selection, selector resolution,
 * multichannel packing, the antimeridian two-strip merge, and abort
 * propagation. Plus the pure merge helpers used by the two-strip path.
 *
 * Fixture: 2-time x 4-lat x 8-lon index ramp (value = t*32 + y*8 + x) on a
 * global 45-degree grid, lat north-first. Pixel (x, y) has center
 * lon = (x + 0.5) / 8 * 360 - 180 and lat = 90 - (y + 0.5) / 4 * 180.
 */

const WORLD = { xMin: -180, xMax: 180, yMin: -90, yMax: 90 }

async function makeQueryHarness(opts: { gateReads?: boolean } = {}) {
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

  let releaseReads = () => {}
  const readsReleased = new Promise<void>((res) => {
    releaseReads = res
  })
  const customStore = opts.gateReads
    ? {
        get: async (key: string) => {
          if (key.includes('/temp/c/')) await readsReleased
          return memory.get(key)
        },
      }
    : memory

  const store = new ZarrStore({
    customStore,
    variable: 'temp',
    version: 3,
    bounds: [-180, -90, 180, 90],
    latIsAscending: false,
  })
  await store.initialized
  const zarrArray = await store.getArray()

  const context: QueryContext = {
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
  }

  return { context, releaseReads }
}

const point = (lon: number, lat: number): QueryGeometry => ({
  type: 'Point',
  coordinates: [lon, lat],
})

describe('queryData', () => {
  it('returns an empty result without a committed level', async () => {
    const { context } = await makeQueryHarness()
    const result = await queryData(
      { ...context, level: null },
      point(-157.5, 67.5)
    )
    expect(result.temp).toEqual([])
    expect(result.coordinates).toEqual({ lat: [], lon: [] })
  })

  it('resolves point queries to the correct pixel', async () => {
    const { context } = await makeQueryHarness()
    // Pixel (0, 0), time plane 0.
    const northWest = await queryData(context, point(-157.5, 67.5), {
      time: 10,
    })
    expect(northWest.temp).toEqual([0])
    // Pixel (7, 3): 3*8 + 7.
    const southEast = await queryData(context, point(157.5, -67.5), {
      time: 10,
    })
    expect(southEast.temp).toEqual([31])
  })

  it('normalizes a raw selector and resolves it against coordinate arrays', async () => {
    const { context } = await makeQueryHarness()
    // time=20 is coordinate VALUE 20 (index 1): plane offset 32.
    const result = await queryData(context, point(-157.5, 67.5), { time: 20 })
    expect(result.temp).toEqual([32])
  })

  it('packs multi-value selectors into label-keyed channels', async () => {
    const { context } = await makeQueryHarness()
    const result = await queryData(context, point(-157.5, 67.5), {
      time: [10, 20],
    })
    expect(result.temp).toEqual({ 10: [0], 20: [32] })
  })

  it('merges the two strips of an antimeridian-crossing polygon', async () => {
    const { context } = await makeQueryHarness()
    // Unwrapped lon 130..230 (crossing is signalled by out-of-range lons):
    // covers exactly pixel (7, 0) west of the antimeridian and pixel (0, 0)
    // east of it.
    const result = await queryData(
      context,
      {
        type: 'Polygon',
        coordinates: [
          [
            [130, 50],
            [230, 50],
            [230, 80],
            [130, 80],
            [130, 50],
          ],
        ],
      },
      { time: 10 }
    )
    expect(result.temp).toEqual([7, 0])
    expect(result.coordinates.lon).toEqual([157.5, -157.5])
    expect(result.coordinates.lat).toEqual([67.5, 67.5])
  })

  it('warns once and single-fetches antimeridian crossings on proj4 data', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    try {
      const { context } = await makeQueryHarness()
      const proj4Context: QueryContext = {
        ...context,
        projection: createProjectionContext({
          crs: 'EPSG:4326',
          proj4def: '+proj=longlat +datum=WGS84 +no_defs',
          xyLimits: WORLD,
        }),
      }
      const crossing: QueryGeometry = {
        type: 'Polygon',
        coordinates: [
          [
            [130, 50],
            [230, 50],
            [230, 80],
            [130, 80],
            [130, 50],
          ],
        ],
      }
      await queryData(proj4Context, crossing)
      await queryData(proj4Context, crossing)
      expect(warnSpy).toHaveBeenCalledTimes(1)
      expect(proj4Context.antimeridianWarnings.has('proj4-crossing')).toBe(true)
    } finally {
      warnSpy.mockRestore()
    }
  })

  it('propagates aborts as AbortError', async () => {
    const { context, releaseReads } = await makeQueryHarness({
      gateReads: true,
    })
    const controller = new AbortController()
    const pending = queryData(context, point(-157.5, 67.5), undefined, {
      signal: controller.signal,
    })
    controller.abort()
    releaseReads()
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' })
  })
})

/**
 * Merge semantics for antimeridian two-strip queries: west-strip pixels then
 * east-strip pixels, spatial coordinate arrays concatenated, non-spatial
 * coordinates taken from the first strip.
 */

describe('mergeQueryResults', () => {
  it('concatenates values and spatial coordinates', () => {
    const west: QueryResult = {
      temp: [1, 2],
      dimensions: ['lat', 'lon'],
      coordinates: { lat: [10, 10], lon: [179, 179.5], month: [1] },
    }
    const east: QueryResult = {
      temp: [3],
      dimensions: ['lat', 'lon'],
      coordinates: { lat: [10], lon: [-179.5], month: [1] },
    }
    const merged = mergeQueryResults(west, east, 'temp', 'lat', 'lon')

    expect(merged.temp).toEqual([1, 2, 3])
    expect(merged.coordinates.lat).toEqual([10, 10, 10])
    expect(merged.coordinates.lon).toEqual([179, 179.5, -179.5])
    // Non-spatial coordinates come from the first strip unchanged.
    expect(merged.coordinates.month).toEqual([1])
    expect(merged.dimensions).toEqual(['lat', 'lon'])
  })

  it('merges nested multi-band values recursively', () => {
    const west: QueryResult = {
      temp: { jan: [1] },
      dimensions: ['lat', 'lon'],
      coordinates: { lat: [0], lon: [179] },
    }
    const east: QueryResult = {
      temp: { jan: [2], feb: [3] },
      dimensions: ['lat', 'lon'],
      coordinates: { lat: [0], lon: [-179] },
    }
    const merged = mergeQueryResults(west, east, 'temp', 'lat', 'lon')
    expect(merged.temp).toEqual({ jan: [1, 2], feb: [3] })
  })

  it('takes the first side when value shapes disagree', () => {
    const west: QueryResult = {
      temp: [1],
      dimensions: ['lat', 'lon'],
      coordinates: { lat: [0], lon: [179] },
    }
    const east: QueryResult = {
      temp: { jan: [2] },
      dimensions: ['lat', 'lon'],
      coordinates: { lat: [0], lon: [-179] },
    }
    const merged = mergeQueryResults(west, east, 'temp', 'lat', 'lon')
    expect(merged.temp).toEqual([1])
  })
})

describe('mergeNestedValues', () => {
  it('concatenates leaf arrays at any depth', () => {
    const a: NestedValues = { r: { low: [1] }, g: { low: [2] } }
    const b: NestedValues = { r: { low: [3], high: [4] } }
    expect(mergeNestedValues(a, b)).toEqual({
      r: { low: [1, 3], high: [4] },
      g: { low: [2] },
    })
  })

  it('includes keys present only in the second object', () => {
    expect(mergeNestedValues({ a: [1] }, { b: [2] })).toEqual({
      a: [1],
      b: [2],
    })
  })
})
