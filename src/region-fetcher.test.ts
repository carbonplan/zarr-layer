import { describe, it, expect, vi } from 'vitest'
import { RegionFetcher, type RegionFetcherContext } from './region-fetcher'
import { RegionCache, makeRegionKey } from './region-cache'
import { createProjectionContext } from './projection-utils'
import { createRequestCanceller, cancelAllRequests } from './region-utils'
import { ZarrStore } from './zarr-store'
import { buildMemoryZarrStore } from './__fixtures__/memory-zarr'
import type { LevelRuntime } from './region-state'

/**
 * Integration tests for the (GL-free) fetch pipeline against the in-memory
 * Zarr fixture: a real ZarrStore, real zarrita reads, and the real cache.
 * Geometry creation is injected as a spy; GPU upload is out of scope (it
 * happens lazily at render time).
 */

// 4-lat x 8-lon index ramp split into 2x4 chunks -> a 2x2 region grid.
const HEIGHT = 4
const WIDTH = 8
const REGION: [number, number] = [2, 4]

function chunkData(chunkY: number, chunkX: number): number[] {
  const out: number[] = []
  for (let y = chunkY * 2; y < chunkY * 2 + 2; y++) {
    for (let x = chunkX * 4; x < chunkX * 4 + 4; x++) {
      out.push(y * WIDTH + x)
    }
  }
  return out
}

function makeMemoryStore(attributes: Record<string, unknown> = {}) {
  return buildMemoryZarrStore({
    arrays: [
      {
        name: 'temperature',
        shape: [HEIGHT, WIDTH],
        chunkShape: REGION,
        dimensionNames: ['lat', 'lon'],
        attributes,
        chunks: {
          '0/0': chunkData(0, 0),
          '0/1': chunkData(0, 1),
          '1/0': chunkData(1, 0),
          '1/1': chunkData(1, 1),
        },
      },
    ],
  })
}

async function makeHarness(
  opts: {
    attributes?: Record<string, unknown>
    gateReads?: boolean
  } = {}
) {
  const memory = makeMemoryStore(opts.attributes)
  let releaseReads = () => {}
  const readsReleased = new Promise<void>((res) => {
    releaseReads = res
  })
  // Optionally hold chunk reads open so tests can abort/invalidate mid-fetch.
  const customStore = opts.gateReads
    ? {
        get: async (key: string) => {
          if (key.includes('/c/')) await readsReleased
          return memory.get(key)
        },
      }
    : memory

  const store = new ZarrStore({
    customStore,
    variable: 'temperature',
    version: 3,
    bounds: [-180, -90, 180, 90],
    latIsAscending: false,
  })
  await store.initialized
  const zarrArray = await store.getArray()

  let level: LevelRuntime | null = {
    index: 0,
    zarrArray,
    width: WIDTH,
    height: HEIGHT,
    regionSize: REGION,
    baseSliceArgs: [0, 0],
    baseMultiValueDims: [],
  }
  let selectorVersion = 0

  const cache = new RegionCache()
  const requestCanceller = createRequestCanceller()
  const invalidate = vi.fn()
  const createRegionGeometry = vi.fn()

  const context: RegionFetcherContext = {
    zarrStore: store,
    dimIndices: store.describe().dimIndices,
    levels: [],
    projection: createProjectionContext({
      crs: 'EPSG:4326',
      proj4def: null,
      xyLimits: { xMin: -180, xMax: 180, yMin: -90, yMax: 90 },
    }),
    xyLimits: { xMin: -180, xMax: 180, yMin: -90, yMax: 90 },
    latIsAscending: false,
    fixedDataScale: 1,
    regionCache: cache,
    requestCanceller,
    loadingDebouncer: { show: () => {}, hide: () => {} },
    getActiveLevel: () => level,
    getSelectorVersion: () => selectorVersion,
    getBandNames: () => ['temperature'],
    isRemoved: () => false,
    getRegionBounds: () => ({ xMin: 0, xMax: 1, yMin: 0, yMax: 1 }),
    computeRegionMercatorBounds: () => ({ x0: 0, y0: 0, x1: 1, y1: 1 }),
    createRegionGeometry,
    invalidate,
  }

  return {
    fetcher: new RegionFetcher(context),
    context,
    cache,
    requestCanceller,
    invalidate,
    createRegionGeometry,
    releaseReads,
    setLevel: (next: LevelRuntime | null) => {
      level = next
    },
    getLevel: () => level,
    setSelectorVersion: (v: number) => {
      selectorVersion = v
    },
  }
}

describe('RegionFetcher', () => {
  it('fetches region chunks into CPU-side state', async () => {
    const { fetcher, cache, createRegionGeometry, invalidate } =
      await makeHarness()
    await fetcher.fetchRegions([
      { regionX: 0, regionY: 0 },
      { regionX: 1, regionY: 1 },
    ])

    const topLeft = cache.get(makeRegionKey(0, 0, 0))!
    expect(Array.from(topLeft.data!)).toEqual(chunkData(0, 0))
    expect(topLeft.width).toBe(4)
    expect(topLeft.height).toBe(2)
    expect(topLeft.loading).toBe(false)
    expect(topLeft.requestId).toBeNull()
    expect(topLeft.levelMeta).toEqual({
      width: WIDTH,
      height: HEIGHT,
      regionSize: REGION,
    })

    const bottomRight = cache.get(makeRegionKey(0, 1, 1))!
    expect(Array.from(bottomRight.data!)).toEqual(chunkData(1, 1))

    expect(createRegionGeometry).toHaveBeenCalledTimes(2)
    expect(invalidate).toHaveBeenCalled()
  })

  it('leaves GPU upload to render time', async () => {
    const { fetcher, cache } = await makeHarness()
    await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])

    const region = cache.get(makeRegionKey(0, 0, 0))!
    expect(region.texture).toBeNull()
    expect(region.textureUploaded).toBe(false)
    expect(region.bandData.get('temperature')).toBeDefined()
    expect(region.bandTexturesUploaded.size).toBe(0)
  })

  it('applies scale/offset to raw values', async () => {
    const { fetcher, cache } = await makeHarness({
      attributes: { scale_factor: 2, add_offset: 10 },
    })
    await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])

    const region = cache.get(makeRegionKey(0, 0, 0))!
    expect(Array.from(region.data!)).toEqual(
      chunkData(0, 0).map((v) => v * 2 + 10)
    )
  })

  it('never lets an older fetch overwrite newer region data', async () => {
    const { fetcher, cache } = await makeHarness()
    // A region already refreshed by a newer selector version...
    const key = makeRegionKey(0, 0, 0)
    await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
    const region = cache.get(key)!
    region.selectorVersion = 5
    region.data = null

    // ...must ignore a fetch batch stamped with the old version.
    await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
    expect(region.data).toBeNull()
    expect(region.selectorVersion).toBe(5)
  })

  it('cancels the whole batch when the level changed before dispatch', async () => {
    const { fetcher, cache, context, getLevel } = await makeHarness()
    // The getter serves the old level to the snapshot, then the new one to
    // the pre-flight staleness check.
    const original = getLevel()!
    const swapped: LevelRuntime = { ...original, index: 1 }
    let calls = 0
    context.getActiveLevel = () => (calls++ === 0 ? original : swapped)

    await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
    const region = cache.get(makeRegionKey(0, 0, 0))!
    expect(region.data).toBeNull()
    expect(region.loading).toBe(false)
  })

  it('drops data when the level changes mid-fetch', async () => {
    const harness = await makeHarness({ gateReads: true })
    const { fetcher, cache, setLevel, getLevel, releaseReads } = harness
    const original = getLevel()!

    const batch = fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
    setLevel({ ...original, index: 1 })
    releaseReads()
    await batch

    const region = cache.get(makeRegionKey(0, 0, 0))!
    expect(region.data).toBeNull()
    expect(region.loading).toBe(false)
    expect(region.requestId).toBeNull()
  })

  it('aborts silently and re-invalidates so regions refetch on return', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    try {
      const harness = await makeHarness({ gateReads: true })
      const { fetcher, cache, requestCanceller, invalidate, releaseReads } =
        harness

      const batch = fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
      cancelAllRequests(requestCanceller)
      releaseReads()
      await batch

      const region = cache.get(makeRegionKey(0, 0, 0))!
      expect(region.data).toBeNull()
      expect(region.loading).toBe(false)
      expect(region.requestId).toBeNull()
      expect(requestCanceller.controllers.size).toBe(0)
      expect(errorSpy).not.toHaveBeenCalled()
      expect(invalidate).toHaveBeenCalled()
    } finally {
      errorSpy.mockRestore()
    }
  })

  it('does nothing without a committed level', async () => {
    const harness = await makeHarness()
    const { fetcher, cache, setLevel } = harness
    setLevel(null)
    await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
    expect(cache.size).toBe(0)
  })
})
