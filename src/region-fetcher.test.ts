import { describe, it, expect, vi } from 'vitest'
import * as zarr from 'zarrita'
import {
  RegionFetcher,
  contiguousBandRange,
  isCOrder,
  splitBands,
  type RegionFetcherContext,
} from './region-fetcher'
import { RegionCache, makeRegionKey } from './region-cache'
import { createProjectionContext } from './projection-utils'
import {
  createRequestCanceller,
  cancelAllRequests,
  dropQueuedFetches,
} from './region-utils'
import { ZarrStore } from './zarr-store'
import {
  buildMemoryZarrStore,
  type ArraySpec,
} from './__fixtures__/memory-zarr'
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

// Band b of the multi-band fixture is the ramp plus 100 * b.
function bandChunkData(bands: number, chunkY: number, chunkX: number) {
  const out: number[] = []
  for (let b = 0; b < bands; b++) {
    out.push(...chunkData(chunkY, chunkX).map((v) => v + 100 * b))
  }
  return out
}

function makeMultiBandStore(
  bands: number,
  dtype?: ArraySpec['dtype'],
  order?: ArraySpec['order'],
  attributes: Record<string, unknown> = {},
  fillValue?: number
) {
  return buildMemoryZarrStore({
    arrays: [
      {
        name: 'temperature',
        shape: [bands, HEIGHT, WIDTH],
        attributes,
        fillValue,
        chunkShape: [bands, ...REGION],
        dimensionNames: ['band', 'lat', 'lon'],
        dtype,
        order,
        chunks: {
          '0/0/0': bandChunkData(bands, 0, 0),
          '0/0/1': bandChunkData(bands, 0, 1),
          '0/1/0': bandChunkData(bands, 1, 0),
          '0/1/1': bandChunkData(bands, 1, 1),
        },
      },
    ],
  })
}

function makeMemoryStore(
  attributes: Record<string, unknown> = {},
  dtype?: ArraySpec['dtype'],
  fillValue?: number
) {
  return buildMemoryZarrStore({
    arrays: [
      {
        name: 'temperature',
        shape: [HEIGHT, WIDTH],
        chunkShape: REGION,
        dimensionNames: ['lat', 'lon'],
        attributes,
        dtype,
        fillValue,
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
    dtype?: ArraySpec['dtype']
    fillValue?: number
    bandTextures?: boolean
    /** Band count of a (band, lat, lon) array, with these bands selected. */
    bands?: {
      count: number
      selected: number[]
      order?: ArraySpec['order']
    }
  } = {}
) {
  const memory = opts.bands
    ? makeMultiBandStore(
        opts.bands.count,
        opts.dtype,
        opts.bands.order,
        opts.attributes,
        opts.fillValue
      )
    : makeMemoryStore(opts.attributes, opts.dtype, opts.fillValue)
  let releaseReads = () => {}
  const readsReleased = new Promise<void>((res) => {
    releaseReads = res
  })
  // Chunk reads are counted so tests can assert a batch never dispatched, and
  // optionally held open so tests can abort/invalidate mid-fetch.
  const chunkReads: string[] = []
  const customStore = {
    get: async (key: string) => {
      if (key.includes('/c/')) {
        chunkReads.push(key)
        if (opts.gateReads) await readsReleased
      }
      return memory.get(key)
    },
  }

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
    baseSliceArgs: opts.bands ? [0, 0, 0] : [0, 0],
    baseMultiValueDims: opts.bands
      ? [
          {
            dimIndex: 0,
            dimName: 'band',
            values: opts.bands.selected,
            labels: opts.bands.selected,
          },
        ]
      : [],
  }
  let selectorVersion = 0

  const cache = new RegionCache()
  const requestCanceller = createRequestCanceller()
  const invalidate = vi.fn()
  const createRegionGeometry = vi.fn()

  // fetchRegions runs synchronously up to its pre-flight staleness check, so
  // `show()` is the one hook a test can use to simulate a level or selector
  // change landing in that window.
  let onBatchStart = () => {}

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
    loadingDebouncer: {
      show: () => onBatchStart(),
      hide: () => {},
    },
    getActiveLevel: () => level,
    getSelectorVersion: () => selectorVersion,
    getBandNames: () =>
      opts.bands
        ? opts.bands.selected.map((b) => `band_${b}`)
        : ['temperature'],
    usesBandTextures: () => opts.bandTextures ?? false,
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
    chunkReads,
    setLevel: (next: LevelRuntime | null) => {
      level = next
    },
    getLevel: () => level,
    setSelectorVersion: (v: number) => {
      selectorVersion = v
    },
    onBatchStart: (fn: () => void) => {
      onBatchStart = fn
    },
  }
}

/**
 * Wait until a batch has dispatched a chunk read. Fetches start from a queue
 * in a later task, so a test cancelling "mid-fetch" has to let the read
 * begin first, or it only exercises the queue.
 */
async function untilReading(chunkReads: string[]): Promise<void> {
  while (chunkReads.length === 0) {
    await new Promise((resolve) => setTimeout(resolve, 0))
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
    expect(region.bandTexture).toBeNull()
    expect(region.bandTextureKey).toBeNull()
  })

  it('keeps small integer bands raw for band-sampling shaders', async () => {
    const { fetcher, cache } = await makeHarness({
      dtype: 'int8',
      fillValue: -128,
      attributes: { scale_factor: 2, add_offset: 10 },
      bandTextures: true,
    })
    await fetcher.fetchRegions([{ regionX: 1, regionY: 0 }])

    const region = cache.get(makeRegionKey(0, 1, 0))!
    const band = region.bandData.get('temperature')!
    // Stored as the dtype, unscaled: the shader applies the transform.
    expect(band).toBeInstanceOf(Int8Array)
    expect(Array.from(band)).toEqual(chunkData(0, 1))
    expect(region.bandTransform).toEqual({ scale: 2, offset: 10, fill: -128 })
    expect(region.data).toBeNull()
  })

  it('keeps integer bands raw for a scale float32 cannot hold exactly', async () => {
    // 0.1 is inexact in float32, like most real scale factors; the region
    // must stay in the integer format the renderer draws this level with.
    const { fetcher, cache } = await makeHarness({
      dtype: 'int16',
      attributes: { scale_factor: 0.1 },
      bandTextures: true,
    })
    await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])

    const region = cache.get(makeRegionKey(0, 0, 0))!
    expect(region.bandData.get('temperature')).toBeInstanceOf(Int16Array)
    expect(region.bandTransform).toEqual({ scale: 0.1, offset: 0, fill: null })
  })

  it('converts integer bands to float for the main texture', async () => {
    const { fetcher, cache } = await makeHarness({ dtype: 'int8' })
    await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])

    const region = cache.get(makeRegionKey(0, 0, 0))!
    expect(region.bandData.get('temperature')).toBeInstanceOf(Float32Array)
    expect(region.bandTransform).toBeNull()
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

  it('regenerates geometry when a refetch changes the region size', async () => {
    const { fetcher, cache, createRegionGeometry, getLevel, setLevel } =
      await makeHarness()
    await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
    const region = cache.get(makeRegionKey(0, 0, 0))!
    expect(region.width).toBe(4)
    expect(createRegionGeometry).toHaveBeenCalledTimes(1)

    // Same level index and cache key, larger regions: the cached region's mesh
    // no longer matches its data, so it has to be rebuilt (and re-uploaded).
    setLevel({ ...getLevel()!, regionSize: [HEIGHT, WIDTH] })
    await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])

    expect(region.width).toBe(WIDTH)
    expect(region.height).toBe(HEIGHT)
    expect(createRegionGeometry).toHaveBeenCalledTimes(2)
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
    const { fetcher, cache, chunkReads, getLevel, setLevel, onBatchStart } =
      await makeHarness()
    const original = getLevel()!
    onBatchStart(() => setLevel({ ...original, index: 1 }))

    await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
    // No read is issued at all: the pre-flight check drops the batch before
    // dispatch rather than letting each fetch discard its own result.
    expect(chunkReads).toEqual([])
    const region = cache.get(makeRegionKey(0, 0, 0))!
    expect(region.data).toBeNull()
    expect(region.loading).toBe(false)
  })

  it('cancels the whole batch when the selector changed before dispatch', async () => {
    const { fetcher, cache, chunkReads, setSelectorVersion, onBatchStart } =
      await makeHarness()
    onBatchStart(() => setSelectorVersion(1))

    await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
    expect(chunkReads).toEqual([])
    const region = cache.get(makeRegionKey(0, 0, 0))!
    expect(region.data).toBeNull()
    expect(region.loading).toBe(false)
  })

  it('drops data when the level changes mid-fetch', async () => {
    const harness = await makeHarness({ gateReads: true })
    const { fetcher, cache, setLevel, getLevel, releaseReads, chunkReads } =
      harness
    const original = getLevel()!

    const batch = fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
    await untilReading(chunkReads)
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
      const {
        fetcher,
        cache,
        requestCanceller,
        invalidate,
        releaseReads,
        chunkReads,
      } = harness

      const batch = fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
      await untilReading(chunkReads)
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

  it('an aborted fetch does not clobber a newer request that took over the region', async () => {
    const harness = await makeHarness({ gateReads: true })
    const { fetcher, cache, requestCanceller, releaseReads, chunkReads } =
      harness

    const batch = fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
    await untilReading(chunkReads)
    const region = cache.get(makeRegionKey(0, 0, 0))!
    cancelAllRequests(requestCanceller)
    // A newer request takes over the region while the aborted one unwinds.
    region.requestId = 999
    region.loading = true
    releaseReads()
    await batch

    expect(region.requestId).toBe(999)
    expect(region.loading).toBe(true)
  })

  it('reads nothing for a region whose level changed while it was queued', async () => {
    const { fetcher, cache, setLevel, getLevel, chunkReads } =
      await makeHarness()
    const original = getLevel()!

    const batch = fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
    setLevel({ ...original, index: 1 })
    await batch

    expect(chunkReads).toEqual([])
    expect(cache.get(makeRegionKey(0, 0, 0))!.loading).toBe(false)
  })

  it('fetches a region dropped from the queue when it is requested again', async () => {
    const { fetcher, cache, requestCanceller, chunkReads } = await makeHarness()

    const dropped = fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
    dropQueuedFetches(requestCanceller, () => true)
    await dropped
    const region = cache.get(makeRegionKey(0, 0, 0))!
    expect(chunkReads).toEqual([])
    expect(region.loading).toBe(false)

    await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
    expect(Array.from(region.data!)).toEqual(chunkData(0, 0))
    expect(region.loading).toBe(false)
  })

  it('does nothing without a committed level', async () => {
    const harness = await makeHarness()
    const { fetcher, cache, setLevel } = harness
    setLevel(null)
    await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
    expect(cache.size).toBe(0)
  })

  // 2-time x 4-lat x 8-lon ramp: value = t*32 + y*8 + x. Selecting both time
  // steps produces two channels.
  async function makeMultibandHarness(usesBandTextures: boolean) {
    const memory = buildMemoryZarrStore({
      arrays: [
        {
          name: 'temperature',
          shape: [2, HEIGHT, WIDTH],
          chunkShape: [2, HEIGHT, WIDTH],
          dimensionNames: ['time', 'lat', 'lon'],
          chunks: { '0/0/0': Array.from({ length: 64 }, (_, i) => i) },
        },
      ],
    })
    const store = new ZarrStore({
      customStore: memory,
      variable: 'temperature',
      version: 3,
      bounds: [-180, -90, 180, 90],
      latIsAscending: false,
    })
    await store.initialized
    const zarrArray = await store.getArray()

    const level: LevelRuntime = {
      index: 0,
      zarrArray,
      width: WIDTH,
      height: HEIGHT,
      regionSize: REGION,
      baseSliceArgs: [0, 0, 0],
      baseMultiValueDims: [
        { dimIndex: 0, dimName: 'time', values: [0, 1], labels: [10, 20] },
      ],
    }
    const cache = new RegionCache()
    const fetcher = new RegionFetcher({
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
      requestCanceller: createRequestCanceller(),
      loadingDebouncer: { show: () => {}, hide: () => {} },
      getActiveLevel: () => level,
      getSelectorVersion: () => 0,
      // One name provided: the second channel falls back to band_<index>.
      getBandNames: () => ['t10'],
      usesBandTextures: () => usesBandTextures,
      isRemoved: () => false,
      getRegionBounds: () => ({ xMin: 0, xMax: 1, yMin: 0, yMax: 1 }),
      computeRegionMercatorBounds: () => ({ x0: 0, y0: 0, x1: 1, y1: 1 }),
      createRegionGeometry: vi.fn(),
      invalidate: vi.fn(),
    })
    return { fetcher, cache }
  }

  it('fetches multi-value dims as parallel channels and interleaves them', async () => {
    const { fetcher, cache } = await makeMultibandHarness(false)
    await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
    const region = cache.get(makeRegionKey(0, 0, 0))!

    expect(region.channels).toBe(2)
    const t0 = chunkData(0, 0)
    expect(Array.from(region.bandData.get('t10')!)).toEqual(t0)
    expect(Array.from(region.bandData.get('band_1')!)).toEqual(
      t0.map((v) => v + 32)
    )
    // Interleaved pixel-major: [c0[0], c1[0], c0[1], c1[1], ...].
    const interleaved = Array.from(region.data!)
    expect(interleaved).toHaveLength(t0.length * 2)
    expect(interleaved.slice(0, 4)).toEqual([
      t0[0],
      t0[0] + 32,
      t0[1],
      t0[1] + 32,
    ])
  })

  it('skips the interleaved copy when the shader samples band textures', async () => {
    const { fetcher, cache } = await makeMultibandHarness(true)
    await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
    const region = cache.get(makeRegionKey(0, 0, 0))!

    // Every band is still fetched and kept; only the interleaved copy that
    // would feed the unread main texture is skipped.
    const t0 = chunkData(0, 0)
    expect(Array.from(region.bandData.get('t10')!)).toEqual(t0)
    expect(Array.from(region.bandData.get('band_1')!)).toEqual(
      t0.map((v) => v + 32)
    )
    // Left null rather than aliased to one band: a stand-in here would be
    // drawn as the whole dataset if the shader switched back to the main
    // texture before the refetch landed.
    expect(region.data).toBeNull()
    expect(region.channels).toBe(2)
  })

  describe('multi-band regions', () => {
    const expectBands = (
      region: { bandData: Map<string, ArrayLike<number>> },
      selected: number[]
    ) => {
      for (const b of selected) {
        expect(Array.from(region.bandData.get(`band_${b}`)!)).toEqual(
          chunkData(0, 0).map((v) => v + 100 * b)
        )
      }
    }

    it('reads a contiguous band range in one read', async () => {
      const selected = [1, 2, 3]
      const { fetcher, cache } = await makeHarness({
        bands: { count: 4, selected },
        dtype: 'int16',
        bandTextures: true,
      })
      await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
      const region = cache.get(makeRegionKey(0, 0, 0))!
      expectBands(region, selected)
      // Every band is a view into the one result.
      const buffers = new Set(
        selected.map(
          (b) => (region.bandData.get(`band_${b}`) as Int16Array).buffer
        )
      )
      expect(buffers.size).toBe(1)
    })

    it("keeps each band in the store's memory order when it is Fortran-order", async () => {
      // Every band holds its own values, in the order a separate read of
      // that band lays them out: latitude varies fastest here.
      const selected = [0, 1, 2]
      const { fetcher, cache } = await makeHarness({
        bands: { count: 3, selected, order: 'F' },
        dtype: 'int16',
        bandTextures: true,
      })
      await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
      const region = cache.get(makeRegionKey(0, 0, 0))!
      for (const b of selected) {
        const rowMajor = chunkData(0, 0).map((v) => v + 100 * b)
        const expected: number[] = []
        for (let x = 0; x < 4; x++) {
          for (let y = 0; y < 2; y++) expected.push(rowMajor[y * 4 + x])
        }
        expect(Array.from(region.bandData.get(`band_${b}`)!)).toEqual(expected)
      }
    })

    it('fails a region whose selection runs past the last band', async () => {
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      try {
        const selected = [3, 4]
        const { fetcher, cache } = await makeHarness({
          bands: { count: 4, selected },
          dtype: 'int16',
          bandTextures: true,
        })
        await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
        // Band by band, index 4 fails the read: nothing partial is kept.
        const region = cache.get(makeRegionKey(0, 0, 0))!
        expect(region.bandData.size).toBe(0)
        expect(region.loading).toBe(false)
      } finally {
        errorSpy.mockRestore()
      }
    })

    it('scales float bands read in one get', async () => {
      const selected = [0, 1]
      const { fetcher, cache } = await makeHarness({
        bands: { count: 2, selected },
        attributes: { scale_factor: 2, add_offset: 1 },
        fillValue: 0,
        bandTextures: true,
      })
      await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
      const region = cache.get(makeRegionKey(0, 0, 0))!
      for (const b of selected) {
        // The ramp's first pixel of band 0 is the fill value, so it is NaN.
        const expected = chunkData(0, 0).map((v) => {
          const raw = v + 100 * b
          return raw === 0 ? NaN : raw * 2 + 1
        })
        expect(Array.from(region.bandData.get(`band_${b}`)!)).toEqual(expected)
      }
    })

    it('reads a bool band selection band by band', async () => {
      const selected = [0, 1]
      const { fetcher, cache } = await makeHarness({
        bands: { count: 2, selected },
        dtype: 'bool',
        bandTextures: true,
      })
      await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
      const region = cache.get(makeRegionKey(0, 0, 0))!
      // Band 0 of the fixture is the ramp, true everywhere but its first
      // pixel; band 1 adds 100, so it is all true.
      const truthy = (b: number) =>
        chunkData(0, 0).map((v) => (v + 100 * b ? 1 : 0))
      expect(Array.from(region.bandData.get('band_0')!)).toEqual(truthy(0))
      expect(Array.from(region.bandData.get('band_1')!)).toEqual(truthy(1))
    })

    it('reads a run of negative indices band by band', async () => {
      const selected = [-2, -1]
      const { fetcher, cache } = await makeHarness({
        bands: { count: 4, selected },
        dtype: 'int16',
        bandTextures: true,
      })
      await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
      const region = cache.get(makeRegionKey(0, 0, 0))!
      expect(Array.from(region.bandData.get('band_-2')!)).toEqual(
        chunkData(0, 0).map((v) => v + 200)
      )
      expect(Array.from(region.bandData.get('band_-1')!)).toEqual(
        chunkData(0, 0).map((v) => v + 300)
      )
    })

    it('reads each band of a scattered selection', async () => {
      const selected = [3, 0]
      const { fetcher, cache } = await makeHarness({
        bands: { count: 4, selected },
        dtype: 'int16',
        bandTextures: true,
      })
      await fetcher.fetchRegions([{ regionX: 0, regionY: 0 }])
      expectBands(cache.get(makeRegionKey(0, 0, 0))!, selected)
    })
  })
})

describe('contiguousBandRange', () => {
  // time, band, lat, lon
  const SHAPE = [1, 10, 4, 4]
  const dims = (values: number[], dimIndex = 1) => [
    { dimIndex, dimName: 'band', values, labels: values },
  ]

  it('accepts an ascending run after integer-indexed dimensions', () => {
    expect(contiguousBandRange(dims([4, 5, 6]), [0, 0, 0, 0], SHAPE)).toEqual({
      dimIndex: 1,
      start: 4,
    })
  })

  it('rejects a run starting at a negative index', () => {
    expect(contiguousBandRange(dims([-2, -1]), [0, 0, 0, 0], SHAPE)).toBeNull()
    expect(contiguousBandRange(dims([-1, 0]), [0, 0, 0, 0], SHAPE)).toBeNull()
  })

  it('rejects a run starting at a fractional index', () => {
    expect(
      contiguousBandRange(dims([0.5, 1.5]), [0, 0, 0, 0], SHAPE)
    ).toBeNull()
  })

  it('rejects a run past the end of the dimension', () => {
    expect(contiguousBandRange(dims([8, 9]), [0, 0, 0, 0], SHAPE)).toEqual({
      dimIndex: 1,
      start: 8,
    })
    expect(contiguousBandRange(dims([9, 10]), [0, 0, 0, 0], SHAPE)).toBeNull()
  })

  it('rejects gaps, descending runs and single bands', () => {
    expect(contiguousBandRange(dims([0, 2]), [0, 0, 0, 0], SHAPE)).toBeNull()
    expect(contiguousBandRange(dims([2, 1]), [0, 0, 0, 0], SHAPE)).toBeNull()
    expect(contiguousBandRange(dims([2]), [0, 0, 0, 0], SHAPE)).toBeNull()
  })

  it('rejects a band dimension after another sliced dimension', () => {
    expect(
      contiguousBandRange(dims([0, 1]), [zarr.slice(0, 2), 0, 0, 0], SHAPE)
    ).toBeNull()
  })

  it('rejects more than one multi-value dimension', () => {
    expect(
      contiguousBandRange(
        [...dims([0, 1], 0), ...dims([0, 1], 1)],
        [0, 0, 0, 0],
        SHAPE
      )
    ).toBeNull()
  })
})

describe('splitBands', () => {
  // Two bands of 2x3, band b holding 10 * b + (row-major position).
  const expected = [
    [0, 1, 2, 3, 4, 5],
    [10, 11, 12, 13, 14, 15],
  ]

  it('returns views of a C-order result', () => {
    const data = new Int16Array(expected.flat())
    const bands = splitBands({ data, shape: [2, 2, 3], stride: [6, 3, 1] })
    expect(bands.map((b) => Array.from(b))).toEqual(expected)
    expect((bands[1] as Int16Array).buffer).toBe(data.buffer)
  })

  it('gathers each band of a Fortran-order result in memory order', () => {
    // (band, lon, lat) in Fortran order: longitude varies fastest after the
    // band, so each band comes out as rows of latitude, which is how the
    // renderer draws it.
    const bands = [0, 1].map((c) => [0, 1, 2, 3, 4, 5].map((v) => v + 10 * c))
    const data = new Int16Array(12)
    for (let c = 0; c < 2; c++)
      for (let lon = 0; lon < 3; lon++)
        for (let lat = 0; lat < 2; lat++)
          data[c + 2 * lon + 6 * lat] = bands[c][lat * 3 + lon]
    const split = splitBands({ data, shape: [2, 3, 2], stride: [1, 2, 6] })
    expect(split.map((b) => Array.from(b))).toEqual(bands)
    expect(split[0]).toBeInstanceOf(Int16Array)
  })
})

describe('isCOrder', () => {
  it('accepts row-major strides and ignores size-1 dimensions', () => {
    expect(isCOrder([3, 2, 4], [8, 4, 1])).toBe(true)
    expect(isCOrder([1, 2, 4], [99, 4, 1])).toBe(true)
  })

  it('rejects Fortran-order strides', () => {
    expect(isCOrder([3, 2, 4], [1, 3, 6])).toBe(false)
  })
})
