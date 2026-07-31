import { describe, it, expect } from 'vitest'
import { ZarrStore } from './zarr-store'
import {
  buildMemoryZarrStore,
  ramp,
  type MemoryStore,
} from './__fixtures__/memory-zarr'

/**
 * Integration test for the metadata → description pipeline. A hand-authored,
 * in-memory Zarr v3 store (no network/fs/GL) is driven through the real
 * `ZarrStore._initialize()` path, asserting that `describe()` reflects the
 * fixture's structure and that bounds / orientation are correctly DETECTED
 * from the coordinate arrays (not supplied explicitly).
 */

// 4-lat × 8-lon grid; coordinates are pixel centers on a 45° grid, lat ordered
// north-first (descending). Half-pixel expansion + global snap should yield an
// exactly-global extent, and the descending lat should be detected.
const LAT_CENTERS = [67.5, 22.5, -22.5, -67.5]
const LON_CENTERS = [-157.5, -112.5, -67.5, -22.5, 22.5, 67.5, 112.5, 157.5]

function makeStore(): MemoryStore {
  return buildMemoryZarrStore({
    arrays: [
      {
        name: 'temperature',
        shape: [2, 4, 8],
        chunkShape: [2, 4, 8],
        dtype: 'float32',
        fillValue: -9999,
        dimensionNames: ['time', 'lat', 'lon'],
        attributes: { scale_factor: 0.1, add_offset: 5 },
        chunks: { '0/0/0': ramp(2 * 4 * 8) },
      },
      {
        name: 'lat',
        shape: [4],
        chunkShape: [4],
        dimensionNames: ['lat'],
        chunks: { '0': LAT_CENTERS },
      },
      {
        name: 'lon',
        shape: [8],
        chunkShape: [8],
        dimensionNames: ['lon'],
        chunks: { '0': LON_CENTERS },
      },
    ],
  })
}

describe('ZarrStore (in-memory v3 fixture)', () => {
  it('reads array metadata into the store description', async () => {
    const store = new ZarrStore({
      customStore: makeStore(),
      variable: 'temperature',
      version: 3,
    })
    await store.initialized
    const d = store.describe()

    expect(d.dimensions).toEqual(['time', 'lat', 'lon'])
    expect(d.shape).toEqual([2, 4, 8])
    expect(d.chunks).toEqual([2, 4, 8])
    expect(d.dtype).toBe('float32')
    expect(d.fill_value).toBe(-9999)
    expect(d.scaleFactor).toBe(0.1)
    expect(d.addOffset).toBe(5)
    expect(d.multiscaleType).toBe('none')
  })

  it('identifies spatial dimension indices', async () => {
    const store = new ZarrStore({
      customStore: makeStore(),
      variable: 'temperature',
      version: 3,
    })
    await store.initialized
    const { dimIndices } = store.describe()

    expect(dimIndices.lat?.index).toBe(1)
    expect(dimIndices.lon?.index).toBe(2)
    expect(dimIndices.time?.index).toBe(0)
  })

  it('detects bounds and orientation from coordinate arrays', async () => {
    const store = new ZarrStore({
      customStore: makeStore(),
      variable: 'temperature',
      version: 3,
    })
    await store.initialized
    const d = store.describe()

    // Half-pixel expansion (45° spacing → ±22.5°) + global snap → exactly global.
    expect(d.xyLimits).toEqual({ xMin: -180, xMax: 180, yMin: -90, yMax: 90 })
    // lat is north-first (67.5 → -67.5), so row 0 is north.
    expect(d.latIsAscending).toBe(false)
    // Degree-range bounds → EPSG:4326 inferred.
    expect(d.crs).toBe('EPSG:4326')
  })

  it('opens the variable array and decodes a chunk', async () => {
    const store = new ZarrStore({
      customStore: makeStore(),
      variable: 'temperature',
      version: 3,
    })
    await store.initialized

    const array = await store.getArray()
    const chunk = await array.getChunk([0, 0, 0])
    expect(Array.from(chunk.data as ArrayLike<number>)).toEqual(ramp(2 * 4 * 8))
  })

  it('infers EPSG:3857 from metric coordinates', async () => {
    const metric = buildMemoryZarrStore({
      arrays: [
        {
          name: 'temperature',
          shape: [4, 8],
          chunkShape: [4, 8],
          dimensionNames: ['lat', 'lon'],
        },
        {
          name: 'lat',
          shape: [4],
          chunkShape: [4],
          dimensionNames: ['lat'],
          chunks: { '0': [6e6, 2e6, -2e6, -6e6] },
        },
        {
          name: 'lon',
          shape: [8],
          chunkShape: [8],
          dimensionNames: ['lon'],
          chunks: { '0': [-1.4e7, -1e7, -6e6, -2e6, 2e6, 6e6, 1e7, 1.4e7] },
        },
      ],
    })
    const store = new ZarrStore({
      customStore: metric,
      variable: 'temperature',
      version: 3,
    })
    await store.initialized
    expect(store.describe().crs).toBe('EPSG:3857')
  })
})

/**
 * The multiscale classifier (`_getPyramidMetadata`/`_parseUntiledMultiscale`)
 * sniffs three metadata layouts and picks the render mode + CRS. A wrong guess
 * fails silently (wrong mode/CRS, never throws), so these pin each layout's
 * classification. Untiled cases pass explicit bounds + orientation so the path
 * skips coordinate-array reads and isolates the classification logic.
 */

/** Build a pyramid store: root `multiscales` attr + one variable array per level. */
function pyramidStore(multiscales: unknown, levels: string[]): MemoryStore {
  return buildMemoryZarrStore({
    attributes: { multiscales },
    arrays: levels.map((lvl) => ({
      name: `${lvl}/temperature`,
      shape: [256, 256],
      chunkShape: [256, 256],
      dimensionNames: ['lat', 'lon'],
    })),
  })
}

describe('ZarrStore multiscale classification', () => {
  it('classifies an OME-NGFF pyramid with pixels_per_tile as tiled', async () => {
    const store = new ZarrStore({
      customStore: pyramidStore(
        [
          {
            datasets: [
              { path: '0', pixels_per_tile: 256, crs: 'EPSG:4326' },
              { path: '1', pixels_per_tile: 256, crs: 'EPSG:4326' },
            ],
          },
        ],
        ['0', '1']
      ),
      variable: 'temperature',
      version: 3,
    })
    await store.initialized
    const d = store.describe()

    expect(d.multiscaleType).toBe('tiled')
    expect(d.levels).toEqual(['0', '1'])
    expect(d.crs).toBe('EPSG:4326')
    // Tiled pyramids default to a global extent, north-first.
    expect(d.xyLimits).toEqual({ xMin: -180, xMax: 180, yMin: -90, yMax: 90 })
    expect(d.latIsAscending).toBe(false)
  })

  it('classifies an OME-NGFF pyramid without pixels_per_tile as untiled', async () => {
    const store = new ZarrStore({
      customStore: pyramidStore(
        [{ datasets: [{ path: '0' }, { path: '1' }] }],
        ['0', '1']
      ),
      variable: 'temperature',
      version: 3,
      bounds: [-180, -90, 180, 90],
      latIsAscending: false,
    })
    await store.initialized
    const d = store.describe()

    expect(d.multiscaleType).toBe('untiled')
    expect(d.levels).toEqual(['0', '1'])
    expect(d.untiledLevels.map((l) => l.asset)).toEqual(['0', '1'])
    // Intentionally NOT pinned here:
    //  - crs: an untiled pyramid with absent CRS currently resolves to EPSG:3857
    //    (the classifier defaults there and bounds inference never demotes to
    //    EPSG:4326). That looks incidental rather than intentional, so we don't
    //    cement it — see the CRS-resolution test for the tiled default.
  })

  it('classifies a zarr-conventions layout pyramid as untiled with transforms', async () => {
    const store = new ZarrStore({
      customStore: pyramidStore(
        {
          layout: [
            { asset: '0', transform: { scale: [1, 1], translation: [0, 0] } },
            { asset: '1', transform: { scale: [2, 2], translation: [0, 0] } },
          ],
          crs: 'EPSG:4326',
        },
        ['0', '1']
      ),
      variable: 'temperature',
      version: 3,
      bounds: [-180, -90, 180, 90],
      latIsAscending: false,
    })
    await store.initialized
    const d = store.describe()

    expect(d.multiscaleType).toBe('untiled')
    expect(d.levels).toEqual(['0', '1'])
    expect(d.crs).toBe('EPSG:4326')
    expect(d.untiledLevels).toEqual([
      { asset: '0', scale: [1, 1], translation: [0, 0] },
      { asset: '1', scale: [2, 2], translation: [0, 0] },
    ])
  })

  it('resolves the CRS of a tiled OME-NGFF pyramid from the dataset crs field', async () => {
    // All datasets here are tiled (pixels_per_tile present). A tiled (slippy-map)
    // pyramid with absent CRS conventionally defaults to Web Mercator, so pinning
    // the tiled default is intentional — unlike the untiled-absent case above.
    const crsFor = async (crs?: string) => {
      const dataset = {
        path: '0',
        pixels_per_tile: 256,
        ...(crs ? { crs } : {}),
      }
      const store = new ZarrStore({
        customStore: pyramidStore([{ datasets: [dataset] }], ['0']),
        variable: 'temperature',
        version: 3,
      })
      await store.initialized
      return store.describe().crs
    }
    expect(await crsFor(undefined)).toBe('EPSG:3857')
    expect(await crsFor('EPSG:4326')).toBe('EPSG:4326')
    expect(await crsFor('EPSG:3857')).toBe('EPSG:3857')
  })
})
