import { describe, it, expect, vi } from 'vitest'
import { ZarrLayer } from './zarr-layer'
import { buildMemoryZarrStore } from './__fixtures__/memory-zarr'
import { createRecordingGl } from './__fixtures__/fake-gl'
import type { MapLike } from './types'

/**
 * minRegionSize reaches the renderer: a 4x8 array in 2x4 chunks draws as
 * four regions by default and as fewer, larger ones when grown, with the
 * region past the array edge clipped to it.
 */

/** A height x width ramp (value = y * width + x) in 2x4 chunks. */
function store(height = 4, width = 8) {
  const chunks: Record<string, number[]> = {}
  for (let cy = 0; cy < Math.ceil(height / 2); cy++) {
    for (let cx = 0; cx < Math.ceil(width / 4); cx++) {
      const values: number[] = []
      for (let y = cy * 2; y < cy * 2 + 2; y++) {
        for (let x = cx * 4; x < cx * 4 + 4; x++) values.push(y * width + x)
      }
      chunks[`${cy}/${cx}`] = values
    }
  }
  const memory = buildMemoryZarrStore({
    arrays: [
      {
        name: 'temperature',
        shape: [height, width],
        chunkShape: [2, 4],
        dimensionNames: ['lat', 'lon'],
        chunks,
      },
    ],
  })
  return { get: (key: string) => memory.get(key) }
}

const map = {
  getProjection: () => ({ type: 'mercator' }),
  getTerrain: () => null,
  getZoom: () => 3,
  getBounds: () => ({
    getWest: () => -180,
    getEast: () => 180,
    toArray: () => [
      [-180, -85],
      [180, 85],
    ],
  }),
  getRenderWorldCopies: () => true,
  triggerRepaint: vi.fn(),
  on: vi.fn(),
  off: vi.fn(),
} as unknown as MapLike

type LoadedRegion = {
  regionX: number
  regionY: number
  width: number
  height: number
  loading: boolean
  data: Float32Array | null
}
type Internals = {
  regionRenderer: { regionCache: Map<string, LoadedRegion> }
}

async function loadRegions(
  minRegionSize: number | undefined,
  shape: [number, number] = [4, 8]
): Promise<LoadedRegion[]> {
  let ready: () => void
  const initialized = new Promise<void>((resolve) => {
    ready = resolve
  })
  const layer = new ZarrLayer({
    id: 'zarr',
    store: store(...shape),
    variable: 'temperature',
    colormap: [
      [0, 0, 0],
      [255, 255, 255],
    ],
    clim: [0, shape[0] * shape[1]],
    bounds: [-180, -90, 180, 90],
    latIsAscending: false,
    minRegionSize,
    onLoadingStateChange: (state) => {
      if (!state.metadata) ready()
    },
  })
  layer.onAdd(map, createRecordingGl())
  await initialized
  const cache = (layer as unknown as Internals).regionRenderer.regionCache
  do {
    await new Promise((resolve) => setTimeout(resolve, 0))
  } while ([...cache.values()].some((region) => region.loading))
  return [...cache.values()]
}

async function loadedRegions(minRegionSize?: number) {
  return (await loadRegions(minRegionSize)).map(({ width, height }) => [
    height,
    width,
  ])
}

describe('minRegionSize', () => {
  it('draws one region per chunk by default', async () => {
    expect(await loadedRegions()).toEqual([
      [2, 4],
      [2, 4],
      [2, 4],
      [2, 4],
    ])
  })

  it('groups chunks into larger regions', async () => {
    expect(await loadedRegions(4)).toEqual([
      [4, 4],
      [4, 4],
    ])
  })

  it('clips a grown region to the array edge', async () => {
    expect(await loadedRegions(6)).toEqual([[4, 8]])
  })

  it('assembles a clipped trailing region from the right pixels', async () => {
    // 10x14 in 2x4 chunks grows to 6x8 regions: a 2x2 grid whose
    // bottom-right region is clipped to rows 6-9 and columns 8-13.
    const regions = await loadRegions(6, [10, 14])
    expect(regions).toHaveLength(4)
    const corner = regions.find((r) => r.regionX === 1 && r.regionY === 1)!
    expect([corner.height, corner.width]).toEqual([4, 6])
    const expected: number[] = []
    for (let y = 6; y < 10; y++) {
      for (let x = 8; x < 14; x++) expected.push(y * 14 + x)
    }
    // Region data is normalized by the data scale, here the clim maximum.
    expect(Array.from(corner.data!, (v) => Math.round(v * 140))).toEqual(
      expected
    )
  })

  it('rejects a size that is not a non-negative number', () => {
    for (const minRegionSize of [NaN, -1, Infinity]) {
      expect(
        () =>
          new ZarrLayer({
            id: 'zarr',
            store: store(),
            variable: 'temperature',
            colormap: [
              [0, 0, 0],
              [255, 255, 255],
            ],
            clim: [0, 31],
            minRegionSize,
          })
      ).toThrow('minRegionSize must be a non-negative number')
    }
  })
})

describe('maxRegionFetches', () => {
  const layer = (maxRegionFetches?: number) =>
    new ZarrLayer({
      id: 'zarr',
      store: store(),
      variable: 'temperature',
      colormap: [
        [0, 0, 0],
        [255, 255, 255],
      ],
      clim: [0, 31],
      bounds: [-180, -90, 180, 90],
      latIsAscending: false,
      maxRegionFetches,
    })
  const limit = (l: ZarrLayer) =>
    (
      l as unknown as {
        regionRenderer: { requestCanceller: { maxActive: number } }
      }
    ).regionRenderer.requestCanceller.maxActive

  it('sets how many region fetches run at once', async () => {
    const custom = layer(3)
    custom.onAdd(map, createRecordingGl())
    await custom.ready
    expect(limit(custom)).toBe(3)

    const byDefault = layer()
    byDefault.onAdd(map, createRecordingGl())
    await byDefault.ready
    expect(limit(byDefault)).toBe(16)
  })

  it('rejects a limit that is not a positive integer', () => {
    for (const bad of [0, -1, 1.5, NaN]) {
      expect(() => layer(bad)).toThrow(
        'maxRegionFetches must be a positive integer'
      )
    }
  })
})
