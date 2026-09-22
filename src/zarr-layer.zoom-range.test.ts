import { describe, it, expect, vi } from 'vitest'
import { ZarrLayer } from './zarr-layer'
import { buildMemoryZarrStore } from './__fixtures__/memory-zarr'
import { createRecordingGl } from './__fixtures__/fake-gl'
import type { MapLike } from './types'

/**
 * A layer outside its zoom range must not fetch. The per-frame path checks
 * the range, but initialization finishes asynchronously and then updates
 * once more: a map zoomed out while the layer was still loading would
 * otherwise enumerate and fetch every visible region at a zoom the layer was
 * told to skip. On a full-resolution store with no pyramid that is thousands
 * of regions, started synchronously, which locks the page.
 */

const CHUNKS: Record<string, number[]> = {
  '0/0': [0, 1, 2, 3, 4, 5, 6, 7],
  '0/1': [8, 9, 10, 11, 12, 13, 14, 15],
  '1/0': [16, 17, 18, 19, 20, 21, 22, 23],
  '1/1': [24, 25, 26, 27, 28, 29, 30, 31],
}

/**
 * A two-level pyramid: the 4x8 array above and its 8x16 upsampling, one
 * chunk each.
 */
function pyramidStore() {
  const coarse = Object.values(CHUNKS).flat()
  const fine = Array.from({ length: 8 * 16 }, (_, i) => {
    const y = Math.floor(i / 16)
    const x = i % 16
    return Math.floor(y / 2) * 8 + Math.floor(x / 2)
  })
  return buildMemoryZarrStore({
    attributes: {
      multiscales: {
        layout: [{ asset: '0' }, { asset: '1' }],
        crs: 'EPSG:4326',
      },
    },
    arrays: [
      {
        name: '0/temperature',
        shape: [4, 8],
        chunkShape: [4, 8],
        dimensionNames: ['lat', 'lon'],
        chunks: { '0/0': coarse },
      },
      {
        name: '1/temperature',
        shape: [8, 16],
        chunkShape: [8, 16],
        dimensionNames: ['lat', 'lon'],
        chunks: { '0/0': fine },
      },
    ],
  })
}

// With `times`, the array gains a leading time dimension, one chunk per step.
function countingStore(times?: number, pyramid = false) {
  const chunks: Record<string, number[]> = {}
  for (let t = 0; t < (times ?? 1); t++) {
    for (const [key, values] of Object.entries(CHUNKS)) {
      chunks[times ? `${t}/${key}` : key] = values
    }
  }
  const memory = pyramid
    ? pyramidStore()
    : buildMemoryZarrStore({
        arrays: [
          {
            name: 'temperature',
            shape: times ? [times, 4, 8] : [4, 8],
            chunkShape: times ? [1, 2, 4] : [2, 4],
            dimensionNames: times ? ['time', 'lat', 'lon'] : ['lat', 'lon'],
            chunks,
          },
        ],
      })
  const chunkReads: string[] = []
  return {
    chunkReads,
    store: {
      get: async (key: string) => {
        if (key.includes('/c/')) chunkReads.push(key)
        return memory.get(key)
      },
    },
  }
}

function fakeMap(zoom: number) {
  const state = { zoom }
  const map = {
    getProjection: () => ({ type: 'mercator' }),
    getTerrain: () => null,
    getZoom: () => state.zoom,
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
  }
  return { map: map as unknown as MapLike, state }
}

async function addLayer(
  zoom: number,
  minzoom: number,
  opts: {
    times?: number
    pyramid?: boolean
    selector?: ConstructorParameters<typeof ZarrLayer>[0]['selector']
  } = {}
) {
  const { store, chunkReads } = countingStore(opts.times, opts.pyramid)
  const { map, state } = fakeMap(zoom)
  const gl = createRecordingGl()

  let ready: () => void
  const initialized = new Promise<void>((resolve) => {
    ready = resolve
  })
  const layer = new ZarrLayer({
    id: 'zarr',
    store,
    variable: 'temperature',
    colormap: [
      [0, 0, 0],
      [255, 255, 255],
    ],
    clim: [0, 31],
    bounds: [-180, -90, 180, 90],
    latIsAscending: false,
    minzoom,
    selector: opts.selector,
    onLoadingStateChange: (loadingState) => {
      if (!loadingState.metadata) ready()
    },
  })
  layer.onAdd(map, gl)
  await initialized
  // Let any fetch the initialization kicked off reach the store.
  await new Promise((resolve) => setTimeout(resolve, 20))
  return { layer, gl, map, state, chunkReads }
}

/**
 * Wait, a macrotask at a time, until `done` holds. Deliberately unbounded:
 * a wall-clock cutoff would turn a slow machine into a confusing assertion
 * failure, and vitest's testTimeout reports a real hang as one.
 */
async function until(done: () => boolean): Promise<void> {
  while (!done()) await new Promise((resolve) => setTimeout(resolve, 0))
}

describe('zoom range', () => {
  it('fetches nothing when added below minzoom', async () => {
    const { chunkReads } = await addLayer(2, 5)
    expect(chunkReads).toEqual([])
  })

  it('fetches once the map zooms into range', async () => {
    const { layer, gl, state, chunkReads } = await addLayer(2, 5)

    state.zoom = 6
    layer.prerender(gl, {})
    await until(() => chunkReads.length > 0)
  })

  it('fetches after entering range on a pyramid, once the level commits', async () => {
    const { layer, gl, map, state, chunkReads } = await addLayer(2, 5, {
      pyramid: true,
    })
    expect(chunkReads).toEqual([])
    const repaint = map.triggerRepaint as ReturnType<typeof vi.fn>

    // Frames run only while the map repaints, so a frame is drawn here only
    // when the layer asks for one. A layer that stopped asking before it
    // fetched would hang this loop, and the test timeout reports it.
    state.zoom = 6
    let frames = 0
    while (chunkReads.length === 0) {
      expect(frames++).toBeLessThan(10)
      const before = repaint.mock.calls.length
      layer.prerender(gl, {})
      await until(
        () => chunkReads.length > 0 || repaint.mock.calls.length > before
      )
    }
  })

  it('fetches on add when the map is already in range', async () => {
    const { chunkReads } = await addLayer(6, 5)
    expect(chunkReads.length).toBeGreaterThan(0)
  })

  it('fetches the current selector after a change made below minzoom', async () => {
    const { layer, gl, state, chunkReads } = await addLayer(2, 5, {
      times: 2,
      selector: { time: { selected: 0, type: 'index' } },
    })
    await layer.setSelector({ time: { selected: 1, type: 'index' } })

    state.zoom = 6
    layer.prerender(gl, {})
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(chunkReads.length).toBeGreaterThan(0)
    expect(chunkReads.every((key) => key.includes('/c/1/'))).toBe(true)
  })
})
