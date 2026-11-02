import { describe, it, expect, vi } from 'vitest'
import { ZarrLayer } from './zarr-layer'
import { buildMemoryZarrStore, ramp } from './__fixtures__/memory-zarr'
import { createRecordingGl } from './__fixtures__/fake-gl'
import type { MapLike, ZarrLayerOptions } from './types'
import type { QueryGeometry } from './query/types'

function staticMap() {
  return {
    getProjection: () => ({ type: 'mercator' }),
    getTerrain: () => null,
    getZoom: () => 0,
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
}

// Two 4x8 index ramps on a global grid; `humidity` declares fill 0.
function store() {
  const array = (name: string, fillValue?: number) => ({
    name,
    shape: [4, 8],
    chunkShape: [4, 8],
    dimensionNames: ['lat', 'lon'],
    fillValue,
    chunks: { '0/0': ramp(32) },
  })
  return buildMemoryZarrStore({
    arrays: [array('temperature', 7), array('humidity', 0)],
  })
}

function makeLayer(overrides: Partial<ZarrLayerOptions>) {
  const layer = new ZarrLayer({
    id: 'zarr',
    variable: 'temperature',
    colormap: [
      [0, 0, 0],
      [255, 255, 255],
    ],
    clim: [0, 32],
    crs: 'EPSG:4326',
    bounds: [-180, -90, 180, 90],
    latIsAscending: false,
    store: store(),
    ...overrides,
  } as ZarrLayerOptions)
  layer.onAdd(staticMap(), createRecordingGl())
  return layer
}

// Center of pixel (x, 0) of the 4x8 grid.
const topRow = (x: number): QueryGeometry => ({
  type: 'Point',
  coordinates: [((x + 0.5) / 8) * 360 - 180, 67.5],
})

describe('the fillValue option', () => {
  it("replaces the store's fill", async () => {
    const layer = makeLayer({ fillValue: 3 })
    await layer.ready

    expect(layer.fillValue).toBe(3)
    expect((await layer.queryData(topRow(3))).temperature).toEqual([])
    expect((await layer.queryData(topRow(7))).temperature).toEqual([7])
  })

  it('reports the store fill when not set', async () => {
    const layer = makeLayer({})
    await layer.ready

    expect(layer.fillValue).toBe(7)
    expect((await layer.queryData(topRow(7))).temperature).toEqual([])
  })

  it('is cleared by setVariable', async () => {
    const layer = makeLayer({ fillValue: 3 })
    await layer.ready
    await layer.setVariable('humidity')
    await layer.ready

    expect(layer.fillValue).toBe(0)
    expect((await layer.queryData(topRow(3))).humidity).toEqual([3])
    expect((await layer.queryData(topRow(0))).humidity).toEqual([])
  })
})
