import { describe, it, expect, vi } from 'vitest'
import { ZarrLayer } from './zarr-layer'
import type { RegionRenderer } from './region-renderer'
import { buildMemoryZarrStore } from './__fixtures__/memory-zarr'
import { createRecordingGl } from './__fixtures__/fake-gl'
import type { MapLike } from './types'

/**
 * MapLibre's terrain draping contract. With terrain on, MapLibre calls
 * renderToTerrainTile in place of render, once per terrain tile, with clip
 * space (-1, -1) at the tile's south-west corner, and it redraws a tile only
 * after terrainTileRevision changes. Mapbox's renderToTile shares the tile
 * render path but keeps its own orientation and its shouldRerenderTiles poll.
 */

const memoryStore = () =>
  buildMemoryZarrStore({
    arrays: [
      {
        name: 'temperature',
        shape: [4, 8],
        chunkShape: [2, 4],
        dimensionNames: ['lat', 'lon'],
        chunks: {
          '0/0': [0, 1, 2, 3, 4, 5, 6, 7],
          '0/1': [8, 9, 10, 11, 12, 13, 14, 15],
          '1/0': [16, 17, 18, 19, 20, 21, 22, 23],
          '1/1': [24, 25, 26, 27, 28, 29, 30, 31],
        },
      },
    ],
  })

function fakeMap({ globe = false, zoom = 0 } = {}) {
  const state = { globe, zoom }
  const repaints: number[] = []
  const holder: { layer: ZarrLayer | null } = { layer: null }

  const map = {
    getProjection: () => ({ type: state.globe ? 'globe' : 'mercator' }),
    getTerrain: () => ({ source: 'dem' }),
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
    // Records the revision MapLibre would read when it handles the repaint.
    triggerRepaint: vi.fn(() => {
      repaints.push(holder.layer?.terrainTileRevision ?? -1)
    }),
    on: vi.fn(),
    off: vi.fn(),
  }

  return { map: map as unknown as MapLike, state, repaints, holder }
}

type Seam = { regionRenderer: RegionRenderer }

async function addLayer(
  mapOptions: { globe?: boolean; zoom?: number } = {},
  layerOptions: { maxzoom?: number } = {}
) {
  const { map, state, repaints, holder } = fakeMap(mapOptions)
  const gl = createRecordingGl()

  let ready: () => void
  const initialized = new Promise<void>((resolve) => {
    ready = resolve
  })

  const layer = new ZarrLayer({
    id: 'zarr',
    store: memoryStore(),
    variable: 'temperature',
    colormap: [
      [0, 0, 0],
      [255, 255, 255],
    ],
    clim: [0, 31],
    bounds: [-180, -90, 180, 90],
    latIsAscending: false,
    crs: 'EPSG:4326',
    maxzoom: layerOptions.maxzoom,
    onLoadingStateChange: (loadingState) => {
      if (!loadingState.metadata) ready()
    },
  })
  holder.layer = layer

  layer.onAdd(map, gl)
  await initialized

  const renderToTile = vi.spyOn(
    (layer as unknown as Seam).regionRenderer,
    'renderToTile'
  )

  return { layer, gl, map, state, repaints, renderToTile }
}

const terrainTile = (
  canonical: { z: number; x: number; y: number },
  wrap?: number
) => ({ tileID: { canonical, wrap }, width: 512, height: 512 })

describe('renderToTerrainTile', () => {
  it('draws the canonical tile north-up through the tile render path', async () => {
    const { layer, gl, renderToTile } = await addLayer()

    layer.renderToTerrainTile(gl, terrainTile({ z: 3, x: 2, y: 5 }))

    expect(renderToTile).toHaveBeenCalledTimes(1)
    const [, tileId, context, options] = renderToTile.mock.calls[0]
    expect(tileId).toEqual({ z: 3, x: 2, y: 5 })
    expect(options).toEqual({ northUp: true })
    expect(context.worldOffsets).toEqual([0])
  })

  it('draws the same content for every world copy', async () => {
    const { layer, gl, renderToTile } = await addLayer()

    layer.renderToTerrainTile(gl, terrainTile({ z: 1, x: 0, y: 0 }, -1))
    layer.renderToTerrainTile(gl, terrainTile({ z: 1, x: 0, y: 0 }, 1))

    const tileIds = renderToTile.mock.calls.map(([, tileId]) => tileId)
    expect(tileIds).toEqual([
      { z: 1, x: 0, y: 0 },
      { z: 1, x: 0, y: 0 },
    ])
  })

  it('draws a flat Mercator tile even on the globe', async () => {
    // MapLibre drapes the tile over its own globe; the globe blend belongs to
    // Mapbox's draped globe tiles only.
    const { layer, gl, renderToTile } = await addLayer({ globe: true })

    layer.renderToTerrainTile(gl, terrainTile({ z: 0, x: 0, y: 0 }))

    expect(renderToTile.mock.calls[0][2].isGlobe).toBe(false)
  })

  it('does nothing outside the zoom range', async () => {
    const { layer, gl, renderToTile } = await addLayer(
      { zoom: 6 },
      { maxzoom: 4 }
    )

    layer.renderToTerrainTile(gl, terrainTile({ z: 6, x: 0, y: 0 }))

    expect(renderToTile).not.toHaveBeenCalled()
  })

  it('does nothing after the layer is removed', async () => {
    const { layer, gl, map, renderToTile } = await addLayer()
    layer.onRemove(map, gl)

    layer.renderToTerrainTile(gl, terrainTile({ z: 0, x: 0, y: 0 }))

    expect(renderToTile).not.toHaveBeenCalled()
  })

  it('leaves the revision alone when a tile draws, so drawing cannot loop', async () => {
    const { layer, gl, map, renderToTile } = await addLayer()
    renderToTile.mockReturnValue(true) // more data still to come
    const revision = layer.terrainTileRevision
    vi.mocked(map.triggerRepaint!).mockClear()

    layer.renderToTerrainTile(gl, terrainTile({ z: 0, x: 0, y: 0 }))

    expect(layer.terrainTileRevision).toBe(revision)
    expect(map.triggerRepaint).not.toHaveBeenCalled()
  })
})

describe('terrainTileRevision', () => {
  it('advances before each repaint request when the drawn content changes', async () => {
    const { layer, repaints } = await addLayer()
    const revision = layer.terrainTileRevision
    repaints.length = 0

    layer.setOpacity(0.5)
    layer.setClim([1, 2])

    expect(layer.terrainTileRevision).toBe(revision + 2)
    expect(repaints).toEqual([revision + 1, revision + 2])
  })

  it('advances when a region load lands', async () => {
    // Region loads call the same invalidate hook as the setters; this is what
    // redraws a terrain tile that drew before its data arrived.
    const { layer, repaints } = await addLayer()
    expect(repaints.length).toBeGreaterThan(0)
    expect(layer.terrainTileRevision).toBe(repaints[repaints.length - 1])
  })
})

describe('zoom range with cached terrain tiles', () => {
  it('forces a redraw when the zoom crosses the layer range', async () => {
    // MapLibre reuses a terrain tile's texture across zoom changes while the
    // map moves, so leaving or entering the range must change the revision,
    // or the tile keeps showing (or keeps lacking) the layer.
    const { layer, gl, state, renderToTile, settled } = await addLayer(
      { zoom: 3 },
      { maxzoom: 4 }
    )
    await settled()

    // A tile cache that redraws only when the revision changes.
    const drawnAt = new Map<string, number>()
    const frame = () => {
      layer.prerender(gl, {})
      if (drawnAt.get('0/0/0') !== layer.terrainTileRevision) {
        layer.renderToTerrainTile(gl, terrainTile(WORLD_TILE))
        drawnAt.set('0/0/0', layer.terrainTileRevision)
      }
    }

    frame()
    expect(renderToTile).toHaveBeenCalledTimes(1)
    const inRange = layer.terrainTileRevision
    frame()
    expect(layer.terrainTileRevision).toBe(inRange)

    state.zoom = 6
    frame()
    expect(layer.terrainTileRevision).toBe(inRange + 1)
    // The tile was redrawn and the hook left it empty.
    expect(renderToTile).toHaveBeenCalledTimes(1)
    frame()
    expect(layer.terrainTileRevision).toBe(inRange + 1)

    state.zoom = 3
    frame()
    expect(layer.terrainTileRevision).toBe(inRange + 2)
    expect(renderToTile).toHaveBeenCalledTimes(2)
  })
})

describe('Mapbox renderToTile', () => {
  it('draws north-down and keeps the shouldRerenderTiles poll', async () => {
    const { layer, gl, renderToTile } = await addLayer()
    renderToTile.mockReturnValue(true)

    layer.renderToTile(gl, { z: 2, x: 1, y: 1 })

    const [, tileId, , options] = renderToTile.mock.calls[0]
    expect(tileId).toEqual({ z: 2, x: 1, y: 1 })
    expect(options).toEqual({ northUp: false })
    expect(layer.shouldRerenderTiles()).toBe(true)
    expect(layer.shouldRerenderTiles()).toBe(false)
  })

  it('keeps the globe blend for draped globe tiles', async () => {
    const { layer, gl, renderToTile } = await addLayer({ globe: true })

    layer.renderToTile(gl, { z: 0, x: 0, y: 0 })

    expect(renderToTile.mock.calls[0][2].isGlobe).toBe(true)
  })
})
