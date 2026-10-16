import { describe, it, expect, vi } from 'vitest'
import { ZarrLayer } from './zarr-layer'
import type { RegionRenderer } from './region-renderer'
import { buildMemoryZarrStore } from './__fixtures__/memory-zarr'
import { createRecordingGl } from './__fixtures__/fake-gl'
import type { RegionCache } from './region-cache'
import type { LoadingState, MapLike } from './types'

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

/** The memory store with chunk reads held back until `release` is called. */
function gatedStore() {
  const memory = memoryStore()
  let release!: () => void
  const gate = new Promise<void>((resolve) => {
    release = resolve
  })
  return {
    store: {
      get: async (key: string) => {
        if (key.includes('/temperature/c/')) await gate
        return memory.get(key)
      },
    },
    release,
  }
}

type Canonical = { z: number; x: number; y: number }
type Bounds = [[number, number], [number, number]]
const WORLD: Bounds = [
  [-180, -85],
  [180, 85],
]
/** A viewport inside the south-east region only. */
const SOUTH_EAST: Bounds = [
  [100, -80],
  [170, -10],
]

function fakeMap({
  globe = false,
  zoom = 0,
  terrain = true,
  terrainTileSize,
  cover,
  bounds = WORLD,
}: {
  globe?: boolean
  zoom?: number
  terrain?: boolean
  terrainTileSize?: number
  cover?: Canonical[]
  bounds?: Bounds
} = {}) {
  const state = { globe, zoom, terrain }
  const repaints: number[] = []
  const holder: { layer: ZarrLayer | null } = { layer: null }

  const map = {
    getProjection: () => ({ type: state.globe ? 'globe' : 'mercator' }),
    getTerrain: () => (state.terrain ? { source: 'dem' } : null),
    getSource: (id: string) =>
      id === 'dem' && terrainTileSize
        ? { tileSize: terrainTileSize }
        : undefined,
    ...(cover
      ? { coveringTiles: () => cover.map((canonical) => ({ canonical })) }
      : {}),
    getZoom: () => state.zoom,
    getBounds: () => ({
      getWest: () => bounds[0][0],
      getEast: () => bounds[1][0],
      toArray: () => bounds,
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
type RendererInternals = {
  regionCache: RegionCache
  drawnTiles: Map<string, unknown>
}

/** Hop the macrotask queue until the condition holds. */
async function waitFor(condition: () => boolean): Promise<void> {
  while (!condition()) {
    await new Promise((resolve) => setTimeout(resolve, 0))
  }
}

async function addLayer(
  mapOptions: Parameters<typeof fakeMap>[0] = {},
  layerOptions: {
    maxzoom?: number
    store?: { get(key: string): Promise<Uint8Array | undefined> }
  } = {}
) {
  const { map, state, repaints, holder } = fakeMap(mapOptions)
  const gl = createRecordingGl()

  let ready: () => void
  const initialized = new Promise<void>((resolve) => {
    ready = resolve
  })

  const layer = new ZarrLayer({
    id: 'zarr',
    store: layerOptions.store ?? memoryStore(),
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
    onLoadingStateChange: (loadingState: LoadingState) => {
      if (!loadingState.metadata) ready()
    },
  })
  holder.layer = layer

  layer.onAdd(map, gl)
  await initialized

  const regionRenderer = (layer as unknown as Seam).regionRenderer
  const internals = regionRenderer as unknown as RendererInternals
  const renderToTile = vi.spyOn(regionRenderer, 'renderToTile')
  // Fetches mark their regions loading before they yield, so waiting on the
  // cache sees a batch dispatched by the previous call.
  const settled = () =>
    waitFor(() => {
      for (const cached of internals.regionCache.values()) {
        if (cached.loading) return false
      }
      return true
    })

  return {
    layer,
    gl,
    map,
    state,
    repaints,
    renderToTile,
    regionRenderer,
    internals,
    settled,
  }
}

const terrainTile = (
  canonical: { z: number; x: number; y: number },
  wrap?: number
) => ({ tileID: { canonical, wrap }, width: 512, height: 512 })

const WORLD_TILE = { z: 0, x: 0, y: 0 }

describe('renderToTerrainTile', () => {
  it('draws the canonical tile north-up through the tile render path', async () => {
    const { layer, gl, renderToTile } = await addLayer()

    layer.renderToTerrainTile(gl, terrainTile({ z: 3, x: 2, y: 5 }))

    expect(renderToTile).toHaveBeenCalledTimes(1)
    const [, tile, context, options] = renderToTile.mock.calls[0]
    expect(tile.tileId).toEqual({ z: 3, x: 2, y: 5 })
    expect(options).toEqual({ northUp: true })
    expect(context.worldOffsets).toEqual([0])
  })

  it('draws the same content for every world copy', async () => {
    const { layer, gl, renderToTile } = await addLayer()

    layer.renderToTerrainTile(gl, terrainTile({ z: 1, x: 0, y: 0 }, -1))
    layer.renderToTerrainTile(gl, terrainTile({ z: 1, x: 0, y: 0 }, 1))

    const tileIds = renderToTile.mock.calls.map(([, tile]) => tile.tileId)
    expect(tileIds).toEqual([
      { z: 1, x: 0, y: 0 },
      { z: 1, x: 0, y: 0 },
    ])
  })

  it('draws a flat Mercator tile even on the globe', async () => {
    // MapLibre drapes the tile over its own globe; the globe blend belongs to
    // Mapbox's draped globe tiles only.
    const { layer, gl, renderToTile } = await addLayer({ globe: true })

    layer.renderToTerrainTile(gl, terrainTile(WORLD_TILE))

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

    layer.renderToTerrainTile(gl, terrainTile(WORLD_TILE))

    expect(renderToTile).not.toHaveBeenCalled()
  })

  it('leaves the revision alone when a tile draws, so drawing cannot loop', async () => {
    const { layer, gl, map, renderToTile, settled } = await addLayer()
    await settled()
    renderToTile.mockReturnValue(true) // more data still to come
    const revision = layer.terrainTileRevision
    vi.mocked(map.triggerRepaint!).mockClear()

    layer.renderToTerrainTile(gl, terrainTile(WORLD_TILE))

    expect(layer.terrainTileRevision).toBe(revision)
    expect(map.triggerRepaint).not.toHaveBeenCalled()
  })
})

describe('terrainTileRevision', () => {
  it('advances before each repaint request when the drawn content changes', async () => {
    const { layer, repaints, settled } = await addLayer()
    await settled()
    const revision = layer.terrainTileRevision
    repaints.length = 0

    layer.setOpacity(0.5)
    layer.setClim([1, 2])

    expect(layer.terrainTileRevision).toBe(revision + 2)
    expect(repaints).toEqual([revision + 1, revision + 2])
  })

  it('advances when a region load lands, so a tile drawn while loading is redrawn complete', async () => {
    const { store, release } = gatedStore()
    const { layer, gl, map, renderToTile, settled } = await addLayer(
      {},
      { store }
    )

    // Chunks are held back, so the tile draws with nothing to show and asks
    // for more data.
    layer.renderToTerrainTile(gl, terrainTile(WORLD_TILE))
    expect(renderToTile.mock.results[0].value).toBe(true)
    const revision = layer.terrainTileRevision
    vi.mocked(map.triggerRepaint!).mockClear()

    release()
    await settled()

    expect(layer.terrainTileRevision).toBeGreaterThan(revision)
    expect(map.triggerRepaint).toHaveBeenCalled()
    layer.renderToTerrainTile(gl, terrainTile(WORLD_TILE))
    expect(renderToTile.mock.results[1].value).toBe(false)
  })
})

describe('per-tile levels', () => {
  it('hands the map tile cover to the renderer with each tile at its zoom', async () => {
    // A tile at the center zoom maps back onto the map zoom itself; a tile
    // k zooms coarser maps onto zoom - k.
    const { layer, gl, regionRenderer } = await addLayer({
      zoom: 3.5,
      terrain: false,
      cover: [
        { z: 3, x: 1, y: 1 },
        { z: 1, x: 0, y: 0 },
      ],
    })
    const setCoveringTiles = vi.spyOn(regionRenderer, 'setCoveringTiles')

    layer.prerender(gl, {})

    expect(setCoveringTiles).toHaveBeenCalledWith([
      { tileId: { z: 3, x: 1, y: 1 }, zoom: 3.5, poles: false, wrap: 0 },
      { tileId: { z: 1, x: 0, y: 0 }, zoom: 1.5, poles: false, wrap: 0 },
    ])
  })

  it('extends the cover to the poles when drawn directly on the globe', async () => {
    // Drawn on the globe the layer reaches the poles, which Mercator tiles
    // stop short of; draped terrain tiles stay Mercator.
    const { layer, gl, state, regionRenderer } = await addLayer({
      globe: true,
      terrain: false,
      cover: [{ z: 1, x: 0, y: 0 }],
    })
    const setCoveringTiles = vi.spyOn(regionRenderer, 'setCoveringTiles')

    layer.prerender(gl, {})
    expect(setCoveringTiles.mock.calls[0][0]?.[0].poles).toBe(true)

    state.terrain = true
    layer.prerender(gl, {})
    expect(setCoveringTiles.mock.calls[1][0]?.[0].poles).toBe(false)
  })

  it('maps terrain tiles through twice the terrain source tile size', async () => {
    // MapLibre covers its terrain at twice the source's tile size, so with a
    // 256-pixel DEM the terrain tiles are the 512-pixel cover and a center
    // tile maps straight back onto the map zoom.
    const { layer, gl, regionRenderer } = await addLayer({
      zoom: 3.5,
      terrainTileSize: 256,
    })
    const noteDrawnTile = vi.spyOn(regionRenderer, 'noteDrawnTile')

    layer.renderToTerrainTile(gl, terrainTile({ z: 3, x: 2, y: 3 }))

    expect(noteDrawnTile).toHaveBeenCalledWith(
      { tileId: { z: 3, x: 2, y: 3 }, zoom: 3.5 },
      layer.terrainTileRevision
    )
  })

  it('forgets hook tiles once Mapbox stops draping the layer', async () => {
    // Mapbox exposes no tile cover; it drapes the layer with terrain and on
    // its globe, and draws it directly otherwise. Tiles asked for while
    // draped must not go on steering selection after that.
    const { layer, gl, state, internals } = await addLayer()
    layer.renderToTile(gl, { z: 2, x: 1, y: 1 })
    expect(internals.drawnTiles.size).toBe(1)

    state.terrain = false
    layer.prerender(gl, {})
    expect(internals.drawnTiles.size).toBe(0)

    // The draped globe keeps them until the direct globe path takes over.
    state.globe = true
    layer.renderToTile(gl, { z: 2, x: 1, y: 1 })
    layer.prerender(gl, {})
    expect(internals.drawnTiles.size).toBe(1)
    ;(
      layer as unknown as { usingDirectMapboxGlobePath: boolean }
    ).usingDirectMapboxGlobePath = true
    layer.prerender(gl, {})
    expect(internals.drawnTiles.size).toBe(0)
  })

  it('asks for a frame when a drawn tile needs regions, and not otherwise', async () => {
    // MapLibre runs prerender before it draws tiles and schedules nothing
    // after a tile redraw, so a draw that lacks data requests the frame
    // whose prerender fetches it. The viewport covers the south-east region
    // only; the zoom-3 tile (1, 1) lies inside the north-west one.
    const { layer, gl, map, settled } = await addLayer({ bounds: SOUTH_EAST })
    await settled()
    vi.mocked(map.triggerRepaint!).mockClear()

    layer.renderToTerrainTile(gl, terrainTile({ z: 3, x: 1, y: 1 }))
    expect(map.triggerRepaint).toHaveBeenCalledTimes(1)

    // The frame MapLibre schedules for that request.
    layer.prerender(gl, {})
    await settled()
    vi.mocked(map.triggerRepaint!).mockClear()
    layer.renderToTerrainTile(gl, terrainTile({ z: 3, x: 1, y: 1 }))
    expect(map.triggerRepaint).not.toHaveBeenCalled()
  })

  it('forgets hook tiles once MapLibre terrain is off', async () => {
    const { layer, gl, state, internals } = await addLayer({
      cover: [{ z: 0, x: 0, y: 0 }],
    })
    layer.renderToTerrainTile(gl, terrainTile(WORLD_TILE))
    expect(internals.drawnTiles.size).toBe(1)

    state.terrain = false
    layer.prerender(gl, {})

    expect(internals.drawnTiles.size).toBe(0)
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

    const [, tile, , options] = renderToTile.mock.calls[0]
    expect(tile).toEqual({ tileId: { z: 2, x: 1, y: 1 }, zoom: 2 })
    expect(options).toEqual({ northUp: false })
    expect(layer.shouldRerenderTiles()).toBe(true)
    expect(layer.shouldRerenderTiles()).toBe(false)
  })

  it('keeps the globe blend for draped globe tiles', async () => {
    const { layer, gl, renderToTile } = await addLayer({ globe: true })

    layer.renderToTile(gl, WORLD_TILE)

    expect(renderToTile.mock.calls[0][2].isGlobe).toBe(true)
  })
})
