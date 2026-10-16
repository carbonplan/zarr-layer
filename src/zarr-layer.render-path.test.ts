import { describe, it, expect, vi } from 'vitest'
import { ZarrLayer } from './zarr-layer'
import { buildMemoryZarrStore } from './__fixtures__/memory-zarr'
import { createRecordingGl } from './__fixtures__/fake-gl'
import type { MapLike } from './types'

/**
 * The layer-level render-path gate (configureMapboxRenderPath).
 *
 * Mapbox classifies a custom layer as *draped* whenever it exposes
 * renderToTile: Mapbox renders the layer into tile textures and drapes them
 * over the globe or over terrain. Dropping renderToTile switches the layer to
 * the *direct* path, where it draws itself onto the sphere in ECEF and can
 * reach the poles. The presence of `renderToTile` on the layer object is the
 * whole contract, so it is what these tests assert.
 *
 * Getting the gate wrong is invisible until you look at a map: on terrain the
 * layer detaches from the ground, mid-morph it pops, and on MapLibre — which
 * has no draping API at all — engaging the Mapbox-only path would break the
 * layer outright. The gate has five inputs, each covered below:
 *
 *   renderPoles opt-in, dataset ECEF-eligibility, terrain off, globe
 *   projection, and Mapbox having passed its public render parameters (the
 *   8th render/prerender argument, mapbox-gl >= 3.32) at least once.
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

/**
 * mapbox:        mapbox-gl >= 3.32, render callbacks carry the public
 *                render parameters as their 8th argument.
 * mapbox-legacy: older mapbox-gl, callbacks stop at the 7th argument.
 * maplibre:      a single params object, nothing Mapbox-shaped at all.
 */
type Provider = 'mapbox' | 'mapbox-legacy' | 'maplibre'

interface FakeMapOptions {
  globe?: boolean
  terrain?: boolean
  zoom?: number
  provider?: Provider
}

function fakeMap(options: FakeMapOptions = {}) {
  const {
    globe = false,
    terrain = false,
    zoom = 0,
    provider = 'mapbox',
  } = options

  const state = { globe, terrain, zoom }
  const handlers = new Map<string, Array<(...args: unknown[]) => void>>()

  const map = {
    getProjection: () =>
      // Mapbox reports `name`; a MapLibre map reports `type`.
      provider === 'maplibre'
        ? { type: state.globe ? 'globe' : 'mercator' }
        : { name: state.globe ? 'globe' : 'mercator' },
    getTerrain: () => (state.terrain ? { source: 'dem' } : null),
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
    on: (event: string, handler: (...args: unknown[]) => void) => {
      const list = handlers.get(event) ?? []
      list.push(handler)
      handlers.set(event, list)
    },
    off: vi.fn(),
  }

  const identity = () => {
    const m = new Float64Array(16)
    m[0] = m[5] = m[10] = m[15] = 1
    return m
  }

  /** The arguments the provider's painter passes to prerender/render. */
  const renderArgs = (): unknown[] => {
    if (provider === 'maplibre') return [{}]
    const mapboxArgs: unknown[] = [
      Array.from(identity()),
      { name: state.globe ? 'globe' : 'mercator' },
      Array.from(identity()),
      state.globe ? 0 : 1,
      [0.5, 0.5],
      1,
    ]
    if (provider === 'mapbox') {
      mapboxArgs.push({
        projectionMatrix: identity(),
        viewMatrix: identity(),
        globeClippingPlane: null,
        globeCenterInScreenPixels: null,
      })
    }
    return mapboxArgs
  }

  return {
    map: map as unknown as MapLike,
    state,
    renderArgs,
    /** Re-run the layer's gate the way a real map event would. */
    emit: (event: string) => {
      for (const handler of handlers.get(event) ?? []) handler()
    },
  }
}

async function addLayer(
  mapOptions: FakeMapOptions = {},
  layerOptions: {
    renderPoles?: boolean
    crs?: string
    /** Skip the first frame so a test can observe the pre-callback state. */
    frame?: boolean
  } = {}
) {
  const { renderPoles = true, crs = 'EPSG:4326', frame = true } = layerOptions
  const { map, state, emit, renderArgs } = fakeMap(mapOptions)
  const gl = createRecordingGl()

  // onAdd initializes asynchronously without returning a handle, so the layer's
  // own loading callback is what says the store, renderer, and first gate
  // evaluation are done. Waiting on it rather than on a clock keeps the test
  // from racing parallel workers.
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
    crs,
    renderPoles,
    onLoadingStateChange: (loadingState) => {
      if (!loadingState.metadata) ready()
    },
  })

  layer.onAdd(map, gl)
  await initialized
  await settle()

  type Callback = (
    gl: WebGL2RenderingContext,
    params: unknown,
    ...rest: unknown[]
  ) => void

  /** Mapbox runs prerender for draped layers as well, so the first frame is
   *  where the layer learns whether render parameters exist at all. */
  const prerender = () => {
    const [params, ...rest] = renderArgs()
    ;(layer.prerender as Callback)(gl, params, ...rest)
  }
  const render = () => {
    const [params, ...rest] = renderArgs()
    ;(layer.render as Callback)(gl, params, ...rest)
  }
  if (frame) {
    prerender()
    await settle()
  }

  /** True when the layer is draped (Mapbox renders it into tile textures). */
  const isDraped = () => typeof layer.renderToTile === 'function'

  /**
   * The texture Mapbox caches for the draped batch this layer shares with
   * the basemap. It is reused frame to frame while the source tiles are
   * unchanged; the only thing that drops it is a visible custom layer
   * answering shouldRerenderTiles() with true, which Mapbox asks every custom
   * layer, draped or not, right after batching.
   */
  let drapedCache: { holdsLayer: boolean } | null = null

  /**
   * One Mapbox frame, in the order the painter runs it: the layer is sorted
   * into draped batches first (reading `renderToTile` once), the cache is
   * dropped if the layer asks for a re-drape, then prerender runs in the
   * offscreen pass, then the translucent pass either drapes the batch (from
   * cache when it has one, otherwise tile by tile) or calls render. drawCustom
   * re-reads `renderToTile` for each tile, so a layer that loses it mid-frame
   * silently stops drawing.
   */
  const frameLike = (tiles = 3) => {
    const draped = isDraped()
    if (layer.shouldRerenderTiles()) drapedCache = null
    prerender()
    let tilesDrawn = 0
    let tilesSkipped = 0
    let rendered = false
    if (draped) {
      if (!drapedCache) {
        for (let i = 0; i < tiles; i++) {
          if (typeof layer.renderToTile === 'function') {
            layer.renderToTile(gl, { z: 1, x: i, y: 0 })
            tilesDrawn++
          } else {
            tilesSkipped++
          }
        }
        drapedCache = { holdsLayer: tilesDrawn > 0 }
      }
    } else {
      drapedCache ??= { holdsLayer: false }
      render()
      rendered = true
    }
    return { draped, tilesDrawn, tilesSkipped, rendered }
  }

  /** True when the frame just drawn showed the layer both in the cached
   *  draped texture and through its own direct draw. */
  const doubleDrawn = () => !isDraped() && drapedCache?.holdsLayer === true

  return {
    layer,
    map,
    state,
    emit,
    isDraped,
    prerender,
    frameLike,
    doubleDrawn,
  }
}

/** Let the layer's deferred render-path switch run (it lands on a microtask
 *  queued before this one). */
const settle = () => new Promise<void>((resolve) => queueMicrotask(resolve))

describe('Mapbox draped vs direct globe path', () => {
  it('undrapes on the globe so the layer can reach the poles', async () => {
    const { isDraped } = await addLayer({ globe: true, zoom: 0 })
    expect(isDraped()).toBe(false)
  })

  it('stays draped in mercator', async () => {
    const { isDraped } = await addLayer({ globe: false })
    expect(isDraped()).toBe(true)
  })

  it('stays draped on the globe when terrain is on', async () => {
    // Terrain draping is the whole point of renderToTile; taking the direct
    // path here would float the layer off the terrain surface.
    const { isDraped } = await addLayer({ globe: true, terrain: true, zoom: 0 })
    expect(isDraped()).toBe(true)
  })

  it('re-drapes when terrain is switched on, and undrapes when switched off', async () => {
    const { state, emit, isDraped } = await addLayer({ globe: true, zoom: 0 })
    expect(isDraped()).toBe(false)

    state.terrain = true
    emit('move')
    await settle()
    expect(isDraped()).toBe(true)

    state.terrain = false
    emit('move')
    await settle()
    expect(isDraped()).toBe(false)
  })

  it('re-drapes when the projection switches back to mercator', async () => {
    const { state, emit, isDraped } = await addLayer({ globe: true, zoom: 0 })
    expect(isDraped()).toBe(false)

    state.globe = false
    emit('projectionchange')
    await settle()
    expect(isDraped()).toBe(true)
  })

  it('hands the globe-to-mercator morph back to Mapbox', async () => {
    // Mapbox blends globe and mercator across zoom 5-6 using internal matrices
    // the custom-layer callback never sees, so the direct path only holds at
    // the fully-globe endpoint.
    const { state, emit, isDraped } = await addLayer({ globe: true, zoom: 0 })
    expect(isDraped()).toBe(false)

    state.zoom = 5.5
    emit('move')
    await settle()
    expect(isDraped()).toBe(true)

    state.zoom = 8
    emit('move')
    await settle()
    expect(isDraped()).toBe(true)

    state.zoom = 4
    emit('move')
    await settle()
    expect(isDraped()).toBe(false)
  })

  it('holds the direct path right up to the start of the morph', async () => {
    const { state, emit, isDraped } = await addLayer({ globe: true, zoom: 0 })

    // smoothstep(5, 6, z) is still 0 at z=5.
    state.zoom = 5
    emit('move')
    await settle()
    expect(isDraped()).toBe(false)
  })

  it('stays draped when the caller has not opted into pole rendering', async () => {
    const { isDraped } = await addLayer(
      { globe: true, zoom: 0 },
      { renderPoles: false }
    )
    expect(isDraped()).toBe(true)
  })

  it('stays draped for a dataset the ECEF path cannot place', async () => {
    // The direct path builds vertices from lon/lat, so it needs either a proj4
    // definition or EPSG:4326 source coordinates.
    const { isDraped } = await addLayer(
      { globe: true, zoom: 0 },
      { crs: 'EPSG:3857' }
    )
    expect(isDraped()).toBe(true)
  })
})

describe('Mapbox render parameters as the direct-path key', () => {
  it('starts draped and undrapes once a callback carries render parameters', async () => {
    // Nothing outside render/prerender can tell which Mapbox this is, so the
    // layer begins on the path every version supports.
    const { isDraped, prerender } = await addLayer(
      { globe: true, zoom: 0 },
      { frame: false }
    )
    expect(isDraped()).toBe(true)

    prerender()
    await settle()
    expect(isDraped()).toBe(false)
  })

  it('remembers the parameters across map events that never see them', async () => {
    // Terrain toggles re-run the gate from a map event with no callback
    // arguments in sight; the availability learned on the first frame holds.
    const { state, emit, isDraped } = await addLayer({ globe: true, zoom: 0 })
    expect(isDraped()).toBe(false)

    state.terrain = true
    emit('move')
    await settle()
    expect(isDraped()).toBe(true)

    state.terrain = false
    emit('move')
    await settle()
    expect(isDraped()).toBe(false)
  })

  it('learns them in mercator and applies them on the switch to globe', async () => {
    const { state, emit, isDraped } = await addLayer({ globe: false })
    expect(isDraped()).toBe(true)

    state.globe = true
    emit('projectionchange')
    await settle()
    expect(isDraped()).toBe(false)
  })

  it('stays draped for good on a Mapbox older than 3.32', async () => {
    // Without the public matrices the direct path has nothing to project
    // with, so the layer keeps the draped tiles (pole holes and all).
    const { isDraped, prerender, emit } = await addLayer({
      globe: true,
      zoom: 0,
      provider: 'mapbox-legacy',
    })
    expect(isDraped()).toBe(true)

    prerender()
    emit('move')
    await settle()
    expect(isDraped()).toBe(true)
  })
})

describe('render-path switches never split a Mapbox frame', () => {
  // Mapbox batches draped layers before any callback runs and re-reads
  // renderToTile per tile, so a switch applied inside a callback leaves the
  // rest of that frame undrawn. Every switch therefore lands between frames.

  it('drapes the whole first frame and goes direct on the next', async () => {
    const { frameLike } = await addLayer(
      { globe: true, zoom: 0 },
      { frame: false }
    )
    // Render parameters are first seen in this frame's prerender, after the
    // layer was already batched as draped.
    expect(frameLike()).toEqual({
      draped: true,
      tilesDrawn: 3,
      tilesSkipped: 0,
      rendered: false,
    })

    await settle()
    expect(frameLike()).toEqual({
      draped: false,
      tilesDrawn: 0,
      tilesSkipped: 0,
      rendered: true,
    })
  })

  it('finishes the frame direct when terrain is discovered inside render', async () => {
    // setTerrain fires none of the events the layer listens to, so the change
    // is only noticed by the gate run from render.
    const { state, frameLike } = await addLayer({ globe: true, zoom: 0 })
    state.terrain = true
    expect(frameLike()).toEqual({
      draped: false,
      tilesDrawn: 0,
      tilesSkipped: 0,
      rendered: true,
    })

    await settle()
    expect(frameLike()).toEqual({
      draped: true,
      tilesDrawn: 3,
      tilesSkipped: 0,
      rendered: false,
    })
  })

  it('drapes every tile of the frame in which terrain is found to be gone', async () => {
    const { state, frameLike } = await addLayer({
      globe: true,
      terrain: true,
      zoom: 0,
    })
    state.terrain = false
    // The gate runs from the first renderToTile; the remaining tiles of this
    // frame must still be drawn.
    expect(frameLike()).toEqual({
      draped: true,
      tilesDrawn: 3,
      tilesSkipped: 0,
      rendered: false,
    })

    await settle()
    expect(frameLike()).toEqual({
      draped: false,
      tilesDrawn: 0,
      tilesSkipped: 0,
      rendered: true,
    })
  })
})

describe('switching paths invalidates the shared draped batch cache', () => {
  // Crossing the zoom-morph threshold changes which path the layer takes but
  // not the tile coverage, so Mapbox keeps the cached batch texture unless
  // the layer asks for a re-drape.

  /** Draw draped frames until Mapbox would be serving the batch from cache. */
  const settleDrapedCache = async (
    frameLike: () => { draped: boolean; tilesDrawn: number }
  ) => {
    for (let i = 0; i < 50; i++) {
      const frame = frameLike()
      if (frame.draped && frame.tilesDrawn === 0) return
      await new Promise((resolve) => setTimeout(resolve, 10))
    }
    throw new Error('draped cache never settled')
  }

  it('drops the cached texture that still holds the layer when it goes direct', async () => {
    const { state, emit, frameLike, doubleDrawn } = await addLayer({
      globe: true,
      zoom: 5.5,
    })
    await settleDrapedCache(frameLike)

    state.zoom = 4
    emit('move')
    await settle()
    expect(frameLike().rendered).toBe(true)
    // Otherwise the stale draped pixels sit under the direct draw.
    expect(doubleDrawn()).toBe(false)
  })

  it('redraws the batch with the layer when it goes back to draped', async () => {
    const { state, emit, frameLike } = await addLayer({ globe: true, zoom: 4 })
    expect(frameLike().rendered).toBe(true)

    state.zoom = 5.5
    emit('move')
    await settle()
    // A reused texture would be the one drawn without the layer: a gap.
    expect(frameLike()).toEqual({
      draped: true,
      tilesDrawn: 3,
      tilesSkipped: 0,
      rendered: false,
    })
  })
})

describe('MapLibre never takes the Mapbox direct path', () => {
  it('stays draped on a globe map', async () => {
    // MapLibre has no draping API, so renderToTile is inert there — but the
    // render parameters are also the discriminator: MapLibre never passes
    // them, which keeps the Mapbox-specific ECEF program off a MapLibre map.
    const { isDraped, prerender, emit } = await addLayer({
      globe: true,
      zoom: 0,
      provider: 'maplibre',
    })
    expect(isDraped()).toBe(true)

    prerender()
    emit('move')
    await settle()
    expect(isDraped()).toBe(true)
  })

  it('stays draped in mercator too', async () => {
    const { isDraped } = await addLayer({
      globe: false,
      provider: 'maplibre',
    })
    expect(isDraped()).toBe(true)
  })
})
