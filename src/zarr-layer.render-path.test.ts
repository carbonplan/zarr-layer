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

  /** Mapbox runs prerender for draped layers as well, so the first frame is
   *  where the layer learns whether render parameters exist at all. */
  const prerender = () => {
    const [params, ...rest] = renderArgs()
    ;(
      layer.prerender as (
        gl: WebGL2RenderingContext,
        params: unknown,
        ...rest: unknown[]
      ) => void
    )(gl, params, ...rest)
  }
  if (frame) prerender()

  /** True when the layer is draped (Mapbox renders it into tile textures). */
  const isDraped = () => typeof layer.renderToTile === 'function'

  return { layer, map, state, emit, isDraped, prerender }
}

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
    expect(isDraped()).toBe(true)

    state.terrain = false
    emit('move')
    expect(isDraped()).toBe(false)
  })

  it('re-drapes when the projection switches back to mercator', async () => {
    const { state, emit, isDraped } = await addLayer({ globe: true, zoom: 0 })
    expect(isDraped()).toBe(false)

    state.globe = false
    emit('projectionchange')
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
    expect(isDraped()).toBe(true)

    state.zoom = 8
    emit('move')
    expect(isDraped()).toBe(true)

    state.zoom = 4
    emit('move')
    expect(isDraped()).toBe(false)
  })

  it('holds the direct path right up to the start of the morph', async () => {
    const { state, emit, isDraped } = await addLayer({ globe: true, zoom: 0 })

    // smoothstep(5, 6, z) is still 0 at z=5.
    state.zoom = 5
    emit('move')
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
    expect(isDraped()).toBe(false)
  })

  it('remembers the parameters across map events that never see them', async () => {
    // Terrain toggles re-run the gate from a map event with no callback
    // arguments in sight; the availability learned on the first frame holds.
    const { state, emit, isDraped } = await addLayer({ globe: true, zoom: 0 })
    expect(isDraped()).toBe(false)

    state.terrain = true
    emit('move')
    expect(isDraped()).toBe(true)

    state.terrain = false
    emit('move')
    expect(isDraped()).toBe(false)
  })

  it('learns them in mercator and applies them on the switch to globe', async () => {
    const { state, emit, isDraped } = await addLayer({ globe: false })
    expect(isDraped()).toBe(true)

    state.globe = true
    emit('projectionchange')
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
    expect(isDraped()).toBe(true)
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
