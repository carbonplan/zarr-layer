import { describe, it, expect, vi } from 'vitest'
import { ZarrLayer } from './zarr-layer'
import { ZarrLayerNotReadyError } from './errors'
import { buildMemoryZarrStore } from './__fixtures__/memory-zarr'
import { createRecordingGl } from './__fixtures__/fake-gl'
import type { MapLike, ZarrLayerOptions } from './types'

/**
 * `queryData` readiness on a map that never renders.
 *
 * A multiscale store deliberately commits no level during init — the level is
 * picked from the map zoom, which only the render path knows. On a map that
 * never paints, nothing ever commits one, so a query had no level to index
 * into. The failure was silent: "layer isn't ready" and "point has no data"
 * were the same empty result, so callers rendered "no data" over data.
 *
 * These pin the contract that a query resolves against real data whenever it
 * is issued after `onAdd`, with no render pass and no caller-side polling.
 */

/** A map that never paints: `triggerRepaint` is recorded but runs no frame. */
function staticMap(zoom = 0) {
  return {
    getProjection: () => ({ type: 'mercator' }),
    getTerrain: () => null,
    getZoom: () => zoom,
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

function makeLayer(overrides: Partial<ZarrLayerOptions> = {}) {
  return new ZarrLayer({
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
    ...overrides,
  } as ZarrLayerOptions)
}

/**
 * A 2-level global pyramid carrying the same field at two resolutions: the
 * coarse level is a 4x8 index ramp (value = y*8 + x) and the fine level is its
 * 8x16 nearest-neighbour upsampling. A point therefore has one correct value
 * no matter which level answers, so value assertions test the query rather
 * than the level-selection heuristic.
 */
const COARSE = { width: 8, height: 4 }

function samefieldPyramid({ declareShapes = false } = {}) {
  const shapeOf = (level: 0 | 1) =>
    declareShapes
      ? {
          'spatial:shape': [
            COARSE.height * (level + 1),
            COARSE.width * (level + 1),
          ],
        }
      : {}
  const coarse = Array.from(
    { length: COARSE.height * COARSE.width },
    (_, i) => i
  )
  const fine = Array.from(
    { length: COARSE.height * 2 * COARSE.width * 2 },
    (_, i) => {
      const y = Math.floor(i / (COARSE.width * 2))
      const x = i % (COARSE.width * 2)
      return Math.floor(y / 2) * COARSE.width + Math.floor(x / 2)
    }
  )
  return buildMemoryZarrStore({
    attributes: {
      multiscales: {
        layout: [
          { asset: '0', ...shapeOf(0) },
          { asset: '1', ...shapeOf(1) },
        ],
        crs: 'EPSG:4326',
      },
    },
    arrays: [
      {
        name: '0/temperature',
        shape: [COARSE.height, COARSE.width],
        chunkShape: [COARSE.height, COARSE.width],
        dimensionNames: ['lat', 'lon'],
        chunks: { '0/0': coarse },
      },
      {
        name: '1/temperature',
        shape: [COARSE.height * 2, COARSE.width * 2],
        chunkShape: [COARSE.height * 2, COARSE.width * 2],
        dimensionNames: ['lat', 'lon'],
        chunks: { '0/0': fine },
      },
    ],
  })
}

/** Center of coarse-grid pixel (x, y) as [lon, lat] on a global extent. */
function pixelCenter(x: number, y: number): [number, number] {
  return [
    ((x + 0.5) / COARSE.width) * 360 - 180,
    90 - ((y + 0.5) / COARSE.height) * 180,
  ]
}

describe('queryData readiness', () => {
  it('resolves with data when called immediately after onAdd on a map that never renders', async () => {
    const layer = makeLayer({ store: samefieldPyramid() })
    layer.onAdd(staticMap(), createRecordingGl())

    const result = await layer.queryData({
      type: 'Point',
      coordinates: pixelCenter(3, 1),
    })

    expect(result.temperature).toEqual([1 * COARSE.width + 3])
  })

  it('queries without the caller awaiting any loading-state callback', async () => {
    const onLoadingStateChange = vi.fn()
    const layer = makeLayer({
      store: samefieldPyramid(),
      onLoadingStateChange,
    })
    layer.onAdd(staticMap(), createRecordingGl())

    const result = await layer.queryData({
      type: 'Point',
      coordinates: pixelCenter(0, 0),
    })

    expect(result.temperature).toEqual([0])
  })

  it('returns a genuine empty result for a point outside coverage', async () => {
    const layer = makeLayer({
      store: samefieldPyramid(),
      bounds: [0, 0, 10, 10],
    })
    layer.onAdd(staticMap(), createRecordingGl())

    const result = await layer.queryData({
      type: 'Point',
      coordinates: [100, 50],
    })

    expect(result.temperature).toEqual([])
  })

  it('resolves `ready` after the first level commits', async () => {
    const layer = makeLayer({ store: samefieldPyramid() })
    layer.onAdd(staticMap(), createRecordingGl())

    await layer.ready

    const result = await layer.queryData({
      type: 'Point',
      coordinates: pixelCenter(7, 3),
    })
    expect(result.temperature).toEqual([3 * COARSE.width + 7])
  })

  it('holds `ready` open until the level commits, not just until metadata lands', async () => {
    // Init opens level 0's array to learn the variable's dimensions, and
    // declaring `spatial:shape` stops it probing the rest. The fine level is
    // therefore first read when it commits, so gating it separates the two
    // readiness stages. Zoom 0 picks the fine level here: both are narrower
    // than a 256px world, so selection falls through to the finest.
    let releaseLevel = () => {}
    const levelGate = new Promise<void>((resolve) => {
      releaseLevel = resolve
    })
    const backing = samefieldPyramid({ declareShapes: true })

    let metadataSettled = () => {}
    const metadataDone = new Promise<void>((resolve) => {
      metadataSettled = resolve
    })

    const layer = makeLayer({
      store: {
        get: async (key: string) => {
          if (key === '/1/temperature/zarr.json') await levelGate
          return backing.get(key)
        },
      },
      onLoadingStateChange: (state) => {
        if (!state.metadata) metadataSettled()
      },
    })
    const map = staticMap()
    layer.onAdd(map, createRecordingGl())

    let resolved = false
    void layer.ready.then(() => {
      resolved = true
    })

    await metadataDone
    await new Promise((r) => setTimeout(r, 0))
    expect(resolved).toBe(false)
    const repaintsBeforeCommit = vi.mocked(map.triggerRepaint!).mock.calls
      .length

    releaseLevel()
    await layer.ready
    expect(resolved).toBe(true)
    // The commit asks the map to paint. Without it a layer added to an idle
    // map never gets the frame whose update() starts chunk fetches, so it
    // stays blank however long you wait.
    expect(vi.mocked(map.triggerRepaint!).mock.calls.length).toBeGreaterThan(
      repaintsBeforeCommit
    )
  })

  it('serves concurrent queries issued before any level is committed', async () => {
    const layer = makeLayer({ store: samefieldPyramid() })
    layer.onAdd(staticMap(), createRecordingGl())

    const results = await Promise.all([
      layer.queryData({ type: 'Point', coordinates: pixelCenter(1, 0) }),
      layer.queryData({ type: 'Point', coordinates: pixelCenter(2, 2) }),
      layer.queryData({ type: 'Point', coordinates: pixelCenter(6, 1) }),
    ])

    expect(results.map((r) => r.temperature)).toEqual([
      [1],
      [2 * COARSE.width + 2],
      [1 * COARSE.width + 6],
    ])
  })
})

/**
 * The query commits the level the render loop would have picked, not a
 * hardcoded index. Fixture: two levels wide enough that zoom decides between
 * them (level selection compares a level's pixel width against
 * 256 * 2^zoom), each filled with a constant that names the level.
 */
describe('queryData level selection', () => {
  const LEVEL_MARKER = { coarse: 1, fine: 2 }

  function markedPyramid() {
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
          shape: [128, 256],
          chunkShape: [128, 256],
          dimensionNames: ['lat', 'lon'],
          chunks: { '0/0': new Float32Array(128 * 256).fill(1) },
        },
        {
          name: '1/temperature',
          shape: [256, 512],
          chunkShape: [256, 512],
          dimensionNames: ['lat', 'lon'],
          chunks: { '0/0': new Float32Array(256 * 512).fill(2) },
        },
      ],
    })
  }

  it('uses the coarse level at low zoom', async () => {
    const layer = makeLayer({ store: markedPyramid() })
    layer.onAdd(staticMap(0), createRecordingGl())

    const result = await layer.queryData({ type: 'Point', coordinates: [0, 0] })

    expect(result.temperature).toEqual([LEVEL_MARKER.coarse])
  })

  it('uses the fine level at higher zoom', async () => {
    const layer = makeLayer({ store: markedPyramid() })
    layer.onAdd(staticMap(1), createRecordingGl())

    const result = await layer.queryData({ type: 'Point', coordinates: [0, 0] })

    expect(result.temperature).toEqual([LEVEL_MARKER.fine])
  })
})

describe('queryData on a layer that cannot become ready', () => {
  it('throws when the layer was never added to a map', async () => {
    const layer = makeLayer({ store: samefieldPyramid() })

    await expect(
      layer.queryData({ type: 'Point', coordinates: [0, 0] })
    ).rejects.toBeInstanceOf(ZarrLayerNotReadyError)
  })

  it('throws when the layer has been removed', async () => {
    const layer = makeLayer({ store: samefieldPyramid() })
    const gl = createRecordingGl()
    layer.onAdd(staticMap(), gl)
    await layer.ready
    layer.onRemove(staticMap(), gl)

    await expect(
      layer.queryData({ type: 'Point', coordinates: pixelCenter(0, 0) })
    ).rejects.toThrow(/removed from the map/)
  })

  it('throws when initialization failed, carrying the cause', async () => {
    const layer = makeLayer({
      store: { get: async () => undefined },
      bounds: undefined,
    })
    layer.onAdd(staticMap(), createRecordingGl())

    const error = await layer
      .queryData({ type: 'Point', coordinates: [0, 0] })
      .then(
        () => null,
        (e) => e
      )

    expect(error).toBeInstanceOf(ZarrLayerNotReadyError)
    expect((error as ZarrLayerNotReadyError).cause).toBeInstanceOf(Error)
  })

  it('rejects with the abort reason when the query signal aborts mid-wait', async () => {
    let releaseMetadata = () => {}
    const gate = new Promise<void>((resolve) => {
      releaseMetadata = resolve
    })
    const backing = samefieldPyramid()
    const layer = makeLayer({
      store: {
        get: async (key: string) => {
          if (key.includes('/1/')) await gate
          return backing.get(key)
        },
      },
    })
    layer.onAdd(staticMap(), createRecordingGl())

    const controller = new AbortController()
    const pending = layer.queryData(
      { type: 'Point', coordinates: pixelCenter(0, 0) },
      undefined,
      { signal: controller.signal }
    )
    controller.abort()

    await expect(pending).rejects.toThrow(/abort/i)
    releaseMetadata()
  })
})

/**
 * `options.level` decides which resolution a query reads. The default follows
 * the map so results agree with what is drawn; `'finest'` pins the answer to
 * the highest-resolution level regardless of zoom.
 *
 * Fixture: levels wide enough that zoom picks between them (selection compares
 * a level's pixel width against 256 * 2^zoom), each filled with a constant
 * naming the level.
 */
describe('queryData level option', () => {
  const MARKER = { coarse: 1, fine: 2 }

  function markedPyramid({ reversed = false } = {}) {
    const levels = [
      {
        asset: '0',
        shape: [128, 256] as [number, number],
        fill: MARKER.coarse,
      },
      { asset: '1', shape: [256, 512] as [number, number], fill: MARKER.fine },
    ]
    const layout = reversed
      ? [{ asset: '1' }, { asset: '0' }]
      : [{ asset: '0' }, { asset: '1' }]
    return buildMemoryZarrStore({
      attributes: { multiscales: { layout, crs: 'EPSG:4326' } },
      arrays: levels.map((l) => ({
        name: `${l.asset}/temperature`,
        shape: l.shape,
        chunkShape: l.shape,
        dimensionNames: ['lat', 'lon'],
        chunks: {
          '0/0': new Float32Array(l.shape[0] * l.shape[1]).fill(l.fill),
        },
      })),
    })
  }

  const queryOrigin = (layer: ZarrLayer, level?: 'current' | 'finest') =>
    layer.queryData({ type: 'Point', coordinates: [0, 0] }, undefined, {
      level,
    })

  it('defaults to the level the map is drawing', async () => {
    const layer = makeLayer({ store: markedPyramid() })
    layer.onAdd(staticMap(0), createRecordingGl())

    expect((await queryOrigin(layer)).temperature).toEqual([MARKER.coarse])
  })

  it('reads the finest level at a zoom that would draw the coarse one', async () => {
    const layer = makeLayer({ store: markedPyramid() })
    layer.onAdd(staticMap(0), createRecordingGl())

    expect((await queryOrigin(layer, 'finest')).temperature).toEqual([
      MARKER.fine,
    ])
  })

  it('leaves the rendered level alone when a finest query reads past it', async () => {
    const layer = makeLayer({ store: markedPyramid() })
    layer.onAdd(staticMap(0), createRecordingGl())

    expect((await queryOrigin(layer, 'finest')).temperature).toEqual([
      MARKER.fine,
    ])
    // The finest read is query-local: a default query still answers from the
    // level the renderer committed, so it never dragged the render loop along.
    expect((await queryOrigin(layer)).temperature).toEqual([MARKER.coarse])
  })

  it('finds the finest level by resolution, not by position in the pyramid', async () => {
    const layer = makeLayer({ store: markedPyramid({ reversed: true }) })
    layer.onAdd(staticMap(0), createRecordingGl())

    expect((await queryOrigin(layer, 'finest')).temperature).toEqual([
      MARKER.fine,
    ])
  })

  it('is a no-op on a single-level store', async () => {
    const layer = makeLayer({
      clim: [0, 4],
      store: buildMemoryZarrStore({
        arrays: [
          {
            name: 'temperature',
            shape: [2, 2],
            chunkShape: [2, 2],
            dimensionNames: ['lat', 'lon'],
            chunks: { '0/0': [1, 2, 3, 4] },
          },
        ],
      }),
    })
    layer.onAdd(staticMap(0), createRecordingGl())

    expect((await queryOrigin(layer, 'finest')).temperature).toEqual(
      (await queryOrigin(layer)).temperature
    )
  })
})

/**
 * Failures on the readiness path must surface as rejections. Reporting them as
 * an empty result would recreate the exact bug this work exists to fix: a
 * layer that holds no usable data answering identically to a point that
 * genuinely has none.
 */
describe('queryData failure reporting', () => {
  /** Backing store with every key under `prefix` missing, so opening that
   *  level fails while the rest of the store stays intact. */
  const withMissingLevel = (prefix: string) => {
    const backing = samefieldPyramid({ declareShapes: true })
    return {
      get: async (key: string) =>
        key.startsWith(prefix) ? undefined : backing.get(key),
    }
  }

  const point = { type: 'Point' as const, coordinates: pixelCenter(0, 0) }

  it('rejects rather than answering empty when no level can be loaded', async () => {
    // Zoom 0 targets the fine level here, so making it unreadable leaves the
    // renderer with nothing committed.
    const layer = makeLayer({ store: withMissingLevel('/1/') })
    layer.onAdd(staticMap(), createRecordingGl())

    await expect(layer.queryData(point)).rejects.toBeInstanceOf(
      ZarrLayerNotReadyError
    )
    await expect(layer.ready).rejects.toThrow(/no resolution level/)
  })

  it('rejects when the layer is removed while it is becoming ready', async () => {
    let releaseLevel = () => {}
    const levelGate = new Promise<void>((resolve) => {
      releaseLevel = resolve
    })
    const backing = samefieldPyramid({ declareShapes: true })
    const layer = makeLayer({
      store: {
        get: async (key: string) => {
          if (key === '/1/temperature/zarr.json') await levelGate
          return backing.get(key)
        },
      },
    })
    const map = staticMap()
    const gl = createRecordingGl()
    layer.onAdd(map, gl)

    const pending = layer.queryData(point)
    await new Promise((r) => setTimeout(r, 0))
    layer.onRemove(map, gl)
    releaseLevel()

    // Reported as a removal, not as "no level could be loaded" — disposal
    // clears the committed level too, so both branches would fire and only
    // the earlier one names the actual cause.
    await expect(pending).rejects.toThrow(/removed while it was becoming ready/)
  })

  it('rejects when the finest level cannot be opened', async () => {
    // Levels wide enough that zoom 0 draws the coarse one, so the render
    // level commits normally and only the finest read fails. Shapes are
    // declared so the finest level is identifiable without opening it.
    const backing = buildMemoryZarrStore({
      attributes: {
        multiscales: {
          layout: [
            { asset: '0', 'spatial:shape': [128, 256] },
            { asset: '1', 'spatial:shape': [256, 512] },
          ],
          crs: 'EPSG:4326',
        },
      },
      arrays: [
        {
          name: '0/temperature',
          shape: [128, 256],
          chunkShape: [128, 256],
          dimensionNames: ['lat', 'lon'],
          chunks: { '0/0': new Float32Array(128 * 256).fill(1) },
        },
      ],
    })
    const layer = makeLayer({
      store: {
        get: async (key: string) =>
          key.startsWith('/1/') ? undefined : backing.get(key),
      },
    })
    layer.onAdd(staticMap(0), createRecordingGl())

    // The default query still works — only the finest read is broken.
    expect((await layer.queryData(point)).temperature).toEqual([1])
    await expect(
      layer.queryData(point, undefined, { level: 'finest' })
    ).rejects.toThrow(/failed to open level 1 for query/)
  })

  it('wraps an initialization that rejected before its own error handling', async () => {
    // resolveGl throws on a context that isn't WebGL2, before _onAddAsync
    // reaches the try block that would have recorded the failure.
    const layer = makeLayer({ store: samefieldPyramid() })
    layer.onAdd(staticMap(), {} as WebGL2RenderingContext)

    const error = await layer.queryData(point).then(
      () => null,
      (e) => e
    )

    expect(error).toBeInstanceOf(ZarrLayerNotReadyError)
    expect((error as ZarrLayerNotReadyError).cause).toBeInstanceOf(Error)
    expect((error as ZarrLayerNotReadyError).message).toMatch(/WebGL2/)
  })
})
