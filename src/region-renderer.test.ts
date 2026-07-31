import { describe, it, expect, vi } from 'vitest'
import { RegionRenderer } from './region-renderer'
import { createRegionState, type RegionCache } from './region-cache'
import { ZarrStore } from './zarr-store'
import { buildMemoryZarrStore } from './__fixtures__/memory-zarr'
import type { MapLike } from './types'
import type { RegionRenderState } from './renderer-types'
import type { RegionState } from './region-state'

/**
 * Wiring tests for the orchestrator: a real ZarrStore over the in-memory Zarr
 * fixture driven through the public surface (initialize / update / setSelector
 * / queryData / dispose), with a recording fake for the GL context. Everything
 * update() touches is GL-free apart from resource creation and disposal, so
 * the whole viewport -> level -> region -> fetch -> cache -> upload path runs
 * here; only the draw call itself is out of reach.
 */

// 2-time x 4-lat x 8-lon ramp, value = t*32 + y*8 + x, chunked 2x4 in lat/lon
// so the level is a 2x2 region grid and both time steps share a chunk.
const HEIGHT = 4
const WIDTH = 8

function chunk(chunkY: number, chunkX: number): number[] {
  const out: number[] = []
  for (let t = 0; t < 2; t++) {
    for (let y = chunkY * 2; y < chunkY * 2 + 2; y++) {
      for (let x = chunkX * 4; x < chunkX * 4 + 4; x++) {
        out.push(t * 32 + y * WIDTH + x)
      }
    }
  }
  return out
}

function fakeGl({ failTextures = false }: { failTextures?: boolean } = {}) {
  let textures = 0
  let buffers = 0
  return {
    TEXTURE0: 0x84c0,
    TEXTURE_2D: 0x0de1,
    createTexture: vi.fn(() => (failTextures ? null : { tex: ++textures })),
    createBuffer: vi.fn(() => ({ buf: ++buffers })),
    deleteTexture: vi.fn(),
    deleteBuffer: vi.fn(),
    bindTexture: vi.fn(),
    bindBuffer: vi.fn(),
    bufferData: vi.fn(),
    texImage2D: vi.fn(),
    texParameteri: vi.fn(),
    activeTexture: vi.fn(),
  } as unknown as WebGL2RenderingContext & {
    createTexture: ReturnType<typeof vi.fn>
    deleteTexture: ReturnType<typeof vi.fn>
    texImage2D: ReturnType<typeof vi.fn>
    bufferData: ReturnType<typeof vi.fn>
  }
}

const mapAt = (
  west: number,
  south: number,
  east: number,
  north: number
): MapLike => ({
  getBounds: () => ({
    getWest: () => west,
    getEast: () => east,
    toArray: () => [
      [west, south],
      [east, north],
    ],
  }),
  getZoom: () => 0,
})

/**
 * getRegionStates is the render seam — the point where CPU-side regions become
 * draw-ready GPU state. Reaching it through render() would need a real
 * ZarrRenderer and a compiled program, so the test calls it directly.
 */
type RenderSeam = {
  getRegionStates(gl: WebGL2RenderingContext): RegionRenderState[]
  createRegionGeometry(
    regionX: number,
    regionY: number,
    region: RegionState
  ): void
  regionCache: RegionCache
}
const seam = (renderer: RegionRenderer) => renderer as unknown as RenderSeam

/**
 * A drawable region from another level, standing in for coverage left over
 * from before a zoom change. Seeded directly because reaching this state
 * needs a second level to have been fetched, uploaded, and then zoomed away
 * from.
 */
function seedFallbackRegion(
  renderer: RegionRenderer,
  {
    levelIndex = 1,
    uploaded = true,
    bands,
  }: { levelIndex?: number; uploaded?: boolean; bands?: string[] } = {}
) {
  const region = createRegionState(levelIndex, 0, 0, false, 0)
  region.data = new Float32Array([1, 2, 3, 4])
  region.width = 2
  region.height = 2
  region.vertexArr = new Float32Array(8)
  region.pixCoordArr = new Float32Array(8)
  region.mercatorBounds = { x0: 0, y0: 0, x1: 1, y1: 1 }
  region.levelMeta = { width: 2, height: 2, regionSize: [2, 2] }
  // A fallback was fetched under the same selector, so it carries the same
  // bands as the current level.
  for (const band of bands ?? []) {
    region.bandData.set(band, new Float32Array([1, 2, 3, 4]))
  }
  if (uploaded) {
    region.texture = {} as WebGLTexture
    region.vertexBuffer = {} as WebGLBuffer
    region.pixCoordBuffer = {} as WebGLBuffer
    region.textureUploaded = true
    region.geometryUploaded = true
    for (const band of bands ?? []) {
      region.bandTextures.set(band, {} as WebGLTexture)
      region.bandTexturesUploaded.add(band)
      region.bandTexturesConfigured.add(band)
    }
  }

  const cache = seam(renderer).regionCache
  cache.set(region.key, region)
  cache.rebuildProtection([region.key], { retainKeysNotMatching: '' })
  return region
}

/**
 * Let the fire-and-forget fetch chain started by update() run to quiescence.
 * `fetchRegions` marks its batch loading before it yields, so an in-flight
 * batch is already visible here. Hopping the macrotask queue until nothing is
 * loading keeps the tests independent of how many awaits deep the chain runs.
 *
 * Deliberately unbounded: a wall-clock bail-out would return early whenever
 * parallel workers slow a fetch down, turning contention into a confusing
 * assertion failure somewhere below. Vitest's own testTimeout is the backstop,
 * and it reports a hang as a hang.
 */
async function settle(renderer: RegionRenderer): Promise<void> {
  const fetching = () => {
    for (const region of seam(renderer).regionCache.values()) {
      if (region.loading) return true
    }
    return false
  }
  do {
    await new Promise((resolve) => setTimeout(resolve, 0))
  } while (fetching())
}

async function makeRenderer() {
  const memory = buildMemoryZarrStore({
    arrays: [
      {
        name: 'temperature',
        shape: [2, HEIGHT, WIDTH],
        chunkShape: [2, 2, 4],
        dimensionNames: ['time', 'lat', 'lon'],
        chunks: {
          '0/0/0': chunk(0, 0),
          '0/0/1': chunk(0, 1),
          '0/1/0': chunk(1, 0),
          '0/1/1': chunk(1, 1),
        },
      },
      {
        name: 'time',
        shape: [2],
        chunkShape: [2],
        dimensionNames: ['time'],
        chunks: { '0': [10, 20] },
      },
    ],
  })
  const chunkReads: string[] = []
  const counting = {
    get: async (key: string) => {
      if (key.includes('/temperature/c/')) chunkReads.push(key)
      return memory.get(key)
    },
  }
  const store = new ZarrStore({
    customStore: counting,
    variable: 'temperature',
    version: 3,
    bounds: [-180, -90, 180, 90],
    latIsAscending: false,
  })
  await store.initialized

  const invalidate = vi.fn()
  const renderer = new RegionRenderer(store, 'temperature', {}, invalidate)
  await renderer.initialize()
  return {
    renderer,
    invalidate,
    chunkReads,
    gl: fakeGl(),
    map: mapAt(-180, -85, 180, 85),
  }
}

describe('RegionRenderer', () => {
  it('fetches the visible regions of the committed level on update', async () => {
    const { renderer, invalidate, gl, map } = await makeRenderer()

    renderer.update(map, gl)
    await settle(renderer)

    // 2x2 region grid, all visible at a whole-world viewport.
    expect(seam(renderer).getRegionStates(gl)).toHaveLength(4)
    expect(invalidate).toHaveBeenCalled()
  })

  it('holds GPU uploads until the render seam asks for them', async () => {
    const { renderer, gl, map } = await makeRenderer()

    renderer.update(map, gl)
    await settle(renderer)
    // Fetch is complete, but nothing has been drawn yet.
    expect(gl.createTexture).not.toHaveBeenCalled()
    expect(gl.texImage2D).not.toHaveBeenCalled()

    const states = seam(renderer).getRegionStates(gl)
    expect(gl.createTexture).toHaveBeenCalledTimes(4)
    expect(gl.texImage2D).toHaveBeenCalledTimes(4)
    for (const state of states) {
      expect(state.texture).toBeTruthy()
      expect(state.vertexBuffer).toBeTruthy()
      expect(state.pixCoordBuffer).toBeTruthy()
      expect(state.mercatorBounds).toBeTruthy()
      expect(state.width).toBe(4)
      expect(state.height).toBe(2)
    }
  })

  it('reuses uploaded resources across frames', async () => {
    const { renderer, gl, map } = await makeRenderer()

    renderer.update(map, gl)
    await settle(renderer)
    seam(renderer).getRegionStates(gl)
    seam(renderer).getRegionStates(gl)
    seam(renderer).getRegionStates(gl)

    expect(gl.createTexture).toHaveBeenCalledTimes(4)
    expect(gl.texImage2D).toHaveBeenCalledTimes(4)
  })

  it('renders nothing before a level is committed', async () => {
    const { renderer, gl } = await makeRenderer()
    // No update() yet: regions were never requested.
    expect(seam(renderer).getRegionStates(gl)).toEqual([])
    expect(gl.createTexture).not.toHaveBeenCalled()
  })

  it('serves queries from the same level the renderer committed', async () => {
    const { renderer, gl, map } = await makeRenderer()
    renderer.update(map, gl)
    await settle(renderer)

    // Pixel (0, 0) of the 8x4 grid: lon -157.5, lat 67.5.
    const first = await renderer.queryData(
      { type: 'Point', coordinates: [-157.5, 67.5] },
      { time: 10 }
    )
    expect(first.temperature).toEqual([0])
    // Second time step of the same pixel.
    const second = await renderer.queryData(
      { type: 'Point', coordinates: [-157.5, 67.5] },
      { time: 20 }
    )
    expect(second.temperature).toEqual([32])
  })

  it('refetches and re-uploads every region after a selector change', async () => {
    const { renderer, gl, map } = await makeRenderer()
    renderer.update(map, gl)
    await settle(renderer)
    seam(renderer).getRegionStates(gl)
    expect(gl.texImage2D).toHaveBeenCalledTimes(4)
    const geometryUploads = gl.bufferData.mock.calls.length

    await renderer.setSelector({ time: { selected: 20, type: 'value' } })
    renderer.update(map, gl)
    await settle(renderer)

    const states = seam(renderer).getRegionStates(gl)
    expect(states).toHaveLength(4)
    // Re-uploaded in place: new data, same textures.
    expect(gl.texImage2D).toHaveBeenCalledTimes(8)
    expect(gl.createTexture).toHaveBeenCalledTimes(4)
    // The mesh doesn't depend on the selector, so it isn't re-sent.
    expect(gl.bufferData).toHaveBeenCalledTimes(geometryUploads)
  })

  it('sends regenerated geometry to the GPU on the next frame', async () => {
    const { renderer, gl, map } = await makeRenderer()
    renderer.update(map, gl)
    await settle(renderer)
    seam(renderer).getRegionStates(gl)
    const uploadsBefore = gl.bufferData.mock.calls.length

    // A refetch at new region dimensions rebuilds the mesh in place. Drawing
    // the region afterwards has to pick up the new arrays, or fresh data ends
    // up drawn against the previous mesh.
    const region = seam(renderer).regionCache.get('0:0,0')!
    seam(renderer).createRegionGeometry(0, 0, region)
    expect(region.geometryUploaded).toBe(false)

    seam(renderer).getRegionStates(gl)
    const uploaded = gl.bufferData.mock.calls
      .slice(uploadsBefore)
      .map((call) => call[1])
    expect(uploaded).toContain(region.vertexArr)
    expect(uploaded).toContain(region.pixCoordArr)
    expect(region.geometryUploaded).toBe(true)
  })

  it('recomputes mercator bounds with the regenerated geometry', async () => {
    const { renderer, gl, map } = await makeRenderer()
    renderer.update(map, gl)
    await settle(renderer)
    const region = seam(renderer).regionCache.get('0:0,0')!
    // A quarter of the world: the north-west region of a 2x2 grid.
    expect(region.mercatorBounds).toMatchObject({ x0: 0, x1: 0.5 })

    // Rebuild at a region size covering the whole level, as a refetch under a
    // changed region size would. Shader bounds have to follow the new mesh.
    region.levelMeta = {
      width: WIDTH,
      height: HEIGHT,
      regionSize: [HEIGHT, WIDTH],
    }
    seam(renderer).createRegionGeometry(0, 0, region)
    expect(region.mercatorBounds).toMatchObject({ x0: 0, x1: 1 })
  })

  it('keeps drawing fallback coverage when the current level cannot upload', async () => {
    const { renderer, map } = await makeRenderer()
    const gl = fakeGl({ failTextures: true })
    renderer.update(map, gl)
    await settle(renderer)
    const fallback = seedFallbackRegion(renderer)

    // The current level has data but its textures won't allocate. Dropping the
    // fallback here on CPU readiness alone would leave the viewport blank.
    const states = seam(renderer).getRegionStates(gl)
    expect(states).toHaveLength(1)
    expect(states[0].texture).toBe(fallback.texture)
  })

  it('uploads a fallback region that was fetched but never drawn', async () => {
    const { renderer, gl } = await makeRenderer()
    // Nothing fetched at the current level, so the fallback carries the frame.
    const fallback = seedFallbackRegion(renderer, { uploaded: false })

    const states = seam(renderer).getRegionStates(gl)
    expect(states).toHaveLength(1)
    expect(states[0].texture).toBe(fallback.texture)
    expect(fallback.texture).not.toBeNull()
    expect(gl.texImage2D).toHaveBeenCalledTimes(1)
  })

  it('draws nothing from a band-mode region until the shader switch refetches', async () => {
    const { renderer, gl, map } = await makeRenderer()
    renderer.update(map, gl)
    await settle(renderer)
    await renderer.setSelector({ time: { selected: [10, 20], type: 'value' } })
    renderer.setRendersFromBandTextures(true)
    renderer.update(map, gl)
    await settle(renderer)
    expect(seam(renderer).getRegionStates(gl)).toHaveLength(4)

    // Switching back to the main texture leaves cached regions holding band
    // data only. Drawing them through the main path would put one band's
    // values on screen as if they were the whole dataset.
    renderer.setRendersFromBandTextures(false)
    expect(seam(renderer).getRegionStates(gl)).toEqual([])

    // The refetch triggered by the switch restores them.
    renderer.update(map, gl)
    await settle(renderer)
    const states = seam(renderer).getRegionStates(gl)
    expect(states).toHaveLength(4)
    for (const state of states) {
      expect(state.texture).not.toBeNull()
    }
  })

  it('stops fetching once band-mode regions have their data', async () => {
    const { renderer, invalidate, gl, map } = await makeRenderer()
    renderer.update(map, gl)
    await settle(renderer)
    await renderer.setSelector({ time: { selected: [10, 20], type: 'value' } })
    renderer.setRendersFromBandTextures(true)
    renderer.update(map, gl)
    await settle(renderer)

    // Every fetch batch ends in invalidate(), so a settled viewport that keeps
    // invalidating is redispatching. Store reads can't see this: the decoded
    // chunk cache serves the refetches without touching the store.
    invalidate.mockClear()

    // Band-mode regions carry no interleaved copy, so a has-data check that
    // only looks at `data` queues every visible region on every frame.
    for (let frame = 0; frame < 3; frame++) {
      renderer.update(map, gl)
      await settle(renderer)
    }
    expect(invalidate).not.toHaveBeenCalled()
  })

  it('keeps fallback coverage when a band texture cannot be allocated', async () => {
    const { renderer, map } = await makeRenderer()
    const working = fakeGl()
    renderer.update(map, working)
    await settle(renderer)
    await renderer.setSelector({ time: { selected: [10, 20], type: 'value' } })
    renderer.setRendersFromBandTextures(true)
    renderer.update(map, working)
    await settle(renderer)

    const fallback = seedFallbackRegion(renderer, {
      bands: ['time_10', 'time_20'],
    })

    // Band textures are what this shader samples, so failing to allocate them
    // has to read as undrawable even though geometry uploads fine.
    const failing = fakeGl({ failTextures: true })
    const states = seam(renderer).getRegionStates(failing)
    expect(states).toHaveLength(1)
    expect(states[0].bandTextures).toBe(fallback.bandTextures)
  })

  it('holds other-level eviction protection until the level is drawable', async () => {
    const { renderer, gl, map } = await makeRenderer()
    renderer.update(map, gl)
    await settle(renderer)
    const fallback = seedFallbackRegion(renderer)
    const cache = seam(renderer).regionCache

    // Regions are fetched but never drawn, so the level can't be shown yet and
    // its fallback has to stay protected — eviction here is unrecoverable.
    renderer.update(map, gl)
    expect(cache.isProtected(fallback.key)).toBe(true)

    // Once a frame uploads them, the level stands on its own.
    seam(renderer).getRegionStates(gl)
    renderer.update(map, gl)
    expect(cache.isProtected(fallback.key)).toBe(false)
  })

  it('drops the interleaved copy once the shader samples band textures', async () => {
    const { renderer, gl, map } = await makeRenderer()
    // setSelector rebuilds the level only once a gl context has been cached,
    // so the first update has to land before the multi-value selector.
    renderer.update(map, gl)
    await settle(renderer)
    await renderer.setSelector({ time: { selected: [10, 20], type: 'value' } })
    renderer.update(map, gl)
    await settle(renderer)

    const region = seam(renderer).regionCache.get('0:0,0')!
    expect(region.channels).toBe(2)
    expect(region.bandData.size).toBe(2)
    expect(region.data).not.toBe(region.bandData.get('time_10'))

    // The layer switches to the band-sampling shader, which never reads the
    // main texture, so the interleaved copy stops being built.
    renderer.setRendersFromBandTextures(true)
    renderer.update(map, gl)
    await settle(renderer)

    const refetched = seam(renderer).regionCache.get('0:0,0')!
    expect(refetched.bandData.size).toBe(2)
    expect(refetched.data).toBeNull()

    // Rendering uploads the bands and no main texture at all, which is only
    // true if the renderer passes its band list down to the upload.
    const states = seam(renderer).getRegionStates(gl)
    expect(states).toHaveLength(4)
    for (const state of states) {
      expect(state.texture).toBeNull()
    }
    expect([...refetched.bandTexturesUploaded]).toEqual(['time_10', 'time_20'])
    // Two bands per region across the 2x2 grid, no main textures.
    expect(gl.createTexture).toHaveBeenCalledTimes(8)
  })

  it('disposes GPU resources and stops rendering after dispose', async () => {
    const { renderer, gl, map } = await makeRenderer()
    renderer.update(map, gl)
    await settle(renderer)
    seam(renderer).getRegionStates(gl)

    renderer.dispose(gl)
    expect(gl.deleteTexture).toHaveBeenCalledTimes(4)
    expect(seam(renderer).getRegionStates(gl)).toEqual([])
  })
})
