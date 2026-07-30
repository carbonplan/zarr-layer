import { describe, it, expect, vi } from 'vitest'
import { RegionRenderer } from './region-renderer'
import { createRegionState, type RegionCache } from './region-cache'
import { ZarrStore } from './zarr-store'
import { buildMemoryZarrStore } from './__fixtures__/memory-zarr'
import { createRecordingGl, FAKE_SHADER_DATA } from './__fixtures__/fake-gl'
import {
  createShaderProgram,
  resolveProjectionMode,
  type ShaderProgram,
} from './shader-program'
import { maplibreFragmentShaderSource } from './shaders'
import type { RenderContext } from './renderer-types'
import type { ZarrRenderer } from './zarr-renderer'
import type { RegionState } from './region-state'

/**
 * Per-frame render-path selection: the branch in RegionRenderer.render() that
 * turns "which map is this, in which projection, drawn how" into a shader
 * variant, a world-offset list, and an eye-coords matrix.
 *
 * These are the settings that produce visual-only regressions — a globe that
 * renders flat, a mercator map that draws duplicate copies, terrain that loses
 * the layer — because nothing throws when the wrong variant is picked. The
 * inputs the branch reads are all in RenderContext plus onProjectionChange:
 *
 *   provider    context.mapbox present (Mapbox) or absent (MapLibre)
 *   globe       MapLibre: projectionData.projectionTransition > 0
 *               Mapbox:   onProjectionChange(true) + mapbox.directGlobePathActive
 *   terrain     Mapbox with terrain stays draped, so directGlobePathActive is
 *               false and the frame must not take the ECEF path
 *
 * The stub renderer compiles the variant it is asked for against the recording
 * GL, so uniform locations are the ones the real program would expose and the
 * assertions below see the uploads that actually reach the GPU.
 */

const MATRIX = Array.from({ length: 16 }, (_, i) => i + 1)
const MAIN_MATRIX = Array.from({ length: 16 }, (_, i) => 100 + i)

async function makeRenderer() {
  const memory = buildMemoryZarrStore({
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
  const store = new ZarrStore({
    customStore: memory,
    variable: 'temperature',
    version: 3,
    bounds: [-180, -90, 180, 90],
    latIsAscending: false,
  })
  await store.initialized

  const renderer = new RegionRenderer(store, 'temperature', {}, vi.fn())
  await renderer.initialize()
  return renderer
}

type Seam = { regionCache: RegionCache }

/**
 * A drawable region carrying a source-projected mesh. Seeded rather than
 * fetched: the path selection under test runs off the render context, not off
 * how the region's data arrived.
 */
function seedRegion(renderer: RegionRenderer): RegionState {
  const region = createRegionState(0, 0, 0, false, 0)
  region.data = new Float32Array([1, 2, 3, 4])
  region.width = 2
  region.height = 2
  region.vertexArr = new Float32Array(8)
  region.pixCoordArr = new Float32Array(8)
  region.indexArr = new Uint32Array([0, 1, 2])
  region.useIndexedMesh = true
  region.vertexCount = 3
  region.mercatorBounds = {
    x0: 0,
    y0: 0,
    x1: 1,
    y1: 1,
    latMin: -85,
    latMax: 85,
  }
  region.wgs84Bounds = { x0: 0, y0: 0, x1: 1, y1: 1 }
  region.levelMeta = { width: 2, height: 2, regionSize: [2, 2] }

  const cache = (renderer as unknown as Seam).regionCache
  cache.set(region.key, region)
  cache.rebuildProtection([region.key], { retainKeysNotMatching: '' })
  return region
}

/**
 * Stands in for ZarrRenderer, recording the variant selection while returning
 * a real program compiled for that variant.
 */
function stubRenderer(gl: ReturnType<typeof createRecordingGl>) {
  const requests: Array<{
    useMapbox: boolean
    useWgs84: boolean
    useDirectEcef: boolean
  }> = []
  let program: ShaderProgram | null = null

  const renderer = {
    gl,
    getProgram: (
      shaderData: unknown,
      customShaderConfig: unknown,
      useMapbox = false,
      useWgs84 = false,
      useDirectEcef = false
    ) => {
      requests.push({ useMapbox, useWgs84, useDirectEcef })
      const built = createShaderProgram(gl, {
        fragmentShaderSource: maplibreFragmentShaderSource,
        shaderData: FAKE_SHADER_DATA,
        projectionMode: resolveProjectionMode(
          useMapbox,
          useWgs84,
          useDirectEcef
        ),
      })
      program = built.shaderProgram
      return built.shaderProgram
    },
    applyCommonUniforms: vi.fn(),
  }

  return {
    renderer: renderer as unknown as ZarrRenderer,
    requests,
    mode: () => program?.projectionMode,
  }
}

const uniforms: RenderContext['uniforms'] = {
  clim: [0, 1],
  opacity: 1,
  fillValue: null,
  scaleFactor: 1,
  offset: 0,
  fixedDataScale: 1,
}

function maplibreContext(
  gl: WebGL2RenderingContext,
  projectionTransition: number,
  worldOffsets = [0]
): RenderContext {
  return {
    gl,
    matrix: MAIN_MATRIX,
    uniforms,
    colormapTexture: {} as WebGLTexture,
    worldOffsets,
    shaderData: FAKE_SHADER_DATA,
    projectionData: {
      mainMatrix: MAIN_MATRIX,
      fallbackMatrix: MATRIX,
      tileMercatorCoords: [0, 0, 1, 1],
      clippingPlane: [0, 0, 1, 0],
      projectionTransition,
    },
  }
}

function mapboxContext(
  gl: WebGL2RenderingContext,
  {
    globe = false,
    directGlobePathActive = false,
    worldOffsets = [0],
  }: {
    globe?: boolean
    directGlobePathActive?: boolean
    worldOffsets?: number[]
  } = {}
): RenderContext {
  return {
    gl,
    matrix: MATRIX,
    uniforms,
    colormapTexture: {} as WebGLTexture,
    worldOffsets,
    mapbox: {
      projection: { name: globe ? 'globe' : 'mercator' },
      globeToMercatorMatrix: new Float32Array(16),
      transition: globe ? 0 : 1,
      directGlobePathActive,
    },
  }
}

/** World offsets actually drawn, read back off the u_worldXOffset uploads. */
const drawnWorldOffsets = (gl: ReturnType<typeof createRecordingGl>) =>
  gl.calls
    .filter((call) => call.args[0] === 'u_worldXOffset')
    .map((call) => call.args[1])

const eyeMatrixUpload = (gl: ReturnType<typeof createRecordingGl>) =>
  gl.calls.find((call) => call.args[0] === 'u_eye_matrix')?.args[1]

async function frame(
  build: (gl: ReturnType<typeof createRecordingGl>) => RenderContext,
  { isGlobe = false }: { isGlobe?: boolean } = {}
) {
  const renderer = await makeRenderer()
  seedRegion(renderer)
  renderer.onProjectionChange(isGlobe)

  const gl = createRecordingGl()
  const stub = stubRenderer(gl)
  const context = build(gl)
  // Uploads happen on the first pass; clear so assertions see the draw frame.
  renderer.render(stub.renderer, context)
  gl.calls.length = 0
  renderer.render(stub.renderer, context)

  return { gl, mode: stub.mode(), requests: stub.requests }
}

describe('MapLibre render path', () => {
  it('uses the flat source-projected variant in mercator', async () => {
    const { mode } = await frame((gl) => maplibreContext(gl, 0))
    expect(mode).toBe('maplibre-proj4')
  })

  it('switches to ECEF as soon as the globe transition starts', async () => {
    // MapLibre morphs continuously; any nonzero transition means part of the
    // frame is on the sphere, and mercator-space vertices clip at the poles.
    const { mode } = await frame((gl) => maplibreContext(gl, 0.01))
    expect(mode).toBe('maplibre-ecef')
  })

  it('stays flat at transition exactly zero', async () => {
    const { mode } = await frame((gl) => maplibreContext(gl, 0))
    expect(mode).not.toBe('maplibre-ecef')
  })

  it('never asks for a mapbox variant', async () => {
    for (const transition of [0, 0.5, 1]) {
      const { requests } = await frame((gl) => maplibreContext(gl, transition))
      expect(requests.every((r) => !r.useMapbox)).toBe(true)
    }
  })

  it('draws every world copy in mercator', async () => {
    const { gl } = await frame((g) => maplibreContext(g, 0, [-1, 0, 1]))
    expect(drawnWorldOffsets(gl)).toEqual([-1, 0, 1])
  })

  it('collapses to one world copy on the globe', async () => {
    // isGlobeProjection can flip to false before the transition reaches 0, so
    // the context can still carry wrapped offsets while ECEF is active.
    // Drawing them would stack duplicate geometry on the sphere.
    const { gl } = await frame((g) => maplibreContext(g, 0.5, [-1, 0, 1]))
    expect(drawnWorldOffsets(gl)).toEqual([0])
  })

  it('anchors the flat path on the projection main matrix', async () => {
    // MapLibre's flat view-projection lives in projectionData.mainMatrix; the
    // per-region anchor is computed from it at full precision.
    const { gl } = await frame((g) => maplibreContext(g, 0))
    expect(eyeMatrixUpload(gl)).toEqual(new Float32Array(MAIN_MATRIX))
  })

  it('uploads no eye matrix on the globe', async () => {
    const { gl } = await frame((g) => maplibreContext(g, 0.5))
    expect(eyeMatrixUpload(gl)).toBeUndefined()
  })
})

describe('Mapbox render path', () => {
  it('uses the flat source-projected variant in mercator', async () => {
    const { mode } = await frame((gl) => mapboxContext(gl))
    expect(mode).toBe('mapbox-proj4')
  })

  it('takes the direct ECEF path when the layer has undraped itself', async () => {
    const { mode } = await frame(
      (gl) => mapboxContext(gl, { globe: true, directGlobePathActive: true }),
      { isGlobe: true }
    )
    expect(mode).toBe('mapbox-ecef')
  })

  it('stays draped on the globe while the layer is still draped', async () => {
    // This is the terrain case and the zoom-morph case: the layer keeps
    // renderToTile, so Mapbox drives the projection and the direct ECEF
    // program would fight it for depth.
    const { mode } = await frame(
      (gl) => mapboxContext(gl, { globe: true, directGlobePathActive: false }),
      { isGlobe: true }
    )
    expect(mode).toBe('mapbox-proj4')
  })

  it('ignores the direct-path flag when the map is not on the globe', async () => {
    const { mode } = await frame(
      (gl) => mapboxContext(gl, { globe: false, directGlobePathActive: true }),
      { isGlobe: false }
    )
    expect(mode).toBe('mapbox-proj4')
  })

  it('never asks for a maplibre variant', async () => {
    const { requests } = await frame((gl) => mapboxContext(gl))
    expect(requests.every((r) => r.useMapbox)).toBe(true)
  })

  it('anchors the flat path on the mapbox custom-layer matrix', async () => {
    // Mapbox passes its matrix directly rather than in a projection-data block.
    const { gl } = await frame((g) => mapboxContext(g))
    expect(eyeMatrixUpload(gl)).toEqual(new Float32Array(MATRIX))
  })

  it('collapses to one world copy on the direct globe path', async () => {
    const { gl } = await frame(
      (g) =>
        mapboxContext(g, {
          globe: true,
          directGlobePathActive: true,
          worldOffsets: [-1, 0, 1],
        }),
      { isGlobe: true }
    )
    expect(drawnWorldOffsets(gl)).toEqual([0])
  })

  it('keeps world copies while draped in mercator', async () => {
    const { gl } = await frame((g) =>
      mapboxContext(g, { worldOffsets: [0, 1] })
    )
    expect(drawnWorldOffsets(gl)).toEqual([0, 1])
  })
})

describe('per-region anchor across world copies', () => {
  it('recomputes the anchor for each world copy', async () => {
    // A wrapped copy has to anchor on its own origin; reusing the unwrapped
    // anchor makes the eye-coords sum cancel two near-world-sized clip values
    // and reintroduces high-zoom jitter on the wrapped side.
    const { gl } = await frame((g) => maplibreContext(g, 0, [0, 1]))
    const anchors = gl.calls
      .filter((call) => call.args[0] === 'u_anchor_clip')
      .map((call) => call.args.slice(1))
    expect(anchors).toHaveLength(2)
    expect(anchors[0]).not.toEqual(anchors[1])
  })
})
