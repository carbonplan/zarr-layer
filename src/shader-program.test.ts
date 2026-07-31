import { describe, it, expect } from 'vitest'
import {
  applyProjectionUniforms,
  createShaderProgram,
  makeShaderVariantKey,
  resolveProjectionMode,
  type ShaderProgram,
} from './shader-program'
import { maplibreFragmentShaderSource } from './shaders'
import type { ProjectionMode } from './renderer-types'
import { createRecordingGl, FAKE_SHADER_DATA } from './__fixtures__/fake-gl'

/**
 * The map-settings decision table. Every visible difference between
 * mapbox/maplibre, mercator/globe, and flat/polar-preserving rendering is a
 * different ProjectionMode, and each mode has to reach the GPU as a specific
 * vertex shader with a specific set of uploaded uniforms. Getting the table
 * right but the shader or uniform routing wrong (or vice versa) shows up only
 * as a wrong-looking map, so all three layers are pinned here:
 *
 *   1. (useMapbox, useWgs84, useDirectEcef) -> ProjectionMode
 *   2. ProjectionMode                       -> the GLSL that gets compiled
 *   3. ProjectionMode                       -> which uniforms get uploaded
 */

const ALL_MODES: ProjectionMode[] = [
  'maplibre',
  'maplibre-proj4',
  'maplibre-ecef',
  'mapbox',
  'mapbox-proj4',
  'mapbox-ecef',
]

const buildProgram = (projectionMode: ProjectionMode) => {
  const gl = createRecordingGl()
  const { shaderProgram } = createShaderProgram(gl, {
    fragmentShaderSource: maplibreFragmentShaderSource,
    shaderData: FAKE_SHADER_DATA,
    projectionMode,
  })
  return { gl, shaderProgram, source: gl.sourcesFor(shaderProgram.program) }
}

describe('resolveProjectionMode', () => {
  it('maps every (provider, source-projection, globe) combination', () => {
    // useMapbox, useWgs84, useDirectEcef
    expect(resolveProjectionMode(false, false, false)).toBe('maplibre')
    expect(resolveProjectionMode(false, true, false)).toBe('maplibre-proj4')
    expect(resolveProjectionMode(false, false, true)).toBe('maplibre-ecef')
    expect(resolveProjectionMode(false, true, true)).toBe('maplibre-ecef')
    expect(resolveProjectionMode(true, false, false)).toBe('mapbox')
    expect(resolveProjectionMode(true, true, false)).toBe('mapbox-proj4')
    expect(resolveProjectionMode(true, false, true)).toBe('mapbox-ecef')
    expect(resolveProjectionMode(true, true, true)).toBe('mapbox-ecef')
  })

  it('defaults to the plain maplibre mode', () => {
    expect(resolveProjectionMode()).toBe('maplibre')
  })

  it('lets the provider flag win over the input space', () => {
    // A mapbox frame must never land on a maplibre mode: the maplibre variants
    // call projectTile() from a prelude Mapbox does not supply.
    expect(resolveProjectionMode(true, true, false).startsWith('mapbox')).toBe(
      true
    )
    expect(resolveProjectionMode(true, false, true).startsWith('mapbox')).toBe(
      true
    )
  })
})

describe('makeShaderVariantKey', () => {
  it('gives every projection mode its own cache entry', () => {
    const keys = ALL_MODES.map((projectionMode) =>
      makeShaderVariantKey({ projectionMode, shaderData: FAKE_SHADER_DATA })
    )
    expect(new Set(keys).size).toBe(ALL_MODES.length)
  })

  it('recompiles when MapLibre swaps its prelude mid-transition', () => {
    // MapLibre hands a different prelude (and variantName) across the
    // globe->mercator morph; reusing the cached program would keep projecting
    // with the old prelude's projectTile().
    const globe = makeShaderVariantKey({
      projectionMode: 'maplibre',
      shaderData: { ...FAKE_SHADER_DATA, variantName: 'globe' },
    })
    const mercator = makeShaderVariantKey({
      projectionMode: 'maplibre',
      shaderData: { ...FAKE_SHADER_DATA, variantName: 'mercator' },
    })
    expect(globe).not.toBe(mercator)
  })

  it('keys custom band shaders separately from the base shader', () => {
    const base = makeShaderVariantKey({ projectionMode: 'mapbox' })
    const banded = makeShaderVariantKey({
      projectionMode: 'mapbox',
      customShaderConfig: { bands: ['red', 'nir'] },
    })
    expect(banded).not.toBe(base)
    expect(banded).toContain('red_nir')
  })
})

describe('projection mode -> compiled vertex shader', () => {
  it('maplibre flat projects through the prelude’s projectTile', () => {
    const { source } = buildProgram('maplibre')
    expect(source.vertex).toContain('gl_Position = projectTile(merc);')
    expect(source.vertex).toContain('#define ZARR_FAKE 1')
  })

  it('maplibre source-projected uses the eye-coords flat output', () => {
    const { source } = buildProgram('maplibre-proj4')
    // Region-local mercator deltas plus a precomputed anchor: the high-zoom
    // jitter fix. Falling back to projectTile(merc) here would reintroduce it.
    expect(source.vertex).toContain('vec2 mercDelta')
    expect(source.vertex).toContain('gl_Position = u_anchor_clip + deltaClip;')
  })

  it('maplibre globe builds vertices on the sphere and blends to flat', () => {
    const { source } = buildProgram('maplibre-ecef')
    expect(source.vertex).toContain('vec3 ecef')
    expect(source.vertex).toContain('u_projection_matrix * vec4(ecef, 1.0)')
    // The flat fallback the globe morph blends toward.
    expect(source.vertex).toContain('u_projection_fallback_matrix')
    expect(source.vertex).toContain('u_projection_transition')
  })

  it('mapbox flat and globe share one shader driven by the transition', () => {
    const { source } = buildProgram('mapbox')
    // Mapbox drives mercator/globe from a single program: the mix() endpoint is
    // chosen by u_globe_transition, so a mercator map is transition=1 here.
    expect(source.vertex).toContain('u_globe_transition')
    expect(source.vertex).toContain('mix(globeClip, mercClip')
    expect(source.vertex).toContain('u_tile_render == 1')
    expect(source.vertex).not.toContain('projectTile')
  })

  it('mapbox globe uses the Y-DOWN ECEF sphere with a depth bias', () => {
    const { source } = buildProgram('mapbox-ecef')
    expect(source.vertex).toContain('-GLOBE_RADIUS * sin(latRad)')
    expect(source.vertex).toContain('matrix * (u_globe_to_merc * vec4(ecef')
    // Keeps the custom layer off the globe surface it z-fights with.
    expect(source.vertex).toContain('gl_Position.z -=')
  })

  it('only the ECEF variants carry WGS84 coords to the fragment shader', () => {
    for (const mode of ALL_MODES) {
      const { source } = buildProgram(mode)
      const carriesWgs84 = source.vertex.includes(
        'v_wgs84Pos = vec2(normLon, normLat);'
      )
      expect(carriesWgs84).toBe(mode.endsWith('ecef'))
    }
  })

  it('resolves eye-coords uniforms only on the source-projected flat variants', () => {
    // Other variants never reference them, so the compiler drops them and the
    // per-region uploads in renderRegion become no-ops.
    for (const mode of ALL_MODES) {
      const { shaderProgram } = buildProgram(mode)
      const usesEyeCoords = mode === 'maplibre-proj4' || mode === 'mapbox-proj4'
      expect(!!shaderProgram.eyeMatrixLoc).toBe(usesEyeCoords)
      expect(!!shaderProgram.anchorClipLoc).toBe(usesEyeCoords)
    }
  })

  it('gives mapbox modes the globe-to-mercator matrix and maplibre modes none', () => {
    for (const mode of ALL_MODES) {
      const { shaderProgram } = buildProgram(mode)
      const isMapbox = mode.startsWith('mapbox')
      expect(!!shaderProgram.globeToMercMatrixLoc).toBe(isMapbox)
      expect(!!shaderProgram.projMatrixLoc).toBe(!isMapbox)
    }
  })

  it('drops the globe blend uniforms on the direct mapbox globe path', () => {
    // mapbox-ecef only ever draws the fully-globe endpoint, so it has no
    // mercator blend and no draped-tile branch to switch on.
    const direct = buildProgram('mapbox-ecef').shaderProgram
    expect(direct.globeTransitionLoc).toBeNull()
    expect(direct.tileRenderLoc).toBeNull()

    // The draped modes carry both, since one program serves the whole morph.
    for (const mode of ['mapbox', 'mapbox-proj4'] as const) {
      const draped = buildProgram(mode).shaderProgram
      expect(draped.globeTransitionLoc).not.toBeNull()
      expect(draped.tileRenderLoc).not.toBeNull()
    }
  })
})

describe('applyProjectionUniforms', () => {
  const MATRIX = Array.from({ length: 16 }, (_, i) => i)
  const EXPANDED = new Float32Array(16).fill(9)
  const GLOBE_TO_MERC = new Float32Array(16).fill(3)

  const projectionData = {
    mainMatrix: MATRIX,
    fallbackMatrix: new Float32Array(16).fill(2),
    tileMercatorCoords: [0, 0, 1, 1] as [number, number, number, number],
    clippingPlane: [0, 0, 1, 0] as [number, number, number, number],
    projectionTransition: 0.6,
  }

  const mapbox = {
    projection: { name: 'globe' },
    globeToMercatorMatrix: GLOBE_TO_MERC,
    transition: 0.25,
    expandedFarZMercatorMatrix: EXPANDED,
  }

  const apply = (
    mode: ProjectionMode,
    options: {
      withProjectionData?: boolean
      withMapbox?: boolean
      isGlobeTileRender?: boolean
    } = {}
  ) => {
    const { gl, shaderProgram } = buildProgram(mode)
    gl.calls.length = 0
    applyProjectionUniforms(
      gl,
      shaderProgram,
      MATRIX,
      options.withProjectionData === false ? undefined : projectionData,
      options.withMapbox === false ? undefined : mapbox,
      options.isGlobeTileRender
    )
    return { gl, shaderProgram }
  }

  const uploadFor = (gl: ReturnType<typeof createRecordingGl>, name: string) =>
    gl.calls.find((call) => call.args[0] === name)

  it('routes maplibre modes to the projectTile uniforms', () => {
    for (const mode of [
      'maplibre',
      'maplibre-proj4',
      'maplibre-ecef',
    ] as const) {
      const { gl } = apply(mode)
      expect(uploadFor(gl, 'u_projection_matrix')?.args[1]).toEqual(
        new Float32Array(MATRIX)
      )
      expect(uploadFor(gl, 'u_projection_fallback_matrix')).toBeDefined()
      expect(uploadFor(gl, 'u_projection_transition')?.args[1]).toBe(0.6)
      // Mapbox's `matrix` does not exist in these programs; uploading it would
      // mean the provider detection upstream went wrong.
      expect(uploadFor(gl, 'matrix')).toBeUndefined()
      expect(uploadFor(gl, 'u_globe_to_merc')).toBeUndefined()
    }
  })

  it('uploads nothing for maplibre when projection data is absent', () => {
    // Legacy MapLibre without a prelude has no projection data; a partial
    // upload would leave stale matrices from the previous frame in place.
    const { gl } = apply('maplibre', { withProjectionData: false })
    expect(gl.calls).toHaveLength(0)
  })

  it('routes mapbox modes to the matrix and globe uniforms', () => {
    for (const mode of ['mapbox', 'mapbox-proj4'] as const) {
      const { gl } = apply(mode)
      expect(uploadFor(gl, 'matrix')?.args[1]).toEqual(new Float32Array(MATRIX))
      expect(uploadFor(gl, 'u_globe_to_merc')?.args[1]).toBe(GLOBE_TO_MERC)
      expect(uploadFor(gl, 'u_globe_transition')?.args[1]).toBe(0.25)
      expect(uploadFor(gl, 'u_projection_matrix')).toBeUndefined()
    }
  })

  it('defaults a missing mapbox transition to mercator', () => {
    const { gl } = apply('mapbox', { withMapbox: false })
    expect(uploadFor(gl, 'u_globe_transition')?.args[1]).toBe(1)
  })

  it('gives the direct globe path the expanded far-plane matrix', () => {
    // Mapbox's globe raster pass uses expandedFarZProjMatrix; the public
    // custom-layer matrix uses the regular far plane, so the ECEF path would
    // z-fight against the globe surface without this swap.
    const { gl } = apply('mapbox-ecef')
    expect(uploadFor(gl, 'matrix')?.args[1]).toBe(EXPANDED)
  })

  it('keeps the tile matrix when the direct globe path renders a tile', () => {
    const { gl } = apply('mapbox-ecef', { isGlobeTileRender: true })
    expect(uploadFor(gl, 'matrix')?.args[1]).toEqual(new Float32Array(MATRIX))
  })

  it('never swaps in the expanded matrix on the draped mapbox modes', () => {
    for (const mode of ['mapbox', 'mapbox-proj4'] as const) {
      const { gl } = apply(mode)
      expect(uploadFor(gl, 'matrix')?.args[1]).not.toBe(EXPANDED)
    }
  })

  it('flags tile rendering so the shader takes its flat branch', () => {
    // renderToTile draws into a flat tile texture that Mapbox then drapes; the
    // shader must not apply the globe projection a second time.
    const draped = apply('mapbox', { isGlobeTileRender: true })
    expect(uploadFor(draped.gl, 'u_tile_render')?.args[1]).toBe(1)

    const direct = apply('mapbox', { isGlobeTileRender: false })
    expect(uploadFor(direct.gl, 'u_tile_render')?.args[1]).toBe(0)
  })
})

describe('shader program uniform surface', () => {
  it('exposes the reprojection uniforms on every mode', () => {
    // Polar / EPSG:4326 datasets need the fragment-side latitude lookup no
    // matter which projection is on screen.
    for (const mode of ALL_MODES) {
      const { shaderProgram } = buildProgram(mode)
      const program: ShaderProgram = shaderProgram
      expect(program.reprojectLoc).not.toBeNull()
      expect(program.latBoundsLoc).not.toBeNull()
      expect(program.latIsAscendingLoc).not.toBeNull()
    }
  })
})
