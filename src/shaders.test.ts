import { describe, it, expect } from 'vitest'
import {
  createVertexShader,
  createFragmentShaderSource,
  maplibreFragmentShaderSource,
  type ShaderData,
} from './shaders'
import { latToMercatorNorm } from './map-utils'

/** CPU-side mercator inverse (normalized Y -> latitude in degrees). */
function mercatorNormToLat(mercY: number): number {
  const t = Math.PI * (1 - 2 * mercY)
  return (180 / Math.PI) * Math.atan(Math.sinh(t))
}

/**
 * Shader behavior is verified without a GPU in two complementary layers:
 *
 *  1. STRUCTURE — `createVertexShader`/`createFragmentShaderSource` compose
 *     GLSL from parts based on (inputSpace × projection) and band/customFrag
 *     options. We assert the right transform, uniforms, and projection output
 *     land in each variant, and that the customFrag uniform-extraction +
 *     gl_FragColor rewrite behave.
 *
 *  2. MATH PARITY — the projection math the shaders run on the GPU is mirrored
 *     here in JS (a faithful port of the exact GLSL lines) and proven equal to
 *     the CPU implementation used for tiling and queries (map-utils). If the
 *     CPU and GPU mercator math ever diverge, rendered rasters and queried
 *     values would disagree; this catches that. The substring assertions pin
 *     the GLSL formula so the JS mirror below can't silently drift from it.
 */

const FAKE_SHADER_DATA: ShaderData = {
  vertexShaderPrelude: '// MAPLIBRE_PRELUDE projectTile()',
  define: '#define ZARR_TEST 1',
  variantName: 'test',
}

describe('createVertexShader — structure', () => {
  it('emits a #version 300 es header', () => {
    const src = createVertexShader({
      inputSpace: 'mercator',
      projection: 'maplibre',
      shaderData: FAKE_SHADER_DATA,
    })
    expect(src.startsWith('#version 300 es')).toBe(true)
  })

  it('maplibre mercator uses projectTile and the maplibre prelude', () => {
    const src = createVertexShader({
      inputSpace: 'mercator',
      projection: 'maplibre',
      shaderData: FAKE_SHADER_DATA,
    })
    expect(src).toContain('gl_Position = projectTile(merc);')
    expect(src).toContain('MAPLIBRE_PRELUDE')
    expect(src).toContain('#define ZARR_TEST 1')
    expect(src).toContain('uniform float scale;')
  })

  it('wgs84 input space uses eye-coords mercator deltas', () => {
    const src = createVertexShader({
      inputSpace: 'wgs84',
      projection: 'mapbox',
    })
    // Vertices are region-local mercator deltas; the flat path projects them
    // through the eye-coords decomposition (u_anchor_clip + deltaClip).
    expect(src).toContain(
      'vec2 mercDelta = vec2(vertex.x * sx, vertex.y * sy);'
    )
    expect(src).toContain('u_eye_matrix * vec4(mercDelta, 0.0, 0.0)')
    expect(src).toContain('uniform mat4 matrix;') // mapbox globe uniforms
  })

  it('mapbox globe (non-direct) includes the mercator-Y->lat helper', () => {
    const src = createVertexShader({
      inputSpace: 'mercator',
      projection: 'mapbox',
    })
    expect(src).toContain('mercatorYToLatRad')
    expect(src).toContain('GLOBE_RADIUS')
  })

  it('wgs84-direct (maplibre) takes the ECEF path and carries wgs84 pos', () => {
    const src = createVertexShader({
      inputSpace: 'wgs84-direct',
      projection: 'maplibre',
      shaderData: FAKE_SHADER_DATA,
    })
    expect(src).toContain('u_projection_matrix')
    expect(src).toContain('v_wgs84Pos = vec2(normLon, normLat);')
  })

  it('wgs84-direct (mapbox) inverts mercator deltas onto the Y-DOWN ECEF sphere', () => {
    const src = createVertexShader({
      inputSpace: 'wgs84-direct',
      projection: 'mapbox',
    })
    expect(src).toContain('u_globe_to_merc')
    expect(src).toContain('mercatorYToLatRad')
    expect(src).toContain('-GLOBE_RADIUS * sin(latRad)')
  })

  it('throws when a maplibre variant is missing shaderData', () => {
    expect(() =>
      createVertexShader({ inputSpace: 'mercator', projection: 'maplibre' })
    ).toThrow(/shaderData required/)
    expect(() =>
      createVertexShader({ inputSpace: 'wgs84-direct', projection: 'maplibre' })
    ).toThrow(/shaderData required/)
  })
})

describe('createFragmentShaderSource — structure', () => {
  it('builds a single-band colormap shader', () => {
    const src = createFragmentShaderSource({ bands: ['temp'] })
    expect(src).toContain('uniform sampler2DArray u_zl_bands;')
    expect(src).toContain('texture(colormap, vec2(rescaled, 0.5))')
    expect(src).toContain('isnan(temp_tex)')
    expect(src).toContain('out vec4 fragColor;')
  })

  it('reads every band from one sampler, one layer per band', () => {
    const src = createFragmentShaderSource({ bands: ['a', 'b'] })
    expect(src.match(/uniform\s+sampler2DArray/g)).toHaveLength(1)
    expect(src).not.toMatch(/uniform\s+sampler2D\s+[ab];/)
    expect(src).toContain(
      'float a_tex = texture(u_zl_bands, vec3(sample_coord, 0.0)).r;'
    )
    expect(src).toContain(
      'float b_tex = texture(u_zl_bands, vec3(sample_coord, 1.0)).r;'
    )
  })

  it('samples integer bands through an integer sampler', () => {
    const src = createFragmentShaderSource({ bands: ['a'], bandFormat: 'int' })
    expect(src).toContain('precision highp isampler2DArray;')
    expect(src).toContain('uniform isampler2DArray u_zl_bands;')
    expect(src).toContain('float a_tex = zl_readBand(sample_coord, 0.0);')
    // Scale/offset are per region for raw integers.
    expect(src).toContain('float a_val = a_raw * u_bandScale + u_bandOffset;')
  })

  it('samples unsigned bands through an unsigned sampler', () => {
    const src = createFragmentShaderSource({ bands: ['a'], bandFormat: 'uint' })
    expect(src).toContain('uniform usampler2DArray u_zl_bands;')
  })

  it('turns the integer fill value into NaN for customFrag', () => {
    const src = createFragmentShaderSource({ bands: ['a'], bandFormat: 'int' })
    expect(src).toContain('return raw == u_bandFill ? u_zl_nan : raw;')
  })

  it('omits the integer read helper for float bands', () => {
    const src = createFragmentShaderSource({ bands: ['a'] })
    expect(src).not.toContain('zl_readBand')
    expect(src).not.toContain('u_bandScale')
  })

  it('declares a precision for the band array sampler', () => {
    // GLSL ES 3.00 gives sampler2DArray no default precision; omitting it
    // is a compile error.
    const src = createFragmentShaderSource({ bands: ['a'] })
    expect(src).toContain('precision highp sampler2DArray;')
  })

  it('hoists uniforms out of customFrag and rewrites gl_FragColor', () => {
    const src = createFragmentShaderSource({
      bands: ['temp'],
      customFrag:
        'uniform float gain;\ngl_FragColor = vec4(temp * gain, 0.0, 0.0, 1.0);',
    })
    expect(src).toContain('uniform float gain;')
    expect(src).toContain('fragColor = vec4(temp * gain')
    expect(src).not.toContain('gl_FragColor')
  })

  it('declares explicitly-listed customUniforms', () => {
    const src = createFragmentShaderSource({
      bands: ['temp'],
      customUniforms: { gain: 1 },
      customFrag: 'gl_FragColor = vec4(temp * gain, 0.0, 0.0, 1.0);',
    })
    expect(src).toContain('uniform float gain;')
    expect(src).not.toContain('gl_FragColor')
  })

  it('declares array uniforms at the length of their value', () => {
    const src = createFragmentShaderSource({
      bands: ['temp'],
      customUniforms: { weights: [1, 2, 3], query: new Float32Array(64) },
      customFrag: 'fragColor = vec4(temp * weights[0] * query[63]);',
    })
    expect(src).toContain('uniform float weights[3];')
    expect(src).toContain('uniform float query[64];')
  })

  it('hoists array uniforms declared in customFrag out of main', () => {
    const src = createFragmentShaderSource({
      bands: ['temp'],
      customFrag: 'uniform float w[4];\nfragColor = vec4(temp * w[0]);',
    })
    const main = src.slice(src.indexOf('void main()'))
    expect(src).toContain('uniform float w[4];')
    expect(main).not.toContain('uniform')
  })

  it('ignores declarations inside comments', () => {
    const src = createFragmentShaderSource({
      bands: ['temp'],
      customFrag: [
        '// uniform float opacity[2];',
        '/* uniform float w[4];',
        '   uniform float gain; */',
        'fragColor = vec4(temp);',
      ].join('\n'),
    })
    expect(src).not.toContain('uniform float opacity[2];')
    expect(src).not.toContain('uniform float w[4];')
    expect(src).not.toContain('uniform float gain;')
  })

  it('keeps code that comments sit between or around', () => {
    const src = createFragmentShaderSource({
      bands: ['temp'],
      customFrag: [
        'float/* units */gain = 2.0;',
        '// /*',
        'float offset = 1.0;',
        '// */',
        'fragColor = vec4(temp * gain + offset);',
      ].join('\n'),
    })
    expect(src).toMatch(/float\s+gain = 2\.0;/)
    expect(src).toContain('float offset = 1.0;')
  })

  it('drops a customFrag declaration of a uniform the layer declares', () => {
    const src = createFragmentShaderSource({
      bands: ['temp'],
      customUniforms: { w: [1, 2, 3, 4] },
      customFrag: 'uniform float w[4];\nfragColor = vec4(temp * w[0]);',
    })
    expect(src.match(/uniform float w\[4\];/g)).toHaveLength(1)
  })
})

describe('mercator math parity (GPU shader <-> CPU map-utils)', () => {
  // The mercator FORWARD runs on the CPU (mesh encoding via latToMercatorNorm);
  // the shaders only INVERT (fragment texture lookup, ECEF vertex paths). If
  // the GPU inverse and CPU forward disagree, texture lookups shift.

  // Faithful JS port of the fragment shader's mercator inverse
  // (FRAGMENT_SHADER_REPROJECT + FUNC_MERCATOR_INVERT in shaders.ts).
  function shaderInvertLatDeg(mercYNorm: number): number {
    const PI = Math.PI
    const y = PI * (1 - 2 * mercYNorm)
    const phi = 2 * Math.atan(Math.exp(y)) - PI / 2
    return (180 / PI) * phi
  }

  it('fragment-shader inverse matches mercatorNormToLat', () => {
    for (const mercY of [0.05, 0.25, 0.5, 0.75, 0.95]) {
      expect(shaderInvertLatDeg(mercY)).toBeCloseTo(
        mercatorNormToLat(mercY),
        10
      )
    }
  })

  it('GPU inverse inverts the CPU forward (latToMercatorNorm)', () => {
    for (const lat of [-80, -30, 0, 30, 80]) {
      expect(shaderInvertLatDeg(latToMercatorNorm(lat))).toBeCloseTo(lat, 9)
    }
  })

  it('pins the GLSL formulas the JS port mirrors', () => {
    // If these substrings change, the JS ports above must be updated in lockstep.
    const vtx = createVertexShader({
      inputSpace: 'wgs84-direct',
      projection: 'mapbox',
    })
    // Vertex-side inverse (FUNC_MERCATOR_Y_TO_LAT, used by the ECEF paths).
    expect(vtx).toContain('float t = PI * (1.0 - 2.0 * y);')
    expect(vtx).toContain('atan(sinh(t))')
    // Fragment-side inverse (FUNC_MERCATOR_INVERT).
    expect(maplibreFragmentShaderSource).toContain(
      '2.0 * atan(exp(y)) - PI / 2.0'
    )
    expect(maplibreFragmentShaderSource).toContain(
      'PI * (1.0 - 2.0 * v_mercatorPos.y)'
    )
  })
})
