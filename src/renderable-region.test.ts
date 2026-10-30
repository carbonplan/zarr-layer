import { describe, it, expect } from 'vitest'
import { renderRegion, type RenderableRegion } from './renderable-region'
import { bandTextureKey } from './render-helpers'
import { createRecordingGl } from './__fixtures__/fake-gl'
import type { BandTransform } from './band-format'
import type { ShaderProgram } from './zarr-renderer'
import type { CustomShaderConfig } from './renderer-types'

/**
 * Raw integer bands are scaled and fill-checked in the shader, so each region
 * uploads its own transform right before it draws.
 */

const BAND_UNIFORMS = ['u_bandScale', 'u_bandOffset', 'u_bandFill', 'u_zl_nan']

// Locations are named stubs so the recording GL can report which uniform
// each upload went to; every other location is null.
const program = new Proxy(
  {
    useCustomShader: true,
    bandScaleLoc: { name: 'u_bandScale' },
    bandOffsetLoc: { name: 'u_bandOffset' },
    bandFillLoc: { name: 'u_bandFill' },
    nanLoc: { name: 'u_zl_nan' },
    clipBoxesLoc: { name: 'u_clip_boxes' },
    clipCountLoc: { name: 'u_clip_count' },
  } as Record<string, unknown>,
  { get: (target, key) => (key in target ? target[key as string] : null) }
) as unknown as ShaderProgram

const config: CustomShaderConfig = {
  bands: ['band_0'],
  bandFormat: 'int',
} as CustomShaderConfig

function region(bandTransform: BandTransform | null): RenderableRegion {
  return {
    mercatorBounds: { x0: 0, y0: 0, x1: 1, y1: 1 },
    meshBounds: { x0: 0, y0: 0, x1: 1, y1: 1 },
    vertexBuffer: {} as WebGLBuffer,
    pixCoordBuffer: {} as WebGLBuffer,
    indexBuffer: {} as WebGLBuffer,
    indexCount: 3,
    latIsAscending: false,
    texture: null,
    bandTexture: {} as WebGLTexture,
    bandTextureKey: bandTextureKey(['band_0'], 'int'),
    bandTransform,
  }
}

function bandUploads(gl: ReturnType<typeof createRecordingGl>) {
  return gl
    .callsTo('uniform1f')
    .filter((call) => BAND_UNIFORMS.includes(call.args[0] as string))
    .map((call) => call.args)
}

describe('renderRegion band transforms', () => {
  it('uploads each region its own scale, offset and fill', () => {
    const gl = createRecordingGl()
    renderRegion(
      gl,
      program,
      region({ scale: 2, offset: 10, fill: -128 }),
      [0],
      config
    )
    renderRegion(
      gl,
      program,
      region({ scale: 0.5, offset: 0, fill: 0 }),
      [0],
      config
    )
    expect(bandUploads(gl)).toEqual([
      ['u_bandScale', 2],
      ['u_bandOffset', 10],
      ['u_bandFill', -128],
      ['u_zl_nan', NaN],
      ['u_bandScale', 0.5],
      ['u_bandOffset', 0],
      ['u_bandFill', 0],
      ['u_zl_nan', NaN],
    ])
  })

  it('uploads a missing fill value as NaN, which matches nothing', () => {
    const gl = createRecordingGl()
    renderRegion(
      gl,
      program,
      region({ scale: 1, offset: 0, fill: null }),
      [0],
      config
    )
    const fill = bandUploads(gl).find(([name]) => name === 'u_bandFill')
    expect(fill?.[1]).toBeNaN()
  })

  it('uploads nothing for float bands', () => {
    const gl = createRecordingGl()
    const floatRegion = {
      ...region(null),
      bandTextureKey: bandTextureKey(['band_0'], 'float'),
    }
    renderRegion(gl, program, floatRegion, [0], {
      ...config,
      bandFormat: 'float',
    })
    expect(bandUploads(gl)).toEqual([])
  })
})

describe('renderRegion clip boxes', () => {
  it('uploads the tile boxes a region is clipped to', () => {
    // Each pixel shows exactly one level: a region keeps only the fragments
    // inside the tiles that show its level.
    const gl = createRecordingGl()
    const clipped = region(null)
    clipped.clipBoxes = new Float32Array([0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1])
    renderRegion(gl, program, clipped, [0], config)

    expect(
      gl.callsTo('uniform1i').find((c) => c.args[0] === 'u_clip_count')?.args
    ).toEqual(['u_clip_count', 2])
    expect(
      gl.callsTo('uniform4fv').find((c) => c.args[0] === 'u_clip_boxes')?.args
    ).toEqual(['u_clip_boxes', [0, 0, 0.5, 0.5, 0.5, 0.5, 1, 1]])
  })

  it('clears the count for a region drawn whole', () => {
    const gl = createRecordingGl()
    renderRegion(gl, program, region(null), [0], config)

    expect(
      gl.callsTo('uniform1i').find((c) => c.args[0] === 'u_clip_count')?.args
    ).toEqual(['u_clip_count', 0])
    expect(gl.callsTo('uniform4fv')).toHaveLength(0)
  })
})
