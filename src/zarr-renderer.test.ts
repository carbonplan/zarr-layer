import { describe, it, expect } from 'vitest'
import { ZarrRenderer } from './zarr-renderer'
import { maplibreFragmentShaderSource } from './shaders'
import type { CustomShaderConfig, RendererUniforms } from './renderer-types'
import { createRecordingGl, FAKE_SHADER_DATA } from './__fixtures__/fake-gl'

const uniforms: RendererUniforms = {
  clim: [0, 1],
  opacity: 1,
  fillValue: null,
  scaleFactor: 1,
  offset: 0,
  fixedDataScale: 1,
}

const customFrag = 'fragColor = vec4(temp * gain * weights[0]) * opacity;'

function draw(
  gl: ReturnType<typeof createRecordingGl>,
  renderer: ZarrRenderer,
  config: CustomShaderConfig
) {
  const program = renderer.getProgram(FAKE_SHADER_DATA, config)
  renderer.applyCommonUniforms(program, {} as WebGLTexture, uniforms, config)
  return program
}

describe('custom uniforms', () => {
  it('uploads floats with uniform1f and arrays with uniform1fv', () => {
    const gl = createRecordingGl()
    const config: CustomShaderConfig = {
      bands: ['temp'],
      customFrag,
      customUniforms: { gain: 2, weights: new Float32Array([0.5, 0.25]) },
    }
    const renderer = new ZarrRenderer(gl, maplibreFragmentShaderSource, config)
    draw(gl, renderer, config)

    expect(gl.callsTo('uniform1f')).toContainEqual({
      name: 'uniform1f',
      args: ['gain', 2],
    })
    expect(gl.callsTo('uniform1fv')).toEqual([
      { name: 'uniform1fv', args: ['weights', [0.5, 0.25]] },
    ])
  })

  it('recompiles when an array changes length, not when its values change', () => {
    const gl = createRecordingGl()
    const config: CustomShaderConfig = {
      bands: ['temp'],
      customFrag,
      customUniforms: { gain: 1, weights: [1, 2] },
    }
    const renderer = new ZarrRenderer(gl, maplibreFragmentShaderSource, config)
    const first = draw(gl, renderer, config)

    config.customUniforms = { gain: 1, weights: [3, 4] }
    expect(draw(gl, renderer, config)).toBe(first)

    config.customUniforms = { gain: 1, weights: [1, 2, 3] }
    const resized = draw(gl, renderer, config)
    expect(resized).not.toBe(first)
    expect(gl.sourcesFor(resized.program).fragment).toContain(
      'uniform float weights[3];'
    )
  })

  it('rejects arrays that exceed the device uniform budget', () => {
    const gl = createRecordingGl({ maxFragmentUniformVectors: 16 })
    const config: CustomShaderConfig = {
      bands: ['temp'],
      customFrag,
      customUniforms: { gain: 1, weights: new Float32Array(64) },
    }
    const renderer = new ZarrRenderer(gl, maplibreFragmentShaderSource, config)
    expect(() => renderer.getProgram(FAKE_SHADER_DATA, config)).toThrow(
      'Uniform arrays need 64 fragment uniform vectors, but this device supports 16'
    )
  })
})
