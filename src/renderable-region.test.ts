import { describe, it, expect } from 'vitest'
import {
  latLookupAnchor,
  renderRegion,
  type RenderableRegion,
} from './renderable-region'
import { latToMercatorNorm } from './map-utils'
import { bandTextureKey } from './render-helpers'
import { createRecordingGl } from './__fixtures__/fake-gl'
import type { ValueTransform } from './value-transform'
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
    latLookupLoc: { name: 'u_latLookup' },
    latAnchorLoc: { name: 'u_latAnchor' },
    latShiftLoc: { name: 'u_latShift' },
  } as Record<string, unknown>,
  { get: (target, key) => (key in target ? target[key as string] : null) }
) as unknown as ShaderProgram

const config: CustomShaderConfig = {
  bands: ['band_0'],
  bandFormat: 'int',
} as CustomShaderConfig

function region(bandTransform: ValueTransform | null): RenderableRegion {
  return {
    mercatorBounds: { x0: 0, y0: 0, x1: 1, y1: 1 },
    meshBounds: { x0: 0, y0: 0, x1: 1, y1: 1 },
    vertexBuffer: {} as WebGLBuffer,
    pixCoordBuffer: {} as WebGLBuffer,
    indexBuffer: {} as WebGLBuffer,
    indexCount: 3,
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

describe('renderRegion latitude lookup', () => {
  const uniform1i = (gl: ReturnType<typeof createRecordingGl>, name: string) =>
    gl.callsTo('uniform1i').find((c) => c.args[0] === name)?.args[1]

  it('enables the lookup with the region-anchored constants', () => {
    const gl = createRecordingGl()
    const r = {
      ...region(null),
      meshBounds: { x0: 0.5, y0: 0.3, x1: 0.6, y1: 0.4 },
      texLatEdges: [21.9, 72.6] as [number, number],
    }
    renderRegion(gl, program, r, [0], config)

    expect(uniform1i(gl, 'u_latLookup')).toBe(1)
    const anchor = gl
      .callsTo('uniform4f')
      .find((c) => c.args[0] === 'u_latAnchor')?.args
    expect(anchor?.slice(1)).toEqual(latLookupAnchor(0.35, [21.9, 72.6]))
  })

  it('samples interpolated texture coordinates without lat edges', () => {
    const gl = createRecordingGl()
    renderRegion(gl, program, region(null), [0], config)

    expect(uniform1i(gl, 'u_latLookup')).toBe(0)
    expect(
      gl.callsTo('uniform4f').find((c) => c.args[0] === 'u_latAnchor')
    ).toBeUndefined()
  })
})

describe('latLookupAnchor', () => {
  // JS port of FRAGMENT_SAMPLE_COORD's lookup branch, run in Float32 so the
  // test sees the precision the GPU does.
  const f = Math.fround
  function sinhSmall(x: number): number {
    if (Math.abs(x) < 0.5) {
      const x2 = f(x * x)
      return f(x * f(1 + f(x2 * f(1 / 6 + f(x2 * f(1 / 120 + f(x2 / 5040)))))))
    }
    return f(Math.sinh(x))
  }
  function shaderV(
    clipY: number,
    [s0, c0, vAnchor, vPerRad]: [number, number, number, number],
    latShift = 0
  ): number {
    const dy = f(-2 * f(Math.PI) * f(f(clipY) - f(latShift)))
    const sh = sinhSmall(f(0.5 * dy))
    const dTan = f(f(2 * f(s0) * sh * sh) + f(f(c0) * sinhSmall(dy)))
    const dLat = f(Math.atan2(dTan, f(1 + f(f(s0) * f(f(s0) + dTan)))))
    return f(f(vAnchor) + f(dLat * f(vPerRad)))
  }

  const edges: [number, number] = [21.9, 72.6]
  const vOf = (lat: number) => (lat - edges[0]) / (edges[1] - edges[0])
  const anchorY = (latToMercatorNorm(21.9) + latToMercatorNorm(72.6)) / 2
  const anchor = latLookupAnchor(anchorY, edges)

  it('maps each latitude to its texture row', () => {
    for (const lat of [21.9, 30, 46.7, 46.8, 60, 72.5, 72.6]) {
      const v = shaderV(latToMercatorNorm(lat) - anchorY, anchor)
      // 507 rows: a thousandth of a row
      expect(Math.abs(v - vOf(lat)) * 507).toBeLessThan(1e-3)
    }
  })

  it('stays within a Float32 texture coordinate on a small region', () => {
    // A 0.5° region of 512 rows: an absolute Float32 Mercator Y would be off
    // by ~0.01 rows here, the region-local offset by Float32 v rounding only.
    const small: [number, number] = [46.5, 47]
    const y0 = (latToMercatorNorm(46.5) + latToMercatorNorm(47)) / 2
    const a = latLookupAnchor(y0, small)
    for (let lat = 46.5; lat <= 47; lat += 0.01) {
      const v = shaderV(latToMercatorNorm(lat) - y0, a)
      expect(Math.abs(v - (lat - 46.5) / 0.5) * 512).toBeLessThan(1e-4)
    }
  })

  it('follows descending rows', () => {
    const descending: [number, number] = [72.6, 21.9]
    const a = latLookupAnchor(anchorY, descending)
    const v = shaderV(latToMercatorNorm(46.7) - anchorY, a)
    expect(v).toBeCloseTo((72.6 - 46.7) / 50.7, 5)
  })

  it('maps rows of a region whose mesh reaches past the Mercator world', () => {
    // The top row of a global pyramid: its mesh runs nearly to the pole.
    const edges: [number, number] = [90, 67.5]
    const meshBounds = { x0: 0.5, y0: -2.45342, x1: 0.625, y1: 0.24298 }
    const gl = createRecordingGl()
    renderRegion(
      gl,
      program,
      { ...region(null), meshBounds, texLatEdges: edges },
      [0],
      config
    )
    const a = gl
      .callsTo('uniform4f')
      .find((c) => c.args[0] === 'u_latAnchor')!
      .args.slice(1) as [number, number, number, number]
    const latShift = Number(
      gl.callsTo('uniform1f').find((c) => c.args[0] === 'u_latShift')
        ?.args[1] ?? 0
    )
    const meshY = (meshBounds.y0 + meshBounds.y1) / 2
    for (let lat = 67.5; lat <= 85; lat += 0.5) {
      const v = shaderV(latToMercatorNorm(lat) - meshY, a, latShift)
      expect(Math.abs(v - (90 - lat) / 22.5) * 128).toBeLessThan(1e-3)
    }
  })

  it.each([
    ['crossing the equator', -30, 30, 720],
    ['reaching the Mercator limit', 60, 85, 300],
    ['in the far south', -85, -60, 300],
    ['spanning the Mercator world', -85, 85, 1700],
  ])('maps rows of a region %s', (_, south, north, rows) => {
    const y0 = (latToMercatorNorm(south) + latToMercatorNorm(north)) / 2
    for (const edges of [
      [south, north],
      [north, south],
    ] as [number, number][]) {
      const a = latLookupAnchor(y0, edges)
      for (let i = 0; i <= 20; i++) {
        const lat = south + ((north - south) * i) / 20
        const v = shaderV(latToMercatorNorm(lat) - y0, a)
        const expected = (lat - edges[0]) / (edges[1] - edges[0])
        expect(Math.abs(v - expected) * rows).toBeLessThan(1e-3)
      }
    }
  })
})
