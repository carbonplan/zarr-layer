import { describe, it, expect, vi } from 'vitest'
import {
  bandTextureKey,
  bindBandTexture,
  ensureRegionGpuResources,
} from './render-helpers'
import { createRegionState } from './region-cache'
import type { RegionState } from './region-state'

/**
 * The lazy GPU upload contract: fetch leaves only CPU state; the render paths
 * call ensureRegionGpuResources per frame to create and upload texture and
 * geometry buffers on the drawing context, re-uploading when the data was
 * refreshed (textureUploaded reset by fetch) and doing nothing when complete.
 * Custom-shader bands are uploaded the same way, into one texture array.
 */

// Texture unit constants matter: the band array binds to unit 2.
const TEXTURE0 = 0x84c0
const TEXTURE_2D_ARRAY = 0x8c1a
const R32F = 0x822e
const R8I = 0x8231
const R16UI = 0x8234
const RED_INTEGER = 0x8d94
const BYTE = 0x1400
const UNSIGNED_SHORT = 0x1403
const UNPACK_ALIGNMENT = 0x0cf5

const MAX_ARRAY_TEXTURE_LAYERS = 0x88ff
const MAX_TEXTURE_SIZE = 0x0d33

function fakeGl({
  failTextures = false,
  maxLayers = 256,
}: { failTextures?: boolean; maxLayers?: number } = {}) {
  let textureCount = 0
  let bufferCount = 0
  return {
    TEXTURE0,
    TEXTURE_2D: 0x0de1,
    TEXTURE_2D_ARRAY,
    R32F,
    RED: 0x1903,
    FLOAT: 0x1406,
    R8I,
    R8UI: 0x8232,
    R16I: 0x8233,
    R16UI,
    RED_INTEGER,
    BYTE,
    UNSIGNED_BYTE: 0x1401,
    SHORT: 0x1402,
    UNSIGNED_SHORT,
    UNPACK_ALIGNMENT,
    MAX_ARRAY_TEXTURE_LAYERS,
    MAX_TEXTURE_SIZE,
    getParameter: vi.fn((pname: number) =>
      pname === MAX_ARRAY_TEXTURE_LAYERS
        ? maxLayers
        : pname === MAX_TEXTURE_SIZE
        ? 4096
        : pname === UNPACK_ALIGNMENT
        ? 4
        : null
    ),
    pixelStorei: vi.fn(),
    createTexture: vi.fn(() => (failTextures ? null : { tex: ++textureCount })),
    createBuffer: vi.fn(() => ({ buf: ++bufferCount })),
    deleteTexture: vi.fn(),
    bindTexture: vi.fn(),
    bindBuffer: vi.fn(),
    bufferData: vi.fn(),
    texImage2D: vi.fn(),
    texImage3D: vi.fn(),
    texParameteri: vi.fn(),
    activeTexture: vi.fn(),
  } as unknown as WebGL2RenderingContext & {
    createTexture: ReturnType<typeof vi.fn>
    createBuffer: ReturnType<typeof vi.fn>
    deleteTexture: ReturnType<typeof vi.fn>
    bufferData: ReturnType<typeof vi.fn>
    texImage2D: ReturnType<typeof vi.fn>
    texImage3D: ReturnType<typeof vi.fn>
    pixelStorei: ReturnType<typeof vi.fn>
    bindTexture: ReturnType<typeof vi.fn>
    texParameteri: ReturnType<typeof vi.fn>
    activeTexture: ReturnType<typeof vi.fn>
  }
}

function fetchedRegion(): RegionState {
  const region = createRegionState(0, 0, 0, false, 0)
  region.data = new Float32Array([1, 2, 3, 4])
  region.width = 2
  region.height = 2
  region.channels = 1
  region.vertexArr = new Float32Array(8)
  region.pixCoordArr = new Float32Array(8)
  region.indexArr = new Uint32Array([0, 1, 2])
  region.indexCount = 3
  return region
}

describe('ensureRegionGpuResources', () => {
  it('returns false while CPU-side state is incomplete', () => {
    const gl = fakeGl()
    const region = createRegionState(0, 0, 0, false, 0)
    expect(ensureRegionGpuResources(gl, region)).toBe(false)
    expect(gl.createTexture).not.toHaveBeenCalled()
    expect(gl.createBuffer).not.toHaveBeenCalled()
  })

  it('creates and uploads texture and buffers on first call', () => {
    const gl = fakeGl()
    const region = fetchedRegion()

    expect(ensureRegionGpuResources(gl, region)).toBe(true)
    expect(region.texture).not.toBeNull()
    expect(region.textureUploaded).toBe(true)
    expect(region.vertexBuffer).not.toBeNull()
    expect(region.pixCoordBuffer).not.toBeNull()
    expect(region.indexBuffer).not.toBeNull()
    expect(gl.createTexture).toHaveBeenCalledTimes(1)
    expect(gl.createBuffer).toHaveBeenCalledTimes(3)
    expect(gl.texImage2D).toHaveBeenCalledTimes(1)
    expect(gl.bufferData).toHaveBeenCalledTimes(3)
  })

  it('is idempotent once resources exist', () => {
    const gl = fakeGl()
    const region = fetchedRegion()
    ensureRegionGpuResources(gl, region)

    expect(ensureRegionGpuResources(gl, region)).toBe(true)
    expect(gl.createTexture).toHaveBeenCalledTimes(1)
    expect(gl.createBuffer).toHaveBeenCalledTimes(3)
    expect(gl.texImage2D).toHaveBeenCalledTimes(1)
    expect(gl.bufferData).toHaveBeenCalledTimes(3)
  })

  it('re-uploads the texture after a data refresh without recreating it', () => {
    const gl = fakeGl()
    const region = fetchedRegion()
    ensureRegionGpuResources(gl, region)

    // Fetch wrote new data and reset the flag (selector change refetch).
    region.textureUploaded = false
    expect(ensureRegionGpuResources(gl, region)).toBe(true)
    expect(gl.createTexture).toHaveBeenCalledTimes(1)
    expect(gl.texImage2D).toHaveBeenCalledTimes(2)
    // Geometry is untouched by a data-only refresh.
    expect(gl.bufferData).toHaveBeenCalledTimes(3)
  })

  it('re-uploads regenerated geometry into the existing buffers', () => {
    const gl = fakeGl()
    const region = fetchedRegion()
    ensureRegionGpuResources(gl, region)
    const buffers = [
      region.vertexBuffer,
      region.pixCoordBuffer,
      region.indexBuffer,
    ]

    // A refetch at new dimensions regenerates the mesh: createRegionGeometry
    // replaces the arrays and clears the flag. Without the re-upload the GPU
    // would keep the old mesh and draw the new data misaligned against it.
    const vertexArr = new Float32Array([9, 9, 9, 9, 9, 9, 9, 9])
    const pixCoordArr = new Float32Array([8, 8, 8, 8, 8, 8, 8, 8])
    const indexArr = new Uint32Array([2, 1, 0])
    region.vertexArr = vertexArr
    region.pixCoordArr = pixCoordArr
    region.indexArr = indexArr
    region.geometryUploaded = false

    expect(ensureRegionGpuResources(gl, region)).toBe(true)
    expect(gl.bufferData).toHaveBeenCalledTimes(6)
    expect(gl.bufferData.mock.calls.slice(3).map((call) => call[1])).toEqual([
      vertexArr,
      pixCoordArr,
      indexArr,
    ])
    // Reused, not reallocated.
    expect(gl.createBuffer).toHaveBeenCalledTimes(3)
    expect([
      region.vertexBuffer,
      region.pixCoordBuffer,
      region.indexBuffer,
    ]).toEqual(buffers)
    expect(region.geometryUploaded).toBe(true)
  })

  it('leaves clean geometry alone across frames', () => {
    const gl = fakeGl()
    const region = fetchedRegion()
    ensureRegionGpuResources(gl, region)
    ensureRegionGpuResources(gl, region)
    ensureRegionGpuResources(gl, region)
    expect(gl.bufferData).toHaveBeenCalledTimes(3)
  })

  // Every region draws with gl.drawElements, so an index buffer is as
  // load-bearing as the vertex buffer: without one there is nothing to draw.
  it('holds a region undrawable until it has indices', () => {
    const gl = fakeGl()
    const region = fetchedRegion()
    region.indexArr = null

    expect(ensureRegionGpuResources(gl, region)).toBe(false)
    expect(region.indexBuffer).toBeNull()
    expect(region.geometryUploaded).toBe(false)
  })

  it('uploads an index buffer alongside the vertex buffers', () => {
    const gl = fakeGl()
    const region = fetchedRegion()

    expect(ensureRegionGpuResources(gl, region)).toBe(true)
    expect(region.indexBuffer).not.toBeNull()
    expect(gl.createBuffer).toHaveBeenCalledTimes(3)
  })
})

describe('ensureRegionGpuResources with band rendering', () => {
  function bandRegion(): RegionState {
    const region = fetchedRegion()
    region.bandData.set('red', new Float32Array([1, 2, 3, 4]))
    region.bandData.set('green', new Float32Array([5, 6, 7, 8]))
    return region
  }

  it('uploads every sampled band into one texture array, one band per layer', () => {
    const gl = fakeGl()
    const region = bandRegion()

    expect(ensureRegionGpuResources(gl, region, ['red', 'green'])).toBe(true)
    expect(gl.createTexture).toHaveBeenCalledTimes(1)
    expect(gl.texImage3D).toHaveBeenCalledTimes(1)
    const [target, , internalFormat, width, height, depth, , , , data] =
      gl.texImage3D.mock.calls[0]
    expect(target).toBe(TEXTURE_2D_ARRAY)
    expect(internalFormat).toBe(R32F)
    expect([width, height, depth]).toEqual([2, 2, 2])
    expect([...data]).toEqual([1, 2, 3, 4, 5, 6, 7, 8])
    expect(region.bandTextureKey).toBe(bandTextureKey(['red', 'green']))
    // Nothing samples the main texture in this mode.
    expect(region.texture).toBeNull()
    expect(region.textureUploaded).toBe(false)
  })

  it('layers bands in shader order, not fetch order', () => {
    const gl = fakeGl()
    const region = bandRegion()

    ensureRegionGpuResources(gl, region, ['green', 'red'])
    expect([...gl.texImage3D.mock.calls[0][9]]).toEqual([
      5, 6, 7, 8, 1, 2, 3, 4,
    ])
  })

  it('costs one texture regardless of band count', () => {
    const gl = fakeGl()
    const region = fetchedRegion()
    const bands = Array.from({ length: 64 }, (_, i) => `band_${i}`)
    for (const band of bands) region.bandData.set(band, new Float32Array(4))

    expect(ensureRegionGpuResources(gl, region, bands)).toBe(true)
    expect(gl.createTexture).toHaveBeenCalledTimes(1)
    expect(gl.texImage3D.mock.calls[0][5]).toBe(64)
  })

  it('refuses more bands than the texture array limit, once per context', () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    try {
      const gl = fakeGl({ maxLayers: 4 })
      const region = fetchedRegion()
      const bands = Array.from({ length: 5 }, (_, i) => `band_${i}`)
      for (const band of bands) region.bandData.set(band, new Float32Array(4))

      expect(ensureRegionGpuResources(gl, region, bands)).toBe(false)
      expect(ensureRegionGpuResources(gl, region, bands)).toBe(false)
      expect(gl.texImage3D).not.toHaveBeenCalled()
      expect(region.bandTextureKey).toBeNull()
      expect(errorSpy).toHaveBeenCalledTimes(1)
    } finally {
      errorSpy.mockRestore()
    }
  })

  it('creates no texture while a sampled band is missing', () => {
    const gl = fakeGl()
    const region = bandRegion()
    expect(ensureRegionGpuResources(gl, region, ['red', 'blue'])).toBe(false)
    expect(gl.createTexture).not.toHaveBeenCalled()
    expect(gl.texImage3D).not.toHaveBeenCalled()
  })

  it('is idempotent once the band texture is resident', () => {
    const gl = fakeGl()
    const region = bandRegion()
    ensureRegionGpuResources(gl, region, ['red', 'green'])

    expect(ensureRegionGpuResources(gl, region, ['red', 'green'])).toBe(true)
    expect(gl.texImage3D).toHaveBeenCalledTimes(1)
  })

  it('re-uploads into the same texture after a data refresh', () => {
    const gl = fakeGl()
    const region = bandRegion()
    ensureRegionGpuResources(gl, region, ['red', 'green'])
    const texture = region.bandTexture

    // Fetch wrote new data and cleared the key (selector change refetch).
    region.bandTextureKey = null
    expect(ensureRegionGpuResources(gl, region, ['red', 'green'])).toBe(true)
    expect(gl.texImage3D).toHaveBeenCalledTimes(2)
    expect(gl.createTexture).toHaveBeenCalledTimes(1)
    expect(region.bandTexture).toBe(texture)
  })

  it('re-uploads when the band list changes at the same size', () => {
    const gl = fakeGl()
    const region = bandRegion()
    ensureRegionGpuResources(gl, region, ['red', 'green'])

    region.bandData.clear()
    region.bandData.set('nir', new Float32Array([1, 2, 3, 4]))
    region.bandData.set('swir', new Float32Array([5, 6, 7, 8]))
    expect(ensureRegionGpuResources(gl, region, ['nir', 'swir'])).toBe(true)
    expect(gl.texImage3D).toHaveBeenCalledTimes(2)
    expect(region.bandTextureKey).toBe(bandTextureKey(['nir', 'swir']))
  })

  it('reports not-ready when the band texture cannot be allocated', () => {
    const gl = fakeGl({ failTextures: true })
    const region = bandRegion()

    // Coverage is decided by this result, so a failed allocation has to read
    // as undrawable rather than as a covered level.
    expect(ensureRegionGpuResources(gl, region, ['red', 'green'])).toBe(false)
  })

  it('reports not-ready and uploads nothing when a required band has no data', () => {
    const gl = fakeGl()
    const region = bandRegion()
    expect(ensureRegionGpuResources(gl, region, ['red', 'missing'])).toBe(false)
    expect(gl.texImage3D).not.toHaveBeenCalled()
    expect(region.bandTextureKey).toBeNull()
  })

  it('releases the band texture when switching back to the main texture', () => {
    const gl = fakeGl()
    const region = bandRegion()
    ensureRegionGpuResources(gl, region, ['red', 'green'])
    const released = region.bandTexture

    expect(ensureRegionGpuResources(gl, region)).toBe(true)
    expect(gl.deleteTexture).toHaveBeenCalledWith(released)
    expect(region.bandTexture).toBeNull()
    expect(region.bandTextureKey).toBeNull()
    expect(region.texture).not.toBeNull()
  })

  it('releases the main texture when switching to band rendering', () => {
    const gl = fakeGl()
    const region = bandRegion()
    ensureRegionGpuResources(gl, region)
    const mainTexture = region.texture

    expect(ensureRegionGpuResources(gl, region, ['red', 'green'])).toBe(true)
    expect(gl.deleteTexture).toHaveBeenCalledWith(mainTexture)
    expect(region.texture).toBeNull()
  })
})

describe('ensureRegionGpuResources with integer bands', () => {
  function int8Region(): RegionState {
    const region = fetchedRegion()
    region.bandData.set('a', new Int8Array([-128, -1, 0, 127]))
    region.bandData.set('b', new Int8Array([1, 2, 3, 4]))
    return region
  }

  it('uploads raw integers to an integer texture array', () => {
    const gl = fakeGl()
    const region = int8Region()

    expect(ensureRegionGpuResources(gl, region, ['a', 'b'], 'int')).toBe(true)
    const [, , internalFormat, , , depth, , format, type, data] =
      gl.texImage3D.mock.calls[0]
    expect([internalFormat, format, type]).toEqual([R8I, RED_INTEGER, BYTE])
    expect(depth).toBe(2)
    expect(data).toBeInstanceOf(Int8Array)
    expect([...data]).toEqual([-128, -1, 0, 127, 1, 2, 3, 4])
    expect(region.bandTextureKey).toBe(bandTextureKey(['a', 'b'], 'int'))
  })

  it('picks the texture format from the array type', () => {
    const gl = fakeGl()
    const region = fetchedRegion()
    region.bandData.set('a', new Uint16Array([0, 1, 2, 65535]))

    expect(ensureRegionGpuResources(gl, region, ['a'], 'uint')).toBe(true)
    const [, , internalFormat, , , , , format, type] =
      gl.texImage3D.mock.calls[0]
    expect([internalFormat, format, type]).toEqual([
      R16UI,
      RED_INTEGER,
      UNSIGNED_SHORT,
    ])
  })

  it('unpacks unaligned rows and restores the alignment', () => {
    const gl = fakeGl()
    ensureRegionGpuResources(gl, int8Region(), ['a', 'b'], 'int')

    expect(gl.pixelStorei.mock.calls).toEqual([
      [UNPACK_ALIGNMENT, 1],
      [UNPACK_ALIGNMENT, 4],
    ])
  })

  it('leaves unpack state alone for float bands', () => {
    const gl = fakeGl()
    const region = fetchedRegion()
    region.bandData.set('a', new Float32Array(4))
    ensureRegionGpuResources(gl, region, ['a'])
    expect(gl.pixelStorei).not.toHaveBeenCalled()
  })

  it('reports a region stored in another format as undrawable', () => {
    const gl = fakeGl()
    const region = int8Region()

    // A fallback from a float level must not be sampled as integers, nor
    // integers as floats.
    expect(ensureRegionGpuResources(gl, region, ['a', 'b'], 'float')).toBe(
      false
    )
    expect(ensureRegionGpuResources(gl, region, ['a', 'b'], 'uint')).toBe(false)
    expect(gl.texImage3D).not.toHaveBeenCalled()
  })

  it('refuses to mix array types within one texture', () => {
    const gl = fakeGl()
    const region = int8Region()
    region.bandData.set('b', new Int16Array([1, 2, 3, 4]))

    expect(ensureRegionGpuResources(gl, region, ['a', 'b'], 'int')).toBe(false)
    expect(gl.texImage3D).not.toHaveBeenCalled()
  })
})

describe('bindBandTexture', () => {
  const texture = { tex: 1 } as unknown as WebGLTexture

  it('binds the array to unit 2 when it holds the sampled bands', () => {
    const gl = fakeGl()
    const region = {
      bandTexture: texture,
      bandTextureKey: bandTextureKey(['red', 'green']),
    }

    expect(bindBandTexture(gl, region, ['red', 'green'])).toBe(true)
    expect(gl.activeTexture).toHaveBeenCalledWith(TEXTURE0 + 2)
    expect(gl.bindTexture).toHaveBeenCalledWith(TEXTURE_2D_ARRAY, texture)
  })

  it('refuses a texture holding a different band list', () => {
    const gl = fakeGl()
    const region = {
      bandTexture: texture,
      bandTextureKey: bandTextureKey(['green', 'red']),
    }

    expect(bindBandTexture(gl, region, ['red', 'green'])).toBe(false)
    expect(gl.bindTexture).not.toHaveBeenCalled()
  })

  it('refuses a texture stored in another format', () => {
    const gl = fakeGl()
    const region = {
      bandTexture: texture,
      bandTextureKey: bandTextureKey(['red'], 'int'),
    }

    expect(bindBandTexture(gl, region, ['red'])).toBe(false)
    expect(bindBandTexture(gl, region, ['red'], 'int')).toBe(true)
  })

  it('refuses a region whose bands are not uploaded', () => {
    const gl = fakeGl()
    expect(
      bindBandTexture(gl, { bandTexture: null, bandTextureKey: null }, ['red'])
    ).toBe(false)
    expect(
      bindBandTexture(gl, { bandTexture: texture, bandTextureKey: null }, [
        'red',
      ])
    ).toBe(false)
  })
})
