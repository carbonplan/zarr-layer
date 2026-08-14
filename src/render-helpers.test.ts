import { describe, it, expect, vi } from 'vitest'
import { bindBandTextures, ensureRegionGpuResources } from './render-helpers'
import { createRegionState } from './region-cache'
import type { RegionState } from './region-state'

/**
 * The lazy GPU upload contract: fetch leaves only CPU state; the render paths
 * call ensureRegionGpuResources per frame to create and upload texture and
 * geometry buffers on the drawing context, re-uploading when the data was
 * refreshed (textureUploaded reset by fetch) and doing nothing when complete.
 * Custom-shader bands take the parallel per-band path in bindBandTextures.
 */

// Texture unit constants matter: bands bind from unit 2 up, in config order.
const TEXTURE0 = 0x84c0

function fakeGl({ failTextures = false }: { failTextures?: boolean } = {}) {
  let textureCount = 0
  let bufferCount = 0
  return {
    TEXTURE0,
    TEXTURE_2D: 0x0de1,
    createTexture: vi.fn(() => (failTextures ? null : { tex: ++textureCount })),
    createBuffer: vi.fn(() => ({ buf: ++bufferCount })),
    deleteTexture: vi.fn(),
    bindTexture: vi.fn(),
    bindBuffer: vi.fn(),
    bufferData: vi.fn(),
    texImage2D: vi.fn(),
    texParameteri: vi.fn(),
    activeTexture: vi.fn(),
  } as unknown as WebGL2RenderingContext & {
    createTexture: ReturnType<typeof vi.fn>
    createBuffer: ReturnType<typeof vi.fn>
    deleteTexture: ReturnType<typeof vi.fn>
    bufferData: ReturnType<typeof vi.fn>
    texImage2D: ReturnType<typeof vi.fn>
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

  it('uploads the band textures the shader will sample', () => {
    const gl = fakeGl()
    const region = bandRegion()

    expect(ensureRegionGpuResources(gl, region, ['red', 'green'])).toBe(true)
    expect([...region.bandTexturesUploaded]).toEqual(['red', 'green'])
    expect(region.bandTextures.size).toBe(2)
    // Nothing samples the main texture in this mode.
    expect(region.texture).toBeNull()
    expect(region.textureUploaded).toBe(false)
    expect(gl.createTexture).toHaveBeenCalledTimes(2)
  })

  it('reports not-ready when a band texture cannot be allocated', () => {
    const gl = fakeGl({ failTextures: true })
    const region = bandRegion()

    // Coverage is decided by this result, so a failed band allocation has to
    // read as undrawable rather than as a covered level.
    expect(ensureRegionGpuResources(gl, region, ['red', 'green'])).toBe(false)
  })

  it('reports not-ready when a required band has no data', () => {
    const gl = fakeGl()
    const region = bandRegion()
    expect(ensureRegionGpuResources(gl, region, ['red', 'missing'])).toBe(false)
  })

  it('releases band textures when switching back to the main texture', () => {
    const gl = fakeGl()
    const region = bandRegion()
    ensureRegionGpuResources(gl, region, ['red', 'green'])
    const released = [...region.bandTextures.values()]

    // The shader no longer samples bands. bindBandTextures is not called at
    // all in that mode, so this is the only chance to free them.
    expect(ensureRegionGpuResources(gl, region)).toBe(true)
    for (const tex of released) {
      expect(gl.deleteTexture).toHaveBeenCalledWith(tex)
    }
    expect(region.bandTextures.size).toBe(0)
    expect(region.bandTexturesUploaded.size).toBe(0)
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

  it('drops bands that leave the set on the same call', () => {
    const gl = fakeGl()
    const region = bandRegion()
    ensureRegionGpuResources(gl, region, ['red', 'green'])
    const stale = [...region.bandTextures.values()]

    // Same-size swap: a count-based prune would miss this entirely.
    region.bandData.clear()
    region.bandData.set('nir', new Float32Array([1, 2, 3, 4]))
    region.bandData.set('swir', new Float32Array([5, 6, 7, 8]))
    expect(ensureRegionGpuResources(gl, region, ['nir', 'swir'])).toBe(true)

    for (const tex of stale) {
      expect(gl.deleteTexture).toHaveBeenCalledWith(tex)
    }
    expect([...region.bandTextures.keys()]).toEqual(['nir', 'swir'])
  })
})

describe('bindBandTextures', () => {
  const bandOptions = (region: RegionState, bands: string[]) => ({
    bandData: region.bandData,
    bandTextures: region.bandTextures,
    bandTexturesUploaded: region.bandTexturesUploaded,
    bandTexturesConfigured: region.bandTexturesConfigured,
    customShaderConfig: { bands } as never,
    width: region.width,
    height: region.height,
  })

  function twoBandRegion(): RegionState {
    const region = fetchedRegion()
    region.bandData.set('red', new Float32Array([1, 2, 3, 4]))
    region.bandData.set('green', new Float32Array([5, 6, 7, 8]))
    return region
  }

  it('creates, configures, and uploads one texture per band', () => {
    const gl = fakeGl()
    const region = twoBandRegion()

    expect(bindBandTextures(gl, bandOptions(region, ['red', 'green']))).toBe(
      true
    )
    expect(gl.createTexture).toHaveBeenCalledTimes(2)
    expect(gl.texImage2D).toHaveBeenCalledTimes(2)
    expect(region.bandTextures.size).toBe(2)
    expect([...region.bandTexturesUploaded]).toEqual(['red', 'green'])
    expect([...region.bandTexturesConfigured]).toEqual(['red', 'green'])
    // Bands occupy consecutive units from 2 up, in shader-config order.
    expect(gl.activeTexture.mock.calls.map(([unit]) => unit)).toEqual([
      TEXTURE0 + 2,
      TEXTURE0 + 3,
    ])
    // Uploaded in the shader's band order, not insertion order.
    expect(gl.texImage2D.mock.calls.map((call) => call[8])).toEqual([
      region.bandData.get('red'),
      region.bandData.get('green'),
    ])
  })

  it('rebinds without re-uploading once every band is resident', () => {
    const gl = fakeGl()
    const region = twoBandRegion()
    bindBandTextures(gl, bandOptions(region, ['red', 'green']))

    expect(bindBandTextures(gl, bandOptions(region, ['red', 'green']))).toBe(
      true
    )
    expect(gl.createTexture).toHaveBeenCalledTimes(2)
    expect(gl.texImage2D).toHaveBeenCalledTimes(2)
    expect(gl.texParameteri).toHaveBeenCalledTimes(2 * 4)
    // Still bound every frame — only the upload is skipped.
    expect(gl.bindTexture).toHaveBeenCalledTimes(4)
  })

  it('uploads only the bands that are not yet resident', () => {
    const gl = fakeGl()
    const region = twoBandRegion()
    bindBandTextures(gl, bandOptions(region, ['red']))

    expect(bindBandTextures(gl, bandOptions(region, ['red', 'green']))).toBe(
      true
    )
    expect(gl.texImage2D).toHaveBeenCalledTimes(2)
    expect(gl.texImage2D.mock.calls[1][8]).toBe(region.bandData.get('green'))
  })

  it('reports failure and uploads nothing when a band has no data', () => {
    const gl = fakeGl()
    const region = twoBandRegion()

    expect(bindBandTextures(gl, bandOptions(region, ['red', 'blue']))).toBe(
      false
    )
    // 'red' still binds before the missing band aborts the loop; 'blue' does
    // not get a texture, so the caller must skip the draw entirely.
    expect(region.bandTextures.has('blue')).toBe(false)
    expect(region.bandTexturesUploaded.has('blue')).toBe(false)
  })

  it('releases textures for bands that are no longer requested', () => {
    const gl = fakeGl()
    const region = twoBandRegion()
    bindBandTextures(gl, bandOptions(region, ['red', 'green']))
    const greenTexture = region.bandTextures.get('green')

    // The selector changed: only 'red' is in the band set now.
    region.bandData.delete('green')
    expect(bindBandTextures(gl, bandOptions(region, ['red']))).toBe(true)

    expect(gl.deleteTexture).toHaveBeenCalledWith(greenTexture)
    expect(region.bandTextures.has('green')).toBe(false)
    expect(region.bandTexturesUploaded.has('green')).toBe(false)
    expect(region.bandTexturesConfigured.has('green')).toBe(false)
    // The retained band keeps its texture and is not re-uploaded.
    expect(region.bandTextures.has('red')).toBe(true)
    expect(gl.texImage2D).toHaveBeenCalledTimes(2)
  })

  it('leaves caller-owned textures alone', () => {
    const gl = fakeGl()
    const region = twoBandRegion()
    const ensureTexture = vi.fn(
      (name: string) => ({ pooled: name } as unknown as WebGLTexture)
    )
    bindBandTextures(gl, {
      ...bandOptions(region, ['red', 'green']),
      ensureTexture,
    })

    bindBandTextures(gl, { ...bandOptions(region, ['red']), ensureTexture })
    expect(gl.deleteTexture).not.toHaveBeenCalled()
    expect(region.bandTextures.has('green')).toBe(true)
  })

  it('routes texture creation through ensureTexture when provided', () => {
    const gl = fakeGl()
    const region = twoBandRegion()
    const ensureTexture = vi.fn(() => ({ pooled: true } as WebGLTexture))

    expect(
      bindBandTextures(gl, {
        ...bandOptions(region, ['red']),
        ensureTexture,
      })
    ).toBe(true)
    expect(ensureTexture).toHaveBeenCalledWith('red')
    expect(gl.createTexture).not.toHaveBeenCalled()
    expect(region.bandTextures.get('red')).toEqual({ pooled: true })
  })
})
