import { describe, it, expect, vi } from 'vitest'
import { ensureRegionGpuResources } from './render-helpers'
import { createRegionState } from './region-cache'
import type { RegionState } from './region-state'

/**
 * The lazy GPU upload contract: fetch leaves only CPU state; the render paths
 * call ensureRegionGpuResources per frame to create and upload texture and
 * geometry buffers on the drawing context, re-uploading when the data was
 * refreshed (textureUploaded reset by fetch) and doing nothing when complete.
 */

function fakeGl() {
  let textureCount = 0
  let bufferCount = 0
  return {
    createTexture: vi.fn(() => ({ tex: ++textureCount })),
    createBuffer: vi.fn(() => ({ buf: ++bufferCount })),
    bindTexture: vi.fn(),
    bindBuffer: vi.fn(),
    bufferData: vi.fn(),
    texImage2D: vi.fn(),
    texParameteri: vi.fn(),
    activeTexture: vi.fn(),
  } as unknown as WebGL2RenderingContext & {
    createTexture: ReturnType<typeof vi.fn>
    createBuffer: ReturnType<typeof vi.fn>
    bufferData: ReturnType<typeof vi.fn>
    texImage2D: ReturnType<typeof vi.fn>
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
  region.useIndexedMesh = true
  region.vertexCount = 3
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
  })

  it('skips the index buffer for non-indexed meshes', () => {
    const gl = fakeGl()
    const region = fetchedRegion()
    region.useIndexedMesh = false
    region.indexArr = null

    expect(ensureRegionGpuResources(gl, region)).toBe(true)
    expect(region.indexBuffer).toBeNull()
    expect(gl.createBuffer).toHaveBeenCalledTimes(2)
  })
})
