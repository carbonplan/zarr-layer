import { describe, it, expect, vi } from 'vitest'
import {
  MAX_CACHED_REGIONS,
  RegionCache,
  createRegionState,
  disposeRegion,
  isRegionCpuReady,
  isRegionGpuReady,
  makeRegionKey,
} from './region-cache'
import type { RegionState } from './region-state'

/**
 * Contract tests for the region cache: FIFO eviction under a hard cap,
 * viewport-driven protection, and GL resource disposal. Protection semantics
 * pin the eviction fix (PR #75 review): the protected set is rebuilt from the
 * current viewport each update, never accumulated, so panning at a fixed zoom
 * can't grow the cache without bound.
 */

function fakeGl() {
  return {
    deleteTexture: vi.fn(),
    deleteBuffer: vi.fn(),
  } as unknown as WebGL2RenderingContext
}

function region(levelIndex: number, x: number, y: number): RegionState {
  return createRegionState(levelIndex, x, y, false, 0)
}

function fill(cache: RegionCache, count: number, levelIndex = 0): string[] {
  const keys: string[] = []
  for (let i = 0; i < count; i++) {
    const r = region(levelIndex, i, 0)
    cache.set(r.key, r)
    keys.push(r.key)
  }
  return keys
}

describe('makeRegionKey', () => {
  it('encodes level and coordinates', () => {
    expect(makeRegionKey(3, 7, 2)).toBe('3:7,2')
  })
})

describe('createRegionState', () => {
  it('stamps orientation and selector version with empty resources', () => {
    const r = createRegionState(2, 1, 4, true, 7)
    expect(r.key).toBe('2:1,4')
    expect(r.latIsAscending).toBe(true)
    expect(r.selectorVersion).toBe(7)
    expect(r.data).toBeNull()
    expect(r.texture).toBeNull()
    expect(r.vertexBuffer).toBeNull()
    expect(r.loading).toBe(false)
  })
})

function cpuReadyRegion() {
  const r = region(0, 0, 0)
  r.data = new Float32Array(4)
  r.vertexArr = new Float32Array(8)
  r.pixCoordArr = new Float32Array(8)
  r.mercatorBounds = { x0: 0, y0: 0, x1: 1, y1: 1 }
  r.levelMeta = { width: 2, height: 2, regionSize: [2, 2] }
  return r
}

describe('isRegionCpuReady', () => {
  it('requires CPU-side data, geometry arrays, bounds, and level meta', () => {
    expect(isRegionCpuReady(region(0, 0, 0))).toBe(false)
    expect(isRegionCpuReady(cpuReadyRegion())).toBe(true)
  })

  it('does not require GPU resources (uploads happen lazily at render)', () => {
    const r = cpuReadyRegion()
    expect(r.texture).toBeNull()
    expect(r.vertexBuffer).toBeNull()
    expect(isRegionCpuReady(r)).toBe(true)
  })
})

describe('isRegionGpuReady', () => {
  it('requires both uploads on top of CPU readiness', () => {
    const r = cpuReadyRegion()
    expect(isRegionGpuReady(r)).toBe(false)

    r.textureUploaded = true
    expect(isRegionGpuReady(r)).toBe(false)

    r.geometryUploaded = true
    expect(isRegionGpuReady(r)).toBe(true)
  })

  it('is false for an uploaded region whose data was dropped', () => {
    const r = cpuReadyRegion()
    r.textureUploaded = true
    r.geometryUploaded = true
    r.data = null
    expect(isRegionGpuReady(r)).toBe(false)
  })

  it('goes stale again when a refetch marks the uploads dirty', () => {
    const r = cpuReadyRegion()
    r.textureUploaded = true
    r.geometryUploaded = true

    // A refetch clears the texture flag; a regenerated mesh clears the other.
    r.textureUploaded = false
    expect(isRegionGpuReady(r)).toBe(false)
    expect(isRegionCpuReady(r)).toBe(true)
  })
})

describe('disposeRegion', () => {
  it('deletes the texture, geometry buffers, and band textures', () => {
    const gl = fakeGl()
    const r = region(0, 0, 0)
    r.texture = {} as WebGLTexture
    r.vertexBuffer = {} as WebGLBuffer
    r.pixCoordBuffer = {} as WebGLBuffer
    r.indexBuffer = {} as WebGLBuffer
    r.bandTextures.set('a', {} as WebGLTexture)
    r.bandTextures.set('b', {} as WebGLTexture)

    disposeRegion(gl, r)
    expect(gl.deleteTexture).toHaveBeenCalledTimes(3) // main + 2 bands
    expect(gl.deleteBuffer).toHaveBeenCalledTimes(3)
  })
})

describe('RegionCache.evict', () => {
  it('prunes oldest-first down to the cap', () => {
    const cache = new RegionCache()
    const keys = fill(cache, MAX_CACHED_REGIONS + 10)
    cache.evict(fakeGl())

    expect(cache.size).toBe(MAX_CACHED_REGIONS)
    // The 10 oldest (insertion order) are gone; the newest all survive.
    for (const key of keys.slice(0, 10)) expect(cache.get(key)).toBeUndefined()
    for (const key of keys.slice(10)) expect(cache.get(key)).toBeDefined()
  })

  it('skips protected regions and evicts the oldest unprotected instead', () => {
    const cache = new RegionCache()
    const keys = fill(cache, MAX_CACHED_REGIONS + 5)
    cache.rebuildProtection(keys.slice(0, 5), { retainKeysNotMatching: '' })
    cache.evict(fakeGl())

    expect(cache.size).toBe(MAX_CACHED_REGIONS)
    for (const key of keys.slice(0, 5)) expect(cache.get(key)).toBeDefined()
    for (const key of keys.slice(5, 10)) expect(cache.get(key)).toBeUndefined()
  })

  it('stops when every region is protected (low-zoom safety valve)', () => {
    const cache = new RegionCache()
    const keys = fill(cache, MAX_CACHED_REGIONS + 20)
    cache.rebuildProtection(keys, { retainKeysNotMatching: '' })
    cache.evict(fakeGl())
    expect(cache.size).toBe(MAX_CACHED_REGIONS + 20)
  })

  it('disposes GL resources on eviction', () => {
    const gl = fakeGl()
    const cache = new RegionCache()
    const keys = fill(cache, MAX_CACHED_REGIONS + 1)
    const oldest = cache.get(keys[0])!
    oldest.texture = {} as WebGLTexture
    cache.evict(gl)
    expect(gl.deleteTexture).toHaveBeenCalledTimes(1)
  })
})

describe('RegionCache.rebuildProtection', () => {
  it('replaces protection with the visible set when the level covers the viewport', () => {
    const cache = new RegionCache()
    cache.rebuildProtection(['0:0,0', '1:5,5'], { retainKeysNotMatching: '' })
    // Covered viewport: retain nothing from before (every key matches '').
    cache.rebuildProtection(['1:1,1', '1:2,1'], { retainKeysNotMatching: '' })

    expect(cache.isProtected('1:1,1')).toBe(true)
    expect(cache.isProtected('1:2,1')).toBe(true)
    expect(cache.isProtected('0:0,0')).toBe(false)
    expect(cache.isProtected('1:5,5')).toBe(false)
  })

  it('retains cross-level fallback keys while the level is not covered', () => {
    const cache = new RegionCache()
    cache.rebuildProtection(['0:0,0', '1:5,5'], { retainKeysNotMatching: '' })
    // Now level 1 is current but not yet covering: keep other levels' keys,
    // drop level-1 keys that are no longer visible.
    cache.rebuildProtection(['1:1,1'], { retainKeysNotMatching: '1:' })

    expect(cache.isProtected('1:1,1')).toBe(true)
    expect(cache.isProtected('0:0,0')).toBe(true) // other-level fallback kept
    expect(cache.isProtected('1:5,5')).toBe(false) // same-level, not visible
  })
})

describe('RegionCache.clear', () => {
  it('disposes everything and resets protection', () => {
    const gl = fakeGl()
    const cache = new RegionCache()
    const keys = fill(cache, 3)
    for (const key of keys) cache.get(key)!.texture = {} as WebGLTexture
    cache.rebuildProtection(keys, { retainKeysNotMatching: '' })

    cache.clear(gl)
    expect(cache.size).toBe(0)
    expect(gl.deleteTexture).toHaveBeenCalledTimes(3)
    for (const key of keys) expect(cache.isProtected(key)).toBe(false)
  })
})
