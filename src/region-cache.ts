import type { RegionState } from './region-state'

/** Maximum number of regions to keep in cache (LRU eviction) */
export const MAX_CACHED_REGIONS = 128

export function makeRegionKey(
  levelIndex: number,
  regionX: number,
  regionY: number
): string {
  return `${levelIndex}:${regionX},${regionY}`
}

export function createRegionState(
  levelIndex: number,
  regionX: number,
  regionY: number,
  latIsAscending: boolean,
  selectorVersion: number
): RegionState {
  return {
    key: makeRegionKey(levelIndex, regionX, regionY),
    levelIndex,
    regionX,
    regionY,
    data: null,
    width: 0,
    height: 0,
    loading: false,
    requestId: null,
    channels: 1,
    texture: null,
    textureUploaded: false,
    vertexBuffer: null,
    pixCoordBuffer: null,
    indexBuffer: null,
    geometryUploaded: false,
    vertexArr: null,
    pixCoordArr: null,
    indexArr: null,
    vertexCount: 0,
    useIndexedMesh: false,
    mercatorBounds: null,
    wgs84Bounds: null,
    latIsAscending,
    selectorVersion,
    bandData: new Map(),
    bandTextures: new Map(),
    bandTexturesUploaded: new Set(),
    bandTexturesConfigured: new Set(),
    levelMeta: null, // Set from snapshot in fetchRegion
  }
}

/**
 * The region has everything an upload needs. Gates whether
 * `ensureRegionGpuResources` is worth attempting — not whether the region can
 * be drawn, which also requires that upload to have succeeded.
 */
export function isRegionCpuReady(region: RegionState): boolean {
  return !!(
    region.data &&
    region.vertexArr &&
    region.pixCoordArr &&
    region.mercatorBounds &&
    region.levelMeta
  )
}

/**
 * The region is drawable right now. Use this, never `isRegionCpuReady`, to
 * decide that a level covers the viewport: a level that displaces its
 * lower-resolution fallbacks on CPU state alone leaves nothing on screen if
 * its uploads then fail.
 */
export function isRegionGpuReady(region: RegionState): boolean {
  return (
    isRegionCpuReady(region) &&
    region.textureUploaded &&
    region.geometryUploaded
  )
}

export function disposeRegion(
  gl: WebGL2RenderingContext | WebGLRenderingContext,
  region: RegionState
): void {
  if (region.texture) gl.deleteTexture(region.texture)
  if (region.vertexBuffer) gl.deleteBuffer(region.vertexBuffer)
  if (region.pixCoordBuffer) gl.deleteBuffer(region.pixCoordBuffer)
  if (region.indexBuffer) gl.deleteBuffer(region.indexBuffer)
  for (const tex of region.bandTextures.values()) gl.deleteTexture(tex)
}

export class RegionCache {
  private regions = new Map<string, RegionState>()
  private protectedKeys = new Set<string>()

  get size(): number {
    return this.regions.size
  }
  get(key: string): RegionState | undefined {
    return this.regions.get(key)
  }
  set(key: string, region: RegionState): void {
    this.regions.set(key, region)
  }
  delete(key: string): boolean {
    return this.regions.delete(key)
  }
  values(): MapIterator<RegionState> {
    return this.regions.values()
  }
  entries(): MapIterator<[string, RegionState]> {
    return this.regions.entries()
  }
  [Symbol.iterator](): MapIterator<[string, RegionState]> {
    return this.regions[Symbol.iterator]()
  }
  isProtected(key: string): boolean {
    return this.protectedKeys.has(key)
  }

  rebuildProtection(
    visibleKeys: Iterable<string>,
    { retainKeysNotMatching }: { retainKeysNotMatching: string }
  ): void {
    const nextProtected = new Set(visibleKeys)
    for (const key of this.protectedKeys) {
      if (!key.startsWith(retainKeysNotMatching)) nextProtected.add(key)
    }
    this.protectedKeys = nextProtected
  }

  evict(gl: WebGL2RenderingContext): void {
    // Uses Map iteration order (oldest first). Never evicts currently visible regions.
    while (this.regions.size > MAX_CACHED_REGIONS) {
      let evictedKey: string | null = null
      for (const key of this.regions.keys()) {
        if (!this.protectedKeys.has(key)) {
          evictedKey = key
          break
        }
      }
      if (!evictedKey) break // All regions are visible, stop
      const region = this.regions.get(evictedKey)
      if (region) disposeRegion(gl, region)
      this.regions.delete(evictedKey)
    }
  }

  clear(gl: WebGL2RenderingContext | WebGLRenderingContext): void {
    for (const region of this.regions.values()) disposeRegion(gl, region)
    this.regions.clear()
    this.protectedKeys.clear()
  }
}
