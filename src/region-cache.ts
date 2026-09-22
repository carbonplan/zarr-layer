import type { RegionState } from './region-state'
import { bandTextureKey } from './render-helpers'

/** Maximum number of regions to keep in cache (LRU eviction) */
export const MAX_CACHED_REGIONS = 128

/**
 * Maximum decoded bytes held across cached regions. Region cost varies with
 * band count: one 256x256 float32 band is 256 KB, but 64 bands of it are
 * 16 MB, so the count cap alone would let the cache grow to gigabytes. The
 * GPU holds a copy of the same size.
 */
export const MAX_CACHED_REGION_BYTES = 512 * 1024 * 1024

/** CPU bytes a region's pixel data occupies. */
export function regionByteLength(region: RegionState): number {
  let bytes = region.data?.byteLength ?? 0
  for (const band of region.bandData.values()) bytes += band.byteLength
  return bytes
}

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
    indexCount: 0,
    mercatorBounds: null,
    meshBounds: null,
    latIsAscending,
    selectorVersion,
    bandData: new Map(),
    bandTexture: null,
    bandTextureKey: null,
    levelMeta: null, // Set from snapshot in fetchRegion
  }
}

/**
 * The region's pixels have arrived, in whichever form the active shader
 * reads them. Band-sampling regions carry no interleaved copy, so `data`
 * alone is not the test for whether a region still needs fetching.
 */
export function hasRegionData(region: RegionState): boolean {
  return !!region.data || region.bandData.size > 0
}

/**
 * The region has all CPU-side data required to upload and later draw its mesh.
 * Gates whether `ensureRegionGpuResources` is worth attempting — not whether
 * the region can be drawn, which also requires that upload to have succeeded.
 */
export function isRegionCpuReady(region: RegionState): boolean {
  return !!(
    hasRegionData(region) &&
    region.vertexArr &&
    region.pixCoordArr &&
    region.indexArr &&
    region.mercatorBounds &&
    region.meshBounds &&
    region.levelMeta
  )
}

/**
 * The region is drawable right now. Use this, never `isRegionCpuReady`, to
 * decide that a level covers the viewport: a level that displaces its
 * lower-resolution fallbacks on CPU state alone leaves nothing on screen if
 * its uploads then fail.
 *
 * `requiredBands` names the textures a custom shader samples. Pass the same
 * list the draw call uses, or a region whose main texture is resident but
 * whose bands are not would count towards coverage and then fail to draw.
 */
export function isRegionGpuReady(
  region: RegionState,
  requiredBands?: readonly string[]
): boolean {
  if (!isRegionCpuReady(region) || !region.geometryUploaded) return false
  if (requiredBands && requiredBands.length > 0) {
    return region.bandTextureKey === bandTextureKey(requiredBands)
  }
  return region.textureUploaded
}

export function disposeRegion(
  gl: WebGL2RenderingContext | WebGLRenderingContext,
  region: RegionState
): void {
  if (region.texture) gl.deleteTexture(region.texture)
  if (region.vertexBuffer) gl.deleteBuffer(region.vertexBuffer)
  if (region.pixCoordBuffer) gl.deleteBuffer(region.pixCoordBuffer)
  if (region.indexBuffer) gl.deleteBuffer(region.indexBuffer)
  if (region.bandTexture) gl.deleteTexture(region.bandTexture)
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

  evict(
    gl: WebGL2RenderingContext,
    maxBytes: number = MAX_CACHED_REGION_BYTES
  ): void {
    // Uses Map iteration order (oldest first). Never evicts currently visible regions.
    let totalBytes = 0
    for (const region of this.regions.values()) {
      totalBytes += regionByteLength(region)
    }
    while (this.regions.size > MAX_CACHED_REGIONS || totalBytes > maxBytes) {
      let evictedKey: string | null = null
      for (const key of this.regions.keys()) {
        if (!this.protectedKeys.has(key)) {
          evictedKey = key
          break
        }
      }
      if (!evictedKey) break // All regions are visible, stop
      const region = this.regions.get(evictedKey)
      if (region) {
        totalBytes -= regionByteLength(region)
        disposeRegion(gl, region)
      }
      this.regions.delete(evictedKey)
    }
  }

  clear(gl: WebGL2RenderingContext | WebGLRenderingContext): void {
    for (const region of this.regions.values()) disposeRegion(gl, region)
    this.regions.clear()
    this.protectedKeys.clear()
  }
}
