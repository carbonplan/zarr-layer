import type { RegionState } from './region-state'
import { bandTextureKey } from './render-helpers'
import type { BandFormat } from './band-format'

/**
 * Pixel bytes the cache holds across its regions, CPU arrays and GPU
 * textures alike, the limit that governs it: region cost varies from
 * kilobytes (one small band) to tens of megabytes (many bands of a large
 * chunk), so no region count fits both.
 */
export const MAX_CACHED_REGION_BYTES = 512 * 1024 * 1024

/**
 * Regions the cache may keep: at least the floor, and twice the regions an
 * update wants, so fallbacks fit beside them, up to a hard bound that keeps
 * the count of GL objects (a texture and three buffers each) sane.
 */
export const MIN_CACHED_REGIONS = 128
export const MAX_CACHED_REGIONS = 1024
const CACHED_PER_WANTED_REGION = 2

/**
 * Share of the cache one update may want; the rest is headroom for the
 * fallbacks shown while the wanted regions load.
 */
export const WANTED_CACHE_SHARE = 1 / 2
export const MAX_WANTED_REGION_BYTES =
  MAX_CACHED_REGION_BYTES * WANTED_CACHE_SHARE
export const MAX_WANTED_REGIONS = MAX_CACHED_REGIONS * WANTED_CACHE_SHARE

export interface RegionCacheLimits {
  maxBytes: number
  minRegions: number
  maxRegions: number
}

export const DEFAULT_REGION_CACHE_LIMITS: RegionCacheLimits = {
  maxBytes: MAX_CACHED_REGION_BYTES,
  minRegions: MIN_CACHED_REGIONS,
  maxRegions: MAX_CACHED_REGIONS,
}

/**
 * Bytes a region's pixel data occupies, on the CPU and on the GPU: the
 * interleaved data and its main texture, and the bands, as arrays until a
 * band texture takes them. A single band doubling as the interleaved data
 * is counted once.
 */
export function regionByteLength(region: RegionState): number {
  let bytes =
    (region.data?.byteLength ?? 0) +
    region.textureBytes +
    region.bandTextureBytes
  for (const band of region.bandData.values()) {
    if (band !== region.data) bytes += band.byteLength
  }
  return bytes
}

/**
 * Bytes `regionByteLength` will count for a region once fetched and
 * uploaded, before it is. A band texture holds each band at its stored
 * size, and its arrays are released. The main texture path keeps the bands
 * as float32 arrays, an interleaved copy of them when there is more than
 * one, and the main texture, the same size again.
 */
export function estimateRegionBytes({
  pixels,
  bands,
  bytesPerValue,
  interleaved,
}: {
  pixels: number
  bands: number
  bytesPerValue: number
  interleaved: boolean
}): number {
  const bandBytes = pixels * bands * bytesPerValue
  if (!interleaved) return bandBytes
  return bandBytes * (bands > 1 ? 3 : 2)
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
    failedSelectorVersion: null,
    channels: 1,
    texture: null,
    textureUploaded: false,
    textureBytes: 0,
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
    bandTransform: null,
    bandTexture: null,
    bandTextureKey: null,
    bandTextureBytes: 0,
    levelMeta: null, // Set from snapshot in fetchRegion
  }
}

/**
 * The region's pixels have arrived, in whichever form the active shader
 * reads them. Band-sampling regions carry no interleaved copy, and release
 * their band arrays once the texture holds them, so `data` alone is not the
 * test for whether a region still needs fetching.
 */
export function hasRegionData(region: RegionState): boolean {
  return (
    !!region.data || region.bandData.size > 0 || region.bandTextureKey !== null
  )
}

/**
 * The region's fetch failed under this selector. It is left alone until the
 * selector changes, the level is reloaded, or the entry leaves the cache.
 */
export function isRegionFailed(
  region: RegionState,
  selectorVersion: number
): boolean {
  return region.failedSelectorVersion === selectorVersion
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
 * Likewise `bandFormat` must be the format the shader's band sampler reads.
 */
export function isRegionGpuReady(
  region: RegionState,
  requiredBands?: readonly string[],
  bandFormat: BandFormat = 'float'
): boolean {
  if (!isRegionCpuReady(region) || !region.geometryUploaded) return false
  if (requiredBands && requiredBands.length > 0) {
    return region.bandTextureKey === bandTextureKey(requiredBands, bandFormat)
  }
  return region.textureUploaded
}

/**
 * Free the region's textures and stop counting their bytes. Geometry buffers
 * stay: the mesh does not depend on the data that was dropped.
 */
export function releaseRegionTextures(
  gl: WebGL2RenderingContext | WebGLRenderingContext,
  region: RegionState
): void {
  if (region.texture) gl.deleteTexture(region.texture)
  if (region.bandTexture) gl.deleteTexture(region.bandTexture)
  region.texture = null
  region.textureUploaded = false
  region.textureBytes = 0
  region.bandTexture = null
  region.bandTextureKey = null
  region.bandTextureBytes = 0
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
  readonly limits: RegionCacheLimits

  constructor(limits: Partial<RegionCacheLimits> = {}) {
    this.limits = { ...DEFAULT_REGION_CACHE_LIMITS, ...limits }
  }

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

  /** Let every failed region be requested again. */
  clearFailures(): void {
    for (const region of this.regions.values()) {
      region.failedSelectorVersion = null
    }
  }

  /**
   * Protect `visibleKeys`, plus previously protected keys that `retain`
   * keeps, or, without it, that match none of the `retainKeysNotMatching`
   * prefixes (an empty prefix matches every key, so `''` retains nothing).
   */
  rebuildProtection(
    visibleKeys: Iterable<string>,
    {
      retainKeysNotMatching = '',
      retain,
    }: {
      retainKeysNotMatching?: string | string[]
      retain?: (key: string) => boolean
    }
  ): void {
    const prefixes =
      typeof retainKeysNotMatching === 'string'
        ? [retainKeysNotMatching]
        : retainKeysNotMatching
    const nextProtected = new Set(visibleKeys)
    for (const key of this.protectedKeys) {
      const keep = retain
        ? retain(key)
        : !prefixes.some((prefix) => key.startsWith(prefix))
      if (keep) nextProtected.add(key)
    }
    this.protectedKeys = nextProtected
  }

  /** Regions the cache may keep while an update wants `wantedCount`. */
  maxRegionsFor(wantedCount: number): number {
    const { minRegions, maxRegions } = this.limits
    return Math.min(
      maxRegions,
      Math.max(minRegions, wantedCount * CACHED_PER_WANTED_REGION)
    )
  }

  /**
   * Drop the oldest unprotected regions until the cache is within its byte
   * limit and the region count `wantedCount` allows. Protected regions stay
   * even past both.
   */
  evict(
    gl: WebGL2RenderingContext,
    {
      wantedCount = 0,
      maxBytes = this.limits.maxBytes,
    }: { wantedCount?: number; maxBytes?: number } = {}
  ): void {
    const maxRegions = this.maxRegionsFor(wantedCount)
    let totalBytes = 0
    for (const region of this.regions.values()) {
      totalBytes += regionByteLength(region)
    }
    while (this.regions.size > maxRegions || totalBytes > maxBytes) {
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
