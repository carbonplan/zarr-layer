/**
 * BatchedFetchStore — shard index caching + range request batching.
 *
 * Adapted from zarrita.js packages/@zarrita-storage/src/batched-fetch.ts (MIT).
 * That file is not yet published in @zarrita/storage v0.1.4, so it lives here
 * until a new zarrita release ships it.
 *
 * Two optimizations, directly analogous to GDAL Zarr work:
 *
 * 1. Shard index caching (suffix-length requests).
 *    zarrita.js creates a new shard-index cache per `open(array)` call.  For
 *    an RGB layer with three separate band arrays, the same shard index bytes
 *    are fetched three times per viewport.  This store caches suffix-length
 *    range reads (always shard-index reads) in an LRU so they are served from
 *    memory on the 2nd and 3rd band open.  Equivalent to GDAL PR #14021
 *    (81 HTTP GETs → 1 per shard).
 *
 * 2. Macrotask range batching (offset+length requests).
 *    Concurrent getRange calls that land within the same event-loop tick are
 *    accumulated, sorted by offset, grouped into merged ranges (gap threshold
 *    32 KB), fetched as fewer HTTP requests, then sliced back to callers.
 *    Modelled on geotiff.js BlockedSource; equivalent to GDAL's ReadMultiRange
 *    batching (PR #13871).
 */

import type { AsyncReadable } from '@zarrita/storage'

type AbsolutePath = `/${string}`
type RangeQuery = { offset: number; length: number } | { suffixLength: number }

/** Simple LRU cache using Map insertion order. */
class LRUCache<V> {
  #map = new Map<string, V>()
  #max: number

  constructor(max: number) {
    this.#max = max
  }

  has(key: string): boolean {
    return this.#map.has(key)
  }

  get(key: string): V | undefined {
    if (!this.#map.has(key)) return undefined
    const value = this.#map.get(key) as V
    this.#map.delete(key)
    this.#map.set(key, value)
    return value
  }

  set(key: string, value: V): void {
    this.#map.delete(key)
    this.#map.set(key, value)
    if (this.#map.size > this.#max) {
      const first = this.#map.keys().next().value
      if (first !== undefined) this.#map.delete(first)
    }
  }
}

interface PendingRequest {
  offset: number
  length: number
  resolve: (value: Uint8Array | undefined) => void
  reject: (reason: Error) => void
}

interface RangeGroup {
  offset: number
  length: number
  requests: PendingRequest[]
}

/** Maximum gap (bytes) between two requests before splitting into separate groups. */
const GAP_THRESHOLD = 32768

function groupRequests(sorted: PendingRequest[]): RangeGroup[] {
  if (sorted.length === 0) return []
  const groups: RangeGroup[] = []
  let current = [sorted[0]]
  let groupStart = sorted[0].offset
  let groupEnd = sorted[0].offset + sorted[0].length

  for (let i = 1; i < sorted.length; i++) {
    const req = sorted[i]
    const reqEnd = req.offset + req.length
    if (req.offset <= groupEnd + GAP_THRESHOLD) {
      current.push(req)
      groupEnd = Math.max(groupEnd, reqEnd)
    } else {
      groups.push({
        offset: groupStart,
        length: groupEnd - groupStart,
        requests: current,
      })
      current = [req]
      groupStart = req.offset
      groupEnd = reqEnd
    }
  }
  groups.push({
    offset: groupStart,
    length: groupEnd - groupStart,
    requests: current,
  })
  return groups
}

export default class BatchedFetchStore implements AsyncReadable<RequestInit> {
  #inner: AsyncReadable<RequestInit>
  #innerGetRange: NonNullable<AsyncReadable<RequestInit>['getRange']>
  #pending: Map<AbsolutePath, PendingRequest[]> = new Map()
  #scheduled = false
  #cache: LRUCache<Uint8Array | undefined>
  #inflight: Map<string, Promise<Uint8Array | undefined>> = new Map()

  stats = {
    hits: 0,
    inflightHits: 0,
    misses: 0,
    mergedRequests: 0,
    batchedRequests: 0,
  }

  constructor(
    inner: AsyncReadable<RequestInit>,
    options?: { cacheSize?: number }
  ) {
    if (!inner.getRange) {
      throw new Error(
        'BatchedFetchStore requires a store with getRange support'
      )
    }
    this.#inner = inner
    this.#innerGetRange = inner.getRange.bind(inner)
    this.#cache = new LRUCache(options?.cacheSize ?? 256)
  }

  get(
    key: AbsolutePath,
    options?: RequestInit
  ): Promise<Uint8Array | undefined> {
    return this.#inner.get(key, options)
  }

  getRange(
    key: AbsolutePath,
    range: RangeQuery,
    options?: RequestInit
  ): Promise<Uint8Array | undefined> {
    // Suffix requests = shard index reads. Bypass batching; serve from LRU cache.
    if ('suffixLength' in range) {
      const cacheKey = `${key}:suffix:${range.suffixLength}`
      if (this.#cache.has(cacheKey)) {
        this.stats.hits++
        return Promise.resolve(this.#cache.get(cacheKey))
      }
      const inflight = this.#inflight.get(cacheKey)
      if (inflight) {
        this.stats.inflightHits++
        return inflight
      }
      this.stats.misses++
      const promise = this.#innerGetRange(key, range, options)
        .then((data) => {
          this.#cache.set(cacheKey, data)
          this.#inflight.delete(cacheKey)
          return data
        })
        .catch((err: Error) => {
          this.#inflight.delete(cacheKey)
          throw err
        })
      this.#inflight.set(cacheKey, promise)
      return promise
    }

    const { offset, length } = range
    const cacheKey = `${key}:${offset}:${length}`
    if (this.#cache.has(cacheKey)) {
      this.stats.hits++
      return Promise.resolve(this.#cache.get(cacheKey))
    }

    this.stats.misses++
    this.stats.batchedRequests++

    return new Promise((resolve, reject) => {
      let pending = this.#pending.get(key)
      if (!pending) {
        pending = []
        this.#pending.set(key, pending)
      }
      pending.push({ offset, length, resolve, reject })
      if (!this.#scheduled) {
        this.#scheduled = true
        setTimeout(() => this.#flush(options), 0)
      }
    })
  }

  async #flush(options?: RequestInit): Promise<void> {
    const work = new Map(this.#pending)
    this.#pending.clear()
    this.#scheduled = false

    const pathPromises: Promise<void>[] = []
    for (const [path, requests] of work) {
      requests.sort((a, b) => a.offset - b.offset)
      const groups = groupRequests(requests)
      this.stats.mergedRequests += groups.length
      pathPromises.push(this.#fetchGroups(path, groups, options))
    }
    await Promise.all(pathPromises)
  }

  async #fetchGroups(
    path: AbsolutePath,
    groups: RangeGroup[],
    options?: RequestInit
  ): Promise<void> {
    for (const group of groups) {
      try {
        const data = await this.#innerGetRange(
          path,
          { offset: group.offset, length: group.length },
          options
        )
        for (const req of group.requests) {
          const cacheKey = `${path}:${req.offset}:${req.length}`
          if (!data) {
            this.#cache.set(cacheKey, undefined)
            req.resolve(undefined)
            continue
          }
          const start = req.offset - group.offset
          const slice = data.slice(start, start + req.length)
          this.#cache.set(cacheKey, slice)
          req.resolve(slice)
        }
      } catch (err) {
        for (const req of group.requests) req.reject(err as Error)
      }
    }
  }
}
