import * as zarr from 'zarrita'

/** Insertion-order LRU bounded by entry count. */
const createLRU = <V>(maxEntries: number) => {
  const store = new Map<string, V>()
  return {
    get(key: string): V | undefined {
      if (!store.has(key)) return undefined
      const hit = store.get(key) as V
      store.delete(key)
      store.set(key, hit)
      return hit
    },
    set(key: string, value: V): void {
      if (store.has(key)) store.delete(key)
      store.set(key, value)
      while (store.size > maxEntries) {
        const oldest = store.keys().next().value
        if (oldest === undefined) break
        store.delete(oldest)
      }
    },
  }
}

/**
 * Cache decoded chunks at the zarr.Array level so render
 * (`array.getChunk`), scrub (same chunks, new selector slice), and query
 * (`zarr.get`, internally driven by `array.getChunk`) skip decompression
 * on re-reads. Concurrent callers for the same chunk share one fetch via
 * the in-flight map so they don't each decompress independently.
 *
 * Cached chunks are returned by reference; callers must treat `chunk.data`
 * as read-only. Downstream code in this repo only reads from it (slice +
 * extract into fresh arrays), which keeps the cache safe. Mutating a
 * cached chunk will corrupt every future read that hits the same key.
 *
 * Keyed on `(array.path, chunkCoords)`, per-store, count-bounded LRU.
 */

type AnyChunk = zarr.Chunk<zarr.DataType>

const chunkCacheKey = (path: string, coords: number[]): string =>
  `${path}\0${coords.join(',')}`

const decodedChunkExtension = zarr.defineArrayExtension(
  (
    array,
    opts: {
      cache: ReturnType<typeof createLRU<AnyChunk>>
      pending: Map<string, Promise<AnyChunk>>
    }
  ) => ({
    async getChunk(coords, options) {
      const key = chunkCacheKey(array.path, coords)
      const hit = opts.cache.get(key)
      if (hit) return hit
      const inflight = opts.pending.get(key)
      if (inflight) return inflight
      const promise = array
        .getChunk(coords, options)
        .then((chunk) => {
          opts.cache.set(key, chunk)
          return chunk
        })
        .finally(() => {
          if (opts.pending.get(key) === promise) {
            opts.pending.delete(key)
          }
        })
      opts.pending.set(key, promise)
      return promise
    },
  })
)

/**
 * Wrap a store so every zarr.Array it produces memoizes decoded `getChunk`
 * results. The extension is attached via the store's `arrayExtensions`
 * field so `zarr.open` auto-applies it to every array.
 */
export const withDecodedChunkCaching = zarr.defineStoreExtension(
  (_inner, opts: { maxEntries?: number } = {}) => {
    const cache = createLRU<AnyChunk>(opts.maxEntries ?? 512)
    const pending = new Map<string, Promise<AnyChunk>>()
    return {
      arrayExtensions: [
        (array) => decodedChunkExtension(array, { cache, pending }),
      ],
    }
  }
)
