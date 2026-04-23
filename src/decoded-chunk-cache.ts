import * as zarr from 'zarrita'

/**
 * Small insertion-order LRU. Exposes `has`/`get`/`set` so it can be used
 * both for this module's decoded-chunk extension and as the `cache`
 * container for zarrita's `withByteCaching` (which calls `cache.has(k)`
 * before `cache.get(k)` and therefore needs to distinguish "missing" from
 * "stored undefined value").
 */
export const createLRU = <V>(maxEntries: number) => {
  const store = new Map<string, V>()
  return {
    has(key: string): boolean {
      return store.has(key)
    },
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
 * Cache decoded chunks at the zarr.Array level so both render (which calls
 * `store.getChunk` / `array.getChunk`) and query (which calls `zarr.get`,
 * internally driven by `array.getChunk`) skip decompression on re-reads.
 *
 * Complements `withByteCaching` (which skips the HTTP round-trip): byte
 * caching removes network cost, decoded caching removes decompression cost.
 * Scrubbing a selector within already-rendered tiles is the dominant
 * beneficiary — the same chunk is asked for on every scrub tick and the
 * decode is the heavy piece.
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
  (array, opts: { cache: ReturnType<typeof createLRU<AnyChunk>> }) => ({
    async getChunk(coords, options) {
      const key = chunkCacheKey(array.path, coords)
      const hit = opts.cache.get(key)
      if (hit) return hit
      const chunk = await array.getChunk(coords, options)
      opts.cache.set(key, chunk)
      return chunk
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
    const cache = createLRU<AnyChunk>(opts.maxEntries ?? 128)
    return {
      arrayExtensions: [(array) => decodedChunkExtension(array, { cache })],
    }
  }
)
