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

const createAbortError = () =>
  new DOMException('The operation was aborted.', 'AbortError')

interface PendingEntry {
  promise: Promise<AnyChunk>
  // Used to abort the underlying fetch only when every awaiter has given
  // up. One caller's abort must not cancel another caller's wait.
  ac: AbortController
  refCount: number
}

const decodedChunkExtension = zarr.defineArrayExtension(
  (
    array,
    opts: {
      cache: ReturnType<typeof createLRU<AnyChunk>>
      pending: Map<string, PendingEntry>
    }
  ) => ({
    async getChunk(coords, options) {
      const callerSignal = options?.signal
      if (callerSignal?.aborted) {
        throw createAbortError()
      }

      const key = chunkCacheKey(array.path, coords)
      const hit = opts.cache.get(key)
      if (hit) return hit

      let entry = opts.pending.get(key)
      if (!entry) {
        const ac = new AbortController()
        // Fresh placeholder; `promise` is assigned before we publish the
        // entry so the `.finally` closure can reference the final object.
        const fresh: PendingEntry = {
          promise: undefined as unknown as Promise<AnyChunk>,
          ac,
          refCount: 0,
        }
        fresh.promise = array
          .getChunk(coords, { ...(options ?? {}), signal: ac.signal })
          .then((chunk) => {
            opts.cache.set(key, chunk)
            return chunk
          })
          .finally(() => {
            if (opts.pending.get(key) === fresh) {
              opts.pending.delete(key)
            }
          })
        opts.pending.set(key, fresh)
        entry = fresh
      }

      const ownedEntry = entry
      ownedEntry.refCount++

      const releaseRef = () => {
        ownedEntry.refCount--
      }
      const abandonRef = () => {
        ownedEntry.refCount--
        if (ownedEntry.refCount <= 0) {
          if (opts.pending.get(key) === ownedEntry) {
            opts.pending.delete(key)
          }
          ownedEntry.ac.abort()
        }
      }

      if (!callerSignal) {
        try {
          const chunk = await ownedEntry.promise
          releaseRef()
          return chunk
        } catch (err) {
          releaseRef()
          throw err
        }
      }

      return new Promise<AnyChunk>((resolve, reject) => {
        let settled = false
        const onAbort = () => {
          if (settled) return
          settled = true
          callerSignal.removeEventListener('abort', onAbort)
          abandonRef()
          reject(createAbortError())
        }
        callerSignal.addEventListener('abort', onAbort, { once: true })
        ownedEntry.promise.then(
          (chunk) => {
            if (settled) return
            settled = true
            callerSignal.removeEventListener('abort', onAbort)
            releaseRef()
            resolve(chunk)
          },
          (err) => {
            if (settled) return
            settled = true
            callerSignal.removeEventListener('abort', onAbort)
            releaseRef()
            reject(err)
          }
        )
      })
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
    const pending = new Map<string, PendingEntry>()
    return {
      arrayExtensions: [
        (array) => decodedChunkExtension(array, { cache, pending }),
      ],
    }
  }
)
