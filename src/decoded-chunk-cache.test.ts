import { describe, it, expect, vi } from 'vitest'
import * as zarr from 'zarrita'
import {
  abortWaiterCount,
  withDecodedChunkCaching,
} from './decoded-chunk-cache'
import { buildMemoryZarrStore, ramp } from './__fixtures__/memory-zarr'

/**
 * The cache is exercised through real zarrita reads against the in-memory
 * fixture, counting store reads: a chunk served from cache issues no read, an
 * evicted one issues another. Chunks here are 4x4 float32, so 64 bytes each.
 */

const CHUNK_BYTES = 4 * 4 * 4

async function makeArray(opts: { maxEntries?: number; maxBytes?: number }) {
  const memory = buildMemoryZarrStore({
    arrays: [
      {
        name: 'temperature',
        shape: [4, 16],
        chunkShape: [4, 4],
        dimensionNames: ['lat', 'lon'],
        chunks: {
          '0/0': ramp(16),
          '0/1': ramp(16),
          '0/2': ramp(16),
          '0/3': ramp(16),
        },
      },
    ],
  })

  const reads: string[] = []
  const counting = {
    get: async (key: string) => {
      if (key.includes('/c/')) reads.push(key)
      return memory.get(key)
    },
  }

  const store = await zarr.extendStore(counting, (inner) =>
    withDecodedChunkCaching(inner, opts)
  )
  const array = await zarr.open.v3(zarr.root(store).resolve('temperature'), {
    kind: 'array',
  })
  return { array, reads }
}

describe('withDecodedChunkCaching', () => {
  it('serves a repeated chunk read from cache', async () => {
    const { array, reads } = await makeArray({})
    await array.getChunk([0, 0])
    await array.getChunk([0, 0])
    expect(reads).toHaveLength(1)
  })

  it('holds a viewport worth of ordinary chunks by default', async () => {
    // The default budget has to clear the working set: one rendered region
    // reads one chunk, so a cache below that turns panning and every selector
    // change back into network reads.
    const { array, reads } = await makeArray({})
    for (const x of [0, 1, 2, 3]) await array.getChunk([0, x])
    expect(reads).toHaveLength(4)

    for (const x of [0, 1, 2, 3]) await array.getChunk([0, x])
    expect(reads).toHaveLength(4)
  })

  it('shares one read between concurrent callers', async () => {
    const { array, reads } = await makeArray({})
    await Promise.all([array.getChunk([0, 0]), array.getChunk([0, 0])])
    expect(reads).toHaveLength(1)
  })

  it('evicts by total bytes, not just entry count', async () => {
    // Room for two chunks. The entry cap is far higher, so only the byte
    // budget can force the eviction below.
    const { array, reads } = await makeArray({
      maxEntries: 512,
      maxBytes: CHUNK_BYTES * 2,
    })
    await array.getChunk([0, 0])
    await array.getChunk([0, 1])
    await array.getChunk([0, 2])
    expect(reads).toHaveLength(3)

    // The oldest is gone, the two newest are still resident.
    await array.getChunk([0, 1])
    await array.getChunk([0, 2])
    expect(reads).toHaveLength(3)

    await array.getChunk([0, 0])
    expect(reads).toHaveLength(4)
  })

  it('keeps a chunk larger than the whole budget rather than caching nothing', async () => {
    const { array, reads } = await makeArray({ maxBytes: 1 })
    await array.getChunk([0, 0])
    await array.getChunk([0, 0])
    expect(reads).toHaveLength(1)
  })

  it('never retains more than the byte budget', async () => {
    // The budget is the hard bound: no entry floor keeps chunks resident past
    // it, which is what would let a large chunk size retain gigabytes.
    const { array, reads } = await makeArray({ maxBytes: CHUNK_BYTES })
    for (const x of [0, 1, 2, 3]) await array.getChunk([0, x])
    expect(reads).toHaveLength(4)

    // Only the newest survived, so every earlier chunk reads again.
    for (const x of [0, 1, 2]) await array.getChunk([0, x])
    expect(reads).toHaveLength(7)
  })

  it('serves in-flight readers from the shared fetch, not the cache', async () => {
    // Awaiters resolve from the pending promise, so eviction between the
    // fetch starting and finishing cannot break them. Worth pinning: it is
    // the reason the byte bound needs no carve-out for active chunks.
    const { array, reads } = await makeArray({ maxBytes: 1 })
    const [a, b] = await Promise.all([
      array.getChunk([0, 0]),
      array.getChunk([0, 0]),
    ])
    expect(reads).toHaveLength(1)
    expect(a).toBe(b)
    expect(Array.from(a.data as Float32Array)).toEqual(ramp(16))
  })

  it('still honors the entry cap when chunks are small', async () => {
    const { array, reads } = await makeArray({
      maxEntries: 2,
      maxBytes: 1024 * 1024,
    })
    await array.getChunk([0, 0])
    await array.getChunk([0, 1])
    await array.getChunk([0, 2])
    await array.getChunk([0, 0])
    expect(reads).toHaveLength(4)
  })

  it('refreshes recency on a cache hit', async () => {
    const { array, reads } = await makeArray({ maxBytes: CHUNK_BYTES * 2 })
    await array.getChunk([0, 0])
    await array.getChunk([0, 1])
    // Touch the oldest so the next insert evicts [0, 1] instead.
    await array.getChunk([0, 0])
    await array.getChunk([0, 2])
    expect(reads).toHaveLength(3)

    await array.getChunk([0, 0])
    expect(reads).toHaveLength(3)
    await array.getChunk([0, 1])
    expect(reads).toHaveLength(4)
  })

  it('listens once per signal however many reads share it', async () => {
    const { array } = await makeArray({})
    const { signal } = new AbortController()
    const add = vi.spyOn(signal, 'addEventListener')
    const remove = vi.spyOn(signal, 'removeEventListener')
    await Promise.all(
      [0, 1, 2, 3].map((x) => array.getChunk([0, x], { signal }))
    )
    await array.getChunk([0, 0], { signal })
    expect(add).toHaveBeenCalledTimes(1)
    expect(remove).not.toHaveBeenCalled()
  })

  it('rejects every read waiting on an aborted signal', async () => {
    const { array } = await makeArray({})
    const controller = new AbortController()
    const reads = [0, 1, 2].map((x) =>
      array.getChunk([0, x], { signal: controller.signal })
    )
    controller.abort()
    for (const read of reads) {
      await expect(read).rejects.toMatchObject({ name: 'AbortError' })
    }
  })

  it('keeps one caller waiting when another sharing the read aborts', async () => {
    // A store that honors abort signals and holds reads open until released,
    // so cancelling the shared read would reject the surviving caller.
    const memory = buildMemoryZarrStore({
      arrays: [
        {
          name: 'temperature',
          shape: [4, 4],
          chunkShape: [4, 4],
          dimensionNames: ['lat', 'lon'],
          chunks: { '0/0': ramp(16) },
        },
      ],
    })
    let release = () => {}
    const released = new Promise<void>((res) => {
      release = res
    })
    const readSignals: (AbortSignal | undefined)[] = []
    const gated = {
      get: async (key: string, options?: { signal?: AbortSignal }) => {
        if (key.includes('/c/')) {
          readSignals.push(options?.signal)
          await released
          if (options?.signal?.aborted) throw createAbortError()
        }
        return memory.get(key)
      },
    }
    const store = await zarr.extendStore(gated, (inner) =>
      withDecodedChunkCaching(inner, {})
    )
    const array = await zarr.open.v3(zarr.root(store).resolve('temperature'), {
      kind: 'array',
    })

    const first = new AbortController()
    const abandoned = array.getChunk([0, 0], { signal: first.signal })
    const kept = array.getChunk([0, 0], {
      signal: new AbortController().signal,
    })
    first.abort()
    await expect(abandoned).rejects.toMatchObject({ name: 'AbortError' })
    await Promise.resolve()
    expect(readSignals).toHaveLength(1)
    expect(readSignals[0]?.aborted ?? false).toBe(false)

    release()
    await expect(kept).resolves.toBeDefined()
  })

  it('ignores an abort after the read has finished', async () => {
    const { array } = await makeArray({})
    const controller = new AbortController()
    const chunk = await array.getChunk([0, 0], { signal: controller.signal })
    controller.abort()
    expect(chunk.data).toHaveLength(16)
  })

  it('forgets each read on a long-lived signal once it settles', async () => {
    const { array } = await makeArray({})
    const { signal } = new AbortController()
    for (const x of [0, 1, 2, 3]) await array.getChunk([0, x], { signal })
    expect(abortWaiterCount(signal)).toBe(0)
  })

  it('forgets a read that fails', async () => {
    const memory = buildMemoryZarrStore({
      arrays: [
        {
          name: 'temperature',
          shape: [4, 4],
          chunkShape: [4, 4],
          dimensionNames: ['lat', 'lon'],
          chunks: { '0/0': ramp(16) },
        },
      ],
    })
    const failing = {
      get: async (key: string) => {
        if (key.includes('/c/')) throw new Error('network down')
        return memory.get(key)
      },
    }
    const store = await zarr.extendStore(failing, (inner) =>
      withDecodedChunkCaching(inner, {})
    )
    const array = await zarr.open.v3(zarr.root(store).resolve('temperature'), {
      kind: 'array',
    })
    const { signal } = new AbortController()
    await expect(array.getChunk([0, 0], { signal })).rejects.toThrow(
      'network down'
    )
    expect(abortWaiterCount(signal)).toBe(0)
  })

  it('holds a read on its signal only while it is pending', async () => {
    const memory = buildMemoryZarrStore({
      arrays: [
        {
          name: 'temperature',
          shape: [4, 4],
          chunkShape: [4, 4],
          dimensionNames: ['lat', 'lon'],
          chunks: { '0/0': ramp(16) },
        },
      ],
    })
    let release = () => {}
    const released = new Promise<void>((res) => {
      release = res
    })
    const gated = {
      get: async (key: string) => {
        if (key.includes('/c/')) await released
        return memory.get(key)
      },
    }
    const store = await zarr.extendStore(gated, (inner) =>
      withDecodedChunkCaching(inner, {})
    )
    const array = await zarr.open.v3(zarr.root(store).resolve('temperature'), {
      kind: 'array',
    })
    const { signal } = new AbortController()

    const read = array.getChunk([0, 0], { signal })
    await Promise.resolve()
    expect(abortWaiterCount(signal)).toBe(1)
    release()
    await read
    expect(abortWaiterCount(signal)).toBe(0)
  })

  it('rejects a read whose signal is already aborted', async () => {
    const { array, reads } = await makeArray({})
    await array.getChunk([0, 0])
    const controller = new AbortController()
    controller.abort()

    // Neither a cached chunk nor a new one is read for an aborted caller.
    for (const coords of [
      [0, 0],
      [0, 1],
    ]) {
      await expect(
        array.getChunk(coords, { signal: controller.signal })
      ).rejects.toMatchObject({ name: 'AbortError' })
    }
    expect(reads).toHaveLength(1)
    expect(abortWaiterCount(controller.signal)).toBe(0)
  })
})

const createAbortError = () =>
  new DOMException('The operation was aborted.', 'AbortError')
