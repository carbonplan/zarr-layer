import { describe, it, expect } from 'vitest'
import * as zarr from 'zarrita'
import { withDecodedChunkCaching } from './decoded-chunk-cache'
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
})
