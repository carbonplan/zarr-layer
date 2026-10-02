import { describe, it, expect, vi, afterEach } from 'vitest'
import {
  cancelAllRequests,
  createRequestCanceller,
  dropQueuedFetches,
  enqueueFetches,
  hasActiveRequests,
  prioritizeQueuedFetches,
  type QueuedFetch,
} from './region-utils'

/** Queued fetches that record when they start and finish when told to. */
function makeFetches(coords: number[][]) {
  const started: string[] = []
  const dropped: string[] = []
  const finishers = new Map<string, () => void>()
  const fetches: QueuedFetch[] = coords.map(([regionX, regionY]) => {
    const key = `0:${regionX},${regionY}`
    return {
      key,
      regionX,
      regionY,
      start: () =>
        new Promise<void>((resolve) => {
          started.push(key)
          finishers.set(key, resolve)
        }),
      drop: () => {
        dropped.push(key)
      },
    }
  })
  const finish = async (key: string) => {
    finishers.get(key)!()
    // Let the settled fetch free its slot.
    await Promise.resolve()
    await Promise.resolve()
  }
  return { fetches, started, dropped, finish }
}

const row = (n: number) => Array.from({ length: n }, (_, i) => [i, 0])

describe('region fetch queue', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('starts fetches in a later task, not inside the caller', () => {
    vi.useFakeTimers()
    const canceller = createRequestCanceller(4)
    const { fetches, started } = makeFetches([[0, 0]])
    enqueueFetches(canceller, fetches)
    expect(started).toEqual([])
    vi.runAllTimers()
    expect(started).toEqual(['0:0,0'])
  })

  it('keeps no more than maxActive fetches in flight', async () => {
    vi.useFakeTimers()
    const canceller = createRequestCanceller(2)
    const { fetches, started, finish } = makeFetches(row(4))
    enqueueFetches(canceller, fetches)
    vi.runAllTimers()
    expect(started).toEqual(['0:0,0', '0:1,0'])

    await finish('0:0,0')
    vi.runAllTimers()
    expect(started).toEqual(['0:0,0', '0:1,0', '0:2,0'])
    expect(hasActiveRequests(canceller)).toBe(true)
  })

  it('starts the queued fetches nearest a point first', () => {
    vi.useFakeTimers()
    const canceller = createRequestCanceller(1)
    const { fetches, started } = makeFetches([
      [0, 0],
      [5, 5],
      [2, 2],
    ])
    enqueueFetches(canceller, fetches)
    prioritizeQueuedFetches(canceller, 5, 5)
    vi.runAllTimers()
    expect(started).toEqual(['0:5,5'])
  })

  it('drops queued fetches without starting them', () => {
    vi.useFakeTimers()
    const canceller = createRequestCanceller(1)
    const { fetches, started, dropped } = makeFetches(row(2))
    enqueueFetches(canceller, fetches)
    dropQueuedFetches(canceller, (f) => f.key === '0:0,0')
    vi.runAllTimers()
    expect(dropped).toEqual(['0:0,0'])
    expect(started).toEqual(['0:1,0'])
  })

  it('counts queued fetches as active until cancelled', () => {
    vi.useFakeTimers()
    const canceller = createRequestCanceller(1)
    const { fetches, dropped } = makeFetches([[0, 0]])
    enqueueFetches(canceller, fetches)
    expect(hasActiveRequests(canceller)).toBe(true)
    cancelAllRequests(canceller)
    expect(dropped).toEqual(['0:0,0'])
    expect(hasActiveRequests(canceller)).toBe(false)
  })
})
