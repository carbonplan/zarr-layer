import { describe, it, expect, vi } from 'vitest'
import { LevelLoader, type LevelLoaderContext } from './level-loader'
import { SelectorResolutionError } from './selector-resolution'
import type { NormalizedSelector } from './types'
import type * as zarr from 'zarrita'

/**
 * Race tests for the level-load state machine: dedupe, token-based
 * supersession, selector atomicity, and dispose. The context's `resolveArray`
 * is gated on per-call deferreds so each test controls exactly when a load's
 * async work completes relative to competing calls.
 */

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (err: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

type Resolved = {
  zarrArray: zarr.Array<zarr.DataType>
  width: number
  height: number
  regionSize: [number, number]
  reusedArray: boolean
}

function makeHarness(
  opts: {
    isMultiscale?: boolean
    levelCount?: number
    buildSliceArgs?: LevelLoaderContext['buildSliceArgs']
  } = {}
) {
  const { isMultiscale = true, levelCount = 3 } = opts
  const gates: Array<ReturnType<typeof deferred<void>>> = []
  let selector: NormalizedSelector = {}
  const calls = {
    resolveArray: [] as Array<{ levelIndex: number; reuse: boolean }>,
    cancels: 0,
    commits: 0,
    invalidates: 0,
    errors: [] as Array<Error | null>,
  }

  const context: LevelLoaderContext = {
    isMultiscale: () => isMultiscale,
    getLevelCount: () => (isMultiscale ? levelCount : 0),
    resolveArray: async (levelIndex, reuse) => {
      calls.resolveArray.push({ levelIndex, reuse })
      const gate = deferred<void>()
      gates.push(gate)
      await gate.promise
      return {
        zarrArray: { shape: [10, 20] } as unknown as zarr.Array<zarr.DataType>,
        width: 20,
        height: 10,
        regionSize: [5, 5],
        reusedArray: reuse,
      } satisfies Resolved
    },
    buildSliceArgs:
      opts.buildSliceArgs ??
      (async () => ({
        baseSliceArgs: [0, 0],
        baseMultiValueDims: [],
      })),
    getSelector: () => selector,
    isRemoved: () => false,
    onCancelInflight: () => {
      calls.cancels++
    },
    onNewArrayCommitted: () => {
      calls.commits++
    },
    onLoadErrorChange: (error) => {
      calls.errors.push(error)
    },
    invalidate: () => {
      calls.invalidates++
    },
    getAssetLabel: (levelIndex) => String(levelIndex),
  }

  const loader = new LevelLoader(context)
  return {
    loader,
    calls,
    gates,
    setSelector: (next: NormalizedSelector) => {
      selector = next
    },
  }
}

describe('LevelLoader.loadLevel', () => {
  it('commits an atomic level runtime', async () => {
    const { loader, gates } = makeHarness()
    loader.desiredIndex = 1
    const load = loader.loadLevel(1)
    expect(loader.active).toBeNull()
    expect(loader.loadingIndex).toBe(1)
    gates[0].resolve()
    await load

    expect(loader.active).toMatchObject({
      index: 1,
      width: 20,
      height: 10,
      regionSize: [5, 5],
    })
    expect(loader.loadingIndex).toBeNull()
  })

  it('dedupes loads for the target already in flight', async () => {
    const { loader, calls, gates } = makeHarness()
    loader.desiredIndex = 1
    const first = loader.loadLevel(1)
    // Per-frame update() calls must not restart the in-flight load.
    const second = loader.loadLevel(1)
    const third = loader.loadLevel(1)
    gates[0].resolve()
    await Promise.all([first, second, third])

    expect(calls.resolveArray).toHaveLength(1)
    expect(loader.active?.index).toBe(1)
  })

  it('rejects out-of-range targets without touching state', async () => {
    const { loader, calls } = makeHarness({ levelCount: 3 })
    await loader.loadLevel(5)
    await loader.loadLevel(-1)
    expect(calls.resolveArray).toHaveLength(0)
    expect(loader.loadingIndex).toBeNull()
  })

  it('only loads level 0 for single-level datasets', async () => {
    const { loader, calls, gates } = makeHarness({ isMultiscale: false })
    await loader.loadLevel(2)
    expect(calls.resolveArray).toHaveLength(0)
    const load = loader.loadLevel(0)
    gates[0].resolve()
    await load
    expect(loader.active?.index).toBe(0)
  })

  it('drops a superseded load at the commit guard (token bump)', async () => {
    const { loader, gates } = makeHarness()
    loader.desiredIndex = 1
    const first = loader.loadLevel(1)
    // A selector rebuild for the same level supersedes via reuseArray.
    const second = loader.loadLevel(1, { reuseArray: true })
    // Resolve the FIRST load; its token is stale so the commit must drop.
    gates[0].resolve()
    await first
    expect(loader.active).toBeNull()

    gates[1].resolve()
    await second
    expect(loader.active?.index).toBe(1)
  })

  it('drops the commit when the selector changed mid-load', async () => {
    const { loader, gates, setSelector } = makeHarness()
    loader.desiredIndex = 1
    const load = loader.loadLevel(1)
    setSelector({ month: { selected: 6, type: 'value' } })
    gates[0].resolve()
    await load
    expect(loader.active).toBeNull()
  })

  it('drops the commit when the zoom target moved on', async () => {
    const { loader, gates } = makeHarness()
    loader.desiredIndex = 1
    const load = loader.loadLevel(1)
    loader.desiredIndex = 2
    gates[0].resolve()
    await load
    expect(loader.active).toBeNull()
  })

  it('reuseArray commits even when the desired index moved (selector rebuild)', async () => {
    const { loader, gates } = makeHarness()
    loader.desiredIndex = 1
    const first = loader.loadLevel(1)
    gates[0].resolve()
    await first

    loader.desiredIndex = 2
    const rebuild = loader.loadLevel(1, { reuseArray: true })
    gates[1].resolve()
    await rebuild
    expect(loader.active?.index).toBe(1)
  })

  it('signals cache reset only when a new array was fetched', async () => {
    const { loader, calls, gates } = makeHarness()
    loader.desiredIndex = 1
    const first = loader.loadLevel(1)
    gates[0].resolve()
    await first
    expect(calls.commits).toBe(1)

    const rebuild = loader.loadLevel(1, { reuseArray: true })
    gates[1].resolve()
    await rebuild
    // resolveArray reported reusedArray: true -> no cache reset.
    expect(calls.commits).toBe(1)
    expect(calls.resolveArray[1]).toEqual({ levelIndex: 1, reuse: true })
  })

  it('cancels in-flight region fetches at the start of every load', async () => {
    const { loader, calls, gates } = makeHarness()
    loader.desiredIndex = 1
    const load = loader.loadLevel(1)
    gates[0].resolve()
    await load
    expect(calls.cancels).toBe(1)
  })

  it('dispose during a load drops the commit and clears state', async () => {
    const { loader, gates } = makeHarness()
    loader.desiredIndex = 1
    const load = loader.loadLevel(1)
    loader.dispose()
    gates[0].resolve()
    await load
    expect(loader.active).toBeNull()
    expect(loader.loadingIndex).toBeNull()
  })

  it('a stale load does not clobber the newer load state', async () => {
    const { loader, gates } = makeHarness()
    loader.desiredIndex = 1
    const first = loader.loadLevel(1)
    loader.desiredIndex = 2
    const second = loader.loadLevel(2)
    expect(loader.loadingIndex).toBe(2)

    // The stale first load finishing must not null the newer loadingIndex.
    gates[0].resolve()
    await first
    expect(loader.loadingIndex).toBe(2)

    gates[1].resolve()
    await second
    expect(loader.active?.index).toBe(2)
    expect(loader.loadingIndex).toBeNull()
  })

  it('latches a selector-resolution failure: one attempt, one error log', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    try {
      const { loader, calls, gates } = makeHarness({
        buildSliceArgs: async () => {
          throw new SelectorResolutionError('bad band name')
        },
      })
      loader.desiredIndex = 1
      const first = loader.loadLevel(1)
      gates[0].resolve()
      expect(await first).toBe('failed')
      expect(errorSpy).toHaveBeenCalledTimes(1)
      expect(calls.errors).toHaveLength(1)
      expect(calls.errors[0]).toBeInstanceOf(SelectorResolutionError)

      expect(await loader.loadLevel(1)).toBe('ignored')
      expect(await loader.loadLevel(1)).toBe('ignored')
      expect(calls.resolveArray).toHaveLength(1)
      expect(errorSpy).toHaveBeenCalledTimes(1)
      expect(calls.errors).toHaveLength(1)

      loader.desiredIndex = 2
      const other = loader.loadLevel(2)
      expect(calls.resolveArray).toHaveLength(2)
      gates[1].resolve()
      expect(await other).toBe('failed')
      expect(errorSpy).toHaveBeenCalledTimes(2)
    } finally {
      errorSpy.mockRestore()
    }
  })

  it('retries a non-selector failure on the next call', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    try {
      const { loader, calls, gates } = makeHarness()
      loader.desiredIndex = 1
      const first = loader.loadLevel(1)
      gates[0].reject(new Error('503'))
      expect(await first).toBe('failed')
      expect(errorSpy).toHaveBeenCalledTimes(1)
      expect(calls.errors[0]?.message).toBe('503')

      const second = loader.loadLevel(1)
      expect(calls.resolveArray).toHaveLength(2)
      gates[1].reject(new Error('503'))
      expect(await second).toBe('failed')
      // The same failure again is neither re-reported nor re-logged.
      expect(calls.errors).toHaveLength(1)
      expect(errorSpy).toHaveBeenCalledTimes(1)

      const third = loader.loadLevel(1)
      expect(calls.resolveArray).toHaveLength(3)
      gates[2].resolve()
      expect(await third).toBe('committed')
      expect(loader.active?.index).toBe(1)
      expect(calls.errors).toEqual([expect.any(Error), null])
    } finally {
      errorSpy.mockRestore()
    }
  })

  it('clearSelectorFailures lifts the latch so a new selector can load', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    try {
      let resolvable = false
      const { loader, calls, gates } = makeHarness({
        buildSliceArgs: async () => {
          if (!resolvable) throw new SelectorResolutionError('bad band name')
          return { baseSliceArgs: [0, 0], baseMultiValueDims: [] }
        },
      })
      loader.desiredIndex = 1
      const first = loader.loadLevel(1)
      gates[0].resolve()
      expect(await first).toBe('failed')
      expect(await loader.loadLevel(1)).toBe('ignored')

      resolvable = true
      loader.clearSelectorFailures()
      expect(calls.errors[calls.errors.length - 1]).toBeNull()
      const retry = loader.loadLevel(1)
      expect(calls.resolveArray).toHaveLength(2)
      gates[1].resolve()
      expect(await retry).toBe('committed')
      expect(loader.active?.index).toBe(1)
    } finally {
      errorSpy.mockRestore()
    }
  })

  it('does not latch a selector-resolution failure from a stale selector', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    try {
      const bad: NormalizedSelector = {
        band: { selected: 'bleu', type: 'value' },
      }
      const { loader, calls, gates, setSelector } = makeHarness({
        buildSliceArgs: async (selector) => {
          if (selector === bad) throw new SelectorResolutionError('bad band')
          return { baseSliceArgs: [0, 0], baseMultiValueDims: [] }
        },
      })
      setSelector(bad)
      loader.desiredIndex = 1
      const first = loader.loadLevel(1)

      // The replacement deduplicates onto the in-flight load.
      setSelector({ band: { selected: 'blue', type: 'value' } })
      loader.clearSelectorFailures()
      const replacement = loader.loadLevel(1)
      expect(replacement).toBe(first)
      gates[0].resolve()
      expect(await first).toBe('superseded')
      expect(errorSpy).not.toHaveBeenCalled()
      expect(calls.errors).toEqual([])

      const retry = loader.loadLevel(1)
      expect(calls.resolveArray).toHaveLength(2)
      gates[1].resolve()
      expect(await retry).toBe('committed')
      expect(loader.active?.index).toBe(1)
    } finally {
      errorSpy.mockRestore()
    }
  })

  it('drops the committed level when a selector rebuild fails', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    try {
      let fail = false
      const { loader, gates } = makeHarness({
        buildSliceArgs: async () => {
          if (fail) throw new SelectorResolutionError('bad band')
          return { baseSliceArgs: [0, 0], baseMultiValueDims: [] }
        },
      })
      loader.desiredIndex = 1
      const first = loader.loadLevel(1)
      gates[0].resolve()
      expect(await first).toBe('committed')

      fail = true
      const rebuild = loader.loadLevel(1, { reuseArray: true })
      gates[1].resolve()
      expect(await rebuild).toBe('failed')
      expect(loader.active).toBeNull()
      expect(await loader.loadLevel(1)).toBe('ignored')
    } finally {
      errorSpy.mockRestore()
    }
  })

  it('reports the same error instance again when a different level fails', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    try {
      const shared = new Error('store offline')
      const { loader, calls, gates } = makeHarness()
      loader.desiredIndex = 1
      const first = loader.loadLevel(1)
      gates[0].reject(shared)
      expect(await first).toBe('failed')

      loader.desiredIndex = 2
      const second = loader.loadLevel(2)
      gates[1].reject(shared)
      expect(await second).toBe('failed')

      expect(errorSpy).toHaveBeenCalledTimes(2)
      expect(calls.errors).toEqual([shared, shared])
    } finally {
      errorSpy.mockRestore()
    }
  })

  it('logs errors only for the load that still owns the token', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    try {
      const { loader, gates } = makeHarness()
      loader.desiredIndex = 1
      const first = loader.loadLevel(1)
      const second = loader.loadLevel(1, { reuseArray: true })

      // Stale load fails: silent.
      gates[0].reject(new Error('network down'))
      await first
      expect(errorSpy).not.toHaveBeenCalled()

      // Current load fails: logged.
      gates[1].reject(new Error('network down'))
      await second
      expect(errorSpy).toHaveBeenCalledTimes(1)
      expect(loader.loadingIndex).toBeNull()
    } finally {
      errorSpy.mockRestore()
    }
  })
})
