import { describe, it, expect, vi, beforeEach } from 'vitest'
import * as zarr from 'zarrita'
import {
  buildChannelCombinations,
  buildSliceArgsForSelector,
  resolveSelectionIndex,
  SelectorResolutionError,
  type SelectorResolutionContext,
} from './selector-resolution'
import { loadDimensionValues } from './zarr-utils'
import type { ZarrStore } from './zarr-store'

vi.mock('./zarr-utils', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./zarr-utils')>()),
  loadDimensionValues: vi.fn(),
}))

const mockedLoadDimensionValues = vi.mocked(loadDimensionValues)

beforeEach(() => {
  mockedLoadDimensionValues.mockReset()
})

function makeContext(
  overrides: Partial<SelectorResolutionContext> = {}
): SelectorResolutionContext {
  return {
    zarrStore: {
      coordinates: {},
      root: null,
      version: 3,
    } as unknown as ZarrStore,
    dimIndices: {
      time: { index: 0, name: 'time', array: null },
      lat: { index: 1, name: 'lat', array: null },
      lon: { index: 2, name: 'lon', array: null },
    },
    levels: [],
    isMultiscale: false,
    dimensionValues: {},
    coordLevelIndex: 0,
    warnedDimensions: new Set(),
    ...overrides,
  }
}

const DIM_INFO = { index: 0, name: 'time', array: null }

describe('resolveSelectionIndex', () => {
  it('passes numeric values through for type "index"', async () => {
    const context = makeContext()
    expect(
      await resolveSelectionIndex(context, 'time', DIM_INFO, 3, 'index')
    ).toBe(3)
    expect(
      await resolveSelectionIndex(context, 'time', DIM_INFO, undefined, 'index')
    ).toBe(0)
  })

  it('prefers the store-preloaded coordinate arrays', async () => {
    const context = makeContext({
      zarrStore: {
        coordinates: { time: [2000, 2001, 2002] },
        root: {},
        version: 3,
      } as unknown as ZarrStore,
    })
    expect(await resolveSelectionIndex(context, 'time', DIM_INFO, 2001)).toBe(1)
    expect(mockedLoadDimensionValues).not.toHaveBeenCalled()
  })

  it('treats a numeric value missing from preloaded coords as an index', async () => {
    const context = makeContext({
      zarrStore: {
        coordinates: { time: [2000, 2001] },
        root: {},
        version: 3,
      } as unknown as ZarrStore,
    })
    expect(await resolveSelectionIndex(context, 'time', DIM_INFO, 7)).toBe(7)
  })

  it('throws for a string missing from preloaded coords, listing values', async () => {
    const context = makeContext({
      zarrStore: {
        coordinates: { band: ['red', 'green', 'blue'] },
        root: {},
        version: 3,
      } as unknown as ZarrStore,
    })
    await expect(
      resolveSelectionIndex(context, 'band', DIM_INFO, 'bleu')
    ).rejects.toThrow(/'bleu'.*'band'.*red, green, blue/)
    await expect(
      resolveSelectionIndex(context, 'band', DIM_INFO, 'bleu')
    ).rejects.toBeInstanceOf(SelectorResolutionError)
  })

  it('treats a number as a direct index without a store root', async () => {
    const context = makeContext()
    expect(await resolveSelectionIndex(context, 'time', DIM_INFO, 4)).toBe(4)
  })

  it('throws for a string without a store root', async () => {
    const context = makeContext()
    await expect(
      resolveSelectionIndex(context, 'time', DIM_INFO, 'jan')
    ).rejects.toThrow(/'jan'.*'time'/)
  })

  it('resolves against root coordinate arrays for single-level datasets', async () => {
    mockedLoadDimensionValues.mockResolvedValue([10, 20, 30])
    const context = makeContext({
      zarrStore: {
        coordinates: {},
        root: {},
        version: 3,
      } as unknown as ZarrStore,
    })
    expect(await resolveSelectionIndex(context, 'time', DIM_INFO, 20)).toBe(1)
    // Single-level: no level path, coords open at the root.
    expect(mockedLoadDimensionValues.mock.calls[0][1]).toBeNull()
    // Resolved coords are cached for subsequent lookups.
    expect(context.dimensionValues['time']).toEqual([10, 20, 30])
  })

  it('opens coords under the coordLevelIndex level for multiscale datasets', async () => {
    mockedLoadDimensionValues.mockResolvedValue(['red', 'green', 'blue'])
    const context = makeContext({
      isMultiscale: true,
      levels: [{ asset: '0' }, { asset: '1' }],
      coordLevelIndex: 1,
      zarrStore: {
        coordinates: {},
        root: {},
        version: 3,
      } as unknown as ZarrStore,
    })
    expect(
      await resolveSelectionIndex(context, 'band', DIM_INFO, 'green')
    ).toBe(1)
    expect(mockedLoadDimensionValues.mock.calls[0][1]).toBe('1')
  })

  it('clamps an out-of-range coordLevelIndex to the last level', async () => {
    mockedLoadDimensionValues.mockResolvedValue([1, 2])
    const context = makeContext({
      isMultiscale: true,
      levels: [{ asset: '0' }, { asset: '1' }],
      coordLevelIndex: 9,
      zarrStore: {
        coordinates: {},
        root: {},
        version: 3,
      } as unknown as ZarrStore,
    })
    await resolveSelectionIndex(context, 'band', DIM_INFO, 2)
    expect(mockedLoadDimensionValues.mock.calls[0][1]).toBe('1')
  })

  it('warns and treats a number as a direct index when the coordinate load fails', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    try {
      mockedLoadDimensionValues.mockRejectedValue(new Error('404'))
      const context = makeContext({
        zarrStore: {
          coordinates: {},
          root: {},
          version: 3,
        } as unknown as ZarrStore,
      })
      expect(await resolveSelectionIndex(context, 'time', DIM_INFO, 5)).toBe(5)
      expect(warnSpy).toHaveBeenCalledTimes(1)
      expect(warnSpy.mock.calls[0][0]).toContain('[zarr-layer]')
      expect(warnSpy.mock.calls[0][0]).toContain("'time'")
    } finally {
      warnSpy.mockRestore()
    }
  })

  it('warns and throws a non-latching error when coordinate loading fails', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    try {
      mockedLoadDimensionValues.mockRejectedValue(
        new Error('Unknown or unsupported dataType')
      )
      const context = makeContext({
        zarrStore: {
          coordinates: {},
          root: {},
          version: 3,
        } as unknown as ZarrStore,
      })
      await expect(
        resolveSelectionIndex(context, 'band', DIM_INFO, 'red')
      ).rejects.toThrow(/'red'.*'band'.*Unknown or unsupported dataType/)
      await expect(
        resolveSelectionIndex(context, 'band', DIM_INFO, 'red')
      ).rejects.not.toBeInstanceOf(SelectorResolutionError)
      expect(warnSpy).toHaveBeenCalledTimes(1)
    } finally {
      warnSpy.mockRestore()
    }
  })

  it('warns once per dimension across repeated load failures', async () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    try {
      mockedLoadDimensionValues.mockRejectedValue(new Error('404'))
      const context = makeContext({
        zarrStore: {
          coordinates: {},
          root: {},
          version: 3,
        } as unknown as ZarrStore,
      })
      await resolveSelectionIndex(context, 'time', DIM_INFO, 5)
      await resolveSelectionIndex(context, 'time', DIM_INFO, 6)
      expect(warnSpy).toHaveBeenCalledTimes(1)
      await resolveSelectionIndex(context, 'band', DIM_INFO, 2)
      expect(warnSpy).toHaveBeenCalledTimes(2)
    } finally {
      warnSpy.mockRestore()
    }
  })
})

describe('buildSliceArgsForSelector', () => {
  const array = { shape: [5, 180, 360] } as unknown as zarr.Array<zarr.DataType>

  it('sets spatial dims to full slices when requested', async () => {
    const { sliceArgs } = await buildSliceArgsForSelector(
      makeContext(),
      {},
      {
        includeSpatialSlices: true,
        trackMultiValue: false,
        array,
      }
    )
    expect(sliceArgs[0]).toBe(0) // unselected non-spatial dim
    // Full extent of each spatial dim, from the array shape [5, 180, 360].
    expect(sliceArgs[1]).toMatchObject({ start: 0, stop: 180 })
    expect(sliceArgs[2]).toMatchObject({ start: 0, stop: 360 })
  })

  it('sets spatial dims to placeholder 0 when slices are excluded', async () => {
    const { sliceArgs } = await buildSliceArgsForSelector(
      makeContext(),
      {},
      {
        includeSpatialSlices: false,
        trackMultiValue: false,
        array,
      }
    )
    expect(sliceArgs).toEqual([0, 0, 0])
  })

  it('applies explicit spatial bounds as slices', async () => {
    const { sliceArgs } = await buildSliceArgsForSelector(
      makeContext(),
      {},
      {
        includeSpatialSlices: false,
        trackMultiValue: false,
        spatialBounds: { minX: 10, maxX: 20, minY: 30, maxY: 40 },
        array,
      }
    )
    expect(sliceArgs[1]).toMatchObject({ start: 30, stop: 40 })
    expect(sliceArgs[2]).toMatchObject({ start: 10, stop: 20 })
  })

  it('resolves single-value selectors to an index', async () => {
    const { sliceArgs, multiValueDims } = await buildSliceArgsForSelector(
      makeContext(),
      { time: { selected: 3, type: 'index' } },
      { includeSpatialSlices: false, trackMultiValue: true, array }
    )
    expect(sliceArgs[0]).toBe(3)
    expect(multiValueDims).toEqual([])
  })

  it('tracks multi-value dimensions and pins the first index', async () => {
    const { sliceArgs, multiValueDims } = await buildSliceArgsForSelector(
      makeContext(),
      { time: { selected: [1, 3], type: 'index' } },
      { includeSpatialSlices: false, trackMultiValue: true, array }
    )
    expect(sliceArgs[0]).toBe(1)
    expect(multiValueDims).toEqual([
      { dimIndex: 0, dimName: 'time', values: [1, 3], labels: [1, 3] },
    ])
  })

  it('selects a dimension whose name merely contains lat, lon, or time', async () => {
    const array4d = {
      shape: [5, 6, 180, 360],
    } as unknown as zarr.Array<zarr.DataType>
    for (const name of ['simulation', 'longwave', 'population', 'lifetime']) {
      const context = makeContext({
        dimIndices: {
          time: { index: 0, name: 'time', array: null },
          [name]: { index: 1, name, array: null },
          lat: { index: 2, name: 'lat', array: null },
          lon: { index: 3, name: 'lon', array: null },
        },
      })
      const { sliceArgs } = await buildSliceArgsForSelector(
        context,
        { [name]: { selected: 4, type: 'index' } },
        { includeSpatialSlices: false, trackMultiValue: false, array: array4d }
      )
      expect(sliceArgs[1]).toBe(4)
    }
  })

  it('applies a time selector to the time dimension only', async () => {
    const array4d = {
      shape: [5, 6, 180, 360],
    } as unknown as zarr.Array<zarr.DataType>
    const context = makeContext({
      dimIndices: {
        time: { index: 0, name: 'time', array: null },
        lead_time: { index: 1, name: 'lead_time', array: null },
        lat: { index: 2, name: 'lat', array: null },
        lon: { index: 3, name: 'lon', array: null },
      },
    })
    const { sliceArgs } = await buildSliceArgsForSelector(
      context,
      { time: { selected: 3, type: 'index' } },
      { includeSpatialSlices: false, trackMultiValue: false, array: array4d }
    )
    expect(sliceArgs[0]).toBe(3)
    expect(sliceArgs[1]).toBe(0)
  })

  it('treats a one-element array as single-value unless asked to label it', async () => {
    const selector = { time: { selected: [3], type: 'index' as const } }
    const plain = await buildSliceArgsForSelector(makeContext(), selector, {
      includeSpatialSlices: false,
      trackMultiValue: true,
      array,
    })
    expect(plain.sliceArgs[0]).toBe(3)
    expect(plain.multiValueDims).toEqual([])

    const labelled = await buildSliceArgsForSelector(makeContext(), selector, {
      includeSpatialSlices: false,
      trackMultiValue: true,
      labelSingleElementArrays: true,
      array,
    })
    expect(labelled.sliceArgs[0]).toBe(3)
    expect(labelled.multiValueDims).toEqual([
      { dimIndex: 0, dimName: 'time', values: [3], labels: [3] },
    ])
  })

  it('rejects a selector key that names no dimension, listing the real ones', async () => {
    const context = makeContext({
      dimIndices: {
        valid_time: { index: 0, name: 'valid_time', array: null },
        lat: { index: 1, name: 'lat', array: null },
        lon: { index: 2, name: 'lon', array: null },
      },
    })
    const build = buildSliceArgsForSelector(
      context,
      { time: { selected: 2, type: 'index' } },
      { includeSpatialSlices: false, trackMultiValue: false, array }
    )
    await expect(build).rejects.toBeInstanceOf(SelectorResolutionError)
    await expect(build).rejects.toThrow(/'time'.*\[valid_time\]/)
  })

  it('rejects a selector on a spatial axis', async () => {
    await expect(
      buildSliceArgsForSelector(
        makeContext(),
        { lat: { selected: 2, type: 'index' } },
        { includeSpatialSlices: false, trackMultiValue: false, array }
      )
    ).rejects.toBeInstanceOf(SelectorResolutionError)
  })
})

describe('buildChannelCombinations', () => {
  it('yields a single empty combination without multi-value dims', () => {
    expect(buildChannelCombinations([])).toEqual({
      combinations: [[]],
      labelCombinations: [[]],
    })
  })

  it('expands one dimension into per-value combinations', () => {
    const { combinations, labelCombinations } = buildChannelCombinations([
      { values: [4, 9], labels: ['red', 'blue'] },
    ])
    expect(combinations).toEqual([[4], [9]])
    expect(labelCombinations).toEqual([['red'], ['blue']])
  })

  it('builds the cartesian product across dimensions', () => {
    const { combinations } = buildChannelCombinations([
      { values: [1, 2], labels: [1, 2] },
      { values: [10, 20], labels: [10, 20] },
    ])
    expect(combinations).toEqual([
      [1, 10],
      [2, 10],
      [1, 20],
      [2, 20],
    ])
  })
})
