import * as zarr from 'zarrita'
import type {
  DimIndicesProps,
  NormalizedSelector,
  ResolutionLevel,
} from './types'
import { loadDimensionValues } from './zarr-utils'
import type { ZarrStore } from './zarr-store'

export type DimensionValuesCache = {
  [key: string]: Float64Array | number[] | string[]
}

export type SelectorResolutionContext = {
  zarrStore: ZarrStore
  dimIndices: DimIndicesProps
  levels: ResolutionLevel[]
  isMultiscale: boolean
  dimensionValues: DimensionValuesCache
  coordLevelIndex: number
  /** Dimensions already warned about after coordinate-read failures. */
  warnedDimensions: Set<string>
}

/**
 * Reject selector keys that name no selectable dimension. The spatial axes
 * are filed under the `lat` and `lon` keys of `dimIndices`, and every other
 * dimension under its own name, which is the only name a selector can use.
 */
export function assertSelectorKeysAreDimensions(
  selector: NormalizedSelector,
  dimIndices: DimIndicesProps
): void {
  const selectable = Object.keys(dimIndices).filter(
    (key) => key !== 'lat' && key !== 'lon'
  )
  const unknown = Object.keys(selector).filter(
    (key) => !selectable.includes(key)
  )
  if (unknown.length === 0) return
  throw new SelectorResolutionError(
    `[ZarrLayer] selector ${unknown
      .map((k) => `'${k}'`)
      .join(
        ', '
      )} does not name a dimension of this variable. Selectable dimensions: [${selectable.join(
      ', '
    )}]`
  )
}

/**
 * The values a selection spreads across channels, or null when it picks a
 * single index. Rendering needs two or more values to pack channels. Queries
 * label every non-empty array, with repeats collapsed.
 */
function multiValueSelection<T>(
  selected: T | T[],
  queryLabelling?: boolean
): T[] | null {
  if (!Array.isArray(selected)) return null
  if (!queryLabelling) return selected.length > 1 ? selected : null
  const unique = [...new Set(selected)]
  return unique.length > 0 ? unique : null
}

/**
 * The dimensions a query selector spreads across labelled series, and their
 * labels, under the same rules `buildSliceArgsForSelector` applies to a read.
 */
export function findQueryMultiValueDims(
  selector: NormalizedSelector,
  dimIndices: DimIndicesProps
): Array<{ dimName: string; labels: (number | string)[] }> {
  const result: Array<{ dimName: string; labels: (number | string)[] }> = []
  for (const dimName of Object.keys(dimIndices)) {
    if (dimName === 'lat' || dimName === 'lon') continue
    const spec = selector[dimName]
    if (spec === undefined) continue
    const labels = multiValueSelection(spec.selected, true)
    if (labels) result.push({ dimName, labels })
  }
  return result
}

/**
 * Build slice arguments from a selector for all dimensions.
 * Shared logic used by both display (buildBaseSliceArgs) and queries (fetchDataForSelector).
 */
export async function buildSliceArgsForSelector(
  context: SelectorResolutionContext,
  selector: NormalizedSelector,
  options: {
    /** If true, set spatial dims to full slices; if false, set to 0 placeholder */
    includeSpatialSlices: boolean
    /** If true, track multi-value dimensions for channel packing */
    trackMultiValue: boolean
    /**
     * Query semantics for array selectors: any non-empty array is multi-value,
     * so it always yields labelled channels whatever its length, and repeated
     * values collapse to one channel so each label names one series.
     */
    queryLabelling?: boolean
    /** Spatial bounds for fetch - bbox for region subset */
    spatialBounds?: {
      minX: number
      maxX: number
      minY: number
      maxY: number
    }
    /**
     * Array to derive shape from. Caller pins this so we don't read
     * `this.activeLevel?.zarrArray` mid-flight (which can swap during
     * `loadLevel` or zoom). Pass the new array during `loadLevel`, or
     * a snapshot of the active array for query/render paths.
     */
    array: zarr.Array<zarr.DataType>
  }
): Promise<{
  sliceArgs: (number | zarr.Slice)[]
  multiValueDims: Array<{
    dimIndex: number
    dimName: string
    values: number[]
    labels: (number | string)[]
  }>
}> {
  const { array } = options
  const sliceArgs: (number | zarr.Slice)[] = new Array(array.shape.length).fill(
    0
  )
  const multiValueDims: Array<{
    dimIndex: number
    dimName: string
    values: number[]
    labels: (number | string)[]
  }> = []

  assertSelectorKeysAreDimensions(selector, context.dimIndices)

  for (const dimName of Object.keys(context.dimIndices)) {
    const dimInfo = context.dimIndices[dimName]

    if (dimName === 'lon') {
      if (options.spatialBounds) {
        sliceArgs[dimInfo.index] = zarr.slice(
          options.spatialBounds.minX,
          options.spatialBounds.maxX
        )
      } else {
        sliceArgs[dimInfo.index] = options.includeSpatialSlices
          ? zarr.slice(0, array.shape[dimInfo.index] ?? 0)
          : 0
      }
    } else if (dimName === 'lat') {
      if (options.spatialBounds) {
        sliceArgs[dimInfo.index] = zarr.slice(
          options.spatialBounds.minY,
          options.spatialBounds.maxY
        )
      } else {
        sliceArgs[dimInfo.index] = options.includeSpatialSlices
          ? zarr.slice(0, array.shape[dimInfo.index] ?? 0)
          : 0
      }
    } else {
      const selectionSpec = selector[dimName]

      if (selectionSpec !== undefined) {
        const selectionValue = selectionSpec.selected
        const selectionType = selectionSpec.type
        const multiValues = options.trackMultiValue
          ? multiValueSelection(selectionValue, options.queryLabelling)
          : null

        if (multiValues) {
          const resolvedIndices: number[] = []
          const labelValues: (number | string)[] = []
          for (const val of multiValues) {
            const idx = await resolveSelectionIndex(
              context,
              dimName,
              dimInfo,
              val,
              selectionType
            )
            resolvedIndices.push(idx)
            labelValues.push(val)
          }
          multiValueDims.push({
            dimIndex: dimInfo.index,
            dimName,
            values: resolvedIndices,
            labels: labelValues,
          })
          sliceArgs[dimInfo.index] = resolvedIndices[0]
        } else {
          const primaryValue = Array.isArray(selectionValue)
            ? selectionValue[0]
            : selectionValue
          sliceArgs[dimInfo.index] = await resolveSelectionIndex(
            context,
            dimName,
            dimInfo,
            primaryValue,
            selectionType
          )
        }
      } else {
        sliceArgs[dimInfo.index] = 0
      }
    }
  }

  return { sliceArgs, multiValueDims }
}

/** A selector value was not found in an available coordinate array. */
export class SelectorResolutionError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'SelectorResolutionError'
  }
}

function unresolvedSelectionIndex(
  dimName: string,
  value: number | string,
  coords: (number | string)[] | null
): number {
  if (typeof value === 'number') return value
  const detail = coords
    ? `Available values: [${coords.slice(0, 10).join(', ')}${
        coords.length > 10 ? ', ...' : ''
      }]. `
    : `The store has no root to read the coordinate array from, so string values cannot be matched. `
  throw new SelectorResolutionError(
    `[ZarrLayer] Selector value '${value}' not found in coordinate array for dimension '${dimName}'. ` +
      detail +
      `Use { selected: <index>, type: 'index' } to select by array index instead.`
  )
}

/** Coordinate reads may recover, so this deliberately avoids the latching error. */
function unreadableCoordinateIndex(
  dimName: string,
  value: number | string,
  cause: unknown
): number {
  if (typeof value === 'number') return value
  const reason = cause instanceof Error ? cause.message : String(cause)
  throw new Error(
    `[ZarrLayer] Selector value '${value}' for dimension '${dimName}' cannot be matched: ` +
      `the coordinate array could not be read (${reason}). ` +
      `Use { selected: <index>, type: 'index' } to select by array index instead.`
  )
}

export async function resolveSelectionIndex(
  context: SelectorResolutionContext,
  dimName: string,
  dimInfo: {
    index: number
    name: string
    array: zarr.Array<zarr.DataType> | null
  },
  value: number | string | [number, number] | undefined,
  type?: 'index' | 'value'
): Promise<number> {
  if (type === 'index') {
    return typeof value === 'number' ? value : 0
  }
  if (typeof value !== 'number' && typeof value !== 'string') {
    return 0
  }

  // Resolution pyramids keep their non-spatial
  // coordinate arrays inside each level directory (e.g. "0/month"), not at
  // the store root. ZarrStore preloads those from the level-0 directory into
  // `coordinates`, so prefer them — opening the same arrays at the root (as
  // the fallback below does) would 404. The fallback covers single-level
  // datasets, whose coordinate arrays live at the root and aren't preloaded.
  const storeCoords = context.zarrStore.coordinates[dimName] as
    | (number | string)[]
    | undefined
  if (storeCoords && storeCoords.length > 0) {
    const idx = storeCoords.indexOf(value)
    if (idx >= 0) return idx
    return unresolvedSelectionIndex(dimName, value, storeCoords)
  }

  if (!context.zarrStore.root) {
    return unresolvedSelectionIndex(dimName, value, null)
  }

  // Multiscale coordinate arrays live beneath each level; single-level
  // coordinates remain at the root.
  let levelInfo: string | null = null
  if (context.isMultiscale && context.levels.length > 0) {
    const safeIdx = Math.max(
      0,
      Math.min(context.coordLevelIndex, context.levels.length - 1)
    )
    levelInfo = context.levels[safeIdx]?.asset ?? null
  }

  let coords: (number | string)[]
  try {
    const loaded = await loadDimensionValues(
      context.dimensionValues,
      levelInfo,
      dimInfo,
      context.zarrStore.root,
      context.zarrStore.version
    )
    context.dimensionValues[dimName] = loaded
    coords = loaded as (number | string)[]
  } catch (err) {
    // Failed reads are retried, so warn once per dimension.
    if (!context.warnedDimensions.has(dimName)) {
      context.warnedDimensions.add(dimName)
      console.warn(
        `[zarr-layer] Failed to load coordinate array for dimension '${dimName}':`,
        err
      )
    }
    return unreadableCoordinateIndex(dimName, value, err)
  }

  const coordIdx = coords.indexOf(value)
  if (coordIdx >= 0) return coordIdx
  return unresolvedSelectionIndex(dimName, value, coords)
}

/**
 * Build all index combinations from multi-value dimensions.
 * Returns cartesian product of all dimension value arrays.
 */
export function buildChannelCombinations(
  multiValueDims: Array<{ values: number[]; labels: (number | string)[] }>
): { combinations: number[][]; labelCombinations: (number | string)[][] } {
  let combinations: number[][] = [[]]
  let labelCombinations: (number | string)[][] = [[]]

  for (const { values, labels } of multiValueDims) {
    const nextCombos: number[][] = []
    const nextLabels: (number | string)[][] = []
    for (let idx = 0; idx < values.length; idx++) {
      for (let c = 0; c < combinations.length; c++) {
        nextCombos.push([...combinations[c], values[idx]])
        nextLabels.push([...labelCombinations[c], labels[idx]])
      }
    }
    combinations = nextCombos
    labelCombinations = nextLabels
  }

  return { combinations, labelCombinations }
}
