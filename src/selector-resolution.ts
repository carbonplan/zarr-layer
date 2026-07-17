import * as zarr from 'zarrita'
import type { DimIndicesProps, NormalizedSelector, UntiledLevel } from './types'
import { loadDimensionValues } from './zarr-utils'
import type { ZarrStore } from './zarr-store'

export type DimensionValuesCache = {
  [key: string]: Float64Array | number[] | string[]
}

export type SelectorResolutionContext = {
  zarrStore: ZarrStore
  dimIndices: DimIndicesProps
  levels: UntiledLevel[]
  isMultiscale: boolean
  dimensionValues: DimensionValuesCache
  coordLevelIndex: number
}

/**
 * Classify a dimension by its name.
 * Used to identify spatial (lat/lon) vs non-spatial dimensions.
 */
export function classifyDimension(
  dimKey: string
): 'lon' | 'lat' | 'time' | 'other' {
  const key = dimKey.toLowerCase()
  if (key === 'lon' || key === 'x' || key === 'lng' || key.includes('lon')) {
    return 'lon'
  }
  if (key === 'lat' || key === 'y' || key.includes('lat')) {
    return 'lat'
  }
  if (key.includes('time')) {
    return 'time'
  }
  return 'other'
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

  for (const dimName of Object.keys(context.dimIndices)) {
    const dimInfo = context.dimIndices[dimName]
    const dimType = classifyDimension(dimName)

    if (dimType === 'lon') {
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
    } else if (dimType === 'lat') {
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
      const selectionSpec =
        selector[dimName] || (dimType === 'time' ? selector['time'] : undefined)

      if (selectionSpec !== undefined) {
        const selectionValue = selectionSpec.selected
        const selectionType = selectionSpec.type

        if (
          options.trackMultiValue &&
          Array.isArray(selectionValue) &&
          selectionValue.length > 1
        ) {
          const resolvedIndices: number[] = []
          const labelValues: (number | string)[] = []
          for (const val of selectionValue) {
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

  // Multiscale pyramids (tiled and untiled alike) keep their non-spatial
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
    // Value not present in the preloaded coordinate array: treat a numeric
    // selector as a direct index.
    return typeof value === 'number' ? value : 0
  }

  if (!context.zarrStore.root) {
    return typeof value === 'number' ? value : 0
  }

  // Multiscale pyramids keep per-dimension coordinate arrays under per-level
  // paths (e.g. `0/band/c/0`), not at the root. Passing `null` here resolves
  // the coordinate array from the root, which 404s for those datasets; the
  // catch below then falls back to index 0 for every selector value. An RGB
  // selection (red/green/blue by name) therefore collapses all three channels
  // to band 0 and renders greyscale. Resolve the in-flight level's path
  // instead, gated on `isMultiscale` so single-level datasets (coordinates at
  // root) are unaffected. `loadLevel` calls `buildSliceArgsForSelector`
  // (and so this) before committing `activeLevel`, so fall through
  // activeLevel.index -> loadingLevelIndex -> desiredLevelIndex -> 0 to use
  // the level the in-flight load is actually targeting.
  let levelInfo: string | null = null
  if (context.isMultiscale && context.levels.length > 0) {
    const safeIdx = Math.max(
      0,
      Math.min(context.coordLevelIndex, context.levels.length - 1)
    )
    levelInfo = context.levels[safeIdx]?.asset ?? null
  }

  try {
    const coords = await loadDimensionValues(
      context.dimensionValues,
      levelInfo,
      dimInfo,
      context.zarrStore.root,
      context.zarrStore.version
    )
    context.dimensionValues[dimName] = coords

    const coordIdx = (coords as (number | string)[]).indexOf(value)
    if (coordIdx >= 0) return coordIdx
    throw new Error(
      `[ZarrLayer] Selector value '${value}' not found in coordinate array for dimension '${dimName}'. ` +
        `Available values: [${(coords as (number | string)[])
          .slice(0, 10)
          .join(', ')}${coords.length > 10 ? ', ...' : ''}]. ` +
        `Use { selected: <index>, type: 'index' } to select by array index instead.`
    )
  } catch (err) {
    console.debug(`Could not resolve coordinate for '${dimName}':`, err)
  }

  return typeof value === 'number' ? value : 0
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
