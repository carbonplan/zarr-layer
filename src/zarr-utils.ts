/**
 * @module zarr-utils
 *
 * Utility functions for reading, interpreting, and slicing Zarr datasets
 * adapted from zarr-cesium/src/zarr-utils.ts
 */

import * as zarr from 'zarrita'
import {
  type SelectorSpec,
  type SelectorValue,
  type NormalizedSelector,
  type Selector,
  type DimensionNamesProps,
  type DimIndicesProps,
} from './types'

type CoordinateArray = zarr.Array<zarr.DataType> & {
  attrs?: Record<string, unknown>
}
type CoordinatesMap = Record<string, CoordinateArray>

const resolveOpenFunc = (zarrVersion: 2 | 3 | null): typeof zarr.open => {
  if (zarrVersion === 2) return zarr.open.v2 as typeof zarr.open
  if (zarrVersion === 3) return zarr.open.v3 as typeof zarr.open
  return zarr.open
}

const DIMENSION_ALIASES_DEFAULT: {
  [key in keyof DimensionNamesProps]: string[]
} = {
  lat: ['lat', 'latitude', 'y', 'Latitude', 'Y'],
  lon: ['lon', 'longitude', 'x', 'Longitude', 'X', 'lng'],
  time: ['time', 't', 'Time', 'time_counter'],
  elevation: [
    'depth',
    'z',
    'Depth',
    'level',
    'lev',
    'deptht',
    'elevation',
    'depthu',
    'depthv',
  ],
}

const CF_MAPPINGS: { [key in keyof DimensionNamesProps]: string[] } = {
  lat: ['latitude'],
  lon: ['longitude'],
  time: ['time'],
  elevation: [
    'height',
    'depth',
    'altitude',
    'air_pressure',
    'pressure',
    'geopotential_height',
  ],
} as const

/**
 * Identify the indices of common dimensions (lat, lon, time, elevation)
 * in a Zarr array, optionally using CF-compliant standard names or custom dimension mappings.
 *
 * @param dimNames - Names of the array dimensions.
 * @param dimensionNames - Optional explicit mapping of dimension names (see {@link DimensionNamesProps}).
 * @param coordinates - Optional coordinate variable dictionary.
 * @returns A {@link DimIndicesProps} object describing each dimension’s index and name.
 */
export function identifyDimensionIndices(
  dimNames: string[],
  dimensionNames?: DimensionNamesProps,
  coordinates?: CoordinatesMap
): DimIndicesProps {
  let DIMENSION_ALIASES = { ...DIMENSION_ALIASES_DEFAULT }
  const names = ['lat', 'lon', 'time', 'elevation']

  if (coordinates) {
    Object.keys(coordinates).forEach((coordName) => {
      const coordArr = coordinates[coordName]
      const coordAttrs = (coordArr.attrs ?? {}) as Record<string, unknown>
      const standardName =
        typeof coordAttrs?.standard_name === 'string'
          ? coordAttrs.standard_name
          : undefined
      if (standardName) {
        for (const [dimKey, cfNames] of Object.entries(CF_MAPPINGS)) {
          if (cfNames.includes(standardName)) {
            DIMENSION_ALIASES[dimKey as keyof DimensionNamesProps] = [coordName]
          }
        }
      }
    })
  }

  if (dimensionNames) {
    names.forEach((name) => {
      const dimName = name as keyof DimensionNamesProps
      if (dimensionNames[dimName]) {
        DIMENSION_ALIASES[dimName] = [dimensionNames[dimName]] as string[]
      }
    })
    if (dimensionNames.others) {
      dimensionNames.others.forEach((otherName) => {
        DIMENSION_ALIASES[otherName as keyof DimensionNamesProps] = [otherName]
      })
    }
  }

  const indices: DimIndicesProps = {}
  for (const [key, aliases] of Object.entries(DIMENSION_ALIASES)) {
    for (let i = 0; i < dimNames.length; i++) {
      const name = dimNames[i].toLowerCase()
      if (aliases.map((a) => a.toLowerCase()).includes(name)) {
        indices[key] = {
          name,
          index: i,
          array: coordinates ? coordinates[dimNames[i]] : null,
        }
        break
      }
    }
  }
  return indices
}

/**
 * Finds the index of the value in `values` nearest to `target`.
 * @param values - Array of numeric values.
 * @param target - Target value to find.
 * @returns Index of the nearest value.
 */
export function calculateNearestIndex(
  values: Float64Array | number[],
  target: number
): number {
  const selectedValue = target
  let nearestIdx = 0
  let minDiff = Infinity
  values.forEach((val, i) => {
    const diff = Math.abs(val - selectedValue)
    if (diff < minDiff) {
      minDiff = diff
      nearestIdx = i
    }
  })
  return nearestIdx
}

/**
 * Loads the coordinate values for a specific dimension.
 *
 * Behavior:
 * - Uses cached values if available (does not reload unless the caller resets the cache).
 * - Resolves the correct multiscale level if `levelInfo` is provided.
 * - Converts Zarr buffers into plain JavaScript number arrays.
 * - Converts bigint values to number.
 * - If a slice `[start, end]` is supplied, only a sub-range is returned.
 *
 * @param dimensionValues  Cache of already-loaded coordinate arrays.
 * @param levelInfo        Optional multiscale subpath.
 * @param dimIndices      Dimension index info. See {@link DimIndicesProps}.
 * @param root            Root Zarr group location.
 * @param zarrVersion     Zarr version (2 or 3).
 * @param slice           Optional index range `[start, end]` to slice the loaded values.
 *
 * @returns The loaded coordinate array for the dimension.
 */
export async function loadDimensionValues(
  dimensionValues: Record<string, Float64Array | number[]>,
  levelInfo: string | null,
  dimIndices: DimIndicesProps[string],
  root: zarr.Location<zarr.FetchStore>,
  zarrVersion: 2 | 3 | null,
  slice?: [number, number]
): Promise<Float64Array | number[]> {
  if (dimensionValues[dimIndices.name]) return dimensionValues[dimIndices.name]
  let targetRoot
  if (levelInfo) {
    targetRoot = await root.resolve(levelInfo)
  } else {
    targetRoot = root
  }
  let coordArr
  if (dimIndices.array) {
    coordArr = dimIndices.array
  } else {
    const coordVar = await targetRoot.resolve(dimIndices.name)
    const localFunc = resolveOpenFunc(zarrVersion)
    coordArr = await localFunc(coordVar, { kind: 'array' })
  }
  const coordData = await zarr.get(coordArr)
  const coordArray = Array.from(
    coordData.data as ArrayLike<number | bigint>,
    (v: number | bigint) => (typeof v === 'bigint' ? Number(v) : v)
  )
  if (slice) {
    return coordArray.slice(slice[0], slice[1])
  }
  return coordArray
}

interface BandInfo {
  band: number | string
  index: number
}

function getBandInformation(
  selector: NormalizedSelector
): Record<string, BandInfo> {
  const result: Record<string, BandInfo> = {}

  for (const [key, value] of Object.entries(selector)) {
    const selected = value?.selected
    const normalized = Array.isArray(selected) ? selected : null

    if (normalized && Array.isArray(normalized)) {
      normalized.forEach((v, idx) => {
        const bandValue = v as string | number
        const bandName =
          typeof bandValue === 'string' ? bandValue : `${key}_${bandValue}`
        result[bandName] = { band: bandValue, index: idx }
      })
    }
  }

  return result
}

export function getBands(
  variable: string,
  selector: NormalizedSelector
): string[] {
  const bandInfo = getBandInformation(selector)
  const bandNames = Object.keys(bandInfo)

  if (bandNames.length === 0) {
    return [variable]
  }

  return bandNames
}

export function toSelectorProps(
  value: SelectorValue | SelectorSpec
): SelectorSpec {
  if (
    value &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    'selected' in value
  ) {
    const normalized = value as SelectorSpec
    return {
      selected: normalized.selected,
      type: normalized.type ?? 'value',
    }
  }

  return { selected: value as SelectorValue, type: 'value' }
}

export function normalizeSelector(selector: Selector): NormalizedSelector {
  return (
    Object.entries(selector) as [string, SelectorValue | SelectorSpec][]
  ).reduce((acc, [dimName, value]) => {
    acc[dimName] = toSelectorProps(value)
    return acc
  }, {} as NormalizedSelector)
}

/**
 * Resolves a selector value for a dimension, handling the common pattern of
 * looking up by dimension key, dimension name, or the name stored in dimIndices.
 *
 * This consolidates the triple-fallback pattern used throughout the codebase:
 * ```
 * selector[dimKey] ?? selector[dimName] ?? selector[dimIndices[dimKey]?.name]
 * ```
 *
 * @param selector - The normalized selector object
 * @param dimKey - The canonical dimension key (e.g., 'lat', 'lon', 'time')
 * @param dimName - The actual dimension name from the dataset
 * @param dimIndices - Optional dimension indices mapping
 * @returns The resolved SelectorSpec or undefined if not found
 */
export function resolveSelectorValue(
  selector: NormalizedSelector,
  dimKey: string,
  dimName?: string,
  dimIndices?: DimIndicesProps
): SelectorSpec | undefined {
  // Try canonical key first (e.g., 'time', 'lat')
  if (selector[dimKey] !== undefined) {
    return selector[dimKey]
  }

  // Try the actual dimension name from the dataset
  if (dimName && selector[dimName] !== undefined) {
    return selector[dimName]
  }

  // Try the name stored in dimIndices for this key
  if (dimIndices && dimIndices[dimKey]?.name) {
    const indexedName = dimIndices[dimKey].name
    if (selector[indexedName] !== undefined) {
      return selector[indexedName]
    }
  }

  return undefined
}
