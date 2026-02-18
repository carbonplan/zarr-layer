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
  type SpatialDimensions,
  type DimIndicesProps,
} from './types'
import { SPATIAL_DIMENSION_ALIASES } from './constants'

type CoordinateArray = zarr.Array<zarr.DataType> & {
  attrs?: Record<string, unknown>
}
type CoordinatesMap = Record<string, CoordinateArray>

const resolveOpenFunc = (zarrVersion: 2 | 3 | null): typeof zarr.open => {
  if (zarrVersion === 2) return zarr.open.v2 as typeof zarr.open
  if (zarrVersion === 3) return zarr.open.v3 as typeof zarr.open
  return zarr.open
}

/**
 * Sanitizes a string to be a valid GLSL identifier.
 * GLSL identifiers must start with a letter or underscore,
 * and contain only letters, digits, and underscores.
 *
 * @example
 * sanitizeGlslName('nir:B08') // returns 'nir_B08'
 * sanitizeGlslName('123abc') // returns '_123abc'
 * sanitizeGlslName('band-1') // returns 'band_1'
 */
export function sanitizeGlslName(name: string): string {
  // Replace any non-alphanumeric character (except underscore) with underscore
  let sanitized = name.replace(/[^a-zA-Z0-9_]/g, '_')

  // If it starts with a digit, prefix with underscore
  if (/^[0-9]/.test(sanitized)) {
    sanitized = '_' + sanitized
  }

  return sanitized
}

/**
 * Identify the indices of spatial dimensions (lat, lon) in a Zarr array.
 *
 * Auto-detects common dimension names (lat, latitude, y, lon, longitude, x, lng).
 * Use spatialDimensions to override if your dataset uses non-standard names.
 *
 * @param dimNames - Names of the array dimensions.
 * @param spatialDimensions - Optional explicit mapping for non-standard dimension names.
 * @param coordinates - Optional coordinate variable dictionary (for attaching arrays to dimIndices).
 * @returns A {@link DimIndicesProps} object with lat/lon indices if found.
 */
export function identifyDimensionIndices(
  dimNames: string[],
  spatialDimensions?: SpatialDimensions,
  coordinates?: CoordinatesMap
): DimIndicesProps {
  const aliases: Record<'lat' | 'lon', string[]> = {
    lat: [...SPATIAL_DIMENSION_ALIASES.lat],
    lon: [...SPATIAL_DIMENSION_ALIASES.lon],
  }

  // Apply explicit overrides from spatialDimensions
  if (spatialDimensions?.lat) {
    aliases.lat = [spatialDimensions.lat]
  }
  if (spatialDimensions?.lon) {
    aliases.lon = [spatialDimensions.lon]
  }

  const indices: DimIndicesProps = {}

  for (const [key, aliasList] of Object.entries(aliases)) {
    for (let i = 0; i < dimNames.length; i++) {
      const name = dimNames[i].toLowerCase()
      if (aliasList.map((a) => a.toLowerCase()).includes(name)) {
        indices[key] = {
          name: dimNames[i],
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
  dimensionValues: Record<string, Float64Array | number[] | string[]>,
  levelInfo: string | null,
  dimIndices: DimIndicesProps[string],
  root: zarr.Location<zarr.FetchStore>,
  zarrVersion: 2 | 3 | null,
  slice?: [number, number]
): Promise<Float64Array | number[] | string[]> {
  if (dimensionValues[dimIndices.name]) return dimensionValues[dimIndices.name]
  const targetRoot = levelInfo ? root.resolve(levelInfo) : root
  let coordArr
  if (dimIndices.array) {
    coordArr = dimIndices.array
  } else {
    const coordVar = targetRoot.resolve(dimIndices.name)
    const localFunc = resolveOpenFunc(zarrVersion)
    coordArr = await localFunc(coordVar, { kind: 'array' })
  }
  const coordData = await zarr.get(coordArr)
  const data = coordData.data

  // Handle string arrays (zarrita returns Array<string> for vlen-utf8)
  if (Array.isArray(data) && data.length > 0 && typeof data[0] === 'string') {
    const stringArray = data as string[]
    if (slice) {
      return stringArray.slice(slice[0], slice[1])
    }
    return stringArray
  }

  // Handle numeric arrays
  const coordArray = Array.from(
    data as ArrayLike<number | bigint>,
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
        // Sanitize band names to be valid GLSL identifiers
        const rawName =
          typeof bandValue === 'string' ? bandValue : `${key}_${bandValue}`
        const bandName = sanitizeGlslName(rawName)
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
