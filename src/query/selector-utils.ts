/**
 * @module selector-utils
 *
 * Utility functions for handling selectors in region queries.
 * Follows patterns from carbonplan/maps.
 */

import type { ZarrSelectorsProps } from '../types'
import type { QuerySelector, PointValueEntry, RegionValues } from './types'

/**
 * Checks if a selector contains any array values.
 * Array values mean results should be nested.
 */
export function hasArraySelector(selector: QuerySelector): boolean {
  for (const key of Object.keys(selector)) {
    const value = selector[key]
    if (Array.isArray(value)) return true
    if (
      typeof value === 'object' &&
      value !== null &&
      'selected' in value &&
      Array.isArray((value as ZarrSelectorsProps).selected)
    ) {
      return true
    }
  }
  return false
}

/**
 * Normalizes a selector value to an array of indices or values.
 */
export function normalizeSelectorValue(
  value: number | number[] | string | string[] | ZarrSelectorsProps | undefined,
  coordinates?: (string | number)[]
): (number | string)[] {
  if (value === undefined) return []

  // Handle ZarrSelectorsProps format
  if (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    'selected' in value
  ) {
    const selected = (value as ZarrSelectorsProps).selected
    const type = (value as ZarrSelectorsProps).type
    const values = Array.isArray(selected) ? selected : [selected]

    if (type === 'index') {
      return values as (number | string)[]
    }
    // Value-based lookup
    if (coordinates) {
      return values.map((v) => {
        const idx = coordinates.indexOf(v as string | number)
        return idx >= 0 ? idx : (v as number | string)
      })
    }
    return values as (number | string)[]
  }

  // Handle simple array or single value
  if (Array.isArray(value)) {
    return value
  }

  return [value]
}

/**
 * Gets the index for a selector value in coordinates.
 */
export function getSelectorIndex(
  value: number | string,
  coordinates: (string | number)[]
): number {
  if (typeof value === 'number' && !coordinates.includes(value)) {
    // Assume it's already an index
    return value
  }
  const idx = coordinates.indexOf(value)
  return idx >= 0 ? idx : 0
}

/**
 * Computes which chunks are needed for a selector.
 * Adapted from carbonplan/maps getChunks().
 */
export function getChunksForSelector(
  selector: QuerySelector,
  dimensions: string[],
  coordinates: Record<string, (string | number)[]>,
  shape: number[],
  chunks: number[],
  tileX: number,
  tileY: number
): number[][] {
  const chunkIndicesToUse = dimensions.map((dimension, i) => {
    const dimLower = dimension.toLowerCase()

    // Spatial dimensions use tile coordinates
    if (['x', 'lon', 'longitude'].includes(dimLower)) {
      return [tileX]
    }
    if (['y', 'lat', 'latitude'].includes(dimLower)) {
      return [tileY]
    }

    const selectorValue = selector[dimension]
    const coords = coordinates[dimension]
    const chunkSize = chunks[i]

    let indices: number[]

    if (selectorValue === undefined) {
      // No selector - use all indices
      indices = Array(shape[i])
        .fill(null)
        .map((_, j) => j)
    } else if (Array.isArray(selectorValue)) {
      // Array of values - get index for each
      indices = selectorValue.map((v) => {
        if (coords) {
          const idx = coords.indexOf(v)
          return idx >= 0 ? idx : (typeof v === 'number' ? v : 0)
        }
        return typeof v === 'number' ? v : 0
      })
    } else if (
      typeof selectorValue === 'object' &&
      'selected' in selectorValue
    ) {
      // ZarrSelectorsProps format
      const selected = selectorValue.selected
      const type = selectorValue.type
      const values = Array.isArray(selected) ? selected : [selected]

      if (type === 'index') {
        indices = values.map((v) => (typeof v === 'number' ? v : 0))
      } else {
        indices = values.map((v) => {
          if (coords) {
            const idx = coords.indexOf(v as string | number)
            return idx >= 0 ? idx : (typeof v === 'number' ? v : 0)
          }
          return typeof v === 'number' ? v : 0
        })
      }
    } else {
      // Single value
      if (coords) {
        const idx = coords.indexOf(selectorValue)
        indices = [idx >= 0 ? idx : (typeof selectorValue === 'number' ? selectorValue : 0)]
      } else {
        indices = [typeof selectorValue === 'number' ? selectorValue : 0]
      }
    }

    // Convert indices to chunk indices and deduplicate
    return indices
      .map((index) => Math.floor(index / chunkSize))
      .filter((v, i, a) => a.indexOf(v) === i)
  })

  // Generate cartesian product of all chunk index combinations
  let result: number[][] = [[]]
  for (const indices of chunkIndicesToUse) {
    const updatedResult: number[][] = []
    for (const index of indices) {
      for (const prev of result) {
        updatedResult.push([...prev, index])
      }
    }
    result = updatedResult
  }

  return result
}

/**
 * Gets point values for all selector dimension combinations.
 * Adapted from carbonplan/maps Tile.getPointValues().
 */
export function getPointValues(
  data: Float32Array,
  pixelX: number,
  pixelY: number,
  tileSize: number,
  channels: number,
  selector: QuerySelector,
  dimensions: string[],
  coordinates: Record<string, (string | number)[]>,
  shape: number[],
  chunks: number[],
  chunkIndices: number[]
): PointValueEntry[] {
  const result: PointValueEntry[] = []

  // Build combined indices for all selector combinations
  let combinedIndices: number[][] = [[]]
  const keys: ((string | number)[])[] = [[]]

  for (let i = 0; i < dimensions.length; i++) {
    const dimension = dimensions[i]
    const dimLower = dimension.toLowerCase()
    const chunkOffset = chunkIndices[i] * chunks[i]
    const coords = coordinates[dimension]

    if (['x', 'lon', 'longitude'].includes(dimLower)) {
      combinedIndices = combinedIndices.map((prev) => [...prev, pixelX])
      // No keys for spatial dimensions
    } else if (['y', 'lat', 'latitude'].includes(dimLower)) {
      combinedIndices = combinedIndices.map((prev) => [...prev, pixelY])
      // No keys for spatial dimensions
    } else {
      const selectorValue = selector[dimension]
      let selectorIndices: number[]
      let selectorKeys: (string | number)[]

      if (selectorValue === undefined) {
        // No selector - use all values in this chunk
        selectorIndices = []
        selectorKeys = []
        for (let j = 0; j < chunks[i]; j++) {
          const globalIndex = chunkOffset + j
          if (globalIndex < shape[i]) {
            selectorIndices.push(globalIndex)
            if (coords) {
              selectorKeys.push(coords[globalIndex])
            }
          }
        }
      } else if (Array.isArray(selectorValue)) {
        // Array selector - get indices for values in this chunk
        selectorIndices = []
        selectorKeys = []
        for (const v of selectorValue) {
          let idx: number
          if (coords) {
            idx = coords.indexOf(v)
            if (idx < 0) idx = typeof v === 'number' ? v : 0
          } else {
            idx = typeof v === 'number' ? v : 0
          }
          if (idx >= chunkOffset && idx < chunkOffset + chunks[i]) {
            selectorIndices.push(idx)
            selectorKeys.push(v)
          }
        }
      } else if (
        typeof selectorValue === 'object' &&
        'selected' in selectorValue
      ) {
        const selected = selectorValue.selected
        const type = selectorValue.type
        const values = Array.isArray(selected) ? selected : [selected]

        selectorIndices = []
        selectorKeys = []
        for (const v of values) {
          let idx: number
          if (type === 'index') {
            idx = typeof v === 'number' ? v : 0
          } else if (coords) {
            idx = coords.indexOf(v as string | number)
            if (idx < 0) idx = typeof v === 'number' ? v : 0
          } else {
            idx = typeof v === 'number' ? v : 0
          }
          if (idx >= chunkOffset && idx < chunkOffset + chunks[i]) {
            selectorIndices.push(idx)
            if (Array.isArray(selected)) {
              selectorKeys.push(v as string | number)
            }
          }
        }
      } else {
        // Single value
        let idx: number
        if (coords) {
          idx = coords.indexOf(selectorValue)
          if (idx < 0) idx = typeof selectorValue === 'number' ? selectorValue : 0
        } else {
          idx = typeof selectorValue === 'number' ? selectorValue : 0
        }
        selectorIndices = [idx]
        selectorKeys = [] // No keys for single value
      }

      // Expand combined indices with selector indices
      const newCombined: number[][] = []
      const newKeys: ((string | number)[])[] = []
      for (let j = 0; j < selectorIndices.length; j++) {
        for (let k = 0; k < combinedIndices.length; k++) {
          newCombined.push([...combinedIndices[k], selectorIndices[j]])
          if (selectorKeys.length > 0) {
            newKeys.push([...keys[k], selectorKeys[j]])
          } else {
            newKeys.push([...keys[k]])
          }
        }
      }
      combinedIndices = newCombined.length > 0 ? newCombined : combinedIndices.map((prev) => [...prev, 0])
      keys.length = 0
      keys.push(...(newKeys.length > 0 ? newKeys : keys.map(() => [])))
    }
  }

  // Extract values for each combination
  for (let i = 0; i < combinedIndices.length; i++) {
    const indices = combinedIndices[i]
    const entryKeys = keys[i] || []

    // Convert global indices to local chunk indices for non-spatial dimensions
    const localIndices = indices.map((idx, j) => {
      const dimLower = dimensions[j].toLowerCase()
      if (['x', 'lon', 'longitude', 'y', 'lat', 'latitude'].includes(dimLower)) {
        return idx
      }
      return idx - chunkIndices[j] * chunks[j]
    })

    // Calculate flat index in data array
    // For simple 2D lat/lon data: index = y * width + x
    // For higher-dimensional data, we need to compute based on shape
    const latIdx = dimensions.findIndex((d) =>
      ['y', 'lat', 'latitude'].includes(d.toLowerCase())
    )
    const lonIdx = dimensions.findIndex((d) =>
      ['x', 'lon', 'longitude'].includes(d.toLowerCase())
    )

    if (latIdx >= 0 && lonIdx >= 0) {
      const dataIndex =
        localIndices[latIdx] * tileSize * channels +
        localIndices[lonIdx] * channels

      const value = data[dataIndex]
      result.push({ keys: entryKeys, value })
    }
  }

  return result
}

/**
 * Mutates an object by adding a value to an array at a nested location.
 * Adapted from carbonplan/maps setObjectValues().
 */
export function setObjectValues(
  obj: RegionValues,
  keys: (string | number)[],
  value: number
): RegionValues {
  if (keys.length === 0) {
    if (Array.isArray(obj)) {
      obj.push(value)
    }
    return obj
  }

  let ref = obj as Record<string | number, RegionValues>
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i]
    if (i === keys.length - 1) {
      if (!ref[key]) {
        ref[key] = []
      }
      const arr = ref[key]
      if (Array.isArray(arr)) {
        arr.push(value)
      }
    } else {
      if (!ref[key]) {
        ref[key] = {}
      }
      ref = ref[key] as Record<string | number, RegionValues>
    }
  }

  return obj
}

/**
 * Computes a hash string for a selector to use as cache key.
 */
export function getSelectorHash(selector: QuerySelector): string {
  return JSON.stringify(selector, Object.keys(selector).sort())
}
