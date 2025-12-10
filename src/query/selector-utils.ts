/**
 * @module selector-utils
 *
 * Utility functions for handling selectors in region queries.
 * Follows patterns from carbonplan/maps.
 */

import type { ZarrSelectorsProps } from '../types'
import type { QuerySelector, PointValueEntry, QueryDataValues } from './types'

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
          return idx >= 0 ? idx : typeof v === 'number' ? v : 0
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
            return idx >= 0 ? idx : typeof v === 'number' ? v : 0
          }
          return typeof v === 'number' ? v : 0
        })
      }
    } else {
      // Single value
      if (coords) {
        const idx = coords.indexOf(selectorValue)
        indices = [
          idx >= 0
            ? idx
            : typeof selectorValue === 'number'
            ? selectorValue
            : 0,
        ]
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
  const keys: (string | number)[][] = [[]]

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
          if (idx < 0)
            idx = typeof selectorValue === 'number' ? selectorValue : 0
        } else {
          idx = typeof selectorValue === 'number' ? selectorValue : 0
        }
        selectorIndices = [idx]
        selectorKeys = [] // No keys for single value
      }

      // Expand combined indices with selector indices
      const newCombined: number[][] = []
      const newKeys: (string | number)[][] = []
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
      combinedIndices =
        newCombined.length > 0
          ? newCombined
          : combinedIndices.map((prev) => [...prev, 0])
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
      if (
        ['x', 'lon', 'longitude', 'y', 'lat', 'latitude'].includes(dimLower)
      ) {
        return idx
      }
      return idx - chunkIndices[j] * chunks[j]
    })

    // Calculate flat index in data array
    // For multi-dimensional data, we need to compute strides for each dimension
    // Formula: index = sum(localIndices[i] * stride[i]) where stride[i] = product of all subsequent dimension sizes

    // The chunk size is just the chunks array - data array contains the full chunk
    // Calculate strides (product of subsequent chunk dimensions)
    const strides = new Array(dimensions.length)
    strides[dimensions.length - 1] = 1
    for (let j = dimensions.length - 2; j >= 0; j--) {
      strides[j] = strides[j + 1] * chunks[j + 1]
    }

    // Compute flat index
    let dataIndex = 0
    for (let j = 0; j < dimensions.length; j++) {
      dataIndex += localIndices[j] * strides[j]
    }

    const value = data[dataIndex]
    result.push({ keys: entryKeys, value })
  }

  return result
}

/**
 * Mutates an object by adding a value to an array at a nested location.
 * Adapted from carbonplan/maps setObjectValues().
 */
export function setObjectValues(
  obj: QueryDataValues,
  keys: (string | number)[],
  value: number
): QueryDataValues {
  if (keys.length === 0) {
    if (Array.isArray(obj)) {
      obj.push(value)
    }
    return obj
  }

  let ref = obj as Record<string | number, QueryDataValues>
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
      ref = ref[key] as Record<string | number, QueryDataValues>
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

/**
 * Get all chunk indices for a selector (cartesian product).
 * Directly ported from carbonplan/maps getChunks().
 *
 * @param selector - The query selector (can have multi-valued dimensions)
 * @param dimensions - Dimension names in order
 * @param coordinates - Coordinate values for each dimension
 * @param shape - Array shape
 * @param chunks - Chunk sizes for each dimension
 * @param x - Tile x coordinate (for lon dimension)
 * @param y - Tile y coordinate (for lat dimension)
 * @returns Array of chunk index arrays (cartesian product)
 */
export function getChunks(
  selector: QuerySelector,
  dimensions: string[],
  coordinates: Record<string, (string | number)[]>,
  shape: number[],
  chunks: number[],
  x: number,
  y: number
): number[][] {
  // Map each dimension to its relevant chunk indices
  const chunkIndicesToUse = dimensions.map((dimension, i) => {
    const dimLower = dimension.toLowerCase()

    // Spatial dimensions: use tile coordinates
    if (['x', 'lon', 'longitude'].includes(dimLower)) {
      return [x]
    } else if (['y', 'lat', 'latitude'].includes(dimLower)) {
      return [y]
    }

    const selectorValue = selector[dimension]
    const coords = coordinates[dimension]
    const chunkSize = chunks[i]
    let indices: number[]

    if (selectorValue === undefined) {
      // Unconstrained dimension: span entire dimension
      indices = Array(shape[i])
        .fill(null)
        .map((_, j) => j)
    } else if (Array.isArray(selectorValue)) {
      // Multi-value: find ALL indices
      indices = selectorValue.map((v) => {
        const idx = coords ? coords.indexOf(v) : typeof v === 'number' ? v : 0
        return idx >= 0 ? idx : typeof v === 'number' ? v : 0
      })
    } else if (
      typeof selectorValue === 'object' &&
      'selected' in selectorValue
    ) {
      // ZarrSelectorsProps format
      const selected = selectorValue.selected
      const type = selectorValue.type
      const values = Array.isArray(selected) ? selected : [selected]

      indices = values.map((v) => {
        if (type === 'index') {
          return typeof v === 'number' ? v : 0
        }
        if (coords) {
          const idx = coords.indexOf(v as string | number)
          return idx >= 0 ? idx : typeof v === 'number' ? (v as number) : 0
        }
        return typeof v === 'number' ? (v as number) : 0
      })
    } else {
      // Single value
      if (coords) {
        const idx = coords.indexOf(selectorValue)
        indices = [
          idx >= 0
            ? idx
            : typeof selectorValue === 'number'
            ? selectorValue
            : 0,
        ]
      } else {
        indices = [typeof selectorValue === 'number' ? selectorValue : 0]
      }
    }

    // Map indices to chunk indices and deduplicate
    const chunkIndices = indices
      .map((index) => Math.floor(index / chunkSize))
      .filter((v, i, a) => a.indexOf(v) === i)

    return chunkIndices
  })

  // Compute cartesian product of all chunk indices
  let result: number[][] = [[]]
  chunkIndicesToUse.forEach((chunkIndices) => {
    const updatedResult: number[][] = []
    chunkIndices.forEach((chunkIndex) => {
      result.forEach((prev) => {
        updatedResult.push([...prev, chunkIndex])
      })
    })
    result = updatedResult
  })

  return result
}
