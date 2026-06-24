/**
 * @module selector-utils
 *
 * Utility functions for handling selectors in region queries.
 * Follows patterns from carbonplan/maps.
 */

import type { QueryDataValues } from './types'

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
