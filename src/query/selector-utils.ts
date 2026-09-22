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

  // Labels come from user selectors, so they are only ever read and written
  // as own properties: `constructor` or `__proto__` is a label like any other.
  const own = (target: object, key: string | number) =>
    Object.prototype.hasOwnProperty.call(target, key)
      ? (target as Record<string | number, QueryDataValues>)[key]
      : undefined
  const define = (
    target: object,
    key: string | number,
    value: QueryDataValues
  ) => {
    Object.defineProperty(target, key, {
      value,
      enumerable: true,
      writable: true,
      configurable: true,
    })
    return value
  }

  let ref: object = obj
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i]
    if (i === keys.length - 1) {
      const leaf = own(ref, key) ?? define(ref, key, [])
      if (Array.isArray(leaf)) leaf.push(value)
    } else {
      ref = own(ref, key) ?? define(ref, key, {})
    }
  }

  return obj
}
