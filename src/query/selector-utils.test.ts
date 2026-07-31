import { describe, it, expect } from 'vitest'
import { setObjectValues } from './selector-utils'
import type { QueryDataValues } from './types'

describe('setObjectValues', () => {
  it('pushes onto a flat array for empty keys', () => {
    const obj: QueryDataValues = []
    setObjectValues(obj, [], 5)
    expect(obj).toEqual([5])
  })

  it('accumulates values under a single key', () => {
    const obj: QueryDataValues = {}
    setObjectValues(obj, ['a'], 5)
    setObjectValues(obj, ['a'], 6)
    expect(obj).toEqual({ a: [5, 6] })
  })

  it('builds nested structure for multi-level keys', () => {
    const obj: QueryDataValues = {}
    setObjectValues(obj, ['a', 'b'], 5)
    expect(obj).toEqual({ a: { b: [5] } })
  })
})
