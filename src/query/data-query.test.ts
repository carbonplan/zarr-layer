import { describe, it, expect } from 'vitest'
import { mergeQueryResults, mergeNestedValues } from './data-query'
import type { NestedValues, QueryResult } from './types'

/**
 * Merge semantics for antimeridian two-strip queries: west-strip pixels then
 * east-strip pixels, spatial coordinate arrays concatenated, non-spatial
 * coordinates taken from the first strip.
 */

describe('mergeQueryResults', () => {
  it('concatenates values and spatial coordinates', () => {
    const west: QueryResult = {
      temp: [1, 2],
      dimensions: ['lat', 'lon'],
      coordinates: { lat: [10, 10], lon: [179, 179.5], month: [1] },
    }
    const east: QueryResult = {
      temp: [3],
      dimensions: ['lat', 'lon'],
      coordinates: { lat: [10], lon: [-179.5], month: [1] },
    }
    const merged = mergeQueryResults(west, east, 'temp', 'lat', 'lon')

    expect(merged.temp).toEqual([1, 2, 3])
    expect(merged.coordinates.lat).toEqual([10, 10, 10])
    expect(merged.coordinates.lon).toEqual([179, 179.5, -179.5])
    // Non-spatial coordinates come from the first strip unchanged.
    expect(merged.coordinates.month).toEqual([1])
    expect(merged.dimensions).toEqual(['lat', 'lon'])
  })

  it('merges nested multi-band values recursively', () => {
    const west: QueryResult = {
      temp: { jan: [1] },
      dimensions: ['lat', 'lon'],
      coordinates: { lat: [0], lon: [179] },
    }
    const east: QueryResult = {
      temp: { jan: [2], feb: [3] },
      dimensions: ['lat', 'lon'],
      coordinates: { lat: [0], lon: [-179] },
    }
    const merged = mergeQueryResults(west, east, 'temp', 'lat', 'lon')
    expect(merged.temp).toEqual({ jan: [1, 2], feb: [3] })
  })

  it('takes the first side when value shapes disagree', () => {
    const west: QueryResult = {
      temp: [1],
      dimensions: ['lat', 'lon'],
      coordinates: { lat: [0], lon: [179] },
    }
    const east: QueryResult = {
      temp: { jan: [2] },
      dimensions: ['lat', 'lon'],
      coordinates: { lat: [0], lon: [-179] },
    }
    const merged = mergeQueryResults(west, east, 'temp', 'lat', 'lon')
    expect(merged.temp).toEqual([1])
  })
})

describe('mergeNestedValues', () => {
  it('concatenates leaf arrays at any depth', () => {
    const a: NestedValues = { r: { low: [1] }, g: { low: [2] } }
    const b: NestedValues = { r: { low: [3], high: [4] } }
    expect(mergeNestedValues(a, b)).toEqual({
      r: { low: [1, 3], high: [4] },
      g: { low: [2] },
    })
  })

  it('includes keys present only in the second object', () => {
    expect(mergeNestedValues({ a: [1] }, { b: [2] })).toEqual({
      a: [1],
      b: [2],
    })
  })
})
