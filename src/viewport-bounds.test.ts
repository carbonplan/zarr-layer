import { describe, it, expect } from 'vitest'
import { viewportBounds } from './region-renderer'
import type { MapLike } from './types'

const mapWith = (padding?: {
  top?: number
  bottom?: number
  left?: number
  right?: number
}): MapLike => ({
  getBounds: () => ({
    getWest: () => 0,
    getEast: () => 10,
    toArray: () => [
      [0, 0],
      [10, 10],
    ],
  }),
  getContainer: () => ({ clientWidth: 1000, clientHeight: 800 }),
  ...(padding ? { getPadding: () => padding } : {}),
})

describe('viewportBounds', () => {
  it("is the map's bounds without padding", () => {
    expect(viewportBounds(mapWith())).toEqual({
      west: 0,
      south: 0,
      east: 10,
      north: 10,
    })
    expect(viewportBounds(mapWith({ top: 0, left: 0 }))).toEqual({
      west: 0,
      south: 0,
      east: 10,
      north: 10,
    })
  })

  it('reaches into the padded edges the bounds leave out', () => {
    // 500 of 1000 pixels padded on the left: the bounds span the other 500,
    // so the padded half spans as much again.
    const bounds = viewportBounds(mapWith({ left: 500 }))!
    expect(bounds.west).toBeCloseTo(-10)
    expect(bounds.east).toBe(10)
    expect(bounds.south).toBe(0)
    expect(bounds.north).toBe(10)
  })

  it('widens latitude through Mercator', () => {
    const bounds = viewportBounds(mapWith({ top: 400 }))!
    expect(bounds.north).toBeGreaterThan(19)
    expect(bounds.north).toBeLessThan(21)
    expect(bounds.south).toBe(0)
  })

  it('carries east past west across the antimeridian', () => {
    const map: MapLike = {
      getBounds: () => ({
        getWest: () => 170,
        getEast: () => -170,
        toArray: () => [
          [170, 0],
          [-170, 10],
        ],
      }),
    }
    expect(viewportBounds(map)).toEqual({
      west: 170,
      south: 0,
      east: 190,
      north: 10,
    })
  })
})
