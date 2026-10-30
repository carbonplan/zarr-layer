import { describe, it, expect } from 'vitest'
import { createTileMatrix, getTileBounds, boundsIntersect } from './tile-utils'

/**
 * The tile matrix decides which way up a draped tile is drawn. Mapbox's
 * renderToTile and MapLibre's renderToTerrainTile disagree on the vertical
 * direction of tile clip space, and getting it wrong draws the data upside
 * down within every tile, which nothing but a map would reveal.
 */

/** Apply the column-major matrix to a normalized Mercator point. */
function project(matrix: Float32Array, x: number, y: number): [number, number] {
  return [matrix[0] * x + matrix[12], matrix[5] * y + matrix[13]]
}

// Tile z=1 x=0 y=0: the north-west quarter of the world.
const tile = { x0: 0, y0: 0, x1: 0.5, y1: 0.5 }
const southWest = [tile.x0, tile.y1] as const
const northEast = [tile.x1, tile.y0] as const

describe('createTileMatrix', () => {
  it('puts the south-west corner at (-1, -1) when north is up', () => {
    const m = createTileMatrix(tile.x0, tile.y0, tile.x1, tile.y1, true)
    expect(project(m, ...southWest)).toEqual([-1, -1])
    expect(project(m, ...northEast)).toEqual([1, 1])
  })

  it('puts the north edge at clip y = -1 when north is down', () => {
    const m = createTileMatrix(tile.x0, tile.y0, tile.x1, tile.y1, false)
    expect(project(m, ...southWest)).toEqual([-1, 1])
    expect(project(m, ...northEast)).toEqual([1, -1])
  })

  it('keeps the horizontal mapping the same either way', () => {
    const up = createTileMatrix(tile.x0, tile.y0, tile.x1, tile.y1, true)
    const down = createTileMatrix(tile.x0, tile.y0, tile.x1, tile.y1, false)
    expect(project(up, 0.25, 0.25)[0]).toBe(project(down, 0.25, 0.25)[0])
  })

  it('centers the tile at the clip origin', () => {
    const m = createTileMatrix(0.75, 0.25, 1, 0.5, true)
    expect(project(m, 0.875, 0.375)).toEqual([0, 0])
  })
})

describe('getTileBounds', () => {
  it('returns the normalized Mercator extent of a canonical tile', () => {
    expect(getTileBounds({ z: 2, x: 3, y: 1 })).toEqual({
      x0: 0.75,
      x1: 1,
      y0: 0.25,
      y1: 0.5,
    })
  })

  it('covers the world at zoom 0', () => {
    expect(getTileBounds({ z: 0, x: 0, y: 0 })).toEqual({
      x0: 0,
      x1: 1,
      y0: 0,
      y1: 1,
    })
  })
})

describe('boundsIntersect', () => {
  const a = { x0: 0, y0: 0, x1: 0.5, y1: 0.5 }

  it('is true for overlapping bounds', () => {
    expect(boundsIntersect(a, { x0: 0.25, y0: 0.25, x1: 1, y1: 1 })).toBe(true)
  })

  it('is false for bounds that only share an edge', () => {
    expect(boundsIntersect(a, { x0: 0.5, y0: 0, x1: 1, y1: 0.5 })).toBe(false)
  })
})
