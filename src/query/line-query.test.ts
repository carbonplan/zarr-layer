import { describe, it, expect } from 'vitest'
import {
  groupCellsIntoRuns,
  haversineMeters,
  lineToPixelPaths,
  segmentLengthMeters,
  splitLineAtAntimeridian,
  traceLineCells,
  validateLineCoordinates,
  type TracedCell,
} from './line-query'
import type { DensifiedVertex } from './query-utils'
import type { Bounds } from '../types'

const WORLD: Bounds = [-180, -90, 180, 90]

/** Pixel-space path with lon/lat set so distance is well defined. */
function path(points: [number, number][]): DensifiedVertex[] {
  return points.map(([px, py]) => ({ px, py, lon: px, lat: py }))
}

const xy = (cells: TracedCell[]) => cells.map((c) => [c.x, c.y])

describe('splitLineAtAntimeridian', () => {
  it('leaves an in-range line whole and literal', () => {
    const pieces = splitLineAtAntimeridian([
      [170, 0],
      [-170, 0],
    ])
    expect(pieces).toEqual([
      [
        [170, 0],
        [-170, 0],
      ],
    ])
  })

  it('splits an explicitly unwrapped eastward crossing into ordered pieces', () => {
    const pieces = splitLineAtAntimeridian([
      [170, 10],
      [190, 30],
    ])
    expect(pieces).toEqual([
      [
        [170, 10],
        [180, 20],
      ],
      [
        [-180, 20],
        [-170, 30],
      ],
    ])
  })

  it('splits a westward crossing with the far side first', () => {
    const pieces = splitLineAtAntimeridian([
      [190, 0],
      [170, 0],
    ])
    expect(pieces).toEqual([
      [
        [-170, 0],
        [-180, 0],
      ],
      [
        [180, 0],
        [170, 0],
      ],
    ])
  })

  it('unwraps wrapped input once any vertex is out of range', () => {
    // 170 -> -170 would be literal alone, but the trailing 200 marks the
    // whole line as explicit, so the first edge is read as a crossing too.
    const pieces = splitLineAtAntimeridian([
      [170, 0],
      [-170, 0],
      [200, 0],
    ])
    expect(pieces.map((p) => p.length)).toEqual([2, 3])
    expect(pieces[1][0]).toEqual([-180, 0])
    expect(pieces[1][2]).toEqual([-160, 0])
  })

  it('does not split a line that only touches 180', () => {
    expect(
      splitLineAtAntimeridian([
        [170, 0],
        [180, 0],
      ])
    ).toHaveLength(1)
    expect(
      splitLineAtAntimeridian([
        [-180, 0],
        [-170, 0],
      ])
    ).toHaveLength(1)
  })

  it('does not split a line that touches the boundary from the far frame', () => {
    const pieces = splitLineAtAntimeridian([
      [190, 0],
      [180, 0],
      [190, 0],
    ])
    expect(pieces).toHaveLength(1)
    expect(pieces[0][0]).toEqual([-170, 0])
    expect(pieces[0][pieces[0].length - 1]).toEqual([-170, 0])
  })

  it('splits at a raster seam other than the antimeridian', () => {
    // Frame [-202.5, 157.5]: 170 belongs one turn west, at -190.
    const pieces = splitLineAtAntimeridian(
      [
        [150, 0],
        [170, 0],
      ],
      -202.5
    )
    expect(pieces).toEqual([
      [
        [150, 0],
        [157.5, 0],
      ],
      [
        [-202.5, 0],
        [-190, 0],
      ],
    ])
  })

  it('crosses back and forth into alternating pieces', () => {
    const pieces = splitLineAtAntimeridian([
      [170, 0],
      [190, 0],
      [170, 0],
    ])
    expect(pieces).toEqual([
      [
        [170, 0],
        [180, 0],
      ],
      [
        [-180, 0],
        [-170, 0],
        [-180, 0],
      ],
      [
        [180, 0],
        [170, 0],
      ],
    ])
  })
})

describe('validateLineCoordinates', () => {
  it('rejects coordinates the tracer could not walk in bounded time', () => {
    for (const bad of [NaN, Infinity, -Infinity, 1e20]) {
      expect(() =>
        validateLineCoordinates([
          [0, 0],
          [bad, 0],
        ])
      ).toThrow(RangeError)
    }
    expect(() =>
      validateLineCoordinates([
        [0, 0],
        [0, NaN],
      ])
    ).toThrow(RangeError)
    expect(() => validateLineCoordinates([[0, 0], [] as number[]])).toThrow(
      RangeError
    )
    for (const lat of [90.5, -91, 1e20]) {
      expect(() =>
        validateLineCoordinates([
          [0, 0],
          [1, lat],
        ])
      ).toThrow(RangeError)
    }
  })

  it('accepts explicitly unwrapped longitudes', () => {
    expect(() =>
      validateLineCoordinates([
        [170, 0],
        [190, 0],
        [-200, 90],
      ])
    ).not.toThrow()
  })
})

describe('traceLineCells', () => {
  it('visits each cell of a horizontal line once, in order', () => {
    const { cells } = traceLineCells(
      path([
        [0.5, 0.5],
        [4.5, 0.5],
      ]),
      10,
      10
    )
    expect(xy(cells)).toEqual([
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0],
      [4, 0],
    ])
  })

  it('steps diagonally through exact cell corners, whatever the direction or vertices', () => {
    const diagonal = [
      [0, 0],
      [1, 1],
      [2, 2],
    ]
    const forward = traceLineCells(
      path([
        [0.5, 0.5],
        [2.5, 2.5],
      ]),
      3,
      3
    )
    expect(xy(forward.cells)).toEqual(diagonal)

    const viaCorner = traceLineCells(
      path([
        [0.5, 0.5],
        [1, 1],
        [2.5, 2.5],
      ]),
      3,
      3
    )
    expect(xy(viaCorner.cells)).toEqual(diagonal)

    const backward = traceLineCells(
      path([
        [2.5, 2.5],
        [0.5, 0.5],
      ]),
      3,
      3
    )
    expect(xy(backward.cells)).toEqual([...diagonal].reverse())
  })

  it('steps through edge-adjacent cells on an off-corner diagonal', () => {
    const { cells } = traceLineCells(
      path([
        [0.5, 0.25],
        [3.5, 3.25],
      ]),
      10,
      10
    )
    for (let i = 1; i < cells.length; i++) {
      const dx = Math.abs(cells[i].x - cells[i - 1].x)
      const dy = Math.abs(cells[i].y - cells[i - 1].y)
      expect(dx + dy).toBe(1)
    }
    expect(cells).toHaveLength(7)
  })

  it('ignores a repeated vertex, even one sitting on a cell edge', () => {
    const plain = traceLineCells(
      path([
        [0.5, 0.5],
        [1, 0.5],
        [0.5, 0.5],
      ]),
      3,
      3
    )
    const repeated = traceLineCells(
      path([
        [0.5, 0.5],
        [1, 0.5],
        [1, 0.5],
        [0.5, 0.5],
      ]),
      3,
      3
    )
    expect(xy(plain.cells)).toEqual([[0, 0]])
    expect(xy(repeated.cells)).toEqual([[0, 0]])
    expect(repeated.endDistance).toBeCloseTo(plain.endDistance)
  })

  it('keeps every cell when the segment starts absurdly far outside', () => {
    const { cells } = traceLineCells(
      [
        { px: -1e17, py: 0.5, lon: 0, lat: 0 },
        { px: 1e17, py: 0.5, lon: 1, lat: 0 },
      ],
      3,
      3
    )
    expect(xy(cells)).toEqual([
      [0, 0],
      [1, 0],
      [2, 0],
    ])
  })

  it('samples nothing for a line lying exactly on the far grid edge', () => {
    const right = traceLineCells(
      path([
        [3, 0.5],
        [3, 2.5],
      ]),
      3,
      3
    )
    expect(right.cells).toEqual([])
  })

  it('does not repeat the shared cell at a vertex between segments', () => {
    const { cells } = traceLineCells(
      path([
        [0.5, 0.5],
        [2.5, 0.5],
        [2.5, 2.5],
      ]),
      10,
      10
    )
    expect(xy(cells)).toEqual([
      [0, 0],
      [1, 0],
      [2, 0],
      [2, 1],
      [2, 2],
    ])
  })

  it('returns to a cell it visited earlier as a distinct sample', () => {
    const { cells } = traceLineCells(
      path([
        [0.5, 0.5],
        [2.5, 0.5],
        [0.5, 0.5],
      ]),
      10,
      10
    )
    expect(xy(cells)).toEqual([
      [0, 0],
      [1, 0],
      [2, 0],
      [1, 0],
      [0, 0],
    ])
  })

  it('drops cells outside the grid but keeps distance running through them', () => {
    const { cells } = traceLineCells(
      path([
        [-2.5, 0.5],
        [2.5, 0.5],
      ]),
      3,
      3
    )
    expect(xy(cells)).toEqual([
      [0, 0],
      [1, 0],
      [2, 0],
    ])
    // The line enters cell 0 after travelling 2.5 units from its start.
    expect(cells[0].distance).toBeCloseTo(haversineMeters(-2.5, 0.5, 0, 0.5))
  })

  it('reports monotone distance and continues from an offset', () => {
    const first = traceLineCells(
      path([
        [0.5, 0.5],
        [3.5, 0.5],
      ]),
      10,
      10
    )
    for (let i = 1; i < first.cells.length; i++) {
      expect(first.cells[i].distance).toBeGreaterThan(
        first.cells[i - 1].distance
      )
    }
    expect(first.cells[0].distance).toBe(0)
    expect(first.endDistance).toBeCloseTo(haversineMeters(0.5, 0.5, 3.5, 0.5))

    const second = traceLineCells(
      path([
        [3.5, 0.5],
        [5.5, 0.5],
      ]),
      10,
      10,
      first.endDistance
    )
    expect(second.cells[0].distance).toBe(first.endDistance)
  })

  it('assigns a start on a cell boundary heading down to the lower cell', () => {
    const { cells } = traceLineCells(
      path([
        [3, 0.5],
        [1.5, 0.5],
      ]),
      10,
      10
    )
    expect(xy(cells)).toEqual([
      [2, 0],
      [1, 0],
    ])
  })

  it('skips the off-grid part of a segment without walking it', () => {
    const start = performance.now()
    const { cells, endDistance } = traceLineCells(
      [
        { px: -1e9, py: 0.5, lon: -60, lat: 10 },
        { px: 1e9, py: 0.5, lon: 60, lat: 10 },
      ],
      3,
      3
    )
    expect(performance.now() - start).toBeLessThan(50)
    expect(xy(cells)).toEqual([
      [0, 0],
      [1, 0],
      [2, 0],
    ])
    // The grid sits at the middle of the line, and the axis covers all of it.
    expect(cells[0].distance).toBeCloseTo(
      segmentLengthMeters(-60, 10, 0, 10),
      0
    )
    expect(endDistance).toBeCloseTo(segmentLengthMeters(-60, 10, 60, 10), 0)
  })

  it('returns nothing, quickly, for a line that never touches the grid', () => {
    const start = performance.now()
    const { cells, endDistance } = traceLineCells(
      [
        { px: 5000, py: 5000, lon: 0, lat: 0 },
        { px: 1e9, py: 3e8, lon: 40, lat: 20 },
      ],
      1000,
      1000
    )
    expect(performance.now() - start).toBeLessThan(50)
    expect(cells).toEqual([])
    expect(endDistance).toBeCloseTo(segmentLengthMeters(0, 0, 40, 20), 0)
  })

  it('samples a cell again when the line leaves the grid and comes back', () => {
    const { cells } = traceLineCells(
      path([
        [0.5, 0.5],
        [-5, 0.5],
        [0.5, 0.5],
      ]),
      3,
      3
    )
    expect(xy(cells)).toEqual([
      [0, 0],
      [0, 0],
    ])
    expect(cells[1].distance).toBeGreaterThan(cells[0].distance)
  })

  it('clips a diagonal that cuts a corner of the grid', () => {
    const { cells } = traceLineCells(
      path([
        [-1.5, 1.5],
        [1.5, -1.5],
      ]),
      3,
      3
    )
    expect(xy(cells)).toContainEqual([0, 0])
    for (const [x, y] of xy(cells)) {
      expect(x).toBeGreaterThanOrEqual(0)
      expect(y).toBeGreaterThanOrEqual(0)
    }
  })

  it('samples one cell for a degenerate single-point path', () => {
    const { cells } = traceLineCells(path([[4.2, 4.8]]), 10, 10)
    expect(xy(cells)).toEqual([[4, 4]])
  })
})

describe('segmentLengthMeters', () => {
  it('stays bounded for a span no real line has', () => {
    const start = performance.now()
    const length = segmentLengthMeters(0, 0, 1, 1e20)
    expect(performance.now() - start).toBeLessThan(200)
    expect(Number.isNaN(length) || Number.isFinite(length)).toBe(true)
  })

  it('does not depend on how the segment is split', () => {
    const whole = segmentLengthMeters(-100, 60, 40, 60)
    const halves =
      segmentLengthMeters(-100, 60, -30, 60) +
      segmentLengthMeters(-30, 60, 40, 60)
    expect(halves / whole).toBeCloseTo(1, 6)
  })

  it('measures along the lon/lat line, which is longer than the great circle', () => {
    expect(segmentLengthMeters(-100, 60, 40, 60)).toBeGreaterThan(
      haversineMeters(-100, 60, 40, 60)
    )
  })
})

describe('lineToPixelPaths', () => {
  it('projects a lon/lat line into the pixel grid and keeps lon/lat', () => {
    const { sections, trailingGap } = lineToPixelPaths(
      [
        [-180, 90],
        [180, -90],
      ],
      WORLD,
      10,
      10,
      'EPSG:4326',
      false
    )
    expect(sections).toHaveLength(1)
    expect(sections[0].gapBefore).toBe(0)
    expect(trailingGap).toBe(0)
    const path = sections[0].path
    expect(path[0]).toMatchObject({ px: 0, py: 0, lon: -180, lat: 90 })
    const last = path[path.length - 1]
    expect(last).toMatchObject({ px: 10, py: 10, lon: 180, lat: -90 })
  })

  it('breaks the line at a vertex the projection cannot place', () => {
    // Orthographic centred on 0,0 cannot place the far side of the globe.
    const ortho = '+proj=ortho +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs'
    const bounds: Bounds = [-6378137, -6378137, 6378137, 6378137]
    const { sections, trailingGap } = lineToPixelPaths(
      [
        [-10, 0],
        [180, 0],
        [10, 0],
      ],
      bounds,
      100,
      100,
      ortho,
      false
    )
    expect(sections).toHaveLength(2)
    expect(sections[0].path).toHaveLength(1)
    expect(sections[1].path).toHaveLength(1)
    expect(sections[0].gapBefore).toBe(0)
    // Both edges touching the lost vertex are skipped, and their length kept.
    expect(sections[1].gapBefore).toBeCloseTo(
      segmentLengthMeters(-10, 0, 180, 0) + segmentLengthMeters(180, 0, 10, 0),
      0
    )
    expect(trailingGap).toBe(0)
  })
})

describe('groupCellsIntoRuns', () => {
  const cell = (x: number, y: number): TracedCell => ({ x, y, distance: 0 })

  it('keeps cells in one run while they fit within the span', () => {
    const runs = groupCellsIntoRuns([cell(0, 0), cell(1, 0), cell(2, 0)], 3)
    expect(runs).toHaveLength(1)
    expect(runs[0].rect).toEqual({ minX: 0, maxX: 3, minY: 0, maxY: 1 })
  })

  it('starts a new run when either axis would exceed the span', () => {
    const runs = groupCellsIntoRuns(
      [cell(0, 0), cell(1, 0), cell(2, 0), cell(3, 0), cell(3, 1), cell(3, 2)],
      3
    )
    expect(runs.map((r) => r.cells.length)).toEqual([3, 3])
    expect(runs[1].rect).toEqual({ minX: 3, maxX: 4, minY: 0, maxY: 3 })
  })

  it('bounds a run by extent, not by count, for a line that doubles back', () => {
    const runs = groupCellsIntoRuns(
      [cell(0, 0), cell(1, 0), cell(0, 0), cell(1, 0), cell(0, 0)],
      2
    )
    expect(runs).toHaveLength(1)
    expect(runs[0].cells).toHaveLength(5)
  })
})
