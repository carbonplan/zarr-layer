import { describe, it, expect } from 'vitest'
import {
  groupCellsIntoRuns,
  haversineMeters,
  lineToPixelPath,
  splitLineAtAntimeridian,
  traceLineCells,
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

  it('walks a diagonal without skipping corners', () => {
    const { cells } = traceLineCells(
      path([
        [0.5, 0.5],
        [3.5, 3.5],
      ]),
      10,
      10
    )
    const set = new Set(cells.map((c) => `${c.x},${c.y}`))
    for (const c of ['0,0', '1,1', '2,2', '3,3']) expect(set.has(c)).toBe(true)
    // Every step moves to an edge-adjacent cell.
    for (let i = 1; i < cells.length; i++) {
      const dx = Math.abs(cells[i].x - cells[i - 1].x)
      const dy = Math.abs(cells[i].y - cells[i - 1].y)
      expect(dx + dy).toBe(1)
    }
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

  it('samples one cell for a degenerate single-point path', () => {
    const { cells } = traceLineCells(path([[4.2, 4.8]]), 10, 10)
    expect(xy(cells)).toEqual([[4, 4]])
  })
})

describe('lineToPixelPath', () => {
  it('projects a lon/lat line into the pixel grid and keeps lon/lat', () => {
    const path = lineToPixelPath(
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
    expect(path[0]).toMatchObject({ px: 0, py: 0, lon: -180, lat: 90 })
    const last = path[path.length - 1]
    expect(last).toMatchObject({ px: 10, py: 10, lon: 180, lat: -90 })
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
