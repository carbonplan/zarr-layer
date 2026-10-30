import { describe, it, expect } from 'vitest'
import { tileClipBounds, tileClipPasses, type ClipTile } from './clip-boxes'
import { MAX_CLIP_BOXES } from './shaders'
import { getTileBounds } from './tile-utils'
import type { MeshMercatorBounds } from './map-utils'

type Box = { x0: number; y0: number; x1: number; y1: number }

/** The boxes of a pass back in absolute Mercator. */
function absolute(pass: Float32Array, mesh: MeshMercatorBounds): Box[] {
  const ax = (mesh.x0 + mesh.x1) / 2
  const ay = (mesh.y0 + mesh.y1) / 2
  const boxes: Box[] = []
  for (let i = 0; i < pass.length; i += 4) {
    boxes.push({
      x0: pass[i] + ax,
      y0: pass[i + 1] + ay,
      x1: pass[i + 2] + ax,
      y1: pass[i + 3] + ay,
    })
  }
  return boxes
}

const expectBox = (box: Box, expected: Box) => {
  expect(box.x0).toBeCloseTo(expected.x0, 6)
  expect(box.y0).toBeCloseTo(expected.y0, 6)
  expect(box.x1).toBeCloseTo(expected.x1, 6)
  expect(box.y1).toBeCloseTo(expected.y1, 6)
}

const tile = (z: number, x: number, y: number): ClipTile => ({
  tileId: { z, x, y },
})

describe('tileClipPasses', () => {
  it('boxes a region by its owning tiles, whatever its outline', () => {
    // A region may be wider than a tile or have a curved footprint; its box
    // is the tile's exact Mercator rectangle either way.
    const mesh = { x0: 0.2, y0: 0.2, x1: 0.6, y1: 0.5 }
    const owner = tile(3, 2, 2)

    const passes = tileClipPasses(mesh, [owner, tile(3, 7, 7)])

    expect(passes).toHaveLength(1)
    const boxes = absolute(passes[0], mesh)
    expect(boxes).toHaveLength(1)
    expectBox(boxes[0], getTileBounds(owner.tileId))
  })

  it('merges tiles along a row and splits past the box budget', () => {
    const mesh = { x0: 0, y0: 0, x1: 1, y1: 1 }
    const block: ClipTile[] = []
    for (let x = 0; x < 4; x++) {
      for (let y = 0; y < 3; y++) block.push(tile(3, x, y))
    }
    const [pass] = tileClipPasses(mesh, block)
    const rows = absolute(pass, mesh)
    expect(rows).toHaveLength(3)
    for (const row of rows) {
      // The rows start on the mesh's west edge, which opens.
      expect(row.x0).toBeLessThan(0)
      expect(row.x1).toBeCloseTo(0.5, 6)
    }

    const column: ClipTile[] = []
    for (let y = 0; y < MAX_CLIP_BOXES + 8; y++) column.push(tile(6, 0, y))
    const passes = tileClipPasses(mesh, column)
    expect(passes.map((p) => p.length / 4)).toEqual([MAX_CLIP_BOXES, 8])
  })

  it('opens box edges that lie on the mesh edge, so neighbours leave no gap', () => {
    // A region whose east edge is a tile edge: the box there would decide
    // the region's edge pixels by a rounding error against the delta the
    // shader interpolates, so it moves out a mesh width. Its west edge,
    // inside the mesh, still clips.
    const owner = tile(2, 1, 1)
    const bounds = getTileBounds(owner.tileId)
    const mesh = { x0: 0.2, y0: 0.2, x1: bounds.x1, y1: 0.6 }

    const [pass] = tileClipPasses(mesh, [owner])

    const [box] = absolute(pass, mesh)
    expect(box.x0).toBeCloseTo(bounds.x0, 6)
    expect(box.y0).toBeCloseTo(bounds.y0, 6)
    expect(box.y1).toBeCloseTo(bounds.y1, 6)
    expect(box.x1).toBeCloseTo(mesh.x1 + (mesh.x1 - mesh.x0), 6)
  })

  it('meets tiles across the antimeridian through the next world copy', () => {
    // A region unwrapped past 180 degrees spans x = 0.99..1.01. The tile on
    // the far side lives at x = 0..1/64 in the base world; its copy one
    // world east is what overlaps the mesh, and it joins the tile west of
    // the antimeridian into one box across it.
    const mesh = { x0: 0.99, y0: 0.4, x1: 1.01, y1: 0.45 }
    const west = tile(6, 63, 28)
    const east = tile(6, 0, 28)

    const [pass] = tileClipPasses(mesh, [west, east])

    const boxes = absolute(pass, mesh)
    expect(boxes).toHaveLength(1)
    const westBounds = getTileBounds(west.tileId)
    const eastBounds = getTileBounds(east.tileId)
    expectBox(boxes[0], { ...westBounds, x1: eastBounds.x1 + 1 })

    // The same tile on its own, only through the copy.
    const [alone] = tileClipPasses(mesh, [east])
    expectBox(absolute(alone, mesh)[0], {
      ...eastBounds,
      x0: 1,
      x1: eastBounds.x1 + 1,
    })
  })

  it('matches a tile only in the world copy it is drawn in', () => {
    // The map may cover copies at different zooms; a tile of one copy must
    // not clip the region's draw in another, or two owners would show.
    const mesh = { x0: 0.2, y0: 0.2, x1: 0.6, y1: 0.5 }
    const west = { tileId: { z: 3, x: 2, y: 2 }, wrap: -1 }
    const bounds = getTileBounds(west.tileId)

    expect(tileClipPasses(mesh, [west], 0)).toHaveLength(0)
    const [pass] = tileClipPasses(mesh, [west], -1)
    expectBox(absolute(pass, mesh)[0], bounds)

    // A tile of every copy follows the draw instead.
    const every = { tileId: { z: 3, x: 2, y: 2 } }
    for (const offset of [-1, 0, 1]) {
      const [p] = tileClipPasses(mesh, [every], offset)
      expectBox(absolute(p, mesh)[0], bounds)
    }
  })

  it('keeps shared edges exact at high zoom', () => {
    // At zoom 20 a tile is 2^-20 of the world. The vertex shader's absolute
    // position carries the region anchor as a float32 uniform, which near
    // x = 0.7 is off by a few percent of such a tile; the box relative to
    // the anchor is exact to well under a millionth of it, and so is the
    // region-local delta it is compared against.
    const z = 20
    const n = Math.floor(0.7 * 2 ** z)
    const owner = tile(z, n, n)
    const exact = getTileBounds(owner.tileId)
    const width = exact.x1 - exact.x0
    const mesh = {
      x0: exact.x0 - width / 3,
      y0: exact.y0 - width / 3,
      x1: exact.x1 + width / 7,
      y1: exact.y1 + width / 7,
    }
    const anchorX = (mesh.x0 + mesh.x1) / 2
    expect(Math.abs(Math.fround(anchorX) - anchorX)).toBeGreaterThan(
      width * 1e-3
    )

    const [pass] = tileClipPasses(mesh, [owner])

    const [box] = absolute(pass, mesh)
    expect(Math.abs(box.x0 - exact.x0)).toBeLessThan(width * 1e-6)
    expect(Math.abs(box.x1 - exact.x1)).toBeLessThan(width * 1e-6)
  })

  it('extends polar edge tiles to the pole on the globe', () => {
    const north = tileClipBounds({ tileId: { z: 2, x: 1, y: 0 }, poles: true })
    const south = tileClipBounds({ tileId: { z: 2, x: 1, y: 3 }, poles: true })
    const flat = tileClipBounds({ tileId: { z: 2, x: 1, y: 0 } })

    expect(north.y0).toBeLessThan(-1)
    expect(north.y1).toBeCloseTo(0.25, 6)
    expect(south.y0).toBeCloseTo(0.75, 6)
    expect(south.y1).toBeGreaterThan(2)
    expect(flat).toEqual(getTileBounds({ z: 2, x: 1, y: 0 }))

    const mesh = { x0: 0.25, y0: -0.3, x1: 0.5, y1: 0.3 }
    const [pass] = tileClipPasses(mesh, [
      { tileId: { z: 2, x: 1, y: 0 }, poles: true },
    ])
    expect(absolute(pass, mesh)[0].y0).toBeLessThan(-1)
  })
})
