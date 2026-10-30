/**
 * @module clip-boxes
 *
 * Fragment clip boxes for the direct render path. The map's tile cover
 * partitions the view, and each tile shows one level, so a region is drawn
 * only inside the tiles that show its level. The boxes are those tiles in
 * normalized Mercator, expressed relative to the region's mesh anchor: the
 * shader compares them against the region-local delta it already carries, so
 * a shared tile edge lands on the same float32 value in every draw and the
 * absolute coordinate's float32 rounding never reaches the pixel.
 */

import type { MeshMercatorBounds } from './map-utils'
import type { TileId } from './renderer-types'
import { MAX_CLIP_BOXES } from './shaders'
import { getTileBounds } from './tile-utils'

/** A tile a region is clipped to. */
export interface ClipTile {
  tileId: TileId
  /** Extend a polar edge tile to the pole: the globe draws there. */
  poles?: boolean
  /**
   * The world copy the tile is drawn in, as the map's cover numbers them;
   * undefined stands for every copy.
   */
  wrap?: number
}

interface Box {
  x0: number
  y0: number
  x1: number
  y1: number
}

/** Far beyond any Mercator y a rendered polar vertex reaches. */
const POLE_EXTENT = 1e3

/**
 * A tile's Mercator bounds for clipping. On the globe a polar edge tile
 * reaches past y = 0 or y = 1 to cover the pole cap drawn there.
 */
export function tileClipBounds({ tileId, poles }: ClipTile): Box {
  const bounds: Box = getTileBounds(tileId)
  if (!poles) return bounds
  if (tileId.y === 0) bounds.y0 = -POLE_EXTENT
  if (tileId.y === 2 ** tileId.z - 1) bounds.y1 = POLE_EXTENT
  return bounds
}

const overlaps = (a: Box, b: Box) =>
  a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0

/**
 * Merge boxes that share a row (same y extent) and touch along x, so a block
 * of n by m tiles costs m boxes rather than n by m.
 */
function mergeRows(boxes: Box[]): Box[] {
  const rows = new Map<string, Box[]>()
  for (const box of boxes) {
    const row = `${box.y0}:${box.y1}`
    const list = rows.get(row)
    if (list) list.push(box)
    else rows.set(row, [box])
  }
  const merged: Box[] = []
  for (const list of rows.values()) {
    list.sort((a, b) => a.x0 - b.x0)
    let current = { ...list[0] }
    for (let i = 1; i < list.length; i++) {
      const box = list[i]
      if (box.x0 <= current.x1 + 1e-9) {
        current.x1 = Math.max(current.x1, box.x1)
      } else {
        merged.push(current)
        current = { ...box }
      }
    }
    merged.push(current)
  }
  return merged
}

/**
 * Clip boxes for one draw of a region at a world offset, split into passes
 * of at most MAX_CLIP_BOXES. Each pass draws the region once; together they
 * show it in every owning tile. A tile is matched in the world copy it is
 * drawn in, so copies the map covers at different zooms keep one owner per
 * point each; a tile of every copy is matched in the copy drawn and its
 * neighbours, so a region unwrapped past the antimeridian meets the tiles
 * on the far side. Returns no passes when no tile touches the mesh.
 */
export function tileClipPasses(
  mesh: MeshMercatorBounds,
  tiles: ClipTile[],
  worldOffset = 0
): Float32Array[] {
  const anchorX = (mesh.x0 + mesh.x1) / 2
  const anchorY = (mesh.y0 + mesh.y1) / 2
  const boxes: Box[] = []
  for (const tile of tiles) {
    const bounds = tileClipBounds(tile)
    const copies =
      tile.wrap === undefined
        ? [worldOffset - 1, worldOffset, worldOffset + 1]
        : [tile.wrap]
    for (const copy of copies) {
      const box = {
        x0: bounds.x0 + copy - worldOffset,
        y0: bounds.y0,
        x1: bounds.x1 + copy - worldOffset,
        y1: bounds.y1,
      }
      if (overlaps(box, mesh)) boxes.push(box)
    }
  }
  const merged = mergeRows(boxes)
  const passes: Float32Array[] = []
  for (let start = 0; start < merged.length; start += MAX_CLIP_BOXES) {
    const chunk = merged.slice(start, start + MAX_CLIP_BOXES)
    const pass = new Float32Array(chunk.length * 4)
    chunk.forEach((box, i) => {
      pass[i * 4] = box.x0 - anchorX
      pass[i * 4 + 1] = box.y0 - anchorY
      pass[i * 4 + 2] = box.x1 - anchorX
      pass[i * 4 + 3] = box.y1 - anchorY
    })
    passes.push(pass)
  }
  return passes
}
