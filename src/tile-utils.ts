/**
 * @module tile-utils
 *
 * Helpers for drawing one slippy-map tile into a framebuffer that spans it:
 * Mapbox's renderToTile and MapLibre's renderToTerrainTile both work that way.
 */

import type { TileId } from './renderer-types'

/**
 * Creates a 4x4 matrix mapping normalized Mercator [0,1] to the tile's clip
 * space [-1,1]. The providers disagree vertically: Mapbox puts the tile's
 * north edge at clip y = -1, while MapLibre's terrain tiles put it at
 * clip y = 1 (`northUp`).
 *
 * @param tileX0 - Left edge of tile in normalized Mercator X [0,1]
 * @param tileY0 - Top (north) edge of tile in normalized Mercator Y [0,1]
 * @param tileX1 - Right edge of tile in normalized Mercator X [0,1]
 * @param tileY1 - Bottom (south) edge of tile in normalized Mercator Y [0,1]
 * @param northUp - True to place the north edge at clip y = 1
 * @returns 4x4 column-major transformation matrix
 */
export function createTileMatrix(
  tileX0: number,
  tileY0: number,
  tileX1: number,
  tileY1: number,
  northUp: boolean
): Float32Array {
  const x0 = Math.max(0, tileX0)
  const x1 = Math.min(1, tileX1)
  const y0 = Math.max(0, tileY0)
  const y1 = Math.min(1, tileY1)
  const width = x1 - x0
  const height = y1 - y0
  const ySign = northUp ? -1 : 1

  return new Float32Array([
    2 / width,
    0,
    0,
    0,
    0,
    (ySign * 2) / height,
    0,
    0,
    0,
    0,
    1,
    0,
    -(x0 + x1) / width,
    (-ySign * (y0 + y1)) / height,
    0,
    1,
  ])
}

/**
 * Get normalized Mercator bounds for a tile.
 *
 * @param tileId - Tile coordinates (z, x, y)
 * @returns Bounds in normalized Mercator space [0,1]
 */
export function getTileBounds(tileId: TileId): {
  x0: number
  y0: number
  x1: number
  y1: number
} {
  const tilesPerSide = 2 ** tileId.z
  return {
    x0: tileId.x / tilesPerSide,
    x1: (tileId.x + 1) / tilesPerSide,
    y0: tileId.y / tilesPerSide,
    y1: (tileId.y + 1) / tilesPerSide,
  }
}

/**
 * Get the WGS84 extent of a tile.
 *
 * @param tileId - Tile coordinates (z, x, y)
 * @returns Bounds in degrees
 */
export function getTileLonLatBounds(tileId: TileId): {
  west: number
  south: number
  east: number
  north: number
} {
  const { x0, y0, x1, y1 } = getTileBounds(tileId)
  const toLat = (y: number) =>
    (Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * 180) / Math.PI
  return {
    west: x0 * 360 - 180,
    east: x1 * 360 - 180,
    north: toLat(y0),
    south: toLat(y1),
  }
}

/**
 * Check if two bounds intersect.
 */
export function boundsIntersect(
  a: { x0: number; y0: number; x1: number; y1: number },
  b: { x0: number; y0: number; x1: number; y1: number }
): boolean {
  return a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0
}
