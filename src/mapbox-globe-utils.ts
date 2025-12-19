/**
 * @module mapbox-globe-utils
 *
 * Utility functions for Mapbox GL JS globe tile rendering (renderToTile API).
 * These are NOT used by MapLibre globe, which uses projectTile() in the shader.
 */

import type { TileId } from './zarr-mode'

/** Identity matrix for Mapbox globe tile rendering (no additional transformation) */
export const MAPBOX_GLOBE_IDENTITY_MATRIX = new Float32Array([
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
])

/**
 * Creates a 4x4 transformation matrix for Mapbox globe tile rendering.
 * Used by renderToTile() to transform from normalized Mercator [0,1] space to clip space [-1,1].
 *
 * @param tileX0 - Left edge of tile in normalized Mercator X [0,1]
 * @param tileY0 - Top edge of tile in normalized Mercator Y [0,1]
 * @param tileX1 - Right edge of tile in normalized Mercator X [0,1]
 * @param tileY1 - Bottom edge of tile in normalized Mercator Y [0,1]
 * @returns 4x4 column-major transformation matrix
 */
export function createMapboxTileMatrix(
  tileX0: number,
  tileY0: number,
  tileX1: number,
  tileY1: number
): Float32Array {
  const x0 = Math.max(0, tileX0)
  const x1 = Math.min(1, tileX1)
  const y0 = Math.max(0, tileY0)
  const y1 = Math.min(1, tileY1)
  const width = x1 - x0
  const height = y1 - y0

  return new Float32Array([
    2 / width,
    0,
    0,
    0,
    0,
    2 / height,
    0,
    0,
    0,
    0,
    1,
    0,
    -(x0 + x1) / width,
    -(y0 + y1) / height,
    0,
    1,
  ])
}

/**
 * Get normalized Mercator bounds for a Mapbox globe tile.
 * Used by renderToTile() to determine tile coverage.
 *
 * @param tileId - Tile coordinates (z, x, y)
 * @returns Bounds in normalized Mercator space [0,1]
 */
export function getMapboxTileBounds(tileId: TileId): {
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
 * Check if two bounds intersect.
 */
export function boundsIntersect(
  a: { x0: number; y0: number; x1: number; y1: number },
  b: { x0: number; y0: number; x1: number; y1: number }
): boolean {
  return a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0
}
