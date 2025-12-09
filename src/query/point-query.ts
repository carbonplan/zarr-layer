/**
 * @module point-query
 *
 * Point query implementation for zarr-layer.
 * Queries a single geographic point and returns the data value.
 */

import type { TileTuple, MercatorBounds, XYLimits } from '../map-utils'
import { tileToKey } from '../map-utils'
import type { Tiles, TileDataCache } from '../tiles'
import type { ZarrStore } from '../zarr-store'
import type { CRS } from '../types'
import type { PointQueryResult, QuerySelector } from './types'
import {
  geoToTile,
  geoToTileFraction,
  mercatorBoundsToPixel,
} from './query-utils'
import { getSelectorHash } from './selector-utils'

/**
 * Query a point in tiled mode.
 */
export async function queryPointTiled(
  lng: number,
  lat: number,
  tilesManager: Tiles,
  zarrStore: ZarrStore,
  selector: QuerySelector,
  crs: CRS,
  xyLimits: XYLimits,
  maxZoom: number,
  tileSize: number
): Promise<PointQueryResult> {
  // Use maxZoom for highest resolution query
  const tile = geoToTile(lng, lat, maxZoom, crs, xyLimits)
  const [z, x, y] = tile

  // Get fractional position within tile
  const { fracX, fracY } = geoToTileFraction(lng, lat, tile, crs, xyLimits)

  // Convert to pixel indices
  const pixelX = Math.floor(fracX * tileSize)
  const pixelY = Math.floor(fracY * tileSize)

  // Clamp to valid range
  const clampedPixelX = Math.max(0, Math.min(pixelX, tileSize - 1))
  const clampedPixelY = Math.max(0, Math.min(pixelY, tileSize - 1))

  // Check if pixel is within tile bounds
  if (pixelX < 0 || pixelX >= tileSize || pixelY < 0 || pixelY >= tileSize) {
    return {
      lng,
      lat,
      value: null,
      tile: { z, x, y },
      pixel: { x: clampedPixelX, y: clampedPixelY },
    }
  }

  // Get or fetch tile data
  const selectorHash = getSelectorHash(selector)
  let tileData: TileDataCache | null = tilesManager.getTile(tile) || null

  if (!tileData || !tileData.data || tileData.selectorHash !== selectorHash) {
    // Fetch tile if not cached or selector changed
    tileData = await tilesManager.fetchTile(tile, selectorHash)
  }

  if (!tileData || !tileData.data) {
    return {
      lng,
      lat,
      value: null,
      tile: { z, x, y },
      pixel: { x: clampedPixelX, y: clampedPixelY },
    }
  }

  // Extract value from tile data
  const channels = tileData.channels || 1
  const dataIndex = clampedPixelY * tileSize * channels + clampedPixelX * channels
  const value = tileData.data[dataIndex]

  // Get band values if multi-band
  let bandValues: Record<string, number | null> | undefined
  if (tileData.bandData && tileData.bandData.size > 0) {
    bandValues = {}
    for (const [bandName, bandData] of tileData.bandData) {
      const bandIndex = clampedPixelY * tileSize + clampedPixelX
      bandValues[bandName] = bandData[bandIndex] ?? null
    }
  }

  return {
    lng,
    lat,
    value: value ?? null,
    bandValues,
    tile: { z, x, y },
    pixel: { x: clampedPixelX, y: clampedPixelY },
  }
}

/**
 * Query a point in single-image mode.
 */
export function queryPointSingleImage(
  lng: number,
  lat: number,
  data: Float32Array | null,
  width: number,
  height: number,
  bounds: MercatorBounds,
  crs: CRS
): PointQueryResult {
  if (!data) {
    return { lng, lat, value: null }
  }

  const pixel = mercatorBoundsToPixel(lng, lat, bounds, width, height, crs)

  if (!pixel) {
    return { lng, lat, value: null }
  }

  const { x, y } = pixel
  const index = y * width + x
  const value = data[index]

  return {
    lng,
    lat,
    value: value ?? null,
    pixel: { x, y },
  }
}
