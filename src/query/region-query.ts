/**
 * @module region-query
 *
 * Region query implementation for zarr-layer.
 * Queries all data points within a GeoJSON polygon.
 */

import type { TileTuple, MercatorBounds, XYLimits } from '../map-utils'
import { tileToKey } from '../map-utils'
import type { Tiles, TileDataCache } from '../tiles'
import type { ZarrStore } from '../zarr-store'
import type { CRS } from '../types'
import { queryPointTiled } from './point-query'
import type {
  QueryGeometry,
  QuerySelector,
  RegionQueryResult,
  RegionValues,
} from './types'
import {
  computeBoundingBox,
  getTilesForPolygon,
  pointInGeoJSON,
  tilePixelToLatLon,
  mercatorBoundsToPixel,
} from './query-utils'
import {
  hasArraySelector,
  setObjectValues,
  getSelectorHash,
  getChunksForSelector,
} from './selector-utils'

/**
 * Query a region in tiled mode.
 */
export async function queryRegionTiled(
  geometry: QueryGeometry,
  selector: QuerySelector,
  tilesManager: Tiles,
  zarrStore: ZarrStore,
  crs: CRS,
  xyLimits: XYLimits,
  maxZoom: number,
  tileSize: number
): Promise<RegionQueryResult> {
  const desc = zarrStore.describe()
  const dimensions = desc.dimensions
  const coordinates = desc.coordinates

  // Determine if results should be nested based on selector
  const useNestedResults = hasArraySelector(selector)
  let results: RegionValues = useNestedResults ? {} : []

  const lat: number[] = []
  const lon: number[] = []
  const bandResults: Record<string, number[]> = {}

  // Get tiles that intersect the polygon
  const tiles = getTilesForPolygon(geometry, maxZoom, crs, xyLimits)

  // Fetch all required tiles
  const selectorHash = getSelectorHash(selector)
  await Promise.all(
    tiles.map(async (tileTuple) => {
      let tileData = tilesManager.getTile(tileTuple) || null
      if (
        !tileData ||
        !tileData.data ||
        tileData.selectorHash !== selectorHash
      ) {
        await tilesManager.fetchTile(tileTuple, selectorHash)
      }
    })
  )

  // Iterate over tiles and pixels
  for (const tileTuple of tiles) {
    const tileData = tilesManager.getTile(tileTuple)
    if (!tileData || !tileData.data) continue

    const [z, x, y] = tileTuple
    const channels = tileData.channels || 1

    for (let pixelY = 0; pixelY < tileSize; pixelY++) {
      for (let pixelX = 0; pixelX < tileSize; pixelX++) {
        // Convert pixel to geographic coordinates
        const geo = tilePixelToLatLon(
          tileTuple,
          pixelX,
          pixelY,
          tileSize,
          crs,
          xyLimits
        )

        // Test if point is inside polygon
        if (!pointInGeoJSON([geo.lon, geo.lat], geometry)) {
          continue
        }

        // Point is inside - collect values
        lon.push(geo.lon)
        lat.push(geo.lat)

        const dataIndex = pixelY * tileSize * channels + pixelX * channels
        const value = tileData.data[dataIndex]

        if (useNestedResults) {
          // For nested results, we need to determine the keys based on selector
          const keys = getSelectorKeys(selector, dimensions, coordinates)
          if (keys.length > 0) {
            setObjectValues(results, keys, value)
          } else {
            // Should not happen with array selector, but fallback
            if (Array.isArray(results)) {
              results.push(value)
            }
          }
        } else {
          // Flat array results
          if (Array.isArray(results)) {
            results.push(value)
          }
        }

        if (tileData.bandData && tileData.bandData.size > 0) {
          const bandIndex = pixelY * tileSize + pixelX
          for (const [bandName, bandData] of tileData.bandData) {
            const arr = bandResults[bandName] || []
            arr.push(bandData[bandIndex])
            bandResults[bandName] = arr
          }
        }
      }
    }
  }

  // Build coordinates object
  const coordsResult: Record<string, (number | string)[]> = { lat, lon }

  // Add non-spatial dimension coordinates from selector
  for (const dim of dimensions) {
    const dimLower = dim.toLowerCase()
    if (['x', 'lon', 'longitude', 'y', 'lat', 'latitude'].includes(dimLower)) {
      continue
    }

    const selectorValue = selector[dim]
    if (selectorValue !== undefined) {
      if (Array.isArray(selectorValue)) {
        coordsResult[dim] = selectorValue
      } else if (
        typeof selectorValue === 'object' &&
        'selected' in selectorValue
      ) {
        const selected = selectorValue.selected
        coordsResult[dim] = Array.isArray(selected) ? selected : [selected]
      } else {
        coordsResult[dim] = [selectorValue]
      }
    } else if (coordinates[dim]) {
      coordsResult[dim] = coordinates[dim]
    }
  }

  const hasValues =
    (Array.isArray(results) && results.length > 0) ||
    (!Array.isArray(results) && Object.keys(results).length > 0)

  let fallbackBandValues: Record<string, RegionValues> | undefined

  // Fallback: if the polygon is smaller than a pixel and missed all centers,
  // sample the polygon center with a point query so viewport queries still return.
  if (!hasValues) {
    const bbox = computeBoundingBox(geometry)
    const centerLng = (bbox.west + bbox.east) / 2
    const centerLat = (bbox.south + bbox.north) / 2
    const pointResult = await queryPointTiled(
      centerLng,
      centerLat,
      tilesManager,
      zarrStore,
      selector,
      crs,
      xyLimits,
      maxZoom,
      tileSize
    )

    lat.push(pointResult.lat)
    lon.push(pointResult.lng)

    const value = pointResult.value ?? Number.NaN
    if (useNestedResults) {
      const keys = getSelectorKeys(selector, dimensions, coordinates)
      setObjectValues(results, keys, value)
    } else {
      if (Array.isArray(results)) {
        results.push(value)
      }
    }

    if (pointResult.bandValues) {
      fallbackBandValues = {}
      for (const [band, bandValue] of Object.entries(pointResult.bandValues)) {
        fallbackBandValues[band] = [bandValue ?? Number.NaN]
      }
    }
  }

  return {
    values: results,
    dimensions: dimensions.map((d) => {
      const lower = d.toLowerCase()
      if (['x', 'lon', 'longitude'].includes(lower)) return 'lon'
      if (['y', 'lat', 'latitude'].includes(lower)) return 'lat'
      return d
    }),
    coordinates: coordsResult as RegionQueryResult['coordinates'],
    bandValues:
      Object.keys(bandResults).length > 0 ? bandResults : fallbackBandValues,
  }
}

/**
 * Query a region in single-image mode.
 */
export function queryRegionSingleImage(
  geometry: QueryGeometry,
  selector: QuerySelector,
  data: Float32Array | null,
  width: number,
  height: number,
  bounds: MercatorBounds,
  crs: CRS,
  dimensions: string[],
  coordinates: Record<string, (string | number)[]>
): RegionQueryResult {
  const useNestedResults = hasArraySelector(selector)
  let results: RegionValues = useNestedResults ? {} : []

  const lat: number[] = []
  const lon: number[] = []

  if (!data) {
    return {
      values: results,
      dimensions: ['lat', 'lon'],
      coordinates: { lat, lon },
    }
  }

  // Iterate over all pixels in the image
  for (let pixelY = 0; pixelY < height; pixelY++) {
    for (let pixelX = 0; pixelX < width; pixelX++) {
      // Convert pixel to geographic coordinates
      const geo = pixelToGeo(pixelX, pixelY, width, height, bounds, crs)
      if (!geo) continue

      // Test if point is inside polygon
      if (!pointInGeoJSON([geo.lon, geo.lat], geometry)) {
        continue
      }

      // Point is inside - collect values
      lon.push(geo.lon)
      lat.push(geo.lat)

      const dataIndex = pixelY * width + pixelX
      const value = data[dataIndex]

      if (useNestedResults) {
        const keys = getSelectorKeys(selector, dimensions, coordinates)
        if (keys.length > 0) {
          setObjectValues(results, keys, value)
        } else {
          if (Array.isArray(results)) {
            results.push(value)
          }
        }
      } else {
        if (Array.isArray(results)) {
          results.push(value)
        }
      }
    }
  }

  // Build coordinates object
  const coordsResult: Record<string, (number | string)[]> = { lat, lon }

  for (const dim of dimensions) {
    const dimLower = dim.toLowerCase()
    if (['x', 'lon', 'longitude', 'y', 'lat', 'latitude'].includes(dimLower)) {
      continue
    }

    const selectorValue = selector[dim]
    if (selectorValue !== undefined) {
      if (Array.isArray(selectorValue)) {
        coordsResult[dim] = selectorValue
      } else if (
        typeof selectorValue === 'object' &&
        'selected' in selectorValue
      ) {
        const selected = selectorValue.selected
        coordsResult[dim] = Array.isArray(selected) ? selected : [selected]
      } else {
        coordsResult[dim] = [selectorValue]
      }
    } else if (coordinates[dim]) {
      coordsResult[dim] = coordinates[dim]
    }
  }

  return {
    values: results,
    dimensions: dimensions.map((d) => {
      const lower = d.toLowerCase()
      if (['x', 'lon', 'longitude'].includes(lower)) return 'lon'
      if (['y', 'lat', 'latitude'].includes(lower)) return 'lat'
      return d
    }),
    coordinates: coordsResult as RegionQueryResult['coordinates'],
  }
}

/**
 * Gets the keys for nested results based on selector.
 */
function getSelectorKeys(
  selector: QuerySelector,
  dimensions: string[],
  coordinates: Record<string, (string | number)[]>
): (string | number)[] {
  const keys: (string | number)[] = []

  for (const dim of dimensions) {
    const dimLower = dim.toLowerCase()
    if (['x', 'lon', 'longitude', 'y', 'lat', 'latitude'].includes(dimLower)) {
      continue
    }

    const selectorValue = selector[dim]
    if (selectorValue === undefined) {
      // No selector - include all coordinate values as keys
      if (coordinates[dim] && coordinates[dim].length > 1) {
        // This would need iteration - handled differently in full implementation
      }
    } else if (Array.isArray(selectorValue)) {
      // Array selector - first value as key (full implementation needs iteration)
      if (selectorValue.length > 0) {
        keys.push(selectorValue[0])
      }
    } else if (
      typeof selectorValue === 'object' &&
      'selected' in selectorValue
    ) {
      const selected = selectorValue.selected
      if (Array.isArray(selected) && selected.length > 0) {
        keys.push(selected[0])
      }
    }
    // Single value selectors don't contribute keys
  }

  return keys
}

/**
 * Converts pixel coordinates to geographic coordinates for single-image mode.
 */
function pixelToGeo(
  pixelX: number,
  pixelY: number,
  width: number,
  height: number,
  bounds: MercatorBounds,
  crs: CRS
): { lat: number; lon: number } | null {
  const normX = pixelX / width
  const normY = pixelY / height

  // Convert normalized coordinates to mercator
  const mercX = bounds.x0 + normX * (bounds.x1 - bounds.x0)
  const mercY = bounds.y0 + normY * (bounds.y1 - bounds.y0)

  // Convert mercator to geographic
  const lon = mercX * 360 - 180

  if (
    crs === 'EPSG:4326' &&
    bounds.latMin !== undefined &&
    bounds.latMax !== undefined
  ) {
    // For equirectangular data, use linear lat mapping
    const lat = bounds.latMax - normY * (bounds.latMax - bounds.latMin)
    return { lat, lon }
  }

  // Invert mercator Y to latitude
  const y2 = 180 - mercY * 360
  const lat = (360 / Math.PI) * Math.atan(Math.exp((y2 * Math.PI) / 180)) - 90

  return { lat, lon }
}
