/**
 * @module region-query
 *
 * Region query implementation for zarr-layer.
 * Queries all data points within a GeoJSON polygon.
 * Matches carbonplan/maps structure and behavior.
 */

import type { MercatorBounds, XYLimits } from '../map-utils'
import {
  latToMercatorNorm,
  lonToMercatorNorm,
  mercatorNormToLat,
  mercatorNormToLon,
  parseLevelZoom,
  tileToKey,
} from '../map-utils'
import type { ZarrStore } from '../zarr-store'
import type { Bounds, CRS, Selector } from '../types'
import type {
  QueryGeometry,
  QueryResult,
  QueryDataValues,
  QueryTransformOptions,
} from './types'
import {
  computeBoundingBox,
  getTilesForPolygon,
  pixelIntersectsGeometrySingle,
  pixelIntersectsGeometryTiled,
  tilePixelToLatLon,
  pixelToLatLon,
  CachedTransformer,
} from './query-utils'
import { createWGS84ToSourceTransformer } from '../projection-utils'
import { setObjectValues, getChunks, getPointValues } from './selector-utils'

/**
 * Apply scale_factor/add_offset transforms and filter invalid values.
 * Returns null if value should be filtered out.
 */
function transformValue(
  value: number,
  transforms?: QueryTransformOptions
): number | null {
  if (!Number.isFinite(value)) return null

  if (transforms?.fillValue !== undefined && transforms.fillValue !== null) {
    if (value === transforms.fillValue) return null
  }

  let result = value
  if (transforms?.scaleFactor !== undefined && transforms.scaleFactor !== 1) {
    result *= transforms.scaleFactor
  }
  if (transforms?.addOffset !== undefined && transforms.addOffset !== 0) {
    result += transforms.addOffset
  }

  return result
}

/**
 * Query a region in tiled mode.
 * Returns structure matching carbonplan/maps: { [variable]: values, dimensions, coordinates }
 */
export async function queryRegionTiled(
  variable: string,
  geometry: QueryGeometry,
  selector: Selector,
  zarrStore: ZarrStore,
  crs: CRS,
  xyLimits: XYLimits,
  levelIndex: number,
  tileSize: number,
  transforms?: QueryTransformOptions
): Promise<QueryResult> {
  const desc = zarrStore.describe()
  const dimensions = desc.dimensions
  const coordinates = desc.coordinates
  const shape = desc.shape || []
  const chunks = desc.chunks || []

  // Calculate result dimension based on selector
  // resultDim = total dims - (number of single-valued dims)
  const isMultiValSelector = (value: Selector[string]) => {
    if (Array.isArray(value)) return true
    if (value && typeof value === 'object' && 'selected' in value) {
      return Array.isArray((value as any).selected)
    }
    return false
  }
  const singleValuedDims = Object.keys(selector).filter(
    (k) => !isMultiValSelector(selector[k])
  ).length
  const resultDim = dimensions.length - singleValuedDims

  // Determine if results should be nested
  const useNestedResults = resultDim > 2
  let results: QueryDataValues = useNestedResults ? {} : []

  const latCoords: number[] = []
  const lonCoords: number[] = []

  const mappedDimensions = dimensions.map((d) => {
    const dimLower = d.toLowerCase()
    if (['x', 'lon', 'longitude'].includes(dimLower)) return 'lon'
    if (['y', 'lat', 'latitude'].includes(dimLower)) return 'lat'
    return d
  })

  const resultDimensions = useNestedResults ? mappedDimensions : ['lat', 'lon']

  const buildResultCoordinates = (): Record<string, (number | string)[]> => {
    const coords: Record<string, (number | string)[]> = {
      lat: latCoords,
      lon: lonCoords,
    }

    if (useNestedResults) {
      const addDimCoordinates = (dim: string) => {
        const dimLower = dim.toLowerCase()
        if (
          ['x', 'lon', 'longitude', 'y', 'lat', 'latitude'].includes(dimLower)
        ) {
          return
        }

        const sel = selector[dim]
        let values: (number | string)[] | undefined

        if (Array.isArray(sel)) {
          values = sel as (number | string)[]
        } else if (sel && typeof sel === 'object' && 'selected' in sel) {
          const selected = sel.selected
          values = Array.isArray(selected) ? selected : [selected]
        } else if (sel !== undefined && typeof sel !== 'object') {
          values = [sel]
        } else if (coordinates[dim]) {
          values = coordinates[dim]
        }

        if (values) {
          coords[dim] = values
        }
      }

      dimensions.forEach(addDimCoordinates)
    }

    return coords
  }

  // Get level path for chunk fetching
  const levelPath = zarrStore.levels[levelIndex]
  if (!levelPath) {
    throw new Error(`No level path found for level index ${levelIndex}`)
  }

  // Parse actual zoom from level path to handle pyramids that don't start at 0
  const actualZoom = parseLevelZoom(levelPath, levelIndex)

  // Get tiles that intersect the polygon
  const tiles = getTilesForPolygon(geometry, actualZoom, crs, xyLimits)
  if (tiles.length === 0) {
    // Return empty result in carbonplan/maps format
    const result = {
      [variable]: results,
      dimensions: resultDimensions,
      coordinates: buildResultCoordinates(),
    } as QueryResult
    return result
  }

  // For each tile, determine which chunks we need based on selector and fetch them
  const tileChunkData = new Map<string, Map<string, Float32Array>>()

  for (const tileTuple of tiles) {
    const [, x, y] = tileTuple
    const chunksToFetch = getChunks(
      selector,
      dimensions,
      coordinates,
      shape,
      chunks,
      x,
      y
    )

    const tileKey = tileToKey(tileTuple)
    const chunkDataMap = new Map<string, Float32Array>()

    // Fetch all chunks for this tile
    await Promise.all(
      chunksToFetch.map(async (chunkIndices) => {
        try {
          const chunk = await zarrStore.getChunk(levelPath, chunkIndices)
          // Make a proper copy to avoid buffer sharing issues
          const chunkData = new Float32Array(chunk.data as ArrayLike<number>)

          // Store chunk with indices as key
          const chunkKey = chunkIndices.join(',')
          chunkDataMap.set(chunkKey, chunkData)
        } catch (err) {
          console.warn(
            `Failed to fetch chunk ${chunkIndices} for tile ${tileKey}:`,
            err
          )
        }
      })
    )

    tileChunkData.set(tileKey, chunkDataMap)
  }

  // Iterate over tiles and pixels to extract values
  for (const tileTuple of tiles) {
    const tileKey = tileToKey(tileTuple)
    const chunkDataMap = tileChunkData.get(tileKey)
    if (!chunkDataMap || chunkDataMap.size === 0) continue

    // Get all chunk indices for this tile
    const [, x, y] = tileTuple
    const chunksForTile = getChunks(
      selector,
      dimensions,
      coordinates,
      shape,
      chunks,
      x,
      y
    )

    for (let pixelY = 0; pixelY < tileSize; pixelY++) {
      for (let pixelX = 0; pixelX < tileSize; pixelX++) {
        if (
          !pixelIntersectsGeometryTiled(
            tileTuple,
            pixelX,
            pixelY,
            tileSize,
            crs,
            xyLimits,
            geometry
          )
        ) {
          continue
        }

        // Collect all values for this pixel first
        const pixelValues: { keys: (string | number)[]; value: number }[] = []

        for (const chunkIndices of chunksForTile) {
          const chunkKey = chunkIndices.join(',')
          const chunkData = chunkDataMap.get(chunkKey)
          if (!chunkData) continue

          const valuesToSet = getPointValues(
            chunkData,
            pixelX,
            pixelY,
            selector,
            dimensions,
            coordinates,
            shape,
            chunks,
            chunkIndices
          )

          for (const { keys, value } of valuesToSet) {
            const transformed = transformValue(value, transforms)
            if (transformed !== null) {
              pixelValues.push({ keys, value: transformed })
            }
          }
        }

        // Only add coordinates and values if we have valid data
        if (pixelValues.length === 0) continue

        const geo = tilePixelToLatLon(
          tileTuple,
          pixelX + 0.5,
          pixelY + 0.5,
          tileSize,
          crs,
          xyLimits
        )

        latCoords.push(geo.lat)
        lonCoords.push(geo.lon)

        for (const { keys, value } of pixelValues) {
          if (keys.length > 0) {
            setObjectValues(results, keys, value)
          } else if (Array.isArray(results)) {
            results.push(value)
          }
        }
      }
    }
  }

  const result = {
    [variable]: results,
    dimensions: resultDimensions,
    coordinates: buildResultCoordinates(),
  } as QueryResult

  return result
}

/**
 * Query a region in single-image mode.
 * Returns structure matching carbonplan/maps: { [variable]: values, dimensions, coordinates }
 */
export async function queryRegionSingleImage(
  variable: string,
  geometry: QueryGeometry,
  selector: Selector,
  data: Float32Array | null,
  width: number,
  height: number,
  bounds: MercatorBounds,
  _crs: CRS,
  dimensions: string[],
  coordinates: Record<string, (string | number)[]>,
  channels: number = 1,
  channelLabels?: (string | number)[][],
  multiValueDimNames?: string[],
  latIsAscending?: boolean,
  transforms?: QueryTransformOptions,
  proj4def?: string | null,
  sourceBounds?: Bounds | null
): Promise<QueryResult> {
  // Calculate result dimension
  const isMultiValSelector = (value: Selector[string]) => {
    if (Array.isArray(value)) return true
    if (value && typeof value === 'object' && 'selected' in value) {
      return Array.isArray((value as any).selected)
    }
    return false
  }
  const singleValuedDims = Object.keys(selector).filter(
    (k) => !isMultiValSelector(selector[k])
  ).length
  const resultDim = dimensions.length - singleValuedDims

  // Determine if results should be nested
  const useNestedResults = resultDim > 2
  let results: QueryDataValues = useNestedResults ? {} : []

  const latCoords: number[] = []
  const lonCoords: number[] = []

  const mappedDimensions = dimensions.map((d) => {
    const dimLower = d.toLowerCase()
    if (['x', 'lon', 'longitude'].includes(dimLower)) return 'lon'
    if (['y', 'lat', 'latitude'].includes(dimLower)) return 'lat'
    return d
  })

  const resultDimensions = useNestedResults ? mappedDimensions : ['lat', 'lon']

  const buildResultCoordinates = (): Record<string, (number | string)[]> => {
    const coords: Record<string, (number | string)[]> = {
      lat: latCoords,
      lon: lonCoords,
    }

    if (useNestedResults) {
      const addDimCoordinates = (dim: string) => {
        const dimLower = dim.toLowerCase()
        if (
          ['x', 'lon', 'longitude', 'y', 'lat', 'latitude'].includes(dimLower)
        ) {
          return
        }

        const sel = selector[dim]
        let values: (number | string)[] | undefined

        if (Array.isArray(sel)) {
          values = sel as (number | string)[]
        } else if (sel && typeof sel === 'object' && 'selected' in sel) {
          const selected = sel.selected
          values = Array.isArray(selected) ? selected : [selected]
        } else if (sel !== undefined && typeof sel !== 'object') {
          values = [sel]
        } else if (coordinates[dim]) {
          values = coordinates[dim]
        }

        if (values) {
          coords[dim] = values
        }
      }

      dimensions.forEach(addDimCoordinates)
    }

    return coords
  }

  if (!data) {
    const result = {
      [variable]: results,
      dimensions: resultDimensions,
      coordinates: buildResultCoordinates(),
    } as QueryResult
    return result
  }

  // Compute iteration bounds
  // For proj4, the data is already the subset matching the geometry bbox,
  // so iterate over all pixels. For standard CRS, compute overlap.
  let xStart = 0
  let xEnd = width
  let yStart = 0
  let yEnd = height

  if (!proj4def) {
    const bbox = computeBoundingBox(geometry)

    // Compute overlap of polygon bbox with image bounds in mercator space
    const polyX0 = lonToMercatorNorm(bbox.west)
    const polyX1 = lonToMercatorNorm(bbox.east)
    const polyY0 = latToMercatorNorm(bbox.north)
    const polyY1 = latToMercatorNorm(bbox.south)

    const overlapX0 = Math.max(bounds.x0, Math.min(polyX0, polyX1))
    const overlapX1 = Math.min(bounds.x1, Math.max(polyX0, polyX1))

    if (
      _crs === 'EPSG:4326' &&
      bounds.latMin !== undefined &&
      bounds.latMax !== undefined
    ) {
      // For equirectangular data, compute Y overlap in linear latitude space.
      const latMax = bounds.latMax
      const latMin = bounds.latMin
      const clampedNorth = Math.min(Math.max(bbox.north, latMin), latMax)
      const clampedSouth = Math.min(Math.max(bbox.south, latMin), latMax)

      const latRange = latMax - latMin
      if (latRange === 0) {
        const result = {
          [variable]: results,
          dimensions: resultDimensions,
          coordinates: buildResultCoordinates(),
        } as QueryResult
        return result
      }
      const toFrac = (latVal: number) =>
        latIsAscending
          ? (latVal - latMin) / latRange
          : (latMax - latVal) / latRange
      const yStartFracRaw = toFrac(clampedNorth)
      const yEndFracRaw = toFrac(clampedSouth)
      const yFracMin = Math.min(yStartFracRaw, yEndFracRaw)
      const yFracMax = Math.max(yStartFracRaw, yEndFracRaw)

      if (overlapX1 <= overlapX0 || yFracMax <= yFracMin) {
        const result = {
          [variable]: results,
          dimensions: resultDimensions,
          coordinates: buildResultCoordinates(),
        } as QueryResult
        return result
      }

      const minX = ((overlapX0 - bounds.x0) / (bounds.x1 - bounds.x0)) * width
      const maxX = ((overlapX1 - bounds.x0) / (bounds.x1 - bounds.x0)) * width

      xStart = Math.max(0, Math.floor(minX))
      xEnd = Math.min(width, Math.ceil(maxX + 1))
      yStart = Math.max(0, Math.floor(yFracMin * height))
      yEnd = Math.min(height, Math.ceil(yFracMax * height))
    } else {
      const overlapY0 = Math.max(bounds.y0, Math.min(polyY0, polyY1))
      const overlapY1 = Math.min(bounds.y1, Math.max(polyY0, polyY1))

      if (overlapX1 <= overlapX0 || overlapY1 <= overlapY0) {
        const result = {
          [variable]: results,
          dimensions: resultDimensions,
          coordinates: buildResultCoordinates(),
        } as QueryResult
        return result
      }

      const minX = ((overlapX0 - bounds.x0) / (bounds.x1 - bounds.x0)) * width
      const maxX = ((overlapX1 - bounds.x0) / (bounds.x1 - bounds.x0)) * width
      const minY = ((overlapY0 - bounds.y0) / (bounds.y1 - bounds.y0)) * height
      const maxY = ((overlapY1 - bounds.y0) / (bounds.y1 - bounds.y0)) * height

      xStart = Math.max(0, Math.floor(minX))
      xEnd = Math.min(width, Math.ceil(maxX + 1))
      yStart = Math.max(0, Math.floor(minY))
      yEnd = Math.min(height, Math.ceil(maxY + 1))
    }

    if (xEnd <= xStart || yEnd <= yStart) {
      const result = {
        [variable]: results,
        dimensions: resultDimensions,
        coordinates: buildResultCoordinates(),
      } as QueryResult
      return result
    }
  }

  // Create transformer once for all pixels
  const cachedTransformer: CachedTransformer | undefined = proj4def
    ? createWGS84ToSourceTransformer(proj4def)
    : undefined

  // Iterate pixels within bounding box
  for (let y = yStart; y < yEnd; y++) {
    for (let x = xStart; x < xEnd; x++) {
      if (
        !pixelIntersectsGeometrySingle(
          bounds,
          width,
          height,
          x,
          y,
          _crs,
          geometry,
          latIsAscending,
          proj4def,
          sourceBounds,
          cachedTransformer
        )
      ) {
        continue
      }

      // Use consolidated helper for pixel → lat/lon conversion
      const { lat, lon } = pixelToLatLon(
        x,
        y,
        bounds,
        width,
        height,
        _crs,
        latIsAscending,
        proj4def,
        sourceBounds,
        cachedTransformer
      )

      const baseIndex = (y * width + x) * channels

      // Collect valid values for this pixel first
      const pixelValues: { keys: (string | number)[]; value: number }[] = []

      for (let c = 0; c < channels; c++) {
        const rawValue = data[baseIndex + c]
        const transformed = transformValue(rawValue, transforms)
        if (transformed === null) continue

        if (useNestedResults && multiValueDimNames) {
          const labels = channelLabels?.[c]
          const keys =
            labels && labels.length === multiValueDimNames.length ? labels : [c]
          pixelValues.push({ keys, value: transformed })
        } else {
          pixelValues.push({ keys: [], value: transformed })
        }
      }

      // Only add coordinates and values if we have valid data
      if (pixelValues.length === 0) continue

      latCoords.push(lat)
      lonCoords.push(lon)

      for (const { keys, value } of pixelValues) {
        if (keys.length > 0) {
          setObjectValues(results, keys, value)
        } else if (Array.isArray(results)) {
          results.push(value)
        }
      }
    }
  }

  const result = {
    [variable]: results,
    dimensions: resultDimensions,
    coordinates: buildResultCoordinates(),
  } as QueryResult

  return result
}
