/**
 * @module region-query
 *
 * Region query implementation for zarr-layer.
 * Queries all data points within a GeoJSON polygon.
 * Matches carbonplan/maps structure and behavior.
 */

import type { MercatorBounds, TileTuple, XYLimits } from '../map-utils'
import { parseLevelZoom, tileToKey } from '../map-utils'
import type { ZarrStore } from '../zarr-store'
import type { Bounds, CRS, Selector } from '../types'
import { pixelToSourceCRS } from '../projection-utils'
import type {
  QueryGeometry,
  QueryOptions,
  QueryResult,
  QueryDataValues,
  QueryTransformOptions,
} from './types'
import {
  getTilesForPolygon,
  tilePixelToLatLon,
  pixelToLatLon,
  transformGeometryToPixelSpace,
  transformGeometryToTilePixelSpace,
  buildScanlineTable,
  CachedTransformer,
} from './query-utils'
import { createWGS84ToSourceTransformer } from '../projection-utils'
import { setObjectValues, getChunks, getPointValues } from './selector-utils'
import { SPATIAL_DIMENSION_ALIASES, SPATIAL_DIM_NAMES } from '../constants'

/**
 * Determine spatial coordinate keys for query results.
 *
 * For proj4 data we emit source-CRS values, so keys match the store's
 * original axis names (e.g. 'y'/'x').
 *
 * For standard CRS the values are always WGS84 lat/lon (from pixelToLatLon),
 * so keys are always 'lat'/'lon' regardless of what the store calls its axes.
 */
function findSpatialDimNames(
  dimensions: string[],
  isProj4: boolean
): {
  yDim: string
  xDim: string
} {
  if (!isProj4) return { yDim: 'lat', xDim: 'lon' }
  const yAliases = SPATIAL_DIMENSION_ALIASES.lat
  const xAliases = SPATIAL_DIMENSION_ALIASES.lon
  const yDim =
    dimensions.find((d) => yAliases.includes(d.toLowerCase())) ?? 'lat'
  const xDim =
    dimensions.find((d) => xAliases.includes(d.toLowerCase())) ?? 'lon'
  return { yDim, xDim }
}

function isMultiValSelector(value: Selector[string]): boolean {
  if (Array.isArray(value)) return true
  if (value && typeof value === 'object' && 'selected' in value) {
    return Array.isArray((value as any).selected)
  }
  return false
}

function checkAborted(signal?: AbortSignal) {
  if (signal?.aborted) {
    throw new DOMException('The operation was aborted.', 'AbortError')
  }
}

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
 * Process a single pixel in tiled mode: extract values from chunks and emit results.
 */
function processPixelTiled(
  pixelX: number,
  pixelY: number,
  tileTuple: [number, number, number],
  chunksForTile: number[][],
  chunkDataMap: Map<string, Float32Array>,
  selector: Selector,
  dimensions: string[],
  coordinates: Record<string, (number | string)[]>,
  shape: number[],
  chunks: number[],
  transforms: QueryTransformOptions | undefined,
  includeSpatialCoordinates: boolean,
  tileSize: number,
  crs: CRS,
  xyLimits: XYLimits,
  yCoords: number[],
  xCoords: number[],
  results: QueryDataValues,
  useNestedResults: boolean
) {
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

  if (pixelValues.length === 0) return

  if (includeSpatialCoordinates) {
    const geo = tilePixelToLatLon(
      tileTuple,
      pixelX + 0.5,
      pixelY + 0.5,
      tileSize,
      crs,
      xyLimits
    )
    yCoords.push(geo.lat)
    xCoords.push(geo.lon)
  }

  for (const { keys, value } of pixelValues) {
    if (keys.length > 0) {
      setObjectValues(results, keys, value)
    } else if (Array.isArray(results)) {
      results.push(value)
    }
  }
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
  transforms?: QueryTransformOptions,
  options?: QueryOptions
): Promise<QueryResult> {
  const { signal, includeSpatialCoordinates = true } = options ?? {}
  const desc = zarrStore.describe()
  const dimensions = desc.dimensions
  const coordinates = desc.coordinates
  const shape = desc.shape || []
  const chunks = desc.chunks || []

  // Calculate result dimension based on selector
  // resultDim = total dims - (number of single-valued dims)
  const singleValuedDims = Object.keys(selector).filter(
    (k) => !isMultiValSelector(selector[k])
  ).length
  const resultDim = dimensions.length - singleValuedDims

  // Determine if results should be nested
  const useNestedResults = resultDim > 2
  let results: QueryDataValues = useNestedResults ? {} : []

  const { yDim, xDim } = findSpatialDimNames(dimensions, false)
  const yCoords: number[] = []
  const xCoords: number[] = []

  const resultDimensions = useNestedResults ? dimensions : [yDim, xDim]

  const buildResultCoordinates = (): Record<string, (number | string)[]> => {
    const coords: Record<string, (number | string)[]> = {
      [yDim]: yCoords,
      [xDim]: xCoords,
    }

    if (useNestedResults) {
      const addDimCoordinates = (dim: string) => {
        if (SPATIAL_DIM_NAMES.has(dim.toLowerCase())) {
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

  const buildResult = () =>
    ({
      [variable]: results,
      dimensions: resultDimensions,
      coordinates: buildResultCoordinates(),
    } as QueryResult)

  // Get level path for chunk fetching
  const levelPath = zarrStore.levels[levelIndex]
  if (!levelPath) {
    throw new Error(`No level path found for level index ${levelIndex}`)
  }

  // Parse actual zoom from level path to handle pyramids that don't start at 0
  const actualZoom = parseLevelZoom(levelPath, levelIndex)

  // Get tiles that intersect the polygon
  const tiles = getTilesForPolygon(geometry, actualZoom, crs, xyLimits)
  if (tiles.length === 0) return buildResult()

  checkAborted(signal)

  // For each tile, determine which chunks we need based on selector and fetch them
  const tileChunkData = new Map<string, Map<string, Float32Array>>()

  // Parallelize tile fetching - process all tiles concurrently instead of serially
  await Promise.all(
    tiles.map(async (tileTuple) => {
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
            const chunk = await zarrStore.getChunk(
              levelPath,
              chunkIndices,
              signal ? { signal } : undefined
            )
            // Make a proper copy to avoid buffer sharing issues
            const chunkData = new Float32Array(chunk.data as ArrayLike<number>)

            // Store chunk with indices as key
            const chunkKey = chunkIndices.join(',')
            chunkDataMap.set(chunkKey, chunkData)
          } catch (err) {
            if (err instanceof DOMException && err.name === 'AbortError') {
              throw err
            }
            console.warn(
              `Failed to fetch chunk ${chunkIndices} for tile ${tileKey}:`,
              err
            )
          }
        })
      )

      tileChunkData.set(tileKey, chunkDataMap)
    })
  )

  // Iterate over tiles, using scanline rasterization per tile
  for (const tileTuple of tiles) {
    const tileKey = tileToKey(tileTuple)
    const chunkDataMap = tileChunkData.get(tileKey)
    if (!chunkDataMap || chunkDataMap.size === 0) continue

    // Transform geometry into this tile's pixel space once
    const tileGeometry = transformGeometryToTilePixelSpace(
      geometry,
      tileTuple,
      tileSize,
      crs,
      xyLimits
    )
    if (!tileGeometry) continue

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

    // Point geometry: process the single pixel directly
    if (tileGeometry.type === 'Point') {
      const px = Math.min(Math.floor(tileGeometry.coordinates[0]), tileSize - 1)
      const py = Math.min(Math.floor(tileGeometry.coordinates[1]), tileSize - 1)
      if (px >= 0 && py >= 0) {
        processPixelTiled(
          px,
          py,
          tileTuple,
          chunksForTile,
          chunkDataMap,
          selector,
          dimensions,
          coordinates,
          shape,
          chunks,
          transforms,
          includeSpatialCoordinates,
          tileSize,
          crs,
          xyLimits,
          yCoords,
          xCoords,
          results,
          useNestedResults
        )
      }
      continue
    }

    // Build scanline table for this tile's pixel range
    const scanlines = buildScanlineTable(tileGeometry, 0, tileSize)

    for (let pixelY = 0; pixelY < tileSize; pixelY++) {
      checkAborted(signal)
      const crossings = scanlines.get(pixelY)
      if (!crossings || crossings.length < 2) continue

      for (let i = 0; i < crossings.length - 1; i += 2) {
        const xFrom = Math.max(0, Math.ceil(crossings[i] - 0.5))
        const xTo = Math.min(tileSize, Math.floor(crossings[i + 1] - 0.5) + 1)

        for (let pixelX = xFrom; pixelX < xTo; pixelX++) {
          processPixelTiled(
            pixelX,
            pixelY,
            tileTuple,
            chunksForTile,
            chunkDataMap,
            selector,
            dimensions,
            coordinates,
            shape,
            chunks,
            transforms,
            includeSpatialCoordinates,
            tileSize,
            crs,
            xyLimits,
            yCoords,
            xCoords,
            results,
            useNestedResults
          )
        }
      }
    }
  }

  return buildResult()
}

/**
 * Query a region in single-image mode.
 * Returns structure matching carbonplan/maps: { [variable]: values, dimensions, coordinates }
 */
export function queryRegionUntiled(
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
  sourceBounds?: Bounds | null,
  options?: QueryOptions
): QueryResult {
  const { signal, includeSpatialCoordinates = true } = options ?? {}

  // Calculate result dimension
  const singleValuedDims = Object.keys(selector).filter(
    (k) => !isMultiValSelector(selector[k])
  ).length
  const resultDim = dimensions.length - singleValuedDims

  // Determine if results should be nested
  const useNestedResults = resultDim > 2
  let results: QueryDataValues = useNestedResults ? {} : []

  const { yDim, xDim } = findSpatialDimNames(dimensions, !!proj4def)
  const yCoords: number[] = []
  const xCoords: number[] = []

  const resultDimensions = useNestedResults ? dimensions : [yDim, xDim]

  const buildResultCoordinates = (): Record<string, (number | string)[]> => {
    const coords: Record<string, (number | string)[]> = {
      [yDim]: yCoords,
      [xDim]: xCoords,
    }

    if (useNestedResults) {
      const addDimCoordinates = (dim: string) => {
        if (SPATIAL_DIM_NAMES.has(dim.toLowerCase())) {
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

  const buildResult = () =>
    ({
      [variable]: results,
      dimensions: resultDimensions,
      coordinates: buildResultCoordinates(),
    } as QueryResult)

  if (!data) return buildResult()

  checkAborted(signal)

  // Create transformer once for all pixels
  const cachedTransformer: CachedTransformer | undefined = proj4def
    ? createWGS84ToSourceTransformer(proj4def)
    : undefined

  // Transform the query polygon into pixel-space coordinates once.
  // This eliminates per-pixel proj4 calls during the intersection test.
  const pixelGeometry = transformGeometryToPixelSpace(
    geometry,
    bounds,
    width,
    height,
    _crs,
    latIsAscending,
    proj4def,
    sourceBounds,
    cachedTransformer
  )
  if (!pixelGeometry) return buildResult()

  // For proj4 data, emit source CRS coordinates via pure linear math (no inverse transform).
  // For standard CRS, emit lat/lon via pixelToLatLon.
  const emitCoords =
    proj4def && sourceBounds
      ? (x: number, y: number) => {
          const [srcX, srcY] = pixelToSourceCRS(
            x + 0.5,
            y + 0.5,
            sourceBounds,
            width,
            height,
            latIsAscending
          )
          yCoords.push(srcY)
          xCoords.push(srcX)
        }
      : (x: number, y: number) => {
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
          yCoords.push(lat)
          xCoords.push(lon)
        }

  // Helper to process a single pixel
  const processPixel = (x: number, y: number) => {
    const baseIndex = (y * width + x) * channels

    // Single-channel fast path
    if (channels === 1 && !useNestedResults) {
      const rawValue = data![baseIndex]
      const transformed = transformValue(rawValue, transforms)
      if (transformed === null) return

      if (includeSpatialCoordinates) emitCoords(x, y)
      ;(results as number[]).push(transformed)
      return
    }

    // Multi-channel path
    let hasValid = false
    for (let c = 0; c < channels; c++) {
      const rawValue = data![baseIndex + c]
      const transformed = transformValue(rawValue, transforms)
      if (transformed === null) continue

      if (!hasValid) {
        if (includeSpatialCoordinates) emitCoords(x, y)
        hasValid = true
      }

      if (useNestedResults && multiValueDimNames) {
        const labels = channelLabels?.[c]
        const keys =
          labels && labels.length === multiValueDimNames.length ? labels : [c]
        setObjectValues(results, keys, transformed)
      } else if (Array.isArray(results)) {
        results.push(transformed)
      }
    }
  }

  // Point geometry: process the single pixel directly
  if (pixelGeometry.type === 'Point') {
    const px = Math.min(Math.floor(pixelGeometry.coordinates[0]), width - 1)
    const py = Math.min(Math.floor(pixelGeometry.coordinates[1]), height - 1)
    if (px >= 0 && py >= 0) {
      processPixel(px, py)
    }
    return buildResult()
  }

  // Polygon/MultiPolygon: compute tight bbox and scanline table
  let pxMinX = Infinity
  let pxMaxX = -Infinity
  let pxMinY = Infinity
  let pxMaxY = -Infinity

  const scanRings = (rings: number[][][]) => {
    for (const ring of rings) {
      for (const [px, py] of ring) {
        if (px < pxMinX) pxMinX = px
        if (px > pxMaxX) pxMaxX = px
        if (py < pxMinY) pxMinY = py
        if (py > pxMaxY) pxMaxY = py
      }
    }
  }
  if (pixelGeometry.type === 'Polygon') {
    scanRings(pixelGeometry.coordinates)
  } else {
    for (const poly of pixelGeometry.coordinates) scanRings(poly)
  }

  const xStart = Math.max(0, Math.floor(pxMinX))
  const xEnd = Math.min(width, Math.ceil(pxMaxX))
  const yStart = Math.max(0, Math.floor(pxMinY))
  const yEnd = Math.min(height, Math.ceil(pxMaxY))

  if (xEnd <= xStart || yEnd <= yStart) return buildResult()

  // Build scanline intersection table: for each row Y, sorted X-crossings of polygon edges.
  // Pixels between consecutive pairs of crossings are inside the polygon.
  // Scanline table eliminates the O(V) per-pixel cost of point-in-polygon tests.
  const scanlines = buildScanlineTable(pixelGeometry, yStart, yEnd)

  // Iterate rows using the scanline table
  for (let y = yStart; y < yEnd; y++) {
    checkAborted(signal)
    const crossings = scanlines.get(y)
    if (!crossings || crossings.length < 2) continue

    // Walk crossing pairs: include pixels whose center (x+0.5) is inside the interval.
    for (let i = 0; i < crossings.length - 1; i += 2) {
      const xFrom = Math.max(xStart, Math.ceil(crossings[i] - 0.5))
      const xTo = Math.min(xEnd, Math.floor(crossings[i + 1] - 0.5) + 1)

      for (let x = xFrom; x < xTo; x++) {
        processPixel(x, y)
      }
    }
  }

  return buildResult()
}
