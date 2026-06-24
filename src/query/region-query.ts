/**
 * @module region-query
 *
 * Region query implementation for zarr-layer.
 * Queries all data points within a GeoJSON polygon.
 * Matches carbonplan/maps structure and behavior.
 */

import type { Bounds, DimIndicesProps, Selector } from '../types'
import { pixelToSourceCRS } from '../projection-utils'
import type {
  QueryGeometry,
  QueryOptions,
  QueryResult,
  QueryDataValues,
  QueryTransformOptions,
} from './types'
import {
  transformGeometryToPixelSpace,
  buildScanlineTable,
  CachedTransformer,
} from './query-utils'
import { createWGS84ToSourceTransformer } from '../projection-utils'
import { setObjectValues } from './selector-utils'
import { SPATIAL_DIMENSION_ALIASES } from '../constants'

/**
 * Resolve the store's spatial axis names for query result keys.
 *
 * Uses dimIndices (which incorporates spatialDimensions overrides) when
 * available, falling back to alias matching on the raw dimension names.
 */
export function findSpatialDimNames(
  dimensions: string[],
  dimIndices?: DimIndicesProps
): { yDim: string; xDim: string } {
  const yDim = dimIndices?.lat?.name ?? findByAlias(dimensions, 'lat')
  const xDim = dimIndices?.lon?.name ?? findByAlias(dimensions, 'lon')
  return { yDim, xDim }
}

function findByAlias(dimensions: string[], axis: 'lat' | 'lon'): string {
  const aliases = SPATIAL_DIMENSION_ALIASES[axis]
  return dimensions.find((d) => aliases.includes(d.toLowerCase())) ?? axis
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
  dimensions: string[],
  coordinates: Record<string, (string | number)[]>,
  sourceBounds: Bounds,
  proj4def: string,
  channels: number = 1,
  channelLabels?: (string | number)[][],
  multiValueDimNames?: string[],
  latIsAscending?: boolean,
  transforms?: QueryTransformOptions,
  options?: QueryOptions,
  dimIndices?: DimIndicesProps,
  cachedTransformer?: CachedTransformer
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

  const { yDim, xDim } = findSpatialDimNames(dimensions, dimIndices)
  const yCoords: number[] = []
  const xCoords: number[] = []

  const resultDimensions = useNestedResults ? [...dimensions] : [yDim, xDim]

  const buildResultCoordinates = (): Record<string, (number | string)[]> => {
    const coords: Record<string, (number | string)[]> = {
      [yDim]: yCoords,
      [xDim]: xCoords,
    }

    if (useNestedResults) {
      for (const dim of dimensions) {
        if (dim === yDim || dim === xDim) continue

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
  const transformer: CachedTransformer =
    cachedTransformer ?? createWGS84ToSourceTransformer(proj4def)

  // Transform the query polygon into pixel-space coordinates once.
  // This eliminates per-pixel proj4 calls during the intersection test.
  const pixelGeometry = transformGeometryToPixelSpace(
    geometry,
    sourceBounds,
    width,
    height,
    proj4def,
    latIsAscending,
    transformer
  )
  if (!pixelGeometry) return buildResult()

  // Emit pixel-center coordinates in the source CRS.
  const emitCoords = (x: number, y: number) => {
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
