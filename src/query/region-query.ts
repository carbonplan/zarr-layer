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
  AreaQueryGeometry,
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

/** An array selector of any length, which nests the result by label. */
function isMultiValSelector(value: Selector[string]): boolean {
  const selected =
    value && typeof value === 'object' && 'selected' in value
      ? (value as { selected: unknown }).selected
      : value
  return Array.isArray(selected) && selected.length > 0
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

  return Number.isFinite(result) ? result : null
}

export interface ResultBuilderParams {
  variable: string
  selector: Selector
  data: Float32Array
  width: number
  height: number
  dimensions: string[]
  coordinates: Record<string, (string | number)[]>
  sourceBounds: Bounds
  channels: number
  channelLabels?: (string | number)[][]
  multiValueDimNames?: string[]
  latIsAscending?: boolean
  transforms?: QueryTransformOptions
  includeSpatialCoordinates: boolean
  dimIndices?: DimIndicesProps
}

export interface ResultBuilder {
  /**
   * Read the pixel at (x, y) of the fetched window into the result.
   * Returns false when every series was fill or non-finite and nothing was
   * emitted. A cell with a value in at least one series is emitted to all of
   * them, with NaN standing in where a series has none.
   */
  processPixel: (x: number, y: number) => boolean
  /** Assemble the result from everything processed so far. */
  buildResult: () => QueryResult
  /** Names the result uses for its spatial axes. */
  yDim: string
  xDim: string
}

/**
 * Accumulate per-pixel samples from one fetched window into the
 * carbonplan/maps result shape: { [variable]: values, dimensions, coordinates }.
 *
 * Values are scale/offset transformed and fill filtered. Results nest by
 * channel label when the selector holds more than one value along any
 * non-spatial dimension.
 */
export function createResultBuilder(
  params: ResultBuilderParams
): ResultBuilder {
  const {
    variable,
    selector,
    data,
    width,
    height,
    dimensions,
    coordinates,
    sourceBounds,
    channels,
    channelLabels,
    multiValueDimNames,
    latIsAscending,
    transforms,
    includeSpatialCoordinates,
    dimIndices,
  } = params

  // Nesting follows the selector's syntax: an array value nests by label, a
  // scalar or an unselected dimension (read at index 0) stays flat.
  const useNestedResults = Object.values(selector).some(isMultiValSelector)
  const results: QueryDataValues = useNestedResults ? {} : []

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

  const processPixel = (x: number, y: number): boolean => {
    const baseIndex = (y * width + x) * channels

    if (channels === 1 && !useNestedResults) {
      const transformed = transformValue(data[baseIndex], transforms)
      if (transformed === null) return false

      if (includeSpatialCoordinates) emitCoords(x, y)
      ;(results as number[]).push(transformed)
      return true
    }

    // Every series gets an entry for every emitted cell, NaN where that
    // series has no value, so each stays index-aligned with the coordinates.
    const values: number[] = new Array(channels)
    let hasValid = false
    for (let c = 0; c < channels; c++) {
      const transformed = transformValue(data[baseIndex + c], transforms)
      values[c] = transformed ?? NaN
      if (transformed !== null) hasValid = true
    }
    if (!hasValid) return false

    if (includeSpatialCoordinates) emitCoords(x, y)
    for (let c = 0; c < channels; c++) {
      if (useNestedResults && multiValueDimNames) {
        const labels = channelLabels?.[c]
        const keys =
          labels && labels.length === multiValueDimNames.length ? labels : [c]
        setObjectValues(results, keys, values[c])
      } else if (Array.isArray(results)) {
        results.push(values[c])
      }
    }
    return true
  }

  return { processPixel, buildResult, yDim, xDim }
}

/**
 * Result for a query that selected nothing, shaped like any other result for
 * the same selector: the same dimensions, selector coordinates, and flat or
 * nested values.
 */
export function buildEmptyResult(
  variable: string,
  selector: Selector,
  dimensions: string[],
  coordinates: Record<string, (string | number)[]>,
  dimIndices?: DimIndicesProps
): QueryResult {
  return createResultBuilder({
    variable,
    selector,
    data: new Float32Array(0),
    width: 0,
    height: 0,
    dimensions,
    coordinates,
    sourceBounds: [0, 0, 0, 0],
    channels: 1,
    includeSpatialCoordinates: false,
    dimIndices,
  }).buildResult()
}

/**
 * Query a raster region.
 * Returns structure matching carbonplan/maps: { [variable]: values, dimensions, coordinates }
 */
export function queryRegion(
  variable: string,
  geometry: AreaQueryGeometry,
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

  const { processPixel, buildResult } = createResultBuilder({
    variable,
    selector,
    data: data ?? new Float32Array(0),
    width,
    height,
    dimensions,
    coordinates,
    sourceBounds,
    channels,
    channelLabels,
    multiValueDimNames,
    latIsAscending,
    transforms,
    includeSpatialCoordinates,
    dimIndices,
  })

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
