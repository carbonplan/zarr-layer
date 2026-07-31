import * as zarr from 'zarrita'
import type {
  Bounds,
  NormalizedSelector,
  Selector,
  UntiledLevel,
} from '../types'
import type { MercatorBounds, XYLimits } from '../map-utils'
import type { ProjectionContext } from '../projection-utils'
import { pixelToSourceCRS } from '../projection-utils'
import type { QueryLevelSnapshot } from '../region-state'
import type { ZarrStore } from '../zarr-store'
import {
  buildChannelCombinations,
  buildSliceArgsForSelector,
  type DimensionValuesCache,
} from '../selector-resolution'
import { normalizeSelector } from '../zarr-utils'
import { queryRegionUntiled, findSpatialDimNames } from './region-query'
import {
  computePixelBoundsFromGeometry,
  preprocessQueryGeometry,
  wrappedBboxToPixelSpans,
  rasterExtentCrossesAntimeridian,
  type PixelRect,
} from './query-utils'
import type {
  NestedValues,
  QueryDataValues,
  QueryGeometry,
  QueryOptions,
  QueryResult,
} from './types'

export type QueryContext = {
  zarrStore: ZarrStore
  variable: string
  selector: NormalizedSelector
  xyLimits: XYLimits | null
  mercatorBounds: MercatorBounds | null
  latIsAscending: boolean
  levels: UntiledLevel[]
  level: QueryLevelSnapshot | null
  projection: ProjectionContext
  antimeridianWarnings: Set<string>
  dimensionValues: DimensionValuesCache
  isMultiscale: boolean
  coordLevelIndex: number
}

/**
 * Unified method to fetch query data for either point or region queries.
 * Handles multi-value dimensions and channel combinations.
 */
export async function fetchQueryData(
  context: QueryContext,
  level: QueryLevelSnapshot,
  selector: NormalizedSelector,
  spatialQuery: {
    minX: number
    maxX: number
    minY: number
    maxY: number
  },
  signal?: AbortSignal
): Promise<{
  data: Float32Array
  width: number
  height: number
  channels: number
  channelLabels: (string | number)[][]
  multiValueDimNames: string[]
} | null> {
  try {
    const { sliceArgs: baseSliceArgs, multiValueDims } =
      await buildSliceArgsForSelector(
        {
          zarrStore: context.zarrStore,
          dimIndices: context.zarrStore.describe().dimIndices,
          levels: context.levels,
          isMultiscale: context.isMultiscale,
          dimensionValues: context.dimensionValues,
          coordLevelIndex: context.coordLevelIndex,
        },
        selector,
        {
          includeSpatialSlices: false,
          trackMultiValue: true,
          spatialBounds: spatialQuery,
          array: level.zarrArray,
        }
      )

    const {
      combinations: channelCombinations,
      labelCombinations: channelLabelCombinations,
    } = buildChannelCombinations(multiValueDims)
    const numChannels = channelCombinations.length || 1
    const multiValueDimNames = multiValueDims.map((d) => d.dimName)
    const getOpts = signal ? { signal } : undefined
    const fetchWidth = spatialQuery.maxX - spatialQuery.minX
    const fetchHeight = spatialQuery.maxY - spatialQuery.minY

    if (numChannels === 1) {
      const result = (await zarr.get(
        level.zarrArray,
        baseSliceArgs,
        getOpts
      )) as { data: ArrayLike<number> }
      return {
        data: new Float32Array(result.data),
        width: fetchWidth,
        height: fetchHeight,
        channels: 1,
        channelLabels: channelLabelCombinations,
        multiValueDimNames,
      }
    }

    const packedData = new Float32Array(fetchWidth * fetchHeight * numChannels)
    for (let c = 0; c < numChannels; c++) {
      const sliceArgs = [...baseSliceArgs]
      const combo = channelCombinations[c]
      for (let i = 0; i < multiValueDims.length; i++) {
        sliceArgs[multiValueDims[i].dimIndex] = combo[i]
      }

      const bandData = (await zarr.get(
        level.zarrArray,
        sliceArgs,
        getOpts
      )) as { data: ArrayLike<number> }
      for (let pixIdx = 0; pixIdx < fetchWidth * fetchHeight; pixIdx++) {
        packedData[pixIdx * numChannels + c] = bandData.data[pixIdx]
      }
    }

    return {
      data: packedData,
      width: fetchWidth,
      height: fetchHeight,
      channels: numChannels,
      channelLabels: channelLabelCombinations,
      multiValueDimNames,
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err
    console.error('Error fetching query data:', err)
    return null
  }
}

/** Query data for point or region geometries. */
export async function queryData(
  context: QueryContext,
  geometry: QueryGeometry,
  selector?: Selector,
  options?: QueryOptions
): Promise<QueryResult> {
  const desc = context.zarrStore.describe()
  // Queries index into the active level, so they must map through whatever
  // extent that level is drawn against.
  const queryLimits = context.level?.xyLimits ?? context.xyLimits
  const sourceBounds: Bounds | null = queryLimits
    ? [queryLimits.xMin, queryLimits.yMin, queryLimits.xMax, queryLimits.yMax]
    : null
  const { yDim: emptyYDim, xDim: emptyXDim } = findSpatialDimNames(
    desc.dimensions,
    desc.dimIndices
  )
  const emptyResult = (): QueryResult => ({
    [context.variable]: [],
    dimensions: [],
    coordinates: { [emptyYDim]: [], [emptyXDim]: [] },
  })

  const level = context.level
  if (!context.mercatorBounds || !level || !sourceBounds) {
    return emptyResult()
  }
  const projectionDef = context.projection.def
  if (!projectionDef) return emptyResult()

  const normalizedSelector = selector
    ? normalizeSelector(selector)
    : context.selector
  const currentLevel = context.levels[level.index]
  const transforms = {
    scaleFactor: currentLevel?.scaleFactor ?? desc.scaleFactor,
    addOffset: currentLevel?.addOffset ?? desc.addOffset,
    fillValue: currentLevel?.fillValue ?? desc.fill_value,
  }

  // Closure for running a single pixel-bounds strip query.
  // Captures request-scoped locals (not instance state) to avoid races
  // when queryData is called concurrently on the same instance.
  const runStrip = async (
    geom: QueryGeometry,
    pixelBounds: PixelRect,
    opts?: QueryOptions
  ): Promise<QueryResult | null> => {
    const fetched = await fetchQueryData(
      context,
      level,
      normalizedSelector,
      pixelBounds,
      opts?.signal
    )
    if (!fetched) return null

    const { minX, minY, maxX, maxY } = pixelBounds
    const [xMin0, yMin0] = pixelToSourceCRS(
      minX,
      minY,
      sourceBounds,
      level.width,
      level.height,
      context.latIsAscending
    )
    const [xMax0, yMax0] = pixelToSourceCRS(
      maxX,
      maxY,
      sourceBounds,
      level.width,
      level.height,
      context.latIsAscending
    )
    const subsetSourceBounds: Bounds = [
      Math.min(xMin0, xMax0),
      Math.min(yMin0, yMax0),
      Math.max(xMin0, xMax0),
      Math.max(yMin0, yMax0),
    ]

    return queryRegionUntiled(
      context.variable,
      geom,
      normalizedSelector,
      fetched.data,
      fetched.width,
      fetched.height,
      desc.dimensions,
      desc.coordinates,
      subsetSourceBounds,
      projectionDef,
      fetched.channels,
      fetched.channelLabels,
      fetched.multiValueDimNames,
      context.latIsAscending,
      transforms,
      opts,
      desc.dimIndices,
      context.projection.toWGS84 ?? undefined
    )
  }

  const singleFetch = async (geom: QueryGeometry): Promise<QueryResult> => {
    const pixelBounds = computePixelBoundsFromGeometry(
      geom,
      sourceBounds,
      level.width,
      level.height,
      projectionDef,
      context.latIsAscending,
      context.projection.toWGS84 ?? undefined
    )
    if (!pixelBounds) return emptyResult()
    const result = await runStrip(geom, pixelBounds, options)
    return result ?? emptyResult()
  }

  const { geometry: processedGeometry, bbox: wrappedBbox } =
    preprocessQueryGeometry(geometry)
  const supportsWrappedLongitude =
    context.projection.kind === 'epsg4326' ||
    context.projection.kind === 'epsg3857'
  const queryGeometry = supportsWrappedLongitude ? processedGeometry : geometry

  if (!wrappedBbox.crossesAntimeridian) return singleFetch(queryGeometry)

  if (!supportsWrappedLongitude) {
    if (!context.antimeridianWarnings.has('proj4-crossing')) {
      context.antimeridianWarnings.add('proj4-crossing')
      console.warn(
        'Antimeridian-crossing polygon queries are not supported for proj4 projections; results may be incorrect'
      )
    }
    return singleFetch(queryGeometry)
  }

  // Crossing: raster extent guard (EPSG:4326 only — 3857 xyLimits are in meters)
  if (
    context.projection.kind === 'epsg4326' &&
    rasterExtentCrossesAntimeridian('EPSG:4326', context.xyLimits)
  ) {
    if (!context.antimeridianWarnings.has('raster-extent-crossing')) {
      context.antimeridianWarnings.add('raster-extent-crossing')
      console.warn(
        'Antimeridian-crossing polygon queries are not supported for rasters whose own extent crosses the antimeridian; results may be incorrect'
      )
    }
    return singleFetch(geometry)
  }

  // Crossing: two-strip fetch
  const spans = wrappedBboxToPixelSpans(
    wrappedBbox,
    sourceBounds,
    level.width,
    level.height,
    projectionDef,
    context.latIsAscending,
    context.projection.toWGS84 ?? undefined
  )
  const westResult = spans.west
    ? await runStrip(processedGeometry, spans.west, options)
    : null
  const eastResult = spans.east
    ? await runStrip(processedGeometry, spans.east, options)
    : null

  // If either requested strip failed, return empty rather than partial data
  if ((spans.west && !westResult) || (spans.east && !eastResult)) {
    return emptyResult()
  }
  if (!westResult && !eastResult) return emptyResult()
  if (!westResult || !eastResult) return (westResult ?? eastResult)!

  const { yDim, xDim } = findSpatialDimNames(desc.dimensions, desc.dimIndices)
  return mergeQueryResults(westResult, eastResult, context.variable, yDim, xDim)
}

/**
 * Merge two QueryResult objects from west and east strips.
 *
 * Ordering: west-strip pixels first, then east-strip pixels. This does NOT
 * preserve row-major scan order. The QueryResult contract provides parallel
 * coordinate arrays so consumers index by position, not implicit grid layout.
 *
 * Spatial coordinate arrays (yDim, xDim) are concatenated.
 * Non-spatial coordinate arrays are taken from the first result unchanged.
 */
export function mergeQueryResults(
  a: QueryResult,
  b: QueryResult,
  variable: string,
  yDim: string,
  xDim: string
): QueryResult {
  const spatialKeys = new Set([yDim, xDim])

  // Merge coordinates: concatenate spatial, take first for non-spatial
  const coordinates: Record<string, (number | string)[]> = {}
  for (const key of Object.keys(a.coordinates)) {
    coordinates[key] = spatialKeys.has(key)
      ? [...a.coordinates[key], ...b.coordinates[key]]
      : a.coordinates[key]
  }

  const aVals = a[variable] as QueryDataValues
  const bVals = b[variable] as QueryDataValues
  let merged: QueryDataValues
  if (Array.isArray(aVals) && Array.isArray(bVals)) {
    merged = [...aVals, ...bVals]
  } else if (!Array.isArray(aVals) && !Array.isArray(bVals)) {
    merged = mergeNestedValues(aVals as NestedValues, bVals as NestedValues)
  } else {
    merged = aVals // Mismatched types: take first
  }

  return { [variable]: merged, dimensions: a.dimensions, coordinates }
}

/** Recursively merge two NestedValues objects by concatenating leaf arrays. */
export function mergeNestedValues(
  a: NestedValues,
  b: NestedValues
): NestedValues {
  const result: NestedValues = {}
  for (const key of Object.keys(a)) {
    const aVal = a[key]
    const bVal = b[key]
    if (Array.isArray(aVal) && Array.isArray(bVal)) {
      result[key] = [...aVal, ...bVal]
    } else if (
      aVal &&
      bVal &&
      !Array.isArray(aVal) &&
      !Array.isArray(bVal) &&
      typeof aVal === 'object' &&
      typeof bVal === 'object'
    ) {
      result[key] = mergeNestedValues(
        aVal as NestedValues,
        bVal as NestedValues
      )
    } else {
      result[key] = aVal
    }
  }
  // Include keys only in b
  for (const key of Object.keys(b)) {
    if (!(key in result)) result[key] = b[key]
  }
  return result
}
