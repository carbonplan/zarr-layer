import * as zarr from 'zarrita'
import type {
  Bounds,
  NormalizedSelector,
  Selector,
  ResolutionLevel,
} from '../types'
import type { MercatorBounds, XYLimits } from '../map-utils'
import type { ProjectionContext } from '../projection-utils'
import { pixelToSourceCRS } from '../projection-utils'
import type { QueryLevelSnapshot } from '../region-state'
import type { ZarrStore } from '../zarr-store'
import {
  buildChannelCombinations,
  assertSelectorKeysAreDimensions,
  buildSliceArgsForSelector,
  findQueryMultiValueDims,
  type DimensionValuesCache,
} from '../selector-resolution'
import { normalizeSelector } from '../zarr-utils'
import { wrapError } from '../errors'
import {
  queryRegion,
  findSpatialDimNames,
  createResultBuilder,
  buildEmptyResult,
} from './region-query'
import {
  computePixelBoundsFromGeometry,
  preprocessQueryGeometry,
  wrappedBboxToPixelSpans,
  rasterExtentCrossesAntimeridian,
  shiftGeometryIntoExtent,
  type PixelRect,
} from './query-utils'
import {
  groupCellsIntoRuns,
  lineToPixelPaths,
  splitLineAtAntimeridian,
  traceLineCells,
  validateLineCoordinates,
  type TracedCell,
} from './line-query'
import { QUERY_LINE_RUN_MAX_PX } from '../constants'
import type {
  AreaQueryGeometry,
  NestedValues,
  QueryDataValues,
  QueryGeometry,
  QueryOptions,
  QueryResult,
  QueryTransformOptions,
} from './types'

const DEFAULT_LINE_DISTANCE_KEY = 'distance'

/**
 * Key that a LineString result reports distance under. Every other key in
 * `coordinates` is a store dimension name, so a clash would overwrite that
 * dimension's values and is rejected.
 */
function resolveLineDistanceKey(
  dimensions: string[],
  dimIndices: Parameters<typeof findSpatialDimNames>[1],
  options?: QueryOptions
): string {
  const key = options?.distanceKey ?? DEFAULT_LINE_DISTANCE_KEY
  const { yDim, xDim } = findSpatialDimNames(dimensions, dimIndices)
  if (key === yDim || key === xDim || dimensions.includes(key)) {
    throw new Error(
      `[ZarrLayer] LineString queries report distance under the result coordinate \`${key}\`, which this store already uses as a dimension name. Pass a different \`distanceKey\` in the query options.`
    )
  }
  return key
}

export type QueryContext = {
  zarrStore: ZarrStore
  variable: string
  selector: NormalizedSelector
  xyLimits: XYLimits | null
  mercatorBounds: MercatorBounds | null
  latIsAscending: boolean
  levels: ResolutionLevel[]
  level: QueryLevelSnapshot | null
  projection: ProjectionContext
  antimeridianWarnings: Set<string>
  dimensionValues: DimensionValuesCache
  isMultiscale: boolean
  coordLevelIndex: number
  /** Dimensions already warned about after coordinate-read failures. */
  warnedDimensions: Set<string>
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
}> {
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
          warnedDimensions: context.warnedDimensions,
        },
        selector,
        {
          includeSpatialSlices: false,
          trackMultiValue: true,
          queryLabelling: true,
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
    // Propagated, not swallowed: a read that failed is not a region that holds
    // no data, and returning empty here makes the two indistinguishable.
    throw wrapError('[ZarrLayer] failed to read query data', err)
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
  // A query selector overrides the layer's selector key by key, so a
  // dimension it leaves out stays on the slice the layer is showing.
  const normalizedSelector = selector
    ? { ...context.selector, ...normalizeSelector(selector) }
    : context.selector
  assertSelectorKeysAreDimensions(normalizedSelector, desc.dimIndices)
  if (geometry.type === 'LineString') {
    validateLineCoordinates(geometry.coordinates)
  }
  const lineDistanceKey =
    geometry.type === 'LineString'
      ? resolveLineDistanceKey(desc.dimensions, desc.dimIndices, options)
      : null
  const emptyResult = (): QueryResult => {
    const result = buildEmptyResult(
      context.variable,
      normalizedSelector,
      desc.dimensions,
      desc.coordinates,
      findQueryMultiValueDims(normalizedSelector, desc.dimIndices),
      desc.dimIndices
    )
    if (lineDistanceKey !== null) result.coordinates[lineDistanceKey] = []
    return result
  }

  const level = context.level
  if (!context.mercatorBounds || !level || !sourceBounds) {
    return emptyResult()
  }
  const projectionDef = context.projection.def
  if (!projectionDef) return emptyResult()

  const currentLevel = context.levels[level.index]
  const transforms = {
    scaleFactor: currentLevel?.scaleFactor ?? desc.scaleFactor,
    addOffset: currentLevel?.addOffset ?? desc.addOffset,
    fillValue: currentLevel?.fillValue ?? desc.fill_value,
  }

  if (geometry.type === 'LineString') {
    return queryLineString(
      context,
      geometry.coordinates,
      level,
      sourceBounds,
      projectionDef,
      normalizedSelector,
      transforms,
      emptyResult,
      lineDistanceKey!,
      options
    )
  }

  // Closure for running a single pixel-bounds strip query.
  // Captures request-scoped locals (not instance state) to avoid races
  // when queryData is called concurrently on the same instance.
  const runStrip = async (
    geom: AreaQueryGeometry,
    pixelBounds: PixelRect,
    opts?: QueryOptions
  ): Promise<QueryResult> => {
    const fetched = await fetchQueryData(
      context,
      level,
      normalizedSelector,
      pixelBounds,
      opts?.signal
    )
    const subsetSourceBounds = pixelRectToSourceBounds(
      pixelBounds,
      sourceBounds,
      level.width,
      level.height,
      context.latIsAscending
    )

    return queryRegion(
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

  const singleFetch = async (geom: AreaQueryGeometry): Promise<QueryResult> => {
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
    return runStrip(geom, pixelBounds, options)
  }

  const { geometry: processedGeometry, bbox: wrappedBbox } =
    preprocessQueryGeometry(geometry)
  const supportsWrappedLongitude =
    context.projection.kind === 'epsg4326' ||
    context.projection.kind === 'epsg3857'
  const queryGeometry = supportsWrappedLongitude ? processedGeometry : geometry

  const extentPastAntimeridian =
    context.projection.kind === 'epsg4326' &&
    rasterExtentCrossesAntimeridian('EPSG:4326', queryLimits)

  if (!wrappedBbox.crossesAntimeridian) {
    if (!extentPastAntimeridian || !queryLimits)
      return singleFetch(queryGeometry)
    const { xMin, xMax } = queryLimits
    const shifted = shiftGeometryIntoExtent(
      queryGeometry,
      wrappedBbox,
      xMin,
      xMax
    )
    const fits = (lon: number) => lon >= xMin && lon <= xMax
    if (
      shifted === queryGeometry &&
      !(fits(wrappedBbox.west) && fits(wrappedBbox.east)) &&
      !context.antimeridianWarnings.has('raster-extent-partial')
    ) {
      context.antimeridianWarnings.add('raster-extent-partial')
      console.warn(
        'Queries only partly inside the part of a raster extent beyond ±180 read the side inside it; results may be incomplete'
      )
    }
    return singleFetch(shifted)
  }

  if (!supportsWrappedLongitude) {
    if (!context.antimeridianWarnings.has('proj4-crossing')) {
      context.antimeridianWarnings.add('proj4-crossing')
      console.warn(
        'Antimeridian-crossing polygon queries are not supported for proj4 projections; results may be incorrect'
      )
    }
    return singleFetch(queryGeometry)
  }

  // Crossing: raster extent guard (EPSG:4326 only — 3857 xyLimits are in
  // meters). Checked against the same extent the pixel-span mapping uses.
  if (extentPastAntimeridian) {
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

  // A strip is null only when the wrapped bbox produced no span on that side;
  // a strip that failed to read threw rather than coming back empty.
  if (!westResult && !eastResult) return emptyResult()
  if (!westResult || !eastResult) return (westResult ?? eastResult)!

  const { yDim, xDim } = findSpatialDimNames(desc.dimensions, desc.dimIndices)
  return mergeQueryResults(westResult, eastResult, context.variable, yDim, xDim)
}

/** Source-CRS extent of a pixel rectangle within a level. */
function pixelRectToSourceBounds(
  rect: PixelRect,
  sourceBounds: Bounds,
  width: number,
  height: number,
  latIsAscending: boolean
): Bounds {
  const [x0, y0] = pixelToSourceCRS(
    rect.minX,
    rect.minY,
    sourceBounds,
    width,
    height,
    latIsAscending
  )
  const [x1, y1] = pixelToSourceCRS(
    rect.maxX,
    rect.maxY,
    sourceBounds,
    width,
    height,
    latIsAscending
  )
  return [
    Math.min(x0, x1),
    Math.min(y0, y1),
    Math.max(x0, x1),
    Math.max(y0, y1),
  ]
}

/**
 * Sample every cell a line passes through, in path order.
 *
 * The line is traced once through the level's full pixel grid, and the
 * traced cells are read in runs of bounded extent so a long line costs a
 * chain of small windows rather than the rectangle it spans. Each sample
 * carries the distance along the line from its start in `distance`.
 */
async function queryLineString(
  context: QueryContext,
  coords: number[][],
  level: QueryLevelSnapshot,
  sourceBounds: Bounds,
  projectionDef: string,
  selector: NormalizedSelector,
  transforms: QueryTransformOptions,
  emptyResult: () => QueryResult,
  distanceKey: string,
  options?: QueryOptions
): Promise<QueryResult> {
  const desc = context.zarrStore.describe()
  const { yDim, xDim } = findSpatialDimNames(desc.dimensions, desc.dimIndices)
  const includeSpatialCoordinates = options?.includeSpatialCoordinates ?? true
  const signal = options?.signal

  const supportsWrappedLongitude =
    context.projection.kind === 'epsg4326' ||
    context.projection.kind === 'epsg3857'
  const queryLimits = level.xyLimits ?? context.xyLimits
  const extentPastAntimeridian =
    context.projection.kind === 'epsg4326' &&
    rasterExtentCrossesAntimeridian('EPSG:4326', queryLimits)

  let pieces: number[][][]
  if (!supportsWrappedLongitude) {
    pieces = [coords]
    if (
      coords.some(([lon]) => lon > 180 || lon < -180) &&
      !context.antimeridianWarnings.has('proj4-crossing')
    ) {
      context.antimeridianWarnings.add('proj4-crossing')
      console.warn(
        'Antimeridian-crossing queries are not supported for proj4 projections; results may be incorrect'
      )
    }
  } else if (extentPastAntimeridian && queryLimits) {
    if (queryLimits.xMin < queryLimits.xMax) {
      // The raster's seam sits at its own west edge, so pieces are cut there
      // and land on its grid wherever it reaches past ±180.
      pieces = splitLineAtAntimeridian(coords, queryLimits.xMin)
    } else {
      pieces = [coords]
      if (!context.antimeridianWarnings.has('raster-extent-crossing')) {
        context.antimeridianWarnings.add('raster-extent-crossing')
        console.warn(
          'Antimeridian-crossing queries are not supported for rasters whose own extent crosses the antimeridian; results may be incorrect'
        )
      }
    }
  } else {
    pieces = splitLineAtAntimeridian(coords)
  }

  const cells: TracedCell[] = []
  let distance = 0
  for (const piece of pieces) {
    const { sections, trailingGap } = lineToPixelPaths(
      piece,
      sourceBounds,
      level.width,
      level.height,
      projectionDef,
      context.latIsAscending,
      context.projection.toWGS84 ?? undefined
    )
    for (const section of sections) {
      const traced = traceLineCells(
        section.path,
        level.width,
        level.height,
        distance + section.gapBefore
      )
      for (const cell of traced.cells) cells.push(cell)
      distance = traced.endDistance
    }
    distance += trailingGap
  }

  const runs = groupCellsIntoRuns(cells, QUERY_LINE_RUN_MAX_PX)
  if (runs.length === 0) return emptyResult()

  const runResults: QueryResult[] = []
  for (const run of runs) {
    if (signal?.aborted) {
      throw new DOMException('The operation was aborted.', 'AbortError')
    }
    const fetched = await fetchQueryData(
      context,
      level,
      selector,
      run.rect,
      signal
    )
    const builder = createResultBuilder({
      variable: context.variable,
      selector,
      data: fetched.data,
      width: fetched.width,
      height: fetched.height,
      dimensions: desc.dimensions,
      coordinates: desc.coordinates,
      sourceBounds: pixelRectToSourceBounds(
        run.rect,
        sourceBounds,
        level.width,
        level.height,
        context.latIsAscending
      ),
      channels: fetched.channels,
      channelLabels: fetched.channelLabels,
      multiValueDimNames: fetched.multiValueDimNames,
      latIsAscending: context.latIsAscending,
      transforms,
      includeSpatialCoordinates,
      dimIndices: desc.dimIndices,
    })

    const distances: number[] = []
    for (const cell of run.cells) {
      const emitted = builder.processPixel(
        cell.x - run.rect.minX,
        cell.y - run.rect.minY
      )
      if (emitted && includeSpatialCoordinates) distances.push(cell.distance)
    }
    const result = builder.buildResult()
    result.coordinates[distanceKey] = distances

    runResults.push(result)
  }

  return concatQueryResults(runResults, context.variable, [
    yDim,
    xDim,
    distanceKey,
  ])
}

/**
 * Join per-window results, in order, into one. Each per-pixel array is built
 * once at its final length, so the cost is linear in the samples however
 * many windows a long line was read in.
 *
 * `perPixelKeys` names the coordinate arrays that hold one entry per sample.
 * Other coordinates describe the selection and are taken from the first
 * result.
 */
export function concatQueryResults(
  results: QueryResult[],
  variable: string,
  perPixelKeys: string[]
): QueryResult {
  const [first] = results
  if (results.length === 1) return first

  const coordinates: Record<string, (number | string)[]> = {}
  for (const key of Object.keys(first.coordinates)) {
    coordinates[key] = perPixelKeys.includes(key)
      ? concatArrays(results.map((r) => r.coordinates[key] ?? []))
      : first.coordinates[key]
  }

  return {
    [variable]: concatValues(
      results.map((r) => r[variable] as QueryDataValues)
    ),
    dimensions: first.dimensions,
    coordinates,
  }
}

function concatArrays<T>(parts: T[][]): T[] {
  let total = 0
  for (const part of parts) total += part.length
  const out = new Array<T>(total)
  let offset = 0
  for (const part of parts) {
    for (let i = 0; i < part.length; i++) out[offset + i] = part[i]
    offset += part.length
  }
  return out
}

/**
 * Concatenate flat value arrays, or nested series leaf by leaf. A window that
 * emitted no samples has no series keys and adds nothing to any leaf.
 */
function concatValues(parts: QueryDataValues[]): QueryDataValues {
  if (parts.every((part) => Array.isArray(part))) {
    return concatArrays(parts as number[][])
  }
  const nested = parts.filter(
    (part): part is NestedValues => !Array.isArray(part)
  )
  const result: NestedValues = {}
  const has = (target: object, key: string) =>
    Object.prototype.hasOwnProperty.call(target, key)
  for (const part of nested) {
    for (const key of Object.keys(part)) {
      if (has(result, key)) continue
      Object.defineProperty(result, key, {
        value: concatValues(
          nested.filter((p) => has(p, key)).map((p) => p[key])
        ),
        enumerable: true,
        writable: true,
        configurable: true,
      })
    }
  }
  return result
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
    if (!Object.prototype.hasOwnProperty.call(result, key)) result[key] = b[key]
  }
  return result
}
