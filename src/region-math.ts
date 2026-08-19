import type * as zarr from 'zarrita'
import { WEB_MERCATOR_EXTENT } from './constants'
import {
  boundsToMercatorNorm,
  lonRangeOverlaps,
  type MercatorBounds,
  type XYLimits,
} from './map-utils'
import type { ProjectionContext } from './projection-utils'
import { sampleEdgesToMercatorBounds } from './projection-utils'
import type { LevelMeta } from './region-state'
import type { DimIndicesProps, MapLike, ResolutionLevel } from './types'

export type RegionCoordinate = { regionX: number; regionY: number }
type SourceBounds = { xMin: number; xMax: number; yMin: number; yMax: number }

/** Detect optimal region size from array metadata. */
export function getRegionSize(
  array: zarr.Array<zarr.DataType>,
  dimIndices: DimIndicesProps
): [number, number] | null {
  const latIdx = dimIndices.lat?.index
  const lonIdx = dimIndices.lon?.index
  if (latIdx === undefined || lonIdx === undefined) return null
  const codecs = (array as any).codecs || []
  for (const codec of codecs) {
    if (codec.name === 'sharding_indexed' && codec.configuration?.chunk_shape) {
      const shardShape = codec.configuration.chunk_shape as number[]
      return [shardShape[latIdx], shardShape[lonIdx]]
    }
  }
  const chunks = array.chunks as number[] | undefined
  if (chunks && chunks.length > Math.max(latIdx, lonIdx)) {
    const chunkH = chunks[latIdx]
    const chunkW = chunks[lonIdx]
    // Only use region-based loading if chunks are smaller than the array
    const shape = array.shape as number[]
    if (chunkH < shape[latIdx] || chunkW < shape[lonIdx]) {
      return [chunkH, chunkW]
    }
  }
  return null // No chunking or single chunk
}

export function getVisibleRegions({
  map,
  xyLimits,
  levelMeta,
  projection,
  latIsAscending,
}: {
  map: MapLike
  xyLimits: XYLimits | null
  levelMeta: LevelMeta | null
  projection: ProjectionContext
  latIsAscending: boolean
}): RegionCoordinate[] {
  const bounds = map.getBounds?.()?.toArray?.()
  if (!bounds || !xyLimits || !levelMeta) return []
  if (!projection.def || !projection.toWGS84) {
    // No proj4 transformer was set up — either because the CRS isn't
    // one proj4js handles natively and no `proj4` prop was supplied
    // (constructor warned), or because proj4 init threw on a malformed
    // string. Either way, computing region indices here would be wrong;
    // skip rather than render a misleading partial result.
    return []
  }
  const { width, height, regionSize } = levelMeta
  const [[west, south], [east, north]] = bounds
  const [regionH, regionW] = regionSize
  // Candidate index math and the verification below must agree on the extent,
  // or a level placed against its own bounds can generate no candidates at all.
  const limits = levelMeta.xyLimits ?? xyLimits
  // For projected data, use a two-pass approach:
  // 1. Forward-transform viewport edges to source CRS to find candidate
  //    regions via index math (O(1) proj4 cost, may include false
  //    positives for non-bijective projections like UTM outside their zone)
  // 2. Inverse-transform candidate region bounds to WGS84 for precise overlap
  const transformer = projection.toWGS84
  const candidates = getCandidateRegions({
    west,
    south,
    east,
    north,
    transformer,
    numRegionsX: Math.ceil(width / regionW),
    numRegionsY: Math.ceil(height / regionH),
    regionW,
    regionH,
    width,
    height,
    xyLimits: limits,
    latIsAscending,
  })

  // Verify candidates via inverse transform to WGS84 for precise overlap.
  // This handles non-bijective projections where forward transforms can
  // produce false positives.
  const regions: RegionCoordinate[] = []
  for (const { regionX, regionY } of candidates) {
    const regBounds = getRegionBounds({
      regionX,
      regionY,
      levelMeta,
      xyLimits,
      latIsAscending,
    })
    const xMid = (regBounds.xMin + regBounds.xMax) / 2
    const yMid = (regBounds.yMin + regBounds.yMax) / 2
    const samplePoints = [
      transformer.inverse(regBounds.xMin, regBounds.yMin),
      transformer.inverse(regBounds.xMax, regBounds.yMin),
      transformer.inverse(regBounds.xMax, regBounds.yMax),
      transformer.inverse(regBounds.xMin, regBounds.yMax),
      transformer.inverse(xMid, regBounds.yMin),
      transformer.inverse(xMid, regBounds.yMax),
      transformer.inverse(regBounds.xMin, yMid),
      transformer.inverse(regBounds.xMax, yMid),
    ]
    let regWest = Infinity
    let regEast = -Infinity
    let regSouth = Infinity
    let regNorth = -Infinity
    let hasValid = false
    for (const [lon, lat] of samplePoints) {
      if (!isFinite(lon) || !isFinite(lat)) continue
      hasValid = true
      regWest = Math.min(regWest, lon)
      regEast = Math.max(regEast, lon)
      regSouth = Math.min(regSouth, lat)
      regNorth = Math.max(regNorth, lat)
    }
    if (
      hasValid &&
      lonRangeOverlaps(west, east, regWest, regEast) &&
      regNorth >= south &&
      regSouth <= north
    ) {
      regions.push({ regionX, regionY })
    }
  }
  return regions
}

/** Fast projected viewport prefilter; callers verify inverse overlap. */
export function getCandidateRegions({
  west,
  south,
  east,
  north,
  transformer,
  numRegionsX,
  numRegionsY,
  regionW,
  regionH,
  width,
  height,
  xyLimits,
  latIsAscending,
}: {
  west: number
  south: number
  east: number
  north: number
  transformer: { forward: (lon: number, lat: number) => [number, number] }
  numRegionsX: number
  numRegionsY: number
  regionW: number
  regionH: number
  width: number
  height: number
  xyLimits: XYLimits
  latIsAscending: boolean
}): RegionCoordinate[] {
  const { xMin, xMax, yMin, yMax } = xyLimits
  const edgeSamples = 16
  let srcXMin = Infinity
  let srcXMax = -Infinity
  let srcYMin = Infinity
  let srcYMax = -Infinity
  let validCount = 0
  let totalCount = 0
  const record = ([x, y]: [number, number]) => {
    totalCount++
    if (!isFinite(x) || !isFinite(y)) return
    validCount++
    srcXMin = Math.min(srcXMin, x)
    srcXMax = Math.max(srcXMax, x)
    srcYMin = Math.min(srcYMin, y)
    srcYMax = Math.max(srcYMax, y)
  }
  // Densely sample viewport edges and interior to capture projection curvature
  // and extrema that may fall inside the viewport (e.g., pole in polar stereo).
  for (let i = 0; i <= edgeSamples; i++) {
    const t = i / edgeSamples
    const lon = west + t * (east - west)
    const lat = south + t * (north - south)
    record(transformer.forward(lon, south))
    record(transformer.forward(lon, north))
    record(transformer.forward(west, lat))
    record(transformer.forward(east, lat))
  }
  // Sample interior grid to catch extrema inside viewport (e.g., pole in polar stereo)
  const interiorSamples = 4
  for (let iy = 1; iy <= interiorSamples; iy++) {
    for (let ix = 1; ix <= interiorSamples; ix++) {
      record(
        transformer.forward(
          west + (ix / (interiorSamples + 1)) * (east - west),
          south + (iy / (interiorSamples + 1)) * (north - south)
        )
      )
    }
  }
  // No valid points — fall back to all regions
  if (validCount === 0) {
    const all: RegionCoordinate[] = []
    for (let ry = 0; ry < numRegionsY; ry++)
      for (let rx = 0; rx < numRegionsX; rx++)
        all.push({ regionX: rx, regionY: ry })
    return all
  }
  // Widen margin when some samples failed (projection boundary)
  const margin = validCount < totalCount ? 8 : 2
  const pxXMin = ((srcXMin - xMin) / (xMax - xMin)) * width
  const pxXMax = ((srcXMax - xMin) / (xMax - xMin)) * width
  const pxYMin = ((srcYMin - yMin) / (yMax - yMin)) * height
  const pxYMax = ((srcYMax - yMin) / (yMax - yMin)) * height
  let rYMin = Math.floor(pxYMin / regionH) - margin
  let rYMax = Math.floor(pxYMax / regionH) + margin
  if (!latIsAscending) {
    rYMin = Math.floor((height - pxYMax) / regionH) - margin
    rYMax = Math.floor((height - pxYMin) / regionH) + margin
  }
  let rXMin = Math.max(0, Math.floor(pxXMin / regionW) - margin)
  let rXMax = Math.min(numRegionsX - 1, Math.floor(pxXMax / regionW) + margin)
  rYMin = Math.max(0, rYMin)
  rYMax = Math.min(numRegionsY - 1, rYMax)
  // When the viewport straddles the antimeridian, forward-projecting the
  // sampled longitudes folds source X back on itself (e.g. proj4 adjust_lon),
  // so the srcX min/max span no longer bounds the visible columns. Treat
  // every X region as a candidate; the antimeridian-aware overlap check in
  // getVisibleRegions then keeps only the columns that are actually visible,
  // so this widens the search without over-fetching the result (issue #64).
  if (east < west || east > 180 || west < -180) {
    rXMin = 0
    rXMax = numRegionsX - 1
  }
  const candidates: RegionCoordinate[] = []
  for (let ry = rYMin; ry <= rYMax; ry++)
    for (let rx = rXMin; rx <= rXMax; rx++)
      candidates.push({ regionX: rx, regionY: ry })
  return candidates
}

export function getRegionBounds({
  regionX,
  regionY,
  levelMeta,
  xyLimits,
  latIsAscending,
}: {
  regionX: number
  regionY: number
  levelMeta: LevelMeta
  xyLimits: XYLimits | null
  latIsAscending: boolean
}): SourceBounds {
  const { width, height, regionSize } = levelMeta
  // A level that declares its own extent is placed against that; the dataset
  // extent describes the base level and would stretch any level covering less.
  const limits = levelMeta.xyLimits ?? xyLimits
  if (!limits) return { xMin: 0, xMax: 1, yMin: 0, yMax: 1 }
  const [regionH, regionW] = regionSize
  const { xMin, xMax, yMin, yMax } = limits
  const pxXStart = regionX * regionW
  const pxXEnd = Math.min(pxXStart + regionW, width)
  const pxYStart = regionY * regionH
  const pxYEnd = Math.min(pxYStart + regionH, height)
  const geoXMin = xMin + (pxXStart / width) * (xMax - xMin)
  const geoXMax = xMin + (pxXEnd / width) * (xMax - xMin)
  let geoYMin: number
  let geoYMax: number
  if (!latIsAscending) {
    geoYMax = yMax - (pxYStart / height) * (yMax - yMin)
    geoYMin = yMax - (pxYEnd / height) * (yMax - yMin)
  } else {
    geoYMin = yMin + (pxYStart / height) * (yMax - yMin)
    geoYMax = yMin + (pxYEnd / height) * (yMax - yMin)
  }
  return { xMin: geoXMin, xMax: geoXMax, yMin: geoYMin, yMax: geoYMax }
}

export function longitudeWorldFraction(bounds: { xMin: number; xMax: number }) {
  const rawSpan = bounds.xMax - bounds.xMin
  if (!Number.isFinite(rawSpan) || rawSpan === 0 || Math.abs(rawSpan) >= 360)
    return 1
  const span = rawSpan > 0 ? rawSpan : rawSpan + 360
  return Math.max(span / 360, Number.EPSILON)
}

export function selectLevelForZoom({
  mapZoom,
  xyLimits,
  levels,
  projection,
  lonIndex,
}: {
  mapZoom: number
  xyLimits: XYLimits | null
  levels: ResolutionLevel[]
  projection: ProjectionContext
  lonIndex?: number
}): number {
  if (!xyLimits || levels.length === 0) return 0
  const mapPixelsPerWorld = 256 * Math.pow(2, mapZoom)
  let worldFraction: number
  if (projection.kind === 'epsg4326') {
    // proj4's adjust_lon folds inputs outside [-180, 180] back into range
    // (e.g. forward(360,·)→0, forward(190,·)→-170), which collapses or
    // distorts the width for 0-360° stores and antimeridian-crossing
    // extents. Use the source longitude span for level selection because
    // render bounds may conservatively expand antimeridian-crossing regions
    // to full-world X for tile intersection.
    worldFraction = longitudeWorldFraction(xyLimits)
  } else if (projection.kind === 'epsg3857') {
    worldFraction = (xyLimits.xMax - xyLimits.xMin) / (2 * WEB_MERCATOR_EXTENT)
  } else if (projection.def && projection.toMercator) {
    const [minMercX] = projection.toMercator.forward(
      xyLimits.xMin,
      xyLimits.yMin
    )
    const [maxMercX] = projection.toMercator.forward(
      xyLimits.xMax,
      xyLimits.yMax
    )
    let dataWidthMeters = Math.abs(maxMercX - minMercX)
    const fullWorldMeters = 2 * WEB_MERCATOR_EXTENT
    // worldFraction > 1 means the corner projection failed: for projections
    // like MODIS sinusoidal the rectangular bbox corners fall outside the
    // valid CRS domain (near-polar latitudes give out-of-domain longitudes),
    // yielding a garbage or non-finite width. Fall back to the equatorial
    // strip (y=0), which stays in-domain for any cylindrical-like projection
    // and gives the correct horizontal extent.
    if (!isFinite(dataWidthMeters) || dataWidthMeters > fullWorldMeters) {
      const [eqMinX] = projection.toMercator.forward(xyLimits.xMin, 0)
      const [eqMaxX] = projection.toMercator.forward(xyLimits.xMax, 0)
      dataWidthMeters = Math.abs(eqMaxX - eqMinX)
    }
    worldFraction = Math.min(1, dataWidthMeters / fullWorldMeters)
  } else {
    worldFraction = (xyLimits.xMax - xyLimits.xMin) / 360
  }
  const levelResolutions: Array<{ index: number; effectivePixels: number }> = []
  for (let i = 0; i < levels.length; i++) {
    const shape = levels[i].shape
    if (!shape) continue
    const index = lonIndex ?? shape.length - 1
    levelResolutions.push({
      index: i,
      effectivePixels: shape[index] / worldFraction,
    })
  }
  if (levelResolutions.length === 0) return levels.length - 1
  levelResolutions.sort((a, b) => a.effectivePixels - b.effectivePixels)
  for (const level of levelResolutions)
    if (level.effectivePixels >= mapPixelsPerWorld) return level.index
  return levelResolutions[levelResolutions.length - 1].index
}

export function computeMercatorBoundsFromProjection(
  xyLimits: XYLimits | null,
  projection: ProjectionContext
): MercatorBounds {
  if (!xyLimits) return { x0: 0, y0: 0, x1: 1, y1: 1 }
  if (projection.kind === 'epsg4326')
    return boundsToMercatorNorm(xyLimits, 'EPSG:4326')
  if (projection.kind === 'epsg3857')
    return boundsToMercatorNorm(xyLimits, 'EPSG:3857')
  if (!projection.def || !projection.toMercator)
    return { x0: 0, y0: 0, x1: 1, y1: 1 }
  const result = sampleEdgesToMercatorBounds(
    xyLimits,
    projection.toMercator,
    20
  )
  if (!result) {
    console.warn('computeMercatorBoundsFromProjection: No valid samples found')
    return { x0: 0, y0: 0, x1: 1, y1: 1 }
  }
  return result
}

export function computeRegionMercatorBounds(
  bounds: SourceBounds,
  projection: ProjectionContext
): MercatorBounds {
  if (projection.kind === 'epsg4326')
    return boundsToMercatorNorm(bounds, 'EPSG:4326')
  if (projection.kind === 'epsg3857')
    return boundsToMercatorNorm(bounds, 'EPSG:3857')
  if (!projection.def || !projection.toMercator)
    return { x0: 0, y0: 0, x1: 1, y1: 1 }
  const result = sampleEdgesToMercatorBounds(bounds, projection.toMercator, 5)
  if (!result) {
    console.warn('computeRegionMercatorBounds: No valid samples found')
    return { x0: 0, y0: 0, x1: 1, y1: 1 }
  }
  return result
}
