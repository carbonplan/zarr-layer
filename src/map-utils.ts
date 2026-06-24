/**
 * @module map-utils
 *
 * Utility functions for custom layer integration.
 * Provides tile management, zoom level conversion,
 * coordinate transformations, and projection handling.
 * adapted from zarr-cesium/src/map-utils.ts
 */

import { MERCATOR_LAT_LIMIT, WEB_MERCATOR_EXTENT } from './constants'
import { MAPBOX_IDENTITY_MATRIX } from './mapbox-utils'
import type { ProjectionData, ShaderData } from './shaders'
import type { MapLike } from './types'

export interface MercatorBounds {
  x0: number
  y0: number
  x1: number
  y1: number
  latMin?: number
  latMax?: number
  lonMin?: number
  lonMax?: number
}

/**
 * Bounds for the source-projected ("wgs84" inputSpace) mesh path, in normalized
 * Web Mercator [0, 1] world coords. `y` can exceed [0, 1] for near-polar
 * vertices. The renderer derives its scale/shift uniforms from these.
 */
export interface Wgs84Bounds {
  x0: number
  y0: number
  x1: number
  y1: number
}

/**
 * Longitude-interval overlap test that is aware of antimeridian wrapping.
 *
 * The viewport range [viewWest, viewEast] reported by `map.getBounds()` may be
 * wrapped (viewWest > viewEast) or unwrapped with values well outside
 * [-180, 180]: it straddles ±180 (e.g. 175→195), and with renderWorldCopies the
 * user can pan into a neighbouring world copy, so the whole range can be offset
 * by any multiple of 360 (e.g. 535→555). The region range [regWest, regEast] is
 * expressed in the data's own longitude domain (typically [-180, 180] or
 * [0, 360]).
 *
 * A region overlaps the viewport when some world copy of it — shifted by k·360
 * for integer k — lands inside the viewport:
 *   regEast + k·360 >= viewWest   AND   regWest + k·360 <= viewEast
 * Solving each inequality for k yields a closed interval [kMin, kMax]; an
 * overlapping copy exists iff that interval contains an integer. Testing the
 * whole interval (rather than a fixed ±360) covers arbitrarily distant world
 * copies and viewports wider than 360° (issues #53/#64).
 */
export function lonRangeOverlaps(
  viewWest: number,
  viewEast: number,
  regWest: number,
  regEast: number
): boolean {
  let east = viewEast
  if (east < viewWest) east += 360
  const kMin = Math.ceil((viewWest - regEast) / 360)
  const kMax = Math.floor((east - regWest) / 360)
  return kMax >= kMin
}

/**
 * Converts longitude in degrees to normalized Web Mercator X coordinate [0, 1].
 * Clamps longitude to [-180, 180] range to handle coordinates that slightly
 * exceed bounds due to half-pixel expansion at coordinate array edges.
 * @param lon - Longitude in degrees.
 * @returns Normalized mercator X coordinate.
 */
export function lonToMercatorNorm(lon: number): number {
  const clamped = Math.max(-180, Math.min(180, lon))
  return (clamped + 180) / 360
}

/**
 * Converts latitude in degrees to normalized Web Mercator Y coordinate [0, 1].
 * Clamps latitude to valid Web Mercator range (±85.05112878°).
 * Note: Y=0 is at the north pole, Y=1 is at the south pole.
 * @param lat - Latitude in degrees.
 * @returns Normalized mercator Y coordinate.
 */
export function latToMercatorNorm(lat: number): number {
  const clamped = Math.max(
    -MERCATOR_LAT_LIMIT,
    Math.min(MERCATOR_LAT_LIMIT, lat)
  )
  return (
    (1 -
      Math.log(
        Math.tan((clamped * Math.PI) / 180) +
          1 / Math.cos((clamped * Math.PI) / 180)
      ) /
        Math.PI) /
    2
  )
}

export interface XYLimits {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
}

export function boundsToMercatorNorm(
  xyLimits: { xMin: number; xMax: number; yMin: number; yMax: number },
  crs: 'EPSG:4326' | 'EPSG:3857' | null
): MercatorBounds {
  if (crs === 'EPSG:3857') {
    return {
      x0: (xyLimits.xMin + WEB_MERCATOR_EXTENT) / (2 * WEB_MERCATOR_EXTENT),
      y0: (WEB_MERCATOR_EXTENT - xyLimits.yMax) / (2 * WEB_MERCATOR_EXTENT),
      x1: (xyLimits.xMax + WEB_MERCATOR_EXTENT) / (2 * WEB_MERCATOR_EXTENT),
      y1: (WEB_MERCATOR_EXTENT - xyLimits.yMin) / (2 * WEB_MERCATOR_EXTENT),
    }
  }

  let yMin = xyLimits.yMin
  let yMax = xyLimits.yMax
  if (yMin > yMax) {
    ;[yMin, yMax] = [yMax, yMin]
  }

  let x0: number
  let x1: number
  const lonSpan = xyLimits.xMax - xyLimits.xMin
  if (Math.abs(lonSpan) >= 360) {
    x0 = 0
    x1 = 1
  } else {
    let lonMin = xyLimits.xMin
    let lonMax = xyLimits.xMax

    // Keep 0-360-style intervals continuous when they sit entirely outside
    // [-180, 180]. Intervals that still cross the wrap boundary cannot be
    // represented as one MercatorBounds X span, so use full-world X bounds.
    while (lonMin >= 180 && lonMax > 180) {
      lonMin -= 360
      lonMax -= 360
    }
    while (lonMin < -180 && lonMax <= -180) {
      lonMin += 360
      lonMax += 360
    }

    x0 = lonToMercatorNormWrapped(lonMin)
    x1 = lonToMercatorNormWrapped(lonMax)
    if (lonSpan !== 0 && x1 <= x0) {
      x0 = 0
      x1 = 1
    }
  }

  return {
    x0,
    y0: latToMercatorNorm(yMax),
    x1,
    y1: latToMercatorNorm(yMin),
  }
}

function lonToMercatorNormWrapped(lon: number): number {
  if (lon >= -180 && lon <= 180) {
    return lonToMercatorNorm(lon)
  }
  const wrapped = ((((lon + 180) % 360) + 360) % 360) - 180
  return (wrapped + 180) / 360
}

// === Untiled mode utilities ===

/**
 * Convert a geographic coordinate to an array index.
 * Used for mapping viewport bounds to array pixel coordinates.
 * @param geo - Geographic coordinate value (lon or lat).
 * @param geoMin - Minimum geographic extent.
 * @param geoMax - Maximum geographic extent.
 * @param arraySize - Size of the array in this dimension.
 * @returns Array index (integer).
 */
export function geoToArrayIndex(
  geo: number,
  geoMin: number,
  geoMax: number,
  arraySize: number
): number {
  const normalized = (geo - geoMin) / (geoMax - geoMin)
  return Math.floor(
    Math.max(0, Math.min(arraySize - 1, normalized * arraySize))
  )
}

// === Projection utilities ===

/**
 * Detects if the given projection is a globe projection.
 * Works with both Mapbox (projection.name) and MapLibre (projection.type).
 */
export function isGlobeProjection(
  projection: { type?: unknown; name?: string } | null | undefined
): boolean {
  return projection?.type === 'globe' || projection?.name === 'globe'
}

interface ProjectionResolution {
  matrix: number[] | Float32Array | Float64Array | null
  shaderData?: ShaderData
  projectionData?: ProjectionData
  mapbox?:
    | {
        projection: { name: string }
        globeToMercatorMatrix: number[] | Float32Array | Float64Array
        transition: number
      }
    | undefined
}

export function resolveProjectionParams(
  params: unknown,
  projection?: { name: string },
  projectionToMercatorMatrix?: number[] | Float32Array | Float64Array,
  projectionToMercatorTransition?: number
): ProjectionResolution {
  type MatrixLike = number[] | Float32Array | Float64Array
  type ProjectionParams = {
    shaderData?: ShaderData
    defaultProjectionData?: {
      mainMatrix?: MatrixLike
      fallbackMatrix?: MatrixLike
      tileMercatorCoords?: number[]
      clippingPlane?: number[]
      projectionTransition?: number
    }
    modelViewProjectionMatrix?: MatrixLike
    projectionMatrix?: MatrixLike
  }

  const paramsObj =
    params &&
    typeof params === 'object' &&
    !Array.isArray(params) &&
    !ArrayBuffer.isView(params)
      ? (params as ProjectionParams)
      : null

  const shaderData = paramsObj?.shaderData
  let projectionData: ProjectionData | undefined
  const defaultProj = paramsObj?.defaultProjectionData
  if (
    defaultProj &&
    defaultProj.mainMatrix &&
    defaultProj.fallbackMatrix &&
    defaultProj.tileMercatorCoords &&
    defaultProj.clippingPlane &&
    typeof defaultProj.projectionTransition === 'number'
  ) {
    projectionData = {
      mainMatrix: defaultProj.mainMatrix,
      fallbackMatrix: defaultProj.fallbackMatrix,
      tileMercatorCoords: defaultProj.tileMercatorCoords as [
        number,
        number,
        number,
        number
      ],
      clippingPlane: defaultProj.clippingPlane as [
        number,
        number,
        number,
        number
      ],
      projectionTransition: defaultProj.projectionTransition,
    }
  }
  let matrix: number[] | Float32Array | Float64Array | null = null
  if (projectionData?.mainMatrix && projectionData.mainMatrix.length) {
    matrix = projectionData.mainMatrix
  } else if (
    Array.isArray(params) ||
    params instanceof Float32Array ||
    params instanceof Float64Array
  ) {
    matrix = params as number[] | Float32Array | Float64Array
  } else if (paramsObj?.modelViewProjectionMatrix) {
    matrix = paramsObj.modelViewProjectionMatrix
  } else if (paramsObj?.projectionMatrix) {
    matrix = paramsObj.projectionMatrix
  }

  // Mapbox detection: passes projection param (globe mode) or matrix directly (mercator mode)
  const paramsIsMatrix =
    Array.isArray(params) ||
    params instanceof Float32Array ||
    params instanceof Float64Array
  const isMapbox = !!projection || paramsIsMatrix

  // For Mapbox, always provide mapbox params (even in mercator mode) to avoid special-case logic
  // In mercator mode: use identity matrix and transition=1 (pure mercator)
  // In globe mode: use provided values
  const mapbox = isMapbox
    ? {
        projection: projection ?? { name: 'mercator' },
        globeToMercatorMatrix:
          projectionToMercatorMatrix ?? MAPBOX_IDENTITY_MATRIX,
        transition:
          typeof projectionToMercatorTransition === 'number'
            ? projectionToMercatorTransition
            : 1, // Default to mercator (transition=1) when not in globe mode
      }
    : undefined

  return { matrix, shaderData, projectionData, mapbox }
}

export function computeWorldOffsets(
  map: MapLike | null,
  isGlobe: boolean
): number[] {
  if (!map) return [0]

  const bounds = map.getBounds ? map.getBounds() : null
  if (!bounds) return [0]

  const renderWorldCopies =
    typeof map.getRenderWorldCopies === 'function'
      ? map.getRenderWorldCopies()
      : true
  if (isGlobe || !renderWorldCopies) return [0]

  const west = bounds.getWest()
  const east = bounds.getEast()

  let effectiveEast = east
  if (west > east) {
    effectiveEast = east + 360
  }

  const minWorld = Math.floor((west + 180) / 360)
  const maxWorld = Math.floor((effectiveEast + 180) / 360)

  const worldOffsets: number[] = []
  for (let i = minWorld; i <= maxWorld; i++) {
    worldOffsets.push(i)
  }
  return worldOffsets.length > 0 ? worldOffsets : [0]
}
