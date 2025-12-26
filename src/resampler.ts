/**
 * CPU-based texture resampling for CRS → EPSG:3857 conversion.
 *
 * Resamples source tile data into Mercator space before GPU upload using
 * nearest neighbor sampling to preserve exact pixel values. This allows
 * simple linear texture coordinates to work correctly on Mercator maps.
 *
 * Supports:
 * - EPSG:4326 (lat/lon) → EPSG:3857 (Web Mercator)
 * - Arbitrary CRS via proj4 → EPSG:3857 (Web Mercator)
 */

import { mercatorNormToLat, mercatorNormToLon } from './map-utils'
import { MERCATOR_LAT_LIMIT } from './constants'
import { createTransformer, sourceCRSToPixel } from './projection-utils'
import type { Bounds } from './types'

// Small epsilon for floating point comparisons at region boundaries
// Using 1e-9 degrees is ~0.1mm at the equator - plenty of tolerance for fp errors
const BOUNDS_EPSILON = 1e-9

export interface ResampleOptions {
  /** Source data in source CRS space */
  sourceData: Float32Array
  /** Source dimensions [width, height] */
  sourceSize: [number, number]
  /** Source geographic bounds [west, south, east, north] in degrees (for EPSG:4326) */
  sourceBounds: Bounds
  /** Target dimensions [width, height] - typically same as source */
  targetSize: [number, number]
  /** Target mercator bounds [x0, y0, x1, y1] in normalized [0,1] space */
  targetMercatorBounds: Bounds
  /** Fill value for nodata pixels */
  fillValue: number
  /** Whether source latitude is ascending (row 0 = south) */
  latIsAscending: boolean | null
  /** Proj4 definition string for arbitrary CRS reprojection (optional) */
  proj4?: string | null
}

/**
 * Nearest neighbor sampling at fractional coordinates.
 * Uses pixel-center model where coordinate 0 is the center of pixel 0.
 * Valid coordinate range is [-0.5, N-0.5] for N pixels.
 * Coordinates outside this range are clamped (CLAMP_TO_EDGE behavior).
 */
function nearestSample(
  data: Float32Array,
  width: number,
  height: number,
  x: number,
  y: number
): number {
  // Round to nearest pixel (pixel-center model)
  const px = Math.max(0, Math.min(width - 1, Math.round(x)))
  const py = Math.max(0, Math.min(height - 1, Math.round(y)))
  return data[py * width + px]
}

import { WEB_MERCATOR_EXTENT } from './constants'

/**
 * Convert normalized mercator [0,1] to Web Mercator meters.
 * In normalized mercator: Y=0 is north (top), Y=1 is south (bottom)
 * In Web Mercator meters: positive Y is north, negative Y is south
 */
function normMercatorToMeters(normX: number, normY: number): [number, number] {
  const x = (normX - 0.5) * 2 * WEB_MERCATOR_EXTENT
  // Invert Y: normY=0 (top/north) → +worldSize, normY=1 (bottom/south) → -worldSize
  const y = (0.5 - normY) * 2 * WEB_MERCATOR_EXTENT
  return [x, y]
}

/**
 * Resample source data into EPSG:3857 (Web Mercator) space.
 *
 * For EPSG:4326 source data:
 * 1. Convert pixel coord → normalized Mercator [0,1]
 * 2. Convert normalized Mercator → lat/lon
 * 3. Convert lat/lon → source pixel coord
 * 4. Sample source with nearest neighbor
 *
 * For arbitrary CRS (via proj4):
 * 1. Convert pixel coord → normalized Mercator [0,1]
 * 2. Convert normalized Mercator → Web Mercator meters
 * 3. Use proj4 inverse to convert Web Mercator → source CRS
 * 4. Convert source CRS coords → source pixel coord
 * 5. Sample source with nearest neighbor
 */
export function resampleToMercator(
  options: ResampleOptions
): Float32Array<ArrayBuffer> {
  const {
    sourceData,
    sourceSize: [srcW, srcH],
    sourceBounds: [west, south, east, north],
    targetSize: [tgtW, tgtH],
    targetMercatorBounds: [mercX0, mercY0, mercX1, mercY1],
    fillValue,
    latIsAscending,
    proj4: proj4def,
  } = options

  // If proj4 is provided, use proj4-based reprojection
  if (proj4def) {
    return resampleWithProj4(
      sourceData,
      srcW,
      srcH,
      tgtW,
      tgtH,
      mercX0,
      mercY0,
      mercX1,
      mercY1,
      fillValue,
      proj4def,
      [west, south, east, north], // Pass region bounds in source CRS
      latIsAscending
    )
  }

  const result = new Float32Array(tgtW * tgtH)
  result.fill(fillValue)

  // Source geographic range
  const lonRange = east - west
  const latRange = north - south

  // Handle antimeridian crossing (west > east means crossing ±180)
  const crossesAntimeridian = west > east

  for (let tgtY = 0; tgtY < tgtH; tgtY++) {
    for (let tgtX = 0; tgtX < tgtW; tgtX++) {
      // Convert target pixel to normalized mercator [0,1]
      // Use pixel-center model: pixel i covers [(i/N)*range, ((i+1)/N)*range]
      // with center at ((i+0.5)/N)*range. This ensures the texture fully covers
      // the mercator bounds when sampled with UV 0-1.
      const normMercX = mercX0 + ((tgtX + 0.5) / tgtW) * (mercX1 - mercX0)
      const normMercY = mercY0 + ((tgtY + 0.5) / tgtH) * (mercY1 - mercY0)

      // Convert normalized mercator to lat/lon
      const lon = mercatorNormToLon(normMercX)
      const lat = mercatorNormToLat(normMercY)

      // Clamp latitude to Mercator limits
      if (lat < -MERCATOR_LAT_LIMIT || lat > MERCATOR_LAT_LIMIT) {
        continue // Leave as fill value
      }

      // Check if within source longitude bounds
      let adjustedLon = lon
      if (crossesAntimeridian) {
        // Source spans antimeridian: west > east (e.g., 170 to -170)
        // Adjust lon to be in the same "space" as source bounds
        if (lon < 0 && west > 0) {
          adjustedLon = lon + 360
        }
        if (
          adjustedLon < west - BOUNDS_EPSILON ||
          adjustedLon > east + 360 + BOUNDS_EPSILON
        ) {
          continue // Leave as fill value
        }
        // Compute source X in the adjusted space using pixel-center model
        // Source pixel i is centered at ((i+0.5)/srcW) of the geographic range
        const effectiveWest = west
        const effectiveEast = east + 360
        const srcX =
          ((adjustedLon - effectiveWest) / (effectiveEast - effectiveWest)) *
            srcW -
          0.5

        // Check latitude bounds (with epsilon for floating point precision)
        if (lat < south - BOUNDS_EPSILON || lat > north + BOUNDS_EPSILON) {
          continue
        }

        // Compute source Y based on latIsAscending using pixel-center model
        let srcY: number
        if (latIsAscending === false) {
          // Row 0 = north (latMax), row N-1 = south (latMin)
          srcY = ((north - lat) / latRange) * srcH - 0.5
        } else {
          // Row 0 = south (latMin), row N-1 = north (latMax)
          srcY = ((lat - south) / latRange) * srcH - 0.5
        }

        result[tgtY * tgtW + tgtX] = nearestSample(
          sourceData,
          srcW,
          srcH,
          srcX,
          srcY
        )
      } else {
        // Normal case: source doesn't cross antimeridian
        // Handle 0-360° longitude convention: if source uses 0-360 (west > 180),
        // convert negative lon to 0-360 range
        let checkLon = lon
        if (west > 180 && lon < 0) {
          checkLon = lon + 360
        }

        // Use epsilon tolerance for boundary checks to handle floating point precision
        // at region edges (prevents gaps between adjacent regions)
        if (
          checkLon < west - BOUNDS_EPSILON ||
          checkLon > east + BOUNDS_EPSILON ||
          lat < south - BOUNDS_EPSILON ||
          lat > north + BOUNDS_EPSILON
        ) {
          continue // Leave as fill value
        }

        // Convert lon/lat to source pixel coordinates using pixel-center model
        // Source pixel i is centered at ((i+0.5)/srcN) of the geographic range
        const srcX = ((checkLon - west) / lonRange) * srcW - 0.5

        // Y coordinate depends on data orientation
        let srcY: number
        if (latIsAscending === false) {
          // Row 0 = north (latMax), row N-1 = south (latMin)
          srcY = ((north - lat) / latRange) * srcH - 0.5
        } else {
          // Row 0 = south (latMin), row N-1 = north (latMax) - default
          srcY = ((lat - south) / latRange) * srcH - 0.5
        }

        result[tgtY * tgtW + tgtX] = nearestSample(
          sourceData,
          srcW,
          srcH,
          srcX,
          srcY
        )
      }
    }
  }

  return result
}

/**
 * Check if resampling is needed for a given CRS.
 * Resampling is needed for:
 * - EPSG:4326 data (lat/lon needs Mercator correction)
 * - Any custom projection via proj4
 */
export function needsResampling(
  crs: string,
  proj4def?: string | null
): boolean {
  if (proj4def) return true
  return crs === 'EPSG:4326'
}

/**
 * Resample source data using proj4 for arbitrary CRS → Web Mercator conversion.
 *
 * For each pixel in the target (mercator) space:
 * 1. Convert pixel coord → normalized Mercator [0,1]
 * 2. Convert normalized Mercator → Web Mercator meters
 * 3. Use proj4 inverse to convert Web Mercator → source CRS coords
 * 4. Map source CRS coords → source pixel indices (using REGION bounds)
 * 5. Sample source with nearest neighbor
 */
function resampleWithProj4(
  sourceData: Float32Array,
  srcW: number,
  srcH: number,
  tgtW: number,
  tgtH: number,
  mercX0: number,
  mercY0: number,
  mercX1: number,
  mercY1: number,
  fillValue: number,
  proj4def: string,
  regionBounds: Bounds, // Region bounds in source CRS
  latIsAscending: boolean | null
): Float32Array<ArrayBuffer> {
  const result = new Float32Array(tgtW * tgtH)
  result.fill(fillValue)

  // Create proj4 transformer (source CRS ↔ Web Mercator)
  const transformer = createTransformer(proj4def, regionBounds)

  // Use REGION bounds for pixel mapping, not full grid bounds
  const [xMin, yMin, xMax, yMax] = regionBounds

  for (let tgtY = 0; tgtY < tgtH; tgtY++) {
    for (let tgtX = 0; tgtX < tgtW; tgtX++) {
      // Convert target pixel to normalized mercator [0,1]
      const normMercX = mercX0 + ((tgtX + 0.5) / tgtW) * (mercX1 - mercX0)
      const normMercY = mercY0 + ((tgtY + 0.5) / tgtH) * (mercY1 - mercY0)

      // Convert normalized mercator to Web Mercator meters
      const [mercMetersX, mercMetersY] = normMercatorToMeters(
        normMercX,
        normMercY
      )

      // Transform Web Mercator → source CRS
      const [srcCrsX, srcCrsY] = transformer.inverse(mercMetersX, mercMetersY)

      // Guard against NaN/Infinity from proj4 inverse transforms
      if (!isFinite(srcCrsX) || !isFinite(srcCrsY)) {
        continue // Leave as fill value
      }

      // Check if within REGION bounds (not full grid bounds)
      if (
        srcCrsX < xMin ||
        srcCrsX > xMax ||
        srcCrsY < yMin ||
        srcCrsY > yMax
      ) {
        continue // Leave as fill value
      }

      // Convert source CRS coords to pixel indices using REGION bounds
      const [srcX, srcY] = sourceCRSToPixel(
        srcCrsX,
        srcCrsY,
        regionBounds,
        srcW,
        srcH,
        latIsAscending
      )

      // Sample source data
      result[tgtY * tgtW + tgtX] = nearestSample(
        sourceData,
        srcW,
        srcH,
        srcX,
        srcY
      )
    }
  }

  return result
}
