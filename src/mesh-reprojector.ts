/**
 * Mesh generation for client-side raster reprojection.
 *
 * Uses Delaunay triangulation on a uniform grid for reliable mesh generation.
 * Outputs EPSG:4326 coordinates for two-stage GPU reprojection
 * (source CRS → WGS84 on CPU, WGS84 → Mercator in shader).
 */

import Delaunator from 'delaunator'
import { RasterReprojector } from '@developmentseed/raster-reproject'
import { lonToMercatorNorm, type Wgs84Bounds } from './map-utils'
import {
  pixelToSourceCRS,
  sourceCRSToPixel,
  type ProjectionTransformer,
} from './projection-utils'
import { DEFAULT_MESH_MAX_ERROR } from './constants'

// ============================================================================
// Module constants
// ============================================================================

/** Maximum vertices for adaptive mesh refinement (prevents hanging on polar data) */
const MAX_ADAPTIVE_VERTICES = 10000

/** Maximum iterations for adaptive mesh refinement */
const MAX_ITERATIONS = 1000

/**
 * Longitude coverage threshold (degrees) for polar data detection.
 * Polar projections (e.g., EPSG:3031) span most longitudes when transformed
 * to WGS84. If coverage exceeds this threshold, skip antimeridian detection
 * to avoid false positives from projection singularities.
 */
const POLAR_LON_COVERAGE_THRESHOLD = 270

// ============================================================================
// Interfaces
// ============================================================================

interface ReprojectorConfig {
  bounds: [number, number, number, number]
  width: number
  height: number
  latIsAscending: boolean
  transformer: ProjectionTransformer
}

export interface AdaptiveMeshResult {
  positions: Float32Array // Normalized 4326 coords [-1,1] for shader
  texCoords: Float32Array // UVs for texture sampling
  indices: Uint32Array // Triangle indices
  wgs84Bounds: Wgs84Bounds
}

export interface HybridMeshOptions {
  geoBounds: { xMin: number; xMax: number; yMin: number; yMax: number }
  width: number
  height: number
  subdivisions: number
  transformer: ProjectionTransformer
  latIsAscending: boolean
  maxError?: number
}

// ============================================================================
// Shared helpers
// ============================================================================

/**
 * Normalize longitude to [-180, 180) range using true modulo wrapping.
 * Handles any input range (e.g., 540° → -180°, -540° → 180°).
 */
function normalizeLon180(lon: number): number {
  if (!isFinite(lon)) return lon
  // Shift to [0, 360) then back to [-180, 180)
  return ((((lon + 180) % 360) + 360) % 360) - 180
}

/**
 * Create a configured RasterReprojector with proper pixel-to-CRS transforms.
 */
function createReprojector(config: ReprojectorConfig): RasterReprojector {
  const { bounds, width, height, latIsAscending, transformer } = config

  // The reprojector converts UV [0,1] → pixel [0, width-1] internally.
  // Our edge-based pixelToSourceCRS expects [0, width] → [xMin, xMax].
  // We scale the pixel values to bridge this: scaledPx = px * width / (width - 1)
  const scaleX = width > 1 ? width / (width - 1) : 1
  const scaleY = height > 1 ? height / (height - 1) : 1

  return new RasterReprojector(
    {
      // Pixel coords [0, width-1] → source CRS coords (scaled to edge-based model)
      forwardTransform: (px: number, py: number) =>
        pixelToSourceCRS(
          px * scaleX,
          py * scaleY,
          bounds,
          width,
          height,
          latIsAscending
        ),
      // Source CRS coords → pixel coords [0, width-1] (unscale from edge-based model)
      inverseTransform: (x: number, y: number) => {
        const [scaledPx, scaledPy] = sourceCRSToPixel(
          x,
          y,
          bounds,
          width,
          height,
          latIsAscending
        )
        return [scaledPx / scaleX, scaledPy / scaleY]
      },
      // Source CRS → EPSG:4326 (lon, lat)
      forwardReproject: (x: number, y: number) => transformer.forward(x, y),
      // EPSG:4326 → source CRS
      inverseReproject: (lon: number, lat: number) =>
        transformer.inverse(lon, lat),
    },
    width,
    height
  )
}

/**
 * Compute normalized WGS84 bounds from min/max lon/lat values.
 * For antimeridian-crossing data, minLon > maxLon (e.g., 170° to -170°).
 *
 * Note: For Mercator rendering, latitudes beyond ±85.05° will be clamped
 * in the shader. This may cause visual artifacts for polar data.
 * Globe mode can display full polar data without issues.
 */
function computeWgs84Bounds(
  minLon: number,
  maxLon: number,
  minLat: number,
  maxLat: number,
  crossesAntimeridian: boolean = false
): Wgs84Bounds {
  return {
    lon0: lonToMercatorNorm(minLon),
    lat0: (minLat + 90) / 180,
    lon1: lonToMercatorNorm(maxLon),
    lat1: (maxLat + 90) / 180,
    crossesAntimeridian,
  }
}

/**
 * Normalize lon/lat positions to local [-1, 1] coordinates.
 * This matches the standard mercator path where createSubdividedQuad outputs [-1, 1].
 *
 * For antimeridian-crossing bounds (minLon > maxLon in degrees, e.g., 170° to -170°):
 * - Actual range is (360 - minLon + maxLon), e.g., (360 - 170 + (-170)) = 20°
 * - Longitudes < minLon get +360 to create a continuous range
 */
function normalizeToLocalCoords(
  positions: ArrayLike<number>,
  minLon: number,
  maxLon: number,
  minLat: number,
  maxLat: number,
  crossesAntimeridian: boolean = false
): Float32Array {
  const numVerts = positions.length / 2
  const normalized = new Float32Array(numVerts * 2)

  // For antimeridian crossing: actual range wraps through 180°
  // e.g., minLon=170, maxLon=-170 means range of 20° (not -340°)
  const lonRange = crossesAntimeridian
    ? 360 - minLon + maxLon || 1
    : maxLon - minLon || 1
  const latRange = maxLat - minLat || 1

  for (let i = 0; i < numVerts; i++) {
    let lon = positions[i * 2]
    const lat = positions[i * 2 + 1]

    // Normalize longitude to [-180, 180) range first
    // This handles the case where proj4 outputs 180° which equals -180°
    lon = normalizeLon180(lon)

    // For antimeridian crossing, shift negative longitudes up by 360
    // so they're in a continuous range with positive longitudes
    if (crossesAntimeridian && lon < minLon) {
      lon += 360
    }

    // Map [minLon, maxLon] → [-1, 1] and [minLat, maxLat] → [-1, 1]
    normalized[i * 2] = ((lon - minLon) / lonRange) * 2 - 1
    normalized[i * 2 + 1] = ((lat - minLat) / latRange) * 2 - 1
  }

  return normalized
}

// ============================================================================
// Antimeridian handling
// ============================================================================

// --- Detection ---

/**
 * Check if an edge crosses the antimeridian (spans > 180° longitude).
 */
function edgeCrossesAntimeridian(lon1: number, lon2: number): boolean {
  return Math.abs(lon1 - lon2) > 180
}

// --- Triangle splitting ---

/**
 * Compute the intersection point of an edge with the antimeridian.
 * Returns the latitude at which the edge crosses lon = ±180°.
 */
function computeAntimeridianIntersection(
  lon1: number,
  lat1: number,
  lon2: number,
  lat2: number
): { lat: number; t: number } {
  // Shift longitudes so they're continuous across the antimeridian
  // If lon1 is positive and lon2 is negative (or vice versa) with large span,
  // shift the negative one up by 360
  let l1 = lon1
  let l2 = lon2
  if (lon1 > 0 && lon2 < 0 && lon1 - lon2 > 180) {
    l2 += 360 // e.g., -170 becomes 190
  } else if (lon2 > 0 && lon1 < 0 && lon2 - lon1 > 180) {
    l1 += 360
  }

  // Now interpolate to find where lon = 180 (the antimeridian)
  const t = (180 - l1) / (l2 - l1)
  const lat = lat1 + t * (lat2 - lat1)
  return { lat, t }
}

/**
 * Result of processing triangles at the antimeridian.
 */
interface SplitResult {
  positions: Float64Array
  texCoords: Float64Array
  indices: Uint32Array
}

/**
 * Process triangles that span the antimeridian by splitting them.
 * Filters out triangles with non-finite coordinates.
 * When canCrossAntimeridian is false, skips crossing checks for better performance.
 */
function splitAntimeridianTriangles(
  wgs84Positions: Float64Array,
  texCoords: Float64Array,
  triangles: ArrayLike<number>,
  canCrossAntimeridian: boolean
): SplitResult {
  const numVerts = wgs84Positions.length / 2

  // Pre-compute which vertices are valid (have finite coords)
  const validVertex = new Uint8Array(numVerts)
  for (let i = 0; i < numVerts; i++) {
    const lon = wgs84Positions[i * 2]
    const lat = wgs84Positions[i * 2 + 1]
    validVertex[i] = isFinite(lon) && isFinite(lat) ? 1 : 0
  }

  // Fast path: no crossing possible, just filter invalid triangles
  if (!canCrossAntimeridian) {
    const newIndices: number[] = []
    for (let i = 0; i < triangles.length; i += 3) {
      const i0 = triangles[i]
      const i1 = triangles[i + 1]
      const i2 = triangles[i + 2]
      if (validVertex[i0] && validVertex[i1] && validVertex[i2]) {
        newIndices.push(i0, i1, i2)
      }
    }
    return {
      positions: wgs84Positions,
      texCoords,
      indices: new Uint32Array(newIndices),
    }
  }

  // Estimate capacity: most triangles won't split
  const newPositions: number[] = new Array(wgs84Positions.length)
  for (let i = 0; i < wgs84Positions.length; i++)
    newPositions[i] = wgs84Positions[i]
  const newTexCoords: number[] = new Array(texCoords.length)
  for (let i = 0; i < texCoords.length; i++) newTexCoords[i] = texCoords[i]
  const newIndices: number[] = []

  for (let i = 0; i < triangles.length; i += 3) {
    const i0 = triangles[i]
    const i1 = triangles[i + 1]
    const i2 = triangles[i + 2]

    // Skip triangles with invalid vertices
    if (!validVertex[i0] || !validVertex[i1] || !validVertex[i2]) {
      continue
    }

    // Inline vertex access (avoid object allocation)
    const lon0raw = wgs84Positions[i0 * 2]
    const lat0 = wgs84Positions[i0 * 2 + 1]
    const u0 = texCoords[i0 * 2]
    const v0 = texCoords[i0 * 2 + 1]

    const lon1raw = wgs84Positions[i1 * 2]
    const lat1 = wgs84Positions[i1 * 2 + 1]
    const u1 = texCoords[i1 * 2]
    const v1 = texCoords[i1 * 2 + 1]

    const lon2raw = wgs84Positions[i2 * 2]
    const lat2 = wgs84Positions[i2 * 2 + 1]
    const u2 = texCoords[i2 * 2]
    const v2 = texCoords[i2 * 2 + 1]

    // Normalize longitudes
    const lon0 = normalizeLon180(lon0raw)
    const lon1 = normalizeLon180(lon1raw)
    const lon2 = normalizeLon180(lon2raw)

    // Check which edges cross the antimeridian
    const cross01 = edgeCrossesAntimeridian(lon0, lon1)
    const cross12 = edgeCrossesAntimeridian(lon1, lon2)
    const cross20 = edgeCrossesAntimeridian(lon2, lon0)
    const crossCount = (cross01 ? 1 : 0) + (cross12 ? 1 : 0) + (cross20 ? 1 : 0)

    if (crossCount === 0) {
      newIndices.push(i0, i1, i2)
    } else if (crossCount === 2) {
      // Determine vertex order: [alone, other1, other2]
      const vertexOrder = !cross01
        ? [2, 0, 1]
        : !cross12
        ? [0, 1, 2]
        : [1, 2, 0]
      const [ai, o1i, o2i] = vertexOrder

      const idxArr = [i0, i1, i2]
      const lonArr = [lon0, lon1, lon2]
      const latArr = [lat0, lat1, lat2]
      const uArr = [u0, u1, u2]
      const vArr = [v0, v1, v2]

      const alone = idxArr[ai]
      const other1 = idxArr[o1i]
      const other2 = idxArr[o2i]
      const lonAlone = lonArr[ai]
      const latAlone = latArr[ai]
      const uAlone = uArr[ai]
      const vAlone = vArr[ai]
      const lonOther1 = lonArr[o1i]
      const latOther1 = latArr[o1i]
      const uOther1 = uArr[o1i]
      const vOther1 = vArr[o1i]
      const lonOther2 = lonArr[o2i]
      const latOther2 = latArr[o2i]
      const uOther2 = uArr[o2i]
      const vOther2 = vArr[o2i]

      // Compute intersection points
      const int1 = computeAntimeridianIntersection(
        lonAlone,
        latAlone,
        lonOther1,
        latOther1
      )
      const int2 = computeAntimeridianIntersection(
        lonAlone,
        latAlone,
        lonOther2,
        latOther2
      )

      // Interpolate texture coordinates
      const intU1 = uAlone + int1.t * (uOther1 - uAlone)
      const intV1 = vAlone + int1.t * (vOther1 - vAlone)
      const intU2 = uAlone + int2.t * (uOther2 - uAlone)
      const intV2 = vAlone + int2.t * (vOther2 - vAlone)

      // Determine which side the alone vertex is on
      const aloneOnEast = lonAlone > 0
      const lonAloneSide = aloneOnEast ? 179.9999 : -179.9999
      const lonOtherSide = aloneOnEast ? -179.9999 : 179.9999

      // Add new vertices at intersection points
      const baseIdx = newPositions.length / 2
      newPositions.push(lonAloneSide, int1.lat, lonAloneSide, int2.lat)
      newPositions.push(lonOtherSide, int1.lat, lonOtherSide, int2.lat)
      newTexCoords.push(intU1, intV1, intU2, intV2)
      newTexCoords.push(intU1, intV1, intU2, intV2)

      const intAlone1 = baseIdx
      const intAlone2 = baseIdx + 1
      const intOther1 = baseIdx + 2
      const intOther2 = baseIdx + 3

      // Create triangles
      newIndices.push(alone, intAlone1, intAlone2)
      newIndices.push(intOther1, other1, other2)
      newIndices.push(intOther1, other2, intOther2)
    } else {
      // crossCount === 1 or 3: preserve triangle to avoid holes
      newIndices.push(i0, i1, i2)
    }
  }

  return {
    positions: new Float64Array(newPositions),
    texCoords: new Float64Array(newTexCoords),
    indices: new Uint32Array(newIndices),
  }
}

// ============================================================================
// Public API
// ============================================================================

/**
 * Create a hybrid mesh that combines adaptive refinement with uniform grid vertices.
 * This gives both: accurate reprojection (adaptive) + even coverage for globe curvature (uniform).
 *
 * Algorithm:
 * 1. Run adaptive mesh to get error-driven vertices in UV space
 * 2. Generate uniform grid vertices in UV space
 * 3. Merge and re-triangulate with Delaunator
 * 4. Transform all vertices to WGS84
 */
export function createHybridMesh(
  options: HybridMeshOptions
): AdaptiveMeshResult {
  const {
    geoBounds,
    width,
    height,
    subdivisions,
    transformer,
    latIsAscending,
    maxError = DEFAULT_MESH_MAX_ERROR,
  } = options
  const { xMin, xMax, yMin, yMax } = geoBounds
  const bounds: [number, number, number, number] = [xMin, yMin, xMax, yMax]

  const reprojector = createReprojector({
    bounds,
    width,
    height,
    latIsAscending,
    transformer,
  })

  // Run adaptive refinement with vertex limit to prevent hanging on polar data.
  for (
    let i = 0;
    i < MAX_ITERATIONS && reprojector.getMaxError() > maxError;
    i++
  ) {
    const prevVertCount = reprojector.uvs.length / 2
    reprojector.refine()

    const newVertCount = reprojector.uvs.length / 2
    if (
      newVertCount >= MAX_ADAPTIVE_VERTICES ||
      newVertCount === prevVertCount
    ) {
      break
    }
  }

  // Collect adaptive mesh UVs
  const adaptiveUVs = reprojector.uvs // [u0, v0, u1, v1, ...]

  // Merge adaptive + uniform grid UVs directly into pre-sized array
  // (duplicates are harmless for Delaunator)
  const uniformVertices = (subdivisions + 1) ** 2
  const mergedUVs = new Float64Array(adaptiveUVs.length + uniformVertices * 2)
  mergedUVs.set(adaptiveUVs)

  let offset = adaptiveUVs.length
  for (let row = 0; row <= subdivisions; row++) {
    for (let col = 0; col <= subdivisions; col++) {
      mergedUVs[offset++] = col / subdivisions
      mergedUVs[offset++] = row / subdivisions
    }
  }

  // Triangulate merged UVs with Delaunator
  const delaunay = new Delaunator(mergedUVs)
  const triangles = delaunay.triangles

  // Transform all UVs to WGS84, collect normalized lons, and track lat bounds
  const numVerts = mergedUVs.length / 2
  const wgs84Positions = new Float64Array(numVerts * 2)
  const lons: number[] = []
  let minLat = Infinity,
    maxLat = -Infinity

  for (let i = 0; i < numVerts; i++) {
    const u = mergedUVs[i * 2]
    const v = mergedUVs[i * 2 + 1]

    // UV → source CRS
    const srcX = xMin + u * (xMax - xMin)
    const srcY = latIsAscending
      ? yMin + v * (yMax - yMin)
      : yMax - v * (yMax - yMin)

    // Source CRS → WGS84
    const [lon, lat] = transformer.forward(srcX, srcY)

    wgs84Positions[i * 2] = lon
    wgs84Positions[i * 2 + 1] = lat

    if (isFinite(lon) && isFinite(lat)) {
      lons.push(normalizeLon180(lon))
      minLat = Math.min(minLat, lat)
      maxLat = Math.max(maxLat, lat)
    }
  }

  // Fallback if no valid coords found
  if (!isFinite(minLat)) minLat = -90
  if (!isFinite(maxLat)) maxLat = 90

  // Detect antimeridian crossing and compute lon bounds from sorted list
  let minLon = -180,
    maxLon = 180,
    crossesAntimeridian = false
  if (lons.length > 0) {
    lons.sort((a, b) => a - b)

    // Set bounds from normalized sorted list
    minLon = lons[0]
    maxLon = lons[lons.length - 1]
    const lonCoverage = maxLon - minLon

    // Skip crossing detection if no coverage (crossesAntimeridian stays false)
    if (lonCoverage > 0) {
      // Find largest internal gap
      let maxGap = 0
      let gapEndIndex = 0
      for (let i = 0; i < lons.length - 1; i++) {
        const gap = lons[i + 1] - lons[i]
        if (gap > maxGap) {
          maxGap = gap
          gapEndIndex = i + 1
        }
      }

      // Wrap-around gap
      const wrapGap = lons[0] + 360 - lons[lons.length - 1]

      // Crossing if wrap gap < max internal gap (and not polar data)
      if (wrapGap < maxGap && lonCoverage < POLAR_LON_COVERAGE_THRESHOLD) {
        crossesAntimeridian = true
        minLon = lons[gapEndIndex]
        maxLon = lons[gapEndIndex - 1]
      }
    }
  }

  // Normalize edge case: if minLon is exactly 180°, it's the same as -180°
  if (crossesAntimeridian && minLon >= 180) {
    minLon = minLon - 360
    if (minLon <= maxLon) {
      crossesAntimeridian = false
    }
  }

  // Split triangles at antimeridian (also filters invalid triangles)
  // If lon range < 180°, no edge can cross antimeridian, so skip crossing checks
  const canCrossAntimeridian = crossesAntimeridian || maxLon - minLon >= 180
  const texCoords = new Float64Array(mergedUVs)
  const splitResult = splitAntimeridianTriangles(
    wgs84Positions,
    texCoords,
    triangles,
    canCrossAntimeridian
  )

  // Normalize with antimeridian awareness
  const positions = normalizeToLocalCoords(
    splitResult.positions,
    minLon,
    maxLon,
    minLat,
    maxLat,
    crossesAntimeridian
  )

  const wgs84Bounds = computeWgs84Bounds(
    minLon,
    maxLon,
    minLat,
    maxLat,
    crossesAntimeridian
  )

  return {
    positions,
    texCoords: new Float32Array(splitResult.texCoords),
    indices: splitResult.indices,
    wgs84Bounds,
  }
}
