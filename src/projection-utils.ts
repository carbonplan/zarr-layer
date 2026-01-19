import proj4 from 'proj4'
import type { MercatorBounds } from './map-utils'
import { WEB_MERCATOR_EXTENT } from './constants'
import type { Bounds } from './types'

/**
 * Formats a proj4 error with helpful context.
 */
function formatProj4Error(proj4def: string, err: unknown): string {
  const msg = err instanceof Error ? err.message : String(err)
  return (
    `[zarr-layer] Invalid proj4 string: "${proj4def.slice(0, 50)}${
      proj4def.length > 50 ? '...' : ''
    }". ` +
    `Error: ${msg}. Check your dataset metadata or find CRS definitions at https://epsg.io/`
  )
}

/**
 * A transformer for converting coordinates between source CRS and Web Mercator.
 */
export interface ProjectionTransformer {
  /** Transform from source CRS to Web Mercator [x, y] */
  forward: (x: number, y: number) => [number, number]
  /** Transform from Web Mercator to source CRS [x, y] */
  inverse: (x: number, y: number) => [number, number]
  /** Source projection bounds in source CRS units */
  bounds: Bounds
}

/**
 * Creates a reusable transformer for converting between source CRS and Web Mercator.
 */
export function createTransformer(
  proj4def: string,
  bounds: Bounds
): ProjectionTransformer {
  let converter: proj4.Converter
  try {
    converter = proj4(proj4def, 'EPSG:3857')
  } catch (err) {
    throw new Error(formatProj4Error(proj4def, err))
  }

  return {
    forward: (x: number, y: number) =>
      converter.forward([x, y]) as [number, number],
    inverse: (x: number, y: number) =>
      converter.inverse([x, y]) as [number, number],
    bounds,
  }
}

/**
 * A transformer for converting coordinates between source CRS and EPSG:4326 (WGS84).
 */
export interface Wgs84Transformer {
  /** Transform from source CRS to EPSG:4326 [lon, lat] */
  forward: (x: number, y: number) => [number, number]
  /** Transform from EPSG:4326 to source CRS [x, y] */
  inverse: (lon: number, lat: number) => [number, number]
  /** Source projection bounds in source CRS units */
  bounds: Bounds
}

/**
 * Creates a reusable transformer for converting between source CRS and EPSG:4326.
 * Used for the two-stage reprojection pipeline where Stage 1 targets 4326.
 */
export function createTransformerTo4326(
  proj4def: string,
  bounds: Bounds
): Wgs84Transformer {
  let converter: proj4.Converter
  try {
    converter = proj4(proj4def, 'EPSG:4326')
  } catch (err) {
    throw new Error(formatProj4Error(proj4def, err))
  }

  return {
    forward: (x: number, y: number) =>
      converter.forward([x, y]) as [number, number],
    inverse: (lon: number, lat: number) =>
      converter.inverse([lon, lat]) as [number, number],
    bounds,
  }
}

/**
 * Validates that bounds have positive extent (max > min).
 */
function validateBounds(bounds: Bounds, fnName: string): boolean {
  const [xMin, yMin, xMax, yMax] = bounds
  if (xMax <= xMin || yMax <= yMin) {
    console.warn(
      `[zarr-layer] Invalid bounds in ${fnName}: max must be greater than min`
    )
    return false
  }
  return true
}

/**
 * Converts source CRS coordinates to pixel indices given grid shape and bounds.
 * Bounds are edge-to-edge (xMin = left edge, xMax = right edge).
 * Returns [xPixel, yPixel] as floating-point values for interpolation.
 *
 * Uses edge-based model: xMin → 0, xMax → width (consistent with getRegionBounds).
 * For pixel centers, the result will be at integer + 0.5 positions.
 *
 * @param latIsAscending - If true, row 0 = yMin (south). If false, row 0 = yMax (north).
 */
export function sourceCRSToPixel(
  x: number,
  y: number,
  bounds: Bounds,
  width: number,
  height: number,
  latIsAscending: boolean = true
): [number, number] {
  if (!validateBounds(bounds, 'sourceCRSToPixel')) {
    return [width / 2, height / 2]
  }

  const [xMin, yMin, xMax, yMax] = bounds

  // Map source CRS coords to normalized [0, 1]
  const xNorm = (x - xMin) / (xMax - xMin)
  const yNorm = (y - yMin) / (yMax - yMin)

  // Convert to pixel coordinates using edge-to-edge model
  const xPixel = xNorm * width

  // Y depends on data orientation:
  // - latIsAscending true: row 0 = yMin (south)
  // - latIsAscending false: row 0 = yMax (north)
  const yPixel = latIsAscending ? yNorm * height : (1 - yNorm) * height

  return [xPixel, yPixel]
}

/**
 * Converts pixel position to source CRS coordinates given grid shape and bounds.
 * Bounds are edge-to-edge (xMin = left edge, xMax = right edge).
 *
 * Uses edge-based model: pixel 0 → xMin, pixel width → xMax.
 * For pixel centers, pass pixel + 0.5 (e.g., 0.5 for center of first pixel).
 *
 * @param latIsAscending - If true, row 0 = yMin (south). If false, row 0 = yMax (north).
 */
export function pixelToSourceCRS(
  xPixel: number,
  yPixel: number,
  bounds: Bounds,
  width: number,
  height: number,
  latIsAscending: boolean = true
): [number, number] {
  const [xMin, yMin, xMax, yMax] = bounds

  if (!validateBounds(bounds, 'pixelToSourceCRS')) {
    return [(xMin + xMax) / 2, (yMin + yMax) / 2]
  }

  // Convert pixel to normalized [0, 1] using edge-to-edge model
  const xNorm = width <= 1 ? 0.5 : xPixel / width
  const yNorm = height <= 1 ? 0.5 : yPixel / height

  // Map to source CRS
  const x = xMin + xNorm * (xMax - xMin)

  // Y depends on data orientation:
  // - latIsAscending true: row 0 = yMin (south)
  // - latIsAscending false: row 0 = yMax (north)
  const y = latIsAscending
    ? yMin + yNorm * (yMax - yMin)
    : yMax - yNorm * (yMax - yMin)

  return [x, y]
}

/**
 * Creates a transformer for converting WGS84 lat/lon to source CRS.
 * Useful for query coordinate transforms.
 */
export function createWGS84ToSourceTransformer(proj4def: string): {
  forward: (lon: number, lat: number) => [number, number]
  inverse: (x: number, y: number) => [number, number]
} {
  let converter: proj4.Converter
  try {
    converter = proj4('EPSG:4326', proj4def)
  } catch (err) {
    throw new Error(formatProj4Error(proj4def, err))
  }

  return {
    forward: (lon: number, lat: number) =>
      converter.forward([lon, lat]) as [number, number],
    inverse: (x: number, y: number) =>
      converter.inverse([x, y]) as [number, number],
  }
}

/**
 * Sample edge points of bounds and transform to normalized mercator bounds.
 * Samples along all 4 edges to capture curved extent for non-Mercator projections.
 *
 * @param bounds - Source CRS bounds
 * @param transformer - Transformer with forward(x, y) method to Web Mercator
 * @param numSamples - Number of sample points per edge (more = more accurate for curved projections)
 * @returns Normalized mercator bounds [0,1] or null if no valid samples
 */
export function sampleEdgesToMercatorBounds(
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number },
  transformer: { forward: (x: number, y: number) => [number, number] },
  numSamples: number
): MercatorBounds | null {
  const { xMin, yMin, xMax, yMax } = bounds

  let minMercX = Infinity
  let maxMercX = -Infinity
  let minMercY = Infinity
  let maxMercY = -Infinity

  for (let i = 0; i <= numSamples; i++) {
    const t = i / numSamples
    const edgePoints: [number, number][] = [
      [xMin + t * (xMax - xMin), yMin], // Bottom
      [xMin + t * (xMax - xMin), yMax], // Top
      [xMin, yMin + t * (yMax - yMin)], // Left
      [xMax, yMin + t * (yMax - yMin)], // Right
    ]
    for (const [srcX, srcY] of edgePoints) {
      const [mercX, mercY] = transformer.forward(srcX, srcY)
      if (!isFinite(mercX) || !isFinite(mercY)) continue
      const normX = (mercX + WEB_MERCATOR_EXTENT) / (2 * WEB_MERCATOR_EXTENT)
      const normY = (WEB_MERCATOR_EXTENT - mercY) / (2 * WEB_MERCATOR_EXTENT)
      minMercX = Math.min(minMercX, normX)
      maxMercX = Math.max(maxMercX, normX)
      minMercY = Math.min(minMercY, normY)
      maxMercY = Math.max(maxMercY, normY)
    }
  }

  if (!isFinite(minMercX)) return null
  return { x0: minMercX, y0: minMercY, x1: maxMercX, y1: maxMercY }
}
