/**
 * @module geozarr
 *
 * Readers for the zarr-conventions geospatial attribute sets: `proj:` names the
 * CRS, `spatial:` places the grid within it.
 *
 * Pure and I/O-free — every function takes already-fetched attribute objects.
 * Malformed values are warned about and dropped rather than thrown; this runs
 * during layer init, where one bad attribute must not take the layer down.
 *
 * @see https://github.com/zarr-conventions/geo-proj
 * @see https://github.com/zarr-conventions/spatial
 * @see https://github.com/zarr-conventions/multiscales
 */

import type { Bounds } from './types'

type Attrs = Record<string, unknown> | undefined

/** `spatial:transform` coefficients [a, b, c, d, e, f]. */
export type AffineTransform = [number, number, number, number, number, number]

export type Registration = 'pixel' | 'node'

export interface GeoZarrCrs {
  /** `proj:code`, e.g. "EPSG:4326". */
  code?: string
  /** `proj:wkt2`, a WKT2 (ISO 19162) string. */
  wkt2?: string
  /** `proj:projjson`, a PROJJSON object. */
  projjson?: Record<string, unknown>
}

export interface GeoZarrAttrs {
  /** Present when at least one of the three `proj:` forms was found. */
  crs?: GeoZarrCrs
  /** `spatial:bbox` as [xMin, yMin, xMax, yMax]. */
  bbox?: Bounds
  /**
   * `spatial:transform`, mapping index to coordinate as
   * `x = a*col + b*row + c` and `y = d*col + e*row + f`. `a` and `e` are the
   * axis resolutions (`e` negative for north-up), `b` and `d` the rotations,
   * `c` and `f` the westernmost/northernmost coordinate.
   */
  transform?: AffineTransform
  /** `spatial:shape` as [height, width]. */
  shape?: [number, number]
  /** `spatial:dimensions`, ordered by axis role as [yName, xName]. */
  dimensions?: [string, string]
  registration: Registration
  transformType: string
}

/** The `spatial:` fields a multiscales layout entry may carry per level. */
export type LayoutItemSpatial = Pick<GeoZarrAttrs, 'transform' | 'shape'>

const warn = (message: string): void => {
  console.warn(`[zarr-layer] ${message}`)
}

/**
 * The nearest source declaring `key` wins, sources being ordered outermost
 * first. `spatial:` is overridden a property at a time, so a level that
 * restates only its own transform keeps the rest of what it inherits.
 */
const pick = (sources: Attrs[], key: string): unknown => {
  for (let i = sources.length - 1; i >= 0; i--) {
    const value = sources[i]?.[key]
    if (value !== undefined) return value
  }
  return undefined
}

function asString(value: unknown, key: string): string | undefined {
  if (value === undefined || value === null) return undefined
  if (typeof value === 'string' && value.trim()) return value.trim()
  warn(`Ignoring '${key}': expected a non-empty string.`)
  return undefined
}

function asObject(
  value: unknown,
  key: string
): Record<string, unknown> | undefined {
  if (value === undefined || value === null) return undefined
  if (typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }
  warn(`Ignoring '${key}': expected an object.`)
  return undefined
}

function asNumbers(
  value: unknown,
  key: string,
  length: number
): number[] | undefined {
  if (value === undefined || value === null) return undefined
  if (
    !Array.isArray(value) ||
    value.length !== length ||
    !value.every((v) => typeof v === 'number' && Number.isFinite(v))
  ) {
    warn(`Ignoring '${key}': expected an array of ${length} finite numbers.`)
    return undefined
  }
  return value as number[]
}

function asBbox(value: unknown): Bounds | undefined {
  const nums = asNumbers(value, 'spatial:bbox', 4)
  if (!nums) return undefined
  const [xMin, yMin, xMax, yMax] = nums
  if (xMax <= xMin || yMax <= yMin) {
    warn(`Ignoring 'spatial:bbox': expected [xMin, yMin, xMax, yMax].`)
    return undefined
  }
  return [xMin, yMin, xMax, yMax]
}

function asTransform(value: unknown): AffineTransform | undefined {
  const nums = asNumbers(value, 'spatial:transform', 6)
  if (!nums) return undefined
  const [a, b, c, d, e, f] = nums
  // An axis with zero resolution and zero rotation maps every index to one
  // coordinate. A zero resolution alone is a quarter-turn rotation, kept
  // intact so the rotation handling sees it.
  if ((a === 0 && b === 0) || (d === 0 && e === 0)) {
    warn(`Ignoring 'spatial:transform': it collapses an axis.`)
    return undefined
  }
  return [a, b, c, d, e, f]
}

function asShape(value: unknown): [number, number] | undefined {
  const nums = asNumbers(value, 'spatial:shape', 2)
  if (!nums) return undefined
  if (!nums.every((n) => Number.isInteger(n) && n > 0)) {
    warn(`Ignoring 'spatial:shape': expected two positive integers.`)
    return undefined
  }
  return [nums[0], nums[1]]
}

function asDimensions(value: unknown): [string, string] | undefined {
  if (value === undefined || value === null) return undefined
  if (
    !Array.isArray(value) ||
    value.length !== 2 ||
    !value.every((v) => typeof v === 'string' && v.trim())
  ) {
    warn(
      `Ignoring 'spatial:dimensions': expected two dimension names ordered [y, x].`
    )
    return undefined
  }
  return [value[0].trim(), value[1].trim()]
}

function asRegistration(value: unknown): Registration {
  if (value === undefined || value === null) return 'pixel'
  if (value === 'pixel' || value === 'node') return value
  warn(
    `Ignoring 'spatial:registration' value ${JSON.stringify(value)}: ` +
      `expected 'pixel' or 'node'. Using 'pixel'.`
  )
  return 'pixel'
}

const PROJ_KEYS = ['proj:code', 'proj:wkt2', 'proj:projjson'] as const

/**
 * Unlike `spatial:`, a `proj:` declaration is taken whole from the nearest
 * source that makes one. The three forms describe a single CRS, so blending a
 * level's `proj:code` with an ancestor's `proj:wkt2` could silently compose two
 * different coordinate systems.
 */
function parseCrs(sources: Attrs[]): GeoZarrCrs | undefined {
  for (let i = sources.length - 1; i >= 0; i--) {
    const source = sources[i]
    if (!source || !PROJ_KEYS.some((k) => source[k] !== undefined)) continue
    const code = asString(source['proj:code'], 'proj:code')
    const wkt2 = asString(source['proj:wkt2'], 'proj:wkt2')
    const projjson = asObject(source['proj:projjson'], 'proj:projjson')
    if (!code && !wkt2 && !projjson) return undefined
    return { code, wkt2, projjson }
  }
  return undefined
}

/**
 * Read the `proj:` and `spatial:` attributes that apply to one array.
 *
 * Sources are given outermost first -- typically the store root, then the
 * level's own group, then the array -- and the nearest declaration wins. The
 * `proj:` convention inherits to a group's direct child arrays only, so a
 * pyramid may legitimately restate it on each level group rather than at the
 * root; passing the whole chain covers either placement.
 *
 * `registration` and `transformType` always come back populated, defaulting to
 * the spec's `'pixel'` and `'affine'`. Everything else is absent when nothing
 * declares it or what is declared is malformed.
 */
export function parseGeoZarrAttrs(...sources: Attrs[]): GeoZarrAttrs {
  const spatial = (key: string) => pick(sources, key)
  return {
    crs: parseCrs(sources),
    bbox: asBbox(spatial('spatial:bbox')),
    transform: asTransform(spatial('spatial:transform')),
    shape: asShape(spatial('spatial:shape')),
    dimensions: asDimensions(spatial('spatial:dimensions')),
    registration: asRegistration(spatial('spatial:registration')),
    transformType:
      asString(spatial('spatial:transform_type'), 'spatial:transform_type') ??
      'affine',
  }
}

export interface SpatialExtent {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
  /** `null` when nothing declared settles which edge row 0 sits on. */
  latIsAscending: boolean | null
}

/**
 * Derive the grid's outer edges from what the `spatial:` convention declares.
 *
 * The result is edge-to-edge, matching what the renderer means by bounds, which
 * is why `spatial:registration` is load-bearing: under `'pixel'` the declared
 * coordinates already sit on cell boundaries, under `'node'` they sit on cell
 * centers and everything moves out by half a cell.
 *
 * `spatial:bbox` wins the extent when both are declared, since it states the
 * extent directly; `spatial:transform` still supplies the cell size and the row
 * direction. Returns `null` when nothing declared pins the extent down: a
 * rotated transform with no bbox, or a node-registered bbox on a grid too small
 * to recover a cell size from.
 */
export function boundsFromSpatialAttrs(
  attrs: GeoZarrAttrs,
  grid: { nCols: number; nRows: number }
): SpatialExtent | null {
  if (attrs.transformType !== 'affine') return null

  // Rotated grids have no axis-aligned placement at all. The renderer maps rows
  // and columns linearly across the extent with no rotation term, so a bbox --
  // which merely encloses the rotated footprint -- would render the raster
  // unrotated and stretched to fill it. Decline rather than misplace it.
  if (
    attrs.transform &&
    (attrs.transform[1] !== 0 || attrs.transform[3] !== 0)
  ) {
    return null
  }

  const transform = attrs.transform
  const latIsAscending = transform ? transform[4] > 0 : null

  if (attrs.bbox) {
    const [xMin, yMin, xMax, yMax] = attrs.bbox
    if (attrs.registration === 'pixel') {
      return { xMin, yMin, xMax, yMax, latIsAscending }
    }

    // Node registration puts the bbox on the centers of the border cells, so
    // half a cell is missing from each side. The transform gives the cell size
    // outright; otherwise the bbox spans nCols - 1 whole cells.
    const halfCell = (
      resolution: number | undefined,
      span: number,
      count: number
    ): number | null => {
      if (resolution !== undefined) return Math.abs(resolution) / 2
      return count > 1 ? span / (count - 1) / 2 : null
    }
    const halfX = halfCell(transform?.[0], xMax - xMin, grid.nCols)
    const halfY = halfCell(transform?.[4], yMax - yMin, grid.nRows)
    if (halfX === null || halfY === null) return null

    return {
      xMin: xMin - halfX,
      xMax: xMax + halfX,
      yMin: yMin - halfY,
      yMax: yMax + halfY,
      latIsAscending,
    }
  }

  if (!transform) return null

  const [a, , c, , e, f] = transform
  // Under node registration the transform lands on the first cell's center.
  const originX = attrs.registration === 'node' ? c - a / 2 : c
  const originY = attrs.registration === 'node' ? f - e / 2 : f
  const endX = originX + a * grid.nCols
  const endY = originY + e * grid.nRows

  return {
    xMin: Math.min(originX, endX),
    xMax: Math.max(originX, endX),
    yMin: Math.min(originY, endY),
    yMax: Math.max(originY, endY),
    latIsAscending,
  }
}

/**
 * Read the per-level `spatial:` attributes from one `multiscales.layout` entry.
 *
 * These sit alongside `asset` on the entry itself, not inside its `transform`
 * object, which holds only the relative scale/translation between levels.
 */
export function parseLayoutItemSpatial(item: unknown): LayoutItemSpatial {
  const attrs =
    item && typeof item === 'object' && !Array.isArray(item)
      ? (item as Record<string, unknown>)
      : undefined
  return {
    transform: asTransform(attrs?.['spatial:transform']),
    shape: asShape(attrs?.['spatial:shape']),
  }
}
