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
 * Array attributes override the group's, following `proj:`'s inheritance model:
 * a group's attributes apply to its direct child arrays unless the array
 * declares its own.
 */
const pick = (group: Attrs, array: Attrs, key: string): unknown =>
  array?.[key] !== undefined ? array[key] : group?.[key]

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
  if (a === 0 || e === 0) {
    warn(
      `Ignoring 'spatial:transform': resolution coefficients must be non-zero.`
    )
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

function parseCrs(group: Attrs, array: Attrs): GeoZarrCrs | undefined {
  const code = asString(pick(group, array, 'proj:code'), 'proj:code')
  const wkt2 = asString(pick(group, array, 'proj:wkt2'), 'proj:wkt2')
  const projjson = asObject(
    pick(group, array, 'proj:projjson'),
    'proj:projjson'
  )
  if (!code && !wkt2 && !projjson) return undefined
  return { code, wkt2, projjson }
}

/**
 * Read the `proj:` and `spatial:` attributes a store declares for one array.
 *
 * `registration` and `transformType` always come back populated, defaulting to
 * the spec's `'pixel'` and `'affine'`. Everything else is absent when the store
 * doesn't declare it or declares it malformed.
 */
export function parseGeoZarrAttrs(
  groupAttrs: Attrs,
  arrayAttrs: Attrs
): GeoZarrAttrs {
  const spatial = (key: string) => pick(groupAttrs, arrayAttrs, key)
  return {
    crs: parseCrs(groupAttrs, arrayAttrs),
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
