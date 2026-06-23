export const DEFAULT_TILE_SIZE = 128
export const MAX_CACHED_TILES = 64
export const TILE_SUBDIVISIONS = 32
export const MERCATOR_LAT_LIMIT = 85.05112878

/** Default maximum error threshold for adaptive mesh refinement (in pixels) */
export const DEFAULT_MESH_MAX_ERROR = 0.125

/** Default maximum error for query polygon edge densification (in pixels) */
export const DEFAULT_QUERY_DENSIFY_MAX_ERROR = DEFAULT_MESH_MAX_ERROR

/**
 * Minimum subdivisions per axis for region geometry tessellation (globe
 * projection). 8 (not 2) so a thin strip chunk — e.g. a single data row
 * spanning many degrees — still gets enough vertices across its short axis to
 * keep hard data/nodata edges from smearing on the globe.
 */
export const MIN_SUBDIVISIONS = 8

/** Maximum subdivisions per axis for region geometry tessellation (globe projection) */
export const MAX_SUBDIVISIONS = 128

/** Web Mercator world extent in meters (half of full world width) */
export const WEB_MERCATOR_EXTENT = 20037508.342789244

/** Common names for spatial dimensions. These are matched case-insensitively. */
export const SPATIAL_DIMENSION_ALIASES: Record<'lat' | 'lon', string[]> = {
  lat: ['lat', 'latitude', 'y'],
  lon: ['lon', 'longitude', 'x', 'lng'],
}

/** Flat set of all spatial dimension names */
export const SPATIAL_DIM_NAMES = new Set([
  ...SPATIAL_DIMENSION_ALIASES.lat,
  ...SPATIAL_DIMENSION_ALIASES.lon,
])
