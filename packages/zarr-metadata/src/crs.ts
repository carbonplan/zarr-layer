/**
 * @module crs
 *
 * CRS (Coordinate Reference System) extraction utilities.
 * Extracts CRS from explicit metadata, CF grid_mapping attributes,
 * and crs_wkt strings.
 *
 * Supports all 15 CF grid_mapping_name projections that have proj4js equivalents:
 * @see http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#appendix-grid-mappings
 * @see https://cfconventions.org/wkt-proj-4.html
 */

import type {
  CFGridMappingAttributes,
  CRSInfo,
  ZarrV2Attributes,
  ZarrV2ConsolidatedMetadata,
  ZarrV3GroupMetadata,
} from './types'

/** Internal result from buildProj4FromCF */
interface CFConversionResult {
  proj4def: string
  coordinateScale: number | null
}

/**
 * Extract CRS from zarr-conventions multiscale metadata.
 */
export function extractCrsFromZarrConventions(multiscales: {
  crs?: string
}): CRSInfo | null {
  if (!multiscales.crs) return null

  return {
    code: multiscales.crs.toUpperCase(),
    proj4def: null,
    source: 'explicit',
  }
}

/**
 * Extract CRS from OME-NGFF dataset metadata.
 * OME-NGFF doesn't have a standard CRS field, but some implementations use custom attributes.
 */
export function extractCrsFromOmeNgff(
  datasets: Array<{ crs?: string }>
): CRSInfo | null {
  if (!datasets[0]?.crs) return null

  return {
    code: datasets[0].crs.toUpperCase(),
    proj4def: null,
    source: 'explicit',
  }
}

/**
 * Extract CRS from CF grid_mapping variable.
 *
 * Handles three cases:
 * 1. `crs_wkt` present — passed through as proj4def (proj4js can parse WKT directly)
 * 2. `grid_mapping_name` present — converted to proj4 string
 * 3. Neither — returns null
 *
 * @see http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#appendix-grid-mappings
 */
export function extractCrsFromGridMapping(
  attrs: CFGridMappingAttributes
): CRSInfo | null {
  // Check for CRS WKT first — proj4js can parse WKT directly
  if (attrs.crs_wkt) {
    return {
      code: null,
      proj4def: attrs.crs_wkt,
      source: 'grid_mapping',
    }
  }

  // Try to build proj4 from CF parameters
  const result = buildProj4FromCF(attrs)
  if (result) {
    return {
      code: null,
      proj4def: result.proj4def,
      coordinateScale: result.coordinateScale,
      source: 'grid_mapping',
    }
  }

  return null
}

/**
 * Build a proj4 string from CF grid_mapping parameters.
 *
 * Supports all CF grid_mapping_name values that have proj4js equivalents:
 * - latitude_longitude → longlat
 * - transverse_mercator → tmerc
 * - lambert_conformal_conic → lcc
 * - geostationary → geos (returns coordinateScale for scanning angle conversion)
 * - polar_stereographic / stereographic → stere
 * - mercator → merc
 * - albers_conical_equal_area → aea
 * - lambert_azimuthal_equal_area → laea
 * - azimuthal_equidistant → aeqd
 * - sinusoidal → sinu
 * - orthographic → ortho
 * - lambert_cylindrical_equal_area → cea
 * - oblique_mercator → omerc
 * - vertical_perspective → nsper
 * - rotated_latitude_longitude → ob_tran
 *
 * @see https://cfconventions.org/wkt-proj-4.html
 */
function buildProj4FromCF(
  attrs: CFGridMappingAttributes
): CFConversionResult | null {
  const name = attrs.grid_mapping_name?.toLowerCase()
  if (!name) return null

  const ellipsoidParts = buildEllipsoidParams(attrs)

  // CF uses both longitude_of_projection_origin and longitude_of_central_meridian
  // depending on the projection. Both map to proj4 +lon_0.
  const lon0 =
    attrs.longitude_of_projection_origin ?? attrs.longitude_of_central_meridian

  function result(
    proj4def: string,
    coordinateScale?: number | null
  ): CFConversionResult {
    return { proj4def, coordinateScale: coordinateScale ?? null }
  }

  function pushCommon(parts: string[]): void {
    if (attrs.false_easting !== undefined)
      parts.push(`+x_0=${attrs.false_easting}`)
    if (attrs.false_northing !== undefined)
      parts.push(`+y_0=${attrs.false_northing}`)
    parts.push(ellipsoidParts ?? '+datum=WGS84')
    parts.push('+units=m +no_defs')
  }

  function pushStandardParallels(parts: string[]): void {
    const stdPar = attrs.standard_parallel
    if (Array.isArray(stdPar) && stdPar.length >= 2) {
      parts.push(`+lat_1=${stdPar[0]}`)
      parts.push(`+lat_2=${stdPar[1]}`)
    } else if (typeof stdPar === 'number') {
      parts.push(`+lat_1=${stdPar}`)
      parts.push(`+lat_2=${stdPar}`)
    }
  }

  switch (name) {
    case 'latitude_longitude': {
      // Only return proj4 if we have ellipsoid info, otherwise return null
      // and let consumers decide (we can't assume WGS84)
      if (ellipsoidParts) {
        return result(`+proj=longlat ${ellipsoidParts} +no_defs`)
      }
      return null
    }

    case 'transverse_mercator': {
      const parts = ['+proj=tmerc']
      if (attrs.latitude_of_projection_origin !== undefined) {
        parts.push(`+lat_0=${attrs.latitude_of_projection_origin}`)
      }
      if (attrs.longitude_of_central_meridian !== undefined) {
        parts.push(`+lon_0=${attrs.longitude_of_central_meridian}`)
      }
      if (attrs.scale_factor_at_central_meridian !== undefined) {
        parts.push(`+k=${attrs.scale_factor_at_central_meridian}`)
      }
      pushCommon(parts)
      return result(parts.join(' '))
    }

    case 'lambert_conformal_conic': {
      const parts = ['+proj=lcc']
      if (attrs.latitude_of_projection_origin !== undefined) {
        parts.push(`+lat_0=${attrs.latitude_of_projection_origin}`)
      }
      if (attrs.longitude_of_central_meridian !== undefined) {
        parts.push(`+lon_0=${attrs.longitude_of_central_meridian}`)
      }
      pushStandardParallels(parts)
      pushCommon(parts)
      return result(parts.join(' '))
    }

    case 'geostationary': {
      const parts = ['+proj=geos']
      const h = attrs.perspective_point_height
      if (h !== undefined) parts.push(`+h=${h}`)
      if (lon0 !== undefined) parts.push(`+lon_0=${lon0}`)
      if (attrs.sweep_angle_axis) parts.push(`+sweep=${attrs.sweep_angle_axis}`)
      pushCommon(parts)
      // Return perspective_point_height as coordinateScale.
      // GOES-R ABI stores x/y as scanning angles in radians,
      // but proj4 +proj=geos expects meters.
      // Consumers should multiply coordinate bounds by this value.
      return result(parts.join(' '), h ?? null)
    }

    case 'polar_stereographic':
    case 'stereographic': {
      const parts = ['+proj=stere']
      if (attrs.latitude_of_projection_origin !== undefined) {
        parts.push(`+lat_0=${attrs.latitude_of_projection_origin}`)
      }
      // CF uses straight_vertical_longitude_from_pole for polar stereo
      const straightLon = attrs.straight_vertical_longitude_from_pole
      if (straightLon !== undefined) {
        parts.push(`+lon_0=${straightLon}`)
      } else if (lon0 !== undefined) {
        parts.push(`+lon_0=${lon0}`)
      }
      // Two variants: standard_parallel (lat_ts) or scale_factor
      const stdPar = attrs.standard_parallel
      if (stdPar !== undefined) {
        const latTs = Array.isArray(stdPar) ? stdPar[0] : stdPar
        parts.push(`+lat_ts=${latTs}`)
      } else if (attrs.scale_factor_at_projection_origin !== undefined) {
        parts.push(`+k=${attrs.scale_factor_at_projection_origin}`)
      }
      pushCommon(parts)
      return result(parts.join(' '))
    }

    case 'mercator': {
      const parts = ['+proj=merc']
      if (lon0 !== undefined) parts.push(`+lon_0=${lon0}`)
      const stdPar = attrs.standard_parallel
      if (stdPar !== undefined) {
        const latTs = Array.isArray(stdPar) ? stdPar[0] : stdPar
        parts.push(`+lat_ts=${latTs}`)
      } else if (attrs.scale_factor_at_projection_origin !== undefined) {
        parts.push(`+k=${attrs.scale_factor_at_projection_origin}`)
      }
      pushCommon(parts)
      return result(parts.join(' '))
    }

    case 'albers_conical_equal_area': {
      const parts = ['+proj=aea']
      if (attrs.latitude_of_projection_origin !== undefined) {
        parts.push(`+lat_0=${attrs.latitude_of_projection_origin}`)
      }
      if (lon0 !== undefined) parts.push(`+lon_0=${lon0}`)
      pushStandardParallels(parts)
      pushCommon(parts)
      return result(parts.join(' '))
    }

    case 'lambert_azimuthal_equal_area': {
      const parts = ['+proj=laea']
      if (attrs.latitude_of_projection_origin !== undefined) {
        parts.push(`+lat_0=${attrs.latitude_of_projection_origin}`)
      }
      if (lon0 !== undefined) parts.push(`+lon_0=${lon0}`)
      pushCommon(parts)
      return result(parts.join(' '))
    }

    case 'azimuthal_equidistant': {
      const parts = ['+proj=aeqd']
      if (attrs.latitude_of_projection_origin !== undefined) {
        parts.push(`+lat_0=${attrs.latitude_of_projection_origin}`)
      }
      if (lon0 !== undefined) parts.push(`+lon_0=${lon0}`)
      pushCommon(parts)
      return result(parts.join(' '))
    }

    case 'sinusoidal': {
      const parts = ['+proj=sinu']
      if (lon0 !== undefined) parts.push(`+lon_0=${lon0}`)
      pushCommon(parts)
      return result(parts.join(' '))
    }

    case 'orthographic': {
      const parts = ['+proj=ortho']
      if (attrs.latitude_of_projection_origin !== undefined) {
        parts.push(`+lat_0=${attrs.latitude_of_projection_origin}`)
      }
      if (lon0 !== undefined) parts.push(`+lon_0=${lon0}`)
      pushCommon(parts)
      return result(parts.join(' '))
    }

    case 'lambert_cylindrical_equal_area': {
      const parts = ['+proj=cea']
      if (lon0 !== undefined) parts.push(`+lon_0=${lon0}`)
      const stdPar = attrs.standard_parallel
      if (stdPar !== undefined) {
        const latTs = Array.isArray(stdPar) ? stdPar[0] : stdPar
        parts.push(`+lat_ts=${latTs}`)
      }
      pushCommon(parts)
      return result(parts.join(' '))
    }

    case 'oblique_mercator': {
      const parts = ['+proj=omerc']
      if (attrs.latitude_of_projection_origin !== undefined) {
        parts.push(`+lat_0=${attrs.latitude_of_projection_origin}`)
      }
      if (lon0 !== undefined) parts.push(`+lonc=${lon0}`)
      if (attrs.azimuth_of_central_line !== undefined) {
        parts.push(`+alpha=${attrs.azimuth_of_central_line}`)
      }
      if (attrs.scale_factor_at_projection_origin !== undefined) {
        parts.push(`+k=${attrs.scale_factor_at_projection_origin}`)
      }
      pushCommon(parts)
      return result(parts.join(' '))
    }

    case 'vertical_perspective': {
      const parts = ['+proj=nsper']
      const h = attrs.perspective_point_height
      if (h !== undefined) parts.push(`+h=${h}`)
      if (attrs.latitude_of_projection_origin !== undefined) {
        parts.push(`+lat_0=${attrs.latitude_of_projection_origin}`)
      }
      if (lon0 !== undefined) parts.push(`+lon_0=${lon0}`)
      pushCommon(parts)
      return result(parts.join(' '))
    }

    case 'rotated_latitude_longitude': {
      const parts = ['+proj=ob_tran', '+o_proj=longlat']
      if (attrs.grid_north_pole_latitude !== undefined) {
        parts.push(`+o_lat_p=${attrs.grid_north_pole_latitude}`)
      }
      if (attrs.grid_north_pole_longitude !== undefined) {
        parts.push(`+o_lon_p=${attrs.grid_north_pole_longitude}`)
      }
      if (attrs.north_pole_grid_longitude !== undefined) {
        parts.push(`+lon_0=${attrs.north_pole_grid_longitude}`)
      }
      parts.push(ellipsoidParts ?? '+datum=WGS84')
      parts.push('+no_defs')
      return result(parts.join(' '))
    }

    default:
      return null
  }
}

/**
 * Build proj4 ellipsoid parameters from CF attributes.
 * Returns null if no ellipsoid info is provided.
 *
 * @see http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#ellipsoid
 */
function buildEllipsoidParams(attrs: CFGridMappingAttributes): string | null {
  const parts: string[] = []

  if (attrs.semi_major_axis !== undefined) {
    parts.push(`+a=${attrs.semi_major_axis}`)
  }

  if (attrs.inverse_flattening !== undefined) {
    parts.push(`+rf=${attrs.inverse_flattening}`)
  } else if (attrs.semi_minor_axis !== undefined) {
    parts.push(`+b=${attrs.semi_minor_axis}`)
  }

  if (attrs.longitude_of_prime_meridian !== undefined) {
    parts.push(`+pm=${attrs.longitude_of_prime_meridian}`)
  }

  return parts.length > 0 ? parts.join(' ') : null
}

/**
 * Attempt to find and parse CF grid_mapping attributes from metadata.
 */
export function findGridMapping(
  arrayAttrs: ZarrV2Attributes | Record<string, unknown> | undefined,
  metadata: ZarrV2ConsolidatedMetadata | ZarrV3GroupMetadata | null
): CRSInfo | null {
  if (!arrayAttrs || !metadata) return null

  // Get the grid_mapping variable name
  const gridMappingName = arrayAttrs.grid_mapping as string | undefined
  if (!gridMappingName) return null

  // Try to find the grid_mapping variable in metadata
  let gridMappingAttrs: CFGridMappingAttributes | null = null

  // V2: look in .zattrs
  const v2Meta = metadata as ZarrV2ConsolidatedMetadata
  if (v2Meta.metadata) {
    const attrsKey = `${gridMappingName}/.zattrs`
    if (v2Meta.metadata[attrsKey]) {
      gridMappingAttrs = v2Meta.metadata[attrsKey] as CFGridMappingAttributes
    }
  }

  // V3: look in consolidated_metadata
  const v3Meta = metadata as ZarrV3GroupMetadata
  if (v3Meta.consolidated_metadata?.metadata) {
    const arrayMeta = v3Meta.consolidated_metadata.metadata[gridMappingName]
    if (arrayMeta?.attributes) {
      gridMappingAttrs = arrayMeta.attributes as CFGridMappingAttributes
    }
  }

  if (gridMappingAttrs) {
    return extractCrsFromGridMapping(gridMappingAttrs)
  }

  return null
}

/**
 * Create CRSInfo from an explicit user-provided CRS.
 */
export function createExplicitCrs(crs: string, proj4def?: string): CRSInfo {
  return {
    code: crs.toUpperCase(),
    proj4def: proj4def ?? null,
    source: 'explicit',
  }
}

/**
 * Extract CRS from group-level attributes.
 *
 * Supports the GeoZarr/zarr-conventions geo-proj convention:
 * - `proj:wkt2`: WKT2 string (highest fidelity, passed as proj4def)
 * - `proj:code`: Authority:code string (e.g., "EPSG:4326", "EPSG:32632")
 *
 * @see https://github.com/zarr-conventions/geo-proj
 * @see https://github.com/zarr-conventions/multiscales
 */
export function extractCrsFromGroupAttributes(
  metadata: ZarrV2ConsolidatedMetadata | ZarrV3GroupMetadata | null
): CRSInfo | null {
  if (!metadata) return null

  // V3: check attributes on group
  const v3Meta = metadata as ZarrV3GroupMetadata
  if (v3Meta.attributes) {
    const result = extractCrsFromProjAttributes(v3Meta.attributes)
    if (result) return result
  }

  // V2: check root .zattrs
  const v2Meta = metadata as ZarrV2ConsolidatedMetadata
  if (v2Meta.metadata) {
    const rootAttrs = v2Meta.metadata['.zattrs'] as
      | Record<string, unknown>
      | undefined
    if (rootAttrs) {
      const result = extractCrsFromProjAttributes(rootAttrs)
      if (result) return result
    }
  }

  return null
}

/**
 * Extract CRS from proj: attributes (GeoZarr geo-proj convention).
 * Priority: proj:wkt2 > proj:code (wkt2 carries full CRS definition).
 */
function extractCrsFromProjAttributes(
  attrs: Record<string, unknown>
): CRSInfo | null {
  // proj:wkt2 — full WKT2 CRS definition, proj4js can parse directly
  const wkt2 = attrs['proj:wkt2'] as string | undefined
  if (wkt2) {
    // Also grab the code if available for labeling
    const code = attrs['proj:code'] as string | undefined
    return {
      code: code?.toUpperCase() ?? null,
      proj4def: wkt2,
      source: 'explicit',
    }
  }

  // proj:code — authority:code string
  const projCode = attrs['proj:code'] as string | undefined
  if (projCode) {
    return {
      code: projCode.toUpperCase(),
      proj4def: null,
      source: 'explicit',
    }
  }

  return null
}
