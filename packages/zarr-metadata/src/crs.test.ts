import { describe, expect, it } from 'vitest'
import {
  extractCrsFromGridMapping,
  extractCrsFromGroupAttributes,
  findGridMapping,
} from './crs'
import {
  cfAlbers,
  cfGeostationary,
  cfLaea,
  cfLambertConformalConic,
  cfLatLonGridMapping,
  cfMercator,
  cfPolarStereographic,
  cfTransverseMercator,
  cfWithCrsWkt,
  v2WithGridMapping,
  v2WithProjCode,
  v3GroupMetadata,
} from './test-fixtures'

describe('extractCrsFromGridMapping', () => {
  it('returns null for latitude_longitude without ellipsoid params', () => {
    const result = extractCrsFromGridMapping(cfLatLonGridMapping)

    expect(result).toBeNull()
  })

  it('builds proj4 string for latitude_longitude with ellipsoid params', () => {
    const result = extractCrsFromGridMapping({
      grid_mapping_name: 'latitude_longitude',
      semi_major_axis: 6378137,
      inverse_flattening: 298.257223563,
    })

    expect(result).not.toBeNull()
    expect(result?.code).toBeNull()
    expect(result?.proj4def).toContain('+proj=longlat')
    expect(result?.proj4def).toContain('+a=6378137')
    expect(result?.proj4def).toContain('+rf=298.257223563')
    expect(result?.source).toBe('grid_mapping')
  })

  it('builds proj4 string for transverse_mercator', () => {
    const result = extractCrsFromGridMapping(cfTransverseMercator)

    expect(result).not.toBeNull()
    expect(result?.proj4def).toContain('+proj=tmerc')
    expect(result?.proj4def).toContain('+lon_0=-93')
    expect(result?.proj4def).toContain('+k=0.9996')
    expect(result?.proj4def).toContain('+x_0=500000')
    expect(result?.source).toBe('grid_mapping')
  })

  it('builds proj4 string for lambert_conformal_conic', () => {
    const result = extractCrsFromGridMapping(cfLambertConformalConic)

    expect(result).not.toBeNull()
    expect(result?.proj4def).toContain('+proj=lcc')
    expect(result?.proj4def).toContain('+lat_0=25')
    expect(result?.proj4def).toContain('+lon_0=-95')
    expect(result?.proj4def).toContain('+lat_1=25')
    expect(result?.proj4def).toContain('+lat_2=25')
  })

  it('passes crs_wkt through as proj4def', () => {
    const result = extractCrsFromGridMapping(cfWithCrsWkt)

    expect(result).not.toBeNull()
    expect(result?.code).toBeNull()
    expect(result?.proj4def).toBe('PROJCS["WGS 84 / UTM zone 32N"...]')
    expect(result?.source).toBe('grid_mapping')
  })

  it('returns null for unknown grid_mapping_name', () => {
    const result = extractCrsFromGridMapping({
      grid_mapping_name: 'unknown_projection',
    })

    expect(result).toBeNull()
  })

  // --- New projection tests ---

  it('builds proj4 string for geostationary (GOES-R ABI)', () => {
    const result = extractCrsFromGridMapping(cfGeostationary)

    expect(result).not.toBeNull()
    expect(result?.proj4def).toContain('+proj=geos')
    expect(result?.proj4def).toContain('+h=35786023')
    expect(result?.proj4def).toContain('+lon_0=-75')
    expect(result?.proj4def).toContain('+sweep=x')
    expect(result?.proj4def).toContain('+a=6378137')
    expect(result?.proj4def).toContain('+rf=298.2572221')
    // coordinateScale should be the satellite height for scanning angle conversion
    expect(result?.coordinateScale).toBe(35786023.0)
  })

  it('builds proj4 string for polar_stereographic', () => {
    const result = extractCrsFromGridMapping(cfPolarStereographic)

    expect(result).not.toBeNull()
    expect(result?.proj4def).toContain('+proj=stere')
    expect(result?.proj4def).toContain('+lat_0=-90')
    expect(result?.proj4def).toContain('+lon_0=0')
    expect(result?.proj4def).toContain('+lat_ts=-71')
    expect(result?.coordinateScale).toBeNull()
  })

  it('builds proj4 string for polar_stereographic with scale_factor variant', () => {
    const result = extractCrsFromGridMapping({
      grid_mapping_name: 'polar_stereographic',
      latitude_of_projection_origin: 90,
      straight_vertical_longitude_from_pole: -45,
      scale_factor_at_projection_origin: 0.994,
    })

    expect(result?.proj4def).toContain('+proj=stere')
    expect(result?.proj4def).toContain('+lat_0=90')
    expect(result?.proj4def).toContain('+lon_0=-45')
    expect(result?.proj4def).toContain('+k=0.994')
    // Should NOT contain lat_ts since scale_factor was used
    expect(result?.proj4def).not.toContain('+lat_ts')
  })

  it('builds proj4 string for mercator', () => {
    const result = extractCrsFromGridMapping(cfMercator)

    expect(result).not.toBeNull()
    expect(result?.proj4def).toContain('+proj=merc')
    expect(result?.proj4def).toContain('+lon_0=0')
    expect(result?.proj4def).toContain('+lat_ts=0')
  })

  it('builds proj4 string for albers_conical_equal_area', () => {
    const result = extractCrsFromGridMapping(cfAlbers)

    expect(result).not.toBeNull()
    expect(result?.proj4def).toContain('+proj=aea')
    expect(result?.proj4def).toContain('+lat_0=23')
    expect(result?.proj4def).toContain('+lon_0=-96')
    expect(result?.proj4def).toContain('+lat_1=29.5')
    expect(result?.proj4def).toContain('+lat_2=45.5')
  })

  it('builds proj4 string for lambert_azimuthal_equal_area', () => {
    const result = extractCrsFromGridMapping(cfLaea)

    expect(result).not.toBeNull()
    expect(result?.proj4def).toContain('+proj=laea')
    expect(result?.proj4def).toContain('+lat_0=52')
    expect(result?.proj4def).toContain('+lon_0=10')
    expect(result?.proj4def).toContain('+x_0=4321000')
    expect(result?.proj4def).toContain('+y_0=3210000')
  })

  it('builds proj4 string for sinusoidal', () => {
    const result = extractCrsFromGridMapping({
      grid_mapping_name: 'sinusoidal',
      longitude_of_central_meridian: 0,
      false_easting: 0,
      false_northing: 0,
    })

    expect(result?.proj4def).toContain('+proj=sinu')
    expect(result?.proj4def).toContain('+lon_0=0')
  })

  it('builds proj4 string for orthographic', () => {
    const result = extractCrsFromGridMapping({
      grid_mapping_name: 'orthographic',
      latitude_of_projection_origin: 45,
      longitude_of_projection_origin: -100,
    })

    expect(result?.proj4def).toContain('+proj=ortho')
    expect(result?.proj4def).toContain('+lat_0=45')
    expect(result?.proj4def).toContain('+lon_0=-100')
  })

  it('builds proj4 string for rotated_latitude_longitude', () => {
    const result = extractCrsFromGridMapping({
      grid_mapping_name: 'rotated_latitude_longitude',
      grid_north_pole_latitude: 39.25,
      grid_north_pole_longitude: -162,
      north_pole_grid_longitude: 0,
    })

    expect(result?.proj4def).toContain('+proj=ob_tran')
    expect(result?.proj4def).toContain('+o_proj=longlat')
    expect(result?.proj4def).toContain('+o_lat_p=39.25')
    expect(result?.proj4def).toContain('+o_lon_p=-162')
    expect(result?.proj4def).toContain('+lon_0=0')
  })

  it('geostationary without perspective_point_height has null coordinateScale', () => {
    const result = extractCrsFromGridMapping({
      grid_mapping_name: 'geostationary',
      longitude_of_projection_origin: -75,
    })

    expect(result?.proj4def).toContain('+proj=geos')
    expect(result?.coordinateScale).toBeNull()
  })
})

describe('extractCrsFromGroupAttributes', () => {
  it('extracts proj:code from V3 group attributes', () => {
    const result = extractCrsFromGroupAttributes(v3GroupMetadata)

    expect(result).not.toBeNull()
    expect(result?.code).toBe('EPSG:4326')
    expect(result?.source).toBe('explicit')
  })

  it('extracts proj:code from V2 root .zattrs', () => {
    const result = extractCrsFromGroupAttributes(v2WithProjCode)

    expect(result).not.toBeNull()
    expect(result?.code).toBe('EPSG:32632')
  })

  it('extracts proj:wkt2 from V3 group attributes', () => {
    const wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984"]]'
    const result = extractCrsFromGroupAttributes({
      zarr_format: 3,
      node_type: 'group',
      attributes: { 'proj:wkt2': wkt },
    })

    expect(result).not.toBeNull()
    expect(result?.proj4def).toBe(wkt)
    expect(result?.code).toBeNull()
    expect(result?.source).toBe('explicit')
  })

  it('prefers proj:wkt2 over proj:code when both present', () => {
    const wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984"]]'
    const result = extractCrsFromGroupAttributes({
      zarr_format: 3,
      node_type: 'group',
      attributes: { 'proj:wkt2': wkt, 'proj:code': 'EPSG:4326' },
    })

    expect(result?.proj4def).toBe(wkt)
    expect(result?.code).toBe('EPSG:4326')
  })

  it('extracts proj:wkt2 from V2 root .zattrs', () => {
    const wkt = 'PROJCS["UTM zone 32N"]'
    const result = extractCrsFromGroupAttributes({
      metadata: { '.zattrs': { 'proj:wkt2': wkt } },
    })

    expect(result?.proj4def).toBe(wkt)
  })

  it('returns null when proj:code is not present', () => {
    const result = extractCrsFromGroupAttributes({
      zarr_format: 3,
      node_type: 'group',
      attributes: {},
    })

    expect(result).toBeNull()
  })

  it('returns null for null metadata', () => {
    const result = extractCrsFromGroupAttributes(null)

    expect(result).toBeNull()
  })
})

describe('findGridMapping', () => {
  it('finds grid_mapping from V2 consolidated metadata', () => {
    const arrayAttrs = { grid_mapping: 'crs' }
    const result = findGridMapping(arrayAttrs, v2WithGridMapping)

    expect(result).not.toBeNull()
    expect(result?.proj4def).toContain('+proj=tmerc')
  })

  it('returns null when grid_mapping attribute is missing', () => {
    const result = findGridMapping({}, v2WithGridMapping)

    expect(result).toBeNull()
  })

  it('returns null when grid_mapping variable not found', () => {
    const result = findGridMapping(
      { grid_mapping: 'nonexistent' },
      v2WithGridMapping
    )

    expect(result).toBeNull()
  })

  it('returns null for null metadata', () => {
    const result = findGridMapping({ grid_mapping: 'crs' }, null)

    expect(result).toBeNull()
  })
})
