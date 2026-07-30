import { describe, it, expect, vi, afterEach } from 'vitest'
import proj4 from 'proj4'
import { ZarrStore } from './zarr-store'
import {
  buildMemoryZarrStore,
  type MemoryStore,
} from './__fixtures__/memory-zarr'

/**
 * Integration coverage for the zarr-conventions attributes, driven through the
 * real `ZarrStore._initialize()` against an in-memory v3 fixture.
 *
 * Bounds are supplied explicitly on the CRS cases so nothing depends on
 * coordinate arrays, isolating what `proj:` alone settles.
 */

// EPSG:5070, which proj4 has no built-in definition for — the store has to
// carry the full description for it to resolve.
const ALBERS_WKT2 =
  'PROJCRS["NAD83 / Conus Albers",BASEGEOGCRS["NAD83",DATUM["North American Datum 1983",ELLIPSOID["GRS 1980",6378137,298.257222101,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]]],CONVERSION["Conus Albers",METHOD["Albers Equal Area",ID["EPSG",9822]],PARAMETER["Latitude of false origin",23,ANGLEUNIT["degree",0.0174532925199433]],PARAMETER["Longitude of false origin",-96,ANGLEUNIT["degree",0.0174532925199433]],PARAMETER["Latitude of 1st standard parallel",29.5,ANGLEUNIT["degree",0.0174532925199433]],PARAMETER["Latitude of 2nd standard parallel",45.5,ANGLEUNIT["degree",0.0174532925199433]],PARAMETER["Easting at false origin",0,LENGTHUNIT["metre",1]],PARAMETER["Northing at false origin",0,LENGTHUNIT["metre",1]]],CS[Cartesian,2],AXIS["easting (X)",east,ORDER[1],LENGTHUNIT["metre",1]],AXIS["northing (Y)",north,ORDER[2],LENGTHUNIT["metre",1]],ID["EPSG",5070]]'

const ALBERS_PROJ4 =
  '+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +datum=NAD83 +units=m'

const UTM31_PROJJSON = {
  type: 'ProjectedCRS',
  name: 'WGS 84 / UTM zone 31N',
  base_crs: {
    name: 'WGS 84',
    datum: {
      type: 'GeodeticReferenceFrame',
      name: 'World Geodetic System 1984',
      ellipsoid: {
        name: 'WGS 84',
        semi_major_axis: 6378137,
        inverse_flattening: 298.257223563,
      },
    },
    coordinate_system: {
      subtype: 'ellipsoidal',
      axis: [
        { name: 'Geodetic latitude', direction: 'north', unit: 'degree' },
        { name: 'Geodetic longitude', direction: 'east', unit: 'degree' },
      ],
    },
  },
  conversion: {
    name: 'UTM zone 31N',
    method: {
      name: 'Transverse Mercator',
      id: { authority: 'EPSG', code: 9807 },
    },
    parameters: [
      { name: 'Latitude of natural origin', value: 0, unit: 'degree' },
      { name: 'Longitude of natural origin', value: 3, unit: 'degree' },
      { name: 'Scale factor at natural origin', value: 0.9996, unit: 'unity' },
      { name: 'False easting', value: 500000, unit: 'metre' },
      { name: 'False northing', value: 0, unit: 'metre' },
    ],
  },
  coordinate_system: {
    subtype: 'Cartesian',
    axis: [
      { name: 'Easting', direction: 'east', unit: 'metre' },
      { name: 'Northing', direction: 'north', unit: 'metre' },
    ],
  },
  id: { authority: 'EPSG', code: 32631 },
}

const METRIC_BOUNDS: [number, number, number, number] = [-2e6, 1e6, 2e6, 3e6]

interface GeoStoreSpec {
  groupAttrs?: Record<string, unknown>
  arrayAttrs?: Record<string, unknown>
  /** Attributes of a CF grid-mapping variable named `spatial_ref`. */
  gridMappingAttrs?: Record<string, unknown>
}

function geoStore({
  groupAttrs,
  arrayAttrs,
  gridMappingAttrs,
}: GeoStoreSpec): MemoryStore {
  return buildMemoryZarrStore({
    attributes: groupAttrs ?? {},
    arrays: [
      {
        name: 'temperature',
        shape: [4, 8],
        chunkShape: [4, 8],
        dimensionNames: ['y', 'x'],
        attributes: arrayAttrs ?? {},
      },
      {
        name: 'y',
        shape: [4],
        chunkShape: [4],
        dimensionNames: ['y'],
        chunks: { '0': [2.75e6, 2.25e6, 1.75e6, 1.25e6] },
      },
      {
        name: 'x',
        shape: [8],
        chunkShape: [8],
        dimensionNames: ['x'],
        chunks: {
          '0': [-1.75e6, -1.25e6, -7.5e5, -2.5e5, 2.5e5, 7.5e5, 1.25e6, 1.75e6],
        },
      },
      ...(gridMappingAttrs
        ? [
            {
              name: 'spatial_ref',
              shape: [1],
              chunkShape: [1],
              attributes: gridMappingAttrs,
            },
          ]
        : []),
    ],
  })
}

/**
 * `latIsAscending: false` keeps the CRS cases off the coordinate-read path
 * entirely. The cases that exercise bounds-based CRS inference pass `null`
 * instead, since that inference only runs once the coordinate path has been
 * walked.
 */
async function describeGeoStore(
  spec: GeoStoreSpec,
  options: {
    crs?: string
    proj4?: string
    latIsAscending?: boolean | null
  } = {}
) {
  const store = new ZarrStore({
    customStore: geoStore(spec),
    variable: 'temperature',
    version: 3,
    bounds: METRIC_BOUNDS,
    latIsAscending: false,
    ...options,
  })
  await store.initialized
  return store.describe()
}

const silenceWarnings = () =>
  vi.spyOn(console, 'warn').mockImplementation(() => {})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('CRS from the proj: convention', () => {
  it.each([
    ['EPSG:4326', 'EPSG:4326'],
    ['EPSG:3857', 'EPSG:3857'],
    ['epsg:4326', 'EPSG:4326'],
    // WGS84 lon/lat under its OGC identity, same axis order the renderer assumes.
    ['OGC:CRS84', 'EPSG:4326'],
  ])('keeps %s on the native path', async (code, expected) => {
    const d = await describeGeoStore({ arrayAttrs: { 'proj:code': code } })

    expect(d.crs).toBe(expected)
    expect(d.proj4).toBeNull()
  })

  it('resolves a proj:code proj4 already ships a definition for', async () => {
    const d = await describeGeoStore({
      arrayAttrs: { 'proj:code': 'EPSG:32631' },
    })

    expect(d.proj4).toBe('EPSG:32631')
  })

  it('registers proj:wkt2 and reprojects through it', async () => {
    const d = await describeGeoStore({
      arrayAttrs: { 'proj:wkt2': ALBERS_WKT2 },
    })

    expect(d.proj4).toBeTruthy()
    const [lon, lat] = proj4(d.proj4!, 'EPSG:4326').forward([0, 0])
    const [refLon, refLat] = proj4(ALBERS_PROJ4, 'EPSG:4326').forward([0, 0])
    expect(lon).toBeCloseTo(refLon, 9)
    expect(lat).toBeCloseTo(refLat, 9)
  })

  it('registers proj:projjson and reprojects through it', async () => {
    const d = await describeGeoStore({
      arrayAttrs: { 'proj:projjson': UTM31_PROJJSON },
    })

    expect(d.proj4).toBeTruthy()
    // The false easting of UTM zone 31N sits on its central meridian, 3°E.
    const [lon] = proj4(d.proj4!, 'EPSG:4326').forward([500000, 0])
    expect(lon).toBeCloseTo(3, 9)
  })

  it('gives each store its own key so two layers cannot collide', async () => {
    const [first, second] = await Promise.all([
      describeGeoStore({ arrayAttrs: { 'proj:wkt2': ALBERS_WKT2 } }),
      describeGeoStore({ arrayAttrs: { 'proj:projjson': UTM31_PROJJSON } }),
    ])

    expect(first.proj4).not.toBe(second.proj4)
  })

  it('prefers a built-in proj:code over a wkt2 describing the same CRS', async () => {
    const d = await describeGeoStore({
      arrayAttrs: { 'proj:code': 'EPSG:4326', 'proj:wkt2': ALBERS_WKT2 },
    })

    expect(d.crs).toBe('EPSG:4326')
    expect(d.proj4).toBeNull()
  })

  it('lets array attributes override the group they inherit from', async () => {
    const d = await describeGeoStore({
      groupAttrs: { 'proj:code': 'EPSG:4326' },
      arrayAttrs: { 'proj:code': 'EPSG:3857' },
    })

    expect(d.crs).toBe('EPSG:3857')
  })

  it('reads a CF grid mapping when no proj: attribute is declared', async () => {
    const d = await describeGeoStore({
      arrayAttrs: { grid_mapping: 'spatial_ref' },
      gridMappingAttrs: { crs_wkt: ALBERS_WKT2 },
    })

    expect(d.proj4).toBeTruthy()
    const [lon, lat] = proj4(d.proj4!, 'EPSG:4326').forward([0, 0])
    const [refLon, refLat] = proj4(ALBERS_PROJ4, 'EPSG:4326').forward([0, 0])
    expect(lon).toBeCloseTo(refLon, 9)
    expect(lat).toBeCloseTo(refLat, 9)
  })

  it('reads the spatial_ref attribute rioxarray writes alongside crs_wkt', async () => {
    const d = await describeGeoStore({
      arrayAttrs: { grid_mapping: 'spatial_ref' },
      gridMappingAttrs: { spatial_ref: ALBERS_WKT2 },
    })

    expect(d.proj4).toBeTruthy()
  })
})

describe('CRS resolution precedence', () => {
  it('lets the crs option win over the store attributes', async () => {
    const d = await describeGeoStore(
      { arrayAttrs: { 'proj:wkt2': ALBERS_WKT2 } },
      { crs: 'EPSG:3857' }
    )

    expect(d.crs).toBe('EPSG:3857')
    expect(d.proj4).toBeNull()
  })

  it('lets the proj4 option win over the store attributes', async () => {
    const d = await describeGeoStore(
      { arrayAttrs: { 'proj:wkt2': ALBERS_WKT2 } },
      { proj4: ALBERS_PROJ4 }
    )

    expect(d.proj4).toBe(ALBERS_PROJ4)
  })
})

describe('an unresolvable proj:code', () => {
  it('warns naming the code and the proj4.defs remedy', async () => {
    const warn = silenceWarnings()
    await describeGeoStore({ arrayAttrs: { 'proj:code': 'EPSG:5070' } })

    const message = warn.mock.calls.map((c) => String(c[0])).join('\n')
    expect(message).toContain('EPSG:5070')
    expect(message).toContain('proj4.defs')
  })

  it('does not fall back to guessing a CRS from the bounds', async () => {
    silenceWarnings()
    // Metric bounds otherwise infer as EPSG:3857, as the next case shows. The
    // store said EPSG:5070, so contradicting it is worse than leaving it
    // unresolved.
    const d = await describeGeoStore(
      { arrayAttrs: { 'proj:code': 'EPSG:5070' } },
      { latIsAscending: null }
    )

    expect(d.crs).toBe('EPSG:4326')
    expect(d.proj4).toBeNull()
  })

  it('still infers from bounds when nothing declares a CRS', async () => {
    const d = await describeGeoStore({}, { latIsAscending: null })

    expect(d.crs).toBe('EPSG:3857')
  })
})
