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

  it('normalizes a lowercase proj:code before looking it up', async () => {
    // proj4's registry is keyed on the uppercase form, and the built-in codes
    // already accept either case.
    const d = await describeGeoStore({
      arrayAttrs: { 'proj:code': 'epsg:32631' },
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

/**
 * A 4-row by 8-column global grid on a 45 deg cell. Coordinates are cell
 * centers ordered north-first, so the coordinate-array path yields an exactly
 * global extent with row 0 at the north. The equivalent declaration is
 * `spatial:transform` [45, 0, -180, 0, -45, 90] under pixel registration.
 */
const LAT_CENTERS = [67.5, 22.5, -22.5, -67.5]
const LON_CENTERS = [-157.5, -112.5, -67.5, -22.5, 22.5, 67.5, 112.5, 157.5]
const GLOBAL_TRANSFORM = [45, 0, -180, 0, -45, 90]
const GLOBAL_LIMITS = { xMin: -180, xMax: 180, yMin: -90, yMax: 90 }

function spatialStore(arrayAttrs: Record<string, unknown>): MemoryStore {
  return buildMemoryZarrStore({
    arrays: [
      {
        name: 'temperature',
        shape: [4, 8],
        chunkShape: [4, 8],
        dimensionNames: ['lat', 'lon'],
        attributes: arrayAttrs,
      },
      {
        name: 'lat',
        shape: [4],
        chunkShape: [4],
        dimensionNames: ['lat'],
        chunks: { '0': LAT_CENTERS },
      },
      {
        name: 'lon',
        shape: [8],
        chunkShape: [8],
        dimensionNames: ['lon'],
        chunks: { '0': LON_CENTERS },
      },
    ],
  })
}

/** Wraps a store so every key it is asked for can be inspected afterwards. */
function recordReads(inner: MemoryStore) {
  const keys: string[] = []
  return {
    keys,
    store: {
      get: async (key: string) => {
        keys.push(key)
        return inner.get(key)
      },
    },
  }
}

async function describeSpatialStore(
  arrayAttrs: Record<string, unknown>,
  options: { bounds?: [number, number, number, number] } = {}
) {
  const { keys, store: recorded } = recordReads(spatialStore(arrayAttrs))
  const store = new ZarrStore({
    customStore: recorded,
    variable: 'temperature',
    version: 3,
    ...options,
  })
  await store.initialized
  return { d: store.describe(), keys }
}

describe('bounds from the spatial: convention', () => {
  it('derives the extent and row direction from a transform', async () => {
    const { d } = await describeSpatialStore({
      'spatial:transform': GLOBAL_TRANSFORM,
    })

    expect(d.xyLimits).toEqual(GLOBAL_LIMITS)
    expect(d.latIsAscending).toBe(false)
  })

  it('never touches the coordinate arrays when a transform settles both', async () => {
    const { keys } = await describeSpatialStore({
      'spatial:transform': GLOBAL_TRANSFORM,
    })

    expect(
      keys.filter((k) => k.startsWith('/lat') || k.startsWith('/lon'))
    ).toEqual([])
  })

  it('lands exactly where the coordinate arrays would have', async () => {
    // Half a cell of disagreement here would shift the raster silently.
    const declared = await describeSpatialStore({
      'spatial:transform': GLOBAL_TRANSFORM,
    })
    const read = await describeSpatialStore({})

    expect(declared.d.xyLimits).toEqual(read.d.xyLimits)
    expect(declared.d.latIsAscending).toBe(read.d.latIsAscending)
  })

  it('reads a bbox as the extent but still checks orientation', async () => {
    const { d, keys } = await describeSpatialStore({
      'spatial:bbox': [-180, -90, 180, 90],
    })

    expect(d.xyLimits).toEqual(GLOBAL_LIMITS)
    // A bbox says nothing about which edge row 0 sits on, so the coordinate
    // read still has to happen. Setting `latIsAscending` skips it.
    expect(keys.some((k) => k.startsWith('/lat'))).toBe(true)
  })

  it('folds a 0-360 bbox into the range the renderer works in', async () => {
    const { d } = await describeSpatialStore({
      'spatial:bbox': [200, -90, 340, 90],
      'spatial:transform': GLOBAL_TRANSFORM,
    })

    expect(d.xyLimits?.xMin).toBe(-160)
    expect(d.xyLimits?.xMax).toBe(-20)
  })

  it('snaps a global bbox that misses the antimeridian by a hair', async () => {
    const { d } = await describeSpatialStore({
      'spatial:bbox': [-179.9999, -90, 180.0001, 90],
      'spatial:transform': GLOBAL_TRANSFORM,
    })

    expect(d.xyLimits?.xMin).toBe(-180)
    expect(d.xyLimits?.xMax).toBe(180)
  })

  it('shifts a node-registered grid out by half a cell', async () => {
    // A 1 deg regional grid, well clear of the global snap below.
    const { d } = await describeSpatialStore({
      'spatial:transform': [1, 0, -100, 0, -1, 40],
      'spatial:registration': 'node',
    })

    expect(d.xyLimits).toEqual({
      xMin: -100.5,
      xMax: -92.5,
      yMin: 36.5,
      yMax: 40.5,
    })
  })

  it('snaps a node-registered global grid onto the antimeridian', async () => {
    // Node registration puts this grid's edges at -202.5..157.5, a full 360 deg
    // of coverage offset by half a cell. The global snap pulls it onto +/-180
    // rather than leaving a seam. Its tolerance is a whole cell, so the half
    // cell here is well inside it.
    const { d } = await describeSpatialStore({
      'spatial:transform': GLOBAL_TRANSFORM,
      'spatial:registration': 'node',
    })

    expect(d.xyLimits?.xMin).toBe(-180)
    expect(d.xyLimits?.xMax).toBe(180)
  })

  it('lets the bounds option win over a declared bbox', async () => {
    const { d } = await describeSpatialStore(
      { 'spatial:bbox': [-180, -90, 180, 90] },
      { bounds: [-10, -10, 10, 10] }
    )

    expect(d.xyLimits).toEqual({ xMin: -10, xMax: 10, yMin: -10, yMax: 10 })
  })

  it('falls back to the coordinate arrays for a transform type it cannot map', async () => {
    const warn = silenceWarnings()
    const { d, keys } = await describeSpatialStore({
      'spatial:transform': GLOBAL_TRANSFORM,
      'spatial:transform_type': 'rpc',
    })

    expect(d.xyLimits).toEqual(GLOBAL_LIMITS)
    expect(keys.some((k) => k.startsWith('/lat'))).toBe(true)
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('rpc'))
  })

  it('falls back to the coordinate arrays for a rotated transform', async () => {
    const warn = silenceWarnings()
    const { d, keys } = await describeSpatialStore({
      'spatial:transform': [45, 1, -180, 1, -45, 90],
    })

    expect(d.xyLimits).toEqual(GLOBAL_LIMITS)
    expect(keys.some((k) => k.startsWith('/lat'))).toBe(true)
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('rotated'))
  })
})

/**
 * A grid whose axes are named something the alias list has never heard of, so
 * only `spatial:dimensions` can identify them.
 */
function namedAxesStore(arrayAttrs: Record<string, unknown>): MemoryStore {
  return buildMemoryZarrStore({
    arrays: [
      {
        name: 'temperature',
        shape: [2, 4, 8],
        chunkShape: [2, 4, 8],
        dimensionNames: ['band', 'northing', 'easting'],
        attributes: arrayAttrs,
      },
    ],
  })
}

async function describeNamedAxes(
  arrayAttrs: Record<string, unknown>,
  spatialDimensions?: { lat?: string; lon?: string }
) {
  const store = new ZarrStore({
    customStore: namedAxesStore(arrayAttrs),
    variable: 'temperature',
    version: 3,
    bounds: [-100, 30, -92, 40],
    latIsAscending: false,
    spatialDimensions,
  })
  await store.initialized
  return store.describe()
}

describe('axis identification from spatial:dimensions', () => {
  it('identifies axes the alias list cannot', async () => {
    const d = await describeNamedAxes({
      'spatial:dimensions': ['northing', 'easting'],
    })

    expect(d.dimIndices.lat?.name).toBe('northing')
    expect(d.dimIndices.lat?.index).toBe(1)
    expect(d.dimIndices.lon?.name).toBe('easting')
    expect(d.dimIndices.lon?.index).toBe(2)
  })

  it('leaves the alias heuristic in charge when nothing is declared', async () => {
    const d = await describeNamedAxes({})

    expect(d.dimIndices.lat).toBeUndefined()
    expect(d.dimIndices.lon).toBeUndefined()
  })

  it('reads the declared order as [y, x]', async () => {
    // Declared the other way round, the axes swap with it.
    const d = await describeNamedAxes({
      'spatial:dimensions': ['easting', 'northing'],
    })

    expect(d.dimIndices.lat?.name).toBe('easting')
    expect(d.dimIndices.lon?.name).toBe('northing')
  })

  // Pointing an axis at the band dimension is nonsense geographically; it is
  // picked precisely because the store would never resolve that axis there.
  it.each([
    ['lat' as const, 'lon' as const, 'easting'],
    ['lon' as const, 'lat' as const, 'northing'],
  ])(
    'lets the spatialDimensions option win for %s alone',
    async (overridden, untouched, stillDeclared) => {
      const d = await describeNamedAxes(
        { 'spatial:dimensions': ['northing', 'easting'] },
        { [overridden]: 'band' }
      )

      expect(d.dimIndices[overridden]?.name).toBe('band')
      // The axis the caller left alone still comes from the store.
      expect(d.dimIndices[untouched]?.name).toBe(stillDeclared)
    }
  )

  it('lets an override repair an axis the store declared wrongly', async () => {
    const d = await describeNamedAxes(
      { 'spatial:dimensions': ['bogus', 'easting'] },
      { lat: 'northing' }
    )

    expect(d.dimIndices.lat?.name).toBe('northing')
    expect(d.dimIndices.lon?.name).toBe('easting')
  })

  it('rejects a declaration naming dimensions the array does not have', async () => {
    await expect(
      describeNamedAxes({ 'spatial:dimensions': ['y', 'x'] })
    ).rejects.toThrow(/spatial:dimensions names \[y, x\]/)
  })
})

/**
 * A two-level pyramid carrying a time dimension, so a level's shape cannot just
 * be the declared [height, width] pair — the non-spatial dimension has to
 * survive and the spatial ones have to land in the array's own axis order.
 */
function levelShapeStore(
  layout: unknown[],
  rootAttrs: Record<string, unknown> = {}
): MemoryStore {
  return buildMemoryZarrStore({
    attributes: { multiscales: { layout, crs: 'EPSG:4326' }, ...rootAttrs },
    arrays: [0, 1].map((i) => ({
      name: `${i}/temperature`,
      shape: [2, 512 >> i, 1024 >> i],
      chunkShape: [2, 512 >> i, 1024 >> i],
      dimensionNames: ['time', 'lat', 'lon'],
    })),
  })
}

/**
 * `bounds` defaults to the dataset extent for layouts that declare only
 * shapes. Cases exercising per-level extents pass `null`, since explicit
 * bounds deliberately suppress them.
 */
async function describeLevelsWithStore(
  layout: unknown[],
  {
    rootAttrs = {},
    bounds = [-180, -90, 180, 90] as [number, number, number, number] | null,
  }: {
    rootAttrs?: Record<string, unknown>
    bounds?: [number, number, number, number] | null
  } = {}
) {
  const store = new ZarrStore({
    customStore: levelShapeStore(layout, rootAttrs),
    variable: 'temperature',
    version: 3,
    ...(bounds ? { bounds } : {}),
    latIsAscending: false,
  })
  await store.initialized
  return { d: store.describe(), store }
}

async function describeLevels(
  layout: unknown[],
  {
    rootAttrs = {},
    bounds = [-180, -90, 180, 90] as [number, number, number, number] | null,
  }: {
    rootAttrs?: Record<string, unknown>
    bounds?: [number, number, number, number] | null
  } = {}
) {
  const { keys, store: recorded } = recordReads(
    levelShapeStore(layout, rootAttrs)
  )
  const store = new ZarrStore({
    customStore: recorded,
    variable: 'temperature',
    version: 3,
    ...(bounds ? { bounds } : {}),
    latIsAscending: false,
  })
  await store.initialized
  return { d: store.describe(), keys }
}

describe('level shapes from spatial:shape', () => {
  it('substitutes the declared pair into the array axis order', async () => {
    const { d } = await describeLevels([
      { asset: '0', 'spatial:shape': [512, 1024] },
      { asset: '1', 'spatial:shape': [256, 512] },
    ])

    // The time dimension is carried over from the base shape.
    expect(d.untiledLevels).toEqual([
      { asset: '0', shape: [2, 512, 1024] },
      { asset: '1', shape: [2, 256, 512] },
    ])
  })

  it('matches what opening the level array would have reported', async () => {
    const { d } = await describeLevels([
      { asset: '0', 'spatial:shape': [512, 1024] },
      { asset: '1', 'spatial:shape': [256, 512] },
    ])
    const declared = d.untiledLevels[1].shape

    const opened = await new ZarrStore({
      customStore: levelShapeStore([{ asset: '0' }, { asset: '1' }]),
      variable: 'temperature',
      version: 3,
      bounds: [-180, -90, 180, 90],
      latIsAscending: false,
    }).initialized.then((s) => s.getUntiledLevelMetadata('1'))

    expect(declared).toEqual(opened.shape)
  })

  it('never opens the deeper level to size the pyramid', async () => {
    const { keys } = await describeLevels([
      { asset: '0', 'spatial:shape': [512, 1024] },
      { asset: '1', 'spatial:shape': [256, 512] },
    ])

    expect(keys).not.toContain('/1/temperature/zarr.json')
  })

  it('leaves a level that declares nothing for the renderer to fetch', async () => {
    const { d } = await describeLevels([
      { asset: '0', 'spatial:shape': [512, 1024] },
      { asset: '1' },
    ])

    expect(d.untiledLevels[0].shape).toEqual([2, 512, 1024])
    expect(d.untiledLevels[1].shape).toBeUndefined()
  })
})

describe('georeferencing declared on the layout entries', () => {
  /**
   * A pyramid whose absolute placement lives only on its `multiscales.layout`
   * entries, which is where the multiscales convention puts it. Nothing on the
   * group or the array says where the grid sits.
   */
  const layoutGeoStore = (
    layout: unknown[],
    groupAttrs: Record<string, unknown> = {}
  ) =>
    buildMemoryZarrStore({
      attributes: { multiscales: { layout }, ...groupAttrs },
      arrays: [0, 1].map((i) => ({
        name: `${i}/temperature`,
        shape: [4 >> i, 8 >> i],
        chunkShape: [4 >> i, 8 >> i],
        dimensionNames: ['lat', 'lon'],
      })),
    })

  const describeLayoutGeo = async (
    layout: unknown[],
    groupAttrs: Record<string, unknown> = {}
  ) => {
    const { keys, store: recorded } = recordReads(
      layoutGeoStore(layout, groupAttrs)
    )
    const store = new ZarrStore({
      customStore: recorded,
      variable: 'temperature',
      version: 3,
    })
    await store.initialized
    return { d: store.describe(), keys }
  }

  it('places the grid from the base level transform', async () => {
    const { d } = await describeLayoutGeo([
      {
        asset: '0',
        'spatial:transform': GLOBAL_TRANSFORM,
        'spatial:shape': [4, 8],
      },
      {
        asset: '1',
        'spatial:transform': [90, 0, -180, 0, -90, 90],
        'spatial:shape': [2, 4],
      },
    ])

    expect(d.xyLimits).toEqual(GLOBAL_LIMITS)
    expect(d.latIsAscending).toBe(false)
  })

  it('needs no coordinate arrays to do it', async () => {
    const { keys } = await describeLayoutGeo([
      {
        asset: '0',
        'spatial:transform': GLOBAL_TRANSFORM,
        'spatial:shape': [4, 8],
      },
      { asset: '1', 'spatial:shape': [2, 4] },
    ])

    expect(keys.filter((k) => k.includes('lat') || k.includes('lon'))).toEqual(
      []
    )
  })

  it('takes the extent from a group bbox and the row direction from the entry', async () => {
    // How EOPF Sentinel-2 declares itself: a bbox on the group, the transform
    // only on the layout entries. Neither settles the grid alone.
    const { d, keys } = await describeLayoutGeo(
      [{ asset: '0', 'spatial:transform': GLOBAL_TRANSFORM }, { asset: '1' }],
      { 'spatial:bbox': [-180, -90, 180, 90] }
    )

    expect(d.xyLimits).toEqual(GLOBAL_LIMITS)
    expect(d.latIsAscending).toBe(false)
    expect(keys.filter((k) => k.includes('lat') || k.includes('lon'))).toEqual(
      []
    )
  })

  it('lets a group-level declaration win over the layout entry', async () => {
    const store = new ZarrStore({
      customStore: buildMemoryZarrStore({
        attributes: {
          multiscales: {
            layout: [
              { asset: '0', 'spatial:transform': GLOBAL_TRANSFORM },
              { asset: '1' },
            ],
          },
          'spatial:bbox': [-10, -10, 10, 10],
        },
        arrays: [0, 1].map((i) => ({
          name: `${i}/temperature`,
          shape: [4 >> i, 8 >> i],
          chunkShape: [4 >> i, 8 >> i],
          dimensionNames: ['lat', 'lon'],
        })),
      }),
      variable: 'temperature',
      version: 3,
      latIsAscending: false,
    })
    await store.initialized

    expect(store.describe().xyLimits).toEqual({
      xMin: -10,
      xMax: 10,
      yMin: -10,
      yMax: 10,
    })
  })
})

/**
 * `proj:` inherits only to a group's direct child arrays, so a pyramid whose
 * levels are groups may restate it on each level group rather than at the
 * store root. Both placements are sanctioned by the convention.
 */
describe('a declaration on the level group', () => {
  const levelGroupStore = (
    rootAttrs: Record<string, unknown>,
    levelAttrs: Record<string, unknown>
  ) => {
    const spec = buildMemoryZarrStore({
      attributes: { multiscales: { layout: [{ asset: '0' }] }, ...rootAttrs },
      arrays: [
        {
          name: '0/temperature',
          shape: [4, 8],
          chunkShape: [4, 8],
          dimensionNames: ['lat', 'lon'],
        },
      ],
    })
    // Give the level group its own attributes.
    const enc = new TextEncoder()
    const inner = spec.get.bind(spec)
    return {
      get: async (key: string) => {
        if (key === '/0/zarr.json') {
          return enc.encode(
            JSON.stringify({
              zarr_format: 3,
              node_type: 'group',
              attributes: levelAttrs,
            })
          )
        }
        return inner(key)
      },
    }
  }

  const describeLevelGroup = async (
    rootAttrs: Record<string, unknown>,
    levelAttrs: Record<string, unknown>
  ) => {
    const store = new ZarrStore({
      customStore: levelGroupStore(rootAttrs, levelAttrs),
      variable: 'temperature',
      version: 3,
      latIsAscending: false,
    })
    await store.initialized
    return store.describe()
  }

  it('is read when the root declares nothing', async () => {
    const d = await describeLevelGroup(
      {},
      {
        'proj:code': 'EPSG:32631',
        'spatial:transform': [45, 0, -180, 0, -45, 90],
      }
    )

    expect(d.proj4).toBe('EPSG:32631')
    expect(d.xyLimits).toEqual(GLOBAL_LIMITS)
  })

  it('wins over the root, being nearer the array', async () => {
    const d = await describeLevelGroup(
      { 'proj:code': 'EPSG:4326', 'spatial:bbox': [-180, -90, 180, 90] },
      { 'proj:code': 'EPSG:32631' }
    )

    expect(d.proj4).toBe('EPSG:32631')
  })

  it('leaves the root in charge of what it does not restate', async () => {
    const d = await describeLevelGroup(
      {
        'proj:code': 'EPSG:4326',
        'spatial:bbox': [-180, -90, 180, 90],
        'spatial:registration': 'pixel',
      },
      { 'spatial:transform': [45, 0, -180, 0, -45, 90] }
    )

    expect(d.crs).toBe('EPSG:4326')
    expect(d.xyLimits).toEqual(GLOBAL_LIMITS)
    expect(d.latIsAscending).toBe(false)
  })
})

describe('per-level extents from the layout entries', () => {
  it('gives each level the extent its own transform describes', async () => {
    // Level 1 is floor-divided: 4 columns of 90 covers 360, but 2 columns of
    // 180 starting at -180 stops at 180 -- while a 3-column level would not.
    const { d } = await describeLevels(
      [
        {
          asset: '0',
          'spatial:transform': GLOBAL_TRANSFORM,
          'spatial:shape': [4, 8],
        },
        {
          asset: '1',
          'spatial:transform': [90, 0, -180, 0, -90, 90],
          'spatial:shape': [2, 3],
        },
      ],
      { bounds: null }
    )

    expect(d.untiledLevels[0].xyLimits).toEqual(GLOBAL_LIMITS)
    // 3 columns x 90 = 270 wide, not the dataset's 360.
    expect(d.untiledLevels[1].xyLimits).toEqual({
      xMin: -180,
      xMax: 90,
      yMin: -90,
      yMax: 90,
    })
  })

  it('declines a level whose row direction contradicts the dataset', async () => {
    // The renderer draws every level in the dataset's row direction. A level
    // declaring the opposite cannot be honored, so its extent is dropped with
    // a warning rather than placed as if its rows ran the other way.
    const warn = silenceWarnings()
    const { d } = await describeLevels(
      [
        {
          asset: '0',
          'spatial:transform': GLOBAL_TRANSFORM,
          'spatial:shape': [4, 8],
        },
        {
          asset: '1',
          'spatial:transform': [90, 0, -180, 0, 90, -90],
          'spatial:shape': [2, 4],
        },
      ],
      { bounds: null }
    )

    expect(d.untiledLevels[0].xyLimits).toEqual(GLOBAL_LIMITS)
    expect(d.untiledLevels[1].xyLimits).toBeUndefined()
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('row direction'))
  })

  it('keeps extents of a consistent store under a latIsAscending override', async () => {
    // The harness overrides latIsAscending to false against south-up
    // transforms. The override changes how rows are drawn, not what the
    // store's transforms declare about placement, so extents survive.
    const warn = silenceWarnings()
    const { d } = await describeLevels(
      [
        {
          asset: '0',
          'spatial:transform': [45, 0, -180, 0, 45, -90],
          'spatial:shape': [4, 8],
        },
        {
          asset: '1',
          'spatial:transform': [90, 0, -180, 0, 90, -90],
          'spatial:shape': [2, 3],
        },
      ],
      { bounds: null }
    )

    expect(d.latIsAscending).toBe(false)
    expect(d.untiledLevels[0].xyLimits).toEqual(GLOBAL_LIMITS)
    expect(d.untiledLevels[1].xyLimits).toEqual({
      xMin: -180,
      xMax: 90,
      yMin: -90,
      yMax: 90,
    })
    expect(warn).not.toHaveBeenCalledWith(
      expect.stringContaining('row direction')
    )
  })

  it('leaves a level with no declared transform without one', async () => {
    const { d } = await describeLevels(
      [
        {
          asset: '0',
          'spatial:transform': GLOBAL_TRANSFORM,
          'spatial:shape': [4, 8],
        },
        { asset: '1', 'spatial:shape': [2, 4] },
      ],
      { bounds: null }
    )

    expect(d.untiledLevels[0].xyLimits).toEqual(GLOBAL_LIMITS)
    expect(d.untiledLevels[1].xyLimits).toBeUndefined()
  })
})

describe('a level extent competing with a dataset bbox', () => {
  it("uses the level's own transform, not the dataset bbox", async () => {
    // The bbox describes the base level. Applying it to level 1 would stretch
    // that level's 270 units of coverage across the full 360.
    const { d } = await describeLevels(
      [
        {
          asset: '0',
          'spatial:transform': GLOBAL_TRANSFORM,
          'spatial:shape': [4, 8],
        },
        {
          asset: '1',
          'spatial:transform': [90, 0, -180, 0, -90, 90],
          'spatial:shape': [2, 3],
        },
      ],
      { rootAttrs: { 'spatial:bbox': [-180, -90, 180, 90] }, bounds: null }
    )

    expect(d.untiledLevels[1].xyLimits).toEqual({
      xMin: -180,
      xMax: 90,
      yMin: -90,
      yMax: 90,
    })
  })
})

describe('explicit bounds against per-level extents', () => {
  it('suppresses per-level extents when the caller supplies bounds', async () => {
    // `bounds` overrides the store's georeferencing. Re-deriving level extents
    // from the same metadata would quietly reinstate what was overridden.
    const { keys, store: recorded } = recordReads(
      levelShapeStore([
        {
          asset: '0',
          'spatial:transform': GLOBAL_TRANSFORM,
          'spatial:shape': [4, 8],
        },
        {
          asset: '1',
          'spatial:transform': [90, 0, -180, 0, -90, 90],
          'spatial:shape': [2, 3],
        },
      ])
    )
    const store = new ZarrStore({
      customStore: recorded,
      variable: 'temperature',
      version: 3,
      bounds: [-10, -10, 10, 10],
      latIsAscending: false,
    })
    await store.initialized
    const d = store.describe()

    expect(d.xyLimits).toEqual({ xMin: -10, xMax: 10, yMin: -10, yMax: 10 })
    expect(d.untiledLevels.every((l) => l.xyLimits === undefined)).toBe(true)
    expect(keys.length).toBeGreaterThan(0)
  })
})

describe('a level transform without a declared shape', () => {
  it('derives the extent once the real shape is loaded', async () => {
    // The fixture's arrays are 512x1024 and 256x512, so a global extent means
    // cells of 360/1024 and 360/512 degrees.
    const { d, store } = await describeLevelsWithStore(
      [
        {
          asset: '0',
          'spatial:transform': [0.3515625, 0, -180, 0, -0.3515625, 90],
          'spatial:shape': [512, 1024],
        },
        // Transform but no spatial:shape: the extent cannot be worked out
        // until the array itself is opened.
        {
          asset: '1',
          'spatial:transform': [0.703125, 0, -180, 0, -0.703125, 90],
        },
      ],
      { bounds: null }
    )

    expect(d.untiledLevels[1].xyLimits).toBeUndefined()

    await store.getUntiledLevelMetadata('1')

    expect(d.untiledLevels[1].xyLimits).toEqual({
      xMin: -180,
      xMax: 180,
      yMin: -90,
      yMax: 90,
    })
  })
})
