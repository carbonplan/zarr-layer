import { describe, it, expect, vi, afterEach } from 'vitest'
import {
  parseGeoZarrAttrs,
  parseLayoutItemSpatial,
  boundsFromSpatialAttrs,
  type GeoZarrAttrs,
} from './geozarr'

/**
 * Unit tests for the zarr-conventions attribute readers. Everything here is
 * pure: attribute objects in, parsed values out, with malformed input expected
 * to warn and drop rather than throw.
 */

const silenceWarnings = () =>
  vi.spyOn(console, 'warn').mockImplementation(() => {})

afterEach(() => {
  vi.restoreAllMocks()
})

// A north-up 10 m grid at the origin of a UTM-style projected CRS.
const TRANSFORM = [10, 0, 500000, 0, -10, 5000000]

describe('parseGeoZarrAttrs', () => {
  it('defaults registration and transform type when nothing is declared', () => {
    const attrs = parseGeoZarrAttrs(undefined, undefined)

    expect(attrs.registration).toBe('pixel')
    expect(attrs.transformType).toBe('affine')
    expect(attrs.crs).toBeUndefined()
    expect(attrs.bbox).toBeUndefined()
    expect(attrs.transform).toBeUndefined()
    expect(attrs.shape).toBeUndefined()
    expect(attrs.dimensions).toBeUndefined()
  })

  it('reads every proj: and spatial: field', () => {
    const projjson = {
      type: 'ProjectedCRS',
      id: { authority: 'EPSG', code: 32631 },
    }
    const attrs = parseGeoZarrAttrs(undefined, {
      'proj:code': 'EPSG:32631',
      'proj:wkt2': 'PROJCRS["WGS 84 / UTM zone 31N"]',
      'proj:projjson': projjson,
      'spatial:bbox': [500000, 4990000, 600000, 5000000],
      'spatial:transform': TRANSFORM,
      'spatial:shape': [1000, 10000],
      'spatial:dimensions': ['y', 'x'],
      'spatial:registration': 'node',
      'spatial:transform_type': 'affine',
    })

    expect(attrs.crs).toEqual({
      code: 'EPSG:32631',
      wkt2: 'PROJCRS["WGS 84 / UTM zone 31N"]',
      projjson,
    })
    expect(attrs.bbox).toEqual([500000, 4990000, 600000, 5000000])
    expect(attrs.transform).toEqual(TRANSFORM)
    expect(attrs.shape).toEqual([1000, 10000])
    expect(attrs.dimensions).toEqual(['y', 'x'])
    expect(attrs.registration).toBe('node')
    expect(attrs.transformType).toBe('affine')
  })

  it('lets array attributes override the group they inherit from', () => {
    const attrs = parseGeoZarrAttrs(
      { 'proj:code': 'EPSG:4326', 'spatial:registration': 'node' },
      { 'proj:code': 'EPSG:3857' }
    )

    expect(attrs.crs?.code).toBe('EPSG:3857')
    // Untouched by the array, so the group's value still applies.
    expect(attrs.registration).toBe('node')
  })

  it('carries a code-only CRS through without a wkt2 or projjson', () => {
    const attrs = parseGeoZarrAttrs({ 'proj:code': 'EPSG:5070' }, undefined)

    expect(attrs.crs).toEqual({
      code: 'EPSG:5070',
      wkt2: undefined,
      projjson: undefined,
    })
  })

  it('preserves a rotated transform for the caller to reject', () => {
    const rotated = [10, 1, 500000, 1, -10, 5000000]
    const attrs = parseGeoZarrAttrs(undefined, {
      'spatial:transform': rotated,
    })

    expect(attrs.transform).toEqual(rotated)
  })

  it('reports a non-affine transform type verbatim', () => {
    const attrs = parseGeoZarrAttrs(undefined, {
      'spatial:transform_type': 'rpc',
    })

    expect(attrs.transformType).toBe('rpc')
  })

  const dropped: [string, unknown, (a: GeoZarrAttrs) => unknown][] = [
    ['spatial:bbox', [1, 2, 3], (a) => a.bbox],
    ['spatial:bbox', [1, 2, 'three', 4], (a) => a.bbox],
    ['spatial:bbox', [10, 0, 0, 10], (a) => a.bbox],
    ['spatial:transform', [1, 2, 3], (a) => a.transform],
    ['spatial:transform', [0, 0, 500000, 0, -10, 5000000], (a) => a.transform],
    ['spatial:shape', [10.5, 10], (a) => a.shape],
    ['spatial:shape', [0, 10], (a) => a.shape],
    ['spatial:shape', [10], (a) => a.shape],
    ['spatial:dimensions', ['y'], (a) => a.dimensions],
    ['spatial:dimensions', ['y', ''], (a) => a.dimensions],
    ['proj:code', 42, (a) => a.crs],
    ['proj:wkt2', '', (a) => a.crs],
    ['proj:projjson', ['not', 'an', 'object'], (a) => a.crs],
  ]

  it.each(dropped)(
    'drops a malformed %s with a warning',
    (key, value, read) => {
      const warn = silenceWarnings()

      expect(
        read(parseGeoZarrAttrs(undefined, { [key]: value }))
      ).toBeUndefined()
      expect(warn).toHaveBeenCalledWith(expect.stringContaining(key))
    }
  )

  it('falls back to pixel registration when the value is unrecognized', () => {
    const warn = silenceWarnings()
    const attrs = parseGeoZarrAttrs(undefined, {
      'spatial:registration': 'centre',
    })

    expect(attrs.registration).toBe('pixel')
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining('spatial:registration')
    )
  })
})

describe('parseLayoutItemSpatial', () => {
  it('reads the per-level fields sitting alongside asset', () => {
    expect(
      parseLayoutItemSpatial({
        asset: 'r10m',
        transform: { scale: [1, 1], translation: [0, 0] },
        'spatial:transform': TRANSFORM,
        'spatial:shape': [10000, 10000],
      })
    ).toEqual({ transform: TRANSFORM, shape: [10000, 10000] })
  })

  it('returns nothing for an entry carrying only relative transforms', () => {
    expect(
      parseLayoutItemSpatial({
        asset: '0',
        transform: { scale: [2, 2], translation: [0, 0] },
      })
    ).toEqual({ transform: undefined, shape: undefined })
  })

  it.each([[null], [undefined], ['0'], [[1, 2]]])(
    'returns nothing for a non-object entry (%s)',
    (item) => {
      expect(parseLayoutItemSpatial(item)).toEqual({
        transform: undefined,
        shape: undefined,
      })
    }
  )
})

/**
 * A 100-column by 50-row grid on a 10 m north-up transform, the shape
 * `spatial:transform` is written for. Its edges are x 500000..501000 and
 * y 4999500..5000000.
 */
const GRID = { nCols: 100, nRows: 50 }

const attrs = (over: Partial<GeoZarrAttrs> = {}): GeoZarrAttrs => ({
  registration: 'pixel',
  transformType: 'affine',
  ...over,
})

describe('boundsFromSpatialAttrs', () => {
  it('walks a north-up transform out to the grid edges', () => {
    expect(
      boundsFromSpatialAttrs(
        attrs({ transform: [10, 0, 5e5, 0, -10, 5e6] }),
        GRID
      )
    ).toEqual({
      xMin: 5e5,
      xMax: 501000,
      yMin: 4999500,
      yMax: 5e6,
      latIsAscending: false,
    })
  })

  it('reads row direction off the sign of the y resolution', () => {
    const south = boundsFromSpatialAttrs(
      attrs({ transform: [10, 0, 5e5, 0, 10, 4999500] }),
      GRID
    )

    expect(south).toEqual({
      xMin: 5e5,
      xMax: 501000,
      yMin: 4999500,
      yMax: 5e6,
      latIsAscending: true,
    })
  })

  it('pushes a node-registered transform out by half a cell', () => {
    // The transform lands on the first cell center, so every edge moves out 5 m.
    expect(
      boundsFromSpatialAttrs(
        attrs({ registration: 'node', transform: [10, 0, 5e5, 0, -10, 5e6] }),
        GRID
      )
    ).toEqual({
      xMin: 499995,
      xMax: 500995,
      yMin: 4999505,
      yMax: 5000005,
      latIsAscending: false,
    })
  })

  it('takes a pixel-registered bbox as the edges outright', () => {
    expect(
      boundsFromSpatialAttrs(attrs({ bbox: [-180, -90, 180, 90] }), GRID)
    ).toEqual({
      xMin: -180,
      yMin: -90,
      xMax: 180,
      yMax: 90,
      latIsAscending: null,
    })
  })

  it('expands a node-registered bbox using the span between border centers', () => {
    // 10 columns of centers span 9 cells, so half a cell is 90/9/2 = 5.
    expect(
      boundsFromSpatialAttrs(
        attrs({ registration: 'node', bbox: [0, 0, 90, 45] }),
        { nCols: 10, nRows: 5 }
      )
    ).toEqual({
      xMin: -5,
      xMax: 95,
      yMin: -5.625,
      yMax: 50.625,
      latIsAscending: null,
    })
  })

  it('prefers the transform resolution over the bbox span for a node expansion', () => {
    const both = boundsFromSpatialAttrs(
      attrs({
        registration: 'node',
        bbox: [0, 0, 90, 45],
        transform: [10, 0, 0, 0, -10, 45],
      }),
      { nCols: 10, nRows: 5 }
    )

    expect(both).toEqual({
      xMin: -5,
      xMax: 95,
      yMin: -5,
      yMax: 50,
      latIsAscending: false,
    })
  })

  it('lets the bbox set the extent while the transform sets the row direction', () => {
    expect(
      boundsFromSpatialAttrs(
        attrs({ bbox: [0, 0, 90, 45], transform: [10, 0, 5e5, 0, -10, 5e6] }),
        GRID
      )
    ).toEqual({
      xMin: 0,
      yMin: 0,
      xMax: 90,
      yMax: 45,
      latIsAscending: false,
    })
  })

  it('rejects a rotated transform, whose corners are not its extent', () => {
    expect(
      boundsFromSpatialAttrs(
        attrs({ transform: [10, 1, 5e5, 1, -10, 5e6] }),
        GRID
      )
    ).toBeNull()
  })

  it('rejects a rotated grid even when a bbox encloses it', () => {
    // The bbox bounds the rotated footprint, but the renderer maps rows and
    // columns linearly across it, which would draw the raster unrotated and
    // stretched to the corners.
    expect(
      boundsFromSpatialAttrs(
        attrs({ bbox: [0, 0, 90, 45], transform: [10, 1, 5e5, 1, -10, 5e6] }),
        GRID
      )
    ).toBeNull()
  })

  it('rejects a transform type it cannot map to a grid', () => {
    expect(
      boundsFromSpatialAttrs(
        attrs({ transformType: 'rpc', bbox: [0, 0, 90, 45] }),
        GRID
      )
    ).toBeNull()
  })

  it('rejects a node-registered bbox on a grid too small to size a cell from', () => {
    expect(
      boundsFromSpatialAttrs(
        attrs({ registration: 'node', bbox: [0, 0, 90, 45] }),
        { nCols: 1, nRows: 1 }
      )
    ).toBeNull()
  })

  it('returns nothing when the store declares no extent at all', () => {
    expect(boundsFromSpatialAttrs(attrs(), GRID)).toBeNull()
  })
})
