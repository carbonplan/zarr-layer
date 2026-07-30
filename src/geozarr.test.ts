import { describe, it, expect, vi, afterEach } from 'vitest'
import {
  parseGeoZarrAttrs,
  parseLayoutItemSpatial,
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
