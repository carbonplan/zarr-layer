import { describe, it, expect } from 'vitest'
import type * as zarr from 'zarrita'
import {
  computeRegionMercatorBounds,
  getCandidateRegions,
  getRegionBounds,
  getRegionSize,
  getVisibleRegions,
  longitudeWorldFraction,
  selectLevelForZoom,
} from './region-math'
import { createProjectionContext } from './projection-utils'
import { boundsToMercatorNorm } from './map-utils'
import type { UntiledLevel } from './types'

/**
 * Pure viewport/level math: region grid enumeration, pixel↔geo bounds,
 * resolution-matched level selection. These functions take all state as
 * explicit inputs, so the tests double as a signature spec.
 */

const WORLD = { xMin: -180, xMax: 180, yMin: -90, yMax: 90 }

const identityTransformer = {
  forward: (lon: number, lat: number): [number, number] => [lon, lat],
  inverse: (x: number, y: number): [number, number] => [x, y],
}

function fakeArray(spec: {
  shape: number[]
  chunks?: number[]
  codecs?: unknown[]
}): zarr.Array<zarr.DataType> {
  return {
    shape: spec.shape,
    chunks: spec.chunks,
    codecs: spec.codecs ?? [],
  } as unknown as zarr.Array<zarr.DataType>
}

const LATLON_INDICES = {
  lat: { index: 0, name: 'lat', array: null },
  lon: { index: 1, name: 'lon', array: null },
}

describe('getRegionSize', () => {
  it('uses the chunk shape when chunks subdivide the array', () => {
    const array = fakeArray({ shape: [100, 200], chunks: [50, 100] })
    expect(getRegionSize(array, LATLON_INDICES)).toEqual([50, 100])
  })

  it('prefers the shard shape from a sharding codec', () => {
    const array = fakeArray({
      shape: [100, 200],
      chunks: [10, 10],
      codecs: [
        {
          name: 'sharding_indexed',
          configuration: { chunk_shape: [64, 128] },
        },
      ],
    })
    expect(getRegionSize(array, LATLON_INDICES)).toEqual([64, 128])
  })

  it('returns null for a single-chunk array', () => {
    const array = fakeArray({ shape: [100, 200], chunks: [100, 200] })
    expect(getRegionSize(array, LATLON_INDICES)).toBeNull()
  })

  it('returns null without spatial dimension indices', () => {
    const array = fakeArray({ shape: [100, 200], chunks: [50, 100] })
    expect(getRegionSize(array, {})).toBeNull()
  })
})

describe('longitudeWorldFraction', () => {
  it('is 1 for global and degenerate extents', () => {
    expect(longitudeWorldFraction({ xMin: -180, xMax: 180 })).toBe(1)
    expect(longitudeWorldFraction({ xMin: 0, xMax: 360 })).toBe(1)
    expect(longitudeWorldFraction({ xMin: 10, xMax: 10 })).toBe(1)
    expect(longitudeWorldFraction({ xMin: NaN, xMax: 10 })).toBe(1)
  })

  it('is the longitude span over 360 for regional extents', () => {
    expect(longitudeWorldFraction({ xMin: -90, xMax: 90 })).toBeCloseTo(0.5, 12)
  })

  it('unwraps antimeridian-crossing extents', () => {
    expect(longitudeWorldFraction({ xMin: 170, xMax: -170 })).toBeCloseTo(
      20 / 360,
      12
    )
  })
})

describe('getRegionBounds', () => {
  const levelMeta = {
    width: 360,
    height: 180,
    regionSize: [90, 180] as [number, number],
  }

  it('maps region (0,0) to the north-west corner when lat is descending', () => {
    const b = getRegionBounds({
      regionX: 0,
      regionY: 0,
      levelMeta,
      xyLimits: WORLD,
      latIsAscending: false,
    })
    expect(b).toEqual({ xMin: -180, xMax: 0, yMin: 0, yMax: 90 })
  })

  it('maps region (0,0) to the south-west corner when lat is ascending', () => {
    const b = getRegionBounds({
      regionX: 0,
      regionY: 0,
      levelMeta,
      xyLimits: WORLD,
      latIsAscending: true,
    })
    expect(b).toEqual({ xMin: -180, xMax: 0, yMin: -90, yMax: 0 })
  })

  it('clamps edge regions to the array extent', () => {
    const meta = {
      width: 300,
      height: 180,
      regionSize: [90, 180] as [number, number],
    }
    const b = getRegionBounds({
      regionX: 1,
      regionY: 0,
      levelMeta: meta,
      xyLimits: WORLD,
      latIsAscending: false,
    })
    // Pixel columns 180..300 of a 300px world span.
    expect(b.xMin).toBeCloseTo(-180 + (180 / 300) * 360, 9)
    expect(b.xMax).toBeCloseTo(180, 9)
  })

  it('falls back to the unit square without xyLimits', () => {
    const b = getRegionBounds({
      regionX: 0,
      regionY: 0,
      levelMeta,
      xyLimits: null,
      latIsAscending: false,
    })
    expect(b).toEqual({ xMin: 0, xMax: 1, yMin: 0, yMax: 1 })
  })
})

describe('getCandidateRegions', () => {
  const grid = {
    transformer: identityTransformer,
    numRegionsX: 40,
    numRegionsY: 20,
    regionW: 90,
    regionH: 90,
    width: 3600,
    height: 1800,
    xyLimits: WORLD,
    latIsAscending: false,
  }

  it('returns a margin-padded block around the viewport', () => {
    const candidates = getCandidateRegions({
      ...grid,
      west: -10,
      south: -10,
      east: 10,
      north: 10,
    })
    const xs = candidates.map((c) => c.regionX)
    const ys = candidates.map((c) => c.regionY)
    // Viewport around the grid center, well away from the edges.
    expect(Math.min(...xs)).toBeGreaterThan(10)
    expect(Math.max(...xs)).toBeLessThan(30)
    expect(Math.min(...ys)).toBeGreaterThan(2)
    expect(Math.max(...ys)).toBeLessThan(18)
  })

  it('widens to every X column when the viewport crosses the antimeridian', () => {
    const candidates = getCandidateRegions({
      ...grid,
      west: 170,
      south: -10,
      east: -170,
      north: 10,
    })
    const xs = new Set(candidates.map((c) => c.regionX))
    expect(xs.size).toBe(grid.numRegionsX)
  })

  it('falls back to every region when no viewport sample projects', () => {
    const candidates = getCandidateRegions({
      ...grid,
      transformer: { forward: () => [NaN, NaN] },
      west: -10,
      south: -10,
      east: 10,
      north: 10,
    })
    expect(candidates).toHaveLength(grid.numRegionsX * grid.numRegionsY)
  })
})

describe('getVisibleRegions', () => {
  const projection = createProjectionContext({
    crs: 'EPSG:4326',
    proj4def: null,
    xyLimits: WORLD,
  })
  const levelMeta = {
    width: 360,
    height: 180,
    regionSize: [90, 90] as [number, number],
  }
  const mapAt = (west: number, south: number, east: number, north: number) =>
    ({
      getBounds: () => ({
        toArray: () => [
          [west, south],
          [east, north],
        ],
      }),
    } as never)

  it('returns exactly the regions overlapping the viewport', () => {
    const regions = getVisibleRegions({
      map: mapAt(-10, -10, 10, 10),
      xyLimits: WORLD,
      levelMeta,
      projection,
      latIsAscending: false,
    })
    // 4x2 grid of 90° regions; a viewport straddling (0,0) touches the two
    // middle columns in both rows.
    const keys = regions.map((r) => `${r.regionX},${r.regionY}`).sort()
    expect(keys).toEqual(['1,0', '1,1', '2,0', '2,1'])
  })

  it('returns empty without bounds, limits, level, or transformer', () => {
    const noBoundsMap = { getBounds: () => undefined } as never
    expect(
      getVisibleRegions({
        map: noBoundsMap,
        xyLimits: WORLD,
        levelMeta,
        projection,
        latIsAscending: false,
      })
    ).toEqual([])
    expect(
      getVisibleRegions({
        map: mapAt(-10, -10, 10, 10),
        xyLimits: null,
        levelMeta,
        projection,
        latIsAscending: false,
      })
    ).toEqual([])
    expect(
      getVisibleRegions({
        map: mapAt(-10, -10, 10, 10),
        xyLimits: WORLD,
        levelMeta: null,
        projection,
        latIsAscending: false,
      })
    ).toEqual([])
  })
})

describe('selectLevelForZoom', () => {
  const level = (asset: string, lonSize: number): UntiledLevel => ({
    asset,
    scale: [1, 1],
    translation: [0, 0],
    shape: [lonSize / 2, lonSize],
  })
  const projection4326 = createProjectionContext({
    crs: 'EPSG:4326',
    proj4def: null,
    xyLimits: null,
  })

  it('matches slippy zoom levels for a 256px global pyramid', () => {
    // Levels 0..3 of a 256px-tile pyramid: lon sizes 256, 512, 1024, 2048.
    const levels = [0, 1, 2, 3].map((i) => level(String(i), 256 * 2 ** i))
    for (const zoom of [0, 1, 2, 3]) {
      expect(
        selectLevelForZoom({
          mapZoom: zoom,
          xyLimits: WORLD,
          levels,
          projection: projection4326,
          lonIndex: 1,
        })
      ).toBe(zoom)
    }
  })

  it('picks one level finer for a 128px pyramid', () => {
    const levels = [0, 1, 2, 3].map((i) => level(String(i), 128 * 2 ** i))
    expect(
      selectLevelForZoom({
        mapZoom: 1,
        xyLimits: WORLD,
        levels,
        projection: projection4326,
        lonIndex: 1,
      })
    ).toBe(2)
  })

  it('scales resolution by the longitude world fraction for regional data', () => {
    // Half-world data: each pixel covers half the ground, so effective
    // resolution doubles and a coarser level suffices at the same zoom.
    const levels = [0, 1, 2].map((i) => level(String(i), 256 * 2 ** i))
    const regional = { xMin: -90, xMax: 90, yMin: -45, yMax: 45 }
    expect(
      selectLevelForZoom({
        mapZoom: 1,
        xyLimits: regional,
        levels,
        projection: projection4326,
        lonIndex: 1,
      })
    ).toBe(0)
  })

  it('returns the finest level when the zoom exceeds every level', () => {
    const levels = [0, 1].map((i) => level(String(i), 256 * 2 ** i))
    expect(
      selectLevelForZoom({
        mapZoom: 10,
        xyLimits: WORLD,
        levels,
        projection: projection4326,
        lonIndex: 1,
      })
    ).toBe(1)
  })

  it('uses meter extents for EPSG:3857 data', () => {
    const E = 20037508.342789244
    const projection3857 = createProjectionContext({
      crs: 'EPSG:3857',
      proj4def: null,
      xyLimits: null,
    })
    const levels = [0, 1, 2].map((i) => level(String(i), 256 * 2 ** i))
    expect(
      selectLevelForZoom({
        mapZoom: 2,
        xyLimits: { xMin: -E, xMax: E, yMin: -E, yMax: E },
        levels,
        projection: projection3857,
        lonIndex: 1,
      })
    ).toBe(2)
  })

  it('returns level 0 without limits or levels', () => {
    expect(
      selectLevelForZoom({
        mapZoom: 3,
        xyLimits: null,
        levels: [],
        projection: projection4326,
        lonIndex: 1,
      })
    ).toBe(0)
  })
})

describe('computeRegionMercatorBounds', () => {
  it('matches boundsToMercatorNorm for the builtin CRSes', () => {
    const bounds = { xMin: -90, xMax: 90, yMin: -45, yMax: 45 }
    const p4326 = createProjectionContext({
      crs: 'EPSG:4326',
      proj4def: null,
      xyLimits: WORLD,
    })
    expect(computeRegionMercatorBounds(bounds, p4326)).toEqual(
      boundsToMercatorNorm(bounds, 'EPSG:4326')
    )

    const E = 20037508.342789244
    const meters = { xMin: -E / 2, xMax: E / 2, yMin: -E / 2, yMax: E / 2 }
    const p3857 = createProjectionContext({
      crs: 'EPSG:3857',
      proj4def: null,
      xyLimits: null,
    })
    expect(computeRegionMercatorBounds(meters, p3857)).toEqual(
      boundsToMercatorNorm(meters, 'EPSG:3857')
    )
  })
})
