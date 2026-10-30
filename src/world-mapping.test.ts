import { describe, it, expect } from 'vitest'
import proj4 from 'proj4'
import {
  worldBoundsOf,
  worldMappingFor,
  worldFractionOf,
  type MapProjection,
  type WorldMapping,
} from './world-mapping'
import { getRegionsInWorldBox, selectLevelForZoom } from './region-math'
import { createWorldMesh } from './mesh-reprojector'
import { createProjectionContext } from './projection-utils'
import type { MapLike } from './types'

/**
 * A map in a projection registered with MapLibre's `addProjection` places
 * custom layers in the world square of its tile matrix set. These fixtures
 * stand in for a registered projection's mapping with closed forms.
 */

/** Plate carrée over the top half of the world square. */
const PLATE: WorldMapping = {
  toWorld: (lng, lat) => [(lng + 180) / 360, (90 - lat) / 360],
  toLngLat: (x, y) => {
    const lat = 90 - y * 360
    return lat < -90 || lat > 90 ? null : [x * 360 - 180, lat]
  },
}

/** A south polar azimuthal projection, centered in the world square. */
const POLAR: WorldMapping = {
  toWorld: (lng, lat) => {
    const r = (lat + 90) / 360
    const a = (lng * Math.PI) / 180
    return [0.5 + r * Math.sin(a), 0.5 - r * Math.cos(a)]
  },
  toLngLat: (x, y) => {
    const dx = x - 0.5
    const dy = 0.5 - y
    return [(Math.atan2(dx, dy) * 180) / Math.PI, Math.hypot(dx, dy) * 360 - 90]
  },
}

/**
 * South polar stereographic: the north pole lands a finite but enormous
 * distance away, as proj4 places it.
 */
const STEREO: WorldMapping = {
  toWorld: (lng, lat) => {
    const r = Math.tan((((lat + 90) / 2) * Math.PI) / 180) / 4
    const a = (lng * Math.PI) / 180
    return [0.5 + r * Math.sin(a), 0.5 - r * Math.cos(a)]
  },
  toLngLat: (x, y) => {
    const dx = x - 0.5
    const dy = 0.5 - y
    return [
      (Math.atan2(dx, dy) * 180) / Math.PI,
      (Math.atan(Math.hypot(dx, dy) * 4) * 360) / Math.PI - 90,
    ]
  },
}

const IDENTITY_4326 = {
  forward: (x: number, y: number): [number, number] => [x, y],
  inverse: (x: number, y: number): [number, number] => [x, y],
}

/** Plate carrée in degrees, tile 0/0/0 spanning lon -180..180 from lat 90 down. */
const plateCarree = (
  overrides: Partial<MapProjection['projection']> = {}
): MapProjection => ({
  name: 'custom:plate',
  projection: {
    forward: ([lng, lat]) => [lng, lat],
    inverse: ([x, y]) => [x, y],
    ...overrides,
  },
  tileMatrixSet: { origin: [-180, 90], extentAtZoom0: 360 },
})

const mapIn = (type: string) =>
  ({ getProjection: () => ({ type }) } as unknown as MapLike)

describe('worldMappingFor', () => {
  it('maps the projection the map is in from its definition', () => {
    const mapping = worldMappingFor(mapIn('custom:plate'), [plateCarree()])!

    // World y grows downwards from the tile matrix set's top-left origin.
    expect(mapping.toWorld(0, 90)).toEqual([0.5, 0])
    expect(mapping.toWorld(-180, 0)).toEqual([0, 0.25])
    expect(mapping.toLngLat(0.75, 0.125)).toEqual([90, 45])
  })

  it('gives the same mapping each frame for the same projection', () => {
    const projections = [plateCarree()]
    expect(worldMappingFor(mapIn('custom:plate'), projections)).toBe(
      worldMappingFor(mapIn('custom:plate'), projections)
    )
  })

  it('returns null for built-in projections and ones it was not given', () => {
    const projections = [plateCarree()]
    expect(worldMappingFor(mapIn('mercator'), projections)).toBe(null)
    expect(worldMappingFor(mapIn('globe'), projections)).toBe(null)
    expect(worldMappingFor(mapIn('EPSG:3031'), projections)).toBe(null)
    expect(worldMappingFor(null, projections)).toBe(null)
  })

  it('reports positions the projection cannot place', () => {
    const mapping = worldMappingFor(mapIn('custom:plate'), [
      plateCarree({
        forward: ([lng, lat]) => {
          if (lng > 90) throw new Error('outside')
          return [lng > 45 ? Infinity : lng, lat]
        },
        inverse: ([x, y]) => (x > 0 ? null : [x, y]),
      }),
    ])!

    expect(mapping.toWorld(60, 0)).toEqual([NaN, NaN])
    expect(mapping.toWorld(120, 0)).toEqual([NaN, NaN])
    expect(mapping.toLngLat(0.75, 0.25)).toBe(null)
  })

  it('has no answer where the inverse does not lead back to the position', () => {
    const mapping = worldMappingFor(mapIn('custom:plate'), [
      plateCarree({ inverse: ([x, y]) => (x > 0 ? [x - 360, y] : [x, y]) }),
    ])!

    expect(mapping.toLngLat(0.25, 0.25)).toEqual([-90, 0])
    expect(mapping.toLngLat(0.75, 0.25)).toBe(null)
  })

  it('takes an inverse that only approximately leads back, as MapLibre does', () => {
    const rounded = (v: number) => Math.round(v * 100) / 100
    const mapping = worldMappingFor(mapIn('custom:plate'), [
      plateCarree({ inverse: ([x, y]) => [rounded(x), rounded(y)] }),
    ])!

    expect(mapping.toLngLat(0.5 + 0.004 / 360, 0.25)).toEqual([0, 0])
  })
})

describe('worldBoundsOf', () => {
  it('maps a rectangle under a linear mapping exactly', () => {
    expect(
      worldBoundsOf(
        { xMin: -180, xMax: 0, yMin: 0, yMax: 90 },
        PLATE,
        IDENTITY_4326
      )
    ).toEqual({ x0: 0, y0: 0, x1: 0.5, y1: 0.25 })
  })

  it('reaches the curved edges of a cap around the pole', () => {
    const bounds = worldBoundsOf(
      { xMin: -180, xMax: 180, yMin: -90, yMax: -72 },
      POLAR,
      IDENTITY_4326
    )!
    const r = 18 / 360

    for (const [edge, cap] of [
      [0.5 - bounds.x0, r],
      [bounds.x1 - 0.5, r],
      [0.5 - bounds.y0, r],
      [bounds.y1 - 0.5, r],
    ]) {
      expect(edge).toBeGreaterThanOrEqual(cap)
      expect(edge).toBeLessThan(cap * 1.05)
    }
  })

  it('holds a curved edge that bulges out between samples', () => {
    // The equator from lon 10 to 190 reaches farthest east at lon 90, between
    // the samples at 55 and 100.
    const bounds = worldBoundsOf(
      { xMin: 10, xMax: 190, yMin: -90, yMax: 0 },
      POLAR,
      IDENTITY_4326,
      4
    )!

    expect(bounds.x1).toBeGreaterThanOrEqual(0.75)
    expect(bounds.x1).toBeLessThan(0.76)
  })

  it('is null when nothing lands in the world', () => {
    const nowhere: WorldMapping = { ...PLATE, toWorld: () => [NaN, NaN] }
    expect(
      worldBoundsOf(
        { xMin: 0, xMax: 1, yMin: 0, yMax: 1 },
        nowhere,
        IDENTITY_4326
      )
    ).toBe(null)
  })
})

describe('getRegionsInWorldBox', () => {
  const projection = createProjectionContext({
    crs: 'EPSG:4326',
    proj4def: null,
    xyLimits: { xMin: -180, xMax: 180, yMin: -90, yMax: 90 },
  })
  // 4x4 regions of 90 by 45 degrees.
  const levelMeta = {
    width: 32,
    height: 16,
    regionSize: [4, 8] as [number, number],
  }
  const regionsIn = (
    box: { x0: number; y0: number; x1: number; y1: number },
    mapping = PLATE
  ) =>
    getRegionsInWorldBox({
      box,
      xyLimits: { xMin: -180, xMax: 180, yMin: -90, yMax: 90 },
      levelMeta,
      projection,
      mapping,
      latIsAscending: false,
    })
      .map(({ regionX, regionY }) => `${regionX},${regionY}`)
      .sort()

  it('selects the regions under a tile, leaving out those that only touch it', () => {
    // Tile 1/0/0 is the western hemisphere in this world.
    expect(regionsIn({ x0: 0, y0: 0, x1: 0.5, y1: 0.5 })).toEqual([
      '0,0',
      '0,1',
      '0,2',
      '0,3',
      '1,0',
      '1,1',
      '1,2',
      '1,3',
    ])
  })

  it('keeps a region whose curved edge bulges into the box between samples', () => {
    // One region, lon 10..190 and lat -90..0; its equator edge reaches past
    // x 0.748 only around lon 90.
    const limits = { xMin: 10, xMax: 190, yMin: -90, yMax: 0 }
    expect(
      getRegionsInWorldBox({
        box: { x0: 0.747, y0: 0.499, x1: 0.749, y1: 0.501 },
        xyLimits: limits,
        levelMeta: { width: 1, height: 1, regionSize: [1, 1] },
        projection: createProjectionContext({
          crs: 'EPSG:4326',
          proj4def: null,
          xyLimits: limits,
        }),
        mapping: POLAR,
        latIsAscending: false,
      })
    ).toEqual([{ regionX: 0, regionY: 0 }])
  })

  it('selects regions of a 0-360° dataset west of the antimeridian', () => {
    const limits = { xMin: 0, xMax: 360, yMin: -90, yMax: 90 }
    // Lon 260 (-100) at 60°S, in region 2,3 (lon 180..270, lat -45..-90).
    const [x, y] = POLAR.toWorld(-100, -60)
    expect(
      getRegionsInWorldBox({
        box: { x0: x - 0.002, y0: y - 0.002, x1: x + 0.002, y1: y + 0.002 },
        xyLimits: limits,
        levelMeta,
        projection: createProjectionContext({
          crs: 'EPSG:4326',
          proj4def: null,
          xyLimits: limits,
        }),
        mapping: POLAR,
        latIsAscending: false,
      })
    ).toEqual([{ regionX: 2, regionY: 3 }])
  })

  it('selects the ring of regions around the pole of a polar world', () => {
    // A box at the pole holds the southernmost row, every longitude.
    expect(
      regionsIn({ x0: 0.45, y0: 0.45, x1: 0.55, y1: 0.55 }, POLAR)
    ).toEqual(['0,3', '1,3', '2,3', '3,3'])
  })
})

describe('worldFractionOf', () => {
  it('is the share of the world width the data spans', () => {
    expect(
      worldFractionOf(
        { xMin: -180, xMax: 0, yMin: 0, yMax: 90 },
        PLATE,
        IDENTITY_4326
      )
    ).toBeCloseTo(0.5, 9)
  })

  it('counts only the world square, past which far points are placed', () => {
    // Global data reaches the north pole, which the projection places far
    // outside the world square.
    expect(
      worldFractionOf(
        { xMin: -180, xMax: 180, yMin: -90, yMax: 90 },
        STEREO,
        IDENTITY_4326
      )
    ).toBeCloseTo(1, 1)
  })

  it('counts only the part of a narrow strip inside the world square', () => {
    // A one-degree strip reaching the north pole: inside the square it is a
    // sliver from the south pole out to the square's edge.
    const fraction = worldFractionOf(
      { xMin: 0, xMax: 1, yMin: -90, yMax: 90 },
      STEREO,
      IDENTITY_4326
    )!

    expect(fraction).toBeGreaterThan(0.005)
    expect(fraction).toBeLessThanOrEqual(0.5 * Math.sin(Math.PI / 180))
  })

  it('is undefined when nothing lands in the world', () => {
    expect(
      worldFractionOf(
        { xMin: 0, xMax: 1, yMin: 0, yMax: 1 },
        { ...PLATE, toWorld: () => [NaN, NaN] },
        IDENTITY_4326
      )
    ).toBe(undefined)
  })
})

describe('selectLevelForZoom in a registered projection', () => {
  const projection = createProjectionContext({
    crs: 'EPSG:4326',
    proj4def: null,
    xyLimits: { xMin: -180, xMax: 180, yMin: -90, yMax: 90 },
  })
  const levels = [
    { asset: '0', shape: [128, 256] },
    { asset: '1', shape: [256, 512] },
  ]
  const select = (worldFraction?: number) =>
    selectLevelForZoom({
      mapZoom: 1,
      xyLimits: { xMin: -180, xMax: 180, yMin: -90, yMax: 90 },
      levels,
      projection,
      worldFraction,
    })

  it('reads coarser data when the data spans less of the world', () => {
    expect(select()).toBe(1)
    expect(select(1)).toBe(1)
    expect(select(0.5)).toBe(0)
  })
})

describe('createWorldMesh', () => {
  const geoBounds = { xMin: -180, xMax: 0, yMin: 0, yMax: 90 }
  const mesh = (mapping: WorldMapping) =>
    createWorldMesh({
      geoBounds,
      width: 16,
      height: 8,
      lonSubdivisions: 4,
      latSubdivisions: 4,
      toWorld: (x, y) => mapping.toWorld(x, y),
      fromWorld: (x, y) => mapping.toLngLat(x, y) ?? [NaN, NaN],
      latIsAscending: false,
    })
  const worldPositions = (result: ReturnType<typeof createWorldMesh>) => {
    const { x0, y0, x1, y1 } = result.meshBounds
    const points: Array<[number, number]> = []
    for (let i = 0; i < result.positions.length; i += 2) {
      points.push([
        (x0 + x1) / 2 + (result.positions[i] * (x1 - x0)) / 2,
        (y0 + y1) / 2 + (result.positions[i + 1] * (y1 - y0)) / 2,
      ])
    }
    return points
  }

  it('places vertices at their world positions, around the region anchor', () => {
    const result = mesh(PLATE)

    expect(result.meshBounds.x0).toBeCloseTo(0, 9)
    expect(result.meshBounds.x1).toBeCloseTo(0.5, 9)
    expect(result.meshBounds.y0).toBeCloseTo(0, 9)
    expect(result.meshBounds.y1).toBeCloseTo(0.25, 9)
    const points = worldPositions(result)
    for (let i = 0; i < points.length; i++) {
      const u = result.texCoords[i * 2]
      const v = result.texCoords[i * 2 + 1]
      // v runs north to south, as the data rows do.
      expect(points[i][0]).toBeCloseTo(u * 0.5, 6)
      expect(points[i][1]).toBeCloseTo(v * 0.25, 6)
    }
    expect(result.indices.length).toBeGreaterThan(0)
  })

  it('drops triangles with a vertex the projection cannot place', () => {
    const western: WorldMapping = {
      ...PLATE,
      toWorld: (lng, lat) => (lng > -90 ? [NaN, NaN] : PLATE.toWorld(lng, lat)),
    }
    const result = mesh(western)
    const points = worldPositions(result)

    expect(result.indices.length).toBeGreaterThan(0)
    for (const index of result.indices) {
      expect(points[index][0]).toBeLessThanOrEqual(0.25 + 1e-9)
    }
    expect(result.meshBounds.x1).toBeCloseTo(0.25, 9)
  })

  it('meets the neighboring region along a curved edge without a gap', () => {
    const region = (yMin: number, yMax: number) =>
      createWorldMesh({
        geoBounds: { xMin: -90, xMax: 0, yMin, yMax },
        width: 16,
        height: 8,
        lonSubdivisions: 4,
        latSubdivisions: 4,
        toWorld: (x, y) => POLAR.toWorld(x, y),
        fromWorld: (x, y) => POLAR.toLngLat(x, y) ?? [NaN, NaN],
        latIsAscending: true,
      })
    // The edge at 60°S, a circle around the pole.
    const edge = (result: ReturnType<typeof createWorldMesh>, v: number) => {
      const points = worldPositions(result)
      return [...new Set(result.indices)]
        .filter((i) => result.texCoords[i * 2 + 1] === v)
        .map((i) => points[i])
    }
    const inner = edge(region(-90, -60), 1)
    const outer = edge(region(-60, -30), 0)
    const onPolyline = (
      [x, y]: [number, number],
      polyline: Array<[number, number]>
    ) =>
      polyline.some(([ax, ay], k) => {
        const [bx, by] = polyline[(k + 1) % polyline.length]
        const t = Math.max(
          0,
          Math.min(
            1,
            ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) /
              ((bx - ax) ** 2 + (by - ay) ** 2 || 1)
          )
        )
        return Math.hypot(x - ax - t * (bx - ax), y - ay - t * (by - ay)) < 1e-6
      })
    const byAngle = (points: Array<[number, number]>) =>
      [...points].sort(
        (a, b) =>
          Math.atan2(a[0] - 0.5, 0.5 - a[1]) -
          Math.atan2(b[0] - 0.5, 0.5 - b[1])
      )

    for (const point of outer)
      expect(onPolyline(point, byAngle(inner))).toBe(true)
    for (const point of inner)
      expect(onPolyline(point, byAngle(outer))).toBe(true)
  })

  it('meshes a region reaching the pole the projection sends far away', () => {
    // Antarctic polar stereographic places the north pole ~1e23 m out, where
    // proj4's inverse no longer leads back.
    const antarctic = worldMappingFor(mapIn('EPSG:3031'), [
      {
        name: 'EPSG:3031',
        projection: proj4(
          'EPSG:4326',
          '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
        ),
        tileMatrixSet: {
          origin: [-4194304, 4194304],
          extentAtZoom0: 8388608,
        },
      },
    ])!
    const result = createWorldMesh({
      geoBounds: { xMin: -36, xMax: 36, yMin: 30, yMax: 90 },
      width: 360,
      height: 300,
      lonSubdivisions: 8,
      latSubdivisions: 8,
      toWorld: (x, y) => antarctic.toWorld(x, y),
      fromWorld: (x, y) => antarctic.toLngLat(x, y) ?? [NaN, NaN],
      latIsAscending: true,
    })

    expect(result.indices.length).toBeGreaterThan(0)
  })

  it('drops triangles spanning a seam in the projection', () => {
    // East of lon -90 the world wraps to the far side, as past a seam.
    const seam: WorldMapping = {
      ...PLATE,
      toWorld: (lng, lat) => {
        const [x, y] = PLATE.toWorld(lng, lat)
        return [lng > -90 ? x + 1 : x, y]
      },
    }
    const result = mesh(seam)
    const points = worldPositions(result)

    for (let t = 0; t < result.indices.length; t += 3) {
      const xs = [0, 1, 2].map((k) => points[result.indices[t + k]][0])
      expect(Math.max(...xs) - Math.min(...xs)).toBeLessThan(0.5)
    }
  })
})
