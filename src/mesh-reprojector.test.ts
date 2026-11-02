import { describe, it, expect } from 'vitest'
import { createHybridMesh } from './mesh-reprojector'
import { latToMercatorNorm } from './map-utils'
import {
  createTransformerTo4326,
  type ProjectionTransformer,
} from './projection-utils'
import { WEB_MERCATOR_EXTENT } from './constants'
import type { Bounds } from './types'

/**
 * Tests for the client-side reprojection mesh — the geometry that the GPU
 * drapes the raster onto. createHybridMesh is pure (no GL): it builds an
 * adaptive + uniform vertex grid, triangulates it (Delaunay), reprojects to
 * WGS84, and splits triangles at the antimeridian.
 *
 * Exact vertex positions depend on adaptive refinement, so these assert
 * invariants that pin the contract and catch the regression classes this
 * module has historically hit: non-finite vertices at projection edges,
 * corrupt/inconsistent triangulation, and non-determinism.
 */

/** Identity transformer: treats the source CRS as WGS84 (forward = passthrough). */
function identityTransformer(bounds: Bounds): ProjectionTransformer {
  return {
    forward: (x, y) => [x, y],
    inverse: (x, y) => [x, y],
    bounds,
  }
}

/** Assert a typed array contains only finite numbers. */
function expectAllFinite(arr: ArrayLike<number>, label: string) {
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i])) {
      throw new Error(`${label}[${i}] is not finite: ${arr[i]}`)
    }
  }
}

/** Shared structural invariants every mesh must satisfy. */
function expectValidMesh(mesh: ReturnType<typeof createHybridMesh>) {
  const numVerts = mesh.positions.length / 2
  expect(numVerts).toBeGreaterThan(0)
  // positions and texCoords are paired (x,y) / (u,v) per vertex.
  expect(mesh.texCoords.length).toBe(mesh.positions.length)
  // Triangles: a whole number of them, all referencing valid vertices.
  expect(mesh.indices.length % 3).toBe(0)
  expect(mesh.indices.length).toBeGreaterThan(0)
  for (let i = 0; i < mesh.indices.length; i++) {
    expect(mesh.indices[i]).toBeLessThan(numVerts)
  }
}

describe('createHybridMesh — identity (EPSG:4326 passthrough)', () => {
  const bounds: Bounds = [-90, -45, 90, 45]
  const mesh = createHybridMesh({
    geoBounds: { xMin: -90, yMin: -45, xMax: 90, yMax: 45 },
    width: 128,
    height: 64,
    lonSubdivisions: 8,
    latSubdivisions: 8,
    transformer: identityTransformer(bounds),
    latIsAscending: false,
  })

  it('produces a structurally valid, all-finite mesh', () => {
    expectValidMesh(mesh)
    expectAllFinite(mesh.positions, 'positions')
    expectAllFinite(mesh.texCoords, 'texCoords')
  })

  it('covers the region in anchor-relative space', () => {
    // Positions are mercator deltas from the region anchor, normalized by the
    // half-extent — so the region's own edges land exactly at ±1.
    const xs: number[] = []
    const ys: number[] = []
    for (let i = 0; i < mesh.positions.length; i += 2) {
      xs.push(mesh.positions[i])
      ys.push(mesh.positions[i + 1])
    }
    expect(Math.min(...xs)).toBeCloseTo(-1, 5)
    expect(Math.max(...xs)).toBeCloseTo(1, 5)
    expect(Math.min(...ys)).toBeCloseTo(-1, 5)
    expect(Math.max(...ys)).toBeCloseTo(1, 5)
  })

  it('emits texture UVs spanning the full [0, 1] range', () => {
    let min = Infinity
    let max = -Infinity
    for (let i = 0; i < mesh.texCoords.length; i++) {
      min = Math.min(min, mesh.texCoords[i])
      max = Math.max(max, mesh.texCoords[i])
    }
    expect(min).toBeCloseTo(0, 5)
    expect(max).toBeCloseTo(1, 5)
  })

  it('meshBounds carries the region extent in normalized mercator', () => {
    // lon ±90 -> mercator X 0.25..0.75; lat ±45 -> mercator Y via the
    // gudermannian forward (smaller Y = north).
    expect(mesh.meshBounds.x0).toBeCloseTo(0.25, 9)
    expect(mesh.meshBounds.x1).toBeCloseTo(0.75, 9)
    expect(mesh.meshBounds.y0).toBeCloseTo(latToMercatorNorm(45), 9)
    expect(mesh.meshBounds.y1).toBeCloseTo(latToMercatorNorm(-45), 9)
  })

  it('is deterministic for identical input', () => {
    const again = createHybridMesh({
      geoBounds: { xMin: -90, yMin: -45, xMax: 90, yMax: 45 },
      width: 128,
      height: 64,
      lonSubdivisions: 8,
      latSubdivisions: 8,
      transformer: identityTransformer(bounds),
      latIsAscending: false,
    })
    expect(Array.from(again.positions)).toEqual(Array.from(mesh.positions))
    expect(Array.from(again.indices)).toEqual(Array.from(mesh.indices))
  })

  it('adds vertices as subdivisions increase', () => {
    const opts = (subdivisions: number) => ({
      geoBounds: { xMin: -90, yMin: -45, xMax: 90, yMax: 45 },
      width: 128,
      height: 64,
      lonSubdivisions: subdivisions,
      latSubdivisions: subdivisions,
      transformer: identityTransformer(bounds),
      latIsAscending: false,
    })
    const coarse = createHybridMesh(opts(2))
    const fine = createHybridMesh(opts(16))
    expect(fine.positions.length).toBeGreaterThan(coarse.positions.length)
  })
})

describe('createHybridMesh — real EPSG:3857 reprojection', () => {
  it('reprojects mercator meters to a finite, valid mesh', () => {
    const E = WEB_MERCATOR_EXTENT
    const bounds: Bounds = [-E / 2, -E / 2, E / 2, E / 2]
    const transformer = createTransformerTo4326('EPSG:3857', bounds)

    const mesh = createHybridMesh({
      geoBounds: { xMin: -E / 2, yMin: -E / 2, xMax: E / 2, yMax: E / 2 },
      width: 256,
      height: 256,
      lonSubdivisions: 8,
      latSubdivisions: 8,
      transformer,
      latIsAscending: false,
    })

    expectValidMesh(mesh)
    // The key regression guard: no NaN vertices from the projection path.
    expectAllFinite(mesh.positions, 'positions')
    expectAllFinite(mesh.texCoords, 'texCoords')
  })
})

describe('createHybridMesh — refineInMercator', () => {
  // ETRS89 LAEA Europe, 1 km pixels: a 1024 km region over central Europe.
  const def =
    '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs'
  const geoBounds = { xMin: 4.0e6, yMin: 3.5e6, xMax: 5.024e6, yMax: 4.524e6 }
  const size = 1024
  const transformer = createTransformerTo4326(def, [
    geoBounds.xMin,
    geoBounds.yMin,
    geoBounds.xMax,
    geoBounds.yMax,
  ])
  const build = (refineInMercator: boolean) =>
    createHybridMesh({
      geoBounds,
      width: size,
      height: size,
      lonSubdivisions: 8,
      latSubdivisions: 8,
      transformer,
      latIsAscending: false,
      refineInMercator,
    })

  /**
   * Largest gap, in source pixels, between the texel the flat map draws at a
   * point (texture coordinates interpolated linearly in Mercator, as the GPU
   * does) and the texel that truly belongs there, over each triangle's
   * centroid and edge midpoints.
   */
  function maxDrawnError(mesh: ReturnType<typeof createHybridMesh>): number {
    const { x0, y0, x1, y1 } = mesh.meshBounds
    const cx = (x0 + x1) / 2
    const cy = (y0 + y1) / 2
    const hx = (x1 - x0) / 2
    const hy = (y1 - y0) / 2
    const weights = [
      [1 / 3, 1 / 3, 1 / 3],
      [0.5, 0.5, 0],
      [0.5, 0, 0.5],
      [0, 0.5, 0.5],
    ]
    let max = 0
    for (let t = 0; t < mesh.indices.length; t += 3) {
      const ids = [mesh.indices[t], mesh.indices[t + 1], mesh.indices[t + 2]]
      for (const w of weights) {
        let mx = 0
        let my = 0
        let u = 0
        let v = 0
        ids.forEach((id, k) => {
          mx += w[k] * (cx + mesh.positions[id * 2] * hx)
          my += w[k] * (cy + mesh.positions[id * 2 + 1] * hy)
          u += w[k] * mesh.texCoords[id * 2]
          v += w[k] * mesh.texCoords[id * 2 + 1]
        })
        const lon = mx * 360 - 180
        const lat =
          (Math.atan(Math.sinh(Math.PI * (1 - 2 * my))) * 180) / Math.PI
        const [sx, sy] = transformer.inverse(lon, lat)
        const px =
          ((sx - geoBounds.xMin) / (geoBounds.xMax - geoBounds.xMin)) * size
        const py =
          ((geoBounds.yMax - sy) / (geoBounds.yMax - geoBounds.yMin)) * size
        max = Math.max(max, Math.hypot(px - u * size, py - v * size))
      }
    }
    return max
  }

  it('bounds the drawn error on the flat map', () => {
    expect(maxDrawnError(build(true))).toBeLessThan(0.15)
  })

  it('tightens a mesh refined against WGS84', () => {
    const wgs84 = maxDrawnError(build(false))
    expect(wgs84).toBeGreaterThan(0.25)
    expect(maxDrawnError(build(true))).toBeLessThan(wgs84 / 2)
  })

  it('adds no vertices for EPSG:3857 data, already linear in Mercator', () => {
    const merc = {
      xMin: 0,
      yMin: 5.0e6,
      xMax: 2.0e6,
      yMax: 7.0e6,
    }
    const t = createTransformerTo4326('EPSG:3857', [
      merc.xMin,
      merc.yMin,
      merc.xMax,
      merc.yMax,
    ])
    const options = {
      geoBounds: merc,
      width: 512,
      height: 512,
      lonSubdivisions: 8,
      latSubdivisions: 8,
      transformer: t,
      latIsAscending: false,
    }
    const uniformVerts = 9 * 9
    const refined = createHybridMesh({ ...options, refineInMercator: true })
    const wgs84 = createHybridMesh({ ...options, refineInMercator: false })
    expect(refined.positions.length / 2).toBeLessThan(uniformVerts + 8)
    expect(wgs84.positions.length / 2).toBeGreaterThan(
      refined.positions.length / 2
    )
  })

  it('stays bounded over a pole', () => {
    const polar = { xMin: -2.0e6, yMin: -2.0e6, xMax: 2.0e6, yMax: 2.0e6 }
    const stere =
      '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
    const options = {
      geoBounds: polar,
      width: 512,
      height: 512,
      lonSubdivisions: 8,
      latSubdivisions: 8,
      transformer: createTransformerTo4326(
        stere,
        [-2.0e6, -2.0e6, 2.0e6, 2.0e6]
      ),
      latIsAscending: false,
    }
    const refined = createHybridMesh({ ...options, refineInMercator: true })
    const wgs84 = createHybridMesh({ ...options, refineInMercator: false })
    expectValidMesh(refined)
    expect(refined.positions.length).toBeLessThan(wgs84.positions.length * 1.5)
  })
})
