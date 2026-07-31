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

  it('wgs84Bounds carries the region extent in normalized mercator', () => {
    // lon ±90 -> mercator X 0.25..0.75; lat ±45 -> mercator Y via the
    // gudermannian forward (smaller Y = north).
    expect(mesh.wgs84Bounds.x0).toBeCloseTo(0.25, 9)
    expect(mesh.wgs84Bounds.x1).toBeCloseTo(0.75, 9)
    expect(mesh.wgs84Bounds.y0).toBeCloseTo(latToMercatorNorm(45), 9)
    expect(mesh.wgs84Bounds.y1).toBeCloseTo(latToMercatorNorm(-45), 9)
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
