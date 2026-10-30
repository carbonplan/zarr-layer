/**
 * @module world-mapping
 *
 * The flat world of a map in a projection registered with MapLibre's
 * `addProjection` (maplibre-gl >= 6.15). Such a map lays its tiles over a
 * square tile matrix set in the CRS's own plane, and custom layers take
 * positions in that square: [0, 0] is the top-left of tile 0/0/0 and [1, 1]
 * its bottom-right, with y growing downwards like tile rows. The layer
 * places regions there in place of normalized Web Mercator, and the tile
 * math it shares with Mercator (tile ids, clip boxes) carries over unchanged.
 */

import { customProjectionName } from './map-utils'
import type { MapLike } from './types'

/**
 * A projection registered with MapLibre's `addProjection`, in the shape of
 * its `CRS` type, so the same object can be passed to both.
 */
export interface MapProjection {
  /** The registered name, e.g. `EPSG:3031`. */
  name: string
  /** WGS84 degrees to CRS units and back, like a proj4 converter. */
  projection: {
    forward(position: number[]): number[]
    inverse(position: number[]): number[] | null
  }
  tileMatrixSet: {
    /** Top-left corner of tile 0/0/0, in CRS units. */
    origin: [number, number]
    /** Width of tile 0/0/0, in CRS units. */
    extentAtZoom0: number
  }
}

export interface WorldMapping {
  /** WGS84 degrees to the world square; NaN where the CRS has no answer. */
  toWorld(lng: number, lat: number): [number, number]
  /** World square to WGS84 degrees; null where the CRS has no answer. */
  toLngLat(x: number, y: number): [number, number] | null
}

const mappings = new WeakMap<MapProjection, WorldMapping>()

/**
 * How far, in world units, a round trip may land from where it started for
 * the inverse to count, MapLibre's own tolerance: loose enough for a
 * converter that is only approximately invertible, tight enough to reject an
 * answer from the other side of the world.
 */
const ROUND_TRIP_WORLD = 1e-3

function worldMappingOf(definition: MapProjection): WorldMapping {
  let mapping = mappings.get(definition)
  if (mapping) return mapping
  const {
    projection,
    tileMatrixSet: {
      origin: [originX, originY],
      extentAtZoom0: extent,
    },
  } = definition
  const toWorld = (lng: number, lat: number): [number, number] => {
    if (!Number.isFinite(lng) || !Number.isFinite(lat)) return [NaN, NaN]
    try {
      const [x, y] = projection.forward([lng, lat])
      return Number.isFinite(x) && Number.isFinite(y)
        ? [(x - originX) / extent, (originY - y) / extent]
        : [NaN, NaN]
    } catch {
      return [NaN, NaN]
    }
  }
  // An inverse that forward does not take back to the position has no
  // answer there, as MapLibre treats it.
  const toLngLat = (x: number, y: number): [number, number] | null => {
    try {
      const lngLat = projection.inverse([
        originX + x * extent,
        originY - y * extent,
      ])
      if (
        !lngLat ||
        !Number.isFinite(lngLat[0]) ||
        !Number.isFinite(lngLat[1])
      ) {
        return null
      }
      const [backX, backY] = toWorld(lngLat[0], lngLat[1])
      return Math.hypot(backX - x, backY - y) <= ROUND_TRIP_WORLD
        ? [lngLat[0], lngLat[1]]
        : null
    } catch {
      return null
    }
  }
  mapping = { toWorld, toLngLat }
  mappings.set(definition, mapping)
  return mapping
}

/**
 * The world of the map's projection when it is one registered with
 * `addProjection` and among the projections given, or null otherwise. The
 * same projection gives the same object each frame.
 */
export function worldMappingFor(
  map: MapLike | null,
  projections: readonly MapProjection[]
): WorldMapping | null {
  const name = customProjectionName(map?.getProjection?.())
  if (!name) return null
  const definition = projections.find((p) => p.name === name)
  return definition ? worldMappingOf(definition) : null
}

/** Source CRS to WGS84 and back, as the layer's projection context holds it. */
export interface LngLatTransformer {
  forward(x: number, y: number): [number, number]
  inverse(lng: number, lat: number): [number, number]
}

/** A box in the world square. */
export interface WorldBounds {
  x0: number
  y0: number
  x1: number
  y1: number
}

/** Source CRS to the world square, through WGS84; NaN where either has no answer. */
export function sourceToWorld(
  mapping: WorldMapping,
  to4326: LngLatTransformer
): (x: number, y: number) => [number, number] {
  return (x, y) => {
    const [lng, lat] = to4326.forward(x, y)
    return mapping.toWorld(lng, lat)
  }
}

/** World square to source CRS, through WGS84; NaN where either has no answer. */
export function worldToSource(
  mapping: WorldMapping,
  to4326: LngLatTransformer
): (x: number, y: number) => [number, number] {
  return (x, y) => {
    const lngLat = mapping.toLngLat(x, y)
    if (!lngLat) return [NaN, NaN]
    const source = to4326.inverse(lngLat[0], lngLat[1])
    return Number.isFinite(source[0]) && Number.isFinite(source[1])
      ? source
      : [NaN, NaN]
  }
}

/**
 * The world-square box around a source-CRS rectangle, from a grid of
 * samples across it, so a pole inside the rectangle counts. Its edges are
 * sampled twice as densely, and the box grows by half the farthest any
 * in-between sample bends from its neighbors, twice what a smooth edge can
 * bulge between samples, so an edge that curves out between them stays
 * inside. Null when no sample has a place in the world.
 */
export function worldBoundsOf(
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number },
  mapping: WorldMapping,
  to4326: LngLatTransformer,
  samples = 8
): WorldBounds | null {
  const toWorld = sourceToWorld(mapping, to4326)
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  const record = (wx: number, wy: number) => {
    if (!Number.isFinite(wx) || !Number.isFinite(wy)) return
    x0 = Math.min(x0, wx)
    y0 = Math.min(y0, wy)
    x1 = Math.max(x1, wx)
    y1 = Math.max(y1, wy)
  }
  const at = (u: number, v: number) =>
    toWorld(
      bounds.xMin + u * (bounds.xMax - bounds.xMin),
      bounds.yMin + v * (bounds.yMax - bounds.yMin)
    )
  for (let j = 0; j <= samples; j++) {
    for (let i = 0; i <= samples; i++) {
      record(...at(i / samples, j / samples))
    }
  }
  let bend = 0
  const steps = 2 * samples
  for (const edge of [
    (t: number) => at(t, 0),
    (t: number) => at(t, 1),
    (t: number) => at(0, t),
    (t: number) => at(1, t),
  ]) {
    const points = Array.from({ length: steps + 1 }, (_, k) => edge(k / steps))
    for (let k = 1; k < steps; k += 2) {
      const [ax, ay] = points[k - 1]
      const [mx, my] = points[k]
      const [bx, by] = points[k + 1]
      record(mx, my)
      const offset = Math.hypot(mx - (ax + bx) / 2, my - (ay + by) / 2)
      if (Number.isFinite(offset)) bend = Math.max(bend, offset)
    }
  }
  if (!Number.isFinite(x0)) return null
  const pad = bend / 2
  return { x0: x0 - pad, y0: y0 - pad, x1: x1 + pad, y1: y1 + pad }
}

/**
 * The share of the world square's width a source-CRS rectangle spans inside
 * it, from a grid of samples across it that land in the square. Points past
 * the square, such as a pole the projection sends far away, do not count.
 * Undefined when the samples in the square span no width.
 */
export function worldFractionOf(
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number },
  mapping: WorldMapping,
  to4326: LngLatTransformer,
  samples = 32
): number | undefined {
  const toWorld = sourceToWorld(mapping, to4326)
  let x0 = Infinity
  let x1 = -Infinity
  for (let j = 0; j <= samples; j++) {
    const y = bounds.yMin + (j / samples) * (bounds.yMax - bounds.yMin)
    for (let i = 0; i <= samples; i++) {
      const x = bounds.xMin + (i / samples) * (bounds.xMax - bounds.xMin)
      const [wx, wy] = toWorld(x, y)
      if (!(wx >= 0 && wx <= 1 && wy >= 0 && wy <= 1)) continue
      x0 = Math.min(x0, wx)
      x1 = Math.max(x1, wx)
    }
  }
  return x1 > x0 ? x1 - x0 : undefined
}
