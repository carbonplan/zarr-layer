/**
 * @module line-query
 *
 * Geometry for LineString queries: antimeridian splitting, cell-by-cell
 * tracing through the pixel grid, and grouping traced cells into fetch runs.
 */

import type { Bounds } from '../types'
import {
  densifyAndTransformPath,
  lonLatToPixel,
  type CachedTransformer,
  type DensifiedVertex,
  type PixelRect,
} from './query-utils'

const EARTH_RADIUS_M = 6371008.8

/** Great-circle distance in meters between two lon/lat positions. */
export function haversineMeters(
  lon0: number,
  lat0: number,
  lon1: number,
  lat1: number
): number {
  const toRad = Math.PI / 180
  const dLat = (lat1 - lat0) * toRad
  const dLon = (lon1 - lon0) * toRad
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat0 * toRad) * Math.cos(lat1 * toRad) * Math.sin(dLon / 2) ** 2
  return 2 * EARTH_RADIUS_M * Math.asin(Math.min(1, Math.sqrt(a)))
}

/** Longest lon/lat span measured as a single great-circle chord. */
const MAX_CHORD_DEGREES = 0.25

/**
 * Length in meters of the straight lon/lat segment between two positions.
 *
 * Query edges are straight in lon/lat, which is not a great circle, so long
 * spans are measured as a chain of short chords. The result does not depend
 * on how finely a caller happens to split the segment.
 */
export function segmentLengthMeters(
  lon0: number,
  lat0: number,
  lon1: number,
  lat1: number
): number {
  const span = Math.max(Math.abs(lon1 - lon0), Math.abs(lat1 - lat0))
  const steps = Math.max(1, Math.ceil(span / MAX_CHORD_DEGREES))
  if (steps === 1) return haversineMeters(lon0, lat0, lon1, lat1)

  let total = 0
  let prevLon = lon0
  let prevLat = lat0
  for (let i = 1; i <= steps; i++) {
    const t = i / steps
    const lon = lon0 + t * (lon1 - lon0)
    const lat = lat0 + t * (lat1 - lat0)
    total += haversineMeters(prevLon, prevLat, lon, lat)
    prevLon = lon
    prevLat = lat
  }
  return total
}

/**
 * Liang–Barsky clip of a segment against the rectangle [0, width] x
 * [0, height]. Returns the parameter range of the part inside, or null when
 * the segment misses the rectangle.
 */
function clipSegmentToGrid(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  width: number,
  height: number
): [number, number] | null {
  const dx = bx - ax
  const dy = by - ay
  let t0 = 0
  let t1 = 1
  const edges: [number, number][] = [
    [-dx, ax],
    [dx, width - ax],
    [-dy, ay],
    [dy, height - ay],
  ]
  for (const [p, q] of edges) {
    if (p === 0) {
      if (q < 0) return null
      continue
    }
    const r = q / p
    if (p < 0) {
      if (r > t1) return null
      if (r > t0) t0 = r
    } else {
      if (r < t0) return null
      if (r < t1) t1 = r
    }
  }
  return [t0, t1]
}

/**
 * Split a lon/lat line at the antimeridian into ordered pieces whose
 * longitudes all lie within [-180, 180].
 *
 * Longitude semantics match polygon queries: a line whose vertices all lie in
 * [-180, 180] is read literally, so a segment from 170 to -170 runs across
 * the prime meridian. Any vertex outside that range marks an explicit
 * crossing, and edges are unwrapped for continuity before splitting.
 */
export function splitLineAtAntimeridian(coords: number[][]): number[][][] {
  if (coords.length === 0) return []

  const explicit = coords.some(([lon]) => lon > 180 || lon < -180)
  const unwrapped: number[][] = [[coords[0][0], coords[0][1]]]
  for (let i = 1; i < coords.length; i++) {
    let lon = coords[i][0]
    if (explicit) {
      const delta = lon - unwrapped[i - 1][0]
      if (delta > 180) lon -= 360
      else if (delta < -180) lon += 360
    }
    unwrapped.push([lon, coords[i][1]])
  }

  const pieces: number[][][] = []
  // Frame k holds longitudes in (k*360 - 180, k*360 + 180], except frame 0
  // which also owns -180, so both ends of the literal range stay unsplit.
  const frameOf = (lon: number) => {
    const f = Math.floor((lon + 180) / 360)
    return lon - f * 360 === -180 && f > 0 ? f - 1 : f
  }
  let piece: number[][] = []
  let frame = frameOf(unwrapped[0][0])
  const shift = (lon: number, f: number) => lon - f * 360

  piece.push([shift(unwrapped[0][0], frame), unwrapped[0][1]])
  for (let i = 1; i < unwrapped.length; i++) {
    const [lon0, lat0] = unwrapped[i - 1]
    const [lon1, lat1] = unwrapped[i]
    const targetFrame = frameOf(lon1)
    // Walk across every frame boundary the edge crosses, in order.
    while (frame !== targetFrame) {
      const step = targetFrame > frame ? 1 : 0
      const boundary = frame * 360 + (step ? 180 : -180)
      const t = (boundary - lon0) / (lon1 - lon0)
      const latB = lat0 + t * (lat1 - lat0)
      piece.push([shift(boundary, frame), latB])
      pieces.push(piece)
      frame += targetFrame > frame ? 1 : -1
      piece = [[shift(boundary, frame), latB]]
    }
    piece.push([shift(lon1, frame), lat1])
  }
  pieces.push(piece)

  if (pieces.length === 1) return pieces
  const isDegenerate = (p: number[][]) =>
    p.every(([lon, lat]) => lon === p[0][0] && lat === p[0][1])
  return pieces.filter((p) => !isDegenerate(p))
}

/** A grid cell the line passes through, with distance along the line. */
export interface TracedCell {
  x: number
  y: number
  /** Meters along the line from its start to where the line enters the cell. */
  distance: number
}

/**
 * Project a lon/lat line into the raster's pixel grid, densifying edges so
 * curvature under nonlinear projections is preserved.
 */
export function lineToPixelPath(
  coords: number[][],
  sourceBounds: Bounds,
  width: number,
  height: number,
  proj4def: string,
  latIsAscending?: boolean,
  cachedTransformer?: CachedTransformer
): DensifiedVertex[] {
  return densifyAndTransformPath(coords, (lon, lat) => {
    const px = lonLatToPixel(
      lon,
      lat,
      sourceBounds,
      width,
      height,
      proj4def,
      latIsAscending,
      cachedTransformer
    )
    return px ?? [NaN, NaN]
  })
}

/**
 * Walk a pixel-space path cell by cell (Amanatides–Woo grid traversal) and
 * return every cell it passes through, in path order, with consecutive
 * repeats removed.
 *
 * Each segment is clipped to the grid before it is walked, so the cost
 * follows the cells returned and not the length of the line. `distance`
 * still accumulates along the whole path, including the clipped-away parts,
 * so a line that leaves and re-enters the raster keeps a continuous axis.
 * `distanceOffset` seeds the accumulator for a path that continues an earlier
 * one, and `endDistance` is where the next continuation should start.
 */
export function traceLineCells(
  path: DensifiedVertex[],
  width: number,
  height: number,
  distanceOffset = 0
): { cells: TracedCell[]; endDistance: number } {
  const cells: TracedCell[] = []
  if (path.length === 0) return { cells, endDistance: distanceOffset }

  let lastX = NaN
  let lastY = NaN
  let distance = distanceOffset
  let prevLon = path[0].lon
  let prevLat = path[0].lat

  const advanceTo = (lon: number, lat: number) => {
    distance += segmentLengthMeters(prevLon, prevLat, lon, lat)
    prevLon = lon
    prevLat = lat
  }
  // A re-entry after time off the grid is a new sample even in the same cell.
  const forgetLastCell = () => {
    lastX = NaN
    lastY = NaN
  }

  const emit = (x: number, y: number, lon: number, lat: number) => {
    advanceTo(lon, lat)
    if (x === lastX && y === lastY) return
    lastX = x
    lastY = y
    if (x < 0 || y < 0 || x >= width || y >= height) return
    cells.push({ x, y, distance })
  }

  if (path.length === 1) {
    const v = path[0]
    emit(Math.floor(v.px), Math.floor(v.py), v.lon, v.lat)
    return { cells, endDistance: distance }
  }

  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i]
    const b = path[i + 1]

    const clip = clipSegmentToGrid(a.px, a.py, b.px, b.py, width, height)
    if (!clip) {
      forgetLastCell()
      advanceTo(b.lon, b.lat)
      continue
    }
    const [t0, t1] = clip
    const at = (t: number) => ({
      px: a.px + t * (b.px - a.px),
      py: a.py + t * (b.py - a.py),
      lon: a.lon + t * (b.lon - a.lon),
      lat: a.lat + t * (b.lat - a.lat),
    })
    const start = t0 > 0 ? at(t0) : a
    const end = t1 < 1 ? at(t1) : b
    if (t0 > 0) forgetLastCell()

    const dx = end.px - start.px
    const dy = end.py - start.py

    let x = Math.floor(start.px)
    let y = Math.floor(start.py)
    // A start on a cell boundary heading down belongs to the lower cell.
    if (dx < 0 && x === start.px) x -= 1
    if (dy < 0 && y === start.py) y -= 1

    const stepX = dx > 0 ? 1 : dx < 0 ? -1 : 0
    const stepY = dy > 0 ? 1 : dy < 0 ? -1 : 0
    let tMaxX =
      dx > 0 ? (x + 1 - start.px) / dx : dx < 0 ? (x - start.px) / dx : Infinity
    let tMaxY =
      dy > 0 ? (y + 1 - start.py) / dy : dy < 0 ? (y - start.py) / dy : Infinity
    const tDeltaX = dx !== 0 ? Math.abs(1 / dx) : Infinity
    const tDeltaY = dy !== 0 ? Math.abs(1 / dy) : Infinity

    emit(x, y, start.lon, start.lat)

    for (;;) {
      let t: number
      if (tMaxX < tMaxY) {
        if (tMaxX >= 1) break
        t = tMaxX
        x += stepX
        tMaxX += tDeltaX
      } else {
        if (tMaxY >= 1) break
        t = tMaxY
        y += stepY
        tMaxY += tDeltaY
      }
      emit(
        x,
        y,
        start.lon + t * (end.lon - start.lon),
        start.lat + t * (end.lat - start.lat)
      )
    }

    if (t1 < 1) forgetLastCell()
    advanceTo(b.lon, b.lat)
  }

  return { cells, endDistance: distance }
}

/** Consecutive traced cells that fit in one fetch window. */
export interface LineRun {
  rect: PixelRect
  cells: TracedCell[]
}

/**
 * Group traced cells, in order, into runs whose bounding rectangle spans at
 * most `maxSpan` pixels per axis.
 */
export function groupCellsIntoRuns(
  cells: TracedCell[],
  maxSpan: number
): LineRun[] {
  const runs: LineRun[] = []
  let current: TracedCell[] = []
  let minX = 0
  let maxX = 0
  let minY = 0
  let maxY = 0

  const flush = () => {
    if (current.length === 0) return
    runs.push({
      rect: { minX, maxX: maxX + 1, minY, maxY: maxY + 1 },
      cells: current,
    })
    current = []
  }

  for (const cell of cells) {
    if (current.length === 0) {
      minX = maxX = cell.x
      minY = maxY = cell.y
      current.push(cell)
      continue
    }
    const nextMinX = Math.min(minX, cell.x)
    const nextMaxX = Math.max(maxX, cell.x)
    const nextMinY = Math.min(minY, cell.y)
    const nextMaxY = Math.max(maxY, cell.y)
    if (nextMaxX - nextMinX >= maxSpan || nextMaxY - nextMinY >= maxSpan) {
      flush()
      minX = maxX = cell.x
      minY = maxY = cell.y
      current.push(cell)
      continue
    }
    minX = nextMinX
    maxX = nextMaxX
    minY = nextMinY
    maxY = nextMaxY
    current.push(cell)
  }
  flush()

  return runs
}

/** Total length of a lon/lat line in meters. */
export function lineLengthMeters(coords: number[][]): number {
  let total = 0
  for (let i = 1; i < coords.length; i++) {
    total += segmentLengthMeters(
      coords[i - 1][0],
      coords[i - 1][1],
      coords[i][0],
      coords[i][1]
    )
  }
  return total
}
