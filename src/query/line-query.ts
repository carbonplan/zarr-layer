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
 * repeats removed and cells outside the grid dropped.
 *
 * `distance` accumulates along the whole path, including parts outside the
 * grid, so a line that leaves and re-enters the raster keeps a continuous
 * axis. `distanceOffset` seeds the accumulator for a path that continues an
 * earlier one, and `endDistance` is where the next continuation should start.
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

  const emit = (x: number, y: number, lon: number, lat: number) => {
    distance += haversineMeters(prevLon, prevLat, lon, lat)
    prevLon = lon
    prevLat = lat
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
    const dx = b.px - a.px
    const dy = b.py - a.py

    let x = Math.floor(a.px)
    let y = Math.floor(a.py)
    // A start on a cell boundary heading down belongs to the lower cell.
    if (dx < 0 && x === a.px) x -= 1
    if (dy < 0 && y === a.py) y -= 1

    const stepX = dx > 0 ? 1 : dx < 0 ? -1 : 0
    const stepY = dy > 0 ? 1 : dy < 0 ? -1 : 0
    let tMaxX =
      dx > 0 ? (x + 1 - a.px) / dx : dx < 0 ? (x - a.px) / dx : Infinity
    let tMaxY =
      dy > 0 ? (y + 1 - a.py) / dy : dy < 0 ? (y - a.py) / dy : Infinity
    const tDeltaX = dx !== 0 ? Math.abs(1 / dx) : Infinity
    const tDeltaY = dy !== 0 ? Math.abs(1 / dy) : Infinity

    emit(x, y, a.lon, a.lat)

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
      emit(x, y, a.lon + t * (b.lon - a.lon), a.lat + t * (b.lat - a.lat))
    }

    // Close out the segment so distance reaches the vertex itself.
    distance += haversineMeters(prevLon, prevLat, b.lon, b.lat)
    prevLon = b.lon
    prevLat = b.lat
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

/** Total great-circle length of a lon/lat line in meters. */
export function lineLengthMeters(coords: number[][]): number {
  let total = 0
  for (let i = 1; i < coords.length; i++) {
    total += haversineMeters(
      coords[i - 1][0],
      coords[i - 1][1],
      coords[i][0],
      coords[i][1]
    )
  }
  return total
}
