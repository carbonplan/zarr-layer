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
 * Most chords one segment is measured in. Validated query input never needs
 * more, and the cap keeps any other caller's cost bounded.
 */
const MAX_CHORDS = 8192

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
  if (!Number.isFinite(span)) return NaN
  const steps = Math.min(
    MAX_CHORDS,
    Math.max(1, Math.ceil(span / MAX_CHORD_DEGREES))
  )
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

interface ClippedSegment {
  t0: number
  t1: number
  /** Grid edge the segment enters and leaves through, when it is clipped. */
  entry: GridEdge | null
  exit: GridEdge | null
}

type GridEdge = 'left' | 'right' | 'top' | 'bottom'

/**
 * Liang–Barsky clip of a segment against the rectangle [0, width] x
 * [0, height]. Returns the parameter range of the part inside and the edges
 * it was cut at, or null when the segment misses the rectangle.
 */
function clipSegmentToGrid(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  width: number,
  height: number
): ClippedSegment | null {
  const dx = bx - ax
  const dy = by - ay
  const clip: ClippedSegment = { t0: 0, t1: 1, entry: null, exit: null }
  const edges: [number, number, GridEdge][] = [
    [-dx, ax, 'left'],
    [dx, width - ax, 'right'],
    [-dy, ay, 'top'],
    [dy, height - ay, 'bottom'],
  ]
  for (const [p, q, edge] of edges) {
    if (p === 0) {
      if (q < 0) return null
      continue
    }
    const r = q / p
    if (p < 0) {
      if (r > clip.t1) return null
      if (r > clip.t0) {
        clip.t0 = r
        clip.entry = edge
      }
    } else {
      if (r < clip.t0) return null
      if (r < clip.t1) {
        clip.t1 = r
        clip.exit = edge
      }
    }
  }
  return clip
}

/**
 * Reject line coordinates the tracer cannot walk in bounded time: anything
 * non-finite, latitudes off the globe, and longitudes far enough out that
 * unwrapping them would mean circling it many times over.
 */
export function validateLineCoordinates(coords: number[][]): void {
  for (const position of coords) {
    const lon = position?.[0]
    const lat = position?.[1]
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
      throw new RangeError(
        '[ZarrLayer] LineString coordinates must be finite [lon, lat] pairs'
      )
    }
    if (Math.abs(lon) > MAX_LINE_LONGITUDE) {
      throw new RangeError(
        `[ZarrLayer] LineString longitudes must lie within ±${MAX_LINE_LONGITUDE}`
      )
    }
    if (Math.abs(lat) > 90) {
      throw new RangeError(
        '[ZarrLayer] LineString latitudes must lie within ±90'
      )
    }
  }
}

const MAX_LINE_LONGITUDE = 720

/**
 * Split a lon/lat line into ordered pieces whose longitudes all lie within
 * one 360-degree frame, [westEdge, westEdge + 360]. The default frame is
 * [-180, 180], which splits at the antimeridian. A raster whose extent
 * reaches past ±180 passes its own west edge, so pieces land on its grid.
 *
 * Longitude semantics match polygon queries: a line whose vertices all lie in
 * [-180, 180] is read literally, so a segment from 170 to -170 runs across
 * the prime meridian. Any vertex outside that range marks an explicit
 * crossing, and edges are unwrapped for continuity before splitting.
 *
 * A line that only touches a frame boundary and turns back is not split.
 */
export function splitLineAtAntimeridian(
  coords: number[][],
  westEdge = -180
): number[][][] {
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

  // Frame k holds (westEdge + 360k, westEdge + 360(k + 1)], except frame 0
  // which also owns its west edge, so both ends of that range stay unsplit.
  const frameOf = (lon: number) => {
    const f = Math.floor((lon - westEdge) / 360)
    return lon - f * 360 === westEdge && f > 0 ? f - 1 : f
  }
  const shift = (lon: number, f: number) => lon - f * 360

  const pieces: { frame: number; coords: number[][] }[] = []
  let frame = frameOf(unwrapped[0][0])
  let piece: number[][] = [[shift(unwrapped[0][0], frame), unwrapped[0][1]]]
  for (let i = 1; i < unwrapped.length; i++) {
    const [lon0, lat0] = unwrapped[i - 1]
    const [lon1, lat1] = unwrapped[i]
    const targetFrame = frameOf(lon1)
    // Walk across every frame boundary the edge crosses, in order.
    while (frame !== targetFrame) {
      const up = targetFrame > frame
      const boundary = westEdge + 360 * (up ? frame + 1 : frame)
      const t = (boundary - lon0) / (lon1 - lon0)
      const latB = lat0 + t * (lat1 - lat0)
      piece.push([shift(boundary, frame), latB])
      pieces.push({ frame, coords: piece })
      frame += up ? 1 : -1
      piece = [[shift(boundary, frame), latB]]
    }
    piece.push([shift(lon1, frame), lat1])
  }
  pieces.push({ frame, coords: piece })

  if (pieces.length === 1) return [pieces[0].coords]

  // A piece that is a single repeated point is a touch of the boundary, not
  // a crossing. Dropping it leaves its neighbours in the same frame, and
  // they rejoin into one continuous piece.
  const isPoint = (p: number[][]) =>
    p.every(([lon, lat]) => lon === p[0][0] && lat === p[0][1])
  const merged: { frame: number; coords: number[][] }[] = []
  for (const current of pieces) {
    if (isPoint(current.coords)) continue
    const previous = merged[merged.length - 1]
    if (previous && previous.frame === current.frame) {
      previous.coords = [...previous.coords, ...current.coords]
    } else {
      merged.push({ frame: current.frame, coords: [...current.coords] })
    }
  }
  if (merged.length === 0) return [pieces[0].coords]
  return merged.map((p) => p.coords)
}

/** A grid cell the line passes through, with distance along the line. */
export interface TracedCell {
  x: number
  y: number
  /** Meters along the line from its start to where the line enters the cell. */
  distance: number
}

/** A stretch of a line that projects onto the raster's grid. */
export interface PixelPathSection {
  path: DensifiedVertex[]
  /** Meters of line skipped between the previous section and this one. */
  gapBefore: number
}

/**
 * Project a lon/lat line into the raster's pixel grid, densifying edges so
 * curvature under nonlinear projections is preserved.
 *
 * A vertex the projection cannot place breaks the line there. The edges on
 * either side of it are left out, never bridged, and their length is carried
 * in `gapBefore` and `trailingGap` so distance stays continuous.
 */
export function lineToPixelPaths(
  coords: number[][],
  sourceBounds: Bounds,
  width: number,
  height: number,
  proj4def: string,
  latIsAscending?: boolean,
  cachedTransformer?: CachedTransformer
): { sections: PixelPathSection[]; trailingGap: number } {
  const transformVertex = (lon: number, lat: number): [number, number] => {
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
  }
  const projects = coords.map(([lon, lat]) => {
    const [px, py] = transformVertex(lon, lat)
    return Number.isFinite(px) && Number.isFinite(py)
  })

  const sections: PixelPathSection[] = []
  let gap = 0
  let gapBeforeRun = 0
  let run: number[][] = []
  const flush = () => {
    if (run.length === 0) return
    sections.push({
      path: densifyAndTransformPath(run, transformVertex),
      gapBefore: gapBeforeRun,
    })
    run = []
  }

  for (let i = 0; i < coords.length; i++) {
    if (i > 0 && !(projects[i] && projects[i - 1])) {
      gap += segmentLengthMeters(
        coords[i - 1][0],
        coords[i - 1][1],
        coords[i][0],
        coords[i][1]
      )
    }
    if (!projects[i]) {
      flush()
      continue
    }
    if (run.length === 0) {
      gapBeforeRun = gap
      gap = 0
    }
    run.push(coords[i])
  }
  flush()

  return { sections, trailingGap: gap }
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

  const isSinglePoint = path.every(
    (v) => v.px === path[0].px && v.py === path[0].py
  )
  if (isSinglePoint) {
    const v = path[0]
    emit(Math.floor(v.px), Math.floor(v.py), v.lon, v.lat)
    const last = path[path.length - 1]
    advanceTo(last.lon, last.lat)
    return { cells, endDistance: distance }
  }

  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i]
    const b = path[i + 1]

    // A repeated vertex covers no ground, so it visits nothing new.
    if (a.px === b.px && a.py === b.py) {
      advanceTo(b.lon, b.lat)
      continue
    }

    const clip = clipSegmentToGrid(a.px, a.py, b.px, b.py, width, height)
    if (!clip) {
      forgetLastCell()
      advanceTo(b.lon, b.lat)
      continue
    }
    const { t0, t1 } = clip
    // A clipped end sits exactly on the edge it was cut at, however far
    // outside the grid the segment started.
    const at = (t: number, edge: GridEdge | null): DensifiedVertex => {
      const v = {
        px: a.px + t * (b.px - a.px),
        py: a.py + t * (b.py - a.py),
        lon: a.lon + t * (b.lon - a.lon),
        lat: a.lat + t * (b.lat - a.lat),
      }
      if (edge === 'left') v.px = 0
      else if (edge === 'right') v.px = width
      else if (edge === 'top') v.py = 0
      else if (edge === 'bottom') v.py = height
      return v
    }
    const start = clip.entry ? at(t0, clip.entry) : a
    const end = clip.exit ? at(t1, clip.exit) : b
    if (clip.entry) forgetLastCell()

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
      const t = Math.min(tMaxX, tMaxY)
      if (t >= 1) break
      // Through an exact cell corner the line steps diagonally, so it never
      // samples a cell it only touches at a point.
      if (tMaxX === t) {
        x += stepX
        tMaxX += tDeltaX
      }
      if (tMaxY === t) {
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

    if (clip.exit) forgetLastCell()
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
