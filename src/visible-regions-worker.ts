/**
 * Web Worker for calculating visible regions with proj4 reprojection.
 *
 * This worker offloads the expensive proj4 inverse transformations from
 * the main thread, preventing UI jank during panning with non-Web-Mercator
 * CRS datasets.
 *
 * Message protocol:
 * - Main → Worker: VisibleRegionsRequest
 * - Worker → Main: VisibleRegionsResponse
 */

import proj4 from 'proj4'

/** Request message from main thread */
export interface VisibleRegionsRequest {
  type: 'calculateVisibleRegions'
  requestId: number
  proj4def: string
  viewport: {
    west: number
    south: number
    east: number
    north: number
  }
  regions: Array<{
    regionX: number
    regionY: number
    bounds: {
      xMin: number
      xMax: number
      yMin: number
      yMax: number
    }
  }>
}

/** Response message to main thread */
export interface VisibleRegionsResponse {
  type: 'visibleRegionsResult'
  requestId: number
  visibleRegions: Array<{ regionX: number; regionY: number }>
}

/** Error response */
export interface VisibleRegionsError {
  type: 'visibleRegionsError'
  requestId: number
  error: string
}

/** Initialization error response */
export interface InitError {
  type: 'initError'
  error: string
}

export type WorkerMessage =
  | VisibleRegionsRequest
  | { type: 'init'; proj4def: string }

export type WorkerResponse =
  | VisibleRegionsResponse
  | VisibleRegionsError
  | InitError
  | { type: 'ready' }

// Cache the transformer to avoid recreating it for each request
let cachedProj4def: string | null = null
let cachedTransformer: {
  inverse: (x: number, y: number) => [number, number]
} | null = null

type Transformer = { inverse: (x: number, y: number) => [number, number] }

function getTransformer(proj4def: string): Transformer {
  if (proj4def !== cachedProj4def || !cachedTransformer) {
    const converter = proj4(proj4def, 'EPSG:4326')
    cachedTransformer = {
      inverse: (x: number, y: number): [number, number] => {
        const result = converter.forward([x, y])
        return [result[0], result[1]]
      },
    }
    cachedProj4def = proj4def
  }
  return cachedTransformer
}

function calculateVisibleRegions(
  request: VisibleRegionsRequest
): VisibleRegionsResponse {
  const { proj4def, viewport, regions, requestId } = request
  const { west, south, east, north } = viewport

  const transformer = getTransformer(proj4def)
  const visibleRegions: Array<{ regionX: number; regionY: number }> = []

  for (const region of regions) {
    const { regionX, regionY, bounds } = region
    const xMid = (bounds.xMin + bounds.xMax) / 2
    const yMid = (bounds.yMin + bounds.yMax) / 2

    // Transform region corners and edge midpoints to WGS84
    const samplePoints = [
      // Corners
      transformer.inverse(bounds.xMin, bounds.yMin),
      transformer.inverse(bounds.xMax, bounds.yMin),
      transformer.inverse(bounds.xMax, bounds.yMax),
      transformer.inverse(bounds.xMin, bounds.yMax),
      // Edge midpoints
      transformer.inverse(xMid, bounds.yMin),
      transformer.inverse(xMid, bounds.yMax),
      transformer.inverse(bounds.xMin, yMid),
      transformer.inverse(bounds.xMax, yMid),
    ]

    // Filter out invalid points
    const validPoints = samplePoints.filter(
      (c) => isFinite(c[0]) && isFinite(c[1])
    )
    if (validPoints.length === 0) {
      continue
    }

    // Get geographic bounds of this region
    const regWest = Math.min(...validPoints.map((c) => c[0]))
    const regEast = Math.max(...validPoints.map((c) => c[0]))
    const regSouth = Math.min(...validPoints.map((c) => c[1]))
    const regNorth = Math.max(...validPoints.map((c) => c[1]))

    // Check if region overlaps with viewport
    if (
      regEast >= west &&
      regWest <= east &&
      regNorth >= south &&
      regSouth <= north
    ) {
      visibleRegions.push({ regionX, regionY })
    }
  }

  return {
    type: 'visibleRegionsResult',
    requestId,
    visibleRegions,
  }
}

// Worker message handler
self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const message = event.data

  switch (message.type) {
    case 'init':
      // Pre-initialize the transformer
      try {
        getTransformer(message.proj4def)
        self.postMessage({ type: 'ready' })
      } catch (err) {
        self.postMessage({
          type: 'initError',
          error: err instanceof Error ? err.message : String(err),
        } as InitError)
      }
      break

    case 'calculateVisibleRegions':
      try {
        const response = calculateVisibleRegions(message)
        self.postMessage(response)
      } catch (err) {
        self.postMessage({
          type: 'visibleRegionsError',
          requestId: message.requestId,
          error: err instanceof Error ? err.message : String(err),
        } as VisibleRegionsError)
      }
      break
  }
}
