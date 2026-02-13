/**
 * @module visible-regions-manager
 *
 * Manages asynchronous visible region calculations using a Web Worker.
 * Provides a non-blocking API for calculating which regions are visible
 * in the current viewport when using proj4 reprojection.
 *
 * Key features:
 * - Non-blocking: Main thread stays responsive during expensive proj4 calculations
 * - Debouncing: Only sends request when viewport stops changing (or after timeout)
 * - Caching: Returns previous results immediately while waiting for worker response
 */

import type {
  VisibleRegionsRequest,
  VisibleRegionsResponse,
  VisibleRegionsError,
  InitError,
  WorkerResponse,
} from './visible-regions-worker'

import { createVisibleRegionsWorker } from './visible-regions-worker-inline'

/** Region with bounds for worker calculation */
export interface RegionWithBounds {
  regionX: number
  regionY: number
  bounds: {
    xMin: number
    xMax: number
    yMin: number
    yMax: number
  }
}

/** Viewport bounds in WGS84 */
export interface Viewport {
  west: number
  south: number
  east: number
  north: number
}

/** Result of a visible regions calculation */
export interface VisibleRegionsResult {
  /** Cached regions from the last completed calculation */
  cachedRegions: Array<{ regionX: number; regionY: number }>
  /** Sequence number for this result, used to detect stale responses */
  sequence: number
}

/** Callback when new visible regions are calculated */
export type VisibleRegionsCallback = (result: VisibleRegionsResult) => void

/** Options for VisibleRegionsManager */
export interface VisibleRegionsManagerOptions {
  /** Callback invoked when new visible regions are calculated */
  onVisibleRegionsUpdate?: VisibleRegionsCallback
  /** Debounce delay in ms before sending request to worker (default: 50) */
  debounceMs?: number
}

/**
 * Manages asynchronous visible region calculations.
 */
export class VisibleRegionsManager {
  private worker: Worker | null = null
  private isReady = false
  private pendingInit: Promise<void> | null = null

  // Debouncing: track pending viewport and use timeout
  private pendingViewport: Viewport | null = null
  private pendingRegions: RegionWithBounds[] | null = null
  private debounceTimer: ReturnType<typeof setTimeout> | null = null
  private debounceMs: number

  // Track if a request is in-flight
  private requestInFlight = false
  private currentRequestId = 0
  private hasCalculatedOnce = false // Track if we've done the first calculation
  private resultSequence = 0 // Monotonically increasing sequence for results

  // Last known result (returned immediately while waiting for new worker calculation)
  private lastVisibleRegions: Array<{ regionX: number; regionY: number }> = []

  // Callback for async updates
  private onVisibleRegionsUpdate?: VisibleRegionsCallback

  // Current proj4 definition
  private proj4def: string | null = null

  constructor(options: VisibleRegionsManagerOptions = {}) {
    this.onVisibleRegionsUpdate = options.onVisibleRegionsUpdate
    this.debounceMs = options.debounceMs ?? 50 // 50ms debounce by default
  }

  /**
   * Initialize the worker with a proj4 definition.
   * Must be called before requestVisibleRegions.
   *
   * Note: Initialization is async and non-blocking. While the worker is
   * initializing, requestVisibleRegions() returns an empty array. Once ready,
   * the worker triggers a repaint via the onVisibleRegionsUpdate callback.
   *
   * @param proj4def - The proj4 projection definition string
   * @throws Error if worker initialization fails (e.g., invalid proj4def)
   */
  async init(proj4def: string): Promise<void> {
    this.proj4def = proj4def

    // If already initializing, wait for that
    if (this.pendingInit) {
      await this.pendingInit
      return
    }

    this.pendingInit = this.initWorker(proj4def)
    await this.pendingInit
    this.pendingInit = null
  }

  private async initWorker(proj4def: string): Promise<void> {
    // Terminate existing worker if any
    if (this.worker) {
      this.worker.terminate()
      this.worker = null
      this.isReady = false
    }

    return new Promise((resolve, reject) => {
      try {
        // Create worker using inline code (embedded in the library)
        const worker = createVisibleRegionsWorker()

        this.worker = worker

        this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
          const message = event.data

          switch (message.type) {
            case 'ready':
              this.isReady = true
              resolve()
              break

            case 'visibleRegionsResult':
              this.handleResult(message)
              break

            case 'visibleRegionsError':
              this.handleError(message)
              break

            case 'initError':
              console.error(
                '[VisibleRegionsManager] Init error:',
                message.error
              )
              reject(new Error(message.error))
              break
          }
        }

        this.worker.onerror = (error) => {
          console.error('[VisibleRegionsManager] Worker error:', error)
          if (!this.isReady) {
            reject(error)
          }
        }

        // Initialize the worker with proj4 definition
        this.worker.postMessage({ type: 'init', proj4def })
      } catch (err) {
        reject(err)
      }
    })
  }

  private handleResult(response: VisibleRegionsResponse): void {
    this.requestInFlight = false
    this.lastVisibleRegions = response.visibleRegions
    this.resultSequence++ // Increment sequence for new result

    // Call the update callback
    if (this.onVisibleRegionsUpdate) {
      this.onVisibleRegionsUpdate({
        cachedRegions: response.visibleRegions,
        sequence: this.resultSequence,
      })
    }

    // If there's a pending viewport that came in while we were calculating,
    // send it now
    if (this.pendingViewport && this.pendingRegions) {
      this.sendRequest(this.pendingViewport, this.pendingRegions)
      this.pendingViewport = null
      this.pendingRegions = null
    }
  }

  private handleError(error: VisibleRegionsError): void {
    this.requestInFlight = false
    console.error(
      '[VisibleRegionsManager] Calculation error:',
      error.error,
      'requestId:',
      error.requestId
    )
  }

  /**
   * Send request to worker immediately
   */
  private sendRequest(
    viewport: Viewport,
    allRegions: RegionWithBounds[]
  ): void {
    if (!this.worker || !this.proj4def) return

    this.requestInFlight = true
    this.currentRequestId++

    const request: VisibleRegionsRequest = {
      type: 'calculateVisibleRegions',
      requestId: this.currentRequestId,
      proj4def: this.proj4def,
      viewport,
      regions: allRegions,
    }

    this.worker.postMessage(request)
  }

  /**
   * Request visible regions calculation asynchronously.
   *
   * Returns immediately with cached results from the previous calculation.
   * When the worker completes, onVisibleRegionsUpdate callback is invoked.
   *
   * @param viewport - Current viewport bounds in WGS84
   * @param allRegions - All possible regions with their bounds in source CRS
   * @returns Immediate result with cached regions from last calculation
   */
  requestVisibleRegions(
    viewport: Viewport,
    allRegions: RegionWithBounds[]
  ): VisibleRegionsResult {
    // If worker not ready, return last known result
    if (!this.isReady || !this.worker || !this.proj4def) {
      return {
        cachedRegions: this.lastVisibleRegions,
        sequence: this.resultSequence,
      }
    }

    // If a request is already in-flight, queue this viewport for later
    if (this.requestInFlight) {
      this.pendingViewport = viewport
      this.pendingRegions = allRegions
      return {
        cachedRegions: this.lastVisibleRegions,
        sequence: this.resultSequence,
      }
    }

    // First request: send immediately without debounce for fast initial load
    if (!this.hasCalculatedOnce) {
      this.hasCalculatedOnce = true
      this.sendRequest(viewport, allRegions)
      return {
        cachedRegions: this.lastVisibleRegions,
        sequence: this.resultSequence,
      }
    }

    // Subsequent requests: debounce to avoid flooding worker during panning
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer)
    }

    this.pendingViewport = viewport
    this.pendingRegions = allRegions

    this.debounceTimer = setTimeout(() => {
      this.debounceTimer = null
      // Guard against race where worker was disposed or not ready between setTimeout and callback
      if (!this.worker || !this.isReady) return
      if (this.pendingViewport && this.pendingRegions) {
        this.sendRequest(this.pendingViewport, this.pendingRegions)
        this.pendingViewport = null
        this.pendingRegions = null
      }
    }, this.debounceMs)

    // Return last known result immediately
    return {
      cachedRegions: this.lastVisibleRegions,
      sequence: this.resultSequence,
    }
  }

  /**
   * Clear stored results and reset calculation state.
   * Call this when region layout changes (e.g., level switch).
   * The next requestVisibleRegions() call will send immediately (no debounce).
   */
  clear(): void {
    this.lastVisibleRegions = []
    this.hasCalculatedOnce = false // Reset so next calculation sends immediately
  }

  /**
   * Set or update the callback for async updates.
   *
   * @param callback - Function called when worker completes a calculation,
   *                   or undefined to remove the callback
   */
  setCallback(callback: VisibleRegionsCallback | undefined): void {
    this.onVisibleRegionsUpdate = callback
  }

  /**
   * Check if the worker is ready to process requests.
   *
   * @returns true if the worker has been initialized and is ready
   */
  get ready(): boolean {
    return this.isReady
  }

  /**
   * Dispose of the worker and release all resources.
   * Call this when the manager is no longer needed (e.g., layer removal).
   * After disposal, the manager cannot be reused.
   */
  dispose(): void {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer)
      this.debounceTimer = null
    }
    if (this.worker) {
      this.worker.terminate()
      this.worker = null
    }
    this.isReady = false
    this.lastVisibleRegions = []
    this.hasCalculatedOnce = false
    this.proj4def = null
  }
}
