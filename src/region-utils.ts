/**
 * @module region-utils
 *
 * Shared utilities for the region renderer (RegionRenderer).
 * Provides common patterns for throttling, request cancellation, and loading state management.
 */

import type { LoadingStateCallback, LoadingState } from './types'

// ============================================================================
// Request Cancellation
// ============================================================================

/**
 * Region fetches in flight at once, per layer. Each landing region is
 * decoded and assembled on the main thread, so starting a heavy viewport all
 * together makes one long burst of that work; a limit spreads it across
 * frames. Sixteen is enough for an ordinary viewport to start at once.
 */
export const MAX_ACTIVE_REGION_FETCHES = 16

/** A region fetch waiting to start. */
export interface QueuedFetch {
  key: string
  regionX: number
  regionY: number
  /** Begin the fetch; settles when it has finished or been aborted. */
  start: () => Promise<void>
  /** Settle without fetching. */
  drop: () => void
}

export interface RequestCanceller {
  controllers: Map<number, AbortController>
  currentVersion: number
  /** Fetches waiting to start, in the order they will start. */
  queue: QueuedFetch[]
  /** Fetches started and not yet settled. */
  activeCount: number
  maxActive: number
  pumpScheduled: boolean
}

export function createRequestCanceller(
  maxActive: number = MAX_ACTIVE_REGION_FETCHES
): RequestCanceller {
  return {
    controllers: new Map(),
    currentVersion: 0,
    queue: [],
    activeCount: 0,
    maxActive,
    pumpScheduled: false,
  }
}

/**
 * Cancel all pending requests, queued ones included.
 */
export function cancelAllRequests(canceller: RequestCanceller): void {
  for (const controller of canceller.controllers.values()) {
    controller.abort()
  }
  canceller.controllers.clear()
  dropQueuedFetches(canceller, () => true)
}

/**
 * Check if any requests are still pending (not aborted), queued ones
 * included.
 */
export function hasActiveRequests(canceller: RequestCanceller): boolean {
  if (canceller.queue.length > 0) return true
  for (const controller of canceller.controllers.values()) {
    if (!controller.signal.aborted) return true
  }
  return false
}

/** Queue region fetches and schedule them to start. */
export function enqueueFetches(
  canceller: RequestCanceller,
  fetches: QueuedFetch[]
): void {
  canceller.queue.push(...fetches)
  scheduleFetches(canceller)
}

/**
 * Start queued fetches in a later task, never inside the render callback
 * that queued them, up to `maxActive` in flight. Fetches started in one pass
 * share a microtask, so a range-coalescing store can still merge their
 * reads.
 */
function scheduleFetches(canceller: RequestCanceller): void {
  if (canceller.pumpScheduled || canceller.queue.length === 0) return
  canceller.pumpScheduled = true
  setTimeout(() => {
    canceller.pumpScheduled = false
    while (
      canceller.queue.length > 0 &&
      canceller.activeCount < canceller.maxActive
    ) {
      const next = canceller.queue.shift()!
      canceller.activeCount++
      next.start().finally(() => {
        canceller.activeCount--
        scheduleFetches(canceller)
      })
    }
  }, 0)
}

/** Drop queued fetches that match, settling them without fetching. */
export function dropQueuedFetches(
  canceller: RequestCanceller,
  matches: (fetch: QueuedFetch) => boolean
): void {
  const kept: QueuedFetch[] = []
  const dropped: QueuedFetch[] = []
  for (const fetch of canceller.queue) {
    ;(matches(fetch) ? dropped : kept).push(fetch)
  }
  canceller.queue = kept
  for (const fetch of dropped) fetch.drop()
}

/** Order queued fetches nearest a region-grid point first. */
export function prioritizeQueuedFetches(
  canceller: RequestCanceller,
  centerX: number,
  centerY: number
): void {
  const distance = (f: QueuedFetch) =>
    (f.regionX - centerX) ** 2 + (f.regionY - centerY) ** 2
  canceller.queue.sort((a, b) => distance(a) - distance(b))
}

// ============================================================================
// Loading State Management
// ============================================================================

export interface LoadingManager {
  callback: LoadingStateCallback | undefined
  metadataLoading: boolean
  chunksLoading: boolean
  error: Error | null
}

export function createLoadingManager(): LoadingManager {
  return {
    callback: undefined,
    metadataLoading: false,
    chunksLoading: false,
    error: null,
  }
}

export function setLoadingCallback(
  manager: LoadingManager,
  callback: LoadingStateCallback | undefined
): void {
  manager.callback = callback
}

export function emitLoadingState(manager: LoadingManager): void {
  if (!manager.callback) return
  const state: LoadingState = {
    loading: manager.metadataLoading || manager.chunksLoading,
    metadata: manager.metadataLoading,
    chunks: manager.chunksLoading,
    error: manager.error,
  }
  manager.callback(state)
}

/**
 * Spinner debouncer for chunk loading: flips `chunksLoading` on only after
 * a short delay, so cache-hit refetches (e.g. scrubbing a selector within
 * already-fetched tiles, or any <80ms fetch) never trigger it; flips it
 * off immediately so the UI stays honest when real work finishes.
 *
 * `show()` is idempotent — calling it repeatedly while a timer is pending
 * or while the spinner is already on is a no-op. `hide()` cancels any
 * pending show and turns the spinner off if it was on.
 */
export interface ChunkLoadingDebouncer {
  show(): void
  hide(): void
}

export function createChunkLoadingDebouncer(
  manager: LoadingManager,
  showDelayMs: number = 80
): ChunkLoadingDebouncer {
  let showTimer: ReturnType<typeof setTimeout> | null = null

  return {
    show() {
      if (manager.chunksLoading) return
      if (showTimer) return
      showTimer = setTimeout(() => {
        showTimer = null
        if (!manager.chunksLoading) {
          manager.chunksLoading = true
          emitLoadingState(manager)
        }
      }, showDelayMs)
    },
    hide() {
      if (showTimer) {
        clearTimeout(showTimer)
        showTimer = null
      }
      if (manager.chunksLoading) {
        manager.chunksLoading = false
        emitLoadingState(manager)
      }
    },
  }
}
