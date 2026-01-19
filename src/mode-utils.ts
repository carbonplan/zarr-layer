/**
 * @module mode-utils
 *
 * Shared utilities for ZarrMode implementations (TiledMode and UntiledMode).
 * Provides common patterns for throttling, request cancellation, and loading state management.
 */

import type { LoadingStateCallback, LoadingState } from './types'

// ============================================================================
// Throttle Management
// ============================================================================

export interface ThrottleState {
  lastFetchTime: number
  throttleTimeout: ReturnType<typeof setTimeout> | null
  throttledPending: boolean
}

export function createThrottleState(): ThrottleState {
  return {
    lastFetchTime: 0,
    throttleTimeout: null,
    throttledPending: false,
  }
}

/**
 * Check if we should throttle (wait before fetching).
 * Returns the wait time in ms if we should throttle, or 0 if we can proceed.
 */
export function getThrottleWaitTime(
  state: ThrottleState,
  throttleMs: number
): number {
  if (throttleMs <= 0) return 0
  const now = Date.now()
  const timeSinceLastFetch = now - state.lastFetchTime
  if (timeSinceLastFetch < throttleMs) {
    return throttleMs - timeSinceLastFetch
  }
  return 0
}

/**
 * Schedule a throttled update callback after the wait time.
 * Only schedules if no timeout is already pending.
 */
export function scheduleThrottledUpdate(
  state: ThrottleState,
  waitTime: number,
  invalidate: () => void
): void {
  if (state.throttleTimeout) return // Already scheduled

  state.throttledPending = true
  state.throttleTimeout = setTimeout(() => {
    state.throttleTimeout = null
    state.throttledPending = false
    invalidate()
  }, waitTime)
}

/**
 * Mark that a fetch is starting (update timestamp).
 */
export function markFetchStart(state: ThrottleState): void {
  state.lastFetchTime = Date.now()
}

/**
 * Clear any pending throttle timeout.
 */
export function clearThrottle(state: ThrottleState): void {
  if (state.throttleTimeout) {
    clearTimeout(state.throttleTimeout)
    state.throttleTimeout = null
  }
  state.throttledPending = false
}

// ============================================================================
// Request Cancellation
// ============================================================================

export interface RequestCanceller {
  controllers: Map<number, AbortController>
  currentVersion: number
}

export function createRequestCanceller(): RequestCanceller {
  return {
    controllers: new Map(),
    currentVersion: 0,
  }
}

/**
 * Get a new AbortController for a new request, incrementing the version.
 */
export function getNewController(canceller: RequestCanceller): {
  controller: AbortController
  version: number
} {
  const version = ++canceller.currentVersion
  const controller = new AbortController()
  canceller.controllers.set(version, controller)
  return { controller, version }
}

/**
 * Remove a controller from tracking (typically after request completes).
 */
export function removeController(
  canceller: RequestCanceller,
  version: number
): void {
  canceller.controllers.delete(version)
}

/**
 * Cancel all requests older than the completed version.
 */
export function cancelOlderRequests(
  canceller: RequestCanceller,
  completedVersion: number
): void {
  for (const [version, controller] of canceller.controllers) {
    if (version < completedVersion) {
      controller.abort()
      canceller.controllers.delete(version)
    }
  }
}

/**
 * Cancel all pending requests.
 */
export function cancelAllRequests(canceller: RequestCanceller): void {
  for (const controller of canceller.controllers.values()) {
    controller.abort()
  }
  canceller.controllers.clear()
}

/**
 * Check if any requests are still pending (not aborted).
 */
export function hasActiveRequests(canceller: RequestCanceller): boolean {
  for (const controller of canceller.controllers.values()) {
    if (!controller.signal.aborted) {
      return true
    }
  }
  return false
}

// ============================================================================
// Loading State Management
// ============================================================================

export interface LoadingManager {
  callback: LoadingStateCallback | undefined
  metadataLoading: boolean
  chunksLoading: boolean
}

export function createLoadingManager(): LoadingManager {
  return {
    callback: undefined,
    metadataLoading: false,
    chunksLoading: false,
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
    error: null,
  }
  manager.callback(state)
}

export function setMetadataLoading(
  manager: LoadingManager,
  loading: boolean
): void {
  manager.metadataLoading = loading
  emitLoadingState(manager)
}

export function setChunksLoading(
  manager: LoadingManager,
  loading: boolean
): void {
  manager.chunksLoading = loading
  emitLoadingState(manager)
}
