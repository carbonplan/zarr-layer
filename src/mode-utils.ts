/**
 * @module mode-utils
 *
 * Shared utilities for the ZarrMode implementation (UntiledMode).
 * Provides common patterns for throttling, request cancellation, and loading state management.
 */

import type { LoadingStateCallback, LoadingState } from './types'

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
