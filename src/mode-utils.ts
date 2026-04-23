/**
 * @module mode-utils
 *
 * Shared utilities for ZarrMode implementations (TiledMode and UntiledMode).
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
