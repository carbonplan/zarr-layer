/** Re-throw with context attached. `cause` is assigned rather than passed to
 *  `super`, which needs a newer lib target than this package builds against. */
export function wrapError(message: string, cause: unknown): Error {
  const error = new Error(
    `${message}: ${cause instanceof Error ? cause.message : String(cause)}`
  )
  ;(error as Error & { cause?: unknown }).cause = cause
  return error
}

/**
 * Thrown when a layer can never answer a query: initialization failed, no
 * resolution level could be loaded, the layer was removed from the map, or it
 * was never added to one.
 *
 * Queries wait out a layer that is merely still initializing, so an unready
 * layer never comes back as an empty result.
 */
export class ZarrLayerNotReadyError extends Error {
  readonly name = 'ZarrLayerNotReadyError'
  /** The underlying failure, when one was available. Initialization failures
   *  carry the error they failed with; the other cases have none to attach. */
  readonly cause?: unknown

  constructor(
    /** The layer's `id`, so callers juggling several can tell which failed. */
    readonly layerId: string,
    message: string,
    options?: { cause?: unknown }
  ) {
    super(`[ZarrLayer:${layerId}] ${message}`)
    this.cause = options?.cause
  }
}
