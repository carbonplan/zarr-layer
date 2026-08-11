/**
 * Thrown when a layer can never answer a query: initialization failed, the
 * layer was removed from the map, or it was never added to one.
 *
 * Queries wait for a layer that is merely still initializing, so this is
 * distinct from an empty result — an empty result means the geometry found no
 * data, never that the layer wasn't ready.
 */
export class ZarrLayerNotReadyError extends Error {
  readonly name = 'ZarrLayerNotReadyError'
  /** The error that caused initialization to fail, when there was one. */
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
