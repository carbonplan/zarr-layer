/**
 * How a failed region fetch is retried. Network errors, timeouts,
 * throttling and server errors pass, so they are retried for as long as the
 * region is wanted. A client error other than those will not change on
 * retry, so it is final at once. Anything else, such as a decode error, gets
 * a few attempts before it is final.
 */
export type FetchFailureKind = 'transient' | 'permanent' | 'unknown'

export const RETRY_BASE_DELAY_MS = 1000
export const RETRY_MAX_DELAY_MS = 30_000
export const MAX_UNKNOWN_FAILURE_ATTEMPTS = 3

const TRANSIENT_STATUSES = new Set([408, 425, 429])
const STATUS_PATTERN = /\bstatus (\d{3})\b/i

function httpStatus(err: unknown): number | null {
  for (let e: unknown = err, depth = 0; e && depth < 5; depth++) {
    const status = (e as { status?: unknown }).status
    if (typeof status === 'number') return status
    const message = (e as { message?: unknown }).message
    if (typeof message === 'string') {
      const match = STATUS_PATTERN.exec(message)
      if (match) return Number(match[1])
    }
    e = (e as { cause?: unknown }).cause
  }
  return null
}

export function classifyFetchFailure(err: unknown): FetchFailureKind {
  const status = httpStatus(err)
  if (status !== null) {
    if (status >= 500 || TRANSIENT_STATUSES.has(status)) return 'transient'
    if (status >= 400) return 'permanent'
  }
  // fetch() rejects with a TypeError when the request never got a response.
  if (err instanceof TypeError) return 'transient'
  const name = (err as { name?: unknown })?.name
  if (name === 'TimeoutError' || name === 'NetworkError') return 'transient'
  return 'unknown'
}

/** Backoff before attempt `attempts + 1`, with jitter so retries spread out. */
export function retryDelay(attempts: number, random = Math.random): number {
  const ceiling = Math.min(
    RETRY_MAX_DELAY_MS,
    RETRY_BASE_DELAY_MS * 2 ** Math.max(0, attempts - 1)
  )
  return ceiling * (0.5 + 0.5 * random())
}

/** Whether a failure after `attempts` tries should be retried. */
export function shouldRetry(kind: FetchFailureKind, attempts: number): boolean {
  if (kind === 'transient') return true
  if (kind === 'unknown') return attempts < MAX_UNKNOWN_FAILURE_ATTEMPTS
  return false
}
