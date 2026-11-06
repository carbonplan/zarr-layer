import { describe, it, expect } from 'vitest'
import {
  RETRY_MAX_DELAY_MS,
  classifyFetchFailure,
  retryDelay,
  shouldRetry,
} from './fetch-retry'

describe('classifyFetchFailure', () => {
  it.each([
    ['Unexpected response status 500 Internal Server Error', 'transient'],
    ['Unexpected response status 503 Service Unavailable', 'transient'],
    ['Unexpected response status 429 Too Many Requests', 'transient'],
    ['Unexpected response status 408 Request Timeout', 'transient'],
    ['Unexpected response status 403 Forbidden', 'permanent'],
    ['Unexpected response status 400 Bad Request', 'permanent'],
    ['Invalid codec configuration', 'unknown'],
  ])('%s is %s', (message, kind) => {
    expect(classifyFetchFailure(new Error(message))).toBe(kind)
  })

  it('treats a request that got no response as transient', () => {
    expect(classifyFetchFailure(new TypeError('Failed to fetch'))).toBe(
      'transient'
    )
    const timeout = new Error('signal timed out')
    timeout.name = 'TimeoutError'
    expect(classifyFetchFailure(timeout)).toBe('transient')
  })

  it('reads a status from a wrapped cause or a status field', () => {
    const wrapped = Object.assign(new Error('read failed'), {
      cause: new Error('Unexpected response status 502 Bad Gateway'),
    })
    expect(classifyFetchFailure(wrapped)).toBe('transient')
    expect(classifyFetchFailure({ status: 404 })).toBe('permanent')
  })
})

describe('retryDelay', () => {
  it('doubles from one second up to a ceiling, with jitter below it', () => {
    expect(retryDelay(1, () => 1)).toBe(1000)
    expect(retryDelay(3, () => 1)).toBe(4000)
    expect(retryDelay(20, () => 1)).toBe(RETRY_MAX_DELAY_MS)
    expect(retryDelay(3, () => 0)).toBe(2000)
  })
})

describe('shouldRetry', () => {
  it('retries transient failures indefinitely and permanent ones never', () => {
    expect(shouldRetry('transient', 50)).toBe(true)
    expect(shouldRetry('permanent', 1)).toBe(false)
    expect(shouldRetry('unknown', 1)).toBe(true)
    expect(shouldRetry('unknown', 3)).toBe(false)
  })
})
