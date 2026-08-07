import { describe, it, expect, vi, afterEach } from 'vitest'
import { ZarrStore } from './zarr-store'

/**
 * `createFetchStore` is internal, so these drive it through a real `ZarrStore`
 * over a stubbed `fetch`. Initialization always fails (nothing resolves), so
 * every case awaits `initialized` under a catch.
 */

const SOURCE = 'https://example.com/store.zarr'

async function readThrough({
  status,
  onAuthError,
  transformRequest = (url: string) => ({ url }),
}: {
  status: number
  onAuthError?: (status: number) => void
  transformRequest?: (url: string) => { url: string }
}) {
  const fetchStub = vi.fn(async () => new Response(null, { status }))
  vi.stubGlobal('fetch', fetchStub)

  const store = new ZarrStore({
    source: SOURCE,
    variable: 'temperature',
    transformRequest,
    onAuthError,
  })
  await store.initialized.catch(() => {})

  return { calls: fetchStub.mock.calls.length }
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('onAuthError', () => {
  it('fires once for a burst of credential-shaped failures', async () => {
    const onAuthError = vi.fn()
    const { calls } = await readThrough({ status: 401, onAuthError })

    expect(calls).toBeGreaterThan(1)
    expect(onAuthError).toHaveBeenCalledTimes(1)
    expect(onAuthError).toHaveBeenCalledWith(401)
  })

  it('fires for 400, which is what expired temporary credentials return', async () => {
    const onAuthError = vi.fn()
    await readThrough({ status: 400, onAuthError })

    expect(onAuthError).toHaveBeenCalledWith(400)
  })

  it('ignores statuses that are not credential-shaped', async () => {
    const onAuthError = vi.fn()
    await readThrough({ status: 403, onAuthError })
    await readThrough({ status: 404, onAuthError })
    await readThrough({ status: 500, onAuthError })

    expect(onAuthError).not.toHaveBeenCalled()
  })

  it('leaves 400 unmasked when no handler is set', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(null, { status: 400 }))
    )

    const store = new ZarrStore({
      source: SOURCE,
      variable: 'temperature',
      transformRequest: (url: string) => ({ url }),
    })
    const err = await store.initialized.then(
      () => null,
      (e: Error) => e
    )

    expect(err?.message).toMatch(/400/)
  })
})
