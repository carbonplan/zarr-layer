import { describe, expect, it, vi } from 'vitest'
import { ZarrLayer } from './zarr-layer'
import type { MapLike } from './types'

type Listener = (...args: unknown[]) => void

const fakeMap = () => {
  const map = {
    on: vi.fn<(event: string, handler: Listener) => void>(),
    off: vi.fn<(event: string, handler: Listener) => void>(),
    triggerRepaint: vi.fn(),
    getZoom: () => 0,
  }
  const handlersFor = (spy: typeof map.on, event: string) =>
    spy.mock.calls
      .filter(([name]) => name === event)
      .map(([, handler]) => handler)
  return {
    map: map as unknown as MapLike,
    subscribed: (event: string) => handlersFor(map.on, event),
    unsubscribed: (event: string) => handlersFor(map.off, event),
  }
}

// resolveGl only probes for these two members before accepting the context.
// Anything past that fails, which drives the layer down its init-error path.
const fakeGl = () =>
  ({
    getUniformLocation: () => null,
    drawBuffers: () => {},
  } as unknown as WebGL2RenderingContext)

const makeLayer = () =>
  new ZarrLayer({
    id: 'zarr-layer',
    source: 'https://example.invalid/missing.zarr',
    variable: 'foo',
    colormap: [
      [0, 0, 0],
      [255, 255, 255],
    ],
    clim: [0, 1],
  })

/**
 * `map.remove()` reaches `Style._remove()`, which tears the style down without
 * calling `onRemove` on custom layers. The layer subscribes to the map's own
 * `remove` event so it releases its store either way.
 */
describe('ZarrLayer map lifecycle', () => {
  it('subscribes to the map remove event when added', () => {
    const { map, subscribed } = fakeMap()

    makeLayer().onAdd(map, fakeGl())

    expect(subscribed('remove')).toHaveLength(1)
  })

  it('subscribes before initialization so a failed init still has a hook', () => {
    const { map, subscribed, unsubscribed } = fakeMap()

    // This gl stub cannot satisfy colormap upload, so init throws and the
    // layer disposes itself during onAdd.
    makeLayer().onAdd(map, fakeGl())

    expect(subscribed('remove')).toHaveLength(1)
    expect(unsubscribed('remove')).toEqual(subscribed('remove'))
  })

  it('unsubscribes on removal so the map stops retaining it', () => {
    const { map, subscribed, unsubscribed } = fakeMap()
    const layer = makeLayer()

    layer.onAdd(map, fakeGl())
    layer.onRemove(map, fakeGl())

    expect(unsubscribed('remove')).toEqual(subscribed('remove'))
  })
})
