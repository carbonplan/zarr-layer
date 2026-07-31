import { describe, it, expect } from 'vitest'
import { ColormapState } from './colormap'

// ColormapState.build() runs in the constructor and produces the CPU-side
// representation (colors + Float32 texture data). GL upload is not exercised
// here — it needs a WebGL context — but the data layout that the shader
// samples is fully testable.

describe('ColormapState build', () => {
  it('parses hex colors and normalizes to [0, 1] float data', () => {
    const cm = new ColormapState(['#ff0000', '#00ff00'])
    expect(cm.length).toBe(2)
    expect(cm.colors).toEqual([
      [255, 0, 0],
      [0, 255, 0],
    ])
    expect(Array.from(cm.floatData)).toEqual([1, 0, 0, 0, 1, 0])
  })

  it('scales 0-255 RGB triples down to [0, 1]', () => {
    const cm = new ColormapState([
      [255, 0, 0],
      [0, 0, 0],
    ])
    expect(Array.from(cm.floatData)).toEqual([1, 0, 0, 0, 0, 0])
  })

  it('leaves already-normalized [0, 1] data unscaled', () => {
    const cm = new ColormapState([
      [1, 0, 0],
      [0, 0, 1],
    ])
    expect(Array.from(cm.floatData)).toEqual([1, 0, 0, 0, 0, 1])
  })

  it('replaces the ramp via apply()', () => {
    const cm = new ColormapState(['#ffffff'])
    cm.apply(['#000000', '#ffffff'])
    expect(cm.length).toBe(2)
    expect(Array.from(cm.floatData)).toEqual([0, 0, 0, 1, 1, 1])
  })

  it('rejects invalid input', () => {
    expect(() => new ColormapState([])).toThrow(/non-empty/)
    expect(() => new ColormapState(['#ff' as string])).toThrow(
      /Invalid hex color/
    )
    expect(() => new ColormapState([[1, 2] as unknown as number[]])).toThrow(
      /\[r, g, b\]/
    )
  })
})
