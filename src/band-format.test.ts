import { describe, it, expect } from 'vitest'
import { bandFormatForDtype, bandFormatOf } from './band-format'

describe('bandFormatForDtype', () => {
  it('keeps 8- and 16-bit integers native', () => {
    expect(bandFormatForDtype('int8')).toBe('int')
    expect(bandFormatForDtype('int16')).toBe('int')
    expect(bandFormatForDtype('uint8')).toBe('uint')
    expect(bandFormatForDtype('uint16')).toBe('uint')
  })

  it('uploads everything else as float', () => {
    // 32-bit fill values are not exact in the shader's float comparison.
    for (const dtype of ['int32', 'uint32', 'float32', 'float64', 'bool']) {
      expect(bandFormatForDtype(dtype)).toBe('float')
    }
    expect(bandFormatForDtype(null)).toBe('float')
    expect(bandFormatForDtype(undefined)).toBe('float')
  })
})

describe('bandFormatOf', () => {
  it('reads the format from the array type', () => {
    expect(bandFormatOf(new Int8Array(1))).toBe('int')
    expect(bandFormatOf(new Uint16Array(1))).toBe('uint')
    expect(bandFormatOf(new Float32Array(1))).toBe('float')
  })
})
