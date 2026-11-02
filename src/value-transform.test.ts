import { describe, it, expect } from 'vitest'
import { BoolArray } from 'zarrita'
import {
  decodeValues,
  resolveValueTransform,
  type ValueTransform,
} from './value-transform'

const dataset = { scaleFactor: 2, addOffset: 1, fill_value: -9999 }

describe('resolveValueTransform', () => {
  it("uses the dataset's transform when the level declares none", () => {
    expect(resolveValueTransform('int16', undefined, dataset, null)).toEqual({
      scale: 2,
      offset: 1,
      fill: -9999,
    })
    expect(
      resolveValueTransform('int16', { asset: '0' }, dataset, null)
    ).toEqual({ scale: 2, offset: 1, fill: -9999 })
  })

  it("prefers the level's own transform", () => {
    const level = { asset: '1', scaleFactor: 3, addOffset: 0, fillValue: -1 }
    expect(resolveValueTransform('int16', level, dataset, null)).toEqual({
      scale: 3,
      offset: 0,
      fill: -1,
    })
  })

  it('keeps a level that declares no fill without one', () => {
    const level = { asset: '1', fillValue: null }
    expect(resolveValueTransform('int16', level, dataset, null).fill).toBe(null)
  })

  it('lets the layer option replace any store fill', () => {
    const level = { asset: '1', fillValue: -1 }
    expect(resolveValueTransform('int16', level, dataset, 7).fill).toBe(7)
    expect(
      resolveValueTransform(
        'int16',
        { asset: '1', fillValue: null },
        dataset,
        7
      ).fill
    ).toBe(7)
  })

  it('rounds the fill of a float32 array to float32 only', () => {
    const ds = { ...dataset, fill_value: 9.96921e36 }
    expect(resolveValueTransform('float32', undefined, ds, null).fill).toBe(
      Math.fround(9.96921e36)
    )
    expect(resolveValueTransform('float64', undefined, ds, null).fill).toBe(
      9.96921e36
    )
  })
})

// Present wherever zarrita can read float16.
const Float16 = (
  globalThis as unknown as { Float16Array?: Float32ArrayConstructor }
).Float16Array

describe.runIf(Float16)('float16 fills', () => {
  it('match stored values once rounded to float16', () => {
    const t = resolveValueTransform(
      'float16',
      undefined,
      { ...dataset, fill_value: 0.1 },
      null
    )
    const out = decodeValues(new Float16!([0.1, 0.5]), t, new Float32Array(2))
    expect(Array.from(out)).toEqual([NaN, 0.5 * 2 + 1])
  })
})

const identity: ValueTransform = { scale: 1, offset: 0, fill: null }

describe('decodeValues', () => {
  it('matches the fill on the raw value, before scaling', () => {
    const t = { scale: 1e-4, offset: 0, fill: -32768 }
    const out = decodeValues(
      new Int16Array([-32768, 0, 12345]),
      t,
      new Float32Array(3)
    )
    expect(Array.from(out)).toEqual([NaN, 0, Math.fround(1.2345)])
  })

  it('matches a float64 fill before rounding to float32', () => {
    // 1e20 + 1e12 rounds to the same float32 as 1e20.
    const t = { ...identity, fill: 1e20 }
    const out = decodeValues(
      new Float64Array([1e20, 1e20 + 1e12]),
      t,
      new Float32Array(2)
    )
    expect(Math.fround(1e20 + 1e12)).toBe(Math.fround(1e20))
    expect(Array.from(out)).toEqual([NaN, Math.fround(1e20)])
  })

  it('matches a 32-bit integer fill exactly', () => {
    // Both values round to 2^31 in float32.
    const t = { ...identity, fill: 2147483647 }
    const out = decodeValues(
      new Int32Array([2147483647, 2147483646]),
      t,
      new Float32Array(2)
    )
    expect(Array.from(out)).toEqual([NaN, 2147483648])
  })

  it('writes NaN for missing and non-finite values', () => {
    const out = decodeValues(
      new Float32Array([NaN, Infinity, -Infinity, 1]),
      { scale: 10, offset: 0, fill: null },
      new Float64Array(4)
    )
    expect(Array.from(out)).toEqual([NaN, NaN, NaN, 10])
  })

  it('drops values scaling pushes past the float64 range', () => {
    const out = decodeValues(
      new Float64Array([1e300, 1]),
      { scale: 1e10, offset: 0, fill: null },
      new Float64Array(2)
    )
    expect(Array.from(out)).toEqual([NaN, 1e10])
  })

  it('keeps zero when there is no fill', () => {
    const out = decodeValues(
      new Uint8Array([0, 1]),
      identity,
      new Float32Array(2)
    )
    expect(Array.from(out)).toEqual([0, 1])
  })

  it('divides transformed values by the divisor', () => {
    const out = decodeValues(
      new Int16Array([4, -1]),
      { scale: 2, offset: 2, fill: -1 },
      new Float32Array(2),
      5
    )
    expect(Array.from(out)).toEqual([2, NaN])
  })

  it('reads bool arrays', () => {
    const out = decodeValues(
      new BoolArray([true, false]) as unknown as ArrayLike<number>,
      identity,
      new Float32Array(2)
    )
    expect(Array.from(out)).toEqual([1, 0])
  })
})
