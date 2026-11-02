/**
 * @module value-transform
 *
 * How stored values become data values, shared by rendering and queries so
 * the two always agree: the fill value and non-finite values are missing
 * (NaN), everything else is `raw * scale + offset`.
 */

import type { ResolutionLevel } from './types'

export interface ValueTransform {
  scale: number
  offset: number
  /** The fill as the array's dtype stores it, so raw values match it exactly. */
  fill: number | null
}

/**
 * The transform for reads of an array of `dtype` at `level`. A layer's
 * `fillValue` option replaces the store's fill; otherwise a level that
 * declares its own metadata uses its own fill, even when that is none.
 */
export function resolveValueTransform(
  dtype: string,
  level: ResolutionLevel | undefined,
  dataset: {
    scaleFactor: number
    addOffset: number
    fill_value: number | null
  },
  fillOverride: number | null
): ValueTransform {
  const fill =
    fillOverride ??
    (level?.fillValue !== undefined ? level.fillValue : dataset.fill_value)
  return {
    scale: level?.scaleFactor ?? dataset.scaleFactor,
    offset: level?.addOffset ?? dataset.addOffset,
    fill: fill === null ? null : storedFill(fill, dtype),
  }
}

/**
 * Metadata may write a float32 fill with fewer digits than it takes to
 * round-trip, so the value the array holds is the fill rounded to float32.
 */
function storedFill(fill: number, dtype: string): number {
  return dtype === 'float32' ? Math.fround(fill) : fill
}

/**
 * Apply `transform` to raw values in their stored dtype, before any rounding
 * to float32, then divide by `divisor`. Missing values are written as NaN.
 */
export function decodeValues<T extends Float32Array | Float64Array>(
  raw: ArrayLike<number>,
  transform: ValueTransform,
  out: T,
  divisor: number = 1
): T {
  // zarrita's bool arrays can only be iterated, not indexed.
  const values = ArrayBuffer.isView(raw)
    ? raw
    : Float32Array.from(raw as unknown as Iterable<number>)
  const { scale, offset } = transform
  const fill = transform.fill ?? NaN
  for (let i = 0; i < values.length; i++) {
    const r = values[i]
    const v = r === fill ? NaN : r * scale + offset
    // v - v is 0 only for finite v.
    out[i] = v - v === 0 ? v / divisor : NaN
  }
  return out
}
