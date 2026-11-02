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
 * `fillValue` option replaces the store's fill; otherwise the level's own
 * metadata wins, falling back to the dataset's where the level has none.
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
  const fill = fillOverride ?? level?.fillValue ?? dataset.fill_value
  return {
    scale: level?.scaleFactor ?? dataset.scaleFactor,
    offset: level?.addOffset ?? dataset.addOffset,
    fill: fill === null ? null : storedFill(fill, dtype),
  }
}

// zarrita reads float16 only where Float16Array exists, which ships with it.
const f16round = (Math as { f16round?: (x: number) => number }).f16round

/**
 * Metadata may write a float fill with fewer digits than it takes to
 * round-trip, so the value the array holds is the fill rounded to its dtype.
 */
function storedFill(fill: number, dtype: string): number {
  if (dtype === 'float32') return Math.fround(fill)
  if (dtype === 'float16' && f16round) return f16round(fill)
  return fill
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
