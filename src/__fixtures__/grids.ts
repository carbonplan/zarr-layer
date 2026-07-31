/**
 * @module __fixtures__/grids
 *
 * Synthetic raster fixtures for query/projection tests.
 *
 * `indexRamp` fills each pixel with its own row-major flat index, so a test
 * can map a returned value back to the (x, y) pixel that produced it — the
 * cleanest way to assert *which* pixels a query selected, independent of any
 * coordinate transform.
 */

export interface Grid {
  width: number
  height: number
  data: Float32Array
}

/** Row-major grid where `data[y * width + x] === y * width + x`. */
export function indexRamp(width: number, height: number): Grid {
  const data = new Float32Array(width * height)
  for (let i = 0; i < data.length; i++) data[i] = i
  return { width, height, data }
}

/** Grid filled with a single constant value. */
export function constant(width: number, height: number, value: number): Grid {
  const data = new Float32Array(width * height).fill(value)
  return { width, height, data }
}

/** Convert a flat index (as produced by `indexRamp`) back to `[x, y]`. */
export function indexToXY(index: number, width: number): [number, number] {
  return [index % width, Math.floor(index / width)]
}
