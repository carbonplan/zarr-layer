/**
 * @module band-format
 *
 * How custom-shader bands are stored on the GPU. Small integer dtypes are
 * uploaded as-is to integer textures, which a float copy would make 2-4x
 * larger; scale/offset and fill detection then happen in the shader.
 * Everything else is uploaded as float32.
 *
 * 32-bit integers stay float: the shader compares against the fill value in
 * float, which is only exact up to 2^24.
 */

export type BandFormat = 'float' | 'int' | 'uint'

export type BandArray =
  | Float32Array
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array

/** Per-region values the shader needs to turn raw integer texels into data. */
export interface BandTransform {
  scale: number
  offset: number
  fill: number | null
}

const NATIVE_FORMATS: Record<string, BandFormat> = {
  int8: 'int',
  int16: 'int',
  uint8: 'uint',
  uint16: 'uint',
}

export function bandFormatForDtype(
  dtype: string | null | undefined
): BandFormat {
  return (dtype && NATIVE_FORMATS[dtype]) || 'float'
}

export function bandFormatOf(array: BandArray): BandFormat {
  if (array instanceof Int8Array || array instanceof Int16Array) return 'int'
  if (array instanceof Uint8Array || array instanceof Uint16Array) return 'uint'
  return 'float'
}

/** Is `data` an integer array that can be uploaded without conversion? */
export function isNativeBandArray(data: unknown): data is BandArray {
  return (
    data instanceof Int8Array ||
    data instanceof Uint8Array ||
    data instanceof Int16Array ||
    data instanceof Uint16Array
  )
}

export interface BandTextureFormats {
  internalFormat: GLenum
  format: GLenum
  type: GLenum
}

export function bandTextureFormats(
  gl: WebGL2RenderingContext,
  array: BandArray
): BandTextureFormats {
  if (array instanceof Int8Array)
    return { internalFormat: gl.R8I, format: gl.RED_INTEGER, type: gl.BYTE }
  if (array instanceof Uint8Array)
    return {
      internalFormat: gl.R8UI,
      format: gl.RED_INTEGER,
      type: gl.UNSIGNED_BYTE,
    }
  if (array instanceof Int16Array)
    return { internalFormat: gl.R16I, format: gl.RED_INTEGER, type: gl.SHORT }
  if (array instanceof Uint16Array)
    return {
      internalFormat: gl.R16UI,
      format: gl.RED_INTEGER,
      type: gl.UNSIGNED_SHORT,
    }
  return { internalFormat: gl.R32F, format: gl.RED, type: gl.FLOAT }
}

/** Allocate an empty array of the same kind as `like`. */
export function allocateLike(like: BandArray, length: number): BandArray {
  return new (like.constructor as new (length: number) => BandArray)(length)
}
