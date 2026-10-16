/**
 * @module mapbox-globe-matrix
 *
 * Builds the clip matrix for the direct Mapbox globe path from the public
 * custom-layer render parameters that mapbox-gl >= 3.32 passes to render().
 */

type Mat4Like = number[] | Float32Array | Float64Array

/** The subset of Mapbox's `CustomLayerRenderParameters` the globe path reads. */
export interface MapboxRenderParameters {
  /** Camera-to-clip perspective with Mapbox's regular far plane. */
  projectionMatrix: Mat4Like
  /** World-to-camera. On the globe, world space is the ECEF sphere of radius
   *  8192 / (2π) with Y down, the same frame the ECEF vertex shader builds. */
  viewMatrix: Mat4Like
}

function isMat4(value: unknown): value is Mat4Like {
  return (
    (Array.isArray(value) ||
      value instanceof Float64Array ||
      value instanceof Float32Array) &&
    value.length === 16
  )
}

/** Accept the 8th render argument only when it carries both matrices; older
 *  Mapbox passes nothing and MapLibre passes a single params object. */
export function parseMapboxRenderParameters(
  value: unknown
): MapboxRenderParameters | undefined {
  if (!value || typeof value !== 'object') return undefined
  const { projectionMatrix, viewMatrix } = value as Record<string, unknown>
  if (!isMat4(projectionMatrix) || !isMat4(viewMatrix)) return undefined
  return { projectionMatrix, viewMatrix }
}

/**
 * The same OpenGL-style perspective with its far plane moved to infinity.
 * Mapbox draws the globe surface through such a matrix, so the direct path
 * has to share it to land at matching depths. Only the depth row changes:
 * m[10] = -(f+n)/(f-n) and m[14] = -2fn/(f-n) give n = m[14] / (m[10] - 1),
 * and the infinite limit is m[10] = -1, m[14] = -2n. The off-center terms
 * m[8], m[9] and everything else carry over untouched.
 */
export function infiniteFarPerspective(m: Mat4Like): Float64Array {
  const out = Float64Array.from(m)
  const near = m[14] / (m[10] - 1)
  out[10] = -1
  out[14] = -2 * near
  return out
}

/** Column-major 4x4 product a * b. */
export function multiplyMat4(a: Mat4Like, b: Mat4Like): Float64Array {
  const out = new Float64Array(16)
  for (let col = 0; col < 4; col++) {
    const b0 = b[col * 4]
    const b1 = b[col * 4 + 1]
    const b2 = b[col * 4 + 2]
    const b3 = b[col * 4 + 3]
    for (let row = 0; row < 4; row++) {
      out[col * 4 + row] =
        a[row] * b0 + a[4 + row] * b1 + a[8 + row] * b2 + a[12 + row] * b3
    }
  }
  return out
}

/** Globe world (ECEF) to clip for the direct globe path, computed in float64
 *  and handed to the GPU as float32. */
export function mapboxEcefClipMatrix(
  params: MapboxRenderParameters
): Float32Array {
  return new Float32Array(
    multiplyMat4(
      infiniteFarPerspective(params.projectionMatrix),
      params.viewMatrix
    )
  )
}
