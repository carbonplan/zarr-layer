/**
 * @module renderable-region
 *
 * Unified rendering abstraction for data regions.
 * RegionRenderer (flat/globe direct rendering) and the Mapbox tile renderer
 * convert their regions to this interface for a single render path.
 */

import type { MercatorBounds, MeshMercatorBounds } from './map-utils'
import type { CustomShaderConfig } from './renderer-types'
import type { ShaderProgram } from './shader-program'
import { bindBandTextures, bindGeometryBuffers } from './render-helpers'

/**
 * A region ready for rendering.
 * This is the common interface for both tiles and untiled regions.
 * Textures are expected to be uploaded before rendering.
 */
export interface RenderableRegion {
  mercatorBounds: MercatorBounds

  // Geometry
  vertexBuffer: WebGLBuffer
  pixCoordBuffer: WebGLBuffer
  indexBuffer: WebGLBuffer
  indexCount: number

  // Normalized-Mercator bounds used to reconstruct region-local mesh positions
  meshBounds: MeshMercatorBounds

  // Data orientation: true = row 0 is south (latitude ascending)
  // Resolved by ZarrStore during init
  latIsAscending: boolean

  // Render-time sampling mode. Defaults to linear.
  sampleMode?: 'linear' | 'mercator-invert' | 'wgs84-lookup'

  // Main texture (pre-uploaded). Null when band textures are sampled instead.
  texture: WebGLTexture | null

  // Band textures (for custom shaders)
  bandData: Map<string, Float32Array>
  bandTextures: Map<string, WebGLTexture>
  bandTexturesUploaded: Set<string>
  bandTexturesConfigured: Set<string>
  width: number
  height: number

  // Callbacks for lazy resource creation
  ensureBandTexture?: (bandName: string) => WebGLTexture | null
}

/**
 * Render a single region with the given shader program.
 * Handles uniform setup, texture binding, and draw call.
 *
 * @param gl - WebGL2 context
 * @param shaderProgram - Compiled shader program
 * @param region - Region to render
 * @param worldOffsets - X offsets for map wrapping
 * @param customShaderConfig - Optional custom shader configuration
 * @returns true if rendered successfully, false if skipped (missing data)
 */
export function renderRegion(
  gl: WebGL2RenderingContext,
  shaderProgram: ShaderProgram,
  region: RenderableRegion,
  worldOffsets: number[],
  customShaderConfig?: CustomShaderConfig,
  // Flat projection matrix for the eye-coords path (source-projected 'wgs84'
  // shader only), at the renderer's native precision (NOT pre-cast to Float32 —
  // see the anchor_clip computation below). When provided, anchor_clip is
  // computed PER WORLD OFFSET as (shiftX + worldOffset, shiftY), so the eye
  // origin sits near the on-screen region — and near-camera for wrapped copies,
  // not the off-screen layer center: small clip magnitudes, no pan/zoom jitter.
  eyeMatrix?: number[] | Float32Array | Float64Array | null
): boolean {
  const meshBounds = region.meshBounds
  if (!meshBounds) return false

  const hasLatBounds =
    region.mercatorBounds.latMin !== undefined &&
    region.mercatorBounds.latMax !== undefined
  const sampMode = region.sampleMode ?? 'linear'

  const scaleX = (meshBounds.x1 - meshBounds.x0) / 2
  const scaleY = (meshBounds.y1 - meshBounds.y0) / 2
  const shiftX = (meshBounds.x0 + meshBounds.x1) / 2
  const shiftY = (meshBounds.y0 + meshBounds.y1) / 2

  gl.uniform1f(shaderProgram.scaleLoc, 0)
  gl.uniform1f(shaderProgram.scaleXLoc, scaleX)
  gl.uniform1f(shaderProgram.scaleYLoc, scaleY)
  gl.uniform1f(shaderProgram.shiftXLoc, shiftX)
  gl.uniform1f(shaderProgram.shiftYLoc, shiftY)

  // Eye-coords path for flat source-projected shaders (see
  // VERTEX_TO_WGS84_TO_MERCATOR). Upload u_eye_matrix once; u_anchor_clip is
  // computed PER WORLD OFFSET in the draw loop below, so a wrapped copy's anchor
  // is its near-camera wrapped origin (no large-term cancellation). The anchor is
  // computed from the original (un-cast) matrix; cast to Float32 only for upload.
  const eyeM = eyeMatrix
  const eyeActive = !!(
    eyeM &&
    eyeM.length >= 16 &&
    shaderProgram.eyeMatrixLoc &&
    shaderProgram.anchorClipLoc
  )
  if (eyeActive && eyeM) {
    gl.uniformMatrix4fv(
      shaderProgram.eyeMatrixLoc,
      false,
      eyeM instanceof Float32Array ? eyeM : new Float32Array(eyeM)
    )
  }

  // Set fragment shader reprojection uniforms based on sample mode
  const needsLatLookup =
    sampMode === 'mercator-invert' || sampMode === 'wgs84-lookup'
  if (shaderProgram.reprojectLoc !== null) {
    switch (sampMode) {
      case 'linear':
        gl.uniform1i(shaderProgram.reprojectLoc, 0)
        break
      case 'mercator-invert':
        gl.uniform1i(shaderProgram.reprojectLoc, 1)
        break
      case 'wgs84-lookup':
        gl.uniform1i(shaderProgram.reprojectLoc, 2)
        break
    }
  }
  if (needsLatLookup && shaderProgram.latBoundsLoc !== null && hasLatBounds) {
    gl.uniform2f(
      shaderProgram.latBoundsLoc,
      region.mercatorBounds.latMin!,
      region.mercatorBounds.latMax!
    )
  }
  if (needsLatLookup && shaderProgram.latIsAscendingLoc !== null) {
    gl.uniform1i(shaderProgram.latIsAscendingLoc, region.latIsAscending ? 1 : 0)
  }

  // Bind geometry
  bindGeometryBuffers(
    gl,
    shaderProgram,
    region.vertexBuffer,
    region.pixCoordBuffer
  )

  if (!region.indexBuffer) return false
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, region.indexBuffer)

  // Bind textures. The main texture must already be uploaded; bindBandTextures
  // uploads any band whose contents are not resident yet.
  if (shaderProgram.useCustomShader && customShaderConfig) {
    const bandsBound = bindBandTextures(gl, {
      bandData: region.bandData,
      bandTextures: region.bandTextures,
      bandTexturesUploaded: region.bandTexturesUploaded,
      bandTexturesConfigured: region.bandTexturesConfigured,
      customShaderConfig,
      width: region.width,
      height: region.height,
      ensureTexture: region.ensureBandTexture,
    })
    if (!bandsBound) {
      return false
    }
  } else {
    if (!region.texture) return false
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, region.texture)
    if (shaderProgram.texLoc !== null) {
      gl.uniform1i(shaderProgram.texLoc, 0)
    }
  }

  // Draw for each world offset (map wrapping)
  for (const worldOffset of worldOffsets) {
    gl.uniform1f(shaderProgram.worldXOffsetLoc, worldOffset)
    if (eyeActive && eyeM) {
      // u_anchor_clip = M * vec4(shiftX + worldOffset, shiftY, 0, 1), in Float64
      // then cast by uniform4f. Folding the offset in here (vs. a separate shader
      // term) keeps a wrapped copy's anchor near the camera — no large-term
      // cancellation. Column-major: result[i] = m[i]*x + m[4+i]*y + m[12+i].
      const ax = shiftX + worldOffset
      gl.uniform4f(
        shaderProgram.anchorClipLoc,
        eyeM[0] * ax + eyeM[4] * shiftY + eyeM[12],
        eyeM[1] * ax + eyeM[5] * shiftY + eyeM[13],
        eyeM[2] * ax + eyeM[6] * shiftY + eyeM[14],
        eyeM[3] * ax + eyeM[7] * shiftY + eyeM[15]
      )
    }
    gl.drawElements(gl.TRIANGLES, region.indexCount, gl.UNSIGNED_INT, 0)
  }

  return true
}
