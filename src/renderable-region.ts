/**
 * @module renderable-region
 *
 * Unified rendering abstraction for both tiles and regions.
 * Both TiledMode and UntiledMode convert their data to this interface
 * for a single render path.
 */

import type { MercatorBounds } from './map-utils'
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
  vertexCount: number

  // Main texture (pre-uploaded)
  texture: WebGLTexture

  // Band textures (for custom shaders)
  bandData: Map<string, Float32Array>
  bandTextures: Map<string, WebGLTexture>
  bandTexturesUploaded: Set<string>
  bandTexturesConfigured: Set<string>
  width: number
  height: number

  // Texture transform (for parent tile fallback, defaults to identity)
  texScale?: [number, number]
  texOffset?: [number, number]

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
  customShaderConfig?: CustomShaderConfig
): boolean {
  const bounds = region.mercatorBounds

  // Set position uniforms from mercator bounds
  const scaleX = (bounds.x1 - bounds.x0) / 2
  const scaleY = (bounds.y1 - bounds.y0) / 2
  const shiftX = (bounds.x0 + bounds.x1) / 2
  const shiftY = (bounds.y0 + bounds.y1) / 2

  gl.uniform1f(shaderProgram.scaleLoc, 0)
  gl.uniform1f(shaderProgram.scaleXLoc, scaleX)
  gl.uniform1f(shaderProgram.scaleYLoc, scaleY)
  gl.uniform1f(shaderProgram.shiftXLoc, shiftX)
  gl.uniform1f(shaderProgram.shiftYLoc, shiftY)

  // Set texture transform (for parent tile fallback)
  const texScale = region.texScale ?? [1, 1]
  const texOffset = region.texOffset ?? [0, 0]
  gl.uniform2f(shaderProgram.texScaleLoc, texScale[0], texScale[1])
  gl.uniform2f(shaderProgram.texOffsetLoc, texOffset[0], texOffset[1])

  // Bind geometry
  bindGeometryBuffers(
    gl,
    shaderProgram,
    region.vertexBuffer,
    region.pixCoordBuffer
  )

  // Bind textures (upload happens at fetch time for both tiles and regions)
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
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, region.texture)
    if (shaderProgram.texLoc !== null) {
      gl.uniform1i(shaderProgram.texLoc, 0)
    }
  }

  // Draw for each world offset (map wrapping)
  for (const worldOffset of worldOffsets) {
    gl.uniform1f(shaderProgram.worldXOffsetLoc, worldOffset)
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, region.vertexCount)
  }

  return true
}
