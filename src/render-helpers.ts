/**
 * @module render-helpers
 *
 * Shared rendering utilities for both tiled and untiled modes.
 * Handles band texture setup, binding, and geometry buffer binding.
 */

import type { ShaderProgram } from './shader-program'
import type { CustomShaderConfig } from './renderer-types'
import { configureDataTexture, getTextureFormats } from './webgl-utils'

/**
 * Set up band texture uniform locations.
 * Called once per frame before rendering any tiles/regions.
 *
 * @param gl - WebGL context
 * @param shaderProgram - Shader program with band texture uniform locations
 * @param customShaderConfig - Custom shader configuration with band names
 */
export function setupBandTextureUniforms(
  gl: WebGL2RenderingContext,
  shaderProgram: ShaderProgram,
  customShaderConfig?: CustomShaderConfig
): void {
  if (!shaderProgram.useCustomShader || !customShaderConfig) return

  let textureUnit = 2 // 0 = main texture, 1 = colormap
  for (const bandName of customShaderConfig.bands) {
    const loc = shaderProgram.bandTexLocs.get(bandName)
    if (loc) {
      gl.uniform1i(loc, textureUnit)
    }
    textureUnit++
  }
}

/** Options for band texture binding */
export interface BindBandTexturesOptions {
  /** Band data arrays by name */
  bandData: Map<string, Float32Array>
  /** Band textures by name */
  bandTextures: Map<string, WebGLTexture>
  /** Set of band names that have been uploaded */
  bandTexturesUploaded: Set<string>
  /** Set of band names that have been configured */
  bandTexturesConfigured: Set<string>
  /** Custom shader config with band names */
  customShaderConfig: CustomShaderConfig
  /** Texture width */
  width: number
  /** Texture height */
  height: number
  /** Optional function to ensure a texture exists for a band */
  ensureTexture?: (bandName: string) => WebGLTexture | null
}

/**
 * Bind and upload band textures for a single tile/region.
 * Returns false if any required band data is missing.
 *
 * @param gl - WebGL context
 * @param options - Band texture binding options
 * @returns true if all bands bound successfully, false if missing data
 */
export function bindBandTextures(
  gl: WebGL2RenderingContext,
  options: BindBandTexturesOptions
): boolean {
  const {
    bandData,
    bandTextures,
    bandTexturesUploaded,
    bandTexturesConfigured,
    customShaderConfig,
    width,
    height,
    ensureTexture,
  } = options

  let textureUnit = 2
  for (const bandName of customShaderConfig.bands) {
    const data = bandData.get(bandName)
    if (!data) {
      return false // Missing band data
    }

    let bandTex = bandTextures.get(bandName)
    if (!bandTex) {
      if (ensureTexture) {
        const newTex = ensureTexture(bandName)
        if (newTex) {
          bandTex = newTex
          bandTextures.set(bandName, bandTex)
        }
      } else {
        // Create texture directly
        bandTex = gl.createTexture()
        if (bandTex) {
          bandTextures.set(bandName, bandTex)
        }
      }
    }
    if (!bandTex) {
      return false // Failed to create texture
    }

    gl.activeTexture(gl.TEXTURE0 + textureUnit)
    gl.bindTexture(gl.TEXTURE_2D, bandTex)

    if (!bandTexturesConfigured.has(bandName)) {
      configureDataTexture(gl)
      bandTexturesConfigured.add(bandName)
    }

    if (!bandTexturesUploaded.has(bandName)) {
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.R32F,
        width,
        height,
        0,
        gl.RED,
        gl.FLOAT,
        data
      )
      bandTexturesUploaded.add(bandName)
    }

    textureUnit++
  }

  return true
}

/**
 * Bind geometry buffers and set up vertex attribute pointers.
 *
 * @param gl - WebGL context
 * @param shaderProgram - Shader program with attribute locations
 * @param vertexBuffer - Buffer containing vertex positions
 * @param pixCoordBuffer - Buffer containing texture coordinates
 */
export function bindGeometryBuffers(
  gl: WebGL2RenderingContext,
  shaderProgram: ShaderProgram,
  vertexBuffer: WebGLBuffer,
  pixCoordBuffer: WebGLBuffer
): void {
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer)
  gl.enableVertexAttribArray(shaderProgram.vertexLoc)
  gl.vertexAttribPointer(shaderProgram.vertexLoc, 2, gl.FLOAT, false, 0, 0)

  gl.bindBuffer(gl.ARRAY_BUFFER, pixCoordBuffer)
  gl.enableVertexAttribArray(shaderProgram.pixCoordLoc)
  gl.vertexAttribPointer(shaderProgram.pixCoordLoc, 2, gl.FLOAT, false, 0, 0)
}

/** Options for uploading a data texture */
export interface UploadTextureOptions {
  texture: WebGLTexture
  data: Float32Array
  width: number
  height: number
  channels: number
  configured: boolean
}

/** Result of texture upload with updated state */
export interface UploadTextureResult {
  configured: boolean
  uploaded: boolean
}

/**
 * Upload data to a texture, configuring it if needed.
 * Handles both initial upload and re-upload scenarios.
 *
 * @param gl - WebGL context
 * @param options - Texture upload options
 * @returns Updated configuration state
 */
export function uploadDataTexture(
  gl: WebGL2RenderingContext,
  options: UploadTextureOptions
): UploadTextureResult {
  const { texture, data, width, height, channels, configured } = options

  gl.activeTexture(gl.TEXTURE0)
  gl.bindTexture(gl.TEXTURE_2D, texture)

  if (!configured) {
    configureDataTexture(gl)
  }

  const { format, internalFormat } = getTextureFormats(gl, channels)
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    internalFormat,
    width,
    height,
    0,
    format,
    gl.FLOAT,
    data
  )

  return { configured: true, uploaded: true }
}
