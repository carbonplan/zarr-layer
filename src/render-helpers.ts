/**
 * @module render-helpers
 *
 * Shared rendering utilities for both tiled and untiled modes.
 * Handles band texture setup, binding, and geometry buffer binding.
 */

import type { ShaderProgram } from './shader-program'
import type { CustomShaderConfig } from './renderer-types'
import type { RegionState } from './region-state'
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
interface BindBandTexturesOptions {
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

/** Shared empty list so the no-bands prune allocates nothing per frame. */
const EMPTY_BANDS: readonly string[] = []

/** The per-band GPU bookkeeping carried on a region. */
interface BandTextureState {
  bandTextures: Map<string, WebGLTexture>
  bandTexturesUploaded: Set<string>
  bandTexturesConfigured: Set<string>
}

/**
 * Delete every band texture whose name is not in `wanted`. Runs per region on
 * every render call, so it exits without allocating in the common case where
 * the resident set already matches.
 */
function pruneBandTextures(
  gl: WebGL2RenderingContext,
  wanted: readonly string[],
  state: BandTextureState
): void {
  const { bandTextures } = state
  if (bandTextures.size === 0) return
  if (
    bandTextures.size === wanted.length &&
    wanted.every((name) => bandTextures.has(name))
  ) {
    return
  }

  const keep = new Set(wanted)
  for (const [name, tex] of bandTextures) {
    if (keep.has(name)) continue
    gl.deleteTexture(tex)
    bandTextures.delete(name)
    state.bandTexturesUploaded.delete(name)
    state.bandTexturesConfigured.delete(name)
  }
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

  // Band names track the selector on some datasets, so the set changes as the
  // user scrubs. Prune by membership rather than count: a same-size swap
  // (red, green -> nir, swir) replaces every name without changing the size.
  // Skipped when the caller owns the textures.
  if (!ensureTexture) {
    pruneBandTextures(gl, customShaderConfig.bands, {
      bandTextures,
      bandTexturesUploaded,
      bandTexturesConfigured,
    })
  }

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
interface UploadTextureOptions {
  texture: WebGLTexture
  data: Float32Array
  width: number
  height: number
  channels: number
  configured: boolean
}

/** Result of texture upload with updated state */
interface UploadTextureResult {
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

/**
 * Create and upload the band textures a custom shader samples, dropping any
 * that are no longer requested. Returns false if a band's data is missing or
 * a texture cannot be allocated, which makes the region undrawable.
 */
function ensureBandTextures(
  gl: WebGL2RenderingContext,
  region: RegionState,
  bands: readonly string[]
): boolean {
  pruneBandTextures(gl, bands, region)

  for (const name of bands) {
    const data = region.bandData.get(name)
    if (!data) return false

    let texture = region.bandTextures.get(name)
    if (!texture) {
      texture = gl.createTexture()
      if (!texture) return false
      region.bandTextures.set(name, texture)
    }
    if (region.bandTexturesUploaded.has(name)) continue

    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, texture)
    if (!region.bandTexturesConfigured.has(name)) {
      configureDataTexture(gl)
      region.bandTexturesConfigured.add(name)
    }
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.R32F,
      region.width,
      region.height,
      0,
      gl.RED,
      gl.FLOAT,
      data
    )
    region.bandTexturesUploaded.add(name)
  }
  return true
}

/**
 * Lazily create and upload a region's GPU resources from its CPU-side state.
 * Fetch produces only data and geometry arrays; the render paths call this
 * per frame so uploads happen on the context that is actually drawing.
 * Returns false while the region isn't renderable yet.
 *
 * `requiredBands` names the textures a custom shader will sample. It has to be
 * the same list the draw call uses: this function's return value decides
 * whether the level is considered covered, and a level that displaces its
 * fallbacks and then fails to bind a band leaves the viewport blank. The main
 * texture is not created in that mode, since nothing samples it.
 */
export function ensureRegionGpuResources(
  gl: WebGL2RenderingContext,
  region: RegionState,
  requiredBands?: readonly string[]
): boolean {
  if (!region.data || !region.vertexArr || !region.pixCoordArr) return false

  const bandRendering = !!requiredBands && requiredBands.length > 0
  let texturesReady: boolean

  if (bandRendering) {
    if (region.texture) {
      // Switched from main-texture rendering; nothing samples it now.
      gl.deleteTexture(region.texture)
      region.texture = null
      region.textureUploaded = false
    }
    texturesReady = ensureBandTextures(gl, region, requiredBands)
  } else {
    pruneBandTextures(gl, EMPTY_BANDS, region)
    if (!region.texture) region.texture = gl.createTexture()
    if (!region.texture) return false
    if (!region.textureUploaded) {
      const result = uploadDataTexture(gl, {
        texture: region.texture,
        data: region.data,
        width: region.width,
        height: region.height,
        channels: region.channels,
        configured: false,
      })
      region.textureUploaded = result.uploaded
    }
    texturesReady = region.textureUploaded
  }

  // Buffer objects are reused across re-uploads, so the dirty flag — not the
  // presence of a buffer — decides whether the GPU has the current mesh. A
  // region refetched at new dimensions regenerates its arrays, and without
  // this its data would be drawn against the previous mesh.
  if (!region.geometryUploaded) {
    if (!region.vertexBuffer) region.vertexBuffer = gl.createBuffer()
    if (!region.pixCoordBuffer) region.pixCoordBuffer = gl.createBuffer()
    if (region.vertexBuffer) {
      gl.bindBuffer(gl.ARRAY_BUFFER, region.vertexBuffer)
      gl.bufferData(gl.ARRAY_BUFFER, region.vertexArr, gl.STATIC_DRAW)
    }
    if (region.pixCoordBuffer) {
      gl.bindBuffer(gl.ARRAY_BUFFER, region.pixCoordBuffer)
      gl.bufferData(gl.ARRAY_BUFFER, region.pixCoordArr, gl.STATIC_DRAW)
    }
    if (region.useIndexedMesh && region.indexArr) {
      if (!region.indexBuffer) region.indexBuffer = gl.createBuffer()
      if (region.indexBuffer) {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, region.indexBuffer)
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, region.indexArr, gl.STATIC_DRAW)
      }
    }
    region.geometryUploaded = !!(
      region.vertexBuffer &&
      region.pixCoordBuffer &&
      (!region.useIndexedMesh || region.indexBuffer)
    )
  }
  return texturesReady && region.geometryUploaded
}
