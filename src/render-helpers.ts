/**
 * @module render-helpers
 *
 * Rendering utilities shared by the region and Mapbox draped-tile paths.
 * Handles band texture setup, binding, geometry buffer binding, and the lazy
 * upload of a region's textures and geometry to the GPU.
 */

import type { ShaderProgram } from './shader-program'
import type { RegionState } from './region-state'
import { configureDataTexture, getTextureFormats } from './webgl-utils'
import {
  allocateLike,
  bandFormatOf,
  bandTextureFormats,
  type BandFormat,
} from './band-format'

/** Texture unit for the band array (0 = main texture, 1 = colormap). */
const BAND_TEXTURE_UNIT = 2

/**
 * Identifies which bands, in which order and storage format, a band texture
 * holds. A texture is only drawable by a shader whose band list and sampler
 * type produce the same key.
 */
export function bandTextureKey(
  bands: readonly string[],
  format: BandFormat = 'float'
): string {
  return `${format}:${bands.join(',')}`
}

/**
 * Point the band sampler at its texture unit.
 * Called once per frame before rendering any tiles/regions.
 */
export function setupBandTextureUniforms(
  gl: WebGL2RenderingContext,
  shaderProgram: ShaderProgram
): void {
  if (!shaderProgram.useCustomShader || !shaderProgram.bandTexLoc) return
  gl.uniform1i(shaderProgram.bandTexLoc, BAND_TEXTURE_UNIT)
}

/**
 * Bind a region's band texture array for drawing. Returns false unless the
 * resident texture holds exactly the bands the shader samples.
 */
export function bindBandTexture(
  gl: WebGL2RenderingContext,
  region: {
    bandTexture: WebGLTexture | null
    bandTextureKey: string | null
  },
  bands: readonly string[],
  format: BandFormat = 'float'
): boolean {
  if (!region.bandTexture) return false
  if (region.bandTextureKey !== bandTextureKey(bands, format)) return false
  gl.activeTexture(gl.TEXTURE0 + BAND_TEXTURE_UNIT)
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, region.bandTexture)
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

function deleteBandTexture(
  gl: WebGL2RenderingContext,
  region: RegionState
): void {
  if (region.bandTexture) gl.deleteTexture(region.bandTexture)
  region.bandTexture = null
  region.bandTextureKey = null
  region.bandTextureBytes = 0
}

const bandTextureLimits = new WeakMap<
  WebGL2RenderingContext,
  { layers: number; size: number; warned: boolean }
>()

/**
 * Whether a texture array of this size fits the context. An upload past the
 * limits fails without an exception, so it is refused up front, with one
 * error per context.
 */
function fitsBandTexture(
  gl: WebGL2RenderingContext,
  width: number,
  height: number,
  layers: number
): boolean {
  let limits = bandTextureLimits.get(gl)
  if (!limits) {
    limits = {
      layers: gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number,
      size: gl.getParameter(gl.MAX_TEXTURE_SIZE) as number,
      warned: false,
    }
    bandTextureLimits.set(gl, limits)
  }
  if (layers <= limits.layers && width <= limits.size && height <= limits.size)
    return true
  if (!limits.warned) {
    limits.warned = true
    console.error(
      `[zarr-layer] ${layers} bands of ${width}x${height} exceed this GPU's ` +
        `texture array limits (${limits.layers} layers, ${limits.size} px).`
    )
  }
  return false
}

/**
 * Upload every band a custom shader samples into one texture array, one band
 * per layer, in the bands' own storage format. Returns false if a band's data
 * is missing, is not stored as `format`, or the texture cannot be allocated,
 * which makes the region undrawable.
 *
 * The band arrays are released once uploaded: the texture is the only copy
 * the region keeps, and anything that needs different bands refetches (from
 * the decoded chunk cache, when it still holds the chunk).
 */
function ensureBandTexture(
  gl: WebGL2RenderingContext,
  region: RegionState,
  bands: readonly string[],
  format: BandFormat
): boolean {
  const key = bandTextureKey(bands, format)
  if (region.bandTexture && region.bandTextureKey === key) return true

  // Checked before allocating: a region missing a band is retried every
  // frame until it is refetched or evicted.
  if (!bands.every((band) => region.bandData.has(band))) return false
  if (!fitsBandTexture(gl, region.width, region.height, bands.length)) {
    return false
  }

  const first = region.bandData.get(bands[0])
  if (!first || bandFormatOf(first) !== format) return false

  const layerSize = region.width * region.height
  const packed = allocateLike(first, layerSize * bands.length)
  for (let layer = 0; layer < bands.length; layer++) {
    const data = region.bandData.get(bands[layer])
    if (!data || data.constructor !== first.constructor) return false
    packed.set(data, layer * layerSize)
  }

  if (!region.bandTexture) region.bandTexture = gl.createTexture()
  if (!region.bandTexture) return false

  const {
    internalFormat,
    format: glFormat,
    type,
  } = bandTextureFormats(gl, first)
  gl.activeTexture(gl.TEXTURE0)
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, region.bandTexture)
  configureDataTexture(gl, gl.TEXTURE_2D_ARRAY)
  // Rows of 1- and 2-byte texels need not be 4-byte aligned.
  const alignment =
    format === 'float' ? null : gl.getParameter(gl.UNPACK_ALIGNMENT)
  if (alignment !== null) gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
  gl.texImage3D(
    gl.TEXTURE_2D_ARRAY,
    0,
    internalFormat,
    region.width,
    region.height,
    bands.length,
    0,
    glFormat,
    type,
    packed
  )
  if (alignment !== null) gl.pixelStorei(gl.UNPACK_ALIGNMENT, alignment)
  region.bandTextureKey = key
  region.bandTextureBytes = packed.byteLength
  region.bandData.clear()
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
 *
 * `bandFormat` is the storage format the shader's band sampler reads. A
 * region stored in another format (a fallback from a level with a different
 * dtype) is reported undrawable rather than drawn through the wrong sampler.
 */
export function ensureRegionGpuResources(
  gl: WebGL2RenderingContext,
  region: RegionState,
  requiredBands?: readonly string[],
  bandFormat: BandFormat = 'float'
): boolean {
  if (!region.vertexArr || !region.pixCoordArr || !region.indexArr) return false

  const bandRendering = !!requiredBands && requiredBands.length > 0
  let texturesReady: boolean

  if (bandRendering) {
    if (region.texture) {
      // Switched from main-texture rendering; nothing samples it now.
      gl.deleteTexture(region.texture)
      region.texture = null
      region.textureUploaded = false
    }
    texturesReady = ensureBandTexture(gl, region, requiredBands, bandFormat)
  } else {
    deleteBandTexture(gl, region)
    // A region fetched for a band-sampling shader has no interleaved copy, so
    // it stays undrawable here until the refetch that follows the switch.
    if (!region.data) return false
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
    if (!region.indexBuffer) region.indexBuffer = gl.createBuffer()
    if (region.indexBuffer) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, region.indexBuffer)
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, region.indexArr, gl.STATIC_DRAW)
    }
    region.geometryUploaded = !!(
      region.vertexBuffer &&
      region.pixCoordBuffer &&
      region.indexBuffer
    )
  }
  return texturesReady && region.geometryUploaded
}
