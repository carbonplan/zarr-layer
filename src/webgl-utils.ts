// utils borrowed from zarr-cesium/src/webgl-utils.ts

/**
 * Creates and compiles a WebGL shader from source code.
 *
 * @param gl - The WebGL2 rendering context.
 * @param type - Shader type (`gl.VERTEX_SHADER` or `gl.FRAGMENT_SHADER`).
 * @param source - GLSL source code for the shader.
 * @returns The compiled {@link WebGLShader} instance, or `null` if compilation failed.
 *
 * @example
 * ```ts
 * const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
 * const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
 * ```
 */
export function createShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string
): WebGLShader | null {
  const shader = gl.createShader(type)
  if (!shader) return null

  gl.shaderSource(shader, source)
  gl.compileShader(shader)

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('Shader compile error:', gl.getShaderInfoLog(shader))
    gl.deleteShader(shader)
    return null
  }

  return shader
}

/**
 * Creates and links a WebGL program using the specified vertex and fragment shaders.
 *
 * @param gl - The WebGL2 rendering context.
 * @param vertexShader - Compiled vertex shader.
 * @param fragmentShader - Compiled fragment shader.
 * @returns The linked {@link WebGLProgram}, or `null` if linking failed.
 *
 * @example
 * ```ts
 * const program = createProgram(gl, vertexShader, fragmentShader);
 * gl.useProgram(program);
 * ```
 */
export function createProgram(
  gl: WebGL2RenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader
): WebGLProgram | null {
  const program = gl.createProgram()
  if (!program) return null

  gl.attachShader(program, vertexShader)
  gl.attachShader(program, fragmentShader)
  gl.linkProgram(program)

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('Program link error:', gl.getProgramInfoLog(program))
    gl.deleteProgram(program)
    return null
  }

  return program
}

/**
 * Utility to fetch a uniform location with a helpful error if missing.
 */
export function mustGetUniformLocation(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  name: string
): WebGLUniformLocation {
  const loc = gl.getUniformLocation(program, name)
  if (!loc) {
    throw new Error(`Failed to get uniform location for ${name}`)
  }
  return loc
}

/**
 * Utility to create a texture or throw.
 */
export function mustCreateTexture(gl: WebGL2RenderingContext): WebGLTexture {
  const tex = gl.createTexture()
  if (!tex) {
    throw new Error('Failed to create texture')
  }
  return tex
}

/**
 * Utility to create a buffer or throw.
 */
export function mustCreateBuffer(gl: WebGL2RenderingContext): WebGLBuffer {
  const buf = gl.createBuffer()
  if (!buf) {
    throw new Error('Failed to create buffer')
  }
  return buf
}

/**
 * Returns the WebGL texture format and internal format for a given number of channels.
 * Used for uploading Float32 data textures with varying channel counts.
 *
 * @param gl - The WebGL2 rendering context.
 * @param channels - Number of channels (1-4).
 * @returns Object with `format` and `internalFormat` WebGL constants.
 */
export function getTextureFormats(
  gl: WebGL2RenderingContext,
  channels: number
): { format: GLenum; internalFormat: GLenum } {
  const format =
    channels === 2
      ? gl.RG
      : channels === 3
      ? gl.RGB
      : channels >= 4
      ? gl.RGBA
      : gl.RED
  const internalFormat =
    channels === 2
      ? gl.RG32F
      : channels === 3
      ? gl.RGB32F
      : channels >= 4
      ? gl.RGBA32F
      : gl.R32F
  return { format, internalFormat }
}

/**
 * Configures a data texture with NEAREST filtering and CLAMP_TO_EDGE wrapping.
 * Call after binding the texture with gl.bindTexture(gl.TEXTURE_2D, texture).
 */
export function configureDataTexture(gl: WebGL2RenderingContext) {
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
}

/**
 * Normalize data for texture upload to ensure half-float safe range on mobile GPUs.
 * Uses fixedScale to normalize data (frozen to initial clim).
 * Fill values are converted to NaN for reliable detection.
 *
 * @param data - Raw float data to normalize
 * @param fillValue - Fill value to convert to NaN
 * @param fixedScale - Fixed scale factor for normalization
 * @returns Object with normalized data and scale factor
 */
export function normalizeDataForTexture(
  data: Float32Array,
  fillValue: number | null,
  fixedScale: number
): { normalized: Float32Array; scale: number } {
  const normalized = new Float32Array(data.length)

  for (let i = 0; i < data.length; i++) {
    const v = data[i]
    // Convert fill values to NaN (fill values may exceed half-float range on mobile GPUs)
    // NaN is preserved through texture reads and caught by the shader's NaN check
    if ((fillValue !== null && v === fillValue) || v !== v) {
      normalized[i] = NaN
    } else {
      normalized[i] = v / fixedScale
    }
  }

  return { normalized, scale: fixedScale }
}

/**
 * Interleave separate band arrays into a single packed array.
 * For single-channel data, returns the band directly (no copy).
 */
export function interleaveBands(
  bands: Float32Array[],
  channels: number
): Float32Array {
  if (channels === 1 && bands.length === 1) {
    return bands[0]
  }
  const pixelCount = bands[0].length
  const result = new Float32Array(pixelCount * channels)
  for (let i = 0; i < pixelCount; i++) {
    for (let c = 0; c < channels; c++) {
      result[i * channels + c] = bands[c][i]
    }
  }
  return result
}

export function createSubdividedQuad(subdivisions: number): {
  vertexArr: Float32Array
  texCoordArr: Float32Array
} {
  const vertices: number[] = []
  const texCoords: number[] = []
  const step = 2 / subdivisions
  const texStep = 1 / subdivisions

  const pushVertex = (col: number, row: number) => {
    const x = -1 + col * step
    const y = 1 - row * step
    const u = col * texStep
    const v = row * texStep
    vertices.push(x, y)
    texCoords.push(u, v)
  }

  for (let row = 0; row < subdivisions; row++) {
    for (let col = 0; col <= subdivisions; col++) {
      pushVertex(col, row)
      pushVertex(col, row + 1)
    }
    if (row < subdivisions - 1) {
      pushVertex(subdivisions, row + 1)
      pushVertex(0, row + 1)
    }
  }

  return {
    vertexArr: new Float32Array(vertices),
    texCoordArr: new Float32Array(texCoords),
  }
}
