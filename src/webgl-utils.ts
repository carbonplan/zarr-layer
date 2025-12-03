/**
 * @module gl-utils
 *
 * Low-level WebGL2 utility functions for shader creation, program linking,
 * and generation of color ramp textures used in Cesium and Zarr visualization.
 *
 * These helpers are used internally by rendering providers (e.g., {@link ZarrCubeProvider})
 * to generate color-mapped textures and dynamic shader programs.
 */

import { type ColorScaleProps } from './types';

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
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('Shader compile error:', gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
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
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('Program link error:', gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

/**
 * Creates a flexible 1D color-ramp texture supporting either normalized (0–1)
 * or integer (0–255) color definitions.
 *
 * @param gl - The WebGL2 rendering context.
 * @param colors - Array of RGB colors in normalized `[0–1]` or integer `[0–255]` format.
 * @param opacity - Opacity multiplier between 0 and 1.
 * @returns A {@link WebGLTexture} representing the color ramp, or `null` if creation failed.
 *
 * @example
 * ```ts
 * const texture = createColorRampTexture(gl, [[1, 0, 0], [0, 0, 1]], 0.8);
 * ```
 */
export function createColorRampTexture(
  gl: WebGL2RenderingContext,
  colors: number[][],
  opacity: number
): WebGLTexture | null {
  if (!gl) return null;

  const colorTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, colorTexture);

  const flat = new Uint8Array(colors.length * 4);
  const useFloat = colors[0][0] <= 1.0;

  for (let i = 0; i < colors.length; i++) {
    const c = colors[i];
    flat[i * 4 + 0] = useFloat ? Math.round(c[0] * 255) : c[0];
    flat[i * 4 + 1] = useFloat ? Math.round(c[1] * 255) : c[1];
    flat[i * 4 + 2] = useFloat ? Math.round(c[2] * 255) : c[2];
    flat[i * 4 + 3] = Math.floor(opacity * 255);
  }

  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, colors.length, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, flat);

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  gl.bindTexture(gl.TEXTURE_2D, null);
  return colorTexture;
}

/**
 * Updates an {@link ImageData} object with a new pixel color
 * derived from a numeric value and a {@link ColorScaleProps} colormap.
 *
 * @param value - Numeric data value to visualize.
 * @param pixelIdx - Byte offset within the `ImageData.data` array.
 * @param imgData - Target {@link ImageData} object to update.
 * @param colorScale - Colormap definition with `min`, `max`, and `colors` (see {@link ColorScaleProps}).
 * @param opacity - Global opacity multiplier between 0 and 1.
 * @returns The updated {@link ImageData} with modified pixel values.
 *
 * @example
 * ```ts
 * updateImgData(3.2, idx, imgData, { min: 0, max: 10, colors: viridis }, 0.9);
 * ```
 */
export function updateImgData(
  value: number,
  pixelIdx: number,
  imgData: ImageData,
  colorScale: ColorScaleProps,
  opacity: number
): ImageData {
  if (isNaN(value) || value === null || value === undefined) {
    imgData.data[pixelIdx + 0] = 0;
    imgData.data[pixelIdx + 1] = 0;
    imgData.data[pixelIdx + 2] = 0;
    imgData.data[pixelIdx + 3] = 0;
  } else {
    const normalized = (value - colorScale.min) / (colorScale.max - colorScale.min);
    const [r, g, b, a] = addColor(normalized, colorScale, opacity);
    imgData.data[pixelIdx + 0] = Math.floor(r);
    imgData.data[pixelIdx + 1] = Math.floor(g);
    imgData.data[pixelIdx + 2] = Math.floor(b);
    imgData.data[pixelIdx + 3] = Math.floor(a * 255);
  }
  return imgData;
}

function addColor(
  value: number,
  colorScale: ColorScaleProps,
  opacity: number
): [number, number, number, number] {
  const clamped = Math.min(Math.max(value, 0), 1);

  const colors = colorScale.colors as number[][];
  const n = colors.length;

  if (n === 0) {
    return [1, 1, 1, opacity];
  }

  const scaled = clamped * (n - 1);
  const idx = Math.floor(scaled);
  const t = scaled - idx;

  const [r1, g1, b1] = colors[idx];
  const [r2, g2, b2] = colors[Math.min(idx + 1, n - 1)];

  const r = r1 + (r2 - r1) * t;
  const g = g1 + (g2 - g1) * t;
  const b = b1 + (b2 - b1) * t;
  return [r, g, b, opacity];
}

/**
 * Utility to fetch a uniform location with a helpful error if missing.
 */
export function mustGetUniformLocation(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  name: string
): WebGLUniformLocation {
  const loc = gl.getUniformLocation(program, name);
  if (!loc) {
    throw new Error(`Failed to get uniform location for ${name}`);
  }
  return loc;
}

/**
 * Utility to create a texture or throw.
 */
export function mustCreateTexture(gl: WebGL2RenderingContext): WebGLTexture {
  const tex = gl.createTexture();
  if (!tex) {
    throw new Error('Failed to create texture');
  }
  return tex;
}

/**
 * Utility to create a buffer or throw.
 */
export function mustCreateBuffer(gl: WebGL2RenderingContext): WebGLBuffer {
  const buf = gl.createBuffer();
  if (!buf) {
    throw new Error('Failed to create buffer');
  }
  return buf;
}

/**
 * Utility to create an RGBA framebuffer/texture pair sized to the canvas.
 */
export function mustCreateFramebuffer(
  gl: WebGL2RenderingContext,
  width: number,
  height: number
): { framebuffer: WebGLFramebuffer; texture: WebGLTexture } {
  const framebuffer = gl.createFramebuffer();
  if (!framebuffer) {
    throw new Error('Failed to create framebuffer');
  }

  const texture = gl.createTexture();
  if (!texture) {
    throw new Error('Failed to create texture for framebuffer');
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

  return { framebuffer, texture };
}
