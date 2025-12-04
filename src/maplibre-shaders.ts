/**
 * @module maplibre-shaders
 *
 * WebGL shaders for MapLibre/MapBox custom layer rendering.
 * Adapted from zarr-gl to work with zarr-cesium's colormap and nodata handling.
 */

export interface ShaderData {
  vertexShaderPrelude: string
  define: string
  variantName: string
}

export interface ProjectionData {
  mainMatrix: Float32Array | Float64Array | number[]
  fallbackMatrix: Float32Array | Float64Array | number[]
  tileMercatorCoords: [number, number, number, number]
  clippingPlane: [number, number, number, number]
  projectionTransition: number
}

export function createVertexShaderSource(shaderData?: ShaderData): string {
  if (shaderData && shaderData.vertexShaderPrelude) {
    return `#version 300 es
${shaderData.vertexShaderPrelude}
${shaderData.define}

uniform float scale;
uniform float scale_x;
uniform float scale_y;
uniform float shift_x;
uniform float shift_y;
uniform float u_worldXOffset;

in vec2 pix_coord_in;
in vec2 vertex;

out vec2 pix_coord;

void main() {
  float sx = scale_x > 0.0 ? scale_x : scale;
  float sy = scale_y > 0.0 ? scale_y : scale;
  vec2 a = vec2(vertex.x * sx + shift_x + u_worldXOffset, -vertex.y * sy + shift_y);
  gl_Position = projectTile(a);
  pix_coord = pix_coord_in;
}
`
  }
  return maplibreVertexShaderSource
}

/**
 * Vertex shader for tile rendering (mercator fallback).
 * Transforms tile vertices using scale, shift, and projection matrix uniforms.
 * Vertices are in [-1, 1] and represent a full tile quad.
 * Scale and shift position the tile in mercator [0, 1] space.
 *
 * Note: Y is negated because vertex Y increases upward (+1 is top)
 * but mercator Y increases downward (0 is north, 1 is south).
 */
export const maplibreVertexShaderSource = `#version 300 es
uniform float scale;
uniform float scale_x;
uniform float scale_y;
uniform float shift_x;
uniform float shift_y;
uniform float u_worldXOffset;
uniform mat4 matrix;

in vec2 pix_coord_in;
in vec2 vertex;

out vec2 pix_coord;

void main() {
  float sx = scale_x > 0.0 ? scale_x : scale;
  float sy = scale_y > 0.0 ? scale_y : scale;
  vec2 a = vec2(vertex.x * sx + shift_x + u_worldXOffset, -vertex.y * sy + shift_y);
  gl_Position = matrix * vec4(a, 0.0, 1.0);
  pix_coord = pix_coord_in;
}
`

/**
 * Fragment shader for tile rendering with colormap and fillValue handling.
 * Mirrors carbonplan/maps approach with clim (vec2) and single fillValue.
 */
export const maplibreFragmentShaderSource = `#version 300 es
precision highp float;

uniform vec2 clim;
uniform float opacity;
uniform float fillValue;
uniform float u_scaleFactor;
uniform float u_addOffset;
uniform vec2 u_texScale;
uniform vec2 u_texOffset;

uniform sampler2D tex;
uniform sampler2D cmap;

in vec2 pix_coord;
out vec4 color;

void main() {
  vec2 sample_coord = pix_coord * u_texScale + u_texOffset;
  float raw = texture(tex, sample_coord).r;
  float value = raw * u_scaleFactor + u_addOffset;
  
  if (raw == fillValue || raw != raw || value != value) {
    discard;
  }
  
  float rescaled = (value - clim.x) / (clim.y - clim.x);
  vec4 c = texture(cmap, vec2(rescaled, 0.5));
  color = vec4(c.rgb, opacity);
  color.rgb *= color.a;
}
`

/**
 * Simple vertex shader for rendering framebuffer to screen.
 */
export const renderVertexShaderSource = `#version 300 es
in vec2 vertex;
out vec2 texCoord;
void main() {
  gl_Position = vec4(vertex, 0.0, 1.0);
  texCoord = vertex * 0.5 + 0.5;
}
`

/**
 * Simple fragment shader for rendering framebuffer texture to screen.
 */
export const renderFragmentShaderSource = `#version 300 es
precision highp float;
uniform sampler2D tex;
in vec2 texCoord;
out vec4 fragColor;
void main() {
  fragColor = texture(tex, texCoord);
}
`

export interface FragmentShaderOptions {
  bands: string[]
  customUniforms?: string[]
  customFrag?: string
}

export function createFragmentShaderSource(
  options: FragmentShaderOptions
): string {
  const { bands, customUniforms = [], customFrag } = options
  const hasBands = bands.length > 0

  const bandSamplers = bands
    .map((name) => `uniform sampler2D ${name};`)
    .join('\n')

  const customUniformDecls = customUniforms
    .map((name) => `uniform float ${name};`)
    .join('\n')

  let processedFragBody = customFrag || ''
  const uniformRegex = /uniform\s+\w+\s+(\w+)\s*;/g
  let match
  const extractedUniforms: string[] = []

  while ((match = uniformRegex.exec(processedFragBody)) !== null) {
    if (!customUniforms.includes(match[1])) {
      extractedUniforms.push(match[0])
    }
  }

  processedFragBody = processedFragBody.replace(uniformRegex, '')

  const extraUniformsDecl = extractedUniforms.join('\n')

  const bandReads = bands
    .map(
      (name) =>
        `  float ${name}_raw = texture(${name}, sample_coord).r;
  float ${name}_val = ${name}_raw * u_scaleFactor + u_addOffset;`
    )
    .join('\n')

  const bandAliases = bands
    .map((name) => `  float ${name} = ${name}_val;`)
    .join('\n')

  const fillValueChecks = bands
    .map(
      (name) =>
        `(${name}_raw == fillValue || ${name}_raw != ${name}_raw || ${name}_val != ${name}_val)`
    )
    .join(' || ')

  const commonDiscardChecks = hasBands
    ? `
  if (${fillValueChecks}) {
    discard;
  }
`
    : ''

  return `#version 300 es
precision highp float;

uniform float opacity;
uniform vec2 clim;
uniform float fillValue;
uniform float u_scaleFactor;
uniform float u_addOffset;
uniform vec2 u_texScale;
uniform vec2 u_texOffset;

uniform sampler2D colormap;

${bandSamplers}
${customUniformDecls}
${extraUniformsDecl}

in vec2 pix_coord;
out vec4 fragColor;

void main() {
  vec2 sample_coord = pix_coord * u_texScale + u_texOffset;
${bandReads}
${bandAliases}
${
  processedFragBody
    ? `
${commonDiscardChecks}
${processedFragBody.replace(/gl_FragColor/g, 'fragColor')}`
    : bands.length === 1
    ? `
  float value = ${bands[0]};
  float raw = ${bands[0]}_raw;
  
  if (raw == fillValue || raw != raw || value != value) {
    discard;
  }
  
  float rescaled = (value - clim.x) / (clim.y - clim.x);
  vec4 c = texture(colormap, vec2(rescaled, 0.5));
  fragColor = vec4(c.rgb, opacity);
  fragColor.rgb *= fragColor.a;
`
    : `
  if (${fillValueChecks}) {
    discard;
  }
  
  float value = ${bands[0]};
  float rescaled = (value - clim.x) / (clim.y - clim.x);
  vec4 c = texture(colormap, vec2(rescaled, 0.5));
  fragColor = vec4(c.rgb, opacity);
  fragColor.rgb *= fragColor.a;
`
}
}
`
}
