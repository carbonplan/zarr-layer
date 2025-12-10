/**
 * @module maplibre-shaders
 *
 * WebGL shaders for MapLibre/MapBox custom layer rendering.
 * Adapted from zarr-gl and zarr-cesium.
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

const maplibreVertexShaderSource = `#version 300 es
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

const FRAGMENT_SHADER_PRELUDE = `
uniform bool u_isEquirectangular;
uniform float u_latMin;
uniform float u_latMax;
uniform float scale_y;
uniform float shift_y;

#define PI 3.1415926535897932384626433832795

float mercatorToLat(float y) {
  return 2.0 * atan(exp((0.5 - y) * 2.0 * PI)) - PI / 2.0;
}
`

const FRAGMENT_SHADER_REPROJECT = `
  vec2 sample_coord = pix_coord;
  if (u_isEquirectangular) {
    float sy = scale_y;
    // pix_coord.y is in [0, 1]. 0 is top.
    // mercator Y: 0 (North) -> 1 (South).
    // Top of tile: shift_y - sy. Bottom: shift_y + sy.
    float mercY = (shift_y - sy) + pix_coord.y * 2.0 * sy;
    
    float latRad = mercatorToLat(mercY);
    float latDeg = degrees(latRad);
    
    // Map latDeg to V [0, 1].
    // V=0 should be latMax (North). V=1 should be latMin (South).
    float v = (u_latMax - latDeg) / (u_latMax - u_latMin);
    
    sample_coord.y = v;
  }
  sample_coord = sample_coord * u_texScale + u_texOffset;
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

${FRAGMENT_SHADER_PRELUDE}

in vec2 pix_coord;
out vec4 color;

void main() {
  ${FRAGMENT_SHADER_REPROJECT}
  
  float raw = texture(tex, sample_coord).r;
  float value = raw * u_scaleFactor + u_addOffset;
  
  if (raw == fillValue || isnan(raw) || isnan(value)) {
    discard;
  }
  
  float rescaled = (value - clim.x) / (clim.y - clim.x);
  vec4 c = texture(cmap, vec2(rescaled, 0.5));
  color = vec4(c.rgb, opacity);
  color.rgb *= color.a;
}
`

interface FragmentShaderOptions {
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
        `  float ${name}_raw = texture(${name}, sample_coord).r;\n  float ${name}_val = ${name}_raw * u_scaleFactor + u_addOffset;`
    )
    .join('\n')

  const bandAliases = bands
    .map((name) => `  float ${name} = ${name}_val;`)
    .join('\n')

  const fillValueChecks = bands
    .map(
      (name) =>
        `(${name}_raw == fillValue || isnan(${name}_raw) || isnan(${name}_val))`
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
${FRAGMENT_SHADER_PRELUDE}

in vec2 pix_coord;
out vec4 fragColor;

void main() {
  ${FRAGMENT_SHADER_REPROJECT}
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
  
  if (raw == fillValue || isnan(raw) || isnan(value)) {
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

/**
 * Vertex shader used for Mapbox globe custom layers.
 * Supports two modes:
 * - Tile render mode (u_tile_render = 1): Simple clip space output for renderToTile
 * - Globe render mode (u_tile_render = 0): Full ECEF calculations for regular render
 */
export function createMapboxGlobeVertexShaderSource(): string {
  return `#version 300 es
uniform float scale;
uniform float scale_x;
uniform float scale_y;
uniform float shift_x;
uniform float shift_y;
uniform float u_worldXOffset;
uniform mat4 matrix;
uniform mat4 u_globe_to_merc;
uniform float u_globe_transition;
uniform int u_tile_render;

in vec2 pix_coord_in;
in vec2 vertex;

out vec2 pix_coord;

const float PI = 3.14159265358979323846;
const float GLOBE_RADIUS = 1303.7972938088067;

float mercatorYToLatRad(float y) {
  float t = PI * (1.0 - 2.0 * y);
  return atan(sinh(t));
}

void main() {
  float sx = scale_x > 0.0 ? scale_x : scale;
  float sy = scale_y > 0.0 ? scale_y : scale;

  vec2 merc = vec2(vertex.x * sx + shift_x + u_worldXOffset, -vertex.y * sy + shift_y);

  if (u_tile_render == 1) {
    gl_Position = matrix * vec4(merc, 0.0, 1.0);
  } else {
    vec4 mercClip = matrix * vec4(merc, 0.0, 1.0);
    mercClip /= mercClip.w;

    float lon = (merc.x - 0.5) * 2.0 * PI;
    float lat = mercatorYToLatRad(merc.y);
    float cosLat = cos(lat);
    vec3 ecef = vec3(
      GLOBE_RADIUS * cosLat * sin(lon),
      -GLOBE_RADIUS * sin(lat),
      GLOBE_RADIUS * cosLat * cos(lon)
    );

    vec4 globeClip = matrix * (u_globe_to_merc * vec4(ecef, 1.0));
    globeClip /= globeClip.w;

    gl_Position = mix(globeClip, mercClip, clamp(u_globe_transition, 0.0, 1.0));
  }
  pix_coord = pix_coord_in;
}
`
}
