/**
 * @module maplibre-shaders
 *
 * WebGL shaders for MapLibre/Mapbox custom layer rendering.
 * Consolidated vertex shaders built from reusable components.
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

// Small clip-space depth bias keeps the direct Mapbox globe path from
// z-fighting with the globe surface. This is only applied to the custom ECEF path.
const MAPBOX_ECEF_DEPTH_BIAS = 5e-4

// ============================================================================
// Reusable Shader Components
// ============================================================================

/** Common uniforms for all vertex shaders */
const UNIFORMS_COMMON = `
uniform float scale;
uniform float scale_x;
uniform float scale_y;
uniform float shift_x;
uniform float shift_y;
uniform float u_worldXOffset;`

/** Additional uniforms for Mapbox globe projection */
const UNIFORMS_MAPBOX_GLOBE = `
uniform mat4 matrix;
uniform mat4 u_globe_to_merc;
uniform float u_globe_transition;
uniform int u_tile_render;`

/** Common vertex inputs and outputs */
const INPUTS_OUTPUTS = `
in vec2 pix_coord_in;
in vec2 vertex;

out vec2 pix_coord;
out vec2 v_mercatorPos;
out vec2 v_wgs84Pos;`

/** Scale handling (shared by all shaders) */
const SCALE_HANDLING = `
  float sx = scale_x > 0.0 ? scale_x : scale;
  float sy = scale_y > 0.0 ? scale_y : scale;`

/** Transform vertex from local space to normalized Mercator coordinates */
const VERTEX_TO_MERCATOR = `
  vec2 merc = vec2(vertex.x * sx + shift_x + u_worldXOffset, -vertex.y * sy + shift_y);`

/** Transform vertex from local space to normalized WGS84, then to Mercator */
const VERTEX_TO_WGS84_TO_MERCATOR = `
  // vertex.xy are in local [-1, 1] space for this region
  // scale/shift transform to absolute normalized 4326 [0,1] on world
  float normLon = vertex.x * sx + shift_x + u_worldXOffset;
  float normLat = vertex.y * sy + shift_y;

  // Convert normalized [0,1] to degrees
  float lon = normLon * 360.0 - 180.0;
  float lat = normLat * 180.0 - 90.0;

  // Clamp latitude to Mercator limits to avoid infinity at poles
  lat = clamp(lat, -MERCATOR_LAT_LIMIT, MERCATOR_LAT_LIMIT);

  // Mercator projection
  float lambda = radians(lon);
  float phi = radians(lat);
  float mercY_raw = log(tan((PI / 2.0 + phi) / 2.0));

  // Normalize mercator output to [0,1]
  float mercX = (lambda / PI + 1.0) / 2.0;
  float mercY = (1.0 - mercY_raw / PI) / 2.0;
  vec2 merc = vec2(mercX, mercY);`

/** Individual shader constants (composed as needed) */
const CONST_PI = `const float PI = 3.14159265358979323846;`
const CONST_MERCATOR_LAT_LIMIT = `const float MERCATOR_LAT_LIMIT = 85.05112878;`
const CONST_GLOBE_RADIUS = `const float GLOBE_RADIUS = 1303.7972938088067;`

/** Helper function for Mapbox globe: convert Mercator Y to latitude radians */
const FUNC_MERCATOR_Y_TO_LAT = `
float mercatorYToLatRad(float y) {
  float t = PI * (1.0 - 2.0 * y);
  return atan(sinh(t));
}`

/** MapLibre globe projection output (uses projectTile from prelude) */
const PROJECT_MAPLIBRE_GLOBE = `
  gl_Position = projectTile(merc);`

/** Mapbox globe projection output (handles tile render vs globe render) */
const PROJECT_MAPBOX_GLOBE = `
  if (u_tile_render == 1) {
    gl_Position = matrix * vec4(merc, 0.0, 1.0);
  } else {
    vec4 mercClip = matrix * vec4(merc, 0.0, 1.0);
    mercClip /= mercClip.w;

    float lonRad = (merc.x - 0.5) * 2.0 * PI;
    float latRad = mercatorYToLatRad(merc.y);
    float cosLat = cos(latRad);
    vec3 ecef = vec3(
      GLOBE_RADIUS * cosLat * sin(lonRad),
      -GLOBE_RADIUS * sin(latRad),
      GLOBE_RADIUS * cosLat * cos(lonRad)
    );

    vec4 globeClip = matrix * (u_globe_to_merc * vec4(ecef, 1.0));
    globeClip /= globeClip.w;

    gl_Position = mix(globeClip, mercClip, clamp(u_globe_transition, 0.0, 1.0));
  }`

/** Transform vertex from local space to WGS84 normalized coords, then compute ECEF + Mercator fallback */
const VERTEX_WGS84_TO_ECEF = `
  float normLon = vertex.x * sx + shift_x + u_worldXOffset;
  float normLat = vertex.y * sy + shift_y;

  // WGS84 normalized [0,1] to radians
  float lonRad = (normLon - 0.5) * 2.0 * PI;
  float latDeg = normLat * 180.0 - 90.0;
  float latRad = latDeg * PI / 180.0;

  // ECEF unit sphere (MapLibre Y-UP convention)
  float cosLat = cos(latRad);
  vec3 ecef = vec3(sin(lonRad) * cosLat, sin(latRad), cos(lonRad) * cosLat);

  // Clamped Mercator fallback for flat-map transition
  float clampedLatDeg = clamp(latDeg, -MERCATOR_LAT_LIMIT, MERCATOR_LAT_LIMIT);
  float clampedLatRad = clampedLatDeg * PI / 180.0;
  float mercY_raw = log(tan((PI / 2.0 + clampedLatRad) / 2.0));
  vec2 merc = vec2(normLon, (1.0 - mercY_raw / PI) / 2.0);`

/** Transform vertex from local space to WGS84 normalized coords, then compute ECEF (Mapbox Y-DOWN + radius) */
const VERTEX_WGS84_TO_ECEF_MAPBOX = `
  float normLon = vertex.x * sx + shift_x + u_worldXOffset;
  float normLat = vertex.y * sy + shift_y;

  // WGS84 normalized [0,1] to radians
  float lonRad = (normLon - 0.5) * 2.0 * PI;
  float latDeg = normLat * 180.0 - 90.0;
  float latRad = latDeg * PI / 180.0;

  // Mapbox ECEF: Y-DOWN with explicit radius
  float cosLat = cos(latRad);
  vec3 ecef = vec3(
    GLOBE_RADIUS * cosLat * sin(lonRad),
    -GLOBE_RADIUS * sin(latRad),
    GLOBE_RADIUS * cosLat * cos(lonRad)
  );`

/** Mapbox ECEF projection output for the direct untiled globe path. */
const PROJECT_MAPBOX_ECEF = `
  // This path is only used at the fully-globe endpoint. During Mapbox's
  // globe->mercator zoom morph, the layer switches back to the draped path so
  // Mapbox can handle the transition with its internal globe/mercator matrices.
  gl_Position = matrix * (u_globe_to_merc * vec4(ecef, 1.0));
  gl_Position.z -= ${MAPBOX_ECEF_DEPTH_BIAS.toExponential(6)} * gl_Position.w;`

/** MapLibre ECEF projection output with globe/flat transition blend.
 * Unlike projectTile(merc), this path starts from WGS84-derived sphere coords,
 * which preserves polar coverage for untiled EPSG:4326/proj4 data.
 */
const PROJECT_MAPLIBRE_ECEF = `
  vec4 globePos = u_projection_matrix * vec4(ecef, 1.0);

  // Backface clipping
  float clipZ = 1.0 - (dot(ecef, u_projection_clipping_plane.xyz) + u_projection_clipping_plane.w);
  globePos.z = clipZ * globePos.w;

  // Mercator fallback for flat-map transition
  vec4 flatPos = u_projection_fallback_matrix * vec4(merc, 0.0, 1.0);

  // Transition blend (matches MapLibre's interpolateProjection)
  float t = u_projection_transition;
  vec4 result;
  result.xyw = mix(flatPos.xyw, globePos.xyw, t);
  result.z = mix(0.0, globePos.z, clamp((t - 0.2) / 0.8, 0.0, 1.0));
  gl_Position = result;`

// ============================================================================
// Vertex Shader Types
// ============================================================================

export type VertexShaderInputSpace = 'mercator' | 'wgs84' | 'wgs84-direct'
export type VertexShaderProjection = 'maplibre' | 'mapbox'

interface VertexShaderOptions {
  inputSpace: VertexShaderInputSpace
  projection: VertexShaderProjection
  shaderData?: ShaderData
}

// ============================================================================
// Unified Vertex Shader Generator
// ============================================================================

/**
 * Create a vertex shader from composable parts.
 *
 * @param options.inputSpace - 'mercator', 'wgs84' (needs Mercator transform), or 'wgs84-direct' (ECEF)
 * @param options.projection - 'maplibre' or 'mapbox'
 * @param options.shaderData - Required for maplibre projection (provides projectTile prelude)
 */
export function createVertexShader(options: VertexShaderOptions): string {
  const { inputSpace, projection, shaderData } = options
  const isDirectEcef = inputSpace === 'wgs84-direct'
  const isMapboxDirectEcef = isDirectEcef && projection === 'mapbox'

  // Build uniforms section
  let uniforms: string
  let prelude = ''
  let define = ''

  if (isDirectEcef && projection === 'mapbox') {
    // Mapbox ECEF: no prelude, use Mapbox uniforms
    uniforms = UNIFORMS_COMMON + UNIFORMS_MAPBOX_GLOBE
  } else if (isDirectEcef || projection === 'maplibre') {
    // MapLibre paths (both projectTile and direct ECEF) use the prelude which
    // declares PI, projection uniforms, and projectTile(). The ECEF path
    // references those uniforms directly instead of calling projectTile().
    if (!shaderData) {
      throw new Error('shaderData required for MapLibre projection modes')
    }
    prelude = shaderData.vertexShaderPrelude
    define = shaderData.define
    uniforms = UNIFORMS_COMMON
  } else {
    // mapbox non-ECEF
    uniforms = UNIFORMS_COMMON + UNIFORMS_MAPBOX_GLOBE
  }

  // Build constants section
  // PI: only needed for Mapbox (no prelude). MapLibre prelude defines PI for all other paths.
  // MERCATOR_LAT_LIMIT: needed for wgs84 and wgs84-direct (Mercator fallback clamping)
  const constants = [
    !shaderData ? CONST_PI : '',
    inputSpace === 'wgs84' || (isDirectEcef && projection !== 'mapbox')
      ? CONST_MERCATOR_LAT_LIMIT
      : '',
    projection === 'mapbox' ? CONST_GLOBE_RADIUS : '',
  ]
    .filter(Boolean)
    .join('\n')

  // Build helper functions
  // FUNC_MERCATOR_Y_TO_LAT only needed for regular Mapbox globe (inverts Mercator Y to lat).
  // Mapbox ECEF gets latitude directly from WGS84 input, so it doesn't need this helper.
  let helpers = ''
  if (projection === 'mapbox' && !isDirectEcef) {
    helpers = FUNC_MERCATOR_Y_TO_LAT
  }

  // Build coordinate transform
  let coordTransform: string
  if (isMapboxDirectEcef) {
    coordTransform = VERTEX_WGS84_TO_ECEF_MAPBOX
  } else if (isDirectEcef) {
    coordTransform = VERTEX_WGS84_TO_ECEF
  } else if (inputSpace === 'wgs84') {
    coordTransform = VERTEX_TO_WGS84_TO_MERCATOR
  } else {
    coordTransform = VERTEX_TO_MERCATOR
  }

  // Build projection output
  let projectionOutput: string
  if (isMapboxDirectEcef) {
    projectionOutput = PROJECT_MAPBOX_ECEF
  } else if (isDirectEcef) {
    projectionOutput = PROJECT_MAPLIBRE_ECEF
  } else if (projection === 'maplibre') {
    projectionOutput = PROJECT_MAPLIBRE_GLOBE
  } else {
    projectionOutput = PROJECT_MAPBOX_GLOBE
  }

  // Set v_wgs84Pos: meaningful for direct ECEF, zero for other paths
  const wgs84PosAssignment = isDirectEcef
    ? '  v_wgs84Pos = vec2(normLon, normLat);'
    : '  v_wgs84Pos = vec2(0.0);'
  const mercatorPosAssignment = isMapboxDirectEcef
    ? '  v_mercatorPos = vec2(0.0);'
    : '  v_mercatorPos = merc;'

  // Compose final shader
  return `#version 300 es
${prelude}
${define}
${uniforms}
${INPUTS_OUTPUTS}
${constants}
${helpers}

void main() {
${SCALE_HANDLING}
${coordTransform}
${projectionOutput}
  pix_coord = pix_coord_in;
${mercatorPosAssignment}
${wgs84PosAssignment}
}
`
}

// ============================================================================
// Fragment Shaders
// ============================================================================

// Fragment shader constants
const FRAG_CONST_PI = `const float PI = 3.14159265358979323846;`

/** Inverse Mercator projection: (x, y) in radians -> (lon, lat) in degrees */
const FUNC_MERCATOR_INVERT = `
vec2 mercatorInvert(float x, float y) {
  float lambda = x;
  float phi = 2.0 * atan(exp(y)) - PI / 2.0;
  return vec2(degrees(lambda), degrees(phi));
}
`

/** Fragment shader reprojection logic for EPSG:4326 data */
const FRAGMENT_SHADER_REPROJECT = `
  vec2 sample_coord;

  if (u_reproject == 1) {
    // EPSG:4326 reprojection: invert Mercator to lat/lon for texture lookup
    // v_mercatorPos is normalized [0,1] where y=0 is north, y=1 is south
    // Convert to Mercator radians: y=0 -> PI (north), y=1 -> -PI (south)
    float mercY = PI * (1.0 - 2.0 * v_mercatorPos.y);
    vec2 lonLat = mercatorInvert(0.0, mercY);
    float lat = lonLat.y;

    // Map latitude to texture V coordinate based on data orientation
    float latRange = u_latBounds.y - u_latBounds.x;
    float texV;
    if (u_latIsAscending == 1) {
      // Row 0 = south (latMin), row N = north (latMax)
      texV = (lat - u_latBounds.x) / latRange;
    } else {
      // Row 0 = north (latMax), row N = south (latMin)
      texV = (u_latBounds.y - lat) / latRange;
    }

    // X coordinate is linear (longitude)
    sample_coord = vec2(pix_coord.x, texV) * u_texScale + u_texOffset;
  } else if (u_reproject == 2) {
    // WGS84 direct lookup: v_wgs84Pos carries normalized WGS84 coords from ECEF vertex shader
    float lat = v_wgs84Pos.y * 180.0 - 90.0;
    float latRange = u_latBounds.y - u_latBounds.x;
    float texV;
    if (u_latIsAscending == 1) {
      texV = (lat - u_latBounds.x) / latRange;
    } else {
      texV = (u_latBounds.y - lat) / latRange;
    }
    sample_coord = vec2(pix_coord.x, texV) * u_texScale + u_texOffset;
  } else {
    // Standard linear texture lookup
    sample_coord = pix_coord * u_texScale + u_texOffset;
  }
`

/**
 * Fragment shader for tile rendering with colormap and fillValue handling.
 * Supports EPSG:4326 reprojection via Mercator inversion in fragment shader.
 */
export const maplibreFragmentShaderSource = `#version 300 es
precision highp float;

uniform vec2 clim;
uniform float opacity;
uniform float fillValue;
uniform float u_scaleFactor;
uniform float u_addOffset;
uniform float u_dataScale;
uniform vec2 u_texScale;
uniform vec2 u_texOffset;

// EPSG:4326 reprojection uniforms
uniform int u_reproject;      // 0 = no reprojection, 1 = Mercator inversion, 2 = WGS84 direct lookup
uniform vec2 u_latBounds;     // (latMin, latMax) in degrees
uniform int u_latIsAscending; // 1 = row 0 is south, 0 = row 0 is north

uniform sampler2D tex;
uniform sampler2D cmap;

in vec2 pix_coord;
in vec2 v_mercatorPos;
in vec2 v_wgs84Pos;
out vec4 color;

${FRAG_CONST_PI}
${FUNC_MERCATOR_INVERT}

void main() {
${FRAGMENT_SHADER_REPROJECT}
  float texVal = texture(tex, sample_coord).r;

  // NaN check (fill values converted to NaN during normalization)
  if (isnan(texVal)) {
    discard;
  }

  float raw = texVal * u_dataScale;
  float value = raw * u_scaleFactor + u_addOffset;

  if (isnan(value)) {
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

// Compiled once at module load to avoid recompilation on every shader creation
const UNIFORM_REGEX = /uniform\s+\w+\s+(\w+)\s*;/g

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
  // Reset lastIndex since we reuse the regex
  UNIFORM_REGEX.lastIndex = 0
  let match
  const extractedUniforms: string[] = []

  while ((match = UNIFORM_REGEX.exec(processedFragBody)) !== null) {
    if (!customUniforms.includes(match[1])) {
      extractedUniforms.push(match[0])
    }
  }

  processedFragBody = processedFragBody.replace(UNIFORM_REGEX, '')

  const extraUniformsDecl = extractedUniforms.join('\n')

  const bandReads = bands
    .map(
      (name) =>
        `  float ${name}_tex = texture(${name}, sample_coord).r;\n  float ${name}_raw = ${name}_tex * u_dataScale;\n  float ${name}_val = ${name}_raw * u_scaleFactor + u_addOffset;`
    )
    .join('\n')

  const bandAliases = bands
    .map((name) => `  float ${name} = ${name}_val;`)
    .join('\n')

  const fillValueChecks = bands
    .map((name) => `(isnan(${name}_tex) || isnan(${name}_val))`)
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
uniform float u_dataScale;
uniform vec2 u_texScale;
uniform vec2 u_texOffset;

// EPSG:4326 reprojection uniforms
uniform int u_reproject;      // 0 = no reprojection, 1 = Mercator inversion, 2 = WGS84 direct lookup
uniform vec2 u_latBounds;     // (latMin, latMax) in degrees
uniform int u_latIsAscending; // 1 = row 0 is south, 0 = row 0 is north

uniform sampler2D colormap;

${bandSamplers}
${customUniformDecls}
${extraUniformsDecl}

in vec2 pix_coord;
in vec2 v_mercatorPos;
in vec2 v_wgs84Pos;
out vec4 fragColor;

${FRAG_CONST_PI}
${FUNC_MERCATOR_INVERT}

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
  if (isnan(${bands[0]}_tex) || isnan(${bands[0]})) {
    discard;
  }

  float rescaled = (${bands[0]} - clim.x) / (clim.y - clim.x);
  vec4 c = texture(colormap, vec2(rescaled, 0.5));
  fragColor = vec4(c.rgb, opacity);
  fragColor.rgb *= fragColor.a;
`
    : `
  if (${fillValueChecks}) {
    discard;
  }

  float rescaled = (${bands[0]} - clim.x) / (clim.y - clim.x);
  vec4 c = texture(colormap, vec2(rescaled, 0.5));
  fragColor = vec4(c.rgb, opacity);
  fragColor.rgb *= fragColor.a;
`
}
}
`
}
