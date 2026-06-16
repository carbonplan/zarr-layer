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
uniform float u_worldXOffset;
// Eye-coords uniforms (source-projected FLAT path only). u_eye_matrix is the
// flat projection matrix (MapLibre mainMatrix / Mapbox matrix); u_anchor_clip
// is matrix * vec4(regionOrigin, 0, 1) for THIS region, computed per region per
// frame in JS Float64 (regionOrigin = the region's shift_x/shift_y). A visible
// region is near the camera, so anchor_clip is small in clip space:
// Float32-exact and jitter-free. Unused by other variants (location null).
uniform mat4 u_eye_matrix;
uniform vec4 u_anchor_clip;`

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

/**
 * Source-projected FLAT path, eye-coords (render-relative-to-eye) decomposition.
 *
 * vertex.xy are region-local MERCATOR deltas (pre-projected in JS Float64, see
 * encodeMercDelta); scale/shift would restore the region-anchored
 * mercator value. Instead of forming an absolute world coord in Float32 and
 * multiplying by the high-zoom matrix (which quantizes to ~4 px of jitter at
 * z≈19), decompose, anchored at this region's near-camera origin:
 *
 *   matrix·vec4(anchor + delta, 0, 1)
 *     ≡ matrix·vec4(anchor, 0, 1)  +  matrix·vec4(delta, 0, 0)
 *     ≡ u_anchor_clip (JS Float64, near origin)  +  deltaClip (small)
 *
 * Both terms are small in clip space, so their Float32 sum keeps sub-pixel
 * precision. The world-wrap offset (+/-1 for wrapped copies near the
 * antimeridian) is folded into u_anchor_clip on the CPU (per world offset, in
 * Float64) rather than added here as a separate matrix term — otherwise the
 * anchor and the offset would be two ~one-world clip values that cancel in
 * Float32 for the wrapped copy. `merc` is reconstructed at low precision ONLY
 * for the fragment varyings.
 */
const VERTEX_TO_WGS84_TO_MERCATOR = `
  vec2 mercDelta = vec2(vertex.x * sx, vertex.y * sy);
  vec4 deltaClip = u_eye_matrix * vec4(mercDelta, 0.0, 0.0);
  vec2 merc = vec2(
    shift_x + mercDelta.x + u_worldXOffset,
    shift_y + mercDelta.y
  );`

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

/**
 * Mapbox globe projection output (handles tile render vs globe render).
 *
 * For source-projected `wgs84` input, the linear flat/tile endpoint uses the
 * eye-coords sum (u_anchor_clip + deltaClip), which is sub-pixel precise at
 * high zoom. During the globe morph, keep Mapbox's existing absolute Mercator
 * path because the ECEF blend is nonlinear and only active at low/mid zoom.
 */
const projectMapboxGlobe = (eyeCoords: boolean): string => {
  const flatClip = eyeCoords
    ? 'u_anchor_clip + deltaClip'
    : 'matrix * vec4(merc, 0.0, 1.0)'
  const mercClip = eyeCoords ? 'matrix * vec4(merc, 0.0, 1.0)' : flatClip
  const flatEndpoint = eyeCoords
    ? `
  } else if (u_globe_transition >= 0.999999) {
    vec4 flatClip = ${flatClip};
    flatClip /= flatClip.w;
    gl_Position = flatClip;`
    : ''
  return `
  if (u_tile_render == 1) {
    gl_Position = ${flatClip};
${flatEndpoint}
  } else {
    vec4 mercClip = ${mercClip};
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
}

/**
 * ECEF globe path (MapLibre Y-UP). vertex.xy are layer-anchored MERCATOR deltas;
 * restore absolute mercator (the globe is only active at low/mid zoom, where
 * Float32 absolute coords are ample) and invert mercator-Y → latitude. The
 * inverse gudermannian is defined for all finite mercY, so polar vertices
 * (mercY outside [0,1]) reconstruct their true latitude → poles preserved.
 */
const VERTEX_WGS84_TO_ECEF = `
  float mercX = vertex.x * sx + shift_x + u_worldXOffset;
  float mercY = vertex.y * sy + shift_y;

  float lonRad = (mercX - 0.5) * 2.0 * PI;
  float latRad = mercatorYToLatRad(mercY);

  // ECEF unit sphere (MapLibre Y-UP convention)
  float cosLat = cos(latRad);
  vec3 ecef = vec3(sin(lonRad) * cosLat, sin(latRad), cos(lonRad) * cosLat);

  // Flat-map fallback for the globe->mercator transition: mercator coords direct.
  vec2 merc = vec2(mercX, mercY);

  // Geographic normalized coords for the wgs84-lookup fragment path.
  float normLon = mercX;
  float normLat = latRad / PI + 0.5;`

/**
 * ECEF globe path (Mapbox Y-DOWN + radius). Same mercator-delta input and
 * merc→lat inversion as VERTEX_WGS84_TO_ECEF (see there).
 */
const VERTEX_WGS84_TO_ECEF_MAPBOX = `
  float mercX = vertex.x * sx + shift_x + u_worldXOffset;
  float mercY = vertex.y * sy + shift_y;

  float lonRad = (mercX - 0.5) * 2.0 * PI;
  float latRad = mercatorYToLatRad(mercY);

  // Mapbox ECEF: Y-DOWN with explicit radius
  float cosLat = cos(latRad);
  vec3 ecef = vec3(
    GLOBE_RADIUS * cosLat * sin(lonRad),
    -GLOBE_RADIUS * sin(latRad),
    GLOBE_RADIUS * cosLat * cos(lonRad)
  );

  // Geographic normalized coords for the wgs84-lookup fragment path.
  float normLon = mercX;
  float normLat = latRad / PI + 0.5;`

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
  // FUNC_MERCATOR_Y_TO_LAT (inverts Mercator Y → latitude) is needed by:
  //  - regular Mapbox globe (PROJECT_MAPBOX_GLOBE), and
  //  - both ECEF paths, which now reconstruct latitude from mercator-encoded
  //    vertices to preserve polar coverage.
  let helpers = ''
  if (isDirectEcef || projection === 'mapbox') {
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
  } else if (inputSpace === 'wgs84' && projection === 'maplibre') {
    // Eye-coords FLAT path. VERTEX_TO_WGS84_TO_MERCATOR produced
    // deltaClip (matrix * vec4(mercDelta, 0, 0)); u_anchor_clip is the
    // JS-precomputed matrix * vec4(regionOrigin + worldOffset, 0, 1). Their
    // Float32 sum is sub-pixel precise. Skip projectTile so we never form an
    // absolute world coord in Float32.
    //
    // Scoped to MapLibre because MapLibre routes ALL globe transition to the
    // ECEF path above (this branch is only reached when the map is truly flat,
    // where mainMatrix is a linear world->clip map and the decomposition holds).
    projectionOutput = `  gl_Position = u_anchor_clip + deltaClip;`
  } else if (projection === 'maplibre') {
    projectionOutput = PROJECT_MAPLIBRE_GLOBE
  } else {
    // Mapbox flat + globe morph. For the source-projected path, the flat/tile
    // endpoint uses eye-coords; the globe morph keeps absolute mercator.
    projectionOutput = projectMapboxGlobe(inputSpace === 'wgs84')
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
