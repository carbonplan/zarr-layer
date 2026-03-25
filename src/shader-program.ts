import {
  createProgram,
  createShader,
  mustGetUniformLocation,
} from './webgl-utils'
import {
  createVertexShader,
  createFragmentShaderSource,
  type ProjectionData,
  type ShaderData,
  type VertexShaderInputSpace,
  type VertexShaderProjection,
} from './shaders'
import type {
  CustomShaderConfig,
  MapboxParams,
  ProjectionMode,
} from './renderer-types'

export interface ShaderProgram {
  program: WebGLProgram
  scaleLoc: WebGLUniformLocation
  scaleXLoc: WebGLUniformLocation
  scaleYLoc: WebGLUniformLocation
  shiftXLoc: WebGLUniformLocation
  shiftYLoc: WebGLUniformLocation
  worldXOffsetLoc: WebGLUniformLocation
  matrixLoc: WebGLUniformLocation | null
  projMatrixLoc: WebGLUniformLocation | null
  fallbackMatrixLoc: WebGLUniformLocation | null
  tileMercatorCoordsLoc: WebGLUniformLocation | null
  clippingPlaneLoc: WebGLUniformLocation | null
  projectionTransitionLoc: WebGLUniformLocation | null
  climLoc: WebGLUniformLocation | null
  opacityLoc: WebGLUniformLocation
  fillValueLoc: WebGLUniformLocation | null
  scaleFactorLoc: WebGLUniformLocation | null
  addOffsetLoc: WebGLUniformLocation | null
  cmapLoc: WebGLUniformLocation | null
  colormapLoc: WebGLUniformLocation | null
  texLoc: WebGLUniformLocation | null
  texScaleLoc: WebGLUniformLocation
  texOffsetLoc: WebGLUniformLocation
  vertexLoc: number
  pixCoordLoc: number
  projectionMode: ProjectionMode
  useCustomShader: boolean
  bandTexLocs: Map<string, WebGLUniformLocation>
  customUniformLocs: Map<string, WebGLUniformLocation>
  globeToMercMatrixLoc?: WebGLUniformLocation | null
  globeTransitionLoc?: WebGLUniformLocation | null
  tileRenderLoc?: WebGLUniformLocation | null
  dataScaleLoc: WebGLUniformLocation | null
  // EPSG:4326 reprojection uniforms
  reprojectLoc: WebGLUniformLocation | null
  latBoundsLoc: WebGLUniformLocation | null
  latIsAscendingLoc: WebGLUniformLocation | null
}

export function resolveProjectionMode(
  useMapbox: boolean = false,
  useWgs84: boolean = false,
  useDirectEcef: boolean = false
): ProjectionMode {
  // For Mapbox (ECEF, proj4, and non-proj4)
  if (useMapbox && useDirectEcef) return 'mapbox-ecef'
  if (useMapbox && useWgs84) return 'mapbox-proj4'
  if (useMapbox) return 'mapbox'
  // For MapLibre: projectTile() handles the standard Mercator-driven paths.
  // The ECEF variant is separate because untiled 4326/proj4 globe rendering
  // needs vertices built directly on the sphere to avoid Mercator pole clipping.
  // Requires MapLibre 3.0+ which provides vertexShaderPrelude.
  if (useDirectEcef) return 'maplibre-ecef'
  if (useWgs84) return 'maplibre-proj4'
  return 'maplibre'
}

export function makeShaderVariantKey(options: {
  projectionMode: ProjectionMode
  shaderData?: ShaderData
  customShaderConfig?: CustomShaderConfig | null
}) {
  const { projectionMode, shaderData, customShaderConfig } = options
  const useCustomShader =
    customShaderConfig && customShaderConfig.bands.length > 0
  // Include shaderData.variantName for both paths to ensure custom shaders
  // recompile when MapLibre changes the vertex shader prelude during globe to merc transitions
  const shaderVariant = shaderData?.variantName ?? 'base'

  const baseVariant =
    useCustomShader && customShaderConfig
      ? ['custom', customShaderConfig.bands.join('_'), shaderVariant].join('_')
      : shaderVariant
  return [baseVariant, projectionMode].join('_')
}

const toFloat32Array = (
  arr: number[] | Float32Array | Float64Array
): Float32Array => {
  if (arr instanceof Float32Array) return arr
  return new Float32Array(arr)
}

// Projection mode helpers - modes are combinations of input space + projection target
const isMapboxMode = (mode: ProjectionMode) =>
  mode === 'mapbox' || mode === 'mapbox-proj4' || mode === 'mapbox-ecef'
const isMaplibreMode = (mode: ProjectionMode) =>
  mode === 'maplibre' || mode === 'maplibre-proj4' || mode === 'maplibre-ecef'

/** Map ProjectionMode to vertex shader options */
function getVertexShaderOptions(projectionMode: ProjectionMode): {
  inputSpace: VertexShaderInputSpace
  projection: VertexShaderProjection
} {
  const inputSpace: VertexShaderInputSpace =
    projectionMode === 'maplibre-ecef' || projectionMode === 'mapbox-ecef'
      ? 'wgs84-direct'
      : projectionMode.includes('proj4')
      ? 'wgs84'
      : 'mercator'

  const projection: VertexShaderProjection = isMapboxMode(projectionMode)
    ? 'mapbox'
    : 'maplibre'

  return { inputSpace, projection }
}

export function createShaderProgram(
  gl: WebGL2RenderingContext,
  options: {
    fragmentShaderSource: string
    shaderData?: ShaderData
    customShaderConfig?: CustomShaderConfig | null
    projectionMode: ProjectionMode
    variantName?: string
  }
): { shaderProgram: ShaderProgram; variantName: string } {
  const {
    fragmentShaderSource,
    shaderData,
    customShaderConfig,
    projectionMode,
  } = options

  const config = customShaderConfig || undefined
  const useCustomShader = config && config.bands.length > 0
  const variantName =
    options.variantName ||
    makeShaderVariantKey({ projectionMode, shaderData, customShaderConfig })

  const { inputSpace, projection } = getVertexShaderOptions(projectionMode)
  const vertexSource = createVertexShader({
    inputSpace,
    projection,
    shaderData,
  })

  const fragmentSource =
    useCustomShader && config
      ? createFragmentShaderSource({
          bands: config.bands,
          customUniforms: config.customUniforms
            ? Object.keys(config.customUniforms)
            : [],
          customFrag: config.customFrag,
        })
      : fragmentShaderSource

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource)
  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource)
  if (!vertexShader || !fragmentShader) {
    throw new Error(`Failed to create shaders for variant: ${variantName}`)
  }

  const program = createProgram(gl, vertexShader, fragmentShader)
  if (!program) {
    throw new Error(`Failed to create program for variant: ${variantName}`)
  }

  const bandTexLocs = new Map<string, WebGLUniformLocation>()
  const customUniformLocs = new Map<string, WebGLUniformLocation>()

  if (useCustomShader && config) {
    for (const bandName of config.bands) {
      const loc = gl.getUniformLocation(program, bandName)
      if (loc) {
        bandTexLocs.set(bandName, loc)
      }
    }
    if (config.customUniforms) {
      for (const uniformName of Object.keys(config.customUniforms)) {
        const loc = gl.getUniformLocation(program, uniformName)
        if (loc) {
          customUniformLocs.set(uniformName, loc)
        }
      }
    }
  }

  // MapLibre modes use projectTile uniforms, Mapbox has its own
  const needsMaplibre = isMaplibreMode(projectionMode)
  const needsMapbox = isMapboxMode(projectionMode)
  const maplibreUniform = (name: string) =>
    needsMaplibre ? gl.getUniformLocation(program, name) : null
  const mapboxUniform = (name: string) =>
    needsMapbox ? gl.getUniformLocation(program, name) : null

  const shaderProgram: ShaderProgram = {
    program,
    scaleLoc: mustGetUniformLocation(gl, program, 'scale'),
    scaleXLoc: mustGetUniformLocation(gl, program, 'scale_x'),
    scaleYLoc: mustGetUniformLocation(gl, program, 'scale_y'),
    shiftXLoc: mustGetUniformLocation(gl, program, 'shift_x'),
    shiftYLoc: mustGetUniformLocation(gl, program, 'shift_y'),
    worldXOffsetLoc: mustGetUniformLocation(gl, program, 'u_worldXOffset'),
    // MapLibre modes use projectTile instead of matrix
    matrixLoc: needsMaplibre
      ? null
      : mustGetUniformLocation(gl, program, 'matrix'),
    projMatrixLoc: maplibreUniform('u_projection_matrix'),
    fallbackMatrixLoc: maplibreUniform('u_projection_fallback_matrix'),
    tileMercatorCoordsLoc: maplibreUniform('u_projection_tile_mercator_coords'),
    clippingPlaneLoc: maplibreUniform('u_projection_clipping_plane'),
    projectionTransitionLoc: maplibreUniform('u_projection_transition'),

    opacityLoc: mustGetUniformLocation(gl, program, 'opacity'),
    texScaleLoc: mustGetUniformLocation(gl, program, 'u_texScale'),
    texOffsetLoc: mustGetUniformLocation(gl, program, 'u_texOffset'),
    vertexLoc: gl.getAttribLocation(program, 'vertex'),
    pixCoordLoc: gl.getAttribLocation(program, 'pix_coord_in'),

    climLoc: gl.getUniformLocation(program, 'clim'),
    fillValueLoc: gl.getUniformLocation(program, 'fillValue'),
    scaleFactorLoc: gl.getUniformLocation(program, 'u_scaleFactor'),
    addOffsetLoc: gl.getUniformLocation(program, 'u_addOffset'),

    cmapLoc: useCustomShader ? null : gl.getUniformLocation(program, 'cmap'),
    colormapLoc: gl.getUniformLocation(program, 'colormap'),
    texLoc: useCustomShader ? null : gl.getUniformLocation(program, 'tex'),

    projectionMode,
    useCustomShader: !!useCustomShader,
    bandTexLocs,
    customUniformLocs,
    globeToMercMatrixLoc: mapboxUniform('u_globe_to_merc'),
    globeTransitionLoc: mapboxUniform('u_globe_transition'),
    tileRenderLoc: mapboxUniform('u_tile_render'),
    dataScaleLoc: gl.getUniformLocation(program, 'u_dataScale'),
    // EPSG:4326 reprojection uniforms
    reprojectLoc: gl.getUniformLocation(program, 'u_reproject'),
    latBoundsLoc: gl.getUniformLocation(program, 'u_latBounds'),
    latIsAscendingLoc: gl.getUniformLocation(program, 'u_latIsAscending'),
  }

  gl.deleteShader(vertexShader)
  gl.deleteShader(fragmentShader)

  return { shaderProgram, variantName }
}

export function applyProjectionUniforms(
  gl: WebGL2RenderingContext,
  shaderProgram: ShaderProgram,
  matrix: number[] | Float32Array | Float64Array,
  projectionData?: ProjectionData,
  mapbox?: MapboxParams,
  isGlobeTileRender?: boolean
) {
  const setMatrix4 = (
    loc: WebGLUniformLocation | null | undefined,
    value?: number[] | Float32Array | Float64Array
  ) => {
    if (loc && value) {
      gl.uniformMatrix4fv(loc, false, toFloat32Array(value))
    }
  }
  const setVec4 = (
    loc: WebGLUniformLocation | null | undefined,
    value?: [number, number, number, number]
  ) => {
    if (loc && value) {
      gl.uniform4f(loc, ...value)
    }
  }
  const setFloat = (
    loc: WebGLUniformLocation | null | undefined,
    value?: number
  ) => {
    if (loc && value !== undefined) {
      gl.uniform1f(loc, value)
    }
  }

  switch (shaderProgram.projectionMode) {
    case 'maplibre':
    case 'maplibre-proj4':
    case 'maplibre-ecef': {
      // All MapLibre modes use projectTile uniforms
      if (!projectionData) return

      setMatrix4(shaderProgram.projMatrixLoc, projectionData.mainMatrix)
      setMatrix4(shaderProgram.fallbackMatrixLoc, projectionData.fallbackMatrix)
      setVec4(
        shaderProgram.tileMercatorCoordsLoc,
        projectionData.tileMercatorCoords
      )
      setVec4(shaderProgram.clippingPlaneLoc, projectionData.clippingPlane)
      setFloat(
        shaderProgram.projectionTransitionLoc,
        projectionData.projectionTransition
      )
      break
    }
    case 'mapbox':
    case 'mapbox-proj4':
    case 'mapbox-ecef': {
      // Mapbox's public custom-layer matrix uses the standard far plane, but
      // the direct globe ECEF path needs the expanded-far variant to match
      // the globe surface depth contract. Tile rendering still uses the tile matrix.
      const mapboxMatrix =
        shaderProgram.projectionMode === 'mapbox-ecef' &&
        !isGlobeTileRender &&
        mapbox?.expandedFarZMercatorMatrix
          ? mapbox.expandedFarZMercatorMatrix
          : matrix
      setMatrix4(shaderProgram.matrixLoc, mapboxMatrix)
      setMatrix4(
        shaderProgram.globeToMercMatrixLoc,
        mapbox?.globeToMercatorMatrix
      )
      setFloat(shaderProgram.globeTransitionLoc, mapbox?.transition ?? 1)
      if (shaderProgram.tileRenderLoc) {
        gl.uniform1i(shaderProgram.tileRenderLoc, isGlobeTileRender ? 1 : 0)
      }
      break
    }
    default: {
      setMatrix4(shaderProgram.matrixLoc, matrix)
      break
    }
  }
}
