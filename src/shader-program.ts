import {
  createProgram,
  createShader,
  mustGetUniformLocation,
} from './webgl-utils'
import {
  createVertexShaderSource,
  createFragmentShaderSource,
  createMapboxGlobeVertexShaderSource,
  type ProjectionData,
  type ShaderData,
} from './shaders'
import type {
  CustomShaderConfig,
  MapboxGlobeParams,
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
  isEquirectangularLoc: WebGLUniformLocation | null
  latMinLoc: WebGLUniformLocation | null
  latMaxLoc: WebGLUniformLocation | null
  dataScaleLoc: WebGLUniformLocation | null
}

export function resolveProjectionMode(
  shaderData?: ShaderData,
  useMapboxGlobe: boolean = false
): ProjectionMode {
  if (useMapboxGlobe) return 'mapbox-globe'
  if (shaderData?.vertexShaderPrelude) return 'maplibre-globe'
  return 'mercator'
}

export function makeShaderVariantKey(options: {
  projectionMode: ProjectionMode
  shaderData?: ShaderData
  customShaderConfig?: CustomShaderConfig | null
}) {
  const { projectionMode, shaderData, customShaderConfig } = options
  const useCustomShader =
    customShaderConfig && customShaderConfig.bands.length > 0
  const baseVariant =
    useCustomShader && customShaderConfig
      ? ['custom', customShaderConfig.bands.join('_')].join('_')
      : shaderData?.variantName ?? 'base'
  return [baseVariant, projectionMode].join('_')
}

const toFloat32Array = (
  arr: number[] | Float32Array | Float64Array
): Float32Array => {
  if (arr instanceof Float32Array) return arr
  return new Float32Array(arr)
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

  const vertexSource =
    projectionMode === 'mapbox-globe'
      ? createMapboxGlobeVertexShaderSource()
      : createVertexShaderSource(shaderData)

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

  const needsGlobeProjection = projectionMode === 'maplibre-globe'
  const globeUniform = (name: string) =>
    needsGlobeProjection ? gl.getUniformLocation(program, name) : null

  const shaderProgram: ShaderProgram = {
    program,
    scaleLoc: mustGetUniformLocation(gl, program, 'scale'),
    scaleXLoc: mustGetUniformLocation(gl, program, 'scale_x'),
    scaleYLoc: mustGetUniformLocation(gl, program, 'scale_y'),
    shiftXLoc: mustGetUniformLocation(gl, program, 'shift_x'),
    shiftYLoc: mustGetUniformLocation(gl, program, 'shift_y'),
    worldXOffsetLoc: mustGetUniformLocation(gl, program, 'u_worldXOffset'),
    matrixLoc:
      projectionMode === 'maplibre-globe'
        ? null
        : mustGetUniformLocation(gl, program, 'matrix'),
    projMatrixLoc: globeUniform('u_projection_matrix'),
    fallbackMatrixLoc: globeUniform('u_projection_fallback_matrix'),
    tileMercatorCoordsLoc: globeUniform('u_projection_tile_mercator_coords'),
    clippingPlaneLoc: globeUniform('u_projection_clipping_plane'),
    projectionTransitionLoc: globeUniform('u_projection_transition'),

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
    globeToMercMatrixLoc:
      projectionMode === 'mapbox-globe'
        ? gl.getUniformLocation(program, 'u_globe_to_merc')
        : null,
    globeTransitionLoc:
      projectionMode === 'mapbox-globe'
        ? gl.getUniformLocation(program, 'u_globe_transition')
        : null,
    tileRenderLoc:
      projectionMode === 'mapbox-globe'
        ? gl.getUniformLocation(program, 'u_tile_render')
        : null,
    isEquirectangularLoc: gl.getUniformLocation(program, 'u_isEquirectangular'),
    latMinLoc: gl.getUniformLocation(program, 'u_latMin'),
    latMaxLoc: gl.getUniformLocation(program, 'u_latMax'),
    dataScaleLoc: gl.getUniformLocation(program, 'u_dataScale'),
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
  mapboxGlobe?: MapboxGlobeParams,
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
    case 'maplibre-globe': {
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
    case 'mapbox-globe': {
      setMatrix4(shaderProgram.matrixLoc, matrix)
      setMatrix4(
        shaderProgram.globeToMercMatrixLoc,
        mapboxGlobe?.globeToMercatorMatrix
      )
      setFloat(shaderProgram.globeTransitionLoc, mapboxGlobe?.transition ?? 0)
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
