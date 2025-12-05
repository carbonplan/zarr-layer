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
import { tileToKey, tileToScale, type TileTuple } from './map-utils'
import type { MercatorBounds } from './map-utils'
import type { TileRenderCache, TileRenderData } from './zarr-tile-cache'

interface RendererUniforms {
  clim: [number, number]
  opacity: number
  fillValue: number | null
  scaleFactor: number
  offset: number
}

export interface CustomShaderConfig {
  bands: string[]
  customFrag?: string
  customUniforms?: Record<string, number>
}

function toFloat32Array(
  arr: number[] | Float32Array | Float64Array
): Float32Array {
  if (arr instanceof Float32Array) {
    return arr
  }
  return new Float32Array(arr)
}

interface SingleImageParams {
  data: Float32Array | null
  width: number
  height: number
  bounds: MercatorBounds | null
  texture: WebGLTexture | null
  vertexBuffer: WebGLBuffer | null
  pixCoordBuffer: WebGLBuffer | null
  pixCoordArr: Float32Array
  geometryVersion: number
}

interface MapboxGlobeParams {
  projection: { name: string }
  globeToMercatorMatrix: number[] | Float32Array | Float64Array
  transition: number
}

type ProjectionMode = 'mercator' | 'maplibre-globe' | 'mapbox-globe'

interface RenderParams {
  matrix: number[] | Float32Array | Float64Array
  colormapTexture: WebGLTexture
  uniforms: RendererUniforms
  worldOffsets: number[]
  isMultiscale: boolean
  visibleTiles: TileTuple[]
  tileSize: number
  vertexArr: Float32Array
  pixCoordArr: Float32Array
  tileBounds?: Record<string, MercatorBounds>
  tileCache?: TileRenderCache
  singleImage?: SingleImageParams
  shaderData?: ShaderData
  projectionData?: ProjectionData
  customShaderConfig?: CustomShaderConfig
  mapboxGlobe?: MapboxGlobeParams
  mapboxTileRender?: boolean
  tileTexOverrides?: Record<
    string,
    { texScale: [number, number]; texOffset: [number, number] }
  >
  tileOverride?: {
    scaleX: number
    scaleY: number
    shiftX: number
    shiftY: number
    texScale: [number, number]
    texOffset: [number, number]
  }
}

interface ShaderProgram {
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
  // Mapbox globe specific uniforms
  globeToMercMatrixLoc?: WebGLUniformLocation | null
  globeTransitionLoc?: WebGLUniformLocation | null
  tileRenderLoc?: WebGLUniformLocation | null
  isEquirectangularLoc: WebGLUniformLocation | null
  latMinLoc: WebGLUniformLocation | null
  latMaxLoc: WebGLUniformLocation | null
}

export class ZarrRenderer {
  private gl: WebGL2RenderingContext
  private fragmentShaderSource: string
  private shaderCache: Map<string, ShaderProgram> = new Map()
  private singleImageGeometryUploaded = false
  private singleImageGeometryVersion: number | null = null
  private customShaderConfig: CustomShaderConfig | null = null
  private canUseLinearFloat: boolean = false
  private canUseLinearHalfFloat: boolean = false

  constructor(
    gl: WebGL2RenderingContext,
    fragmentShaderSource: string,
    customShaderConfig?: CustomShaderConfig
  ) {
    this.gl = ZarrRenderer.resolveGl(gl)
    this.canUseLinearFloat = !!this.gl.getExtension('OES_texture_float_linear')
    this.canUseLinearHalfFloat = !!this.gl.getExtension(
      'OES_texture_half_float_linear'
    )
    this.fragmentShaderSource = fragmentShaderSource
    this.customShaderConfig = customShaderConfig || null
    this.getOrCreateProgram(undefined, customShaderConfig)
  }

  updateMultiBandConfig(config: CustomShaderConfig | null) {
    if (config && this.customShaderConfig) {
      const bandsChanged =
        JSON.stringify(config.bands) !==
        JSON.stringify(this.customShaderConfig.bands)
      const fragChanged =
        config.customFrag !== this.customShaderConfig.customFrag
      if (bandsChanged || fragChanged) {
        this.shaderCache.clear()
      }
    } else if (config !== this.customShaderConfig) {
      this.shaderCache.clear()
    }
    this.customShaderConfig = config
  }

  private static resolveGl(gl: WebGL2RenderingContext): WebGL2RenderingContext {
    const hasWebGL2Methods =
      gl &&
      typeof gl.getUniformLocation === 'function' &&
      typeof gl.drawBuffers === 'function'
    if (hasWebGL2Methods) {
      gl.getExtension('EXT_color_buffer_float')
      gl.getExtension('OES_texture_float_linear')
      return gl
    }
    throw new Error('Invalid WebGL2 context: missing required WebGL2 methods')
  }

  private resolveProjectionMode(
    shaderData?: ShaderData,
    useMapboxGlobe: boolean = false
  ): ProjectionMode {
    if (useMapboxGlobe) return 'mapbox-globe'
    if (shaderData?.vertexShaderPrelude) return 'maplibre-globe'
    return 'mercator'
  }

  private applyProjectionUniforms(
    shaderProgram: ShaderProgram,
    matrix: number[] | Float32Array | Float64Array,
    projectionData?: ProjectionData,
    mapboxGlobe?: MapboxGlobeParams,
    mapboxTileRender?: boolean
  ) {
    const gl = this.gl
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
        setMatrix4(
          shaderProgram.fallbackMatrixLoc,
          projectionData.fallbackMatrix
        )
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
          gl.uniform1i(shaderProgram.tileRenderLoc, mapboxTileRender ? 1 : 0)
        }
        break
      }
      default: {
        setMatrix4(shaderProgram.matrixLoc, matrix)
        break
      }
    }
  }

  private getOrCreateProgram(
    shaderData?: ShaderData,
    customShaderConfig?: CustomShaderConfig,
    useMapboxGlobe: boolean = false
  ): ShaderProgram {
    const projectionMode = this.resolveProjectionMode(
      shaderData,
      useMapboxGlobe
    )
    const config = customShaderConfig || this.customShaderConfig
    const useCustomShader = config && config.bands.length > 0
    const baseVariant =
      useCustomShader && config
        ? ['custom', config.bands.join('_')].join('_')
        : shaderData?.variantName ?? 'base'
    const variantName = [baseVariant, projectionMode].join('_')

    const cached = this.shaderCache.get(variantName)
    if (cached) {
      return cached
    }

    const vertexSource =
      projectionMode === 'mapbox-globe'
        ? createMapboxGlobeVertexShaderSource()
        : createVertexShaderSource(shaderData)

    let fragmentSource: string
    if (useCustomShader && config) {
      fragmentSource = createFragmentShaderSource({
        bands: config.bands,
        customUniforms: config.customUniforms
          ? Object.keys(config.customUniforms)
          : [],
        customFrag: config.customFrag,
      })
    } else {
      fragmentSource = this.fragmentShaderSource
    }

    const vertexShader = createShader(
      this.gl,
      this.gl.VERTEX_SHADER,
      vertexSource
    )
    const fragmentShader = createShader(
      this.gl,
      this.gl.FRAGMENT_SHADER,
      fragmentSource
    )
    if (!vertexShader || !fragmentShader) {
      throw new Error(`Failed to create shaders for variant: ${variantName}`)
    }

    const program = createProgram(this.gl, vertexShader, fragmentShader)
    if (!program) {
      throw new Error(`Failed to create program for variant: ${variantName}`)
    }

    const bandTexLocs = new Map<string, WebGLUniformLocation>()
    const customUniformLocs = new Map<string, WebGLUniformLocation>()

    if (useCustomShader && config) {
      for (const bandName of config.bands) {
        const loc = this.gl.getUniformLocation(program, bandName)
        if (loc) {
          bandTexLocs.set(bandName, loc)
        }
      }

      if (config.customUniforms) {
        for (const uniformName of Object.keys(config.customUniforms)) {
          const loc = this.gl.getUniformLocation(program, uniformName)
          if (loc) {
            customUniformLocs.set(uniformName, loc)
          }
        }
      }
    }

    const needsGlobeProjection = projectionMode === 'maplibre-globe'
    const globeUniform = (name: string) =>
      needsGlobeProjection ? this.gl.getUniformLocation(program, name) : null

    const shaderProgram: ShaderProgram = {
      program,
      scaleLoc: mustGetUniformLocation(this.gl, program, 'scale'),
      scaleXLoc: mustGetUniformLocation(this.gl, program, 'scale_x'),
      scaleYLoc: mustGetUniformLocation(this.gl, program, 'scale_y'),
      shiftXLoc: mustGetUniformLocation(this.gl, program, 'shift_x'),
      shiftYLoc: mustGetUniformLocation(this.gl, program, 'shift_y'),
      worldXOffsetLoc: mustGetUniformLocation(
        this.gl,
        program,
        'u_worldXOffset'
      ),
      matrixLoc:
        projectionMode === 'maplibre-globe'
          ? null
          : mustGetUniformLocation(this.gl, program, 'matrix'),
      projMatrixLoc: globeUniform('u_projection_matrix'),
      fallbackMatrixLoc: globeUniform('u_projection_fallback_matrix'),
      tileMercatorCoordsLoc: globeUniform('u_projection_tile_mercator_coords'),
      clippingPlaneLoc: globeUniform('u_projection_clipping_plane'),
      projectionTransitionLoc: globeUniform('u_projection_transition'),

      opacityLoc: mustGetUniformLocation(this.gl, program, 'opacity'),
      texScaleLoc: mustGetUniformLocation(this.gl, program, 'u_texScale'),
      texOffsetLoc: mustGetUniformLocation(this.gl, program, 'u_texOffset'),
      vertexLoc: this.gl.getAttribLocation(program, 'vertex'),
      pixCoordLoc: this.gl.getAttribLocation(program, 'pix_coord_in'),

      climLoc: this.gl.getUniformLocation(program, 'clim'),
      fillValueLoc: this.gl.getUniformLocation(program, 'fillValue'),
      scaleFactorLoc: this.gl.getUniformLocation(program, 'u_scaleFactor'),
      addOffsetLoc: this.gl.getUniformLocation(program, 'u_addOffset'),

      cmapLoc: useCustomShader
        ? null
        : this.gl.getUniformLocation(program, 'cmap'),
      colormapLoc: this.gl.getUniformLocation(program, 'colormap'),
      texLoc: useCustomShader
        ? null
        : this.gl.getUniformLocation(program, 'tex'),

      projectionMode,
      useCustomShader: !!useCustomShader,
      bandTexLocs,
      customUniformLocs,
      globeToMercMatrixLoc:
        projectionMode === 'mapbox-globe'
          ? this.gl.getUniformLocation(program, 'u_globe_to_merc')
          : null,
      globeTransitionLoc:
        projectionMode === 'mapbox-globe'
          ? this.gl.getUniformLocation(program, 'u_globe_transition')
          : null,
      tileRenderLoc:
        projectionMode === 'mapbox-globe'
          ? this.gl.getUniformLocation(program, 'u_tile_render')
          : null,
      isEquirectangularLoc: this.gl.getUniformLocation(
        program,
        'u_isEquirectangular'
      ),
      latMinLoc: this.gl.getUniformLocation(program, 'u_latMin'),
      latMaxLoc: this.gl.getUniformLocation(program, 'u_latMax'),
    }

    this.gl.deleteShader(vertexShader)
    this.gl.deleteShader(fragmentShader)

    this.shaderCache.set(variantName, shaderProgram)
    return shaderProgram
  }

  render(params: RenderParams) {
    const {
      matrix,
      colormapTexture,
      uniforms,
      worldOffsets,
      isMultiscale,
      visibleTiles,
      tileCache,
      tileSize,
      vertexArr,
      pixCoordArr,
      tileBounds,
      singleImage,
      shaderData,
      projectionData,
      customShaderConfig,
      mapboxGlobe,
      mapboxTileRender,
      tileTexOverrides,
      tileOverride,
    } = params

    const shaderProgram = this.getOrCreateProgram(
      shaderData,
      customShaderConfig,
      !!mapboxGlobe || !!mapboxTileRender
    )

    const gl = this.gl
    gl.useProgram(shaderProgram.program)

    gl.enable(gl.BLEND)
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)

    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, colormapTexture)
    if (shaderProgram.cmapLoc) {
      gl.uniform1i(shaderProgram.cmapLoc, 1)
    }
    if (shaderProgram.colormapLoc) {
      gl.uniform1i(shaderProgram.colormapLoc, 1)
    }

    if (shaderProgram.climLoc) {
      gl.uniform2f(shaderProgram.climLoc, uniforms.clim[0], uniforms.clim[1])
    }

    gl.uniform1f(shaderProgram.opacityLoc, uniforms.opacity)
    if (shaderProgram.fillValueLoc) {
      gl.uniform1f(shaderProgram.fillValueLoc, uniforms.fillValue ?? NaN)
    }
    if (shaderProgram.scaleFactorLoc) {
      gl.uniform1f(shaderProgram.scaleFactorLoc, uniforms.scaleFactor)
    }
    if (shaderProgram.addOffsetLoc) {
      gl.uniform1f(shaderProgram.addOffsetLoc, uniforms.offset)
    }
    gl.uniform2f(shaderProgram.texScaleLoc, 1.0, 1.0)
    gl.uniform2f(shaderProgram.texOffsetLoc, 0.0, 0.0)

    if (customShaderConfig?.customUniforms) {
      for (const [name, value] of Object.entries(
        customShaderConfig.customUniforms
      )) {
        const loc = shaderProgram.customUniformLocs.get(name)
        if (loc) {
          gl.uniform1f(loc, value)
        }
      }
    }

    this.applyProjectionUniforms(
      shaderProgram,
      matrix,
      projectionData,
      mapboxGlobe,
      mapboxTileRender
    )

    if (isMultiscale) {
      if (!tileCache) {
        console.warn('Missing tile cache for multiscale render, skipping frame')
        return
      }
      this.renderTiles(
        shaderProgram,
        visibleTiles,
        worldOffsets,
        tileCache,
        tileSize,
        vertexArr,
        pixCoordArr,
        tileBounds,
        customShaderConfig,
        !!mapboxTileRender,
        tileTexOverrides
      )
    } else if (singleImage) {
      this.renderSingleImage(
        shaderProgram,
        worldOffsets,
        singleImage,
        vertexArr,
        tileOverride
      )
    }
  }

  dispose() {
    const gl = this.gl
    for (const [, shader] of this.shaderCache) {
      gl.deleteProgram(shader.program)
    }
    this.shaderCache.clear()
  }

  resetSingleImageGeometry() {
    this.singleImageGeometryUploaded = false
  }

  private renderSingleImage(
    shaderProgram: ShaderProgram,
    worldOffsets: number[],
    params: SingleImageParams,
    vertexArr: Float32Array,
    tileOverride?: {
      scaleX: number
      scaleY: number
      shiftX: number
      shiftY: number
      texScale: [number, number]
      texOffset: [number, number]
    }
  ) {
    const {
      data,
      bounds,
      texture,
      vertexBuffer,
      pixCoordBuffer,
      width,
      height,
      pixCoordArr,
      geometryVersion,
    } = params

    if (
      this.singleImageGeometryVersion === null ||
      this.singleImageGeometryVersion !== geometryVersion
    ) {
      this.singleImageGeometryUploaded = false
      this.singleImageGeometryVersion = geometryVersion
    }

    if (!data || !bounds || !texture || !vertexBuffer || !pixCoordBuffer) {
      return
    }

    const gl = this.gl

    const scaleX =
      tileOverride?.scaleX !== undefined
        ? tileOverride.scaleX
        : (bounds.x1 - bounds.x0) / 2
    const scaleY =
      tileOverride?.scaleY !== undefined
        ? tileOverride.scaleY
        : (bounds.y1 - bounds.y0) / 2
    const shiftX =
      tileOverride?.shiftX !== undefined
        ? tileOverride.shiftX
        : (bounds.x0 + bounds.x1) / 2
    const shiftY =
      tileOverride?.shiftY !== undefined
        ? tileOverride.shiftY
        : (bounds.y0 + bounds.y1) / 2

    gl.uniform1f(shaderProgram.scaleLoc, 0)
    gl.uniform1f(shaderProgram.scaleXLoc, scaleX)
    gl.uniform1f(shaderProgram.scaleYLoc, scaleY)
    gl.uniform1f(shaderProgram.shiftXLoc, shiftX)
    gl.uniform1f(shaderProgram.shiftYLoc, shiftY)
    const texScale = tileOverride?.texScale ?? [1.0, 1.0]
    const texOffset = tileOverride?.texOffset ?? [0.0, 0.0]
    gl.uniform2f(shaderProgram.texScaleLoc, texScale[0], texScale[1])
    gl.uniform2f(shaderProgram.texOffsetLoc, texOffset[0], texOffset[1])

    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer)
    if (!this.singleImageGeometryUploaded) {
      gl.bufferData(gl.ARRAY_BUFFER, vertexArr, gl.STATIC_DRAW)
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, pixCoordBuffer)
    if (!this.singleImageGeometryUploaded) {
      gl.bufferData(gl.ARRAY_BUFFER, pixCoordArr, gl.STATIC_DRAW)
      this.singleImageGeometryUploaded = true
    }

    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.uniform1i(shaderProgram.texLoc, 0)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
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

    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer)
    gl.enableVertexAttribArray(shaderProgram.vertexLoc)
    gl.vertexAttribPointer(shaderProgram.vertexLoc, 2, gl.FLOAT, false, 0, 0)

    gl.bindBuffer(gl.ARRAY_BUFFER, pixCoordBuffer)
    gl.enableVertexAttribArray(shaderProgram.pixCoordLoc)
    gl.vertexAttribPointer(shaderProgram.pixCoordLoc, 2, gl.FLOAT, false, 0, 0)

    const vertexCount = vertexArr.length / 2

    for (const worldOffset of worldOffsets) {
      gl.uniform1f(shaderProgram.worldXOffsetLoc, worldOffset)
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, vertexCount)
    }
  }

  private renderTiles(
    shaderProgram: ShaderProgram,
    visibleTiles: TileTuple[],
    worldOffsets: number[],
    tileCache: TileRenderCache,
    tileSize: number,
    vertexArr: Float32Array,
    pixCoordArr: Float32Array,
    tileBounds?: Record<string, MercatorBounds>,
    customShaderConfig?: CustomShaderConfig,
    mapboxTileRender: boolean = false,
    tileTexOverrides?: Record<
      string,
      { texScale: [number, number]; texOffset: [number, number] }
    >
  ) {
    const gl = this.gl

    if (shaderProgram.useCustomShader && customShaderConfig) {
      let textureUnit = 2
      for (const bandName of customShaderConfig.bands) {
        const loc = shaderProgram.bandTexLocs.get(bandName)
        if (loc) {
          gl.uniform1i(loc, textureUnit)
        }
        textureUnit++
      }
    }

    const vertexCount = vertexArr.length / 2

    for (const worldOffset of worldOffsets) {
      gl.uniform1f(
        shaderProgram.worldXOffsetLoc,
        mapboxTileRender ? 0 : worldOffset
      )

      for (const tileTuple of visibleTiles) {
        const [z, x, y] = tileTuple
        const tileKey = tileToKey(tileTuple)
        const tile = tileCache.get(tileKey)
        const bounds = tileBounds?.[tileKey]

        let tileToRender: TileRenderData | null = null
        let renderTileKey = tileKey
        let texScale: [number, number] = [1, 1]
        let texOffset: [number, number] = [0, 0]

        if (tile && tile.data) {
          tileToRender = tile
        } else {
          const parent = this.findBestParentTile(z, x, y, tileCache)
          if (parent) {
            tileToRender = parent.tile
            renderTileKey = tileToKey([
              parent.ancestorZ,
              parent.ancestorX,
              parent.ancestorY,
            ])
            const levelDiff = z - parent.ancestorZ
            const divisor = Math.pow(2, levelDiff)
            const localX = x % divisor
            const localY = y % divisor
            texScale = [1 / divisor, 1 / divisor]
            texOffset = [localX / divisor, localY / divisor]
          }
        }

        if (!tileToRender || !tileToRender.data) continue

        if (bounds) {
          const scaleX = (bounds.x1 - bounds.x0) / 2
          const scaleY = (bounds.y1 - bounds.y0) / 2
          const shiftX = (bounds.x0 + bounds.x1) / 2
          const shiftY = (bounds.y0 + bounds.y1) / 2
          gl.uniform1f(shaderProgram.scaleLoc, 0)
          gl.uniform1f(shaderProgram.scaleXLoc, scaleX)
          gl.uniform1f(shaderProgram.scaleYLoc, scaleY)
          gl.uniform1f(shaderProgram.shiftXLoc, shiftX)
          gl.uniform1f(shaderProgram.shiftYLoc, shiftY)

          if (shaderProgram.isEquirectangularLoc) {
            gl.uniform1i(
              shaderProgram.isEquirectangularLoc,
              bounds.latMin !== undefined ? 1 : 0
            )
          }
          if (shaderProgram.latMinLoc && bounds.latMin !== undefined) {
            gl.uniform1f(shaderProgram.latMinLoc, bounds.latMin)
          }
          if (shaderProgram.latMaxLoc && bounds.latMax !== undefined) {
            gl.uniform1f(shaderProgram.latMaxLoc, bounds.latMax)
          }
        } else {
          const [scale, shiftX, shiftY] = tileToScale(tileTuple)
          gl.uniform1f(shaderProgram.scaleLoc, scale)
          gl.uniform1f(shaderProgram.scaleXLoc, 0)
          gl.uniform1f(shaderProgram.scaleYLoc, 0)
          gl.uniform1f(shaderProgram.shiftXLoc, shiftX)
          gl.uniform1f(shaderProgram.shiftYLoc, shiftY)

          if (shaderProgram.isEquirectangularLoc) {
            gl.uniform1i(shaderProgram.isEquirectangularLoc, 0)
          }
        }
        if (mapboxTileRender && tileTexOverrides?.[tileKey]) {
          const override = tileTexOverrides[tileKey]
          gl.uniform2f(
            shaderProgram.texScaleLoc,
            override.texScale[0],
            override.texScale[1]
          )
          gl.uniform2f(
            shaderProgram.texOffsetLoc,
            override.texOffset[0],
            override.texOffset[1]
          )
        } else {
          gl.uniform2f(shaderProgram.texScaleLoc, texScale[0], texScale[1])
          gl.uniform2f(shaderProgram.texOffsetLoc, texOffset[0], texOffset[1])
        }

        gl.bindBuffer(gl.ARRAY_BUFFER, tileToRender.vertexBuffer)
        if (!tileToRender.geometryUploaded) {
          gl.bufferData(gl.ARRAY_BUFFER, vertexArr, gl.STATIC_DRAW)
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, tileToRender.pixCoordBuffer)
        if (!tileToRender.geometryUploaded) {
          gl.bufferData(gl.ARRAY_BUFFER, pixCoordArr, gl.STATIC_DRAW)
          tileToRender.geometryUploaded = true
        }

        if (shaderProgram.useCustomShader && customShaderConfig) {
          let textureUnit = 2
          let missingBandData = false
          for (const bandName of customShaderConfig.bands) {
            const bandData = tileToRender.bandData.get(bandName)
            if (!bandData) {
              missingBandData = true
              break
            }

            let bandTex = tileToRender.bandTextures.get(bandName)
            if (!bandTex) {
              const newTex = tileCache.ensureBandTexture(
                renderTileKey,
                bandName
              )
              if (newTex) {
                bandTex = newTex
                tileToRender.bandTextures.set(bandName, bandTex)
              }
            }
            if (!bandTex) {
              missingBandData = true
              break
            }

            gl.activeTexture(gl.TEXTURE0 + textureUnit)
            gl.bindTexture(gl.TEXTURE_2D, bandTex)
            if (!tileToRender.bandTexturesConfigured.has(bandName)) {
              gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
              gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
              gl.texParameteri(
                gl.TEXTURE_2D,
                gl.TEXTURE_WRAP_S,
                gl.CLAMP_TO_EDGE
              )
              gl.texParameteri(
                gl.TEXTURE_2D,
                gl.TEXTURE_WRAP_T,
                gl.CLAMP_TO_EDGE
              )
              tileToRender.bandTexturesConfigured.add(bandName)
            }
            if (!tileToRender.bandTexturesUploaded.has(bandName)) {
              gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                gl.R32F,
                tileSize,
                tileSize,
                0,
                gl.RED,
                gl.FLOAT,
                bandData
              )
              tileToRender.bandTexturesUploaded.add(bandName)
            }

            textureUnit++
          }
          if (missingBandData) {
            continue
          }
        } else {
          gl.activeTexture(gl.TEXTURE0)
          gl.bindTexture(gl.TEXTURE_2D, tileToRender.tileTexture)
          if (shaderProgram.texLoc) {
            gl.uniform1i(shaderProgram.texLoc, 0)
          }
          if (!tileToRender.textureConfigured) {
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
            tileToRender.textureConfigured = true
          }
          const channels = tileToRender.channels ?? 1
          const format =
            channels === 2
              ? gl.RG
              : channels === 3
              ? gl.RGB
              : channels >= 4
              ? gl.RGBA
              : gl.RED
          // Use full 32-bit float textures so large sentinel values (fillValue)
          // survive the upload and can be discarded in the shader.
          const internalFormat =
            channels === 2
              ? gl.RG32F
              : channels === 3
              ? gl.RGB32F
              : channels >= 4
              ? gl.RGBA32F
              : gl.R32F

          if (!tileToRender.textureUploaded) {
            gl.texImage2D(
              gl.TEXTURE_2D,
              0,
              internalFormat,
              tileSize,
              tileSize,
              0,
              format,
              gl.FLOAT,
              tileToRender.data
            )
            tileToRender.textureUploaded = true
          }
        }

        gl.bindBuffer(gl.ARRAY_BUFFER, tileToRender.vertexBuffer)
        gl.enableVertexAttribArray(shaderProgram.vertexLoc)
        gl.vertexAttribPointer(
          shaderProgram.vertexLoc,
          2,
          gl.FLOAT,
          false,
          0,
          0
        )

        gl.bindBuffer(gl.ARRAY_BUFFER, tileToRender.pixCoordBuffer)
        gl.enableVertexAttribArray(shaderProgram.pixCoordLoc)
        gl.vertexAttribPointer(
          shaderProgram.pixCoordLoc,
          2,
          gl.FLOAT,
          false,
          0,
          0
        )

        gl.drawArrays(gl.TRIANGLE_STRIP, 0, vertexCount)
      }
    }
  }

  private findBestParentTile(
    z: number,
    x: number,
    y: number,
    tileCache: TileRenderCache
  ): {
    tile: TileRenderData
    ancestorZ: number
    ancestorX: number
    ancestorY: number
  } | null {
    let ancestorZ = z - 1
    let ancestorX = Math.floor(x / 2)
    let ancestorY = Math.floor(y / 2)

    while (ancestorZ >= 0) {
      const parentKey = tileToKey([ancestorZ, ancestorX, ancestorY])
      const parentTile = tileCache.get(parentKey)
      if (parentTile && parentTile.data) {
        return { tile: parentTile, ancestorZ, ancestorX, ancestorY }
      }
      ancestorZ--
      ancestorX = Math.floor(ancestorX / 2)
      ancestorY = Math.floor(ancestorY / 2)
    }
    return null
  }
}
