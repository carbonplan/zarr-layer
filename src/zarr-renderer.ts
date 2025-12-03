import {
  createProgram,
  createShader,
  mustCreateBuffer,
  mustGetUniformLocation,
} from './webgl-utils'
import {
  createVertexShaderSource,
  createFragmentShaderSource,
  type ProjectionData,
  type ShaderData,
} from './maplibre-shaders'
import { tileToKey, tileToScale, type TileTuple } from './maplibre-utils'
import type { MercatorBounds } from './maplibre-utils'
import type { TileRenderCache, TileRenderData } from './zarr-tile-cache'

interface RendererUniforms {
  vmin: number
  vmax: number
  opacity: number
  fillValue: number
  useFillValue: boolean
  noDataMin: number
  noDataMax: number
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
}

interface RenderParams {
  matrix: number[] | Float32Array | Float64Array
  colormapTexture: WebGLTexture
  uniforms: RendererUniforms
  worldOffsets: number[]
  isMultiscale: boolean
  visibleTiles: TileTuple[]
  tileCache: TileRenderCache
  tileSize: number
  vertexArr: Float32Array
  pixCoordArr: Float32Array
  singleImage?: SingleImageParams
  shaderData?: ShaderData
  projectionData?: ProjectionData
  customShaderConfig?: CustomShaderConfig
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
  vminLoc: WebGLUniformLocation | null
  vmaxLoc: WebGLUniformLocation | null
  climLoc: WebGLUniformLocation | null
  opacityLoc: WebGLUniformLocation
  noDataLoc: WebGLUniformLocation | null
  noDataMinLoc: WebGLUniformLocation | null
  noDataMaxLoc: WebGLUniformLocation | null
  useFillValueLoc: WebGLUniformLocation | null
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
  isGlobe: boolean
  useCustomShader: boolean
  bandTexLocs: Map<string, WebGLUniformLocation>
  customUniformLocs: Map<string, WebGLUniformLocation>
}

export class ZarrRenderer {
  private gl: WebGL2RenderingContext
  private fragmentShaderSource: string
  private shaderCache: Map<string, ShaderProgram> = new Map()
  private singleImageGeometryUploaded = false
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

  private getOrCreateProgram(
    shaderData?: ShaderData,
    customShaderConfig?: CustomShaderConfig
  ): ShaderProgram {
    const config = customShaderConfig || this.customShaderConfig
    const useCustomShader = config && config.bands.length > 0
    const variantName = useCustomShader
      ? `custom_${config.bands.join('_')}${shaderData?.variantName ?? ''}`
      : shaderData?.variantName ?? 'mercator'

    const cached = this.shaderCache.get(variantName)
    if (cached) {
      return cached
    }

    const isGlobe = shaderData && shaderData.vertexShaderPrelude ? true : false
    const vertexSource = createVertexShaderSource(shaderData)

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
      matrixLoc: isGlobe
        ? null
        : mustGetUniformLocation(this.gl, program, 'matrix'),
      projMatrixLoc: isGlobe
        ? this.gl.getUniformLocation(program, 'u_projection_matrix')
        : null,
      fallbackMatrixLoc: isGlobe
        ? this.gl.getUniformLocation(program, 'u_projection_fallback_matrix')
        : null,
      tileMercatorCoordsLoc: isGlobe
        ? this.gl.getUniformLocation(
            program,
            'u_projection_tile_mercator_coords'
          )
        : null,
      clippingPlaneLoc: isGlobe
        ? this.gl.getUniformLocation(program, 'u_projection_clipping_plane')
        : null,
      projectionTransitionLoc: isGlobe
        ? this.gl.getUniformLocation(program, 'u_projection_transition')
        : null,
      // Shared uniforms
      opacityLoc: mustGetUniformLocation(this.gl, program, 'opacity'),
      texScaleLoc: mustGetUniformLocation(this.gl, program, 'u_texScale'),
      texOffsetLoc: mustGetUniformLocation(this.gl, program, 'u_texOffset'),
      vertexLoc: this.gl.getAttribLocation(program, 'vertex'),
      pixCoordLoc: this.gl.getAttribLocation(program, 'pix_coord_in'),

      // Conditional uniforms (single vs multi-band)
      vminLoc: useCustomShader
        ? null
        : this.gl.getUniformLocation(program, 'vmin'),
      vmaxLoc: useCustomShader
        ? null
        : this.gl.getUniformLocation(program, 'vmax'),
      climLoc: useCustomShader
        ? this.gl.getUniformLocation(program, 'clim')
        : null,

      noDataLoc: this.gl.getUniformLocation(program, 'nodata'),
      noDataMinLoc: this.gl.getUniformLocation(program, 'u_noDataMin'),
      noDataMaxLoc: this.gl.getUniformLocation(program, 'u_noDataMax'),
      useFillValueLoc: this.gl.getUniformLocation(program, 'u_useFillValue'),
      fillValueLoc: this.gl.getUniformLocation(program, 'u_fillValue'),
      scaleFactorLoc: this.gl.getUniformLocation(program, 'u_scaleFactor'),
      addOffsetLoc: this.gl.getUniformLocation(program, 'u_addOffset'),

      cmapLoc: useCustomShader
        ? null
        : this.gl.getUniformLocation(program, 'cmap'),
      colormapLoc: this.gl.getUniformLocation(program, 'colormap'),
      texLoc: useCustomShader
        ? null
        : this.gl.getUniformLocation(program, 'tex'),

      isGlobe,
      useCustomShader: !!useCustomShader,
      bandTexLocs,
      customUniformLocs,
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
      singleImage,
      shaderData,
      projectionData,
      customShaderConfig,
    } = params

    const shaderProgram = this.getOrCreateProgram(
      shaderData,
      customShaderConfig
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
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

    if (shaderProgram.useCustomShader) {
      if (shaderProgram.climLoc) {
        gl.uniform2f(shaderProgram.climLoc, uniforms.vmin, uniforms.vmax)
      }
    } else {
      if (shaderProgram.vminLoc) {
        gl.uniform1f(shaderProgram.vminLoc, uniforms.vmin)
      }
      if (shaderProgram.vmaxLoc) {
        gl.uniform1f(shaderProgram.vmaxLoc, uniforms.vmax)
      }
    }

    gl.uniform1f(shaderProgram.opacityLoc, uniforms.opacity)
    if (shaderProgram.noDataLoc) {
      gl.uniform1f(shaderProgram.noDataLoc, uniforms.fillValue)
    }
    if (shaderProgram.noDataMinLoc) {
      gl.uniform1f(shaderProgram.noDataMinLoc, uniforms.noDataMin)
    }
    if (shaderProgram.noDataMaxLoc) {
      gl.uniform1f(shaderProgram.noDataMaxLoc, uniforms.noDataMax)
    }
    if (shaderProgram.useFillValueLoc) {
      gl.uniform1i(shaderProgram.useFillValueLoc, uniforms.useFillValue ? 1 : 0)
    }
    if (shaderProgram.fillValueLoc) {
      gl.uniform1f(shaderProgram.fillValueLoc, uniforms.fillValue)
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

    if (shaderProgram.isGlobe && projectionData) {
      if (shaderProgram.projMatrixLoc) {
        gl.uniformMatrix4fv(
          shaderProgram.projMatrixLoc,
          false,
          toFloat32Array(projectionData.mainMatrix)
        )
      }
      if (shaderProgram.fallbackMatrixLoc) {
        gl.uniformMatrix4fv(
          shaderProgram.fallbackMatrixLoc,
          false,
          toFloat32Array(projectionData.fallbackMatrix)
        )
      }
      if (shaderProgram.tileMercatorCoordsLoc) {
        gl.uniform4f(
          shaderProgram.tileMercatorCoordsLoc,
          ...projectionData.tileMercatorCoords
        )
      }
      if (shaderProgram.clippingPlaneLoc) {
        gl.uniform4f(
          shaderProgram.clippingPlaneLoc,
          ...projectionData.clippingPlane
        )
      }
      if (shaderProgram.projectionTransitionLoc) {
        gl.uniform1f(
          shaderProgram.projectionTransitionLoc,
          projectionData.projectionTransition
        )
      }
    } else if (shaderProgram.matrixLoc) {
      gl.uniformMatrix4fv(
        shaderProgram.matrixLoc,
        false,
        toFloat32Array(matrix)
      )
    }

    if (isMultiscale) {
      this.renderTiles(
        shaderProgram,
        visibleTiles,
        worldOffsets,
        tileCache,
        tileSize,
        vertexArr,
        pixCoordArr,
        customShaderConfig
      )
    } else if (singleImage) {
      this.renderSingleImage(
        shaderProgram,
        worldOffsets,
        singleImage,
        vertexArr
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
    vertexArr: Float32Array
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
    } = params

    if (!data || !bounds || !texture || !vertexBuffer || !pixCoordBuffer) {
      return
    }

    const gl = this.gl

    const scaleX = (bounds.x1 - bounds.x0) / 2
    const scaleY = (bounds.y1 - bounds.y0) / 2
    const shiftX = (bounds.x0 + bounds.x1) / 2
    const shiftY = (bounds.y0 + bounds.y1) / 2

    gl.uniform1f(shaderProgram.scaleLoc, 0)
    gl.uniform1f(shaderProgram.scaleXLoc, scaleX)
    gl.uniform1f(shaderProgram.scaleYLoc, scaleY)
    gl.uniform1f(shaderProgram.shiftXLoc, shiftX)
    gl.uniform1f(shaderProgram.shiftYLoc, shiftY)
    gl.uniform2f(shaderProgram.texScaleLoc, 1.0, 1.0)
    gl.uniform2f(shaderProgram.texOffsetLoc, 0.0, 0.0)

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
    customShaderConfig?: CustomShaderConfig
  ) {
    const gl = this.gl

    gl.uniform1f(shaderProgram.scaleXLoc, 0)
    gl.uniform1f(shaderProgram.scaleYLoc, 0)

    const vertexCount = vertexArr.length / 2

    for (const worldOffset of worldOffsets) {
      gl.uniform1f(shaderProgram.worldXOffsetLoc, worldOffset)

      for (const tileTuple of visibleTiles) {
        const [z, x, y] = tileTuple
        const tileKey = tileToKey(tileTuple)
        const tile = tileCache.get(tileKey)

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

        const [scale, shiftX, shiftY] = tileToScale(tileTuple)
        gl.uniform1f(shaderProgram.scaleLoc, scale)
        gl.uniform1f(shaderProgram.shiftXLoc, shiftX)
        gl.uniform1f(shaderProgram.shiftYLoc, shiftY)
        gl.uniform2f(shaderProgram.texScaleLoc, texScale[0], texScale[1])
        gl.uniform2f(shaderProgram.texOffsetLoc, texOffset[0], texOffset[1])

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
            gl.texParameteri(
              gl.TEXTURE_2D,
              gl.TEXTURE_MIN_FILTER,
              this.canUseLinearFloat ? gl.LINEAR : gl.NEAREST
            )
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
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

            const loc = shaderProgram.bandTexLocs.get(bandName)
            if (loc) {
              gl.uniform1i(loc, textureUnit)
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
          gl.texParameteri(
            gl.TEXTURE_2D,
            gl.TEXTURE_MIN_FILTER,
            this.canUseLinearHalfFloat ? gl.LINEAR : gl.NEAREST
          )
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
          const channels = tileToRender.channels ?? 1
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
              ? gl.RG16F
              : channels === 3
              ? gl.RGB16F
              : channels >= 4
              ? gl.RGBA16F
              : gl.R16F

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
