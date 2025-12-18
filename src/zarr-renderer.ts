import {
  applyProjectionUniforms,
  createShaderProgram,
  makeShaderVariantKey,
  resolveProjectionMode,
  type ShaderProgram,
} from './shader-program'
import type { ProjectionData, ShaderData } from './shaders'
import type {
  CustomShaderConfig,
  MapboxGlobeParams,
  RendererUniforms,
} from './renderer-types'
import { renderTiles } from './tile-renderer'
import type { TileTuple, MercatorBounds } from './map-utils'
import type { Tiles } from './tiles'

export { type ShaderProgram } from './shader-program'

export class ZarrRenderer {
  readonly gl: WebGL2RenderingContext
  private fragmentShaderSource: string
  private shaderCache: Map<string, ShaderProgram> = new Map()
  private customShaderConfig: CustomShaderConfig | null = null

  constructor(
    gl: WebGL2RenderingContext,
    fragmentShaderSource: string,
    customShaderConfig?: CustomShaderConfig
  ) {
    this.gl = ZarrRenderer.resolveGl(gl)
    this.fragmentShaderSource = fragmentShaderSource
    this.customShaderConfig = customShaderConfig || null
    this.getProgram(undefined, customShaderConfig)
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

  getProgram(
    shaderData?: ShaderData,
    customShaderConfig?: CustomShaderConfig,
    useMapboxGlobe: boolean = false
  ): ShaderProgram {
    const projectionMode = resolveProjectionMode(shaderData, useMapboxGlobe)
    const config = customShaderConfig || this.customShaderConfig
    const variantName = makeShaderVariantKey({
      projectionMode,
      shaderData,
      customShaderConfig: config,
    })

    const cached = this.shaderCache.get(variantName)
    if (cached) {
      return cached
    }

    const { shaderProgram } = createShaderProgram(this.gl, {
      fragmentShaderSource: this.fragmentShaderSource,
      shaderData,
      customShaderConfig: config,
      projectionMode,
      variantName,
    })

    this.shaderCache.set(variantName, shaderProgram)
    return shaderProgram
  }

  applyCommonUniforms(
    shaderProgram: ShaderProgram,
    colormapTexture: WebGLTexture,
    uniforms: RendererUniforms,
    customShaderConfig?: CustomShaderConfig,
    projectionData?: ProjectionData,
    mapboxGlobe?: MapboxGlobeParams,
    matrix?: number[] | Float32Array | Float64Array,
    isGlobeTileRender: boolean = false
  ): void {
    const gl = this.gl

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
    if (shaderProgram.dataScaleLoc) {
      // Compute data scale from clim (same formula used in normalizeDataForTexture)
      const dataScale = Math.max(
        Math.abs(uniforms.clim[0]),
        Math.abs(uniforms.clim[1]),
        1
      )
      gl.uniform1f(shaderProgram.dataScaleLoc, dataScale)
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

    if (matrix) {
      applyProjectionUniforms(
        gl,
        shaderProgram,
        matrix,
        projectionData,
        mapboxGlobe,
        isGlobeTileRender
      )
    }
  }

  renderTiles(
    shaderProgram: ShaderProgram,
    visibleTiles: TileTuple[],
    worldOffsets: number[],
    tileCache: Tiles,
    tileSize: number,
    vertexArr: Float32Array,
    pixCoordArr: Float32Array,
    latIsAscending: boolean | null,
    tileBounds?: Record<string, MercatorBounds>,
    customShaderConfig?: CustomShaderConfig,
    isGlobeTileRender: boolean = false,
    tileTexOverrides?: Record<
      string,
      { texScale: [number, number]; texOffset: [number, number] }
    >
  ): void {
    renderTiles(
      this.gl,
      shaderProgram,
      visibleTiles,
      worldOffsets,
      tileCache,
      tileSize,
      vertexArr,
      pixCoordArr,
      latIsAscending,
      tileBounds,
      customShaderConfig,
      isGlobeTileRender,
      tileTexOverrides
    )
  }

  dispose() {
    const gl = this.gl
    for (const [, shader] of this.shaderCache) {
      gl.deleteProgram(shader.program)
    }
    this.shaderCache.clear()
  }
}
