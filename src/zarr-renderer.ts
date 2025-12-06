import {
  applyProjectionUniforms,
  createShaderProgram,
  makeShaderVariantKey,
  resolveProjectionMode,
  type ShaderProgram,
} from './shader-program'
import type { ShaderData } from './shaders'
import type { CustomShaderConfig, RenderParams } from './renderer-types'
import {
  renderSingleImage,
  type SingleImageState,
} from './single-image-renderer'
import { renderTiles } from './tile-renderer'

export class ZarrRenderer {
  private gl: WebGL2RenderingContext
  private fragmentShaderSource: string
  private shaderCache: Map<string, ShaderProgram> = new Map()
  private singleImageState: SingleImageState = {
    uploaded: false,
    version: null,
  }
  private customShaderConfig: CustomShaderConfig | null = null

  constructor(
    gl: WebGL2RenderingContext,
    fragmentShaderSource: string,
    customShaderConfig?: CustomShaderConfig
  ) {
    this.gl = ZarrRenderer.resolveGl(gl)
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
      mode,
    } = params

    const isMapboxTile = mode.type === 'mapboxTile'
    const tileTexOverrides =
      mode.type === 'mapboxTile' ? mode.tileTexOverrides : undefined
    const tileOverride =
      mode.type === 'mapboxTile' ? mode.tileOverride : undefined

    const shaderProgram = this.getOrCreateProgram(
      shaderData,
      customShaderConfig,
      !!mapboxGlobe || isMapboxTile
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

    applyProjectionUniforms(
      this.gl,
      shaderProgram,
      matrix,
      projectionData,
      mapboxGlobe,
      isMapboxTile
    )

    if (isMultiscale) {
      if (!tileCache) {
        console.warn('Missing tile cache for multiscale render, skipping frame')
        return
      }
      renderTiles(
        this.gl,
        shaderProgram,
        visibleTiles,
        worldOffsets,
        tileCache,
        tileSize,
        vertexArr,
        pixCoordArr,
        tileBounds,
        customShaderConfig,
        isMapboxTile,
        tileTexOverrides
      )
    } else if (singleImage) {
      this.singleImageState = renderSingleImage(
        this.gl,
        shaderProgram,
        worldOffsets,
        singleImage,
        vertexArr,
        this.singleImageState,
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
    this.singleImageState.uploaded = false
  }
}
