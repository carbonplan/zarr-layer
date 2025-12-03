import {
  createProgram,
  createShader,
  mustCreateBuffer,
  mustGetUniformLocation,
} from './webgl-utils'
import {
  createVertexShaderSource,
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
  vminLoc: WebGLUniformLocation
  vmaxLoc: WebGLUniformLocation
  opacityLoc: WebGLUniformLocation
  noDataLoc: WebGLUniformLocation
  noDataMinLoc: WebGLUniformLocation
  noDataMaxLoc: WebGLUniformLocation
  useFillValueLoc: WebGLUniformLocation
  fillValueLoc: WebGLUniformLocation
  scaleFactorLoc: WebGLUniformLocation
  addOffsetLoc: WebGLUniformLocation
  cmapLoc: WebGLUniformLocation
  texLoc: WebGLUniformLocation
  texScaleLoc: WebGLUniformLocation
  texOffsetLoc: WebGLUniformLocation
  vertexLoc: number
  pixCoordLoc: number
  isGlobe: boolean
}

export class ZarrRenderer {
  private gl: WebGL2RenderingContext
  private fragmentShaderSource: string
  private shaderCache: Map<string, ShaderProgram> = new Map()
  private singleImageGeometryUploaded = false

  constructor(gl: WebGL2RenderingContext, fragmentShaderSource: string) {
    this.gl = ZarrRenderer.resolveGl(gl)
    this.fragmentShaderSource = fragmentShaderSource
    this.getOrCreateProgram(undefined)
  }

  private static resolveGl(gl: WebGL2RenderingContext): WebGL2RenderingContext {
    const hasWebGL2Methods =
      gl &&
      typeof gl.getUniformLocation === 'function' &&
      typeof gl.drawBuffers === 'function'
    if (hasWebGL2Methods) {
      return gl
    }
    throw new Error('Invalid WebGL2 context: missing required WebGL2 methods')
  }

  private getOrCreateProgram(shaderData?: ShaderData): ShaderProgram {
    const variantName = shaderData?.variantName ?? 'mercator'

    const cached = this.shaderCache.get(variantName)
    if (cached) {
      return cached
    }

    const isGlobe = shaderData && shaderData.vertexShaderPrelude ? true : false
    const vertexSource = createVertexShaderSource(shaderData)

    const vertexShader = createShader(
      this.gl,
      this.gl.VERTEX_SHADER,
      vertexSource
    )
    const fragmentShader = createShader(
      this.gl,
      this.gl.FRAGMENT_SHADER,
      this.fragmentShaderSource
    )
    if (!vertexShader || !fragmentShader) {
      throw new Error(`Failed to create shaders for variant: ${variantName}`)
    }

    const program = createProgram(this.gl, vertexShader, fragmentShader)
    if (!program) {
      throw new Error(`Failed to create program for variant: ${variantName}`)
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
      vminLoc: mustGetUniformLocation(this.gl, program, 'vmin'),
      vmaxLoc: mustGetUniformLocation(this.gl, program, 'vmax'),
      opacityLoc: mustGetUniformLocation(this.gl, program, 'opacity'),
      noDataLoc: mustGetUniformLocation(this.gl, program, 'nodata'),
      noDataMinLoc: mustGetUniformLocation(this.gl, program, 'u_noDataMin'),
      noDataMaxLoc: mustGetUniformLocation(this.gl, program, 'u_noDataMax'),
      useFillValueLoc: mustGetUniformLocation(
        this.gl,
        program,
        'u_useFillValue'
      ),
      fillValueLoc: mustGetUniformLocation(this.gl, program, 'u_fillValue'),
      scaleFactorLoc: mustGetUniformLocation(this.gl, program, 'u_scaleFactor'),
      addOffsetLoc: mustGetUniformLocation(this.gl, program, 'u_addOffset'),
      cmapLoc: mustGetUniformLocation(this.gl, program, 'cmap'),
      texLoc: mustGetUniformLocation(this.gl, program, 'tex'),
      texScaleLoc: mustGetUniformLocation(this.gl, program, 'u_texScale'),
      texOffsetLoc: mustGetUniformLocation(this.gl, program, 'u_texOffset'),
      vertexLoc: this.gl.getAttribLocation(program, 'vertex'),
      pixCoordLoc: this.gl.getAttribLocation(program, 'pix_coord_in'),
      isGlobe,
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
    } = params

    const shaderProgram = this.getOrCreateProgram(shaderData)

    const gl = this.gl
    gl.useProgram(shaderProgram.program)

    gl.enable(gl.BLEND)
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)

    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, colormapTexture)
    gl.uniform1i(shaderProgram.cmapLoc, 1)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)

    gl.uniform1f(shaderProgram.vminLoc, uniforms.vmin)
    gl.uniform1f(shaderProgram.vmaxLoc, uniforms.vmax)
    gl.uniform1f(shaderProgram.opacityLoc, uniforms.opacity)
    gl.uniform1f(shaderProgram.noDataLoc, uniforms.fillValue)
    gl.uniform1f(shaderProgram.noDataMinLoc, uniforms.noDataMin)
    gl.uniform1f(shaderProgram.noDataMaxLoc, uniforms.noDataMax)
    gl.uniform1i(shaderProgram.useFillValueLoc, uniforms.useFillValue ? 1 : 0)
    gl.uniform1f(shaderProgram.fillValueLoc, uniforms.fillValue)
    gl.uniform1f(shaderProgram.scaleFactorLoc, uniforms.scaleFactor)
    gl.uniform1f(shaderProgram.addOffsetLoc, uniforms.offset)
    gl.uniform2f(shaderProgram.texScaleLoc, 1.0, 1.0)
    gl.uniform2f(shaderProgram.texOffsetLoc, 0.0, 0.0)

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
        pixCoordArr
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
    pixCoordArr: Float32Array
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
        let texScale: [number, number] = [1, 1]
        let texOffset: [number, number] = [0, 0]

        if (tile && tile.data) {
          tileToRender = tile
        } else {
          const parent = this.findBestParentTile(z, x, y, tileCache)
          if (parent) {
            tileToRender = parent.tile
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

        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D, tileToRender.tileTexture)
        gl.uniform1i(shaderProgram.texLoc, 0)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.R16F,
          tileSize,
          tileSize,
          0,
          gl.RED,
          gl.FLOAT,
          tileToRender.data
        )

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
