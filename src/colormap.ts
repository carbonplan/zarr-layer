import { mustCreateTexture } from './webgl-utils'
import type { ColormapArray } from './types'

function hexToRgb(hex: string): [number, number, number] {
  const cleaned = hex.replace('#', '')
  if (cleaned.length !== 6) {
    throw new Error(`Invalid hex color: ${hex}`)
  }
  const num = parseInt(cleaned, 16)
  return [(num >> 16) & 255, (num >> 8) & 255, num & 255]
}

export class ColormapState {
  colors: number[][]
  floatData: Float32Array
  length: number
  texture: WebGLTexture | null = null
  private dirty: boolean = true

  constructor(colormap: ColormapArray) {
    const { colors, floatData, length } = this.build(colormap)
    this.colors = colors
    this.floatData = floatData
    this.length = length
    this.dirty = true
  }

  apply(colormap: ColormapArray) {
    const { colors, floatData, length } = this.build(colormap)
    this.colors = colors
    this.floatData = floatData
    this.length = length
    this.dirty = true
  }

  ensureTexture(gl: WebGL2RenderingContext): WebGLTexture {
    if (!this.texture || this.dirty) {
      this.upload(gl)
    }
    return this.texture!
  }

  upload(gl: WebGL2RenderingContext) {
    if (!this.texture) {
      this.texture = mustCreateTexture(gl)
    }
    gl.bindTexture(gl.TEXTURE_2D, this.texture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGB16F,
      this.length,
      1,
      0,
      gl.RGB,
      gl.FLOAT,
      this.floatData
    )
    gl.bindTexture(gl.TEXTURE_2D, null)
    this.dirty = false
  }

  dispose(gl: WebGL2RenderingContext) {
    if (this.texture) {
      gl.deleteTexture(this.texture)
      this.texture = null
    }
  }

  private build(colormap: ColormapArray) {
    if (!Array.isArray(colormap) || colormap.length === 0) {
      throw new Error(
        'colormap must be a non-empty array of [r, g, b] values or hex strings'
      )
    }

    const normalized: number[][] = []

    for (const entry of colormap) {
      if (typeof entry === 'string') {
        normalized.push(hexToRgb(entry))
      } else if (Array.isArray(entry) && entry.length >= 3) {
        normalized.push([entry[0], entry[1], entry[2]])
      } else {
        throw new Error(
          'colormap entries must be arrays shaped like [r, g, b] or hex strings'
        )
      }
    }

    const flattened = normalized.flat()
    const needsScaling = flattened.some((value) => value > 1)
    const floatData = new Float32Array(
      flattened.map((value) => (needsScaling ? value / 255.0 : value))
    )

    return { colors: normalized, floatData, length: normalized.length }
  }
}
