import type { SingleImageParams } from './renderer-types'
import type { ShaderProgram } from './shader-program'
import { configureDataTexture } from './webgl-utils'

export interface SingleImageState {
  uploaded: boolean
  version: number | null
}

export function renderSingleImage(
  gl: WebGL2RenderingContext,
  shaderProgram: ShaderProgram,
  worldOffsets: number[],
  params: SingleImageParams,
  vertexArr: Float32Array,
  state: SingleImageState,
  tileOverride?: {
    scaleX: number
    scaleY: number
    shiftX: number
    shiftY: number
    texScale: [number, number]
    texOffset: [number, number]
  }
): SingleImageState {
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

  let uploaded = state.uploaded
  let version = state.version

  if (version === null || version !== geometryVersion) {
    uploaded = false
    version = geometryVersion
  }

  if (!data || !bounds || !texture || !vertexBuffer || !pixCoordBuffer) {
    return { uploaded, version }
  }

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
  if (!uploaded) {
    gl.bufferData(gl.ARRAY_BUFFER, vertexArr, gl.STATIC_DRAW)
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, pixCoordBuffer)
  if (!uploaded) {
    gl.bufferData(gl.ARRAY_BUFFER, pixCoordArr, gl.STATIC_DRAW)
    uploaded = true
  }

  gl.activeTexture(gl.TEXTURE0)
  gl.bindTexture(gl.TEXTURE_2D, texture)
  gl.uniform1i(shaderProgram.texLoc, 0)
  configureDataTexture(gl)
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

  return { uploaded, version }
}
