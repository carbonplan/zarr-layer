import type { SingleImageParams } from './renderer-types'
import type { ShaderProgram } from './shader-program'
import { configureDataTexture } from './webgl-utils'

export interface SingleImageState {
  uploaded: boolean
  geometryVersion: number | null
  dataVersion: number | null
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
    channels = 1,
    pixCoordArr,
    geometryVersion,
    dataVersion,
    texScale: baseTexScale = [1, 1],
    texOffset: baseTexOffset = [0, 0],
  } = params

  let uploaded = state.uploaded
  let currentGeometryVersion = state.geometryVersion
  let currentDataVersion = state.dataVersion

  const geometryChanged =
    currentGeometryVersion === null ||
    currentGeometryVersion !== geometryVersion
  const dataChanged =
    currentDataVersion === null || currentDataVersion !== dataVersion

  if (geometryChanged) {
    uploaded = false
    currentGeometryVersion = geometryVersion
  }

  if (!data || !bounds || !texture || !vertexBuffer || !pixCoordBuffer) {
    return {
      uploaded,
      geometryVersion: currentGeometryVersion,
      dataVersion: currentDataVersion,
    }
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
  const overrideScale = tileOverride?.texScale ?? [1.0, 1.0]
  const overrideOffset = tileOverride?.texOffset ?? [0.0, 0.0]
  const texScale: [number, number] = [
    baseTexScale[0] * overrideScale[0],
    baseTexScale[1] * overrideScale[1],
  ]
  const texOffset: [number, number] = [
    baseTexOffset[0] * overrideScale[0] + overrideOffset[0],
    baseTexOffset[1] * overrideScale[1] + overrideOffset[1],
  ]
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

  if (dataChanged) {
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
        ? gl.RG32F
        : channels === 3
        ? gl.RGB32F
        : channels >= 4
        ? gl.RGBA32F
        : gl.R32F

    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      internalFormat,
      width,
      height,
      0,
      format,
      gl.FLOAT,
      data
    )
    currentDataVersion = dataVersion
  }

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

  return {
    uploaded,
    geometryVersion: currentGeometryVersion,
    dataVersion: currentDataVersion,
  }
}
