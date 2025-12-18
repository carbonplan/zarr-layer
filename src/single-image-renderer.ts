import type { SingleImageParams } from './renderer-types'
import type { ShaderProgram } from './shader-program'
import {
  configureDataTexture,
  getTextureFormats,
  normalizeDataForTexture,
} from './webgl-utils'

export interface SingleImageState {
  uploaded: boolean
  geometryVersion: number | null
  dataVersion: number | null
  normalizedData: Float32Array | null
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
    fillValue = null,
    clim,
  } = params

  // For region-based rendering (data=null, texture already uploaded), bypass state tracking
  // Regions have pre-uploaded buffers and textures, just render directly
  const isRegionRender = data === null

  let uploaded = state.uploaded
  let currentGeometryVersion = state.geometryVersion
  let currentDataVersion = state.dataVersion
  let normalizedData = state.normalizedData

  // For region renders, don't use state tracking - each region is independent
  // For regular single-image renders, track state to avoid redundant uploads
  let dataChanged = false
  if (!isRegionRender) {
    const geometryChanged =
      currentGeometryVersion === null ||
      currentGeometryVersion !== geometryVersion
    dataChanged =
      currentDataVersion === null || currentDataVersion !== dataVersion

    if (geometryChanged) {
      uploaded = false
      currentGeometryVersion = geometryVersion
    }

    // Normalize data when it changes (use clim to determine scale)
    if (dataChanged || !normalizedData) {
      normalizedData = normalizeDataForTexture(data, fillValue, clim).normalized
    }
  }

  // Early return if required resources are missing
  if (!bounds || !texture || !vertexBuffer || !pixCoordBuffer) {
    return {
      uploaded,
      geometryVersion: currentGeometryVersion,
      dataVersion: currentDataVersion,
      normalizedData,
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

  // For region-based rendering, buffers are pre-uploaded, just bind them
  // For regular single-image rendering, upload if geometry changed
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer)
  if (!uploaded && !isRegionRender) {
    gl.bufferData(gl.ARRAY_BUFFER, vertexArr, gl.STATIC_DRAW)
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, pixCoordBuffer)
  if (!uploaded && !isRegionRender) {
    gl.bufferData(gl.ARRAY_BUFFER, pixCoordArr, gl.STATIC_DRAW)
    uploaded = true
  }

  gl.activeTexture(gl.TEXTURE0)
  gl.bindTexture(gl.TEXTURE_2D, texture)
  gl.uniform1i(shaderProgram.texLoc, 0)
  configureDataTexture(gl)

  // Only upload texture data if we have normalized data to upload
  // Skip if data was null (texture already uploaded externally, e.g., region-based rendering)
  if (dataChanged && normalizedData) {
    const { format, internalFormat } = getTextureFormats(gl, channels)

    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      internalFormat,
      width,
      height,
      0,
      format,
      gl.FLOAT,
      normalizedData
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
    normalizedData,
  }
}
