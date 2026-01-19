/**
 * @module renderable-region
 *
 * Unified rendering abstraction for both tiles and regions.
 * Both TiledMode and UntiledMode convert their data to this interface
 * for a single render path.
 */

import type { MercatorBounds, Wgs84Bounds } from './map-utils'
import type { CustomShaderConfig } from './renderer-types'
import type { ShaderProgram } from './shader-program'
import { bindBandTextures, bindGeometryBuffers } from './render-helpers'

/**
 * A region ready for rendering.
 * This is the common interface for both tiles and untiled regions.
 * Textures are expected to be uploaded before rendering.
 */
export interface RenderableRegion {
  mercatorBounds: MercatorBounds

  // Geometry
  vertexBuffer: WebGLBuffer
  pixCoordBuffer: WebGLBuffer
  vertexCount: number

  // Indexed mesh support (for adaptive mesh with proj4 datasets)
  indexBuffer?: WebGLBuffer | null
  useIndexedMesh?: boolean

  // WGS84 bounds for two-stage reprojection (proj4 datasets)
  wgs84Bounds?: Wgs84Bounds | null

  // Data orientation: true = row 0 is south (latitude ascending)
  // Resolved by ZarrStore during init
  latIsAscending: boolean

  // Main texture (pre-uploaded)
  texture: WebGLTexture

  // Band textures (for custom shaders)
  bandData: Map<string, Float32Array>
  bandTextures: Map<string, WebGLTexture>
  bandTexturesUploaded: Set<string>
  bandTexturesConfigured: Set<string>
  width: number
  height: number

  // Texture transform (for parent tile fallback, defaults to identity)
  texScale?: [number, number]
  texOffset?: [number, number]

  // Callbacks for lazy resource creation
  ensureBandTexture?: (bandName: string) => WebGLTexture | null
}

/**
 * Render a single region with the given shader program.
 * Handles uniform setup, texture binding, and draw call.
 *
 * @param gl - WebGL2 context
 * @param shaderProgram - Compiled shader program
 * @param region - Region to render
 * @param worldOffsets - X offsets for map wrapping
 * @param customShaderConfig - Optional custom shader configuration
 * @returns true if rendered successfully, false if skipped (missing data)
 */
export function renderRegion(
  gl: WebGL2RenderingContext,
  shaderProgram: ShaderProgram,
  region: RenderableRegion,
  worldOffsets: number[],
  customShaderConfig?: CustomShaderConfig
): boolean {
  // Determine bounds for scale/shift uniforms
  // For wgs84 shader (two-stage reprojection), use wgs84Bounds
  // Otherwise use mercatorBounds
  const wgs84Bounds = region.wgs84Bounds ?? null

  // Set position uniforms
  // For wgs84: bounds are in normalized 4326 [0,1], shader transforms to Mercator
  // For mercator: bounds are in normalized Mercator [0,1]
  let scaleX: number, scaleY: number, shiftX: number, shiftY: number
  if (wgs84Bounds) {
    scaleX = (wgs84Bounds.lon1 - wgs84Bounds.lon0) / 2
    scaleY = (wgs84Bounds.lat1 - wgs84Bounds.lat0) / 2
    shiftX = (wgs84Bounds.lon0 + wgs84Bounds.lon1) / 2
    shiftY = (wgs84Bounds.lat0 + wgs84Bounds.lat1) / 2
  } else {
    scaleX = (region.mercatorBounds.x1 - region.mercatorBounds.x0) / 2
    scaleY = (region.mercatorBounds.y1 - region.mercatorBounds.y0) / 2
    shiftX = (region.mercatorBounds.x0 + region.mercatorBounds.x1) / 2
    shiftY = (region.mercatorBounds.y0 + region.mercatorBounds.y1) / 2
  }

  gl.uniform1f(shaderProgram.scaleLoc, 0)
  gl.uniform1f(shaderProgram.scaleXLoc, scaleX)
  gl.uniform1f(shaderProgram.scaleYLoc, scaleY)
  gl.uniform1f(shaderProgram.shiftXLoc, shiftX)
  gl.uniform1f(shaderProgram.shiftYLoc, shiftY)

  // Set texture transform (for parent tile fallback)
  const texScale = region.texScale ?? [1, 1]
  const texOffset = region.texOffset ?? [0, 0]
  gl.uniform2f(shaderProgram.texScaleLoc, texScale[0], texScale[1])
  gl.uniform2f(shaderProgram.texOffsetLoc, texOffset[0], texOffset[1])

  // Set EPSG:4326 reprojection uniforms for fragment shader reprojection
  // This is used when mercatorBounds has lat bounds but we're NOT using wgs84 vertex shader
  const hasLatBounds =
    region.mercatorBounds.latMin !== undefined &&
    region.mercatorBounds.latMax !== undefined
  const useFragmentReproject = hasLatBounds && !wgs84Bounds

  if (shaderProgram.reprojectLoc !== null) {
    gl.uniform1i(shaderProgram.reprojectLoc, useFragmentReproject ? 1 : 0)
  }
  if (
    useFragmentReproject &&
    shaderProgram.latBoundsLoc !== null &&
    hasLatBounds
  ) {
    gl.uniform2f(
      shaderProgram.latBoundsLoc,
      region.mercatorBounds.latMin!,
      region.mercatorBounds.latMax!
    )
  }
  if (useFragmentReproject && shaderProgram.latIsAscendingLoc !== null) {
    gl.uniform1i(shaderProgram.latIsAscendingLoc, region.latIsAscending ? 1 : 0)
  }

  // Bind geometry
  bindGeometryBuffers(
    gl,
    shaderProgram,
    region.vertexBuffer,
    region.pixCoordBuffer
  )

  // Bind index buffer for indexed mesh
  if (region.useIndexedMesh && region.indexBuffer) {
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, region.indexBuffer)
  }

  // Bind textures (upload happens at fetch time for both tiles and regions)
  if (shaderProgram.useCustomShader && customShaderConfig) {
    const bandsBound = bindBandTextures(gl, {
      bandData: region.bandData,
      bandTextures: region.bandTextures,
      bandTexturesUploaded: region.bandTexturesUploaded,
      bandTexturesConfigured: region.bandTexturesConfigured,
      customShaderConfig,
      width: region.width,
      height: region.height,
      ensureTexture: region.ensureBandTexture,
    })
    if (!bandsBound) {
      return false
    }
  } else {
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, region.texture)
    if (shaderProgram.texLoc !== null) {
      gl.uniform1i(shaderProgram.texLoc, 0)
    }
  }

  // Draw for each world offset (map wrapping)
  for (const worldOffset of worldOffsets) {
    gl.uniform1f(shaderProgram.worldXOffsetLoc, worldOffset)
    if (region.useIndexedMesh && region.indexBuffer) {
      gl.drawElements(gl.TRIANGLES, region.vertexCount, gl.UNSIGNED_INT, 0)
    } else {
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, region.vertexCount)
    }
  }

  return true
}
