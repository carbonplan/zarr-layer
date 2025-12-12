import {
  findBestParentTile,
  tileToKey,
  tileToScale,
  type MercatorBounds,
  type TileTuple,
} from './map-utils'
import type { CustomShaderConfig } from './renderer-types'
import type { ShaderProgram } from './shader-program'
import type { Tiles, TileData } from './tiles'
import { configureDataTexture, getTextureFormats } from './webgl-utils'

export function renderTiles(
  gl: WebGL2RenderingContext,
  shaderProgram: ShaderProgram,
  visibleTiles: TileTuple[],
  worldOffsets: number[],
  tileCache: Tiles,
  tileSize: number,
  vertexArr: Float32Array,
  pixCoordArr: Float32Array,
  tileBounds?: Record<string, MercatorBounds>,
  customShaderConfig?: CustomShaderConfig,
  isGlobeTileRender: boolean = false,
  tileTexOverrides?: Record<
    string,
    { texScale: [number, number]; texOffset: [number, number] }
  >
) {
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
      isGlobeTileRender ? 0 : worldOffset
    )

    for (const tileTuple of visibleTiles) {
      const [z, x, y] = tileTuple
      const tileKey = tileToKey(tileTuple)
      const tile = tileCache.get(tileKey)
      const bounds = tileBounds?.[tileKey]

      let tileToRender: TileData | null = null
      let renderTileKey = tileKey
      let texScale: [number, number] = [1, 1]
      let texOffset: [number, number] = [0, 0]

      if (tile && tile.data) {
        tileToRender = tile
      } else {
        const parent = findBestParentTile(tileCache, z, x, y)
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

      // Skip tiles without data or WebGL resources
      if (
        !tileToRender ||
        !tileToRender.data ||
        !tileToRender.vertexBuffer ||
        !tileToRender.pixCoordBuffer ||
        !tileToRender.tileTexture
      ) {
        continue
      }

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
      if (isGlobeTileRender && tileTexOverrides?.[tileKey]) {
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
            const newTex = tileCache.ensureBandTexture(renderTileKey, bandName)
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
            configureDataTexture(gl)
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
          configureDataTexture(gl)
          tileToRender.textureConfigured = true
        }
        const channels = tileToRender.channels ?? 1
        const { format, internalFormat } = getTextureFormats(gl, channels)

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
      gl.vertexAttribPointer(shaderProgram.vertexLoc, 2, gl.FLOAT, false, 0, 0)

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
