import { mustCreateBuffer, mustCreateTexture } from './webgl-utils'

export interface TileRenderData {
  data: Float32Array | null
  bandData: Map<string, Float32Array>
  channels: number
  selectorHash: string | null
  tileTexture: WebGLTexture
  bandTextures: Map<string, WebGLTexture>
  bandTexturesUploaded: Set<string>
  bandTexturesConfigured: Set<string>
  textureUploaded: boolean
  textureConfigured: boolean
  vertexBuffer: WebGLBuffer
  pixCoordBuffer: WebGLBuffer
  geometryUploaded?: boolean
  lastUsed: number
}

export class TileRenderCache {
  private tiles: Map<string, TileRenderData> = new Map()
  private accessOrder: string[] = []

  constructor(private gl: WebGL2RenderingContext, private maxTiles: number) {}

  get(tileKey: string): TileRenderData | undefined {
    return this.tiles.get(tileKey)
  }

  upsert(tileKey: string): TileRenderData {
    const gl = this.gl
    let tile = this.tiles.get(tileKey)

    if (!tile) {
      tile = {
        data: null,
        bandData: new Map(),
        channels: 1,
        selectorHash: null,
        tileTexture: mustCreateTexture(gl),
        bandTextures: new Map(),
        bandTexturesUploaded: new Set(),
        bandTexturesConfigured: new Set(),
        textureUploaded: false,
        textureConfigured: false,
        vertexBuffer: mustCreateBuffer(gl),
        pixCoordBuffer: mustCreateBuffer(gl),
        geometryUploaded: false,
        lastUsed: Date.now(),
      }
      this.tiles.set(tileKey, tile)
      this.accessOrder.push(tileKey)
      this.evictOldTiles()
    } else {
      tile.lastUsed = Date.now()
      const idx = this.accessOrder.indexOf(tileKey)
      if (idx > -1) {
        this.accessOrder.splice(idx, 1)
        this.accessOrder.push(tileKey)
      }
    }

    return tile
  }

  ensureBandTexture(tileKey: string, bandName: string): WebGLTexture | null {
    const tile = this.tiles.get(tileKey)
    if (!tile) return null

    let tex = tile.bandTextures.get(bandName)
    if (!tex) {
      tex = mustCreateTexture(this.gl)
      tile.bandTextures.set(bandName, tex)
    }
    return tex
  }

  clear() {
    for (const tile of this.tiles.values()) {
      this.gl.deleteTexture(tile.tileTexture)
      for (const tex of tile.bandTextures.values()) {
        this.gl.deleteTexture(tex)
      }
      this.gl.deleteBuffer(tile.vertexBuffer)
      this.gl.deleteBuffer(tile.pixCoordBuffer)
    }
    this.tiles.clear()
    this.accessOrder = []
  }

  private evictOldTiles() {
    while (this.tiles.size > this.maxTiles) {
      const oldestKey = this.accessOrder.shift()
      if (!oldestKey) break
      const tile = this.tiles.get(oldestKey)
      if (tile) {
        this.gl.deleteTexture(tile.tileTexture)
        for (const tex of tile.bandTextures.values()) {
          this.gl.deleteTexture(tex)
        }
        this.gl.deleteBuffer(tile.vertexBuffer)
        this.gl.deleteBuffer(tile.pixCoordBuffer)
      }
      this.tiles.delete(oldestKey)
    }
  }

  markGeometryDirty() {
    for (const tile of this.tiles.values()) {
      tile.geometryUploaded = false
    }
  }
}
