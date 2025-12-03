export interface RenderData {
  isMultiscale: boolean
  vertexArr?: Float32Array
  pixCoordArr?: Float32Array
  // Tiled specific
  visibleTiles?: any[]
  tileCache?: any
  tileSize?: number
  // Single image specific
  singleImage?: {
    data: Float32Array | null
    width: number
    height: number
    bounds: any
    texture: WebGLTexture | null
    vertexBuffer: WebGLBuffer | null
    pixCoordBuffer: WebGLBuffer | null
    pixCoordArr: Float32Array
  }
}

export interface DataManager {
  isMultiscale: boolean
  initialize(): Promise<void>
  update(map: any, gl: WebGL2RenderingContext): void
  getRenderData(): RenderData
  dispose(gl: WebGL2RenderingContext): void
  setSelector(selector: Record<string, number | number[]>): Promise<void>
  onProjectionChange(isGlobe: boolean): void
}
