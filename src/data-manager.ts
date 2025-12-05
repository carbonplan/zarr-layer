import type { MercatorBounds, TileTuple } from './map-utils'
import type { TileRenderCache } from './zarr-tile-cache'
import type { LoadingStateCallback, MapLike } from './types'

export interface RenderData {
  isMultiscale: boolean
  vertexArr?: Float32Array
  pixCoordArr?: Float32Array
  tileBounds?: Record<string, MercatorBounds>
  visibleTiles?: TileTuple[]
  tileCache?: TileRenderCache
  tileSize?: number
  singleImage?: {
    data: Float32Array | null
    width: number
    height: number
    bounds: MercatorBounds | null
    texture: WebGLTexture | null
    vertexBuffer: WebGLBuffer | null
    pixCoordBuffer: WebGLBuffer | null
    pixCoordArr: Float32Array
    geometryVersion: number
  }
}

export interface DataManager {
  isMultiscale: boolean
  initialize(): Promise<void>
  update(map: MapLike, gl: WebGL2RenderingContext): void
  getRenderData(): RenderData
  dispose(gl: WebGL2RenderingContext): void
  setSelector(
    selector: Record<string, number | number[] | string | string[]>
  ): Promise<void>
  onProjectionChange(isGlobe: boolean): void
  setLoadingCallback(callback: LoadingStateCallback | undefined): void
}
