import type { MercatorBounds, TileTuple, XYLimits } from './map-utils'
import type { ProjectionData, ShaderData } from './shaders'
import type { Tiles } from './tiles'
import type {
  CRS,
  LoadingStateCallback,
  MapLike,
  NormalizedSelector,
  Selector,
} from './types'
import type {
  CustomShaderConfig,
  MapboxGlobeParams,
  RendererUniforms,
  SingleImageParams,
} from './renderer-types'
import type { ZarrRenderer } from './zarr-renderer'
import type { QueryGeometry, QueryResult } from './query/types'

export interface RenderContext {
  gl: WebGL2RenderingContext
  matrix: number[] | Float32Array | Float64Array
  uniforms: RendererUniforms
  colormapTexture: WebGLTexture
  worldOffsets: number[]
  customShaderConfig?: CustomShaderConfig
  shaderData?: ShaderData
  projectionData?: ProjectionData
  mapboxGlobe?: MapboxGlobeParams
}

export interface TileId {
  z: number
  x: number
  y: number
}

export interface TiledRenderState {
  tileCache: Tiles
  visibleTiles: TileTuple[]
  tileSize: number
  vertexArr: Float32Array
  pixCoordArr: Float32Array
  tileBounds?: Record<string, MercatorBounds>
}

export interface SingleImageRenderState {
  singleImage: SingleImageParams
  vertexArr: Float32Array
}

export interface ZarrMode {
  isMultiscale: boolean
  initialize(): Promise<void>
  update(map: MapLike, gl: WebGL2RenderingContext): void
  render(renderer: ZarrRenderer, context: RenderContext): void
  renderToTile?(
    renderer: ZarrRenderer,
    tileId: TileId,
    context: RenderContext
  ): boolean
  dispose(gl: WebGL2RenderingContext): void
  setSelector(selector: NormalizedSelector): Promise<void>
  onProjectionChange(isGlobe: boolean): void
  setLoadingCallback(callback: LoadingStateCallback | undefined): void
  getCRS(): CRS
  getXYLimits(): XYLimits | null
  getMaxLevelIndex(): number
  getTiledState?(): TiledRenderState | null
  getSingleImageState?(): SingleImageRenderState | null
  updateClim(clim: [number, number]): void

  // Query methods (optional)
  queryData?(
    geometry: QueryGeometry,
    selector?: Selector
  ): Promise<QueryResult>
}
