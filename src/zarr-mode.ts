import type { Bounds } from './types'
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
  MapboxParams,
  RendererUniforms,
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
  mapbox?: MapboxParams
  isGlobe?: boolean
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
  latIsAscending: boolean
}

export interface RegionRenderState {
  texture: WebGLTexture
  vertexBuffer: WebGLBuffer
  /** Texture coordinate buffer for sampling resampled data */
  pixCoordBuffer: WebGLBuffer
  vertexArr: Float32Array
  mercatorBounds: MercatorBounds
  width: number
  height: number
  channels: number
  /** Data orientation: true = row 0 is south */
  latIsAscending: boolean
  /** Band textures for multi-band custom shaders */
  bandData?: Map<string, Float32Array>
  bandTextures?: Map<string, WebGLTexture>
  bandTexturesUploaded?: Set<string>
  bandTexturesConfigured?: Set<string>
  /** Index buffer for adaptive mesh (proj4 datasets) */
  indexBuffer?: WebGLBuffer
  /** Number of vertices/indices to draw */
  vertexCount?: number
  /** Whether to use indexed mesh rendering (gl.drawElements) */
  useIndexedMesh?: boolean
  /** WGS84 bounds for two-stage reprojection */
  wgs84Bounds?: import('./map-utils').Wgs84Bounds
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
  dispose(gl: WebGL2RenderingContext | WebGLRenderingContext): void
  setSelector(selector: NormalizedSelector): Promise<void>
  onProjectionChange(isGlobe: boolean): void
  setLoadingCallback(callback: LoadingStateCallback | undefined): void
  getCRS(): CRS
  getXYLimits(): XYLimits | null
  /** Get bounds in EPSG:4326 [west, south, east, north] for map.fitBounds() */
  getLonLatBounds(): Bounds | null
  getMaxLevelIndex(): number
  getLevels(): string[]
  getTiledState?(): TiledRenderState | null

  // Query methods (optional)
  queryData?(geometry: QueryGeometry, selector?: Selector): Promise<QueryResult>
}
