import type { MercatorBounds, TileTuple } from './map-utils'
import type { ProjectionData, ShaderData } from './shaders'
import type { TileRenderCache } from './zarr-tile-cache'

export interface RendererUniforms {
  clim: [number, number]
  opacity: number
  fillValue: number | null
  scaleFactor: number
  offset: number
}

export interface CustomShaderConfig {
  bands: string[]
  customFrag?: string
  customUniforms?: Record<string, number>
}

export interface SingleImageParams {
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

export interface MapboxGlobeParams {
  projection: { name: string }
  globeToMercatorMatrix: number[] | Float32Array | Float64Array
  transition: number
}

export type ProjectionMode = 'mercator' | 'maplibre-globe' | 'mapbox-globe'

interface TileOverride {
  scaleX: number
  scaleY: number
  shiftX: number
  shiftY: number
  texScale: [number, number]
  texOffset: [number, number]
}

interface TileTexOverrides {
  [key: string]: { texScale: [number, number]; texOffset: [number, number] }
}

export type RenderMode =
  | { type: 'standard' }
  | {
      type: 'mapboxTile'
      tileOverride?: TileOverride
      tileTexOverrides?: TileTexOverrides
    }

export interface RenderParams {
  matrix: number[] | Float32Array | Float64Array
  colormapTexture: WebGLTexture
  uniforms: RendererUniforms
  worldOffsets: number[]
  isMultiscale: boolean
  visibleTiles: TileTuple[]
  tileSize: number
  vertexArr: Float32Array
  pixCoordArr: Float32Array
  tileBounds?: Record<string, MercatorBounds>
  tileCache?: TileRenderCache
  singleImage?: SingleImageParams
  shaderData?: ShaderData
  projectionData?: ProjectionData
  customShaderConfig?: CustomShaderConfig
  mapboxGlobe?: MapboxGlobeParams
  mode: RenderMode
}
