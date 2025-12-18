import type { MercatorBounds } from './map-utils'

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
  channels?: number
  bounds: MercatorBounds | null
  texture: WebGLTexture | null
  vertexBuffer: WebGLBuffer | null
  pixCoordBuffer: WebGLBuffer | null
  pixCoordArr?: Float32Array
  geometryVersion: number
  dataVersion: number
  texScale?: [number, number]
  texOffset?: [number, number]
  fillValue?: number | null
  clim: [number, number]
}

export interface MapboxGlobeParams {
  projection: { name: string }
  globeToMercatorMatrix: number[] | Float32Array | Float64Array
  transition: number
}

export type ProjectionMode = 'mercator' | 'maplibre-globe' | 'mapbox-globe'
