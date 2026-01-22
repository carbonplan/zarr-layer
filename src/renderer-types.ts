export interface RendererUniforms {
  clim: [number, number]
  opacity: number
  fillValue: number | null
  scaleFactor: number
  offset: number
  fixedDataScale: number
}

export interface CustomShaderConfig {
  bands: string[]
  customFrag?: string
  customUniforms?: Record<string, number>
}

export interface MapboxParams {
  projection: { name: string }
  globeToMercatorMatrix: number[] | Float32Array | Float64Array
  transition: number
}

export type ProjectionMode =
  | 'wgs84-globe'
  | 'maplibre-globe'
  | 'mapbox'
  | 'mapbox-wgs84'
