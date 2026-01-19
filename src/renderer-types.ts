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

export interface MapboxGlobeParams {
  projection: { name: string }
  globeToMercatorMatrix: number[] | Float32Array | Float64Array
  transition: number
}

export type ProjectionMode =
  | 'mercator'
  | 'wgs84'
  | 'wgs84-globe'
  | 'maplibre-globe'
  | 'mapbox-globe'
  | 'mapbox-globe-wgs84'
