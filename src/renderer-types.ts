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
  /** True when this frame is using the direct untiled globe path, not draped tiles. */
  directGlobePathActive?: boolean
  /** Internal Mapbox globe matrix needed for direct custom-layer ECEF depth parity. */
  expandedFarZMercatorMatrix?: number[] | Float32Array | Float64Array
}

/**
 * Projection modes: {backend}-{input path}
 *
 * 'maplibre'       — Mercator-input path for MapLibre (EPSG:3857, EPSG:4326 via projectTile)
 * 'maplibre-proj4' — WGS84-input path for MapLibre (proj4 vertices → Mercator in shader)
 * 'maplibre-ecef'  — ECEF path for MapLibre globe (proj4 or EPSG:4326 vertices → sphere)
 * 'mapbox'         — Mercator-input path for Mapbox
 * 'mapbox-proj4'   — WGS84-input path for Mapbox (proj4 vertices → Mercator in shader)
 * 'mapbox-ecef'    — ECEF path for Mapbox globe (proj4 or EPSG:4326 vertices → sphere)
 */
export type ProjectionMode =
  | 'maplibre'
  | 'maplibre-proj4'
  | 'maplibre-ecef'
  | 'mapbox'
  | 'mapbox-proj4'
  | 'mapbox-ecef'
