export interface RendererUniforms {
  clim: [number, number]
  opacity: number
  fillValue: number | null
  scaleFactor: number
  offset: number
  fixedDataScale: number
  /**
   * Layer-level mercator anchor used by the eye-coords (render-relative-to-eye)
   * path for proj4 datasets. When set, applyCommonUniforms uploads
   * `u_eye_matrix` (= matrix) and `u_anchor_clip` (= matrix · vec4(anchor, 0, 1),
   * computed in JS Float64) so the shader can avoid forming an absolute
   * world-coord Float32. See VERTEX_TO_WGS84_TO_MERCATOR.
   *
   * `x` and `y` are in normalized mercator [0, 1] world coords.
   */
  eyeAnchorMerc?: { x: number; y: number } | null
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
 * 'maplibre-proj4' — WGS84-input path for MapLibre (proj4 vertices → Mercator via projectTile)
 * 'maplibre-ecef'  — ECEF path for MapLibre globe (proj4 or EPSG:4326 vertices → sphere).
 *                    Needed for untiled globe rendering that must reach the poles,
 *                    since the regular MapLibre paths still start from Mercator-style geometry.
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
