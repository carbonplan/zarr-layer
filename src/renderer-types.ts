import type { MercatorBounds, Wgs84Bounds } from './map-utils'
import type { ProjectionData, ShaderData } from './shaders'

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

/** Identifies a Mapbox tile for the draped renderToTile path. */
export interface TileId {
  z: number
  x: number
  y: number
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
  /** Data orientation: true = row 0 is south */
  latIsAscending: boolean
  /** Band textures for multi-band custom shaders */
  bandData?: Map<string, Float32Array>
  bandTextures?: Map<string, WebGLTexture>
  bandTexturesUploaded?: Set<string>
  bandTexturesConfigured?: Set<string>
  /** Index buffer for adaptive source-projected meshes */
  indexBuffer?: WebGLBuffer
  /** Number of vertices/indices to draw */
  vertexCount?: number
  /** Whether to use indexed mesh rendering (gl.drawElements) */
  useIndexedMesh?: boolean
  /** WGS84 bounds for two-stage reprojection */
  wgs84Bounds?: Wgs84Bounds
}
