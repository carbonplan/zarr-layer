import type { BandFormat, BandTransform } from './band-format'
import type { MercatorBounds, MeshMercatorBounds } from './map-utils'
import type { ProjectionData, ShaderData } from './shaders'

export interface RendererUniforms {
  clim: [number, number]
  opacity: number
  fillValue: number | null
  scaleFactor: number
  offset: number
  fixedDataScale: number
}

/** A custom-shader uniform: a float, or a float array declared at its length. */
export type UniformValue = number | number[] | Float32Array

export interface CustomShaderConfig {
  bands: string[]
  customFrag?: string
  customUniforms?: Record<string, UniformValue>
  /** Storage format of the band texture; decides the sampler type. */
  bandFormat?: BandFormat
}

export interface MapboxParams {
  projection: { name: string }
  globeToMercatorMatrix: number[] | Float32Array | Float64Array
  transition: number
  /** True when this frame uses the direct globe path instead of draped tiles. */
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
 *                    Needed for globe rendering that must reach the poles,
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
  /** Null when a custom shader samples band textures instead. */
  texture: WebGLTexture | null
  vertexBuffer: WebGLBuffer
  /** Texture coordinate buffer for sampling resampled data */
  pixCoordBuffer: WebGLBuffer
  mercatorBounds: MercatorBounds
  width: number
  height: number
  /** Data orientation: true = row 0 is south */
  latIsAscending: boolean
  /** Band texture array for multi-band custom shaders */
  bandTexture: WebGLTexture | null
  bandTextureKey: string | null
  bandTransform: BandTransform | null
  /** Index buffer for the adaptive mesh */
  indexBuffer: WebGLBuffer
  /** Number of indices to draw */
  indexCount: number
  /** Normalized-Mercator bounds for reconstructing region-local mesh positions */
  meshBounds: MeshMercatorBounds
}
