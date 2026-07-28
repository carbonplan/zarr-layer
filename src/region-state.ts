import type * as zarr from 'zarrita'
import type { MercatorBounds, Wgs84Bounds } from './map-utils'

/** State for a single region (chunk/shard) in region-based loading */
export interface RegionState {
  key: string
  levelIndex: number
  regionX: number
  regionY: number
  // Data
  data: Float32Array | null
  width: number
  height: number
  loading: boolean
  requestId: number | null
  channels: number
  // WebGL resources
  texture: WebGLTexture | null
  textureUploaded: boolean
  vertexBuffer: WebGLBuffer | null
  pixCoordBuffer: WebGLBuffer | null
  indexBuffer: WebGLBuffer | null // For adaptive mesh indexed triangles
  // False whenever the geometry arrays below are newer than the buffers above
  geometryUploaded: boolean
  // Geometry arrays for this region's quad
  vertexArr: Float32Array | null
  pixCoordArr: Float32Array | null // Texture coordinates for sampling resampled data
  indexArr: Uint32Array | null // Triangle indices for adaptive mesh
  vertexCount: number // Number of vertices (for triangle strip) or indices (for indexed)
  useIndexedMesh: boolean // Whether to use indexed triangles (adaptive mesh)
  // Mercator bounds for this region (for shader uniforms)
  mercatorBounds: MercatorBounds | null
  // WGS84 bounds for vertex shader positioning (source-projected path, ECEF globe)
  wgs84Bounds: Wgs84Bounds | null
  // Data orientation: true = row 0 is south
  latIsAscending: boolean
  // Version tracking for selector changes
  selectorVersion: number
  // Multi-band support
  bandData: Map<string, Float32Array>
  bandTextures: Map<string, WebGLTexture>
  bandTexturesUploaded: Set<string>
  bandTexturesConfigured: Set<string>
  // Level-specific dimensions for region geometry bounds.
  // Set from LevelSnapshot during fetch to avoid races with level switching.
  levelMeta: LevelMeta | null
}

/** Level-specific dimensions for geometry bounds calculation */
export type LevelMeta = {
  width: number
  height: number
  regionSize: [number, number]
  // xyLimits omitted - assumed constant across levels for now.
  // TODO: If heterogeneous pyramids with per-level bounds are needed,
  // add xyLimits here and store per-region.
}

/** Snapshot of level state captured at fetch start to prevent race conditions */
export interface LevelSnapshot {
  index: number
  zarrArray: zarr.Array<zarr.DataType>
  baseSliceArgs: (number | zarr.Slice)[]
  baseMultiValueDims: Array<{
    dimIndex: number
    dimName: string
    values: number[]
    labels: (number | string)[]
  }>
  width: number
  height: number
  regionSize: [number, number]
  selectorVersion: number
  bandNames: string[]
}

/**
 * Fully-committed per-level state. Replaces six top-level fields that were
 * previously mutated independently across `initializeLevel`, `switchToLevel`,
 * `_initialize`, and `setSelector` — any of which was one `await` away from
 * a half-commit race.
 *
 * `RegionRenderer.activeLevel` is either `null` (nothing loaded) or a fully-
 * formed runtime; readers never see a partial level.
 */
export interface LevelRuntime {
  index: number
  zarrArray: zarr.Array<zarr.DataType>
  width: number
  height: number
  regionSize: [number, number]
  baseSliceArgs: (number | zarr.Slice)[]
  baseMultiValueDims: Array<{
    dimIndex: number
    dimName: string
    values: number[]
    labels: (number | string)[]
  }>
}

export type QueryLevelSnapshot = Pick<
  LevelRuntime,
  'index' | 'zarrArray' | 'width' | 'height'
>
