import * as zarr from 'zarrita'

/** Bounds tuple: [xMin, yMin, xMax, yMax] */
export type Bounds = [number, number, number, number]

export interface RequestParameters extends Omit<RequestInit, 'headers'> {
  url: string
  headers?: { [key: string]: string }
}

/**
 * Options passed to transformRequest
 */
export interface TransformRequestOptions {
  /** HTTP method that will be used for this request */
  method?: 'GET' | 'HEAD'
}

export type TransformRequest = (
  url: string,
  options?: TransformRequestOptions
) => RequestParameters | Promise<RequestParameters>

export type ColormapArray = number[][] | string[]

export type SelectorValue = number | number[] | string | string[]

export interface SelectorSpec {
  selected: SelectorValue
  type?: 'index' | 'value'
}

// Public shape for callers: full selector object (per-dimension value or spec).
export type Selector = Record<string, SelectorValue | SelectorSpec>

// Internal normalized form (object per dimension).
export type NormalizedSelector = Record<string, SelectorSpec>

/**
 * Override the names used to identify spatial dimensions (lat/lon).
 * Only needed if your dataset uses non-standard names that aren't auto-detected.
 * Standard names (lat, latitude, y, lon, longitude, x) are detected automatically.
 */
export interface SpatialDimensions {
  lat?: string
  lon?: string
}

export interface DimIndicesProps {
  [key: string]: {
    name: string
    index: number
    array: zarr.Array<zarr.DataType> | null
  }
}

export interface LoadingState {
  loading: boolean
  metadata: boolean
  chunks: boolean
  error?: Error | null
}

export type LoadingStateCallback = (state: LoadingState) => void

export interface ZarrLayerOptions {
  id: string
  /**
   * URL to the Zarr store. Required unless `store` is provided.
   */
  source?: string
  variable: string
  /**
   * Custom zarrita-compatible store to use instead of creating a FetchStore from source.
   * Useful for IcechunkStore or other custom storage backends.
   *
   * The store must implement the zarrita Readable interface with at minimum:
   * - `get(key: string): Promise<Uint8Array | undefined>` - fetch data at path
   *
   * Optionally implement AsyncReadable for range requests:
   * - `getRange(key: string, range: RangeQuery): Promise<Uint8Array | undefined>`
   *
   * When provided:
   * - `source` becomes optional (falls back to layer id for identification)
   * - Metadata caching is bypassed (each layer fetches fresh metadata)
   *
   * @example
   * ```ts
   * import { IcechunkStore } from '@icechunk/icechunk-python'
   * const store = await IcechunkStore.open(...)
   * new ZarrLayer({ id: 'my-layer', store, variable: 'temperature', ... })
   * ```
   */
  store?: zarr.Readable<unknown>
  selector?: Selector
  colormap: ColormapArray
  clim: [number, number]
  opacity?: number
  minzoom?: number
  maxzoom?: number
  zarrVersion?: 2 | 3
  spatialDimensions?: SpatialDimensions
  /**
   * Explicit spatial bounds [xMin, yMin, xMax, yMax].
   * Units depend on CRS: degrees for EPSG:4326, source CRS units (e.g. meters) when proj4 is provided.
   * If not provided, bounds are read from coordinate arrays or default to global.
   */
  bounds?: Bounds
  /**
   * CRS identifier for built-in projections (EPSG:4326 or EPSG:3857).
   * For any other CRS, provide a matching proj4 definition.
   */
  crs?: string
  latIsAscending?: boolean | null
  fillValue?: number
  customFrag?: string
  uniforms?: Record<string, number>
  renderingMode?: '2d' | '3d'
  onLoadingStateChange?: LoadingStateCallback
  /**
   * Throttle interval in milliseconds for data fetching during rapid selector changes.
   * Higher values reduce network requests when scrubbing through e.g. time sliders.
   * Set to 0 to disable throttling. Default: 100ms.
   */
  throttleMs?: number
  /**
   * Proj4 definition string for reprojection (untiled mode only).
   * When provided, bounds are interpreted as source CRS units and data is reprojected to Web Mercator.
   * Example: "+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs"
   */
  proj4?: string
  /**
   * Function to transform request URLs and add custom headers/credentials.
   * Useful for authentication, proxy routing, or request customization.
   * When provided, the store cache is bypassed to prevent credential sharing between layers.
   */
  transformRequest?: TransformRequest
  /**
   * Enable full polar coverage in Mapbox globe view for untiled EPSG:4326 or
   * proj4 datasets. Has no effect on tiled or EPSG:3857 data.
   *
   * MapLibre globe always renders to the poles automatically.
   *
   * For Mapbox, this enables an experimental direct ECEF path that bypasses
   * tile draping. Only activates at the fully-globe zoom endpoint; during
   * the globe-to-mercator zoom morph the layer falls back to the standard
   * draped path. Incompatible with Mapbox terrain — when terrain is enabled
   * the layer uses the draped tile path. Relies on Mapbox internal APIs and
   * may break across Mapbox GL JS versions.
   *
   * Default: `false`
   */
  renderPoles?: boolean
}

export type CRS = 'EPSG:4326' | 'EPSG:3857'

// Untiled multiscale types (zarr-conventions/multiscales format)
export interface UntiledLevel {
  asset: string
  scale: [number, number]
  translation: [number, number]
  shape?: number[]
  chunks?: number[]
  scaleFactor?: number
  addOffset?: number
  fillValue?: number | null
  dtype?: string | null
}

export interface BoundsLike {
  getWest(): number
  getEast(): number
  toArray(): [number, number][]
}

export interface MapLike {
  // type can be string, array expression, or complex PropertyValueSpecification
  getProjection?(): { type?: unknown; name?: string } | null
  getRenderWorldCopies?(): boolean
  getTerrain?(): unknown
  on?(event: string, handler: (...args: unknown[]) => void): void
  off?(event: string, handler: (...args: unknown[]) => void): void
  triggerRepaint?(): void
  getBounds?(): BoundsLike | null
  getZoom?(): number
  painter?: { context?: { gl?: unknown } }
  renderer?: { getContext?: () => unknown }
}
