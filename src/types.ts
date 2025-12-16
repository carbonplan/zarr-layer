import * as zarr from 'zarrita'

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
}

export type LoadingStateCallback = (state: LoadingState) => void

export interface ZarrLayerOptions {
  id: string
  source: string
  variable: string
  selector?: Selector
  colormap: ColormapArray
  clim: [number, number]
  opacity?: number
  minzoom?: number
  maxzoom?: number
  zarrVersion?: 2 | 3
  spatialDimensions?: SpatialDimensions
  /**
   * Explicit spatial bounds [west, south, east, north] in degrees.
   * Used when coordinate arrays aren't available in the zarr store.
   * If not provided, bounds are read from coordinate arrays or default to global.
   */
  bounds?: [number, number, number, number]
  latIsAscending?: boolean | null
  fillValue?: number
  customFrag?: string
  uniforms?: Record<string, number>
  renderingMode?: '2d' | '3d'
  onLoadingStateChange?: LoadingStateCallback
}

export type CRS = 'EPSG:4326' | 'EPSG:3857'

export interface BoundsLike {
  getWest(): number
  getEast(): number
  toArray(): [[number, number], [number, number]]
}

export interface MapLike {
  getProjection?(): { type?: string; name?: string } | null
  getRenderWorldCopies?(): boolean
  on?(event: string, handler: (...args: unknown[]) => void): void
  off?(event: string, handler: (...args: unknown[]) => void): void
  triggerRepaint?(): void
  getBounds?(): BoundsLike | null
  getZoom?(): number
  painter?: { context?: { gl?: unknown } }
  renderer?: { getContext?: () => unknown }
}
