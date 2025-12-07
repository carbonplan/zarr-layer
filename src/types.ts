import * as zarr from 'zarrita'

export type ColormapArray = number[][] | string[]

export type SelectorValue =
  | number
  | number[]
  | string
  | string[]
  | [number, number]

export interface ZarrSelectorsProps {
  selected: SelectorValue
  type?: 'index' | 'value'
}

export type SelectorMap = Record<string, ZarrSelectorsProps>

export interface XYLimits {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
}

export interface XYLimitsProps extends XYLimits {}

export interface DimensionNamesProps {
  time?: string
  elevation?: string
  lat?: string
  lon?: string
  others?: string[]
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
  selector?: Record<string, number | number[] | string | string[]>
  colormap: ColormapArray
  clim: [number, number]
  opacity?: number
  minRenderZoom?: number
  zarrVersion?: 2 | 3
  dimensionNames?: DimensionNamesProps
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
