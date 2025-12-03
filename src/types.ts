import { allColorScales } from './jsColormaps'
import * as zarr from 'zarrita'

export type ColorMapName = (typeof allColorScales)[number]

export interface ColorMapInfo {
  [key: string]: {
    interpolate: boolean
    colors: number[][]
  }
}

export interface ZarrSelectorsProps {
  selected: number | number[] | string | [number, number]
  type?: 'index' | 'value'
}

export interface XYLimits {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
}

export interface XYLimitsProps extends XYLimits {}

export interface ZarrLevelMetadata {
  width: number
  height: number
}

export interface DimensionNamesProps {
  time?: string
  elevation?: string
  lat?: string
  lon?: string
  others?: string[]
}

export interface DimIndicesProps {
  [key: string]: { name: string; index: number; array: zarr.Array<any> | null }
}

export interface MaplibreLayerOptions {
  id: string
  source: string
  variable: string
  selector?: Record<string, number | number[]>
  colormap?: ColorMapName | number[][] | string[]
  vmin: number
  vmax: number
  opacity?: number
  minRenderZoom?: number
  zarrVersion?: 2 | 3
  dimensionNames?: DimensionNamesProps
  noDataMin?: number
  noDataMax?: number
  customFragmentSource?: string
  customFrag?: string
  uniforms?: Record<string, number>
  renderingMode?: '2d' | '3d'
}

export type CRS = 'EPSG:4326' | 'EPSG:3857'

export interface DataSliceProps {
  startX: number
  endX: number
  startY: number
  endY: number
  startElevation?: number
  endElevation?: number
}

export interface SliceArgs {
  [key: number]: number | zarr.Slice
}

export interface ColorScaleProps {
  min: number
  max: number
  colors: number[][]
}
