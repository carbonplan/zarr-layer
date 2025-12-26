import React from 'react'
import type { Selector } from '@carbonplan/zarr-layer'

export type DatasetConfig = {
  id: string
  source: string
  variable: string
  clim: [number, number]
  colormap: string
  zarrVersion: 2 | 3
  info: string
  sourceInfo: string
  fillValue?: number
  latIsAscending?: boolean
  center?: [number, number]
  zoom?: number
  spatialDimensions?: {
    lat?: string
    lon?: string
  }
  bounds?: [number, number, number, number]
  proj4?: string
}

export type ControlsProps<State> = {
  state: State
  setState: (updates: Partial<State>) => void
}

export type LayerProps = {
  selector: Selector
  variable?: string
  customFrag?: string
  uniforms?: Record<string, number>
}

export type Dataset<State = Record<string, unknown>> = DatasetConfig & {
  defaultState: State
  Controls: React.FC<ControlsProps<State>>
  buildLayerProps: (state: State) => LayerProps
}
