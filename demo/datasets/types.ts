import React from 'react'
import type { Selector } from '@carbonplan/zarr-layer'

export type DatasetCommonConfig = {
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
  /** Explicit bounds [west, south, east, north] when coordinate arrays aren't available */
  bounds?: [number, number, number, number]
}

export type DatasetControlsProps<State> = {
  state: State
  setState: (updates: Partial<State>) => void
}

export type BuildLayerResult = {
  selector: Selector
  variable?: string
  customFrag?: string
  uniforms?: Record<string, number>
}

export type DatasetModule<
  State extends Record<string, unknown> = Record<string, unknown>,
> = DatasetCommonConfig & {
  defaultState: State
  Controls: React.FC<DatasetControlsProps<State>>
  buildLayerProps: (args: { state: State }) => BuildLayerResult
}

export type TimeDatasetState = { time: number }

export const defineDatasetModule = <State extends Record<string, unknown>>(
  module: DatasetModule<State>,
) => module

export const createDatasetList = <
  const Modules extends readonly DatasetModule<any>[],
>(
  ...modules: Modules
) => modules

export const defineModules = <
  Modules extends readonly DatasetModule<any>[],
  Ids extends Modules[number]['id'],
  ModuleMap extends { [K in Ids]: Extract<Modules[number], { id: K }> },
>(
  modules: Modules,
) => {
  const map = modules.reduce(
    (acc, module) => {
      acc[module.id as Ids] = module as Modules[number]
      return acc
    },
    {} as Record<Ids, Modules[number]>,
  )
  return map as ModuleMap
}
