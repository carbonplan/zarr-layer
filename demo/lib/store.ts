import { create } from 'zustand'
import {
  DATASET_MODULES,
  DEFAULT_DATASET_ID,
  DatasetId,
  DatasetModuleMap,
  DatasetStateMap,
} from './constants'
import { MapProvider } from '../components/map-shared'
import type { PointQueryResult, RegionQueryResult } from '../../src/query/types'
import type { MapInstance } from '../components/map-shared'
import type { ZarrLayer } from '@carbonplan/zarr-layer'
import type { LoadingState } from '@carbonplan/zarr-layer'

type DatasetStateStore = {
  [K in DatasetId]?: Partial<DatasetStateMap[K]>
}

interface AppState {
  sidebarWidth: number
  datasetId: DatasetId
  opacity: number
  clim: [number, number]
  colormap: string
  globeProjection: boolean
  mapProvider: MapProvider
  datasetState: DatasetStateStore
  loadingState: LoadingState
  pointResult: PointQueryResult | null
  regionResult: RegionQueryResult | null
  mapInstance: MapInstance | null
  zarrLayer: InstanceType<typeof ZarrLayer> | null
  setSidebarWidth: (width: number) => void
  setDatasetId: (id: DatasetId) => void
  setOpacity: (opacity: number) => void
  setClim: (clim: [number, number]) => void
  setColormap: (colormap: string) => void
  setGlobeProjection: (globeProjection: boolean) => void
  setMapProvider: (provider: MapProvider) => void
  setDatasetState: <K extends DatasetId>(
    id: K,
    updates: Partial<DatasetStateMap[K]>,
  ) => void
  setActiveDatasetState: (updates: Partial<DatasetStateMap[DatasetId]>) => void
  setLoadingState: (state: LoadingState) => void
  setPointResult: (result: PointQueryResult | null) => void
  setRegionResult: (result: RegionQueryResult | null) => void
  setMapInstance: (map: MapInstance | null) => void
  setZarrLayer: (layer: InstanceType<typeof ZarrLayer> | null) => void
  getDatasetModule: () => DatasetModuleMap[DatasetId]
  getDatasetState: () => DatasetStateMap[DatasetId]
}

export const useAppStore = create<AppState>((set, get) => ({
  sidebarWidth: 0,
  datasetId: DEFAULT_DATASET_ID,
  opacity: 1,
  clim: DATASET_MODULES[DEFAULT_DATASET_ID].clim,
  colormap: DATASET_MODULES[DEFAULT_DATASET_ID].colormap,
  globeProjection: true,
  mapProvider: 'maplibre',
  datasetState: {
    [DEFAULT_DATASET_ID]: {
      ...DATASET_MODULES[DEFAULT_DATASET_ID].defaultState,
    },
  },
  loadingState: { loading: false, metadata: false, chunks: false },
  pointResult: null,
  regionResult: null,
  mapInstance: null,
  zarrLayer: null,
  setSidebarWidth: (width) => set({ sidebarWidth: width }),
  setLoadingState: (loadingState) => set({ loadingState }),
  setDatasetId: (id) => {
    const module = DATASET_MODULES[id]
    if (!module) return

    set((state) => ({
      datasetId: id,
      clim: module.clim,
      colormap: module.colormap,
      datasetState: {
        ...state.datasetState,
        [id]: state.datasetState[id] ?? module.defaultState,
      },
    }))
  },
  setOpacity: (opacity) => set({ opacity }),
  setClim: (clim) => set({ clim }),
  setColormap: (colormap) => set({ colormap }),
  setGlobeProjection: (globeProjection) => set({ globeProjection }),
  setMapProvider: (mapProvider) => set({ mapProvider }),
  setDatasetState: (id, updates) =>
    set((state): Partial<AppState> => {
      const module = DATASET_MODULES[id]
      const current = state.datasetState[id] ?? module?.defaultState
      const nextDatasetState: DatasetStateStore = {
        ...state.datasetState,
        [id]: {
          ...(current ?? {}),
          ...updates,
        } as DatasetStateStore[typeof id],
      }
      return { datasetState: nextDatasetState }
    }),
  setActiveDatasetState: (updates) => {
    const activeId = get().datasetId
    get().setDatasetState(activeId, updates)
  },
  setPointResult: (pointResult) => set({ pointResult }),
  setRegionResult: (regionResult) => set({ regionResult }),
  setMapInstance: (mapInstance) => set({ mapInstance }),
  setZarrLayer: (zarrLayer) => set({ zarrLayer }),
  getDatasetModule: () => DATASET_MODULES[get().datasetId],
  getDatasetState: () => {
    const id = get().datasetId
    const module = DATASET_MODULES[id]
    const stored = get().datasetState[id]
    return (stored ?? module.defaultState) as DatasetStateMap[DatasetId]
  },
}))
