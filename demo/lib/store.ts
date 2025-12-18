import { create } from 'zustand'
import { DATASET_MAP, DEFAULT_DATASET_ID } from '../datasets'
import type { Dataset } from '../datasets'
import type { MapProvider, MapInstance } from '../components/map-shared'
import type {
  ZarrLayer,
  QueryResult,
  LoadingState,
} from '@carbonplan/zarr-layer'

interface AppState {
  sidebarWidth: number
  datasetId: string
  opacity: number
  clim: [number, number]
  colormap: string
  globeProjection: boolean
  mapProvider: MapProvider
  datasetState: Record<string, Record<string, unknown>>
  loadingState: LoadingState
  pointResult: QueryResult | null
  regionResult: QueryResult | null
  mapInstance: MapInstance | null
  zarrLayer: InstanceType<typeof ZarrLayer> | null
  setSidebarWidth: (width: number) => void
  setDatasetId: (id: string) => void
  setOpacity: (opacity: number) => void
  setClim: (clim: [number, number]) => void
  setColormap: (colormap: string) => void
  setGlobeProjection: (globeProjection: boolean) => void
  setMapProvider: (provider: MapProvider) => void
  setActiveDatasetState: (updates: Record<string, unknown>) => void
  setLoadingState: (state: LoadingState) => void
  setPointResult: (result: QueryResult | null) => void
  setRegionResult: (result: QueryResult | null) => void
  setMapInstance: (map: MapInstance | null) => void
  setZarrLayer: (layer: InstanceType<typeof ZarrLayer> | null) => void
  getDatasetModule: () => Dataset<any>
  getDatasetState: () => Record<string, unknown>
}

const defaultModule = DATASET_MAP[DEFAULT_DATASET_ID]

export const useAppStore = create<AppState>((set, get) => ({
  sidebarWidth: 0,
  datasetId: DEFAULT_DATASET_ID,
  opacity: 1,
  clim: defaultModule.clim,
  colormap: defaultModule.colormap,
  globeProjection: true,
  mapProvider: 'maplibre',
  datasetState: {
    [DEFAULT_DATASET_ID]: { ...defaultModule.defaultState },
  },
  loadingState: { loading: false, metadata: false, chunks: false },
  pointResult: null,
  regionResult: null,
  mapInstance: null,
  zarrLayer: null,
  setSidebarWidth: (width) => set({ sidebarWidth: width }),
  setLoadingState: (loadingState) => set({ loadingState }),
  setDatasetId: (id) => {
    const module = DATASET_MAP[id]
    if (!module) return

    set((state) => ({
      datasetId: id,
      clim: module.clim,
      colormap: module.colormap,
      datasetState: {
        ...state.datasetState,
        [id]: state.datasetState[id] ?? { ...module.defaultState },
      },
    }))
  },
  setOpacity: (opacity) => set({ opacity }),
  setClim: (clim) => set({ clim }),
  setColormap: (colormap) => set({ colormap }),
  setGlobeProjection: (globeProjection) => set({ globeProjection }),
  setMapProvider: (mapProvider) => set({ mapProvider }),
  setActiveDatasetState: (updates) => {
    const id = get().datasetId
    set((state) => ({
      datasetState: {
        ...state.datasetState,
        [id]: { ...(state.datasetState[id] ?? {}), ...updates },
      },
    }))
  },
  setPointResult: (pointResult) => set({ pointResult }),
  setRegionResult: (regionResult) => set({ regionResult }),
  setMapInstance: (mapInstance) => set({ mapInstance }),
  setZarrLayer: (zarrLayer) => set({ zarrLayer }),
  getDatasetModule: () => DATASET_MAP[get().datasetId],
  getDatasetState: () => {
    const id = get().datasetId
    const module = DATASET_MAP[id]
    return get().datasetState[id] ?? module.defaultState
  },
}))
