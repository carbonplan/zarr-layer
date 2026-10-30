import { create } from 'zustand'
import { DATASET_MAP, DEFAULT_DATASET_ID } from '../datasets'
import type { Dataset } from '../datasets/types'
import type { MapProvider, MapInstance } from '../components/map-shared'
import type {
  ZarrLayer,
  QueryResult,
  LoadingState,
} from '@carbonplan/zarr-layer'

export type DemoProjection = 'globe' | 'mercator' | 'antarctic'

interface AppState {
  sidebarWidth: number | null
  datasetId: string
  opacity: number
  clim: [number, number]
  colormap: string
  projection: DemoProjection
  terrainEnabled: boolean
  renderPoles: boolean
  mapProvider: MapProvider
  datasetState: Record<string, unknown>
  loadingState: LoadingState
  pointResult: QueryResult | null
  regionResult: QueryResult | null
  hoverQueryEnabled: boolean
  mapInstance: MapInstance | null
  zarrLayer: InstanceType<typeof ZarrLayer> | null
  setSidebarWidth: (width: number | null) => void
  setDatasetId: (id: string) => void
  setOpacity: (opacity: number) => void
  setClim: (clim: [number, number]) => void
  setColormap: (colormap: string) => void
  setProjection: (projection: DemoProjection) => void
  setTerrainEnabled: (terrainEnabled: boolean) => void
  setRenderPoles: (renderPoles: boolean) => void
  setMapProvider: (provider: MapProvider) => void
  setActiveDatasetState: (updates: Record<string, unknown>) => void
  setLoadingState: (state: LoadingState) => void
  setPointResult: (result: QueryResult | null) => void
  setRegionResult: (result: QueryResult | null) => void
  setHoverQueryEnabled: (enabled: boolean) => void
  setMapInstance: (map: MapInstance | null) => void
  setZarrLayer: (layer: InstanceType<typeof ZarrLayer> | null) => void
  getDatasetModule: () => Dataset
}

const defaultModule = DATASET_MAP[DEFAULT_DATASET_ID]

export const useAppStore = create<AppState>((set, get) => ({
  sidebarWidth: null,
  datasetId: DEFAULT_DATASET_ID,
  opacity: 1,
  clim: defaultModule.clim,
  colormap: defaultModule.colormap,
  projection: 'globe',
  terrainEnabled: false,
  renderPoles: false,
  mapProvider: 'maplibre',
  datasetState: { ...defaultModule.defaultState },
  loadingState: { loading: false, metadata: false, chunks: false },
  pointResult: null,
  regionResult: null,
  hoverQueryEnabled: false,
  mapInstance: null,
  zarrLayer: null,
  setSidebarWidth: (width) => set({ sidebarWidth: width }),
  setLoadingState: (loadingState) => set({ loadingState }),
  setDatasetId: (id) => {
    const module = DATASET_MAP[id]
    if (!module || id === get().datasetId) return

    set({
      datasetId: id,
      clim: module.clim,
      colormap: module.colormap,
      datasetState: { ...module.defaultState },
    })
  },
  setOpacity: (opacity) => set({ opacity }),
  setClim: (clim) => set({ clim }),
  setColormap: (colormap) => set({ colormap }),
  setProjection: (projection) => set({ projection }),
  setTerrainEnabled: (terrainEnabled) => set({ terrainEnabled }),
  setRenderPoles: (renderPoles) => set({ renderPoles }),
  // The Antarctic projection is MapLibre's addProjection, which Mapbox lacks.
  setMapProvider: (mapProvider) =>
    set((state) => ({
      mapProvider,
      projection:
        mapProvider === 'mapbox' && state.projection === 'antarctic'
          ? 'globe'
          : state.projection,
    })),
  setActiveDatasetState: (updates) => {
    set((state) => ({
      datasetState: { ...state.datasetState, ...updates },
    }))
  },
  setPointResult: (pointResult) => set({ pointResult }),
  setRegionResult: (regionResult) => set({ regionResult }),
  setHoverQueryEnabled: (hoverQueryEnabled) => set({ hoverQueryEnabled }),
  setMapInstance: (mapInstance) => set({ mapInstance }),
  setZarrLayer: (zarrLayer) => set({ zarrLayer }),
  getDatasetModule: () => DATASET_MAP[get().datasetId],
}))
