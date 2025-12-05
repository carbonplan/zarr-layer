import carbonplan4d from './carbonplan-4d'
import { createTimeDatasetModule } from './time-series'
import { createDatasetList, defineModules } from './types'

const DATASET_MODULE_LIST = createDatasetList(
  carbonplan4d,
  createTimeDatasetModule({
    id: 'salinity_v2',
    source:
      'https://atlantis-vis-o.s3-ext.jc.rl.ac.uk/nemotest101/pyramid2/T1d/sos_abs.zarr',
    variable: 'sos_abs',
    clim: [30, 37],
    colormap: 'blues',
    zarrVersion: 2,
    info: 'NEMO NPD-EORCA1 Salinity',
    sourceInfo: 'Zarr v2 pyramid format (EPSG:3857)',
  }),
  createTimeDatasetModule({
    id: 'temperature_v3',
    source:
      'https://atlantis-vis-o.s3-ext.jc.rl.ac.uk/noc-npd-era5-demo/npd-eorca1-era5v1/gn/T1y/tos_con',
    variable: 'tos_con',
    clim: [0, 27],
    colormap: 'fire',
    zarrVersion: 3,
    info: 'NEMO NPD-EORCA1 Temperature',
    sourceInfo: 'Zarr v3 pyramid format (EPSG:3857)',
  }),
  createTimeDatasetModule({
    id: 'pressure_v3',
    source:
      'https://atlantis-vis-o.s3-ext.jc.rl.ac.uk/hurricanes/era5/florence',
    variable: 'surface_pressure',
    clim: [75000, 104000],
    colormap: 'cool',
    zarrVersion: 3,
    info: 'Hurricane Florence Surface Pressure',
    sourceInfo: 'Zarr v3 format (EPSG:4326)',
  }),
  createTimeDatasetModule({
    id: 'tasmax_pyramid_4326',
    source:
      'https://carbonplan-benchmarks.s3.us-west-2.amazonaws.com/data/NEX-GDDP-CMIP6/ACCESS-CM2/historical/r1i1p1f1/tasmax/tasmax_day_ACCESS-CM2_historical_r1i1p1f1_gn/pyramids-v2-4326-True-128-1-0-0-f4-0-0-0-gzipL1-100',
    variable: 'tasmax',
    clim: [220, 320],
    colormap: 'fire',
    zarrVersion: 2,
    info: 'tasmax v2 pyramid (EPSG:4326)',
    sourceInfo: 'Zarr v2 pyramid (EPSG:4326)',
    maxTime: 729,
  }),
  createTimeDatasetModule({
    id: 'tasmax_pyramid_v3_4326',
    source:
      'https://carbonplan-benchmarks.s3.us-west-2.amazonaws.com/data/NEX-GDDP-CMIP6/ACCESS-CM2/historical/r1i1p1f1/tasmax/tasmax_day_ACCESS-CM2_historical_r1i1p1f1_gn/pyramids-v3-4326-True-128-1-0-0-f4-0-0-gzipL1-100',
    variable: 'tasmax',
    clim: [220, 320],
    colormap: 'fire',
    zarrVersion: 3,
    info: 'tasmax v3 pyramid (EPSG:4326)',
    sourceInfo: 'Zarr v3 pyramid (EPSG:4326)',
    maxTime: 729,
  }),
)

export { DATASET_MODULE_LIST }
export const DATASET_MODULES = defineModules(DATASET_MODULE_LIST)
export const DEFAULT_DATASET_ID = DATASET_MODULE_LIST[0].id

export type DatasetModuleMap = typeof DATASET_MODULES
export type DatasetId = keyof DatasetModuleMap
export type DatasetStateMap = {
  [K in DatasetId]: DatasetModuleMap[K]['defaultState']
}
export type AnyDatasetModule = DatasetModuleMap[keyof DatasetModuleMap]
export type { DatasetModule } from './types'
