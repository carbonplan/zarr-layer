import antarcticEra5 from './antarctic-era5'
import carbonplan4d from './carbonplan-4d'
import hrrr from './hrrr'
import usgsdem from './usgs-dem'
import icechunkPrecip from './icechunk'

import hurricane from './hurricane'
import polar from './polar'
import { createSimpleDataset } from './simple'
import { createTimeDataset } from './time'
import type { Dataset } from './types'

export const DATASETS: Dataset<any>[] = [
  carbonplan4d,
  hrrr,
  icechunkPrecip,
  hurricane,
  createTimeDataset({
    id: 'temperature_v3',
    source:
      'https://atlantis-vis-o.s3-ext.jc.rl.ac.uk/noc-npd-era5-demo/npd-eorca1-era5v1/gn/T1y/tos_con',
    variable: 'tos_con',
    clim: [0, 50],
    colormap: 'fire',
    zarrVersion: 3,
    info: 'Ocean Temperature (EPSG:3857)',
    sourceInfo: 'Zarr v3 tiled pyramid. EPSG:3857, time dimension.',
  }),
  createTimeDataset({
    id: 'tasmax_pyramid_4326',
    source:
      'https://carbonplan-benchmarks.s3.us-west-2.amazonaws.com/data/NEX-GDDP-CMIP6/ACCESS-CM2/historical/r1i1p1f1/tasmax/tasmax_day_ACCESS-CM2_historical_r1i1p1f1_gn/pyramids-v2-4326-True-128-1-0-0-f4-0-0-0-gzipL1-100',
    variable: 'tasmax',
    clim: [220, 320],
    colormap: 'fire',
    zarrVersion: 2,
    info: 'CMIP6 tasmax (EPSG:4326)',
    sourceInfo:
      'NEX-GDDP-CMIP6 tasmax, Zarr v2 tiled pyramid. EPSG:4326, daily time dimension.',
    maxTime: 729,
  }),
  createTimeDataset({
    id: 'delta_fg_co2',
    source:
      'https://carbonplan-oae-efficiency.s3.us-west-2.amazonaws.com/fgco2-2021-180x360.zarr',
    variable: 'FG_CO2_2',
    clim: [-5, 5],
    colormap: 'orangeblue',
    zarrVersion: 2,
    maxTime: 365,
    timeSelectorType: 'index',
    spatialDimensions: { lat: 'nlat', lon: 'nlon' },
    bounds: [-180, -90, 180, 90],
    info: 'Delta FG CO2 (EPSG:4326)',
    sourceInfo:
      'OAE efficiency delta FG CO2, Zarr v2 single resolution. EPSG:4326, daily time dimension.',
    latIsAscending: true,
  }),
  usgsdem,
  polar,
  antarcticEra5,
]

export const DATASET_MAP = Object.fromEntries(
  DATASETS.map((d) => [d.id, d])
) as Record<string, Dataset>

export const DEFAULT_DATASET_ID = usgsdem.id

export type { Dataset } from './types'
