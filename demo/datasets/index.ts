import carbonplan4d from './carbonplan-4d'
import hurricane from './hurricane'
import { createSimpleDataset } from './simple'
import { createTimeDataset } from './time'
import untiled4326 from './untiled-4326'
import type { Dataset } from './types'

export const DATASETS: Dataset<any>[] = [
  carbonplan4d,
  createTimeDataset({
    id: 'salinity_v2',
    source:
      'https://atlantis-vis-o.s3-ext.jc.rl.ac.uk/nemotest101/pyramid2/T1d/sos_abs.zarr',
    variable: 'sos_abs',
    clim: [30, 37],
    colormap: 'blues',
    zarrVersion: 2,
    info: 'Ocean salinity (v2 pyramid, EPSG:3857)',
    sourceInfo: 'v2 pyramid format (EPSG:3857)',
  }),
  createTimeDataset({
    id: 'temperature_v3',
    source:
      'https://atlantis-vis-o.s3-ext.jc.rl.ac.uk/noc-npd-era5-demo/npd-eorca1-era5v1/gn/T1y/tos_con',
    variable: 'tos_con',
    clim: [0, 50],
    colormap: 'fire',
    zarrVersion: 3,
    info: 'Ocean temperature (v3 pyramid, EPSG:3857)',
    sourceInfo: 'v3 pyramid (EPSG:3857)',
  }),
  hurricane,
  createTimeDataset({
    id: 'tasmax_pyramid_4326',
    source:
      'https://carbonplan-benchmarks.s3.us-west-2.amazonaws.com/data/NEX-GDDP-CMIP6/ACCESS-CM2/historical/r1i1p1f1/tasmax/tasmax_day_ACCESS-CM2_historical_r1i1p1f1_gn/pyramids-v2-4326-True-128-1-0-0-f4-0-0-0-gzipL1-100',
    variable: 'tasmax',
    clim: [220, 320],
    colormap: 'fire',
    zarrVersion: 2,
    info: 'tasmax v2 pyramid (EPSG:4326)',
    sourceInfo: 'v2 pyramid (EPSG:4326)',
    maxTime: 729,
  }),
  createTimeDataset({
    id: 'tasmax_pyramid_v3_4326',
    source:
      'https://carbonplan-benchmarks.s3.us-west-2.amazonaws.com/data/NEX-GDDP-CMIP6/ACCESS-CM2/historical/r1i1p1f1/tasmax/tasmax_day_ACCESS-CM2_historical_r1i1p1f1_gn/pyramids-v3-4326-True-128-1-0-0-f4-0-0-gzipL1-100',
    variable: 'tasmax',
    clim: [220, 320],
    colormap: 'fire',
    zarrVersion: 3,
    info: 'tasmax v3 pyramid (EPSG:4326)',
    sourceInfo: 'v3 pyramid (EPSG:4326)',
    maxTime: 729,
  }),
  createTimeDataset({
    id: 'pr single image',
    source:
      'https://carbonplan-scratch.s3.us-west-2.amazonaws.com/zarr-pyramids/zarr-v3-single-layer-default.zarr',
    variable: 'pr',
    clim: [0, 20],
    colormap: 'blues',
    zarrVersion: 3,
    info: 'Precipitation (single image, global)',
    sourceInfo: 'v3 single image (global)',
  }),
  createTimeDataset({
    id: 'delta_FG_CO2',
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
    info: 'Delta FG CO2 (single image, global)',
    sourceInfo: 'v2 single image (global)',
  }),
  untiled4326,
  createSimpleDataset({
    id: 'untiled_2level_3857',
    source:
      'https://carbonplan-share.s3.us-west-2.amazonaws.com/scratch/ndpyramid/2-lvl-test-web-mercator.zarr',
    variable: 'pr',
    clim: [0, 20],
    colormap: 'blues',
    zarrVersion: 3,
    info: 'Untiled 2-level (EPSG:3857)',
    sourceInfo:
      'zarr-conventions/multiscales format. Loads different resolutions based on current zoom and requests chunks based on current viewport.',
  }),
]

export const DATASET_MAP = Object.fromEntries(
  DATASETS.map((d) => [d.id, d])
) as Record<string, Dataset<any>>

export const DEFAULT_DATASET_ID = DATASETS[0].id

export type { Dataset } from './types'
