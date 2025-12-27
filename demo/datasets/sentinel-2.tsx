import React from 'react'
import type { Selector } from '@carbonplan/zarr-layer'
import type { Dataset, ControlsProps } from './types'
import { useAppStore } from '../lib/store'

type Sentinel2State = Record<string, never>

const Controls = ({ state, setState }: ControlsProps<Sentinel2State>) => {
  return null
}

// UTM Zone 37N projection parameters (EPSG:32637)
const UTM_37N_PROJ4 = '+proj=utm +zone=37 +datum=WGS84 +units=m +no_defs'

// Sentinel-2 bounds in UTM Zone 37N meters from zarr metadata [xMin, yMin, xMax, yMax]
const SENTINEL2_BOUNDS: [number, number, number, number] = [
  199980.0, 4590240.0, 309780.0, 4700040.0,
]

const sentinel2_blue: Dataset<Sentinel2State> = {
  id: 'sentinel_2_l2a',
  source:
    'https://s3.explorer.eopf.copernicus.eu/esa-zarr-sentinel-explorer-fra/tests-output/sentinel-2-l2a/S2C_MSIL2A_20251218T083401_N0511_R021_T37TBG_20251218T112007.zarr/measurements/reflectance',
  variable: 'b04', // Blue
  colormap: 'blues',
  clim: [0, 1],
  latIsAscending: false,
  zarrVersion: 3,
  // Map UTM coordinate dimension names to lat/lon
  spatialDimensions: {
    lat: 'y',
    lon: 'x',
  },
  proj4: UTM_37N_PROJ4,
  bounds: SENTINEL2_BOUNDS,
  info: 'Sentinel-2 L2A RGB (B04/B03/B02) - UTM 37N',
  sourceInfo:
    'Sentinel-2 Level 2A reflectance data from Copernicus EOPF. UTM Zone 37N (T37TBG) reprojected on-the-fly using proj4. RGB = B04 (Red), B03 (Green), B02 (Blue).',
  defaultState: {},
  Controls,
  buildLayerProps: () => ({ selector: { band: ['b04', 'b03', 'b02'] } }),
}

export default sentinel2_blue
