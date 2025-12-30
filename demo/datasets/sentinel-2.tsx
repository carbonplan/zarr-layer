import React from 'react'
import type { Dataset, ControlsProps, LayerProps } from './types'
import { BandSelector } from '../components/shared-controls'

const BANDS = ['b04', 'b03', 'b02'] as const
type Band = (typeof BANDS)[number]

type State = {
  band: Band
}

const Controls = ({ state, setState }: ControlsProps<State>) => {
  return (
    <BandSelector
      value={state.band}
      options={BANDS}
      onChange={(band) => setState({ band })}
      label='Band'
    />
  )
}

const buildLayerProps = (state: State): LayerProps => {
  return {
    selector: {},
    variable: state.band,
  }
}

const sentinel2: Dataset<State> = {
  id: 'sentinel_2_eopf',
  source:
    'https://s3.explorer.eopf.copernicus.eu/esa-zarr-sentinel-explorer-fra/tests-output/sentinel-2-l2a/S2C_MSIL2A_20251218T083401_N0511_R021_T37TBG_20251218T112007.zarr/measurements/reflectance',
  variable: 'b04',
  clim: [0, 0.1],
  colormap: 'greys',
  zarrVersion: 3,
  latIsAscending: false,
  info: 'Sentinel-2 L2A (EOPF, UTM projection)',
  sourceInfo:
    'Sentinel-2 Level-2A reflectance from ESA EOPF. 6-level pyramid (10m-720m) in UTM Zone 37N with proj4 reprojection. Multiscale levels are heterogeneous.',
  bounds: [199980.0, 4590240.0, 309780.0, 4700040.0],
  proj4: '+proj=utm +zone=37 +datum=WGS84 +units=m +no_defs',
  spatialDimensions: { lat: 'y', lon: 'x' },
  center: [36, 42],
  zoom: 8,
  defaultState: {
    band: 'b04',
  },
  Controls,
  buildLayerProps,
}

export default sentinel2
