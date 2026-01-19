import React from 'react'
import type { Dataset, ControlsProps } from './types'
import { BandSelector } from '../components/shared-controls'
import { useAppStore } from '../lib/store'

const VARIABLES = {
  DEM: {
    clim: [0, 4000],
    colormap: 'warm',
    label: 'DEM (m)',
  },
  hillshade: {
    clim: [0, 1],
    colormap: 'wind',
    label: 'Hillshade',
  },
  slope: {
    clim: [0, 90],
    colormap: 'warm',
    label: 'Slope angle (degrees)',
  },
  aspect: {
    clim: [0, 360],
    colormap: 'cool',
    label: 'Aspect (degrees)',
  },
} as const

type VariableKey = keyof typeof VARIABLES

const VARIABLE_KEYS = Object.keys(VARIABLES) as VariableKey[]

type DEMState = {
  variable: VariableKey
}

const Controls = ({ state, setState }: ControlsProps<DEMState>) => {
  const setClim = useAppStore((s) => s.setClim)
  const setColormap = useAppStore((s) => s.setColormap)

  const handleVariableChange = (variable: VariableKey) => {
    setState({ variable })
    const config = VARIABLES[variable]
    setClim([...config.clim] as [number, number])
    setColormap(config.colormap)
  }

  return (
    <BandSelector
      value={state.variable}
      options={VARIABLE_KEYS}
      onChange={handleVariableChange}
      label='Variable'
    />
  )
}

const usgsdem: Dataset<DEMState> = {
  id: 'usgsdem',
  source: `https://carbonplan-share.s3.us-west-2.amazonaws.com/zarr-layer-examples/USGS-CONUS-DEM-10m.zarr`,
  variable: 'DEM',
  clim: [0, 4000],
  bounds: [
    -125.5958334317414, 24.75000001939236, -66.62101857071806,
    49.09379633476274,
  ],
  latIsAscending: false,
  sourceInfo:
    'USGS 10m DEM with derived hillshade, aspect and slope angle shading.',
  colormap: 'warm',
  zarrVersion: 3,
  info: 'USGS 10m DEM',
  center: [-98, 39],
  zoom: 3,
  defaultState: {
    variable: 'DEM',
  },
  Controls,
  buildLayerProps: (state) => {
    return {
      selector: {},
      variable: state.variable,
    }
  },
}

export default usgsdem
