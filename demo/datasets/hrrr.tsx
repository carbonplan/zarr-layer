import React from 'react'
import type { Selector } from '@carbonplan/zarr-layer'
import type { Dataset, ControlsProps } from './types'
import { BandSelector } from '../components/shared-controls'
import { useAppStore } from '../lib/store'

const VARIABLES = {
  TMP: {
    level: '2m_above_ground',
    clim: [250, 310],
    colormap: 'fire',
    label: 'Temperature (2m)',
  },
  GUST: {
    level: 'surface',
    clim: [0, 30],
    colormap: 'cool',
    label: 'Wind Gust',
  },
  RH: {
    level: '2m_above_ground',
    clim: [0, 100],
    colormap: 'blues',
    label: 'Rel. Humidity',
  },
  PRES: {
    level: 'surface',
    clim: [80000, 105000],
    colormap: 'cool',
    label: 'Surface Pressure',
  },
} as const

type VariableKey = keyof typeof VARIABLES

const VARIABLE_KEYS = Object.keys(VARIABLES) as VariableKey[]

type HRRRState = {
  variable: VariableKey
}

const Controls = ({ state, setState }: ControlsProps<HRRRState>) => {
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

// HRRR Lambert Conformal Conic projection parameters
const HRRR_PROJ4 =
  '+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs'

// HRRR grid bounds in Lambert meters [xMin, yMin, xMax, yMax]
const HRRR_BOUNDS: [number, number, number, number] = [
  -2697520, -1587306, 2697480, 1586694,
]

const getYesterdayDateString = () => {
  const now = new Date()
  now.setUTCDate(now.getUTCDate() - 1)
  const year = now.getUTCFullYear()
  const month = String(now.getUTCMonth() + 1).padStart(2, '0')
  const day = String(now.getUTCDate()).padStart(2, '0')
  return `${year}${month}${day}`
}

const HRRR_DATE = getYesterdayDateString()

const hrrr: Dataset<HRRRState> = {
  id: 'hrrr_weather',
  source: `https://hrrrzarr.s3.amazonaws.com/sfc/${HRRR_DATE}/${HRRR_DATE}_12z_anl.zarr`,
  variable: '2m_above_ground/TMP/2m_above_ground/TMP',
  clim: [250, 310],
  colormap: 'fire',
  zarrVersion: 2,
  info: 'HRRR Weather Model (CONUS) - Yesterday 12Z',
  sourceInfo:
    'NOAA High-Resolution Rapid Refresh model via MesoWest. 3km resolution weather analysis over CONUS, updated daily. Reprojected from Lambert Conformal Conic using proj4.',
  fillValue: -9999,
  // Map projection coordinate dimension names to lat/lon
  spatialDimensions: {
    lat: 'projection_y_coordinate',
    lon: 'projection_x_coordinate',
  },
  proj4: HRRR_PROJ4,
  bounds: HRRR_BOUNDS,
  defaultState: {
    variable: 'TMP',
  },
  Controls,
  buildLayerProps: (state) => {
    const config = VARIABLES[state.variable]
    const selector: Selector = {}
    return {
      selector,
      variable: `${config.level}/${state.variable}/${config.level}/${state.variable}`,
    }
  },
}

export default hrrr
