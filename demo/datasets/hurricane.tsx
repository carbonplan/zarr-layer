import React from 'react'
import type { Selector } from '@carbonplan/zarr-layer'
import type { Dataset, ControlsProps } from './types'
import { Slider, BandSelector } from '../components/shared-controls'
import { useAppStore } from '../lib/store'

const VARIABLES = ['surface_pressure', 'velocity'] as const
type Variable = (typeof VARIABLES)[number]

type HurricaneState = {
  variable: Variable
  time: number
  level: number
}

const is4D = (variable: Variable) => variable !== 'surface_pressure'

const Controls = ({ state, setState }: ControlsProps<HurricaneState>) => {
  const setClim = useAppStore((s) => s.setClim)
  const setColormap = useAppStore((s) => s.setColormap)

  const handleVariableChange = (variable: Variable) => {
    setState({ variable })
    if (variable === 'surface_pressure') {
      setClim([100000, 102500])
      setColormap('cool')
    } else {
      setClim([0, 50])
      setColormap('warm')
    }
  }

  return (
    <>
      <BandSelector
        value={state.variable}
        options={VARIABLES}
        onChange={handleVariableChange}
        label='Variable'
      />
      <Slider
        value={state.time}
        onChange={(v) => setState({ time: v })}
        min={0}
        max={95}
        label='Time'
      />
      {is4D(state.variable) && (
        <Slider
          value={state.level}
          onChange={(v) => setState({ level: v })}
          min={0}
          max={36}
          label='Level'
        />
      )}
    </>
  )
}

const hurricane: Dataset<HurricaneState> = {
  id: 'hurricane_florence',
  source: 'https://atlantis-vis-o.s3-ext.jc.rl.ac.uk/hurricanes/era5/florence',
  variable: 'surface_pressure',
  clim: [100000, 102500],
  colormap: 'cool',
  zarrVersion: 3,
  info: 'Hurricane Florence (single image, 4D)',
  sourceInfo:
    'Single image Zarr v3 with time and level dimensions. Switch between surface pressure and wind velocity.',
  defaultState: {
    variable: 'surface_pressure',
    time: 0,
    level: 15,
  },
  Controls,
  buildLayerProps: (state) => {
    const selector: Selector = is4D(state.variable)
      ? {
          time: { selected: state.time, type: 'index' },
          level: { selected: state.level, type: 'index' },
        }
      : { time: { selected: state.time, type: 'index' } }
    return { selector, variable: state.variable }
  },
}

export default hurricane
