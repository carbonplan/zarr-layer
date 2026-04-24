import React, { useEffect, useRef, useState } from 'react'
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

const useDebouncedCommit = <T,>(value: T, commit: (v: T) => void, ms = 100) => {
  const commitRef = useRef(commit)
  commitRef.current = commit
  useEffect(() => {
    const id = setTimeout(() => commitRef.current(value), ms)
    return () => clearTimeout(id)
  }, [value, ms])
}

const Controls = ({ state, setState }: ControlsProps<HurricaneState>) => {
  const setClim = useAppStore((s) => s.setClim)
  const setColormap = useAppStore((s) => s.setColormap)

  const [time, setTime] = useState(state.time)
  const [level, setLevel] = useState(state.level)
  useDebouncedCommit(time, (v) => setState({ time: v }))
  useDebouncedCommit(level, (v) => setState({ level: v }))

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
      <Slider value={time} onChange={setTime} min={0} max={95} label='Time' />
      {is4D(state.variable) && (
        <Slider
          value={level}
          onChange={setLevel}
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
  info: 'Hurricane Florence (EPSG:4326)',
  sourceInfo:
    'ERA5 Hurricane Florence, Zarr v3 single resolution. EPSG:4326, time and level dimensions.',
  center: [-65, 35],
  zoom: 3,
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
