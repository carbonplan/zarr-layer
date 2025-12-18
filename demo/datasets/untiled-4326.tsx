import React from 'react'
import { Box } from 'theme-ui'
import type { Dataset, ControlsProps, LayerProps } from './types'
import { Slider, BandSelector } from '../components/shared-controls'

const weightedCustomFrag = `
  // Custom fragment shader with weight uniform
  uniform float u_weight;
  float weighted = pr * u_weight;
  float norm = (weighted - clim.x) / (clim.y - clim.x);
  float cla = clamp(norm, 0.0, 1.0);
  vec4 c = texture(colormap, vec2(cla, 0.5));
  fragColor = vec4(c.r, c.g, c.b, opacity);
`

const MODES = ['normal', 'weighted'] as const
type Mode = (typeof MODES)[number]

type State = {
  mode: Mode
  weight: number
}

const Controls = ({ state, setState }: ControlsProps<State>) => {
  return (
    <>
      <BandSelector
        value={state.mode}
        options={MODES}
        onChange={(mode) => setState({ mode })}
        label='Mode'
      />

      {state.mode === 'weighted' && (
        <>
          <Box
            as='code'
            sx={{
              fontSize: 0,
              color: 'secondary',
              whiteSpace: 'pre-wrap',
              display: 'block',
              mb: 3,
            }}
          >
            {weightedCustomFrag}
          </Box>

          <Slider
            value={state.weight}
            onChange={(v) => setState({ weight: v })}
            min={0.1}
            max={3}
            step={0.1}
            label='Weight'
          />
        </>
      )}
    </>
  )
}

const buildLayerProps = (state: State): LayerProps => {
  if (state.mode === 'weighted') {
    return {
      selector: {},
      customFrag: weightedCustomFrag,
      uniforms: { u_weight: state.weight },
    }
  }

  return {
    selector: {},
  }
}

const untiled4326: Dataset<State> = {
  id: 'untiled_2level_4326',
  source:
    'https://carbonplan-share.s3.us-west-2.amazonaws.com/scratch/ndpyramid/2-lvl-test-4326.zarr',
  variable: 'pr',
  clim: [0, 20],
  colormap: 'blues',
  zarrVersion: 3,
  info: 'Untiled 2-level (EPSG:4326)',
  sourceInfo:
    'zarr-conventions/multiscales format with custom shader support. Loads different resolutions based on current zoom and requests chunks based on current viewport.',
  defaultState: {
    mode: 'normal',
    weight: 1.0,
  },
  Controls,
  buildLayerProps,
}

export default untiled4326
