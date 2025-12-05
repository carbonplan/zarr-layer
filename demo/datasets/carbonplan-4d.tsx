import React from 'react'
// @ts-expect-error - carbonplan components types not available
import { Select, Slider } from '@carbonplan/components'
import { Box } from 'theme-ui'
import { BuildLayerResult, DatasetControlsProps, DatasetModule } from './types'
import { useAppStore } from '../lib/store'

const ALL_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

export const combinedBandsCustomFrag = `
  uniform float u_precipWeight;
  float combined = tavg + prec * u_precipWeight;
  float norm = (combined - clim.x) / (clim.y - clim.x);
  float cla = clamp(norm, 0.0, 1.0);
  vec4 c = texture(colormap, vec2(cla, 0.5));
  fragColor = vec4(c.r, c.g, c.b, opacity);
`

export const monthRangeAverageFrag = `
  uniform float u_monthStart;
  uniform float u_monthEnd;
  float sum = 0.0;
  float count = 0.0;
  ${ALL_MONTHS.map(
    (month) => `
  float inRange_${month} = step(u_monthStart, ${month.toFixed(1)}) * step(${month.toFixed(1)}, u_monthEnd);
  sum += month_${month} * inRange_${month};
  count += inRange_${month};`,
  ).join('')}
  float average = sum / max(count, 1.0);
  float rescaled = (average - clim.x) / (clim.y - clim.x);
  vec4 c = texture(colormap, vec2(clamp(rescaled, 0.0, 1.0), 0.5));
  fragColor = vec4(c.r, c.g, c.b, opacity);
`

export type Carbonplan4dState = {
  band: 'tavg' | 'prec' | 'tavg_range_avg' | 'prec_range_avg' | 'combined'
  month: number
  monthStart: number
  monthEnd: number
  precipWeight: number
}

const buildLayerProps = ({ state }: { state: Carbonplan4dState }) => {
  const isCombined = state.band === 'combined'
  const isRangeAverage =
    state.band === 'tavg_range_avg' || state.band === 'prec_range_avg'
  let baseRangeBand: 'prec' | 'tavg' | null = null
  if (isRangeAverage) {
    baseRangeBand = state.band.startsWith('prec') ? 'prec' : 'tavg'
  } else {
    baseRangeBand = null
  }

  let selector: BuildLayerResult['selector']
  if (isCombined) {
    selector = { band: ['tavg', 'prec'], month: state.month }
  } else if (isRangeAverage && baseRangeBand) {
    selector = { band: baseRangeBand, month: ALL_MONTHS }
  } else {
    selector = { band: state.band, month: state.month }
  }

  const result: BuildLayerResult = {
    selector,
  }

  if (isCombined) {
    result.customFrag = combinedBandsCustomFrag
    result.uniforms = { u_precipWeight: state.precipWeight }
  } else if (isRangeAverage) {
    result.customFrag = monthRangeAverageFrag
    result.uniforms = {
      u_monthStart: state.monthStart,
      u_monthEnd: state.monthEnd,
    }
  }

  return result
}

const Carbonplan4dControls = ({
  state,
  setState,
}: DatasetControlsProps<Carbonplan4dState>) => {
  const setColormap = useAppStore((state) => state.setColormap)
  const setClim = useAppStore((state) => state.setClim)

  const handleBandChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const band = e.target.value as Carbonplan4dState['band']
    setState({ band })
    if (band === 'prec' || band === 'prec_range_avg') {
      setColormap('cool')
      setClim([0, 300])
    } else {
      setColormap('warm')
      setClim([-20, 30])
    }
  }

  const isRangeAverage =
    state.band === 'tavg_range_avg' || state.band === 'prec_range_avg'

  return (
    <>
      <Box>
        <Box>Band</Box>
        <Select value={state.band} onChange={handleBandChange}>
          <option value='tavg'>tavg</option>
          <option value='prec'>prec</option>
          <option value='tavg_range_avg'>
            tavg (custom frag average range)
          </option>
          <option value='prec_range_avg'>
            prec (custom frag average range)
          </option>
          <option value='combined'>combined (custom frag w/ uniform)</option>
        </Select>
        {state.band === 'combined' && (
          <Box
            as='code'
            sx={{ fontSize: 0, color: 'secondary', whiteSpace: 'pre-wrap' }}
          >
            {combinedBandsCustomFrag}
          </Box>
        )}
      </Box>

      {isRangeAverage ? (
        <Box>
          Month range: {state.monthStart} – {state.monthEnd}
          <Box sx={{ position: 'relative', mt: 2 }}>
            <Slider
              min={1}
              max={12}
              step={1}
              value={state.monthStart}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                setState({ monthStart: parseInt(e.target.value) })
              }
            />
            <Slider
              min={1}
              max={12}
              step={1}
              value={state.monthEnd}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                setState({ monthEnd: parseInt(e.target.value) })
              }
              sx={{ mt: 3 }}
            />
          </Box>
        </Box>
      ) : (
        <Box>
          Month: {state.month}
          <Slider
            min={1}
            max={12}
            step={1}
            value={state.month}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setState({ month: parseInt(e.target.value) })
            }
          />
        </Box>
      )}

      {state.band === 'combined' && (
        <Box>
          Precip Weight: {state.precipWeight}
          <Slider
            min={0}
            max={5}
            step={0.1}
            value={state.precipWeight}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setState({ precipWeight: parseFloat(e.target.value) })
            }
          />
        </Box>
      )}
    </>
  )
}

const carbonplan4dDataset: DatasetModule<Carbonplan4dState> = {
  id: 'carbonplan_4d',
  source:
    'https://carbonplan-maps.s3.us-west-2.amazonaws.com/v2/demo/4d/tavg-prec-month',
  variable: 'climate',
  clim: [-20, 30],
  colormap: 'warm',
  zarrVersion: 2,
  info: 'CarbonPlan Climate Demo (4D)',
  sourceInfo: 'Zarr v2 pyramid - temp avg & precipitation by month',
  dimensionNames: {
    lat: 'y',
    lon: 'x',
  },
  defaultState: {
    band: 'tavg',
    month: 1,
    monthStart: 1,
    monthEnd: 6,
    precipWeight: 1.0,
  },
  Controls: Carbonplan4dControls,
  buildLayerProps,
}

export default carbonplan4dDataset
