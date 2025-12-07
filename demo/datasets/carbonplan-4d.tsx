import React from 'react'
// @ts-expect-error - carbonplan components types not available
import { Filter, Slider, Row, Column } from '@carbonplan/components'
import { Box, Flex } from 'theme-ui'
import type { Theme } from 'theme-ui'
import { BuildLayerResult, DatasetControlsProps, DatasetModule } from './types'
import { useAppStore } from '../lib/store'

const ALL_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

const subheadingSx = {
  mt: 3,
  mb: 1,
  fontFamily: 'mono',
  letterSpacing: 'smallcaps',
  textTransform: 'uppercase' as const,
  fontSize: [1, 1, 1, 2],
  color: 'secondary',
}

export const combinedBandsCustomFrag = `
  // custom fragment shader w/ uniform example
  uniform float u_precipWeight;
  float combined = prec * u_precipWeight;
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
  band: 'tavg' | 'prec' | 'tavg_range' | 'prec_range' | 'weighted'
  month: number
  monthStart: number
  monthEnd: number
  precipWeight: number
}

const buildLayerProps = ({ state }: { state: Carbonplan4dState }) => {
  const isCombined = state.band === 'weighted'
  const isRangeAverage =
    state.band === 'tavg_range' || state.band === 'prec_range'
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

  const handleBandChange = (band: Carbonplan4dState['band']) => {
    setState({ band })
    if (band === 'prec' || band === 'prec_range') {
      setColormap('cool')
      setClim([0, 300])
    } else {
      setColormap('warm')
      setClim([-20, 30])
    }
  }

  const isRangeAverage =
    state.band === 'tavg_range' || state.band === 'prec_range'

  return (
    <>
      <Row columns={[4, 4, 4, 4]} sx={{ mb: 3, alignItems: 'baseline' }}>
        <Column start={1} width={1}>
          <Box sx={subheadingSx}>Selector</Box>
        </Column>
        <Column start={2} width={3}>
          <Filter
            values={{
              tavg: state.band === 'tavg',
              prec: state.band === 'prec',
              tavg_range: state.band === 'tavg_range',
              prec_range: state.band === 'prec_range',
              weighted: state.band === 'weighted',
            }}
            setValues={(obj: Record<string, boolean>) => {
              if (obj.tavg) handleBandChange('tavg')
              if (obj.prec) handleBandChange('prec')
              if (obj.tavg_range) handleBandChange('tavg_range')
              if (obj.prec_range) handleBandChange('prec_range')
              if (obj.weighted) handleBandChange('weighted')
            }}
          />
        </Column>
      </Row>
      {state.band === 'weighted' && (
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
          {combinedBandsCustomFrag}
        </Box>
      )}

      {isRangeAverage ? (
        <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline', mb: 3 }}>
          <Column start={1} width={1}>
            <Box sx={subheadingSx}>Month range</Box>
          </Column>
          <Column start={2} width={3} sx={{ position: 'relative' }}>
            <Box
              sx={{
                mt: ['8px'],
                bg: 'primary',
                position: 'absolute',
                top: 0,
                left: `calc((${state.monthStart - 1}) * 100% / 11)`,
                opacity: 1,
                width: `calc((${state.monthEnd - state.monthStart}) * 100% / 11)`,
                height: '4px',
                zIndex: 2,
                pointerEvents: 'none',
              }}
            />
            <Slider
              sx={{
                color: 'primary',
                position: 'absolute',
                left: 0,
                top: 0,
                pointerEvents: 'none',
                ':focus': {
                  color: 'primary',
                  '&::-webkit-slider-thumb': {
                    boxShadow: ({ colors }: Theme) =>
                      `0 0 0 4px ${colors?.secondary}`,
                  },
                  '&::-moz-range-thumb': {
                    boxShadow: ({ colors }: Theme) =>
                      `0 0 0 4px ${colors?.secondary}`,
                  },
                },
                '&::-webkit-slider-thumb': {
                  height: [22, 18, 16],
                  width: [22, 18, 16],
                  boxShadow: ({ colors }: Theme) =>
                    `0 0 0 0px ${colors?.secondary}`,
                  transition: 'box-shadow .15s ease',
                  pointerEvents: 'auto',
                  zIndex: 3,
                },
                '&::-moz-range-thumb': {
                  height: [22, 18, 16],
                  width: [22, 18, 16],
                  boxShadow: ({ colors }: Theme) =>
                    `0 0 0 0px ${colors?.secondary}`,
                  transition: 'box-shadow .15s ease',
                  pointerEvents: 'auto',
                  zIndex: 3,
                },
              }}
              min={1}
              max={12}
              step={1}
              value={state.monthStart}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                setState({
                  monthStart: Math.min(
                    parseFloat(e.target.value),
                    state.monthEnd,
                  ),
                })
              }
            />
            <Slider
              sx={{
                color: 'primary',
                position: 'absolute',
                left: 0,
                top: 0,
                pointerEvents: 'none',
                bg: 'transparent',
                ':focus': {
                  color: 'primary',
                  bg: 'transparent',
                  '&::-webkit-slider-thumb': {
                    boxShadow: ({ colors }: Theme) =>
                      `0 0 0 4px ${colors?.secondary}`,
                  },
                  '&::-moz-range-thumb': {
                    boxShadow: ({ colors }: Theme) =>
                      `0 0 0 4px ${colors?.secondary}`,
                  },
                },
                ':focus-visible': {
                  outline: 'none !important',
                  background: `transparent !important`,
                },
                '&::-webkit-slider-thumb': {
                  height: [22, 18, 16],
                  width: [22, 18, 16],
                  boxShadow: ({ colors }: Theme) =>
                    `0 0 0 0px ${colors?.secondary}`,
                  transition: 'box-shadow .15s ease',
                  pointerEvents: 'auto',
                  zIndex: 3,
                },
                '&::-moz-range-thumb': {
                  height: [22, 18, 16],
                  width: [22, 18, 16],
                  boxShadow: ({ colors }: Theme) =>
                    `0 0 0 0px ${colors?.secondary}`,
                  transition: 'box-shadow .15s ease',
                  pointerEvents: 'auto',
                  zIndex: 3,
                },
              }}
              min={1}
              max={12}
              step={1}
              value={state.monthEnd}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                setState({
                  monthEnd: Math.max(
                    parseFloat(e.target.value),
                    state.monthStart,
                  ),
                })
              }
            />
          </Column>
          <Column start={2} width={3}>
            <Box sx={{ width: '100%', textAlign: 'center' }}>
              {state.monthStart} - {state.monthEnd}
            </Box>
          </Column>
        </Row>
      ) : (
        <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline', mb: 3 }}>
          <Column start={1} width={1}>
            <Box sx={subheadingSx}>Month</Box>
          </Column>
          <Column start={2} width={3}>
            <Flex sx={{ flexDirection: 'column' }}>
              <Slider
                min={1}
                max={12}
                step={1}
                value={state.month}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                  setState({ month: parseInt(e.target.value) })
                }
              />
              <Box sx={{ textAlign: 'center' }}>{state.month}</Box>
            </Flex>
          </Column>
        </Row>
      )}

      {state.band === 'weighted' && (
        <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline', mt: 2 }}>
          <Column start={1} width={1}>
            <Box sx={subheadingSx}>Precip weight</Box>
          </Column>
          <Column start={2} width={3}>
            <Flex sx={{ flexDirection: 'column' }}>
              <Slider
                min={0}
                max={5}
                step={0.1}
                value={state.precipWeight}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                  setState({ precipWeight: parseFloat(e.target.value) })
                }
              />
              <Box sx={{ textAlign: 'center' }}>{state.precipWeight}</Box>
            </Flex>
          </Column>
        </Row>
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
  info: '4d pyramid, temp/precip by month',
  sourceInfo:
    'Zarr v2 pyramid. Select different bands for demonstrations of custom fragment shaders and uniform variables.',
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
