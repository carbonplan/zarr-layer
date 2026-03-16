import React from 'react'
// @ts-expect-error - carbonplan components types not available
import { Slider, Row, Column } from '@carbonplan/components'
import { Box } from 'theme-ui'
import type { Theme } from 'theme-ui'
import { subheadingSx } from './styles'

export type RangeSliderProps = {
  startValue: number
  endValue: number
  onStartChange: (value: number) => void
  onEndChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  label?: string
}

export const RangeSlider: React.FC<RangeSliderProps> = ({
  startValue,
  endValue,
  onStartChange,
  onEndChange,
  min = 0,
  max = 10,
  step = 1,
  label = 'Range',
}) => {
  const range = max - min

  const sliderThumbSx = {
    height: [22, 18, 16],
    width: [22, 18, 16],
    boxShadow: ({ colors }: Theme) => `0 0 0 0px ${colors?.secondary}`,
    transition: 'box-shadow .15s ease',
    pointerEvents: 'auto' as const,
    zIndex: 3,
  }

  const focusSx = {
    color: 'primary',
    '&::-webkit-slider-thumb': {
      boxShadow: ({ colors }: Theme) => `0 0 0 4px ${colors?.secondary}`,
    },
    '&::-moz-range-thumb': {
      boxShadow: ({ colors }: Theme) => `0 0 0 4px ${colors?.secondary}`,
    },
  }

  return (
    <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline' }}>
      <Column start={1} width={1}>
        <Box sx={subheadingSx}>{label}</Box>
      </Column>
      <Column start={2} width={3} sx={{ position: 'relative' }}>
        {/* Visual highlight bar showing selected range */}
        <Box
          sx={{
            mt: ['8px'],
            bg: 'primary',
            position: 'absolute',
            top: 0,
            left: `calc((${startValue - min}) * 100% / ${range})`,
            opacity: 1,
            width: `calc((${endValue - startValue}) * 100% / ${range})`,
            height: '4px',
            zIndex: 2,
            pointerEvents: 'none',
          }}
        />
        {/* Start slider */}
        <Slider
          sx={{
            color: 'primary',
            position: 'absolute',
            left: 0,
            top: 0,
            pointerEvents: 'none',
            ':focus': focusSx,
            '&::-webkit-slider-thumb': sliderThumbSx,
            '&::-moz-range-thumb': sliderThumbSx,
          }}
          min={min}
          max={max}
          step={step}
          value={startValue}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            onStartChange(Math.min(parseFloat(e.target.value), endValue))
          }
        />
        {/* End slider (overlaid) */}
        <Slider
          sx={{
            color: 'primary',
            position: 'absolute',
            left: 0,
            top: 0,
            pointerEvents: 'none',
            bg: 'transparent',
            ':focus': { ...focusSx, bg: 'transparent' },
            ':focus-visible': {
              outline: 'none !important',
              background: 'transparent !important',
            },
            '&::-webkit-slider-thumb': sliderThumbSx,
            '&::-moz-range-thumb': sliderThumbSx,
          }}
          min={min}
          max={max}
          step={step}
          value={endValue}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            onEndChange(Math.max(parseFloat(e.target.value), startValue))
          }
        />
      </Column>
      <Column start={2} width={3}>
        <Box sx={{ width: '100%', textAlign: 'center' }}>
          {startValue} - {endValue}
        </Box>
      </Column>
    </Row>
  )
}
