import React from 'react'
import { Slider as CarbonSlider, Row, Column } from '@carbonplan/components'
import { Box, Flex } from 'theme-ui'
import { subheadingSx } from './styles'

export type SliderProps = {
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  label?: string
}

export const Slider: React.FC<SliderProps> = ({
  value,
  onChange,
  min = 0,
  max = 10,
  step = 1,
  label = 'Value',
}) => {
  return (
    <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline', mt: 2 }}>
      <Column start={1} width={1}>
        <Box sx={subheadingSx}>{label}</Box>
      </Column>
      <Column start={2} width={3}>
        <Flex sx={{ flexDirection: 'column' }}>
          <CarbonSlider
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              onChange(parseFloat(e.target.value))
            }
          />
          <Box sx={{ textAlign: 'center' }}>{value}</Box>
        </Flex>
      </Column>
    </Row>
  )
}
