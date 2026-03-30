import React from 'react'
import { Filter, Row, Column } from '@carbonplan/components'
import { Box } from 'theme-ui'
import { subheadingSx } from './styles'

export type BandSelectorProps<T extends string> = {
  value: T
  options: readonly T[]
  onChange: (value: T) => void
  label?: string
}

export function BandSelector<T extends string>({
  value,
  options,
  onChange,
  label = 'Band',
}: BandSelectorProps<T>) {
  const values = options.reduce((acc, opt) => {
    acc[opt] = opt === value
    return acc
  }, {} as Record<T, boolean>)

  return (
    <Row columns={[4, 4, 4, 4]} sx={{ alignItems: 'baseline' }}>
      <Column start={1} width={1}>
        <Box sx={{ ...subheadingSx, my: [1, 1, 1, 2] }}>{label}</Box>
      </Column>
      <Column start={2} width={3}>
        <Filter
          values={values}
          setValues={(obj: Record<string, boolean>) => {
            const selected = options.find((opt) => obj[opt])
            if (selected) onChange(selected)
          }}
        />
      </Column>
    </Row>
  )
}
