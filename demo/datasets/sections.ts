export type SelectorSection = {
  label: string
  datasetIds: string[]
}

export const SELECTOR_SECTIONS: SelectorSection[] = [
  {
    label: 'Tiled Pyramids',
    datasetIds: ['carbonplan_4d', 'temperature_v3', 'tasmax_pyramid_4326'],
  },
  {
    label: 'Single Image',
    datasetIds: [
      'hrrr_weather',
      'hurricane_florence',
      'pr single image',
      'delta_FG_CO2',
    ],
  },
  {
    label: 'Untiled Multiscale',
    datasetIds: ['untiled_2level_4326', 'Burn Probability over CONUS'],
  },
]
