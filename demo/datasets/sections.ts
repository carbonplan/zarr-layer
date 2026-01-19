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
      'polar_antarctic',
      'delta_FG_CO2',
    ],
  },
  {
    label: 'Untiled Multiscale',
    datasetIds: [
      'usgsdem',
      'untiled_2level_4326',
      'sentinel_2_eopf',
      'Burn Probability over CONUS',
    ],
  },
]
