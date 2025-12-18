export type SelectorSection = {
  label: string
  datasetIds: string[]
}

export const SELECTOR_SECTIONS: SelectorSection[] = [
  // {
  //   label: 'Advanced',
  //   datasetIds: ['carbonplan_4d'],
  // },
  {
    label: 'Tiled Pyramids',
    datasetIds: [
      'carbonplan_4d',
      'salinity_v2',
      'temperature_v3',
      'tasmax_pyramid_4326',
      'tasmax_pyramid_v3_4326',
    ],
  },
  {
    label: 'Single Image',
    datasetIds: ['hurricane_florence', 'pr single image', 'delta_FG_CO2'],
  },
  {
    label: 'Untiled Multiscale',
    datasetIds: ['untiled_2level_4326', 'untiled_2level_3857'],
  },
]
