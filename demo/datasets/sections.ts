export type SelectorSection = {
  label: string
  description: string
  datasetIds: string[]
}

export const SELECTOR_SECTIONS: SelectorSection[] = [
  {
    label: 'Multiscale',
    description:
      'Multiscale Zarr stores. Uses the zarr-conventions/multiscales format. See @carbonplan/topozarr for creation.',
    datasetIds: ['usgsdem', 'sentinel_2_eopf', 'Burn Probability over CONUS'],
  },
  {
    label: 'Single Resolution',
    description: 'Single-resolution datasets. Reprojected if needed.',
    datasetIds: [
      'hrrr_weather',
      'hurricane_florence',
      'polar_antarctic',
      'antarctic_era5',
      'delta_FG_CO2',
    ],
  },
  {
    label: 'Icechunk',
    description:
      'Datasets served from Icechunk, a transactional storage engine for Zarr that supports virtual datasets via VirtualiZarr. Uses @carbonplan/icechunk-js reader.',
    datasetIds: ['icechunk_prec'],
  },
  {
    label: 'Legacy Tiled Pyramids',
    description:
      'Legacy format. Zarr stores resampled and rechunked to follow slippy-map tile pyramid conventions (zxy). See @carbonplan/ndpyramid for creation.',
    datasetIds: ['carbonplan_4d', 'temperature_v3', 'tasmax_pyramid_4326'],
  },
]
