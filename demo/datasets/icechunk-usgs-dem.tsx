import { IcechunkStore } from 'icechunk-js'
import type { Dataset, LayerProps } from './types'

const ICECHUNK_URL =
  'https://carbonplan-share.s3.us-west-2.amazonaws.com/zarr-layer-examples/usgs10m_dem_subset_multiscale_icechunk_14lvl.icechunk'

let _store: IcechunkStore | null = null

const icechunkUsgsDem: Dataset<Record<string, never>> = {
  id: 'icechunk_usgs_dem',
  source: ICECHUNK_URL,
  variable: 'DEM',
  clim: [0, 4000],
  colormap: 'warm',
  zarrVersion: 3,
  info: 'USGS 10m DEM (Icechunk, 14-level multiscale)',
  sourceInfo:
    'USGS 10m DEM stored in Icechunk format with 14-level multiscale pyramid.',
  center: [-117, 45],
  zoom: 5,
  get store() {
    return (_store ??= new IcechunkStore(ICECHUNK_URL, {
      branch: 'main',
      formatVersion: 'v1',
    }))
  },
  defaultState: {},
  Controls: () => null,
  buildLayerProps: (): LayerProps => ({ selector: {} }),
}

export default icechunkUsgsDem
