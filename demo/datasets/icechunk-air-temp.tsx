import { IcechunkStore } from '@carbonplan/icechunk-js'
import type { Dataset, LayerProps } from './types'

const ICECHUNK_URL =
  'https://carbonplan-share.s3.us-west-2.amazonaws.com/zarr-layer-examples/virtual_icechunk/air_temperature.icechunk'

let _storePromise: Promise<IcechunkStore> | null = null

const icechunkAirTemp: Dataset<Record<string, never>> = {
  id: 'icechunk_air_temp',
  source: ICECHUNK_URL,
  variable: 'air',
  clim: [185, 322],
  colormap: 'warm',
  zarrVersion: 3,
  info: 'Air temperature (Icechunk)',
  sourceInfo: 'Air temperature stored in an Icechunk store',
  center: [-100, 45],
  zoom: 2,
  get store() {
    return (_storePromise ??= IcechunkStore.open(ICECHUNK_URL, {
      branch: 'main',
      formatVersion: 'v1',
    }).catch((err) => {
      _storePromise = null
      throw err
    }))
  },
  defaultState: {},
  Controls: () => null,
  buildLayerProps: (): LayerProps => ({
    selector: { time: { selected: 0, type: 'index' } },
  }),
}

export default icechunkAirTemp
