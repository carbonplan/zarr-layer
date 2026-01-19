/**
 * Polar Antarctic Dataset (EPSG:3031)
 *
 * Subset of Antarctica InSync ice velocity data covering Thwaites Glacier.
 * Source: https://discourse-earthcode.eox.at/t/antartica-insync-data-cubes/107
 */

import { createSimpleDataset } from './simple'

const POLAR_SOURCE =
  'https://carbonplan-share.s3.us-west-2.amazonaws.com/zarr-layer-examples/polar-subset.zarr'

const polar = createSimpleDataset({
  id: 'polar_antarctic',
  source: POLAR_SOURCE,
  variable: 'velocity',
  clim: [0, 1.5],
  colormap: 'cool',
  zarrVersion: 2,
  info: 'Antarctic Ice Velocity (EPSG:3031 Polar Stereographic)',
  sourceInfo:
    'Antarctica InSync ice velocity. 100m resolution subset. EPSG:3031 Antarctic Polar Stereographic projection.',
  proj4:
    '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',
  bounds: [-1720900, -983400, -1147500, -492000],
  latIsAscending: true,
  spatialDimensions: { lat: 'y', lon: 'x' },
  center: [-117, -76.5],
  zoom: 4,
})

export default polar
