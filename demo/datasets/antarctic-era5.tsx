import { createSimpleDataset } from './simple'

const antarcticEra5 = createSimpleDataset({
  id: 'antarctic_era5',
  source: 'https://carbonplan-share.s3.us-west-2.amazonaws.com/zarr-layer-examples/antarctic_era5.zarr',
  variable: 'wind_speed',
  clim: [0, 25],
  colormap: 'cool',
  zarrVersion: 3,
  info: 'Antarctic ERA5 (EPSG:3031)',
  sourceInfo:
    'ERA5 Reanalysis wind speed in EPSG:3031 (Antarctic Polar Stereographic)',
  proj4:
    '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',
  center: [0, -80],
  zoom: 0.1,
})

export default antarcticEra5
