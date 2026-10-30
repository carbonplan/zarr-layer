import proj4 from 'proj4'
import {
  addProjection,
  addProtocol,
  type Map as MapLibreMap,
} from 'maplibre-gl'

/**
 * Antarctic Polar Stereographic (EPSG:3031) as a MapLibre custom projection
 * (maplibre-gl >= 6.15), with NASA GIBS imagery cut in the same CRS as its
 * basemap. GIBS's EPSG:3031 tile matrix set starts with 2x2 tiles over the
 * square below, so MapLibre tile z is GIBS level z - 1.
 */
export const ANTARCTIC_PROJECTION = 'EPSG:3031'
const ANTARCTIC_PROJ4 =
  '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
const GIBS_HALF_EXTENT = 4194304

const BASEMAP_SOURCE = 'antarctic-basemap'
const BASEMAP_LAYER = 'antarctic-basemap'
const GIBS_PROTOCOL = 'gibs3031'
const GIBS_LAYER = 'BlueMarble_ShadedRelief_Bathymetry'
const GIBS_MATRIX_SET = '500m'
const GIBS_MAX_LEVEL = 4

let registered = false

function register() {
  if (registered) return
  registered = true
  const converter = proj4('EPSG:4326', ANTARCTIC_PROJ4)
  addProjection({
    name: ANTARCTIC_PROJECTION,
    projection: {
      forward: (position) => converter.forward(position),
      inverse: (position) => converter.inverse(position),
    },
    tileMatrixSet: {
      origin: [-GIBS_HALF_EXTENT, GIBS_HALF_EXTENT],
      extentAtZoom0: 2 * GIBS_HALF_EXTENT,
    },
    definition: ANTARCTIC_PROJ4,
  })
  addProtocol(GIBS_PROTOCOL, async (params, abortController) => {
    const [z, x, y] = params.url.replace(`${GIBS_PROTOCOL}://`, '').split('/')
    const response = await fetch(
      `https://gibs.earthdata.nasa.gov/wmts/epsg3031/best/${GIBS_LAYER}/default/${GIBS_MATRIX_SET}/${
        Number(z) - 1
      }/${y}/${x}.jpeg`,
      { signal: abortController.signal }
    )
    if (!response.ok)
      throw new Error(`GIBS tile ${params.url}: ${response.status}`)
    return { data: await response.arrayBuffer() }
  })
}

/**
 * Switch a MapLibre map into or out of the Antarctic projection, swapping
 * the Web Mercator vector basemap for the GIBS imagery, which is cut in
 * EPSG:3031. Returns whether the map changed between the two.
 */
export function setAntarctic(map: MapLibreMap, antarctic: boolean): boolean {
  if (antarctic) register()
  if (!map.getSource(BASEMAP_SOURCE)) {
    map.addSource(BASEMAP_SOURCE, {
      type: 'raster',
      tiles: [`${GIBS_PROTOCOL}://{z}/{x}/{y}`],
      tileSize: 512,
      minzoom: 1,
      maxzoom: GIBS_MAX_LEVEL + 1,
      attribution:
        '<a href="https://earthdata.nasa.gov/gibs">NASA GIBS</a>, Blue Marble',
    })
    const firstLayer = map
      .getStyle()
      .layers.find((l) => l.type !== 'background')
    map.addLayer(
      {
        id: BASEMAP_LAYER,
        type: 'raster',
        source: BASEMAP_SOURCE,
        layout: { visibility: 'none' },
      },
      firstLayer?.id
    )
  }
  const wasAntarctic =
    map.getLayoutProperty(BASEMAP_LAYER, 'visibility') === 'visible'
  for (const layer of map.getStyle().layers) {
    if ('source' in layer && layer.source === 'protomaps') {
      map.setLayoutProperty(
        layer.id,
        'visibility',
        antarctic ? 'none' : 'visible'
      )
    }
  }
  map.setLayoutProperty(
    BASEMAP_LAYER,
    'visibility',
    antarctic ? 'visible' : 'none'
  )
  return wasAntarctic !== antarctic
}
