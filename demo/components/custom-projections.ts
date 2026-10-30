import proj4 from 'proj4'
import {
  addProjection,
  importScriptInWorkers,
  type CRS,
  type Map as MapLibreMap,
} from 'maplibre-gl'
import { isCustomProjection, type CustomProjectionId } from '../lib/projections'

/**
 * Map projections registered with MapLibre's experimental `addProjection`
 * (maplibre-gl >= 6.15), over Natural Earth land and borders drawn as GeoJSON.
 */
interface CustomProjection {
  crs: CRS
  view: { center: [number, number]; zoom: number }
}

const POLAR_HALF_EXTENT = 4194304

const polar = (
  name: string,
  definition: string,
  center: [number, number]
): CustomProjection => ({
  crs: {
    name,
    projection: proj4('EPSG:4326', definition),
    tileMatrixSet: {
      origin: [-POLAR_HALF_EXTENT, POLAR_HALF_EXTENT],
      extentAtZoom0: 2 * POLAR_HALF_EXTENT,
    },
    definition,
  },
  view: { center, zoom: 1.2 },
})

const LONGLAT = '+proj=longlat +datum=WGS84 +no_defs'
const EQUAL_EARTH = '+proj=eqearth +lon_0=0 +datum=WGS84 +units=m +no_defs'
const EQUAL_EARTH_HALF_WIDTH = 17243960
const CONUS_ALBERS =
  '+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs'

export const CUSTOM_PROJECTIONS: Record<CustomProjectionId, CustomProjection> =
  {
    antarctic: polar(
      'EPSG:3031',
      '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',
      [0, -90]
    ),
    arctic: polar(
      'EPSG:3413',
      '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs',
      [0, 90]
    ),
    'plate carrée': {
      crs: {
        name: 'EPSG:4326',
        projection: proj4('EPSG:4326', LONGLAT),
        tileMatrixSet: { origin: [-180, 180], extentAtZoom0: 360 },
        definition: LONGLAT,
      },
      view: { center: [0, 0], zoom: 0.5 },
    },
    'equal earth': {
      crs: {
        name: 'ESRI:54035',
        projection: proj4('EPSG:4326', EQUAL_EARTH),
        tileMatrixSet: {
          origin: [-EQUAL_EARTH_HALF_WIDTH, EQUAL_EARTH_HALF_WIDTH],
          extentAtZoom0: 2 * EQUAL_EARTH_HALF_WIDTH,
        },
        definition: EQUAL_EARTH,
      },
      view: { center: [0, 0], zoom: 0.7 },
    },
    'conus albers': {
      crs: {
        name: 'EPSG:5070',
        projection: proj4('EPSG:4326', CONUS_ALBERS),
        tileMatrixSet: { origin: [-3000000, 4750000], extentAtZoom0: 6000000 },
        definition: CONUS_ALBERS,
      },
      view: { center: [-97, 38], zoom: 1 },
    },
  }

/** Passed to the layer's `mapProjections`, the same objects `addProjection` gets. */
export const MAP_PROJECTIONS: CRS[] = Object.values(CUSTOM_PROJECTIONS).map(
  (p) => p.crs
)

const BASEMAP_PREFIX = 'custom-basemap'
const NATURAL_EARTH =
  'https://cdn.jsdelivr.net/gh/nvkelso/natural-earth-vector@v5.1.2/geojson'
const LAND_COLOR = '#2a2e35'
const BORDER_COLOR = '#5b6069'

let registered = false
let projectionHandler: Promise<void> | null = null
/** Bumped on each projection change, so a basemap still loading can tell it is stale. */
const basemapGenerations = new WeakMap<MapLibreMap, number>()

function register() {
  if (registered) return
  registered = true
  for (const { crs } of Object.values(CUSTOM_PROJECTIONS)) addProjection(crs)
}

/** GeoJSON sources project in the workers, which build proj4 from `definition`. */
function loadProjectionHandler(): Promise<void> {
  projectionHandler ??= importScriptInWorkers(
    new URL('/projection-handler.mjs', window.location.href).href
  ).then(
    () => undefined,
    (error) => {
      // Let the next projection change try again.
      projectionHandler = null
      throw error
    }
  )
  return projectionHandler
}

function removeBasemap(map: MapLibreMap) {
  const style = map.getStyle()
  for (const layer of style.layers) {
    if (layer.id.startsWith(BASEMAP_PREFIX)) map.removeLayer(layer.id)
  }
  for (const id of Object.keys(style.sources)) {
    if (id.startsWith(BASEMAP_PREFIX)) map.removeSource(id)
  }
}

/** The layer under everything but the background, and the first label. */
function anchors(map: MapLibreMap) {
  const { layers } = map.getStyle()
  return {
    bottom: layers.find((l) => l.type !== 'background')?.id,
    labels: layers.find((l) => l.type === 'symbol')?.id,
  }
}

function addBasemap(map: MapLibreMap, generation: number) {
  loadProjectionHandler().then(
    () => {
      // The map may have moved on, or gone, while the workers loaded it.
      if (basemapGenerations.get(map) !== generation) return
      // Borders draw over the data, under the labels.
      const { bottom, labels } = anchors(map)
      const attribution =
        '<a href="https://www.naturalearthdata.com">Natural Earth</a>'
      map.addSource(`${BASEMAP_PREFIX}-land`, {
        type: 'geojson',
        data: `${NATURAL_EARTH}/ne_50m_land.geojson`,
        attribution,
      })
      // The land polygons are cut at 180°, which would show at the poles.
      map.addSource(`${BASEMAP_PREFIX}-coast`, {
        type: 'geojson',
        data: `${NATURAL_EARTH}/ne_50m_coastline.geojson`,
      })
      map.addSource(`${BASEMAP_PREFIX}-borders`, {
        type: 'geojson',
        data: `${NATURAL_EARTH}/ne_50m_admin_0_boundary_lines_land.geojson`,
      })
      map.addLayer(
        {
          id: `${BASEMAP_PREFIX}-land`,
          type: 'fill',
          source: `${BASEMAP_PREFIX}-land`,
          paint: { 'fill-color': LAND_COLOR },
        },
        bottom
      )
      map.addLayer(
        {
          id: `${BASEMAP_PREFIX}-coast`,
          type: 'line',
          source: `${BASEMAP_PREFIX}-coast`,
          paint: { 'line-color': BORDER_COLOR, 'line-width': 0.75 },
        },
        labels
      )
      map.addLayer(
        {
          id: `${BASEMAP_PREFIX}-borders`,
          type: 'line',
          source: `${BASEMAP_PREFIX}-borders`,
          paint: { 'line-color': BORDER_COLOR, 'line-width': 0.5 },
        },
        labels
      )
    },
    (error) =>
      console.warn(
        '[demo] The GeoJSON basemap could not load its projection handler:',
        error
      )
  )
}

/**
 * Put a MapLibre map in one of these projections, or a built-in one, with
 * the basemap to match: the Web Mercator vector basemap shows only in the
 * built-in projections.
 */
export function setMapProjection(
  map: MapLibreMap,
  projection: 'globe' | 'mercator' | CustomProjectionId
) {
  const custom = isCustomProjection(projection)
    ? CUSTOM_PROJECTIONS[projection]
    : null
  if (custom) register()
  if (!basemapGenerations.has(map)) {
    map.once('remove', () => basemapGenerations.delete(map))
  }
  const generation = (basemapGenerations.get(map) ?? 0) + 1
  basemapGenerations.set(map, generation)
  const previous = map.getProjection()?.type
  const next = custom ? custom.crs.name : projection
  removeBasemap(map)
  for (const layer of map.getStyle().layers) {
    if ('source' in layer && layer.source === 'protomaps') {
      map.setLayoutProperty(layer.id, 'visibility', custom ? 'none' : 'visible')
    }
  }
  map.setProjection({ type: next })
  if (custom) addBasemap(map, generation)
  const wasCustom = MAP_PROJECTIONS.some((crs) => crs.name === previous)
  if (custom && previous !== next) map.jumpTo(custom.view)
  else if (!custom && wasCustom) map.jumpTo({ center: [0, 20], zoom: 2 })
}
