# @carbonplan/zarr-layer

![NPM Version](https://img.shields.io/npm/v/@carbonplan/zarr-layer)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Custom layer for rendering Zarr datasets in MapLibre or Mapbox GL, inspired (and borrowing significant code and concepts from)
[zarr-gl](https://github.com/carderne/zarr-gl), [zarr-cesium](https://github.com/NOC-OI/zarr-cesium), [@carbonplan/maps](https://github.com/carbonplan/maps), and [deck-gl-raster](https://github.com/developmentseed/deck.gl-raster). Uses [CustomLayerInterface](https://maplibre.org/maplibre-gl-js/docs/API/interfaces/CustomLayerInterface/) to render data directly to the map and supports rendering to globe and mercator projections for both MapLibre and Mapbox. Input data are reprojected on the fly.

This is an active experiment so expect to run into some bugs! Please report them.

## demo

See the [demo](https://zarr-layer.demo.carbonplan.org/) for a quick tour of capabilities. Code for the demo is in the `/demo` folder.

## data requirements

Supports v2 and v3 zarr stores via [zarrita](https://github.com/manzt/zarrita.js). Arbitrary CRS support via [proj4](https://github.com/proj4js/proj4js) reprojection for 'untiled' data. Tiled data need to be in EPSG:4326 or EPSG:3857.

### Multiscales

High resolution datasets require multiscales. There are two main ways to store these. We recommend the untiled path for ease of data creation. Performance appears similar between the two options in most cases.

#### Tiled

Classic approach that requires reprojection to either web mercator or WGS84. Breaks data down into chunks that correspond exactly to web map slippy map tile conventions (XYZ). See [ndpyramid](https://github.com/carbonplan/ndpyramid). Data creation for this format can be resource intensive, see the untiled section below for an easier alternative.

- Limited to EPSG:4326 or EPSG:3857

#### Untiled

Support for the emerging [multiscales](https://github.com/zarr-conventions/multiscales) convention (non-slippy map conforming) is experimental! We also try to interpret other multiscale formats. Chunks are loaded based on viewport intersection and zoom level.

- Supports client side CRS reprojection via `proj4` and `@developmentseed/raster-reproject`

See [topozarr](https://github.com/norlandrhagen/topozarr) for a look at how to create these datasets.

## install

```bash
npm install @carbonplan/zarr-layer
```

## build locally

```bash
npm install
npm run build
```

## usage

```ts
import maplibregl from 'maplibre-gl' // or mapbox
import { ZarrLayer } from '@carbonplan/zarr-layer'

const map = new maplibregl.Map({container: 'map'})
const layer = new ZarrLayer({
  id: 'zarr-layer',
  source: 'https://example.com/my.zarr',
  variable: 'temperature',
  clim: [270, 310],
  colormap: ['#000000', '#ffffff', ...],
  selector: { month: 1 },
})
map.on('load', () => {
  map.addLayer(layer)
  // optionally add before id to slot data into map layer stack.
  // map.addLayer(layer, 'beforeID')
})
```

## options

**Required:**
| Option | Type | Description |
|--------|------|-------------|
| id | string | Unique layer identifier |
| source | string | Zarr store URL |
| variable | string | Variable name to render |
| colormap | array | Array of hex strings or `[r,g,b]` values |
| clim | [min, max] | Color scale limits |

**Optional:**
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| selector | object | `{}` | Dimension selector (unspecified dims default to index 0) |
| opacity | number | `1` | Layer opacity (0-1) |
| zarrVersion | `2` \| `3` | auto | Zarr format version (tries v3 first, falls back to v2) |
| minzoom | number | `0` | Minimum zoom level for rendering |
| maxzoom | number | `Infinity` | Maximum zoom level for rendering |
| fillValue | number | auto | No-data value (from metadata if not set) |
| spatialDimensions | object | auto | Custom `{ lat, lon }` dim names |
| crs | string | auto | CRS identifier for built-in projections (`EPSG:4326` or `EPSG:3857`). For other CRS, use `proj4`. |
| proj4 | string | - | Proj4 definition string for CRS reprojection (`bounds` recommended, else derived from coordinates) |
| bounds | array | auto | `[xMin, yMin, xMax, yMax]` in source CRS units (degrees for EPSG:4326, meters for EPSG:3857). These are interpreted as edge bounds (not center-to-center) |
| latIsAscending | boolean | auto | Latitude orientation |
| renderingMode | `'2d'` \| `'3d'` | `'3d'` | Custom layer rendering mode |
| customFrag | string | - | Custom fragment shader |
| uniforms | object | - | Shader uniform values (requires `customFrag`) |
| onLoadingStateChange | function | - | Loading state callback |
| throttleMs | number | `100` | Throttle interval (ms) for data fetching during rapid selector changes. Set to `0` to disable. |
| transformRequest | function | - | Transform request URLs and add headers/credentials (see [authentication](#authentication)) |

## methods

```ts
layer.setOpacity(0.8)
layer.setClim([0, 100])
layer.setColormap(['#000', '#fff'])
layer.setSelector({ time: 5 })
layer.setVariable('precipitation') // async - reloads metadata
layer.setUniforms({ u_weight: 1.5 }) // no-op unless layer has customFrag
```

## selectors

Selectors specify which slice of your multidimensional data to render. Dimensions not specified default to index 0.

**Basic syntax:**

```ts
// Simple value - matches exact value in coordinate array
{ time: 5 }
{ time: '2024-01-15' }

// Explicit index - uses array index directly (no coordinate lookup)
{ time: { selected: 5, type: 'index' } }

// Explicit value - same as simple syntax, matches exact value
{ time: { selected: 5, type: 'value' } }
```

**Multi-band selection (for custom shaders):**

```ts
// String values use the string directly as the shader variable name
{ band: ['tavg', 'prec'] }
// exposes as: tavg, prec

// Numeric values are prefixed with the dimension key (required for valid GLSL identifiers)
{ month: [1, 2, 3] }
// exposes as: month_1, month_2, month_3

// Mix with other dimensions
{ band: ['red', 'green', 'blue'], time: 0 }
```

**Query-specific selectors:**

```ts
// Array of values for time series queries
const result = await layer.queryData(
  { type: 'Point', coordinates: [lng, lat] },
  { time: [0, 1, 2, 3, 4] } // returns data for all 5 time steps
)
```

**Type options:**

| Type                | Behavior                                                      |
| ------------------- | ------------------------------------------------------------- |
| `'value'` (default) | Matches exact value in coordinate array (throws if not found) |
| `'index'`           | Uses value directly as array index                            |

## custom shaders and uniforms

Custom fragment shaders let you do math on your data to change how it's displayed. This can be useful for things like log scales, combining bands, or aggregating data over a time window. In tiled mode, data for all selected bands must be in the same chunk. In untiled mode, bands can span separate chunks — each band is fetched in parallel and combined for rendering. You can pass in `uniforms` to allow user interaction to influence the custom shader code.

```ts
new ZarrLayer({
  // ...
  customFrag: `
    uniform float u_weight;
    float val = band_a * u_weight;
    float norm = (val - clim.x) / (clim.y - clim.x);
    vec4 c = texture(colormap, vec2(clamp(norm, 0.0, 1.0), 0.5));
    fragColor = vec4(c.rgb, opacity);
  `,
  uniforms: { u_weight: 1.0 },
})
```

## band math helpers

For common multi-band operations, we provide helper functions that return ready-to-use configuration objects. These include the `selector`, `customFrag`, and `clim` settings needed for the calculation. The default `clim` values can be overridden by setting `clim` after spreading the config.

### NDVI (Normalized Difference Vegetation Index)

```ts
import { ZarrLayer, ndvi } from '@carbonplan/zarr-layer'

const config = ndvi({ nir: 'B08', red: 'B04' })
// Returns: { selector: { band: ['B08', 'B04'] }, customFrag: '...', clim: [-1, 1] }

new ZarrLayer({
  source: 'https://example.com/sentinel2.zarr',
  variable: 'data',
  colormap: 'rdylgn',
  ...config,
  // Override clim to focus on vegetation range:
  clim: [0, 0.8],
  // Merge additional selector dimensions as needed:
  selector: { ...config.selector, time: 0 },
})
```

### True Color RGB

```ts
import { ZarrLayer, trueColor } from '@carbonplan/zarr-layer'

const config = trueColor({ red: 'B04', green: 'B03', blue: 'B02' })
// Returns: { selector: { band: ['B04', 'B03', 'B02'] }, customFrag: '...', clim: [0, 1] }

new ZarrLayer({
  source: 'https://example.com/sentinel2.zarr',
  variable: 'data',
  ...config,
  // For raw reflectance data (e.g., Sentinel-2 0-10000), override clim:
  clim: [0, 3000],
})
```

### Options

Both helpers accept a `dimension` parameter (default: `'band'`) to specify which dimension contains the band values:

```ts
ndvi({ nir: 'B08', red: 'B04', dimension: 'wavelength' })
trueColor({ red: 'B04', green: 'B03', blue: 'B02', dimension: 'wavelength' })
```

## custom projections

For datasets in non-standard projections (e.g., Lambert Conformal Conic, UTM), provide a `proj4` definition string. Specifying `bounds` in source CRS units is recommended for performance (otherwise derived from coordinate arrays). If you set `crs` to a non-`EPSG:4326`/`EPSG:3857` value without `proj4`, the renderer will warn and fall back to inferred CRS.

```ts
new ZarrLayer({
  // ...
  spatialDimensions: {
    lat: 'projection_y_coordinate',
    lon: 'projection_x_coordinate',
  },
  proj4:
    '+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs',
  bounds: [-2697520, -1587306, 2697480, 1586694], // recommended: edge bounds [xMin, yMin, xMax, yMax] in source CRS units
})
```

The data will be reprojected to Web Mercator for display using GPU-accelerated mesh reprojection powered by [@developmentseed/raster-reproject](https://github.com/developmentseed/deck.gl-raster). Find proj4 strings at [epsg.io](https://epsg.io/) or in your dataset's metadata.

## queries

Supports `Point`, `Polygon`, and `MultiPolygon` geometries in geojson format. You can optionally pass in a custom `selector` to override the visualization `selector`.

```ts
// Point query
const result = await layer.queryData(
  { type: 'Point', coordinates: [lng, lat] },
  // optional selector override (useful for e.g. time series creation)
  { time: [0, 1, 2] }
)

// Polygon query
const result = await layer.queryData({
  type: 'Polygon',
  coordinates: [[...]],
})

// Returns:
// {
//   [variable]: number[],
//   dimensions: ['lat', 'lon'],
//   coordinates: { lat: number[], lon: number[] }
// }
```

**Note:** Query results match rendered values (`scale_factor`/`add_offset` applied, `fillValue`/NaN filtered). For datasets rendered via `proj4` reprojection, queries sample the underlying source grid; because reprojection/resampling occurs for display, a visual pixel click may not align perfectly with the nearest source pixel.

## authentication

Use `transformRequest` to add headers or credentials to requests. The function receives the fully resolved URL for each request, enabling per-path authentication like presigned S3 URLs. Supports any [fetch options](https://developer.mozilla.org/en-US/docs/Web/API/fetch#options).

```ts
// Static auth (same headers for all requests)
transformRequest: (url) => ({
  url,
  headers: { Authorization: `Bearer ${token}` },
})

// Presigned URLs (path-specific signatures)
transformRequest: async (url) => ({
  url: await getPresignedUrl(url),
})
```

## thanks

This experiment is only possible following in the footsteps of other work in this space. [zarr-gl](https://github.com/carderne/zarr-gl) showed that custom layers are a viable rendering option and [zarr-cesium](https://github.com/NOC-OI/zarr-cesium) showed how flexible web rendering can be. We borrow code and concepts from both. This library also leans on our prior work on [@carbonplan/maps](https://github.com/carbonplan/maps) for many of its patterns. Custom projection support uses [@developmentseed/raster-reproject](https://github.com/developmentseed/deck.gl-raster) for adaptive mesh generation. LLMs of several makes aided in the coding and debugging of this library.

## license

All the code in this repository is [MIT](https://choosealicense.com/licenses/mit/)-licensed, but we request that you please provide attribution if reusing any of our digital content (graphics, logo, articles, etc.).

## about us

CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions with open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/zarr-layer/issues/new) or [sending us an email](mailto:hello@carbonplan.org).
