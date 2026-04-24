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

## globe rendering and polar coverage

Web Mercator rendering clips near ±85° latitude, leaving visible "pole holes" on globe projections. For untiled EPSG:4326 and `proj4` datasets:

**MapLibre** — Full polar coverage is always enabled via a direct ECEF rendering path. No configuration needed.

**Mapbox** — Set `renderPoles: true` to enable an experimental direct ECEF path that bypasses tile draping. This relies on Mapbox internal APIs and may break across Mapbox GL JS versions. During Mapbox's globe-to-mercator zoom morph the layer automatically falls back to the standard draped path (with pole holes visible). Incompatible with draping the zarr layer over Mapbox terrain — when terrain is enabled the layer always uses the draped tile path.

```ts
new ZarrLayer({
  // ...
  renderPoles: true, // Mapbox only — MapLibre always renders to the poles
})
```

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
| source | string | Zarr store URL (required unless `store` is provided) |
| variable | string | Variable name to render |
| colormap | array | Array of hex strings or `[r,g,b]` values |
| clim | [min, max] | Color scale limits |

**Optional:**
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| store | Readable | - | Custom zarrita-compatible store (e.g., IcechunkStore). When provided, `source` becomes optional. |
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
| transformRequest | function | - | Transform request URLs and add headers/credentials (see [authentication](#authentication)) |
| renderPoles | boolean | `false` | Enable polar coverage in Mapbox globe for untiled EPSG:4326/proj4 datasets (see [globe rendering](#globe-rendering-and-polar-coverage)). No effect on tiled or EPSG:3857 data. MapLibre always renders to the poles. |

## methods

```ts
layer.setOpacity(0.8)
layer.setClim([0, 100])
layer.setColormap(['#000', '#fff'])
layer.setSelector({ time: 5 })
layer.setVariable('precipitation') // async - reloads metadata
layer.setUniforms({ u_weight: 1.5 }) // no-op unless layer has customFrag
```

### throttling rapid selector changes

`setSelector` is synchronous and kicks off a fetch on every call. For UIs that fire rapid updates (e.g. dragging a time slider), debounce the value on your side before handing it to the layer so you aren't firing one abort-cancelled fetch per pointer event.

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

Band names are automatically sanitized to valid GLSL identifiers: any characters that aren't letters, digits, or underscores are replaced with underscores, and names starting with a digit are prefixed with an underscore. For example, `s2med_harvest:B02` becomes `s2med_harvest_B02` and `123band` becomes `_123band`.

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

### NDVI example

Here's an example of computing NDVI (Normalized Difference Vegetation Index) using custom shaders:

```ts
new ZarrLayer({
  source: 'https://example.com/sentinel2.zarr',
  variable: 'data',
  colormap: [
    /* gradient */
  ],
  selector: { band: ['B08', 'B04'], time: 0 },
  clim: [-1, 1],
  customFrag: `
    float ndvi = (B08 - B04) / (B08 + B04);
    float norm = (ndvi - clim.x) / (clim.y - clim.x);
    vec4 c = texture(colormap, vec2(clamp(norm, 0.0, 1.0), 0.5));
    fragColor = vec4(c.rgb, opacity);
  `,
})
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

Datasets using a custom projection (via the `proj4` option) return coordinates in the source coordinate system, with keys matching the store's axis names (e.g. `y`/`x`). All other datasets (EPSG:4326, EPSG:3857) return `lat`/`lon` keys with WGS84 degree values.

You can pass a third `options` argument to control query behavior:

```ts
const result = await layer.queryData(geometry, selector, {
  signal: abortController.signal, // cancel in-flight query
  includeSpatialCoordinates: false, // omit per-pixel coordinates for slimmer results
})
```

**Note:** Query results match rendered values (`scale_factor`/`add_offset` applied, `fillValue`/NaN filtered).

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

## custom stores

For advanced use cases like [Icechunk](https://icechunk.io/), you can pass a custom zarrita-compatible store directly. When using a custom store, `source` becomes optional:

```ts
import { IcechunkStore } from 'wip-tk'

const store = await IcechunkStore.open(...)

new ZarrLayer({
  id: 'icechunk-layer',
  store,
  variable: 'temperature',
  colormap: [...],
  clim: [0, 100],
})
```

The store must implement the zarrita `Readable` interface with at minimum a `get(key: string)` method.

## codecs

The library uses [zarrita](https://github.com/manzt/zarrita.js) for Zarr data access. zarrita includes built-in codecs for `bytes`, `zlib`, `gzip`, `blosc`, `lz4`, `zstd`, `transpose`, `crc32c`, and `bitround`. You can add more if needed!

### adding custom codecs

Virtualized NetCDF data may use `numcodecs.*`-prefixed codec names (e.g., `numcodecs.zlib`, `numcodecs.shuffle`) that zarrita doesn't recognize by default. Use `codecRegistry` to register them before creating layers:

```ts
import { codecRegistry } from '@carbonplan/zarr-layer'

// Alias numcodecs.zlib → zarrita's built-in zlib
const zlibFactory = codecRegistry.get('zlib')
if (zlibFactory) codecRegistry.set('numcodecs.zlib', zlibFactory)

// numcodecs.shuffle — byte un-shuffle by element size
codecRegistry.set('numcodecs.shuffle', async () => ({
  fromConfig(config: { elementsize?: number }) {
    const elementsize = config?.elementsize ?? 1
    return {
      kind: 'bytes_to_bytes',
      decode(bytes: Uint8Array): Uint8Array {
        if (elementsize <= 1) return bytes
        const n = bytes.length
        const count = Math.floor(n / elementsize)
        const out = new Uint8Array(n)
        for (let i = 0; i < count; i++) {
          for (let j = 0; j < elementsize; j++) {
            out[i * elementsize + j] = bytes[j * count + i]
          }
        }
        for (let i = count * elementsize; i < n; i++) {
          out[i] = bytes[i]
        }
        return out
      },
    }
  },
}))
```

## thanks

This experiment is only possible following in the footsteps of other work in this space. [zarr-gl](https://github.com/carderne/zarr-gl) showed that custom layers are a viable rendering option and [zarr-cesium](https://github.com/NOC-OI/zarr-cesium) showed how flexible web rendering can be. We borrow code and concepts from both. This library also leans on our prior work on [@carbonplan/maps](https://github.com/carbonplan/maps) for many of its patterns. Custom projection support uses [@developmentseed/raster-reproject](https://github.com/developmentseed/deck.gl-raster) for adaptive mesh generation. LLMs of several makes aided in the coding and debugging of this library.

## license

All the code in this repository is [MIT](https://choosealicense.com/licenses/mit/)-licensed, but we request that you please provide attribution if reusing any of our digital content (graphics, logo, articles, etc.).

## about us

CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions with open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/zarr-layer/issues/new) or [sending us an email](mailto:hello@carbonplan.org).
