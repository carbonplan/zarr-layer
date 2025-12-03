# zarr-maplibre

Minimal MapLibre GL custom layer for rendering Zarr datasets, extracted from the
[zarr-cesium](https://github.com/NOC-OI/zarr-cesium) project with elements of
[zarr-gl](https://github.com/carderne/zarr-gl) and
[carbonplan/maps](https://github.com/carbonplan/maps).

## Install

```bash
npm install zarr-maplibre maplibre-gl
```

## Build locally

```bash
npm install
npm run build
```

## Usage

```ts
import maplibregl from "maplibre-gl";
import { ZarrLayer } from "zarr-maplibre";

const map = new maplibregl.Map({
  container: "map",
  style: "https://demotiles.maplibre.org/style.json",
  center: [0, 20],
  zoom: 2,
});

map.on("load", () => {
  map.addLayer(
    new ZarrLayer({
      id: "zarr-layer",
      source: "https://example.com/my.zarr",
      variable: "temperature",
      vmin: 270,
      vmax: 310,
      colormap: "viridis",
      selector: { time: 0 },
      zarrVersion: 3,
    })
  );
});
```

## Demo

A static demo page is included. Build the library, then serve the repo root:

```bash
npm install
npm run build
python -m http.server 8000
# open http://localhost:8000/maplibre-demo.html
```

The demo pulls the prebuilt bundle from `dist/index.mjs`.
