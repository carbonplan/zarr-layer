// Loaded into MapLibre's workers so GeoJSON sources follow a registered
// projection: the workers build its converter from the CRS's `definition`.
import proj4 from 'https://cdn.jsdelivr.net/npm/proj4@2.22.0/+esm'

self.setProjectionHandler((definition) => proj4(definition))
