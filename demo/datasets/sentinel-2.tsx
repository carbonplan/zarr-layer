import React from 'react'
import type { Selector } from '@carbonplan/zarr-layer'
import type { Dataset, ControlsProps } from './types'
import { useAppStore } from '../lib/store'

type Sentinel2State = Record<string, never>

const Controls = ({ state, setState }: ControlsProps<Sentinel2State>) => {
  return null
}

// UTM Zone 37N projection parameters (EPSG:32637)
const UTM_37N_PROJ4 = '+proj=utm +zone=37 +datum=WGS84 +units=m +no_defs'

// Sentinel-2 bounds in UTM Zone 37N meters from zarr metadata [xMin, yMin, xMax, yMax]
const SENTINEL2_BOUNDS: [number, number, number, number] = [
  199980.0, 4590240.0, 309780.0, 4700040.0,
]

const sentinel2: Dataset<Sentinel2State> = {
  id: 'sentinel_2_l2a',
  source:
    'https://s3.explorer.eopf.copernicus.eu/esa-zarr-sentinel-explorer-fra/tests-output/sentinel-2-l2a/S2C_MSIL2A_20251218T083401_N0511_R021_T37TBG_20251218T112007.zarr/measurements/reflectance',
  variable: 'b04',
  colormap: 'blues',
  clim: [0, 0.4],
  latIsAscending: false,
  zarrVersion: 3,
  // Map UTM coordinate dimension names to lat/lon
  spatialDimensions: {
    lat: 'y',
    lon: 'x',
  },
  proj4: UTM_37N_PROJ4,
  bounds: SENTINEL2_BOUNDS,
  info: 'Sentinel-2 L2A B04 (Red band) - UTM 37N',
  sourceInfo:
    'Sentinel-2 Level 2A reflectance data from Copernicus EOPF. UTM Zone 37N (T37TBG) reprojected on-the-fly using proj4.',
  defaultState: {},
  Controls,
  buildLayerProps: () => ({ selector: {} }),
}

// Debug multiscale detection (client-side only)
if (typeof window !== 'undefined') {
  console.log('=== MULTISCALE DEBUG ===');
  console.log('Expected: Sentinel-2 should be detected as untiled multiscale with multiple resolution levels (r10m, r20m, r60m, etc.)');

  // Add a debug function to check the zarr layer after initialization
  (window as any).debugZarrLayer = function() {
    // Access the layer from the map or global state if available
    const layer = (window as any).currentZarrLayer; // You may need to set this in your map component
    if (layer) {
      console.log('=== ZARR LAYER DEBUG ===');
      console.log('Mode type:', layer.mode?.constructor.name);
      console.log('Is multiscale:', layer.mode?.isMultiscale);
      console.log('Levels:', layer.mode?.getLevels?.() || 'No getLevels method');
      console.log('Current level index:', layer.mode?.currentLevelIndex);
      console.log('Store description:', layer.zarrStore?.describe());
      console.log('Available methods on mode:', Object.getOwnPropertyNames(Object.getPrototypeOf(layer.mode || {})));
    } else {
      console.log('No zarr layer found - set window.currentZarrLayer = your_layer_instance');
    }
  };

  console.log('Run debugZarrLayer() in console to check multiscale detection');
}

export default sentinel2
