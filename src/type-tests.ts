/**
 * Type compatibility tests for MapLike and ZarrLayer interfaces.
 *
 * This file validates that:
 * 1. MapLibre and Mapbox Map types are assignable to MapLike
 * 2. ZarrLayer is assignable to CustomLayerInterface from both libraries
 *
 * It's not executed at runtime - only checked by TypeScript during typecheck.
 * If this file has type errors, our types have drifted out of sync with the
 * actual library types and need to be updated.
 */

import type {
  Map as MapLibreMap,
  CustomLayerInterface as MapLibreCustomLayer,
} from 'maplibre-gl'
import type {
  Map as MapboxMap,
  CustomLayerInterface as MapboxCustomLayer,
} from 'mapbox-gl'
import type { MapLike } from './types'
import { ZarrLayer } from './zarr-layer'

// Type assertion helpers - these create compile-time errors if types are incompatible
type AssertAssignable<T, U extends T> = U

// =============================================================================
// MapLike compatibility tests
// Ensures onAdd(map: MapLike) accepts both MapLibre and Mapbox maps
// =============================================================================

type _MapLibreMapCompatible = AssertAssignable<MapLike, MapLibreMap>
type _MapboxMapCompatible = AssertAssignable<MapLike, MapboxMap>

// =============================================================================
// CustomLayerInterface compatibility tests
// Ensures map.addLayer(zarrLayer) works with both MapLibre and Mapbox
// =============================================================================

type _MapLibreLayerCompatible = AssertAssignable<MapLibreCustomLayer, ZarrLayer>
type _MapboxLayerCompatible = AssertAssignable<MapboxCustomLayer, ZarrLayer>

// Suppress unused variable warnings
export type {
  _MapLibreMapCompatible,
  _MapboxMapCompatible,
  _MapLibreLayerCompatible,
  _MapboxLayerCompatible,
}
