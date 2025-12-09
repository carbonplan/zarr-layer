export { ZarrLayer } from './zarr-layer'
export type {
  ZarrLayerOptions,
  ColormapArray,
  DimensionNamesProps,
  LoadingState,
  LoadingStateCallback,
} from './types'

// Query interface exports
export type {
  PointQueryResult,
  RegionQueryResult,
  RegionValues,
  QuerySelector,
  QueryGeometry,
  GeoJSONPolygon,
  GeoJSONMultiPolygon,
  BoundingBox,
} from './query/types'

export { mercatorYFromLat } from './query/query-utils'
