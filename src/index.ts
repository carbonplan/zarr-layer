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
  QueryDataResult,
  QueryDataValues,
  QuerySelector,
  QueryGeometry,
  QueryDataGeometry,
  GeoJSONPolygon,
  GeoJSONMultiPolygon,
  GeoJSONPoint,
  BoundingBox,
} from './query/types'

export { mercatorYFromLat } from './query/query-utils'
