export { ZarrLayer } from './zarr-layer'
export { ZarrLayerNotReadyError } from './errors'
export type {
  ZarrLayerOptions,
  ColormapArray,
  SpatialDimensions,
  LoadingState,
  LoadingStateCallback,
  Selector,
  TransformRequest,
  OnAuthError,
  RequestParameters,
} from './types'

// Query interface exports
export type {
  QueryResult,
  QueryDataValues,
  QueryGeometry,
  QueryLevel,
  QueryOptions,
} from './query/types'

// Codec registry — re-export for registering custom codecs
export { registry as codecRegistry } from 'zarrita'
