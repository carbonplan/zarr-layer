export { ZarrLayer } from './zarr-layer'
export type {
  ZarrLayerOptions,
  ColormapArray,
  SpatialDimensions,
  LoadingState,
  LoadingStateCallback,
  Selector,
  TransformRequest,
  RequestParameters,
} from './types'

// Query interface exports
export type { QueryResult, QueryDataValues, QueryGeometry } from './query/types'

// Band math utilities
export { ndvi, trueColor } from './band-math'
export type { BandMathConfig, NdviOptions, TrueColorOptions } from './band-math'
