import type { Dataset, DatasetConfig } from './types'

type EmptyState = Record<string, never>

export const createSimpleDataset = (
  config: DatasetConfig
): Dataset<EmptyState> => ({
  ...config,
  defaultState: {},
  Controls: () => null,
  buildLayerProps: () => ({ selector: {} }),
})
