import * as zarr from 'zarrita'
import type { Readable, AsyncReadable } from '@zarrita/storage'
import type {
  Bounds,
  SpatialDimensions,
  DimIndicesProps,
  CRS,
  UntiledLevel,
  TransformRequest,
} from './types'
import type { XYLimits } from './map-utils'
import { identifyDimensionIndices } from './zarr-utils'

const textDecoder = new TextDecoder()

const decodeJSON = (bytes: Uint8Array | undefined): unknown => {
  if (!bytes) return null
  return JSON.parse(textDecoder.decode(bytes))
}

interface PyramidMetadata {
  levels: string[]
  maxLevelIndex: number
  tileSize: number
  crs: CRS
}

interface MultiscaleDataset {
  path: string
  pixels_per_tile?: number
  crs?: string
}

interface Multiscale {
  datasets: MultiscaleDataset[]
}

// zarr-conventions/multiscales format (untiled multiscales)
interface UntiledMultiscaleLayoutEntry {
  asset: string
  transform?: {
    scale?: [number, number]
    translation?: [number, number]
  }
  derived_from?: string
}

interface UntiledMultiscaleMetadata {
  layout: UntiledMultiscaleLayoutEntry[]
  resampling_method?: string
  crs?: 'EPSG:4326' | 'EPSG:3857'
}

interface ZarrV2ConsolidatedMetadata {
  metadata: Record<string, unknown>
  zarr_consolidated_format?: number
}

interface ZarrV2ArrayMetadata {
  shape: number[]
  chunks: number[]
  fill_value: number | null
  dtype: string
}

interface ZarrV2Attributes {
  _ARRAY_DIMENSIONS?: string[]
  multiscales?: Multiscale[] | UntiledMultiscaleMetadata
  scale_factor?: number
  add_offset?: number
}

interface ZarrV3GroupMetadata {
  zarr_format: 3
  node_type: 'group'
  attributes?: {
    multiscales?: Multiscale[] | UntiledMultiscaleMetadata
  }
  consolidated_metadata?: {
    metadata?: Record<string, ZarrV3ArrayMetadata>
  }
}

interface ZarrV3ArrayMetadata {
  zarr_format: 3
  node_type: 'array'
  shape: number[]
  dimension_names?: string[]
  data_type?: string
  fill_value: number | null
  chunk_grid?: {
    name?: string
    configuration?: {
      chunk_shape?: number[]
    }
  }
  chunks?: number[]
  chunk_key_encoding?: {
    name: string
    configuration?: Record<string, unknown>
  }
  codecs?: Array<{
    name: string
    configuration?: {
      chunk_shape?: number[]
    }
  }>
  storage_transformers?: Array<{
    name: string
    configuration?: Record<string, unknown>
  }>
  attributes?: Record<string, unknown>
}

type AbsolutePath = `/${string}`
type RangeQuery = { offset: number; length: number } | { suffixLength: number }

/**
 * Merge RequestInit objects, properly combining headers instead of replacing.
 * Request overrides take precedence over store overrides.
 */
const mergeInit = (
  storeOverrides: RequestInit,
  requestOverrides?: RequestInit
): RequestInit => {
  if (!requestOverrides) return storeOverrides
  return {
    ...storeOverrides,
    ...requestOverrides,
    headers: {
      ...(storeOverrides.headers as Record<string, string>),
      ...(requestOverrides.headers as Record<string, string>),
    },
  }
}

/**
 * Handle fetch response, returning bytes or undefined for 404.
 */
const handleResponse = async (
  response: Response
): Promise<Uint8Array | undefined> => {
  if (response.status === 404) return undefined
  if (response.status === 200 || response.status === 206) {
    return new Uint8Array(await response.arrayBuffer())
  }
  throw new Error(
    `Unexpected response status ${response.status} ${response.statusText}`
  )
}

/**
 * Fetch a byte range from a URL.
 */
const fetchRange = (
  url: string | URL,
  offset: number,
  length: number,
  opts: RequestInit = {}
): Promise<Response> => {
  return fetch(url, {
    ...opts,
    headers: {
      ...(opts.headers as Record<string, string>),
      Range: `bytes=${offset}-${offset + length - 1}`,
    },
  })
}

/**
 * Fetch suffix bytes from a URL.
 * Uses HEAD + range fallback for backends that don't support suffix ranges.
 */
const fetchSuffix = async (
  url: string | URL,
  suffixLength: number,
  init: RequestInit
): Promise<Response> => {
  // Use HEAD + range fallback (matches zarrita's default behavior)
  // Some backends (like S3) don't support bytes=-n suffix requests well
  const headResponse = await fetch(url, { ...init, method: 'HEAD' })
  if (!headResponse.ok) {
    return headResponse // will be picked up by handleResponse
  }
  const contentLength = headResponse.headers.get('Content-Length')
  const length = Number(contentLength)
  return fetchRange(url, length - suffixLength, suffixLength, init)
}

/**
 * Custom store that calls transformRequest for each request with the fully resolved URL.
 * This enables per-path authentication like presigned S3 URLs.
 */
class TransformingFetchStore implements AsyncReadable<RequestInit> {
  private baseUrl: URL
  private transformRequest: TransformRequest

  constructor(url: string, transformRequest: TransformRequest) {
    this.baseUrl = new URL(url)
    if (!this.baseUrl.pathname.endsWith('/')) {
      this.baseUrl.pathname += '/'
    }
    this.transformRequest = transformRequest
  }

  private resolveUrl(key: AbsolutePath): string {
    const resolved = new URL(key.slice(1), this.baseUrl)
    resolved.search = this.baseUrl.search
    return resolved.href
  }

  async get(
    key: AbsolutePath,
    opts?: RequestInit
  ): Promise<Uint8Array | undefined> {
    const resolvedUrl = this.resolveUrl(key)
    const { url: transformedUrl, ...overrides } = await this.transformRequest(
      resolvedUrl
    )

    const merged = mergeInit(overrides, opts)
    const response = await fetch(transformedUrl, merged)
    return handleResponse(response)
  }

  async getRange(
    key: AbsolutePath,
    range: RangeQuery,
    opts?: RequestInit
  ): Promise<Uint8Array | undefined> {
    const resolvedUrl = this.resolveUrl(key)
    const { url: transformedUrl, ...overrides } = await this.transformRequest(
      resolvedUrl
    )

    const merged = mergeInit(overrides, opts)
    let response: Response

    if ('suffixLength' in range) {
      response = await fetchSuffix(transformedUrl, range.suffixLength, merged)
    } else {
      response = await fetchRange(
        transformedUrl,
        range.offset,
        range.length,
        merged
      )
    }

    return handleResponse(response)
  }
}

type ConsolidatedStore = zarr.Listable<zarr.FetchStore>
type ZarrStoreType =
  | zarr.FetchStore
  | TransformingFetchStore
  | ConsolidatedStore

interface ZarrStoreOptions {
  source: string
  version?: 2 | 3 | null
  variable: string
  spatialDimensions?: SpatialDimensions
  bounds?: Bounds
  coordinateKeys?: string[]
  latIsAscending?: boolean | null
  proj4?: string
  transformRequest?: TransformRequest
}

interface StoreDescription {
  metadata: ZarrV2ConsolidatedMetadata | ZarrV3GroupMetadata | null
  dimensions: string[]
  shape: number[]
  chunks: number[]
  fill_value: number | null
  dtype: string | null
  levels: string[]
  maxLevelIndex: number
  tileSize: number
  crs: CRS
  multiscaleType: 'tiled' | 'untiled' | 'none'
  untiledLevels: UntiledLevel[]
  dimIndices: DimIndicesProps
  xyLimits: XYLimits | null
  scaleFactor: number
  addOffset: number
  coordinates: Record<string, (string | number)[]>
  latIsAscending: boolean | null
  proj4: string | null
}

/**
 * Factory function to create a store with optional request transformation.
 * When transformRequest is provided, returns a TransformingFetchStore that
 * calls the transform function for each request with the fully resolved URL.
 * This enables per-path authentication like presigned S3 URLs.
 */
const createFetchStore = (
  url: string,
  transformRequest?: TransformRequest
): zarr.FetchStore | TransformingFetchStore => {
  if (!transformRequest) {
    return new zarr.FetchStore(url)
  }
  return new TransformingFetchStore(url, transformRequest)
}

export class ZarrStore {
  private static _cache = new Map<
    string,
    ZarrV2ConsolidatedMetadata | ZarrV3GroupMetadata | ZarrV3ArrayMetadata
  >()
  private static _storeCache = new Map<string, Promise<ZarrStoreType>>()

  source: string
  version: 2 | 3 | null
  variable: string
  spatialDimensions: SpatialDimensions
  private explicitBounds: Bounds | null
  coordinateKeys: string[]
  private transformRequest?: TransformRequest

  metadata: ZarrV2ConsolidatedMetadata | ZarrV3GroupMetadata | null = null
  arrayMetadata: ZarrV3ArrayMetadata | null = null
  dimensions: string[] = []
  shape: number[] = []
  chunks: number[] = []
  fill_value: number | null = null
  dtype: string | null = null
  levels: string[] = []
  maxLevelIndex: number = 0
  tileSize: number = 128
  crs: CRS = 'EPSG:4326'
  multiscaleType: 'tiled' | 'untiled' | 'none' = 'none'
  untiledLevels: UntiledLevel[] = []
  dimIndices: DimIndicesProps = {}
  xyLimits: XYLimits | null = null
  scaleFactor: number = 1
  addOffset: number = 0
  coordinates: Record<string, (string | number)[]> = {}
  latIsAscending: boolean | null = null
  proj4: string | null = null
  private _crsFromMetadata: boolean = false // Track if CRS was explicitly set from metadata

  /**
   * Returns the coarsest (lowest resolution) level path.
   * - Tiled pyramids: level 0 is coarsest
   * - Untiled multiscale: last level (maxLevelIndex) is coarsest
   */
  get coarsestLevel(): string | undefined {
    if (this.levels.length === 0) return undefined
    return this.multiscaleType === 'untiled'
      ? this.levels[this.maxLevelIndex]
      : this.levels[0]
  }

  store: ZarrStoreType | null = null
  root: zarr.Location<ZarrStoreType> | null = null
  private _arrayHandles = new Map<
    string,
    Promise<zarr.Array<zarr.DataType, Readable>>
  >()

  initialized: Promise<this>

  constructor({
    source,
    version = null,
    variable,
    spatialDimensions = {},
    bounds,
    coordinateKeys = [],
    latIsAscending = null,
    proj4,
    transformRequest,
  }: ZarrStoreOptions) {
    if (!source) {
      throw new Error('source is a required parameter')
    }
    if (!variable) {
      throw new Error('variable is a required parameter')
    }
    this.source = source
    this.version = version
    this.variable = variable
    this.spatialDimensions = spatialDimensions
    this.explicitBounds = bounds ?? null
    this.coordinateKeys = coordinateKeys
    this.latIsAscending = latIsAscending
    this.proj4 = proj4 ?? null
    this.transformRequest = transformRequest

    this.initialized = this._initialize()
  }

  private async _initialize(): Promise<this> {
    const storeCacheKey = `${this.source}:${this.version ?? 'auto'}`
    let storeHandle: Promise<ZarrStoreType> | undefined

    if (this.transformRequest) {
      // Bypass cache when transformRequest is provided (unique credentials per layer)
      const baseStore = createFetchStore(this.source, this.transformRequest)
      if (this.version === 3) {
        storeHandle = Promise.resolve(baseStore)
      } else {
        storeHandle = zarr.tryWithConsolidated(baseStore).catch(() => baseStore)
      }
    } else {
      // Use cached store for standard requests
      storeHandle = ZarrStore._storeCache.get(storeCacheKey)
      if (!storeHandle) {
        const baseStore = new zarr.FetchStore(this.source)
        if (this.version === 3) {
          storeHandle = Promise.resolve(baseStore)
        } else {
          storeHandle = zarr
            .tryWithConsolidated(baseStore)
            .catch(() => baseStore)
        }
        ZarrStore._storeCache.set(storeCacheKey, storeHandle)
      }
    }

    this.store = await storeHandle
    this.root = zarr.root(this.store)

    if (this.version === 2) {
      await this._loadV2()
    } else if (this.version === 3) {
      await this._loadV3()
    } else {
      try {
        await this._loadV3()
      } catch {
        await this._loadV2()
      }
    }

    await this._loadSpatialMetadata()
    await this._loadCoordinates()

    return this
  }

  private async _loadCoordinates(): Promise<void> {
    if (!this.coordinateKeys.length || !this.levels.length) return

    await Promise.all(
      this.coordinateKeys.map(async (key) => {
        try {
          const coordPath = `${this.levels[0]}/${key}`
          const coordArray = await this._getArray(coordPath)
          const chunk = await coordArray.getChunk([0])
          this.coordinates[key] = Array.from(
            chunk.data as ArrayLike<number | string>
          )
        } catch (err) {
          console.warn(`Failed to load coordinate array for '${key}':`, err)
        }
      })
    )
  }

  cleanup() {
    this._arrayHandles.clear()
    this.store = null
    this.root = null
  }

  describe(): StoreDescription {
    return {
      metadata: this.metadata,
      dimensions: this.dimensions,
      shape: this.shape,
      chunks: this.chunks,
      fill_value: this.fill_value,
      dtype: this.dtype,
      levels: this.levels,
      maxLevelIndex: this.maxLevelIndex,
      tileSize: this.tileSize,
      crs: this.crs,
      multiscaleType: this.multiscaleType,
      untiledLevels: this.untiledLevels,
      dimIndices: this.dimIndices,
      xyLimits: this.xyLimits,
      scaleFactor: this.scaleFactor,
      addOffset: this.addOffset,
      coordinates: this.coordinates,
      latIsAscending: this.latIsAscending,
      proj4: this.proj4,
    }
  }

  async getChunk(
    level: string,
    chunkIndices: number[],
    options?: { signal?: AbortSignal }
  ): Promise<zarr.Chunk<zarr.DataType>> {
    const key = `${level}/${this.variable}`
    const array = await this._getArray(key)
    return array.getChunk(chunkIndices, options)
  }

  async getLevelArray(
    level: string
  ): Promise<zarr.Array<zarr.DataType, Readable>> {
    const key = `${level}/${this.variable}`
    return this._getArray(key)
  }

  async getArray(): Promise<zarr.Array<zarr.DataType, Readable>> {
    return this._getArray(this.variable)
  }

  /**
   * Get metadata (shape, chunks, scale/offset/fill) for a specific untiled level.
   * Used by UntiledMode to determine chunk boundaries and data transforms.
   */
  async getUntiledLevelMetadata(levelAsset: string): Promise<{
    shape: number[]
    chunks: number[]
    scaleFactor: number
    addOffset: number
    fillValue: number | null
    dtype: string | null
  }> {
    const array = await this.getLevelArray(levelAsset)
    const arrayKey = `${levelAsset}/${this.variable}`

    // Try to get metadata from zarr.json for v3, or .zattrs for v2
    let scaleFactor = 1
    let addOffset = 0
    let fillValue: number | null = null
    let dtype: string | null = null

    try {
      if (this.version === 3) {
        const meta = (await this._getJSON(`/${arrayKey}/zarr.json`)) as {
          attributes?: Record<string, unknown>
          fill_value?: unknown
          data_type?: string
        }
        dtype = meta.data_type ?? null
        fillValue = this.normalizeFillValue(meta.fill_value)

        // Float types typically store physical values - skip scaling
        // Integer types store raw values - apply scale/offset
        const isFloatData =
          dtype?.includes('float') || dtype === 'float32' || dtype === 'float64'

        if (isFloatData) {
          scaleFactor = 1
          addOffset = 0
        } else {
          const attrs = meta.attributes
          scaleFactor = (attrs?.scale_factor as number) ?? 1
          addOffset = (attrs?.add_offset as number) ?? 0
        }
      } else {
        const zattrs = (await this._getJSON(`/${arrayKey}/.zattrs`).catch(
          () => ({})
        )) as { scale_factor?: number; add_offset?: number }
        const zarray = (await this._getJSON(`/${arrayKey}/.zarray`)) as {
          fill_value?: unknown
          dtype?: string
        }
        scaleFactor = zattrs.scale_factor ?? 1
        addOffset = zattrs.add_offset ?? 0
        fillValue = this.normalizeFillValue(zarray.fill_value)
        dtype = zarray.dtype ?? null
      }
    } catch (err) {
      console.warn(
        `[ZarrStore] Failed to load per-level metadata for ${arrayKey}:`,
        err
      )
    }

    return {
      shape: array.shape,
      chunks: array.chunks,
      scaleFactor,
      addOffset,
      fillValue,
      dtype,
    }
  }

  private async _getArray(
    key: string
  ): Promise<zarr.Array<zarr.DataType, Readable>> {
    if (!this.root) {
      throw new Error('Zarr store accessed before initialization completed')
    }

    let handle = this._arrayHandles.get(key)

    if (!handle) {
      const location = this.root.resolve(key)
      const openArray = (loc: zarr.Location<Readable>) => {
        if (this.version === 2) {
          return zarr.open.v2(loc, { kind: 'array' })
        } else if (this.version === 3) {
          return zarr.open.v3(loc, { kind: 'array' })
        }
        return zarr.open(loc, { kind: 'array' })
      }
      handle = openArray(location).catch((err: Error) => {
        this._arrayHandles.delete(key)
        throw err
      })
      this._arrayHandles.set(key, handle)
    }

    return handle
  }

  private async _getJSON(path: string): Promise<unknown> {
    if (!this.store) {
      throw new Error('Zarr store accessed before initialization completed')
    }
    if (!path.startsWith('/')) {
      throw new Error(`Expected absolute Zarr path. Received '${path}'.`)
    }

    const bytes = await this.store.get(path)
    const parsed = decodeJSON(bytes)
    if (parsed === null) {
      throw new Error(`Missing metadata at path '${path}'.`)
    }
    return parsed
  }

  private isConsolidatedStore(
    store: ZarrStoreType | null
  ): store is ConsolidatedStore {
    return (
      store !== null &&
      typeof (store as ConsolidatedStore).contents === 'function'
    )
  }

  private async _loadV2() {
    const cacheKey = `v2:${this.source}`
    // Bypass cache when transformRequest is provided (unique credentials per layer)
    let zmetadata = this.transformRequest
      ? undefined
      : (ZarrStore._cache.get(cacheKey) as
          | ZarrV2ConsolidatedMetadata
          | undefined)
    if (!zmetadata) {
      if (this.isConsolidatedStore(this.store)) {
        const rootZattrsBytes = await this.store.get('/.zattrs')
        const rootZattrs = rootZattrsBytes ? decodeJSON(rootZattrsBytes) : {}
        zmetadata = { metadata: { '.zattrs': rootZattrs } }
        if (!this.transformRequest) ZarrStore._cache.set(cacheKey, zmetadata)
      } else {
        try {
          zmetadata = (await this._getJSON(
            '/.zmetadata'
          )) as ZarrV2ConsolidatedMetadata
          if (!this.transformRequest) ZarrStore._cache.set(cacheKey, zmetadata)
        } catch {
          const zattrs = await this._getJSON('/.zattrs')
          zmetadata = { metadata: { '.zattrs': zattrs } }
        }
      }
    }

    this.metadata = { metadata: zmetadata.metadata }

    const rootAttrs = zmetadata.metadata['.zattrs'] as
      | ZarrV2Attributes
      | undefined
    if (rootAttrs?.multiscales) {
      const pyramid = this._getPyramidMetadata(rootAttrs.multiscales)
      this.levels = pyramid.levels
      this.maxLevelIndex = pyramid.maxLevelIndex
      this.tileSize = pyramid.tileSize
      this.crs = pyramid.crs
    }

    const basePath =
      this.levels.length > 0
        ? `${this.levels[0]}/${this.variable}`
        : this.variable
    const v2Metadata = this.metadata as ZarrV2ConsolidatedMetadata
    let zattrs = v2Metadata.metadata[`${basePath}/.zattrs`] as
      | ZarrV2Attributes
      | undefined
    let zarray = v2Metadata.metadata[`${basePath}/.zarray`] as
      | ZarrV2ArrayMetadata
      | undefined

    if (!zattrs || !zarray) {
      ;[zattrs, zarray] = await Promise.all([
        zattrs
          ? Promise.resolve(zattrs)
          : (this._getJSON(`/${basePath}/.zattrs`).catch(
              () => ({})
            ) as Promise<ZarrV2Attributes>),
        zarray
          ? Promise.resolve(zarray)
          : (this._getJSON(
              `/${basePath}/.zarray`
            ) as Promise<ZarrV2ArrayMetadata>),
      ])
      v2Metadata.metadata[`${basePath}/.zattrs`] = zattrs
      v2Metadata.metadata[`${basePath}/.zarray`] = zarray
    }

    this.dimensions = zattrs?._ARRAY_DIMENSIONS || []
    this.shape = zarray?.shape || []
    this.chunks = zarray?.chunks || []
    this.fill_value = this.normalizeFillValue(zarray?.fill_value ?? null)
    this.dtype = zarray?.dtype || null
    this.scaleFactor = zattrs?.scale_factor ?? 1
    this.addOffset = zattrs?.add_offset ?? 0

    await this._computeDimIndices()
  }

  private async _loadV3() {
    const metadataCacheKey = `v3:${this.source}`
    // Bypass cache when transformRequest is provided (unique credentials per layer)
    let metadata = this.transformRequest
      ? undefined
      : (ZarrStore._cache.get(metadataCacheKey) as
          | ZarrV3GroupMetadata
          | undefined)
    if (!metadata) {
      metadata = (await this._getJSON('/zarr.json')) as ZarrV3GroupMetadata
      if (!this.transformRequest) {
        ZarrStore._cache.set(metadataCacheKey, metadata)

        if (metadata.consolidated_metadata?.metadata) {
          for (const [key, arrayMeta] of Object.entries(
            metadata.consolidated_metadata.metadata
          )) {
            const arrayCacheKey = `v3:${this.source}/${key}`
            ZarrStore._cache.set(arrayCacheKey, arrayMeta)
          }
        }
      }
    }
    this.metadata = metadata
    this.version = 3

    if (metadata.attributes?.multiscales) {
      const pyramid = this._getPyramidMetadata(metadata.attributes.multiscales)
      this.levels = pyramid.levels
      this.maxLevelIndex = pyramid.maxLevelIndex
      this.tileSize = pyramid.tileSize
      this.crs = pyramid.crs
    }

    const arrayKey =
      this.levels.length > 0
        ? `${this.levels[0]}/${this.variable}`
        : this.variable
    const arrayCacheKey = `v3:${this.source}/${arrayKey}`
    let arrayMetadata = this.transformRequest
      ? undefined
      : (ZarrStore._cache.get(arrayCacheKey) as ZarrV3ArrayMetadata | undefined)
    if (!arrayMetadata) {
      arrayMetadata = (await this._getJSON(
        `/${arrayKey}/zarr.json`
      )) as ZarrV3ArrayMetadata
      if (!this.transformRequest)
        ZarrStore._cache.set(arrayCacheKey, arrayMetadata)
    }
    this.arrayMetadata = arrayMetadata

    const attrs = arrayMetadata.attributes as
      | Record<string, unknown>
      | undefined
    // Legacy v3 support: attributes._ARRAY_DIMENSIONS.
    const legacyDims =
      Array.isArray(attrs?._ARRAY_DIMENSIONS) && attrs?._ARRAY_DIMENSIONS

    this.dimensions = arrayMetadata.dimension_names || legacyDims || []
    this.shape = arrayMetadata.shape

    const isSharded = arrayMetadata.codecs?.[0]?.name === 'sharding_indexed'
    const shardedChunkShape =
      isSharded && arrayMetadata.codecs?.[0]?.configuration
        ? (arrayMetadata.codecs[0].configuration as { chunk_shape?: number[] })
            .chunk_shape
        : undefined
    const gridChunkShape = arrayMetadata.chunk_grid?.configuration?.chunk_shape
    // Some pre-spec pyramids used top-level chunks; keep as a fallback.
    const legacyChunks = Array.isArray(arrayMetadata.chunks)
      ? arrayMetadata.chunks
      : undefined
    this.chunks =
      shardedChunkShape || gridChunkShape || legacyChunks || this.shape

    this.fill_value = this.normalizeFillValue(arrayMetadata.fill_value)
    this.dtype = arrayMetadata.data_type || null
    this.scaleFactor =
      typeof attrs?.scale_factor === 'number' ? attrs.scale_factor : 1
    this.addOffset =
      typeof attrs?.add_offset === 'number' ? attrs.add_offset : 0

    await this._computeDimIndices()
  }

  private async _computeDimIndices() {
    if (this.dimensions.length === 0) return

    this.dimIndices = identifyDimensionIndices(
      this.dimensions,
      this.spatialDimensions
    )

    // Collect the actual names of identified spatial dimensions
    // (e.g., 'projection_y_coordinate' if mapped to 'lat')
    const spatialDimNames = new Set(
      ['lat', 'lon']
        .filter((key) => this.dimIndices[key])
        .map((key) => this.dimIndices[key].name.toLowerCase())
    )

    // Add ALL dimensions to dimIndices so selectors can reference them by name
    // (e.g., 'time', 'level', etc. - not just lat/lon)
    for (let i = 0; i < this.dimensions.length; i++) {
      const dimName = this.dimensions[i]
      // Skip if already added (e.g., 'lat' was already mapped with its coordinate array)
      if (this.dimIndices[dimName] || this.dimIndices[dimName.toLowerCase()]) {
        continue
      }
      // Skip if this is the name of an identified spatial dimension
      // (already tracked under 'lat' or 'lon' keys)
      if (spatialDimNames.has(dimName.toLowerCase())) {
        continue
      }
      this.dimIndices[dimName] = {
        name: dimName,
        index: i,
        array: null,
      }
    }
  }

  private normalizeFillValue(value: unknown): number | null {
    if (value === undefined || value === null) return null
    if (typeof value === 'string') {
      const lower = value.toLowerCase()
      if (lower === 'nan') return Number.NaN
      const parsed = Number(value)
      return Number.isNaN(parsed) ? null : parsed
    }
    if (typeof value === 'number') {
      return value
    }
    return null
  }

  /**
   * Find the highest resolution level for untiled bounds detection.
   * Coarse levels have larger pixels, so padding offsets are more significant.
   * Trade-off is higher bandwidth. Users can provide explicit `bounds` to skip.
   */
  private async _findBoundsLevel(): Promise<string | undefined> {
    if (this.levels.length === 0 || !this.root) return undefined
    if (this.levels.length === 1) return this.levels[0]

    const openArray = (loc: zarr.Location<Readable>) => {
      if (this.version === 2) {
        return zarr.open.v2(loc, { kind: 'array' })
      } else if (this.version === 3) {
        return zarr.open.v3(loc, { kind: 'array' })
      }
      return zarr.open(loc, { kind: 'array' })
    }

    // Compare first and last levels to determine which has higher resolution
    const firstLevel = this.levels[0]
    const lastLevel = this.levels[this.levels.length - 1]

    try {
      const firstArray = await openArray(
        this.root.resolve(`${firstLevel}/${this.variable}`)
      )
      const lastArray = await openArray(
        this.root.resolve(`${lastLevel}/${this.variable}`)
      )

      // Compare total pixels (product of dimensions)
      const firstSize = firstArray.shape.reduce((a, b) => a * b, 1)
      const lastSize = lastArray.shape.reduce((a, b) => a * b, 1)

      return firstSize >= lastSize ? firstLevel : lastLevel
    } catch {
      // If we can't determine, default to first level
      return firstLevel
    }
  }

  private async _loadSpatialMetadata() {
    // Apply explicit bounds first (takes precedence for all multiscale types)
    // Bounds are in source CRS units (degrees for EPSG:4326, meters for EPSG:3857/proj4)
    if (this.explicitBounds) {
      const [west, south, east, north] = this.explicitBounds
      this.xyLimits = { xMin: west, xMax: east, yMin: south, yMax: north }
    }

    // Tiled pyramids: use standard global extent if no explicit bounds
    if (this.multiscaleType === 'tiled') {
      if (!this.xyLimits) {
        this.xyLimits = { xMin: -180, xMax: 180, yMin: -90, yMax: 90 }
      }
      if (this.latIsAscending === null) {
        this.latIsAscending = false // Tiled pyramids: row 0 = north
      }
      return
    }

    // For untiled: determine what we still need to detect
    const needsBounds = !this.xyLimits
    const needsLatAscending = this.latIsAscending === null

    // If explicit bounds provided and user doesn't need latIsAscending detection, skip coord fetch
    // (respects user intent to avoid coord reads by providing bounds)
    if (!needsBounds && !needsLatAscending) {
      return
    }

    // Can't fetch coords without dimension info - return silently (best-effort)
    if (!this.dimIndices.lon || !this.dimIndices.lat || !this.root) {
      return
    }

    try {
      const boundsLevel = await this._findBoundsLevel()
      const levelRoot = boundsLevel ? this.root.resolve(boundsLevel) : this.root

      const openArray = (loc: zarr.Location<Readable>) => {
        if (this.version === 2) return zarr.open.v2(loc, { kind: 'array' })
        if (this.version === 3) return zarr.open.v3(loc, { kind: 'array' })
        return zarr.open(loc, { kind: 'array' })
      }

      const lonName = this.spatialDimensions.lon ?? this.dimIndices.lon.name
      const latName = this.spatialDimensions.lat ?? this.dimIndices.lat.name

      const xarr = await openArray(levelRoot.resolve(lonName))
      const yarr = await openArray(levelRoot.resolve(latName))

      const xLen = xarr.shape[0]
      const yLen = yarr.shape[0]

      type ZarrResult = { data: ArrayLike<number> }
      const [xFirst, xLast, yFirstTwo, yLast] = (await Promise.all([
        zarr.get(xarr, [zarr.slice(0, 1)]),
        zarr.get(xarr, [zarr.slice(xLen - 1, xLen)]),
        zarr.get(yarr, [zarr.slice(0, 2)]),
        zarr.get(yarr, [zarr.slice(yLen - 1, yLen)]),
      ])) as ZarrResult[]

      const x0 = xFirst.data[0]
      const x1 = xLast.data[0]
      const y0 = yFirstTwo.data[0]
      const y1 = yFirstTwo.data[1]
      const yN = yLast.data[0]

      // Detect latIsAscending from first two y values
      const detectedLatAscending = y1 > y0
      if (needsLatAscending) {
        this.latIsAscending = detectedLatAscending
      }

      // Compute bounds from coordinate extents
      const xMin = Math.min(x0, x1)
      const xMax = Math.max(x0, x1)
      const yMin = Math.min(y0, yN)
      const yMax = Math.max(y0, yN)

      if (needsBounds) {
        this.xyLimits = { xMin, xMax, yMin, yMax }
      }

      // Warn users to set explicit values to skip future coordinate fetches
      if (this.multiscaleType === 'untiled') {
        const hints: string[] = []
        if (needsBounds)
          hints.push(`bounds: [${xMin}, ${yMin}, ${xMax}, ${yMax}]`)
        if (needsLatAscending && !detectedLatAscending)
          hints.push('latIsAscending: false')

        if (hints.length > 0) {
          console.warn(
            `[zarr-layer] Detected from coordinate arrays. ` +
              `Set explicitly to skip this fetch: ${hints.join(', ')}`
          )
        }
      }
    } catch (err) {
      if (needsBounds) {
        throw new Error(
          `Failed to load bounds from coordinate arrays. ` +
            `Provide explicit bounds via the 'bounds' option. ` +
            `Error: ${err instanceof Error ? err.message : err}`
        )
      }
      // Bounds were explicit but latIsAscending detection failed - just log
      console.debug(
        '[zarr-layer] Could not detect latIsAscending from coordinates:',
        err instanceof Error ? err.message : err
      )
    }

    // Infer CRS from bounds for untiled multiscales if not explicitly set
    // Only classify as meters if clearly outside degree range (> 360)
    // This handles both [-180, 180] and [0, 360] degree conventions
    if (
      this.multiscaleType === 'untiled' &&
      !this._crsFromMetadata &&
      this.xyLimits
    ) {
      const maxAbsX = Math.max(
        Math.abs(this.xyLimits.xMin),
        Math.abs(this.xyLimits.xMax)
      )
      if (maxAbsX > 360) {
        this.crs = 'EPSG:3857'
      }
    }
  }

  /**
   * Parse multiscale metadata to determine pyramid structure.
   *
   * Supports three multiscale formats:
   *
   * 1. **zarr-conventions/multiscales** (layout format):
   *    Uses `layout` array with transform info. Parsed by `_parseUntiledMultiscale()`.
   *    Example: `{ layout: [{ asset: "0", transform: { scale: [...] } }, ...] }`
   *
   * 2. **OME-NGFF style** (datasets format):
   *    Uses `datasets` array. If `pixels_per_tile` is present, treated as tiled pyramid.
   *    Otherwise treated as untiled multi-level.
   *    Example: `[{ datasets: [{ path: "0", crs: "EPSG:4326" }, ...] }]`
   *
   * 3. **Single level**: No multiscale metadata, treated as single untiled image.
   *
   * For untiled formats, shapes are extracted from consolidated metadata when available
   * to avoid per-level network requests.
   */
  private _getPyramidMetadata(
    multiscales: Multiscale[] | UntiledMultiscaleMetadata | undefined
  ): PyramidMetadata {
    if (!multiscales) {
      // No multiscale metadata - single level untiled dataset
      this.multiscaleType = 'untiled'
      return {
        levels: [],
        maxLevelIndex: 0,
        tileSize: 128,
        crs: this.crs,
      }
    }

    // Format 1: zarr-conventions/multiscales (has 'layout' key)
    // See: https://github.com/zarr-conventions/multiscales
    if ('layout' in multiscales && Array.isArray(multiscales.layout)) {
      return this._parseUntiledMultiscale(multiscales)
    }

    // Format 2: OME-NGFF style (array with 'datasets' key)
    // See: https://ngff.openmicroscopy.org/latest/
    if (Array.isArray(multiscales) && multiscales[0]?.datasets?.length) {
      const datasets = multiscales[0].datasets
      const levels = datasets.map((dataset) => String(dataset.path))
      const maxLevelIndex = levels.length - 1
      const tileSize = datasets[0].pixels_per_tile
      // If CRS is absent, default to EPSG:3857 to match pyramid (mercator) tiling.
      const crs: CRS =
        (datasets[0].crs as CRS) === 'EPSG:4326' ? 'EPSG:4326' : 'EPSG:3857'

      // If pixels_per_tile is present, this is a tiled pyramid (slippy map tiles).
      // Otherwise, treat as untiled multi-level (each level is a complete image).
      if (tileSize) {
        this.multiscaleType = 'tiled'
        return { levels, maxLevelIndex, tileSize, crs }
      } else {
        // Multi-level but not tiled - use UntiledMode
        // Try to extract shapes from consolidated metadata to avoid per-level fetches
        const consolidatedMeta = (this.metadata as ZarrV3GroupMetadata)
          ?.consolidated_metadata?.metadata

        this.untiledLevels = levels.map((level) => {
          const untiledLevel: UntiledLevel = {
            asset: level,
            scale: [1.0, 1.0] as [number, number],
            translation: [0.0, 0.0] as [number, number],
          }

          // Extract shape/chunks from consolidated metadata if available
          if (consolidatedMeta) {
            const arrayKey = `${level}/${this.variable}`
            const arrayMeta = consolidatedMeta[arrayKey] as
              | ZarrV3ArrayMetadata
              | undefined
            if (arrayMeta?.shape) {
              untiledLevel.shape = arrayMeta.shape
              // Extract chunks from chunk_grid or sharding codec
              const gridChunks =
                arrayMeta.chunk_grid?.configuration?.chunk_shape
              const shardChunks = arrayMeta.codecs?.find(
                (c) => c.name === 'sharding_indexed'
              )?.configuration?.chunk_shape as number[] | undefined
              untiledLevel.chunks = shardChunks || gridChunks || arrayMeta.shape
            }
          }

          return untiledLevel
        })
        this.multiscaleType = 'untiled'
        return { levels, maxLevelIndex, tileSize: 128, crs }
      }
    }

    // Unrecognized multiscale format - treat as single level untiled
    this.multiscaleType = 'untiled'
    return {
      levels: [],
      maxLevelIndex: 0,
      tileSize: 128,
      crs: this.crs,
    }
  }

  /**
   * Parse zarr-conventions/multiscales format (layout-based).
   *
   * This format uses a `layout` array where each entry specifies:
   * - `asset`: path to the level (e.g., "0", "1", ...)
   * - `transform`: optional scale/translation for georeferencing
   *
   * Example metadata:
   * ```json
   * {
   *   "layout": [
   *     { "asset": "0", "transform": { "scale": [1.0, 1.0], "translation": [0, 0] } },
   *     { "asset": "1", "transform": { "scale": [2.0, 2.0], "translation": [0, 0] } }
   *   ],
   *   "crs": "EPSG:4326"
   * }
   * ```
   *
   * @see https://github.com/zarr-conventions/multiscales
   */
  private _parseUntiledMultiscale(
    metadata: UntiledMultiscaleMetadata
  ): PyramidMetadata {
    const layout = metadata.layout
    if (!layout || layout.length === 0) {
      this.multiscaleType = 'untiled'
      return {
        levels: [],
        maxLevelIndex: 0,
        tileSize: 128,
        crs: this.crs,
      }
    }

    // Extract levels from layout
    const levels = layout.map((entry) => entry.asset)
    const maxLevelIndex = levels.length - 1

    // Try to extract shapes from consolidated metadata to avoid per-level fetches
    const consolidatedMeta = (this.metadata as ZarrV3GroupMetadata)
      ?.consolidated_metadata?.metadata

    // Build untiledLevels with transform info and shapes from consolidated metadata
    this.untiledLevels = layout.map((entry) => {
      const level: UntiledLevel = {
        asset: entry.asset,
        scale: entry.transform?.scale ?? [1.0, 1.0],
        translation: entry.transform?.translation ?? [0.0, 0.0],
      }

      // Extract shape/chunks from consolidated metadata if available
      if (consolidatedMeta) {
        const arrayKey = `${entry.asset}/${this.variable}`
        const arrayMeta = consolidatedMeta[arrayKey] as
          | ZarrV3ArrayMetadata
          | undefined
        if (arrayMeta?.shape) {
          level.shape = arrayMeta.shape
          // Extract chunks from chunk_grid or sharding codec
          const gridChunks = arrayMeta.chunk_grid?.configuration?.chunk_shape
          const shardChunks = arrayMeta.codecs?.find(
            (c) => c.name === 'sharding_indexed'
          )?.configuration?.chunk_shape as number[] | undefined
          level.chunks = shardChunks || gridChunks || arrayMeta.shape
        }
      }

      return level
    })

    this.multiscaleType = 'untiled'

    // Check for explicit CRS in metadata, otherwise use configured CRS
    // (bounds-based inference will happen after coordinate arrays are loaded)
    const crs: CRS = metadata.crs ?? this.crs
    if (metadata.crs) {
      this._crsFromMetadata = true
    }

    return {
      levels,
      maxLevelIndex,
      tileSize: 128, // Will be overridden by chunk shape
      crs,
    }
  }

  static clearCache() {
    ZarrStore._cache.clear()
    ZarrStore._storeCache.clear()
  }
}
