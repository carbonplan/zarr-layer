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
      resolvedUrl,
      { method: 'GET' }
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

    let response: Response

    if ('suffixLength' in range) {
      // For suffix queries, we need separate signed URLs for HEAD and GET
      const { url: headUrl, ...headOverrides } = await this.transformRequest(
        resolvedUrl,
        { method: 'HEAD' }
      )
      const headMerged = mergeInit(headOverrides, opts)
      const headResponse = await fetch(headUrl, {
        ...headMerged,
        method: 'HEAD',
      })
      if (!headResponse.ok) {
        return handleResponse(headResponse)
      }
      const contentLength = headResponse.headers.get('Content-Length')
      const length = Number(contentLength)

      // Now get the actual range with a GET-signed URL
      const { url: getUrl, ...getOverrides } = await this.transformRequest(
        resolvedUrl,
        { method: 'GET' }
      )
      const getMerged = mergeInit(getOverrides, opts)
      response = await fetchRange(
        getUrl,
        length - range.suffixLength,
        range.suffixLength,
        getMerged
      )
    } else {
      const { url: transformedUrl, ...overrides } = await this.transformRequest(
        resolvedUrl,
        { method: 'GET' }
      )
      const merged = mergeInit(overrides, opts)
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
  crs?: string
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
  latIsAscending: boolean
  proj4: string | null
  tileOffsets: Map<number, { x: number; y: number }>
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
  latIsAscending: boolean = true // Default: row 0 = south; overridden by detection
  private _latIsAscendingUserSet: boolean = false
  proj4: string | null = null
  private _crsFromMetadata: boolean = false // Track if CRS was explicitly set from metadata
  private _crsOverride: boolean = false // Track if CRS was explicitly set by user
  tileOffsets: Map<number, { x: number; y: number }> = new Map() // Per-zoom tile coordinate offsets for regional pyramids

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
    crs,
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
    if (latIsAscending !== null) {
      this.latIsAscending = latIsAscending
      this._latIsAscendingUserSet = true
    }
    this.proj4 = proj4 ?? null
    if (crs) {
      const normalized = crs.toUpperCase()
      if (normalized === 'EPSG:4326' || normalized === 'EPSG:3857') {
        this.crs = normalized
        this._crsOverride = true
      } else if (!this.proj4) {
        console.warn(
          `[zarr-layer] CRS "${crs}" requires 'proj4' to render correctly. ` +
            `Falling back to inferred CRS.`
        )
      }
    }
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
      tileOffsets: this.tileOffsets,
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
    scaleFactor: number | undefined
    addOffset: number | undefined
    fillValue: number | null
    dtype: string | null
  }> {
    const array = await this.getLevelArray(levelAsset)
    const arrayKey = `${levelAsset}/${this.variable}`

    // Try to get metadata from zarr.json for v3, or .zattrs for v2
    // Return undefined for scaleFactor/addOffset when not specified,
    // allowing caller to fall back to dataset-level values
    let scaleFactor: number | undefined = undefined
    let addOffset: number | undefined = undefined
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

        // Float data typically stores already-physical values (e.g., pyramid levels
        // created by averaging). Integer data stores raw counts needing conversion.
        // For heterogeneous pyramids like Sentinel-2, lower-res float levels inherit
        // scale_factor attributes but shouldn't have them re-applied.
        const isFloatData =
          dtype?.includes('float') || dtype === 'float32' || dtype === 'float64'

        if (isFloatData) {
          // Float data: assume already physical, use 1/0
          scaleFactor = 1
          addOffset = 0
        } else {
          // Integer data: apply scale_factor/add_offset if present
          const attrs = meta.attributes
          if (attrs?.scale_factor !== undefined) {
            scaleFactor = attrs.scale_factor as number
          }
          if (attrs?.add_offset !== undefined) {
            addOffset = attrs.add_offset as number
          }
        }
      } else {
        // Zarr v2 path
        const zattrs = (await this._getJSON(`/${arrayKey}/.zattrs`).catch(
          () => ({})
        )) as { scale_factor?: number; add_offset?: number }
        const zarray = (await this._getJSON(`/${arrayKey}/.zarray`)) as {
          fill_value?: unknown
          dtype?: string
        }
        fillValue = this.normalizeFillValue(zarray.fill_value)
        dtype = zarray.dtype ?? null

        // Same float logic as v3: float data is already physical, integer needs scaling
        const isFloatData =
          dtype?.includes('float') || dtype === 'float32' || dtype === 'float64'

        if (isFloatData) {
          scaleFactor = 1
          addOffset = 0
        } else {
          // Only set if attributes actually exist - leave undefined for fallback
          if (zattrs.scale_factor !== undefined) {
            scaleFactor = zattrs.scale_factor
          }
          if (zattrs.add_offset !== undefined) {
            addOffset = zattrs.add_offset
          }
        }
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
      handle = this._openArray(location).catch((err: Error) => {
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
      if (!this._crsOverride) {
        this.crs = pyramid.crs
      }
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
      if (!this._crsOverride) {
        this.crs = pyramid.crs
      }
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

  // Track in-flight offset calculations to avoid duplicate requests
  private _pendingOffsetCalculations = new Map<
    number,
    Promise<{ x: number; y: number } | null>
  >()

  /**
   * Calculate tile offsets using only consolidated metadata (fast, synchronous).
   * Called during initialization. Levels without consolidated metadata are
   * computed lazily via getTileOffset() when first requested.
   */
  private _calculateTileOffsetsFromConsolidatedMetadata() {
    if (this.crs !== 'EPSG:3857') return

    for (const levelPath of this.levels) {
      const zoom = parseInt(levelPath, 10)
      if (isNaN(zoom)) continue

      const spatialRefPath = `${levelPath}/spatial_ref`
      const variablePath = `${levelPath}/${this.variable}`

      const fromConsolidated = this._getGeoTransformFromConsolidatedMetadata(
        spatialRefPath,
        variablePath
      )

      if (fromConsolidated) {
        const extent = this._parseGeoTransformExtent(
          fromConsolidated.geoTransform,
          fromConsolidated.shape
        )
        if (extent) {
          const lonLat = this._extentToLonLat(extent)
          this.tileOffsets.set(zoom, {
            x: this._lonToTile(lonLat.lonMin, zoom),
            y: this._latToTile(lonLat.latMax, zoom),
          })
          if (!this.xyLimits) {
            this.xyLimits = {
              xMin: lonLat.lonMin,
              xMax: lonLat.lonMax,
              yMin: lonLat.latMin,
              yMax: lonLat.latMax,
            }
          }
        }
      }
      // Levels without consolidated metadata will be computed lazily
    }
  }

  /**
   * Get tile offset for a zoom level. Uses cached value if available,
   * otherwise computes lazily (only for levels without consolidated metadata).
   */
  async getTileOffset(zoom: number): Promise<{ x: number; y: number }> {
    // Return cached offset if available
    const cached = this.tileOffsets.get(zoom)
    if (cached) return cached

    // EPSG:4326 doesn't need offsets
    if (this.crs !== 'EPSG:3857') {
      return { x: 0, y: 0 }
    }

    // Check if calculation is already in progress
    let pending = this._pendingOffsetCalculations.get(zoom)
    if (pending) {
      const result = await pending
      return result ?? { x: 0, y: 0 }
    }

    // Start new calculation
    pending = this._calculateTileOffsetForZoom(zoom)
    this._pendingOffsetCalculations.set(zoom, pending)

    try {
      const result = await pending
      return result ?? { x: 0, y: 0 }
    } finally {
      this._pendingOffsetCalculations.delete(zoom)
    }
  }

  /**
   * Calculate tile offset for a single zoom level (lazy, async).
   */
  private async _calculateTileOffsetForZoom(
    zoom: number
  ): Promise<{ x: number; y: number } | null> {
    const levelPath = String(zoom)
    if (!this.levels.includes(levelPath)) {
      return null
    }

    const extent = await this._getLevelExtent(levelPath)
    if (extent) {
      const offset = {
        x: this._lonToTile(extent.lonMin, zoom),
        y: this._latToTile(extent.latMax, zoom),
      }
      this.tileOffsets.set(zoom, offset)

      if (!this.xyLimits) {
        this.xyLimits = {
          xMin: extent.lonMin,
          xMax: extent.lonMax,
          yMin: extent.latMin,
          yMax: extent.latMax,
        }
      }
      return offset
    }

    // Fallback to bounds-based calculation
    if (this.xyLimits) {
      const { xMin, yMax } = this.xyLimits
      const offset = {
        x: this._lonToTile(xMin, zoom),
        y: this._latToTile(yMax, zoom),
      }
      this.tileOffsets.set(zoom, offset)
      return offset
    }

    return null
  }

  /**
   * Parse GeoTransform into spatial extent.
   */
  private _parseGeoTransformExtent(
    geoTransform: string | number[],
    shape: number[]
  ): { xMin: number; xMax: number; yMin: number; yMax: number } | null {
    let gt: number[]
    if (typeof geoTransform === 'string') {
      gt = geoTransform.split(/\s+/).map(Number)
    } else if (Array.isArray(geoTransform)) {
      gt = geoTransform.map(Number)
    } else {
      return null
    }

    if (gt.length < 6 || gt.some(isNaN)) return null

    const [xOrigin, xPixelSize, , yOrigin, , yPixelSize] = gt
    const xDimIdx = this.dimIndices.lon?.index ?? shape.length - 1
    const yDimIdx = this.dimIndices.lat?.index ?? shape.length - 2
    const width = shape[xDimIdx]
    const height = shape[yDimIdx]

    const halfPixelX = xPixelSize / 2
    const halfPixelY = yPixelSize / 2

    return {
      xMin: xOrigin + halfPixelX,
      xMax: xOrigin + width * xPixelSize - halfPixelX,
      yMax: yOrigin + halfPixelY,
      yMin: yOrigin + height * yPixelSize - halfPixelY,
    }
  }

  /**
   * Get the spatial extent for a pyramid level in lon/lat degrees.
   * First tries spatial_ref GeoTransform (fast), then falls back to coordinate arrays.
   */
  private async _getLevelExtent(levelPath: string): Promise<{
    lonMin: number
    lonMax: number
    latMin: number
    latMax: number
  } | null> {
    // Try spatial_ref first (fast - metadata only)
    const spatialRefExtent = await this._getExtentFromSpatialRef(levelPath)
    if (spatialRefExtent) {
      return this._extentToLonLat(spatialRefExtent)
    }

    // Fallback: read coordinate arrays
    const coordExtent = await this._getExtentFromCoordArrays(levelPath)
    if (coordExtent) {
      this.crs
      return this._extentToLonLat(coordExtent)
    }

    return null
  }

  /**
   * Get extent from spatial_ref GeoTransform attribute (fast - metadata only).
   * First checks consolidated metadata, then falls back to opening the array.
   */
  private async _getExtentFromSpatialRef(levelPath: string): Promise<{
    xMin: number
    xMax: number
    yMin: number
    yMax: number
  } | null> {
    if (!this.root) return null

    const spatialRefPath = `${levelPath}/spatial_ref`
    const variablePath = `${levelPath}/${this.variable}`

    try {
      // Try to get GeoTransform and shape from consolidated metadata first
      const fromConsolidated = this._getGeoTransformFromConsolidatedMetadata(
        spatialRefPath,
        variablePath
      )

      let geoTransform: string | number[] | undefined
      let shape: number[]

      if (fromConsolidated) {
        geoTransform = fromConsolidated.geoTransform
        shape = fromConsolidated.shape
      } else {
        // Fall back to opening arrays
        const spatialRefLoc = this.root.resolve(spatialRefPath)
        const spatialRefArray = await this._openArray(spatialRefLoc)

        const attrs = (
          spatialRefArray as unknown as { attrs?: Record<string, unknown> }
        ).attrs
        if (!attrs) return null

        geoTransform = attrs.GeoTransform as string | number[] | undefined
        if (!geoTransform) return null

        const variableArray = await this._openArray(
          this.root.resolve(variablePath)
        )
        shape = variableArray.shape
      }

      if (!geoTransform) return null

      return this._parseGeoTransformExtent(geoTransform, shape)
    } catch {
      return null
    }
  }

  /**
   * Try to get GeoTransform and variable shape from consolidated metadata.
   * Returns null if not available in consolidated metadata.
   */
  private _getGeoTransformFromConsolidatedMetadata(
    spatialRefPath: string,
    variablePath: string
  ): { geoTransform: string | number[]; shape: number[] } | null {
    if (!this.metadata) return null

    if (this.version === 2) {
      const v2Meta = this.metadata as ZarrV2ConsolidatedMetadata
      if (!v2Meta.metadata) return null

      // Check for spatial_ref attributes
      const spatialRefAttrs = v2Meta.metadata[`${spatialRefPath}/.zattrs`] as
        | Record<string, unknown>
        | undefined
      const geoTransform = spatialRefAttrs?.GeoTransform as
        | string
        | number[]
        | undefined
      if (!geoTransform) return null

      // Check for variable array metadata
      const variableArray = v2Meta.metadata[`${variablePath}/.zarray`] as
        | ZarrV2ArrayMetadata
        | undefined
      if (!variableArray?.shape) return null

      return { geoTransform, shape: variableArray.shape }
    }

    if (this.version === 3) {
      const v3Meta = this.metadata as ZarrV3GroupMetadata
      const consolidated = v3Meta.consolidated_metadata?.metadata
      if (!consolidated) return null

      // Check for spatial_ref metadata
      const spatialRefMeta = consolidated[spatialRefPath] as
        | ZarrV3ArrayMetadata
        | undefined
      const geoTransform = spatialRefMeta?.attributes?.GeoTransform as
        | string
        | number[]
        | undefined
      if (!geoTransform) return null

      // Check for variable array metadata
      const variableMeta = consolidated[variablePath] as
        | ZarrV3ArrayMetadata
        | undefined
      if (!variableMeta?.shape) return null

      return { geoTransform, shape: variableMeta.shape }
    }

    return null
  }

  /**
   * Get extent from coordinate arrays (slower - requires data reads).
   */
  private async _getExtentFromCoordArrays(levelPath: string): Promise<{
    xMin: number
    xMax: number
    yMin: number
    yMax: number
  } | null> {
    if (!this.root) return null

    const xCoordName =
      this.spatialDimensions.lon ?? this.dimIndices.lon?.name ?? 'x'
    const yCoordName =
      this.spatialDimensions.lat ?? this.dimIndices.lat?.name ?? 'y'

    try {
      const levelRoot = this.root.resolve(levelPath)
      const xArray = await this._openArray(levelRoot.resolve(xCoordName))
      const yArray = await this._openArray(levelRoot.resolve(yCoordName))

      type ZarrResult = { data: ArrayLike<number> }
      const xLen = xArray.shape[0]
      const yLen = yArray.shape[0]

      const [xFirst, xLast, yFirst, yLast] = (await Promise.all([
        zarr.get(xArray, [zarr.slice(0, 1)]),
        zarr.get(xArray, [zarr.slice(xLen - 1, xLen)]),
        zarr.get(yArray, [zarr.slice(0, 1)]),
        zarr.get(yArray, [zarr.slice(yLen - 1, yLen)]),
      ])) as ZarrResult[]

      return {
        xMin: Math.min(xFirst.data[0], xLast.data[0]),
        xMax: Math.max(xFirst.data[0], xLast.data[0]),
        yMin: Math.min(yFirst.data[0], yLast.data[0]),
        yMax: Math.max(yFirst.data[0], yLast.data[0]),
      }
    } catch {
      return null
    }
  }

  /**
   * Convert extent from source CRS to lon/lat degrees.
   */
  private _extentToLonLat(extent: {
    xMin: number
    xMax: number
    yMin: number
    yMax: number
  }): { lonMin: number; lonMax: number; latMin: number; latMax: number } {
    const { xMin, xMax, yMin, yMax } = extent

    // EPSG:3857 coordinates are in meters - convert to degrees
    if (this.crs === 'EPSG:3857') {
      const swCorner = this._mercatorToLonLat(xMin, yMin)
      const neCorner = this._mercatorToLonLat(xMax, yMax)
      return {
        lonMin: swCorner.lon,
        lonMax: neCorner.lon,
        latMin: swCorner.lat,
        latMax: neCorner.lat,
      }
    }

    // EPSG:4326 coordinates are already in degrees
    return { lonMin: xMin, lonMax: xMax, latMin: yMin, latMax: yMax }
  }

  /**
   * Convert Web Mercator meters to lon/lat degrees.
   */
  private _mercatorToLonLat(
    x: number,
    y: number
  ): { lon: number; lat: number } {
    const EARTH_RADIUS = 6378137
    const lon = (x / EARTH_RADIUS) * (180 / Math.PI)
    const lat =
      (Math.PI / 2 - 2 * Math.atan(Math.exp(-y / EARTH_RADIUS))) *
      (180 / Math.PI)
    return { lon, lat }
  }

  /**
   * Convert longitude to tile X coordinate for a given zoom level.
   */
  private _lonToTile(lon: number, zoom: number): number {
    return Math.floor(((lon + 180) / 360) * Math.pow(2, zoom))
  }

  /**
   * Convert latitude to tile Y coordinate for a given zoom level.
   */
  private _latToTile(lat: number, zoom: number): number {
    const MERCATOR_LAT_LIMIT = 85.0511287798066
    const clamped = Math.max(
      -MERCATOR_LAT_LIMIT,
      Math.min(MERCATOR_LAT_LIMIT, lat)
    )
    const z2 = Math.pow(2, zoom)
    return Math.floor(
      ((1 -
        Math.log(
          Math.tan((clamped * Math.PI) / 180) +
            1 / Math.cos((clamped * Math.PI) / 180)
        ) /
          Math.PI) /
        2) *
        z2
    )
  }

  /**
   * Helper to open a zarr array with version-appropriate method.
   */
  private _openArray(loc: zarr.Location<Readable>) {
    if (this.version === 2) return zarr.open.v2(loc, { kind: 'array' })
    if (this.version === 3) return zarr.open.v3(loc, { kind: 'array' })
    return zarr.open(loc, { kind: 'array' })
  }

  /**
   * Find the highest resolution level using consolidated metadata (no network requests).
   * Falls back to network requests only if metadata doesn't have shape info.
   * Users can provide explicit `bounds` to skip this detection entirely.
   */
  private async _findBoundsLevel(): Promise<string | undefined> {
    if (this.levels.length === 0 || !this.root) return undefined
    if (this.levels.length === 1) return this.levels[0]

    // Try to get shapes from consolidated metadata first (no network requests)
    const getShapeFromMetadata = (level: string): number[] | null => {
      const key = `${level}/${this.variable}`

      // V2 metadata
      const v2Meta = this.metadata as ZarrV2ConsolidatedMetadata
      if (v2Meta?.metadata?.[`${key}/.zarray`]) {
        const arrayMeta = v2Meta.metadata[`${key}/.zarray`] as {
          shape?: number[]
        }
        return arrayMeta.shape ?? null
      }

      // V3 metadata
      const v3Meta = this.metadata as ZarrV3GroupMetadata
      if (v3Meta?.consolidated_metadata?.metadata?.[key]) {
        const arrayMeta = v3Meta.consolidated_metadata.metadata[key] as {
          shape?: number[]
        }
        return arrayMeta.shape ?? null
      }

      return null
    }

    const firstLevel = this.levels[0]
    const lastLevel = this.levels[this.levels.length - 1]

    // Try metadata first
    const firstShape = getShapeFromMetadata(firstLevel)
    const lastShape = getShapeFromMetadata(lastLevel)

    if (firstShape && lastShape) {
      const firstSize = firstShape.reduce((a, b) => a * b, 1)
      const lastSize = lastShape.reduce((a, b) => a * b, 1)
      return firstSize >= lastSize ? firstLevel : lastLevel
    }

    // Fallback: network requests if metadata doesn't have shapes
    const openArray = (loc: zarr.Location<Readable>) => {
      if (this.version === 2) return zarr.open.v2(loc, { kind: 'array' })
      if (this.version === 3) return zarr.open.v3(loc, { kind: 'array' })
      return zarr.open(loc, { kind: 'array' })
    }

    try {
      const firstArray = await this._openArray(
        this.root.resolve(`${firstLevel}/${this.variable}`)
      )
      const lastArray = await this._openArray(
        this.root.resolve(`${lastLevel}/${this.variable}`)
      )

      const firstSize = firstArray.shape.reduce((a, b) => a * b, 1)
      const lastSize = lastArray.shape.reduce((a, b) => a * b, 1)
      return firstSize >= lastSize ? firstLevel : lastLevel
    } catch {
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
      if (!this._latIsAscendingUserSet) {
        this.latIsAscending = false // Tiled pyramids: row 0 = north
      }

      // For EPSG:3857 regional tile pyramids, calculate tile offsets from actual coords
      // This maps global tile coordinates to zarr array indices
      // EPSG:4326 uses extent-relative coordinates, so no offset is needed
      if (this.crs === 'EPSG:3857') {
        // Use fast path (consolidated metadata only) during initialization
        // Expensive fallback is deferred to getTileOffset() when actually needed
        this._calculateTileOffsetsFromConsolidatedMetadata()
      }
      return
    }

    // For untiled: determine what we still need to detect
    const needsBounds = !this.xyLimits
    const needsLatAscending = !this._latIsAscendingUserSet

    // If explicit bounds provided and user doesn't need latIsAscending detection, skip coord fetch
    // (respects user intent to avoid coord reads by providing bounds)
    if (!needsBounds && !needsLatAscending) {
      return
    }

    // Can't fetch coords without dimension info - default already set
    if (!this.dimIndices.lon || !this.dimIndices.lat || !this.root) {
      return
    }

    try {
      const boundsLevel = await this._findBoundsLevel()
      const levelRoot = boundsLevel ? this.root.resolve(boundsLevel) : this.root

      const lonName = this.spatialDimensions.lon ?? this.dimIndices.lon.name
      const latName = this.spatialDimensions.lat ?? this.dimIndices.lat.name

      // Helper to open array with correct zarr version
      const openArray = (loc: zarr.Location<Readable>) => {
        if (this.version === 2) return zarr.open.v2(loc, { kind: 'array' })
        if (this.version === 3) return zarr.open.v3(loc, { kind: 'array' })
        return zarr.open(loc, { kind: 'array' })
      }

      // Find the HIGHEST RESOLUTION coordinate array path from consolidated metadata.
      // This ensures we get the most accurate bounds regardless of level naming conventions.
      const findCoordPath = (dimName: string): string | null => {
        if (!this.metadata) return null

        type CoordCandidate = { path: string; size: number }
        const candidates: CoordCandidate[] = []

        // V2: keys are like "lat/.zarray" or "surface/lat/.zarray"
        const v2Meta = this.metadata as ZarrV2ConsolidatedMetadata
        if (v2Meta.metadata) {
          const suffix = `/${dimName}/.zarray`
          const rootKey = `${dimName}/.zarray`
          for (const key of Object.keys(v2Meta.metadata)) {
            if (key === rootKey || key.endsWith(suffix)) {
              const meta = v2Meta.metadata[key] as { shape?: number[] }
              const size = meta.shape?.[0] ?? 0
              candidates.push({
                path: key.slice(0, -'/.zarray'.length),
                size,
              })
            }
          }
        }

        // V3: keys are like "lat" or "surface/lat" with node_type: 'array'
        const v3Meta = this.metadata as ZarrV3GroupMetadata
        if (v3Meta.consolidated_metadata?.metadata) {
          const suffix = `/${dimName}`
          for (const [key, value] of Object.entries(
            v3Meta.consolidated_metadata.metadata
          )) {
            if (
              (key === dimName || key.endsWith(suffix)) &&
              value.node_type === 'array'
            ) {
              const size = (value as { shape?: number[] }).shape?.[0] ?? 0
              candidates.push({ path: key, size })
            }
          }
        }

        // Return the highest resolution (largest size) coordinate array
        if (candidates.length === 0) return null

        const pickLargest = (list: CoordCandidate[]) => {
          if (list.length === 0) return null
          const sorted = [...list].sort((a, b) => b.size - a.size)
          return sorted[0].path
        }

        // Prefer coord arrays within the bounds level to avoid cross-variable grids.
        // Fallback to root-level coords, then the global maximum.
        if (boundsLevel) {
          const levelPrefix = `${boundsLevel}/`
          const levelCandidates = candidates.filter((c) =>
            c.path.startsWith(levelPrefix)
          )
          const levelPick = pickLargest(levelCandidates)
          if (levelPick) return levelPick

          const rootCandidates = candidates.filter((c) => !c.path.includes('/'))
          const rootPick = pickLargest(rootCandidates)
          if (rootPick) return rootPick
        } else if (this.variable) {
          const varCandidates = candidates.filter((c) =>
            c.path.startsWith(`${this.variable}/`)
          )
          const varPick = pickLargest(varCandidates)
          if (varPick) return varPick
        }

        return pickLargest(candidates)
      }

      // Find highest resolution coordinate arrays from metadata (handles all multiscale conventions)
      const xPath = findCoordPath(lonName)
      const yPath = findCoordPath(latName)

      // Open coord arrays: use metadata path if found, otherwise try levelRoot
      const xarr = await openArray(
        xPath ? this.root!.resolve(xPath) : levelRoot.resolve(lonName)
      )
      const yarr = await openArray(
        yPath ? this.root!.resolve(yPath) : levelRoot.resolve(latName)
      )

      const xLen = xarr.shape[0]
      const yLen = yarr.shape[0]

      type ZarrResult = { data: ArrayLike<number> }
      const [xFirstTwo, xLast, yFirstTwo, yLast] = (await Promise.all([
        zarr.get(xarr, [zarr.slice(0, 2)]),
        zarr.get(xarr, [zarr.slice(xLen - 1, xLen)]),
        zarr.get(yarr, [zarr.slice(0, 2)]),
        zarr.get(yarr, [zarr.slice(yLen - 1, yLen)]),
      ])) as ZarrResult[]

      const x0 = xFirstTwo.data[0]
      const x1 = xFirstTwo.data[1] ?? x0
      const xN = xLast.data[0]
      const y0 = yFirstTwo.data[0]
      const y1 = yFirstTwo.data[1]
      const yN = yLast.data[0]

      // Detect latIsAscending from first two y values
      const detectedLatAscending = y1 > y0
      if (needsLatAscending) {
        this.latIsAscending = detectedLatAscending
      }

      // Coordinate extents from coordinate arrays (these are pixel centers)
      const coordXMin = Math.min(x0, xN)
      const coordXMax = Math.max(x0, xN)
      const coordYMin = Math.min(y0, yN)
      const coordYMax = Math.max(y0, yN)

      // Use coordinate array's own spacing for half-pixel expansion.
      // Coords represent pixel centers; extent is [first - halfPixel, last + halfPixel]
      const dx = Math.abs(x1 - x0)
      const dy = Math.abs(y1 - y0)

      // Apply half-pixel expansion (coords are pixel centers, we need edge bounds)
      const xMin = coordXMin - (Number.isFinite(dx) ? dx / 2 : 0)
      const xMax = coordXMax + (Number.isFinite(dx) ? dx / 2 : 0)
      const yMin = coordYMin - (Number.isFinite(dy) ? dy / 2 : 0)
      const yMax = coordYMax + (Number.isFinite(dy) ? dy / 2 : 0)

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
      if (needsLatAscending) {
        console.warn(
          `[zarr-layer] Could not detect latIsAscending from coordinates. ` +
            `Defaulting to true (row 0 = south). Set explicitly if data appears flipped.`
        )
      }
    }

    // Infer CRS from bounds if not explicitly set
    // Only classify as meters if clearly outside degree range (> 360)
    // This handles both [-180, 180] and [0, 360] degree conventions
    // Applies to untiled multiscales and single-level datasets (multiscaleType === 'none')
    if (
      (this.multiscaleType === 'untiled' || this.multiscaleType === 'none') &&
      !this._crsFromMetadata &&
      !this._crsOverride &&
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

          // Extract shape/chunks/dtype/fillValue/scaleFactor/addOffset from consolidated metadata
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

              // Extract dtype and fillValue
              if (arrayMeta.data_type) {
                untiledLevel.dtype = arrayMeta.data_type
              }
              if (arrayMeta.fill_value !== undefined) {
                untiledLevel.fillValue = this.normalizeFillValue(
                  arrayMeta.fill_value
                )
              }

              // Float data typically stores already-physical values (e.g., pyramid levels
              // created by averaging). Integer data stores raw counts needing conversion.
              // For heterogeneous pyramids like Sentinel-2, lower-res float levels inherit
              // scale_factor attributes but shouldn't have them re-applied.
              const isFloatData =
                arrayMeta.data_type?.includes('float') ||
                arrayMeta.data_type === 'float32' ||
                arrayMeta.data_type === 'float64'

              if (isFloatData) {
                // Float data: assume already physical, use 1/0
                untiledLevel.scaleFactor = 1
                untiledLevel.addOffset = 0
              } else if (arrayMeta.attributes) {
                // Integer data: apply scale_factor/add_offset if present
                if (arrayMeta.attributes.scale_factor !== undefined) {
                  untiledLevel.scaleFactor = arrayMeta.attributes
                    .scale_factor as number
                }
                if (arrayMeta.attributes.add_offset !== undefined) {
                  untiledLevel.addOffset = arrayMeta.attributes
                    .add_offset as number
                }
              }
              // If non-float without attributes, leave undefined for dataset-level fallback
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
    if (metadata.crs && !this._crsOverride) {
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
