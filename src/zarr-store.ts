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
import {
  parseZarrMetadata,
  normalizeFillValue,
  findCoordinatePath,
  findHighestResolutionLevel,
} from '../packages/zarr-metadata/src'
import type {
  ZarrMultiscaleMetadata,
  ZarrV2ConsolidatedMetadata,
  ZarrV3GroupMetadata,
  ZarrV3ArrayMetadata,
} from '../packages/zarr-metadata/src/types'

const textDecoder = new TextDecoder()

const decodeJSON = (bytes: Uint8Array | undefined): unknown => {
  if (!bytes) return null
  return JSON.parse(textDecoder.decode(bytes))
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
 * Handle fetch response, returning bytes or undefined for 404/403.
 * 403 is treated as "not found" for S3/CloudFront compatibility: these
 * services return 403 (not 404) for missing or inaccessible paths.
 */
const handleResponse = async (
  response: Response
): Promise<Uint8Array | undefined> => {
  if (response.status === 404 || response.status === 403) return undefined
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
  | Readable<unknown>
  | AsyncReadable<unknown>

interface ZarrStoreOptions {
  /** URL to Zarr store. Required unless customStore is provided. */
  source?: string
  version?: 2 | 3 | null
  variable: string
  spatialDimensions?: SpatialDimensions
  bounds?: Bounds
  crs?: string
  coordinateKeys?: string[]
  latIsAscending?: boolean | null
  proj4?: string
  transformRequest?: TransformRequest
  /** Custom store to use instead of FetchStore. When provided, source becomes optional. */
  customStore?: Readable<unknown> | AsyncReadable<unknown>
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
  coordinateScale: number | null
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
  private customStore?: Readable<unknown> | AsyncReadable<unknown>

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
  coordinateScale: number | null = null
  private _crsFromMetadata: boolean = false // Track if CRS was explicitly set from metadata
  private _crsOverride: boolean = false // Track if CRS was explicitly set by user

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
    customStore,
  }: ZarrStoreOptions) {
    if (!source && !customStore) {
      throw new Error('source is required when customStore is not provided')
    }
    if (!variable) {
      throw new Error('variable is a required parameter')
    }
    this.source = source ?? 'custom-store'
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
    this.customStore = customStore

    this.initialized = this._initialize()
  }

  private async _initialize(): Promise<this> {
    const storeCacheKey = `${this.source}:${this.version ?? 'auto'}`
    let storeHandle: Promise<ZarrStoreType> | undefined

    if (this.customStore) {
      // Validate that custom store implements required Readable interface
      if (typeof this.customStore.get !== 'function') {
        throw new Error(
          'customStore must implement Readable interface with get() method'
        )
      }
      // Use custom store directly (e.g., IcechunkStore)
      storeHandle = Promise.resolve(this.customStore as ZarrStoreType)
    } else if (this.transformRequest) {
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

    // Step 1: Load raw metadata (fetch + cache)
    const preloadedMetadata = await this._loadRawMetadata()

    // Step 2: Parse via zarr-metadata package
    const parsed = await parseZarrMetadata(this.store, {
      variable: this.variable,
      version: this.version ?? undefined,
      spatialDimensions: this.spatialDimensions
        ? { lat: this.spatialDimensions.lat, lon: this.spatialDimensions.lon }
        : undefined,
      crs: this._crsOverride ? this.crs : undefined,
      proj4: this.proj4 ?? undefined,
      sourceUrl: this.source,
      preloadedMetadata,
    })

    // Step 3: Map parsed output to store properties
    this._applyParsedMetadata(parsed)

    // Step 4: Build full dim indices (all dims, not just spatial)
    await this._computeDimIndices()

    // Step 5: Load spatial metadata (bounds normalization, CRS inference)
    await this._loadSpatialMetadata()

    // Step 6: Load coordinate arrays for selectors
    await this._loadCoordinates()

    return this
  }

  /**
   * Load raw metadata from the store, using cache when possible.
   * Returns the preloaded metadata object to pass to parseZarrMetadata.
   */
  private async _loadRawMetadata(): Promise<
    ZarrV2ConsolidatedMetadata | ZarrV3GroupMetadata
  > {
    const bypassCache = !!(this.transformRequest || this.customStore)

    // Try V3 first (or forced), then V2
    if (this.version === 3) {
      return this._loadRawV3(bypassCache)
    }
    if (this.version === 2) {
      return this._loadRawV2(bypassCache)
    }

    // Auto-detect: try V3 first, then V2
    try {
      return await this._loadRawV3(bypassCache)
    } catch {
      return this._loadRawV2(bypassCache)
    }
  }

  private async _loadRawV2(
    bypassCache: boolean
  ): Promise<ZarrV2ConsolidatedMetadata> {
    const cacheKey = `v2:${this.source}`
    let zmetadata = bypassCache
      ? undefined
      : (ZarrStore._cache.get(cacheKey) as
          | ZarrV2ConsolidatedMetadata
          | undefined)

    if (!zmetadata) {
      if (this.isConsolidatedStore(this.store)) {
        const rootZattrsBytes = await this.store.get('/.zattrs')
        const rootZattrs = rootZattrsBytes ? decodeJSON(rootZattrsBytes) : {}
        zmetadata = { metadata: { '.zattrs': rootZattrs } }
      } else {
        try {
          zmetadata = (await this._getJSON(
            '/.zmetadata'
          )) as ZarrV2ConsolidatedMetadata
        } catch {
          try {
            const zattrs = await this._getJSON('/.zattrs')
            zmetadata = { metadata: { '.zattrs': zattrs } }
          } catch {
            zmetadata = { metadata: { '.zattrs': {} } }
          }
        }
      }
      if (!bypassCache) ZarrStore._cache.set(cacheKey, zmetadata)
    }

    this.metadata = zmetadata
    return zmetadata
  }

  private async _loadRawV3(bypassCache: boolean): Promise<ZarrV3GroupMetadata> {
    const cacheKey = `v3:${this.source}`
    let metadata = bypassCache
      ? undefined
      : (ZarrStore._cache.get(cacheKey) as ZarrV3GroupMetadata | undefined)

    if (!metadata) {
      metadata = (await this._getJSON('/zarr.json')) as ZarrV3GroupMetadata
      if (!bypassCache) {
        ZarrStore._cache.set(cacheKey, metadata)

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
    return metadata
  }

  /**
   * Map parseZarrMetadata output to ZarrStore properties.
   */
  private _applyParsedMetadata(parsed: ZarrMultiscaleMetadata): void {
    // Version
    this.version = parsed.version

    // Format → multiscaleType
    switch (parsed.format) {
      case 'ndpyramid-tiled':
        this.multiscaleType = 'tiled'
        break
      case 'single-level':
        this.multiscaleType = 'none'
        break
      default:
        // zarr-conventions, ome-ngff → untiled
        this.multiscaleType = 'untiled'
        break
    }

    // For OME-NGFF without pixels_per_tile but with multiple levels,
    // it could be tiled or untiled. The package returns 'ome-ngff' for both.
    // Check if levels have tileSize (pixels_per_tile was present).
    if (parsed.format === 'ome-ngff' && parsed.tileSize) {
      this.multiscaleType = 'tiled'
    }

    // Single-level datasets still use untiled rendering
    if (this.multiscaleType === 'none') {
      this.multiscaleType = 'untiled'
    }

    // Base level properties
    this.shape = parsed.base.shape
    this.chunks = parsed.base.chunks
    this.dtype = parsed.base.dtype
    this.fill_value = parsed.base.fillValue
    this.dimensions = parsed.base.dimensions
    this.scaleFactor = parsed.base.scaleFactor ?? 1
    this.addOffset = parsed.base.addOffset ?? 0

    // Levels
    this.levels = parsed.levels.map((l) => l.path)
    this.maxLevelIndex = Math.max(0, parsed.levels.length - 1)
    this.tileSize = parsed.tileSize ?? 128

    // Build untiledLevels from parsed levels
    if (this.multiscaleType === 'untiled') {
      this.untiledLevels = parsed.levels.map((level) => {
        const untiledLevel: UntiledLevel = {
          asset: level.path,
          scale: level.resolution,
          translation: level.translation ?? [0.0, 0.0],
        }

        if (level.shape.length > 0) {
          untiledLevel.shape = level.shape
        }
        if (level.chunks.length > 0) {
          untiledLevel.chunks = level.chunks
        }

        // Float data scaling: float data is already physical, use 1/0
        const dtype = level.dtype ?? parsed.base.dtype
        const isFloatData =
          dtype?.includes('float') || dtype === 'float32' || dtype === 'float64'

        if (isFloatData) {
          untiledLevel.scaleFactor = 1
          untiledLevel.addOffset = 0
        } else {
          if (level.scaleFactor !== undefined) {
            untiledLevel.scaleFactor = level.scaleFactor
          }
          if (level.addOffset !== undefined) {
            untiledLevel.addOffset = level.addOffset
          }
        }

        if (level.dtype !== undefined) {
          untiledLevel.dtype = level.dtype
        }
        if (level.fillValue !== undefined) {
          untiledLevel.fillValue = level.fillValue
        }

        return untiledLevel
      })
    }

    // Tiled pyramids without explicit CRS default to Web Mercator
    // (slippy-map tiles are always Mercator-projected)
    if (!this._crsOverride && this.multiscaleType === 'tiled' && !parsed.crs) {
      this.crs = 'EPSG:3857'
      this._crsFromMetadata = true
    }

    // CRS (respect user override)
    if (!this._crsOverride && parsed.crs) {
      const code = parsed.crs.code?.toUpperCase()
      if (code === 'EPSG:4326' || code === 'EPSG:3857') {
        this.crs = code as CRS
        this._crsFromMetadata = true
      } else if (parsed.crs.proj4def) {
        // Non-standard CRS with proj4 definition (e.g., from grid_mapping)
        this.proj4 = parsed.crs.proj4def
        this.crs = 'custom' as CRS
        this._crsFromMetadata = true
      } else if (code) {
        // Non-standard CRS code without proj4 — warn user
        if (!this.proj4) {
          console.warn(
            `[zarr-layer] Detected ${code} from metadata but no proj4 definition available. ` +
              `Set explicitly to enable reprojection: proj4: '<proj4 string for ${code}>'`
          )
        }
      }
      if (parsed.crs.coordinateScale) {
        this.coordinateScale = parsed.crs.coordinateScale
      }
    }

    // Apply spatial bounds from metadata (spatial:bbox or spatial:transform).
    // When present, _loadSpatialMetadata will skip the coordinate array fetch.
    if (parsed.bounds && !this.explicitBounds) {
      const [xMin, yMin, xMax, yMax] = parsed.bounds
      this.xyLimits = { xMin, yMin, xMax, yMax }
    }
    if (parsed.latIsAscending !== null && !this._latIsAscendingUserSet) {
      this.latIsAscending = parsed.latIsAscending
    }
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
      coordinateScale: this.coordinateScale,
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
        fillValue = normalizeFillValue(meta.fill_value)

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
        fillValue = normalizeFillValue(zarray.fill_value)
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
      // Use package's findHighestResolutionLevel instead of inline _findBoundsLevel
      const boundsLevel = findHighestResolutionLevel(
        this.levels,
        this.variable,
        this.metadata
      )
      const levelRoot = boundsLevel ? this.root.resolve(boundsLevel) : this.root

      const openArray = (loc: zarr.Location<Readable>) => {
        if (this.version === 2) return zarr.open.v2(loc, { kind: 'array' })
        if (this.version === 3) return zarr.open.v3(loc, { kind: 'array' })
        return zarr.open(loc, { kind: 'array' })
      }

      const lonName = this.spatialDimensions.lon ?? this.dimIndices.lon.name
      const latName = this.spatialDimensions.lat ?? this.dimIndices.lat.name

      // Use package's findCoordinatePath instead of inline closure
      const xPath = findCoordinatePath(
        lonName,
        this.metadata,
        boundsLevel ?? undefined,
        this.variable
      )
      const yPath = findCoordinatePath(
        latName,
        this.metadata,
        boundsLevel ?? undefined,
        this.variable
      )

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
      let xMin = coordXMin - (Number.isFinite(dx) ? dx / 2 : 0)
      let xMax = coordXMax + (Number.isFinite(dx) ? dx / 2 : 0)
      const yMin = coordYMin - (Number.isFinite(dy) ? dy / 2 : 0)
      const yMax = coordYMax + (Number.isFinite(dy) ? dy / 2 : 0)

      // Normalize 0–360° longitude convention to -180–180°.
      // Only applies when both bounds are > 180 (clearly 0–360° data, not
      // projected meters) and within the degree range (xMax <= 360).
      if (
        xMin > 180 &&
        xMax > 180 &&
        xMax <= 360 &&
        !this.proj4 &&
        this.crs !== 'EPSG:3857'
      ) {
        xMin -= 360
        xMax -= 360
      }

      // For global datasets, snap bounds to exactly ±180 to avoid antimeridian
      // seams caused by grid alignment not landing on ±180. A truly global grid
      // has extent = N * dx = 360°; use dx/2 tolerance for float32 precision.
      // A dataset one cell short has extent = 360 - dx, which fails the check.
      const lonExtent = xMax - xMin
      if (Number.isFinite(dx) && Math.abs(lonExtent - 360) < dx / 2) {
        if (Math.abs(xMin + 180) < dx) xMin = -180
        if (Math.abs(xMax - 180) < dx) xMax = 180
      }

      if (needsBounds) {
        this.xyLimits = { xMin, xMax, yMin, yMax }
      }

      // Apply coordinate scaling for projections where coordinate arrays
      // use different units than the proj4 definition expects.
      // E.g., GOES geostationary stores scanning angles in radians,
      // but proj4 +proj=geos expects meters.
      // Only scale auto-detected bounds — explicit bounds are assumed correct.
      if (needsBounds && this.coordinateScale && this.xyLimits) {
        const s = this.coordinateScale
        this.xyLimits = {
          xMin: this.xyLimits.xMin * s,
          yMin: this.xyLimits.yMin * s,
          xMax: this.xyLimits.xMax * s,
          yMax: this.xyLimits.yMax * s,
        }
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
    if (!this._crsFromMetadata && !this._crsOverride && this.xyLimits) {
      const maxAbsX = Math.max(
        Math.abs(this.xyLimits.xMin),
        Math.abs(this.xyLimits.xMax)
      )
      if (maxAbsX > 360) {
        this.crs = 'EPSG:3857'
      }
    }
  }

  static clearCache() {
    ZarrStore._cache.clear()
    ZarrStore._storeCache.clear()
  }
}
