import * as zarr from 'zarrita'
import type { Readable } from '@zarrita/storage'
import type {
  SpatialDimensions,
  DimIndicesProps,
  CRS,
  UntiledLevel,
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

type ConsolidatedStore = zarr.Listable<zarr.FetchStore>
type ZarrStoreType = zarr.FetchStore | ConsolidatedStore

interface ZarrStoreOptions {
  source: string
  version?: 2 | 3 | null
  variable: string
  spatialDimensions?: SpatialDimensions
  bounds?: [number, number, number, number]
  coordinateKeys?: string[]
  latIsAscending?: boolean | null
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
  private explicitBounds: [number, number, number, number] | null
  coordinateKeys: string[]

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

    this.initialized = this._initialize()
  }

  private async _initialize(): Promise<this> {
    const storeCacheKey = `${this.source}:${this.version ?? 'auto'}`
    let storeHandle = ZarrStore._storeCache.get(storeCacheKey)

    if (!storeHandle) {
      const baseStore = new zarr.FetchStore(this.source)
      if (this.version === 3) {
        storeHandle = Promise.resolve(baseStore)
      } else {
        storeHandle = zarr.tryWithConsolidated(baseStore).catch(() => baseStore)
      }
      ZarrStore._storeCache.set(storeCacheKey, storeHandle)
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

    await this._loadXYLimits()
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
   * Get metadata (shape, chunks) for a specific untiled level.
   * Used by UntiledMode to determine chunk boundaries for viewport-based loading.
   */
  async getUntiledLevelMetadata(
    levelAsset: string
  ): Promise<{ shape: number[]; chunks: number[] }> {
    const array = await this.getLevelArray(levelAsset)
    return {
      shape: array.shape,
      chunks: array.chunks,
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
    let zmetadata = ZarrStore._cache.get(cacheKey) as
      | ZarrV2ConsolidatedMetadata
      | undefined
    if (!zmetadata) {
      if (this.isConsolidatedStore(this.store)) {
        const rootZattrsBytes = await this.store.get('/.zattrs')
        const rootZattrs = rootZattrsBytes ? decodeJSON(rootZattrsBytes) : {}
        zmetadata = { metadata: { '.zattrs': rootZattrs } }
        ZarrStore._cache.set(cacheKey, zmetadata)
      } else {
        try {
          zmetadata = (await this._getJSON(
            '/.zmetadata'
          )) as ZarrV2ConsolidatedMetadata
          ZarrStore._cache.set(cacheKey, zmetadata)
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
    let metadata = ZarrStore._cache.get(metadataCacheKey) as
      | ZarrV3GroupMetadata
      | undefined
    if (!metadata) {
      metadata = (await this._getJSON('/zarr.json')) as ZarrV3GroupMetadata
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
    let arrayMetadata = ZarrStore._cache.get(arrayCacheKey) as
      | ZarrV3ArrayMetadata
      | undefined
    if (!arrayMetadata) {
      arrayMetadata = (await this._getJSON(
        `/${arrayKey}/zarr.json`
      )) as ZarrV3ArrayMetadata
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

    const coordinates: Record<string, zarr.Array<zarr.DataType, Readable>> = {}

    // Load coordinate arrays for spatial dimensions (used for bounds calculation)
    for (const dimName of this.dimensions) {
      if (
        ['x', 'lon', 'longitude', 'y', 'lat', 'latitude'].includes(
          dimName.toLowerCase()
        )
      ) {
        try {
          const coordKey = this.coarsestLevel
            ? `${this.coarsestLevel}/${dimName}`
            : dimName
          const coordArray = await this._getArray(coordKey)
          coordinates[dimName] = coordArray
        } catch (err) {
          console.debug(
            `Could not load coordinate array for '${dimName}':`,
            err
          )
        }
      }
    }

    // Use identifyDimensionIndices for spatial dimensions (lat, lon)
    this.dimIndices = identifyDimensionIndices(
      this.dimensions,
      this.spatialDimensions,
      coordinates
    )

    // Add ALL dimensions to dimIndices so selectors can reference them by name
    // (e.g., 'time', 'level', etc. - not just lat/lon)
    for (let i = 0; i < this.dimensions.length; i++) {
      const dimName = this.dimensions[i]
      // Skip if already added (e.g., 'lat' was already mapped with its coordinate array)
      if (this.dimIndices[dimName] || this.dimIndices[dimName.toLowerCase()]) {
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

  private async _loadXYLimits() {
    if (!this.dimIndices.lon || !this.dimIndices.lat || !this.root) return

    try {
      // Use coarsest level for bounds detection:
      // - Tiled pyramids: level 0 is coarsest
      // - Untiled multiscale: last level (maxLevelIndex) is coarsest
      const coarsestLevel =
        this.multiscaleType === 'untiled'
          ? this.levels[this.maxLevelIndex]
          : this.levels[0]
      const levelRoot =
        this.levels.length > 0 && coarsestLevel
          ? this.root.resolve(coarsestLevel)
          : this.root

      const openArray = (loc: zarr.Location<Readable>) => {
        if (this.version === 2) {
          return zarr.open.v2(loc, { kind: 'array' })
        } else if (this.version === 3) {
          return zarr.open.v3(loc, { kind: 'array' })
        }
        return zarr.open(loc, { kind: 'array' })
      }

      const lonName = this.spatialDimensions.lon ?? this.dimIndices.lon.name
      const latName = this.spatialDimensions.lat ?? this.dimIndices.lat.name

      const xarr =
        this.dimIndices.lon.array ||
        (await openArray(levelRoot.resolve(lonName)))
      const yarr =
        this.dimIndices.lat.array ||
        (await openArray(levelRoot.resolve(latName)))

      const xdata = await zarr.get(xarr)
      const ydata = await zarr.get(yarr)

      const xValues = xdata.data as ArrayLike<number>
      const yValues = ydata.data as ArrayLike<number>

      // Determine ascending from first two values
      if (this.latIsAscending === null && yValues.length >= 2) {
        this.latIsAscending = yValues[1] > yValues[0]
      }

      // Use for-loop instead of spread to avoid stack overflow with large arrays
      let xMin = xValues[0],
        xMax = xValues[0]
      for (let i = 1; i < xValues.length; i++) {
        const v = xValues[i]
        if (v < xMin) xMin = v
        if (v > xMax) xMax = v
      }
      let yMin = yValues[0],
        yMax = yValues[0]
      for (let i = 1; i < yValues.length; i++) {
        const v = yValues[i]
        if (v < yMin) yMin = v
        if (v > yMax) yMax = v
      }

      this.xyLimits = { xMin, xMax, yMin, yMax }
    } catch (err) {
      // Use explicit bounds if provided, otherwise throw
      if (this.explicitBounds) {
        const [west, south, east, north] = this.explicitBounds
        this.xyLimits = {
          xMin: west,
          xMax: east,
          yMin: south,
          yMax: north,
        }
        console.debug(
          'Using explicit bounds for XY limits:',
          this.explicitBounds
        )
      } else {
        throw new Error(
          `Failed to load XY limits from coordinate arrays. ` +
            `Provide explicit bounds via the 'bounds' option. ` +
            `Original error: ${err instanceof Error ? err.message : err}`
        )
      }
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

    // Detect zarr-conventions/multiscales format (has 'layout' key)
    if ('layout' in multiscales && Array.isArray(multiscales.layout)) {
      return this._parseUntiledMultiscale(multiscales)
    }

    // OME-NGFF style format (array with 'datasets' key)
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
        this.untiledLevels = levels.map((level) => ({
          asset: level,
          scale: [1.0, 1.0] as [number, number],
          translation: [0.0, 0.0] as [number, number],
        }))
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

    // Build untiledLevels with transform info
    this.untiledLevels = layout.map((entry) => ({
      asset: entry.asset,
      scale: entry.transform?.scale ?? [1.0, 1.0],
      translation: entry.transform?.translation ?? [0.0, 0.0],
    }))

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
