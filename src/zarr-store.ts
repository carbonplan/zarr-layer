import * as zarr from 'zarrita'
import type { Readable } from '@zarrita/storage'
import type {
  Bounds,
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
  bounds?: Bounds
  coordinateKeys?: string[]
  latIsAscending?: boolean | null
  proj4?: string
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
  tileOffsets: Map<number, { x: number; y: number }>
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
    coordinateKeys = [],
    latIsAscending = null,
    proj4,
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
   * Find the highest resolution level for untiled bounds detection.
   * Coarse levels have larger pixels, so padding offsets are more significant.
   * Trade-off is higher bandwidth. Users can provide explicit `bounds` to skip.
   */
  private async _findBoundsLevel(): Promise<string | undefined> {
    if (this.levels.length === 0 || !this.root) return undefined
    if (this.levels.length === 1) return this.levels[0]

    // Compare first and last levels to determine which has higher resolution
    const firstLevel = this.levels[0]
    const lastLevel = this.levels[this.levels.length - 1]

    try {
      const firstArray = await this._openArray(
        this.root.resolve(`${firstLevel}/${this.variable}`)
      )
      const lastArray = await this._openArray(
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

      const lonName = this.spatialDimensions.lon ?? this.dimIndices.lon.name
      const latName = this.spatialDimensions.lat ?? this.dimIndices.lat.name

      const xarr = await this._openArray(levelRoot.resolve(lonName))
      const yarr = await this._openArray(levelRoot.resolve(latName))

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
