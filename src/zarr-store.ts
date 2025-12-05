import * as zarr from 'zarrita'
import type { Readable } from '@zarrita/storage'
import type {
  DimensionNamesProps,
  DimIndicesProps,
  XYLimitsProps,
  CRS,
} from './types'
import { identifyDimensionIndices } from './zarr-utils'

const textDecoder = new TextDecoder()

const decodeJSON = (bytes: Uint8Array | undefined): unknown => {
  if (!bytes) return null
  return JSON.parse(textDecoder.decode(bytes))
}

interface PyramidMetadata {
  levels: string[]
  maxZoom: number
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
  multiscales?: Multiscale[]
  scale_factor?: number
  add_offset?: number
  _FillValue?: number
  missing_value?: number
}

interface ZarrV3GroupMetadata {
  zarr_format: 3
  node_type: 'group'
  attributes?: {
    multiscales?: Multiscale[]
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
    configuration?: {
      chunk_shape?: number[]
    }
  }
  chunks?: number[]
  codecs?: Array<{
    name: string
    configuration?: {
      chunk_shape?: number[]
    }
  }>
  attributes?: {
    _ARRAY_DIMENSIONS?: string[]
    scale_factor?: number
    add_offset?: number
    _FillValue?: number
    missing_value?: number
  }
}

type ConsolidatedStore = zarr.Listable<zarr.FetchStore>
type ZarrStoreType = zarr.FetchStore | ConsolidatedStore

interface ZarrStoreOptions {
  source: string
  version?: 2 | 3 | null
  variable: string
  dimensionNames?: DimensionNamesProps
  coordinateKeys?: string[]
}

interface StoreDescription {
  metadata: ZarrV2ConsolidatedMetadata | ZarrV3GroupMetadata | null
  dimensions: string[]
  shape: number[]
  chunks: number[]
  fill_value: number | null
  dtype: string | null
  levels: string[]
  maxZoom: number
  tileSize: number
  crs: CRS
  dimIndices: DimIndicesProps
  xyLimits: XYLimitsProps | null
  scaleFactor: number
  addOffset: number
  coordinates: Record<string, (string | number)[]>
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
  dimensionNames: DimensionNamesProps
  coordinateKeys: string[]

  metadata: ZarrV2ConsolidatedMetadata | ZarrV3GroupMetadata | null = null
  arrayMetadata: ZarrV3ArrayMetadata | null = null
  dimensions: string[] = []
  shape: number[] = []
  chunks: number[] = []
  fill_value: number | null = null
  dtype: string | null = null
  levels: string[] = []
  maxZoom: number = 0
  tileSize: number = 128
  crs: CRS = 'EPSG:4326'
  dimIndices: DimIndicesProps = {}
  xyLimits: XYLimitsProps | null = null
  scaleFactor: number = 1
  addOffset: number = 0
  coordinates: Record<string, (string | number)[]> = {}

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
    dimensionNames = {},
    coordinateKeys = [],
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
    this.dimensionNames = dimensionNames
    this.coordinateKeys = coordinateKeys

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
      maxZoom: this.maxZoom,
      tileSize: this.tileSize,
      crs: this.crs,
      dimIndices: this.dimIndices,
      xyLimits: this.xyLimits,
      scaleFactor: this.scaleFactor,
      addOffset: this.addOffset,
      coordinates: this.coordinates,
    }
  }

  async getChunk(
    level: string,
    chunkIndices: number[]
  ): Promise<zarr.Chunk<zarr.DataType>> {
    const key = `${level}/${this.variable}`
    const array = await this._getArray(key)
    return array.getChunk(chunkIndices)
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
      this.maxZoom = pyramid.maxZoom
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
    this.fill_value = zarray?.fill_value ?? null
    if (this.fill_value === null && zattrs) {
      if (zattrs._FillValue !== undefined) {
        this.fill_value = zattrs._FillValue
      } else if (zattrs.missing_value !== undefined) {
        this.fill_value = zattrs.missing_value
      }
    }
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
      this.maxZoom = pyramid.maxZoom
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

    this.dimensions =
      arrayMetadata.attributes?._ARRAY_DIMENSIONS ||
      arrayMetadata.dimension_names ||
      []
    this.shape = arrayMetadata.shape

    const isSharded = arrayMetadata.codecs?.[0]?.name === 'sharding_indexed'
    this.chunks = isSharded
      ? arrayMetadata.codecs?.[0]?.configuration?.chunk_shape || this.shape
      : arrayMetadata.chunk_grid?.configuration?.chunk_shape ||
        arrayMetadata.chunks ||
        this.shape

    this.fill_value = arrayMetadata.fill_value
    if (this.fill_value === null && arrayMetadata.attributes) {
      if (arrayMetadata.attributes._FillValue !== undefined) {
        this.fill_value = arrayMetadata.attributes._FillValue
      } else if (arrayMetadata.attributes.missing_value !== undefined) {
        this.fill_value = arrayMetadata.attributes.missing_value
      }
    }
    this.dtype = arrayMetadata.data_type || null
    this.scaleFactor = arrayMetadata.attributes?.scale_factor ?? 1
    this.addOffset = arrayMetadata.attributes?.add_offset ?? 0

    await this._computeDimIndices()
  }

  private async _computeDimIndices() {
    if (this.dimensions.length === 0) return

    const coordinates: Record<string, zarr.Array<zarr.DataType, Readable>> = {}
    for (const dimName of this.dimensions) {
      if (
        !['x', 'lon', 'longitude', 'y', 'lat', 'latitude'].includes(
          dimName.toLowerCase()
        )
      ) {
        continue
      }
      try {
        const coordKey =
          this.levels.length > 0 ? `${this.levels[0]}/${dimName}` : dimName
        const coordArray = await this._getArray(coordKey)
        coordinates[dimName] = coordArray
      } catch {}
    }

    this.dimIndices = identifyDimensionIndices(
      this.dimensions,
      this.dimensionNames,
      coordinates
    )
  }

  private async _loadXYLimits() {
    if (!this.dimIndices.lon || !this.dimIndices.lat || !this.root) return

    try {
      const levelRoot =
        this.levels.length > 0 ? this.root.resolve(this.levels[0]) : this.root

      const openArray = (loc: zarr.Location<Readable>) => {
        if (this.version === 2) {
          return zarr.open.v2(loc, { kind: 'array' })
        } else if (this.version === 3) {
          return zarr.open.v3(loc, { kind: 'array' })
        }
        return zarr.open(loc, { kind: 'array' })
      }

      const xarr =
        this.dimIndices.lon.array ||
        (await openArray(levelRoot.resolve(this.dimIndices.lon.name)))
      const yarr =
        this.dimIndices.lat.array ||
        (await openArray(levelRoot.resolve(this.dimIndices.lat.name)))

      const xdata = await zarr.get(xarr)
      const ydata = await zarr.get(yarr)

      const xValues = Array.from(xdata.data as ArrayLike<number>)
      const yValues = Array.from(ydata.data as ArrayLike<number>)

      this.xyLimits = {
        xMin: Math.min(...xValues),
        xMax: Math.max(...xValues),
        yMin: Math.min(...yValues),
        yMax: Math.max(...yValues),
      }
    } catch (err) {
      console.warn(
        'Failed to load XY limits from coordinate arrays, using defaults:',
        err
      )
      if (this.crs === 'EPSG:3857') {
        const worldExtent = 20037508.342789244
        this.xyLimits = {
          xMin: -worldExtent,
          xMax: worldExtent,
          yMin: -worldExtent,
          yMax: worldExtent,
        }
      } else {
        this.xyLimits = {
          xMin: -180,
          xMax: 180,
          yMin: -90,
          yMax: 90,
        }
      }
    }
  }

  private _getPyramidMetadata(multiscales: Multiscale[]): PyramidMetadata {
    if (!multiscales || !multiscales[0]?.datasets?.length) {
      return {
        levels: [],
        maxZoom: 0,
        tileSize: 128,
        crs: 'EPSG:4326',
      }
    }

    const datasets = multiscales[0].datasets
    const levels = datasets.map((dataset) => String(dataset.path))
    const maxZoom = levels.length - 1
    const tileSize = datasets[0].pixels_per_tile || 128
    // If CRS is absent, default to EPSG:3857 to match pyramid (mercator) tiling.
    // Explicitly handle EPSG:4326 pyramids so we can reproject to mercator on the fly.
    const crs: CRS =
      (datasets[0].crs as CRS) === 'EPSG:4326'
        ? 'EPSG:4326'
        : 'EPSG:3857'

    return { levels, maxZoom, tileSize, crs }
  }

  static clearCache() {
    ZarrStore._cache.clear()
    ZarrStore._storeCache.clear()
  }
}
