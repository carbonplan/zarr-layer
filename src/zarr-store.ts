import * as zarr from 'zarrita'
import proj4 from 'proj4'
import { withDecodedChunkCaching } from './decoded-chunk-cache'
import type { Readable, AsyncReadable } from '@zarrita/storage'
import type {
  Bounds,
  SpatialDimensions,
  DimIndicesProps,
  CRS,
  UntiledLevel,
  TransformRequest,
} from './types'
import { normalizeLongitudeExtent, type XYLimits } from './map-utils'
import { WEB_MERCATOR_EXTENT } from './constants'
import { identifyDimensionIndices, resolveOpenFunc } from './zarr-utils'
import {
  parseGeoZarrAttrs,
  parseLayoutItemSpatial,
  boundsFromSpatialAttrs,
  type AffineTransform,
  type GeoZarrAttrs,
  type SpatialExtent,
} from './geozarr'
import { normalizeBuiltinProjectionDef } from './projection-utils'

interface PyramidMetadata {
  levels: string[]
  maxLevelIndex: number
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
  derived_from?: string
}

interface UntiledMultiscaleMetadata {
  layout: UntiledMultiscaleLayoutEntry[]
  resampling_method?: string
  crs?: 'EPSG:4326' | 'EPSG:3857'
}

/**
 * `proj4.defs` is a module-global registry, so a definition read out of one
 * store's metadata needs a key no other layer on the page will reuse.
 */
let syntheticProjectionCount = 0

/**
 * A `proj:code` naming WGS84 lon/lat under its OGC identity. Same axis order
 * as the renderer already assumes, so it maps straight onto the native path.
 */
const CRS84 = 'OGC:CRS84'

type ZarrStoreType =
  | zarr.FetchStore
  | zarr.Listable<zarr.FetchStore>
  | Readable
  | AsyncReadable

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
  customStore?: Readable | AsyncReadable
}

interface StoreDescription {
  dimensions: string[]
  shape: number[]
  chunks: number[]
  fill_value: number | null
  dtype: string | null
  levels: string[]
  maxLevelIndex: number
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
}

/**
 * Factory function to create a store with optional request transformation.
 * When transformRequest is provided, uses FetchStore's native fetch handler
 * to intercept each request with the fully resolved URL.
 * This enables per-path authentication like presigned S3 URLs.
 */
const createFetchStore = (
  url: string,
  transformRequest?: TransformRequest
): zarr.FetchStore => {
  if (!transformRequest) {
    return new zarr.FetchStore(url)
  }
  return new zarr.FetchStore(url, {
    async fetch(request: Request): Promise<Response> {
      const { url: transformedUrl, ...overrides } = await transformRequest(
        request.url,
        { method: request.method as 'GET' | 'HEAD' }
      )
      const mergedHeaders = new Headers(request.headers)
      if (overrides.headers) {
        for (const [k, v] of Object.entries(
          overrides.headers as Record<string, string>
        )) {
          mergedHeaders.set(k, v)
        }
      }
      // Use `request` as the base init so signal/body/credentials/etc. carry
      // over (Request's own properties aren't spread-friendly), then overlay
      // transformRequest overrides with merged headers last.
      const response = await fetch(
        new Request(new Request(transformedUrl, request), {
          ...overrides,
          headers: mergedHeaders,
        })
      )
      // Remap 403 to 404 for S3/CloudFront compatibility: these services
      // return 403 (not 404) for missing or inaccessible paths.
      if (response.status === 403) {
        return new Response(null, { status: 404 })
      }
      return response
    },
  })
}

export class ZarrStore {
  source: string
  version: 2 | 3 | null
  variable: string
  spatialDimensions: SpatialDimensions
  private explicitBounds: Bounds | null
  coordinateKeys: string[]
  private transformRequest?: TransformRequest
  private customStore?: Readable | AsyncReadable

  dimensions: string[] = []
  shape: number[] = []
  chunks: number[] = []
  fill_value: number | null = null
  dtype: string | null = null
  levels: string[] = []
  maxLevelIndex: number = 0
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
  private _proj4Override: boolean = false // Track if proj4 was explicitly set by user
  private geoZarr: GeoZarrAttrs | null = null
  /** Per-level `spatial:shape` as declared, [height, width], before axis mapping. */
  private _declaredLevelShapes: ([number, number] | undefined)[] = []
  /** Per-level `spatial:transform` as declared, indexed alongside the levels. */
  private _declaredLevelTransforms: (AffineTransform | undefined)[] = []

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
    this._proj4Override = !!proj4
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
    if (this.customStore) {
      // Validate that custom store implements required Readable interface
      if (typeof this.customStore.get !== 'function') {
        throw new Error(
          'customStore must implement Readable interface with get() method'
        )
      }
      // Skip consolidated metadata: custom stores typically have their
      // own efficient metadata layer. `withRangeCoalescing` eagerly
      // asserts `getRange`, so only install it when the store has one.
      const hasGetRange =
        typeof (this.customStore as { getRange?: unknown }).getRange ===
        'function'
      this.store = hasGetRange
        ? ((await zarr.extendStore(
            this.customStore as AsyncReadable,
            (store) => zarr.withRangeCoalescing(store),
            (store) => withDecodedChunkCaching(store)
          )) as ZarrStoreType)
        : ((await zarr.extendStore(this.customStore as AsyncReadable, (store) =>
            withDecodedChunkCaching(store)
          )) as ZarrStoreType)
    } else {
      // Layered data access:
      // - Consolidated metadata: one-shot fetch of the `.zmetadata`/`zarr.json`
      //   blob so array opens are in-memory lookups. Falls back to the raw
      //   store if the wrapper trips (e.g. v3 experimental).
      // - Range coalescing: batches concurrent HTTP range requests in a
      //   microtask tick so many tile fetches become few round-trips.
      // - Decoded-chunk caching: memoizes the decompressed `getChunk`
      //   ndarray so selector scrubs and hover queries within
      //   already-fetched chunks skip decompression. Concurrent requests
      //   for the same chunk share one fetch via an in-flight map.
      const consolidatedOpts: zarr.ConsolidatedMetadataOptions | undefined =
        this.version === 2
          ? { format: 'v2' }
          : this.version === 3
          ? { format: 'v3' }
          : undefined
      this.store = (await zarr.extendStore(
        createFetchStore(this.source, this.transformRequest),
        (store) =>
          zarr
            .withMaybeConsolidatedMetadata(store, consolidatedOpts)
            .catch(() => store),
        (store) => zarr.withRangeCoalescing(store),
        (store) => withDecodedChunkCaching(store)
      )) as ZarrStoreType
    }

    this.root = zarr.root(this.store)
    await this._loadMetadata()

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
      dimensions: this.dimensions,
      shape: this.shape,
      chunks: this.chunks,
      fill_value: this.fill_value,
      dtype: this.dtype,
      levels: this.levels,
      maxLevelIndex: this.maxLevelIndex,
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
   * Uses zarrita's array properties — no manual JSON fetching needed.
   * On consolidated stores, metadata is served from cache (no network).
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
    const attrs = array.attrs as Record<string, unknown>
    const dtype = (array.dtype as string) || null
    const fillValue = this.normalizeFillValue(array.fillValue)

    // Float data typically stores already-physical values (e.g., pyramid levels
    // created by averaging). Integer data stores raw counts needing conversion.
    const isFloatData = !!dtype?.includes('float')

    let scaleFactor: number | undefined = undefined
    let addOffset: number | undefined = undefined

    if (isFloatData) {
      scaleFactor = 1
      addOffset = 0
    } else {
      if (attrs?.scale_factor !== undefined) {
        scaleFactor = attrs.scale_factor as number
      }
      if (attrs?.add_offset !== undefined) {
        addOffset = attrs.add_offset as number
      }
    }

    // A level may declare its transform without a `spatial:shape`, in which
    // case its extent can only be worked out once the real shape is known.
    const index = this.untiledLevels.findIndex((l) => l.asset === levelAsset)
    const level = this.untiledLevels[index]
    const { lat, lon } = this.dimIndices
    if (level && !level.xyLimits && lat && lon) {
      level.xyLimits = this._deriveLevelExtent(index, [
        array.shape[lat.index],
        array.shape[lon.index],
      ])
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
      const openFunc = resolveOpenFunc(this.version)
      handle = openFunc(location, { kind: 'array' }).catch((err: Error) => {
        this._arrayHandles.delete(key)
        throw err
      })
      this._arrayHandles.set(key, handle)
    }

    return handle
  }

  private isConsolidatedStore(store: ZarrStoreType | null): store is {
    contents(): { path: `/${string}`; kind: 'array' | 'group' }[]
  } {
    return (
      store !== null &&
      typeof (store as { contents?: unknown }).contents === 'function'
    )
  }

  /**
   * Unified metadata loading using zarrita's built-in APIs.
   * zarrita auto-detects Zarr v2/v3 format and provides parsed metadata
   * via group.attrs and array.shape/chunks/dtype/fillValue/dimensionNames/attrs.
   */
  private async _loadMetadata(): Promise<void> {
    if (!this.root) throw new Error('Zarr store not initialized')

    // Open root group to get multiscales metadata from attrs
    const openFunc = resolveOpenFunc(this.version)
    const group = await openFunc(this.root, { kind: 'group' })
    const rootAttrs = group.attrs as Record<string, unknown>

    if (rootAttrs?.multiscales) {
      const pyramid = this._getPyramidMetadata(
        rootAttrs.multiscales as Multiscale[] | UntiledMultiscaleMetadata
      )
      this.levels = pyramid.levels
      this.maxLevelIndex = pyramid.maxLevelIndex
      if (!this._crsOverride) {
        this.crs = pyramid.crs
      }
    }

    // Open target array to get shape, chunks, dtype, fill_value, dimensions
    const basePath =
      this.levels.length > 0
        ? `${this.levels[0]}/${this.variable}`
        : this.variable
    const array = await this._getArray(basePath)
    const arrayAttrs = array.attrs as Record<string, unknown>

    this.geoZarr = parseGeoZarrAttrs(
      rootAttrs,
      await this._readBaseLevelGroupAttrs(),
      arrayAttrs
    )
    await this._applyGeoZarrCrs(arrayAttrs)

    // zarrita's dimensionNames returns the unified answer for v2
    // (_ARRAY_DIMENSIONS) and v3 (dimension_names).
    this.dimensions = array.dimensionNames ?? []
    this.shape = array.shape
    // zarrita's array.chunks already handles sharding (inner chunk shape)
    this.chunks = array.chunks
    this.fill_value = this.normalizeFillValue(array.fillValue)
    this.dtype = (array.dtype as string) || null
    this.scaleFactor =
      typeof arrayAttrs?.scale_factor === 'number' ? arrayAttrs.scale_factor : 1
    this.addOffset =
      typeof arrayAttrs?.add_offset === 'number' ? arrayAttrs.add_offset : 0

    await this._computeDimIndices()
    this._applyDeclaredLevelShapes()
    this._applyDeclaredLevelExtents()
  }

  /**
   * Fill in each level's shape from the `spatial:shape` its layout entry
   * declares, sparing the renderer an array open per level just to size the
   * pyramid.
   *
   * `spatial:shape` is [height, width]; a level's shape follows the array's own
   * dimension order and carries its non-spatial dimensions too, so the declared
   * pair is substituted into the base shape rather than used directly.
   */
  private _applyDeclaredLevelShapes(): void {
    if (this._declaredLevelShapes.length === 0 || this.shape.length === 0)
      return

    const { lat, lon } = this.dimIndices
    if (!lat || !lon) return

    this._declaredLevelShapes.forEach((declared, i) => {
      const level = this.untiledLevels[i]
      if (!declared || !level) return
      const shape = [...this.shape]
      shape[lat.index] = declared[0]
      shape[lon.index] = declared[1]
      level.shape = shape
    })
  }

  /**
   * Give each level the extent its own `spatial:transform` describes.
   *
   * Levels of a pyramid normally share the dataset extent, but floor division
   * leaves a coarse level covering a partial trailing cell less, and the
   * convention permits levels to differ outright. Placing a level against the
   * dataset extent instead of its own stretches it by that difference.
   */
  private _applyDeclaredLevelExtents(): void {
    this._declaredLevelTransforms.forEach((_, i) => {
      const shape = this._declaredLevelShapes[i]
      if (!shape) return
      const extent = this._deriveLevelExtent(i, shape)
      if (extent) this.untiledLevels[i].xyLimits = extent
    })
  }

  /**
   * The extent level `index` covers, from its own `spatial:transform` and the
   * given `[rows, cols]`.
   *
   * Returns nothing when the caller supplied explicit `bounds`: those override
   * the store's georeferencing wholesale, and re-deriving a level extent from
   * the same metadata would quietly reinstate what was overridden.
   */
  private _deriveLevelExtent(
    index: number,
    shape: [number, number]
  ): XYLimits | undefined {
    const attrs = this.geoZarr
    const transform = this._declaredLevelTransforms[index]
    if (!attrs || !transform || this.explicitBounds) return undefined

    const [nRows, nCols] = shape
    // The level's own transform places it; a dataset-wide bbox describes the
    // base level and would defeat the point.
    const extent = boundsFromSpatialAttrs(
      { ...attrs, transform, bbox: undefined },
      { nCols, nRows }
    )
    if (!extent) return undefined

    const { xMin, xMax, yMin, yMax } = extent
    return this.isGeographic()
      ? {
          yMin,
          yMax,
          ...normalizeLongitudeExtent(xMin, xMax, (xMax - xMin) / nCols),
        }
      : { xMin, xMax, yMin, yMax }
  }

  /**
   * Attributes of the group holding the base level's arrays.
   *
   * `proj:` inherits only to a group's direct child arrays, so a pyramid whose
   * levels are groups may restate it on each of them instead of at the root.
   * That group sits between the two we already read, and is one metadata
   * lookup -- served from cache on a consolidated store, and never per level.
   */
  private async _readBaseLevelGroupAttrs(): Promise<
    Record<string, unknown> | undefined
  > {
    if (this.levels.length === 0 || !this.root) return undefined
    try {
      const openFunc = resolveOpenFunc(this.version)
      const group = await openFunc(this.root.resolve(this.levels[0]), {
        kind: 'group',
      })
      return group.attrs as Record<string, unknown>
    } catch {
      // A level that isn't a group at all just contributes nothing.
      return undefined
    }
  }

  /**
   * Resolve the CRS the store declares through the `proj:` convention, falling
   * back to the CF grid-mapping variable when it declares none.
   *
   * A `proj:code` naming one of the two built-in CRSs is honored first: those
   * render through a native path that needs no proj4 transformer, and a store
   * naming one has already said everything we need. WKT2 and PROJJSON come
   * next, since they describe the CRS in full and proj4 parses both without a
   * lookup table. Only then does an unfamiliar code get looked up against
   * proj4's built-in definitions.
   *
   * A declared CRS that stays unresolved is left alone rather than guessed at
   * from bounds. Inferring a different CRS would silently contradict what the
   * store said; rendering visibly wrong is the honest outcome, and
   * `proj4.defs()` is the caller's remedy.
   */
  private async _applyGeoZarrCrs(
    arrayAttrs: Record<string, unknown>
  ): Promise<void> {
    if (this._crsOverride || this._proj4Override) return

    const declared = this.geoZarr?.crs
    const code = declared?.code
    const builtin =
      code?.trim().toUpperCase() === CRS84
        ? 'EPSG:4326'
        : normalizeBuiltinProjectionDef(code)
    if (builtin) {
      this.crs = builtin
      this._crsFromMetadata = true
      return
    }

    if (
      declared?.wkt2 &&
      this._registerProjection(declared.wkt2, 'proj:wkt2')
    ) {
      return
    }
    if (
      declared?.projjson &&
      this._registerProjection(declared.projjson, 'proj:projjson')
    ) {
      return
    }

    if (code) {
      this._crsFromMetadata = true
      // proj4's registry is keyed on the canonical uppercase form, so a store
      // writing "epsg:32631" needs normalizing to find it.
      const registered = proj4.defs(code) ? code : code.toUpperCase()
      if (proj4.defs(registered)) {
        this.proj4 = registered
        return
      }
      console.warn(
        `[zarr-layer] Store declares proj:code "${code}", which proj4 does not ` +
          `know and which the store gives no proj:wkt2 or proj:projjson for. ` +
          `Register it before creating the layer with ` +
          `proj4.defs('${code}', '<proj4 string>') to render it correctly.`
      )
      return
    }

    await this._applyCfGridMappingCrs(arrayAttrs)
  }

  /**
   * Read a WKT definition off the CF grid-mapping variable the data array
   * points at. rioxarray writes the same WKT under both `crs_wkt` and
   * `spatial_ref`.
   */
  private async _applyCfGridMappingCrs(
    arrayAttrs: Record<string, unknown>
  ): Promise<void> {
    const gridMapping = arrayAttrs?.grid_mapping
    if (typeof gridMapping !== 'string' || !gridMapping.trim()) return

    const prefix = this.levels.length > 0 ? `${this.levels[0]}/` : ''
    try {
      const mapping = await this._getArray(`${prefix}${gridMapping.trim()}`)
      const attrs = mapping.attrs as Record<string, unknown>
      const wkt = [attrs?.crs_wkt, attrs?.spatial_ref].find(
        (value): value is string =>
          typeof value === 'string' && value.trim().length > 0
      )
      if (wkt) this._registerProjection(wkt, `${gridMapping}.crs_wkt`)
    } catch (err) {
      console.warn(
        `[zarr-layer] Could not read grid mapping variable '${gridMapping}': `,
        err
      )
    }
  }

  /**
   * Register a full CRS definition under a key of our own and point the store
   * at it. proj4 parses WKT2 strings and PROJJSON objects directly, so a store
   * carrying either is self-describing with no lookup table.
   */
  private _registerProjection(
    def: string | Record<string, unknown>,
    source: string
  ): boolean {
    const key = `ZARRLAYER:${++syntheticProjectionCount}`
    try {
      proj4.defs(key, def as string)
      if (!proj4.defs(key)) throw new Error('definition did not parse')
      proj4(key, 'EPSG:4326')
    } catch (err) {
      console.warn(
        `[zarr-layer] Could not use '${source}' from store metadata: ` +
          `${err instanceof Error ? err.message : err}`
      )
      return false
    }
    this.proj4 = key
    this._crsFromMetadata = true
    return true
  }

  /**
   * The spatial dimension names to identify axes by.
   *
   * `spatial:dimensions` names them outright, ordered [y, x], which beats
   * guessing from an alias list and is the only thing that works for a store
   * whose axes are named something the alias list has never heard of. The
   * constructor option still wins per axis.
   */
  private _resolveSpatialDimensions(): SpatialDimensions {
    const declared = this.geoZarr?.dimensions
    if (!declared) return this.spatialDimensions

    const [declaredLat, declaredLon] = declared
    const lat = this.spatialDimensions.lat ?? declaredLat
    const lon = this.spatialDimensions.lon ?? declaredLon

    // Only the declarations actually being used are worth rejecting. An axis
    // the caller overrode is repaired already, and a bad override is the
    // caller's own error, which `identifyDimensionIndices` reports.
    const known = this.dimensions.map((d) => d.toLowerCase())
    const missing = [
      this.spatialDimensions.lat ? null : declaredLat,
      this.spatialDimensions.lon ? null : declaredLon,
    ].filter((n): n is string => !!n && !known.includes(n.toLowerCase()))
    if (missing.length > 0) {
      throw new Error(
        `spatial:dimensions names [${missing.join(
          ', '
        )}], which the array does not have. Available: [${this.dimensions.join(
          ', '
        )}]`
      )
    }

    return { lat, lon }
  }

  private async _computeDimIndices() {
    if (this.dimensions.length === 0) return

    this.dimIndices = identifyDimensionIndices(
      this.dimensions,
      this._resolveSpatialDimensions()
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

  /** Whether x/y are longitude/latitude in degrees rather than projected units. */
  private isGeographic(): boolean {
    return !this.proj4 && this.crs !== 'EPSG:3857'
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
   * Find the highest resolution level by comparing array shapes.
   * On consolidated stores, zarr.open serves metadata from cache (no network).
   * Users can provide explicit `bounds` to skip this detection entirely.
   */
  private async _findBoundsLevel(): Promise<string | undefined> {
    if (this.levels.length === 0 || !this.root) return undefined
    if (this.levels.length === 1) return this.levels[0]

    const firstLevel = this.levels[0]
    const lastLevel = this.levels[this.levels.length - 1]

    try {
      const [firstArray, lastArray] = await Promise.all([
        this._getArray(`${firstLevel}/${this.variable}`),
        this._getArray(`${lastLevel}/${this.variable}`),
      ])

      const firstSize = firstArray.shape.reduce((a, b) => a * b, 1)
      const lastSize = lastArray.shape.reduce((a, b) => a * b, 1)
      return firstSize >= lastSize ? firstLevel : lastLevel
    } catch {
      return firstLevel
    }
  }

  /**
   * The declaration to place the grid from.
   *
   * A pyramid's absolute georeferencing belongs on its `multiscales.layout`
   * entries, one per resolution, so the base entry's transform stands in
   * whenever the group or array declares none of its own. The two sources
   * combine rather than compete: a group that states only a `spatial:bbox`
   * still gets its row direction from that transform, which is the difference
   * between placing the grid outright and falling back to coordinate reads.
   *
   * The grid size comes from the base array either way, which is the level
   * that entry describes.
   */
  private _effectiveSpatialAttrs(): GeoZarrAttrs | null {
    const attrs = this.geoZarr
    if (!attrs) return null

    const transform = attrs.transform ?? this._declaredLevelTransforms[0]
    if (!attrs.bbox && !transform) return null
    return transform === attrs.transform ? attrs : { ...attrs, transform }
  }

  /**
   * The grid extent the store declares through the `spatial:` convention, in
   * the renderer's edge-to-edge terms and its -180–180 longitude range.
   *
   * Returns null whenever the declaration doesn't settle the extent, leaving
   * the coordinate-array read as the fallback.
   */
  private _declaredSpatialExtent(): SpatialExtent | null {
    const attrs = this._effectiveSpatialAttrs()
    if (!attrs) return null

    if (attrs.transformType !== 'affine') {
      console.warn(
        `[zarr-layer] Store declares spatial:transform_type "${attrs.transformType}", ` +
          `which this layer cannot map to a grid. Reading bounds from coordinate arrays.`
      )
      return null
    }

    const { lon, lat } = this.dimIndices
    if (!lon || !lat) return null

    const extent = boundsFromSpatialAttrs(attrs, {
      nCols: this.shape[lon.index],
      nRows: this.shape[lat.index],
    })
    if (!extent) {
      console.warn(
        `[zarr-layer] Could not derive bounds from the store's spatial: attributes. ` +
          `A rotated transform has no axis-aligned placement this layer can render, ` +
          `and a node-registered bbox needs a cell size to expand by. ` +
          `Reading bounds from coordinate arrays.`
      )
      return null
    }

    if (!this.isGeographic()) return extent

    const cellWidth = (extent.xMax - extent.xMin) / this.shape[lon.index]
    return {
      ...extent,
      ...normalizeLongitudeExtent(extent.xMin, extent.xMax, cellWidth),
    }
  }

  private async _loadSpatialMetadata() {
    // Apply explicit bounds first (takes precedence for all multiscale types)
    // Bounds are in source CRS units (degrees for EPSG:4326, meters for EPSG:3857/proj4)
    if (this.explicitBounds) {
      const [west, south, east, north] = this.explicitBounds
      this.xyLimits = { xMin: west, xMax: east, yMin: south, yMax: north }
    }

    // Tiled pyramids: use the standard global slippy-map extent if no explicit
    // bounds. The extent units depend on CRS — EPSG:3857 covers the full square
    // Web Mercator world in meters, EPSG:4326 the full lon/lat range in degrees.
    if (this.multiscaleType === 'tiled') {
      if (!this.xyLimits) {
        this.xyLimits =
          this.crs === 'EPSG:3857'
            ? {
                xMin: -WEB_MERCATOR_EXTENT,
                xMax: WEB_MERCATOR_EXTENT,
                yMin: -WEB_MERCATOR_EXTENT,
                yMax: WEB_MERCATOR_EXTENT,
              }
            : { xMin: -180, xMax: 180, yMin: -90, yMax: 90 }
      }
      if (!this._latIsAscendingUserSet) {
        this.latIsAscending = false // Tiled pyramids: row 0 = north
      }
      return
    }

    // For untiled: determine what we still need to detect
    let needsBounds = !this.xyLimits
    let needsLatAscending = !this._latIsAscendingUserSet

    // A store declaring the spatial: convention has already said where its grid
    // sits, so take it at its word and skip the coordinate reads.
    if (needsBounds || needsLatAscending) {
      const declared = this._declaredSpatialExtent()
      if (declared) {
        if (needsBounds) {
          this.xyLimits = {
            xMin: declared.xMin,
            xMax: declared.xMax,
            yMin: declared.yMin,
            yMax: declared.yMax,
          }
          needsBounds = false
        }
        if (needsLatAscending && declared.latIsAscending !== null) {
          this.latIsAscending = declared.latIsAscending
          needsLatAscending = false
        }
      }
    }

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

      const lonName = this.spatialDimensions.lon ?? this.dimIndices.lon.name
      const latName = this.spatialDimensions.lat ?? this.dimIndices.lat.name

      // Find the best coordinate array path from consolidated store listings.
      // On consolidated stores, uses store.contents() to enumerate all arrays;
      // on non-consolidated stores, returns null (triggers default fallback).
      const findCoordPath = async (dimName: string): Promise<string | null> => {
        const store = this.store
        if (!this.isConsolidatedStore(store)) return null

        const entries = store.contents()
        // Find all array entries whose path ends with the dimension name
        const matchingPaths = entries
          .filter(
            (e) =>
              e.kind === 'array' &&
              (e.path === `/${dimName}` || e.path.endsWith(`/${dimName}`))
          )
          .map((e) => e.path.slice(1)) // Remove leading '/'

        if (matchingPaths.length === 0) return null
        if (matchingPaths.length === 1) return matchingPaths[0]

        // Multiple matches: open each to find highest resolution (largest shape[0])
        const withSizes = await Promise.all(
          matchingPaths.map(async (path) => {
            try {
              const arr = await this._getArray(path)
              return { path, size: arr.shape[0] }
            } catch {
              return { path, size: 0 }
            }
          })
        )

        type Candidate = { path: string; size: number }
        const largest = (
          predicate: (c: Candidate) => boolean
        ): Candidate | undefined =>
          withSizes.reduce<Candidate | undefined>(
            (best, c) =>
              predicate(c) && (!best || c.size > best.size) ? c : best,
            undefined
          )

        // Prefer coord arrays within the bounds level, then root-level, then largest
        if (boundsLevel) {
          const levelPrefix = `${boundsLevel}/`
          const levelPick = largest((c) => c.path.startsWith(levelPrefix))
          if (levelPick) return levelPick.path

          const rootPick = largest((c) => !c.path.includes('/'))
          if (rootPick) return rootPick.path
        } else if (this.variable) {
          const varPick = largest((c) => c.path.startsWith(`${this.variable}/`))
          if (varPick) return varPick.path
        }

        return largest(() => true)?.path ?? null
      }

      // Find highest resolution coordinate arrays from store listings
      const [xPath, yPath] = await Promise.all([
        findCoordPath(lonName),
        findCoordPath(latName),
      ])

      // Open coord arrays: use metadata path if found, otherwise try level/dimName
      const defaultPrefix = boundsLevel ? `${boundsLevel}/` : ''
      const xarr = await this._getArray(xPath ?? `${defaultPrefix}${lonName}`)
      const yarr = await this._getArray(yPath ?? `${defaultPrefix}${latName}`)

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
      const rawXMin = coordXMin - (Number.isFinite(dx) ? dx / 2 : 0)
      const rawXMax = coordXMax + (Number.isFinite(dx) ? dx / 2 : 0)
      const yMin = coordYMin - (Number.isFinite(dy) ? dy / 2 : 0)
      const yMax = coordYMax + (Number.isFinite(dy) ? dy / 2 : 0)

      const { xMin, xMax } = this.isGeographic()
        ? normalizeLongitudeExtent(rawXMin, rawXMax, dx)
        : { xMin: rawXMin, xMax: rawXMax }

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
    // Default for missing or unrecognized multiscale metadata: single-level untiled
    const singleLevelUntiled = (): PyramidMetadata => {
      this.multiscaleType = 'untiled'
      return {
        levels: [],
        maxLevelIndex: 0,
        crs: this.crs,
      }
    }

    if (!multiscales) return singleLevelUntiled()

    // Format 1: zarr-conventions/multiscales (has 'layout' key)
    // See: https://github.com/zarr-conventions/multiscales
    if ('layout' in multiscales && Array.isArray(multiscales.layout)) {
      return this._parseUntiledMultiscale(multiscales, singleLevelUntiled)
    }

    // Format 2: OME-NGFF style (array with 'datasets' key)
    // See: https://ngff.openmicroscopy.org/latest/
    if (Array.isArray(multiscales) && multiscales[0]?.datasets?.length) {
      const datasets = multiscales[0].datasets
      const levels = datasets.map((dataset) => String(dataset.path))
      const maxLevelIndex = levels.length - 1
      const pixelsPerTile = datasets[0].pixels_per_tile
      // If CRS is absent, default to EPSG:3857 to match pyramid (mercator) tiling.
      const crs: CRS =
        (datasets[0].crs as CRS) === 'EPSG:4326' ? 'EPSG:4326' : 'EPSG:3857'

      // Both shapes expose their levels as untiledLevels; only the
      // classification differs. `pixels_per_tile` marks a slippy-map pyramid,
      // kept as 'tiled' purely to drive its metadata defaults in
      // `_loadSpatialMetadata` (global extent, latIsAscending=false, no
      // coordinate-array reads).
      this.multiscaleType = pixelsPerTile ? 'tiled' : 'untiled'
      this.untiledLevels = levels.map((level) => ({ asset: level }))
      return { levels, maxLevelIndex, crs }
    }

    return singleLevelUntiled()
  }

  /**
   * Parse zarr-conventions/multiscales format (layout-based).
   *
   * This format uses a `layout` array where each entry specifies:
   * - `asset`: path to the level (e.g., "0", "1", ...)
   * - `spatial:shape`: optional level dimensions as [height, width]
   *
   * Example metadata:
   * ```json
   * {
   *   "layout": [
   *     { "asset": "0", "spatial:shape": [1024, 2048] },
   *     { "asset": "1", "spatial:shape": [512, 1024] }
   *   ],
   *   "crs": "EPSG:4326"
   * }
   * ```
   *
   * @see https://github.com/zarr-conventions/multiscales
   */
  private _parseUntiledMultiscale(
    metadata: UntiledMultiscaleMetadata,
    singleLevelUntiled: () => PyramidMetadata
  ): PyramidMetadata {
    const layout = metadata.layout
    if (!layout || layout.length === 0) return singleLevelUntiled()

    // Extract levels from layout
    const levels = layout.map((entry) => entry.asset)
    const maxLevelIndex = levels.length - 1

    this.untiledLevels = layout.map((entry) => ({ asset: entry.asset }))
    const perLevel = layout.map((entry) => parseLayoutItemSpatial(entry))
    // Applied once the dimension order is known; see `_applyDeclaredLevelShapes`.
    this._declaredLevelShapes = perLevel.map((s) => s.shape)
    this._declaredLevelTransforms = perLevel.map((s) => s.transform)

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
      crs,
    }
  }
}
