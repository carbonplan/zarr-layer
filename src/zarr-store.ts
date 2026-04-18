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
import { DEFAULT_TILE_SIZE } from './constants'
import { identifyDimensionIndices, resolveOpenFunc } from './zarr-utils'

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
  private static _storeCache = new Map<string, Promise<ZarrStoreType>>()

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
  tileSize: number = DEFAULT_TILE_SIZE
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

    if (this.customStore) {
      // Validate that custom store implements required Readable interface
      if (typeof this.customStore.get !== 'function') {
        throw new Error(
          'customStore must implement Readable interface with get() method'
        )
      }
      // Use custom store directly (e.g., IcechunkStore)
      this.store = this.customStore as ZarrStoreType
    } else {
      const bypassCache = !!this.transformRequest
      let storePromise = bypassCache
        ? undefined
        : ZarrStore._storeCache.get(storeCacheKey)

      if (!storePromise) {
        const baseStore = createFetchStore(this.source, this.transformRequest)
        // When the version is known, tell the consolidated-metadata wrapper
        // to only try that format — avoids a wasted .zmetadata fetch on v3
        // stores (and vice versa). Falls back to auto-detect when unknown.
        // v3 consolidated metadata support is experimental; the outer
        // `.catch` keeps us on the raw store if the wrapper trips.
        const consolidatedOpts: zarr.ConsolidatedMetadataOptions | undefined =
          this.version === 2
            ? { format: 'v2' }
            : this.version === 3
            ? { format: 'v3' }
            : undefined
        // Range coalescing groups concurrent HTTP range requests into fewer
        // round-trips, reducing latency when fetching many tiles in parallel.
        storePromise = zarr.extendStore(
          baseStore,
          (store) =>
            zarr
              .withMaybeConsolidatedMetadata(store, consolidatedOpts)
              .catch(() => store),
          (store) => zarr.withRangeCoalescing(store)
        ) as Promise<ZarrStoreType>
        if (!bypassCache) {
          ZarrStore._storeCache.set(storeCacheKey, storePromise)
        }
      }

      this.store = await storePromise
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
      this.tileSize = pyramid.tileSize
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
        tileSize: DEFAULT_TILE_SIZE,
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
      const tileSize = datasets[0].pixels_per_tile
      // If CRS is absent, default to EPSG:3857 to match pyramid (mercator) tiling.
      const crs: CRS =
        (datasets[0].crs as CRS) === 'EPSG:4326' ? 'EPSG:4326' : 'EPSG:3857'

      // If pixels_per_tile is present, this is a tiled pyramid (slippy map tiles).
      // Otherwise, treat as untiled multi-level (each level is a complete image).
      if (tileSize) {
        this.multiscaleType = 'tiled'
        return { levels, maxLevelIndex, tileSize, crs }
      }
      // Multi-level but not tiled - use UntiledMode
      this.untiledLevels = levels.map((level) => ({
        asset: level,
        scale: [1.0, 1.0] as [number, number],
        translation: [0.0, 0.0] as [number, number],
      }))
      this.multiscaleType = 'untiled'
      return { levels, maxLevelIndex, tileSize: DEFAULT_TILE_SIZE, crs }
    }

    return singleLevelUntiled()
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
    metadata: UntiledMultiscaleMetadata,
    singleLevelUntiled: () => PyramidMetadata
  ): PyramidMetadata {
    const layout = metadata.layout
    if (!layout || layout.length === 0) return singleLevelUntiled()

    // Extract levels from layout
    const levels = layout.map((entry) => entry.asset)
    const maxLevelIndex = levels.length - 1

    // Build untiledLevels with transform info (shapes loaded lazily via getUntiledLevelMetadata)
    this.untiledLevels = layout.map((entry) => ({
      asset: entry.asset,
      scale: entry.transform?.scale ?? [1.0, 1.0],
      translation: entry.transform?.translation ?? [0.0, 0.0],
    }))

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
      tileSize: DEFAULT_TILE_SIZE, // Will be overridden by chunk shape
      crs,
    }
  }

  static clearCache() {
    ZarrStore._storeCache.clear()
  }
}
