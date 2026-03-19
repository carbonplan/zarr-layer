import * as zarr from 'zarrita'
import parseWkt from 'wkt-parser'
import proj4 from 'proj4'
import type { Readable, AsyncReadable } from '@zarrita/storage'
import type {
  Bounds,
  SpatialDimensions,
  DimIndicesProps,
  CRS,
  UntiledLevel,
  TransformRequest,
  ResolveProj4,
  ResolveProj4Context,
} from './types'
import type { XYLimits } from './map-utils'
import { identifyDimensionIndices } from './zarr-utils'

const textDecoder = new TextDecoder()

const decodeJSON = (bytes: Uint8Array | undefined): unknown => {
  if (!bytes) return null
  return JSON.parse(textDecoder.decode(bytes))
}

const EPSG_AUTHORITY_REGEX = /AUTHORITY\["EPSG","(\d+)"\]/g
const PROJ4_EXTENSION_REGEX = /EXTENSION\["PROJ4","([^"]+)"\]/

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (typeof value !== 'object' || value === null) return null
  return value as Record<string, unknown>
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
  crs?: string
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

const fetchSuffix = (
  url: string | URL,
  suffixLength: number,
  opts: RequestInit = {}
): Promise<Response> => {
  return fetch(url, {
    ...opts,
    headers: {
      ...(opts.headers as Record<string, string>),
      Range: `bytes=-${suffixLength}`,
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
      // Use a single suffix-range request to avoid extra HEAD probes.
      const { url: getUrl, ...getOverrides } = await this.transformRequest(
        resolvedUrl,
        { method: 'GET' }
      )
      const getMerged = mergeInit(getOverrides, opts)
      response = await fetchSuffix(getUrl, range.suffixLength, getMerged)
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
  resolveProj4?: boolean | ResolveProj4
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
    return new zarr.FetchStore(url, { useSuffixRequest: true })
  }
  return new TransformingFetchStore(url, transformRequest)
}

export class ZarrStore {
  private static _cache = new Map<
    string,
    ZarrV2ConsolidatedMetadata | ZarrV3GroupMetadata | ZarrV3ArrayMetadata
  >()
  private static _storeCache = new Map<string, Promise<ZarrStoreType>>()
  private static _proj4LookupCache = new Map<string, Promise<string | null>>()

  source: string
  version: 2 | 3 | null
  variable: string
  spatialDimensions: SpatialDimensions
  private explicitBounds: Bounds | null
  coordinateKeys: string[]
  private transformRequest?: TransformRequest
  private resolveProj4?: ResolveProj4
  private enableOnlineProj4Lookup: boolean = true
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
  private _crsFromMetadata: boolean = false // Track if CRS was explicitly set from metadata
  private _crsOverride: boolean = false // Track if CRS was explicitly set by user
  private _variableMetadataMissing: boolean = false
  private _pendingUserCrs: string | null = null
  private _pendingMetadataCrs: string | null = null

  private normalizeCrsCode(crs: unknown): string | null {
    if (typeof crs === 'number' && Number.isFinite(crs)) {
      return `EPSG:${Math.trunc(crs)}`
    }
    if (typeof crs !== 'string') return null

    const value = crs.trim()
    if (!value) return null

    if (/^EPSG:\d+$/i.test(value)) {
      return value.toUpperCase()
    }

    if (/^\d+$/.test(value)) {
      return `EPSG:${value}`
    }

    const urnMatch = value.match(/EPSG(?:::|\/0\/)(\d+)$/i)
    if (urnMatch) {
      return `EPSG:${urnMatch[1]}`
    }

    return value
  }

  private logCrsCandidate(
    source: string,
    kind: 'proj' | 'spatial_ref',
    candidate: { crs?: string; proj4?: string }
  ): void {
    const hasProj4 =
      typeof candidate.proj4 === 'string' && candidate.proj4.trim().length > 0
    console.info(
      `[zarr-layer] CRS candidate (${kind}) from ${source}: ` +
        `crs="${candidate.crs ?? 'none'}", proj4=${
          hasProj4 ? 'present' : 'missing'
        }`
    )
  }

  private normalizeProj4Definition(
    proj4Candidate: unknown,
    source: string
  ): string | null {
    if (typeof proj4Candidate !== 'string') return null
    const value = proj4Candidate.trim()
    if (!value) return null

    // Accept EPSG codes only when they are already registered in proj4.
    if (/^EPSG:\d+$/i.test(value)) {
      const code = value.toUpperCase()
      const existing = proj4.defs(code)
      if (existing) {
        try {
          // Ensure registered definition is actually usable by proj4.
          proj4(code, 'EPSG:4326')
          return code
        } catch (err) {
          console.warn(
            `[zarr-layer] Registered proj4 definition for "${code}" is unusable: ` +
              `${err instanceof Error ? err.message : err}`
          )
        }
      }
      console.warn(
        `[zarr-layer] Ignoring proj4 candidate "${value}" from ${source}: ` +
          `EPSG code is not registered in proj4.`
      )
      return null
    }

    try {
      // Validate parseability early so invalid strings don't poison CRS inference.
      proj4(value, 'EPSG:4326')
      return value
    } catch (err) {
      console.warn(
        `[zarr-layer] Ignoring invalid proj4 candidate from ${source}: ` +
          `${err instanceof Error ? err.message : err}`
      )
      return null
    }
  }

  private extractFromWkt(wkt: unknown): { crs?: string; proj4?: string } {
    if (typeof wkt !== 'string' || !wkt.trim()) return {}

    // Prefer parser-based extraction via proj4js/wkt-parser.
    // Keep regex fallback for malformed or non-standard WKT strings.
    try {
      const parsed = asRecord(parseWkt(wkt))
      if (parsed) {
        const directTitle = this.normalizeCrsCode(parsed.title)
        const authorityObj = asRecord(parsed.AUTHORITY)
        const authorityCode = this.normalizeCrsCode(authorityObj?.EPSG)

        const extension = asRecord(parsed.EXTENSION)
        const proj4FromExtension =
          typeof extension?.PROJ4 === 'string' ? extension.PROJ4 : undefined

        const crs = authorityCode ?? directTitle
        const proj4FromParsed =
          !proj4FromExtension && crs
            ? this.registerProjectionDefinition(
                crs,
                parsed as proj4.ProjectionDefinition,
                'parsed WKT'
              )
            : undefined
        const proj4FromWkt =
          !proj4FromExtension && !proj4FromParsed && crs
            ? this.registerProjectionDefinition(crs, wkt, 'raw WKT')
            : undefined
        const inferredProj4 =
          proj4FromExtension ?? proj4FromParsed ?? proj4FromWkt
        if (crs || inferredProj4) {
          return { crs: crs ?? undefined, proj4: inferredProj4 }
        }
      }
    } catch {
      // Fall through to regex-based fallback.
    }

    const epsgMatches = [...wkt.matchAll(EPSG_AUTHORITY_REGEX)]
    const last = epsgMatches[epsgMatches.length - 1]
    const crs = last ? `EPSG:${last[1]}` : undefined

    const proj4Match = wkt.match(PROJ4_EXTENSION_REGEX)
    const proj4 =
      proj4Match?.[1] ??
      (crs ? this.registerProjectionDefinition(crs, wkt, 'raw WKT') : undefined)

    return { crs, proj4 }
  }

  private registerProjectionDefinition(
    crs: string,
    definition: string | proj4.ProjectionDefinition,
    sourceLabel: string
  ): string | undefined {
    if (!/^EPSG:\d+$/i.test(crs)) return undefined

    try {
      const code = crs.toUpperCase()
      proj4.defs(code, definition)
      // Validate immediately: proj4.defs can accept unusable definitions.
      proj4(code, 'EPSG:4326')
      console.info(
        `[zarr-layer] Registered proj4 definition for "${code}" from ${sourceLabel}.`
      )
      return code
    } catch (err) {
      console.warn(
        `[zarr-layer] Failed to register projection for "${crs}" from ${sourceLabel}: ` +
          `${err instanceof Error ? err.message : err}`
      )
      return undefined
    }
  }

  private applyInferredProjection(
    candidate: { crs?: string; proj4?: string },
    source: string
  ): void {
    if (this._crsOverride) return

    const inferredProj4 = this.normalizeProj4Definition(candidate.proj4, source)
    if (inferredProj4 && !this.proj4) {
      this.proj4 = inferredProj4
      if (!candidate.crs) {
        console.info(
          `[zarr-layer] Inferred proj4 from ${source} without CRS code; ` +
            `continuing CRS inference from other metadata.`
        )
      }
    }

    if (!candidate.crs) return
    const normalized = this.normalizeCrsCode(candidate.crs)
    if (!normalized) return

    if (normalized === 'EPSG:4326' || normalized === 'EPSG:3857') {
      this.crs = normalized
      this._crsFromMetadata = true
      this._pendingMetadataCrs = null
      console.info(
        `[zarr-layer] Using inferred CRS "${normalized}" from ${source}.`
      )
      return
    }

    if (this.proj4) {
      this.crs = normalized as CRS
      this._crsFromMetadata = true
      this._pendingMetadataCrs = null
      console.info(
        `[zarr-layer] Using inferred CRS "${normalized}" from ${source} ` +
          `with proj4 reprojection.`
      )
      return
    }

    const preRegistered = this.normalizeProj4Definition(
      normalized,
      `${source} (pre-registered proj4 definition)`
    )
    if (preRegistered) {
      this.proj4 = preRegistered
      this.crs = normalized as CRS
      this._crsFromMetadata = true
      this._pendingMetadataCrs = null
      console.info(
        `[zarr-layer] Using inferred CRS "${normalized}" from ${source} ` +
          `with pre-registered proj4 definition.`
      )
      return
    }

    this._pendingMetadataCrs = normalized
    console.warn(
      `[zarr-layer] Inferred CRS "${normalized}" from ${source}, but no proj4 definition was found. ` +
        `Will attempt resolver/online lookup before final fallback.`
    )
  }

  private inferFromProjConvention(
    attrs: Record<string, unknown> | null
  ): { crs?: string; proj4?: string } | null {
    if (!attrs) return null

    const fromCode = this.normalizeCrsCode(attrs['proj:code'])
    const fromEpsg = this.normalizeCrsCode(attrs['proj:epsg'])
    const fromCrs = this.normalizeCrsCode(attrs['proj:crs'])

    const proj4 =
      typeof attrs['proj:proj4'] === 'string' ? attrs['proj:proj4'] : undefined

    const wktCandidate =
      attrs['proj:wkt2'] ?? attrs['proj:wkt'] ?? attrs['proj:wkt2_2019']
    const fromWkt = this.extractFromWkt(wktCandidate)

    let fromProjJson: string | null = null
    const projJson = asRecord(attrs['proj:projjson'])
    if (projJson) {
      const id = asRecord(projJson.id)
      const authority =
        typeof id?.authority === 'string' ? id.authority.toUpperCase() : null
      const code = id?.code
      if (authority === 'EPSG') {
        fromProjJson = this.normalizeCrsCode(code)
      }
    }

    const crs = fromCode ?? fromEpsg ?? fromCrs ?? fromProjJson ?? fromWkt.crs
    const inferredProj4 = proj4 ?? fromWkt.proj4

    if (!crs && !inferredProj4) return null
    return { crs: crs ?? undefined, proj4: inferredProj4 }
  }

  private inferFromSpatialRef(
    attrs: Record<string, unknown> | null
  ): { crs?: string; proj4?: string } | null {
    if (!attrs) return null
    const wkt = attrs.crs_wkt ?? attrs.spatial_ref
    const parsed = this.extractFromWkt(wkt)
    if (!parsed.crs && !parsed.proj4) return null
    return parsed
  }

  private listAncestorPrefixes(path: string): string[] {
    const cleaned = path.replace(/^\/+|\/+$/g, '')
    if (!cleaned) return ['']
    const parts = cleaned.split('/')
    const prefixes: string[] = []
    for (let i = parts.length; i >= 0; i--) {
      const prefix = parts.slice(0, i).join('/')
      prefixes.push(prefix)
    }
    return Array.from(new Set(prefixes))
  }

  private inferCrsFromAttrs(
    projCandidates: Array<{
      source: string
      attrs: Record<string, unknown> | null
    }>,
    spatialRefCandidates: Array<{
      source: string
      attrs: Record<string, unknown> | null
    }>
  ): void {
    if (this._crsOverride) return

    for (const candidate of projCandidates) {
      const inferred = this.inferFromProjConvention(candidate.attrs)
      if (!inferred) continue
      this.logCrsCandidate(candidate.source, 'proj', inferred)
      this.applyInferredProjection(inferred, candidate.source)
      if (this._crsFromMetadata) return
    }

    for (const candidate of spatialRefCandidates) {
      const inferred = this.inferFromSpatialRef(candidate.attrs)
      if (!inferred) continue
      this.logCrsCandidate(candidate.source, 'spatial_ref', inferred)
      this.applyInferredProjection(inferred, candidate.source)
      if (this._crsFromMetadata) return
    }
  }

  private async lookupProj4Online(crs: string): Promise<string | null> {
    const normalized = this.normalizeCrsCode(crs)
    const match = normalized?.match(/^EPSG:(\d+)$/i)
    if (!match) return null

    const code = `EPSG:${match[1]}`
    let pending = ZarrStore._proj4LookupCache.get(code)
    if (!pending) {
      pending = (async () => {
        try {
          const response = await fetch(`https://epsg.io/${match[1]}.proj4`)
          if (!response.ok) return null
          const text = (await response.text()).trim()
          return text || null
        } catch {
          return null
        }
      })()
      ZarrStore._proj4LookupCache.set(code, pending)
    }
    return pending
  }

  private async resolveProj4IfNeeded(): Promise<void> {
    if (this.proj4) {
      const validated = this.normalizeProj4Definition(
        this.proj4,
        'existing proj4 definition'
      )
      if (validated) {
        this.proj4 = validated
        return
      }
      this.proj4 = null
    }

    let targetCrs: string | null = null
    let reason: ResolveProj4Context['reason'] = 'metadata-crs'

    if (this._pendingUserCrs) {
      targetCrs = this._pendingUserCrs
      reason = 'user-crs'
    } else if (this._pendingMetadataCrs) {
      targetCrs = this._pendingMetadataCrs
      reason = 'metadata-crs'
    } else if (this.crs !== 'EPSG:4326' && this.crs !== 'EPSG:3857') {
      targetCrs = this.crs
      reason = 'metadata-crs'
    }

    if (!targetCrs) return

    const context: ResolveProj4Context = {
      source: this.source,
      variable: this.variable,
      reason,
    }

    let resolved: string | null | undefined = null
    if (this.resolveProj4) {
      try {
        resolved = await this.resolveProj4(targetCrs, context)
      } catch (err) {
        console.warn(
          `[zarr-layer] resolveProj4 callback failed for "${targetCrs}": ` +
            `${err instanceof Error ? err.message : err}`
        )
      }
    } else if (this.enableOnlineProj4Lookup) {
      console.info(
        `[zarr-layer] Attempting online EPSG lookup for "${targetCrs}".`
      )
      resolved = await this.lookupProj4Online(targetCrs)
      if (resolved) {
        console.info(
          `[zarr-layer] Resolved proj4 for "${targetCrs}" from online EPSG lookup: ${resolved}`
        )
      } else {
        console.warn(
          `[zarr-layer] Online EPSG lookup returned no proj4 definition for "${targetCrs}".`
        )
      }
    }

    const normalized = this.normalizeProj4Definition(
      resolved,
      this.resolveProj4 ? 'resolveProj4 callback' : 'online EPSG lookup'
    )
    if (!normalized) {
      console.warn(
        `[zarr-layer] Could not resolve a usable proj4 definition for "${targetCrs}".`
      )
      return
    }

    this.proj4 = normalized
    this.crs = targetCrs as CRS
    if (reason === 'user-crs') {
      this._crsOverride = true
      this._pendingUserCrs = null
    } else {
      this._crsFromMetadata = true
      this._pendingMetadataCrs = null
    }

    console.info(
      `[zarr-layer] Using resolved proj4 definition for "${targetCrs}".`
    )
  }

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
    resolveProj4,
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
    this.proj4 =
      proj4 !== undefined
        ? this.normalizeProj4Definition(proj4, 'layer options')
        : null
    if (typeof resolveProj4 === 'function') {
      this.resolveProj4 = resolveProj4
      this.enableOnlineProj4Lookup = false
    } else {
      this.enableOnlineProj4Lookup = resolveProj4 !== false
    }
    if (crs) {
      const normalized = this.normalizeCrsCode(crs) ?? crs.toUpperCase()
      if (normalized === 'EPSG:4326' || normalized === 'EPSG:3857') {
        this.crs = normalized
        this._crsOverride = true
        console.info(
          `[zarr-layer] Using user-provided CRS override "${this.crs}" for "${this.variable}".`
        )
      } else if (this.proj4) {
        this.crs = normalized as CRS
        this._crsOverride = true
        console.info(
          `[zarr-layer] Using user-provided CRS "${normalized}" with proj4 definition.`
        )
      } else {
        this._pendingUserCrs = normalized
        console.warn(
          `[zarr-layer] CRS "${crs}" requires 'proj4' to render correctly. ` +
            `Attempting resolver and metadata inference.`
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
        const baseStore = new zarr.FetchStore(this.source, {
          useSuffixRequest: true,
        })
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

    await this.resolveProj4IfNeeded()
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

  hasMissingVariableMetadata(): boolean {
    return this._variableMetadataMissing
  }

  getVariablePath(): string {
    return this.variable
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
    // Bypass cache when transformRequest or customStore is provided
    const bypassCache = this.transformRequest || this.customStore
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
        if (!bypassCache) ZarrStore._cache.set(cacheKey, zmetadata)
      } else {
        try {
          zmetadata = (await this._getJSON(
            '/.zmetadata'
          )) as ZarrV2ConsolidatedMetadata
          if (!bypassCache) ZarrStore._cache.set(cacheKey, zmetadata)
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
      try {
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
      } catch (err) {
        const hasChildVariable = Object.keys(v2Metadata.metadata).some((key) =>
          key.endsWith(`/${this.variable}/.zarray`)
        )
        if (hasChildVariable) {
          this._variableMetadataMissing = true
          this.dimensions = []
          this.shape = []
          this.chunks = []
          this.fill_value = null
          this.dtype = null
          this.scaleFactor = 1
          this.addOffset = 0
          return
        }
        throw err
      }
    }

    this.dimensions = zattrs?._ARRAY_DIMENSIONS || []
    this.shape = zarray?.shape || []
    this.chunks = zarray?.chunks || []
    this.fill_value = this.normalizeFillValue(zarray?.fill_value ?? null)
    this.dtype = zarray?.dtype || null
    this.scaleFactor = zattrs?.scale_factor ?? 1
    this.addOffset = zattrs?.add_offset ?? 0

    const groupPath = basePath.includes('/')
      ? basePath.slice(0, basePath.lastIndexOf('/'))
      : ''
    const prefixes = this.listAncestorPrefixes(groupPath)
    const spatialRefAttrs =
      (v2Metadata.metadata[
        prefixes
          .map((prefix) =>
            prefix ? `${prefix}/spatial_ref/.zattrs` : 'spatial_ref/.zattrs'
          )
          .find((key) => Boolean(v2Metadata.metadata[key])) ?? ''
      ] as Record<string, unknown> | undefined) ?? null
    const nearestGroupAttrs =
      (v2Metadata.metadata[
        prefixes
          .map((prefix) => (prefix ? `${prefix}/.zattrs` : '.zattrs'))
          .find((key) => Boolean(v2Metadata.metadata[key])) ?? '.zattrs'
      ] as Record<string, unknown> | undefined) ?? null

    this.inferCrsFromAttrs(
      [
        { source: `${basePath}/.zattrs`, attrs: asRecord(zattrs) },
        { source: 'nearest group .zattrs', attrs: asRecord(nearestGroupAttrs) },
        { source: 'root .zattrs', attrs: asRecord(rootAttrs) },
      ],
      [
        { source: 'spatial_ref/.zattrs', attrs: asRecord(spatialRefAttrs) },
        { source: `${basePath}/.zattrs`, attrs: asRecord(zattrs) },
        {
          source: 'nearest group .zattrs',
          attrs: asRecord(nearestGroupAttrs),
        },
        { source: 'root .zattrs', attrs: asRecord(rootAttrs) },
      ]
    )

    await this._computeDimIndices()
  }

  private async _loadV3() {
    const metadataCacheKey = `v3:${this.source}`
    // Bypass cache when transformRequest or customStore is provided
    const bypassCache = this.transformRequest || this.customStore
    let metadata = bypassCache
      ? undefined
      : (ZarrStore._cache.get(metadataCacheKey) as
          | ZarrV3GroupMetadata
          | undefined)
    if (!metadata) {
      metadata = (await this._getJSON('/zarr.json')) as ZarrV3GroupMetadata
      if (!bypassCache) {
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
    let arrayMetadata = bypassCache
      ? undefined
      : (ZarrStore._cache.get(arrayCacheKey) as ZarrV3ArrayMetadata | undefined)
    if (!arrayMetadata) {
      try {
        arrayMetadata = (await this._getJSON(
          `/${arrayKey}/zarr.json`
        )) as ZarrV3ArrayMetadata
      } catch (err) {
        const consolidated = metadata.consolidated_metadata?.metadata
        const hasChildVariable = consolidated
          ? Object.entries(consolidated).some(
              ([key, value]) =>
                key.endsWith(`/${this.variable}`) &&
                (value as { node_type?: string }).node_type === 'array'
            )
          : false
        if (hasChildVariable) {
          this._variableMetadataMissing = true
          this.dimensions = []
          this.shape = []
          this.chunks = []
          this.fill_value = null
          this.dtype = null
          this.scaleFactor = 1
          this.addOffset = 0
          return
        }
        throw err
      }
      if (!bypassCache) ZarrStore._cache.set(arrayCacheKey, arrayMetadata)
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

    const consolidated = metadata.consolidated_metadata?.metadata
    const groupPath = arrayKey.includes('/')
      ? arrayKey.slice(0, arrayKey.lastIndexOf('/'))
      : ''
    const prefixes = this.listAncestorPrefixes(groupPath)

    const firstMatchingSpatialRefPath = prefixes
      .map((prefix) => (prefix ? `${prefix}/spatial_ref` : 'spatial_ref'))
      .find((key) => Boolean(consolidated?.[key]))
    const spatialRefAttrs = firstMatchingSpatialRefPath
      ? asRecord(
          (
            consolidated?.[firstMatchingSpatialRefPath] as {
              attributes?: unknown
            }
          )?.attributes
        )
      : null

    const nearestGroupAttrs = (() => {
      const key = prefixes.find((prefix) => Boolean(consolidated?.[prefix]))
      if (!key) return null
      return asRecord(
        (consolidated?.[key] as { attributes?: unknown })?.attributes
      )
    })()

    this.inferCrsFromAttrs(
      [
        { source: `${arrayKey}.zarr.json attributes`, attrs: asRecord(attrs) },
        { source: 'nearest group attributes', attrs: nearestGroupAttrs },
        { source: 'root attributes', attrs: asRecord(metadata.attributes) },
      ],
      [
        { source: 'spatial_ref attributes', attrs: spatialRefAttrs },
        { source: `${arrayKey}.zarr.json attributes`, attrs: asRecord(attrs) },
        { source: 'nearest group attributes', attrs: nearestGroupAttrs },
        { source: 'root attributes', attrs: asRecord(metadata.attributes) },
      ]
    )

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
      const firstArray = await openArray(
        this.root.resolve(`${firstLevel}/${this.variable}`)
      )
      const lastArray = await openArray(
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

      const openArray = (loc: zarr.Location<Readable>) => {
        if (this.version === 2) return zarr.open.v2(loc, { kind: 'array' })
        if (this.version === 3) return zarr.open.v3(loc, { kind: 'array' })
        return zarr.open(loc, { kind: 'array' })
      }

      const lonName = this.spatialDimensions.lon ?? this.dimIndices.lon.name
      const latName = this.spatialDimensions.lat ?? this.dimIndices.lat.name

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
          const variablePrefix = this.variable.includes('/')
            ? this.variable.slice(0, this.variable.lastIndexOf('/'))
            : this.variable
          const varCandidates = candidates.filter((c) =>
            c.path.startsWith(`${variablePrefix}/`)
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
        console.warn(
          `[zarr-layer] Inferred CRS "${this.crs}" from bounds heuristic ` +
            `(max |x| > 360) for "${this.variable}".`
        )
      } else if (this.crs === 'EPSG:4326') {
        console.warn(
          `[zarr-layer] CRS defaulted to "${this.crs}" for "${this.variable}". ` +
            `No usable CRS code was inferred from proj/spatial_ref metadata.`
        )
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
    let crs: CRS = this.crs
    const normalizedMetadataCrs = this.normalizeCrsCode(metadata.crs)
    if (normalizedMetadataCrs && !this._crsOverride) {
      if (
        normalizedMetadataCrs === 'EPSG:4326' ||
        normalizedMetadataCrs === 'EPSG:3857'
      ) {
        crs = normalizedMetadataCrs
        this._crsFromMetadata = true
      } else if (this.proj4) {
        crs = normalizedMetadataCrs as CRS
        this._crsFromMetadata = true
      }
    }

    return {
      levels,
      maxLevelIndex,
      tileSize: 128, // Will be overridden by chunk shape
      crs,
    }
  }

  /**
   * Discover child dataset groups within a datatree root.
   *
   * Walks consolidated metadata to find child groups that contain the given
   * variable as a child array. Returns an empty array if the source is a
   * single-dataset store (variable lives directly at root or root has multiscales).
   *
   * For child groups with multiscales layout metadata, bounds are derived from
   * translation/scale/shape. Otherwise, bounds are null and will be resolved
   * per-store from coordinate arrays during initialization.
   */
  static discoverDatasets(
    metadata: ZarrV2ConsolidatedMetadata | ZarrV3GroupMetadata | null,
    variable: string,
    version: 2 | 3 | null
  ): import('./types').DatasetDescriptor[] {
    if (!metadata) return []

    // If root has multiscales, it's a single-dataset store — not a datatree
    if (version === 3 || version === null) {
      const v3 = metadata as ZarrV3GroupMetadata
      if (v3.attributes?.multiscales) return []
    }
    if (version === 2 || version === null) {
      const v2 = metadata as ZarrV2ConsolidatedMetadata
      const rootAttrs = v2.metadata?.['.zattrs'] as
        | { multiscales?: unknown }
        | undefined
      if (rootAttrs?.multiscales) return []
      if (v2.metadata?.[`${variable}/.zarray`]) return []
    }

    // Collect child groups that contain `variable` as a direct child array
    const groups = new Map<string, import('./types').DatasetDescriptor>()

    if (version === 3 || version === null) {
      const v3 = metadata as ZarrV3GroupMetadata
      const consolidated = v3.consolidated_metadata?.metadata
      if (consolidated) {
        const rootVariable = consolidated[variable] as
          | ZarrV3ArrayMetadata
          | undefined
        if (rootVariable?.node_type === 'array') return []

        const varSuffix = `/${variable}`
        for (const [key, value] of Object.entries(consolidated)) {
          // Match patterns like "region_a/temperature" (one-level group + variable)
          if (!key.endsWith(varSuffix)) continue
          const v3Meta = value as ZarrV3ArrayMetadata
          if (v3Meta.node_type !== 'array') continue

          const groupPath = key.slice(0, -varSuffix.length)
          // Skip if variable is at root level (no group prefix)
          if (!groupPath || groupPath.includes('/')) continue

          if (!groups.has(groupPath)) {
            groups.set(groupPath, { path: groupPath, bounds: null })
          }
        }

        // Try to derive bounds from multiscales layout metadata on each child group
        for (const [groupPath, descriptor] of groups) {
          const groupMeta = consolidated[groupPath] as
            | { node_type?: string; attributes?: Record<string, unknown> }
            | undefined
          if (!groupMeta?.attributes?.multiscales) continue

          const ms = groupMeta.attributes.multiscales as
            | UntiledMultiscaleMetadata
            | Multiscale[]
          if (
            'layout' in ms &&
            Array.isArray(ms.layout) &&
            ms.layout.length > 0
          ) {
            // Use the finest resolution level (first in layout) to compute bounds
            const finest = ms.layout[0]
            const arrayKey = `${groupPath}/${finest.asset}/${variable}`
            const arrayMeta = consolidated[arrayKey] as
              | ZarrV3ArrayMetadata
              | undefined
            if (
              arrayMeta?.shape &&
              finest.transform?.scale &&
              finest.transform?.translation
            ) {
              const [sx, sy] = finest.transform.scale
              const [tx, ty] = finest.transform.translation
              // shape is [..., y, x] — take last two dims
              const height = arrayMeta.shape[arrayMeta.shape.length - 2]
              const width = arrayMeta.shape[arrayMeta.shape.length - 1]
              if (height && width) {
                const xMin = tx
                const yMax = ty
                const xMax = tx + sx * width
                const yMin = ty + sy * height // sy is typically negative
                descriptor.bounds = [
                  Math.min(xMin, xMax),
                  Math.min(yMin, yMax),
                  Math.max(xMin, xMax),
                  Math.max(yMin, yMax),
                ]
              }
            }
          }
        }
      }
    }

    if ((version === 2 || version === null) && groups.size === 0) {
      const v2 = metadata as ZarrV2ConsolidatedMetadata
      if (v2.metadata) {
        const varArraySuffix = `/${variable}/.zarray`
        for (const key of Object.keys(v2.metadata)) {
          if (!key.endsWith(varArraySuffix)) continue
          const groupPath = key.slice(0, -varArraySuffix.length)
          if (!groupPath || groupPath.includes('/')) continue

          if (!groups.has(groupPath)) {
            groups.set(groupPath, { path: groupPath, bounds: null })
          }
        }
      }
    }

    return Array.from(groups.values())
  }

  static clearCache() {
    ZarrStore._cache.clear()
    ZarrStore._storeCache.clear()
  }
}
