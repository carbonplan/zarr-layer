/**
 * @module zarr-layer
 *
 * MapLibre/Mapbox custom layer implementation for rendering Zarr datasets.
 * Implements CustomLayerInterface for direct WebGL rendering.
 */

import type { Readable } from '@zarrita/storage'
import {
  loadDimensionValues,
  getBands,
  toSelectorProps,
  normalizeSelector,
  hashSelector,
} from './zarr-utils'
import { ZarrStore } from './zarr-store'
import { maplibreFragmentShaderSource } from './shaders'
import { ColormapState } from './colormap'
import { ZarrRenderer } from './zarr-renderer'
import type { CustomShaderConfig, UniformValue } from './renderer-types'
import type {
  Bounds,
  ColormapArray,
  SpatialDimensions,
  DimIndicesProps,
  LoadingState,
  LoadingStateCallback,
  MapLike,
  Selector,
  NormalizedSelector,
  ZarrLayerOptions,
  TransformRequest,
  OnAuthError,
} from './types'
import type { RenderContext, TileId } from './renderer-types'
import { RegionRenderer, type TileRequest } from './region-renderer'
import { MAX_ACTIVE_REGION_FETCHES } from './region-utils'
import {
  computeWorldOffsets,
  resolveProjectionParams,
  isGlobeProjection as checkGlobeProjection,
} from './map-utils'
import { MAPBOX_IDENTITY_MATRIX } from './mapbox-utils'
import {
  mapboxEcefClipMatrix,
  parseMapboxRenderParameters,
} from './mapbox-globe-matrix'
import type { QueryGeometry, QueryOptions, QueryResult } from './query/types'
import { SPATIAL_DIM_NAMES } from './constants'
import { ZarrLayerNotReadyError } from './errors'

/** MapLibre's terrain tile ID. `wrap` is the world copy, which draws the
 *  same content as every other copy, so only `canonical` is used. */
type TerrainTileID = { canonical: TileId; wrap?: number }

/** Screen pixels per tile when asking the map for its tile cover, and the
 *  size draped tiles are assumed to have when the map does not say. */
const COVERING_TILE_SIZE = 512

/** Scale-mismatch ratio beyond which a locked fixedDataScale is re-derived
 *  from the clim and cached tiles are re-normalized. Conservative: float32
 *  texture data only truly degrades several more orders of magnitude out,
 *  but refetches go through the decoded-chunk cache so an eager threshold
 *  costs little. */
const DATA_SCALE_REFRESH_RATIO = 1e6

/** Settle as soon as `signal` aborts, so a caller tearing the layer down isn't
 *  held by a wait it can no longer cancel from the inside. */
function abortable<T>(work: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return work
  if (signal.aborted) return Promise.reject(signal.reason)
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(signal.reason)
    signal.addEventListener('abort', onAbort, { once: true })
    work.then(resolve, reject).finally(() => {
      signal.removeEventListener('abort', onAbort)
    })
  })
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)))
  return t * t * (3 - 2 * t)
}

function mapboxGlobeToMercatorTransition(zoom: number): number {
  // Mirrors Mapbox's current globeToMercatorTransition(zoom) helper.
  return smoothstep(5, 6, zoom)
}

/** GLSL has no zero-length arrays, so an empty one could never compile. */
function checkUniforms(
  uniforms: Record<string, UniformValue>
): Record<string, UniformValue> {
  // Values arrive from JavaScript too, so the declared type is not trusted.
  for (const [name, value] of Object.entries(uniforms) as [string, unknown][]) {
    if (typeof value === 'number') continue
    if (
      !Array.isArray(value) &&
      !(value instanceof Float32Array) &&
      !(value instanceof Float64Array)
    ) {
      throw new Error(
        `[ZarrLayer] Uniform '${name}' must be a number or an array of numbers.`
      )
    }
    if (value.length === 0) {
      throw new Error(`[ZarrLayer] Uniform array '${name}' is empty.`)
    }
    for (const element of value) {
      if (typeof element !== 'number') {
        throw new Error(
          `[ZarrLayer] Uniform array '${name}' holds a non-number: ${String(
            element
          )}.`
        )
      }
    }
  }
  return uniforms
}

export class ZarrLayer {
  readonly type: 'custom' = 'custom'
  readonly renderingMode: '2d' | '3d'
  readonly wrapTileId = true

  id: string
  private url: string
  private variable: string
  private zarrVersion: 2 | 3 | null = null
  private spatialDimensions: SpatialDimensions
  private bounds: Bounds | undefined
  private crs: string | undefined
  private latIsAscending: boolean | null = null
  private selector: Selector
  private invalidate: () => void

  private colormap: ColormapState
  private clim: [number, number]
  private opacity: number
  private minZoom: number
  private maxZoom: number
  private selectorHash: string = ''

  private _fillValue: number | null = null
  private scaleFactor: number = 1
  private offset: number = 0
  private fixedDataScale: number
  private minRegionSize: number
  private maxRegionFetches: number
  // Once true, fixedDataScale is locked (the renderer has captured it)
  private dataScaleLocked: boolean = false

  private gl: WebGL2RenderingContext | undefined
  private map: MapLike | null = null
  private renderer: ZarrRenderer | null = null
  private regionRenderer: RegionRenderer | null = null
  private tileNeedsRender: boolean = true
  /** MapLibre redraws its terrain tiles when this changes (see
   *  `renderToTerrainTile`). Bumped whenever the drawn content changes. */
  terrainTileRevision: number = 0

  private projectionChangeHandler: (() => void) | null = null
  private mapRemoveHandler: (() => void) | null = null
  private resolveGl(
    map: MapLike,
    gl: WebGL2RenderingContext | WebGLRenderingContext | null
  ): WebGL2RenderingContext {
    const isWebGL2 =
      gl &&
      typeof gl.getUniformLocation === 'function' &&
      typeof (gl as WebGL2RenderingContext).drawBuffers === 'function'
    if (isWebGL2) {
      return gl as WebGL2RenderingContext
    }

    const describe = (obj: unknown) =>
      obj
        ? {
            type: obj.constructor?.name,
            keys: Object.keys(obj),
          }
        : null
    console.error('Invalid WebGL2 context passed to onAdd', {
      providedGl: describe(gl),
      painterGl: describe(map?.painter?.context?.gl),
      rendererGl: describe(map?.renderer?.getContext?.()),
    })
    throw new Error('`map` did not provide a valid WebGL2 context')
  }

  private zarrStore: ZarrStore | null = null
  private levelAssets: string[] = []
  private dimIndices: DimIndicesProps = {}
  private dimensionValues: {
    [key: string]: Float64Array | number[] | string[]
  } = {}
  private normalizedSelector: NormalizedSelector = {}
  private isRemoved: boolean = false
  /** Settles when the current dataset's metadata init finishes; never rejects
   *  (failures land in `initError`). Null until the layer is added to a map. */
  private initPromise: Promise<void> | null = null
  private readyPromise: Promise<void> | null = null
  private fragmentShaderSource: string = maplibreFragmentShaderSource
  private customFrag: string | undefined
  private customUniforms: Record<string, UniformValue> = {}
  private bandNames: string[] = []
  private customShaderConfig: CustomShaderConfig | null = null
  private onLoadingStateChange: LoadingStateCallback | undefined
  private metadataLoading: boolean = false
  private chunksLoading: boolean = false
  private initError: Error | null = null
  private levelLoadError: Error | null = null
  private proj4: string | undefined
  private transformRequest: TransformRequest | undefined
  private onAuthError: OnAuthError | undefined
  private customStore: Readable | undefined
  private renderPoles: boolean
  private lastIsGlobe: boolean | null = null
  private lastZoomInRange: boolean | null = null
  private usingDirectMapboxGlobePath: boolean = false
  private renderPathSwitchPending: boolean = false
  private mapboxRenderParametersSeen: boolean = false

  /** Latch whether this map hands the public render parameters (mapbox-gl >=
   *  3.32) to render/prerender. The gate below also runs from map events and
   *  renderToTile, which never see them, so the latest callback decides.
   *  Until the first callback the layer is draped, which is also the path an
   *  older Mapbox keeps for good. */
  private noteMapboxRenderParameters(renderParameters: unknown): void {
    if (
      !this.mapboxRenderParametersSeen &&
      parseMapboxRenderParameters(renderParameters)
    ) {
      this.mapboxRenderParametersSeen = true
      this.configureMapboxRenderPath()
    }
  }

  /** MapLibre calls render with a single params object, so the render
   *  parameters are only ever seen on Mapbox; this doubles as the Mapbox
   *  discriminator for the direct globe path. */
  private canUseMapboxDirectGlobePath(): boolean {
    return this.mapboxRenderParametersSeen
  }

  /** Mapbox sorts custom layers into draped batches at the start of every
   *  frame, before prerender/render/renderToTile run, and only re-reads
   *  `renderToTile` for the next frame. Flipping it from inside a callback
   *  would leave this frame with a batch that no longer matches the layer and
   *  nothing drawn, so the switch is applied on a microtask: Mapbox's frame is
   *  synchronous, so that lands after the frame is drawn and before the next
   *  one is batched. The decision is re-evaluated when it runs, so a stale
   *  verdict is never applied. */
  private configureMapboxRenderPath(): void {
    if (this.renderPathSwitchPending) return
    if (
      this.shouldUseMapboxDirectGlobePath() === this.usingDirectMapboxGlobePath
    )
      return
    this.renderPathSwitchPending = true
    queueMicrotask(() => {
      this.renderPathSwitchPending = false
      if (this.isRemoved) return
      this.applyMapboxRenderPath(this.shouldUseMapboxDirectGlobePath())
    })
  }

  private shouldUseMapboxDirectGlobePath(): boolean {
    if (!this.map || !this.regionRenderer) return false

    const resolvedProj4 = this.zarrStore?.proj4 ?? this.proj4
    const resolvedCrs = this.zarrStore?.crs ?? this.crs
    const isEcefEligible = !!resolvedProj4 || resolvedCrs === 'EPSG:4326'
    const isGlobe = this.isGlobeProjection()
    const transition =
      this.map.getZoom && isGlobe
        ? mapboxGlobeToMercatorTransition(this.map.getZoom())
        : 1
    // FRAGILE: Mapbox classifies a custom layer as draped whenever renderToTile
    // exists (undocumented behaviour with no public alternative). We use the
    // direct globe path only for terrain-off, ECEF-eligible datasets at the
    // fully-globe endpoint on a Mapbox that passes the public render
    // parameters. During the zoom morph, restore the draped path so Mapbox
    // handles the transition itself with its internal globe/mercator matrices.
    // This means polar coverage can snap back to the draped path near the 5-6
    // zoom morph window; that is intentional, and is currently more stable
    // than trying to carry the direct ECEF path through Mapbox's custom-layer
    // transition contract.
    return (
      this.renderPoles &&
      isEcefEligible &&
      !this.map.getTerrain?.() &&
      isGlobe &&
      this.canUseMapboxDirectGlobePath() &&
      transition <= 1e-3
    )
  }

  private applyMapboxRenderPath(useDirectGlobePath: boolean): void {
    if (useDirectGlobePath === this.usingDirectMapboxGlobePath) return
    this.usingDirectMapboxGlobePath = useDirectGlobePath

    if (useDirectGlobePath) {
      Object.defineProperty(this, 'renderToTile', {
        value: undefined,
        configurable: true,
        writable: true,
      })
    } else if (Object.prototype.hasOwnProperty.call(this, 'renderToTile')) {
      delete (this as { renderToTile?: unknown }).renderToTile
    }
    // Mapbox caches each draped batch's texture and reuses it while the
    // source tiles are unchanged, so after the layer leaves or joins a batch
    // the cached texture still shows the old membership: stale draped pixels
    // under the direct draw one way, a one-frame gap the other. Right after
    // batching, Mapbox asks visible custom layers (draped or not, stopping at
    // the first yes) whether they need a re-drape via shouldRerenderTiles(),
    // and a yes drops the whole cache for that frame. A hidden layer keeps
    // the flag until it is visible again.
    this.tileNeedsRender = true
    this.map?.triggerRepaint?.()
  }

  get fillValue(): number | null {
    return this._fillValue
  }

  private isGlobeProjection(): boolean {
    const projection = this.map?.getProjection ? this.map.getProjection() : null
    return checkGlobeProjection(projection)
  }

  /** Check for projection changes and notify mode. Returns current isGlobe state. */
  private syncProjectionState(): boolean {
    const isGlobe = this.isGlobeProjection()
    if (this.lastIsGlobe !== null && this.lastIsGlobe !== isGlobe) {
      this.regionRenderer?.onProjectionChange(isGlobe)
    }
    this.lastIsGlobe = isGlobe
    return isGlobe
  }

  constructor({
    id,
    source,
    variable,
    selector = {},
    colormap,
    clim,
    opacity = 1,
    minzoom = 0,
    maxzoom = Infinity,
    zarrVersion,
    spatialDimensions = {},
    bounds,
    crs,
    latIsAscending = null,
    fillValue,
    customFrag,
    uniforms,
    minRegionSize = 0,
    maxRegionFetches = MAX_ACTIVE_REGION_FETCHES,
    renderingMode = '3d',
    onLoadingStateChange,
    proj4,
    transformRequest,
    onAuthError,
    store,
    renderPoles = false,
  }: ZarrLayerOptions) {
    if (!id) {
      throw new Error('[ZarrLayer] id is required')
    }
    if (!source && !store) {
      throw new Error(
        '[ZarrLayer] source is required when store is not provided'
      )
    }
    if (!variable) {
      throw new Error('[ZarrLayer] variable is required')
    }
    if (!colormap || !Array.isArray(colormap) || colormap.length === 0) {
      throw new Error(
        '[ZarrLayer] colormap is required and must be an array of [r, g, b] or hex string values'
      )
    }
    if (!clim || !Array.isArray(clim) || clim.length !== 2) {
      throw new Error('[ZarrLayer] clim is required and must be [min, max]')
    }
    if (proj4 && !bounds) {
      console.warn(
        `[ZarrLayer] proj4 provided without explicit bounds. ` +
          `Bounds will be derived from coordinate arrays if available (see subsequent log for values). ` +
          `For best performance, provide bounds in source CRS units.`
      )
    }

    this.id = id
    this.url = source ?? id // Use id as fallback identifier when using custom store
    this.variable = variable
    this.zarrVersion = zarrVersion ?? null
    this.spatialDimensions = spatialDimensions
    this.bounds = bounds
    this.crs = crs
    this.latIsAscending = latIsAscending ?? null
    this.selector = selector
    this.normalizedSelector = normalizeSelector(selector)
    this.selectorHash = hashSelector(this.normalizedSelector)
    this.renderingMode = renderingMode
    this.invalidate = () => {}
    this.colormap = new ColormapState(colormap)
    this.clim = clim
    this.fixedDataScale = Math.max(Math.abs(clim[0]), Math.abs(clim[1]), 1)
    this.opacity = opacity
    this.minZoom = minzoom
    this.maxZoom = maxzoom
    if (!Number.isFinite(minRegionSize) || minRegionSize < 0) {
      throw new Error(
        `[ZarrLayer] minRegionSize must be a non-negative number, got ${minRegionSize}.`
      )
    }
    this.minRegionSize = minRegionSize
    if (!Number.isInteger(maxRegionFetches) || maxRegionFetches < 1) {
      throw new Error(
        `[ZarrLayer] maxRegionFetches must be a positive integer, got ${maxRegionFetches}.`
      )
    }
    this.maxRegionFetches = maxRegionFetches

    this.customFrag = customFrag
    this.customUniforms = checkUniforms(uniforms || {})

    this.refreshCustomShaderConfig()

    if (fillValue !== undefined) this._fillValue = fillValue
    this.onLoadingStateChange = onLoadingStateChange
    this.proj4 = proj4
    this.transformRequest = transformRequest
    this.onAuthError = onAuthError
    this.customStore = store
    this.renderPoles = renderPoles
  }

  /**
   * Recompute the band list and the shader variant it implies. More than one
   * band, or any custom fragment, means the shader samples per-band textures
   * rather than the main one, which the renderer needs to know so it can skip
   * building data only the main texture would consume.
   */
  private refreshCustomShaderConfig(): void {
    this.bandNames = getBands(this.variable, this.normalizedSelector)
    this.customShaderConfig =
      this.bandNames.length > 1 || this.customFrag
        ? {
            bands: this.bandNames,
            customFrag: this.customFrag,
            customUniforms: this.customUniforms,
          }
        : null
    this.regionRenderer?.setRendersFromBandTextures(!!this.customShaderConfig)
  }

  private emitLoadingState(): void {
    if (!this.onLoadingStateChange) return
    this.onLoadingStateChange({
      loading: this.metadataLoading || this.chunksLoading,
      metadata: this.metadataLoading,
      chunks: this.chunksLoading,
      error: this.initError ?? this.levelLoadError,
    })
  }

  private handleChunkLoadingChange = (state: LoadingState): void => {
    this.chunksLoading = state.chunks
    this.levelLoadError = state.error ?? null
    this.emitLoadingState()
  }

  setOpacity(opacity: number) {
    this.opacity = opacity
    this.invalidate()
  }

  setClim(clim: [number, number]) {
    this.clim = clim
    const nextScale = Math.max(Math.abs(clim[0]), Math.abs(clim[1]), 1)
    // Allow fixedDataScale to update until the renderer captures it
    if (!this.dataScaleLocked) {
      this.fixedDataScale = nextScale
    } else if (
      nextScale * DATA_SCALE_REFRESH_RATIO < this.fixedDataScale ||
      this.fixedDataScale * DATA_SCALE_REFRESH_RATIO < nextScale
    ) {
      // Texture data is normalized by fixedDataScale at upload. When the
      // locked scale is orders of magnitude off the new clim (e.g. the layer
      // was created with a dtype-range clim like ±3.4e38 before the caller
      // knew the data range), normalized values underflow float32 and the
      // data can never render correctly again. Adopt the new scale and
      // refetch cached tiles through the decoded-chunk cache so textures
      // are re-normalized.
      this.fixedDataScale = nextScale
      this.regionRenderer?.setDataScale(nextScale)
    }
    this.invalidate()
  }

  setColormap(colormap: ColormapArray) {
    this.colormap.apply(colormap)
    if (this.gl) {
      this.colormap.upload(this.gl)
    }
    this.invalidate()
  }

  setUniforms(uniforms: Record<string, UniformValue>) {
    if (!this.customShaderConfig) {
      console.warn(
        '[ZarrLayer] setUniforms() called but layer was not created with customFrag. ' +
          'Uniforms will not be applied. Recreate the layer with customFrag and uniforms options.'
      )
      return
    }
    this.customUniforms = { ...this.customUniforms, ...checkUniforms(uniforms) }
    this.customShaderConfig.customUniforms = this.customUniforms
    this.invalidate()
  }

  async setVariable(variable: string) {
    if (variable === this.variable) return
    // Queries and `ready` track the new dataset from here on, so a call that
    // lands mid-swap waits for the new store rather than reading the old one.
    this.readyPromise = null
    this.initPromise = this.trackInit(this._setVariableAsync(variable))
    await this.initPromise
  }

  private async _setVariableAsync(variable: string) {
    this.metadataLoading = true
    this.levelLoadError = null
    this.emitLoadingState()

    try {
      this.initError = null
      this.variable = variable
      if (this.zarrStore) {
        this.zarrStore.cleanup()
        this.zarrStore = null
      }
      this.dimensionValues = {}
      this._fillValue = null
      // Reset and recompute fixedDataScale from current clim for the new dataset
      this.dataScaleLocked = false
      this.fixedDataScale = Math.max(
        Math.abs(this.clim[0]),
        Math.abs(this.clim[1]),
        1
      )
      await this.initialize()
      await this.initializeRenderer()
      this.invalidate()
    } catch (err) {
      this.initError = err instanceof Error ? err : new Error(String(err))
      console.error('[zarr-layer] Failed to reset:', this.initError.message)
      if (this.regionRenderer && this.gl) {
        this.regionRenderer.dispose(this.gl)
        this.regionRenderer = null
      }
      if (this.zarrStore) {
        this.zarrStore.cleanup()
        this.zarrStore = null
      }
    } finally {
      this.metadataLoading = false
      this.emitLoadingState()
    }
  }

  async setSelector(selector: Selector) {
    const normalized = normalizeSelector(selector)
    const nextHash = hashSelector(normalized)
    if (nextHash === this.selectorHash) {
      return
    }
    this.selectorHash = nextHash
    this.selector = selector
    this.normalizedSelector = normalized

    this.refreshCustomShaderConfig()

    if (this.regionRenderer) {
      await this.regionRenderer.setSelector(this.normalizedSelector)
    }
    this.invalidate()
  }

  onAdd(
    map: MapLike,
    gl: WebGL2RenderingContext | WebGLRenderingContext
  ): void {
    this.initPromise = this.trackInit(this._onAddAsync(map, gl))
  }

  /**
   * Funnel an initialization run into `initError` so `initPromise` always
   * fulfills. `_onAddAsync` does its own error handling once running, but it
   * can reject before reaching that — `resolveGl` throws on a context that
   * isn't WebGL2 — and a rejection here would otherwise escape `ready` and
   * `queryData` raw, and go unhandled when nobody awaits either.
   */
  private trackInit(run: Promise<void>): Promise<void> {
    return run.catch((err) => {
      this.initError = err instanceof Error ? err : new Error(String(err))
      console.error(
        `[zarr-layer] Failed to initialize: ${this.initError.message}`
      )
      this.metadataLoading = false
      this.emitLoadingState()
    })
  }

  private async _onAddAsync(
    map: MapLike,
    gl: WebGL2RenderingContext | WebGLRenderingContext
  ): Promise<void> {
    this.map = map
    const resolvedGl = this.resolveGl(map, gl)
    this.gl = resolvedGl

    // `map.remove()` tears down the style without calling `onRemove`, so the
    // layer releases its store and GPU resources off the map's own event.
    // Registered before initialization so a failed init still leaves a hook.
    this.mapRemoveHandler = () => this._disposeResources(resolvedGl)
    if (typeof map.on === 'function') {
      map.on('remove', this.mapRemoveHandler)
    }

    this.invalidate = () => {
      this.tileNeedsRender = true
      this.terrainTileRevision++
      if (map.triggerRepaint) map.triggerRepaint()
    }

    this.initError = null
    this.levelLoadError = null
    this.metadataLoading = true
    this.emitLoadingState()

    try {
      this.colormap.upload(resolvedGl as WebGL2RenderingContext)
      this.renderer = new ZarrRenderer(
        resolvedGl as WebGL2RenderingContext,
        this.fragmentShaderSource
      )

      this.projectionChangeHandler = () => {
        const isGlobe = this.isGlobeProjection()
        if (this.lastIsGlobe !== isGlobe) {
          this.regionRenderer?.onProjectionChange(isGlobe)
          this.lastIsGlobe = isGlobe
        }
        this.configureMapboxRenderPath()
      }
      if (typeof map.on === 'function' && this.projectionChangeHandler) {
        map.on('projectionchange', this.projectionChangeHandler)
        map.on('style.load', this.projectionChangeHandler)
        map.on('move', this.projectionChangeHandler)
      }

      await this.initialize()
      await this.initializeRenderer()
      this.configureMapboxRenderPath()

      const isGlobe = this.isGlobeProjection()
      this.lastIsGlobe = isGlobe
      this.regionRenderer?.onProjectionChange(isGlobe)

      if (this.isZoomInRange()) this.regionRenderer?.update(this.map, this.gl!)
      else this.regionRenderer?.followZoom(this.map, this.gl!)
    } catch (err) {
      this.initError = err instanceof Error ? err : new Error(String(err))
      console.error(
        `[zarr-layer] Failed to initialize: ${this.initError.message}. ` +
          `Use onLoadingStateChange callback to handle errors and call map.removeLayer('${this.id}') to clean up.`
      )
      this._disposeResources(resolvedGl)
    } finally {
      this.metadataLoading = false
      this.emitLoadingState()
    }

    if (!this.initError) {
      this.invalidate()
    }
  }

  private async initializeRenderer() {
    if (!this.zarrStore || !this.gl) return

    if (this.regionRenderer) {
      this.regionRenderer.dispose(this.gl)
    }

    // One renderer for every dataset — no per-layout branching. A legacy
    // slippy-map pyramid reaches it as resolution arrays chunked at tile size.
    this.regionRenderer = new RegionRenderer(
      this.zarrStore,
      this.variable,
      this.normalizedSelector,
      this.invalidate,
      this.fixedDataScale,
      this.minRegionSize,
      this.maxRegionFetches
    )

    // Lock immediately after the renderer captures the value, before async initialize()
    this.dataScaleLocked = true

    // The shader variant was resolved during initialize(), before this
    // renderer existed, so hand it over before the first fetch.
    this.regionRenderer.setRendersFromBandTextures(!!this.customShaderConfig)
    this.regionRenderer.setLoadingCallback(this.handleChunkLoadingChange)
    await this.regionRenderer.initialize()

    // The map may have zoomed out of range while this was loading; the
    // per-frame update picks up again once it zooms back in. Until then the
    // level still follows the zoom, for queries and `ready`.
    if (this.map && this.gl) {
      if (this.isZoomInRange()) this.regionRenderer.update(this.map, this.gl)
      else this.regionRenderer.followZoom(this.map, this.gl)
    }
  }

  private async initialize(): Promise<void> {
    try {
      this.zarrStore = new ZarrStore({
        source: this.url,
        version: this.zarrVersion,
        variable: this.variable,
        spatialDimensions: this.spatialDimensions,
        bounds: this.bounds,
        crs: this.crs,
        latIsAscending: this.latIsAscending,
        coordinateKeys: Object.keys(this.selector),
        proj4: this.proj4,
        transformRequest: this.transformRequest,
        onAuthError: this.onAuthError,
        customStore: this.customStore,
      })

      await this.zarrStore.initialized

      const desc = this.zarrStore.describe()

      this.levelAssets = desc.levelAssets
      this.dimIndices = desc.dimIndices
      this.scaleFactor = desc.scaleFactor
      this.offset = desc.addOffset

      if (
        this._fillValue === null &&
        desc.fill_value !== null &&
        desc.fill_value !== undefined
      ) {
        this._fillValue = desc.fill_value
      }

      this.normalizedSelector = normalizeSelector(this.selector)
      await this.loadInitialDimensionValues()

      this.refreshCustomShaderConfig()
    } catch (err) {
      // Clean up partially-initialized store before re-throwing
      if (this.zarrStore) {
        this.zarrStore.cleanup()
        this.zarrStore = null
      }
      throw err
    }
  }

  private async loadInitialDimensionValues(): Promise<void> {
    if (!this.zarrStore?.root) return

    const multiscaleLevel =
      this.levelAssets.length > 0 ? this.levelAssets[0] : null

    for (const [dimName, value] of Object.entries(this.selector)) {
      this.normalizedSelector[dimName] = toSelectorProps(value)
    }
    for (const dimName of Object.keys(this.dimIndices)) {
      // Skip spatial dimensions - don't load coordinate arrays for these
      if (!SPATIAL_DIM_NAMES.has(dimName.toLowerCase())) {
        try {
          this.dimensionValues[dimName] = await loadDimensionValues(
            this.dimensionValues,
            multiscaleLevel,
            this.dimIndices[dimName],
            this.zarrStore.root,
            this.zarrStore.version
          )

          if (!this.normalizedSelector[dimName]) {
            this.normalizedSelector[dimName] = { selected: 0 }
          }
        } catch (err) {
          console.warn(`Failed to load dimension values for ${dimName}:`, err)
        }
      }
    }
  }

  private isZoomInRange(): boolean {
    if (!this.map?.getZoom) return true
    // In MapLibre globe mode, pole enlargement compensation can push
    // getZoom() negative. Clamp to 0 so that artifact alone doesn't
    // hide the layer.
    const zoom = Math.max(0, this.map.getZoom())
    return zoom >= this.minZoom && zoom <= this.maxZoom
  }

  prerender(
    _gl: WebGL2RenderingContext | WebGLRenderingContext,
    _params: unknown,
    _projection?: { name: string },
    _projectionToMercatorMatrix?: number[] | Float32Array | Float64Array,
    _projectionToMercatorTransition?: number,
    _centerInMercator?: number[],
    _pixelsPerMeterRatio?: number,
    renderParameters?: unknown
  ) {
    if (this.isRemoved || !this.gl || !this.regionRenderer || !this.map) return
    // Mapbox runs prerender for draped layers too, so this is where the
    // render parameters first show up when the layer starts out draped.
    this.noteMapboxRenderParameters(renderParameters)

    // Draped terrain tiles are cached by the map and reused across zoom
    // changes while it moves, so crossing the zoom range has to force a
    // redraw or the tiles keep (or lack) the layer until movement stops.
    const inRange = this.isZoomInRange()
    if (this.lastZoomInRange !== null && this.lastZoomInRange !== inRange) {
      this.invalidate()
    }
    this.lastZoomInRange = inRange
    if (!inRange) return

    this.syncProjectionState()
    const cover = this.coveringTiles()
    // Tiles the draped hooks asked for are only being drawn while the map
    // drapes the layer: MapLibre with terrain, Mapbox with terrain or on its
    // draped globe. Off that path the map draws the layer directly and the
    // tiles would otherwise go on steering selection.
    const draped =
      !!this.map.getTerrain?.() ||
      (!cover && this.isGlobeProjection() && !this.usingDirectMapboxGlobePath)
    if (!draped) this.regionRenderer.clearDrawnTiles()
    this.regionRenderer.setCoveringTiles(cover)
    this.regionRenderer.update(this.map, this.gl)
  }

  /**
   * The tiles the map draws this frame, each with the map zoom its pixel
   * density matches, when the map exposes its tile cover (MapLibre does).
   * Tilted, the cover reaches lower zooms towards the horizon, so the layer
   * reads coarser levels there than at the center.
   */
  private coveringTiles(): TileRequest[] | null {
    const map = this.map
    if (!map?.coveringTiles) return null
    // Drawn directly on the globe, the layer reaches the poles, so the
    // cover's polar edge tiles must too; draped tiles stay Mercator.
    const poles = this.isGlobeProjection() && !map.getTerrain?.()
    return map
      .coveringTiles({ tileSize: COVERING_TILE_SIZE })
      .map(({ canonical }) => ({
        tileId: { z: canonical.z, x: canonical.x, y: canonical.y },
        zoom: this.tileZoom(canonical.z, COVERING_TILE_SIZE),
        poles,
      }))
  }

  /**
   * The map zoom whose pixel density a tile of this size matches. The map
   * picks tile zoom floor(zoom + log2(512 / tileSize)) at the center, so the
   * fractional part of the map zoom is carried over: a center tile maps back
   * onto the map zoom itself, and a tile k zooms coarser onto zoom - k.
   */
  private tileZoom(z: number, tileSize: number): number {
    const zoom = this.map?.getZoom?.() ?? z
    const shift = Math.log2(512 / tileSize)
    const target = zoom + shift
    return z - shift + (target - Math.floor(target))
  }

  /**
   * Screen size of the tiles the map covers its terrain with: MapLibre
   * covers at twice the terrain source's tile size.
   */
  private terrainTileSize(): number {
    const terrain = this.map?.getTerrain?.() as
      | { source?: string }
      | null
      | undefined
    const source = terrain?.source
      ? (this.map?.getSource?.(terrain.source) as
          | { tileSize?: number }
          | undefined)
      : undefined
    return source?.tileSize ? source.tileSize * 2 : COVERING_TILE_SIZE
  }

  render(
    _gl: WebGL2RenderingContext | WebGLRenderingContext,
    params: unknown,
    projection?: { name: string },
    projectionToMercatorMatrix?: number[] | Float32Array | Float64Array,
    projectionToMercatorTransition?: number,
    _centerInMercator?: number[],
    _pixelsPerMeterRatio?: number,
    renderParameters?: unknown
  ) {
    if (
      this.isRemoved ||
      !this.renderer ||
      !this.gl ||
      !this.regionRenderer ||
      !this.map
    ) {
      return
    }

    this.noteMapboxRenderParameters(renderParameters)
    if (!this.isZoomInRange()) {
      return
    }

    this.configureMapboxRenderPath()

    const projectionParams = resolveProjectionParams(
      params,
      projection,
      projectionToMercatorMatrix,
      projectionToMercatorTransition,
      renderParameters
    )

    if (!projectionParams.matrix) {
      return
    }

    // Legacy MapLibre (no shaderData): fall back to mapbox-style shader path
    // using identity globe-to-merc matrix + transition=1 (pure mercator).
    const legacyMapboxFallback =
      !projectionParams.mapbox && !projectionParams.shaderData
        ? {
            projection: { name: 'mercator' },
            globeToMercatorMatrix: MAPBOX_IDENTITY_MATRIX,
            transition: 1,
          }
        : undefined

    const isGlobe = this.isGlobeProjection()
    const worldOffsets = computeWorldOffsets(this.map, isGlobe)
    const colormapTexture = this.colormap.ensureTexture(this.gl)
    // The direct globe path draws straight onto the sphere, so it needs the
    // per-frame globe-world-to-clip matrix; without one this frame it falls
    // back to the regular globe program rather than drawing with a mercator
    // matrix.
    const ecefClipMatrix =
      this.usingDirectMapboxGlobePath &&
      projectionParams.mapbox?.renderParameters
        ? mapboxEcefClipMatrix(projectionParams.mapbox.renderParameters)
        : undefined

    const context: RenderContext = {
      gl: this.gl,
      matrix: projectionParams.matrix,
      uniforms: {
        clim: this.clim,
        opacity: this.opacity,
        fillValue: this._fillValue,
        scaleFactor: this.scaleFactor,
        offset: this.offset,
        fixedDataScale: this.fixedDataScale,
      },
      colormapTexture,
      worldOffsets,
      customShaderConfig: this.customShaderConfig || undefined,
      shaderData: projectionParams.shaderData,
      projectionData: projectionParams.projectionData,
      mapbox: projectionParams.mapbox
        ? {
            projection: projectionParams.mapbox.projection,
            globeToMercatorMatrix:
              projectionParams.mapbox.globeToMercatorMatrix,
            transition: projectionParams.mapbox.transition,
            directGlobePathActive: !!ecefClipMatrix,
            ecefClipMatrix,
          }
        : legacyMapboxFallback,
    }

    this.regionRenderer.render(this.renderer, context)

    this.tileNeedsRender = false
  }

  renderToTile(
    _gl: WebGL2RenderingContext | WebGLRenderingContext,
    tileId: { z: number; x: number; y: number }
  ) {
    if (
      this.isRemoved ||
      !this.renderer ||
      !this.gl ||
      !this.regionRenderer ||
      !this.map
    ) {
      return
    }

    this.configureMapboxRenderPath()

    const isGlobe = this.syncProjectionState()
    const tile = { tileId, zoom: this.tileZoom(tileId.z, COVERING_TILE_SIZE) }
    this.regionRenderer.noteDrawnTile(tile, this.terrainTileRevision)
    this.regionRenderer.update(this.map, this.gl)

    this.tileNeedsRender = this.regionRenderer.renderToTile(
      this.renderer,
      tile,
      this.tileRenderContext(this.gl, isGlobe),
      { northUp: false }
    )
  }

  /**
   * MapLibre's experimental terrain hook (maplibre-gl >= 6.13), called in
   * place of `render` while terrain is on. MapLibre drapes the tile over the
   * terrain (and the globe) itself, so the tile is always drawn as a flat
   * Mercator tile. MapLibre still calls `prerender` each frame, which drives
   * region fetching; a tile drawn before its data arrived is redrawn once the
   * load bumps `terrainTileRevision`.
   */
  renderToTerrainTile(
    _gl: WebGL2RenderingContext | WebGLRenderingContext,
    { tileID }: { tileID: TerrainTileID }
  ) {
    if (
      this.isRemoved ||
      !this.renderer ||
      !this.gl ||
      !this.regionRenderer ||
      !this.map
    ) {
      return
    }

    if (!this.isZoomInRange()) {
      return
    }

    const tile = {
      tileId: tileID.canonical,
      zoom: this.tileZoom(tileID.canonical.z, this.terrainTileSize()),
    }
    const needsFetch = this.regionRenderer.noteDrawnTile(
      tile,
      this.terrainTileRevision
    )
    // MapLibre ran `prerender` before drawing this tile and schedules no
    // frame after a tile redraw, so ask for one to fetch what the tile
    // lacks. The revision stays put: that frame only needs the update.
    if (needsFetch) this.map.triggerRepaint?.()
    this.regionRenderer.renderToTile(
      this.renderer,
      tile,
      this.tileRenderContext(this.gl, false),
      { northUp: true }
    )
  }

  private tileRenderContext(
    gl: WebGL2RenderingContext,
    isGlobe: boolean
  ): RenderContext {
    return {
      gl,
      matrix: new Float32Array(16),
      uniforms: {
        clim: this.clim,
        opacity: this.opacity,
        fillValue: this._fillValue,
        scaleFactor: this.scaleFactor,
        offset: this.offset,
        fixedDataScale: this.fixedDataScale,
      },
      colormapTexture: this.colormap.ensureTexture(gl),
      worldOffsets: [0],
      customShaderConfig: this.customShaderConfig || undefined,
      isGlobe,
    }
  }

  // Mapbox specific custom layer method required to trigger rerender on eg dataset update.
  shouldRerenderTiles() {
    const needsRender = this.tileNeedsRender
    this.tileNeedsRender = false
    return needsRender
  }

  /**
   * Dispose all GL resources and internal state.
   * Does NOT remove the layer from the map - call map.removeLayer(id) for that.
   */
  private _disposeResources(
    gl: WebGL2RenderingContext | WebGLRenderingContext
  ): void {
    if (this.isRemoved) return
    this.isRemoved = true

    this.renderer?.dispose()
    this.renderer = null

    this.colormap.dispose(gl)

    this.regionRenderer?.dispose(gl)
    this.regionRenderer = null

    if (this.zarrStore) {
      this.zarrStore.cleanup()
      this.zarrStore = null
    }

    if (this.map && typeof this.map.off === 'function') {
      if (this.projectionChangeHandler) {
        this.map.off('projectionchange', this.projectionChangeHandler)
        this.map.off('style.load', this.projectionChangeHandler)
        this.map.off('move', this.projectionChangeHandler)
      }
      if (this.mapRemoveHandler) {
        this.map.off('remove', this.mapRemoveHandler)
      }
    }
    this.projectionChangeHandler = null
    this.mapRemoveHandler = null
  }

  onRemove(_map: MapLike, gl: WebGL2RenderingContext | WebGLRenderingContext) {
    const resolvedGl = this.gl ?? this.resolveGl(_map, gl)
    this._disposeResources(resolvedGl)
  }

  // ========== Query Interface ==========

  /**
   * Resolves once the layer can serve queries and draw: metadata loaded and a
   * level committed. Rejects with `ZarrLayerNotReadyError` if initialization
   * or level loading failed, or if the layer was removed.
   *
   * This is not the same signal as `onLoadingStateChange`. That one is a
   * spinner — it flaps as chunks come and go, and it reports nothing about the
   * level commit, so `loading: false` can be emitted while the layer still has
   * no level. `ready` settles once, after both stages.
   *
   * Awaiting it is optional for queries; `queryData` waits on its own.
   */
  get ready(): Promise<void> {
    this.readyPromise ??= this.resolveReadiness().then(
      () => {},
      (err) => {
        // Only success is cached. Caching the rejection would freeze a
        // transient failure — a level load that lost a race, a store blip —
        // into a layer that reports itself permanently unready even after
        // the render loop has gone on to commit a level.
        this.readyPromise = null
        throw err
      }
    )
    return this.readyPromise
  }

  /** Wait out initialization and commit a level, or explain why neither can
   *  happen. Shared by `ready` and `queryData`. */
  private async resolveReadiness(): Promise<RegionRenderer> {
    if (!this.initPromise) {
      throw new ZarrLayerNotReadyError(
        this.id,
        'layer has not been added to a map — call map.addLayer() first'
      )
    }
    await this.initPromise
    if (this.initError) {
      throw new ZarrLayerNotReadyError(
        this.id,
        `layer failed to initialize: ${this.initError.message}`,
        { cause: this.initError }
      )
    }
    const regionRenderer = this.regionRenderer
    if (this.isRemoved || !regionRenderer) {
      throw new ZarrLayerNotReadyError(
        this.id,
        'layer has been removed from the map'
      )
    }
    // Held in a local because a removal landing during the await nulls the
    // field out from under us.
    const level = await regionRenderer.ensureQueryableLevel()
    if (this.isRemoved || this.regionRenderer !== regionRenderer) {
      throw new ZarrLayerNotReadyError(
        this.id,
        'layer was removed while it was becoming ready'
      )
    }
    if (!level) {
      const cause = regionRenderer.levelLoadError
      // No level means the load failed or was superseded and never retried.
      // Returning here would let a query answer empty off a layer that holds
      // no data at all, which is exactly the silent wrong answer the
      // readiness contract exists to prevent.
      throw new ZarrLayerNotReadyError(
        this.id,
        `no resolution level could be loaded${
          cause ? `: ${cause.message}` : ''
        }`,
        cause ? { cause } : undefined
      )
    }
    return regionRenderer
  }

  /**
   * Query all data values within a geographic region.
   *
   * Waits for metadata and a committed resolution level, so it can be called
   * straight after `map.addLayer()` with no render pass in between. Neither an
   * unready layer nor a failed read answers with an empty result, so an empty
   * result means the geometry found no data.
   *
   * @param geometry - GeoJSON Point, LineString, Polygon or MultiPolygon
   *   geometry. A LineString returns one sample per cell along its path, in
   *   order, with a `distance` coordinate in meters from the line's start.
   * @param selector - Optional selector to override the layer's selector.
   * @throws {ZarrLayerNotReadyError} if the layer failed to initialize, could
   *   not load a level, was removed, or was never added to a map.
   * @returns Promise resolving to the query result matching carbonplan/maps structure.
   */
  async queryData(
    geometry: QueryGeometry,
    selector?: Selector,
    options?: QueryOptions
  ): Promise<QueryResult> {
    const regionRenderer = await abortable(
      this.resolveReadiness(),
      options?.signal
    )
    const result = await regionRenderer.queryData(geometry, selector, options)
    // Readiness held when the query started, but a removal landing while it ran
    // disposes the renderer and clears its level, and the inner path answers
    // empty once there is no level to index. Checked here so a layer that went
    // away can never come back as "no data here".
    if (this.isRemoved || this.regionRenderer !== regionRenderer) {
      throw new ZarrLayerNotReadyError(
        this.id,
        'layer was removed while the query was in flight'
      )
    }
    return result
  }
}
