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
} from './zarr-utils'
import { ZarrStore } from './zarr-store'
import { maplibreFragmentShaderSource, type ShaderData } from './shaders'
import { ColormapState } from './colormap'
import { ZarrRenderer } from './zarr-renderer'
import type { CustomShaderConfig } from './renderer-types'
import type {
  Bounds,
  ColormapArray,
  DatasetDescriptor,
  SpatialDimensions,
  DimIndicesProps,
  LoadingState,
  LoadingStateCallback,
  MapLike,
  Selector,
  NormalizedSelector,
  ZarrLayerOptions,
  TransformRequest,
  ResolveProj4,
} from './types'
import type { ZarrMode, RenderContext } from './zarr-mode'
import { TiledMode } from './tiled-mode'
import { UntiledMode } from './untiled-mode'
import {
  computeWorldOffsets,
  resolveProjectionParams,
  isGlobeProjection as checkGlobeProjection,
} from './map-utils'
import { MAPBOX_IDENTITY_MATRIX } from './mapbox-utils'
import type { QueryDataValues, QueryGeometry, QueryResult } from './query/types'
import { SPATIAL_DIM_NAMES } from './constants'

export class ZarrLayer {
  readonly type: 'custom' = 'custom'
  readonly renderingMode: '2d' | '3d'

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
  // Once true, fixedDataScale is locked (mode has captured it)
  private dataScaleLocked: boolean = false

  private gl: WebGL2RenderingContext | undefined
  private map: MapLike | null = null
  private renderer: ZarrRenderer | null = null
  private mode: ZarrMode | null = null
  private tileNeedsRender: boolean = true

  private projectionChangeHandler: (() => void) | null = null
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
  // Datatree support: multiple stores/modes for child datasets
  private datatreeStores: ZarrStore[] = []
  private datatreeModes: ZarrMode[] = []
  private datatreeDatasets: DatasetDescriptor[] = []
  private datatreeModeLoading: Map<number, boolean> = new Map()
  private isDatatree: boolean = false
  private levelInfos: string[] = []
  private dimIndices: DimIndicesProps = {}
  private dimensionValues: {
    [key: string]: Float64Array | number[] | string[]
  } = {}
  private normalizedSelector: NormalizedSelector = {}
  private isRemoved: boolean = false
  private fragmentShaderSource: string = maplibreFragmentShaderSource
  private customFrag: string | undefined
  private customUniforms: Record<string, number> = {}
  private bandNames: string[] = []
  private customShaderConfig: CustomShaderConfig | null = null
  private onLoadingStateChange: LoadingStateCallback | undefined
  private metadataLoading: boolean = false
  private chunksLoading: boolean = false
  private initError: Error | null = null
  private throttleMs: number
  private proj4: string | undefined
  private resolveProj4: boolean | ResolveProj4 | undefined
  private transformRequest: TransformRequest | undefined
  private customStore: Readable<unknown> | undefined
  private lastIsGlobe: boolean | null = null

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
      if (this.isDatatree) {
        this.datatreeModes.forEach((m) => m.onProjectionChange(isGlobe))
      } else {
        this.mode?.onProjectionChange(isGlobe)
      }
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
    renderingMode = '3d',
    onLoadingStateChange,
    throttleMs = 100,
    proj4,
    resolveProj4,
    transformRequest,
    store,
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
    this.selectorHash = this.computeSelectorHash(this.normalizedSelector)
    this.renderingMode = renderingMode
    this.invalidate = () => {}
    this.colormap = new ColormapState(colormap)
    this.clim = clim
    this.fixedDataScale = Math.max(Math.abs(clim[0]), Math.abs(clim[1]), 1)
    this.opacity = opacity
    this.minZoom = minzoom
    this.maxZoom = maxzoom

    this.customFrag = customFrag
    this.customUniforms = uniforms || {}

    this.bandNames = getBands(variable, this.normalizedSelector)
    if (this.bandNames.length > 1 || customFrag) {
      this.customShaderConfig = {
        bands: this.bandNames,
        customFrag: customFrag,
        customUniforms: this.customUniforms,
      }
    }

    if (fillValue !== undefined) this._fillValue = fillValue
    this.onLoadingStateChange = onLoadingStateChange
    this.throttleMs = throttleMs
    this.proj4 = proj4
    this.resolveProj4 = resolveProj4
    this.transformRequest = transformRequest
    this.customStore = store
  }

  private emitLoadingState(): void {
    if (!this.onLoadingStateChange) return
    this.onLoadingStateChange({
      loading: this.metadataLoading || this.chunksLoading,
      metadata: this.metadataLoading,
      chunks: this.chunksLoading,
      error: this.initError,
    })
  }

  private handleChunkLoadingChange = (state: LoadingState): void => {
    this.chunksLoading = state.chunks
    this.emitLoadingState()
  }

  private handleDatatreeModeLoadingChange = (
    modeIndex: number,
    state: LoadingState
  ): void => {
    this.datatreeModeLoading.set(modeIndex, state.chunks)
    this.chunksLoading = Array.from(this.datatreeModeLoading.values()).some(
      (loading) => loading
    )
    this.emitLoadingState()
  }

  setOpacity(opacity: number) {
    this.opacity = opacity
    this.invalidate()
  }

  setClim(clim: [number, number]) {
    this.clim = clim
    // Allow fixedDataScale to update until mode captures it
    if (!this.dataScaleLocked) {
      this.fixedDataScale = Math.max(Math.abs(clim[0]), Math.abs(clim[1]), 1)
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

  setUniforms(uniforms: Record<string, number>) {
    if (!this.customShaderConfig) {
      console.warn(
        '[ZarrLayer] setUniforms() called but layer was not created with customFrag. ' +
          'Uniforms will not be applied. Recreate the layer with customFrag and uniforms options.'
      )
      return
    }
    this.customUniforms = { ...this.customUniforms, ...uniforms }
    this.customShaderConfig.customUniforms = this.customUniforms
    this.invalidate()
  }

  async setVariable(variable: string) {
    if (variable === this.variable) return

    this.metadataLoading = true
    this.emitLoadingState()

    try {
      this.initError = null
      this.variable = variable
      this._disposeModes()
      this._cleanupStores()
      this.dimensionValues = {}
      this._fillValue = null
      // Reset and recompute fixedDataScale from current clim for new mode
      this.dataScaleLocked = false
      this.fixedDataScale = Math.max(
        Math.abs(this.clim[0]),
        Math.abs(this.clim[1]),
        1
      )
      await this.initialize()
      await this.initializeMode()
      this.invalidate()
    } catch (err) {
      this.initError = err instanceof Error ? err : new Error(String(err))
      console.error('[zarr-layer] Failed to reset:', this.initError.message)
      this._disposeModes()
      this._cleanupStores()
    } finally {
      this.metadataLoading = false
      this.emitLoadingState()
    }
  }

  async setSelector(selector: Selector) {
    const normalized = normalizeSelector(selector)
    const nextHash = this.computeSelectorHash(normalized)
    if (nextHash === this.selectorHash) {
      return
    }
    this.selectorHash = nextHash
    this.selector = selector
    this.normalizedSelector = normalized

    this.bandNames = getBands(this.variable, this.normalizedSelector)
    if (this.bandNames.length > 1 || this.customFrag) {
      this.customShaderConfig = {
        bands: this.bandNames,
        customFrag: this.customFrag,
        customUniforms: this.customUniforms,
      }
    } else {
      this.customShaderConfig = null
    }

    if (this.isDatatree) {
      await Promise.all(
        this.datatreeModes.map((m) => m.setSelector(this.normalizedSelector))
      )
    } else if (this.mode) {
      await this.mode.setSelector(this.normalizedSelector)
    }
    this.invalidate()
  }

  onAdd(
    map: MapLike,
    gl: WebGL2RenderingContext | WebGLRenderingContext
  ): void {
    this._onAddAsync(map, gl)
  }

  private async _onAddAsync(
    map: MapLike,
    gl: WebGL2RenderingContext | WebGLRenderingContext
  ): Promise<void> {
    this.map = map
    const resolvedGl = this.resolveGl(map, gl)
    this.gl = resolvedGl
    this.invalidate = () => {
      this.tileNeedsRender = true
      if (map.triggerRepaint) map.triggerRepaint()
    }

    this.initError = null
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
        if (this.isDatatree) {
          this.datatreeModes.forEach((m) => m.onProjectionChange(isGlobe))
        } else {
          this.mode?.onProjectionChange(isGlobe)
        }
      }
      if (typeof map.on === 'function' && this.projectionChangeHandler) {
        map.on('projectionchange', this.projectionChangeHandler)
        map.on('style.load', this.projectionChangeHandler)
      }

      await this.initialize()
      await this.initializeMode()

      const isGlobe = this.isGlobeProjection()
      this.lastIsGlobe = isGlobe
      if (this.isDatatree) {
        this.datatreeModes.forEach((m) => m.onProjectionChange(isGlobe))
        this.getRenderableModes().forEach((m) => m.update(this.map!, this.gl!))
      } else {
        this.mode?.onProjectionChange(isGlobe)
        this.mode?.update(this.map, this.gl!)
      }
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

  private computeSelectorHash(selector: NormalizedSelector): string {
    const sortKeys = (value: unknown): unknown => {
      if (Array.isArray(value) || value === null) return value
      if (typeof value !== 'object') return value

      const obj = value as Record<string, unknown>
      const sorted: Record<string, unknown> = {}
      Object.keys(obj)
        .sort()
        .forEach((k) => {
          sorted[k] = sortKeys(obj[k])
        })
      return sorted
    }

    return JSON.stringify(sortKeys(selector))
  }

  private _disposeModes(): void {
    if (this.gl) {
      this.mode?.dispose(this.gl)
      this.datatreeModes.forEach((m) => m.dispose(this.gl!))
    }
    this.mode = null
    this.datatreeModes = []
    this.datatreeModeLoading.clear()
    this.chunksLoading = false
  }

  private _cleanupStores(): void {
    const stores = new Set<ZarrStore>()
    if (this.zarrStore) stores.add(this.zarrStore)
    this.datatreeStores.forEach((store) => stores.add(store))
    stores.forEach((store) => store.cleanup())

    this.zarrStore = null
    this.datatreeStores = []
    this.datatreeDatasets = []
    this.isDatatree = false
  }

  private xyLimitsToBounds(
    xyLimits: ReturnType<ZarrMode['getXYLimits']>
  ): Bounds | null {
    if (!xyLimits) return null
    return [xyLimits.xMin, xyLimits.yMin, xyLimits.xMax, xyLimits.yMax]
  }

  private getMapViewBounds(map: MapLike): Bounds | null {
    if (!map.getBounds) return null
    const bounds = map.getBounds()
    if (!bounds) return null

    const corners = bounds.toArray()
    if (!Array.isArray(corners) || corners.length < 2) return null
    const south = Math.min(corners[0][1], corners[1][1])
    const north = Math.max(corners[0][1], corners[1][1])
    return [bounds.getWest(), south, bounds.getEast(), north]
  }

  private getWrappedLonIntervals(minLon: number, maxLon: number): number[][] {
    if (!Number.isFinite(minLon) || !Number.isFinite(maxLon)) return []

    if (maxLon - minLon >= 360) {
      return [[-180, 180]]
    }

    const base =
      minLon <= maxLon
        ? [[minLon, maxLon]]
        : [
            [minLon, 180],
            [-180, maxLon],
          ]

    return base.flatMap(([a, b]) => [
      [a - 360, b - 360],
      [a, b],
      [a + 360, b + 360],
    ])
  }

  private boundsIntersect(a: Bounds, b: Bounds): boolean {
    const yIntersects = !(a[3] < b[1] || a[1] > b[3])
    if (!yIntersects) return false

    const aLonIntervals = this.getWrappedLonIntervals(a[0], a[2])
    const bLonIntervals = this.getWrappedLonIntervals(b[0], b[2])

    return aLonIntervals.some(([aMin, aMax]) =>
      bLonIntervals.some(([bMin, bMax]) => !(aMax < bMin || aMin > bMax))
    )
  }

  private isDatasetVisible(
    bounds: Bounds | null | undefined,
    map: MapLike,
    crs?: string
  ): boolean {
    if (!bounds) return true
    if (crs && crs !== 'EPSG:4326') return true

    const viewBounds = this.getMapViewBounds(map)
    if (!viewBounds) return true

    return this.boundsIntersect(bounds, viewBounds)
  }

  private getDatasetBounds(index: number): Bounds | null {
    const discoveredBounds = this.datatreeDatasets[index]?.bounds
    if (discoveredBounds) return discoveredBounds

    const modeBounds = this.xyLimitsToBounds(
      this.datatreeModes[index]?.getXYLimits()
    )
    if (modeBounds) return modeBounds

    const storeBounds = this.xyLimitsToBounds(
      this.datatreeStores[index]?.describe().xyLimits ?? null
    )
    if (storeBounds) return storeBounds

    return null
  }

  private getRenderableModes(): ZarrMode[] {
    if (!this.isDatatree) {
      return this.mode ? [this.mode] : []
    }
    if (!this.map) return this.datatreeModes

    return this.datatreeModes.filter((mode, index) =>
      this.isDatasetVisible(
        this.getDatasetBounds(index),
        this.map!,
        mode.getCRS()
      )
    )
  }

  private async initializeMode() {
    if (!this.gl) return

    this._disposeModes()

    if (this.isDatatree) {
      for (const store of this.datatreeStores) {
        const desc = store.describe()
        let mode: ZarrMode
        if (desc.multiscaleType === 'tiled') {
          mode = new TiledMode(
            store,
            this.variable,
            this.normalizedSelector,
            this.invalidate,
            this.throttleMs,
            this.fixedDataScale
          )
        } else {
          mode = new UntiledMode(
            store,
            this.variable,
            this.normalizedSelector,
            this.invalidate,
            this.throttleMs,
            this.fixedDataScale
          )
        }
        const modeIndex = this.datatreeModes.length
        mode.setLoadingCallback((state) =>
          this.handleDatatreeModeLoadingChange(modeIndex, state)
        )
        this.datatreeModes.push(mode)
      }

      this.dataScaleLocked = true

      await Promise.all(this.datatreeModes.map((m) => m.initialize()))

      if (this.map && this.gl) {
        this.getRenderableModes().forEach((m) => m.update(this.map!, this.gl!))
      }
      return
    }

    if (!this.zarrStore) return

    const desc = this.zarrStore.describe()

    // Mode selection based on auto-detected metadata format:
    // - 'tiled' = OME-NGFF style with slippy map tile convention
    // - 'untiled' = zarr-conventions/multiscales format or single-level
    // - 'none' = single-level dataset (also uses UntiledMode)
    if (desc.multiscaleType === 'tiled') {
      this.mode = new TiledMode(
        this.zarrStore,
        this.variable,
        this.normalizedSelector,
        this.invalidate,
        this.throttleMs,
        this.fixedDataScale
      )
    } else {
      // Use UntiledMode for untiled multiscales and single-level datasets
      this.mode = new UntiledMode(
        this.zarrStore,
        this.variable,
        this.normalizedSelector,
        this.invalidate,
        this.throttleMs,
        this.fixedDataScale
      )
    }

    // Lock immediately after mode captures the value, before async initialize()
    this.dataScaleLocked = true

    this.mode.setLoadingCallback(this.handleChunkLoadingChange)
    await this.mode.initialize()

    if (this.map && this.gl) {
      this.mode.update(this.map, this.gl)
    }
  }

  private async initialize(): Promise<void> {
    try {
      this.isDatatree = false
      this.datatreeStores = []
      this.datatreeDatasets = []

      // First, create a store for the root to load metadata
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
        resolveProj4: this.resolveProj4,
        transformRequest: this.transformRequest,
        customStore: this.customStore,
      })

      await this.zarrStore.initialized
      const rootStore = this.zarrStore

      // Check for datatree: child groups containing the variable
      const datasets = ZarrStore.discoverDatasets(
        rootStore.metadata,
        this.variable,
        rootStore.version
      )

      if (datasets.length > 0) {
        // Datatree mode: create a store per child dataset
        this.isDatatree = true
        this.datatreeDatasets = datasets

        if (this.crs || this.proj4 || this.bounds) {
          console.warn(
            `[zarr-layer] Datatree mode detected; ignoring layer-level crs/proj4/bounds overrides ` +
              `so each child dataset can resolve its own spatial metadata.`
          )
        }

        this.datatreeStores = datasets.map((dataset) => {
          return new ZarrStore({
            // Keep all child datasets on the root source so consolidated metadata
            // and store handle can be shared across the datatree.
            source: this.url,
            version: rootStore.version,
            variable: `${dataset.path}/${this.variable}`,
            spatialDimensions: this.spatialDimensions,
            bounds: dataset.bounds ?? undefined,
            crs: undefined,
            latIsAscending: this.latIsAscending,
            coordinateKeys: Object.keys(this.selector),
            proj4: undefined,
            resolveProj4: this.resolveProj4,
            transformRequest: this.transformRequest,
            customStore: this.customStore,
          })
        })
        await Promise.all(this.datatreeStores.map((store) => store.initialized))

        if (this.datatreeStores.length === 0) {
          throw new Error('Datatree discovered no readable child datasets')
        }

        // Use first child store for shared dimension metadata in layer-level state.
        this.zarrStore = this.datatreeStores[0]
        rootStore.cleanup()

        // Use first store for shared metadata (dimensions, fill_value, etc.)
        const firstDesc = this.datatreeStores[0].describe()
        this.levelInfos = firstDesc.levels
        this.dimIndices = firstDesc.dimIndices
        this.scaleFactor = firstDesc.scaleFactor
        this.offset = firstDesc.addOffset

        if (
          this._fillValue === null &&
          firstDesc.fill_value !== null &&
          firstDesc.fill_value !== undefined
        ) {
          this._fillValue = firstDesc.fill_value
        }
      } else {
        // Single-dataset mode (existing path)
        this.isDatatree = false
        this.datatreeDatasets = []
        this.datatreeStores = []
        if (rootStore.hasMissingVariableMetadata()) {
          throw new Error(
            `Variable "${this.variable}" was not found at root and no datatree child datasets were discovered.`
          )
        }
        const desc = rootStore.describe()

        this.levelInfos = desc.levels
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
      }

      this.normalizedSelector = normalizeSelector(this.selector)
      await this.loadInitialDimensionValues()

      this.bandNames = getBands(this.variable, this.normalizedSelector)
      if (this.bandNames.length > 1 || this.customFrag) {
        this.customShaderConfig = {
          bands: this.bandNames,
          customFrag: this.customFrag,
          customUniforms: this.customUniforms,
        }
      } else {
        this.customShaderConfig = null
      }
    } catch (err) {
      // Clean up partially-initialized stores before re-throwing
      this._cleanupStores()
      throw err
    }
  }

  private async loadInitialDimensionValues(): Promise<void> {
    if (!this.zarrStore?.root) return

    const variablePath = this.zarrStore.getVariablePath()
    const variablePrefix = variablePath.includes('/')
      ? variablePath.slice(0, variablePath.lastIndexOf('/'))
      : ''
    const multiscaleLevel =
      this.levelInfos.length > 0 ? this.levelInfos[0] : null
    const dimensionLevel =
      variablePrefix && multiscaleLevel
        ? `${variablePrefix}/${multiscaleLevel}`
        : variablePrefix
        ? variablePrefix
        : multiscaleLevel

    for (const [dimName, value] of Object.entries(this.selector)) {
      this.normalizedSelector[dimName] = toSelectorProps(value)
    }
    for (const dimName of Object.keys(this.dimIndices)) {
      // Skip spatial dimensions - don't load coordinate arrays for these
      if (!SPATIAL_DIM_NAMES.has(dimName.toLowerCase())) {
        try {
          this.dimensionValues[dimName] = await loadDimensionValues(
            this.dimensionValues,
            dimensionLevel,
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
    const zoom = this.map.getZoom()
    return zoom >= this.minZoom && zoom <= this.maxZoom
  }

  prerender(
    _gl: WebGL2RenderingContext | WebGLRenderingContext,
    _params: unknown
  ) {
    if (this.isRemoved || !this.gl || !this.map) return
    if (!this.isZoomInRange()) return

    this.syncProjectionState()
    this.getRenderableModes().forEach((mode) =>
      mode.update(this.map!, this.gl!)
    )
  }

  render(
    _gl: WebGL2RenderingContext | WebGLRenderingContext,
    params: unknown,
    projection?: { name: string },
    projectionToMercatorMatrix?: number[] | Float32Array | Float64Array,
    projectionToMercatorTransition?: number,
    _centerInMercator?: number[],
    _pixelsPerMeterRatio?: number
  ) {
    if (this.isRemoved || !this.renderer || !this.gl || !this.map) {
      return
    }

    if (!this.isZoomInRange()) {
      return
    }

    const projectionParams = resolveProjectionParams(
      params,
      projection,
      projectionToMercatorMatrix,
      projectionToMercatorTransition
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
      mapbox: projectionParams.mapbox ?? legacyMapboxFallback,
    }

    const renderableModes = this.getRenderableModes()
    if (renderableModes.length === 0) return

    renderableModes.forEach((mode) => mode.render(this.renderer!, context))

    this.tileNeedsRender = false
  }

  renderToTile(
    _gl: WebGL2RenderingContext | WebGLRenderingContext,
    tileId: { z: number; x: number; y: number }
  ) {
    if (this.isRemoved || !this.renderer || !this.gl || !this.map) {
      return
    }

    const modes = this.isDatatree
      ? this.datatreeModes
      : this.mode
      ? [this.mode]
      : []
    if (modes.length === 0) return

    const isGlobe = this.syncProjectionState()
    modes.forEach((mode) => mode.update(this.map!, this.gl!))

    const colormapTexture = this.colormap.ensureTexture(this.gl)

    const context: RenderContext = {
      gl: this.gl,
      matrix: new Float32Array(16),
      uniforms: {
        clim: this.clim,
        opacity: this.opacity,
        fillValue: this._fillValue,
        scaleFactor: this.scaleFactor,
        offset: this.offset,
        fixedDataScale: this.fixedDataScale,
      },
      colormapTexture,
      worldOffsets: [0],
      customShaderConfig: this.customShaderConfig || undefined,
      isGlobe,
    }

    this.tileNeedsRender = modes.some(
      (mode) => mode.renderToTile?.(this.renderer!, tileId, context) ?? false
    )
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
    this.isRemoved = true

    this.renderer?.dispose()
    this.renderer = null

    this.colormap.dispose(gl)

    this._disposeModes()
    this._cleanupStores()

    if (
      this.map &&
      this.projectionChangeHandler &&
      typeof this.map.off === 'function'
    ) {
      this.map.off('projectionchange', this.projectionChangeHandler)
      this.map.off('style.load', this.projectionChangeHandler)
    }

    this.projectionChangeHandler = null
    this.gl = undefined
    this.map = null
  }

  onRemove(_map: MapLike, gl: WebGL2RenderingContext | WebGLRenderingContext) {
    const resolvedGl = this.gl ?? this.resolveGl(_map, gl)
    this._disposeResources(resolvedGl)
  }

  private emptyQueryResult(): QueryResult {
    return {
      [this.variable]: [],
      dimensions: ['lat', 'lon'],
      coordinates: { lat: [], lon: [] },
    }
  }

  private boundsContainsPoint(
    bounds: Bounds,
    lon: number,
    lat: number
  ): boolean {
    if (lat < bounds[1] || lat > bounds[3]) return false
    return this.getWrappedLonIntervals(bounds[0], bounds[2]).some(
      ([minLon, maxLon]) => lon >= minLon && lon <= maxLon
    )
  }

  private geometryToBounds(geometry: QueryGeometry): Bounds | null {
    if (geometry.type === 'Point') {
      const [lon, lat] = geometry.coordinates
      return [lon, lat, lon, lat]
    }

    const points =
      geometry.type === 'Polygon'
        ? geometry.coordinates.flat()
        : geometry.coordinates.flat(2)

    if (!points.length) return null
    const lons = points.map(([lon]) => lon)
    const lats = points.map(([, lat]) => lat)
    return [
      Math.min(...lons),
      Math.min(...lats),
      Math.max(...lons),
      Math.max(...lats),
    ]
  }

  private getQueryMode(geometry: QueryGeometry): ZarrMode | null {
    if (!this.isDatatree) return this.mode
    if (this.datatreeModes.length === 0) return null

    if (geometry.type === 'Point') {
      const [lon, lat] = geometry.coordinates
      for (let i = 0; i < this.datatreeModes.length; i++) {
        const mode = this.datatreeModes[i]
        if (mode.getCRS() !== 'EPSG:4326') continue
        const bounds = this.getDatasetBounds(i)
        if (bounds && this.boundsContainsPoint(bounds, lon, lat)) {
          return mode
        }
      }
    } else {
      const queryBounds = this.geometryToBounds(geometry)
      if (queryBounds) {
        for (let i = 0; i < this.datatreeModes.length; i++) {
          const mode = this.datatreeModes[i]
          if (mode.getCRS() !== 'EPSG:4326') continue
          const bounds = this.getDatasetBounds(i)
          if (bounds && this.boundsIntersect(bounds, queryBounds)) {
            return mode
          }
        }
      }
    }

    return this.datatreeModes[0] ?? null
  }

  private getQueryModesForGeometry(geometry: QueryGeometry): ZarrMode[] {
    if (!this.isDatatree) return this.mode ? [this.mode] : []
    if (this.datatreeModes.length === 0) return []

    if (geometry.type === 'Point') {
      const [lon, lat] = geometry.coordinates
      const candidates = this.datatreeModes.filter((mode, index) => {
        const bounds = this.getDatasetBounds(index)
        if (!bounds) return true
        if (mode.getCRS() !== 'EPSG:4326') return true
        return this.boundsContainsPoint(bounds, lon, lat)
      })
      return candidates.length > 0 ? candidates : [this.datatreeModes[0]]
    }

    const queryBounds = this.geometryToBounds(geometry)
    if (!queryBounds) return this.datatreeModes

    const candidates = this.datatreeModes.filter((mode, index) => {
      const bounds = this.getDatasetBounds(index)
      if (!bounds) return true
      if (mode.getCRS() !== 'EPSG:4326') return true
      return this.boundsIntersect(bounds, queryBounds)
    })
    return candidates.length > 0 ? candidates : this.datatreeModes
  }

  private isQueryValueEmpty(value: unknown): boolean {
    if (Array.isArray(value)) return value.length === 0
    if (typeof value !== 'object' || value === null) return true
    const obj = value as Record<string, unknown>
    const entries = Object.values(obj)
    if (entries.length === 0) return true
    return entries.every((entry) => this.isQueryValueEmpty(entry))
  }

  private hasQueryResultData(result: QueryResult): boolean {
    return !this.isQueryValueEmpty(result[this.variable])
  }

  private mergeQueryValues(base: unknown, incoming: unknown): unknown {
    if (Array.isArray(base) && Array.isArray(incoming)) {
      return [...base, ...incoming]
    }
    if (
      typeof base === 'object' &&
      base !== null &&
      !Array.isArray(base) &&
      typeof incoming === 'object' &&
      incoming !== null &&
      !Array.isArray(incoming)
    ) {
      const merged: Record<string, unknown> = {
        ...(base as Record<string, unknown>),
      }
      for (const [key, value] of Object.entries(
        incoming as Record<string, unknown>
      )) {
        if (merged[key] === undefined) {
          merged[key] = value
        } else {
          merged[key] = this.mergeQueryValues(merged[key], value)
        }
      }
      return merged
    }
    if (this.isQueryValueEmpty(base)) return incoming
    if (this.isQueryValueEmpty(incoming)) return base
    return incoming
  }

  private mergeQueryResults(results: QueryResult[]): QueryResult {
    const merged = this.emptyQueryResult()
    let mergedValue: QueryDataValues = merged[this.variable] as QueryDataValues

    for (const result of results) {
      if (!this.hasQueryResultData(result)) continue

      if (merged.dimensions.length <= 2 && result.dimensions.length > 0) {
        merged.dimensions = result.dimensions
      }

      merged.coordinates.lat.push(...(result.coordinates?.lat ?? []))
      merged.coordinates.lon.push(...(result.coordinates?.lon ?? []))

      for (const [key, values] of Object.entries(result.coordinates ?? {})) {
        if (key === 'lat' || key === 'lon') continue
        if (!merged.coordinates[key]) {
          merged.coordinates[key] = values
        }
      }

      const value = result[this.variable] as QueryDataValues
      mergedValue = this.mergeQueryValues(mergedValue, value) as QueryDataValues
    }

    merged[this.variable] = mergedValue
    return merged
  }

  // ========== Query Interface ==========

  /**
   * Query all data values within a geographic region.
   * @param geometry - GeoJSON Point, Polygon or MultiPolygon geometry.
   * @param selector - Optional selector to override the layer's selector.
   * @returns Promise resolving to the query result matching carbonplan/maps structure.
   */
  async queryData(
    geometry: QueryGeometry,
    selector?: Selector
  ): Promise<QueryResult> {
    if (!this.isDatatree) {
      const mode = this.getQueryMode(geometry)
      if (!mode?.queryData) return this.emptyQueryResult()
      return mode.queryData(geometry, selector)
    }

    const modes = this.getQueryModesForGeometry(geometry).filter(
      (mode) => !!mode.queryData
    )
    if (modes.length === 0) return this.emptyQueryResult()

    if (geometry.type === 'Point') {
      for (const mode of modes) {
        const result = await mode.queryData!(geometry, selector)
        if (this.hasQueryResultData(result)) return result
      }
      return this.emptyQueryResult()
    }

    const results = await Promise.all(
      modes.map((mode) => mode.queryData!(geometry, selector))
    )
    return this.mergeQueryResults(results)
  }
}
