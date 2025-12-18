/**
 * @module zarr-layer
 *
 * MapLibre/Mapbox custom layer implementation for rendering Zarr datasets.
 * Implements CustomLayerInterface for direct WebGL rendering.
 */

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
  ColormapArray,
  SpatialDimensions,
  DimIndicesProps,
  LoadingStateCallback,
  MapLike,
  Selector,
  NormalizedSelector,
  ZarrLayerOptions,
} from './types'
import type { ZarrMode, RenderContext } from './zarr-mode'
import { TiledMode } from './tiled-mode'
import { SingleImageMode } from './single-image-mode'
import {
  computeWorldOffsets,
  resolveProjectionParams,
  isGlobeProjection as checkGlobeProjection,
} from './render-utils'
import type { QueryGeometry, QueryResult } from './query/types'

export class ZarrLayer {
  readonly type: 'custom' = 'custom'
  readonly renderingMode: '2d' | '3d'

  id: string
  private url: string
  private variable: string
  private zarrVersion: 2 | 3 | null = null
  private spatialDimensions: SpatialDimensions
  private bounds: [number, number, number, number] | undefined
  private latIsAscending: boolean | null = null
  private selector: Selector
  private invalidate: () => void

  private colormap: ColormapState
  private clim: [number, number]
  private opacity: number
  private minzoom: number
  private maxzoom: number
  private selectorHash: string = ''

  private isMultiscale: boolean = true
  private _fillValue: number | null = null
  private scaleFactor: number = 1
  private offset: number = 0

  private gl: WebGL2RenderingContext | undefined
  private map: MapLike | null = null
  private renderer: ZarrRenderer | null = null
  private mode: ZarrMode | null = null
  private tileNeedsRender: boolean = true

  private projectionChangeHandler: (() => void) | null = null
  private resolveGl(
    map: MapLike,
    gl: WebGL2RenderingContext | null
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
  private levelInfos: string[] = []
  private dimIndices: DimIndicesProps = {}
  private dimensionValues: { [key: string]: Float64Array | number[] } = {}
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
  private throttleMs: number

  get fillValue(): number | null {
    return this._fillValue
  }

  private isGlobeProjection(shaderData?: ShaderData): boolean {
    if (shaderData?.vertexShaderPrelude) return true
    const projection = this.map?.getProjection ? this.map.getProjection() : null
    return checkGlobeProjection(projection)
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
    latIsAscending = null,
    fillValue,
    customFrag,
    uniforms,
    renderingMode = '3d',
    onLoadingStateChange,
    throttleMs = 100,
  }: ZarrLayerOptions) {
    if (!id) {
      throw new Error('[ZarrLayer] id is required')
    }
    if (!source) {
      throw new Error('[ZarrLayer] source is required')
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

    this.id = id
    this.url = source
    this.variable = variable
    this.zarrVersion = zarrVersion ?? null
    this.spatialDimensions = spatialDimensions
    this.bounds = bounds
    this.latIsAscending = latIsAscending ?? null
    this.selector = selector
    this.normalizedSelector = normalizeSelector(selector)
    this.selectorHash = this.computeSelectorHash(this.normalizedSelector)
    this.renderingMode = renderingMode
    this.invalidate = () => {}
    this.colormap = new ColormapState(colormap)
    this.clim = clim
    this.opacity = opacity
    this.minzoom = minzoom
    this.maxzoom = maxzoom

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
  }

  private emitLoadingState(): void {
    if (!this.onLoadingStateChange) return
    this.onLoadingStateChange({
      loading: this.metadataLoading || this.chunksLoading,
      metadata: this.metadataLoading,
      chunks: this.chunksLoading,
    })
  }

  private handleChunkLoadingChange = (state: {
    loading: boolean
    chunks: boolean
  }): void => {
    this.chunksLoading = state.chunks
    this.emitLoadingState()
  }

  setOpacity(opacity: number) {
    this.opacity = opacity
    this.invalidate()
  }

  setClim(clim: [number, number]) {
    this.clim = clim
    this.mode?.updateClim(clim)
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
      this.variable = variable
      if (this.zarrStore) {
        this.zarrStore.cleanup()
        this.zarrStore = null
      }
      this.dimensionValues = {}
      this._fillValue = null
      await this.initialize()
      await this.initializeMode()
      this.invalidate()
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

    if (this.mode) {
      await this.mode.setSelector(this.normalizedSelector)
    }
    this.invalidate()
  }

  async onAdd(map: MapLike, gl: WebGL2RenderingContext | null) {
    this.map = map
    const resolvedGl = this.resolveGl(map, gl)
    this.gl = resolvedGl
    this.invalidate = () => {
      this.tileNeedsRender = true
      if (map.triggerRepaint) map.triggerRepaint()
    }

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
        this.mode?.onProjectionChange(isGlobe)
        this.renderer?.resetSingleImageGeometry()
      }
      if (typeof map.on === 'function' && this.projectionChangeHandler) {
        map.on('projectionchange', this.projectionChangeHandler)
        map.on('style.load', this.projectionChangeHandler)
      }

      await this.initialize()
      await this.initializeMode()

      const isGlobe = this.isGlobeProjection()
      this.mode?.onProjectionChange(isGlobe)

      this.mode?.update(this.map, this.gl!)
    } finally {
      this.metadataLoading = false
      this.emitLoadingState()
    }

    this.invalidate()
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

  private async initializeMode() {
    if (!this.zarrStore || !this.gl) return

    if (this.mode) {
      this.mode.dispose(this.gl)
    }

    if (this.isMultiscale) {
      this.mode = new TiledMode(
        this.zarrStore,
        this.variable,
        this.normalizedSelector,
        this.invalidate,
        this.throttleMs
      )
    } else {
      this.mode = new SingleImageMode(
        this.zarrStore,
        this.variable,
        this.normalizedSelector,
        this.invalidate,
        this.throttleMs
      )
    }

    this.mode.setLoadingCallback(this.handleChunkLoadingChange)
    await this.mode.initialize()
    this.mode.updateClim(this.clim)

    if (this.map && this.gl) {
      this.mode.update(this.map, this.gl)
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
        latIsAscending: this.latIsAscending,
        coordinateKeys: Object.keys(this.selector),
      })

      await this.zarrStore.initialized

      const desc = this.zarrStore.describe()

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

      this.isMultiscale = this.levelInfos.length > 0

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
      console.error('Failed to initialize Zarr layer:', err)
      throw err
    }
  }

  private async loadInitialDimensionValues(): Promise<void> {
    if (!this.zarrStore?.root) return

    const multiscaleLevel =
      this.levelInfos.length > 0 ? this.levelInfos[0] : null

    for (const [dimName, value] of Object.entries(this.selector)) {
      this.normalizedSelector[dimName] = toSelectorProps(value)
    }
    for (const dimName of Object.keys(this.dimIndices)) {
      if (dimName !== 'lon' && dimName !== 'lat') {
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
    const zoom = this.map.getZoom()
    return zoom >= this.minzoom && zoom <= this.maxzoom
  }

  prerender(_gl: WebGL2RenderingContext, _params: unknown) {
    if (this.isRemoved || !this.gl || !this.mode || !this.map) return
    if (!this.isZoomInRange()) return

    this.mode.update(this.map, this.gl)
  }

  render(
    _gl: WebGL2RenderingContext,
    params: unknown,
    projection?: { name: string },
    projectionToMercatorMatrix?: number[] | Float32Array | Float64Array,
    projectionToMercatorTransition?: number,
    _centerInMercator?: number[],
    _pixelsPerMeterRatio?: number
  ) {
    if (
      this.isRemoved ||
      !this.renderer ||
      !this.gl ||
      !this.mode ||
      !this.map
    ) {
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
      },
      colormapTexture,
      worldOffsets,
      customShaderConfig: this.customShaderConfig || undefined,
      shaderData: projectionParams.shaderData,
      projectionData: projectionParams.projectionData,
      mapboxGlobe: projectionParams.mapboxGlobe,
    }

    this.mode.render(this.renderer, context)

    this.tileNeedsRender = false
  }

  renderToTile(
    _gl: WebGL2RenderingContext,
    tileId: { z: number; x: number; y: number }
  ) {
    if (
      this.isRemoved ||
      !this.renderer ||
      !this.gl ||
      !this.mode ||
      !this.map
    ) {
      return
    }

    this.mode.update(this.map, this.gl)

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
      },
      colormapTexture,
      worldOffsets: [0],
      customShaderConfig: this.customShaderConfig || undefined,
    }

    this.tileNeedsRender =
      this.mode.renderToTile?.(this.renderer, tileId, context) ?? false
  }

  // Mapbox specific custom layer method required to trigger rerender on eg dataset update.
  shouldRerenderTiles() {
    const needsRender = this.tileNeedsRender
    this.tileNeedsRender = false
    return needsRender
  }

  onRemove(_map: MapLike, gl: WebGL2RenderingContext) {
    this.isRemoved = true

    this.renderer?.dispose()
    this.renderer = null

    this.colormap.dispose(gl)

    this.mode?.dispose(gl)
    this.mode = null

    if (this.zarrStore) {
      this.zarrStore.cleanup()
      this.zarrStore = null
    }

    if (
      this.map &&
      this.projectionChangeHandler &&
      typeof this.map.off === 'function'
    ) {
      this.map.off('projectionchange', this.projectionChangeHandler)
      this.map.off('style.load', this.projectionChangeHandler)
    }
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
    if (!this.mode?.queryData) {
      return {
        [this.variable]: [],
        dimensions: ['lat', 'lon'],
        coordinates: { lat: [], lon: [] },
      }
    }
    return this.mode.queryData(geometry, selector)
  }
}
