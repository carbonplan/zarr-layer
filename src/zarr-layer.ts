/**
 * @module zarr-layer
 *
 * MapLibre/MapBox custom layer implementation for rendering Zarr datasets.
 * Implements CustomLayerInterface for direct WebGL rendering.
 */

import { loadDimensionValues, getBands, toSelectorProps } from './zarr-utils'
import { ZarrStore } from './zarr-store'
import { maplibreFragmentShaderSource, type ShaderData } from './shaders'
import { ColormapState } from './zarr-colormap'
import { ZarrRenderer } from './zarr-renderer'
import type { CustomShaderConfig } from './renderer-types'
import type {
  ColormapArray,
  DimensionNamesProps,
  DimIndicesProps,
  LoadingStateCallback,
  MapLike,
  ZarrLayerOptions,
  ZarrSelectorsProps,
} from './types'
import { DataManager } from './data-manager'
import { TiledDataManager } from './tiled-data-manager'
import { SingleImageDataManager } from './single-image-data-manager'
import { renderMapboxTile } from './mapbox-globe-tile-renderer'
import { computeWorldOffsets, resolveProjectionParams } from './render-utils'

const DEFAULT_TILE_SIZE = 128

export class ZarrLayer {
  readonly type: 'custom' = 'custom'
  readonly renderingMode: '2d' | '3d'

  id: string
  private url: string
  private variable: string
  private zarrVersion: 2 | 3 | null = null
  private dimensionNames: DimensionNamesProps
  private selector: Record<
    string,
    number | number[] | string | string[] | ZarrSelectorsProps
  >
  private invalidate: () => void

  private colormap: ColormapState
  private clim: [number, number]
  private opacity: number
  private minRenderZoom: number
  private selectorHash: string = ''

  private tileSize: number = DEFAULT_TILE_SIZE
  private isMultiscale: boolean = true
  private fillValue: number | null = null
  private scaleFactor: number = 1
  private offset: number = 0

  private gl: WebGL2RenderingContext | undefined
  private map: MapLike | null = null
  private renderer: ZarrRenderer | null = null
  private dataManager: DataManager | null = null
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
    throw new Error('MapLibre did not provide a valid WebGL2 context')
  }

  private zarrStore: ZarrStore | null = null
  private levelInfos: string[] = []
  private dimIndices: DimIndicesProps = {}
  private dimensionValues: { [key: string]: Float64Array | number[] } = {}
  private selectors: { [key: string]: ZarrSelectorsProps } = {}
  private isRemoved: boolean = false
  private fragmentShaderSource: string = maplibreFragmentShaderSource
  private customFrag: string | undefined
  private customUniforms: Record<string, number> = {}
  private bandNames: string[] = []
  private customShaderConfig: CustomShaderConfig | null = null
  private onLoadingStateChange: LoadingStateCallback | undefined
  private metadataLoading: boolean = false
  private chunksLoading: boolean = false

  private isGlobeProjection(shaderData?: ShaderData): boolean {
    if (shaderData?.vertexShaderPrelude) return true
    const projection = this.map?.getProjection ? this.map.getProjection() : null
    return projection?.type === 'globe' || projection?.name === 'globe'
  }

  constructor({
    id,
    source,
    variable,
    selector = {},
    colormap,
    clim,
    opacity = 1,
    minRenderZoom = 0,
    zarrVersion,
    dimensionNames = {},
    fillValue,
    customFrag,
    uniforms,
    renderingMode = '2d',
    onLoadingStateChange,
  }: ZarrLayerOptions) {
    this.id = id
    this.url = source
    this.variable = variable
    this.zarrVersion = zarrVersion ?? null
    this.dimensionNames = dimensionNames
    this.selector = selector
    this.selectorHash = this.computeSelectorHash(selector)
    this.renderingMode = renderingMode
    for (const [dimName, value] of Object.entries(selector)) {
      this.selectors[dimName] = toSelectorProps(value)
    }
    this.invalidate = () => {}

    if (!colormap || !Array.isArray(colormap) || colormap.length === 0) {
      throw new Error(
        '[ZarrLayer] colormap is required and must be an array of [r, g, b] or hex string values'
      )
    }
    this.colormap = new ColormapState(colormap)
    this.clim = clim
    this.opacity = opacity
    this.minRenderZoom = minRenderZoom

    this.customFrag = customFrag
    this.customUniforms = uniforms || {}

    this.bandNames = getBands(variable, selector)
    if (this.bandNames.length > 1 || customFrag) {
      this.customShaderConfig = {
        bands: this.bandNames,
        customFrag: customFrag,
        customUniforms: this.customUniforms,
      }
    }

    if (fillValue !== undefined) this.fillValue = fillValue
    this.onLoadingStateChange = onLoadingStateChange
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
    this.variable = variable
    if (this.zarrStore) {
      this.zarrStore.variable = variable
    }
    await this.initializeManager()
    this.invalidate()
  }

  async setSelector(
    selector: Record<
      string,
      number | number[] | string | string[] | ZarrSelectorsProps
    >
  ) {
    const nextHash = this.computeSelectorHash(selector)
    if (nextHash === this.selectorHash) {
      return
    }
    this.selectorHash = nextHash
    this.selector = selector
    for (const [dimName, value] of Object.entries(selector)) {
      this.selectors[dimName] = toSelectorProps(value)
    }

    this.bandNames = getBands(this.variable, selector)
    if (this.bandNames.length > 1 || this.customFrag) {
      this.customShaderConfig = {
        bands: this.bandNames,
        customFrag: this.customFrag,
        customUniforms: this.customUniforms,
      }
    } else {
      this.customShaderConfig = null
    }

    if (this.dataManager) {
      await this.dataManager.setSelector(selector)
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
        this.dataManager?.onProjectionChange(isGlobe)
        this.renderer?.resetSingleImageGeometry()
      }
      if (typeof map.on === 'function' && this.projectionChangeHandler) {
        map.on('projectionchange', this.projectionChangeHandler)
        map.on('style.load', this.projectionChangeHandler)
      }

      await this.initialize()
      await this.initializeManager()

      const isGlobe = this.isGlobeProjection()
      this.dataManager?.onProjectionChange(isGlobe)

      this.dataManager?.update(this.map, this.gl!)
    } finally {
      this.metadataLoading = false
      this.emitLoadingState()
    }

    this.invalidate()
  }

  private computeSelectorHash(
    selector: Record<
      string,
      number | number[] | string | string[] | ZarrSelectorsProps
    >
  ): string {
    return JSON.stringify(selector, Object.keys(selector).sort())
  }

  private async initializeManager() {
    if (!this.zarrStore || !this.gl) return

    if (this.dataManager) {
      this.dataManager.dispose(this.gl)
    }

    if (this.isMultiscale) {
      this.dataManager = new TiledDataManager(
        this.zarrStore,
        this.variable,
        this.selector,
        this.minRenderZoom,
        this.invalidate
      )
    } else {
      this.dataManager = new SingleImageDataManager(
        this.zarrStore,
        this.variable,
        this.selector,
        this.invalidate
      )
    }

    this.dataManager.setLoadingCallback(this.handleChunkLoadingChange)
    await this.dataManager.initialize()

    if (this.map && this.gl) {
      this.dataManager.update(this.map, this.gl)
    }
  }

  private async initialize(): Promise<void> {
    try {
      this.zarrStore = new ZarrStore({
        source: this.url,
        version: this.zarrVersion,
        variable: this.variable,
        dimensionNames: this.dimensionNames,
        coordinateKeys: Object.keys(this.selector),
      })

      await this.zarrStore.initialized

      const desc = this.zarrStore.describe()

      this.levelInfos = desc.levels
      this.dimIndices = desc.dimIndices
      this.scaleFactor = desc.scaleFactor
      this.offset = desc.addOffset
      this.tileSize = desc.tileSize || DEFAULT_TILE_SIZE

      if (
        this.fillValue === null &&
        desc.fill_value !== null &&
        desc.fill_value !== undefined
      ) {
        this.fillValue = desc.fill_value
      }

      this.isMultiscale = this.levelInfos.length > 0

      // Load initial dimension values for UI if needed (kept from original)
      // But we mostly delegate to manager now for data loading.
      await this.loadInitialDimensionValues()
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
      this.selectors[dimName] = toSelectorProps(value)
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

          if (!this.selectors[dimName]) {
            this.selectors[dimName] = { selected: 0 }
          }
        } catch (err) {
          console.warn(`Failed to load dimension values for ${dimName}:`, err)
        }
      }
    }
  }

  prerender(_gl: WebGL2RenderingContext, _params: unknown) {
    if (this.isRemoved || !this.gl || !this.dataManager || !this.map) return

    // Update data manager (prefetch tiles etc)
    this.dataManager.update(this.map, this.gl)
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
      !this.dataManager ||
      !this.map
    ) {
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
    const uniforms = {
      clim: this.clim,
      opacity: this.opacity,
      fillValue: this.fillValue,
      scaleFactor: this.scaleFactor,
      offset: this.offset,
    }

    const renderData = this.dataManager.getRenderData()

    this.renderer.render({
      matrix: projectionParams.matrix,
      colormapTexture,
      uniforms,
      worldOffsets,
      isMultiscale: renderData.isMultiscale,
      visibleTiles: renderData.visibleTiles || [],
      tileCache: renderData.tileCache,
      tileSize: renderData.tileSize || this.tileSize,
      vertexArr: renderData.vertexArr || new Float32Array(),
      pixCoordArr: renderData.pixCoordArr || new Float32Array(),
      tileBounds: renderData.tileBounds,
      singleImage: renderData.singleImage,
      shaderData: projectionParams.shaderData,
      projectionData: projectionParams.projectionData,
      customShaderConfig: this.customShaderConfig || undefined,
      mapboxGlobe: projectionParams.mapboxGlobe,
      mode: { type: 'standard' },
    })

    // main render path handled; tile path not needed this frame
    this.tileNeedsRender = false
  }

  // Mapbox globe draping path
  renderToTile(
    _gl: WebGL2RenderingContext,
    tileId: { z: number; x: number; y: number }
  ) {
    if (
      this.isRemoved ||
      !this.renderer ||
      !this.gl ||
      !this.dataManager ||
      !this.map
    ) {
      return
    }

    this.dataManager.update(this.map, this.gl)

    const renderData = this.dataManager.getRenderData()
    const colormapTexture = this.colormap.ensureTexture(this.gl)
    const uniforms = {
      clim: this.clim,
      opacity: this.opacity,
      fillValue: this.fillValue,
      scaleFactor: this.scaleFactor,
      offset: this.offset,
    }

    this.tileNeedsRender = renderMapboxTile({
      renderer: this.renderer,
      renderData,
      tileId,
      colormapTexture,
      uniforms,
      tileSize: renderData.tileSize || this.tileSize,
      customShaderConfig: this.customShaderConfig || undefined,
      dataManager: this.dataManager,
    })
  }

  shouldRerenderTiles(): boolean {
    return this.tileNeedsRender
  }

  onRemove(_map: MapLike, gl: WebGL2RenderingContext) {
    this.isRemoved = true

    this.renderer?.dispose()
    this.renderer = null

    this.colormap.dispose(gl)

    this.dataManager?.dispose(gl)
    this.dataManager = null

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
}
