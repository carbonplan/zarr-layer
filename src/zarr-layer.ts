/**
 * @module zarr-layer
 *
 * MapLibre/MapBox custom layer implementation for rendering Zarr datasets.
 * Implements CustomLayerInterface for direct WebGL rendering.
 */

import {
  calculateNearestIndex,
  loadDimensionValues,
  getBands,
} from './zarr-utils'
import { ZarrStore } from './zarr-store'
import {
  maplibreFragmentShaderSource,
  type ProjectionData,
  type ShaderData,
} from './maplibre-shaders'
import { ColormapState } from './zarr-colormap'
import { ZarrRenderer, type CustomShaderConfig } from './zarr-renderer'
import type {
  ColorMapName,
  CRS,
  DimensionNamesProps,
  DimIndicesProps,
  MaplibreLayerOptions,
  XYLimits,
  ZarrLevelMetadata,
  ZarrSelectorsProps,
} from './types'
import { DataManager } from './data-manager'
import { TiledDataManager } from './tiled-data-manager'
import { SingleImageDataManager } from './single-image-data-manager'

const DEFAULT_TILE_SIZE = 128

export class ZarrLayer {
  readonly type: 'custom' = 'custom'
  readonly renderingMode: '2d' | '3d'

  id: string
  private url: string
  private variable: string
  private zarrVersion: 2 | 3 | null = null
  private dimensionNames: DimensionNamesProps
  private selector: Record<string, number | number[]>
  private invalidate: () => void

  private colormap: ColormapState
  private vmin: number
  private vmax: number
  private opacity: number
  private minRenderZoom: number

  private maxZoom: number = 4
  private tileSize: number = DEFAULT_TILE_SIZE
  private isMultiscale: boolean = true
  private fillValue: number = 0
  private useFillValue: boolean = false
  private noDataMin: number = -9999
  private noDataMax: number = 9999
  private scaleFactor: number = 1
  private offset: number = 0

  private gl: WebGL2RenderingContext | undefined
  private map: any
  private renderer: ZarrRenderer | null = null
  private dataManager: DataManager | null = null

  private applyWorldCopiesSetting() {
    if (
      !this.map ||
      typeof this.map.getProjection !== 'function' ||
      typeof this.map.setRenderWorldCopies !== 'function'
    ) {
      return
    }
    const projection = this.map.getProjection()
    const isGlobe = projection?.type === 'globe' || projection?.name === 'globe'
    const target = isGlobe
      ? false
      : this.initialRenderWorldCopies !== undefined
      ? this.initialRenderWorldCopies
      : true

    const current =
      typeof this.map.getRenderWorldCopies === 'function'
        ? this.map.getRenderWorldCopies()
        : undefined
    if (current !== target) {
      this.map.setRenderWorldCopies(target)
    }
  }
  private initialRenderWorldCopies: boolean | undefined
  private projectionChangeHandler: (() => void) | null = null
  private resolveGl(map: any, gl: any): WebGL2RenderingContext {
    const isWebGL2 =
      gl &&
      typeof (gl as any).getUniformLocation === 'function' &&
      typeof (gl as any).drawBuffers === 'function'
    if (isWebGL2) {
      return gl as WebGL2RenderingContext
    }

    const describe = (obj: any) =>
      obj
        ? {
            type: obj.constructor?.name,
            keys: Object.keys(obj),
          }
        : null
    console.error('Invalid WebGL2 context passed to onAdd', {
      providedGl: describe(gl),
      painterGl: describe((map as any)?.painter?.context?.gl),
      rendererGl: describe((map as any)?.renderer?.getContext?.()),
    })
    throw new Error('MapLibre did not provide a valid WebGL2 context')
  }

  private zarrStore: ZarrStore | null = null
  private levelInfos: string[] = []
  private levelMetadata: Map<number, ZarrLevelMetadata> = new Map()
  private dimIndices: DimIndicesProps = {}
  private xyLimits: XYLimits | null = null
  private crs: CRS | null = null
  private dimensionValues: { [key: string]: Float64Array | number[] } = {}
  private selectors: { [key: string]: ZarrSelectorsProps } = {}
  private isRemoved: boolean = false
  private fragmentShaderSource: string = maplibreFragmentShaderSource
  private customFrag: string | undefined
  private customUniforms: Record<string, number> = {}
  private bandNames: string[] = []
  private customShaderConfig: CustomShaderConfig | null = null

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
    colormap = 'viridis',
    vmin,
    vmax,
    opacity = 1,
    minRenderZoom = 3,
    zarrVersion,
    dimensionNames = {},
    noDataMin,
    noDataMax,
    customFragmentSource,
    customFrag,
    uniforms,
    renderingMode = '2d',
  }: MaplibreLayerOptions) {
    this.id = id
    this.url = source
    this.variable = variable
    this.zarrVersion = zarrVersion ?? null
    this.dimensionNames = dimensionNames
    this.selector = selector
    this.renderingMode = renderingMode
    for (const [dimName, value] of Object.entries(selector)) {
      this.selectors[dimName] = { selected: value, type: 'index' }
    }
    this.invalidate = () => {}

    this.colormap = new ColormapState(colormap)
    this.vmin = vmin
    this.vmax = vmax
    this.opacity = opacity
    this.minRenderZoom = minRenderZoom
    if (customFragmentSource) {
      this.fragmentShaderSource = customFragmentSource
    }

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

    if (noDataMin !== undefined) this.noDataMin = noDataMin
    if (noDataMax !== undefined) this.noDataMax = noDataMax
  }

  setOpacity(opacity: number) {
    this.opacity = opacity
    this.invalidate()
  }

  setVminVmax(vmin: number, vmax: number) {
    this.vmin = vmin
    this.vmax = vmax
    this.invalidate()
  }

  setColormap(colormap: ColorMapName | number[][] | string[]) {
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

    // Re-create manager for new variable
    await this.initializeManager()
    this.invalidate()
  }

  async setSelector(selector: Record<string, number | number[]>) {
    this.selector = selector
    for (const [dimName, value] of Object.entries(selector)) {
      this.selectors[dimName] = { selected: value, type: 'index' }
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

  async onAdd(map: any, gl: WebGL2RenderingContext) {
    this.map = map
    const resolvedGl = this.resolveGl(map, gl)
    this.gl = resolvedGl
    this.invalidate = () => map.triggerRepaint()

    this.colormap.upload(resolvedGl as WebGL2RenderingContext)
    this.renderer = new ZarrRenderer(
      resolvedGl as WebGL2RenderingContext,
      this.fragmentShaderSource
    )

    if (typeof map.getRenderWorldCopies === 'function') {
      this.initialRenderWorldCopies = map.getRenderWorldCopies()
    }
    this.projectionChangeHandler = () => {
      const isGlobe = this.isGlobeProjection()
      this.applyWorldCopiesSetting()
      this.dataManager?.onProjectionChange(isGlobe)
      this.renderer?.resetSingleImageGeometry()
    }
    if (typeof map.on === 'function' && this.projectionChangeHandler) {
      map.on('projectionchange', this.projectionChangeHandler)
      map.on('style.load', this.projectionChangeHandler)
    }
    this.applyWorldCopiesSetting()

    await this.initialize()
    await this.initializeManager()

    // Ensure correct initial projection state
    const isGlobe = this.isGlobeProjection()
    this.dataManager?.onProjectionChange(isGlobe)

    // Trigger initial update
    this.dataManager?.update(this.map, this.gl!)
    this.invalidate()
  }

  private async initializeManager() {
    if (!this.zarrStore || !this.gl) return

    // Dispose old manager if exists
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

    await this.dataManager.initialize()

    // Initial update if map is ready
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
      })

      await this.zarrStore.initialized

      const desc = this.zarrStore.describe()

      this.levelInfos = desc.levels
      this.dimIndices = desc.dimIndices
      this.xyLimits = desc.xyLimits
      this.crs = desc.crs
      this.scaleFactor = desc.scaleFactor
      this.offset = desc.addOffset
      this.tileSize = desc.tileSize || DEFAULT_TILE_SIZE

      if (desc.fill_value !== null && desc.fill_value !== undefined) {
        this.fillValue = desc.fill_value
        this.useFillValue = true
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
      this.selectors[dimName] = { selected: value, type: 'index' }
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
            this.selectors[dimName] = { selected: 0, type: 'index' }
          } else if (this.selectors[dimName].type === 'value') {
            this.selectors[dimName].selected = calculateNearestIndex(
              this.dimensionValues[dimName],
              this.selectors[dimName].selected as number
            )
          }
        } catch (err) {
          console.warn(`Failed to load dimension values for ${dimName}:`, err)
        }
      }
    }
  }

  private getWorldOffsets(): number[] {
    const bounds = this.map.getBounds()
    if (!bounds) return [0]

    const projection = this.map.getProjection ? this.map.getProjection() : null
    const isGlobe = projection?.name === 'globe' || projection?.type === 'globe'
    // Honor MapLibre's world copy setting, but always avoid duplicates on globe
    const renderWorldCopies =
      typeof this.map.getRenderWorldCopies === 'function'
        ? this.map.getRenderWorldCopies()
        : true
    if (isGlobe || !renderWorldCopies) return [0]

    const west = bounds.getWest()
    const east = bounds.getEast()

    const minWorld = Math.floor((west + 180) / 360)
    const maxWorld = Math.floor((east + 180) / 360)

    const worldOffsets: number[] = []
    for (let i = minWorld; i <= maxWorld; i++) {
      worldOffsets.push(i)
    }
    return worldOffsets.length > 0 ? worldOffsets : [0]
  }

  private getSelectorHash(): string {
    return JSON.stringify(this.selector)
  }

  prerender(_gl: WebGL2RenderingContext | WebGLRenderingContext, _params: any) {
    if (this.isRemoved || !this.gl || !this.dataManager) return

    // Update data manager (prefetch tiles etc)
    this.dataManager.update(this.map, this.gl)
  }

  render(_gl: WebGL2RenderingContext | WebGLRenderingContext, params: any) {
    if (this.isRemoved || !this.renderer || !this.gl || !this.dataManager)
      return

    const paramsObj =
      params && typeof params === 'object' && !Array.isArray(params)
        ? (params as any)
        : null

    const shaderData = paramsObj?.shaderData
    let projectionData: ProjectionData | undefined
    if (paramsObj?.defaultProjectionData) {
      projectionData = {
        mainMatrix: paramsObj.defaultProjectionData.mainMatrix,
        fallbackMatrix: paramsObj.defaultProjectionData.fallbackMatrix,
        tileMercatorCoords: paramsObj.defaultProjectionData.tileMercatorCoords,
        clippingPlane: paramsObj.defaultProjectionData.clippingPlane,
        projectionTransition:
          paramsObj.defaultProjectionData.projectionTransition,
      }
    }
    let matrix: number[] | Float32Array | Float64Array | null = null
    if (projectionData?.mainMatrix && projectionData.mainMatrix.length) {
      matrix = projectionData.mainMatrix
    } else if (
      Array.isArray(params) ||
      params instanceof Float32Array ||
      params instanceof Float64Array
    ) {
      matrix = params as number[] | Float32Array | Float64Array
    } else if (paramsObj?.modelViewProjectionMatrix) {
      matrix = paramsObj.modelViewProjectionMatrix
    } else if (paramsObj?.projectionMatrix) {
      matrix = paramsObj.projectionMatrix
    }

    if (!matrix) {
      return
    }

    const worldOffsets = this.getWorldOffsets()
    const colormapTexture = this.colormap.ensureTexture(this.gl)

    const uniforms = {
      vmin: this.vmin,
      vmax: this.vmax,
      opacity: this.opacity,
      fillValue: this.fillValue,
      useFillValue: this.useFillValue,
      noDataMin: this.noDataMin,
      noDataMax: this.noDataMax,
      scaleFactor: this.scaleFactor,
      offset: this.offset,
    }

    const renderData = this.dataManager.getRenderData()

    this.renderer.render({
      matrix,
      colormapTexture,
      uniforms,
      worldOffsets,
      isMultiscale: renderData.isMultiscale,
      visibleTiles: renderData.visibleTiles || [],
      tileCache: renderData.tileCache,
      tileSize: renderData.tileSize || this.tileSize,
      vertexArr: renderData.vertexArr || new Float32Array(),
      pixCoordArr: renderData.pixCoordArr || new Float32Array(),
      singleImage: renderData.singleImage,
      shaderData,
      projectionData,
      customShaderConfig: this.customShaderConfig || undefined,
    })
  }

  onRemove(_map: any, gl: WebGL2RenderingContext) {
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
    if (
      this.map &&
      typeof this.map.setRenderWorldCopies === 'function' &&
      this.initialRenderWorldCopies !== undefined
    ) {
      this.map.setRenderWorldCopies(this.initialRenderWorldCopies)
    }
  }
}
