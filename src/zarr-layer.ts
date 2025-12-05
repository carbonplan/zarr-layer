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
} from './shaders'
import { ColormapState } from './zarr-colormap'
import { ZarrRenderer, type CustomShaderConfig } from './zarr-renderer'
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
import {
  mercatorTileToGeoBounds,
  getOverlapping4326Tiles,
  get4326TileGeoBounds,
  tileToKey,
  latToMercatorNorm,
  lonToMercatorNorm,
  mercatorNormToLon,
  zoomToLevel,
} from './map-utils'
import type { TileRenderCache, TileRenderData } from './zarr-tile-cache'

const DEFAULT_TILE_SIZE = 128

export class ZarrLayer {
  readonly type: 'custom' = 'custom'
  readonly renderingMode: '2d' | '3d'

  id: string
  private url: string
  private variable: string
  private zarrVersion: 2 | 3 | null = null
  private dimensionNames: DimensionNamesProps
  private selector: Record<string, number | number[] | string | string[]>
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

  private applyWorldCopiesSetting() {
    if (
      !this.map ||
      typeof this.map.getProjection !== 'function' ||
      typeof this.map.setRenderWorldCopies !== 'function'
    ) {
      return
    }
    const isGlobe = this.isGlobeProjection()
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
  private resolveGl(
    map: MapLike,
    gl: WebGLRenderingContext | WebGL2RenderingContext | null
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
    customFragmentSource,
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
      this.selectors[dimName] = { selected: value, type: 'index' }
    }
    this.invalidate = () => {}

    if (!colormap || !Array.isArray(colormap) || colormap.length === 0) {
      throw new Error(
        '[ZarrLayer] colormap is required and must be an array of [r, g, b] values'
      )
    }
    this.colormap = new ColormapState(colormap)
    this.clim = clim
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

    // Re-create manager for new variable
    await this.initializeManager()
    this.invalidate()
  }

  async setSelector(
    selector: Record<string, number | number[] | string | string[]>
  ) {
    const nextHash = this.computeSelectorHash(selector)
    if (nextHash === this.selectorHash) {
      return
    }
    this.selectorHash = nextHash
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
    selector: Record<string, number | number[] | string | string[]>
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
    const map = this.map
    if (!map) return [0]

    const bounds = map.getBounds ? map.getBounds() : null
    if (!bounds) return [0]

    const isGlobe = this.isGlobeProjection()
    // Honor MapLibre's world copy setting, but always avoid duplicates on globe
    const renderWorldCopies =
      typeof map.getRenderWorldCopies === 'function'
        ? map.getRenderWorldCopies()
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

  prerender(
    _gl: WebGL2RenderingContext | WebGLRenderingContext,
    _params: unknown
  ) {
    if (this.isRemoved || !this.gl || !this.dataManager || !this.map) return

    // Update data manager (prefetch tiles etc)
    this.dataManager.update(this.map, this.gl)
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
    if (
      this.isRemoved ||
      !this.renderer ||
      !this.gl ||
      !this.dataManager ||
      !this.map
    )
      return

    type MatrixLike = number[] | Float32Array | Float64Array
    type ProjectionParams = {
      shaderData?: ShaderData
      defaultProjectionData?: {
        mainMatrix?: MatrixLike
        fallbackMatrix?: MatrixLike
        tileMercatorCoords?: number[]
        clippingPlane?: number[]
        projectionTransition?: number
      }
      modelViewProjectionMatrix?: MatrixLike
      projectionMatrix?: MatrixLike
    }

    const paramsObj =
      params &&
      typeof params === 'object' &&
      !Array.isArray(params) &&
      !ArrayBuffer.isView(params)
        ? (params as ProjectionParams)
        : null

    const shaderData = paramsObj?.shaderData
    let projectionData: ProjectionData | undefined
    const defaultProj = paramsObj?.defaultProjectionData
    if (
      defaultProj &&
      defaultProj.mainMatrix &&
      defaultProj.fallbackMatrix &&
      defaultProj.tileMercatorCoords &&
      defaultProj.clippingPlane &&
      typeof defaultProj.projectionTransition === 'number'
    ) {
      projectionData = {
        mainMatrix: defaultProj.mainMatrix,
        fallbackMatrix: defaultProj.fallbackMatrix,
        tileMercatorCoords: defaultProj.tileMercatorCoords as [
          number,
          number,
          number,
          number
        ],
        clippingPlane: defaultProj.clippingPlane as [
          number,
          number,
          number,
          number
        ],
        projectionTransition: defaultProj.projectionTransition,
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
      clim: this.clim,
      opacity: this.opacity,
      fillValue: this.fillValue,
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
      tileBounds: renderData.tileBounds,
      singleImage: renderData.singleImage,
      shaderData,
      projectionData,
      customShaderConfig: this.customShaderConfig || undefined,
      mapboxGlobe:
        projection && projectionToMercatorMatrix !== undefined
          ? {
              projection,
              globeToMercatorMatrix: projectionToMercatorMatrix,
              transition:
                typeof projectionToMercatorTransition === 'number'
                  ? projectionToMercatorTransition
                  : 0,
            }
          : undefined,
      mapboxTileRender: false,
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

    // Ensure data manager stays up-to-date for the requested tile
    this.dataManager.update(this.map, this.gl)

    const renderData = this.dataManager.getRenderData()
    const uniforms = {
      clim: this.clim,
      opacity: this.opacity,
      fillValue: this.fillValue,
      scaleFactor: this.scaleFactor,
      offset: this.offset,
    }

    // Single image path: render the portion of the image that overlaps with this tile
    if (!renderData.isMultiscale && renderData.singleImage) {
      const bounds = renderData.singleImage.bounds
      if (!bounds) {
        this.tileNeedsRender = false
        return
      }

      const tileSize = 1 / 2 ** tileId.z
      const tileX0 = tileId.x * tileSize
      const tileX1 = (tileId.x + 1) * tileSize
      const tileY0 = tileId.y * tileSize
      const tileY1 = (tileId.y + 1) * tileSize

      const intersects =
        bounds.x0 < tileX1 &&
        bounds.x1 > tileX0 &&
        bounds.y0 < tileY1 &&
        bounds.y1 > tileY0

      if (!intersects) {
        this.tileNeedsRender = false
        return
      }

      const overlapX0 = Math.max(bounds.x0, tileX0)
      const overlapX1 = Math.min(bounds.x1, tileX1)
      const overlapY0 = Math.max(bounds.y0, tileY0)
      const overlapY1 = Math.min(bounds.y1, tileY1)

      const localX0 = (overlapX0 - tileX0) / tileSize
      const localX1 = (overlapX1 - tileX0) / tileSize
      const localY0 = (overlapY0 - tileY0) / tileSize
      const localY1 = (overlapY1 - tileY0) / tileSize

      const clipX0 = localX0 * 2 - 1
      const clipX1 = localX1 * 2 - 1
      const clipY0 = localY0 * 2 - 1
      const clipY1 = localY1 * 2 - 1

      const scaleX = (clipX1 - clipX0) / 2
      const scaleY = (clipY1 - clipY0) / 2
      const shiftX = (clipX0 + clipX1) / 2
      const shiftY = (clipY0 + clipY1) / 2

      const imgWidth = bounds.x1 - bounds.x0
      const imgHeight = bounds.y1 - bounds.y0
      const texScaleX = imgWidth > 0 ? (overlapX1 - overlapX0) / imgWidth : 1
      const texScaleY = imgHeight > 0 ? (overlapY1 - overlapY0) / imgHeight : 1
      const texOffsetX = imgWidth > 0 ? (overlapX0 - bounds.x0) / imgWidth : 0
      const texOffsetY = imgHeight > 0 ? (overlapY0 - bounds.y0) / imgHeight : 0

      const identityMatrix = new Float32Array([
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
      ])

      this.renderer.render({
        matrix: identityMatrix,
        colormapTexture: this.colormap.ensureTexture(this.gl),
        uniforms,
        worldOffsets: [0],
        isMultiscale: false,
        visibleTiles: [],
        tileSize: renderData.tileSize || this.tileSize,
        vertexArr: renderData.vertexArr || new Float32Array(),
        pixCoordArr: renderData.pixCoordArr || new Float32Array(),
        singleImage: renderData.singleImage,
        customShaderConfig: this.customShaderConfig || undefined,
        mapboxGlobe: {
          projection: { name: 'globe' },
          globeToMercatorMatrix: identityMatrix,
          transition: 0,
        },
        mapboxTileRender: true,
        tileOverride: {
          scaleX,
          scaleY,
          shiftX,
          shiftY,
          texScale: [texScaleX, texScaleY],
          texOffset: [texOffsetX, texOffsetY],
        },
      })

      this.tileNeedsRender = false
      return
    }

    if (!renderData.tileCache) {
      this.tileNeedsRender = true
      return
    }

    const identityMatrix = new Float32Array([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    ])

    const tilesPerSide = 2 ** tileId.z
    const mapboxMercX0 = tileId.x / tilesPerSide
    const mapboxMercX1 = (tileId.x + 1) / tilesPerSide
    const mapboxMercY0 = tileId.y / tilesPerSide
    const mapboxMercY1 = (tileId.y + 1) / tilesPerSide
    const EPS = 1e-7
    const x0 = Math.max(0, mapboxMercX0 + EPS)
    const x1 = Math.min(1, mapboxMercX1 - EPS)
    const y0 = Math.max(0, mapboxMercY0 + EPS)
    const y1 = Math.min(1, mapboxMercY1 - EPS)
    const width = x1 - x0
    const height = y1 - y0

    const tileMatrix = new Float32Array([
      2 / width,
      0,
      0,
      0,
      0,
      2 / height,
      0,
      0,
      0,
      0,
      1,
      0,
      -(x0 + x1) / width,
      -(y0 + y1) / height,
      0,
      1,
    ])

    const crs = this.dataManager.getCRS()
    const xyLimits = this.dataManager.getXYLimits()
    const maxZoom = this.dataManager.getMaxZoom()

    if (crs === 'EPSG:4326' && xyLimits) {
      const mapboxGeoBounds = mercatorTileToGeoBounds(
        tileId.z,
        tileId.x,
        tileId.y
      )
      const pyramidLevel = zoomToLevel(tileId.z, maxZoom)
      const overlappingZarrTiles = getOverlapping4326Tiles(
        mapboxGeoBounds,
        xyLimits,
        pyramidLevel
      )

      if (overlappingZarrTiles.length === 0) {
        this.tileNeedsRender = false
        return
      }

      let anyTileRendered = false
      let anyMissing = false
      for (const zarrTile of overlappingZarrTiles) {
        const zarrTileKey = tileToKey(zarrTile)
        let tileData = renderData.tileCache.get(zarrTileKey)
        let renderTileKey = zarrTileKey
        let renderTileTuple = zarrTile
        if (!tileData?.data) {
          anyMissing = true
          const parent = this.findBestParentTile(
            renderData.tileCache,
            zarrTile[0],
            zarrTile[1],
            zarrTile[2]
          )
          if (!parent) continue
          tileData = parent.tile
          renderTileTuple = [
            parent.ancestorZ,
            parent.ancestorX,
            parent.ancestorY,
          ]
          renderTileKey = tileToKey(renderTileTuple)
        }

        const [z, tx, ty] = renderTileTuple
        const zarrGeoBounds = get4326TileGeoBounds(z, tx, ty, xyLimits)

        const zarrMercX0 = lonToMercatorNorm(zarrGeoBounds.west)
        const zarrMercX1 = lonToMercatorNorm(zarrGeoBounds.east)
        const zarrMercY0 = latToMercatorNorm(zarrGeoBounds.north)
        const zarrMercY1 = latToMercatorNorm(zarrGeoBounds.south)

        const overlapX0 = Math.max(zarrMercX0, mapboxMercX0)
        const overlapX1 = Math.min(zarrMercX1, mapboxMercX1)
        const overlapY0 = Math.max(zarrMercY0, mapboxMercY0)
        const overlapY1 = Math.min(zarrMercY1, mapboxMercY1)

        if (overlapX1 <= overlapX0 || overlapY1 <= overlapY0) continue

        const zarrLonWidth = zarrGeoBounds.east - zarrGeoBounds.west
        const overlapWest = mercatorNormToLon(overlapX0)
        const overlapEast = mercatorNormToLon(overlapX1)
        const texScaleX =
          zarrLonWidth > 0 ? (overlapEast - overlapWest) / zarrLonWidth : 1
        const texOffsetX =
          zarrLonWidth > 0
            ? (overlapWest - zarrGeoBounds.west) / zarrLonWidth
            : 0
        const texScaleY = 1.0
        const texOffsetY = 0.0

        const tileBoundsForRender = {
          [renderTileKey]: {
            x0: overlapX0,
            y0: overlapY0,
            x1: overlapX1,
            y1: overlapY1,
            latMin: zarrGeoBounds.south,
            latMax: zarrGeoBounds.north,
          },
        }

        this.renderer.render({
          matrix: tileMatrix,
          colormapTexture: this.colormap.ensureTexture(this.gl),
          uniforms,
          worldOffsets: [0],
          isMultiscale: true,
          visibleTiles: [renderTileTuple],
          tileCache: renderData.tileCache,
          tileSize: renderData.tileSize || this.tileSize,
          vertexArr: renderData.vertexArr || new Float32Array(),
          pixCoordArr: renderData.pixCoordArr || new Float32Array(),
          tileBounds: tileBoundsForRender,
          tileTexOverrides: {
            [renderTileKey]: {
              texScale: [texScaleX, texScaleY],
              texOffset: [texOffsetX, texOffsetY],
            },
          },
          singleImage: renderData.singleImage,
          customShaderConfig: this.customShaderConfig || undefined,
          mapboxGlobe: {
            projection: { name: 'globe' },
            globeToMercatorMatrix: identityMatrix,
            transition: 0,
          },
          mapboxTileRender: true,
        })

        anyTileRendered = true
      }

      this.tileNeedsRender = anyMissing || !anyTileRendered
      return
    }

    const tileTuple: [number, number, number] = [tileId.z, tileId.x, tileId.y]
    const tileKey = tileTuple.join(',')

    const boundsForTile = renderData.tileBounds?.[tileKey]
    const tileBoundsOverride = {
      [tileKey]: {
        x0: mapboxMercX0,
        y0: mapboxMercY0,
        x1: mapboxMercX1,
        y1: mapboxMercY1,
        latMin: boundsForTile?.latMin,
        latMax: boundsForTile?.latMax,
      },
    }

    this.renderer.render({
      matrix: tileMatrix,
      colormapTexture: this.colormap.ensureTexture(this.gl),
      uniforms,
      worldOffsets: [0],
      isMultiscale: renderData.isMultiscale,
      visibleTiles: [tileTuple],
      tileCache: renderData.tileCache,
      tileSize: renderData.tileSize || this.tileSize,
      vertexArr: renderData.vertexArr || new Float32Array(),
      pixCoordArr: renderData.pixCoordArr || new Float32Array(),
      tileBounds: tileBoundsOverride,
      singleImage: renderData.singleImage,
      customShaderConfig: this.customShaderConfig || undefined,
      mapboxGlobe: {
        projection: { name: 'globe' },
        globeToMercatorMatrix: identityMatrix,
        transition: 0,
      },
      mapboxTileRender: true,
    })

    const tileHasData = renderData.tileCache.get(tileKey)?.data
    this.tileNeedsRender = !tileHasData
  }

  private findBestParentTile(
    tileCache: TileRenderCache,
    z: number,
    x: number,
    y: number
  ): {
    tile: TileRenderData
    ancestorZ: number
    ancestorX: number
    ancestorY: number
  } | null {
    let ancestorZ = z - 1
    let ancestorX = Math.floor(x / 2)
    let ancestorY = Math.floor(y / 2)

    while (ancestorZ >= 0) {
      const parentKey = tileToKey([ancestorZ, ancestorX, ancestorY])
      const parentTile = tileCache.get(parentKey)
      if (parentTile && parentTile.data) {
        return { tile: parentTile, ancestorZ, ancestorX, ancestorY }
      }
      ancestorZ--
      ancestorX = Math.floor(ancestorX / 2)
      ancestorY = Math.floor(ancestorY / 2)
    }
    return null
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
    if (
      this.map &&
      typeof this.map.setRenderWorldCopies === 'function' &&
      this.initialRenderWorldCopies !== undefined
    ) {
      this.map.setRenderWorldCopies(this.initialRenderWorldCopies)
    }
  }
}
