/**
 * @module zarr-layer
 *
 * MapLibre/MapBox custom layer implementation for rendering Zarr datasets.
 * Implements CustomLayerInterface for direct WebGL rendering.
 */

import * as zarr from 'zarrita'
import { calculateNearestIndex, loadDimensionValues } from './zarr-utils'
import { ZarrStore } from './zarr-store'
import { Tiles } from './tiles'
import { mustCreateBuffer, mustCreateTexture } from './webgl-utils'
import {
  maplibreFragmentShaderSource,
  type ProjectionData,
  type ShaderData,
} from './maplibre-shaders'
import {
  boundsToMercatorNorm,
  getTilesAtZoom,
  type MercatorBounds,
  tileToKey,
  type TileTuple,
  zoomToLevel,
} from './maplibre-utils'
import { ColormapState } from './zarr-colormap'
import { ZarrRenderer } from './zarr-renderer'
import { TileRenderCache } from './zarr-tile-cache'
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

const DEFAULT_TILE_SIZE = 128
const MAX_CACHED_TILES = 64
const TILE_SUBDIVISIONS = 16

export class ZarrLayer {
  type: 'custom' = 'custom'
  renderingMode: '2d' = '2d'

  id: string
  private url: string
  private variable: string
  private zarrVersion: 2 | 3 | null = null
  private dimensionNames: DimensionNamesProps
  private selector: Record<string, number>
  private invalidate: () => void

  private colormap: ColormapState
  private vmin: number
  private vmax: number
  private opacity: number
  private minRenderZoom: number

  private tileCache: TileRenderCache | null = null
  private tilesManager: Tiles | null = null
  private maxZoom: number = 4
  private tileSize: number = DEFAULT_TILE_SIZE
  private isMultiscale: boolean = true
  private singleImageData: Float32Array | null = null
  private singleImageTexture: WebGLTexture | null = null
  private singleImageVertexBuffer: WebGLBuffer | null = null
  private singleImagePixCoordBuffer: WebGLBuffer | null = null
  private singleImageWidth: number = 0
  private singleImageHeight: number = 0
  private mercatorBounds: MercatorBounds | null = null
  private fillValue: number = 0
  private useFillValue: boolean = false
  private noDataMin: number = -9999
  private noDataMax: number = 9999
  private scaleFactor: number = 1
  private offset: number = 0

  private gl: WebGL2RenderingContext | undefined
  private map: any
  private renderer: ZarrRenderer | null = null
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

  private vertexArr: Float32Array = new Float32Array()
  private pixCoordArr: Float32Array = new Float32Array()
  private singleImagePixCoordArr: Float32Array = new Float32Array()
  private currentSubdivisions: number = 0

  private zarrArray: zarr.Array<any> | null = null
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

  private static createSubdividedQuad(
    subdivisions: number,
    flipY: boolean
  ): { vertexArr: Float32Array; texCoordArr: Float32Array } {
    const vertices: number[] = []
    const texCoords: number[] = []
    const step = 2 / subdivisions
    const texStep = 1 / subdivisions

    const pushVertex = (col: number, row: number) => {
      const x = -1 + col * step
      const y = 1 - row * step
      const u = col * texStep
      const vBase = row * texStep
      const v = flipY ? 1 - vBase : vBase
      vertices.push(x, y)
      texCoords.push(u, v)
    }

    for (let row = 0; row < subdivisions; row++) {
      for (let col = 0; col <= subdivisions; col++) {
        pushVertex(col, row)
        pushVertex(col, row + 1)
      }
      if (row < subdivisions - 1) {
        // Degenerate vertices to connect strips
        pushVertex(subdivisions, row + 1)
        pushVertex(0, row + 1)
      }
    }

    return {
      vertexArr: new Float32Array(vertices),
      texCoordArr: new Float32Array(texCoords),
    }
  }

  private isGlobeProjection(shaderData?: ShaderData): boolean {
    if (shaderData?.vertexShaderPrelude) return true
    const projection = this.map?.getProjection ? this.map.getProjection() : null
    return projection?.type === 'globe' || projection?.name === 'globe'
  }

  private updateGeometryForProjection(isGlobe: boolean) {
    const targetSubdivisions = isGlobe ? TILE_SUBDIVISIONS : 1
    if (this.currentSubdivisions === targetSubdivisions) return

    const subdivided = ZarrLayer.createSubdividedQuad(
      targetSubdivisions,
      false
    )
    this.vertexArr = subdivided.vertexArr
    this.pixCoordArr = subdivided.texCoordArr

    const subdividedSingle = ZarrLayer.createSubdividedQuad(
      targetSubdivisions,
      true
    )
    this.singleImagePixCoordArr = subdividedSingle.texCoordArr

    this.currentSubdivisions = targetSubdivisions
    this.tileCache?.markGeometryDirty()
    this.renderer?.resetSingleImageGeometry()
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
  }: MaplibreLayerOptions) {
    this.id = id
    this.url = source
    this.variable = variable
    this.zarrVersion = zarrVersion ?? null
    this.dimensionNames = dimensionNames
    this.selector = selector
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

    if (noDataMin !== undefined) this.noDataMin = noDataMin
    if (noDataMax !== undefined) this.noDataMax = noDataMax

    // Default to unsubdivided geometry; globe mode will enable finer grids.
    this.updateGeometryForProjection(false)
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

  async setVariable(variable: string) {
    this.variable = variable
    this.clearAllTiles()
    await this.prepareTiles()
    this.getVisibleTiles()
    await this.prefetchTileData()
    this.invalidate()
  }

  private clearAllTiles() {
    if (this.tileCache) {
      this.tileCache.clear()
    }
  }

  async setSelector(selector: Record<string, number>) {
    this.selector = selector
    for (const [dimName, value] of Object.entries(selector)) {
      this.selectors[dimName] = { selected: value, type: 'index' }
    }
    this.tilesManager?.updateSelector(this.selectors)
    if (!this.isMultiscale) {
      this.singleImageData = null
      await this.prefetchTileData()
    } else {
      this.reextractTileSlices()
    }
    this.invalidate()
  }

  private async reextractTileSlices() {
    if (!this.tilesManager) return

    const currentHash = this.getSelectorHash()
    const visibleTiles = this.getVisibleTiles()

    await this.tilesManager.reextractTileSlices(visibleTiles, currentHash)

    for (const tileTuple of visibleTiles) {
      const tileKey = tileToKey(tileTuple)
      const tile = this.tileCache?.get(tileKey)
      const cache = this.tilesManager.getTile(tileTuple)
      if (!tile || !cache) continue
      tile.data = cache.data
      tile.selectorHash = cache.selectorHash
    }

    await this.prefetchTileData()
  }

  async onAdd(map: any, gl: WebGL2RenderingContext) {
    this.map = map
    const resolvedGl = this.resolveGl(map, gl)
    this.gl = resolvedGl
    this.invalidate = () => map.triggerRepaint()
    this.tileCache = new TileRenderCache(
      resolvedGl as WebGL2RenderingContext,
      MAX_CACHED_TILES
    )
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
      this.updateGeometryForProjection(isGlobe)
    }
    if (typeof map.on === 'function' && this.projectionChangeHandler) {
      map.on('projectionchange', this.projectionChangeHandler)
      map.on('style.load', this.projectionChangeHandler)
    }
    this.applyWorldCopiesSetting()
    this.updateGeometryForProjection(this.isGlobeProjection())

    await this.initialize()
    await this.prepareTiles()

    this.prefetchTileData().then(() => {
      this.invalidate()
    })
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

      if (this.levelInfos.length > 0) {
        this.zarrArray = await this.zarrStore.getLevelArray(this.levelInfos[0])
      } else {
        this.zarrArray = await this.zarrStore.getArray()
      }

      for (let i = 0; i < this.levelInfos.length; i++) {
        const levelArr = await this.zarrStore.getLevelArray(this.levelInfos[i])
        const width = levelArr.shape[this.dimIndices.lon?.index ?? 1]
        const height = levelArr.shape[this.dimIndices.lat?.index ?? 0]
        this.levelMetadata.set(i, { width, height })
      }

      await this.loadInitialDimensionValues()

      this.tilesManager = new Tiles({
        store: this.zarrStore,
        selectors: this.selectors,
        fillValue: this.fillValue,
        dimIndices: this.dimIndices,
        maxCachedTiles: MAX_CACHED_TILES,
      })
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

  async prefetchTileData() {
    if (!this.isMultiscale) {
      await this.fetchSingleImageData()
      return
    }

    const tiles = this.getVisibleTiles()
    const fetchPromises = tiles.map((tiletuple) =>
      this.fetchTileData(tiletuple)
    )
    await Promise.all(fetchPromises)
  }

  getVisibleTiles(): TileTuple[] {
    const mapZoom = this.map.getZoom()
    if (mapZoom < this.minRenderZoom) {
      return []
    }
    const pyramidLevel = zoomToLevel(mapZoom, this.maxZoom)

    const bounds = this.map.getBounds()?.toArray()
    if (!bounds) {
      return []
    }
    const tiles = getTilesAtZoom(pyramidLevel, bounds)
    return tiles
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

  async prepareTiles() {
    if (typeof this.gl === 'undefined') {
      throw new Error('Cannot prepareTiles with no GL context set')
    }

    if (this.levelInfos.length === 0) {
      this.isMultiscale = false
      await this.prepareSingleImage()
      return
    }

    this.isMultiscale = true
    this.maxZoom = this.levelInfos.length - 1
  }

  private getSelectorHash(): string {
    return JSON.stringify(this.selector)
  }

  private async prepareSingleImage(): Promise<void> {
    if (!this.gl || !this.zarrArray || !this.xyLimits) {
      console.warn(
        'Cannot prepare single image: missing GL context, zarrArray, or xyLimits'
      )
      return
    }

    const gl = this.gl

    this.mercatorBounds = boundsToMercatorNorm(this.xyLimits, this.crs)

    this.singleImageTexture = mustCreateTexture(gl)
    this.singleImageVertexBuffer = mustCreateBuffer(gl)
    this.singleImagePixCoordBuffer = mustCreateBuffer(gl)

    this.singleImageWidth = this.zarrArray.shape[this.dimIndices.lon.index]
    this.singleImageHeight = this.zarrArray.shape[this.dimIndices.lat.index]
  }

  private async fetchSingleImageData(): Promise<Float32Array | null> {
    if (!this.zarrArray || this.singleImageData || this.isRemoved) {
      return this.singleImageData
    }

    try {
      const sliceArgs: any[] = new Array(this.zarrArray.shape.length).fill(0)

      for (const dimName of Object.keys(this.dimIndices)) {
        const dimInfo = this.dimIndices[dimName]
        if (dimName === 'lon') {
          sliceArgs[dimInfo.index] = zarr.slice(0, this.singleImageWidth)
        } else if (dimName === 'lat') {
          sliceArgs[dimInfo.index] = zarr.slice(0, this.singleImageHeight)
        } else {
          const dimSelection = this.selectors[dimName] || this.selector[dimName]
          if (dimSelection !== undefined) {
            sliceArgs[dimInfo.index] =
              typeof dimSelection === 'object'
                ? (dimSelection.selected as number)
                : dimSelection
          } else {
            sliceArgs[dimInfo.index] = 0
          }
        }
      }

      const data = await zarr.get(this.zarrArray, sliceArgs)
      if (this.isRemoved) return null
      this.singleImageData = new Float32Array(
        (data.data as Float32Array).buffer
      )
      this.invalidate()
      return this.singleImageData
    } catch (err) {
      console.error('Error fetching single image data:', err)
      return null
    }
  }

  private async fetchTileData(
    tileTuple: TileTuple
  ): Promise<Float32Array | null> {
    if (this.isRemoved || !this.tilesManager || !this.gl || !this.tileCache)
      return null

    const tileKey = tileToKey(tileTuple)
    const tile = this.tileCache.upsert(tileKey)
    const currentHash = this.getSelectorHash()

    if (tile.data && tile.selectorHash === currentHash) {
      return tile.data
    }

    const cache = await this.tilesManager.fetchTile(tileTuple, currentHash)
    if (!cache || this.isRemoved) {
      return null
    }

    tile.data = cache.data
    tile.selectorHash = cache.selectorHash
    this.invalidate()

    return tile.data
  }

  prerender(
    _gl: WebGL2RenderingContext,
    _params: number[] | Float32Array | Float64Array | any
  ) {
    if (this.isRemoved || !this.gl || !this.tileCache) return

    if (this.isMultiscale) {
      this.prefetchTileData()
    } else if (!this.singleImageData) {
      this.prefetchTileData()
    }
  }

  render(
    _gl: WebGL2RenderingContext,
    params:
      | number[]
      | Float32Array
      | Float64Array
      | {
          modelViewProjectionMatrix?: Float64Array | Float32Array | number[]
          projectionMatrix?: Float64Array | Float32Array | number[]
          shaderData?: ShaderData
          defaultProjectionData?: ProjectionData
        }
  ) {
    if (this.isRemoved || !this.renderer || !this.gl || !this.tileCache) return

    const paramsObj =
      params && typeof params === 'object' && !Array.isArray(params)
        ? (params as any)
        : null

    const shaderData = paramsObj?.shaderData
    const isGlobe = this.isGlobeProjection(shaderData)
    this.updateGeometryForProjection(isGlobe)
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

    const visibleTiles = this.isMultiscale ? this.getVisibleTiles() : []

    this.renderer.render({
      matrix,
      colormapTexture,
      uniforms,
      worldOffsets,
      isMultiscale: this.isMultiscale,
      visibleTiles,
      tileCache: this.tileCache,
      tileSize: this.tileSize,
      vertexArr: this.vertexArr,
      pixCoordArr: this.pixCoordArr,
      singleImage: this.isMultiscale
        ? undefined
        : {
            data: this.singleImageData,
            width: this.singleImageWidth,
            height: this.singleImageHeight,
            bounds: this.mercatorBounds,
            texture: this.singleImageTexture,
            vertexBuffer: this.singleImageVertexBuffer,
            pixCoordBuffer: this.singleImagePixCoordBuffer,
            pixCoordArr: this.singleImagePixCoordArr,
          },
      shaderData,
      projectionData,
    })
  }

  onRemove(_map: any, gl: WebGL2RenderingContext) {
    this.isRemoved = true

    this.renderer?.dispose()
    this.renderer = null

    this.colormap.dispose(gl)

    this.tileCache?.clear()
    this.tileCache = null

    if (this.singleImageTexture) {
      gl.deleteTexture(this.singleImageTexture)
      this.singleImageTexture = null
    }
    if (this.singleImageVertexBuffer) {
      gl.deleteBuffer(this.singleImageVertexBuffer)
      this.singleImageVertexBuffer = null
    }
    if (this.singleImagePixCoordBuffer) {
      gl.deleteBuffer(this.singleImagePixCoordBuffer)
      this.singleImagePixCoordBuffer = null
    }

    if (this.zarrStore) {
      this.zarrStore.cleanup()
      this.zarrStore = null
    }
    this.singleImageData = null

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
