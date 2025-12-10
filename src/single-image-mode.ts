import * as zarr from 'zarrita'
import type {
  ZarrMode,
  RenderContext,
  TileId,
  SingleImageRenderState,
} from './zarr-mode'
import type {
  QuerySelector,
  QueryDataGeometry,
  QueryDataResult,
} from './query/types'
import { queryRegionSingleImage } from './query/region-query'
import { mercatorBoundsToPixel } from './query/query-utils'
import { ZarrStore } from './zarr-store'
import {
  boundsToMercatorNorm,
  MercatorBounds,
  type XYLimits as MapXYLimits,
} from './map-utils'
import {
  mustCreateBuffer,
  mustCreateTexture,
  createSubdividedQuad,
} from './webgl-utils'
import type {
  CRS,
  DimIndicesProps,
  LoadingStateCallback,
  MapLike,
  XYLimits,
  ZarrSelectorsProps,
} from './types'
import { calculateNearestIndex, loadDimensionValues } from './zarr-utils'
import { TILE_SUBDIVISIONS } from './constants'
import type { ZarrRenderer } from './zarr-renderer'
import { renderMapboxTile } from './mapbox-globe-tile-renderer'

export class SingleImageMode implements ZarrMode {
  isMultiscale: false = false
  private data: Float32Array | null = null
  private width: number = 0
  private height: number = 0
  private channels: number = 1
  private channelLabels: (string | number)[][] = []
  private multiValueDimNames: string[] = []
  private texture: WebGLTexture | null = null
  private vertexBuffer: WebGLBuffer | null = null
  private pixCoordBuffer: WebGLBuffer | null = null

  private vertexArr: Float32Array = new Float32Array()
  private pixCoordArr: Float32Array = new Float32Array()
  private currentSubdivisions: number = 0
  private geometryVersion: number = 0
  private dataVersion: number = 0

  private mercatorBounds: MercatorBounds | null = null
  private zarrStore: ZarrStore
  private variable: string
  private selector: Record<
    string,
    number | number[] | string | string[] | ZarrSelectorsProps
  >
  private invalidate: () => void
  private dimIndices: DimIndicesProps = {}
  private xyLimits: XYLimits | null = null
  private crs: CRS | null = null
  private zarrArray: zarr.Array<zarr.DataType> | null = null
  private isRemoved: boolean = false
  private loadingCallback: LoadingStateCallback | undefined
  private isLoadingData: boolean = false
  private metadataLoading: boolean = false
  private fetchRequestId: number = 0
  private dimensionValues: { [key: string]: Float64Array | number[] } = {}

  constructor(
    store: ZarrStore,
    variable: string,
    selector: Record<
      string,
      number | number[] | string | string[] | ZarrSelectorsProps
    >,
    invalidate: () => void
  ) {
    this.zarrStore = store
    this.variable = variable
    this.selector = selector
    this.invalidate = invalidate
  }

  async initialize(): Promise<void> {
    this.metadataLoading = true
    this.emitLoadingState()

    try {
      const desc = this.zarrStore.describe()
      this.dimIndices = desc.dimIndices
      this.crs = desc.crs
      this.xyLimits = desc.xyLimits

      this.zarrArray = await this.zarrStore.getArray()
      this.width = this.zarrArray.shape[this.dimIndices.lon.index]
      this.height = this.zarrArray.shape[this.dimIndices.lat.index]

      if (this.xyLimits) {
        this.mercatorBounds = boundsToMercatorNorm(this.xyLimits, this.crs)
      } else {
        console.warn('SingleImageMode: No XY limits found')
      }

      this.updateGeometryForProjection(false)
    } finally {
      this.metadataLoading = false
      this.emitLoadingState()
    }
  }

  update(map: MapLike, gl: WebGL2RenderingContext): void {
    if (!this.texture) {
      this.texture = mustCreateTexture(gl)
    }
    if (!this.vertexBuffer) {
      this.vertexBuffer = mustCreateBuffer(gl)
    }
    if (!this.pixCoordBuffer) {
      this.pixCoordBuffer = mustCreateBuffer(gl)
    }

    const projection = map.getProjection ? map.getProjection() : null
    const isGlobe = projection?.type === 'globe' || projection?.name === 'globe'
    this.updateGeometryForProjection(isGlobe)

    if (!this.data && !this.isLoadingData) {
      this.fetchData().then(() => {
        this.invalidate()
      })
    }
  }

  render(renderer: ZarrRenderer, context: RenderContext): void {
    const singleImageState = this.getSingleImageState()
    if (!singleImageState) return

    const isMapboxTile = !!context.mapboxGlobe
    const shaderProgram = renderer.getProgram(
      context.shaderData,
      context.customShaderConfig,
      isMapboxTile
    )

    renderer.gl.useProgram(shaderProgram.program)

    renderer.applyCommonUniforms(
      shaderProgram,
      context.colormapTexture,
      context.uniforms,
      context.customShaderConfig,
      context.projectionData,
      context.mapboxGlobe,
      context.matrix,
      false
    )

    renderer.renderSingleImage(
      shaderProgram,
      context.worldOffsets,
      singleImageState.singleImage,
      singleImageState.vertexArr
    )
  }

  renderToTile(
    renderer: ZarrRenderer,
    tileId: TileId,
    context: RenderContext
  ): boolean {
    return renderMapboxTile({
      renderer,
      mode: this,
      tileId,
      context,
    })
  }

  onProjectionChange(isGlobe: boolean): void {
    this.updateGeometryForProjection(isGlobe)
  }

  getTiledState() {
    return null
  }

  getSingleImageState(): SingleImageRenderState | null {
    if (!this.texture || !this.vertexBuffer || !this.pixCoordBuffer) {
      return null
    }
    return {
      singleImage: {
        data: this.data,
        width: this.width,
        height: this.height,
        channels: this.channels,
        bounds: this.mercatorBounds,
        texture: this.texture,
        vertexBuffer: this.vertexBuffer,
        pixCoordBuffer: this.pixCoordBuffer,
        pixCoordArr: this.pixCoordArr,
        geometryVersion: this.geometryVersion,
        dataVersion: this.dataVersion,
      },
      vertexArr: this.vertexArr,
    }
  }

  dispose(gl: WebGL2RenderingContext): void {
    this.isRemoved = true
    if (this.texture) gl.deleteTexture(this.texture)
    if (this.vertexBuffer) gl.deleteBuffer(this.vertexBuffer)
    if (this.pixCoordBuffer) gl.deleteBuffer(this.pixCoordBuffer)
    this.texture = null
    this.vertexBuffer = null
    this.pixCoordBuffer = null
    this.data = null
    this.isLoadingData = false
    this.emitLoadingState()
  }

  setLoadingCallback(callback: LoadingStateCallback | undefined): void {
    this.loadingCallback = callback
  }

  getCRS(): CRS {
    return this.crs ?? 'EPSG:4326'
  }

  getXYLimits(): MapXYLimits | null {
    return this.xyLimits
  }

  getMaxZoom(): number {
    return 0
  }

  private emitLoadingState(): void {
    if (!this.loadingCallback) return
    this.loadingCallback({
      loading: this.metadataLoading || this.isLoadingData,
      metadata: this.metadataLoading,
      chunks: this.isLoadingData,
    })
  }

  async setSelector(
    selector: Record<
      string,
      number | number[] | string | string[] | ZarrSelectorsProps
    >
  ): Promise<void> {
    this.selector = selector
    await this.fetchData()
    this.invalidate()
  }

  private updateGeometryForProjection(isGlobe: boolean) {
    const targetSubdivisions = isGlobe ? TILE_SUBDIVISIONS : 1
    if (this.currentSubdivisions === targetSubdivisions) return

    const subdivided = createSubdividedQuad(targetSubdivisions)
    this.vertexArr = subdivided.vertexArr
    this.pixCoordArr = subdivided.texCoordArr
    this.currentSubdivisions = targetSubdivisions
    this.geometryVersion += 1
    this.invalidate()
  }

  private async fetchData(): Promise<void> {
    if (!this.zarrArray || this.isRemoved) return

    const requestId = ++this.fetchRequestId
    const selectorSnapshot = { ...this.selector }
    this.isLoadingData = true
    this.emitLoadingState()

    try {
      const baseSliceArgs: (number | zarr.Slice)[] = new Array(
        this.zarrArray.shape.length
      ).fill(0)

      const multiValueDims: Array<{
        dimIndex: number
        dimName: string
        values: number[]
        labels: (number | string)[]
      }> = []

      for (const dimName of Object.keys(this.dimIndices)) {
        const dimInfo = this.dimIndices[dimName]
        if (dimName === 'lon') {
          baseSliceArgs[dimInfo.index] = zarr.slice(0, this.width)
        } else if (dimName === 'lat') {
          baseSliceArgs[dimInfo.index] = zarr.slice(0, this.height)
        } else {
          const dimSelection = selectorSnapshot[dimName] as
            | number
            | number[]
            | string
            | string[]
            | ZarrSelectorsProps
            | undefined
          if (dimSelection !== undefined) {
            const isObj =
              typeof dimSelection === 'object' &&
              dimSelection !== null &&
              !Array.isArray(dimSelection) &&
              'selected' in dimSelection
            const selectionValue = isObj
              ? (dimSelection as ZarrSelectorsProps).selected
              : dimSelection
            const selectionType = isObj
              ? (dimSelection as ZarrSelectorsProps).type
              : undefined

            if (Array.isArray(selectionValue) && selectionValue.length > 1) {
              const resolvedIndices: number[] = []
              const labelValues: (number | string)[] = []
              for (const val of selectionValue) {
                const idx = await this.resolveSelectionIndex(
                  dimName,
                  dimInfo,
                  val as number | string,
                  selectionType
                )
                resolvedIndices.push(idx)
                labelValues.push(val as number | string)
              }
              multiValueDims.push({
                dimIndex: dimInfo.index,
                dimName,
                values: resolvedIndices,
                labels: labelValues,
              })
              baseSliceArgs[dimInfo.index] = resolvedIndices[0]
            } else {
              const primaryValue = Array.isArray(selectionValue)
                ? selectionValue[0]
                : selectionValue

              baseSliceArgs[dimInfo.index] = await this.resolveSelectionIndex(
                dimName,
                dimInfo,
                primaryValue as number | string,
                selectionType
              )
            }
          } else {
            baseSliceArgs[dimInfo.index] = 0
          }
        }
      }

      let channelCombinations: number[][] = [[]]
      let channelLabelCombinations: (number | string)[][] = [[]]
      for (const { values, labels } of multiValueDims) {
        const next: number[][] = []
        const nextLabels: (number | string)[][] = []
        for (let idx = 0; idx < values.length; idx++) {
          const val = values[idx]
          const label = labels[idx]
          for (let c = 0; c < channelCombinations.length; c++) {
            next.push([...channelCombinations[c], val])
            nextLabels.push([...channelLabelCombinations[c], label])
          }
        }
        channelCombinations = next
        channelLabelCombinations = nextLabels
      }

      const numChannels = channelCombinations.length || 1
      this.channels = numChannels
      this.multiValueDimNames = multiValueDims.map((d) => d.dimName)
      this.channelLabels = channelLabelCombinations

      if (numChannels === 1) {
        const data = (await zarr.get(this.zarrArray, baseSliceArgs)) as {
          data: ArrayLike<number>
        }
        if (this.isRemoved || requestId !== this.fetchRequestId) return
        this.data = new Float32Array((data.data as Float32Array).buffer)
        this.dataVersion++
      } else {
        const packedData = new Float32Array(
          this.width * this.height * numChannels
        )

        for (let c = 0; c < numChannels; c++) {
          const sliceArgs = [...baseSliceArgs]
          const combo = channelCombinations[c]

          for (let i = 0; i < multiValueDims.length; i++) {
            sliceArgs[multiValueDims[i].dimIndex] = combo[i]
          }

          const bandData = (await zarr.get(this.zarrArray, sliceArgs)) as {
            data: ArrayLike<number>
          }
          if (this.isRemoved || requestId !== this.fetchRequestId) return

          const bandArray = new Float32Array(
            (bandData.data as Float32Array).buffer
          )
          for (let pixIdx = 0; pixIdx < this.width * this.height; pixIdx++) {
            packedData[pixIdx * numChannels + c] = bandArray[pixIdx]
          }
        }

        this.data = packedData
        this.dataVersion++
      }

      this.invalidate()
    } catch (err) {
      console.error('Error fetching single image data:', err)
    } finally {
      if (requestId === this.fetchRequestId) {
        this.isLoadingData = false
        this.emitLoadingState()
      }
    }
  }

  private async resolveSelectionIndex(
    dimName: string,
    dimInfo: {
      index: number
      name: string
      array: zarr.Array<zarr.DataType> | null
    },
    value: number | string | [number, number] | undefined,
    type?: 'index' | 'value'
  ): Promise<number> {
    if (type === 'index') {
      return typeof value === 'number' ? value : 0
    }

    if (!this.zarrStore.root) {
      return typeof value === 'number' ? value : 0
    }

    try {
      const coords = await loadDimensionValues(
        this.dimensionValues,
        null,
        dimInfo,
        this.zarrStore.root,
        this.zarrStore.version
      )
      this.dimensionValues[dimName] = coords

      if (typeof value === 'number' || typeof value === 'string') {
        const coordIdx = (coords as (number | string)[]).indexOf(value)
        if (coordIdx >= 0) return coordIdx
        if (typeof value === 'number') {
          return calculateNearestIndex(coords, value)
        }
      }
    } catch {
      // fall through
    }

    return typeof value === 'number' ? value : 0
  }

  /**
   * Query data for point or region geometries.
   */
  async queryData(
    geometry: QueryDataGeometry,
    selector?: QuerySelector
  ): Promise<QueryDataResult> {
    if (!this.mercatorBounds) {
      return {
        [this.variable]: [],
        dimensions: [],
        coordinates: { lat: [], lon: [] },
      }
    }

    // Point geometries: sample single pixel and return region-shaped result
    if (geometry.type === 'Point') {
      const [lon, lat] = geometry.coordinates
      const coords = { lat: [lat], lon: [lon] }

      if (!this.data) {
        return {
          [this.variable]: [],
          dimensions: ['lat', 'lon'],
          coordinates: coords,
        }
      }

      const pixel = mercatorBoundsToPixel(
        lon,
        lat,
        this.mercatorBounds,
        this.width,
        this.height,
        this.crs ?? 'EPSG:4326'
      )

      if (!pixel) {
        return {
          [this.variable]: [],
          dimensions: ['lat', 'lon'],
          coordinates: coords,
        }
      }

      const { x, y } = pixel
      const baseIndex = (y * this.width + x) * this.channels
      const values: number[] = []

      for (let c = 0; c < this.channels; c++) {
        const value = this.data[baseIndex + c]
        if (value !== undefined && value !== null && Number.isFinite(value)) {
          values.push(value)
        }
      }

      return {
        [this.variable]: values,
        dimensions: ['lat', 'lon'],
        coordinates: coords,
      }
    }

    const desc = this.zarrStore.describe()
    const querySelector = selector || (this.selector as QuerySelector)

    return queryRegionSingleImage(
      this.variable,
      geometry,
      querySelector,
      this.data,
      this.width,
      this.height,
      this.mercatorBounds,
      this.crs ?? 'EPSG:4326',
      desc.dimensions,
      desc.coordinates,
      this.channels,
      this.channelLabels,
      this.multiValueDimNames
    )
  }
}
