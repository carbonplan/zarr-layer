import * as zarr from 'zarrita'
import { DataManager, RenderData } from './data-manager'
import { ZarrStore } from './zarr-store'
import { boundsToMercatorNorm, MercatorBounds } from './map-utils'
import { mustCreateBuffer, mustCreateTexture } from './webgl-utils'
import type {
  CRS,
  DimIndicesProps,
  LoadingStateCallback,
  MapLike,
  XYLimits,
  ZarrSelectorsProps,
} from './types'

const TILE_SUBDIVISIONS = 16

export class SingleImageDataManager implements DataManager {
  isMultiscale: false = false
  private data: Float32Array | null = null
  private width: number = 0
  private height: number = 0
  private texture: WebGLTexture | null = null
  private vertexBuffer: WebGLBuffer | null = null
  private pixCoordBuffer: WebGLBuffer | null = null

  private vertexArr: Float32Array = new Float32Array()
  private pixCoordArr: Float32Array = new Float32Array()
  private currentSubdivisions: number = 0
  private geometryVersion: number = 0

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
  private fetchRequestId: number = 0

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
      console.warn('SingleImageDataManager: No XY limits found')
    }

    // Initialize with standard quad (will be updated by onProjectionChange usually)
    this.updateGeometryForProjection(false)
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
    const isGlobe =
      projection?.type === 'globe' || projection?.name === 'globe'
    this.updateGeometryForProjection(isGlobe)

    if (!this.data && !this.isLoadingData) {
      this.fetchData().then(() => {
        this.invalidate()
      })
    }
  }

  onProjectionChange(isGlobe: boolean): void {
    this.updateGeometryForProjection(isGlobe)
  }

  private updateGeometryForProjection(isGlobe: boolean) {
    const targetSubdivisions = isGlobe ? TILE_SUBDIVISIONS : 1
    if (this.currentSubdivisions === targetSubdivisions) return

    const subdivided =
      SingleImageDataManager.createSubdividedQuad(targetSubdivisions)
    this.vertexArr = subdivided.vertexArr
    this.pixCoordArr = subdivided.texCoordArr
    this.currentSubdivisions = targetSubdivisions
    this.geometryVersion += 1
    this.invalidate()

    // We rely on ZarrLayer/Renderer to handle buffer update signalling via resetSingleImageGeometry()
    // But since SingleImageDataManager doesn't own the renderer, it just provides updated arrays.
  }

  private static createSubdividedQuad(subdivisions: number): {
    vertexArr: Float32Array
    texCoordArr: Float32Array
  } {
    const vertices: number[] = []
    const texCoords: number[] = []
    const step = 2 / subdivisions
    const texStep = 1 / subdivisions

    const pushVertex = (col: number, row: number) => {
      const x = -1 + col * step
      const y = 1 - row * step
      const u = col * texStep
      const v = row * texStep
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

  getRenderData(): RenderData {
    return {
      isMultiscale: false,
      vertexArr: this.vertexArr,
      pixCoordArr: this.pixCoordArr,
      singleImage: {
        data: this.data,
        width: this.width,
        height: this.height,
      bounds: this.mercatorBounds,
      texture: this.texture,
      vertexBuffer: this.vertexBuffer,
      pixCoordBuffer: this.pixCoordBuffer,
      pixCoordArr: this.pixCoordArr,
      geometryVersion: this.geometryVersion,
    },
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

  private emitLoadingState(): void {
    if (!this.loadingCallback) return
    this.loadingCallback({
      loading: this.isLoadingData,
      metadata: false,
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

  private async fetchData(): Promise<void> {
    if (!this.zarrArray || this.isRemoved) return

    const requestId = ++this.fetchRequestId
    const selectorSnapshot = { ...this.selector }
    this.isLoadingData = true
    this.emitLoadingState()

    try {
      const sliceArgs: (number | zarr.Slice)[] = new Array(
        this.zarrArray.shape.length
      ).fill(0)

      for (const dimName of Object.keys(this.dimIndices)) {
        const dimInfo = this.dimIndices[dimName]
        if (dimName === 'lon') {
          sliceArgs[dimInfo.index] = zarr.slice(0, this.width)
        } else if (dimName === 'lat') {
          sliceArgs[dimInfo.index] = zarr.slice(0, this.height)
        } else {
          const dimSelection = selectorSnapshot[dimName] as
            | number
            | number[]
            | string
            | string[]
            | ZarrSelectorsProps
            | undefined
          if (dimSelection !== undefined) {
            const selectionValue =
              typeof dimSelection === 'object' &&
              dimSelection !== null &&
              !Array.isArray(dimSelection) &&
              'selected' in dimSelection
                ? dimSelection.selected
                : dimSelection
            const normalizedValue = Array.isArray(selectionValue)
              ? selectionValue.find((v) => typeof v === 'number') ?? 0
              : typeof selectionValue === 'number'
              ? selectionValue
              : 0
            sliceArgs[dimInfo.index] = normalizedValue
          } else {
            sliceArgs[dimInfo.index] = 0
          }
        }
      }

      const data = (await zarr.get(this.zarrArray, sliceArgs)) as {
        data: ArrayLike<number>
      }
      if (this.isRemoved || requestId !== this.fetchRequestId) return

      this.data = new Float32Array((data.data as Float32Array).buffer)
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
}
