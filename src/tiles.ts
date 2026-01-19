import * as zarr from 'zarrita'
import { tileToKey, type TileTuple } from './map-utils'
import type {
  DimIndicesProps,
  NormalizedSelector,
  SelectorSpec,
  SelectorValue,
} from './types'
import { ZarrStore } from './zarr-store'
import { resolveSelectorValue } from './zarr-utils'
import {
  configureDataTexture,
  getTextureFormats,
  interleaveBands,
  mustCreateBuffer,
  mustCreateTexture,
  normalizeDataForTexture,
} from './webgl-utils'

/**
 * Tile cache entry containing raw data and WebGL resources.
 */
export interface TileData {
  // Raw chunk data
  chunkData: Float32Array | null
  chunkShape: number[] | null
  chunkIndices?: number[]
  data: Float32Array | null
  bandData: Map<string, Float32Array>
  channels: number
  selectorHash: string | null
  selectorVersion: number
  loading: boolean

  // Data normalization (for half-float precision on mobile GPUs)
  dataScale: number // Scale factor applied to data (1.0 = no normalization)
  bandDataScales: Map<string, number> // Scale factors per band

  // Geographic bounds for fragment shader reprojection (EPSG:4326 only)
  latBounds: { min: number; max: number } | null
  geoBounds: { west: number; south: number; east: number; north: number } | null
  mercatorBounds: { x0: number; y0: number; x1: number; y1: number } | null

  // WebGL resources
  tileTexture: WebGLTexture | null
  bandTextures: Map<string, WebGLTexture>
  bandTexturesUploaded: Set<string>
  bandTexturesConfigured: Set<string>
  textureUploaded: boolean
  textureConfigured: boolean
  vertexBuffer: WebGLBuffer | null
  pixCoordBuffer: WebGLBuffer | null
  geometryUploaded: boolean
}

interface TilesOptions {
  store: ZarrStore
  selector: NormalizedSelector
  fillValue: number
  clim?: [number, number]
  dimIndices: DimIndicesProps
  coordinates: Record<string, (string | number)[]>
  maxCachedTiles?: number
  bandNames?: string[]
  crs?: 'EPSG:4326' | 'EPSG:3857'
}

/**
 * Tile cache managing raw Zarr chunk data, extracted slices, and WebGL resources.
 * Uses Map's insertion-order iteration for O(1) LRU tracking.
 */
export class Tiles {
  private store: ZarrStore
  private selector: NormalizedSelector
  private fillValue: number
  private clim: [number, number]
  private dimIndices: DimIndicesProps
  private coordinates: Record<string, (string | number)[]>
  private maxCachedTiles: number
  private tiles: Map<string, TileData> = new Map()
  private bandNames: string[]
  private gl: WebGL2RenderingContext | null = null

  constructor({
    store,
    selector,
    fillValue,
    clim = [0, 1],
    dimIndices,
    coordinates,
    maxCachedTiles = 64,
    bandNames = [],
  }: TilesOptions) {
    this.store = store
    this.selector = selector
    this.fillValue = fillValue
    this.clim = clim
    this.dimIndices = dimIndices
    this.coordinates = coordinates
    this.maxCachedTiles = maxCachedTiles
    this.bandNames = bandNames
  }

  /**
   * Initialize WebGL resources. Must be called before rendering.
   */
  setGL(gl: WebGL2RenderingContext) {
    this.gl = gl
  }

  updateBandNames(bandNames: string[]) {
    this.bandNames = bandNames
  }

  updateSelector(selector: NormalizedSelector) {
    this.selector = selector
  }

  updateClim(clim: [number, number]) {
    this.clim = clim
  }

  private getDimKeyForName(dimName: string): string {
    const lower = dimName.toLowerCase()
    if (['lat', 'latitude', 'y'].includes(lower)) return 'lat'
    if (['lon', 'longitude', 'x', 'lng'].includes(lower)) return 'lon'
    if (['time', 't', 'time_counter'].includes(lower)) return 'time'
    if (['depth', 'z', 'level', 'lev', 'elevation'].includes(lower))
      return 'elevation'
    return dimName
  }

  private arraysEqual(a: number[] | undefined, b: number[] | undefined) {
    if (!a || !b) return false
    if (a.length !== b.length) return false
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) return false
    }
    return true
  }

  private normalizeSelection(
    dimSelection: SelectorSpec | SelectorValue | undefined,
    dimName?: string
  ): number[] {
    if (dimSelection === undefined) return [0]

    const coords = dimName ? this.coordinates[dimName] : undefined

    const toIndices = (value: SelectorSpec | SelectorValue): number => {
      const isSpec =
        typeof value === 'object' &&
        value !== null &&
        !Array.isArray(value) &&
        'selected' in value
      const selected = isSpec ? (value as SelectorSpec).selected : value
      const mode =
        isSpec && (value as SelectorSpec).type
          ? (value as SelectorSpec).type
          : 'value'

      if (
        mode !== 'index' &&
        coords &&
        (typeof selected === 'number' || typeof selected === 'string')
      ) {
        const idx = coords.indexOf(selected)
        if (idx >= 0) return idx
      }
      return typeof selected === 'number' ? selected : 0
    }

    if (
      typeof dimSelection === 'object' &&
      dimSelection !== null &&
      !Array.isArray(dimSelection) &&
      'selected' in dimSelection
    ) {
      const values = Array.isArray(dimSelection.selected)
        ? dimSelection.selected
        : [dimSelection.selected]
      return values.map((v) =>
        toIndices({ selected: v, type: dimSelection.type })
      )
    }

    if (Array.isArray(dimSelection)) {
      return dimSelection.map((v) => toIndices(v))
    }

    return [toIndices(dimSelection)]
  }

  /**
   * Compute which chunk indices to fetch for a given tile.
   */
  private computeChunkIndices(
    levelArray: zarr.Array<zarr.DataType>,
    tileTuple: TileTuple
  ): number[] {
    const [_, x, y] = tileTuple
    const dimensions = this.store.dimensions || []
    const chunks = levelArray.chunks
    const chunkIndices: number[] = new Array(dimensions.length).fill(0)

    for (let i = 0; i < dimensions.length; i++) {
      const dimName = dimensions[i]
      const dimKey = this.getDimKeyForName(dimName)

      if (dimKey === 'lon') {
        chunkIndices[i] = x
      } else if (dimKey === 'lat') {
        chunkIndices[i] = y
      } else {
        const dimSelection = resolveSelectorValue(
          this.selector,
          dimKey,
          dimName,
          this.dimIndices
        )

        const selectionValues = this.normalizeSelection(dimSelection, dimName)

        const normalized = selectionValues.map((v) =>
          Math.max(0, Math.min(v, levelArray.shape[i] - 1))
        )
        const chunkIdx = Math.floor(normalized[0] / chunks[i])
        const spansMultipleChunks = normalized.some(
          (v) => Math.floor(v / chunks[i]) !== chunkIdx
        )
        if (spansMultipleChunks) {
          console.warn(
            `Selector for dimension '${dimName}' spans multiple chunks – using chunk index ${chunkIdx} for tile ${tileTuple.join(
              ','
            )}`
          )
        }
        const maxChunkIdx = Math.max(
          0,
          Math.ceil(levelArray.shape[i] / chunks[i]) - 1
        )
        chunkIndices[i] = Math.min(chunkIdx, maxChunkIdx)
      }
    }

    return chunkIndices
  }

  /**
   * Extract a 2D slice (+ optional extra channels) from a loaded chunk.
   * Returns band-separate format only; interleaving is done later if needed.
   */
  private extractSliceFromChunk(
    chunkData: Float32Array,
    chunkShape: number[],
    levelArray: zarr.Array<zarr.DataType>,
    chunkIndices: number[]
  ): {
    channels: number
    bandData: Map<string, Float32Array>
  } {
    const tileWidth = this.store.tileSize
    const tileHeight = this.store.tileSize
    let channels = 1

    const dimensions = this.store.dimensions || []
    const chunkSizes = levelArray.chunks

    const selectorIndices: number[] = []
    let latDimIdx = -1
    let lonDimIdx = -1
    let latSize = tileHeight
    let lonSize = tileWidth

    const selectionSets: number[][] = []
    const varyingDims: number[] = []

    for (let i = 0; i < dimensions.length; i++) {
      const dimName = dimensions[i]
      const dimKey = this.getDimKeyForName(dimName)

      if (dimKey === 'lat') {
        latDimIdx = i
        latSize = Math.min(chunkShape[i], tileHeight)
        selectorIndices.push(-1)
      } else if (dimKey === 'lon') {
        lonDimIdx = i
        lonSize = Math.min(chunkShape[i], tileWidth)
        selectorIndices.push(-1)
      } else {
        const dimSelection = resolveSelectorValue(
          this.selector,
          dimKey,
          dimName,
          this.dimIndices
        )

        const selectedValues = this.normalizeSelection(dimSelection, dimName)

        const chunkOffset = chunkIndices[i] * chunkSizes[i]
        const withinChunk = selectedValues.map((v) => {
          const adjusted = Math.max(
            0,
            Math.min(v - chunkOffset, chunkShape[i] - 1)
          )
          return adjusted
        })

        selectionSets[i] = withinChunk
        selectorIndices.push(withinChunk[0])
        if (withinChunk.length > 1) {
          varyingDims.push(i)
        }
      }
    }

    const getChunkIndex = (indices: number[]): number => {
      let idx = 0
      let stride = 1
      for (let i = indices.length - 1; i >= 0; i--) {
        idx += indices[i] * stride
        stride *= chunkShape[i]
      }
      return idx
    }

    let channelSelections: number[][] = [[]]
    varyingDims.forEach((dimIdx) => {
      const choices = selectionSets[dimIdx]
      const next: number[][] = []
      channelSelections.forEach((combo) => {
        choices.forEach((choice) => {
          next.push([...combo, choice])
        })
      })
      channelSelections = next
    })
    channels = channelSelections.length || 1

    // Create only band-separate arrays (interleaving done later if needed)
    const bandData = new Map<string, Float32Array>()
    const bandArrays: Float32Array[] = []
    for (let c = 0; c < channels; c++) {
      const arr = new Float32Array(tileWidth * tileHeight)
      arr.fill(this.fillValue)
      bandArrays.push(arr)
    }

    for (let latIdx = 0; latIdx < latSize; latIdx++) {
      for (let lonIdx = 0; lonIdx < lonSize; lonIdx++) {
        if (latDimIdx >= 0) selectorIndices[latDimIdx] = latIdx
        if (lonDimIdx >= 0) selectorIndices[lonDimIdx] = lonIdx

        channelSelections.forEach((selectionCombo, channelIdx) => {
          const indices = [...selectorIndices]
          let comboIdx = 0
          for (let i = 0; i < varyingDims.length; i++) {
            indices[varyingDims[i]] = selectionCombo[comboIdx++]
          }

          const srcIdx = getChunkIndex(indices)
          const dstIdx = latIdx * tileWidth + lonIdx

          if (srcIdx < chunkData.length) {
            bandArrays[channelIdx][dstIdx] = chunkData[srcIdx]
          }
        })
      }
    }

    for (let c = 0; c < channels; c++) {
      const bandName = this.bandNames[c] || `band_${c}`
      bandData.set(bandName, bandArrays[c])
    }

    return { channels, bandData }
  }

  /**
   * Apply normalization to tile data and upload texture.
   */
  private applyNormalization(
    tile: TileData,
    sliced: {
      channels: number
      bandData: Map<string, Float32Array>
    }
  ): void {
    const bandDataToProcess = sliced.bandData

    // Normalize bands (single pass) and collect for interleaving
    tile.bandData = new Map()
    tile.bandDataScales = new Map()
    tile.bandTexturesUploaded.clear()
    const normalizedBands: Float32Array[] = []

    for (const [bandName, bandData] of bandDataToProcess) {
      const { normalized, scale } = normalizeDataForTexture(
        bandData,
        this.fillValue,
        this.clim
      )
      tile.bandData.set(bandName, normalized)
      tile.bandDataScales.set(bandName, scale)
      normalizedBands.push(normalized)
    }

    // Construct interleaved data from normalized bands
    tile.data = interleaveBands(normalizedBands, sliced.channels)
    tile.dataScale = tile.bandDataScales.values().next().value ?? 1.0
    tile.channels = sliced.channels

    // Upload texture
    if (this.gl && tile.tileTexture) {
      this.uploadTileTexture(tile)
    }
  }

  /**
   * Upload tile data to its texture.
   */
  private uploadTileTexture(tile: TileData): void {
    if (!this.gl || !tile.tileTexture || !tile.data) return

    const gl = this.gl
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, tile.tileTexture)

    if (!tile.textureConfigured) {
      configureDataTexture(gl)
      tile.textureConfigured = true
    }

    const { format, internalFormat } = getTextureFormats(gl, tile.channels)
    const tileSize = this.store.tileSize
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      internalFormat,
      tileSize,
      tileSize,
      0,
      format,
      gl.FLOAT,
      tile.data
    )
    tile.textureUploaded = true
  }

  /**
   * Get or create a tile entry, using Map's insertion order for LRU tracking.
   * O(1) for both access and eviction.
   */
  private getOrCreateTile(tileKey: string): TileData {
    let tile = this.tiles.get(tileKey)

    if (tile) {
      // Move to end of Map iteration order (most recently used)
      this.tiles.delete(tileKey)
      this.tiles.set(tileKey, tile)
      return tile
    }

    // Create new tile entry with WebGL resources if GL context available
    tile = {
      chunkData: null,
      chunkShape: null,
      chunkIndices: undefined,
      data: null,
      bandData: new Map(),
      channels: 1,
      selectorHash: null,
      selectorVersion: 0,
      loading: false,
      dataScale: 1.0,
      bandDataScales: new Map(),
      latBounds: null,
      geoBounds: null,
      mercatorBounds: null,
      tileTexture: this.gl ? mustCreateTexture(this.gl) : null,
      bandTextures: new Map(),
      bandTexturesUploaded: new Set(),
      bandTexturesConfigured: new Set(),
      textureUploaded: false,
      textureConfigured: false,
      vertexBuffer: this.gl ? mustCreateBuffer(this.gl) : null,
      pixCoordBuffer: this.gl ? mustCreateBuffer(this.gl) : null,
      geometryUploaded: false,
    }
    this.tiles.set(tileKey, tile)
    this.evictOldTiles()

    return tile
  }

  /**
   * Evict oldest tiles when cache exceeds limit.
   * Uses Map iteration order (oldest first).
   */
  private evictOldTiles() {
    while (this.tiles.size > this.maxCachedTiles) {
      // Get first key (oldest entry due to Map insertion order)
      const oldestKey = this.tiles.keys().next().value
      if (!oldestKey) break

      const tile = this.tiles.get(oldestKey)
      if (tile && this.gl) {
        // Clean up WebGL resources
        if (tile.tileTexture) this.gl.deleteTexture(tile.tileTexture)
        for (const tex of tile.bandTextures.values()) {
          this.gl.deleteTexture(tex)
        }
        if (tile.vertexBuffer) this.gl.deleteBuffer(tile.vertexBuffer)
        if (tile.pixCoordBuffer) this.gl.deleteBuffer(tile.pixCoordBuffer)
      }
      this.tiles.delete(oldestKey)
    }
  }

  /**
   * Get a tile from the cache. Returns undefined if not found.
   */
  get(tileKey: string): TileData | undefined {
    const tile = this.tiles.get(tileKey)
    if (tile) {
      // Update LRU order
      this.tiles.delete(tileKey)
      this.tiles.set(tileKey, tile)
    }
    return tile
  }

  /**
   * Get or create a tile entry. Creates WebGL resources if not present.
   */
  upsert(tileKey: string): TileData {
    return this.getOrCreateTile(tileKey)
  }

  /**
   * Ensure a band texture exists for a tile.
   */
  ensureBandTexture(tileKey: string, bandName: string): WebGLTexture | null {
    const tile = this.tiles.get(tileKey)
    if (!tile || !this.gl) return null

    let tex = tile.bandTextures.get(bandName)
    if (!tex) {
      tex = mustCreateTexture(this.gl)
      tile.bandTextures.set(bandName, tex)
    }
    return tex
  }

  getTile(tileTuple: TileTuple): TileData | undefined {
    return this.tiles.get(tileToKey(tileTuple))
  }

  /**
   * Set bounds for a tile (used for fragment shader reprojection in EPSG:4326 mode).
   */
  setTileBounds(
    tileKey: string,
    bounds: {
      latMin: number
      latMax: number
      lonMin: number
      lonMax: number
      x0: number
      y0: number
      x1: number
      y1: number
    }
  ): void {
    const tile = this.tiles.get(tileKey)
    if (!tile) return

    const latBounds = { min: bounds.latMin, max: bounds.latMax }
    const geoBounds = {
      west: bounds.lonMin,
      south: bounds.latMin,
      east: bounds.lonMax,
      north: bounds.latMax,
    }
    const mercatorBounds = {
      x0: bounds.x0,
      y0: bounds.y0,
      x1: bounds.x1,
      y1: bounds.y1,
    }

    // Check if any bounds changed
    const boundsChanged =
      tile.latBounds?.min !== latBounds.min ||
      tile.latBounds?.max !== latBounds.max ||
      tile.geoBounds?.west !== geoBounds.west ||
      tile.geoBounds?.east !== geoBounds.east ||
      tile.mercatorBounds?.x0 !== mercatorBounds.x0 ||
      tile.mercatorBounds?.x1 !== mercatorBounds.x1 ||
      tile.mercatorBounds?.y0 !== mercatorBounds.y0 ||
      tile.mercatorBounds?.y1 !== mercatorBounds.y1

    if (boundsChanged) {
      tile.latBounds = latBounds
      tile.geoBounds = geoBounds
      tile.mercatorBounds = mercatorBounds
    }
  }

  async reextractTileSlices(
    visibleTiles: TileTuple[],
    selectorHash: string,
    version: number
  ): Promise<void> {
    for (const tileTuple of visibleTiles) {
      const tileKey = tileToKey(tileTuple)
      const tile = this.tiles.get(tileKey)
      if (!tile) continue
      const levelPath = this.store.levels[tileTuple[0]]
      if (!levelPath) continue
      const levelArray = await this.store.getLevelArray(levelPath)
      const desiredChunkIndices = this.computeChunkIndices(
        levelArray,
        tileTuple
      )
      const canReuseChunk =
        tile.chunkData &&
        tile.chunkShape &&
        this.arraysEqual(tile.chunkIndices, desiredChunkIndices)
      if (canReuseChunk) {
        // Only update if this version is newer than what's already rendered
        if (version < tile.selectorVersion) continue
        const sliced = this.extractSliceFromChunk(
          tile.chunkData!,
          tile.chunkShape!,
          levelArray,
          desiredChunkIndices
        )
        this.applyNormalization(tile, sliced)
        tile.selectorHash = selectorHash
        tile.selectorVersion = version
      } else {
        // Keep old data visible while new chunk loads - don't clear tile.data
        // The stale selectorHash ensures a new fetch will be triggered,
        // and the old data continues to render until replaced
        tile.selectorHash = null
        tile.chunkData = null
        tile.chunkShape = null
        tile.chunkIndices = undefined
      }
    }
  }

  async fetchTile(
    tileTuple: TileTuple,
    selectorHash: string,
    version: number,
    signal?: AbortSignal,
    bounds?: {
      latMin: number
      latMax: number
      lonMin: number
      lonMax: number
      x0: number
      y0: number
      x1: number
      y1: number
    }
  ): Promise<TileData | null> {
    const [z] = tileTuple
    const levelPath = this.store.levels[z]
    if (!levelPath) return null

    const levelArray = await this.store.getLevelArray(levelPath)
    const tileKey = tileToKey(tileTuple)
    const tile = this.getOrCreateTile(tileKey)

    // Store bounds if provided (for EPSG:4326 resampling)
    if (bounds) {
      tile.latBounds = { min: bounds.latMin, max: bounds.latMax }
      tile.geoBounds = {
        west: bounds.lonMin,
        south: bounds.latMin,
        east: bounds.lonMax,
        north: bounds.latMax,
      }
      tile.mercatorBounds = {
        x0: bounds.x0,
        y0: bounds.y0,
        x1: bounds.x1,
        y1: bounds.y1,
      }
    }

    if (tile.data && tile.selectorHash === selectorHash) {
      return tile
    }
    if (tile.loading) return null

    tile.loading = true

    try {
      const chunkIndices = this.computeChunkIndices(levelArray, tileTuple)
      const canReuseChunk =
        tile.chunkData &&
        tile.chunkShape &&
        this.arraysEqual(tile.chunkIndices, chunkIndices)

      if (canReuseChunk) {
        // Only update if this version is newer than what's already rendered
        if (version < tile.selectorVersion) {
          tile.loading = false
          return null
        }
        const sliced = this.extractSliceFromChunk(
          tile.chunkData!,
          tile.chunkShape!,
          levelArray,
          chunkIndices
        )
        this.applyNormalization(tile, sliced)
        tile.selectorHash = selectorHash
        tile.selectorVersion = version
        tile.loading = false
        return tile
      }

      const chunk = await this.store.getChunk(levelPath, chunkIndices, {
        signal,
      })
      const chunkShape = (chunk.shape as number[]).map((n) => Number(n))
      const chunkData =
        chunk.data instanceof Float32Array
          ? new Float32Array(chunk.data.buffer)
          : Float32Array.from(chunk.data as ArrayLike<number>)

      // Only update if this version is newer than what's already rendered
      if (version < tile.selectorVersion) {
        tile.loading = false
        return null
      }

      tile.chunkData = chunkData
      tile.chunkShape = chunkShape
      tile.chunkIndices = chunkIndices
      const sliced = this.extractSliceFromChunk(
        chunkData,
        chunkShape,
        levelArray,
        chunkIndices
      )
      this.applyNormalization(tile, sliced)
      tile.selectorHash = selectorHash
      tile.selectorVersion = version
      tile.loading = false
      return tile
    } catch (err) {
      tile.loading = false
      // AbortError is expected when requests are cancelled - don't log as error
      if (err instanceof DOMException && err.name === 'AbortError') {
        return null
      }
      console.error('Error fetching tile data:', err)
      return null
    }
  }

  /**
   * Mark all tile geometry as needing re-upload (e.g., after subdivision change).
   */
  markGeometryDirty() {
    for (const tile of this.tiles.values()) {
      tile.geometryUploaded = false
    }
  }

  /**
   * Clear all tiles and release WebGL resources.
   */
  clear() {
    if (this.gl) {
      for (const tile of this.tiles.values()) {
        if (tile.tileTexture) this.gl.deleteTexture(tile.tileTexture)
        for (const tex of tile.bandTextures.values()) {
          this.gl.deleteTexture(tex)
        }
        if (tile.vertexBuffer) this.gl.deleteBuffer(tile.vertexBuffer)
        if (tile.pixCoordBuffer) this.gl.deleteBuffer(tile.pixCoordBuffer)
      }
    }
    this.tiles.clear()
  }
}
