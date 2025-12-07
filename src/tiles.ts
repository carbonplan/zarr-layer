import * as zarr from 'zarrita'
import { tileToKey, type TileTuple } from './map-utils'
import type { DimIndicesProps, SelectorMap, ZarrSelectorsProps } from './types'
import { ZarrStore } from './zarr-store'

export interface TileDataCache {
  chunkData: Float32Array | null
  chunkShape: number[] | null
  chunkIndices?: number[]
  data: Float32Array | null
  bandData: Map<string, Float32Array>
  channels: number
  selectorHash: string | null
  loading: boolean
  lastUsed: number
}

interface TilesOptions {
  store: ZarrStore
  selectors: SelectorMap
  fillValue: number
  dimIndices: DimIndicesProps
  coordinates: Record<string, (string | number)[]>
  maxCachedTiles?: number
  bandNames?: string[]
}

export class Tiles {
  private store: ZarrStore
  private selectors: SelectorMap
  private fillValue: number
  private dimIndices: DimIndicesProps
  private coordinates: Record<string, (string | number)[]>
  private maxCachedTiles: number
  private tiles: Map<string, TileDataCache> = new Map()
  private accessOrder: string[] = []
  private bandNames: string[]

  constructor({
    store,
    selectors,
    fillValue,
    dimIndices,
    coordinates,
    maxCachedTiles = 64,
    bandNames = [],
  }: TilesOptions) {
    this.store = store
    this.selectors = selectors
    this.fillValue = fillValue
    this.dimIndices = dimIndices
    this.coordinates = coordinates
    this.maxCachedTiles = maxCachedTiles
    this.bandNames = bandNames
  }

  updateBandNames(bandNames: string[]) {
    this.bandNames = bandNames
  }

  updateSelector(selectors: SelectorMap) {
    this.selectors = selectors
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
    dimSelection:
      | number
      | string
      | number[]
      | string[]
      | ZarrSelectorsProps
      | undefined,
    dimName?: string
  ): number[] {
    if (dimSelection === undefined) return [0]

    const coords = dimName ? this.coordinates[dimName] : undefined

    const toIndices = (
      value: number | string | [number, number],
      type?: 'index' | 'value'
    ): number => {
      if (
        type !== 'index' &&
        coords &&
        (typeof value === 'number' || typeof value === 'string')
      ) {
        const idx = coords.indexOf(value)
        if (idx >= 0) return idx
      }
      return typeof value === 'number' ? value : 0
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
      return values.map((v) => toIndices(v as number | string | [number, number], dimSelection.type))
    }

    if (Array.isArray(dimSelection)) {
      return dimSelection.map((v) => toIndices(v, undefined))
    }

    return [toIndices(dimSelection, undefined)]
  }

  /**
   * Compute which chunk indices to fetch for a given tile.
   *
   * Selectors can be in several formats:
   *   - Direct number: `this.selectors['band'] = 0`
   *   - Wrapped object with single value: `{ selected: 0, type: 'index' }`
   *   - Wrapped object with array (multi-band): `{ selected: [0, 1], type: 'index' }`
   *
   * For multi-band selectors like [0, 1], we use the first value's chunk index.
   * If bands span multiple chunks, a warning is logged and only one chunk is fetched.
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
        const dimSelection =
          this.selectors[dimKey] ??
          this.selectors[dimName] ??
          this.selectors[this.dimIndices[dimKey]?.name]

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
   *
   * When the selector includes multiple values for a non-spatial dimension
   * (e.g., `{ band: [0, 1] }`), this method packs them into separate channels
   * of the output texture:
   *   - 1 value  → R channel only (single band)
   *   - 2 values → R and G channels (e.g., tavg + prec)
   *   - 3 values → R, G, B channels
   *   - 4 values → R, G, B, A channels
   *
   * The fragment shader can then access these via texture(tex, coord).r, .g, etc.
   */

  private extractSliceFromChunk(
    chunkData: Float32Array,
    chunkShape: number[],
    levelArray: zarr.Array<zarr.DataType>,
    chunkIndices: number[]
  ): {
    data: Float32Array
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
        const dimSelection =
          this.selectors[dimKey] ??
          this.selectors[dimName] ??
          this.selectors[this.dimIndices[dimKey]?.name]

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

    const paddedData = new Float32Array(tileWidth * tileHeight * channels)
    paddedData.fill(this.fillValue)

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
          // Pre-calculate destination index to avoid repeated arithmetic
          const baseDstIdx = latIdx * tileWidth + lonIdx
          const dstIdx = baseDstIdx * channels + channelIdx

          if (srcIdx < chunkData.length) {
            const val = chunkData[srcIdx]
            paddedData[dstIdx] = val
            bandArrays[channelIdx][baseDstIdx] = val
          }
        })
      }
    }

    for (let c = 0; c < channels; c++) {
      const bandName = this.bandNames[c] || `band_${c}`
      bandData.set(bandName, bandArrays[c])
    }

    return { data: paddedData, channels, bandData }
  }

  private getOrCreateTile(tileKey: string): TileDataCache {
    let tile = this.tiles.get(tileKey)
    if (!tile) {
      tile = {
        chunkData: null,
        chunkShape: null,
        chunkIndices: undefined,
        data: null,
        bandData: new Map(),
        channels: 1,
        selectorHash: null,
        loading: false,
        lastUsed: Date.now(),
      }
      this.tiles.set(tileKey, tile)
      this.accessOrder.push(tileKey)
      this.evictOldTiles()
    } else {
      tile.lastUsed = Date.now()
      const idx = this.accessOrder.indexOf(tileKey)
      if (idx > -1) {
        this.accessOrder.splice(idx, 1)
        this.accessOrder.push(tileKey)
      }
    }
    return tile
  }

  private evictOldTiles() {
    while (this.tiles.size > this.maxCachedTiles) {
      const oldestKey = this.accessOrder.shift()
      if (!oldestKey) break
      this.tiles.delete(oldestKey)
    }
  }

  getTile(tileTuple: TileTuple): TileDataCache | undefined {
    return this.tiles.get(tileToKey(tileTuple))
  }

  async reextractTileSlices(
    visibleTiles: TileTuple[],
    selectorHash: string
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
        const sliced = this.extractSliceFromChunk(
          tile.chunkData!,
          tile.chunkShape!,
          levelArray,
          desiredChunkIndices
        )
        tile.data = sliced.data
        tile.channels = sliced.channels
        tile.bandData = sliced.bandData
        tile.selectorHash = selectorHash
      } else {
        tile.data = null
        tile.bandData = new Map()
        tile.selectorHash = null
        tile.chunkData = null
        tile.chunkShape = null
        tile.chunkIndices = undefined
      }
    }
  }

  async fetchTile(
    tileTuple: TileTuple,
    selectorHash: string
  ): Promise<TileDataCache | null> {
    const [z] = tileTuple
    const levelPath = this.store.levels[z]
    if (!levelPath) return null

    const levelArray = await this.store.getLevelArray(levelPath)
    const tileKey = tileToKey(tileTuple)
    const tile = this.getOrCreateTile(tileKey)

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
        const sliced = this.extractSliceFromChunk(
          tile.chunkData!,
          tile.chunkShape!,
          levelArray,
          chunkIndices
        )
        tile.data = sliced.data
        tile.channels = sliced.channels
        tile.bandData = sliced.bandData
        tile.selectorHash = selectorHash
        tile.loading = false
        return tile
      }

      const chunk = await this.store.getChunk(levelPath, chunkIndices)
      const chunkShape = (chunk.shape as number[]).map((n) => Number(n))
      const chunkData =
        chunk.data instanceof Float32Array
          ? new Float32Array(chunk.data.buffer)
          : Float32Array.from(chunk.data as ArrayLike<number>)

      tile.chunkData = chunkData
      tile.chunkShape = chunkShape
      tile.chunkIndices = chunkIndices
      const sliced = this.extractSliceFromChunk(
        chunkData,
        chunkShape,
        levelArray,
        chunkIndices
      )
      tile.data = sliced.data
      tile.channels = sliced.channels
      tile.bandData = sliced.bandData
      tile.selectorHash = selectorHash
      tile.loading = false
      return tile
    } catch (err) {
      console.error('Error fetching tile data:', err)
      tile.loading = false
      return null
    }
  }
}
