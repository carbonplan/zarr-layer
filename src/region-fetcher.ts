import * as zarr from 'zarrita'
import type { DimIndicesProps, ResolutionLevel } from './types'
import type { XYLimits } from './map-utils'
import type { ProjectionContext } from './projection-utils'
import type {
  LevelMeta,
  LevelRuntime,
  LevelSnapshot,
  RegionState,
} from './region-state'
import type { ZarrStore } from './zarr-store'
import { buildChannelCombinations } from './selector-resolution'
import {
  bandFormatForDtype,
  isNativeBandArray,
  type BandArray,
} from './band-format'
import { interleaveBands, normalizeDataForTexture } from './webgl-utils'
import {
  type ChunkLoadingDebouncer,
  type RequestCanceller,
  cancelAllRequests,
  enqueueFetches,
  hasActiveRequests,
} from './region-utils'
import { RegionCache, createRegionState, makeRegionKey } from './region-cache'

/**
 * When the only multi-value dimension selects an ascending run of
 * non-negative indices and comes before the other sliced dimensions, one
 * read returns every band, in selection order.
 */
export function contiguousBandRange(
  multiValueDims: LevelSnapshot['baseMultiValueDims'],
  sliceArgs: (number | zarr.Slice | null)[],
  shape: readonly number[]
): { dimIndex: number; start: number } | null {
  if (multiValueDims.length !== 1) return null
  const { dimIndex, values } = multiValueDims[0]
  // A negative index counts from the end, and a fractional one is
  // truncated; as a slice start neither would be.
  if (values.length < 2 || values[0] < 0 || !Number.isInteger(values[0])) {
    return null
  }
  // A slice past the end is clipped where an index would fail, which would
  // quietly return fewer bands than selected.
  if (values[0] + values.length > shape[dimIndex]) return null
  for (let i = 1; i < values.length; i++) {
    if (values[i] !== values[0] + i) return null
  }
  // Integer args drop their dimension from the result; every other
  // dimension must come after this one for its bands to be contiguous.
  for (let d = 0; d < dimIndex; d++) {
    if (typeof sliceArgs[d] !== 'number') return null
  }
  return { dimIndex, start: values[0] }
}

/**
 * Split a read whose first dimension is the band into one array per band,
 * each holding its values in the result's own memory order, as a separate
 * read of that band would return them. In a C-order result each band is a
 * contiguous block and becomes a view; otherwise (a Fortran-order store)
 * each band is gathered by stride.
 */
export function splitBands(result: {
  data: ArrayLike<number> & {
    subarray(begin: number, end: number): ArrayLike<number>
  }
  shape: number[]
  stride: number[]
}): ArrayLike<number>[] {
  const { data, shape, stride } = result
  const bands = shape[0]
  const bandLength = shape.slice(1).reduce((n, size) => n * size, 1)
  if (isCOrder(shape, stride)) {
    return Array.from({ length: bands }, (_, c) =>
      data.subarray(c * bandLength, (c + 1) * bandLength)
    )
  }
  // The band's other dimensions, the one that varies fastest in memory first.
  const dims = shape
    .map((_, d) => d)
    .slice(1)
    .sort((a, b) => stride[a] - stride[b])
  const Typed = data.constructor as new (length: number) => {
    [i: number]: number
    length: number
  }
  return Array.from({ length: bands }, (_, c) => {
    const out = new Typed(bandLength)
    const index = new Array<number>(dims.length).fill(0)
    for (let i = 0; i < bandLength; i++) {
      let offset = c * stride[0]
      for (let k = 0; k < dims.length; k++) {
        offset += index[k] * stride[dims[k]]
      }
      out[i] = data[offset]
      for (let k = 0; k < dims.length; k++) {
        if (++index[k] < shape[dims[k]]) break
        index[k] = 0
      }
    }
    return out as unknown as ArrayLike<number>
  })
}

const NUMERIC_DTYPES = new Set<string>([
  'int8',
  'uint8',
  'int16',
  'uint16',
  'int32',
  'uint32',
  'float16',
  'float32',
  'float64',
])

/** Whether strides describe a dense row-major (C-order) layout. */
export function isCOrder(shape: number[], stride: number[]): boolean {
  let expected = 1
  for (let d = shape.length - 1; d >= 0; d--) {
    if (shape[d] > 1 && stride[d] !== expected) return false
    expected *= shape[d]
  }
  return true
}

export type RegionFetcherContext = {
  zarrStore: ZarrStore
  dimIndices: DimIndicesProps
  levels: ResolutionLevel[]
  projection: ProjectionContext
  xyLimits: XYLimits | null
  latIsAscending: boolean
  fixedDataScale: number
  regionCache: RegionCache
  requestCanceller: RequestCanceller
  loadingDebouncer: ChunkLoadingDebouncer
  getActiveLevel: () => LevelRuntime | null
  getSelectorVersion: () => number
  getBandNames: () => string[]
  /**
   * True when a custom shader samples the per-band textures. The main texture
   * is then never bound, so the interleaved copy of every channel that feeds
   * it is dead weight (12 bands at 128x128 is 786 KB per region).
   */
  usesBandTextures: () => boolean
  isRemoved: () => boolean
  getRegionBounds: (
    regionX: number,
    regionY: number,
    levelMeta: LevelMeta
  ) => {
    xMin: number
    xMax: number
    yMin: number
    yMax: number
  }
  computeRegionMercatorBounds: (bounds: {
    xMin: number
    xMax: number
    yMin: number
    yMax: number
  }) => { x0: number; y0: number; x1: number; y1: number }
  createRegionGeometry: (
    regionX: number,
    regionY: number,
    region: RegionState
  ) => void
  invalidate: () => void
}

export class RegionFetcher {
  constructor(private context: RegionFetcherContext) {}

  /**
   * Clear loading flags for queued-but-not-started regions in a batch.
   * Only touches regions where requestId is null (pre-marked as loading
   * but no fetch was started). In-flight regions (requestId set) are
   * cleaned up by their own finally block.
   */
  private clearBatchLoadingFlags(
    regions: Array<{ regionX: number; regionY: number }>,
    levelIndex: number
  ): void {
    for (const { regionX, regionY } of regions) {
      const region = this.context.regionCache.get(
        makeRegionKey(levelIndex, regionX, regionY)
      )
      if (region && region.requestId === null) region.loading = false
    }
  }

  /**
   * Fetch multiple regions with limited concurrency to avoid overwhelming the browser.
   */
  async fetchRegions(
    regions: Array<{ regionX: number; regionY: number }>
  ): Promise<void> {
    // Can't fetch without a committed level.
    const level = this.context.getActiveLevel()
    if (!level) return

    // Capture ALL level-dependent state at start to pass to fetchRegion.
    // This prevents races where a later `loadLevel` (zoom switch or
    // selector rebuild) swaps `this.activeLevel` mid-batch.
    const snapshot: LevelSnapshot = {
      index: level.index,
      zarrArray: level.zarrArray,
      baseSliceArgs: [...level.baseSliceArgs],
      width: level.width,
      height: level.height,
      regionSize: level.regionSize,
      xyLimits: level.xyLimits,
      selectorVersion: this.context.getSelectorVersion(),
      bandNames: [...this.context.getBandNames()],
      baseMultiValueDims: level.baseMultiValueDims.map((dim) => ({
        dimIndex: dim.dimIndex,
        dimName: dim.dimName,
        values: [...dim.values],
        labels: [...dim.labels],
      })),
    }

    this.context.loadingDebouncer.show()

    // Mark ALL regions as loading upfront to prevent duplicate fetches
    // from subsequent update() calls before we've processed them all
    for (const { regionX, regionY } of regions) {
      const key = makeRegionKey(snapshot.index, regionX, regionY)
      let region = this.context.regionCache.get(key)
      if (!region) {
        region = createRegionState(
          snapshot.index,
          regionX,
          regionY,
          this.context.latIsAscending,
          this.context.getSelectorVersion()
        )
        this.context.regionCache.set(key, region)
      }
      region.loading = true
    }

    // Pre-flight staleness check. Mid-flight changes are handled by the
    // `cancelAllRequests` in `loadLevel`/`setSelector`, which aborts the
    // signals that fetchRegion threads through every await.
    if (
      (this.context.getActiveLevel()?.index ?? -1) !== snapshot.index ||
      this.context.getSelectorVersion() !== snapshot.selectorVersion
    ) {
      cancelAllRequests(this.context.requestCanceller)
      this.clearBatchLoadingFlags(regions, snapshot.index)
    } else {
      // Regions wait in the layer's queue, which starts a few at a time,
      // nearest the viewport center first. A region that leaves the
      // viewport while queued is dropped without a request.
      const fetches = regions.map(
        ({ regionX, regionY }) =>
          new Promise<void>((resolve) => {
            const key = makeRegionKey(snapshot.index, regionX, regionY)
            const release = () => {
              const region = this.context.regionCache.get(key)
              if (region && region.requestId === null) {
                region.loading = false
              }
              resolve()
            }
            enqueueFetches(this.context.requestCanceller, [
              {
                key,
                regionX,
                regionY,
                start: async () => {
                  // The level or selector may have moved on while queued.
                  if (
                    this.context.isRemoved() ||
                    (this.context.getActiveLevel()?.index ?? -1) !==
                      snapshot.index ||
                    this.context.getSelectorVersion() !==
                      snapshot.selectorVersion
                  ) {
                    release()
                    return
                  }
                  await this.fetchRegion(regionX, regionY, snapshot)
                  resolve()
                },
                drop: release,
              },
            ])
          })
      )
      await Promise.all(fetches)
    }

    // Only update loading state if we're still on the same level
    if (!hasActiveRequests(this.context.requestCanceller)) {
      this.context.loadingDebouncer.hide()

      this.context.invalidate()
    }
  }

  /**
   * Fetch data for a single region.
   * Handles multi-band extraction when selector has multi-value dimensions.
   * @param snapshot - Captured level state from when fetch batch started (prevents race conditions)
   */
  private async fetchRegion(
    regionX: number,
    regionY: number,
    snapshot: LevelSnapshot
  ): Promise<void> {
    if ((this.context.getActiveLevel()?.index ?? -1) !== snapshot.index) {
      return
    }

    if (this.context.isRemoved()) {
      return
    }

    const key = makeRegionKey(snapshot.index, regionX, regionY)
    const requestId = ++this.context.requestCanceller.currentVersion
    const fetchSelectorVersion = snapshot.selectorVersion

    const controller = new AbortController()
    this.context.requestCanceller.controllers.set(requestId, controller)

    let region = this.context.regionCache.get(key)
    if (!region) {
      region = createRegionState(
        snapshot.index,
        regionX,
        regionY,
        this.context.latIsAscending,
        this.context.getSelectorVersion()
      )
      this.context.regionCache.set(key, region)
    }
    region.loading = true
    region.requestId = requestId

    const [regionH, regionW] = snapshot.regionSize

    // Calculate pixel bounds for this region
    const yStart = regionY * regionH
    const yEnd = Math.min(yStart + regionH, snapshot.height)
    const xStart = regionX * regionW
    const xEnd = Math.min(xStart + regionW, snapshot.width)
    const actualW = xEnd - xStart
    const actualH = yEnd - yStart

    try {
      // Build base slice args with spatial region bounds
      const baseSliceArgs = [...snapshot.baseSliceArgs]
      const latIdx = this.context.dimIndices.lat.index
      const lonIdx = this.context.dimIndices.lon.index
      baseSliceArgs[latIdx] = zarr.slice(yStart, yEnd)
      baseSliceArgs[lonIdx] = zarr.slice(xStart, xEnd)

      const desc = this.context.zarrStore.describe()
      // Use per-level metadata if available (for heterogeneous pyramids)
      const currentLevel = this.context.levels[snapshot.index]
      const fillValue = currentLevel?.fillValue ?? desc.fill_value

      const { combinations: channelCombinations } = buildChannelCombinations(
        snapshot.baseMultiValueDims
      )
      const numChannels = channelCombinations.length || 1
      // Only typed numeric arrays split into band views; bool arrays are
      // read band by band.
      const contiguousRange = NUMERIC_DTYPES.has(snapshot.zarrArray.dtype)
        ? contiguousBandRange(
            snapshot.baseMultiValueDims,
            baseSliceArgs,
            snapshot.zarrArray.shape
          )
        : null

      // Band-sampling shaders read small integer dtypes straight from
      // integer textures; everything else is converted to float32 here.
      const native =
        this.context.usesBandTextures() &&
        bandFormatForDtype(snapshot.zarrArray.dtype) !== 'float'
      const toBand = (data: ArrayLike<number>): BandArray =>
        native && isNativeBandArray(data)
          ? data
          : new Float32Array(data as ArrayLike<number>)

      // Fetch data for all channels
      const bandArrays: BandArray[] = []

      const isStale = () =>
        controller.signal.aborted ||
        this.context.isRemoved() ||
        (this.context.getActiveLevel()?.index ?? -1) !== snapshot.index

      if (numChannels === 1) {
        // Single channel - simple fetch
        if (isStale()) return

        const result = (await zarr.get(snapshot.zarrArray, baseSliceArgs, {
          signal: controller.signal,
        })) as { data: ArrayLike<number> }

        if (isStale()) return

        bandArrays.push(toBand(result.data))
      } else {
        if (isStale()) return

        // One read covers every band of a contiguous selection, so bands
        // come out of it rather than a read each.
        if (contiguousRange) {
          const sliceArgs = [...baseSliceArgs]
          sliceArgs[contiguousRange.dimIndex] = zarr.slice(
            contiguousRange.start,
            contiguousRange.start + numChannels
          )
          const result = (await zarr.get(snapshot.zarrArray, sliceArgs, {
            signal: controller.signal,
          })) as {
            data: ArrayLike<number> & {
              subarray(begin: number, end: number): ArrayLike<number>
            }
            shape: number[]
            stride: number[]
          }

          if (isStale()) return

          for (const band of splitBands(result)) bandArrays.push(toBand(band))
        } else {
          // Build slice args for all channels upfront
          const allSliceArgs: (number | zarr.Slice)[][] = []
          for (let c = 0; c < numChannels; c++) {
            const sliceArgs = [...baseSliceArgs]
            const combo = channelCombinations[c]

            // Apply channel-specific indices to multi-value dimensions
            for (let i = 0; i < snapshot.baseMultiValueDims.length; i++) {
              sliceArgs[snapshot.baseMultiValueDims[i].dimIndex] = combo[i]
            }
            allSliceArgs.push(sliceArgs)
          }

          // Fetch all bands in parallel
          const results = await Promise.all(
            allSliceArgs.map((sliceArgs) =>
              zarr.get(snapshot.zarrArray, sliceArgs, {
                signal: controller.signal,
              })
            )
          )

          if (isStale()) return

          // Process results in order
          for (let c = 0; c < numChannels; c++) {
            const result = results[c] as { data: ArrayLike<number> }
            bandArrays.push(toBand(result.data))
          }
        }
      }

      // Only render if this is newer than what's already rendered for this region
      if (fetchSelectorVersion < region.selectorVersion) return

      // Update region's selector version
      region.selectorVersion = fetchSelectorVersion

      // GPU handles reprojection. Source-projected data uses an adaptive mesh
      // (source CRS → WGS84), then the GPU projects to Mercator or ECEF.
      const needsProj4MercBounds =
        this.context.projection.def && this.context.projection.toMercator

      if (
        needsProj4MercBounds &&
        this.context.xyLimits &&
        !region.mercatorBounds
      ) {
        const levelMeta: LevelMeta = {
          width: snapshot.width,
          height: snapshot.height,
          regionSize: snapshot.regionSize,
          xyLimits: snapshot.xyLimits,
        }
        const geoBounds = this.context.getRegionBounds(
          regionX,
          regionY,
          levelMeta
        )
        region.mercatorBounds =
          this.context.computeRegionMercatorBounds(geoBounds)
      }

      // Apply per-level scale/offset to convert raw values to physical units
      // Fall back to dataset-level scale/offset for pyramids that only define them at the root
      const scaleFactor = currentLevel?.scaleFactor ?? desc.scaleFactor
      const addOffset = currentLevel?.addOffset ?? desc.addOffset

      region.bandData.clear()
      region.bandTextureKey = null
      region.bandTransform = null
      const normalizedBands: Float32Array[] = []

      for (let c = 0; c < bandArrays.length; c++) {
        const bandName = snapshot.bandNames[c] || `band_${c}`
        const band = bandArrays[c]

        // Raw integers are transformed in the shader, per region.
        if (!(band instanceof Float32Array)) {
          region.bandData.set(bandName, band)
          region.bandTransform = {
            scale: scaleFactor,
            offset: addOffset,
            fill: fillValue,
          }
          continue
        }

        let bandData = band

        // Apply scale/offset if needed (converts raw to physical values)
        if (scaleFactor !== 1 || addOffset !== 0) {
          const scaled = new Float32Array(bandData.length)
          for (let i = 0; i < bandData.length; i++) {
            const raw = bandData[i]
            // Scale all values including fill - normalizeDataForTexture will filter by scaled fill
            if (!Number.isFinite(raw)) {
              scaled[i] = raw // Keep NaN/Inf as-is
            } else {
              scaled[i] = raw * scaleFactor + addOffset
            }
          }
          bandData = scaled
        }

        // Compute the fill value in the same space as the data
        const effectiveFillValue =
          fillValue !== null && (scaleFactor !== 1 || addOffset !== 0)
            ? fillValue * scaleFactor + addOffset
            : fillValue

        const { normalized: bandNormalized } = normalizeDataForTexture(
          bandData,
          effectiveFillValue,
          this.context.fixedDataScale
        )
        region.bandData.set(bandName, bandNormalized)
        normalizedBands.push(bandNormalized)
      }

      // Interleaved data exists only to feed the main texture. Band-sampling
      // shaders never read it, so skip the copy entirely rather than leaving
      // a partial stand-in: if the shader later switches back to the main
      // texture, a stand-in would be uploaded and drawn as if it were the
      // whole dataset until the refetch lands. Null keeps the region
      // undrawable through that path instead.
      const bandRendering = this.context.usesBandTextures()
      region.data = bandRendering
        ? null
        : interleaveBands(normalizedBands, numChannels)

      // Check if geometry needs to be (re)created before updating dimensions
      // The adaptive mesh only depends on spatial bounds and dimensions, not the selector
      const needsGeometry =
        !region.vertexArr ||
        region.width !== actualW ||
        region.height !== actualH

      region.width = actualW
      region.height = actualH
      region.channels = numChannels
      region.loading = false

      // Store level-specific dimensions from snapshot for geometry creation.
      // Must use snapshot (not this.*) to avoid races with level switching.
      // Set before createRegionGeometry is called below.
      region.levelMeta = {
        width: snapshot.width,
        height: snapshot.height,
        regionSize: [...snapshot.regionSize] as [number, number],
        xyLimits: snapshot.xyLimits,
      }

      region.textureUploaded = false

      // Create geometry only if needed (new region or dimensions changed)
      if (needsGeometry) {
        this.context.createRegionGeometry(regionX, regionY, region)
      }

      this.context.invalidate()
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        console.error(`[fetchRegion] Error fetching region ${key}:`, err)
      }
    } finally {
      // Only clear flags if this request still owns the region — a newer
      // request may have taken over while an aborted one was unwinding.
      if (region.requestId === requestId) {
        region.loading = false
        region.requestId = null
      }
      this.context.requestCanceller.controllers.delete(requestId)
      // Re-evaluate visible regions after abort so panned-back regions get re-fetched.
      if (controller.signal.aborted && !this.context.isRemoved()) {
        this.context.invalidate()
      }
    }
  }
}
