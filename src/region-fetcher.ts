import * as zarr from 'zarrita'
import type { DimIndicesProps, UntiledLevel } from './types'
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
import { interleaveBands, normalizeDataForTexture } from './webgl-utils'
import {
  type ChunkLoadingDebouncer,
  type RequestCanceller,
  cancelAllRequests,
  hasActiveRequests,
} from './region-utils'
import { RegionCache, createRegionState, makeRegionKey } from './region-cache'

export type RegionFetcherContext = {
  zarrStore: ZarrStore
  dimIndices: DimIndicesProps
  levels: UntiledLevel[]
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
      // Kick off every region synchronously so their underlying chunk reads
      // land in one microtask drain — that's what lets the range coalescer
      // (icechunk-js + zarrita) merge them into a handful of HTTP fetches
      // instead of one per region. Browser HTTP queueing handles back-
      // pressure on the connection pool; throttling fetchRegion calls here
      // just fragments the coalescer's same-tick batch window.
      const fetches = regions.map(({ regionX, regionY }) =>
        this.fetchRegion(regionX, regionY, snapshot)
      )
      await Promise.allSettled(fetches)
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

      // Fetch data for all channels
      const bandArrays: Float32Array[] = []

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

        const rawData = new Float32Array(result.data as ArrayLike<number>)
        bandArrays.push(rawData)
      } else {
        // Multi-channel - fetch all channels in parallel
        if (isStale()) return

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
          const bandData = new Float32Array(result.data as ArrayLike<number>)
          bandArrays.push(bandData)
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

      // Normalize bands (single pass) and collect for interleaving
      region.bandData.clear()
      region.bandTexturesUploaded.clear()
      const normalizedBands: Float32Array[] = []

      for (let c = 0; c < bandArrays.length; c++) {
        const bandName = snapshot.bandNames[c] || `band_${c}`
        let bandData = bandArrays[c]

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

      // Construct interleaved data from normalized bands
      region.data = interleaveBands(normalizedBands, numChannels)

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
