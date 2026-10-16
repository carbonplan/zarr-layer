import { describe, it, expect, vi } from 'vitest'
import { RegionRenderer, type TileRequest } from './region-renderer'
import { ZarrStore } from './zarr-store'
import { buildMemoryZarrStore } from './__fixtures__/memory-zarr'
import {
  MAX_CACHED_REGIONS,
  hasRegionData,
  makeRegionKey,
  type RegionCache,
} from './region-cache'
import type { LevelLoader } from './level-loader'
import type { MapLike } from './types'
import type { RegionCoordinate } from './region-math'
import type { RegionState } from './region-state'
import { boundsIntersect, getTileBounds } from './tile-utils'

/**
 * Per-tile level selection. The map picks a zoom per tile by distance when
 * pitched (and terrain-aware with terrain on); each tile reads the pyramid
 * level matching its zoom, so one update wants a mix of levels: fine near the
 * camera, coarse at the horizon. The region renderer fetches, protects,
 * evicts and draws that mix, coarser levels beneath finer, and falls back to
 * the whole viewport at the active level when the map gives no tile cover.
 *
 * The pyramid's levels are 256 and 512 pixels wide, so zoom 0 selects level 0
 * and zoom 1 selects level 1. Regions are 2x4 pixels with no chunk data:
 * level 0 is a 64x64 region grid (5.625 by 2.8125 degrees), level 1 128x128.
 */

function fakeGl({ failTextures = false } = {}) {
  let textures = 0
  let buffers = 0
  return {
    TEXTURE0: 0x84c0,
    TEXTURE_2D: 0x0de1,
    TEXTURE_2D_ARRAY: 0x8c1a,
    MAX_ARRAY_TEXTURE_LAYERS: 0x88ff,
    MAX_TEXTURE_SIZE: 0x0d33,
    getParameter: vi.fn((pname: number) =>
      pname === 0x88ff ? 256 : pname === 0x0d33 ? 4096 : 4
    ),
    createTexture: vi.fn(() => (failTextures ? null : { tex: ++textures })),
    createBuffer: vi.fn(() => ({ buf: ++buffers })),
    deleteTexture: vi.fn(),
    deleteBuffer: vi.fn(),
    bindTexture: vi.fn(),
    bindBuffer: vi.fn(),
    bufferData: vi.fn(),
    texImage2D: vi.fn(),
    texImage3D: vi.fn(),
    texParameteri: vi.fn(),
    activeTexture: vi.fn(),
    pixelStorei: vi.fn(),
  } as unknown as WebGL2RenderingContext
}

const mapAt = (
  west: number,
  south: number,
  east: number,
  north: number,
  zoom = 0
): MapLike => ({
  getBounds: () => ({
    getWest: () => west,
    getEast: () => east,
    toArray: () => [
      [west, south],
      [east, north],
    ],
  }),
  getZoom: () => zoom,
})

/** A small viewport in the southern Indian Ocean. */
const viewportAt = (zoom: number) => mapAt(100, -20, 105, -15, zoom)

type Draw = {
  region: RegionState
  clipPasses: Array<{ worldOffset: number; boxes: Float32Array }> | null
}
type Seam = {
  regionCache: RegionCache
  wantedRegions: Map<number, RegionCoordinate[]>
  wantedKeys: Set<string>
  pendingLevels: Set<number>
  levelLoader: LevelLoader
  activeLevel: { index: number } | null
  getLoadedRegions(gl: WebGL2RenderingContext): RegionState[]
  plannedDraws(gl: WebGL2RenderingContext, worldOffsets?: number[]): Draw[]
  planTile(
    tile: TileRequest,
    drawable: (region: RegionState) => boolean
  ): { regions: RegionState[]; ready: boolean }
  wantedCovered(): boolean
}
const seam = (renderer: RegionRenderer) => renderer as unknown as Seam

/** Latch a level as unresolvable under the current selector. */
const failLevel = (renderer: RegionRenderer, levelIndex: number) =>
  (
    seam(renderer).levelLoader as unknown as {
      selectorFailedLevels: Set<number>
    }
  ).selectorFailedLevels.add(levelIndex)

type Box = { x0: number; y0: number; x1: number; y1: number }

/**
 * A draw's clip boxes back in absolute Mercator of the base world, for the
 * passes drawn at one world offset.
 */
const absoluteBoxes = (draw: Draw, worldOffset = 0): Box[] => {
  const mesh = draw.region.meshBounds!
  const ax = (mesh.x0 + mesh.x1) / 2
  const ay = (mesh.y0 + mesh.y1) / 2
  const boxes: Box[] = []
  for (const pass of draw.clipPasses ?? []) {
    if (pass.worldOffset !== worldOffset) continue
    const p = pass.boxes
    for (let i = 0; i < p.length; i += 4) {
      boxes.push({
        x0: p[i] + ax,
        y0: p[i + 1] + ay,
        x1: p[i + 2] + ax,
        y1: p[i + 3] + ay,
      })
    }
  }
  return boxes
}

/** Boxes are uploaded as float32, so match them loosely. */
const sameBox = (a: Box, b: Box) =>
  Math.abs(a.x0 - b.x0) < 1e-6 &&
  Math.abs(a.y0 - b.y0) < 1e-6 &&
  Math.abs(a.x1 - b.x1) < 1e-6 &&
  Math.abs(a.y1 - b.y1) < 1e-6

const wantedAt = (renderer: RegionRenderer, levelIndex: number) =>
  (seam(renderer).wantedRegions.get(levelIndex) ?? [])
    .map((r) => `${r.regionX},${r.regionY}`)
    .sort()

const protectedKeys = (renderer: RegionRenderer) => {
  const cache = seam(renderer).regionCache
  return [...cache].filter(([key]) => cache.isProtected(key)).map(([k]) => k)
}

/** Hop the macrotask queue until no region fetch is in flight. */
async function settle(renderer: RegionRenderer): Promise<void> {
  const fetching = () => {
    for (const region of seam(renderer).regionCache.values()) {
      if (region.loading) return true
    }
    return false
  }
  do {
    await new Promise((resolve) => setTimeout(resolve, 0))
  } while (fetching())
}

/** Hop the macrotask queue until the level is committed as the active one. */
async function waitForActive(
  renderer: RegionRenderer,
  index: number
): Promise<void> {
  while (seam(renderer).activeLevel?.index !== index) {
    await new Promise((resolve) => setTimeout(resolve, 0))
  }
}

/** Hop the macrotask queue until a level opened on demand is available. */
async function waitForRuntime(
  renderer: RegionRenderer,
  index: number
): Promise<void> {
  while (!seam(renderer).levelLoader.runtime(index)) {
    await new Promise((resolve) => setTimeout(resolve, 0))
  }
}

async function makeRenderer({ single = false } = {}) {
  const memory = single
    ? buildMemoryZarrStore({
        arrays: [
          {
            name: 'temperature',
            shape: [64, 128],
            chunkShape: [2, 4],
            dimensionNames: ['lat', 'lon'],
          },
        ],
      })
    : buildMemoryZarrStore({
        attributes: {
          multiscales: {
            layout: [{ asset: '0' }, { asset: '1' }],
            crs: 'EPSG:4326',
          },
        },
        arrays: [
          {
            name: '0/temperature',
            shape: [128, 256],
            chunkShape: [2, 4],
            dimensionNames: ['lat', 'lon'],
          },
          {
            name: '1/temperature',
            shape: [256, 512],
            chunkShape: [2, 4],
            dimensionNames: ['lat', 'lon'],
          },
        ],
      })
  const store = new ZarrStore({
    customStore: memory,
    variable: 'temperature',
    version: 3,
    bounds: [-180, -90, 180, 90],
    latIsAscending: false,
  })
  await store.initialized

  const invalidate = vi.fn()
  const renderer = new RegionRenderer(store, 'temperature', {}, invalidate)
  await renderer.initialize()
  const gl = fakeGl()
  return { renderer, invalidate, gl }
}

/** A renderer on the pyramid with the given level active. */
async function rendererAtZoom(zoom: number) {
  const made = await makeRenderer()
  const map = viewportAt(zoom)
  made.renderer.update(map, made.gl)
  await waitForActive(made.renderer, zoom)
  made.renderer.update(map, made.gl)
  await settle(made.renderer)
  return { ...made, map }
}

/** Two updates: the first opens any level the cover needs, the second uses it. */
async function updateWithCover(
  renderer: RegionRenderer,
  map: MapLike,
  gl: WebGL2RenderingContext,
  tiles: TileRequest[]
) {
  renderer.setCoveringTiles(tiles)
  renderer.update(map, gl)
  for (const tile of tiles) {
    await waitForRuntime(renderer, Math.round(tile.zoom))
  }
  renderer.update(map, gl)
}

// Zoom-6 tiles near the camera (lon -135..-129, lat 52..56: a few level-1
// regions each) and a zoom-5 tile at the horizon (lon -135..-124, lat 49..56:
// a few level-0 regions).
const NEAR: TileRequest = { tileId: { z: 6, x: 8, y: 20 }, zoom: 1 }
const NEAR_EAST: TileRequest = { tileId: { z: 6, x: 9, y: 20 }, zoom: 1 }
const FAR: TileRequest = { tileId: { z: 5, x: 4, y: 10 }, zoom: 0 }
// The zoom-5 tile east of FAR (lon -123.75..-112.5), clear of NEAR, and a
// zoom-6 tile inside it.
const FAR_EAST: TileRequest = { tileId: { z: 5, x: 5, y: 10 }, zoom: 0 }
const NEAR_IN_FAR_EAST: TileRequest = {
  tileId: { z: 6, x: 10, y: 20 },
  zoom: 1,
}

describe('per-tile level selection', () => {
  it('reads each tile from the level matching its zoom', async () => {
    const { renderer, map, gl } = await rendererAtZoom(1)

    await updateWithCover(renderer, map, gl, [NEAR, FAR])

    expect(wantedAt(renderer, 1).length).toBeGreaterThan(0)
    expect(wantedAt(renderer, 0).length).toBeGreaterThan(0)
    // Level 0 regions are 5.625 by 2.8125 degrees, so the far tile's 11 by 7
    // degree span is a handful of them rather than the dozens it would be
    // at level 1.
    expect(wantedAt(renderer, 0).length).toBeLessThanOrEqual(12)
  })

  it('reads only the center level when every tile shares the map zoom', async () => {
    // An untilted view: the cover is uniform, so nothing changes from a
    // whole-viewport selection at the active level.
    const { renderer, map, gl } = await rendererAtZoom(1)

    await updateWithCover(renderer, map, gl, [NEAR, NEAR_EAST])

    expect([...seam(renderer).wantedRegions.keys()]).toEqual([1])
  })

  it('fetches every wanted level and protects all of it', async () => {
    const { renderer, map, gl } = await rendererAtZoom(1)

    await updateWithCover(renderer, map, gl, [NEAR, FAR])
    await settle(renderer)

    const cache = seam(renderer).regionCache
    for (const key of seam(renderer).wantedKeys) {
      const region = cache.get(key)
      expect(region && hasRegionData(region)).toBe(true)
      expect(cache.isProtected(key)).toBe(true)
    }
    expect(protectedKeys(renderer).some((k) => k.startsWith('0:'))).toBe(true)
    expect(protectedKeys(renderer).some((k) => k.startsWith('1:'))).toBe(true)
  })

  it('draws coarser levels beneath finer ones', async () => {
    const { renderer, map, gl } = await rendererAtZoom(1)
    await updateWithCover(renderer, map, gl, [NEAR, FAR])
    await settle(renderer)

    const drawn = seam(renderer).getLoadedRegions(gl)
    const levels = drawn.map((region) => region.levelIndex)
    expect(levels).toContain(0)
    expect(levels).toContain(1)
    expect(levels.lastIndexOf(0)).toBeLessThan(levels.indexOf(1))
  })

  it('releases regions of a wanted level as soon as the cover leaves them', async () => {
    const { renderer, map, gl } = await rendererAtZoom(1)
    await updateWithCover(renderer, map, gl, [NEAR])
    await settle(renderer)
    const before = protectedKeys(renderer)
    expect(before.length).toBeGreaterThan(0)

    await updateWithCover(renderer, map, gl, [NEAR_EAST])

    const stillWanted = seam(renderer).wantedKeys
    for (const key of before) {
      expect(seam(renderer).regionCache.isProtected(key)).toBe(
        stillWanted.has(key)
      )
    }
  })

  it('switches levels mid-load, keeping the coarse coverage until the fine one is drawn', async () => {
    // Zoom in while level 0 is drawn: level 1 is wanted, its regions are in
    // flight, and the level-0 regions stay as fallbacks meanwhile. Once the
    // fine regions are drawn the fallbacks are released and evictable.
    const { renderer, gl } = await rendererAtZoom(0)
    const coarse = viewportAt(0)
    await updateWithCover(renderer, coarse, gl, [FAR])
    await settle(renderer)
    const coarseKeys = protectedKeys(renderer)
    expect(coarseKeys.every((k) => k.startsWith('0:'))).toBe(true)

    const fine = viewportAt(1)
    renderer.setCoveringTiles([NEAR])
    renderer.update(fine, gl)
    await waitForActive(renderer, 1)
    renderer.update(fine, gl)

    // Coarse regions under the fine tile stay as fallbacks; the rest of the
    // far tile is no longer shown anywhere and is released at once.
    const cache = seam(renderer).regionCache
    expect([...cache.values()].some((r) => r.loading)).toBe(true)
    const underNear = coarseKeys.filter((key) =>
      boundsIntersect(
        cache.get(key)!.mercatorBounds!,
        getTileBounds(NEAR.tileId)
      )
    )
    expect(underNear.length).toBeGreaterThan(0)
    for (const key of underNear) {
      expect(cache.isProtected(key)).toBe(true)
    }
    expect(coarseKeys.some((key) => !cache.isProtected(key))).toBe(true)
    expect(wantedAt(renderer, 1).length).toBeGreaterThan(0)
    const drawnWhileLoading = seam(renderer).getLoadedRegions(gl)
    expect(drawnWhileLoading.some((r) => r.levelIndex === 0)).toBe(true)

    await settle(renderer)
    seam(renderer).getLoadedRegions(gl)
    renderer.update(fine, gl)
    for (const key of coarseKeys) {
      expect(cache.isProtected(key)).toBe(false)
    }
    cache.evict(gl, 1)
    expect([...cache].every(([key]) => key.startsWith('1:'))).toBe(true)
  })

  it('keeps a coarse fallback while its fine replacement loads even if the coarse level stays wanted', async () => {
    // Two far tiles at level 0; then one of them is drawn fine. Level 0 is
    // still wanted for the other, so the regions under the fine tile are no
    // longer wanted, but they are what shows there until the fine data can
    // be drawn.
    const { renderer, gl } = await rendererAtZoom(0)
    const coarse = viewportAt(0)
    await updateWithCover(renderer, coarse, gl, [FAR, FAR_EAST])
    await settle(renderer)
    const cache = seam(renderer).regionCache
    const underNear = protectedKeys(renderer).filter((key) =>
      boundsIntersect(
        cache.get(key)!.mercatorBounds!,
        getTileBounds(NEAR.tileId)
      )
    )
    expect(underNear.length).toBeGreaterThan(0)

    const fine = viewportAt(1)
    renderer.setCoveringTiles([NEAR, FAR_EAST])
    renderer.update(fine, gl)
    await waitForActive(renderer, 1)
    renderer.update(fine, gl)

    expect([...seam(renderer).wantedRegions.keys()].sort()).toEqual([0, 1])
    for (const key of underNear) {
      expect(seam(renderer).wantedKeys.has(key)).toBe(false)
      expect(cache.isProtected(key)).toBe(true)
    }
    const drawn = seam(renderer)
      .getLoadedRegions(gl)
      .map((r) => r.key)
    for (const key of underNear) expect(drawn).toContain(key)

    await settle(renderer)
    seam(renderer).getLoadedRegions(gl)
    renderer.update(fine, gl)
    for (const key of underNear) {
      expect(cache.isProtected(key)).toBe(false)
    }
  })

  it('does not count the view as covered while a requested level is still opening', async () => {
    // The near tile's level is open and drawable; the far tile's is not.
    // Releasing fallbacks on the strength of the near tile alone would leave
    // the far tile with nothing.
    const { renderer, map, gl } = await rendererAtZoom(1)
    await updateWithCover(renderer, map, gl, [NEAR])
    await settle(renderer)
    seam(renderer).getLoadedRegions(gl)
    expect(seam(renderer).wantedCovered()).toBe(true)

    renderer.setCoveringTiles([NEAR, FAR])
    renderer.update(map, gl)
    expect(seam(renderer).pendingLevels.has(0)).toBe(true)
    expect(seam(renderer).wantedCovered()).toBe(false)

    await waitForRuntime(renderer, 0)
    renderer.update(map, gl)
    await settle(renderer)
    seam(renderer).getLoadedRegions(gl)
    renderer.update(map, gl)
    expect(seam(renderer).pendingLevels.size).toBe(0)
    expect(seam(renderer).wantedCovered()).toBe(true)
  })

  it('clips each region to the tiles that show its level', async () => {
    // A fine tile beside a far one. Every pixel shows one level, so each
    // region is drawn only inside the tile reading its level, whatever the
    // region's own outline: the box is the tile's exact Mercator rectangle.
    const { renderer, map, gl } = await rendererAtZoom(1)
    await updateWithCover(renderer, map, gl, [NEAR, FAR_EAST])
    await settle(renderer)

    const draws = seam(renderer).plannedDraws(gl)
    expect(draws.some((d) => d.region.levelIndex === 0)).toBe(true)
    expect(draws.some((d) => d.region.levelIndex === 1)).toBe(true)
    for (const draw of draws) {
      const tile = draw.region.levelIndex === 1 ? NEAR : FAR_EAST
      const boxes = absoluteBoxes(draw)
      expect(boxes).toHaveLength(1)
      expect(sameBox(boxes[0], getTileBounds(tile.tileId))).toBe(true)
    }
  })

  it('owns tiles per world copy when copies are covered at different zooms', async () => {
    // A pitched low-zoom view covers one world copy coarsely and the next
    // finely: here the far tile in the base copy and, inside its footprint,
    // the near tile in the copy to the west. Each copy keeps one owner per
    // point: the far tile's regions are drawn only in the base copy and the
    // near tile's only in the west copy, never both over the same pixels.
    const { renderer, map, gl } = await rendererAtZoom(1)
    const base = { ...FAR, wrap: 0 }
    const west = { ...NEAR, wrap: -1 }
    await updateWithCover(renderer, map, gl, [base, west])
    await settle(renderer)

    const draws = seam(renderer).plannedDraws(gl, [-1, 0])
    expect(draws.some((d) => d.region.levelIndex === 0)).toBe(true)
    expect(draws.some((d) => d.region.levelIndex === 1)).toBe(true)
    for (const draw of draws) {
      const offsets = [...new Set(draw.clipPasses!.map((p) => p.worldOffset))]
      if (draw.region.levelIndex === 0) {
        expect(offsets).toEqual([0])
        expect(
          sameBox(absoluteBoxes(draw, 0)[0], getTileBounds(FAR.tileId))
        ).toBe(true)
      } else {
        expect(offsets).toEqual([-1])
        expect(
          sameBox(absoluteBoxes(draw, -1)[0], getTileBounds(NEAR.tileId))
        ).toBe(true)
      }
    }
  })

  it('shows a tile from one complete level rather than a partial mix', async () => {
    // Zoom in over a coarse tile. Until every fine region under the near
    // tile is drawable, the tile shows the coarse data it already has,
    // clipped to the tile, and nothing of the fine level; the coarse
    // regions stay protected meanwhile.
    const { renderer, gl } = await rendererAtZoom(0)
    await updateWithCover(renderer, viewportAt(0), gl, [FAR_EAST])
    await settle(renderer)
    seam(renderer).plannedDraws(gl)

    const fine = viewportAt(1)
    renderer.setCoveringTiles([NEAR_IN_FAR_EAST])
    renderer.update(fine, gl)
    await waitForActive(renderer, 1)
    renderer.update(fine, gl)
    await settle(renderer)
    const cache = seam(renderer).regionCache
    const fineKeys = [...seam(renderer).wantedKeys]
    expect(fineKeys.every((k) => k.startsWith('1:'))).toBe(true)
    cache.delete(fineKeys[0])
    renderer.update(fine, gl)

    const draws = seam(renderer).plannedDraws(gl)
    expect(draws.length).toBeGreaterThan(0)
    expect(draws.every((d) => d.region.levelIndex === 0)).toBe(true)
    const tileBounds = getTileBounds(NEAR_IN_FAR_EAST.tileId)
    for (const draw of draws) {
      expect(cache.isProtected(draw.region.key)).toBe(true)
      expect(sameBox(absoluteBoxes(draw)[0], tileBounds)).toBe(true)
    }
    await settle(renderer)
  })

  it('keeps the drawn fallback protected when its replacement cannot upload', async () => {
    // The fine regions have landed but their textures will not allocate.
    // Retention must judge them as the render plan does, or the coarse data
    // on screen would lose protection and, under cache pressure, be evicted
    // while the tile still has nothing else to show.
    const { renderer, gl } = await rendererAtZoom(0)
    await updateWithCover(renderer, viewportAt(0), gl, [FAR_EAST])
    await settle(renderer)
    seam(renderer).plannedDraws(gl)

    const fine = viewportAt(1)
    renderer.setCoveringTiles([NEAR_IN_FAR_EAST])
    renderer.update(fine, gl)
    await waitForActive(renderer, 1)
    renderer.update(fine, gl)
    await settle(renderer)

    const failing = fakeGl({ failTextures: true })
    renderer.update(fine, failing)
    const cache = seam(renderer).regionCache
    const tileBounds = getTileBounds(NEAR_IN_FAR_EAST.tileId)
    const coarse = [...cache.values()]
      .filter(
        (r) =>
          r.levelIndex === 0 && boundsIntersect(r.mercatorBounds!, tileBounds)
      )
      .map((r) => r.key)
    expect(coarse.length).toBeGreaterThan(0)
    for (const key of coarse) expect(cache.isProtected(key)).toBe(true)

    cache.evict(failing, 1)
    for (const key of coarse) expect(cache.get(key)).toBeDefined()
    const draws = seam(renderer).plannedDraws(failing)
    expect(draws.length).toBeGreaterThan(0)
    expect(draws.every((d) => d.region.levelIndex === 0)).toBe(true)
  })

  it('shows what a tile has of its own level when no other level covers it', async () => {
    const { renderer, map, gl } = await rendererAtZoom(1)
    await updateWithCover(renderer, map, gl, [NEAR])
    await settle(renderer)
    const cache = seam(renderer).regionCache
    // Regions the tile shows some of: wanted regions that only touch its
    // edge are never drawn.
    const tileBounds = getTileBounds(NEAR.tileId)
    const shown = [...seam(renderer).wantedKeys].filter((key) => {
      const m = cache.get(key)!.meshBounds!
      return (
        m.x0 < tileBounds.x1 &&
        m.x1 > tileBounds.x0 &&
        m.y0 < tileBounds.y1 &&
        m.y1 > tileBounds.y0
      )
    })
    expect(shown.length).toBeGreaterThan(1)
    cache.delete(shown[0])

    const draws = seam(renderer).plannedDraws(gl)
    expect(draws.map((d) => d.region.key).sort()).toEqual(shown.slice(1).sort())
  })

  it('reads a tile from the nearest coarser level when its own cannot resolve', async () => {
    // A level whose selector failed to resolve never opens under this
    // selector. Waiting on it would hold fallbacks for ever; the tile reads
    // the coarser level instead and counts as covered once that is drawn.
    const { renderer, map, gl } = await rendererAtZoom(0)
    failLevel(renderer, 1)

    renderer.setCoveringTiles([NEAR, FAR])
    renderer.update(map, gl)

    expect([...seam(renderer).wantedRegions.keys()]).toEqual([0])
    expect(seam(renderer).pendingLevels.size).toBe(0)
    // Level 0 column 8, row 12 lies under NEAR.
    expect(wantedAt(renderer, 0)).toContain('8,12')

    await settle(renderer)
    seam(renderer).plannedDraws(gl)
    renderer.update(map, gl)
    expect(seam(renderer).wantedCovered()).toBe(true)
  })

  it('keeps fetching the coarser level when the centre level cannot resolve', async () => {
    // The map zoom selects the failed level. Were that still the desired
    // level, the update would wait for it to open and never fetch; the
    // centre resolves to the coarser level instead, with and without a
    // tile cover.
    const { renderer, gl } = await rendererAtZoom(0)
    failLevel(renderer, 1)
    const fine = viewportAt(1)

    renderer.setCoveringTiles([NEAR])
    renderer.update(fine, gl)
    expect(seam(renderer).activeLevel?.index).toBe(0)
    expect(wantedAt(renderer, 0)).toContain('8,12')
    await settle(renderer)
    const cached = seam(renderer).regionCache.get(makeRegionKey(0, 8, 12))
    expect(cached && hasRegionData(cached)).toBe(true)

    renderer.setCoveringTiles(null)
    renderer.update(fine, gl)
    expect([...seam(renderer).wantedRegions.keys()]).toEqual([0])
    expect(wantedAt(renderer, 0).length).toBeGreaterThan(0)
  })

  it('targets the finest level a tile can want in full', async () => {
    // A tile whose own level has more regions under it than one update may
    // want could never be wanted in full, so it would never be promoted and
    // the same subset would be requested for ever. It targets the finest
    // coarser level that fits instead, and is shown from it once loaded.
    const { renderer, map, gl } = await rendererAtZoom(1)
    // Lon -135..-112.5, lat 0..21.9: some 150 level-1 regions, 45 level-0.
    const wide: TileRequest = { tileId: { z: 4, x: 2, y: 7 }, zoom: 1 }

    renderer.setCoveringTiles([wide])
    renderer.update(map, gl)
    await waitForRuntime(renderer, 0)
    renderer.update(map, gl)

    expect([...seam(renderer).wantedRegions.keys()]).toEqual([0])
    expect(wantedAt(renderer, 0).length).toBeLessThanOrEqual(MAX_CACHED_REGIONS)
    await settle(renderer)
    seam(renderer).plannedDraws(gl)
    const plan = seam(renderer).planTile(wide, () => true)
    expect(plan.ready).toBe(true)
    expect(plan.regions.length).toBe(wantedAt(renderer, 0).length)
    expect(plan.regions.every((r) => r.levelIndex === 0)).toBe(true)
  })

  it('draws nothing for a tile no level can serve', async () => {
    const { renderer, map, gl } = await rendererAtZoom(0)
    failLevel(renderer, 0)

    renderer.setCoveringTiles([FAR])
    renderer.update(map, gl)

    expect(seam(renderer).wantedKeys.size).toBe(0)
    expect(seam(renderer).pendingLevels.size).toBe(0)
    expect(renderer.noteDrawnTile(FAR, 1)).toBe(false)
    expect(seam(renderer).planTile(FAR, () => true)).toEqual({
      tile: FAR,
      regions: [],
      ready: true,
    })
  })

  it('keeps a polar region as the fallback of a polar tile on the globe', async () => {
    // Row 127 of level 1 lies wholly beyond the Mercator limit, so its
    // Mercator bounds clamp to zero height at y = 1. The globe draws the
    // tile to the pole, so while the tile's level changes the region must
    // still be shown and kept there, inside a box that reaches the pole.
    const { renderer, map, gl } = await rendererAtZoom(1)
    const polar: TileRequest = {
      tileId: { z: 4, x: 0, y: 15 },
      zoom: 1,
      poles: true,
    }
    await updateWithCover(renderer, map, gl, [polar])
    await settle(renderer)
    seam(renderer).plannedDraws(gl)
    const cache = seam(renderer).regionCache
    // Columns 0..7 lie inside the tile; column 8 only touches its edge.
    const polarKeys = [...cache.values()]
      .filter((r) => r.levelIndex === 1 && r.regionY === 127 && r.regionX < 8)
      .map((r) => r.key)
    expect(polarKeys.length).toBeGreaterThan(0)

    // The tile now reads level 0, which is still opening.
    renderer.setCoveringTiles([{ ...polar, zoom: 0 }])
    renderer.update(map, gl)
    expect(seam(renderer).pendingLevels.has(0)).toBe(true)
    for (const key of polarKeys) expect(cache.isProtected(key)).toBe(true)
    const draws = seam(renderer).plannedDraws(gl)
    const polarDraw = draws.find((d) => polarKeys.includes(d.region.key))
    expect(polarDraw).toBeDefined()
    expect(absoluteBoxes(polarDraw!)[0].y1).toBeGreaterThan(1)

    // Once level 0 is drawn there, the level-1 regions are released.
    await waitForRuntime(renderer, 0)
    renderer.update(map, gl)
    await settle(renderer)
    seam(renderer).plannedDraws(gl)
    renderer.update(map, gl)
    for (const key of polarKeys) expect(cache.isProtected(key)).toBe(false)
  })

  it('extends polar edge tiles to the pole when asked', async () => {
    // Level 1 rows are 1.4 degrees: row 127 lies wholly beyond the
    // Mercator limit, so only a pole-extended tile reaches it.
    const { renderer, map, gl } = await rendererAtZoom(1)
    const southEdge = { z: 4, x: 0, y: 15 }
    const rowsOf = () =>
      wantedAt(renderer, 1).map((k) => Number(k.split(',')[1]))

    await updateWithCover(renderer, map, gl, [{ tileId: southEdge, zoom: 1 }])
    expect(Math.max(...rowsOf())).toBeLessThan(126)

    await updateWithCover(renderer, map, gl, [
      { tileId: southEdge, zoom: 1, poles: true },
    ])
    expect(Math.max(...rowsOf())).toBe(127)
  })

  it('falls back to the viewport at the active level without a tile cover', async () => {
    // Mapbox exposes no public tile cover; without terrain its view is read
    // the way it always was.
    const { renderer, map, gl } = await rendererAtZoom(1)

    renderer.setCoveringTiles(null)
    renderer.update(map, gl)

    expect([...seam(renderer).wantedRegions.keys()]).toEqual([1])
    expect(wantedAt(renderer, 1).length).toBeGreaterThan(0)
  })
})

describe('tiles drawn by the draped hooks', () => {
  it('adds hook tiles to the cover at their own level', async () => {
    const { renderer, map, gl } = await rendererAtZoom(1)

    renderer.noteDrawnTile(FAR, 1)
    renderer.update(map, gl)
    await waitForRuntime(renderer, 0)
    renderer.update(map, gl)

    expect(wantedAt(renderer, 0).length).toBeGreaterThan(0)
  })

  it('keeps hook tiles for two revisions and drops them after', async () => {
    const { renderer, map, gl } = await rendererAtZoom(1)

    renderer.noteDrawnTile(NEAR, 1)
    renderer.noteDrawnTile(NEAR_EAST, 2)
    renderer.update(map, gl)
    const both = wantedAt(renderer, 1)

    renderer.noteDrawnTile(NEAR_EAST, 3)
    renderer.update(map, gl)
    const later = wantedAt(renderer, 1)

    expect(both.length).toBeGreaterThan(later.length)
    // With no tiles left, selection is the viewport at the active level,
    // which lies nowhere near the tiles.
    renderer.clearDrawnTiles()
    renderer.update(map, gl)
    const viewportOnly = wantedAt(renderer, 1)
    expect(later.every((k) => !viewportOnly.includes(k))).toBe(true)
  })

  it('reports whether a drawn tile still needs a fetch', async () => {
    const { renderer, map, gl } = await rendererAtZoom(1)

    expect(renderer.noteDrawnTile(NEAR, 1)).toBe(true)
    renderer.update(map, gl)
    // Queued by the update, so a frame is no longer needed for it.
    expect(renderer.noteDrawnTile(NEAR, 1)).toBe(false)
    await settle(renderer)
    expect(renderer.noteDrawnTile(NEAR, 2)).toBe(false)
  })
})

describe('single-resolution datasets', () => {
  it('caps the wanted regions at the cache size, nearest tiles first', async () => {
    // With no coarser level, far tiles map onto many regions of the only
    // level. Wanting more than the cache holds would make it thrash.
    const { renderer, gl } = await makeRenderer({ single: true })
    const map = viewportAt(0)
    renderer.update(map, gl)
    await settle(renderer)

    // Zoom-2 tiles over the northern hemisphere, each some 45 regions of the
    // 32x32 grid, listed far-to-near so priority has to reorder them.
    const tiles: TileRequest[] = []
    for (let x = 0; x < 4; x++) {
      tiles.push({ tileId: { z: 2, x, y: 1 }, zoom: 0 })
    }
    const nearest: TileRequest = { tileId: { z: 3, x: 1, y: 1 }, zoom: 2 }
    tiles.push(nearest)
    renderer.setCoveringTiles(tiles)
    renderer.update(map, gl)

    const wanted = seam(renderer).wantedKeys
    expect(wanted.size).toBe(MAX_CACHED_REGIONS)
    // The nearest tile (lon -135..-90, lat 41..67) covers region column 4..8,
    // rows 4..8; its regions are in despite being listed last.
    expect(wanted.has(makeRegionKey(0, 5, 5))).toBe(true)
  })

  it('shows what it has of the only level when a tile cannot want it in full', async () => {
    // With no coarser level to fall back to, the tile keeps its own level,
    // never counts as complete, and draws the regions the cap let through.
    const { renderer, gl } = await makeRenderer({ single: true })
    const map = viewportAt(0)
    renderer.update(map, gl)
    await settle(renderer)
    const world: TileRequest = { tileId: { z: 0, x: 0, y: 0 }, zoom: 0 }

    renderer.setCoveringTiles([world])
    renderer.update(map, gl)
    expect(seam(renderer).wantedKeys.size).toBe(MAX_CACHED_REGIONS)
    await settle(renderer)
    seam(renderer).plannedDraws(gl)

    const plan = seam(renderer).planTile(world, () => true)
    expect(plan.ready).toBe(false)
    expect(plan.regions.length).toBe(MAX_CACHED_REGIONS)
    expect(seam(renderer).plannedDraws(gl).length).toBeGreaterThan(0)
  })

  it('spends the cap on the regions nearest the map center, not the first rows', async () => {
    // One zoom-0 tile over a 32x32 grid: row-major order would fill the cap
    // with the top rows and never reach the viewport.
    const { renderer, gl } = await makeRenderer({ single: true })
    const map = {
      ...viewportAt(0),
      getCenter: () => ({ lng: 102.5, lat: -17.5 }),
    }
    renderer.update(map, gl)
    await settle(renderer)

    renderer.setCoveringTiles([{ tileId: { z: 0, x: 0, y: 0 }, zoom: 0 }])
    renderer.update(map, gl)

    const wanted = seam(renderer).wantedKeys
    expect(wanted.size).toBe(MAX_CACHED_REGIONS)
    // Column 25 (lon 101..112), row 19 (lat -17..-22) holds the center.
    expect(wanted.has(makeRegionKey(0, 25, 19))).toBe(true)
    expect(wanted.has(makeRegionKey(0, 0, 0))).toBe(false)
  })
})
