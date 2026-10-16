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

function fakeGl() {
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
    createTexture: vi.fn(() => ({ tex: ++textures })),
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

type Seam = {
  regionCache: RegionCache
  wantedRegions: Map<number, RegionCoordinate[]>
  wantedKeys: Set<string>
  levelLoader: LevelLoader
  activeLevel: { index: number } | null
  getLoadedRegions(gl: WebGL2RenderingContext): RegionState[]
}
const seam = (renderer: RegionRenderer) => renderer as unknown as Seam

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

    const cache = seam(renderer).regionCache
    expect([...cache.values()].some((r) => r.loading)).toBe(true)
    for (const key of coarseKeys) {
      expect(cache.isProtected(key)).toBe(true)
    }
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
})
