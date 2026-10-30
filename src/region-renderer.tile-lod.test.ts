import { describe, it, expect, vi } from 'vitest'
import { RegionRenderer, type TileRequest } from './region-renderer'
import { ZarrStore } from './zarr-store'
import { buildMemoryZarrStore } from './__fixtures__/memory-zarr'
import {
  MAX_WANTED_REGIONS,
  hasRegionData,
  makeRegionKey,
  regionByteLength,
  type RegionCache,
  type RegionCacheLimits,
} from './region-cache'
import type { LevelLoader } from './level-loader'
import type { MapLike } from './types'
import { getRegionsInBounds, type RegionCoordinate } from './region-math'
import type { ProjectionContext } from './projection-utils'
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

/**
 * Regions one update may want here: the cache is held to twice that, so
 * the region budget, not bytes, binds on these small fixtures.
 */
const WANTED_REGIONS = 128

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
  viewportTiles(map: MapLike): TileRequest[]
  tileRegions(
    levelIndex: number,
    levelMeta: NonNullable<ReturnType<LevelLoader['runtime']>>,
    tile: TileRequest
  ): RegionCoordinate[]
  projection: ProjectionContext
  wantedCovered(): boolean
  tileLevels: Map<string, number>
  tileLevel(tile: TileRequest): number | null
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

/** A box cut to a region's mesh: what of it the region can draw. */
const withinMesh = (draw: Draw, box: Box): Box => {
  const mesh = draw.region.meshBounds!
  return {
    x0: Math.max(box.x0, mesh.x0),
    y0: Math.max(box.y0, mesh.y0),
    x1: Math.min(box.x1, mesh.x1),
    y1: Math.min(box.y1, mesh.y1),
  }
}

/**
 * A draw's clip boxes back in absolute Mercator of the base world, for the
 * passes drawn at one world offset, each cut to the region's mesh: a box
 * edge on the mesh edge is moved out, and only what lies inside the mesh
 * is drawn.
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
      boxes.push(
        withinMesh(draw, {
          x0: p[i] + ax,
          y0: p[i + 1] + ay,
          x1: p[i + 2] + ax,
          y1: p[i + 3] + ay,
        })
      )
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

async function makeRenderer({
  single = false,
  wide = false,
  strips = false,
  grown = false,
  limits = { maxRegions: 2 * WANTED_REGIONS },
}: {
  single?: boolean
  wide?: boolean
  strips?: boolean
  grown?: boolean
  limits?: Partial<RegionCacheLimits>
} = {}) {
  // `wide`: a global single level of 64x64 regions, 5.625 degrees each.
  // `strips`: a global single level of 72 pole-to-pole regions, 5° wide.
  // `grown`: a global single level of 64x192 pixels whose 32x64 chunks grow
  // to 128x128 regions, past the array: a 64x128 region west of lon 60 and
  // a 64x64 edge region east of it.
  const memory = grown
    ? buildMemoryZarrStore({
        arrays: [
          {
            name: 'temperature',
            shape: [64, 192],
            chunkShape: [32, 64],
            dimensionNames: ['lat', 'lon'],
          },
        ],
      })
    : strips
    ? buildMemoryZarrStore({
        arrays: [
          {
            name: 'temperature',
            shape: [2, 144],
            chunkShape: [2, 2],
            dimensionNames: ['lat', 'lon'],
          },
        ],
      })
    : wide
    ? buildMemoryZarrStore({
        arrays: [
          {
            name: 'temperature',
            shape: [2048, 4096],
            chunkShape: [32, 64],
            dimensionNames: ['lat', 'lon'],
          },
        ],
      })
    : single
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
  const renderer = new RegionRenderer(
    store,
    'temperature',
    {},
    invalidate,
    1,
    grown ? 128 : 0,
    undefined,
    limits
  )
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
    cache.evict(gl, { maxBytes: 1 })
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
      expect(
        sameBox(boxes[0], withinMesh(draw, getTileBounds(tile.tileId)))
      ).toBe(true)
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
          sameBox(
            absoluteBoxes(draw, 0)[0],
            withinMesh(draw, getTileBounds(FAR.tileId))
          )
        ).toBe(true)
      } else {
        expect(offsets).toEqual([-1])
        expect(
          sameBox(
            absoluteBoxes(draw, -1)[0],
            withinMesh(draw, getTileBounds(NEAR.tileId))
          )
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
      expect(
        sameBox(absoluteBoxes(draw)[0], withinMesh(draw, tileBounds))
      ).toBe(true)
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

    cache.evict(failing, { maxBytes: 1 })
    for (const key of coarse) expect(cache.get(key)).toBeDefined()
    const draws = seam(renderer).plannedDraws(failing)
    expect(draws.length).toBeGreaterThan(0)
    expect(draws.every((d) => d.region.levelIndex === 0)).toBe(true)
  })

  it('protects a released fallback again once a tile shows it', async () => {
    // A fine tile is drawn, so the coarse regions under it are released.
    // Panning to a neighbouring fine tile shows those coarse regions again
    // while its own regions load; they must be protected again too, or
    // cache pressure evicts them while they are on screen.
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
    seam(renderer).plannedDraws(gl)
    renderer.update(fine, gl)
    const cache = seam(renderer).regionCache
    const coarseKeys = [...cache]
      .map(([key]) => key)
      .filter((key) => key.startsWith('0:'))
    expect(coarseKeys.length).toBeGreaterThan(0)
    expect(coarseKeys.every((key) => !cache.isProtected(key))).toBe(true)

    const neighbour: TileRequest = { tileId: { z: 6, x: 10, y: 21 }, zoom: 1 }
    renderer.setCoveringTiles([neighbour])
    renderer.update(fine, gl)
    const draws = seam(renderer).plannedDraws(gl)
    expect(draws.length).toBeGreaterThan(0)
    expect(draws.every((d) => d.region.levelIndex === 0)).toBe(true)
    for (const draw of draws) {
      expect(cache.isProtected(draw.region.key)).toBe(true)
    }

    cache.evict(gl, { maxBytes: 1 })
    for (const draw of draws) expect(cache.get(draw.region.key)).toBeDefined()
    await settle(renderer)
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
    // Lon -135..-90, lat 0..41: some 480 level-1 regions, 120 level-0.
    const wide: TileRequest = { tileId: { z: 3, x: 1, y: 3 }, zoom: 1 }

    renderer.setCoveringTiles([wide])
    renderer.update(map, gl)
    await waitForRuntime(renderer, 0)
    renderer.update(map, gl)

    expect([...seam(renderer).wantedRegions.keys()]).toEqual([0])
    expect(wantedAt(renderer, 0).length).toBeLessThanOrEqual(WANTED_REGIONS)
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

  it('tiles the viewport itself without a cover, so levels never overlap', async () => {
    // Mapbox without terrain gives no cover. Zooming in mid-load, partial
    // fine data and the coarse fallback must not both draw whole over the
    // same pixels; the viewport is tiled at the map zoom instead and each
    // tile shows one level, so no box of one level overlaps another's.
    const { renderer, gl } = await rendererAtZoom(0)
    renderer.setCoveringTiles(null)
    renderer.update(viewportAt(0), gl)
    await settle(renderer)
    seam(renderer).plannedDraws(gl)

    const fine = mapAt(100, -20, 105, -15, 6)
    renderer.update(fine, gl)
    await waitForActive(renderer, 1)
    renderer.update(fine, gl)
    await settle(renderer)
    const cache = seam(renderer).regionCache
    const fineKeys = [...seam(renderer).wantedKeys]
    expect(fineKeys.length).toBeGreaterThan(1)
    cache.delete(fineKeys[0])
    renderer.update(fine, gl)

    const draws = seam(renderer).plannedDraws(gl)
    const levels = new Set(draws.map((d) => d.region.levelIndex))
    expect(levels).toEqual(new Set([0, 1]))
    // Boxes are float32 deltas from each region's own anchor, so a shared
    // edge lands within a few 1e-9 of itself across regions.
    const eps = 1e-6
    const overlaps = (a: Box, b: Box) =>
      a.x0 < b.x1 - eps &&
      a.x1 > b.x0 + eps &&
      a.y0 < b.y1 - eps &&
      a.y1 > b.y0 + eps
    for (const a of draws) {
      for (const b of draws) {
        if (a.region.levelIndex === b.region.levelIndex) continue
        for (const boxA of absoluteBoxes(a)) {
          for (const boxB of absoluteBoxes(b)) {
            expect(overlaps(boxA, boxB)).toBe(false)
          }
        }
      }
    }
    await settle(renderer)
  })

  it('covers every column of a nearly global viewport', async () => {
    // Both ends of a 355° viewport wrap into the same tile column. The
    // cover follows the unwrapped span, not the wrapped ends.
    const { renderer } = await rendererAtZoom(0)
    const tiles = seam(renderer).viewportTiles(mapAt(10, -85, 365, 85, 3))
    const z = tiles[0].tileId.z
    expect(new Set(tiles.map((t) => t.tileId.x)).size).toBe(2 ** z)
  })

  it('selects exactly the viewport regions without a cover', async () => {
    // Synthesised tiles overshoot the viewport; counting their whole
    // footprints would let regions off screen take the budget from visible
    // ones. Each tile is cut to the viewport, so the selection is the
    // viewport's regions, as a whole-viewport selection would pick them.
    const { renderer, gl } = await makeRenderer({ wide: true })
    const map = mapAt(-33, -17, 12, 17, 3)
    renderer.update(map, gl)
    await waitForActive(renderer, 0)
    renderer.setCoveringTiles(null)
    renderer.update(map, gl)

    const runtime = seam(renderer).levelLoader.runtime(0)!
    const viewport = getRegionsInBounds({
      bounds: { west: -33, south: -17, east: 12, north: 17 },
      xyLimits: { xMin: -180, xMax: 180, yMin: -90, yMax: 90 },
      levelMeta: runtime,
      projection: seam(renderer).projection,
      latIsAscending: false,
    })
      .map((r) => makeRegionKey(0, r.regionX, r.regionY))
      .sort()
    expect(viewport.length).toBeGreaterThan(0)
    expect([...seam(renderer).wantedKeys].sort()).toEqual(viewport)
  })

  it('selects every visible region of a viewport meeting a tile at both ends', async () => {
    // A view about 355° wide meets the tile column 0°–90° at both ends:
    // [-280, -270] is 80°–90° a turn west, and [0, 75] is the column itself.
    // The tile must own the regions under both parts.
    const { renderer, gl } = await makeRenderer({ strips: true })
    for (const [west, east] of [
      [-280, 75],
      [80, 435],
    ]) {
      const map = mapAt(west, -85, east, 85, 2)
      renderer.update(map, gl)
      await waitForActive(renderer, 0)
      renderer.setCoveringTiles(null)
      renderer.update(map, gl)

      const tiles = seam(renderer).viewportTiles(map)
      const ids = tiles.map((t) => `${t.tileId.z}/${t.tileId.x}/${t.tileId.y}`)
      expect(new Set(ids).size).toBe(ids.length)

      const runtime = seam(renderer).levelLoader.runtime(0)!
      const visible = getRegionsInBounds({
        bounds: { west, south: -85, east, north: 85 },
        xyLimits: { xMin: -180, xMax: 180, yMin: -90, yMax: 90 },
        levelMeta: runtime,
        projection: seam(renderer).projection,
        latIsAscending: false,
        strict: true,
      })
        .map((r) => makeRegionKey(0, r.regionX, r.regionY))
        .sort()
      expect(visible.length).toBe(71)
      expect([...seam(renderer).wantedKeys].sort()).toEqual(visible)
      await settle(renderer)
    }
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

describe('region budget across tiles', () => {
  it('coarsens the farthest tiles first until the whole cover can be wanted in full', async () => {
    // A tilted view at one tile zoom: twelve zoom-5 tiles (lon -135..-67.5,
    // lat 41..56) at zoom 1, the map center over the westmost. Each fits the
    // budget at level 1, but together they need more than one update may
    // want. Spending it nearest first would leave the far tiles with
    // nothing; the far tiles read level 0 instead and the near ones keep
    // level 1, so every tile is wanted in full.
    const { renderer, gl } = await rendererAtZoom(1)
    const map = { ...viewportAt(1), getCenter: () => ({ lng: -130, lat: 50 }) }
    const tiles: TileRequest[] = []
    for (let x = 4; x <= 9; x++) {
      for (let y = 10; y <= 11; y++) {
        tiles.push({ tileId: { z: 5, x, y }, zoom: 1 })
      }
    }
    const fine = seam(renderer).levelLoader.runtime(1)!
    const fineRegions = tiles.map((tile) =>
      seam(renderer).tileRegions(1, fine, tile)
    )
    expect(
      fineRegions.every((regions) => regions.length <= WANTED_REGIONS)
    ).toBe(true)
    expect(
      new Set(fineRegions.flat().map((r) => `${r.regionX},${r.regionY}`)).size
    ).toBeGreaterThan(WANTED_REGIONS)

    renderer.setCoveringTiles(tiles)
    renderer.update(map, gl)
    await waitForRuntime(renderer, 0)
    renderer.update(map, gl)
    const wanted = [...seam(renderer).wantedKeys].sort()
    renderer.update(map, gl)

    // Settled: a further update wants the same.
    expect([...seam(renderer).wantedKeys].sort()).toEqual(wanted)
    expect(wanted.length).toBeLessThanOrEqual(WANTED_REGIONS)
    const levels = tiles.map((tile) => seam(renderer).tileLevel(tile))
    expect(levels.slice(0, 2)).toEqual([1, 1])
    expect(levels.slice(-2)).toEqual([0, 0])
    // Inward from the far end, never finer farther out.
    for (let i = 1; i < levels.length; i++) {
      expect(levels[i]!).toBeLessThanOrEqual(levels[i - 1]!)
    }
    await settle(renderer)
    tiles.forEach((tile, i) => {
      const plan = seam(renderer).planTile(tile, () => true)
      expect(plan.ready).toBe(true)
      expect(plan.regions.length).toBeGreaterThan(0)
      expect(plan.regions.every((r) => r.levelIndex === levels[i])).toBe(true)
      expect(
        plan.regions.every((r) => seam(renderer).wantedKeys.has(r.key))
      ).toBe(true)
    })
  })
})

describe('region budget with hook tiles from an earlier view', () => {
  it('refits once the cover shrinks, whatever hook tiles remain', async () => {
    // A tilted view covers twelve zoom-5 tiles, too many regions at level 1,
    // so the fit drops some to level 0; the hooks draw the same tiles. The
    // view then shrinks to two tiles elsewhere that fit at level 1. Nothing
    // redraws the terrain, so the hook tiles stay; they must neither hold
    // the fit coarse nor take the budget from the cover.
    const { renderer, map, gl } = await rendererAtZoom(1)
    const wide: TileRequest[] = []
    for (let y = 10; y <= 11; y++) {
      for (let x = 4; x <= 9; x++) {
        wide.push({ tileId: { z: 5, x, y }, zoom: 1 })
      }
    }
    renderer.setCoveringTiles(wide)
    renderer.update(map, gl)
    await waitForRuntime(renderer, 0)
    renderer.update(map, gl)
    // Drawn at a finer zoom than the cover, so sorting by zoom alone would
    // put them first.
    const hooked = wide.map((tile) => ({ ...tile, zoom: 1.5 }))
    for (const tile of hooked.slice(0, 6)) renderer.noteDrawnTile(tile, 1)
    for (const tile of hooked.slice(6)) renderer.noteDrawnTile(tile, 2)
    renderer.update(map, gl)
    expect([...seam(renderer).wantedRegions.keys()]).toContain(0)
    await settle(renderer)

    // Lon -45..-22.5, lat 49..56: clear of the hook tiles.
    const narrow: TileRequest[] = [
      { tileId: { z: 5, x: 12, y: 10 }, zoom: 1 },
      { tileId: { z: 5, x: 13, y: 10 }, zoom: 1 },
    ]
    renderer.setCoveringTiles(narrow)
    renderer.update(map, gl)

    expect([...seam(renderer).tileLevels.values()]).toEqual([1, 1])
    expect(seam(renderer).wantedKeys.size).toBe(WANTED_REGIONS)
    const fine = seam(renderer).levelLoader.runtime(1)!
    for (const tile of narrow) {
      for (const { regionX, regionY } of seam(renderer).tileRegions(
        1,
        fine,
        tile
      )) {
        expect(
          seam(renderer).wantedKeys.has(makeRegionKey(1, regionX, regionY))
        ).toBe(true)
      }
    }
    await settle(renderer)
    for (const tile of narrow) {
      const plan = seam(renderer).planTile(tile, () => true)
      expect(plan.ready).toBe(true)
      expect(plan.regions.every((r) => r.levelIndex === 1)).toBe(true)
    }
    // Hook tiles get what budget the cover leaves, at the level the planner
    // gives them, and show a complete coarser level meanwhile.
    expect([...seam(renderer).wantedRegions.keys()]).toEqual([1])
    expect(seam(renderer).tileLevel(hooked[0])).toBe(1)
    expect(
      seam(renderer).planTile(hooked[0], () => true).regions.length
    ).toBeGreaterThan(0)

    renderer.setCoveringTiles(null)
    renderer.clearDrawnTiles()
    renderer.update({ getZoom: () => 1 } as MapLike, gl)
    expect(seam(renderer).tileLevels.size).toBe(0)
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

describe('region byte budget', () => {
  it('lets a dataset of small regions want more than the old region count', async () => {
    // Regions of 2x4 float32 pixels: the bytes allow far more than 128 of
    // them, so four zoom-2 tiles over the northern hemisphere are wanted in
    // full under the default limits.
    const { renderer, gl } = await makeRenderer({ single: true, limits: {} })
    const map = viewportAt(0)
    renderer.update(map, gl)
    await settle(renderer)
    const tiles: TileRequest[] = []
    for (let x = 0; x < 4; x++) {
      tiles.push({ tileId: { z: 2, x, y: 1 }, zoom: 0 })
    }
    const runtime = seam(renderer).levelLoader.runtime(0)!
    const all = new Set(
      tiles.flatMap((tile) =>
        seam(renderer)
          .tileRegions(0, runtime, tile)
          .map((r) => makeRegionKey(0, r.regionX, r.regionY))
      )
    )
    expect(all.size).toBeGreaterThan(128)
    expect(all.size).toBeLessThanOrEqual(MAX_WANTED_REGIONS)

    renderer.setCoveringTiles(tiles)
    renderer.update(map, gl)

    expect([...seam(renderer).wantedKeys].sort()).toEqual([...all].sort())
  })

  it('holds a dataset of large regions to the byte budget', async () => {
    // Regions of 32x64 float32 pixels, 8 KB of data and 8 KB of texture
    // each, against a 1 MB cache: an update may want half of it, 32
    // regions, well short of the region budget. The world tile needs all
    // 4096 regions of the only level.
    const { renderer, gl } = await makeRenderer({
      wide: true,
      limits: { maxBytes: 1024 * 1024 },
    })
    const map = { ...viewportAt(0), getCenter: () => ({ lng: 0, lat: 0 }) }
    renderer.update(map, gl)
    await settle(renderer)

    renderer.setCoveringTiles([{ tileId: { z: 0, x: 0, y: 0 }, zoom: 0 }])
    renderer.update(map, gl)
    expect(seam(renderer).wantedKeys.size).toBe(32)

    // The estimate the budget ran on matches what the regions hold once
    // drawn.
    await settle(renderer)
    seam(renderer).plannedDraws(gl)
    const cache = seam(renderer).regionCache
    let bytes = 0
    for (const key of seam(renderer).wantedKeys) {
      bytes += regionByteLength(cache.get(key)!)
    }
    expect(bytes).toBe(512 * 1024)
  })
})

describe('region byte estimates', () => {
  it('estimates regions grown past the array at the extent they hold', async () => {
    // Grown regions are 128x128 pixels, 128 KB each of float32 data and
    // texture, but the array holds only 64x128 and 64x64 of them, 96 KB
    // together. The update may want 112 KB: estimated at full size not even
    // the first region fits.
    const { renderer, gl } = await makeRenderer({
      grown: true,
      limits: { maxBytes: 2 * 112 * 1024 },
    })
    const map = viewportAt(0)
    renderer.update(map, gl)
    await settle(renderer)

    // The western half-world meets only the first region; the eastern
    // meets both.
    renderer.setCoveringTiles([
      { tileId: { z: 1, x: 0, y: 0 }, zoom: 0 },
      { tileId: { z: 1, x: 1, y: 0 }, zoom: 0 },
    ])
    renderer.update(map, gl)
    expect([...seam(renderer).wantedKeys].sort()).toEqual(['0:0,0', '0:1,0'])

    await settle(renderer)
    seam(renderer).plannedDraws(gl)
    const cache = seam(renderer).regionCache
    const bytes = [...seam(renderer).wantedKeys].map((key) =>
      regionByteLength(cache.get(key)!)
    )
    expect(bytes).toEqual([64 * 1024, 32 * 1024])
  })

  it('wants the single nearest region when it alone exceeds the budget', async () => {
    // 16 KB regions under the world tile against an update budget of 8 KB:
    // a blank view helps no one, so the region nearest the map center is
    // wanted, and nothing more. The world tile shows what it has.
    const world: TileRequest = { tileId: { z: 0, x: 0, y: 0 }, zoom: 0 }
    const { renderer, gl } = await makeRenderer({
      wide: true,
      limits: { maxBytes: 16 * 1024 },
    })
    // Column 32 (lon 0..5.6), row 31 (lat 0..2.8) holds the center.
    let map = { ...viewportAt(0), getCenter: () => ({ lng: 2, lat: 1 }) }
    renderer.update(map, gl)
    await settle(renderer)

    renderer.setCoveringTiles([world])
    renderer.update(map, gl)
    expect([...seam(renderer).wantedKeys]).toEqual(['0:32,31'])

    // Panning swaps the forced region; the one before is neither wanted nor
    // shown, so it is evicted rather than piling up.
    const cache = seam(renderer).regionCache
    for (const lng of [20, 40]) {
      await settle(renderer)
      expect(seam(renderer).plannedDraws(gl).length).toBe(1)
      map = { ...viewportAt(0), getCenter: () => ({ lng, lat: 1 }) }
      renderer.update(map, gl)
      expect(seam(renderer).wantedKeys.size).toBe(1)
      await settle(renderer)
      renderer.update(map, gl)
      expect([...cache.values()].map((r) => r.key)).toEqual([
        ...seam(renderer).wantedKeys,
      ])
    }
  })

  it('wants the next nearest region once the budget fits two', async () => {
    const world: TileRequest = { tileId: { z: 0, x: 0, y: 0 }, zoom: 0 }
    const { renderer, gl } = await makeRenderer({
      wide: true,
      limits: { maxBytes: 64 * 1024 },
    })
    const map = { ...viewportAt(0), getCenter: () => ({ lng: 2, lat: 1 }) }
    renderer.update(map, gl)
    await settle(renderer)

    renderer.setCoveringTiles([world])
    renderer.update(map, gl)
    // Row 32 (lat 0..-2.8) lies just south of the center.
    expect([...seam(renderer).wantedKeys].sort()).toEqual([
      '0:32,31',
      '0:32,32',
    ])
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
    expect(wanted.size).toBe(WANTED_REGIONS)
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
    expect(seam(renderer).wantedKeys.size).toBe(WANTED_REGIONS)
    await settle(renderer)
    seam(renderer).plannedDraws(gl)

    const plan = seam(renderer).planTile(world, () => true)
    expect(plan.ready).toBe(false)
    expect(plan.regions.length).toBe(WANTED_REGIONS)
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
    expect(wanted.size).toBe(WANTED_REGIONS)
    // Column 25 (lon 101..112), row 19 (lat -17..-22) holds the center.
    expect(wanted.has(makeRegionKey(0, 25, 19))).toBe(true)
    expect(wanted.has(makeRegionKey(0, 0, 0))).toBe(false)
  })
})
