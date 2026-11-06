import { describe, it, expect } from 'vitest'
import { viewportBounds } from './region-renderer'
import type { MapLike } from './types'
import type { LonLatBounds } from './region-math'

const WIDTH = 1000
const HEIGHT = 800

/**
 * A flat map centred on 5°E 5°N whose bounds, like Mapbox's, leave the
 * padded edges out. Screen offsets from the padded centre are turned by
 * `bearing` and bent by `warp`, standing in for pitch, then read as
 * degrees at 0.02° a pixel.
 */
const camera = ({
  padding = {},
  bearing = 0,
  warp = (y: number) => y,
}: {
  padding?: { top?: number; bottom?: number; left?: number; right?: number }
  bearing?: number
  warp?: (y: number) => number
}): MapLike => {
  const { top = 0, bottom = 0, left = 0, right = 0 } = padding
  const cx = left + (WIDTH - left - right) / 2
  const cy = top + (HEIGHT - top - bottom) / 2
  const angle = (bearing * Math.PI) / 180
  const unproject = ([x, y]: [number, number]) => {
    const dx = x - cx
    const dy = warp(y - cy)
    const east = dx * Math.cos(angle) - dy * Math.sin(angle)
    const north = -dx * Math.sin(angle) - dy * Math.cos(angle)
    return { lng: 5 + east * 0.02, lat: 5 + north * 0.02 }
  }
  const inner = [
    unproject([left, top]),
    unproject([WIDTH - right, top]),
    unproject([WIDTH - right, HEIGHT - bottom]),
    unproject([left, HEIGHT - bottom]),
  ]
  const lngs = inner.map((p) => p.lng)
  const lats = inner.map((p) => p.lat)
  const sw: [number, number] = [Math.min(...lngs), Math.min(...lats)]
  const ne: [number, number] = [Math.max(...lngs), Math.max(...lats)]
  return {
    getBounds: () => ({
      getWest: () => sw[0],
      getEast: () => ne[0],
      toArray: () => [sw, ne],
    }),
    getContainer: () => ({ clientWidth: WIDTH, clientHeight: HEIGHT }),
    getPadding: () => padding,
    unproject,
  }
}

const expectToContainScreen = (map: MapLike, bounds: LonLatBounds) => {
  for (const x of [0, WIDTH / 2, WIDTH]) {
    for (const y of [0, HEIGHT / 2, HEIGHT]) {
      const { lng, lat } = map.unproject!([x, y])
      expect(lng).toBeGreaterThanOrEqual(bounds.west - 1e-9)
      expect(lng).toBeLessThanOrEqual(bounds.east + 1e-9)
      expect(lat).toBeGreaterThanOrEqual(bounds.south - 1e-9)
      expect(lat).toBeLessThanOrEqual(bounds.north + 1e-9)
    }
  }
}

describe('viewportBounds', () => {
  it("is the map's bounds without padding", () => {
    const map = camera({})
    const [[west, south], [east, north]] = map.getBounds!()!.toArray!()
    const bounds = { west, south, east, north }
    expect(viewportBounds({ ...map, getPadding: undefined })).toEqual(bounds)
    expect(viewportBounds(map)).toEqual(bounds)
  })

  it('reaches into the padded edges the bounds leave out', () => {
    // 500 of 1000 pixels padded on the left: the bounds span 0° to 10°
    // over the other 500, so the padded half spans as much again.
    const bounds = viewportBounds(camera({ padding: { left: 500 } }))!
    expect(bounds.west).toBeCloseTo(-10)
    expect(bounds.east).toBeCloseTo(10)
    expect(bounds.south).toBeCloseTo(-3)
    expect(bounds.north).toBeCloseTo(13)
  })

  it('reaches the padded edges of a rotated view', () => {
    // Turned a quarter, east is up and the screen's left edge faces north.
    const map = camera({ padding: { left: 500 }, bearing: 90 })
    const inner = map.getBounds!()!.toArray!()
    const bounds = viewportBounds(map)!
    expect(bounds.north).toBeGreaterThan(inner[1][1] + 1)
    expectToContainScreen(map, bounds)
  })

  it('reaches the padded edges of a pitched view', () => {
    // Rows nearer the top show ground ever farther away.
    const map = camera({
      padding: { top: 300, left: 200 },
      bearing: 30,
      warp: (y) => y * (1 + Math.max(0, -y) / 200),
    })
    expectToContainScreen(map, viewportBounds(map)!)
  })

  it('carries east past west across the antimeridian', () => {
    const map: MapLike = {
      getBounds: () => ({
        getWest: () => 170,
        getEast: () => -170,
        toArray: () => [
          [170, 0],
          [-170, 10],
        ],
      }),
    }
    expect(viewportBounds(map)).toEqual({
      west: 170,
      south: 0,
      east: 190,
      north: 10,
    })
  })
})
