import { describe, it, expect } from 'vitest'
import {
  computeWorldOffsets,
  isGlobeProjection,
  resolveProjectionParams,
} from './map-utils'
import { MAPBOX_IDENTITY_MATRIX } from './mapbox-utils'
import type { MapLike } from './types'

/**
 * `resolveProjectionParams` is the single funnel that turns a render callback's
 * arguments into the renderer's projection state, and the two map libraries
 * call it with completely different shapes:
 *
 *   MapLibre  render(gl, args)          — args.shaderData + args.defaultProjectionData
 *   Mapbox    render(gl, matrix)        — mercator: a bare 16-float matrix
 *   Mapbox    render(gl, matrix, projection, toMercMatrix, transition) — globe
 *
 * Whether the result carries a `mapbox` block is what decides the whole shader
 * family downstream (see resolveProjectionMode), so provider detection failing
 * silently would put a MapLibre map on Mapbox shaders and render nothing.
 */

const MATRIX = Array.from({ length: 16 }, (_, i) => i)

const maplibreParams = (projectionTransition: number) => ({
  shaderData: {
    vertexShaderPrelude: '// prelude',
    define: '#define X 1',
    variantName: 'globe',
  },
  defaultProjectionData: {
    mainMatrix: MATRIX,
    fallbackMatrix: MATRIX,
    tileMercatorCoords: [0, 0, 1, 1],
    clippingPlane: [0, 0, 1, 0],
    projectionTransition,
  },
})

describe('isGlobeProjection', () => {
  it('reads the globe flag off either library’s projection object', () => {
    // Mapbox reports `name`, MapLibre reports `type`.
    expect(isGlobeProjection({ name: 'globe' })).toBe(true)
    expect(isGlobeProjection({ type: 'globe' })).toBe(true)
    expect(isGlobeProjection({ name: 'mercator' })).toBe(false)
    expect(isGlobeProjection({ type: 'mercator' })).toBe(false)
  })

  it('treats a missing projection as not-globe', () => {
    expect(isGlobeProjection(null)).toBe(false)
    expect(isGlobeProjection(undefined)).toBe(false)
    expect(isGlobeProjection({})).toBe(false)
  })
})

describe('resolveProjectionParams — MapLibre', () => {
  it('takes the matrix and projection data from the params object', () => {
    const resolved = resolveProjectionParams(maplibreParams(0))

    expect(resolved.matrix).toBe(MATRIX)
    expect(resolved.shaderData?.variantName).toBe('globe')
    expect(resolved.projectionData?.projectionTransition).toBe(0)
  })

  it('never produces a mapbox block, in mercator or globe', () => {
    // The mapbox block is the provider discriminator downstream; a MapLibre
    // frame acquiring one would select the Mapbox shader family.
    expect(resolveProjectionParams(maplibreParams(0)).mapbox).toBeUndefined()
    expect(resolveProjectionParams(maplibreParams(1)).mapbox).toBeUndefined()
  })

  it('carries the globe transition through, which is what activates ECEF', () => {
    expect(
      resolveProjectionParams(maplibreParams(0.42)).projectionData
        ?.projectionTransition
    ).toBe(0.42)
  })

  it('drops projection data that is missing any field it needs', () => {
    const params = maplibreParams(1) as {
      defaultProjectionData: Record<string, unknown>
    }
    delete params.defaultProjectionData.clippingPlane

    const resolved = resolveProjectionParams(params)

    // A partially-populated block would upload garbage clipping planes, so it
    // is refused wholesale rather than filled in.
    expect(resolved.projectionData).toBeUndefined()
    expect(resolved.matrix).toBeNull()
  })

  it('falls back to a legacy matrix-carrying params object', () => {
    expect(
      resolveProjectionParams({ modelViewProjectionMatrix: MATRIX }).matrix
    ).toBe(MATRIX)
    expect(resolveProjectionParams({ projectionMatrix: MATRIX }).matrix).toBe(
      MATRIX
    )
  })

  it('reports no matrix when params carry none, so render can bail', () => {
    expect(resolveProjectionParams({}).matrix).toBeNull()
    expect(resolveProjectionParams(undefined).matrix).toBeNull()
  })
})

describe('resolveProjectionParams — Mapbox', () => {
  it('treats a bare matrix as mapbox mercator', () => {
    const matrix = new Float64Array(MATRIX)

    const resolved = resolveProjectionParams(matrix)

    expect(resolved.matrix).toBe(matrix)
    expect(resolved.mapbox?.projection).toEqual({ name: 'mercator' })
    // Mercator is the transition=1 endpoint of the globe shader's blend, with
    // an identity globe->mercator matrix.
    expect(resolved.mapbox?.transition).toBe(1)
    expect(resolved.mapbox?.globeToMercatorMatrix).toBe(MAPBOX_IDENTITY_MATRIX)
    expect(resolved.projectionData).toBeUndefined()
  })

  it('detects mapbox from a plain-array matrix too', () => {
    expect(resolveProjectionParams(MATRIX).mapbox?.projection.name).toBe(
      'mercator'
    )
  })

  it('carries the globe morph arguments through', () => {
    const globeToMerc = new Float32Array(MATRIX)

    const resolved = resolveProjectionParams(
      MATRIX,
      { name: 'globe' },
      globeToMerc,
      0.37
    )

    expect(resolved.mapbox?.projection).toEqual({ name: 'globe' })
    expect(resolved.mapbox?.globeToMercatorMatrix).toBe(globeToMerc)
    expect(resolved.mapbox?.transition).toBe(0.37)
  })

  it('keeps transition 0 rather than defaulting it to mercator', () => {
    // 0 is the fully-globe endpoint. Treating it as absent would flatten the
    // globe to mercator for the whole low-zoom range.
    expect(
      resolveProjectionParams(MATRIX, { name: 'globe' }, MATRIX, 0).mapbox
        ?.transition
    ).toBe(0)
  })

  it('is detected from the projection argument even without a matrix', () => {
    expect(
      resolveProjectionParams(undefined, { name: 'globe' }).mapbox?.projection
        .name
    ).toBe('globe')
  })
})

describe('computeWorldOffsets', () => {
  const mapSpanning = (
    west: number,
    east: number,
    renderWorldCopies = true
  ): MapLike => ({
    getBounds: () => ({
      getWest: () => west,
      getEast: () => east,
      toArray: () => [
        [west, -85],
        [east, 85],
      ],
    }),
    getRenderWorldCopies: () => renderWorldCopies,
  })

  it('draws a single copy on the globe', () => {
    // The globe has no world copies; drawing at ±1 would stack duplicate
    // geometry on the same sphere.
    expect(computeWorldOffsets(mapSpanning(-180, 180), true)).toEqual([0])
    expect(computeWorldOffsets(mapSpanning(-200, 200), true)).toEqual([0])
  })

  it('draws a single copy when the map disables world copies', () => {
    expect(computeWorldOffsets(mapSpanning(-200, 200, false), false)).toEqual([
      0,
    ])
  })

  it('draws one copy for a mercator viewport inside a single world', () => {
    expect(computeWorldOffsets(mapSpanning(-50, 50), false)).toEqual([0])
  })

  it('draws both copies when a mercator viewport crosses a world seam', () => {
    expect(computeWorldOffsets(mapSpanning(100, 260), false)).toEqual([0, 1])
  })

  it('follows the viewport into a far world copy', () => {
    // Panning east repeatedly keeps unwrapping the bounds; the offsets have to
    // track it or the data stops drawing after a few wraps.
    expect(computeWorldOffsets(mapSpanning(900, 1000), false)).toEqual([3])
  })

  it('handles bounds reported wrapped across the antimeridian', () => {
    // getBounds() can report west > east when the viewport straddles ±180.
    expect(computeWorldOffsets(mapSpanning(170, -170), false)).toEqual([0, 1])
  })

  it('covers a viewport wider than one world', () => {
    expect(computeWorldOffsets(mapSpanning(-400, 400), false)).toEqual([
      -1, 0, 1,
    ])
  })

  it('falls back to one copy without a map or bounds', () => {
    expect(computeWorldOffsets(null, false)).toEqual([0])
    expect(computeWorldOffsets({ getBounds: () => null }, false)).toEqual([0])
  })

  it('assumes world copies when the map does not expose the setting', () => {
    const { getBounds } = mapSpanning(100, 260)
    expect(computeWorldOffsets({ getBounds }, false)).toEqual([0, 1])
  })
})
