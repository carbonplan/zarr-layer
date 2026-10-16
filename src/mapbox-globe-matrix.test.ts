import { describe, it, expect } from 'vitest'
import fixture from './__fixtures__/mapbox-globe-matrices.json'
import {
  infiniteFarPerspective,
  mapboxEcefClipMatrix,
  multiplyMat4,
  parseMapboxRenderParameters,
} from './mapbox-globe-matrix'

/**
 * Mapbox draws its globe surface through a perspective whose far plane sits
 * at infinity, while the public render parameters carry the regular finite
 * far plane. The direct globe path rebuilds the infinite variant from the
 * public matrix; these tests pin that rebuild against matrices built the way
 * Mapbox builds them (gl-matrix perspectiveNO with off-center terms written
 * into m[8], m[9] afterwards), and show the resulting world-to-clip product
 * reproduces Mapbox's internal expandedFarZProjMatrix route exactly.
 */

const FOV = 0.6435011087932844
const WIDTH = 1280
const HEIGHT = 720
const ASPECT = WIDTH / HEIGHT
const NEAR = HEIGHT / 50
const FAR = 4567.89
const CENTER_OFFSET = { x: 37, y: -21 }

/** gl-matrix perspectiveNO, as bundled in mapbox-gl. */
function perspective(
  fov: number,
  aspect: number,
  near: number,
  far: number
): Float64Array {
  const out = new Float64Array(16)
  const f = 1 / Math.tan(fov / 2)
  out[0] = f / aspect
  out[5] = f
  out[11] = -1
  if (far !== Infinity) {
    const nf = 1 / (near - far)
    out[10] = (far + near) * nf
    out[14] = 2 * far * near * nf
  } else {
    out[10] = -1
    out[14] = -2 * near
  }
  return out
}

/** Mapbox's camera-to-clip: perspective plus the centerOffset shear. */
function mapboxCameraToClip(far: number): Float64Array {
  const m = perspective(FOV, ASPECT, NEAR, far)
  m[8] = (2 * -CENTER_OFFSET.x) / WIDTH
  m[9] = (2 * CENTER_OFFSET.y) / HEIGHT
  return m
}

const identity = () =>
  new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

const translation = (x: number, y: number, z: number) => {
  const m = identity()
  m[12] = x
  m[13] = y
  m[14] = z
  return m
}

const scaling = (x: number, y: number, z: number) => {
  const m = identity()
  m[0] = x
  m[5] = y
  m[10] = z
  return m
}

const rotationX = (rad: number) => {
  const m = identity()
  const c = Math.cos(rad)
  const s = Math.sin(rad)
  m[5] = c
  m[6] = s
  m[9] = -s
  m[10] = c
  return m
}

const rotationY = (rad: number) => {
  const m = identity()
  const c = Math.cos(rad)
  const s = Math.sin(rad)
  m[0] = c
  m[2] = -s
  m[8] = s
  m[10] = c
  return m
}

const rotationZ = (rad: number) => {
  const m = identity()
  const c = Math.cos(rad)
  const s = Math.sin(rad)
  m[0] = c
  m[1] = s
  m[4] = -s
  m[5] = c
  return m
}

const chain = (...ms: Float64Array[]) =>
  ms.reduce((acc, m) => multiplyMat4(acc, m))

function transformPoint(m: ArrayLike<number>, p: [number, number, number]) {
  const [x, y, z] = p
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
    m[3] * x + m[7] * y + m[11] * z + m[15],
  ]
}

const GLOBE_RADIUS = 8192 / (2 * Math.PI)

function ecef(lng: number, lat: number): [number, number, number] {
  const lngRad = (lng * Math.PI) / 180
  const latRad = (lat * Math.PI) / 180
  const cosLat = Math.cos(latRad)
  return [
    GLOBE_RADIUS * cosLat * Math.sin(lngRad),
    -GLOBE_RADIUS * Math.sin(latRad),
    GLOBE_RADIUS * cosLat * Math.cos(lngRad),
  ]
}

describe('infiniteFarPerspective', () => {
  const finite = mapboxCameraToClip(FAR)

  it('recovers the near plane from the finite matrix', () => {
    const rebuilt = infiniteFarPerspective(finite)
    expect(rebuilt[14] / -2).toBeCloseTo(NEAR, 10)
  })

  it('moves the far plane to infinity and keeps every other element', () => {
    const rebuilt = infiniteFarPerspective(finite)
    expect(rebuilt[10]).toBe(-1)
    for (let i = 0; i < 16; i++) {
      if (i === 10 || i === 14) continue
      expect(rebuilt[i]).toBe(finite[i])
    }
    // The off-center shear Mapbox writes in after building the perspective.
    expect(rebuilt[8]).toBe((2 * -CENTER_OFFSET.x) / WIDTH)
    expect(rebuilt[9]).toBe((2 * CENTER_OFFSET.y) / HEIGHT)
  })

  it('equals the perspective Mapbox builds directly with far = Infinity', () => {
    const rebuilt = infiniteFarPerspective(finite)
    const direct = mapboxCameraToClip(Infinity)
    for (let i = 0; i < 16; i++) {
      expect(rebuilt[i]).toBeCloseTo(direct[i], 12)
    }
  })

  it('is the limit of pushing the finite far plane out', () => {
    const rebuilt = infiniteFarPerspective(finite)
    const distant = mapboxCameraToClip(1e12)
    for (let i = 0; i < 16; i++) {
      expect(distant[i]).toBeCloseTo(rebuilt[i], 6)
    }
  })

  it('leaves an already-infinite perspective unchanged', () => {
    const direct = mapboxCameraToClip(Infinity)
    const again = infiniteFarPerspective(direct)
    for (let i = 0; i < 16; i++) {
      expect(again[i]).toBeCloseTo(direct[i], 12)
    }
  })

  it('does not mutate its input', () => {
    const copy = Float64Array.from(finite)
    infiniteFarPerspective(finite)
    expect(Array.from(finite)).toEqual(Array.from(copy))
  })
})

describe('mapboxEcefClipMatrix parity with the internal expandedFarZ route', () => {
  // A globe view at zoom 2.3, pitched and rotated, off-center. Mapbox's
  // private matrices are assembled here the way Transform._calcMatrices
  // does: expandedFarZProjMatrix = P_inf * worldToCamera, custom-layer
  // matrices scale by worldSize, globeToMercatorMatrix =
  // scale(1/worldSize) * globeMatrix, and the public viewMatrix on the globe
  // is worldToCamera * globeMatrix.
  const zoom = 2.3
  const worldSize = 512 * 2 ** zoom
  const center = { lng: 12.5, lat: -71.2 }
  const centerPixel = {
    x: ((center.lng + 180) / 360) * worldSize,
    y:
      ((1 -
        Math.log(Math.tan(Math.PI / 4 + (center.lat * Math.PI) / 180 / 2)) /
          Math.PI) /
        2) *
      worldSize,
  }
  const cameraDistance = (0.5 / Math.tan(FOV / 2)) * HEIGHT
  const pitch = 0.6
  const bearing = -0.4

  // World (mercator pixels, Y down) to camera, with Mapbox's Y flip.
  const worldToCamera = chain(
    scaling(1, -1, 1),
    translation(0, 0, -cameraDistance),
    rotationX(pitch),
    rotationZ(bearing),
    translation(-centerPixel.x, -centerPixel.y, 0)
  )

  // ECEF sphere (radius 8192 / 2π) to mercator pixel world.
  const globeMatrix = chain(
    translation(centerPixel.x, centerPixel.y, -worldSize / (2 * Math.PI)),
    scaling(worldSize / 8192, worldSize / 8192, worldSize / 8192),
    rotationX((-center.lat * Math.PI) / 180),
    rotationY((-center.lng * Math.PI) / 180)
  )

  const cameraToClip = mapboxCameraToClip(FAR)

  // Old route: expandedFarZProjMatrix scaled by worldSize, then
  // globeToMercatorMatrix, then the vertex.
  const expandedFarZProjMatrix = multiplyMat4(
    mapboxCameraToClip(Infinity),
    worldToCamera
  )
  const scaledByWorldSize = multiplyMat4(
    expandedFarZProjMatrix,
    scaling(worldSize, worldSize, worldSize)
  )
  const globeToMercator = multiplyMat4(
    scaling(1 / worldSize, 1 / worldSize, 1 / worldSize),
    globeMatrix
  )
  const oldMatrix = multiplyMat4(scaledByWorldSize, globeToMercator)

  // New route: the public projectionMatrix and viewMatrix only.
  const viewMatrix = multiplyMat4(worldToCamera, globeMatrix)
  const newMatrix = multiplyMat4(
    infiniteFarPerspective(cameraToClip),
    viewMatrix
  )

  const samples: Array<[number, number]> = [
    [center.lng, center.lat],
    [0, -90],
    [0, 90],
    [-45, -80],
    [170, -60],
    [12.5, -40],
    [-120, 10],
  ]

  it('projects globe-world points to the same clip coordinates (incl. z/w)', () => {
    for (const [lng, lat] of samples) {
      const p = ecef(lng, lat)
      const oldClip = transformPoint(oldMatrix, p)
      const newClip = transformPoint(newMatrix, p)
      for (let i = 0; i < 4; i++) {
        const tolerance = 1e-9 * Math.max(1, Math.abs(oldClip[i]))
        expect(Math.abs(oldClip[i] - newClip[i])).toBeLessThanOrEqual(tolerance)
      }
    }
  })

  it('differs from the finite-far product only in depth', () => {
    // The whole point of the rebuild: x, y, w match the public matrix product
    // while z follows the globe surface's infinite far plane.
    const finiteMatrix = multiplyMat4(cameraToClip, viewMatrix)
    for (const [lng, lat] of samples) {
      const p = ecef(lng, lat)
      const finiteClip = transformPoint(finiteMatrix, p)
      const newClip = transformPoint(newMatrix, p)
      for (const i of [0, 1, 3]) {
        expect(newClip[i]).toBeCloseTo(finiteClip[i], 9)
      }
      expect(newClip[2]).not.toBeCloseTo(finiteClip[2], 3)
    }
  })

  it('uploads the float64 product as float32', () => {
    const uploaded = mapboxEcefClipMatrix({
      projectionMatrix: cameraToClip,
      viewMatrix,
    })
    expect(uploaded).toBeInstanceOf(Float32Array)
    for (let i = 0; i < 16; i++) {
      expect(uploaded[i]).toBeCloseTo(Math.fround(newMatrix[i]), 12)
    }
  })
})

describe('parseMapboxRenderParameters', () => {
  const m = () => new Float64Array(16).fill(1)

  it('accepts the Mapbox render parameters object', () => {
    const parsed = parseMapboxRenderParameters({
      projectionMatrix: m(),
      viewMatrix: m(),
      globeClippingPlane: null,
      globeCenterInScreenPixels: null,
    })
    expect(parsed?.projectionMatrix).toHaveLength(16)
    expect(parsed?.viewMatrix).toHaveLength(16)
  })

  it('accepts plain arrays too', () => {
    const parsed = parseMapboxRenderParameters({
      projectionMatrix: Array.from(m()),
      viewMatrix: Array.from(m()),
    })
    expect(parsed).toBeDefined()
  })

  it('rejects what older Mapbox and MapLibre pass in that position', () => {
    expect(parseMapboxRenderParameters(undefined)).toBeUndefined()
    expect(parseMapboxRenderParameters(null)).toBeUndefined()
    expect(parseMapboxRenderParameters(0.5)).toBeUndefined()
    expect(parseMapboxRenderParameters(Array.from(m()))).toBeUndefined()
    expect(parseMapboxRenderParameters({})).toBeUndefined()
    expect(
      parseMapboxRenderParameters({ projectionMatrix: m() })
    ).toBeUndefined()
    expect(
      parseMapboxRenderParameters({
        projectionMatrix: m(),
        viewMatrix: new Float64Array(9),
      })
    ).toBeUndefined()
  })
})

describe('parity against matrices captured from mapbox-gl 3.32', () => {
  // Recorded in the demo from a real globe map: the public render parameters
  // (8th render argument) alongside the private expandedFarZProjMatrix,
  // worldSize and the globeToMercatorMatrix render argument, for several
  // camera poses including pitch, bearing and padded (off-center) views.
  const samples: Array<[number, number]> = [
    [0, -90],
    [0, 90],
    [-117, -76.5],
    [12.5, -71.2],
    [170, 20],
    [45, 60],
    [-30, 5],
    [100, -45],
  ]

  it('includes padded poses so the off-center terms are exercised', () => {
    expect(fixture.mapboxVersion).toBe('3.32.0')
    expect(
      fixture.captured.some(
        (c) => c.centerOffset[0] !== 0 || c.centerOffset[1] !== 0
      )
    ).toBe(true)
  })

  for (const c of fixture.captured) {
    const label = `zoom ${c.pose.zoom}, pitch ${c.pose.pitch}, bearing ${c.pose.bearing}, offset ${c.centerOffset}`

    it(`recovers Mapbox's near plane and off-center terms (${label})`, () => {
      const rebuilt = infiniteFarPerspective(c.projectionMatrix)
      expect(rebuilt[14] / -2).toBeCloseTo(c.nearZ, 9)
      expect(rebuilt[8]).toBeCloseTo((2 * -c.centerOffset[0]) / c.width, 12)
    })

    it(`projects globe points like the expandedFarZ route (${label})`, () => {
      const ws = c.worldSize
      const oldMatrix = multiplyMat4(
        multiplyMat4(c.expandedFarZProjMatrix, scaling(ws, ws, ws)),
        c.globeToMercatorMatrix
      )
      const newMatrix = multiplyMat4(
        infiniteFarPerspective(c.projectionMatrix),
        c.viewMatrix
      )
      for (const [lng, lat] of samples) {
        const p = ecef(lng, lat)
        const oldClip = transformPoint(oldMatrix, p)
        const newClip = transformPoint(newMatrix, p)
        for (let i = 0; i < 4; i++) {
          const tolerance = 1e-9 * Math.max(1, Math.abs(oldClip[i]))
          expect(Math.abs(oldClip[i] - newClip[i])).toBeLessThanOrEqual(
            tolerance
          )
        }
      }
    })
  }
})
