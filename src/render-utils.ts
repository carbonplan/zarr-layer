import type { ProjectionData, ShaderData } from './shaders'
import type { MapLike } from './types'

interface ProjectionResolution {
  matrix: number[] | Float32Array | Float64Array | null
  shaderData?: ShaderData
  projectionData?: ProjectionData
  mapboxGlobe?:
    | {
        projection: { name: string }
        globeToMercatorMatrix: number[] | Float32Array | Float64Array
        transition: number
      }
    | undefined
}

export function resolveProjectionParams(
  params: unknown,
  projection?: { name: string },
  projectionToMercatorMatrix?: number[] | Float32Array | Float64Array,
  projectionToMercatorTransition?: number
): ProjectionResolution {
  type MatrixLike = number[] | Float32Array | Float64Array
  type ProjectionParams = {
    shaderData?: ShaderData
    defaultProjectionData?: {
      mainMatrix?: MatrixLike
      fallbackMatrix?: MatrixLike
      tileMercatorCoords?: number[]
      clippingPlane?: number[]
      projectionTransition?: number
    }
    modelViewProjectionMatrix?: MatrixLike
    projectionMatrix?: MatrixLike
  }

  const paramsObj =
    params &&
    typeof params === 'object' &&
    !Array.isArray(params) &&
    !ArrayBuffer.isView(params)
      ? (params as ProjectionParams)
      : null

  const shaderData = paramsObj?.shaderData
  let projectionData: ProjectionData | undefined
  const defaultProj = paramsObj?.defaultProjectionData
  if (
    defaultProj &&
    defaultProj.mainMatrix &&
    defaultProj.fallbackMatrix &&
    defaultProj.tileMercatorCoords &&
    defaultProj.clippingPlane &&
    typeof defaultProj.projectionTransition === 'number'
  ) {
    projectionData = {
      mainMatrix: defaultProj.mainMatrix,
      fallbackMatrix: defaultProj.fallbackMatrix,
      tileMercatorCoords: defaultProj.tileMercatorCoords as [
        number,
        number,
        number,
        number
      ],
      clippingPlane: defaultProj.clippingPlane as [
        number,
        number,
        number,
        number
      ],
      projectionTransition: defaultProj.projectionTransition,
    }
  }
  let matrix: number[] | Float32Array | Float64Array | null = null
  if (projectionData?.mainMatrix && projectionData.mainMatrix.length) {
    matrix = projectionData.mainMatrix
  } else if (
    Array.isArray(params) ||
    params instanceof Float32Array ||
    params instanceof Float64Array
  ) {
    matrix = params as number[] | Float32Array | Float64Array
  } else if (paramsObj?.modelViewProjectionMatrix) {
    matrix = paramsObj.modelViewProjectionMatrix
  } else if (paramsObj?.projectionMatrix) {
    matrix = paramsObj.projectionMatrix
  }

  const mapboxGlobe =
    projection && projectionToMercatorMatrix !== undefined
      ? {
          projection,
          globeToMercatorMatrix: projectionToMercatorMatrix,
          transition:
            typeof projectionToMercatorTransition === 'number'
              ? projectionToMercatorTransition
              : 0,
        }
      : undefined

  return { matrix, shaderData, projectionData, mapboxGlobe }
}

export function computeWorldOffsets(
  map: MapLike | null,
  isGlobe: boolean
): number[] {
  if (!map) return [0]

  const bounds = map.getBounds ? map.getBounds() : null
  if (!bounds) return [0]

  // Honor MapLibre's world copy setting, but always avoid duplicates on globe
  const renderWorldCopies =
    typeof map.getRenderWorldCopies === 'function'
      ? map.getRenderWorldCopies()
      : true
  if (isGlobe || !renderWorldCopies) return [0]

  const west = bounds.getWest()
  const east = bounds.getEast()

  const minWorld = Math.floor((west + 180) / 360)
  const maxWorld = Math.floor((east + 180) / 360)

  const worldOffsets: number[] = []
  for (let i = minWorld; i <= maxWorld; i++) {
    worldOffsets.push(i)
  }
  return worldOffsets.length > 0 ? worldOffsets : [0]
}
