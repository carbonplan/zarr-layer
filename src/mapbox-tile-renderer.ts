/**
 * @module mapbox-tile-renderer
 *
 * Specialized rendering for Mapbox GL JS tile API (renderToTile).
 * This is fundamentally different from MapLibre which uses projectTile().
 *
 * Mapbox's renderToTile() asks the custom layer to render individual tiles
 * to offscreen textures, requiring:
 * - Tile-specific transformation matrix (not camera matrix)
 * - Per-region projection handling (the unified region renderer drapes each
 *   loaded region onto the requested Mapbox tile).
 */

import type { ZarrRenderer } from './zarr-renderer'
import type { RenderContext, TileId, RegionRenderState } from './renderer-types'
import { setupBandTextureUniforms } from './render-helpers'
import { renderRegion, type RenderableRegion } from './renderable-region'
import {
  MAPBOX_IDENTITY_MATRIX,
  createMapboxTileMatrix,
  getMapboxTileBounds,
  boundsIntersect,
} from './mapbox-utils'

// ============================================================================
// Main Rendering Functions
// ============================================================================

interface MapboxTileRenderParams {
  renderer: ZarrRenderer
  tileId: TileId
  context: RenderContext
  /** Loaded regions to drape onto the requested Mapbox tile. */
  regions: RegionRenderState[]
}

/**
 * Renders Zarr data to a Mapbox globe tile by draping each loaded region that
 * overlaps the requested tile. Returns true if more data is needed (a region
 * intersecting the tile has not finished loading), false otherwise.
 */
export function renderMapboxTile({
  renderer,
  tileId,
  context,
  regions,
}: MapboxTileRenderParams): boolean {
  if (regions.length === 0) return true // Still loading

  const tileBounds = getMapboxTileBounds(tileId)

  const tileMatrix = createMapboxTileMatrix(
    tileBounds.x0,
    tileBounds.y0,
    tileBounds.x1,
    tileBounds.y1
  )

  const { colormapTexture, uniforms, customShaderConfig } = context

  // Determine if we're in globe mode (default true for backwards compatibility)
  const isGlobe = context.isGlobe ?? true

  // Check if any region uses source-projected mesh positions.
  const useWgs84 = regions.some((r) => !!r.wgs84Bounds)

  // Always use Mapbox globe shader for tile rendering - it handles both globe and mercator
  // via the transition uniform. The shader converts WGS84 → Mercator, then optionally
  // applies globe projection based on transition value.
  const shaderProgram = renderer.getProgram(
    context.shaderData,
    customShaderConfig,
    true, // useMapbox - always true for Mapbox tile rendering
    useWgs84
  )
  renderer.gl.useProgram(shaderProgram.program)
  renderer.applyCommonUniforms(
    shaderProgram,
    colormapTexture,
    uniforms,
    customShaderConfig,
    context.projectionData,
    {
      projection: { name: isGlobe ? 'globe' : 'mercator' },
      globeToMercatorMatrix: MAPBOX_IDENTITY_MATRIX,
      transition: isGlobe ? 0 : 1, // 0 = globe, 1 = mercator (blended)
    },
    tileMatrix,
    true // useMapbox
  )

  setupBandTextureUniforms(renderer.gl, shaderProgram, customShaderConfig)

  let needsMoreData = false
  for (const region of regions) {
    // Use mercatorBounds for tile intersection — always set and has the actual
    // per-region extent. wgs84Bounds carries the source-projected mesh anchor.
    if (!boundsIntersect(region.mercatorBounds, tileBounds)) continue

    // Source-projected regions may use an indexed adaptive mesh.
    const useIndexedMesh = !!region.useIndexedMesh && !!region.indexBuffer

    const renderable: RenderableRegion = {
      mercatorBounds: region.mercatorBounds,
      vertexBuffer: region.vertexBuffer,
      pixCoordBuffer: region.pixCoordBuffer,
      vertexCount: useIndexedMesh
        ? region.vertexCount ?? region.vertexArr.length / 2
        : region.vertexArr.length / 2,
      texture: region.texture,
      bandData: region.bandData ?? new Map(),
      bandTextures: region.bandTextures ?? new Map(),
      bandTexturesUploaded: region.bandTexturesUploaded ?? new Set(),
      bandTexturesConfigured: region.bandTexturesConfigured ?? new Set(),
      width: region.width,
      height: region.height,
      // Include indexed mesh fields for adaptive source-projected meshes.
      indexBuffer: useIndexedMesh ? region.indexBuffer : undefined,
      useIndexedMesh: useIndexedMesh,
      // Include wgs84Bounds for source-projected mesh scale/anchor uniforms.
      wgs84Bounds: region.wgs84Bounds,
      latIsAscending: region.latIsAscending,
    }

    const rendered = renderRegion(
      renderer.gl,
      shaderProgram,
      renderable,
      [0], // Globe tiles don't need world wrapping
      customShaderConfig,
      useWgs84 ? tileMatrix : null
    )
    if (!rendered) {
      // renderRegion returns false when band data is missing
      needsMoreData = true
    }
  }

  // Return true if any region still needs data (triggers re-render when loaded)
  return needsMoreData
}
