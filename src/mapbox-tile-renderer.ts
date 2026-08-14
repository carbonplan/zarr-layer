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

  // Check if any region uses source-projected mesh positions. This becomes
  // unconditional when the projection-mode cleanup removes the old variant.
  const useWgs84 = regions.some((r) => !!r.meshBounds)

  // Always use the Mapbox globe-capable shader for tile rendering. It restores
  // absolute Mercator positions from the region-local mesh, then optionally
  // applies globe projection based on the transition value.
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
    // per-region extent. meshBounds carries the source-projected mesh anchor.
    if (!boundsIntersect(region.mercatorBounds, tileBounds)) continue

    const renderable: RenderableRegion = {
      mercatorBounds: region.mercatorBounds,
      vertexBuffer: region.vertexBuffer,
      pixCoordBuffer: region.pixCoordBuffer,
      indexCount: region.indexCount,
      texture: region.texture,
      bandData: region.bandData ?? new Map(),
      bandTextures: region.bandTextures ?? new Map(),
      bandTexturesUploaded: region.bandTexturesUploaded ?? new Set(),
      bandTexturesConfigured: region.bandTexturesConfigured ?? new Set(),
      width: region.width,
      height: region.height,
      indexBuffer: region.indexBuffer,
      meshBounds: region.meshBounds,
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
      // A drawable region can still become incomplete at this boundary if a
      // texture or another required render resource is missing.
      needsMoreData = true
    }
  }

  // Return true if any region still needs data (triggers re-render when loaded)
  return needsMoreData
}
