/**
 * @module tile-renderer
 *
 * Rendering for the per-tile draping APIs: Mapbox's renderToTile and
 * MapLibre's renderToTerrainTile. Both ask the custom layer to draw one tile
 * into an offscreen texture, which needs:
 * - A tile-specific transformation matrix (not the camera matrix)
 * - Per-region projection handling (each loaded region is draped onto the
 *   requested tile).
 */

import type { ZarrRenderer } from './zarr-renderer'
import type { RenderContext, TileId, RegionRenderState } from './renderer-types'
import { setupBandTextureUniforms } from './render-helpers'
import { renderRegion, type RenderableRegion } from './renderable-region'
import { MAPBOX_IDENTITY_MATRIX } from './mapbox-utils'
import { createTileMatrix, getTileBounds, boundsIntersect } from './tile-utils'

// ============================================================================
// Main Rendering Functions
// ============================================================================

export interface TileRenderOptions {
  /**
   * True when the tile framebuffer puts the tile's north edge at clip y = 1
   * (MapLibre terrain tiles). Mapbox's renderToTile puts it at clip y = -1.
   */
  northUp: boolean
}

interface TileRenderParams extends TileRenderOptions {
  renderer: ZarrRenderer
  tileId: TileId
  context: RenderContext
  /** Loaded regions to drape onto the requested tile. */
  regions: RegionRenderState[]
}

/**
 * Renders Zarr data to a map tile by draping each loaded region that overlaps
 * the requested tile. Returns true if more data is needed (a region
 * intersecting the tile has not finished loading), false otherwise.
 */
export function renderTile({
  renderer,
  tileId,
  context,
  regions,
  northUp,
}: TileRenderParams): boolean {
  if (regions.length === 0) return true // Still loading

  const tileBounds = getTileBounds(tileId)

  const tileMatrix = createTileMatrix(
    tileBounds.x0,
    tileBounds.y0,
    tileBounds.x1,
    tileBounds.y1,
    northUp
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
    true, // useMapbox - always true for tile rendering
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
    true // tile render
  )

  setupBandTextureUniforms(renderer.gl, shaderProgram)

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
      bandTexture: region.bandTexture,
      bandTextureKey: region.bandTextureKey,
      bandTransform: region.bandTransform,
      indexBuffer: region.indexBuffer,
      meshBounds: region.meshBounds,
      latIsAscending: region.latIsAscending,
    }

    const rendered = renderRegion(
      renderer.gl,
      shaderProgram,
      renderable,
      [0], // Tiles don't need world wrapping
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
