/**
 * @module mapbox-utils
 *
 * Constants for the Mapbox-flavored shader path.
 */

/** Identity globe-to-Mercator matrix for tile rendering (no additional transformation) */
export const MAPBOX_IDENTITY_MATRIX = new Float32Array([
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
])
