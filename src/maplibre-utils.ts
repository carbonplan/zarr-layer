/**
 * @module maplibre-utils
 *
 * Utility functions for MapLibre custom layer integration.
 * Provides tile management, zoom level conversion, and coordinate transformations.
 */

export type TileTuple = [number, number, number];

/**
 * Converts longitude to tile X coordinate at given zoom level.
 * @param lon - Longitude in degrees.
 * @param zoom - Zoom level.
 * @returns Tile X coordinate.
 */
export function lon2tile(lon: number, zoom: number): number {
  return Math.floor(((lon + 180) / 360) * Math.pow(2, zoom));
}

/**
 * Converts latitude to tile Y coordinate at given zoom level.
 * Uses Web Mercator projection.
 * @param lat - Latitude in degrees.
 * @param zoom - Zoom level.
 * @returns Tile Y coordinate.
 */
export function lat2tile(lat: number, zoom: number): number {
  return Math.floor(
    ((1 -
      Math.log(Math.tan((lat * Math.PI) / 180) + 1 / Math.cos((lat * Math.PI) / 180)) / Math.PI) /
      2) *
      Math.pow(2, zoom)
  );
}

/**
 * Gets all tiles visible at a given zoom level within geographic bounds.
 * Handles world wrap-around by normalizing tile indices to valid range.
 * Handles antimeridian crossing when west > east.
 * @param zoom - Zoom level.
 * @param bounds - Geographic bounds [[west, south], [east, north]].
 * @returns Array of tile tuples [zoom, x, y].
 */
export function getTilesAtZoom(
  zoom: number,
  bounds: [[number, number], [number, number]]
): TileTuple[] {
  const [[west, south], [east, north]] = bounds;
  let nwX = lon2tile(west, zoom);
  let seX = lon2tile(east, zoom);
  const nwY = lat2tile(north, zoom);
  const seY = lat2tile(south, zoom);

  const maxTiles = Math.pow(2, zoom);
  const tiles: TileTuple[] = [];
  const seenTiles = new Set<string>();

  if (nwX > seX) {
    seX += maxTiles;
  }

  for (let x = nwX; x <= seX; x++) {
    const wrappedX = ((x % maxTiles) + maxTiles) % maxTiles;
    for (let y = nwY; y <= seY; y++) {
      const clampedY = Math.max(0, Math.min(maxTiles - 1, y));
      const key = `${zoom},${wrappedX},${clampedY}`;
      if (!seenTiles.has(key)) {
        seenTiles.add(key);
        tiles.push([zoom, wrappedX, clampedY]);
      }
    }
  }

  return tiles;
}

/**
 * Converts tile tuple to cache key string.
 * @param tile - Tile tuple [zoom, x, y].
 * @returns String key "z,x,y".
 */
export function tileToKey(tile: TileTuple): string {
  return tile.join(',');
}

/**
 * Converts cache key to tile tuple.
 * @param key - String key "z,x,y".
 * @returns Tile tuple [zoom, x, y].
 */
export function keyToTile(key: string): TileTuple {
  return key.split(',').map(d => parseInt(d)) as TileTuple;
}

/**
 * Computes scale and shift parameters for positioning a tile in mercator coordinates.
 * Used in vertex shader to position tiles correctly on the map.
 *
 * Maps vertices from [-1, 1] clip space to [0, 1] mercator coordinate space.
 * For tile (z, x, y):
 *   - Left edge (vertex.x=-1) maps to x / 2^z
 *   - Right edge (vertex.x=1) maps to (x+1) / 2^z
 *   - Top edge (vertex.y=1) maps to y / 2^z (note: Y increases downward in web mercator)
 *   - Bottom edge (vertex.y=-1) maps to (y+1) / 2^z
 *
 * @param tile - Tile tuple [zoom, x, y].
 * @returns [scale, shiftX, shiftY] for vertex shader uniforms.
 */
export function tileToScale(tile: TileTuple): [number, number, number] {
  const [z, x, y] = tile;
  const scale = 1 / 2 ** (z + 1);
  const shiftX = (2 * x + 1) * scale;
  const shiftY = (2 * y + 1) * scale;
  return [scale, shiftX, shiftY];
}

/**
 * Converts map zoom level to pyramid/multiscale level.
 * Clamps zoom to valid range for the dataset.
 * @param zoom - Map zoom level.
 * @param maxZoom - Maximum zoom level available in dataset.
 * @returns Pyramid level (integer).
 */
export function zoomToLevel(zoom: number, maxZoom: number): number {
  if (maxZoom) return Math.min(Math.max(3, Math.floor(zoom)), maxZoom);
  return Math.max(0, Math.floor(zoom));
}

const MERCATOR_LAT_LIMIT = 85.05112878;

/**
 * Converts longitude in degrees to normalized Web Mercator X coordinate [0, 1].
 * Handles wraparound for longitudes outside -180 to 180 range.
 * @param lon - Longitude in degrees.
 * @returns Normalized mercator X coordinate.
 */
export function lonToMercatorNorm(lon: number): number {
  let normalizedLon = lon;
  if (lon > 180) {
    normalizedLon = lon - 360;
  } else if (lon < -180) {
    normalizedLon = lon + 360;
  }
  return (normalizedLon + 180) / 360;
}

/**
 * Converts latitude in degrees to normalized Web Mercator Y coordinate [0, 1].
 * Clamps latitude to valid Web Mercator range (±85.05112878°).
 * Note: Y=0 is at the north pole, Y=1 is at the south pole.
 * @param lat - Latitude in degrees.
 * @returns Normalized mercator Y coordinate.
 */
export function latToMercatorNorm(lat: number): number {
  const clamped = Math.max(-MERCATOR_LAT_LIMIT, Math.min(MERCATOR_LAT_LIMIT, lat));
  return (
    (1 -
      Math.log(Math.tan((clamped * Math.PI) / 180) + 1 / Math.cos((clamped * Math.PI) / 180)) /
        Math.PI) /
    2
  );
}

export interface MercatorBounds {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/**
 * Converts geographic bounds to normalized Web Mercator bounds [0, 1].
 * Handles both EPSG:4326 (lat/lon) and EPSG:3857 (already mercator) coordinate systems.
 * @param xyLimits - Geographic bounds { xMin, xMax, yMin, yMax }.
 * @param crs - Coordinate reference system ('EPSG:4326' or 'EPSG:3857').
 * @returns Normalized mercator bounds { x0, y0, x1, y1 }.
 */
export function boundsToMercatorNorm(
  xyLimits: { xMin: number; xMax: number; yMin: number; yMax: number },
  crs: 'EPSG:4326' | 'EPSG:3857' | null
): MercatorBounds {
  if (crs === 'EPSG:3857') {
    const worldExtent = 20037508.342789244;
    return {
      x0: (xyLimits.xMin + worldExtent) / (2 * worldExtent),
      y0: (worldExtent - xyLimits.yMax) / (2 * worldExtent),
      x1: (xyLimits.xMax + worldExtent) / (2 * worldExtent),
      y1: (worldExtent - xyLimits.yMin) / (2 * worldExtent)
    };
  }

  let yMin = xyLimits.yMin;
  let yMax = xyLimits.yMax;
  if (yMin > yMax) {
    [yMin, yMax] = [yMax, yMin];
  }

  return {
    x0: lonToMercatorNorm(xyLimits.xMin),
    y0: latToMercatorNorm(yMax),
    x1: lonToMercatorNorm(xyLimits.xMax),
    y1: latToMercatorNorm(yMin)
  };
}

/**
 * Computes scale and shift parameters for positioning an arbitrary bounds region in mercator coordinates.
 * Used in vertex shader to position non-tiled data correctly on the map.
 * Maps vertices from [-1, 1] clip space to the mercator bounds.
 * @param bounds - Normalized mercator bounds { x0, y0, x1, y1 }.
 * @returns [scale, shiftX, shiftY] for vertex shader uniforms.
 */
export function boundsToScale(bounds: MercatorBounds): [number, number, number] {
  const scaleX = (bounds.x1 - bounds.x0) / 2;
  const scaleY = (bounds.y1 - bounds.y0) / 2;
  const scale = Math.max(scaleX, scaleY);
  const shiftX = (bounds.x0 + bounds.x1) / 2;
  const shiftY = (bounds.y0 + bounds.y1) / 2;
  return [scale, shiftX, shiftY];
}
