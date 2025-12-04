import * as zarr from 'zarrita';
import { Readable } from '@zarrita/storage';

declare const Accent: (x: number) => number[];
declare const AccentR: (x: number) => number[];
declare const Blues: (x: number) => number[];
declare const BluesR: (x: number) => number[];
declare const BrBG: (x: number) => number[];
declare const BrBGR: (x: number) => number[];
declare const BuGn: (x: number) => number[];
declare const BuGnR: (x: number) => number[];
declare const BuPu: (x: number) => number[];
declare const BuPuR: (x: number) => number[];
declare const CMRmap: (x: number) => number[];
declare const CMRmapR: (x: number) => number[];
declare const Dark2: (x: number) => number[];
declare const Dark2R: (x: number) => number[];
declare const GnBu: (x: number) => number[];
declare const GnBuR: (x: number) => number[];
declare const Greens: (x: number) => number[];
declare const GreensR: (x: number) => number[];
declare const Greys: (x: number) => number[];
declare const GreysR: (x: number) => number[];
declare const OrRd: (x: number) => number[];
declare const OrRdR: (x: number) => number[];
declare const Oranges: (x: number) => number[];
declare const OrangesR: (x: number) => number[];
declare const PRGn: (x: number) => number[];
declare const PRGnR: (x: number) => number[];
declare const Paired: (x: number) => number[];
declare const PairedR: (x: number) => number[];
declare const Pastel1: (x: number) => number[];
declare const Pastel1R: (x: number) => number[];
declare const Pastel2: (x: number) => number[];
declare const Pastel2R: (x: number) => number[];
declare const PiYG: (x: number) => number[];
declare const PiYGR: (x: number) => number[];
declare const PuBu: (x: number) => number[];
declare const PuBuR: (x: number) => number[];
declare const PuBuGn: (x: number) => number[];
declare const PuBuGnR: (x: number) => number[];
declare const PuOr: (x: number) => number[];
declare const PuOrR: (x: number) => number[];
declare const PuRd: (x: number) => number[];
declare const PuRdR: (x: number) => number[];
declare const Purples: (x: number) => number[];
declare const PurplesR: (x: number) => number[];
declare const RdBu: (x: number) => number[];
declare const RdBuR: (x: number) => number[];
declare const RdGy: (x: number) => number[];
declare const RdGyR: (x: number) => number[];
declare const RdPu: (x: number) => number[];
declare const RdPuR: (x: number) => number[];
declare const RdYlBu: (x: number) => number[];
declare const RdYlBuR: (x: number) => number[];
declare const RdYlGn: (x: number) => number[];
declare const RdYlGnR: (x: number) => number[];
declare const Reds: (x: number) => number[];
declare const RedsR: (x: number) => number[];
declare const Set1: (x: number) => number[];
declare const Set1R: (x: number) => number[];
declare const Set2: (x: number) => number[];
declare const Set2R: (x: number) => number[];
declare const Set3: (x: number) => number[];
declare const Set3R: (x: number) => number[];
declare const Spectral: (x: number) => number[];
declare const SpectralR: (x: number) => number[];
declare const Wistia: (x: number) => number[];
declare const WistiaR: (x: number) => number[];
declare const YlGn: (x: number) => number[];
declare const YlGnR: (x: number) => number[];
declare const YlGnBu: (x: number) => number[];
declare const YlGnBuR: (x: number) => number[];
declare const YlOrBr: (x: number) => number[];
declare const YlOrBrR: (x: number) => number[];
declare const YlOrRd: (x: number) => number[];
declare const YlOrRdR: (x: number) => number[];
declare const afmhot: (x: number) => number[];
declare const afmhotR: (x: number) => number[];
declare const autumn: (x: number) => number[];
declare const autumnR: (x: number) => number[];
declare const binary: (x: number) => number[];
declare const binaryR: (x: number) => number[];
declare const bone: (x: number) => number[];
declare const boneR: (x: number) => number[];
declare const brg: (x: number) => number[];
declare const brgR: (x: number) => number[];
declare const bwr: (x: number) => number[];
declare const bwrR: (x: number) => number[];
declare const cividis: (x: number) => number[];
declare const cividisR: (x: number) => number[];
declare const cool: (x: number) => number[];
declare const coolR: (x: number) => number[];
declare const coolwarm: (x: number) => number[];
declare const coolwarmR: (x: number) => number[];
declare const copper: (x: number) => number[];
declare const copperR: (x: number) => number[];
declare const cubehelix: (x: number) => number[];
declare const cubehelixR: (x: number) => number[];
declare const flag: (x: number) => number[];
declare const flagR: (x: number) => number[];
declare const gistEarth: (x: number) => number[];
declare const gistEarthR: (x: number) => number[];
declare const gistGray: (x: number) => number[];
declare const gistGrayR: (x: number) => number[];
declare const gistHeat: (x: number) => number[];
declare const gistHeatR: (x: number) => number[];
declare const gistNcar: (x: number) => number[];
declare const gistNcarR: (x: number) => number[];
declare const gistRainbow: (x: number) => number[];
declare const gistRainbowR: (x: number) => number[];
declare const gistStern: (x: number) => number[];
declare const gistSternR: (x: number) => number[];
declare const gistYarg: (x: number) => number[];
declare const gistYargR: (x: number) => number[];
declare const gnuplot: (x: number) => number[];
declare const gnuplotR: (x: number) => number[];
declare const gnuplot2: (x: number) => number[];
declare const gnuplot2R: (x: number) => number[];
declare const gray: (x: number) => number[];
declare const grayR: (x: number) => number[];
declare const hot: (x: number) => number[];
declare const hotR: (x: number) => number[];
declare const hsv: (x: number) => number[];
declare const hsvR: (x: number) => number[];
declare const inferno: (x: number) => number[];
declare const infernoR: (x: number) => number[];
declare const jet: (x: number) => number[];
declare const jetR: (x: number) => number[];
declare const magma: (x: number) => number[];
declare const magmaR: (x: number) => number[];
declare const nipySpectral: (x: number) => number[];
declare const nipySpectralR: (x: number) => number[];
declare const ocean: (x: number) => number[];
declare const oceanR: (x: number) => number[];
declare const pink: (x: number) => number[];
declare const pinkR: (x: number) => number[];
declare const plasma: (x: number) => number[];
declare const plasmaR: (x: number) => number[];
declare const prism: (x: number) => number[];
declare const prismR: (x: number) => number[];
declare const rainbow: (x: number) => number[];
declare const rainbowR: (x: number) => number[];
declare const seismic: (x: number) => number[];
declare const seismicR: (x: number) => number[];
declare const spring: (x: number) => number[];
declare const springR: (x: number) => number[];
declare const summer: (x: number) => number[];
declare const summerR: (x: number) => number[];
declare const tab10: (x: number) => number[];
declare const tab10R: (x: number) => number[];
declare const tab20: (x: number) => number[];
declare const tab20R: (x: number) => number[];
declare const tab20b: (x: number) => number[];
declare const tab20bR: (x: number) => number[];
declare const tab20c: (x: number) => number[];
declare const tab20cR: (x: number) => number[];
declare const terrain: (x: number) => number[];
declare const terrainR: (x: number) => number[];
declare const turbo: (x: number) => number[];
declare const turboR: (x: number) => number[];
declare const twilight: (x: number) => number[];
declare const twilightR: (x: number) => number[];
declare const twilightShifted: (x: number) => number[];
declare const twilightShiftedR: (x: number) => number[];
declare const viridis: (x: number) => number[];
declare const viridisR: (x: number) => number[];
declare const winter: (x: number) => number[];
declare const winterR: (x: number) => number[];
/**
 * Returns a color scale function for a given colormap name.
 *
 * @remarks
 * This function is automatically generated for every color map and its reversed counterpart.
 * For instance, both `viridis` and `viridis_r` can be used to generate forward or reversed
 * colormap interpolators.
 *
 * @param color - Name of the color scale (e.g., `'viridis'`, `'coolwarm_r'`).
 * @returns A callable function that accepts a normalized value `x ∈ [0, 1]` and returns an RGB array.
 *
 * @example
 * ```ts
 * const viridis = colorScaleByName('viridis');
 * const rgb = viridis(0.5); // [r, g, b]
 * ```
 */
declare const colorScaleByName: (color: string) => (x: number) => number[];
/**
 * List of all available colormap names (including reversed versions).
 */
declare const allColorScales: readonly ["Accent", "Accent_r", "Blues", "Blues_r", "BrBG", "BrBG_r", "BuGn", "BuGn_r", "BuPu", "BuPu_r", "CMRmap", "CMRmap_r", "Dark2", "Dark2_r", "GnBu", "GnBu_r", "Greens", "Greens_r", "Greys", "Greys_r", "OrRd", "OrRd_r", "Oranges", "Oranges_r", "PRGn", "PRGn_r", "Paired", "Paired_r", "Pastel1", "Pastel1_r", "Pastel2", "Pastel2_r", "PiYG", "PiYG_r", "PuBu", "PuBuGn", "PuBuGn_r", "PuBu_r", "PuOr", "PuOr_r", "PuRd", "PuRd_r", "Purples", "Purples_r", "RdBu", "RdBu_r", "RdGy", "RdGy_r", "RdPu", "RdPu_r", "RdYlBu", "RdYlBu_r", "RdYlGn", "RdYlGn_r", "Reds", "Reds_r", "Set1", "Set1_r", "Set2", "Set2_r", "Set3", "Set3_r", "Spectral", "Spectral_r", "Wistia", "Wistia_r", "YlGn", "YlGnBu", "YlGnBu_r", "YlGn_r", "YlOrBr", "YlOrBr_r", "YlOrRd", "YlOrRd_r", "afmhot", "afmhot_r", "autumn", "autumn_r", "binary", "binary_r", "bone", "bone_r", "brg", "brg_r", "bwr", "bwr_r", "cividis", "cividis_r", "cool", "cool_r", "coolwarm", "coolwarm_r", "copper", "copper_r", "cubehelix", "cubehelix_r", "flag", "flag_r", "gist_earth", "gist_earth_r", "gist_gray", "gist_gray_r", "gist_heat", "gist_heat_r", "gist_ncar", "gist_ncar_r", "gist_rainbow", "gist_rainbow_r", "gist_stern", "gist_stern_r", "gist_yarg", "gist_yarg_r", "gnuplot", "gnuplot2", "gnuplot2_r", "gnuplot_r", "gray", "gray_r", "hot", "hot_r", "hsv", "hsv_r", "inferno", "inferno_r", "jet", "jet_r", "magma", "magma_r", "nipy_spectral", "nipy_spectral_r", "ocean", "ocean_r", "pink", "pink_r", "plasma", "plasma_r", "prism", "prism_r", "rainbow", "rainbow_r", "seismic", "seismic_r", "spring", "spring_r", "summer", "summer_r", "tab10", "tab10_r", "tab20", "tab20_r", "tab20b", "tab20b_r", "tab20c", "tab20c_r", "terrain", "terrain_r", "twilight", "twilight_r", "viridis", "viridis_r", "winter", "winter_r"];
/**
 * Builds a color ramp (discrete or continuous) from a specified colormap.
 *
 * @param color - Name of the colormap to use (e.g. `'viridis'`, `'RdBu_r'`).
 * @param convertTo - Optional output format (`'hex'` or `'css'`). Default is raw RGB arrays.
 * @param n - Number of color steps to generate (default: 255).
 * @param opacity - Opacity factor from 0 to 1 (default: 1).
 * @returns Array of colors in the selected format.
 *
 * @example
 * ```ts
 * // Generate a viridis ramp as CSS rgba() strings
 * const colors = colormapBuilder('viridis', 'css', 10, 0.8);
 * console.log(colors[0]); // "rgba(68,1,84,0.8)"
 * ```
 */
declare function colormapBuilder(color: string, convertTo?: string, n?: number, opacity?: number): string[] | number[][];

type ColorMapName = (typeof allColorScales)[number];
interface ColorMapInfo {
    [key: string]: {
        interpolate: boolean;
        colors: number[][];
    };
}
interface ZarrSelectorsProps {
    selected: number | number[] | string | string[] | [number, number];
    type?: 'index' | 'value';
}
interface XYLimits {
    xMin: number;
    xMax: number;
    yMin: number;
    yMax: number;
}
interface XYLimitsProps extends XYLimits {
}
interface ZarrLevelMetadata {
    width: number;
    height: number;
}
interface DimensionNamesProps {
    time?: string;
    elevation?: string;
    lat?: string;
    lon?: string;
    others?: string[];
}
interface DimIndicesProps {
    [key: string]: {
        name: string;
        index: number;
        array: zarr.Array<any> | null;
    };
}
interface MaplibreLayerOptions {
    id: string;
    source: string;
    variable: string;
    selector?: Record<string, number | number[] | string | string[]>;
    colormap?: ColorMapName | number[][] | string[];
    clim: [number, number];
    opacity?: number;
    minRenderZoom?: number;
    zarrVersion?: 2 | 3;
    dimensionNames?: DimensionNamesProps;
    fillValue?: number;
    customFragmentSource?: string;
    customFrag?: string;
    uniforms?: Record<string, number>;
    renderingMode?: '2d' | '3d';
}
type CRS = 'EPSG:4326' | 'EPSG:3857';
interface DataSliceProps {
    startX: number;
    endX: number;
    startY: number;
    endY: number;
    startElevation?: number;
    endElevation?: number;
}
interface SliceArgs {
    [key: number]: number | zarr.Slice;
}
interface ColorScaleProps {
    min: number;
    max: number;
    colors: number[][];
}

/**
 * @module zarr-layer
 *
 * MapLibre/MapBox custom layer implementation for rendering Zarr datasets.
 * Implements CustomLayerInterface for direct WebGL rendering.
 */

declare class ZarrLayer {
    readonly type: 'custom';
    readonly renderingMode: '2d' | '3d';
    id: string;
    private url;
    private variable;
    private zarrVersion;
    private dimensionNames;
    private selector;
    private invalidate;
    private colormap;
    private clim;
    private opacity;
    private minRenderZoom;
    private maxZoom;
    private tileSize;
    private isMultiscale;
    private fillValue;
    private scaleFactor;
    private offset;
    private gl;
    private map;
    private renderer;
    private dataManager;
    private applyWorldCopiesSetting;
    private initialRenderWorldCopies;
    private projectionChangeHandler;
    private resolveGl;
    private zarrStore;
    private levelInfos;
    private levelMetadata;
    private dimIndices;
    private xyLimits;
    private crs;
    private dimensionValues;
    private selectors;
    private isRemoved;
    private fragmentShaderSource;
    private customFrag;
    private customUniforms;
    private bandNames;
    private customShaderConfig;
    private isGlobeProjection;
    constructor({ id, source, variable, selector, colormap, clim, opacity, minRenderZoom, zarrVersion, dimensionNames, fillValue, customFragmentSource, customFrag, uniforms, renderingMode, }: MaplibreLayerOptions);
    setOpacity(opacity: number): void;
    setClim(clim: [number, number]): void;
    setColormap(colormap: ColorMapName | number[][] | string[]): void;
    setUniforms(uniforms: Record<string, number>): void;
    setVariable(variable: string): Promise<void>;
    setSelector(selector: Record<string, number | number[] | string | string[]>): Promise<void>;
    onAdd(map: any, gl: WebGL2RenderingContext): Promise<void>;
    private initializeManager;
    private initialize;
    private loadInitialDimensionValues;
    private getWorldOffsets;
    private getSelectorHash;
    prerender(_gl: WebGL2RenderingContext | WebGLRenderingContext, _params: any): void;
    render(_gl: WebGL2RenderingContext | WebGLRenderingContext, params: any, projection?: {
        name: string;
    }, globeToMercatorMatrix?: number[] | Float32Array | Float64Array, transition?: number): void;
    onRemove(_map: any, gl: WebGL2RenderingContext): void;
}

interface MultiscaleDataset {
    path: string;
    pixels_per_tile?: number;
    crs?: string;
}
interface Multiscale {
    datasets: MultiscaleDataset[];
}
interface ZarrV2ConsolidatedMetadata {
    metadata: Record<string, unknown>;
    zarr_consolidated_format?: number;
}
interface ZarrV3GroupMetadata {
    zarr_format: 3;
    node_type: 'group';
    attributes?: {
        multiscales?: Multiscale[];
    };
    consolidated_metadata?: {
        metadata?: Record<string, ZarrV3ArrayMetadata>;
    };
}
interface ZarrV3ArrayMetadata {
    zarr_format: 3;
    node_type: 'array';
    shape: number[];
    dimension_names?: string[];
    data_type?: string;
    fill_value: number | null;
    chunk_grid?: {
        configuration?: {
            chunk_shape?: number[];
        };
    };
    chunks?: number[];
    codecs?: Array<{
        name: string;
        configuration?: {
            chunk_shape?: number[];
        };
    }>;
    attributes?: {
        _ARRAY_DIMENSIONS?: string[];
        scale_factor?: number;
        add_offset?: number;
        _FillValue?: number;
        missing_value?: number;
    };
}
type ConsolidatedStore = zarr.Listable<zarr.FetchStore>;
type ZarrStoreType = zarr.FetchStore | ConsolidatedStore;
interface ZarrStoreOptions {
    source: string;
    version?: 2 | 3 | null;
    variable: string;
    dimensionNames?: DimensionNamesProps;
    coordinateKeys?: string[];
}
interface StoreDescription {
    metadata: ZarrV2ConsolidatedMetadata | ZarrV3GroupMetadata | null;
    dimensions: string[];
    shape: number[];
    chunks: number[];
    fill_value: number | null;
    dtype: string | null;
    levels: string[];
    maxZoom: number;
    tileSize: number;
    crs: CRS;
    dimIndices: DimIndicesProps;
    xyLimits: XYLimitsProps | null;
    scaleFactor: number;
    addOffset: number;
    coordinates: Record<string, (string | number)[]>;
}
declare class ZarrStore {
    private static _cache;
    private static _storeCache;
    source: string;
    version: 2 | 3 | null;
    variable: string;
    dimensionNames: DimensionNamesProps;
    coordinateKeys: string[];
    metadata: ZarrV2ConsolidatedMetadata | ZarrV3GroupMetadata | null;
    arrayMetadata: ZarrV3ArrayMetadata | null;
    dimensions: string[];
    shape: number[];
    chunks: number[];
    fill_value: number | null;
    dtype: string | null;
    levels: string[];
    maxZoom: number;
    tileSize: number;
    crs: CRS;
    dimIndices: DimIndicesProps;
    xyLimits: XYLimitsProps | null;
    scaleFactor: number;
    addOffset: number;
    coordinates: Record<string, (string | number)[]>;
    store: ZarrStoreType | null;
    root: zarr.Location<ZarrStoreType> | null;
    private _arrayHandles;
    initialized: Promise<this>;
    constructor({ source, version, variable, dimensionNames, coordinateKeys, }: ZarrStoreOptions);
    private _initialize;
    private _loadCoordinates;
    cleanup(): void;
    describe(): StoreDescription;
    getChunk(level: string, chunkIndices: number[]): Promise<zarr.Chunk<zarr.DataType>>;
    getLevelArray(level: string): Promise<zarr.Array<zarr.DataType, Readable>>;
    getArray(): Promise<zarr.Array<zarr.DataType, Readable>>;
    private _getArray;
    private _getJSON;
    private isConsolidatedStore;
    private _loadV2;
    private _loadV3;
    private _computeDimIndices;
    private _loadXYLimits;
    private _getPyramidMetadata;
    static clearCache(): void;
}

/**
 * @module maplibre-utils
 *
 * Utility functions for MapLibre custom layer integration.
 * Provides tile management, zoom level conversion, and coordinate transformations.
 */
type TileTuple = [number, number, number];
/**
 * Converts longitude to tile X coordinate at given zoom level.
 * @param lon - Longitude in degrees.
 * @param zoom - Zoom level.
 * @returns Tile X coordinate.
 */
declare function lon2tile(lon: number, zoom: number): number;
/**
 * Converts latitude to tile Y coordinate at given zoom level.
 * Uses Web Mercator projection.
 * @param lat - Latitude in degrees.
 * @param zoom - Zoom level.
 * @returns Tile Y coordinate.
 */
declare function lat2tile(lat: number, zoom: number): number;
/**
 * Gets all tiles visible at a given zoom level within geographic bounds.
 * Handles world wrap-around by normalizing tile indices to valid range.
 * Handles antimeridian crossing when west > east.
 * @param zoom - Zoom level.
 * @param bounds - Geographic bounds [[west, south], [east, north]].
 * @returns Array of tile tuples [zoom, x, y].
 */
declare function getTilesAtZoom(zoom: number, bounds: [[number, number], [number, number]]): TileTuple[];
/**
 * Converts tile tuple to cache key string.
 * @param tile - Tile tuple [zoom, x, y].
 * @returns String key "z,x,y".
 */
declare function tileToKey(tile: TileTuple): string;
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
declare function tileToScale(tile: TileTuple): [number, number, number];
/**
 * Converts map zoom level to pyramid/multiscale level.
 * Clamps zoom to valid range for the dataset.
 * @param zoom - Map zoom level.
 * @param maxZoom - Maximum zoom level available in dataset.
 * @returns Pyramid level (integer).
 */
declare function zoomToLevel(zoom: number, maxZoom: number): number;
/**
 * Converts longitude in degrees to normalized Web Mercator X coordinate [0, 1].
 * Handles wraparound for longitudes outside -180 to 180 range.
 * @param lon - Longitude in degrees.
 * @returns Normalized mercator X coordinate.
 */
declare function lonToMercatorNorm(lon: number): number;
/**
 * Converts latitude in degrees to normalized Web Mercator Y coordinate [0, 1].
 * Clamps latitude to valid Web Mercator range (±85.05112878°).
 * Note: Y=0 is at the north pole, Y=1 is at the south pole.
 * @param lat - Latitude in degrees.
 * @returns Normalized mercator Y coordinate.
 */
declare function latToMercatorNorm(lat: number): number;
interface MercatorBounds {
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
declare function boundsToMercatorNorm(xyLimits: {
    xMin: number;
    xMax: number;
    yMin: number;
    yMax: number;
}, crs: 'EPSG:4326' | 'EPSG:3857' | null): MercatorBounds;

/**
 * @module maplibre-shaders
 *
 * WebGL shaders for MapLibre/MapBox custom layer rendering.
 * Adapted from zarr-gl to work with zarr-cesium's colormap and nodata handling.
 */
interface ShaderData {
    vertexShaderPrelude: string;
    define: string;
    variantName: string;
}
interface ProjectionData {
    mainMatrix: Float32Array | Float64Array | number[];
    fallbackMatrix: Float32Array | Float64Array | number[];
    tileMercatorCoords: [number, number, number, number];
    clippingPlane: [number, number, number, number];
    projectionTransition: number;
}
declare function createVertexShaderSource(shaderData?: ShaderData): string;
/**
 * Vertex shader for tile rendering (mercator fallback).
 * Transforms tile vertices using scale, shift, and projection matrix uniforms.
 * Vertices are in [-1, 1] and represent a full tile quad.
 * Scale and shift position the tile in mercator [0, 1] space.
 *
 * Note: Y is negated because vertex Y increases upward (+1 is top)
 * but mercator Y increases downward (0 is north, 1 is south).
 */
declare const maplibreVertexShaderSource = "#version 300 es\nuniform float scale;\nuniform float scale_x;\nuniform float scale_y;\nuniform float shift_x;\nuniform float shift_y;\nuniform float u_worldXOffset;\nuniform mat4 matrix;\n\nin vec2 pix_coord_in;\nin vec2 vertex;\n\nout vec2 pix_coord;\n\nvoid main() {\n  float sx = scale_x > 0.0 ? scale_x : scale;\n  float sy = scale_y > 0.0 ? scale_y : scale;\n  vec2 a = vec2(vertex.x * sx + shift_x + u_worldXOffset, -vertex.y * sy + shift_y);\n  gl_Position = matrix * vec4(a, 0.0, 1.0);\n  pix_coord = pix_coord_in;\n}\n";
/**
 * Fragment shader for tile rendering with colormap and fillValue handling.
 * Mirrors carbonplan/maps approach with clim (vec2) and single fillValue.
 */
declare const maplibreFragmentShaderSource = "#version 300 es\nprecision highp float;\n\nuniform vec2 clim;\nuniform float opacity;\nuniform float fillValue;\nuniform float u_scaleFactor;\nuniform float u_addOffset;\nuniform vec2 u_texScale;\nuniform vec2 u_texOffset;\n\nuniform sampler2D tex;\nuniform sampler2D cmap;\n\nin vec2 pix_coord;\nout vec4 color;\n\nvoid main() {\n  vec2 sample_coord = pix_coord * u_texScale + u_texOffset;\n  float raw = texture(tex, sample_coord).r;\n  float value = raw * u_scaleFactor + u_addOffset;\n  \n  if (raw == fillValue || raw != raw || value != value) {\n    discard;\n  }\n  \n  float rescaled = (value - clim.x) / (clim.y - clim.x);\n  vec4 c = texture(cmap, vec2(rescaled, 0.5));\n  color = vec4(c.rgb, opacity);\n  color.rgb *= color.a;\n}\n";
/**
 * Simple vertex shader for rendering framebuffer to screen.
 */
declare const renderVertexShaderSource = "#version 300 es\nin vec2 vertex;\nout vec2 texCoord;\nvoid main() {\n  gl_Position = vec4(vertex, 0.0, 1.0);\n  texCoord = vertex * 0.5 + 0.5;\n}\n";
/**
 * Simple fragment shader for rendering framebuffer texture to screen.
 */
declare const renderFragmentShaderSource = "#version 300 es\nprecision highp float;\nuniform sampler2D tex;\nin vec2 texCoord;\nout vec4 fragColor;\nvoid main() {\n  fragColor = texture(tex, texCoord);\n}\n";
interface FragmentShaderOptions {
    bands: string[];
    customUniforms?: string[];
    customFrag?: string;
}
declare function createFragmentShaderSource(options: FragmentShaderOptions): string;

/**
 * @module gl-utils
 *
 * Low-level WebGL2 utility functions for shader creation, program linking,
 * and generation of color ramp textures used in Cesium and Zarr visualization.
 *
 * These helpers are used internally by rendering providers (e.g., {@link ZarrCubeProvider})
 * to generate color-mapped textures and dynamic shader programs.
 */
/**
 * Creates and compiles a WebGL shader from source code.
 *
 * @param gl - The WebGL2 rendering context.
 * @param type - Shader type (`gl.VERTEX_SHADER` or `gl.FRAGMENT_SHADER`).
 * @param source - GLSL source code for the shader.
 * @returns The compiled {@link WebGLShader} instance, or `null` if compilation failed.
 *
 * @example
 * ```ts
 * const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
 * const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
 * ```
 */
declare function createShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader | null;
/**
 * Creates and links a WebGL program using the specified vertex and fragment shaders.
 *
 * @param gl - The WebGL2 rendering context.
 * @param vertexShader - Compiled vertex shader.
 * @param fragmentShader - Compiled fragment shader.
 * @returns The linked {@link WebGLProgram}, or `null` if linking failed.
 *
 * @example
 * ```ts
 * const program = createProgram(gl, vertexShader, fragmentShader);
 * gl.useProgram(program);
 * ```
 */
declare function createProgram(gl: WebGL2RenderingContext, vertexShader: WebGLShader, fragmentShader: WebGLShader): WebGLProgram | null;
/**
 * Creates a flexible 1D color-ramp texture supporting either normalized (0–1)
 * or integer (0–255) color definitions.
 *
 * @param gl - The WebGL2 rendering context.
 * @param colors - Array of RGB colors in normalized `[0–1]` or integer `[0–255]` format.
 * @param opacity - Opacity multiplier between 0 and 1.
 * @returns A {@link WebGLTexture} representing the color ramp, or `null` if creation failed.
 *
 * @example
 * ```ts
 * const texture = createColorRampTexture(gl, [[1, 0, 0], [0, 0, 1]], 0.8);
 * ```
 */
declare function createColorRampTexture(gl: WebGL2RenderingContext, colors: number[][], opacity: number): WebGLTexture | null;
/**
 * Utility to fetch a uniform location with a helpful error if missing.
 */
declare function mustGetUniformLocation(gl: WebGL2RenderingContext, program: WebGLProgram, name: string): WebGLUniformLocation;
/**
 * Utility to create a texture or throw.
 */
declare function mustCreateTexture(gl: WebGL2RenderingContext): WebGLTexture;
/**
 * Utility to create a buffer or throw.
 */
declare function mustCreateBuffer(gl: WebGL2RenderingContext): WebGLBuffer;

/**
 * @module zarr-utils
 *
 * Utility functions for reading, interpreting, and slicing Zarr datasets
 * used by Cesium visualization components (e.g., {@link ZarrCubeProvider},
 * {@link ZarrCubeVelocityProvider}, {@link ZarrLayerProvider}).
 *
 * Provides:
 * - Dimension detection and CF-compliant alias mapping
 * - Slice generation for multidimensional Zarr arrays
 * - Multiscale (pyramidal) dataset handling
 * - CRS detection and coordinate transformation utilities
 * - Calculation of vertical exaggeration and Cesium-compatible XY indices
 */

/**
 * Identify the indices of common dimensions (lat, lon, time, elevation)
 * in a Zarr array, optionally using CF-compliant standard names or custom dimension mappings.
 *
 * @param dimNames - Names of the array dimensions.
 * @param dimensionNames - Optional explicit mapping of dimension names (see {@link DimensionNamesProps}).
 * @param coordinates - Optional coordinate variable dictionary.
 * @returns A {@link DimIndicesProps} object describing each dimension’s index and name.
 */
declare function identifyDimensionIndices(dimNames: string[], dimensionNames?: DimensionNamesProps, coordinates?: Record<string, any>): DimIndicesProps;
/**
 * Finds the index of the value in `values` nearest to `target`.
 * @param values - Array of numeric values.
 * @param target - Target value to find.
 * @returns Index of the nearest value.
 */
declare function calculateNearestIndex(values: Float64Array | number[], target: number): number;
/**
 * Loads the coordinate values for a specific dimension.
 *
 * Behavior:
 * - Uses cached values if available (does not reload unless the caller resets the cache).
 * - Resolves the correct multiscale level if `levelInfo` is provided.
 * - Converts Zarr buffers into plain JavaScript number arrays.
 * - Converts bigint values to number.
 * - If a slice `[start, end]` is supplied, only a sub-range is returned.
 *
 * @param dimensionValues  Cache of already-loaded coordinate arrays.
 * @param levelInfo        Optional multiscale subpath.
 * @param dimIndices      Dimension index info. See {@link DimIndicesProps}.
 * @param root            Root Zarr group location.
 * @param zarrVersion     Zarr version (2 or 3).
 * @param slice           Optional index range `[start, end]` to slice the loaded values.
 *
 * @returns The loaded coordinate array for the dimension.
 */
declare function loadDimensionValues(dimensionValues: Record<string, Float64Array | number[]>, levelInfo: string | null, dimIndices: DimIndicesProps[string], root: zarr.Location<zarr.FetchStore>, zarrVersion: 2 | 3 | null, slice?: [number, number]): Promise<Float64Array | number[]>;
/**
 * Opens a Zarr variable (single-scale or multiscale pyramid) and prepares its metadata.
 *
 * - Detects and loads multiscale dataset levels (if present).
 * - Computes per-level dimension sizes and stores them in `levelMetadata`.
 * - Scans coordinate variables from `_ARRAY_DIMENSIONS` or consolidated metadata.
 * - Detects CF/alias-based dimension names (lat/lon/time/elevation).
 *
 * @param store             Zarr store (e.g., `FetchStore`).
 * @param root              Root Zarr group location.
 * @param variable          Variable name within the Zarr group.
 * @param dimensions        Optional explicit dimension name mapping. See {@link DimensionNamesProps}.
 * @param levelMetadata     Map to populate with per-level metadata (width/height).
 * @param levelCache        Cache for opened multiscale level arrays.
 * @param zarrVersion      Zarr version (2 or 3).
 * @param multiscaleLevel   Optional initial multiscale level to open.
 *
 * @returns
 *   - `zarrArray` — the opened array for the selected multiscale level.
 *   - `levelInfos` — all multiscale level paths.
 *   - `dimIndices` — discovered dimension index mapping. See {@link DimIndicesProps}.
 *   - `attrs` — variable or group attributes.
 *   - `multiscaleLevel` — updated level if adjusted due to missing levels.
 */
declare function initZarrDataset(store: zarr.FetchStore, root: zarr.Location<zarr.FetchStore>, variable: string, dimensions: DimensionNamesProps, levelMetadata: Map<number, ZarrLevelMetadata>, levelCache: Map<number, any>, zarrVersion: 2 | 3 | null, multiscaleLevel?: number): Promise<{
    zarrArray: zarr.Array<any>;
    levelInfos: string[];
    dimIndices: DimIndicesProps;
    attrs: Record<string, any>;
    multiscaleLevel?: number;
}>;
/**
 * Retrieve the geographic coordinate limits (min/max latitude/longitude) for a Zarr array.
 *
 * @param root - Zarr group root.
 * @param dimIndices - Dimension mapping. See {@link DimIndicesProps}.
 * @param levelInfos - Multiscale level paths.
 * @param multiscale - Whether the dataset is multiscale.
 * @param zarrVersion - Zarr version (2 or 3).
 *
 * @returns A {@link XYLimitsProps} object describing the coordinate bounds.
 */
declare function getXYLimits(root: zarr.Location<zarr.FetchStore>, dimIndices: DimIndicesProps, levelInfos: string[], multiscale: boolean, zarrVersion: 2 | 3 | null): Promise<XYLimitsProps>;
/**
 * Opens and caches a specific multiscale level array.
 * Keeps a small LRU-style cache of up to three levels.
 *
 * @param root        Zarr group root.
 * @param levelPath   Path to the multiscale level.
 * @param variable    Variable name within the level (if any).
 * @param levelCache Cache of opened level arrays.
 * @param zarrVersion Zarr version (2 or 3).
 *
 * @returns The opened Zarr array for the specified level.
 */
declare function openLevelArray(root: zarr.Location<zarr.FetchStore>, levelPath: string, variable: string, levelCache: Map<number, any>, zarrVersion?: 2 | 3 | null): Promise<zarr.Array<any>>;
/**
 * Extracts no-data related metadata from a Zarr array's attributes.
 *
 * Looks for standard NetCDF attributes (`valid_min`, `valid_max`, `_FillValue`, `missing_value`).
 *
 * @param zarrArray - Zarr array to extract metadata from.
 *
 * @returns An object containing:
 *   - `metadataMin`: Valid minimum value (if any).
 *   - `metadataMax`: Valid maximum value (if any).
 *   - `fillValue`: Exact fill/missing value (if any).
 *   - `useFillValue`: Whether to apply exact masking based on fill value.
 */
declare function extractNoDataMetadata(zarrArray: zarr.Array<any>): {
    metadataMin: number | undefined;
    metadataMax: number | undefined;
    fillValue: number | undefined;
    useFillValue: boolean;
};
/**
 * Detects the coordinate reference system (CRS) of a Zarr dataset based on metadata or coordinate range.
 * Defaults to EPSG:4326 (WGS84) if uncertain.
 *
 * @param attrs - Zarr array or group attributes.
 * @param arr - Zarr array (may be null).
 * @param xyLimits - Optional geographic coordinate limits. See {@link XYLimitsProps}.
 * @returns Detected  CRS as a string (e.g., 'EPSG:4326' or 'EPSG:3857'. See {@link CRS}).
 */
declare function detectCRS(attrs: Record<string, any>, arr: zarr.Array<any> | null, xyLimits?: XYLimitsProps): Promise<CRS>;
interface BandInfo {
    band: number | string;
    index: number;
}
declare function getBandInformation(selector: Record<string, any>): Record<string, BandInfo>;
declare function getBands(variable: string, selector: Record<string, any>): string[];

export { Accent, AccentR, type BandInfo, Blues, BluesR, BrBG, BrBGR, BuGn, BuGnR, BuPu, BuPuR, CMRmap, CMRmapR, type CRS, type ColorMapInfo, type ColorMapName, type ColorScaleProps, Dark2, Dark2R, type DataSliceProps, type DimIndicesProps, type DimensionNamesProps, type FragmentShaderOptions, GnBu, GnBuR, Greens, GreensR, Greys, GreysR, type MaplibreLayerOptions, type MercatorBounds, OrRd, OrRdR, Oranges, OrangesR, PRGn, PRGnR, Paired, PairedR, Pastel1, Pastel1R, Pastel2, Pastel2R, PiYG, PiYGR, type ProjectionData, PuBu, PuBuGn, PuBuGnR, PuBuR, PuOr, PuOrR, PuRd, PuRdR, Purples, PurplesR, RdBu, RdBuR, RdGy, RdGyR, RdPu, RdPuR, RdYlBu, RdYlBuR, RdYlGn, RdYlGnR, Reds, RedsR, Set1, Set1R, Set2, Set2R, Set3, Set3R, type ShaderData, type SliceArgs, Spectral, SpectralR, type TileTuple, Wistia, WistiaR, type XYLimits, type XYLimitsProps, YlGn, YlGnBu, YlGnBuR, YlGnR, YlOrBr, YlOrBrR, YlOrRd, YlOrRdR, ZarrLayer, type ZarrLevelMetadata, type ZarrSelectorsProps, ZarrStore, afmhot, afmhotR, allColorScales, autumn, autumnR, binary, binaryR, bone, boneR, boundsToMercatorNorm, brg, brgR, bwr, bwrR, calculateNearestIndex, cividis, cividisR, colorScaleByName, colormapBuilder, cool, coolR, coolwarm, coolwarmR, copper, copperR, createColorRampTexture, createFragmentShaderSource, createProgram, createShader, createVertexShaderSource, cubehelix, cubehelixR, detectCRS, extractNoDataMetadata, flag, flagR, getBandInformation, getBands, getTilesAtZoom, getXYLimits, gistEarth, gistEarthR, gistGray, gistGrayR, gistHeat, gistHeatR, gistNcar, gistNcarR, gistRainbow, gistRainbowR, gistStern, gistSternR, gistYarg, gistYargR, gnuplot, gnuplot2, gnuplot2R, gnuplotR, gray, grayR, hot, hotR, hsv, hsvR, identifyDimensionIndices, inferno, infernoR, initZarrDataset, jet, jetR, lat2tile, latToMercatorNorm, loadDimensionValues, lon2tile, lonToMercatorNorm, magma, magmaR, maplibreFragmentShaderSource, maplibreVertexShaderSource, mustCreateBuffer, mustCreateTexture, mustGetUniformLocation, nipySpectral, nipySpectralR, ocean, oceanR, openLevelArray, pink, pinkR, plasma, plasmaR, prism, prismR, rainbow, rainbowR, renderFragmentShaderSource, renderVertexShaderSource, seismic, seismicR, spring, springR, summer, summerR, tab10, tab10R, tab20, tab20R, tab20b, tab20bR, tab20c, tab20cR, terrain, terrainR, tileToKey, tileToScale, turbo, turboR, twilight, twilightR, twilightShifted, twilightShiftedR, viridis, viridisR, winter, winterR, zoomToLevel };
