type ColormapArray = number[][] | string[];
type SelectorValue = number | number[] | string | string[];
interface SelectorSpec {
    selected: SelectorValue;
    type?: 'index' | 'value';
}
type Selector = Record<string, SelectorValue | SelectorSpec>;
/**
 * Override the names used to identify spatial dimensions (lat/lon).
 * Only needed if your dataset uses non-standard names that aren't auto-detected.
 * Standard names (lat, latitude, y, lon, longitude, x) are detected automatically.
 */
interface SpatialDimensions {
    lat?: string;
    lon?: string;
}
interface LoadingState {
    loading: boolean;
    metadata: boolean;
    chunks: boolean;
}
type LoadingStateCallback = (state: LoadingState) => void;
interface ZarrLayerOptions {
    id: string;
    source: string;
    variable: string;
    selector?: Selector;
    colormap: ColormapArray;
    clim: [number, number];
    opacity?: number;
    minzoom?: number;
    maxzoom?: number;
    zarrVersion?: 2 | 3;
    spatialDimensions?: SpatialDimensions;
    /**
     * Explicit spatial bounds [west, south, east, north] in degrees.
     * Used when coordinate arrays aren't available in the zarr store.
     * If not provided, bounds are read from coordinate arrays or default to global.
     */
    bounds?: [number, number, number, number];
    latIsAscending?: boolean | null;
    fillValue?: number;
    customFrag?: string;
    uniforms?: Record<string, number>;
    renderingMode?: '2d' | '3d';
    onLoadingStateChange?: LoadingStateCallback;
    /**
     * Throttle interval in milliseconds for data fetching during rapid selector changes.
     * Higher values reduce network requests when scrubbing through e.g. time sliders.
     * Set to 0 to disable throttling. Default: 100ms.
     */
    throttleMs?: number;
}
interface BoundsLike {
    getWest(): number;
    getEast(): number;
    toArray(): [[number, number], [number, number]];
}
interface MapLike {
    getProjection?(): {
        type?: string;
        name?: string;
    } | null;
    getRenderWorldCopies?(): boolean;
    on?(event: string, handler: (...args: unknown[]) => void): void;
    off?(event: string, handler: (...args: unknown[]) => void): void;
    triggerRepaint?(): void;
    getBounds?(): BoundsLike | null;
    getZoom?(): number;
    painter?: {
        context?: {
            gl?: unknown;
        };
    };
    renderer?: {
        getContext?: () => unknown;
    };
}

/**
 * Nested values structure for multi-dimensional data queries.
 */
interface NestedValues {
    [key: string]: number[] | NestedValues;
    [key: number]: number[] | NestedValues;
}
/**
 * Values from a data query. Can be flat array or nested when selector has array values.
 *
 * Flat: `number[]` when selector = `{ month: 1 }`
 * Nested: `{ 1: number[], 2: number[] }` when selector = `{ month: [1, 2] }`
 */
type QueryDataValues = number[] | NestedValues;
/**
 * Result from a query (point or region).
 * Matches carbonplan/maps structure: { [variable]: values, dimensions, coordinates }
 */
interface QueryResult {
    /** Variable name mapped to its values (flat array or nested based on selector) */
    [variable: string]: QueryDataValues | string[] | {
        [key: string]: (number | string)[];
    };
    /** Dimension names in order (e.g., ['month', 'lat', 'lon']) */
    dimensions: string[];
    /** Coordinate arrays for each dimension */
    coordinates: {
        lat: number[];
        lon: number[];
        [key: string]: (number | string)[];
    };
}
/**
 * GeoJSON Point geometry.
 */
interface GeoJSONPoint {
    type: 'Point';
    coordinates: [number, number];
}
/**
 * GeoJSON Polygon geometry.
 */
interface GeoJSONPolygon {
    type: 'Polygon';
    coordinates: number[][][];
}
/**
 * GeoJSON MultiPolygon geometry.
 */
interface GeoJSONMultiPolygon {
    type: 'MultiPolygon';
    coordinates: number[][][][];
}
/**
 * Supported GeoJSON geometry types for queries.
 */
type QueryGeometry = GeoJSONPoint | GeoJSONPolygon | GeoJSONMultiPolygon;

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
    private spatialDimensions;
    private bounds;
    private latIsAscending;
    private selector;
    private invalidate;
    private colormap;
    private clim;
    private opacity;
    private minzoom;
    private maxzoom;
    private selectorHash;
    private _fillValue;
    private scaleFactor;
    private offset;
    private gl;
    private map;
    private renderer;
    private mode;
    private tileNeedsRender;
    private projectionChangeHandler;
    private resolveGl;
    private zarrStore;
    private levelInfos;
    private dimIndices;
    private dimensionValues;
    private normalizedSelector;
    private isRemoved;
    private fragmentShaderSource;
    private customFrag;
    private customUniforms;
    private bandNames;
    private customShaderConfig;
    private onLoadingStateChange;
    private metadataLoading;
    private chunksLoading;
    private throttleMs;
    get fillValue(): number | null;
    private isGlobeProjection;
    constructor({ id, source, variable, selector, colormap, clim, opacity, minzoom, maxzoom, zarrVersion, spatialDimensions, bounds, latIsAscending, fillValue, customFrag, uniforms, renderingMode, onLoadingStateChange, throttleMs, }: ZarrLayerOptions);
    private emitLoadingState;
    private handleChunkLoadingChange;
    setOpacity(opacity: number): void;
    setClim(clim: [number, number]): void;
    setColormap(colormap: ColormapArray): void;
    setUniforms(uniforms: Record<string, number>): void;
    setVariable(variable: string): Promise<void>;
    setSelector(selector: Selector): Promise<void>;
    onAdd(map: MapLike, gl: WebGL2RenderingContext | null): Promise<void>;
    private computeSelectorHash;
    private initializeMode;
    private initialize;
    private loadInitialDimensionValues;
    private isZoomInRange;
    prerender(_gl: WebGL2RenderingContext, _params: unknown): void;
    render(_gl: WebGL2RenderingContext, params: unknown, projection?: {
        name: string;
    }, projectionToMercatorMatrix?: number[] | Float32Array | Float64Array, projectionToMercatorTransition?: number, _centerInMercator?: number[], _pixelsPerMeterRatio?: number): void;
    renderToTile(_gl: WebGL2RenderingContext, tileId: {
        z: number;
        x: number;
        y: number;
    }): void;
    shouldRerenderTiles(): boolean;
    onRemove(_map: MapLike, gl: WebGL2RenderingContext): void;
    /**
     * Query all data values within a geographic region.
     * @param geometry - GeoJSON Point, Polygon or MultiPolygon geometry.
     * @param selector - Optional selector to override the layer's selector.
     * @returns Promise resolving to the query result matching carbonplan/maps structure.
     */
    queryData(geometry: QueryGeometry, selector?: Selector): Promise<QueryResult>;
}

export { type ColormapArray, type LoadingState, type LoadingStateCallback, type QueryDataValues, type QueryGeometry, type QueryResult, type Selector, type SpatialDimensions, ZarrLayer, type ZarrLayerOptions };
