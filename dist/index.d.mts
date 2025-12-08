type ColormapArray = number[][] | string[];
type SelectorValue = number | number[] | string | string[] | [number, number];
interface ZarrSelectorsProps {
    selected: SelectorValue;
    type?: 'index' | 'value';
}
interface DimensionNamesProps {
    time?: string;
    elevation?: string;
    lat?: string;
    lon?: string;
    others?: string[];
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
    selector?: Record<string, number | number[] | string | string[] | ZarrSelectorsProps>;
    colormap: ColormapArray;
    clim: [number, number];
    opacity?: number;
    minRenderZoom?: number;
    zarrVersion?: 2 | 3;
    dimensionNames?: DimensionNamesProps;
    fillValue?: number;
    customFrag?: string;
    uniforms?: Record<string, number>;
    renderingMode?: '2d' | '3d';
    onLoadingStateChange?: LoadingStateCallback;
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
    private selectorHash;
    private tileSize;
    private isMultiscale;
    private fillValue;
    private scaleFactor;
    private offset;
    private gl;
    private map;
    private renderer;
    private dataManager;
    private tileNeedsRender;
    private projectionChangeHandler;
    private resolveGl;
    private zarrStore;
    private levelInfos;
    private dimIndices;
    private dimensionValues;
    private selectors;
    private isRemoved;
    private fragmentShaderSource;
    private customFrag;
    private customUniforms;
    private bandNames;
    private customShaderConfig;
    private onLoadingStateChange;
    private metadataLoading;
    private chunksLoading;
    private isGlobeProjection;
    constructor({ id, source, variable, selector, colormap, clim, opacity, minRenderZoom, zarrVersion, dimensionNames, fillValue, customFrag, uniforms, renderingMode, onLoadingStateChange, }: ZarrLayerOptions);
    private emitLoadingState;
    private handleChunkLoadingChange;
    setOpacity(opacity: number): void;
    setClim(clim: [number, number]): void;
    setColormap(colormap: ColormapArray): void;
    setUniforms(uniforms: Record<string, number>): void;
    setVariable(variable: string): Promise<void>;
    setSelector(selector: Record<string, number | number[] | string | string[] | ZarrSelectorsProps>): Promise<void>;
    onAdd(map: MapLike, gl: WebGL2RenderingContext | null): Promise<void>;
    private computeSelectorHash;
    private initializeManager;
    private initialize;
    private loadInitialDimensionValues;
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
}

export { type ColormapArray, type DimensionNamesProps, type LoadingState, type LoadingStateCallback, ZarrLayer, type ZarrLayerOptions };
