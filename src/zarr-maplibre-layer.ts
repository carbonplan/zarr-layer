/**
 * @module zarr-maplibre-layer
 *
 * MapLibre/MapBox custom layer implementation for rendering Zarr datasets.
 * Implements CustomLayerInterface for direct WebGL rendering.
 */

import * as zarr from "zarrita";
import { colormapBuilder } from "./jsColormaps";
import { calculateNearestIndex, loadDimensionValues } from "./zarr-utils";
import { ZarrStore } from "./zarr-store";
import {
  createColorRampTexture,
  createProgram,
  createShader,
  mustCreateBuffer,
  mustCreateFramebuffer,
  mustCreateTexture,
  mustGetUniformLocation,
} from "./webgl-utils";
import {
  maplibreFragmentShaderSource,
  maplibreVertexShaderSource,
  renderFragmentShaderSource,
  renderVertexShaderSource,
} from "./maplibre-shaders";
import {
  boundsToMercatorNorm,
  getTilesAtZoom,
  type MercatorBounds,
  tileToKey,
  tileToScale,
  type TileTuple,
  zoomToLevel,
} from "./maplibre-utils";
import type {
  ColorMapName,
  CRS,
  DimensionNamesProps,
  DimIndicesProps,
  MaplibreLayerOptions,
  XYLimits,
  ZarrLevelMetadata,
  ZarrSelectorsProps,
} from "./types";

const DEFAULT_TILE_SIZE = 128;
const MAX_CACHED_TILES = 64;

interface TileData {
  chunkData: Float32Array | null;
  chunkShape: number[] | null;
  data: Float32Array | null;
  selectorHash: string | null;
  tileTexture: WebGLTexture;
  vertexBuffer: WebGLBuffer;
  pixCoordBuffer: WebGLBuffer;
  loading: boolean;
  lastUsed: number;
}

/**
 * MapLibre/MapBox custom layer for rendering Zarr datasets.
 * Implements the CustomLayerInterface specification.
 *
 * @example
 * ```ts
 * const layer = new ZarrMaplibreLayer({
 *   id: 'my-zarr-layer',
 *   source: 'https://example.com/data.zarr',
 *   variable: 'temperature',
 *   vmin: 0,
 *   vmax: 40,
 *   colormap: 'viridis'
 * });
 * map.addLayer(layer);
 * ```
 */
export class ZarrMaplibreLayer {
  type: "custom" = "custom";
  renderingMode: "2d" = "2d";

  id: string;
  private url: string;
  private variable: string;
  private zarrVersion: 2 | 3 | null = null;
  private dimensionNames: DimensionNamesProps;
  private selector: Record<string, number>;
  private invalidate: () => void;

  private cmap: Float32Array;
  private cmapLength: number;
  private cmapColors: number[][];
  private vmin: number;
  private vmax: number;
  private opacity: number;
  private minRenderZoom: number;

  private tiles: Map<string, TileData> = new Map();
  private tileAccessOrder: string[] = [];
  private maxZoom: number = 4;
  private tileSize: number = DEFAULT_TILE_SIZE;
  private isMultiscale: boolean = true;
  private singleImageData: Float32Array | null = null;
  private singleImageTexture: WebGLTexture | null = null;
  private singleImageVertexBuffer: WebGLBuffer | null = null;
  private singleImagePixCoordBuffer: WebGLBuffer | null = null;
  private singleImageWidth: number = 0;
  private singleImageHeight: number = 0;
  private mercatorBounds: MercatorBounds | null = null;
  private fillValue: number = 0;
  private useFillValue: boolean = false;
  private noDataMin: number = -9999;
  private noDataMax: number = 9999;
  private scaleFactor: number = 1;
  private offset: number = 0;

  private gl: WebGL2RenderingContext | undefined;
  private program: WebGLProgram | null = null;
  private map: any;

  private scaleLoc: WebGLUniformLocation | undefined;
  private scaleXLoc: WebGLUniformLocation | undefined;
  private scaleYLoc: WebGLUniformLocation | undefined;
  private shiftXLoc: WebGLUniformLocation | undefined;
  private shiftYLoc: WebGLUniformLocation | undefined;
  private worldXOffsetLoc: WebGLUniformLocation | undefined;
  private matrixLoc: WebGLUniformLocation | undefined;
  private vminLoc: WebGLUniformLocation | undefined;
  private vmaxLoc: WebGLUniformLocation | undefined;
  private opacityLoc: WebGLUniformLocation | undefined;
  private noDataLoc: WebGLUniformLocation | undefined;
  private noDataMinLoc: WebGLUniformLocation | undefined;
  private noDataMaxLoc: WebGLUniformLocation | undefined;
  private useFillValueLoc: WebGLUniformLocation | undefined;
  private fillValueLoc: WebGLUniformLocation | undefined;
  private scaleFactorLoc: WebGLUniformLocation | undefined;
  private addOffsetLoc: WebGLUniformLocation | undefined;

  private vertexLoc: number = 0;
  private cmapTex: WebGLTexture | null = null;
  private cmapLoc: WebGLUniformLocation | undefined;

  private vertexArr: Float32Array;
  private pixCoordArr: Float32Array;
  private singleImagePixCoordArr: Float32Array = new Float32Array();

  private texLoc: WebGLUniformLocation | undefined;
  private pixCoordLoc: number = 0;

  private frameBuffers: {
    current: {
      framebuffer: WebGLFramebuffer;
      texture: WebGLTexture;
    } | null;
    next: {
      framebuffer: WebGLFramebuffer;
      texture: WebGLTexture;
    } | null;
  };
  private isUpdating: boolean = false;
  private canvasWidth: number = 512;
  private canvasHeight: number = 512;

  private renderProgram: WebGLProgram | null = null;
  private renderVertexLoc: number = 0;
  private renderTexLoc: WebGLUniformLocation | undefined;
  private vertexBuffer: WebGLBuffer | null = null;

  private zarrArray: zarr.Array<any> | null = null;
  private zarrStore: ZarrStore | null = null;
  private levelInfos: string[] = [];
  private levelMetadata: Map<number, ZarrLevelMetadata> = new Map();
  private dimIndices: DimIndicesProps = {};
  private xyLimits: XYLimits | null = null;
  private crs: CRS | null = null;
  private dimensionValues: { [key: string]: Float64Array | number[] } = {};
  private selectors: { [key: string]: ZarrSelectorsProps } = {};
  private isRemoved: boolean = false;

  constructor({
    id,
    source,
    variable,
    selector = {},
    colormap = "viridis",
    vmin,
    vmax,
    opacity = 1,
    minRenderZoom = 3,
    zarrVersion,
    dimensionNames = {},
    noDataMin,
    noDataMax,
  }: MaplibreLayerOptions) {
    this.id = id;
    this.url = source;
    this.variable = variable;
    this.zarrVersion = zarrVersion ?? null;
    this.dimensionNames = dimensionNames;
    this.selector = selector;
    for (const [dimName, value] of Object.entries(selector)) {
      this.selectors[dimName] = { selected: value, type: "index" };
    }
    this.invalidate = () => {};

    const colors = colormapBuilder(colormap);
    this.cmapColors = colors as number[][];
    this.cmap = new Float32Array(this.cmapColors.flat().map((v) => v / 255.0));
    this.cmapLength = this.cmapColors.length;
    this.vmin = vmin;
    this.vmax = vmax;
    this.opacity = opacity;
    this.minRenderZoom = minRenderZoom;

    if (noDataMin !== undefined) this.noDataMin = noDataMin;
    if (noDataMax !== undefined) this.noDataMax = noDataMax;

    this.frameBuffers = { current: null, next: null };

    // Vertices in clip space [-1, 1] representing a tile quad
    // Order: top-left, bottom-left, top-right, bottom-right (triangle strip)
    this.vertexArr = new Float32Array([
      -1.0,
      1.0, // top-left
      -1.0,
      -1.0, // bottom-left
      1.0,
      1.0, // top-right
      1.0,
      -1.0, // bottom-right
    ]);

    // Texture coordinates for sampling the tile texture
    // For multiscale tiles, Y increases downward (north to south)
    this.pixCoordArr = new Float32Array([
      0.0,
      0.0, // top-left
      0.0,
      1.0, // bottom-left
      1.0,
      0.0, // top-right
      1.0,
      1.0, // bottom-right
    ]);

    // Texture coordinates for single image (EPSG:4326 data)
    // Latitude often increases upward in data, so Y is flipped
    this.singleImagePixCoordArr = new Float32Array([
      0.0,
      1.0, // top-left (sample from bottom of texture)
      0.0,
      0.0, // bottom-left (sample from top of texture)
      1.0,
      1.0, // top-right
      1.0,
      0.0, // bottom-right
    ]);
  }

  setOpacity(opacity: number) {
    this.opacity = opacity;
    this.invalidate();
  }

  setVminVmax(vmin: number, vmax: number) {
    this.vmin = vmin;
    this.vmax = vmax;
    this.invalidate();
  }

  setColormap(colormap: ColorMapName) {
    const colors = colormapBuilder(colormap) as number[][];
    this.cmapColors = colors;
    this.cmap = new Float32Array(colors.flat().map((v) => v / 255.0));
    this.cmapLength = colors.length;
    this.updateColormapTexture();
    this.invalidate();
  }

  async setVariable(variable: string) {
    this.variable = variable;
    this.clearAllTiles();
    await this.prepareTiles();
    this.getVisibleTiles();
    await this.prefetchTileData();
    this.invalidate();
  }

  private clearAllTiles() {
    const gl = this.gl;
    if (gl) {
      for (const tile of this.tiles.values()) {
        if (tile.tileTexture) gl.deleteTexture(tile.tileTexture);
        if (tile.vertexBuffer) gl.deleteBuffer(tile.vertexBuffer);
        if (tile.pixCoordBuffer) gl.deleteBuffer(tile.pixCoordBuffer);
      }
    }
    this.tiles.clear();
    this.tileAccessOrder = [];
  }

  async setSelector(selector: Record<string, number>) {
    this.selector = selector;
    for (const [dimName, value] of Object.entries(selector)) {
      this.selectors[dimName] = { selected: value, type: "index" };
    }
    if (!this.isMultiscale) {
      this.singleImageData = null;
      await this.prefetchTileData();
    } else {
      this.reextractTileSlices();
    }
    this.invalidate();
  }

  private async reextractTileSlices() {
    if (!this.zarrStore) return;

    const currentHash = this.getSelectorHash();
    const visibleTiles = this.getVisibleTiles();

    const levelPath = this.levelInfos[0];
    if (!levelPath) return;

    const levelArray = await this.zarrStore.getLevelArray(levelPath);
    const useChunkCaching = this.shouldUseChunkCaching(levelArray);

    if (!useChunkCaching) {
      for (const tileTuple of visibleTiles) {
        const tileKey = tileToKey(tileTuple);
        const tile = this.tiles.get(tileKey);
        if (tile && tile.selectorHash !== currentHash) {
          tile.data = null;
          tile.selectorHash = null;
        }
      }
      this.prefetchTileData();
      return;
    }

    for (const tileTuple of visibleTiles) {
      const tileKey = tileToKey(tileTuple);
      const tile = this.tiles.get(tileKey);

      if (
        tile &&
        tile.chunkData &&
        tile.chunkShape &&
        tile.selectorHash !== currentHash
      ) {
        const tileLevelPath = this.levelInfos[tileTuple[0]];
        if (tileLevelPath) {
          try {
            const tileLevelArray = await this.zarrStore.getLevelArray(
              tileLevelPath
            );
            tile.data = this.extractSliceFromChunk(
              tile.chunkData,
              tile.chunkShape,
              tileLevelArray
            );
            tile.selectorHash = currentHash;
          } catch (err) {
            console.warn("Error re-extracting tile slice:", err);
          }
        }
      }
    }
  }

  private updateColormapTexture() {
    if (!this.gl) return;
    const gl = this.gl;
    if (!this.cmapTex) {
      this.cmapTex = mustCreateTexture(gl);
    }
    gl.bindTexture(gl.TEXTURE_2D, this.cmapTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGB16F,
      this.cmapLength,
      1,
      0,
      gl.RGB,
      gl.FLOAT,
      this.cmap
    );
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  async onAdd(map: any, gl: WebGL2RenderingContext) {
    this.map = map;
    this.gl = gl;
    this.invalidate = () => map.triggerRepaint();

    const vertexShader = createShader(
      gl,
      gl.VERTEX_SHADER,
      maplibreVertexShaderSource
    );
    const fragmentShader = createShader(
      gl,
      gl.FRAGMENT_SHADER,
      maplibreFragmentShaderSource
    );
    if (!vertexShader || !fragmentShader) {
      throw new Error("Failed to create shaders");
    }
    this.program = createProgram(gl, vertexShader, fragmentShader);
    if (!this.program) {
      throw new Error("Failed to create program");
    }

    this.scaleLoc = mustGetUniformLocation(gl, this.program, "scale");
    this.scaleXLoc = mustGetUniformLocation(gl, this.program, "scale_x");
    this.scaleYLoc = mustGetUniformLocation(gl, this.program, "scale_y");
    this.shiftXLoc = mustGetUniformLocation(gl, this.program, "shift_x");
    this.shiftYLoc = mustGetUniformLocation(gl, this.program, "shift_y");
    this.worldXOffsetLoc = mustGetUniformLocation(
      gl,
      this.program,
      "u_worldXOffset"
    );
    this.matrixLoc = mustGetUniformLocation(gl, this.program, "matrix");

    this.vminLoc = mustGetUniformLocation(gl, this.program, "vmin");
    this.vmaxLoc = mustGetUniformLocation(gl, this.program, "vmax");
    this.opacityLoc = mustGetUniformLocation(gl, this.program, "opacity");
    this.noDataLoc = mustGetUniformLocation(gl, this.program, "nodata");

    this.noDataMinLoc = mustGetUniformLocation(gl, this.program, "u_noDataMin");
    this.noDataMaxLoc = mustGetUniformLocation(gl, this.program, "u_noDataMax");
    this.useFillValueLoc = mustGetUniformLocation(
      gl,
      this.program,
      "u_useFillValue"
    );
    this.fillValueLoc = mustGetUniformLocation(gl, this.program, "u_fillValue");
    this.scaleFactorLoc = mustGetUniformLocation(
      gl,
      this.program,
      "u_scaleFactor"
    );
    this.addOffsetLoc = mustGetUniformLocation(gl, this.program, "u_addOffset");

    this.updateColormapTexture();
    this.cmapLoc = mustGetUniformLocation(gl, this.program, "cmap");

    this.texLoc = mustGetUniformLocation(gl, this.program, "tex");

    this.vertexLoc = gl.getAttribLocation(this.program, "vertex");
    this.pixCoordLoc = gl.getAttribLocation(this.program, "pix_coord_in");

    this.canvasWidth = gl.canvas.width;
    this.canvasHeight = gl.canvas.height;
    this.frameBuffers.current = mustCreateFramebuffer(
      gl,
      this.canvasWidth,
      this.canvasHeight
    );
    this.frameBuffers.next = mustCreateFramebuffer(
      gl,
      this.canvasWidth,
      this.canvasHeight
    );

    await this.initialize();
    await this.prepareTiles();

    this.vertexBuffer = mustCreateBuffer(gl);

    this.prefetchTileData().then(() => {
      this.invalidate();
    });

    const renderVertShader = createShader(
      gl,
      gl.VERTEX_SHADER,
      renderVertexShaderSource
    );
    const renderFragShader = createShader(
      gl,
      gl.FRAGMENT_SHADER,
      renderFragmentShaderSource
    );
    if (!renderVertShader || !renderFragShader) {
      throw new Error("Failed to create render shaders");
    }
    this.renderProgram = createProgram(gl, renderVertShader, renderFragShader);
    if (!this.renderProgram) {
      throw new Error("Failed to create render program");
    }
    this.renderVertexLoc = gl.getAttribLocation(this.renderProgram, "vertex");
    this.renderTexLoc = mustGetUniformLocation(gl, this.renderProgram, "tex");

    gl.deleteShader(renderVertShader);
    gl.deleteShader(renderFragShader);
  }

  private async initialize(): Promise<void> {
    try {
      this.zarrStore = new ZarrStore({
        source: this.url,
        version: this.zarrVersion,
        variable: this.variable,
        dimensionNames: this.dimensionNames,
      });

      await this.zarrStore.initialized;

      const desc = this.zarrStore.describe();

      this.levelInfos = desc.levels;
      this.dimIndices = desc.dimIndices;
      this.xyLimits = desc.xyLimits;
      this.crs = desc.crs;
      this.scaleFactor = desc.scaleFactor;
      this.offset = desc.addOffset;
      this.tileSize = desc.tileSize || DEFAULT_TILE_SIZE;

      if (desc.fill_value !== null && desc.fill_value !== undefined) {
        this.fillValue = desc.fill_value;
        this.useFillValue = true;
      }

      if (this.levelInfos.length > 0) {
        this.zarrArray = await this.zarrStore.getLevelArray(this.levelInfos[0]);
      } else {
        this.zarrArray = await this.zarrStore.getArray();
      }

      for (let i = 0; i < this.levelInfos.length; i++) {
        const levelArr = await this.zarrStore.getLevelArray(this.levelInfos[i]);
        const width = levelArr.shape[this.dimIndices.lon?.index ?? 1];
        const height = levelArr.shape[this.dimIndices.lat?.index ?? 0];
        this.levelMetadata.set(i, { width, height });
      }

      await this.loadInitialDimensionValues();
    } catch (err) {
      console.error("Failed to initialize Zarr layer:", err);
      throw err;
    }
  }

  private async loadInitialDimensionValues(): Promise<void> {
    if (!this.zarrStore?.root) return;

    const multiscaleLevel =
      this.levelInfos.length > 0 ? this.levelInfos[0] : null;

    for (const [dimName, value] of Object.entries(this.selector)) {
      this.selectors[dimName] = { selected: value, type: "index" };
    }

    for (const dimName of Object.keys(this.dimIndices)) {
      if (dimName !== "lon" && dimName !== "lat") {
        try {
          this.dimensionValues[dimName] = await loadDimensionValues(
            this.dimensionValues,
            multiscaleLevel,
            this.dimIndices[dimName],
            this.zarrStore.root,
            this.zarrStore.version
          );

          if (!this.selectors[dimName]) {
            this.selectors[dimName] = { selected: 0, type: "index" };
          } else if (this.selectors[dimName].type === "value") {
            this.selectors[dimName].selected = calculateNearestIndex(
              this.dimensionValues[dimName],
              this.selectors[dimName].selected as number
            );
          }
        } catch (err) {
          console.warn(`Failed to load dimension values for ${dimName}:`, err);
        }
      }
    }
  }

  async prefetchTileData() {
    if (!this.isMultiscale) {
      await this.fetchSingleImageData();
      return;
    }

    const tiles = this.getVisibleTiles();
    const fetchPromises = tiles.map((tiletuple) =>
      this.fetchTileData(tiletuple)
    );
    await Promise.all(fetchPromises);
  }

  getVisibleTiles(): TileTuple[] {
    const mapZoom = this.map.getZoom();
    if (mapZoom < this.minRenderZoom) {
      return [];
    }
    const pyramidLevel = zoomToLevel(mapZoom, this.maxZoom);

    const bounds = this.map.getBounds()?.toArray();
    if (!bounds) {
      return [];
    }
    const tiles = getTilesAtZoom(pyramidLevel, bounds);
    return tiles;
  }

  private getWorldOffsets(): number[] {
    const bounds = this.map.getBounds();
    if (!bounds) return [0];

    const west = bounds.getWest();
    const east = bounds.getEast();

    const minWorld = Math.floor((west + 180) / 360);
    const maxWorld = Math.floor((east + 180) / 360);

    const worldOffsets: number[] = [];
    for (let i = minWorld; i <= maxWorld; i++) {
      worldOffsets.push(i);
    }
    return worldOffsets.length > 0 ? worldOffsets : [0];
  }

  async prepareTiles() {
    if (typeof this.gl === "undefined") {
      throw new Error("Cannot prepareTiles with no GL context set");
    }

    if (this.levelInfos.length === 0) {
      this.isMultiscale = false;
      await this.prepareSingleImage();
      return;
    }

    this.isMultiscale = true;
    this.maxZoom = this.levelInfos.length - 1;
  }

  private getOrCreateTile(tileKey: string): TileData {
    const gl = this.gl!;
    let tile = this.tiles.get(tileKey);

    if (!tile) {
      tile = {
        chunkData: null,
        chunkShape: null,
        data: null,
        selectorHash: null,
        tileTexture: mustCreateTexture(gl),
        vertexBuffer: mustCreateBuffer(gl),
        pixCoordBuffer: mustCreateBuffer(gl),
        loading: false,
        lastUsed: Date.now(),
      };
      this.tiles.set(tileKey, tile);
      this.tileAccessOrder.push(tileKey);
      this.evictOldTiles();
    } else {
      tile.lastUsed = Date.now();
      const idx = this.tileAccessOrder.indexOf(tileKey);
      if (idx > -1) {
        this.tileAccessOrder.splice(idx, 1);
        this.tileAccessOrder.push(tileKey);
      }
    }

    return tile;
  }

  private getSelectorHash(): string {
    return JSON.stringify(this.selector);
  }

  private extractSliceFromChunk(
    chunkData: Float32Array,
    chunkShape: number[],
    levelArray: zarr.Array<any>
  ): Float32Array {
    const tileWidth = this.tileSize;
    const tileHeight = this.tileSize;
    const paddedData = new Float32Array(tileWidth * tileHeight);
    paddedData.fill(this.fillValue);

    const dimensions = this.zarrStore?.dimensions || [];

    const selectorIndices: number[] = [];
    let latDimIdx = -1;
    let lonDimIdx = -1;
    let latSize = tileHeight;
    let lonSize = tileWidth;

    for (let i = 0; i < dimensions.length; i++) {
      const dimName = dimensions[i];
      const dimKey = this.getDimKeyForName(dimName);

      if (dimKey === "lat") {
        latDimIdx = i;
        latSize = Math.min(chunkShape[i], tileHeight);
        selectorIndices.push(-1);
      } else if (dimKey === "lon") {
        lonDimIdx = i;
        lonSize = Math.min(chunkShape[i], tileWidth);
        selectorIndices.push(-1);
      } else {
        let idx = 0;
        const dimSelection =
          this.selectors[dimKey] ??
          this.selectors[dimName] ??
          this.selector[dimKey] ??
          this.selector[dimName];
        if (dimSelection !== undefined) {
          idx =
            typeof dimSelection === "object"
              ? (dimSelection.selected as number)
              : dimSelection;
        }
        idx = Math.max(0, Math.min(idx, chunkShape[i] - 1));
        selectorIndices.push(idx);
      }
    }

    const getChunkIndex = (indices: number[]): number => {
      let idx = 0;
      let stride = 1;
      for (let i = indices.length - 1; i >= 0; i--) {
        idx += indices[i] * stride;
        stride *= chunkShape[i];
      }
      return idx;
    };

    for (let latIdx = 0; latIdx < latSize; latIdx++) {
      for (let lonIdx = 0; lonIdx < lonSize; lonIdx++) {
        const indices = [...selectorIndices];
        if (latDimIdx >= 0) indices[latDimIdx] = latIdx;
        if (lonDimIdx >= 0) indices[lonDimIdx] = lonIdx;

        const srcIdx = getChunkIndex(indices);
        const dstIdx = latIdx * tileWidth + lonIdx;

        if (srcIdx < chunkData.length) {
          paddedData[dstIdx] = chunkData[srcIdx];
        }
      }
    }

    return paddedData;
  }

  private getDimKeyForName(dimName: string): string {
    const lower = dimName.toLowerCase();
    if (["lat", "latitude", "y"].includes(lower)) return "lat";
    if (["lon", "longitude", "x", "lng"].includes(lower)) return "lon";
    if (["time", "t", "time_counter"].includes(lower)) return "time";
    if (["depth", "z", "level", "lev", "elevation"].includes(lower))
      return "elevation";
    return dimName;
  }

  private evictOldTiles() {
    const gl = this.gl;
    if (!gl) return;

    while (this.tiles.size > MAX_CACHED_TILES) {
      const oldestKey = this.tileAccessOrder.shift();
      if (!oldestKey) break;

      const tile = this.tiles.get(oldestKey);
      if (tile) {
        if (tile.tileTexture) gl.deleteTexture(tile.tileTexture);
        if (tile.vertexBuffer) gl.deleteBuffer(tile.vertexBuffer);
        if (tile.pixCoordBuffer) gl.deleteBuffer(tile.pixCoordBuffer);
        tile.data = null;
        this.tiles.delete(oldestKey);
      }
    }
  }

  private async prepareSingleImage(): Promise<void> {
    if (!this.gl || !this.zarrArray || !this.xyLimits) {
      console.warn(
        "Cannot prepare single image: missing GL context, zarrArray, or xyLimits"
      );
      return;
    }

    const gl = this.gl;

    this.mercatorBounds = boundsToMercatorNorm(this.xyLimits, this.crs);

    this.singleImageTexture = mustCreateTexture(gl);
    this.singleImageVertexBuffer = mustCreateBuffer(gl);
    this.singleImagePixCoordBuffer = mustCreateBuffer(gl);

    this.singleImageWidth = this.zarrArray.shape[this.dimIndices.lon.index];
    this.singleImageHeight = this.zarrArray.shape[this.dimIndices.lat.index];
  }

  private async fetchSingleImageData(): Promise<Float32Array | null> {
    if (!this.zarrArray || this.singleImageData || this.isRemoved) {
      return this.singleImageData;
    }

    try {
      const sliceArgs: any[] = new Array(this.zarrArray.shape.length).fill(0);

      for (const dimName of Object.keys(this.dimIndices)) {
        const dimInfo = this.dimIndices[dimName];
        if (dimName === "lon") {
          sliceArgs[dimInfo.index] = zarr.slice(0, this.singleImageWidth);
        } else if (dimName === "lat") {
          sliceArgs[dimInfo.index] = zarr.slice(0, this.singleImageHeight);
        } else {
          const dimSelection =
            this.selectors[dimName] || this.selector[dimName];
          if (dimSelection !== undefined) {
            sliceArgs[dimInfo.index] =
              typeof dimSelection === "object"
                ? (dimSelection.selected as number)
                : dimSelection;
          } else {
            sliceArgs[dimInfo.index] = 0;
          }
        }
      }

      const data = await zarr.get(this.zarrArray, sliceArgs);
      if (this.isRemoved) return null;
      this.singleImageData = new Float32Array(
        (data.data as Float32Array).buffer
      );
      this.invalidate();
      return this.singleImageData;
    } catch (err) {
      console.error("Error fetching single image data:", err);
      return null;
    }
  }

  private shouldUseChunkCaching(levelArray: zarr.Array<any>): boolean {
    const dimensions = this.zarrStore?.dimensions || [];
    const MAX_CACHEABLE_DIM_SIZE = 16;

    for (let i = 0; i < dimensions.length; i++) {
      const dimName = dimensions[i];
      const dimKey = this.getDimKeyForName(dimName);
      if (dimKey !== "lon" && dimKey !== "lat") {
        if (levelArray.shape[i] > MAX_CACHEABLE_DIM_SIZE) {
          return false;
        }
      }
    }
    return true;
  }

  private async fetchTileData(
    tileTuple: TileTuple
  ): Promise<Float32Array | null> {
    if (this.isRemoved || !this.zarrStore || !this.gl) return null;

    const tileKey = tileToKey(tileTuple);
    const tile = this.getOrCreateTile(tileKey);
    const currentHash = this.getSelectorHash();

    if (tile.data && tile.selectorHash === currentHash) {
      return tile.data;
    }

    const [z, x, y] = tileTuple;
    const levelPath = this.levelInfos[z];
    if (!levelPath) return null;

    const levelArray = await this.zarrStore.getLevelArray(levelPath);
    const useChunkCaching = this.shouldUseChunkCaching(levelArray);

    if (useChunkCaching && tile.chunkData && tile.chunkShape) {
      tile.data = this.extractSliceFromChunk(
        tile.chunkData,
        tile.chunkShape,
        levelArray
      );
      tile.selectorHash = currentHash;
      return tile.data;
    }

    if (tile.loading) return null;

    tile.loading = true;

    try {
      const dimensions = this.zarrStore.dimensions || [];
      const sliceArgs: any[] = new Array(levelArray.shape.length).fill(0);
      const chunks = levelArray.chunks;
      const chunkShape: number[] = [];

      for (let i = 0; i < dimensions.length; i++) {
        const dimName = dimensions[i];
        const dimKey = this.getDimKeyForName(dimName);

        if (dimKey === "lon") {
          const chunkSize = chunks[i];
          const arrayWidth = levelArray.shape[i];
          const startX = x * chunkSize;
          const endX = Math.min((x + 1) * chunkSize, arrayWidth);
          if (startX >= arrayWidth) {
            tile.loading = false;
            return null;
          }
          sliceArgs[i] = zarr.slice(startX, endX);
          chunkShape.push(endX - startX);
        } else if (dimKey === "lat") {
          const chunkSize = chunks[i];
          const arrayHeight = levelArray.shape[i];
          const startY = y * chunkSize;
          const endY = Math.min((y + 1) * chunkSize, arrayHeight);
          if (startY >= arrayHeight) {
            tile.loading = false;
            return null;
          }
          sliceArgs[i] = zarr.slice(startY, endY);
          chunkShape.push(endY - startY);
        } else {
          if (useChunkCaching) {
            sliceArgs[i] = zarr.slice(0, levelArray.shape[i]);
            chunkShape.push(levelArray.shape[i]);
          } else {
            const dimSelection =
              this.selectors[dimKey] ??
              this.selectors[dimName] ??
              this.selector[dimKey] ??
              this.selector[dimName];
            let idx = 0;
            if (dimSelection !== undefined) {
              idx =
                typeof dimSelection === "object"
                  ? (dimSelection.selected as number)
                  : dimSelection;
            }
            idx = Math.max(0, Math.min(idx, levelArray.shape[i] - 1));
            sliceArgs[i] = idx;
            chunkShape.push(1);
          }
        }
      }

      const data = await zarr.get(levelArray, sliceArgs);
      if (this.isRemoved) {
        tile.loading = false;
        return null;
      }

      const flatData = new Float32Array((data.data as Float32Array).buffer);

      if (useChunkCaching) {
        tile.chunkData = flatData;
        tile.chunkShape = chunkShape;
        tile.data = this.extractSliceFromChunk(
          flatData,
          chunkShape,
          levelArray
        );
      } else {
        tile.chunkData = null;
        tile.chunkShape = null;
        tile.data = this.padTileData(flatData, chunkShape);
      }
      tile.selectorHash = currentHash;
      tile.loading = false;
      this.invalidate();

      return tile.data;
    } catch (err) {
      console.error("Error fetching tile data:", err);
      tile.loading = false;
      return null;
    }
  }

  private padTileData(
    flatData: Float32Array,
    chunkShape: number[]
  ): Float32Array {
    const tileWidth = this.tileSize;
    const tileHeight = this.tileSize;
    const paddedData = new Float32Array(tileWidth * tileHeight);
    paddedData.fill(this.fillValue);

    const dimensions = this.zarrStore?.dimensions || [];
    let latDimIdx = -1;
    let lonDimIdx = -1;

    for (let i = 0; i < dimensions.length; i++) {
      const dimKey = this.getDimKeyForName(dimensions[i]);
      if (dimKey === "lat") latDimIdx = i;
      else if (dimKey === "lon") lonDimIdx = i;
    }

    const latSize =
      latDimIdx >= 0 ? Math.min(chunkShape[latDimIdx], tileHeight) : tileHeight;
    const lonSize =
      lonDimIdx >= 0 ? Math.min(chunkShape[lonDimIdx], tileWidth) : tileWidth;

    for (let latIdx = 0; latIdx < latSize; latIdx++) {
      for (let lonIdx = 0; lonIdx < lonSize; lonIdx++) {
        const srcIdx = latIdx * lonSize + lonIdx;
        const dstIdx = latIdx * tileWidth + lonIdx;
        if (srcIdx < flatData.length) {
          paddedData[dstIdx] = flatData[srcIdx];
        }
      }
    }

    return paddedData;
  }

  prerender(gl: WebGL2RenderingContext, matrix: number[]) {
    if (this.isUpdating || !this.program) return;

    gl.useProgram(this.program);
    if (
      gl.canvas.width !== this.canvasWidth ||
      gl.canvas.height !== this.canvasHeight
    ) {
      this.canvasWidth = gl.canvas.width;
      this.canvasHeight = gl.canvas.height;
      this.frameBuffers.current = mustCreateFramebuffer(
        gl,
        this.canvasWidth,
        this.canvasHeight
      );
      this.frameBuffers.next = mustCreateFramebuffer(
        gl,
        this.canvasWidth,
        this.canvasHeight
      );
    }

    // Render to the next framebuffer (clear it first, don't blit old content)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.frameBuffers.next!.framebuffer);
    gl.viewport(0, 0, this.canvasWidth, this.canvasHeight);

    // Clear the framebuffer completely for a fresh render
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    if (!this.cmapTex) {
      this.updateColormapTexture();
      if (!this.cmapTex) return;
    }

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.cmapTex!);
    gl.uniform1i(this.cmapLoc!, 1);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.uniform1f(this.vminLoc!, this.vmin);
    gl.uniform1f(this.vmaxLoc!, this.vmax);
    gl.uniform1f(this.opacityLoc!, this.opacity);
    gl.uniform1f(this.noDataLoc!, this.fillValue);
    gl.uniform1f(this.noDataMinLoc!, this.noDataMin);
    gl.uniform1f(this.noDataMaxLoc!, this.noDataMax);
    gl.uniform1i(this.useFillValueLoc!, this.useFillValue ? 1 : 0);
    gl.uniform1f(this.fillValueLoc!, this.fillValue);
    gl.uniform1f(this.scaleFactorLoc!, this.scaleFactor);
    gl.uniform1f(this.addOffsetLoc!, this.offset);

    gl.uniformMatrix4fv(this.matrixLoc!, false, matrix);

    const worldOffsets = this.getWorldOffsets();

    if (!this.isMultiscale) {
      this.prerenderSingleImage(gl, worldOffsets);
    } else {
      this.prerenderTiles(gl, worldOffsets);
    }

    const temp = this.frameBuffers.current;
    this.frameBuffers.current = this.frameBuffers.next;
    this.frameBuffers.next = temp;

    this.isUpdating = false;
  }

  private prerenderSingleImage(
    gl: WebGL2RenderingContext,
    worldOffsets: number[]
  ) {
    if (
      !this.singleImageData ||
      !this.mercatorBounds ||
      !this.singleImageTexture
    ) {
      this.prefetchTileData();
      return;
    }

    const bounds = this.mercatorBounds;
    const scaleX = (bounds.x1 - bounds.x0) / 2;
    const scaleY = (bounds.y1 - bounds.y0) / 2;
    const shiftX = (bounds.x0 + bounds.x1) / 2;
    const shiftY = (bounds.y0 + bounds.y1) / 2;

    gl.uniform1f(this.scaleLoc!, 0);
    gl.uniform1f(this.scaleXLoc!, scaleX);
    gl.uniform1f(this.scaleYLoc!, scaleY);
    gl.uniform1f(this.shiftXLoc!, shiftX);
    gl.uniform1f(this.shiftYLoc!, shiftY);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.singleImageVertexBuffer!);
    gl.bufferData(gl.ARRAY_BUFFER, this.vertexArr, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.singleImagePixCoordBuffer!);
    gl.bufferData(gl.ARRAY_BUFFER, this.singleImagePixCoordArr, gl.STATIC_DRAW);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.singleImageTexture);
    gl.uniform1i(this.texLoc!, 0);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.R32F,
      this.singleImageWidth,
      this.singleImageHeight,
      0,
      gl.RED,
      gl.FLOAT,
      this.singleImageData
    );

    gl.bindBuffer(gl.ARRAY_BUFFER, this.singleImageVertexBuffer!);
    gl.enableVertexAttribArray(this.vertexLoc);
    gl.vertexAttribPointer(this.vertexLoc, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.singleImagePixCoordBuffer!);
    gl.enableVertexAttribArray(this.pixCoordLoc);
    gl.vertexAttribPointer(this.pixCoordLoc, 2, gl.FLOAT, false, 0, 0);

    for (const worldOffset of worldOffsets) {
      gl.uniform1f(this.worldXOffsetLoc!, worldOffset);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
  }

  private findBestParentTile(
    z: number,
    x: number,
    y: number
  ): {
    tile: TileData;
    ancestorZ: number;
    ancestorX: number;
    ancestorY: number;
  } | null {
    let ancestorZ = z - 1;
    let ancestorX = Math.floor(x / 2);
    let ancestorY = Math.floor(y / 2);

    while (ancestorZ >= 0) {
      const parentKey = tileToKey([ancestorZ, ancestorX, ancestorY]);
      const parentTile = this.tiles.get(parentKey);
      if (parentTile && parentTile.data) {
        return { tile: parentTile, ancestorZ, ancestorX, ancestorY };
      }
      ancestorZ--;
      ancestorX = Math.floor(ancestorX / 2);
      ancestorY = Math.floor(ancestorY / 2);
    }
    return null;
  }

  private getOverzoomTexCoords(
    targetZ: number,
    targetX: number,
    targetY: number,
    ancestorZ: number
  ): Float32Array {
    const levelDiff = targetZ - ancestorZ;
    const divisor = Math.pow(2, levelDiff);

    const localX = targetX % divisor;
    const localY = targetY % divisor;

    const texX0 = localX / divisor;
    const texX1 = (localX + 1) / divisor;
    const texY0 = localY / divisor;
    const texY1 = (localY + 1) / divisor;

    return new Float32Array([
      texX0,
      texY0,
      texX0,
      texY1,
      texX1,
      texY0,
      texX1,
      texY1,
    ]);
  }

  private prerenderTiles(gl: WebGL2RenderingContext, worldOffsets: number[]) {
    const visibleTiles = this.getVisibleTiles();
    this.prefetchTileData();

    gl.uniform1f(this.scaleXLoc!, 0);
    gl.uniform1f(this.scaleYLoc!, 0);

    const tileWidth = this.tileSize;
    const tileHeight = this.tileSize;

    for (const worldOffset of worldOffsets) {
      gl.uniform1f(this.worldXOffsetLoc!, worldOffset);

      for (const tileTuple of visibleTiles) {
        const [z, x, y] = tileTuple;
        const tileKey = tileToKey(tileTuple);
        const tile = this.tiles.get(tileKey);

        let tileToRender: TileData | null = null;
        let texCoords = this.pixCoordArr;

        if (tile && tile.data) {
          tileToRender = tile;
        } else {
          const parent = this.findBestParentTile(z, x, y);
          if (parent) {
            tileToRender = parent.tile;
            texCoords = this.getOverzoomTexCoords(z, x, y, parent.ancestorZ);
          }
        }

        if (!tileToRender || !tileToRender.data) continue;

        const [scale, shiftX, shiftY] = tileToScale(tileTuple);
        gl.uniform1f(this.scaleLoc!, scale);
        gl.uniform1f(this.shiftXLoc!, shiftX);
        gl.uniform1f(this.shiftYLoc!, shiftY);

        gl.bindBuffer(gl.ARRAY_BUFFER, tileToRender.vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, this.vertexArr, gl.STATIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, tileToRender.pixCoordBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, tileToRender.tileTexture);
        gl.uniform1i(this.texLoc!, 0);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.R16F,
          tileWidth,
          tileHeight,
          0,
          gl.RED,
          gl.FLOAT,
          tileToRender.data
        );

        gl.bindBuffer(gl.ARRAY_BUFFER, tileToRender.vertexBuffer);
        gl.enableVertexAttribArray(this.vertexLoc);
        gl.vertexAttribPointer(this.vertexLoc, 2, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, tileToRender.pixCoordBuffer);
        gl.enableVertexAttribArray(this.pixCoordLoc);
        gl.vertexAttribPointer(this.pixCoordLoc, 2, gl.FLOAT, false, 0, 0);

        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }
    }
  }

  render(gl: WebGL2RenderingContext, _matrix: number[]) {
    if (
      this.isRemoved ||
      !this.frameBuffers.current ||
      !this.renderProgram ||
      !this.vertexBuffer
    )
      return;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvasWidth, this.canvasHeight);

    gl.useProgram(this.renderProgram);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.frameBuffers.current.texture);
    gl.uniform1i(this.renderTexLoc!, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW
    );
    gl.enableVertexAttribArray(this.renderVertexLoc);
    gl.vertexAttribPointer(this.renderVertexLoc, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  onRemove(_map: any, gl: WebGL2RenderingContext) {
    this.isRemoved = true;

    if (this.program) {
      gl.deleteProgram(this.program);
      this.program = null;
    }

    if (this.renderProgram) {
      gl.deleteProgram(this.renderProgram);
      this.renderProgram = null;
    }

    if (this.cmapTex) {
      gl.deleteTexture(this.cmapTex);
      this.cmapTex = null;
    }

    if (this.vertexBuffer) {
      gl.deleteBuffer(this.vertexBuffer);
      this.vertexBuffer = null;
    }

    for (const tile of this.tiles.values()) {
      if (tile.tileTexture) gl.deleteTexture(tile.tileTexture);
      if (tile.vertexBuffer) gl.deleteBuffer(tile.vertexBuffer);
      if (tile.pixCoordBuffer) gl.deleteBuffer(tile.pixCoordBuffer);
    }
    this.tiles.clear();
    this.tileAccessOrder = [];

    if (this.singleImageTexture) {
      gl.deleteTexture(this.singleImageTexture);
      this.singleImageTexture = null;
    }
    if (this.singleImageVertexBuffer) {
      gl.deleteBuffer(this.singleImageVertexBuffer);
      this.singleImageVertexBuffer = null;
    }
    if (this.singleImagePixCoordBuffer) {
      gl.deleteBuffer(this.singleImagePixCoordBuffer);
      this.singleImagePixCoordBuffer = null;
    }

    if (this.frameBuffers.current) {
      gl.deleteFramebuffer(this.frameBuffers.current.framebuffer);
      gl.deleteTexture(this.frameBuffers.current.texture);
      this.frameBuffers.current = null;
    }
    if (this.frameBuffers.next) {
      gl.deleteFramebuffer(this.frameBuffers.next.framebuffer);
      gl.deleteTexture(this.frameBuffers.next.texture);
      this.frameBuffers.next = null;
    }

    if (this.zarrStore) {
      this.zarrStore.cleanup();
      this.zarrStore = null;
    }
    this.singleImageData = null;
  }
}
