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

import proj4 from 'proj4';
import * as zarr from 'zarrita';
import {
  type ZarrSelectorsProps,
  type ZarrLevelMetadata,
  type DimensionNamesProps,
  type XYLimitsProps,
  type CRS,
  type DataSliceProps,
  type DimIndicesProps,
  type SliceArgs
} from './types';

const DIMENSION_ALIASES_DEFAULT: { [key in keyof DimensionNamesProps]: string[] } = {
  lat: ['lat', 'latitude', 'y', 'Latitude', 'Y'],
  lon: ['lon', 'longitude', 'x', 'Longitude', 'X', 'lng'],
  time: ['time', 't', 'Time', 'time_counter'],
  elevation: ['depth', 'z', 'Depth', 'level', 'lev', 'deptht', 'elevation', 'depthu', 'depthv']
};

const CF_MAPPINGS: { [key in keyof DimensionNamesProps]: string[] } = {
  lat: ['latitude'],
  lon: ['longitude'],
  time: ['time'],
  elevation: ['height', 'depth', 'altitude', 'air_pressure', 'pressure', 'geopotential_height']
};

/**
 * Identify the indices of common dimensions (lat, lon, time, elevation)
 * in a Zarr array, optionally using CF-compliant standard names or custom dimension mappings.
 *
 * @param dimNames - Names of the array dimensions.
 * @param dimensionNames - Optional explicit mapping of dimension names (see {@link DimensionNamesProps}).
 * @param coordinates - Optional coordinate variable dictionary.
 * @returns A {@link DimIndicesProps} object describing each dimension’s index and name.
 */
export function identifyDimensionIndices(
  dimNames: string[],
  dimensionNames?: DimensionNamesProps,
  coordinates?: Record<string, any>
): DimIndicesProps {
  let DIMENSION_ALIASES = { ...DIMENSION_ALIASES_DEFAULT };
  const names = ['lat', 'lon', 'time', 'elevation'];

  if (coordinates) {
    Object.keys(coordinates).forEach(coordName => {
      const coordArr = coordinates[coordName];
      const coordAttrs = coordArr.attrs as Record<string, any>;
      const standardName = coordAttrs?.standard_name;
      if (standardName) {
        for (const [dimKey, cfNames] of Object.entries(CF_MAPPINGS)) {
          if (cfNames.includes(standardName)) {
            DIMENSION_ALIASES[dimKey as keyof DimensionNamesProps] = [coordName];
          }
        }
      }
    });
  }

  if (dimensionNames) {
    names.forEach(name => {
      const dimName = name as keyof DimensionNamesProps;
      if (dimensionNames[dimName]) {
        DIMENSION_ALIASES[dimName] = [dimensionNames[dimName]] as string[];
      }
    });
    if (dimensionNames.others) {
      dimensionNames.others.forEach(otherName => {
        DIMENSION_ALIASES[otherName as keyof DimensionNamesProps] = [otherName];
      });
    }
  }

  const indices: DimIndicesProps = {};
  for (const [key, aliases] of Object.entries(DIMENSION_ALIASES)) {
    for (let i = 0; i < dimNames.length; i++) {
      const name = dimNames[i].toLowerCase();
      if (aliases.map(a => a.toLowerCase()).includes(name)) {
        indices[key] = { name, index: i, array: coordinates ? coordinates[dimNames[i]] : null };
        break;
      }
    }
  }
  return indices;
}

/**
 * Constructs Zarr slice arguments for extracting a subregion of a multidimensional array.
 *
 * This function:
 * - Converts geographic / elevation slice ranges into Zarr slice objects.
 * - Converts value-based selectors (e.g. `{type: "value", selected: 2020}`) into nearest index selectors.
 * - Optionally loads dimension coordinate arrays for the selected slice.
 * - Produces a *new* selector map describing index-based selections.
 *
 * @param shape               Full array shape.
 * @param dataSlice           Pixel-space slice ranges `{ startX, endX, startY, endY, startElevation?, endElevation? }` (see {@link DataSliceProps}).
 * @param dimIndices          Mapping of dimension names → indices as returned by `identifyDimensionIndices` (see {@link DimIndicesProps}).
 * @param selectors           User-provided selection map (lat/lon/elevation/time/etc.). See {@link ZarrSelectorsProps}.
 * @param dimensionValues     Cache of already-loaded coordinate arrays (mutated by this function).
 * @param root                Root Zarr group location.
 * @param levelInfo           Optional multiscale subpath.
 * @param zarrVersion         Zarr version (2 or 3).
 * @param updateDimensionValues  If true, rewrites dimensionValues only for the selected ranges.
 *
 * @returns An object containing:
 *   - `sliceArgs`: Array of slice objects/indexes matching the array's dimensions. See {@link SliceArgs}.
 *   - `dimensionValues`: Possibly updated coordinate arrays.
 *   - `selectors`: Updated index-based selectors. See {@link ZarrSelectorsProps}.
 */
export async function calculateSliceArgs(
  shape: number[],
  dataSlice: DataSliceProps,
  dimIndices: DimIndicesProps,
  selectors: { [key: string]: ZarrSelectorsProps },
  dimensionValues: { [key: string]: Float64Array | number[] },
  root: zarr.Location<zarr.FetchStore>,
  levelInfo: string | null,
  zarrVersion: 2 | 3 | null,
  updateDimensionValues: boolean = false
): Promise<{
  sliceArgs: SliceArgs;
  dimensionValues: { [key: string]: Float64Array | number[] };
  selectors: { [key: string]: ZarrSelectorsProps };
}> {
  const sliceArgs: SliceArgs = new Array(shape.length).fill(0);
  const newDimensionValues = structuredClone(dimensionValues);
  const newSelectors = structuredClone(selectors);
  for (const dimName of Object.keys(dimIndices)) {
    const dimInfo = dimIndices[dimName];
    if (dimName === 'lon') {
      sliceArgs[dimInfo.index] = zarr.slice(dataSlice.startX, dataSlice.endX);
      if (updateDimensionValues) {
        newDimensionValues[dimName] = await loadDimensionValues(
          newDimensionValues,
          levelInfo,
          dimInfo,
          root,
          zarrVersion,
          [dataSlice.startX, dataSlice.endX]
        );
      }
    } else if (dimName === 'lat') {
      sliceArgs[dimInfo.index] = zarr.slice(dataSlice.startY, dataSlice.endY);
      if (updateDimensionValues) {
        newDimensionValues[dimName] = await loadDimensionValues(
          newDimensionValues,
          levelInfo,
          dimInfo,
          root,
          zarrVersion,
          [dataSlice.startY, dataSlice.endY]
        );
      }
    } else if (
      dimName === 'elevation' &&
      dataSlice.startElevation !== undefined &&
      dataSlice.endElevation !== undefined
    ) {
      sliceArgs[dimInfo.index] = zarr.slice(dataSlice.startElevation, dataSlice.endElevation);
      newSelectors[dimName] = {
        type: 'index',
        selected: [dataSlice.startElevation, dataSlice.endElevation]
      };
      if (updateDimensionValues) {
        newDimensionValues[dimName] = await loadDimensionValues(
          newDimensionValues,
          levelInfo,
          dimInfo,
          root,
          zarrVersion,
          [dataSlice.startElevation, dataSlice.endElevation]
        );
      }
    } else {
      const dimSelection = newSelectors[dimName];
      if (!dimSelection) {
        newSelectors[dimName] = { type: 'index', selected: 0 };
        sliceArgs[dimInfo.index] = 0;
      } else if (dimSelection.type === 'value') {
        try {
          newDimensionValues[dimName] = await loadDimensionValues(
            newDimensionValues,
            levelInfo,
            dimInfo,
            root,
            zarrVersion
          );
          const nearestIdx = calculateNearestIndex(
            newDimensionValues[dimName],
            dimSelection.selected as number
          );
          newSelectors[dimName] = { type: 'index', selected: nearestIdx };
          sliceArgs[dimInfo.index] = nearestIdx;
        } catch (err) {
          sliceArgs[dimInfo.index] = 0;
        }
      } else {
        newSelectors[dimName] = { type: 'index', selected: dimSelection.selected };
        sliceArgs[dimInfo.index] = dimSelection.selected as number;
      }

      newDimensionValues[dimName] = await loadDimensionValues(
        newDimensionValues,
        levelInfo,
        dimInfo,
        root,
        zarrVersion
      );
    }
  }
  return { sliceArgs, dimensionValues: newDimensionValues, selectors: newSelectors };
}

/**
 * Constructs Zarr slice arguments for extracting a subregion of a multidimensional array.
 *
 * This function:
 * - Converts geographic / elevation slice ranges into Zarr slice objects.
 * - Converts value-based selectors (e.g. `{type: "value", selected: 2020}`) into nearest index selectors.
 * - Optionally loads dimension coordinate arrays for the selected slice.
 * - Produces a *new* selector map describing index-based selections.
 *
 * @param shape               Full array shape.
 * @param dataSlice           Pixel-space slice ranges `{ startX, endX, startY, endY, startElevation?, endElevation? }` (see {@link DataSliceProps}).
 * @param dimIndices          Mapping of dimension names → indices as returned by `identifyDimensionIndices` (see {@link DimIndicesProps}).
 * @param selectors           User-provided selection map (lat/lon/elevation/time/etc.). See {@link ZarrSelectorsProps}.
 *
 * @returns An object containing:
 *   - `sliceArgs`: Array of slice objects/indexes matching the array's dimensions. See {@link SliceArgs}.
 *   - `dimensionValues`: Possibly updated coordinate arrays.
 *   - `selectors`: Updated index-based selectors. See {@link ZarrSelectorsProps}.
 */
export function calculateSliceArgsRequestImage(
  shape: number[],
  dataSlice: DataSliceProps,
  dimIndices: DimIndicesProps,
  selectors: { [key: string]: ZarrSelectorsProps }
): SliceArgs {
  const sliceArgs: SliceArgs = new Array(shape.length).fill(0);
  for (const dimName of Object.keys(dimIndices)) {
    const dimInfo = dimIndices[dimName];
    if (dimName === 'lon') {
      sliceArgs[dimInfo.index] = zarr.slice(dataSlice.startX, dataSlice.endX);
    } else if (dimName === 'lat') {
      sliceArgs[dimInfo.index] = zarr.slice(dataSlice.startY, dataSlice.endY);
    } else {
      const dimSelection = selectors[dimName];
      sliceArgs[dimInfo.index] = dimSelection.selected as number;
    }
  }
  return sliceArgs;
}

/**
 * Finds the index of the value in `values` nearest to `target`.
 * @param values - Array of numeric values.
 * @param target - Target value to find.
 * @returns Index of the nearest value.
 */
export function calculateNearestIndex(values: Float64Array | number[], target: number): number {
  const selectedValue = target;
  let nearestIdx = 0;
  let minDiff = Infinity;
  values.forEach((val, i) => {
    const diff = Math.abs(val - selectedValue);
    if (diff < minDiff) {
      minDiff = diff;
      nearestIdx = i;
    }
  });
  return nearestIdx;
}

/**
 * Determines the index range of the elevation axis to load from a Zarr cube.
 *
 * Supports two kinds of selectors:
 * - `{ type: "index", selected: [i0, i1] }` — direct index slicing
 * - `{ type: "value", selected: [z0, z1] }` — find nearest elevation values
 *
 * Behavior:
 * - Loads elevation coordinate values (if not already loaded).
 * - Converts value-based ranges into nearest-index ranges.
 * - If `selected` is a single scalar, slices from the lowest elevation to that value.
 * - Returns the index bounds as `[start, endExclusive]`.
 * - Mutates `dimensionValuesWithElevation.elevation` by slicing it to the returned range.
 *
 * @param shapeElevation                 Size of the elevation dimension.
 * @param dimInfo                       Dimension index info for elevation. See {@link DimIndicesProps}.
 * @param selectorsElevation            User-provided elevation selector. See {@link ZarrSelectorsProps}.
 * @param dimensionValuesWithElevation  Cache of already-loaded coordinate arrays (mutated by this function).
 * @param root                          Root Zarr group location.
 * @param levelInfo                     Optional multiscale subpath.
 * @param zarrVersion                   Zarr version (2 or 3).
 *
 * @returns An object containing:
 * - `dimensionValuesWithElevation`: Possibly updated elevation coordinate array.
 * - `elevationSlice`: Index range `[start, endExclusive]` for elevation slicing.
 */
export async function calculateElevationSlice(
  shapeElevation: number,
  dimInfo: DimIndicesProps['elevation'],
  selectorsElevation: ZarrSelectorsProps | undefined,
  dimensionValuesWithElevation: { [key: string]: Float64Array | number[] },
  root: zarr.Location<zarr.FetchStore>,
  levelInfo: string | null,
  zarrVersion: 2 | 3 | null
): Promise<{
  dimensionValuesWithElevation: { [key: string]: Float64Array | number[] };
  elevationSlice: [number, number];
}> {
  dimensionValuesWithElevation['elevation'] = await loadDimensionValues(
    dimensionValuesWithElevation,
    levelInfo,
    dimInfo,
    root,
    zarrVersion
  );

  let startElevation = 0;
  let endElevation = shapeElevation;
  if (selectorsElevation?.selected === undefined) {
    return { dimensionValuesWithElevation, elevationSlice: [startElevation, endElevation] };
  }
  if (selectorsElevation) {
    let firstElevation: number | null = null;
    let secondElevation;
    if (typeof selectorsElevation.selected === 'object') {
      firstElevation = selectorsElevation.selected[0];
      secondElevation = selectorsElevation.selected[1];
    } else {
      secondElevation = selectorsElevation.selected as number;
    }
    if (firstElevation !== null && firstElevation > secondElevation) {
      console.warn('Invalid elevation selection: start value is greater than end value.');
      return { dimensionValuesWithElevation, elevationSlice: [startElevation, endElevation] };
    }
    if (selectorsElevation.type !== 'value') {
      startElevation = firstElevation !== null ? (firstElevation as number) : 0;
      endElevation = Math.min(secondElevation + 1, shapeElevation);
    } else {
      const elevationValues = dimensionValuesWithElevation['elevation'];
      if (firstElevation === null) {
        firstElevation = Math.min(...(elevationValues as number[]));
      }
      let nearestIdxFirst = 0;
      let minDiffStartFirst = Infinity;
      let nearestIdxSecond = 0;
      let minDiffStartSecond = Infinity;
      elevationValues.forEach((val, i) => {
        const diffFirst = Math.abs(val - (firstElevation as number));
        const diffSecond = Math.abs(val - (secondElevation as number));
        if (diffFirst < minDiffStartFirst) {
          minDiffStartFirst = diffFirst;
          nearestIdxFirst = i;
        }
        if (diffSecond < minDiffStartSecond) {
          minDiffStartSecond = diffSecond;
          nearestIdxSecond = i;
        }
      });
      startElevation = nearestIdxFirst;
      endElevation = nearestIdxSecond;
    }
  }
  dimensionValuesWithElevation['elevation'] = dimensionValuesWithElevation['elevation'].slice(
    startElevation,
    endElevation
  );

  return { dimensionValuesWithElevation, elevationSlice: [startElevation, endElevation] };
}

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
export async function loadDimensionValues(
  dimensionValues: Record<string, Float64Array | number[]>,
  levelInfo: string | null,
  dimIndices: DimIndicesProps[string],
  root: zarr.Location<zarr.FetchStore>,
  zarrVersion: 2 | 3 | null,
  slice?: [number, number]
): Promise<Float64Array | number[]> {
  if (dimensionValues[dimIndices.name]) return dimensionValues[dimIndices.name];
  let targetRoot;
  if (levelInfo) {
    targetRoot = await root.resolve(levelInfo);
  } else {
    targetRoot = root;
  }
  let coordArr;
  if (dimIndices.array) {
    coordArr = dimIndices.array;
  } else {
    const coordVar = await targetRoot.resolve(dimIndices.name);
    let localFunc = zarr.open as any;
    if (zarrVersion === 2) {
      localFunc = zarr.open.v2;
    } else if (zarrVersion === 3) {
      localFunc = zarr.open.v3;
    }
    coordArr = await localFunc(coordVar, { kind: 'array' });
  }
  const coordData = await zarr.get(coordArr);
  const coordArray = Array.from(coordData.data as number[], (v: number | bigint) =>
    typeof v === 'bigint' ? Number(v) : v
  );
  if (slice) {
    return coordArray.slice(slice[0], slice[1]);
  }
  return coordArray;
}

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
export async function initZarrDataset(
  store: zarr.FetchStore,
  root: zarr.Location<zarr.FetchStore>,
  variable: string,
  dimensions: DimensionNamesProps,
  levelMetadata: Map<number, ZarrLevelMetadata>,
  levelCache: Map<number, any>,
  zarrVersion: 2 | 3 | null,
  multiscaleLevel?: number
): Promise<{
  zarrArray: zarr.Array<any>;
  levelInfos: string[];
  dimIndices: DimIndicesProps;
  attrs: Record<string, any>;
  multiscaleLevel?: number;
}> {
  let localFunc = zarr.open as any;
  if (zarrVersion === 2) {
    localFunc = zarr.open.v2;
  } else if (zarrVersion === 3) {
    localFunc = zarr.open.v3;
  }
  const zarrGroup = await localFunc(root, { kind: 'group' });
  const attrs = (zarrGroup.attrs ?? {}) as Record<string, any>;
  let zarrArray: zarr.Array<any> | null = null;
  let levelInfos: string[] = [];
  let coordinates: Record<string, any> = {};
  let datasets;
  let pyramidMode = false;
  if (attrs.multiscales && attrs.multiscales[0]?.datasets?.length) {
    pyramidMode = true;
    datasets = attrs.multiscales[0].datasets;
    if (multiscaleLevel) datasets = [datasets[multiscaleLevel]];

    for (let i = 0; i < datasets.length; i++) {
      const levelPath = datasets[i].path;
      levelInfos.push(levelPath);
      const levelArr = await openLevelArray(root, levelPath, variable, levelCache);

      const levelRoot = await root.resolve(levelPath);
      const { existingCoordinates, array_dimensions } = await calculateCoordinatesFromAttrs(
        levelArr,
        levelRoot,
        coordinates,
        store,
        variable,
        zarrVersion,
        pyramidMode
      );
      coordinates = existingCoordinates;
      const dims = identifyDimensionIndices(array_dimensions, dimensions, coordinates);

      const width = levelArr.shape[dims.lon.index];
      const height = levelArr.shape[dims.lat.index];

      levelMetadata.set(i, { width, height });
    }
    if (multiscaleLevel) {
      datasets = attrs.multiscales[0].datasets;
      levelInfos = [];
      for (let i = 0; i < datasets.length; i++) {
        levelInfos.push(datasets[i].path);
      }
    }
    let levelInfo = levelInfos[multiscaleLevel || 0];
    if (!levelInfo) {
      console.error(
        'No level info found for multiscale level:',
        multiscaleLevel,
        '. Using 0 instead.'
      );
      multiscaleLevel = 0;
      levelInfo = levelInfos[multiscaleLevel];
    }
    zarrArray = await openLevelArray(root, levelInfo, variable, levelCache);
  } else {
    const arrayLocation = await root.resolve(variable);
    let localFunc = zarr.open as any;
    if (zarrVersion === 2) {
      localFunc = zarr.open.v2;
    } else if (zarrVersion === 3) {
      localFunc = zarr.open.v3;
    }
    zarrArray = await localFunc(arrayLocation, { kind: 'array' });
  }
  if (!zarrArray) {
    throw new Error('Failed to initialize Zarr array');
  }
  const { existingCoordinates, array_dimensions } = await calculateCoordinatesFromAttrs(
    zarrArray,
    root,
    coordinates,
    store,
    variable,
    zarrVersion,
    pyramidMode
  );
  const dimIndices = await identifyDimensionIndices(
    array_dimensions,
    dimensions,
    existingCoordinates
  );
  return {
    zarrArray,
    levelInfos,
    dimIndices,
    attrs,
    multiscaleLevel
  };
}

/**
 * Retrieves coordinate arrays from `_ARRAY_DIMENSIONS` attributes in the Zarr array metadata.
 * Reuses existing coordinates if already computed.
 *
 * @param arr - Zarr array.
 * @param root - Zarr group root.
 * @param existingCoordinates - Already loaded coordinate arrays.
 * @param store - Zarr store.
 * @param variable - Variable name.
 * @param zarrVersion - Zarr version (2 or 3).
 * @param isPyramid - Whether the dataset is multiscale.
 *
 * @returns An object containing:
 * - `existingCoordinates`: Loaded coordinate arrays.
 * - `array_dimensions`: Names of the array dimensions.
 */
async function calculateCoordinatesFromAttrs(
  arr: zarr.Array<any>,
  root: zarr.Location<zarr.FetchStore>,
  existingCoordinates: Record<string, any>,
  store: zarr.FetchStore,
  variable: string,
  zarrVersion: 2 | 3 | null,
  isPyramid: boolean = false
): Promise<{ existingCoordinates: Record<string, any>; array_dimensions: string[] }> {
  let array_dimensions: string[] = (arr.attrs['_ARRAY_DIMENSIONS'] as string[]) || [];
  if (array_dimensions.length === 0) {
    const location = zarr.root(store);
    const rootMetadata = JSON.parse(
      new TextDecoder().decode(await store.get(location.resolve('zarr.json').path))
    );
    if (isPyramid) {
      array_dimensions =
        rootMetadata.consolidated_metadata.metadata[`0/${variable}`]['dimension_names'];
    } else {
      array_dimensions = rootMetadata.consolidated_metadata.metadata[variable]['dimension_names'];
    }
  }
  if (Object.keys(existingCoordinates).length > 0) return { existingCoordinates, array_dimensions };

  for (let i = 0; i < array_dimensions.length; i++) {
    const dimName = array_dimensions[i];
    const coordVar = await root.resolve(dimName);
    let localFunc = zarr.open as any;
    if (zarrVersion === 2) {
      localFunc = zarr.open.v2;
    } else if (zarrVersion === 3) {
      localFunc = zarr.open.v3;
    }
    existingCoordinates[dimName] = await localFunc(coordVar, { kind: 'array' });
  }
  return { existingCoordinates, array_dimensions };
}

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
export async function getXYLimits(
  root: zarr.Location<zarr.FetchStore>,
  dimIndices: DimIndicesProps,
  levelInfos: string[],
  multiscale: boolean,
  zarrVersion: 2 | 3 | null
): Promise<XYLimitsProps> {
  const levelRoot = multiscale ? await root.resolve(levelInfos[0]) : root;
  let localFunc = zarr.open as any;
  if (zarrVersion === 2) {
    localFunc = zarr.open.v2;
  } else if (zarrVersion === 3) {
    localFunc = zarr.open.v3;
  }
  const xarr =
    dimIndices.lon.array ||
    (await localFunc(await levelRoot.resolve(dimIndices.lon.name), { kind: 'array' }));
  const yarr =
    dimIndices.lat.array ||
    (await localFunc(await levelRoot.resolve(dimIndices.lat.name), { kind: 'array' }));

  const xdata = (await zarr.get(xarr)) as any;
  const ydata = (await zarr.get(yarr)) as any;

  const xyLimits = {
    xMin: Math.min(...xdata.data),
    xMax: Math.max(...xdata.data),
    yMin: Math.min(...ydata.data),
    yMax: Math.max(...ydata.data)
  };
  return xyLimits;
}

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
export async function openLevelArray(
  root: zarr.Location<zarr.FetchStore>,
  levelPath: string,
  variable: string,
  levelCache: Map<number, any>,
  zarrVersion: 2 | 3 | null = null
): Promise<zarr.Array<any>> {
  const existing = Array.from(levelCache.entries()).find(([_, val]) => val.path === levelPath);
  if (existing) return existing[1];

  const levelRoot = await root.resolve(levelPath);
  const arrayLoc = variable ? await levelRoot.resolve(variable) : levelRoot;
  let localFunc = zarr.open as any;
  if (zarrVersion === 2) {
    localFunc = zarr.open.v2;
  } else if (zarrVersion === 3) {
    localFunc = zarr.open.v3;
  }
  const arr = await localFunc(arrayLoc, { kind: 'array' });

  const levelIndex = levelCache.size;
  levelCache.set(levelIndex, arr);
  if (levelCache.size > 3) {
    const firstKey = levelCache.keys().next().value as number;
    levelCache.delete(firstKey);
  }

  return arr;
}

/**
 * Resolves the no-data value range for masking dataset values.
 *
 * Priority order:
 * 1. User-specified min/max
 * 2. Dataset metadata min/max
 * 3. Hardcoded fallback (-9999 to 9999)
 *
 * @param userMin - User-defined no-data minimum value.
 * @param userMax - User-defined no-data maximum value.
 * @param metadataMin - Metadata-defined valid minimum value.
 * @param metadataMax - Metadata-defined valid maximum value.
 *
 * @returns An object containing:
 *  - `noDataMin`: Resolved no-data minimum value.
 *  - `noDataMax`: Resolved no-data maximum value.
 */
export function resolveNoDataRange(
  userMin: number | undefined,
  userMax: number | undefined,
  metadataMin: number | undefined,
  metadataMax: number | undefined
): { noDataMin: number; noDataMax: number } {
  if (userMin !== undefined && userMax !== undefined) {
    return { noDataMin: userMin, noDataMax: userMax };
  }

  if (metadataMin !== undefined && metadataMax !== undefined) {
    return { noDataMin: metadataMin, noDataMax: metadataMax };
  }

  return { noDataMin: -9999, noDataMax: 9999 };
}

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
export function extractNoDataMetadata(zarrArray: zarr.Array<any>): {
  metadataMin: number | undefined;
  metadataMax: number | undefined;
  fillValue: number | undefined;
  useFillValue: boolean;
} {
  const attrs = zarrArray.attrs || {};

  let metadataMin: number | undefined = undefined;
  let metadataMax: number | undefined = undefined;
  let fillValue: number | undefined = undefined;
  let useFillValue = false;

  if (attrs.valid_min !== undefined) metadataMin = attrs.valid_min as number;
  if (attrs.valid_max !== undefined) metadataMax = attrs.valid_max as number;

  if (attrs._FillValue !== undefined) {
    fillValue = attrs._FillValue as number;
    useFillValue = true;
  } else if (attrs.missing_value !== undefined) {
    fillValue = attrs.missing_value as number;
    useFillValue = true;
  }

  return { metadataMin, metadataMax, fillValue, useFillValue };
}

/**
 * Detects the coordinate reference system (CRS) of a Zarr dataset based on metadata or coordinate range.
 * Defaults to EPSG:4326 (WGS84) if uncertain.
 *
 * @param attrs - Zarr array or group attributes.
 * @param arr - Zarr array (may be null).
 * @param xyLimits - Optional geographic coordinate limits. See {@link XYLimitsProps}.
 * @returns Detected  CRS as a string (e.g., 'EPSG:4326' or 'EPSG:3857'. See {@link CRS}).
 */
export async function detectCRS(
  attrs: Record<string, any>,
  arr: zarr.Array<any> | null,
  xyLimits?: XYLimitsProps
): Promise<CRS> {
  const attrCRS = attrs?.multiscales?.[0]?.datasets?.[0]?.crs ?? arr?.attrs?.crs;
  if (attrCRS) {
    return attrCRS;
  }
  if (!xyLimits) {
    return 'EPSG:4326';
  }
  const xMax = xyLimits.xMax;
  return xMax && Math.abs(xMax) > 360 ? 'EPSG:3857' : 'EPSG:4326';
}

/**
 * Computes cube dimension ordering and strides based on dimension indices.
 * Useful for reshaping 3D Zarr arrays into Cesium-renderable layouts.
 *
 * @param cubeDimensions - Original cube dimensions [nx, ny, nz].
 * @param dimIndices - Dimension index mapping. See {@link DimIndicesProps}.
 * @returns An object containing:
 *   - `nx`, `ny`, `nz`: Original cube dimensions.
 *   - `indicesOrder`: Ordered dimension names.
 *   - `strides`: Stride values for each dimension.
 */
export function getCubeDimensions(
  cubeDimensions: [number, number, number],
  dimIndices: DimIndicesProps
): { nx: number; ny: number; nz: number; indicesOrder: string[]; strides: Record<string, number> } {
  const [nx, ny, nz] = cubeDimensions;
  const names = ['lat', 'lon', 'elevation'];
  const indicesOrder = names.slice().sort((a, b) => dimIndices[a].index - dimIndices[b].index);
  const dims: Record<string, number> = { lon: nx, lat: ny, elevation: nz };
  const strides: Record<string, number> = {};
  strides[indicesOrder[2]] = 1;
  strides[indicesOrder[1]] = dims[indicesOrder[2]];
  strides[indicesOrder[0]] = dims[indicesOrder[1]] * dims[indicesOrder[2]];

  return { nx, ny, nz, indicesOrder, strides };
}

/**
 * Converts elevation index to Cesium height (meters),
 * applying vertical exaggeration and optional below-sea-level offset.
 *
 * @param elevationValue - Scalar elevation value or index.
 * @param elevationArray - Full elevation coordinate array.
 * @param verticalExaggeration - Exaggeration multiplier.
 * @param belowSeaLevel - Whether elevations are below sea level.
 * @param flipElevation - If true, inverts elevation direction.
 *
 * @returns Height in meters for Cesium rendering.
 */
export function calculateHeightMeters(
  elevationValue: number,
  elevationArray: Float64Array | number[] | undefined,
  verticalExaggeration: number,
  belowSeaLevel: boolean | undefined,
  flipElevation: boolean = false
): number {
  const maxElevationValue = Math.max(...(elevationArray as number[]));
  let firstElement: number;
  if (belowSeaLevel) {
    firstElement = -(flipElevation ? elevationValue : maxElevationValue - elevationValue);
  } else {
    firstElement = flipElevation ? maxElevationValue - elevationValue : elevationValue;
  }
  return firstElement * verticalExaggeration;
}

/**
 * Converts geographic bounds (lat/lon) to pixel-space indices for slicing Zarr arrays.
 * Supports both EPSG:4326 and EPSG:3857 projections.
 *
 * @param bounds - Geographic bounding box.
 * @param width - Array width (longitude dimension).
 * @param height - Array height (latitude dimension).
 * @param crs - Coordinate reference system. See {@link CRS}.
 * @returns Start and end pixel indices for X and Y axes.
 */
export function calculateXYFromBounds(
  bounds: { west: number; south: number; east: number; north: number },
  width: number,
  height: number,
  crs: CRS | null
): { x: [number, number]; y: [number, number] } {
  if (crs === 'EPSG:3857') {
    const sourceCRS: CRS = 'EPSG:4326';
    const [xWest, ySouth] = proj4(sourceCRS, crs, [bounds.west, bounds.south]);
    const [xEast, yNorth] = proj4(sourceCRS, crs, [bounds.east, bounds.north]);
    const worldExtent = 20037508.342789244;
    const xMin = Math.floor(((xWest + worldExtent) / (2 * worldExtent)) * width);
    const xMax = Math.floor(((xEast + worldExtent) / (2 * worldExtent)) * width);
    const yMin = Math.floor(((worldExtent - yNorth) / (2 * worldExtent)) * height);
    const yMax = Math.floor(((worldExtent - ySouth) / (2 * worldExtent)) * height);
    return { x: [xMin, xMax], y: [yMin, yMax] };
  } else {
    const xMin = Math.floor(((bounds.west + 180) / 360) * width);
    const xMax = Math.floor(((bounds.east + 180) / 360) * width);
    const yMin = Math.floor(((90 - bounds.north) / 180) * height);
    const yMax = Math.floor(((90 - bounds.south) / 180) * height);
    return {
      x: [xMin, xMax],
      y: [yMin, yMax]
    };
  }
}
