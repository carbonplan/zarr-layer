import type { Selector, SelectorValue } from './types'
import { sanitizeGlslName } from './zarr-utils'

/**
 * Configuration object returned by band math helper functions.
 * Spread this into your ZarrLayer options. The returned `clim` is a sensible
 * default that can be overridden by setting `clim` after spreading.
 */
export interface BandMathConfig {
  selector: Selector
  customFrag: string
  clim: [number, number]
}

/**
 * Options for the NDVI calculation.
 */
export interface NdviOptions {
  /** Band name/value for near-infrared (default: 'B08' for Sentinel-2) */
  nir: string | number
  /** Band name/value for red (default: 'B04' for Sentinel-2) */
  red: string | number
  /** Dimension name for bands (default: 'band') */
  dimension?: string
}

/**
 * Options for true color RGB display.
 */
export interface TrueColorOptions {
  /** Band name/value for red channel (default: 'B04' for Sentinel-2) */
  red: string | number
  /** Band name/value for green channel (default: 'B03' for Sentinel-2) */
  green: string | number
  /** Band name/value for blue channel (default: 'B02' for Sentinel-2) */
  blue: string | number
  /** Dimension name for bands (default: 'band') */
  dimension?: string
}

/**
 * Converts a band value to a valid GLSL variable name.
 * Numbers become `${dimension}_${value}`, strings are sanitized to valid GLSL identifiers.
 */
function toGlslName(dimension: string, value: string | number): string {
  if (typeof value === 'number') {
    return sanitizeGlslName(`${dimension}_${value}`)
  }
  return sanitizeGlslName(value)
}

/**
 * Helper to build a selector value array with proper typing.
 * SelectorValue requires homogeneous arrays (string[] or number[]), so we
 * assert the type based on the first element when types are mixed.
 */
function buildSelectorArray(values: (string | number)[]): SelectorValue {
  const allStrings = values.every((v) => typeof v === 'string')
  const allNumbers = values.every((v) => typeof v === 'number')

  if (allStrings) return values as string[]
  if (allNumbers) return values as number[]
  // Mixed types: SelectorValue doesn't support (string|number)[], so we
  // coerce to string[] which will work at runtime (values are used for lookup)
  return values as unknown as string[]
}

/**
 * Creates a configuration for NDVI (Normalized Difference Vegetation Index).
 *
 * NDVI = (NIR - Red) / (NIR + Red)
 *
 * Returns a default `clim` of `[-1, 1]` which covers the full NDVI range.
 * Override by setting `clim` after spreading the config.
 *
 * @example
 * ```ts
 * import { ZarrLayer, ndvi } from '@carbonplan/zarr-layer'
 *
 * const config = ndvi({ nir: 'B08', red: 'B04' })
 * new ZarrLayer({
 *   source: 'https://example.com/data.zarr',
 *   variable: 'data',
 *   colormap: 'rdylgn',
 *   ...config,
 *   // Optionally override clim or merge additional selector dimensions:
 *   clim: [0, 0.8], // focus on vegetation range
 *   selector: { ...config.selector, time: 0 },
 * })
 * ```
 */
export function ndvi(options: NdviOptions): BandMathConfig {
  const { nir, red, dimension = 'band' } = options

  const nirName = toGlslName(dimension, nir)
  const redName = toGlslName(dimension, red)

  return {
    selector: {
      [dimension]: buildSelectorArray([nir, red]),
    },
    customFrag: `
  float ndvi = (${nirName} - ${redName}) / (${nirName} + ${redName});
  float norm = (ndvi - clim.x) / (clim.y - clim.x);
  vec4 c = texture(colormap, vec2(clamp(norm, 0.0, 1.0), 0.5));
  fragColor = vec4(c.rgb, opacity);
`,
    clim: [-1, 1],
  }
}

/**
 * Creates a configuration for true color RGB display.
 *
 * Maps three bands to RGB channels for natural color visualization.
 *
 * Returns a default `clim` of `[0, 1]` which assumes normalized reflectance values.
 * For raw satellite data (e.g., Sentinel-2 surface reflectance 0-10000), override
 * `clim` after spreading the config: `clim: [0, 3000]`.
 *
 * @example
 * ```ts
 * import { ZarrLayer, trueColor } from '@carbonplan/zarr-layer'
 *
 * const config = trueColor({ red: 'B04', green: 'B03', blue: 'B02' })
 * new ZarrLayer({
 *   source: 'https://example.com/data.zarr',
 *   variable: 'data',
 *   ...config,
 *   // Override clim for raw reflectance data:
 *   clim: [0, 3000],
 *   // Optionally merge additional selector dimensions:
 *   selector: { ...config.selector, time: 0 },
 * })
 * ```
 */
export function trueColor(options: TrueColorOptions): BandMathConfig {
  const { red, green, blue, dimension = 'band' } = options

  const redName = toGlslName(dimension, red)
  const greenName = toGlslName(dimension, green)
  const blueName = toGlslName(dimension, blue)

  return {
    selector: {
      [dimension]: buildSelectorArray([red, green, blue]),
    },
    customFrag: `
  float r = (${redName} - clim.x) / (clim.y - clim.x);
  float g = (${greenName} - clim.x) / (clim.y - clim.x);
  float b = (${blueName} - clim.x) / (clim.y - clim.x);
  fragColor = vec4(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0), opacity);
`,
    clim: [0, 1],
  }
}
