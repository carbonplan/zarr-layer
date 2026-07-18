/**
 * @module __fixtures__/memory-zarr
 *
 * Builds an in-memory, zarrita-compatible Zarr v3 store from a declarative
 * spec — no network, filesystem, or GL. The returned object implements the
 * zarrita `Readable` interface (`get(key)`), so it can be passed to
 * `new ZarrStore({ customStore })` to exercise the real metadata → description
 * pipeline against frozen fixture bytes.
 *
 * Store keys use zarrita's leading-slash convention (`/zarr.json`,
 * `/lat/zarr.json`, `/lat/c/0`). Arrays are written uncompressed with the v3
 * `bytes` codec, so chunk bytes are just the raw little-endian typed-array
 * buffer.
 */

export interface ArraySpec {
  /** Array name / path segment (e.g. 'temperature', 'lat'). */
  name: string
  shape: number[]
  chunkShape: number[]
  /** Defaults to 'float32'. */
  dtype?: 'float32' | 'float64'
  fillValue?: number | null
  dimensionNames?: string[]
  /** Array-level attributes (e.g. scale_factor, add_offset). */
  attributes?: Record<string, unknown>
  /**
   * Chunk data keyed by chunk index joined with '/' (matching the v3 default
   * chunk-key encoding). A 1D array's only chunk is `'0'`; a 3D array's is
   * `'0/0/0'`.
   */
  chunks?: Record<string, ArrayLike<number>>
}

export interface ZarrSpec {
  /** Root group attributes (e.g. `multiscales`). */
  attributes?: Record<string, unknown>
  arrays: ArraySpec[]
}

const CHUNK_ENCODERS: Record<string, (data: ArrayLike<number>) => Uint8Array> =
  {
    float32: (data) => new Uint8Array(Float32Array.from(data).buffer),
    float64: (data) => new Uint8Array(Float64Array.from(data).buffer),
  }

/** Minimal Readable shape; matches zarrita's `Readable` structurally. */
export interface MemoryStore {
  get(key: string): Promise<Uint8Array | undefined>
}

export function buildMemoryZarrStore(spec: ZarrSpec): MemoryStore {
  const map = new Map<string, Uint8Array>()
  const enc = new TextEncoder()
  const writeJson = (key: string, value: unknown) =>
    map.set(key, enc.encode(JSON.stringify(value)))

  writeJson('/zarr.json', {
    zarr_format: 3,
    node_type: 'group',
    attributes: spec.attributes ?? {},
  })

  for (const a of spec.arrays) {
    const dtype = a.dtype ?? 'float32'
    writeJson(`/${a.name}/zarr.json`, {
      zarr_format: 3,
      node_type: 'array',
      shape: a.shape,
      data_type: dtype,
      chunk_grid: {
        name: 'regular',
        configuration: { chunk_shape: a.chunkShape },
      },
      chunk_key_encoding: {
        name: 'default',
        configuration: { separator: '/' },
      },
      codecs: [{ name: 'bytes', configuration: { endian: 'little' } }],
      fill_value: a.fillValue ?? null,
      dimension_names: a.dimensionNames,
      attributes: a.attributes ?? {},
    })

    const encode = CHUNK_ENCODERS[dtype]
    for (const [idx, data] of Object.entries(a.chunks ?? {})) {
      map.set(`/${a.name}/c/${idx}`, encode(data))
    }
  }

  return { get: async (key: string) => map.get(key) }
}

/** Build a row-major ramp [0, 1, ..., n-1] of the given length. */
export function ramp(length: number): number[] {
  return Array.from({ length }, (_, i) => i)
}
