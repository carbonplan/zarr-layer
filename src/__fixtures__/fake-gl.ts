/**
 * @module __fixtures__/fake-gl
 *
 * A recording stand-in for WebGL2RenderingContext. Programs "compile" and
 * "link" unconditionally; what the stub actually models is the one GL behavior
 * the renderer's control flow depends on: which uniforms resolve to a location.
 *
 * A real driver reports a uniform as active only when it is declared AND
 * reachable from main(), and the renderer leans on that — `matrixLoc`,
 * `eyeMatrixLoc`, and `anchorClipLoc` are expected to come back null on the
 * variants that don't use them, which is what makes their uploads no-op. The
 * stub approximates the rule by looking for a declaration plus a reference
 * outside it. It does not dead-code-eliminate, so a uniform referenced only
 * from an uncalled function still resolves here (real GL may drop it); nothing
 * in the renderer distinguishes the two cases, since every such upload is
 * null-guarded.
 */

import { vi } from 'vitest'

export interface UniformLocationStub {
  name: string
}

/** Comments are not code: the uniform blocks document each other by name, and
 *  a mention there must not read as a reference. */
const stripComments = (source: string) =>
  source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '')

const isActiveUniform = (source: string, name: string): boolean => {
  const code = stripComments(source)
  const declaration = new RegExp(`uniform\\s+\\w+\\s+${name}\\s*;`)
  if (!declaration.test(code)) return false
  return new RegExp(`\\b${name}\\b`).test(code.replace(declaration, ''))
}

/** A MapLibre-shaped vertex prelude: declares the projection uniforms the
 *  maplibre variants read and references them from projectTile(). */
export const FAKE_MAPLIBRE_PRELUDE = `
const float PI = 3.14159265358979323846;
uniform mat4 u_projection_matrix;
uniform mat4 u_projection_fallback_matrix;
uniform vec4 u_projection_tile_mercator_coords;
uniform vec4 u_projection_clipping_plane;
uniform float u_projection_transition;
vec4 projectTile(vec2 p) {
  vec4 flat_pos = u_projection_fallback_matrix * vec4(p, 0.0, 1.0);
  vec4 globe_pos = u_projection_matrix * vec4(p, 0.0, 1.0);
  globe_pos.z += dot(u_projection_clipping_plane, u_projection_tile_mercator_coords);
  return mix(flat_pos, globe_pos, u_projection_transition);
}`

export const FAKE_SHADER_DATA = {
  vertexShaderPrelude: FAKE_MAPLIBRE_PRELUDE,
  define: '#define ZARR_FAKE 1',
  variantName: 'fake-variant',
}

export interface RecordedCall {
  name: string
  args: unknown[]
}

export interface RecordingGl extends WebGL2RenderingContext {
  /** Every uniform/draw call in order, for assertions on what was uploaded. */
  calls: RecordedCall[]
  /** Vertex + fragment sources handed to each linked program. */
  sourcesFor(program: WebGLProgram): { vertex: string; fragment: string }
  /** Uniform names uploaded via any uniform*/
  uploadedUniforms(): string[]
  callsTo(name: string): RecordedCall[]
}

/**
 * @param failTextures - make createTexture return null, to drive the
 *   "region cannot be uploaded" branches.
 */
export function createRecordingGl({
  failTextures = false,
}: { failTextures?: boolean } = {}): RecordingGl {
  const calls: RecordedCall[] = []
  const record = (name: string, ...args: unknown[]) => {
    calls.push({ name, args })
  }

  type ShaderStub = { type: number; source: string }
  type ProgramStub = { shaders: ShaderStub[] }

  const programSources = new Map<
    ProgramStub,
    { vertex: string; fragment: string }
  >()
  let textures = 0
  let buffers = 0

  const VERTEX_SHADER = 0x8b31
  const FRAGMENT_SHADER = 0x8b30

  const sourcesOf = (program: ProgramStub) => {
    const cached = programSources.get(program)
    if (cached) return cached
    const resolved = {
      vertex:
        program.shaders.find((s) => s.type === VERTEX_SHADER)?.source ?? '',
      fragment:
        program.shaders.find((s) => s.type === FRAGMENT_SHADER)?.source ?? '',
    }
    programSources.set(program, resolved)
    return resolved
  }

  const gl = {
    // Constants the renderer reads off the context.
    VERTEX_SHADER,
    FRAGMENT_SHADER,
    COMPILE_STATUS: 0x8b81,
    LINK_STATUS: 0x8b82,
    TEXTURE0: 0x84c0,
    TEXTURE1: 0x84c1,
    TEXTURE_2D: 0x0de1,
    TEXTURE_MIN_FILTER: 0x2801,
    TEXTURE_MAG_FILTER: 0x2800,
    TEXTURE_WRAP_S: 0x2802,
    TEXTURE_WRAP_T: 0x2803,
    NEAREST: 0x2600,
    LINEAR: 0x2601,
    CLAMP_TO_EDGE: 0x812f,
    ARRAY_BUFFER: 0x8892,
    ELEMENT_ARRAY_BUFFER: 0x8893,
    STATIC_DRAW: 0x88e4,
    TRIANGLES: 0x0004,
    TRIANGLE_STRIP: 0x0005,
    UNSIGNED_INT: 0x1405,
    FLOAT: 0x1406,
    BLEND: 0x0be2,
    ONE: 1,
    ONE_MINUS_SRC_ALPHA: 0x0303,
    RED: 0x1903,
    RG: 0x8227,
    RGB: 0x1907,
    RGBA: 0x1908,
    R32F: 0x822e,
    RG32F: 0x8230,
    RGB32F: 0x8815,
    RGBA32F: 0x8814,
    RGB16F: 0x881b,

    getExtension: vi.fn(() => null),
    drawBuffers: vi.fn(),

    createShader: vi.fn((type: number) => ({ type, source: '' })),
    shaderSource: vi.fn((shader: ShaderStub, source: string) => {
      shader.source = source
    }),
    compileShader: vi.fn(),
    getShaderParameter: vi.fn(() => true),
    getShaderInfoLog: vi.fn(() => ''),
    deleteShader: vi.fn(),

    createProgram: vi.fn((): ProgramStub => ({ shaders: [] })),
    attachShader: vi.fn((program: ProgramStub, shader: ShaderStub) => {
      program.shaders.push(shader)
    }),
    linkProgram: vi.fn(),
    getProgramParameter: vi.fn(() => true),
    getProgramInfoLog: vi.fn(() => ''),
    deleteProgram: vi.fn(),
    useProgram: vi.fn(),

    getUniformLocation: vi.fn(
      (program: ProgramStub, name: string): UniformLocationStub | null => {
        const { vertex, fragment } = sourcesOf(program)
        return isActiveUniform(vertex, name) || isActiveUniform(fragment, name)
          ? { name }
          : null
      }
    ),
    getAttribLocation: vi.fn((_program: ProgramStub, name: string) =>
      name === 'vertex' ? 0 : 1
    ),

    uniform1i: vi.fn((loc: UniformLocationStub, v: number) =>
      record('uniform1i', loc?.name, v)
    ),
    uniform1f: vi.fn((loc: UniformLocationStub, v: number) =>
      record('uniform1f', loc?.name, v)
    ),
    uniform2f: vi.fn((loc: UniformLocationStub, a: number, b: number) =>
      record('uniform2f', loc?.name, a, b)
    ),
    uniform4f: vi.fn((loc: UniformLocationStub, ...v: number[]) =>
      record('uniform4f', loc?.name, ...v)
    ),
    uniformMatrix4fv: vi.fn(
      (loc: UniformLocationStub, _transpose: boolean, value: Float32Array) =>
        record('uniformMatrix4fv', loc?.name, value)
    ),

    createTexture: vi.fn(() => (failTextures ? null : { tex: ++textures })),
    deleteTexture: vi.fn(),
    bindTexture: vi.fn(),
    activeTexture: vi.fn(),
    texParameteri: vi.fn(),
    texImage2D: vi.fn(),

    createBuffer: vi.fn(() => ({ buf: ++buffers })),
    deleteBuffer: vi.fn(),
    bindBuffer: vi.fn(),
    bufferData: vi.fn(),
    enableVertexAttribArray: vi.fn(),
    vertexAttribPointer: vi.fn(),

    drawArrays: vi.fn((...args: unknown[]) => record('drawArrays', ...args)),
    drawElements: vi.fn((...args: unknown[]) =>
      record('drawElements', ...args)
    ),

    enable: vi.fn(),
    blendFunc: vi.fn(),

    calls,
    callsTo: (name: string) => calls.filter((c) => c.name === name),
    uploadedUniforms: () =>
      calls
        .filter((c) => c.name.startsWith('uniform'))
        .map((c) => c.args[0] as string),
    sourcesFor: (program: unknown) => sourcesOf(program as ProgramStub),
  }

  return gl as unknown as RecordingGl
}
