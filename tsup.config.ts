import { defineConfig } from 'tsup'

export default defineConfig({
  // Main library build
  entry: ['src/index.ts'],
  format: ['cjs', 'esm'],
  dts: true,
  sourcemap: true,
  splitting: false,
  clean: true,
  external: ['maplibre-gl', 'mapbox-gl'],
  noExternal: ['zarrita'],
})
