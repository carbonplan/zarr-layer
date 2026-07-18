import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    // Pure-logic unit tests run in Node; no DOM/WebGL is required because
    // shader behavior is verified by math parity rather than a live GL context
    // (see src/shaders.test.ts).
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
