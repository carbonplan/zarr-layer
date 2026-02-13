#!/usr/bin/env node
/**
 * Build script that generates an inline worker module.
 *
 * This script:
 * 1. Bundles the worker with proj4 into a single file
 * 2. Generates a TypeScript module that exports the worker code as a string
 *
 * This approach works better for library distribution because the worker
 * code is embedded in the library itself, avoiding URL resolution issues.
 *
 * Bundle size note: The generated worker bundle is ~128KB (minified) because
 * it includes proj4. This effectively duplicates the proj4 code (once for
 * main thread, once for worker) since workers cannot share modules with the
 * main thread. This trade-off is necessary for the non-blocking benefits.
 */

import { build } from 'esbuild'
import { writeFileSync, mkdirSync } from 'fs'
import { dirname } from 'path'

async function buildWorkerInline() {
  console.log('Building inline worker...')

  // Bundle the worker with esbuild (self-contained with proj4)
  const result = await build({
    entryPoints: ['src/visible-regions-worker.ts'],
    bundle: true,
    format: 'iife', // Use IIFE for worker (self-contained)
    minify: true,
    write: false, // Don't write to disk, we want the code
    target: 'es2020',
    define: {
      'process.env.NODE_ENV': '"production"',
    },
  })

  const workerCode = result.outputFiles[0].text

  // Generate TypeScript module that exports the worker code
  const moduleContent = `/**
 * Auto-generated inline worker code.
 * Do not edit directly - run 'npm run build:worker' to regenerate.
 */

// Worker code bundled with proj4
const WORKER_CODE = ${JSON.stringify(workerCode)};

/**
 * Creates a Blob URL for the worker.
 * This approach works for library distribution because the worker
 * code is embedded in the library itself.
 */
export function createWorkerBlobURL(): string {
  const blob = new Blob([WORKER_CODE], { type: 'application/javascript' });
  return URL.createObjectURL(blob);
}

/**
 * Creates a Worker instance using the embedded worker code.
 */
export function createVisibleRegionsWorker(): Worker {
  const blobUrl = createWorkerBlobURL();
  const worker = new Worker(blobUrl);
  // Revoke the blob URL immediately - the worker has already loaded the code
  // and no longer needs the URL. This prevents memory leaks.
  URL.revokeObjectURL(blobUrl);
  return worker;
}
`

  // Ensure the directory exists
  const outPath = 'src/visible-regions-worker-inline.ts'
  mkdirSync(dirname(outPath), { recursive: true })

  // Write the module
  writeFileSync(outPath, moduleContent)
  console.log(
    `Generated ${outPath} (${(workerCode.length / 1024).toFixed(
      1
    )}KB of worker code)`
  )
}

buildWorkerInline().catch((err) => {
  console.error('Failed to build inline worker:', err)
  process.exit(1)
})
