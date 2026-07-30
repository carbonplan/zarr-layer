/**
 * Render vitest's JSON report as markdown for the Actions run summary.
 *
 * Usage: node .github/scripts/test-summary.mjs <report.json>
 *
 * Always exits 0 — the vitest step is what fails the job.
 */

import { readFileSync } from 'node:fs'

const reportPath = process.argv[2]

let report
try {
  report = JSON.parse(readFileSync(reportPath, 'utf8'))
} catch (err) {
  // Vitest died before writing a report (crash, OOM, bad config). Saying so
  // beats reporting zero tests, which reads like a clean run.
  process.stdout.write(
    `### Tests\n\nNo test report was produced — vitest exited before writing \`${reportPath}\`.\n\n> ${err.message}\n`
  )
  process.exit(0)
}

const {
  numTotalTests = 0,
  numPassedTests = 0,
  numFailedTests = 0,
  numPendingTests = 0,
  numTodoTests = 0,
  testResults = [],
  success,
} = report

const elapsed = (file) =>
  Math.max(0, (file.endTime ?? 0) - (file.startTime ?? 0))
const seconds = (ms) => `${(ms / 1000).toFixed(1)}s`
const relative = (name) => name.replace(`${process.cwd()}/`, '')

const skipped = numPendingTests + numTodoTests
const duration = testResults.reduce((total, file) => total + elapsed(file), 0)

const lines = [
  `### ${success ? '✅' : '❌'} Tests`,
  '',
  `**${numPassedTests}/${numTotalTests} passed**` +
    (numFailedTests ? ` · ${numFailedTests} failed` : '') +
    (skipped ? ` · ${skipped} skipped` : '') +
    ` · ${seconds(duration)} · ${testResults.length} files`,
  '',
]

const failedFiles = testResults.filter((file) => file.status === 'failed')

if (failedFiles.length) {
  lines.push('| Test | File |', '| --- | --- |')
  for (const file of failedFiles) {
    const failures = (file.assertionResults ?? []).filter(
      (assertion) => assertion.status === 'failed'
    )
    // An import-time throw fails the file without failing any single test.
    if (!failures.length) {
      lines.push(`| _suite failed to run_ | \`${relative(file.name)}\` |`)
      continue
    }
    for (const failure of failures) {
      const title = [...(failure.ancestorTitles ?? []), failure.title]
        .filter(Boolean)
        .join(' › ')
      lines.push(
        `| ${title.replace(/\|/g, '\\|')} | \`${relative(file.name)}\` |`
      )
    }
  }
  lines.push(
    '',
    'Failure output is annotated inline on the diff and printed in full in the job log.'
  )
} else {
  const slowest = [...testResults]
    .sort((a, b) => elapsed(b) - elapsed(a))
    .slice(0, 5)
  lines.push(
    '<details><summary>Slowest files</summary>',
    '',
    '| File | Duration |',
    '| --- | --- |',
    ...slowest.map(
      (file) => `| \`${relative(file.name)}\` | ${seconds(elapsed(file))} |`
    ),
    '',
    '</details>'
  )
}

lines.push('')
process.stdout.write(lines.join('\n'))
