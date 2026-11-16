// Checks the curated programs (curated/<lib>.txt) after curated_real.py ran them on the real library: each must print
// the same in the sandbox, and draw something there. Those that pass go to curated/<lib>.json for gen.ts.
//   python3 scripts/api-coverage/curated_real.py matplotlib real.json
//   npx esbuild scripts/api-coverage/curated.ts --bundle --platform=node --format=esm --outfile=curated.mjs && node curated.mjs matplotlib real.json
import fs from 'node:fs'
import { runPython } from '../../src/lib/minipy'
import { CV2 } from '../../src/lib/pyCv2'
import { KERAS } from '../../src/lib/pyKeras'
import { PANDAS } from '../../src/lib/pyPandas'
import { MATPLOTLIB } from '../../src/lib/pyPlot'
import { SKLEARN } from '../../src/lib/pySklearn'
import { STDLIB_MORE } from '../../src/lib/pyStdlibMore'
import { TORCH } from '../../src/lib/pyTorch'
import { norm, same } from '../../src/lib/apiCompare'
import type { FigureSpec } from '../../src/lib/pyEvents'

const [lib, realFile] = process.argv.slice(2)
const libs = [TORCH, KERAS, PANDAS, MATPLOTLIB, SKLEARN, CV2, STDLIB_MORE]
const sections = JSON.parse(fs.readFileSync(realFile, 'utf8')) as { keys: string[]; src: string; real: string; realError: string | null }[]
/** what a figure shows: lines, patches, images, texts, legends, titles */
const drawn = (f: FigureSpec) => f.texts.length + f.legends.length + f.axes.reduce((s, a) => s + (a.visible ? a.artists.length + a.legends.length + (a.title ? 1 : 0) + a.titles.length : 0), 0)
const kept: { keys: string[]; src: string; out: string }[] = []
let bad = 0
for (const s of sections) {
  const r = runPython(s.src, { libs, maxSize: 2_000_000, timeBudget: 20_000 })
  const got = r.error ? `${r.error.type}: ${r.error.message}` : norm(r.stdout)
  const figures = r.displays.filter((d) => d.type === 'figure') as { fig: FigureSpec }[]
  const marks = figures.reduce((n, d) => n + drawn(d.fig), 0)
  // what the playground's right side shows: a figure for matplotlib, the traced steps, tables and diagrams for the rest
  const shown = lib === 'python' ? 1 : lib === 'matplotlib' ? marks : marks + r.calls.length + r.events.filter((e) => e.type !== 'note').length + r.displays.length + (r.outDisplay ? 1 : 0)
  const missing = s.keys.filter((k) => !new RegExp(`\\b${k.slice(k.lastIndexOf('.') + 1)}\\b`).test(s.src))
  const why = s.realError ? `real: ${s.realError}` : r.error ? `sandbox: ${got}` : !same(got, norm(s.real)) ? `differs\n  real:    ${norm(s.real).slice(0, 300)}\n  sandbox: ${got.slice(0, 300)}` : !shown ? 'shows nothing' : missing.length ? `does not use ${missing.join(' ')}` : null
  if (why) {
    bad++
    console.log(`✗ ${s.keys.join(' ')}\n  ${why}`)
  } else kept.push({ keys: s.keys, src: s.src, out: norm(s.real) })
}
fs.writeFileSync(`scripts/api-coverage/curated/${lib}.json`, `${JSON.stringify(kept, null, 1)}\n`)
console.log(`${kept.length} kept, ${bad} not`)
