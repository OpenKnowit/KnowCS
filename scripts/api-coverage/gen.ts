// What the sandbox has of each real library, and a program for each name. Regenerate with:
//   python3 scripts/api-real-names.py && ~/repo/minimind/.venv/bin/python scripts/api-real-names.py torch
//   python3 scripts/api-coverage/probe_real.py a.json numpy matplotlib pandas keras tensorflow python sklearn cv2
//   ~/repo/minimind/.venv/bin/python scripts/api-coverage/probe_real.py b.json pytorch
//   REC_OUT=programs.jsonl npx vitest run --config scripts/api-coverage/vitest.record.config.ts
//   npx esbuild scripts/api-coverage/gen.ts --bundle --platform=node --format=esm --outfile=gen.mjs && node gen.mjs programs.jsonl a.json b.json
import { createHash } from 'node:crypto'
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

const libs = [TORCH, KERAS, PANDAS, MATPLOTLIB, SKLEARN, CV2, STDLIB_MORE]
type Group = { lib: string; expr: string; setup: string; names: string[] }
const real = JSON.parse(fs.readFileSync('scripts/api-real-names.json', 'utf8')) as { versions: Record<string, string>; groups: Record<string, Group> }
const hash = (s: string) => createHash('sha1').update(s).digest('hex').slice(0, 12)

// ---- programs whose whole output was compared with the real library
const checked = new Set<string>(JSON.parse(fs.readFileSync('scripts/api-coverage/gallery-verified.json', 'utf8')) as string[])
const add = (s: unknown) => { if (typeof s === 'string') checked.add(hash(s)) }
for (const f of ['pyKerasCases', 'pyTorchCases', 'pyPandasCases', 'minipyCases', 'pyPlotCases']) {
  const d = JSON.parse(fs.readFileSync(`src/lib/${f}.json`, 'utf8'))
  const rows = Array.isArray(d) ? d : (Object.values(d).find(Array.isArray) as unknown[])
  for (const r of rows as (Record<string, unknown> | unknown[])[]) {
    if (Array.isArray(r)) { add(r[0]); add(`import matplotlib.pyplot as plt\nimport numpy as np\n${r[0]}`) }
    else { add(r.src); add(r.code) }
  }
}

// ---- the corpus: each program once, by library, short ones first
const LIB_OF: [RegExp, string][] = [[/\bimport torch|from torch/, 'pytorch'], [/\bimport tensorflow|from tensorflow/, 'tensorflow'], [/\bimport keras|from keras/, 'keras'], [/\bimport cv2/, 'cv2'], [/\bimport sklearn|from sklearn/, 'sklearn'], [/\bimport pandas/, 'pandas'], [/\bimport matplotlib|from matplotlib/, 'matplotlib'], [/\bimport numpy/, 'numpy']]
const importsOf = (s: string) => LIB_OF.filter(([r]) => r.test(s)).map(([, l]) => l)
const programs = [...new Set(fs.readFileSync(process.argv[2], 'utf8').split('\n').filter(Boolean).map((l) => (JSON.parse(l) as { src: string }).src))]
  .filter((s) => !/\b__\w+__\b/.test(s.slice(0, 40)))
  .sort((a, b) => Number(checked.has(hash(b))) - Number(checked.has(hash(a))) || a.length - b.length)
const words = programs.map((s) => ({ s, dotted: new Set([...s.matchAll(/\.\s*([A-Za-z_]\w*)/g)].map((m) => m[1])), bare: new Set(s.match(/\b[A-Za-z_]\w*\b/g) ?? []), libs: importsOf(s) }))

const example = (group: string, g: Group, name: string): typeof words[number] | null => {
  const instance = !/^[\w.]+$/.test(g.expr) || /^(Tensor|ndarray|Series|DataFrame|GroupBy|Series\.str|Axes|Figure|nn\.Module|keras\.Model|tf\.Tensor|tf\.Variable|tf\.data\.Dataset|str|list|dict|set|tuple|int|float)$/.test(group)
  for (const p of words) {
    if (g.lib === 'python') {
      if (p.libs.length && group !== 'builtins') continue
      if (group !== 'builtins' && !instance && !new RegExp(`\\b(import|from)\\s+[\\w., ]*\\b${group.replace('.', '\\.')}\\b`).test(p.s)) continue
      if (group === 'builtins' ? p.bare.has(name) : instance ? p.dotted.has(name) : p.dotted.has(name) || p.bare.has(name)) return p
    } else {
      // keras programs often also import tensorflow, and tf programs keras
      const ok = p.libs.includes(g.lib) || (g.lib === 'numpy' && !p.libs.length)
      if (ok && (p.dotted.has(name) || (!instance && p.bare.has(name)))) return p
    }
  }
  return null
}

// ---- calls that worked on the real library: run each here, keep those that print the same
type Probe = { module: string; name: string; src: string; out: string }

// What a difference means when it is not a bug. These names stay out of "differs", and they do not count as "same" either.
const TYPE_NAME = /^[A-Za-z_][\w.]*$/
const NOT_A_TYPE = new Set(['None', 'True', 'False', 'nan', 'inf'])
const typeNameOnly = (want: string, got: string) => {
  const w = want.trim()
  const g = got.trim()
  // an abstract class has no shape of its own, so the real library printed the descriptor
  if (/^<(attribute 'shape' of|property object)/.test(w)) return TYPE_NAME.test(g)
  return TYPE_NAME.test(w) && TYPE_NAME.test(g) && !NOT_A_TYPE.has(w) && !NOT_A_TYPE.has(g)
}
const NOT_A_BUG: Record<string, string> = Object.fromEntries(([
  ['the probe passes an argument no real program would, and the libraries only happen to accept it', ['DataFrame:astype', 'Series:astype', 'DataFrame:from_dict', 'DataFrame:from_records', 'DataFrame:iloc', 'DataFrame:loc', 'GroupBy:agg', 'GroupBy:aggregate', 'GroupBy:apply', 'cv2:rotate', 'functools:total_ordering', 'functools:wraps', 'keras.models:clone_model', 'nn.Module:parameters', 'nn.Module:zero_grad', 'np:polysub', 'np:fromstring', 'pd:json_normalize', 'plt:get', 'plt:getp', 'plt:setp', 'tf:boolean_mask', 'torch.autograd:Function', 'typing:Generic', 'typing:Protocol', 'utils.data:Subset', 'utils.data:SubsetRandomSampler']],
  ['uninitialised memory: the real output is whatever was there before', ['np:empty', 'np:empty_like', 'np:ndarray']],
  ['documentation text, which the sandbox does not carry', ['inspect:getdoc', 'np:info']],
  ['depends on the machine: needs an accelerator, and this one has MPS', ['Tensor:pin_memory']],
] as [string, string[]][]).flatMap(([why, keys]) => keys.map((k) => [k, why])))
const probes = process.argv.slice(3).flatMap((f) => JSON.parse(fs.readFileSync(f, 'utf8')) as Probe[])
const passed = new Map<string, Probe>()
const differs = new Map<string, { src: string; want: string; got: string }>()
for (const p of probes) {
  const key = `${p.module}:${p.name}`
  if (passed.has(key)) continue
  let got: string
  try {
    const r = runPython(p.src, { libs, maxSize: 2_000_000, timeBudget: 5000 })
    got = r.error ? `${r.error.type}: ${r.error.message}` : norm(r.stdout)
  } catch (e) {
    got = `CRASH ${String(e)}`
  }
  if (same(got, norm(p.out))) {
    passed.set(key, { ...p, out: norm(p.out) })
    differs.delete(key)
  } else if (!differs.has(key)) differs.set(key, { src: p.src, want: norm(p.out).slice(0, 300), got: got.slice(0, 300) })
}
const accepted: Record<string, { want: string; got: string; why: string }> = {}
for (const [key, d] of differs) {
  const why = NOT_A_BUG[key] ?? (typeNameOnly(d.want, d.got) ? 'only the name of the object\'s type differs' : null)
  if (!why) continue
  if (!passed.has(key)) accepted[key] = { want: d.want.slice(0, 120), got: d.got.slice(0, 120), why }
  differs.delete(key)
}
console.log('probes', probes.length, 'names passed', passed.size, 'names differing', differs.size)

const out: { versions: Record<string, string>; groups: { m: string; lib: string; names: string[]; code: string }[] } = { versions: real.versions, groups: [] }
const perLib: Record<string, { programs: string[]; at: Map<string, number>; entries: Record<string, Record<string, [number, number]>> }> = {}
const frozen: { src: string; out: string }[] = []
const backlog: Record<string, { name: string; src: string; want: string; got: string }[]> = {}
for (const [module, g] of Object.entries(real.groups)) {
  const prefix = g.expr === '__builtins__' ? '' : `${g.setup}__o = ${g.expr}\n`
  const test = g.expr === '__builtins__'
    ? `for n in ${JSON.stringify(g.names)}:\n    try:\n        eval(n)\n        print(n)\n    except Exception:\n        pass\n`
    : `${prefix}for n in ${JSON.stringify(g.names)}:\n    try:\n        getattr(__o, n)\n        print(n)\n    except AttributeError:\n        pass\n    except NotImplementedError:\n        pass\n    except Exception:\n        print(n)\n`
  const r = runPython(test, { libs, maxSize: 2_000_000, timeBudget: 120_000 })
  if (r.error) console.log('!!', module, r.error.type, r.error.message.slice(0, 120))
  const have = new Set(r.stdout.split('\n').filter(Boolean))
  const lib = (perLib[g.lib] ??= { programs: [], at: new Map(), entries: {} })
  const codes: string[] = []
  const mine: Record<string, [number, number]> = {}
  const place = (src: string, v: boolean, name: string) => {
    let at = lib.at.get(src)
    if (at === undefined) {
      at = lib.programs.length
      lib.programs.push(src)
      lib.at.set(src, at)
    }
    mine[name] = [at, v ? 1 : 0]
  }
  for (const name of g.names) {
    const key = `${module}:${name}`
    const auto = passed.get(key)
    const bad = differs.get(key)
    if (bad && !auto && have.has(name)) (backlog[module] ??= []).push({ name, ...bad })
    if (!have.has(name)) {
      codes.push('n')
      continue
    }
    const p = example(module, g, name)
    if (p && checked.has(hash(p.s))) {
      place(p.s, true, name)
      codes.push('c')
    } else if (auto) {
      place(auto.src, true, name)
      frozen.push({ src: auto.src, out: auto.out })
      codes.push('c')
    } else if (bad) {
      if (p) place(p.s, false, name)
      codes.push(p ? 'd' : 'x')
    } else if (p) {
      place(p.s, false, name)
      codes.push('e')
    } else codes.push('y')
  }
  lib.entries[module] = mine
  out.groups.push({ m: module, lib: g.lib, names: g.names, code: codes.join('') })
  const count = (c: string) => codes.filter((x) => x === c).length
  console.log(`${g.lib.padEnd(11)} ${module.padEnd(22)} ${String(have.size).padStart(4)}/${String(g.names.length).padEnd(5)} checked ${String(count('c')).padStart(4)}  example ${String(count('e')).padStart(4)}  differs ${String(count('d') + count('x')).padStart(3)}  bare ${String(count('y')).padStart(4)}`)
}
fs.mkdirSync('src/data/apiExamples', { recursive: true })
fs.writeFileSync('src/data/apiCoverage.json', `${JSON.stringify(out)}\n`)
for (const [lib, d] of Object.entries(perLib)) fs.writeFileSync(`src/data/apiExamples/${lib}.json`, `${JSON.stringify({ programs: d.programs, entries: d.entries })}\n`)
fs.writeFileSync('src/lib/apiCases.json', `${JSON.stringify([...new Map(frozen.map((c) => [c.src, c])).values()])}\n`)
fs.writeFileSync('scripts/api-coverage/backlog.json', `${JSON.stringify(backlog, null, 1)}\n`)
fs.writeFileSync('scripts/api-coverage/accepted.json', `${JSON.stringify(accepted, null, 1)}\n`)
const all = out.groups.flatMap((g) => [...g.code])
console.log('TOTAL', all.length, 'have', all.filter((c) => c !== 'n').length, 'checked', all.filter((c) => c === 'c').length, 'example', all.filter((c) => c === 'e').length, 'differs', all.filter((c) => c === 'd' || c === 'x').length, 'bare', all.filter((c) => c === 'y').length)
