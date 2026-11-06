import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import { CV2 } from './pyCv2'
import { KERAS } from './pyKeras'
import { PANDAS } from './pyPandas'
import { MATPLOTLIB } from './pyPlot'
import { SKLEARN } from './pySklearn'
import { STDLIB_MORE } from './pyStdlibMore'
import { TORCH } from './pyTorch'
import cases from './apiCases.json'
import coverage from '../data/apiCoverage.json'
import { norm, same } from './apiCompare'

// Small calls to each name, kept when the sandbox printed what the real library printed (scripts/api-coverage/gen.ts).
const libs = [TORCH, KERAS, PANDAS, MATPLOTLIB, SKLEARN, CV2, STDLIB_MORE]

describe('each frozen call to an API prints what the real library printed', () => {
  it(`all ${(cases as unknown[]).length} of them`, () => {
    const wrong: string[] = []
    for (const c of cases as { src: string; out: string }[]) {
      const r = runPython(c.src, { libs, maxSize: 2_000_000, timeBudget: 5000 })
      const got = r.error ? `${r.error.type}: ${r.error.message}` : norm(r.stdout)
      if (!same(got, c.out)) wrong.push(`${c.src}\n  got ${got.slice(0, 200)}\n want ${c.out.slice(0, 200)}`)
    }
    expect(wrong.slice(0, 5).join('\n\n')).toBe('')
  }, 300_000)
})

describe('the coverage data', () => {
  it('has one status per name, and an example for every name it says has one', () => {
    const files = import.meta.glob<{ entries: Record<string, Record<string, unknown>> }>('../data/apiExamples/*.json', { eager: true, import: 'default' })
    const examples = Object.fromEntries(Object.entries(files).map(([f, d]) => [f.split('/').pop()!.replace('.json', ''), d]))
    for (const g of coverage.groups) {
      expect(g.code.length).toBe(g.names.length)
      g.names.forEach((name, i) => {
        const has = examples[g.lib]?.entries[g.m]?.[name]
        if ('ced'.includes(g.code[i])) expect(has, `${g.m}.${name}`).toBeTruthy()
        if ('yxn'.includes(g.code[i])) expect(has, `${g.m}.${name}`).toBeFalsy()
      })
    }
  })
})
