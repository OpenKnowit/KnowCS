import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import { CV2 } from './pyCv2'
import { KERAS } from './pyKeras'
import { PANDAS } from './pyPandas'
import { MATPLOTLIB } from './pyPlot'
import { SKLEARN } from './pySklearn'
import { STDLIB_MORE } from './pyStdlibMore'
import { TORCH } from './pyTorch'
import { norm, same } from './apiCompare'
import cases from './realUseCases.json'

// Programs written the way people write them, and every example in the playground sidebar and gallery (titled <library>/<id>),
// with the output the real libraries gave (NumPy 2.2, pandas 2.3, matplotlib 3.10, Keras 3.12, TF 2.21, OpenCV 4.13,
// sklearn 1.7, torch 2.14). `value` is what the last line evaluated to, as the playground shows it below the output.
// These caught what the one-call-per-name probes could not. Examples left out print something only the real library's
// own random numbers or layout decide (Keras and torch weights, a 3-D title's position).
const libs = [TORCH, KERAS, PANDAS, MATPLOTLIB, SKLEARN, CV2, STDLIB_MORE]

describe('programs in the shape people write them print what the real libraries printed', () => {
  for (const c of cases as { title: string; src: string; out: string; value?: string }[]) {
    it(c.title, () => {
      const r = runPython(c.src, { libs, maxSize: 2_000_000, timeBudget: 20_000 })
      const got = norm(r.error ? `${r.error.type}: ${r.error.message}` : c.value === undefined ? r.stdout : `${r.stdout}\n${r.out}`)
      const want = norm(c.value === undefined ? c.out : `${c.out}\n${c.value}`)
      if (!same(got, want)) expect(got).toBe(want)
    }, 30_000)
  }
})
