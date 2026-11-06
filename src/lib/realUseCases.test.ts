import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import { CV2 } from './pyCv2'
import { KERAS } from './pyKeras'
import { PANDAS } from './pyPandas'
import { MATPLOTLIB } from './pyPlot'
import { SKLEARN } from './pySklearn'
import { STDLIB_MORE } from './pyStdlibMore'
import { TORCH } from './pyTorch'
import cases from './realUseCases.json'

// Programs written the way people write them, with the output the real libraries gave (NumPy 2.2, pandas 2.3, matplotlib 3.10,
// Keras 3.12, OpenCV 4.13, torch 2.14). These caught what the one-call-per-name probes could not.
const libs = [TORCH, KERAS, PANDAS, MATPLOTLIB, SKLEARN, CV2, STDLIB_MORE]

describe('programs in the shape people write them print what the real libraries printed', () => {
  for (const c of cases as { title: string; src: string; out: string }[]) {
    it(c.title, () => {
      const r = runPython(c.src, { libs, maxSize: 2_000_000, timeBudget: 20_000 })
      expect(r.error ? `${r.error.type}: ${r.error.message}` : r.stdout).toBe(c.out)
    }, 30_000)
  }
})
