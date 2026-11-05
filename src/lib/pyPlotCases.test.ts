import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import { MATPLOTLIB } from './pyPlot'
import cases from './pyPlotCases.json'

// What real matplotlib 3.10.8 printed or raised for each snippet in scripts/gen-plot-cases.py.
// Compared to 11 significant digits: matplotlib sends some limits through display coordinates and back, which leaves
// noise in the last bits. Scalars print as plain numbers here, not np.float64(...), as everywhere in the sandbox.
const normal = (s: string) =>
  s
    .replace(/np\.(float64|int64|uint8)\(([^)]*)\)/g, '$2')
    .replace(/np\.True_/g, 'True')
    .replace(/np\.False_/g, 'False')
    .replace(/0x[0-9a-f]+/g, '0x')
    .replace(' (or it is not available in this sandbox)', '')
    // a warning in matplotlib, which goes to stderr
    .replace(/No artists with labels found to put in legend\.[^\n]*\n/g, '')
    .replace(/-?\d+\.\d+(e-?\d+)?/g, (m) => String(+Number(m).toPrecision(11)))

describe('the sandbox prints what matplotlib prints', () => {
  for (const [code, want] of cases as [string, string][]) {
    it(code.split('\n').slice(0, 3).join(' ; ').slice(0, 110), () => {
      const r = runPython(`import numpy as np\nimport matplotlib.pyplot as plt\n${code}`, { libs: [MATPLOTLIB], maxSize: 4000000 })
      expect(normal(r.stdout + (r.error ? `${r.error.type}: ${r.error.message}` : ''))).toBe(normal(want))
    })
  }
})
