import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import { PANDAS } from './pyPandas'
import cases from './pyPandasCases.json'

// What real pandas 2.3.3 printed for each program in scripts/pandas-cases.txt (scripts/gen-pandas-cases.py freezes it).
// An error must match in type and message; the sandbox may only add a hint in parentheses after the message.
describe('the pandas sandbox prints what pandas prints', () => {
  for (const c of cases) {
    it(c.name, () => {
      const r = runPython(c.code, { libs: [PANDAS], maxSize: 65536, timeBudget: 20000 })
      expect(r.stdout).toBe(c.out)
      expect(r.error?.type ?? null).toBe(c.error?.type ?? null)
      if (r.error && c.error && r.error.message !== c.error.message) expect(r.error.message.startsWith(`${c.error.message} (`)).toBe(true)
    })
  }
})
