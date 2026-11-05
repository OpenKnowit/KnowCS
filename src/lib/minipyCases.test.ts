import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import cases from './minipyCases.json'

// What real CPython printed for each program in scripts/python-cases.txt (scripts/gen-python-cases.py freezes it).
describe('the sandbox prints what CPython prints', () => {
  for (const c of cases) {
    it(c.name, () => {
      const r = runPython(c.code)
      expect(r.stdout).toBe(c.out)
      expect(r.error?.type ?? null).toBe(c.error)
    })
  }
})
