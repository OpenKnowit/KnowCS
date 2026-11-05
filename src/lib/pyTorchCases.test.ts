import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import { TORCH } from './pyTorch'
import data from './pyTorchCases.json'

// What real PyTorch printed or raised for each program in scripts/torch-cases.txt (python scripts/gen-torch-cases.py).
// Numbers may differ by float32 last-digit noise; all other text must be exact.
const NUMBER = /[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?/g

const close = (got: string, want: string): boolean => {
  if (got === want) return true
  const exact = (t: string) => t.replace(NUMBER, '#')
  const loose = (t: string) => exact(t).replace(/ +#/g, '#')
  if (loose(got) !== loose(want)) return false
  const [a, b] = [got.match(NUMBER) ?? [], want.match(NUMBER) ?? []]
  const near = a.every((text, i) => {
    const [x, y] = [Number(text), Number(b[i])]
    return Math.abs(x - y) <= 2e-6 || Math.abs(x - y) <= 4e-6 * Math.abs(y)
  })
  if (!near) return false
  // a column's padding may move only when noise around zero changed a sign
  return exact(got) === exact(want) || a.some((text, i) => text.startsWith('-') !== b[i].startsWith('-'))
}

/*
 * The deliberate differences, each with its reason.
 * - NO_OVERLOAD: PyTorch lists every overload after "but expected"; the sandbox names what it was given and stops there.
 * - SANDBOX_SUFFIX: the core adds a note to AttributeError, that the name may simply not be here yet.
 * - NUMPY_SCALAR: NumPy scalars print as plain numbers everywhere in the sandbox, not np.float32(...).
 */
const NO_OVERLOAD = /(received an invalid combination of arguments - got \(.*?\), but expected)[^]*$/
const SANDBOX_SUFFIX = / \((or it is )?not available in this sandbox\)$/gm
const NUMPY_SCALAR = /np\.(?:float32|float64|int64)\(([^)]*)\)/g

const normal = (s: string) => s.replace(NO_OVERLOAD, '$1').replace(SANDBOX_SUFFIX, '').replace(NUMPY_SCALAR, '$1')

describe(`the sandbox prints what torch ${data.torch} prints`, () => {
  it.each(data.cases.map((c) => [c.src.replace(/^(import .*\n|def fill[^]*?view\(p\.shape\)\)\n)+/, '').slice(0, 120), c] as const))('%s', (_name, c) => {
    const r = runPython(c.src, { libs: [TORCH], maxSize: 2_000_000, timeBudget: 20_000 })
    const got = normal(r.stdout + (r.error ? `${r.error.type}: ${r.error.message}` : ''))
    const want = normal(c.out + (c.error ?? ''))
    if (!close(got, want)) expect(got).toBe(want)
  })
})
