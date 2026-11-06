import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import { KERAS } from './pyKeras'
import { PANDAS } from './pyPandas'
import { MATPLOTLIB } from './pyPlot'
import { STDLIB_MORE } from './pyStdlibMore'
import data from './pyKerasCases.json'

// What real TensorFlow 2.21 and Keras 3.12 printed for scripts/keras-cases.txt, frozen by scripts/gen-keras-cases.py.
const NUM = /[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?/g
const shape = (s: string) => s.replace(NUM, '#').replace(/\s+/g, ' ').replace(/([[(]) /g, '$1').replace(/ ([\]),])/g, '$1')
/** the sandbox computes in double and rounds, TensorFlow in float32: last digits differ, and so may the padding around them */
const close = (a: string, b: string) => {
  if (a === b) return true
  if (shape(a) !== shape(b)) return false
  const x = a.match(NUM) ?? []
  const y = b.match(NUM) ?? []
  return x.every((v, i) => Math.abs(Number(v) - Number(y[i])) <= Math.max(5e-4 * Math.abs(Number(y[i])), 2e-4) && Math.abs(v.length - y[i].length) <= 5)
}
/** the sandbox's own remarks, which real Keras does not print to stdout */
const own = (s: string) => s.replace(/^\(sandbox\)[^\n]*\n/gm, '').replace(/^UserWarning:[^\n]*\n/gm, '').replace(/ \((?:or it is )?not available in this sandbox\)/g, '').replace(/[ \t]+$/gm, '')
const errText = (e: { type: string; message: string } | null) => (e ? `${e.type}: ${e.message}`.replace(/ \((?:the|The) sandbox only[^)]*\)/, '').replace(/ \((?:or it is )?not available in this sandbox\)/, '').trimEnd() : null)

describe(`the sandbox prints what TensorFlow 2.21 and Keras 3.12 print`, () => {
  it.each(data as { title: string; src: string; out: string; error: { type: string; message: string } | null; random: boolean }[])('$title', (c) => {
    const r = runPython(c.src, { libs: [KERAS, PANDAS, MATPLOTLIB, STDLIB_MORE], maxSize: 2_000_000, timeBudget: 30_000 })
    const got = own(r.stdout)
    const want = errText(c.error)
    if (c.random) {
      expect(shape(got)).toBe(shape(c.out))
      expect(want === null ? errText(r.error) : shape(errText(r.error) ?? '')).toBe(want === null ? null : shape(want))
      return
    }
    expect(errText(r.error)).toBe(want)
    if (!close(got, c.out)) expect(got).toBe(c.out)
  })
})
