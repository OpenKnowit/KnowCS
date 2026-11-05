import { describe, expect, it } from 'vitest'
import { planBroadcast, shapeLabel, sourceIndex } from './broadcast'
import { EXAM_QUESTIONS, SYNTAXES, apply, constructCode, examShape, flatOf, makeExam, makePairwise, makeProgram, valA, valB, withoutOnes } from './broadcastDemo'
import type { Op } from './broadcastDemo'
import { runPython } from './minipy'

const shapes: number[][] = [[], [1], [2], [3], [1, 1], [1, 3], [3, 1], [2, 3], [3, 2], [2, 1, 3], [1, 2, 3], [2, 3, 1], [1, 1, 3]]
const indices = (shape: number[]): number[][] => shape.reduce<number[][]>((acc, d) => acc.flatMap((p) => Array.from({ length: d }, (_, i) => [...p, i])), [[]])

describe('the generated broadcasting program', () => {
  for (const { id: syntax } of SYNTAXES) {
    it(`computes exactly what the grids draw, spelled with ${syntax}`, () => {
      for (const a of shapes) {
        for (const b of shapes) {
          const op: Op = '-'
          const p = makeProgram(a, b, op, syntax)
          const r = runPython(p.code.replace('print(C)', 'print(np.array(C).ravel().tolist())'))
          const plan = planBroadcast(a, b)
          const label = `${shapeLabel(a)} ${shapeLabel(b)}`
          if (!plan.ok) {
            expect(r.error?.type, label).toBe('ValueError')
            continue
          }
          expect(r.error, label).toBeNull()
          const cells = indices(plan.outShape!).map((i) => apply(op, valA(sourceIndex(i, a), a), valB(sourceIndex(i, b), b)))
          const [shapeLine, valueLine] = r.stdout.trim().split('\n')
          expect(shapeLine.replace(/\s/g, ''), label).toBe(shapeLabel(plan.outShape!).replace(/\s/g, ''))
          expect(JSON.parse(valueLine), label).toEqual(cells)
        }
      }
    })
  }

  it('writes the spellings the exams use', () => {
    expect(constructCode('a', [3, 1], 'none')).toBe('a[:, None]')
    expect(constructCode('a', [1, 3], 'none')).toBe('a[None, :]')
    expect(constructCode('X', [2, 1, 2], 'newaxis')).toBe('X[:, np.newaxis, :]')
    expect(constructCode('a', [3, 1], 'expand')).toBe('np.expand_dims(a, axis=1)')
    expect(constructCode('a', [1, 1, 3], 'expand')).toBe('np.expand_dims(np.expand_dims(a, axis=0), axis=1)')
    expect(constructCode('a', [1, 3], 'reshape')).toBe('a.reshape(1, 3)')
    expect(constructCode('a', [2, 3], 'none')).toBeNull()
    expect(withoutOnes([1, 1])).toEqual([1])
  })

  it('needs no construction when the shapes have no length-1 axis', () => {
    expect(makeProgram([3, 4], [4], '+', 'none').constructs).toEqual([])
    expect(makeProgram([3, 1], [1, 4], '+', 'none').constructs.map((c) => c.code)).toEqual(['a[:, None]', 'b[None, :]'])
  })

  it('traces the same eight pairwise steps in every spelling', () => {
    const results = SYNTAXES.map(({ id }) => runPython(makePairwise(id).code))
    for (const r of results) {
      expect(r.error).toBeNull()
      expect(r.calls.filter((c) => c.line >= 4 && c.line <= 10)).toHaveLength(8)
      expect(r.stdout).toBe(results[0].stdout)
    }
    expect(results[0].stdout).toContain('7.07106781')
  })

})

describe('the past-paper questions', () => {
  // real NumPy 2.2.6 on the papers' code, word for word
  const REAL: Record<string, string> = {
    e23s_i: '[[1. 1.]\n [2. 2.]\n [3. 3.]\n [4. 4.]]',
    e23s_ii: 'ValueError',
    e23s_iii: '[[ 1  2  3]\n [11 12 13]\n [21 22 23]\n [31 32 33]]',
    e24s_x: 'ValueError',
    e24s_xi:
      '[[[ 0.  1.  2.  3.]\n  [ 4.  5.  6.  7.]]\n\n [[ 8.  9. 10. 11.]\n  [12. 13. 14. 15.]]\n\n [[16. 17. 18. 19.]\n  [20. 21. 22. 23.]]]',
  }
  for (const q of EXAM_QUESTIONS) {
    const syntax = q.syntax ?? 'none'
    it(`${q.id} prints the paper's own line and gives NumPy's answer`, () => {
      const p = makeExam(q, syntax, q.op, null)
      expect(p.lines.at(-1)!.text).toBe(q.line)
      const r = runPython(p.code)
      expect(r.error ? r.error.type : r.stdout.trim()).toBe(REAL[q.id])
    })
    it(`${q.id}: the grids draw what its code computes, in every spelling`, () => {
      const a = examShape(q.l, 'l', null)
      const b = examShape(q.r, 'r', null)
      const plan = planBroadcast(a, b)
      for (const { id } of SYNTAXES) {
        const r = runPython(makeExam(q, id, q.op, null).code.replace(/print\(\s*(.*?)\s*\)$/, 'print(np.array($1).ravel().tolist())'))
        if (!plan.ok) {
          expect(r.error?.type).toBe('ValueError')
          continue
        }
        const cells = indices(plan.outShape!).map((i) => {
          const ia = sourceIndex(i, a)
          const ib = sourceIndex(i, b)
          return apply(q.op, q.l.values[flatOf(ia, a)], q.r.values[flatOf(ib, b)])
        })
        expect(r.error).toBeNull()
        expect(JSON.parse(r.stdout.trim())).toEqual(cells)
      }
    })
  }

  it('keeps the data when a fix makes an impossible question work', () => {
    const q = EXAM_QUESTIONS.find((x) => x.id === 'e23s_ii')!
    const p = makeExam(q, 'none', '+', { side: 'r', shape: [4, 1] })
    expect(p.lines.at(-1)!.text).toBe('print(A + B[:, None])')
    expect(runPython(p.code).stdout).toContain('[10 10 10]')
  })
})
