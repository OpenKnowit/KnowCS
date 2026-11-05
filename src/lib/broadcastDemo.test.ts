import { describe, expect, it } from 'vitest'
import { planBroadcast, shapeLabel, sourceIndex } from './broadcast'
import { EXAM_EXAMPLES, SYNTAXES, apply, constructCode, makePairwise, makeProgram, valA, valB, withoutOnes } from './broadcastDemo'
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

  it('has every past-paper example the right way round', () => {
    expect(EXAM_EXAMPLES.map((e) => planBroadcast(e.a, e.b).outShape?.join() ?? 'error')).toEqual(['2,4,2', '4,2', 'error', '4,3', 'error', '3,2,4'])
  })
})
