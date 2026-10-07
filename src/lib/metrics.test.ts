import { describe, expect, it } from 'vitest'
import { classificationMetrics, confusionFromLabels } from './metrics'

describe('confusionFromLabels', () => {
  it('builds the 2024 Spring midterm matrix (rows = actual)', () => {
    const { classes, m } = confusionFromLabels('2 1 1 2 0 1 0 0 1 1', '2,1,2,1,0,2,1,0,0,0')
    expect(classes).toEqual(['0', '1', '2'])
    expect(m).toEqual([[2, 1, 0], [2, 1, 2], [0, 1, 1]])
  })

  it('rejects arrays of different length', () => {
    expect(() => confusionFromLabels('0 1', '0')).toThrow(/2 labels/)
  })
})

describe('classificationMetrics', () => {
  it('matches the 2024 Spring marking scheme', () => {
    const r = classificationMetrics([[2, 1, 0], [2, 1, 2], [0, 1, 1]])
    expect(r.perClass.map((c) => [c.TP, c.TN, c.FP, c.FN])).toEqual([[2, 5, 2, 1], [1, 3, 2, 4], [1, 6, 2, 1]])
    expect(r.perClass.map((c) => c.f1)).toEqual([4 / 7, 1 / 4, 2 / 5])
    expect(r.accuracy).toBe(0.4)
    expect(r.macroF1).toBeCloseTo(0.407, 3)
  })

  it('shows accuracy vs F1 on the Final 2022 imbalanced matrix', () => {
    const r = classificationMetrics([[1, 20], [5, 100]])
    expect(r.accuracy).toBeCloseTo(0.8016, 4)
    expect(r.perClass[0].f1).toBeCloseTo(0.0741, 4)
  })

  it('gives MCC ≈ 0.14 for the lecture 3 example (TP 90, FP 4, TN 1, FN 5)', () => {
    expect(classificationMetrics([[90, 5], [4, 1]]).mcc).toBeCloseTo(0.135, 3)
  })

  it('does not divide by zero for a class that is never predicted', () => {
    const r = classificationMetrics([[0, 5], [0, 95]])
    expect(Number.isNaN(r.perClass[0].precision)).toBe(true)
    expect(r.perClass[0].f1).toBe(0)
    expect(r.accuracy).toBe(0.95)
  })
})
