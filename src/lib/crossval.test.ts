import { describe, expect, it } from 'vitest'
import { crossValidate, foldSizes, sortedLabels } from './crossval'

describe('crossValidate (sorted, no shuffle)', () => {
  it('reproduces 2024 Spring Q7: 4 classes × 20 → 0%, 50%, 100%', () => {
    const labels = sortedLabels(4, 20)
    expect(crossValidate(labels, 2, 'sorted').mean).toBe(0)
    // folds of 27/27/26 give 23% · 100% · 26% (the paper rounds the folds to 25%)
    expect(crossValidate(labels, 3, 'sorted').mean).toBeCloseTo(0.5, 1)
    expect(crossValidate(labels, 5, 'sorted').mean).toBe(1)
  })

  it('reproduces 2023 Fall Q7: 5 classes → 20%, 0%, 100%', () => {
    const labels = sortedLabels(5, 6)
    expect(crossValidate(labels, 2, 'sorted').mean).toBeCloseTo(0.2, 10)
    expect(crossValidate(labels, 5, 'sorted').mean).toBe(0)
    expect(crossValidate(labels, 6, 'sorted').mean).toBe(1)
  })
})

describe('fixes', () => {
  it('stratified folds keep every class in training', () => {
    const labels = sortedLabels(4, 20)
    for (const d of [2, 3, 5]) expect(crossValidate(labels, d, 'stratified').mean).toBe(1)
  })

  it('shuffling is deterministic for a given seed', () => {
    const labels = sortedLabels(4, 20)
    expect(crossValidate(labels, 2, 'shuffled', 3).order).toEqual(crossValidate(labels, 2, 'shuffled', 3).order)
  })
})

describe('foldSizes', () => {
  it('splits like np.array_split', () => {
    expect(foldSizes(80, 3)).toEqual([27, 27, 26])
    expect(foldSizes(10, 4)).toEqual([3, 3, 2, 2])
  })
})
