import { describe, expect, it } from 'vitest'
import { NDArray } from './ndarray'
import { allclose, average, bincount, diff, histogram, isin, median, meshgrid, pad, padWidths, percentile } from './ndextra'

// Expected values are NumPy 2.2.6's output (checked on 2026-10-08).

const A = (v: number[], shape = [v.length], dtype: 'int64' | 'float64' = 'int64') => NDArray.create(v, shape, dtype)
const out = (r: NDArray | number) => (typeof r === 'number' ? r : r.values())

describe('counting', () => {
  it('bincount counts each value and keeps minlength', () => {
    expect(bincount(A([0, 1, 1, 3])).out.values()).toEqual([1, 2, 0, 1])
    expect(bincount(A([1, 1]), 4).out.values()).toEqual([0, 2, 0, 0])
    expect(bincount(A([0, 1, 1, 3])).groups[1]).toEqual([[0, 1], [0, 2]])
  })
  it('bincount with weights sums them (a weighted KNN vote)', () => {
    const r = bincount(A([0, 2, 2, 1, 2]), 0, A([1, 0.5, 0.5, 2, 1], [5], 'float64'))
    expect(r.out.values()).toEqual([1, 2, 2])
    expect(r.out.dtype).toBe('float64')
  })
  it('bincount rejects negatives', () => {
    expect(() => bincount(A([-1, 2]))).toThrow(/negative/)
  })
  it('histogram: equal-width bins, last bin closed', () => {
    const r = histogram(A([1, 2, 2, 3, 7, 8]), 3)
    expect(r.out.values()).toEqual([4, 0, 2])
    expect(r.edges.values().map((x) => +x.toFixed(8))).toEqual([1, 3.33333333, 5.66666667, 8])
    expect(histogram(A([0.1, 0.5, 0.9], [3], 'float64'), 2, [0, 1]).out.values()).toEqual([1, 2])
  })
})

describe('order statistics', () => {
  it('median and percentile interpolate linearly', () => {
    expect(median(A([3, 1, 2, 10]), null)).toBe(2.5)
    expect(out(median(A([1, 5, 3, 4, 2, 6], [2, 3]), 1))).toEqual([3, 4])
    expect(percentile(A([1, 2, 3, 4, 10]), 25, null)).toBe(2)
    expect(out(percentile(A([1, 2, 3, 4], [2, 2]), 50, 0))).toEqual([2, 3])
    expect(percentile(A([1, 2, 3, 4]), 75, null)).toBe(3.25)
    expect(median(A([1, NaN, 2], [3], 'float64'), null)).toBeNaN()
  })
  it('average with and without weights', () => {
    expect(average(A([1, 2, 3]), null, A([3, 1, 1]))).toBeCloseTo(1.6)
    expect(out(average(A([1, 2, 3, 4], [2, 2]), 0, null))).toEqual([2, 3])
  })
})

describe('grids, differences, padding, membership', () => {
  it('meshgrid repeats x down the rows and y across the columns', () => {
    const { X, Y } = meshgrid(A([0, 1, 2]), A([10, 20]))
    expect(X.out.shape).toEqual([2, 3])
    expect(X.out.values()).toEqual([0, 1, 2, 0, 1, 2])
    expect(Y.out.values()).toEqual([10, 10, 10, 20, 20, 20])
  })
  it('diff along the last axis or axis 0', () => {
    expect(diff(A([1, 4, 9, 16])).out.values()).toEqual([3, 5, 7])
    const m = A([1, 3, 6, 10], [2, 2])
    expect(diff(m, 0).out.values()).toEqual([5, 7])
    expect(diff(m, 0).out.shape).toEqual([1, 2])
    expect(diff(m).out.values()).toEqual([2, 4])
    expect(diff(m).out.shape).toEqual([2, 1])
  })
  it('pad reads pad_width like NumPy and fills the border', () => {
    const m = A([1, 2, 3, 4], [2, 2])
    expect(pad(m, padWidths(1, 2)).out.values()).toEqual([0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0])
    expect(pad(A([1, 2]), padWidths([2, 1], 1), 9).out.values()).toEqual([9, 9, 1, 2, 9])
    const r = pad(m, padWidths([[0, 1], [1, 0]], 2))
    expect(r.out.shape).toEqual([3, 3])
    expect(r.out.values()).toEqual([0, 1, 2, 0, 3, 4, 0, 0, 0])
    expect(r.groups[1]).toEqual([[0, 0]])
    expect(r.groups[0]).toEqual([])
  })
  it('isin marks members and points at the match', () => {
    const r = isin(A([1, 2, 3]), A([2, 3, 5]))
    expect(r.out.values()).toEqual([0, 1, 1])
    expect(r.groups[1]).toEqual([[0, 1], [1, 0]])
  })
  it('allclose uses rtol 1e-5 and atol 1e-8', () => {
    expect(allclose([1, 2], [1, 2.0000001])).toBe(true)
    expect(allclose([1], [1.001])).toBe(false)
  })
})
