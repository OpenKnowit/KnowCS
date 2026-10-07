import { describe, expect, it } from 'vitest'
import { circleMask, convolveValid, flattenConv1d, separableBlur } from './imageNp'
import { convolve } from './conv2d'

describe('Final 2024 Q2 helpers', () => {
  it('np.convolve valid follows the paper’s definition', () => {
    expect(convolveValid([1, 2, 3], [0, 1, 0.5])).toEqual([2.5])
    // the paper prints [2.5, 3.5] for this one; NumPy (and its own convolve_valid definition) give [2.5, 4]
    expect(convolveValid([1, 2, 3, 4], [0, 1, 0.5])).toEqual([2.5, 4])
  })

  it('circle mask includes the boundary', () => {
    const { mask } = circleMask(5, 5, 2, 2, 2)
    expect(mask[0][2]).toBe(1) // distance exactly 2
    expect(mask[0][0]).toBe(0)
    expect(mask.flat().reduce((a, b) => a + b, 0)).toBe(13)
  })

  it('the flatten trick never mixes neighbouring rows', () => {
    const img = [[1, 2, 3], [10, 20, 30]]
    expect(flattenConv1d(img, [1, 1, 1]).out).toEqual([[3, 6, 5], [30, 60, 50]])
  })

  it('two passes equal a zero-padded 3×3 mean filter', () => {
    const img = Array.from({ length: 6 }, (_, i) => Array.from({ length: 7 }, (_, j) => (i * 7 + j * 3) % 11))
    const third = 1 / 3
    const box = convolve(img, Array.from({ length: 3 }, () => [1 / 9, 1 / 9, 1 / 9]), { pad: 1, mode: 'zero', stride: 1, flip: false }).out
    const sep = separableBlur(img, [third, third, third])
    sep.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(box[i][j], 9)))
  })
})
