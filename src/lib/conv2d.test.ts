import { describe, expect, it } from 'vitest'
import { convolve, outputSize, pad, dilatedConv } from './conv2d'

const values = (g: ReturnType<typeof pad>) => g.map((r) => r.map((c) => c.value))
const LECTURE_IMG = [[10, 1, 3, 2, 6], [4, 3, 5, 8, 0], [8, 7, 9, 6, 5]]

describe('pad', () => {
  it('replicates boundary pixels like lecture 7', () => {
    expect(values(pad(LECTURE_IMG, 1, 'replicate'))[0]).toEqual([10, 10, 1, 3, 2, 6, 6])
  })

  it('mirrors without repeating the edge like lecture 7', () => {
    const g = values(pad(LECTURE_IMG, 1, 'mirror'))
    expect(g[0]).toEqual([3, 4, 3, 5, 8, 0, 8])
    expect(g[1]).toEqual([1, 10, 1, 3, 2, 6, 2])
  })

  it('reflects with the edge repeated like Final 2022 (2×2 → 4×4)', () => {
    expect(values(pad([[0, 9], [18, 27]], 1, 'reflect'))).toEqual([[0, 0, 9, 9], [0, 0, 9, 9], [18, 18, 27, 27], [18, 18, 27, 27]])
  })

  it('reflects by 2 like Final 2024 Q5(c)', () => {
    const g = values(pad([[2, 4, 8], [16, 32, 64], [128, 128, 128]], 2, 'reflect'))
    expect(g[0]).toEqual([32, 16, 16, 32, 64, 64, 32])
    expect(g[1]).toEqual([4, 2, 2, 4, 8, 8, 4])
    expect(g[6]).toEqual([32, 16, 16, 32, 64, 64, 32])
  })
})

describe('convolve', () => {
  it('solves the lecture 8 practice problem (flip, zero pad 1)', () => {
    const img = [[5, 1, 3, 7, 8], [2, 4, 6, 1, 7], [3, 4, 9, 7, 2], [4, 9, 4, 5, 9], [7, 9, 8, 7, 9]]
    const r = convolve(img, [[3, 0, 0], [0, 0, 0], [0, 0, -3]], { pad: 1, mode: 'zero', stride: 1, flip: true })
    expect(r.out[0]).toEqual([12, 18, 3, 21, 0])
    expect(r.out[1]).toEqual([12, 12, 18, -3, -21])
  })

  it('gives both accepted answers of Final 2022 Part B Q1(a)', () => {
    const k = [[1, 6, 7], [2, 5, 8], [3, 4, 9]]
    const o = { pad: 1, mode: 'zero' as const, stride: 2 }
    expect(convolve([[2, 4], [1, 3]], k, { ...o, flip: false }).out).toEqual([[73]])
    expect(convolve([[2, 4], [1, 3]], k, { ...o, flip: true }).out).toEqual([[27]])
  })

  it('averages like Final 2022 Part A Q4(b)(ii)', () => {
    const img = [[0, 0, 9, 9], [0, 0, 9, 9], [18, 18, 27, 27], [18, 18, 27, 27]]
    const avg = Array.from({ length: 3 }, () => [1 / 9, 1 / 9, 1 / 9])
    const out = convolve(img, avg, { pad: 1, mode: 'zero', stride: 1, flip: false }).out.map((r) => r.map(Math.round))
    expect(out).toEqual([[0, 2, 4, 4], [4, 9, 12, 10], [8, 15, 18, 14], [8, 14, 16, 12]])
  })
})

describe('outputSize', () => {
  it('matches lecture 8: 7×7 input, 3×3 kernel, stride 1/2/3', () => {
    expect(outputSize(7, 3, 0, 1)).toEqual({ size: 5, exact: true })
    expect(outputSize(7, 3, 0, 2)).toEqual({ size: 3, exact: true })
    expect(outputSize(7, 3, 0, 3).exact).toBe(false)
  })
})

describe('dilatedConv (Final 2024 Q6a)', () => {
  it('reproduces the printed test-script output', () => {
    const img = Array.from({ length: 10 }, (_, r) => Array.from({ length: 10 }, (_, c) => r * 10 + c))
    const res = dilatedConv(img, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 2, 2, 'same')
    expect(res.pad).toBe(2)
    expect(res.effective).toBe(5)
    expect(res.out).toEqual([
      [22, 26, 30, 34, 8],
      [62, 66, 72, 78, 34],
      [102, 126, 132, 138, 74],
      [142, 186, 192, 198, 114],
      [80, 142, 146, 150, 154],
    ])
  })
  it('dilation 1, valid padding is an ordinary convolution', () => {
    const img = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    expect(dilatedConv(img, [[1, 1], [1, 1]]).out).toEqual([[12, 16], [24, 28]])
  })
})
