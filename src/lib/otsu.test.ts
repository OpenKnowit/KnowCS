import { describe, expect, it } from 'vitest'
import { contrastStretch, histogram, otsuIterations, threshold } from './otsu'

describe('otsuIterations', () => {
  it('reproduces Final 2022 Part A Q4(b)(iii): start at the mean 9.375', () => {
    const px = [0, 2, 4, 4, 4, 9, 12, 10, 8, 15, 18, 14, 8, 14, 16, 12]
    const [first] = otsuIterations(px)
    expect(first.T).toBe(9.375)
    expect(first.mu1).toBe(4.875)
    expect(first.mu2).toBe(13.875)
    expect(first.next).toBe(9.375)
  })

  it('reproduces Final 2024 Q5(b): T = 100 → 74.5 and the bottom row turns white', () => {
    const px = [2, 4, 8, 16, 32, 64, 128, 128, 128]
    const [first] = otsuIterations(px, 100)
    expect([first.mu1, first.mu2, first.next]).toEqual([21, 128, 74.5])
    expect(threshold(px, first.next)).toEqual([0, 0, 0, 0, 0, 0, 255, 255, 255])
  })

  it('stops once T stops changing', () => {
    const it = otsuIterations([0, 0, 10, 200, 210, 255])
    expect(it[it.length - 1].next).toBeCloseTo(it[it.length - 1].T, 9)
  })
})

describe('point operations', () => {
  it('stretches to 0…255', () => {
    expect(contrastStretch([100, 150, 200])).toEqual([0, 128, 255])
  })

  it('counts pixels per bin', () => {
    expect(histogram([0, 0, 255], 4)).toEqual([2, 0, 0, 1])
  })
})
