import { describe, expect, it } from 'vitest'
import { bowl, gdPath } from './gradient'

describe('gradient descent on a bowl', () => {
  const { df } = bowl(2)
  it('creeps with a small rate, lands fast with a good one', () => {
    const slow = gdPath(-2, 0.1, 8, df)
    expect(slow[1]).toBeCloseTo(-1.2) // distance shrinks by 1 − 2η = 0.8
    expect(Math.abs(slow[8] - 2)).toBeGreaterThan(0.5)
    expect(Math.abs(gdPath(-2, 0.45, 8, df)[8] - 2)).toBeLessThan(1e-6)
  })
  it('overshoots back and forth, and diverges past η = 1', () => {
    const big = gdPath(-2, 1.05, 6, df)
    expect(Math.sign(big[1] - 2)).toBe(1)
    expect(Math.sign(big[2] - 2)).toBe(-1)
    expect(Math.abs(big[6] - 2)).toBeGreaterThan(Math.abs(big[0] - 2))
  })
})
