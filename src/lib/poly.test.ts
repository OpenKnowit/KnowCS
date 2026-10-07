import { describe, expect, it } from 'vitest'
import { DEFAULT_IV, complexRepr, convert, deriv, integ, lstsq, mapParams, polyRepr, polyStr, polymul, polyval, roots } from './poly'

// Expected strings come from NumPy 2.2.6.
describe('numpy.polynomial subset', () => {
  const p = [1, 2, 3]

  it('evaluates, differentiates and integrates like the note says', () => {
    expect(polyval(p, 2)).toBe(17)
    expect(deriv(p)).toEqual([2, 6])
    expect(integ(p)).toEqual([0, 1, 1, 1])
    expect(polymul(p, [0, 1])).toEqual([0, 1, 2, 3])
  })

  it('prints like numpy', () => {
    expect(polyRepr([1, 2, 3], DEFAULT_IV, DEFAULT_IV)).toBe("Polynomial([1., 2., 3.], domain=[-1.,  1.], window=[-1.,  1.], symbol='x')")
    expect(polyStr([1, 2, 3])).toBe('1.0 + 2.0·x + 3.0·x²')
    expect(polyStr([1, -3, 0.5])).toBe('1.0 - 3.0·x + 0.5·x²')
  })

  it('finds complex roots of 1 + 2x + 3x² and prints them like numpy', () => {
    const r = roots(p)
    expect(r.real).toBe(false)
    expect(complexRepr(r.re, r.im)).toBe('array([-0.33333333-0.47140452j, -0.33333333+0.47140452j])')
  })

  it('returns sorted real roots when they are all real', () => {
    const r = roots([1, -3, 2])
    expect(r.real).toBe(true)
    expect(r.re[0]).toBeCloseTo(0.5, 12)
    expect(r.re[1]).toBeCloseTo(1, 12)
    expect(roots([-6, 11, -6, 1]).re.map((v) => Math.round(v * 1e9) / 1e9)).toEqual([1, 2, 3])
  })

  it('fits in the window variable and converts back (exact quadratic)', () => {
    const x = [0, 1, 2, 3, 4]
    const y = x.map((v) => 0.5 * v * v - 2 * v + 3)
    const { off, scl } = mapParams([0, 4], DEFAULT_IV)
    const c = lstsq(x.map((v) => off + scl * v), y, 2)
    expect(c[0]).toBeCloseTo(1, 10)
    expect(c[2]).toBeCloseTo(2, 10)
    const std = convert(c, [0, 4], DEFAULT_IV)
    expect(std.map((v) => Math.round(v * 1e9) / 1e9)).toEqual([3, -2, 0.5])
  })
})
