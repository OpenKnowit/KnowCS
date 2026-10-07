import { describe, expect, it } from 'vitest'
import { bayesPosterior, naturalCounts } from './bayesRule'

describe('bayesPosterior', () => {
  it('reproduces the lecture 2 virus test (0.1%, 99%, 95%)', () => {
    const r = bayesPosterior({ prior: 0.001, hit: 0.99, falseAlarm: 0.05, observed: 'E' })
    expect(r.evidence).toBeCloseTo(0.05094, 6)
    expect(r.posterior).toBeCloseTo(0.0194, 4)
  })

  it('handles P(B | not E) like the 2022 Spring midterm (answer 5/68)', () => {
    const r = bayesPosterior({ prior: 0.1, hit: 0.5, falseAlarm: 0.3, observed: 'notE' })
    expect(r.posterior).toBeCloseTo(5 / 68, 10)
  })

  it('reproduces the 2022 Fall midterm A+ question (0.12 then 0.63)', () => {
    const r = bayesPosterior({ prior: 0.08, hit: 0.95, falseAlarm: 0.05, observed: 'E' })
    expect(r.evidence).toBeCloseTo(0.122, 3)
    expect(r.posterior).toBeCloseTo(0.623, 3)
  })
})

describe('naturalCounts', () => {
  it('always sums to n', () => {
    for (const [p, h, f] of [[0.001, 0.99, 0.05], [0.333, 0.71, 0.13], [0.5, 0.5, 0.5]]) {
      const c = naturalCounts(p, h, f, 10000)
      expect(c.tp + c.fn + c.fp + c.tn).toBe(10000)
    }
  })

  it('gives 10 true and 500 false positives for the virus test', () => {
    expect(naturalCounts(0.001, 0.99, 0.05)).toEqual({ tp: 10, fn: 0, fp: 500, tn: 9490 })
  })
})
