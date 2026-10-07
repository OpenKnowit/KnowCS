import { describe, expect, it } from 'vitest'
import { XOR_POINTS, classify, hidden, lineScore, orNandAnd, sweepLine } from './xorWarp'

describe('xor explainer network', () => {
  it('solves XOR through the hidden layer', () => {
    for (const p of XOR_POINTS) expect(classify(p.x)).toBe(p.t)
  })

  it('puts the four corners where the video says', () => {
    const h = XOR_POINTS.map((p) => hidden(p.x).map((v) => Number(v.toFixed(3))))
    expect(h).toEqual([
      [0.182, 0.002],
      [0.953, 0.047],
      [0.818, 0.182],
      [0.998, 0.818],
    ])
  })

  it('no single line scores 4 / 4, many score 3', () => {
    let best = 0
    for (let a = 0; a < 2 * Math.PI; a += 0.01)
      for (let d = -2; d <= 2; d += 0.01) best = Math.max(best, sweepLine(a, d).score)
    expect(best).toBe(3)
    expect(lineScore(1, 1, -0.5)).toBe(3) // OR gets (1,1) wrong
  })

  it('OR and NAND feed an AND', () => {
    expect(XOR_POINTS.map((p) => orNandAnd(p.x))).toEqual([
      { h1: 0, h2: 1, y: 0 },
      { h1: 1, h2: 1, y: 1 },
      { h1: 1, h2: 1, y: 1 },
      { h1: 1, h2: 0, y: 0 },
    ])
  })
})
