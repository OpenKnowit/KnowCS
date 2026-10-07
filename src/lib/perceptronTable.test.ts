import { describe, expect, it } from 'vitest'
import { perceptronTable } from './perceptronTable'

const pick = (r: { o: number; w: number[]; theta: number }) => [r.o, ...r.w, r.theta]

describe('perceptronTable', () => {
  it('reproduces the 2022 Fall table (f = 1 if z > 0, else 0)', () => {
    const data = [[1, 1, 0], [3, 2, 1], [4, 5, 0], [3, 4, 1], [2, 3, 1]].map(([a, b, t]) => ({ x: [a, b], t }))
    const { rows } = perceptronTable(data, [0, 0], -1, 1, { high: 1, low: 0, rule: 'gt' }, 1)
    expect(rows.map(pick)).toEqual([
      [0, 0, 0, -1],
      [0, 3, 2, 0],
      [1, -1, -3, -1],
      [0, 2, 1, 0],
      [1, 2, 1, 0],
    ])
  })

  it('reproduces the 2022 Spring table (f = 1 if z ≥ 0, else −1)', () => {
    const xs = [[10, 10], [0, 0], [8, 4], [3, 3], [4, 8], [0.5, 0.5], [4, 3], [2, 5]]
    const ts = [1, -1, 1, -1, 1, -1, 1, 1]
    const { rows, converged } = perceptronTable(xs.map((x, i) => ({ x, t: ts[i] })), [1, 1], 0, 1, { high: 1, low: -1, rule: 'ge' }, 1)
    expect(rows.map(pick)).toEqual([
      [1, 1, 1, 0],
      [1, 1, 1, -2],
      [1, 1, 1, -2],
      [1, -5, -5, -4],
      [-1, 3, 11, -2],
      [1, 2, 10, -4],
      [1, 2, 10, -4],
      [1, 2, 10, -4],
    ])
    expect(converged).toBe(false)
  })

  it('learns AND from the lecture 5 start (w = 0.1, 0.5, θ = −0.8, η = 0.2) and stops', () => {
    const data = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]].map(([a, b, t]) => ({ x: [a, b], t }))
    const r = perceptronTable(data, [0.1, 0.5], -0.8, 0.2, { high: 1, low: 0, rule: 'ge' }, 20)
    expect(r.converged).toBe(true)
    const last = r.rows[r.rows.length - 1]
    for (const s of data) {
      const z = s.x[0] * last.w[0] + s.x[1] * last.w[1] + last.theta
      expect(z >= 0 ? 1 : 0).toBe(s.t)
    }
    expect(r.rows[1]).toMatchObject({ o: 0, w: [0.1, 0.5], theta: -0.8 })
  })

  it('never converges on XOR', () => {
    const data = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]].map(([a, b, t]) => ({ x: [a, b], t }))
    expect(perceptronTable(data, [0, 0], 0, 1, { high: 1, low: 0, rule: 'ge' }, 50).converged).toBe(false)
  })
})
