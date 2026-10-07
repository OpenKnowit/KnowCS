import { describe, expect, it } from 'vitest'
import { bceWithLogits, forwardBackward, lectureExample } from './autogradDemo'

describe('lecture 9 autograd example', () => {
  const ex = lectureExample()
  it('has the lecture’s shapes and is reproducible', () => {
    expect(ex.w).toHaveLength(5)
    expect(ex.w[0]).toHaveLength(3)
    expect(lectureExample()).toEqual(ex)
  })
  it('gradients match finite differences', () => {
    const h = 1e-6
    for (let i = 0; i < 5; i++)
      for (let j = 0; j < 3; j++) {
        const w2 = ex.w.map((row) => [...row])
        w2[i][j] += h
        const num = (forwardBackward(ex.x, ex.y, w2, ex.b).loss - ex.loss) / h
        expect(ex.wGrad[i][j]).toBeCloseTo(num, 5)
      }
    for (let j = 0; j < 3; j++) {
      const b2 = [...ex.b]
      b2[j] += h
      expect(ex.bGrad[j]).toBeCloseTo((forwardBackward(ex.x, ex.y, ex.w, b2).loss - ex.loss) / h, 5)
    }
  })
  it('with x all ones every row of w.grad equals b.grad (as in the lecture output)', () => {
    for (const row of ex.wGrad) expect(row).toEqual(ex.bGrad)
  })
  it('BCE with logits is stable for large logits', () => {
    expect(bceWithLogits([100, -100], [1, 0])).toBeCloseTo(0, 10)
    expect(Number.isFinite(bceWithLogits([1000], [0]))).toBe(true)
  })
})
