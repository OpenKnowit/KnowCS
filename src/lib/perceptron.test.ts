import { describe, expect, it } from 'vitest'
import { countErrors, predict, trainPerceptron } from './perceptron'
import { PERCEPTRON_DATA } from '../data/constants'

describe('perceptron', () => {
  it('predict 用 sign(w·x+b)', () => {
    expect(predict([1, 1], 0, { x: 2, y: 3, label: 1 })).toBe(1)
    expect(predict([1, 1], 0, { x: -2, y: -3, label: -1 })).toBe(-1)
    expect(predict([1, 1], 0, { x: 0, y: 0, label: 1 })).toBe(1) // 0 归 +1
  })

  it('在线性可分数据上收敛到 0 错分', () => {
    const steps = trainPerceptron(PERCEPTRON_DATA, 0.1, 50)
    const last = steps.at(-1)!
    expect(countErrors(PERCEPTRON_DATA, last.w, last.b)).toBe(0)
  })

  it('错分时才更新权重，分对时权重不变', () => {
    const steps = trainPerceptron(PERCEPTRON_DATA, 0.1, 50)
    for (let i = 1; i < steps.length; i++) {
      const prev = steps[i - 1]
      const cur = steps[i]
      if (cur.correct) {
        expect(cur.w).toEqual(prev.w)
        expect(cur.b).toBe(prev.b)
      }
    }
  })

  it('提前停止：最后一个 epoch 无错分', () => {
    const steps = trainPerceptron(PERCEPTRON_DATA, 0.1, 50)
    // 收敛后不应再有多余 epoch（最后 N 步覆盖一整个 epoch 且 errors 不增）
    const last = steps.at(-1)!
    expect(last.errors).toBe(0)
  })
})
