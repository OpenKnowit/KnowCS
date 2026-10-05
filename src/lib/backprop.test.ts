import { describe, expect, it } from 'vitest'
import { numericalGrad } from './autograd'
import { applyUpdate, sigmoid, traceNeuron } from './backprop'
import type { NeuronParams } from './backprop'

const P: NeuronParams = { oi: 1, oj: 0.5, wik: 0.4, wjk: -0.6, target: 1, eta: 0.5 }

/** 以给定权重计算 E = ½(T − σ(net))²，供数值梯度校验 */
const errorAt = (p: NeuronParams) => traceNeuron(p).error

describe('traceNeuron', () => {
  it('黄金值：net = 0.4·1 + (−0.6)·0.5 = 0.1，O = σ(0.1)', () => {
    const t = traceNeuron(P)
    expect(t.net).toBeCloseTo(0.1, 12)
    expect(t.out).toBeCloseTo(sigmoid(0.1), 12)
    expect(t.out).toBeCloseTo(0.524979, 6)
    expect(t.fprime).toBeCloseTo(t.out * (1 - t.out), 12)
    expect(t.error).toBeCloseTo(0.5 * (1 - t.out) ** 2, 12)
  })

  it('δ_k = −∂E/∂net_k（与数值梯度一致）', () => {
    const t = traceNeuron(P)
    const eOfNet = (net: number) => 0.5 * (P.target - sigmoid(net)) ** 2
    expect(t.delta).toBeCloseTo(-numericalGrad(eOfNet, t.net), 8)
  })

  it('Δw = −η·∂E/∂w（链式法则结果与数值梯度一致）', () => {
    const t = traceNeuron(P)
    const dEdwik = numericalGrad((w) => errorAt({ ...P, wik: w }), P.wik)
    const dEdwjk = numericalGrad((w) => errorAt({ ...P, wjk: w }), P.wjk)
    expect(t.dwik).toBeCloseTo(-P.eta * dEdwik, 8)
    expect(t.dwjk).toBeCloseTo(-P.eta * dEdwjk, 8)
  })

  it('输出低于目标时 δ > 0、权重增大；输出高于目标时相反（注意 T − O 的符号）', () => {
    expect(traceNeuron({ ...P, target: 1 }).delta).toBeGreaterThan(0)
    expect(traceNeuron({ ...P, target: 0 }).delta).toBeLessThan(0)
  })

  it('输入 O_x = 0 时对应权重不更新（Δw ∝ O_x）', () => {
    expect(traceNeuron({ ...P, oj: 0 }).dwjk).toBe(0)
  })
})

describe('applyUpdate', () => {
  it('w ← w + Δw', () => {
    const t = traceNeuron(P)
    const next = applyUpdate(P)
    expect(next.wik).toBeCloseTo(P.wik + t.dwik, 12)
    expect(next.wjk).toBeCloseTo(P.wjk + t.dwjk, 12)
    expect(next.oi).toBe(P.oi)
  })

  it('反复更新使误差单调下降（梯度下降）', () => {
    let p = P
    let prev = errorAt(p)
    for (let i = 0; i < 20; i++) {
      p = applyUpdate(p)
      const e = errorAt(p)
      expect(e).toBeLessThan(prev)
      prev = e
    }
  })
})
