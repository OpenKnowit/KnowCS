import { describe, expect, it } from 'vitest'
import { numericalGrad, Value } from './autograd'

describe('autograd 反向传播', () => {
  it('加法梯度均为 1', () => {
    const a = new Value(3)
    const b = new Value(4)
    const c = a.add(b)
    c.backward()
    expect(a.grad).toBe(1)
    expect(b.grad).toBe(1)
  })

  it('乘法梯度为对方的值', () => {
    const a = new Value(3)
    const b = new Value(4)
    const c = a.mul(b)
    c.backward()
    expect(a.grad).toBe(4)
    expect(b.grad).toBe(3)
  })

  it('ReLU 在负输入处梯度为 0', () => {
    const x = new Value(-2)
    const y = x.relu()
    y.backward()
    expect(y.data).toBe(0)
    expect(x.grad).toBe(0)
  })

  it('复合表达式 L = relu(a*b + c) 的梯度正确', () => {
    // a=2, b=3, c=-1 → d=6, e=5, L=relu(5)=5
    const a = new Value(2)
    const b = new Value(3)
    const c = new Value(-1)
    const L = a.mul(b).add(c).relu()
    L.backward()
    expect(L.data).toBe(5)
    // dL/dc = 1, dL/da = b = 3, dL/db = a = 2
    expect(a.grad).toBe(3)
    expect(b.grad).toBe(2)
    expect(c.grad).toBe(1)
  })

  it('解析梯度与数值梯度一致（tanh 神经元）', () => {
    // o = tanh(w*x + b)，对 w 求导
    const x = 1.5
    const b = 0.3
    const forward = (w: number) => Math.tanh(w * x + b)
    const w = new Value(0.8)
    const xv = new Value(x)
    const bv = new Value(b)
    const o = w.mul(xv).add(bv).tanh()
    o.backward()
    expect(w.grad).toBeCloseTo(numericalGrad(forward, 0.8), 5)
  })

  it('共享节点的梯度会累加', () => {
    // y = a + a = 2a → dy/da = 2
    const a = new Value(5)
    const y = a.add(a)
    y.backward()
    expect(a.grad).toBe(2)
  })
})
