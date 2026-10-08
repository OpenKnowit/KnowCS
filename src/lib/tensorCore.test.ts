import { describe, expect, it } from 'vitest'
import { SandboxRandom } from './ndops'
import { Engine, TT, backward, bce, binary, conv2d, crossEntropy, formatTensor, linear, matmul, mseLoss, pool2d, reduceSum, softmax, transposeT, reshapeT, unary } from './tensorCore'

const E = () => new Engine(new SandboxRandom(1))
const T = (v: number[], shape: number[], dtype: TT['dtype'] = 'float32', grad = false) => {
  const t = TT.of(v, shape, dtype)
  t.requiresGrad = grad
  return t
}

describe('printing matches PyTorch 2.12', () => {
  const cases: [TT, string][] = [
    [T([1, 3, 5], [3], 'int64'), 'tensor([1, 3, 5])'],
    [T([1, 2, 3, 4], [2, 2]), 'tensor([[1., 2.],\n        [3., 4.]])'],
    [T([0.1, 0.25], [2]), 'tensor([0.1000, 0.2500])'],
    [T([22], [], 'int64'), 'tensor(22)'],
    [T([2], []), 'tensor(2.)'],
    [T([1e5, 2e-3], [2]), 'tensor([1.0000e+05, 2.0000e-03])'],
    [T([1, 0], [2], 'bool'), 'tensor([ True, False])'],
    [T([1, 2], [2], 'float64'), 'tensor([1., 2.], dtype=torch.float64)'],
    [T([1, 2], [2], 'int32'), 'tensor([1, 2], dtype=torch.int32)'],
    [T([1, 2], [2], 'float32', true), 'tensor([1., 2.], requires_grad=True)'],
    [T([0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2]), 'tensor([[[0., 0.],\n         [0., 0.]],\n\n        [[0., 0.],\n         [0., 0.]]])'],
    [T([1, -20, 300, 4], [2, 2], 'int64'), 'tensor([[  1, -20],\n        [300,   4]])'],
    [T([-0.5, 1.25, 100], [3]), 'tensor([ -0.5000,   1.2500, 100.0000])'],
    [T([], [0]), 'tensor([])'],
    [T([123456, 1], [2]), 'tensor([1.2346e+05, 1.0000e+00])'],
    [T([1.123456789], [1]), 'tensor([1.1235])'],
  ]
  it.each(cases.map(([t, s]) => [s, t] as const))('%s', (s, t) => expect(formatTensor(t)).toBe(s))

  it('float32 storage keeps float32 precision', () => {
    expect(T([0.1], []).values()[0]).toBe(0.10000000149011612)
  })

  it('a non-leaf shows its grad_fn', () => {
    const e = E()
    const x = T([2], [], 'float32', true)
    expect(formatTensor(binary(e, '*', x, T([3], [], 'int64'), false, true))).toBe('tensor(6., grad_fn=<MulBackward0>)')
  })
})

/** compare autograd with central differences for f(x) where x is input k */
const gradCheck = (build: (e: Engine, xs: TT[]) => TT, inputs: TT[], tol = 1e-3) => {
  const e = E()
  const xs = inputs.map((x) => { const c = TT.of(x.values(), x.shape, 'float64'); c.requiresGrad = true; return c })
  const out = build(e, xs)
  backward(e, out, null, false)
  xs.forEach((x, k) => {
    const v = x.values()
    v.forEach((_, i) => {
      const at = (d: number) => {
        const ys = xs.map((y, j) => TT.of(j === k ? v.map((w, q) => (q === i ? w + d : w)) : y.values(), y.shape, 'float64'))
        const e2 = E()
        e2.grad = false
        return build(e2, ys).values()[0]
      }
      const num = (at(1e-5) - at(-1e-5)) / 2e-5
      expect(x.grad!.values()[i]).toBeCloseTo(num, -Math.log10(tol))
    })
  })
}

describe('gradients match finite differences', () => {
  const r = new SandboxRandom(7)
  const rnd = (shape: number[]) => T(Array.from({ length: shape.reduce((a, b) => a * b, 1) }, () => r.normal()), shape, 'float64')
  it('broadcast * and / ', () => gradCheck((e, [a, b]) => reduceSum(e, binary(e, '/', binary(e, '*', a, b), binary(e, '+', b, T([3], [], 'float64'))), null, false, false), [rnd([2, 3]), rnd([3])]))
  it('matmul', () => gradCheck((e, [a, b]) => reduceSum(e, matmul(e, a, b), null, false, false), [rnd([2, 3]), rnd([3, 4])]))
  it('transpose + view', () => gradCheck((e, [a]) => reduceSum(e, unary(e, 'square', reshapeT(e, transposeT(e, a, 0, 1), [6], false)), null, false, false), [rnd([2, 3])]))
  it('linear', () => gradCheck((e, [x, w, b]) => reduceSum(e, unary(e, 'tanh', linear(e, x, w, b)), null, false, false), [rnd([2, 3]), rnd([4, 3]), rnd([4])]))
  it('conv2d with stride and padding', () => gradCheck((e, [x, w, b]) => reduceSum(e, unary(e, 'square', conv2d(e, x, w, b, { stride: [2, 1], padding: [1, 1] })), null, false, false), [rnd([1, 2, 4, 4]), rnd([3, 2, 3, 3]), rnd([3])]))
  it('max_pool2d', () => gradCheck((e, [x]) => reduceSum(e, unary(e, 'square', pool2d(e, x, [2, 2], [2, 2], [0, 0], 'max')), null, false, false), [rnd([1, 1, 4, 4])]))
  it('softmax', () => gradCheck((e, [x]) => reduceSum(e, binary(e, '*', softmax(e, x, 1, false), T([1, 2, 3, 4, 5, 6], [2, 3], 'float64')), null, false, false), [rnd([2, 3])]))
  it('cross_entropy', () => gradCheck((e, [x]) => crossEntropy(e, x, T([2, 0], [2], 'int64')), [rnd([2, 3])]))
  it('mse and bce with logits', () => {
    gradCheck((e, [a]) => mseLoss(e, a, T([1, 2, 3], [3], 'float64')), [rnd([3])])
    gradCheck((e, [z]) => bce(e, z, T([1, 0, 1], [3], 'float64'), true), [rnd([3])])
  })
})

describe('autograd bookkeeping', () => {
  it('leaf gradients accumulate; a freed graph cannot be reused', () => {
    const e = E()
    const x = T([2], [], 'float32', true)
    backward(e, binary(e, '*', x, x), null, true)
    expect(x.grad!.values()).toEqual([4])
    const y = binary(e, '*', x, x)
    backward(e, y, null, false)
    expect(x.grad!.values()).toEqual([8])
    expect(() => backward(e, y, null, false)).toThrow('Trying to backward through the graph a second time')
  })
  it('no grad → error; non-scalar root → error', () => {
    const e = E()
    expect(() => backward(e, T([1, 2, 3], [3]), null, false)).toThrow('does not require grad')
    expect(() => backward(e, binary(e, '*', T([1, 2], [2], 'float32', true), T([2], [])), null, false)).toThrow('only for scalar outputs')
  })
  it('the lecture 9 example: z = x·w + b, BCE with logits', () => {
    const e = E()
    const x = T([1, 1, 1, 1, 1], [5])
    const y = T([0, 0, 0], [3])
    const w = T(Array.from({ length: 15 }, (_, i) => (i % 3) * 0.1), [5, 3], 'float32', true)
    const b = T([0.1, 0.2, 0.3], [3], 'float32', true)
    const loss = bce(e, binary(e, '+', matmul(e, x, w), b), y, true)
    backward(e, loss, null, false)
    // d loss / d b_j = sigmoid(z_j) / 3, and every row of w.grad equals b.grad (x is all ones)
    const z = [0.1, 0.7, 1.3]
    b.grad!.values().forEach((g, j) => expect(g).toBeCloseTo(1 / (1 + Math.exp(-z[j])) / 3, 5))
    expect(w.grad!.values().slice(0, 3)).toEqual(b.grad!.values())
  })
})
