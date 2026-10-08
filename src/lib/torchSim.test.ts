import { describe, expect, it } from 'vitest'
import { bridge, graph, lectureTensor, loopError, matmul, mul, replay, sizeStr, transpose } from './torchSim'

// Checked on 2026-10-08 against PyTorch 2.14.1: dtypes and sizes, @ vs *, the NumPy bridge, w.grad -1 then -2 after
// a second backward(), and requires_grad False under no_grad all match.

describe('tensor attributes', () => {
  it('prints sizes like PyTorch', () => {
    expect(sizeStr([3])).toBe('torch.Size([3])')
    expect(sizeStr([])).toBe('torch.Size([])')
  })
})

describe('@ versus * on the lecture example', () => {
  const t = lectureTensor()
  it('tensor @ tensor.T counts the shared ones (3 everywhere)', () => {
    expect(matmul(t, transpose(t))).toEqual([[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]])
  })
  it('tensor * tensor is element-wise and keeps the zero column', () => {
    expect(mul(t, t)).toEqual(t)
  })
  it('agg = tensor.sum() gives 12', () => {
    expect(t.flat().reduce((a, b) => a + b, 0)).toBe(12)
  })
})

describe('NumPy bridge', () => {
  it('add_ changes the array too (the lecture’s output: both become 2)', () => {
    expect(bridge(['add_'])).toEqual({ tensor: [2, 2, 2, 2, 2], array: [2, 2, 2, 2, 2], shared: true })
  })
  it('np.add(..., out=array) changes the tensor', () => {
    expect(bridge(['np_add_out']).tensor).toEqual([2, 2, 2, 2, 2])
  })
  it('x = x + 1 makes a new tensor; later in-place ops no longer reach the array', () => {
    expect(bridge(['plus', 'add_'])).toEqual({ tensor: [3, 3, 3, 3, 3], array: [1, 1, 1, 1, 1], shared: false })
  })
})

describe('scalar autograd', () => {
  it('dL/dz = σ(z) − y, dL/dw = (σ(z) − y)·x, dL/db = σ(z) − y', () => {
    const g = graph(0.5, 2, -1, 1) // z = 0
    expect(g.z).toBe(0)
    expect(g.p).toBeCloseTo(0.5)
    expect(g.loss).toBeCloseTo(Math.log(2))
    expect(g.dw).toBeCloseTo(-1)
    expect(g.db).toBeCloseTo(-0.5)
  })
  it('matches a numerical derivative', () => {
    const [w, x, b, y, h] = [0.3, 1.7, 0.2, 0, 1e-6]
    const num = (graph(w + h, x, b, y).loss - graph(w - h, x, b, y).loss) / (2 * h)
    expect(graph(w, x, b, y).dw).toBeCloseTo(num, 6)
  })
  it('backward() twice doubles the gradient until zero_grad()', () => {
    const one = replay({ w: 0.5, b: -1 }, 2, 1, 0.1, ['backward'])
    const two = replay({ w: 0.5, b: -1 }, 2, 1, 0.1, ['backward', 'backward'])
    expect(two.wGrad).toBeCloseTo(2 * one.wGrad)
    expect(replay({ w: 0.5, b: -1 }, 2, 1, 0.1, ['backward', 'backward', 'zero_grad']).wGrad).toBe(0)
  })
  it('step() moves w against the gradient', () => {
    const r = replay({ w: 0.5, b: -1 }, 2, 1, 0.1, ['backward', 'step'])
    expect(r.w).toBeCloseTo(0.5 + 0.1)
    expect(r.b).toBeCloseTo(-1 + 0.05)
  })
})

describe('training loop order', () => {
  it('accepts the lecture order and zero_grad at the end', () => {
    expect(loopError(['zero_grad', 'forward', 'loss', 'backward', 'step'])).toBeNull()
    expect(loopError(['forward', 'loss', 'backward', 'step', 'zero_grad'])).toBeNull()
  })
  it('names the mistake', () => {
    expect(loopError(['zero_grad', 'loss', 'forward', 'backward', 'step'])).toBe('loss_first')
    expect(loopError(['zero_grad', 'forward', 'backward', 'loss', 'step'])).toBe('backward_first')
    expect(loopError(['zero_grad', 'forward', 'loss', 'step', 'backward'])).toBe('step_early')
    expect(loopError(['forward', 'loss', 'backward', 'zero_grad', 'step'])).toBe('zero_late')
  })
})
