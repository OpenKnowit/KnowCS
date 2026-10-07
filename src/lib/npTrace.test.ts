import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import { dependents, explainCall } from './npTrace'

const run = (code: string) => {
  const r = runPython('import numpy as np\n' + code)
  if (r.error) throw new Error(`${r.error.type}: ${r.error.message}`)
  return r
}
const last = (code: string) => {
  const r = run(code)
  return r.calls[r.calls.length - 1]
}

describe('call recording', () => {
  it('records functions, methods, operators, .T and indexing with their source text', () => {
    const r = run('a = np.arange(6).reshape(2, 3)\nb = a.T\nc = a[0] + 10\nnp.sum(a, axis=0)')
    expect(r.calls.map((c) => [c.api, c.kind])).toEqual([
      ['np.arange', 'create'],
      ['ndarray.reshape', 'move'],
      ['ndarray.T', 'move'],
      ['index', 'move'],
      ['op:+', 'elementwise'],
      ['np.sum', 'reduce'],
    ])
    const sum = r.calls[5]
    expect(sum.code).toBe('np.sum(a, axis=0)')
    expect(sum.operands.map((o) => o.label)).toEqual(['a'])
    expect(sum.axis).toBe(0)
    expect(sum.result?.values).toEqual([3, 5, 7])
  })

  it('labels operands inside a list literal', () => {
    const c = last('a = np.zeros((2, 2))\nb = np.ones((2, 2))\nnp.concatenate([a, b], axis=1)')
    expect(c.operands.map((o) => o.label)).toEqual(['a', 'b'])
    expect(c.axis).toBe(1)
  })

  it('snapshots inputs before the call', () => {
    const c = last('a = np.array([3, 1, 2])\nnp.sort(a)')
    expect(c.operands[0].snap.values).toEqual([3, 1, 2])
  })
})

describe('explainCall', () => {
  it('reduce along axis 0 groups each column', () => {
    const e = explainCall(last('a = np.arange(6).reshape(2, 3)\na.sum(axis=0)'))
    expect(e.sources).toEqual([[[0, 0], [0, 3]], [[0, 1], [0, 4]], [[0, 2], [0, 5]]])
    expect(e.axis).toBe(0)
  })

  it('argmax picks the winning element of each row', () => {
    const e = explainCall(last('a = np.array([[1, 9, 3], [7, 2, 8]])\nnp.argmax(a, axis=1)'))
    expect(e.pick).toEqual([[0, 1], [0, 5]])
  })

  it('reduce with no axis feeds one output from everything', () => {
    expect(explainCall(last('a = np.ones((2, 2))\na.mean()')).sources).toEqual([[[0, 0], [0, 1], [0, 2], [0, 3]]])
  })

  it('broadcasting maps each output to one cell of each operand', () => {
    const e = explainCall(last('a = np.arange(3)\nb = np.arange(2).reshape(2, 1)\na + b'))
    expect(e.sources[4]).toEqual([[0, 1], [1, 1]]) // out[1, 1] = a[1] + b[1, 0]
  })

  it('scalars are operands of element-wise ops', () => {
    const c = last('a = np.arange(3)\nnp.maximum(a, 1)')
    expect(c.operands.map((o) => o.label)).toEqual(['a', '1'])
    expect(explainCall(c).sources[0]).toEqual([[0, 0], [1, 0]])
  })

  it('transpose, reshape, flip and fancy indexing trace every element back', () => {
    expect(explainCall(last('a = np.arange(6).reshape(2, 3)\na.T')).sources.flat()).toEqual([[0, 0], [0, 3], [0, 1], [0, 4], [0, 2], [0, 5]])
    expect(explainCall(last('a = np.arange(4)\nnp.flip(a)')).sources.flat()).toEqual([[0, 3], [0, 2], [0, 1], [0, 0]])
    expect(explainCall(last('a = np.arange(5) * 10\na[[4, 0]]')).sources.flat()).toEqual([[0, 4], [0, 0]])
  })

  it('concatenate and stack remember which array each element came from', () => {
    const e = explainCall(last('a = np.array([1, 2])\nb = np.array([3, 4])\nnp.stack([a, b], axis=1)'))
    expect(e.sources.flat()).toEqual([[0, 0], [1, 0], [0, 1], [1, 1]])
  })

  it('tile and repeat reuse cells', () => {
    expect(explainCall(last('a = np.array([7, 8])\nnp.tile(a, 2)')).sources.flat()).toEqual([[0, 0], [0, 1], [0, 0], [0, 1]])
    expect(explainCall(last('a = np.array([7, 8])\na.repeat(2)')).sources.flat()).toEqual([[0, 0], [0, 0], [0, 1], [0, 1]])
  })

  it('cumsum accumulates along the axis', () => {
    const e = explainCall(last('a = np.arange(6).reshape(2, 3)\nnp.cumsum(a, axis=1)'))
    expect(e.sources[5]).toEqual([[0, 3], [0, 4], [0, 5]])
    expect(explainCall(last('a = np.arange(3)\nnp.cumsum(a)')).sources[2]).toEqual([[0, 0], [0, 1], [0, 2]])
  })

  it('matmul uses row i of A and column j of B', () => {
    const e = explainCall(last('A = np.arange(6).reshape(2, 3)\nB = np.arange(6).reshape(3, 2)\nA @ B'))
    expect(e.sources[1]).toEqual([[0, 0], [0, 1], [0, 2], [1, 1], [1, 3], [1, 5]])
    expect(explainCall(last('np.outer(np.arange(2), np.arange(3))')).sources[5]).toEqual([[0, 1], [1, 2]])
  })

  it('sort maps positions to where they came from', () => {
    expect(explainCall(last('a = np.array([30, 10, 20])\nnp.sort(a)')).sources.flat()).toEqual([[0, 1], [0, 2], [0, 0]])
    expect(explainCall(last('a = np.array([[3, 1], [0, 2]])\nnp.sort(a, axis=0)')).sources.flat()).toEqual([[0, 2], [0, 1], [0, 0], [0, 3]])
  })

  it('unique and trace', () => {
    expect(explainCall(last('a = np.array([2, 1, 2])\nnp.unique(a)')).sources).toEqual([[[0, 1]], [[0, 0], [0, 2]]])
    expect(explainCall(last('a = np.arange(9).reshape(3, 3)\nnp.trace(a)')).sources).toEqual([[[0, 0], [0, 4], [0, 8]]])
  })

  it('dependents inverts the mapping', () => {
    const e = explainCall(last('a = np.arange(4).reshape(2, 2)\na.sum(axis=1)'))
    expect(dependents(e, 1)[0]).toEqual([[0], [0], [1], [1]])
  })
})
