import { describe, expect, it } from 'vitest'
import { NDArray, PyError, arrayRepr, arrayStr, getIndex, planIndex, reshape, setIndex, sharesMemory, sliceIndices, transpose } from './ndarray'
import type { IndexItem } from './ndarray'

const arange = (n: number) => NDArray.create(Array.from({ length: n }, (_, i) => i), [n], 'int64')
const grid = () => reshape(arange(16), [4, 4])
const S = (start: number | null, stop: number | null, step: number | null = null): IndexItem => ({ kind: 'slice', start, stop, step })
const I = (value: number): IndexItem => ({ kind: 'int', value })
const A = (vals: number[], dtype: 'int64' | 'bool' = 'int64', shape = [vals.length]): IndexItem => ({ kind: 'array', arr: NDArray.create(vals, shape, dtype) })

describe('sliceIndices（Python slice.indices 语义）', () => {
  it('正向 / 负步长 / 越界截断', () => {
    expect(sliceIndices(1, 3, null, 4)).toEqual({ start: 1, step: 1, len: 2 })
    expect(sliceIndices(null, null, -1, 4)).toEqual({ start: 3, step: -1, len: 4 })
    expect(sliceIndices(-1, -5, -2, 4)).toEqual({ start: 3, step: -2, len: 2 })
    expect(sliceIndices(10, null, null, 5).len).toBe(0)
  })
  it('step 为 0 报 ValueError', () => {
    expect(() => sliceIndices(null, null, 0, 4)).toThrow('slice step cannot be zero')
  })
})

describe('基本索引 → 视图', () => {
  it('a[1:3, ::2] 共享缓冲区，offset/strides 正确', () => {
    const a = grid()
    const { plan, value } = getIndex(a, [S(1, 3), S(null, null, 2)])
    const v = value as NDArray
    expect(plan.advanced).toBe(false)
    expect(plan.view).toEqual({ shape: [2, 2], strides: [4, 2], offset: 4 })
    expect(v.values()).toEqual([4, 6, 8, 10])
    expect(v.data).toBe(a.data)
    expect(sharesMemory(a, v)).toBe(true)
  })

  it('全部整数索引返回标量', () => {
    const { plan, value } = getIndex(grid(), [I(1), I(-1)])
    expect(plan.scalar).toBe(true)
    expect(value).toBe(7)
  })

  it('None / Ellipsis 改变形状但不复制', () => {
    const a = reshape(arange(24), [2, 3, 4])
    expect(planIndex(a, [{ kind: 'ellipsis' }, I(1)]).shape).toEqual([2, 3])
    expect(planIndex(a, [I(1), { kind: 'ellipsis' }, { kind: 'newaxis' }]).shape).toEqual([3, 4, 1])
  })

  it('越界与维度过多给出 numpy 同款报错', () => {
    expect(() => planIndex(grid(), [I(4)])).toThrow('index 4 is out of bounds for axis 0 with size 4')
    expect(() => planIndex(grid(), [I(0), I(0), I(0)])).toThrow('too many indices for array: array is 2-dimensional, but 3 were indexed')
    expect(() => planIndex(grid(), [{ kind: 'ellipsis' }, { kind: 'ellipsis' }])).toThrow(PyError)
  })
})

describe('高级索引 → 复制', () => {
  it('a[[0, 2]] 取整行，新缓冲区', () => {
    const a = grid()
    const { plan, value } = getIndex(a, [A([0, 2])])
    expect(plan.advanced).toBe(true)
    expect((value as NDArray).values()).toEqual([0, 1, 2, 3, 8, 9, 10, 11])
    expect((value as NDArray).data).not.toBe(a.data)
  })

  it('a[[0, 2, 3], [1, 3, 0]] 成对取点', () => {
    expect(planIndex(grid(), [A([0, 2, 3]), A([1, 3, 0])]).addresses).toEqual([1, 11, 12])
  })

  it('高级索引被切片隔开 → 广播维度移到最前（a[0, :, [1, 3]] 形状为 (2, 3)）', () => {
    const a = reshape(arange(24), [2, 3, 4])
    const plan = planIndex(a, [I(0), S(null, null), A([1, 3])])
    expect(plan.shape).toEqual([2, 3])
    expect(plan.advFront).toBe(true)
    expect(plan.addresses).toEqual([1, 5, 9, 3, 7, 11])
  })

  it('相邻的高级索引原位替换（a[:, [0, 2], 1:3] 形状为 (2, 2, 2)）', () => {
    const a = reshape(arange(24), [2, 3, 4])
    const plan = planIndex(a, [S(null, null), A([0, 2]), S(1, 3)])
    expect(plan.shape).toEqual([2, 2, 2])
    expect(plan.addresses).toEqual([1, 2, 9, 10, 13, 14, 21, 22])
  })

  it('布尔掩码按行优先收集 True 的位置', () => {
    const a = grid()
    const mask = A(a.values().map((v) => (v % 5 === 0 ? 1 : 0)), 'bool', [4, 4])
    const plan = planIndex(a, [mask])
    expect(plan.shape).toEqual([4])
    expect(plan.addresses).toEqual([0, 5, 10, 15])
    expect(plan.mask?.axes).toEqual([0, 1])
  })

  it('掩码形状不匹配 / 索引数组无法广播', () => {
    expect(() => planIndex(grid(), [A([1, 0, 1], 'bool')])).toThrow('size of axis is 4 but size of corresponding boolean axis is 3')
    expect(() => planIndex(grid(), [A([0, 1]), A([1, 2, 3])])).toThrow('shape mismatch')
  })
})

describe('写入与视图', () => {
  it('通过视图写入会改到原数组', () => {
    const a = grid()
    const v = getIndex(a, [S(1, 3), S(1, 3)]).value as NDArray
    setIndex(v, [I(0), I(0)], NDArray.create([99], [], 'int64'))
    expect(a.values()[5]).toBe(99)
  })

  it('掩码赋值写回原数组并按 dtype 截断', () => {
    const a = grid()
    setIndex(a, [A(a.values().map((x) => (x < 3 ? 1 : 0)), 'bool', [4, 4])], NDArray.create([2.7], [], 'float64'))
    expect(a.values().slice(0, 4)).toEqual([2, 2, 2, 3])
  })

  it('形状不兼容报错', () => {
    expect(() => setIndex(grid(), [S(0, 2)], arange(3))).toThrow('could not broadcast input array from shape (3,) into shape (2,4)')
  })

  it('转置是视图，转置后 reshape 需要复制', () => {
    const a = grid()
    const t = transpose(a)
    expect(t.data).toBe(a.data)
    expect(t.isCContiguous()).toBe(false)
    expect(reshape(t, [16]).data).not.toBe(a.data)
  })
})

describe('repr / str 与 numpy 一致', () => {
  it('整数、布尔、浮点对齐', () => {
    expect(arrayRepr(reshape(arange(6), [2, 3]))).toBe('array([[0, 1, 2],\n       [3, 4, 5]])')
    expect(arrayStr(reshape(arange(6), [2, 3]))).toBe('[[0 1 2]\n [3 4 5]]')
    expect(arrayRepr(NDArray.create([1, 0, 1], [3], 'bool'))).toBe('array([ True, False,  True])')
    expect(arrayRepr(NDArray.create([0, 0.25, 0.5, 1], [4], 'float64'))).toBe('array([0.  , 0.25, 0.5 , 1.  ])')
    expect(arrayRepr(NDArray.create([1e-5, 1], [2], 'float64'))).toBe('array([1.e-05, 1.e+00])')
  })
  it('长行在 75 列换行，空数组带 dtype', () => {
    expect(arrayRepr(arange(20))).toBe(
      'array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n       17, 18, 19])',
    )
    expect(arrayRepr(NDArray.create([], [0], 'int64'))).toBe('array([], dtype=int64)')
  })
})
