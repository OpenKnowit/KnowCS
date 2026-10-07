import { describe, expect, it } from 'vitest'
import {
  EXAM_TEST, EXAM_TRAIN, kNearest, pairwiseDiff, pairwiseDist, pairwiseDistExpansion, pairwiseDistLoop, pairwiseSqDist, shapeTrace, sumShape,
} from './pairwise'
import { runPython } from './minipy'

// 2022S 期中 Q2(c) 题面给出的输出
const EXAM_OUTPUT = [
  [7.07106781, 5.65685425, 4.24264069, 2.82842712],
  [9.89949494, 8.48528137, 7.07106781, 5.65685425],
]

const close = (a: number[][], b: number[][]) => a.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(b[i][j], 7)))

describe('pairwise distance', () => {
  it('与原题输出一致，形状为 (num_test, num_train)', () => {
    const D = pairwiseDist(EXAM_TEST, EXAM_TRAIN)
    expect(D.length).toBe(2)
    expect(D[0].length).toBe(4)
    close(D, EXAM_OUTPUT)
  })

  it('广播版 = 双重循环版 = 展开式版', () => {
    const test = [[1, -2, 3], [0, 0, 0], [2.5, 1, -1]]
    const train = [[0, 1, 1], [4, 4, 4], [-1, 0, 2], [1, -2, 3]]
    const D = pairwiseDist(test, train)
    close(pairwiseDistLoop(test, train), D)
    close(pairwiseDistExpansion(test, train), D)
  })

  it('中间结果 diff 的形状为 (n, m, d)，diff[i][j] = test[i] - train[j]', () => {
    const diff = pairwiseDiff(EXAM_TEST, EXAM_TRAIN)
    expect([diff.length, diff[0].length, diff[0][0].length]).toEqual([2, 4, 2])
    expect(diff[0][0]).toEqual([5, 5])
    expect(diff[1][3]).toEqual([4, 4])
    expect(pairwiseSqDist(EXAM_TEST, EXAM_TRAIN)[0]).toEqual([50, 32, 18, 8])
  })

  it('kNearest 等价于 np.argsort(D, axis=1)[:, :k]，平局按下标', () => {
    expect(kNearest(pairwiseDist(EXAM_TEST, EXAM_TRAIN), 2)).toEqual([[3, 2], [3, 2]])
    expect(kNearest([[2, 1, 1, 0]], 3)).toEqual([[3, 1, 2]])
  })

  it('形状追踪与错误轴', () => {
    expect(shapeTrace(2, 4, 2).map((s) => s.shape)).toEqual([[2, 1, 2], [1, 4, 2], [2, 4, 2], [2, 4, 2], [2, 4], [2, 4]])
    expect(sumShape([2, 4, 2], -1)).toEqual([2, 4])
    expect(sumShape([2, 4, 2], 1)).toEqual([2, 2])
    expect(sumShape([2, 4, 2], 0)).toEqual([4, 2])
  })
})

describe('站内解释器能跑通考试写法', () => {
  const setup = 'import numpy as np\nX_train = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])\nX_test = np.array([[5, 6], [7, 8]])\n'

  it('一行广播写法与 np.expand_dims / np.square 写法', () => {
    const r = runPython(`${setup}a = np.sqrt(((X_test[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2))
b = np.sqrt(np.sum(np.square(np.expand_dims(X_test, 1) - X_train), axis=2))
c = np.sqrt(np.sum(np.square(X_test[:, None] - X_train[None]), axis=-1))
print((X_test[:, None, :] - X_train[None, :, :]).shape, np.expand_dims(X_test, 1).shape)
print(np.all(a == b), np.all(a == c))
a`)
    expect(r.error).toBeNull()
    expect(r.stdout).toBe('(2, 4, 2) (2, 1, 2)\nTrue True\n')
    expect(r.out).toBe(`array([[7.07106781, 5.65685425, 4.24264069, 2.82842712],
       [9.89949494, 8.48528137, 7.07106781, 5.65685425]])`)
  })

  it('不加新轴直接相减会报广播错误', () => {
    const r = runPython(`${setup}X_test - X_train`)
    expect(r.error).toMatchObject({ type: 'ValueError' })
    expect(r.error!.message).toContain('(2,2) (4,2)')
  })

  it('np.argsort 与 ndarray.argsort：默认 axis=-1，axis=None 拉平', () => {
    const r = runPython(`import numpy as np
D = np.array([[3, 1, 2], [0, 5, 0]])
print(np.argsort(D, axis=1)[:, :2])
print(D.argsort(axis=0))
print(np.argsort(D, axis=None))
np.square(np.array([1, -2, 3]))`)
    expect(r.error).toBeNull()
    expect(r.stdout).toBe('[[1 2]\n [0 2]]\n[[1 0 1]\n [0 1 0]]\n[3 5 1 2 0 4]\n')
    expect(r.out).toBe('array([1, 4, 9])')
  })

  it('np.expand_dims 返回视图，负轴与越界轴', () => {
    const r = runPython(`import numpy as np
a = np.arange(6).reshape(2, 3)
e = np.expand_dims(a, -1)
e[0, 0, 0] = 99
print(e.shape, a[0, 0], np.expand_dims(a, 0).shape)
np.expand_dims(a, 3)`)
    expect(r.stdout).toBe('(2, 3, 1) 99 (1, 2, 3)\n')
    expect(r.error).toMatchObject({ type: 'AxisError' })
  })
})
