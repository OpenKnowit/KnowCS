// --- 成对距离：一行 NumPy（X_test[:, None, :] - X_train[None, :, :]）背后的每一步，供讲解视频逐帧取值 ---

export type Points = readonly (readonly number[])[]

/** 2022S 期中 Q2(c) 的原题数据：4 个训练点、2 个测试点，输出形状 (num_test, num_train) = (2, 4) */
export const EXAM_TRAIN: Points = [[0, 1], [1, 2], [2, 3], [3, 4]]
export const EXAM_TEST: Points = [[5, 6], [7, 8]]

/** diff[i][j][k] = test[i][k] - train[j][k]，形状 (n, m, d)：广播后的中间结果 */
export const pairwiseDiff = (test: Points, train: Points): number[][][] =>
  test.map((x) => train.map((y) => x.map((v, k) => v - y[k])))

/** 平方后沿最后一轴（坐标轴，axis=-1）求和，形状 (n, m) */
export const pairwiseSqDist = (test: Points, train: Points): number[][] =>
  pairwiseDiff(test, train).map((row) => row.map((v) => v.reduce((s, x) => s + x * x, 0)))

export const pairwiseDist = (test: Points, train: Points): number[][] =>
  pairwiseSqDist(test, train).map((row) => row.map(Math.sqrt))

/** 原题给出的双重循环版本，作为对照 */
export const pairwiseDistLoop = (test: Points, train: Points): number[][] => {
  const out: number[][] = []
  for (let i = 0; i < test.length; i++) {
    out.push([])
    for (let j = 0; j < train.length; j++) {
      let s = 0
      for (let k = 0; k < test[i].length; k++) s += (test[i][k] - train[j][k]) ** 2
      out[i].push(Math.sqrt(s))
    }
  }
  return out
}

/** 官方答案的展开式 ‖x‖² + ‖y‖² − 2x·y：不生成 (n, m, d) 中间数组 */
export const pairwiseDistExpansion = (test: Points, train: Points): number[][] => {
  const sq = (p: readonly number[]) => p.reduce((s, x) => s + x * x, 0)
  return test.map((x) => train.map((y) => Math.sqrt(Math.max(0, sq(x) + sq(y) - 2 * x.reduce((s, v, k) => s + v * y[k], 0)))))
}

/** np.argsort(D, axis=1)[:, :k]：每个测试点最近的 k 个训练点下标（距离相同按下标稳定排序） */
export const kNearest = (dist: readonly (readonly number[])[], k: number): number[][] =>
  dist.map((row) => row.map((_, j) => j).sort((a, b) => row[a] - row[b] || a - b).slice(0, k))

/** 沿错误的轴求和得到的形状：用来说明 axis 选错时形状如何「看起来能跑但意义全错」 */
export const sumShape = (shape: readonly number[], axis: number): number[] => {
  const ax = axis < 0 ? axis + shape.length : axis
  return shape.filter((_, i) => i !== ax)
}

export interface ShapeStep {
  /** 代码片段的键，用于 i18n 讲解 */
  id: 'test' | 'train' | 'diff' | 'square' | 'sum' | 'sqrt'
  code: string
  shape: number[]
}

/** 一行代码逐段的形状追踪：n 个测试点、m 个训练点、d 维 */
export const shapeTrace = (n: number, m: number, d: number): ShapeStep[] => [
  { id: 'test', code: 'X_test[:, None, :]', shape: [n, 1, d] },
  { id: 'train', code: 'X_train[None, :, :]', shape: [1, m, d] },
  { id: 'diff', code: 'X_test[:, None, :] - X_train[None, :, :]', shape: [n, m, d] },
  { id: 'square', code: '(…) ** 2', shape: [n, m, d] },
  { id: 'sum', code: '.sum(axis=-1)', shape: [n, m] },
  { id: 'sqrt', code: 'np.sqrt(…)', shape: [n, m] },
]
