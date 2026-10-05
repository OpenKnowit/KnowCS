import type { KnnPoint, KnnStats, StudentClass, TestPoint } from '../types'

// --- KNN 纯计算逻辑（KnnModule） ---

export interface KnnPointWithDist extends KnnPoint {
  idx: number // 在原始数据中的下标（同坐标的点也能区分）
  dist: number
}

export interface KnnResult {
  data: KnnPointWithDist[]
  neighborsMap: Map<number, number> // 数据下标 → 名次（1..K）
  radiusDist: number
  prediction: StudentClass
  isTie: boolean // 票数相同：由最近邻的类别裁决
  mCount: number
  lCount: number
  topK: KnnPointWithDist[]
}

/** 线性映射：把 [min, max] 区间的值缩放到 [0, target] */
export const scaleLinear = (val: number, min: number, max: number, target: number): number =>
  ((val - min) / (max - min)) * target

/** 两点距离：原始欧氏距离，或先按 Z-score 标准化再算欧氏距离 */
const distance = (a: TestPoint, b: TestPoint, isStandardized: boolean, stats: KnnStats): number => {
  const dh = isStandardized ? (a.h - b.h) / stats.stdH : a.h - b.h
  const dw = isStandardized ? (a.w - b.w) / stats.stdW : a.w - b.w
  return Math.hypot(dh, dw)
}

/**
 * 计算每个样本到测试点的距离（可选 Z-score 标准化），排序取前 K 并多数投票。
 * 平票（K 为偶数时可能出现）不再默认偏向某一类，而是交给最近的邻居裁决，并通过 isTie 标出。
 */
export const computeKnn = (
  rawData: KnnPoint[],
  testPoint: TestPoint,
  k: number,
  isStandardized: boolean,
  stats: KnnStats
): KnnResult => {
  const dataWithDist = rawData.map((d, idx) => ({ ...d, idx, dist: distance(d, testPoint, isStandardized, stats) }))

  // 稳定排序：等距时保持原始顺序，结果可复现
  const sorted = [...dataWithDist].sort((a, b) => a.dist - b.dist)
  const topK = sorted.slice(0, k)
  const neighborsMap = new Map(topK.map((n, rank) => [n.idx, rank + 1]))

  // 决策圆半径（到第 K 个邻居的距离）
  const radiusDist = topK.length > 0 ? topK[topK.length - 1].dist : 0

  const mCount = topK.filter((n) => n.s === 'M').length
  const lCount = topK.length - mCount
  const isTie = topK.length > 0 && mCount === lCount
  const prediction: StudentClass = isTie ? topK[0].s : mCount > lCount ? 'M' : 'L'
  return { data: dataWithDist, neighborsMap, radiusDist, prediction, isTie, mCount, lCount, topK }
}

/**
 * 留一法（LOO）交叉验证误差曲线：对每个 K，依次把每个样本拿出来当测试点、
 * 用其余样本做 KNN 预测，统计预测错误的比例。这是真实计算结果，不是示意曲线。
 */
export const looErrorCurve = (
  data: KnnPoint[],
  ks: number[],
  isStandardized: boolean,
  stats: KnnStats
): { k: number; error: number }[] =>
  ks.map((k) => {
    const wrong = data.filter((p, i) => {
      const rest = data.filter((_, j) => j !== i)
      return computeKnn(rest, p, k, isStandardized, stats).prediction !== p.s
    }).length
    return { k, error: data.length ? wrong / data.length : 0 }
  })

/** 把 SVG 画布点击坐标映射回数据空间（身高/体重） */
export const svgToDataPoint = (
  x: number,
  y: number,
  width: number,
  height: number,
  stats: KnnStats
): TestPoint => {
  const h = (x / width) * (stats.maxH - stats.minH) + stats.minH
  const w = stats.maxW - (y / height) * (stats.maxW - stats.minW)
  return { h: Math.round(h), w: Math.round(w) }
}

/** 键盘移动测试点：方向键每次 1 个单位，限制在画布数据范围内 */
export const nudgeTestPoint = (p: TestPoint, dh: number, dw: number, stats: KnnStats): TestPoint => ({
  h: Math.min(stats.maxH, Math.max(stats.minH, p.h + dh)),
  w: Math.min(stats.maxW, Math.max(stats.minW, p.w + dw)),
})
