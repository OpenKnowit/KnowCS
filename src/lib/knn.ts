import type { KnnPoint, KnnStats, StudentClass, TestPoint } from '../types'

// --- KNN 纯计算逻辑（从 KnnModule 提取，行为保持不变） ---

export interface KnnPointWithDist extends KnnPoint {
  dist: number
}

export interface KnnResult {
  data: KnnPointWithDist[]
  neighborsMap: Map<string, number>
  radiusDist: number
  prediction: StudentClass
  mCount: number
  lCount: number
  topK: KnnPointWithDist[]
}

/** 线性映射：把 [min, max] 区间的值缩放到 [0, target] */
export const scaleLinear = (val: number, min: number, max: number, target: number): number =>
  ((val - min) / (max - min)) * target

/** 计算每个样本到测试点的距离（可选 Z-score 标准化），排序取前 K 并投票 */
export const computeKnn = (
  rawData: KnnPoint[],
  testPoint: TestPoint,
  k: number,
  isStandardized: boolean,
  stats: KnnStats
): KnnResult => {
  const dataWithDist = rawData.map((d) => {
    let dist: number
    if (isStandardized) {
      const sh1 = (d.h - stats.meanH) / stats.stdH
      const sw1 = (d.w - stats.meanW) / stats.stdW
      const sh2 = (testPoint.h - stats.meanH) / stats.stdH
      const sw2 = (testPoint.w - stats.meanW) / stats.stdW
      dist = Math.sqrt(Math.pow(sh1 - sh2, 2) + Math.pow(sw1 - sw2, 2))
    } else {
      dist = Math.sqrt(Math.pow(d.h - testPoint.h, 2) + Math.pow(d.w - testPoint.w, 2))
    }
    return { ...d, dist }
  })

  const sorted = [...dataWithDist].sort((a, b) => a.dist - b.dist)
  const topK = sorted.slice(0, k)
  const neighborsMap = new Map(topK.map((n, idx) => [`${n.h}-${n.w}`, idx + 1]))

  // 决策圆半径（到第 K 个邻居的距离）
  const radiusDist = topK.length > 0 ? topK[topK.length - 1].dist : 0

  const mCount = topK.filter((n) => n.s === 'M').length
  const lCount = topK.filter((n) => n.s === 'L').length
  const prediction: StudentClass = mCount >= lCount ? 'M' : 'L'
  return { data: dataWithDist, neighborsMap, radiusDist, prediction, mCount, lCount, topK }
}

/** 合成「模型复杂度 vs 错误率」曲线（确定性 U 形 + 伪随机抖动） */
export const complexityCurve = (length = 15): { k: number; error: number }[] =>
  Array.from({ length }, (_, i) => ({
    k: i + 1,
    error: 0.1 + Math.pow(i + 1 - 4, 2) * 0.01 + ((Math.sin((i + 1) * 99.71) + 1) / 2) * 0.02,
  }))

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
