import type { KMeansIteration, Vec2 } from '../types'

export const dist2 = (a: Vec2, b: Vec2): number => (a.x - b.x) ** 2 + (a.y - b.y) ** 2

/** 把每个点分配到最近的质心，返回簇索引数组。 */
export function assign(data: Vec2[], centroids: Vec2[]): number[] {
  return data.map((p) => {
    let best = 0
    let bestD = Infinity
    centroids.forEach((c, i) => {
      const d = dist2(p, c)
      if (d < bestD) {
        bestD = d
        best = i
      }
    })
    return best
  })
}

/** 依据分配重新计算质心；空簇保持原位。 */
export function recompute(data: Vec2[], assignments: number[], centroids: Vec2[]): Vec2[] {
  return centroids.map((c, k) => {
    const members = data.filter((_, i) => assignments[i] === k)
    if (members.length === 0) return c
    const sx = members.reduce((s, p) => s + p.x, 0)
    const sy = members.reduce((s, p) => s + p.y, 0)
    return { x: sx / members.length, y: sy / members.length }
  })
}

/** 簇内平方和（WCSS / inertia）。 */
export function inertia(data: Vec2[], assignments: number[], centroids: Vec2[]): number {
  return data.reduce((s, p, i) => s + dist2(p, centroids[assignments[i]]), 0)
}

/**
 * 跑完整 K-Means（Lloyd / EM 迭代），返回每一轮的快照，便于动画逐步播放。
 * 第 0 项为初始状态（用初始质心做一次分配）。
 */
export function runKMeans(
  data: Vec2[],
  initCentroids: Vec2[],
  maxIter = 20,
  tol = 1e-6
): KMeansIteration[] {
  let centroids = initCentroids.map((c) => ({ ...c }))
  const history: KMeansIteration[] = []

  let assignments = assign(data, centroids)
  history.push({ centroids, assignments, inertia: inertia(data, assignments, centroids), moved: Infinity })

  for (let iter = 0; iter < maxIter; iter++) {
    const newCentroids = recompute(data, assignments, centroids)
    const moved = newCentroids.reduce((s, c, k) => s + Math.sqrt(dist2(c, centroids[k])), 0)
    centroids = newCentroids
    assignments = assign(data, centroids)
    history.push({ centroids, assignments, inertia: inertia(data, assignments, centroids), moved })
    if (moved < tol) break
  }
  return history
}

/**
 * 最远优先（farthest-first）确定性初始化：先取首点，
 * 之后每次选「到已选质心集合最小距离最大」的点。
 * 比「取前 k 个点」更分散，能避免落入退化的局部最优，
 * 从而让 Elbow 曲线随 K 增大单调下降。
 */
export function farthestFirstInit(data: Vec2[], k: number): Vec2[] {
  const chosen: Vec2[] = [{ ...data[0] }]
  while (chosen.length < k) {
    let bestIdx = 0
    let bestD = -1
    data.forEach((p, i) => {
      const minD = Math.min(...chosen.map((c) => dist2(p, c)))
      if (minD > bestD) {
        bestD = minD
        bestIdx = i
      }
    })
    chosen.push({ ...data[bestIdx] })
  }
  return chosen
}

/** 对一系列 K 值跑 K-Means，返回最终 inertia，用于 Elbow Method 选 K。 */
export function elbowCurve(data: Vec2[], ks: number[]): { k: number; inertia: number }[] {
  return ks.map((k) => {
    const init = farthestFirstInit(data, k)
    const hist = runKMeans(data, init)
    return { k, inertia: hist.at(-1)!.inertia }
  })
}

/** Z-score 标准化（按维度），返回标准化后的数据与统计量。 */
export function standardize(data: Vec2[]): { data: Vec2[]; mean: Vec2; std: Vec2 } {
  const n = data.length
  const mean: Vec2 = {
    x: data.reduce((s, p) => s + p.x, 0) / n,
    y: data.reduce((s, p) => s + p.y, 0) / n,
  }
  const std: Vec2 = {
    x: Math.sqrt(data.reduce((s, p) => s + (p.x - mean.x) ** 2, 0) / n) || 1,
    y: Math.sqrt(data.reduce((s, p) => s + (p.y - mean.y) ** 2, 0) / n) || 1,
  }
  return {
    data: data.map((p) => ({ x: (p.x - mean.x) / std.x, y: (p.y - mean.y) / std.y })),
    mean,
    std,
  }
}
