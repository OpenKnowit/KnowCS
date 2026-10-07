/**
 * K-Means exactly as the exam tables run it (lecture 4): fixed initial centroids, a distance table per
 * round, ties go to the lower-numbered centroid, stop when no point changes cluster.
 * Works in any dimension (the midterms use 1-D and 2-D data).
 */

export type KMetric = 'euclidean' | 'squared' | 'manhattan'

export const distance = (a: number[], b: number[], metric: KMetric): number => {
  if (metric === 'manhattan') return a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0)
  const sq = a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0)
  return metric === 'squared' ? sq : Math.sqrt(sq)
}

export interface KMeansRound {
  round: number
  /** centroids used to measure distances in this round */
  centroids: number[][]
  /** dist[point][cluster] */
  dist: number[][]
  assign: number[]
  /** means of the clusters formed this round (an empty cluster keeps its centroid) */
  next: number[][]
  /** Σ squared Euclidean distance of each point to its new centroid */
  sse: number
  /** points whose cluster differs from the previous round (all points in round 1) */
  moved: number[]
}

export interface KMeansTableResult {
  rounds: KMeansRound[]
  converged: boolean
  emptyClusters: boolean
}

export function kmeansTable(points: number[][], init: number[][], metric: KMetric = 'euclidean', maxRounds = 12): KMeansTableResult {
  const rounds: KMeansRound[] = []
  let centroids = init.map((c) => [...c])
  let prev: number[] | null = null
  let emptyClusters = false
  for (let r = 1; r <= maxRounds; r++) {
    const dist = points.map((p) => centroids.map((c) => distance(p, c, metric)))
    // strict < keeps the first (lowest-numbered) centroid on ties
    const assign = dist.map((row) => row.reduce((best, d, k) => (d < row[best] - 1e-12 ? k : best), 0))
    const next = centroids.map((c, k) => {
      const mine = points.filter((_, i) => assign[i] === k)
      if (!mine.length) {
        emptyClusters = true
        return [...c]
      }
      return c.map((_, j) => mine.reduce((s, p) => s + p[j], 0) / mine.length)
    })
    const sse = points.reduce((s, p, i) => s + distance(p, next[assign[i]], 'squared'), 0)
    const moved = prev ? assign.flatMap((a, i) => (a !== prev![i] ? [i] : [])) : assign.map((_, i) => i)
    rounds.push({ round: r, centroids, dist, assign, next, sse, moved })
    if (prev && moved.length === 0) return { rounds, converged: true, emptyClusters }
    prev = assign
    centroids = next
  }
  return { rounds, converged: false, emptyClusters }
}
