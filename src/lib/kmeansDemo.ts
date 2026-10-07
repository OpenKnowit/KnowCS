/** Data and helpers for the K-Means explainer (/watch/kmeans/). Clustering itself is kmeansTable. */
import { mulberry32 } from './crossval'
import { kmeansTable } from './kmeansTable'

/** Three round blobs of 12 points each in [0, 10]², reproducible. */
export function makeBlobs(seed = 7): number[][] {
  const r = mulberry32(seed)
  const centres = [
    [2.6, 7.2],
    [7.4, 7.6],
    [5.2, 2.6],
  ]
  const gauss = () => {
    // Box–Muller
    const u = Math.max(1e-9, r())
    const v = r()
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  }
  return centres.flatMap(([cx, cy]) => Array.from({ length: 12 }, () => [Number((cx + gauss() * 0.85).toFixed(2)), Number((cy + gauss() * 0.85).toFixed(2))]))
}

/** Lowest final SSE over `tries` runs started from random data points. */
export function bestSse(points: number[][], k: number, tries = 20, seed = 1): number {
  const r = mulberry32(seed)
  let best = Infinity
  for (let t = 0; t < tries; t++) {
    const idx = new Set<number>()
    while (idx.size < k) idx.add(Math.floor(r() * points.length))
    const res = kmeansTable(points, [...idx].map((i) => points[i]), 'euclidean', 50)
    best = Math.min(best, res.rounds[res.rounds.length - 1].sse)
  }
  return best
}
