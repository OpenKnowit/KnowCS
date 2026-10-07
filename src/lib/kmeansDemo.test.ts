import { describe, expect, it } from 'vitest'
import { bestSse, makeBlobs } from './kmeansDemo'
import { kmeansTable } from './kmeansTable'

describe('k-means explainer data', () => {
  const pts = makeBlobs()
  it('is reproducible and inside the plot', () => {
    expect(makeBlobs()).toEqual(pts)
    expect(pts).toHaveLength(36)
    for (const [x, y] of pts) {
      expect(x).toBeGreaterThan(0)
      expect(x).toBeLessThan(10)
      expect(y).toBeGreaterThan(0)
      expect(y).toBeLessThan(10)
    }
  })
  it('has its elbow at k = 3', () => {
    const sse = [1, 2, 3, 4, 5].map((k) => bestSse(pts, k))
    for (let i = 1; i < sse.length; i++) expect(sse[i]).toBeLessThanOrEqual(sse[i - 1])
    const drop = (k: number) => sse[k - 2] - sse[k - 1]
    expect(drop(3)).toBeGreaterThan(3 * drop(4))
  })
  it('SSE never rises from one round to the next', () => {
    const res = kmeansTable(pts, [pts[0], pts[1], pts[2]], 'euclidean', 50)
    for (let i = 1; i < res.rounds.length; i++) expect(res.rounds[i].sse).toBeLessThanOrEqual(res.rounds[i - 1].sse + 1e-9)
    expect(res.converged).toBe(true)
  })
})
