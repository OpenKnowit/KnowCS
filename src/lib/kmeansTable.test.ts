import { describe, expect, it } from 'vitest'
import { kmeansTable } from './kmeansTable'

const col = (xs: number[]) => xs.map((x) => [x])

describe('kmeansTable', () => {
  it('reproduces the 2022 Fall 1-D question (seeds A, B)', () => {
    const r = kmeansTable(col([0.2, 0.5, 0.7, 2.5, 3.5]), [[0.2], [0.5]])
    expect(r.rounds[0].assign).toEqual([0, 1, 1, 1, 1])
    expect(r.rounds[0].next.map((c) => c[0])).toEqual([0.2, 1.8])
    expect(r.rounds[1].assign).toEqual([0, 0, 0, 1, 1])
    expect(r.rounds[1].next[0][0]).toBeCloseTo(0.4667, 4)
    expect(r.rounds[1].next[1][0]).toBe(3)
    expect(r.converged).toBe(true)
  })

  it('reproduces the 2022 Spring 1-D question (c1 = 3, c2 = 4)', () => {
    const r = kmeansTable(col([0, 3, 6, 9, 27, 30]), [[3], [4]])
    expect(r.rounds[0].dist.map((d) => d[0])).toEqual([3, 0, 3, 6, 24, 27])
    expect(r.rounds[0].next).toEqual([[1.5], [18]])
    expect(r.rounds[1].next).toEqual([[4.5], [28.5]])
  })

  it('reproduces the 2023 Spring 2-D question and converges in round 2', () => {
    const pts = [[1, 4], [2, 3], [4, 6], [5, 7], [8, 3]]
    const r = kmeansTable(pts, [[1, 4], [5, 7]])
    expect(r.rounds[0].assign).toEqual([0, 0, 1, 1, 1])
    expect(r.rounds[0].next[1][0]).toBeCloseTo(5.67, 2)
    expect(r.rounds[0].next[1][1]).toBeCloseTo(5.33, 2)
    expect(r.rounds[1].dist[2][1]).toBeCloseTo(1.8, 2)
    expect(r.rounds).toHaveLength(2)
    expect(r.converged).toBe(true)
  })

  it('reports squared distances and SSE like 2023 Fall (SSE = 8)', () => {
    const pts = [[1, 2], [2, 3], [7, 1], [8, 2], [3, 1], [9, 3]]
    const r = kmeansTable(pts, [[0, 0], [8, 0]], 'squared')
    expect(r.rounds[0].dist.map((d) => d[0])).toEqual([5, 13, 50, 68, 10, 90])
    expect(r.rounds[0].next).toEqual([[2, 2], [8, 2]])
    expect(r.rounds[0].sse).toBe(8)
  })

  it('solves the lecture 4 medicine practice problem', () => {
    const r = kmeansTable([[1, 1], [2, 1], [4, 3], [5, 4]], [[1, 1], [2, 1]])
    const last = r.rounds[r.rounds.length - 1]
    expect(last.next).toEqual([[1.5, 1], [4.5, 3.5]])
    expect(r.rounds).toHaveLength(3)
  })

  it('breaks ties toward centroid 1', () => {
    expect(kmeansTable([[2]], [[1], [3]]).rounds[0].assign).toEqual([0])
  })
})
