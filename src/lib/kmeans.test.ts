import { describe, expect, it } from 'vitest'
import { assign, elbowCurve, inertia, recompute, runKMeans, standardize } from './kmeans'
import { KMEANS_DATA, KMEANS_INIT_CENTROIDS } from '../data/constants'
import type { Vec2 } from '../types'

describe('kmeans 基础算子', () => {
  it('assign 把点分到最近质心', () => {
    const c: Vec2[] = [{ x: 0, y: 0 }, { x: 10, y: 10 }]
    expect(assign([{ x: 1, y: 1 }, { x: 9, y: 8 }], c)).toEqual([0, 1])
  })

  it('recompute 求簇内均值', () => {
    const data: Vec2[] = [{ x: 0, y: 0 }, { x: 2, y: 2 }, { x: 10, y: 10 }]
    const out = recompute(data, [0, 0, 1], [{ x: 0, y: 0 }, { x: 10, y: 10 }])
    expect(out[0]).toEqual({ x: 1, y: 1 })
    expect(out[1]).toEqual({ x: 10, y: 10 })
  })

  it('空簇保持原质心不变', () => {
    const data: Vec2[] = [{ x: 0, y: 0 }]
    const out = recompute(data, [0], [{ x: 0, y: 0 }, { x: 5, y: 5 }])
    expect(out[1]).toEqual({ x: 5, y: 5 })
  })
})

describe('runKMeans 收敛性', () => {
  it('inertia 随迭代单调不增', () => {
    const hist = runKMeans(KMEANS_DATA, KMEANS_INIT_CENTROIDS)
    for (let i = 1; i < hist.length; i++) {
      expect(hist[i].inertia).toBeLessThanOrEqual(hist[i - 1].inertia + 1e-9)
    }
  })

  it('收敛后质心不再移动（moved≈0）', () => {
    const hist = runKMeans(KMEANS_DATA, KMEANS_INIT_CENTROIDS)
    expect(hist.at(-1)!.moved).toBeLessThan(1e-6)
  })

  it('三簇数据最终把 18 个点分成 3 个非空簇', () => {
    const hist = runKMeans(KMEANS_DATA, KMEANS_INIT_CENTROIDS)
    const final = hist.at(-1)!
    const sizes = [0, 1, 2].map((k) => final.assignments.filter((a) => a === k).length)
    expect(sizes.every((s) => s > 0)).toBe(true)
    expect(sizes.reduce((a, b) => a + b, 0)).toBe(18)
  })
})

describe('elbow 与标准化', () => {
  it('elbowCurve 随 K 增大 inertia 递减', () => {
    const curve = elbowCurve(KMEANS_DATA, [1, 2, 3, 4])
    for (let i = 1; i < curve.length; i++) {
      expect(curve[i].inertia).toBeLessThanOrEqual(curve[i - 1].inertia + 1e-9)
    }
  })

  it('standardize 后各维均值≈0、标准差≈1', () => {
    const { data } = standardize(KMEANS_DATA)
    const n = data.length
    const mx = data.reduce((s, p) => s + p.x, 0) / n
    const sx = Math.sqrt(data.reduce((s, p) => s + (p.x - mx) ** 2, 0) / n)
    expect(Math.abs(mx)).toBeLessThan(1e-9)
    expect(Math.abs(sx - 1)).toBeLessThan(1e-9)
  })

  it('inertia 非负', () => {
    expect(inertia(KMEANS_DATA, assign(KMEANS_DATA, KMEANS_INIT_CENTROIDS), KMEANS_INIT_CENTROIDS)).toBeGreaterThan(0)
  })
})
