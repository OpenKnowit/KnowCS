import { describe, expect, it } from 'vitest'
import { KNN_RAW_DATA } from '../data/constants'
import type { KnnStats } from '../types'
import { computeKnn, looErrorCurve, nudgeTestPoint, scaleLinear, svgToDataPoint } from './knn'

const STATS: KnnStats = { meanH: 164, meanW: 62.33, stdH: 4.33, stdW: 2.63, minH: 155, maxH: 175, minW: 55, maxW: 70 }
const TEST_POINT = { h: 161, w: 61 } // 模块默认测试点

describe('scaleLinear', () => {
  it('线性映射区间端点与中点', () => {
    expect(scaleLinear(155, 155, 175, 400)).toBe(0)
    expect(scaleLinear(175, 155, 175, 400)).toBe(400)
    expect(scaleLinear(165, 155, 175, 400)).toBe(200)
  })
})

describe('computeKnn', () => {
  it('黄金值：默认测试点 K=5 原始数据 → M 4 票 : L 1 票，预测 M（对应 UI 投票面板）', () => {
    const r = computeKnn(KNN_RAW_DATA, TEST_POINT, 5, false, STATS)
    expect(r.mCount).toBe(4)
    expect(r.lCount).toBe(1)
    expect(r.prediction).toBe('M')
  })

  it('最近邻是 (160, 60)，距离 √2；决策圆半径为第 K 个邻居距离', () => {
    const r = computeKnn(KNN_RAW_DATA, TEST_POINT, 5, false, STATS)
    expect(r.topK[0]).toMatchObject({ h: 160, w: 60 })
    expect(r.topK[0].dist).toBeCloseTo(Math.SQRT2, 10)
    expect(r.radiusDist).toBeCloseTo(r.topK[4].dist, 10)
  })

  it('neighborsMap 以数据下标为键，按距离名次编号 1..K', () => {
    const r = computeKnn(KNN_RAW_DATA, TEST_POINT, 3, false, STATS)
    expect(r.neighborsMap.size).toBe(3)
    const nearestIdx = KNN_RAW_DATA.findIndex((p) => p.h === 160 && p.w === 60)
    expect(r.neighborsMap.get(nearestIdx)).toBe(1)
  })

  it('坐标完全相同的两个样本各自有独立名次（不会因键冲突丢失）', () => {
    const dup = [{ h: 160, w: 60, s: 'M' as const }, { h: 160, w: 60, s: 'L' as const }]
    const r = computeKnn(dup, TEST_POINT, 2, false, STATS)
    expect(r.neighborsMap.get(0)).toBe(1)
    expect(r.neighborsMap.get(1)).toBe(2)
  })

  it('K 大于样本数时取全部样本', () => {
    const r = computeKnn(KNN_RAW_DATA, TEST_POINT, 99, false, STATS)
    expect(r.topK).toHaveLength(KNN_RAW_DATA.length)
    expect(r.mCount + r.lCount).toBe(KNN_RAW_DATA.length)
  })

  it('平票时由最近邻裁决，并标记 isTie（不再默认偏向 M）', () => {
    // 最近邻为 L（距离 1），次近邻为 M（距离 2）→ 1:1 平票，预测 L
    const nearestL = computeKnn(
      [{ h: 163, w: 61, s: 'M' }, { h: 160, w: 61, s: 'L' }],
      { h: 161, w: 61 }, 2, false, STATS
    )
    expect(nearestL.mCount).toBe(1)
    expect(nearestL.lCount).toBe(1)
    expect(nearestL.isTie).toBe(true)
    expect(nearestL.prediction).toBe('L')
    // 交换类别后结论随之翻转
    const nearestM = computeKnn(
      [{ h: 163, w: 61, s: 'L' }, { h: 160, w: 61, s: 'M' }],
      { h: 161, w: 61 }, 2, false, STATS
    )
    expect(nearestM.prediction).toBe('M')
  })

  it('非平票时 isTie 为 false，按多数票预测', () => {
    const r = computeKnn(KNN_RAW_DATA, TEST_POINT, 5, false, STATS)
    expect(r.isTie).toBe(false)
  })

  it('标准化改变距离度量：身高差被 σ 压缩，邻居排序可与原始模式不同', () => {
    const raw = computeKnn(KNN_RAW_DATA, TEST_POINT, 5, false, STATS)
    const std = computeKnn(KNN_RAW_DATA, TEST_POINT, 5, true, STATS)
    // 标准化下距离均为正且与原始值不同
    expect(std.topK[0].dist).toBeGreaterThan(0)
    expect(std.topK[0].dist).not.toBeCloseTo(raw.topK[0].dist, 5)
  })

  it('标准化距离公式校验：单点差 1 个标准差 → 距离 1', () => {
    const r = computeKnn(
      [{ h: STATS.meanH + STATS.stdH, w: STATS.meanW, s: 'M' }],
      { h: STATS.meanH, w: STATS.meanW }, 1, true, STATS
    )
    expect(r.topK[0].dist).toBeCloseTo(1, 10)
  })
})

describe('looErrorCurve', () => {
  const KS = Array.from({ length: 15 }, (_, i) => i + 1)

  it('每个 K 一个点，误差率位于 [0, 1] 且是 1/n 的整数倍（真实计数）', () => {
    const curve = looErrorCurve(KNN_RAW_DATA, KS, false, STATS)
    expect(curve.map((p) => p.k)).toEqual(KS)
    for (const { error } of curve) {
      expect(error).toBeGreaterThanOrEqual(0)
      expect(error).toBeLessThanOrEqual(1)
      const wrong = error * KNN_RAW_DATA.length
      expect(wrong).toBeCloseTo(Math.round(wrong), 10)
    }
  })

  it('手算校验：完全可分的两簇 LOO 误差为 0；翻转一个标签后 K=3 误差 = 3/6', () => {
    const clean = [
      { h: 155, w: 55, s: 'L' as const }, { h: 156, w: 55, s: 'L' as const }, { h: 155, w: 56, s: 'L' as const },
      { h: 175, w: 70, s: 'M' as const }, { h: 174, w: 70, s: 'M' as const }, { h: 175, w: 69, s: 'M' as const },
    ]
    expect(looErrorCurve(clean, [1], false, STATS)[0].error).toBe(0)
    const noisy = clean.map((p, i) => (i === 0 ? { ...p, s: 'M' as const } : p))
    // 留出 (155,55)[被翻成 M]：邻居 L, L, 远处 M → 判 L，错
    // 留出 (156,55)：邻居 (155,55)M d=1、(155,56)L d=√2、远处 M d≈23 → 判 M，错
    // 留出 (155,56)：同理 → 判 M，错
    // 远处三个 M：各有两个 M 邻居 → 全对。合计 3/6
    expect(looErrorCurve(noisy, [3], false, STATS)[0].error).toBeCloseTo(3 / 6, 10)
  })

  it('K 等于其余全部样本时退化为多数类分类器', () => {
    const n = KNN_RAW_DATA.length
    const [{ error }] = looErrorCurve(KNN_RAW_DATA, [n - 1], false, STATS)
    expect(error).toBeGreaterThan(0)
  })
})

describe('nudgeTestPoint', () => {
  it('方向键移动并限制在数据范围内', () => {
    expect(nudgeTestPoint({ h: 160, w: 60 }, 1, -1, STATS)).toEqual({ h: 161, w: 59 })
    expect(nudgeTestPoint({ h: 175, w: 55 }, 1, -1, STATS)).toEqual({ h: 175, w: 55 })
  })
})

describe('svgToDataPoint', () => {
  it('画布角点映射回数据空间边界（y 轴翻转）', () => {
    expect(svgToDataPoint(0, 400, 400, 400, STATS)).toEqual({ h: 155, w: 55 })
    expect(svgToDataPoint(400, 0, 400, 400, STATS)).toEqual({ h: 175, w: 70 })
  })

  it('结果四舍五入为整数', () => {
    const p = svgToDataPoint(123, 217, 400, 400, STATS)
    expect(Number.isInteger(p.h)).toBe(true)
    expect(Number.isInteger(p.w)).toBe(true)
  })
})
