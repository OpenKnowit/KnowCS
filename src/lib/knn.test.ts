import { describe, expect, it } from 'vitest'
import { KNN_RAW_DATA } from '../data/constants'
import type { KnnStats } from '../types'
import { complexityCurve, computeKnn, scaleLinear, svgToDataPoint } from './knn'

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

  it('neighborsMap 按距离名次编号 1..K', () => {
    const r = computeKnn(KNN_RAW_DATA, TEST_POINT, 3, false, STATS)
    expect(r.neighborsMap.size).toBe(3)
    expect(r.neighborsMap.get('160-60')).toBe(1)
  })

  it('K 大于样本数时取全部样本', () => {
    const r = computeKnn(KNN_RAW_DATA, TEST_POINT, 99, false, STATS)
    expect(r.topK).toHaveLength(KNN_RAW_DATA.length)
    expect(r.mCount + r.lCount).toBe(KNN_RAW_DATA.length)
  })

  it('平票时预测 M（mCount >= lCount 规则）', () => {
    const data = KNN_RAW_DATA.slice(0, 8) // 7 M + 1 L
    const r = computeKnn(data, { h: 161, w: 63 }, 2, false, STATS)
    if (r.mCount === r.lCount) expect(r.prediction).toBe('M')
    // 构造确定平票：1 M + 1 L 等距
    const tie = computeKnn(
      [{ h: 160, w: 61, s: 'M' }, { h: 162, w: 61, s: 'L' }],
      { h: 161, w: 61 }, 2, false, STATS
    )
    expect(tie.mCount).toBe(1)
    expect(tie.lCount).toBe(1)
    expect(tie.prediction).toBe('M')
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

describe('complexityCurve', () => {
  it('确定性：两次调用结果一致，长度 15，k 从 1 到 15', () => {
    const a = complexityCurve()
    const b = complexityCurve()
    expect(a).toEqual(b)
    expect(a).toHaveLength(15)
    expect(a[0].k).toBe(1)
    expect(a[14].k).toBe(15)
  })

  it('U 形：k=4 附近误差低于两端（过拟合/欠拟合演示）', () => {
    const c = complexityCurve()
    const errAt = (k: number) => c[k - 1].error
    expect(errAt(4)).toBeLessThan(errAt(1))
    expect(errAt(4)).toBeLessThan(errAt(15))
    c.forEach((p) => expect(p.error).toBeGreaterThanOrEqual(0.1))
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
