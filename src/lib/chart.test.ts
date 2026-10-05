import { describe, expect, it } from 'vitest'
import { linearScale, nearestIndex, niceTicks } from './chart'

describe('linearScale', () => {
  it('端点与中点映射，支持反向 range（SVG y 轴向下）', () => {
    const s = linearScale([0, 10], [100, 0])
    expect(s(0)).toBe(100)
    expect(s(10)).toBe(0)
    expect(s(5)).toBe(50)
  })

  it('domain 退化时映射到 range 中点，不产生 NaN', () => {
    expect(linearScale([3, 3], [0, 200])(3)).toBe(100)
  })
})

describe('niceTicks', () => {
  it('步长取 1/2/5×10ⁿ，首尾向外取整覆盖数据', () => {
    // 原始步长 0.25 → 向上取到 0.5
    expect(niceTicks(0, 1, 4)).toEqual([0, 0.5, 1])
    const t = niceTicks(0.06, 0.44, 4)
    expect(t[0]).toBeLessThanOrEqual(0.06)
    expect(t.at(-1)!).toBeGreaterThanOrEqual(0.44)
    expect(t).toEqual([0, 0.1, 0.2, 0.3, 0.4, 0.5])
  })

  it('大数值：WCSS 量级', () => {
    expect(niceTicks(12.3, 287.9, 4)).toEqual([0, 100, 200, 300])
  })

  it('无浮点累加误差', () => {
    for (const v of niceTicks(0, 0.7, 7)) expect(String(v).length).toBeLessThanOrEqual(4)
  })

  it('min = max 或非有限值时仍返回可用刻度', () => {
    const t = niceTicks(5, 5)
    expect(t[0]).toBeLessThan(5)
    expect(t.at(-1)!).toBeGreaterThan(5)
    expect(niceTicks(NaN, 1)).toEqual([0, 1])
  })
})

describe('nearestIndex', () => {
  it('返回最近点下标', () => {
    expect(nearestIndex([1, 2, 3, 4], 2.4)).toBe(1)
    expect(nearestIndex([1, 2, 3, 4], 9)).toBe(3)
    expect(nearestIndex([1, 2, 3, 4], -1)).toBe(0)
  })
})
