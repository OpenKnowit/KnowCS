import { describe, expect, it } from 'vitest'
import { broadcastStrides, isRealCell, planBroadcast, shapeLabel, sourceIndex, suggestFix } from './broadcast'

describe('planBroadcast（从最后一维往前逐轴比对）', () => {
  it('(3,1) + (1,4) → (3,4)：两边各拉伸一次', () => {
    const p = planBroadcast([3, 1], [1, 4])
    expect(p.ok).toBe(true)
    expect(p.outShape).toEqual([3, 4])
    expect(p.steps.map((s) => s.verdict)).toEqual(['stretchA', 'stretchB'])
  })

  it('(3,4) + (4,) → (3,4)：缺失的维度左侧补 1', () => {
    const p = planBroadcast([3, 4], [4])
    expect(p.outShape).toEqual([3, 4])
    expect(p.steps[1]).toMatchObject({ axis: -2, a: 3, b: 1, padB: true, verdict: 'stretchB' })
  })

  it('(3,4) + (3,) 不兼容：在 axis -1 处 4 ≠ 3', () => {
    const p = planBroadcast([3, 4], [3])
    expect(p.ok).toBe(false)
    expect(p.outShape).toBeNull()
    expect(p.steps[0]).toMatchObject({ axis: -1, a: 4, b: 3, verdict: 'mismatch', out: null })
  })

  it('标量与三维', () => {
    expect(planBroadcast([3, 4], []).outShape).toEqual([3, 4])
    expect(planBroadcast([2, 3, 4], [3, 1]).outShape).toEqual([2, 3, 4])
    expect(planBroadcast([3], [3, 1]).outShape).toEqual([3, 3])
  })
})

describe('下标映射与虚拟副本', () => {
  it('被拉伸的轴总是读取下标 0', () => {
    expect(sourceIndex([2, 3], [3, 1])).toEqual([2, 0])
    expect(sourceIndex([2, 3], [1, 4])).toEqual([0, 3])
    expect(sourceIndex([1, 2, 3], [4])).toEqual([3])
    expect(sourceIndex([1, 2], [])).toEqual([])
  })

  it('isRealCell：只有原数组自己的位置为真', () => {
    expect(isRealCell([2, 0], [3, 1])).toBe(true)
    expect(isRealCell([2, 1], [3, 1])).toBe(false)
    expect(isRealCell([1, 0, 2], [3, 4])).toBe(false) // 补出来的轴上只有下标 0 是真的
    expect(isRealCell([0, 0, 2], [3, 4])).toBe(true)
  })

  it('broadcast_to 的步长：拉伸轴为 0', () => {
    expect(broadcastStrides([3, 1], [3, 4])).toEqual([1, 0])
    expect(broadcastStrides([4], [3, 4])).toEqual([0, 1])
    expect(broadcastStrides([], [2, 2])).toEqual([0, 0])
  })
})

describe('suggestFix', () => {
  it('(3,4) + (3,) → 建议 B[:, None]', () => {
    expect(suggestFix([3, 4], [3])).toEqual({ side: 'B', shape: [3, 1], code: 'B[:, None]' })
  })
  it('(3,) + (3,4) → 建议 A[:, None]', () => {
    expect(suggestFix([3], [3, 4])).toEqual({ side: 'A', shape: [3, 1], code: 'A[:, None]' })
  })
  it('已兼容或无解时返回 null', () => {
    expect(suggestFix([3, 1], [1, 4])).toBeNull()
    expect(suggestFix([2, 3], [4, 5])).toBeNull()
  })
  it('shapeLabel', () => {
    expect([[], [3], [3, 4]].map(shapeLabel)).toEqual(['()', '(3,)', '(3, 4)'])
  })
})
