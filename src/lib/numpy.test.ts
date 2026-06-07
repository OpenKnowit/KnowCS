import { describe, expect, it } from 'vitest'
import { INITIAL_MATRIX } from '../data/constants'
import { isHighlighted } from './numpy'

describe('isHighlighted', () => {
  it('slice 模式 a[1:2, :]：仅第 1 行选中', () => {
    expect(isHighlighted('slice', 1, 4)).toBe(true)
    expect(isHighlighted('slice', 1, 7)).toBe(true)
    expect(isHighlighted('slice', 0, 0)).toBe(false)
    expect(isHighlighted('slice', 2, 8)).toBe(false)
  })

  it('fancy 模式 a[[0, 2]]：第 0、2 行选中', () => {
    expect(isHighlighted('fancy', 0, 0)).toBe(true)
    expect(isHighlighted('fancy', 2, 8)).toBe(true)
    expect(isHighlighted('fancy', 1, 4)).toBe(false)
    expect(isHighlighted('fancy', 3, 12)).toBe(false)
  })

  it('mask 模式 a[a % 2 == 0]：偶数值选中，与行无关', () => {
    expect(isHighlighted('mask', 0, 0)).toBe(true)
    expect(isHighlighted('mask', 3, 14)).toBe(true)
    expect(isHighlighted('mask', 0, 1)).toBe(false)
    expect(isHighlighted('mask', 2, 11)).toBe(false)
  })

  it('none 模式：全部不选中', () => {
    INITIAL_MATRIX.forEach((row, r) =>
      row.forEach((val) => expect(isHighlighted('none', r, val)).toBe(false))
    )
  })

  it('对 4×4 演示矩阵：mask 选中 8 格，slice 选中 4 格，fancy 选中 8 格', () => {
    const count = (type: 'slice' | 'fancy' | 'mask') =>
      INITIAL_MATRIX.flatMap((row, r) => row.filter((val) => isHighlighted(type, r, val))).length
    expect(count('slice')).toBe(4)
    expect(count('fancy')).toBe(8)
    expect(count('mask')).toBe(8)
  })
})
