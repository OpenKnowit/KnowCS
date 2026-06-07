import type { SliceType } from '../types'

// --- NumPy 切片高亮逻辑（从 NumpyModule 提取，行为保持不变） ---

/** 判断矩阵 (r, c) 处值为 val 的单元格在当前切片模式下是否选中
 *  - slice  a[1:2, :]      → 第 1 行（View）
 *  - fancy  a[[0, 2]]      → 第 0、2 行（Copy）
 *  - mask   a[a % 2 == 0]  → 偶数值（Copy）
 */
export const isHighlighted = (sliceType: SliceType, r: number, val: number): boolean => {
  if (sliceType === 'slice') return r === 1
  if (sliceType === 'fancy') return r === 0 || r === 2
  if (sliceType === 'mask') return val % 2 === 0
  return false
}
