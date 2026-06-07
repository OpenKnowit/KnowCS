import { describe, expect, it } from 'vitest'
import { KERNEL_PRESETS } from '../data/constants'
import { buildSourceImage, convolvePixel } from './kernel'

describe('buildSourceImage', () => {
  it('8×8 矩阵，中央 [2,6)×[2,6) 为 100，其余为 0', () => {
    const img = buildSourceImage()
    expect(img).toHaveLength(8)
    img.forEach((row) => expect(row).toHaveLength(8))
    expect(img[2][2]).toBe(100)
    expect(img[5][5]).toBe(100)
    expect(img[1][2]).toBe(0)
    expect(img[6][5]).toBe(0)
    expect(img[0][0]).toBe(0)
  })
})

describe('convolvePixel', () => {
  const img = buildSourceImage()

  it('Identity 核输出等于窗口中心像素', () => {
    const identity = KERNEL_PRESETS['Identity']
    for (let ri = 0; ri < 6; ri++) {
      for (let ci = 0; ci < 6; ci++) {
        expect(convolvePixel(img, identity, ri, ci)).toBe(img[ri + 1][ci + 1])
      }
    }
  })

  it('Sobel-X 在均匀区域内部输出 0（无梯度）', () => {
    // 输出 (3,3) 窗口覆盖 rows 3-5 × cols 3-5，全部为 100
    expect(convolvePixel(img, KERNEL_PRESETS['Sobel-X'], 3, 3)).toBe(0)
    // (0,5) 窗口仅左列触及亮块角（img[2][5]=100×-1）→ |−100| = 100
    expect(convolvePixel(img, KERNEL_PRESETS['Sobel-X'], 0, 5)).toBe(100)
  })

  it('Sobel-X 在垂直边缘响应：400 clamp 到 255', () => {
    // 输出 (2,0) 窗口覆盖 rows 2-4 × cols 0-2：仅 col 2 为 100 → (1+2+1)×100 = 400
    expect(convolvePixel(img, KERNEL_PRESETS['Sobel-X'], 2, 0)).toBe(255)
  })

  it('负响应取绝对值：右侧边缘 -400 → 255', () => {
    // 输出 (2,4) 窗口覆盖 rows 2-4 × cols 4-6：col 4-5 为 100，col 6 为 0 → -100×3... 取右缘
    expect(convolvePixel(img, KERNEL_PRESETS['Sobel-X'], 2, 4)).toBe(255)
  })

  it('Laplacian 黄金值：块左上角斜对位输出 200（二阶导响应，未触 clamp）', () => {
    // 输出 (1,1) 窗口覆盖 rows 1-3 × cols 1-3：
    // 中心 img[2][2]=100×(-4)=-400，img[2][3]=100，img[3][2]=100 → |−200| = 200
    expect(convolvePixel(img, KERNEL_PRESETS['Laplacian'], 1, 1)).toBe(200)
  })

  it('Laplacian 在均匀区域与全零区域输出 0', () => {
    expect(convolvePixel(img, KERNEL_PRESETS['Laplacian'], 3, 3)).toBe(0)
    expect(convolvePixel(img, KERNEL_PRESETS['Laplacian'], 0, 0)).toBe(0)
  })

  it('全零核输出 0；大系数核 clamp 到 255 上限', () => {
    const zero = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    expect(convolvePixel(img, zero, 2, 2)).toBe(0)
    const big = [[9, 9, 9], [9, 9, 9], [9, 9, 9]]
    expect(convolvePixel(img, big, 2, 2)).toBe(255)
  })
})
