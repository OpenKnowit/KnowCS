import { describe, expect, it } from 'vitest'
import { KERNEL_PRESETS } from '../data/constants'
import { buildSourceImage, convolve, convolvePixel, kernelGain, kernelPreset, presetImage, presetNormalized, resample, resizeKernel } from './kernel'

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

describe('kernels of any size', () => {
  it('keeps the course kernels at 3 × 3', () => {
    for (const name of Object.keys(KERNEL_PRESETS) as (keyof typeof KERNEL_PRESETS)[]) expect(kernelPreset(name, 3)).toEqual(KERNEL_PRESETS[name])
  })

  it('builds Sobel the way OpenCV does', () => {
    // cv2.getDerivKernels(1, 0, k): the derivative across, the binomial smoothing down
    expect(kernelPreset('Sobel-X', 5)[0]).toEqual([-1, -2, 0, 2, 1])
    expect(kernelPreset('Sobel-X', 5).map((r) => r[4])).toEqual([1, 4, 6, 4, 1])
    expect(kernelPreset('Sobel-X', 7)[0]).toEqual([-1, -4, -5, 0, 5, 4, 1])
    expect(kernelPreset('Sobel-X', 7).map((r) => r[6])).toEqual([1, 6, 15, 20, 15, 6, 1])
    // same sign as the course's Sobel-Y: positive on top
    expect(kernelPreset('Sobel-Y', 5)[0]).toEqual([1, 4, 6, 4, 1])
    expect(kernelPreset('Sobel-Y', 5)[4]).toEqual([-1, -4, -6, -4, -1])
  })

  it('has edge kernels that sum to 0, so a flat picture gives nothing', () => {
    const flat = Array.from({ length: 16 }, () => new Array<number>(16).fill(120))
    for (const k of [3, 5, 7]) {
      for (const name of ['Sobel-X', 'Sobel-Y', 'Laplacian'] as const) {
        const m = kernelPreset(name, k)
        expect(m.flat().reduce((a, b) => a + b, 0)).toBe(0)
        expect(convolve(flat, m).flat().every((v) => v === 0)).toBe(true)
      }
    }
  })

  it('leaves a flat picture alone when a blur is normalized', () => {
    const flat = Array.from({ length: 16 }, () => new Array<number>(16).fill(120))
    for (const k of [3, 5, 7]) {
      for (const name of ['Box blur', 'Gaussian', 'Identity'] as const) {
        const m = kernelPreset(name, k)
        expect(convolve(flat, m, kernelGain(m)).flat().every((v) => Math.abs(v - 120) < 1e-9)).toBe(true)
      }
    }
  })

  it('shrinks the output to N − k + 1, and matches NumPy on a worked example', () => {
    expect(convolve(presetImage('square', 8), kernelPreset('Sobel-X', 3))).toHaveLength(6)
    expect(convolve(presetImage('square', 32), kernelPreset('Sobel-X', 7))).toHaveLength(26)
    expect(convolve(presetImage('square', 8), kernelPreset('Identity', 7))).toHaveLength(2)
    // the 3 × 3 result the course demo has always shown
    const row2 = convolve(buildSourceImage(), KERNEL_PRESETS['Sobel-X'])[2]
    expect(row2).toEqual([255, 255, 0, 0, 255, 255])
  })

  it('normalizes the presets whose weights would saturate, not the course ones', () => {
    expect(presetNormalized('Sobel-X', 3)).toBe(false)
    expect(presetNormalized('Sobel-X', 5)).toBe(true)
    expect(presetNormalized('Gaussian', 3)).toBe(true)
    expect(presetNormalized('Identity', 7)).toBe(false)
  })

  it('keeps the middle of a hand-edited kernel when its size changes', () => {
    const m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    expect(resizeKernel(m, 5)[2]).toEqual([0, 4, 5, 6, 0])
    expect(resizeKernel(resizeKernel(m, 5), 3)).toEqual(m)
  })
})

describe('pictures', () => {
  it('draws the course demo as the square preset', () => {
    expect(presetImage('square', 8)).toEqual(buildSourceImage())
  })
  it('stays in 0–255 at every size', () => {
    for (const name of ['square', 'circle', 'stripes', 'checker', 'gradient', 'house'] as const) {
      for (const n of [8, 16, 128]) expect(presetImage(name, n).flat().every((v) => v >= 0 && v <= 255)).toBe(true)
    }
  })
  it('averages when it scales down', () => {
    expect(resample([[0, 100], [100, 200]], 1)).toEqual([[100]])
    expect(resample([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30], [20, 20, 30, 30]], 2)).toEqual([[0, 10], [20, 30]])
    expect(resample(presetImage('gradient', 256), 64)).toHaveLength(64)
  })
})
