// --- 卷积纯计算逻辑（从 KernelModule 提取）：任意奇数大小的卷积核、预设核、演示图与降采样 ---
import { KERNEL_PRESETS } from '../data/constants'
import type { KernelPresetName } from '../types'

export const KERNEL_SIZES = [3, 5, 7] as const
export const IMAGE_SIZES = [8, 16, 32, 64, 128] as const

/** 生成 8×8 演示源图：中央 4×4 区域亮度 100，其余 0 */
export const buildSourceImage = (): number[][] => presetImage('square', 8)

/** 计算输出特征图 (ri, ci) 处的卷积值：核滑窗求和（可除以 divisor），取绝对值并 clamp 到 [0, 255] */
export const convolvePixel = (sourceImage: number[][], kernel: number[][], ri: number, ci: number, divisor = 1): number => {
  let sum = 0
  const k = kernel.length
  for (let kr = 0; kr < k; kr++) {
    for (let kc = 0; kc < k; kc++) {
      sum += sourceImage[ri + kr][ci + kc] * kernel[kr][kc]
    }
  }
  return Math.max(0, Math.min(255, Math.abs(sum / divisor)))
}

/** the whole output map, without padding: an N × N image and a k × k kernel give N − k + 1 per side */
export const convolve = (image: number[][], kernel: number[][], divisor = 1): number[][] => {
  const n = Math.max(0, image.length - kernel.length + 1)
  return Array.from({ length: n }, (_, r) => Array.from({ length: n }, (_, c) => convolvePixel(image, kernel, r, c, divisor)))
}

/** what keeps a result in range: the sum of the positive weights, so that white stays white at most */
export const kernelGain = (kernel: number[][]): number => Math.max(1, kernel.flat().reduce((s, v) => (v > 0 ? s + v : s), 0))

const binomial = (n: number): number[] => {
  let row = [1]
  for (let i = 0; i < n; i++) row = [0, ...row].map((v, j) => v + (row[j] ?? 0))
  return row
}
/** the derivative side of a Sobel kernel: binomial(k − 3) slid over [−1, 0, 1], as OpenCV builds it */
const sobelDerivative = (k: number): number[] => {
  const b = binomial(k - 3)
  return Array.from({ length: k }, (_, i) => (b[i - 2] ?? 0) - (b[i] ?? 0))
}
const outer = (col: number[], row: number[]) => col.map((a) => row.map((b) => a * b))
const clean = (m: number[][]) => m.map((row) => row.map((v) => (v === 0 ? 0 : v)))

/** a preset at any size. At 3 × 3 it is exactly the course's kernel */
export const kernelPreset = (name: KernelPresetName, k: number): number[][] => {
  if (k === 3) return KERNEL_PRESETS[name]
  const mid = (k - 1) / 2
  switch (name) {
    case 'Sobel-X':
      return clean(outer(binomial(k - 1), sobelDerivative(k)))
    case 'Sobel-Y':
      return clean(outer(sobelDerivative(k).map((v) => -v), binomial(k - 1)))
    case 'Laplacian':
      if (k === 5) return [[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, -16, 2, 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]]
      return Array.from({ length: k }, (_, r) => Array.from({ length: k }, (_, c) => (r === mid && c === mid ? 1 - k * k : 1)))
    case 'Identity':
      return Array.from({ length: k }, (_, r) => Array.from({ length: k }, (_, c) => (r === mid && c === mid ? 1 : 0)))
    case 'Box blur':
      return Array.from({ length: k }, () => new Array<number>(k).fill(1))
    case 'Gaussian':
      return outer(binomial(k - 1), binomial(k - 1))
  }
}

/** blurs always, and edges past 3 × 3, whose weights would otherwise turn nearly everything white */
export const presetNormalized = (name: KernelPresetName, k: number) => name === 'Box blur' || name === 'Gaussian' || (k > 3 && name !== 'Identity')

/** a hand-edited kernel at a new size: the middle stays, new cells are 0 */
export const resizeKernel = (kernel: number[][], k: number): number[][] => {
  const shift = (kernel.length - k) / 2
  return Array.from({ length: k }, (_, r) => Array.from({ length: k }, (_, c) => kernel[r + shift]?.[c + shift] ?? 0))
}

// ---------------------------------------------------------------- the built-in pictures, drawn from numbers

export const IMAGE_PRESETS = ['square', 'circle', 'stripes', 'checker', 'gradient', 'house'] as const
export type ImagePreset = (typeof IMAGE_PRESETS)[number]

/** every preset is defined over 0–1, so it looks the same at any resolution; square is the course demo at 8 × 8 */
export const presetImage = (name: ImagePreset, n: number): number[][] =>
  Array.from({ length: n }, (_, r) =>
    Array.from({ length: n }, (_, c) => {
      const y = (r + 0.5) / n
      const x = (c + 0.5) / n
      switch (name) {
        case 'square':
          return x > 0.25 && x < 0.75 && y > 0.25 && y < 0.75 ? 100 : 0
        case 'circle':
          return (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.1 ? 220 : 30
        case 'stripes':
          return Math.floor(x * 4) % 2 ? 220 : 30
        case 'checker':
          return (Math.floor(x * 4) + Math.floor(y * 4)) % 2 ? 230 : 20
        case 'gradient':
          return Math.round(((x + y) / 2) * 255)
        case 'house': {
          const sun = (x - 0.8) ** 2 + (y - 0.2) ** 2 < 0.01
          const roof = y > 0.25 && y < 0.5 && Math.abs(x - 0.42) < y - 0.2
          const wall = y >= 0.5 && y < 0.85 && x > 0.2 && x < 0.64
          const door = wall && y > 0.62 && x > 0.36 && x < 0.48
          if (y >= 0.85) return 70
          if (door) return 40
          if (roof) return 90
          if (wall) return 200
          return sun ? 250 : 150
        }
      }
    }),
  )

/** average a square picture down to n × n, each new pixel the mean of the cells it covers */
export const resample = (image: number[][], n: number): number[][] => {
  const m = image.length
  return Array.from({ length: n }, (_, r) =>
    Array.from({ length: n }, (_, c) => {
      const r0 = Math.floor((r * m) / n)
      const r1 = Math.max(r0 + 1, Math.floor(((r + 1) * m) / n))
      const c0 = Math.floor((c * m) / n)
      const c1 = Math.max(c0 + 1, Math.floor(((c + 1) * m) / n))
      let sum = 0
      for (let i = r0; i < r1; i++) for (let j = c0; j < c1; j++) sum += image[i][j]
      return Math.round(sum / ((r1 - r0) * (c1 - c0)))
    }),
  )
}
