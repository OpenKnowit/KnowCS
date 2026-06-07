// --- 卷积纯计算逻辑（从 KernelModule 提取，行为保持不变） ---

/** 生成 8×8 演示源图：中央 4×4 区域亮度 100，其余 0 */
export const buildSourceImage = (): number[][] => {
  const img: number[][] = Array.from({ length: 8 }, () => Array<number>(8).fill(0))
  for (let i = 2; i < 6; i++) {
    for (let j = 2; j < 6; j++) img[i][j] = 100
  }
  return img
}

/** 计算输出特征图 (ri, ci) 处的卷积值：3×3 核滑窗求和，取绝对值并 clamp 到 [0, 255] */
export const convolvePixel = (sourceImage: number[][], kernel: number[][], ri: number, ci: number): number => {
  let sum = 0
  for (let kr = 0; kr < 3; kr++) {
    for (let kc = 0; kc < 3; kc++) {
      sum += sourceImage[ri + kr][ci + kc] * kernel[kr][kc]
    }
  }
  return Math.max(0, Math.min(255, Math.abs(sum)))
}
