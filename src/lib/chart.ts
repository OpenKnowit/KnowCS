// --- 轻量折线图的纯计算：线性比例尺 + 「好看」的刻度（替代 Recharts，省去 ~300KB 依赖） ---

/** 线性比例尺：把 domain 区间映射到 range 区间；domain 退化（min = max）时映射到 range 中点。 */
export const linearScale = (
  [d0, d1]: [number, number],
  [r0, r1]: [number, number]
): ((v: number) => number) => {
  if (d0 === d1) return () => (r0 + r1) / 2
  return (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0)
}

/** 1/2/5×10ⁿ 序列中不小于 raw 的最小步长 */
const niceStep = (raw: number): number => {
  const pow = 10 ** Math.floor(Math.log10(raw))
  const frac = raw / pow
  return (frac <= 1 ? 1 : frac <= 2 ? 2 : frac <= 5 ? 5 : 10) * pow
}

/**
 * 生成覆盖 [min, max] 的整齐刻度（步长取 1/2/5×10ⁿ），约 count 个。
 * 返回的首尾刻度会向外取整，可直接作为 y 轴 domain。
 */
export const niceTicks = (min: number, max: number, count = 4): number[] => {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1]
  if (min === max) {
    const pad = Math.abs(min) || 1
    return niceTicks(min - pad, max + pad, count)
  }
  const step = niceStep((max - min) / Math.max(1, count))
  const start = Math.floor(min / step) * step
  const end = Math.ceil(max / step) * step
  const ticks: number[] = []
  // 用整数计数避免浮点累加误差（0.1 + 0.2 ≠ 0.3）
  const n = Math.round((end - start) / step)
  for (let i = 0; i <= n; i++) ticks.push(Number((start + i * step).toPrecision(12)))
  return ticks
}

/** 在已排序的 x 序列中找离 x 最近的下标（悬停提示用） */
export const nearestIndex = (xs: number[], x: number): number => {
  let best = 0
  for (let i = 1; i < xs.length; i++) {
    if (Math.abs(xs[i] - x) < Math.abs(xs[best] - x)) best = i
  }
  return best
}
