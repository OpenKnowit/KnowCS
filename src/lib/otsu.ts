/**
 * Point operations and the iterative threshold method that lecture 7 calls "Otsu's method":
 *   T₀ = mean intensity; split into R1 (≤ T) and R2 (> T); T ← (μ1 + μ2) / 2; repeat until stable.
 */

export interface OtsuIteration {
  T: number
  mu1: number
  mu2: number
  count1: number
  count2: number
  next: number
}

export function otsuIterations(pixels: number[], T0?: number, maxIter = 50): OtsuIteration[] {
  const mean = pixels.reduce((a, b) => a + b, 0) / pixels.length
  let T = T0 ?? mean
  const out: OtsuIteration[] = []
  for (let i = 0; i < maxIter; i++) {
    const g1 = pixels.filter((p) => p <= T)
    const g2 = pixels.filter((p) => p > T)
    const mu1 = g1.length ? g1.reduce((a, b) => a + b, 0) / g1.length : NaN
    const mu2 = g2.length ? g2.reduce((a, b) => a + b, 0) / g2.length : NaN
    const next = Number.isFinite(mu1) && Number.isFinite(mu2) ? (mu1 + mu2) / 2 : T
    out.push({ T, mu1, mu2, count1: g1.length, count2: g2.length, next })
    if (Math.abs(next - T) < 1e-9) break
    T = next
  }
  return out
}

export const threshold = (pixels: number[], T: number, hi = 255): number[] => pixels.map((p) => (p > T ? hi : 0))

/** I_new = (I − I_min) / (I_max − I_min) × 255 */
export function contrastStretch(pixels: number[], lo = 0, hi = 255): number[] {
  const mn = Math.min(...pixels)
  const mx = Math.max(...pixels)
  if (mx === mn) return pixels.map(() => lo)
  return pixels.map((p) => Math.round(lo + ((p - mn) / (mx - mn)) * (hi - lo)))
}

export function histogram(pixels: number[], bins = 256, max = 255): number[] {
  const h = new Array(bins).fill(0)
  for (const p of pixels) h[Math.min(bins - 1, Math.max(0, Math.floor((p / (max + 1)) * bins)))]++
  return h
}
