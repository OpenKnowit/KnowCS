// The distributions np.random offers. np.random and RandomState draw from NumPy's own generator (mt19937.ts), so uniform,
// normal, exponential and small-mean Poisson draws match NumPy; the other distributions use simpler methods than NumPy's.
import type { SandboxRandom } from './ndops'

export const uniform = (r: SandboxRandom) => r.next()
export const standardNormal = (r: SandboxRandom) => r.normal()
export const exponential = (r: SandboxRandom) => -Math.log(1 - r.next())

/** Marsaglia and Tsang's method */
export const gamma = (r: SandboxRandom, shape: number): number => {
  if (shape <= 0) return 0
  if (shape < 1) return gamma(r, shape + 1) * r.next() ** (1 / shape)
  const d = shape - 1 / 3
  const c = 1 / Math.sqrt(9 * d)
  for (;;) {
    const x = r.normal()
    const v = (1 + c * x) ** 3
    if (v <= 0) continue
    const u = 1 - r.next()
    if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v
  }
}
export const beta = (r: SandboxRandom, a: number, b: number) => {
  const x = gamma(r, a)
  return x / (x + gamma(r, b))
}
export const chisquare = (r: SandboxRandom, df: number) => 2 * gamma(r, df / 2)
export const studentT = (r: SandboxRandom, df: number) => r.normal() / Math.sqrt(chisquare(r, df) / df)

export const poisson = (r: SandboxRandom, lam: number): number => {
  if (lam <= 0) return 0
  if (lam < 30) {
    const limit = Math.exp(-lam)
    let k = 0
    let p = r.next()
    while (p > limit) {
      k++
      p *= r.next()
    }
    return k
  }
  // split the mean: a sum of independent Poissons is Poisson
  return poisson(r, lam / 2) + poisson(r, lam / 2)
}
export const binomial = (r: SandboxRandom, n: number, p: number): number => {
  if (n > 2000) {
    // far past where it can be told apart from the normal approximation
    const x = Math.round(n * p + Math.sqrt(n * p * (1 - p)) * r.normal())
    return Math.min(n, Math.max(0, x))
  }
  let k = 0
  for (let i = 0; i < n; i++) if (r.next() < p) k++
  return k
}
export const geometric = (r: SandboxRandom, p: number) => (p >= 1 ? 1 : Math.ceil(Math.log(1 - r.next()) / Math.log(1 - p)))
export const multinomial = (r: SandboxRandom, n: number, ps: number[]): number[] => {
  let left = n
  let mass = 1
  return ps.map((p, i) => {
    if (i === ps.length - 1) return left
    const k = binomial(r, left, mass > 0 ? Math.min(1, p / mass) : 0)
    left -= k
    mass -= p
    return k
  })
}
export const dirichlet = (r: SandboxRandom, alpha: number[]): number[] => {
  const xs = alpha.map((a) => gamma(r, a))
  const total = xs.reduce((s, x) => s + x, 0)
  return xs.map((x) => x / total)
}
export const triangular = (r: SandboxRandom, lo: number, mode: number, hi: number) => {
  const u = r.next()
  const cut = (mode - lo) / (hi - lo)
  return u < cut ? lo + Math.sqrt(u * (hi - lo) * (mode - lo)) : hi - Math.sqrt((1 - u) * (hi - lo) * (hi - mode))
}

/** a random order of 0 … n−1 */
export const permutationOf = (r: SandboxRandom, n: number): number[] => {
  const order = Array.from({ length: n }, (_, i) => i)
  for (let i = n - 1; i > 0; i--) {
    const j = r.interval(i)
    ;[order[i], order[j]] = [order[j], order[i]]
  }
  return order
}
