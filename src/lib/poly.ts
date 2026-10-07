// --- numpy.polynomial 子集：系数从低次到高次；domain / window 映射与 numpy 相同 ---
import { NDArray, arrayRepr } from './ndarray'

export type Interval = [number, number]
export const DEFAULT_IV: Interval = [-1, 1]

/** 去掉末尾（最高次）为 0 的系数，至少保留一个 */
export const trim = (c: number[]): number[] => {
  const out = [...c]
  while (out.length > 1 && out[out.length - 1] === 0) out.pop()
  return out.length ? out : [0]
}

/** Horner 求值：c[0] + c[1]·x + c[2]·x² … */
export const polyval = (c: number[], x: number): number => c.reduceRight((acc, ci) => acc * x + ci, 0)

export const polyadd = (a: number[], b: number[]): number[] => trim(Array.from({ length: Math.max(a.length, b.length) }, (_, i) => (a[i] ?? 0) + (b[i] ?? 0)))
export const polysub = (a: number[], b: number[]): number[] => trim(Array.from({ length: Math.max(a.length, b.length) }, (_, i) => (a[i] ?? 0) - (b[i] ?? 0)))
export const polymul = (a: number[], b: number[]): number[] => {
  const out = new Array(a.length + b.length - 1).fill(0)
  a.forEach((x, i) => b.forEach((y, j) => (out[i + j] += x * y)))
  return trim(out)
}
export const polypow = (a: number[], n: number): number[] => {
  let out = [1]
  for (let i = 0; i < n; i++) out = polymul(out, a)
  return out
}

/** m 阶导数（在 window 变量上求导后乘以映射比例 scl^m，与 numpy 一致） */
export const deriv = (c: number[], m = 1, scl = 1): number[] => {
  let out = [...c]
  for (let k = 0; k < m; k++) out = out.length <= 1 ? [0] : out.slice(1).map((v, i) => v * (i + 1) * scl)
  return out
}

/** m 次积分，积分常数 k（下界 0） */
export const integ = (c: number[], m = 1, k = 0, scl = 1): number[] => {
  let out = [...c]
  for (let t = 0; t < m; t++) out = [k, ...out.map((v, i) => (v * scl) / (i + 1))]
  return out
}

/** x' = off + scl·x 把 domain 映射到 window */
export const mapParams = (domain: Interval, window: Interval): { off: number; scl: number } => {
  const scl = (window[1] - window[0]) / (domain[1] - domain[0])
  return { off: window[0] - scl * domain[0], scl }
}

/** 把 window 变量下的系数还原成普通 x 的系数（Polynomial.convert()） */
export const convert = (c: number[], domain: Interval, window: Interval): number[] => {
  const { off, scl } = mapParams(domain, window)
  // p(off + scl·x) = Σ c_k (off + scl·x)^k
  return c.reduce((acc, ck, k) => polyadd(acc, polymul([ck], polypow([off, scl], k))), [0])
}

/** 最小二乘拟合（window 变量上的范德蒙矩阵，Householder QR 求解） */
export const lstsq = (x: number[], y: number[], deg: number): number[] => {
  const n = x.length
  const m = deg + 1
  if (n < m) throw new Error('not enough points for this degree')
  const A = x.map((xi) => Array.from({ length: m }, (_, j) => xi ** j))
  const b = [...y]
  // Householder QR
  for (let k = 0; k < m; k++) {
    let norm = 0
    for (let i = k; i < n; i++) norm += A[i][k] ** 2
    norm = Math.sqrt(norm)
    if (norm === 0) continue
    const alpha = A[k][k] > 0 ? -norm : norm
    const v = A.map((row, i) => (i < k ? 0 : row[k]))
    v[k] -= alpha
    const vv = v.reduce((s, t) => s + t * t, 0)
    if (vv === 0) continue
    for (let j = k; j < m; j++) {
      const d = v.reduce((s, t, i) => s + t * A[i][j], 0) * (2 / vv)
      for (let i = k; i < n; i++) A[i][j] -= d * v[i]
    }
    const d = v.reduce((s, t, i) => s + t * b[i], 0) * (2 / vv)
    for (let i = k; i < n; i++) b[i] -= d * v[i]
  }
  const c = new Array(m).fill(0)
  for (let i = m - 1; i >= 0; i--) {
    let s = b[i]
    for (let j = i + 1; j < m; j++) s -= A[i][j] * c[j]
    c[i] = s / A[i][i]
  }
  return c
}

// ---------------- 求根（Durand–Kerner，实根时返回实数） ----------------

type C = [number, number]
const cmul = (a: C, b: C): C => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
const csub = (a: C, b: C): C => [a[0] - b[0], a[1] - b[1]]
const cdiv = (a: C, b: C): C => {
  const d = b[0] * b[0] + b[1] * b[1]
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]
}

export interface Roots {
  re: number[]
  im: number[]
  /** 所有根都是实数时为 true（numpy 此时返回 float 数组） */
  real: boolean
}

export const roots = (coef: number[]): Roots => {
  const c = trim(coef)
  const n = c.length - 1
  if (n < 1) return { re: [], im: [], real: true }
  if (n === 1) return { re: [-c[0] / c[1]], im: [0], real: true }
  const monic = c.map((v) => v / c[n])
  const evalC = (z: C): C => monic.reduceRight<C>((acc, ci) => [acc[0] * z[0] - acc[1] * z[1] + ci, acc[0] * z[1] + acc[1] * z[0]], [0, 0])
  let zs: C[] = Array.from({ length: n }, (_, k) => {
    let p: C = [1, 0]
    for (let t = 0; t < k; t++) p = cmul(p, [0.4, 0.9])
    return p
  })
  for (let iter = 0; iter < 800; iter++) {
    let moved = 0
    zs = zs.map((z, i) => {
      let den: C = [1, 0]
      zs.forEach((w, j) => { if (j !== i) den = cmul(den, csub(z, w)) })
      const nz = csub(z, cdiv(evalC(z), den))
      moved = Math.max(moved, Math.abs(nz[0] - z[0]) + Math.abs(nz[1] - z[1]))
      return nz
    })
    if (moved < 1e-15) break
  }
  const scale = Math.max(1, ...zs.map((z) => Math.hypot(z[0], z[1])))
  const real = zs.every((z) => Math.abs(z[1]) < 1e-9 * scale)
  // numpy sorts complex roots by real part, then imaginary part
  zs.sort((a, b) => (Math.abs(a[0] - b[0]) > 1e-9 * scale ? a[0] - b[0] : a[1] - b[1]))
  return { re: zs.map((z) => z[0]), im: real ? zs.map(() => 0) : zs.map((z) => z[1]), real }
}

// ---------------- 输出格式（对齐 numpy 2.x） ----------------

/** Python float 的 repr：整数值带 .0 */
export const pyFloat = (v: number): string => {
  if (Number.isNaN(v)) return 'nan'
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf'
  if (Number.isInteger(v) && Math.abs(v) < 1e16) return `${v}.0`
  return String(v)
}

const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹'
const sup = (n: number) => [...String(n)].map((d) => SUP[Number(d)]).join('')

/** str(Polynomial)：1.0 + 2.0·x + 3.0·x² */
export const polyStr = (c: number[]): string =>
  c
    .map((v, i) => {
      const body = i === 0 ? pyFloat(Math.abs(v)) : `${pyFloat(Math.abs(v))}·x${i > 1 ? sup(i) : ''}`
      if (i === 0) return (v < 0 || Object.is(v, -0) ? '-' : '') + body
      return `${v < 0 ? '-' : '+'} ${body}`
    })
    .join(' ')

const inner = (vals: number[]): string => arrayRepr(NDArray.create(vals, [vals.length], 'float64')).slice(6, -1)

/** repr(Polynomial) */
export const polyRepr = (c: number[], domain: Interval, window: Interval): string =>
  `Polynomial(${inner(c)}, domain=${inner(domain)}, window=${inner(window)}, symbol='x')`

/** 复数数组的 repr：array([-0.33333333-0.47140452j, -0.33333333+0.47140452j]) */
export const complexRepr = (re: number[], im: number[]): string => {
  const fmt = (v: number) => (Math.abs(v) < 1e-15 ? '0.' : String(Number(v.toFixed(8))).replace(/^(-?\d+)$/, '$1.'))
  const items = re.map((r, i) => `${fmt(r)}${im[i] < 0 ? '-' : '+'}${fmt(Math.abs(im[i]))}j`)
  return `array([${items.join(', ')}])`
}
