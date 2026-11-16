// NumPy's legacy RandomState: the MT19937 generator and the exact way np.random.seed / rand / randn / randint / choice /
// shuffle / permutation turn its 32-bit words into numbers, so a seeded program prints what real NumPy prints.
// sklearn's random_state and pandas' sample run on it too, as they do in the real libraries.
import { SandboxRandom } from './ndops'
import { repr } from './minipy'
import type { Host, Value } from './minipy'

const N = 624

export const cumNormalised = (p: number[]): number[] => {
  let c = 0
  const out = p.map((x) => (c += x))
  const last = out[out.length - 1]
  return out.map((x) => x / last)
}
export const searchRight = (sorted: number[], x: number): number => {
  let lo = 0
  let hi = sorted.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (x < sorted[mid]) hi = mid
    else lo = mid + 1
  }
  return lo
}
export const searchLeft = (sorted: number[], x: number): number => {
  let lo = 0
  let hi = sorted.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (sorted[mid] < x) lo = mid + 1
    else hi = mid
  }
  return lo
}

export interface LegacyState { key: number[]; pos: number; gauss: number | null }

export class LegacyRandom extends SandboxRandom {
  private key = new Uint32Array(N)
  private pos = N
  private gauss: number | null = null
  constructor(seed: number | number[] = 0) {
    super()
    this.seed(seed)
  }
  /** mt19937_seed for one integer, init_by_array for a list (both drop a cached normal) */
  override seed(seed: number | number[]) {
    if (typeof seed === 'number') this.init(seed)
    else this.initByArray(seed)
    this.pos = N
    this.gauss = null
  }
  private init(seed: number) {
    let s = seed >>> 0
    for (let i = 0; i < N; i++) {
      this.key[i] = s
      s = (Math.imul(1812433253, s ^ (s >>> 30)) + i + 1) >>> 0
    }
  }
  private initByArray(words: number[]) {
    const mt = this.key
    this.init(19650218)
    let i = 1
    let j = 0
    for (let k = Math.max(N, words.length); k; k--) {
      mt[i] = ((mt[i] ^ Math.imul(mt[i - 1] ^ (mt[i - 1] >>> 30), 1664525)) + words[j] + j) >>> 0
      i++
      j++
      if (i >= N) { mt[0] = mt[N - 1]; i = 1 }
      if (j >= words.length) j = 0
    }
    for (let k = N - 1; k; k--) {
      mt[i] = ((mt[i] ^ Math.imul(mt[i - 1] ^ (mt[i - 1] >>> 30), 1566083941)) - i) >>> 0
      i++
      if (i >= N) { mt[0] = mt[N - 1]; i = 1 }
    }
    mt[0] = 0x80000000
  }
  getState(): LegacyState {
    return { key: [...this.key], pos: this.pos, gauss: this.gauss }
  }
  setState(s: LegacyState) {
    this.key = Uint32Array.from(s.key)
    this.pos = s.pos
    this.gauss = s.gauss
  }
  next32(): number {
    if (this.pos === N) {
      const k = this.key
      for (let i = 0; i < N; i++) {
        const y = (k[i] & 0x80000000) | (k[(i + 1) % N] & 0x7fffffff)
        k[i] = k[(i + 397) % N] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0)
      }
      this.pos = 0
    }
    let y = this.key[this.pos++]
    y ^= y >>> 11
    y ^= (y << 7) & 0x9d2c5680
    y ^= (y << 15) & 0xefc60000
    y ^= y >>> 18
    return y >>> 0
  }
  /** a double in [0, 1) from 53 random bits */
  double(): number {
    const a = this.next32() >>> 5
    const b = this.next32() >>> 6
    return (a * 67108864 + b) / 9007199254740992
  }
  override next(): number {
    return this.double()
  }
  /** random_interval: uniform on 0..max by rejection through a bit mask */
  override interval(max: number): number {
    if (max === 0) return 0
    if (max > 0xffffffff) return super.interval(max)
    let mask = max
    for (let s = 1; s < 32; s <<= 1) mask |= mask >>> s
    mask >>>= 0
    for (;;) {
      const v = (this.next32() & mask) >>> 0
      if (v <= max) return v
    }
  }
  /** randint(low, high): the masked 32-bit generator NumPy's legacy interface uses */
  randint(low: number, high: number): number {
    return low + this.interval(high - low - 1)
  }
  shuffle<T>(a: T[]) {
    for (let i = a.length - 1; i >= 1; i--) {
      const j = this.interval(i)
      const t = a[j]
      a[j] = a[i]
      a[i] = t
    }
  }
  permutation(n: number): number[] {
    const a = Array.from({ length: n }, (_, i) => i)
    this.shuffle(a)
    return a
  }
  uniform(lo = 0, hi = 1): number {
    return lo + (hi - lo) * this.double()
  }
  /** legacy_gauss: the polar method, which hands out its numbers in pairs */
  override normal(): number {
    if (this.gauss !== null) {
      const g = this.gauss
      this.gauss = null
      return g
    }
    for (;;) {
      const x1 = 2 * this.double() - 1
      const x2 = 2 * this.double() - 1
      const r2 = x1 * x1 + x2 * x2
      if (r2 < 1 && r2 !== 0) {
        const f = Math.sqrt((-2 * Math.log(r2)) / r2)
        this.gauss = f * x1
        return f * x2
      }
    }
  }
  /** choice(n, p=p): one index by inverting the cumulative distribution */
  choiceP(p: number[]): number {
    return searchRight(cumNormalised(p), this.double())
  }
  /** choice(n, size=k, replace=False, p=p) */
  choiceNoReplace(k: number, p0: number[]): number[] {
    const p = [...p0]
    const found: number[] = []
    while (found.length < k) {
      const draws = Array.from({ length: k - found.length }, () => this.double())
      for (const f of found) p[f] = 0
      const cdf = cumNormalised(p)
      const fresh = draws.map((x) => searchRight(cdf, x))
      // np.unique(new, return_index=True), back in the order of first appearance
      fresh.filter((v, i) => fresh.indexOf(v) === i).forEach((v) => found.push(v))
    }
    return found
  }
  /** RandomState.choice(n, size, replace, p) as indices: what each case draws, in NumPy's order */
  choice(n: number, count: number, replace: boolean, p: number[] | null): number[] {
    if (replace) {
      if (!p) return Array.from({ length: count }, () => this.interval(n - 1))
      const cdf = cumNormalised(p)
      return Array.from({ length: count }, () => searchRight(cdf, this.double()))
    }
    if (p) return this.choiceNoReplace(count, p)
    return this.permutation(n).slice(0, count)
  }
}

/** check_random_state, as sklearn and pandas apply it: an integer seeds a fresh generator, None is np.random's global one,
 * and a RandomState is itself (so drawing from it moves it on) */
export const randomStateOf = (h: Host, v: Value | undefined): LegacyRandom => {
  if (!v || v.k === 'none') return h.npRng
  if (v.k === 'int' || v.k === 'bool' || (v.k === 'array' && v.a.ndim === 0)) {
    const s = h.toInt(v)
    if (s < 0 || s > 4294967295) throw h.err('ValueError', 'Seed must be between 0 and 2**32 - 1')
    return new LegacyRandom(s)
  }
  const r = v.k === 'obj' ? (v.o as { r?: unknown }).r : null
  if (r instanceof LegacyRandom) return r
  throw h.err('ValueError', `${repr(v)} cannot be used to seed a numpy.random.RandomState instance`)
}
