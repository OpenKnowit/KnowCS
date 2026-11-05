/**
 * matplotlib's tick machinery as pure functions (ported from matplotlib 3.10 ticker.py and transforms.py), so tick
 * positions and limits come out number for number: nonsingular, MaxNLocator, LogLocator, AutoMinorLocator,
 * ScalarFormatter, LogFormatterSciNotation.
 */

const TINY = 2.2250738585072014e-308

export const nonsingular = (a: number, b: number, expander = 0.001, tiny = 1e-15): [number, number] => {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return [-expander, expander]
  let vmin = Math.min(a, b)
  let vmax = Math.max(a, b)
  const m = Math.max(Math.abs(vmin), Math.abs(vmax))
  if (m < (1e6 / tiny) * TINY) {
    vmin = -expander
    vmax = expander
  } else if (vmax - vmin <= m * tiny) {
    if (vmax === 0 && vmin === 0) {
      vmin = -expander
      vmax = expander
    } else {
      vmin -= expander * Math.abs(vmin)
      vmax += expander * Math.abs(vmax)
    }
  }
  return [vmin, vmax]
}

/** Python's float divmod, which differs from Math.floor(a / b) right at multiples */
const divmod = (a: number, b: number): [number, number] => {
  let mod = a % b
  let div = (a - mod) / b
  if (mod !== 0 && mod < 0 !== b < 0) {
    mod += b
    div -= 1
  }
  if (div === 0) return [0, mod]
  let fl = Math.floor(div)
  if (div - fl > 0.5) fl += 1
  return [fl, mod]
}

const edgeClose = (ms: number, edge: number, step: number, offset: number) => {
  let tol = 1e-10
  if (offset > 0) tol = Math.min(0.4999, Math.max(1e-10, 10 ** (Math.log10(offset / step) - 12)))
  return Math.abs(ms - edge) < tol
}
const edgeLe = (x: number, step: number, offset: number) => {
  const [d, m] = divmod(x, step)
  return edgeClose(m / step, 1, step, offset) ? d + 1 : d
}
const edgeGe = (x: number, step: number, offset: number) => {
  const [d, m] = divmod(x, step)
  return edgeClose(m / step, 0, step, offset) ? d : d + 1
}

export interface MaxN {
  nbins: number | 'auto'
  steps: number[]
  integer: boolean
  minN: number
  prune: 'upper' | 'lower' | 'both' | null
  symmetric: boolean
  /** 3D axes ask for a slightly smaller step (23/24), to keep matplotlib 3.8's look */
  rawScale?: number
}

/** AutoLocator's settings */
export const AUTO: MaxN = { nbins: 'auto', steps: [1, 2, 2.5, 5, 10], integer: false, minN: 2, prune: null, symmetric: false }
/** MaxNLocator's defaults when constructed directly */
export const MAXN_STEPS = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]

export const validateSteps = (steps: number[]): number[] => {
  const s = [...steps]
  if (s[0] !== 1) s.unshift(1)
  if (s[s.length - 1] !== 10) s.push(10)
  return s
}

const rawTicks = (vmin: number, vmax: number, m: MaxN, space: number): number[] => {
  const nbins = m.nbins === 'auto' ? Math.min(9, Math.max(Math.max(1, m.minN - 1), space)) : m.nbins
  const dv = Math.abs(vmax - vmin)
  const meanv = (vmax + vmin) / 2
  const offset = Math.abs(meanv) / dv < 100 ? 0 : Math.sign(meanv) * 10 ** divmod(Math.log10(Math.abs(meanv)), 1)[0]
  const scale = 10 ** divmod(Math.log10(dv / nbins), 1)[0]
  const lo = vmin - offset
  const hi = vmax - offset
  const st = m.steps
  let steps = [...st.slice(0, -1).map((s) => s * 0.1), ...st, 10 * st[1]].map((s) => s * scale)
  if (m.integer) steps = steps.filter((s) => s < 1 || Math.abs(s - Math.round(s)) < 0.001)
  const raw = ((hi - lo) / nbins) * (m.rawScale ?? 1)
  let istep = steps.findIndex((s) => s >= raw)
  if (istep < 0) istep = steps.length - 1
  let ticks: number[] = []
  for (let k = istep; k >= 0; k--) {
    let step = steps[k]
    if (m.integer && Math.floor(hi) - Math.ceil(lo) >= m.minN - 1) step = Math.max(1, step)
    const best = divmod(lo, step)[0] * step
    const low = edgeLe(lo - best, step, Math.abs(offset))
    const high = edgeGe(hi - best, step, Math.abs(offset))
    ticks = []
    for (let i = low; i < high + 1; i++) ticks.push(i * step + best)
    if (ticks.filter((t) => t <= hi && t >= lo).length >= m.minN) break
  }
  return ticks.map((t) => t + offset)
}

export const maxNTicks = (a: number, b: number, m: MaxN, space = 9): number[] => {
  let [lo, hi] = [a, b]
  if (m.symmetric) {
    hi = Math.max(Math.abs(lo), Math.abs(hi))
    lo = -hi
  }
  ;[lo, hi] = nonsingular(lo, hi, 1e-13, 1e-14)
  const locs = rawTicks(lo, hi, m, space)
  if (m.prune === 'lower') return locs.slice(1)
  if (m.prune === 'upper') return locs.slice(0, -1)
  if (m.prune === 'both') return locs.slice(1, -1)
  return locs
}

export const multipleTicks = (a: number, b: number, base: number, offset: number): number[] => {
  const hi0 = Math.max(a, b) - offset
  const lo = edgeGe(Math.min(a, b) - offset, base, 0) * base
  const n = divmod(hi0 - lo + 0.001 * base, base)[0]
  const out: number[] = []
  for (let i = 0; i < n + 3; i++) out.push(lo - base + i * base + offset)
  return out
}

export const linearTicks = (a: number, b: number, n: number): number[] => {
  let [lo, hi] = nonsingular(a, b, 0.05)
  if (a > b) [lo, hi] = [hi, lo]
  if (n === 0) return []
  if (n === 1) return [lo]
  const step = (hi - lo) / (n - 1)
  return Array.from({ length: n }, (_, i) => (i === n - 1 ? hi : lo + i * step))
}

/** LogLocator.tick_values; subs 'auto' is the default minor locator, null the major one */
export const logTicks = (a: number, b: number, base: number, subs: number[] | 'auto' | null, space: number, minpos: number): number[] => {
  const numticks = Math.min(9, Math.max(2, space))
  let vmin = a
  let vmax = b
  if (vmin <= 0) vmin = minpos
  if (vmin <= 0 || !Number.isFinite(vmin)) return []
  if (vmax < vmin) [vmin, vmax] = [vmax, vmin]
  const lmin = Math.log(vmin) / Math.log(base)
  const lmax = Math.log(vmax) / Math.log(base)
  const numdec = Math.floor(lmax) - Math.ceil(lmin)
  let sub: number[]
  if (subs === 'auto') {
    if (numdec > 10 || base < 3) return []
    sub = []
    for (let s = 2; s < base; s++) sub.push(s)
  } else sub = subs ?? [1]
  let stride = divmod(numdec, numticks)[0] + 1
  if (stride >= numdec) stride = Math.max(1, numdec - 1)
  const haveSubs = sub.length > 1 || (sub.length === 1 && sub[0] !== 1)
  const decades: number[] = []
  for (let d = Math.floor(lmin) - stride; d < Math.ceil(lmax) + 2 * stride; d += stride) decades.push(d)
  let ticks: number[]
  if (haveSubs) ticks = stride === 1 ? decades.flatMap((d) => sub.map((s) => s * base ** d)) : []
  else ticks = decades.map((d) => base ** d)
  if (sub.length > 1 && stride === 1 && ticks.filter((t) => vmin <= t && t <= vmax).length <= 1) return maxNTicks(vmin, vmax, AUTO, 9)
  return ticks
}

export const logNonsingular = (a: number, b: number, base: number, minpos: number): [number, number] => {
  let [vmin, vmax] = a > b ? [b, a] : [a, b]
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax) || vmax <= 0) return [1, 10]
  const mp = Number.isFinite(minpos) ? minpos : 1e-300
  if (vmin <= 0) vmin = mp
  if (vmin === vmax) {
    const lower = base ** Math.floor(Math.log(vmin) / Math.log(base))
    const upper = base ** Math.ceil(Math.log(vmax) / Math.log(base))
    vmin = lower === vmin ? lower / base : lower
    vmax = upper === vmax ? upper * base : upper
  }
  return [vmin, vmax]
}

/** AutoMinorLocator */
export const autoMinor = (major: number[], a: number, b: number, ndivs: number | null = null): number[] => {
  const m = [...new Set(major)].sort((p, q) => p - q)
  if (m.length < 2) return []
  const stepM = m[1] - m[0]
  let n = ndivs
  if (n === null) {
    const l = Math.log10(stepM)
    const mant = 10 ** (l - Math.floor(l))
    n = [1, 2.5, 5, 10].some((c) => Math.abs(mant - c) <= 1e-8 + 1e-5 * c) ? 5 : 4
  }
  const minor = stepM / n
  const vmin = Math.min(a, b)
  const vmax = Math.max(a, b)
  const tmin = pyRound((vmin - m[0]) / minor)
  const tmax = pyRound((vmax - m[0]) / minor) + 1
  const out: number[] = []
  for (let i = tmin; i < tmax; i++) out.push(i * minor + m[0])
  return out
}

/** round half to even */
export const pyRound = (x: number): number => {
  const r = Math.round(x)
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r
}

const MINUS = '−'
const fixMinus = (s: string) => s.replace(/-/g, MINUS)

const ptp = (xs: number[]) => Math.max(...xs) - Math.min(...xs)

/** ScalarFormatter.set_locs + format_ticks: the tick strings and the offset text ('1e6', '+1.5e3') */
export const scalarLabels = (locs: number[], view: [number, number], opts: { useOffset?: boolean; limits?: [number, number] | null; sci?: boolean } = {}): { labels: string[]; offset: string } => {
  if (!locs.length) return { labels: [], offset: '' }
  const [vmin, vmax] = view[0] < view[1] ? view : [view[1], view[0]]
  const shown = locs.filter((l) => vmin <= l && l <= vmax)
  let offset = 0
  if (shown.length && opts.useOffset !== false) {
    const lmin = Math.min(...shown)
    const lmax = Math.max(...shown)
    if (!(lmin === lmax || (lmin <= 0 && 0 <= lmax))) {
      const absMin = Math.min(Math.abs(lmin), Math.abs(lmax))
      const absMax = Math.max(Math.abs(lmin), Math.abs(lmax))
      const sign = lmin < 0 ? -1 : 1
      const fd = (x: number, o: number) => divmod(x, 10 ** o)[0]
      const oomMax = Math.ceil(Math.log10(absMax))
      let oom = oomMax
      while (fd(absMin, oom) === fd(absMax, oom)) oom--
      oom += 1
      if ((absMax - absMin) / 10 ** oom <= 1e-2) {
        oom = oomMax
        while (!(fd(absMax, oom) - fd(absMin, oom) > 1)) oom--
        oom += 1
      }
      if (fd(absMax, oom) >= 1000) offset = sign * fd(absMax, oom) * 10 ** oom
    }
  }
  let order = 0
  const [limLo, limHi] = opts.limits ?? [-5, 6]
  if (opts.sci === false) order = 0
  else if (limLo === limHi && limLo !== 0) order = limLo
  else if (shown.length) {
    let oom: number
    if (offset) oom = Math.floor(Math.log10(vmax - vmin))
    else {
      const val = Math.max(...shown.map(Math.abs))
      oom = val === 0 ? 0 : Math.floor(Math.log10(val))
    }
    if (oom <= limLo || oom >= limHi) order = oom
  }
  const scale = 10 ** order
  const raw = locs.length < 2 ? [...locs, ...view] : locs
  const all = raw.map((l) => (l - offset) / scale)
  let range = ptp(all)
  if (range === 0) range = Math.max(...all.map(Math.abs))
  if (range === 0) range = 1
  const pts = locs.length < 2 ? all.slice(0, -2) : all
  const rangeOom = Math.floor(Math.log10(range))
  let sig = Math.max(0, 3 - rangeOom)
  const thresh = 1e-3 * 10 ** rangeOom
  while (sig >= 0) {
    const p = 10 ** sig
    if (Math.max(...pts.map((v) => Math.abs(v - rint(v * p) / p))) < thresh) sig--
    else break
  }
  sig += 1
  const labels = locs.map((l) => {
    let xp = (l - offset) / scale
    if (Math.abs(xp) < 1e-8) xp = 0
    return fixMinus(xp.toFixed(sig))
  })
  let off = ''
  if (order || offset) {
    let s = order ? `1e${order}` : ''
    if (offset) s += (offset > 0 ? '+' : '') + shortG(offset)
    off = fixMinus(s)
  }
  return { labels, offset: off }
}

const rint = (x: number) => {
  const r = Math.round(x)
  return Math.abs(x - Math.trunc(x)) === 0.5 && r % 2 !== 0 ? r - 1 : r
}

/** '%g' */
const shortG = (x: number): string => {
  if (x === 0) return '0'
  const e = Math.floor(Math.log10(Math.abs(x)))
  if (e < -4 || e >= 6) {
    const [m, ex] = x.toExponential(5).split('e')
    const n = Number(ex)
    return `${String(Number(m))}e${n < 0 ? '-' : '+'}${String(Math.abs(n)).padStart(2, '0')}`
  }
  return String(Number(x.toPrecision(6)))
}
export const formatG = shortG

/** LogFormatterSciNotation for ticks (mathtext, as matplotlib returns it) */
export const logLabels = (locs: number[], view: [number, number], base: number): string[] => {
  const lo = Math.min(...view)
  const hi = Math.max(...view)
  const sub = (() => {
    if (lo <= 0) return new Set([1])
    const numdec = Math.abs(Math.log(hi) / Math.log(base) - Math.log(lo) / Math.log(base))
    if (numdec > 1) return new Set([1])
    if (numdec > 0.4) return new Set([1, 2, 3, 4, 6, 10])
    return new Set(Array.from({ length: base }, (_, i) => i + 1))
  })()
  return locs.map((x0) => {
    if (x0 === 0) return '$\\mathdefault{0}$'
    const sign = x0 < 0 ? '-' : ''
    const x = Math.abs(x0)
    const fx = Math.log(x) / Math.log(base)
    const isDecade = Math.abs(fx - Math.round(fx)) < 1e-10
    const exponent = isDecade ? Math.round(fx) : Math.floor(fx)
    const coeff = Math.round(base ** (fx - exponent))
    if (!sub.has(coeff)) return ''
    if (isDecade) return `$\\mathdefault{${sign}${base}^{${Math.round(fx)}}}$`
    const c = base ** (fx - exponent)
    const cs = Math.abs(c - Math.round(c)) < 1e-10 ? Math.round(c) : c
    return `$\\mathdefault{${sign}${shortG(cs)}\\times${base}^{${exponent}}}$`
  })
}
