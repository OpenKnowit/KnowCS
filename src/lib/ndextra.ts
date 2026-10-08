// --- More NumPy for the sandbox: counting, histograms, order statistics, grids, differences, padding, membership ---
// Each function returns its result and, where a result cell comes from a group of input cells, that group
// (groups[o] = [operand, flat index] cells), so the API panel can show where every number came from.
import { NDArray, PyError, cStrides, normAxis, prod, transpose, unravel } from './ndarray'

export type Cell = [number, number]
export interface Grouped { out: NDArray; groups: Cell[][] }

const ints = (a: NDArray, what: string): number[] => {
  const v = a.values()
  if (a.dtype === 'float64' && v.some((x) => !Number.isInteger(x))) throw new PyError('TypeError', `Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe' (${what})`)
  return v
}

/** np.bincount: result[k] = how many times k occurs (or the sum of the weights where it occurs) */
export function bincount(a: NDArray, minlength = 0, weights: NDArray | null = null): Grouped {
  if (a.ndim !== 1) throw new PyError('ValueError', 'object too deep for desired array')
  const v = ints(a, 'bincount')
  if (v.some((x) => x < 0)) throw new PyError('ValueError', "'list' argument must have no negative elements")
  if (weights && weights.size !== a.size) throw new PyError('ValueError', 'The weights and list don\'t have the same length.')
  const n = Math.max(minlength, v.length ? Math.max(...v) + 1 : 0)
  const w = weights?.values()
  const out = new Array(n).fill(0)
  const groups: Cell[][] = Array.from({ length: n }, () => [])
  v.forEach((x, f) => {
    out[x] += w ? w[f] : 1
    groups[x].push([0, f])
  })
  return { out: NDArray.create(out, [n], w ? 'float64' : 'int64'), groups }
}

/** np.histogram: counts per bin (the last bin includes its right edge) and the bin edges */
export function histogram(a: NDArray, bins = 10, range: [number, number] | null = null): Grouped & { edges: NDArray } {
  if (bins < 1) throw new PyError('ValueError', '`bins` must be positive, when an integer')
  const v = a.values()
  let [lo, hi] = range ?? (v.length ? [Math.min(...v), Math.max(...v)] : [0, 1])
  if (lo > hi) throw new PyError('ValueError', 'max must be larger than min in range parameter.')
  if (lo === hi) [lo, hi] = [lo - 0.5, hi + 0.5]
  const edges = Array.from({ length: bins + 1 }, (_, i) => lo + ((hi - lo) * i) / bins)
  const out = new Array(bins).fill(0)
  const groups: Cell[][] = Array.from({ length: bins }, () => [])
  v.forEach((x, f) => {
    if (x < lo || x > hi || Number.isNaN(x)) return
    let b = Math.floor(((x - lo) / (hi - lo)) * bins)
    if (b >= bins) b = bins - 1
    // floating-point edges: put x in the bin whose edges really contain it
    while (b > 0 && x < edges[b]) b--
    while (b < bins - 1 && x >= edges[b + 1]) b++
    out[b]++
    groups[b].push([0, f])
  })
  return { out: NDArray.create(out, [bins], 'int64'), groups, edges: NDArray.create(edges, [bins + 1], 'float64') }
}

/** apply f to each lane along axis (null = the whole array): the result drops that axis */
function alongAxis(a: NDArray, axis: number | null, f: (xs: number[]) => number): NDArray | number {
  if (axis === null) return f(a.values())
  const ax = normAxis(axis, a.ndim)
  const rest = a.shape.map((_, i) => i).filter((i) => i !== ax)
  const vals = transpose(a, [...rest, ax]).values()
  const n = a.shape[ax]
  const outShape = rest.map((i) => a.shape[i])
  return NDArray.create(Array.from({ length: prod(outShape) }, (_, i) => f(vals.slice(i * n, i * n + n))), outShape, 'float64')
}

/** the q-th percentile of xs (0–100), NumPy's default linear interpolation; NaN anywhere gives NaN */
export const percentileOf = (xs: number[], q: number): number => {
  if (q < 0 || q > 100) throw new PyError('ValueError', 'Percentiles must be in the range [0, 100]')
  if (!xs.length || xs.some(Number.isNaN)) return NaN
  const s = [...xs].sort((x, y) => x - y)
  const pos = ((s.length - 1) * q) / 100
  const i = Math.floor(pos)
  return i + 1 < s.length ? s[i] + (s[i + 1] - s[i]) * (pos - i) : s[i]
}

export const median = (a: NDArray, axis: number | null) => alongAxis(a, axis, (xs) => percentileOf(xs, 50))
export const percentile = (a: NDArray, q: number, axis: number | null) => alongAxis(a, axis, (xs) => percentileOf(xs, q))

/** np.average: the mean, or Σ w·x / Σ w with weights (same shape as a, or 1-D along axis) */
export function average(a: NDArray, axis: number | null, weights: NDArray | null): NDArray | number {
  if (!weights) return alongAxis(a, axis, (xs) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : NaN))
  const w = weights.values()
  if (weights.size === a.size && (axis === null || weights.ndim === a.ndim)) {
    if (axis === null) {
      const sw = w.reduce((s, x) => s + x, 0)
      if (sw === 0) throw new PyError('ZeroDivisionError', 'Weights sum to zero, can\'t be normalized')
      return a.values().reduce((s, x, i) => s + x * w[i], 0) / sw
    }
  }
  if (axis === null) throw new PyError('TypeError', 'Axis must be specified when shapes of a and weights differ.')
  const ax = normAxis(axis, a.ndim)
  if (weights.ndim !== 1 || weights.size !== a.shape[ax]) throw new PyError('ValueError', 'Length of weights not compatible with specified axis.')
  const sw = w.reduce((s, x) => s + x, 0)
  if (sw === 0) throw new PyError('ZeroDivisionError', 'Weights sum to zero, can\'t be normalized')
  return alongAxis(a, ax, (xs) => xs.reduce((s, x, i) => s + x * w[i], 0) / sw)
}

/** np.meshgrid(x, y) (indexing='xy'): X repeats x along the rows, Y repeats y along the columns */
export function meshgrid(x: NDArray, y: NDArray): { X: Grouped; Y: Grouped } {
  const xv = x.values()
  const yv = y.values()
  const shape = [yv.length, xv.length]
  return {
    X: { out: NDArray.create(yv.flatMap(() => xv), shape, x.dtype), groups: yv.flatMap(() => xv.map((_, j) => [[0, j]] as Cell[])) },
    Y: { out: NDArray.create(yv.flatMap((v) => xv.map(() => v)), shape, y.dtype), groups: yv.flatMap((_, i) => xv.map(() => [[1, i]] as Cell[])) },
  }
}

/** np.diff (n = 1): neighbouring differences along axis (default: the last one) */
export function diff(a: NDArray, axis = -1): Grouped {
  if (a.ndim === 0) throw new PyError('ValueError', 'diff requires input that is at least one dimensional')
  const ax = normAxis(axis, a.ndim)
  const shape = a.shape.map((d, i) => (i === ax ? Math.max(0, d - 1) : d))
  const st = cStrides(a.shape)
  const v = a.values()
  const out: number[] = []
  const groups: Cell[][] = []
  for (let o = 0; o < prod(shape); o++) {
    const idx = unravel(o, shape)
    const f = idx.reduce((s, x, k) => s + x * st[k], 0)
    out.push(v[f + st[ax]] - v[f])
    groups.push([[0, f], [0, f + st[ax]]])
  }
  return { out: NDArray.create(out, shape, a.dtype === 'bool' ? 'bool' : a.dtype), groups }
}

/** pad_width as NumPy reads it: n, (before, after), or one (before, after) pair per axis */
export function padWidths(spec: number | number[] | number[][], ndim: number): [number, number][] {
  if (typeof spec === 'number') return Array.from({ length: ndim }, () => [spec, spec])
  if (spec.length && typeof spec[0] === 'number') {
    const s = spec as number[]
    if (s.length === 1) return Array.from({ length: ndim }, () => [s[0], s[0]])
    if (s.length === 2) return Array.from({ length: ndim }, () => [s[0], s[1]])
    throw new PyError('ValueError', `operands could not be broadcast together with remapped shapes [original->remapped]: (${s.length},) and requested shape (${ndim},2)`)
  }
  const p = spec as number[][]
  if (p.length === 1) return Array.from({ length: ndim }, () => [p[0][0], p[0][1] ?? p[0][0]])
  if (p.length !== ndim) throw new PyError('ValueError', `operands could not be broadcast together with remapped shapes [original->remapped]: (${p.length},2) and requested shape (${ndim},2)`)
  return p.map((q) => [q[0], q[1] ?? q[0]])
}

/** np.pad with mode='constant': a border of `value` around the array (the zero padding of a convolution) */
export function pad(a: NDArray, widths: [number, number][], value = 0): Grouped {
  if (widths.some(([b, e]) => b < 0 || e < 0)) throw new PyError('ValueError', 'index can\'t contain negative values')
  const shape = a.shape.map((d, i) => d + widths[i][0] + widths[i][1])
  const st = cStrides(a.shape)
  const v = a.values()
  const out: number[] = []
  const groups: Cell[][] = []
  for (let o = 0; o < prod(shape); o++) {
    const idx = unravel(o, shape).map((x, k) => x - widths[k][0])
    if (idx.every((x, k) => x >= 0 && x < a.shape[k])) {
      const f = idx.reduce((s, x, k) => s + x * st[k], 0)
      out.push(v[f])
      groups.push([[0, f]])
    } else {
      out.push(value)
      groups.push([])
    }
  }
  return { out: NDArray.create(out, shape, a.dtype), groups }
}

/** np.isin: is each element of a among the test values? (each result cell also points at its match) */
export function isin(a: NDArray, test: NDArray): Grouped {
  const tv = test.values()
  const groups: Cell[][] = []
  const out = a.values().map((x, f) => {
    const hits = tv.flatMap((y, g) => (y === x ? [[1, g] as Cell] : []))
    groups.push([[0, f], ...hits])
    return hits.length ? 1 : 0
  })
  return { out: NDArray.create(out, a.shape, 'bool'), groups }
}

/** np.allclose with NumPy's defaults: |a − b| ≤ atol + rtol·|b| everywhere (after broadcasting) */
export const allclose = (a: number[], b: number[], rtol = 1e-5, atol = 1e-8): boolean =>
  a.length === b.length && a.every((x, i) => (Number.isNaN(x) || Number.isNaN(b[i]) ? false : x === b[i] || Math.abs(x - b[i]) <= atol + rtol * Math.abs(b[i])))
