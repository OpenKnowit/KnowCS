/**
 * scikit-learn for the sandbox: the estimators, preprocessing, model selection and metrics of an intro ML course.
 * Randomness runs on a port of NumPy's legacy RandomState (MT19937), so a given random_state gives real sklearn's answer:
 * the same split, the same k-means++ seeds, the same Perceptron shuffle. With random_state=None the seed comes from
 * np.random.seed instead, which cannot match NumPy's global stream.
 *
 * NDArray has no text dtype, so labels and category names live in ObjArray, which prints the way NumPy prints them.
 */
import { NDArray, pyFloatRepr } from './ndarray'
import { PyDict, PyObj, isNum, numOf, py, repr, toArray } from './minipy'
import type { ApiKind, ClassValue, Host, Kw, PyLib, Value } from './minipy'
import { Frame, Series } from './pyPandas'
import { solveM, svdThin, transposeM } from './npLinalg'

type Fn = (args: Value[], kw: Kw) => Value
type Lab = number | string

// ---------------------------------------------------------------- numpy.random.RandomState

class RandomState {
  private key = new Uint32Array(624)
  private pos = 624
  private gauss: number | null = null
  constructor(seed: number) {
    let s = seed >>> 0
    for (let i = 0; i < 624; i++) {
      this.key[i] = s
      s = (Math.imul(1812433253, s ^ (s >>> 30)) + i + 1) >>> 0
    }
  }
  next32(): number {
    if (this.pos === 624) {
      const k = this.key
      for (let i = 0; i < 624; i++) {
        const y = (k[i] & 0x80000000) | (k[(i + 1) % 624] & 0x7fffffff)
        k[i] = k[(i + 397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0)
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
  double(): number {
    const a = this.next32() >>> 5
    const b = this.next32() >>> 6
    return (a * 67108864 + b) / 9007199254740992
  }
  /** random_interval: uniform on 0..max by rejection through a bit mask */
  interval(max: number): number {
    if (max === 0) return 0
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
  normal(): number {
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
}

const cumNormalised = (p: number[]): number[] => {
  let c = 0
  const out = p.map((x) => (c += x))
  const last = out[out.length - 1]
  return out.map((x) => x / last)
}
const searchRight = (sorted: number[], x: number): number => {
  let lo = 0
  let hi = sorted.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (x < sorted[mid]) hi = mid
    else lo = mid + 1
  }
  return lo
}
const searchLeft = (sorted: number[], x: number): number => {
  let lo = 0
  let hi = sorted.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (sorted[mid] < x) lo = mid + 1
    else hi = mid
  }
  return lo
}

/** check_random_state: an integer seeds a fresh generator; None takes its seed from the sandbox's np.random */
const rngOf = (h: Host, v: Value | undefined): RandomState => {
  if (!v || v.k === 'none') return new RandomState(Math.floor(h.rng.next() * 4294967296))
  if (v.k === 'int' || (v.k === 'array' && v.a.ndim === 0)) {
    const s = h.toInt(v)
    if (s < 0 || s > 4294967295) throw h.err('ValueError', 'Seed must be between 0 and 2**32 - 1')
    return new RandomState(s)
  }
  // np.random.RandomState is the sandbox's own stream here, so it only hands over a seed
  if (v.k === 'obj' && v.o.cls === 'RandomState') return new RandomState(h.toInt(h.call(h.getAttr(v, 'randint'), [py.int(2147483647)])))
  throw h.err('ValueError', `${repr(v)} cannot be used to seed a numpy.random.RandomState instance`)
}

/** our_rand_r: the xorshift sklearn's Cython code draws from */
const ourRandR = (st: { s: number }): number => {
  if (st.s === 0) st.s = 1
  st.s = (st.s ^ (st.s << 13)) >>> 0
  st.s = (st.s ^ (st.s >>> 17)) >>> 0
  st.s = (st.s ^ (st.s << 5)) >>> 0
  return st.s % 2147483648
}

const MAX_INT = 2147483647

// ---------------------------------------------------------------- numbers

/** np.add.reduce on a 1-D array: pairwise summation, which sets the rounding of every printed sum and mean */
const pairwise = (a: number[], lo: number, n: number): number => {
  if (n < 8) {
    let r = -0
    for (let i = 0; i < n; i++) r += a[lo + i]
    return r
  }
  if (n <= 128) {
    const r = [a[lo], a[lo + 1], a[lo + 2], a[lo + 3], a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]]
    let i = 8
    for (; i < n - (n % 8); i += 8) for (let k = 0; k < 8; k++) r[k] += a[lo + i + k]
    let res = r[0] + r[1] + (r[2] + r[3]) + (r[4] + r[5] + (r[6] + r[7]))
    for (; i < n; i++) res += a[lo + i]
    return res
  }
  let n2 = Math.floor(n / 2)
  n2 -= n2 % 8
  return pairwise(a, lo, n2) + pairwise(a, lo + n2, n - n2)
}
const npSum = (a: number[]): number => 0 + pairwise(a, 0, a.length)
const npMean = (a: number[]): number => npSum(a) / a.length
const dot = (a: number[], b: number[]): number => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}
const colsOf = (rows: number[][], m: number): number[][] => Array.from({ length: m }, (_, j) => rows.map((r) => r[j]))
/** X.sum(axis=0): whole rows at a time, so plain running sums; a single column is one long inner loop, so pairwise */
const sumAxis0 = (rows: number[][], m: number): number[] => {
  if (m === 1) return [npSum(rows.map((r) => r[0]))]
  const s = new Array<number>(m).fill(0)
  for (const r of rows) for (let j = 0; j < m; j++) s[j] += r[j]
  return s
}
const meanAxis0 = (rows: number[][], m: number): number[] => sumAxis0(rows, m).map((x) => x / rows.length)
const varAxis0 = (rows: number[][], m: number): number[] => {
  const mu = meanAxis0(rows, m)
  return sumAxis0(rows.map((r) => r.map((x, j) => (x - mu[j]) ** 2)), m).map((x) => x / rows.length)
}
const argmax = (a: number[]): number => {
  let k = 0
  for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i
  return k
}
const logsumexp = (a: number[]): number => {
  const k = argmax(a)
  const m = a[k]
  if (!Number.isFinite(m)) return m
  let s = 0
  a.forEach((x, i) => { if (i !== k) s += Math.exp(x - m) })
  return Math.log1p(s) + m
}

const mat = (rows: number[][], m?: number, dtype: 'float64' | 'int64' = 'float64'): Value => py.arr(NDArray.create(rows.flat(), [rows.length, m ?? rows[0]?.length ?? 0], dtype))
const vec = (xs: number[], dtype: 'float64' | 'int64' | 'bool' = 'float64'): Value => py.arr(NDArray.create(xs, [xs.length], dtype))
const rowsOf = (a: NDArray): number[][] => {
  const v = a.values()
  const m = a.shape[1]
  return Array.from({ length: a.shape[0] }, (_, i) => v.slice(i * m, (i + 1) * m))
}
const dictOf = (entries: [string, Value][]): Value => ({ k: 'dict', d: PyDict.from(entries.map(([k, v]) => [py.str(k), v])) })

// ---------------------------------------------------------------- text arrays

const reprStr = (s: string) => repr(py.str(s))

/** an array of strings (dtype <U.. or object): NDArray only holds numbers */
export class ObjArray extends PyObj {
  readonly cls = 'ndarray'
  items: string[]
  shape: number[]
  dtype: string
  constructor(items: string[], dtype: string | null = null, shape: number[] | null = null) {
    super()
    this.items = items
    this.shape = shape ?? [items.length]
    this.dtype = dtype ?? uDtype(items)
  }
  private body(sep: string, lead: number): string {
    const line = (xs: string[], indent: number) => {
      const lines: string[] = []
      let cur = ''
      xs.forEach((w, i) => {
        const word = i < xs.length - 1 ? w + sep : w
        const next = cur ? `${cur} ${word}` : word
        if (cur && indent + next.length + (i === xs.length - 1 ? 1 : 0) > 75) {
          lines.push(cur)
          cur = word
        } else cur = next
      })
      lines.push(cur)
      return `[${lines.join(`\n${' '.repeat(indent)}`)}]`
    }
    const words = this.items.map(reprStr)
    if (this.shape.length === 1) return line(words, lead + 1)
    const m = this.shape[1]
    const rows = Array.from({ length: this.shape[0] }, (_, i) => line(words.slice(i * m, (i + 1) * m), lead + 2))
    return `[${rows.join(`${sep}\n${' '.repeat(lead + 1)}`)}]`
  }
  repr() {
    const tail = this.dtype === 'object' ? 'dtype=object' : `dtype='${this.dtype}'`
    if (!this.items.length) return `array([], ${tail})`
    const body = this.body(',', 6)
    const last = body.split('\n').pop()!
    return last.length + tail.length + 3 > 75 ? `array(${body},\n      ${tail})` : `array(${body}, ${tail})`
  }
  str() { return this.body('', 0) }
  len() { return this.shape[0] }
  iter() {
    if (this.shape.length === 1) return this.items.map((s) => py.str(s))
    const m = this.shape[1]
    return Array.from({ length: this.shape[0] }, (_, i) => py.obj(new ObjArray(this.items.slice(i * m, (i + 1) * m), this.dtype)))
  }
  contains(v: Value) { return v.k === 'str' && this.items.includes(v.v) }
  take(rows: number[]): ObjArray {
    if (this.shape.length === 1) return new ObjArray(rows.map((r) => this.items[r]), this.dtype)
    const m = this.shape[1]
    return new ObjArray(rows.flatMap((r) => this.items.slice(r * m, (r + 1) * m)), this.dtype, [rows.length, m])
  }
  getItem(idx: Value, h: Host): Value {
    const n = this.shape[0]
    if (idx.k === 'int' || (idx.k === 'array' && idx.a.ndim === 0 && idx.a.dtype === 'int64')) {
      const i0 = h.toInt(idx)
      const i = i0 < 0 ? i0 + n : i0
      if (i < 0 || i >= n) throw h.err('IndexError', `index ${i0} is out of bounds for axis 0 with size ${n}`)
      return this.shape.length === 1 ? py.str(this.items[i]) : py.obj(this.take([i]).flatten())
    }
    if (idx.k === 'slice') {
      const part = (x: Value, d: number) => (x.k === 'none' ? d : h.toInt(x))
      const step = part(idx.step, 1)
      const norm = (i: number) => (i < 0 ? Math.max(i + n, step < 0 ? -1 : 0) : Math.min(i, step < 0 ? n - 1 : n))
      const rows: number[] = []
      const stop = idx.stop.k === 'none' ? (step < 0 ? -1 : n) : norm(h.toInt(idx.stop))
      for (let i = idx.start.k === 'none' ? (step < 0 ? n - 1 : 0) : norm(h.toInt(idx.start)); step < 0 ? i > stop : i < stop; i += step) rows.push(i)
      return py.obj(this.take(rows))
    }
    if (idx.k === 'list' || idx.k === 'array') {
      const a = toArray(idx)
      const rows = a.dtype === 'bool' ? a.values().flatMap((b, i) => (b ? [i] : [])) : a.values().map((i) => (i < 0 ? i + n : i))
      return py.obj(this.take(rows))
    }
    throw h.err('IndexError', 'only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices')
  }
  flatten() { return new ObjArray(this.items, this.dtype) }
  binop(op: string, other: Value, _reflected: boolean, h: Host): Value | undefined {
    if (op !== '==' && op !== '!=') return undefined
    const eq = op === '=='
    if (other.k === 'str') return vec(this.items.map((s) => Number((s === other.v) === eq)), 'bool')
    const o = labelsOf(h, other)
    if (o.vals.length !== this.items.length) throw h.err('ValueError', `operands could not be broadcast together with shapes (${this.items.length},) (${o.vals.length},)`)
    return vec(this.items.map((s, i) => Number((s === o.vals[i]) === eq)), 'bool')
  }
  isa(name: string) { return name === 'ndarray' }
  getAttr(name: string): Value | undefined {
    const fn = (call: Fn): Value => ({ k: 'fn', name, call })
    switch (name) {
      case 'shape': return py.tuple(this.shape.map((d) => py.int(d)))
      case 'size': return py.int(this.items.length)
      case 'ndim': return py.int(this.shape.length)
      case 'dtype': return py.str(this.dtype)
      case 'tolist': return fn(() => py.list(this.iter().map((v) => (v.k === 'obj' && v.o instanceof ObjArray ? py.list(v.o.items.map((s) => py.str(s))) : v))))
      case 'ravel': case 'flatten': return fn(() => py.obj(this.flatten()))
      case 'copy': return fn(() => py.obj(new ObjArray([...this.items], this.dtype, [...this.shape])))
      case 'reshape': return fn((args) => {
        const dims = (args.length === 1 && (args[0].k === 'tuple' || args[0].k === 'list') ? args[0].items : args).map((v) => numOf(v))
        const known = dims.filter((d) => d >= 0).reduce((p, d) => p * d, 1)
        const shape = dims.map((d) => (d < 0 ? this.items.length / known : d))
        return py.obj(new ObjArray([...this.items], this.dtype, shape))
      })
    }
    return undefined
  }
}
const uDtype = (items: string[]) => `<U${Math.max(1, ...items.map((s) => s.length))}`

/** scipy's csr_matrix, as OneHotEncoder hands it back */
class Csr extends PyObj {
  readonly cls = 'csr_matrix'
  rows: number[][]
  m: number
  constructor(rows: number[][], m: number) {
    super()
    this.rows = rows
    this.m = m
  }
  get nnz() { return this.rows.reduce((n, r) => n + r.filter((x) => x !== 0).length, 0) }
  repr() { return `<Compressed Sparse Row sparse matrix of dtype 'float64'\n\twith ${this.nnz} stored elements and shape (${this.rows.length}, ${this.m})>` }
  str() {
    const lines: string[] = []
    this.rows.forEach((r, i) => r.forEach((x, j) => { if (x !== 0) lines.push(`  (${i}, ${j})\t${pyFloatRepr(x)}`) }))
    return lines.length ? `${this.repr()}\n  Coords\tValues\n${lines.join('\n')}` : this.repr()
  }
  dense() { return NDArray.create(this.rows.flat(), [this.rows.length, this.m], 'float64') }
  take(rows: number[]) { return new Csr(rows.map((r) => this.rows[r]), this.m) }
  len(h: Host): number { throw h.err('TypeError', 'sparse array length is ambiguous; use getnnz() or shape[0]') }
  isa(name: string) { return name === 'csr_matrix' }
  getAttr(name: string): Value | undefined {
    switch (name) {
      case 'toarray': return { k: 'fn', name, call: () => py.arr(this.dense()) }
      case 'todense': return { k: 'fn', name, call: () => { const a = this.dense(); a.matrix = true; return py.arr(a) } }
      case 'shape': return py.tuple([py.int(this.rows.length), py.int(this.m)])
      case 'nnz': return py.int(this.nnz)
      case 'dtype': return py.str('float64')
      case 'ndim': return py.int(2)
    }
    return undefined
  }
}

// ---------------------------------------------------------------- reading inputs

/** a DataFrame's column labels as text, whether pandas keeps them in a list or an Index */
const colNames = (f: Frame): string[] => {
  const c = f.columns as unknown
  return (Array.isArray(c) ? c : (c as { labels: unknown[] }).labels).map(String)
}

/** an object array from another library (df['c'].values): anything that calls itself an ndarray but is not numeric */
const isForeignArray = (v: Value): boolean => v.k === 'obj' && v.o.cls === 'ndarray' && !(v.o instanceof ObjArray) && !v.o.toArray

/** sklearn's own exceptions, which an except ValueError also catches */
const excOf = (h: Host, name: string, bases: string[]): ClassValue => {
  const key = `sklearn.${name}`
  let c = h.state.get(key) as ClassValue | undefined
  if (!c) {
    const b = bases.map((n) => h.excClass(n))
    c = { k: 'class', name, bases: b, ns: new Map(), host: b[0].host, builtin: true }
    h.state.set(key, c)
  }
  return c
}
const notFittedClass = (h: Host) => excOf(h, 'NotFittedError', ['ValueError', 'AttributeError'])
const notFitted = (h: Host, name: string): never => h.raise(notFittedClass(h), [py.str(`This ${name} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.`)])
const badParam = (h: Host, msg: string): never => h.raise(excOf(h, 'InvalidParameterError', ['ValueError', 'TypeError']), [py.str(msg)])

const firstString = (v: Value): string | null => {
  if (v.k === 'str') return v.v
  if (v.k === 'list' || v.k === 'tuple') {
    for (const it of v.items) {
      const s = firstString(it)
      if (s !== null) return s
    }
  }
  if (v.k === 'obj' && v.o instanceof ObjArray) return v.o.items[0] ?? null
  return null
}

/** X as numbers, or the error check_array gives */
const numericArray = (h: Host, v: Value): NDArray => {
  if (v.k === 'obj' && v.o instanceof Csr) throw h.err('TypeError', "Sparse data was passed for X, but dense data is required. Use '.toarray()' to convert to a dense numpy array.")
  if (v.k === 'obj' && v.o instanceof Frame) {
    const bad = v.o.data.find((s) => s.dtype === 'object')
    if (bad) throw h.err('ValueError', `could not convert string to float: ${reprStr(String(bad.values.find((c) => typeof c === 'string') ?? ''))}`)
  }
  if (v.k === 'obj' && v.o instanceof Series && v.o.dtype === 'object') throw h.err('ValueError', `could not convert string to float: ${reprStr(String(v.o.values.find((c) => typeof c === 'string') ?? ''))}`)
  const s = firstString(v)
  if (s !== null) throw h.err('ValueError', `could not convert string to float: ${reprStr(s)}`)
  try {
    return toArray(v)
  } catch {
    throw h.err('TypeError', `Expected a numeric array-like, got ${repr(v).slice(0, 60)}`)
  }
}

const NAN_HELP = (est: string) => `${est} does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values`

const finiteCheck = (h: Host, xs: number[], name: string, est: string) => {
  let nan = false
  let inf = false
  for (const x of xs) {
    if (Number.isNaN(x)) nan = true
    else if (!Number.isFinite(x)) inf = true
  }
  if (nan) throw h.err('ValueError', `Input ${name} contains NaN.\n${NAN_HELP(est)}`)
  if (inf) throw h.err('ValueError', `Input ${name} contains infinity or a value too large for dtype('float64').`)
}

interface XData { rows: number[][]; n: number; m: number; names: string[] | null; a: NDArray }

/** these convert X to float before they look at it, which shows in the 1-D error */
const FLOAT_INPUT = new Set(['StandardScaler', 'MinMaxScaler', 'Normalizer', 'PolynomialFeatures', 'KMeans', 'Perceptron', 'LogisticRegression', 'Ridge', 'PCA', 'DecisionTreeClassifier'])

/** check_array for a 2-D sample matrix */
const readX = (h: Host, v: Value | undefined, est: string, name = 'X'): XData => {
  if (!v) throw h.err('TypeError', `${est}.fit() missing 1 required positional argument: '${name}'`)
  if (v.k === 'none') throw h.err('ValueError', `Expected 2D array, got scalar array instead:\narray=None.\nReshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.`)
  const a = numericArray(h, v)
  if (a.ndim === 0) throw h.err('ValueError', `Expected 2D array, got scalar array instead:\narray=${h.str(py.arr(a))}.\nReshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.`)
  if (a.ndim === 1) throw h.err('ValueError', `Expected 2D array, got 1D array instead:\narray=${h.str(py.arr(FLOAT_INPUT.has(est) ? NDArray.create(a.values(), a.shape, 'float64') : a))}.\nReshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.`)
  if (a.ndim > 2) throw h.err('ValueError', `Found array with dim ${a.ndim}, while dim <= 2 is required by ${est}.`)
  if (a.shape[0] === 0) throw h.err('ValueError', `Found array with 0 sample(s) (shape=(0, ${a.shape[1]})) while a minimum of 1 is required by ${est}.`)
  if (a.shape[1] === 0) throw h.err('ValueError', `Found array with 0 feature(s) (shape=(${a.shape[0]}, 0)) while a minimum of 1 is required by ${est}.`)
  finiteCheck(h, a.values(), name, est)
  return { rows: rowsOf(a), n: a.shape[0], m: a.shape[1], names: v.k === 'obj' && v.o instanceof Frame ? colNames(v.o) : null, a }
}

interface Labels { vals: Lab[]; isStr: boolean; dtype: string }

/** a target or label column, numeric or text */
const labelsOf = (h: Host, v: Value): Labels => {
  if (v.k === 'obj' && v.o instanceof ObjArray) {
    if (v.o.shape.length === 2 && v.o.shape[1] !== 1) throw h.err('ValueError', `y should be a 1d array, got an array of shape (${v.o.shape.join(', ')}) instead.`)
    return { vals: v.o.items, isStr: true, dtype: v.o.dtype }
  }
  if (v.k === 'obj' && v.o instanceof Series) {
    const s = v.o
    if (s.dtype === 'object') {
      const kinds = new Set(s.values.map((c) => (typeof c === 'string' ? 'str' : c === null ? 'NoneType' : typeof c === 'number' ? 'float' : 'bool')))
      if (kinds.size > 1) throw h.err('TypeError', `Encoders require their input argument must be uniformly strings or numbers. Got ${JSON.stringify([...kinds].sort()).replace(/"/g, "'").replace(/,/g, ', ')}`)
      return { vals: s.values as string[], isStr: true, dtype: 'object' }
    }
    return { vals: s.values.map((c) => Number(c)), isStr: false, dtype: s.dtype }
  }
  if (v.k === 'obj' && v.o instanceof Frame && v.o.columns.length === 1) return labelsOf(h, py.obj(v.o.data[0]))
  const items = v.k === 'list' || v.k === 'tuple' ? v.items : v.k === 'iter' || (v.k === 'obj' && v.o.iter && !v.o.toArray) ? h.iterate(v) : null
  if (items && items.some((x) => x.k === 'str')) {
    if (items.some((x) => x.k === 'list' || x.k === 'tuple')) throw h.err('ValueError', 'y should be a 1d array of labels')
    const vals = items.map((x) => h.str(x))
    return { vals, isStr: true, dtype: v.k === 'obj' ? 'object' : uDtype(vals) }
  }
  let a: NDArray
  try {
    a = toArray(v)
  } catch {
    throw h.err('TypeError', `Expected array-like (array or non-string sequence), got ${repr(v).slice(0, 60)}`)
  }
  if (a.ndim === 2 && a.shape[1] === 1) return { vals: a.values(), isStr: false, dtype: a.dtype }
  if (a.ndim !== 1) throw h.err('ValueError', `y should be a 1d array, got an array of shape (${a.shape.join(', ')}) instead.`)
  return { vals: a.values(), isStr: false, dtype: a.dtype }
}

const uniqueSorted = (vals: Lab[], isStr: boolean): Lab[] => {
  const u = [...new Set(vals)]
  return isStr ? (u as string[]).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)) : (u as number[]).sort((a, b) => a - b)
}

/** labels as a NumPy-looking array of the same kind as the input */
const labelArray = (vals: Lab[], like: Pick<Labels, 'isStr' | 'dtype'>): Value =>
  like.isStr ? py.obj(new ObjArray(vals as string[], like.dtype)) : py.arr(NDArray.create(vals as number[], [vals.length], like.dtype as 'int64'))

const typeOfTarget = (y: Labels): string => {
  if (!y.isStr && y.dtype !== 'int64' && y.dtype !== 'bool' && y.dtype !== 'uint8' && (y.vals as number[]).some((x) => !Number.isInteger(x))) return 'continuous'
  return new Set(y.vals).size <= 2 ? 'binary' : 'multiclass'
}

const numSamples = (h: Host, v: Value): number => {
  if (v.k === 'array') {
    if (v.a.ndim === 0) throw h.err('TypeError', `Input should have at least 1 dimension i.e. satisfy \`len(x.shape) > 0\`, got scalar \`array(${h.str(v)})\` instead.`)
    return v.a.shape[0]
  }
  if (v.k === 'list' || v.k === 'tuple') return v.items.length
  if (v.k === 'range') return h.iterate(v).length
  if (v.k === 'obj' && v.o instanceof Frame) return v.o.nrows
  if (v.k === 'obj' && v.o instanceof Series) return v.o.length
  if (v.k === 'obj' && v.o instanceof ObjArray) return v.o.shape[0]
  if (v.k === 'obj' && v.o instanceof Csr) return v.o.rows.length
  if (isForeignArray(v)) return h.iterate(v).length
  throw h.err('TypeError', `Expected sequence or array-like, got ${h.str(py.str(`<class '${v.k === 'obj' ? v.o.cls : v.k}'>`))}`)
}

const checkLengths = (h: Host, vs: Value[]) => {
  const lens = vs.map((v) => numSamples(h, v))
  if (new Set(lens).size > 1) throw h.err('ValueError', `Found input variables with inconsistent numbers of samples: [${lens.join(', ')}]`)
}

/** _safe_indexing along the samples */
const takeSamples = (h: Host, v: Value, idx: number[]): Value => {
  if (v.k === 'array') {
    const a = v.a
    const inner = a.shape.slice(1).reduce((p, d) => p * d, 1)
    const vals = a.values()
    const out: number[] = []
    for (const i of idx) for (let k = 0; k < inner; k++) out.push(vals[i * inner + k])
    return py.arr(NDArray.create(out, [idx.length, ...a.shape.slice(1)], a.dtype))
  }
  if (v.k === 'list' || v.k === 'tuple') return py.list(idx.map((i) => v.items[i]))
  if (v.k === 'range') { const it = h.iterate(v); return py.list(idx.map((i) => it[i])) }
  if (v.k === 'obj' && v.o instanceof Frame) return py.obj(v.o.take(idx, null))
  if (v.k === 'obj' && v.o instanceof Series) return py.obj(v.o.pick(idx))
  if (v.k === 'obj' && (v.o instanceof ObjArray || v.o instanceof Csr)) return py.obj(v.o.take(idx))
  if (isForeignArray(v)) return (v as Value & { k: 'obj' }).o.getItem!(py.list(idx.map((i) => py.int(i))), h)
  throw h.err('TypeError', `Expected sequence or array-like, got ${v.k}`)
}

const floatArg = (h: Host, v: Value | undefined, d: number): number => (v && v.k !== 'none' ? h.num(v) : d)
const isNone = (v: Value | undefined) => !v || v.k === 'none'
const strOf = (v: Value | undefined): string | null => (v && v.k === 'str' ? v.v : null)

/** format(x, '.nf'): exact ties go to the even digit, which Number.toFixed gets wrong */
const fixedStr = (x: number, d: number): string => {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? 'nan' : x > 0 ? 'inf' : '-inf'
  const s = x.toFixed(d)
  const exact = Math.abs(x).toFixed(Math.min(d + 40, 100))
  const tail = exact.slice(exact.indexOf('.') + 1 + d)
  if (/^50*$/.test(tail)) {
    const scale = 10 ** d
    const up = Math.ceil(Math.abs(x) * scale)
    const chosen = up % 2 === 0 ? up : up - 1
    return `${x < 0 ? '-' : ''}${(chosen / scale).toFixed(d)}`
  }
  return s
}

// ---------------------------------------------------------------- estimators

interface Spec {
  name: string
  /** parameters that may be given by position */
  pos?: string[]
  /** every __init__ parameter and its default; null = required */
  params: [string, Value | null][]
  /** parameters that change the result but are not implemented here */
  locked?: string[]
  kind?: 'classifier' | 'regressor'
  methods: (e: Est, h: Host) => Record<string, Fn>
  extra?: (e: Est, name: string, h: Host) => Value | undefined
  /** the sub-estimators get_params(deep=True) and set_params reach by name */
  named?: (e: Est, h: Host) => [string, Value][]
  setNamed?: (e: Est, name: string, v: Value, h: Host) => boolean
  getItem?: (e: Est, idx: Value, h: Host) => Value
  len?: (e: Est, h: Host) => number
}

const TRACED = new Set(['fit', 'predict', 'transform', 'fit_transform', 'fit_predict', 'predict_proba', 'predict_log_proba', 'inverse_transform', 'kneighbors', 'decision_function', 'score'])

class Est extends PyObj {
  readonly cls: string
  spec: Spec
  params = new Map<string, Value>()
  fitted = new Map<string, Value>()
  /** what fit computed, for the other methods */
  st: Record<string, unknown> = {}
  private ms: Record<string, Fn> | null = null
  constructor(spec: Spec) {
    super()
    this.spec = spec
    this.cls = spec.name
  }
  defaultOf(name: string): Value | null | undefined { return this.spec.params.find(([k]) => k === name)?.[1] }
  /** ms is bound to this object: the copy makes its own */
  pickled(copy: <T>(part: T) => T) {
    const e = new Est(this.spec)
    e.params = copy(this.params)
    e.fitted = copy(this.fitted)
    e.st = copy(this.st)
    return e
  }
  repr() {
    const out: string[] = []
    pprintEst(this, 0, 0, out)
    return out.join('')
  }
  isa(name: string) { return name === this.cls || name === 'BaseEstimator' }
  get isFitted() { return this.st.fitted === true }
  param(name: string): Value { return this.params.get(name) ?? py.NONE }
  num(name: string, h: Host): number { return h.num(this.param(name)) }
  int(name: string, h: Host): number { return h.toInt(this.param(name)) }
  flag(name: string, h: Host): boolean { return h.truthy(this.param(name)) }
  str(): string { return this.repr() }
  /** raises for a parameter that would change the answer and is not implemented */
  checkLocked(h: Host) {
    for (const k of this.spec.locked ?? []) {
      const d = this.defaultOf(k)
      if (d !== undefined && d !== null && repr(this.param(k)) !== repr(d)) throw h.err('NotImplementedError', `${this.cls}: ${k}=${repr(this.param(k))} is not available in this sandbox`)
    }
  }
  getAttr(name: string, h: Host): Value | undefined {
    if (name === 'get_params') return { k: 'fn', name, call: (a, kw) => getParams(this, h, !(kw.deep ?? a[0]) || h.truthy(kw.deep ?? a[0])) }
    if (name === 'set_params') return { k: 'fn', name, call: (_a, kw) => { setParams(this, h, kw); return py.obj(this) } }
    const ms = this.spec.named ? this.spec.methods(this, h) : (this.ms ??= this.spec.methods(this, h))
    const m = ms[name]
    if (m) return { k: 'fn', name, call: m, api: TRACED.has(name) ? { name: `sklearn.${this.cls}.${name}`, kind: 'create' as ApiKind } : undefined } as Value
    return this.fitted.get(name) ?? this.params.get(name) ?? this.spec.extra?.(this, name, h)
  }
  setAttr(name: string, v: Value) {
    if (this.params.has(name)) this.params.set(name, v)
    else this.fitted.set(name, v)
    return true
  }
  getItem(idx: Value, h: Host): Value {
    if (!this.spec.getItem) throw h.err('TypeError', `'${this.cls}' object is not subscriptable`)
    return this.spec.getItem(this, idx, h)
  }
  len(h: Host): number {
    if (!this.spec.len) throw h.err('TypeError', `object of type '${this.cls}' has no len()`)
    return this.spec.len(this, h)
  }
}

const estOf = (v: Value): Est | null => (v.k === 'obj' && v.o instanceof Est ? v.o : null)

const makeEst = (spec: Spec, args: Value[], kw: Kw, h: Host): Est => {
  const e = new Est(spec)
  const pos = spec.pos ?? []
  if (args.length > pos.length) {
    const takes = pos.length === 0 ? 'takes 1 positional argument' : pos.some((p) => spec.params.find(([k]) => k === p)?.[1] === null) && pos.length === 1 ? `takes 2 positional arguments` : `takes from 1 to ${pos.length + 1} positional arguments`
    throw h.err('TypeError', `${spec.name}.__init__() ${takes} but ${args.length + 1} ${args.length === 0 ? 'was' : 'were'} given`)
  }
  for (const [k, d] of spec.params) if (d !== null) e.params.set(k, d)
  args.forEach((v, i) => e.params.set(pos[i], v))
  for (const [k, v] of Object.entries(kw)) {
    if (!spec.params.some(([p]) => p === k)) throw h.err('TypeError', `${spec.name}.__init__() got an unexpected keyword argument '${k}'`)
    e.params.set(k, v)
  }
  const missing = spec.params.filter(([k, d]) => d === null && !e.params.has(k)).map(([k]) => `'${k}'`)
  if (missing.length) throw h.err('TypeError', `${spec.name}.__init__() missing ${missing.length} required positional argument${missing.length > 1 ? 's' : ''}: ${missing.join(' and ')}`)
  return e
}

const cloneValue = (v: Value): Value => {
  const e = estOf(v)
  if (e) return py.obj(cloneEst(e))
  if (v.k === 'list' || v.k === 'tuple') return { k: v.k, items: v.items.map(cloneValue) } as Value
  return v
}
const cloneEst = (e: Est): Est => {
  const c = new Est(e.spec)
  for (const [k, v] of e.params) c.params.set(k, cloneValue(v))
  return c
}

const getParams = (e: Est, h: Host, deep: boolean): Value => {
  const names = [...e.params.keys()].sort()
  const out: [string, Value][] = []
  for (const k of names) {
    const v = e.params.get(k)!
    out.push([k, v])
    const sub = estOf(v)
    if (deep && sub) {
      const nested = getParams(sub, h, true)
      if (nested.k === 'dict') for (const [nk, nv] of nested.d.items()) out.push([`${k}__${h.str(nk)}`, nv])
    }
  }
  if (deep && e.spec.named) {
    for (const [name, v] of e.spec.named(e, h)) {
      out.push([name, v])
      const sub = estOf(v)
      if (sub) {
        const nested = getParams(sub, h, true)
        if (nested.k === 'dict') for (const [nk, nv] of nested.d.items()) out.push([`${name}__${h.str(nk)}`, nv])
      }
    }
  }
  return dictOf(out)
}

const setParams = (e: Est, h: Host, kw: Kw) => {
  const valid = [...e.params.keys()].sort()
  for (const [k, v] of Object.entries(kw)) {
    const at = k.indexOf('__')
    const head = at < 0 ? k : k.slice(0, at)
    const named = e.spec.named?.(e, h).find(([n]) => n === head)
    if (at >= 0) {
      const sub = estOf(named?.[1] ?? e.params.get(head) ?? py.NONE)
      if (!sub) throw h.err('ValueError', `Invalid parameter ${reprStr(head)} for estimator ${e.repr()}. Valid parameters are: [${valid.map(reprStr).join(', ')}].`)
      setParams(sub, h, { [k.slice(at + 2)]: v })
    } else if (named && e.spec.setNamed?.(e, k, v, h)) {
      continue
    } else {
      if (!e.params.has(k)) throw h.err('ValueError', `Invalid parameter ${reprStr(k)} for estimator ${e.repr()}. Valid parameters are: [${valid.map(reprStr).join(', ')}].`)
      e.params.set(k, v)
    }
  }
}

// ---- repr: a port of sklearn.utils._pprint with compact=True, indent=1, indent_at_name=True, width=80

const changedParams = (e: Est): [string, Value][] =>
  [...e.params.entries()].filter(([k, v]) => {
    const d = e.defaultOf(k)
    if (d === null || d === undefined) return true
    const a = repr(d)
    return safeRepr(v) !== a && !(a === 'nan' && repr(v) === 'nan')
  }).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))

const safeRepr = (v: Value): string => {
  const e = estOf(v)
  if (e) return `${e.cls}(${changedParams(e).map(([k, x]) => `${k}=${safeRepr(x)}`).join(', ')})`
  if (v.k === 'list') return `[${v.items.map(safeRepr).join(', ')}]`
  if (v.k === 'tuple') return v.items.length === 1 ? `(${safeRepr(v.items[0])},)` : `(${v.items.map(safeRepr).join(', ')})`
  if (v.k === 'dict') return `{${[...v.d.items()].sort(([a], [b]) => (repr(a) < repr(b) ? -1 : 1)).map(([k, x]) => `${repr(k)}: ${safeRepr(x)}`).join(', ')}}`
  return repr(v)
}

const PP_WIDTH = 80
const PP_MAX_ITEMS = 30

const ppFormat = (v: Value, indent: number, allowance: number, out: string[]) => {
  const rep = safeRepr(v)
  if (rep.length <= PP_WIDTH - indent - allowance) {
    out.push(rep)
    return
  }
  const e = estOf(v)
  if (e) pprintEst(e, indent, allowance, out)
  else if (v.k === 'list') {
    out.push('[')
    ppItems(v.items.map((x) => ({ rep: safeRepr(x), v: x, key: null })), indent, allowance + 1, out)
    out.push(']')
  } else if (v.k === 'tuple') {
    const end = v.items.length === 1 ? ',)' : ')'
    out.push('(')
    ppItems(v.items.map((x) => ({ rep: safeRepr(x), v: x, key: null })), indent, allowance + end.length, out)
    out.push(end)
  } else if (v.k === 'dict') {
    out.push('{')
    const items = [...v.d.items()].sort(([a], [b]) => (repr(a) < repr(b) ? -1 : 1))
    ppItems(items.map(([k, x]) => ({ rep: `${repr(k)}: ${safeRepr(x)}`, v: x, key: `${repr(k)}: ` })), indent, allowance + 1, out)
    out.push('}')
  } else out.push(rep)
}

const ppItems = (items: { rep: string; v: Value; key: string | null }[], indent0: number, allowance: number, out: string[]) => {
  const indent = indent0 + 1
  const delimnl = `,\n${' '.repeat(indent)}`
  let delim = ''
  let width = PP_WIDTH - indent + 1
  let maxWidth = width
  for (let i = 0; i < items.length; i++) {
    if (i === PP_MAX_ITEMS) {
      out.push(', ...')
      break
    }
    const last = i === items.length - 1
    if (last) {
      maxWidth -= allowance
      width -= allowance
    }
    const it = items[i]
    const w = it.rep.length + 2
    if (width < w) {
      width = maxWidth
      if (delim) delim = delimnl
    }
    if (width >= w) {
      width -= w
      out.push(delim)
      delim = ', '
      out.push(it.rep)
      continue
    }
    out.push(delim)
    delim = delimnl
    const allow = last ? allowance : 1
    if (it.key === null) ppFormat(it.v, indent, allow, out)
    else if (it.rep.length <= PP_WIDTH - indent - allow) out.push(it.rep)
    else {
      out.push(it.key)
      ppFormat(it.v, indent + it.key.length, allow, out)
    }
  }
}

function pprintEst(e: Est, indent: number, allowance: number, out: string[]) {
  out.push(`${e.cls}(`)
  ppItems(changedParams(e).map(([k, v]) => ({ rep: `${k}=${safeRepr(v)}`, v, key: `${k}=` })), indent + e.cls.length, allowance + 1, out)
  out.push(')')
}

// ---- helpers for the methods

const requireFit = (e: Est, h: Host) => { if (!e.isFitted) throw notFitted(h, e.cls) }

const rememberFeatures = (e: Est, X: XData) => {
  e.st.m = X.m
  e.st.names = X.names
  e.fitted.set('n_features_in_', py.int(X.m))
  if (X.names) e.fitted.set('feature_names_in_', py.obj(new ObjArray(X.names, 'object')))
  else e.fitted.delete('feature_names_in_')
}

const checkNames = (e: Est, X: { names: string[] | null }, h: Host) => {
  const seen = e.st.names as string[] | null | undefined
  if (!seen || !X.names || seen.join('\0') === X.names.join('\0')) return
  const unseen = X.names.filter((n) => !seen.includes(n))
  const gone = seen.filter((n) => !X.names!.includes(n))
  let msg = 'The feature names should match those that were passed during fit.\n'
  if (unseen.length) msg += `Feature names unseen at fit time:\n${unseen.map((n) => `- ${n}\n`).join('')}`
  if (gone.length) msg += `Feature names seen at fit time, yet now missing:\n${gone.map((n) => `- ${n}\n`).join('')}`
  if (!unseen.length && !gone.length) msg += 'Feature names must be in the same order as they were in fit.\n'
  throw h.err('ValueError', msg)
}

const readTest = (e: Est, v: Value | undefined, h: Host): XData => {
  requireFit(e, h)
  const X = readX(h, v, e.cls)
  if (X.m !== e.st.m) throw h.err('ValueError', `X has ${X.m} features, but ${e.cls} is expecting ${e.st.m} features as input.`)
  checkNames(e, X, h)
  return X
}

const rejectKw = (h: Host, who: string, kw: Kw, ok: string[]) => {
  for (const k of Object.keys(kw)) if (!ok.includes(k)) throw h.err('TypeError', `${who}() got an unexpected keyword argument '${k}'`)
}

/** only sample_weight=None is on offer: the weighted fits are not implemented */
const noSampleWeight = (h: Host, who: string, kw: Kw, v?: Value) => {
  const w = kw.sample_weight ?? v
  if (w && w.k !== 'none') throw h.err('NotImplementedError', `${who}: sample_weight is not available in this sandbox`)
}

interface FitTarget { y: Labels; classes: Lab[]; enc: number[] }

const classifierTarget = (h: Host, yV: Value | undefined): FitTarget => {
  if (!yV || yV.k === 'none') throw h.err('ValueError', 'y cannot be None')
  const y = labelsOf(h, yV)
  const t = typeOfTarget(y)
  if (t === 'continuous') throw h.err('ValueError', `Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.`)
  const classes = uniqueSorted(y.vals, y.isStr)
  const at = new Map(classes.map((c, i) => [c, i]))
  return { y, classes, enc: y.vals.map((v) => at.get(v)!) }
}

const storeClasses = (e: Est, t: FitTarget) => {
  e.st.classes = t.classes
  e.st.labels = { isStr: t.y.isStr, dtype: t.y.dtype }
  e.fitted.set('classes_', labelArray(t.classes, t.y))
}

const predictLabels = (e: Est, idx: number[]): Value => labelArray(idx.map((i) => (e.st.classes as Lab[])[i]), e.st.labels as Labels)

const accuracyOf = (h: Host, yTrue: Value, yPred: Value): number => {
  const a = labelsOf(h, yTrue)
  const b = labelsOf(h, yPred)
  return countMatches(a, b) / a.vals.length
}
const countMatches = (a: Labels, b: Labels) => a.vals.reduce((n: number, v, i) => n + (v === b.vals[i] ? 1 : 0), 0)

/** ClassifierMixin.score */
const classifierScore = (e: Est, h: Host, predict: Fn): Fn => (args, kw) => {
  noSampleWeight(h, `${e.cls}.score`, kw, args[2])
  return py.float(accuracyOf(h, args[1] ?? kw.y, predict([args[0] ?? kw.X], {})))
}

const r2Of = (yT: number[], yP: number[]): number => {
  const mean = npMean(yT)
  const num = npSum(yT.map((y, i) => (y - yP[i]) ** 2))
  const den = npSum(yT.map((y) => (y - mean) ** 2))
  if (den === 0) return num === 0 ? 1 : 0
  return 1 - num / den
}

const regressionTarget = (h: Host, yV: Value | undefined): { rows: number[][]; multi: boolean; t: number } => {
  if (!yV || yV.k === 'none') throw h.err('ValueError', 'y cannot be None')
  const a = numericArray(h, yV)
  finiteCheck(h, a.values(), 'y', 'Estimator')
  if (a.ndim === 1) return { rows: a.values().map((x) => [x]), multi: false, t: 1 }
  if (a.ndim === 2) return { rows: rowsOf(a), multi: true, t: a.shape[1] }
  throw h.err('ValueError', `Found array with dim ${a.ndim}, while dim <= 2 is required by the target.`)
}

const inRange = (h: Host, e: Est, name: string, ok: boolean, what: string) => {
  if (!ok) throw badParam(h, `The '${name}' parameter of ${e.cls} must be ${what}. Got ${repr(e.param(name))} instead.`)
}

// ---------------------------------------------------------------- model selection

const shuffleSplitSizes = (h: Host, n: number, testV: Value | undefined, trainV: Value | undefined, dflt = 0.25): [number, number] => {
  const kind = (v: Value | undefined) => (isNone(v) ? null : v!.k === 'int' || (v!.k === 'array' && v!.a.dtype === 'int64' && v!.a.ndim === 0) ? 'i' : v!.k === 'float' ? 'f' : 'x')
  const tk0 = kind(testV)
  const trk = kind(trainV)
  const useDefault = tk0 === null && trk === null
  const tk = useDefault ? 'f' : tk0
  const test = useDefault ? dflt : tk ? h.num(testV!) : 0
  const train = trk && trk !== 'x' ? h.num(trainV!) : 0
  const txt = (v: Value | undefined, x: number) => (isNone(v) && !useDefault ? 'None' : v && v.k !== 'none' ? h.str(v) : String(x))
  if ((tk === 'i' && (test >= n || test <= 0)) || (tk === 'f' && (test <= 0 || test >= 1))) throw h.err('ValueError', `test_size=${txt(testV, test)} should be either positive and smaller than the number of samples ${n} or a float in the (0, 1) range`)
  if ((trk === 'i' && (train >= n || train <= 0)) || (trk === 'f' && (train <= 0 || train >= 1))) throw h.err('ValueError', `train_size=${txt(trainV, train)} should be either positive and smaller than the number of samples ${n} or a float in the (0, 1) range`)
  if (trk === 'x') throw h.err('ValueError', `Invalid value for train_size: ${repr(trainV!)}`)
  if (tk === 'x') throw h.err('ValueError', `Invalid value for test_size: ${repr(testV!)}`)
  if (trk === 'f' && tk === 'f' && train + test > 1) throw h.err('ValueError', `The sum of test_size and train_size = ${pyFloatRepr(train + test)}, should be in the (0, 1) range. Reduce test_size and/or train_size.`)
  let nTest = tk === 'f' ? Math.ceil(test * n) : tk === 'i' ? test : 0
  let nTrain = trk === 'f' ? Math.floor(train * n) : trk === 'i' ? train : 0
  if (!trk) nTrain = n - nTest
  else if (!tk) nTest = n - nTrain
  if (nTrain + nTest > n) throw h.err('ValueError', `The sum of train_size and test_size = ${nTrain + nTest}, should be smaller than the number of samples ${n}. Reduce test_size and/or train_size.`)
  if (nTrain === 0) throw h.err('ValueError', `With n_samples=${n}, test_size=${txt(testV, test)} and train_size=${txt(trainV, train)}, the resulting train set will be empty. Adjust any of the aforementioned parameters.`)
  return [nTrain, nTest]
}

/** _approximate_mode: how many of each class a stratified sample of n draws gets */
const approximateMode = (counts: number[], n: number, rng: RandomState): number[] => {
  const total = counts.reduce((a, b) => a + b, 0)
  const cont = counts.map((c) => (c / total) * n)
  const floored = cont.map(Math.floor)
  let need = n - floored.reduce((a, b) => a + b, 0)
  if (need > 0) {
    const rem = cont.map((c, i) => c - floored[i])
    const values = [...new Set(rem)].sort((a, b) => b - a)
    for (const value of values) {
      const inds = rem.flatMap((r, i) => (r === value ? [i] : []))
      const now = Math.min(inds.length, need)
      const pick = rng.permutation(inds.length).slice(0, now).map((k) => inds[k])
      for (const p of pick) floored[p]++
      need -= now
      if (need === 0) break
    }
  }
  return floored
}

const stratifiedShuffle = (h: Host, y: Labels, nTrain: number, nTest: number, rng: RandomState): [number[], number[]] => {
  const classes = uniqueSorted(y.vals, y.isStr)
  const at = new Map(classes.map((c, i) => [c, i]))
  const code = y.vals.map((v) => at.get(v)!)
  const counts = classes.map(() => 0)
  code.forEach((c) => counts[c]++)
  if (Math.min(...counts) < 2) throw h.err('ValueError', 'The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.')
  if (nTrain < classes.length) throw h.err('ValueError', `The train_size = ${nTrain} should be greater or equal to the number of classes = ${classes.length}`)
  if (nTest < classes.length) throw h.err('ValueError', `The test_size = ${nTest} should be greater or equal to the number of classes = ${classes.length}`)
  const members = classes.map((_, k) => code.flatMap((c, i) => (c === k ? [i] : [])))
  const ni = approximateMode(counts, nTrain, rng)
  const ti = approximateMode(counts.map((c, i) => c - ni[i]), nTest, rng)
  const train: number[] = []
  const test: number[] = []
  classes.forEach((_, i) => {
    const perm = rng.permutation(counts[i]).map((k) => members[i][k])
    train.push(...perm.slice(0, ni[i]))
    test.push(...perm.slice(ni[i], ni[i] + ti[i]))
  })
  return [rng.permutation(train.length).map((k) => train[k]), rng.permutation(test.length).map((k) => test[k])]
}

class Splitter extends PyObj {
  readonly cls: string
  params: [string, Value][]
  kind: 'KFold' | 'StratifiedKFold' | 'LeaveOneOut' | 'ShuffleSplit'
  constructor(kind: Splitter['kind'], params: [string, Value][]) {
    super()
    this.cls = kind
    this.kind = kind
    this.params = params
  }
  repr() { return `${this.cls}(${this.params.map(([k, v]) => `${k}=${repr(v)}`).join(', ')})` }
  get(name: string): Value { return this.params.find(([k]) => k === name)?.[1] ?? py.NONE }
  splits(h: Host, X: Value, yV: Value | undefined): [number[], number[]][] {
    const n = numSamples(h, X)
    const all = Array.from({ length: n }, (_, i) => i)
    const fromTests = (tests: number[][]): [number[], number[]][] => tests.map((t) => {
      const mask = new Set(t)
      return [all.filter((i) => !mask.has(i)), all.filter((i) => mask.has(i))]
    })
    if (this.kind === 'LeaveOneOut') {
      if (n <= 1) throw h.err('ValueError', `Cannot perform LeaveOneOut with n_samples=${n}.`)
      return fromTests(all.map((i) => [i]))
    }
    if (this.kind === 'ShuffleSplit') {
      const [nTrain, nTest] = shuffleSplitSizes(h, n, this.get('test_size'), this.get('train_size'), 0.1)
      const rng = rngOf(h, this.get('random_state'))
      return Array.from({ length: h.toInt(this.get('n_splits')) }, () => {
        const p = rng.permutation(n)
        return [p.slice(nTest, nTest + nTrain), p.slice(0, nTest)] as [number[], number[]]
      })
    }
    const k = h.toInt(this.get('n_splits'))
    if (k > n) throw h.err('ValueError', `Cannot have number of splits n_splits=${k} greater than the number of samples: n_samples=${n}.`)
    const shuffle = h.truthy(this.get('shuffle'))
    if (this.kind === 'KFold') {
      const order = [...all]
      if (shuffle) rngOf(h, this.get('random_state')).shuffle(order)
      const sizes = Array.from({ length: k }, (_, i) => Math.floor(n / k) + (i < n % k ? 1 : 0))
      let at = 0
      return fromTests(sizes.map((s) => {
        const part = order.slice(at, at + s)
        at += s
        return part
      }))
    }
    if (!yV || yV.k === 'none') throw h.err('TypeError', "StratifiedKFold.split() missing 1 required positional argument: 'y'")
    const y = labelsOf(h, yV)
    checkLengths(h, [X, yV])
    const t = typeOfTarget(y)
    if (t !== 'binary' && t !== 'multiclass') throw h.err('ValueError', `Supported target types are: ('binary', 'multiclass'). Got ${reprStr(t)} instead.`)
    const firstAt = new Map<Lab, number>()
    y.vals.forEach((v) => { if (!firstAt.has(v)) firstAt.set(v, firstAt.size) })
    const enc = y.vals.map((v) => firstAt.get(v)!)
    const nc = firstAt.size
    const counts = new Array<number>(nc).fill(0)
    enc.forEach((c) => counts[c]++)
    if (counts.every((c) => k > c)) throw h.err('ValueError', `n_splits=${k} cannot be greater than the number of members in each class.`)
    const sorted = [...enc].sort((a, b) => a - b)
    const alloc = Array.from({ length: k }, (_, i) => {
      const c = new Array<number>(nc).fill(0)
      for (let j = i; j < n; j += k) c[sorted[j]]++
      return c
    })
    const rng = shuffle ? rngOf(h, this.get('random_state')) : null
    const fold = new Array<number>(n)
    for (let c = 0; c < nc; c++) {
      const folds = alloc.flatMap((a, f) => new Array<number>(a[c]).fill(f))
      rng?.shuffle(folds)
      let at = 0
      enc.forEach((e, i) => { if (e === c) fold[i] = folds[at++] })
    }
    return fromTests(Array.from({ length: k }, (_, f) => all.filter((i) => fold[i] === f)))
  }
  getAttr(name: string, h: Host): Value | undefined {
    if (name === 'split') {
      return { k: 'fn', name, api: { name: `sklearn.${this.cls}.split`, kind: 'create' }, call: (args, kw) => {
        // a generator, so its errors wait for the first next() as they do in Python
        let parts: [number[], number[]][] | null = null
        let at = 0
        return { k: 'iter', name: 'generator', next: () => {
          parts ??= this.splits(h, args[0] ?? kw.X, args[1] ?? kw.y)
          const p = parts[at++]
          return p ? py.tuple([vec(p[0], 'int64'), vec(p[1], 'int64')]) : undefined
        } } as Value
      } } as Value
    }
    if (name === 'get_n_splits') {
      return { k: 'fn', name, call: (args, kw) => {
        if (this.kind === 'LeaveOneOut') {
          const X = args[0] ?? kw.X
          if (isNone(X)) throw h.err('ValueError', "The 'X' parameter should not be None.")
          return py.int(numSamples(h, X))
        }
        return this.get('n_splits')
      } }
    }
    return this.params.find(([k]) => k === name)?.[1]
  }
  isa(name: string) { return name === this.cls }
}

const makeKFold = (kind: 'KFold' | 'StratifiedKFold') => (h: Host, args: Value[], kw: Kw): Value => {
  const nv = args[0] ?? kw.n_splits ?? py.int(5)
  const k = h.toInt(nv, 'an integer')
  for (const key of Object.keys(kw)) if (!['n_splits', 'shuffle', 'random_state'].includes(key)) throw h.err('TypeError', `${kind}.__init__() got an unexpected keyword argument '${key}'`)
  if (k <= 1) throw h.err('ValueError', `k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits=${k}.`)
  const shuffle = kw.shuffle ?? py.bool(false)
  if (shuffle.k !== 'bool') throw h.err('TypeError', `shuffle must be True or False; got ${repr(shuffle)}`)
  const rs = kw.random_state ?? py.NONE
  if (!h.truthy(shuffle) && rs.k !== 'none') throw h.err('ValueError', 'Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.')
  return py.obj(new Splitter(kind, [['n_splits', py.int(k)], ['random_state', rs], ['shuffle', shuffle]]))
}

/** check_cv */
const cvOf = (h: Host, cv: Value | undefined, y: Value | undefined, classifier: boolean): Splitter => {
  const v = cv && cv.k !== 'none' ? cv : py.int(5)
  if (v.k === 'obj' && v.o instanceof Splitter) return v.o
  if (v.k !== 'int') throw h.err('ValueError', `Expected cv as an integer, cross-validation object (from sklearn.model_selection) or an iterable. Got ${repr(v)}.`)
  const useStrat = classifier && y && y.k !== 'none' && typeOfTarget(labelsOf(h, y)) !== 'continuous'
  const made = makeKFold(useStrat ? 'StratifiedKFold' : 'KFold')(h, [v], {})
  return (made as Value & { k: 'obj' }).o as Splitter
}

const isClassifier = (h: Host, e: Est): boolean => {
  if (e.spec.kind === 'classifier') return true
  if (e.cls === 'Pipeline') {
    const steps = h.iterate(e.param('steps'))
    const last = h.iterate(steps[steps.length - 1])[1]
    const le = estOf(last)
    return !!le && isClassifier(h, le)
  }
  if (e.cls === 'GridSearchCV') {
    const inner = estOf(e.param('estimator'))
    return !!inner && isClassifier(h, inner)
  }
  return false
}

/** the score of a fitted estimator on (X, y), through est.score or a named scorer */
const scoreWith = (h: Host, est: Value, scoring: Value | undefined, X: Value, y: Value | undefined): number => {
  if (!scoring || scoring.k === 'none') return h.num(h.call(h.getAttr(est, 'score'), y ? [X, y] : [X]))
  if (scoring.k !== 'str') return h.num(h.call(scoring, y ? [est, X, y] : [est, X]))
  const name = scoring.v
  const pred = () => h.call(h.getAttr(est, 'predict'), [X])
  const yV = y as Value
  const bin = (m: 'precision' | 'recall' | 'f1', avg: string) => prfScore(h, yV, pred(), m, { average: py.str(avg) })
  switch (name) {
    case 'accuracy': return accuracyOf(h, yV, pred())
    case 'f1': return bin('f1', 'binary')
    case 'f1_macro': return bin('f1', 'macro')
    case 'f1_micro': return bin('f1', 'micro')
    case 'f1_weighted': return bin('f1', 'weighted')
    case 'precision': return bin('precision', 'binary')
    case 'precision_macro': return bin('precision', 'macro')
    case 'precision_micro': return bin('precision', 'micro')
    case 'precision_weighted': return bin('precision', 'weighted')
    case 'recall': return bin('recall', 'binary')
    case 'recall_macro': return bin('recall', 'macro')
    case 'recall_micro': return bin('recall', 'micro')
    case 'recall_weighted': return bin('recall', 'weighted')
    case 'r2': return r2Of(numbersOf(h, yV), numbersOf(h, pred()))
    case 'neg_mean_squared_error': return -npMean(diffs(h, yV, pred()).map((d) => d * d))
    case 'neg_root_mean_squared_error': return -Math.sqrt(npMean(diffs(h, yV, pred()).map((d) => d * d)))
    case 'neg_mean_absolute_error': return -npMean(diffs(h, yV, pred()).map(Math.abs))
  }
  throw badParam(h, `The 'scoring' parameter of check_scoring must be a str among the scorer names. Got ${reprStr(name)} instead.`)
}

const numbersOf = (h: Host, v: Value): number[] => {
  const a = numericArray(h, v)
  return a.values()
}
const diffs = (h: Host, a: Value, b: Value): number[] => {
  const x = numbersOf(h, a)
  const y = numbersOf(h, b)
  return x.map((v, i) => v - y[i])
}

const fitOn = (h: Host, e: Est, X: Value, y: Value | undefined, train: number[]) => {
  const c = cloneEst(e)
  const cv = py.obj(c)
  h.call(h.getAttr(cv, 'fit'), y && y.k !== 'none' ? [takeSamples(h, X, train), takeSamples(h, y, train)] : [takeSamples(h, X, train)])
  return cv
}

const crossValScores = (h: Host, e: Est, X: Value, y: Value | undefined, scoring: Value | undefined, cv: Value | undefined): number[] => {
  const s = cvOf(h, cv, y, isClassifier(h, e))
  return s.splits(h, X, y).map(([train, test]) => scoreWith(h, fitOn(h, e, X, y, train), scoring, takeSamples(h, X, test), y && y.k !== 'none' ? takeSamples(h, y, test) : undefined))
}

const paramCandidates = (h: Host, grid: Value): [string, Value][][] => {
  const grids = grid.k === 'list' ? grid.items : [grid]
  const out: [string, Value][][] = []
  for (const g of grids) {
    if (g.k !== 'dict') throw h.err('TypeError', `Parameter grid should be a dict or a list, got: ${repr(g)} of type ${g.k}`)
    const items = g.d.items().map(([k, v]) => [h.str(k), h.iterate(v)] as [string, Value[]]).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    let combos: [string, Value][][] = [[]]
    for (const [k, vals] of items) combos = combos.flatMap((c) => vals.map((v) => [...c, [k, v] as [string, Value]]))
    out.push(...combos)
  }
  return out
}

const GRID_SEARCH: Spec = {
  name: 'GridSearchCV',
  pos: ['estimator', 'param_grid'],
  params: [['estimator', null], ['param_grid', null], ['scoring', py.NONE], ['n_jobs', py.NONE], ['refit', py.bool(true)], ['cv', py.NONE], ['verbose', py.int(0)], ['pre_dispatch', py.str('2*n_jobs')], ['error_score', py.float(NaN)], ['return_train_score', py.bool(false)]],
  locked: ['return_train_score'],
  methods: (e, h) => {
    const best = (): Value => {
      if (!e.isFitted) throw notFitted(h, e.cls)
      const b = e.fitted.get('best_estimator_')
      if (!b) throw h.err('AttributeError', "This GridSearchCV instance was initialized with `refit=False`. predict is available only after refitting on the best parameters. You can refit an estimator manually using the `best_params_` attribute")
      return b
    }
    const delegate = (name: string): Fn => (args, kw) => h.call(h.getAttr(best(), name), args, kw)
    return {
      fit: (args, kw) => {
        const X = args[0] ?? kw.X
        const y = args[1] ?? kw.y
        const base = estOf(e.param('estimator'))
        if (!base) throw h.err('TypeError', 'estimator should be an estimator implementing the fit method')
        const cands = paramCandidates(h, e.param('param_grid'))
        const scoring = e.param('scoring')
        const splitter = cvOf(h, e.param('cv'), y, isClassifier(h, base))
        const splits = splitter.splits(h, X, y)
        const scores = cands.map((c) => splits.map(([train, test]) => {
          const m = cloneEst(base)
          setParams(m, h, Object.fromEntries(c))
          const cv = fitOn(h, m, X, y, train)
          return scoreWith(h, cv, scoring, takeSamples(h, X, test), y && y.k !== 'none' ? takeSamples(h, y, test) : undefined)
        }))
        const means = scores.map(npMean)
        const stds = scores.map((s, i) => Math.sqrt(npMean(s.map((x) => (x - means[i]) ** 2))))
        const sortedM = [...means].sort((a, b) => b - a)
        const ranks = means.map((m) => sortedM.indexOf(m) + 1)
        let bi = 0
        ranks.forEach((r, i) => { if (r < ranks[bi]) bi = i })
        const paramsV = cands.map((c) => dictOf(c))
        const results: [string, Value][] = []
        results.push(['params', py.list(paramsV)])
        splits.forEach((_, f) => results.push([`split${f}_test_score`, vec(scores.map((s) => s[f]))]))
        results.push(['mean_test_score', vec(means)], ['std_test_score', vec(stds)], ['rank_test_score', vec(ranks, 'int64')])
        e.fitted.set('cv_results_', dictOf(results))
        e.fitted.set('best_index_', py.int(bi))
        e.fitted.set('best_score_', py.float(means[bi]))
        e.fitted.set('best_params_', paramsV[bi])
        e.fitted.set('n_splits_', py.int(splits.length))
        e.fitted.set('scorer_', scoring)
        e.st.fitted = true
        if (h.truthy(e.param('refit'))) {
          const m = cloneEst(base)
          setParams(m, h, Object.fromEntries(cands[bi]))
          const mv = py.obj(m)
          h.call(h.getAttr(mv, 'fit'), y && y.k !== 'none' ? [X, y] : [X])
          e.fitted.set('best_estimator_', mv)
          const cl = m.fitted.get('classes_')
          if (cl) e.fitted.set('classes_', cl)
        }
        return py.obj(e)
      },
      predict: delegate('predict'),
      predict_proba: delegate('predict_proba'),
      transform: delegate('transform'),
      decision_function: delegate('decision_function'),
      score: (args, kw) => py.float(scoreWith(h, best(), e.param('scoring'), args[0] ?? kw.X, args[1] ?? kw.y)),
    }
  },
}

const trainTestSplit: (h: Host) => Fn = (h) => (args, kw) => {
  rejectKw(h, 'train_test_split', kw, ['test_size', 'train_size', 'random_state', 'shuffle', 'stratify'])
  if (!args.length) throw h.err('ValueError', 'At least one array required as input')
  checkLengths(h, args)
  for (const name of ['test_size', 'train_size']) {
    const v = kw[name]
    if (v && v.k !== 'none' && !((v.k === 'float' && v.v > 0 && v.v < 1) || (v.k === 'int' && v.v >= 1))) throw badParam(h, `The '${name}' parameter of train_test_split must be a float in the range (0.0, 1.0), an int in the range [1, inf) or None. Got ${repr(v)} instead.`)
  }
  const n = numSamples(h, args[0])
  const [nTrain, nTest] = shuffleSplitSizes(h, n, kw.test_size, kw.train_size)
  const strat = kw.stratify
  let train: number[]
  let test: number[]
  if (kw.shuffle && !h.truthy(kw.shuffle)) {
    if (!isNone(strat)) throw h.err('ValueError', 'Stratified train/test split is not implemented for shuffle=False')
    train = Array.from({ length: nTrain }, (_, i) => i)
    test = Array.from({ length: nTest }, (_, i) => nTrain + i)
  } else if (!isNone(strat)) {
    ;[train, test] = stratifiedShuffle(h, labelsOf(h, strat), nTrain, nTest, rngOf(h, kw.random_state))
  } else {
    const p = rngOf(h, kw.random_state).permutation(n)
    test = p.slice(0, nTest)
    train = p.slice(nTest, nTest + nTrain)
  }
  return py.list(args.flatMap((a) => [takeSamples(h, a, train), takeSamples(h, a, test)]))
}

// ---------------------------------------------------------------- preprocessing

/** _handle_zeros_in_scale */
const safeScale = (xs: number[]): number[] => xs.map((x) => (x < 10 * Number.EPSILON ? 1 : x))

const transformerMixin = (e: Est, h: Host, m: { fit: Fn; transform: Fn }): Record<string, Fn> => ({
  fit: m.fit,
  transform: m.transform,
  fit_transform: (args, kw) => {
    const X = args[0] ?? kw.X
    m.fit([X], {})
    return m.transform([X], {})
  },
  set_output: () => { throw h.err('NotImplementedError', `${e.cls}.set_output is not available in this sandbox: transformers return NumPy arrays here`) },
})

const featureNames = (e: Est, h: Host, given: Value | undefined): string[] => {
  const m = e.st.m as number
  const seen = e.st.names as string[] | null
  if (!isNone(given)) {
    const list = labelsOf(h, given!).vals.map(String)
    if (list.length !== m) throw h.err('ValueError', 'input_features should have length equal to number of features.')
    if (seen && seen.join('\0') !== list.join('\0')) throw h.err('ValueError', 'input_features is not equal to feature_names_in_')
    return list
  }
  return seen ?? Array.from({ length: m }, (_, i) => `x${i}`)
}

const nanTransformCheck = (h: Host, a: NDArray, est: string) => {
  if (a.values().some(Number.isNaN)) throw h.err('NotImplementedError', `${est}: missing values (NaN) are not available in this sandbox`)
  if (a.values().some((x) => !Number.isFinite(x))) throw h.err('ValueError', "Input X contains infinity or a value too large for dtype('float64').")
}

const readXAllowNan = (h: Host, v: Value | undefined, est: string): XData => {
  const a = v && v.k !== 'none' ? numericArray(h, v) : null
  if (a) nanTransformCheck(h, a, est)
  return readX(h, v, est)
}

const STANDARD_SCALER: Spec = {
  name: 'StandardScaler',
  params: [['copy', py.bool(true)], ['with_mean', py.bool(true)], ['with_std', py.bool(true)]],
  methods: (e, h) => {
    const m = {
      fit: ((args, kw) => {
        noSampleWeight(h, 'StandardScaler.fit', kw, args[2])
        const X = readXAllowNan(h, args[0] ?? kw.X, 'StandardScaler')
        rememberFeatures(e, X)
        const withMean = e.flag('with_mean', h)
        const withStd = e.flag('with_std', h)
        e.fitted.set('n_samples_seen_', py.int(X.n))
        const sum = sumAxis0(X.rows, X.m)
        const mean = sum.map((s) => s / X.n)
        e.st.mean = withMean ? mean : null
        e.st.scale = null
        e.fitted.set('mean_', !withMean && !withStd ? py.NONE : vec(mean))
        e.fitted.set('var_', py.NONE)
        e.fitted.set('scale_', py.NONE)
        if (withStd) {
          const T = sum.map((s) => s / X.n)
          const temp = X.rows.map((r) => r.map((x, j) => x - T[j]))
          const corr = sumAxis0(temp, X.m)
          const sq = sumAxis0(temp.map((r) => r.map((x) => x * x)), X.m)
          const vr = sq.map((s, j) => (s - (corr[j] * corr[j]) / X.n) / X.n)
          const eps = Number.EPSILON
          const scale = vr.map((v, j) => (v <= X.n * eps * v + (X.n * mean[j] * eps) ** 2 ? 1 : Math.sqrt(v)))
          e.st.scale = scale
          e.fitted.set('var_', vec(vr))
          e.fitted.set('scale_', vec(scale))
        }
        e.st.fitted = true
        return py.obj(e)
      }) as Fn,
      transform: ((args, kw) => {
        requireFit(e, h)
        const X = readXAllowNan(h, args[0] ?? kw.X, 'StandardScaler')
        if (X.m !== e.st.m) throw h.err('ValueError', `X has ${X.m} features, but StandardScaler is expecting ${e.st.m} features as input.`)
        const mean = e.st.mean as number[] | null
        const scale = e.st.scale as number[] | null
        return mat(X.rows.map((r) => r.map((x, j) => {
          let v = x
          if (mean) v -= mean[j]
          if (scale) v /= scale[j]
          return v
        })), X.m)
      }) as Fn,
    }
    return {
      ...transformerMixin(e, h, m),
      inverse_transform: (args, kw) => {
        requireFit(e, h)
        const a = numericArray(h, args[0] ?? kw.X)
        const mean = e.st.mean as number[] | null
        const scale = e.st.scale as number[] | null
        const f = (x: number, j: number) => {
          let v = x
          if (scale) v *= scale[j]
          if (mean) v += mean[j]
          return v
        }
        if (a.ndim === 1) return vec(a.values().map(f))
        return mat(rowsOf(a).map((r) => r.map(f)), a.shape[1])
      },
      get_feature_names_out: (args, kw) => { requireFit(e, h); return py.obj(new ObjArray(featureNames(e, h, args[0] ?? kw.input_features), 'object')) },
    }
  },
}

const MINMAX_SCALER: Spec = {
  name: 'MinMaxScaler',
  pos: ['feature_range'],
  params: [['feature_range', py.tuple([py.int(0), py.int(1)])], ['copy', py.bool(true)], ['clip', py.bool(false)]],
  methods: (e, h) => {
    const range = () => h.iterate(e.param('feature_range')).map((v) => h.num(v))
    const m = {
      fit: ((args, kw) => {
        const [lo, hi] = range()
        if (lo >= hi) throw h.err('ValueError', `Minimum of desired feature range must be smaller than maximum. Got ${repr(e.param('feature_range'))}.`)
        const X = readXAllowNan(h, args[0] ?? kw.X, 'MinMaxScaler')
        rememberFeatures(e, X)
        const cols = colsOf(X.rows, X.m)
        const dmin = cols.map((c) => Math.min(...c))
        const dmax = cols.map((c) => Math.max(...c))
        const drange = dmax.map((x, j) => x - dmin[j])
        const safe = safeScale(drange)
        const scale = safe.map((d) => (hi - lo) / d)
        const min = dmin.map((d, j) => lo - d * scale[j])
        e.st.scale = scale
        e.st.min = min
        e.fitted.set('n_samples_seen_', py.int(X.n))
        e.fitted.set('data_min_', vec(dmin))
        e.fitted.set('data_max_', vec(dmax))
        e.fitted.set('data_range_', vec(drange))
        e.fitted.set('scale_', vec(scale))
        e.fitted.set('min_', vec(min))
        e.st.fitted = true
        return py.obj(e)
      }) as Fn,
      transform: ((args, kw) => {
        requireFit(e, h)
        const X = readXAllowNan(h, args[0] ?? kw.X, 'MinMaxScaler')
        if (X.m !== e.st.m) throw h.err('ValueError', `X has ${X.m} features, but MinMaxScaler is expecting ${e.st.m} features as input.`)
        const scale = e.st.scale as number[]
        const min = e.st.min as number[]
        const clip = e.flag('clip', h)
        const [lo, hi] = range()
        return mat(X.rows.map((r) => r.map((x, j) => {
          const v = x * scale[j] + min[j]
          return clip ? Math.min(hi, Math.max(lo, v)) : v
        })), X.m)
      }) as Fn,
    }
    return {
      ...transformerMixin(e, h, m),
      inverse_transform: (args, kw) => {
        requireFit(e, h)
        const a = readX(h, args[0] ?? kw.X, 'MinMaxScaler')
        const scale = e.st.scale as number[]
        const min = e.st.min as number[]
        return mat(a.rows.map((r) => r.map((x, j) => (x - min[j]) / scale[j])), a.m)
      },
      get_feature_names_out: (args, kw) => { requireFit(e, h); return py.obj(new ObjArray(featureNames(e, h, args[0] ?? kw.input_features), 'object')) },
    }
  },
}

const NORMALIZER: Spec = {
  name: 'Normalizer',
  pos: ['norm'],
  params: [['norm', py.str('l2')], ['copy', py.bool(true)]],
  methods: (e, h) => {
    const m = {
      fit: ((args, kw) => {
        const X = readX(h, args[0] ?? kw.X, 'Normalizer')
        rememberFeatures(e, X)
        e.st.fitted = true
        return py.obj(e)
      }) as Fn,
      transform: ((args, kw) => {
        const X = readX(h, args[0] ?? kw.X, 'Normalizer')
        if (e.st.m !== undefined && X.m !== e.st.m) throw h.err('ValueError', `X has ${X.m} features, but Normalizer is expecting ${e.st.m} features as input.`)
        const kind = strOf(e.param('norm'))
        if (kind !== 'l1' && kind !== 'l2' && kind !== 'max') throw badParam(h, `The 'norm' parameter of Normalizer must be a str among {'l1', 'l2', 'max'}. Got ${repr(e.param('norm'))} instead.`)
        const out = X.rows.map((r) => {
          const n = kind === 'l2' ? Math.sqrt(dot(r, r)) : kind === 'l1' ? npSum(r.map(Math.abs)) : Math.max(...r.map(Math.abs))
          const s = safeScale([n])[0]
          return r.map((x) => x / s)
        })
        return mat(out, X.m)
      }) as Fn,
    }
    return { ...transformerMixin(e, h, m), get_feature_names_out: (args, kw) => py.obj(new ObjArray(featureNames(e, h, args[0] ?? kw.input_features), 'object')) }
  },
}

const combosOf = (m: number, d: number, interaction: boolean): number[][] => {
  const out: number[][] = []
  const rec = (start: number, cur: number[]) => {
    if (cur.length === d) { out.push(cur); return }
    for (let i = start; i < m; i++) rec(interaction ? i + 1 : i, [...cur, i])
  }
  rec(0, [])
  return out
}

const POLY: Spec = {
  name: 'PolynomialFeatures',
  pos: ['degree'],
  params: [['degree', py.int(2)], ['interaction_only', py.bool(false)], ['include_bias', py.bool(true)], ['order', py.str('C')]],
  locked: ['order'],
  methods: (e, h) => {
    const degrees = (): [number, number] => {
      const d = e.param('degree')
      if (d.k === 'tuple' || d.k === 'list') {
        const [lo, hi] = h.iterate(d).map((v) => h.toInt(v))
        if (lo < 0 || hi < lo) throw h.err('ValueError', `degree=(min_degree, max_degree) must be non-negative integers that fulfil min_degree <= max_degree, got ${repr(d)}.`)
        return [lo, hi]
      }
      const k = h.toInt(d)
      if (k < 0) throw h.err('ValueError', `degree must be a non-negative integer or tuple (min_degree, max_degree), got ${k}.`)
      return [0, k]
    }
    const combos = (m: number): number[][] => {
      const [lo, hi] = degrees()
      const inter = e.flag('interaction_only', h)
      const out: number[][] = e.flag('include_bias', h) ? [[]] : []
      for (let d = Math.max(1, lo); d <= hi; d++) out.push(...combosOf(m, d, inter))
      return out
    }
    const m = {
      fit: ((args, kw) => {
        e.checkLocked(h)
        const X = readX(h, args[0] ?? kw.X, 'PolynomialFeatures')
        rememberFeatures(e, X)
        const cs = combos(X.m)
        if (!cs.length) throw h.err('ValueError', degrees()[1] === 0 ? 'Setting degree to zero and include_bias to False would result in an empty output array.' : 'need at least one array to concatenate')
        e.fitted.set('n_output_features_', py.int(cs.length))
        e.fitted.set('powers_', mat(cs.map((c) => { const p = new Array<number>(X.m).fill(0); c.forEach((i) => p[i]++); return p }), X.m, 'int64'))
        e.st.fitted = true
        return py.obj(e)
      }) as Fn,
      transform: ((args, kw) => {
        requireFit(e, h)
        const X = readX(h, args[0] ?? kw.X, 'PolynomialFeatures')
        if (X.m !== e.st.m) throw h.err('ValueError', `X has ${X.m} features, but PolynomialFeatures is expecting ${e.st.m} features as input.`)
        const [lo, hi] = degrees()
        const inter = e.flag('interaction_only', h)
        const bias = e.flag('include_bias', h)
        const out = X.rows.map((r) => {
          // sklearn builds each degree from the one before, which fixes the rounding
          const xp: number[] = bias ? [1] : []
          if (hi === 0) return xp
          let cur = xp.length
          xp.push(...r)
          let index = Array.from({ length: X.m }, (_, i) => cur + i)
          cur += X.m
          index.push(cur)
          for (let d = 2; d <= hi; d++) {
            const next: number[] = []
            const end = index[index.length - 1]
            for (let f = 0; f < X.m; f++) {
              let start = index[f]
              next.push(cur)
              if (inter) start += index[f + 1] - index[f]
              if (cur + end - start <= cur) break
              for (let k = start; k < end; k++) xp.push(xp[k] * r[f])
              cur += end - start
            }
            next.push(cur)
            index = next
          }
          if (lo > 1) {
            const total = combos(X.m).length
            return bias ? [1, ...xp.slice(xp.length - total + 1)] : xp.slice(xp.length - total)
          }
          return xp
        })
        return mat(out, combos(X.m).length)
      }) as Fn,
    }
    return {
      ...transformerMixin(e, h, m),
      get_feature_names_out: (args, kw) => {
        requireFit(e, h)
        const names = featureNames(e, h, args[0] ?? kw.input_features)
        return py.obj(new ObjArray(combos(e.st.m as number).map((c) => {
          if (!c.length) return '1'
          const counts = new Map<number, number>()
          c.forEach((i) => counts.set(i, (counts.get(i) ?? 0) + 1))
          return [...counts.entries()].map(([i, p]) => (p > 1 ? `${names[i]}^${p}` : names[i])).join(' ')
        }), 'object'))
      },
    }
  },
}

const FUNCTION_TRANSFORMER: Spec = {
  name: 'FunctionTransformer',
  pos: ['func', 'inverse_func'],
  params: [['func', py.NONE], ['inverse_func', py.NONE], ['validate', py.bool(false)], ['accept_sparse', py.bool(false)], ['check_inverse', py.bool(true)], ['feature_names_out', py.NONE], ['kw_args', py.NONE], ['inv_kw_args', py.NONE]],
  methods: (e, h) => {
    const kwOf = (v: Value): Kw => (v.k === 'dict' ? Object.fromEntries(v.d.items().map(([k, x]) => [h.str(k), x])) : {})
    const input = (v: Value): Value => {
      if (!e.flag('validate', h)) return v
      const X = readX(h, v, 'FunctionTransformer')
      return py.arr(X.a)
    }
    const apply = (X: Value, f: Value, kwv: Value) => (f.k === 'none' ? X : h.call(f, [X], kwOf(kwv)))
    const m = {
      fit: ((args, kw) => {
        const X = input(args[0] ?? kw.X)
        if (X.k === 'obj' && X.o instanceof Frame) rememberFeatures(e, { m: colNames(X.o).length, names: colNames(X.o), rows: [], n: 0, a: null as never })
        else if (X.k === 'array' && X.a.ndim === 2) rememberFeatures(e, { m: X.a.shape[1], names: null, rows: [], n: 0, a: X.a })
        e.st.fitted = true
        return py.obj(e)
      }) as Fn,
      transform: ((args, kw) => apply(input(args[0] ?? kw.X), e.param('func'), e.param('kw_args'))) as Fn,
    }
    return {
      ...transformerMixin(e, h, m),
      inverse_transform: (args, kw) => apply(input(args[0] ?? kw.X), e.param('inverse_func'), e.param('inv_kw_args')),
      get_feature_names_out: (args, kw) => {
        const how = e.param('feature_names_out')
        if (how.k === 'none') throw h.err('AttributeError', "This 'FunctionTransformer' has no attribute 'get_feature_names_out'")
        if (how.k === 'str' && how.v === 'one-to-one') return py.obj(new ObjArray(featureNames(e, h, args[0] ?? kw.input_features), 'object'))
        throw h.err('NotImplementedError', 'FunctionTransformer: a callable feature_names_out is not available in this sandbox')
      },
    }
  },
}

const LABEL_ENCODER: Spec = {
  name: 'LabelEncoder',
  params: [],
  methods: (e, h) => {
    const fit = (yV: Value | undefined): Labels => {
      if (!yV) throw h.err('TypeError', "LabelEncoder.fit() missing 1 required positional argument: 'y'")
      const y = labelsOf(h, yV)
      const cl = uniqueSorted(y.vals, y.isStr)
      e.st.classes = cl
      e.st.labels = y
      e.fitted.set('classes_', labelArray(cl, y))
      e.st.fitted = true
      return y
    }
    const encode = (y: Labels): Value => {
      const cl = e.st.classes as Lab[]
      const at = new Map(cl.map((c, i) => [c, i]))
      const unseen = y.vals.filter((v) => !at.has(v))
      if (unseen.length) {
        // text fails on the first label it cannot look up; numbers report them all
        const first = String(unseen[0])
        const text = y.isStr ? (y.dtype === 'object' ? reprStr(first) : `np.str_(${reprStr(first)})`) : `[${uniqueSorted(unseen, false).map((d) => `np.${y.dtype}(${d})`).join(', ')}]`
        throw h.err('ValueError', `y contains previously unseen labels: ${text}`)
      }
      return vec(y.vals.map((v) => at.get(v)!), 'int64')
    }
    return {
      fit: (args, kw) => { fit(args[0] ?? kw.y); return py.obj(e) },
      fit_transform: (args, kw) => encode(fit(args[0] ?? kw.y)),
      transform: (args, kw) => {
        requireFit(e, h)
        const y = labelsOf(h, args[0] ?? kw.y)
        if (!y.vals.length) return vec([], 'int64')
        if (y.isStr !== (e.st.labels as Labels).isStr) throw h.err('ValueError', `y contains previously unseen labels: ${reprStr(String(y.vals[0]))}`)
        return encode(y)
      },
      inverse_transform: (args, kw) => {
        requireFit(e, h)
        const a = toArray(args[0] ?? kw.y)
        const cl = e.st.classes as Lab[]
        if (a.ndim !== 1 && !(a.ndim === 2 && a.shape[1] === 1)) throw h.err('ValueError', `y should be a 1d array, got an array of shape (${a.shape.join(', ')}) instead.`)
        const bad = uniqueSorted(a.values().filter((i) => !(Number.isInteger(i) && i >= 0 && i < cl.length)), false)
        if (bad.length) throw h.err('ValueError', `y contains previously unseen labels: ${h.str(py.arr(NDArray.create(bad as number[], [bad.length], a.dtype)))}`)
        return labelArray(a.values().map((i) => cl[i]), e.st.labels as Labels)
      },
    }
  },
}

interface Table { cols: Labels[]; n: number; names: string[] | null }

/** check_array(dtype=None) for encoders: one column of labels per feature */
const readTable = (h: Host, v: Value | undefined, est: string): Table => {
  if (!v) throw h.err('TypeError', `${est}.fit() missing 1 required positional argument: 'X'`)
  const need2d = (arr: string) => h.err('ValueError', `Expected 2D array, got 1D array instead:\narray=${arr}.\nReshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.`)
  if (v.k === 'obj' && v.o instanceof Frame) return { cols: v.o.data.map((s) => labelsOf(h, py.obj(s))), n: v.o.nrows, names: colNames(v.o) }
  if (isForeignArray(v)) {
    const rows = h.iterate(v).map((x) => (isForeignArray(x) ? py.list(h.iterate(x)) : x))
    return readTable(h, py.list(rows), est)
  }
  if (v.k === 'obj' && v.o instanceof Series) {
    if (v.o.dtype === 'object') throw need2d(new ObjArray(v.o.values.map(String), 'object').str())
    throw need2d(h.str(py.arr(v.o.toArray())))
  }
  if (v.k === 'obj' && v.o instanceof ObjArray) {
    if (v.o.shape.length === 1) throw need2d(v.o.str())
    const m = v.o.shape[1]
    const items = v.o.items
    return { cols: Array.from({ length: m }, (_, j) => ({ vals: items.filter((_x, i) => i % m === j), isStr: true, dtype: (v.o as ObjArray).dtype })), n: v.o.shape[0], names: null }
  }
  if ((v.k === 'list' || v.k === 'tuple') && firstString(v) !== null) {
    if (v.items.some((r) => r.k !== 'list' && r.k !== 'tuple')) throw need2d(new ObjArray(v.items.map((x) => h.str(x))).str())
    const rows = v.items.map((r) => h.iterate(r))
    const m = rows[0]?.length ?? 0
    if (rows.some((r) => r.length !== m)) throw h.err('ValueError', 'setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions.')
    return {
      cols: Array.from({ length: m }, (_, j) => {
        const col = rows.map((r) => r[j])
        return col.some((c) => c.k === 'str') ? { vals: col.map((c) => h.str(c)), isStr: true, dtype: 'object' } : { vals: col.map((c) => h.num(c)), isStr: false, dtype: col.some((c) => c.k === 'float') ? 'float64' : 'int64' }
      }),
      n: rows.length,
      names: null,
    }
  }
  const a = numericArray(h, v)
  if (a.ndim === 1) throw need2d(h.str(py.arr(a)))
  if (a.ndim !== 2) throw h.err('ValueError', `Found array with dim ${a.ndim}, while dim <= 2 is required by ${est}.`)
  return { cols: colsOf(rowsOf(a), a.shape[1]).map((c) => ({ vals: c, isStr: false, dtype: a.dtype })), n: a.shape[0], names: null }
}

const pyDtypeIsFloat = (h: Host, v: Value): boolean => {
  if (v.k === 'type' && v.dtype) return v.dtype === 'float64' || v.dtype === 'float32'
  if (v.k === 'dtype') return v.d === 'float64'
  if (v.k === 'str') return v.v.startsWith('float')
  throw h.err('TypeError', `Cannot interpret ${repr(v)} as a data type`)
}

/** the categories of each column: learned, or given by the user */
const fitCategories = (h: Host, e: Est, t: Table): Lab[][] => {
  for (const c of t.cols) if (!c.isStr && (c.vals as number[]).some(Number.isNaN)) throw h.err('NotImplementedError', `${e.cls}: missing values (NaN) are not available in this sandbox`)
  const given = e.param('categories')
  let cats: Lab[][]
  if (given.k === 'str') cats = t.cols.map((c) => uniqueSorted(c.vals, c.isStr))
  else {
    const lists = h.iterate(given).map((x) => labelsOf(h, x))
    if (lists.length !== t.cols.length) throw h.err('ValueError', 'Shape mismatch: if categories is an array, it has to be of shape (n_features,).')
    cats = lists.map((l) => l.vals)
    cats.forEach((cat, j) => {
      const sorted = !t.cols[j].isStr ? [...(cat as number[])].sort((a, b) => a - b) : null
      if (sorted && sorted.join() !== cat.join()) throw h.err('ValueError', 'Unsorted categories are not supported for numerical categories')
      if (e.param('handle_unknown').k === 'str' && strOf(e.param('handle_unknown')) === 'error') {
        const bad = t.cols[j].vals.filter((v) => !cat.includes(v))
        if (bad.length) throw h.err('ValueError', `Found unknown categories ${unknownText([...new Set(bad)], t.cols[j])} in column ${j} during fit`)
      }
    })
  }
  e.st.cats = cats
  e.st.catTypes = t.cols.map((c) => ({ isStr: c.isStr, dtype: c.dtype }))
  e.fitted.set('categories_', py.list(cats.map((c, j) => labelArray(c, t.cols[j]))))
  e.st.names = t.names
  e.st.m = t.cols.length
  e.fitted.set('n_features_in_', py.int(t.cols.length))
  if (t.names) e.fitted.set('feature_names_in_', py.obj(new ObjArray(t.names, 'object')))
  e.st.fitted = true
  return cats
}

const unknownText = (bad: Lab[], col: Labels): string => {
  const items = uniqueSorted(bad, col.isStr)
  return `[${items.map((x) => (typeof x === 'string' ? (col.dtype === 'object' ? reprStr(x) : `np.str_(${reprStr(x)})`) : `np.${col.dtype}(${x})`)).join(', ')}]`
}

const readTransformTable = (h: Host, e: Est, v: Value | undefined): Table => {
  requireFit(e, h)
  const t = readTable(h, v, e.cls)
  if (t.cols.length !== e.st.m) throw h.err('ValueError', `X has ${t.cols.length} features, but ${e.cls} is expecting ${e.st.m} features as input.`)
  checkNames(e, t, h)
  return t
}

const ONE_HOT: Spec = {
  name: 'OneHotEncoder',
  params: [['categories', py.str('auto')], ['drop', py.NONE], ['sparse_output', py.bool(true)], ['dtype', { k: 'type', name: 'numpy.float64', dtype: 'float64', call: () => py.NONE }], ['handle_unknown', py.str('error')], ['min_frequency', py.NONE], ['max_categories', py.NONE], ['feature_name_combiner', py.str('concat')]],
  locked: ['min_frequency', 'max_categories', 'feature_name_combiner'],
  methods: (e, h) => {
    const unknown = () => {
      const u = strOf(e.param('handle_unknown'))
      if (u !== 'error' && u !== 'ignore') {
        if (u === 'infrequent_if_exist' || u === 'warn') throw h.err('NotImplementedError', `OneHotEncoder: handle_unknown=${reprStr(u)} is not available in this sandbox`)
        throw badParam(h, `The 'handle_unknown' parameter of OneHotEncoder must be a str among {'error', 'ignore', 'infrequent_if_exist', 'warn'}. Got ${repr(e.param('handle_unknown'))} instead.`)
      }
      return u
    }
    const dropIdx = (cats: Lab[][]): (number | null)[] => {
      const d = e.param('drop')
      if (d.k === 'none') return cats.map(() => null)
      if (d.k === 'str' && d.v === 'first') return cats.map(() => 0)
      if (d.k === 'str' && d.v === 'if_binary') return cats.map((c) => (c.length === 2 ? 0 : null))
      throw h.err('NotImplementedError', `OneHotEncoder: drop=${repr(d)} is not available in this sandbox`)
    }
    const m = {
      fit: ((args, kw) => {
        e.checkLocked(h)
        unknown()
        const cats = fitCategories(h, e, readTable(h, args[0] ?? kw.X, 'OneHotEncoder'))
        const di = dropIdx(cats)
        e.st.drop = di
        e.fitted.set('drop_idx_', e.param('drop').k === 'none' ? py.NONE : py.arr(NDArray.create(di.map((x) => x ?? -1), [di.length], 'int64')))
        return py.obj(e)
      }) as Fn,
      transform: ((args, kw) => {
        const t = readTransformTable(h, e, args[0] ?? kw.X)
        const cats = e.st.cats as Lab[][]
        const di = e.st.drop as (number | null)[]
        const u = unknown()
        const widths = cats.map((c, j) => c.length - (di[j] === null ? 0 : 1))
        const total = widths.reduce((a, b) => a + b, 0)
        const rows = Array.from({ length: t.n }, () => new Array<number>(total).fill(0))
        let off = 0
        t.cols.forEach((col, j) => {
          const at = new Map(cats[j].map((c, i) => [c, i]))
          const bad: Lab[] = []
          col.vals.forEach((v, r) => {
            const c = at.get(v)
            if (c === undefined) { bad.push(v); return }
            const d = di[j]
            if (d !== null && c === d) return
            rows[r][off + (d !== null && c > d ? c - 1 : c)] = 1
          })
          if (bad.length && u === 'error') throw h.err('ValueError', `Found unknown categories ${unknownText(bad, col)} in column ${j} during transform`)
          off += widths[j]
        })
        if (e.flag('sparse_output', h)) {
          if (!pyDtypeIsFloat(h, e.param('dtype'))) throw h.err('NotImplementedError', 'OneHotEncoder: a sparse result is float64 in this sandbox; use sparse_output=False for another dtype')
          return py.obj(new Csr(rows, total))
        }
        return mat(rows, total, pyDtypeIsFloat(h, e.param('dtype')) ? 'float64' : 'int64')
      }) as Fn,
    }
    return {
      ...transformerMixin(e, h, m),
      inverse_transform: (args, kw) => {
        requireFit(e, h)
        const a0 = args[0] ?? kw.X
        const a = a0.k === 'obj' && a0.o instanceof Csr ? a0.o.dense() : numericArray(h, a0)
        const cats = e.st.cats as Lab[][]
        const di = e.st.drop as (number | null)[]
        const types = e.st.catTypes as Labels[]
        const widths = cats.map((c, j) => c.length - (di[j] === null ? 0 : 1))
        if (a.ndim !== 2 || a.shape[1] !== widths.reduce((x, y) => x + y, 0)) throw h.err('ValueError', `Shape of the passed X data is not correct. Expected ${cats.length} columns, got ${a.shape[1]}.`)
        const rows = rowsOf(a)
        const out: Lab[][] = rows.map((r) => {
          let off = 0
          return cats.map((c, j) => {
            const part = r.slice(off, off + widths[j])
            off += widths[j]
            const hit = part.some((x) => x !== 0) ? argmax(part) : -1
            if (hit < 0) {
              if (di[j] === null) throw h.err('NotImplementedError', 'OneHotEncoder.inverse_transform: an all-zero row (an unknown category) gives None in scikit-learn, which this sandbox cannot hold in a text array')
              return c[di[j]!]
            }
            const d = di[j]
            return c[d !== null && hit >= d ? hit + 1 : hit]
          })
        })
        if (types.some((t) => t.isStr)) return py.obj(new ObjArray(out.flat().map(String), types.every((t) => t.dtype === 'object' || !t.isStr) ? 'object' : null, [rows.length, cats.length]))
        return mat(out as number[][], cats.length, types.every((t) => t.dtype === 'int64') ? 'int64' : 'float64')
      },
      get_feature_names_out: (args, kw) => {
        requireFit(e, h)
        const names = featureNames(e, h, args[0] ?? kw.input_features)
        const cats = e.st.cats as Lab[][]
        const di = e.st.drop as (number | null)[]
        return py.obj(new ObjArray(cats.flatMap((c, j) => c.filter((_x, i) => di[j] === null || i !== di[j]).map((x) => `${names[j]}_${x}`)), 'object'))
      },
    }
  },
}

const ORDINAL: Spec = {
  name: 'OrdinalEncoder',
  params: [['categories', py.str('auto')], ['dtype', { k: 'type', name: 'numpy.float64', dtype: 'float64', call: () => py.NONE }], ['handle_unknown', py.str('error')], ['unknown_value', py.NONE], ['encoded_missing_value', py.float(NaN)], ['min_frequency', py.NONE], ['max_categories', py.NONE]],
  locked: ['min_frequency', 'max_categories'],
  methods: (e, h) => {
    const m = {
      fit: ((args, kw) => {
        e.checkLocked(h)
        const u = strOf(e.param('handle_unknown'))
        const uv = e.param('unknown_value')
        if (u === 'use_encoded_value' && uv.k === 'none') throw h.err('TypeError', `unknown_value should be an integer or np.nan when handle_unknown is 'use_encoded_value', got None.`)
        if (u === 'error' && uv.k !== 'none') throw h.err('TypeError', `unknown_value should only be set when handle_unknown is 'use_encoded_value', got ${repr(uv)}.`)
        fitCategories(h, e, readTable(h, args[0] ?? kw.X, 'OrdinalEncoder'))
        return py.obj(e)
      }) as Fn,
      transform: ((args, kw) => {
        const t = readTransformTable(h, e, args[0] ?? kw.X)
        const cats = e.st.cats as Lab[][]
        const useValue = strOf(e.param('handle_unknown')) === 'use_encoded_value'
        const uv = useValue ? h.num(e.param('unknown_value')) : NaN
        const out = Array.from({ length: t.n }, () => new Array<number>(cats.length).fill(0))
        t.cols.forEach((col, j) => {
          const at = new Map(cats[j].map((c, i) => [c, i]))
          const bad: Lab[] = []
          col.vals.forEach((v, r) => {
            const c = at.get(v)
            if (c === undefined) {
              bad.push(v)
              out[r][j] = uv
            } else out[r][j] = c
          })
          if (bad.length && !useValue) throw h.err('ValueError', `Found unknown categories ${unknownText(bad, col)} in column ${j} during transform`)
        })
        return mat(out, cats.length, pyDtypeIsFloat(h, e.param('dtype')) ? 'float64' : 'int64')
      }) as Fn,
    }
    return {
      ...transformerMixin(e, h, m),
      inverse_transform: (args, kw) => {
        requireFit(e, h)
        const a = readX(h, args[0] ?? kw.X, 'OrdinalEncoder')
        const cats = e.st.cats as Lab[][]
        const types = e.st.catTypes as Labels[]
        if (a.m !== cats.length) throw h.err('ValueError', `Shape of the passed X data is not correct. Expected ${cats.length} columns, got ${a.m}.`)
        const out = a.rows.map((r) => r.map((x, j) => cats[j][x]))
        if (types.some((t) => t.isStr)) return py.obj(new ObjArray(out.flat().map(String), 'object', [a.n, a.m]))
        return mat(out as number[][], a.m, types.every((t) => t.dtype === 'int64') ? 'int64' : 'float64')
      },
      get_feature_names_out: (args, kw) => { requireFit(e, h); return py.obj(new ObjArray(featureNames(e, h, args[0] ?? kw.input_features), 'object')) },
    }
  },
}

// ---------------------------------------------------------------- compose and pipeline

const denseOf = (h: Host, v: Value): { rows: number[][]; m: number; sparse: boolean; nnz: number } => {
  if (v.k === 'obj' && v.o instanceof Csr) return { rows: v.o.rows, m: v.o.m, sparse: true, nnz: v.o.nnz }
  const a = numericArray(h, v)
  if (a.ndim !== 2) throw h.err('ValueError', 'The output of the transformer should be 2D (numpy array, scipy sparse array, dataframe).')
  return { rows: rowsOf(a), m: a.shape[1], sparse: false, nnz: a.size }
}

const callFitTransform = (h: Host, t: Value, X: Value, y?: Value): Value => {
  const est = estOf(t)
  const has = (n: string) => (est ? !!est.spec.methods(est, h)[n] : t.k === 'inst' && !!(t as unknown as { cls: { ns: Map<string, Value> } }).cls.ns.get(n))
  const args = y && y.k !== 'none' ? [X, y] : [X]
  if (has('fit_transform')) return h.call(h.getAttr(t, 'fit_transform'), args)
  const fitted = h.call(h.getAttr(t, 'fit'), args)
  return h.call(h.getAttr(fitted, 'transform'), [X])
}

const selectColumns = (h: Host, X: Value, col: Value): { value: Value; idx: number[] } => {
  const nCols = X.k === 'obj' && X.o instanceof Frame ? colNames(X.o).length : X.k === 'array' && X.a.ndim === 2 ? X.a.shape[1] : -1
  if (nCols < 0) throw h.err('ValueError', 'ColumnTransformer needs a 2D array or a DataFrame')
  const c = col.k === 'func' || col.k === 'fn' ? h.call(col, [X]) : col
  const frame = X.k === 'obj' && X.o instanceof Frame ? X.o : null
  const byName = (n: string) => {
    if (!frame) throw h.err('ValueError', 'Specifying the columns using strings is only supported for dataframes.')
    const k = colNames(frame).indexOf(n)
    if (k < 0) throw h.err('ValueError', 'A given column is not a column of the dataframe')
    return k
  }
  const pick = (idx: number[], single: boolean): { value: Value; idx: number[] } => {
    if (frame) return { value: single ? py.obj(frame.data[idx[0]]) : py.obj(frame.take(null, idx)), idx }
    const a = (X as { a: NDArray }).a
    const rows = rowsOf(a)
    return single ? { value: py.arr(NDArray.create(rows.map((r) => r[idx[0]]), [rows.length], a.dtype)), idx } : { value: py.arr(NDArray.create(rows.flatMap((r) => idx.map((j) => r[j])), [rows.length, idx.length], a.dtype)), idx }
  }
  if (c.k === 'str') return pick([byName(c.v)], true)
  if (c.k === 'int') return pick([c.v < 0 ? c.v + nCols : c.v], true)
  if (c.k === 'slice') {
    const part = (x: Value, d: number) => (x.k === 'none' ? d : x.k === 'str' ? byName(x.v) : h.toInt(x))
    const lo = part(c.start, 0)
    const hi = c.stop.k === 'str' ? byName(c.stop.v) + 1 : part(c.stop, nCols)
    return pick(Array.from({ length: Math.max(0, hi - lo) }, (_, i) => lo + i), false)
  }
  const items = c.k === 'obj' && c.o instanceof ObjArray ? c.o.items.map((s) => py.str(s)) : c.k === 'obj' && c.o instanceof Series ? h.iterate(c) : c.k === 'array' ? h.iterate(c) : c.k === 'obj' && c.o.iter ? c.o.iter(h) : h.iterate(c)
  if (items.every((x) => x.k === 'str')) return pick(items.map((x) => byName((x as { v: string }).v)), false)
  if (items.every((x) => x.k === 'bool' || (x.k === 'array' && x.a.dtype === 'bool'))) return pick(items.flatMap((x, i) => (h.truthy(x) ? [i] : [])), false)
  return pick(items.map((x) => { const i = h.toInt(x); return i < 0 ? i + nCols : i }), false)
}

const COLUMN_TRANSFORMER: Spec = {
  name: 'ColumnTransformer',
  pos: ['transformers'],
  params: [['transformers', null], ['remainder', py.str('drop')], ['sparse_threshold', py.float(0.3)], ['n_jobs', py.NONE], ['transformer_weights', py.NONE], ['verbose', py.bool(false)], ['verbose_feature_names_out', py.bool(true)], ['force_int_remainder_cols', py.str('deprecated')]],
  locked: ['transformer_weights'],
  named: (e, h) => h.iterate(e.param('transformers')).map((t) => { const [n, tr] = h.iterate(t); return [h.str(n), tr] as [string, Value] }),
  methods: (e, h) => {
    const specOf = (X: Value) => {
      const list = h.iterate(e.param('transformers')).map((t) => {
        const parts = h.iterate(t)
        if (parts.length !== 3) throw h.err('ValueError', 'not enough values to unpack (expected 3, got ' + parts.length + ')')
        const s = selectColumns(h, X, parts[2])
        return { name: h.str(parts[0]), tr: parts[1], sel: s, col: parts[2] }
      })
      const nCols = X.k === 'obj' && X.o instanceof Frame ? colNames(X.o).length : (X as { a: NDArray }).a.shape[1]
      const used = new Set(list.flatMap((t) => t.sel.idx))
      const rest = Array.from({ length: nCols }, (_, i) => i).filter((i) => !used.has(i))
      return { list, rest }
    }
    const isDrop = (t: Value) => t.k === 'str' && t.v === 'drop'
    const isPass = (t: Value) => t.k === 'str' && t.v === 'passthrough'
    const stack = (outs: { rows: number[][]; m: number; sparse: boolean; nnz: number }[], n: number): Value => {
      const total = outs.reduce((a, o) => a + o.m, 0)
      const rows = Array.from({ length: n }, (_, i) => outs.flatMap((o) => o.rows[i]))
      if (outs.some((o) => o.sparse)) {
        const nnz = outs.reduce((a, o) => a + o.nnz, 0)
        const cells = outs.reduce((a, o) => a + o.rows.length * o.m, 0)
        const out = nnz / cells < e.num('sparse_threshold', h)
        e.fitted.set('sparse_output_', py.bool(out))
        if (out) return py.obj(new Csr(rows, total))
      } else e.fitted.set('sparse_output_', py.bool(false))
      return mat(rows, total)
    }
    const fitTransform: Fn = (args, kw) => {
      e.checkLocked(h)
      const X = args[0] ?? kw.X
      const y = args[1] ?? kw.y
      if (X.k === 'list' || X.k === 'tuple') throw h.err('ValueError', 'ColumnTransformer takes an array or a DataFrame in this sandbox')
      const { list, rest } = specOf(X)
      const rem = e.param('remainder')
      const n = numSamples(h, X)
      const outs: ReturnType<typeof denseOf>[] = []
      const fittedList: Value[] = []
      const indices: [string, Value][] = []
      const features: { name: string; tr: Value; idx: number[]; width: number; col: Value }[] = []
      let at = 0
      const run = (name: string, tr: Value, sel: { value: Value; idx: number[] }, col: Value) => {
        if (isDrop(tr) || !sel.idx.length) {
          fittedList.push(py.tuple([py.str(name), tr, col]))
          indices.push([name, { k: 'slice', start: py.int(0), stop: py.int(0), step: py.NONE }])
          return
        }
        const clone = isPass(tr) ? tr : cloneValue(tr)
        const out = denseOf(h, isPass(tr) ? sel.value : callFitTransform(h, clone, sel.value, y))
        outs.push(out)
        fittedList.push(py.tuple([py.str(name), clone, col]))
        indices.push([name, { k: 'slice', start: py.int(at), stop: py.int(at + out.m), step: py.NONE }])
        features.push({ name, tr: clone, idx: sel.idx, width: out.m, col })
        at += out.m
      }
      for (const t of list) run(t.name, t.tr, t.sel, t.col)
      const frame = X.k === 'obj' && X.o instanceof Frame ? X.o : null
      const restCols = py.list(rest.map((i) => (frame ? py.str(colNames(frame)[i]) : py.int(i))))
      if (rem.k === 'str' && rem.v === 'drop') {
        if (rest.length) fittedList.push(py.tuple([py.str('remainder'), py.str('drop'), restCols]))
        indices.push(['remainder', { k: 'slice', start: py.int(0), stop: py.int(0), step: py.NONE }])
      } else if (rest.length) run('remainder', rem, selectColumns(h, X, restCols), restCols)
      else indices.push(['remainder', { k: 'slice', start: py.int(0), stop: py.int(0), step: py.NONE }])
      e.st.rest = rest
      e.st.features = features
      e.st.names = frame ? colNames(frame) : null
      e.st.m = frame ? colNames(frame).length : (X as { a: NDArray }).a.shape[1]
      e.fitted.set('n_features_in_', py.int(e.st.m as number))
      if (frame) e.fitted.set('feature_names_in_', py.obj(new ObjArray(colNames(frame), 'object')))
      e.fitted.set('transformers_', py.list(fittedList))
      e.fitted.set('named_transformers_', py.obj(new BunchObj(fittedList.map((t) => { const [nm, tr] = h.iterate(t); return [h.str(nm), tr] as [string, Value] }))))
      e.fitted.set('output_indices_', dictOf(indices))
      e.st.fitted = true
      if (!outs.length) return mat(Array.from({ length: n }, () => []), 0)
      return stack(outs, n)
    }
    return {
      fit: (args, kw) => { fitTransform(args, kw); return py.obj(e) },
      fit_transform: fitTransform,
      transform: (args, kw) => {
        requireFit(e, h)
        const X = args[0] ?? kw.X
        const n = numSamples(h, X)
        const feats = e.st.features as { name: string; tr: Value; col: Value; idx: number[] }[]
        const outs = feats.map((f) => {
          const sel = selectColumns(h, X, f.col)
          return denseOf(h, f.tr.k === 'str' ? sel.value : h.call(h.getAttr(f.tr, 'transform'), [sel.value]))
        })
        if (!outs.length) return mat(Array.from({ length: n }, () => []), 0)
        return stack(outs, n)
      },
      get_feature_names_out: () => {
        requireFit(e, h)
        const feats = e.st.features as { name: string; tr: Value; idx: number[] }[]
        const names = (e.st.names as string[] | null) ?? Array.from({ length: e.st.m as number }, (_, i) => `x${i}`)
        const pref = e.flag('verbose_feature_names_out', h)
        return py.obj(new ObjArray(feats.flatMap((f) => {
          const inputs = f.idx.map((i) => names[i])
          const inner = f.tr.k === 'str' ? inputs : labelsOf(h, h.call(h.getAttr(f.tr, 'get_feature_names_out'), [py.list(inputs.map((s) => py.str(s)))])).vals.map(String)
          return inner.map((s) => (pref ? `${f.name}__${s}` : s))
        }), 'object'))
      },
    }
  },
}

/** sklearn.utils.Bunch: a dict whose keys are also attributes */
class BunchObj extends PyObj {
  readonly cls = 'Bunch'
  items: [string, Value][]
  constructor(items: [string, Value][]) {
    super()
    this.items = items
  }
  repr() { return `{${this.items.map(([k, v]) => `${reprStr(k)}: ${repr(v)}`).join(', ')}}` }
  getAttr(name: string): Value | undefined {
    const hit = this.items.find(([k]) => k === name)
    if (hit) return hit[1]
    if (name === 'keys') return { k: 'fn', name, call: () => py.list(this.items.map(([k]) => py.str(k))) }
    if (name === 'values') return { k: 'fn', name, call: () => py.list(this.items.map(([, v]) => v)) }
    if (name === 'items') return { k: 'fn', name, call: () => py.list(this.items.map(([k, v]) => py.tuple([py.str(k), v]))) }
    return undefined
  }
  getItem(idx: Value, h: Host): Value {
    if (idx.k === 'str') {
      const hit = this.items.find(([k]) => k === idx.v)
      if (hit) return hit[1]
    }
    throw h.err('KeyError', repr(idx))
  }
  iter() { return this.items.map(([k]) => py.str(k)) }
  len() { return this.items.length }
  contains(v: Value) { return v.k === 'str' && this.items.some(([k]) => k === v.v) }
}

const PIPELINE: Spec = {
  name: 'Pipeline',
  pos: ['steps'],
  params: [['steps', null], ['transform_input', py.NONE], ['memory', py.NONE], ['verbose', py.bool(false)]],
  locked: ['transform_input'],
  named: (e, h) => h.iterate(e.param('steps')).map((s) => { const [n, t] = h.iterate(s); return [h.str(n), t] as [string, Value] }),
  setNamed: (e, name, v, h) => {
    const steps = h.iterate(e.param('steps')).map((s) => { const [n, t] = h.iterate(s); return h.str(n) === name ? py.tuple([n, v]) : py.tuple([n, t]) })
    e.params.set('steps', py.list(steps))
    return true
  },
  len: (e, h) => h.iterate(e.param('steps')).length,
  getItem: (e, idx, h) => {
    const steps = h.iterate(e.param('steps'))
    if (idx.k === 'str') {
      const hit = steps.find((s) => h.str(h.iterate(s)[0]) === idx.v)
      if (!hit) throw h.err('KeyError', repr(idx))
      return h.iterate(hit)[1]
    }
    if (idx.k === 'slice') {
      const part = (x: Value, d: number) => { if (x.k === 'none') return d; const i = h.toInt(x); return i < 0 ? Math.max(0, i + steps.length) : Math.min(i, steps.length) }
      if (idx.step.k !== 'none' && h.toInt(idx.step) !== 1) throw h.err('ValueError', 'Pipeline slicing only supports a step of 1')
      const sub = new Est(PIPELINE)
      sub.params = new Map(e.params)
      sub.params.set('steps', py.list(steps.slice(part(idx.start, 0), part(idx.stop, steps.length))))
      sub.st.fitted = e.st.fitted
      return py.obj(sub)
    }
    const i = h.toInt(idx)
    const at = i < 0 ? i + steps.length : i
    if (at < 0 || at >= steps.length) throw h.err('IndexError', 'list index out of range')
    return h.iterate(steps[at])[1]
  },
  extra: (e, name, h) => {
    if (name === 'named_steps') return py.obj(new BunchObj(PIPELINE.named!(e, h)))
    const steps = PIPELINE.named!(e, h)
    if (name === 'classes_' || name === 'n_features_in_' || name === 'feature_names_in_') {
      const holder = name === 'classes_' ? steps[steps.length - 1][1] : steps[0][1]
      return holder.k === 'str' ? undefined : h.getAttr(holder, name)
    }
    return undefined
  },
  methods: (e, h) => {
    const steps = () => PIPELINE.named!(e, h)
    const checkSteps = () => {
      const s = steps()
      s.slice(0, -1).forEach(([, t]) => {
        if (t.k === 'none' || (t.k === 'str' && t.v === 'passthrough')) return
        const est = estOf(t)
        if (est && (!est.spec.methods(est, h).transform)) throw h.err('TypeError', `All intermediate steps should be transformers and implement fit and transform or be the string 'passthrough' '${est.repr()}' (type <class '${est.cls}'>) doesn't`)
      })
      return s
    }
    const skip = (t: Value) => t.k === 'none' || (t.k === 'str' && t.v === 'passthrough')
    /** fit the transformers, in place, and return what reaches the final step */
    const fitHead = (X: Value, y?: Value): Value => {
      let cur = X
      const s = checkSteps()
      for (const [, t] of s.slice(0, -1)) if (!skip(t)) cur = callFitTransform(h, t, cur, y)
      return cur
    }
    const through = (X: Value): Value => {
      let cur = X
      for (const [, t] of steps().slice(0, -1)) if (!skip(t)) cur = h.call(h.getAttr(t, 'transform'), [cur])
      return cur
    }
    const finalCall = (name: string): Fn => (args, kw) => {
      const s = steps()
      return h.call(h.getAttr(s[s.length - 1][1], name), [through(args[0] ?? kw.X), ...args.slice(1)], Object.fromEntries(Object.entries(kw).filter(([k]) => k !== 'X')))
    }
    const lastHas = (name: string) => {
      const s = steps()
      const last = estOf(s[s.length - 1][1])
      return !last || !!last.spec.methods(last, h)[name]
    }
    const ms: Record<string, Fn> = {
      fit: (args, kw) => {
        const y = args[1] ?? kw.y
        const Xt = fitHead(args[0] ?? kw.X, y)
        const s = steps()
        const last = s[s.length - 1][1]
        if (!skip(last)) h.call(h.getAttr(last, 'fit'), y && y.k !== 'none' ? [Xt, y] : [Xt])
        e.st.fitted = true
        return py.obj(e)
      },
      score: (args, kw) => {
        const s = steps()
        const y = args[1] ?? kw.y
        return h.call(h.getAttr(s[s.length - 1][1], 'score'), y ? [through(args[0] ?? kw.X), y] : [through(args[0] ?? kw.X)])
      },
      get_feature_names_out: (args, kw) => {
        let names = args[0] ?? kw.input_features ?? py.NONE
        for (const [, t] of steps()) if (!skip(t)) names = h.call(h.getAttr(t, 'get_feature_names_out'), [names])
        return names
      },
    }
    if (lastHas('predict')) ms.predict = finalCall('predict')
    if (lastHas('predict_proba')) ms.predict_proba = finalCall('predict_proba')
    if (lastHas('predict_log_proba')) ms.predict_log_proba = finalCall('predict_log_proba')
    if (lastHas('decision_function')) ms.decision_function = finalCall('decision_function')
    const s0 = steps()
    const lastV = s0[s0.length - 1][1]
    if (skip(lastV) || lastHas('transform')) {
      ms.transform = (args, kw) => {
        const s = steps()
        const last = s[s.length - 1][1]
        const Xt = through(args[0] ?? kw.X)
        return skip(last) ? Xt : h.call(h.getAttr(last, 'transform'), [Xt])
      }
      ms.fit_transform = (args, kw) => {
        const y = args[1] ?? kw.y
        const Xt = fitHead(args[0] ?? kw.X, y)
        const s = steps()
        const last = s[s.length - 1][1]
        e.st.fitted = true
        return skip(last) ? Xt : callFitTransform(h, last, Xt, y)
      }
      ms.inverse_transform = (args, kw) => {
        let cur = args[0] ?? kw.X
        for (const [, t] of [...steps()].reverse()) if (!skip(t)) cur = h.call(h.getAttr(t, 'inverse_transform'), [cur])
        return cur
      }
    }
    if (lastHas('fit_predict')) {
      ms.fit_predict = (args, kw) => {
        const y = args[1] ?? kw.y
        const Xt = fitHead(args[0] ?? kw.X, y)
        const s = steps()
        e.st.fitted = true
        return h.call(h.getAttr(s[s.length - 1][1], 'fit_predict'), y ? [Xt, y] : [Xt])
      }
    }
    return ms
  },
}

// ---------------------------------------------------------------- nearest neighbours

const dualSwap = (d: number[], ix: number[], lo: number, a: number, b: number) => {
  const t = d[lo + a]
  d[lo + a] = d[lo + b]
  d[lo + b] = t
  const u = ix[lo + a]
  ix[lo + a] = ix[lo + b]
  ix[lo + b] = u
}

/** simultaneous_sort: not stable, so the order among equal distances is sklearn's own */
const simultaneousSort = (d: number[], ix: number[], lo: number, size: number) => {
  if (size <= 1) return
  if (size === 2) {
    if (d[lo] > d[lo + 1]) dualSwap(d, ix, lo, 0, 1)
    return
  }
  if (size === 3) {
    if (d[lo] > d[lo + 1]) dualSwap(d, ix, lo, 0, 1)
    if (d[lo + 1] > d[lo + 2]) {
      dualSwap(d, ix, lo, 1, 2)
      if (d[lo] > d[lo + 1]) dualSwap(d, ix, lo, 0, 1)
    }
    return
  }
  let p = size >> 1
  if (d[lo] > d[lo + size - 1]) dualSwap(d, ix, lo, 0, size - 1)
  if (d[lo + size - 1] > d[lo + p]) {
    dualSwap(d, ix, lo, size - 1, p)
    if (d[lo] > d[lo + size - 1]) dualSwap(d, ix, lo, 0, size - 1)
  }
  const pivot = d[lo + size - 1]
  let store = 0
  for (let i = 0; i < size - 1; i++) {
    if (d[lo + i] < pivot) {
      dualSwap(d, ix, lo, i, store)
      store++
    }
  }
  dualSwap(d, ix, lo, store, size - 1)
  p = store
  if (p > 1) simultaneousSort(d, ix, lo, p)
  if (p + 2 < size) simultaneousSort(d, ix, lo + p + 1, size - p - 1)
}

/** heap_push on a max-heap of the k closest so far */
const heapPush = (d: number[], ix: number[], size: number, val: number, idx: number) => {
  if (val >= d[0]) return
  let cur = 0
  for (;;) {
    const l = 2 * cur + 1
    const r = l + 1
    let swap: number
    if (l >= size) break
    else if (r >= size) {
      if (d[l] > val) swap = l
      else break
    } else if (d[l] >= d[r]) {
      if (val < d[l]) swap = l
      else break
    } else {
      if (val < d[r]) swap = r
      else break
    }
    d[cur] = d[swap]
    ix[cur] = ix[swap]
    cur = swap
  }
  d[cur] = val
  ix[cur] = idx
}

/** np.argpartition's introselect, which decides among tied distances when ArgKmin cannot be used */
const argIntroselect = (v: number[], ix: number[], kth: number) => {
  const swap = (a: number, b: number) => {
    const t = ix[a]
    ix[a] = ix[b]
    ix[b] = t
  }
  const num = ix.length
  let low = 0
  let high = num - 1
  if (kth < 3) {
    for (let i = 0; i <= kth; i++) {
      let m = i
      for (let k = i + 1; k < num; k++) if (v[ix[k]] < v[ix[m]]) m = k
      swap(i, m)
    }
    return
  }
  if (kth === num - 1) {
    let m = 0
    for (let k = 1; k < num; k++) if (!(v[ix[k]] < v[ix[m]])) m = k
    swap(kth, m)
    return
  }
  while (low + 1 < high) {
    let ll = low + 1
    let hh = high
    const mid = low + ((high - low) >> 1)
    if (v[ix[high]] < v[ix[mid]]) swap(high, mid)
    if (v[ix[high]] < v[ix[low]]) swap(high, low)
    if (v[ix[low]] < v[ix[mid]]) swap(low, mid)
    swap(mid, low + 1)
    const pivot = v[ix[low]]
    for (;;) {
      do ll++
      while (v[ix[ll]] < pivot)
      do hh--
      while (pivot < v[ix[hh]])
      if (hh < ll) break
      swap(ll, hh)
    }
    swap(low, hh)
    if (hh >= kth) high = hh - 1
    if (hh <= kth) low = ll
  }
  if (high === low + 1 && v[ix[high]] < v[ix[low]]) swap(high, low)
}

type MetricFn = (a: number[], b: number[]) => number

/** the distance the heap compares (squared for euclidean, p-th power for minkowski) and the step back to a true distance */
const metricOf = (h: Host, e: Est): { rdist: MetricFn; finish: (x: number) => number; effective: string; p: number | null } => {
  const metric = e.param('metric')
  if (metric.k !== 'str') throw h.err('NotImplementedError', `${e.cls}: only named metrics are available in this sandbox (euclidean, manhattan, minkowski, chebyshev)`)
  const p = h.num(e.param('p'))
  let name = metric.v
  if (name === 'minkowski') {
    if (p < 1) throw h.err('NotImplementedError', `${e.cls}: minkowski with p < 1 is not available in this sandbox`)
    name = p === 1 ? 'manhattan' : p === 2 ? 'euclidean' : p === Infinity ? 'chebyshev' : 'minkowski'
  } else if (name === 'l2') name = 'euclidean'
  else if (name === 'l1' || name === 'cityblock') name = 'manhattan'
  if (name === 'euclidean') return { rdist: (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return s }, finish: Math.sqrt, effective: 'euclidean', p: null }
  if (name === 'manhattan') return { rdist: (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s }, finish: (x) => x, effective: 'manhattan', p: null }
  if (name === 'chebyshev') return { rdist: (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s = Math.max(s, Math.abs(a[i] - b[i])); return s }, finish: (x) => x, effective: 'chebyshev', p: null }
  if (name === 'minkowski') return { rdist: (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]) ** p; return s }, finish: (x) => x ** (1 / p), effective: 'minkowski', p }
  throw h.err('NotImplementedError', `${e.cls}: metric=${reprStr(metric.v)} is not available in this sandbox (euclidean, manhattan, minkowski, chebyshev)`)
}

const kneighborsOf = (e: Est, query: { rows: number[][]; isFloat: boolean }, skipSelf: boolean, k0: number): { dist: number[][]; ind: number[][] } => {
  const q = query.rows
  const fit = e.st.X as number[][]
  const { rdist, finish, effective } = e.st.metric as ReturnType<typeof metricOf>
  const k = skipSelf ? k0 + 1 : k0
  // brute force reduces euclidean distances through X² + Y² - 2XY; a tree computes them directly
  const isBrute = e.st.brute as boolean
  const useGemm = isBrute && effective === 'euclidean'
  const onY = isBrute && !(fit.length < q.length) && !(4 * 256 < q.length)
  const sqFit = useGemm ? fit.map((r) => dot(r, r)) : []
  const dist: number[][] = []
  const ind: number[][] = []
  q.forEach((x, qi) => {
    if (isBrute && !(query.isFloat && (e.st.floatFit as boolean))) {
      // integer data cannot use ArgKmin: sklearn computes all distances, argpartitions, and sorts the first k
      const all = fit.map((y, i) => (skipSelf && i === qi ? 0 : effective === 'euclidean' ? Math.max(0, -2 * dot(x, y) + dot(x, x) + dot(y, y)) : finish(rdist(x, y))))
      const order = fit.map((_, i) => i)
      argIntroselect(all, order, k - 1)
      const top = order.slice(0, k).sort((a, b) => all[a] - all[b])
      if (skipSelf) {
        const at = top.indexOf(qi)
        top.splice(at >= 0 ? at : 0, 1)
      }
      dist.push(top.map((i) => (effective === 'euclidean' ? Math.sqrt(all[i]) : all[i])))
      ind.push(top)
      return
    }
    const d = new Array<number>(k).fill(Infinity)
    const ix = new Array<number>(k).fill(0)
    const xx = useGemm ? dot(x, x) : 0
    for (let i = 0; i < fit.length; i++) heapPush(d, ix, k, useGemm ? Math.max(0, xx + sqFit[i] - 2 * dot(x, fit[i])) : rdist(x, fit[i]), i)
    let dd = d
    let ii = ix
    if (onY) {
      // ArgKmin parallel on the training set: the thread's heap is pushed once more into the final one, which reorders ties
      dd = new Array<number>(k).fill(Infinity)
      ii = new Array<number>(k).fill(0)
      for (let j = 0; j < k; j++) heapPush(dd, ii, k, d[j], ix[j])
    }
    simultaneousSort(dd, ii, 0, k)
    if (skipSelf) {
      // each point drops itself, or its first neighbour when more duplicates than neighbours hide it
      const at = ii.indexOf(qi)
      const drop = at >= 0 ? at : 0
      dd.splice(drop, 1)
      ii.splice(drop, 1)
    }
    dist.push(dd.map(finish))
    ind.push(ii)
  })
  return { dist, ind }
}

/** _get_weights: None for uniform; for 'distance', 1/d with exact matches taking all the weight */
const neighbourWeights = (h: Host, e: Est, dist: number[][]): number[][] | null => {
  const w = e.param('weights')
  if (w.k === 'none' || (w.k === 'str' && w.v === 'uniform')) return null
  if (w.k === 'str' && w.v === 'distance') {
    return dist.map((row) => {
      const inv = row.map((d) => 1 / d)
      const anyInf = inv.some((x) => !Number.isFinite(x))
      return anyInf ? inv.map((x) => (Number.isFinite(x) ? 0 : 1)) : inv
    })
  }
  if (w.k === 'func' || w.k === 'fn') {
    const out = numericArray(h, h.call(w, [mat(dist)]))
    return rowsOf(out)
  }
  throw badParam(h, `The 'weights' parameter of ${e.cls} must be a str among {'distance', 'uniform'}, a callable or None. Got ${repr(w)} instead.`)
}

const queryOf = (e: Est, Xv: Value | undefined, h: Host): { rows: number[][]; isFloat: boolean } => {
  if (isNone(Xv)) return { rows: e.st.X as number[][], isFloat: e.st.floatFit as boolean }
  const X = readTest(e, Xv, h)
  return { rows: X.rows, isFloat: X.a.dtype === 'float64' || X.a.dtype === 'float32' }
}

const NEIGHBOR_PARAMS: [string, Value][] = [['n_neighbors', py.int(5)], ['weights', py.str('uniform')], ['algorithm', py.str('auto')], ['leaf_size', py.int(30)], ['p', py.int(2)], ['metric', py.str('minkowski')], ['metric_params', py.NONE], ['n_jobs', py.NONE]]

const fitNeighbors = (e: Est, h: Host, X: XData) => {
  const nn = e.param('n_neighbors')
  if (nn.k !== 'none') inRange(h, e, 'n_neighbors', nn.k === 'int' && nn.v >= 1, 'an int in the range [1, inf) or None')
  const algo = strOf(e.param('algorithm'))
  if (!algo || !['auto', 'ball_tree', 'kd_tree', 'brute'].includes(algo)) throw badParam(h, `The 'algorithm' parameter of ${e.cls} must be a str among {'auto', 'ball_tree', 'brute', 'kd_tree'}. Got ${repr(e.param('algorithm'))} instead.`)
  if (e.param('metric_params').k !== 'none') throw h.err('NotImplementedError', `${e.cls}: metric_params is not available in this sandbox`)
  const metric = metricOf(h, e)
  rememberFeatures(e, X)
  e.st.X = X.rows
  e.st.floatFit = X.a.dtype === 'float64' || X.a.dtype === 'float32'
  e.st.metric = metric
  e.st.brute = algo === 'brute' || (algo === 'auto' && (X.m > 15 || (nn.k === 'int' && nn.v >= Math.floor(X.n / 2))))
  e.fitted.set('n_samples_fit_', py.int(X.n))
  e.fitted.set('effective_metric_', py.str(metric.effective))
  e.fitted.set('effective_metric_params_', metric.p === null ? dictOf([]) : dictOf([['p', e.param('p')]]))
  e.fitted.set('_fit_method', py.str(algo === 'auto' ? (e.st.brute ? 'brute' : 'kd_tree') : algo))
  e.st.fitted = true
}

const kneighborsMethod = (e: Est, h: Host): Fn => (args, kw) => {
  rejectKw(h, `${e.cls}.kneighbors`, kw, ['X', 'n_neighbors', 'return_distance'])
  requireFit(e, h)
  const Xv = args[0] ?? kw.X
  const nv = args[1] ?? kw.n_neighbors ?? py.NONE
  const n = nv.k === 'none' ? e.int('n_neighbors', h) : h.toInt(nv)
  if (n <= 0) throw h.err('ValueError', `Expected n_neighbors > 0. Got ${n}`)
  const rd = args[2] ?? kw.return_distance
  const self = isNone(Xv)
  const q = queryOf(e, Xv, h)
  const fitN = (e.st.X as number[][]).length
  const queries = n + (self ? 1 : 0)
  if (queries > fitN) throw h.err('ValueError', `Expected n_neighbors <= n_samples_fit, but n_neighbors = ${queries}, n_samples_fit = ${fitN}, n_samples = ${q.rows.length}`)
  const { dist, ind } = kneighborsOf(e, q, self, n)
  const indV = mat(ind, n, 'int64')
  return rd && !h.truthy(rd) ? indV : py.tuple([mat(dist, n), indV])
}

const KNN_CLASSIFIER: Spec = {
  name: 'KNeighborsClassifier',
  kind: 'classifier',
  pos: ['n_neighbors'],
  params: NEIGHBOR_PARAMS,
  methods: (e, h) => {
    const votes = (args: Value[], kw: Kw) => {
      requireFit(e, h)
      const Xv = args[0] ?? kw.X
      const self = isNone(Xv)
      const q = queryOf(e, Xv, h)
      const k = e.int('n_neighbors', h)
      const queries = k + (self ? 1 : 0)
      if (queries > (e.st.X as number[][]).length) throw h.err('ValueError', `Expected n_neighbors <= n_samples_fit, but n_neighbors = ${queries}, n_samples_fit = ${(e.st.X as number[][]).length}, n_samples = ${q.rows.length}`)
      const { dist, ind } = kneighborsOf(e, q, self, k)
      const w = neighbourWeights(h, e, dist)
      if (w && w.some((r) => r.every((x) => x === 0))) throw h.err('ValueError', "All neighbors of some sample is getting zero weights. Please modify 'weights' to avoid this case if you are using a user-defined function.")
      const enc = e.st.enc as number[]
      const nc = (e.st.classes as Lab[]).length
      return ind.map((row, i) => {
        const c = new Array<number>(nc).fill(0)
        row.forEach((nb, j) => { c[enc[nb]] += w ? w[i][j] : 1 })
        return c
      })
    }
    const predict: Fn = (args, kw) => predictLabels(e, votes(args, kw).map(argmax))
    return {
      fit: (args, kw) => {
        rejectKw(h, 'KNeighborsClassifier.fit', kw, ['X', 'y'])
        const X = readX(h, args[0] ?? kw.X, e.cls)
        const yV = args[1] ?? kw.y
        if (yV) checkLengths(h, [py.arr(X.a), yV])
        const t = classifierTarget(h, yV)
        fitNeighbors(e, h, X)
        storeClasses(e, t)
        e.st.enc = t.enc
        e.fitted.set('outputs_2d_', py.bool(false))
        return py.obj(e)
      },
      predict,
      predict_proba: (args, kw) => {
        const v = votes(args, kw)
        const nc = (e.st.classes as Lab[]).length
        return mat(v.map((c) => {
          const s = npSum(c) || 1
          return c.map((x) => x / s)
        }), nc)
      },
      score: classifierScore(e, h, predict),
      kneighbors: kneighborsMethod(e, h),
    }
  },
}

const KNN_REGRESSOR: Spec = {
  name: 'KNeighborsRegressor',
  kind: 'regressor',
  pos: ['n_neighbors'],
  params: NEIGHBOR_PARAMS,
  methods: (e, h) => {
    const predictRows = (args: Value[], kw: Kw): number[][] => {
      requireFit(e, h)
      const Xv = args[0] ?? kw.X
      const self = isNone(Xv)
      const q = queryOf(e, Xv, h)
      const k = e.int('n_neighbors', h)
      const queries = k + (self ? 1 : 0)
      if (queries > (e.st.X as number[][]).length) throw h.err('ValueError', `Expected n_neighbors <= n_samples_fit, but n_neighbors = ${queries}, n_samples_fit = ${(e.st.X as number[][]).length}, n_samples = ${q.rows.length}`)
      const { dist, ind } = kneighborsOf(e, q, self, k)
      const w = neighbourWeights(h, e, dist)
      const Y = e.st.y as number[][]
      const t = e.st.t as number
      return ind.map((row, i) => Array.from({ length: t }, (_, j) => {
        const vals = row.map((nb) => Y[nb][j])
        if (!w) return npMean(vals)
        return npSum(vals.map((v, c) => v * w[i][c])) / npSum(w[i])
      }))
    }
    const predict: Fn = (args, kw) => {
      const r = predictRows(args, kw)
      return e.st.multi ? mat(r, e.st.t as number) : vec(r.map((x) => x[0]))
    }
    return {
      fit: (args, kw) => {
        rejectKw(h, 'KNeighborsRegressor.fit', kw, ['X', 'y'])
        const X = readX(h, args[0] ?? kw.X, e.cls)
        const yV = args[1] ?? kw.y
        if (yV) checkLengths(h, [py.arr(X.a), yV])
        const y = regressionTarget(h, yV)
        fitNeighbors(e, h, X)
        e.st.y = y.rows
        e.st.multi = y.multi
        e.st.t = y.t
        return py.obj(e)
      },
      predict,
      score: (args, kw) => {
        noSampleWeight(h, 'KNeighborsRegressor.score', kw, args[2])
        const yT = regressionTarget(h, args[1] ?? kw.y)
        const yP = predictRows([args[0] ?? kw.X], {})
        return py.float(npMean(Array.from({ length: yT.t }, (_, j) => r2Of(yT.rows.map((r) => r[j]), yP.map((r) => r[j])))))
      },
      kneighbors: kneighborsMethod(e, h),
    }
  },
}

// ---------------------------------------------------------------- k-means

/** _euclidean_dense_dense with squared=True: four terms at a time */
const sqEuclid = (a: number[], b: number[]): number => {
  const n = a.length
  const rem = n % 4
  let r = 0
  let i = 0
  for (; i < n - rem; i += 4) r += (a[i] - b[i]) ** 2 + (a[i + 1] - b[i + 1]) ** 2 + (a[i + 2] - b[i + 2]) ** 2 + (a[i + 3] - b[i + 3]) ** 2
  for (; i < n; i++) r += (a[i] - b[i]) ** 2
  return r
}

/** the E step: nearest centre by ||C||² - 2 X·C, as sklearn compares them */
const assign = (X: number[][], C: number[][]): number[] => {
  const cc = C.map((c) => dot(c, c))
  return X.map((x) => {
    let best = cc[0] - 2 * dot(x, C[0])
    let label = 0
    for (let j = 1; j < C.length; j++) {
      const d = cc[j] - 2 * dot(x, C[j])
      if (d < best) {
        best = d
        label = j
      }
    }
    return label
  })
}

/** one Lloyd iteration: new centres (empty clusters relocated), and how far each centre moved */
const lloydStep = (X: number[][], C: number[][], m: number) => {
  const k = C.length
  const labels = assign(X, C)
  const sum = Array.from({ length: k }, () => new Array<number>(m).fill(0))
  const weight = new Array<number>(k).fill(0)
  X.forEach((x, i) => {
    weight[labels[i]] += 1
    for (let j = 0; j < m; j++) sum[labels[i]][j] += x[j]
  })
  const empty = weight.flatMap((w, j) => (w === 0 ? [j] : []))
  if (empty.length) {
    const dist = X.map((x, i) => sqEuclid(x, C[labels[i]]))
    if (Math.max(...dist) !== 0) {
      const far = X.map((_, i) => i)
      argIntroselect(dist, far, X.length - empty.length)
      empty.forEach((nc, t) => {
        const idx = far[X.length - 1 - t]
        const old = labels[idx]
        for (let j = 0; j < m; j++) {
          sum[old][j] -= X[idx][j]
          sum[nc][j] = X[idx][j]
        }
        weight[nc] = 1
        weight[old] -= 1
      })
    }
  }
  const next = sum.map((s, j) => {
    if (weight[j] <= 0) return s
    const alpha = 1 / weight[j]
    return s.map((x) => x * alpha)
  })
  const shift = next.map((c, j) => Math.sqrt(sqEuclid(c, C[j])))
  return { labels, next, shift }
}

const lloyd = (X: number[][], init: number[][], m: number, maxIter: number, tol: number) => {
  let C = init
  let labels: number[] = new Array<number>(X.length).fill(-1)
  let old = labels
  let strict = false
  let it = 0
  for (it = 0; it < maxIter; it++) {
    const s = lloydStep(X, C, m)
    labels = s.labels
    C = s.next
    if (labels.every((l, i) => l === old[i])) {
      strict = true
      it++
      break
    }
    if (npSum(s.shift.map((x) => x * x)) <= tol) {
      it++
      break
    }
    old = labels
  }
  if (!strict) labels = assign(X, C)
  let inertia = 0
  X.forEach((x, i) => { inertia += sqEuclid(x, C[labels[i]]) })
  return { labels, inertia, C, iters: it }
}

/** euclidean_distances(A, B, squared=True), through -2 A·B + ||A||² + ||B||² */
const sqDistances = (A: number[][], B: number[][], bb: number[]): number[][] =>
  A.map((a) => {
    const aa = dot(a, a)
    return B.map((b, j) => Math.max(0, -2 * dot(a, b) + aa + bb[j]))
  })

const kmeansPlusPlus = (X: number[][], k: number, rng: RandomState, xx: number[]): number[][] => {
  const n = X.length
  const trials = 2 + Math.floor(Math.log(k))
  const centers: number[][] = []
  const first = rng.choiceP(new Array<number>(n).fill(1 / npSum(new Array<number>(n).fill(1))))
  centers.push(X[first])
  let closest = sqDistances([X[first]], X, xx)[0]
  let pot = 0
  for (const d of closest) pot += d
  for (let c = 1; c < k; c++) {
    const rand = Array.from({ length: trials }, () => rng.double() * pot)
    let cum = 0
    const cdf = closest.map((d) => (cum += d))
    const ids = rand.map((r) => Math.min(searchLeft(cdf, r), n - 1))
    const dists = sqDistances(ids.map((i) => X[i]), X, xx).map((row) => row.map((d, i) => Math.min(closest[i], d)))
    const pots = dists.map((row) => {
      let s = 0
      for (const d of row) s += d
      return s
    })
    let best = 0
    pots.forEach((p, i) => { if (p < pots[best]) best = i })
    pot = pots[best]
    closest = dists[best]
    centers.push(X[ids[best]])
  }
  return centers.map((c) => [...c])
}

const sameClustering = (a: number[], b: number[], k: number): boolean => {
  const map = new Array<number>(k).fill(-1)
  for (let i = 0; i < a.length; i++) {
    if (map[a[i]] === -1) map[a[i]] = b[i]
    else if (map[a[i]] !== b[i]) return false
  }
  return true
}

const KMEANS: Spec = {
  name: 'KMeans',
  pos: ['n_clusters'],
  params: [['n_clusters', py.int(8)], ['init', py.str('k-means++')], ['n_init', py.str('auto')], ['max_iter', py.int(300)], ['tol', py.float(0.0001)], ['verbose', py.int(0)], ['random_state', py.NONE], ['copy_x', py.bool(true)], ['algorithm', py.str('lloyd')]],
  locked: ['algorithm'],
  methods: (e, h) => {
    const fit = (args: Value[], kw: Kw): XData => {
      rejectKw(h, 'KMeans.fit', kw, ['X', 'y', 'sample_weight'])
      noSampleWeight(h, 'KMeans.fit', kw, args[2])
      e.checkLocked(h)
      const k = e.param('n_clusters')
      inRange(h, e, 'n_clusters', k.k === 'int' && k.v >= 1, 'an int in the range [1, inf)')
      const nClusters = h.toInt(k)
      const mi = e.param('max_iter')
      inRange(h, e, 'max_iter', mi.k === 'int' && mi.v >= 1, 'an int in the range [1, inf)')
      const tolV = e.param('tol')
      inRange(h, e, 'tol', isNum(tolV) && numOf(tolV) >= 0, 'a float in the range [0.0, inf)')
      const X = readX(h, args[0] ?? kw.X, 'KMeans')
      if (X.n < nClusters) throw h.err('ValueError', `n_samples=${X.n} should be >= n_clusters=${nClusters}.`)
      const tol0 = numOf(tolV)
      const tol = tol0 === 0 ? 0 : (npSum(varAxis0(X.rows, X.m)) / X.m) * tol0
      const init = e.param('init')
      const isArr = init.k === 'array' || init.k === 'list' || init.k === 'tuple'
      if (!isArr && init.k !== 'str') throw h.err('NotImplementedError', 'KMeans: a callable init is not available in this sandbox')
      if (init.k === 'str' && init.v !== 'k-means++' && init.v !== 'random') throw badParam(h, `The 'init' parameter of KMeans must be a str among {'k-means++', 'random'}, a callable or an array-like. Got ${repr(init)} instead.`)
      const nInitV = e.param('n_init')
      if (!(nInitV.k === 'str' && nInitV.v === 'auto') && !(nInitV.k === 'int' && nInitV.v >= 1)) throw badParam(h, `The 'n_init' parameter of KMeans must be a str among {'auto'} or an int in the range [1, inf). Got ${repr(nInitV)} instead.`)
      let nInit = nInitV.k === 'int' ? nInitV.v : init.k === 'str' && init.v === 'random' ? 10 : 1
      if (isArr) nInit = 1
      const mean = meanAxis0(X.rows, X.m)
      const centred = X.rows.map((r) => r.map((x, j) => x - mean[j]))
      let initArr: number[][] | null = null
      if (isArr) {
        const a = readX(h, init, 'KMeans', 'X')
        if (a.n !== nClusters) throw h.err('ValueError', `The shape of the initial centers (${a.n}, ${a.m}) does not match the number of clusters ${nClusters}.`)
        if (a.m !== X.m) throw h.err('ValueError', `The shape of the initial centers (${a.n}, ${a.m}) does not match the number of features of the data ${X.m}.`)
        initArr = a.rows.map((r) => r.map((x, j) => x - mean[j]))
      }
      const rng = rngOf(h, e.param('random_state'))
      const xx = centred.map((r) => dot(r, r))
      let best: ReturnType<typeof lloyd> | null = null
      for (let i = 0; i < nInit; i++) {
        let C0: number[][]
        if (initArr) C0 = initArr
        else if (init.k === 'str' && init.v === 'random') {
          const p = new Array<number>(X.n).fill(1 / X.n)
          C0 = rng.choiceNoReplace(nClusters, p).map((s) => centred[s])
        } else C0 = kmeansPlusPlus(centred, nClusters, rng, xx)
        const run = lloyd(centred, C0, X.m, h.toInt(mi), tol)
        if (!best || (run.inertia < best.inertia && !sameClustering(run.labels, best.labels, nClusters))) best = run
      }
      const centers = best!.C.map((c) => c.map((x, j) => x + mean[j]))
      rememberFeatures(e, X)
      e.st.centers = centers
      e.st.labels = best!.labels
      e.st.fitted = true
      e.fitted.set('cluster_centers_', mat(centers, X.m))
      e.fitted.set('labels_', vec(best!.labels, 'int64'))
      e.fitted.set('inertia_', py.float(best!.inertia))
      e.fitted.set('n_iter_', py.int(best!.iters))
      e.fitted.set('_n_features_out', py.int(nClusters))
      return X
    }
    const distances = (X: number[][]) => {
      const C = e.st.centers as number[][]
      const cc = C.map((c) => dot(c, c))
      return sqDistances(X, C, cc).map((r) => r.map(Math.sqrt))
    }
    return {
      fit: (args, kw) => { fit(args, kw); return py.obj(e) },
      fit_predict: (args, kw) => { fit(args, kw); return vec(e.st.labels as number[], 'int64') },
      fit_transform: (args, kw) => { const X = fit(args, kw); return mat(distances(X.rows), (e.st.centers as number[][]).length) },
      predict: (args, kw) => { noSampleWeight(h, 'KMeans.predict', kw); return vec(assign(readTest(e, args[0] ?? kw.X, h).rows, e.st.centers as number[][]), 'int64') },
      transform: (args, kw) => mat(distances(readTest(e, args[0] ?? kw.X, h).rows), (e.st.centers as number[][]).length),
      score: (args, kw) => {
        const X = readTest(e, args[0] ?? kw.X, h)
        const C = e.st.centers as number[][]
        const l = assign(X.rows, C)
        let s = 0
        X.rows.forEach((x, i) => { s += sqEuclid(x, C[l[i]]) })
        return py.float(-s)
      },
      get_feature_names_out: () => { requireFit(e, h); return py.obj(new ObjArray((e.st.centers as number[][]).map((_, i) => `kmeans${i}`), 'object')) },
    }
  },
}

// ---------------------------------------------------------------- naive Bayes

/** everything the Naive Bayes classifiers share: predict and friends from a joint log likelihood */
const nbMethods = (e: Est, h: Host, fit: Fn, jll: (X: XData) => number[][]): Record<string, Fn> => {
  const predictLog = (X: XData) => jll(X).map((r) => {
    const m = logsumexp(r)
    return r.map((x) => x - m)
  })
  const predict: Fn = (args, kw) => predictLabels(e, jll(readTest(e, args[0] ?? kw.X, h)).map(argmax))
  const nc = () => (e.st.classes as Lab[]).length
  return {
    fit,
    predict,
    predict_log_proba: (args, kw) => mat(predictLog(readTest(e, args[0] ?? kw.X, h)), nc()),
    predict_proba: (args, kw) => mat(predictLog(readTest(e, args[0] ?? kw.X, h)).map((r) => r.map(Math.exp)), nc()),
    predict_joint_log_proba: (args, kw) => mat(jll(readTest(e, args[0] ?? kw.X, h)), nc()),
    score: classifierScore(e, h, predict),
  }
}

const GAUSSIAN_NB: Spec = {
  name: 'GaussianNB',
  kind: 'classifier',
  params: [['priors', py.NONE], ['var_smoothing', py.float(1e-9)]],
  methods: (e, h) => {
    const fit: Fn = (args, kw) => {
      rejectKw(h, 'GaussianNB.fit', kw, ['X', 'y', 'sample_weight'])
      noSampleWeight(h, 'GaussianNB.fit', kw, args[2])
      const X = readX(h, args[0] ?? kw.X, 'GaussianNB')
      const yV = args[1] ?? kw.y
      if (yV) checkLengths(h, [py.arr(X.a), yV])
      const t = classifierTarget(h, yV)
      const eps = e.num('var_smoothing', h) * Math.max(...varAxis0(X.rows, X.m))
      const theta: number[][] = []
      const vars: number[][] = []
      const counts: number[] = []
      t.classes.forEach((_, c) => {
        const rows = X.rows.filter((_r, i) => t.enc[i] === c)
        theta.push(meanAxis0(rows, X.m))
        vars.push(varAxis0(rows, X.m).map((v) => v + eps))
        counts.push(rows.length)
      })
      let prior = counts.map((c) => c / X.n)
      const pv = e.param('priors')
      if (pv.k !== 'none') {
        const p = toArray(pv, 'float64').values()
        if (p.length !== t.classes.length) throw h.err('ValueError', 'Number of priors must match number of classes.')
        const total = p.reduce((a, b) => a + b, 0)
        if (Math.abs(total - 1) > 1e-8 + 1e-5) throw h.err('ValueError', 'The sum of the priors should be 1.')
        if (p.some((x) => x < 0)) throw h.err('ValueError', 'Priors must be non-negative.')
        prior = p
      }
      storeClasses(e, t)
      rememberFeatures(e, X)
      e.st.theta = theta
      e.st.vars = vars
      e.st.prior = prior
      e.st.fitted = true
      e.fitted.set('theta_', mat(theta, X.m))
      e.fitted.set('var_', mat(vars, X.m))
      e.fitted.set('class_count_', vec(counts))
      e.fitted.set('class_prior_', vec(prior))
      e.fitted.set('epsilon_', py.float(eps))
      return py.obj(e)
    }
    const jll = (X: XData) => {
      const theta = e.st.theta as number[][]
      const vars = e.st.vars as number[][]
      const prior = e.st.prior as number[]
      const base = vars.map((v) => -0.5 * npSum(v.map((x) => Math.log(2 * Math.PI * x))))
      return X.rows.map((r) => theta.map((t, c) => Math.log(prior[c]) + (base[c] - 0.5 * npSum(r.map((x, j) => (x - t[j]) ** 2 / vars[c][j]))))) 
    }
    return nbMethods(e, h, fit, jll)
  },
}

const classLogPrior = (h: Host, e: Est, counts: number[]): number[] => {
  const cp = e.param('class_prior')
  const n = counts.length
  if (cp.k !== 'none') {
    const p = toArray(cp, 'float64').values()
    if (p.length !== n) throw h.err('ValueError', 'Number of priors must match number of classes.')
    return p.map(Math.log)
  }
  if (e.flag('fit_prior', h)) {
    const total = Math.log(npSum(counts))
    return counts.map((c) => Math.log(c) - total)
  }
  return new Array<number>(n).fill(-Math.log(n))
}

const checkAlpha = (h: Host, e: Est) => {
  const a = e.param('alpha')
  if (!isNum(a)) throw h.err('NotImplementedError', `${e.cls}: an array alpha is not available in this sandbox`)
  inRange(h, e, 'alpha', numOf(a) >= 0, 'a float in the range [0.0, inf) or an array-like')
  const alpha = numOf(a)
  return alpha < 1e-10 && !e.flag('force_alpha', h) ? 1e-10 : alpha
}

const CATEGORICAL_NB: Spec = {
  name: 'CategoricalNB',
  kind: 'classifier',
  params: [['alpha', py.float(1)], ['force_alpha', py.bool(true)], ['fit_prior', py.bool(true)], ['class_prior', py.NONE], ['min_categories', py.NONE]],
  methods: (e, h) => {
    const readInts = (v: Value | undefined): XData => {
      const X = readX(h, v, 'CategoricalNB')
      const rows = X.rows.map((r) => r.map(Math.trunc))
      if (rows.some((r) => r.some((x) => x < 0))) throw h.err('ValueError', 'Negative values in data passed to CategoricalNB (input X).')
      return { ...X, rows }
    }
    const fit: Fn = (args, kw) => {
      rejectKw(h, 'CategoricalNB.fit', kw, ['X', 'y', 'sample_weight'])
      noSampleWeight(h, 'CategoricalNB.fit', kw, args[2])
      const X = readInts(args[0] ?? kw.X)
      const yV = args[1] ?? kw.y
      if (yV) checkLengths(h, [py.arr(X.a), yV])
      const t = classifierTarget(h, yV)
      const alpha = checkAlpha(h, e)
      const nc = t.classes.length
      const counts = new Array<number>(nc).fill(0)
      t.enc.forEach((c) => counts[c]++)
      const minC = e.param('min_categories')
      const mins = minC.k === 'none' ? new Array<number>(X.m).fill(0) : isNum(minC) ? new Array<number>(X.m).fill(numOf(minC)) : toArray(minC).values()
      if (mins.length !== X.m) throw h.err('ValueError', `'min_categories' should have shape (${X.m},) when an array-like is provided. Got ${toArray(minC).shape.join(', ')} instead.`)
      const cols = colsOf(X.rows, X.m)
      const nCat = cols.map((c, j) => Math.max(Math.max(...c) + 1, mins[j]))
      const catCount = cols.map((c, j) => {
        const table = Array.from({ length: nc }, () => new Array<number>(nCat[j]).fill(0))
        c.forEach((v, i) => { table[t.enc[i]][v]++ })
        return table
      })
      const flp = catCount.map((table) => table.map((row) => {
        const smoothed = row.map((x) => x + alpha)
        const total = Math.log(npSum(smoothed))
        return smoothed.map((x) => Math.log(x) - total)
      }))
      storeClasses(e, t)
      rememberFeatures(e, X)
      e.st.flp = flp
      e.st.prior = classLogPrior(h, e, counts)
      e.st.fitted = true
      e.fitted.set('class_count_', vec(counts))
      e.fitted.set('category_count_', py.list(catCount.map((table, j) => mat(table, nCat[j]))))
      e.fitted.set('feature_log_prob_', py.list(flp.map((table, j) => mat(table, nCat[j]))))
      e.fitted.set('class_log_prior_', vec(e.st.prior as number[]))
      e.fitted.set('n_categories_', vec(nCat, 'int64'))
      return py.obj(e)
    }
    const jll = (X0: XData) => {
      const flp = e.st.flp as number[][][]
      const prior = e.st.prior as number[]
      const rows = X0.rows.map((r) => r.map(Math.trunc))
      if (rows.some((r) => r.some((x) => x < 0))) throw h.err('ValueError', 'Negative values in data passed to CategoricalNB (input X).')
      return rows.map((r) => prior.map((p, c) => {
        let s = 0
        r.forEach((v, j) => {
          if (v >= flp[j][c].length) throw h.err('IndexError', `index ${v} is out of bounds for axis 1 with size ${flp[j][c].length}`)
          s += flp[j][c][v]
        })
        return s + p
      }))
    }
    return nbMethods(e, h, fit, jll)
  },
}

/** MultinomialNB and BernoulliNB count features per class the same way */
const countingNB = (name: 'MultinomialNB' | 'BernoulliNB'): Spec => ({
  name,
  kind: 'classifier',
  params: name === 'MultinomialNB'
    ? [['alpha', py.float(1)], ['force_alpha', py.bool(true)], ['fit_prior', py.bool(true)], ['class_prior', py.NONE]]
    : [['alpha', py.float(1)], ['force_alpha', py.bool(true)], ['binarize', py.float(0)], ['fit_prior', py.bool(true)], ['class_prior', py.NONE]],
  methods: (e, h) => {
    const prep = (X: XData): XData => {
      if (name === 'MultinomialNB') {
        if (X.rows.some((r) => r.some((x) => x < 0))) throw h.err('ValueError', 'Negative values in data passed to MultinomialNB (input X).')
        return X
      }
      const b = e.param('binarize')
      if (b.k === 'none') return X
      const t = h.num(b)
      return { ...X, rows: X.rows.map((r) => r.map((x) => (x > t ? 1 : 0))) }
    }
    const fit: Fn = (args, kw) => {
      rejectKw(h, `${name}.fit`, kw, ['X', 'y', 'sample_weight'])
      noSampleWeight(h, `${name}.fit`, kw, args[2])
      const raw = readX(h, args[0] ?? kw.X, name)
      const yV = args[1] ?? kw.y
      if (yV) checkLengths(h, [py.arr(raw.a), yV])
      const t = classifierTarget(h, yV)
      const X = prep(raw)
      const alpha = checkAlpha(h, e)
      const nc = t.classes.length
      const counts = new Array<number>(nc).fill(0)
      const fc = Array.from({ length: nc }, () => new Array<number>(X.m).fill(0))
      X.rows.forEach((r, i) => {
        counts[t.enc[i]]++
        r.forEach((x, j) => { fc[t.enc[i]][j] += x })
      })
      const flp = fc.map((row, c) => {
        const smoothed = row.map((x) => x + alpha)
        const denom = Math.log(name === 'MultinomialNB' ? npSum(smoothed) : counts[c] + alpha * 2)
        return smoothed.map((x) => Math.log(x) - denom)
      })
      storeClasses(e, t)
      rememberFeatures(e, X)
      e.st.flp = flp
      e.st.prior = classLogPrior(h, e, counts)
      e.st.fitted = true
      e.fitted.set('class_count_', vec(counts))
      e.fitted.set('feature_count_', mat(fc, X.m))
      e.fitted.set('feature_log_prob_', mat(flp, X.m))
      e.fitted.set('class_log_prior_', vec(e.st.prior as number[]))
      return py.obj(e)
    }
    const jll = (X0: XData) => {
      const X = prep(X0)
      const flp = e.st.flp as number[][]
      const prior = e.st.prior as number[]
      if (name === 'MultinomialNB') return X.rows.map((r) => flp.map((f, c) => dot(r, f) + prior[c]))
      const neg = flp.map((f) => f.map((x) => Math.log(1 - Math.exp(x))))
      const add = neg.map((n, c) => prior[c] + npSum(n))
      return X.rows.map((r) => flp.map((f, c) => dot(r, f.map((x, j) => x - neg[c][j])) + add[c]))
    }
    return nbMethods(e, h, fit, jll)
  },
})

// ---------------------------------------------------------------- linear models

const classWeights = (h: Host, cw: Value, classes: Lab[], enc: number[], est: string): number[] => {
  if (cw.k === 'none') return classes.map(() => 1)
  if (cw.k === 'str') {
    if (cw.v !== 'balanced') throw badParam(h, `The 'class_weight' parameter of ${est} must be an instance of 'dict', a str among {'balanced'} or None. Got ${repr(cw)} instead.`)
    const counts = classes.map(() => 0)
    enc.forEach((c) => counts[c]++)
    return counts.map((c) => enc.length / (classes.length * c))
  }
  if (cw.k !== 'dict') throw badParam(h, `The 'class_weight' parameter of ${est} must be an instance of 'dict', a str among {'balanced'} or None. Got ${repr(cw)} instead.`)
  const given = new Map(cw.d.items().map(([k, v]) => [k.k === 'str' ? k.v : h.num(k), h.num(v)] as [Lab, number]))
  const missing = classes.filter((c) => !given.has(c))
  if (missing.length && classes.length - missing.length !== given.size) throw h.err('ValueError', `The classes, [${missing.map((c) => (typeof c === 'string' ? reprStr(c) : c)).join(', ')}], are not in class_weight`)
  return classes.map((c) => given.get(c) ?? 1)
}

/** _plain_sgd for the perceptron loss: the exact sequential update */
const plainSgd = (X: number[][], y: number[], m: number, o: { eta: number; alpha: number; l2: boolean; fitIntercept: boolean; maxIter: number; tol: number; shuffle: boolean; seed: number; noChange: number; wPos: number; wNeg: number }) => {
  const w = new Array<number>(m).fill(0)
  let wscale = 1
  let b = 0
  const n = X.length
  const order = Array.from({ length: n }, (_, i) => i)
  let best = Infinity
  let stall = 0
  let epoch = 0
  for (epoch = 0; epoch < o.maxIter; epoch++) {
    let sumloss = 0
    if (o.shuffle) {
      const st = { s: o.seed >>> 0 }
      for (let i = 0; i < n - 1; i++) {
        const j = i + (ourRandR(st) % (n - i))
        const t = order[i]
        order[i] = order[j]
        order[j] = t
      }
    }
    for (const idx of order) {
      const x = X[idx]
      const yi = y[idx]
      let p = 0
      for (let j = 0; j < m; j++) p += w[j] * x[j]
      p = p * wscale + b
      const z = p * yi
      if (z <= 0) sumloss += -z
      const update = (z <= 0 ? o.eta * yi : 0) * (yi > 0 ? o.wPos : o.wNeg)
      if (o.l2) {
        const c = Math.max(0, 1 - o.eta * o.alpha)
        wscale *= c
        if (wscale < 1e-9) {
          for (let j = 0; j < m; j++) w[j] *= wscale
          wscale = 1
        }
      }
      if (update !== 0) {
        const c = update / wscale
        for (let j = 0; j < m; j++) w[j] += x[j] * c
      }
      if (o.fitIntercept) b += update
    }
    if (o.tol > -Infinity && sumloss > best - o.tol * n) stall++
    else stall = 0
    if (sumloss < best) best = sumloss
    if (stall >= o.noChange) {
      epoch++
      break
    }
  }
  return { w: w.map((x) => x * wscale), b, iters: epoch }
}

const decisionMethods = (e: Est, h: Host): { scores: (v: Value | undefined) => number[][]; predict: Fn; decision: Fn } => {
  const scores = (v: Value | undefined) => {
    const X = readTest(e, v, h)
    const coef = e.st.coef as number[][]
    const b = e.st.intercept as number[]
    return X.rows.map((r) => coef.map((c, k) => dot(r, c) + b[k]))
  }
  return {
    scores,
    predict: (args, kw) => {
      const s = scores(args[0] ?? kw.X)
      return predictLabels(e, s.map((r) => (r.length === 1 ? (r[0] > 0 ? 1 : 0) : argmax(r))))
    },
    decision: (args, kw) => {
      const s = scores(args[0] ?? kw.X)
      return s[0].length === 1 ? vec(s.map((r) => r[0])) : mat(s, s[0].length)
    },
  }
}

const PERCEPTRON: Spec = {
  name: 'Perceptron',
  kind: 'classifier',
  params: [['penalty', py.NONE], ['alpha', py.float(0.0001)], ['l1_ratio', py.float(0.15)], ['fit_intercept', py.bool(true)], ['max_iter', py.int(1000)], ['tol', py.float(0.001)], ['shuffle', py.bool(true)], ['verbose', py.int(0)], ['eta0', py.float(1)], ['n_jobs', py.NONE], ['random_state', py.int(0)], ['early_stopping', py.bool(false)], ['validation_fraction', py.float(0.1)], ['n_iter_no_change', py.int(5)], ['class_weight', py.NONE], ['warm_start', py.bool(false)]],
  locked: ['early_stopping', 'warm_start'],
  methods: (e, h) => {
    const { predict, decision } = decisionMethods(e, h)
    return {
      fit: (args, kw) => {
        rejectKw(h, 'Perceptron.fit', kw, ['X', 'y', 'coef_init', 'intercept_init', 'sample_weight'])
        noSampleWeight(h, 'Perceptron.fit', kw)
        if (!isNone(kw.coef_init) || !isNone(kw.intercept_init)) throw h.err('NotImplementedError', 'Perceptron.fit: coef_init and intercept_init are not available in this sandbox')
        e.checkLocked(h)
        const pen = e.param('penalty')
        if (pen.k !== 'none' && !(pen.k === 'str' && pen.v === 'l2')) throw h.err('NotImplementedError', `Perceptron: penalty=${repr(pen)} is not available in this sandbox (None and 'l2' are)`)
        const X = readX(h, args[0] ?? kw.X, 'Perceptron')
        const yV = args[1] ?? kw.y
        if (yV) checkLengths(h, [py.arr(X.a), yV])
        const t = classifierTarget(h, yV)
        if (t.classes.length < 2) throw h.err('ValueError', `The number of classes has to be greater than one; got ${t.classes.length} class`)
        const eta = e.num('eta0', h)
        inRange(h, e, 'eta0', eta > 0, 'a float in the range (0.0, inf)')
        const weights = classWeights(h, e.param('class_weight'), t.classes, t.enc, 'Perceptron')
        const rng = rngOf(h, e.param('random_state'))
        const tolV = e.param('tol')
        const base = {
          eta,
          alpha: e.num('alpha', h),
          l2: pen.k === 'str',
          fitIntercept: e.flag('fit_intercept', h),
          maxIter: e.int('max_iter', h),
          tol: tolV.k === 'none' ? -Infinity : h.num(tolV),
          shuffle: e.flag('shuffle', h),
          noChange: e.int('n_iter_no_change', h),
        }
        // fit_binary draws once for the dataset and once for the shuffle
        const runOne = (cls: number, r2: RandomState, wPos: number, wNeg: number) => {
          r2.randint(1, MAX_INT)
          const sgdSeed = r2.randint(0, MAX_INT)
          return plainSgd(X.rows, t.enc.map((c) => (c === cls ? 1 : -1)), X.m, { ...base, seed: sgdSeed, wPos, wNeg })
        }
        let coef: number[][]
        let intercept: number[]
        let iters: number
        if (t.classes.length === 2) {
          const r = runOne(1, rng, weights[1], weights[0])
          coef = [r.w]
          intercept = [r.b]
          iters = r.iters
        } else {
          const seeds = t.classes.map(() => rng.randint(0, MAX_INT))
          const runs = t.classes.map((_, k) => runOne(k, new RandomState(seeds[k]), weights[k], 1))
          coef = runs.map((r) => r.w)
          intercept = runs.map((r) => r.b)
          iters = Math.max(...runs.map((r) => r.iters))
        }
        storeClasses(e, t)
        rememberFeatures(e, X)
        e.st.coef = coef
        e.st.intercept = intercept
        e.st.fitted = true
        e.fitted.set('coef_', mat(coef, X.m))
        e.fitted.set('intercept_', vec(intercept))
        e.fitted.set('n_iter_', py.int(iters))
        e.fitted.set('t_', py.float(1 + iters * X.n))
        return py.obj(e)
      },
      predict,
      decision_function: decision,
      score: classifierScore(e, h, predict),
    }
  },
}

/** the thin SVD, with U and V as lists of vectors */
const svd = (A: number[][]): { U: number[][]; S: number[]; V: number[][] } => {
  const { u, s, v } = svdThin(A)
  return { U: transposeM(u), S: s, V: transposeM(v) }
}

/** centre X and y as _preprocess_data does; returns the means to rebuild the intercept */
const centreXY = (X: XData, Y: number[][], t: number, on: boolean) => {
  if (!on) return { Xc: X.rows, Yc: Y, xm: new Array<number>(X.m).fill(0), ym: new Array<number>(t).fill(0) }
  const xm = meanAxis0(X.rows, X.m)
  const ym = meanAxis0(Y, t)
  return { Xc: X.rows.map((r) => r.map((x, j) => x - xm[j])), Yc: Y.map((r) => r.map((x, j) => x - ym[j])), xm, ym }
}

const regressorResult = (e: Est, X: XData, y: { multi: boolean; t: number }, coef: number[][], xm: number[], ym: number[], on: boolean) => {
  const intercept = coef.map((c, k) => (on ? ym[k] - dot(xm, c) : 0))
  rememberFeatures(e, X)
  e.st.coef = coef
  e.st.intercept = intercept
  e.st.multi = y.multi
  e.st.t = y.t
  e.st.fitted = true
  e.fitted.set('coef_', y.multi ? mat(coef, X.m) : vec(coef[0]))
  e.fitted.set('intercept_', y.multi ? vec(intercept) : py.float(intercept[0]))
}

const regressorMethods = (e: Est, h: Host): Record<string, Fn> => {
  const predictRows = (v: Value | undefined) => {
    const X = readTest(e, v, h)
    const coef = e.st.coef as number[][]
    const b = e.st.intercept as number[]
    return X.rows.map((r) => coef.map((c, k) => dot(r, c) + b[k]))
  }
  return {
    predict: (args, kw) => {
      const r = predictRows(args[0] ?? kw.X)
      return e.st.multi ? mat(r, e.st.t as number) : vec(r.map((x) => x[0]))
    },
    score: (args, kw) => {
      noSampleWeight(h, `${e.cls}.score`, kw, args[2])
      const yT = regressionTarget(h, args[1] ?? kw.y)
      const yP = predictRows(args[0] ?? kw.X)
      return py.float(npMean(Array.from({ length: yT.t }, (_, j) => r2Of(yT.rows.map((r) => r[j]), yP.map((r) => r[j])))))
    },
  }
}

const LINEAR_REGRESSION: Spec = {
  name: 'LinearRegression',
  kind: 'regressor',
  params: [['fit_intercept', py.bool(true)], ['copy_X', py.bool(true)], ['tol', py.float(1e-6)], ['n_jobs', py.NONE], ['positive', py.bool(false)]],
  locked: ['positive'],
  methods: (e, h) => ({
    fit: (args, kw) => {
      rejectKw(h, 'LinearRegression.fit', kw, ['X', 'y', 'sample_weight'])
      noSampleWeight(h, 'LinearRegression.fit', kw, args[2])
      e.checkLocked(h)
      const X = readX(h, args[0] ?? kw.X, 'LinearRegression')
      const yV = args[1] ?? kw.y
      if (yV) checkLengths(h, [py.arr(X.a), yV])
      const y = regressionTarget(h, yV)
      const on = e.flag('fit_intercept', h)
      const { Xc, Yc, xm, ym } = centreXY(X, y.rows, y.t, on)
      const { U, S, V } = svd(Xc)
      const cutoff = Math.max(X.n, X.m) * Number.EPSILON * (S[0] ?? 0)
      const rank = S.filter((s) => s > cutoff).length
      const coef = Array.from({ length: y.t }, (_, t) => {
        const col = Yc.map((r) => r[t])
        const c = new Array<number>(X.m).fill(0)
        for (let k = 0; k < rank; k++) {
          const w = dot(U[k], col) / S[k]
          for (let j = 0; j < X.m; j++) c[j] += w * V[k][j]
        }
        return c
      })
      regressorResult(e, X, y, coef, xm, ym, on)
      e.fitted.set('rank_', py.int(rank))
      e.fitted.set('singular_', vec(S))
      return py.obj(e)
    },
    ...regressorMethods(e, h),
  }),
}

const solveSym = (h: Host, A: number[][], b: number[]): number[] => {
  const x = solveM(A, b.map((v) => [v]))
  if (!x) throw h.err('LinAlgError', 'Singular matrix')
  return x.map((r) => r[0])
}

const RIDGE: Spec = {
  name: 'Ridge',
  kind: 'regressor',
  pos: ['alpha'],
  params: [['alpha', py.float(1)], ['fit_intercept', py.bool(true)], ['copy_X', py.bool(true)], ['max_iter', py.NONE], ['tol', py.float(0.0001)], ['solver', py.str('auto')], ['positive', py.bool(false)], ['random_state', py.NONE]],
  locked: ['positive'],
  methods: (e, h) => ({
    fit: (args, kw) => {
      rejectKw(h, 'Ridge.fit', kw, ['X', 'y', 'sample_weight'])
      noSampleWeight(h, 'Ridge.fit', kw, args[2])
      e.checkLocked(h)
      const solver = strOf(e.param('solver'))
      if (solver !== 'auto' && solver !== 'cholesky' && solver !== 'svd') throw h.err('NotImplementedError', `Ridge: solver=${repr(e.param('solver'))} is not available in this sandbox (auto, cholesky and svd give the exact solution)`)
      const av = e.param('alpha')
      if (!isNum(av)) throw h.err('NotImplementedError', 'Ridge: one alpha per target is not available in this sandbox')
      const alpha = numOf(av)
      inRange(h, e, 'alpha', alpha >= 0, 'a float in the range [0.0, inf) or an array-like')
      const X = readX(h, args[0] ?? kw.X, 'Ridge')
      const yV = args[1] ?? kw.y
      if (yV) checkLengths(h, [py.arr(X.a), yV])
      const y = regressionTarget(h, yV)
      const on = e.flag('fit_intercept', h)
      const { Xc, Yc, xm, ym } = centreXY(X, y.rows, y.t, on)
      const coef = Array.from({ length: y.t }, (_, t) => {
        const col = Yc.map((r) => r[t])
        if (X.m > X.n) {
          const K = Xc.map((a, i) => Xc.map((b, j) => dot(a, b) + (i === j ? alpha : 0)))
          const dual = solveSym(h, K, col)
          return Array.from({ length: X.m }, (_x, j) => { let s = 0; Xc.forEach((r, i) => { s += r[j] * dual[i] }); return s })
        }
        const cols = colsOf(Xc, X.m)
        const A = cols.map((a, i) => cols.map((b, j) => dot(a, b) + (i === j ? alpha : 0)))
        return solveSym(h, A, cols.map((c) => dot(c, col)))
      })
      regressorResult(e, X, y, coef, xm, ym, on)
      e.fitted.set('n_iter_', py.NONE)
      e.fitted.set('solver_', py.str('cholesky'))
      return py.obj(e)
    },
    ...regressorMethods(e, h),
  }),
}

/** minimise f with L-BFGS and a backtracking line search; returns x and whether the gradient vanished */
const lbfgs = (f: (x: number[]) => { v: number; g: number[] }, x0: number[], gtol: number, maxIter: number): { x: number[]; ok: boolean } => {
  const n = x0.length
  let x = x0
  let cur = f(x)
  const S: number[][] = []
  const Y: number[][] = []
  const R: number[] = []
  let flat = 0
  for (let it = 0; it < maxIter; it++) {
    if (Math.max(...cur.g.map(Math.abs)) < gtol) return { x, ok: true }
    const q = [...cur.g]
    const a: number[] = []
    for (let i = S.length - 1; i >= 0; i--) {
      a[i] = R[i] * dot(S[i], q)
      for (let j = 0; j < n; j++) q[j] -= a[i] * Y[i][j]
    }
    const last = S.length - 1
    const gam = last >= 0 ? dot(S[last], Y[last]) / dot(Y[last], Y[last]) : 1 / Math.max(1, Math.sqrt(dot(cur.g, cur.g)))
    const d = q.map((v) => v * gam)
    for (let i = 0; i < S.length; i++) {
      const b = R[i] * dot(Y[i], d)
      for (let j = 0; j < n; j++) d[j] += S[i][j] * (a[i] - b)
    }
    const dir = d.map((v) => -v)
    const slope = dot(cur.g, dir)
    let step = 1
    let next = null as { v: number; g: number[] } | null
    let xn = x
    for (let ls = 0; ls < 60; ls++) {
      xn = x.map((v, j) => v + step * dir[j])
      next = f(xn)
      if (next.v <= cur.v + 1e-4 * step * slope) break
      step *= 0.5
      next = null
    }
    if (!next) return { x, ok: Math.max(...cur.g.map(Math.abs)) < gtol * 1e4 }
    const s = xn.map((v, j) => v - x[j])
    const y = next.g.map((v, j) => v - cur.g[j])
    const sy = dot(s, y)
    if (sy > 1e-14) {
      S.push(s)
      Y.push(y)
      R.push(1 / sy)
      if (S.length > 20) {
        S.shift()
        Y.shift()
        R.shift()
      }
    }
    // the loss has stopped moving at machine precision, so more steps only burn the page's time
    flat = cur.v - next.v <= 1e-15 * Math.abs(cur.v) ? flat + 1 : 0
    x = xn
    cur = next
    if (flat >= 5 && Math.max(...cur.g.map(Math.abs)) < gtol * 1e4) return { x, ok: true }
  }
  return { x, ok: Math.max(...cur.g.map(Math.abs)) < gtol * 1e4 }
}

const LOGISTIC_REGRESSION: Spec = {
  name: 'LogisticRegression',
  kind: 'classifier',
  pos: ['penalty'],
  params: [['penalty', py.str('l2')], ['dual', py.bool(false)], ['tol', py.float(0.0001)], ['C', py.float(1)], ['fit_intercept', py.bool(true)], ['intercept_scaling', py.int(1)], ['class_weight', py.NONE], ['random_state', py.NONE], ['solver', py.str('lbfgs')], ['max_iter', py.int(100)], ['multi_class', py.str('deprecated')], ['verbose', py.int(0)], ['warm_start', py.bool(false)], ['n_jobs', py.NONE], ['l1_ratio', py.NONE]],
  locked: ['dual', 'warm_start', 'l1_ratio', 'multi_class'],
  methods: (e, h) => {
    const { scores, predict, decision } = decisionMethods(e, h)
    const proba = (v: Value | undefined): number[][] => scores(v).map((r) => {
      if (r.length === 1) {
        const p = 1 / (1 + Math.exp(-r[0]))
        return [1 - p, p]
      }
      const m = Math.max(...r)
      const ex = r.map((x) => Math.exp(x - m))
      const s = npSum(ex)
      return ex.map((x) => x / s)
    })
    return {
      fit: (args, kw) => {
        rejectKw(h, 'LogisticRegression.fit', kw, ['X', 'y', 'sample_weight'])
        noSampleWeight(h, 'LogisticRegression.fit', kw, args[2])
        e.checkLocked(h)
        const pen = e.param('penalty')
        if (pen.k !== 'none' && !(pen.k === 'str' && pen.v === 'l2')) throw h.err('NotImplementedError', `LogisticRegression: penalty=${repr(pen)} is not available in this sandbox (None and 'l2' are)`)
        const solver = strOf(e.param('solver'))
        if (solver !== 'lbfgs' && solver !== 'newton-cg' && solver !== 'newton-cholesky') throw h.err('NotImplementedError', `LogisticRegression: solver=${repr(e.param('solver'))} is not available in this sandbox (lbfgs, newton-cg and newton-cholesky reach the same optimum)`)
        const maxIter = e.int('max_iter', h)
        if (maxIter < 100) throw h.err('NotImplementedError', `LogisticRegression: max_iter=${maxIter} stops the optimiser early, at a point this sandbox cannot reproduce: it always fits to convergence`)
        if (e.num('tol', h) > 1e-3) throw h.err('NotImplementedError', `LogisticRegression: tol=${repr(e.param('tol'))} stops the optimiser early, at a point this sandbox cannot reproduce: it always fits to convergence`)
        const C = e.num('C', h)
        inRange(h, e, 'C', C > 0, 'a float in the range (0.0, inf)')
        const X = readX(h, args[0] ?? kw.X, 'LogisticRegression')
        const yV = args[1] ?? kw.y
        if (yV) checkLengths(h, [py.arr(X.a), yV])
        const t = classifierTarget(h, yV)
        if (t.classes.length < 2) throw h.err('ValueError', `This solver needs samples of at least 2 classes in the data, but the data contains only one class: ${repr(py.str(String(t.classes[0])))}`)
        const on = e.flag('fit_intercept', h)
        const reg = pen.k === 'none' ? 0 : 1
        const cw = classWeights(h, e.param('class_weight'), t.classes, t.enc, 'LogisticRegression')
        const sw = t.enc.map((c) => C * cw[c])
        const { rows } = X
        const m = X.m
        const k = t.classes.length
        let coef: number[][]
        let intercept: number[]
        if (k === 2) {
          const f = (w: number[]) => {
            let v = 0
            const g = new Array<number>(m + 1).fill(0)
            rows.forEach((r, i) => {
              const yi = t.enc[i] === 1 ? 1 : -1
              const z = yi * (dot(r, w) + (on ? w[m] : 0))
              v += sw[i] * Math.log1p(Math.exp(-z) === Infinity ? Number.MAX_VALUE : Math.exp(-z))
              const q = -yi * sw[i] / (1 + Math.exp(z))
              for (let j = 0; j < m; j++) g[j] += q * r[j]
              if (on) g[m] += q
            })
            for (let j = 0; j < m; j++) {
              v += (0.5 * reg * w[j] * w[j])
              g[j] += reg * w[j]
            }
            return { v, g }
          }
          const r = lbfgs(f, new Array<number>(m + 1).fill(0), 1e-9 * Math.max(1, C), 5000)
          if (!r.ok) throw h.err('NotImplementedError', 'LogisticRegression: this fit has no finite optimum (separable data without a penalty?), so the answer depends on where the real solver stops')
          coef = [r.x.slice(0, m)]
          intercept = [on ? r.x[m] : 0]
        } else {
          const size = k * (m + 1)
          const f = (w: number[]) => {
            let v = 0
            const g = new Array<number>(size).fill(0)
            const eta = new Array<number>(k)
            rows.forEach((r, i) => {
              let mx = -Infinity
              for (let c = 0; c < k; c++) {
                let s = on ? w[c * (m + 1) + m] : 0
                for (let j = 0; j < m; j++) s += w[c * (m + 1) + j] * r[j]
                eta[c] = s
                if (s > mx) mx = s
              }
              let tot = 0
              for (let c = 0; c < k; c++) tot += Math.exp(eta[c] - mx)
              v += sw[i] * (Math.log(tot) + mx - eta[t.enc[i]])
              for (let c = 0; c < k; c++) {
                const q = sw[i] * (Math.exp(eta[c] - mx) / tot - (c === t.enc[i] ? 1 : 0))
                for (let j = 0; j < m; j++) g[c * (m + 1) + j] += q * r[j]
                if (on) g[c * (m + 1) + m] += q
              }
            })
            for (let c = 0; c < k; c++) {
              for (let j = 0; j < m; j++) {
                const x = w[c * (m + 1) + j]
                v += 0.5 * reg * x * x
                g[c * (m + 1) + j] += reg * x
              }
            }
            return { v, g }
          }
          const r = lbfgs(f, new Array<number>(size).fill(0), 1e-9 * Math.max(1, C), 5000)
          if (!r.ok) throw h.err('NotImplementedError', 'LogisticRegression: this fit has no finite optimum (separable data without a penalty?), so the answer depends on where the real solver stops')
          coef = t.classes.map((_, c) => r.x.slice(c * (m + 1), c * (m + 1) + m))
          intercept = t.classes.map((_, c) => (on ? r.x[c * (m + 1) + m] : 0))
        }
        storeClasses(e, t)
        rememberFeatures(e, X)
        e.st.coef = coef
        e.st.intercept = intercept
        e.st.fitted = true
        e.fitted.set('coef_', mat(coef, m))
        e.fitted.set('intercept_', vec(intercept))
        return py.obj(e)
      },
      predict,
      decision_function: decision,
      predict_proba: (args, kw) => mat(proba(args[0] ?? kw.X), (e.st.classes as Lab[]).length),
      predict_log_proba: (args, kw) => mat(proba(args[0] ?? kw.X).map((r) => r.map(Math.log)), (e.st.classes as Lab[]).length),
      score: classifierScore(e, h, predict),
    }
  },
}

// ---------------------------------------------------------------- decision tree

interface TreeNode { feature: number; threshold: number; left: number; right: number; impurity: number; n: number; counts: number[] }

const nodeImpurity = (counts: number[], n: number, entropy: boolean): number => {
  if (entropy) {
    let s = 0
    for (const c of counts) {
      if (c > 0) {
        const p = c / n
        s -= p * (Math.log(p) / Math.LN2)
      }
    }
    return s
  }
  let sq = 0
  for (const c of counts) sq += c * c
  return 1 - sq / (n * n)
}

/** DepthFirstTreeBuilder with BestSplitter: features are drawn in sklearn's random order, which decides between equally good splits */
const buildTree = (rows64: number[][], enc: number[], nc: number, o: { entropy: boolean; maxDepth: number; minSplit: number; minLeaf: number; minDecrease: number; rand: { s: number } }): TreeNode[] => {
  // trees store X as float32, which shows in the thresholds
  const rows = rows64.map((r) => r.map(Math.fround))
  const m = rows[0].length
  const nodes: TreeNode[] = []
  const samples = rows.map((_, i) => i)
  const total = rows.length
  const features = Array.from({ length: m }, (_, j) => j)
  const constants = new Array<number>(m).fill(0)
  const EPS = Number.EPSILON
  const closeTo = (a: number, b: number) => a <= Math.fround(b + Math.fround(1e-7))
  const stack: { start: number; end: number; depth: number; parent: number; left: boolean; impurity: number; nConst: number }[] = [{ start: 0, end: total, depth: 0, parent: -1, left: false, impurity: Infinity, nConst: 0 }]
  let first = true
  while (stack.length) {
    const rec = stack.pop()!
    const { start, end, depth, parent, left } = rec
    const n = end - start
    const counts = new Array<number>(nc).fill(0)
    for (let i = start; i < end; i++) counts[enc[samples[i]]]++
    let impurity = rec.impurity
    let isLeaf = depth >= o.maxDepth || n < o.minSplit || n < 2 * o.minLeaf
    if (first) {
      impurity = nodeImpurity(counts, n, o.entropy)
      first = false
    }
    isLeaf = isLeaf || impurity <= EPS
    let bestPos = end
    let bestFeat = 0
    let bestThr = 0
    let bestImp = -Infinity
    let impL = Infinity
    let impR = Infinity
    let improvement = -Infinity
    let nConstAfter = rec.nConst
    if (!isLeaf) {
      let fi = m
      let visited = 0
      let foundC = 0
      let drawnC = 0
      const knownC = rec.nConst
      let totalC = knownC
      while (fi > totalC && (visited < m || visited <= foundC + drawnC)) {
        visited++
        const fj = drawnC + ourRandR(o.rand) % (fi - foundC - drawnC)
        if (fj < knownC) {
          const t = features[drawnC]
          features[drawnC] = features[fj]
          features[fj] = t
          drawnC++
          continue
        }
        const fjj = fj + foundC
        const feat = features[fjj]
        const vals: [number, number][] = []
        for (let i = start; i < end; i++) vals.push([rows[samples[i]][feat], samples[i]])
        vals.sort((a, b) => a[0] - b[0])
        if (closeTo(vals[vals.length - 1][0], vals[0][0])) {
          features[fjj] = features[totalC]
          features[totalC] = feat
          foundC++
          totalC++
          continue
        }
        fi--
        features[fjj] = features[fi]
        features[fi] = feat
        const lc = new Array<number>(nc).fill(0)
        const rc = [...counts]
        for (let p = 1; p < n; p++) {
          const c = enc[vals[p - 1][1]]
          lc[c]++
          rc[c]--
          if (closeTo(vals[p][0], vals[p - 1][0])) continue
          const nl = p
          const nr = n - p
          if (nl < o.minLeaf || nr < o.minLeaf) continue
          const il = nodeImpurity(lc, nl, o.entropy)
          const ir = nodeImpurity(rc, nr, o.entropy)
          const proxy = -nr * ir - nl * il
          if (proxy > bestImp) {
            bestImp = proxy
            let thr = vals[p - 1][0] / 2 + vals[p][0] / 2
            if (thr === vals[p][0] || !Number.isFinite(thr)) thr = vals[p - 1][0]
            bestThr = thr
            bestFeat = feat
            bestPos = start + p
            impL = il
            impR = ir
            improvement = 0
          }
        }
      }
      if (bestPos < end) {
        let l = start
        let r = end
        while (l < r) {
          if (rows[samples[l]][bestFeat] <= bestThr) l++
          else {
            r--
            const t = samples[l]
            samples[l] = samples[r]
            samples[r] = t
          }
        }
        const nl = bestPos - start
        const nr = end - bestPos
        improvement = (n / total) * (impurity - (nr / n) * impR - (nl / n) * impL)
      }
      for (let j = 0; j < knownC; j++) features[j] = constants[j]
      for (let j = knownC; j < totalC; j++) constants[j] = features[j]
      nConstAfter = totalC
      isLeaf = bestPos >= end || improvement + EPS < o.minDecrease
    }
    const id = nodes.length
    nodes.push({ feature: isLeaf ? -2 : bestFeat, threshold: isLeaf ? -2 : bestThr, left: -1, right: -1, impurity, n, counts })
    if (parent >= 0) {
      if (left) nodes[parent].left = id
      else nodes[parent].right = id
    }
    if (!isLeaf) {
      stack.push({ start: bestPos, end, depth: depth + 1, parent: id, left: false, impurity: impR, nConst: nConstAfter })
      stack.push({ start, end: bestPos, depth: depth + 1, parent: id, left: true, impurity: impL, nConst: nConstAfter })
    }
  }
  return nodes
}

const treeLeaf = (nodes: TreeNode[], r: number[]): TreeNode => {
  let n = nodes[0]
  while (n.left >= 0) n = nodes[Math.fround(r[n.feature]) <= n.threshold ? n.left : n.right]
  return n
}

class TreeView extends PyObj {
  readonly cls = 'Tree'
  nodes: TreeNode[]
  constructor(nodes: TreeNode[]) {
    super()
    this.nodes = nodes
  }
  repr() { return '<sklearn.tree._tree.Tree object>' }
  getAttr(name: string): Value | undefined {
    const ns = this.nodes
    switch (name) {
      case 'node_count': return py.int(ns.length)
      case 'n_leaves': return py.int(ns.filter((n) => n.left < 0).length)
      case 'max_depth': return py.int(treeDepth(ns))
      case 'feature': return vec(ns.map((n) => n.feature), 'int64')
      case 'threshold': return vec(ns.map((n) => n.threshold))
      case 'children_left': return vec(ns.map((n) => n.left), 'int64')
      case 'children_right': return vec(ns.map((n) => n.right), 'int64')
      case 'impurity': return vec(ns.map((n) => n.impurity))
      case 'n_node_samples': return vec(ns.map((n) => n.n), 'int64')
      case 'value': return py.arr(NDArray.create(ns.flatMap((n) => n.counts.map((c) => c / n.n)), [ns.length, 1, ns[0].counts.length], 'float64'))
    }
    return undefined
  }
}

const treeDepth = (ns: TreeNode[]): number => {
  const rec = (i: number): number => (ns[i].left < 0 ? 0 : 1 + Math.max(rec(ns[i].left), rec(ns[i].right)))
  return rec(0)
}

const DECISION_TREE: Spec = {
  name: 'DecisionTreeClassifier',
  kind: 'classifier',
  params: [['criterion', py.str('gini')], ['splitter', py.str('best')], ['max_depth', py.NONE], ['min_samples_split', py.int(2)], ['min_samples_leaf', py.int(1)], ['min_weight_fraction_leaf', py.float(0)], ['max_features', py.NONE], ['random_state', py.NONE], ['max_leaf_nodes', py.NONE], ['min_impurity_decrease', py.float(0)], ['class_weight', py.NONE], ['ccp_alpha', py.float(0)], ['monotonic_cst', py.NONE]],
  locked: ['splitter', 'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'class_weight', 'ccp_alpha', 'monotonic_cst'],
  methods: (e, h) => {
    const leafCounts = (v: Value | undefined) => {
      const X = readTest(e, v, h)
      return X.rows.map((r) => treeLeaf(e.st.nodes as TreeNode[], r).counts)
    }
    const predict: Fn = (args, kw) => predictLabels(e, leafCounts(args[0] ?? kw.X).map(argmax))
    return {
      fit: (args, kw) => {
        rejectKw(h, 'DecisionTreeClassifier.fit', kw, ['X', 'y', 'sample_weight', 'check_input'])
        noSampleWeight(h, 'DecisionTreeClassifier.fit', kw, args[2])
        e.checkLocked(h)
        const crit = strOf(e.param('criterion'))
        if (crit !== 'gini' && crit !== 'entropy' && crit !== 'log_loss') throw badParam(h, `The 'criterion' parameter of DecisionTreeClassifier must be a str among {'entropy', 'gini', 'log_loss'}. Got ${repr(e.param('criterion'))} instead.`)
        const md = e.param('max_depth')
        if (md.k !== 'none') inRange(h, e, 'max_depth', md.k === 'int' && md.v >= 1, 'an int in the range [1, inf) or None')
        const mss = e.param('min_samples_split')
        const msl = e.param('min_samples_leaf')
        const X = readX(h, args[0] ?? kw.X, 'DecisionTreeClassifier')
        const yV = args[1] ?? kw.y
        if (yV) checkLengths(h, [py.arr(X.a), yV])
        const t = classifierTarget(h, yV)
        const minLeaf = msl.k === 'float' ? Math.ceil(msl.v * X.n) : h.toInt(msl)
        const minSplit0 = mss.k === 'float' ? Math.ceil(mss.v * X.n) : h.toInt(mss)
        const nodes = buildTree(X.rows, t.enc, t.classes.length, {
          entropy: crit !== 'gini',
          maxDepth: md.k === 'none' ? Infinity : h.toInt(md),
          minSplit: Math.max(minSplit0, 2 * minLeaf),
          minLeaf,
          minDecrease: e.num('min_impurity_decrease', h),
          rand: { s: rngOf(h, e.param('random_state')).randint(0, MAX_INT) },
        })
        storeClasses(e, t)
        rememberFeatures(e, X)
        e.st.nodes = nodes
        e.st.fitted = true
        e.fitted.set('n_classes_', py.int(t.classes.length))
        e.fitted.set('n_outputs_', py.int(1))
        e.fitted.set('max_features_', py.int(X.m))
        e.fitted.set('tree_', py.obj(new TreeView(nodes)))
        const imp = new Array<number>(X.m).fill(0)
        for (const n of nodes) {
          if (n.left >= 0) {
            const l = nodes[n.left]
            const r = nodes[n.right]
            imp[n.feature] += n.n * n.impurity - l.n * l.impurity - r.n * r.impurity
          }
        }
        const norm = imp.map((x) => x / nodes[0].n)
        const tot = norm.reduce((a, b) => a + b, 0)
        e.fitted.set('feature_importances_', vec(tot > 0 ? norm.map((x) => x / tot) : norm))
        return py.obj(e)
      },
      predict,
      predict_proba: (args, kw) => mat(leafCounts(args[0] ?? kw.X).map((c) => {
        const s = c.reduce((a, b) => a + b, 0)
        return c.map((x) => x / s)
      }), (e.st.classes as Lab[]).length),
      predict_log_proba: (args, kw) => mat(leafCounts(args[0] ?? kw.X).map((c) => {
        const s = c.reduce((a, b) => a + b, 0)
        return c.map((x) => Math.log(x / s))
      }), (e.st.classes as Lab[]).length),
      apply: (args, kw) => {
        const X = readTest(e, args[0] ?? kw.X, h)
        const ns = e.st.nodes as TreeNode[]
        return vec(X.rows.map((r) => {
          let i = 0
          while (ns[i].left >= 0) i = Math.fround(r[ns[i].feature]) <= ns[i].threshold ? ns[i].left : ns[i].right
          return i
        }), 'int64')
      },
      score: classifierScore(e, h, predict),
      get_depth: () => { requireFit(e, h); return py.int(treeDepth(e.st.nodes as TreeNode[])) },
      get_n_leaves: () => { requireFit(e, h); return py.int((e.st.nodes as TreeNode[]).filter((n) => n.left < 0).length) },
    }
  },
}

/** sklearn.tree.export_text */
const exportText = (h: Host, v: Value, kw: Kw): string => {
  const e = estOf(v)
  if (!e || e.cls !== 'DecisionTreeClassifier') throw h.err('TypeError', 'export_text takes a fitted DecisionTreeClassifier here')
  requireFit(e, h)
  for (const k of Object.keys(kw)) if (k !== 'feature_names' && k !== 'decimals' && k !== 'max_depth') throw h.err('NotImplementedError', `export_text: ${k} is not available in this sandbox`)
  const ns = e.st.nodes as TreeNode[]
  const names = kw.feature_names ? labelsOf(h, kw.feature_names).vals.map(String) : Array.from({ length: e.st.m as number }, (_, i) => `feature_${i}`)
  const dec = kw.decimals ? h.toInt(kw.decimals) : 2
  const maxD = kw.max_depth ? h.toInt(kw.max_depth) : 10
  const cl = e.st.classes as Lab[]
  const lines: string[] = []
  const rec = (i: number, depth: number) => {
    const pad = '|   '.repeat(depth)
    const n = ns[i]
    if (depth > maxD) return
    if (n.left < 0) {
      lines.push(`${pad}|--- class: ${cl[argmax(n.counts)]}`)
      return
    }
    if (depth === maxD) {
      lines.push(`${pad}|--- truncated branch of depth ${subDepth(ns, i)}`)
      return
    }
    lines.push(`${pad}|--- ${names[n.feature]} <= ${fixedStr(n.threshold, dec)}`)
    rec(n.left, depth + 1)
    lines.push(`${pad}|--- ${names[n.feature]} >  ${fixedStr(n.threshold, dec)}`)
    rec(n.right, depth + 1)
  }
  rec(0, 0)
  return `${lines.join('\n')}\n`
}
const subDepth = (ns: TreeNode[], i: number): number => (ns[i].left < 0 ? 1 : 1 + Math.max(subDepth(ns, ns[i].left), subDepth(ns, ns[i].right)))

// ---------------------------------------------------------------- PCA

const PCA: Spec = {
  name: 'PCA',
  pos: ['n_components'],
  params: [['n_components', py.NONE], ['copy', py.bool(true)], ['whiten', py.bool(false)], ['svd_solver', py.str('auto')], ['tol', py.float(0)], ['iterated_power', py.str('auto')], ['n_oversamples', py.int(10)], ['power_iteration_normalizer', py.str('auto')], ['random_state', py.NONE]],
  methods: (e, h) => {
    const fit = (v: Value | undefined): XData => {
      const solver = strOf(e.param('svd_solver'))
      if (solver !== 'auto' && solver !== 'full' && solver !== 'covariance_eigh') throw h.err('NotImplementedError', `PCA: svd_solver=${repr(e.param('svd_solver'))} is not available in this sandbox (auto, full and covariance_eigh are)`)
      const X = readX(h, v, 'PCA')
      const lim = Math.min(X.n, X.m)
      const nc = e.param('n_components')
      if (nc.k === 'str') throw h.err('NotImplementedError', `PCA: n_components=${repr(nc)} is not available in this sandbox`)
      const solverName = solver === 'auto' ? (X.m <= 1000 && X.n >= 10 * X.m ? 'covariance_eigh' : 'full') : solver
      if (nc.k === 'int' && (nc.v < 0 || nc.v > lim)) throw h.err('ValueError', `n_components=${nc.v} must be between 0 and min(n_samples, n_features)=${lim} with svd_solver=${reprStr(solverName)}`)
      if (nc.k === 'float' && !(nc.v > 0 && nc.v < 1)) throw badParam(h, `The 'n_components' parameter of PCA must be an int in the range [0, inf), a float in the range (0.0, 1.0), a str among {'mle'} or None. Got ${repr(nc)} instead.`)
      const mean = meanAxis0(X.rows, X.m)
      const centred = X.rows.map((r) => r.map((x, j) => x - mean[j]))
      const { S, V } = svd(centred)
      // svd_flip(u_based_decision=False): the largest entry of each component is positive
      const comps = V.map((c) => {
        const big = argmax(c.map(Math.abs))
        const sign = c[big] < 0 ? -1 : 1
        return c.map((x) => x * sign)
      })
      const ev = S.map((s) => (s * s) / (X.n - 1))
      const totalVar = npSum(ev)
      const ratio = ev.map((x) => x / totalVar)
      let k: number
      if (nc.k === 'none') k = lim
      else if (nc.k === 'float') {
        let c = 0
        const cum = ratio.map((x) => (c += x))
        k = searchRight(cum, nc.v) + 1
      } else k = h.toInt(nc)
      const noise = k < lim ? npMean(ev.slice(k)) : 0
      rememberFeatures(e, X)
      e.st.mean = mean
      e.st.comps = comps.slice(0, k)
      e.st.ev = ev.slice(0, k)
      e.st.fitted = true
      e.fitted.set('components_', mat(comps.slice(0, k), X.m))
      e.fitted.set('explained_variance_', vec(ev.slice(0, k)))
      e.fitted.set('explained_variance_ratio_', vec(ratio.slice(0, k)))
      e.fitted.set('singular_values_', vec(S.slice(0, k)))
      e.fitted.set('mean_', vec(mean))
      e.fitted.set('n_components_', py.int(k))
      e.fitted.set('n_samples_', py.int(X.n))
      e.fitted.set('noise_variance_', py.float(noise))
      return X
    }
    const project = (rows: number[][]) => {
      const mean = e.st.mean as number[]
      const comps = e.st.comps as number[][]
      const ev = e.st.ev as number[]
      const whiten = e.flag('whiten', h)
      return rows.map((r) => {
        const c = r.map((x, j) => x - mean[j])
        return comps.map((p, k) => (whiten ? dot(c, p) / Math.sqrt(ev[k]) : dot(c, p)))
      })
    }
    return {
      fit: (args, kw) => { fit(args[0] ?? kw.X); return py.obj(e) },
      fit_transform: (args, kw) => {
        const X = fit(args[0] ?? kw.X)
        return mat(project(X.rows), (e.st.comps as number[][]).length)
      },
      transform: (args, kw) => mat(project(readTest(e, args[0] ?? kw.X, h).rows), (e.st.comps as number[][]).length),
      inverse_transform: (args, kw) => {
        requireFit(e, h)
        const Z = readX(h, args[0] ?? kw.X, 'PCA')
        const mean = e.st.mean as number[]
        const comps = e.st.comps as number[][]
        const ev = e.st.ev as number[]
        const whiten = e.flag('whiten', h)
        return mat(Z.rows.map((z) => mean.map((m, j) => {
          let s = 0
          z.forEach((v, k) => { s += v * comps[k][j] * (whiten ? Math.sqrt(ev[k]) : 1) })
          return s + m
        })), mean.length)
      },
      get_covariance: () => {
        requireFit(e, h)
        const comps = e.st.comps as number[][]
        const ev = e.st.ev as number[]
        const noise = h.num(e.fitted.get('noise_variance_') ?? py.float(0))
        const m = e.st.m as number
        return mat(Array.from({ length: m }, (_, i) => Array.from({ length: m }, (_x, j) => {
          let s = i === j ? noise : 0
          comps.forEach((c, k) => { s += c[i] * c[j] * (ev[k] - noise) })
          return s
        })), m)
      },
      get_feature_names_out: () => { requireFit(e, h); return py.obj(new ObjArray((e.st.comps as number[][]).map((_, i) => `pca${i}`), 'object')) },
    }
  },
}

// ---------------------------------------------------------------- metrics

const targetPair = (h: Host, tV: Value | undefined, pV: Value | undefined): { t: Labels; p: Labels; type: string } => {
  if (!tV || !pV) throw h.err('TypeError', "missing a required argument: 'y_pred'")
  checkLengths(h, [tV, pV])
  const t = labelsOf(h, tV)
  const p = labelsOf(h, pV)
  const a = typeOfTarget(t)
  const b = typeOfTarget(p)
  const kinds = new Set([a, b])
  if (kinds.has('binary') && kinds.has('multiclass')) kinds.delete('binary')
  if (kinds.size > 1) throw h.err('ValueError', `Classification metrics can't handle a mix of ${a} and ${b} targets`)
  const type = [...kinds][0]
  if (type !== 'binary' && type !== 'multiclass') throw h.err('ValueError', `${type} is not supported`)
  if (t.isStr !== p.isStr) {
    if ([t, p].some((l) => l.dtype === 'object')) throw h.err('TypeError', `Labels in y_true and y_pred should be of the same type. Got y_true=${labelArrayStr(t)} and y_pred=${labelArrayStr(p)}. Make sure that the predictions provided by the classifier coincides with the true labels.`)
    // NumPy turns a list of numbers into text next to a text array
    const asText = (l: Labels): Labels => ({ vals: l.vals.map(String), isStr: true, dtype: l.dtype })
    return { t: asText(t), p: asText(p), type: 'multiclass' }
  }
  return { t, p, type }
}
const labelArrayStr = (l: Labels) => (l.isStr ? new ObjArray(uniqueSorted(l.vals, true) as string[], l.dtype).str() : `[${uniqueSorted(l.vals, false).join(' ')}]`)

const labelsArg = (h: Host, v: Value | undefined): Lab[] | null => (isNone(v) ? null : labelsOf(h, v!).vals)

const noWeights = (h: Host, who: string, kw: Kw) => {
  if (!isNone(kw.sample_weight)) throw h.err('NotImplementedError', `${who}: sample_weight is not available in this sandbox`)
}

const confusionCounts = (t: Labels, p: Labels, labels: Lab[]): number[][] => {
  const at = new Map(labels.map((l, i) => [l, i]))
  const out = labels.map(() => new Array<number>(labels.length).fill(0))
  t.vals.forEach((v, i) => {
    const a = at.get(v)
    const b = at.get(p.vals[i])
    if (a !== undefined && b !== undefined) out[a][b]++
  })
  return out
}

/** tp, predicted and true counts per label */
const perLabel = (t: Labels, p: Labels, labels: Lab[]) => {
  const at = new Map(labels.map((l, i) => [l, i]))
  const tp = labels.map(() => 0)
  const pred = labels.map(() => 0)
  const tru = labels.map(() => 0)
  t.vals.forEach((v, i) => {
    const a = at.get(v)
    const b = at.get(p.vals[i])
    if (a !== undefined) tru[a]++
    if (b !== undefined) pred[b]++
    if (a !== undefined && v === p.vals[i]) tp[a]++
  })
  return { tp, pred, tru }
}

const zeroDivOf = (h: Host, v: Value | undefined): number => {
  if (isNone(v) || (v!.k === 'str' && v!.v === 'warn')) return 0
  return h.num(v!)
}

const prfCore = (h: Host, tV: Value | undefined, pV: Value | undefined, kw: Kw, who: string, beta = 1): { p: number | number[]; r: number | number[]; f: number | number[]; s: number | number[] | null; labels: Lab[] } => {
  noWeights(h, who, kw)
  const { t, p, type } = targetPair(h, tV, pV)
  const avg = kw.average === undefined ? 'binary' : kw.average.k === 'none' ? null : strOf(kw.average)
  if (avg !== null && !['binary', 'micro', 'macro', 'weighted'].includes(avg ?? '')) {
    if (avg === 'samples') throw h.err('ValueError', 'Samplewise metrics are not available outside of multilabel classification.')
    throw badParam(h, `The 'average' parameter of ${who} must be a str among {'binary', 'macro', 'micro', 'samples', 'weighted'} or None. Got ${repr(kw.average!)} instead.`)
  }
  const present = uniqueSorted([...t.vals, ...p.vals], t.isStr)
  const posLabel: Lab = kw.pos_label ? (kw.pos_label.k === 'str' ? kw.pos_label.v : h.num(kw.pos_label)) : 1
  let labels = labelsArg(h, kw.labels)
  if (avg === 'binary') {
    if (type === 'binary') {
      if (!present.includes(posLabel) && present.length >= 2) throw h.err('ValueError', `pos_label=${typeof posLabel === 'string' ? reprStr(posLabel) : posLabel} is not a valid label. It should be one of ${t.isStr ? `[${present.map((x) => reprStr(String(x))).join(', ')}]` : `[${present.join(', ')}]`}`)
      labels = [posLabel]
    } else {
      throw h.err('ValueError', `Target is ${type} but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].`)
    }
  }
  const use = labels ?? present
  const c = perLabel(t, p, use)
  const divide = (a: number[], b: number[], z: number) => a.map((x, i) => (b[i] === 0 ? z : x / b[i]))
  const z = zeroDivOf(h, kw.zero_division)
  const b2 = beta * beta
  const fOf = (tp: number[], pr: number[], tr: number[]) => tp.map((x, i) => {
    const denom = b2 * tr[i] + pr[i]
    return denom === 0 ? z : ((1 + b2) * x) / denom
  })
  if (avg === 'micro') {
    const tp = [npSum(c.tp)]
    const pr = [npSum(c.pred)]
    const tr = [npSum(c.tru)]
    return { p: divide(tp, pr, z)[0], r: divide(tp, tr, z)[0], f: fOf(tp, pr, tr)[0], s: null, labels: use }
  }
  const pr = divide(c.tp, c.pred, z)
  const re = divide(c.tp, c.tru, z)
  const ff = fOf(c.tp, c.pred, c.tru)
  if (avg === null) return { p: pr, r: re, f: ff, s: c.tru, labels: use }
  if (avg === 'binary') return { p: pr[0], r: re[0], f: ff[0], s: null, labels: use }
  const w = avg === 'weighted' ? c.tru : c.tru.map(() => 1)
  const sw = npSum(w)
  const mean = (xs: number[]) => (sw === 0 ? z : npSum(xs.map((x, i) => x * w[i])) / sw)
  return { p: mean(pr), r: mean(re), f: mean(ff), s: null, labels: use }
}

const prfScore = (h: Host, tV: Value, pV: Value, which: 'precision' | 'recall' | 'f1', kw: Kw): number => {
  const r = prfCore(h, tV, pV, kw, `${which}_score`)
  const v = which === 'precision' ? r.p : which === 'recall' ? r.r : r.f
  return v as number
}

const classificationReport = (h: Host, tV: Value | undefined, pV: Value | undefined, kw: Kw): Value => {
  noWeights(h, 'classification_report', kw)
  const { t, p } = targetPair(h, tV, pV)
  const labelsGiven = !isNone(kw.labels)
  const labels = labelsGiven ? labelsOf(h, kw.labels!).vals : uniqueSorted([...t.vals, ...p.vals], t.isStr)
  const present = uniqueSorted([...t.vals, ...p.vals], t.isStr)
  const microIsAccuracy = !labelsGiven || (labels.length >= present.length && present.every((x) => labels.includes(x)))
  const names0 = isNone(kw.target_names) ? labels.map(String) : labelsOf(h, kw.target_names!).vals.map(String)
  if (names0.length !== labels.length) throw h.err('ValueError', `Number of classes, ${labels.length}, does not match size of target_names, ${names0.length}. Try specifying the labels parameter`)
  const digits = kw.digits ? h.toInt(kw.digits) : 2
  const c = perLabel(t, p, labels)
  const z = zeroDivOf(h, kw.zero_division)
  const div = (a: number, b: number) => (b === 0 ? z : a / b)
  const rows = labels.map((_, i) => {
    const pr = div(c.tp[i], c.pred[i])
    const re = div(c.tp[i], c.tru[i])
    const denom = c.tru[i] + c.pred[i]
    return { p: pr, r: re, f: denom === 0 ? z : (2 * c.tp[i]) / denom, s: c.tru[i] }
  })
  // with no correct prediction at all, multilabel_confusion_matrix falls into its float branch and support prints as 3.0
  const noneRight = countMatches(t, p) === 0
  const count = (x: number) => (noneRight ? pyFloatRepr(x) : String(x))
  const totalS = npSum(c.tru)
  const macro = (k: 'p' | 'r' | 'f') => npSum(rows.map((r) => r[k])) / rows.length
  const weighted = (k: 'p' | 'r' | 'f') => (totalS === 0 ? z : npSum(rows.map((r) => r[k] * r.s)) / totalS)
  const micro = () => {
    const tp = npSum(c.tp)
    const pr = npSum(c.pred)
    const tr = npSum(c.tru)
    const f = tr + pr === 0 ? z : (2 * tp) / (tr + pr)
    return { p: div(tp, pr), r: div(tp, tr), f, s: tr }
  }
  const avgs: [string, { p: number; r: number; f: number; s: number }][] = []
  const m = micro()
  avgs.push([microIsAccuracy ? 'accuracy' : 'micro avg', m])
  avgs.push(['macro avg', { p: macro('p'), r: macro('r'), f: macro('f'), s: totalS }])
  avgs.push(['weighted avg', { p: weighted('p'), r: weighted('r'), f: weighted('f'), s: totalS }])
  if (kw.output_dict && h.truthy(kw.output_dict)) {
    const entry = (r: { p: number; r: number; f: number; s: number }) => dictOf([['precision', py.float(r.p)], ['recall', py.float(r.r)], ['f1-score', py.float(r.f)], ['support', py.float(r.s)]])
    const items: [string, Value][] = names0.map((n, i) => [n, entry(rows[i])])
    for (const [n, r] of avgs) items.push(n === 'accuracy' ? ['accuracy', py.float(r.p)] : [n, entry(r)])
    return dictOf(items)
  }
  const width = Math.max(...names0.map((n) => n.length), 'weighted avg'.length, digits)
  const num = (x: number) => fixedStr(x, digits).padStart(9)
  const lines = [`${''.padStart(width)}  ${'precision'.padStart(9)} ${'recall'.padStart(9)} ${'f1-score'.padStart(9)} ${'support'.padStart(9)}`, '']
  names0.forEach((n, i) => { lines.push(`${n.padStart(width)}  ${num(rows[i].p)} ${num(rows[i].r)} ${num(rows[i].f)} ${count(rows[i].s).padStart(9)}`) })
  lines.push('')
  for (const [n, r] of avgs) {
    if (n === 'accuracy') lines.push(`${n.padStart(width)}  ${''.padStart(9)} ${''.padStart(9)} ${num(r.f)} ${count(r.s).padStart(9)}`)
    else lines.push(`${n.padStart(width)}  ${num(r.p)} ${num(r.r)} ${num(r.f)} ${count(r.s).padStart(9)}`)
  }
  return py.str(`${lines.join('\n')}\n`)
}

/** _binary_clf_curve and roc_curve: false and true positive rates at each distinct score */
const rocCurve = (h: Host, tV: Value | undefined, sV: Value | undefined, posLabel?: Value): { fpr: number[]; tpr: number[]; thr: number[] } => {
  if (!tV || !sV) throw h.err('TypeError', "missing a required argument: 'y_score'")
  checkLengths(h, [tV, sV])
  const t = labelsOf(h, tV)
  const scores = numbersOf(h, sV)
  finiteCheck(h, scores, 'y_score', 'roc_curve')
  const classes = uniqueSorted(t.vals, t.isStr)
  if (classes.length > 2) throw h.err('ValueError', 'multiclass format is not supported')
  const pos: Lab = posLabel ? (posLabel.k === 'str' ? posLabel.v : h.num(posLabel)) : 1
  if (isNone(posLabel) && (t.isStr || !classes.every((c) => c === 0 || c === 1 || c === -1))) throw h.err('ValueError', `y_true takes value in ${t.isStr ? `{${classes.map((c) => reprStr(String(c))).join(', ')}}` : `{${classes.join(', ')}}`} and pos_label is not specified: either make y_true take value in {0, 1} or {-1, 1} or pass pos_label explicitly.`)
  const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a] || a - b)
  let tp = 0
  let fp = 0
  const tps: number[] = []
  const fps: number[] = []
  const thr: number[] = []
  order.forEach((i, k) => {
    if (t.vals[i] === pos) tp++
    else fp++
    if (k === order.length - 1 || scores[order[k + 1]] !== scores[i]) {
      tps.push(tp)
      fps.push(fp)
      thr.push(scores[i])
    }
  })
  // drop_intermediate: points with no second difference in either rate go
  if (tps.length > 2) {
    const keep = tps.flatMap((_, i) => {
      if (i === 0 || i === tps.length - 1) return [i]
      return fps[i - 1] - 2 * fps[i] + fps[i + 1] !== 0 || tps[i - 1] - 2 * tps[i] + tps[i + 1] !== 0 ? [i] : []
    })
    const pick = <T,>(a: T[]) => keep.map((i) => a[i])
    const a = pick(tps)
    const b = pick(fps)
    const c = pick(thr)
    tps.length = 0; tps.push(...a)
    fps.length = 0; fps.push(...b)
    thr.length = 0; thr.push(...c)
  }
  const T = tps[tps.length - 1]
  const F = fps[fps.length - 1]
  return { fpr: [0, ...fps].map((x) => (F > 0 ? x / F : NaN)), tpr: [0, ...tps].map((x) => (T > 0 ? x / T : NaN)), thr: [Infinity, ...thr] }
}

const trapezoid = (x: number[], y: number[]): number => npSum(x.slice(1).map((v, i) => ((v - x[i]) * (y[i] + y[i + 1])) / 2))

const regressionPair = (h: Host, tV: Value | undefined, pV: Value | undefined, kw: Kw, who: string) => {
  noWeights(h, who, kw)
  if (!isNone(kw.multioutput) && strOf(kw.multioutput) !== 'uniform_average') throw h.err('NotImplementedError', `${who}: multioutput=${repr(kw.multioutput!)} is not available in this sandbox`)
  if (!tV || !pV) throw h.err('TypeError', "missing a required argument: 'y_pred'")
  checkLengths(h, [tV, pV])
  const a = regressionTarget(h, tV)
  const b = regressionTarget(h, pV)
  if (a.t !== b.t) throw h.err('ValueError', `y_true and y_pred have different number of output (${a.t}!=${b.t})`)
  return { t: a.t, cols: Array.from({ length: a.t }, (_, j) => [a.rows.map((r) => r[j]), b.rows.map((r) => r[j])] as [number[], number[]]) }
}

// ---------------------------------------------------------------- datasets

const IRIS = '513514024930140247321302463115025036140254391704463414035034150244291402493115015437150248341602483014014330110158401202574415045439130451351403573817035138150354341702513715044636100251331705483419025030160250341604523515025234140247321602483116025434150452411501554214024931150250321202553513024936140144301302513415025035130345231303443213025035160651381904483014035138160246321402533715025033140270324714643245156931491555234013652846155728451363334716492433106629461352273914502035105930421560224010612947145629361367314414563045155827411062224515562539115932481861284013632549156128471264294313663044146828481467305017602945155726351055243811552437105827391260275116543045156034451667314715632344135630411355254013552644126130461458264012502333105627421357304212572942136229431351253011572841136333602558275119713059216329561865305822763066214925451773296318672558187236612565325120642753196830552157255020582851246432532365305518773867227726692360225015693257235628492077286720632749186733572172326018622848186130491864285621723058167428611979386420642856226328511561265614773061236334562464315518603048186931542167315624693151235827511968325923673357256730522363255019653052206234542359305118'
const IRIS_TARGET = (i: number) => Math.floor(i / 50)

const sampleNoReplace = (pop: number, k: number, rng: RandomState): number[] => {
  const ratio = k / pop
  if (ratio > 0.01 && ratio < 0.99) return rng.permutation(pop).slice(0, k)
  if (ratio < 0.2) {
    const seen = new Set<number>()
    const out: number[] = []
    while (out.length < k) {
      const j = rng.randint(0, pop)
      if (!seen.has(j)) {
        seen.add(j)
        out.push(j)
      }
    }
    return out
  }
  const out = Array.from({ length: k }, (_, i) => i)
  for (let i = k; i < pop; i++) {
    const j = rng.randint(0, i + 1)
    if (j < k) out[j] = i
  }
  return out
}

const makeBlobs = (h: Host): Fn => (args, kw) => {
  const names = ['n_samples', 'n_features', 'centers', 'cluster_std', 'center_box', 'shuffle', 'random_state', 'return_centers']
  rejectKw(h, 'make_blobs', kw, names)
  if (args.length) throw h.err('TypeError', `make_blobs() takes 0 positional arguments but ${args.length} were given`)
  const rng = rngOf(h, kw.random_state)
  const ns = kw.n_samples ?? py.int(100)
  const cv = kw.centers ?? py.NONE
  const box = kw.center_box ? h.iterate(kw.center_box).map((x) => h.num(x)) : [-10, 10]
  let nf = kw.n_features ? h.toInt(kw.n_features) : 2
  let centers: number[][]
  let perCenter: number[]
  const uniform = (k: number) => Array.from({ length: k }, () => Array.from({ length: nf }, () => rng.uniform(box[0], box[1])))
  if (ns.k === 'int') {
    if (cv.k === 'none' || cv.k === 'int') centers = uniform(cv.k === 'int' ? cv.v : 3)
    else {
      const a = readX(h, cv, 'make_blobs')
      centers = a.rows
      nf = a.m
    }
    const k = centers.length
    perCenter = Array.from({ length: k }, (_, i) => Math.floor(ns.v / k) + (i < ns.v % k ? 1 : 0))
  } else {
    perCenter = h.iterate(ns).map((x) => h.toInt(x))
    if (cv.k === 'none') centers = uniform(perCenter.length)
    else {
      const a = readX(h, cv, 'make_blobs')
      centers = a.rows
      nf = a.m
    }
    if (centers.length !== perCenter.length) throw h.err('ValueError', `Length of \`n_samples\` not consistent with number of centers. Got n_samples = ${repr(ns)} and centers = ${repr(cv)}`)
  }
  const stdV = kw.cluster_std ?? py.float(1)
  const stds = isNum(stdV) ? centers.map(() => numOf(stdV)) : h.iterate(stdV).map((x) => h.num(x))
  if (stds.length !== centers.length) throw h.err('ValueError', `Length of \`clusters_std\` not consistent with number of centers. Got centers = ${repr(cv)} and cluster_std = ${repr(stdV)}`)
  let X: number[][] = []
  let y: number[] = []
  perCenter.forEach((n, i) => {
    for (let s = 0; s < n; s++) {
      X.push(centers[i].map((c) => c + stds[i] * rng.normal()))
      y.push(i)
    }
  })
  if (!kw.shuffle || h.truthy(kw.shuffle)) {
    const ix = X.map((_, i) => i)
    rng.shuffle(ix)
    X = ix.map((i) => X[i])
    y = ix.map((i) => y[i])
  }
  const out = [mat(X, nf), vec(y, 'int64')]
  if (kw.return_centers && h.truthy(kw.return_centers)) out.push(mat(centers, nf))
  return py.tuple(out)
}

const makeClassification = (h: Host): Fn => (args, kw) => {
  const names = ['n_samples', 'n_features', 'n_informative', 'n_redundant', 'n_repeated', 'n_classes', 'n_clusters_per_class', 'weights', 'flip_y', 'class_sep', 'hypercube', 'shift', 'scale', 'shuffle', 'random_state']
  rejectKw(h, 'make_classification', kw, names)
  if (args.length) throw h.err('TypeError', `make_classification() takes 0 positional arguments but ${args.length} were given`)
  const i = (k: string, d: number) => (kw[k] ? h.toInt(kw[k]) : d)
  const n = i('n_samples', 100)
  const nf = i('n_features', 20)
  const ni = i('n_informative', 2)
  const nr = i('n_redundant', 2)
  const nrep = i('n_repeated', 0)
  const nc = i('n_classes', 2)
  const ncl = i('n_clusters_per_class', 2)
  const flip = floatArg(h, kw.flip_y, 0.01)
  const sep = floatArg(h, kw.class_sep, 1)
  const rng = rngOf(h, kw.random_state)
  if (ni + nr + nrep > nf) throw h.err('ValueError', 'Number of informative, redundant and repeated features must sum to less than the number of total features')
  if (ni < Math.log2(nc * ncl)) throw h.err('ValueError', `n_classes(${nc}) * n_clusters_per_class(${ncl}) must be smaller or equal 2**n_informative(${ni})=${2 ** ni}`)
  if (ni > 30) throw h.err('NotImplementedError', 'make_classification: n_informative above 30 is not available in this sandbox')
  let w = new Array<number>(nc).fill(1 / nc)
  if (!isNone(kw.weights)) {
    const given = toArray(kw.weights!, 'float64').values()
    if (given.length !== nc && given.length !== nc - 1) throw h.err('ValueError', 'Weights specified but incompatible with number of classes.')
    w = given.length === nc - 1 ? [...given, 1 - given.reduce((a, b) => a + b, 0)] : given
  }
  const nRandom = nf - ni - nr - nrep
  const clusters = nc * ncl
  const per = Array.from({ length: clusters }, (_, k) => Math.trunc((n * w[k % nc]) / ncl))
  const extra = n - per.reduce((a, b) => a + b, 0)
  for (let k = 0; k < extra; k++) per[k % clusters]++
  const X = Array.from({ length: n }, () => new Array<number>(nf).fill(0))
  const y = new Array<number>(n).fill(0)
  const verts = sampleNoReplace(2 ** ni, clusters, rng)
  let cents = verts.map((v) => Array.from({ length: ni }, (_, b) => (Math.floor(v / 2 ** (ni - 1 - b)) % 2) * (2 * sep) - sep))
  if (kw.hypercube && !h.truthy(kw.hypercube)) {
    const a = Array.from({ length: clusters }, () => rng.double())
    const b = Array.from({ length: ni }, () => rng.double())
    cents = cents.map((c, k) => c.map((x, j) => x * a[k] * b[j]))
  }
  for (let r = 0; r < n; r++) for (let j = 0; j < ni; j++) X[r][j] = rng.normal()
  let stop = 0
  cents.forEach((cen, k) => {
    const start = stop
    stop += per[k]
    const A = Array.from({ length: ni }, () => Array.from({ length: ni }, () => 2 * rng.double() - 1))
    for (let r = start; r < stop; r++) {
      y[r] = k % nc
      const row = X[r].slice(0, ni)
      for (let j = 0; j < ni; j++) {
        let s = 0
        for (let q = 0; q < ni; q++) s += row[q] * A[q][j]
        X[r][j] = s + cen[j]
      }
    }
  })
  if (nr > 0) {
    const B = Array.from({ length: ni }, () => Array.from({ length: nr }, () => 2 * rng.double() - 1))
    for (const row of X) {
      for (let j = 0; j < nr; j++) {
        let s = 0
        for (let q = 0; q < ni; q++) s += row[q] * B[q][j]
        row[ni + j] = s
      }
    }
  }
  const base = ni + nr
  if (nrep > 0) {
    const idx = Array.from({ length: nrep }, () => Math.trunc((base - 1) * rng.double() + 0.5))
    for (const row of X) idx.forEach((q, j) => { row[base + j] = row[q] })
  }
  if (nRandom > 0) for (const row of X) for (let j = nf - nRandom; j < nf; j++) row[j] = rng.normal()
  if (flip >= 0) {
    const mask = Array.from({ length: n }, () => rng.double() < flip)
    mask.forEach((m, r) => { if (m) y[r] = rng.randint(0, nc) })
  }
  // shift=None and scale=None draw at random; the defaults are 0 and 1
  const given = (v: Value | undefined, d: number, draw: () => number) => (v && v.k === 'none' ? Array.from({ length: nf }, draw) : !v ? new Array<number>(nf).fill(d) : isNum(v) ? new Array<number>(nf).fill(numOf(v)) : toArray(v, 'float64').values())
  const shift = given(kw.shift, 0, () => (2 * rng.double() - 1) * sep)
  for (const row of X) for (let j = 0; j < nf; j++) row[j] += shift[j]
  const scale = given(kw.scale, 1, () => 1 + 100 * rng.double())
  for (const row of X) for (let j = 0; j < nf; j++) row[j] *= scale[j]
  let Xo = X
  let yo = y
  if (!kw.shuffle || h.truthy(kw.shuffle)) {
    const ix = X.map((_, r) => r)
    rng.shuffle(ix)
    Xo = ix.map((r) => X[r])
    yo = ix.map((r) => y[r])
    const fi = Array.from({ length: nf }, (_, j) => j)
    rng.shuffle(fi)
    Xo = Xo.map((row) => fi.map((j) => row[j]))
  }
  return py.tuple([mat(Xo, nf), vec(yo, 'int64')])
}

// ---------------------------------------------------------------- the module

const build = (h: Host): Record<string, Value> => {
  const fn = (name: string, call: Fn, kind: ApiKind = 'create'): Value => ({ k: 'fn', name, call, api: { name: `sklearn.${name}`, kind } } as Value)
  const type = (name: string, call: Fn): Value => ({ k: 'type', name: `sklearn.${name}`, call } as Value)
  const estType = (spec: Spec): Value => {
    const t: Value = { k: 'type', name: spec.name, call: (args, kw) => py.obj(makeEst(spec, args, kw, h)) }
    return t
  }
  const mod = (name: string, attrs: Record<string, Value>): Value => ({ k: 'module', name: `sklearn.${name}`, attrs })

  const metrics: Record<string, Value> = {
    accuracy_score: fn('accuracy_score', (args, kw) => {
      noWeights(h, 'accuracy_score', kw)
      const { t, p } = targetPair(h, args[0] ?? kw.y_true, args[1] ?? kw.y_pred)
      const c = countMatches(t, p)
      const norm = args[2] ?? kw.normalize
      return norm && !h.truthy(norm) ? py.float(c) : py.float(npMean(t.vals.map((v, i) => (v === p.vals[i] ? 1 : 0))))
    }, 'reduce'),
    confusion_matrix: fn('confusion_matrix', (args, kw) => {
      noWeights(h, 'confusion_matrix', kw)
      const { t, p } = targetPair(h, args[0] ?? kw.y_true, args[1] ?? kw.y_pred)
      const labels = labelsArg(h, args[2] ?? kw.labels) ?? uniqueSorted([...t.vals, ...p.vals], t.isStr)
      if (!labels.length) throw h.err('ValueError', "'labels' should contains at least one label.")
      if (!labels.some((l) => t.vals.includes(l))) throw h.err('ValueError', 'At least one label specified must be in y_true')
      const cm = confusionCounts(t, p, labels)
      const nv = kw.normalize
      if (isNone(nv)) return mat(cm, labels.length, 'int64')
      const how = strOf(nv)
      if (how !== 'true' && how !== 'pred' && how !== 'all') throw badParam(h, `The 'normalize' parameter of confusion_matrix must be a str among {'all', 'pred', 'true'} or None. Got ${repr(nv!)} instead.`)
      const rowT = cm.map((r) => r.reduce((a, b) => a + b, 0))
      const colT = labels.map((_, j) => cm.reduce((a, r) => a + r[j], 0))
      const all = rowT.reduce((a, b) => a + b, 0)
      return mat(cm.map((r, i) => r.map((x, j) => {
        const d = how === 'true' ? rowT[i] : how === 'pred' ? colT[j] : all
        return d === 0 ? 0 : x / d
      })), labels.length)
    }),
    precision_score: fn('precision_score', (args, kw) => {
      const r = prfCore(h, args[0] ?? kw.y_true, args[1] ?? kw.y_pred, { labels: args[2] ?? kw.labels, ...kw }, 'precision_score')
      return Array.isArray(r.p) ? vec(r.p) : py.float(r.p as number)
    }, 'reduce'),
    recall_score: fn('recall_score', (args, kw) => {
      const r = prfCore(h, args[0] ?? kw.y_true, args[1] ?? kw.y_pred, { labels: args[2] ?? kw.labels, ...kw }, 'recall_score')
      return Array.isArray(r.r) ? vec(r.r) : py.float(r.r as number)
    }, 'reduce'),
    f1_score: fn('f1_score', (args, kw) => {
      const r = prfCore(h, args[0] ?? kw.y_true, args[1] ?? kw.y_pred, { labels: args[2] ?? kw.labels, ...kw }, 'f1_score')
      return Array.isArray(r.f) ? vec(r.f) : py.float(r.f as number)
    }, 'reduce'),
    fbeta_score: fn('fbeta_score', (args, kw) => {
      const { beta, ...rest } = kw
      const r = prfCore(h, args[0] ?? kw.y_true, args[1] ?? kw.y_pred, rest, 'fbeta_score', h.num(beta ?? args[2]))
      return Array.isArray(r.f) ? vec(r.f) : py.float(r.f as number)
    }, 'reduce'),
    precision_recall_fscore_support: fn('precision_recall_fscore_support', (args, kw) => {
      const r = prfCore(h, args[0] ?? kw.y_true, args[1] ?? kw.y_pred, { average: py.NONE, ...kw }, 'precision_recall_fscore_support')
      if (Array.isArray(r.p)) return py.tuple([vec(r.p), vec(r.r as number[]), vec(r.f as number[]), vec(r.s as number[], 'int64')])
      return py.tuple([py.float(r.p), py.float(r.r as number), py.float(r.f as number), py.NONE])
    }),
    classification_report: fn('classification_report', (args, kw) => classificationReport(h, args[0] ?? kw.y_true, args[1] ?? kw.y_pred, { labels: args[2] ?? kw.labels, target_names: args[3] ?? kw.target_names, ...kw })),
    mean_squared_error: fn('mean_squared_error', (args, kw) => {
      for (const k of Object.keys(kw)) if (!['y_true', 'y_pred', 'sample_weight', 'multioutput'].includes(k)) throw h.err('TypeError', `got an unexpected keyword argument '${k}'`)
      const { cols } = regressionPair(h, args[0] ?? kw.y_true, args[1] ?? kw.y_pred, kw, 'mean_squared_error')
      return py.float(npMean(cols.map(([a, b]) => npMean(a.map((x, i) => (x - b[i]) ** 2)))))
    }, 'reduce'),
    root_mean_squared_error: fn('root_mean_squared_error', (args, kw) => {
      const { cols } = regressionPair(h, args[0] ?? kw.y_true, args[1] ?? kw.y_pred, kw, 'root_mean_squared_error')
      return py.float(npMean(cols.map(([a, b]) => Math.sqrt(npMean(a.map((x, i) => (x - b[i]) ** 2))))))
    }, 'reduce'),
    mean_absolute_error: fn('mean_absolute_error', (args, kw) => {
      const { cols } = regressionPair(h, args[0] ?? kw.y_true, args[1] ?? kw.y_pred, kw, 'mean_absolute_error')
      return py.float(npMean(cols.map(([a, b]) => npMean(a.map((x, i) => Math.abs(x - b[i]))))))
    }, 'reduce'),
    r2_score: fn('r2_score', (args, kw) => {
      const { cols } = regressionPair(h, args[0] ?? kw.y_true, args[1] ?? kw.y_pred, kw, 'r2_score')
      return py.float(npMean(cols.map(([a, b]) => r2Of(a, b))))
    }, 'reduce'),
    roc_curve: fn('roc_curve', (args, kw) => {
      const r = rocCurve(h, args[0] ?? kw.y_true, args[1] ?? kw.y_score, kw.pos_label)
      return py.tuple([vec(r.fpr), vec(r.tpr), vec(r.thr)])
    }),
    auc: fn('auc', (args, kw) => {
      const x = numbersOf(h, args[0] ?? kw.x)
      const y = numbersOf(h, args[1] ?? kw.y)
      if (x.length < 2) throw h.err('ValueError', `At least 2 points are needed to compute area under curve, but x.shape = (${x.length},)`)
      return py.float(Math.abs(trapezoid(x, y)))
    }, 'reduce'),
    roc_auc_score: fn('roc_auc_score', (args, kw) => {
      noWeights(h, 'roc_auc_score', kw)
      const tV = args[0] ?? kw.y_true
      const sV = args[1] ?? kw.y_score
      for (const k of ['max_fpr', 'labels']) if (!isNone(kw[k])) throw h.err('NotImplementedError', `roc_auc_score: ${k} is not available in this sandbox`)
      const y = labelsOf(h, tV)
      const nc = new Set(y.vals).size
      if (nc > 2) {
        if (isNone(kw.multi_class) || strOf(kw.multi_class) === 'raise') throw h.err('ValueError', "multi_class must be in ('ovo', 'ovr')")
        throw h.err('NotImplementedError', 'roc_auc_score: multi_class is not available in this sandbox')
      }
      if (nc !== 2) return py.float(NaN)
      const classes = uniqueSorted(y.vals, y.isStr)
      const bin = vec(y.vals.map((v) => (v === classes[1] ? 1 : 0)), 'int64')
      const r = rocCurve(h, bin, sV)
      return py.float(trapezoid(r.fpr, r.tpr))
    }, 'reduce'),
    silhouette_score: fn('silhouette_score', (args, kw) => {
      if (!isNone(kw.metric) && strOf(kw.metric) !== 'euclidean') throw h.err('NotImplementedError', `silhouette_score: metric=${repr(kw.metric!)} is not available in this sandbox`)
      if (!isNone(kw.sample_size)) throw h.err('NotImplementedError', 'silhouette_score: sample_size is not available in this sandbox')
      const X = readX(h, args[0] ?? kw.X, 'silhouette_score')
      const lab = labelsOf(h, args[1] ?? kw.labels)
      const classes = uniqueSorted(lab.vals, lab.isStr)
      if (classes.length < 2 || classes.length > X.n - 1) throw h.err('ValueError', `Number of labels is ${classes.length}. Valid values are 2 to n_samples - 1 (inclusive)`)
      const at = new Map(classes.map((c, i) => [c, i]))
      const enc = lab.vals.map((v) => at.get(v)!)
      const counts = classes.map(() => 0)
      enc.forEach((c) => counts[c]++)
      const sil = X.rows.map((x, i) => {
        const sums = classes.map(() => 0)
        X.rows.forEach((y, j) => { sums[enc[j]] += Math.sqrt(sqEuclid(x, y)) })
        const own = enc[i]
        if (counts[own] === 1) return 0
        const a = sums[own] / (counts[own] - 1)
        let b = Infinity
        classes.forEach((_, c) => { if (c !== own) b = Math.min(b, sums[c] / counts[c]) })
        return (b - a) / Math.max(a, b)
      })
      return py.float(npMean(sil))
    }, 'reduce'),
  }

  const modelSelection: Record<string, Value> = {
    train_test_split: fn('train_test_split', trainTestSplit(h)),
    KFold: type('KFold', (args, kw) => makeKFold('KFold')(h, args, kw)),
    StratifiedKFold: type('StratifiedKFold', (args, kw) => makeKFold('StratifiedKFold')(h, args, kw)),
    LeaveOneOut: type('LeaveOneOut', () => py.obj(new Splitter('LeaveOneOut', []))),
    ShuffleSplit: type('ShuffleSplit', (args, kw) => py.obj(new Splitter('ShuffleSplit', [['n_splits', args[0] ?? kw.n_splits ?? py.int(10)], ['random_state', kw.random_state ?? py.NONE], ['test_size', kw.test_size ?? py.NONE], ['train_size', kw.train_size ?? py.NONE]]))),
    cross_val_score: fn('cross_val_score', (args, kw) => {
      rejectKw(h, 'cross_val_score', kw, ['estimator', 'X', 'y', 'groups', 'scoring', 'cv', 'n_jobs', 'verbose', 'params', 'pre_dispatch', 'error_score'])
      const e = estOf(args[0] ?? kw.estimator)
      if (!e) throw h.err('TypeError', "estimator should be an estimator implementing 'fit' method")
      const X = args[1] ?? kw.X
      const y = args[2] ?? kw.y
      if (y && y.k !== 'none') checkLengths(h, [X, y])
      return vec(crossValScores(h, e, X, y, kw.scoring, kw.cv))
    }),
    GridSearchCV: estType(GRID_SEARCH),
  }

  const iris = (args: Value[], kw: Kw): Value => {
    rejectKw(h, 'load_iris', kw, ['return_X_y', 'as_frame'])
    const rows = Array.from({ length: 150 }, (_, i) => Array.from({ length: 4 }, (_x, j) => Number(IRIS.slice((i * 4 + j) * 2, (i * 4 + j) * 2 + 2)) / 10))
    const target = rows.map((_, i) => IRIS_TARGET(i))
    const names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    const asFrame = !!(kw.as_frame && h.truthy(kw.as_frame))
    const retXY = !!((args[0] ?? kw.return_X_y) && h.truthy(args[0] ?? kw.return_X_y))
    const data: Value = asFrame ? py.obj(new Frame(names, colsOf(rows, 4).map((c, j) => new Series(c, null, names[j], 'float64')), null)) : mat(rows, 4)
    const tv: Value = asFrame ? py.obj(new Series(target, null, 'target', 'int64')) : vec(target, 'int64')
    if (retXY) return py.tuple([data, tv])
    const frame = asFrame ? py.obj(new Frame([...names, 'target'], [...colsOf(rows, 4).map((c, j) => new Series(c, null, names[j], 'float64')), new Series(target, null, 'target', 'int64')], null)) : py.NONE
    return py.obj(new BunchObj([['data', data], ['target', tv], ['frame', frame], ['target_names', py.obj(new ObjArray(['setosa', 'versicolor', 'virginica'], '<U10'))], ['feature_names', py.list(names.map((s) => py.str(s)))], ['filename', py.str('iris.csv')], ['data_module', py.str('sklearn.datasets.data')]]))
  }
  const notEmbedded = (name: string): Value => ({ k: 'fn', name, call: () => { throw h.err('NotImplementedError', `sklearn.datasets.${name}: only load_iris is embedded in this sandbox; make_blobs and make_classification generate data`) } })

  const pipe = (args: Value[], kw: Kw): Value => {
    for (const k of Object.keys(kw)) if (k !== 'memory' && k !== 'verbose' && k !== 'transform_input') throw h.err('TypeError', `make_pipeline() got an unexpected keyword argument '${k}'`)
    const names = args.map((a) => (estOf(a)?.cls ?? (a.k === 'str' ? a.v : typeof a)).toLowerCase())
    const counts = new Map<string, number>()
    names.forEach((n) => counts.set(n, (counts.get(n) ?? 0) + 1))
    const seen = new Map<string, number>()
    const steps = args.map((a, i) => {
      const n = names[i]
      let label = n
      if ((counts.get(n) ?? 0) > 1) {
        const c = (seen.get(n) ?? 0) + 1
        seen.set(n, c)
        label = `${n}-${c}`
      }
      return py.tuple([py.str(label), a])
    })
    return py.obj(makeEst(PIPELINE, [py.list(steps)], kw, h))
  }

  const modules: Record<string, Value> = {
    'sklearn.model_selection': mod('model_selection', modelSelection),
    'sklearn.preprocessing': mod('preprocessing', {
      LabelEncoder: estType(LABEL_ENCODER), OneHotEncoder: estType(ONE_HOT), OrdinalEncoder: estType(ORDINAL), StandardScaler: estType(STANDARD_SCALER), MinMaxScaler: estType(MINMAX_SCALER), Normalizer: estType(NORMALIZER), PolynomialFeatures: estType(POLY), FunctionTransformer: estType(FUNCTION_TRANSFORMER),
    }),
    'sklearn.compose': mod('compose', { ColumnTransformer: estType(COLUMN_TRANSFORMER) }),
    'sklearn.pipeline': mod('pipeline', { Pipeline: estType(PIPELINE), make_pipeline: { k: 'fn', name: 'make_pipeline', call: pipe } }),
    'sklearn.neighbors': mod('neighbors', { KNeighborsClassifier: estType(KNN_CLASSIFIER), KNeighborsRegressor: estType(KNN_REGRESSOR) }),
    'sklearn.cluster': mod('cluster', { KMeans: estType(KMEANS) }),
    'sklearn.naive_bayes': mod('naive_bayes', { GaussianNB: estType(GAUSSIAN_NB), CategoricalNB: estType(CATEGORICAL_NB), MultinomialNB: estType(countingNB('MultinomialNB')), BernoulliNB: estType(countingNB('BernoulliNB')) }),
    'sklearn.linear_model': mod('linear_model', { Perceptron: estType(PERCEPTRON), LinearRegression: estType(LINEAR_REGRESSION), Ridge: estType(RIDGE), LogisticRegression: estType(LOGISTIC_REGRESSION) }),
    'sklearn.tree': mod('tree', { DecisionTreeClassifier: estType(DECISION_TREE), export_text: { k: 'fn', name: 'export_text', call: (args, kw) => py.str(exportText(h, args[0] ?? kw.decision_tree, kw)) } }),
    'sklearn.decomposition': mod('decomposition', { PCA: estType(PCA) }),
    'sklearn.metrics': mod('metrics', metrics),
    'sklearn.datasets': mod('datasets', {
      load_iris: { k: 'fn', name: 'load_iris', call: iris },
      make_blobs: { k: 'fn', name: 'make_blobs', call: makeBlobs(h) },
      make_classification: { k: 'fn', name: 'make_classification', call: makeClassification(h) },
      load_digits: notEmbedded('load_digits'), load_wine: notEmbedded('load_wine'), load_breast_cancer: notEmbedded('load_breast_cancer'), load_diabetes: notEmbedded('load_diabetes'), fetch_openml: notEmbedded('fetch_openml'),
    }),
    'sklearn.base': mod('base', { clone: { k: 'fn', name: 'clone', call: (args) => cloneValue(args[0]) } }),
    'sklearn.exceptions': mod('exceptions', { NotFittedError: notFittedClass(h), ConvergenceWarning: excOf(h, 'ConvergenceWarning', ['UserWarning']) }),
  }
  const root: Record<string, Value> = { __version__: py.str('1.7.2') }
  for (const [path, m] of Object.entries(modules)) root[path.split('.')[1]] = m
  return { sklearn: { k: 'module', name: 'sklearn', attrs: root }, ...modules }
}

export const SKLEARN: PyLib = {
  modules: ['sklearn', 'sklearn.model_selection', 'sklearn.preprocessing', 'sklearn.compose', 'sklearn.pipeline', 'sklearn.neighbors', 'sklearn.cluster', 'sklearn.naive_bayes', 'sklearn.linear_model', 'sklearn.tree', 'sklearn.decomposition', 'sklearn.metrics', 'sklearn.datasets', 'sklearn.base', 'sklearn.exceptions'],
  load: build,
}
