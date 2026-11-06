/**
 * The methods of Series, DataFrame and Index. Each entry names the parameters it implements; any other parameter raises
 * NotImplementedError instead of being ignored. Numeric kernels follow pandas' nanops (and numpy's pairwise sum and
 * percentile) so results agree to the last digit where the arithmetic allows it.
 */
import { PyDict, PyObj, parse, py, repr } from './minipy'
import type { Host, Node, Value } from './minipy'
import { NDArray } from './ndarray'
import { Frame, IndexObj, IntervalIndexObj, ObjArray, sharedWidth, DtypeObj, Series, PyCell, boxFloat, cellOf, cellsOf, commonDT, convertCells, dtName, frameIndexer, frameOp, frameToString, inferDT, inferValues, isListLike, isMissing, labelStr, method, numCell, positions, reindexCells, seriesIndexer, seriesOp, seriesToString, sortOrder, typedCells, unionIndex, valueOf, seriesIn, frameIn } from './pyPandas'
import type { Cats, Cell, Ctx, DT, Method, Params } from './pyPandas'
import { OPTIONS, fixedStr, formatArray } from './pyPandasFormat'
import { groupByFrame, groupBySeries, windowOf } from './pyPandasGroup'
import { frameToCsv, frameToJson, seriesToCsv, seriesToJson } from './pyPandasIO'
import { joinFrames, meltFrame, mergeFrames, pivotFrame, pivotTable, stackFrame, unstackFrame, unstackSeries } from './pyPandasReshape'
import { pyRegex, strAccessor, indexStrAccessor } from './pyPandasStr'

// ---------------------------------------------------------------- argument helpers

/** a parameter that was given and is not None */
export const given = (v: Value | undefined): Value | undefined => (v && v.k !== 'none' ? v : undefined)
export const flag = (h: Host, v: Value | undefined, d = false) => (v ? h.truthy(v) : d)
export const axisOf = (h: Host, v: Value | undefined, cls: string, d: 0 | 1 = 0): 0 | 1 => {
  if (!given(v)) return d
  const bad = () => h.err('ValueError', `No axis named ${v!.k === 'str' ? v!.v : repr(v!)} for object type ${cls}`)
  if (v!.k === 'str') {
    if (v!.v === 'index' || v!.v === 'rows') return 0
    if (v!.v === 'columns' && cls === 'DataFrame') return 1
    throw bad()
  }
  const i = h.toInt(v!)
  if (i === 0) return 0
  if (i === 1 && cls === 'DataFrame') return 1
  throw bad()
}
export const need = (h: Host, v: Value | undefined, fn: string, name: string): Value => {
  if (!v) throw h.err('TypeError', `${h.state.get('pandas.cls') ?? 'DataFrame'}.${fn}() missing 1 required positional argument: '${name}'`)
  return v
}
/** a parameter whose non-default values are not implemented */
const refuse = (h: Host, v: Value | undefined, fn: string, name: string) => {
  if (given(v)) throw h.err('NotImplementedError', `${fn}(${name}=...) is not available in this sandbox`)
}
const oneOf = (h: Host, v: Value | undefined, d: string, allowed: string[], fn: string, name: string): string => {
  const s = given(v) ? h.str(v!) : d
  if (!allowed.includes(s)) throw h.err('NotImplementedError', `${fn}(${name}='${s}') is not available in this sandbox`)
  return s
}
const callableValue = (v: Value | undefined) => !!v && (v.k === 'func' || v.k === 'fn' || v.k === 'bound' || v.k === 'type' || v.k === 'class')
const rows = (n: number) => Array.from({ length: n }, (_, i) => i)
/** inplace=True swaps the result in and returns None */
const finish = <S extends Series | Frame>(h: Host, self: S, out: S, inplace: Value | undefined): Value => {
  if (flag(h, inplace)) {
    ;(self as Frame).adopt(out as Frame)
    return py.NONE
  }
  return py.obj(out)
}
const noteOnce = (h: Host, key: string) => {
  const seen = (h.state.get('pandas.notes') as Set<string> | undefined) ?? new Set<string>()
  h.state.set('pandas.notes', seen)
  if (seen.has(key) || !h.tracing) return
  seen.add(key)
  h.emit({ type: 'note', tone: 'info', key, params: {} })
}
const fn = (name: string, call: (args: Value[], kw: Record<string, Value>) => Value): Value => ({ k: 'fn', name, call })
const has = (t: object, k: string) => Object.prototype.hasOwnProperty.call(t, k)

/** the Python name of an aggregation: 'sum', np.sum → 'sum', a lambda → '<lambda>' */
export const funcName = (h: Host, f: Value): string => (f.k === 'str' ? f.v : f.k === 'fn' || f.k === 'func' || f.k === 'type' ? (f.name === 'lambda' ? '<lambda>' : f.name.replace(/^numpy\./, '')) : h.str(f))

// ---------------------------------------------------------------- numeric kernels

/** numpy's pairwise summation */
const pairwise = (a: number[], lo: number, n: number): number => {
  if (n < 8) {
    let r = -0
    for (let i = 0; i < n; i++) r += a[lo + i]
    return r
  }
  if (n <= 128) {
    const r = [a[lo], a[lo + 1], a[lo + 2], a[lo + 3], a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]]
    let i = 8
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[lo + i + j]
    let res = r[0] + r[1] + (r[2] + r[3]) + (r[4] + r[5] + (r[6] + r[7]))
    for (; i < n; i++) res += a[lo + i]
    return res
  }
  let n2 = Math.floor(n / 2)
  n2 -= n2 % 8
  return pairwise(a, lo, n2) + pairwise(a, lo + n2, n - n2)
}
/** np.add.reduce */
export const npSum = (a: number[]): number => (a.length ? pairwise(a, 0, a.length) : 0)

const zeroFperr = (x: number) => (Math.abs(x) < 1e-14 ? 0 : x)

const median = (present: number[]) => {
  const v = [...present].sort((a, b) => a - b)
  const n = v.length
  if (!n) return NaN
  return n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2
}

/** numpy's percentile on sorted values */
export const npQuantile = (sorted: number[], q: number, interp = 'linear'): number => {
  const n = sorted.length
  if (!n) return NaN
  const quant = (100.0 * q) / 100
  const vi = (n - 1) * quant
  if (interp === 'lower') return sorted[Math.floor(vi)]
  if (interp === 'higher') return sorted[Math.ceil(vi)]
  if (interp === 'nearest') {
    const f = Math.floor(vi)
    const d = vi - f
    return sorted[d === 0.5 ? (f % 2 === 0 ? f : f + 1) : Math.round(vi)]
  }
  if (interp === 'midpoint') return (sorted[Math.floor(vi)] + sorted[Math.ceil(vi)]) / 2
  if (vi >= n - 1) return sorted[n - 1]
  const lo = Math.floor(vi)
  const t = vi - lo
  const a = sorted[lo]
  const b = sorted[lo + 1]
  const diff = b - a
  return t >= 0.5 ? b - diff * (1 - t) : a + diff * t
}

const momentStats = (vals: number[], mask: boolean[], count: number) => {
  const zeroed = vals.map((v, i) => (mask[i] ? 0 : v))
  const mean = npSum(zeroed) / count
  const adjusted = zeroed.map((v, i) => (mask[i] ? 0 : v - mean))
  return adjusted
}

export interface ReduceOpts { skipna?: boolean; ddof?: number; minCount?: number; frame?: boolean }

const reduceText = (h: Host, s: Series, how: string, o: ReduceOpts): Value => {
  const vs = s.values
  const firstText = vs.find((c) => typeof c === 'string') as string
  const missing = vs.some(isMissing)
  const list = (xs: Cell[]) => `[${xs.map((x) => (typeof x === 'string' ? `'${x}'` : labelStr(x))).join(' ')}]`
  switch (how) {
    case 'sum':
      if (missing) throw h.err('TypeError', 'can only concatenate str (not "int") to str')
      if (vs.some((c) => typeof c !== 'string')) throw h.err('TypeError', `can only concatenate str (not "${typeof vs.find((c) => typeof c !== 'string') === 'boolean' ? 'bool' : 'int'}") to str`)
      return py.str(vs.join(''))
    case 'min': case 'max': {
      if (missing) throw h.err('TypeError', `'${how === 'max' ? '>=' : '<='}' not supported between instances of 'str' and '${vs.find(isMissing) === null ? 'NoneType' : 'float'}'`)
      if (vs.some((c) => typeof c !== 'string')) throw h.err('TypeError', `'${how === 'max' ? '>=' : '<='}' not supported between instances of 'str' and 'int'`)
      const t = vs as string[]
      return py.str(t.reduce((a, b) => (how === 'min' ? (b < a ? b : a) : b > a ? b : a)))
    }
    case 'mean':
      if (missing) throw h.err('TypeError', 'can only concatenate str (not "int") to str')
      throw h.err('TypeError', o.frame ? `Could not convert ['${vs.join('')}'] to numeric` : `Could not convert string '${vs.join('')}' to numeric`)
    case 'median':
      throw h.err('TypeError', o.frame ? `Cannot convert [${list(vs)}] to numeric` : `Cannot convert ${list(vs)} to numeric`)
    case 'prod': throw h.err('TypeError', "can't multiply sequence by non-int of type 'str'")
  }
  throw h.err('TypeError', `could not convert string to float: '${firstText}'`)
}

/** one column reduced to one value, skipping missing values unless skipna=False */
export const reduceCol = (h: Host, s: Series, how: string, o: ReduceOpts = {}): Value => {
  const skipna = o.skipna ?? true
  const ddof = o.ddof ?? 1
  if (how === 'count') return py.int(s.values.filter((c) => !isMissing(c)).length)
  if (s.cats) {
    if ((how === 'min' || how === 'max') && s.cats.ordered) {
      const labels = s.cats.labels
      const pos = s.values.filter((c) => !isMissing(c)).map((c) => labels.indexOf(c))
      if (!pos.length) return py.float(NaN)
      return valueOf(labels[how === 'min' ? Math.min(...pos) : Math.max(...pos)], s.dtype)
    }
    throw h.err('TypeError', `'Categorical' with dtype category does not support reduction '${how}'`)
  }
  const text = s.dtype === 'object' && s.values.some((c) => typeof c === 'string' || (c instanceof PyCell && c.v.k !== 'float' && c.v.k !== 'int'))
  if (text) return reduceText(h, s, how, o)
  const vals = s.values.map(numCell)
  const mask = vals.map(Number.isNaN)
  const count = mask.reduce((a, m) => a + (m ? 0 : 1), 0)
  if (!skipna && count < vals.length) return py.float(NaN)
  const isIntLike = s.dtype === 'int64' || s.dtype === 'bool' || (s.dtype === 'object' && !s.values.some((c) => c instanceof PyCell || (typeof c === 'number' && !Number.isInteger(c) && !Number.isNaN(c))))
  const asDT = (x: number) => (isIntLike ? py.int(x) : py.float(x))
  const below = o.minCount !== undefined && count < o.minCount
  switch (how) {
    case 'sum': return below ? py.float(NaN) : asDT(npSum(vals.map((v, i) => (mask[i] ? 0 : v))))
    case 'prod': {
      if (below) return py.float(NaN)
      let p = 1
      vals.forEach((v, i) => { if (!mask[i]) p *= v })
      return asDT(p)
    }
    case 'mean': return py.float(count ? npSum(vals.map((v, i) => (mask[i] ? 0 : v))) / count : NaN)
    case 'median': return py.float(median(vals.filter((_, i) => !mask[i])))
    case 'min': case 'max': {
      if (!count) return py.float(NaN)
      let best = NaN
      vals.forEach((v, i) => { if (!mask[i] && (Number.isNaN(best) || (how === 'min' ? v < best : v > best))) best = v })
      if (s.dtype === 'object') {
        const at = vals.findIndex((v, i) => !mask[i] && v === best)
        return valueOf(s.values[at], 'object')
      }
      return s.dtype === 'bool' ? py.bool(best !== 0) : asDT(best)
    }
    case 'var': case 'std': case 'sem': {
      if (count <= ddof) return py.float(NaN)
      const zeroed = vals.map((v, i) => (mask[i] ? 0 : v))
      const avg = npSum(zeroed) / count
      const sqr = zeroed.map((v, i) => (mask[i] ? 0 : (avg - v) * (avg - v)))
      const va = npSum(sqr) / (count - ddof)
      return py.float(how === 'var' ? va : how === 'std' ? Math.sqrt(va) : Math.sqrt(va) / Math.sqrt(count))
    }
    case 'skew': {
      if (count < 3) return py.float(NaN)
      const adj = momentStats(vals, mask, count)
      const m2 = zeroFperr(npSum(adj.map((x) => x * x)))
      const m3 = zeroFperr(npSum(adj.map((x) => x * x * x)))
      if (m2 === 0) return py.float(0)
      return py.float(((count * (count - 1) ** 0.5) / (count - 2)) * (m3 / m2 ** 1.5))
    }
    case 'kurt': {
      if (count < 4) return py.float(NaN)
      const adj = momentStats(vals, mask, count)
      const m2 = npSum(adj.map((x) => x * x))
      const m4 = npSum(adj.map((x) => x * x * (x * x)))
      const a = (3 * (count - 1) ** 2) / ((count - 2) * (count - 3))
      const numer = zeroFperr(count * (count + 1) * (count - 1) * m4)
      const denom = zeroFperr((count - 2) * (count - 3) * (m2 * m2))
      if (denom === 0) return py.float(0)
      return py.float(numer / denom - a)
    }
  }
  throw h.err('AttributeError', how)
}

const REDUCERS: Record<string, [string, string]> = {
  mean: ['mean', 'axis skipna numeric_only'], median: ['median', 'axis skipna numeric_only'], min: ['min', 'axis skipna numeric_only'], max: ['max', 'axis skipna numeric_only'],
  sum: ['sum', 'axis skipna numeric_only min_count'], prod: ['prod', 'axis skipna numeric_only min_count'], product: ['prod', 'axis skipna numeric_only min_count'],
  std: ['std', 'axis skipna ddof numeric_only'], var: ['var', 'axis skipna ddof numeric_only'], sem: ['sem', 'axis skipna ddof numeric_only'],
  skew: ['skew', 'axis skipna numeric_only'], kurt: ['kurt', 'axis skipna numeric_only'], kurtosis: ['kurt', 'axis skipna numeric_only'],
}

const reduceOpts = (h: Host, p: Params): ReduceOpts => ({ skipna: flag(h, p.skipna, true), ddof: given(p.ddof) ? h.toInt(p.ddof!) : 1, minCount: given(p.min_count) ? h.toInt(p.min_count!) : undefined })

/** results that keep their Python types in an object column (a row of mixed dtypes reduces to these) */
const objectResults = (h: Host, vals: Value[]) => vals.map((v) => (v.k === 'float' ? boxFloat(v.v) : cellOf(h, v)))

const numericCols = (f: Frame) => f.data.map((_, k) => k).filter((k) => f.data[k].dtype !== 'object' && !f.data[k].cats)

const frameReduce = (h: Host, f: Frame, how: string, p: Params): Value => {
  const axis = given(p.axis) ? axisOf(h, p.axis, 'DataFrame') : p.axis ? null : 0
  const cols = flag(h, p.numeric_only) ? numericCols(f) : f.data.map((_, k) => k)
  const o = { ...reduceOpts(h, p), frame: true }
  if (axis === null) {
    // sum, prod, std, var, sem still reduce per column with axis=None (deprecated in pandas 2.x); the others over everything
    if (!['sum', 'prod', 'std', 'var', 'sem'].includes(how)) {
      const dt = commonDT(cols.map((k) => f.data[k].dtype))
      const all = new Series(cols.flatMap((k) => (dt === 'object' ? convertCells(f.data[k].values, f.data[k].dtype, dt) : f.data[k].values)), null, null, dt)
      return reduceCol(h, all, how, o)
    }
    return frameReduce(h, f, how, { ...p, axis: py.int(0) })
  }
  if (axis === 1) {
    const vals = rows(f.nrows).map((r) => reduceCol(h, f.row(r, cols), how, o))
    if (how !== 'count' && commonDT(cols.map((k) => f.data[k].dtype)) === 'object') return py.obj(new Series(objectResults(h, vals), f.index, null, 'object'))
    const t = inferValues(h, vals)
    return py.obj(new Series(t.cells, f.index, null, how === 'count' ? 'int64' : t.dt))
  }
  const vals = cols.map((k) => reduceCol(h, f.data[k], how, o))
  const t = inferValues(h, vals)
  const out = new Series(t.cells, f.columns.pick(cols, true), null, how === 'count' ? 'int64' : t.dt)
  const width = sharedWidth(cols.map((k) => f.data[k]))
  if (width && t.dt !== 'object' && (how === 'min' || how === 'max')) out.width = width
  else if (width?.startsWith('u') && (how === 'sum' || how === 'prod')) out.width = 'uint64'
  return py.obj(out)
}

// ---------------------------------------------------------------- column kernels

const cumulate = (h: Host, s: Series, how: string, skipna: boolean): Series => {
  if (s.dtype === 'object') {
    if (how !== 'cumsum' || !s.values.every((c) => typeof c === 'string' || isMissing(c))) throw h.err('TypeError', `can't multiply sequence by non-int of type 'str'`)
    let acc = ''
    return s.with(s.values.map((c) => {
      if (isMissing(c)) return c
      acc += c as string
      return acc
    }), 'object')
  }
  const dt: DT = s.dtype === 'bool' ? (how === 'cummax' || how === 'cummin' ? 'bool' : 'int64') : s.dtype
  let acc: number | null = null
  let dead = false
  const out = s.values.map((c) => {
    const v = numCell(c)
    if (Number.isNaN(v)) {
      if (!skipna) dead = true
      return NaN
    }
    if (dead) return NaN
    if (acc === null) acc = v
    else acc = how === 'cumsum' ? acc + v : how === 'cumprod' ? acc * v : how === 'cummax' ? Math.max(acc, v) : Math.min(acc, v)
    return acc
  })
  const res = dt === 'bool' ? out.map((x) => !!x) : out
  return s.with(res, dt === 'int64' && out.some(Number.isNaN) ? 'float64' : dt)
}

const shiftCol = (h: Host, s: Series, periods: number, fill?: Value): Series => {
  const n = s.length
  const fc: Cell = fill ? cellOf(h, fill) : s.dtype === 'object' && !s.cats ? null : NaN
  const vals = rows(n).map((i) => (i - periods >= 0 && i - periods < n ? s.values[i - periods] : fc))
  if (fill || !periods || !n) {
    const dt = fill ? (s.dtype === 'int64' && typeof fc === 'number' && !Number.isInteger(fc) ? 'float64' : s.dtype === 'object' || typeof fc === 'string' ? 'object' : s.dtype) : s.dtype
    return s.with(vals, dt)
  }
  const dt: DT = s.dtype === 'int64' ? 'float64' : s.dtype === 'bool' ? 'object' : s.dtype
  const out = s.with(vals, dt)
  if (s.cats) out.cats = s.cats
  return out
}

const diffCol = (h: Host, s: Series, periods: number): Series => {
  if (s.dtype === 'object') throw h.err('TypeError', "unsupported operand type(s) for -: 'str' and 'str'")
  const n = s.length
  const vals = s.values.map(numCell)
  const out = rows(n).map((i) => (i - periods >= 0 && i - periods < n ? (s.dtype === 'bool' ? Number(vals[i] !== vals[i - periods]) : vals[i] - vals[i - periods]) : NaN))
  return s.with(out, 'float64')
}

const fillDirection = (s: Series, forward: boolean, limit: number | null): Series => {
  const vals = [...s.values]
  const n = vals.length
  let last: Cell = null
  let have = false
  let run = 0
  for (let k = 0; k < n; k++) {
    const i = forward ? k : n - 1 - k
    if (isMissing(vals[i])) {
      if (have && (limit === null || run < limit)) {
        vals[i] = last
        run++
      }
    } else {
      last = vals[i]
      have = true
      run = 0
    }
  }
  const out = s.with(vals, s.dtype)
  out.cats = s.cats
  return out
}

const pctChange = (h: Host, s: Series, periods: number, fillMethod: string | null, limit: number | null): Series => {
  const base = fillMethod === 'bfill' || fillMethod === 'backfill' ? fillDirection(s, false, limit) : fillMethod ? fillDirection(s, true, limit) : s
  const data = base.values.map(numCell)
  if (s.dtype === 'object' && s.values.some((c) => typeof c === 'string')) throw h.err('TypeError', "unsupported operand type(s) for /: 'str' and 'str'")
  const n = data.length
  return s.with(data.map((v, i) => (i - periods >= 0 && i - periods < n ? v / data[i - periods] - 1 : NaN)), 'float64')
}

/** pandas' rank: returns NaN at missing values with na_option='keep' */
const rankCol = (s: Series, how: string, asc: boolean, naOption: string, pct: boolean): number[] => {
  const n = s.length
  const present = rows(n).filter((i) => !isMissing(s.values[i]))
  const absent = rows(n).filter((i) => isMissing(s.values[i]))
  const order = sortOrder([present.map((i) => s.values[i])], [asc]).map((k) => present[k])
  const out = new Array<number>(n).fill(NaN)
  const offset = naOption === 'top' ? absent.length : 0
  let k = 0
  let dense = 0
  while (k < order.length) {
    let e = k
    while (e + 1 < order.length && s.values[order[e + 1]] === s.values[order[k]]) e++
    dense++
    for (let j = k; j <= e; j++) {
      const i = order[j]
      out[i] = (how === 'average' ? (k + e) / 2 + 1 : how === 'min' ? k + 1 : how === 'max' ? e + 1 : how === 'first' ? j + 1 : dense) + offset
    }
    k = e + 1
  }
  if (naOption !== 'keep') {
    const base = naOption === 'top' ? 0 : order.length
    if (how === 'average') absent.forEach((i) => { out[i] = base + (absent.length + 1) / 2 })
    else if (how === 'min') absent.forEach((i) => { out[i] = base + 1 })
    else if (how === 'max') absent.forEach((i) => { out[i] = base + absent.length })
    else if (how === 'first') absent.forEach((i, j) => { out[i] = base + j + 1 })
    else absent.forEach((i) => { out[i] = (naOption === 'top' ? 1 : dense + 1) })
  }
  if (pct) {
    const total = how === 'dense' ? dense + (naOption !== 'keep' && absent.length ? 1 : 0) : naOption === 'keep' ? present.length : n
    return out.map((x) => x / total)
  }
  return out
}

const roundCell = (v: number, d: number) => {
  if (!Number.isFinite(v)) return v
  const rint = (x: number) => {
    const fl = Math.floor(x)
    const diff = x - fl
    return diff < 0.5 ? fl : diff > 0.5 ? fl + 1 : fl % 2 === 0 ? fl : fl + 1
  }
  if (d >= 0) {
    const f = 10 ** d
    const r = rint(v * f) / f
    return Number.isFinite(r) ? r : v
  }
  const f = 10 ** -d
  return rint(v / f) * f
}

const sameCell = (a: Cell, b: Cell) => {
  if (isMissing(a) || isMissing(b)) return isMissing(a) && isMissing(b)
  if (a instanceof PyCell || b instanceof PyCell) {
    const x = numCell(a)
    const y = numCell(b)
    if (!Number.isNaN(x) && !Number.isNaN(y)) return x === y
    return labelStr(a) === labelStr(b)
  }
  return a === b || ((typeof a === 'number' || typeof a === 'boolean') && (typeof b === 'number' || typeof b === 'boolean') && Number(a) === Number(b))
}

/** pandas keeps the object dtype only if what remains needs it: replace(), fillna() */
const settleObject = (cells: Cell[], dt: DT): { cells: Cell[]; dt: DT } => {
  if (dt !== 'object') return { cells, dt }
  const inferred = inferDT(cells)
  if (inferred === 'object' || !cells.length || cells.every(isMissing)) return { cells, dt }
  return { cells: inferred === 'float64' ? cells.map(numCell) : cells, dt: inferred }
}

const replaceCol = (h: Host, s: Series, pairs: [Cell, Cell, DT][], regex: boolean): Series => {
  let dt = s.dtype
  let changed = false
  const cells: Cell[] = s.values.map((c) => {
    for (const [from, to, tdt] of pairs) {
      const hit = regex && typeof from === 'string' ? typeof c === 'string' && pyRegex(h, from).test(c) : sameCell(c, from)
      if (hit) {
        changed = true
        dt = dt === tdt ? dt : !isMissing(to) || dt !== 'object' ? (isMissing(to) ? (dt === 'int64' ? 'float64' : dt === 'bool' ? 'object' : dt) : commonDT([dt, tdt])) : dt
        if (regex && typeof from === 'string' && typeof to === 'string') return (c as string).replace(pyRegex(h, from, 'g'), to.replace(/\\(\d)/g, '$$$1'))
        return to
      }
    }
    return c
  })
  if (!changed) return s.copy()
  const conv = dt !== s.dtype && dt === 'float64' ? cells.map(numCell) : dt === 'object' && s.dtype === 'float64' ? cells.map((c) => (typeof c === 'number' ? boxFloat(c) : c)) : cells
  const settled = settleObject(conv, dt)
  return s.with(settled.cells, settled.dt)
}

const scalarPair = (h: Host, from: Value, to: Value): [Cell, Cell, DT] => {
  const t = inferValues(h, [to])
  return [cellOf(h, from), t.cells[0], t.dt]
}

const replaceArgs = (h: Host, p: Params, fnName: string): [Cell, Cell, DT][] | null => {
  const a = p.to_replace
  const b = p.value
  // nothing to replace: a copy
  if (!a && !b) return []
  if (!a) throw h.err('ValueError', `${fnName}: to_replace is required`)
  if (a.k === 'dict') return null
  if (isListLike(a)) {
    const from = h.iterate(a)
    if (b && isListLike(b)) {
      const to = h.iterate(b)
      if (from.length !== to.length) throw h.err('ValueError', `Replacement lists must match in length. Expecting ${from.length} got ${to.length} `)
      return from.map((x, i) => scalarPair(h, x, to[i]))
    }
    return from.map((x) => scalarPair(h, x, b ?? py.float(NaN)))
  }
  return [scalarPair(h, a, b ?? py.float(NaN))]
}

/** the condition of where / mask for one column, aligned to it */
const condCells = (h: Host, cond: Value, s: Series, whole: Series | Frame): boolean[] => {
  const c = callableValue(cond) ? h.call(cond, [py.obj(whole)]) : cond
  if (c.k === 'obj' && c.o instanceof Frame) {
    const col = c.o.col(s.name)
    const badDt = c.o.data.find((x) => x.dtype !== 'bool')
    if (badDt) throw h.err('ValueError', `Boolean array expected for the condition, not ${badDt.dtype}`)
    return s.index.labels.map((l) => {
      if (!col) return false
      const at = c.o instanceof Frame ? c.o.index.locate(l) : []
      return at.length ? !!col.values[at[0]] && !isMissing(col.values[at[0]]) : false
    })
  }
  if (c.k === 'obj' && c.o instanceof Series) {
    const m = c.o
    if (m.dtype !== 'bool' && !(m.dtype === 'object' && m.values.every((x) => typeof x === 'boolean' || isMissing(x)))) throw h.err('ValueError', `Boolean array expected for the condition, not ${m.dtype}`)
    return s.index.labels.map((l) => {
      const at = m.index.locate(l)
      return at.length ? !!m.values[at[0]] && !isMissing(m.values[at[0]]) : false
    })
  }
  if (c.k === 'array' && c.a.ndim === 2) {
    const [r, m] = c.a.shape
    const k = (whole as Frame).columns.locate(s.name)[0]
    return Array.from({ length: r }, (_, i) => c.a.values()[i * m + k] !== 0)
  }
  const cells = cellsOf(h, c)
  if (cells.length !== s.length) throw h.err('ValueError', 'Array conditional must be same shape as self')
  return cells.map((x) => !!x)
}

const whereCol = (h: Host, s: Series, keep: boolean[], other: Value | undefined, whole: Series | Frame, axisV?: Value): Series => {
  const out = s.copy()
  const rs = rows(s.length).filter((i) => !keep[i])
  const o = other && callableValue(other) ? h.call(other, [py.obj(whole)]) : other
  if (whole instanceof Frame && o?.k === 'obj' && o.o instanceof Series && !given(axisV)) throw h.err('ValueError', 'Must specify axis=0 or 1')
  if (!rs.length) return out
  let cells: Cell[]
  if (!o) cells = rs.map(() => NaN)
  else if (o.k === 'obj' && o.o instanceof Frame) {
    const src = o.o.col(s.name)
    const f = o.o
    cells = rs.map((r) => {
      const at = f.index.locate(s.index.labels[r])
      return src && at.length ? src.values[at[0]] : NaN
    })
  } else if (o.k === 'obj' && o.o instanceof Series) {
    const src = o.o
    if (whole instanceof Frame && axisOf(h, axisV, 'DataFrame') === 1) {
      const at = src.index.locate(s.name)
      const c = at.length ? src.values[at[0]] : NaN
      cells = rs.map(() => c)
    } else cells = rs.map((r) => {
      const at = src.index.locate(s.index.labels[r])
      return at.length ? src.values[at[0]] : NaN
    })
  } else if (isListLike(o)) {
    const all = cellsOf(h, o)
    if (all.length !== s.length) throw h.err('ValueError', 'Length of replacements must equal series length')
    cells = rs.map((r) => all[r])
  } else {
    const c = cellOf(h, o)
    cells = rs.map(() => c)
  }
  const dt0 = s.dtype
  out.store(rs, cells)
  if (o && o.k === 'float' && dt0 === 'int64' && Number.isInteger(o.v)) out.dtype = 'int64'
  return out
}

const clipCol = (h: Host, s: Series, lower: Value | undefined, upper: Value | undefined): Series => {
  if (s.dtype === 'object') throw h.err('TypeError', "'>=' not supported between instances of 'str' and 'int'")
  const boundAt = (b: Value | undefined): ((i: number) => number) => {
    if (!given(b)) return () => NaN
    if (b!.k === 'obj' && b!.o instanceof Series) {
      const src = b!.o
      return (i) => {
        const at = src.index.locate(s.index.labels[i])
        return at.length ? numCell(src.values[at[0]]) : NaN
      }
    }
    if (isListLike(b!)) {
      const all = cellsOf(h, b!)
      return (i) => numCell(all[i])
    }
    const x = h.num(b!)
    return () => x
  }
  const lo = boundAt(lower)
  const hi = boundAt(upper)
  const rs: number[] = []
  const cells: Cell[] = []
  s.values.forEach((c, i) => {
    const v = numCell(c)
    if (Number.isNaN(v)) return
    if (!Number.isNaN(lo(i)) && v < lo(i)) {
      rs.push(i)
      cells.push(lo(i))
    } else if (!Number.isNaN(hi(i)) && v > hi(i)) {
      rs.push(i)
      cells.push(hi(i))
    }
  })
  const out = s.copy()
  if (rs.length) out.store(rs, cells)
  return out
}

/** the sandbox's numbers for astype */
const WIDTHS: Record<string, { dt: DT; bits?: number; signed?: boolean }> = {
  int: { dt: 'int64' }, int64: { dt: 'int64' }, int32: { dt: 'int64', bits: 32, signed: true }, int16: { dt: 'int64', bits: 16, signed: true }, int8: { dt: 'int64', bits: 8, signed: true },
  uint8: { dt: 'int64', bits: 8, signed: false }, uint16: { dt: 'int64', bits: 16, signed: false }, uint32: { dt: 'int64', bits: 32, signed: false }, uint64: { dt: 'int64' },
  float: { dt: 'float64' }, float64: { dt: 'float64' }, float32: { dt: 'float64', bits: 32 }, float16: { dt: 'float64', bits: 16 },
}

export const typeArg = (h: Host, v: Value): string => {
  if (v.k === 'type') return v.name.replace(/^numpy\./, '')
  if (v.k === 'dtype') return v.d
  if (v.k === 'obj' && v.o instanceof DtypeObj) return v.o.name
  return h.str(v)
}

export const categoricalOf = (s: Series, ordered = false, categories?: Cell[]): Series => {
  if (s.cats && !categories) return s.copy()
  const present = s.values.filter((c) => !isMissing(c))
  const uniq = categories ?? (() => {
    const u = [...new Map(present.map((c) => [c instanceof PyCell ? labelStr(c) : c, c])).values()]
    return sortOrder([u], [true]).map((i) => u[i])
  })()
  const vals = categories ? s.values.map((c) => (uniq.includes(c) ? c : NaN)) : s.values
  const out = new Series([...vals], s.index, s.name, s.dtype === 'bool' ? 'object' : s.dtype)
  const kind = uniq.every((c) => typeof c === 'string') ? 'object' : uniq.every((c) => typeof c === 'number' && Number.isInteger(c)) ? 'int64' : uniq.every((c) => typeof c === 'number') ? 'float64' : 'object'
  out.cats = { labels: uniq, kind, ordered }
  return out
}

export const astypeCol = (h: Host, s: Series, t: string, ignore = false): Series => {
  try {
    if (t === 'category') return categoricalOf(s)
    if (t === 'object' || t === 'O') {
      const out = s.with(s.dtype === 'object' ? [...s.values] : s.dtype === 'float64' ? s.values.map((c) => boxFloat(c as number)) : [...s.values], 'object')
      return out
    }
    if (t === 'str') return s.with(s.values.map((c) => (c === null ? 'None' : typeof c === 'number' && Number.isNaN(c) ? 'nan' : s.dtype === 'float64' ? repr(py.float(c as number)) : c instanceof PyCell ? h.str(c.v) : labelStr(c))), 'object')
    if (t === 'bool') return s.with(s.values.map((c) => (typeof c === 'string' ? c.length > 0 : c === null ? false : typeof c === 'number' ? c !== 0 : c instanceof PyCell ? h.truthy(c.v) : !!c)), 'bool')
    const w = WIDTHS[t]
    if (!w) {
      if (['string', 'Int64', 'Int32', 'Float64', 'boolean'].includes(t)) throw h.err('NotImplementedError', `astype('${t}'): nullable dtypes are not available in this sandbox`)
      throw h.err('TypeError', `data type '${t}' not understood`)
    }
    if (w.dt === 'int64') {
      const vals = s.values.map((c) => {
        if (isMissing(c)) throw h.err(s.dtype === 'float64' ? 'IntCastingNaNError' : 'ValueError', s.dtype === 'float64' ? 'Cannot convert non-finite values (NA or inf) to integer' : 'cannot convert float NaN to integer')
        if (typeof c === 'string') {
          if (!/^\s*[+-]?\d+(_\d+)*\s*$/.test(c)) throw h.err('ValueError', `invalid literal for int() with base 10: '${c}'`)
          return Number(c.replace(/_/g, ''))
        }
        const v = numCell(c)
        if (!Number.isFinite(v)) throw h.err('IntCastingNaNError', 'Cannot convert non-finite values (NA or inf) to integer')
        const tv = Math.trunc(v)
        if (!w.bits) return tv
        const m = 2 ** w.bits
        const u = ((tv % m) + m) % m
        return w.signed && u >= m / 2 ? u - m : u
      })
      const out = s.with(vals, 'int64')
      if (t !== 'int' && t !== 'int64') out.width = t
      return out
    }
    const vals = s.values.map((c) => {
      if (typeof c === 'string') {
        const x = c.trim().toLowerCase()
        const v = x === 'nan' ? NaN : x === 'inf' || x === 'infinity' || x === '+inf' ? Infinity : x === '-inf' || x === '-infinity' ? -Infinity : x === '' ? NaN : Number(x)
        if (Number.isNaN(v) && x !== 'nan') throw h.err('ValueError', `could not convert string to float: '${c}'`)
        return w.bits === 32 ? Math.fround(v) : v
      }
      return w.bits === 32 ? Math.fround(numCell(c)) : numCell(c)
    })
    const out = s.with(vals, 'float64')
    if (t !== 'float' && t !== 'float64') out.width = t
    return out
  } catch (e) {
    if (ignore) return s.copy()
    throw e
  }
}

/** NDArray where the values allow it; a dtype=object array for text */
export const arrayOf = (cells: Cell[], dt: DT, width: string | null = null): Value => {
  if (dt === 'object') return py.obj(new ObjArray([...cells]))
  return py.arr(NDArray.create(cells.map(numCell), [cells.length], dt === 'bool' ? 'bool' : width === 'uint8' ? 'uint8' : dt === 'int64' ? 'int64' : 'float64'))
}

const uniqueCells = (cells: Cell[]) => {
  const seen = new Set<unknown>()
  const out: Cell[] = []
  for (const c of cells) {
    const k = isMissing(c) ? 'NaN\0' : c instanceof PyCell ? (c.v.k === 'float' ? c.v.v : `\0${repr(c.v)}`) : c
    if (!seen.has(k)) {
      seen.add(k)
      out.push(c)
    }
  }
  return out
}

export const valueCountsOf = (s: Series, o: { normalize?: boolean; sort?: boolean; ascending?: boolean; dropna?: boolean } = {}): Series => {
  const dropna = o.dropna ?? true
  const counts = new Map<unknown, { c: Cell; n: number }>()
  for (const c of s.values) {
    if (isMissing(c) && dropna) continue
    const k = isMissing(c) ? 'NaN\0' : c instanceof PyCell ? `\0${repr(c.v)}` : c
    const e = counts.get(k)
    if (e) e.n++
    else counts.set(k, { c, n: 1 })
  }
  if (s.cats) for (const l of s.cats.labels) if (!counts.has(l)) counts.set(l, { c: l, n: 0 })
  let entries = [...counts.values()]
  if (s.cats && (o.sort === false)) {
    const labels = s.cats.labels
    entries = sortOrder([entries.map((e) => labels.indexOf(e.c))], [true]).map((i) => entries[i])
  }
  if (s.cats && o.sort !== false) {
    const labels = s.cats.labels
    entries = sortOrder([entries.map((e) => labels.indexOf(e.c))], [true]).map((i) => entries[i])
  }
  if (o.sort ?? true) {
    const order = sortOrder([entries.map((e) => e.n)], [o.ascending ?? false])
    entries = order.map((i) => entries[i])
  }
  const total = entries.reduce((a, e) => a + e.n, 0)
  const dt = s.dtype === 'bool' && entries.some((e) => isMissing(e.c)) ? 'object' : s.dtype
  const index = new IndexObj(entries.map((e) => e.c), s.name, dt)
  return new Series(entries.map((e) => (o.normalize ? e.n / total : e.n)), index, o.normalize ? 'proportion' : 'count', o.normalize ? 'float64' : 'int64')
}

export const duplicatedMask = (keys: string[], keep: string | false): boolean[] => {
  const n = keys.length
  const dup = new Array<boolean>(n).fill(false)
  if (keep === false) {
    const count = new Map<string, number>()
    keys.forEach((k) => count.set(k, (count.get(k) ?? 0) + 1))
    keys.forEach((k, i) => { dup[i] = count.get(k)! > 1 })
    return dup
  }
  const seen = new Set<string>()
  for (const i of keep === 'last' ? rows(n).reverse() : rows(n)) {
    if (seen.has(keys[i])) dup[i] = true
    else seen.add(keys[i])
  }
  return dup
}

const keepArg = (h: Host, v: Value | undefined): string | false => {
  if (v && v.k === 'bool' && !v.v) return false
  const k = v ? h.str(v) : 'first'
  if (k !== 'first' && k !== 'last') throw h.err('ValueError', 'keep must be either "first", "last" or False')
  return k
}

const cellKey = (c: Cell) => (isMissing(c) ? 'NaN' : c instanceof PyCell ? repr(c.v) : `${typeof c}:${c}`)

/** format_percentiles for describe's row labels */
const formatPercentiles = (ps: number[]): string[] => {
  const pcts = ps.map((p) => 100 * p)
  const precision = (a: number[]) => {
    const diffs: number[] = []
    if (a[0] > 0) diffs.push(a[0])
    for (let i = 1; i < a.length; i++) diffs.push(a[i] - a[i - 1])
    if (a[a.length - 1] < 100) diffs.push(100 - a[a.length - 1])
    return Math.max(1, -Math.floor(Math.log10(Math.min(...diffs.map(Math.abs)))))
  }
  const prec = precision(pcts)
  const f = 10 ** prec
  const rounded = pcts.map((p) => Math.trunc(Math.round(p * f) / f))
  const close = (a: number, b: number) => Math.abs(a - b) <= 1e-8 + 1e-5 * Math.abs(b)
  if (pcts.every((p, i) => close(rounded[i], p))) return rounded.map((r) => `${r}%`)
  const uniq = [...new Set(pcts)].sort((a, b) => a - b)
  const p2 = precision(uniq)
  const f2 = 10 ** p2
  return pcts.map((p, i) => (close(rounded[i], p) ? `${Math.round(p)}%` : `${repr(py.float(Math.round(p * f2) / f2))}%`))
}

const describeNumeric = (h: Host, s: Series, ps: number[]): Series => {
  const vals = s.values.map(numCell)
  const present = vals.filter((v) => !Number.isNaN(v)).sort((a, b) => a - b)
  const stat = (how: string) => numCell(cellOf(h, reduceCol(h, s, how)))
  const labels = ['count', 'mean', 'std', 'min', ...formatPercentiles(ps), 'max']
  const out = [present.length, stat('mean'), stat('std'), present.length ? present[0] : NaN, ...ps.map((q) => npQuantile(present, q)), present.length ? present[present.length - 1] : NaN]
  return new Series(out, new IndexObj(labels, null, 'object'), s.name, 'float64')
}

const describeObject = (s: Series): Series => {
  const counts = valueCountsOf(s)
  const count = s.values.filter((c) => !isMissing(c)).length
  const nonzero = counts.values.filter((n) => (n as number) > 0).length
  const cells: Cell[] = [count, s.cats ? nonzero : counts.length]
  const labels = ['count', 'unique']
  if (counts.length) {
    cells.push(counts.index.labels[0], counts.values[0])
    labels.push('top', 'freq')
  } else {
    cells.push(null, NaN)
    labels.push('top', 'freq')
  }
  return new Series(cells, new IndexObj(labels, null, 'object'), s.name, 'object')
}

const percentilesArg = (h: Host, v: Value | undefined): number[] => {
  if (!given(v)) return [0.25, 0.5, 0.75]
  const ps = cellsOf(h, v!).map(numCell)
  if (ps.some((x) => x < 0 || x > 1)) throw h.err('ValueError', 'percentiles should all be in the interval [0, 1]')
  return [...new Set([...ps, 0.5])].sort((a, b) => a - b)
}

const isNumberCol = (s: Series) => !s.cats && (s.dtype === 'int64' || s.dtype === 'float64')

const describeCol = (h: Host, s: Series, ps: number[]) => (isNumberCol(s) ? describeNumeric(h, s, ps) : describeObject(s))

const dtypeMatches = (s: Series, t: string): boolean => {
  const name = dtName(s)
  switch (t) {
    case 'number': return isNumberCol(s)
    case 'int': case 'int_': return s.dtype === 'int64' && !s.cats && (!s.width || s.width === 'int32')
    case 'integer': return s.dtype === 'int64' && !s.cats
    case 'signedinteger': return s.dtype === 'int64' && !s.cats && !s.width?.startsWith('u')
    case 'unsignedinteger': return s.dtype === 'int64' && !s.cats && !!s.width?.startsWith('u')
    case 'float': case 'float_': return s.dtype === 'float64' && !s.cats && (!s.width || s.width === 'float32')
    case 'floating': return s.dtype === 'float64' && !s.cats
    case 'bool': case 'bool_': return s.dtype === 'bool'
    case 'object': case 'O': case 'str': return s.dtype === 'object' && !s.cats
    case 'category': return !!s.cats
    default: return name === t
  }
}

const describeFrame = (h: Host, f: Frame, p: Params): Frame => {
  const ps = percentilesArg(h, p.percentiles)
  const typeList = (v: Value) => (isListLike(v) ? h.iterate(v) : [v]).map((x) => typeArg(h, x))
  let cols: number[]
  const inc = given(p.include)
  const exc = given(p.exclude)
  if (inc && inc.k === 'str' && inc.v === 'all') {
    if (exc) throw h.err('ValueError', "exclude must be None when include is 'all'")
    cols = f.data.map((_, k) => k)
  } else if (inc || exc) {
    const incs = inc ? typeList(inc) : null
    const excs = exc ? typeList(exc) : []
    cols = f.data.map((_, k) => k).filter((k) => (!incs || incs.some((t) => dtypeMatches(f.data[k], t))) && !excs.some((t) => dtypeMatches(f.data[k], t)))
  } else {
    cols = numericColsStrict(f)
    if (!cols.length) cols = f.data.map((_, k) => k)
  }
  if (!cols.length) throw h.err('ValueError', 'Cannot describe a DataFrame without columns')
  const parts = cols.map((k) => describeCol(h, f.data[k], ps))
  const bySize = [...parts].sort((a, b) => a.length - b.length)
  const names: Cell[] = []
  for (const x of bySize) for (const l of x.index.labels) if (!names.includes(l)) names.push(l)
  const index = new IndexObj(names, null, 'object')
  const out = parts.map((x) => {
    const t = reindexCells(x, index)
    return new Series(t.cells, index, x.name, t.dt)
  })
  return new Frame(f.columns.pick(cols, true), out, index)
}

const numericColsStrict = (f: Frame) => f.data.map((_, k) => k).filter((k) => isNumberCol(f.data[k]))

const quantileArgs = (h: Host, p: Params) => {
  const interp = oneOf(h, p.interpolation, 'linear', ['linear', 'lower', 'higher', 'midpoint', 'nearest'], 'quantile', 'interpolation')
  const qv = p.q ?? py.float(0.5)
  const many = isListLike(qv)
  const qs = many ? cellsOf(h, qv).map(numCell) : [h.num(qv)]
  if (qs.some((q) => q < 0 || q > 1)) throw h.err('ValueError', 'Percentiles must be in the range [0, 100]')
  return { interp, many, qs }
}

const quantileCol = (h: Host, s: Series, qs: number[], interp: string): { vals: number[]; dt: DT } => {
  if (s.dtype === 'object' && s.values.some((c) => typeof c === 'string')) throw h.err('TypeError', "unsupported operand type(s) for -: 'str' and 'str'")
  if (s.dtype === 'bool') throw h.err('TypeError', 'numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.')
  const present = s.values.map(numCell).filter((v) => !Number.isNaN(v)).sort((a, b) => a - b)
  return { vals: qs.map((q) => npQuantile(present, q, interp)), dt: s.dtype === 'int64' && ['lower', 'higher', 'nearest'].includes(interp) ? 'int64' : 'float64' }
}

/** the Welford loop of pandas' nancorr, on rows both columns have */
const welford = (x: number[], y: number[], cov: boolean, minp: number, ddof = 1): number => {
  let n = 0
  let mx = 0
  let my = 0
  let ssx = 0
  let ssy = 0
  let cxy = 0
  for (let i = 0; i < x.length; i++) {
    const vx = x[i]
    const vy = y[i]
    if (Number.isNaN(vx) || Number.isNaN(vy)) continue
    n++
    const dx = vx - mx
    const dy = vy - my
    mx += (1 / n) * dx
    my += (1 / n) * dy
    ssx += (vx - mx) * dx
    ssy += (vy - my) * dy
    cxy += (vx - mx) * dy
  }
  if (n < minp) return NaN
  const div = cov ? n - ddof : Math.sqrt(ssx * ssy)
  if (div === 0) return NaN
  const val = cxy / div
  return cov ? val : Math.max(-1, Math.min(1, val))
}

/** np.cov / np.corrcoef of two columns */
const npCov = (a: number[], b: number[], ddof: number, corr: boolean): number => {
  const n = a.length
  const ma = npSum(a) / n
  const mb = npSum(b) / n
  const x = a.map((v) => v - ma)
  const y = b.map((v) => v - mb)
  const dot = (p: number[], q: number[]) => p.reduce((acc, v, i) => acc + v * q[i], 0)
  const fact = 1 / (n - ddof)
  const cxy = dot(x, y) * fact
  if (!corr) return n - ddof <= 0 ? NaN : cxy
  const sx = Math.sqrt(dot(x, x) * fact)
  const sy = Math.sqrt(dot(y, y) * fact)
  return Math.max(-1, Math.min(1, cxy / sx / sy))
}

/** two Series on their shared labels, both present */
const pairOn = (s: Series, o: Series): [number[], number[]] => {
  const shared = s.index.equals(o.index) ? rows(s.length).map((i) => [i, i]) : rows(s.length).flatMap((i) => o.index.locate(s.index.labels[i]).map((j) => [i, j]))
  const a: number[] = []
  const b: number[] = []
  for (const [i, j] of shared) {
    const x = numCell(s.values[i])
    const y = numCell(o.values[j])
    if (!Number.isNaN(x) && !Number.isNaN(y)) {
      a.push(x)
      b.push(y)
    }
  }
  return [a, b]
}

const rankedPair = (x: number[], y: number[]) => {
  const keep = rows(x.length).filter((i) => !Number.isNaN(x[i]) && !Number.isNaN(y[i]))
  const rank = (a: number[]) => rankCol(new Series(keep.map((i) => a[i]), null, null, 'float64'), 'average', true, 'keep', false)
  return [rank(x), rank(y)]
}

const corrOf = (x: number[], y: number[], method2: string, minp: number) => {
  if (method2 === 'spearman') {
    const [a, b] = rankedPair(x, y)
    return welford(a, b, false, minp)
  }
  return welford(x, y, false, minp)
}

/** a table's numeric columns as numbers, or the error text-holding columns raise */
const floatColumns = (h: Host, f: Frame, numericOnly: boolean) => {
  const cols = numericOnly ? numericCols(f) : f.data.map((_, k) => k)
  for (const k of cols) {
    const t = f.data[k].values.find((c) => typeof c === 'string')
    if (t !== undefined) throw h.err('ValueError', `could not convert string to float: '${t}'`)
  }
  return { cols, vals: cols.map((k) => f.data[k].values.map(numCell)) }
}

export const transposeFrame = (f: Frame): Frame => {
  const dt = f.data.some((s) => s.cats) ? 'object' : commonDT(f.data.map((s) => s.dtype))
  const width = sharedWidth(f.data)
  const data = f.index.labels.map((_, r) => {
    const vals = f.data.map((s) => (dt === 'object' && s.dtype === 'float64' ? boxFloat(s.values[r] as number) : s.values[r]))
    const out = new Series(vals, f.columns, f.index.labels[r], dt)
    out.width = width
    return out
  })
  return new Frame(f.index, data, f.columns)
}

/** a kernel over columns, or over rows through the transpose (axis=1), as pandas itself does */
const byAxis = (f: Frame, axis: 0 | 1, kernel: (s: Series) => Series): Frame => {
  if (axis === 0) return f.same(f.data.map(kernel))
  const t = transposeFrame(f)
  return transposeFrame(t.same(t.data.map(kernel)))
}

const PY_KEYWORDS = ['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']

/** a row of itertuples(): a namedtuple whose bad field names become positions */
const namedRow = (f: Frame, r: number, indexLabel: boolean, tname: string): Value => {
  const names: string[] = []
  const items: Value[] = []
  if (indexLabel) {
    names.push('Index')
    items.push(valueOf(f.index.labels[r], f.index.dt))
  }
  f.data.forEach((s, k) => {
    const raw = labelStr(f.columns.labels[k])
    names.push(/^[A-Za-z][A-Za-z0-9_]*$/.test(raw) && !PY_KEYWORDS.includes(raw) ? raw : `_${names.length}`)
    items.push(valueOf(s.values[r], s.dtype))
  })
  return { k: 'tuple', items, names, tname }
}

/** what a generator-returning pandas method gives back */
const generatorOf = (name: string, items: Value[]): Value => {
  let i = 0
  return { k: 'iter', name, next: () => items[i++] }
}

/** df.to_records() */
class RecArray extends PyObj {
  readonly cls = 'recarray'
  f: Frame
  names: string[]
  rows: Value[][]
  constructor(f: Frame, names: string[], cols: { cells: Cell[]; dt: DT }[]) {
    super()
    this.f = f
    this.names = names
    this.rows = rows(f.nrows).map((r) => cols.map((c) => valueOf(c.cells[r], c.dt)))
    this.dtypes = cols.map((c) => (c.dt === 'int64' ? '<i8' : c.dt === 'float64' ? '<f8' : c.dt === 'bool' ? '?' : 'O'))
    this.cols = cols
  }
  dtypes: string[]
  cols: { cells: Cell[]; dt: DT }[]
  repr() {
    const body = this.rows.map((r) => `(${r.map(repr).join(', ')}${r.length === 1 ? ',' : ''})`).join(', ')
    const dtype = this.names.map((n, i) => `('${n}', '${this.dtypes[i]}')`).join(', ')
    return `rec.array([${body}],\n          dtype=[${dtype}])`
  }
  str() { return `[${this.rows.map((r) => `(${r.map(repr).join(', ')}${r.length === 1 ? ',' : ''})`).join(' ')}]` }
  len() { return this.rows.length }
  iter() { return this.rows.map((r) => py.tuple(r)) }
  getItem(idx: Value, h: Host): Value {
    if (idx.k === 'str') {
      const k = this.names.indexOf(idx.v)
      if (k < 0) throw h.err('ValueError', `no field of name ${idx.v}`)
      return arrayOf(this.cols[k].cells, this.cols[k].dt)
    }
    const i = positions(h, idx, this.rows.length, `index ${repr(idx)} is out of bounds for axis 0 with size ${this.rows.length}`)[0]
    return py.tuple(this.rows[i])
  }
}

/** s.array */
class PandasArray extends PyObj {
  readonly cls = 'NumpyExtensionArray'
  s: Series
  constructor(s: Series) {
    super()
    this.s = s
  }
  repr() {
    const s = this.s
    const n = s.length
    const quote = (c: Cell) => (typeof c === 'string' ? `'${c}'` : c === null ? 'None' : typeof c === 'number' && Number.isNaN(c) ? 'nan' : s.dtype === 'float64' ? repr(py.float(c as number)) : labelStr(c))
    const items = s.values.map(quote)
    const shown = n > 100 ? [...items.slice(0, 10), '...', ...items.slice(-10)] : items
    return `<NumpyExtensionArray>\n[${shown.join(', ')}]\nLength: ${n}, dtype: ${s.dtype}`
  }
  len() { return this.s.length }
  iter() { return this.s.iter() }
  getItem(idx: Value, h: Host): Value { return seriesIndexer(this.s, 'iloc').getItem(idx, h) }
  toArray() { return this.s.toArray() }
  getAttr(name: string): Value | undefined {
    switch (name) {
      case 'shape': return py.tuple([py.int(this.s.length)])
      case 'ndim': return py.int(1)
      case 'size': return py.int(this.s.length)
    }
    return undefined
  }
}

class CatAccessor extends PyObj {
  readonly cls = 'CategoricalAccessor'
  s: Series
  constructor(s: Series) {
    super()
    this.s = s
  }
  repr() { return '<pandas.core.arrays.categorical.CategoricalAccessor object>' }
  getAttr(name: string): Value | undefined {
    const c = this.s.cats!
    switch (name) {
      case 'codes': {
        const out = this.s.with(this.s.values.map((x) => (isMissing(x) ? -1 : c.labels.indexOf(x))), 'int64', null)
        out.width = 'int8'
        return py.obj(out)
      }
      case 'categories':
        if (c.kind.startsWith('interval')) return py.obj(new IntervalIndexObj(c.labels, c.kind))
        return py.obj(new IndexObj(c.labels, null, c.kind === 'int64' ? 'int64' : c.kind === 'float64' ? 'float64' : 'object'))
      case 'ordered': return py.bool(c.ordered)
    }
    return undefined
  }
}

/** the "Categories (n, kind): [...]" line under a categorical Series */
export const categoricalFooter = (c: Cats): string => {
  const width = OPTIONS.width ?? 80
  const interval = c.kind.startsWith('interval')
  const fmtAll = (xs: Cell[]) => (interval ? xs.map(labelStr) : c.kind === 'object' ? xs.map((x) => (typeof x === 'string' ? `'${x}'` : labelStr(x))) : formatArray(xs, c.kind as DT).map((x) => x.trim()))
  const items = c.labels.length > 8 ? [...fmtAll(c.labels.slice(0, 4)), '...', ...fmtAll(c.labels.slice(-4))] : fmtAll(c.labels)
  const head = `Categories (${c.labels.length}, ${c.kind}): `
  const sep = c.ordered ? ' < ' : ', '
  let out = ''
  let col = head.length
  let start = true
  for (const v of items) {
    if (col + sep.length + v.length > width) {
      out += `${sep.trimEnd()}\n${' '.repeat(head.length + 1)}`
      col = head.length + 1
    } else if (!start) {
      out += sep
      col += v.length
    }
    out += v
    start = false
  }
  return `${head}[${out.replace(' < ... < ', ' ... ')}]`
}

// ---------------------------------------------------------------- calling Python functions on columns

const unboxArgs = (h: Host, p: Params): { args: Value[]; kw: Record<string, Value> } => ({
  args: p.__args && p.__args.k === 'tuple' ? p.__args.items : p.args && (p.args.k === 'tuple' || p.args.k === 'list') ? p.args.items : [],
  kw: p.__kw && p.__kw.k === 'dict' ? Object.fromEntries(p.__kw.d.items().map(([k, v]) => [h.str(k), v])) : {},
})

const npFunc = (h: Host, name: string): Value | null => {
  const np = h.module('numpy')
  return np.k === 'module' && np.attrs[name] ? np.attrs[name] : null
}

/** Numpy reductions a column handles itself in agg, with pandas' ddof */
const AGG_ALIASES: Record<string, string> = { sum: 'sum', mean: 'mean', prod: 'prod', std: 'std', var: 'var', median: 'median', max: 'max', min: 'min', amax: 'max', amin: 'min', nanmean: 'mean', nansum: 'sum', nanstd: 'std', nanvar: 'var', nanmax: 'max', nanmin: 'min', all: 'all', any: 'any', cumsum: 'cumsum', cumprod: 'cumprod' }

/** obj.name(): a method by name, or numpy's function of that name */
const applyNamed = (h: Host, obj: Value, name: string, args: Value[] = [], kw: Record<string, Value> = {}): Value => {
  const o = (obj as { o: unknown }).o as Series | Frame
  const attr = o.getAttr(name, h)
  if (attr) return attr.k === 'fn' ? h.call(attr, args, kw) : attr
  const np = npFunc(h, name)
  if (np) return h.call(np, [obj, ...args], kw)
  throw h.err('AttributeError', `'${name}' is not a valid function for '${o.cls}' object`)
}

/** one aggregation of one column */
export const aggColumn = (h: Host, s: Series, f: Value): Value => {
  const self = py.obj(s)
  if (f.k === 'str') return applyNamed(h, self, f.v)
  if (f.k === 'fn' && AGG_ALIASES[f.name]) return applyNamed(h, self, AGG_ALIASES[f.name])
  try {
    return py.obj(seriesApply(h, s, f, [], {}) as Series)
  } catch (e) {
    if (!(e instanceof Error) || !('pyType' in e) || !['ValueError', 'AttributeError', 'TypeError'].includes((e as unknown as { pyType: string }).pyType)) throw e
    return h.call(f, [self])
  }
}

const seriesApply = (h: Host, s: Series, f: Value, args: Value[], kw: Record<string, Value>): Series | Frame => {
  const results = s.iter().map((v) => h.call(f, [v, ...args], kw))
  if (results.length && results.every((r) => r.k === 'obj' && r.o instanceof Series)) {
    const rs = results.map((r) => seriesIn(r))
    let cols = rs[0].index
    for (const r of rs.slice(1)) cols = unionIndex(cols, r.index)
    return new Frame(cols, cols.labels.map((c) => {
      const t = inferValues(h, rs.map((r) => {
        const at = r.index.locate(c)
        return at.length ? valueOf(r.values[at[0]], r.dtype) : py.float(NaN)
      }))
      return new Series(t.cells, s.index, c, t.dt)
    }), s.index)
  }
  const t = inferValues(h, results)
  return new Series(t.cells, s.index, s.name, t.dt)
}

/** results keyed by a list of names → a Series or a table */
const assembleColumn = (h: Host, labels: Cell[], vals: Value[], name: Cell): Series => {
  const t = inferValues(h, vals)
  return new Series(t.cells, new IndexObj(labels, null, inferDT(labels)), name, t.dt)
}

const seriesAgg = (h: Host, s: Series, f: Value): Value => {
  if (f.k === 'dict') return py.obj(assembleColumn(h, f.d.keys().map((k) => cellOf(h, k)), f.d.values().map((x) => seriesAgg(h, s, x)), s.name))
  if (isListLike(f)) {
    const fs = h.iterate(f)
    return py.obj(assembleColumn(h, fs.map((x) => funcName(h, x)), fs.map((x) => aggColumn(h, s, x)), s.name))
  }
  return aggColumn(h, s, f)
}

const frameAgg = (h: Host, fr: Frame, f: Value, axis: 0 | 1, named: Record<string, Value>): Value => {
  if (axis === 1) {
    const t = transposeFrame(fr)
    const res = frameAgg(h, t, f, 0, named)
    return res.k === 'obj' && res.o instanceof Frame ? py.obj(transposeFrame(res.o)) : res
  }
  if (Object.keys(named).length) {
    const labels = Object.keys(named)
    const cols: Cell[] = []
    for (const k of labels) {
      const tup = named[k] as { items: Value[] }
      const c = cellOf(h, tup.items[0])
      if (!cols.includes(c)) cols.push(c)
    }
    const index = new IndexObj(labels, null, 'object')
    return py.obj(new Frame(cols, cols.map((c) => {
      const t = inferValues(h, labels.map((k) => {
        const tup = named[k] as { items: Value[] }
        return cellOf(h, tup.items[0]) === c ? aggColumn(h, fr.byColumnSeries(h, tup.items[0]), tup.items[1]) : py.float(NaN)
      }))
      return new Series(t.cells, index, c, t.dt)
    }), index))
  }
  if (f.k === 'dict') {
    const entries = f.d.items()
    const anyList = entries.some(([, x]) => isListLike(x))
    if (!anyList) return py.obj(assembleColumn(h, entries.map(([k]) => cellOf(h, k)), entries.map(([k, x]) => aggColumn(h, fr.byColumnSeries(h, k), x)), null))
    const parts = entries.map(([k, x]) => {
      const fs = isListLike(x) ? h.iterate(x) : [x]
      return { col: cellOf(h, k), res: assembleColumn(h, fs.map((g) => funcName(h, g)), fs.map((g) => aggColumn(h, fr.byColumnSeries(h, k), g)), cellOf(h, k)) }
    })
    const names: Cell[] = []
    for (const q of parts) for (const l of q.res.index.labels) if (!names.includes(l)) names.push(l)
    const index = new IndexObj(names, null, 'object')
    return py.obj(new Frame(parts.map((q) => q.col), parts.map((q) => {
      const t = reindexCells(q.res, index)
      return new Series(t.cells, index, q.col, t.dt)
    }), index))
  }
  if (isListLike(f)) {
    const fs = h.iterate(f)
    const labels = fs.map((x) => funcName(h, x))
    const index = new IndexObj(labels, null, 'object')
    return py.obj(new Frame(fr.columns, fr.data.map((s) => {
      const t = inferValues(h, fs.map((x) => aggColumn(h, s, x)))
      return new Series(t.cells, index, s.name, t.dt)
    }), index))
  }
  if (f.k === 'str') return applyNamed(h, py.obj(fr), f.v)
  if (f.k === 'fn' && AGG_ALIASES[f.name]) return applyNamed(h, py.obj(fr), AGG_ALIASES[f.name])
  return frameApply(h, fr, f, 0, null, false, [], {})
}

/** DataFrame.apply: scalars make a Series; Series or equal-length lists make a table (pandas' wrap_results) */
const frameApply = (h: Host, fr: Frame, f: Value, axis: 0 | 1, resultType: string | null, raw: boolean, args: Value[], kw: Record<string, Value>): Value => {
  const n = axis === 0 ? fr.ncols : fr.nrows
  const arg = (i: number): Value => {
    const s = axis === 0 ? fr.data[i] : fr.row(i)
    return raw ? arrayOf(s.values, s.dtype) : py.obj(s)
  }
  const results = rows(n).map((i) => h.call(f, [arg(i), ...args], kw))
  const resIndex = axis === 0 ? fr.columns : fr.index
  const other = axis === 0 ? fr.index : fr.columns
  const isSeq = (r: Value) => r.k === 'list' || r.k === 'tuple' || (r.k === 'array' && r.a.ndim > 0) || (r.k === 'obj' && (r.o instanceof Series || r.o instanceof ObjArray))
  const scalarSeries = () => {
    const t = inferValues(h, results)
    return py.obj(new Series(t.cells, resIndex, null, t.dt))
  }
  if (!results.length) return py.obj(fr.copy())
  if (!isSeq(results[0])) {
    if (results[0].k === 'dict') return py.obj(new Series(results.map((r) => cellOf(h, r)), resIndex, null, 'object'))
    return scalarSeries()
  }
  if (resultType === 'reduce') return py.obj(new Series(results.map((r) => cellOf(h, r)), resIndex, null, 'object'))
  const toSeries = (r: Value): Series => {
    if (r.k === 'obj' && r.o instanceof Series) return r.o
    const t = typedCells(h, r)
    return new Series(t.cells, null, null, t.dt)
  }
  const parts = results.map((r) => toSeries(r))
  if (axis === 1 && !(results[0].k === 'obj' && results[0].o instanceof Series) && resultType !== 'expand') return py.obj(new Series(results.map((r) => cellOf(h, r)), fr.index, null, 'object'))
  const lens = new Set(parts.map((x) => x.length))
  if (lens.size > 1 && axis === 0 && !(results[0].k === 'obj')) return py.obj(new Series(results.map((r) => cellOf(h, r)), resIndex, null, 'object'))
  let labels = parts[0].index
  for (const x of parts.slice(1)) labels = unionIndex(labels, x.index)
  const useOther = axis === 0 && !(results[0].k === 'obj') && labels.length === other.length
  const inner = useOther ? other : labels
  const table = (axis === 0)
    ? new Frame(resIndex, parts.map((x, k) => {
      const t = useOther ? { cells: x.values, dt: x.dtype } : reindexCells(x, labels)
      return new Series(t.cells, inner, resIndex.labels[k], t.dt)
    }), inner)
    : new Frame(inner, inner.labels.map((_, j) => {
      const t = inferValues(h, parts.map((x) => {
        const c = useOther ? x.values[j] : (() => {
          const at = x.index.locate(labels.labels[j])
          return at.length ? x.values[at[0]] : NaN
        })()
        return valueOf(c, x.dtype)
      }))
      return new Series(t.cells, fr.index, inner.labels[j], t.dt)
    }), fr.index)
  return py.obj(table)
}

const seriesTransform = (h: Host, s: Series, f: Value, args: Value[], kw: Record<string, Value>): Series | Frame => {
  if (f.k === 'dict' || isListLike(f)) {
    const items: [Cell, Value][] = f.k === 'dict' ? f.d.items().map(([k, v]) => [cellOf(h, k), v]) : h.iterate(f).map((x) => [funcName(h, x), x])
    const parts = items.map(([, g]) => seriesTransform(h, s, g, args, kw) as Series)
    return new Frame(items.map(([l]) => l), parts, s.index)
  }
  let res: Value
  if (f.k === 'str') res = applyNamed(h, py.obj(s), f.v, args, kw)
  else {
    try {
      res = py.obj(seriesApply(h, s, f, args, kw) as Series)
    } catch (e) {
      if (!(e instanceof Error) || !('pyType' in e)) throw e
      res = h.call(f, [py.obj(s), ...args], kw)
    }
  }
  if (res.k === 'array') res = py.obj(new Series(typedCells(h, res).cells, s.index, s.name, typedCells(h, res).dt))
  if (!(res.k === 'obj' && res.o instanceof Series) || res.o.length !== s.length) throw h.err('ValueError', 'Function did not transform')
  return res.o
}

const frameTransform = (h: Host, fr: Frame, f: Value, axis: 0 | 1, args: Value[], kw: Record<string, Value>): Frame => {
  if (axis === 1) return transposeFrame(frameTransform(h, transposeFrame(fr), f, 0, args, kw))
  if (f.k === 'dict') {
    const entries = f.d.items()
    return new Frame(entries.map(([k]) => cellOf(h, k)), entries.map(([k, g]) => seriesTransform(h, fr.byColumnSeries(h, k), g, args, kw) as Series), fr.index)
  }
  if (isListLike(f)) {
    const fs = h.iterate(f)
    const names = fs.map((g) => funcName(h, g))
    const arrays: Cell[][] = [fr.columns.labels.flatMap((c) => names.map(() => c)), fr.columns.labels.flatMap(() => names)]
    const cols = IndexObj.fromArrays(arrays, [null, null], [fr.columns.dt, 'object'])
    return new Frame(cols, fr.data.flatMap((s) => fs.map((g) => seriesTransform(h, s, g, args, kw) as Series)), fr.index)
  }
  if (f.k === 'str') {
    const res = applyNamed(h, py.obj(fr), f.v, args, kw)
    if (res.k === 'obj' && res.o instanceof Frame && res.o.nrows === fr.nrows) return res.o
    if (res.k === 'array' && res.a.ndim === 2) return fr.same(fr.data.map((s, k) => {
      const [r, c] = res.a.shape
      return s.with(Array.from({ length: r }, (_, i) => res.a.values()[i * c + k]), res.a.dtype === 'bool' ? 'bool' : res.a.dtype === 'float64' ? 'float64' : 'int64')
    }))
    throw h.err('ValueError', 'Function did not transform')
  }
  const res = frameApply(h, fr, f, 0, null, false, args, kw)
  if (!(res.k === 'obj' && res.o instanceof Frame) || !res.o.index.equals(fr.index)) throw h.err('ValueError', 'Function did not transform')
  return res.o
}

const mapCol = (h: Host, s: Series, f: Value, ignoreNa: boolean, kw: Record<string, Value> = {}): Series => {
  const results = s.values.map((c): Value => (ignoreNa && isMissing(c) ? valueOf(c, s.dtype) : h.call(f, [valueOf(c, s.dtype)], kw)))
  const t = inferValues(h, results)
  return s.with(t.cells, t.dt)
}

// ---------------------------------------------------------------- query / eval

const queryValue = (h: Host, f: Frame, n: Node): Value => {
  const go = (x: Node) => queryValue(h, f, x)
  const bin = (op: string, l: Value, r: Value): Value => {
    const sides = [l, r]
    for (let i = 0; i < 2; i++) {
      const o = sides[i]
      if (o.k === 'obj' && (o.o instanceof Series || o.o instanceof Frame)) return o.o.binop!(op, sides[1 - i], i === 1, h)!
    }
    if (l.k === 'str' && r.k === 'str' && ['==', '!='].includes(op)) return py.bool((l.v === r.v) === (op === '=='))
    const a = h.num(l)
    const b = h.num(r)
    const out = op === '+' ? a + b : op === '-' ? a - b : op === '*' ? a * b : op === '/' ? a / b : op === '**' ? a ** b : op === '%' ? a % b : op === '//' ? Math.floor(a / b) : NaN
    if (['==', '!=', '<', '<=', '>', '>='].includes(op)) return py.bool(op === '==' ? a === b : op === '!=' ? a !== b : op === '<' ? a < b : op === '<=' ? a <= b : op === '>' ? a > b : a >= b)
    return l.k === 'float' || r.k === 'float' || op === '/' ? py.float(out) : py.int(out)
  }
  const both = (op: '&' | '|', l: Value, r: Value) => (l.k === 'bool' && r.k === 'bool' ? py.bool(op === '&' ? l.v && r.v : l.v || r.v) : bin(op, l, r))
  switch (n.k) {
    case 'num': return n.isFloat ? py.float(n.v) : py.int(n.v)
    case 'str': return py.str(n.v)
    case 'const': return n.v === 'None' ? py.NONE : py.bool(n.v === 'True')
    case 'list': return py.list(n.items.map(go))
    case 'tuple': return py.tuple(n.items.map(go))
    case 'name': {
      if (n.id.startsWith('__at_')) {
        const name = n.id.slice(5)
        const v = h.lookup(name)
        if (!v) throw h.err('UndefinedVariableError', `local variable '${name}' is not defined`)
        return v
      }
      const label = n.id.startsWith('__bt_') ? QUERY_NAMES.get(n.id)! : n.id
      const at = f.columns.locate(label)
      if (at.length) return py.obj(f.data[at[0]])
      if (label === 'index') return py.obj(new Series(f.index.labels, f.index, f.index.name, f.index.dt))
      if (f.index.name === label) return py.obj(new Series(f.index.labels, f.index, f.index.name, f.index.dt))
      if (label === 'True' || label === 'False') return py.bool(label === 'True')
      throw h.err('UndefinedVariableError', `name '${label}' is not defined`)
    }
    case 'unary': {
      const x = go(n.x)
      if (x.k === 'obj' && x.o.unary) return x.o.unary(n.op, h)
      const v = h.num(x)
      return x.k === 'float' ? py.float(n.op === '-' ? -v : v) : py.int(n.op === '-' ? -v : v)
    }
    case 'not': {
      const x = go(n.x)
      if (x.k === 'obj' && x.o instanceof Series) return x.o.unary('~', h)
      return py.bool(!h.truthy(x))
    }
    case 'bin': {
      if (n.op === '&' || n.op === '|') return both(n.op, go(n.l), go(n.r))
      return bin(n.op, go(n.l), go(n.r))
    }
    case 'boolop': return both(n.op === 'and' ? '&' : '|', go(n.l), go(n.r))
    case 'compare': {
      let acc: Value | null = null
      for (let i = 0; i < n.ops.length; i++) {
        const l = go(n.xs[i])
        const r = go(n.xs[i + 1])
        let res: Value
        const op = n.ops[i]
        if (op === 'in' || op === 'not in') {
          const isin = seriesIn(l).getAttr('isin', h)!
          res = h.call(isin, [r])
          if (op === 'not in') res = seriesIn(res).unary('~', h)
        } else if ((op === '==' || op === '!=') && (r.k === 'list' || r.k === 'tuple') && l.k === 'obj' && l.o instanceof Series) {
          res = h.call(l.o.getAttr('isin', h)!, [r])
          if (op === '!=') res = seriesIn(res).unary('~', h)
        } else res = bin(op, l, r)
        acc = acc ? both('&', acc, res) : res
      }
      return acc!
    }
    case 'attr': {
      const o = go(n.obj)
      return h.getAttr(o, n.name)
    }
    case 'call': {
      const callee = go(n.fn)
      const args = n.args.map(go)
      const kw = Object.fromEntries(n.kw.map((k) => [k.name, go(k.v)]))
      return h.call(callee, args, kw)
    }
    case 'sub': {
      const o = go(n.obj)
      if (o.k === 'obj' && o.o.getItem) return o.o.getItem(go(n.idx), h)
      break
    }
  }
  throw h.err('NotImplementedError', `query/eval: this expression is not available in this sandbox`)
}

const QUERY_NAMES = new Map<string, string>()

const parseQuery = (h: Host, expr: string): Node => {
  // outside quotes: `name` becomes an identifier, @var a marked one, and & | bind like and / or
  const src = expr.replace(/(['"])(?:\\.|(?!\1).)*\1|`([^`]+)`|@([A-Za-z_][A-Za-z0-9_]*)|[&|]/g, (m, _q, name?: string, at?: string) => {
    if (name) {
      const id = `__bt_${QUERY_NAMES.size}`
      QUERY_NAMES.set(id, name)
      return id
    }
    if (at) return `__at_${at}`
    return m === '&' ? ' and ' : m === '|' ? ' or ' : m
  })
  let stmts
  try {
    stmts = parse(src.trim())
  } catch {
    throw h.err('SyntaxError', 'invalid syntax')
  }
  const st = stmts[0]
  if (stmts.length !== 1 || st.k !== 'expr') throw h.err('NotImplementedError', 'query/eval take one expression in this sandbox')
  return st.x
}

// ---------------------------------------------------------------- flex arithmetic

const FLEX: Record<string, [string, boolean]> = {
  add: ['+', false], radd: ['+', true], sub: ['-', false], subtract: ['-', false], rsub: ['-', true], mul: ['*', false], multiply: ['*', false], rmul: ['*', true],
  div: ['/', false], divide: ['/', false], truediv: ['/', false], rdiv: ['/', true], rtruediv: ['/', true], floordiv: ['//', false], rfloordiv: ['//', true],
  mod: ['%', false], rmod: ['%', true], pow: ['**', false], rpow: ['**', true],
}
const FLEX_CMP: Record<string, string> = { eq: '==', ne: '!=', gt: '>', ge: '>=', lt: '<', le: '<=' }

// ---------------------------------------------------------------- Series methods

const headTail = <S extends Series | Frame>(self: S, which: 'head' | 'tail', nv: Value | undefined, h: Host, ctx: Ctx): Value => {
  const n = nv ? h.toInt(nv) : 5
  const total = self instanceof Frame ? self.nrows : (self as Series).length
  const all = rows(total)
  const pick = which === 'head' ? (n >= 0 ? all.slice(0, n) : all.slice(0, Math.max(0, total + n))) : n >= 0 ? (n === 0 ? [] : all.slice(-n)) : all.slice(-n)
  ctx.picks = { rows: pick, cols: null }
  return py.obj(self instanceof Frame ? self.take(pick, null, true) : (self as Series).pick(pick, true))
}

const sampleRows = (h: Host, total: number, p: Params): number[] => {
  const nv = given(p.n)
  const fv = given(p.frac)
  if (nv && fv) throw h.err('ValueError', 'Please enter a value for `frac` OR `n`, not both')
  const replace = flag(h, p.replace)
  const n = fv ? Math.round(h.num(fv) * total) : nv ? h.toInt(nv) : 1
  if (fv && h.num(fv) > 1 && !replace) throw h.err('ValueError', 'Replace has to be set to `True` when upsampling the population `frac` > 1.')
  if (n < 0) throw h.err('ValueError', 'A negative number of rows requested. Please provide `n` >= 0.')
  if (n > total && !replace) throw h.err('ValueError', "Cannot take a larger sample than population when 'replace=False'")
  if (given(p.random_state)) h.rng.seed(h.toInt(p.random_state!))
  const w = given(p.weights) ? cellsOf(h, p.weights!).map(numCell) : null
  noteOnce(h, 'pandas_sample')
  if (w) {
    const pool = rows(total)
    const out: number[] = []
    const weights = [...w]
    for (let k = 0; k < n; k++) {
      const sum = weights.reduce((a, b) => a + b, 0)
      let x = h.rng.next() * sum
      let i = 0
      while (i < pool.length - 1 && x >= weights[i]) x -= weights[i++]
      out.push(pool[i])
      if (!replace) {
        pool.splice(i, 1)
        weights.splice(i, 1)
      }
    }
    return out
  }
  if (replace) return Array.from({ length: n }, () => Math.floor(h.rng.next() * total))
  const order = rows(total)
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(h.rng.next() * (i + 1))
    ;[order[i], order[j]] = [order[j], order[i]]
  }
  return order.slice(0, n)
}

const reindexSeries = (h: Host, s: Series, target: IndexObj, p: Params): Series => {
  const meth = given(p.method) ? h.str(p.method!) : null
  const fillv = given(p.fill_value)
  const fill = fillv ? cellOf(h, fillv) : NaN
  refuse(h, p.limit, 'reindex', 'limit')
  refuse(h, p.tolerance, 'reindex', 'tolerance')
  if (!s.index.isUnique) throw h.err('ValueError', 'cannot reindex on an axis with duplicate labels')
  if (meth) {
    const labels = s.index.labels
    const asc = labels.every((l, i) => i === 0 || (l as number) >= (labels[i - 1] as number))
    const desc = labels.every((l, i) => i === 0 || (l as number) <= (labels[i - 1] as number))
    if (!asc && !desc) throw h.err('ValueError', 'index must be monotonic increasing or decreasing')
    const nums = labels.map(numCell)
    const pos = (x: Cell): number => {
      const v = numCell(x)
      const exact = s.index.locate(x)
      if (exact.length) return exact[0]
      if (meth === 'ffill' || meth === 'pad') {
        let j = -1
        nums.forEach((a, i) => { if (a <= v) j = i })
        return j
      }
      if (meth === 'bfill' || meth === 'backfill') return nums.findIndex((a) => a >= v)
      if (meth === 'nearest') {
        let best = -1
        nums.forEach((a, i) => { if (best < 0 || Math.abs(a - v) < Math.abs(nums[best] - v)) best = i })
        return best
      }
      throw h.err('ValueError', `Invalid fill method. Expecting pad (ffill), backfill (bfill) or nearest. Got ${meth}`)
    }
    const at = target.labels.map(pos)
    const cells = at.map((i) => (i < 0 ? fill : s.values[i]))
    const anyMissing = at.some((i) => i < 0) && !fillv
    return new Series(cells, target, s.name, anyMissing && s.dtype === 'int64' ? 'float64' : anyMissing && s.dtype === 'bool' ? 'object' : s.dtype)
  }
  const t = reindexCells(s, target, fill)
  return new Series(t.cells === s.values ? [...t.cells] : t.cells, target, s.name, t.dt)
}

const toIndexObj = (h: Host, v: Value): IndexObj => {
  if (v.k === 'obj' && v.o instanceof IndexObj) return v.o
  const t = typedCells(h, v)
  return new IndexObj(t.cells, null, t.dt)
}

/** a plain list in a set operation does not disturb the Index's name */
const sameNamed = (h: Host, ix: IndexObj, v: Value): IndexObj => (v.k === 'obj' && v.o instanceof IndexObj ? v.o : toIndexObj(h, v).withName(ix.name))

const renameLabels = (h: Host, ix: IndexObj, m: Value): IndexObj => {
  const conv = (l: Cell): Cell => {
    if (m.k === 'dict') {
      const v = m.d.get(valueOf(l, ix.dt))
      return v ? cellOf(h, v) : l
    }
    if (m.k === 'obj' && m.o instanceof Series) {
      const at = m.o.index.locate(l)
      return at.length ? m.o.values[at[0]] : l
    }
    return cellOf(h, h.call(m, [valueOf(l, ix.dt)]))
  }
  if (ix.multi) {
    const mi = ix.multi
    const arrays = mi.arrays.map((a) => a.map(conv))
    return IndexObj.fromArrays(arrays, mi.names, arrays.map((a) => inferDT(a)))
  }
  const labels = ix.labels.map(conv)
  const dt = inferDT(labels)
  return new IndexObj(labels, ix.name, dt === 'int64' && ix.dt === 'float64' ? 'float64' : dt)
}

const SERIES: Record<string, Method<Series>> = {
  tolist: ['', (s) => py.list(s.iter())],
  to_list: ['', (s) => py.list(s.iter())],
  to_numpy: ['dtype copy na_value', (s, p, h) => {
    if (given(p.dtype)) return arrayOf(astypeCol(h, s, typeArg(h, p.dtype!)).values, astypeCol(h, s, typeArg(h, p.dtype!)).dtype)
    if (given(p.na_value)) {
      const v = cellOf(h, p.na_value!)
      return arrayOf(s.values.map((c) => (isMissing(c) ? v : c)), s.dtype === 'float64' ? 'float64' : inferDT(s.values.map((c) => (isMissing(c) ? v : c))))
    }
    return arrayOf(s.values, s.dtype, s.width)
  }],
  ravel: ['order', (s) => arrayOf(s.values, s.dtype, s.width)],
  head: ['n', (s, p, h, c) => headTail(s, 'head', p.n, h, c)],
  tail: ['n', (s, p, h, c) => headTail(s, 'tail', p.n, h, c)],
  take: ['indices axis', (s, p, h) => py.obj(s.pick(positions(h, p.indices!, s.length, 'indices are out-of-bounds')))],
  copy: ['deep', (s, p, h) => {
    if (p.deep && !h.truthy(p.deep)) {
      const out = new Series(s.values, s.index, s.name, s.dtype)
      out.values = s.values
      return py.obj(out)
    }
    return py.obj(s.copy())
  }],
  sample: ['n frac replace weights random_state axis ignore_index', (s, p, h, c) => {
    const pick = sampleRows(h, s.length, p)
    c.picks = { rows: pick, cols: null }
    const out = s.pick(pick)
    if (flag(h, p.ignore_index)) out.index = IndexObj.range(pick.length)
    return py.obj(out)
  }],
  count: ['', (s, _p, h) => reduceCol(h, s, 'count')],
  describe: ['percentiles include exclude', (s, p, h) => py.obj(describeCol(h, s, percentilesArg(h, p.percentiles)))],
  quantile: ['q interpolation', (s, p, h) => {
    const { interp, many, qs } = quantileArgs(h, p)
    const { vals, dt } = quantileCol(h, s, qs, interp)
    if (many) return py.obj(new Series(vals, new IndexObj(qs, null, 'float64'), s.name, dt))
    return dt === 'int64' ? py.int(vals[0]) : py.float(vals[0])
  }],
  all: ['axis bool_only skipna', (s, p, h) => {
    if (flag(h, p.bool_only)) throw h.err('ValueError', 'Series.all does not allow bool_only=True.')
    return py.bool(anyAll(h, s, 'all', flag(h, p.skipna, true)))
  }],
  any: ['axis bool_only skipna', (s, p, h) => {
    if (flag(h, p.bool_only)) throw h.err('ValueError', 'Series.any does not allow bool_only=True.')
    return py.bool(anyAll(h, s, 'any', flag(h, p.skipna, true)))
  }],
  idxmax: ['axis skipna', (s, p, h) => idxOf(h, s, 'max', flag(h, p.skipna, true))],
  idxmin: ['axis skipna', (s, p, h) => idxOf(h, s, 'min', flag(h, p.skipna, true))],
  argmax: ['axis skipna', (s, p, h) => py.int(argOf(h, s, 'max', flag(h, p.skipna, true)))],
  argmin: ['axis skipna', (s, p, h) => py.int(argOf(h, s, 'min', flag(h, p.skipna, true)))],
  argsort: ['axis kind order', (s) => {
    const present = rows(s.length).filter((i) => !isMissing(s.values[i]))
    const order = sortOrder([present.map((i) => s.values[i])], [true])
    const out = new Array<number>(s.length).fill(-1)
    present.forEach((i, k) => { out[i] = order[k] })
    return py.obj(s.with(out, 'int64'))
  }],
  value_counts: ['normalize sort ascending bins dropna', (s, p, h, c) => {
    const opts = { normalize: flag(h, p.normalize), sort: flag(h, p.sort, true), ascending: flag(h, p.ascending), dropna: flag(h, p.dropna, true) }
    c.api = 'value_counts'
    if (given(p.bins)) {
      const cut = (h.getAttr(h.module('pandas'), 'cut'))
      const out = valueCountsOf(seriesIn(h.call(cut, [py.obj(s), p.bins!], { include_lowest: py.bool(true) })), opts)
      out.index = out.index.withName(null)
      return py.obj(out)
    }
    return py.obj(valueCountsOf(s, opts))
  }],
  unique: ['', (s) => arrayOf(uniqueCells(s.values), s.dtype, s.width)],
  nunique: ['dropna', (s, p, h) => py.int(uniqueCells(flag(h, p.dropna, true) ? s.values.filter((c) => !isMissing(c)) : s.values).length)],
  mode: ['dropna', (s, p, h) => py.obj(modeCol(s, flag(h, p.dropna, true)))],
  factorize: ['sort use_na_sentinel', (s, p, h) => factorize(s.values, s.dtype, flag(h, p.sort), flag(h, p.use_na_sentinel, true))],
  isna: ['', (s) => py.obj(s.with(s.values.map(isMissing), 'bool'))],
  isnull: ['', (s) => py.obj(s.with(s.values.map(isMissing), 'bool'))],
  notna: ['', (s) => py.obj(s.with(s.values.map((c) => !isMissing(c)), 'bool'))],
  notnull: ['', (s) => py.obj(s.with(s.values.map((c) => !isMissing(c)), 'bool'))],
  fillna: ['value | method axis inplace limit', (s, p, h, c) => {
    const missing = rows(s.length).filter((i) => isMissing(s.values[i]))
    c.picks = { rows: missing, cols: null }
    return finish(h, s, fillCol(h, s, p), p.inplace)
  }],
  ffill: ['axis inplace limit limit_area', (s, p, h) => {
    refuse(h, p.limit_area, 'ffill', 'limit_area')
    return finish(h, s, fillDirection(s, true, given(p.limit) ? h.toInt(p.limit!) : null), p.inplace)
  }],
  pad: ['axis inplace limit', (s, p, h) => finish(h, s, fillDirection(s, true, given(p.limit) ? h.toInt(p.limit!) : null), p.inplace)],
  bfill: ['axis inplace limit limit_area', (s, p, h) => {
    refuse(h, p.limit_area, 'bfill', 'limit_area')
    return finish(h, s, fillDirection(s, false, given(p.limit) ? h.toInt(p.limit!) : null), p.inplace)
  }],
  backfill: ['axis inplace limit', (s, p, h) => finish(h, s, fillDirection(s, false, given(p.limit) ? h.toInt(p.limit!) : null), p.inplace)],
  dropna: ['| axis inplace how ignore_index', (s, p, h, c) => {
    const keep = rows(s.length).filter((i) => !isMissing(s.values[i]))
    c.picks = { rows: keep, cols: null }
    const out = s.pick(keep)
    if (flag(h, p.ignore_index)) out.index = IndexObj.range(keep.length)
    return finish(h, s, out, p.inplace)
  }],
  interpolate: ['method axis limit inplace limit_direction limit_area downcast', (s, p, h) => finish(h, s, interpolateCol(h, s, p), p.inplace)],
  replace: ['to_replace value inplace limit regex method', (s, p, h) => {
    refuse(h, p.limit, 'replace', 'limit')
    refuse(h, p.method, 'replace', 'method')
    const a = p.to_replace
    const pairs = a && a.k === 'dict' ? a.d.items().map(([k, v]) => scalarPair(h, k, v)) : replaceArgs(h, p, 'replace')!
    return finish(h, s, replaceCol(h, s, pairs, flag(h, p.regex)), p.inplace)
  }],
  where: ['cond other inplace axis level', (s, p, h) => {
    refuse(h, p.level, 'where', 'level')
    return finish(h, s, whereCol(h, s, condCells(h, need(h, p.cond, 'where', 'cond'), s, s), p.other, s), p.inplace)
  }],
  mask: ['cond other inplace axis level', (s, p, h) => {
    refuse(h, p.level, 'mask', 'level')
    return finish(h, s, whereCol(h, s, condCells(h, need(h, p.cond, 'mask', 'cond'), s, s).map((x) => !x), p.other, s), p.inplace)
  }],
  clip: ['lower upper axis inplace', (s, p, h) => {
    const [lo, up] = swapBounds(h, p)
    return finish(h, s, clipCol(h, s, lo, up), p.inplace)
  }],
  abs: ['', (s, _p, h) => py.obj(s.with(s.numeric(h, 'abs').map(Math.abs), s.dtype === 'bool' ? 'bool' : s.dtype))],
  round: ['decimals', (s, p, h) => {
    if (s.dtype === 'object') throw h.err('TypeError', 'Expected numeric dtype, got object instead.')
    const d = p.decimals ? h.toInt(p.decimals) : 0
    if (s.dtype === 'bool' || (s.dtype === 'int64' && d >= 0)) return py.obj(s.copy())
    return py.obj(s.with(s.values.map((c) => roundCell(c as number, d)), s.dtype))
  }],
  between: ['left right inclusive', (s, p, h) => {
    const inc = oneOf(h, p.inclusive, 'both', ['both', 'neither', 'left', 'right'], 'between', 'inclusive')
    const lo = seriesOp(h, s, inc === 'both' || inc === 'left' ? '>=' : '>', need(h, p.left, 'between', 'left'), false, undefined, true)
    const hi = seriesOp(h, s, inc === 'both' || inc === 'right' ? '<=' : '<', need(h, p.right, 'between', 'right'), false, undefined, true)
    return py.obj(s.with(lo.values.map((x, i) => !!x && !!hi.values[i]), 'bool'))
  }],
  isin: ['values', (s, p, h, c) => {
    const out = isinCol(h, s, need(h, p.values, 'isin', 'values'))
    c.picks = { rows: out.values.flatMap((b, i) => (b ? [i] : [])), cols: null }
    return py.obj(out)
  }],
  map: ['arg na_action', (s, p, h) => {
    const m = need(h, p.arg, 'map', 'arg')
    const ignore = given(p.na_action) ? h.str(p.na_action!) === 'ignore' : false
    if (m.k === 'dict') {
      const t = inferValues(h, s.values.map((c): Value => (isMissing(c) && ignore ? py.float(NaN) : m.d.get(valueOf(c, s.dtype)) ?? py.float(NaN))))
      return py.obj(s.with(t.cells, t.dt))
    }
    if (m.k === 'obj' && m.o instanceof Series) {
      const src = m.o
      const cells = s.values.map((c) => {
        const at = src.index.locate(c)
        return at.length ? src.values[at[0]] : NaN
      })
      return py.obj(s.with(cells, cells.some(isMissing) && src.dtype === 'int64' ? 'float64' : src.dtype))
    }
    return py.obj(mapCol(h, s, m, ignore))
  }],
  apply: ['func convert_dtype args by_row **', (s, p, h) => {
    const f = need(h, p.func, 'apply', 'func')
    if (p.by_row && p.by_row.k !== 'str') refuse(h, p.by_row, 'apply', 'by_row')
    const { args, kw } = unboxArgs(h, p)
    if (f.k === 'str' || f.k === 'dict' || isListLike(f)) return f.k === 'str' ? applyNamed(h, py.obj(s), f.v, args, kw) : seriesAgg(h, s, f)
    return py.obj(seriesApply(h, s, f, args, kw))
  }],
  agg: ['func axis', (s, p, h) => seriesAgg(h, s, need(h, p.func, 'agg', 'func'))],
  aggregate: ['func axis', (s, p, h) => seriesAgg(h, s, need(h, p.func, 'aggregate', 'func'))],
  transform: ['func axis * **', (s, p, h) => {
    const { args, kw } = unboxArgs(h, p)
    return py.obj(seriesTransform(h, s, need(h, p.func, 'transform', 'func'), args, kw))
  }],
  pipe: ['func * **', (s, p, h) => pipeCall(h, py.obj(s), p)],
  astype: ['dtype copy errors', (s, p, h) => {
    const d = need(h, p.dtype, 'astype', 'dtype')
    return py.obj(astypeCol(h, s, typeArg(h, d), given(p.errors) ? h.str(p.errors!) === 'ignore' : false))
  }],
  infer_objects: ['copy', (s) => {
    const t = settleObject(s.values, s.dtype)
    return py.obj(s.with(t.cells, t.dt))
  }],
  cumsum: ['axis skipna', (s, p, h) => py.obj(cumulate(h, s, 'cumsum', flag(h, p.skipna, true)))],
  cumprod: ['axis skipna', (s, p, h) => py.obj(cumulate(h, s, 'cumprod', flag(h, p.skipna, true)))],
  cummax: ['axis skipna', (s, p, h) => py.obj(cumulate(h, s, 'cummax', flag(h, p.skipna, true)))],
  cummin: ['axis skipna', (s, p, h) => py.obj(cumulate(h, s, 'cummin', flag(h, p.skipna, true)))],
  diff: ['periods', (s, p, h) => py.obj(diffCol(h, s, p.periods ? h.toInt(p.periods) : 1))],
  shift: ['periods freq axis fill_value', (s, p, h) => {
    refuse(h, p.freq, 'shift', 'freq')
    return py.obj(shiftCol(h, s, p.periods ? h.toInt(p.periods) : 1, given(p.fill_value)))
  }],
  pct_change: ['periods fill_method limit freq', (s, p, h) => {
    refuse(h, p.freq, 'pct_change', 'freq')
    return py.obj(pctChange(h, s, p.periods ? h.toInt(p.periods) : 1, fillMethodArg(h, p), given(p.limit) ? h.toInt(p.limit!) : null))
  }],
  rank: ['axis method numeric_only na_option ascending pct', (s, p, h) => {
    const how = oneOf(h, p.method, 'average', ['average', 'min', 'max', 'first', 'dense'], 'rank', 'method')
    const na = oneOf(h, p.na_option, 'keep', ['keep', 'top', 'bottom'], 'rank', 'na_option')
    return py.obj(s.with(rankCol(s, how, flag(h, p.ascending, true), na, flag(h, p.pct)), 'float64'))
  }],
  sort_values: ['| axis ascending inplace kind na_position ignore_index key', (s, p, h) => {
    const key = given(p.key) ? seriesIn(h.call(p.key!, [py.obj(s)])).values : s.values
    const order = sortOrder([key], [flag(h, p.ascending, true)], given(p.na_position) ? h.str(p.na_position!) === 'first' : false)
    const out = s.pick(order)
    if (flag(h, p.ignore_index)) out.index = IndexObj.range(order.length)
    return finish(h, s, out, p.inplace)
  }],
  sort_index: ['| axis level ascending inplace kind na_position sort_remaining ignore_index key', (s, p, h) => {
    refuse(h, p.level, 'sort_index', 'level')
    refuse(h, p.key, 'sort_index', 'key')
    const order = sortOrder([s.index.labels], [flag(h, p.ascending, true)], given(p.na_position) ? h.str(p.na_position!) === 'first' : false)
    const out = s.pick(order)
    if (flag(h, p.ignore_index)) out.index = IndexObj.range(order.length)
    return finish(h, s, out, p.inplace)
  }],
  nlargest: ['n keep', (s, p, h, c) => nExtreme(h, s, p, 'nlargest', c)],
  nsmallest: ['n keep', (s, p, h, c) => nExtreme(h, s, p, 'nsmallest', c)],
  duplicated: ['keep', (s, p, h) => py.obj(s.with(duplicatedMask(s.values.map(cellKey), keepArg(h, p.keep)), 'bool'))],
  drop_duplicates: ['| keep inplace ignore_index', (s, p, h, c) => {
    const dup = duplicatedMask(s.values.map(cellKey), keepArg(h, p.keep))
    c.picks = { rows: rows(s.length).filter((i) => dup[i]), cols: null }
    const out = s.pick(rows(s.length).filter((i) => !dup[i]))
    if (flag(h, p.ignore_index)) out.index = IndexObj.range(out.length)
    return finish(h, s, out, p.inplace)
  }],
  drop: ['labels | axis index columns level inplace errors', (s, p, h) => {
    refuse(h, p.level, 'drop', 'level')
    refuse(h, p.columns, 'drop', 'columns')
    const tv = p.index ?? p.labels
    if (!tv) throw h.err('ValueError', "Need to specify at least one of 'labels', 'index' or 'columns'")
    const labels = isListLike(tv) ? cellsOf(h, tv) : [cellOf(h, tv)]
    const ignore = given(p.errors) && h.str(p.errors!) === 'ignore'
    const missing = labels.filter((l) => !s.index.has(l))
    if (missing.length && !ignore) throw h.err('KeyError', repr(py.str(`[${missing.map(reprCell).join(', ')}] not found in axis`)))
    const dropping = new Set(labels.flatMap((l) => s.index.locate(l)))
    return finish(h, s, s.pick(rows(s.length).filter((i) => !dropping.has(i))), p.inplace)
  }],
  rename: ['index | axis copy inplace level errors', (s, p, h) => {
    refuse(h, p.level, 'rename', 'level')
    const m = p.index
    const out = s.copy()
    if (m === undefined || m.k === 'none') out.name = null
    else if (m.k === 'dict' || callableValue(m) || (m.k === 'obj' && m.o instanceof Series)) out.index = renameLabels(h, s.index, m)
    else out.name = cellOf(h, m)
    return finish(h, s, out, p.inplace)
  }],
  rename_axis: ['mapper index axis copy inplace', (s, p, h) => {
    const m = p.mapper ?? p.index
    const out = s.copy()
    out.index = s.index.withName(given(m) ? cellOf(h, m!) : null)
    return finish(h, s, out, p.inplace)
  }],
  set_axis: ['labels axis copy', (s, p, h) => {
    const ix = toIndexObj(h, need(h, p.labels, 'set_axis', 'labels'))
    if (ix.length !== s.length) throw h.err('ValueError', `Length mismatch: Expected axis has ${s.length} elements, new values have ${ix.length} elements`)
    const out = s.copy()
    out.index = ix
    return py.obj(out)
  }],
  add_prefix: ['prefix axis', (s, p, h) => py.obj(prefixed(s, h.str(p.prefix!), ''))],
  add_suffix: ['suffix axis', (s, p, h) => py.obj(prefixed(s, '', h.str(p.suffix!)))],
  reindex: ['index method copy level fill_value limit tolerance', (s, p, h) => {
    refuse(h, p.level, 'reindex', 'level')
    return py.obj(reindexSeries(h, s, toIndexObj(h, need(h, p.index, 'reindex', 'index')), p))
  }],
  reset_index: ['level | drop name inplace allow_duplicates', (s, p, h) => {
    refuse(h, p.level, 'reset_index', 'level')
    if (flag(h, p.drop)) {
      const out = s.copy()
      out.index = IndexObj.range(s.length)
      if (given(p.name)) out.name = cellOf(h, p.name!)
      return finish(h, s, out, p.inplace)
    }
    if (flag(h, p.inplace)) throw h.err('TypeError', 'Cannot reset_index inplace on a Series to create a DataFrame')
    const name = given(p.name) ? cellOf(h, p.name!) : s.name ?? 0
    return py.obj(indexToColumns(h, new Frame([name], [s], s.index)))
  }],
  to_frame: ['name', (s, p, h) => py.obj(new Frame([given(p.name) ? cellOf(h, p.name!) : s.name ?? 0], [s.copy()], s.index))],
  to_dict: ['into', (s, p, h) => (refuseInto(h, p), { k: 'dict', d: dictFromPairs(s.index.iter().map((k, i) => [k, valueOf(s.values[i], s.dtype)])) })],
  to_string: ['buf na_rep float_format header index length dtype name max_rows min_rows', (s, p, h) => {
    refuse(h, p.buf, 'to_string', 'buf')
    return py.str(seriesToString(s, { maxRows: given(p.max_rows) ? h.toInt(p.max_rows!) : null, minRows: given(p.min_rows) ? h.toInt(p.min_rows!) : null, index: flag(h, p.index, true), header: flag(h, p.header, true), name: flag(h, p.name), dtype: flag(h, p.dtype), length: flag(h, p.length), naRep: given(p.na_rep) ? h.str(p.na_rep!) : undefined, floatFormat: floatFormatArg(h, p.float_format) }))
  }],
  to_csv: ['path_or_buf sep na_rep float_format columns header index index_label mode encoding compression quoting quotechar lineterminator', (s, p, h) => seriesToCsv(h, s, p)],
  to_json: ['path_or_buf orient date_format double_precision force_ascii date_unit default_handler lines compression index indent', (s, p, h) => seriesToJson(h, s, p)],
  items: ['', (s) => generatorOf('zip', s.index.iter().map((k, i) => py.tuple([k, valueOf(s.values[i], s.dtype)])))],
  keys: ['', (s) => py.obj(s.index)],
  get: ['key default', (s, p, h) => seriesGet(h, s, p)],
  item: ['', (s, _p, h) => {
    if (s.length !== 1) throw h.err('ValueError', 'can only convert an array of size 1 to a Python scalar')
    return valueOf(s.values[0], s.dtype)
  }],
  pop: ['item', (s, p, h) => {
    const c = cellOf(h, need(h, p.item, 'pop', 'item'))
    const at = s.index.locate(c)
    if (!at.length) throw h.err('KeyError', repr(p.item!))
    const v = valueOf(s.values[at[0]], s.dtype)
    s.adopt(s.pick(rows(s.length).filter((i) => !at.includes(i))))
    return v
  }],
  repeat: ['repeats axis', (s, p, h) => {
    const r = need(h, p.repeats, 'repeat', 'repeats')
    const given0 = isListLike(r) ? cellsOf(h, r).map(Number) : null
    const times = given0 && given0.length !== 1 ? given0 : rows(s.length).map(() => (given0 ? given0[0] : h.toInt(r)))
    if (times.length !== s.length) throw h.err('ValueError', `operands could not be broadcast together with shape (${s.length},) (${times.length},)`)
    return py.obj(s.pick(rows(s.length).flatMap((i) => Array.from({ length: times[i] }, () => i))))
  }],
  squeeze: ['axis', (s) => (s.length === 1 ? valueOf(s.values[0], s.dtype) : py.obj(s))],
  transpose: ['*', (s) => py.obj(s)],
  xs: ['key axis level drop_level', (s, p, h) => {
    const key = need(h, p.key, 'xs', 'key')
    if (!given(p.level)) return seriesIndexer(s, 'loc').getItem(key, h)
    const { rows: at, rest } = crossSection(h, s.index, key, p)
    const out = s.pick(at)
    out.index = rest
    return py.obj(out)
  }],
  filter: ['items like regex axis', (s, p, h) => py.obj(s.pick(filterLabels(h, s.index, p)))],
  droplevel: ['level axis', (s, p, h) => {
    const out = s.copy()
    out.index = dropIndexLevel(h, s.index, need(h, p.level, 'droplevel', 'level'))
    return py.obj(out)
  }],
  equals: ['other', (s, p) => py.bool(seriesEquals(s, p.other))],
  dot: ['other', (s, p, h) => dotSeries(h, s, need(h, p.other, 'dot', 'other'))],
  corr: ['other method min_periods', (s, p, h) => {
    const how = oneOf(h, p.method, 'pearson', ['pearson', 'spearman'], 'corr', 'method')
    const o = need(h, p.other, 'corr', 'other')
    if (!(o.k === 'obj' && o.o instanceof Series)) throw h.err('TypeError', `unsupported type: <class '${o.k === 'obj' ? o.o.cls : o.k}'>`)
    const [a, b] = pairOn(s, o.o)
    const minp = given(p.min_periods) ? h.toInt(p.min_periods!) : 1
    if (a.length < minp || !a.length) return py.float(NaN)
    if (how === 'spearman') {
      const [x, y] = rankedPair(a, b)
      return py.float(npCov(x, y, 1, true))
    }
    return py.float(npCov(a, b, 1, true))
  }],
  cov: ['other min_periods ddof', (s, p, h) => {
    const o = need(h, p.other, 'cov', 'other')
    if (!(o.k === 'obj' && o.o instanceof Series)) throw h.err('TypeError', "unsupported operand type(s) for -: 'list' and 'float'")
    const [a, b] = pairOn(s, o.o)
    if (a.length < (given(p.min_periods) ? h.toInt(p.min_periods!) : 1) || !a.length) return py.float(NaN)
    return py.float(npCov(a, b, given(p.ddof) ? h.toInt(p.ddof!) : 1, false))
  }],
  autocorr: ['lag', (s, p, h) => {
    const lag = p.lag ? h.toInt(p.lag) : 1
    const [a, b] = pairOn(s, shiftCol(h, s, lag))
    return py.float(a.length ? npCov(a, b, 1, true) : NaN)
  }],
  explode: ['ignore_index', (s, p, h) => py.obj(explodeCol(h, s, flag(h, p.ignore_index)))],
  update: ['other', (s, p, h) => {
    const o = need(h, p.other, 'update', 'other')
    // one value is a Series of one row, so it lands on label 0
    const scalar = o.k === 'int' || o.k === 'float' || o.k === 'str' || o.k === 'bool'
    const src = o.k === 'obj' && o.o instanceof Series ? o.o : new Series(typedCells(h, scalar ? py.list([o]) : o).cells, null, null, typedCells(h, scalar ? py.list([o]) : o).dt)
    const rs: number[] = []
    const cells: Cell[] = []
    s.index.labels.forEach((l, i) => {
      const at = src.index.locate(l)
      if (at.length && !isMissing(src.values[at[0]])) {
        rs.push(i)
        cells.push(src.values[at[0]])
      }
    })
    if (rs.length) s.store(rs, cells)
    return py.NONE
  }],
  combine_first: ['other', (s, p, h) => {
    const o = seriesIn(need(h, p.other, 'combine_first', 'other'))
    const ix = unionIndex(s.index, o.index)
    const a = reindexCells(s, ix)
    const b = reindexCells(o, ix)
    const vals = a.cells.map((c, i) => (isMissing(c) ? b.cells[i] : c))
    return py.obj(new Series(vals, ix, s.name, commonDT([a.dt, b.dt])))
  }],
  groupby: ['by level as_index sort group_keys observed dropna', (s, p, h) => groupBySeries(h, s, p)],
  rolling: ['window min_periods center win_type on closed step method axis', (s, p, h) => windowOf(h, s, p, 'rolling')],
  expanding: ['min_periods axis method', (s, p, h) => windowOf(h, s, p, 'expanding')],
  unstack: ['level fill_value sort', (s, p, h) => py.obj(unstackSeries(h, s, p))],
  first_valid_index: ['', (s) => {
    const i = s.values.findIndex((c) => !isMissing(c))
    return i < 0 ? py.NONE : valueOf(s.index.labels[i], s.index.dt)
  }],
  last_valid_index: ['', (s) => {
    for (let i = s.length - 1; i >= 0; i--) if (!isMissing(s.values[i])) return valueOf(s.index.labels[i], s.index.dt)
    return py.NONE
  }],
  searchsorted: ['value side sorter', (s, p, h) => {
    refuse(h, p.sorter, 'searchsorted', 'sorter')
    const right = given(p.side) && h.str(p.side!) === 'right'
    const find = (v: number) => {
      let i = 0
      const a = s.values.map(numCell)
      while (i < a.length && (right ? a[i] <= v : a[i] < v)) i++
      return i
    }
    const x = need(h, p.value, 'searchsorted', 'value')
    return isListLike(x) ? py.arr(NDArray.create(cellsOf(h, x).map((c) => find(numCell(c))), [cellsOf(h, x).length], 'int64')) : py.int(find(h.num(x)))
  }],
  info: ['verbose buf max_cols memory_usage show_counts', (s, p, h) => {
    for (const k of ['buf', 'max_cols', 'memory_usage', 'show_counts']) refuse(h, p[k], 'info', k)
    if (p.verbose && !h.truthy(p.verbose)) refuse(h, p.verbose, 'info', 'verbose')
    const nn = s.values.filter((c) => !isMissing(c)).length
    const dw = Math.max(5, dtName(s).length)
    h.print(`<class 'pandas.core.series.Series'>\n${indexInfoLine(s.index)}\nSeries name: ${s.name === null ? 'None' : labelStr(s.name)}\nNon-Null Count  ${'Dtype'.padEnd(dw)}\n--------------  ${'-----'.padEnd(dw)}\n${`${nn} non-null`.padEnd(14)}  ${dtName(s).padEnd(dw)}\ndtypes: ${dtName(s)}(1)\nmemory usage: ${sizeFmt(memOfIndex(s.index) + memOfSeries(s), s.dtype === 'object' ? '+' : '')}\n`)
    return py.NONE
  }],
  memory_usage: ['index deep', (s, p, h) => {
    if (flag(h, p.deep)) refuse(h, p.deep, 'memory_usage', 'deep')
    return py.int((flag(h, p.index, true) ? memOfIndex(s.index) : 0) + memOfSeries(s))
  }],
}

const refuseInto = (h: Host, p: Params) => {
  if (p.into && !(p.into.k === 'type' && p.into.name === 'dict')) refuse(h, p.into, 'to_dict', 'into')
}

const reprCell = (c: Cell) => (typeof c === 'string' ? `'${c}'` : labelStr(c))
const dictFromPairs = (pairs: [Value, Value][]) => PyDict.from(pairs)

const seriesGet = (h: Host, s: Series, p: Params): Value => {
  const key = need(h, p.key, 'get', 'key')
  const dflt = p.default ?? py.NONE
  if (isListLike(key)) {
    const labels = cellsOf(h, key)
    if (!labels.every((l) => s.index.has(l))) return dflt
    return py.obj(s.pick(labels.flatMap((l) => s.index.locate(l))))
  }
  const c = cellOf(h, key)
  if (!s.index.has(c)) return dflt
  return s.byLabel(h, key, c)
}

const swapBounds = (h: Host, p: Params): [Value | undefined, Value | undefined] => {
  const lo = given(p.lower)
  const up = given(p.upper)
  if (lo && up && !isListLike(lo) && !isListLike(up) && h.num(lo) > h.num(up)) return [up, lo]
  return [lo, up]
}

const fillMethodArg = (h: Host, p: Params): string | null => {
  if (p.fill_method === undefined) return 'pad'
  return p.fill_method.k === 'none' ? null : h.str(p.fill_method)
}

const floatFormatArg = (h: Host, v: Value | undefined): ((x: number) => string) | undefined => {
  if (!given(v)) return undefined
  if (v!.k === 'str') {
    const m = /^%(\+?)(\.(\d+))?([feg])$/.exec(v!.v)
    if (!m) throw h.err('NotImplementedError', `float_format='${v!.v}' is not available in this sandbox`)
    const d = m[3] ? Number(m[3]) : 6
    if (m[4] === 'f') return (x) => fixedStr(x, d)
    if (m[4] === 'e') return (x) => x.toExponential(d).replace(/e([+-])(\d)$/, 'e$10$2')
    return (x) => String(Number(x.toPrecision(d)))
  }
  return (x) => h.str(h.call(v!, [py.float(x)]))
}

const anyAll = (h: Host, s: Series, how: 'any' | 'all', skipna: boolean): boolean => {
  const truth = (c: Cell) => {
    if (isMissing(c)) return skipna ? how === 'all' : true
    if (typeof c === 'string') return c.length > 0
    if (c instanceof PyCell) return h.truthy(c.v)
    return !!c
  }
  return how === 'any' ? s.values.some(truth) : s.values.every(truth)
}

const bestPos = (s: Series, how: 'max' | 'min'): number => {
  let k = -1
  s.values.forEach((c, i) => {
    if (isMissing(c)) return
    if (k < 0) k = i
    else {
      const a = c as number
      const b = s.values[k] as number
      if (how === 'max' ? a > b : a < b) k = i
    }
  })
  return k
}

const idxOf = (h: Host, s: Series, how: 'max' | 'min', skipna: boolean): Value => {
  if (!s.length) throw h.err('ValueError', `attempt to get arg${how} of an empty sequence`)
  if (!skipna && s.values.some(isMissing)) return py.float(NaN)
  const k = bestPos(s, how)
  return k < 0 ? py.float(NaN) : valueOf(s.index.labels[k], s.index.dt)
}

const argOf = (h: Host, s: Series, how: 'max' | 'min', skipna: boolean): number => {
  if (!s.length) throw h.err('ValueError', `attempt to get arg${how} of an empty sequence`)
  if (!skipna && s.values.some(isMissing)) return -1
  return bestPos(s, how)
}

const modeCol = (s: Series, dropna: boolean): Series => {
  const counts = new Map<unknown, { c: Cell; n: number }>()
  for (const c of s.values) {
    if (isMissing(c) && dropna) continue
    const k = isMissing(c) ? 'NaN\0' : c
    const e = counts.get(k)
    if (e) e.n++
    else counts.set(k, { c, n: 1 })
  }
  const best = Math.max(0, ...[...counts.values()].map((e) => e.n))
  const modes = [...counts.values()].filter((e) => e.n === best).map((e) => e.c)
  const sorted = sortOrder([modes], [true]).map((i) => modes[i])
  return new Series(sorted, IndexObj.range(sorted.length), s.name, s.dtype)
}

export const factorize = (cells: Cell[], dt: DT, sort: boolean, sentinel: boolean): Value => {
  const seen = new Map<unknown, number>()
  const uniques: Cell[] = []
  const codes = cells.map((c) => {
    if (isMissing(c) && sentinel) return -1
    const k = isMissing(c) ? 'NaN\0' : c
    let at = seen.get(k)
    if (at === undefined) {
      at = uniques.length
      seen.set(k, at)
      uniques.push(c)
    }
    return at
  })
  let final = codes
  let us = uniques.map((c) => (c === null ? NaN : c))
  if (sort) {
    const order = sortOrder([uniques], [true])
    const rank = new Array<number>(order.length)
    order.forEach((o, i) => { rank[o] = i })
    final = codes.map((x) => (x < 0 ? x : rank[x]))
    us = order.map((i) => us[i])
  }
  return py.tuple([py.arr(NDArray.create(final, [final.length], 'int64')), py.obj(new IndexObj(us, null, dt))])
}

export const isinCol = (h: Host, s: Series, values: Value): Series => {
  if (values.k === 'str' || values.k === 'int' || values.k === 'float' || values.k === 'bool' || values.k === 'none') throw h.err('TypeError', `only list-like objects are allowed to be passed to isin(), you passed a \`${values.k === 'none' ? 'NoneType' : values.k}\``)
  const want = values.k === 'dict' ? values.d.keys().map((x) => cellOf(h, x)) : cellsOf(h, values)
  const set = new Set(want.map((c) => (isMissing(c) ? 'NaN\0' : c instanceof PyCell ? `\0${repr(c.v)}` : typeof c === 'boolean' ? Number(c) : c)))
  return s.with(s.values.map((c) => set.has(isMissing(c) ? 'NaN\0' : c instanceof PyCell ? `\0${repr(c.v)}` : typeof c === 'boolean' ? Number(c) : c)), 'bool')
}

const nExtreme = (h: Host, s: Series, p: Params, name: string, c: Ctx): Value => {
  if (s.dtype === 'object') throw h.err('TypeError', `Cannot use method '${name}' with dtype object`)
  oneOf(h, p.keep, 'first', ['first'], name, 'keep')
  const n = p.n ? h.toInt(p.n) : 5
  const present = rows(s.length).filter((i) => !isMissing(s.values[i]))
  const absent = rows(s.length).filter((i) => isMissing(s.values[i]))
  const order = [...sortOrder([present.map((i) => s.values[i])], [name === 'nsmallest']).map((k) => present[k]), ...absent].slice(0, Math.max(0, n))
  c.picks = { rows: order, cols: null }
  return py.obj(s.pick(order))
}

const fillCol = (h: Host, s: Series, p: Params): Series => {
  const limit = given(p.limit) ? h.toInt(p.limit!) : null
  if (given(p.method)) {
    const m = h.str(p.method!)
    if (m === 'ffill' || m === 'pad') return fillDirection(s, true, limit)
    if (m === 'bfill' || m === 'backfill') return fillDirection(s, false, limit)
    throw h.err('ValueError', `Invalid fill method. Expecting pad (ffill) or backfill (bfill). Got ${m}`)
  }
  const v = p.value
  if (!v || v.k === 'none') throw h.err('ValueError', "Must specify a fill 'value' or 'method'.")
  if (v.k === 'list' || v.k === 'tuple') throw h.err('TypeError', `"value" parameter must be a scalar or dict, but you passed a "${v.k}"`)
  const at = (i: number): Cell | undefined => {
    if (v.k === 'dict') {
      const x = v.d.get(valueOf(s.index.labels[i], s.index.dt))
      return x ? cellOf(h, x) : undefined
    }
    if (v.k === 'obj' && v.o instanceof Series) {
      const pos = v.o.index.locate(s.index.labels[i])
      return pos.length ? v.o.values[pos[0]] : undefined
    }
    return cellOf(h, v)
  }
  const rs: number[] = []
  const cells: Cell[] = []
  let used = 0
  s.values.forEach((c, i) => {
    if (!isMissing(c) || (limit !== null && used >= limit)) return
    const x = at(i)
    if (x === undefined || isMissing(x)) return
    rs.push(i)
    cells.push(x)
    used++
  })
  const out = s.copy()
  if (!rs.length) return out
  const dt0 = s.dtype
  out.store(rs, cells)
  if (s.cats) {
    if (!cells.every((x) => s.cats!.labels.includes(x))) throw h.err('TypeError', `Cannot setitem on a Categorical with a new category (${labelStr(cells[0])}), set the categories first`)
    out.cats = s.cats
  }
  if (dt0 === 'object') {
    const t = settleObject(out.values, out.dtype)
    out.values = t.cells
    out.dtype = t.dt
  }
  return out
}

const interpolateCol = (h: Host, s: Series, p: Params): Series => {
  if (s.dtype === 'bool') return s.copy()
  const how = oneOf(h, p.method, 'linear', ['linear', 'index', 'values'], 'interpolate', 'method')
  const dir = oneOf(h, p.limit_direction, 'forward', ['forward', 'backward', 'both'], 'interpolate', 'limit_direction')
  refuse(h, p.limit_area, 'interpolate', 'limit_area')
  refuse(h, p.downcast, 'interpolate', 'downcast')
  if (s.dtype === 'object') return s.copy()
  const limit = given(p.limit) ? h.toInt(p.limit!) : null
  const vals = s.values.map(numCell)
  if (!vals.some(Number.isNaN)) return s.copy()
  const n = vals.length
  const xs = how === 'linear' ? rows(n) : s.index.labels.map(numCell)
  const valid = rows(n).filter((i) => !Number.isNaN(vals[i]))
  if (!valid.length) return s.copy()
  const out = [...vals]
  const first = valid[0]
  const last = valid[valid.length - 1]
  // which gaps may be filled at all, by distance from the last valid value in the allowed direction
  const allowed = new Array<boolean>(n).fill(true)
  if (dir === 'forward' || limit !== null) {
    const fw = new Array<boolean>(n).fill(false)
    let run = 0
    let seen = false
    for (let i = 0; i < n; i++) {
      if (!Number.isNaN(vals[i])) {
        seen = true
        run = 0
      } else {
        run++
        fw[i] = seen && (limit === null || run <= limit)
      }
    }
    const bw = new Array<boolean>(n).fill(false)
    run = 0
    seen = false
    for (let i = n - 1; i >= 0; i--) {
      if (!Number.isNaN(vals[i])) {
        seen = true
        run = 0
      } else {
        run++
        bw[i] = seen && (limit === null || run <= limit)
      }
    }
    for (let i = 0; i < n; i++) allowed[i] = dir === 'forward' ? fw[i] : dir === 'backward' ? bw[i] : fw[i] || bw[i]
  } else if (dir === 'backward') {
    for (let i = last + 1; i < n; i++) allowed[i] = false
  }
  let j = 0
  for (let i = 0; i < n; i++) {
    if (!Number.isNaN(vals[i]) || !allowed[i]) continue
    while (j + 1 < valid.length && valid[j + 1] < i) j++
    if (i < first) out[i] = vals[first]
    else if (i > last) out[i] = vals[last]
    else {
      const a = valid[j]
      const b = valid[j + 1]
      const t = (xs[i] - xs[a]) / (xs[b] - xs[a])
      out[i] = t * (vals[b] - vals[a]) + vals[a]
    }
  }
  return s.with(out, 'float64')
}

const prefixed = (s: Series, pre: string, suf: string): Series => {
  const out = s.copy()
  out.index = new IndexObj(s.index.labels.map((l) => `${pre}${labelStr(l)}${suf}`), s.index.name, 'object')
  return out
}

const seriesEquals = (s: Series, other: Value | undefined): boolean => {
  if (!(other && other.k === 'obj' && other.o instanceof Series)) return false
  const o = other.o
  return s.dtype === o.dtype && s.index.equals(o.index) && s.length === o.length && s.values.every((c, i) => sameCell(c, o.values[i]))
}

const dotSeries = (h: Host, s: Series, o: Value): Value => {
  if (o.k === 'obj' && o.o instanceof Series) {
    const ix = unionIndex(s.index, o.o.index)
    if (ix.length > s.length || ix.length > o.o.length) throw h.err('ValueError', 'matrices are not aligned')
    const a = reindexCells(s, ix).cells.map(numCell)
    const b = reindexCells(o.o, ix).cells.map(numCell)
    const out = a.reduce((acc, v, i) => acc + v * b[i], 0)
    return s.dtype === 'float64' || o.o.dtype === 'float64' ? py.float(out) : py.int(out)
  }
  if (o.k === 'obj' && o.o instanceof Frame) {
    const f = o.o
    return py.obj(new Series(f.data.map((c) => s.values.reduce<number>((acc, v, i) => acc + numCell(v) * numCell(c.values[i]), 0)), f.columns, null, commonDT([s.dtype, ...f.data.map((c) => c.dtype)])))
  }
  const a = (h.getAttr(py.obj(s), 'to_numpy'))
  const dot = npFunc(h, 'dot')!
  return h.call(dot, [h.call(a, []), o])
}

const pipeCall = (h: Host, self: Value, p: Params): Value => {
  const f = need(h, p.func, 'pipe', 'func')
  const { args, kw } = unboxArgs(h, p)
  if (f.k === 'tuple') {
    const [g, target] = f.items
    return h.call(g, args, { ...kw, [h.str(target)]: self })
  }
  return h.call(f, [self, ...args], kw)
}

const explodeCol = (h: Host, s: Series, ignore: boolean): Series => {
  // nothing list-like: the same column, with its dtype
  if (!s.values.some((c) => c instanceof PyCell && (c.v.k === 'list' || c.v.k === 'tuple' || c.v.k === 'array'))) return new Series([...s.values], ignore ? null : s.index, s.name, s.dtype)
  const cells: Cell[] = []
  const at: number[] = []
  s.values.forEach((c, i) => {
    if (c instanceof PyCell && (c.v.k === 'list' || c.v.k === 'tuple' || c.v.k === 'array')) {
      const items = h.iterate(c.v)
      if (!items.length) {
        cells.push(NaN)
        at.push(i)
      }
      for (const x of items) {
        cells.push(x.k === 'float' ? boxFloat(x.v) : cellOf(h, x))
        at.push(i)
      }
    } else {
      cells.push(c)
      at.push(i)
    }
  })
  return new Series(cells, ignore ? IndexObj.range(cells.length) : s.index.pick(at), s.name, 'object')
}

/** reset_index: the index levels become the first columns */
export const indexToColumns = (h: Host, f: Frame, levels?: number[], allowDuplicates = false): Frame => {
  const ix = f.index
  const n = ix.nlevels
  const chosen = levels ?? Array.from({ length: n }, (_, i) => i)
  const names = ix.names.map((x, j) => x ?? (n > 1 ? `level_${j}` : f.columns.has('index') ? 'level_0' : 'index'))
  if (!allowDuplicates) for (const j of chosen) if (f.columns.has(names[j])) throw h.err('ValueError', `cannot insert ${labelStr(names[j])}, already exists`)
  const inserts = chosen.map((j) => {
    const lev = ix.level(j)
    return new Series([...lev.labels], null, names[j], lev.dt)
  })
  const rest = Array.from({ length: n }, (_, i) => i).filter((j) => !chosen.includes(j))
  const newIndex = !rest.length ? IndexObj.range(f.nrows) : rest.length === 1 ? ix.level(rest[0]) : IndexObj.fromArrays(rest.map((j) => ix.multi!.arrays[j]), rest.map((j) => ix.multi!.names[j]), rest.map((j) => ix.multi!.dts[j]))
  const columnsBase = f.columns.multi
    ? IndexObj.fromArrays(f.columns.multi.arrays.map((a, j) => [...chosen.map((k) => (j === 0 ? names[k] : '')), ...a]), f.columns.multi.names, f.columns.multi.dts.map((d, j) => (j === 0 ? d : 'object')))
    : new IndexObj([...chosen.map((j) => names[j]), ...f.columns.labels], f.columns.name, chosen.length ? commonDT([f.columns.dt, 'object']) : f.columns.dt)
  return new Frame(columnsBase, [...inserts, ...f.data.map((s) => s.copy())], newIndex)
}

// ---------------------------------------------------------------- Series attributes

export const seriesAttr = (s: Series, name: string, h: Host): Value | undefined => {
  if (has(SERIES, name)) return method(s, 'Series', name, SERIES[name], h)
  if (has(REDUCERS, name)) {
    const [how, spec] = REDUCERS[name]
    return method(s, 'Series', name, [spec, (x, p, hh) => {
      if ((how === 'std' || how === 'var') && !p.ddof) noteOnce(hh, 'pandas_ddof')
      axisOf(hh, p.axis, 'Series')
      if (flag(hh, p.numeric_only) && x.dtype === 'object') throw hh.err('TypeError', `Series.${name} does not allow numeric_only=True with non-numeric dtypes.`)
      return reduceCol(hh, x, how, reduceOpts(hh, p))
    }], h)
  }
  if (has(FLEX, name)) {
    const [op, reflected] = FLEX[name]
    return method(s, 'Series', name, ['other level fill_value axis', (x, p, hh) => {
      refuse(hh, p.level, name, 'level')
      return py.obj(seriesOp(hh, x, op, need(hh, p.other, name, 'other'), reflected, given(p.fill_value) ? cellOf(hh, p.fill_value!) : undefined, true))
    }], h)
  }
  if (has(FLEX_CMP, name)) {
    return method(s, 'Series', name, ['other level fill_value axis', (x, p, hh) => {
      refuse(hh, p.level, name, 'level')
      return py.obj(seriesOp(hh, x, FLEX_CMP[name], need(hh, p.other, name, 'other'), false, given(p.fill_value) ? cellOf(hh, p.fill_value!) : undefined, true))
    }], h)
  }
  switch (name) {
    case 'name': return s.name === null ? py.NONE : valueOf(s.name, inferDT([s.name]))
    case 'dtype': case 'dtypes': return py.obj(new DtypeObj(dtName(s)))
    case 'shape': return py.tuple([py.int(s.length)])
    case 'size': return py.int(s.length)
    case 'ndim': return py.int(1)
    case 'empty': return py.bool(s.length === 0)
    case 'index': return py.obj(s.index)
    case 'axes': return py.list([py.obj(s.index)])
    case 'values': return arrayOf(s.values, s.dtype, s.width)
    case 'array': return py.obj(new PandasArray(s))
    case 'T': return py.obj(s)
    case 'hasnans': return py.bool(s.values.some(isMissing))
    case 'is_unique': return py.bool(uniqueCells(s.values).length === s.length)
    case 'is_monotonic_increasing': return py.bool(monotonic(s.values, true))
    case 'is_monotonic_decreasing': return py.bool(monotonic(s.values, false))
    case 'nbytes': return py.int(memOfSeries(s))
    case 'attrs': return { k: 'dict', d: dictFromPairs([]) }
    case 'loc': case 'iloc': case 'at': case 'iat': return py.obj(seriesIndexer(s, name))
    case 'str': return py.obj(strAccessor(h, s))
    case 'cat':
      if (!s.cats) throw h.err('AttributeError', 'Can only use .cat accessor with a \'category\' dtype')
      return py.obj(new CatAccessor(s))
  }
  return undefined
}

const monotonic = (cells: Cell[], inc: boolean) => {
  if (cells.some(isMissing)) return false
  for (let i = 1; i < cells.length; i++) {
    const a = cells[i - 1] as number
    const b = cells[i] as number
    if (inc ? a > b : a < b) return false
  }
  return true
}

const memOfSeries = (s: Series) => (s.cats ? s.length + 8 * s.cats.labels.length : (s.dtype === 'bool' ? 1 : s.width ? Number(/\d+/.exec(s.width)?.[0] ?? 64) / 8 : 8) * s.length)
const memOfIndex = (ix: IndexObj) => {
  if (!ix.isRange) return 8 * ix.length
  // sys.getsizeof of the range and its start, stop and step: 0 is a smaller int
  const size = (x: number) => (x === 0 ? 24 : 28)
  const start = (ix.labels[0] as number | undefined) ?? 0
  const step = ix.length > 1 ? (ix.labels[1] as number) - start : 1
  return 48 + size(start) + size(start + ix.length * step) + size(step)
}
const sizeFmt = (num0: number, q: string) => {
  let x = num0
  for (const u of ['bytes', 'KB', 'MB', 'GB', 'TB']) {
    if (x < 1024) return `${fixedStr(x, 1)}${q} ${u}`
    x /= 1024
  }
  return `${fixedStr(x, 1)}${q} PB`
}
const indexInfoLine = (ix: IndexObj) => {
  if (ix.isRange) return `RangeIndex: ${ix.length} entries, ${labelStr(ix.labels[0] ?? 0)} to ${labelStr(ix.labels[ix.length - 1] ?? -1)}`
  if (!ix.length) return 'Index: 0 entries'
  return `${ix.cls}: ${ix.length} entries, ${labelStr(ix.labels[0])} to ${labelStr(ix.labels[ix.length - 1])}`
}

// ---------------------------------------------------------------- DataFrame methods

const cellsToFrameKeys = (h: Host, f: Frame, v: Value | undefined): number[] => {
  if (!given(v)) return f.data.map((_, k) => k)
  const labels = isListLike(v!) ? cellsOf(h, v!) : [cellOf(h, v!)]
  const missing = labels.filter((l) => !f.columns.has(l))
  if (missing.length) throw h.err('KeyError', `Index([${missing.map(reprCell).join(', ')}], dtype='${inferDT(missing)}')`)
  return labels.flatMap((l) => f.columns.locate(l))
}

const axisLabels = (h: Host, v: Value): Cell[] => (isListLike(v) ? cellsOf(h, v) : [cellOf(h, v)])

const dropFrame = (h: Host, f: Frame, p: Params): Frame => {
  refuse(h, p.level, 'drop', 'level')
  const ignore = given(p.errors) && h.str(p.errors!) === 'ignore'
  const axis = axisOf(h, p.axis, 'DataFrame')
  let rowLabels = given(p.index)
  let colLabels = given(p.columns)
  if (p.labels) {
    if (rowLabels || colLabels) throw h.err('ValueError', "Cannot specify both 'labels' and 'index'/'columns'")
    if (axis === 1) colLabels = p.labels
    else rowLabels = p.labels
  }
  if (!rowLabels && !colLabels) throw h.err('ValueError', "Need to specify at least one of 'labels', 'index' or 'columns'")
  const positionsToDrop = (ix: IndexObj, labels: Cell[]) => {
    const missing = labels.filter((l) => !ix.has(l) && !(ix.multi && ix.multi.arrays[0].includes(l)))
    if (missing.length && !ignore) throw h.err('KeyError', repr(py.str(`[${missing.map(reprCell).join(', ')}] not found in axis`)))
    return new Set(labels.flatMap((l) => (ix.has(l) ? ix.locate(l) : ix.multi ? rows(ix.length).filter((i) => ix.multi!.arrays[0][i] === l) : [])))
  }
  let out = f
  if (colLabels) {
    const d = positionsToDrop(f.columns, axisLabels(h, colLabels))
    out = out.take(null, rows(f.ncols).filter((k) => !d.has(k)))
  }
  if (rowLabels) {
    const d = positionsToDrop(f.index, axisLabels(h, rowLabels))
    out = out.take(rows(f.nrows).filter((r) => !d.has(r)), null)
  }
  return out === f ? f.copy() : out
}


const renameFrame = (h: Host, f: Frame, p: Params): Frame => {
  refuse(h, p.level, 'rename', 'level')
  const axis = axisOf(h, p.axis, 'DataFrame')
  const colsM = given(p.columns) ?? (given(p.mapper) && axis === 1 ? p.mapper : undefined)
  const idxM = given(p.index) ?? (given(p.mapper) && axis === 0 ? p.mapper : undefined)
  if (!colsM && !idxM) throw h.err('TypeError', "must pass an index to rename")
  const strict = given(p.errors) && h.str(p.errors!) === 'raise'
  const check = (m: Value | undefined, ix: IndexObj) => {
    if (strict && m && m.k === 'dict') {
      const miss = m.d.keys().map((k) => cellOf(h, k)).filter((c) => !ix.has(c))
      if (miss.length) throw h.err('KeyError', repr(py.str(`[${miss.map(reprCell).join(', ')}] not found in axis`)))
    }
  }
  check(colsM, f.columns)
  check(idxM, f.index)
  const cols = colsM ? renameLabels(h, f.columns, colsM) : f.columns
  const idx = idxM ? renameLabels(h, f.index, idxM) : f.index
  return new Frame(cols, f.data.map((s) => s.copy()), idx)
}

const setIndexFrame = (h: Host, f: Frame, p: Params): Frame => {
  const keys = need(h, p.keys, 'set_index', 'keys')
  const list = keys.k === 'list' ? keys.items : [keys]
  const drop = flag(h, p.drop, true)
  const toDrop = new Set<number>()
  const arrays: { cells: Cell[]; name: Cell; dt: DT }[] = []
  const missing: Cell[] = []
  for (const k of list) {
    if (k.k === 'obj' && (k.o instanceof Series || k.o instanceof IndexObj)) {
      const t = typedCells(h, k)
      arrays.push({ cells: t.cells, name: k.o.name, dt: t.dt })
    } else if (k.k === 'array' || k.k === 'range' || k.k === 'list') {
      const t = typedCells(h, k)
      arrays.push({ cells: t.cells, name: null, dt: t.dt })
    } else {
      const c = cellOf(h, k)
      const at = f.columns.locate(c)
      if (!at.length) {
        missing.push(c)
        continue
      }
      arrays.push({ cells: [...f.data[at[0]].values], name: c, dt: f.data[at[0]].dtype })
      if (drop) toDrop.add(at[0])
    }
  }
  if (missing.length) throw h.err('KeyError', repr(py.str(`None of [${missing.map(reprCell).join(', ')}] are in the columns`)))
  const append = flag(h, p.append)
  const existing = append ? (f.index.multi ? f.index.multi.arrays.map((a, j) => ({ cells: a, name: f.index.multi!.names[j], dt: f.index.multi!.dts[j] })) : [{ cells: f.index.labels, name: f.index.name, dt: f.index.dt }]) : []
  const all = [...existing, ...arrays]
  const index = all.length === 1 ? new IndexObj(all[0].cells, all[0].name, all[0].dt) : IndexObj.fromArrays(all.map((a) => a.cells), all.map((a) => a.name), all.map((a) => a.dt))
  if (index.length !== f.nrows) throw h.err('ValueError', `Length mismatch: Expected ${f.nrows} rows, received array of length ${index.length}`)
  if (flag(h, p.verify_integrity) && !index.isUnique) throw h.err('ValueError', 'Index has duplicate keys')
  const keep = rows(f.ncols).filter((k) => !toDrop.has(k))
  return new Frame(f.columns.pick(keep, true), keep.map((k) => f.data[k].copy()), index)
}

const toDictFrame = (h: Host, f: Frame, orientV: Value | undefined, indexFlag: boolean): Value => {
  const orient = orientV ? h.str(orientV) : 'dict'
  const dict = (pairs: [Value, Value][]): Value => ({ k: 'dict', d: dictFromPairs(pairs) })
  const colKeys = f.columns.iter()
  const rowKeys = f.index.iter()
  const n = f.nrows
  const cell = (k: number, r: number) => valueOf(f.data[k].values[r], f.data[k].dtype)
  switch (orient) {
    case 'dict': return dict(colKeys.map((c, k) => [c, dict(rowKeys.map((l, r) => [l, cell(k, r)]))]))
    case 'list': return dict(colKeys.map((c, k) => [c, py.list(rows(n).map((r) => cell(k, r)))]))
    case 'series': return dict(colKeys.map((c, k) => [c, py.obj(f.data[k])]))
    case 'split': return dict([...(indexFlag ? [[py.str('index'), py.list(rowKeys)] as [Value, Value]] : []), [py.str('columns'), py.list(colKeys)], [py.str('data'), py.list(rows(n).map((r) => py.list(f.data.map((_, k) => cell(k, r)))))]])
    case 'tight': return dict([...(indexFlag ? [[py.str('index'), py.list(rowKeys)] as [Value, Value]] : []), [py.str('columns'), py.list(colKeys)], [py.str('data'), py.list(rows(n).map((r) => py.list(f.data.map((_, k) => cell(k, r)))))], ...(indexFlag ? [[py.str('index_names'), py.list(f.index.names.map((x) => valueOf(x, 'object')))] as [Value, Value]] : []), [py.str('column_names'), py.list(f.columns.names.map((x) => valueOf(x, 'object')))]])
    case 'records': return py.list(rows(n).map((r) => dict(colKeys.map((c, k) => [c, cell(k, r)]))))
    case 'index':
      if (!f.index.isUnique) throw h.err('ValueError', "DataFrame index must be unique for orient='index'.")
      return dict(rowKeys.map((l, r) => [l, dict(colKeys.map((c, k) => [c, cell(k, r)]))]))
  }
  throw h.err('ValueError', `orient '${orient}' not understood`)
}

/** a covariance-like table: each pair computed once, the later column first, as pandas' nancorr does */
const symmetric = (f: Frame, cols: number[], vals: number[][], g: (x: number[], y: number[]) => number): Frame => {
  const names = f.columns.pick(cols, true)
  const grid = vals.map(() => new Array<number>(vals.length).fill(NaN))
  for (let xi = 0; xi < vals.length; xi++) for (let yi = 0; yi <= xi; yi++) grid[xi][yi] = grid[yi][xi] = g(vals[xi], vals[yi])
  return new Frame(names, grid.map((col, k) => new Series(col, names, names.labels[k], 'float64')), names)
}

const FRAME: Record<string, Method<Frame>> = {
  to_numpy: ['dtype copy na_value', (f, p, h) => frameArray(h, f, p)],
  head: ['n', (f, p, h, c) => headTail(f, 'head', p.n, h, c)],
  tail: ['n', (f, p, h, c) => headTail(f, 'tail', p.n, h, c)],
  take: ['indices axis', (f, p, h) => {
    const axis = axisOf(h, p.axis, 'DataFrame')
    const idx = need(h, p.indices, 'take', 'indices')
    return py.obj(axis === 0 ? f.take(positions(h, idx, f.nrows, 'indices are out-of-bounds'), null) : f.take(null, positions(h, idx, f.ncols, 'indices are out-of-bounds')))
  }],
  info: ['verbose buf max_cols memory_usage show_counts', (f, p, h) => {
    for (const k of ['buf', 'max_cols']) refuse(h, p[k], 'info', k)
    if (p.verbose && !h.truthy(p.verbose)) refuse(h, p.verbose, 'info', 'verbose')
    if (p.memory_usage?.k === 'str') refuse(h, p.memory_usage, 'info', 'memory_usage')
    return frameInfo(h, f, flag(h, p.show_counts, true), !p.memory_usage || h.truthy(p.memory_usage))
  }],
  memory_usage: ['index deep', (f, p, h) => {
    if (flag(h, p.deep)) refuse(h, p.deep, 'memory_usage', 'deep')
    const useIndex = flag(h, p.index, true)
    const labels = [...(useIndex ? ['Index'] : []), ...f.columns.labels]
    const cells = [...(useIndex ? [memOfIndex(f.index)] : []), ...f.data.map(memOfSeries)]
    return py.obj(new Series(cells, new IndexObj(labels, null, inferDT(labels)), null, 'int64'))
  }],
  describe: ['percentiles include exclude', (f, p, h) => py.obj(describeFrame(h, f, p))],
  count: ['axis numeric_only', (f, p, h) => frameReduce(h, f, 'count', p)],
  quantile: ['q axis numeric_only interpolation method', (f, p, h) => {
    oneOf(h, p.method, 'single', ['single'], 'quantile', 'method')
    const { interp, many, qs } = quantileArgs(h, p)
    const axis = axisOf(h, p.axis, 'DataFrame')
    const src = axis === 1 ? transposeFrame(f) : f
    const cols = flag(h, p.numeric_only) ? numericCols(src) : src.data.map((_, k) => k)
    const parts = cols.map((k) => quantileCol(h, src.data[k], qs, interp))
    if (many) {
      const ix = new IndexObj(qs, null, 'float64')
      const out = new Frame(src.columns.pick(cols, true), parts.map((x, i) => new Series(x.vals, ix, src.columns.labels[cols[i]], x.dt)), ix)
      return py.obj(axis === 1 ? transposeFrame(out) : out)
    }
    return py.obj(new Series(parts.map((x) => x.vals[0]), src.columns.pick(cols, true), qs[0], commonDT(parts.map((x) => x.dt))))
  }],
  all: ['axis bool_only skipna', (f, p, h) => frameAnyAll(h, f, 'all', p)],
  any: ['axis bool_only skipna', (f, p, h) => frameAnyAll(h, f, 'any', p)],
  idxmax: ['axis skipna numeric_only', (f, p, h) => frameIdx(h, f, 'max', p)],
  idxmin: ['axis skipna numeric_only', (f, p, h) => frameIdx(h, f, 'min', p)],
  nunique: ['axis dropna', (f, p, h) => {
    const dropna = flag(h, p.dropna, true)
    const count = (s: Series) => uniqueCells(dropna ? s.values.filter((c) => !isMissing(c)) : s.values).length
    const axis = axisOf(h, p.axis, 'DataFrame')
    if (axis === 1) return py.obj(new Series(rows(f.nrows).map((r) => count(f.row(r))), f.index, null, 'int64'))
    return py.obj(new Series(f.data.map(count), f.columns, null, 'int64'))
  }],
  value_counts: ['subset normalize sort ascending dropna', (f, p, h) => py.obj(frameValueCounts(h, f, p))],
  mode: ['axis numeric_only dropna', (f, p, h) => {
    if (axisOf(h, p.axis, 'DataFrame') === 1) refuse(h, py.int(1), 'mode', 'axis')
    const cols = flag(h, p.numeric_only) ? numericCols(f) : f.data.map((_, k) => k)
    const parts = cols.map((k) => modeCol(f.data[k], flag(h, p.dropna, true)))
    const n = Math.max(0, ...parts.map((x) => x.length))
    const ix = IndexObj.range(n)
    return py.obj(new Frame(f.columns.pick(cols, true), parts.map((x) => {
      const t = reindexCells(x, ix)
      return new Series(t.cells, ix, x.name, t.dt)
    }), ix))
  }],
  isna: ['', (f, _p, _h, c) => {
    c.api = 'isna'
    return py.obj(f.same(f.data.map((s) => s.with(s.values.map(isMissing), 'bool'))))
  }],
  isnull: ['', (f, _p, _h, c) => {
    c.api = 'isna'
    return py.obj(f.same(f.data.map((s) => s.with(s.values.map(isMissing), 'bool'))))
  }],
  notna: ['', (f, _p, _h, c) => {
    c.api = 'isna'
    return py.obj(f.same(f.data.map((s) => s.with(s.values.map((x) => !isMissing(x)), 'bool'))))
  }],
  notnull: ['', (f, _p, _h, c) => {
    c.api = 'isna'
    return py.obj(f.same(f.data.map((s) => s.with(s.values.map((x) => !isMissing(x)), 'bool'))))
  }],
  fillna: ['value | method axis inplace limit', (f, p, h, c) => {
    const filled = new Set<number>()
    f.data.forEach((s) => s.values.forEach((x, r) => { if (isMissing(x)) filled.add(r) }))
    c.picks = { rows: [...filled], cols: null }
    if (axisOf(h, p.axis, 'DataFrame') === 1) refuse(h, py.int(1), 'fillna', 'axis')
    const v = p.value
    if (v && v.k === 'obj' && v.o instanceof Frame) {
      const o = v.o
      return finish(h, f, f.same(f.data.map((s) => {
        const src = o.col(s.name)
        return src ? fillCol(h, s, { value: py.obj(src) }) : s.copy()
      })), p.inplace)
    }
    const perCol = (s: Series): Series => {
      if (v && v.k === 'dict') {
        const x = v.d.get(valueOf(s.name, f.columns.dt))
        return x ? fillCol(h, s, { ...p, value: x }) : s.copy()
      }
      if (v && v.k === 'obj' && v.o instanceof Series) {
        const at = v.o.index.locate(s.name)
        return at.length ? fillCol(h, s, { ...p, value: valueOf(v.o.values[at[0]], v.o.dtype) }) : s.copy()
      }
      return fillCol(h, s, p)
    }
    return finish(h, f, f.same(f.data.map(perCol)), p.inplace)
  }],
  ffill: ['axis inplace limit limit_area', (f, p, h) => {
    refuse(h, p.limit_area, 'ffill', 'limit_area')
    return finish(h, f, byAxis(f, axisOf(h, p.axis, 'DataFrame'), (s) => fillDirection(s, true, given(p.limit) ? h.toInt(p.limit!) : null)), p.inplace)
  }],
  pad: ['axis inplace limit', (f, p, h) => finish(h, f, byAxis(f, axisOf(h, p.axis, 'DataFrame'), (s) => fillDirection(s, true, given(p.limit) ? h.toInt(p.limit!) : null)), p.inplace)],
  bfill: ['axis inplace limit limit_area', (f, p, h) => {
    refuse(h, p.limit_area, 'bfill', 'limit_area')
    return finish(h, f, byAxis(f, axisOf(h, p.axis, 'DataFrame'), (s) => fillDirection(s, false, given(p.limit) ? h.toInt(p.limit!) : null)), p.inplace)
  }],
  backfill: ['axis inplace limit', (f, p, h) => finish(h, f, byAxis(f, axisOf(h, p.axis, 'DataFrame'), (s) => fillDirection(s, false, given(p.limit) ? h.toInt(p.limit!) : null)), p.inplace)],
  dropna: ['| axis how thresh subset inplace ignore_index', (f, p, h, c) => {
    const axis = axisOf(h, p.axis, 'DataFrame')
    if (given(p.how) && p.thresh) throw h.err('TypeError', 'You cannot set both the how and thresh arguments at the same time.')
    const how = given(p.how) ? h.str(p.how!) : 'any'
    if (how !== 'any' && how !== 'all') throw h.err('ValueError', `invalid how option: ${how}`)
    const thresh = given(p.thresh) ? h.toInt(p.thresh!) : null
    const test = (present: number, total: number) => (thresh !== null ? present >= thresh : how === 'any' ? present === total : present > 0)
    if (axis === 1) {
      const keepRows = given(p.subset) ? (isListLike(p.subset!) ? cellsOf(h, p.subset!) : [cellOf(h, p.subset!)]).flatMap((l) => f.index.locate(l)) : rows(f.nrows)
      const keep = rows(f.ncols).filter((k) => test(keepRows.filter((r) => !isMissing(f.data[k].values[r])).length, keepRows.length))
      return finish(h, f, f.take(null, keep), p.inplace)
    }
    const cols = cellsToFrameKeys(h, f, p.subset)
    const keep = rows(f.nrows).filter((r) => test(cols.filter((k) => !isMissing(f.data[k].values[r])).length, cols.length))
    c.picks = { rows: keep, cols: null }
    const out = f.take(keep, null)
    if (flag(h, p.ignore_index)) out.index = IndexObj.range(keep.length)
    return finish(h, f, flag(h, p.ignore_index) ? new Frame(out.columns, out.data, IndexObj.range(keep.length)) : out, p.inplace)
  }],
  interpolate: ['method axis limit inplace limit_direction limit_area downcast', (f, p, h) => (f.ncols && f.data.every((s) => s.dtype === 'object') ? (() => { throw h.err('TypeError', 'Cannot interpolate with all object-dtype columns in the DataFrame. Try setting at least one column to a numeric dtype.') })() : finish(h, f, byAxis(f, axisOf(h, p.axis, 'DataFrame'), (s) => interpolateCol(h, s, p)), p.inplace))],
  replace: ['to_replace value inplace limit regex method', (f, p, h) => {
    refuse(h, p.limit, 'replace', 'limit')
    refuse(h, p.method, 'replace', 'method')
    const a = p.to_replace
    const regex = flag(h, p.regex)
    if (a && a.k === 'dict') {
      const items = a.d.items()
      const nested = items.length > 0 && items.every(([, v]) => v.k === 'dict')
      const byCol = !nested && p.value !== undefined
      return finish(h, f, f.same(f.data.map((s) => {
        const key = valueOf(s.name, f.columns.dt)
        if (nested) {
          const m = a.d.get(key)
          return m && m.k === 'dict' ? replaceCol(h, s, m.d.items().map(([k, v]) => scalarPair(h, k, v)), regex) : s.copy()
        }
        if (byCol) {
          const from = a.d.get(key)
          return from ? replaceCol(h, s, [scalarPair(h, from, p.value!)], regex) : s.copy()
        }
        return replaceCol(h, s, items.map(([k, v]) => scalarPair(h, k, v)), regex)
      })), p.inplace)
    }
    const pairs = replaceArgs(h, p, 'replace')!
    return finish(h, f, f.same(f.data.map((s) => replaceCol(h, s, pairs, regex))), p.inplace)
  }],
  where: ['cond other inplace axis level', (f, p, h) => {
    refuse(h, p.level, 'where', 'level')
    const cond = need(h, p.cond, 'where', 'cond')
    return finish(h, f, f.same(f.data.map((s) => whereCol(h, s, condCells(h, cond, s, f), p.other, f, p.axis))), p.inplace)
  }],
  mask: ['cond other inplace axis level', (f, p, h) => {
    refuse(h, p.level, 'mask', 'level')
    const cond = need(h, p.cond, 'mask', 'cond')
    return finish(h, f, f.same(f.data.map((s) => whereCol(h, s, condCells(h, cond, s, f).map((x) => !x), p.other, f, p.axis))), p.inplace)
  }],
  clip: ['lower upper axis inplace', (f, p, h) => {
    const [lo, up] = swapBounds(h, p)
    const axis = given(p.axis) ? axisOf(h, p.axis, 'DataFrame') : null
    if ((lo?.k === 'obj' || up?.k === 'obj') && axis === null) throw h.err('ValueError', 'Must specify axis=0 or 1')
    const perCol = (s: Series, b: Value | undefined): Value | undefined => {
      if (axis === 1 && b && b.k === 'obj' && b.o instanceof Series) {
        const at = b.o.index.locate(s.name)
        return at.length ? valueOf(b.o.values[at[0]], b.o.dtype) : undefined
      }
      return b
    }
    return finish(h, f, f.same(f.data.map((s) => clipCol(h, s, perCol(s, lo), perCol(s, up)))), p.inplace)
  }],
  abs: ['', (f, _p, h) => py.obj(f.same(f.data.map((s) => {
    if (s.dtype === 'object') throw h.err('TypeError', "bad operand type for abs(): 'str'")
    return s.with(s.values.map((x) => Math.abs(numCell(x))), s.dtype === 'bool' ? 'bool' : s.dtype)
  })))],
  round: ['decimals', (f, p, h) => {
    const d = p.decimals
    const dOf = (s: Series): number | null => {
      if (!d) return 0
      if (d.k === 'dict') {
        const x = d.d.get(valueOf(s.name, f.columns.dt))
        return x ? h.toInt(x) : null
      }
      if (d.k === 'obj' && d.o instanceof Series) {
        const at = d.o.index.locate(s.name)
        return at.length ? Number(d.o.values[at[0]]) : null
      }
      return h.toInt(d)
    }
    return py.obj(f.same(f.data.map((s) => {
      const k = dOf(s)
      if (k === null || s.dtype === 'object' || s.dtype === 'bool' || (s.dtype === 'int64' && k >= 0)) return s.copy()
      return s.with(s.values.map((c) => roundCell(c as number, k)), s.dtype)
    })))
  }],
  isin: ['values', (f, p, h) => {
    const v = need(h, p.values, 'isin', 'values')
    return py.obj(f.same(f.data.map((s) => {
      if (v.k === 'dict') {
        const x = v.d.get(valueOf(s.name, f.columns.dt))
        return x ? isinCol(h, s, x) : s.with(s.values.map(() => false), 'bool')
      }
      if (v.k === 'obj' && v.o instanceof Series) {
        const o = v.o
        return s.with(s.values.map((c, r) => {
          const at = o.index.locate(f.index.labels[r])
          return at.length && !isMissing(c) ? sameCell(c, o.values[at[0]]) : false
        }), 'bool')
      }
      if (v.k === 'obj' && v.o instanceof Frame) {
        const o = v.o
        const src = o.col(s.name)
        return s.with(s.values.map((c, r) => {
          const at = o.index.locate(f.index.labels[r])
          return src && at.length && !isMissing(c) ? sameCell(c, src.values[at[0]]) : false
        }), 'bool')
      }
      return isinCol(h, s, v)
    })))
  }],
  map: ['func na_action **', (f, p, h) => {
    const g = need(h, p.func, 'map', 'func')
    if (!callableValue(g)) throw h.err('TypeError', `the first argument must be callable`)
    const { kw } = unboxArgs(h, p)
    return py.obj(f.same(f.data.map((s) => mapCol(h, s, g, given(p.na_action) ? h.str(p.na_action!) === 'ignore' : false, kw))))
  }],
  applymap: ['func na_action **', (f, p, h) => {
    const g = need(h, p.func, 'applymap', 'func')
    const { kw } = unboxArgs(h, p)
    return py.obj(f.same(f.data.map((s) => mapCol(h, s, g, given(p.na_action) ? h.str(p.na_action!) === 'ignore' : false, kw))))
  }],
  apply: ['func axis raw result_type args by_row **', (f, p, h) => {
    const g = need(h, p.func, 'apply', 'func')
    if (p.by_row && p.by_row.k !== 'str') refuse(h, p.by_row, 'apply', 'by_row')
    if (given(p.result_type) && h.str(p.result_type!) === 'broadcast') refuse(h, p.result_type, 'apply', 'result_type')
    const axis = axisOf(h, p.axis, 'DataFrame')
    const { args, kw } = unboxArgs(h, p)
    if (g.k === 'str' || g.k === 'dict' || isListLike(g)) return frameAgg(h, f, g, axis, {})
    return frameApply(h, f, g, axis, given(p.result_type) ? h.str(p.result_type!) : null, flag(h, p.raw), args, kw)
  }],
  agg: ['func axis **', (f, p, h) => frameAggCall(h, f, p)],
  aggregate: ['func axis **', (f, p, h) => frameAggCall(h, f, p)],
  transform: ['func axis * **', (f, p, h) => {
    const { args, kw } = unboxArgs(h, p)
    return py.obj(frameTransform(h, f, need(h, p.func, 'transform', 'func'), axisOf(h, p.axis, 'DataFrame'), args, kw))
  }],
  pipe: ['func * **', (f, p, h) => pipeCall(h, py.obj(f), p)],
  assign: ['**', (f, p, h, c) => {
    const out = f.copy()
    const items = (p.__kw as { d: { items(): [Value, Value][] } }).d.items()
    for (const [k, v] of items) out.assignColumn(h, cellOf(h, k), callableValue(v) ? h.call(v, [py.obj(out)]) : v, false)
    c.picks = { rows: null, cols: items.map(([k]) => out.columns.locate(cellOf(h, k))[0]) }
    return py.obj(out)
  }],
  astype: ['dtype copy errors', (f, p, h) => {
    const t = need(h, p.dtype, 'astype', 'dtype')
    const ignore = given(p.errors) ? h.str(p.errors!) === 'ignore' : false
    if (t.k === 'dict') {
      const miss = t.d.keys().map((k) => cellOf(h, k)).filter((c) => !f.columns.has(c))
      if (miss.length) throw h.err('KeyError', 'Only a column name can be used for the key in a dtype mappings argument.')
    }
    return py.obj(f.same(f.data.map((s) => {
      const want = t.k === 'dict' ? t.d.get(valueOf(s.name, f.columns.dt)) : t
      return want ? astypeCol(h, s, typeArg(h, want), ignore) : s.copy()
    })))
  }],
  infer_objects: ['copy', (f) => py.obj(f.same(f.data.map((s) => {
    const t = settleObject(s.values, s.dtype)
    return s.with(t.cells, t.dt)
  })))],
  copy: ['deep', (f, p, h) => py.obj(p.deep && !h.truthy(p.deep) ? f.same(f.data) : f.copy())],
  cumsum: ['axis skipna', (f, p, h) => py.obj(byAxis(f, axisOf(h, p.axis, 'DataFrame'), (s) => cumulate(h, s, 'cumsum', flag(h, p.skipna, true))))],
  cumprod: ['axis skipna', (f, p, h) => py.obj(byAxis(f, axisOf(h, p.axis, 'DataFrame'), (s) => cumulate(h, s, 'cumprod', flag(h, p.skipna, true))))],
  cummax: ['axis skipna', (f, p, h) => py.obj(byAxis(f, axisOf(h, p.axis, 'DataFrame'), (s) => cumulate(h, s, 'cummax', flag(h, p.skipna, true))))],
  cummin: ['axis skipna', (f, p, h) => py.obj(byAxis(f, axisOf(h, p.axis, 'DataFrame'), (s) => cumulate(h, s, 'cummin', flag(h, p.skipna, true))))],
  diff: ['periods axis', (f, p, h) => py.obj(byAxis(f, axisOf(h, p.axis, 'DataFrame'), (s) => diffCol(h, s, p.periods ? h.toInt(p.periods) : 1)))],
  shift: ['periods freq axis fill_value suffix', (f, p, h) => {
    refuse(h, p.freq, 'shift', 'freq')
    refuse(h, p.suffix, 'shift', 'suffix')
    const periods = p.periods ? h.toInt(p.periods) : 1
    if (axisOf(h, p.axis, 'DataFrame') === 1) {
      const fill = given(p.fill_value)
      const n = f.ncols
      return py.obj(f.same(rows(n).map((k) => {
        const from = k - periods
        if (from >= 0 && from < n) return f.data[from].with([...f.data[from].values], f.data[from].dtype, f.columns.labels[k])
        const c = fill ? cellOf(h, fill) : NaN
        return new Series(f.index.labels.map(() => c), f.index, f.columns.labels[k], fill ? inferDT([c]) : 'float64')
      })))
    }
    return py.obj(f.same(f.data.map((s) => shiftCol(h, s, periods, given(p.fill_value)))))
  }],
  pct_change: ['periods fill_method limit freq', (f, p, h) => {
    refuse(h, p.freq, 'pct_change', 'freq')
    return py.obj(f.same(f.data.map((s) => pctChange(h, s, p.periods ? h.toInt(p.periods) : 1, fillMethodArg(h, p), given(p.limit) ? h.toInt(p.limit!) : null))))
  }],
  rank: ['axis method numeric_only na_option ascending pct', (f, p, h) => {
    const how = oneOf(h, p.method, 'average', ['average', 'min', 'max', 'first', 'dense'], 'rank', 'method')
    const na = oneOf(h, p.na_option, 'keep', ['keep', 'top', 'bottom'], 'rank', 'na_option')
    const asc = flag(h, p.ascending, true)
    const pct = flag(h, p.pct)
    const src = flag(h, p.numeric_only) ? f.take(null, numericCols(f)) : f
    return py.obj(byAxis(src, axisOf(h, p.axis, 'DataFrame'), (s) => s.with(rankCol(s, how, asc, na, pct), 'float64')))
  }],
  sort_values: ['by | axis ascending inplace kind na_position ignore_index key', (f, p, h, c) => {
    const byV = need(h, p.by, 'sort_values', 'by')
    const labels = byV.k === 'list' || byV.k === 'tuple' ? cellsOf(h, byV) : [cellOf(h, byV)]
    const naFirst = given(p.na_position) ? h.str(p.na_position!) === 'first' : false
    if (axisOf(h, p.axis, 'DataFrame') === 1) {
      const keys = labels.map((l) => {
        const at = f.index.locate(l)
        if (!at.length) throw h.err('KeyError', repr(valueOf(l, 'object')))
        return f.row(at[0]).values
      })
      return finish(h, f, f.take(null, sortOrder(keys, ascList(h, p.ascending, keys.length), naFirst)), p.inplace)
    }
    const by = labels.map((l) => {
      const at = f.columns.locate(l)
      if (!at.length) throw h.err('KeyError', repr(valueOf(l, 'object')))
      return at[0]
    })
    const keyOf = (k: number) => (given(p.key) ? seriesIn(h.call(p.key!, [py.obj(f.data[k])])).values : f.data[k].cats ? f.data[k].values.map((x) => f.data[k].cats!.labels.indexOf(x)) : f.data[k].values)
    const order = sortOrder(by.map(keyOf), ascList(h, p.ascending, by.length), naFirst)
    c.picks = { rows: null, cols: by }
    const out = f.take(order, null)
    return finish(h, f, flag(h, p.ignore_index) ? new Frame(out.columns, out.data, IndexObj.range(order.length)) : out, p.inplace)
  }],
  sort_index: ['| axis level ascending inplace kind na_position sort_remaining ignore_index key', (f, p, h) => {
    refuse(h, p.level, 'sort_index', 'level')
    refuse(h, p.key, 'sort_index', 'key')
    const asc = flag(h, p.ascending, true)
    const naFirst = given(p.na_position) ? h.str(p.na_position!) === 'first' : false
    if (axisOf(h, p.axis, 'DataFrame') === 1) return finish(h, f, f.take(null, sortOrder([f.columns.labels], [asc], naFirst)), p.inplace)
    const order = sortOrder([f.index.labels], [asc], naFirst)
    const out = f.take(order, null)
    return finish(h, f, flag(h, p.ignore_index) ? new Frame(out.columns, out.data, IndexObj.range(order.length)) : out, p.inplace)
  }],
  groupby: ['by axis level as_index sort group_keys observed dropna', (f, p, h) => groupByFrame(h, f, p)],
  drop: ['labels | axis index columns level inplace errors', (f, p, h) => finish(h, f, dropFrame(h, f, p), p.inplace)],
  rename: ['mapper | index columns axis copy inplace level errors', (f, p, h) => finish(h, f, renameFrame(h, f, p), p.inplace)],
  rename_axis: ['mapper index columns axis copy inplace', (f, p, h) => {
    const axis = axisOf(h, p.axis, 'DataFrame')
    let idx = f.index
    let cols = f.columns
    if (p.mapper) {
      if (axis === 0) idx = idx.withName(given(p.mapper) ? cellOf(h, p.mapper) : null)
      else cols = cols.withName(given(p.mapper) ? cellOf(h, p.mapper) : null)
    }
    if (p.index) idx = idx.withName(given(p.index) ? cellOf(h, p.index) : null)
    if (p.columns) cols = cols.withName(given(p.columns) ? cellOf(h, p.columns) : null)
    return finish(h, f, new Frame(cols, f.data.map((s) => s.copy()), idx), p.inplace)
  }],
  set_axis: ['labels axis copy', (f, p, h) => {
    const axis = axisOf(h, p.axis, 'DataFrame')
    const ix = toIndexObj(h, need(h, p.labels, 'set_axis', 'labels'))
    const n = axis === 0 ? f.nrows : f.ncols
    if (ix.length !== n) throw h.err('ValueError', `Length mismatch: Expected axis has ${n} elements, new values have ${ix.length} elements`)
    return py.obj(axis === 0 ? new Frame(f.columns, f.data.map((s) => s.copy()), ix) : new Frame(ix, f.data.map((s) => s.copy()), f.index))
  }],
  add_prefix: ['prefix axis', (f, p, h) => {
    const pre = h.str(need(h, p.prefix, 'add_prefix', 'prefix'))
    const on = (l: Cell) => `${pre}${labelStr(l)}`
    return py.obj(axisOf(h, p.axis, 'DataFrame', 1) === 1 ? new Frame(new IndexObj(f.columns.labels.map(on), f.columns.name, 'object'), f.data.map((s) => s.copy()), f.index) : new Frame(f.columns, f.data.map((s) => s.copy()), new IndexObj(f.index.labels.map(on), f.index.name, 'object')))
  }],
  add_suffix: ['suffix axis', (f, p, h) => {
    const suf = h.str(need(h, p.suffix, 'add_suffix', 'suffix'))
    const on = (l: Cell) => `${labelStr(l)}${suf}`
    return py.obj(axisOf(h, p.axis, 'DataFrame', 1) === 1 ? new Frame(new IndexObj(f.columns.labels.map(on), f.columns.name, 'object'), f.data.map((s) => s.copy()), f.index) : new Frame(f.columns, f.data.map((s) => s.copy()), new IndexObj(f.index.labels.map(on), f.index.name, 'object')))
  }],
  reindex: ['labels index columns axis method copy level fill_value limit tolerance', (f, p, h) => {
    refuse(h, p.level, 'reindex', 'level')
    const axis = axisOf(h, p.axis, 'DataFrame')
    const idxV = given(p.index) ?? (given(p.labels) && axis === 0 ? p.labels : undefined)
    const colV = given(p.columns) ?? (given(p.labels) && axis === 1 ? p.labels : undefined)
    let out = f
    if (idxV) {
      const ix = toIndexObj(h, idxV)
      out = new Frame(out.columns, out.data.map((s) => reindexSeries(h, s, ix, p)), ix)
    }
    if (colV) {
      const cx = toIndexObj(h, colV)
      const fillv = given(p.fill_value)
      const src = out
      const data = cx.labels.map((c) => {
        const s = src.col(c)
        if (s) return s.copy()
        const fc = fillv ? cellOf(h, fillv) : NaN
        return new Series(src.index.labels.map(() => fc), src.index, c, fillv ? inferDT([fc]) : 'float64')
      })
      out = new Frame(cx.withName(cx.name ?? f.columns.name), data, src.index)
    }
    return py.obj(out === f ? f.copy() : out)
  }],
  reset_index: ['level | drop inplace col_level col_fill allow_duplicates names', (f, p, h) => {
    refuse(h, p.col_level, 'reset_index', 'col_level')
    refuse(h, p.col_fill, 'reset_index', 'col_fill')
    refuse(h, p.names, 'reset_index', 'names')
    const levels = given(p.level) ? axisLabels(h, p.level!).map((l) => (typeof l === 'number' ? l : f.index.names.indexOf(l))) : undefined
    if (levels && levels.some((j) => j < 0)) throw h.err('KeyError', `'Level ${labelStr(axisLabels(h, p.level!)[0])} not found'`)
    if (flag(h, p.drop)) {
      const rest = levels ? Array.from({ length: f.index.nlevels }, (_, i) => i).filter((j) => !levels.includes(j)) : []
      const ix = !rest.length ? IndexObj.range(f.nrows) : rest.length === 1 ? f.index.level(rest[0]) : IndexObj.fromArrays(rest.map((j) => f.index.multi!.arrays[j]), rest.map((j) => f.index.multi!.names[j]), rest.map((j) => f.index.multi!.dts[j]))
      return finish(h, f, new Frame(f.columns, f.data.map((s) => s.copy()), ix), p.inplace)
    }
    return finish(h, f, indexToColumns(h, f, levels, flag(h, p.allow_duplicates)), p.inplace)
  }],
  set_index: ['keys | drop append inplace verify_integrity', (f, p, h) => finish(h, f, setIndexFrame(h, f, p), p.inplace)],
  merge: ['right how on left_on right_on left_index right_index sort suffixes indicator validate copy', (f, p, h) => py.obj(mergeFrames(h, f, need(h, p.right, 'merge', 'right'), p, h.nameOf(py.obj(f), 'left')))],
  join: ['other on how lsuffix rsuffix sort validate', (f, p, h) => py.obj(joinFrames(h, f, p))],
  corr: ['method min_periods numeric_only', (f, p, h) => {
    const how = oneOf(h, p.method, 'pearson', ['pearson', 'spearman'], 'corr', 'method')
    const minp = given(p.min_periods) ? h.toInt(p.min_periods!) : 1
    const { cols, vals } = floatColumns(h, f, flag(h, p.numeric_only))
    return py.obj(symmetric(f, cols, vals, (x, y) => corrOf(x, y, how, minp)))
  }],
  cov: ['min_periods ddof numeric_only', (f, p, h) => {
    const minp = given(p.min_periods) ? h.toInt(p.min_periods!) : 1
    const ddof = given(p.ddof) ? h.toInt(p.ddof!) : 1
    const { cols, vals } = floatColumns(h, f, flag(h, p.numeric_only))
    const anyMissing = vals.some((c) => c.some(Number.isNaN))
    if (anyMissing) return py.obj(symmetric(f, cols, vals, (x, y) => welford(x, y, true, minp)))
    if (minp > f.nrows) return py.obj(symmetric(f, cols, vals, () => NaN))
    return py.obj(symmetric(f, cols, vals, (x, y) => npCov(x, y, ddof, false)))
  }],
  corrwith: ['other axis drop method numeric_only', (f, p, h) => {
    const how = oneOf(h, p.method, 'pearson', ['pearson', 'spearman'], 'corrwith', 'method')
    const o = need(h, p.other, 'corrwith', 'other')
    if (axisOf(h, p.axis, 'DataFrame') === 1) refuse(h, py.int(1), 'corrwith', 'axis')
    const cols = flag(h, p.numeric_only) ? numericCols(f) : f.data.map((_, k) => k)
    if (o.k === 'obj' && o.o instanceof Series) {
      const src = o.o
      const ix = unionIndex(f.index, src.index)
      const b = reindexCells(src, ix).cells.map(numCell)
      return py.obj(new Series(cols.map((k) => corrOf(reindexCells(f.data[k], ix).cells.map(numCell), b, how, 1)), f.columns.pick(cols, true), null, 'float64'))
    }
    if (!(o.k === 'obj' && o.o instanceof Frame)) throw h.err('TypeError', 'corrwith takes a Series or a DataFrame')
    const other = o.o
    const shared = f.columns.labels.filter((l, k) => cols.includes(k) && other.columns.has(l))
    const drop = flag(h, p.drop)
    const missing = [...f.columns.labels.filter((l) => !other.columns.has(l)), ...other.columns.labels.filter((l) => !f.columns.has(l))]
    const labels = drop ? shared : [...shared, ...missing]
    return py.obj(new Series(labels.map((l) => {
      const a = f.col(l)
      const b = other.col(l)
      return a && b ? corrOf(a.values.map(numCell), reindexCells(b, a.index).cells.map(numCell), how, 1) : NaN
    }), new IndexObj(labels, null, f.columns.dt), null, 'float64'))
  }],
  dot: ['other', (f, p, h) => dotFrame(h, f, need(h, p.other, 'dot', 'other'))],
  equals: ['other', (f, p) => {
    const o = p.other
    if (!(o && o.k === 'obj' && o.o instanceof Frame)) return py.bool(false)
    const other = o.o
    return py.bool(f.columns.equals(other.columns) && f.index.equals(other.index) && f.data.every((s, k) => s.dtype === other.data[k].dtype && s.values.every((c, i) => sameCell(c, other.data[k].values[i]))))
  }],
  filter: ['items like regex axis', (f, p, h) => {
    const axis = axisOf(h, p.axis, 'DataFrame', 1)
    const ix = axis === 1 ? f.columns : f.index
    const keep = filterLabels(h, ix, p)
    return py.obj(axis === 1 ? f.take(null, keep) : f.take(keep, null))
  }],
  get: ['key default', (f, p, h) => {
    const key = need(h, p.key, 'get', 'key')
    const dflt = p.default ?? py.NONE
    if (isListLike(key)) {
      const labels = cellsOf(h, key)
      if (!labels.every((l) => f.columns.has(l))) return dflt
      return py.obj(f.take(null, labels.flatMap((l) => f.columns.locate(l))))
    }
    const c = cellOf(h, key)
    if (!f.columns.has(c)) return dflt
    return f.byColumn(h, key, c)
  }],
  insert: ['loc column value allow_duplicates', (f, p, h) => {
    const label = cellOf(h, need(h, p.column, 'insert', 'column'))
    const loc = h.toInt(need(h, p.loc, 'insert', 'loc'))
    if (f.columns.has(label) && !flag(h, p.allow_duplicates)) throw h.err('ValueError', `cannot insert ${labelStr(label)}, already exists`)
    if (loc < 0 || loc > f.ncols) throw h.err('IndexError', `index ${loc} is out of bounds for axis 0 with size ${f.ncols}`)
    const scratch = new Frame(f.columns.pick([], true), [], f.index)
    scratch.assignColumn(h, label, need(h, p.value, 'insert', 'value'), false)
    const labels = [...f.columns.labels]
    labels.splice(loc, 0, label)
    const data = [...f.data]
    data.splice(loc, 0, scratch.data[0])
    f.adopt(new Frame(new IndexObj(labels, f.columns.name, commonDT([f.columns.dt, inferDT([label])])), data, scratch.index))
    return py.NONE
  }],
  pop: ['item', (f, p, h) => {
    const k = f.colPos(h, need(h, p.item, 'pop', 'item'))
    const s = f.data[k]
    f.adopt(f.take(null, rows(f.ncols).filter((j) => j !== k)))
    return py.obj(s)
  }],
  squeeze: ['axis', (f, p, h) => {
    const axis = given(p.axis) ? axisOf(h, p.axis, 'DataFrame') : null
    if ((axis === null || axis === 1) && f.ncols === 1) {
      if ((axis === null) && f.nrows === 1) return valueOf(f.data[0].values[0], f.data[0].dtype)
      return py.obj(f.data[0])
    }
    if ((axis === null || axis === 0) && f.nrows === 1) return py.obj(f.row(0))
    return py.obj(f)
  }],
  transpose: ['* copy', (f) => py.obj(transposeFrame(f))],
  items: ['', (f) => generatorOf('generator', f.columns.iter().map((c, k) => py.tuple([c, py.obj(f.data[k])])))],
  iterrows: ['', (f) => generatorOf('generator', f.index.iter().map((l, r) => py.tuple([l, py.obj(f.row(r))])))],
  itertuples: ['index name', (f, p, h) => {
    const nameV = p.name
    const indexFlag = flag(h, p.index, true)
    if (nameV && nameV.k === 'none') return generatorOf('zip', rows(f.nrows).map((r) => py.tuple([...(indexFlag ? [valueOf(f.index.labels[r], f.index.dt)] : []), ...f.data.map((s) => valueOf(s.values[r], s.dtype))])))
    return generatorOf('map', rows(f.nrows).map((r) => namedRow(f, r, indexFlag, nameV ? h.str(nameV) : 'Pandas')))
  }],
  keys: ['', (f) => py.obj(f.columns)],
  duplicated: ['subset keep', (f, p, h, c) => {
    const cols = cellsToFrameKeys(h, f, p.subset)
    const dup = duplicatedMask(rows(f.nrows).map((r) => cols.map((k) => cellKey(f.data[k].values[r])).join('\0')), keepArg(h, p.keep))
    c.picks = { rows: rows(f.nrows).filter((r) => dup[r]), cols }
    return py.obj(new Series(dup, f.index, null, 'bool'))
  }],
  drop_duplicates: ['subset | keep inplace ignore_index', (f, p, h, c) => {
    const cols = cellsToFrameKeys(h, f, p.subset)
    const dup = duplicatedMask(rows(f.nrows).map((r) => cols.map((k) => cellKey(f.data[k].values[r])).join('\0')), keepArg(h, p.keep))
    c.picks = { rows: rows(f.nrows).filter((r) => dup[r]), cols }
    const out = f.take(rows(f.nrows).filter((r) => !dup[r]), null)
    return finish(h, f, flag(h, p.ignore_index) ? new Frame(out.columns, out.data, IndexObj.range(out.nrows)) : out, p.inplace)
  }],
  nlargest: ['n columns keep', (f, p, h, c) => frameExtreme(h, f, p, 'nlargest', c)],
  nsmallest: ['n columns keep', (f, p, h, c) => frameExtreme(h, f, p, 'nsmallest', c)],
  sample: ['n frac replace weights random_state axis ignore_index', (f, p, h, c) => {
    const axis = axisOf(h, p.axis, 'DataFrame')
    const pick = sampleRows(h, axis === 0 ? f.nrows : f.ncols, { ...p, weights: p.weights && p.weights.k === 'str' ? py.obj(f.byColumnSeries(h, p.weights)) : p.weights })
    if (axis === 1) return py.obj(f.take(null, pick))
    c.picks = { rows: pick, cols: null }
    const out = f.take(pick, null)
    return py.obj(flag(h, p.ignore_index) ? new Frame(out.columns, out.data, IndexObj.range(out.nrows)) : out)
  }],
  pivot: ['columns index values', (f, p, h) => py.obj(pivotFrame(h, f, p))],
  pivot_table: ['values index columns aggfunc fill_value margins dropna margins_name observed sort', (f, p, h, c) => {
    c.api = 'pivot_table'
    return py.obj(pivotTable(h, f, p))
  }],
  melt: ['id_vars value_vars var_name value_name col_level ignore_index', (f, p, h) => py.obj(meltFrame(h, f, p))],
  stack: ['level dropna sort future_stack', (f, p, h) => py.obj(stackFrame(h, f, p))],
  unstack: ['level fill_value sort', (f, p, h) => py.obj(unstackFrame(h, f, p))],
  explode: ['column ignore_index', (f, p, h) => {
    const cols = axisLabels(h, need(h, p.column, 'explode', 'column')).map((l) => {
      const at = f.columns.locate(l)
      if (!at.length) throw h.err('KeyError', repr(valueOf(l, 'object')))
      return at[0]
    })
    const first = explodeCol(h, new Series(f.data[cols[0]].values, IndexObj.range(f.nrows), null, f.data[cols[0]].dtype), false)
    const at = first.index.labels as number[]
    const data = f.data.map((s, k) => {
      if (k === cols[0]) return first.with(first.values, 'object', s.name)
      if (!cols.includes(k)) return s.pick(at)
      return explodeCol(h, new Series(s.values, IndexObj.range(f.nrows), s.name, s.dtype), false)
    })
    const ix = flag(h, p.ignore_index) ? IndexObj.range(at.length) : f.index.pick(at)
    return py.obj(new Frame(f.columns, data.map((s) => new Series(s.values, ix, s.name, s.dtype)), ix))
  }],
  select_dtypes: ['include exclude', (f, p, h) => {
    const inc = given(p.include)
    const exc = given(p.exclude)
    if (!inc && !exc) throw h.err('ValueError', 'at least one of include or exclude must be nonempty')
    const list = (v: Value) => (isListLike(v) ? h.iterate(v) : [v]).map((x) => typeArg(h, x))
    const incs = inc ? list(inc) : null
    const excs = exc ? list(exc) : []
    const overlap = incs?.filter((t) => excs.includes(t)) ?? []
    if (overlap.length) throw h.err('ValueError', `include and exclude overlap on frozenset({${overlap.map((t) => `'${t}'`).join(', ')}})`)
    const keep = rows(f.ncols).filter((k) => (!incs || incs.some((t) => dtypeMatches(f.data[k], t))) && !excs.some((t) => dtypeMatches(f.data[k], t)))
    return py.obj(f.take(null, keep))
  }],
  query: ['expr inplace', (f, p, h, c) => {
    const expr = h.str(need(h, p.expr, 'query', 'expr'))
    const res = queryValue(h, f, parseQuery(h, expr))
    if (!(res.k === 'obj' && res.o instanceof Series)) throw h.err('ValueError', 'Expected a boolean expression')
    const out = res.o.dtype === 'bool' ? f.getItem(res, h) : f.getItem(res, h)
    c.picks = { rows: res.o.values.flatMap((b, i) => (b ? [i] : [])), cols: null }
    return finish(h, f, frameIn(out), p.inplace)
  }],
  eval: ['expr inplace', (f, p, h) => {
    const expr = h.str(need(h, p.expr, 'eval', 'expr'))
    const assign = /^\s*`?([A-Za-z_][A-Za-z0-9_ ]*)`?\s*=(?!=)(.*)$/s.exec(expr)
    if (assign) {
      const out = f.copy()
      out.assignColumn(h, assign[1].trim(), queryValue(h, f, parseQuery(h, assign[2])), false)
      return finish(h, f, out, p.inplace)
    }
    return queryValue(h, f, parseQuery(h, expr))
  }],
  to_dict: ['orient into index', (f, p, h) => {
    refuseInto(h, p)
    return toDictFrame(h, f, p.orient, flag(h, p.index, true))
  }],
  to_string: ['buf columns col_space header index na_rep formatters float_format sparsify index_names justify max_rows max_cols show_dimensions decimal line_width min_rows max_colwidth encoding', (f, p, h) => {
    for (const k of ['buf', 'formatters', 'sparsify', 'index_names', 'decimal', 'max_colwidth', 'encoding']) refuse(h, p[k], 'to_string', k)
    const src = given(p.columns) ? f.take(null, cellsToFrameKeys(h, f, p.columns)) : f
    const header = p.header ? (isListLike(p.header) ? cellsOf(h, p.header).map(labelStr) : h.truthy(p.header)) : true
    const just = oneOf(h, p.justify, 'right', ['left', 'right', 'center'], 'to_string', 'justify') as 'left' | 'right' | 'center'
    return py.str(frameToString(src, { maxRows: given(p.max_rows) ? h.toInt(p.max_rows!) : null, minRows: given(p.min_rows) ? h.toInt(p.min_rows!) : null, maxCols: given(p.max_cols) ? h.toInt(p.max_cols!) : null, lineWidth: given(p.line_width) ? h.toInt(p.line_width!) : null, index: flag(h, p.index, true), header, showDims: flag(h, p.show_dimensions), naRep: given(p.na_rep) ? h.str(p.na_rep!) : undefined, floatFormat: floatFormatArg(h, p.float_format), colSpace: given(p.col_space) ? h.toInt(p.col_space!) : undefined, justifyHeader: just }))
  }],
  to_csv: ['path_or_buf sep na_rep float_format columns header index index_label mode encoding compression quoting quotechar lineterminator', (f, p, h) => frameToCsv(h, f, p)],
  to_json: ['path_or_buf orient date_format double_precision force_ascii date_unit default_handler lines compression index indent', (f, p, h) => frameToJson(h, f, p)],
  to_records: ['index column_dtypes index_dtypes', (f, p, h) => {
    refuse(h, p.column_dtypes, 'to_records', 'column_dtypes')
    refuse(h, p.index_dtypes, 'to_records', 'index_dtypes')
    const useIndex = flag(h, p.index, true)
    const names = [...(useIndex ? [f.index.name === null ? 'index' : labelStr(f.index.name)] : []), ...f.columns.labels.map(labelStr)]
    const cols = [...(useIndex ? [{ cells: f.index.labels, dt: f.index.dt }] : []), ...f.data.map((s) => ({ cells: s.values, dt: s.dtype }))]
    return py.obj(new RecArray(f, names, cols))
  }],
  update: ['other join overwrite filter_func errors', (f, p, h) => {
    oneOf(h, p.join, 'left', ['left'], 'update', 'join')
    refuse(h, p.filter_func, 'update', 'filter_func')
    if (given(p.errors) && h.str(p.errors!) !== 'ignore') refuse(h, p.errors, 'update', 'errors')
    const overwrite = flag(h, p.overwrite, true)
    const o = need(h, p.other, 'update', 'other')
    const other = o.k === 'obj' && o.o instanceof Frame ? o.o : o.k === 'obj' && o.o instanceof Series ? new Frame([o.o.name ?? 0], [o.o], o.o.index) : null
    if (!other) throw h.err('TypeError', 'update takes a DataFrame or a Series')
    f.data.forEach((s) => {
      const src = other.col(s.name)
      if (!src) return
      const rs: number[] = []
      const cells: Cell[] = []
      f.index.labels.forEach((l, r) => {
        const at = other.index.locate(l)
        if (!at.length || isMissing(src.values[at[0]]) || (!overwrite && !isMissing(s.values[r]))) return
        rs.push(r)
        cells.push(src.values[at[0]])
      })
      if (rs.length) s.store(rs, cells)
    })
    return py.NONE
  }],
  combine_first: ['other', (f, p, h) => {
    const o = frameIn(need(h, p.other, 'combine_first', 'other'))
    const ix = unionIndex(f.index, o.index)
    const cols = unionIndex(f.columns, o.columns)
    return py.obj(new Frame(cols, cols.labels.map((c) => {
      const a = f.col(c) ? reindexCells(f.col(c)!, ix) : null
      const b = o.col(c) ? reindexCells(o.col(c)!, ix) : null
      const cells = ix.labels.map((_, i) => {
        const x = a ? a.cells[i] : NaN
        return isMissing(x) && b ? b.cells[i] : x
      })
      return new Series(cells, ix, c, commonDT([...(a ? [a.dt] : []), ...(b ? [b.dt] : [])]))
    }), ix))
  }],
  rolling: ['window min_periods center win_type on closed step method axis', (f, p, h) => windowOf(h, f, p, 'rolling')],
  expanding: ['min_periods axis method', (f, p, h) => windowOf(h, f, p, 'expanding')],
  first_valid_index: ['', (f) => {
    const r = rows(f.nrows).find((i) => f.data.some((s) => !isMissing(s.values[i])))
    return r === undefined ? py.NONE : valueOf(f.index.labels[r], f.index.dt)
  }],
  last_valid_index: ['', (f) => {
    const r = rows(f.nrows).reverse().find((i) => f.data.some((s) => !isMissing(s.values[i])))
    return r === undefined ? py.NONE : valueOf(f.index.labels[r], f.index.dt)
  }],
  droplevel: ['level axis', (f, p, h) => {
    const axis = axisOf(h, p.axis, 'DataFrame')
    const out = dropIndexLevel(h, axis === 0 ? f.index : f.columns, need(h, p.level, 'droplevel', 'level'))
    return py.obj(axis === 0 ? new Frame(f.columns, f.data.map((s) => s.copy()), out) : new Frame(out, f.data.map((s) => s.copy()), f.index))
  }],
  xs: ['key axis level drop_level', (f, p, h) => {
    const key = need(h, p.key, 'xs', 'key')
    const axis = axisOf(h, p.axis, 'DataFrame')
    if (given(p.level)) {
      const { rows: at, rest } = crossSection(h, axis === 0 ? f.index : f.columns, key, p)
      const t = axis === 0 ? f.take(at, null) : f.take(null, at)
      return py.obj(axis === 0 ? new Frame(t.columns, t.data, rest) : new Frame(rest, t.data, t.index))
    }
    if (axis === 1) return f.byColumn(h, key, cellOf(h, key))
    return frameIndexer(f, 'loc').getItem(key, h)
  }],
}

/** the index without a level */
const dropIndexLevel = (h: Host, ix: IndexObj, level: Value): IndexObj => {
  if (!ix.multi) throw h.err('ValueError', 'Cannot remove 1 levels from an index with 1 levels: at least one level must be left.')
  const lv = axisLabels(h, level).map((l) => (typeof l === 'number' ? (l < 0 ? l + ix.nlevels : l) : ix.names.indexOf(l)))
  const keep = Array.from({ length: ix.nlevels }, (_, i) => i).filter((j) => !lv.includes(j))
  return keep.length === 1 ? ix.level(keep[0]) : IndexObj.fromArrays(keep.map((j) => ix.multi!.arrays[j]), keep.map((j) => ix.multi!.names[j]), keep.map((j) => ix.multi!.dts[j]))
}

/** xs(key, level=…): the rows where that level holds key, and the index they keep */
const crossSection = (h: Host, ix: IndexObj, key: Value, p: Params): { rows: number[]; rest: IndexObj } => {
  if (!ix.multi) throw h.err('TypeError', 'Index must be a MultiIndex')
  const l = cellOf(h, p.level!)
  const j = typeof l === 'number' ? (l < 0 ? l + ix.nlevels : l) : ix.names.indexOf(l)
  if (j < 0) throw h.err('KeyError', repr(py.str(`Level ${labelStr(l)} not found`)))
  const k = cellOf(h, key)
  const at = rows(ix.length).filter((r) => ix.multi!.arrays[j][r] === k)
  if (!at.length) throw h.err('KeyError', repr(key))
  const picked = ix.pick(at)
  return { rows: at, rest: flag(h, p.drop_level, true) ? dropIndexLevel(h, picked, p.level!) : picked }
}

const frameAggCall = (h: Host, f: Frame, p: Params): Value => {
  const { kw } = unboxArgs(h, p)
  const named = p.func ? {} : kw
  const axis = axisOf(h, p.axis, 'DataFrame')
  if (!p.func && !Object.keys(named).length) throw h.err('TypeError', 'Must provide \'func\' or tuples of \'(column, aggfunc).')
  return frameAgg(h, f, p.func ?? py.NONE, axis, named)
}

const ascList = (h: Host, v: Value | undefined, n: number): boolean[] => {
  if (!v) return Array.from({ length: n }, () => true)
  if (v.k === 'list' || v.k === 'tuple') {
    if (v.items.length !== n) throw h.err('ValueError', `Length of ascending (${v.items.length}) != length of by (${n})`)
    return v.items.map((x) => h.truthy(x))
  }
  return Array.from({ length: n }, () => h.truthy(v))
}

const frameAnyAll = (h: Host, f: Frame, how: 'any' | 'all', p: Params): Value => {
  const skipna = flag(h, p.skipna, true)
  const cols = flag(h, p.bool_only) ? rows(f.ncols).filter((k) => f.data[k].dtype === 'bool') : rows(f.ncols)
  if (p.axis && p.axis.k === 'none') {
    const vals = cols.map((k) => anyAll(h, f.data[k], how, skipna))
    return py.bool(how === 'any' ? vals.some(Boolean) : vals.every(Boolean))
  }
  if (axisOf(h, p.axis, 'DataFrame') === 1) return py.obj(new Series(rows(f.nrows).map((r) => anyAll(h, f.row(r, cols), how, skipna)), f.index, null, 'bool'))
  return py.obj(new Series(cols.map((k) => anyAll(h, f.data[k], how, skipna)), f.columns.pick(cols, true), null, 'bool'))
}

const frameIdx = (h: Host, f: Frame, how: 'max' | 'min', p: Params): Value => {
  const skipna = flag(h, p.skipna, true)
  const cols = flag(h, p.numeric_only) ? numericCols(f) : rows(f.ncols)
  if (axisOf(h, p.axis, 'DataFrame') === 1) {
    const t = inferValues(h, rows(f.nrows).map((r) => {
      const row = f.row(r, cols)
      return idxOf(h, row, how, skipna)
    }))
    return py.obj(new Series(t.cells, f.index, null, t.dt))
  }
  const t = inferValues(h, cols.map((k) => idxOf(h, f.data[k], how, skipna)))
  return py.obj(new Series(t.cells, f.columns.pick(cols, true), null, t.dt))
}

const frameExtreme = (h: Host, f: Frame, p: Params, name: string, c: Ctx): Value => {
  oneOf(h, p.keep, 'first', ['first'], name, 'keep')
  const n = h.toInt(need(h, p.n, name, 'n'))
  const by = cellsToFrameKeys(h, f, need(h, p.columns, name, 'columns'))
  for (const k of by) if (f.data[k].dtype === 'object') throw h.err('TypeError', `Column ${reprCell(f.columns.labels[k])} has dtype object, cannot use method '${name}' with this dtype`)
  const order = sortOrder(by.map((k) => f.data[k].values), by.map(() => name === 'nsmallest')).slice(0, Math.max(0, n))
  c.picks = { rows: order, cols: by }
  return py.obj(f.take(order, null))
}

const filterLabels = (h: Host, ix: IndexObj, p: Params): number[] => {
  const nGiven = [p.items, p.like, p.regex].filter((x) => given(x)).length
  if (nGiven > 1) throw h.err('TypeError', 'Keyword arguments `items`, `like`, or `regex` are mutually exclusive')
  if (!nGiven) throw h.err('TypeError', 'Must pass either `items`, `like`, or `regex`')
  if (given(p.items)) return cellsOf(h, p.items!).flatMap((l) => ix.locate(l))
  if (given(p.like)) {
    const w = h.str(p.like!)
    return rows(ix.length).filter((i) => labelStr(ix.labels[i]).includes(w))
  }
  const re = pyRegex(h, h.str(p.regex!))
  return rows(ix.length).filter((i) => re.test(labelStr(ix.labels[i])))
}

const frameArray = (h: Host, f: Frame, p: Params): Value => {
  const src = given(p.dtype) ? f.same(f.data.map((s) => astypeCol(h, s, typeArg(h, p.dtype!)))) : f
  const anyObj = src.data.some((s) => s.dtype === 'object' || s.cats) || (src.data.some((s) => s.dtype === 'bool') && src.data.some((s) => s.dtype !== 'bool'))
  if (anyObj) {
    const cells: Cell[] = []
    for (let r = 0; r < src.nrows; r++) for (const s of src.data) cells.push(s.dtype === 'float64' ? boxFloat(s.values[r] as number) : s.values[r])
    return py.obj(new ObjArray(cells, [src.nrows, src.ncols]))
  }
  if (given(p.na_value)) {
    const nv = h.num(p.na_value!)
    const a = src.toArray()
    return py.arr(NDArray.create(a.values().map((x) => (Number.isNaN(x) ? nv : x)), a.shape, a.dtype))
  }
  return py.arr(src.toArray())
}

const frameInfo = (h: Host, f: Frame, showCounts: boolean, showMem: boolean): Value => {
  const lines = [`<class 'pandas.core.frame.DataFrame'>`, indexInfoLine(f.index)]
  if (!f.ncols) {
    lines.push(`Empty DataFrame`)
    h.print(`${lines.join('\n')}\n`)
    return py.NONE
  }
  lines.push(`Data columns (total ${f.ncols} columns):`)
  const names = f.columns.labels.map(labelStr)
  const dts = f.data.map((s) => dtName(s))
  const cw = Math.max(6, ...names.map((c) => c.length))
  const dw = Math.max(5, ...dts.map((d) => d.length))
  const idw = Math.max(2, String(f.ncols - 1).length)
  const header = showCounts ? ['Non-Null Count', '--------------'] : []
  const lead = ` ${'#'.padEnd(idw)}  `
  lines.push(`${lead}${'Column'.padEnd(cw)}${showCounts ? `  ${header[0]}` : ''}  ${'Dtype'.padEnd(dw)}`)
  lines.push(`${'-'.repeat(idw + 1)}  ${'------'.padEnd(cw)}${showCounts ? `  ${header[1]}` : ''}  ${'-----'.padEnd(dw)}`)
  f.data.forEach((s, k) => {
    const count = `${s.values.filter((c) => !isMissing(c)).length} non-null`
    lines.push(` ${String(k).padEnd(idw)}  ${names[k].padEnd(cw)}${showCounts ? `  ${count.padEnd(14)}` : ''}  ${dts[k].padEnd(dw)}`)
  })
  const counts = new Map<string, number>()
  dts.forEach((d) => counts.set(d, (counts.get(d) ?? 0) + 1))
  lines.push(`dtypes: ${[...counts].sort(([a], [b]) => (a < b ? -1 : 1)).map(([d, n]) => `${d}(${n})`).join(', ')}`)
  if (showMem) {
    const bytes = memOfIndex(f.index) + f.data.reduce((a, s) => a + memOfSeries(s), 0)
    lines.push(`memory usage: ${sizeFmt(bytes, f.data.some((s) => s.dtype === 'object' && !s.cats) || f.index.dt === 'object' ? '+' : '')}\n`)
  }
  h.print(lines.join('\n'))
  return py.NONE
}

const frameValueCounts = (h: Host, f: Frame, p: Params): Series => {
  const cols = cellsToFrameKeys(h, f, p.subset)
  const dropna = flag(h, p.dropna, true)
  const normalize = flag(h, p.normalize)
  const counts = new Map<string, { r: number; n: number }>()
  for (let r = 0; r < f.nrows; r++) {
    if (dropna && cols.some((k) => isMissing(f.data[k].values[r]))) continue
    const key = cols.map((k) => cellKey(f.data[k].values[r])).join('\0')
    const e = counts.get(key)
    if (e) e.n++
    else counts.set(key, { r, n: 1 })
  }
  let entries = [...counts.values()]
  // groupby sorts the keys first; a stable sort by count then keeps that order among ties
  entries = sortOrder(cols.map((k) => entries.map((e) => f.data[k].values[e.r])), cols.map(() => true)).map((i) => entries[i])
  if (flag(h, p.sort, true)) entries = sortOrder([entries.map((e) => e.n)], [flag(h, p.ascending)]).map((i) => entries[i])
  const total = entries.reduce((a, e) => a + e.n, 0)
  const arrays = cols.map((k) => entries.map((e) => f.data[k].values[e.r]))
  const index = cols.length === 1 ? IndexObj.fromArrays(arrays, [f.columns.labels[cols[0]]], [f.data[cols[0]].dtype]) : IndexObj.fromArrays(arrays, cols.map((k) => f.columns.labels[k]), cols.map((k) => f.data[k].dtype))
  return new Series(entries.map((e) => (normalize ? e.n / total : e.n)), index, normalize ? 'proportion' : 'count', normalize ? 'float64' : 'int64')
}

const dotFrame = (h: Host, f: Frame, o: Value): Value => {
  const a = f.data.map((s) => s.values.map(numCell))
  const dt = (extra: DT[]) => {
    const all = [...f.data.map((s) => s.dtype), ...extra]
    return all.includes('float64') ? 'float64' : 'int64'
  }
  if (o.k === 'obj' && o.o instanceof Frame) {
    const other = o.o
    const shared = unionIndex(f.columns, other.index)
    if (shared.length > f.ncols || shared.length > other.nrows) throw h.err('ValueError', 'matrices are not aligned')
    return py.obj(new Frame(other.columns, other.data.map((c) => {
      const vals = rows(f.nrows).map((r) => f.columns.labels.reduce<number>((acc, l, k) => acc + a[k][r] * numCell(c.values[other.index.locate(l)[0]]), 0))
      return new Series(vals, f.index, c.name, dt([c.dtype]))
    }), f.index))
  }
  if (o.k === 'obj' && o.o instanceof Series) {
    const s = o.o
    const shared = unionIndex(f.columns, s.index)
    if (shared.length > f.ncols || shared.length > s.length) throw h.err('ValueError', 'matrices are not aligned')
    return py.obj(new Series(rows(f.nrows).map((r) => f.columns.labels.reduce<number>((acc, l, k) => acc + a[k][r] * numCell(s.values[s.index.locate(l)[0]]), 0)), f.index, null, dt([s.dtype])))
  }
  const dotFn = npFunc(h, 'dot')!
  const arr = py.arr(f.toArray())
  const res = h.call(dotFn, [arr, o])
  if (res.k === 'array' && res.a.ndim === 2) {
    const [r, c] = res.a.shape
    return py.obj(new Frame(IndexObj.range(c), rows(c).map((j) => new Series(rows(r).map((i) => res.a.values()[i * c + j]), f.index, j, res.a.dtype === 'float64' ? 'float64' : 'int64')), f.index))
  }
  if (res.k === 'array') return py.obj(new Series(res.a.values(), f.index, null, res.a.dtype === 'float64' ? 'float64' : 'int64'))
  return res
}

// ---------------------------------------------------------------- DataFrame attributes

export const frameAttr = (f: Frame, name: string, h: Host): Value | undefined => {
  if (has(FRAME, name)) return method(f, 'DataFrame', name, FRAME[name], h)
  if (has(REDUCERS, name)) {
    const [how, spec] = REDUCERS[name]
    return method(f, 'DataFrame', name, [spec, (x, p, hh) => {
      if ((how === 'std' || how === 'var') && !p.ddof) noteOnce(hh, 'pandas_ddof')
      return frameReduce(hh, x, how, p)
    }], h)
  }
  if (has(FLEX, name)) {
    const [op, reflected] = FLEX[name]
    return method(f, 'DataFrame', name, ['other axis level fill_value', (x, p, hh) => {
      refuse(hh, p.level, name, 'level')
      return py.obj(frameOp(hh, x, op, need(hh, p.other, name, 'other'), reflected, { axis: axisOf(hh, p.axis, 'DataFrame', 1), fill: given(p.fill_value) ? cellOf(hh, p.fill_value!) : undefined, flex: true }))
    }], h)
  }
  if (has(FLEX_CMP, name)) {
    return method(f, 'DataFrame', name, ['other axis level', (x, p, hh) => {
      refuse(hh, p.level, name, 'level')
      return py.obj(frameOp(hh, x, FLEX_CMP[name], need(hh, p.other, name, 'other'), false, { axis: axisOf(hh, p.axis, 'DataFrame', 1), flex: true }))
    }], h)
  }
  switch (name) {
    case 'shape': return py.tuple([py.int(f.nrows), py.int(f.ncols)])
    case 'columns': return py.obj(f.columns)
    case 'index': return py.obj(f.index)
    case 'axes': return py.list([py.obj(f.index), py.obj(f.columns)])
    case 'dtypes': return py.obj(new Series(f.data.map((s) => dtName(s)), f.columns, null, 'object'))
    case 'size': return py.int(f.nrows * f.ncols)
    case 'ndim': return py.int(2)
    case 'empty': return py.bool(f.nrows === 0 || f.ncols === 0)
    case 'T': return py.obj(transposeFrame(f))
    case 'values': return frameArray(h, f, {})
    case 'loc': case 'iloc': case 'at': case 'iat': return py.obj(frameIndexer(f, name))
    case 'attrs': return { k: 'dict', d: dictFromPairs([]) }
    case 'from_dict': case 'from_records': return (h.getAttr(h.getAttr(h.module('pandas'), 'DataFrame'), name))
  }
  return undefined
}

// ---------------------------------------------------------------- numpy functions on pandas objects

const NP_UFUNCS = new Set(['sqrt', 'exp', 'exp2', 'expm1', 'log', 'log2', 'log10', 'log1p', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh', 'abs', 'absolute', 'fabs', 'floor', 'ceil', 'rint', 'trunc', 'sign', 'square', 'negative', 'positive', 'cbrt', 'reciprocal', 'degrees', 'radians', 'deg2rad', 'rad2deg', 'power', 'float_power', 'add', 'subtract', 'multiply', 'divide', 'true_divide', 'floor_divide', 'mod', 'remainder', 'maximum', 'minimum', 'fmax', 'fmin', 'isnan', 'isfinite', 'isinf', 'logical_and', 'logical_or', 'logical_xor', 'logical_not', 'greater', 'greater_equal', 'less', 'less_equal', 'equal', 'not_equal'])
/** numpy forwards these to the object's own method, so skipna and pandas' index apply */
const NP_METHODS: Record<string, string> = { sum: 'sum', mean: 'mean', std: 'std', var: 'var', max: 'max', amax: 'max', min: 'min', amin: 'min', prod: 'prod', product: 'prod', any: 'any', all: 'all', cumsum: 'cumsum', cumprod: 'cumprod', argmax: 'argmax', argmin: 'argmin', argsort: 'argsort', round: 'round', around: 'round', clip: 'clip' }

const arrayResultSeries = (h: Host, res: Value, s: Series): Series => {
  if (res.k !== 'array') return s.with(s.values.map(() => cellOf(h, res)), inferDT([cellOf(h, res)]))
  const t = typedCells(h, res)
  const out = s.with(t.cells, t.dt)
  out.width = t.width ?? null
  return out
}

const forwardToMethod = (self: Series | Frame, name: string, args: Value[], kw: Record<string, Value>, h: Host): Value | undefined => {
  const m = NP_METHODS[name]
  if (!m || args[0].k !== 'obj' || args[0].o !== self) return undefined
  if (m === 'argmax' || m === 'argmin' || m === 'argsort') {
    if (self instanceof Frame) return undefined
    return h.call(self.getAttr(m, h)!, [])
  }
  const attr = self.getAttr(m, h)!
  if (m === 'round') return h.call(attr, args.slice(1, 2), kw.decimals ? { decimals: kw.decimals } : {})
  if (m === 'clip') return h.call(attr, [args[1] ?? kw.a_min ?? kw.min ?? py.NONE, args[2] ?? kw.a_max ?? kw.max ?? py.NONE])
  const out: Record<string, Value> = {}
  if (m === 'std' || m === 'var') out.ddof = kw.ddof ?? py.int(0)
  const axis = args[1] ?? kw.axis
  if (axis) out.axis = axis
  else if (self instanceof Frame && m !== 'cumsum' && m !== 'cumprod') out.axis = py.NONE
  return h.call(attr, [], out)
}

const plainArgs = (args: Value[]) => args.map((a) => (a.k === 'obj' && (a.o instanceof Series || a.o instanceof Frame) ? py.arr(a.o.toArray()) : a))
/** these see plain arrays and give plain results */
const NP_ARRAY_ONLY = new Set(['dot', 'matmul', 'inner', 'outer', 'vdot', 'cross'])
/** a two-argument ufunc with no pandas operator behind it works on the table as one array, so its columns share a dtype */
const NP_COMMON = new Set(['maximum', 'minimum', 'fmax', 'fmin', 'logical_and', 'logical_or', 'logical_xor'])

/** np.where(mask, 'yes', 'no'): text the numeric arrays cannot hold */
const whereText = (h: Host, args: Value[]): Value | undefined => {
  if (args.length !== 3 || !args.slice(1).some((a) => a.k === 'str' || (a.k === 'obj' && a.o instanceof Series && a.o.dtype === 'object'))) return undefined
  const cond = cellsOf(h, args[0])
  const at = (v: Value, i: number): string => {
    const c = isListLike(v) ? cellsOf(h, v)[i] : cellOf(h, v)
    return c === null ? 'None' : typeof c === 'number' && Number.isNaN(c) ? 'nan' : typeof c === 'number' && v.k === 'float' ? repr(v) : labelStr(c)
  }
  return py.obj(new ObjArray(cond.map((c, i) => at(c ? args[1] : args[2], i)), [cond.length], true))
}

export const seriesNp = (s: Series, name: string, args: Value[], kw: Record<string, Value>, h: Host, impl: (a: Value[], k: Record<string, Value>) => Value): Value | undefined => {
  if (name === 'where') return whereText(h, args)
  if (s.dtype === 'object' && args[0].k === 'obj' && args[0].o === s) {
    const text = new ObjArray([...s.values])
    const res = text.npCall(name, [py.obj(text), ...args.slice(1)], kw, h)
    if (res) return res
  }
  const fw = forwardToMethod(s, name, args, kw, h)
  if (fw) return fw
  if (NP_ARRAY_ONLY.has(name)) return impl(plainArgs(args), kw)
  if (!NP_UFUNCS.has(name)) return undefined
  const plain = args.map((a) => (a.k === 'obj' && a.o instanceof Series ? py.arr(a.o.toArray()) : a))
  const first = args.find((a) => a.k === 'obj' && a.o instanceof Series)
  return py.obj(arrayResultSeries(h, impl(plain, kw), seriesIn(first!)))
}

export const frameNp = (f: Frame, name: string, args: Value[], kw: Record<string, Value>, h: Host, impl: (a: Value[], k: Record<string, Value>) => Value): Value | undefined => {
  if ((name === 'array' || name === 'asarray') && args[0].k === 'obj' && args[0].o === f && f.data.some((s) => s.dtype === 'object')) return frameArray(h, f, {})
  const fw = forwardToMethod(f, name, args, kw, h)
  if (fw) return fw
  if (NP_ARRAY_ONLY.has(name)) return impl(plainArgs(args), kw)
  if (!NP_UFUNCS.has(name)) return undefined
  const out = f.same(f.data.map((s, k) => {
    const plain = args.map((a) => {
      if (a.k === 'obj' && a.o instanceof Frame) {
        const c = a.o === f ? s : a.o.col(f.columns.labels[k])
        if (!c) throw h.err('ValueError', 'cannot align DataFrames of different columns in a numpy function in this sandbox')
        return py.arr(c.toArray())
      }
      return a.k === 'obj' && a.o instanceof Series ? py.arr(a.o.toArray()) : a
    })
    return arrayResultSeries(h, impl(plain, kw), s)
  }))
  if (NP_COMMON.has(name)) {
    const dt = commonDT(out.data.map((s) => s.dtype))
    return py.obj(out.same(out.data.map((s) => s.with(convertCells(s.values, s.dtype, dt), dt))))
  }
  return py.obj(out)
}

// ---------------------------------------------------------------- Index attributes

export const indexAttr = (ix: IndexObj, name: string, h: Host): Value | undefined => {
  const m = (call: (args: Value[], kw: Record<string, Value>) => Value): Value => fn(name, call)
  const asSeries = () => ix.asSeries()
  const flatArgs = (a: Value[], kw: Record<string, Value>, key: string) => kw[key] ?? a[0]
  switch (name) {
    case 'tolist': case 'to_list': return m(() => py.list(ix.iter()))
    case 'name': return ix.name === null ? py.NONE : valueOf(ix.name, 'object')
    case 'names': return py.list(ix.names.map((x) => valueOf(x, 'object')))
    case 'nlevels': return py.int(ix.nlevels)
    case 'levels': {
      if (!ix.multi) return undefined
      return py.list(ix.multi.arrays.map((a, j) => {
        const u = uniqueCells(a)
        return py.obj(new IndexObj(sortOrder([u], [true]).map((i) => u[i]), ix.multi!.names[j], ix.multi!.dts[j]))
      }))
    }
    case 'values': return arrayOf(ix.labels, ix.dt)
    case 'array': return arrayOf(ix.labels, ix.dt)
    case 'to_numpy': return m(() => arrayOf(ix.labels, ix.dt))
    case 'dtype': return py.obj(new DtypeObj(ix.dt))
    case 'size': return py.int(ix.length)
    case 'shape': return py.tuple([py.int(ix.length)])
    case 'ndim': return py.int(1)
    case 'empty': return py.bool(ix.length === 0)
    case 'T': return py.obj(ix)
    case 'is_unique': return py.bool(ix.isUnique)
    case 'hasnans': return py.bool(ix.labels.some(isMissing))
    case 'is_monotonic_increasing': return py.bool(monotonic(ix.labels, true))
    case 'is_monotonic_decreasing': return py.bool(monotonic(ix.labels, false))
    case 'str': return py.obj(indexStrAccessor(h, ix))
    case 'copy': return m((_a, kw) => py.obj(given(kw.name) ? ix.withName(cellOf(h, kw.name)) : ix))
    case 'rename': return m((a, kw) => py.obj(ix.withName(cellOf(h, flatArgs(a, kw, 'name') ?? py.NONE))))
    case 'set_names': return m((a, kw) => py.obj(ix.withName(cellOf(h, flatArgs(a, kw, 'names') ?? py.NONE))))
    case 'astype': return m((a, kw) => {
      const s = astypeCol(h, asSeries(), typeArg(h, flatArgs(a, kw, 'dtype')))
      return py.obj(new IndexObj(s.values, ix.name, s.dtype))
    })
    case 'map': return m((a) => {
      const res = seriesIn(h.call(h.getAttr(py.obj(asSeries()), 'map'), [a[0]]))
      return py.obj(new IndexObj(res.values, ix.name, res.dtype))
    })
    case 'isin': return m((a, kw) => py.arr(isinCol(h, asSeries(), flatArgs(a, kw, 'values')).toArray()))
    case 'isna': case 'isnull': return m(() => py.arr(NDArray.create(ix.labels.map((c) => (isMissing(c) ? 1 : 0)), [ix.length], 'bool')))
    case 'notna': case 'notnull': return m(() => py.arr(NDArray.create(ix.labels.map((c) => (isMissing(c) ? 0 : 1)), [ix.length], 'bool')))
    case 'unique': return m(() => {
      if (ix.multi) {
        const seen = new Set<string>()
        const keep = rows(ix.length).filter((i) => {
          const k = labelStr(ix.labels[i])
          if (seen.has(k)) return false
          seen.add(k)
          return true
        })
        return py.obj(ix.pick(keep))
      }
      return py.obj(new IndexObj(uniqueCells(ix.labels), ix.name, ix.dt))
    })
    case 'nunique': return m(() => py.int(uniqueCells(ix.labels.filter((c) => !isMissing(c))).length))
    case 'duplicated': return m((a, kw) => py.arr(NDArray.create(duplicatedMask(ix.labels.map(cellKey), keepArg(h, flatArgs(a, kw, 'keep'))).map(Number), [ix.length], 'bool')))
    case 'drop_duplicates': return m((a, kw) => {
      const dup = duplicatedMask(ix.labels.map(cellKey), keepArg(h, flatArgs(a, kw, 'keep')))
      return py.obj(ix.pick(rows(ix.length).filter((i) => !dup[i])))
    })
    case 'sort_values': return m((_a, kw) => py.obj(ix.pick(sortOrder([ix.labels], [flag(h, kw.ascending, true)]))))
    case 'get_loc': return m((a, kw) => {
      const v = flatArgs(a, kw, 'key')
      const at = ix.locate(cellOf(h, v))
      if (!at.length) throw h.err('KeyError', repr(v))
      if (at.length === 1) return py.int(at[0])
      return py.arr(NDArray.create(rows(ix.length).map((i) => (at.includes(i) ? 1 : 0)), [ix.length], 'bool'))
    })
    case 'get_level_values': return m((a, kw) => {
      const l = cellOf(h, flatArgs(a, kw, 'level'))
      const j = typeof l === 'number' ? (l < 0 ? l + ix.nlevels : l) : ix.names.indexOf(l)
      if (j < 0 || j >= ix.nlevels) throw h.err('KeyError', `'Level ${labelStr(l)} not found'`)
      return py.obj(ix.level(j))
    })
    case 'to_flat_index': return m(() => py.obj(new IndexObj(ix.labels, null, 'object')))
    case 'to_series': return m(() => py.obj(new Series([...ix.labels], ix, ix.name, ix.dt)))
    case 'to_frame': return m((_a, kw) => {
      const nameV = flag(h, kw.index, true) ? ix : IndexObj.range(ix.length)
      return py.obj(new Frame([given(kw.name) ? cellOf(h, kw.name) : ix.name ?? 0], [new Series([...ix.labels], nameV, null, ix.dt)], nameV))
    })
    case 'append': return m((a) => {
      const others = a[0].k === 'list' ? a[0].items : [a[0]]
      const all = [ix, ...others.map((o) => toIndexObj(h, o))]
      return py.obj(new IndexObj(all.flatMap((x) => x.labels), all.every((x) => x.name === ix.name) ? ix.name : null, commonDT(all.map((x) => x.dt))))
    })
    case 'union': return m((a) => py.obj(unionIndex(ix, sameNamed(h, ix, a[0]))))
    case 'intersection': return m((a) => {
      const o = sameNamed(h, ix, a[0])
      const u = uniqueCells(ix.labels.filter((l) => o.has(l)))
      return py.obj(new IndexObj(u, ix.name === o.name ? ix.name : null, ix.dt))
    })
    case 'difference': return m((a) => {
      const o = toIndexObj(h, a[0])
      const u = uniqueCells(ix.labels.filter((l) => !o.has(l)))
      return py.obj(new IndexObj(sortOrder([u], [true]).map((i) => u[i]), ix.name, ix.dt))
    })
    case 'drop': return m((a, kw) => {
      const labels = axisLabels(h, flatArgs(a, kw, 'labels'))
      const miss = labels.filter((l) => !ix.has(l))
      if (miss.length && !(given(kw.errors) && h.str(kw.errors) === 'ignore')) throw h.err('KeyError', repr(py.str(`[${miss.map(reprCell).join(', ')}] not found in axis`)))
      return py.obj(ix.pick(rows(ix.length).filter((i) => !labels.includes(ix.labels[i]))))
    })
    case 'delete': return m((a) => {
      const del = new Set(positions(h, a[0], ix.length))
      return py.obj(ix.pick(rows(ix.length).filter((i) => !del.has(i))))
    })
    case 'insert': return m((a) => {
      const labels = [...ix.labels]
      labels.splice(h.toInt(a[0]), 0, cellOf(h, a[1]))
      return py.obj(new IndexObj(labels, ix.name, commonDT([ix.dt, inferDT([cellOf(h, a[1])])])))
    })
    case 'equals': return m((a) => py.bool(a[0].k === 'obj' && a[0].o instanceof IndexObj && ix.equals(a[0].o)))
    case 'max': case 'min': case 'sum': case 'mean': case 'argmax': case 'argmin': case 'value_counts': return h.getAttr(py.obj(asSeries()), name)
  }
  return undefined
}

