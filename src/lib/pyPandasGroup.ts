/**
 * groupby and rolling / expanding windows. Group statistics follow pandas' cython loops (Kahan sums, Welford variance) and
 * windows its add/remove updates, so floating-point results agree with the real thing rather than with a naive recompute.
 */
import { PyDict, PyObj, py, repr } from './minipy'
import type { Host, Value } from './minipy'
import { NDArray } from './ndarray'
import { Frame, IndexObj, Series, bindParams, cellOf, commonDT, inferDT, inferValues, isListLike, isMissing, labelStr, numCell, quietly, sortOrder, tableOfFrame, tableOfSeries, traceFrame, typedCells, valueOf, seriesIn, frameIn } from './pyPandas'
import type { Cats, Cell, DT, Params } from './pyPandas'
import { PyCell } from './pyPandasFormat'
import { flag, funcName, given, need, npQuantile, reduceCol } from './pyPandasMethods'

interface GKey { name: Cell; values: Cell[]; dt: DT; cats: Cats | null }
interface Groups { keys: Cell[][]; members: number[][]; ofRow: number[] }
interface GOpts { asIndex: boolean; sort: boolean; dropna: boolean; groupKeys: boolean; observed: boolean }

const AGG_OPTS = 'numeric_only min_count'
const NUMERIC_ONLY_HOWS = ['mean', 'median', 'std', 'var', 'sem', 'skew', 'prod']

const cellStr = (c: Cell) => (isMissing(c) ? 'NaN' : c instanceof PyCell ? repr(c.v) : `${typeof c}:${c}`)

/** one group statistic for each group, as pandas' cython group_* functions compute it */
const groupStat = (h: Host, s: Series, members: number[][], how: string, o: { ddof?: number; minCount?: number; skipna?: boolean; q?: number; interp?: string } = {}): { cells: Cell[]; dt: DT } => {
  const dt = s.dtype
  const isText = dt === 'object' && s.values.some((c) => typeof c === 'string' || (c instanceof PyCell && c.v.k !== 'float' && c.v.k !== 'int'))
  const vals = (m: number[]) => m.map((r) => s.values[r]).filter((c) => !isMissing(c))
  if (isText && (NUMERIC_ONLY_HOWS.includes(how) || how === 'quantile')) {
    if (how === 'quantile') throw h.err('TypeError', "dtype 'object' does not support operation 'quantile'")
    if (how === 'prod') {
      const ok = members.every((m) => vals(m).length <= 1)
      if (!ok) throw h.err('TypeError', "can't multiply sequence by non-int of type 'str'")
      return { cells: members.map((m) => vals(m)[0] ?? 1), dt: 'object' }
    }
    if (['std', 'sem', 'skew'].includes(how)) throw h.err('ValueError', `could not convert string to float: '${s.values.find((c) => typeof c === 'string')}'`)
    throw h.err('TypeError', `agg function failed [how->${how},dtype->object]`)
  }
  switch (how) {
    case 'count': return { cells: members.map((m) => vals(m).length), dt: 'int64' }
    case 'size': return { cells: members.map((m) => m.length), dt: 'int64' }
    case 'nunique': return { cells: members.map((m) => new Set(vals(m).map(cellStr)).size), dt: 'int64' }
    case 'first': case 'last': {
      const cells = members.map((m) => {
        const v = vals(m)
        return v.length && !(o.minCount !== undefined && v.length < o.minCount) ? v[how === 'first' ? 0 : v.length - 1] : NaN
      })
      return { cells, dt: dt === 'int64' && cells.some(isMissing) ? 'float64' : dt === 'bool' && cells.some(isMissing) ? 'object' : dt }
    }
    case 'any': case 'all': {
      const truth = (c: Cell) => (typeof c === 'string' ? c.length > 0 : !!c)
      return { cells: members.map((m) => {
        const skipna = o.skipna ?? true
        const v = m.map((r) => s.values[r]).filter((c) => !(skipna && isMissing(c))).map((c) => (isMissing(c) ? true : truth(c)))
        return how === 'any' ? v.some(Boolean) : v.every(Boolean)
      }), dt: 'bool' }
    }
    case 'min': case 'max': {
      if (isText) {
        const cells = members.map((m) => {
          const v = vals(m) as string[]
          return v.length ? v.reduce((a, b) => (how === 'min' ? (b < a ? b : a) : b > a ? b : a)) : NaN
        })
        return { cells, dt: 'object' }
      }
      const cells = members.map((m) => {
        const v = vals(m).map(numCell)
        if (!v.length) return NaN
        let best = v[0]
        for (const x of v) if (how === 'min' ? x < best : x > best) best = x
        return best
      })
      if (dt === 'bool') return { cells: cells.map((c) => !!c), dt: 'bool' }
      return { cells, dt: dt === 'int64' && !cells.some(isMissing) ? 'int64' : 'float64' }
    }
    case 'sum': {
      if (isText) return { cells: members.map((m) => (vals(m).length ? (vals(m) as string[]).join('') : 0)), dt: 'object' }
      const cells = members.map((m) => {
        const v = vals(m).map(numCell)
        if (o.minCount !== undefined && v.length < o.minCount) return NaN
        let sum = 0
        let comp = 0
        for (const x of v) {
          const y = x - comp
          const t = sum + y
          comp = t - sum - y
          if (Number.isNaN(comp)) comp = 0
          sum = t
        }
        return sum
      })
      return { cells, dt: dt === 'float64' || cells.some(isMissing) ? 'float64' : 'int64' }
    }
    case 'prod': {
      const cells = members.map((m) => {
        const v = vals(m).map(numCell)
        if (o.minCount !== undefined && v.length < o.minCount) return NaN
        return v.reduce((a, b) => a * b, 1)
      })
      return { cells, dt: dt === 'float64' || cells.some(isMissing) ? 'float64' : 'int64' }
    }
    case 'mean': return { cells: members.map((m) => {
      const v = vals(m).map(numCell)
      if (!v.length) return NaN
      let sum = 0
      let comp = 0
      for (const x of v) {
        const y = x - comp
        const t = sum + y
        comp = t - sum - y
        if (Number.isNaN(comp)) comp = 0
        sum = t
      }
      return sum / v.length
    }), dt: 'float64' }
    case 'median': return { cells: members.map((m) => {
      const v = vals(m).map(numCell).sort((a, b) => a - b)
      const n = v.length
      return !n ? NaN : n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2
    }), dt: 'float64' }
    case 'var': case 'std': case 'sem': {
      const ddof = o.ddof ?? 1
      return { cells: members.map((m) => {
        let n = 0
        let mean = 0
        let out = 0
        for (const x of vals(m).map(numCell)) {
          n++
          const old = mean
          mean += (x - old) / n
          out += (x - mean) * (x - old)
        }
        if (n <= ddof) return NaN
        return how === 'var' ? out / (n - ddof) : how === 'std' ? Math.sqrt(out / (n - ddof)) : Math.sqrt(out / (n - ddof) / n)
      }), dt: 'float64' }
    }
    case 'skew': return { cells: members.map((m) => numCell(cellOf(h, reduceCol(h, s.pick(m), 'skew', { skipna: o.skipna })))), dt: 'float64' }
    case 'quantile': {
      const q = o.q ?? 0.5
      const interp = o.interp ?? 'linear'
      return { cells: members.map((m) => {
        const v = vals(m).map(numCell).sort((a, b) => a - b)
        const n = v.length
        if (!n) return NaN
        const qi = q * (n - 1)
        const idx = Math.trunc(qi)
        const frac = qi % 1
        if (frac === 0 || interp === 'lower') return v[idx]
        const next = v[idx + 1]
        if (interp === 'linear') return v[idx] + (next - v[idx]) * frac
        if (interp === 'higher') return next
        if (interp === 'midpoint') return (v[idx] + next) / 2
        return frac > 0.5 || (frac === 0.5 && (idx & 1) === 1) ? next : v[idx]
      }), dt: dt === 'int64' && ['lower', 'higher', 'nearest'].includes(interp) ? 'int64' : 'float64' }
    }
    case 'idxmax': case 'idxmin': {
      const skipna = o.skipna ?? true
      const cells = members.map((m) => {
        let k = -1
        for (const r of m) {
          const c = s.values[r]
          if (isMissing(c)) {
            if (!skipna) return NaN
            continue
          }
          if (k < 0 || (how === 'idxmax' ? (c as number) > (s.values[k] as number) : (c as number) < (s.values[k] as number))) k = r
        }
        return k < 0 ? NaN : s.index.labels[k]
      })
      const idt = s.index.dt
      return { cells, dt: cells.some(isMissing) && idt === 'int64' ? 'float64' : idt }
    }
  }
  throw h.err('AttributeError', `'SeriesGroupBy' object has no attribute '${how}'`)
}

const AGG_NAMES = ['count', 'size', 'nunique', 'first', 'last', 'any', 'all', 'min', 'max', 'sum', 'prod', 'mean', 'median', 'var', 'std', 'sem', 'skew', 'idxmax', 'idxmin']
/** numpy functions that agg treats as pandas' own */
const NP_ALIASES: Record<string, string> = { sum: 'sum', mean: 'mean', prod: 'prod', std: 'std', var: 'var', median: 'median', max: 'max', min: 'min', amax: 'max', amin: 'min', nanmean: 'mean', nansum: 'sum', all: 'all', any: 'any', len: 'size' }

/** labels for several functions: pandas numbers the lambdas once there is more than one function */
const funcLabels = (h: Host, fs: Value[]): string[] => {
  let n = 0
  return fs.map((x) => {
    const name = funcName(h, x)
    return name === '<lambda>' && fs.length > 1 ? `<lambda_${n++}>` : name
  })
}

const aggName = (f: Value): string | null => {
  if (f.k === 'str') return f.v === 'product' ? 'prod' : f.v
  if (f.k === 'fn' && NP_ALIASES[f.name]) return NP_ALIASES[f.name]
  return null
}

export class GroupBy extends PyObj {
  readonly f: Frame
  readonly gkeys: GKey[]
  readonly select: Cell[] | null
  readonly single: boolean
  readonly opts: GOpts
  readonly label: string
  /** the frame only wraps a Series: results keep that Series' shape */
  readonly fromSeries: boolean
  private memo: Groups | null = null
  constructor(f: Frame, gkeys: GKey[], select: Cell[] | null, single: boolean, opts: GOpts, label: string, fromSeries = false) {
    super()
    this.f = f
    this.gkeys = gkeys
    this.select = select
    this.single = single
    this.opts = opts
    this.label = label
    this.fromSeries = fromSeries
  }
  get cls() { return this.single ? 'SeriesGroupBy' : 'DataFrameGroupBy' }
  repr() { return `<pandas.core.groupby.generic.${this.cls} object>` }
  /** groups in key order, and the group number of every row */
  groups(): Groups {
    if (this.memo) return this.memo
    const map = new Map<string, { key: Cell[]; rows: number[] }>()
    const n = this.f.nrows
    for (let r = 0; r < n; r++) {
      const k = this.gkeys.map((g) => g.values[r])
      if (this.opts.dropna && k.some(isMissing)) continue
      const s = k.map(cellStr).join('\0')
      for (let j = 0; j < k.length; j++) if (k[j] === null) k[j] = NaN
      const e = map.get(s)
      if (e) e.rows.push(r)
      else map.set(s, { key: k, rows: [r] })
    }
    let entries = [...map.values()]
    const first = this.gkeys[0]
    if (this.gkeys.length === 1 && first.cats && !this.opts.observed) {
      for (const l of first.cats.labels) if (!map.has(cellStr(l))) entries.push({ key: [l], rows: [] })
    }
    if (this.opts.sort || this.gkeys.some((g) => g.cats)) {
      const col = (j: number) => {
        const g = this.gkeys[j]
        return entries.map((e) => (g.cats ? g.cats.labels.indexOf(e.key[j]) : e.key[j]))
      }
      entries = sortOrder(this.gkeys.map((_, j) => col(j)), this.gkeys.map(() => true)).map((i) => entries[i])
    }
    const ofRow = new Array<number>(n).fill(-1)
    entries.forEach((e, g) => e.rows.forEach((r) => { ofRow[r] = g }))
    this.memo = { keys: entries.map((e) => e.key), members: entries.map((e) => e.rows), ofRow }
    return this.memo
  }
  /** the index of an aggregated result */
  keyIndex(g: Groups): IndexObj {
    if (this.gkeys.length === 1) {
      const k = this.gkeys[0]
      return new IndexObj(g.keys.map((x) => x[0]), k.name, k.cats ? 'object' : k.dt === 'bool' && g.keys.some((x) => isMissing(x[0])) ? 'object' : k.dt === 'int64' && g.keys.some((x) => isMissing(x[0])) ? 'float64' : k.dt)
    }
    return IndexObj.fromArrays(this.gkeys.map((_, j) => g.keys.map((x) => x[j])), this.gkeys.map((k) => k.name), this.gkeys.map((k) => (k.cats ? 'object' : k.dt)))
  }
  valueCols(): number[] {
    const f = this.f
    if (this.select) return this.select.flatMap((l) => f.columns.locate(l))
    const keys = this.gkeys.filter((g) => g.name !== null && inFrame(f, g))
    return f.data.map((_, k) => k).filter((k) => !keys.some((g) => g.name === f.columns.labels[k]))
  }
  numericOnly(cols: number[]) { return cols.filter((k) => this.f.data[k].dtype !== 'object' && !this.f.data[k].cats) }
  /** keys become ordinary columns with as_index=False */
  finish(out: Frame | Series, g: Groups): Value {
    if (this.opts.asIndex) return py.obj(out)
    const frame = out instanceof Series ? new Frame([out.name ?? 0], [out], out.index) : out
    const key = new Frame(this.gkeys.map((k) => k.name ?? 'index'), this.gkeys.map((k, j) => new Series(g.keys.map((x) => x[j]), null, k.name, k.dt)), null)
    return py.obj(new Frame([...key.columns.labels, ...frame.columns.labels], [...key.data, ...frame.data], IndexObj.range(g.keys.length)))
  }
  /** one statistic per group, for one column or a table of them */
  stat(h: Host, how: string, cols: number[], o: Parameters<typeof groupStat>[4] = {}): Value {
    const g = this.groups()
    const index = this.keyIndex(g)
    if (this.single) {
      const k = cols[0]
      const r = groupStat(h, this.f.data[k], g.members, how, o)
      return this.finish(new Series(r.cells, index, this.f.columns.labels[k], r.dt), g)
    }
    const parts = cols.map((k) => {
      const r = groupStat(h, this.f.data[k], g.members, how, o)
      return new Series(r.cells, index, this.f.columns.labels[k], r.dt)
    })
    return this.finish(new Frame(this.f.columns.pick(cols, true), parts, index), g)
  }
  /** rows back in their original order, one result cell per row: cumsum, rank, shift … */
  rowwise(cols: number[], kernel: (s: Series, rows: number[]) => { cells: Cell[]; dt?: DT }): Value {
    const g = this.groups()
    const parts = cols.map((k) => {
      const s = this.f.data[k]
      const out: Cell[] = new Array(this.f.nrows).fill(NaN)
      const dts: DT[] = []
      g.members.forEach((rows) => {
        if (!rows.length) return
        const res = kernel(s.pick(rows), rows)
        if (res.dt) dts.push(res.dt)
        rows.forEach((r, i) => { out[r] = res.cells[i] })
      })
      const missingKey = g.ofRow.some((x) => x < 0)
      const t = dts.length ? commonDT(dts) : inferDT(out)
      const dt: DT = missingKey && t === 'int64' ? 'float64' : missingKey && t === 'bool' ? 'object' : t
      return new Series(out, this.f.index, this.f.columns.labels[k], dt)
    })
    return this.single ? py.obj(parts[0]) : py.obj(new Frame(this.f.columns.pick(cols, true), parts, this.f.index))
  }
  groupColumn(k: number, rows: number[]) { return this.f.data[k].pick(rows) }
  /** a group as the callback sees it */
  groupObject(cols: number[] | null, rows: number[]): Value {
    if (this.single) return py.obj(this.f.data[this.valueCols()[0]].pick(rows))
    const out = this.f.take(rows, cols)
    return py.obj(out)
  }
  /** agg() with one function, for one column */
  aggOne(h: Host, f: Value, k: number, g: Groups): { cells: Cell[]; dt: DT } {
    const name = aggName(f)
    const s = this.f.data[k]
    if (name) {
      if (name === 'size' || AGG_NAMES.includes(name)) return groupStat(h, s, g.members, name)
      const sub = new GroupBy(this.f, this.gkeys, [this.f.columns.labels[k]], true, { ...this.opts, asIndex: true }, this.label)
      const res = sub.run(h, name, [], {})
      if (res.k === 'obj' && res.o instanceof Series && res.o.length === g.keys.length) return { cells: res.o.values, dt: res.o.dtype }
      throw h.err('AttributeError', `'${name}' is not a valid function for 'SeriesGroupBy' object`)
    }
    const results = g.members.map((rows) => h.call(f, [py.obj(s.pick(rows))]))
    return inferValues(h, results)
  }
  agg(h: Host, f: Value | undefined, named: [string, Value][]): Value {
    const g = this.groups()
    const index = this.keyIndex(g)
    if (named.length) {
      const parts = named.map(([label, spec]) => {
        const [col, how] = spec.k === 'tuple' || spec.k === 'list' ? spec.items : [null, spec]
        const k = col === null ? this.valueCols()[0] : this.colPos(h, col)
        const r = this.aggOne(h, how, k, g)
        return new Series(r.cells, index, label, r.dt)
      })
      return this.finish(new Frame(named.map(([l]) => l), parts, index), g)
    }
    if (!f) throw h.err('TypeError', "Must provide 'func' or tuples of '(column, aggfunc).")
    if (f.k === 'dict') {
      const entries = f.d.items()
      if (this.single) throw h.err('SpecificationError', 'nested renamer is not supported')
      const anyList = entries.some(([, x]) => isListLike(x))
      for (const [c] of entries) if (!this.f.columns.has(cellOf(h, c))) throw h.err('KeyError', `"Column(s) ['${labelStr(cellOf(h, c))}'] do not exist"`)
      if (!anyList) {
        const parts = entries.map(([c, x]) => {
          const k = this.colPos(h, c)
          const r = this.aggOne(h, x, k, g)
          return new Series(r.cells, index, this.f.columns.labels[k], r.dt)
        })
        return this.finish(new Frame(entries.map(([c]) => cellOf(h, c)), parts, index), g)
      }
      const flat = entries.flatMap(([c, x]) => {
        const fs = isListLike(x) ? h.iterate(x) : [x]
        const names = funcLabels(h, fs)
        return fs.map((fn, i) => ({ k: this.colPos(h, c), c: cellOf(h, c), fn, name: names[i] }))
      })
      const cols = IndexObj.fromArrays([flat.map((e) => e.c), flat.map((e) => e.name)], [null, null], [this.f.columns.dt, 'object'])
      return this.finish(new Frame(cols, flat.map((e) => {
        const r = this.aggOne(h, e.fn, e.k, g)
        return new Series(r.cells, index, null, r.dt)
      }), index), g)
    }
    if (isListLike(f)) {
      const fs = h.iterate(f)
      if (this.single) {
        const k = this.valueCols()[0]
        return this.finish(new Frame(funcLabels(h, fs), fs.map((x) => {
          const r = this.aggOne(h, x, k, g)
          return new Series(r.cells, index, null, r.dt)
        }), index), g)
      }
      const cols = this.valueCols()
      const names = funcLabels(h, fs)
      const labels = cols.flatMap((k) => fs.map((x, i) => ({ k, fn: x, name: names[i] })))
      const mi = IndexObj.fromArrays([labels.map((e) => this.f.columns.labels[e.k]), labels.map((e) => e.name)], [null, null], [this.f.columns.dt, 'object'])
      return this.finish(new Frame(mi, labels.map((e) => {
        const r = this.aggOne(h, e.fn, e.k, g)
        return new Series(r.cells, index, null, r.dt)
      }), index), g)
    }
    const name = aggName(f)
    if (name) return this.run(h, name, [], {})
    const cols = this.valueCols()
    if (this.single) {
      const r = this.aggOne(h, f, cols[0], g)
      return this.finish(new Series(r.cells, index, this.f.columns.labels[cols[0]], r.dt), g)
    }
    return this.finish(new Frame(this.f.columns.pick(cols, true), cols.map((k) => {
      const r = this.aggOne(h, f, k, g)
      return new Series(r.cells, index, this.f.columns.labels[k], r.dt)
    }), index), g)
  }
  colPos(h: Host, c: Value) {
    const at = this.f.columns.locate(cellOf(h, c))
    if (!at.length) throw h.err('KeyError', `"Column(s) ['${labelStr(cellOf(h, c))}'] do not exist"`)
    return at[0]
  }
  apply(h: Host, f: Value, args: Value[], kw: Record<string, Value>): Value {
    const g = this.groups()
    const nonEmpty = g.members.map((rows, i) => ({ rows, i })).filter((e) => e.rows.length)
    const cols = this.select ? this.valueCols() : null
    const called = nonEmpty.map((e) => ({ i: e.i, rows: e.rows, res: h.call(f, [this.groupObject(cols, e.rows), ...args], kw) }))
    // None drops a group from a table of results, but among scalars it is a missing value
    const sample = called.find((e) => e.res.k !== 'none')?.res
    const results = sample && sample.k === 'obj' && (sample.o instanceof Frame || sample.o instanceof Series) ? called.filter((e) => e.res.k !== 'none') : called
    const keyOf = (e: { i: number }) => g.keys[e.i]
    const subIndex = this.keyIndex({ keys: results.map(keyOf), members: [], ofRow: [] })
    if (!results.length || !sample) return py.obj(new Series([], IndexObj.range(0), null, 'float64'))
    const first = results[0].res
    /** (keys..., inner labels) */
    const withKeys = (inner: IndexObj[]): IndexObj => {
      const kn = this.gkeys.length
      const levels = inner[0].nlevels
      const arrays: Cell[][] = Array.from({ length: kn + levels }, () => [])
      results.forEach((e, i) => {
        const ix = inner[i]
        const k = keyOf(e)
        for (let r = 0; r < ix.length; r++) {
          for (let j = 0; j < kn; j++) arrays[j].push(k[j])
          for (let j = 0; j < levels; j++) arrays[kn + j].push(ix.level(j).labels[r])
        }
      })
      const dts: DT[] = [...this.gkeys.map((k) => k.dt), ...Array.from({ length: levels }, (_, j) => inner[0].level(j).dt)]
      return IndexObj.fromArrays(arrays, [...this.gkeys.map((k) => k.name), ...inner[0].names], dts)
    }
    if (first.k === 'obj' && first.o instanceof Frame) {
      const frames = results.map((e) => frameIn(e.res))
      const columns = frames[0].columns
      const data = columns.labels.map((c) => {
        const vals = frames.flatMap((x) => {
          const col = x.col(c)
          return col ? col.values : x.index.labels.map(() => NaN)
        })
        const dt = commonDT(frames.map((x) => x.col(c)?.dtype ?? 'float64'))
        return new Series(vals, null, c, dt)
      })
      if (this.opts.groupKeys) return py.obj(new Frame(columns, data, withKeys(frames.map((x) => x.index))))
      const flat = new IndexObj(frames.flatMap((x) => x.index.labels), frames[0].index.name, frames[0].index.dt)
      const out = new Frame(columns, data, flat)
      const indexedSame = frames.every((x, i) => x.nrows === results[i].rows.length && x.index.labels.every((l, j) => l === this.f.index.labels[results[i].rows[j]]))
      if (!indexedSame) return py.obj(out)
      const positions = results.flatMap((e) => e.rows)
      return py.obj(out.take(sortOrder([positions], [true]), null))
    }
    if (first.k === 'obj' && first.o instanceof Series) {
      const parts = results.map((e) => seriesIn(e.res))
      const sameIdx = parts.every((x) => x.index.equals(parts[0].index))
      if (!this.single && sameIdx) {
        const t = parts[0].index
        const cols2 = t.labels.map((_, j) => {
          const v = inferValues(h, parts.map((x) => valueOf(x.values[j], x.dtype)))
          return new Series(v.cells, subIndex, t.labels[j], v.dt)
        })
        return py.obj(new Frame(t, cols2, subIndex))
      }
      const vals = parts.flatMap((x) => x.values)
      const dt = commonDT(parts.map((x) => x.dtype))
      if (!this.opts.groupKeys) {
        const positions = results.flatMap((e) => e.rows)
        const out = new Series(vals, new IndexObj(parts.flatMap((x) => x.index.labels), parts[0].index.name, parts[0].index.dt), parts[0].name, dt)
        const indexedSame = parts.every((x, i) => x.length === results[i].rows.length)
        return py.obj(indexedSame ? out.pick(sortOrder([positions], [true])) : out)
      }
      return py.obj(new Series(vals, withKeys(parts.map((x) => x.index)), parts[0].name, dt))
    }
    const t = inferValues(h, results.map((e) => (e.res.k === 'none' ? py.float(NaN) : e.res)))
    const name = this.single ? this.f.columns.labels[this.valueCols()[0]] : null
    return this.finish(new Series(t.cells, subIndex, name, t.dt), { keys: results.map(keyOf), members: [], ofRow: [] })
  }
  transform(h: Host, f: Value, args: Value[], kw: Record<string, Value>): Value {
    const cols = this.valueCols()
    const name = aggName(f)
    const g = this.groups()
    if (name) {
      if (AGG_NAMES.includes(name)) {
        const parts = cols.map((k) => {
          const r = groupStat(h, this.f.data[k], g.members, name)
          const cells = g.ofRow.map((gi) => (gi < 0 ? NaN : r.cells[gi]))
          return new Series(cells, this.f.index, this.f.columns.labels[k], g.ofRow.some((x) => x < 0) && r.dt === 'int64' ? 'float64' : r.dt)
        })
        return this.single ? py.obj(parts[0]) : py.obj(new Frame(this.f.columns.pick(cols, true), parts, this.f.index))
      }
      return this.run(h, name, args, kw)
    }
    return this.rowwise(cols, (s, rows) => {
      const res = h.call(f, [py.obj(s), ...args], kw)
      if (res.k === 'obj' && res.o instanceof Series) {
        if (res.o.length !== rows.length) throw h.err('ValueError', 'Length of values does not match length of index')
        return { cells: res.o.values, dt: res.o.dtype }
      }
      if (isListLike(res)) {
        const t = typedCells(h, res)
        if (t.cells.length !== rows.length) throw h.err('ValueError', 'Length of values does not match length of index')
        return { cells: t.cells, dt: t.dt }
      }
      const t = inferValues(h, [res])
      return { cells: rows.map(() => t.cells[0]), dt: t.dt }
    })
  }
  /** run a named group method (used by agg('x') and the attribute table) */
  run(h: Host, name: string, args: Value[], kw: Record<string, Value>): Value {
    const m = METHODS[name]
    if (!m) throw h.err('AttributeError', `'${this.cls}' object has no attribute '${name}'`)
    return m[1](this, bindParams(h, this.cls, name, m[0], args, kw), h)
  }
  getItem(idx: Value, h: Host): Value {
    if (this.single) throw h.err('IndexError', `Column(s) ${h.str(idx)} already selected`)
    const isList = isListLike(idx)
    const labels = isList ? typedCells(h, idx).cells : [cellOf(h, idx)]
    const miss = labels.filter((c) => !this.f.columns.has(c))
    if (miss.length) throw h.err('KeyError', isList ? `"Columns not found: ${miss.map((c) => `'${labelStr(c)}'`).join(', ')}"` : `'Column not found: ${labelStr(miss[0])}'`)
    return py.obj(new GroupBy(this.f, this.gkeys, labels, !isList, this.opts, this.label))
  }
  iter(): Value[] {
    const g = this.groups()
    const cols = this.select ? this.valueCols() : null
    return g.keys.flatMap((k, i) => {
      if (!g.members[i].length) return []
      const key = this.gkeys.length === 1 && !this.keyIsList ? valueOf(k[0], this.gkeys[0].dt) : py.tuple(k.map((x, j) => valueOf(x, this.gkeys[j].dt)))
      return [py.tuple([key, this.groupObject(cols, g.members[i])])]
    })
  }
  keyIsList = false
  len() { return this.groups().keys.length }
  getAttr(name: string, h: Host): Value | undefined {
    const g = () => this.groups()
    switch (name) {
      case 'ngroups': return py.int(g().keys.length)
      case 'groups': {
        const d = new PyDict()
        g().keys.forEach((k, i) => d.set(this.keyValue(k), py.obj(this.f.index.pick(g().members[i]))))
        return py.obj(new PrettyDict(d))
      }
      case 'indices': {
        const d = new PyDict()
        g().keys.forEach((k, i) => d.set(this.keyValue(k), py.arr(NDArray.create(g().members[i], [g().members[i].length], 'int64'))))
        return { k: 'dict', d }
      }
      case 'ndim': return py.int(this.single ? 1 : 2)
      case 'dtypes': {
        // one row per group, one column per column of the frame, keys included
        const gr = this.groups()
        const index = this.keyIndex(gr)
        if (this.single) return py.obj(new Series(gr.keys.map(() => this.f.data[this.valueCols()[0]].dtype), index, this.f.columns.labels[this.valueCols()[0]], 'object'))
        return py.obj(new Frame(this.f.columns, this.f.data.map((c) => new Series(gr.keys.map(() => c.dtype), index, null, 'object')), index))
      }
    }
    if (METHODS[name]) {
      return { k: 'fn', name, call: (args, kw) => {
        const out = quietly(h, () => this.run(h, name, args, kw))
        traceFrame(h, 'groupby', () => {
          const gr = this.groups()
          const res = out.k === 'obj' && out.o instanceof Frame ? tableOfFrame(out.o) : out.k === 'obj' && out.o instanceof Series ? tableOfSeries(out.o) : null
          const cols = this.gkeys.flatMap((k) => this.f.columns.locate(k.name))
          return { inputs: [{ label: this.label, table: tableOfFrame(this.f) }], result: res, picks: { rows: null, cols }, groups: { keys: gr.keys.map((k) => k.map(labelStr).join(', ')), ofRow: gr.ofRow }, resultText: res ? '' : repr(out) }
        })
        return out
      } }
    }
    if (!this.single && this.f.columns.has(name) && !this.gkeys.some((k) => k.name === name)) return this.getItem(py.str(name), h)
    return undefined
  }
  keyValue(k: Cell[]): Value {
    return this.gkeys.length === 1 ? valueOf(k[0], this.gkeys[0].dt) : py.tuple(k.map((x, j) => valueOf(x, this.gkeys[j].dt)))
  }
}

/** g.groups: a dict of Index objects, printed as lists of labels */
class PrettyDict extends PyObj {
  readonly cls = 'PrettyDict'
  readonly d: PyDict
  constructor(d: PyDict) {
    super()
    this.d = d
  }
  repr() {
    return `{${this.d.items().map(([k, v]) => `${repr(k)}: [${h2list(v)}]`).join(', ')}}`
  }
  len() { return this.d.size }
  iter() { return this.d.keys() }
  contains(v: Value) { return this.d.has(v) }
  getItem(idx: Value, h: Host): Value {
    const v = this.d.get(idx)
    if (!v) throw h.err('KeyError', repr(idx))
    return v
  }
  getAttr(name: string): Value | undefined {
    const fn = (call: () => Value): Value => ({ k: 'fn', name, call })
    if (name === 'keys') return fn(() => py.list(this.d.keys()))
    if (name === 'values') return fn(() => py.list(this.d.values()))
    if (name === 'items') return fn(() => py.list(this.d.items().map(([k, v]) => py.tuple([k, v]))))
    return undefined
  }
}

const h2list = (v: Value) => ((v as { o: unknown }).o as IndexObj).labels.map(labelStrQuoted).join(', ')
const labelStrQuoted = (c: Cell) => (typeof c === 'string' ? `'${c}'` : labelStr(c))

const inFrame = (f: Frame, g: GKey) => {
  const c = f.col(g.name)
  return !!c && c.values === g.values
}

type GM = [string, (g: GroupBy, p: Params, h: Host) => Value]

const colsFor = (g: GroupBy, h: Host, p: Params) => (flag(h, p.numeric_only) && !g.single ? g.numericOnly(g.valueCols()) : g.valueCols())

const statMethod = (how: string, spec = AGG_OPTS): GM => [spec, (g, p, h) => g.stat(h, how, colsFor(g, h, p), { ddof: given(p.ddof) ? h.toInt(p.ddof!) : undefined, minCount: given(p.min_count) ? h.toInt(p.min_count!) : undefined, skipna: flag(h, p.skipna, true) })]

const fillKernel = (forward: boolean, limit: number | null) => (s: Series) => {
  const vals = [...s.values]
  let last: Cell = null
  let have = false
  let run = 0
  for (let k = 0; k < vals.length; k++) {
    const i = forward ? k : vals.length - 1 - k
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
  return { cells: vals, dt: s.dtype }
}

const cumKernel = (h: Host, how: string, skipna: boolean) => (s: Series) => {
  if (s.dtype === 'object') throw h.err('NotImplementedError', `function is not implemented for this dtype: [how->${how},dtype->object]`)
  const res = seriesIn(h.call(s.getAttr(how, h)!, [], { skipna: py.bool(skipna) }))
  return { cells: res.values, dt: res.dtype }
}

/** DataFrameGroupBy.value_counts: how often each row of the other columns occurs in each group */
const frameValueCounts = (g: GroupBy, p: Params, h: Host): Value => {
  // in this form the first positional argument is subset
  const first = p.normalize
  const positional = first && (first.k === 'str' || first.k === 'list' || first.k === 'tuple') ? first : undefined
  const subset = positional ?? (given(p.subset) ? p.subset : undefined)
  const names = subset ? (subset.k === 'str' ? [subset] : h.iterate(subset)) : null
  const cols = names ? names.map((n) => {
    const at = g.f.columns.locate(n.k === 'str' ? n.v : h.str(n))
    if (!at.length) throw h.err('ValueError', `Keys ${h.str(n)} in subset do not exist in the DataFrame.`)
    return at[0]
  }) : g.valueCols()
  const norm = positional ? false : flag(h, p.normalize)
  const sort = flag(h, p.sort, true)
  const asc = flag(h, p.ascending)
  const dropna = flag(h, p.dropna, true)
  const gr = g.groups()
  const arrays: Cell[][] = [...g.gkeys.map(() => [] as Cell[]), ...cols.map(() => [] as Cell[])]
  const vals: number[] = []
  gr.members.forEach((rows, i) => {
    const present = rows.filter((r) => !dropna || cols.every((c) => !isMissing(g.f.data[c].values[r])))
    const counts = new Map<string, { cells: Cell[]; n: number }>()
    for (const r of present) {
      const cells = cols.map((c) => g.f.data[c].values[r])
      const key = cells.map(cellStr).join('\u0000')
      const e = counts.get(key)
      if (e) e.n++
      else counts.set(key, { cells, n: 1 })
    }
    let entries = [...counts.values()]
    entries = sortOrder(cols.map((_, c) => entries.map((e) => e.cells[c])), cols.map(() => true)).map((x) => entries[x])
    if (sort) entries = sortOrder([entries.map((e) => e.n)], [asc]).map((x) => entries[x])
    for (const e of entries) {
      g.gkeys.forEach((_, j) => arrays[j].push(gr.keys[i][j]))
      e.cells.forEach((c, j) => arrays[g.gkeys.length + j].push(c))
      vals.push(norm ? e.n / present.length : e.n)
    }
  })
  const index = IndexObj.fromArrays(arrays, [...g.gkeys.map((x) => x.name), ...cols.map((c) => g.f.columns.labels[c])], [...g.gkeys.map((x) => x.dt), ...cols.map((c) => g.f.data[c].dtype)])
  return py.obj(new Series(vals, index, norm ? 'proportion' : 'count', norm ? 'float64' : 'int64'))
}

const METHODS: Record<string, GM> = {
  count: statMethod('count', ''),
  size: ['', (g) => {
    const gr = g.groups()
    const name = !g.opts.asIndex ? 'size' : g.single && g.f.columns.labels[g.valueCols()[0]] !== '__values__' ? g.f.columns.labels[g.valueCols()[0]] : null
    return g.finish(new Series(gr.members.map((m) => m.length), g.keyIndex(gr), name, 'int64'), gr)
  }],
  nunique: ['dropna', (g, _p, h) => {
    if (_p.dropna && !h.truthy(_p.dropna)) throw h.err('NotImplementedError', 'groupby.nunique(dropna=False) is not available in this sandbox')
    return g.stat(h, 'nunique', g.valueCols())
  }],
  first: statMethod('first', 'numeric_only min_count'),
  last: statMethod('last', 'numeric_only min_count'),
  sum: statMethod('sum'),
  prod: statMethod('prod'),
  product: statMethod('prod'),
  min: statMethod('min'),
  max: statMethod('max'),
  mean: statMethod('mean', 'numeric_only engine engine_kwargs'),
  median: statMethod('median', 'numeric_only'),
  std: statMethod('std', 'ddof engine engine_kwargs numeric_only'),
  var: statMethod('var', 'ddof engine engine_kwargs numeric_only'),
  sem: statMethod('sem', 'ddof numeric_only'),
  skew: statMethod('skew', 'skipna numeric_only'),
  any: statMethod('any', 'skipna'),
  all: statMethod('all', 'skipna'),
  idxmax: statMethod('idxmax', 'skipna numeric_only'),
  idxmin: statMethod('idxmin', 'skipna numeric_only'),
  quantile: ['q interpolation numeric_only', (g, p, h) => {
    const interp = given(p.interpolation) ? h.str(p.interpolation!) : 'linear'
    const qv = p.q ?? py.float(0.5)
    if (!isListLike(qv)) return g.stat(h, 'quantile', colsFor(g, h, p), { q: h.num(qv), interp })
    const qs = typedCells(h, qv).cells.map(numCell)
    const gr = g.groups()
    const cols = colsFor(g, h, p)
    const n = gr.keys.length
    const arrays: Cell[][] = [...g.gkeys.map((_, j) => gr.keys.flatMap((k) => qs.map(() => k[j]))), Array.from({ length: n }, () => qs).flat()]
    const index = IndexObj.fromArrays(arrays, [...g.gkeys.map((k) => k.name), null], [...g.gkeys.map((k) => k.dt), 'float64'])
    const parts = cols.map((k) => {
      const per = qs.map((q) => groupStat(h, g.f.data[k], gr.members, 'quantile', { q, interp }).cells)
      return new Series(gr.keys.flatMap((_, i) => qs.map((_q, j) => per[j][i])), index, g.f.columns.labels[k], 'float64')
    })
    return g.single ? py.obj(parts[0]) : py.obj(new Frame(g.f.columns.pick(cols, true), parts, index))
  }],
  agg: ['func * **', (g, p, h) => {
    const named = p.__kw && p.__kw.k === 'dict' ? p.__kw.d.items().map(([k, v]): [string, Value] => [h.str(k), v]) : []
    return g.agg(h, p.func, p.func ? [] : named)
  }],
  aggregate: ['func * **', (g, p, h) => {
    const named = p.__kw && p.__kw.k === 'dict' ? p.__kw.d.items().map(([k, v]): [string, Value] => [h.str(k), v]) : []
    return g.agg(h, p.func, p.func ? [] : named)
  }],
  apply: ['func * include_groups **', (g, p, h) => {
    const f = need(h, p.func, 'apply', 'func')
    const args = p.__args && p.__args.k === 'tuple' ? p.__args.items : []
    const kw = p.__kw && p.__kw.k === 'dict' ? Object.fromEntries(p.__kw.d.items().map(([k, v]) => [h.str(k), v])) : {}
    const useKeys = flag(h, p.include_groups, true)
    const target = useKeys || g.single || g.select ? g : new GroupBy(g.f, g.gkeys, g.valueCols().map((k) => g.f.columns.labels[k]), false, g.opts, g.label)
    if (f.k === 'str') return g.run(h, f.v, args, kw)
    return target.apply(h, f, args, kw)
  }],
  transform: ['func * **', (g, p, h) => {
    const args = p.__args && p.__args.k === 'tuple' ? p.__args.items : []
    const kw = p.__kw && p.__kw.k === 'dict' ? Object.fromEntries(p.__kw.d.items().map(([k, v]) => [h.str(k), v])) : {}
    return g.transform(h, need(h, p.func, 'transform', 'func'), args, kw)
  }],
  filter: ['func dropna * **', (g, p, h) => {
    const f = need(h, p.func, 'filter', 'func')
    const gr = g.groups()
    const dropna = flag(h, p.dropna, true)
    const keep = new Set<number>()
    gr.members.forEach((rows) => {
      if (!rows.length) return
      const r = h.call(f, [g.groupObject(g.select ? g.valueCols() : null, rows), ...(p.__args && p.__args.k === 'tuple' ? p.__args.items : [])], p.__kw && p.__kw.k === 'dict' ? Object.fromEntries(p.__kw.d.items().map(([k, v]) => [h.str(k), v])) : {})
      if (r.k !== 'bool' && !(r.k === 'array' && r.a.size === 1) && r.k !== 'int' && r.k !== 'float') throw h.err('TypeError', 'filter function returned a ' + (r.k === 'obj' ? r.o.cls : r.k) + ', but expected a scalar bool')
      if (h.truthy(r)) rows.forEach((x) => keep.add(x))
    })
    const all = gr.ofRow.map((_, i) => i)
    const rows = all.filter((r) => keep.has(r))
    const src = g.single ? new Frame([g.f.columns.labels[g.valueCols()[0]]], [g.f.data[g.valueCols()[0]]], g.f.index) : g.select ? g.f.take(null, g.valueCols()) : g.f
    if (!dropna) {
      const out = new Frame(src.columns, src.data.map((s) => {
        const cells = s.values.map((c, r) => (keep.has(r) ? c : NaN))
        return s.with(cells, s.dtype === 'int64' ? 'float64' : s.dtype === 'bool' ? 'object' : s.dtype)
      }), src.index)
      return g.single ? py.obj(out.data[0]) : py.obj(out)
    }
    return g.single ? py.obj(src.data[0].pick(rows)) : py.obj(src.take(rows, null))
  }],
  cumcount: ['ascending', (g, p, h) => {
    const asc = flag(h, p.ascending, true)
    const gr = g.groups()
    const out = new Array<number>(g.f.nrows).fill(0)
    gr.members.forEach((rows) => rows.forEach((r, i) => { out[r] = asc ? i : rows.length - 1 - i }))
    const dropped = gr.ofRow.some((x) => x < 0)
    return py.obj(new Series(dropped ? out.map((x, r) => (gr.ofRow[r] < 0 ? NaN : x)) : out, g.f.index, null, dropped ? 'float64' : 'int64'))
  }],
  ngroup: ['ascending', (g, p, h) => {
    const asc = flag(h, p.ascending, true)
    const gr = g.groups()
    const n = gr.keys.length
    return py.obj(new Series(gr.ofRow.map((x) => (x < 0 ? NaN : asc ? x : n - 1 - x)), g.f.index, null, gr.ofRow.some((x) => x < 0) ? 'float64' : 'int64'))
  }],
  cumsum: ['skipna numeric_only', (g, p, h) => g.rowwise(colsFor(g, h, p), cumKernel(h, 'cumsum', flag(h, p.skipna, true)))],
  cumprod: ['skipna numeric_only', (g, p, h) => g.rowwise(colsFor(g, h, p), cumKernel(h, 'cumprod', flag(h, p.skipna, true)))],
  cummax: ['skipna numeric_only', (g, p, h) => g.rowwise(colsFor(g, h, p), cumKernel(h, 'cummax', flag(h, p.skipna, true)))],
  cummin: ['skipna numeric_only', (g, p, h) => g.rowwise(colsFor(g, h, p), cumKernel(h, 'cummin', flag(h, p.skipna, true)))],
  diff: ['periods', (g, p, h) => g.rowwise(g.valueCols(), (s) => {
    const res = seriesIn(h.call(s.getAttr('diff', h)!, [p.periods ?? py.int(1)]))
    return { cells: res.values, dt: 'float64' }
  })],
  shift: ['periods freq fill_value', (g, p, h) => {
    if (given(p.freq)) throw h.err('NotImplementedError', 'groupby.shift(freq=...) is not available in this sandbox')
    return g.rowwise(g.valueCols(), (s) => {
      const res = seriesIn(h.call(s.getAttr('shift', h)!, [p.periods ?? py.int(1)], p.fill_value ? { fill_value: p.fill_value } : {}))
      return { cells: res.values, dt: res.dtype }
    })
  }],
  pct_change: ['periods fill_method limit freq', (g, p, h) => {
    if (given(p.freq)) throw h.err('NotImplementedError', 'groupby.pct_change(freq=...) is not available in this sandbox')
    return g.rowwise(g.valueCols(), (s) => {
      const kw: Record<string, Value> = {}
      if (p.fill_method) kw.fill_method = p.fill_method
      if (p.limit) kw.limit = p.limit
      const res = seriesIn(h.call(s.getAttr('pct_change', h)!, [p.periods ?? py.int(1)], kw))
      return { cells: res.values, dt: 'float64' }
    })
  }],
  rank: ['method ascending na_option pct', (g, p, h) => g.rowwise(g.valueCols(), (s) => {
    const kw: Record<string, Value> = {}
    for (const k of ['method', 'ascending', 'na_option', 'pct']) if (p[k]) kw[k] = p[k]!
    const res = seriesIn(h.call(s.getAttr('rank', h)!, [], kw))
    return { cells: res.values, dt: 'float64' }
  })],
  ffill: ['limit', (g, p, h) => g.rowwise(g.valueCols(), fillKernel(true, given(p.limit) ? h.toInt(p.limit!) : null))],
  bfill: ['limit', (g, p, h) => g.rowwise(g.valueCols(), fillKernel(false, given(p.limit) ? h.toInt(p.limit!) : null))],
  fillna: ['value method', (g, p, h) => {
    if (given(p.method)) return h.str(p.method!).startsWith('f') || h.str(p.method!) === 'pad' ? g.rowwise(g.valueCols(), fillKernel(true, null)) : g.rowwise(g.valueCols(), fillKernel(false, null))
    const v = cellOf(h, need(h, p.value, 'fillna', 'value'))
    return g.rowwise(g.valueCols(), (s) => ({ cells: s.values.map((c) => (isMissing(c) ? v : c)), dt: commonDT([s.dtype, inferDT([v])]) }))
  }],
  head: ['n', (g, p, h) => {
    const n = p.n ? h.toInt(p.n) : 5
    const gr = g.groups()
    const keep = gr.members.flatMap((rows) => (n >= 0 ? rows.slice(0, n) : rows.slice(0, Math.max(0, rows.length + n))))
    return pickRows(g, keep)
  }],
  tail: ['n', (g, p, h) => {
    const n = p.n ? h.toInt(p.n) : 5
    const gr = g.groups()
    const keep = gr.members.flatMap((rows) => (n === 0 ? [] : n > 0 ? rows.slice(-n) : rows.slice(-n)))
    return pickRows(g, keep)
  }],
  nth: ['n dropna', (g, p, h) => {
    if (given(p.dropna)) throw h.err('NotImplementedError', 'groupby.nth(dropna=...) is not available in this sandbox')
    const nv = need(h, p.n, 'nth', 'n')
    const ns = isListLike(nv) ? typedCells(h, nv).cells.map(Number) : [h.toInt(nv)]
    const gr = g.groups()
    const keep = gr.members.flatMap((rows) => ns.flatMap((n) => {
      const r = rows[n < 0 ? rows.length + n : n]
      return r === undefined ? [] : [r]
    }))
    return pickRows(g, keep)
  }],
  get_group: ['name obj', (g, p, h) => {
    if (given(p.obj)) throw h.err('NotImplementedError', 'get_group(obj=...) is not available in this sandbox')
    const nameV = need(h, p.name, 'get_group', 'name')
    const want = nameV.k === 'tuple' ? nameV.items.map((x) => cellOf(h, x)) : [cellOf(h, nameV)]
    const gr = g.groups()
    const i = gr.keys.findIndex((k) => k.length === want.length && k.every((c, j) => c === want[j]))
    if (i < 0 || !gr.members[i].length) throw h.err('KeyError', repr(nameV))
    return g.single ? py.obj(g.f.data[g.valueCols()[0]].pick(gr.members[i])) : py.obj(g.f.take(gr.members[i], g.select ? g.valueCols() : null))
  }],
  describe: ['percentiles', (g, p, h) => {
    const gr = g.groups()
    const index = g.keyIndex(gr)
    const kw: Record<string, Value> = {}
    if (p.percentiles) kw.percentiles = p.percentiles
    const describeOf = (s: Series) => seriesIn(h.call(s.getAttr('describe', h)!, [], kw))
    const cols = g.single ? g.valueCols() : (() => {
      const num = g.numericOnly(g.valueCols())
      return num.length ? num.filter((k) => g.f.data[k].dtype !== 'bool') : g.valueCols()
    })()
    const stats = cols.map((k) => gr.members.filter((m) => m.length).map((m) => describeOf(g.f.data[k].pick(m))))
    const labels = stats[0][0].index.labels
    const groupIdx = gr.members.every((m) => m.length) ? index : g.keyIndex({ keys: gr.keys.filter((_, i) => gr.members[i].length), members: [], ofRow: [] })
    if (g.single) {
      return py.obj(new Frame(labels, labels.map((l, j) => new Series(stats[0].map((d) => d.values[j]), groupIdx, l, stats[0][0].dtype)), groupIdx))
    }
    const mi = IndexObj.fromArrays([cols.flatMap((k) => labels.map(() => g.f.columns.labels[k])), cols.flatMap(() => labels)], [null, null], [g.f.columns.dt, 'object'])
    const data = cols.flatMap((_, ci) => labels.map((_l, j) => new Series(stats[ci].map((d) => d.values[j]), groupIdx, null, stats[ci][0].dtype)))
    return py.obj(new Frame(mi, data, groupIdx))
  }],
  value_counts: ['normalize sort ascending dropna bins subset', (g, p, h) => {
    if (given(p.bins)) throw h.err('NotImplementedError', 'groupby.value_counts(bins=...) is not available in this sandbox')
    if (!g.single) return frameValueCounts(g, p, h)
    const gr = g.groups()
    const k = g.valueCols()[0]
    const s = g.f.data[k]
    const norm = flag(h, p.normalize)
    const sort = flag(h, p.sort, true)
    const asc = flag(h, p.ascending)
    const dropna = flag(h, p.dropna, true)
    const arrays: Cell[][] = [...g.gkeys.map(() => [] as Cell[]), []]
    const vals: number[] = []
    gr.members.forEach((rows, i) => {
      const present = rows.filter((r) => !dropna || !isMissing(s.values[r]))
      const counts = new Map<string, { c: Cell; n: number }>()
      for (const r of present) {
        const key = cellStr(s.values[r])
        const e = counts.get(key)
        if (e) e.n++
        else counts.set(key, { c: s.values[r], n: 1 })
      }
      let entries = [...counts.values()]
      entries = sortOrder([entries.map((e) => e.c)], [true]).map((x) => entries[x])
      if (sort) entries = sortOrder([entries.map((e) => e.n)], [asc]).map((x) => entries[x])
      for (const e of entries) {
        g.gkeys.forEach((_, j) => arrays[j].push(gr.keys[i][j]))
        arrays[g.gkeys.length].push(e.c)
        vals.push(norm ? e.n / present.length : e.n)
      }
    })
    const index = IndexObj.fromArrays(arrays, [...g.gkeys.map((x) => x.name), g.f.columns.labels[k]], [...g.gkeys.map((x) => x.dt), s.dtype])
    return py.obj(new Series(vals, index, norm ? 'proportion' : 'count', norm ? 'float64' : 'int64'))
  }],
  pipe: ['func * **', (g, p, h) => {
    const args = p.__args && p.__args.k === 'tuple' ? p.__args.items : []
    const kw = p.__kw && p.__kw.k === 'dict' ? Object.fromEntries(p.__kw.d.items().map(([k, v]) => [h.str(k), v])) : {}
    return h.call(need(h, p.func, 'pipe', 'func'), [py.obj(g), ...args], kw)
  }],
}

const pickRows = (g: GroupBy, rows: number[]): Value => {
  const sorted = [...rows].sort((a, b) => a - b)
  return g.single ? py.obj(g.f.data[g.valueCols()[0]].pick(sorted)) : py.obj(g.f.take(sorted, g.select ? g.valueCols() : null))
}

/** by= into grouping keys */
const keysFrom = (h: Host, f: Frame, by: Value | undefined, level: Value | undefined, isSeries: boolean): { keys: GKey[]; isList: boolean } => {
  const n = f.nrows
  const fromLevel = (l: Value): GKey => {
    const c = cellOf(h, l)
    const j = typeof c === 'number' ? (c < 0 ? c + f.index.nlevels : c) : f.index.names.indexOf(c)
    if (j < 0 || j >= f.index.nlevels) throw h.err(typeof c === 'number' ? 'ValueError' : 'KeyError', typeof c === 'number' ? `level > 0 or level < -1 only valid with MultiIndex` : `'Level ${labelStr(c)} not found'`)
    const lev = f.index.level(j)
    return { name: lev.name, values: lev.labels, dt: lev.dt, cats: null }
  }
  if (given(level)) {
    const ls = isListLike(level!) ? h.iterate(level!) : [level!]
    return { keys: ls.map(fromLevel), isList: isListLike(level!) }
  }
  if (!given(by)) throw h.err('TypeError', "You have to supply one of 'by' and 'level'")
  const one = (b: Value): GKey => {
    if (b.k === 'obj' && b.o instanceof Series) {
      const s = b.o
      const vals = s.index.equals(f.index) ? s.values : f.index.labels.map((l) => {
        const at = s.index.locate(l)
        return at.length ? s.values[at[0]] : NaN
      })
      return { name: s.name, values: vals, dt: s.dtype, cats: s.cats }
    }
    if (b.k === 'obj' && b.o instanceof IndexObj) return { name: b.o.name, values: b.o.labels, dt: b.o.dt, cats: null }
    if (b.k === 'array') {
      const t = typedCells(h, b)
      if (t.cells.length !== n) throw h.err('ValueError', `Length of values (${t.cells.length}) does not match length of index (${n})`)
      return { name: null, values: t.cells, dt: t.dt, cats: null }
    }
    if (b.k === 'dict') {
      const cells = f.index.labels.map((l) => {
        const v = b.d.get(valueOf(l, f.index.dt))
        return v ? cellOf(h, v) : NaN
      })
      return { name: null, values: cells, dt: inferDT(cells), cats: null }
    }
    if (b.k === 'func' || b.k === 'fn' || b.k === 'bound' || b.k === 'type') {
      const t = inferValues(h, f.index.labels.map((l) => h.call(b, [valueOf(l, f.index.dt)])))
      return { name: null, values: t.cells, dt: t.dt, cats: null }
    }
    const c = cellOf(h, b)
    const col = isSeries ? null : f.col(c)
    if (!col) {
      if (f.index.names.includes(c) && c !== null) return fromLevel(b)
      throw h.err('KeyError', repr(b))
    }
    return { name: c, values: col.values, dt: col.dtype, cats: col.cats }
  }
  if (by!.k === 'list' || by!.k === 'tuple') {
    const items = by!.items
    const allLabels = items.every((x) => !(x.k === 'obj' || x.k === 'array' || x.k === 'list' || x.k === 'func' || x.k === 'fn') && (isSeries ? false : f.columns.has(cellOf(h, x)) || f.index.names.includes(cellOf(h, x))))
    const hasStructured = items.some((x) => x.k === 'obj' || x.k === 'array' || x.k === 'list' || x.k === 'func' || x.k === 'fn')
    if (!allLabels && !hasStructured && items.length === n && (isSeries || !items.some((x) => x.k === 'str' && f.columns.has(x.v)))) {
      const t = inferValues(h, items)
      return { keys: [{ name: null, values: t.cells, dt: t.dt, cats: null }], isList: false }
    }
    return { keys: items.map((x) => (x.k === 'list' ? one(py.arr(NDArray.create(typedCells(h, x).cells.map(numCell), [x.items.length], 'float64'))) : one(x))), isList: true }
  }
  return { keys: [one(by!)], isList: false }
}

const optsFrom = (h: Host, p: Params): GOpts => ({ asIndex: flag(h, p.as_index, true), sort: flag(h, p.sort, true), dropna: flag(h, p.dropna, true), groupKeys: flag(h, p.group_keys, true), observed: flag(h, p.observed, false) })

export const groupByFrame = (h: Host, f: Frame, p: Params): Value => {
  if (given(p.axis) && h.str(p.axis!) !== '0' && h.str(p.axis!) !== 'index') throw h.err('NotImplementedError', 'groupby(axis=1) is not available in this sandbox')
  const { keys, isList } = keysFrom(h, f, p.by, p.level, false)
  const g = new GroupBy(f, keys, null, false, optsFrom(h, p), h.nameOf(py.obj(f), 'df'))
  g.keyIsList = isList
  return py.obj(g)
}

export const groupBySeries = (h: Host, s: Series, p: Params): Value => {
  const f = new Frame([s.name ?? '__values__'], [s], s.index)
  const { keys, isList } = keysFrom(h, f, p.by, p.level, true)
  if (!flag(h, p.as_index, true)) throw h.err('TypeError', 'as_index=False only valid with DataFrame')
  const g = new GroupBy(f, keys.map((k) => ({ ...k })), [f.columns.labels[0]], true, optsFrom(h, p), h.nameOf(py.obj(s), 's'), true)
  g.keyIsList = isList
  return py.obj(new GroupByNamed(g, s.name))
}

/** a Series' groupby: results carry the Series' own (possibly missing) name */
class GroupByNamed extends PyObj {
  readonly g: GroupBy
  readonly name: Cell
  constructor(g: GroupBy, name: Cell) {
    super()
    this.g = g
    this.name = name
  }
  get cls() { return 'SeriesGroupBy' }
  repr() { return this.g.repr() }
  len() { return this.g.len() }
  iter() {
    return this.g.iter().map((t) => {
      const [k, v] = (t as { items: Value[] }).items
      return py.tuple([k, this.fix(v)])
    })
  }
  private fix(v: Value): Value {
    if (v.k === 'obj' && v.o instanceof Series && v.o.name === '__values__') {
      const out = v.o.copy()
      out.name = this.name
      return py.obj(out)
    }
    return v
  }
  getAttr(name: string, h: Host): Value | undefined {
    const a = this.g.getAttr(name, h)
    if (a && a.k === 'fn') return { k: 'fn', name, call: (args, kw) => this.fix(a.call(args.map((x) => x), kw)) }
    return a
  }
}

// ---------------------------------------------------------------- windows

/** [start, end) of each window, as pandas' FixedWindowIndexer / ExpandingIndexer */
const windowBounds = (n: number, size: number | null, center: boolean, closed: string): { start: number[]; end: number[] } => {
  if (size === null) return { start: new Array(n).fill(0), end: Array.from({ length: n }, (_, i) => i + 1) }
  const offset = center ? Math.floor((size - 1) / 2) : 0
  const clip = (x: number) => Math.max(0, Math.min(n, x))
  const end0 = Array.from({ length: n }, (_, i) => i + 1 + offset)
  const start = end0.map((e) => clip(e - size - (closed === 'left' || closed === 'both' ? 1 : 0)))
  const end = end0.map((e) => clip(e - (closed === 'left' || closed === 'neither' ? 1 : 0)))
  return { start, end }
}

const rollSum = (v: number[], start: number[], end: number[], minp: number): number[] => {
  const out: number[] = []
  let nobs = 0
  let sum = 0
  let compAdd = 0
  let compRem = 0
  let same = 0
  let prev = 0
  const add = (x: number) => {
    if (Number.isNaN(x)) return
    nobs++
    const y = x - compAdd
    const t = sum + y
    compAdd = t - sum - y
    sum = t
    if (x === prev) same++
    else same = 1
    prev = x
  }
  const rem = (x: number) => {
    if (Number.isNaN(x)) return
    nobs--
    const y = -x - compRem
    const t = sum + y
    compRem = t - sum - y
    sum = t
  }
  for (let i = 0; i < start.length; i++) {
    const s = start[i]
    const e = end[i]
    if (i === 0 || s >= end[i - 1]) {
      prev = v[s]
      same = 0
      sum = compAdd = compRem = 0
      nobs = 0
      for (let j = s; j < e; j++) add(v[j])
    } else {
      for (let j = start[i - 1]; j < s; j++) rem(v[j])
      for (let j = end[i - 1]; j < e; j++) add(v[j])
    }
    out.push(nobs === 0 && minp === 0 ? 0 : nobs >= minp ? (same >= nobs ? prev * nobs : sum) : NaN)
  }
  return out
}

const rollMean = (v: number[], start: number[], end: number[], minp: number): number[] => {
  const out: number[] = []
  let nobs = 0
  let neg = 0
  let sum = 0
  let compAdd = 0
  let compRem = 0
  let same = 0
  let prev = 0
  const add = (x: number) => {
    if (Number.isNaN(x)) return
    nobs++
    const y = x - compAdd
    const t = sum + y
    compAdd = t - sum - y
    sum = t
    if (x < 0 || Object.is(x, -0)) neg++
    if (x === prev) same++
    else same = 1
    prev = x
  }
  const rem = (x: number) => {
    if (Number.isNaN(x)) return
    nobs--
    const y = -x - compRem
    const t = sum + y
    compRem = t - sum - y
    sum = t
    if (x < 0 || Object.is(x, -0)) neg--
  }
  for (let i = 0; i < start.length; i++) {
    const s = start[i]
    const e = end[i]
    if (i === 0 || s >= end[i - 1]) {
      prev = v[s]
      same = 0
      sum = compAdd = compRem = 0
      nobs = neg = 0
      for (let j = s; j < e; j++) add(v[j])
    } else {
      for (let j = start[i - 1]; j < s; j++) rem(v[j])
      for (let j = end[i - 1]; j < e; j++) add(v[j])
    }
    let r = NaN
    if (nobs >= minp && nobs > 0) {
      r = sum / nobs
      if (same >= nobs) r = prev
      else if (neg === 0 && r < 0) r = 0
      else if (neg === nobs && r > 0) r = 0
    }
    out.push(r)
  }
  return out
}

const rollVar = (v: number[], start: number[], end: number[], minp0: number, ddof: number): number[] => {
  const minp = Math.max(minp0, 1)
  const out: number[] = []
  let nobs = 0
  let mean = 0
  let ssq = 0
  let compAdd = 0
  let compRem = 0
  let same = 0
  let prev = 0
  const add = (x: number) => {
    if (Number.isNaN(x)) return
    nobs++
    if (x === prev) same++
    else same = 1
    prev = x
    const pm = mean - compAdd
    const y = x - compAdd
    const t = y - mean
    compAdd = t + mean - y
    const delta = t
    mean = nobs ? mean + delta / nobs : 0
    ssq += (x - pm) * (x - mean)
  }
  const rem = (x: number) => {
    if (Number.isNaN(x)) return
    nobs--
    if (nobs) {
      const pm = mean - compRem
      const y = x - compRem
      const t = y - mean
      compRem = t + mean - y
      const delta = t
      mean -= delta / nobs
      ssq -= (x - pm) * (x - mean)
    } else {
      mean = 0
      ssq = 0
    }
  }
  for (let i = 0; i < start.length; i++) {
    const s = start[i]
    const e = end[i]
    if (i === 0 || s >= end[i - 1]) {
      prev = v[s]
      same = 0
      mean = ssq = nobs = compAdd = compRem = 0
      for (let j = s; j < e; j++) add(v[j])
    } else {
      for (let j = start[i - 1]; j < s; j++) rem(v[j])
      for (let j = end[i - 1]; j < e; j++) add(v[j])
    }
    let r = NaN
    if (nobs >= minp && nobs > ddof) r = nobs === 1 || same >= nobs ? 0 : ssq / (nobs - ddof)
    out.push(r)
  }
  return out
}

const rollEach = (v: number[], start: number[], end: number[], minp: number, f: (present: number[]) => number): number[] => start.map((s, i) => {
  const present: number[] = []
  for (let j = s; j < end[i]; j++) if (!Number.isNaN(v[j])) present.push(v[j])
  return present.length >= minp && present.length ? f(present) : NaN
})

export class Window extends PyObj {
  readonly obj: Series | Frame
  readonly size: number | null
  readonly minp: number | null
  readonly center: boolean
  readonly closed: string
  constructor(obj: Series | Frame, size: number | null, minp: number | null, center: boolean, closed: string) {
    super()
    this.obj = obj
    this.size = size
    this.minp = minp
    this.center = center
    this.closed = closed
  }
  get cls() { return this.size === null ? 'Expanding' : 'Rolling' }
  repr() {
    if (this.size === null) return `Expanding [min_periods=${this.minp ?? 1},axis=0,method=single]`
    const parts = [`window=${this.size}`, ...(this.minp !== null ? [`min_periods=${this.minp}`] : []), `center=${this.center ? 'True' : 'False'}`, 'axis=0', ...(this.closed !== 'right' ? [`closed=${this.closed}`] : []), 'method=single']
    return `Rolling [${parts.join(',')}]`
  }
  private run(h: Host, kernel: (v: number[], s: number[], e: number[], minp: number) => number[], minpOverride?: number, numeric = (s: Series) => s.values.map(numCell)): Value {
    const one = (s: Series): Series => {
      if (s.dtype === 'object' && s.values.some((c) => typeof c === 'string')) throw h.err('DataError', 'Cannot aggregate non-numeric type: object')
      const { start, end } = windowBounds(s.length, this.size, this.center, this.closed)
      const minp = minpOverride ?? this.minp ?? (this.size ?? 1)
      return s.with(kernel(numeric(s), start, end, minp), 'float64')
    }
    if (this.obj instanceof Series) return py.obj(one(this.obj))
    const f = this.obj
    return py.obj(f.same(f.data.map(one)))
  }
  getAttr(name: string, h: Host): Value | undefined {
    const m = (spec: string, impl: (p: Params) => Value): Value => ({ k: 'fn', name, call: (args, kw) => {
      const p = bindParams(h, this.cls, name, spec, args, kw)
      if (flag(h, p.numeric_only)) {
        if (this.obj instanceof Frame) {
          const cols = this.obj.data.map((_, k) => k).filter((k) => (this.obj as Frame).data[k].dtype !== 'object')
          const w = new Window(this.obj.take(null, cols), this.size, this.minp, this.center, this.closed)
          const rest = { ...kw }
          delete rest.numeric_only
          return h.call(w.getAttr(name, h)!, args, rest)
        }
      }
      return quietly(h, () => impl(p))
    } })
    switch (name) {
      case 'sum': return m('numeric_only engine engine_kwargs', () => this.run(h, rollSum))
      case 'mean': return m('numeric_only engine engine_kwargs', () => this.run(h, rollMean))
      case 'var': return m('ddof numeric_only engine engine_kwargs', (p) => this.run(h, (v, s, e, minp) => rollVar(v, s, e, minp, given(p.ddof) ? h.toInt(p.ddof!) : 1)))
      case 'std': return m('ddof numeric_only engine engine_kwargs', (p) => this.run(h, (v, s, e, minp) => rollVar(v, s, e, minp, given(p.ddof) ? h.toInt(p.ddof!) : 1).map((x) => (x < 0 ? 0 : Math.sqrt(x)))))
      case 'min': return m('numeric_only engine engine_kwargs', () => this.run(h, (v, s, e, minp) => rollEach(v, s, e, minp, (x) => Math.min(...x))))
      case 'max': return m('numeric_only engine engine_kwargs', () => this.run(h, (v, s, e, minp) => rollEach(v, s, e, minp, (x) => Math.max(...x))))
      case 'median': return m('numeric_only engine engine_kwargs', () => this.run(h, (v, s, e, minp) => rollEach(v, s, e, minp, (x) => npQuantile([...x].sort((a, b) => a - b), 0.5))))
      case 'count': return m('numeric_only', () => this.run(h, rollSum, this.minp ?? (this.size === null ? 0 : undefined), (x) => x.values.map((c) => (isMissing(c) ? 0 : 1))))
      case 'quantile': return m('q interpolation numeric_only', (p) => {
        const q = h.num(need(h, p.q, 'quantile', 'q'))
        const interp = given(p.interpolation) ? h.str(p.interpolation!) : 'linear'
        return this.run(h, (v, s, e, minp) => rollEach(v, s, e, minp, (x) => {
          const sorted = [...x].sort((a, b) => a - b)
          const idx = q * (sorted.length - 1)
          const lo = Math.trunc(idx)
          if (idx === lo || interp === 'lower') return sorted[lo]
          if (interp === 'higher') return sorted[lo + 1]
          if (interp === 'midpoint') return (sorted[lo] + sorted[lo + 1]) / 2
          if (interp === 'nearest') return idx - lo < 0.5 || (idx - lo === 0.5 && lo % 2 === 0) ? sorted[lo] : sorted[lo + 1]
          return sorted[lo] + (sorted[lo + 1] - sorted[lo]) * (idx - lo)
        }))
      })
      case 'apply': return m('func raw engine engine_kwargs args kwargs', (p) => {
        const f = need(h, p.func, 'apply', 'func')
        const raw = flag(h, p.raw)
        const extra = p.args && (p.args.k === 'tuple' || p.args.k === 'list') ? p.args.items : []
        const kwargs = p.kwargs && p.kwargs.k === 'dict' ? Object.fromEntries(p.kwargs.d.items().map(([k, v]) => [h.str(k), v])) : {}
        const one = (s: Series): Series => {
          if (s.dtype === 'object' && s.values.some((c) => typeof c === 'string')) throw h.err('DataError', 'Cannot aggregate non-numeric type: object')
          const { start, end } = windowBounds(s.length, this.size, this.center, this.closed)
          const minp = this.minp ?? (this.size ?? 1)
          const vals = s.values.map(numCell)
          const out = start.map((a, i) => {
            const rows = Array.from({ length: end[i] - a }, (_, k) => a + k)
            const nobs = rows.filter((r) => !Number.isNaN(vals[r])).length
            if (nobs < minp || !rows.length) return NaN
            const w = s.pick(rows)
            return h.num(h.call(f, [raw ? py.arr(w.toArray()) : py.obj(w), ...extra], kwargs))
          })
          return s.with(out, 'float64')
        }
        if (this.obj instanceof Series) return py.obj(one(this.obj))
        return py.obj(this.obj.same(this.obj.data.map(one)))
      })
      case 'agg': case 'aggregate': return m('func', (p) => {
        const f = need(h, p.func, name, 'func')
        const fs = isListLike(f) ? h.iterate(f) : [f]
        const results = fs.map((x) => (x.k === 'str' ? h.call(this.getAttr(x.v, h) ?? (() => { throw h.err('AttributeError', `'${x.v}' is not a valid function for '${this.cls}' object`) })(), []) : h.call(this.getAttr('apply', h)!, [x])))
        if (!isListLike(f)) return results[0]
        if (this.obj instanceof Series) return py.obj(new Frame(fs.map((x) => funcName(h, x)), results.map((r) => seriesIn(r)), this.obj.index))
        const fr = this.obj
        const mi = IndexObj.fromArrays([fr.columns.labels.flatMap((c) => fs.map(() => c)), fr.columns.labels.flatMap(() => fs.map((x) => funcName(h, x)))], [null, null], [fr.columns.dt, 'object'])
        return py.obj(new Frame(mi, fr.columns.labels.flatMap((c) => results.map((r) => frameIn(r).col(c)!)), fr.index))
      })
    }
    return undefined
  }
}

export const windowOf = (h: Host, obj: Series | Frame, p: Params, kind: 'rolling' | 'expanding'): Value => {
  if (kind === 'expanding') {
    if (given(p.axis) && h.toInt(p.axis!) !== 0) throw h.err('NotImplementedError', 'expanding(axis=1) is not available in this sandbox')
    if (given(p.method) && h.str(p.method!) !== 'single') throw h.err('NotImplementedError', `expanding(method='${h.str(p.method!)}') is not available in this sandbox`)
    return py.obj(new Window(obj, null, given(p.min_periods) ? h.toInt(p.min_periods!) : 1, false, 'right'))
  }
  for (const k of ['win_type', 'on', 'step', 'axis']) if (given(p[k])) throw h.err('NotImplementedError', `rolling(${k}=...) is not available in this sandbox`)
  if (given(p.method) && h.str(p.method!) !== 'single') throw h.err('NotImplementedError', `rolling(method='${h.str(p.method!)}') is not available in this sandbox`)
  const wv = need(h, p.window, 'rolling', 'window')
  if (wv.k !== 'int') throw h.err('ValueError', 'window must be an integer 0 or greater')
  if (wv.v < 0) throw h.err('ValueError', 'window must be an integer 0 or greater')
  const minp = given(p.min_periods) ? h.toInt(p.min_periods!) : null
  if (minp !== null && minp < 0) throw h.err('ValueError', 'min_periods must be >= 0')
  if (minp !== null && minp > wv.v) throw h.err('ValueError', `min_periods ${minp} must be <= window ${wv.v}`)
  const closed = given(p.closed) ? h.str(p.closed!) : 'right'
  if (!['right', 'left', 'both', 'neither'].includes(closed)) throw h.err('ValueError', "closed must be 'right', 'left', 'both' or 'neither'")
  return py.obj(new Window(obj, wv.v, minp, flag(h, p.center), closed))
}

