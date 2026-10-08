/**
 * pandas for the sandbox: Series and DataFrame with pandas' printing, selection (df[…], loc, iloc, boolean masks),
 * cleaning (isna / fillna / dropna), groupby, value_counts, crosstab, get_dummies, merge and describe — the table work
 * the course does around its models (Naive Bayes counts, standardising features, K-Means centroids, confusion
 * matrices). Each table operation is recorded with the rows / columns it picked or the groups it formed.
 */
import { NDArray, PyError } from './ndarray'
import { PyDict, PyObj, isNum, py, repr } from './minipy'
import type { Host, Kw, PyLib, Value } from './minipy'
import type { TableSpec } from './pyEvents'

type Cell = number | string | boolean | null
type DT = 'int64' | 'float64' | 'bool' | 'object'

/** a teaching note, once per run */
const noteOnce = (h: Host, key: string) => {
  const seen = (h.state.get('pandas.notes') as Set<string> | undefined) ?? new Set<string>()
  h.state.set('pandas.notes', seen)
  if (seen.has(key) || !h.tracing) return
  seen.add(key)
  h.emit({ type: 'note', tone: 'info', key, params: {} })
}

const isMissing = (c: Cell) => c === null || (typeof c === 'number' && Number.isNaN(c))

// ---------------------------------------------------------------- dtype and conversion

/** pandas' dtype for a column of Python values */
const inferDT = (cells: Cell[]): DT => {
  const present = cells.filter((c) => !isMissing(c))
  const missing = present.length < cells.length
  if (!present.length) return cells.length ? 'float64' : 'object'
  if (present.every((c) => typeof c === 'boolean')) return missing ? 'object' : 'bool'
  if (present.every((c) => typeof c === 'number')) return !missing && present.every((c) => Number.isInteger(c)) ? 'int64' : 'float64'
  return 'object'
}

/** numeric columns store missing values as NaN */
const normalise = (cells: Cell[], dt: DT): Cell[] => (dt === 'float64' ? cells.map((c) => (c === null ? NaN : typeof c === 'boolean' ? Number(c) : c)) : cells)

const cellOf = (h: Host, v: Value): Cell => {
  switch (v.k) {
    case 'int': case 'float': return v.v
    case 'bool': return v.v
    case 'str': return v.v
    case 'none': return null
    case 'array': if (v.a.size === 1) return v.a.dtype === 'bool' ? v.a.values()[0] !== 0 : v.a.values()[0]
  }
  return h.str(v)
}

const valueOf = (c: Cell, dt: DT): Value => {
  if (c === null) return py.NONE
  if (typeof c === 'boolean') return py.bool(c)
  if (typeof c === 'string') return py.str(c)
  return dt === 'int64' && Number.isInteger(c) ? py.int(c) : py.float(c)
}

/** a list / array / Series as cells */
const cellsOf = (h: Host, v: Value): Cell[] => {
  if (v.k === 'obj' && v.o instanceof Series) return [...v.o.values]
  if (v.k === 'array') return v.a.dtype === 'bool' ? v.a.values().map((x) => x !== 0) : v.a.values()
  if (v.k === 'range' || v.k === 'list' || v.k === 'tuple') return h.iterate(v).map((x) => cellOf(h, x))
  if (v.k === 'obj' && v.o.toArray) return v.o.toArray().values()
  throw h.err('TypeError', `expected a list, array or Series, got ${v.k === 'obj' ? v.o.cls : v.k}`)
}

// ---------------------------------------------------------------- formatting (pandas' repr)

/** a column's cells as pandas prints them: floats share one number of decimals (6, trailing zeros trimmed together) */
const formatCells = (cells: Cell[], dt: DT): string[] => {
  if (dt === 'float64') {
    const fin = cells.filter((c): c is number => typeof c === 'number' && Number.isFinite(c))
    const big = fin.some((v) => Math.abs(v) >= 1e16) || (fin.some((v) => v !== 0 && Math.abs(v) < 1e-4) && fin.some((v) => Math.abs(v) >= 1))
    if (big) return cells.map((c) => (typeof c === 'number' && !Number.isNaN(c) ? c.toExponential(6).replace(/e([+-])(\d)$/, 'e$10$2') : 'NaN'))
    let dec = 6
    const fixed = (d: number) => fin.map((v) => v.toFixed(d))
    while (dec > 1 && fixed(dec).every((s) => s.endsWith('0'))) dec--
    return cells.map((c) => (typeof c === 'number' ? (Number.isNaN(c) ? 'NaN' : c === Infinity ? 'inf' : c === -Infinity ? '-inf' : c.toFixed(dec)) : 'NaN'))
  }
  return cells.map((c) => (c === null ? 'None' : typeof c === 'boolean' ? (c ? 'True' : 'False') : typeof c === 'number' ? (Number.isNaN(c) ? 'NaN' : String(c)) : c))
}

/** cells as they sit in a printed column: a leading space, or the minus sign of a negative number */
const printCells = (cells: Cell[], dt: DT): string[] =>
  formatCells(cells, dt).map((t, i) => (dt !== 'object' && typeof cells[i] === 'number' && t.startsWith('-') ? t : ` ${t}`))

const labelStr = (c: Cell) => (c === null ? 'None' : typeof c === 'boolean' ? (c ? 'True' : 'False') : typeof c === 'number' && Number.isNaN(c) ? 'NaN' : String(c))

const MAX_SHOW = 60

/** rows to print: all, or the first and last 5 with a '...' row */
const shownRows = (n: number): (number | -1)[] => (n > MAX_SHOW ? [0, 1, 2, 3, 4, -1, n - 5, n - 4, n - 3, n - 2, n - 1] : Array.from({ length: n }, (_, i) => i))

// ---------------------------------------------------------------- Index

class IndexObj extends PyObj {
  readonly cls = 'Index'
  labels: Cell[]
  name: string | null
  range: boolean
  constructor(labels: Cell[], name: string | null, range = false) {
    super()
    this.labels = labels
    this.name = name
    this.range = range
  }
  repr() {
    if (this.range) return `RangeIndex(start=0, stop=${this.labels.length}, step=1)`
    const dt = inferDT(this.labels)
    const items = this.labels.map((l) => (typeof l === 'string' ? `'${l}'` : labelStr(l))).join(', ')
    return `Index([${items}], dtype='${dt}'${this.name ? `, name='${this.name}'` : ''})`
  }
  len() { return this.labels.length }
  iter() { return this.labels.map((l) => valueOf(l, inferDT(this.labels))) }
  getItem(idx: Value, h: Host) {
    const i = h.toInt(idx)
    const n = this.labels.length
    if (i < -n || i >= n) throw h.err('IndexError', `index ${i} is out of bounds for axis 0 with size ${n}`)
    return valueOf(this.labels[i < 0 ? i + n : i], inferDT(this.labels))
  }
  contains(v: Value, h: Host) { return this.labels.includes(cellOf(h, v)) }
  getAttr(name: string) {
    if (name === 'tolist' || name === 'to_list') return { k: 'fn' as const, name, call: () => py.list(this.iter()) }
    if (name === 'name') return this.name === null ? py.NONE : py.str(this.name)
    if (name === 'values') return py.arr(NDArray.create(this.labels.map((l) => (typeof l === 'number' ? l : NaN)), [this.labels.length], 'int64'))
    return undefined
  }
}

const isRangeIndex = (idx: Cell[]) => idx.every((l, i) => l === i)

// ---------------------------------------------------------------- Series

export class Series extends PyObj {
  readonly cls = 'Series'
  values: Cell[]
  dtype: DT
  index: Cell[]
  name: Cell
  indexName: string | null
  constructor(values: Cell[], index: Cell[] | null, name: Cell = null, dtype?: DT, indexName: string | null = null) {
    super()
    this.dtype = dtype ?? inferDT(values)
    this.values = normalise(values, this.dtype)
    this.index = index ?? values.map((_, i) => i)
    this.name = name
    this.indexName = indexName
  }
  get length() { return this.values.length }
  repr() {
    if (!this.length) return `Series([], ${this.name !== null ? `Name: ${labelStr(this.name)}, ` : ''}dtype: ${this.dtype})`
    const rows = shownRows(this.length)
    const idx = rows.map((r) => (r < 0 ? '..' : labelStr(this.index[r])))
    const vals = printCells(this.values, this.dtype)
    const shownVals = rows.map((r) => (r < 0 ? ' ...' : vals[r]))
    const iw = Math.max(...idx.map((x) => x.length))
    const vw = Math.max(...shownVals.map((x) => x.length))
    const lines = rows.map((_, k) => `${idx[k].padEnd(iw)}   ${shownVals[k].padStart(vw)}`)
    const head = this.indexName ? [this.indexName] : []
    const foot = `${this.name !== null ? `Name: ${labelStr(this.name)}, ` : ''}${this.length > MAX_SHOW ? `Length: ${this.length}, ` : ''}dtype: ${this.dtype}`
    return [...head, ...lines, foot].join('\n')
  }
  display() { return { type: 'table' as const, table: tableOfSeries(this) } }
  len() { return this.length }
  iter() { return this.values.map((c) => valueOf(c, this.dtype)) }
  contains(v: Value, h: Host) { return this.index.includes(cellOf(h, v)) }
  toArray(): NDArray {
    if (this.dtype === 'object') throw new PyError('TypeError', `the Series '${labelStr(this.name)}' holds text, so it cannot become a numeric array`)
    return NDArray.create(this.values.map((c) => (typeof c === 'boolean' ? Number(c) : (c as number))), [this.length], this.dtype === 'bool' ? 'bool' : this.dtype === 'int64' ? 'int64' : 'float64')
  }
  isa(name: string) { return name === 'Series' }
  numeric(h: Host, what: string): number[] {
    if (this.dtype === 'object') throw h.err('TypeError', `Could not convert ${JSON.stringify(this.values.slice(0, 3).join(''))} to numeric (the Series '${labelStr(this.name)}' holds text; ${what} needs numbers)`)
    return this.values.map((c) => (typeof c === 'boolean' ? Number(c) : (c as number)))
  }
  with(values: Cell[], dtype?: DT, name: Cell = this.name) { return new Series(values, [...this.index], name, dtype, this.indexName) }
  /** position of each of my labels in other, for aligned arithmetic */
  align(o: Series): { idx: Cell[]; a: number[]; b: number[] } {
    if (o.index.length === this.index.length && o.index.every((l, i) => l === this.index[i])) return { idx: this.index, a: this.index.map((_, i) => i), b: this.index.map((_, i) => i) }
    const idx = [...this.index, ...o.index.filter((l) => !this.index.includes(l))]
    return { idx, a: idx.map((l) => this.index.indexOf(l)), b: idx.map((l) => o.index.indexOf(l)) }
  }
  binop(op: string, other: Value, reflected: boolean, h: Host): Value | undefined {
    if (other.k === 'obj' && other.o instanceof Frame) return undefined
    const f = binFn(h, op)
    if (!f) return undefined
    if (other.k === 'obj' && other.o instanceof Series) {
      const { idx, a, b } = this.align(other.o)
      const vals = idx.map((_, i) => {
        const x = a[i] < 0 ? NaN : this.values[a[i]]
        const y = b[i] < 0 ? NaN : other.o instanceof Series ? other.o.values[b[i]] : NaN
        return reflected ? f(y, x) : f(x, y)
      })
      return py.obj(new Series(vals, idx, this.name === (other.o as Series).name ? this.name : null, resultDT(op, this.dtype, (other.o as Series).dtype, vals)))
    }
    let ys: Cell[] | null = null
    if (other.k === 'list' || other.k === 'tuple' || other.k === 'array') {
      ys = cellsOf(h, other)
      if (ys.length !== this.length) throw h.err('ValueError', `Lengths must match to compare (${this.length} vs ${ys.length})`)
    }
    const y = ys ? null : cellOf(h, other)
    const vals = this.values.map((x, i) => (reflected ? f(ys ? ys[i] : y, x) : f(x, ys ? ys[i] : y)))
    const odt: DT = typeof y === 'string' ? 'object' : typeof y === 'number' && !Number.isInteger(y) ? 'float64' : this.dtype === 'bool' && typeof y === 'number' ? 'int64' : this.dtype
    const out = new Series(vals, [...this.index], this.name, resultDT(op, this.dtype, odt, vals), this.indexName)
    traceFrame(h, op.match(/^[<>=!]/) ? 'compare' : 'arith', [{ label: h.nameOf(py.obj(this), 's'), table: tableOfSeries(this) }], tableOfSeries(out))
    return py.obj(out)
  }
  unary(op: '-' | '+' | '~', h: Host): Value {
    if (op === '~') {
      if (this.dtype !== 'bool') throw h.err('TypeError', "bad operand type for unary ~: the Series is not boolean")
      return py.obj(this.with(this.values.map((c) => !c), 'bool'))
    }
    return py.obj(this.with(this.numeric(h, 'negation').map((v) => (op === '-' ? -v : v))))
  }
  getItem(idx: Value, h: Host): Value {
    if (idx.k === 'obj' && idx.o instanceof Series && idx.o.dtype === 'bool') return py.obj(this.pick(idx.o.values.flatMap((b, i) => (b ? [i] : []))))
    if (idx.k === 'slice') {
      const part = (x: Value) => (x.k === 'none' ? null : h.toInt(x))
      const s = part(idx.start) ?? 0
      const e = part(idx.stop) ?? this.length
      const n = this.length
      const lo = s < 0 ? Math.max(0, s + n) : Math.min(s, n)
      const hi = e < 0 ? Math.max(0, e + n) : Math.min(e, n)
      return py.obj(this.pick(Array.from({ length: Math.max(0, hi - lo) }, (_, k) => lo + k)))
    }
    const label = cellOf(h, idx)
    const at = this.index.indexOf(label)
    if (at < 0) {
      if (typeof label === 'number' && !this.index.some((l) => typeof l === 'number') && Number.isInteger(label)) return valueOf(this.values[label < 0 ? label + this.length : label], this.dtype)
      throw h.err('KeyError', repr(idx))
    }
    return valueOf(this.values[at], this.dtype)
  }
  setItem(idx: Value, v: Value, h: Host) {
    const c = cellOf(h, v)
    if (idx.k === 'obj' && idx.o instanceof Series && idx.o.dtype === 'bool') {
      idx.o.values.forEach((b, i) => { if (b) this.values[i] = c })
    } else {
      const at = this.index.indexOf(cellOf(h, idx))
      if (at < 0) {
        this.index.push(cellOf(h, idx))
        this.values.push(c)
      } else this.values[at] = c
    }
    this.dtype = inferDT(this.values)
    this.values = normalise(this.values, this.dtype)
  }
  pick(rows: number[]): Series {
    return new Series(rows.map((r) => this.values[r]), rows.map((r) => this.index[r]), this.name, this.dtype, this.indexName)
  }
  getAttr(name: string, h: Host): Value | undefined {
    const fn = (call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })
    const self = py.obj(this)
    const label = () => h.nameOf(self, 's')
    switch (name) {
      case 'name': return this.name === null ? py.NONE : valueOf(this.name, inferDT([this.name]))
      case 'dtype': return py.str(this.dtype)
      case 'shape': return py.tuple([py.int(this.length)])
      case 'size': return py.int(this.length)
      case 'index': return py.obj(new IndexObj(this.index, this.indexName, isRangeIndex(this.index)))
      case 'values': case 'to_numpy': {
        const a = this.dtype === 'object' ? null : this.toArray()
        if (!a) throw h.err('NotImplementedError', 'text Series cannot become NumPy arrays in this sandbox: use .tolist()')
        return name === 'values' ? py.arr(a) : fn(() => py.arr(a))
      }
      case 'tolist': case 'to_list': return fn(() => py.list(this.iter()))
      case 'str': return py.obj(new StrAccessor(this))
      case 'loc': case 'iloc': return py.obj(new SeriesIndexer(this, name))
      case 'head': case 'tail': return fn((args) => {
        const n = args[0] ? h.toInt(args[0]) : 5
        const rows = Array.from({ length: this.length }, (_, i) => i)
        return py.obj(this.pick(name === 'head' ? rows.slice(0, n) : rows.slice(Math.max(0, this.length - n))))
      })
      case 'mean': case 'sum': case 'min': case 'max': case 'std': case 'var': case 'median': case 'count': case 'prod':
        return fn((_a, kw) => {
          if ((name === 'std' || name === 'var') && !kw.ddof) noteOnce(h, 'pandas_ddof')
          return reduceCells(h, this.values, this.dtype, name, kw.ddof ? h.toInt(kw.ddof) : 1, `the Series '${labelStr(this.name)}'`)
        })
      case 'idxmax': case 'idxmin': return fn(() => {
        const v = this.numeric(h, name)
        let k = -1
        v.forEach((x, i) => { if (!Number.isNaN(x) && (k < 0 || (name === 'idxmax' ? x > v[k] : x < v[k]))) k = i })
        return valueOf(this.index[k], inferDT(this.index))
      })
      case 'value_counts': return fn((_a, kw) => {
        const norm = !!(kw.normalize && h.truthy(kw.normalize))
        const out = valueCounts(this, norm)
        traceFrame(h, 'value_counts', [{ label: label(), table: tableOfSeries(this) }], tableOfSeries(out))
        return py.obj(out)
      })
      case 'unique': return fn(() => py.list([...new Map(this.values.map((c) => [labelStr(c), c])).values()].map((c) => valueOf(c, this.dtype))))
      case 'nunique': return fn(() => py.int(new Set(this.values.filter((c) => !isMissing(c)).map(labelStr)).size))
      case 'isna': case 'isnull': case 'notna': case 'notnull': return fn(() => py.obj(this.with(this.values.map((c) => (name.startsWith('not') ? !isMissing(c) : isMissing(c))), 'bool')))
      case 'fillna': return fn((args, kw) => {
        const v = cellOf(h, kw.value ?? args[0])
        const out = this.with(this.values.map((c) => (isMissing(c) ? v : c)), inferDT(this.values.map((c) => (isMissing(c) ? v : c))))
        traceFrame(h, 'fillna', [{ label: label(), table: tableOfSeries(this) }], tableOfSeries(out), { rows: this.values.flatMap((c, i) => (isMissing(c) ? [i] : [])), cols: null })
        return py.obj(out)
      })
      case 'dropna': return fn(() => py.obj(this.pick(this.values.flatMap((c, i) => (isMissing(c) ? [] : [i])))))
      case 'map': return fn((args) => {
        const m = args[0]
        const vals = this.values.map((c) => {
          if (m.k === 'dict') {
            const hit = m.d.get(valueOf(c, this.dtype))
            return hit ? cellOf(h, hit) : null
          }
          return cellOf(h, h.call(m, [valueOf(c, this.dtype)]))
        })
        const out = this.with(vals, inferDT(vals))
        traceFrame(h, 'map', [{ label: label(), table: tableOfSeries(this) }], tableOfSeries(out))
        return py.obj(out)
      })
      case 'apply': return fn((args) => {
        const vals = this.values.map((c) => cellOf(h, h.call(args[0], [valueOf(c, this.dtype)])))
        return py.obj(this.with(vals, inferDT(vals)))
      })
      case 'astype': return fn((args) => py.obj(astypeSeries(h, this, h.str(args[0].k === 'type' ? py.str(args[0].name) : args[0]))))
      case 'round': return fn((args) => {
        const d = args[0] ? h.toInt(args[0]) : 0
        return py.obj(this.with(this.values.map((c) => (typeof c === 'number' ? roundHalfEven(c, d) : c)), this.dtype))
      })
      case 'abs': return fn(() => py.obj(this.with(this.numeric(h, 'abs').map(Math.abs))))
      case 'cumsum': return fn(() => {
        let acc = 0
        return py.obj(this.with(this.numeric(h, 'cumsum').map((v) => (acc += v))))
      })
      case 'sort_values': return fn((_a, kw) => {
        const asc = !kw.ascending || h.truthy(kw.ascending)
        const order = sortOrder([this.values], [asc])
        return py.obj(this.pick(order))
      })
      case 'sort_index': return fn(() => py.obj(this.pick(sortOrder([this.index], [true]))))
      case 'describe': return fn(() => py.obj(describeSeries(h, this)))
      case 'between': return fn((args) => {
        const lo = h.num(args[0])
        const hi = h.num(args[1])
        return py.obj(this.with(this.numeric(h, 'between').map((v) => v >= lo && v <= hi), 'bool'))
      })
      case 'copy': return fn(() => py.obj(this.with([...this.values])))
      case 'reset_index': return fn((_a, kw) => {
        if (kw.drop && h.truthy(kw.drop)) return py.obj(new Series([...this.values], null, this.name, this.dtype))
        const df = new Frame([this.indexName ?? 'index', labelStr(this.name ?? 0)], [new Series([...this.index], null), new Series([...this.values], null, null, this.dtype)], null)
        return py.obj(df)
      })
      case 'to_frame': return fn(() => py.obj(new Frame([labelStr(this.name ?? 0)], [this.with([...this.values])], [...this.index], this.indexName)))
      case 'plot': return undefined
    }
    return undefined
  }
}

const roundHalfEven = (v: number, d: number) => {
  const f = 10 ** d
  const x = v * f
  return (Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : Math.round(x)) / f
}

/** element-wise operation of two cells (missing propagates; comparisons with missing are False) */
const binFn = (h: Host, op: string): ((x: Cell, y: Cell) => Cell) | null => {
  const num = (c: Cell) => (typeof c === 'boolean' ? Number(c) : c)
  switch (op) {
    case '+': return (x, y) => (typeof x === 'string' && typeof y === 'string' ? x + y : isMissing(x) || isMissing(y) ? NaN : (num(x) as number) + (num(y) as number))
    case '-': return (x, y) => (isMissing(x) || isMissing(y) ? NaN : (num(x) as number) - (num(y) as number))
    case '*': return (x, y) => (isMissing(x) || isMissing(y) ? NaN : (num(x) as number) * (num(y) as number))
    case '/': return (x, y) => (isMissing(x) || isMissing(y) ? NaN : (num(x) as number) / (num(y) as number))
    case '//': return (x, y) => (isMissing(x) || isMissing(y) ? NaN : Math.floor((num(x) as number) / (num(y) as number)))
    case '%': return (x, y) => (isMissing(x) || isMissing(y) ? NaN : (num(x) as number) - Math.floor((num(x) as number) / (num(y) as number)) * (num(y) as number))
    case '**': return (x, y) => (isMissing(x) || isMissing(y) ? NaN : (num(x) as number) ** (num(y) as number))
    case '>': return (x, y) => !isMissing(x) && !isMissing(y) && (num(x) as number) > (num(y) as number)
    case '<': return (x, y) => !isMissing(x) && !isMissing(y) && (num(x) as number) < (num(y) as number)
    case '>=': return (x, y) => !isMissing(x) && !isMissing(y) && (num(x) as number) >= (num(y) as number)
    case '<=': return (x, y) => !isMissing(x) && !isMissing(y) && (num(x) as number) <= (num(y) as number)
    case '==': return (x, y) => x === y
    case '!=': return (x, y) => x !== y
    case '&': return (x, y) => !!x && !!y
    case '|': return (x, y) => !!x || !!y
  }
  void h
  return null
}

const resultDT = (op: string, a: DT, b: DT, vals: Cell[]): DT => {
  if (['>', '<', '>=', '<=', '==', '!=', '&', '|'].includes(op)) return 'bool'
  if (a === 'object' || b === 'object') return inferDT(vals)
  if (op === '/' || a === 'float64' || b === 'float64' || vals.some((v) => typeof v === 'number' && !Number.isInteger(v))) return 'float64'
  return 'int64'
}

const quantile = (sorted: number[], q: number) => {
  if (!sorted.length) return NaN
  const p = (sorted.length - 1) * q
  const lo = Math.floor(p)
  return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (p - lo)
}

/** mean / sum / std … of one column, skipping missing values (std and var use ddof=1, unlike NumPy) */
const reduceCells = (h: Host, cells: Cell[], dt: DT, how: string, ddof: number, what: string): Value => {
  if (how === 'count') return py.int(cells.filter((c) => !isMissing(c)).length)
  if (dt === 'object') {
    if (how === 'min' || how === 'max') {
      const s = cells.filter((c): c is string => typeof c === 'string').sort()
      return py.str(how === 'min' ? s[0] : s[s.length - 1])
    }
    if (how === 'sum') return py.str(cells.map((c) => labelStr(c)).join(''))
    throw h.err('TypeError', `Could not convert ${JSON.stringify(cells.slice(0, 4).map(labelStr).join(''))} to numeric: ${what} holds text`)
  }
  const v = cells.filter((c) => !isMissing(c)).map((c) => (typeof c === 'boolean' ? Number(c) : (c as number)))
  const n = v.length
  const sum = v.reduce((a, b) => a + b, 0)
  const asDT = (x: number) => (dt === 'int64' && Number.isInteger(x) && (how === 'sum' || how === 'min' || how === 'max' || how === 'prod') ? py.int(x) : dt === 'bool' && how === 'sum' ? py.int(x) : py.float(x))
  switch (how) {
    case 'sum': return asDT(sum)
    case 'prod': return asDT(v.reduce((a, b) => a * b, 1))
    case 'mean': return py.float(n ? sum / n : NaN)
    case 'min': return n ? asDT(Math.min(...v)) : py.float(NaN)
    case 'max': return n ? asDT(Math.max(...v)) : py.float(NaN)
    case 'median': return py.float(quantile([...v].sort((a, b) => a - b), 0.5))
    case 'std': case 'var': {
      if (n - ddof <= 0) return py.float(NaN)
      const m = sum / n
      const va = v.reduce((a, b) => a + (b - m) ** 2, 0) / (n - ddof)
      return py.float(how === 'std' ? Math.sqrt(va) : va)
    }
  }
  throw h.err('AttributeError', how)
}

const valueCounts = (s: Series, normalize: boolean): Series => {
  const counts = new Map<string, { c: Cell; n: number; first: number }>()
  s.values.forEach((c, i) => {
    if (isMissing(c)) return
    const k = labelStr(c)
    const e = counts.get(k)
    if (e) e.n++
    else counts.set(k, { c, n: 1, first: i })
  })
  const entries = [...counts.values()].sort((a, b) => b.n - a.n || a.first - b.first)
  const total = entries.reduce((a, e) => a + e.n, 0)
  return new Series(entries.map((e) => (normalize ? e.n / total : e.n)), entries.map((e) => e.c), normalize ? 'proportion' : 'count', normalize ? 'float64' : 'int64', s.name === null ? null : labelStr(s.name))
}

const describeSeries = (h: Host, s: Series): Series => {
  const v = s.numeric(h, 'describe').filter((x) => !Number.isNaN(x)).sort((a, b) => a - b)
  const n = v.length
  const mean = v.reduce((a, b) => a + b, 0) / n
  const std = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1))
  return new Series([n, mean, std, v[0], quantile(v, 0.25), quantile(v, 0.5), quantile(v, 0.75), v[n - 1]], ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], s.name, 'float64')
}

const astypeSeries = (h: Host, s: Series, t: string): Series => {
  switch (t) {
    case 'int': case 'int64': case 'int32': {
      if (s.values.some(isMissing)) throw h.err('ValueError', 'cannot convert float NaN to integer')
      return s.with(s.values.map((c) => (typeof c === 'string' ? (Number.isNaN(Number(c)) ? (() => { throw h.err('ValueError', `invalid literal for int() with base 10: '${c}'`) })() : Math.trunc(Number(c))) : Math.trunc(Number(c)))), 'int64')
    }
    case 'float': case 'float64': case 'float32': return s.with(s.values.map((c) => (c === null ? NaN : Number(c))), 'float64')
    case 'str': case 'object': return s.with(s.values.map((c) => labelStr(c)), 'object')
    case 'bool': return s.with(s.values.map((c) => !!c && c !== 0), 'bool')
    case 'category': return s
  }
  throw h.err('TypeError', `data type '${t}' not understood`)
}

/** stable sort order by several key columns */
const sortOrder = (keys: Cell[][], asc: boolean[]): number[] => {
  const n = keys[0]?.length ?? 0
  const cmp = (x: Cell, y: Cell) => {
    if (isMissing(x)) return isMissing(y) ? 0 : 1
    if (isMissing(y)) return -1
    if (typeof x === 'string' || typeof y === 'string') return String(x) < String(y) ? -1 : String(x) > String(y) ? 1 : 0
    return Number(x) - Number(y)
  }
  return Array.from({ length: n }, (_, i) => i).sort((i, j) => {
    for (let k = 0; k < keys.length; k++) {
      const c = cmp(keys[k][i], keys[k][j])
      // missing values go last whatever the direction
      if (c) return isMissing(keys[k][i]) || isMissing(keys[k][j]) ? c : asc[k] ? c : -c
    }
    return i - j
  })
}

class StrAccessor extends PyObj {
  readonly cls = 'StringMethods'
  s: Series
  constructor(s: Series) {
    super()
    this.s = s
  }
  repr() { return '<pandas.core.strings.accessor.StringMethods object>' }
  getAttr(name: string, h: Host): Value | undefined {
    const map = (f: (x: string, args: Value[]) => Cell, dt?: DT) => ({ k: 'fn' as const, name, call: (args: Value[]) => {
      const vals = this.s.values.map((c) => (typeof c === 'string' ? f(c, args) : null))
      return py.obj(this.s.with(vals, dt ?? inferDT(vals)))
    } })
    switch (name) {
      case 'lower': return map((x) => x.toLowerCase())
      case 'upper': return map((x) => x.toUpperCase())
      case 'strip': return map((x) => x.trim())
      case 'len': return map((x) => x.length, 'int64')
      case 'contains': return map((x, a) => x.includes(h.str(a[0])), 'bool')
      case 'startswith': return map((x, a) => x.startsWith(h.str(a[0])), 'bool')
      case 'endswith': return map((x, a) => x.endsWith(h.str(a[0])), 'bool')
      case 'replace': return map((x, a) => x.split(h.str(a[0])).join(h.str(a[1])))
    }
    return undefined
  }
}

class SeriesIndexer extends PyObj {
  readonly cls: string
  s: Series
  constructor(s: Series, kind: string) {
    super()
    this.s = s
    this.cls = kind === 'loc' ? '_LocIndexer' : '_iLocIndexer'
  }
  repr() { return `<pandas.core.indexing.${this.cls} object>` }
  getItem(idx: Value, h: Host): Value {
    const s = this.s
    if (this.cls === '_iLocIndexer') {
      if (idx.k === 'int') return valueOf(s.values[idx.v < 0 ? idx.v + s.length : idx.v], s.dtype)
      return py.obj(s.pick(positions(h, idx, s.length)))
    }
    if (idx.k === 'list' || (idx.k === 'obj' && idx.o instanceof Series)) {
      if (idx.k === 'obj' && idx.o instanceof Series && idx.o.dtype === 'bool') return s.getItem(idx, h)
      return py.obj(s.pick(h.iterate(idx).map((v) => s.index.indexOf(cellOf(h, v)))))
    }
    return s.getItem(idx, h)
  }
}

// ---------------------------------------------------------------- DataFrame

/** positions from an int / slice / list / boolean selector (iloc) */
const positions = (h: Host, sel: Value, n: number): number[] => {
  if (sel.k === 'slice') {
    const part = (x: Value) => (x.k === 'none' ? null : h.toInt(x))
    const st = part(sel.step) ?? 1
    const norm = (v: number | null, d: number) => (v === null ? d : v < 0 ? Math.max(v + n, 0) : Math.min(v, n))
    const a = norm(part(sel.start), st > 0 ? 0 : n - 1)
    const b = norm(part(sel.stop), st > 0 ? n : -1)
    const out: number[] = []
    for (let i = a; st > 0 ? i < b : i > b; i += st) out.push(i)
    return out
  }
  if (sel.k === 'int') {
    if (sel.v < -n || sel.v >= n) throw h.err('IndexError', 'single positional indexer is out-of-bounds')
    return [sel.v < 0 ? sel.v + n : sel.v]
  }
  if (sel.k === 'obj' && sel.o instanceof Series && sel.o.dtype === 'bool') return sel.o.values.flatMap((b, i) => (b ? [i] : []))
  const items = cellsOf(h, sel)
  if (items.every((c) => typeof c === 'boolean')) return items.flatMap((b, i) => (b ? [i] : []))
  return items.map((c) => {
    const i = Number(c)
    if (i < -n || i >= n) throw h.err('IndexError', 'positional indexers are out-of-bounds')
    return i < 0 ? i + n : i
  })
}

export class Frame extends PyObj {
  readonly cls = 'DataFrame'
  columns: string[]
  data: Series[]
  index: Cell[]
  indexName: string | null
  columnsName: string | null = null
  constructor(columns: string[], data: Series[], index: Cell[] | null, indexName: string | null = null) {
    super()
    this.columns = columns
    const n = data[0]?.length ?? (index ? index.length : 0)
    this.index = index ?? Array.from({ length: n }, (_, i) => i)
    this.indexName = indexName
    this.data = data.map((s, k) => new Series(s.values, this.index, columns[k], s.dtype, indexName))
  }
  get nrows() { return this.index.length }
  col(name: string): Series | null {
    const k = this.columns.indexOf(name)
    return k < 0 ? null : this.data[k]
  }
  repr() {
    if (!this.columns.length) return `Empty DataFrame\nColumns: []\nIndex: [${this.index.map(labelStr).join(', ')}]`
    const rows = shownRows(this.nrows)
    const idx = rows.map((r) => (r < 0 ? '..' : labelStr(this.index[r])))
    const iw = Math.max(this.columnsName?.length ?? 0, this.indexName?.length ?? 0, ...idx.map((x) => x.length))
    const cols = this.data.map((sr, k) => {
      const all = printCells(sr.values, sr.dtype)
      const cells = rows.map((r) => (r < 0 ? '...' : all[r]))
      // a non-text column also keeps a sign slot in front of its header
      const w = Math.max(this.columns[k].length + (sr.dtype === 'object' ? 0 : 1), ...cells.map((c) => c.length))
      return { head: this.columns[k].padStart(w), cells: cells.map((c) => c.padStart(w)) }
    })
    const lines = [`${(this.columnsName ?? '').padEnd(iw)} ${cols.map((c) => c.head).join(' ')}`]
    if (this.indexName) lines.push(`${this.indexName.padEnd(iw)} ${cols.map((c) => ' '.repeat(c.head.length)).join(' ')}`)
    rows.forEach((_, k) => lines.push(`${idx[k].padEnd(iw)} ${cols.map((c) => c.cells[k]).join(' ')}`))
    if (this.nrows > MAX_SHOW) lines.push('', `[${this.nrows} rows x ${this.columns.length} columns]`)
    return lines.join('\n')
  }
  display() { return { type: 'table' as const, table: tableOfFrame(this) } }
  len() { return this.nrows }
  iter() { return this.columns.map((c) => py.str(c)) }
  contains(v: Value, h: Host) { return this.columns.includes(h.str(v)) }
  isa(name: string) { return name === 'DataFrame' }
  toArray(): NDArray {
    const obj = this.data.find((s) => s.dtype === 'object')
    if (obj) throw new PyError('TypeError', `column '${labelStr(obj.name)}' holds text: select the numeric columns before converting to an array`)
    const anyFloat = this.data.some((s) => s.dtype === 'float64')
    const allBool = this.data.every((s) => s.dtype === 'bool')
    const vals: number[] = []
    for (let r = 0; r < this.nrows; r++) for (const s of this.data) vals.push(Number(s.values[r]))
    return NDArray.create(vals, [this.nrows, this.columns.length], allBool ? 'bool' : anyFloat ? 'float64' : 'int64')
  }
  /** a new frame with some rows (positions) and columns (positions) */
  take(rows: number[] | null, cols: number[] | null): Frame {
    const cs = cols ?? this.columns.map((_, k) => k)
    const rs = rows ?? this.index.map((_, i) => i)
    const f = new Frame(cs.map((k) => this.columns[k]), cs.map((k) => this.data[k].pick(rs)), rs.map((r) => this.index[r]), this.indexName)
    return f
  }
  colPos(h: Host, name: Value): number {
    const k = this.columns.indexOf(h.str(name))
    if (k < 0) throw h.err('KeyError', repr(name))
    return k
  }
  /** loc rows: a label, labels, an inclusive label slice, or a boolean mask */
  rowsByLabel(h: Host, sel: Value): number[] | number {
    if (sel.k === 'slice') {
      const at = (v: Value, d: number) => {
        if (v.k === 'none') return d
        const i = this.index.indexOf(cellOf(h, v))
        if (i < 0) throw h.err('KeyError', repr(v))
        return i
      }
      const a = at(sel.start, 0)
      const b = at(sel.stop, this.nrows - 1)
      return Array.from({ length: Math.max(0, b - a + 1) }, (_, k) => a + k)
    }
    if (sel.k === 'obj' && sel.o instanceof Series && sel.o.dtype === 'bool') return sel.o.values.flatMap((b, i) => (b ? [i] : []))
    if (sel.k === 'list' || sel.k === 'array' || (sel.k === 'obj' && sel.o instanceof Series)) {
      const items = cellsOf(h, sel)
      if (items.every((c) => typeof c === 'boolean')) return items.flatMap((b, i) => (b ? [i] : []))
      return items.map((c) => {
        const i = this.index.indexOf(c)
        if (i < 0) throw h.err('KeyError', `"None of [Index([${labelStr(c)}])] are in the [index]"`)
        return i
      })
    }
    const i = this.index.indexOf(cellOf(h, sel))
    if (i < 0) throw h.err('KeyError', repr(sel))
    return i
  }
  colsByLabel(h: Host, sel: Value): number[] | number {
    if (sel.k === 'slice') {
      const at = (v: Value, d: number) => (v.k === 'none' ? d : this.colPos(h, v))
      const a = at(sel.start, 0)
      const b = at(sel.stop, this.columns.length - 1)
      return Array.from({ length: Math.max(0, b - a + 1) }, (_, k) => a + k)
    }
    if (sel.k === 'list' || sel.k === 'tuple') return h.iterate(sel).map((v) => this.colPos(h, v))
    return this.colPos(h, sel)
  }
  getItem(idx: Value, h: Host): Value {
    const me = h.nameOf(py.obj(this), 'df')
    if (idx.k === 'str' || idx.k === 'int' || idx.k === 'float' || idx.k === 'bool') {
      const name = idx.k === 'str' ? idx.v : labelStr(cellOf(h, idx))
      const s = this.col(name)
      if (!s) throw h.err('KeyError', repr(idx))
      idx = py.str(name)
      traceFrame(h, 'column', [{ label: me, table: tableOfFrame(this) }], tableOfSeries(s), { rows: null, cols: [this.columns.indexOf(name)] })
      return py.obj(s)
    }
    if (idx.k === 'list') {
      const cols = h.iterate(idx).map((v) => this.colPos(h, v))
      const out = this.take(null, cols)
      traceFrame(h, 'columns', [{ label: me, table: tableOfFrame(this) }], tableOfFrame(out), { rows: null, cols })
      return py.obj(out)
    }
    if (idx.k === 'obj' && idx.o instanceof Series && idx.o.dtype === 'bool') {
      if (idx.o.length !== this.nrows) throw h.err('ValueError', `Item wrong length ${idx.o.length} instead of ${this.nrows}.`)
      const rows = idx.o.values.flatMap((b, i) => (b ? [i] : []))
      const out = this.take(rows, null)
      traceFrame(h, 'filter', [{ label: me, table: tableOfFrame(this) }], tableOfFrame(out), { rows, cols: null })
      return py.obj(out)
    }
    if (idx.k === 'slice') return py.obj(this.take(positions(h, idx, this.nrows), null))
    throw h.err('KeyError', repr(idx))
  }
  setItem(idx: Value, v: Value, h: Host) {
    this.assignColumn(h, h.str(idx), v, true)
  }
  assignColumn(h: Host, name: string, v: Value, trace: boolean) {
    let s: Series
    if (v.k === 'obj' && v.o instanceof Series) {
      const src = v.o
      const vals = this.index.map((l) => {
        const i = src.index.indexOf(l)
        return i < 0 ? NaN : src.values[i]
      })
      s = new Series(vals, this.index, name, inferDT(vals))
    } else if (v.k === 'list' || v.k === 'tuple' || v.k === 'array' || v.k === 'range') {
      const vals = cellsOf(h, v)
      if (vals.length !== this.nrows) throw h.err('ValueError', `Length of values (${vals.length}) does not match length of index (${this.nrows})`)
      s = new Series(vals, this.index, name)
    } else {
      const c = cellOf(h, v)
      s = new Series(this.index.map(() => c), this.index, name)
    }
    const k = this.columns.indexOf(name)
    if (k < 0) {
      this.columns.push(name)
      this.data.push(s)
    } else this.data[k] = s
    if (trace) traceFrame(h, 'assign', [], tableOfFrame(this), { rows: null, cols: [this.columns.indexOf(name)] })
  }
  /** reduce every column (axis=0) or every row (axis=1) */
  reduce(h: Host, how: string, kw: Kw): Value {
    const axis = kw.axis ? (kw.axis.k === 'str' ? (kw.axis.v === 'columns' ? 1 : 0) : h.toInt(kw.axis)) : 0
    const numericOnly = !!(kw.numeric_only && h.truthy(kw.numeric_only))
    const ddof = kw.ddof ? h.toInt(kw.ddof) : 1
    const cols = this.data.map((s, k) => ({ s, k })).filter(({ s }) => !numericOnly || s.dtype !== 'object')
    if (axis === 1) {
      const vals = this.index.map((_, r) => {
        const cells = cols.map(({ s }) => s.values[r])
        const v = reduceCells(h, cells, inferDT(cells), how, ddof, `row ${labelStr(this.index[r])}`)
        return cellOf(h, v)
      })
      return py.obj(new Series(vals, [...this.index], null, undefined, this.indexName))
    }
    const vals = cols.map(({ s }) => cellOf(h, reduceCells(h, s.values, s.dtype, how, ddof, `the column '${labelStr(s.name)}'`)))
    const floats = ['mean', 'std', 'var', 'median'].includes(how) || (how !== 'count' && cols.length > 0 && cols.every(({ s }) => s.dtype === 'float64'))
    return py.obj(new Series(vals, cols.map(({ s }) => s.name), null, how === 'count' ? 'int64' : floats && vals.every((v) => typeof v === 'number') ? 'float64' : inferDT(vals)))
  }
  binop(op: string, other: Value, reflected: boolean, h: Host): Value | undefined {
    const f = binFn(h, op)
    if (!f) return undefined
    const me = h.nameOf(py.obj(this), 'df')
    const apply = (x: Cell, y: Cell) => (reflected ? f(y, x) : f(x, y))
    let data: Series[]
    if (other.k === 'obj' && other.o instanceof Frame) {
      const o = other.o
      data = this.data.map((s, k) => {
        const os = o.col(this.columns[k])
        const vals = s.values.map((x, r) => (os ? apply(x, os.values[r]) : NaN))
        return new Series(vals, this.index, s.name, resultDT(op, s.dtype, os?.dtype ?? 'float64', vals))
      })
    } else if (other.k === 'obj' && other.o instanceof Series) {
      // a Series lines up with the columns (df - df.mean())
      const o = other.o
      data = this.data.map((s, k) => {
        const i = o.index.indexOf(this.columns[k])
        const y = i < 0 ? NaN : o.values[i]
        const vals = s.values.map((x) => apply(x, y))
        return new Series(vals, this.index, s.name, resultDT(op, s.dtype, o.dtype, vals))
      })
    } else {
      const y = cellOf(h, other)
      data = this.data.map((s) => {
        if (s.dtype === 'object' && typeof y !== 'string' && !['==', '!='].includes(op)) throw h.err('TypeError', `unsupported operand type(s) for ${op}: 'str' and '${typeof y === 'number' ? (Number.isInteger(y) ? 'int' : 'float') : typeof y}' (column '${labelStr(s.name)}')`)
        const vals = s.values.map((x) => apply(x, y))
        return new Series(vals, this.index, s.name, resultDT(op, s.dtype, typeof y === 'number' && !Number.isInteger(y) ? 'float64' : 'int64', vals))
      })
    }
    const out = new Frame([...this.columns], data, [...this.index], this.indexName)
    traceFrame(h, 'arith', [{ label: me, table: tableOfFrame(this) }, ...(other.k === 'obj' && other.o instanceof Series ? [{ label: h.nameOf(other, 'other'), table: tableOfSeries(other.o) }] : [])], tableOfFrame(out))
    return py.obj(out)
  }
  getAttr(name: string, h: Host): Value | undefined {
    const fn = (call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })
    const self = py.obj(this)
    const me = () => h.nameOf(self, 'df')
    const src = () => [{ label: me(), table: tableOfFrame(this) }]
    const col = this.col(name)
    switch (name) {
      case 'shape': return py.tuple([py.int(this.nrows), py.int(this.columns.length)])
      case 'columns': return py.obj(new IndexObj([...this.columns], this.columnsName))
      case 'index': return py.obj(new IndexObj([...this.index], this.indexName, isRangeIndex(this.index)))
      case 'dtypes': return py.obj(new Series(this.data.map((s) => s.dtype), [...this.columns], null, 'object'))
      case 'size': return py.int(this.nrows * this.columns.length)
      case 'ndim': return py.int(2)
      case 'empty': return py.bool(this.nrows === 0 || !this.columns.length)
      case 'T': return py.obj(transpose(this))
      case 'values': return py.arr(this.toArray())
      case 'to_numpy': return fn(() => py.arr(this.toArray()))
      case 'loc': case 'iloc': return py.obj(new FrameIndexer(this, name))
      case 'head': case 'tail': return fn((args) => {
        const n = args[0] ? h.toInt(args[0]) : 5
        const rows = this.index.map((_, i) => i)
        const pick = name === 'head' ? rows.slice(0, n) : rows.slice(Math.max(0, this.nrows - n))
        const out = this.take(pick, null)
        traceFrame(h, name, src(), tableOfFrame(out), { rows: pick, cols: null })
        return py.obj(out)
      })
      case 'info': return fn(() => {
        const lines = [`<class 'pandas.core.frame.DataFrame'>`, `RangeIndex: ${this.nrows} entries, 0 to ${this.nrows - 1}`, `Data columns (total ${this.columns.length} columns):`, ' #   Column  Non-Null Count  Dtype  ', '---  ------  --------------  -----  ']
        const cw = Math.max(6, ...this.columns.map((c) => c.length))
        const dw = Math.max(5, ...this.data.map((s) => s.dtype.length))
        lines[3] = ` #   ${'Column'.padEnd(cw)}  Non-Null Count  ${'Dtype'.padEnd(dw)}`
        lines[4] = `---  ${'------'.padEnd(cw)}  --------------  ${'-----'.padEnd(dw)}`
        this.data.forEach((s, k) => lines.push(` ${String(k).padEnd(3)} ${this.columns[k].padEnd(cw)}  ${`${s.values.filter((c) => !isMissing(c)).length} non-null`.padEnd(14)}  ${s.dtype.padEnd(dw)}`))
        const counts = new Map<string, number>()
        this.data.forEach((s) => counts.set(s.dtype, (counts.get(s.dtype) ?? 0) + 1))
        lines.push(`dtypes: ${[...counts].sort(([a], [b]) => (a < b ? -1 : 1)).map(([d, n]) => `${d}(${n})`).join(', ')}`)
        // pandas counts 8 bytes per value (a RangeIndex is 128); text columns are only their pointers, hence "+"
        const bytes = (isRangeIndex(this.index) ? 128 : 8 * this.nrows) + 8 * this.nrows * this.columns.length
        lines.push(`memory usage: ${bytes.toFixed(1)}${this.data.some((s) => s.dtype === 'object') ? '+' : ''} bytes`)
        h.print(lines.join('\n') + '\n')
        return py.NONE
      })
      case 'describe': return fn(() => {
        const num = this.data.filter((s) => s.dtype === 'int64' || s.dtype === 'float64')
        if (!num.length) throw h.err('ValueError', 'describe() here summarises numeric columns, and this DataFrame has none')
        const ds = num.map((s) => describeSeries(h, s))
        const out = new Frame(num.map((s) => labelStr(s.name)), ds, ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        traceFrame(h, 'describe', src(), tableOfFrame(out))
        return py.obj(out)
      })
      case 'mean': case 'sum': case 'min': case 'max': case 'std': case 'var': case 'median': case 'count': case 'prod':
        return fn((_a, kw) => {
          if ((name === 'std' || name === 'var') && !kw.ddof) noteOnce(h, 'pandas_ddof')
          return this.reduce(h, name, kw)
        })
      case 'nunique': return fn(() => py.obj(new Series(this.data.map((s) => new Set(s.values.filter((c) => !isMissing(c)).map(labelStr)).size), [...this.columns], null, 'int64')))
      case 'isna': case 'isnull': case 'notna': case 'notnull': return fn(() => {
        const out = new Frame([...this.columns], this.data.map((s) => s.with(s.values.map((c) => (name.startsWith('not') ? !isMissing(c) : isMissing(c))), 'bool')), [...this.index])
        traceFrame(h, 'isna', src(), tableOfFrame(out))
        return py.obj(out)
      })
      case 'fillna': return fn((args, kw) => {
        const v = kw.value ?? args[0]
        const fills = (k: number): Cell | undefined => (v.k === 'dict' ? (() => { const x = v.d.get(py.str(this.columns[k])); return x ? cellOf(h, x) : undefined })() : v.k === 'obj' && v.o instanceof Series ? (() => { const i = v.o.index.indexOf(this.columns[k]); return i < 0 ? undefined : v.o.values[i] })() : cellOf(h, v))
        const filled: number[] = []
        const data = this.data.map((s, k) => {
          const f = fills(k)
          if (f === undefined) return s
          const vals = s.values.map((c, r) => {
            if (!isMissing(c)) return c
            filled.push(r)
            return f
          })
          return s.with(vals, inferDT(vals))
        })
        const out = new Frame([...this.columns], data, [...this.index], this.indexName)
        traceFrame(h, 'fillna', src(), tableOfFrame(out), { rows: [...new Set(filled)], cols: null })
        return py.obj(out)
      })
      case 'dropna': return fn((_a, kw) => {
        const subset = kw.subset ? h.iterate(kw.subset).map((v) => this.colPos(h, v)) : this.columns.map((_, k) => k)
        const keep = this.index.map((_, r) => r).filter((r) => subset.every((k) => !isMissing(this.data[k].values[r])))
        const out = this.take(keep, null)
        traceFrame(h, 'dropna', src(), tableOfFrame(out), { rows: keep, cols: null })
        return py.obj(out)
      })
      case 'sort_values': return fn((args, kw) => {
        const byV = kw.by ?? args[0]
        const by = byV.k === 'list' || byV.k === 'tuple' ? h.iterate(byV).map((v) => this.colPos(h, v)) : [this.colPos(h, byV)]
        const ascV = kw.ascending
        const asc = !ascV ? by.map(() => true) : ascV.k === 'list' ? h.iterate(ascV).map((v) => h.truthy(v)) : by.map(() => h.truthy(ascV))
        const order = sortOrder(by.map((k) => this.data[k].values), asc)
        const out = this.take(order, null)
        traceFrame(h, 'sort_values', src(), tableOfFrame(out), { rows: null, cols: by })
        return py.obj(out)
      })
      case 'sort_index': return fn(() => py.obj(this.take(sortOrder([this.index], [true]), null)))
      case 'groupby': return fn((args, kw) => {
        const byV = kw.by ?? args[0]
        const by = byV.k === 'list' ? h.iterate(byV).map((v) => h.str(v)) : [h.str(byV)]
        for (const b of by) this.colPos(h, py.str(b))
        return py.obj(new GroupBy(this, by, null, me()))
      })
      case 'drop': return fn((args, kw) => {
        const target = kw.columns ?? (kw.axis && (h.str(kw.axis) === '1' || h.str(kw.axis) === 'columns') ? args[0] ?? kw.labels : null)
        if (target) {
          const names = target.k === 'list' ? h.iterate(target).map((v) => h.str(v)) : [h.str(target)]
          names.forEach((n) => this.colPos(h, py.str(n)))
          return py.obj(this.take(null, this.columns.map((_, k) => k).filter((k) => !names.includes(this.columns[k]))))
        }
        const rowsV = kw.index ?? args[0]
        const labels = rowsV.k === 'list' ? cellsOf(h, rowsV) : [cellOf(h, rowsV)]
        return py.obj(this.take(this.index.map((_, r) => r).filter((r) => !labels.includes(this.index[r])), null))
      })
      case 'rename': return fn((_a, kw) => {
        const m = kw.columns
        if (!m || m.k !== 'dict') throw h.err('TypeError', 'rename(columns={old: new}) is the form supported here')
        const cols = this.columns.map((c) => {
          const v = m.d.get(py.str(c))
          return v ? h.str(v) : c
        })
        return py.obj(new Frame(cols, this.data, [...this.index], this.indexName))
      })
      case 'assign': return fn((_a, kw) => {
        const out = new Frame([...this.columns], this.data.map((s) => s.with([...s.values], s.dtype)), [...this.index], this.indexName)
        for (const [k, v] of Object.entries(kw)) out.assignColumn(h, k, v.k === 'func' || v.k === 'fn' ? h.call(v, [py.obj(out)]) : v, false)
        traceFrame(h, 'assign', src(), tableOfFrame(out), { rows: null, cols: Object.keys(kw).map((k) => out.columns.indexOf(k)) })
        return py.obj(out)
      })
      case 'apply': return fn((args, kw) => {
        const axis = kw.axis ? (kw.axis.k === 'str' ? (kw.axis.v === 'columns' ? 1 : 0) : h.toInt(kw.axis)) : 0
        if (axis === 1) {
          const vals = this.index.map((_, r) => cellOf(h, h.call(args[0], [py.obj(new Series(this.data.map((s) => s.values[r]), [...this.columns], this.index[r]))])))
          return py.obj(new Series(vals, [...this.index], null))
        }
        const vals = this.data.map((s) => cellOf(h, h.call(args[0], [py.obj(s)])))
        return py.obj(new Series(vals, [...this.columns], null))
      })
      case 'astype': return fn((args) => {
        const t = args[0]
        const data = this.data.map((s, k) => {
          const want = t.k === 'dict' ? t.d.get(py.str(this.columns[k])) : t
          return want ? astypeSeries(h, s, want.k === 'type' ? want.name : h.str(want)) : s
        })
        return py.obj(new Frame([...this.columns], data, [...this.index], this.indexName))
      })
      case 'copy': return fn(() => py.obj(new Frame([...this.columns], this.data.map((s) => s.with([...s.values], s.dtype)), [...this.index], this.indexName)))
      case 'round': return fn((args) => {
        const d = args[0] ? h.toInt(args[0]) : 0
        return py.obj(new Frame([...this.columns], this.data.map((s) => (s.dtype === 'float64' ? s.with(s.values.map((c) => (typeof c === 'number' ? roundHalfEven(c, d) : c)), 'float64') : s)), [...this.index], this.indexName))
      })
      case 'reset_index': return fn((_a, kw) => {
        if (kw.drop && h.truthy(kw.drop)) return py.obj(new Frame([...this.columns], this.data, null))
        const idxCol = new Series([...this.index], null, this.indexName ?? 'index')
        return py.obj(new Frame([this.indexName ?? 'index', ...this.columns], [idxCol, ...this.data], null))
      })
      case 'set_index': return fn((args) => {
        const k = this.colPos(h, args[0])
        const rest = this.columns.map((_, j) => j).filter((j) => j !== k)
        return py.obj(new Frame(rest.map((j) => this.columns[j]), rest.map((j) => this.data[j]), [...this.data[k].values], this.columns[k]))
      })
      case 'merge': return fn((args, kw) => py.obj(merge(h, this, args[0], kw, me())))
      case 'corr': return fn(() => {
        const num = this.data.filter((s) => s.dtype !== 'object')
        const names = num.map((s) => labelStr(s.name))
        const corr = (a: number[], b: number[]) => {
          const ma = a.reduce((x, y) => x + y, 0) / a.length
          const mb = b.reduce((x, y) => x + y, 0) / b.length
          const cov = a.reduce((s, x, i) => s + (x - ma) * (b[i] - mb), 0)
          return cov / Math.sqrt(a.reduce((s, x) => s + (x - ma) ** 2, 0) * b.reduce((s, x) => s + (x - mb) ** 2, 0))
        }
        const vals = num.map((s) => s.numeric(h, 'corr'))
        return py.obj(new Frame(names, vals.map((a) => new Series(vals.map((b) => corr(a, b)), null, null, 'float64')), names))
      })
      case 'iterrows': return fn(() => py.list(this.index.map((l, r) => py.tuple([valueOf(l, inferDT(this.index)), py.obj(new Series(this.data.map((s) => s.values[r]), [...this.columns], l))]))))
      case 'items': return fn(() => py.list(this.data.map((s, k) => py.tuple([py.str(this.columns[k]), py.obj(s)]))))
      case 'value_counts': return undefined
      case 'to_csv': return fn((args) => {
        const lines = [[this.indexName ?? '', ...this.columns].join(','), ...this.index.map((l, r) => [labelStr(l), ...this.data.map((s) => (isMissing(s.values[r]) ? '' : labelStr(s.values[r])))].join(','))]
        const text = lines.join('\n') + '\n'
        if (!args[0] || args[0].k === 'none') return py.str(text)
        h.print(`(sandbox) would write ${h.str(args[0])}:\n${text}`)
        return py.NONE
      })
    }
    if (col) return py.obj(col)
    return undefined
  }
}

const transpose = (f: Frame): Frame => {
  const cols = f.index.map(labelStr)
  const data = f.index.map((_, r) => {
    const vals = f.data.map((s) => s.values[r])
    return new Series(vals, null, null)
  })
  return new Frame(cols, data, [...f.columns])
}

class FrameIndexer extends PyObj {
  readonly cls: string
  f: Frame
  kind: 'loc' | 'iloc'
  constructor(f: Frame, kind: string) {
    super()
    this.f = f
    this.kind = kind as 'loc' | 'iloc'
    this.cls = kind === 'loc' ? '_LocIndexer' : '_iLocIndexer'
  }
  repr() { return `<pandas.core.indexing.${this.cls} object>` }
  resolve(h: Host, idx: Value): { rows: number[] | number; cols: number[] | number } {
    const [r, c] = idx.k === 'tuple' ? [idx.items[0], idx.items[1]] : [idx, null]
    const all = { k: 'slice' as const, start: py.NONE, stop: py.NONE, step: py.NONE }
    if (this.kind === 'iloc') {
      const rows = r.k === 'int' ? positions(h, r, this.f.nrows)[0] : positions(h, r, this.f.nrows)
      const cols = !c ? positions(h, all, this.f.columns.length) : c.k === 'int' ? positions(h, c, this.f.columns.length)[0] : positions(h, c, this.f.columns.length)
      return { rows, cols }
    }
    return { rows: this.f.rowsByLabel(h, r), cols: c ? this.f.colsByLabel(h, c) : this.f.columns.map((_, k) => k) }
  }
  getItem(idx: Value, h: Host): Value {
    const { rows, cols } = this.resolve(h, idx)
    const f = this.f
    const label = h.nameOf(py.obj(f), 'df')
    const picks = { rows: typeof rows === 'number' ? [rows] : rows, cols: typeof cols === 'number' ? [cols] : cols }
    let out: Value
    if (typeof rows === 'number' && typeof cols === 'number') out = valueOf(f.data[cols].values[rows], f.data[cols].dtype)
    else if (typeof rows === 'number') {
      const cs = cols as number[]
      const vals = cs.map((k) => f.data[k].values[rows])
      out = py.obj(new Series(vals, cs.map((k) => f.columns[k]), f.index[rows]))
    } else if (typeof cols === 'number') out = py.obj(f.data[cols].pick(rows))
    else out = py.obj(f.take(rows, cols))
    const table = out.k === 'obj' && out.o instanceof Frame ? tableOfFrame(out.o) : out.k === 'obj' && out.o instanceof Series ? tableOfSeries(out.o) : null
    traceFrame(h, this.kind, [{ label, table: tableOfFrame(f) }], table, picks, null, repr(out))
    return out
  }
  setItem(idx: Value, v: Value, h: Host) {
    // assigning to a new column label with loc creates it (filled with NaN elsewhere)
    if (this.kind === 'loc' && idx.k === 'tuple' && idx.items[1].k === 'str' && !this.f.columns.includes(idx.items[1].v)) {
      this.f.columns.push(idx.items[1].v)
      this.f.data.push(new Series(this.f.index.map(() => NaN), this.f.index, idx.items[1].v, 'float64'))
    }
    const { rows, cols } = this.resolve(h, idx)
    const rs = typeof rows === 'number' ? [rows] : rows
    const cs = typeof cols === 'number' ? [cols] : cols
    const vals = v.k === 'list' || v.k === 'array' ? cellsOf(h, v) : null
    for (const k of cs) {
      const s = this.f.data[k]
      rs.forEach((r, i) => { s.values[r] = vals ? vals[i] : cellOf(h, v) })
      s.dtype = inferDT(s.values)
      s.values = normalise(s.values, s.dtype)
    }
  }
}

// ---------------------------------------------------------------- groupby

const AGGS = ['mean', 'sum', 'count', 'min', 'max', 'std', 'var', 'median', 'size', 'first', 'last', 'nunique']

class GroupBy extends PyObj {
  readonly cls: string
  f: Frame
  by: string[]
  select: string[] | null
  label: string
  constructor(f: Frame, by: string[], select: string[] | null, label: string) {
    super()
    this.f = f
    this.by = by
    this.select = select
    this.label = label
    this.cls = select && select.length === 1 ? 'SeriesGroupBy' : 'DataFrameGroupBy'
  }
  repr() { return `<pandas.core.groupby.generic.${this.cls} object>` }
  /** groups in sorted key order, and the group number of every row */
  groups(): { keys: Cell[][]; members: number[][]; ofRow: number[] } {
    const keyOf = (r: number) => this.by.map((b) => this.f.col(b)!.values[r])
    const map = new Map<string, { key: Cell[]; rows: number[] }>()
    this.f.index.forEach((_, r) => {
      const k = keyOf(r)
      if (k.some(isMissing)) return
      const s = JSON.stringify(k.map(labelStr))
      const e = map.get(s)
      if (e) e.rows.push(r)
      else map.set(s, { key: k, rows: [r] })
    })
    const entries = [...map.values()]
    const order = sortOrder(this.by.map((_, j) => entries.map((e) => e.key[j])), this.by.map(() => true))
    const sorted = order.map((i) => entries[i])
    const ofRow = this.f.index.map(() => -1)
    sorted.forEach((e, g) => e.rows.forEach((r) => (ofRow[r] = g)))
    return { keys: sorted.map((e) => e.key), members: sorted.map((e) => e.rows), ofRow }
  }
  valueCols(): string[] {
    return this.select ?? this.f.columns.filter((c) => !this.by.includes(c))
  }
  agg(h: Host, how: string | Record<string, string>): Value {
    const { keys, members, ofRow } = this.groups()
    const index = keys.map((k) => (k.length === 1 ? k[0] : k.map(labelStr).join(', ')))
    const indexName = this.by.length === 1 ? this.by[0] : this.by.join(', ')
    const cols = this.valueCols()
    const series = this.select && this.select.length === 1 && typeof how === 'string'
    let out: Frame | Series
    if (how === 'size') {
      out = new Series(members.map((m) => m.length), index, null, 'int64', indexName)
    } else {
      const data = cols.map((c) => {
        const s = this.f.col(c)!
        const fn = typeof how === 'string' ? how : how[c]
        if (!fn) return null
        if (!AGGS.includes(fn)) throw h.err('AttributeError', `'${this.cls}' object has no attribute '${fn}'`)
        const vals = members.map((rows) => {
          const cells = rows.map((r) => s.values[r])
          if (fn === 'first') return cells[0]
          if (fn === 'last') return cells[cells.length - 1]
          if (fn === 'nunique') return new Set(cells.map(labelStr)).size
          if (s.dtype === 'object' && !['count', 'min', 'max'].includes(fn)) throw h.err('TypeError', `agg function failed [how->${fn},dtype->object] (column '${c}' holds text: select the numeric columns, e.g. df.groupby('${this.by[0]}')[['…']].${fn}())`)
          return cellOf(h, reduceCells(h, cells, s.dtype, fn, 1, `column '${c}'`))
        })
        const dt: DT | undefined = fn === 'count' || fn === 'nunique' ? 'int64' : ['mean', 'std', 'var', 'median'].includes(fn) ? 'float64' : undefined
        return { c, s: new Series(vals, index, c, dt, indexName) }
      }).filter((x): x is { c: string; s: Series } => !!x)
      out = series ? data[0].s : new Frame(data.map((d) => d.c), data.map((d) => d.s), index, indexName)
    }
    traceFrame(h, 'groupby', [{ label: this.label, table: tableOfFrame(this.f) }], out instanceof Frame ? tableOfFrame(out) : tableOfSeries(out), { rows: null, cols: this.by.map((b) => this.f.columns.indexOf(b)) }, { keys: keys.map((k) => k.map(labelStr).join(', ')), ofRow })
    return py.obj(out)
  }
  getItem(idx: Value, h: Host): Value {
    const cols = idx.k === 'list' ? h.iterate(idx).map((v) => h.str(v)) : [h.str(idx)]
    for (const c of cols) this.f.colPos(h, py.str(c))
    const g = new GroupBy(this.f, this.by, cols, this.label)
    if (idx.k === 'list') (g as { cls: string }).cls = 'DataFrameGroupBy'
    return py.obj(g)
  }
  iter() {
    const { keys, members } = this.groups()
    return keys.map((k, g) => py.tuple([valueOf(k[0], inferDT([k[0]])), py.obj(this.f.take(members[g], null))]))
  }
  getAttr(name: string, h: Host): Value | undefined {
    if (AGGS.includes(name)) return { k: 'fn', name, call: (_a, kw) => {
      if (kw.numeric_only && h.truthy(kw.numeric_only)) {
        const num = this.valueCols().filter((c) => this.f.col(c)!.dtype !== 'object')
        return new GroupBy(this.f, this.by, num, this.label).agg(h, name)
      }
      return this.agg(h, name)
    } }
    if (name === 'agg' || name === 'aggregate') return { k: 'fn', name, call: (args) => {
      const a = args[0]
      if (a.k === 'str') return this.agg(h, a.v)
      if (a.k === 'dict') return this.agg(h, Object.fromEntries(a.d.items().map(([k, v]) => [h.str(k), h.str(v)])))
      throw h.err('NotImplementedError', "agg takes a name ('mean') or a dict ({'col': 'mean'}) in this sandbox")
    } }
    if (name === 'value_counts') return { k: 'fn', name, call: (_a, kw) => {
      const c = this.valueCols()[0]
      const { keys, members, ofRow } = this.groups()
      const norm = !!(kw.normalize && h.truthy(kw.normalize))
      const vals: Cell[] = []
      const index: Cell[] = []
      keys.forEach((k, g) => {
        const vc = valueCounts(this.f.col(c)!.pick(members[g]), norm)
        vc.values.forEach((v, i) => {
          vals.push(v)
          index.push(`${labelStr(k[0])}, ${labelStr(vc.index[i])}`)
        })
      })
      const out = new Series(vals, index, norm ? 'proportion' : 'count', norm ? 'float64' : 'int64', `${this.by[0]}, ${c}`)
      traceFrame(h, 'groupby', [{ label: this.label, table: tableOfFrame(this.f) }], tableOfSeries(out), null, { keys: keys.map((k) => labelStr(k[0])), ofRow })
      return py.obj(out)
    } }
    if (name === 'groups') return { k: 'dict', d: PyDict.from(this.groups().keys.map((k, g) => [valueOf(k[0], inferDT([k[0]])), py.list(this.groups().members[g].map((r) => valueOf(this.f.index[r], inferDT(this.f.index))))])) }
    return undefined
  }
}

// ---------------------------------------------------------------- module-level functions

const merge = (h: Host, left: Frame, rightV: Value, kw: Kw, label: string): Frame => {
  if (rightV.k !== 'obj' || !(rightV.o instanceof Frame)) throw h.err('TypeError', 'merge needs another DataFrame')
  const right = rightV.o
  const on = kw.on ? h.str(kw.on) : left.columns.find((c) => right.columns.includes(c))
  if (!on) throw h.err('MergeError', 'No common columns to perform merge on.')
  const how = kw.how ? h.str(kw.how) : 'inner'
  const lc = left.col(on)!
  const rc = right.col(on)
  if (!rc) throw h.err('KeyError', `'${on}'`)
  const rightCols = right.columns.filter((c) => c !== on)
  const rows: { l: number; r: number }[] = []
  lc.values.forEach((v, i) => {
    const matches = rc.values.flatMap((w, j) => (w === v ? [j] : []))
    if (matches.length) matches.forEach((j) => rows.push({ l: i, r: j }))
    else if (how === 'left' || how === 'outer') rows.push({ l: i, r: -1 })
  })
  if (how === 'right' || how === 'outer') rc.values.forEach((w, j) => { if (!lc.values.includes(w)) rows.push({ l: -1, r: j }) })
  const cols = [...left.columns, ...rightCols.map((c) => (left.columns.includes(c) ? `${c}_y` : c))]
  const data = [
    ...left.data.map((s) => {
      const vals = rows.map(({ l, r }) => (l >= 0 ? s.values[l] : s.name === on ? rc.values[r] : NaN))
      return new Series(vals, null, s.name, inferDT(vals))
    }),
    ...rightCols.map((c) => {
      const s = right.col(c)!
      const vals = rows.map(({ r }) => (r >= 0 ? s.values[r] : NaN))
      return new Series(vals, null, c, inferDT(vals))
    }),
  ]
  const out = new Frame(cols, data, null)
  traceFrame(h, 'merge', [{ label, table: tableOfFrame(left) }, { label: h.nameOf(rightV, 'right'), table: tableOfFrame(right) }], tableOfFrame(out))
  return out
}

const frameFrom = (h: Host, data: Value | undefined, kw: Kw): Frame => {
  const colsArg = kw.columns ? h.iterate(kw.columns).map((v) => h.str(v)) : null
  const indexArg = kw.index ? cellsOf(h, kw.index) : null
  if (!data || data.k === 'none') return new Frame(colsArg ?? [], (colsArg ?? []).map(() => new Series([], null)), indexArg ?? [])
  if (data.k === 'dict') {
    const entries = data.d.items()
    const cols = entries.map(([k]) => h.str(k))
    const series = entries.map(([, v]) => {
      if (v.k === 'obj' && v.o instanceof Series) return v.o
      if (isNum(v) || v.k === 'str' || v.k === 'none') return null
      return new Series(cellsOf(h, v), null)
    })
    const n = series.find((s) => s)?.length ?? (indexArg?.length ?? 1)
    const filled = series.map((s, k) => s ?? new Series(Array.from({ length: n }, () => cellOf(h, entries[k][1])), null))
    const lens = filled.map((s) => s.length)
    if (lens.some((l) => l !== lens[0])) throw h.err('ValueError', 'All arrays must be of the same length')
    const pick = colsArg ?? cols
    return new Frame(pick, pick.map((c) => filled[cols.indexOf(c)] ?? new Series(Array.from({ length: n }, () => NaN), null)), indexArg ?? filled[0]?.index ?? null)
  }
  if (data.k === 'obj' && data.o instanceof Frame) return data.o
  if (data.k === 'array' || data.k === 'list' || data.k === 'tuple') {
    const rows = h.iterate(data)
    if (rows.length && rows[0].k === 'dict') {
      const cols = [...new Set(rows.flatMap((r) => (r.k === 'dict' ? r.d.keys().map((k) => h.str(k)) : [])))]
      return new Frame(colsArg ?? cols, (colsArg ?? cols).map((c) => new Series(rows.map((r) => (r.k === 'dict' ? cellOf(h, r.d.get(py.str(c)) ?? py.NONE) : null)), null)), indexArg)
    }
    const grid = rows.map((r) => (r.k === 'list' || r.k === 'tuple' || r.k === 'array' ? cellsOf(h, r) : [cellOf(h, r)]))
    const width = grid[0]?.length ?? 0
    if (grid.some((r) => r.length !== width)) throw h.err('ValueError', 'rows of different lengths')
    const cols = colsArg ?? Array.from({ length: width }, (_, i) => String(i))
    if (cols.length !== width) throw h.err('ValueError', `Shape of passed values is (${grid.length}, ${width}), indices imply (${grid.length}, ${cols.length})`)
    return new Frame(cols, cols.map((_, k) => new Series(grid.map((r) => r[k]), null)), indexArg)
  }
  throw h.err('TypeError', `cannot build a DataFrame from ${data.k === 'obj' ? data.o.cls : data.k}`)
}

/** pd.read_csv(io.StringIO(text)): split on commas, infer numbers */
const readCsv = (h: Host, text: string, kw: Kw): Frame => {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length)
  if (!lines.length) throw h.err('EmptyDataError', 'No columns to parse from file')
  const sep = kw.sep ? h.str(kw.sep) : ','
  const head = lines[0].split(sep).map((s) => s.trim())
  const rows = lines.slice(1).map((l) => l.split(sep).map((s) => s.trim()))
  const parse = (s: string): Cell => (s === '' || s === 'NaN' || s === 'NA' ? null : s === 'True' ? true : s === 'False' ? false : s !== '' && !Number.isNaN(Number(s)) ? Number(s) : s)
  return new Frame(head, head.map((_, k) => new Series(rows.map((r) => parse(r[k] ?? '')), null)), null)
}

class StringIO extends PyObj {
  readonly cls = 'StringIO'
  text: string
  constructor(text: string) {
    super()
    this.text = text
  }
  repr() { return '<_io.StringIO object>' }
}

// ---------------------------------------------------------------- tables for the step view

export const tableOfFrame = (f: Frame): TableSpec => {
  const cols = f.data.map((s) => formatCells(s.values, s.dtype))
  return {
    columns: [...f.columns],
    index: f.index.map(labelStr),
    cells: f.index.map((_, r) => f.data.map((s, k) => (isMissing(s.values[r]) ? null : cols[k][r]))),
    dtypes: f.data.map((s) => s.dtype),
    series: false,
    name: null,
    indexName: f.indexName,
  }
}

export const tableOfSeries = (s: Series): TableSpec => {
  const all = formatCells(s.values, s.dtype)
  return {
    columns: [],
    index: s.index.map(labelStr),
    cells: s.values.map((c, i) => [isMissing(c) ? null : all[i]]),
    dtypes: [s.dtype],
    series: true,
    name: s.name === null ? '' : labelStr(s.name),
    indexName: s.indexName,
  }
}

const traceFrame = (h: Host, api: string, inputs: { label: string; table: TableSpec }[], result: TableSpec | null, picks: { rows: number[] | null; cols: number[] | null } | null = null, groups: { keys: string[]; ofRow: number[] } | null = null, resultText = '') => {
  if (!h.tracing) return
  if (inputs.some((i) => i.table.index.length > 200)) return
  h.emit({ type: 'frame', api, inputs, result, resultText, picks, groups })
}

// ---------------------------------------------------------------- the module

const build = (h: Host): Record<string, Value> => {
  const fn = (name: string, call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })
  const crosstab = fn('crosstab', (args, kw) => {
    const a = args[0] ?? kw.index
    const b = args[1] ?? kw.columns
    if (!(a.k === 'obj' && a.o instanceof Series) || !(b.k === 'obj' && b.o instanceof Series)) throw h.err('TypeError', 'crosstab(index, columns) takes two Series here')
    const ra = a.o
    const rb = b.o
    const rk = [...new Set(ra.values.map(labelStr))].map((x) => ra.values.find((c) => labelStr(c) === x)!)
    const ck = [...new Set(rb.values.map(labelStr))].map((x) => rb.values.find((c) => labelStr(c) === x)!)
    const rows = sortOrder([rk], [true]).map((i) => rk[i])
    const cols = sortOrder([ck], [true]).map((i) => ck[i])
    const counts = rows.map((r) => cols.map((c) => ra.values.filter((v, i) => v === r && rb.values[i] === c).length))
    const norm = kw.normalize ? (kw.normalize.k === 'str' ? kw.normalize.v : h.truthy(kw.normalize) ? 'all' : '') : ''
    const total = counts.flat().reduce((x, y) => x + y, 0)
    const val = (i: number, j: number) => {
      const c = counts[i][j]
      if (norm === 'index') return c / counts[i].reduce((x, y) => x + y, 0)
      if (norm === 'columns') return c / counts.reduce((s, row) => s + row[j], 0)
      if (norm === 'all') return c / total
      return c
    }
    const out = new Frame(cols.map(labelStr), cols.map((_, j) => new Series(rows.map((_, i) => val(i, j)), null, null, norm ? 'float64' : 'int64')), rows, ra.name === null ? null : labelStr(ra.name))
    out.columnsName = rb.name === null ? null : labelStr(rb.name)
    traceFrame(h, 'crosstab', [{ label: h.nameOf(a, 'index'), table: tableOfSeries(ra) }, { label: h.nameOf(b, 'columns'), table: tableOfSeries(rb) }], tableOfFrame(out))
    return py.obj(out)
  })
  const pd: Record<string, Value> = {
    __version__: py.str('2.3 (sandbox)'),
    DataFrame: { k: 'type', name: 'DataFrame', call: (args, kw) => py.obj(frameFrom(h, args[0] ?? kw.data, kw)) },
    Series: { k: 'type', name: 'Series', call: (args, kw) => {
      const d = args[0] ?? kw.data
      if (d && d.k === 'dict') return py.obj(new Series(d.d.values().map((v) => cellOf(h, v)), d.d.keys().map((k) => cellOf(h, k)), kw.name ? cellOf(h, kw.name) : null))
      const vals = d ? cellsOf(h, d) : []
      return py.obj(new Series(vals, kw.index ? cellsOf(h, kw.index) : null, kw.name ? cellOf(h, kw.name) : null))
    } },
    read_csv: fn('read_csv', (args, kw) => {
      const src = args[0] ?? kw.filepath_or_buffer
      if (src && src.k === 'obj' && src.o instanceof StringIO) return py.obj(readCsv(h, src.o.text, kw))
      throw h.err('FileNotFoundError', `[Errno 2] No such file or directory: '${src ? h.str(src) : ''}' (there are no files in the sandbox: wrap the CSV text in io.StringIO(...) instead)`)
    }),
    concat: fn('concat', (args, kw) => {
      const frames = h.iterate(args[0]).map((v) => (v.k === 'obj' && v.o instanceof Frame ? v.o : null))
      if (frames.some((f) => !f)) throw h.err('TypeError', 'concat takes a list of DataFrames here')
      const fs = frames as Frame[]
      const cols = [...new Set(fs.flatMap((f) => f.columns))]
      const ignore = !!(kw.ignore_index && h.truthy(kw.ignore_index))
      const data = cols.map((c) => {
        const vals = fs.flatMap((f) => (f.col(c) ? f.col(c)!.values : f.index.map(() => NaN)))
        return new Series(vals, null, c, inferDT(vals))
      })
      return py.obj(new Frame(cols, data, ignore ? null : fs.flatMap((f) => f.index)))
    }),
    get_dummies: fn('get_dummies', (args, kw) => {
      const d = args[0] ?? kw.data
      const asInt = kw.dtype ? h.str(kw.dtype.k === 'type' ? py.str(kw.dtype.name) : kw.dtype).startsWith('int') : false
      const dummies = (s: Series, prefix: string | null) => {
        const uniq = [...new Set(s.values.filter((c) => !isMissing(c)).map(labelStr))]
        const cats = sortOrder([uniq], [true]).map((i) => uniq[i])
        return cats.map((c) => ({ name: prefix ? `${prefix}_${c}` : c, s: new Series(s.values.map((v) => (asInt ? Number(labelStr(v) === c) : labelStr(v) === c)), [...s.index], null, asInt ? 'int64' : 'bool') }))
      }
      let out: Frame
      if (d.k === 'obj' && d.o instanceof Series) {
        const parts = dummies(d.o, null)
        out = new Frame(parts.map((p) => p.name), parts.map((p) => p.s), [...d.o.index])
      } else if (d.k === 'obj' && d.o instanceof Frame) {
        const f = d.o
        const which = kw.columns ? h.iterate(kw.columns).map((v) => h.str(v)) : f.columns.filter((c) => f.col(c)!.dtype === 'object')
        const cols: string[] = []
        const data: Series[] = []
        f.columns.forEach((c, k) => {
          if (!which.includes(c)) {
            cols.push(c)
            data.push(f.data[k])
          }
        })
        for (const c of which) for (const p of dummies(f.col(c)!, c)) {
          cols.push(p.name)
          data.push(p.s)
        }
        out = new Frame(cols, data, [...f.index])
      } else throw h.err('TypeError', 'get_dummies takes a Series or a DataFrame')
      traceFrame(h, 'get_dummies', [{ label: h.nameOf(d, 'data'), table: d.k === 'obj' && d.o instanceof Series ? tableOfSeries(d.o) : tableOfFrame(d.o as Frame) }], tableOfFrame(out))
      return py.obj(out)
    }),
    crosstab,
    merge: fn('merge', (args, kw) => {
      const l = args[0]
      if (!(l.k === 'obj' && l.o instanceof Frame)) throw h.err('TypeError', 'merge(left, right, on=...)')
      return py.obj(merge(h, l.o, args[1], kw, h.nameOf(l, 'left')))
    }),
    isna: fn('isna', (args) => {
      const v = args[0]
      if (v.k === 'obj' && (v.o instanceof Series || v.o instanceof Frame)) return h.call(h.getAttr(v, 'isna'), [])
      return py.bool(isMissing(cellOf(h, v)))
    }),
    notna: fn('notna', (args) => {
      const v = args[0]
      if (v.k === 'obj' && (v.o instanceof Series || v.o instanceof Frame)) return h.call(h.getAttr(v, 'notna'), [])
      return py.bool(!isMissing(cellOf(h, v)))
    }),
    NA: py.float(NaN),
  }
  const io: Value = { k: 'module', name: 'io', attrs: { StringIO: { k: 'type', name: 'StringIO', call: (args) => py.obj(new StringIO(args[0] ? h.str(args[0]) : '')) } } }
  return { pandas: { k: 'module', name: 'pandas', attrs: pd }, io }
}

export const PANDAS: PyLib = { modules: ['pandas', 'io'], load: build }

