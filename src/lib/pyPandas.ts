/**
 * pandas for the sandbox: the data model (Index, Series, DataFrame), pandas' printing, selection (df[…], loc, iloc, at,
 * iat, boolean masks), aligned arithmetic and the `pandas` module itself. The methods live in pyPandasMethods.ts,
 * groupby and windows in pyPandasGroup.ts, reshaping in pyPandasReshape.ts, text in pyPandasStr.ts, files in
 * pyPandasIO.ts. Each table operation is recorded with the rows / columns it picked or the groups it formed.
 */
import { NDArray, PyError } from './ndarray'
import { PyDict, PyObj, isNum, py, repr } from './minipy'
import type { Host, Kw, PyLib, Value } from './minipy'
import type { TableSpec } from './pyEvents'
import { DEFAULT_OPTIONS, OPTIONS, PyCell, TERMINAL_WIDTH, adjoin, fixedWidth, formatArray, formatLabels, isMissing, joinMultiline, justify, objectSummary, pprint, sparsify } from './pyPandasFormat'
import type { Cell, DT } from './pyPandasFormat'
import { categoricalFooter, frameAttr, frameNp, indexAttr, seriesAttr, seriesNp } from './pyPandasMethods'
import { pandasModule } from './pyPandasModule'

export { PyCell, isMissing }
export type { Cell, DT }

// ---------------------------------------------------------------- dtype and conversion

/** pandas' dtype for a column of values whose Python types are gone */
export const inferDT = (cells: Cell[]): DT => {
  const present = cells.filter((c) => !isMissing(c))
  const missing = present.length < cells.length
  if (!present.length) return cells.some((c) => c === null) || !cells.length ? 'object' : 'float64'
  if (present.every((c) => typeof c === 'boolean')) return missing ? 'object' : 'bool'
  if (present.every((c) => typeof c === 'number')) return !missing && present.every((c) => Number.isInteger(c)) ? 'int64' : 'float64'
  return 'object'
}

/** numeric columns store missing values as NaN */
export const normalise = (cells: Cell[], dt: DT): Cell[] => (dt === 'float64' ? cells.map(numCell) : cells)

/** a cell as a number: booleans count, a boxed float unboxes, None is NaN */
export const numCell = (c: Cell): number => (typeof c === 'number' ? c : c === null ? NaN : typeof c === 'boolean' ? Number(c) : c instanceof PyCell && (c.v.k === 'float' || c.v.k === 'int') ? c.v.v : NaN)

export const cellOf = (h: Host, v: Value): Cell => {
  switch (v.k) {
    case 'int': case 'float': return v.v
    case 'bool': return v.v
    case 'str': return v.v
    case 'none': return null
    case 'array':
      if (v.a.ndim === 0 || v.a.size === 1) return v.a.dtype === 'bool' ? v.a.values()[0] !== 0 : v.a.values()[0]
      return new PyCell(v)
    case 'obj':
      if (v.o.toArray && !(v.o instanceof Series) && !(v.o instanceof Frame) && !(v.o instanceof IndexObj)) {
        const a = v.o.toArray()
        if (a.size === 1) return a.values()[0]
      }
      return new PyCell(v)
  }
  void h
  return new PyCell(v)
}

export const valueOf = (c: Cell, dt: DT): Value => {
  if (c === null) return py.NONE
  if (typeof c === 'boolean') return py.bool(c)
  if (typeof c === 'string') return py.str(c)
  if (c instanceof PyCell) return c.v
  if (dt === 'float64') return py.float(c)
  return Number.isInteger(c) ? py.int(c) : py.float(c)
}

/** a float beside text stays a float when printed (row['score'] is 70.0, not 70) */
export const boxFloat = (v: number): Cell => (Number.isInteger(v) ? new PyCell(py.float(v)) : v)

/** the cells of a typed column as they sit in an object column */
export const toObjectCells = (cells: Cell[], dt: DT): Cell[] => (dt === 'float64' ? cells.map((c) => boxFloat(c as number)) : [...cells])

/** the dtype and cells pandas infers from Python values (constructors, map, apply) */
export const inferValues = (h: Host, items: Value[]): { cells: Cell[]; dt: DT } => {
  let ints = 0
  let floats = 0
  let bools = 0
  let nones = 0
  let nans = 0
  let other = 0
  const norm = items.map((v0) => {
    const c = v0.k === 'array' && (v0.a.ndim === 0 || v0.a.size === 1) ? cellOf(h, v0) : null
    if (c !== null && v0.k === 'array') return typeof c === 'boolean' ? py.bool(c) : v0.a.dtype === 'float64' ? py.float(c as number) : py.int(c as number)
    return v0
  })
  for (const v of norm) {
    switch (v.k) {
      case 'int': ints++; break
      case 'float': if (Number.isNaN(v.v)) nans++; else floats++; break
      case 'bool': bools++; break
      case 'none': nones++; break
      default: other++
    }
  }
  let dt: DT
  if (!items.length || other) dt = 'object'
  else if (bools) dt = ints + floats + nones + nans ? 'object' : 'bool'
  else if (ints + floats) dt = floats || nones || nans ? 'float64' : 'int64'
  else if (nans) dt = 'float64'
  else dt = 'object'
  const cells = norm.map((v) => (dt === 'object' && v.k === 'float' ? boxFloat(v.v) : cellOf(h, v)))
  return { cells: normalise(cells, dt), dt }
}

const arrayDT = (a: NDArray): DT => (a.dtype === 'bool' ? 'bool' : a.dtype === 'float64' ? 'float64' : 'int64')
/** the sized dtype name of a uint8 array; everything else the sandbox stores is 64-bit */
export const arrayWidth = (a: NDArray): string | null => (a.dtype === 'uint8' ? 'uint8' : null)

/** a list / array / Series / Index as cells with their dtype */
export const typedCells = (h: Host, v: Value): { cells: Cell[]; dt: DT; width?: string | null } => {
  if (v.k === 'obj') {
    if (v.o instanceof Series) return { cells: [...v.o.values], dt: v.o.dtype, width: v.o.width }
    if (v.o instanceof IndexObj) return { cells: [...v.o.labels], dt: v.o.dt }
    if (v.o instanceof ObjArray) {
      if (v.o.shape.length !== 1) throw h.err('ValueError', 'Data must be 1-dimensional')
      return { cells: [...v.o.cells], dt: 'object' }
    }
    if (v.o.toArray) {
      const a = v.o.toArray()
      return { cells: a.dtype === 'bool' ? a.values().map((x) => x !== 0) : a.values(), dt: arrayDT(a), width: arrayWidth(a) }
    }
  }
  if (v.k === 'array') {
    if (v.a.ndim > 1) throw h.err('ValueError', 'Data must be 1-dimensional')
    return { cells: v.a.dtype === 'bool' ? v.a.values().map((x) => x !== 0) : v.a.values(), dt: arrayDT(v.a), width: arrayWidth(v.a) }
  }
  if (v.k === 'range') return { cells: h.iterate(v).map((x) => (x as { v: number }).v), dt: 'int64' }
  if (v.k === 'str' || isNum(v) || v.k === 'none') throw h.err('TypeError', `expected a list, array or Series, got ${v.k}`)
  if (v.k === 'dict') return inferValues(h, v.d.keys())
  return inferValues(h, h.iterate(v))
}

export const seriesIn = (v: Value) => (v as { o: unknown }).o as Series
export const frameIn = (v: Value) => (v as { o: unknown }).o as Frame
export const objArrayIn = (v: Value) => (v as { o: unknown }).o as ObjArray

export const cellsOf = (h: Host, v: Value): Cell[] => typedCells(h, v).cells

/** a list-like selector or argument (not a string, not a scalar) */
export const isListLike = (v: Value) => v.k === 'list' || v.k === 'tuple' || v.k === 'array' || v.k === 'range' || v.k === 'set' || v.k === 'iter' || (v.k === 'obj' && (v.o instanceof Series || v.o instanceof IndexObj || v.o instanceof ObjArray))

export const labelStr = (c: Cell): string => (c === null ? 'None' : typeof c === 'boolean' ? (c ? 'True' : 'False') : typeof c === 'number' ? (Number.isNaN(c) ? 'NaN' : String(c)) : c instanceof PyCell ? pprint(c.v) : c)

/** the widest dtype that holds them all, as a DataFrame row or a concat needs */
export const commonDT = (dts: DT[]): DT => {
  if (!dts.length) return 'object'
  if (dts.includes('object')) return 'object'
  if (dts.every((d) => d === 'bool')) return 'bool'
  if (dts.includes('bool')) return 'object'
  return dts.includes('float64') ? 'float64' : 'int64'
}

/** the dtype after storing one more value in a column */
const widenDT = (dt: DT, c: Cell): DT => {
  if (dt === 'object') return 'object'
  if (isMissing(c)) return dt === 'bool' ? 'object' : 'float64'
  if (typeof c === 'number') return dt === 'bool' ? 'object' : dt === 'int64' && !Number.isInteger(c) ? 'float64' : dt
  if (typeof c === 'boolean') return dt === 'bool' ? 'bool' : 'object'
  return 'object'
}

/** convert cells between dtypes without losing what they print as */
export const convertCells = (cells: Cell[], from: DT, to: DT): Cell[] => {
  if (from === to) return cells
  if (to === 'object') return toObjectCells(cells, from)
  if (to === 'float64') return cells.map(numCell)
  return cells
}

// ---------------------------------------------------------------- Index

export interface Multi { arrays: Cell[][]; names: Cell[]; dts: DT[] }

const keyOf = (c: Cell): unknown => (c instanceof PyCell ? `\0${repr(c.v)}` : c === null ? NaN : c)

const reprLabel = (c: Cell): string => (typeof c === 'string' ? `'${c}'` : typeof c === 'number' ? (Number.isNaN(c) ? 'nan' : labelStr(c)) : c instanceof PyCell ? repr(c.v) : labelStr(c))

const floatRepr = (c: Cell): string => (typeof c === 'number' ? repr(py.float(c)) : reprLabel(c))

export class IndexObj extends PyObj {
  labels: Cell[]
  name: Cell
  dt: DT
  isRange: boolean
  multi: Multi | null
  private map: Map<unknown, number[]> | null = null
  constructor(labels: Cell[], name: Cell = null, dt?: DT, isRange = false, multi: Multi | null = null) {
    super()
    this.dt = dt ?? inferDT(labels)
    this.labels = normalise(labels, this.dt)
    this.name = name
    this.isRange = isRange
    this.multi = multi
  }
  static range(n: number, start = 0, step = 1, name: Cell = null) {
    return new IndexObj(Array.from({ length: n }, (_, i) => start + i * step), name, 'int64', true)
  }
  static fromArrays(arrays: Cell[][], names: Cell[], dts: DT[]) {
    const n = arrays[0]?.length ?? 0
    const labels = Array.from({ length: n }, (_, r) => new PyCell(py.tuple(arrays.map((a, j) => valueOf(a[r], dts[j])))))
    return new IndexObj(labels, null, 'object', false, { arrays, names, dts })
  }
  get cls() { return this.multi ? 'MultiIndex' : this.isRange ? 'RangeIndex' : 'Index' }
  get length() { return this.labels.length }
  isa(name: string) { return name === 'Index' || name === this.cls }
  private lookup() {
    if (!this.map) {
      this.map = new Map()
      this.labels.forEach((l, i) => {
        const k = keyOf(l)
        const e = this.map!.get(k)
        if (e) e.push(i)
        else this.map!.set(k, [i])
      })
    }
    return this.map
  }
  /** every position holding this label */
  locate(label: Cell): number[] { return this.lookup().get(keyOf(label)) ?? [] }
  has(label: Cell) { return this.lookup().has(keyOf(label)) }
  get isUnique() { return this.lookup().size === this.labels.length }
  /** the labels at these positions; a RangeIndex survives a slice */
  pick(rows: number[], keepRange = false): IndexObj {
    if (this.multi) {
      const m = this.multi
      return IndexObj.fromArrays(m.arrays.map((a) => rows.map((r) => a[r])), m.names, m.dts)
    }
    const labels = rows.map((r) => this.labels[r])
    let range = false
    if (keepRange && this.isRange) {
      const step = labels.length > 1 ? (labels[1] as number) - (labels[0] as number) : 1
      range = labels.every((l, i) => l === (labels[0] as number) + i * step)
    }
    return new IndexObj(labels, this.name, this.dt, range)
  }
  withName(name: Cell): IndexObj {
    if (this.multi) return IndexObj.fromArrays(this.multi.arrays, this.multi.names, this.multi.dts)
    const out = new IndexObj(this.labels, name, this.dt, this.isRange)
    out.map = this.map
    return out
  }
  equals(o: IndexObj) {
    if (o === this || o.labels === this.labels) return true
    if (o.length !== this.length) return false
    return this.labels.every((l, i) => keyOf(l) === keyOf(o.labels[i]) || (isMissing(l) && isMissing(o.labels[i])))
  }
  /** one level of a MultiIndex, or the index itself */
  level(i: number): IndexObj {
    if (!this.multi) return this
    return new IndexObj(this.multi.arrays[i], this.multi.names[i], this.multi.dts[i])
  }
  get nlevels() { return this.multi ? this.multi.arrays.length : 1 }
  get names(): Cell[] { return this.multi ? this.multi.names : [this.name] }
  /** the strings printed for each level: one array for a flat index */
  formatted(): string[][] {
    if (this.multi) {
      const m = this.multi
      return m.arrays.map((a, j) => formatLabels(a, m.dts[j]))
    }
    return [formatLabels(this.labels, this.dt)]
  }
  repr() {
    if (this.isRange) {
      const start = (this.labels[0] as number | undefined) ?? 0
      const step = this.labels.length > 1 ? (this.labels[1] as number) - start : 1
      return `RangeIndex(start=${start}, stop=${start + this.labels.length * step}, step=${step}${this.name !== null ? `, name=${reprLabel(this.name)}` : ''})`
    }
    const max = OPTIONS.max_seq_items
    const n = this.labels.length
    const truncated = n > max
    const name = this.cls
    let fmt: string[]
    if (this.multi) {
      const m = this.multi
      const rows = Array.from({ length: n }, (_, r) => m.arrays.map((a, j) => (m.dts[j] === 'float64' ? floatRepr(a[r]) : reprLabel(a[r]))))
      const widths = m.arrays.map((_, j) => Math.max(...rows.map((row) => row[j].length)))
      fmt = rows.map((row) => `(${row.map((x, j) => x.padStart(widths[j])).join(', ')}${row.length === 1 ? ',' : ''})`)
    } else fmt = this.labels.map((l) => (this.dt === 'float64' ? floatRepr(l) : reprLabel(l)))
    const head = truncated ? fmt.slice(0, Math.min(Math.floor(max / 2), 10)) : []
    const tail = truncated ? fmt.slice(-Math.min(Math.floor(max / 2), 10)) : fmt
    const isJustify = !this.labels.every((l) => typeof l === 'string' || isMissing(l))
    const body = objectSummary(head, tail, truncated, name, isJustify, !!this.multi)
    const attrs: string[] = []
    if (!this.multi) attrs.push(`dtype='${this.dt}'`)
    if (this.multi ? this.multi.names.some((x) => x !== null) : this.name !== null) attrs.push(this.multi ? `names=[${this.multi.names.map(reprLabel).join(', ')}]` : `name=${reprLabel(this.name)}`)
    if (truncated) attrs.push(`length=${n}`)
    return `${name}(${body}${attrs.join(', ')})`
  }
  len() { return this.labels.length }
  iter() { return this.labels.map((l) => valueOf(l, this.dt)) }
  contains(v: Value, h: Host) { return this.has(cellOf(h, v)) }
  toArray(): NDArray {
    if (this.dt === 'object') throw new PyError('TypeError', 'this Index holds text, so it cannot become a numeric array')
    return NDArray.create(this.labels.map(numCell), [this.labels.length], this.dt === 'bool' ? 'bool' : this.dt === 'int64' ? 'int64' : 'float64')
  }
  asSeries(): Series { return new Series([...this.labels], IndexObj.range(this.labels.length), this.name, this.dt) }
  getItem(idx: Value, h: Host): Value {
    if (idx.k === 'int' || (idx.k === 'array' && idx.a.ndim === 0)) {
      const i = h.toInt(idx)
      const n = this.labels.length
      if (i < -n || i >= n) throw h.err('IndexError', `index ${i} is out of bounds for axis 0 with size ${n}`)
      return valueOf(this.labels[i < 0 ? i + n : i], this.dt)
    }
    return py.obj(this.pick(positions(h, idx, this.labels.length), idx.k === 'slice'))
  }
  binop(op: string, other: Value, reflected: boolean, h: Host): Value | undefined {
    if (other.k === 'obj' && (other.o instanceof Series || other.o instanceof Frame)) return undefined
    const y = other.k === 'obj' && other.o instanceof IndexObj ? py.obj(other.o.asSeries()) : other
    const res = this.asSeries().binop(op, y, reflected, h)
    if (!res || res.k !== 'obj' || !(res.o instanceof Series)) return res
    const s = res.o
    if (CMP.includes(op)) return py.arr(s.toArray())
    return py.obj(new IndexObj(s.values, this.name, s.dtype))
  }
  getAttr(name: string, h: Host): Value | undefined { return indexAttr(this, name, h) }
  setAttr(name: string, v: Value, h: Host): boolean {
    if (name === 'name') {
      this.name = cellOf(h, v)
      return true
    }
    if (name === 'names') {
      const ns = cellsOf(h, v)
      if (this.multi) this.multi = { ...this.multi, names: ns }
      else this.name = ns[0]
      return true
    }
    return false
  }
}

/** the categories of a binned column */
export class IntervalIndexObj extends IndexObj {
  readonly kind: string
  constructor(labels: Cell[], kind: string) {
    super(labels, null, 'object')
    this.kind = kind
  }
  repr() { return `IntervalIndex([${this.labels.map(labelStr).join(', ')}], dtype='${this.kind}')` }
}

const toIndex = (index: IndexObj | Cell[] | null, n: number): IndexObj => (index instanceof IndexObj ? index : index ? new IndexObj(index) : IndexObj.range(n))

// ---------------------------------------------------------------- text arrays

/** a NumPy array of dtype object: text the numeric NDArray cannot hold (s.values, s.unique(), df.columns.values) */
export class ObjArray extends PyObj {
  readonly cls = 'ndarray'
  cells: Cell[]
  shape: number[]
  /** fixed-width text (dtype='<U3') instead of dtype=object */
  text: boolean
  constructor(cells: Cell[], shape?: number[], text = false) {
    super()
    this.cells = cells
    this.shape = shape ?? [cells.length]
    this.text = text
  }
  private dtypeName() { return `<U${Math.max(0, ...this.cells.map((c) => String(c).length))}` }
  private elem(c: Cell) { return c === null ? 'None' : typeof c === 'number' && Number.isNaN(c) ? 'nan' : typeof c === 'string' ? repr(py.str(c)) : c instanceof PyCell ? repr(c.v) : typeof c === 'number' ? (Number.isInteger(c) ? String(c) : repr(py.float(c))) : labelStr(c) }
  private lines(indent: string, sep: string): string {
    const wrap = (items: string[], pad: string, first: number) => {
      const out: string[] = []
      let line = ''
      let room = first
      items.forEach((x, i) => {
        const word = x + (i < items.length - 1 ? sep.trimEnd() : '')
        if (line && line.length + 1 + word.length + room > 75) {
          out.push(line)
          line = ''
          room = pad.length
        }
        line += (line ? ' ' : '') + word
      })
      out.push(line)
      return out.join(`\n${pad}`)
    }
    if (this.shape.length === 1) return `[${wrap(this.cells.map((c) => this.elem(c)), `${indent} `, indent.length + 1)}]`
    const [r, c] = this.shape
    const rows = Array.from({ length: r }, (_, i) => `[${wrap(this.cells.slice(i * c, (i + 1) * c).map((x) => this.elem(x)), `${indent}  `, indent.length + 2)}]`)
    return `[${rows.join(`${sep.trimEnd()}\n${indent} `)}]`
  }
  repr() {
    const body = `array(${this.lines('      ', ', ')}`
    const dtype = `dtype=${this.text ? `'${this.dtypeName()}'` : 'object'})`
    const last = body.length - (body.lastIndexOf('\n') + 1)
    return `${body},${last + dtype.length + 1 > 75 ? '\n      ' : ' '}${dtype}`
  }
  str() { return this.lines('', ' ') }
  len() { return this.shape[0] }
  iter() {
    if (this.shape.length === 1) return this.cells.map((c) => valueOf(c, 'object'))
    const [r, c] = this.shape
    return Array.from({ length: r }, (_, i) => py.obj(new ObjArray(this.cells.slice(i * c, (i + 1) * c))))
  }
  contains(v: Value, h: Host) {
    const c = cellOf(h, v)
    return this.cells.some((x) => x === c)
  }
  getItem(idx: Value, h: Host): Value {
    if (this.shape.length === 1) {
      if (idx.k === 'int') {
        const i = positions(h, idx, this.cells.length, `index ${idx.v} is out of bounds for axis 0 with size ${this.cells.length}`)[0]
        return valueOf(this.cells[i], 'object')
      }
      return py.obj(new ObjArray(positions(h, idx, this.cells.length).map((i) => this.cells[i])))
    }
    const [r, c] = this.shape
    const [ri, ci] = idx.k === 'tuple' ? [idx.items[0], idx.items[1] ?? null] : [idx, null]
    const rows = positions(h, ri, r)
    const cols = ci ? positions(h, ci, c) : Array.from({ length: c }, (_, j) => j)
    const cells = rows.flatMap((i) => cols.map((j) => this.cells[i * c + j]))
    if (ri.k === 'int' && ci?.k === 'int') return valueOf(cells[0], 'object')
    if (ri.k === 'int' || ci?.k === 'int') return py.obj(new ObjArray(cells))
    return py.obj(new ObjArray(cells, [rows.length, cols.length]))
  }
  binop(op: string, other: Value, reflected: boolean, h: Host): Value | undefined {
    if (this.shape.length !== 1 || (op !== '==' && op !== '!=')) return undefined
    const res = new Series([...this.cells], null, null, 'object').binop(op, other, reflected, h)
    return res && py.arr(seriesIn(res).toArray())
  }
  getAttr(name: string): Value | undefined {
    const fn = (call: () => Value): Value => ({ k: 'fn', name, call })
    switch (name) {
      case 'shape': return py.tuple(this.shape.map(py.int))
      case 'ndim': return py.int(this.shape.length)
      case 'size': return py.int(this.cells.length)
      case 'dtype': return py.obj(new DtypeObj(this.text ? this.dtypeName() : 'object'))
      case 'T': return py.obj(this)
      case 'tolist': return fn(() => (this.shape.length === 1 ? py.list(this.iter()) : py.list(this.iter().map((v) => py.list(objArrayIn(v).iter())))))
      case 'copy': return fn(() => py.obj(new ObjArray([...this.cells], this.shape)))
      case 'flatten': case 'ravel': return fn(() => py.obj(new ObjArray([...this.cells])))
    }
    return undefined
  }
}

/** s.dtype: prints like numpy's dtype, and compares equal to its name or to the type */
export class DtypeObj extends PyObj {
  readonly cls = 'dtype'
  name: string
  constructor(name: string) {
    super()
    this.name = name
  }
  repr() { return this.name === 'category' ? 'CategoricalDtype' : `dtype('${this.name === 'object' ? 'O' : this.name}')` }
  str() { return this.name }
  private matches(o: Value): boolean {
    if (o.k === 'str') return o.v === this.name || (o.v === 'O' && this.name === 'object') || (o.v === 'int' && this.name === 'int64') || (o.v === 'float' && this.name === 'float64')
    if (o.k === 'type') {
      const t = o.name.replace('numpy.', '')
      return t === this.name || (t === 'int' && this.name === 'int64') || (t === 'float' && this.name === 'float64') || (t === 'object' && this.name === 'object') || (t === 'bool_' && this.name === 'bool')
    }
    if (o.k === 'dtype') return o.d === this.name
    return o.k === 'obj' && o.o instanceof DtypeObj && o.o.name === this.name
  }
  binop(op: string, other: Value) {
    if (op === '==') return py.bool(this.matches(other))
    if (op === '!=') return py.bool(!this.matches(other))
    return undefined
  }
  getAttr(name: string): Value | undefined {
    if (name === 'name') return py.str(this.name)
    if (name === 'kind') return py.str(this.name === 'int64' ? 'i' : this.name === 'float64' ? 'f' : this.name === 'bool' ? 'b' : 'O')
    return undefined
  }
}

// ---------------------------------------------------------------- printing

/** what a categorical prints: a missing value turns whole-number categories into floats, or into objects */
const displayCells = (s: Series, rows: number[]): { cells: Cell[]; dt: DT } => {
  const cells = rows.map((r) => s.values[r])
  if (!s.cats || !cells.some(isMissing)) return { cells, dt: s.dtype }
  if (s.cats.kind === 'int64') return { cells, dt: 'object' }
  if (s.cats.kind.startsWith('interval[int64')) return { cells: cells.map((c) => (typeof c === 'string' ? c.replace(/-?\d+/g, '$&.0') : c)), dt: 'object' }
  return { cells, dt: s.dtype }
}

/** a Series as text (SeriesFormatter) */
export const seriesToString = (s: Series, o: { maxRows?: number | null; minRows?: number | null; index?: boolean; header?: boolean; name?: boolean; dtype?: boolean; length?: boolean | 'truncate'; naRep?: string; floatFormat?: (v: number) => string } = {}): string => {
  const maxRows = o.maxRows === undefined ? OPTIONS.max_rows : o.maxRows
  const minRows = o.minRows === undefined ? OPTIONS.min_rows : o.minRows
  const showIndex = o.index ?? true
  const n = s.length
  let rows: number[] = Array.from({ length: n }, (_, i) => i)
  let trRow: number | null = null
  if (maxRows && n > maxRows) {
    const lim = minRows ? Math.min(minRows, maxRows) : maxRows
    if (lim === 1) {
      trRow = 1
      rows = rows.slice(0, 1)
    } else {
      trRow = Math.floor(lim / 2)
      rows = [...rows.slice(0, trRow), ...rows.slice(-trRow)]
    }
  }
  const parts: string[] = []
  if (o.name !== false && s.name !== null) parts.push(`Name: ${labelStr(s.name)}`)
  if (o.length === true || ((o.length ?? 'truncate') === 'truncate' && trRow !== null)) parts.push(`Length: ${n}`)
  if (o.dtype !== false) parts.push(`dtype: ${dtName(s)}`)
  let footer = parts.join(', ')
  if (s.cats) footer += `${footer ? '\n' : ''}${categoricalFooter(s.cats)}`
  if (!n) return `${s.cls}([], ${footer})`
  const tr = s.index.pick(rows)
  const hasNames = tr.names.some((x) => x !== null)
  let fmtIndex: string[]
  if (tr.multi) {
    const levels = tr.formatted().map((l, j) => [tr.names[j] === null ? '' : labelStr(tr.names[j]), ...l])
    fmtIndex = adjoin(2, ...sparsify(levels, 1)).split('\n')
  } else fmtIndex = [s.index.name === null ? '' : labelStr(s.index.name), ...tr.formatted()[0]]
  const shown = displayCells(s, rows)
  const raw = formatArray(shown.cells, shown.dt, showIndex, o.naRep, o.floatFormat)
  const fmtValues = fixedWidth(raw, 'right')
  if (trRow !== null) {
    const width = fmtValues[trRow - 1].length
    fmtValues.splice(trRow, 0, justify(width > 3 ? '...' : '..', width, 'center'))
    fmtIndex.splice(trRow + 1, 0, '')
  }
  let result = showIndex ? adjoin(3, fmtIndex.slice(1), fmtValues) : adjoin(0, fmtValues)
  if ((o.header ?? true) && hasNames && showIndex) result = `${fmtIndex[0]}\n${result}`
  return footer ? `${result}\n${footer}` : result
}

interface FrameFormat { maxRows?: number | null; minRows?: number | null; maxCols?: number | null; lineWidth?: number | null; index?: boolean; header?: boolean | string[]; showDims?: boolean | string; naRep?: string; floatFormat?: (v: number) => string; colSpace?: number; justifyHeader?: 'left' | 'right' | 'center' }

const isNumericDT = (s: Series) => !s.cats && s.dtype !== 'object'

/** the text columns of the rows and columns kept: index first */
const strCols = (f: Frame, rows: number[], cols: number[], o: FrameFormat): string[][] => {
  const showIndex = o.index ?? true
  const header = o.header ?? true
  const trColumns = f.columns.pick(cols)
  const hasIdxNames = f.index.names.some((x) => x !== null) && showIndex
  const hasColNames = f.columns.names.some((x) => x !== null) && header !== false
  const space = o.colSpace ?? 0
  let headers: string[][]
  if (header === false) headers = []
  else if (Array.isArray(header)) headers = cols.map((k) => [` ${header[k]}`])
  else if (trColumns.multi) {
    const fmt = trColumns.formatted()
    const sparse = OPTIONS.multi_sparse ? sparsify(fmt) : fmt
    headers = cols.map((_, i) => sparse.map((l) => l[i]))
  } else {
    const fmt = trColumns.formatted()[0]
    // keyed by label and zipped with the whole frame's dtypes, as pandas does: a cut table takes the dtype at that position
    const lead = new Map<string, boolean>()
    fmt.forEach((x, i) => {
      if (i < f.data.length) lead.set(x, isNumericDT(f.data[i]))
    })
    headers = fmt.map((x) => [lead.get(x) ? ` ${x}` : x])
  }
  if (hasIdxNames && header !== false) headers = headers.map((x) => [...x, ''])
  const out: string[][] = cols.map((k, i) => {
    const shown = displayCells(f.data[k], rows)
    const raw = formatArray(shown.cells, shown.dt, showIndex, o.naRep, o.floatFormat)
    const mode = o.justifyHeader ?? 'right'
    if (header === false) return fixedWidth(raw, mode, space)
    const cheader = headers[i]
    const hw = Math.max(space, ...cheader.map((x) => x.length))
    const cells = fixedWidth(raw, mode, hw)
    const w = Math.max(hw, ...cells.map((x) => x.length))
    return [...cheader.map((x) => justify(x, w, mode)), ...cells]
  })
  if (!showIndex) return out
  const tr = f.index.pick(rows)
  let idx: string[]
  if (tr.multi) {
    const levels = tr.formatted().map((l, j) => (hasIdxNames ? [tr.names[j] === null ? '' : labelStr(tr.names[j]), ...l] : l))
    const sp = OPTIONS.multi_sparse ? sparsify(levels, hasIdxNames ? 1 : 0) : levels
    idx = adjoin(1, ...sp.map((l) => fixedWidth(l, 'left'))).split('\n')
  } else {
    const l = hasIdxNames ? [f.index.name === null ? '' : labelStr(f.index.name), ...tr.formatted()[0]] : tr.formatted()[0]
    idx = adjoin(1, fixedWidth(l, 'left', space)).split('\n')
  }
  if (header !== false) {
    const colHeader = hasColNames ? f.columns.names.map((x) => (x === null ? '' : labelStr(x))) : Array.from({ length: f.columns.nlevels }, () => '')
    idx = [...colHeader, ...idx]
  }
  return [idx, ...out]
}

/** a DataFrame as text (DataFrameFormatter + StringFormatter) */
export const frameToString = (f: Frame, o: FrameFormat = {}): string => {
  const maxRows = o.maxRows === undefined ? OPTIONS.max_rows : o.maxRows
  const minRows = o.minRows === undefined ? OPTIONS.min_rows : o.minRows
  const maxCols = o.maxCols === undefined ? OPTIONS.max_columns : o.maxCols
  const lineWidth = o.lineWidth === undefined ? (OPTIONS.expand_frame_repr ? OPTIONS.width ?? TERMINAL_WIDTH : null) : o.lineWidth
  const showIndex = o.index ?? true
  const n = f.nrows
  const m = f.ncols
  if (!n || !m) {
    const list = (x: IndexObj) => `[${x.labels.map(labelStr).join(', ')}]`
    const cols = f.columns.multi ? f.columns : null
    return `Empty DataFrame\nColumns: ${cols ? list(cols) : list(f.columns)}\nIndex: ${list(f.index)}`
  }
  const inTerminal = maxCols === 0 || maxRows === 0
  let colsFitted: number | null = inTerminal && maxCols === 0 && m > TERMINAL_WIDTH ? TERMINAL_WIDTH : maxCols
  let rowsFitted: number | null = maxRows
  if (maxRows && n > maxRows && minRows) rowsFitted = Math.min(minRows, maxRows)
  const all = (k: number) => Array.from({ length: k }, (_, i) => i)
  const build = () => {
    const truncH = !!colsFitted && m > colsFitted
    const truncV = !!rowsFitted && n > rowsFitted
    let cols = all(m)
    let trCol = 0
    if (truncH) {
      trCol = Math.floor(colsFitted! / 2)
      if (trCol >= 1) cols = [...cols.slice(0, trCol), ...cols.slice(-trCol)]
      else {
        trCol = maxCols!
        cols = cols.slice(0, trCol)
      }
    }
    let rows = all(n)
    let trRow = 0
    if (truncV) {
      trRow = Math.floor(rowsFitted! / 2)
      if (trRow >= 1) rows = [...rows.slice(0, trRow), ...rows.slice(-trRow)]
      else {
        trRow = maxRows!
        rows = rows.slice(0, trRow)
      }
    }
    const sc = strCols(f, rows, cols, o)
    if (truncH || truncV) {
      const indexLength = sc[0].length
      if (truncH) sc.splice(trCol + (showIndex ? 1 : 0), 0, Array.from({ length: indexLength }, () => ' ...'))
      if (truncV) {
        const nHeader = indexLength - rows.length
        sc.forEach((col, ix) => {
          let cw = col[trRow]?.length ?? 0
          const isDot = truncH && ix === trCol + (showIndex ? 1 : 0)
          const dots = cw > 3 || isDot ? '...' : '..'
          let mode: 'left' | 'right' = 'right'
          if (ix === 0 && showIndex) mode = 'left'
          else if (isDot) cw = 4
          col.splice(trRow + nHeader, 0, justify(dots, cw, mode))
        })
      }
    }
    return { sc, truncated: truncH || truncV }
  }
  let { sc, truncated } = build()
  let text: string
  if (lineWidth === null) text = adjoin(1, ...sc)
  else if (maxCols === null || maxCols > 0) text = joinMultiline(sc, lineWidth, showIndex)
  else {
    const lines = adjoin(1, ...sc).split('\n')
    const maxLen = Math.max(...lines.map((l) => l.length))
    let adj = maxLen - TERMINAL_WIDTH + 1
    let lens = sc.map((c) => Math.max(...c.map((x) => x.length)))
    let count = lens.length
    while (adj > 0 && count > 1) {
      // Python's round(): half to even
      const half = count / 2
      const mid = half % 1 === 0.5 ? 2 * Math.round(half / 2) : Math.round(half)
      adj -= lens[mid] + 1
      lens = lens.filter((_, i) => i !== mid)
      count = lens.length
    }
    colsFitted = Math.max(count - 1, 2)
    ;({ sc, truncated } = build())
    text = adjoin(1, ...sc)
  }
  const dims = o.showDims === undefined ? OPTIONS.show_dimensions : o.showDims
  if (dims === true || (dims === 'truncate' && truncated)) text += `\n\n[${n} rows x ${m} columns]`
  return text
}

/** the dtype pandas prints: 'category' for a cut result */
export const dtName = (s: Series): string => (s.cats ? 'category' : s.width ?? s.dtype)

// ---------------------------------------------------------------- traced methods

export type Params = Record<string, Value | undefined>
type Picks = { rows: number[] | null; cols: number[] | null }
type Groups = { keys: string[]; ofRow: number[] }
/** what a method tells the step view besides its result */
export interface Ctx { api?: string; picks?: Picks | null; groups?: Groups | null }
export type Method<S> = [string, (self: S, p: Params, h: Host, ctx: Ctx) => Value]

const depth = (h: Host) => (h.state.get('pandas.depth') as number | undefined) ?? 0

/** run library work whose inner calls should not show up as steps of their own */
export const quietly = <T>(h: Host, f: () => T): T => {
  h.state.set('pandas.depth', depth(h) + 1)
  try {
    return f()
  } finally {
    h.state.set('pandas.depth', depth(h) - 1)
  }
}

export const tracingOn = (h: Host) => h.tracing && depth(h) === 0

const tableOfValue = (v: Value): TableSpec | null => (v.k === 'obj' && v.o instanceof Frame ? tableOfFrame(v.o) : v.k === 'obj' && v.o instanceof Series ? tableOfSeries(v.o) : null)

/** bind a call to a method's supported parameters; anything else is refused rather than ignored */
export const bindParams = (h: Host, cls: string, name: string, spec: string, args: Value[], kw: Kw): Params => {
  const names = spec ? spec.split(' ') : []
  const star = names.indexOf('*')
  const bar = names.indexOf('|')
  const positional = star >= 0 ? names.slice(0, star) : bar >= 0 ? names.slice(0, bar) : names.filter((x) => x !== '**')
  const anyKw = names.includes('**')
  const p: Params = {}
  if (args.length > positional.length && star < 0) throw h.err('TypeError', `${cls}.${name}() takes ${bar >= 0 ? `from 1 to ${positional.length + 1}` : positional.length + 1} positional argument${positional.length ? 's' : ''} but ${args.length + 1} were given`)
  args.slice(0, positional.length).forEach((v, i) => { p[positional[i]] = v })
  if (star >= 0) p.__args = py.tuple(args.slice(positional.length))
  const extra = new PyDict()
  for (const [k, v] of Object.entries(kw)) {
    if (names.includes(k)) {
      if (p[k]) throw h.err('TypeError', `${cls}.${name}() got multiple values for argument '${k}'`)
      p[k] = v
    } else if (anyKw) extra.set(py.str(k), v)
    else throw h.err('NotImplementedError', `${cls}.${name}(${k}=...) is not available in this sandbox`)
  }
  if (anyKw) p.__kw = { k: 'dict', d: extra }
  return p
}

/** a method as Python sees it: binds its arguments, runs, and records one step (inner pandas calls stay silent) */
export const method = <S extends Series | Frame>(self: S, cls: string, name: string, m: Method<S>, h: Host): Value => ({
  k: 'fn', name,
  call: (args, kw) => {
    const p = bindParams(h, cls, name, m[0], args, kw)
    h.state.set('pandas.cls', cls)
    // a groupby or a window does nothing yet: the step is the aggregation that follows
    const tracing = tracingOn(h) && !LAZY_METHODS.includes(name) && (self instanceof Frame ? self.nrows : self.length) <= MAX_TRACE_ROWS
    const input = tracing ? (self instanceof Frame ? tableOfFrame(self) : tableOfSeries(self as Series)) : null
    const ctx: Ctx = {}
    const out = quietly(h, () => m[1](self, p, h, ctx))
    if (input) {
      const label = h.nameOf(py.obj(self), self instanceof Frame ? 'df' : 's')
      const res = out.k === 'none' ? (self instanceof Frame ? tableOfFrame(self) : tableOfSeries(self as Series)) : tableOfValue(out)
      emitFrame(h, ctx.api ?? name, [{ label, table: input }], res, ctx.picks ?? null, ctx.groups ?? null, res ? '' : repr(out))
    }
    return out
  },
})

const MAX_TRACE_ROWS = 200
const LAZY_METHODS = ['groupby', 'rolling', 'expanding']

const emitFrame = (h: Host, api: string, inputs: { label: string; table: TableSpec }[], result: TableSpec | null, picks: Picks | null, groups: Groups | null, resultText: string) => {
  h.emit({ type: 'frame', api, inputs, result, resultText, picks, groups })
}

/** record a step; the tables are only built when the step view will keep them */
export const traceFrame = (h: Host, api: string, make: () => { inputs: { label: string; table: TableSpec }[]; result: TableSpec | null; picks?: Picks | null; groups?: Groups | null; resultText?: string }) => {
  if (!tracingOn(h)) return
  const t = make()
  if (t.inputs.some((i) => i.table.index.length > MAX_TRACE_ROWS)) return
  emitFrame(h, api, t.inputs, t.result, t.picks ?? null, t.groups ?? null, t.resultText ?? '')
}

// ---------------------------------------------------------------- arithmetic

export const CMP = ['>', '<', '>=', '<=', '==', '!=']
const LOGIC = ['&', '|', '^']
const OP_DUNDER: Record<string, string> = { '+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '//': 'floordiv', '%': 'mod', '**': 'pow' }

/** numpy's floor_divide for floats */
const floorDiv = (a: number, b: number): number => {
  if (b === 0) return a / b
  const mod0 = a % b
  let mod = mod0
  let div = (a - mod) / b
  if (mod && b < 0 !== mod < 0) {
    mod += b
    div -= 1
  }
  if (!div) return 0 * (a / b)
  const fl = Math.floor(div)
  return div - fl > 0.5 ? fl + 1 : fl
}

const remainder = (a: number, b: number): number => {
  const mod = a % b
  if (b === 0) return mod
  if (mod) return b < 0 !== mod < 0 ? mod + b : mod
  return b < 0 ? -0 : 0
}

const numOp = (op: string, x: number, y: number): number => {
  switch (op) {
    case '+': return x + y
    case '-': return x - y
    case '*': return x * y
    case '/': return x / y
    case '//': return floorDiv(x, y)
    case '%': return remainder(x, y)
    case '**': return x === 1 || y === 0 ? 1 : x ** y
  }
  return NaN
}

const pyTypeName = (c: Cell): string => (c === null ? 'NoneType' : typeof c === 'string' ? 'str' : typeof c === 'boolean' ? 'bool' : typeof c === 'number' ? (Number.isInteger(c) ? 'int' : 'float') : c.v.k === 'float' ? 'float' : c.v.k)

/** an element of an object column in Python arithmetic */
const objArith = (h: Host, op: string, x: Cell, y: Cell): Cell => {
  if (isMissing(x) || isMissing(y)) return NaN
  const floaty = (c: Cell) => (c instanceof PyCell && c.v.k === 'float') || (typeof c === 'number' && !Number.isInteger(c))
  if (floaty(x) || floaty(y) || (op === '/' && (typeof x === 'number' || typeof x === 'boolean') && (typeof y === 'number' || typeof y === 'boolean'))) {
    const a = numCell(x)
    const b = numCell(y)
    if (!Number.isNaN(a) && !Number.isNaN(b)) return boxFloat(numOp(op, a, b))
  }
  if (typeof x === 'string' && typeof y === 'string' && op === '+') return x + y
  if (typeof x === 'string' && typeof y === 'number' && op === '*') return x.repeat(Math.max(0, y))
  if (typeof x === 'number' && typeof y === 'string' && op === '*') return y.repeat(Math.max(0, x))
  if (typeof x === 'string' && op === '+') throw h.err('TypeError', `can only concatenate str (not "${pyTypeName(y)}") to str`)
  if ((typeof x === 'number' || typeof x === 'boolean') && (typeof y === 'number' || typeof y === 'boolean')) return numOp(op, Number(x), Number(y))
  throw h.err('TypeError', `unsupported operand type(s) for ${op}: '${pyTypeName(x)}' and '${pyTypeName(y)}'`)
}

/** Python's comparison, in which NaN and None are never equal */
const objCompare = (h: Host, op: string, x: Cell, y: Cell): boolean => {
  const a = x instanceof PyCell && (x.v.k === 'float' || x.v.k === 'int') ? x.v.v : x
  const b = y instanceof PyCell && (y.v.k === 'float' || y.v.k === 'int') ? y.v.v : y
  if (op === '==') return !isMissing(a) && !isMissing(b) && (a === b || (typeof a !== 'object' && typeof b !== 'object' && typeof a !== 'string' && typeof b !== 'string' && Number(a) === Number(b)))
  if (op === '!=') return !objCompare(h, '==', a, b)
  if (isMissing(a) || isMissing(b)) return false
  const num = (c: Cell) => typeof c === 'number' || typeof c === 'boolean'
  if (!(typeof a === 'string' && typeof b === 'string') && !(num(a) && num(b))) throw h.err('TypeError', `'${op}' not supported between instances of '${pyTypeName(a)}' and '${pyTypeName(b)}'`)
  const p = typeof a === 'boolean' ? Number(a) : a
  const q = typeof b === 'boolean' ? Number(b) : b
  return op === '>' ? p! > q! : op === '<' ? p! < q! : op === '>=' ? p! >= q! : p! <= q!
}

const UFUNC_NAMES: Record<string, string> = { '+': 'add', '-': 'subtract', '*': 'multiply', '/': 'divide', '//': 'floor_divide', '%': 'remainder', '**': 'power' }
const BOOL_OP_NAMES: Record<string, string> = { '/': 'truediv', '//': 'floordiv', '**': 'pow', '%': 'mod' }

const logicOp = (op: string, x: boolean, y: boolean) => (op === '&' ? x && y : op === '|' ? x || y : x !== y)

/** element-wise op with dtype rules. ys is one cell per row, or one cell for all */
export const applyOp = (h: Host, op: string, xs: Cell[], xdt: DT, ys: Cell[] | { scalar: Cell }, ydt: DT, reflected: boolean): { vals: Cell[]; dt: DT } => {
  const at = 'scalar' in ys ? () => ys.scalar : (i: number) => ys[i]
  const n = xs.length
  const vals: Cell[] = new Array(n)
  if (CMP.includes(op)) {
    for (let i = 0; i < n; i++) vals[i] = reflected ? objCompare(h, op, at(i), xs[i]) : objCompare(h, op, xs[i], at(i))
    return { vals, dt: 'bool' }
  }
  if (LOGIC.includes(op)) {
    if (xdt === 'int64' && ydt === 'int64') {
      for (let i = 0; i < n; i++) {
        const a = xs[i] as number
        const b = at(i) as number
        vals[i] = op === '&' ? a & b : op === '|' ? a | b : a ^ b
      }
      return { vals, dt: 'int64' }
    }
    for (let i = 0; i < n; i++) {
      const a = isMissing(xs[i]) ? false : !!xs[i]
      const b = isMissing(at(i)) ? false : !!at(i)
      vals[i] = logicOp(op, a, b)
    }
    return { vals, dt: 'bool' }
  }
  if ((xdt === 'int64' || xdt === 'float64' || xdt === 'bool') && 'scalar' in ys && typeof ys.scalar === 'string') {
    if (op === '*') throw h.err('TypeError', "The 'out' kwarg is necessary. Use numpy.strings.multiply without it.")
    if (op !== '+' && op !== '-') throw h.err('TypeError', `ufunc '${UFUNC_NAMES[op]}' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''`)
    const a = `dtype('${xdt}')`
    const b = `dtype('<U${ys.scalar.length}')`
    throw h.err('UFuncTypeError', `ufunc '${UFUNC_NAMES[op]}' did not contain a loop with signature matching types (${reflected ? b : a}, ${reflected ? a : b}) -> None`)
  }
  if (xdt === 'object' || ydt === 'object') {
    for (let i = 0; i < n; i++) vals[i] = reflected ? objArith(h, op, at(i), xs[i]) : objArith(h, op, xs[i], at(i))
    return { vals, dt: 'object' }
  }
  if (xdt === 'bool' && ydt === 'bool') {
    if (op === '+' || op === '*') {
      for (let i = 0; i < n; i++) vals[i] = logicOp(op === '+' ? '|' : '&', xs[i] as boolean, at(i) as boolean)
      return { vals, dt: 'bool' }
    }
    if (op === '-') throw h.err('TypeError', 'numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.')
    throw h.err('NotImplementedError', `operator '${BOOL_OP_NAMES[op] ?? op}' not implemented for bool dtypes`)
  }
  let zeroDiv = false
  for (let i = 0; i < n; i++) {
    const a = numCell(xs[i])
    const b = numCell(at(i))
    const [l, r] = reflected ? [b, a] : [a, b]
    if (r === 0 && (op === '//' || op === '%')) zeroDiv = true
    if (op === '**' && r < 0 && xdt !== 'float64' && ydt !== 'float64') throw h.err('ValueError', 'Integers to negative integer powers are not allowed.')
    vals[i] = numOp(op, l, r)
  }
  const isInt = xdt !== 'float64' && ydt !== 'float64' && op !== '/' && !zeroDiv
  return { vals, dt: isInt ? 'int64' : 'float64' }
}

/** the union pandas aligns two axes on: kept if they match, sorted if it can be */
export const unionIndex = (a: IndexObj, b: IndexObj): IndexObj => {
  if (a.equals(b)) return a
  const labels = [...a.labels, ...b.labels.filter((l) => !a.has(l))]
  const order = sortOrder([labels], [true])
  const dt = commonDT([a.dt, b.dt])
  if (a.multi && b.multi) return IndexObj.fromArrays(a.multi.arrays.map((_, j) => order.map((i) => (labels[i] as PyCell).v).map((t) => cellOfTuple(t, j))), a.multi.names, a.multi.dts)
  return new IndexObj(order.map((i) => labels[i]), a.name === b.name ? a.name : null, dt)
}

const cellOfTuple = (t: Value, j: number): Cell => {
  const x = (t as { items: Value[] }).items[j]
  return x.k === 'int' || x.k === 'float' ? x.v : x.k === 'bool' ? x.v : x.k === 'str' ? x.v : x.k === 'none' ? null : new PyCell(x)
}

/** the cells of a Series on another index; missing labels are NaN */
export const reindexCells = (s: Series, to: IndexObj, fill: Cell = NaN): { cells: Cell[]; dt: DT } => {
  if (s.index.equals(to)) return { cells: s.values, dt: s.dtype }
  if (!s.index.isUnique) throw new PyError('ValueError', 'cannot reindex on an axis with duplicate labels')
  let missing = false
  const cells = to.labels.map((l) => {
    const at = s.index.locate(l)
    if (!at.length) {
      missing = true
      return fill
    }
    return s.values[at[0]]
  })
  if (!missing) return { cells, dt: s.dtype }
  const dt = isMissing(fill) ? (s.dtype === 'int64' ? 'float64' : s.dtype === 'bool' ? 'object' : s.dtype) : commonDT([s.dtype, inferDT([fill])])
  return { cells: dt === 'float64' ? cells.map(numCell) : cells, dt }
}

/** one side of an operation, scalar or per-row */
const operand = (h: Host, other: Value, n: number): { ys: Cell[] | { scalar: Cell }; dt: DT } => {
  if (other.k === 'array' && (other.a.ndim === 0 || (other.a.size === 1 && n !== 1))) return { ys: { scalar: cellOf(h, other) }, dt: arrayDT(other.a) }
  if (other.k === 'list' || other.k === 'tuple' || other.k === 'array' || other.k === 'range' || (other.k === 'obj' && (other.o instanceof IndexObj || other.o instanceof ObjArray))) {
    const t = typedCells(h, other)
    if (t.cells.length === 1 && n !== 1 && other.k !== 'obj') return { ys: { scalar: t.cells[0] }, dt: t.dt }
    return { ys: t.cells, dt: t.dt }
  }
  const dt: DT = other.k === 'int' ? 'int64' : other.k === 'float' ? 'float64' : other.k === 'bool' ? 'bool' : 'object'
  return { ys: { scalar: cellOf(h, other) }, dt }
}

/** a sized column keeps its dtype beside a Python number, wrapping like NumPy; anything wider takes over */
const keepWidth = (out: Cell[], dt: DT, op: string, s: Series, other: Value): Series | null => {
  const w = s.width
  if (!w || CMP.includes(op) || LOGIC.includes(op)) return null
  const pyInt = other.k === 'int'
  const pyFloat = other.k === 'float'
  const same = other.k === 'obj' && other.o instanceof Series && other.o.width === w
  if (w.startsWith('float') ? !(pyInt || pyFloat || same) : !(pyInt || same) || dt !== 'int64') return null
  if (w === 'float32') return s.with(out.map((c) => Math.fround(numCell(c))), 'float64')
  const bits = Number(/\d+/.exec(w)![0])
  const m = 2 ** bits
  const signed = !w.startsWith('u')
  return s.with(out.map((c) => {
    const u = ((numCell(c) % m) + m) % m
    return signed && u >= m / 2 ? u - m : u
  }), 'int64')
}

/** s + other (flex=false rejects mislabelled comparisons, as the operators do) */
export const seriesOp = (h: Host, s: Series, op: string, other: Value, reflected: boolean, fill?: Cell, flex = false): Series => {
  if (other.k === 'none' && !CMP.includes(op)) {
    const [l, r] = reflected ? ['NoneType', s.values[0] === undefined ? 'float' : pyTypeName(s.values[0])] : [s.values[0] === undefined ? 'float' : pyTypeName(s.values[0]), 'NoneType']
    throw h.err('TypeError', `unsupported operand type(s) for ${op}: '${l}' and '${r}'`)
  }
  if (other.k === 'obj' && other.o instanceof Series) {
    const o = other.o
    if (CMP.includes(op) && !flex && !s.index.equals(o.index)) throw h.err('ValueError', 'Can only compare identically-labeled Series objects')
    const index = unionIndex(s.index, o.index)
    let a = reindexCells(s, index)
    let b = reindexCells(o, index)
    if (fill !== undefined) {
      const ac = a.cells.map((c, i) => (isMissing(c) && !isMissing(b.cells[i]) ? fill : c))
      const bc = b.cells.map((c, i) => (isMissing(c) && !isMissing(a.cells[i]) ? fill : c))
      a = { cells: ac, dt: a.dt }
      b = { cells: bc, dt: b.dt }
    }
    const res = applyOp(h, op, a.cells, a.dt, b.cells, b.dt, reflected)
    const sized = index === s.index ? keepWidth(res.vals, res.dt, op, s, other) : null
    if (sized) {
      sized.width = s.width
      sized.name = s.name === o.name ? s.name : null
      return sized
    }
    return new Series(res.vals, index, s.name === o.name ? s.name : null, res.dt)
  }
  const isCmpList = CMP.includes(op) && (other.k === 'list' || other.k === 'tuple')
  const { ys, dt } = isCmpList ? { ys: typedCells(h, other).cells, dt: typedCells(h, other).dt } : operand(h, other, s.length)
  if (!('scalar' in ys) && ys.length !== s.length) {
    if (CMP.includes(op)) throw h.err('ValueError', `('Lengths must match to compare', (${s.length},), (${ys.length},))`)
    throw h.err('ValueError', `operands could not be broadcast together with shapes (${s.length},) (${ys.length},) `)
  }
  const xs = fill !== undefined ? s.values.map((c) => (isMissing(c) ? fill : c)) : s.values
  const res = applyOp(h, op, xs, s.dtype, ys, dt, reflected)
  const sized = keepWidth(res.vals, res.dt, op, s, other)
  if (sized) {
    sized.width = s.width
    return sized
  }
  return new Series(res.vals, s.index, s.name, res.dt)
}

/** df + other: columns line up by label, a Series lines up with the columns unless axis=0 */
export const frameOp = (h: Host, f: Frame, op: string, other: Value, reflected: boolean, opts: { axis?: 0 | 1; fill?: Cell; flex?: boolean } = {}): Frame => {
  const { axis = 1, fill, flex = false } = opts
  if (other.k === 'none' && !CMP.includes(op)) throw h.err('TypeError', `unsupported operand type(s) for ${op}: 'int' and 'NoneType'`)
  const col = (s: Series, ys: Cell[] | { scalar: Cell }, dt: DT, index: IndexObj): Series => {
    const xs = fill !== undefined ? s.values.map((c, i) => (isMissing(c) && !('scalar' in ys && isMissing(ys.scalar)) && !(!('scalar' in ys) && isMissing(ys[i])) ? fill : c)) : s.values
    const yy = fill !== undefined && !('scalar' in ys) ? ys.map((c, i) => (isMissing(c) && !isMissing(s.values[i]) ? fill : c)) : ys
    const res = applyOp(h, op, xs, s.dtype, yy, dt, reflected)
    const sized = index === s.index ? keepWidth(res.vals, res.dt, op, s, other) : null
    if (sized) {
      sized.width = s.width
      return sized
    }
    return new Series(res.vals, index, s.name, res.dt)
  }
  if (other.k === 'obj' && other.o instanceof Frame) {
    const o = other.o
    if (CMP.includes(op) && !flex && !(f.index.equals(o.index) && f.columns.equals(o.columns))) throw h.err('ValueError', 'Can only compare identically-labeled (both index and columns) DataFrame objects')
    const index = unionIndex(f.index, o.index)
    const columns = unionIndex(f.columns, o.columns)
    const data = columns.labels.map((c) => {
      const a = f.col(c)
      const b = o.col(c)
      const left = a ? reindexCells(a, index) : { cells: index.labels.map(() => NaN), dt: 'float64' as DT }
      const right = b ? reindexCells(b, index) : { cells: index.labels.map(() => NaN), dt: 'float64' as DT }
      return col(new Series(left.cells, index, c, left.dt), right.cells, right.dt, index)
    })
    return new Frame(columns, data, index)
  }
  if (other.k === 'obj' && other.o instanceof Series) {
    const o = other.o
    if (axis === 0) {
      if (CMP.includes(op) && !flex && !f.index.equals(o.index)) throw h.err('ValueError', 'Operands are not aligned. Do `left, right = left.align(right, axis=1, copy=False)` before operating.')
      const index = unionIndex(f.index, o.index)
      const right = reindexCells(o, index)
      return new Frame(f.columns, f.data.map((s) => {
        const left = reindexCells(s, index)
        return col(new Series(left.cells, index, s.name, left.dt), right.cells, right.dt, index)
      }), index)
    }
    if (CMP.includes(op) && !flex && !f.columns.equals(o.index)) throw h.err('ValueError', 'Operands are not aligned. Do `left, right = left.align(right, axis=1, copy=False)` before operating.')
    const columns = unionIndex(f.columns, o.index)
    const aligned = reindexCells(o, columns)
    const data = columns.labels.map((c, j) => {
      const s = f.col(c) ?? new Series(f.index.labels.map(() => NaN), f.index, c, 'float64')
      const y = aligned.cells[j]
      const ydt: DT = aligned.dt !== 'object' ? aligned.dt : typeof y === 'number' ? (Number.isInteger(y) ? 'int64' : 'float64') : y instanceof PyCell ? 'float64' : typeof y === 'boolean' ? 'bool' : 'object'
      return col(s, { scalar: y instanceof PyCell ? numCell(y) : y }, ydt, f.index)
    })
    return new Frame(columns, data, f.index)
  }
  if (other.k === 'array' && other.a.ndim === 2) {
    const [r, c] = other.a.shape
    if (r !== f.nrows || c !== f.ncols) throw h.err('ValueError', `Unable to coerce to DataFrame, shape must be (${f.nrows}, ${f.ncols}): given (${r}, ${c})`)
    const v = other.a.values()
    const dt = arrayDT(other.a)
    return f.same(f.data.map((s, k) => col(s, Array.from({ length: r }, (_, i) => (dt === 'bool' ? v[i * c + k] !== 0 : v[i * c + k])), dt, f.index)))
  }
  if (other.k === 'list' || other.k === 'tuple' || other.k === 'array' || (other.k === 'obj' && other.o instanceof IndexObj)) {
    const t = typedCells(h, other)
    if (t.cells.length !== f.ncols) throw h.err('ValueError', `Unable to coerce to Series, length must be ${f.ncols}: given ${t.cells.length}`)
    return f.same(f.data.map((s, k) => col(s, { scalar: t.cells[k] }, t.dt, f.index)))
  }
  const { ys, dt } = operand(h, other, f.nrows)
  return f.same(f.data.map((s) => col(s, ys, dt, f.index)))
}

// ---------------------------------------------------------------- selection helpers

/** Python's slice arithmetic */
export const sliceRange = (h: Host, sel: Value & { k: 'slice' }, n: number): number[] => {
  const part = (x: Value) => (x.k === 'none' ? null : h.toInt(x))
  const st = part(sel.step) ?? 1
  if (st === 0) throw h.err('ValueError', 'slice step cannot be zero')
  const norm = (v: number | null, d: number) => (v === null ? d : v < 0 ? Math.max(v + n, st > 0 ? 0 : -1) : Math.min(v, st > 0 ? n : n - 1))
  const a = norm(part(sel.start), st > 0 ? 0 : n - 1)
  const b = norm(part(sel.stop), st > 0 ? n : -1)
  const out: number[] = []
  for (let i = a; st > 0 ? i < b : i > b; i += st) out.push(i)
  return out
}

/** positions from an int / slice / list / boolean selector (iloc) */
export const positions = (h: Host, sel: Value, n: number, oob = 'single positional indexer is out-of-bounds'): number[] => {
  if (sel.k === 'slice') return sliceRange(h, sel, n)
  if (sel.k === 'int' || (sel.k === 'array' && sel.a.ndim === 0)) {
    const i = h.toInt(sel)
    if (i < -n || i >= n) throw h.err('IndexError', oob)
    return [i < 0 ? i + n : i]
  }
  if (sel.k === 'obj' && sel.o instanceof Series && sel.o.dtype === 'bool') {
    if (sel.o.index.dt === 'int64') throw h.err('NotImplementedError', 'iLocation based boolean indexing on an integer type is not available')
    throw h.err('ValueError', 'iLocation based boolean indexing cannot use an indexable as a mask')
  }
  const t = typedCells(h, sel)
  if (t.dt === 'bool') {
    if (t.cells.length !== n) throw h.err('IndexError', `Boolean index has wrong length: ${t.cells.length} instead of ${n}`)
    return t.cells.flatMap((b, i) => (b ? [i] : []))
  }
  return t.cells.map((c) => {
    const i = Number(c)
    if (i < -n || i >= n) throw h.err('IndexError', 'positional indexers are out-of-bounds')
    return i < 0 ? i + n : i
  })
}

/** the rows a boolean selector keeps, or null if it is not one */
const maskRows = (h: Host, sel: Value, index: IndexObj): number[] | null => {
  if (sel.k === 'obj' && sel.o instanceof Series && sel.o.dtype === 'bool') {
    const m = sel.o
    if (m.index.equals(index)) return m.values.flatMap((b, i) => (b ? [i] : []))
    return index.labels.flatMap((l, i) => {
      const at = m.index.locate(l)
      if (!at.length) throw h.err('IndexingError', 'Unalignable boolean Series provided as indexer (index of the boolean Series and of the indexed object do not match).')
      return m.values[at[0]] ? [i] : []
    })
  }
  const list = sel.k === 'list' || sel.k === 'tuple' ? sel.items.length > 0 && sel.items.every((x) => x.k === 'bool') : sel.k === 'array' ? sel.a.dtype === 'bool' : false
  if (!list) return null
  const cells = cellsOf(h, sel)
  if (cells.length !== index.length) throw h.err('IndexError', `Boolean index has wrong length: ${cells.length} instead of ${index.length}`)
  return cells.flatMap((b, i) => (b ? [i] : []))
}

/** an inclusive label slice (monotonic labels may be absent) */
const labelSlice = (h: Host, sel: Value & { k: 'slice' }, index: IndexObj): number[] => {
  const n = index.length
  const isMono = () => index.labels.every((l, i) => i === 0 || compareCells(index.labels[i - 1], l) <= 0)
  const bound = (v: Value, side: 'start' | 'stop'): number => {
    if (v.k === 'none') return side === 'start' ? 0 : n - 1
    const c = cellOf(h, v)
    const at = index.locate(c)
    if (at.length) return side === 'start' ? at[0] : at[at.length - 1]
    if (!isMono()) throw h.err('KeyError', repr(v))
    if (typeof c !== typeof index.labels[0] && n) throw h.err('TypeError', `cannot do slice indexing on Index with these indexers [${repr(v)}] of type ${pyTypeName(c)}`)
    if (side === 'start') {
      const i = index.labels.findIndex((l) => compareCells(l, c) >= 0)
      return i < 0 ? n : i
    }
    let j = -1
    index.labels.forEach((l, i) => { if (compareCells(l, c) <= 0) j = i })
    return j
  }
  const a = bound(sel.start, 'start')
  const b = bound(sel.stop, 'stop')
  const step = sel.step.k === 'none' ? 1 : h.toInt(sel.step)
  const out: number[] = []
  for (let i = a; i <= b; i += Math.abs(step)) out.push(i)
  return step < 0 ? out.reverse() : out
}

/** labels → positions: list-likes must all exist, with pandas' two KeyError forms */
const positionsOfLabels = (h: Host, labels: Cell[], dt: DT, index: IndexObj, axisName: string): number[] => {
  const missing = labels.filter((l) => !index.has(l))
  if (missing.length) {
    if (missing.length === labels.length) throw h.err('KeyError', repr(py.str(`None of [${new IndexObj(labels, null, dt).repr()}] are in the [${axisName}]`)))
    throw h.err('KeyError', repr(py.str(`[${missing.map(reprLabel).join(', ')}] not in index`)))
  }
  return labels.flatMap((l) => index.locate(l))
}

/** [] on a Series or the rows of df[a:b]: integers are positions unless the index is float */
const itemSlice = (h: Host, sel: Value & { k: 'slice' }, index: IndexObj): number[] => {
  const isPos = (x: Value) => x.k === 'none' || x.k === 'int'
  return isPos(sel.start) && isPos(sel.stop) && index.dt !== 'float64' ? sliceRange(h, sel, index.length) : labelSlice(h, sel, index)
}

/** partial MultiIndex key: the positions whose first levels match */
const multiPrefix = (index: IndexObj, key: Cell[]): number[] => {
  const m = index.multi!
  const out: number[] = []
  for (let r = 0; r < index.length; r++) if (key.every((k, j) => m.arrays[j][r] === k)) out.push(r)
  return out
}

/** the rest of a MultiIndex after dropping its first levels */
const dropLevels = (index: IndexObj, rows: number[], nDrop: number): IndexObj => {
  const m = index.multi!
  const arrays = m.arrays.slice(nDrop).map((a) => rows.map((r) => a[r]))
  if (arrays.length === 1) return new IndexObj(arrays[0], m.names[nDrop], m.dts[nDrop])
  return IndexObj.fromArrays(arrays, m.names.slice(nDrop), m.dts.slice(nDrop))
}

const compareCells = (x: Cell, y: Cell): number => {
  if (isMissing(x)) return isMissing(y) ? 0 : 1
  if (isMissing(y)) return -1
  if (x instanceof PyCell || y instanceof PyCell) {
    if (x instanceof PyCell && y instanceof PyCell && x.v.k === 'tuple' && y.v.k === 'tuple') {
      for (let j = 0; j < x.v.items.length; j++) {
        const c = compareCells(cellOfTuple(x.v, j), cellOfTuple(y.v, j))
        if (c) return c
      }
      return 0
    }
    const a = numCell(x)
    const b = numCell(y)
    return !Number.isNaN(a) && !Number.isNaN(b) ? a - b : String(labelStr(x)) < String(labelStr(y)) ? -1 : 1
  }
  if (typeof x === 'string' || typeof y === 'string') return String(x) < String(y) ? -1 : String(x) > String(y) ? 1 : 0
  return Number(x) - Number(y)
}

/** stable sort order by several key columns; missing values last whatever the direction */
export const sortOrder = (keys: Cell[][], asc: boolean[], naFirst = false): number[] => {
  const n = keys[0]?.length ?? 0
  return Array.from({ length: n }, (_, i) => i).sort((i, j) => {
    for (let k = 0; k < keys.length; k++) {
      const a = keys[k][i]
      const b = keys[k][j]
      const ma = isMissing(a)
      const mb = isMissing(b)
      if (ma || mb) {
        if (ma && mb) continue
        return (ma ? 1 : -1) * (naFirst ? -1 : 1)
      }
      const c = compareCells(a, b)
      if (c) return asc[k] ? c : -c
    }
    return i - j
  })
}

// ---------------------------------------------------------------- Series

export interface Cats { labels: Cell[]; kind: string; ordered: boolean }

export class Series extends PyObj {
  readonly cls = 'Series'
  values: Cell[]
  dtype: DT
  index: IndexObj
  name: Cell
  /** a categorical Series (cut, astype('category')): its categories */
  cats: Cats | null = null
  /** a sized dtype name (int32, float32) printed in place of the stored dtype */
  width: string | null = null
  /** a Categorical rather than a Series: pd.cut(list) */
  bare = false
  constructor(values: Cell[], index: IndexObj | Cell[] | null = null, name: Cell = null, dtype?: DT) {
    super()
    this.dtype = dtype ?? inferDT(values)
    this.values = normalise(values, this.dtype)
    this.index = toIndex(index, values.length)
    this.name = name
  }
  get length() { return this.values.length }
  repr() {
    if (!this.bare) return seriesToString(this)
    const footer = categoricalFooter(this.cats!)
    const n = this.length
    const list = (rows: number[]) => {
      const t = displayCells(this, rows)
      const quoted = t.cells.map((c, i) => (typeof c === 'string' && !this.cats!.kind.startsWith('interval') ? `'${c}'` : formatArray([c], t.dt)[0].trim() + (i < 0 ? '' : '')))
      return `[${quoted.join(', ')}]`
    }
    const all = Array.from({ length: n }, (_, i) => i)
    if (n > 10) return `${list(all.slice(0, 5)).slice(0, -1)}, ..., ${list(all.slice(-5)).slice(1)}\nLength: ${n}\n${footer}`
    return n ? `${list(all)}\n${footer}` : `[], ${footer}`
  }
  display() { return { type: 'table' as const, table: tableOfSeries(this) } }
  len() { return this.length }
  iter() { return this.values.map((c) => valueOf(c, this.dtype)) }
  contains(v: Value, h: Host) { return this.index.has(cellOf(h, v)) }
  truthy(h: Host): boolean { throw h.err('ValueError', 'The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().') }
  toArray(): NDArray {
    if (this.dtype === 'object') {
      if (this.values.some((c) => typeof c === 'string')) throw new PyError('TypeError', `the Series '${labelStr(this.name)}' holds text, so it cannot become a numeric array`)
      return NDArray.create(this.values.map(numCell), [this.length], 'float64')
    }
    return NDArray.create(this.values.map(numCell), [this.length], this.dtype === 'bool' ? 'bool' : this.width === 'uint8' ? 'uint8' : this.dtype === 'int64' ? 'int64' : 'float64')
  }
  isa(name: string) { return name === 'Series' }
  /** the values as numbers; text is a TypeError naming what needed numbers */
  numeric(h: Host, what: string): number[] {
    if (this.dtype === 'object' && this.values.some((c) => typeof c === 'string')) throw h.err('TypeError', `Could not convert ${JSON.stringify(this.values.slice(0, 3).map(labelStr).join(''))} to numeric (the Series '${labelStr(this.name)}' holds text; ${what} needs numbers)`)
    return this.values.map(numCell)
  }
  /** same labels, new values */
  with(values: Cell[], dtype?: DT, name: Cell = this.name) { return new Series(values, this.index, name, dtype) }
  /** the rows at these positions; keepRange for a slice, which keeps a RangeIndex */
  pick(rows: number[], keepRange = false): Series {
    const out = new Series(rows.map((r) => this.values[r]), this.index.pick(rows, keepRange), this.name, this.dtype)
    out.cats = this.cats
    out.width = this.width
    out.bare = this.bare
    return out
  }
  copy(): Series {
    const out = new Series([...this.values], this.index.withName(this.index.name), this.name, this.dtype)
    out.cats = this.cats
    out.width = this.width
    out.bare = this.bare
    return out
  }
  /** take over another Series' contents: inplace=True */
  adopt(o: Series) {
    this.values = o.values
    this.dtype = o.dtype
    this.index = o.index
    this.name = o.name
    this.cats = o.cats
    this.width = o.width
  }
  /** store cells at positions, widening the dtype only as far as they need */
  store(rows: number[], cells: Cell[]) {
    let dt = this.dtype
    for (const c of cells) dt = widenDT(dt, c)
    if (dt !== this.dtype) {
      this.values = convertCells(this.values, this.dtype, dt)
      this.dtype = dt
    } else if (this.values === this.index.labels) this.values = [...this.values]
    this.cats = null
    this.width = null
    rows.forEach((r, i) => {
      const c = cells[i]
      if (dt === 'float64') this.values[r] = numCell(c)
      else if (dt === 'object' && typeof c === 'number' && Number.isNaN(c)) this.values[r] = c
      else this.values[r] = c
    })
  }
  binop(op: string, other: Value, reflected: boolean, h: Host): Value | undefined {
    if (other.k === 'obj' && other.o instanceof Frame) return undefined
    if (!OP_DUNDER[op] && !CMP.includes(op) && !LOGIC.includes(op)) return undefined
    if (other.k === 'obj' && !(other.o instanceof Series || other.o instanceof IndexObj || other.o instanceof ObjArray) && !other.o.toArray) return undefined
    const out = quietly(h, () => seriesOp(h, this, op, other, reflected))
    traceFrame(h, CMP.includes(op) ? 'compare' : 'arith', () => ({ inputs: [{ label: h.nameOf(py.obj(this), 's'), table: tableOfSeries(this) }, ...(other.k === 'obj' && other.o instanceof Series ? [{ label: h.nameOf(other, 'other'), table: tableOfSeries(other.o) }] : [])], result: tableOfSeries(out) }))
    return py.obj(out)
  }
  unary(op: '-' | '+' | '~', h: Host): Value {
    if (op === '~') {
      if (this.dtype === 'bool') return py.obj(this.with(this.values.map((c) => !c), 'bool'))
      if (this.dtype === 'int64') return py.obj(this.with(this.values.map((c) => ~(c as number)), 'int64'))
      throw h.err('TypeError', "ufunc 'invert' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''")
    }
    if (this.dtype === 'bool') return py.obj(op === '-' ? this.with(this.values.map((c) => !c), 'bool') : this.copy())
    return py.obj(this.with(this.numeric(h, 'negation').map((v) => (op === '-' ? -v : v)), this.dtype))
  }
  /** s[label]: scalar for a unique label, Series for a repeated one */
  byLabel(h: Host, idx: Value, label: Cell): Value {
    const at = this.index.locate(label)
    if (at.length === 1) return valueOf(this.values[at[0]], this.dtype)
    if (at.length > 1) return py.obj(this.pick(at))
    if (this.index.multi && label !== null && !(label instanceof PyCell)) {
      const rows = multiPrefix(this.index, [label])
      if (rows.length) {
        const out = this.pick(rows)
        out.index = dropLevels(this.index, rows, 1)
        return py.obj(out)
      }
    }
    if (this.index.multi && idx.k === 'tuple') {
      const key = idx.items.map((x) => cellOf(h, x))
      const rows = multiPrefix(this.index, key)
      if (rows.length) {
        const out = this.pick(rows)
        out.index = dropLevels(this.index, rows, key.length)
        return py.obj(out)
      }
    }
    throw h.err('KeyError', repr(idx))
  }
  getItem(idx: Value, h: Host): Value {
    const m = maskRows(h, idx, this.index)
    if (m) return py.obj(this.pick(m))
    if (idx.k === 'slice') return py.obj(this.pick(itemSlice(h, idx, this.index), true))
    if (idx.k === 'func' || idx.k === 'fn') return this.getItem(h.call(idx, [py.obj(this)]), h)
    if (idx.k === 'list' || idx.k === 'array' || idx.k === 'range' || (idx.k === 'obj' && (idx.o instanceof Series || idx.o instanceof IndexObj))) {
      const t = typedCells(h, idx)
      return py.obj(this.pick(positionsOfLabels(h, t.cells, t.dt, this.index, 'index')))
    }
    const label = cellOf(h, idx)
    if (typeof label === 'number' && Number.isInteger(label) && !this.index.has(label) && this.index.dt === 'object' && !this.index.labels.some((l) => typeof l === 'number')) {
      const n = this.length
      if (label < -n || label >= n) throw h.err('IndexError', `index ${label} is out of bounds for axis 0 with size ${n}`)
      return valueOf(this.values[label < 0 ? label + n : label], this.dtype)
    }
    return this.byLabel(h, idx, label)
  }
  /** cells for the positions an assignment touches */
  valuesFor(h: Host, rows: number[], v: Value): Cell[] {
    if (v.k === 'obj' && v.o instanceof Series) {
      const o = v.o
      return rows.map((r) => {
        const at = o.index.locate(this.index.labels[r])
        return at.length ? o.values[at[0]] : NaN
      })
    }
    if (isListLike(v)) {
      const c = cellsOf(h, v)
      if (c.length !== rows.length) throw h.err('ValueError', `cannot set using a list-like indexer with a different length than the value`)
      return c
    }
    const c = cellOf(h, v)
    return rows.map(() => c)
  }
  setItem(idx: Value, v: Value, h: Host) {
    const m = maskRows(h, idx, this.index)
    if (m) return this.store(m, this.valuesFor(h, m, v))
    if (idx.k === 'slice') {
      const rows = itemSlice(h, idx, this.index)
      return this.store(rows, this.valuesFor(h, rows, v))
    }
    if (isListLike(idx)) {
      const t = typedCells(h, idx)
      const rows = positionsOfLabels(h, t.cells, t.dt, this.index, 'index')
      return this.store(rows, this.valuesFor(h, rows, v))
    }
    const label = cellOf(h, idx)
    const at = this.index.locate(label)
    if (at.length) return this.store(at, this.valuesFor(h, at, v))
    if (typeof label === 'number' && Number.isInteger(label) && this.index.dt === 'object') {
      const i = label < 0 ? label + this.length : label
      if (i >= 0 && i < this.length) return this.store([i], [cellOf(h, v)])
    }
    this.grow(label, cellOf(h, v))
  }
  /** a new label appends a row */
  grow(label: Cell, c: Cell) {
    const dt = widenDT(this.dtype, c)
    this.values = [...convertCells(this.values, this.dtype, dt), dt === 'float64' ? numCell(c) : c]
    this.dtype = dt
    this.index = new IndexObj([...this.index.labels, label], this.index.name, commonDT([this.index.dt, inferDT([label])]))
  }
  npCall(name: string, args: Value[], kw: Kw, h: Host, impl: (a: Value[], k: Kw) => Value) { return seriesNp(this, name, args, kw, h, impl) }
  getAttr(name: string, h: Host): Value | undefined {
    const m = seriesAttr(this, name, h)
    if (m) return m
    // a row from df.apply(…, axis=1) or iterrows(): its values by column name (row.height)
    const at = this.index.dt === 'object' ? this.index.locate(name) : []
    if (at.length === 1) return valueOf(this.values[at[0]], this.dtype)
    return undefined
  }
  setAttr(name: string, v: Value, h: Host): boolean {
    if (name === 'name') {
      this.name = cellOf(h, v)
      return true
    }
    if (name === 'index') {
      this.index = axisFrom(h, v, this.length, 'Length mismatch: Expected axis has')
      return true
    }
    return false
  }
}

/** df.index = … / df.columns = … */
const axisFrom = (h: Host, v: Value, n: number, what: string): IndexObj => {
  const ix = v.k === 'obj' && v.o instanceof IndexObj ? v.o : (() => {
    const t = typedCells(h, v)
    if (t.cells.length && t.cells.every((c) => c instanceof PyCell && c.v.k === 'tuple')) {
      const k = ((t.cells[0] as PyCell).v as { items: Value[] }).items.length
      const arrays = Array.from({ length: k }, (_, j) => t.cells.map((c) => cellOfTuple((c as PyCell).v, j)))
      return IndexObj.fromArrays(arrays, arrays.map(() => null), arrays.map((a) => inferDT(a)))
    }
    return new IndexObj(t.cells, null, t.dt)
  })()
  if (ix.length !== n) throw h.err('ValueError', `${what} ${n} elements, new values have ${ix.length} elements`)
  return ix
}

class SeriesIndexer extends PyObj {
  readonly cls: string
  s: Series
  kind: string
  constructor(s: Series, kind: string) {
    super()
    this.s = s
    this.kind = kind
    this.cls = kind === 'loc' ? '_LocIndexer' : kind === 'iloc' ? '_iLocIndexer' : kind === 'at' ? '_AtIndexer' : '_iAtIndexer'
  }
  repr() { return `<pandas.core.indexing.${this.cls} object>` }
  private rows(h: Host, idx: Value): number[] {
    const s = this.s
    if (idx.k === 'slice') return labelSlice(h, idx, s.index)
    const m = maskRows(h, idx, s.index)
    if (m) return m
    const t = typedCells(h, idx)
    return positionsOfLabels(h, t.cells, t.dt, s.index, 'index')
  }
  getItem(idx: Value, h: Host): Value {
    const s = this.s
    if (idx.k === 'func' || idx.k === 'fn') return this.getItem(h.call(idx, [py.obj(s)]), h)
    if (this.kind === 'iloc' || this.kind === 'iat') {
      if (idx.k === 'int' || (idx.k === 'array' && idx.a.ndim === 0)) return valueOf(s.values[positions(h, idx, s.length)[0]], s.dtype)
      return py.obj(s.pick(positions(h, idx, s.length), idx.k === 'slice'))
    }
    if (idx.k === 'slice' || isListLike(idx) && !(idx.k === 'tuple' && s.index.multi)) return py.obj(s.pick(this.rows(h, idx), idx.k === 'slice'))
    return s.byLabel(h, idx, cellOf(h, idx))
  }
  setItem(idx: Value, v: Value, h: Host) {
    const s = this.s
    if (this.kind === 'iloc' || this.kind === 'iat') {
      const rows = positions(h, idx, s.length)
      s.store(rows, s.valuesFor(h, rows, v))
      return
    }
    if (idx.k === 'slice' || isListLike(idx)) {
      const rows = this.rows(h, idx)
      s.store(rows, s.valuesFor(h, rows, v))
      return
    }
    const label = cellOf(h, idx)
    const at = s.index.locate(label)
    if (at.length) s.store(at, s.valuesFor(h, at, v))
    else s.grow(label, cellOf(h, v))
  }
}

export const seriesIndexer = (s: Series, kind: string) => new SeriesIndexer(s, kind)

// ---------------------------------------------------------------- DataFrame

export class Frame extends PyObj {
  readonly cls = 'DataFrame'
  columns: IndexObj
  data: Series[]
  index: IndexObj
  /** plain attributes set on the frame (df.foo = 1) */
  extra: Map<string, Value> | null = null
  constructor(columns: IndexObj | Cell[], data: Series[], index: IndexObj | Cell[] | null = null) {
    super()
    this.columns = columns instanceof IndexObj ? columns : new IndexObj(columns)
    const n = data[0]?.length ?? (index ? index.length : 0)
    this.index = toIndex(index, n)
    this.data = data.map((s, k) => this.member(s, k))
  }
  private member(s: Series, k: number) {
    const out = new Series(s.values, this.index, this.columns.labels[k], s.dtype)
    out.cats = s.cats
    out.width = s.width
    return out
  }
  get nrows() { return this.index.length }
  get ncols() { return this.columns.length }
  col(label: Cell): Series | null {
    const at = this.columns.locate(label)
    return at.length ? this.data[at[0]] : null
  }
  repr() { return frameToString(this) }
  display() { return { type: 'table' as const, table: tableOfFrame(this) } }
  len() { return this.nrows }
  iter() { return this.columns.iter() }
  contains(v: Value, h: Host) { return this.columns.has(cellOf(h, v)) }
  truthy(h: Host): boolean { throw h.err('ValueError', 'The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().') }
  isa(name: string) { return name === 'DataFrame' }
  toArray(): NDArray {
    const text = this.data.find((s) => s.dtype === 'object' && s.values.some((c) => typeof c === 'string'))
    if (text) throw new PyError('TypeError', `column '${labelStr(text.name)}' holds text: select the numeric columns before converting to an array`)
    const dt = commonDT(this.data.map((s) => s.dtype))
    const vals: number[] = new Array(this.nrows * this.ncols)
    const m = this.ncols
    this.data.forEach((s, k) => s.values.forEach((c, r) => { vals[r * m + k] = numCell(c) }))
    const allUint8 = m > 0 && this.data.every((s) => s.width === 'uint8')
    return NDArray.create(vals, [this.nrows, m], dt === 'bool' ? 'bool' : allUint8 ? 'uint8' : dt === 'int64' ? 'int64' : 'float64')
  }
  /** same labels, new columns */
  same(data: Series[]): Frame { return new Frame(this.columns, data, this.index) }
  /** a new frame with some rows and columns (positions); keepRange for a slice */
  take(rows: number[] | null, cols: number[] | null, keepRange = false): Frame {
    const cs = cols ?? this.data.map((_, k) => k)
    const data = cs.map((k) => (rows ? this.data[k].pick(rows) : this.data[k].copy()))
    return new Frame(cols ? this.columns.pick(cs, true) : this.columns, data, rows ? this.index.pick(rows, keepRange) : this.index)
  }
  /** a deep copy; its axes are its own, so that naming them stays here */
  copy(): Frame { return new Frame(this.columns.withName(this.columns.name), this.data.map((s) => s.copy()), this.index.withName(this.index.name)) }
  /** take over another frame's contents: inplace=True */
  adopt(o: Frame) {
    this.columns = o.columns
    this.index = o.index
    this.data = o.data.map((s, k) => this.member(s, k))
  }
  /** one row as a Series, in the dtype its columns share */
  row(r: number, keep?: number[]): Series {
    const cs = keep ?? this.data.map((_, k) => k)
    const dts = cs.map((k) => this.data[k].dtype)
    const dt = cs.some((k) => this.data[k].cats) ? 'object' : commonDT(dts)
    const vals = cs.map((k) => {
      const c = this.data[k].values[r]
      if (dt === 'float64') return numCell(c)
      return dt === 'object' && this.data[k].dtype === 'float64' ? boxFloat(c as number) : c
    })
    const out = new Series(vals, keep ? this.columns.pick(cs, true) : this.columns, this.index.labels[r], dt)
    out.width = sharedWidth(cs.map((k) => this.data[k]))
    return out
  }
  colPos(h: Host, name: Value): number {
    const c = cellOf(h, name)
    const at = this.columns.locate(c)
    if (!at.length) throw h.err('KeyError', repr(name))
    return at[0]
  }
  /** a column that must exist */
  byColumnSeries(h: Host, name: Value): Series { return this.data[this.colPos(h, name)] }
  /** a label on the columns: the column, or the sub-table under a first-level label of a MultiIndex */
  byColumn(h: Host, idx: Value, label: Cell): Value {
    const at = this.columns.locate(label)
    if (at.length === 1) return py.obj(this.data[at[0]])
    if (at.length > 1) return py.obj(this.take(null, at))
    if (this.columns.multi) {
      const key = idx.k === 'tuple' ? idx.items.map((x) => cellOf(h, x)) : [label]
      const cols = multiPrefix(this.columns, key)
      if (cols.length) {
        const out = this.take(null, cols)
        return py.obj(new Frame(dropLevels(this.columns, cols, key.length), out.data, this.index))
      }
    }
    throw h.err('KeyError', repr(idx))
  }
  getItem(idx: Value, h: Host): Value {
    const me = () => h.nameOf(py.obj(this), 'df')
    if (idx.k === 'func' || idx.k === 'fn') return this.getItem(quietly(h, () => h.call(idx, [py.obj(this)])), h)
    if (idx.k === 'slice') return py.obj(this.take(itemSlice(h, idx, this.index), null, true))
    if (idx.k === 'obj' && idx.o instanceof Frame) {
      const cond = idx.o
      if (!cond.data.every((s) => s.dtype === 'bool')) throw h.err('ValueError', 'Boolean array expected for the condition, not object')
      return py.obj(this.same(this.data.map((s, k) => {
        const m = cond.col(this.columns.labels[k])
        const vals = s.values.map((c, r) => (m && m.values[r] ? c : NaN))
        const dt = vals.some(isMissing) ? (s.dtype === 'int64' ? 'float64' : s.dtype === 'bool' ? 'object' : s.dtype) : s.dtype
        return s.with(vals, dt)
      })))
    }
    const mask = idx.k === 'obj' && idx.o instanceof Series && idx.o.dtype === 'bool' ? idx.o : null
    const listMask = !mask && ((idx.k === 'list' && idx.items.length > 0 && idx.items.every((x) => x.k === 'bool')) || (idx.k === 'array' && idx.a.dtype === 'bool'))
    if (mask || listMask) {
      if (listMask && cellsOf(h, idx).length !== this.nrows) throw h.err('ValueError', `Item wrong length ${cellsOf(h, idx).length} instead of ${this.nrows}.`)
      const rows = maskRows(h, idx, this.index)!
      const out = this.take(rows, null)
      traceFrame(h, 'filter', () => ({ inputs: [{ label: me(), table: tableOfFrame(this) }], result: tableOfFrame(out), picks: { rows, cols: null } }))
      return py.obj(out)
    }
    if (idx.k === 'list' || idx.k === 'array' || (idx.k === 'obj' && (idx.o instanceof Series || idx.o instanceof IndexObj || idx.o instanceof ObjArray))) {
      const t = typedCells(h, idx)
      const cols = positionsOfLabels(h, t.cells, t.dt, this.columns, 'columns')
      const out = this.take(null, cols)
      traceFrame(h, 'columns', () => ({ inputs: [{ label: me(), table: tableOfFrame(this) }], result: tableOfFrame(out), picks: { rows: null, cols } }))
      return py.obj(out)
    }
    const res = this.byColumn(h, idx, cellOf(h, idx))
    traceFrame(h, 'column', () => ({ inputs: [{ label: me(), table: tableOfFrame(this) }], result: tableOfValue(res), picks: { rows: null, cols: this.columns.locate(cellOf(h, idx)) } }))
    return res
  }
  setItem(idx: Value, v: Value, h: Host) {
    if (idx.k === 'obj' && idx.o instanceof Frame) {
      const cond = idx.o
      this.data.forEach((s, k) => {
        const m = cond.col(this.columns.labels[k])
        if (!m) return
        const rows = m.values.flatMap((b, r) => (b ? [r] : []))
        if (rows.length) s.store(rows, s.valuesFor(h, rows, v))
      })
      return
    }
    if (idx.k === 'slice' || (idx.k === 'obj' && idx.o instanceof Series && idx.o.dtype === 'bool')) {
      const rows = idx.k === 'slice' ? itemSlice(h, idx, this.index) : maskRows(h, idx, this.index)!
      assignGrid(h, this, rows, this.data.map((_, k) => k), v)
      return
    }
    if (idx.k === 'list' || idx.k === 'array' || (idx.k === 'obj' && idx.o instanceof IndexObj)) {
      const labels = cellsOf(h, idx)
      if (v.k === 'obj' && v.o instanceof Frame) {
        const o = v.o
        labels.forEach((l, j) => this.assignColumn(h, l, py.obj(o.data[j]), false))
      } else if (v.k === 'array' && v.a.ndim === 2) {
        const [r, c] = v.a.shape
        const vals = v.a.values()
        labels.forEach((l, j) => this.assignColumn(h, l, py.arr(NDArray.create(Array.from({ length: r }, (_, i) => vals[i * c + j]), [r], v.a.dtype)), false))
      } else if ((v.k === 'list' || v.k === 'tuple') && v.items.length && v.items.every(isListLike)) {
        const grid = v.items.map((x) => h.iterate(x))
        if (grid.length !== this.nrows || grid.some((r) => r.length !== labels.length)) throw h.err('ValueError', 'Columns must be same length as key')
        labels.forEach((l, j) => this.assignColumn(h, l, py.list(grid.map((r) => r[j])), false))
      } else if ((v.k === 'list' || v.k === 'tuple') && labels.length) {
        if (v.items.length !== labels.length) throw h.err('ValueError', 'Columns must be same length as key')
        labels.forEach((l, j) => this.assignColumn(h, l, v.items[j], false))
      } else labels.forEach((l) => this.assignColumn(h, l, v, false))
      return
    }
    this.assignColumn(h, cellOf(h, idx), v, true)
  }
  assignColumn(h: Host, name: Cell, v: Value, trace: boolean): void {
    let s: Series
    const adoptIndex = !this.ncols && !this.nrows
    if (v.k === 'obj' && v.o instanceof Series) {
      const src = v.o
      if (adoptIndex) this.index = src.index
      const t = reindexCells(src, this.index)
      s = new Series(t.dt === src.dtype && t.cells === src.values ? [...t.cells] : t.cells, this.index, name, t.dt)
      if (t.dt === src.dtype) {
        s.cats = src.cats
        s.width = src.width
      }
    } else if (v.k === 'obj' && v.o instanceof Frame) {
      if (v.o.ncols !== 1) throw h.err('ValueError', `Cannot set a DataFrame with multiple columns to the single column ${labelStr(name)}`)
      return this.assignColumn(h, name, py.obj(v.o.data[0]), trace)
    } else if (v.k === 'array' && v.a.ndim === 2 && v.a.shape[1] !== 1) {
      throw h.err('ValueError', `Expected a 1D array, got an array with shape (${v.a.shape.join(', ')})`)
    } else if ((isListLike(v) && !(v.k === 'array' && v.a.ndim === 0)) || !(v.k === 'str' || v.k === 'none' || isNum(v) || v.k === 'dict' || v.k === 'array' || v.k === 'obj' || v.k === 'func' || v.k === 'fn')) {
      const t = v.k === 'array' && v.a.ndim === 2 ? { cells: v.a.values() as Cell[], dt: arrayDT(v.a) } : typedCells(h, v)
      if (adoptIndex) this.index = IndexObj.range(t.cells.length)
      if (t.cells.length !== this.nrows) throw h.err('ValueError', `Length of values (${t.cells.length}) does not match length of index (${this.nrows})`)
      s = new Series(t.cells, this.index, name, t.dt)
      if ('width' in t) s.width = t.width ?? null
    } else if (v.k === 'array' && (v.a.ndim === 0 || v.a.size === 1)) {
      const c = cellOf(h, v)
      s = new Series(this.index.labels.map(() => c), this.index, name, arrayDT(v.a))
    } else {
      const t = inferValues(h, this.index.labels.map(() => v))
      s = new Series(t.cells, this.index, name, this.nrows ? t.dt : v.k === 'int' ? 'int64' : v.k === 'float' ? 'float64' : v.k === 'bool' ? 'bool' : 'object')
    }
    const k = this.columns.locate(name)
    if (!k.length) {
      this.columns = new IndexObj([...this.columns.labels], this.columns.name, this.ncols ? this.columns.dt : undefined)
      this.columns = new IndexObj([...this.columns.labels, name], this.columns.name, this.ncols ? commonDT([this.columns.dt, inferDT([name])]) : inferDT([name]))
      this.data.push(s)
      if (adoptIndex) this.data = this.data.map((x, i) => this.member(x, i))
    } else this.data[k[0]] = s
    if (trace) traceFrame(h, 'assign', () => ({ inputs: [], result: tableOfFrame(this), picks: { rows: null, cols: this.columns.locate(name) } }))
  }
  binop(op: string, other: Value, reflected: boolean, h: Host): Value | undefined {
    if (!OP_DUNDER[op] && !CMP.includes(op) && !LOGIC.includes(op)) return undefined
    if (other.k === 'obj' && !(other.o instanceof Series || other.o instanceof Frame || other.o instanceof IndexObj) && !other.o.toArray) return undefined
    const out = quietly(h, () => frameOp(h, this, op, other, reflected))
    traceFrame(h, CMP.includes(op) ? 'compare' : 'arith', () => ({ inputs: [{ label: h.nameOf(py.obj(this), 'df'), table: tableOfFrame(this) }, ...(other.k === 'obj' && other.o instanceof Series ? [{ label: h.nameOf(other, 'other'), table: tableOfSeries(other.o) }] : [])], result: tableOfFrame(out) }))
    return py.obj(out)
  }
  unary(op: '-' | '+' | '~', h: Host): Value {
    return py.obj(this.same(this.data.map((s) => seriesIn(s.unary(op, h)))))
  }
  npCall(name: string, args: Value[], kw: Kw, h: Host, impl: (a: Value[], k: Kw) => Value) { return frameNp(this, name, args, kw, h, impl) }
  getAttr(name: string, h: Host): Value | undefined {
    const m = frameAttr(this, name, h)
    if (m) return m
    const ex = this.extra?.get(name)
    if (ex) return ex
    const at = this.columns.locate(name)
    if (at.length === 1) return py.obj(this.data[at[0]])
    if (at.length > 1) return py.obj(this.take(null, at))
    return undefined
  }
  setAttr(name: string, v: Value, h: Host): boolean {
    if (name === 'columns') {
      this.columns = axisFrom(h, v, this.ncols, 'Length mismatch: Expected axis has')
      this.data = this.data.map((s, k) => this.member(s, k))
      return true
    }
    if (name === 'index') {
      const ix = axisFrom(h, v, this.nrows, 'Length mismatch: Expected axis has')
      this.index = ix
      this.data = this.data.map((s, k) => this.member(s, k))
      return true
    }
    if (this.columns.has(name)) {
      this.assignColumn(h, name, v, false)
      return true
    }
    ;(this.extra ??= new Map()).set(name, v)
    return true
  }
}

/** the sized dtype columns have in common, if they all have the same one */
export const sharedWidth = (cols: Series[]): string | null => (cols.length && cols.every((s) => s.width === cols[0].width) ? cols[0].width : null)

/** store a value over a rectangle of a frame: a scalar, a row, a column, a grid, or a labelled Series / DataFrame */
export const assignGrid = (h: Host, f: Frame, rows: number[], cols: number[], v: Value) => {
  if (!rows.length || !cols.length) return
  const fill = (k: number, cells: Cell[]) => f.data[k].store(rows, cells)
  if (v.k === 'obj' && v.o instanceof Frame) {
    const o = v.o
    cols.forEach((k) => {
      const src = o.col(f.columns.labels[k])
      fill(k, rows.map((r) => {
        if (!src) return NaN
        const at = o.index.locate(f.index.labels[r])
        return at.length ? src.values[at[0]] : NaN
      }))
    })
    return
  }
  if (v.k === 'obj' && v.o instanceof Series) {
    const o = v.o
    if (cols.length === 1) {
      fill(cols[0], rows.map((r) => {
        const at = o.index.locate(f.index.labels[r])
        return at.length ? o.values[at[0]] : NaN
      }))
      return
    }
    cols.forEach((k) => {
      const at = o.index.locate(f.columns.labels[k])
      const c = at.length ? o.values[at[0]] : NaN
      fill(k, rows.map(() => c))
    })
    return
  }
  if (v.k === 'array' && v.a.ndim === 2) {
    const [r, c] = v.a.shape
    const vals = v.a.values()
    if (r !== rows.length || c !== cols.length) throw h.err('ValueError', `shape mismatch: value array of shape (${r},${c}) could not be broadcast to indexing result of shape (${rows.length},${cols.length})`)
    cols.forEach((k, j) => fill(k, rows.map((_, i) => vals[i * c + j])))
    return
  }
  if (isListLike(v)) {
    const items = v.k === 'list' || v.k === 'tuple' ? v.items : null
    if (items && items.length && items.every((x) => isListLike(x))) {
      const grid = items.map((x) => cellsOf(h, x))
      if (grid.length !== rows.length || grid.some((row) => row.length !== cols.length)) throw h.err('ValueError', 'cannot set using a list-like indexer with a different length than the value')
      cols.forEach((k, j) => fill(k, rows.map((_, i) => grid[i][j])))
      return
    }
    const cells = cellsOf(h, v)
    if (cols.length === 1 && cells.length === rows.length) fill(cols[0], cells)
    else if (rows.length === 1 && cells.length === cols.length) cols.forEach((k, j) => fill(k, [cells[j]]))
    else if (cells.length === cols.length) cols.forEach((k, j) => fill(k, rows.map(() => cells[j])))
    else throw h.err('ValueError', 'cannot set using a list-like indexer with a different length than the value')
    return
  }
  const c = cellOf(h, v)
  cols.forEach((k) => fill(k, rows.map(() => c)))
}

const typedItem = (h: Host, v: Value, i: number): Value => h.iterate(v)[i]

class FrameIndexer extends PyObj {
  readonly cls: string
  f: Frame
  kind: string
  constructor(f: Frame, kind: string) {
    super()
    this.f = f
    this.kind = kind
    this.cls = kind === 'loc' ? '_LocIndexer' : kind === 'iloc' ? '_iLocIndexer' : kind === 'at' ? '_AtIndexer' : '_iAtIndexer'
  }
  get positional() { return this.kind === 'iloc' || this.kind === 'iat' }
  repr() { return `<pandas.core.indexing.${this.cls} object>` }
  /** rows by label: a number for a single row, positions otherwise */
  private rowSel(h: Host, sel: Value): number[] | number {
    const f = this.f
    if (sel.k === 'func' || sel.k === 'fn') return this.rowSel(h, quietly(h, () => h.call(sel, [py.obj(f)])))
    if (sel.k === 'slice') return labelSlice(h, sel, f.index)
    const m = maskRows(h, sel, f.index)
    if (m) return m
    if (sel.k === 'tuple' && f.index.multi) {
      const at = f.index.locate(cellOf(h, sel))
      if (at.length === 1) return at[0]
      const rows = multiPrefix(f.index, sel.items.map((x) => cellOf(h, x)))
      if (!rows.length) throw h.err('KeyError', repr(sel))
      return rows
    }
    if (isListLike(sel)) {
      const t = typedCells(h, sel)
      if (f.index.multi) {
        const rows = t.cells.flatMap((c) => (f.index.has(c) ? f.index.locate(c) : multiPrefix(f.index, [c])))
        if (!rows.length) throw h.err('KeyError', `"None of [${new IndexObj(t.cells, null, t.dt).repr()}] are in the [index]"`)
        return rows
      }
      return positionsOfLabels(h, t.cells, t.dt, f.index, 'index')
    }
    const c = cellOf(h, sel)
    const at = f.index.locate(c)
    if (at.length === 1) return at[0]
    if (at.length) return at
    if (f.index.multi) {
      const rows = multiPrefix(f.index, [c])
      if (rows.length) return rows
    }
    throw h.err('KeyError', repr(sel))
  }
  private colSel(h: Host, sel: Value): number[] | number {
    const f = this.f
    if (sel.k === 'func' || sel.k === 'fn') return this.colSel(h, quietly(h, () => h.call(sel, [py.obj(f)])))
    if (sel.k === 'slice') return labelSlice(h, sel, f.columns)
    const m = maskRows(h, sel, f.columns)
    if (m) return m
    if (isListLike(sel) && !(sel.k === 'tuple' && f.columns.multi)) {
      const t = typedCells(h, sel)
      return positionsOfLabels(h, t.cells, t.dt, f.columns, 'columns')
    }
    const c = cellOf(h, sel)
    const at = f.columns.locate(c)
    if (at.length === 1) return at[0]
    if (at.length) return at
    if (f.columns.multi) {
      const cols = multiPrefix(f.columns, sel.k === 'tuple' ? sel.items.map((x) => cellOf(h, x)) : [c])
      if (cols.length) return cols
    }
    throw h.err('KeyError', repr(sel))
  }
  resolve(h: Host, idx: Value): { rows: number[] | number; cols: number[] | number; rowDrop: number; colDrop: number } {
    const f = this.f
    if (this.positional) {
      const [r, c] = idx.k === 'tuple' ? [idx.items[0], idx.items[1] ?? null] : [idx, null]
      if (idx.k === 'tuple' && idx.items.length > 2) throw h.err('IndexingError', 'Too many indexers')
      const res = (v: Value, n: number): number[] | number => {
        if (v.k === 'func' || v.k === 'fn') return res(quietly(h, () => h.call(v, [py.obj(f)])), n)
        const oob = c ? `index ${v.k === 'int' ? v.v : ''} is out of bounds for axis 0 with size ${n}` : undefined
        return v.k === 'int' || (v.k === 'array' && v.a.ndim === 0) ? positions(h, v, n, oob)[0] : positions(h, v, n)
      }
      return { rows: res(r, f.nrows), cols: c ? res(c, f.ncols) : f.data.map((_, k) => k), rowDrop: 0, colDrop: 0 }
    }
    if (idx.k === 'tuple' && f.index.multi && !f.columns.has(cellOf(h, idx.items[1] ?? py.NONE)) && idx.items.every((x) => x.k !== 'slice' && x.k !== 'list') && idx.items.length <= f.index.nlevels) {
      const rows = this.rowSel(h, idx)
      return { rows, cols: f.data.map((_, k) => k), rowDrop: typeof rows === 'number' ? 0 : idx.items.length, colDrop: 0 }
    }
    if (idx.k === 'tuple' && idx.items.length === 2) {
      const r = this.rowSel(h, idx.items[0])
      const multiRow = f.index.multi && typeof r !== 'number' && idx.items[0].k !== 'slice' && !isListLike(idx.items[0]) && !(idx.items[0].k === 'obj')
      const multiCol = f.columns.multi && idx.items[1].k !== 'slice' && !isListLike(idx.items[1])
      const c = this.colSel(h, idx.items[1])
      return { rows: r, cols: c, rowDrop: multiRow ? 1 : 0, colDrop: multiCol && typeof c !== 'number' ? 1 : 0 }
    }
    if (idx.k === 'tuple' && idx.items.length > 2) throw h.err('IndexingError', 'Too many indexers')
    const rows = this.rowSel(h, idx)
    const partial = f.index.multi && typeof rows !== 'number' && idx.k !== 'slice' && !isListLike(idx) && !(idx.k === 'obj' && idx.o instanceof Series)
    return { rows, cols: f.data.map((_, k) => k), rowDrop: partial ? 1 : 0, colDrop: 0 }
  }
  getItem(idx: Value, h: Host): Value {
    const { rows, cols, rowDrop, colDrop } = this.resolve(h, idx)
    const f = this.f
    const label = h.nameOf(py.obj(f), 'df')
    const picks = { rows: typeof rows === 'number' ? [rows] : rows, cols: typeof cols === 'number' ? [cols] : cols }
    const sliced = idx.k === 'slice' || (idx.k === 'tuple' && idx.items[0]?.k === 'slice')
    const allCols = typeof cols !== 'number' && cols.length === f.ncols && cols.every((k, i) => k === i)
    let out: Value
    if (typeof rows === 'number' && typeof cols === 'number') out = valueOf(f.data[cols].values[rows], f.data[cols].dtype)
    else if (typeof rows === 'number') out = py.obj(allCols ? f.row(rows) : f.row(rows).pick(cols as number[], true))
    else if (typeof cols === 'number') {
      const s = f.data[cols].pick(rows, sliced)
      if (rowDrop) s.index = dropLevels(f.index, rows, rowDrop)
      out = py.obj(s)
    } else {
      const t = f.take(rows, allCols ? null : cols, sliced)
      if (rowDrop) t.index = dropLevels(f.index, rows, rowDrop)
      const fixed = colDrop ? new Frame(dropLevels(f.columns, cols, colDrop), t.data, t.index) : rowDrop ? new Frame(t.columns, t.data, t.index) : t
      out = py.obj(fixed)
    }
    traceFrame(h, this.positional ? 'iloc' : 'loc', () => ({ inputs: [{ label, table: tableOfFrame(f) }], result: tableOfValue(out), picks, resultText: repr(out) }))
    return out
  }
  setItem(idx: Value, v: Value, h: Host) {
    const f = this.f
    const created = new Set<Cell>()
    if (!this.positional) {
      const [r, c] = idx.k === 'tuple' && idx.items.length === 2 && !f.index.multi ? [idx.items[0], idx.items[1]] : [idx, null]
      // a new column label creates the column, filled with NaN elsewhere
      if (c && (c.k === 'str' || c.k === 'int') && !f.columns.has(cellOf(h, c))) {
        created.add(cellOf(h, c))
        f.assignColumn(h, cellOf(h, c), py.float(NaN), false)
      } else if (c && c.k === 'list') {
        for (const x of c.items) {
          if (!f.columns.has(cellOf(h, x))) f.assignColumn(h, cellOf(h, x), py.float(NaN), false)
        }
      }
      // a new row label appends a row: with its values if the whole row is given, NaN elsewhere if not
      if (r.k !== 'slice' && !isListLike(r) && !(r.k === 'func' || r.k === 'fn') && !f.index.has(cellOf(h, r)) && !f.index.multi) {
        const lab = cellOf(h, r)
        const row: Cell[] | null = c ? null : v.k === 'obj' && v.o instanceof Series ? f.columns.labels.map((l) => {
          const at = (v.o as Series).index.locate(l)
          return at.length ? (v.o as Series).values[at[0]] : NaN
        }) : isListLike(v) ? cellsOf(h, v) : f.columns.labels.map(() => cellOf(h, v))
        if (row && row.length !== f.ncols) throw h.err('ValueError', 'cannot set a row with mismatched columns')
        f.index = new IndexObj([...f.index.labels, lab], f.index.name, f.nrows ? commonDT([f.index.dt, inferDT([lab])]) : inferDT([lab]))
        f.data.forEach((s, k) => {
          s.grow(lab, row ? row[k] : NaN)
          s.index = f.index
        })
        if (row) return
      }
    }
    const { rows, cols } = this.resolve(h, idx)
    const rs = typeof rows === 'number' ? [rows] : rows
    const cs = typeof cols === 'number' ? [cols] : cols
    assignGrid(h, f, rs, cs, v)
    // a brand-new column filled in whole takes the dtype of what it now holds, not the NaN it began as
    for (const k of cs) {
      if (rs.length === f.nrows && created.has(f.columns.labels[k])) {
        const x = f.data[k]
        const t = inferValues(h, rs.map((_, i) => (isListLike(v) ? typedItem(h, v, i) : v)))
        x.values = t.cells
        x.dtype = t.dt
      }
    }
  }
}

export const frameIndexer = (f: Frame, kind: string) => new FrameIndexer(f, kind)

// ---------------------------------------------------------------- tables for the step view

const cellsForTable = (cells: Cell[], dt: DT): (string | null)[] => {
  const txt = formatArray(cells, dt, false)
  return cells.map((c, i) => (isMissing(c) ? null : txt[i]))
}

export const tableOfFrame = (f: Frame): TableSpec => {
  const cols = f.data.map((s) => cellsForTable(s.values, s.dtype))
  return {
    columns: f.columns.labels.map(labelStr),
    index: f.index.labels.map(labelStr),
    cells: f.index.labels.map((_, r) => cols.map((c) => c[r])),
    dtypes: f.data.map((s) => dtName(s)),
    series: false,
    name: null,
    indexName: f.index.name === null ? null : labelStr(f.index.name),
  }
}

export const tableOfSeries = (s: Series): TableSpec => {
  const all = cellsForTable(s.values, s.dtype)
  return {
    columns: [],
    index: s.index.labels.map(labelStr),
    cells: all.map((c) => [c]),
    dtypes: [dtName(s)],
    series: true,
    name: s.name === null ? '' : labelStr(s.name),
    indexName: s.index.name === null ? null : labelStr(s.index.name),
  }
}


// ---------------------------------------------------------------- the module

export const PANDAS: PyLib = {
  modules: ['pandas'],
  load: (h: Host) => {
    Object.assign(OPTIONS, DEFAULT_OPTIONS)
    return pandasModule(h)
  },
}
