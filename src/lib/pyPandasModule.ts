/** The names in `pandas` itself: constructors, readers and the module-level functions. */
import { PyDict, PyObj, isNum, py } from './minipy'
import type { Host, Kw, Value } from './minipy'
import { NDArray } from './ndarray'
import { Frame, IndexObj, ObjArray, Series, cellOf, cellsOf, inferDT, inferValues, isListLike, isMissing, reindexCells, tableOfFrame, tableOfSeries, traceFrame, typedCells, unionIndex, valueOf, seriesIn } from './pyPandas'
import type { Cell, DT, Params } from './pyPandas'
import { DEFAULT_OPTIONS, OPTIONS } from './pyPandasFormat'
import { jsonNormalize, readCsv, readJson, textFrom } from './pyPandasIO'
import { arrayOf, astypeCol, categoricalOf, factorize, flag, given, need, typeArg, valueCountsOf } from './pyPandasMethods'
import { concatObjects, crosstab, cutSeries, getDummies, meltFrame, mergeFrames, pivotFrame, pivotTable, qcutSeries } from './pyPandasReshape'

const fromDictOfColumns = (h: Host, d: PyDict, colsArg: IndexObj | null, indexArg: IndexObj | null, dtype: Value | undefined): Frame => {
  const entries = d.items()
  const cols = entries.map(([k]) => cellOf(h, k))
  const isScalar = (v: Value) => isNum(v) || v.k === 'str' || v.k === 'none'
  const isNested = (v: Value) => v.k === 'dict'
  const structured = entries.filter(([, v]) => !isScalar(v))
  if (!structured.length && !indexArg && entries.length) throw h.err('ValueError', 'If using all scalar values, you must pass an index')
  // Series align on their labels; everything else takes position
  const labelled = entries.filter(([, v]) => (v.k === 'obj' && v.o instanceof Series) || isNested(v)).map(([, v]) => (v.k === 'dict' ? seriesFromDict(h, v.d, null) : seriesIn(v)))
  let index = indexArg
  if (!index && labelled.length) {
    index = labelled[0].index
    for (const s of labelled.slice(1)) index = unionIndex(index, s.index)
  }
  const data = entries.map(([, v], i) => {
    if (v.k === 'dict') v = py.obj(seriesFromDict(h, v.d, null))
    if (v.k === 'obj' && v.o instanceof Series) {
      const t = reindexCells(v.o, index!)
      const s = new Series(t.cells === v.o.values ? [...t.cells] : t.cells, null, cols[i], t.dt)
      if (t.dt === v.o.dtype) s.cats = v.o.cats
      return s
    }
    if (isScalar(v)) return null
    if (v.k === 'array' && v.a.ndim === 2) throw h.err('ValueError', 'Per-column arrays must each be 1-dimensional')
    const t = typedCells(h, v)
    const col = new Series(t.cells, null, cols[i], t.dt)
    col.width = t.width ?? null
    return col
  })
  const lens = new Set(data.filter((s): s is Series => !!s && !labelled.length).map((s) => s.length))
  const n = index ? index.length : data.find((s) => s)?.length ?? 0
  if (lens.size > 1) throw h.err('ValueError', 'All arrays must be of the same length')
  for (const s of data) if (s && s.length !== n) throw h.err('ValueError', index ? `Length of values (${s.length}) does not match length of index (${n})` : 'All arrays must be of the same length')
  const filled = data.map((s, i) => {
    if (s) return s
    const t = inferValues(h, Array.from({ length: n }, () => entries[i][1]))
    return new Series(t.cells, null, cols[i], t.dt)
  })
  const ix = index ?? IndexObj.range(n)
  let out = new Frame(new IndexObj(cols, null, cols.length ? inferDT(cols) : 'object'), filled, ix)
  if (colsArg) out = selectColumns(out, colsArg)
  return applyDtype(h, out, dtype)
}

const selectColumns = (f: Frame, cols: IndexObj): Frame => new Frame(cols, cols.labels.map((c) => {
  const s = f.col(c)
  return s ?? new Series(f.index.labels.map(() => NaN), null, c, 'object')
}), f.index)

const applyDtype = (h: Host, f: Frame, dtype: Value | undefined): Frame => (given(dtype) ? f.same(f.data.map((s) => astypeCol(h, s, typeArg(h, dtype!)))) : f)

const seriesFromDict = (h: Host, d: PyDict, index: IndexObj | null): Series => {
  const t = inferValues(h, d.values())
  const keys = d.keys().map((k) => cellOf(h, k))
  const s = new Series(t.cells, new IndexObj(keys, null, keys.length ? inferDT(keys) : 'object'), null, t.dt)
  if (!index) return s
  const r = reindexCells(s, index)
  return new Series(r.cells, index, null, r.dt)
}

const toIndexArg = (h: Host, v: Value | undefined): IndexObj | null => {
  if (!given(v)) return null
  if (v!.k === 'obj' && v!.o instanceof IndexObj) return v!.o
  const t = typedCells(h, v!)
  return new IndexObj(t.cells, null, t.dt)
}

/** pd.DataFrame(...) */
export const frameFrom = (h: Host, data: Value | undefined, kw: Kw): Frame => {
  for (const k of Object.keys(kw)) if (!['data', 'index', 'columns', 'dtype', 'copy'].includes(k)) throw h.err('TypeError', `DataFrame.__init__() got an unexpected keyword argument '${k}'`)
  const colsArg = toIndexArg(h, kw.columns)
  const indexArg = toIndexArg(h, kw.index)
  const dtype = kw.dtype
  if (!given(data)) {
    const cols = colsArg ?? new IndexObj([], null, 'object')
    const ix = indexArg ?? IndexObj.range(0)
    return new Frame(cols, cols.labels.map((c) => new Series(ix.labels.map(() => NaN), null, c, ix.length ? 'float64' : 'object')), ix)
  }
  const d = data!
  if (d.k === 'dict') return fromDictOfColumns(h, d.d, colsArg, indexArg, dtype)
  if (d.k === 'obj' && d.o instanceof Frame) {
    let out = d.o.copy()
    if (indexArg) out = new Frame(out.columns, out.data.map((s) => {
      const t = reindexCells(s, indexArg)
      return new Series(t.cells, null, s.name, t.dt)
    }), indexArg)
    if (colsArg) out = selectColumns(out, colsArg)
    return applyDtype(h, out, dtype)
  }
  if (d.k === 'obj' && d.o instanceof Series) {
    const s = d.o
    const name = colsArg ? colsArg.labels[0] : s.name ?? 0
    const t = indexArg ? reindexCells(s, indexArg) : { cells: [...s.values], dt: s.dtype }
    return applyDtype(h, new Frame(colsArg ?? (s.name === null ? IndexObj.range(1) : new IndexObj([name])), [new Series(t.cells, null, name, t.dt)], indexArg ?? s.index), dtype)
  }
  if (d.k === 'array' && d.a.ndim === 2) {
    const [r, c] = d.a.shape
    const v = d.a.values()
    const cols = colsArg ?? IndexObj.range(c)
    if (cols.length !== c) throw h.err('ValueError', `Shape of passed values is (${r}, ${c}), indices imply (${r}, ${cols.length})`)
    if (indexArg && indexArg.length !== r) throw h.err('ValueError', `Shape of passed values is (${r}, ${c}), indices imply (${indexArg.length}, ${c})`)
    const dt: DT = d.a.dtype === 'bool' ? 'bool' : d.a.dtype === 'float64' ? 'float64' : 'int64'
    const width = d.a.dtype === 'uint8' ? 'uint8' : null
    return applyDtype(h, new Frame(cols, Array.from({ length: c }, (_, j) => {
      const col = new Series(Array.from({ length: r }, (_i, i) => (dt === 'bool' ? v[i * c + j] !== 0 : v[i * c + j])), null, cols.labels[j], dt)
      col.width = width
      return col
    }), indexArg ?? IndexObj.range(r)), dtype)
  }
  if (d.k === 'obj' && d.o instanceof ObjArray && d.o.shape.length === 2) {
    const [r, c] = d.o.shape
    const cells = d.o.cells
    const cols = colsArg ?? IndexObj.range(c)
    return applyDtype(h, new Frame(cols, Array.from({ length: c }, (_, j) => new Series(Array.from({ length: r }, (_i, i) => cells[i * c + j]), null, cols.labels[j], 'object')), indexArg ?? IndexObj.range(r)), dtype)
  }
  if (d.k === 'obj' && d.o instanceof IndexObj) {
    const cols = colsArg ?? IndexObj.range(1)
    return new Frame(cols, [new Series([...d.o.labels], null, cols.labels[0], d.o.dt)], indexArg ?? IndexObj.range(d.o.length))
  }
  if (d.k === 'array' || d.k === 'list' || d.k === 'tuple' || d.k === 'range' || d.k === 'obj' || d.k === 'iter') {
    if (d.k === 'obj' && !d.o.iter && !d.o.toArray) throw h.err('ValueError', 'DataFrame constructor not properly called!')
    const rows = d.k === 'obj' && d.o.toArray && d.o.toArray().ndim === 2 ? null : h.iterate(d)
    if (!rows) return frameFrom(h, py.arr((d as { o: PyObj }).o.toArray!()), kw)
    if (!rows.length) {
      const cols = colsArg ?? IndexObj.range(0)
      const ix = indexArg ?? IndexObj.range(0)
      return new Frame(cols, cols.labels.map((c) => new Series([], null, c, 'object')), ix)
    }
    if (rows.every((r) => r.k === 'dict')) {
      const labels: Cell[] = []
      for (const r of rows) for (const k of (r as { d: PyDict }).d.keys()) if (!labels.includes(cellOf(h, k))) labels.push(cellOf(h, k))
      const pick = colsArg ?? new IndexObj(labels, null, inferDT(labels))
      return applyDtype(h, new Frame(pick, pick.labels.map((c) => {
        const t = inferValues(h, rows.map((r) => (r as { d: PyDict }).d.get(valueOf(c, 'object')) ?? py.float(NaN)))
        return new Series(t.cells, null, c, t.dt)
      }), indexArg ?? IndexObj.range(rows.length)), dtype)
    }
    if (rows.every((r) => r.k === 'obj' && r.o instanceof Series)) {
      const ss = rows.map((r) => seriesIn(r))
      let labels = ss[0].index
      for (const s of ss.slice(1)) labels = unionIndex(labels, s.index)
      const pick = colsArg ?? labels
      const ix = indexArg ?? (ss.every((s) => s.name !== null) ? new IndexObj(ss.map((s) => s.name)) : IndexObj.range(ss.length))
      return applyDtype(h, new Frame(pick, pick.labels.map((c) => {
        const t = inferValues(h, ss.map((s) => {
          const at = s.index.locate(c)
          return at.length ? valueOf(s.values[at[0]], s.dtype) : py.float(NaN)
        }))
        return new Series(t.cells, null, c, t.dt)
      }), ix), dtype)
    }
    const first = rows[0]
    if (!colsArg && first.k === 'tuple' && first.names) return frameFrom(h, py.list(rows), { ...kw, columns: py.list(first.names.map(py.str)) })
    const nested = rows.some((r) => r.k === 'list' || r.k === 'tuple' || r.k === 'array' || r.k === 'range')
    if (!nested) {
      const t = inferValues(h, rows)
      const cols = colsArg ?? IndexObj.range(1)
      if (cols.length !== 1) throw h.err('ValueError', `Shape of passed values is (${rows.length}, 1), indices imply (${rows.length}, ${cols.length})`)
      return applyDtype(h, new Frame(cols, [new Series(t.cells, null, cols.labels[0], t.dt)], indexArg ?? IndexObj.range(rows.length)), dtype)
    }
    const grid = rows.map((r) => (r.k === 'list' || r.k === 'tuple' || r.k === 'array' || r.k === 'range' ? h.iterate(r) : [r]))
    const width = Math.max(...grid.map((r) => r.length))
    const cols = colsArg ?? IndexObj.range(width)
    if (cols.length !== width) {
      if (grid.every((r) => r.length === width)) throw h.err('ValueError', `${cols.length} columns passed, passed data had ${width} columns`)
      throw h.err('ValueError', `${cols.length} columns passed, passed data had ${width} columns`)
    }
    if (indexArg && indexArg.length !== grid.length) throw h.err('ValueError', `Shape of passed values is (${grid.length}, ${width}), indices imply (${indexArg.length}, ${width})`)
    return applyDtype(h, new Frame(cols, cols.labels.map((c, j) => {
      const t = inferValues(h, grid.map((r) => r[j] ?? py.NONE))
      return new Series(t.cells, null, c, t.dt)
    }), indexArg ?? IndexObj.range(grid.length)), dtype)
  }
  throw h.err('ValueError', 'DataFrame constructor not properly called!')
}

const seriesFrom = (h: Host, d: Value | undefined, kw: Kw): Series => {
  for (const k of Object.keys(kw)) if (!['data', 'index', 'dtype', 'name', 'copy'].includes(k)) throw h.err('TypeError', `Series.__init__() got an unexpected keyword argument '${k}'`)
  const index = toIndexArg(h, kw.index)
  const name = given(kw.name) ? cellOf(h, kw.name) : null
  let out: Series
  if (!given(d)) {
    out = new Series(index ? index.labels.map(() => NaN) : [], index, name, index ? 'float64' : 'object')
  } else if (d!.k === 'dict') {
    out = seriesFromDict(h, d!.d, index)
    out.name = name
  } else if (d!.k === 'obj' && d!.o instanceof Series) {
    const src = d!.o
    const t = index ? reindexCells(src, index) : { cells: [...src.values], dt: src.dtype }
    out = new Series(t.cells, index ?? src.index, name ?? src.name, t.dt)
    if (t.dt === src.dtype) out.cats = src.cats
  } else if (d!.k === 'array' && d!.a.ndim !== 1) {
    if (d!.a.ndim === 0 || d!.a.size === 1 && d!.a.ndim === 0) {
      const c = cellOf(h, d!)
      out = new Series(index ? index.labels.map(() => c) : [c], index, name)
    } else throw h.err('ValueError', 'Data must be 1-dimensional')
  } else if (isNum(d!) || d!.k === 'str' || d!.k === 'none') {
    const n = index ? index.length : 1
    const t = inferValues(h, Array.from({ length: n }, () => d!))
    out = new Series(t.cells, index, name, t.dt)
  } else {
    const t = typedCells(h, d!)
    if (index && index.length !== t.cells.length) throw h.err('ValueError', `Length of values (${t.cells.length}) does not match length of index (${index.length})`)
    out = new Series(t.cells, index, name, t.dt)
    out.width = t.width ?? null
  }
  if (given(kw.dtype)) return astypeCol(h, out, typeArg(h, kw.dtype))
  return out
}

/** pd.isna and friends on whatever they are given */
const missingOf = (h: Host, v: Value, not: boolean): Value => {
  const f = (c: Cell) => (not ? !isMissing(c) : isMissing(c))
  if (v.k === 'obj' && v.o instanceof Frame) return py.obj(v.o.same(v.o.data.map((s) => s.with(s.values.map(f), 'bool'))))
  if (v.k === 'obj' && v.o instanceof Series) return py.obj(v.o.with(v.o.values.map(f), 'bool'))
  if (isListLike(v) || v.k === 'array') {
    const cells = cellsOf(h, v)
    return py.arr(NDArray.create(cells.map((c) => (f(c) ? 1 : 0)), [cells.length], 'bool'))
  }
  return py.bool(f(cellOf(h, v)))
}

class OptionsNode extends PyObj {
  readonly cls = 'DictWrapper'
  prefix: string
  constructor(prefix: string) {
    super()
    this.prefix = prefix
  }
  repr() { return `<pandas options: ${this.prefix}>` }
  getAttr(name: string): Value | undefined {
    if (this.prefix === '' && name === 'display') return py.obj(new OptionsNode('display'))
    if (this.prefix === 'display' && name in OPTIONS) return optionValue(name as keyof typeof OPTIONS)
    return undefined
  }
  setAttr(name: string, v: Value, h: Host): boolean {
    if (this.prefix !== 'display') return false
    setOption(h, `display.${name}`, v)
    return true
  }
}

const optionValue = (k: keyof typeof OPTIONS): Value => {
  const x = OPTIONS[k]
  return x === null ? py.NONE : typeof x === 'number' ? py.int(x) : typeof x === 'boolean' ? py.bool(x) : py.str(x)
}

const optKey = (h: Host, full: string): keyof typeof OPTIONS => {
  // these short names also match styler.* options in pandas
  if (['max_rows', 'max_columns', 'precision'].includes(full)) throw h.err('OptionError', "'Pattern matched multiple keys'")
  const k = full.replace(/^display\./, '')
  if (!(k in OPTIONS)) {
    throw h.err(/^(display\.)?(float_format|colheader_justify|html\.|latex\.|large_repr|memory_usage|unicode\.|date_|notebook_repr_html|max_info|pprint_nest_depth|chop_threshold|encoding|column_space)/.test(full) ? 'NotImplementedError' : 'OptionError', /^(display\.)?(float_format|colheader_justify|html\.|latex\.|large_repr|memory_usage|unicode\.|date_|notebook_repr_html|max_info|pprint_nest_depth|chop_threshold|encoding|column_space)/.test(full) ? `pd.set_option('${full}') is not available in this sandbox` : `"No such keys(s): '${full}'"`)
  }
  return k as keyof typeof OPTIONS
}

const setOption = (h: Host, name: string, v: Value) => {
  const k = optKey(h, name)
  const val: unknown = v.k === 'none' ? null : v.k === 'bool' ? v.v : v.k === 'str' ? v.v : h.num(v)
  ;(OPTIONS as Record<string, unknown>)[k] = val
}

/** with pd.option_context(pat, val, …): the options hold inside the block */
class OptionContext extends PyObj {
  readonly cls = 'option_context'
  readonly pairs: Value[]
  saved: typeof OPTIONS | null = null
  constructor(pairs: Value[]) {
    super()
    this.pairs = pairs
  }
  repr() { return '<pandas._config.config.option_context object>' }
  enter(h: Host) {
    this.saved = { ...OPTIONS }
    for (let i = 0; i < this.pairs.length; i += 2) setOption(h, h.str(this.pairs[i]), this.pairs[i + 1])
    return py.obj(this)
  }
  exit() {
    if (this.saved) Object.assign(OPTIONS, this.saved)
  }
}

export const pandasModule = (h: Host): Record<string, Value> => {
  const fn = (name: string, call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })
  const pd: Record<string, Value> = {
    __version__: py.str('2.3 (sandbox)'),
    DataFrame: {
      k: 'type', name: 'DataFrame', call: (args, kw) => {
        if (args.length > 5) throw h.err('TypeError', 'DataFrame.__init__() takes at most 5 arguments')
        const kw2 = { ...kw }
        if (args[1]) kw2.index = args[1]
        if (args[2]) kw2.columns = args[2]
        if (args[3]) kw2.dtype = args[3]
        return py.obj(frameFrom(h, args[0] ?? kw.data, kw2))
      },
      attrs: new Map(Object.entries({
        from_dict: fn('from_dict', (args, kw) => {
          const data = args[0] ?? kw.data
          if (!data || data.k !== 'dict') throw h.err('TypeError', 'from_dict takes a dict')
          const orient = given(args[1] ?? kw.orient) ? h.str(args[1] ?? kw.orient) : 'columns'
          const dtype = args[2] ?? kw.dtype
          const columns = args[3] ?? kw.columns
          if (orient === 'columns') {
            if (given(columns)) throw h.err('ValueError', "cannot use columns parameter with orient='columns'")
            return py.obj(frameFrom(h, data, given(dtype) ? { dtype } : {}))
          }
          if (orient === 'index') {
            const vals = data.d.values()
            const keys = py.list(data.d.keys())
            if (vals.length && vals.every((v) => v.k === 'dict')) {
              const f = frameFrom(h, py.list(vals), { index: keys, ...(given(dtype) ? { dtype } : {}) })
              return py.obj(given(columns) ? selectColumns(f, toIndexArg(h, columns)!) : f)
            }
            return py.obj(frameFrom(h, py.list(vals), { index: keys, ...(given(columns) ? { columns } : {}), ...(given(dtype) ? { dtype } : {}) }))
          }
          if (orient === 'tight') throw h.err('NotImplementedError', "from_dict(orient='tight') is not available in this sandbox")
          throw h.err('ValueError', `Expected 'index', 'columns' or 'tight' for orient parameter. Got '${orient}' instead`)
        }),
        from_records: fn('from_records', (args, kw) => {
          for (const k of Object.keys(kw)) if (!['data', 'index', 'exclude', 'columns', 'coerce_float', 'nrows'].includes(k)) throw h.err('NotImplementedError', `from_records(${k}=...) is not available in this sandbox`)
          const data = args[0] ?? kw.data
          let f = data.k === 'dict' ? frameFrom(h, data, {}) : frameFrom(h, data, given(kw.columns ?? args[3]) ? { columns: kw.columns ?? args[3] } : {})
          if (given(kw.nrows)) f = f.take(Array.from({ length: Math.min(h.toInt(kw.nrows), f.nrows) }, (_, i) => i), null, true)
          if (given(kw.exclude)) {
            const ex = cellsOf(h, kw.exclude)
            f = f.take(null, f.columns.labels.map((_, k) => k).filter((k) => !ex.includes(f.columns.labels[k])))
          }
          const ixV = given(kw.index ?? args[1])
          if (ixV) {
            if (ixV.k === 'str') {
              const k = f.colPos(h, ixV)
              const rest = f.columns.labels.map((_, j) => j).filter((j) => j !== k)
              f = new Frame(f.columns.pick(rest, true), rest.map((j) => f.data[j]), new IndexObj(f.data[k].values, f.columns.labels[k], f.data[k].dtype))
            } else f = new Frame(f.columns, f.data, toIndexArg(h, ixV))
          }
          return py.obj(f)
        }),
      })),
    },
    Series: { k: 'type', name: 'Series', call: (args, kw) => {
      const kw2 = { ...kw }
      if (args[1]) kw2.index = args[1]
      if (args[2]) kw2.dtype = args[2]
      if (args[3]) kw2.name = args[3]
      return py.obj(seriesFrom(h, args[0] ?? kw.data, kw2))
    } },
    Index: { k: 'type', name: 'Index', call: (args, kw) => {
      for (const k of Object.keys(kw)) if (!['data', 'dtype', 'copy', 'name', 'tupleize_cols'].includes(k)) throw h.err('TypeError', `Index() got an unexpected keyword argument '${k}'`)
      const d = args[0] ?? kw.data
      const name = given(kw.name ?? args[3]) ? cellOf(h, kw.name ?? args[3]) : null
      const t = d ? typedCells(h, d) : { cells: [], dt: 'object' as DT }
      if (!t.cells.length && !d) return py.obj(new IndexObj([], name, 'object'))
      const s = given(kw.dtype ?? args[1]) ? astypeCol(h, new Series(t.cells, null, null, t.dt), typeArg(h, kw.dtype ?? args[1])) : null
      return py.obj(new IndexObj(s ? s.values : t.cells, name ?? (d && d.k === 'obj' && (d.o instanceof Series || d.o instanceof IndexObj) ? d.o.name : null), s ? s.dtype : t.dt))
    } },
    RangeIndex: { k: 'type', name: 'RangeIndex', call: (args, kw) => {
      const a = args[0] ?? kw.start
      const b = args[1] ?? kw.stop
      const c = args[2] ?? kw.step
      const name = given(kw.name) ? cellOf(h, kw.name) : null
      const start = b ? h.toInt(a) : 0
      const stop = b ? h.toInt(b) : a ? h.toInt(a) : 0
      const step = c ? h.toInt(c) : 1
      return py.obj(IndexObj.range(Math.max(0, Math.ceil((stop - start) / step)), start, step, name))
    } },
    read_csv: fn('read_csv', (args, kw) => readerCall(h, args, kw, ',')),
    read_table: fn('read_table', (args, kw) => readerCall(h, args, kw, '\t')),
    read_json: fn('read_json', (args, kw) => {
      const src = args[0] ?? kw.path_or_buf
      if (!src) throw h.err('TypeError', "read_json() missing 1 required positional argument: 'path_or_buf'")
      return py.obj(readJson(h, src, kw))
    }),
    json_normalize: fn('json_normalize', (args, kw) => py.obj(jsonNormalize(h, args[0] ?? kw.data, kw))),
    concat: fn('concat', (args, kw) => concatObjects(h, args, kw)),
    get_dummies: fn('get_dummies', (args, kw) => {
      const out = getDummies(h, args, kw)
      const d = args[0] ?? kw.data
      traceFrame(h, 'get_dummies', () => ({ inputs: [{ label: h.nameOf(d, 'data'), table: d.k === 'obj' && d.o instanceof Series ? tableOfSeries(d.o) : d.k === 'obj' && d.o instanceof Frame ? tableOfFrame(d.o) : tableOfFrame(out) }], result: tableOfFrame(out) }))
      return py.obj(out)
    }),
    crosstab: fn('crosstab', (args, kw) => {
      const out = crosstab(h, args, kw)
      const a = args[0] ?? kw.index
      const b = args[1] ?? kw.columns
      traceFrame(h, 'crosstab', () => ({ inputs: [a, b].filter((x) => x.k === 'obj' && x.o instanceof Series).map((x) => ({ label: h.nameOf(x, 'x'), table: tableOfSeries(seriesIn(x)) })), result: tableOfFrame(out) }))
      return py.obj(out)
    }),
    cut: fn('cut', (args, kw) => cutSeries(h, args, kw)),
    qcut: fn('qcut', (args, kw) => qcutSeries(h, args, kw)),
    merge: fn('merge', (args, kw) => {
      const l = args[0] ?? kw.left
      const r = args[1] ?? kw.right
      if (!l || !r) throw h.err('TypeError', "merge() missing required arguments: 'left' and 'right'")
      if (!(l.k === 'obj' && (l.o instanceof Frame || l.o instanceof Series))) throw h.err('TypeError', `Can only merge Series or DataFrame objects, a <class '${l.k}'> was passed`)
      const left = l.o instanceof Series ? new Frame([l.o.name ?? 0], [l.o], l.o.index) : l.o
      const p: Params = { ...kw }
      if (args[2]) p.how = args[2]
      if (args[3]) p.on = args[3]
      delete p.left
      delete p.right
      const allowed = ['how', 'on', 'left_on', 'right_on', 'left_index', 'right_index', 'sort', 'suffixes', 'indicator', 'validate', 'copy']
      for (const k of Object.keys(p)) if (!allowed.includes(k)) throw h.err('TypeError', `merge() got an unexpected keyword argument '${k}'`)
      return py.obj(mergeFrames(h, left, r, p, h.nameOf(l, 'left')))
    }),
    melt: fn('melt', (args, kw) => {
      const f = args[0] ?? kw.frame
      if (!(f && f.k === 'obj' && f.o instanceof Frame)) throw h.err('TypeError', 'melt takes a DataFrame')
      const names = ['id_vars', 'value_vars', 'var_name', 'value_name', 'col_level', 'ignore_index']
      const p: Params = { ...kw }
      delete p.frame
      args.slice(1).forEach((v, i) => { p[names[i]] = v })
      return py.obj(meltFrame(h, f.o, p))
    }),
    pivot: fn('pivot', (args, kw) => {
      const f = args[0] ?? kw.data
      if (!(f && f.k === 'obj' && f.o instanceof Frame)) throw h.err('TypeError', 'pivot takes a DataFrame')
      const p: Params = { ...kw }
      delete p.data
      return py.obj(pivotFrame(h, f.o, p))
    }),
    pivot_table: fn('pivot_table', (args, kw) => {
      const f = args[0] ?? kw.data
      if (!(f && f.k === 'obj' && f.o instanceof Frame)) throw h.err('TypeError', 'pivot_table takes a DataFrame')
      const names = ['values', 'index', 'columns', 'aggfunc', 'fill_value', 'margins', 'dropna', 'margins_name', 'observed', 'sort']
      const p: Params = { ...kw }
      delete p.data
      args.slice(1).forEach((v, i) => { p[names[i]] = v })
      return py.obj(pivotTable(h, f.o, p))
    }),
    isna: fn('isna', (args, kw) => missingOf(h, need(h, args[0] ?? kw.obj, 'isna', 'obj'), false)),
    isnull: fn('isnull', (args, kw) => missingOf(h, need(h, args[0] ?? kw.obj, 'isnull', 'obj'), false)),
    notna: fn('notna', (args, kw) => missingOf(h, need(h, args[0] ?? kw.obj, 'notna', 'obj'), true)),
    notnull: fn('notnull', (args, kw) => missingOf(h, need(h, args[0] ?? kw.obj, 'notnull', 'obj'), true)),
    unique: fn('unique', (args, kw) => {
      const v = need(h, args[0] ?? kw.values, 'unique', 'values')
      const t = typedCells(h, v)
      return h.call(h.getAttr(py.obj(new Series(t.cells, null, null, t.dt)), 'unique'), [])
    }),
    value_counts: fn('value_counts', (args, kw) => {
      for (const k of Object.keys(kw)) if (!['values', 'sort', 'ascending', 'normalize', 'bins', 'dropna'].includes(k)) throw h.err('TypeError', `value_counts() got an unexpected keyword argument '${k}'`)
      if (given(kw.bins)) throw h.err('NotImplementedError', 'pd.value_counts(bins=...) is not available in this sandbox')
      const v = need(h, args[0] ?? kw.values, 'value_counts', 'values')
      const t = typedCells(h, v)
      const s = v.k === 'obj' && v.o instanceof Series ? v.o : new Series(t.cells, null, null, t.dt)
      return py.obj(valueCountsOf(s, { normalize: flag(h, kw.normalize), sort: flag(h, kw.sort, true), ascending: flag(h, kw.ascending), dropna: flag(h, kw.dropna, true) }))
    }),
    factorize: fn('factorize', (args, kw) => {
      for (const k of Object.keys(kw)) if (!['values', 'sort', 'use_na_sentinel', 'size_hint'].includes(k)) throw h.err('TypeError', `factorize() got an unexpected keyword argument '${k}'`)
      const v = need(h, args[0] ?? kw.values, 'factorize', 'values')
      const t = typedCells(h, v)
      const res = factorize(h, t.cells, t.dt, flag(h, kw.sort), flag(h, kw.use_na_sentinel, true))
      if (v.k === 'obj' && (v.o instanceof Series || v.o instanceof IndexObj)) return res
      // only a pandas object gets its uniques back as an Index
      const [codes, uniques] = (res as { items: Value[] }).items
      const ix = (uniques as { o: unknown }).o as IndexObj
      return py.tuple([codes, arrayOf(ix.labels, ix.dt)])
    }),
    to_numeric: fn('to_numeric', (args, kw) => toNumeric(h, args, kw)),
    Categorical: { k: 'type', name: 'Categorical', call: (args, kw) => {
      const v = need(h, args[0] ?? kw.values, 'Categorical', 'values')
      for (const k of Object.keys(kw)) if (!['values', 'categories', 'ordered', 'dtype'].includes(k) || k === 'dtype') throw h.err('NotImplementedError', `Categorical(${k}=...) is not available in this sandbox`)
      const t = typedCells(h, v)
      const out = categoricalOf(new Series(t.cells, null, null, t.dt), flag(h, kw.ordered ?? args[2]), given(kw.categories ?? args[1]) ? cellsOf(h, kw.categories ?? args[1]) : undefined)
      out.bare = true
      return py.obj(out)
    } },
    option_context: fn('option_context', (args) => {
      if (!args.length || args.length % 2) throw h.err('ValueError', 'Need to invoke as option_context(pat, val, [(pat, val), ...]).')
      return py.obj(new OptionContext(args))
    }),
    NamedAgg: { k: 'type', name: 'NamedAgg', call: (args, kw) => py.tuple([args[0] ?? kw.column, args[1] ?? kw.aggfunc]) },
    NA: py.float(NaN),
    set_option: fn('set_option', (args) => {
      if (!args.length || args.length % 2) throw h.err('ValueError', 'Must provide an even number of non-keyword arguments')
      for (let i = 0; i < args.length; i += 2) setOption(h, h.str(args[i]), args[i + 1])
      return py.NONE
    }),
    get_option: fn('get_option', (args) => optionValue(optKey(h, h.str(args[0])))),
    reset_option: fn('reset_option', (args) => {
      const name = h.str(args[0])
      if (name === 'all' || name === 'display') Object.assign(OPTIONS, DEFAULT_OPTIONS)
      else {
        const k = optKey(h, name)
        ;(OPTIONS as Record<string, unknown>)[k] = DEFAULT_OPTIONS[k]
      }
      return py.NONE
    }),
    options: py.obj(new OptionsNode('')),
    errors: { k: 'module', name: 'pandas.errors', attrs: Object.fromEntries(['MergeError', 'ParserError', 'EmptyDataError', 'DataError', 'IntCastingNaNError', 'UndefinedVariableError', 'OptionError', 'IndexingError', 'SpecificationError'].map((n) => [n, h.excClass(n)])) },
  }
  return { pandas: { k: 'module', name: 'pandas', attrs: pd } }
}

const readerCall = (h: Host, args: Value[], kw: Kw, sep: string): Value => {
  const src = args[0] ?? kw.filepath_or_buffer
  if (!src) throw h.err('TypeError', "read_csv() missing 1 required positional argument: 'filepath_or_buffer'")
  const text = textFrom(h, src)
  if (text === undefined) throw h.err('FileNotFoundError', `[Errno 2] No such file or directory: '${h.str(src)}' (this sandbox starts with no files: write one first, or wrap the CSV text in io.StringIO(...))`)
  const kw2 = { ...kw }
  if (args[1]) kw2.sep = args[1]
  return py.obj(readCsv(h, text, kw2, sep))
}

const toNumeric = (h: Host, args: Value[], kw: Kw): Value => {
  for (const k of Object.keys(kw)) if (!['arg', 'errors', 'downcast', 'dtype_backend'].includes(k)) throw h.err('TypeError', `to_numeric() got an unexpected keyword argument '${k}'`)
  for (const k of ['downcast', 'dtype_backend']) if (given(kw[k])) throw h.err('NotImplementedError', `to_numeric(${k}=...) is not available in this sandbox`)
  const a = args[0] ?? kw.arg
  const errors = given(kw.errors ?? args[1]) ? h.str(kw.errors ?? args[1]) : 'raise'
  if (!['raise', 'coerce', 'ignore'].includes(errors)) throw h.err('ValueError', 'invalid error value specified')
  const conv = (c: Cell, i: number): { v: number; isFloat: boolean } | null => {
    if (typeof c === 'number') return { v: c, isFloat: !Number.isInteger(c) || Number.isNaN(c) }
    if (typeof c === 'boolean') return { v: Number(c), isFloat: false }
    if (c === null) return { v: NaN, isFloat: true }
    if (typeof c === 'string') {
      const t = c.trim()
      if (/^[+-]?\d+$/.test(t)) return { v: Number(t), isFloat: false }
      if (/^[+-]?(\d+\.?\d*([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?)$/.test(t)) return { v: Number(t), isFloat: true }
      if (/^[+-]?(inf|infinity)$/i.test(t)) return { v: t.startsWith('-') ? -Infinity : Infinity, isFloat: true }
      if (t === '') return { v: NaN, isFloat: true }
    }
    if (errors === 'coerce') return { v: NaN, isFloat: true }
    if (errors === 'ignore') return null
    if (typeof c === 'string') throw h.err('ValueError', `Unable to parse string "${c}" at position ${i}`)
    throw h.err('TypeError', `Invalid object type at position ${i}`)
  }
  if (a.k === 'obj' && a.o instanceof Series && a.o.dtype !== 'object') return py.obj(a.o.copy())
  const scalar = !(isListLike(a) || a.k === 'obj')
  const cells = scalar ? [cellOf(h, a)] : typedCells(h, a).cells
  const res = cells.map(conv)
  if (res.some((x) => x === null)) return a
  const anyFloat = (res as { v: number; isFloat: boolean }[]).some((x) => x.isFloat)
  const srcDT = a.k === 'obj' && a.o instanceof Series ? a.o.dtype : null
  const dt: DT = anyFloat || srcDT === 'float64' ? 'float64' : 'int64'
  const nums = (res as { v: number }[]).map((x) => x.v)
  if (scalar) return dt === 'int64' ? py.int(nums[0]) : py.float(nums[0])
  if (a.k === 'obj' && a.o instanceof Series) return py.obj(a.o.with(nums, dt))
  return arrayOf(nums, dt)
}

