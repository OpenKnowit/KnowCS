/** Tables from tables: merge, join, concat, pivot, pivot_table, crosstab, melt, stack, unstack, get_dummies, cut and qcut. */
import { py, repr } from './minipy'
import type { Host, Value } from './minipy'
import { NDArray, pyFloatRepr } from './ndarray'
import { Frame, IndexObj, Series, cellOf, cellsOf, commonDT, convertCells, inferDT, isListLike, isMissing, labelStr, numCell, reindexCells, sortOrder, tableOfFrame, tableOfSeries, traceFrame, typedCells, valueOf, seriesIn } from './pyPandas'
import type { Cats, Cell, DT, Params } from './pyPandas'
import { PyCell } from './pyPandasFormat'
import { GroupBy } from './pyPandasGroup'
import { flag, funcName, given, need, npQuantile, transposeFrame, typeArg } from './pyPandasMethods'

const rows = (n: number) => Array.from({ length: n }, (_, i) => i)
const keyStr = (c: Cell) => (isMissing(c) ? 'NaN' : c instanceof PyCell ? repr(c.v) : typeof c === 'boolean' ? `b${c}` : typeof c === 'number' ? `n${c}` : `s${c}`)
const uniq = (cells: Cell[]) => [...new Map(cells.map((c) => [keyStr(c), c])).values()]
const sortedUniq = (cells: Cell[]) => {
  const u = uniq(cells)
  return sortOrder([u], [true]).map((i) => u[i])
}
/** the dtype a column becomes once rows without a value appear */
const withMissing = (dt: DT, hasMissing: boolean): DT => (!hasMissing ? dt : dt === 'int64' ? 'float64' : dt === 'bool' ? 'object' : dt)
const pickOrMissing = (s: Series, at: number[]): Series => {
  const miss = at.some((i) => i < 0)
  const dt = withMissing(s.dtype, miss)
  const base = convertCells(s.values, s.dtype, dt)
  const out = new Series(at.map((i) => (i < 0 ? NaN : base[i])), null, s.name, dt)
  if (!miss) out.cats = s.cats
  return out
}

// ---------------------------------------------------------------- merge

const labelsArg = (v: Value): Value[] => (v.k === 'list' || v.k === 'tuple' ? v.items : [v])

/** one join key side: cells per row, and the column it came from (if any) */
const keyColumns = (h: Host, f: Frame, vs: Value[], side: string): { cells: Cell[]; dt: DT; label: Cell | null }[] => vs.map((v) => {
  if (isListLike(v)) {
    const t = typedCells(h, v)
    return { cells: t.cells, dt: t.dt, label: null }
  }
  const c = cellOf(h, v)
  const col = f.col(c)
  if (!col) {
    if (f.index.names.includes(c) && c !== null) {
      const lev = f.index.level(f.index.names.indexOf(c))
      return { cells: lev.labels, dt: lev.dt, label: null }
    }
    void side
    throw h.err('KeyError', repr(v))
  }
  return { cells: col.values, dt: col.dtype, label: c }
})

const seriesIn0 = (v: Value) => (v as { o: unknown }).o as Frame

const asFrame = (h: Host, v: Value): Frame => {
  if (v.k === 'obj' && v.o instanceof Frame) return v.o
  if (v.k === 'obj' && v.o instanceof Series) {
    if (v.o.name === null) throw h.err('ValueError', 'Cannot merge a Series without a name')
    return new Frame([v.o.name], [v.o], v.o.index)
  }
  throw h.err('TypeError', `Can only merge Series or DataFrame objects, a <class '${v.k === 'obj' ? v.o.cls : v.k}'> was passed`)
}

export const mergeFrames = (h: Host, left: Frame, rightV: Value, p: Params, label: string): Frame => {
  const right = asFrame(h, rightV)
  const how = given(p.how) ? h.str(p.how!) : 'inner'
  if (!['inner', 'left', 'right', 'outer', 'cross'].includes(how)) throw h.err('ValueError', `do not recognize join method ${how}`)
  if (given(p.validate)) throw h.err('NotImplementedError', 'merge(validate=...) is not available in this sandbox')
  const suffixes = given(p.suffixes) ? h.iterate(p.suffixes!).map((x) => (x.k === 'none' ? '' : h.str(x))) : ['_x', '_y']
  const leftIdx = flag(h, p.left_index)
  const rightIdx = flag(h, p.right_index)
  const onV = given(p.on)
  const lonV = given(p.left_on)
  const ronV = given(p.right_on)
  if (how === 'cross') {
    if (onV || lonV || ronV || leftIdx || rightIdx) throw h.err('MergeError', "Can not pass on, right_on, left_on or set right_index=True or left_index=True")
    const pairs = rows(left.nrows).flatMap((l) => rows(right.nrows).map((r) => [l, r]))
    return assemble(h, left, right, pairs, [], [], suffixes, 'none', p, label)
  }
  if (onV && (lonV || ronV)) throw h.err('MergeError', 'Can only pass argument "on" OR "left_on" and "right_on", not a combination of both.')
  if (onV && (leftIdx || rightIdx)) throw h.err('MergeError', 'Can only pass argument "on" OR "left_index" and "right_index", not a combination of both.')
  let lk: { cells: Cell[]; dt: DT; label: Cell | null }[]
  let rk: { cells: Cell[]; dt: DT; label: Cell | null }[]
  const indexKeys = (f: Frame) => Array.from({ length: f.index.nlevels }, (_, j) => ({ cells: f.index.level(j).labels, dt: f.index.level(j).dt, label: null as Cell | null }))
  if (leftIdx && rightIdx) {
    lk = indexKeys(left)
    rk = indexKeys(right)
  } else if (leftIdx || rightIdx) {
    if (leftIdx && !ronV) throw h.err('MergeError', 'Must pass "right_on" OR "right_index".')
    if (rightIdx && !lonV) throw h.err('MergeError', 'Must pass "left_on" OR "left_index".')
    lk = leftIdx ? indexKeys(left) : keyColumns(h, left, labelsArg(lonV!), 'left')
    rk = rightIdx ? indexKeys(right) : keyColumns(h, right, labelsArg(ronV!), 'right')
  } else if (lonV || ronV) {
    if (!lonV || !ronV) throw h.err('MergeError', lonV ? 'Must pass "right_on" OR "right_index".' : 'Must pass "left_on" OR "left_index".')
    lk = keyColumns(h, left, labelsArg(lonV), 'left')
    rk = keyColumns(h, right, labelsArg(ronV), 'right')
  } else {
    const names = onV ? labelsArg(onV) : left.columns.labels.filter((c) => right.columns.has(c)).map((c) => valueOf(c, 'object'))
    if (!names.length) throw h.err('MergeError', `No common columns to perform merge on. Merge options: left_on=None, right_on=None, left_index=${leftIdx ? 'True' : 'False'}, right_index=${rightIdx ? 'True' : 'False'}`)
    lk = keyColumns(h, left, names, 'left')
    rk = keyColumns(h, right, names, 'right')
  }
  if (lk.length !== rk.length) throw h.err('ValueError', 'len(right_on) must equal len(left_on)')
  const lkey = (r: number) => lk.map((k) => keyStr(k.cells[r])).join('\0')
  const rkey = (r: number) => rk.map((k) => keyStr(k.cells[r])).join('\0')
  const rightRows = new Map<string, number[]>()
  for (const r of rows(right.nrows)) {
    const e = rightRows.get(rkey(r))
    if (e) e.push(r)
    else rightRows.set(rkey(r), [r])
  }
  const leftRows = new Map<string, number[]>()
  for (const r of rows(left.nrows)) {
    const e = leftRows.get(lkey(r))
    if (e) e.push(r)
    else leftRows.set(lkey(r), [r])
  }
  let pairs: number[][] = []
  if (how === 'right') {
    for (const r of rows(right.nrows)) {
      const m = leftRows.get(rkey(r))
      if (m) m.forEach((l) => pairs.push([l, r]))
      else pairs.push([-1, r])
    }
  } else {
    for (const l of rows(left.nrows)) {
      const m = rightRows.get(lkey(l))
      if (m) m.forEach((r) => pairs.push([l, r]))
      else if (how !== 'inner') pairs.push([l, -1])
    }
    if (how === 'outer') for (const r of rows(right.nrows)) if (!leftRows.has(rkey(r))) pairs.push([-1, r])
  }
  if (how === 'outer' || flag(h, p.sort)) {
    const cols = lk.map((k, j) => pairs.map(([l, r]) => (l >= 0 ? k.cells[l] : rk[j].cells[r])))
    const order = sortOrder(cols, cols.map(() => true))
    pairs = order.map((i) => pairs[i])
  }
  const mode = leftIdx && rightIdx ? 'both' : leftIdx ? 'left' : rightIdx ? 'right' : 'none'
  return assemble(h, left, right, pairs, lk, rk, suffixes, mode, p, label)
}

const assemble = (h: Host, left: Frame, right: Frame, pairs: number[][], lk: { cells: Cell[]; dt: DT; label: Cell | null }[], rk: { cells: Cell[]; dt: DT; label: Cell | null }[], suffixes: string[], indexMode: string, p: Params, label: string): Frame => {
  const dropRight = new Set<Cell>()
  const coalesce = new Map<Cell, { lk: (typeof lk)[0]; rk: (typeof rk)[0] }>()
  lk.forEach((k, j) => {
    if (k.label !== null && rk[j].label !== null && k.label === rk[j].label) {
      dropRight.add(rk[j].label)
      coalesce.set(k.label, { lk: k, rk: rk[j] })
    }
  })
  const rightCols = right.columns.labels.filter((c) => !dropRight.has(c))
  const overlap = (c: Cell, other: Cell[]) => other.includes(c)
  const leftOut = left.columns.labels
  const lat = pairs.map((x) => x[0])
  const rat = pairs.map((x) => x[1])
  const labels: Cell[] = []
  const data: Series[] = []
  leftOut.forEach((c, k) => {
    const named = !dropRight.has(c) && overlap(c, rightCols) ? `${labelStr(c)}${suffixes[0]}` : c
    labels.push(named)
    const co = coalesce.get(c)
    if (co) {
      // the key keeps the left dtype unless rows from the right alone appear
      const dt = pairs.some(([l]) => l < 0) ? commonDT([co.lk.dt, co.rk.dt]) : co.lk.dt
      const a = convertCells(co.lk.cells, co.lk.dt, dt)
      const b = convertCells(co.rk.cells, co.rk.dt, dt)
      data.push(new Series(pairs.map(([l, r]) => (l >= 0 ? a[l] : b[r])), null, named, dt))
    } else data.push(pickOrMissing(left.data[k], lat))
  })
  for (const c of rightCols) {
    const named = overlap(c, leftOut) ? `${labelStr(c)}${suffixes[1]}` : c
    labels.push(named)
    data.push(pickOrMissing(right.col(c)!, rat))
  }
  const ind = given(p.indicator)
  if (ind && !(ind.k === 'bool' && !ind.v)) {
    const name = ind.k === 'str' ? ind.v : '_merge'
    const s = new Series(pairs.map(([l, r]) => (l < 0 ? 'right_only' : r < 0 ? 'left_only' : 'both')), null, name, 'object')
    s.cats = { labels: ['left_only', 'right_only', 'both'], kind: 'object', ordered: false }
    labels.push(name)
    data.push(s)
  }
  let index: IndexObj | null = null
  if (indexMode === 'both') {
    const cells = lk.map((k, j) => pairs.map(([l, r]) => (l >= 0 ? k.cells[l] : rk[j].cells[r])))
    index = lk.length === 1 ? new IndexObj(cells[0], left.index.name === right.index.name ? left.index.name : null, commonDT([lk[0].dt, rk[0].dt])) : IndexObj.fromArrays(cells, left.index.names, lk.map((k) => k.dt))
  } else if (indexMode === 'right') {
    const hasMissing = lat.some((x) => x < 0)
    index = hasMissing ? null : left.index.pick(lat)
  } else if (indexMode === 'left') {
    const hasMissing = rat.some((x) => x < 0)
    index = hasMissing ? null : right.index.pick(rat)
  }
  const out = new Frame(new IndexObj(labels, null, commonDT([left.columns.dt, right.columns.dt])), data, index)
  traceFrame(h, 'merge', () => ({ inputs: [{ label, table: tableOfFrame(left) }, { label: 'right', table: tableOfFrame(right) }], result: tableOfFrame(out) }))
  return out
}

export const joinFrames = (h: Host, f: Frame, p: Params): Frame => {
  const other = need(h, p.other, 'join', 'other')
  if (other.k === 'list' || other.k === 'tuple') {
    if (given(p.on)) throw h.err('ValueError', 'Joining multiple DataFrames only supported for joining on index')
    // pandas joins several frames by an outer concat on the index, then takes the rows the join asks for
    const how = given(p.how) ? h.str(p.how!) : 'left'
    const all = seriesIn0(concatObjects(h, [py.list([py.obj(f), ...other.items])], { axis: py.int(1), ...(how === 'inner' ? { join: py.str('inner') } : {}) }))
    if (how === 'left') return new Frame(all.columns, all.data.map((c) => {
      const t = reindexCells(c, f.index)
      return new Series(t.cells, f.index, c.name, c.dtype === 'int64' && !all.index.equals(f.index) ? 'float64' : t.dt)
    }), f.index)
    return all
  }
  const right = asFrame(h, other)
  const lsuf = given(p.lsuffix) ? h.str(p.lsuffix!) : ''
  const rsuf = given(p.rsuffix) ? h.str(p.rsuffix!) : ''
  const overlap = f.columns.labels.filter((c) => right.columns.has(c))
  if (overlap.length && !lsuf && !rsuf) throw h.err('ValueError', `columns overlap but no suffix specified: ${new IndexObj(overlap, null, inferDT(overlap)).repr()}`)
  const mp: Params = { how: p.how ?? py.str('left'), suffixes: py.tuple([py.str(lsuf), py.str(rsuf)]), sort: p.sort, validate: p.validate, right_index: py.bool(true) }
  if (given(p.on)) mp.left_on = p.on
  else mp.left_index = py.bool(true)
  return mergeFrames(h, f, py.obj(right), mp, h.nameOf(py.obj(f), 'left'))
}

// ---------------------------------------------------------------- concat

export const concatObjects = (h: Host, args: Value[], kw: Record<string, Value>): Value => {
  const objsV = args[0] ?? kw.objs
  if (!objsV) throw h.err('TypeError', "concat() missing 1 required positional argument: 'objs'")
  for (const k of Object.keys(kw)) if (!['objs', 'axis', 'join', 'ignore_index', 'keys', 'names', 'sort', 'copy', 'verify_integrity', 'levels'].includes(k)) throw h.err('TypeError', `concat() got an unexpected keyword argument '${k}'`)
  if (given(kw.levels)) throw h.err('NotImplementedError', 'concat(levels=...) is not available in this sandbox')
  if (objsV.k === 'obj' && (objsV.o instanceof Frame || objsV.o instanceof Series)) throw h.err('TypeError', `first argument must be an iterable of pandas objects, you passed an object of type "${objsV.o.cls}"`)
  const dictKeys = objsV.k === 'dict' ? objsV.d.keys() : null
  const items = (objsV.k === 'dict' ? objsV.d.values() : h.iterate(objsV)).filter((v) => v.k !== 'none')
  if (!items.length) throw h.err('ValueError', 'No objects to concatenate')
  for (const v of items) if (!(v.k === 'obj' && (v.o instanceof Frame || v.o instanceof Series))) throw h.err('TypeError', `cannot concatenate object of type '${v.k === 'obj' ? `<class '${v.o.cls}'>` : `<class '${v.k}'>`}'; only Series and DataFrame objs are valid`)
  const objs = items.map((v) => (v as { o: unknown }).o as Frame | Series)
  const axis = given(kw.axis) ? (kw.axis.k === 'str' ? (kw.axis.v === 'columns' ? 1 : 0) : h.toInt(kw.axis)) : 0
  const join = given(kw.join) ? h.str(kw.join) : 'outer'
  if (join !== 'outer' && join !== 'inner') throw h.err('ValueError', 'Only can inner (intersect) or outer (union) join the other axis')
  const ignore = flag(h, kw.ignore_index)
  const sortCols = flag(h, kw.sort)
  const keysV = given(kw.keys) ?? (dictKeys ? py.list(dictKeys) : undefined)
  const keys = keysV ? typedCells(h, keysV) : null
  const namesArg = given(kw.names) ? cellsOf(h, kw.names) : null
  const allSeries = objs.every((o) => o instanceof Series)
  if (axis === 0 && allSeries) {
    const ss = objs as Series[]
    const dt = commonDT(ss.map((s) => s.dtype))
    const vals = ss.flatMap((s) => convertCells(s.values, s.dtype, dt))
    const name = ss.every((s) => s.name === ss[0].name) ? ss[0].name : null
    const index = ignore ? IndexObj.range(vals.length) : keys ? keyedIndex(ss.map((s) => s.index), keys, namesArg) : combineIndex(ss.map((s) => s.index))
    const out = new Series(vals, index, name, dt)
    if (ss.every((s) => s.cats && s.cats === ss[0].cats)) out.cats = ss[0].cats
    if (flag(h, kw.verify_integrity) && !index.isUnique) throw overlapError(h, index)
    return py.obj(out)
  }
  const frames = objs.map((o, i) => (o instanceof Frame ? o : new Frame([o.name ?? (axis === 1 ? (keys ? keys.cells[i] : i) : 0)], [o], o.index)))
  if (axis === 1) {
    let index = frames[0].index
    for (const f of frames.slice(1)) index = join === 'inner' ? intersectIndex(index, f.index) : frames.every((x) => x.index.equals(frames[0].index)) ? index : unionUnsorted(index, f.index)
    if (sortCols) index = index.pick(sortOrder([index.labels], [true]))
    const data: Series[] = []
    const lab: Cell[] = []
    const topKeys: Cell[] = []
    const subs: Cell[] = []
    frames.forEach((f, i) => f.data.forEach((s) => {
      const t = reindexCells(s, index)
      data.push(new Series(t.cells === s.values ? [...t.cells] : t.cells, index, s.name, t.dt))
      lab.push(s.name)
      if (keys) {
        topKeys.push(keys.cells[i])
        subs.push(s.name)
      }
    }))
    const objsAllSeries = allSeries
    let columns: IndexObj
    if (ignore) columns = IndexObj.range(lab.length)
    else if (keys && objsAllSeries) columns = new IndexObj(keys.cells, namesArg ? namesArg[0] : null, keys.dt)
    else if (keys) columns = IndexObj.fromArrays([topKeys, subs], namesArg ? [namesArg[0], namesArg[1] ?? null] : [null, null], [keys.dt, inferDT(subs)])
    else if (allSeries && (objs as Series[]).every((s) => s.name === null)) columns = IndexObj.range(lab.length)
    else columns = new IndexObj(lab.map((l, i) => (allSeries && (objs as Series[])[i].name === null ? i : l)))
    return py.obj(new Frame(columns, data, index))
  }
  let cols = frames[0].columns
  for (const f of frames.slice(1)) cols = join === 'inner' ? intersectIndex(cols, f.columns) : cols.equals(f.columns) ? cols : unionUnsorted(cols, f.columns)
  if (sortCols) cols = cols.pick(sortOrder([cols.labels], [true]))
  const data = cols.labels.map((c) => {
    const parts = frames.map((f) => f.col(c))
    const present = parts.filter((x): x is Series => !!x)
    const missing = parts.length > present.length
    const dt = withMissing(commonDT(present.map((s) => s.dtype)), missing)
    const vals = frames.flatMap((f, i) => {
      const s = parts[i]
      return s ? convertCells(s.values, s.dtype, dt) : f.index.labels.map(() => NaN)
    })
    return new Series(vals, null, c, dt)
  })
  const n = frames.reduce((a, f) => a + f.nrows, 0)
  const index = ignore ? IndexObj.range(n) : keys ? keyedIndex(frames.map((f) => f.index), keys, namesArg) : combineIndex(frames.map((f) => f.index))
  if (flag(h, kw.verify_integrity) && !index.isUnique) throw overlapError(h, index)
  return py.obj(new Frame(cols, data, index))
}

const overlapError = (h: Host, ix: IndexObj) => {
  const seen = new Set<string>()
  const dup: Cell[] = []
  for (const l of ix.labels) {
    if (seen.has(keyStr(l)) && !dup.includes(l)) dup.push(l)
    seen.add(keyStr(l))
  }
  return h.err('ValueError', `Indexes have overlapping values: ${new IndexObj(dup, null, ix.dt).repr()}`)
}

const combineIndex = (ixs: IndexObj[]): IndexObj => {
  const name = ixs.every((x) => x.name === ixs[0].name) ? ixs[0].name : null
  if (ixs.every((x) => x.multi)) {
    const m0 = ixs[0].multi!
    return IndexObj.fromArrays(m0.arrays.map((_, j) => ixs.flatMap((x) => x.multi!.arrays[j])), m0.names, m0.dts)
  }
  return new IndexObj(ixs.flatMap((x) => x.labels), name, commonDT(ixs.map((x) => x.dt)))
}

const keyedIndex = (ixs: IndexObj[], keys: { cells: Cell[]; dt: DT }, names: Cell[] | null): IndexObj => {
  const top = ixs.flatMap((x, i) => x.labels.map(() => keys.cells[i]))
  const levels = ixs[0].nlevels
  const inner = Array.from({ length: levels }, (_, j) => ixs.flatMap((x) => x.level(j).labels))
  const nm: Cell[] = [names ? names[0] : null, ...Array.from({ length: levels }, (_, j) => (names && names[j + 1] !== undefined ? names[j + 1] : ixs[0].names[j]))]
  return IndexObj.fromArrays([top, ...inner], nm, [keys.dt, ...Array.from({ length: levels }, (_, j) => ixs[0].level(j).dt)])
}

const unionUnsorted = (a: IndexObj, b: IndexObj): IndexObj => new IndexObj([...a.labels, ...b.labels.filter((l) => !a.has(l))], a.name === b.name ? a.name : null, commonDT([a.dt, b.dt]))
const intersectIndex = (a: IndexObj, b: IndexObj): IndexObj => new IndexObj(a.labels.filter((l) => b.has(l)), a.name === b.name ? a.name : null, a.dt)

// ---------------------------------------------------------------- reshaping into a grid

/** (row key, column key) → cell, filling a grid in sorted key order */
const levelsIndex = (arrays: Cell[][], names: Cell[], dts: DT[]) => (arrays.length === 1 ? new IndexObj(arrays[0], names[0], dts[0]) : IndexObj.fromArrays(arrays, names, dts))

export const pivotFrame = (h: Host, f: Frame, p: Params): Frame => {
  if (p.columns === undefined) throw h.err('TypeError', "DataFrame.pivot() missing 1 required keyword-only argument: 'columns'")
  const colKeys = labelsArg(p.columns).map((v) => {
    const c = f.col(cellOf(h, v))
    if (!c) throw h.err('KeyError', repr(py.str(`None of [${repr(v)}] are in the columns`)))
    return c
  })
  const rowKeys = given(p.index) ? labelsArg(p.index!).map((v) => {
    const c = f.col(cellOf(h, v))
    if (!c) {
      if (f.index.names.includes(cellOf(h, v))) return new Series(f.index.level(f.index.names.indexOf(cellOf(h, v))).labels, null, cellOf(h, v), f.index.level(f.index.names.indexOf(cellOf(h, v))).dt)
      throw h.err('KeyError', repr(v))
    }
    return c
  }) : [new Series(f.index.labels, null, f.index.name, f.index.dt)]
  for (const a of rowKeys) for (const b of colKeys) if (a.name === b.name && a.name !== null) throw h.err('ValueError', `The name ${labelStr(a.name)} occurs multiple times, use a level number`)
  const given_ = given(p.values)
  const usedNames = [...colKeys, ...rowKeys].map((s) => s.name)
  const valueCols = given_ ? labelsArg(given_).map((v) => {
    const c = f.col(cellOf(h, v))
    if (!c) throw h.err('KeyError', repr(v))
    return c
  }) : f.data.filter((s) => !usedNames.includes(s.name))
  const topLevel = !given_ || (given_ && isListLike(given_))
  const rowKey = (r: number) => rowKeys.map((s) => keyStr(s.values[r])).join('\0')
  const colKey = (r: number) => colKeys.map((s) => keyStr(s.values[r])).join('\0')
  const allRows = rows(f.nrows)
  const seen = new Set<string>()
  for (const r of allRows) {
    const k = `${rowKey(r)}\u0001${colKey(r)}`
    if (seen.has(k)) throw h.err('ValueError', 'Index contains duplicate entries, cannot reshape')
    seen.add(k)
  }
  const uniqRows = new Map<string, number>()
  const uniqCols = new Map<string, number>()
  for (const r of allRows) {
    if (!uniqRows.has(rowKey(r))) uniqRows.set(rowKey(r), r)
    if (!uniqCols.has(colKey(r))) uniqCols.set(colKey(r), r)
  }
  const rowReps = [...uniqRows.values()]
  const colReps = [...uniqCols.values()]
  const rowOrder = sortOrder(rowKeys.map((s) => rowReps.map((r) => s.values[r])), rowKeys.map(() => true)).map((i) => rowReps[i])
  const colOrder = sortOrder(colKeys.map((s) => colReps.map((r) => s.values[r])), colKeys.map(() => true)).map((i) => colReps[i])
  const index = levelsIndex(rowKeys.map((s) => rowOrder.map((r) => s.values[r])), rowKeys.map((s) => s.name), rowKeys.map((s) => s.dtype))
  const lookup = new Map<string, number>()
  for (const r of allRows) lookup.set(`${rowKey(r)}\u0001${colKey(r)}`, r)
  const data: Series[] = []
  for (const v of valueCols) {
    const per = colOrder.map((cr) => rowOrder.map((rr) => lookup.get(`${rowKey(rr)}\u0001${colKey(cr)}`) ?? -1))
    const anyMissing = per.some((x) => x.some((i) => i < 0))
    const dt = withMissing(v.dtype, anyMissing)
    const base = convertCells(v.values, v.dtype, dt)
    per.forEach((at) => data.push(new Series(at.map((i) => (i < 0 ? NaN : base[i])), index, null, dt)))
  }
  const colArrays = colKeys.map((s) => colOrder.map((r) => s.values[r]))
  let columns: IndexObj
  if (topLevel) {
    const tops = valueCols.flatMap((v) => colOrder.map(() => v.name))
    columns = IndexObj.fromArrays([tops, ...colKeys.map((_, j) => valueCols.flatMap(() => colArrays[j]))], [null, ...colKeys.map((s) => s.name)], [inferDT(tops), ...colKeys.map((s) => s.dtype)])
  } else columns = levelsIndex(colArrays, colKeys.map((s) => s.name), colKeys.map((s) => s.dtype))
  return new Frame(columns, data, index)
}

/** pivot_table: agg over (index, columns), then one column per key combination */
export const pivotTable = (h: Host, f: Frame, p: Params): Frame => {
  const toKeys = (v: Value | undefined): Value[] => (given(v) ? labelsArg(v!) : [])
  const rowVs = toKeys(p.index)
  const colVs = toKeys(p.columns)
  if (!rowVs.length && !colVs.length) throw h.err('ValueError', 'No group keys passed!')
  const keyCols = [...rowVs, ...colVs].map((v) => {
    if (v.k === 'obj' && v.o instanceof Series) return v.o
    const c = f.col(cellOf(h, v))
    if (!c) throw h.err('KeyError', repr(v))
    return c
  })
  const rowKeys = keyCols.slice(0, rowVs.length)
  const colKeys = keyCols.slice(rowVs.length)
  const valsV = given(p.values)
  const multiValues = !valsV || isListLike(valsV)
  const keyNames = keyCols.map((s) => s.name)
  const valueNames = valsV ? labelsArg(valsV).map((v) => {
    const c = cellOf(h, v)
    if (!f.columns.has(c)) throw h.err('KeyError', repr(v))
    return c
  }) : f.columns.labels.filter((c) => !keyNames.includes(c))
  const aggV = p.aggfunc ?? py.str('mean')
  if (!rowKeys.length) {
    const table = pivotTable(h, f, { ...p, index: p.columns, columns: undefined })
    return transposeFrame(table)
  }
  const dropna = flag(h, p.dropna, true)
  const fill = given(p.fill_value)
  const margins = flag(h, p.margins)
  const marginName = given(p.margins_name) ? cellOf(h, p.margins_name!) : 'All'
  const sortKeys = flag(h, p.sort, true)
  if (isListLike(aggV) || aggV.k === 'dict') {
    const dictHasList = aggV.k === 'dict' && aggV.d.values().some(isListLike)
    const entries: { label: Cell | null; value: Cell; f: Value }[] = aggV.k === 'dict'
      ? aggV.d.items().flatMap(([c, x]): { label: Cell | null; value: Cell; f: Value }[] => (isListLike(x) ? h.iterate(x) : [x]).map((g) => ({ label: dictHasList ? funcName(h, g) : null, value: cellOf(h, c), f: g })))
      : h.iterate(aggV).flatMap((g) => valueNames.map((v) => ({ label: funcName(h, g), value: v, f: g })))
    return buildPivot(h, f, rowKeys, colKeys, entries, true, valsV ? multiValues : true, dropna, fill, margins, marginName, sortKeys, aggV.k !== 'dict')
  }
  const entries = valueNames.map((v) => ({ label: null, value: v, f: aggV }))
  return buildPivot(h, f, rowKeys, colKeys, entries, false, multiValues, dropna, fill, margins, marginName, sortKeys, false)
}

const statOf = (h: Host, g: GroupBy, s: Series, members: number[][], fn: Value): { cells: Cell[]; dt: DT } => {
  const k = g.f.columns.labels.indexOf(s.name)
  const fixed = { keys: members.map(() => []), members, ofRow: [] }
  return g.aggOne(h, fn, k, fixed)
}

const buildPivot = (h: Host, f0: Frame, rowKeys: Series[], colKeys: Series[], entries: { label: Cell | null; value: Cell; f: Value }[], hasFuncLevel: boolean, keepValueLevel: boolean, dropna: boolean, fill: Value | undefined, margins: boolean, marginName: Cell, sortKeys: boolean, funcFirst: boolean): Frame => {
  // rows with a missing key or value leave the table
  const keep = rows(f0.nrows).filter((r) => [...rowKeys, ...colKeys].every((s) => !isMissing(s.values[r])))
  const f = keep.length === f0.nrows ? f0 : f0.take(keep, null)
  const rk = rowKeys.map((s) => (keep.length === f0.nrows ? s : s.pick(keep)))
  const ck = colKeys.map((s) => (keep.length === f0.nrows ? s : s.pick(keep)))
  const rowOf = (r: number) => rk.map((s) => keyStr(s.values[r])).join('\0')
  const colOf = (r: number) => ck.map((s) => keyStr(s.values[r])).join('\0')
  const orderOf = (keys: Series[], key: (r: number) => string) => {
    const reps = new Map<string, number>()
    rows(f.nrows).forEach((r) => { if (!reps.has(key(r))) reps.set(key(r), r) })
    const list = [...reps.values()]
    const cols = keys.map((s) => list.map((r) => (s.cats ? s.cats.labels.indexOf(s.values[r]) : s.values[r])))
    if (!sortKeys && !keys.some((s) => s.cats)) return list
    return sortOrder(cols, keys.map(() => true)).map((i) => list[i])
  }
  const rowReps = orderOf(rk, rowOf)
  const colReps = ck.length ? orderOf(ck, colOf) : [-1]
  const buckets = new Map<string, number[]>()
  for (const x of rows(f.nrows)) {
    const k = `${rowOf(x)}\u0001${colOf(x)}`
    const b = buckets.get(k)
    if (b) b.push(x)
    else buckets.set(k, [x])
  }
  const group = (r: number, c: number): number[] => (r < 0 || c < 0 ? rows(f.nrows).filter((x) => (r < 0 || rowOf(x) === rowOf(r)) && (c < 0 || colOf(x) === colOf(c))) : buckets.get(`${rowOf(r)}\u0001${colOf(c)}`) ?? [])
  const gb = new GroupBy(f, [], null, false, { asIndex: true, sort: true, dropna: true, groupKeys: true, observed: true }, 'df')
  const columnsOut: { top: Cell | null; val: Cell; sub: number; series: Series }[] = []
  const makeIndex = (extra: boolean) => {
    const arrays = rk.map((s, j) => {
      const base = rowReps.map((r) => s.values[r])
      return extra ? [...base, j === 0 ? marginName : ''] : base
    })
    const dts = rk.map((s) => (extra ? (s.dtype === 'object' ? 'object' : 'object') : s.dtype) as DT)
    return arrays.length ? levelsIndex(arrays, rk.map((s) => s.name), dts) : null
  }
  const baseIndex = makeIndex(false)
  const entryColumns = entries.map((e) => {
    const s = f.col(e.value)!
    const per: Cell[][] = []
    const perDt: DT[] = []
    const rowMembers = (rr: number) => (rk.length ? rr : -1)
    for (const c of colReps) {
      const members = (rk.length ? rowReps : [-1]).map((r) => group(rowMembers(r), c).filter((x) => !(dropna && isMissing(s.values[x]) && false)))
      const nonEmpty = members.map((m) => m.length > 0)
      const st = statOf(h, gb, s, members.map((m) => m), e.f)
      per.push(st.cells.map((x, i) => (nonEmpty[i] ? x : NaN)))
      perDt.push(st.dt)
    }
    return { e, s, per, perDt }
  })
  const nRows = rk.length ? rowReps.length : 1
  const keepRow = rows(nRows).filter((i) => !dropna || !rk.length || entryColumns.some((ec) => ec.per.some((c) => !isMissing(c[i]))))
  const keepCol = colReps.map((_, j) => !dropna || !ck.length || entryColumns.some((ec) => ec.per[j].some((x) => !isMissing(x))))
  const index = keepRow.length === nRows ? baseIndex : baseIndex ? baseIndex.pick(keepRow) : null
  const fillCell = fill ? cellOf(h, fill) : null
  for (const ec of entryColumns) {
    const kept = ec.per.map((c, j) => ({ c: keepRow.map((i) => c[i]), j })).filter((x) => keepCol[x.j])
    const anyMissing = kept.some((x) => x.c.some(isMissing))
    const dts = ec.perDt
    let blockDT: DT = commonDT(dts)
    if (anyMissing) blockDT = withMissing(blockDT === 'bool' ? 'object' : blockDT, true)
    for (const { c, j } of kept) {
      const filled = fillCell !== null ? c.map((x) => (isMissing(x) ? fillCell : x)) : c
      const dt = fillCell !== null && blockDT === 'float64' && filled.every((x) => typeof x === 'number' && Number.isInteger(x)) && commonDT(dts) === 'int64' ? 'int64' : blockDT
      columnsOut.push({ top: ec.e.label, val: ec.e.value, sub: j, series: new Series(filled, index, null, dt) })
    }
  }
  const colArrays = ck.map((s) => columnsOut.map((x) => s.values[colReps[x.sub]]))
  const colDts = ck.map((s) => s.dtype)
  const colNames = ck.map((s) => s.name)
  const hasTop = hasFuncLevel && columnsOut.some((x) => x.top !== null)
  // sort_index(axis=1): by value, then function, then column key; a list of functions stays in the order given
  const sortKeysCols: Cell[][] = []
  if (hasTop && funcFirst) sortKeysCols.push(columnsOut.map((x) => entries.findIndex((e) => e.label === x.top)))
  sortKeysCols.push(columnsOut.map((x) => x.val))
  if (hasTop && !funcFirst) sortKeysCols.push(columnsOut.map((x) => x.top))
  sortKeysCols.push(columnsOut.map((x) => x.sub))
  const order = sortKeys ? sortOrder(sortKeysCols, sortKeysCols.map(() => true)) : rows(columnsOut.length)
  const take = <T>(a: T[]) => order.map((i) => a[i])
  let columns: IndexObj
  const valLevel = keepValueLevel
  const levels: { cells: Cell[]; name: Cell; dt: DT }[] = []
  if (hasTop && funcFirst) levels.push({ cells: take(columnsOut.map((x) => x.top)), name: null, dt: 'object' })
  if (valLevel || (!ck.length && !hasTop) || (hasTop && !ck.length)) levels.push({ cells: take(columnsOut.map((x) => x.val)), name: null, dt: inferDT(columnsOut.map((x) => x.val)) })
  if (hasTop && !funcFirst) levels.push({ cells: take(columnsOut.map((x) => x.top)), name: null, dt: 'object' })
  colArrays.forEach((a, j) => levels.push({ cells: take(a), name: colNames[j], dt: colDts[j] }))
  if (levels.length === 1) columns = new IndexObj(levels[0].cells, levels[0].name, levels[0].dt)
  else if (levels.length) columns = IndexObj.fromArrays(levels.map((l) => l.cells), levels.map((l) => l.name), levels.map((l) => l.dt))
  else columns = new IndexObj([])
  let out = new Frame(columns, take(columnsOut.map((x) => x.series)), index)
  if (margins) out = addMargins(h, out, f, rk, ck, entries, gb, marginName, rowOf, colOf, rowReps, colReps, keepRow, keepCol, keepValueLevel)
  return out
}

/** margins=True: an extra row and column aggregating everything in that row / column */
const addMargins = (h: Host, table: Frame, f: Frame, rk: Series[], ck: Series[], entries: { label: Cell | null; value: Cell; f: Value }[], gb: GroupBy, name: Cell, rowOf: (r: number) => string, colOf: (r: number) => string, rowReps: number[], colReps: number[], keepRow: number[], keepCol: boolean[], keepValueLevel: boolean): Frame => {
  if (entries.some((e) => e.label !== null) || rk.length > 1 || ck.length > 1) throw h.err('NotImplementedError', 'pivot_table(margins=True) takes a single aggfunc, index and columns in this sandbox')
  const all = rows(f.nrows)
  const rowsKept = keepRow.map((i) => rowReps[i])
  const statFor = (e: (typeof entries)[0], members: number[][]) => statOf(h, gb, f.col(e.value)!, members, e.f)
  const labels = table.index.labels
  const index = rk.length ? new IndexObj([...labels, name], table.index.name, table.index.dt === 'object' ? 'object' : 'object') : table.index
  const data: Series[] = []
  let cursor = 0
  const colLabels: Cell[][] = table.columns.multi ? table.columns.multi.arrays.map((a) => [...a]) : [[...table.columns.labels]]
  const newCols: Series[] = []
  for (const e of entries) {
    const colsHere = colReps.map((c, j) => ({ c, j })).filter((x) => keepCol[x.j])
    const bottom = ck.length ? statFor(e, colsHere.map(({ c }) => all.filter((x) => colOf(x) === colOf(c)))) : statFor(e, [all])
    colsHere.forEach((_, j) => {
      const src = table.data[cursor + j]
      const raw = bottom.cells[j]
      const dt = commonDT([src.dtype, bottom.dt === 'int64' && Number.isInteger(raw) ? 'int64' : bottom.dt])
      data.push(new Series([...src.values, raw], index, null, dt))
    })
    cursor += colsHere.length
    if (ck.length) {
      const side = statFor(e, rowsKept.map((r) => all.filter((x) => rowOf(x) === rowOf(r))))
      const grand = statFor(e, [all])
      const sideCells = [...side.cells, grand.cells[0]]
      const dt = sideCells.every((x) => typeof x === 'number' && Number.isInteger(x)) && f.col(e.value)!.dtype === 'int64' ? 'int64' : commonDT([side.dt, grand.dt])
      newCols.push(new Series(sideCells, index, null, dt))
    }
  }
  if (!ck.length) return new Frame(table.columns, data, index)
  const stride = table.ncols / entries.length
  const finalData: Series[] = []
  const finalCols: Cell[][] = colLabels.map(() => [])
  entries.forEach((e, i) => {
    for (let j = 0; j < stride; j++) {
      finalData.push(data[i * stride + j])
      colLabels.forEach((a, lv) => finalCols[lv].push(a[i * stride + j]))
    }
    finalData.push(newCols[i])
    if (colLabels.length > 1) {
      finalCols[0].push(e.value)
      for (let lv = 1; lv < colLabels.length; lv++) finalCols[lv].push(name)
    } else finalCols[0].push(name)
  })
  void keepValueLevel
  const cols = colLabels.length > 1 ? IndexObj.fromArrays(finalCols, table.columns.names, table.columns.multi!.dts.map(() => 'object')) : new IndexObj(finalCols[0], table.columns.name, 'object')
  return new Frame(cols, finalData, index)
}

export const crosstab = (h: Host, args: Value[], kw: Record<string, Value>): Frame => {
  for (const k of Object.keys(kw)) if (!['index', 'columns', 'values', 'rownames', 'colnames', 'aggfunc', 'margins', 'margins_name', 'dropna', 'normalize'].includes(k)) throw h.err('TypeError', `crosstab() got an unexpected keyword argument '${k}'`)
  const a = args[0] ?? kw.index
  const b = args[1] ?? kw.columns
  if (!a || !b) throw h.err('TypeError', "crosstab() missing 2 required positional arguments: 'index' and 'columns'")
  const valuesV = given(args[2] ?? kw.values)
  const aggV = given(kw.aggfunc)
  if (valuesV && !aggV) throw h.err('ValueError', 'values cannot be used without an aggfunc.')
  if (aggV && !valuesV) throw h.err('ValueError', 'aggfunc cannot be used without values.')
  const asList = (v: Value) => (v.k === 'list' && v.items.length && isListLike(v.items[0]) ? v.items : v.k === 'tuple' ? v.items : [v])
  const seriesOf = (v: Value): Series => {
    if (v.k === 'obj' && v.o instanceof Series) return v.o
    const t = typedCells(h, v)
    return new Series(t.cells, null, null, t.dt)
  }
  const ra = asList(a).map(seriesOf)
  const rb = asList(b).map(seriesOf)
  const rn = given(kw.rownames) ? cellsOf(h, kw.rownames) : null
  const cn = given(kw.colnames) ? cellsOf(h, kw.colnames) : null
  const named = (ss: Series[], names: Cell[] | null, pre: string) => ss.map((s, i) => (names ? names[i] : s.name ?? `${pre}_${i}`))
  const rNames = named(ra, rn, 'row')
  const cNames = named(rb, cn, 'col')
  const n = ra[0].length
  const labels = [...rNames, ...cNames]
  const data = [...ra, ...rb].map((s, i) => {
    const out = new Series(s.values, null, labels[i], s.dtype)
    out.cats = s.cats
    return out
  })
  const dummy = '__dummy__'
  const dropnaFlag = flag(h, kw.dropna, true)
  const margins = flag(h, kw.margins)
  const marginName = given(kw.margins_name) ? cellOf(h, kw.margins_name) : 'All'
  const sum = (x: Series) => x.values.reduce<number>((acc, c) => acc + numCell(c), 0)
  const countFn: Value = { k: 'fn', name: 'len', call: (fa) => py.int(seriesIn(fa[0]).length) }
  const src = new Frame([...labels, dummy], [...data, valuesV ? seriesOf(valuesV) : new Series(rows(n).map(() => 1), null, dummy, 'int64')], null)
  const fnV = aggV ?? countFn
  void sum
  const table = pivotTable(h, src, { index: py.list(rNames.map((x) => valueOf(x, 'object'))), columns: py.list(cNames.map((x) => valueOf(x, 'object'))), values: py.str(dummy), aggfunc: fnV, margins: py.bool(margins), margins_name: valueOf(marginName, 'object'), dropna: py.bool(dropnaFlag), ...(aggV ? {} : { fill_value: py.int(0) }) })
  const norm = kw.normalize ? (kw.normalize.k === 'str' ? kw.normalize.v : kw.normalize.k === 'int' ? (kw.normalize.v === 0 ? 'index' : kw.normalize.v === 1 ? 'columns' : '') : h.truthy(kw.normalize) ? 'all' : '') : ''
  if (kw.normalize && kw.normalize.k === 'str' && !['all', 'index', 'columns'].includes(kw.normalize.v)) throw h.err('ValueError', 'Not a valid normalize argument')
  if (!norm) return table
  return normalizeTable(table, norm, margins)
}

const normalizeTable = (t: Frame, how: string, margins: boolean): Frame => {
  const nr = t.nrows - (margins ? 1 : 0)
  const nc = t.ncols - (margins ? 1 : 0)
  const grid = rows(nc).map((j) => rows(nr).map((i) => numCell(t.data[j].values[i])))
  const rowSum = rows(nr).map((i) => rows(nc).reduce((a, j) => a + grid[j][i], 0))
  const colSum = rows(nc).map((j) => rows(nr).reduce((a, i) => a + grid[j][i], 0))
  const total = rowSum.reduce((a, b) => a + b, 0)
  const core = (i: number, j: number) => {
    const x = how === 'index' ? grid[j][i] / rowSum[i] : how === 'columns' ? grid[j][i] / colSum[j] : grid[j][i] / total
    return Number.isNaN(x) ? 0 : x
  }
  if (!margins) return new Frame(t.columns, rows(nc).map((j) => new Series(rows(nr).map((i) => core(i, j)), t.index, null, 'float64')), t.index)
  const lastCol = t.data[nc].values.map(numCell)
  const lastRow = rows(nc).map((j) => numCell(t.data[j].values[nr]))
  const lcSum = rows(nr).reduce((a, i) => a + lastCol[i], 0)
  const lrSum = lastRow.reduce((a, b) => a + b, 0)
  const data: Series[] = []
  for (let j = 0; j < nc; j++) {
    const vals = rows(nr).map((i) => core(i, j))
    if (how === 'index' || how === 'all') vals.push(lastRow[j] / lrSum)
    if (how === 'columns') {
      data.push(new Series(vals, t.index.pick(rows(nr)), null, 'float64'))
      continue
    }
    data.push(new Series(vals, t.index, null, 'float64'))
  }
  if (how === 'index') return new Frame(t.columns.pick(rows(nc), true), data, t.index)
  const side = rows(nr).map((i) => lastCol[i] / lcSum)
  if (how === 'columns') return new Frame(t.columns, [...data, new Series(side, t.index.pick(rows(nr)), null, 'float64')], t.index.pick(rows(nr)))
  return new Frame(t.columns, [...data, new Series([...side, 1], t.index, null, 'float64')], t.index)
}

export const meltFrame = (h: Host, f: Frame, p: Params): Frame => {
  if (f.columns.multi) throw h.err('NotImplementedError', 'melt with MultiIndex columns is not available in this sandbox')
  if (given(p.col_level)) throw h.err('NotImplementedError', 'melt(col_level=...) is not available in this sandbox')
  const listOf = (v: Value | undefined): Cell[] | null => (given(v) ? (isListLike(v!) ? cellsOf(h, v!) : [cellOf(h, v!)]) : null)
  const ids = listOf(p.id_vars) ?? []
  const vals = listOf(p.value_vars)
  const missing = [...ids, ...(vals ?? [])].filter((c) => !f.columns.has(c))
  if (missing.length) throw h.err('KeyError', `"The following id_vars or value_vars are not present in the DataFrame: [${missing.map((c) => (typeof c === 'string' ? `'${c}'` : labelStr(c))).join(', ')}]"`)
  const valueCols = vals ?? f.columns.labels.filter((c) => !ids.includes(c))
  const varName = given(p.var_name) ? cellOf(h, p.var_name!) : f.columns.name ?? 'variable'
  const valueName = given(p.value_name) ? cellOf(h, p.value_name!) : 'value'
  if (f.columns.has(valueName)) throw h.err('ValueError', `value_name (${labelStr(valueName)}) cannot match an element in the DataFrame columns.`)
  const n = f.nrows
  const m = valueCols.length
  const idData = ids.map((c) => {
    const s = f.col(c)!
    return new Series(Array.from({ length: m }, () => s.values).flat(), null, c, s.dtype)
  })
  const varCol = new Series(valueCols.flatMap((c) => rows(n).map(() => c)), null, varName, f.columns.dt)
  const parts = valueCols.map((c) => f.col(c)!)
  const dt = m ? commonDT(parts.map((s) => s.dtype)) : 'float64'
  const valCol = new Series(parts.flatMap((s) => convertCells(s.values, s.dtype, dt)), null, valueName, dt)
  const ignore = flag(h, p.ignore_index, true)
  const index = ignore ? IndexObj.range(n * m) : f.index.pick(Array.from({ length: m }, () => rows(n)).flat())
  return new Frame([...ids, varName, valueName], [...idData, varCol, valCol], index)
}

// ---------------------------------------------------------------- stack / unstack

const levelNumber = (h: Host, ix: IndexObj, v: Value | undefined): number => {
  if (!given(v)) return ix.nlevels - 1
  const c = cellOf(h, v!)
  if (typeof c === 'number' && !ix.names.includes(c)) {
    const j = c < 0 ? c + ix.nlevels : c
    if (j < 0 || j >= ix.nlevels) throw h.err('IndexError', `Too many levels: Index has only ${ix.nlevels} level${ix.nlevels > 1 ? 's' : ''}, not ${c + 1}`)
    return j
  }
  const j = ix.names.indexOf(c)
  if (j < 0) throw h.err('KeyError', `'Level ${labelStr(c)} not found'`)
  return j
}

export const stackFrame = (h: Host, f: Frame, p: Params): Series | Frame => {
  if (p.sort && !h.truthy(p.sort)) throw h.err('NotImplementedError', 'stack(sort=False) is not available in this sandbox')
  const dropna = flag(h, p.future_stack) ? false : p.dropna ? h.truthy(p.dropna) : true
  if (f.columns.multi) {
    const j = levelNumber(h, f.columns, p.level)
    if (f.columns.nlevels !== 2) throw h.err('NotImplementedError', 'stack of more than two column levels is not available in this sandbox')
    const m = f.columns.multi
    const other = 1 - j
    const topLabels = sortedUniq(m.arrays[other])
    const subLabels = sortedUniq(m.arrays[j])
    const rowsOut: { r: number; sub: Cell }[] = []
    for (const r of rows(f.nrows)) for (const sub of subLabels) rowsOut.push({ r, sub })
    const colAt = (top: Cell, sub: Cell) => rows(f.ncols).find((k) => m.arrays[other][k] === top && m.arrays[j][k] === sub)
    const keepRow = rowsOut.filter(({ r, sub }) => !dropna || topLabels.some((t) => {
      const k = colAt(t, sub)
      return k !== undefined && !isMissing(f.data[k].values[r])
    }))
    const base = f.index.nlevels
    const arrays = [...Array.from({ length: base }, (_, l) => keepRow.map(({ r }) => f.index.level(l).labels[r])), keepRow.map((x) => x.sub)]
    const index = IndexObj.fromArrays(arrays, [...f.index.names, m.names[j]], [...Array.from({ length: base }, (_, l) => f.index.level(l).dt), m.dts[j]])
    return new Frame(new IndexObj(topLabels, m.names[other], m.dts[other]), topLabels.map((t) => {
      const cells = keepRow.map(({ r, sub }) => {
        const k = colAt(t, sub)
        return k === undefined ? NaN : f.data[k].values[r]
      })
      const dts = rows(f.ncols).filter((k) => m.arrays[other][k] === t).map((k) => f.data[k].dtype)
      return new Series(cells, index, t, withMissing(commonDT(dts), cells.some(isMissing)))
    }), index)
  }
  const dt = commonDT(f.data.map((s) => s.dtype))
  const cells: Cell[] = []
  const at: number[] = []
  const cols: Cell[] = []
  for (const r of rows(f.nrows)) {
    f.data.forEach((s, k) => {
      const c = s.values[r]
      if (dropna && isMissing(c)) return
      cells.push(dt === 'float64' ? numCell(c) : dt === 'object' && s.dtype === 'float64' ? convertCells([c], 'float64', 'object')[0] : c)
      at.push(r)
      cols.push(f.columns.labels[k])
    })
  }
  const base = f.index.nlevels
  const arrays = [...Array.from({ length: base }, (_, l) => at.map((r) => f.index.level(l).labels[r])), cols]
  const index = IndexObj.fromArrays(arrays, [...f.index.names, f.columns.name], [...Array.from({ length: base }, (_, l) => f.index.level(l).dt), f.columns.dt])
  return new Series(cells, index, null, dt)
}

const unstackGrid = (h: Host, ix: IndexObj, values: { cells: Cell[]; dt: DT; name: Cell }[], level: number, fill: Value | undefined): { index: IndexObj; columnLabels: Cell[]; colName: Cell; colDT: DT; columns: { val: number; label: Cell; cells: Cell[]; dt: DT }[] } => {
  const m = ix.multi!
  const restLevels = m.arrays.map((_, j) => j).filter((j) => j !== level)
  const colLabels = sortedUniq(m.arrays[level].filter((c) => !isMissing(c)))
  const rowKey = (r: number) => restLevels.map((j) => keyStr(m.arrays[j][r])).join('\0')
  const reps = new Map<string, number>()
  rows(ix.length).forEach((r) => { if (!reps.has(rowKey(r))) reps.set(rowKey(r), r) })
  const list = [...reps.values()]
  const order = sortOrder(restLevels.map((j) => list.map((r) => m.arrays[j][r])), restLevels.map(() => true)).map((i) => list[i])
  const index = restLevels.length === 1 ? new IndexObj(order.map((r) => m.arrays[restLevels[0]][r]), m.names[restLevels[0]], m.dts[restLevels[0]]) : IndexObj.fromArrays(restLevels.map((j) => order.map((r) => m.arrays[j][r])), restLevels.map((j) => m.names[j]), restLevels.map((j) => m.dts[j]))
  const lookup = new Map<string, number>()
  rows(ix.length).forEach((r) => {
    const k = `${rowKey(r)}\u0001${keyStr(m.arrays[level][r])}`
    if (lookup.has(k)) throw h.err('ValueError', 'Index contains duplicate entries, cannot reshape')
    lookup.set(k, r)
  })
  const fc = fill ? cellOf(h, fill) : null
  const columns = values.flatMap((v, vi) => {
    const per = colLabels.map((l) => order.map((rr) => lookup.get(`${rowKey(rr)}\u0001${keyStr(l)}`) ?? -1))
    const anyMissing = per.some((a) => a.some((i) => i < 0))
    const dt = fc !== null ? commonDT([v.dt, inferDT([fc])]) : withMissing(v.dt, anyMissing)
    const base = convertCells(v.cells, v.dt, dt)
    return colLabels.map((l, j) => ({ val: vi, label: l, cells: per[j].map((i) => (i < 0 ? fc ?? NaN : base[i])), dt }))
  })
  return { index, columnLabels: colLabels, colName: m.names[level], colDT: m.dts[level], columns }
}

export const unstackSeries = (h: Host, s: Series, p: Params): Frame => {
  if (!s.index.multi) throw h.err('ValueError', `index must be a MultiIndex to unstack, <class 'pandas.core.indexes.${s.index.isRange ? 'range.RangeIndex' : 'base.Index'}'> was passed`)
  if (given(p.sort) && !h.truthy(p.sort!)) throw h.err('NotImplementedError', 'unstack(sort=False) is not available in this sandbox')
  const g = unstackGrid(h, s.index, [{ cells: s.values, dt: s.dtype, name: s.name }], levelNumber(h, s.index, p.level), given(p.fill_value))
  return new Frame(new IndexObj(g.columnLabels, g.colName, g.colDT), g.columns.map((c) => new Series(c.cells, g.index, c.label, c.dt)), g.index)
}

export const unstackFrame = (h: Host, f: Frame, p: Params): Series | Frame => {
  if (given(p.sort) && !h.truthy(p.sort!)) throw h.err('NotImplementedError', 'unstack(sort=False) is not available in this sandbox')
  if (!f.index.multi) {
    const dt = commonDT(f.data.map((s) => s.dtype))
    const cells = f.data.flatMap((s) => convertCells(s.values, s.dtype, dt))
    const index = IndexObj.fromArrays([f.columns.labels.flatMap((c) => f.index.labels.map(() => c)), f.columns.labels.flatMap(() => f.index.labels)], [f.columns.name, f.index.name], [f.columns.dt, f.index.dt])
    return new Series(cells, index, null, dt)
  }
  const g = unstackGrid(h, f.index, f.data.map((s) => ({ cells: s.values, dt: s.dtype, name: s.name })), levelNumber(h, f.index, p.level), given(p.fill_value))
  const cols = IndexObj.fromArrays([g.columns.map((c) => f.columns.labels[c.val]), g.columns.map((c) => c.label)], [f.columns.name, g.colName], [f.columns.dt, g.colDT])
  return new Frame(cols, g.columns.map((c) => new Series(c.cells, g.index, null, c.dt)), g.index)
}

// ---------------------------------------------------------------- dummies

export const getDummies = (h: Host, args: Value[], kw: Record<string, Value>): Frame => {
  for (const k of Object.keys(kw)) if (!['data', 'prefix', 'prefix_sep', 'dummy_na', 'columns', 'sparse', 'drop_first', 'dtype'].includes(k)) throw h.err('TypeError', `get_dummies() got an unexpected keyword argument '${k}'`)
  if (flag(h, kw.sparse)) throw h.err('NotImplementedError', 'get_dummies(sparse=True) is not available in this sandbox')
  const d = args[0] ?? kw.data
  const prefixV = given(args[1] ?? kw.prefix)
  const sep = given(kw.prefix_sep) ? h.str(kw.prefix_sep) : '_'
  const dummyNa = flag(h, kw.dummy_na)
  const dropFirst = flag(h, kw.drop_first)
  const typ = given(kw.dtype) ? typeArg(h, kw.dtype) : 'bool'
  const isFloat = typ.startsWith('float')
  const isBool = typ === 'bool'
  if (!isBool && !isFloat && !typ.startsWith('int') && !typ.startsWith('uint')) throw h.err('NotImplementedError', `get_dummies(dtype='${typ}') is not available in this sandbox`)
  const dummies = (s: Series, prefix: string | null, seen: Cell[]) => {
    const present = s.values.filter((c) => !isMissing(c))
    const cats = s.cats ? s.cats.labels : sortedUniq(present)
    const all = dropFirst ? cats.slice(1) : cats
    const parts = all.map((c) => ({ name: prefix !== null ? `${prefix}${sep}${labelStr(c)}` : c, vals: s.values.map((v) => (isBool ? keyStr(v) === keyStr(c) : Number(keyStr(v) === keyStr(c)))) }))
    if (dummyNa) parts.push({ name: prefix !== null ? `${prefix}${sep}nan` : NaN as Cell, vals: s.values.map((v) => (isBool ? isMissing(v) : Number(isMissing(v)))) })
    void seen
    return parts.map((x) => ({ name: x.name, s: new Series(x.vals, s.index, null, isBool ? 'bool' : isFloat ? 'float64' : 'int64') }))
  }
  if (d.k === 'obj' && d.o instanceof Frame) {
    const f = d.o
    const which = given(kw.columns) ? cellsOf(h, kw.columns) : f.columns.labels.filter((_, k) => f.data[k].dtype === 'object' || !!f.data[k].cats)
    const missing = which.filter((c) => !f.columns.has(c))
    if (missing.length) throw h.err('KeyError', `${new IndexObj(missing).repr()}`)
    const prefixOf = (c: Cell, i: number): string => {
      if (!prefixV) return labelStr(c)
      if (prefixV.k === 'str') return prefixV.v
      if (prefixV.k === 'dict') {
        const v = prefixV.d.get(valueOf(c, 'object'))
        return v ? h.str(v) : labelStr(c)
      }
      const items = h.iterate(prefixV)
      if (items.length !== which.length) throw h.err('ValueError', `Length of 'prefix' (${items.length}) did not match the length of the columns being encoded (${which.length}).`)
      return h.str(items[i])
    }
    const labels: Cell[] = []
    const data: Series[] = []
    f.columns.labels.forEach((c, k) => {
      if (!which.includes(c)) {
        labels.push(c)
        data.push(f.data[k].copy())
      }
    })
    which.forEach((c, i) => {
      for (const x of dummies(f.col(c)!, prefixOf(c, i), [])) {
        labels.push(x.name)
        data.push(x.s)
      }
    })
    return new Frame(labels, data, f.index)
  }
  const s = d.k === 'obj' && d.o instanceof Series ? d.o : new Series(typedCells(h, d).cells, null, null, typedCells(h, d).dt)
  const parts = dummies(s, prefixV ? h.str(prefixV) : null, [])
  return new Frame(parts.map((x) => x.name), parts.map((x) => x.s), s.index)
}

// ---------------------------------------------------------------- cut / qcut

/** pandas' _round_frac: round an interval edge to `precision` significant decimals */
const roundFrac = (x: number, precision: number): number => {
  if (!Number.isFinite(x) || x === 0) return x
  const whole = Math.trunc(x)
  const frac = x - whole
  const digits = whole === 0 ? -Math.floor(Math.log10(Math.abs(frac))) - 1 + precision : precision
  return npRound(x, digits)
}

const npRound = (v: number, d: number) => {
  const rint = (x: number) => {
    const fl = Math.floor(x)
    const diff = x - fl
    return diff < 0.5 ? fl : diff > 0.5 ? fl + 1 : fl % 2 === 0 ? fl : fl + 1
  }
  if (d >= 0) {
    const f = 10 ** d
    return rint(v * f) / f
  }
  const f = 10 ** -d
  return rint(v / f) * f
}

const inferPrecision = (base: number, bins: number[]): number => {
  for (let p = base; p < 20; p++) {
    if (new Set(bins.map((b) => roundFrac(b, p))).size === bins.length) return p
  }
  return base
}

const intervalLabels = (edges: number[], intEdges: boolean, precision: number, right: boolean, lowest: boolean): string[] => {
  const pr = intEdges ? precision : inferPrecision(precision, edges)
  const breaks = edges.map((e) => (intEdges ? e : roundFrac(e, pr)))
  if (right && lowest) breaks[0] -= 10 ** -pr
  const str = (x: number) => (intEdges && Number.isInteger(x) ? String(x) : pyFloatRepr(x))
  const out = intEdges && right && lowest ? breaks.map((x) => pyFloatRepr(x)) : breaks.map(str)
  return Array.from({ length: edges.length - 1 }, (_, i) => (right ? `(${out[i]}, ${out[i + 1]}]` : `[${out[i]}, ${out[i + 1]})`))
}

const binned = (h: Host, src: Series, xv: Value, edges: number[], intEdges: boolean, lv: Value | undefined, p: { right: boolean; lowest: boolean; precision: number; ordered: boolean; retbins: boolean; dupes: string }, api: string): Value => {
  const v = src.values.map(numCell)
  let es = edges
  if (new Set(es).size !== es.length) {
    if (p.dupes === 'raise') throw h.err('ValueError', `Bin edges must be unique: ${new IndexObj(es, null, intEdges ? 'int64' : 'float64').repr()}.\nYou can drop duplicate edges by setting the 'duplicates' kwarg`)
    es = [...new Set(es)]
  }
  const nb = es.length - 1
  const useLabels = lv && lv.k !== 'none' && !(lv.k === 'bool' && !lv.v)
  const noLabels = lv && lv.k === 'bool' && !lv.v
  const labels: Cell[] = useLabels ? typedCells(h, lv!).cells : intervalLabels(es, intEdges, p.precision, p.right, p.lowest)
  if (useLabels && labels.length !== nb) throw h.err('ValueError', 'Bin labels must be one fewer than the number of bin edges')
  const at = v.map((x) => {
    if (Number.isNaN(x)) return -1
    if (p.lowest && p.right && x === es[0]) return 0
    if (p.lowest && !p.right && x === es[nb]) return nb - 1
    return es.findIndex((e, k) => k < nb && (p.right ? x > e && x <= es[k + 1] : x >= e && x < es[k + 1]))
  })
  let out: Series
  if (noLabels) {
    const missing = at.some((i) => i < 0)
    out = new Series(at.map((i) => (i < 0 ? NaN : i)), src.index, src.name, missing ? 'float64' : 'int64')
  } else {
    out = new Series(at.map((i) => (i < 0 ? NaN : labels[i])), src.index, src.name, 'object')
    const allStrings = labels.every((l) => typeof l === 'string')
    const cats: Cats = { labels, kind: useLabels ? (allStrings ? 'object' : inferDT(labels)) : `interval[${intEdges ? (p.lowest && p.right ? 'float64' : 'int64') : 'float64'}, ${p.right ? 'right' : 'left'}]`, ordered: useLabels ? p.ordered : true }
    out.cats = cats
    if (useLabels && !allStrings) out.dtype = inferDT(labels)
    out.bare = !(xv.k === 'obj' && xv.o instanceof Series)
  }
  traceFrame(h, api === 'cut' ? 'cut' : 'qcut', () => ({ inputs: [{ label: h.nameOf(xv, 'x'), table: tableOfSeries(src) }], result: tableOfSeries(out) }))
  if (p.retbins) return py.tuple([py.obj(out), py.arr(NDArray.create(es, [es.length], intEdges ? 'int64' : 'float64'))])
  return py.obj(out)
}

const seriesArg = (h: Host, xv: Value): Series => {
  if (xv.k === 'obj' && xv.o instanceof Series) return xv.o
  const t = typedCells(h, xv)
  return new Series(t.cells, null, null, t.dt)
}

export const cutSeries = (h: Host, args: Value[], kw: Record<string, Value>): Value => {
  for (const k of Object.keys(kw)) if (!['x', 'bins', 'right', 'labels', 'retbins', 'precision', 'include_lowest', 'duplicates', 'ordered'].includes(k)) throw h.err('TypeError', `cut() got an unexpected keyword argument '${k}'`)
  const xv = args[0] ?? kw.x
  const src = seriesArg(h, xv)
  const bv = kw.bins ?? args[1]
  if (bv && (bv.k === 'int' || bv.k === 'float') && src.dtype === 'object' && src.values.some((c) => typeof c === 'string')) throw h.err('TypeError', "bad operand type for abs(): 'str'")
  const v = src.numeric(h, 'cut')
  if (!bv) throw h.err('TypeError', "cut() missing 1 required positional argument: 'bins'")
  const right = !kw.right || h.truthy(kw.right)
  let edges: number[]
  let intEdges = false
  if (bv.k === 'int' || bv.k === 'float') {
    const n = h.toInt(bv)
    if (n < 1) throw h.err('ValueError', '`bins` should be a positive integer.')
    const fin = v.filter(Number.isFinite)
    if (!fin.length) throw h.err('ValueError', 'Cannot cut empty array')
    let lo = Math.min(...fin)
    let hi = Math.max(...fin)
    const same = lo === hi
    if (same) {
      lo -= lo !== 0 ? 0.001 * Math.abs(lo) : 0.001
      hi += hi !== 0 ? 0.001 * Math.abs(hi) : 0.001
    }
    const step = (hi - lo) / n
    edges = Array.from({ length: n + 1 }, (_, i) => (i === n ? hi : i * step + lo))
    if (!same) {
      const adj = (hi - lo) * 0.001
      if (right) edges[0] -= adj
      else edges[n] += adj
    }
  } else {
    const t = typedCells(h, bv)
    edges = t.cells.map(numCell)
    intEdges = t.dt === 'int64'
    if (edges.some((e, i) => i > 0 && e < edges[i - 1])) throw h.err('ValueError', 'bins must increase monotonically.')
  }
  const dupes = given(kw.duplicates) ? h.str(kw.duplicates) : 'raise'
  return binned(h, src, xv, edges, intEdges, kw.labels, { right, lowest: flag(h, kw.include_lowest), precision: given(kw.precision) ? h.toInt(kw.precision) : 3, ordered: flag(h, kw.ordered, true), retbins: flag(h, kw.retbins), dupes }, 'cut')
}

export const qcutSeries = (h: Host, args: Value[], kw: Record<string, Value>): Value => {
  for (const k of Object.keys(kw)) if (!['x', 'q', 'labels', 'retbins', 'precision', 'duplicates'].includes(k)) throw h.err('TypeError', `qcut() got an unexpected keyword argument '${k}'`)
  const xv = args[0] ?? kw.x
  const src = seriesArg(h, xv)
  const v = src.numeric(h, 'qcut')
  const qv = kw.q ?? args[1]
  if (!qv) throw h.err('TypeError', "qcut() missing 1 required positional argument: 'q'")
  const n = qv.k === 'int' ? qv.v : null
  const qs = n !== null ? Array.from({ length: n + 1 }, (_, i) => (i === n ? 1 : i * (1 / n))) : typedCells(h, qv).cells.map(numCell)
  const sorted = v.filter((x) => !Number.isNaN(x)).sort((a, b) => a - b)
  const edges = qs.map((q) => npQuantile(sorted, q))
  const dupes = given(kw.duplicates) ? h.str(kw.duplicates) : 'raise'
  if (dupes !== 'raise' && dupes !== 'drop') throw h.err('ValueError', "invalid value for 'duplicates' parameter, valid options are: raise, drop")
  return binned(h, src, xv, edges, false, kw.labels, { right: true, lowest: true, precision: given(kw.precision) ? h.toInt(kw.precision) : 3, ordered: true, retbins: flag(h, kw.retbins), dupes }, 'qcut')
}

