/** Text in and out: read_csv, read_json, json_normalize, to_csv, to_json, through the sandbox's in-memory files or a file-like object. */
import { PyDict, py } from './minipy'
import type { Host, Value } from './minipy'
import { Frame, IndexObj, Series, cellOf, cellsOf, inferDT, inferValues, isListLike, isMissing, labelStr, numCell, valueOf } from './pyPandas'
import type { Cell, DT, Params } from './pyPandas'
import { PyCell, fixedStr } from './pyPandasFormat'
import { repr } from './minipy'
import { astypeCol, flag, given, typeArg } from './pyPandasMethods'

/** the run's in-memory file system (pyStdlib.ts): a normalised path, relative ones under /sandbox */
const sandboxPath = (p: string) => {
  const parts: string[] = []
  for (const part of (p.startsWith('/') ? p : `/sandbox/${p}`).split('/')) {
    if (part === '..') parts.pop()
    else if (part && part !== '.') parts.push(part)
  }
  return `/${parts.join('/')}`
}

const methodOf = (h: Host, v: Value, name: string): Value | undefined => {
  if (v.k === 'obj') return v.o.getAttr?.(name, h)
  if (v.k !== 'inst') return undefined
  return h.getAttr(v, name)
}

/** the text behind a path or anything with a read method; undefined if the path holds no file */
export const textFrom = (h: Host, src: Value): string | undefined => {
  if (src.k === 'str') return (h.state.get('files') as Map<string, string> | undefined)?.get(sandboxPath(src.v))
  const read = methodOf(h, src, 'read')
  if (!read) throw h.err('ValueError', `Invalid file path or buffer object type: <class '${src.k === 'obj' ? src.o.cls : src.k}'>`)
  return h.str(h.call(read, []))
}

const NA_STRINGS = new Set(['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null'])

/** split CSV text into rows, honouring quotes */
const parseCsv = (text: string, sep: string, quote: string, skipSpace: boolean): { fields: string[]; quoted: boolean[]; line: number }[] => {
  const out: { fields: string[]; quoted: boolean[]; line: number }[] = []
  let lineNo = 1
  let rowLine = 1
  let fields: string[] = []
  let quoted: boolean[] = []
  let cur = ''
  let inQuote = false
  let wasQuoted = false
  let atStart = true
  const endField = () => {
    fields.push(cur)
    quoted.push(wasQuoted)
    cur = ''
    wasQuoted = false
    atStart = true
  }
  const endRow = () => {
    if (fields.length || cur.length || wasQuoted) {
      endField()
      out.push({ fields, quoted, line: rowLine })
    }
    fields = []
    quoted = []
    rowLine = lineNo
  }
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (inQuote) {
      if (c === quote) {
        if (text[i + 1] === quote) {
          cur += quote
          i++
        } else inQuote = false
      } else cur += c
    } else if (c === quote && atStart) {
      inQuote = true
      wasQuoted = true
      atStart = false
    } else if (text.startsWith(sep, i)) {
      endField()
      i += sep.length - 1
    } else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++
      lineNo++
      endRow()
    } else {
      if (skipSpace && atStart && c === ' ') continue
      cur += c
      atStart = false
    }
  }
  endRow()
  return out
}

const INT_RE = /^\s*[+-]?\d+\s*$/
const FLOAT_RE = /^\s*[+-]?(\d+\.?\d*([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?|inf|Inf|INF|infinity|Infinity)\s*$/

/** one column of text to pandas' inferred dtype */
const convertColumn = (raw: (string | null)[], naSet: Set<string>, trues: Set<string>, falses: Set<string>): { cells: Cell[]; dt: DT } => {
  const cells = raw.map((s) => (s === null || naSet.has(s) ? null : s))
  const present = cells.filter((s): s is string => s !== null)
  if (!present.length) return { cells: cells.map(() => NaN), dt: 'float64' }
  if (present.every((s) => INT_RE.test(s))) {
    if (present.length === cells.length) return { cells: cells.map((s) => Number(s)), dt: 'int64' }
    return { cells: cells.map((s) => (s === null ? NaN : Number(s))), dt: 'float64' }
  }
  const asFloat = (s: string) => {
    const t = s.trim().toLowerCase().replace(/^\+/, '')
    if (t === 'inf' || t === 'infinity') return Infinity
    if (t === '-inf' || t === '-infinity') return -Infinity
    return Number(t)
  }
  if (present.every((s) => FLOAT_RE.test(s))) return { cells: cells.map((s) => (s === null ? NaN : asFloat(s))), dt: 'float64' }
  if (present.every((s) => trues.has(s) || falses.has(s))) {
    if (present.length === cells.length) return { cells: cells.map((s) => trues.has(s as string)), dt: 'bool' }
    return { cells: cells.map((s) => (s === null ? NaN : trues.has(s))), dt: 'object' }
  }
  return { cells: cells.map((s) => (s === null ? NaN : s)), dt: 'object' }
}

/** pd.read_csv(io.StringIO(text)) */
export const readCsv = (h: Host, text: string, kw: Record<string, Value>, defaultSep: string): Frame => {
  const known = ['filepath_or_buffer', 'sep', 'delimiter', 'header', 'names', 'index_col', 'usecols', 'dtype', 'true_values', 'false_values', 'skipinitialspace', 'skiprows', 'skipfooter', 'nrows', 'na_values', 'keep_default_na', 'na_filter', 'quotechar', 'encoding', 'engine', 'skip_blank_lines', 'comment', 'low_memory']
  for (const k of Object.keys(kw)) if (!known.includes(k)) throw h.err('NotImplementedError', `read_csv(${k}=...) is not available in this sandbox`)
  const sepV = given(kw.sep) ?? given(kw.delimiter)
  let sep = sepV ? h.str(sepV) : defaultSep
  if (sep === '\\s+') sep = ' '
  const whitespaceSep = sepV && h.str(sepV) === '\\s+'
  const comment = given(kw.comment) ? h.str(kw.comment) : null
  let body = text
  if (comment) body = body.split('\n').map((l) => (l.includes(comment) ? l.slice(0, l.indexOf(comment)) : l)).join('\n')
  if (whitespaceSep) body = body.split('\n').map((l) => l.trim().replace(/\s+/g, ' ')).join('\n')
  const quote = given(kw.quotechar) ? h.str(kw.quotechar) : '"'
  let lines = parseCsv(body, sep, quote, flag(h, kw.skipinitialspace))
  if (flag(h, kw.skip_blank_lines, true)) lines = lines.filter((l) => !(l.fields.length === 1 && l.fields[0] === '' && !l.quoted[0]))
  const skipV = given(kw.skiprows)
  if (skipV) {
    if (skipV.k === 'int') lines = lines.slice(skipV.v)
    else if (isListLike(skipV)) {
      const drop = new Set(cellsOf(h, skipV).map(Number))
      lines = lines.filter((_, i) => !drop.has(i))
    } else throw h.err('NotImplementedError', 'read_csv(skiprows=callable) is not available in this sandbox')
  }
  if (given(kw.skipfooter)) lines = lines.slice(0, lines.length - h.toInt(kw.skipfooter))
  if (!lines.length) throw h.err('EmptyDataError', 'No columns to parse from file')
  const namesV = given(kw.names)
  const headerV = kw.header
  const headerRow = namesV ? (headerV && headerV.k === 'int' ? headerV.v : -1) : headerV && headerV.k === 'none' ? -1 : headerV && headerV.k === 'int' ? headerV.v : 0
  if (headerV && isListLike(headerV)) throw h.err('NotImplementedError', 'read_csv(header=[...]) is not available in this sandbox')
  let head: Cell[]
  let rowsData = lines
  if (headerRow >= 0) {
    if (headerRow >= lines.length) throw h.err('ValueError', `Passed header=${headerRow} but only ${lines.length} lines in file`)
    const seen = new Map<string, number>()
    head = lines[headerRow].fields.map((x, i) => {
      const name = x === '' ? `Unnamed: ${i}` : x
      const n = seen.get(name) ?? 0
      seen.set(name, n + 1)
      return n ? `${name}.${n}` : name
    })
    rowsData = lines.slice(headerRow + 1)
  } else head = []
  const nrowsV = given(kw.nrows)
  if (nrowsV) rowsData = rowsData.slice(0, h.toInt(nrowsV))
  const widest = Math.max(0, ...rowsData.map((r) => r.fields.length))
  const firstWidth = rowsData[0]?.fields.length ?? 0
  if (namesV) head = cellsOf(h, namesV)
  else if (headerRow < 0) head = Array.from({ length: firstWidth }, (_, i) => i)
  // the first data row sets the width: extra leading fields there are the index, a wider row later is an error
  const expected = namesV ? widest : Math.max(firstWidth, head.length)
  const tooWide = rowsData.find((r) => r.fields.length > expected)
  if (tooWide) throw h.err('ParserError', `Error tokenizing data. C error: Expected ${expected} fields in line ${tooWide.line}, saw ${tooWide.fields.length}\n`)
  const keepDefault = flag(h, kw.keep_default_na, true)
  const naSet = new Set<string>(keepDefault ? NA_STRINGS : [])
  const naV = given(kw.na_values)
  const perColNa = new Map<Cell, Set<string>>()
  if (naV) {
    if (naV.k === 'dict') naV.d.items().forEach(([k, v]) => perColNa.set(cellOf(h, k), new Set((isListLike(v) ? cellsOf(h, v) : [cellOf(h, v)]).map(labelStr))))
    else (isListLike(naV) ? cellsOf(h, naV) : [cellOf(h, naV)]).forEach((x) => naSet.add(labelStr(x)))
  }
  const naFilter = flag(h, kw.na_filter, true)
  const tv = new Set(['True', 'TRUE', 'true', ...(given(kw.true_values) ? cellsOf(h, kw.true_values).map(labelStr) : [])])
  const fv = new Set(['False', 'FALSE', 'false', ...(given(kw.false_values) ? cellsOf(h, kw.false_values).map(labelStr) : [])])
  const offset = Math.max(0, expected - head.length)
  const col = (j: number, label: Cell): { cells: Cell[]; dt: DT } => {
    const raw = rowsData.map((r) => r.fields[j] ?? null)
    const set = perColNa.has(label) ? new Set([...naSet, ...perColNa.get(label)!]) : naSet
    const quotedAll = rowsData.map((r) => r.quoted[j])
    const t = convertColumn(raw, naFilter ? set : new Set(), tv, fv)
    if (t.dt === 'object' && quotedAll.length) return t
    return t
  }
  const data = head.map((label, j) => {
    const t = col(j + offset, label)
    return new Series(t.cells, null, label, t.dt)
  })
  let index: IndexObj | null = null
  const dtypeV = given(kw.dtype)
  const typed = data.map((s) => {
    if (!dtypeV) return s
    const want = dtypeV.k === 'dict' ? dtypeV.d.get(valueOf(s.name, 'object')) : dtypeV
    if (!want) return s
    const name = typeArg(h, want)
    if ((name === 'str' || name === 'object') && s.name !== null) {
      const j = head.indexOf(s.name)
      const raw = rowsData.map((r) => r.fields[j + offset] ?? null)
      return new Series(raw.map((x) => (x === null || naSet.has(x) ? NaN : x)), null, s.name, 'object')
    }
    return astypeCol(h, s, name)
  })
  if (offset) {
    const parts = Array.from({ length: offset }, (_, j) => col(j, null))
    index = offset === 1 ? new IndexObj(parts[0].cells, null, parts[0].dt) : IndexObj.fromArrays(parts.map((x) => x.cells), parts.map(() => null), parts.map((x) => x.dt))
  }
  let names = head
  let cols = typed
  const ic = given(kw.index_col)
  if (ic && !(ic.k === 'bool' && !ic.v)) {
    const list = isListLike(ic) ? h.iterate(ic) : [ic]
    const pos = list.map((x) => {
      if (x.k === 'int') return x.v < 0 ? x.v + names.length : x.v
      const at = names.indexOf(cellOf(h, x))
      if (at < 0) throw h.err('ValueError', `Index ${labelStr(cellOf(h, x))} invalid`)
      return at
    })
    const ixs = pos.map((k) => cols[k])
    const unnamed = (c: Cell) => (typeof c === 'string' && /^Unnamed: \d+$/.test(c) ? null : c)
    index = ixs.length === 1 ? new IndexObj(ixs[0].values, unnamed(ixs[0].name), ixs[0].dtype) : IndexObj.fromArrays(ixs.map((s) => s.values), ixs.map((s) => unnamed(s.name)), ixs.map((s) => s.dtype))
    const rest = names.map((_, k) => k).filter((k) => !pos.includes(k))
    names = rest.map((k) => names[k])
    cols = rest.map((k) => cols[k])
  }
  const uc = given(kw.usecols)
  if (uc) {
    if (uc.k === 'func' || uc.k === 'fn') {
      const keep = names.map((_, k) => k).filter((k) => h.truthy(h.call(uc, [valueOf(names[k], 'object')])))
      cols = keep.map((k) => cols[k])
      names = keep.map((k) => names[k])
    } else {
      const want = cellsOf(h, uc)
      const keep = names.map((_, k) => k).filter((k) => want.includes(names[k]) || want.includes(k))
      const miss = want.filter((w) => typeof w === 'string' && !names.includes(w))
      if (miss.length) throw h.err('ValueError', `Usecols do not match columns, columns expected but not found: [${miss.map((x) => `'${labelStr(x)}'`).join(', ')}]`)
      cols = keep.map((k) => cols[k])
      names = keep.map((k) => names[k])
    }
  }
  const allInts = names.every((n, i) => n === i)
  return new Frame(allInts && names.length ? IndexObj.range(names.length) : new IndexObj(names, null, inferDT(names)), cols, index ?? IndexObj.range(rowsData.length))
}

// ---------------------------------------------------------------- to_csv

const csvCell = (c: Cell, dt: DT, na: string, ff: ((x: number) => string) | undefined): string => {
  if (isMissing(c)) return na
  if (dt === 'float64' || (c instanceof PyCell && c.v.k === 'float')) {
    const v = numCell(c)
    return ff ? ff(v) : repr(py.float(v))
  }
  if (typeof c === 'number' && !Number.isInteger(c)) return ff ? ff(c) : repr(py.float(c))
  return labelStr(c)
}

const csvQuote = (s: string, sep: string, quote: string, always: boolean) => (always || s.includes(sep) || s.includes(quote) || s.includes('\n') || s.includes('\r') ? `${quote}${s.split(quote).join(quote + quote)}${quote}` : s)

const csvFloatFormat = (h: Host, v: Value | undefined): ((x: number) => string) | undefined => {
  if (!given(v)) return undefined
  if (v!.k === 'str') {
    const m = /^%\.(\d+)([fe])$/.exec(v!.v)
    if (!m) throw h.err('NotImplementedError', `to_csv(float_format='${v!.v}') is not available in this sandbox`)
    const d = Number(m[1])
    return m[2] === 'f' ? (x) => fixedStr(x, d) : (x) => x.toExponential(d).replace(/e([+-])(\d)$/, 'e$10$2')
  }
  return (x) => h.str(h.call(v!, [py.float(x)]))
}

/** write to a path in the in-memory file system or to anything with a write method; no target returns the text */
const deliver = (h: Host, target: Value | undefined, text: string, mode = 'w'): Value => {
  if (!given(target)) return py.str(text)
  const t = target!
  if (t.k === 'str') {
    let files = h.state.get('files') as Map<string, string> | undefined
    if (!files) h.state.set('files', (files = new Map()))
    const key = sandboxPath(t.v)
    if (mode === 'x' && files.has(key)) throw h.err('FileExistsError', `[Errno 17] File exists: '${t.v}'`)
    files.set(key, mode === 'a' ? (files.get(key) ?? '') + text : text)
    return py.NONE
  }
  const write = methodOf(h, t, 'write')
  if (!write) throw h.err('ValueError', `Invalid file path or buffer object type: <class '${t.k === 'obj' ? t.o.cls : t.k}'>`)
  h.call(write, [py.str(text)])
  return py.NONE
}

const refuseIO = (h: Host, p: Params, fn: string, names: string[]) => {
  for (const k of names) if (given(p[k])) throw h.err('NotImplementedError', `${fn}(${k}=...) is not available in this sandbox`)
}

const tableToCsv = (h: Host, f: Frame, p: Params, headerDefault: boolean): string => {
  refuseIO(h, p, 'to_csv', ['compression'])
  const sep = given(p.sep) ? h.str(p.sep!) : ','
  const na = given(p.na_rep) ? h.str(p.na_rep!) : ''
  const ff = csvFloatFormat(h, p.float_format)
  const quote = given(p.quotechar) ? h.str(p.quotechar!) : '"'
  const eol = given(p.lineterminator) ? h.str(p.lineterminator!) : '\n'
  const qmode = given(p.quoting) ? h.toInt(p.quoting!) : 0
  if (qmode !== 0 && qmode !== 1) throw h.err('NotImplementedError', `to_csv(quoting=${qmode}) is not available in this sandbox`)
  const always = qmode === 1
  const useIndex = flag(h, p.index, true)
  const colKeep = given(p.columns) ? cellsOf(h, p.columns!).map((c) => {
    const at = f.columns.locate(c)
    if (!at.length) throw h.err('KeyError', `"[${typeof c === 'string' ? `'${c}'` : labelStr(c)}] not in index"`)
    return at[0]
  }) : f.data.map((_, k) => k)
  const headerV = p.header
  const header = headerV ? (isListLike(headerV) ? cellsOf(h, headerV).map(labelStr) : h.truthy(headerV)) : headerDefault
  const out: string[] = []
  const q = (s: string) => csvQuote(s, sep, quote, always)
  if (header !== false) {
    const labels = Array.isArray(header) ? header : colKeep.map((k) => labelStr(f.columns.labels[k]))
    const il = given(p.index_label)
    const idxLabels = !useIndex || (il && il.k === 'bool' && !il.v) ? [] : il ? (isListLike(il) ? cellsOf(h, il).map(labelStr) : [h.str(il)]) : f.index.names.map((x) => (x === null ? '' : labelStr(x)))
    out.push([...idxLabels, ...labels].map(q).join(sep))
  }
  const levels = Array.from({ length: f.index.nlevels }, (_, j) => f.index.level(j))
  for (let r = 0; r < f.nrows; r++) {
    const idx = useIndex ? levels.map((l) => q(csvCell(l.labels[r], l.dt, na, undefined))) : []
    out.push([...idx, ...colKeep.map((k) => q(csvCell(f.data[k].values[r], f.data[k].dtype, na, ff)))].join(sep))
  }
  return out.join(eol) + eol
}

const csvMode = (h: Host, p: Params) => {
  const mode = given(p.mode) ? h.str(p.mode!) : 'w'
  if (!['w', 'a', 'x'].includes(mode)) throw h.err('NotImplementedError', `to_csv(mode='${mode}') is not available in this sandbox`)
  return mode
}
export const frameToCsv = (h: Host, f: Frame, p: Params): Value => deliver(h, p.path_or_buf, tableToCsv(h, f, p, true), csvMode(h, p))
export const seriesToCsv = (h: Host, s: Series, p: Params): Value => deliver(h, p.path_or_buf, tableToCsv(h, new Frame([s.name ?? 0], [s], s.index), p, true), csvMode(h, p))

// ---------------------------------------------------------------- JSON

const jsonStr = (s: string, ascii: boolean) => {
  let out = '"'
  for (const c of s) {
    const code = c.codePointAt(0)!
    if (c === '"') out += '\\"'
    else if (c === '\\') out += '\\\\'
    else if (c === '/') out += '\\/'
    else if (c === '\n') out += '\\n'
    else if (c === '\r') out += '\\r'
    else if (c === '\t') out += '\\t'
    else if (c === '\b') out += '\\b'
    else if (c === '\f') out += '\\f'
    else if (code < 0x20 || (ascii && code > 0x7e)) {
      if (code > 0xffff) {
        const v = code - 0x10000
        out += `\\u${(0xd800 + (v >> 10)).toString(16).padStart(4, '0')}\\u${(0xdc00 + (v & 0x3ff)).toString(16).padStart(4, '0')}`
      } else out += `\\u${code.toString(16).padStart(4, '0')}`
    } else out += c
  }
  return `${out}"`
}

/** ujson's Buffer_AppendDoubleUnchecked */
const jsonFloat = (v0: number, prec: number) => {
  if (!Number.isFinite(v0)) return 'null'
  const neg = v0 < 0
  const v = Math.abs(v0)
  if (v > 1e16 - 1 || (v !== 0 && v < 1e-15)) {
    const t = Number(v0.toPrecision(Math.max(1, prec)))
    return t.toExponential().replace(/e([+-])(\d)$/, 'e$10$2')
  }
  const pow10 = 10 ** prec
  let whole = Math.trunc(v)
  const tmp = (v - whole) * pow10
  let frac = Math.trunc(tmp)
  const diff = tmp - frac
  if (diff > 0.5) frac++
  else if (diff === 0.5 && (frac === 0 || frac % 2 === 1)) frac++
  if (frac >= pow10) {
    frac = 0
    whole++
  }
  const sign = neg ? '-' : ''
  if (prec === 0) return `${sign}${diff > 0.5 || (diff === 0.5 && whole % 2 === 1) ? whole + 1 : whole}`
  if (!frac) return `${sign}${whole}.0`
  return `${sign}${whole}.${String(frac).padStart(prec, '0').replace(/0+$/, '')}`
}

const jsonValue = (h: Host, v: Value, prec: number, ascii: boolean): string => {
  switch (v.k) {
    case 'none': return 'null'
    case 'bool': return v.v ? 'true' : 'false'
    case 'int': return String(v.v)
    case 'float': return jsonFloat(v.v, prec)
    case 'str': return jsonStr(v.v, ascii)
    case 'list': case 'tuple': return `[${v.items.map((x) => jsonValue(h, x, prec, ascii)).join(',')}]`
    case 'dict': return `{${v.d.items().map(([k, x]) => `${jsonStr(h.str(k), ascii)}:${jsonValue(h, x, prec, ascii)}`).join(',')}}`
  }
  return jsonStr(h.str(v), ascii)
}

const jsonCell = (h: Host, c: Cell, dt: DT, prec: number, ascii: boolean) => jsonValue(h, valueOf(c, dt), prec, ascii)
const jsonKey = (c: Cell, ascii: boolean) => jsonStr(labelStr(c), ascii)

const jsonOpts = (h: Host, p: Params) => {
  refuseIO(h, p, 'to_json', ['date_format', 'date_unit', 'default_handler', 'compression', 'indent', 'index'])
  return { prec: given(p.double_precision) ? h.toInt(p.double_precision!) : 10, ascii: flag(h, p.force_ascii, true), lines: flag(h, p.lines) }
}

export const frameToJson = (h: Host, f: Frame, p: Params): Value => {
  const { prec, ascii, lines } = jsonOpts(h, p)
  const orient = given(p.orient) ? h.str(p.orient!) : 'columns'
  const cell = (k: number, r: number) => jsonCell(h, f.data[k].values[r], f.data[k].dtype, prec, ascii)
  const n = f.nrows
  const rs = Array.from({ length: n }, (_, i) => i)
  const cs = f.data.map((_, k) => k)
  const col = (k: number) => jsonKey(f.columns.labels[k], ascii)
  const rec = (r: number) => `{${cs.map((k) => `${col(k)}:${cell(k, r)}`).join(',')}}`
  if (lines) {
    if (orient !== 'records') throw h.err('ValueError', "'lines' keyword only valid when 'orient' is records")
    return deliver(h, p.path_or_buf, `${rs.map(rec).join('\n')}\n`)
  }
  if (['columns', 'index'].includes(orient)) {
    if (!f.index.isUnique) throw h.err('ValueError', `DataFrame index must be unique for orient='${orient}'.`)
    if (new Set(f.columns.labels).size !== f.ncols) throw h.err('ValueError', `DataFrame columns must be unique for orient='${orient}'.`)
  }
  let text: string
  switch (orient) {
    case 'columns': text = `{${cs.map((k) => `${col(k)}:{${rs.map((r) => `${jsonKey(f.index.labels[r], ascii)}:${cell(k, r)}`).join(',')}}`).join(',')}}`; break
    case 'index': text = `{${rs.map((r) => `${jsonKey(f.index.labels[r], ascii)}:${rec(r)}`).join(',')}}`; break
    case 'records': text = `[${rs.map(rec).join(',')}]`; break
    case 'values': text = `[${rs.map((r) => `[${cs.map((k) => cell(k, r)).join(',')}]`).join(',')}]`; break
    case 'split': text = `{"columns":[${f.columns.labels.map((c) => jsonValue(h, valueOf(c, f.columns.dt), prec, ascii)).join(',')}],"index":[${f.index.labels.map((c) => jsonValue(h, valueOf(c, f.index.dt), prec, ascii)).join(',')}],"data":[${rs.map((r) => `[${cs.map((k) => cell(k, r)).join(',')}]`).join(',')}]}`; break
    default: throw h.err(orient === 'table' ? 'NotImplementedError' : 'ValueError', orient === 'table' ? "to_json(orient='table') is not available in this sandbox" : `Invalid value '${orient}' for option 'orient'`)
  }
  return deliver(h, p.path_or_buf, text)
}

export const seriesToJson = (h: Host, s: Series, p: Params): Value => {
  const { prec, ascii, lines } = jsonOpts(h, p)
  const orient = given(p.orient) ? h.str(p.orient!) : 'index'
  const cell = (r: number) => jsonCell(h, s.values[r], s.dtype, prec, ascii)
  const rs = s.values.map((_, i) => i)
  if (lines) {
    if (orient !== 'records') throw h.err('ValueError', "'lines' keyword only valid when 'orient' is records")
    return deliver(h, p.path_or_buf, `${rs.map(cell).join('\n')}\n`)
  }
  let text: string
  switch (orient) {
    case 'index': text = `{${rs.map((r) => `${jsonKey(s.index.labels[r], ascii)}:${cell(r)}`).join(',')}}`; break
    case 'records': case 'values': text = `[${rs.map(cell).join(',')}]`; break
    case 'split': text = `{"name":${s.name === null ? 'null' : jsonValue(h, valueOf(s.name, 'object'), prec, ascii)},"index":[${s.index.labels.map((c) => jsonValue(h, valueOf(c, s.index.dt), prec, ascii)).join(',')}],"data":[${rs.map(cell).join(',')}]}`; break
    default: throw h.err(orient === 'table' ? 'NotImplementedError' : 'ValueError', orient === 'table' ? "to_json(orient='table') is not available in this sandbox" : `Invalid value '${orient}' for option 'orient'`)
  }
  return deliver(h, p.path_or_buf, text)
}

/** JSON text as Python values, keeping ints and floats apart */
const parseJson = (h: Host, text: string): Value => {
  let i = 0
  const ws = () => {
    while (i < text.length && ' \t\r\n'.includes(text[i])) i++
  }
  const fail = (): never => {
    throw h.err('ValueError', 'Expected object or value')
  }
  const value = (): Value => {
    ws()
    const c = text[i]
    if (c === '{') {
      i++
      const d = new PyDict()
      ws()
      if (text[i] === '}') {
        i++
        return { k: 'dict', d }
      }
      for (;;) {
        ws()
        const key = value()
        ws()
        if (text[i++] !== ':') fail()
        d.set(key, value())
        ws()
        if (text[i] === ',') {
          i++
          continue
        }
        if (text[i++] !== '}') fail()
        return { k: 'dict', d }
      }
    }
    if (c === '[') {
      i++
      const items: Value[] = []
      ws()
      if (text[i] === ']') {
        i++
        return py.list(items)
      }
      for (;;) {
        items.push(value())
        ws()
        if (text[i] === ',') {
          i++
          continue
        }
        if (text[i++] !== ']') fail()
        return py.list(items)
      }
    }
    if (c === '"') {
      const start = i
      i++
      while (i < text.length && text[i] !== '"') i += text[i] === '\\' ? 2 : 1
      i++
      try {
        return py.str(JSON.parse(text.slice(start, i)) as string)
      } catch {
        return fail()
      }
    }
    const m = /^(-?\d+(\.\d+)?([eE][+-]?\d+)?|true|false|null|NaN|Infinity|-Infinity)/.exec(text.slice(i))
    if (!m) return fail()
    i += m[0].length
    if (m[0] === 'true') return py.bool(true)
    if (m[0] === 'false') return py.bool(false)
    if (m[0] === 'null') return py.NONE
    if (m[0] === 'NaN') return py.float(NaN)
    if (m[0].includes('Infinity')) return py.float(m[0].startsWith('-') ? -Infinity : Infinity)
    return m[2] || m[3] ? py.float(Number(m[0])) : py.int(Number(m[0]))
  }
  const v = value()
  ws()
  if (i < text.length) throw h.err('ValueError', 'Trailing data')
  return v
}

/** read_json's conversion of index labels: "0", "1" become integers */
const axisFromJson = (labels: string[]): IndexObj => {
  if (labels.length && labels.every((l) => /^-?\d+$/.test(l))) {
    const nums = labels.map(Number)
    return new IndexObj(nums, null, 'int64')
  }
  return new IndexObj(labels, null, 'object')
}

const fromRecords = (h: Host, rows: Value[], index: IndexObj | null): Frame => {
  const cols: Cell[] = []
  for (const r of rows) if (r.k === 'dict') for (const k of r.d.keys()) if (!cols.includes(cellOf(h, k))) cols.push(cellOf(h, k))
  return new Frame(cols, cols.map((c) => {
    const t = jsonColumn(h, rows.map((r) => (r.k === 'dict' ? r.d.get(valueOf(c, 'object')) ?? py.float(NaN) : py.float(NaN))))
    return new Series(t.cells, null, c, t.dt)
  }), index)
}

/** read_json's dtype inference: null is NaN, and a whole-number float column becomes int */
const jsonColumn = (h: Host, items: Value[]): { cells: Cell[]; dt: DT } => {
  const numeric = items.every((v) => v.k === 'none' || v.k === 'int' || v.k === 'float')
  const t = inferValues(h, numeric ? items.map((v) => (v.k === 'none' ? py.float(NaN) : v)) : items)
  if (t.dt === 'float64' && t.cells.every((c) => Number.isInteger(c))) return { cells: t.cells, dt: 'int64' }
  return t
}

export const readJson = (h: Host, src: Value, kw: Record<string, Value>): Frame | Series => {
  for (const k of Object.keys(kw)) if (!['path_or_buf', 'orient', 'typ', 'lines', 'encoding', 'precise_float'].includes(k)) throw h.err('NotImplementedError', `read_json(${k}=...) is not available in this sandbox`)
  const text = src.k === 'str' && /^\s*[[{]/.test(src.v) ? src.v : textFrom(h, src)
  if (text === undefined) throw h.err('FileNotFoundError', `File ${h.str(src)} does not exist`)
  const orient = given(kw.orient) ? h.str(kw.orient) : null
  const typ = given(kw.typ) ? h.str(kw.typ) : 'frame'
  if (flag(h, kw.lines)) {
    const rows = text.split('\n').filter((l) => l.trim()).map((l) => parseJson(h, l))
    return fromRecords(h, rows, null)
  }
  const v = parseJson(h, text)
  if (typ === 'series') {
    if (v.k === 'dict') {
      const t = jsonColumn(h, v.d.values())
      return new Series(t.cells, axisFromJson(v.d.keys().map((x) => h.str(x))), null, t.dt)
    }
    if (v.k === 'list') {
      const t = jsonColumn(h, v.items)
      return new Series(t.cells, null, null, t.dt)
    }
    throw h.err('ValueError', 'Expected object or value')
  }
  if (typ !== 'frame') throw h.err('ValueError', `No axis named ${typ}`)
  if (orient === 'split') {
    if (v.k !== 'dict') throw h.err('ValueError', 'Expected object or value')
    const get = (k: string) => v.d.get(py.str(k))
    const columns = get('columns')
    const data = get('data')
    const idx = get('index')
    const rows = data ? h.iterate(data).map((r) => h.iterate(r)) : []
    const labels = columns ? h.iterate(columns).map((c) => cellOf(h, c)) : rows[0]?.map((_, j) => j) ?? []
    return new Frame(labels, labels.map((l, j) => {
      const t = jsonColumn(h, rows.map((r) => r[j]))
      return new Series(t.cells, null, l, t.dt)
    }), idx ? new IndexObj(h.iterate(idx).map((x) => cellOf(h, x))) : null)
  }
  if (orient === 'values') {
    const rows = h.iterate(v).map((r) => h.iterate(r))
    const w = rows[0]?.length ?? 0
    return new Frame(IndexObj.range(w), Array.from({ length: w }, (_, j) => {
      const t = jsonColumn(h, rows.map((r) => r[j]))
      return new Series(t.cells, null, j, t.dt)
    }), null)
  }
  if (v.k === 'list') return fromRecords(h, v.items, null)
  if (v.k !== 'dict') throw h.err('ValueError', 'Expected object or value')
  if (orient === 'index') {
    const rowKeys = v.d.keys().map((x) => h.str(x))
    return fromRecords(h, v.d.values(), axisFromJson(rowKeys))
  }
  const cols = v.d.keys().map((x) => cellOf(h, x))
  const inner = v.d.values()
  if (inner.every((x) => x.k === 'dict')) {
    const rowKeys: string[] = []
    for (const x of inner) for (const k of (x as { d: PyDict }).d.keys()) if (!rowKeys.includes(h.str(k))) rowKeys.push(h.str(k))
    const index = axisFromJson(rowKeys)
    return new Frame(new IndexObj(cols.map((c) => (typeof c === 'string' && /^-?\d+$/.test(c) ? Number(c) : c))), inner.map((x, i) => {
      const t = jsonColumn(h, rowKeys.map((k) => (x as { d: PyDict }).d.get(py.str(k)) ?? py.NONE))
      return new Series(t.cells, null, cols[i], t.dt)
    }), index)
  }
  if (inner.every((x) => x.k === 'list')) return new Frame(cols, inner.map((x, i) => {
    const t = jsonColumn(h, (x as { items: Value[] }).items)
    return new Series(t.cells, null, cols[i], t.dt)
  }), null)
  throw h.err('ValueError', 'If using all scalar values, you must pass an index')
}

/** pd.json_normalize for records and nested dicts: inner keys join with sep */
export const jsonNormalize = (h: Host, data: Value, kw: Record<string, Value>): Frame => {
  for (const k of Object.keys(kw)) if (!['data', 'sep', 'max_level', 'errors'].includes(k)) throw h.err('NotImplementedError', `json_normalize(${k}=...) is not available in this sandbox`)
  const sep = given(kw.sep) ? h.str(kw.sep) : '.'
  const maxLevel = given(kw.max_level) ? h.toInt(kw.max_level) : Infinity
  const rows = data.k === 'dict' ? [data] : h.iterate(data)
  const flatten = (d: PyDict, prefix: string, level: number, out: Map<string, Value>) => {
    for (const [k, v] of d.items()) {
      const key = prefix ? `${prefix}${sep}${h.str(k)}` : h.str(k)
      if (v.k === 'dict' && level < maxLevel) flatten(v.d, key, level + 1, out)
      else out.set(key, v)
    }
  }
  const flat = rows.map((r) => {
    if (r.k !== 'dict') throw h.err('NotImplementedError', `json_normalize takes dicts here, got ${r.k}`)
    const m = new Map<string, Value>()
    flatten(r.d, '', 0, m)
    return m
  })
  const cols: string[] = []
  for (const m of flat) for (const k of m.keys()) if (!cols.includes(k)) cols.push(k)
  return new Frame(cols, cols.map((c) => {
    const t = inferValues(h, flat.map((m) => m.get(c) ?? py.float(NaN)))
    return new Series(t.cells, null, c, t.dt)
  }), null)
}

