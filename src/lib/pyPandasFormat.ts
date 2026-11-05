/**
 * pandas' text layout, ported from pandas/io/formats (format.py, printing.py, string.py): how a column of values becomes
 * strings, how columns are joined, and when a table or an Index wraps. Nothing here knows about Series or DataFrame.
 */
import { str as pyStr } from './minipy'
import type { Value } from './minipy'
import { pyFloatRepr } from './ndarray'

export type DT = 'int64' | 'float64' | 'bool' | 'object'

/** a Python object sitting in an object column: a list from str.split, a tuple, or a float that must stay a float (70.0) */
export class PyCell {
  readonly v: Value
  constructor(v: Value) {
    this.v = v
  }
}
export type Cell = number | string | boolean | null | PyCell

export const isMissing = (c: Cell) => c === null || (typeof c === 'number' && Number.isNaN(c))

/** display.* options; reset when a run imports pandas */
export const DEFAULT_OPTIONS = { max_rows: 60 as number | null, min_rows: 10 as number | null, max_columns: 0 as number | null, width: 80 as number | null, precision: 6, max_colwidth: 50 as number | null, expand_frame_repr: true, max_seq_items: 100, multi_sparse: true, show_dimensions: 'truncate' as boolean | string }
export const OPTIONS = { ...DEFAULT_OPTIONS }
/** shutil.get_terminal_size() of a script with no terminal */
export const TERMINAL_WIDTH = 80

export const escapeText = (s: string) => s.replace(/\t/g, '\\t').replace(/\r/g, '\\r').replace(/\n/g, '\\n')

/** pprint_thing: lists show their strings bare, dicts quote them */
export const pprint = (v: Value, nest = 0, quote = false): string => {
  if (nest < 3) {
    const max = OPTIONS.max_seq_items
    if (v.k === 'dict') {
      const items = v.d.items()
      const body = items.slice(0, max).map(([k, x]) => `${pprint(k, nest + 1, true)}: ${pprint(x, nest + 1, true)}`).join(', ')
      return `{${body}${items.length > max ? ', ...' : ''}}`
    }
    if (v.k === 'list' || v.k === 'tuple') {
      const body = v.items.slice(0, max).map((x) => pprint(x, nest + 1, quote)).join(', ')
      if (v.k === 'list') return `[${body}${v.items.length > max ? ', ...' : ''}]`
      return `(${body}${v.items.length > max ? ', ...' : v.items.length === 1 ? ',' : ''})`
    }
  }
  if (v.k === 'str') return quote ? `'${escapeText(v.v)}'` : escapeText(v.v)
  return escapeText(pyStr(v))
}

/** format(v, '.{d}f'): toFixed, but an exact tie goes to the even digit as in C, and huge values stay positional */
export const fixedStr = (v: number, d: number): string => {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? 'nan' : v > 0 ? 'inf' : '-inf'
  const a = Math.abs(v)
  const neg = v < 0 || Object.is(v, -0)
  let s: string
  if (a >= 1e21) s = BigInt(a).toString() + (d ? `.${'0'.repeat(d)}` : '')
  else {
    s = a.toFixed(d)
    if (Number.isInteger(a * 2 ** (d + 1))) {
      const exact = a.toFixed(d + 1)
      if (exact.endsWith('5')) {
        const down = exact.slice(0, -1).replace(/\.$/, '')
        if (Number(down[down.length - 1]) % 2 === 0) s = down
      }
    }
  }
  return (neg ? '-' : '') + s
}

const expStr = (v: number, d: number) => v.toExponential(d).replace(/e([+-])(\d)$/, 'e$10$2')

const NUMBER_WITH_DOT = /^\s*[+-]?[0-9]+\.[0-9]*$/

/** _trim_zeros_float: drop trailing zeros together, keeping one decimal */
const trimZeros = (xs: string[]): string[] => {
  let out = xs
  const should = () => {
    const nums = out.filter((x) => NUMBER_WITH_DOT.test(x))
    return nums.length > 0 && nums.every((x) => x.endsWith('0'))
  }
  while (should()) out = out.map((x) => (NUMBER_WITH_DOT.test(x) ? x.slice(0, -1) : x))
  return out.map((x) => (NUMBER_WITH_DOT.test(x) && x.endsWith('.') ? `${x}0` : x))
}

/** FloatArrayFormatter.get_result_as_array */
export const formatFloats = (vals: number[], leading: boolean, naRep = 'NaN', floatFormat?: (v: number) => string): string[] => {
  if (floatFormat) return vals.map((v) => (Number.isNaN(v) ? naRep : floatFormat(v)))
  const d = OPTIONS.precision
  const sign = (s: string) => (leading && !s.startsWith('-') ? ` ${s}` : s)
  const fixed = trimZeros(vals.map((v) => (Number.isNaN(v) ? naRep : sign(fixedStr(v, d)))))
  const maxlen = fixed.reduce((m, x) => Math.max(m, x.length), 0)
  const small = Number(`1e-${d}`)
  const large = vals.some((v) => Math.abs(v) > 1e6)
  const tiny = vals.some((v) => Math.abs(v) < small && v !== 0)
  if (tiny || (maxlen > d + 6 && large)) return vals.map((v) => (Number.isNaN(v) ? naRep : !Number.isFinite(v) ? sign(fixedStr(v, d)) : sign(expStr(v, d))))
  return fixed
}

const isFloatCell = (c: Cell) => (typeof c === 'number' && !Number.isNaN(c) && !Number.isInteger(c)) || (c instanceof PyCell && c.v.k === 'float' && !Number.isNaN(c.v.v))
const floatOf = (c: Cell) => (c instanceof PyCell ? (c.v as { v: number }).v : (c as number))

/** a value in an object column, as GenericArrayFormatter._format writes it */
export const cellText = (c: Cell, naRep = 'NaN'): string => {
  if (c === null) return 'None'
  if (typeof c === 'number') return Number.isNaN(c) ? naRep : Number.isInteger(c) ? String(c) : pyFloatRepr(c)
  if (typeof c === 'boolean') return c ? 'True' : 'False'
  if (typeof c === 'string') return escapeText(c)
  return pprint(c.v)
}

/** format_array before justification. leading = room for a sign (off for index=False) */
export const formatArray = (cells: Cell[], dt: DT, leading = true, naRep = 'NaN', floatFormat?: (v: number) => string): string[] => {
  if (dt === 'float64') return formatFloats(cells as number[], leading, naRep, floatFormat)
  if (dt === 'int64') return cells.map((c) => (leading && (c as number) >= 0 ? ` ${c}` : String(c)))
  const d = OPTIONS.precision
  return cells.map((c) => {
    if (isFloatCell(c)) {
      const x = floatOf(c)
      if (floatFormat) return leading ? ` ${floatFormat(x)}` : floatFormat(x)
      const s = (leading && x >= 0 ? ' ' : '') + fixedStr(x, d)
      const t = s.includes('.') ? s.replace(/0+$/, '') : s
      return t.endsWith('.') ? `${t}0` : t
    }
    return (leading ? ' ' : '') + cellText(c, naRep)
  })
}

export const justify = (s: string, width: number, mode: 'left' | 'right' | 'center'): string => {
  if (mode === 'left') return s.padEnd(width)
  if (mode === 'right') return s.padStart(width)
  const marg = width - s.length
  if (marg <= 0) return s
  const left = Math.floor(marg / 2) + (marg & width & 1)
  return ' '.repeat(left) + s + ' '.repeat(marg - left)
}

/** _make_fixed_width */
export const fixedWidth = (strs: string[], mode: 'left' | 'right' | 'center' | 'all', minimum = 0): string[] => {
  if (!strs.length || mode === 'all') return strs
  let w = Math.max(minimum, ...strs.map((x) => x.length))
  const conf = OPTIONS.max_colwidth
  if (conf !== null && w > conf) w = conf
  return strs.map((x) => justify(conf !== null && conf > 3 && x.length > w ? `${x.slice(0, w - 3)}...` : x, w, mode))
}

/** pandas' adjoin: every column is left-justified to its width + space, the last one without the space */
export const adjoin = (space: number, ...lists: string[][]): string => {
  if (!lists.length) return ''
  const widths = lists.map((l, i) => l.reduce((m, x) => Math.max(m, x.length), 0) + (i < lists.length - 1 ? space : 0))
  const rows = Math.max(...lists.map((l) => l.length))
  const padded = lists.map((l, i) => [...Array.from({ length: rows - l.length }, () => ' '.repeat(widths[i])), ...l.map((x) => x.padEnd(widths[i]))])
  return Array.from({ length: rows }, (_, r) => padded.map((p) => p[r]).join('')).join('\n')
}

const trimFront = (strs: string[]): string[] => {
  if (!strs.length) return strs
  const lead = Math.min(...strs.map((x) => x.length - x.trimStart().length))
  return lead > 0 ? strs.map((x) => x.slice(lead)) : strs
}

/** Index._format_with_header: labels as they sit in the index column */
export const formatLabels = (labels: Cell[], dt: DT): string[] => {
  const raw = formatArray(labels, dt)
  return trimFront(dt === 'object' ? raw : fixedWidth(raw, 'left'))
}

/** sparsify_labels: a repeated outer label prints blank. levels are per level, each with `start` header rows */
export const sparsify = (levels: string[][], start = 0): string[][] => {
  if (!levels.length) return levels
  const k = levels.length
  const n = levels[0].length
  if (n <= start) return levels
  const out: string[][] = levels.map((l) => l.slice(0, start + 1))
  for (let r = start + 1; r < n; r++) {
    let same = true
    for (let i = 0; i < k; i++) {
      if (same && i < k - 1 && levels[i][r - 1] === levels[i][r]) out[i].push('')
      else {
        same = false
        out[i].push(levels[i][r])
      }
    }
  }
  return out
}

/** _binify and _join_multiline: columns wrapped at line_width with a trailing backslash */
export const joinMultiline = (strcols: string[][], lineWidth: number, hasIndex: boolean): string => {
  const cols = [...strcols]
  let width = lineWidth
  let idx: string[] = []
  if (hasIndex) {
    idx = cols.shift()!
    width -= Math.max(...idx.map((x) => x.length)) + 1
  }
  const widths = cols.map((c) => c.reduce((m, x) => Math.max(m, x.length), 0))
  const bins: number[] = []
  let cur = 0
  widths.forEach((w, i) => {
    cur += w + 1
    const wrap = (i === widths.length - 1 ? cur + 1 > width : cur + 2 > width) && i > 0
    if (wrap) {
      bins.push(i)
      cur = w + 1
    }
  })
  bins.push(cols.length)
  const parts: string[] = []
  let start = 0
  bins.forEach((end, i) => {
    const row = cols.slice(start, end)
    if (hasIndex) row.unshift(idx)
    if (bins.length > 1) {
      const nrows = row[row.length - 1].length
      if (end <= cols.length && i < bins.length - 1) row.push([' \\', ...Array.from({ length: nrows - 1 }, () => '  ')])
      else row.push(Array.from({ length: nrows }, () => ' '))
    }
    parts.push(adjoin(1, ...row))
    start = end
  })
  return parts.join('\n\n')
}

/** format_object_summary: the body of an Index repr, up to the "dtype=" part */
export const objectSummary = (head: string[], tail: string[], truncated: boolean, name: string, isJustify: boolean, breakEach: boolean): string => {
  const width = OPTIONS.width ?? TERMINAL_WIDTH
  const space1 = `\n${' '.repeat(name.length + 1)}`
  const space2 = `\n${' '.repeat(name.length + 2)}`
  const n = head.length + tail.length
  const close = ', '
  if (n === 0) return `[]${close}`
  if (n === 1 && !breakEach) return `[${tail[0]}]${close}`
  if (n === 2 && !breakEach && !truncated) return `[${tail[0]}, ${tail[1]}]${close}`
  const sep = breakEach ? `,\n ${' '.repeat(name.length)}` : ','
  let hs = head
  let ts = tail
  if (isJustify && !breakEach && (truncated || !(hs.join(', ').length < width && ts.join(', ').length < width))) {
    const w = Math.max(0, ...hs.map((x) => x.length), ...ts.map((x) => x.length))
    hs = hs.map((x) => x.padStart(w))
    ts = ts.map((x) => x.padStart(w))
  }
  let summary = ''
  let line = space2
  const extend = (word: string, w: number) => {
    if (line.trimEnd().length + word.trimEnd().length >= w) {
      summary += line.trimEnd()
      line = space2
    }
    line += word
  }
  for (const v of hs) extend(`${v}${sep} `, width)
  if (truncated) {
    summary += line.trimEnd() + space2 + '...'
    line = space2
  }
  for (const v of ts.slice(0, -1)) extend(`${v}${sep} `, width)
  extend(ts[ts.length - 1], width - 2)
  summary += line
  summary += ']' + close.trimEnd()
  summary += summary.length > width || breakEach ? space1 : ' '
  return `[${summary.slice(space2.length)}`
}
