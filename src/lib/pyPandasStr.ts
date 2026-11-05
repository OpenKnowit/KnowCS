/** Series.str and Index.str: Python's string methods and regular expressions, element by element, keeping missing values. */
import { PyObj, py } from './minipy'
import type { Host, Value } from './minipy'
import { Frame, IndexObj, Series, bindParams, cellsOf, inferValues, isListLike, isMissing, reindexCells, sliceRange, sortOrder, valueOf } from './pyPandas'
import type { Cell, DT, Params } from './pyPandas'
import { PyCell } from './pyPandasFormat'
import { flag, given, need } from './pyPandasMethods'

const RE_I = 2
const RE_M = 8
const RE_S = 16

/** a Python regular expression as a JavaScript one */
export const pyRegex = (h: Host, pat: string, jsFlags = '', pyFlags = 0): RegExp => {
  let src = pat
  let flags = jsFlags
  const inline = /^\(\?([aiLmsux]+)\)/.exec(src)
  if (inline) {
    src = src.slice(inline[0].length)
    if (inline[1].includes('i')) flags += 'i'
    if (inline[1].includes('m')) flags += 'm'
    if (inline[1].includes('s')) flags += 's'
  }
  if (pyFlags & RE_I && !flags.includes('i')) flags += 'i'
  if (pyFlags & RE_M && !flags.includes('m')) flags += 'm'
  if (pyFlags & RE_S && !flags.includes('s')) flags += 's'
  src = src.replace(/\(\?P<([A-Za-z_][A-Za-z0-9_]*)>/g, '(?<$1>').replace(/\(\?P=([A-Za-z_][A-Za-z0-9_]*)\)/g, '\\k<$1>').replace(/\\A/g, '^').replace(/\\Z/g, '$')
  try {
    return new RegExp(src, flags)
  } catch (e) {
    throw h.err('PatternError', e instanceof Error ? e.message : 'bad regular expression')
  }
}

const isCased = (c: string) => c.toLowerCase() !== c.toUpperCase()
const titleCase = (x: string) => {
  let out = ''
  let prev = false
  for (const c of x) {
    const up = c.toUpperCase()
    out += prev ? c.toLowerCase() : up[0] + up.slice(1).toLowerCase()
    prev = isCased(c)
  }
  return out
}
const WS = ' \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f\x85\xa0'
const stripChars = (x: string, chars: string | null, left: boolean, right: boolean) => {
  const set = chars ?? WS
  let a = 0
  let b = x.length
  if (left) while (a < b && set.includes(x[a])) a++
  if (right) while (b > a && set.includes(x[b - 1])) b--
  return x.slice(a, b)
}
const pySplit = (x: string, sep: string | null, n: number): string[] => {
  if (sep === null) {
    const out: string[] = []
    let i = 0
    const len = x.length
    while (i < len) {
      while (i < len && WS.includes(x[i])) i++
      if (i >= len) break
      if (n >= 0 && out.length === n) {
        out.push(stripChars(x.slice(i), null, false, true))
        return out
      }
      let j = i
      while (j < len && !WS.includes(x[j])) j++
      out.push(x.slice(i, j))
      i = j
    }
    return out
  }
  const parts = x.split(sep)
  if (n < 0 || parts.length <= n + 1) return parts
  return [...parts.slice(0, n), parts.slice(n).join(sep)]
}
const pyRsplit = (x: string, sep: string | null, n: number): string[] => {
  if (n < 0) return pySplit(x, sep, -1)
  if (sep === null) {
    const words = pySplit(x, null, -1)
    if (words.length <= n + 1) return words
    let i = x.length
    let seen = 0
    while (i > 0 && seen < n) {
      while (i > 0 && WS.includes(x[i - 1])) i--
      while (i > 0 && !WS.includes(x[i - 1])) i--
      seen++
    }
    return [stripChars(x.slice(0, i), null, true, true), ...words.slice(-n)]
  }
  const parts = x.split(sep)
  if (parts.length <= n + 1) return parts
  return [parts.slice(0, parts.length - n).join(sep), ...parts.slice(parts.length - n)]
}
const pad = (x: string, w: number, side: 'left' | 'right' | 'both', f: string) => {
  const marg = w - x.length
  if (marg <= 0) return x
  if (side === 'left') return f.repeat(marg) + x
  if (side === 'right') return x + f.repeat(marg)
  const left = Math.floor(marg / 2) + (marg & w & 1)
  return f.repeat(left) + x + f.repeat(marg - left)
}
const pyFind = (x: string, sub: string, start: number, end: number | null, right: boolean) => {
  const a = start < 0 ? Math.max(0, start + x.length) : start
  const b = end === null ? x.length : end < 0 ? Math.max(0, end + x.length) : Math.min(end, x.length)
  const part = x.slice(a, b)
  const i = right ? part.lastIndexOf(sub) : part.indexOf(sub)
  return i < 0 ? -1 : a + i
}
const allOf = (x: string, re: RegExp, cased?: 'lower' | 'upper') => {
  if (cased) {
    let any = false
    for (const c of x) {
      if (isCased(c)) {
        any = true
        if (cased === 'lower' ? c !== c.toLowerCase() : c !== c.toUpperCase()) return false
      }
    }
    return any
  }
  return x.length > 0 && re.test(x)
}
const IS: Record<string, (x: string) => boolean> = {
  isalnum: (x) => allOf(x, /^[\p{L}\p{N}]+$/u),
  isalpha: (x) => allOf(x, /^\p{L}+$/u),
  isdecimal: (x) => allOf(x, /^\p{Nd}+$/u),
  isdigit: (x) => allOf(x, /^[\p{Nd}\u00b2\u00b3\u00b9\u2070\u2074-\u2079\u2080-\u2089]+$/u),
  isnumeric: (x) => allOf(x, /^\p{N}+$/u),
  isspace: (x) => x.length > 0 && [...x].every((c) => WS.includes(c) || /\s/u.test(c)),
  islower: (x) => allOf(x, /./, 'lower'),
  isupper: (x) => allOf(x, /./, 'upper'),
  istitle: (x) => {
    let cased = false
    let prev = false
    for (const c of x) {
      if (isCased(c)) {
        const upper = c === c.toUpperCase() && c !== c.toLowerCase()
        if (upper === prev) return false
        cased = true
        prev = true
      } else prev = false
    }
    return cased
  },
}

const fromRepl = (r: string) => r.replace(/\$/g, '$$$$').replace(/\\g<([A-Za-z_][A-Za-z0-9_]*)>/g, '$$<$1>').replace(/\\g<(\d+)>/g, '$$$1').replace(/\\(\d)/g, '$$$1')

const groupNames = (re: RegExp) => {
  const n = new RegExp(`${re.source}|`, re.flags.replace('g', '')).exec('')!.length - 1
  const named: (string | null)[] = new Array(n).fill(null)
  const open = /(?<!\\)\((?!\?)|\(\?<([A-Za-z_][A-Za-z0-9_]*)>/g
  let k = 0
  let m: RegExpExecArray | null
  while ((m = open.exec(re.source)) !== null) {
    if (m[1]) named[k] = m[1]
    k++
  }
  return named
}

export class StrAccessor extends PyObj {
  readonly cls = 'StringMethods'
  s: Series
  fromIndex: IndexObj | null
  constructor(s: Series, fromIndex: IndexObj | null = null) {
    super()
    this.s = s
    this.fromIndex = fromIndex
  }
  repr() { return '<pandas.core.strings.accessor.StringMethods object>' }
  /** the result as what the source was: an Index stays an Index, except for booleans which come back as an array */
  private wrap(cells: Cell[], dt: DT, h: Host): Value {
    void h
    if (this.fromIndex) {
      if (dt === 'bool') return py.arr(new Series(cells, null, null, 'bool').toArray())
      return py.obj(new IndexObj(cells, this.fromIndex.name, dt))
    }
    return py.obj(new Series(cells, this.s.index, this.s.name, dt))
  }
  /** f on each string; a missing value stays as it was, anything else that is not a string gives NaN */
  private each(h: Host, f: (x: string) => Value): Value {
    const res = this.s.values.map((c): Value => {
      if (isMissing(c)) return valueOf(c, 'object')
      if (typeof c !== 'string') return py.float(NaN)
      return f(c)
    })
    const t = inferValues(h, res)
    return this.wrap(t.cells, t.dt, h)
  }
  /** like each, for functions that also take lists (len, get, join) */
  private eachAny(h: Host, f: (x: Cell) => Value): Value {
    const res = this.s.values.map((c): Value => (isMissing(c) ? valueOf(c, 'object') : f(c)))
    const t = inferValues(h, res)
    return this.wrap(t.cells, t.dt, h)
  }
  /** a predicate; na= replaces missing values, which otherwise make the column object */
  private test(h: Host, f: (x: string) => boolean, na: Value | undefined): Value {
    const fill = na && na.k !== 'none' ? na : null
    const res = this.s.values.map((c): Value => {
      if (isMissing(c)) return fill ?? valueOf(c, 'object')
      if (typeof c !== 'string') return fill ?? py.float(NaN)
      return py.bool(f(c))
    })
    const t = inferValues(h, res)
    return this.wrap(t.cells, t.dt, h)
  }
  private expand(h: Host, rows: (Cell | null)[][]): Value {
    const width = Math.max(0, ...rows.map((r) => r.length))
    const data = Array.from({ length: width }, (_, j) => new Series(rows.map((r) => (j < r.length ? r[j] : null)), this.s.index, j, 'object'))
    if (this.fromIndex) throw h.err('NotImplementedError', 'expand=True on an Index is not available in this sandbox')
    return py.obj(new Frame(new IndexObj(data.map((_, j) => j), null, 'int64', true), data, this.s.index))
  }
  getAttr(name: string, h: Host): Value | undefined {
    const def = (spec: string, impl: (p: Params) => Value): Value => ({ k: 'fn', name, call: (args, kw) => impl(bindParams(h, 'str', name, spec, args, kw)) })
    const simple = (f: (x: string) => string) => def('', () => this.each(h, (x) => py.str(f(x))))
    const reOf = (p: Params, jsFlags = '') => pyRegex(h, h.str(need(h, p.pat, name, 'pat')), jsFlags, given(p.flags) ? h.toInt(p.flags!) : 0)
    const widthArg = (p: Params, fill = true) => {
      const wv = need(h, p.width, name, 'width')
      if (wv.k !== 'int') throw h.err('TypeError', `width must be of integer type, not ${wv.k}`)
      const f = fill && given(p.fillchar) ? h.str(p.fillchar!) : ' '
      if (f.length !== 1) throw h.err('TypeError', 'fillchar must be a character, not str')
      return { w: wv.v, f }
    }
    if (IS[name]) return def('', () => this.test(h, IS[name], undefined))
    switch (name) {
      case 'lower': return simple((x) => x.toLowerCase())
      case 'upper': return simple((x) => x.toUpperCase())
      case 'title': return simple(titleCase)
      case 'capitalize': return simple((x) => x.charAt(0).toUpperCase() + x.slice(1).toLowerCase())
      case 'casefold': return simple((x) => x.toLowerCase().replace(/ß/g, 'ss'))
      case 'swapcase': return simple((x) => [...x].map((c) => (c === c.toUpperCase() ? c.toLowerCase() : c.toUpperCase())).join(''))
      case 'strip': case 'lstrip': case 'rstrip': return def('to_strip', (p) => {
        const chars = given(p.to_strip) ? h.str(p.to_strip!) : null
        return this.each(h, (x) => py.str(stripChars(x, chars, name !== 'rstrip', name !== 'lstrip')))
      })
      case 'len': return def('', () => this.eachAny(h, (c) => {
        if (typeof c === 'string') return py.int(c.length)
        if (c instanceof PyCell && (c.v.k === 'list' || c.v.k === 'tuple')) return py.int(c.v.items.length)
        if (c instanceof PyCell && c.v.k === 'dict') return py.int(c.v.d.size)
        return py.float(NaN)
      }))
      case 'contains': return def('pat case flags na regex', (p) => {
        const caseOn = flag(h, p.case, true)
        const pat = h.str(need(h, p.pat, name, 'pat'))
        if (flag(h, p.regex, true)) {
          const re = pyRegex(h, pat, caseOn ? '' : 'i', given(p.flags) ? h.toInt(p.flags!) : 0)
          return this.test(h, (x) => re.test(x), p.na)
        }
        return this.test(h, (x) => (caseOn ? x.includes(pat) : x.toUpperCase().includes(pat.toUpperCase())), p.na)
      })
      case 'match': case 'fullmatch': return def('pat case flags na', (p) => {
        const caseOn = flag(h, p.case, true)
        const re = pyRegex(h, h.str(need(h, p.pat, name, 'pat')), caseOn ? '' : 'i', given(p.flags) ? h.toInt(p.flags!) : 0)
        const anchored = new RegExp(name === 'match' ? `^(?:${re.source})` : `^(?:${re.source})$`, re.flags.replace('m', ''))
        return this.test(h, (x) => anchored.test(x), p.na)
      })
      case 'startswith': case 'endswith': return def('pat na', (p) => {
        const pv = need(h, p.pat, name, 'pat')
        if (pv.k !== 'str' && pv.k !== 'tuple') throw h.err('TypeError', `expected a string or tuple, not ${pv.k}`)
        const pats = pv.k === 'tuple' ? pv.items.map((x) => h.str(x)) : [pv.v]
        return this.test(h, (x) => pats.some((q) => (name === 'startswith' ? x.startsWith(q) : x.endsWith(q))), p.na)
      })
      case 'replace': return def('pat repl n case flags regex', (p) => {
        const pat = h.str(need(h, p.pat, name, 'pat'))
        const rv = need(h, p.repl, name, 'repl')
        const n = given(p.n) ? h.toInt(p.n!) : -1
        const regex = flag(h, p.regex)
        const caseOff = given(p.case) ? !h.truthy(p.case!) : false
        const flagsN = given(p.flags) ? h.toInt(p.flags!) : 0
        if (rv.k !== 'str') {
          if (rv.k === 'func' || rv.k === 'fn') throw h.err('NotImplementedError', 'str.replace with a callable repl is not available in this sandbox')
          throw h.err('TypeError', 'repl must be a string or callable')
        }
        if (regex || caseOff || flagsN) {
          const repl = fromRepl(rv.v)
          const re = pyRegex(h, regex ? pat : pat.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), `g${caseOff ? 'i' : ''}`, flagsN)
          return this.each(h, (x) => {
            let k = 0
            return py.str(n < 0 ? x.replace(re, repl) : x.replace(re, (...m) => {
              if (k++ >= n) return m[0] as string
              return (m[0] as string).replace(new RegExp(re.source, re.flags.replace('g', '')), repl)
            }))
          })
        }
        return this.each(h, (x) => {
          const parts = x.split(pat)
          if (n < 0 || parts.length <= n + 1) return py.str(parts.join(rv.v))
          return py.str(parts.slice(0, n + 1).join(rv.v) + pat + parts.slice(n + 1).join(pat))
        })
      })
      case 'split': case 'rsplit': return def(name === 'split' ? 'pat | n expand regex' : 'pat | n expand', (p) => {
        const patV = given(p.pat)
        const n = given(p.n) ? h.toInt(p.n!) : -1
        const limit = n === 0 ? -1 : n
        const pat = patV ? h.str(patV) : null
        const useRe = name === 'split' && pat !== null && (p.regex && p.regex.k !== 'none' ? h.truthy(p.regex) : pat.length !== 1)
        const re = useRe ? pyRegex(h, pat!, 'g') : null
        const cut = (x: string): string[] => {
          if (name === 'rsplit') return pyRsplit(x, pat, limit)
          if (!re) return pySplit(x, pat, limit)
          const out: string[] = []
          let last = 0
          re.lastIndex = 0
          let m: RegExpExecArray | null
          let count = 0
          while ((m = re.exec(x)) !== null) {
            if (limit >= 0 && count >= limit) break
            if (!m[0].length) {
              re.lastIndex++
              if (m.index >= x.length) break
              if (m.index === 0) continue
            }
            out.push(x.slice(last, m.index), ...m.slice(1).map((g) => g ?? ''))
            last = m.index + m[0].length
            count++
          }
          out.push(x.slice(last))
          return out
        }
        if (flag(h, p.expand)) return this.expand(h, this.s.values.map((c) => (typeof c === 'string' ? cut(c) : [isMissing(c) ? c : NaN])))
        return this.each(h, (x) => py.list(cut(x).map(py.str)))
      })
      case 'partition': case 'rpartition': return def('sep expand', (p) => {
        const sep = given(p.sep) ? h.str(p.sep!) : ' '
        const cut = (x: string): string[] => {
          const i = name === 'partition' ? x.indexOf(sep) : x.lastIndexOf(sep)
          if (i < 0) return name === 'partition' ? [x, '', ''] : ['', '', x]
          return [x.slice(0, i), sep, x.slice(i + sep.length)]
        }
        if (flag(h, p.expand, true)) return this.expand(h, this.s.values.map((c) => (typeof c === 'string' ? cut(c) : [isMissing(c) ? c : NaN, isMissing(c) ? c : NaN, isMissing(c) ? c : NaN])))
        return this.each(h, (x) => py.tuple(cut(x).map(py.str)))
      })
      case 'center': return def('width fillchar', (p) => {
        const { w, f } = widthArg(p)
        return this.each(h, (x) => py.str(pad(x, w, 'both', f)))
      })
      case 'ljust': return def('width fillchar', (p) => {
        const { w, f } = widthArg(p)
        return this.each(h, (x) => py.str(pad(x, w, 'right', f)))
      })
      case 'rjust': return def('width fillchar', (p) => {
        const { w, f } = widthArg(p)
        return this.each(h, (x) => py.str(pad(x, w, 'left', f)))
      })
      case 'pad': return def('width side fillchar', (p) => {
        const { w, f } = widthArg(p)
        const side = given(p.side) ? h.str(p.side!) : 'left'
        if (!['left', 'right', 'both'].includes(side)) throw h.err('ValueError', 'Invalid side')
        return this.each(h, (x) => py.str(pad(x, w, side as 'left', f)))
      })
      case 'zfill': return def('width', (p) => {
        const { w } = widthArg(p, false)
        return this.each(h, (x) => {
          const sign = x[0] === '+' || x[0] === '-' ? x[0] : ''
          const rest = x.slice(sign.length)
          return py.str(sign + rest.padStart(w - sign.length, '0'))
        })
      })
      case 'slice': return def('start stop step', (p) => {
        const sl = { k: 'slice' as const, start: p.start ?? py.NONE, stop: p.stop ?? py.NONE, step: p.step ?? py.NONE }
        return this.each(h, (x) => py.str(sliceRange(h, sl, x.length).map((i) => x[i]).join('')))
      })
      case 'slice_replace': return def('start stop repl', (p) => {
        const repl = given(p.repl) ? h.str(p.repl!) : ''
        const norm = (v: Value | undefined, n: number, d: number) => {
          if (!given(v)) return d
          const i = h.toInt(v!)
          return i < 0 ? Math.max(0, i + n) : Math.min(i, n)
        }
        return this.each(h, (x) => {
          const a = norm(p.start, x.length, 0)
          const b = norm(p.stop, x.length, x.length)
          return py.str(x.slice(0, a) + repl + x.slice(Math.max(a, b)))
        })
      })
      case 'get': return def('i', (p) => {
        const iv = need(h, p.i, name, 'i')
        return this.eachAny(h, (c) => {
          if (typeof c === 'string') {
            const i = h.toInt(iv)
            const at = i < 0 ? i + c.length : i
            return at >= 0 && at < c.length ? py.str(c[at]) : py.float(NaN)
          }
          if (c instanceof PyCell && (c.v.k === 'list' || c.v.k === 'tuple')) {
            const i = h.toInt(iv)
            const at = i < 0 ? i + c.v.items.length : i
            return at >= 0 && at < c.v.items.length ? c.v.items[at] : py.float(NaN)
          }
          if (c instanceof PyCell && c.v.k === 'dict') return c.v.d.get(iv) ?? py.NONE
          return py.float(NaN)
        })
      })
      case 'join': return def('sep', (p) => {
        const sep = h.str(need(h, p.sep, name, 'sep'))
        return this.eachAny(h, (c) => {
          if (typeof c === 'string') return py.str([...c].join(sep))
          if (c instanceof PyCell && (c.v.k === 'list' || c.v.k === 'tuple')) {
            if (!c.v.items.every((x) => x.k === 'str')) return py.float(NaN)
            return py.str(c.v.items.map((x) => h.str(x)).join(sep))
          }
          return py.float(NaN)
        })
      })
      case 'cat': return def('others sep na_rep join', (p) => {
        if (given(p.join) && h.str(p.join!) !== 'left') throw h.err('NotImplementedError', `str.cat(join='${h.str(p.join!)}') is not available in this sandbox`)
        const sep = given(p.sep) ? h.str(p.sep!) : ''
        const na = given(p.na_rep) ? h.str(p.na_rep!) : null
        const text = (c: Cell): string | null => (typeof c === 'string' ? c : isMissing(c) ? na : null)
        if (!given(p.others)) {
          const items = this.s.values.map(text).filter((x): x is string => x !== null)
          return py.str(items.join(sep))
        }
        const o = p.others!
        if (o.k === 'list' && o.items.some((x) => x.k === 'list' || x.k === 'tuple') && !o.items.every((x) => x.k === 'str')) throw h.err('TypeError', 'others must be Series, Index, DataFrame, np.ndarray or list-like (either containing only strings or containing only objects of type Series/Index/np.ndarray[1-dim])')
        const lists: Cell[][] = o.k === 'list' && o.items.length && o.items.every((x) => isListLike(x)) ? o.items.map((x) => this.aligned(h, x)) : o.k === 'obj' && o.o instanceof Frame ? o.o.data.map((s) => this.aligned(h, py.obj(s))) : [this.aligned(h, o)]
        for (const l of lists) if (l.length !== this.s.length) throw h.err('ValueError', `If \`others\` contains arrays or lists (or other list-likes without an index), these must all be of the same length as the calling Series/Index.`)
        const cells = this.s.values.map((c, i) => {
          const parts = [text(c), ...lists.map((l) => text(l[i]))]
          return parts.some((x) => x === null) ? NaN : parts.join(sep)
        })
        return this.wrap(cells, 'object', h)
      })
      case 'count': return def('pat flags', (p) => {
        const re = reOf(p, 'g')
        return this.each(h, (x) => py.int((x.match(re) ?? []).length))
      })
      case 'find': case 'rfind': case 'index': case 'rindex': return def('sub start end', (p) => {
        const sv = need(h, p.sub, name, 'sub')
        if (sv.k !== 'str') throw h.err('TypeError', `expected a string object, not ${sv.k}`)
        const start = given(p.start) ? h.toInt(p.start!) : 0
        const end = given(p.end) ? h.toInt(p.end!) : null
        return this.each(h, (x) => {
          const at = pyFind(x, sv.v, start, end, name.startsWith('r'))
          if (at < 0 && name.endsWith('index')) throw h.err('ValueError', 'substring not found')
          return py.int(at)
        })
      })
      case 'findall': return def('pat flags', (p) => {
        const re = reOf(p, 'g')
        return this.each(h, (x) => py.list([...x.matchAll(re)].map((m) => {
          if (m.length === 1) return py.str(m[0])
          if (m.length === 2) return py.str(m[1] ?? '')
          return py.tuple(m.slice(1).map((g) => py.str(g ?? '')))
        })))
      })
      case 'extract': return def('pat flags expand', (p) => {
        const re = reOf(p)
        const names = groupNames(re)
        if (!names.length) throw h.err('ValueError', 'pattern contains no capture groups')
        const expand = flag(h, p.expand, true)
        if (!expand && names.length === 1) return this.each(h, (x) => {
          const m = re.exec(x)
          return m && m[1] !== undefined ? py.str(m[1]) : py.float(NaN)
        })
        const found = this.s.values.map((c): Cell[] => {
          if (typeof c !== 'string') return names.map(() => NaN)
          const m = re.exec(c)
          return m ? names.map((_, j) => m[j + 1] ?? NaN) : names.map(() => NaN)
        })
        const cols = names.map((n, j) => n ?? j)
        const isRange = names.every((n) => n === null)
        return py.obj(new Frame(isRange ? IndexObj.range(cols.length) : new IndexObj(cols), names.map((_, j) => new Series(found.map((r) => r[j]), this.s.index, cols[j], 'object')), this.s.index))
      })
      case 'extractall': return def('pat flags', (p) => {
        const re = reOf(p, 'g')
        const names = groupNames(re)
        if (!names.length) throw h.err('ValueError', 'pattern contains no capture groups')
        const rows: { at: number; match: number; g: Cell[] }[] = []
        this.s.values.forEach((c, i) => {
          if (typeof c !== 'string') return
          let k = 0
          for (const m of c.matchAll(re)) rows.push({ at: i, match: k++, g: names.map((_, j) => m[j + 1] ?? NaN) })
        })
        const index = IndexObj.fromArrays([rows.map((r) => this.s.index.labels[r.at]), rows.map((r) => r.match)], [this.s.index.name, 'match'], [this.s.index.dt, 'int64'])
        const cols = names.map((n, j) => n ?? j)
        return py.obj(new Frame(names.every((n) => n === null) ? IndexObj.range(cols.length) : new IndexObj(cols), names.map((_, j) => new Series(rows.map((r) => r.g[j]), index, cols[j], 'object')), index))
      })
      case 'get_dummies': return def('sep', (p) => {
        const sep = given(p.sep) ? h.str(p.sep!) : '|'
        const lists = this.s.values.map((c) => (typeof c === 'string' ? c.split(sep) : []))
        const tags = [...new Set(lists.flat())]
        const sorted = sortOrder([tags], [true]).map((i) => tags[i])
        return py.obj(new Frame(sorted, sorted.map((t) => new Series(lists.map((l) => (l.includes(t) ? 1 : 0)), this.s.index, t, 'int64')), this.s.index))
      })
      case 'repeat': return def('repeats', (p) => {
        const r = need(h, p.repeats, name, 'repeats')
        const times = isListLike(r) ? cellsOf(h, r).map(Number) : null
        if (times && times.length !== this.s.length) throw h.err('ValueError', 'operands could not be broadcast together')
        let i = -1
        return this.eachAny(h, (c) => {
          i++
          return typeof c === 'string' ? py.str(c.repeat(times ? times[i] : h.toInt(r))) : py.float(NaN)
        })
      })
      case 'removeprefix': return def('prefix', (p) => {
        const q = h.str(need(h, p.prefix, name, 'prefix'))
        return this.each(h, (x) => py.str(x.startsWith(q) ? x.slice(q.length) : x))
      })
      case 'removesuffix': return def('suffix', (p) => {
        const q = h.str(need(h, p.suffix, name, 'suffix'))
        return this.each(h, (x) => py.str(q && x.endsWith(q) ? x.slice(0, -q.length) : x))
      })
      case 'normalize': return def('form', (p) => {
        const form = h.str(need(h, p.form, name, 'form'))
        return this.each(h, (x) => py.str(x.normalize(form)))
      })
    }
    return undefined
  }
  private aligned(h: Host, o: Value): Cell[] {
    if (o.k === 'obj' && o.o instanceof Series) return reindexCells(o.o, this.s.index).cells
    return cellsOf(h, o)
  }
  /** s.str[0], s.str[:2] */
  getItem(idx: Value, h: Host): Value {
    if (idx.k === 'slice') {
      return this.eachAny(h, (c) => {
        if (typeof c === 'string') return py.str(sliceRange(h, idx, c.length).map((i) => c[i]).join(''))
        if (c instanceof PyCell && c.v.k === 'list') {
          const items = c.v.items
          return py.list(sliceRange(h, idx, items.length).map((i) => items[i]))
        }
        return py.float(NaN)
      })
    }
    return h.call(this.getAttr('get', h)!, [idx])
  }
}

export const strAccessor = (h: Host, s: Series): StrAccessor => {
  if (s.dtype !== 'object' || (s.length && s.values.every((c) => c instanceof PyCell && c.v.k === 'float'))) throw h.err('AttributeError', 'Can only use .str accessor with string values!')
  return new StrAccessor(s)
}

export const indexStrAccessor = (h: Host, ix: IndexObj): StrAccessor => {
  if (ix.dt !== 'object') throw h.err('AttributeError', 'Can only use .str accessor with string values!')
  return new StrAccessor(new Series(ix.labels, IndexObj.range(ix.length), ix.name, 'object'), ix)
}

