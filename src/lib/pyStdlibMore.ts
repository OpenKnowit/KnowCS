// The standard library modules few programs import. They are a chunk of their own, fetched when a program names one
// (pyLazy.ts), so that the interpreter every Python page needs stays small.
import { PyError } from './ndarray'
import { NOT_IMPLEMENTED, PyDict, PyObj, formatSpec, isNum, makeSet, numOf, py, repr, str, typeName } from './minipy'
import type { ClassMeta, ClassValue, Fn, Host, HostClass, InstValue, Kw, PyLib, Value } from './minipy'
import { RARE_STDLIB } from './pyLazy'
import { fullPath, stdlibParts } from './pyStdlib'

const define = () => {
const { NONE, int, float, bool, tuple, str: S, list: L } = py
// (a local name: vitest's module transform loses an imported base class inside a function)
const Base = PyObj
const {
  guarded, F, iter, fromGen, mod, subclassOf, takingKeywords, refuse, TypingAlias, FieldSpec, pad2, strftime,
  filesOf, dirsOf, fdsOf, CWD, TEMP, isDirAt, isFileAt, namesIn, existsAt, noSuchFile, fileExists, isADir, notADir, checkParents,
  pathOr, fspath, TextFile, Pickled, BinaryFile, NO_BYTES, openFile, textOptions, removeTree, mkdirAt, makedirs, renameAt,
} = stdlibParts()
type TextFile = InstanceType<typeof TextFile>
type BinaryFile = InstanceType<typeof BinaryFile>

// ---------------------------------------------------------------- helpers for classes a program can subclass

/** a library class made of plain functions: self comes first, as in Python */
const libClass = (module: string, name: string, methods: Record<string, (args: Value[], kw: Kw) => Value>, bases: Value[] = []): ClassValue => ({
  k: 'class', name, module, bases, host: null, builtin: true,
  ns: new Map(Object.entries(methods).map(([n, call]): [string, Value] => [n, { k: 'fn', name: n, bindable: true, call: guarded(n, call) }])),
})
const excIn = (module: string, name: string, base: ClassValue): ClassValue => ({ k: 'class', name, module, bases: [base], ns: new Map(), host: base.host, builtin: true })
const raiseWith = (h: Host, cls: ClassValue, args: Value[], attrs: Record<string, Value>, state: Record<string, unknown> = {}): never => {
  try {
    return h.raise(cls, args)
  } catch (e) {
    const inst = (e as PyError).exc as InstValue
    for (const n in attrs) inst.attrs.set(n, attrs[n])
    Object.assign(inst.state, state)
    throw e
  }
}
/** is this caught error one of these built-in exceptions, or a subclass of one? */
const caught = (h: Host, e: unknown, ...names: string[]): boolean => {
  if (!(e instanceof PyError) || e.fatal) return false
  const cls = (e.exc as InstValue | undefined)?.cls ?? h.excClass(e.pyType)
  const walk = (c: ClassValue): boolean => names.includes(c.name) || c.bases.some((b) => b.k === 'class' && walk(b))
  return walk(cls)
}
const isCallable = (h: Host, v: Value) => h.truthy(h.call((h.module('builtins') as { attrs: Record<string, Value> }).attrs.callable, [v]))
const randomChars = (alphabet: string, n: number) => Array.from({ length: n }, () => alphabet[Math.floor(Math.random() * alphabet.length)]).join('')

// ---------------------------------------------------------------- statistics

const makeStatistics = (h: Host): Value => {
  const Err = subclassOf(h, 'StatisticsError', 'ValueError')
  const nums = (v: Value, what: string, least = 1) => {
    const items = h.iterate(v)
    if (items.length < least) h.raise(Err, [S(least === 1 ? `${what} requires at least one data point` : `${what} requires at least two data points`)])
    return items
  }
  const allInt = (items: Value[]) => items.every((x) => x.k === 'int' || x.k === 'bool')
  /** statistics returns an int when the data are ints and the result is whole */
  const num = (x: number, items: Value[]) => (allInt(items) && Number.isInteger(x) ? int(x) : float(x))
  const meanOf = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
  const sumSq = (xs: number[]) => {
    const m = meanOf(xs)
    return xs.reduce((s, x) => s + (x - m) ** 2, 0)
  }
  const sortedNums = (items: Value[]) => items.map((x) => h.num(x)).sort((a, b) => a - b)
  return mod('statistics', {
    StatisticsError: Err,
    mean: F('mean', (a) => {
      const items = nums(a[0], 'mean')
      return num(meanOf(items.map((x) => h.num(x))), items)
    }),
    fmean: F('fmean', (a) => float(meanOf(nums(a[0], 'fmean').map((x) => h.num(x))))),
    median: F('median', (a) => {
      const items = nums(a[0], 'median')
      const s = sortedNums(items)
      const n = s.length
      return n % 2 ? num(s[(n - 1) / 2], items) : float((s[n / 2 - 1] + s[n / 2]) / 2)
    }),
    median_low: F('median_low', (a) => {
      const items = nums(a[0], 'median_low')
      const s = sortedNums(items)
      return num(s[Math.floor((s.length - 1) / 2)], items)
    }),
    median_high: F('median_high', (a) => {
      const items = nums(a[0], 'median_high')
      const s = sortedNums(items)
      return num(s[Math.floor(s.length / 2)], items)
    }),
    mode: F('mode', (a) => {
      const items = nums(a[0], 'mode')
      const seen = new PyDict()
      for (const x of items) seen.set(x, int(numOf(seen.get(x) ?? int(0)) + 1))
      return seen.items().reduce((best, e) => (numOf(e[1]) > numOf(best[1]) ? e : best))[0]
    }),
    multimode: F('multimode', (a) => {
      const seen = new PyDict()
      for (const x of h.iterate(a[0])) seen.set(x, int(numOf(seen.get(x) ?? int(0)) + 1))
      const top = Math.max(0, ...seen.values().map(numOf))
      return L(seen.items().filter((e) => numOf(e[1]) === top).map((e) => e[0]))
    }),
    variance: F('variance', (a) => {
      const items = nums(a[0], 'variance', 2)
      return num(sumSq(items.map((x) => h.num(x))) / (items.length - 1), items)
    }),
    stdev: F('stdev', (a) => {
      const items = nums(a[0], 'stdev', 2)
      return float(Math.sqrt(sumSq(items.map((x) => h.num(x))) / (items.length - 1)))
    }),
    pvariance: F('pvariance', (a) => {
      const items = nums(a[0], 'pvariance')
      return num(sumSq(items.map((x) => h.num(x))) / items.length, items)
    }),
    pstdev: F('pstdev', (a) => {
      const items = nums(a[0], 'pstdev')
      return float(Math.sqrt(sumSq(items.map((x) => h.num(x))) / items.length))
    }),
  })
}

/** how two dates, datetimes or timedeltas compare, given the first minus the second */
const compared = (op: string, diff: number): Value | undefined => {
  switch (op) {
    case '<': return bool(diff < 0)
    case '>': return bool(diff > 0)
    case '<=': return bool(diff <= 0)
    case '>=': return bool(diff >= 0)
    case '==': return bool(diff === 0)
    case '!=': return bool(diff !== 0)
  }
  return undefined
}

class TimeDelta extends Base {
  readonly cls = 'datetime.timedelta'
  ms: number
  constructor(ms: number) {
    super()
    this.ms = ms
  }
  pickled() { return this }
  hashKey() { return `timedelta:${this.ms}` }
  repr() {
    const days = Math.floor(this.ms / 86400000)
    const rest = this.ms - days * 86400000
    const seconds = Math.floor(rest / 1000)
    const micros = Math.round((rest - seconds * 1000) * 1000)
    const parts = [days && `days=${days}`, seconds && `seconds=${seconds}`, micros && `microseconds=${micros}`].filter(Boolean)
    return `datetime.timedelta(${parts.join(', ') || '0'})`
  }
  str() {
    const total = Math.abs(this.ms) / 1000
    const days = Math.floor(total / 86400)
    const rest = total - days * 86400
    const h = Math.floor(rest / 3600)
    const m = Math.floor((rest % 3600) / 60)
    const s = rest % 60
    const sec = Number.isInteger(s) ? pad2(s) : s.toFixed(6).padStart(9, '0')
    return `${this.ms < 0 ? '-' : ''}${days ? `${days} day${days > 1 ? 's' : ''}, ` : ''}${h}:${pad2(m)}:${sec}`
  }
  getAttr(name: string): Value | undefined {
    switch (name) {
      case 'total_seconds': return F(name, () => float(this.ms / 1000))
      case 'days': return int(Math.floor(this.ms / 86400000))
      case 'seconds': return int(Math.floor((this.ms % 86400000) / 1000))
      case 'microseconds': return int((this.ms % 1000) * 1000)
    }
    return undefined
  }
  binop(op: string, o: Value, reflected: boolean): Value | undefined {
    if (o.k === 'obj' && o.o instanceof TimeDelta) {
      if (op === '+') return { k: 'obj', o: new TimeDelta(this.ms + o.o.ms) }
      if (op === '-') return { k: 'obj', o: new TimeDelta(this.ms - o.o.ms) }
      return compared(op, reflected ? o.o.ms - this.ms : this.ms - o.o.ms)
    }
    return undefined
  }
}

class DateTime extends Base {
  readonly cls = 'datetime.datetime'
  d: Date
  dateOnly: boolean
  constructor(d: Date, dateOnly = false) {
    super()
    // a date has no time of day, however it was made: two of the same day are equal
    this.d = dateOnly ? new Date(d.getFullYear(), d.getMonth(), d.getDate()) : d
    this.dateOnly = dateOnly
  }
  pickled() { return this }
  // a date and a datetime are never equal, so they never share a key
  hashKey() { return `${this.dateOnly ? 'date' : 'datetime'}:${this.d.getTime()}` }
  repr() {
    const d = this.d
    return this.dateOnly
      ? `datetime.date(${d.getFullYear()}, ${d.getMonth() + 1}, ${d.getDate()})`
      : `datetime.datetime(${d.getFullYear()}, ${d.getMonth() + 1}, ${d.getDate()}, ${d.getHours()}, ${d.getMinutes()}, ${d.getSeconds()})`
  }
  str() { return strftime(this.d, this.dateOnly ? '%Y-%m-%d' : '%Y-%m-%d %H:%M:%S') }
  getAttr(name: string, h: Host): Value | undefined {
    const d = this.d
    switch (name) {
      case 'year': return int(d.getFullYear())
      case 'month': return int(d.getMonth() + 1)
      case 'day': return int(d.getDate())
      case 'hour': return int(d.getHours())
      case 'minute': return int(d.getMinutes())
      case 'second': return int(d.getSeconds())
      case 'microsecond': return int(d.getMilliseconds() * 1000)
      case 'strftime': return F(name, (a) => S(strftime(d, (a[0] as { v: string }).v)))
      case 'isoformat': return F(name, () => S(strftime(d, this.dateOnly ? '%Y-%m-%d' : '%Y-%m-%dT%H:%M:%S')))
      case 'timestamp': return F(name, () => float(d.getTime() / 1000))
      case 'weekday': return F(name, () => int((d.getDay() + 6) % 7))
      case 'date': return F(name, () => ({ k: 'obj', o: new DateTime(d, true) }))
    }
    void h
    return undefined
  }
  binop(op: string, o: Value, reflected: boolean): Value | undefined {
    if (o.k === 'obj' && o.o instanceof DateTime) {
      const diff = (this.d.getTime() - o.o.d.getTime()) * (reflected ? -1 : 1)
      if (this.dateOnly !== o.o.dateOnly) {
        if (op === '==' || op === '!=') return bool(op === '!=')
        return undefined
      }
      if (op === '-') return { k: 'obj', o: new TimeDelta(diff) }
      return compared(op, diff)
    }
    if (o.k === 'obj' && o.o instanceof TimeDelta && (op === '+' || op === '-')) {
      return { k: 'obj', o: new DateTime(new Date(this.d.getTime() + (op === '+' ? o.o.ms : -o.o.ms)), this.dateOnly) }
    }
    return undefined
  }
}

const makeDatetime = (h: Host): Value => {
  const typeWith = (name: string, call: (a: Value[]) => Value, attrs: Record<string, Value>): Value => {
    const t: Value & { k: 'fn' } = { k: 'fn', name, call, attrs: new Map(Object.entries(attrs)) }
    return t
  }
  const now = (): Value => ({ k: 'obj', o: new DateTime(new Date()) })
  const dt = typeWith('datetime', (a) => ({ k: 'obj', o: new DateTime(new Date(h.toInt(a[0]), h.toInt(a[1]) - 1, h.toInt(a[2]), a[3] ? h.toInt(a[3]) : 0, a[4] ? h.toInt(a[4]) : 0, a[5] ? h.toInt(a[5]) : 0)) }), {
    now: F('now', now), today: F('today', now), utcnow: F('utcnow', now),
  })
  const date = typeWith('date', (a) => ({ k: 'obj', o: new DateTime(new Date(h.toInt(a[0]), h.toInt(a[1]) - 1, h.toInt(a[2])), true) }), {
    today: F('today', () => ({ k: 'obj', o: new DateTime(new Date(), true) })),
  })
  return mod('datetime', {
    datetime: dt, date,
    timedelta: F('timedelta', (a, kw) => {
      const n = (v: Value | undefined) => (v ? h.num(v) : 0)
      const days = n(kw.days ?? a[0])
      const secs = n(kw.seconds ?? a[1])
      return { k: 'obj', o: new TimeDelta((days * 86400 + secs + n(kw.minutes) * 60 + n(kw.hours) * 3600 + n(kw.weeks) * 604800) * 1000 + n(kw.milliseconds) + n(kw.microseconds) / 1000) }
    }),
  })
}

const makeCsv = (h: Host): Value => {
  const parseLine = (line: string, delim: string): string[] => {
    const out: string[] = []
    let cur = ''
    let quoted = false
    for (let i = 0; i < line.length; i++) {
      const c = line[i]
      if (quoted) {
        if (c === '"' && line[i + 1] === '"') {
          cur += '"'
          i++
        } else if (c === '"') quoted = false
        else cur += c
      } else if (c === '"') quoted = true
      else if (c === delim) {
        out.push(cur)
        cur = ''
      } else cur += c
    }
    out.push(cur)
    return out
  }
  const delimOf = (kw: Kw) => (kw.delimiter?.k === 'str' ? kw.delimiter.v : ',')
  const lines = (src: Value): (() => string | undefined) => {
    const next = h.iterOf(src)
    return () => {
      for (let x = next(); x !== undefined; x = next()) {
        const t = (x as { v: string }).v.replace(/\r?\n$/, '')
        if (t !== '') return t
      }
      return undefined
    }
  }
  const quote = (v: Value, delim: string) => {
    const t = v.k === 'none' ? '' : h.str(v)
    return t.includes(delim) || t.includes('"') || t.includes('\n') ? `"${t.replace(/"/g, '""')}"` : t
  }
  const lineEnd = (kw: Kw) => (kw.lineterminator?.k === 'str' ? kw.lineterminator.v : '\r\n')
  const writeRow = (f: Value, cells: string[], kw: Kw) => h.call(h.getAttr(f, 'write'), [S(cells.join(delimOf(kw)) + lineEnd(kw))])
  return mod('csv', {
    reader: F('reader', (a, kw) => {
      const next = lines(a[0])
      return iter('_csv.reader', () => {
        const t = next()
        return t === undefined ? undefined : L(parseLine(t, delimOf(kw)).map(S))
      })
    }),
    writer: F('writer', (a, kw) => mod('_csv.writer', {
      writerow: F('writerow', (b) => writeRow(a[0], h.iterate(b[0]).map((x) => quote(x, delimOf(kw))), kw)),
      writerows: F('writerows', (b) => {
        for (const row of h.iterate(b[0])) writeRow(a[0], h.iterate(row).map((x) => quote(x, delimOf(kw))), kw)
        return NONE
      }),
    })),
    DictReader: F('DictReader', (a, kw) => {
      const next = lines(a[0])
      const given = kw.fieldnames && kw.fieldnames.k !== 'none' ? h.iterate(kw.fieldnames).map((x) => (x as { v: string }).v) : null
      const names = given ?? parseLine(next() ?? '', delimOf(kw))
      return iter('csv.DictReader', () => {
        const t = next()
        if (t === undefined) return undefined
        const cells = parseLine(t, delimOf(kw))
        return { k: 'dict', d: PyDict.from(names.map((n, i) => [S(n), i < cells.length ? S(cells[i]) : NONE])) }
      })
    }),
    DictWriter: F('DictWriter', (a, kw) => {
      const names = h.iterate(kw.fieldnames ?? a[1]).map((x) => (x as { v: string }).v)
      const row = (d: Value) => writeRow(a[0], names.map((n) => quote((d as { d: PyDict }).d?.get(S(n)) ?? S(''), delimOf(kw))), kw)
      return mod('csv.DictWriter', {
        writeheader: F('writeheader', () => writeRow(a[0], names.map((n) => quote(S(n), delimOf(kw))), kw)),
        writerow: F('writerow', (b) => row(b[0])),
        writerows: F('writerows', (b) => { for (const d of h.iterate(b[0])) row(d); return NONE }),
      })
    }),
    QUOTE_MINIMAL: int(0),
  })
}

// ---------------------------------------------------------------- fnmatch

const fnmatchRe = (pat: string): RegExp => {
  let out = ''
  for (let i = 0; i < pat.length; i++) {
    const c = pat[i]
    if (c === '*') out += '[\\s\\S]*'
    else if (c === '?') out += '[\\s\\S]'
    else if (c === '[') {
      let j = i + 1
      if (pat[j] === '!') j++
      if (pat[j] === ']') j++
      while (j < pat.length && pat[j] !== ']') j++
      if (j >= pat.length) out += '\\['
      else {
        const inside = pat.slice(i + 1, j).replace(/\\/g, '\\\\').replace(/[[\]]/g, '\\$&')
        out += inside.startsWith('!') ? `[^${inside.slice(1)}]` : `[${inside.startsWith('^') ? '\\' : ''}${inside}]`
        i = j
      }
    } else out += c.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&')
  }
  try {
    return new RegExp(`^(?:${out})$`, 'u')
  } catch {
    // [z-a] names nothing
    return /(?!)/
  }
}
const fnmatchOne = (name: string, pat: string) => fnmatchRe(pat).test(name)

const makeFnmatch = (h: Host): Value => {
  const text = (v: Value) => fspath(h, v)
  return mod('fnmatch', {
    fnmatch: F('fnmatch', (a) => bool(fnmatchOne(text(a[0]), text(a[1])))),
    fnmatchcase: F('fnmatchcase', (a) => bool(fnmatchOne(text(a[0]), text(a[1])))),
    filter: F('filter', (a) => L(h.iterate(a[0]).filter((x) => fnmatchOne(text(x), text(a[1]))))),
  })
}

// ---------------------------------------------------------------- pathlib

interface PathKinds { pure: Value; concrete: Value }

/** PurePosixPath and PosixPath: a root and the parts after it, as pathlib parses them */
class PathObj extends Base {
  readonly cls: string
  pickled() { return this }
  root: string
  segs: string[]
  pure: boolean
  kinds: PathKinds
  constructor(root: string, segs: string[], pure: boolean, kinds: PathKinds) {
    super()
    this.root = root
    this.segs = segs
    this.pure = pure
    this.kinds = kinds
    this.cls = pure ? 'PurePosixPath' : 'PosixPath'
  }
  text() { return this.root + this.segs.join('/') || '.' }
  repr() { return `${this.cls}(${repr(S(this.text()))})` }
  str() { return this.text() }
  format(spec: string) {
    if (spec) throw new PyError('TypeError', `unsupported format string passed to ${this.cls}.__format__`)
    return this.text()
  }
  fspath() { return this.text() }
  hashKey() { return `p${this.text()}` }
  typeValue() { return this.pure ? this.kinds.pure : this.kinds.concrete }
  isa(name: string) { return name === 'pathlib.PurePath' || name === 'pathlib.PurePosixPath' || name === 'os.PathLike' || (!this.pure && (name === 'pathlib.Path' || name === 'pathlib.PosixPath')) }
  make(root: string, segs: string[]) { return new PathObj(root, segs, this.pure, this.kinds) }
  v() { return { k: 'obj', o: this } as Value }
  name() { return this.segs[this.segs.length - 1] ?? '' }
  suffix() {
    const n = this.name()
    const i = n.lastIndexOf('.')
    return i > 0 && i < n.length - 1 ? n.slice(i) : ''
  }
  stem() {
    const n = this.name()
    const i = n.lastIndexOf('.')
    return i > 0 && i < n.length - 1 ? n.slice(0, i) : n
  }
  parts() { return this.root ? [this.root, ...this.segs] : this.segs }
  parent() { return this.segs.length ? this.make(this.root, this.segs.slice(0, -1)) : this }
  fromParts(parts: string[]) {
    return this.root ? this.make(this.root, parts.slice(1)) : this.make('', parts)
  }
  join(others: Value[], h: Host) { return parsePath(h, [this.v(), ...others], this.pure, this.kinds) }
  binop(op: string, other: Value, reflected: boolean, h: Host): Value | undefined {
    if (op === '/') {
      if (pathOr(h, other) === null) return undefined
      return parsePath(h, reflected ? [other, this.v()] : [this.v(), other], this.pure, this.kinds).v()
    }
    if (other.k !== 'obj' || !(other.o instanceof PathObj)) return undefined
    const [a, b] = reflected ? [other.o.parts(), this.parts()] : [this.parts(), other.o.parts()]
    let c = 0
    for (let i = 0; i < Math.min(a.length, b.length) && !c; i++) c = a[i] < b[i] ? -1 : a[i] > b[i] ? 1 : 0
    c ||= a.length - b.length
    switch (op) {
      case '==': return bool(c === 0)
      case '!=': return bool(c !== 0)
      case '<': return bool(c < 0)
      case '>': return bool(c > 0)
      case '<=': return bool(c <= 0)
      case '>=': return bool(c >= 0)
    }
    return undefined
  }
  replaceName(name: string, h: Host) {
    if (!this.name()) throw h.err('ValueError', `${this.repr()} has an empty name`)
    return this.make(this.root, [...this.segs.slice(0, -1), name]).v()
  }
  withName(name: string, h: Host) {
    if (!this.name()) throw h.err('ValueError', `${this.repr()} has an empty name`)
    const parsed = name.split('/').filter((x) => x && x !== '.')
    if (!name || name.endsWith('/') || name.startsWith('/') || parsed.length !== 1) throw h.err('ValueError', `Invalid name ${repr(S(name))}`)
    return this.replaceName(name, h)
  }
  withSuffix(suffix: string, h: Host) {
    if (suffix.includes('/') || (suffix && !suffix.startsWith('.')) || suffix === '.') throw h.err('ValueError', `Invalid suffix ${repr(S(suffix))}`)
    if (!this.name()) throw h.err('ValueError', `${this.repr()} has an empty name`)
    const old = this.suffix()
    return this.replaceName((old ? this.name().slice(0, -old.length) : this.name()) + suffix, h)
  }
  relativeTo(others: Value[], h: Host) {
    if (!others.length) throw h.err('TypeError', 'need at least one argument')
    const to = parsePath(h, others, this.pure, this.kinds)
    const mine = this.root ? ['', this.root, ...this.segs] : this.segs
    const theirs = to.root ? ['', to.root, ...to.segs] : to.segs
    const n = theirs.length
    const fits = n === 0 ? !this.root : theirs.every((x, i) => mine[i] === x)
    if (!fits) return { error: `${repr(S(this.text()))} is not in the subpath of ${repr(S(to.text()))} OR one path is relative and the other is absolute.` }
    return { path: this.make(n === 1 ? this.root : '', mine.slice(n)) }
  }
  match(pattern: string, h: Host) {
    const pat = parsePath(h, [S(pattern)], this.pure, this.kinds)
    const want = pat.parts()
    if (!want.length) throw h.err('ValueError', 'empty pattern')
    if (pat.root && pat.root !== this.root) return false
    const parts = this.parts()
    if (pat.root ? want.length !== parts.length : want.length > parts.length) return false
    for (let i = 1; i <= want.length; i++) if (!fnmatchOne(parts[parts.length - i], want[want.length - i])) return false
    return true
  }
  /** pathlib's selectors: a part with no wildcard, a part with one, and ** */
  *select(parts: string[], h: Host): Generator<PathObj, void, undefined> {
    if (!parts.length) {
      yield this
      return
    }
    if (!isDirAt(h, fullPath(this.text()))) return
    const [first, ...rest] = parts
    const child = (n: string) => this.make(this.root, [...this.segs, n])
    if (first === '**') {
      const seen = new Set<string>()
      const dirs = function* (p: PathObj): Generator<PathObj, void, undefined> {
        yield p
        for (const n of namesIn(h, fullPath(p.text()))) {
          const c = p.make(p.root, [...p.segs, n])
          if (isDirAt(h, fullPath(c.text()))) yield* dirs(c)
        }
      }
      for (const d of dirs(this)) for (const found of d.select(rest, h)) {
        if (seen.has(found.text())) continue
        seen.add(found.text())
        yield found
      }
    } else if (/[*?[]/.test(first)) {
      for (const n of namesIn(h, fullPath(this.text()))) {
        const c = child(n)
        if (rest.length && !isDirAt(h, fullPath(c.text()))) continue
        if (fnmatchOne(n, first)) yield* c.select(rest, h)
      }
    } else {
      const c = child(first)
      if (rest.length ? isDirAt(h, fullPath(c.text())) : existsAt(h, c.text())) yield* c.select(rest, h)
    }
  }
  globbed(pattern: Value | undefined, recursive: boolean, h: Host): Value {
    const text = fspath(h, pattern)
    if (!text && !recursive) throw h.err('ValueError', `Unacceptable pattern: ${repr(S(text))}`)
    if (text.startsWith('/')) throw h.err('NotImplementedError', 'Non-relative patterns are unsupported')
    const parts = text.split('/').filter((x) => x && x !== '.')
    for (const p of parts) if (p !== '**' && p.includes('**')) throw h.err('ValueError', "Invalid pattern: '**' can only be an entire path component")
    if (!parts.length && !recursive) throw h.err('ValueError', `Unacceptable pattern: ${repr(S(text))}`)
    const found = this.select(recursive ? ['**', ...parts] : parts, h)
    return iter('generator', () => {
      const r = found.next()
      return r.done ? undefined : r.value.v()
    })
  }
  resolved() { return new PathObj('/', fullPath(this.text()).split('/').filter(Boolean), false, this.kinds) }
  getAttr(name: string, h: Host): Value | undefined {
    const text = this.text()
    const m = (call: (a: Value[], kw: Kw) => Value) => F(name, call)
    switch (name) {
      case 'name': return S(this.name())
      case 'stem': return S(this.stem())
      case 'suffix': return S(this.suffix())
      case 'suffixes': {
        const n = this.name()
        return n.endsWith('.') ? L([]) : L(n.replace(/^\.+/, '').split('.').slice(1).map((x) => S(`.${x}`)))
      }
      case 'parent': return this.parent().v()
      case 'parents': return { k: 'obj', o: new PathParents(this) }
      case 'parts': return tuple(this.parts().map(S))
      case 'root': return S(this.root)
      case 'anchor': return S(this.root)
      case 'drive': return S('')
      case 'as_posix': return m(() => S(text))
      case '__fspath__': return m(() => S(text))
      case 'as_uri': return m(() => {
        if (!this.root) throw h.err('ValueError', "relative path can't be expressed as a file URI")
        return S(`file://${encodeURI(text)}`)
      })
      case 'is_absolute': return m(() => bool(!!this.root))
      case 'is_reserved': return m(() => bool(false))
      case 'joinpath': return m((a) => this.join(a, h).v())
      case 'with_name': return m((a) => this.withName(h.str(a[0]), h))
      case 'with_stem': return m((a) => this.withName(h.str(a[0]) + this.suffix(), h))
      case 'with_suffix': return m((a) => this.withSuffix(h.str(a[0]), h))
      case 'relative_to': return m((a) => {
        const r = this.relativeTo(a, h)
        if (r.error !== undefined) throw h.err('ValueError', r.error)
        return r.path.v()
      })
      case 'is_relative_to': return m((a) => bool(this.relativeTo(a, h).error === undefined))
      case 'match': return m((a) => bool(this.match(h.str(a[0]), h)))
    }
    if (this.pure) return undefined
    const full = fullPath(text)
    const files = filesOf(h)
    const openIt = (a: Value[], kw: Kw) => openFile(h, this.v(), kw.mode ?? a[0], textOptions(a, kw, 2))
    const read = (kw: Kw) => {
      const f = openFile(h, this.v(), S('r'), textOptions([], kw, 0))
      try {
        return h.call(h.getAttr({ k: 'obj', o: f }, 'read'), [])
      } finally {
        f.finish()
      }
    }
    const statBool = (f: () => boolean) => m(() => bool(f()))
    switch (name) {
      case 'exists': return statBool(() => existsAt(h, text))
      case 'is_file': return statBool(() => isFileAt(h, full))
      case 'is_dir': return statBool(() => isDirAt(h, full))
      case 'is_symlink': case 'is_mount': case 'is_socket': case 'is_fifo': case 'is_block_device': case 'is_char_device': return statBool(() => false)
      case 'stat': case 'lstat': return m(() => {
        if (!existsAt(h, text)) noSuchFile(h, text)
        return h.call(h.getAttr(h.module('os'), 'stat'), [S(text)])
      })
      case 'open': return m((a, kw) => ({ k: 'obj', o: openIt(a, kw) }))
      case 'read_text': return m((_a, kw) => read(kw))
      case 'write_text': return m((a, kw) => {
        const data = kw.data ?? a[0]
        if (data?.k !== 'str') throw h.err('TypeError', `data must be str, not ${typeName(data ?? NONE)}`)
        const f = openFile(h, this.v(), S('w'), textOptions(a, kw, 1))
        ;(f as TextFile).put(data.v)
        f.finish()
        return int(data.v.length)
      })
      case 'read_bytes': return m(() => {
        const f = openFile(h, this.v(), S('rb'), {})
        f.finish()
        return { k: 'obj', o: new Pickled((f as BinaryFile).items) }
      })
      case 'write_bytes': return m((a, kw) => {
        const data = kw.data ?? a[0]
        if (data?.k !== 'obj' || !(data.o instanceof Pickled)) throw h.err('NotImplementedError', NO_BYTES)
        const f = openFile(h, this.v(), S('wb'), {}) as BinaryFile
        f.write(data, h)
        f.finish()
        // the count is not known here: see BinaryFile.write
        return NONE
      })
      case 'touch': return m((a, kw) => {
        const ok = kw.exist_ok ?? a[1]
        if (isFileAt(h, full) || isDirAt(h, full)) {
          if (ok && !h.truthy(ok)) fileExists(h, text)
          return NONE
        }
        checkParents(h, text)
        files.set(full, '')
        return NONE
      })
      case 'mkdir': return m((a, kw) => {
        const parents = kw.parents ?? a[1]
        const existOk = kw.exist_ok ?? a[2]
        const ok = !!existOk && h.truthy(existOk)
        if (ok && isDirAt(h, full)) return NONE
        if (parents && h.truthy(parents)) makedirs(h, text, ok)
        else mkdirAt(h, text)
        return NONE
      })
      case 'rmdir': return m(() => h.call(h.getAttr(h.module('os'), 'rmdir'), [S(text)]))
      case 'unlink': return m((a, kw) => {
        const ok = kw.missing_ok ?? a[0]
        if (ok && h.truthy(ok) && !isFileAt(h, full) && !isDirAt(h, full)) return NONE
        return h.call(h.getAttr(h.module('os'), 'unlink'), [S(text)])
      })
      case 'rename': case 'replace': return m((a, kw) => {
        const target = kw.target ?? a[0]
        renameAt(h, text, fspath(h, target))
        return parsePath(h, [target], false, this.kinds).v()
      })
      case 'iterdir': return m(() => {
        const child = (n: string) => this.make(this.root, [...this.segs, n]).v()
        return fromGen('generator', (function* () {
          if (isFileAt(h, full)) notADir(h, text)
          if (!isDirAt(h, full)) noSuchFile(h, text)
          for (const n of namesIn(h, full)) yield child(n)
        })())
      })
      case 'glob': return m((a, kw) => this.globbed(kw.pattern ?? a[0], false, h))
      case 'rglob': return m((a, kw) => this.globbed(kw.pattern ?? a[0], true, h))
      case 'resolve': return m((a, kw) => {
        const strict = kw.strict ?? a[0]
        if (strict && h.truthy(strict) && !existsAt(h, text)) noSuchFile(h, text)
        return this.resolved().v()
      })
      case 'absolute': return m(() => (this.root ? this.v() : new PathObj('/', [...CWD.split('/').filter(Boolean), ...this.segs], false, this.kinds).v()))
      case 'expanduser': return m(() => (this.segs[0] === '~' && !this.root ? new PathObj('/', [...CWD.split('/').filter(Boolean), ...this.segs.slice(1)], false, this.kinds).v() : this.v()))
      case 'samefile': return m((a) => {
        if (!existsAt(h, text)) noSuchFile(h, text)
        const other = fspath(h, a[0])
        if (!existsAt(h, other)) noSuchFile(h, other)
        return bool(full === fullPath(other))
      })
      case 'chmod': case 'lchmod': return m(() => NONE)
    }
    return undefined
  }
}

class PathParents extends Base {
  readonly cls = '_PathParents'
  p: PathObj
  constructor(p: PathObj) {
    super()
    this.p = p
  }
  repr() { return `<${this.p.cls}.parents>` }
  len() { return this.p.segs.length }
  at(i: number) {
    // _parts[:-i - 1], which is what 3.10 does with a negative index too
    return this.p.fromParts(this.p.parts().slice(0, -(i < 0 ? i + this.len() : i) - 1)).v()
  }
  iter() { return Array.from({ length: this.len() }, (_, i) => this.at(i)) }
  getItem(idx: Value, h: Host): Value {
    if (idx.k === 'slice') {
      const all = this.iter()
      return h.call(h.getAttr(tuple(all), '__getitem__'), [idx])
    }
    const i = h.toInt(idx)
    if (i >= this.len() || i < -this.len()) throw h.raise(h.excClass('IndexError'), [int(i)])
    return this.at(i)
  }
}

const parsePath = (h: Host, args: Value[], pure: boolean, kinds: PathKinds): PathObj => {
  let root = ''
  let segs: string[] = []
  for (const a of args) {
    const t = fspath(h, a)
    if (t.startsWith('/')) {
      root = t.startsWith('//') && !t.startsWith('///') ? '//' : '/'
      segs = []
    }
    segs.push(...t.split('/').filter((x) => x && x !== '.'))
  }
  return new PathObj(root, segs, pure, kinds)
}

const makePathlib = (h: Host): Value => {
  const kinds: PathKinds = { pure: NONE, concrete: NONE }
  const pathType = (name: string, pure: boolean, parents: string[], attrs: Record<string, Value> = {}): Value => ({
    k: 'type', name: `pathlib.${name}`, attrs: new Map(Object.entries(attrs)), parents,
    call: (a) => parsePath(h, a, pure, kinds).v(),
  })
  const here = F('cwd', () => parsePath(h, [S(CWD)], false, kinds).v())
  const classAttrs = { cwd: here, home: F('home', () => parsePath(h, [S(CWD)], false, kinds).v()) }
  kinds.pure = pathType('PurePosixPath', true, ['PurePath'])
  kinds.concrete = pathType('PosixPath', false, ['Path', 'PurePath', 'PurePosixPath'], classAttrs)
  const windows = (name: string): Value => ({ k: 'type', name: `pathlib.${name}`, call: () => { throw h.err('NotImplementedError', `pathlib.${name} is not available in this sandbox: its paths are POSIX paths`) } })
  return mod('pathlib', {
    Path: pathType('Path', false, ['PurePath'], classAttrs),
    PurePath: pathType('PurePath', true, []),
    PosixPath: kinds.concrete,
    PurePosixPath: kinds.pure,
    WindowsPath: windows('WindowsPath'),
    PureWindowsPath: windows('PureWindowsPath'),
  })
}

// ---------------------------------------------------------------- glob, shutil, tempfile

const makeGlob = (h: Host): Value => {
  const hasMagic = (s: string) => /[*?[]/.test(s)
  const hidden = (s: string) => s.startsWith('.')
  const recursiveGlob = (s: string) => s === '**'
  const pathJoin = (a: string, b: string) => (b.startsWith('/') ? b : !a || a.endsWith('/') ? a + b : `${a}/${b}`)
  const dirExists = (p: string) => isDirAt(h, fullPath(p || '.'))
  const listing = (dir: string, dironly: boolean): string[] => {
    const full = fullPath(dir || '.')
    if (!isDirAt(h, full)) return []
    return namesIn(h, full).filter((n) => !dironly || isDirAt(h, `${full === '/' ? '' : full}/${n}`))
  }
  const rlist = function* (dir: string, dironly: boolean): Generator<string, void, undefined> {
    for (const x of listing(dir, dironly)) {
      if (hidden(x)) continue
      yield x
      for (const y of rlist(dir ? pathJoin(dir, x) : x, dironly)) yield pathJoin(x, y)
    }
  }
  const glob1 = function* (dir: string, pat: string, dironly: boolean): Generator<string, void, undefined> {
    for (const x of listing(dir, dironly)) if ((hidden(pat) || !hidden(x)) && fnmatchOne(x, pat)) yield x
  }
  const glob2 = function* (dir: string, _pat: string, dironly: boolean): Generator<string, void, undefined> {
    yield ''
    yield* rlist(dir, dironly)
  }
  const glob0 = function* (dir: string, base: string): Generator<string, void, undefined> {
    if (base ? existsAt(h, pathJoin(dir, base) || '.') : dirExists(dir)) yield base
  }
  const walk = function* (pathname: string, root: string, recursive: boolean, dironly: boolean): Generator<string, void, undefined> {
    const at = pathname.lastIndexOf('/') + 1
    const base = pathname.slice(at)
    const head = pathname.slice(0, at)
    const dirname = head && !/^\/+$/.test(head) ? head.replace(/\/+$/, '') : head
    if (!hasMagic(pathname)) {
      if (base ? existsAt(h, pathJoin(root, pathname)) : dirExists(pathJoin(root, dirname))) yield pathname
      return
    }
    if (!dirname) {
      yield* (recursive && recursiveGlob(base) ? glob2(root, base, dironly) : glob1(root, base, dironly))
      return
    }
    const dirs = dirname !== pathname && hasMagic(dirname) ? walk(dirname, root, recursive, true) : [dirname]
    for (const d of dirs) {
      const inside = pathJoin(root, d)
      const names = hasMagic(base) ? (recursive && recursiveGlob(base) ? glob2(inside, base, dironly) : glob1(inside, base, dironly)) : glob0(inside, base)
      for (const n of names) yield pathJoin(d, n)
    }
  }
  const run = function* (a: Value[], kw: Kw): Generator<Value, void, undefined> {
    refuse(h, 'glob.glob', kw, ['dir_fd'])
    const given = kw.pathname ?? a[0]
    if (given?.k !== 'str') throw h.err('TypeError', `'${typeName(given ?? NONE)}' object is not subscriptable`)
    const pathname = given.v
    const rootV = kw.root_dir ?? a[1]
    const root = rootV && rootV.k !== 'none' ? fspath(h, rootV) : ''
    const recursive = !!kw.recursive && h.truthy(kw.recursive)
    const skipEmpty = !pathname || (recursive && recursiveGlob(pathname.slice(0, 2)))
    for (const x of walk(pathname, root, recursive, false)) if (x || !skipEmpty) yield S(x)
  }
  return takingKeywords(mod('glob', {
    glob: F('glob', (a, kw) => {
      const out: Value[] = []
      for (const x of run(a, kw)) out.push(x)
      return L(out)
    }),
    iglob: F('iglob', (a, kw) => fromGen('generator', run(a, kw))),
    escape: F('escape', (a) => S(fspath(h, a[0]).replace(/[*?[]/g, '[$&]'))),
    has_magic: F('has_magic', (a) => bool(hasMagic(fspath(h, a[0])))),
  }), { glob: ['pathname', 'root_dir', 'dir_fd', 'recursive'], iglob: ['pathname', 'root_dir', 'dir_fd', 'recursive'] })
}

const makeShutil = (h: Host): Value => {
  const Err = excIn('shutil', 'Error', h.excClass('OSError'))
  const Same = excIn('shutil', 'SameFileError', Err)
  const files = filesOf(h)
  const fspath = (v: Value | undefined) => {
    const p = pathOr(h, v)
    if (p === null) throw h.err('TypeError', `stat: path should be string, bytes, os.PathLike or integer, not ${typeName(v ?? NONE)}`)
    return p
  }
  const base = (p: string) => {
    const t = p.replace(/\/+$/, '')
    return t.slice(t.lastIndexOf('/') + 1)
  }
  const copyFile = (src: string, dst: string) => {
    const [a, b] = [fullPath(src), fullPath(dst)]
    if (a === b && isFileAt(h, a)) h.raise(Same, [S(`${repr(S(src))} and ${repr(S(dst))} are the same file`)])
    if (isDirAt(h, a)) isADir(h, src)
    const text = files.get(a)
    if (text === undefined) return noSuchFile(h, src)
    if (isDirAt(h, b)) isADir(h, dst)
    checkParents(h, dst)
    files.set(b, text)
    return undefined
  }
  const copy = (name: string): Value => F(name, (a, kw) => {
    const src = fspath(kw.src ?? a[0])
    const dstV = kw.dst ?? a[1]
    const dst = fspath(dstV)
    if (isDirAt(h, fullPath(dst))) {
      const to = dst.endsWith('/') ? dst + base(src) : `${dst}/${base(src)}`
      copyFile(src, to)
      return S(to)
    }
    copyFile(src, dst)
    return dstV
  })
  const copyTree = (src: string, dst: string, existOk: boolean, ignore: Value | undefined) => {
    const a = fullPath(src)
    if (isFileAt(h, a)) notADir(h, src)
    if (!isDirAt(h, a)) noSuchFile(h, src)
    const walk = (from: string, to: string, fromText: string) => {
      const names = namesIn(h, from)
      const skip = ignore && ignore.k !== 'none' ? new Set(h.iterate(h.call(ignore, [S(fromText), L(names.map(S))])).map((x) => h.str(x))) : new Set<string>()
      if (isFileAt(h, to)) fileExists(h, to)
      if (isDirAt(h, to) && !existOk) fileExists(h, fromText === src ? dst : to)
      if (!isDirAt(h, to)) makedirs(h, to, true)
      for (const n of names) {
        if (skip.has(n)) continue
        const [f, t] = [`${from}/${n}`, `${to}/${n}`]
        if (isDirAt(h, f)) walk(f, t, `${fromText}/${n}`)
        else files.set(t, files.get(f)!)
      }
    }
    walk(a, fullPath(dst), src)
  }
  return takingKeywords(mod('shutil', {
    Error: Err,
    SameFileError: Same,
    copyfile: F('copyfile', (a, kw) => {
      const dstV = kw.dst ?? a[1]
      copyFile(fspath(kw.src ?? a[0]), fspath(dstV))
      return dstV
    }),
    copy: copy('copy'),
    copy2: copy('copy2'),
    copymode: F('copymode', () => NONE),
    copystat: F('copystat', () => NONE),
    move: F('move', (a, kw) => {
      refuse(h, 'shutil.move', kw, ['copy_function'])
      const src = fspath(kw.src ?? a[0])
      const dstV = kw.dst ?? a[1]
      const dst = fspath(dstV)
      let real = dstV
      let realText = dst
      if (isDirAt(h, fullPath(dst))) {
        if (fullPath(src) === fullPath(dst)) {
          renameAt(h, src, dst)
          return dstV
        }
        realText = dst.endsWith('/') ? dst + base(src) : `${dst}/${base(src)}`
        real = S(realText)
        if (existsAt(h, realText)) h.raise(Err, [S(`Destination path '${realText}' already exists`)])
      }
      if (!existsAt(h, src)) noSuchFile(h, src)
      if (fullPath(realText).startsWith(`${fullPath(src)}/`)) h.raise(Err, [S(`Cannot move a directory '${src}' into itself '${realText}'.`)])
      checkParents(h, realText)
      renameAt(h, src, realText)
      return real
    }),
    rmtree: F('rmtree', (a, kw) => {
      refuse(h, 'shutil.rmtree', kw, ['onerror', 'dir_fd'])
      const p = fspath(kw.path ?? a[0])
      const ignore = kw.ignore_errors ?? a[1]
      const full = fullPath(p)
      if (isFileAt(h, full)) return ignore && h.truthy(ignore) ? NONE : notADir(h, p)
      if (!isDirAt(h, full)) return ignore && h.truthy(ignore) ? NONE : noSuchFile(h, p)
      removeTree(h, full)
      return NONE
    }),
    copytree: F('copytree', (a, kw) => {
      refuse(h, 'shutil.copytree', kw, ['symlinks', 'copy_function', 'ignore_dangling_symlinks'])
      const dstV = kw.dst ?? a[1]
      copyTree(fspath(kw.src ?? a[0]), fspath(dstV), !!kw.dirs_exist_ok && h.truthy(kw.dirs_exist_ok), kw.ignore ?? a[3])
      return dstV
    }),
    ignore_patterns: F('ignore_patterns', (pats) => F('_ignore_patterns', (a) => makeSet(h.iterate(a[1]).filter((n) => pats.some((p) => fnmatchOne(fspath(n), fspath(p))))))),
    which: F('which', () => NONE),
    get_terminal_size: F('get_terminal_size', () => ({ k: 'tuple', items: [int(80), int(24)], names: ['columns', 'lines'], tname: 'os.terminal_size' })),
  }), {
    copyfile: ['src', 'dst', 'follow_symlinks'], copy: ['src', 'dst', 'follow_symlinks'], copy2: ['src', 'dst', 'follow_symlinks'], move: ['src', 'dst', 'copy_function'],
    rmtree: ['path', 'ignore_errors', 'onerror', 'dir_fd'], copytree: ['src', 'dst', 'symlinks', 'ignore', 'copy_function', 'ignore_dangling_symlinks', 'dirs_exist_ok'],
  })
}

const makeTempfile = (h: Host): Value => {
  const files = filesOf(h)
  const dirs = dirsOf(h)
  const nameOf = (kw: Kw, a: Value[], at: number) => {
    const opt = (n: string, i: number) => {
      const v = kw[n] ?? a[i]
      return v && v.k !== 'none' ? v : undefined
    }
    const dirV = opt('dir', at + 2)
    const dir = dirV ? fspath(h, dirV) : tempRoot()
    const suffix = opt('suffix', at)
    const prefix = opt('prefix', at + 1)
    const made = `${dir.endsWith('/') ? dir : `${dir}/`}${prefix ? h.str(prefix) : 'tmp'}${randomChars('abcdefghijklmnopqrstuvwxyz0123456789_', 8)}${suffix ? h.str(suffix) : ''}`
    if (!isDirAt(h, fullPath(dir))) noSuchFile(h, made)
    return made
  }
  const tempFile = (a: Value[], kw: Kw, deleting: boolean): Value => {
    const modeV = kw.mode ?? a[0]
    const mode = modeV ? (modeV as { v: string }).v : 'w+b'
    const path = fullPath(nameOf(kw, a, 4))
    files.set(path, '')
    const f = openFile(h, S(path), S(mode), { ...(kw.encoding ?? a[2] ? { encoding: kw.encoding ?? a[2] } : {}), ...(kw.newline ?? a[3] ? { newline: kw.newline ?? a[3] } : {}), ...(kw.errors ? { errors: kw.errors } : {}) })
    if (deleting) f.onClose = () => { files.delete(path) }
    return { k: 'obj', o: f }
  }
  const TemporaryDirectory = libClass('tempfile', 'TemporaryDirectory', {
    __init__: (a, kw) => {
      const self = a[0] as InstValue
      const path = nameOf(kw, a.slice(1), 0)
      dirs.add(fullPath(path))
      self.attrs.set('name', S(path))
      return NONE
    },
    __repr__: (a) => S(`<TemporaryDirectory ${repr((a[0] as InstValue).attrs.get('name')!)}>`),
    __enter__: (a) => (a[0] as InstValue).attrs.get('name')!,
    __exit__: (a) => {
      removeTree(h, fullPath(h.str((a[0] as InstValue).attrs.get('name')!)))
      return NONE
    },
    cleanup: (a) => {
      removeTree(h, fullPath(h.str((a[0] as InstValue).attrs.get('name')!)))
      return NONE
    },
  })
  const module = mod('tempfile', {
    tempdir: NONE,
    gettempdir: F('gettempdir', () => S(tempRoot())),
    gettempprefix: F('gettempprefix', () => S('tmp')),
    mkdtemp: F('mkdtemp', (a, kw) => {
      const path = nameOf(kw, a, 0)
      dirs.add(fullPath(path))
      return S(path)
    }),
    mkstemp: F('mkstemp', (a, kw) => {
      const path = fullPath(nameOf(kw, a, 0))
      files.set(path, '')
      const table = fdsOf(h)
      const fd = table.size + 3
      table.set(fd, path)
      return tuple([int(fd), S(path)])
    }),
    NamedTemporaryFile: F('NamedTemporaryFile', (a, kw) => tempFile(a, kw, !(kw.delete ?? a[7]) || h.truthy(kw.delete ?? a[7]))),
    TemporaryFile: F('TemporaryFile', (a, kw) => tempFile(a, kw, true)),
    TemporaryDirectory,
  }) as Value & { k: 'module' }
  const tempRoot = () => (module.attrs.tempdir.k === 'none' ? TEMP : fspath(h, module.attrs.tempdir))
  const textFileKeywords = ['mode', 'buffering', 'encoding', 'newline', 'suffix', 'prefix', 'dir']
  return takingKeywords(module, {
    mkdtemp: ['suffix', 'prefix', 'dir'], mkstemp: ['suffix', 'prefix', 'dir', 'text'],
    NamedTemporaryFile: [...textFileKeywords, 'delete', 'errors'], TemporaryFile: [...textFileKeywords, 'errors'],
  })
}

// ---------------------------------------------------------------- textwrap

interface WrapOptions {
  width: number
  initialIndent: string
  subsequentIndent: string
  expandTabs: boolean
  tabsize: number
  replaceWhitespace: boolean
  fixSentenceEndings: boolean
  breakLongWords: boolean
  breakOnHyphens: boolean
  dropWhitespace: boolean
  maxLines: number | null
  placeholder: string
}
const WRAP_DEFAULTS: WrapOptions = { width: 70, initialIndent: '', subsequentIndent: '', expandTabs: true, tabsize: 8, replaceWhitespace: true, fixSentenceEndings: false, breakLongWords: true, breakOnHyphens: true, dropWhitespace: true, maxLines: null, placeholder: ' [...]' }

const WS = '[\\t\\n\\x0b\\x0c\\r ]'
const NOWS = '[^\\t\\n\\x0b\\x0c\\r ]'
const WORD_PUNCT = '[\\p{L}\\p{N}_!"\'&.,?]'
const LETTER = '(?:(?!\\p{Nd})[\\p{L}\\p{N}_])'
let wordsep: RegExp | undefined
/** built on first use: an old browser without lookbehind should lose textwrap, not the whole library */
const wordsepRe = () => (wordsep ??= new RegExp(`(${WS}+|(?<=${WORD_PUNCT})-{2,}(?=[\\p{L}\\p{N}_])|${NOWS}+?(?:-(?:(?<=${LETTER}{2}-)|(?<=${LETTER}-${LETTER}-))(?=${LETTER}-?${LETTER})|(?=${WS}|$)|(?<=${WORD_PUNCT})(?=-{2,}[\\p{L}\\p{N}_])))`, 'u'))
const WORDSEP_SIMPLE = new RegExp(`(${WS}+)`)
const expandTabs = (s: string, size: number) => {
  let col = 0
  let out = ''
  for (const c of s) {
    if (c === '\t') {
      const n = size > 0 ? size - (col % size) : 0
      out += ' '.repeat(n)
      col += n
    } else {
      out += c
      col = c === '\n' || c === '\r' ? 0 : col + 1
    }
  }
  return out
}
/** Python's str.strip() == '' for the chunks wrap makes */
const isBlank = (s: string) => s.trim() === ''

const wrapLines = (text0: string, o: WrapOptions, err: (msg: string) => Error): string[] => {
  let text = o.expandTabs ? expandTabs(text0, o.tabsize) : text0
  if (o.replaceWhitespace) text = text.replace(/[\t\n\x0b\x0c\r]/g, ' ') // eslint-disable-line no-control-regex -- string.whitespace
  const chunks = text.split(o.breakOnHyphens ? wordsepRe() : WORDSEP_SIMPLE).filter(Boolean)
  if (o.fixSentenceEndings) {
    for (let i = 0; i < chunks.length - 1; i++) {
      if (chunks[i + 1] === ' ' && /[a-z][.!?]["']?$/.test(chunks[i])) {
        chunks[i + 1] = '  '
        i++
      }
    }
  }
  const lines: string[] = []
  if (o.width <= 0) throw err(`invalid width ${o.width} (must be > 0)`)
  if (o.maxLines !== null) {
    const indent = o.maxLines > 1 ? o.subsequentIndent : o.initialIndent
    if (indent.length + o.placeholder.trimStart().length > o.width) throw err('placeholder too large for max width')
  }
  chunks.reverse()
  while (chunks.length) {
    const line: string[] = []
    let len = 0
    const indent = lines.length ? o.subsequentIndent : o.initialIndent
    const width = o.width - indent.length
    if (o.dropWhitespace && isBlank(chunks[chunks.length - 1]) && lines.length) chunks.pop()
    while (chunks.length) {
      const l = chunks[chunks.length - 1].length
      if (len + l > width) break
      line.push(chunks.pop()!)
      len += l
    }
    if (chunks.length && chunks[chunks.length - 1].length > width) {
      const left = width < 1 ? 1 : width - len
      if (o.breakLongWords) {
        let end = left
        const chunk = chunks[chunks.length - 1]
        if (o.breakOnHyphens && chunk.length > left) {
          const hyphen = chunk.lastIndexOf('-', left - 1)
          if (hyphen > 0 && /[^-]/.test(chunk.slice(0, hyphen))) end = hyphen + 1
        }
        line.push(chunk.slice(0, end))
        chunks[chunks.length - 1] = chunk.slice(end)
      } else if (!line.length) line.push(chunks.pop()!)
      len = line.reduce((n, c) => n + c.length, 0)
    }
    if (o.dropWhitespace && line.length && isBlank(line[line.length - 1])) len -= line.pop()!.length
    if (!line.length) continue
    if (o.maxLines === null || lines.length + 1 < o.maxLines || ((!chunks.length || (o.dropWhitespace && chunks.length === 1 && isBlank(chunks[0]))) && len <= width)) {
      lines.push(indent + line.join(''))
      continue
    }
    let placed = false
    while (line.length) {
      if (!isBlank(line[line.length - 1]) && len + o.placeholder.length <= width) {
        line.push(o.placeholder)
        lines.push(indent + line.join(''))
        placed = true
        break
      }
      len -= line.pop()!.length
    }
    if (!placed) {
      const prev = lines.length ? lines[lines.length - 1].trimEnd() : null
      if (prev !== null && prev.length + o.placeholder.length <= o.width) lines[lines.length - 1] = prev + o.placeholder
      else lines.push(indent + o.placeholder.trimStart())
    }
    break
  }
  return lines
}

const makeTextwrap = (h: Host): Value => {
  const KEYS: Record<string, keyof WrapOptions> = { width: 'width', initial_indent: 'initialIndent', subsequent_indent: 'subsequentIndent', expand_tabs: 'expandTabs', tabsize: 'tabsize', replace_whitespace: 'replaceWhitespace', fix_sentence_endings: 'fixSentenceEndings', break_long_words: 'breakLongWords', break_on_hyphens: 'breakOnHyphens', drop_whitespace: 'dropWhitespace', max_lines: 'maxLines', placeholder: 'placeholder' }
  const optionsFrom = (given: Map<string, Value>): WrapOptions => {
    const o: WrapOptions = { ...WRAP_DEFAULTS }
    for (const [k, v] of given) {
      const key = KEYS[k]
      if (!key) continue
      const d = WRAP_DEFAULTS[key]
      if (key === 'maxLines') o.maxLines = v.k === 'none' ? null : h.toInt(v)
      else if (typeof d === 'number') (o[key] as number) = h.toInt(v)
      else if (typeof d === 'boolean') (o[key] as boolean) = h.truthy(v)
      else (o[key] as string) = h.str(v)
    }
    return o
  }
  const err = (msg: string) => h.err('ValueError', msg)
  const checked = (kw: Kw) => {
    for (const k in kw) if (!KEYS[k]) throw h.err('TypeError', `TextWrapper.__init__() got an unexpected keyword argument '${k}'`)
    if (kw.width && !isNum(kw.width)) throw h.err('TypeError', `'<=' not supported between instances of '${typeName(kw.width)}' and 'int'`)
    return new Map(Object.entries(kw))
  }
  const textArg = (v: Value | undefined): string => {
    if (v?.k !== 'str') throw h.err('AttributeError', `'${typeName(v ?? NONE)}' object has no attribute 'expandtabs'`)
    return v.v
  }
  const withWidth = (a: Value[], kw: Kw) => {
    const rest: Kw = { ...kw, ...(a[1] ? { width: a[1] } : {}) }
    delete rest.text
    return optionsFrom(checked(rest))
  }
  const TextWrapper = libClass('textwrap', 'TextWrapper', {
    __init__: (a, kw) => {
      const self = a[0] as InstValue
      const names = ['width', 'initial_indent', 'subsequent_indent', 'expand_tabs', 'replace_whitespace', 'fix_sentence_endings', 'break_long_words', 'drop_whitespace', 'break_on_hyphens', 'tabsize', 'max_lines', 'placeholder']
      a.slice(1).forEach((v, i) => self.attrs.set(names[i], v))
      checked(kw)
      for (const k in kw) self.attrs.set(k, kw[k])
      for (const k of names) if (!self.attrs.has(k)) {
        const d = WRAP_DEFAULTS[KEYS[k]]
        self.attrs.set(k, d === null ? NONE : typeof d === 'number' ? int(d) : typeof d === 'boolean' ? bool(d) : S(d))
      }
      return NONE
    },
    wrap: (a) => L(wrapLines(textArg(a[1]), optionsFrom((a[0] as InstValue).attrs), err).map(S)),
    fill: (a) => S(wrapLines(textArg(a[1]), optionsFrom((a[0] as InstValue).attrs), err).join('\n')),
  })
  return mod('textwrap', {
    TextWrapper,
    wrap: F('wrap', (a, kw) => L(wrapLines(textArg(kw.text ?? a[0]), withWidth(a, kw), err).map(S))),
    fill: F('fill', (a, kw) => S(wrapLines(textArg(kw.text ?? a[0]), withWidth(a, kw), err).join('\n'))),
    shorten: F('shorten', (a, kw) => {
      if (!a[1] && !kw.width) throw h.err('TypeError', "shorten() missing 1 required positional argument: 'width'")
      if (kw.max_lines) throw h.err('TypeError', "textwrap.TextWrapper() got multiple values for keyword argument 'max_lines'")
      const given = withWidth(a, kw)
      return S(wrapLines(textArg(kw.text ?? a[0]).trim().split(/\s+/).filter(Boolean).join(' '), { ...given, maxLines: 1 }, err).join('\n'))
    }),
    dedent: F('dedent', (a) => {
      if (a[0]?.k !== 'str') throw h.err('TypeError', 'expected string or bytes-like object')
      const text = a[0].v.replace(/^[ \t]+$/gm, '')
      let margin: string | null = null
      for (const m of text.matchAll(/^([ \t]*)[^ \t\n]/gm)) {
        const ind = m[1]
        if (margin === null) margin = ind
        else if (ind.startsWith(margin)) continue
        else if (margin.startsWith(ind)) margin = ind
        else {
          let i = 0
          while (i < margin.length && i < ind.length && margin[i] === ind[i]) i++
          margin = margin.slice(0, i)
        }
      }
      return S(margin ? text.replace(new RegExp(`^${margin}`, 'gm'), '') : text)
    }),
    indent: F('indent', (a, kw) => {
      const prefix = (kw.prefix ?? a[1]) as { v: string }
      const pred = kw.predicate ?? a[2]
      if (a[0]?.k !== 'str') throw h.err('AttributeError', `'${typeName(a[0] ?? NONE)}' object has no attribute 'splitlines'`)
      const lines = a[0].v.match(/[^\n\r\v\f\u001c-\u001e\u0085\u2028\u2029]*(?:\r\n|[\n\r\v\f\u001c-\u001e\u0085\u2028\u2029]|$)/g)?.filter(Boolean) ?? [] // eslint-disable-line no-control-regex -- str.splitlines
      return S(lines.map((line) => ((pred && pred.k !== 'none' ? h.truthy(h.call(pred, [S(line)])) : line.trim() !== '') ? prefix.v + line : line)).join(''))
    }),
  })
}

// ---------------------------------------------------------------- argparse

const SUPPRESS = '==SUPPRESS=='
const isSuppress = (v: Value | undefined | null) => v?.k === 'str' && v.v === SUPPRESS

type ActionKind = 'store' | 'store_const' | 'store_true' | 'store_false' | 'append' | 'append_const' | 'extend' | 'count' | 'help' | 'version' | 'boolean'

/** the parameters each action's constructor takes, beyond option_strings and dest */
const ACTION_PARAMS: Record<ActionKind, string[]> = {
  store: ['nargs', 'const', 'default', 'type', 'choices', 'required', 'help', 'metavar'],
  append: ['nargs', 'const', 'default', 'type', 'choices', 'required', 'help', 'metavar'],
  extend: ['nargs', 'const', 'default', 'type', 'choices', 'required', 'help', 'metavar'],
  store_const: ['const', 'default', 'required', 'help', 'metavar'],
  append_const: ['const', 'default', 'required', 'help', 'metavar'],
  store_true: ['default', 'required', 'help'],
  store_false: ['default', 'required', 'help'],
  count: ['default', 'required', 'help'],
  help: ['default', 'help'],
  version: ['version', 'default', 'help'],
  boolean: ['default', 'type', 'choices', 'required', 'help', 'metavar'],
}
const ACTION_CLASS: Record<ActionKind, string> = { store: '_StoreAction', append: '_AppendAction', extend: '_ExtendAction', store_const: '_StoreConstAction', append_const: '_AppendConstAction', store_true: '_StoreTrueAction', store_false: '_StoreFalseAction', count: '_CountAction', help: '_HelpAction', version: '_VersionAction', boolean: 'BooleanOptionalAction' }

class ArgAction extends Base {
  cls: string
  kind: ActionKind
  optionStrings: string[]
  dest: string
  nargs: number | string | null
  cons: Value
  def: Value
  type: Value
  choices: Value
  required: boolean
  help: Value
  metavar: Value
  version: Value
  constructor(kind: ActionKind, optionStrings: string[], dest: string, nargs: number | string | null, kw: Kw) {
    super()
    this.kind = kind
    this.cls = ACTION_CLASS[kind]
    this.optionStrings = optionStrings
    this.dest = dest
    this.nargs = nargs
    this.cons = kw.const ?? NONE
    this.def = kw.default ?? NONE
    this.type = kw.type ?? NONE
    this.choices = kw.choices ?? NONE
    this.required = false
    this.help = kw.help ?? NONE
    this.metavar = kw.metavar ?? NONE
    this.version = kw.version ?? NONE
  }
  repr() {
    const nargs = this.nargs === null ? 'None' : typeof this.nargs === 'number' ? String(this.nargs) : repr(S(this.nargs))
    return `${this.cls}(option_strings=${repr(L(this.optionStrings.map(S)))}, dest=${repr(S(this.dest))}, nargs=${nargs}, const=${repr(this.cons)}, default=${repr(this.def)}, type=${repr(this.type)}, choices=${repr(this.choices)}, required=${this.required ? 'True' : 'False'}, help=${repr(this.help)}, metavar=${repr(this.metavar)})`
  }
  getAttr(name: string): Value | undefined {
    switch (name) {
      case 'option_strings': return L(this.optionStrings.map(S))
      case 'dest': return S(this.dest)
      case 'nargs': return this.nargs === null ? NONE : typeof this.nargs === 'number' ? int(this.nargs) : S(this.nargs)
      case 'const': return this.cons
      case 'default': return this.def
      case 'type': return this.type
      case 'choices': return this.choices
      case 'required': return bool(this.required)
      case 'help': return this.help
      case 'metavar': return this.metavar
    }
    return undefined
  }
  setAttr(name: string, v: Value): boolean {
    switch (name) {
      case 'default': this.def = v; return true
      case 'help': this.help = v; return true
      case 'metavar': this.metavar = v; return true
      case 'choices': this.choices = v; return true
      case 'type': this.type = v; return true
      case 'const': this.cons = v; return true
      case 'required': this.required = v.k !== 'none' && (v.k !== 'bool' || v.v); return true
    }
    return false
  }
}

class ArgGroup extends Base {
  readonly cls: string = '_ArgumentGroup'
  parser: ArgParser
  title: Value
  description: Value
  actions: ArgAction[] = []
  constructor(parser: ArgParser, title: Value, description: Value) {
    super()
    this.parser = parser
    this.title = title
    this.description = description
  }
  repr() { return `<argparse.${this.cls} object>` }
  add(action: ArgAction) { this.actions.push(action) }
  getAttr(name: string): Value | undefined {
    switch (name) {
      case 'add_argument': return F(name, (a, kw) => ({ k: 'obj', o: this.parser.addArgument(a, kw, this) }))
      case 'add_mutually_exclusive_group': return F(name, (a, kw) => ({ k: 'obj', o: this.parser.addMutex(kw.required ?? a[0], this) }))
      case 'title': return this.title
      case 'description': return this.description
    }
    return undefined
  }
}

class MutexGroup extends ArgGroup {
  readonly cls = '_MutuallyExclusiveGroup'
  requiredGroup: boolean
  container: ArgGroup | null
  constructor(parser: ArgParser, required: boolean, container: ArgGroup | null) {
    super(parser, NONE, NONE)
    this.requiredGroup = required
    this.container = container
  }
}

type OptionTuple = [ArgAction | null, string, string | null]

class ArgParser extends Base {
  readonly cls = 'ArgumentParser'
  h: Host
  mod: { Namespace: ClassValue; ArgumentError: ClassValue; ArgumentTypeError: ClassValue }
  prog: string
  usage: Value
  description: Value
  epilog: Value
  formatter: string
  prefixChars: string
  argumentDefault: Value
  addHelp: boolean
  allowAbbrev: boolean
  exitOnError: boolean
  actions: ArgAction[] = []
  byOption = new Map<string, ArgAction>()
  groups: ArgGroup[] = []
  mutexes: MutexGroup[] = []
  defaults = new Map<string, Value>()
  negativeLike = false
  positionals: ArgGroup
  optionals: ArgGroup
  constructor(h: Host, mods: ArgParser['mod'], a: Value[], kw: Kw) {
    super()
    this.h = h
    this.mod = mods
    const names = ['prog', 'usage', 'description', 'epilog', 'parents', 'formatter_class', 'prefix_chars', 'fromfile_prefix_chars', 'argument_default', 'conflict_handler', 'add_help', 'allow_abbrev', 'exit_on_error']
    const opt: Kw = {}
    a.forEach((v, i) => {
      if (i >= names.length) throw h.err('TypeError', `ArgumentParser.__init__() takes from 1 to 14 positional arguments but ${a.length + 1} were given`)
      opt[names[i]] = v
    })
    for (const k in kw) {
      if (!names.includes(k)) throw h.err('TypeError', `ArgumentParser.__init__() got an unexpected keyword argument '${k}'`)
      opt[k] = kw[k]
    }
    if (opt.parents && h.iterate(opt.parents).length) throw h.err('NotImplementedError', 'ArgumentParser(parents=...) is not available in this sandbox')
    if (opt.fromfile_prefix_chars && opt.fromfile_prefix_chars.k !== 'none') throw h.err('NotImplementedError', 'ArgumentParser(fromfile_prefix_chars=...) is not available in this sandbox')
    if (opt.conflict_handler && h.str(opt.conflict_handler) !== 'error') {
      if (h.str(opt.conflict_handler) !== 'resolve') throw h.err('ValueError', `invalid conflict_resolution value: ${repr(opt.conflict_handler)}`)
      throw h.err('NotImplementedError', "ArgumentParser(conflict_handler='resolve') is not available in this sandbox")
    }
    const argv0 = h.iterate(h.getAttr(h.module('sys'), 'argv'))[0]
    this.prog = opt.prog && opt.prog.k !== 'none' ? h.str(opt.prog) : argv0 ? h.str(argv0).slice(h.str(argv0).lastIndexOf('/') + 1) : ''
    this.usage = opt.usage ?? NONE
    this.description = opt.description ?? NONE
    this.epilog = opt.epilog ?? NONE
    const fc = opt.formatter_class
    this.formatter = fc?.k === 'type' ? fc.name.replace('argparse.', '') : 'HelpFormatter'
    if (fc && (fc.k !== 'type' || !['HelpFormatter', 'RawDescriptionHelpFormatter', 'RawTextHelpFormatter', 'ArgumentDefaultsHelpFormatter', 'MetavarTypeHelpFormatter'].includes(this.formatter))) throw h.err('NotImplementedError', 'ArgumentParser(formatter_class=...) takes only the formatters in argparse in this sandbox')
    this.prefixChars = opt.prefix_chars ? h.str(opt.prefix_chars) : '-'
    this.argumentDefault = opt.argument_default ?? NONE
    this.addHelp = !opt.add_help || h.truthy(opt.add_help)
    this.allowAbbrev = !opt.allow_abbrev || h.truthy(opt.allow_abbrev)
    this.exitOnError = !opt.exit_on_error || h.truthy(opt.exit_on_error)
    this.positionals = this.addGroup(S('positional arguments'), NONE)
    this.optionals = this.addGroup(S('options'), NONE)
    if (this.addHelp) {
      const p = this.prefixChars.includes('-') ? '-' : this.prefixChars[0]
      this.addArgument([S(`${p}h`), S(p + p + 'help')], { action: S('help'), default: S(SUPPRESS), help: S('show this help message and exit') }, null)
    }
  }
  repr() { return `ArgumentParser(prog=${repr(S(this.prog))}, usage=${repr(this.usage)}, description=${repr(this.description)}, formatter_class=<class 'argparse.${this.formatter}'>, conflict_handler='error', add_help=${this.addHelp ? 'True' : 'False'})` }
  addGroup(title: Value, description: Value) {
    const g = new ArgGroup(this, title, description)
    this.groups.push(g)
    return g
  }
  addMutex(required: Value | undefined, container: ArgGroup | null) {
    const g = new MutexGroup(this, !!required && this.h.truthy(required), container)
    this.mutexes.push(g)
    return g
  }
  /** ArgumentError(action, message): the exception that says which argument it is about */
  argError(action: ArgAction | null, message: string): PyError {
    const name = action ? actionName(action) : null
    return this.argErrorText(name, message)
  }
  argErrorText(name: string | null, message: string): PyError {
    try {
      return raiseWith(this.h, this.mod.ArgumentError, [S(name === null ? message : `argument ${name}: ${message}`)], { message: S(message), argument_name: name === null ? NONE : S(name) })
    } catch (e) {
      return e as PyError
    }
  }
  addArgument(args: Value[], kw0: Kw, group: ArgGroup | null): ArgAction {
    const h = this.h
    const kw = { ...kw0 }
    for (const v of args) if (v.k !== 'str') throw h.err('TypeError', `'${typeName(v)}' object is not subscriptable`)
    const texts = args.map((v) => (v as { v: string }).v)
    if (texts.some((t) => t === '')) throw h.err('IndexError', 'string index out of range')
    let optionStrings: string[] = []
    let dest: string
    if (!texts.length || (texts.length === 1 && !this.prefixChars.includes(texts[0][0]))) {
      if (texts.length && kw.dest) throw h.err('ValueError', 'dest supplied twice for positional argument')
      if (!texts.length && !kw.dest) throw h.err('TypeError', "_ActionsContainer._get_positional_kwargs() missing 1 required positional argument: 'dest'")
      if (kw.required) throw h.err('TypeError', "'required' is an invalid argument for positionals")
      dest = texts[0] ?? h.str(kw.dest)
      delete kw.dest
      const nargs = kw.nargs
      const isText = (x: string) => nargs?.k === 'str' && nargs.v === x
      if (!isText('?') && !isText('*')) kw.required = bool(true)
      if (isText('*') && !kw.default) kw.required = bool(true)
    } else {
      for (const t of texts) if (!this.prefixChars.includes(t[0])) throw h.err('ValueError', `invalid option string ${repr(S(t))}: must start with a character ${repr(S(this.prefixChars))}`)
      optionStrings = texts
      const long = texts.filter((t) => t.length > 1 && this.prefixChars.includes(t[1]))
      const given = kw.dest
      delete kw.dest
      if (given && given.k !== 'none') dest = h.str(given)
      else {
        const source = long[0] ?? texts[0]
        let i = 0
        while (i < source.length && this.prefixChars.includes(source[i])) i++
        dest = source.slice(i)
        if (!dest) throw h.err('ValueError', `dest= is required for options like ${repr(S(texts[texts.length - 1]))}`)
        dest = dest.replace(/-/g, '_')
      }
    }
    if (!kw.default) {
      const parserDefault = this.defaults.get(dest)
      if (parserDefault) kw.default = parserDefault
      else if (this.argumentDefault.k !== 'none') kw.default = this.argumentDefault
    }
    const actionV = kw.action
    delete kw.action
    let kind: ActionKind
    if (!actionV || actionV.k === 'none') kind = 'store'
    else if (actionV.k === 'str') {
      if (!(actionV.v in ACTION_PARAMS) || actionV.v === 'boolean') throw h.err('ValueError', `unknown action "${actionV.v}"`)
      kind = actionV.v as ActionKind
    } else if (actionV.k === 'type' && actionV.name === 'argparse.BooleanOptionalAction') kind = 'boolean'
    else if (isCallable(h, actionV)) throw h.err('NotImplementedError', 'add_argument(action=...) with a custom Action class is not available in this sandbox')
    else throw h.err('ValueError', `unknown action "${h.str(actionV)}"`)
    const allowed = ACTION_PARAMS[kind]
    for (const k in kw) {
      if (k === 'required' && (kind === 'help' || kind === 'version')) throw h.err('TypeError', `${ACTION_CLASS[kind]}.__init__() got an unexpected keyword argument 'required'`)
      if (k !== 'required' && !allowed.includes(k)) throw h.err('TypeError', `${ACTION_CLASS[kind]}.__init__() got an unexpected keyword argument '${k}'`)
    }
    if ((kind === 'store_const' || kind === 'append_const') && !kw.const) throw h.err('TypeError', `${ACTION_CLASS[kind]}.__init__() missing 1 required positional argument: 'const'`)
    let nargs: number | string | null = null
    if (kw.nargs && kw.nargs.k !== 'none') nargs = kw.nargs.k === 'str' ? kw.nargs.v : h.toInt(kw.nargs)
    if (kind === 'store' || kind === 'append' || kind === 'extend') {
      if (nargs === 0) throw h.err('ValueError', kind === 'store' ? 'nargs for store actions must be != 0; if you have nothing to store, actions such as store true or store const may be more appropriate' : 'nargs for append actions must be != 0; if arg strings are not supplying the value to append, the append const action may be more appropriate')
      if (kw.const && kw.const.k !== 'none' && nargs !== '?') throw h.err('ValueError', "nargs must be '?' to supply const")
    } else nargs = 0
    if (typeof nargs === 'string' && !['?', '*', '+', 'A...', '...', SUPPRESS].includes(nargs)) throw h.err('ValueError', 'invalid nargs value')
    if (kind === 'store_true') { kw.const = bool(true); kw.default ??= bool(false) }
    if (kind === 'store_false') { kw.const = bool(false); kw.default ??= bool(true) }
    if (kind === 'help' || kind === 'version') kw.default ??= S(SUPPRESS)
    if (kind === 'version') kw.help ??= S("show program's version number and exit")
    if (kind === 'boolean' && kw.help && kw.help.k !== 'none' && kw.default && kw.default.k !== 'none' && !isSuppress(kw.default)) kw.help = S(`${h.str(kw.help)} (default: %(default)s)`)
    const finalDest = kind === 'help' || kind === 'version' ? (kw0.dest ? h.str(kw0.dest) : SUPPRESS) : dest
    if (kind === 'boolean') for (const o of [...optionStrings]) if (o.startsWith('--')) optionStrings.push(`--no-${o.slice(2)}`)
    const action = new ArgAction(kind, optionStrings, finalDest, nargs, kw)
    action.required = !!kw.required && h.truthy(kw.required)
    if (action.type.k !== 'none' && !isCallable(h, action.type)) throw h.err('ValueError', `${repr(action.type)} is not callable`)
    if (action.metavar.k === 'tuple') {
      const n = action.metavar.items.length
      const wants = nargs === null || nargs === '?' || nargs === '*' && n !== 2 ? 1 : nargs === '+' ? 2 : typeof nargs === 'number' ? nargs : nargs === '*' ? 2 : n
      if (n !== wants) throw h.err('ValueError', 'length of metavar tuple does not match nargs')
    }
    return this.register(action, group)
  }
  register(action: ArgAction, group: ArgGroup | null): ArgAction {
    const clash = action.optionStrings.filter((o) => this.byOption.has(o))
    if (clash.length) throw this.argError(action, `conflicting option string${clash.length > 1 ? 's' : ''}: ${clash.join(', ')}`)
    if (group instanceof MutexGroup && action.required) throw this.h.err('ValueError', 'mutually exclusive arguments must be optional')
    this.actions.push(action)
    for (const o of action.optionStrings) {
      this.byOption.set(o, action)
      if (/^-\d+$|^-\d*\.\d+$/.test(o)) this.negativeLike = true
    }
    // a mutually exclusive group's arguments show under their container's heading
    const shown = group instanceof MutexGroup ? group.container : group
    const home = shown ?? (action.optionStrings.length ? this.optionals : this.positionals)
    home.add(action)
    if (group instanceof MutexGroup) group.add(action)
    return action
  }

  // ---- the parse, as argparse does it
  parseOptional(arg: string): OptionTuple | null {
    if (!arg || !this.prefixChars.includes(arg[0])) return null
    const direct = this.byOption.get(arg)
    if (direct) return [direct, arg, null]
    if (arg.length === 1) return null
    if (arg.includes('=')) {
      const at = arg.indexOf('=')
      const found = this.byOption.get(arg.slice(0, at))
      if (found) return [found, arg.slice(0, at), arg.slice(at + 1)]
    }
    const tuples = this.optionTuples(arg)
    if (tuples.length > 1) this.error(`ambiguous option: ${arg} could match ${tuples.map((t) => t[1]).join(', ')}`)
    if (tuples.length === 1) return tuples[0]
    if (/^-\d+$|^-\d*\.\d+$/.test(arg) && !this.negativeLike) return null
    if (arg.includes(' ')) return null
    return [null, arg, null]
  }
  optionTuples(arg: string): OptionTuple[] {
    const out: OptionTuple[] = []
    const chars = this.prefixChars
    if (chars.includes(arg[0]) && chars.includes(arg[1])) {
      if (this.allowAbbrev) {
        const at = arg.indexOf('=')
        const [prefix, explicit] = at >= 0 ? [arg.slice(0, at), arg.slice(at + 1)] : [arg, null]
        for (const [o, action] of this.byOption) if (o.startsWith(prefix)) out.push([action, o, explicit])
      }
    } else if (chars.includes(arg[0])) {
      const short = arg.slice(0, 2)
      const rest = arg.slice(2)
      for (const [o, action] of this.byOption) {
        if (o === short) out.push([action, o, rest])
        else if (o.startsWith(arg)) out.push([action, o, null])
      }
    }
    return out
  }
  nargsPattern(action: ArgAction): string {
    const n = action.nargs
    let p: string
    if (n === null) p = '(-*A-*)'
    else if (n === '?') p = '(-*A?-*)'
    else if (n === '*') p = '(-*[A-]*)'
    else if (n === '+') p = '(-*A[A-]*)'
    else if (n === '...') p = '([-AO]*)'
    else if (n === 'A...') p = '(-*A[-AO]*)'
    else if (n === SUPPRESS) p = '(-*-*)'
    else p = `(-*${Array.from({ length: n as number }, () => 'A').join('-*')}-*)`
    if (action.optionStrings.length) p = p.replace(/-\*/g, '').replace(/-/g, '')
    return p
  }
  matchArgument(action: ArgAction, pattern: string): number {
    const m = new RegExp(`^${this.nargsPattern(action)}`).exec(pattern)
    if (!m) {
      const n = action.nargs
      const msg = n === null ? 'expected one argument' : n === '?' ? 'expected at most one argument' : n === '+' ? 'expected at least one argument' : `expected ${n} argument${n === 1 ? '' : 's'}`
      throw this.argError(action, msg)
    }
    return m[1].length
  }
  matchPartial(actions: ArgAction[], pattern: string): number[] {
    for (let i = actions.length; i > 0; i--) {
      const m = new RegExp(`^${actions.slice(0, i).map((a) => this.nargsPattern(a)).join('')}`).exec(pattern)
      if (m) return m.slice(1).map((s) => s.length)
    }
    return []
  }
  value(action: ArgAction, text: string): Value {
    const h = this.h
    if (action.type.k === 'none') return S(text)
    if (!isCallable(h, action.type)) throw this.argError(action, `${repr(action.type)} is not callable`)
    try {
      return h.call(action.type, [S(text)])
    } catch (e) {
      if (caught(h, e, 'ArgumentTypeError')) throw this.argError(action, h.str((e as PyError).exc as Value))
      if (caught(h, e, 'TypeError', 'ValueError')) {
        const typeName_ = h.getAttrOr(action.type, '__name__') ?? S(repr(action.type))
        throw this.argError(action, `invalid ${h.str(typeName_)} value: ${repr(S(text))}`)
      }
      throw e
    }
  }
  checkValue(action: ArgAction, v: Value) {
    const h = this.h
    if (action.choices.k === 'none' || h.truthy(h.compare('in', v, action.choices))) return
    throw this.argError(action, `invalid choice: ${repr(v)} (choose from ${h.iterate(action.choices).map(repr).join(', ')})`)
  }
  values(action: ArgAction, strings0: string[]): Value {
    const strings = [...strings0]
    if (action.nargs !== 'A...' && action.nargs !== '...') {
      const at = strings.indexOf('--')
      if (at >= 0) strings.splice(at, 1)
    }
    if (!strings.length && action.nargs === '?') {
      let value = action.optionStrings.length ? action.cons : action.def
      if (value.k === 'str') {
        value = this.value(action, value.v)
        this.checkValue(action, value)
      }
      return value
    }
    if (!strings.length && action.nargs === '*' && !action.optionStrings.length) {
      const value = action.def.k !== 'none' ? action.def : L([])
      this.checkValue(action, value)
      return value
    }
    if (strings.length === 1 && (action.nargs === null || action.nargs === '?')) {
      const value = this.value(action, strings[0])
      this.checkValue(action, value)
      return value
    }
    if (action.nargs === '...') return L(strings.map((s) => this.value(action, s)))
    if (action.nargs === 'A...') {
      const out = strings.map((s) => this.value(action, s))
      this.checkValue(action, out[0])
      return L(out)
    }
    if (action.nargs === SUPPRESS) return S(SUPPRESS)
    const out = strings.map((s) => this.value(action, s))
    for (const v of out) this.checkValue(action, v)
    return L(out)
  }
  apply(action: ArgAction, ns: InstValue, values: Value, option: string | null) {
    const copyOf = (v: Value | undefined): Value[] => (!v || v.k === 'none' ? [] : v.k === 'list' ? [...v.items] : [...this.h.iterate(v)])
    switch (action.kind) {
      case 'store': ns.attrs.set(action.dest, values); return
      case 'store_const': case 'store_true': case 'store_false': ns.attrs.set(action.dest, action.cons); return
      case 'append': ns.attrs.set(action.dest, L([...copyOf(ns.attrs.get(action.dest)), values])); return
      case 'append_const': ns.attrs.set(action.dest, L([...copyOf(ns.attrs.get(action.dest)), action.cons])); return
      case 'extend': ns.attrs.set(action.dest, L([...copyOf(ns.attrs.get(action.dest)), ...this.h.iterate(values)])); return
      case 'count': {
        const c = ns.attrs.get(action.dest)
        ns.attrs.set(action.dest, !c || c.k === 'none' ? int(1) : this.h.binop('+', c, int(1)))
        return
      }
      case 'boolean':
        if (option && action.optionStrings.includes(option)) ns.attrs.set(action.dest, bool(!option.startsWith('--no-')))
        return
      case 'help':
        this.write(this.formatHelp(), this.stream('stdout'))
        this.quit(0, null)
        return
      case 'version':
        this.write(new ArgFormatter(this).version(action.version), this.stream('stdout'))
        this.quit(0, null)
        return
    }
  }
  parseKnown(argsV: Value | undefined, nsV: Value | undefined): [Value, Value[]] {
    const h = this.h
    const argv = !argsV || argsV.k === 'none' ? h.iterate(h.getAttr(h.module('sys'), 'argv')).slice(1) : [...h.iterate(argsV)]
    const strings = argv.map((v) => {
      if (v.k !== 'str') throw h.err('TypeError', `'${typeName(v)}' object is not subscriptable`)
      return v.v
    })
    const made = !nsV || nsV.k === 'none' ? h.call(this.mod.Namespace, []) : nsV
    if (made.k !== 'inst') throw h.err('NotImplementedError', 'parse_args(namespace=...) takes an object of a Python class in this sandbox')
    const ns = made
    for (const action of this.actions) {
      if (action.dest !== SUPPRESS && !ns.attrs.has(action.dest) && !isSuppress(action.def)) ns.attrs.set(action.dest, action.def)
    }
    for (const [dest, v] of this.defaults) if (!ns.attrs.has(dest)) ns.attrs.set(dest, v)
    try {
      return [ns, this.parseInner(strings, ns).map(S)]
    } catch (e) {
      if (this.exitOnError && e instanceof PyError && (e.exc as InstValue | undefined)?.cls === this.mod.ArgumentError) return this.error(h.str(e.exc as Value))
      throw e
    }
  }
  parseInner(strings: string[], ns: InstValue): string[] {
    const conflicts = new Map<ArgAction, ArgAction[]>()
    for (const g of this.mutexes) g.actions.forEach((a, i) => {
      conflicts.set(a, [...(conflicts.get(a) ?? []), ...g.actions.slice(0, i), ...g.actions.slice(i + 1)])
    })
    const optionAt = new Map<number, OptionTuple>()
    let pattern = ''
    let dashes = false
    strings.forEach((s, i) => {
      if (dashes) { pattern += 'A'; return }
      if (s === '--') { pattern += '-'; dashes = true; return }
      const t = this.parseOptional(s)
      if (t) optionAt.set(i, t)
      pattern += t ? 'O' : 'A'
    })
    const seen = new Set<ArgAction>()
    const seenReal = new Set<ArgAction>()
    const take = (action: ArgAction, args: string[], option: string | null) => {
      seen.add(action)
      const values = this.values(action, args)
      if (values !== action.def) {
        seenReal.add(action)
        for (const c of conflicts.get(action) ?? []) if (seenReal.has(c)) throw this.argError(action, `not allowed with argument ${actionName(c)}`)
      }
      if (!isSuppress(values)) this.apply(action, ns, values, option)
    }
    const extras: string[] = []
    const consumeOptional = (start: number): number => {
      let [action, option, explicit] = optionAt.get(start)!
      const queue: [ArgAction, string[], string][] = []
      let stop: number
      for (;;) {
        if (!action) {
          extras.push(strings[start])
          return start + 1
        }
        if (explicit !== null) {
          const count = this.matchArgument(action, 'A')
          if (count === 0 && !this.prefixChars.includes(option[1]) && explicit !== '') {
            queue.push([action, [], option])
            option = option[0] + explicit[0]
            const rest = explicit.slice(1) || null
            const next = this.byOption.get(option)
            if (!next) throw this.argError(action, `ignored explicit argument ${repr(S(explicit))}`)
            action = next
            explicit = rest
          } else if (count === 1) {
            stop = start + 1
            queue.push([action, [explicit], option])
            break
          } else throw this.argError(action, `ignored explicit argument ${repr(S(explicit))}`)
        } else {
          const from = start + 1
          const count = this.matchArgument(action, pattern.slice(from))
          stop = from + count
          queue.push([action, strings.slice(from, stop), option])
          break
        }
      }
      for (const [a, args, o] of queue) take(a, args, o)
      return stop
    }
    const positionals = this.actions.filter((a) => !a.optionStrings.length)
    const consumePositionals = (start0: number): number => {
      let start = start0
      const counts = this.matchPartial(positionals, pattern.slice(start))
      counts.forEach((count, i) => {
        const args = strings.slice(start, start + count)
        start += count
        take(positionals[i], args, null)
      })
      positionals.splice(0, counts.length)
      return start
    }
    let start = 0
    const lastOption = optionAt.size ? Math.max(...optionAt.keys()) : -1
    while (start <= lastOption) {
      const next = Math.min(...[...optionAt.keys()].filter((i) => i >= start))
      if (start !== next) {
        const end = consumePositionals(start)
        if (end > start) {
          start = end
          continue
        }
        start = end
      }
      if (!optionAt.has(start)) {
        extras.push(...strings.slice(start, next))
        start = next
      }
      start = consumeOptional(start)
    }
    const stop = consumePositionals(start)
    extras.push(...strings.slice(stop))
    const missing: string[] = []
    for (const action of this.actions) {
      if (seen.has(action)) continue
      if (action.required) missing.push(actionName(action)!)
      else if (action.def.k === 'str' && ns.attrs.get(action.dest) === action.def) ns.attrs.set(action.dest, this.value(action, action.def.v))
    }
    if (missing.length) this.error(`the following arguments are required: ${missing.join(', ')}`)
    for (const g of this.mutexes) {
      if (g.requiredGroup && !g.actions.some((a) => seenReal.has(a))) {
        this.error(`one of the arguments ${g.actions.filter((a) => !isSuppress(a.help)).map((a) => actionName(a)).join(' ')} is required`)
      }
    }
    return extras
  }

  // ---- messages
  stream(which: 'stdout' | 'stderr'): Value { return this.h.getAttr(this.h.module('sys'), which) }
  write(message: string, file: Value) { if (message) this.h.call(this.h.getAttr(file, 'write'), [S(message)]) }
  formatUsage() { return new ArgFormatter(this).usage(this.usage, this.actions, this.mutexes) }
  formatHelp() { return new ArgFormatter(this).help() }
  quit(status: number, message: string | null, said?: string): never {
    if (message) this.write(message, this.stream('stderr'))
    const shown = said ?? (status && message ? message.trimEnd() : undefined)
    return raiseWith(this.h, this.h.excClass('SystemExit'), [int(status)], {}, shown ? { said: shown } : {})
  }
  error(message: string): never {
    const usage = this.formatUsage()
    this.write(usage, this.stream('stderr'))
    const text = `${this.prog}: error: ${message}\n`
    return this.quit(2, text, (usage + text).trimEnd())
  }
  getAttr(name: string, h: Host): Value | undefined {
    const fileOf = (v: Value | undefined) => (v && v.k !== 'none' ? v : this.stream('stdout'))
    switch (name) {
      case 'add_argument': return F(name, (a, kw) => ({ k: 'obj', o: this.addArgument(a, kw, null) }))
      case 'add_argument_group': return F(name, (a, kw) => {
        refuse(h, name, kw, ['prefix_chars', 'argument_default', 'conflict_handler'])
        return { k: 'obj', o: this.addGroup(kw.title ?? a[0] ?? NONE, kw.description ?? a[1] ?? NONE) }
      })
      case 'add_mutually_exclusive_group': return F(name, (a, kw) => ({ k: 'obj', o: this.addMutex(kw.required ?? a[0], null) }))
      case 'add_subparsers': return F(name, () => { throw h.err('NotImplementedError', 'ArgumentParser.add_subparsers is not available in this sandbox') })
      case 'parse_args': return F(name, (a, kw) => {
        const [ns, extras] = this.parseKnown(kw.args ?? a[0], kw.namespace ?? a[1])
        if (extras.length) this.error(`unrecognized arguments: ${extras.map((x) => (x as { v: string }).v).join(' ')}`)
        return ns
      })
      case 'parse_known_args': return F(name, (a, kw) => {
        const [ns, extras] = this.parseKnown(kw.args ?? a[0], kw.namespace ?? a[1])
        return tuple([ns, L(extras)])
      })
      case 'format_help': return F(name, () => S(this.formatHelp()))
      case 'format_usage': return F(name, () => S(this.formatUsage()))
      case 'print_help': return F(name, (a, kw) => { this.write(this.formatHelp(), fileOf(kw.file ?? a[0])); return NONE })
      case 'print_usage': return F(name, (a, kw) => { this.write(this.formatUsage(), fileOf(kw.file ?? a[0])); return NONE })
      case 'error': return F(name, (a, kw) => this.error(h.str(kw.message ?? a[0])))
      case 'exit': return F(name, (a, kw) => {
        const status = kw.status ?? a[0]
        const message = kw.message ?? a[1]
        return this.quit(status ? h.toInt(status) : 0, message && message.k !== 'none' ? h.str(message) : null)
      })
      case 'set_defaults': return F(name, (_a, kw) => {
        for (const k in kw) {
          this.defaults.set(k, kw[k])
          for (const action of this.actions) if (action.dest === k) action.def = kw[k]
        }
        return NONE
      })
      case 'get_default': return F(name, (a) => {
        const dest = h.str(a[0])
        const hit = this.actions.find((x) => x.dest === dest && x.def.k !== 'none')
        return hit ? hit.def : this.defaults.get(dest) ?? NONE
      })
      case 'prog': return S(this.prog)
      case 'usage': return this.usage
      case 'description': return this.description
      case 'epilog': return this.epilog
    }
    return undefined
  }
  setAttr(name: string, v: Value): boolean {
    switch (name) {
      case 'prog': this.prog = this.h.str(v); return true
      case 'usage': this.usage = v; return true
      case 'description': this.description = v; return true
      case 'epilog': this.epilog = v; return true
    }
    return false
  }
}

const actionName = (a: ArgAction): string | null => {
  if (a.optionStrings.length) return a.optionStrings.join('/')
  if (a.metavar.k !== 'none' && !isSuppress(a.metavar)) return str(a.metavar)
  if (a.dest !== SUPPRESS) return a.dest
  return null
}

/** argparse.HelpFormatter, with its width on a terminal that has no COLUMNS */
class ArgFormatter {
  p: ArgParser
  h: Host
  width = 78
  maxHelp = 24
  indent = 0
  maxAction = 0
  constructor(p: ArgParser) {
    this.p = p
    this.h = p.h
  }
  percent(fmt: string, params: PyDict): string { return (this.h.binop('%', S(fmt), { k: 'dict', d: params }) as { v: string }).v }
  progOnly() { return PyDict.from([[S('prog'), S(this.p.prog)]]) }
  metavar(a: ArgAction, def: string): (n: number) => string[] {
    let result: Value
    if (a.metavar.k !== 'none') result = a.metavar
    else if (a.choices.k !== 'none') result = S(`{${this.h.iterate(a.choices).map((c) => this.h.str(c)).join(',')}}`)
    else result = S(def)
    return (n) => (result.k === 'tuple' ? result.items.map((x) => this.h.str(x)) : Array.from({ length: n }, () => this.h.str(result)))
  }
  args(a: ArgAction, def: string): string {
    const get = this.metavar(a, def)
    const n = a.nargs
    if (n === null) return get(1)[0]
    if (n === '?') return `[${get(1)[0]}]`
    if (n === '*') {
      const m = get(1)
      return m.length === 2 ? `[${m[0]} [${m[1]} ...]]` : `[${m[0]} ...]`
    }
    if (n === '+') {
      const m = get(2)
      return `${m[0]} [${m[1]} ...]`
    }
    if (n === '...') return '...'
    if (n === 'A...') return `${get(1)[0]} ...`
    if (n === SUPPRESS) return ''
    return get(n as number).slice(0, n as number).join(' ')
  }
  defaultOptional(a: ArgAction) { return this.p.formatter === 'MetavarTypeHelpFormatter' && a.type.k !== 'none' ? this.h.str(this.h.getAttr(a.type, '__name__')) : a.dest.toUpperCase() }
  defaultPositional(a: ArgAction) { return this.p.formatter === 'MetavarTypeHelpFormatter' && a.type.k !== 'none' ? this.h.str(this.h.getAttr(a.type, '__name__')) : a.dest }
  actionsUsage(actions: ArgAction[], groups: MutexGroup[]): string {
    const inGroup = new Set<ArgAction>()
    const inserts = new Map<number, string>()
    for (const g of groups) {
      if (!g.actions.length) throw this.h.err('ValueError', `empty group ${g.repr()}`)
      const start = actions.indexOf(g.actions[0])
      if (start < 0) continue
      const end = start + g.actions.length
      if (!g.actions.every((a, i) => actions[start + i] === a)) continue
      let hidden = 0
      for (const a of g.actions) {
        inGroup.add(a)
        if (isSuppress(a.help)) hidden++
      }
      const shown = g.actions.length - hidden
      const mark = (at: number, c: string) => inserts.set(at, inserts.has(at) ? `${inserts.get(at)}${c === ']' || c === ')' ? '' : ' '}${c}` : c)
      if (!g.requiredGroup) {
        mark(start, '[')
        mark(end, ']')
      } else if (shown > 1) {
        mark(start, '(')
        mark(end, ')')
      }
      for (let i = start + 1; i < end; i++) inserts.set(i, '|')
    }
    const parts: (string | null)[] = []
    actions.forEach((a, i) => {
      if (isSuppress(a.help)) {
        parts.push(null)
        if (inserts.get(i) === '|') inserts.delete(i)
        else if (inserts.get(i + 1) === '|') inserts.delete(i + 1)
      } else if (!a.optionStrings.length) {
        let part = this.args(a, this.defaultPositional(a))
        if (inGroup.has(a) && part.startsWith('[') && part.endsWith(']')) part = part.slice(1, -1)
        parts.push(part)
      } else {
        const option = a.optionStrings[0]
        let part = a.nargs === 0 ? (a.kind === 'boolean' ? a.optionStrings.join(' | ') : option) : `${option} ${this.args(a, this.defaultOptional(a))}`
        if (!a.required && !inGroup.has(a)) part = `[${part}]`
        parts.push(part)
      }
    })
    for (const i of [...inserts.keys()].sort((x, y) => y - x)) parts.splice(i, 0, inserts.get(i)!)
    return parts.filter((x): x is string => x !== null).join(' ')
      .replace(/([[(]) /g, '$1')
      .replace(/ ([\])])/g, '$1')
      .replace(/[[(] *[\])]/g, '')
      .trim()
  }
  usage(usageV: Value, actions: ArgAction[], groups: MutexGroup[]): string {
    return isSuppress(usageV) ? '' : `${this.usageText(usageV, actions, groups).replace(/\n\n\n+/g, '\n\n').replace(/^\n+|\n+$/g, '')}\n`
  }
  usageText(usageV: Value, actions: ArgAction[], groups: MutexGroup[]): string {
    const prefix = 'usage: '
    let usage: string
    if (usageV.k !== 'none') usage = this.percent(this.h.str(usageV), this.progOnly())
    else if (!actions.length) usage = this.p.prog
    else {
      const prog = this.p.prog
      const optionals = actions.filter((a) => a.optionStrings.length)
      const positionals = actions.filter((a) => !a.optionStrings.length)
      usage = [prog, this.actionsUsage([...optionals, ...positionals], groups)].filter(Boolean).join(' ')
      const textWidth = this.width - this.indent
      if (prefix.length + usage.length > textWidth) {
        const part = /\(.*?\)+(?=\s|$)|\[.*?\]+(?=\s|$)|\S+/g
        const optParts = this.actionsUsage(optionals, groups).match(part) ?? []
        const posParts = this.actionsUsage(positionals, groups).match(part) ?? []
        const getLines = (parts: string[], indent: string, pre?: string) => {
          const lines: string[] = []
          let line: string[] = []
          let len = (pre !== undefined ? pre.length : indent.length) - 1
          for (const p of parts) {
            if (len + 1 + p.length > textWidth && line.length) {
              lines.push(indent + line.join(' '))
              line = []
              len = indent.length - 1
            }
            line.push(p)
            len += p.length + 1
          }
          if (line.length) lines.push(indent + line.join(' '))
          if (pre !== undefined) lines[0] = lines[0].slice(indent.length)
          return lines
        }
        let lines: string[]
        if (prefix.length + prog.length <= 0.75 * textWidth) {
          const indent = ' '.repeat(prefix.length + prog.length + 1)
          if (optParts.length) lines = [...getLines([prog, ...optParts], indent, prefix), ...getLines(posParts, indent)]
          else if (posParts.length) lines = getLines([prog, ...posParts], indent, prefix)
          else lines = [prog]
        } else {
          const indent = ' '.repeat(prefix.length)
          lines = getLines([...optParts, ...posParts], indent)
          if (lines.length > 1) lines = [...getLines(optParts, indent), ...getLines(posParts, indent)]
          lines = [prog, ...lines]
        }
        usage = lines.join('\n')
      }
    }
    return `${prefix}${usage}\n\n`
  }
  invocation(a: ArgAction): string {
    if (!a.optionStrings.length) return this.metavar(a, this.defaultPositional(a))(1)[0]
    if (a.nargs === 0) return a.optionStrings.join(', ')
    const args = this.args(a, this.defaultOptional(a))
    return a.optionStrings.map((o) => `${o} ${args}`).join(', ')
  }
  expandHelp(a: ArgAction): string {
    const h = this.h
    const nargs = a.nargs === null ? NONE : typeof a.nargs === 'number' ? int(a.nargs) : S(a.nargs)
    const raw: [string, Value][] = [['option_strings', L(a.optionStrings.map(S))], ['dest', S(a.dest)], ['nargs', nargs], ['const', a.cons], ['default', a.def], ['type', a.type], ['choices', a.choices], ['required', bool(a.required)], ['help', a.help], ['metavar', a.metavar], ['prog', S(this.p.prog)]]
    const params: [Value, Value][] = raw.filter(([, v]) => !isSuppress(v)).map(([n, v]) => {
      if (n === 'choices' && v.k !== 'none') return [S(n), S(h.iterate(v).map((c) => h.str(c)).join(', '))]
      return [S(n), h.getAttrOr(v, '__name__') ?? v]
    })
    let text = h.str(a.help)
    if (this.p.formatter === 'ArgumentDefaultsHelpFormatter' && !text.includes('%(default)') && !isSuppress(a.def) && (a.optionStrings.length || a.nargs === '?' || a.nargs === '*')) text += ' (default: %(default)s)'
    return this.percent(text, PyDict.from(params))
  }
  splitHelp(text: string, width: number): string[] {
    if (this.p.formatter === 'RawTextHelpFormatter') return text.split(/\r\n|\r|\n/)
    return wrapLines(text.trim().split(/\s+/).filter(Boolean).join(' '), { ...WRAP_DEFAULTS, width }, (m) => this.h.err('ValueError', m))
  }
  textBlock(text: Value): string {
    if (text.k === 'none' || isSuppress(text)) return ''
    let t = this.h.str(text)
    if (t.includes('%(prog)')) t = this.percent(t, this.progOnly())
    const width = Math.max(this.width - this.indent, 11)
    const pad = ' '.repeat(this.indent)
    if (this.p.formatter === 'RawDescriptionHelpFormatter' || this.p.formatter === 'RawTextHelpFormatter') return `${(t.match(/[^\n]*\n|[^\n]+$/g) ?? []).map((l) => pad + l).join('')}\n\n`
    return `${wrapLines(t.trim().split(/\s+/).filter(Boolean).join(' '), { ...WRAP_DEFAULTS, width, initialIndent: pad, subsequentIndent: pad }, (m) => this.h.err('ValueError', m)).join('\n')}\n\n`
  }
  actionHelp(a: ArgAction): string {
    const position = Math.min(this.maxAction + 2, Math.min(this.maxHelp, Math.max(this.width - 20, 4)))
    const helpWidth = Math.max(this.width - position, 11)
    const actionWidth = position - this.indent - 2
    const pad = ' '.repeat(this.indent)
    const label = this.invocation(a)
    const hasHelp = a.help.k !== 'none' && this.h.truthy(a.help)
    let header: string
    let first = 0
    if (!hasHelp) header = `${pad}${label}\n`
    else if (label.length <= actionWidth) header = `${pad}${label.padEnd(actionWidth)}  `
    else {
      header = `${pad}${label}\n`
      first = position
    }
    const parts = [header]
    if (hasHelp && this.h.str(a.help).trim()) {
      const text = this.expandHelp(a)
      if (text) {
        const lines = this.splitHelp(text, helpWidth)
        parts.push(`${' '.repeat(first)}${lines[0]}\n`)
        for (const line of lines.slice(1)) parts.push(`${' '.repeat(position)}${line}\n`)
      }
    } else if (!header.endsWith('\n')) parts.push('\n')
    return parts.join('')
  }
  section(g: ArgGroup): string {
    // the widest entry decides where help text starts, across the whole parser
    this.indent += 2
    const items = [this.textBlock(g.description), ...g.actions.filter((a) => !isSuppress(a.help)).map((a) => this.actionHelp(a))].join('')
    this.indent -= 2
    if (!items) return ''
    const heading = !isSuppress(g.title) && g.title.k !== 'none' ? `${this.h.str(g.title)}:\n` : ''
    return `\n${heading}${items}\n`
  }
  help(): string {
    const p = this.p
    for (const g of p.groups) for (const a of g.actions) if (!isSuppress(a.help)) this.maxAction = Math.max(this.maxAction, this.invocation(a).length + 2)
    const body = (isSuppress(p.usage) ? '' : this.usageText(p.usage, p.actions, p.mutexes)) + this.textBlock(p.description) + p.groups.map((g) => this.section(g)).join('') + this.textBlock(p.epilog)
    return body ? `${body.replace(/\n\n\n+/g, '\n\n').replace(/^\n+|\n+$/g, '')}\n` : ''
  }
  version(v: Value): string {
    const text = this.textBlock(v)
    return text ? `${text.replace(/\n\n\n+/g, '\n\n').replace(/^\n+|\n+$/g, '')}\n` : ''
  }
}

const makeArgparse = (h: Host): Value => {
  const Namespace = libClass('argparse', 'Namespace', {
    __init__: (a, kw) => {
      for (const k in kw) (a[0] as InstValue).attrs.set(k, kw[k])
      return NONE
    },
    __repr__: (a) => {
      const self = a[0] as InstValue
      const plain: string[] = []
      const star: [Value, Value][] = []
      for (const [n, v] of self.attrs) {
        if (/^[\p{L}_][\p{L}\p{N}_]*$/u.test(n)) plain.push(`${n}=${repr(v)}`)
        else star.push([S(n), v])
      }
      if (star.length) plain.push(`**${repr({ k: 'dict', d: PyDict.from(star) })}`)
      return S(`${self.cls.name}(${plain.join(', ')})`)
    },
    __eq__: (a) => {
      const [x, y] = a
      if (x.k !== 'inst' || y.k !== 'inst' || !isNamespace(y)) return NOT_IMPLEMENTED
      return bool(x.attrs.size === y.attrs.size && [...x.attrs].every(([n, v]) => y.attrs.has(n) && h.truthy(h.compare('==', v, y.attrs.get(n)!))))
    },
    __contains__: (a) => bool((a[0] as InstValue).attrs.has(h.str(a[1]))),
    __hash__: (a) => { throw h.err('TypeError', `unhashable type: '${(a[0] as InstValue).cls.name}'`) },
  })
  const isNamespace = (v: InstValue) => {
    const walk = (c: ClassValue): boolean => c === Namespace || c.bases.some((b) => b.k === 'class' && walk(b))
    return walk(v.cls)
  }
  const ArgumentError = excIn('argparse', 'ArgumentError', h.excClass('Exception'))
  const ArgumentTypeError = excIn('argparse', 'ArgumentTypeError', h.excClass('Exception'))
  const mods = { Namespace, ArgumentError, ArgumentTypeError }
  const formatter = (name: string): Value => ({ k: 'type', name: `argparse.${name}`, call: () => { throw h.err('NotImplementedError', `argparse.${name}() cannot be called in this sandbox: pass the class as formatter_class`) } })
  return mod('argparse', {
    ArgumentParser: { k: 'type', name: 'argparse.ArgumentParser', call: (a, kw) => ({ k: 'obj', o: new ArgParser(h, mods, a, kw) }) },
    Namespace,
    ArgumentError,
    ArgumentTypeError,
    SUPPRESS: S(SUPPRESS),
    OPTIONAL: S('?'),
    ZERO_OR_MORE: S('*'),
    ONE_OR_MORE: S('+'),
    REMAINDER: S('...'),
    PARSER: S('A...'),
    HelpFormatter: formatter('HelpFormatter'),
    RawDescriptionHelpFormatter: formatter('RawDescriptionHelpFormatter'),
    RawTextHelpFormatter: formatter('RawTextHelpFormatter'),
    ArgumentDefaultsHelpFormatter: formatter('ArgumentDefaultsHelpFormatter'),
    MetavarTypeHelpFormatter: formatter('MetavarTypeHelpFormatter'),
    BooleanOptionalAction: formatter('BooleanOptionalAction'),
  })
}

// ---------------------------------------------------------------- types

/** types.MappingProxyType: a read-only view of a dict */
class MappingProxy extends Base {
  readonly cls = 'mappingproxy'
  d: Value & { k: 'dict' }
  constructor(d: Value & { k: 'dict' }) {
    super()
    this.d = d
  }
  repr() { return `mappingproxy(${repr(this.d)})` }
  str() { return repr(this.d) }
  iter() { return this.d.d.keys() }
  len() { return this.d.d.size }
  contains(v: Value) { return this.d.d.has(v) }
  hashKey(): string { throw new PyError('TypeError', "unhashable type: 'mappingproxy'") }
  getItem(idx: Value, h: Host): Value {
    const v = this.d.d.get(idx)
    if (!v) throw h.err('KeyError', repr(idx))
    return v
  }
  delItem(_idx: Value, h: Host) { throw h.err('TypeError', "'mappingproxy' object does not support item deletion") }
  binop(op: string, other: Value, _r: boolean, h: Host): Value | undefined {
    if (op !== '==' && op !== '!=') return undefined
    const o = other.k === 'obj' && other.o instanceof MappingProxy ? other.o.d : other
    return h.compare(op, this.d, o)
  }
  getAttr(name: string, h: Host): Value | undefined {
    if (name === '__getitem__') return F(name, (a) => this.getItem(a[0], h))
    return ['keys', 'values', 'items', 'get', 'copy'].includes(name) ? h.getAttr(this.d, name) : undefined
  }
}

const makeTypes = (h: Host): Value => {
  const SimpleNamespace = libClass('types', 'SimpleNamespace', {
    __init__: (a, kw) => {
      if (a.length > 1) throw h.err('TypeError', 'no positional arguments expected')
      for (const k in kw) (a[0] as InstValue).attrs.set(k, kw[k])
      return NONE
    },
    __repr__: (a) => {
      const self = a[0] as InstValue
      const name = self.cls.name === 'SimpleNamespace' ? 'namespace' : self.cls.name
      return S(`${name}(${[...self.attrs].map(([n, v]) => `${n}=${repr(v)}`).join(', ')})`)
    },
    __eq__: (a) => {
      const [x, y] = a
      if (x.k !== 'inst' || y.k !== 'inst' || x.attrs.size !== y.attrs.size) return x.k === 'inst' && y.k === 'inst' ? bool(false) : NOT_IMPLEMENTED
      return bool([...x.attrs].every(([n, v]) => y.attrs.has(n) && h.truthy(h.compare('==', v, y.attrs.get(n)!))))
    },
    __hash__: () => { throw h.err('TypeError', "unhashable type: 'types.SimpleNamespace'") },
  })
  const named = (name: string, call: (a: Value[], kw: Kw) => Value = () => { throw h.err('NotImplementedError', `types.${name} cannot be called in this sandbox`) }, shown = name): Value => ({ k: 'type', name: shown, call })
  const FunctionType = named('FunctionType', undefined, 'function')
  return mod('types', {
    SimpleNamespace,
    FunctionType,
    LambdaType: FunctionType,
    BuiltinFunctionType: named('BuiltinFunctionType', undefined, 'builtin_function_or_method'),
    GeneratorType: named('GeneratorType', undefined, 'generator'),
    NoneType: named('NoneType', () => NONE),
    ModuleType: named('ModuleType', (a, kw) => {
      const name = kw.name ?? a[0]
      if (!name) throw h.err('TypeError', "module() missing required argument 'name' (pos 1)")
      if (name.k !== 'str') throw h.err('TypeError', `module() argument 'name' must be str, not ${typeName(name)}`)
      return mod(name.v, { __name__: name, __doc__: kw.doc ?? a[1] ?? NONE })
    }, 'module'),
    MethodType: named('MethodType', (a) => {
      if (!a[0] || !isCallable(h, a[0])) throw h.err('TypeError', 'first argument must be callable')
      if (!a[1] || a[1].k === 'none') throw h.err('TypeError', 'self must not be None')
      if (a[0].k === 'func') return { k: 'bound', self: a[1], f: a[0] }
      const [f, self] = a
      return { k: 'fn', name: 'method', call: (args, kw) => h.call(f, [self, ...args], kw) }
    }, 'method'),
    MappingProxyType: named('MappingProxyType', (a) => {
      if (a[0]?.k === 'obj' && a[0].o instanceof MappingProxy) return a[0]
      if (a[0]?.k !== 'dict') throw h.err('TypeError', `mappingproxy() argument must be a mapping, not ${typeName(a[0] ?? NONE)}`)
      return { k: 'obj', o: new MappingProxy(a[0]) }
    }, 'mappingproxy'),
  })
}

// ---------------------------------------------------------------- enum

interface EnumInfo { names: string[]; members: Map<string, InstValue>; byValue: PyDict; mixin: (Value & { k: 'type' }) | null }

/** the arithmetic and comparison a member of class Level(int, Enum) has from int */
const NUMBER_OPS: [string, string, string | null][] = [['+', '__add__', '__radd__'], ['-', '__sub__', '__rsub__'], ['*', '__mul__', '__rmul__'], ['/', '__truediv__', '__rtruediv__'], ['//', '__floordiv__', '__rfloordiv__'], ['%', '__mod__', '__rmod__'], ['**', '__pow__', '__rpow__'], ['&', '__and__', '__rand__'], ['|', '__or__', '__ror__'], ['^', '__xor__', '__rxor__'], ['<<', '__lshift__', '__rlshift__'], ['>>', '__rshift__', '__rrshift__']]
const COMPARE_OPS: [string, string][] = [['==', '__eq__'], ['!=', '__ne__'], ['<', '__lt__'], ['<=', '__le__'], ['>', '__gt__'], ['>=', '__ge__']]
const isDunder = (n: string) => n.length > 4 && n.startsWith('__') && n.endsWith('__') && n[2] !== '_' && n[n.length - 3] !== '_'
const isSunder = (n: string) => n.length > 2 && n[0] === '_' && n[n.length - 1] === '_' && n[1] !== '_' && n[n.length - 2] !== '_'

const makeEnum = (h: Host): Value => {
  const builtin = (n: string) => (h.module('builtins') as { attrs: Record<string, Value> }).attrs[n]
  const infos = new WeakMap<ClassValue, EnumInfo>()
  const emptyInfo = (): EnumInfo => ({ names: [], members: new Map(), byValue: new PyDict(), mixin: null })
  const infoOf = (c: ClassValue) => infos.get(c) ?? emptyInfo()
  const isEnum = (c: Value): c is ClassValue => c.k === 'class' && !!c.meta && c.meta === META
  const valueOf = (m: Value): Value => (m as InstValue).attrs.get('_value_') ?? NONE
  const nameOf = (m: Value): Value => (m as InstValue).attrs.get('_name_') ?? NONE
  const isMember = (v: Value, c: ClassValue): v is InstValue => {
    const walk = (x: ClassValue): boolean => x === c || x.bases.some((b) => b.k === 'class' && walk(b))
    return v.k === 'inst' && walk(v.cls)
  }
  const mixinOf = (bases: Value[]): (Value & { k: 'type' }) | null => {
    for (const b of bases) {
      if (b.k === 'type' && b.name !== 'object') return b
      if (isEnum(b) && infoOf(b).mixin) return infoOf(b).mixin
      if (b.k === 'class') {
        const m = mixinOf(b.bases)
        if (m) return m
      }
    }
    return null
  }
  const anyMembers = (c: Value, name: string) => {
    if (c.k !== 'class') return
    if (isEnum(c) && infoOf(c).names.length) throw h.err('TypeError', `${name}: cannot extend enumeration ${repr(S(c.name))}`)
    c.bases.forEach((b) => anyMembers(b, name))
  }

  /** the class body's namespace: what _EnumDict does with each name as it is assigned */
  class EnumNs extends Map<string, Value> {
    clsName: string
    open = true
    memberNames: string[] = []
    lastValues: Value[] = []
    autoCalled = false
    ignore: string[] = []
    gen: Value
    constructor(clsName: string, gen: Value) {
      super()
      this.clsName = clsName
      this.gen = gen
    }
    set(key: string, value0: Value): this {
      if (!this.open) return super.set(key, value0)
      let value = value0
      if (isSunder(key)) {
        if (!['_order_', '_create_pseudo_member_', '_generate_next_value_', '_missing_', '_ignore_'].includes(key)) throw h.err('ValueError', '_names_ are reserved for future Enum use')
        if (key === '_create_pseudo_member_') throw h.err('NotImplementedError', 'an Enum defining _create_pseudo_member_ is not available in this sandbox')
        if (key === '_generate_next_value_') {
          if (this.autoCalled) throw h.err('TypeError', '_generate_next_value_ must be defined before members')
          this.gen = value
        } else if (key === '_ignore_') {
          const names = value.k === 'str' ? value.v.replace(/,/g, ' ').trim().split(/\s+/).filter(Boolean) : h.iterate(value).map((x) => h.str(x))
          const already = names.filter((n) => this.memberNames.includes(n))
          if (already.length) throw h.err('ValueError', `_ignore_ cannot specify already set names: ${repr(tuple(already.map(S)))}`)
          this.ignore = names
        }
      } else if (isDunder(key)) {
        if (key === '__new__') throw h.err('NotImplementedError', 'an Enum defining __new__ is not available in this sandbox')
        if (key === '__order__') return super.set('_order_', value)
      } else if (this.memberNames.includes(key)) {
        throw h.err('TypeError', `Attempted to reuse key: ${repr(S(key))}`)
      } else if (!this.ignore.includes(key) && !isDescriptor(value)) {
        if (this.has(key)) throw h.err('TypeError', `${repr(S(key))} already defined as: ${repr(this.get(key)!)}`)
        if (value.k === 'inst' && value.cls === auto) {
          const given = value.attrs.get('value')
          if (!given) {
            this.autoCalled = true
            value = h.call(this.gen, [S(key), int(1), int(this.memberNames.length), L([...this.lastValues])])
            ;(value0 as InstValue).attrs.set('value', value)
          } else value = given
        }
        this.memberNames.push(key)
        this.lastValues.push(value)
      }
      return super.set(key, value)
    }
  }
  const isDescriptor = (v: Value) => v.k === 'func' || v.k === 'descr' || v.k === 'bound' || (v.k === 'fn' && !!v.bindable) || (v.k === 'inst' && !!h.getAttrOr(v, '__get__'))

  const enumStr = (m: InstValue) => S(`${m.cls.name}.${h.str(nameOf(m))}`)
  const flagName = (m: Value) => (h.truthy(nameOf(m)) ? h.str(nameOf(m)) : repr(valueOf(m)))
  const big = (v: Value) => {
    if (!Number.isInteger(numOf(v))) throw h.err('TypeError', `unsupported operand type(s) for &: 'int' and '${typeName(v)}'`)
    return BigInt(numOf(v))
  }
  const highBit = (n: bigint) => (n === 0n ? -1n : BigInt((n < 0n ? -n : n).toString(2).length - 1))
  /** Flag: the named members a value is made of, and the bits left over */
  const decompose = (c: ClassValue, value: bigint): [InstValue[], bigint] => {
    const info = infoOf(c)
    let notCovered = value
    const members: InstValue[] = []
    for (const m of info.names.map((n) => info.members.get(n)!)) {
      const mv = big(valueOf(m))
      if (mv && (mv & value) === mv) {
        members.push(m)
        notCovered &= ~mv
      }
    }
    if (value >= 0n) {
      let tmp = notCovered
      while (tmp) {
        const bit = 2n ** highBit(tmp)
        const known = info.byValue.get(int(Number(bit)))
        if (known) {
          members.push(known as InstValue)
          notCovered &= ~bit
        }
        tmp &= ~bit
      }
    }
    const exact = info.byValue.get(int(Number(value)))
    if (!members.length && exact) members.push(exact as InstValue)
    members.sort((a, b) => Number(big(valueOf(b)) - big(valueOf(a))))
    if (members.length > 1 && big(valueOf(members[0])) === value) members.shift()
    return [members, notCovered]
  }
  const pseudo = (c: ClassValue, value: Value): InstValue => {
    const made: InstValue = { k: 'inst', cls: c, attrs: new Map([['_value_', value], ['_name_', NONE]]), state: {} }
    infoOf(c).byValue.set(value, made)
    return made
  }
  const notAMember = (c: ClassValue, v: Value) => h.raise(h.excClass('ValueError'), [S(`${repr(v)} is not a valid ${c.name}`)])
  const flagPseudo = (c: ClassValue, value: Value): Value => {
    const known = infoOf(c).byValue.get(value)
    if (known) return known
    const [, extra] = decompose(c, big(value))
    if (extra) notAMember(c, value)
    return pseudo(c, value)
  }
  const intFlagPseudo = (c: ClassValue, value: Value): Value => {
    const info = infoOf(c)
    const known = info.byValue.get(value)
    if (known) return known
    const need = [value]
    let [, extra] = decompose(c, big(value))
    while (extra) {
      const flag = 2n ** highBit(extra)
      const fv = int(Number(flag))
      if (!info.byValue.has(fv) && !need.some((x) => numOf(x) === numOf(fv))) need.push(fv)
      extra = extra === -flag ? 0n : extra ^ flag
    }
    let last: Value = NONE
    for (const v of need.reverse()) last = info.byValue.get(v) ?? pseudo(c, v)
    return last
  }
  const lookup = (c: ClassValue, value: Value): Value => {
    if (value.k === 'inst' && value.cls === c) return value
    const info = infoOf(c)
    try {
      const m = info.byValue.get(value)
      if (m) return m
    } catch (e) {
      if (!caught(h, e, 'TypeError')) throw e
      for (const m of info.members.values()) if (h.truthy(h.compare('==', valueOf(m), value))) return m
    }
    let result: Value = NONE
    let failed: unknown = null
    try {
      result = h.call(h.getAttr(c, '_missing_'), [value])
    } catch (e) {
      if (!caught(h, e, 'Exception')) throw e
      failed = e
    }
    if (isMember(result, c)) return result
    if (failed) throw failed
    if (result.k === 'none') return notAMember(c, value)
    return h.raise(h.excClass('TypeError'), [S(`error in ${c.name}._missing_: returned ${repr(result)} instead of None or a valid member`)])
  }
  const generated = (n: string, f: (a: Value[]) => Value): Value => ({ k: 'descr', kind: 'static', f: F(n, f) })
  const createEnum = (base: ClassValue, nameV: Value, namesV: Value, kw: Kw): Value => {
    const typeV = kw.type
    const start = kw.start ?? int(1)
    const bases: Value[] = typeV && typeV.k !== 'none' ? [typeV, base] : [base]
    const className = h.str(nameV)
    const ns = META.prepare!(className, bases, h) as EnumNs
    let names: Value = namesV
    if (names.k === 'str') names = L(names.v.replace(/,/g, ' ').trim().split(/\s+/).filter(Boolean).map(S))
    if ((names.k === 'tuple' || names.k === 'list') && names.items[0]?.k === 'str') {
      const last: Value[] = []
      names = L(names.items.map((n, count) => {
        const value = h.call(ns.gen, [n, start, int(count), L([...last])])
        last.push(value)
        return tuple([n, value])
      }))
    }
    if (names.k === 'dict') for (const [k, v] of names.d.items()) ns.set(h.str(k), v)
    else for (const item of h.iterate(names)) {
      const [k, v] = h.iterate(item)
      ns.set(h.str(k), v)
    }
    const cls: ClassValue = { k: 'class', name: className, bases, ns, host: base.host, meta: META }
    META.created!(cls, h)
    return cls
  }

  /** class Level(int, Enum): what a member takes from the built-in type */
  const mixIn = (c: ClassValue, type: Value & { k: 'type' }) => {
    const put = (n: string, call: Fn) => { if (!c.ns.has(n)) (c.ns as Map<string, Value>).set(n, { k: 'fn', name: n, bindable: true, call }) }
    const plain = (v: Value): Value | null => {
      if (v.k === 'inst') return v.cls.meta === META && infoOf(v.cls).mixin ? valueOf(v) : null
      return v
    }
    const fits = (other: Value) => (type.name === 'str' ? other.k === 'str' : isNum(other) || other.k === 'array')
    for (const [op, name, rname] of NUMBER_OPS) {
      if (type.name === 'str' && op !== '+' && op !== '*' && op !== '%') continue
      // 'ab' * Level.ONE and Size.ONE * [0]: an int repeats a sequence
      const repeats = (x: Value) => op === '*' && type.name === 'int' && (x.k === 'list' || x.k === 'tuple' || x.k === 'str')
      const operand = (x: Value | null) => x !== null && (fits(x) || repeats(x) || (type.name === 'str' && (op === '*' || op === '%')))
      put(name, (a) => {
        const other = plain(a[1])
        return operand(other) ? h.binop(op, valueOf(a[0]), other!) : NOT_IMPLEMENTED
      })
      if (rname && !(type.name === 'str' && op === '%')) put(rname, (a) => {
        const other = plain(a[1])
        return operand(other) ? h.binop(op, other!, valueOf(a[0])) : NOT_IMPLEMENTED
      })
    }
    for (const [op, name] of COMPARE_OPS) put(name, (a) => {
      const other = plain(a[1])
      return other !== null && fits(other) ? h.compare(op, valueOf(a[0]), other) : NOT_IMPLEMENTED
    })
    put('__hash__', (a) => h.call(builtin('hash'), [valueOf(a[0])]))
    put('__int__', (a) => h.call(builtin('int'), [valueOf(a[0])]))
    if (type.name !== 'str') {
      put('__float__', (a) => h.call(builtin('float'), [valueOf(a[0])]))
      put('__bool__', (a) => bool(h.truthy(valueOf(a[0]))))
      put('__abs__', (a) => h.call(builtin('abs'), [valueOf(a[0])]))
      put('__neg__', (a) => h.binop('*', valueOf(a[0]), int(-1)))
      put('__pos__', (a) => (type.name === 'int' ? int(numOf(valueOf(a[0]))) : float(numOf(valueOf(a[0])))))
      put('__round__', (a) => h.call(builtin('round'), [valueOf(a[0]), ...a.slice(1)]))
    } else {
      put('__len__', (a) => h.call(builtin('len'), [valueOf(a[0])]))
      put('__iter__', (a) => h.call(builtin('iter'), [valueOf(a[0])]))
      put('__getitem__', (a) => h.call(h.getAttr(valueOf(a[0]), '__getitem__'), [a[1]]))
      put('__contains__', (a) => h.compare('in', plain(a[1]) ?? a[1], valueOf(a[0])))
      put('__bool__', (a) => bool(h.truthy(valueOf(a[0]))))
    }
    if (type.name === 'int') {
      put('__index__', (a) => valueOf(a[0]))
      put('__invert__', (a) => int(-numOf(valueOf(a[0])) - 1))
    }
  }

  const META: ClassMeta = {
    prepare(name, bases) {
      for (const b of bases) anyMembers(b, name)
      const last = bases[bases.length - 1]
      if (bases.length && !(last.k === 'class' && isEnumFamily(last))) throw h.err('TypeError', 'new enumerations should be created as `EnumName([mixin_type, ...] [data_type,] enum_type)`')
      return new EnumNs(name, h.getAttr(last, '_generate_next_value_'))
    },
    created(cls) {
      const ns = cls.ns as EnumNs
      ns.open = false
      const members = ns.memberNames.map((n) => [n, ns.get(n)!] as const)
      for (const n of ns.memberNames) ns.delete(n)
      for (const n of ns.ignore) ns.delete(n)
      if (ns.memberNames.includes('mro')) throw h.err('ValueError', 'Invalid enum member name: mro')
      const mixin = mixinOf(cls.bases)
      if (mixin && !['int', 'str', 'float'].includes(mixin.name)) throw h.err('NotImplementedError', `an Enum mixed with '${mixin.name}' is not available in this sandbox (int, str and float are)`)
      if (cls.bases.some((b) => b.k === 'type' && b.name !== 'object') && mixin) mixIn(cls, mixin)
      const info: EnumInfo = { ...emptyInfo(), mixin }
      infos.set(cls, info)
      if (!ns.has('__doc__')) ns.set('__doc__', S('An enumeration.'))
      for (const [name, given] of members) {
        const args = given.k === 'tuple' ? given.items : [given]
        const value = mixin ? h.call(mixin, args) : given
        let member: InstValue = { k: 'inst', cls, attrs: new Map([['_value_', value], ['_name_', S(name)]]), state: {} }
        const init = h.getAttrOr(member, '__init__')
        if (init) h.call(init, args)
        const canonical = [...info.members.values()].find((m) => h.truthy(h.compare('==', valueOf(m), valueOf(member))))
        if (canonical) member = canonical
        else info.names.push(name)
        ns.set(name, member)
        info.members.set(name, member)
        try {
          info.byValue.set(value, member)
        } catch (e) {
          if (!caught(h, e, 'TypeError')) throw e
        }
      }
      const order = ns.get('_order_')
      if (order && order.k !== 'none') {
        const want = order.k === 'str' ? order.v.replace(/,/g, ' ').trim().split(/\s+/).filter(Boolean) : h.iterate(order).map((x) => h.str(x))
        if (want.join('\n') !== info.names.join('\n')) throw h.err('TypeError', 'member order does not match _order_')
      }
    },
    call(cls, args, kw) {
      const value = args[0] ?? kw.value
      if (!value) throw h.err('TypeError', "EnumMeta.__call__() missing 1 required positional argument: 'value'")
      const names = args[1] ?? kw.names
      return names && names.k !== 'none' ? createEnum(cls, value, names, kw) : lookup(cls, value)
    },
    getItem(cls, idx) {
      const m = idx.k === 'str' ? infoOf(cls).members.get(idx.v) : undefined
      if (!m) throw h.err('KeyError', repr(idx))
      return m
    },
    iter: (cls) => infoOf(cls).names.map((n) => infoOf(cls).members.get(n)!),
    len: (cls) => infoOf(cls).names.length,
    contains(cls, v) {
      if (!isMember(v, EnumBase)) throw h.err('TypeError', `unsupported operand type(s) for 'in': '${typeName(v)}' and 'EnumMeta'`)
      const n = nameOf(v)
      return isMember(v, cls) && n.k === 'str' && infoOf(cls).members.has(n.v)
    },
    getAttr(cls, name) {
      const info = infoOf(cls)
      switch (name) {
        case '__members__': return { k: 'obj', o: new MappingProxy({ k: 'dict', d: PyDict.from([...info.members].map(([n, m]) => [S(n), m])) }) }
        case '_member_names_': return L(info.names.map(S))
        case '_member_map_': return { k: 'dict', d: PyDict.from([...info.members].map(([n, m]) => [S(n), m])) }
        case '_value2member_map_': return { k: 'dict', d: info.byValue }
        case '_member_type_': return info.mixin ?? builtin('object')
      }
      return h.raise(h.excClass('AttributeError'), [S(name)])
    },
    setAttr(cls, name) {
      if (infoOf(cls).members.has(name)) throw h.err('AttributeError', 'Cannot reassign members.')
      return false
    },
    repr: (cls) => `<enum ${repr(S(cls.name))}>`,
    scalar: (m) => (infoOf(m.cls).mixin ? valueOf(m) : undefined),
  }
  const HOST: HostClass = {
    init() {},
    // class Mode(str, Enum): Mode.TRAIN.upper()
    getAttr: (inst, name) => (infoOf(inst.cls).mixin ? h.getAttrOr(valueOf(inst), name) : undefined),
    // for the variable pane, once the run has ended
    repr: (inst) => `<${inst.cls.name}.${str(nameOf(inst))}: ${repr(valueOf(inst))}>`,
  }
  const base = (name: string, bases: Value[], methods: Record<string, Value>): ClassValue => ({ k: 'class', name, module: 'enum', bases, ns: new Map(Object.entries(methods)), host: HOST, builtin: true, meta: META })
  const method = (n: string, call: Fn): Value => ({ k: 'fn', name: n, bindable: true, call })
  const readOnly = (f: (m: Value) => Value): Value => ({ k: 'descr', kind: 'prop', f: F('property', (a) => f(a[0])), setter: F('setter', () => { throw h.err('AttributeError', "can't set attribute") }) })
  const auto = libClass('enum', 'auto', {
    __init__: () => NONE,
  })
  const enumStrFn = method('__str__', (a) => enumStr(a[0] as InstValue))
  const flagStrFn = method('__str__', (a) => {
    const m = a[0] as InstValue
    if (nameOf(m).k !== 'none') return enumStr(m)
    const [members] = decompose(m.cls, big(valueOf(m)))
    if (members.length === 1 && nameOf(members[0]).k === 'none') return S(`${m.cls.name}.${repr(valueOf(members[0]))}`)
    return S(`${m.cls.name}.${members.map(flagName).join('|')}`)
  })
  const EnumBase = base('Enum', [], {
    __doc__: S('Generic enumeration.\n\n    Derive from this class to define new enumerations.\n    '),
    name: readOnly(nameOf),
    value: readOnly(valueOf),
    _generate_next_value_: generated('_generate_next_value_', (a) => {
      for (const last of [...h.iterate(a[3])].reverse()) {
        try {
          return h.binop('+', last, int(1))
        } catch (e) {
          if (!caught(h, e, 'TypeError')) throw e
        }
      }
      return a[1]
    }),
    _missing_: { k: 'descr', kind: 'class', f: F('_missing_', () => NONE) },
    __repr__: method('__repr__', (a) => S(`<${(a[0] as InstValue).cls.name}.${h.str(nameOf(a[0]))}: ${repr(valueOf(a[0]))}>`)),
    __str__: enumStrFn,
    __format__: method('__format__', (a) => {
      const m = a[0] as InstValue
      const spec = (a[1] as { v: string }).v
      const own = h.getAttr(m.cls, '__str__')
      const info = infoOf(m.cls)
      if (!info.mixin || (own !== enumStrFn && own !== flagStrFn)) return S(formatSpec(S(h.str(m)), spec))
      return S(formatSpec(valueOf(m), spec))
    }),
    __copy__: method('__copy__', (a) => a[0]),
    __deepcopy__: method('__deepcopy__', (a) => a[0]),
  })
  infos.set(EnumBase, emptyInfo())
  const isEnumFamily = (c: ClassValue) => {
    const walk = (x: ClassValue): boolean => x === EnumBase || x.bases.some((b) => b.k === 'class' && walk(b))
    return walk(c)
  }
  const bitOp = (f: (x: bigint, y: bigint) => bigint) => (a: Value[]): Value => {
    const self = a[0] as InstValue
    if (!isMember(a[1], self.cls)) return NOT_IMPLEMENTED
    return lookup(self.cls, int(Number(f(big(valueOf(self)), big(valueOf(a[1]))))))
  }
  const Flag = base('Flag', [EnumBase], {
    __doc__: S('\n    Support for flags\n    '),
    _generate_next_value_: generated('_generate_next_value_', (a) => {
      if (!numOf(a[2])) return a[1].k === 'none' ? int(1) : a[1]
      const last = [...h.iterate(a[3])].reverse()[0]
      if (last.k !== 'int' && last.k !== 'bool') throw h.err('TypeError', `Invalid Flag value: ${repr(last)}`)
      return int(2 ** (Number(highBit(big(last))) + 1))
    }),
    _missing_: { k: 'descr', kind: 'class', f: F('_missing_', (a) => {
      const c = a[0] as ClassValue
      const negative = h.truthy(h.compare('<', a[1], int(0)))
      const value = negative ? int(-numOf(a[1]) - 1) : a[1]
      const made = flagPseudo(c, value)
      return negative ? h.call(h.getAttr(made, '__invert__'), []) : made
    }) },
    __repr__: method('__repr__', (a) => {
      const m = a[0] as InstValue
      if (nameOf(m).k !== 'none') return S(`<${m.cls.name}.${h.str(nameOf(m))}: ${repr(valueOf(m))}>`)
      const [members] = decompose(m.cls, big(valueOf(m)))
      return S(`<${m.cls.name}.${members.map(flagName).join('|')}: ${repr(valueOf(m))}>`)
    }),
    __str__: flagStrFn,
    __bool__: method('__bool__', (a) => bool(h.truthy(valueOf(a[0])))),
    __contains__: method('__contains__', (a) => {
      const self = a[0] as InstValue
      if (!isMember(a[1], self.cls)) throw h.err('TypeError', `unsupported operand type(s) for 'in': '${typeName(a[1])}' and '${self.cls.name}'`)
      const want = big(valueOf(a[1]))
      return bool((want & big(valueOf(self))) === want)
    }),
    __or__: method('__or__', bitOp((x, y) => x | y)),
    __and__: method('__and__', bitOp((x, y) => x & y)),
    __xor__: method('__xor__', bitOp((x, y) => x ^ y)),
    __invert__: method('__invert__', (a) => {
      const self = a[0] as InstValue
      const c = self.cls
      const value = big(valueOf(self))
      const [members] = decompose(c, value)
      let inverted = lookup(c, int(0))
      const info = infoOf(c)
      for (const m of info.names.map((n) => info.members.get(n)!)) {
        if (!members.includes(m) && !(big(valueOf(m)) & value)) inverted = h.binop('|', inverted, m)
      }
      return lookup(c, inverted)
    }),
  })
  infos.set(Flag, emptyInfo())
  const intType = builtin('int') as Value & { k: 'type' }
  const IntEnum = base('IntEnum', [intType, EnumBase], { __doc__: S('Enum where members are also (and must be) ints') })
  infos.set(IntEnum, { ...emptyInfo(), mixin: intType })
  mixIn(IntEnum, intType)
  const intBit = (f: (x: bigint, y: bigint) => bigint) => (a: Value[]): Value => {
    const self = a[0] as InstValue
    const other = a[1]
    if (!isMember(other, self.cls) && other.k !== 'int' && other.k !== 'bool') return NOT_IMPLEMENTED
    return lookup(self.cls, int(Number(f(big(valueOf(self)), big(valueOf(lookup(self.cls, other)))))))
  }
  const bitNames: [string, (x: bigint, y: bigint) => bigint][] = [['or', (x, y) => x | y], ['and', (x, y) => x & y], ['xor', (x, y) => x ^ y]]
  const IntFlag = base('IntFlag', [intType, Flag], {
    __doc__: S('\n    Support for integer-based Flags\n    '),
    ...Object.fromEntries(bitNames.flatMap(([n, f]) => [[`__${n}__`, method(`__${n}__`, intBit(f))], [`__r${n}__`, method(`__r${n}__`, intBit(f))]])),
    __invert__: method('__invert__', (a) => lookup((a[0] as InstValue).cls, int(-numOf(valueOf(a[0])) - 1))),
    _missing_: { k: 'descr', kind: 'class', f: F('_missing_', (a) => {
      const c = a[0] as ClassValue
      if (a[1].k !== 'int' && a[1].k !== 'bool') return notAMember(c, a[1])
      return intFlagPseudo(c, a[1])
    }) },
  })
  infos.set(IntFlag, { ...emptyInfo(), mixin: intType })
  mixIn(IntFlag, intType)
  return mod('enum', {
    Enum: EnumBase,
    IntEnum,
    Flag,
    IntFlag,
    auto,
    unique: F('unique', (a) => {
      const info = infoOf(a[0] as ClassValue)
      const dups = [...info.members].filter(([n, m]) => n !== h.str(nameOf(m))).map(([n, m]) => `${n} -> ${h.str(nameOf(m))}`)
      if (dups.length) throw h.err('ValueError', `duplicate values found in ${repr(a[0])}: ${dups.join(', ')}`)
      return a[0]
    }),
  })
}

// ---------------------------------------------------------------- contextlib

const makeContextlib = (h: Host): Value => {
  const sys = () => h.module('sys') as Value & { k: 'module' }
  const redirect = (name: string, stream: 'stdout' | 'stderr') => libClass('contextlib', name, {
    __init__: (a, kw) => {
      const self = a[0] as InstValue
      self.attrs.set('_new_target', kw.new_target ?? a[1])
      self.state.old = []
      return NONE
    },
    __enter__: (a) => {
      const self = a[0] as InstValue
      ;(self.state.old as Value[]).push(sys().attrs[stream])
      sys().attrs[stream] = self.attrs.get('_new_target')!
      return self.attrs.get('_new_target')!
    },
    __exit__: (a) => {
      sys().attrs[stream] = ((a[0] as InstValue).state.old as Value[]).pop()!
      return NONE
    },
  })
  /** what a generator-based context manager does at each end of the with block */
  const GeneratorCM = libClass('contextlib', '_GeneratorContextManager', {
    __enter__: (a) => {
      const self = a[0] as InstValue
      // it deletes what it could be remade from, so a second with fails on that
      if (self.state.entered) h.raise(h.excClass('AttributeError'), [S('args')])
      self.state.entered = true
      const gen = self.attrs.get('gen')!
      if (gen.k !== 'iter') throw h.err('TypeError', `'${typeName(gen)}' object is not an iterator`)
      const first = gen.next()
      if (first === undefined) throw h.err('RuntimeError', "generator didn't yield")
      return first
    },
    __exit__: (a) => {
      const gen = (a[0] as InstValue).attrs.get('gen') as Value & { k: 'iter' }
      const [, type, value] = a
      if (type.k === 'none') {
        if (gen.next() === undefined) return bool(false)
        throw h.err('RuntimeError', "generator didn't stop")
      }
      if (!gen.throw) throw h.err('TypeError', 'a contextmanager function must be a generator')
      try {
        h.call(h.getAttr(gen, 'throw'), [value])
      } catch (e) {
        if (!(e instanceof PyError) || e.fatal) throw e
        const thrown = (e.exc as InstValue | undefined) ?? null
        if (thrown === value) return bool(false)
        if (caught(h, e, 'StopIteration')) return bool(true)
        throw e
      }
      throw h.err('RuntimeError', "generator didn't stop after throw()")
    },
    /** @tag('x'): each call runs inside a fresh generator made from the same arguments */
    __call__: (a) => {
      const again = (a[0] as InstValue).state.again as () => Value
      const f = a[1]
      return { k: 'fn', name: f.k === 'func' ? f.name : 'inner', call: (args, kw) => {
        const gen = again() as Value & { k: 'iter' }
        if (gen.next() === undefined) throw h.err('RuntimeError', "generator didn't yield")
        let out: Value
        try {
          out = h.call(f, args, kw)
        } catch (e) {
          if (!(e instanceof PyError) || e.fatal || !gen.throw) throw e
          let next: Value | undefined
          try {
            next = gen.throw(e)
          } catch (again2) {
            if (caught(h, again2, 'StopIteration')) return py.NONE
            throw again2
          }
          if (next === undefined) return py.NONE
          throw h.err('RuntimeError', "generator didn't stop after throw()")
        }
        if (gen.next() !== undefined) throw h.err('RuntimeError', "generator didn't stop")
        return out
      } }
    },
  })
  return mod('contextlib', {
    contextmanager: F('contextmanager', (a) => {
      const f = a[0]
      const name = f.k === 'func' ? f.name : 'helper'
      return { k: 'fn', name, bindable: true, attrs: new Map<string, Value>([['__wrapped__', f]]), call: (args, kw) => {
        const made = h.call(GeneratorCM, []) as InstValue
        made.attrs.set('gen', h.call(f, args, kw))
        made.state.again = () => h.call(f, args, kw)
        return made
      } }
    }),
    redirect_stdout: redirect('redirect_stdout', 'stdout'),
    redirect_stderr: redirect('redirect_stderr', 'stderr'),
    suppress: libClass('contextlib', 'suppress', {
      __init__: (a) => {
        (a[0] as InstValue).attrs.set('_exceptions', tuple(a.slice(1)))
        return NONE
      },
      __enter__: () => NONE,
      __exit__: (a) => {
        const [self, type] = a as [InstValue, Value]
        if (type.k === 'none') return bool(false)
        return h.call((h.module('builtins') as { attrs: Record<string, Value> }).attrs.issubclass, [type, self.attrs.get('_exceptions')!])
      },
    }),
    nullcontext: libClass('contextlib', 'nullcontext', {
      __init__: (a, kw) => {
        (a[0] as InstValue).attrs.set('enter_result', kw.enter_result ?? a[1] ?? NONE)
        return NONE
      },
      __enter__: (a) => (a[0] as InstValue).attrs.get('enter_result')!,
      __exit__: () => NONE,
    }),
    closing: libClass('contextlib', 'closing', {
      __init__: (a, kw) => {
        (a[0] as InstValue).attrs.set('thing', kw.thing ?? a[1])
        return NONE
      },
      __enter__: (a) => (a[0] as InstValue).attrs.get('thing')!,
      __exit__: (a) => {
        h.call(h.getAttr((a[0] as InstValue).attrs.get('thing')!, 'close'), [])
        return NONE
      },
    }),
  })
}

// ---------------------------------------------------------------- inspect

const makeInspect = (h: Host): Value => {
  const Empty: ClassValue = { k: 'class', name: '_empty', module: 'inspect', bases: [], ns: new Map(), host: null, builtin: true }
  const formatNote = (v: Value): string => {
    if (v.k === 'class') return v.builtin && !v.module ? v.name : `${v.module ?? '__main__'}.${v.name}`
    if (v.k === 'type') return v.name
    return v.k === 'obj' && v.o instanceof TypingAlias ? repr(v).replace(/typing\./g, '') : repr(v)
  }
  type Kind = 'POSITIONAL_ONLY' | 'POSITIONAL_OR_KEYWORD' | 'VAR_POSITIONAL' | 'KEYWORD_ONLY' | 'VAR_KEYWORD'
  const KINDS: Kind[] = ['POSITIONAL_ONLY', 'POSITIONAL_OR_KEYWORD', 'VAR_POSITIONAL', 'KEYWORD_ONLY', 'VAR_KEYWORD']
  const DESCRIPTIONS: Record<Kind, string> = { POSITIONAL_ONLY: 'positional-only', POSITIONAL_OR_KEYWORD: 'positional or keyword', VAR_POSITIONAL: 'variadic positional', KEYWORD_ONLY: 'keyword-only', VAR_KEYWORD: 'variadic keyword' }
  class ParamKind extends Base {
    readonly cls = '_ParameterKind'
    kind: Kind
    constructor(kind: Kind) {
      super()
      this.kind = kind
    }
    repr() { return `<_ParameterKind.${this.kind}: ${KINDS.indexOf(this.kind)}>` }
    str() { return this.kind }
    getAttr(name: string): Value | undefined {
      if (name === 'name') return S(this.kind)
      if (name === 'value') return int(KINDS.indexOf(this.kind))
      if (name === 'description') return S(DESCRIPTIONS[this.kind])
      return undefined
    }
    asIndex() { return KINDS.indexOf(this.kind) }
    asNumber() { return KINDS.indexOf(this.kind) }
    binop(op: string, other: Value, reflected: boolean, host: Host): Value | undefined {
      const at = KINDS.indexOf(this.kind)
      const num = other.k === 'obj' && other.o instanceof ParamKind ? int(KINDS.indexOf(other.o.kind)) : other
      if (!isNum(num)) return undefined
      return reflected ? host.binop(op, num, int(at)) : host.binop(op, int(at), num)
    }
  }
  const kinds = Object.fromEntries(KINDS.map((k) => [k, { k: 'obj', o: new ParamKind(k) } as Value])) as Record<Kind, Value>
  class Parameter extends Base {
    readonly cls = 'Parameter'
    name: string
    kind: Kind
    def: Value | null
    note: Value | null
    constructor(name: string, kind: Kind, def: Value | null, note: Value | null) {
      super()
      this.name = name
      this.kind = kind
      this.def = def
      this.note = note
    }
    str() {
      const star = this.kind === 'VAR_POSITIONAL' ? '*' : this.kind === 'VAR_KEYWORD' ? '**' : ''
      const noted = this.note ? `: ${formatNote(this.note)}` : ''
      return `${star}${this.name}${noted}${this.def ? `${this.note ? ' = ' : '='}${repr(this.def)}` : ''}`
    }
    repr() { return `<Parameter "${this.str()}">` }
    getAttr(name: string): Value | undefined {
      switch (name) {
        case 'name': return S(this.name)
        case 'default': return this.def ?? Empty
        case 'annotation': return this.note ?? Empty
        case 'kind': return kinds[this.kind]
        case 'empty': return Empty
      }
      return kinds[name as Kind]
    }
  }
  class Signature extends Base {
    readonly cls = 'Signature'
    params: Parameter[]
    returns: Value | null
    constructor(params: Parameter[], returns: Value | null) {
      super()
      this.params = params
      this.returns = returns
    }
    str() {
      const out: string[] = []
      let renderedPos = false
      let starred = true
      for (const p of this.params) {
        if (p.kind === 'POSITIONAL_ONLY') renderedPos = true
        else if (renderedPos) {
          out.push('/')
          renderedPos = false
        }
        if (p.kind === 'VAR_POSITIONAL') starred = false
        else if (p.kind === 'KEYWORD_ONLY' && starred) {
          out.push('*')
          starred = false
        }
        out.push(p.str())
      }
      if (renderedPos) out.push('/')
      return `(${out.join(', ')})${this.returns ? ` -> ${this.returns.k === 'none' ? 'None' : formatNote(this.returns)}` : ''}`
    }
    repr() { return `<Signature ${this.str()}>` }
    getAttr(name: string): Value | undefined {
      switch (name) {
        case 'parameters': return { k: 'obj', o: new MappingProxy({ k: 'dict', flavor: 'OrderedDict', d: PyDict.from(this.params.map((p) => [S(p.name), { k: 'obj', o: p }])) }) }
        case 'return_annotation': return this.returns ?? Empty
        case 'empty': return Empty
      }
      return undefined
    }
  }
  const fromFunc = (f: Value & { k: 'func' }, dropFirst: boolean): Signature => {
    const params = f.params.map((p, i): Parameter => {
      const kind: Kind = p.kind === 'star' ? 'VAR_POSITIONAL' : p.kind === 'kwstar' ? 'VAR_KEYWORD' : p.kind === 'kwonly' ? 'KEYWORD_ONLY' : p.posOnly ? 'POSITIONAL_ONLY' : 'POSITIONAL_OR_KEYWORD'
      return new Parameter(p.name, kind, f.defaults[i], f.notes?.get(p.name) ?? null)
    })
    return new Signature(dropFirst ? params.slice(1) : params, f.notes?.get('return') ?? null)
  }
  const FACTORY = { k: 'obj', o: new (class extends Base {
    readonly cls = '_HAS_DEFAULT_FACTORY_CLASS'
    repr() { return '<factory>' }
  })() } as Value
  const dataclassSignature = (c: ClassValue, fields: Value): Signature => {
    const noteOf = (n: string, at: Value): Value | null => {
      if (at.k !== 'class') return null
      const notes = at.ns.get('__annotations__')
      const own = notes?.k === 'dict' ? notes.d.get(S(n)) : undefined
      return own ?? at.bases.map((b) => noteOf(n, b)).find(Boolean) ?? null
    }
    const params: Parameter[] = []
    for (const f of h.iterate(fields)) {
      const n = h.str(f)
      const d = h.getAttrOr(c, n)
      const spec = d?.k === 'obj' && d.o instanceof FieldSpec ? d.o : null
      if (spec && !spec.init) continue
      params.push(new Parameter(n, 'POSITIONAL_OR_KEYWORD', spec ? (spec.factory ? FACTORY : spec.def ?? null) : d ?? null, noteOf(n, c)))
    }
    return new Signature(params, NONE)
  }
  const signatureOf = (v: Value, followed = false): Signature => {
    if (v.k === 'func' || v.k === 'fn') {
      const wrapped = v.attrs?.get('__wrapped__')
      if (wrapped && !followed) return signatureOf(wrapped)
    }
    if (v.k === 'func') return fromFunc(v, false)
    if (v.k === 'bound') return fromFunc(v.f, true)
    if (v.k === 'descr' && v.f.k === 'func') return fromFunc(v.f, v.kind !== 'static')
    if (v.k === 'class') {
      const fields = h.getAttrOr(v, '__dataclass_fields__')
      if (fields) return dataclassSignature(v, fields)
      const init = h.getAttrOr(v, '__init__')
      if (init?.k === 'func') return fromFunc(init, true)
      if (!init) return new Signature([], null)
    }
    if (v.k === 'inst') {
      const call = h.getAttrOr(v, '__call__')
      if (call?.k === 'bound') return fromFunc(call.f, true)
      if (!call) throw h.err('TypeError', `${repr(v)} is not a callable object`)
    }
    if (v.k === 'fn' || v.k === 'type' || v.k === 'class' || v.k === 'obj') {
      if (v.k === 'obj' && !v.o.call) throw h.err('TypeError', `${repr(v)} is not a callable object`)
      throw h.err('ValueError', `no signature found for builtin ${repr(v)}`)
    }
    throw h.err('TypeError', `${repr(v)} is not a callable object`)
  }
  const cleandoc = (doc: string) => {
    const lines = doc.replace(/\t/g, '        ').split('\n')
    let margin = Infinity
    for (const line of lines.slice(1)) {
      const content = line.trimStart().length
      if (content) margin = Math.min(margin, line.length - content)
    }
    if (lines.length) lines[0] = lines[0].trimStart()
    if (margin < Infinity) for (let i = 1; i < lines.length; i++) lines[i] = lines[i].slice(margin)
    while (lines.length && !lines[lines.length - 1]) lines.pop()
    while (lines.length && !lines[0]) lines.shift()
    return lines.join('\n')
  }
  const is = (name: string, test: (v: Value) => boolean): Value => F(name, (a) => bool(test(a[0])))
  const memberNames = (v: Value): string[] => {
    const names = new Set<string>()
    if (v.k === 'module') Object.keys(v.attrs).forEach((n) => names.add(n))
    if (v.k === 'inst') for (const n of v.attrs.keys()) names.add(n)
    const walk = (c: Value) => {
      if (c.k !== 'class') return
      for (const n of c.ns.keys()) names.add(n)
      c.bases.forEach(walk)
    }
    walk(v.k === 'inst' ? v.cls : v)
    return [...names]
  }
  return takingKeywords(mod('inspect', {
    Parameter: { k: 'type', name: 'inspect.Parameter', call: () => { throw h.err('NotImplementedError', 'inspect.Parameter cannot be built in this sandbox: inspect.signature gives them') }, attrs: new Map<string, Value>([...Object.entries(kinds), ['empty', Empty]]) },
    Signature: { k: 'type', name: 'inspect.Signature', call: () => { throw h.err('NotImplementedError', 'inspect.Signature cannot be built in this sandbox: use inspect.signature') }, attrs: new Map<string, Value>([['empty', Empty]]) },
    signature: F('signature', (a, kw) => {
      refuse(h, 'inspect.signature', kw, ['globals', 'locals', 'eval_str'])
      const follow = kw.follow_wrapped
      return { k: 'obj', o: signatureOf(kw.obj ?? a[0], !!follow && !h.truthy(follow)) }
    }),
    isfunction: is('isfunction', (v) => v.k === 'func'),
    ismethod: is('ismethod', (v) => v.k === 'bound'),
    isclass: is('isclass', (v) => v.k === 'class' || v.k === 'type'),
    ismodule: is('ismodule', (v) => v.k === 'module'),
    isbuiltin: is('isbuiltin', (v) => v.k === 'fn'),
    isroutine: is('isroutine', (v) => v.k === 'func' || v.k === 'bound' || v.k === 'fn'),
    isgenerator: is('isgenerator', (v) => v.k === 'iter' && v.name === 'generator'),
    isgeneratorfunction: is('isgeneratorfunction', (v) => v.k === 'func' && !!v.isGen),
    getdoc: F('getdoc', (a) => {
      const v = a[0]
      const direct = h.getAttrOr(v, '__doc__')
      if (direct?.k === 'str') return S(cleandoc(direct.v))
      if (v.k === 'class' || v.k === 'inst') {
        const walk = (c: ClassValue): string | null => {
          const d = c.ns.get('__doc__')
          if (d?.k === 'str') return d.v
          for (const b of c.bases) if (b.k === 'class') {
            const found = walk(b)
            if (found !== null) return found
          }
          return null
        }
        const found = walk(v.k === 'class' ? v : v.cls)
        if (found !== null) return S(cleandoc(found))
      }
      return NONE
    }),
    cleandoc: F('cleandoc', (a) => S(cleandoc(h.str(a[0])))),
    getmro: F('getmro', (a) => h.getAttr(a[0], '__mro__')),
    getmembers: F('getmembers', (a, kw) => {
      const pred = kw.predicate ?? a[1]
      const out: Value[] = []
      for (const n of memberNames(a[0]).sort()) {
        const v = h.getAttrOr(a[0], n)
        if (!v) continue
        if (!pred || pred.k === 'none' || h.truthy(h.call(pred, [v]))) out.push(tuple([S(n), v]))
      }
      return L(out)
    }),
    getsource: F('getsource', () => { throw h.err('NotImplementedError', 'inspect.getsource is not available in this sandbox') }),
  }), { signature: ['obj', 'follow_wrapped', 'globals', 'locals', 'eval_str'], getmembers: ['object', 'predicate'], getdoc: ['object'] })
}

// ---------------------------------------------------------------- fractions

const bigGcd = (a0: bigint, b0: bigint) => {
  let [a, b] = [a0 < 0n ? -a0 : a0, b0 < 0n ? -b0 : b0]
  while (b) [a, b] = [b, a % b]
  return a
}
const floorDiv = (a: bigint, b: bigint) => {
  const q = a / b
  return (a % b !== 0n) && ((a < 0n) !== (b < 0n)) ? q - 1n : q
}
/** a float as the exact fraction it is */
const floatRatio = (x: number): [bigint, bigint] => {
  let d = 1n
  let v = x
  while (!Number.isInteger(v)) {
    v *= 2
    d *= 2n
  }
  return [BigInt(v), d]
}
/** n / d, correctly rounded, as Python's int / int */
const ratioToFloat = (n: bigint, d: bigint): number => {
  const neg = n < 0n
  const a = neg ? -n : n
  if (a === 0n) return 0
  const shift = a.toString(2).length - d.toString(2).length
  const scale = 66 - shift
  const scaled = scale >= 0 ? (a << BigInt(scale)) : a
  const denom = scale >= 0 ? d : d << BigInt(-scale)
  const q = scaled / denom
  const sticky = q * denom !== scaled ? 1n : 0n
  const r = Number(q | sticky) * 2 ** -scale
  return neg ? -r : r
}

class Fraction extends Base {
  readonly cls = 'Fraction'
  pickled() { return this }
  n: bigint
  d: bigint
  kind: Value
  constructor(n: bigint, d: bigint, kind: Value) {
    super()
    const g = bigGcd(n, d) || 1n
    const sign = d < 0n ? -1n : 1n
    this.n = (sign * n) / g
    this.d = (sign * d) / g
    this.kind = kind
  }
  repr() { return `Fraction(${this.n}, ${this.d})` }
  str() { return this.d === 1n ? String(this.n) : `${this.n}/${this.d}` }
  v() { return { k: 'obj', o: this } as Value }
  make(n: bigint, d: bigint) { return new Fraction(n, d, this.kind).v() }
  typeValue() { return this.kind }
  isa(name: string) { return name === 'fractions.Fraction' }
  asNumber() { return ratioToFloat(this.n, this.d) }
  truthy() { return this.n !== 0n }
  hashKey() {
    // equal to a number means the same dict entry
    const x = ratioToFloat(this.n, this.d)
    const [a, b] = Number.isFinite(x) ? floatRatio(x) : [0n, 0n]
    return a === this.n && b === this.d ? `n${x}` : `f${this.n}/${this.d}`
  }
  format(spec: string) {
    if (spec) throw new PyError('TypeError', 'unsupported format string passed to Fraction.__format__')
    return this.str()
  }
  unary(op: '-' | '+' | '~', h: Host): Value {
    if (op === '~') throw h.err('TypeError', "bad operand type for unary ~: 'Fraction'")
    return op === '-' ? this.make(-this.n, this.d) : this.v()
  }
  arith(op: string, a: Fraction, b: Fraction, h: Host): Value | undefined {
    switch (op) {
      case '+': return this.make(a.n * b.d + b.n * a.d, a.d * b.d)
      case '-': return this.make(a.n * b.d - b.n * a.d, a.d * b.d)
      case '*': return this.make(a.n * b.n, a.d * b.d)
      case '/':
        if (!b.n) throw h.raise(h.excClass('ZeroDivisionError'), [S(`Fraction(${a.n * b.d}, 0)`)])
        return this.make(a.n * b.d, a.d * b.n)
      case '//':
        if (!b.n) throw h.err('ZeroDivisionError', 'integer division or modulo by zero')
        return int(Number(floorDiv(a.n * b.d, a.d * b.n)))
      case '%': {
        if (!b.n) throw h.err('ZeroDivisionError', 'integer division or modulo by zero')
        const [x, y] = [a.n * b.d, a.d * b.n]
        const m = x - floorDiv(x, y) * y
        return this.make(m, a.d * b.d)
      }
      case '**':
        if (b.d === 1n) {
          const e = b.n
          if (e >= 0n) return this.make(a.n ** e, a.d ** e)
          if (a.n > 0n) return this.make(a.d ** -e, a.n ** -e)
          if (a.n === 0n) throw h.raise(h.excClass('ZeroDivisionError'), [S(`Fraction(${a.d ** -e}, 0)`)])
          return this.make((-a.d) ** -e, (-a.n) ** -e)
        }
        return h.binop('**', float(a.asNumber()), float(b.asNumber()))
    }
    return undefined
  }
  binop(op: string, other: Value, reflected: boolean, h: Host): Value | undefined {
    const compare = ['==', '!=', '<', '>', '<=', '>='].includes(op)
    // 2 ** Fraction(3) stays an int
    if (op === '**' && reflected && this.d === 1n && this.n >= 0n && isNum(other)) return h.binop('**', other, int(Number(this.n)))
    let o: Fraction | null = null
    if (other.k === 'obj' && other.o instanceof Fraction) o = other.o
    else if (other.k === 'int' || other.k === 'bool') o = new Fraction(BigInt(Math.trunc(numOf(other))), 1n, this.kind)
    else if (other.k === 'float') {
      if (!compare) return reflected ? h.binop(op, other, float(this.asNumber())) : h.binop(op, float(this.asNumber()), other)
      if (!Number.isFinite(other.v)) return reflected ? h.compare(op, other, float(0)) : h.compare(op, float(0), other)
      const [n, d] = floatRatio(other.v)
      o = new Fraction(n, d, this.kind)
    }
    if (!o) return compare ? (op === '==' ? bool(false) : op === '!=' ? bool(true) : undefined) : undefined
    const [a, b] = reflected ? [o, this] : [this, o]
    if (compare) {
      const c = a.n * b.d - b.n * a.d
      return bool(op === '==' ? c === 0n : op === '!=' ? c !== 0n : op === '<' ? c < 0n : op === '>' ? c > 0n : op === '<=' ? c <= 0n : c >= 0n)
    }
    return this.arith(op, a, b, h)
  }
  limit(max: bigint, h: Host): Value {
    if (max < 1n) throw h.err('ValueError', 'max_denominator should be at least 1')
    if (this.d <= max) return this.v()
    let [p0, q0, p1, q1] = [0n, 1n, 1n, 0n]
    let [n, d] = [this.n, this.d]
    for (;;) {
      const a = floorDiv(n, d)
      const q2 = q0 + a * q1
      if (q2 > max) break
      ;[p0, q0, p1, q1] = [p1, q1, p0 + a * p1, q2]
      ;[n, d] = [d, n - a * d]
    }
    const k = floorDiv(max - q0, q1)
    const bound1 = new Fraction(p0 + k * p1, q0 + k * q1, this.kind)
    const bound2 = new Fraction(p1, q1, this.kind)
    const dist = (x: Fraction) => {
      const top = x.n * this.d - this.n * x.d
      return new Fraction(top < 0n ? -top : top, x.d * this.d, this.kind)
    }
    const [da, db] = [dist(bound2), dist(bound1)]
    return da.n * db.d <= db.n * da.d ? bound2.v() : bound1.v()
  }
  round(digits: Value | undefined, h: Host): Value {
    const halfEven = (n: bigint, d: bigint) => {
      const fl = floorDiv(n, d)
      const rem = n - fl * d
      const twice = rem * 2n
      return twice < d ? fl : twice > d ? fl + 1n : fl % 2n === 0n ? fl : fl + 1n
    }
    if (!digits || digits.k === 'none') return int(Number(halfEven(this.n, this.d)))
    const nd = h.toInt(digits)
    const shift = 10n ** BigInt(Math.abs(nd))
    return nd > 0 ? this.make(halfEven(this.n * shift, this.d), shift) : this.make(halfEven(this.n, this.d * shift) * shift, 1n)
  }
  getAttr(name: string, h: Host): Value | undefined {
    const m = (call: (a: Value[], kw: Kw) => Value) => F(name, call)
    switch (name) {
      case 'numerator': return int(Number(this.n))
      case 'denominator': return int(Number(this.d))
      case 'real': return this.v()
      case 'imag': return int(0)
      case 'conjugate': return m(() => this.v())
      case 'limit_denominator': return m((a, kw) => {
        const lim = kw.max_denominator ?? a[0]
        return this.limit(lim ? BigInt(h.toInt(lim)) : 1000000n, h)
      })
      case 'as_integer_ratio': return m(() => tuple([int(Number(this.n)), int(Number(this.d))]))
      case '__abs__': return m(() => this.make(this.n < 0n ? -this.n : this.n, this.d))
      case '__round__': return m((a) => this.round(a[0], h))
      case '__floor__': return m(() => int(Number(floorDiv(this.n, this.d))))
      case '__ceil__': return m(() => int(Number(-floorDiv(-this.n, this.d))))
      case '__trunc__': return m(() => int(Number(this.n / this.d)))
    }
    return undefined
  }
}

const makeFractions = (h: Host): Value => {
  const LITERAL = /^\s*(?<sign>[-+]?)(?=\d|\.\d)(?<num>\d*)(?:(?:\/(?<denom>\d+))?|(?:\.(?<decimal>\d*))?(?:[eE](?<exp>[-+]?\d+))?)\s*$/
  const kind: Value & { k: 'type' } = { k: 'type', name: 'fractions.Fraction', call: () => NONE, attrs: new Map() }
  const fromFloat = (x: number): Fraction => {
    if (Number.isNaN(x)) throw h.err('ValueError', 'cannot convert NaN to integer ratio')
    if (!Number.isFinite(x)) throw h.err('OverflowError', 'cannot convert Infinity to integer ratio')
    const [n, d] = floatRatio(x)
    return new Fraction(n, d, kind)
  }
  const rational = (v: Value): Fraction | null => {
    if (v.k === 'obj' && v.o instanceof Fraction) return v.o
    if (v.k === 'int' || v.k === 'bool') return new Fraction(BigInt(numOf(v)), 1n, kind)
    return null
  }
  kind.call = (a, kw) => {
    for (const k in kw) if (k !== 'numerator' && k !== 'denominator') throw h.err('TypeError', `__new__() got an unexpected keyword argument '${k}'`)
    const num = kw.numerator ?? a[0]
    const den = kw.denominator ?? a[1]
    if (!num) return new Fraction(0n, 1n, kind).v()
    if (den && den.k !== 'none') {
      const [x, y] = [rational(num), rational(den)]
      if (!x || !y) throw h.err('TypeError', 'both arguments should be Rational instances')
      if (!y.n) throw h.raise(h.excClass('ZeroDivisionError'), [S(`Fraction(${x.n * y.d}, 0)`)])
      return new Fraction(x.n * y.d, x.d * y.n, kind).v()
    }
    const direct = rational(num)
    if (direct) return direct.v()
    if (num.k === 'float') return fromFloat(num.v).v()
    if (num.k !== 'str') throw h.err('TypeError', 'argument should be a string or a Rational instance')
    const m = LITERAL.exec(num.v)
    if (!m?.groups) throw h.err('ValueError', `Invalid literal for Fraction: ${repr(num)}`)
    let n = BigInt(m.groups.num || '0')
    let d = 1n
    if (m.groups.denom) {
      d = BigInt(m.groups.denom)
      if (!d) throw h.raise(h.excClass('ZeroDivisionError'), [S(`Fraction(${n}, 0)`)])
    } else {
      if (m.groups.decimal) {
        const dec = m.groups.decimal
        const scale = 10n ** BigInt(dec.length)
        n = n * scale + BigInt(dec || '0')
        d *= scale
      }
      if (m.groups.exp) {
        const e = BigInt(m.groups.exp)
        if (e >= 0n) n *= 10n ** e
        else d *= 10n ** -e
      }
    }
    return new Fraction(m.groups.sign === '-' ? -n : n, d, kind).v()
  }
  kind.attrs!.set('from_float', F('from_float', (a) => {
    const direct = a[0].k === 'int' || a[0].k === 'bool' ? rational(a[0]) : null
    if (direct) return direct.v()
    if (a[0].k !== 'float') throw h.err('TypeError', `Fraction.from_float() only takes floats, not ${repr(a[0])} (${typeName(a[0])})`)
    return fromFloat(a[0].v).v()
  }))
  kind.attrs!.set('from_decimal', F('from_decimal', () => { throw h.err('NotImplementedError', 'Fraction.from_decimal is not available in this sandbox: it has no decimal module') }))
  return mod('fractions', { Fraction: kind })
}

// ---------------------------------------------------------------- uuid, secrets, platform, getpass

class Uuid extends Base {
  readonly cls = 'UUID'
  hex: string
  kind: Value
  constructor(hex: string, kind: Value) {
    super()
    this.hex = hex
    this.kind = kind
  }
  str() { return `${this.hex.slice(0, 8)}-${this.hex.slice(8, 12)}-${this.hex.slice(12, 16)}-${this.hex.slice(16, 20)}-${this.hex.slice(20)}` }
  repr() { return `UUID(${repr(S(this.str()))})` }
  format(spec: string) {
    if (spec) throw new PyError('TypeError', 'unsupported format string passed to UUID.__format__')
    return this.str()
  }
  typeValue() { return this.kind }
  isa(name: string) { return name === 'uuid.UUID' }
  hashKey() { return `u${this.hex}` }
  binop(op: string, other: Value): Value | undefined {
    if (other.k !== 'obj' || !(other.o instanceof Uuid)) return undefined
    const [a, b] = [this.hex, other.o.hex]
    switch (op) {
      case '==': return bool(a === b)
      case '!=': return bool(a !== b)
      case '<': return bool(a < b)
      case '>': return bool(a > b)
      case '<=': return bool(a <= b)
      case '>=': return bool(a >= b)
    }
    return undefined
  }
  getAttr(name: string): Value | undefined {
    switch (name) {
      case 'hex': return S(this.hex)
      case 'version': return ((parseInt(this.hex[16], 16) & 0xc) === 8) ? int(parseInt(this.hex[12], 16)) : NONE
      case 'variant': return S((parseInt(this.hex[16], 16) & 0xc) === 8 ? 'specified in RFC 4122' : 'reserved for NCS compatibility')
    }
    return undefined
  }
}

const hexChars = (n: number) => randomChars('0123456789abcdef', n)

const makeUuid = (h: Host): Value => {
  const kind: Value & { k: 'type' } = { k: 'type', name: 'uuid.UUID', call: () => NONE }
  kind.call = (a, kw) => {
    refuse(h, 'uuid.UUID', kw, ['bytes', 'bytes_le', 'fields', 'int', 'version'])
    const given = kw.hex ?? a[0]
    if (!given || given.k === 'none') throw h.err('TypeError', 'one of the hex, bytes, bytes_le, fields, or int arguments must be given')
    const hex = h.str(given).replace(/urn:|uuid:/g, '').replace(/^[{}]+|[{}]+$/g, '').replace(/-/g, '')
    if (hex.length !== 32) throw h.err('ValueError', 'badly formed hexadecimal UUID string')
    if (!/^[0-9a-fA-F]{32}$/.test(hex)) throw h.err('ValueError', `invalid literal for int() with base 16: ${repr(S(hex))}`)
    return { k: 'obj', o: new Uuid(hex.toLowerCase(), kind) }
  }
  return mod('uuid', {
    UUID: kind,
    uuid4: F('uuid4', () => ({ k: 'obj', o: new Uuid(`${hexChars(12)}4${hexChars(3)}${'89ab'[Math.floor(Math.random() * 4)]}${hexChars(15)}`, kind) })),
  })
}

const makeSecrets = (h: Host): Value => {
  const sizeOf = (v: Value | undefined) => (v && v.k !== 'none' ? h.toInt(v) : 32)
  return takingKeywords(mod('secrets', {
    token_hex: F('token_hex', (a, kw) => S(hexChars(2 * sizeOf(kw.nbytes ?? a[0])))),
    token_urlsafe: F('token_urlsafe', (a, kw) => {
      const n = sizeOf(kw.nbytes ?? a[0])
      return S(randomChars('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_', Math.ceil((n * 4) / 3)))
    }),
    randbelow: F('randbelow', (a, kw) => {
      const bound = kw.exclusive_upper_bound ?? a[0]
      if (bound.k !== 'int' && bound.k !== 'bool') throw h.err('AttributeError', `'${typeName(bound)}' object has no attribute 'bit_length'`)
      const n = numOf(bound)
      if (n <= 0) throw h.err('ValueError', 'Upper bound must be positive.')
      return int(Math.floor(Math.random() * n))
    }),
    randbits: F('randbits', (a, kw) => {
      const k = h.toInt(kw.k ?? a[0])
      if (k < 0) throw h.err('ValueError', 'number of bits must be non-negative')
      if (k > 53) throw h.err('NotImplementedError', 'secrets.randbits above 53 bits is not available in this sandbox: its integers are doubles')
      return int(Math.floor(Math.random() * 2 ** k))
    }),
    choice: F('choice', (a, kw) => {
      const seq = kw.seq ?? a[0]
      const items = h.iterate(seq)
      if (!items.length) throw h.err('IndexError', `${seq.k === 'str' ? 'string' : typeName(seq)} index out of range`)
      return items[Math.floor(Math.random() * items.length)]
    }),
    compare_digest: F('compare_digest', (a) => {
      if (a[0]?.k !== 'str' || a[1]?.k !== 'str') throw h.err('TypeError', `unsupported operand types(s) or combination of types: '${typeName(a[0] ?? NONE)}' and '${typeName(a[1] ?? NONE)}'`)
      return bool(a[0].v === a[1].v)
    }),
  }), { token_hex: ['nbytes'], token_urlsafe: ['nbytes'], randbelow: ['exclusive_upper_bound'], randbits: ['k'], choice: ['seq'], compare_digest: [] })
}

const makePlatform = (h: Host): Value => {
  const version = () => h.iterate(h.getAttr(h.module('sys'), 'version_info')).map((x) => h.str(x)).join('.')
  const text = (name: string, value: string): Value => F(name, () => S(value))
  return mod('platform', {
    python_version: F('python_version', () => S(version())),
    python_version_tuple: F('python_version_tuple', () => tuple(version().split('.').map(S))),
    python_implementation: text('python_implementation', 'CPython'),
    system: text('system', 'Linux'),
    machine: text('machine', 'x86_64'),
    processor: text('processor', 'x86_64'),
    node: text('node', 'sandbox'),
    release: text('release', ''),
    version: text('version', ''),
    platform: text('platform', 'Linux-sandbox-x86_64'),
    architecture: F('architecture', () => tuple([S('64bit'), S('ELF')])),
  })
}

const makeGetpass = (h: Host): Value => mod('getpass', {
  getuser: F('getuser', () => S('sandbox')),
  getpass: F('getpass', () => { throw h.err('EOFError', 'this sandbox has no keyboard input: set the value in the code instead') }),
})

const build = (h: Host): Record<string, Value> => ({
  statistics: makeStatistics(h),
  datetime: makeDatetime(h),
  csv: makeCsv(h),
  argparse: makeArgparse(h),
  pathlib: makePathlib(h),
  enum: makeEnum(h),
  contextlib: makeContextlib(h),
  textwrap: makeTextwrap(h),
  types: makeTypes(h),
  tempfile: makeTempfile(h),
  glob: makeGlob(h),
  shutil: makeShutil(h),
  fnmatch: makeFnmatch(h),
  inspect: makeInspect(h),
  fractions: makeFractions(h),
  uuid: makeUuid(h),
  secrets: makeSecrets(h),
  platform: makePlatform(h),
  getpass: makeGetpass(h),
})

return build
}

let load: ReturnType<typeof define> | undefined

export const STDLIB_MORE: PyLib = {
  modules: RARE_STDLIB,
  load: (h) => (load ??= define())(h),
}
