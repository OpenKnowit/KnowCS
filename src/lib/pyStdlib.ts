// The parts of Python's standard library a course's code reaches for, built on the sandbox's own values.
// Nothing here touches files, the network or the clock beyond reading the time.
import { PyError } from './ndarray'
import { NOT_IMPLEMENTED, PyDict, PyObj, formatSpec, isNum, makeSet, numOf, py, repr, setItems, str, typeName } from './minipy'
import type { ClassMeta, ClassValue, Fn, Host, HostClass, InstValue, Kw, PyLib, Value } from './minipy'

const SANDBOX_CWD = '/sandbox'

/** a path in the run's in-memory folder, in the one spelling it is stored under */
export const fullPath = (p: string) => {
  // the empty path names nothing: os.path.exists('') is False
  if (!p) return '/\0'
  const parts: string[] = []
  for (const part of (p.startsWith('/') ? p : `${SANDBOX_CWD}/${p}`).split('/')) {
    if (part === '..') parts.pop()
    else if (part && part !== '.') parts.push(part)
  }
  return `/${parts.join('/')}`
}

// ---------------------------------------------------------------- numeric helpers (also used by math)

export const gcdOf = (a: number, b: number): number => {
  let x = Math.abs(a)
  let y = Math.abs(b)
  while (y) [x, y] = [y, x % y]
  return x
}

/** Abramowitz & Stegun via the continued fraction; good to about 1e-14 */
export const erf = (x: number): number => {
  if (Number.isNaN(x)) return NaN
  const a = Math.abs(x)
  if (a > 6) return Math.sign(x)
  let r: number
  if (a < 2.5) {
    // the Maclaurin series converges quickly here
    let term = a
    let sum = a
    for (let n = 1; n < 200 && Math.abs(term) > 1e-17 * Math.abs(sum); n++) {
      term *= (-a * a) / n
      sum += term / (2 * n + 1)
    }
    r = (2 / Math.sqrt(Math.PI)) * sum
  } else {
    let f = 0
    for (let k = 60; k > 0; k--) f = (k / 2) / (a + f)
    r = 1 - Math.exp(-a * a) / ((a + f) * Math.sqrt(Math.PI))
  }
  return x < 0 ? -r : r
}

const LANCZOS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
export const gammaFn = (x: number): number => {
  if (Number.isInteger(x)) {
    if (x <= 0) return NaN
    if (x < 171) {
      let r = 1
      for (let i = 2; i < x; i++) r *= i
      return r
    }
  }
  if (x < 0.5) return Math.PI / (Math.sin(Math.PI * x) * gammaFn(1 - x))
  const z = x - 1
  let a = LANCZOS[0]
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (z + i)
  const t = z + 7.5
  return Math.sqrt(2 * Math.PI) * t ** (z + 0.5) * Math.exp(-t) * a
}

// minipy.ts imports this file and this file extends minipy's classes, so nothing below may run until the
// first import statement of a program: by then both modules have finished loading.
const define = () => {
const { NONE, int, float, bool, tuple, str: S, list: L } = py
// (a local name: vitest's module transform loses an imported base class inside a function)
const Base = PyObj

// ---------------------------------------------------------------- small objects

/** a library function that read an argument which is not there, or is not what it should be, raises Python's TypeError */
const guarded = (name: string, call: (args: Value[], kw: Kw) => Value) => (args: Value[], kw: Kw): Value => {
  try {
    return call(args, kw)
  } catch (e) {
    if (e instanceof TypeError) throw new PyError('TypeError', `${name}() was given a missing or wrong argument`)
    throw e
  }
}
const F = (name: string, call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call: guarded(name, call) })
const iter = (name: string, next: () => Value | undefined): Value => ({ k: 'iter', name, next })
const fromGen = (name: string, g: Generator<Value, void, undefined>): Value => iter(name, () => {
  const r = g.next()
  return r.done ? undefined : r.value
})
const mod = (name: string, attrs: Record<string, Value>): Value => ({ k: 'module', name, attrs })
const subclassOf = (h: Host, name: string, base: string): ClassValue => {
  const b = h.excClass(base)
  return { k: 'class', name, bases: [b], ns: new Map(), host: b.host, builtin: true }
}

// ---------------------------------------------------------------- re

class Match extends Base {
  readonly cls = 're.Match'
  m: RegExpExecArray
  constructor(m: RegExpExecArray) {
    super()
    this.m = m
  }
  repr() { return `<re.Match object; span=(${this.m.index}, ${this.m.index + this.m[0].length}), match=${repr(S(this.m[0]))}>` }
  group(h: Host, v: Value | undefined): Value {
    if (!v) return S(this.m[0])
    if (v.k === 'str') {
      const g = this.m.groups?.[v.v]
      if (g === undefined && !(this.m.groups && v.v in this.m.groups)) throw h.err('IndexError', 'no such group')
      return g === undefined ? NONE : S(g)
    }
    const i = h.toInt(v)
    if (i < 0 || i >= this.m.length) throw h.err('IndexError', 'no such group')
    return this.m[i] === undefined ? NONE : S(this.m[i])
  }
  getItem(idx: Value, h: Host) { return this.group(h, idx) }
  getAttr(name: string, h: Host): Value | undefined {
    const m = this.m
    switch (name) {
      case 'group': return F(name, (args) => (args.length > 1 ? tuple(args.map((a) => this.group(h, a))) : this.group(h, args[0])))
      case 'groups': return F(name, (args) => tuple(m.slice(1).map((g) => (g === undefined ? args[0] ?? NONE : S(g)))))
      case 'groupdict': return F(name, () => ({ k: 'dict', d: PyDict.from(Object.entries(m.groups ?? {}).map(([a, b]) => [S(a), b === undefined ? NONE : S(b)])) }))
      case 'start': return F(name, () => int(m.index))
      case 'end': return F(name, () => int(m.index + m[0].length))
      case 'span': return F(name, () => tuple([int(m.index), int(m.index + m[0].length)]))
      case 'string': return S(m.input)
    }
    return undefined
  }
}

const I = 2
const M = 8
const DOTALL = 16
const VERBOSE = 64

const translateRegex = (p: string, flags: number): string => {
  let src = p.replace(/\(\?P<(\w+)>/g, '(?<$1>').replace(/\(\?P=(\w+)\)/g, '\\k<$1>').replace(/\\A/g, '^').replace(/\\Z/g, '$')
  if (flags & VERBOSE) src = src.split('\\#').map((part) => part.replace(/#[^\n]*/g, '').replace(/\s+/g, '')).join('\\#')
  return src
}
const jsFlags = (flags: number, extra: string) => `${extra}${flags & I ? 'i' : ''}${flags & M ? 'm' : ''}${flags & DOTALL ? 's' : ''}u`

class Pattern extends Base {
  readonly cls = 're.Pattern'
  source: string
  flags: number
  err: Host
  constructor(source: string, flags: number, h: Host) {
    super()
    this.source = source
    this.flags = flags
    this.err = h
    this.rx('g')
  }
  repr() { return `re.compile(${repr(S(this.source))})` }
  rx(extra: string, wrap?: (s: string) => string): RegExp {
    const s = translateRegex(this.source, this.flags)
    try {
      try {
        return new RegExp(wrap ? wrap(s) : s, jsFlags(this.flags, extra))
      } catch {
        // the u flag is stricter about escapes than Python
        return new RegExp(wrap ? wrap(s) : s, jsFlags(this.flags, extra).replace('u', ''))
      }
    } catch (e) {
      throw this.err.err('ValueError', `bad regular expression: ${(e as Error).message}`)
    }
  }
  match(s: string) { const m = this.rx('y').exec(s); return m ? new Match(m) : null }
  search(s: string) { const m = this.rx('').exec(s); return m ? new Match(m) : null }
  fullmatch(s: string) { const m = this.rx('y', (x) => `(?:${x})$`).exec(s); return m && m[0].length === s.length ? new Match(m) : null }
  all(s: string) { return [...s.matchAll(this.rx('g'))] }
  findall(s: string): Value {
    return L(this.all(s).map((m) => {
      if (m.length === 1) return S(m[0])
      const gs = m.slice(1).map((g) => S(g ?? ''))
      return gs.length === 1 ? gs[0] : tuple(gs)
    }))
  }
  sub(repl: Value, s: string, count: number, h: Host): [string, number] {
    let n = 0
    const out = s.replace(this.rx('g'), (...parts) => {
      if (count && n >= count) return parts[0]
      n++
      const named = typeof parts[parts.length - 1] === 'object' ? (parts.pop() as Record<string, string>) : undefined
      const input = parts.pop() as string
      const index = parts.pop() as number
      const m = Object.assign([...parts], { index, input, groups: named }) as unknown as RegExpExecArray
      if (repl.k !== 'str') return h.str(h.call(repl, [{ k: 'obj', o: new Match(m) }]))
      return repl.v.replace(/\\(\d+)|\\g<(\w+)>|\\n|\\t|\\\\/g, (e, d?: string, g?: string) => {
        if (d) return m[Number(d)] ?? ''
        if (g) return (/^\d+$/.test(g) ? m[Number(g)] : named?.[g]) ?? ''
        return e === '\\n' ? '\n' : e === '\\t' ? '\t' : '\\'
      })
    })
    return [out, n]
  }
  split(s: string, max: number): Value {
    const out: Value[] = []
    let last = 0
    let n = 0
    for (const m of this.all(s)) {
      if (max && n >= max) break
      if (m[0] === '' && (m.index === 0 || m.index === s.length)) continue
      out.push(S(s.slice(last, m.index)))
      for (const g of m.slice(1)) out.push(g === undefined ? NONE : S(g))
      last = m.index + m[0].length
      n++
    }
    out.push(S(s.slice(last)))
    return L(out)
  }
  getAttr(name: string, h: Host): Value | undefined {
    const text = (v: Value) => {
      if (v?.k !== 'str') throw h.err('TypeError', 'expected string or bytes-like object')
      return v.v
    }
    const opt = (o: Match | null): Value => (o ? { k: 'obj', o } : NONE)
    switch (name) {
      case 'match': return F(name, (a) => opt(this.match(text(a[0]))))
      case 'search': return F(name, (a) => opt(this.search(text(a[0]))))
      case 'fullmatch': return F(name, (a) => opt(this.fullmatch(text(a[0]))))
      case 'findall': return F(name, (a) => this.findall(text(a[0])))
      case 'finditer': return F(name, (a) => {
        const ms = this.all(text(a[0]))
        let i = 0
        return iter('callable_iterator', () => (i < ms.length ? { k: 'obj', o: new Match(ms[i++]) } : undefined))
      })
      case 'sub': return F(name, (a, kw) => S(this.sub(a[0], text(a[1]), kw.count ? h.toInt(kw.count) : a[2] ? h.toInt(a[2]) : 0, h)[0]))
      case 'subn': return F(name, (a, kw) => {
        const [out, n] = this.sub(a[0], text(a[1]), kw.count ? h.toInt(kw.count) : a[2] ? h.toInt(a[2]) : 0, h)
        return tuple([S(out), int(n)])
      })
      case 'split': return F(name, (a, kw) => this.split(text(a[0]), kw.maxsplit ? h.toInt(kw.maxsplit) : a[1] ? h.toInt(a[1]) : 0))
      case 'pattern': return S(this.source)
    }
    return undefined
  }
}

const makeRe = (h: Host): Value => {
  const compile = (p: Value, flags?: Value): Pattern => {
    if (p.k === 'obj' && p.o instanceof Pattern) return p.o
    if (p.k !== 'str') throw h.err('TypeError', 'first argument must be string or compiled pattern')
    return new Pattern(p.v, flags ? h.toInt(flags) : 0, h)
  }
  /** re.search(pattern, s, flags) is pattern.search(s) */
  const direct = (name: string, flagAt: number): Value => F(name, (a, kw) => {
    const pat = compile(a[0], kw.flags ?? a[flagAt])
    return h.call(pat.getAttr(name, h)!, a.slice(1, flagAt), kw.flags ? { ...kw, flags: undefined as never } : kw)
  })
  return mod('re', {
    compile: F('compile', (a, kw) => ({ k: 'obj', o: compile(a[0], kw.flags ?? a[1]) })),
    match: direct('match', 2), search: direct('search', 2), fullmatch: direct('fullmatch', 2), findall: direct('findall', 2), finditer: direct('finditer', 2),
    sub: F('sub', (a, kw) => {
      const pat = compile(a[0], kw.flags ?? a[4])
      return S(pat.sub(a[1], (a[2] as { v: string }).v, kw.count ? h.toInt(kw.count) : a[3] ? h.toInt(a[3]) : 0, h)[0])
    }),
    subn: F('subn', (a, kw) => {
      const [out, n] = compile(a[0], kw.flags ?? a[4]).sub(a[1], (a[2] as { v: string }).v, kw.count ? h.toInt(kw.count) : a[3] ? h.toInt(a[3]) : 0, h)
      return tuple([S(out), int(n)])
    }),
    split: F('split', (a, kw) => compile(a[0], kw.flags ?? a[3]).split((a[1] as { v: string }).v, kw.maxsplit ? h.toInt(kw.maxsplit) : a[2] ? h.toInt(a[2]) : 0)),
    escape: F('escape', (a) => S((a[0] as { v: string }).v.replace(/[^\w]/g, '\\$&'))),
    I: int(I), IGNORECASE: int(I), M: int(M), MULTILINE: int(M), S: int(DOTALL), DOTALL: int(DOTALL), X: int(VERBOSE), VERBOSE: int(VERBOSE),
    error: subclassOf(h, 'error', 'Exception'),
  })
}

// ---------------------------------------------------------------- json

// eslint-disable-next-line no-control-regex -- JSON escapes exactly these
const jsonStr = (s: string) => `"${s.replace(/[\\"\u0000-\u001f\u007f-￿]/g, (c) => {
  const short: Record<string, string> = { '"': '\\"', '\\': '\\\\', '\n': '\\n', '\r': '\\r', '\t': '\\t', '\b': '\\b', '\f': '\\f' }
  return short[c] ?? `\\u${c.charCodeAt(0).toString(16).padStart(4, '0')}`
})}"`

const makeJson = (h: Host): Value => {
  const decodeError = subclassOf(h, 'JSONDecodeError', 'ValueError')
  const dump = (v: Value, indent: string | null, sortKeys: boolean, seps: [string, string], fallback: Value | undefined, depth: number): string => {
    const nl = (d: number) => (indent === null ? '' : `\n${indent.repeat(d)}`)
    const join = (items: string[], open: string, close: string) => {
      if (!items.length) return open + close
      return `${open}${nl(depth + 1)}${items.join(seps[0] + nl(depth + 1))}${nl(depth)}${close}`
    }
    switch (v.k) {
      case 'none': return 'null'
      case 'bool': return v.v ? 'true' : 'false'
      case 'int': return String(v.v)
      case 'float': return Number.isNaN(v.v) ? 'NaN' : v.v === Infinity ? 'Infinity' : v.v === -Infinity ? '-Infinity' : repr(v)
      case 'str': return jsonStr(v.v)
      case 'list': case 'tuple': return join(v.items.map((x) => dump(x, indent, sortKeys, seps, fallback, depth + 1)), '[', ']')
      case 'dict': {
        const keyOf = (k0: Value) => {
          const k = (k0.k === 'inst' ? k0.cls.meta?.scalar?.(k0) : undefined) ?? k0
          if (k.k === 'str') return k.v
          if (k.k === 'none') return 'null'
          if (k.k === 'bool') return k.v ? 'true' : 'false'
          if (isNum(k)) return repr(k)
          throw h.err('TypeError', `keys must be str, int, float, bool or None, not ${typeName(k)}`)
        }
        const items = v.d.items().map(([k, x]) => [keyOf(k), x] as const)
        if (sortKeys) items.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
        return join(items.map(([k, x]) => `${jsonStr(k)}${seps[1]}${dump(x, indent, sortKeys, seps, fallback, depth + 1)}`), '{', '}')
      }
    }
    // class Level(int, Enum): json sees the int
    const same = v.k === 'inst' ? v.cls.meta?.scalar?.(v) : undefined
    if (same) return dump(same, indent, sortKeys, seps, fallback, depth)
    if (fallback && fallback.k !== 'none') return dump(h.call(fallback, [v]), indent, sortKeys, seps, fallback, depth)
    throw h.err('TypeError', `Object of type ${typeName(v)} is not JSON serializable`)
  }
  const load = (x: unknown): Value => {
    if (x === null) return NONE
    if (typeof x === 'boolean') return bool(x)
    if (typeof x === 'number') return Number.isInteger(x) ? int(x) : float(x)
    if (typeof x === 'string') return S(x)
    if (Array.isArray(x)) return L(x.map(load))
    return { k: 'dict', d: PyDict.from(Object.entries(x as object).map(([a, b]) => [S(a), load(b)])) }
  }
  const out: Value = mod('json', {
    dumps: F('dumps', (a, kw) => {
      const ind = kw.indent
      const indent = !ind || ind.k === 'none' ? null : ind.k === 'str' ? ind.v : ' '.repeat(h.toInt(ind))
      const sep = kw.separators && kw.separators.k !== 'none' ? h.iterate(kw.separators).map((x) => (x as { v: string }).v) : null
      return S(dump(a[0], indent, !!kw.sort_keys && h.truthy(kw.sort_keys), [sep?.[0] ?? (indent === null ? ', ' : ','), sep?.[1] ?? ': '], kw.default, 0))
    }),
    loads: F('loads', (a) => {
      try {
        return load(JSON.parse((a[0] as { v: string }).v))
      } catch (e) {
        if (e instanceof SyntaxError) return h.raise(decodeError, [S(e.message)])
        throw e
      }
    }),
    dump: F('dump', (a, kw) => {
      const f = a[1]
      const dumps = (out as { attrs: Record<string, Value> }).attrs.dumps
      h.call(h.getAttr(f, 'write'), [h.call(dumps, [a[0]], kw)])
      return NONE
    }),
    load: F('load', (a) => h.call((out as { attrs: Record<string, Value> }).attrs.loads, [h.call(h.getAttr(a[0], 'read'), [])])),
    JSONDecodeError: decodeError,
  })
  return out
}

// ---------------------------------------------------------------- collections

class Deque extends Base {
  readonly cls = 'deque'
  items: Value[]
  maxlen: number | null
  constructor(items: Value[], maxlen: number | null) {
    super()
    this.maxlen = maxlen
    this.items = maxlen === null ? items : items.slice(Math.max(0, items.length - maxlen))
  }
  repr() { return `deque([${this.items.map(repr).join(', ')}]${this.maxlen === null ? '' : `, maxlen=${this.maxlen}`})` }
  iter() { return [...this.items] }
  len() { return this.items.length }
  truthy() { return this.items.length > 0 }
  contains(v: Value, h: Host) { return this.items.some((x) => h.compare('==', x, v) && h.truthy(h.compare('==', x, v))) }
  at(i: number, h: Host) {
    const n = this.items.length
    if (i < -n || i >= n) throw h.err('IndexError', 'deque index out of range')
    return i < 0 ? i + n : i
  }
  getItem(idx: Value, h: Host) { return this.items[this.at(h.toInt(idx), h)] }
  setItem(idx: Value, v: Value, h: Host) { this.items[this.at(h.toInt(idx), h)] = v }
  getAttr(name: string, h: Host): Value | undefined {
    const it = this.items
    const empty = () => { if (!it.length) throw h.err('IndexError', 'pop from an empty deque') }
    const trimRight = () => { if (this.maxlen !== null && it.length > this.maxlen) it.pop() }
    const trimLeft = () => { if (this.maxlen !== null && it.length > this.maxlen) it.shift() }
    switch (name) {
      case 'append': return F(name, (a) => { it.push(a[0]); trimLeft(); return NONE })
      case 'appendleft': return F(name, (a) => { it.unshift(a[0]); trimRight(); return NONE })
      case 'pop': return F(name, () => { empty(); return it.pop()! })
      case 'popleft': return F(name, () => { empty(); return it.shift()! })
      case 'extend': return F(name, (a) => { for (const x of h.iterate(a[0])) { it.push(x); trimLeft() } return NONE })
      case 'extendleft': return F(name, (a) => { for (const x of h.iterate(a[0])) { it.unshift(x); trimRight() } return NONE })
      case 'clear': return F(name, () => { it.length = 0; return NONE })
      case 'rotate': return F(name, (a) => {
        const n = it.length
        if (!n) return NONE
        const k = ((a[0] ? h.toInt(a[0]) : 1) % n + n) % n
        it.unshift(...it.splice(n - k, k))
        return NONE
      })
      case 'reverse': return F(name, () => { it.reverse(); return NONE })
      case 'copy': return F(name, () => ({ k: 'obj', o: new Deque([...it], this.maxlen) }))
      case 'count': return F(name, (a) => int(it.filter((x) => h.truthy(h.compare('==', x, a[0]))).length))
      case 'index': return F(name, (a) => {
        const i = it.findIndex((x) => h.truthy(h.compare('==', x, a[0])))
        if (i < 0) throw h.err('ValueError', `${repr(a[0])} is not in deque`)
        return int(i)
      })
      case 'remove': return F(name, (a) => {
        const i = it.findIndex((x) => h.truthy(h.compare('==', x, a[0])))
        if (i < 0) throw h.err('ValueError', `${repr(a[0])} is not in deque`)
        it.splice(i, 1)
        return NONE
      })
      case 'maxlen': return this.maxlen === null ? NONE : int(this.maxlen)
    }
    return undefined
  }
}

const makeCollections = (h: Host): Value => {
  const ty = (name: string, call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'type', name: `collections.${name}`, call })
  return mod('collections', {
    Counter: ty('Counter', (a, kw) => {
      const d = new PyDict()
      const add = (k: Value, n: Value) => d.set(k, h.binop('+', d.get(k) ?? int(0), n))
      if (a[0]?.k === 'dict') for (const [k, n] of a[0].d.items()) add(k, n)
      else if (a[0]) for (const k of h.iterate(a[0])) add(k, int(1))
      for (const k in kw) add(S(k), kw[k])
      return { k: 'dict', d, flavor: 'Counter' }
    }),
    defaultdict: ty('defaultdict', (a, kw) => {
      const d = a[1]?.k === 'dict' ? PyDict.from(a[1].d.items()) : new PyDict()
      for (const k in kw) d.set(S(k), kw[k])
      return { k: 'dict', d, flavor: 'defaultdict', factory: a[0] ?? NONE }
    }),
    OrderedDict: ty('OrderedDict', (a, kw) => {
      const d = a[0]?.k === 'dict' ? PyDict.from(a[0].d.items()) : new PyDict()
      if (a[0] && a[0].k !== 'dict') for (const p of h.iterate(a[0])) { const [k, v] = h.iterate(p); d.set(k, v) }
      for (const k in kw) d.set(S(k), kw[k])
      return { k: 'dict', d, flavor: 'OrderedDict' }
    }),
    deque: ty('deque', (a, kw) => {
      const m = kw.maxlen ?? a[1]
      return { k: 'obj', o: new Deque(a[0] ? [...h.iterate(a[0])] : [], m && m.k !== 'none' ? h.toInt(m) : null) }
    }),
    namedtuple: F('namedtuple', (a, kw) => {
      const tname = (a[0] as { v: string }).v
      const names = a[1].k === 'str' ? a[1].v.replace(/,/g, ' ').trim().split(/\s+/) : h.iterate(a[1]).map((x) => (x as { v: string }).v)
      const defaults = kw.defaults && kw.defaults.k !== 'none' ? h.iterate(kw.defaults) : []
      const make = (items: Value[]): Value => ({ k: 'tuple', items, names, tname })
      return {
        k: 'type', name: tname,
        attrs: new Map<string, Value>([
          ['_fields', tuple(names.map(S))],
          ['_make', F('_make', (m) => make(h.iterate(m[0])))],
        ]),
        call: (args, ckw) => {
          if (args.length > names.length) throw h.err('TypeError', `${tname}.__new__() takes ${names.length + 1} positional arguments but ${args.length + 1} were given`)
          const items = names.map((n, i) => {
            const fromDefault = i - (names.length - defaults.length)
            const v = i < args.length ? args[i] : ckw[n] ?? (fromDefault >= 0 ? defaults[fromDefault] : undefined)
            if (!v) throw h.err('TypeError', `${tname}.__new__() missing required argument: '${n}'`)
            return v
          })
          return make(items)
        },
      }
    }),
  })
}

// ---------------------------------------------------------------- itertools

const makeItertools = (h: Host): Value => {
  const gen = (name: string, f: () => Generator<Value, void, undefined>): Value => fromGen(name, f())
  const each = function* (v: Value): Generator<Value, void, undefined> {
    const next = h.iterOf(v)
    for (let x = next(); x !== undefined; x = next()) yield x
  }
  const chain: Value & { k: 'fn' } = {
    k: 'fn', name: 'chain',
    call: (a) => gen('chain', function* () { for (const s of a) yield* each(s) }),
    attrs: new Map([['from_iterable', F('from_iterable', (a) => gen('chain', function* () { for (const s of each(a[0])) yield* each(s) }))]]),
  }
  const lt = (a: Value, b: Value) => h.truthy(h.compare('==', a, b))
  return mod('itertools', {
    chain,
    count: F('count', (a, kw) => {
      const start = kw.start ?? a[0] ?? int(0)
      const step = kw.step ?? a[1] ?? int(1)
      return gen('count', function* () {
        let x = start
        for (;;) {
          yield x
          x = h.binop('+', x, step)
        }
      })
    }),
    repeat: F('repeat', (a, kw) => {
      const times = kw.times ?? a[1]
      return gen('repeat', function* () {
        for (let i = 0; !times || i < h.toInt(times); i++) yield a[0]
      })
    }),
    cycle: F('cycle', (a) => gen('cycle', function* () {
      const saved: Value[] = []
      for (const x of each(a[0])) {
        saved.push(x)
        yield x
      }
      if (saved.length) for (;;) yield* saved
    })),
    islice: F('islice', (a) => gen('islice', function* () {
      const [start, stop, step] = a.length === 2 ? [0, a[1], 1] : [a[1].k === 'none' ? 0 : h.toInt(a[1]), a[2], a[3] && a[3].k !== 'none' ? h.toInt(a[3]) : 1]
      const end = stop.k === 'none' ? Infinity : h.toInt(stop)
      if (end <= start) return
      let i = 0
      for (const x of each(a[0])) {
        if (i >= start && (i - start) % step === 0) yield x
        i++
        if (i >= end) return
      }
    })),
    product: F('product', (a, kw) => gen('product', function* () {
      const pools = a.map((s) => [...h.iterate(s)])
      const repeat = kw.repeat ? h.toInt(kw.repeat) : 1
      const all = Array.from({ length: repeat }, () => pools).flat()
      const walk = function* (i: number, row: Value[]): Generator<Value, void, undefined> {
        if (i === all.length) {
          yield tuple(row)
          return
        }
        for (const x of all[i]) yield* walk(i + 1, [...row, x])
      }
      yield* walk(0, [])
    })),
    permutations: F('permutations', (a, kw) => gen('permutations', function* () {
      const pool = [...h.iterate(a[0])]
      const rv = kw.r ?? a[1]
      const r = rv && rv.k !== 'none' ? h.toInt(rv) : pool.length
      const walk = function* (row: number[]): Generator<Value, void, undefined> {
        if (row.length === r) {
          yield tuple(row.map((i) => pool[i]))
          return
        }
        for (let i = 0; i < pool.length; i++) if (!row.includes(i)) yield* walk([...row, i])
      }
      yield* walk([])
    })),
    combinations: F('combinations', (a) => gen('combinations', function* () {
      const pool = [...h.iterate(a[0])]
      const r = h.toInt(a[1])
      const walk = function* (from: number, row: Value[]): Generator<Value, void, undefined> {
        if (row.length === r) {
          yield tuple(row)
          return
        }
        for (let i = from; i < pool.length; i++) yield* walk(i + 1, [...row, pool[i]])
      }
      yield* walk(0, [])
    })),
    combinations_with_replacement: F('combinations_with_replacement', (a) => gen('combinations_with_replacement', function* () {
      const pool = [...h.iterate(a[0])]
      const r = h.toInt(a[1])
      const walk = function* (from: number, row: Value[]): Generator<Value, void, undefined> {
        if (row.length === r) {
          yield tuple(row)
          return
        }
        for (let i = from; i < pool.length; i++) yield* walk(i, [...row, pool[i]])
      }
      yield* walk(0, [])
    })),
    zip_longest: F('zip_longest', (a, kw) => gen('zip_longest', function* () {
      const nexts = a.map((s) => h.iterOf(s))
      const done = nexts.map(() => false)
      for (;;) {
        const row = nexts.map((next, i) => {
          if (done[i]) return kw.fillvalue ?? NONE
          const v = next()
          if (v === undefined) done[i] = true
          return v ?? kw.fillvalue ?? NONE
        })
        if (done.every(Boolean)) return
        yield tuple(row)
      }
    })),
    accumulate: F('accumulate', (a, kw) => gen('accumulate', function* () {
      const op = kw.func ?? a[1]
      let total: Value | undefined = kw.initial && kw.initial.k !== 'none' ? kw.initial : undefined
      if (total) yield total
      for (const x of each(a[0])) {
        const next: Value = total === undefined ? x : op && op.k !== 'none' ? h.call(op, [total, x]) : h.binop('+', total, x)
        total = next
        yield next
      }
    })),
    groupby: F('groupby', (a, kw) => gen('groupby', function* () {
      const key = kw.key ?? a[1]
      let group: Value[] = []
      let cur: Value | undefined
      for (const x of each(a[0])) {
        const k = key && key.k !== 'none' ? h.call(key, [x]) : x
        if (cur !== undefined && !lt(cur, k)) {
          yield tuple([cur, fromList(group)])
          group = []
        }
        cur = k
        group.push(x)
      }
      if (cur !== undefined) yield tuple([cur, fromList(group)])
    })),
    starmap: F('starmap', (a) => gen('starmap', function* () { for (const row of each(a[1])) yield h.call(a[0], h.iterate(row)) })),
    takewhile: F('takewhile', (a) => gen('takewhile', function* () {
      for (const x of each(a[1])) {
        if (!h.truthy(h.call(a[0], [x]))) return
        yield x
      }
    })),
    dropwhile: F('dropwhile', (a) => gen('dropwhile', function* () {
      let dropping = true
      for (const x of each(a[1])) {
        if (dropping && h.truthy(h.call(a[0], [x]))) continue
        dropping = false
        yield x
      }
    })),
    filterfalse: F('filterfalse', (a) => gen('filterfalse', function* () {
      for (const x of each(a[1])) if (!h.truthy(a[0].k === 'none' ? x : h.call(a[0], [x]))) yield x
    })),
    compress: F('compress', (a) => gen('compress', function* () {
      const flags = h.iterOf(a[1])
      for (const x of each(a[0])) {
        const f = flags()
        if (f === undefined) return
        if (h.truthy(f)) yield x
      }
    })),
    pairwise: F('pairwise', (a) => gen('pairwise', function* () {
      let prev: Value | undefined
      for (const x of each(a[0])) {
        if (prev !== undefined) yield tuple([prev, x])
        prev = x
      }
    })),
    tee: F('tee', (a) => {
      const items = h.iterate(a[0])
      return tuple(Array.from({ length: a[1] ? h.toInt(a[1]) : 2 }, () => fromList(items)))
    }),
  })
}
const fromList = (items: Value[]): Value => {
  let i = 0
  return iter('itertools._grouper', () => (i < items.length ? items[i++] : undefined))
}

// ---------------------------------------------------------------- functools, operator

class CmpKey extends Base {
  readonly cls = 'functools.KeyWrapper'
  v: Value
  cmp: Value
  constructor(v: Value, cmp: Value) {
    super()
    this.v = v
    this.cmp = cmp
  }
  repr() { return `<key ${repr(this.v)}>` }
  binop(op: string, other: Value, reflected: boolean, h: Host): Value | undefined {
    if (other.k !== 'obj' || !(other.o instanceof CmpKey)) return undefined
    const [a, b] = reflected ? [other.o.v, this.v] : [this.v, other.o.v]
    const c = h.num(h.call(this.cmp, [a, b]))
    return { '<': bool(c < 0), '>': bool(c > 0), '<=': bool(c <= 0), '>=': bool(c >= 0), '==': bool(c === 0), '!=': bool(c !== 0) }[op]
  }
}

const makeFunctools = (h: Host): Value => {
  const memo = (f: Value): Value => {
    const seen = new Map<string, Value>()
    const wrapper: Value & { k: 'fn' } = {
      k: 'fn', name: f.k === 'func' ? f.name : 'cached', bindable: true,
      call: (args, kw) => {
        const key = PyDict.hash(tuple([...args, ...Object.entries(kw).flatMap(([a, b]) => [S(a), b])]))
        let v = seen.get(key)
        if (!v) seen.set(key, (v = h.call(f, args, kw)))
        return v
      },
      attrs: new Map<string, Value>([['cache_clear', F('cache_clear', () => { seen.clear(); return NONE })], ['__wrapped__', f]]),
    }
    return wrapper
  }
  return mod('functools', {
    reduce: F('reduce', (a) => {
      const items = h.iterate(a[1])
      if (!items.length && !a[2]) throw h.err('TypeError', 'reduce() of empty iterable with no initial value')
      return (a[2] ? items : items.slice(1)).reduce((acc, x) => h.call(a[0], [acc, x]), a[2] ?? items[0])
    }),
    partial: F('partial', (a, kw) => ({ k: 'fn', name: 'partial', call: (args, ckw) => h.call(a[0], [...a.slice(1), ...args], { ...kw, ...ckw }) })),
    lru_cache: F('lru_cache', (a, kw) => {
      if (a[0] && (a[0].k === 'func' || a[0].k === 'bound')) return memo(a[0])
      void kw
      return F('decorator', (b) => memo(b[0]))
    }),
    cache: F('cache', (a) => memo(a[0])),
    cached_property: { k: 'type', name: 'cached_property', call: (a) => {
      const f = a[0]
      const name = f.k === 'func' ? f.name : 'value'
      return { k: 'descr', kind: 'prop', f: F(name, (args) => {
        const inst = args[0] as InstValue
        const key = `__cached_${name}`
        let v = inst.state[key] as Value | undefined
        if (!v) inst.state[key] = v = h.call(f, [inst])
        return v
      }) }
    } },
    wraps: F('wraps', (a) => F('decorator', (b) => {
      const src = a[0]
      const dst = b[0]
      if (src.k === 'func' && dst.k === 'func') {
        dst.name = src.name
        dst.doc = src.doc
      }
      if (dst.k === 'func' || dst.k === 'fn') (dst.attrs ??= new Map()).set('__wrapped__', src)
      return dst
    })),
    cmp_to_key: F('cmp_to_key', (a) => F('K', (b) => ({ k: 'obj', o: new CmpKey(b[0], a[0]) }))),
    total_ordering: F('total_ordering', (a) => {
      // fill in the comparisons a class left out, from __eq__ and whichever one it defined
      const c = a[0] as ClassValue
      const has = (n: string) => c.ns.has(n)
      const lt = (x: Value, y: Value) => {
        if (has('__lt__')) return h.truthy(h.call(c.ns.get('__lt__')!, [x, y]))
        if (has('__gt__')) return h.truthy(h.call(c.ns.get('__gt__')!, [y, x]))
        if (has('__le__')) return h.truthy(h.call(c.ns.get('__le__')!, [x, y])) && !h.truthy(h.call(c.ns.get('__eq__')!, [x, y]))
        return !h.truthy(h.call(c.ns.get('__ge__')!, [x, y]))
      }
      const eq = (x: Value, y: Value) => h.truthy(h.call(c.ns.get('__eq__')!, [x, y]))
      const add = (n: string, f: (x: Value, y: Value) => boolean) => { if (!has(n)) c.ns.set(n, F(n, (b) => bool(f(b[0], b[1])))) }
      add('__lt__', lt)
      add('__gt__', (x, y) => lt(y, x))
      add('__le__', (x, y) => lt(x, y) || eq(x, y))
      add('__ge__', (x, y) => !lt(x, y))
      return c
    }),
  })
}

const makeOperator = (h: Host): Value => {
  const two = (name: string, op: string): Value => F(name, (a) => h.binop(op, a[0], a[1]))
  const cmp = (name: string, op: string): Value => F(name, (a) => h.compare(op, a[0], a[1]))
  const item = (obj: Value, k: Value): Value => h.call(h.getAttr(obj, '__getitem__'), [k])
  const pick = (obj: Value, k: Value): Value => {
    // indexing through the interpreter's own rules, without a syntax-tree node to hand
    if (obj.k === 'list' || obj.k === 'tuple') {
      const n = obj.items.length
      const i = h.toInt(k)
      if (i < -n || i >= n) throw h.err('IndexError', `${obj.k} index out of range`)
      return obj.items[i < 0 ? i + n : i]
    }
    if (obj.k === 'dict') {
      const v = obj.d.get(k)
      if (!v) throw h.err('KeyError', repr(k))
      return v
    }
    if (obj.k === 'str') {
      const n = obj.v.length
      const i = h.toInt(k)
      if (i < -n || i >= n) throw h.err('IndexError', 'string index out of range')
      return S(obj.v[i < 0 ? i + n : i])
    }
    if (obj.k === 'obj' && obj.o.getItem) return obj.o.getItem(k, h)
    return item(obj, k)
  }
  return mod('operator', {
    add: two('add', '+'), sub: two('sub', '-'), mul: two('mul', '*'), truediv: two('truediv', '/'), floordiv: two('floordiv', '//'), mod: two('mod', '%'), pow: two('pow', '**'), matmul: two('matmul', '@'), concat: two('concat', '+'),
    and_: two('and_', '&'), or_: two('or_', '|'), xor: two('xor', '^'), lshift: two('lshift', '<<'), rshift: two('rshift', '>>'),
    eq: cmp('eq', '=='), ne: cmp('ne', '!='), lt: cmp('lt', '<'), le: cmp('le', '<='), gt: cmp('gt', '>'), ge: cmp('ge', '>='),
    neg: F('neg', (a) => h.binop('*', a[0], int(-1))),
    not_: F('not_', (a) => bool(!h.truthy(a[0]))),
    truth: F('truth', (a) => bool(h.truthy(a[0]))),
    contains: F('contains', (a) => h.compare('in', a[1], a[0])),
    getitem: F('getitem', (a) => pick(a[0], a[1])),
    itemgetter: F('itemgetter', (keys) => F('itemgetter', (a) => (keys.length === 1 ? pick(a[0], keys[0]) : tuple(keys.map((k) => pick(a[0], k)))))),
    attrgetter: F('attrgetter', (names) => F('attrgetter', (a) => {
      const one = (n: Value) => (n as { v: string }).v.split('.').reduce((o, p) => h.getAttr(o, p), a[0])
      return names.length === 1 ? one(names[0]) : tuple(names.map(one))
    })),
    methodcaller: F('methodcaller', (m, kw) => F('methodcaller', (a) => h.call(h.getAttr(a[0], (m[0] as { v: string }).v), m.slice(1), kw))),
  })
}

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

// ---------------------------------------------------------------- dataclasses, typing

class TypingAlias extends Base {
  readonly cls = 'typing._GenericAlias'
  name: string
  constructor(name: string) {
    super()
    this.name = name
  }
  repr() { return `typing.${this.name}` }
  getItem(idx: Value) {
    const shown = (v: Value): string => {
      if (v.k === 'type') return v.name
      if (v.k === 'class') return v.builtin && !v.module ? v.name : `${v.module ?? '__main__'}.${v.name}`
      if (v.k === 'none') return 'NoneType'
      if (v.k === 'ellipsis') return '...'
      if (v.k === 'list') return `[${v.items.map(shown).join(', ')}]`
      if (v.k === 'str') return `ForwardRef(${repr(v)})`
      return repr(v)
    }
    return { k: 'obj', o: new TypingAlias(`${this.name}[${(idx.k === 'tuple' ? idx.items : [idx]).map(shown).join(', ')}]`) } as Value
  }
  binop() { return { k: 'obj', o: this } as Value }
  call(args: Value[]) { return args[0] ?? NONE }
}

class FieldSpec extends Base {
  readonly cls = 'dataclasses.Field'
  def?: Value
  factory?: Value
  init: boolean
  useRepr: boolean
  constructor(def: Value | undefined, factory: Value | undefined, init: boolean, useRepr: boolean) {
    super()
    this.def = def
    this.factory = factory
    this.init = init
    this.useRepr = useRepr
  }
  repr() { return 'Field()' }
}

const makeDataclasses = (h: Host): Value => {
  const fieldsOf = (c: ClassValue): string[] => {
    const out: string[] = []
    const walk = (k: Value) => {
      if (k.k !== 'class') return
      k.bases.forEach(walk)
      const notes = k.ns.get('__annotations__')
      if (notes?.k === 'dict') for (const n of notes.d.keys()) if (!out.includes((n as { v: string }).v)) out.push((n as { v: string }).v)
    }
    walk(c)
    return out
  }
  const lookup = (c: ClassValue, n: string): Value | undefined => {
    if (c.ns.has(n)) return c.ns.get(n)
    for (const b of c.bases) if (b.k === 'class') {
      const v = lookup(b, n)
      if (v) return v
    }
    return undefined
  }
  const decorate = (c: ClassValue, opts: Kw): Value => {
    const names = fieldsOf(c)
    const specOf = (n: string) => {
      const v = lookup(c, n)
      return v?.k === 'obj' && v.o instanceof FieldSpec ? v.o : null
    }
    const defaultOf = (n: string): (() => Value) | null => {
      const v = lookup(c, n)
      const spec = specOf(n)
      if (spec) return spec.factory ? () => h.call(spec.factory!, []) : spec.def ? () => spec.def! : null
      return v ? () => v : null
    }
    const inInit = names.filter((n) => specOf(n)?.init !== false)
    let sawDefault = false
    for (const n of inInit) {
      if (defaultOf(n)) sawDefault = true
      else if (sawDefault) throw h.err('TypeError', `non-default argument '${n}' follows default argument`)
    }
    c.ns.set('__dataclass_fields__', tuple(names.map(S)))
    const setter = (inst: InstValue, n: string, v: Value) => inst.attrs.set(n, v)
    c.ns.set('__init__', { k: 'fn', name: '__init__', bindable: true, call: (args, kw) => {
      const inst = args[0] as InstValue
      const pos = args.slice(1)
      if (pos.length > inInit.length) throw h.err('TypeError', `${c.name}.__init__() takes ${inInit.length + 1} positional arguments but ${pos.length + 1} were given`)
      for (const k in kw) if (!inInit.includes(k)) throw h.err('TypeError', `${c.name}.__init__() got an unexpected keyword argument '${k}'`)
      const missing: string[] = []
      for (const n of names) {
        const at = inInit.indexOf(n)
        const given = at >= 0 && at < pos.length ? pos[at] : at >= 0 ? kw[n] : undefined
        const d = defaultOf(n)
        if (given) setter(inst, n, given)
        else if (d) setter(inst, n, d())
        else missing.push(`'${n}'`)
      }
      if (missing.length) throw h.err('TypeError', `${c.name}.__init__() missing ${missing.length} required positional argument${missing.length > 1 ? 's' : ''}: ${missing.join(' and ')}`)
      const post = lookup(c, '__post_init__')
      if (post) h.call(post, [inst])
      return NONE
    } })
    if (!opts.repr || h.truthy(opts.repr)) {
      if (!c.ns.has('__repr__')) c.ns.set('__repr__', { k: 'fn', name: '__repr__', bindable: true, call: (args) => {
        const inst = args[0] as InstValue
        return S(`${inst.cls.name}(${names.filter((n) => specOf(n)?.useRepr !== false).map((n) => `${n}=${repr(inst.attrs.get(n) ?? NONE)}`).join(', ')})`)
      } })
    }
    if (!c.ns.has('__eq__')) c.ns.set('__eq__', { k: 'fn', name: '__eq__', bindable: true, call: (args) => {
      const [a, b] = args
      if (a.k !== 'inst' || b.k !== 'inst' || a.cls !== b.cls) return { k: 'type', name: 'NotImplementedType', call: () => NONE }
      return bool(names.every((n) => h.truthy(h.compare('==', a.attrs.get(n) ?? NONE, b.attrs.get(n) ?? NONE))))
    } })
    if (opts.order && h.truthy(opts.order)) {
      const key = (v: Value) => tuple(names.map((n) => (v as InstValue).attrs.get(n) ?? NONE))
      for (const [n, op] of [['__lt__', '<'], ['__le__', '<='], ['__gt__', '>'], ['__ge__', '>=']] as const) {
        c.ns.set(n, { k: 'fn', name: n, bindable: true, call: (args) => h.binop(op, key(args[0]), key(args[1])) })
      }
    }
    if (opts.frozen && h.truthy(opts.frozen)) c.ns.set('__frozen__', bool(true))
    return c
  }
  const asDict = (v: Value): Value => {
    if (v.k === 'inst' && v.cls.ns.has('__dataclass_fields__')) return { k: 'dict', d: PyDict.from(fieldsOf(v.cls).map((n) => [S(n), asDict(v.attrs.get(n) ?? NONE)])) }
    if (v.k === 'list') return L(v.items.map(asDict))
    if (v.k === 'tuple') return tuple(v.items.map(asDict))
    return v
  }
  const asTuple = (v: Value): Value => {
    if (v.k === 'inst' && v.cls.ns.has('__dataclass_fields__')) return tuple(fieldsOf(v.cls).map((n) => asTuple(v.attrs.get(n) ?? NONE)))
    if (v.k === 'list') return L(v.items.map(asTuple))
    return v
  }
  return mod('dataclasses', {
    dataclass: F('dataclass', (a, kw) => (a[0]?.k === 'class' ? decorate(a[0], kw) : F('dataclass', (b) => decorate(b[0] as ClassValue, kw)))),
    field: F('field', (_a, kw) => ({ k: 'obj', o: new FieldSpec(kw.default, kw.default_factory, !kw.init || h.truthy(kw.init), !kw.repr || h.truthy(kw.repr)) })),
    asdict: F('asdict', (a) => asDict(a[0])),
    astuple: F('astuple', (a) => asTuple(a[0])),
    is_dataclass: F('is_dataclass', (a) => bool((a[0].k === 'inst' && a[0].cls.ns.has('__dataclass_fields__')) || (a[0].k === 'class' && a[0].ns.has('__dataclass_fields__')))),
    replace: F('replace', (a, kw) => {
      const inst = a[0] as InstValue
      const made = fieldsOf(inst.cls).map((n) => kw[n] ?? inst.attrs.get(n) ?? NONE)
      return h.call(inst.cls, made)
    }),
    fields: F('fields', (a) => {
      const c = (a[0].k === 'inst' ? a[0].cls : a[0]) as ClassValue
      return tuple(fieldsOf(c).map((n) => ({ k: 'tuple', items: [S(n)], names: ['name'], tname: 'Field' }) as Value))
    }),
  })
}

// ---------------------------------------------------------------- copy

const makeCopy = (h: Host): Value => {
  const shallow = (v: Value): Value => {
    switch (v.k) {
      case 'list': return { ...v, items: [...v.items], view: undefined }
      case 'dict': return { ...v, d: PyDict.from(v.d.items()) }
      case 'set': return makeSet(setItems(v), v.frozen)
      case 'array': return { k: 'array', a: v.a.copy() }
      case 'inst': {
        const f = h.getAttrOr(v, '__copy__')
        if (f) return h.call(f, [])
        return { ...v, attrs: new Map(v.attrs), state: { ...v.state } }
      }
      case 'obj': return libCopy(v)
    }
    return v
  }
  const libCopy = (v: Value & { k: 'obj' }): Value => {
    const f = h.getAttrOr(v, 'copy') ?? h.getAttrOr(v, 'clone')
    return f ? h.call(f, []) : v
  }
  const deep = (v: Value, memo: Map<unknown, Value>): Value => {
    const known = memo.get(v)
    if (known) return known
    switch (v.k) {
      case 'list': {
        const out: Value = { k: 'list', items: [] }
        memo.set(v, out)
        out.items.push(...v.items.map((x) => deep(x, memo)))
        return out
      }
      case 'tuple': return { ...v, items: v.items.map((x) => deep(x, memo)) }
      case 'dict': {
        const out: Value = { ...v, d: new PyDict() }
        memo.set(v, out)
        for (const [k, x] of v.d.items()) out.d.set(deep(k, memo), deep(x, memo))
        return out
      }
      case 'set': return makeSet(setItems(v).map((x) => deep(x, memo)), v.frozen)
      case 'array': return { k: 'array', a: v.a.copy() }
      case 'inst': {
        const f = h.getAttrOr(v, '__deepcopy__')
        if (f) return h.call(f, [{ k: 'dict', d: new PyDict() }])
        const out: Value = { ...v, attrs: new Map(), state: { ...v.state } }
        memo.set(v, out)
        for (const [n, x] of v.attrs) out.attrs.set(n, deep(x, memo))
        return out
      }
      case 'obj': return libCopy(v)
    }
    return v
  }
  return mod('copy', {
    copy: F('copy', (a) => shallow(a[0])),
    deepcopy: F('deepcopy', (a) => deep(a[0], new Map())),
  })
}

// ---------------------------------------------------------------- time and dates

const pad2 = (n: number, w = 2) => String(n).padStart(w, '0')
const DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
const strftime = (d: Date, fmt: string) => fmt.replace(/%(.)/g, (m, c: string) => {
  switch (c) {
    case 'Y': return String(d.getFullYear())
    case 'y': return pad2(d.getFullYear() % 100)
    case 'm': return pad2(d.getMonth() + 1)
    case 'd': return pad2(d.getDate())
    case 'H': return pad2(d.getHours())
    case 'I': return pad2(d.getHours() % 12 || 12)
    case 'M': return pad2(d.getMinutes())
    case 'S': return pad2(d.getSeconds())
    case 'f': return pad2(d.getMilliseconds() * 1000, 6)
    case 'p': return d.getHours() < 12 ? 'AM' : 'PM'
    case 'A': return DAYS[d.getDay()]
    case 'a': return DAYS[d.getDay()].slice(0, 3)
    case 'B': return MONTHS[d.getMonth()]
    case 'b': return MONTHS[d.getMonth()].slice(0, 3)
    case 'j': return pad2(Math.floor((d.getTime() - new Date(d.getFullYear(), 0, 1).getTime()) / 86400000) + 1, 3)
    case '%': return '%'
  }
  return m
})

class TimeDelta extends Base {
  readonly cls = 'datetime.timedelta'
  ms: number
  constructor(ms: number) {
    super()
    this.ms = ms
  }
  repr() { return `datetime.timedelta(seconds=${this.ms / 1000})` }
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
  binop(op: string, o: Value): Value | undefined {
    if (o.k === 'obj' && o.o instanceof TimeDelta) {
      if (op === '+') return { k: 'obj', o: new TimeDelta(this.ms + o.o.ms) }
      if (op === '-') return { k: 'obj', o: new TimeDelta(this.ms - o.o.ms) }
      if (op === '<') return bool(this.ms < o.o.ms)
      if (op === '>') return bool(this.ms > o.o.ms)
      if (op === '==') return bool(this.ms === o.o.ms)
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
    this.d = d
    this.dateOnly = dateOnly
  }
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
      const diff = this.d.getTime() - o.o.d.getTime()
      if (op === '-') return { k: 'obj', o: new TimeDelta(reflected ? -diff : diff) }
      if (op === '<') return bool(diff < 0)
      if (op === '>') return bool(diff > 0)
      if (op === '==') return bool(diff === 0)
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

// ---------------------------------------------------------------- files: kept in memory for one run

/** the run's files, by normalised path */
const filesOf = (h: Host): Map<string, string> => {
  let m = h.state.get('files') as Map<string, string> | undefined
  if (!m) h.state.set('files', (m = new Map()))
  return m
}
/** the run's folders, by normalised path; pandas can also write into a folder nobody made, and then that one exists too */
const dirsOf = (h: Host): Set<string> => {
  let d = h.state.get('dirs') as Set<string> | undefined
  if (!d) h.state.set('dirs', (d = new Set()))
  return d
}
const CWD = SANDBOX_CWD
const TEMP = '/tmp'
const FIRST_DIRS = new Set(['/', CWD, TEMP, '/content', '/dev'])
const isDirAt = (h: Host, full: string) => FIRST_DIRS.has(full) || dirsOf(h).has(full) || [...filesOf(h).keys()].some((k) => k.startsWith(`${full}/`))
const isFileAt = (h: Host, full: string) => filesOf(h).has(full)
/** what is directly inside a folder, sorted (a real OS gives them in no particular order) */
const namesIn = (h: Host, full: string): string[] => {
  const prefix = full === '/' ? '/' : `${full}/`
  const names = new Set<string>()
  for (const k of [...filesOf(h).keys(), ...dirsOf(h), ...FIRST_DIRS]) if (k !== '/' && k.startsWith(prefix)) names.add(k.slice(prefix.length).split('/')[0])
  return [...names].sort()
}
/** a path as the OS reads it: 'file.txt/' is not a file */
const existsAt = (h: Host, p: string) => isDirAt(h, fullPath(p)) || (!p.endsWith('/') && isFileAt(h, fullPath(p)))

const osError = (h: Host, cls: string, errno: number, text: string, p: string, p2?: string): never =>
  h.raise(h.excClass(cls), p2 === undefined ? [int(errno), S(text), S(p)] : [int(errno), S(text), S(p), NONE, S(p2)])
/** except sees Python's own message; the hint is only for a run that stops here */
const noSuchFile = (h: Host, p: string, p2?: string): never => {
  try {
    return osError(h, 'FileNotFoundError', 2, 'No such file or directory', p, p2)
  } catch (e) {
    ;(e as Error).message += ' (this sandbox starts with no files; a program can write its own, and they last for one run)'
    throw e
  }
}
const fileExists = (h: Host, p: string, p2?: string) => osError(h, 'FileExistsError', 17, 'File exists', p, p2)
const isADir = (h: Host, p: string, p2?: string) => osError(h, 'IsADirectoryError', 21, 'Is a directory', p, p2)
const notADir = (h: Host, p: string, p2?: string) => osError(h, 'NotADirectoryError', 20, 'Not a directory', p, p2)
const notEmpty = (h: Host, p: string, p2?: string) => osError(h, 'OSError', 39, 'Directory not empty', p, p2)
/** the folders above a path must exist, and be folders */
const checkParents = (h: Host, p: string, shown = p, shown2?: string) => {
  if (!p) noSuchFile(h, shown, shown2)
  const parts = fullPath(p).split('/').slice(1, -1)
  parts.forEach((_, i) => {
    const up = `/${parts.slice(0, i + 1).join('/')}`
    if (isFileAt(h, up)) notADir(h, shown, shown2)
    if (!isDirAt(h, up)) noSuchFile(h, shown, shown2)
  })
}

/** the path behind a str, a pathlib path or anything with __fspath__; null if it is none of those */
const pathOr = (h: Host, v0: Value | undefined): string | null => {
  const v = (v0?.k === 'inst' ? v0.cls.meta?.scalar?.(v0) : undefined) ?? v0
  if (v?.k === 'str') return v.v
  if (v?.k === 'obj' && v.o.fspath) return v.o.fspath()
  if (v?.k === 'inst') {
    const f = h.getAttrOr(v, '__fspath__')
    if (f) return h.str(h.call(f, []))
  }
  return null
}
const fspath = (h: Host, v: Value | undefined): string => {
  const p = pathOr(h, v)
  if (p === null) throw h.err('TypeError', `expected str, bytes or os.PathLike object, not ${v ? typeName(v) : 'NoneType'}`)
  return p
}

/** a pickle in a file is this token: the sandbox has no bytes, so the file only says which object it holds */
const PICKLED = /\0pickle:(\d+)\0/g
const holdsPickle = (text: string) => text.startsWith('\0pickle:')

/** a text file or io.StringIO: text held in memory, with a read position */
class TextFile extends Base {
  readonly cls: string
  text: string
  at = 0
  closed = false
  path: string | null
  mode: string
  files: Map<string, string> | null
  /** NamedTemporaryFile deletes its file on close */
  onClose: (() => void) | null = null
  /** open(newline='\r\n'): what \n becomes on the way to the file */
  writeNewline = '\n'
  constructor(cls: string, text: string, path: string | null, mode: string, files: Map<string, string> | null) {
    super()
    this.cls = cls
    this.text = text
    this.path = path
    this.mode = mode
    this.files = files
    if (mode.startsWith('a')) this.at = text.length
  }
  repr() { return this.path === null ? '<_io.StringIO object>' : `<_io.TextIOWrapper name=${repr(S(this.path))} mode=${repr(S(this.mode))} encoding='UTF-8'>` }
  check(h: Host) { if (this.closed) throw h.err('ValueError', this.path === null ? 'I/O operation on closed file' : 'I/O operation on closed file.') }
  isa(name: string) { return name === 'io.IOBase' || name === 'io.TextIOBase' || name === (this.path === null ? 'io.StringIO' : 'io.TextIOWrapper') }
  save() { if (this.files && this.path !== null) this.files.set(fullPath(this.path), this.text) }
  put(s0: string) {
    const s = this.writeNewline === '\n' ? s0 : s0.replace(/\n/g, this.writeNewline)
    this.text = this.text.slice(0, this.at) + s + this.text.slice(this.at + s.length)
    this.at += s.length
    this.save()
  }
  line() {
    const nl = this.text.indexOf('\n', this.at)
    const end = nl < 0 ? this.text.length : nl + 1
    const out = this.text.slice(this.at, end)
    this.at = end
    return out
  }
  canRead(h: Host) {
    this.check(h)
    if (this.path !== null && !/r|\+/.test(this.mode)) unsupported(h, 'not readable')
    if (holdsPickle(this.text)) h.raise(h.excClass('UnicodeDecodeError'), [S("'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte")])
  }
  iter(h: Host) {
    this.canRead(h)
    const out: Value[] = []
    while (this.at < this.text.length) out.push(S(this.line()))
    return out
  }
  nextItem(h: Host) {
    this.canRead(h)
    return this.at < this.text.length ? S(this.line()) : undefined
  }
  enter() { return { k: 'obj', o: this } as Value }
  finish() {
    if (!this.closed) this.onClose?.()
    this.closed = true
  }
  exit() { this.finish() }
  getAttr(name: string, h: Host): Value | undefined {
    switch (name) {
      case 'read': return F(name, (a) => {
        this.canRead(h)
        const n = a[0] && a[0].k !== 'none' ? h.toInt(a[0]) : -1
        const out = n < 0 ? this.text.slice(this.at) : this.text.slice(this.at, this.at + n)
        this.at += out.length
        return S(out)
      })
      case 'readline': return F(name, () => { this.canRead(h); return S(this.line()) })
      case 'readlines': return F(name, () => L(this.iter(h)))
      case 'write': return F(name, (a) => {
        this.check(h)
        if (this.path !== null && this.mode.startsWith('r') && !this.mode.includes('+')) unsupported(h, 'not writable')
        if (a[0]?.k !== 'str') throw h.err('TypeError', `write() argument must be str, not ${typeName(a[0] ?? NONE)}`)
        this.put(a[0].v)
        return int(a[0].v.length)
      })
      case 'writelines': return F(name, (a) => { this.check(h); for (const x of h.iterate(a[0])) this.put((x as { v: string }).v); return NONE })
      case 'getvalue': return this.path === null ? F(name, () => { this.check(h); return S(this.text) }) : undefined
      case 'seek': return F(name, (a, kw) => {
        this.check(h)
        const whence = kw.whence ?? a[1]
        const from = whence ? h.toInt(whence) : 0
        this.at = h.toInt(a[0]) + (from === 1 ? this.at : from === 2 ? this.text.length : 0)
        return int(this.at)
      })
      case 'tell': return F(name, () => int(this.at))
      case 'truncate': return F(name, (a) => {
        const n = a[0] && a[0].k !== 'none' ? h.toInt(a[0]) : this.at
        this.text = this.text.slice(0, n)
        this.save()
        return int(n)
      })
      case 'flush': return F(name, () => NONE)
      case 'close': return F(name, () => { this.finish(); return NONE })
      case 'readable': return F(name, () => bool(this.path === null || /r|\+/.test(this.mode)))
      case 'writable': return F(name, () => bool(this.path === null || !this.mode.startsWith('r') || this.mode.includes('+')))
      case 'seekable': return F(name, () => bool(true))
      case 'isatty': return F(name, () => bool(false))
      case 'name': return this.path === null ? undefined : S(this.path)
      case 'mode': return this.path === null ? undefined : S(this.mode)
      case 'encoding': return this.path === null ? NONE : S('UTF-8')
      case 'closed': return bool(this.closed)
    }
    return undefined
  }
}

const unsupported = (h: Host, what: string): never => h.raise((h.module('io') as { attrs: Record<string, ClassValue> }).attrs.UnsupportedOperation, [S(what)])
const NO_BYTES = "this sandbox has no bytes type (b'...' is a plain string here), so binary files and io.BytesIO hold only pickles: open the file in text mode"

/** what pickle.dumps returns and a binary file holds: bytes in name only, which only pickle can read back */
class Pickled extends Base {
  readonly cls = 'bytes'
  items: Value[]
  constructor(items: Value[]) {
    super()
    this.items = items
  }
  repr() { return this.items.length ? `b'\\x80\\x04<${this.items.length > 1 ? `${this.items.length} pickles` : `pickled ${typeName(this.items[0])}`}: only pickle can read this>'` : "b''" }
  truthy() { return this.items.length > 0 }
  len(h: Host) {
    if (this.items.length) throw h.err('NotImplementedError', 'the size of a pickle is not known in this sandbox: dumps gives a stand-in for the bytes, not the bytes')
    return 0
  }
  binop(op: string, other: Value): Value | undefined {
    if (op !== '==' && op !== '!=') return undefined
    const same = other.k === 'obj' && other.o instanceof Pickled && this.repr() === other.o.repr() && this.items.every((x, i) => repr(x) === repr((other.o as Pickled).items[i]))
    return bool((op === '==') === same)
  }
}

const picklesOf = (h: Host): Value[] => {
  let p = h.state.get('pickles') as Value[] | undefined
  if (!p) h.state.set('pickles', (p = []))
  return p
}

/** a file opened in a binary mode, or an io.BytesIO: a row of pickles, as tokens in the same store as text */
class BinaryFile extends Base {
  readonly cls: string
  items: Value[]
  at = 0
  closed = false
  path: string | null
  mode: string
  files: Map<string, string> | null
  onClose: (() => void) | null = null
  constructor(cls: string, items: Value[], path: string | null, mode: string, files: Map<string, string> | null) {
    super()
    this.cls = cls
    this.items = items
    this.path = path
    this.mode = mode
    this.files = files
    if (mode.startsWith('a')) this.at = items.length
  }
  repr() { return this.path === null ? '<_io.BytesIO object>' : `<_io.${this.cls} name=${repr(S(this.path))}>` }
  check(h: Host, what: string) { if (this.closed) throw h.err('ValueError', this.path === null ? 'I/O operation on closed file.' : `${what} closed file`) }
  isa(name: string) { return name === 'io.IOBase' || name === 'io.BufferedIOBase' || name === (this.path === null ? 'io.BytesIO' : `io.${this.cls}`) }
  canRead(h: Host, what = 'read of') {
    this.check(h, what)
    if (this.path !== null && !/r|\+/.test(this.mode)) unsupported(h, 'read')
  }
  /** the next pickle in the file, for pickle.load */
  nextPickle(h: Host): Value | undefined {
    this.canRead(h, 'peek of')
    return this.items[this.at++]
  }
  write(v: Value, h: Host) {
    this.check(h, 'write to')
    if (this.path !== null && this.mode.startsWith('r') && !this.mode.includes('+')) unsupported(h, 'write')
    if (v.k !== 'obj' || !(v.o instanceof Pickled)) throw h.err('NotImplementedError', NO_BYTES)
    this.items.splice(this.at, v.o.items.length, ...v.o.items)
    this.at += v.o.items.length
    if (this.files && this.path !== null) {
      const table = picklesOf(h)
      this.files.set(fullPath(this.path), this.items.map((x) => `\0pickle:${table.push(x) - 1}\0`).join(''))
    }
  }
  enter() { return { k: 'obj', o: this } as Value }
  finish() {
    if (!this.closed) this.onClose?.()
    this.closed = true
  }
  exit() { this.finish() }
  getAttr(name: string, h: Host): Value | undefined {
    switch (name) {
      case 'read': return F(name, (a) => {
        this.canRead(h)
        if (a[0] && a[0].k !== 'none' && h.toInt(a[0]) >= 0) throw h.err('NotImplementedError', 'f.read(n) on a binary file is not available in this sandbox: the file holds pickles, not bytes')
        const out = this.items.slice(this.at)
        this.at = this.items.length
        return { k: 'obj', o: new Pickled(out) }
      })
      // None, not a count: a pickle's size in bytes is not known here
      case 'write': return F(name, (a) => { this.write(a[0], h); return NONE })
      case 'getvalue': return this.path === null ? F(name, () => ({ k: 'obj', o: new Pickled([...this.items]) })) : undefined
      case 'seek': return F(name, (a) => {
        if (h.toInt(a[0]) !== 0 || (a[1] && h.toInt(a[1]) !== 0)) throw h.err('NotImplementedError', 'a binary file can only seek(0) in this sandbox: it holds pickles, not bytes')
        this.at = 0
        return int(0)
      })
      case 'tell': return F(name, () => {
        if (this.at) throw h.err('NotImplementedError', 'the position in a binary file is not known in this sandbox: it holds pickles, not bytes')
        return int(0)
      })
      case 'flush': return F(name, () => NONE)
      case 'close': return F(name, () => { this.finish(); return NONE })
      case 'readable': return F(name, () => bool(this.path === null || /r|\+/.test(this.mode)))
      case 'writable': return F(name, () => bool(this.path === null || !this.mode.startsWith('r') || this.mode.includes('+')))
      case 'seekable': return F(name, () => bool(true))
      case 'isatty': return F(name, () => bool(false))
      case 'name': return this.path === null ? undefined : S(this.path)
      case 'mode': return this.path === null ? undefined : S(this.mode)
      case 'closed': return bool(this.closed)
    }
    return undefined
  }
}

/** open(): mode, newline and the rest as Python checks them */
const openFile = (h: Host, pathV: Value | undefined, modeV: Value | undefined, kw: Kw): TextFile | BinaryFile => {
  const path = fspath(h, pathV)
  const mode = modeV ? (modeV as { v: string }).v : 'r'
  if (/[^xrwabt+]/.test(mode) || new Set(mode).size !== mode.length) throw h.err('ValueError', `invalid mode: ${repr(S(mode))}`)
  if (mode.includes('b') && mode.includes('t')) throw h.err('ValueError', "can't have text and binary mode at once")
  const opening = [...mode].filter((c) => 'xrwa'.includes(c))
  if (opening.length > 1) throw h.err('ValueError', 'must have exactly one of create/read/write/append mode')
  if (!opening.length) throw h.err('ValueError', 'Must have exactly one of create/read/write/append mode and at most one plus')
  const binary = mode.includes('b')
  if (binary && kw.encoding && kw.encoding.k !== 'none') throw h.err('ValueError', "binary mode doesn't take an encoding argument")
  if (binary && kw.errors && kw.errors.k !== 'none') throw h.err('ValueError', "binary mode doesn't take an errors argument")
  if (binary && kw.newline && kw.newline.k !== 'none') throw h.err('ValueError', "binary mode doesn't take a newline argument")
  const files = filesOf(h)
  const key = fullPath(path)
  const kind = opening[0]
  const have = files.get(key)
  checkParents(h, path)
  if (isDirAt(h, key)) isADir(h, path)
  if (have !== undefined && path.endsWith('/')) notADir(h, path)
  if (kind === 'r' && have === undefined) noSuchFile(h, path)
  if (kind === 'x' && have !== undefined) fileExists(h, path)
  const raw = kind === 'w' || kind === 'x' ? '' : have ?? ''
  if (have === undefined || raw !== have) files.set(key, raw)
  if (!binary) {
    const newline = kw.newline && kw.newline.k !== 'none' ? h.str(kw.newline) : null
    if (newline !== null && !['', '\n', '\r', '\r\n'].includes(newline)) throw h.err('ValueError', `illegal newline value: ${repr(S(newline))}`)
    if (newline === '\r') throw h.err('NotImplementedError', "open(newline='\\r') is not available in this sandbox")
    // text mode reads \r\n as \n, unless newline='' asks for the file as it is
    const text = newline !== null || kind === 'a' ? raw : raw.replace(/\r\n?/g, '\n')
    const f = new TextFile('TextIOWrapper', text, path, mode, files)
    if (newline) f.writeNewline = newline
    return f
  }
  if (raw && !holdsPickle(raw)) throw h.err('NotImplementedError', NO_BYTES)
  const table = picklesOf(h)
  const items = [...raw.matchAll(PICKLED)].map((m) => table[Number(m[1])])
  const plus = mode.includes('+')
  return new BinaryFile(kind === 'r' && !plus ? 'BufferedReader' : plus ? 'BufferedRandom' : 'BufferedWriter', items, path, `${kind === 'x' ? 'x' : kind === 'a' ? 'a' : plus || kind === 'r' ? 'r' : 'w'}b${plus ? '+' : ''}`, files)
}

/** open()'s encoding, errors and newline, however they were passed; `at` is where encoding comes among the arguments */
const textOptions = (a: Value[], kw: Kw, at: number): Kw => {
  const out: Kw = {}
  ;['encoding', 'errors', 'newline'].forEach((n, i) => {
    const v = kw[n] ?? a[at + i]
    if (v) out[n] = v
  })
  return out
}

const makeFiles = (h: Host): Value => mod('_sandbox_files', {
  open: F('open', (a, kw) => {
    refuse(h, 'open', kw, ['opener'])
    return { k: 'obj', o: openFile(h, kw.file ?? a[0], kw.mode ?? a[1], textOptions(a, kw, 3)) }
  }),
})

const makeIo = (h: Host): Value => mod('io', {
  StringIO: { k: 'type', name: 'io.StringIO', call: (a, kw) => {
    const init = kw.initial_value ?? a[0]
    return { k: 'obj', o: new TextFile('StringIO', init?.k === 'str' ? init.v : '', null, 'r+', null) }
  } },
  BytesIO: { k: 'type', name: 'io.BytesIO', call: (a, kw) => {
    const init = kw.initial_bytes ?? a[0]
    if (init && init.k !== 'none' && !(init.k === 'obj' && init.o instanceof Pickled)) throw h.err('NotImplementedError', NO_BYTES)
    return { k: 'obj', o: new BinaryFile('BytesIO', init?.k === 'obj' ? [...(init.o as Pickled).items] : [], null, 'r+b', null) }
  } },
  UnsupportedOperation: { k: 'class', name: 'UnsupportedOperation', module: 'io', bases: [h.excClass('OSError'), h.excClass('ValueError')], ns: new Map(), host: h.excClass('OSError').host, builtin: true },
  ...Object.fromEntries(['IOBase', 'TextIOBase', 'BufferedIOBase', 'TextIOWrapper', 'BufferedReader', 'BufferedWriter', 'BufferedRandom'].map((n): [string, Value] => [n, { k: 'type', name: `io.${n}`, call: () => { throw h.err('NotImplementedError', `io.${n} cannot be built directly in this sandbox: use open()`) } }])),
})

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

// ---------------------------------------------------------------- os, sys

class Stream extends Base {
  readonly cls = 'TextIOWrapper'
  out: boolean
  constructor(out: boolean) {
    super()
    this.out = out
  }
  repr() { return `<_io.TextIOWrapper name='<std${this.out ? 'out' : 'err'}>' mode='w' encoding='utf-8'>` }
  getAttr(name: string, h: Host): Value | undefined {
    if (name === 'write') return F(name, (a) => {
      if (a[0]?.k !== 'str') throw h.err('TypeError', `write() argument must be str, not ${typeName(a[0] ?? NONE)}`)
      const s = a[0].v
      if (this.out) {
        // straight to the run's output, even while sys.stdout is replaced
        const replaced = h.stdoutFile
        h.stdoutFile = null
        h.print(s)
        h.stdoutFile = replaced
      }
      return int(s.length)
    })
    if (name === 'writelines') return F(name, (a) => {
      for (const x of h.iterate(a[0])) h.call(h.getAttr({ k: 'obj', o: this }, 'write'), [x])
      return NONE
    })
    if (name === 'flush') return F(name, () => NONE)
    if (name === 'isatty') return F(name, () => bool(false))
    if (name === 'encoding') return S('utf-8')
    if (name === 'name') return S(`<std${this.out ? 'out' : 'err'}>`)
    if (name === 'closed') return bool(false)
    return undefined
  }
}

const makeOsPath = (h: Host): Value => {
  const files = filesOf(h)
  const stat = (v: Value) => {
    const p = pathOr(h, v)
    if (p === null) throw h.err('TypeError', `stat: path should be string, bytes, os.PathLike or integer, not ${typeName(v ?? NONE)}`)
    return p
  }
  const headOf = (p: string) => {
    const head = p.slice(0, p.lastIndexOf('/') + 1)
    return head && !/^\/+$/.test(head) ? head.replace(/\/+$/, '') : head
  }
  const normpath = (p: string) => {
    if (!p) return '.'
    const slashes = p.startsWith('//') && !p.startsWith('///') ? 2 : p.startsWith('/') ? 1 : 0
    const out: string[] = []
    for (const c of p.split('/')) {
      if (!c || c === '.') continue
      if (c !== '..' || (!slashes && !out.length) || out[out.length - 1] === '..') out.push(c)
      else if (out.length) out.pop()
    }
    return '/'.repeat(slashes) + out.join('/') || '.'
  }
  const join = (a: Value[]) => {
    let path = fspath(h, a[0])
    for (const x of a.slice(1)) {
      const b = pathOr(h, x)
      if (b === null) throw h.err('TypeError', `join() argument must be str, bytes, or os.PathLike object, not '${typeName(x)}'`)
      if (b.startsWith('/')) path = b
      else if (!path || path.endsWith('/')) path += b
      else path += `/${b}`
    }
    return path
  }
  return mod('os.path', {
    sep: S('/'),
    join: F('join', (a) => S(join(a))),
    basename: F('basename', (a) => { const p = fspath(h, a[0]); return S(p.slice(p.lastIndexOf('/') + 1)) }),
    dirname: F('dirname', (a) => S(headOf(fspath(h, a[0])))),
    split: F('split', (a) => { const p = fspath(h, a[0]); return tuple([S(headOf(p)), S(p.slice(p.lastIndexOf('/') + 1))]) }),
    splitext: F('splitext', (a) => {
      const p = fspath(h, a[0])
      const sepAt = p.lastIndexOf('/')
      const dot = p.lastIndexOf('.')
      if (dot > sepAt && /[^.]/.test(p.slice(sepAt + 1, dot))) return tuple([S(p.slice(0, dot)), S(p.slice(dot))])
      return tuple([S(p), S('')])
    }),
    exists: F('exists', (a) => bool(existsAt(h, stat(a[0])))),
    lexists: F('lexists', (a) => bool(existsAt(h, stat(a[0])))),
    isfile: F('isfile', (a) => { const p = stat(a[0]); return bool(!p.endsWith('/') && isFileAt(h, fullPath(p))) }),
    isdir: F('isdir', (a) => bool(isDirAt(h, fullPath(stat(a[0]))))),
    islink: F('islink', (a) => { stat(a[0]); return bool(false) }),
    isabs: F('isabs', (a) => bool(fspath(h, a[0]).startsWith('/'))),
    getsize: F('getsize', (a) => {
      const p = stat(a[0])
      const v = files.get(fullPath(p))
      if (v === undefined) return isDirAt(h, fullPath(p)) ? int(4096) : noSuchFile(h, p)
      if (holdsPickle(v)) throw h.err('NotImplementedError', 'the size of a pickle file is not known in this sandbox: it holds a stand-in for the bytes, not the bytes')
      return int(new TextEncoder().encode(v).length)
    }),
    abspath: F('abspath', (a) => S(fullPath(fspath(h, a[0])))),
    realpath: F('realpath', (a) => S(fullPath(fspath(h, a[0])))),
    normpath: F('normpath', (a) => S(normpath(fspath(h, a[0])))),
    expanduser: F('expanduser', (a) => { const p = fspath(h, a[0]); return S(/^~(\/|$)/.test(p) ? CWD + p.slice(1) : p) }),
    relpath: F('relpath', (a, kw) => {
      const p = fspath(h, a[0])
      if (!p) throw h.err('ValueError', 'no path')
      const start = kw.start ?? a[1]
      const from = fullPath(start && start.k !== 'none' ? fspath(h, start) : '.').split('/').filter(Boolean)
      const to = fullPath(p).split('/').filter(Boolean)
      let i = 0
      while (i < from.length && i < to.length && from[i] === to[i]) i++
      return S([...from.slice(i).map(() => '..'), ...to.slice(i)].join('/') || '.')
    }),
    commonpath: F('commonpath', (a) => {
      const ps = h.iterate(a[0]).map((x) => fspath(h, x))
      if (!ps.length) throw h.err('ValueError', 'commonpath() arg is an empty sequence')
      if (new Set(ps.map((p) => p.startsWith('/'))).size > 1) throw h.err('ValueError', "Can't mix absolute and relative paths")
      const split = ps.map((p) => p.split('/').filter((c) => c && c !== '.'))
      const n = Math.min(...split.map((c) => c.length))
      let i = 0
      while (i < n && split.every((c) => c[i] === split[0][i])) i++
      return S((ps[0].startsWith('/') ? '/' : '') + split[0].slice(0, i).join('/'))
    }),
  })
}

/** every path at or under a folder, for moving or deleting the whole of it */
const treeOf = (h: Host, full: string) => ({
  files: [...filesOf(h).keys()].filter((k) => k === full || k.startsWith(`${full}/`)),
  dirs: [...dirsOf(h)].filter((k) => k === full || k.startsWith(`${full}/`)),
})
const removeTree = (h: Host, full: string) => {
  const t = treeOf(h, full)
  t.files.forEach((k) => filesOf(h).delete(k))
  t.dirs.forEach((k) => dirsOf(h).delete(k))
}
const moveTree = (h: Host, from: string, to: string) => {
  const files = filesOf(h)
  const dirs = dirsOf(h)
  const t = treeOf(h, from)
  const wasDir = isDirAt(h, from)
  for (const k of t.files) {
    const text = files.get(k)!
    files.delete(k)
    files.set(to + k.slice(from.length), text)
  }
  for (const k of t.dirs) {
    dirs.delete(k)
    dirs.add(to + k.slice(from.length))
  }
  if (wasDir) dirs.add(to)
}
const mkdirAt = (h: Host, p: string) => {
  const full = fullPath(p)
  if (isDirAt(h, full) || isFileAt(h, full)) fileExists(h, p)
  checkParents(h, p)
  dirsOf(h).add(full)
}
const makedirs = (h: Host, p: string, existOk: boolean) => {
  if (!p) noSuchFile(h, p)
  const parts = fullPath(p).split('/').filter(Boolean)
  for (let i = 1; i < parts.length; i++) {
    const up = `/${parts.slice(0, i).join('/')}`
    if (isFileAt(h, up)) notADir(h, p)
    if (!isDirAt(h, up)) dirsOf(h).add(up)
  }
  if (existOk && isDirAt(h, fullPath(p))) return
  mkdirAt(h, p)
}
/** os.rename on POSIX */
const renameAt = (h: Host, from: string, to: string) => {
  const [a, b] = [fullPath(from), fullPath(to)]
  const srcDir = isDirAt(h, a)
  if (!srcDir && !isFileAt(h, a)) noSuchFile(h, from, to)
  checkParents(h, to, from, to)
  if (isDirAt(h, b) && !srcDir) isADir(h, from, to)
  if (isFileAt(h, b) && srcDir) notADir(h, from, to)
  if (isDirAt(h, b) && namesIn(h, b).length && a !== b) notEmpty(h, from, to)
  if (a === b) return
  if (b.startsWith(`${a}/`)) osError(h, 'OSError', 22, 'Invalid argument', from, to)
  removeTree(h, b)
  moveTree(h, a, b)
}
/** (root, dirs, files) from the top down; a program may prune dirs in place */
const walkTree = function* (h: Host, top: string): Generator<Value, void, undefined> {
  const names = namesIn(h, fullPath(top))
  const dirs = L(names.filter((n) => isDirAt(h, fullPath(`${top}/${n}`))).map(S))
  yield tuple([S(top), dirs, L(names.filter((n) => isFileAt(h, fullPath(`${top}/${n}`))).map(S))])
  for (const d of (dirs as { items: Value[] }).items) yield* walkTree(h, top.endsWith('/') ? top + (d as { v: string }).v : `${top}/${(d as { v: string }).v}`)
}
const fdsOf = (h: Host): Map<number, string> => {
  let t = h.state.get('fds') as Map<number, string> | undefined
  if (!t) h.state.set('fds', (t = new Map()))
  return t
}

const makeOs = (h: Host): Value => {
  const files = filesOf(h)
  const dirs = dirsOf(h)
  const arg = (v: Value | undefined, fn: string) => {
    const p = pathOr(h, v)
    if (p === null) throw h.err('TypeError', `${fn}: path should be string, bytes or os.PathLike, not ${typeName(v ?? NONE)}`)
    return p
  }
  const removeFile = (fn: string): Value => F(fn, (a, kw) => {
    const p = arg(kw.path ?? a[0], fn)
    const full = fullPath(p)
    if (isDirAt(h, full)) isADir(h, p)
    return files.delete(full) ? NONE : noSuchFile(h, p)
  })
  const rename = (fn: string): Value => F(fn, (a, kw) => {
    renameAt(h, arg(kw.src ?? a[0], fn), arg(kw.dst ?? a[1], fn))
    return NONE
  })
  const environ: Value & { k: 'dict' } = { k: 'dict', d: new PyDict() }
  const statResult = (p: string): Value => {
    const size = (h.getAttr(h.module('os.path'), 'getsize'))
    return { k: 'tuple', items: [h.call(size, [S(p)])], names: ['st_size'], tname: 'os.stat_result' }
  }
  const fd = (v: Value) => {
    const n = h.toInt(v)
    const p = fdsOf(h).get(n)
    return { n, p: p ?? h.raise(h.excClass('OSError'), [int(9), S('Bad file descriptor')]) }
  }
  return takingKeywords(mod('os', {
    path: makeOsPath(h),
    PathLike: { k: 'type', name: 'os.PathLike', call: () => { throw h.err('TypeError', "Can't instantiate abstract class PathLike with abstract method __fspath__") } },
    sep: S('/'),
    linesep: S('\n'),
    curdir: S('.'),
    pardir: S('..'),
    extsep: S('.'),
    pathsep: S(':'),
    devnull: S('/dev/null'),
    name: S('posix'),
    environ,
    getenv: F('getenv', (a, kw) => environ.d.get(kw.key ?? a[0]) ?? kw.default ?? a[1] ?? NONE),
    getcwd: F('getcwd', () => S(CWD)),
    fspath: F('fspath', (a) => S(fspath(h, a[0]))),
    listdir: F('listdir', (a, kw) => {
      const given = kw.path ?? a[0]
      const p = !given || given.k === 'none' ? '.' : pathOr(h, given)
      if (p === null) throw h.err('TypeError', `listdir: path should be string, bytes, os.PathLike, integer or None, not ${typeName(given)}`)
      const full = fullPath(p)
      if (isFileAt(h, full)) notADir(h, p)
      if (!isDirAt(h, full)) noSuchFile(h, p)
      return L(namesIn(h, full).map(S))
    }),
    walk: F('walk', (a, kw) => {
      for (const k of ['topdown', 'onerror', 'followlinks']) if (kw[k] && (k === 'topdown' ? !h.truthy(kw[k]) : kw[k].k !== 'none' && h.truthy(kw[k]))) throw h.err('NotImplementedError', `os.walk(${k}=...) is not available in this sandbox`)
      const top = fspath(h, kw.top ?? a[0])
      return fromGen('generator', isDirAt(h, fullPath(top)) ? walkTree(h, top) : (function* (): Generator<Value, void, undefined> {})())
    }),
    stat: F('stat', (a) => {
      const p = arg(a[0], 'stat')
      if (!existsAt(h, p)) noSuchFile(h, p)
      return statResult(p)
    }),
    cpu_count: F('cpu_count', () => int(1)),
    chdir: F('chdir', () => { throw h.err('NotImplementedError', 'os.chdir is not available in this sandbox: use paths from the current folder') }),
    remove: removeFile('remove'),
    unlink: removeFile('unlink'),
    rmdir: F('rmdir', (a, kw) => {
      const p = arg(kw.path ?? a[0], 'rmdir')
      const full = fullPath(p)
      if (isFileAt(h, full)) notADir(h, p)
      if (!isDirAt(h, full)) noSuchFile(h, p)
      if (namesIn(h, full).length) notEmpty(h, p)
      dirs.delete(full)
      return NONE
    }),
    rename: rename('rename'),
    replace: rename('replace'),
    makedirs: F('makedirs', (a, kw) => {
      const existOk = kw.exist_ok ?? a[2]
      makedirs(h, fspath(h, kw.name ?? a[0]), !!existOk && h.truthy(existOk))
      return NONE
    }),
    mkdir: F('mkdir', (a, kw) => { mkdirAt(h, arg(kw.path ?? a[0], 'mkdir')); return NONE }),
    close: F('close', (a) => { fdsOf(h).delete(fd(a[0]).n); return NONE }),
    fdopen: F('fdopen', (a, kw) => {
      const { n, p } = fd(a[0])
      const f = openFile(h, S(p), kw.mode ?? a[1], kw)
      f.onClose = () => { fdsOf(h).delete(n) }
      return { k: 'obj', o: f }
    }),
  }), {
    listdir: ['path'], walk: ['top', 'topdown', 'onerror', 'followlinks'], stat: ['path', 'dir_fd', 'follow_symlinks'], remove: ['path', 'dir_fd'], unlink: ['path', 'dir_fd'], rmdir: ['path', 'dir_fd'],
    rename: ['src', 'dst', 'src_dir_fd', 'dst_dir_fd'], replace: ['src', 'dst', 'src_dir_fd', 'dst_dir_fd'], makedirs: ['name', 'mode', 'exist_ok'], mkdir: ['path', 'mode', 'dir_fd'], fspath: [],
  })
}

const makeSys = (h: Host): Value => {
  const stdout: Value = { k: 'obj', o: new Stream(true) }
  const stderr: Value = { k: 'obj', o: new Stream(false) }
  const sys = mod('sys', {
    argv: L([S('main.py')]),
    path: L([]),
    modules: { k: 'dict', d: new PyDict() },
    version: S('3.13 (KnowCS sandbox)'),
    version_info: { k: 'tuple', items: [int(3), int(13), int(0)], names: ['major', 'minor', 'micro'], tname: 'sys.version_info' },
    platform: S('sandbox'),
    maxsize: int(Number.MAX_SAFE_INTEGER),
    stderr,
    __stdout__: stdout,
    __stderr__: stderr,
    exit: F('exit', (a) => h.raise(h.excClass('SystemExit'), a.slice(0, 1))),
    getrecursionlimit: F('getrecursionlimit', () => int(1000)),
    byteorder: S('little'),
    setrecursionlimit: F('setrecursionlimit', () => NONE),
    getsizeof: F('getsizeof', (a) => {
      // CPython 3.12 on 64-bit, for what a lesson on memory compares
      const v = a[0]
      switch (v.k) {
        case 'int': return int(v.v === 0 ? 28 : 24 + 4 * Math.ceil(Math.log2(Math.abs(v.v) + 1) / 30))
        case 'bool': return int(28)
        case 'float': return int(24)
        case 'none': return int(16)
        case 'str': return int(41 + v.v.length)
        case 'list': return int(56 + 8 * v.items.length)
        case 'tuple': return int(40 + 8 * v.items.length)
        case 'array': return int(112 + (v.a.base ? 0 : v.a.size * 8))
      }
      throw h.err('NotImplementedError', `sys.getsizeof is not modelled for '${typeName(v)}' in this sandbox`)
    }),
  }) as Value & { k: 'module' }
  // assigning sys.stdout is what redirects print
  Object.defineProperty(sys.attrs, 'stdout', {
    enumerable: true,
    get: () => h.stdoutFile ?? stdout,
    set: (v: Value) => { h.stdoutFile = v === stdout ? null : v },
  })
  return sys
}

// ---------------------------------------------------------------- helpers for classes a program can subclass

/** a library class made of plain functions: self comes first, as in Python */
const libClass = (module: string, name: string, methods: Record<string, (args: Value[], kw: Kw) => Value>, bases: Value[] = []): ClassValue => ({
  k: 'class', name, module, bases, host: null, builtin: true,
  ns: new Map(Object.entries(methods).map(([n, call]): [string, Value] => [n, { k: 'fn', name: n, bindable: true, call: guarded(n, call) }])),
})
/** the keywords each function of a module takes: any other is Python's TypeError, not quietly ignored */
const takingKeywords = (module: Value, table: Record<string, string[]>): Value => {
  const attrs = (module as Value & { k: 'module' }).attrs
  for (const n in table) {
    const f = attrs[n] as Value & { k: 'fn' }
    const inner = f.call
    f.call = (a, kw) => {
      for (const k in kw) if (!table[n].includes(k)) throw new PyError('TypeError', `${n}() got an unexpected keyword argument '${k}'`)
      return inner(a, kw)
    }
  }
  return module
}
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
/** a parameter that changes the result and is not implemented */
const refuse = (h: Host, fn: string, kw: Kw, names: string[]) => {
  for (const n of names) if (kw[n] && kw[n].k !== 'none' && !(kw[n].k === 'bool' && !kw[n].v)) throw h.err('NotImplementedError', `${fn}(${n}=...) is not available in this sandbox`)
}
const randomChars = (alphabet: string, n: number) => Array.from({ length: n }, () => alphabet[Math.floor(Math.random() * alphabet.length)]).join('')

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

// ---------------------------------------------------------------- pickle

const makePickle = (h: Host): Value => {
  const inPickle = (name: string) => {
    const c = h.excClass(name)
    c.module = '_pickle'
    return c
  }
  const [PickleError, PicklingError, UnpicklingError] = ['PickleError', 'PicklingError', 'UnpicklingError'].map(inPickle)
  const cannot = (what: string) => h.err('TypeError', `cannot pickle '${what}' object`)
  /** a library object with no copy of its own: its fields, copied all the way down; functions and the interpreter stay shared */
  const fieldsCopy = (x: unknown, memo: Map<unknown, unknown>): unknown => {
    if (typeof x !== 'object' || x === null || x === h) return x
    if (memo.has(x)) return memo.get(x)
    if (x instanceof Base && x.pickled) {
      const out = x.pickled((part) => fieldsCopy(part, memo) as typeof part)
      memo.set(x, out)
      return out
    }
    if (ArrayBuffer.isView(x)) return (x as unknown as Float64Array).slice()
    if (x instanceof Map) {
      const out = new Map()
      memo.set(x, out)
      for (const [k, v] of x) out.set(fieldsCopy(k, memo), fieldsCopy(v, memo))
      return out
    }
    if (x instanceof Set) {
      const out = new Set()
      memo.set(x, out)
      for (const v of x) out.add(fieldsCopy(v, memo))
      return out
    }
    if (Array.isArray(x)) {
      const out: unknown[] = []
      memo.set(x, out)
      for (const v of x) out.push(fieldsCopy(v, memo))
      return out
    }
    const k = (x as { k?: string }).k
    if (k === 'fn' || k === 'func' || k === 'type' || k === 'class' || k === 'module' || k === 'bound' || k === 'iter') return x
    const out = Object.create(Object.getPrototypeOf(x)) as Record<string, unknown>
    memo.set(x, out)
    for (const key of Object.keys(x)) out[key] = fieldsCopy((x as Record<string, unknown>)[key], memo)
    return out
  }
  /** an instance while it is pickled: its class and its state, with __setstate__ still to come */
  class Frozen extends Base {
    readonly cls: string
    of: InstValue
    held: Value = NONE
    jsState: Record<string, unknown>
    constructor(of: InstValue) {
      super()
      this.of = of
      this.cls = of.cls.name
      this.jsState = fieldsCopy(of.state, new Map()) as Record<string, unknown>
    }
    repr() { return `<pickled ${this.cls}: ${repr(this.held)}>` }
  }
  /** the deep copy on the way into a pickle (freeze) and out of one */
  const copyOf = (v: Value, memo: Map<unknown, Value>, freezing: boolean): Value => {
    const again = (x: Value) => copyOf(x, memo, freezing)
    const known = memo.get(v.k === 'obj' ? v.o : v)
    if (known) return known
    switch (v.k) {
      case 'list': {
        const out: Value = { k: 'list', items: [], ...(v.view ? { view: v.view } : {}) }
        memo.set(v, out)
        out.items.push(...v.items.map(again))
        return out
      }
      case 'tuple': return { ...v, items: v.items.map(again) }
      case 'dict': {
        const out: Value = { ...v, d: new PyDict(), ...(v.factory ? { factory: again(v.factory) } : {}) }
        memo.set(v, out)
        for (const [k, x] of v.d.items()) out.d.set(again(k), again(x))
        return out
      }
      case 'set': return makeSet(setItems(v).map(again), v.frozen)
      case 'array': return { k: 'array', a: v.a.copy() }
      case 'inst': {
        if (v.cls.meta) return v
        if (h.getAttrOr(v, '__reduce__') || h.getAttrOr(v, '__reduce_ex__')) throw h.err('NotImplementedError', `pickling ${v.cls.name} with its own __reduce__ is not available in this sandbox`)
        const frozen = new Frozen(v)
        const out: Value = { k: 'obj', o: frozen }
        memo.set(v, out)
        const getstate = h.getAttrOr(v, '__getstate__')
        const state = getstate ? h.call(getstate, []) : v.attrs.size ? { k: 'dict', d: PyDict.from([...v.attrs].map(([n, x]) => [S(n), x])) } as Value : NONE
        frozen.held = again(state)
        return out
      }
      case 'fn': case 'type': case 'class': return v
      case 'func':
        if (v.closure) throw h.err('AttributeError', `Can't pickle local object '${v.name}'`)
        if (v.name === '<lambda>') return h.raise(PicklingError, [S("Can't pickle <function <lambda>>: attribute lookup <lambda> on __main__ failed")])
        return v
      case 'bound': return { ...v, self: again(v.self) }
      case 'module': throw cannot('module')
      case 'iter': throw cannot(v.name)
      case 'super': throw cannot('super')
      case 'obj': {
        const o = v.o
        if (o instanceof Frozen) {
          const made: InstValue = { k: 'inst', cls: o.of.cls, attrs: new Map(), state: fieldsCopy(o.jsState, new Map()) as Record<string, unknown> }
          memo.set(o, made)
          const state = again(o.held)
          const setstate = h.getAttrOr(made, '__setstate__')
          if (state.k === 'none') return made
          if (setstate) h.call(setstate, [state])
          else if (state.k === 'dict') for (const [n, x] of state.d.items()) made.attrs.set(h.str(n), x)
          else throw h.err('TypeError', `a ${o.cls} with no __setstate__ cannot take the state ${repr(state)}`)
          return made
        }
        if (o instanceof TextFile && o.path !== null) throw cannot('_io.TextIOWrapper')
        if (o instanceof BinaryFile) throw cannot(`_io.${o.cls}`)
        if (o instanceof Stream) throw cannot('_io.TextIOWrapper')
        if (o instanceof Match) throw cannot('re.Match')
        if (o instanceof Deque) return { k: 'obj', o: new Deque(o.items.map(again), o.maxlen) }
        if (o instanceof PathObj || o instanceof Fraction) return v
        const copied: Value = { k: 'obj', o: fieldsCopy(o, new Map()) as PyObj }
        memo.set(o, copied)
        return copied
      }
    }
    return v
  }
  const fileWrite = (f: Value | undefined) => {
    const write = f ? h.getAttrOr(f, 'write') : undefined
    if (!write) throw h.err('TypeError', "file must have a 'write' attribute")
    return write
  }
  const unpickle = (blob: Pickled) => {
    if (!blob.items.length) throw h.err('EOFError', 'Ran out of input')
    return copyOf(blob.items[0], new Map(), false)
  }
  const makeBlob = (obj: Value, protocol: Value | undefined, kw: Kw) => {
    refuse(h, 'pickle.dumps', kw, ['buffer_callback'])
    if (protocol && protocol.k !== 'none' && h.toInt(protocol) > 5) throw h.err('ValueError', 'pickle protocol must be <= 5')
    return new Pickled([copyOf(obj, new Map(), true)])
  }
  return takingKeywords(mod('pickle', {
    HIGHEST_PROTOCOL: int(5),
    DEFAULT_PROTOCOL: int(4),
    PickleError,
    PicklingError,
    UnpicklingError,
    dumps: F('dumps', (a, kw) => ({ k: 'obj', o: makeBlob(kw.obj ?? a[0], kw.protocol ?? a[1], kw) })),
    loads: F('loads', (a, kw) => {
      refuse(h, 'pickle.loads', kw, ['buffers'])
      const data = kw.data ?? a[0]
      if (data?.k === 'obj' && data.o instanceof Pickled) return unpickle(data.o)
      if (data?.k === 'str') throw h.err('TypeError', "a bytes-like object is required, not 'str'")
      throw h.err('TypeError', `a bytes-like object is required, not '${typeName(data ?? NONE)}'`)
    }),
    dump: F('dump', (a, kw) => {
      const write = fileWrite(kw.file ?? a[1])
      h.call(write, [{ k: 'obj', o: makeBlob(kw.obj ?? a[0], kw.protocol ?? a[2], kw) }])
      return NONE
    }),
    load: F('load', (a, kw) => {
      const f = kw.file ?? a[0]
      if (f?.k === 'obj' && f.o instanceof BinaryFile) {
        const next = f.o.nextPickle(h)
        if (!next) throw h.err('EOFError', 'Ran out of input')
        return copyOf(next, new Map(), false)
      }
      if (f?.k === 'obj' && f.o instanceof TextFile) {
        f.o.canRead(h)
        throw h.err('TypeError', "a bytes-like object is required, not 'str'")
      }
      const read = f ? h.getAttrOr(f, 'read') : undefined
      if (!read) throw h.err('TypeError', "file must have 'read' and 'readline' attributes")
      const data = h.call(read, [])
      if (data.k === 'obj' && data.o instanceof Pickled) return unpickle(data.o)
      throw h.err('TypeError', `a bytes-like object is required, not '${typeName(data)}'`)
    }),
  }), {
    dumps: ['obj', 'protocol', 'fix_imports', 'buffer_callback'], dump: ['obj', 'file', 'protocol', 'fix_imports', 'buffer_callback'],
    loads: ['data', 'fix_imports', 'encoding', 'errors', 'buffers'], load: ['file', 'fix_imports', 'encoding', 'errors', 'buffers'],
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

// ---------------------------------------------------------------- small modules

const makeHeapq = (h: Host): Value => {
  const lt = (a: Value, b: Value) => h.truthy(h.compare('<', a, b))
  const items = (v: Value) => {
    if (v.k !== 'list') throw h.err('TypeError', 'heap argument must be a list')
    return v.items
  }
  const up = (x: Value[], i0: number) => {
    let i = i0
    while (i > 0) {
      const p = (i - 1) >> 1
      if (!lt(x[i], x[p])) break
      ;[x[i], x[p]] = [x[p], x[i]]
      i = p
    }
  }
  const down = (x: Value[], i0: number) => {
    let i = i0
    for (;;) {
      const a = 2 * i + 1
      const b = a + 1
      let m = i
      if (a < x.length && lt(x[a], x[m])) m = a
      if (b < x.length && lt(x[b], x[m])) m = b
      if (m === i) return
      ;[x[i], x[m]] = [x[m], x[i]]
      i = m
    }
  }
  const pop = (x: Value[]) => {
    if (!x.length) throw h.err('IndexError', 'index out of range')
    const top = x[0]
    const last = x.pop()!
    if (x.length) {
      x[0] = last
      down(x, 0)
    }
    return top
  }
  return mod('heapq', {
    heappush: F('heappush', (a) => { const x = items(a[0]); x.push(a[1]); up(x, x.length - 1); return NONE }),
    heappop: F('heappop', (a) => pop(items(a[0]))),
    heapify: F('heapify', (a) => { const x = items(a[0]); for (let i = (x.length >> 1) - 1; i >= 0; i--) down(x, i); return NONE }),
    heappushpop: F('heappushpop', (a) => {
      const x = items(a[0])
      if (x.length && lt(x[0], a[1])) {
        const t = x[0]
        x[0] = a[1]
        down(x, 0)
        return t
      }
      return a[1]
    }),
    heapreplace: F('heapreplace', (a) => {
      const x = items(a[0])
      if (!x.length) throw h.err('IndexError', 'index out of range')
      const t = x[0]
      x[0] = a[1]
      down(x, 0)
      return t
    }),
    nlargest: F('nlargest', (a, kw) => L(h.sorted([...h.iterate(a[1])], kw.key, bool(true)).slice(0, h.toInt(a[0])))),
    nsmallest: F('nsmallest', (a, kw) => L(h.sorted([...h.iterate(a[1])], kw.key).slice(0, h.toInt(a[0])))),
  })
}

const makeBisect = (h: Host): Value => {
  const place = (a: Value[], kw: Kw, right: boolean) => {
    const x = (a[0] as { items: Value[] }).items
    const key = kw.key && kw.key.k !== 'none' ? kw.key : null
    const target = a[1]
    let lo = kw.lo ? h.toInt(kw.lo) : a[2] ? h.toInt(a[2]) : 0
    let hi = kw.hi && kw.hi.k !== 'none' ? h.toInt(kw.hi) : a[3] ? h.toInt(a[3]) : x.length
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      const v = key ? h.call(key, [x[mid]]) : x[mid]
      if (right ? h.truthy(h.compare('<', target, v)) : !h.truthy(h.compare('<', v, target))) hi = mid
      else lo = mid + 1
    }
    return lo
  }
  const find = (name: string, right: boolean) => F(name, (a, kw) => int(place(a, kw, right)))
  const ins = (name: string, right: boolean) => F(name, (a, kw) => {
    ;(a[0] as { items: Value[] }).items.splice(place(a, kw, right), 0, a[1])
    return NONE
  })
  return mod('bisect', { bisect: find('bisect', true), bisect_right: find('bisect_right', true), bisect_left: find('bisect_left', false), insort: ins('insort', true), insort_right: ins('insort_right', true), insort_left: ins('insort_left', false) })
}

const makeString = (): Value => mod('string', {
  ascii_lowercase: S('abcdefghijklmnopqrstuvwxyz'),
  ascii_uppercase: S('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
  ascii_letters: S('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'),
  digits: S('0123456789'),
  hexdigits: S('0123456789abcdefABCDEF'),
  octdigits: S('01234567'),
  punctuation: S('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'),
  whitespace: S(' \t\n\r\u000b\f'),
})

const TYPING = ['Any', 'List', 'Dict', 'Tuple', 'Set', 'FrozenSet', 'Optional', 'Union', 'Callable', 'Iterable', 'Iterator', 'Sequence', 'Mapping', 'Generator', 'Literal', 'Type', 'TypeVar', 'Generic', 'NamedTuple', 'Protocol', 'ClassVar', 'Final', 'Annotated', 'Self', 'NoReturn']

const noop = (name: string, value: Value = NONE): Value => F(name, () => value)

const build = (h: Host): Record<string, Value> => ({
  re: makeRe(h),
  json: makeJson(h),
  collections: makeCollections(h),
  itertools: makeItertools(h),
  functools: makeFunctools(h),
  operator: makeOperator(h),
  statistics: makeStatistics(h),
  dataclasses: makeDataclasses(h),
  copy: makeCopy(h),
  datetime: makeDatetime(h),
  os: makeOs(h),
  'os.path': makeOsPath(h),
  sys: makeSys(h),
  heapq: makeHeapq(h),
  bisect: makeBisect(h),
  string: makeString(),
  typing: mod('typing', Object.fromEntries(TYPING.map((n) => [n, { k: 'obj', o: new TypingAlias(n) } as Value]))),
  __future__: mod('__future__', { annotations: bool(true) }),
  time: mod('time', {
    time: F('time', () => float(Date.now() / 1000)),
    perf_counter: F('perf_counter', () => float(performance.now() / 1000)),
    monotonic: F('monotonic', () => float(performance.now() / 1000)),
    process_time: F('process_time', () => float(performance.now() / 1000)),
    // the sandbox does not really wait: a page must not hang on a sleep
    sleep: noop('sleep'),
    strftime: F('strftime', (a) => S(strftime(new Date(), (a[0] as { v: string }).v))),
  }),
  abc: (() => {
    const base = { k: 'class', name: 'ABC', bases: [], ns: new Map(), host: null, builtin: true } satisfies ClassValue
    return mod('abc', { ABC: base, ABCMeta: base, abstractmethod: F('abstractmethod', (a) => a[0]) })
  })(),
  warnings: mod('warnings', { warn: noop('warn'), filterwarnings: noop('filterwarnings'), simplefilter: noop('simplefilter') }),
  io: makeIo(h),
  csv: makeCsv(h),
  _sandbox_files: makeFiles(h),
  argparse: makeArgparse(h),
  pathlib: makePathlib(h),
  enum: makeEnum(h),
  pickle: makePickle(h),
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
  // a notebook's habits, so its first cell runs
  'google.colab': mod('google.colab', {
    drive: mod('google.colab.drive', {
      mount: F('mount', (a) => {
        const at = h.str(a[0] ?? S('/content/drive'))
        for (const d of [at, `${at}/MyDrive`, `${at}/My Drive`]) dirsOf(h).add(fullPath(d))
        h.print(`Mounted at ${at} (nothing is mounted in this sandbox: it starts with no files)\n`)
        return NONE
      }),
    }),
    files: mod('google.colab.files', {
      upload: F('upload', () => { throw h.err('NotImplementedError', 'files.upload needs Colab: this sandbox starts with no files') }),
    }),
  }),
  google: mod('google', {}),
  tqdm: mod('tqdm', { tqdm: F('tqdm', (a) => a[0]), trange: F('trange', (a) => {
      const [from, to] = a.length > 1 ? [h.toInt(a[0]), h.toInt(a[1])] : [0, h.toInt(a[0])]
      let i = from
      return iter('tqdm', () => (i < to ? int(i++) : undefined))
    }) }),
  'IPython.display': mod('IPython.display', { display: F('display', (a) => { for (const x of a) h.print(`${repr(x)}\n`); return NONE }) }),
  IPython: mod('IPython', {}),
  logging: mod('logging', Object.fromEntries([
    ...['debug', 'info', 'warning', 'error', 'critical', 'basicConfig'].map((n) => [n, noop(n)] as const),
    ['getLogger', F('getLogger', () => mod('logging.Logger', Object.fromEntries(['debug', 'info', 'warning', 'error', 'critical', 'setLevel'].map((n) => [n, noop(n)]))))] as const,
    ...Object.entries({ DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50 }).map(([n, v]) => [n, int(v)] as const),
  ])),
  gc: mod('gc', { collect: noop('collect', int(0)), enable: noop('enable'), disable: noop('disable') }),
  traceback: mod('traceback', { print_exc: noop('print_exc'), format_exc: noop('format_exc', S('')) }),
  pprint: mod('pprint', {
    pprint: F('pprint', (a) => { h.print(`${repr(a[0])}\n`); return NONE }),
    pformat: F('pformat', (a) => S(repr(a[0]))),
  }),
})

return build
}

let load: ReturnType<typeof define> | undefined

export const STDLIB: PyLib = {
  modules: ['re', 'json', 'collections', 'itertools', 'functools', 'operator', 'statistics', 'dataclasses', 'copy', 'datetime', 'os', 'os.path', 'sys', 'heapq', 'bisect', 'string', 'typing', '__future__', 'time', 'abc', 'warnings', 'gc', 'traceback', 'pprint', 'io', 'csv', '_sandbox_files', 'google', 'google.colab', 'tqdm', 'IPython', 'IPython.display', 'logging', 'argparse', 'pathlib', 'enum', 'pickle', 'contextlib', 'textwrap', 'types', 'tempfile', 'glob', 'shutil', 'fnmatch', 'inspect', 'fractions', 'uuid', 'secrets', 'platform', 'getpass'],
  load: (h) => (load ??= define())(h),
}
