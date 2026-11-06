// The parts of Python's standard library a course's code reaches for, built on the sandbox's own values.
// Nothing here touches files, the network or the clock beyond reading the time.
import { PyError } from './ndarray'
import { PyDict, PyObj, isNum, makeSet, py, repr, setItems, typeName } from './minipy'
import type { ClassValue, Host, InstValue, Kw, PyLib, Value } from './minipy'

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
  const memo = (f: Value, maxsize: number | null = null): Value => {
    const seen = new Map<string, Value>()
    let hits = 0
    let misses = 0
    const wrapper: Value & { k: 'fn' } = {
      k: 'fn', name: f.k === 'func' ? f.name : 'cached', bindable: true,
      call: (args, kw) => {
        const key = PyDict.hash(tuple([...args, ...Object.entries(kw).flatMap(([a, b]) => [S(a), b])]))
        const known = seen.get(key)
        if (known) {
          hits++
          // the most recently used stays longest
          seen.delete(key)
          seen.set(key, known)
          return known
        }
        misses++
        const v = h.call(f, args, kw)
        seen.set(key, v)
        if (maxsize !== null && seen.size > maxsize) seen.delete(seen.keys().next().value as string)
        return v
      },
      attrs: new Map<string, Value>([
        ['cache_clear', F('cache_clear', () => { seen.clear(); hits = misses = 0; return NONE })],
        ['cache_info', F('cache_info', () => {
          const CacheInfo = h.call(h.getAttr(h.module('collections'), 'namedtuple'), [S('CacheInfo'), S('hits misses maxsize currsize')])
          return h.call(CacheInfo, [int(hits), int(misses), maxsize === null ? NONE : int(maxsize), int(seen.size)])
        })],
        ['__wrapped__', f]]),
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
      // used bare it is lru_cache(maxsize=128)
      if (a[0] && (a[0].k === 'func' || a[0].k === 'bound')) return memo(a[0], 128)
      const size = kw.maxsize ?? a[0]
      return F('decorator', (b) => memo(b[0], !size ? 128 : size.k === 'none' ? null : h.toInt(size)))
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

// ---------------------------------------------------------------- helpers for library modules

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
/** a parameter that changes the result and is not implemented */
const refuse = (h: Host, fn: string, kw: Kw, names: string[]) => {
  for (const n of names) if (kw[n] && kw[n].k !== 'none' && !(kw[n].k === 'bool' && !kw[n].v)) throw h.err('NotImplementedError', `${fn}(${n}=...) is not available in this sandbox`)
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
  dataclasses: makeDataclasses(h),
  copy: makeCopy(h),
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
  _sandbox_files: makeFiles(h),
  pickle: makePickle(h),
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

// what pyStdlibMore.ts builds on
const parts = {
  guarded, F, iter, fromGen, mod, subclassOf, takingKeywords, refuse, TypingAlias, FieldSpec, pad2, strftime,
  filesOf, dirsOf, fdsOf, CWD, TEMP, isDirAt, isFileAt, namesIn, existsAt, noSuchFile, fileExists, isADir, notADir, checkParents,
  pathOr, fspath, TextFile, Pickled, BinaryFile, NO_BYTES, openFile, textOptions, removeTree, mkdirAt, makedirs, renameAt,
}
return { build, parts }
}

let defined: ReturnType<typeof define> | undefined
const defineOnce = () => (defined ??= define())
export const stdlibParts = () => defineOnce().parts

export const STDLIB: PyLib = {
  modules: ['re', 'json', 'collections', 'itertools', 'functools', 'operator', 'dataclasses', 'copy', 'os', 'os.path', 'sys', 'heapq', 'bisect', 'string', 'typing', '__future__', 'time', 'abc', 'warnings', 'gc', 'traceback', 'pprint', 'io', '_sandbox_files', 'google', 'google.colab', 'tqdm', 'IPython', 'IPython.display', 'logging', 'pickle'],
  load: (h) => defineOnce().build(h),
}
