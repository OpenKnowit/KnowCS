// The parts of Python's standard library a course's code reaches for, built on the sandbox's own values.
// Nothing here touches files, the network or the clock beyond reading the time.
import { PyDict, PyObj, isNum, makeSet, numOf, py, repr, setItems, typeName } from './minipy'
import type { ClassValue, Host, InstValue, Kw, PyLib, Value } from './minipy'

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

const F = (name: string, call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })
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
        const keyOf = (k: Value) => {
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
  return mod('json', {
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
    JSONDecodeError: decodeError,
  })
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
  getItem() { return { k: 'obj', o: this } as Value }
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
      const s = h.str(a[0])
      if (this.out) h.print(s)
      return int(s.length)
    })
    if (name === 'flush') return F(name, () => NONE)
    return undefined
  }
}

const makeOsPath = (h: Host): Value => {
  const text = (v: Value) => {
    if (v?.k !== 'str') throw h.err('TypeError', `expected str, not ${v ? typeName(v) : 'nothing'}`)
    return v.v
  }
  const no = F('exists', () => bool(false))
  return mod('os.path', {
    join: F('join', (a) => S(a.map(text).reduce((acc, p) => (p.startsWith('/') ? p : acc === '' || acc.endsWith('/') ? acc + p : `${acc}/${p}`)))),
    basename: F('basename', (a) => S(text(a[0]).slice(text(a[0]).lastIndexOf('/') + 1))),
    dirname: F('dirname', (a) => S(text(a[0]).slice(0, Math.max(0, text(a[0]).lastIndexOf('/'))))),
    split: F('split', (a) => {
      const p = text(a[0])
      const at = p.lastIndexOf('/')
      return tuple([S(p.slice(0, Math.max(0, at))), S(p.slice(at + 1))])
    }),
    splitext: F('splitext', (a) => {
      const p = text(a[0])
      const base = p.slice(p.lastIndexOf('/') + 1)
      const at = base.lastIndexOf('.')
      return at <= 0 ? tuple([S(p), S('')]) : tuple([S(p.slice(0, p.length - base.length + at)), S(base.slice(at))])
    }),
    exists: no, isfile: no, isdir: no,
    abspath: F('abspath', (a) => S(text(a[0]).startsWith('/') ? text(a[0]) : `/sandbox/${text(a[0])}`)),
  })
}

const makeOs = (h: Host): Value => {
  const missing = (v: Value | undefined) => h.raise(h.excClass('FileNotFoundError'), [S(`this sandbox has no files: ${v?.k === 'str' ? repr(v) : ''}`)])
  return mod('os', {
    path: makeOsPath(h),
    sep: S('/'),
    linesep: S('\n'),
    environ: { k: 'dict', d: new PyDict() },
    getenv: F('getenv', (a) => a[1] ?? NONE),
    getcwd: F('getcwd', () => S('/sandbox')),
    listdir: F('listdir', (a) => (!a[0] || (a[0].k === 'str' && (a[0].v === '.' || a[0].v === '/sandbox')) ? L([]) : missing(a[0]))),
    cpu_count: F('cpu_count', () => int(1)),
    chdir: F('chdir', (a) => missing(a[0])),
    remove: F('remove', (a) => missing(a[0])),
    makedirs: F('makedirs', () => NONE),
    mkdir: F('mkdir', () => NONE),
  })
}

const makeSys = (h: Host): Value => mod('sys', {
  argv: L([S('main.py')]),
  path: L([]),
  modules: { k: 'dict', d: new PyDict() },
  version: S('3.13 (KnowCS sandbox)'),
  version_info: { k: 'tuple', items: [int(3), int(13), int(0)], names: ['major', 'minor', 'micro'], tname: 'sys.version_info' },
  platform: S('sandbox'),
  maxsize: int(Number.MAX_SAFE_INTEGER),
  stdout: { k: 'obj', o: new Stream(true) },
  stderr: { k: 'obj', o: new Stream(false) },
  exit: F('exit', (a) => h.raise(h.excClass('SystemExit'), a.slice(0, 1))),
  getrecursionlimit: F('getrecursionlimit', () => int(1000)),
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
  modules: ['re', 'json', 'collections', 'itertools', 'functools', 'operator', 'statistics', 'dataclasses', 'copy', 'datetime', 'os', 'os.path', 'sys', 'heapq', 'bisect', 'string', 'typing', '__future__', 'time', 'abc', 'warnings', 'gc', 'traceback', 'pprint'],
  load: (h) => (load ??= define())(h),
}
