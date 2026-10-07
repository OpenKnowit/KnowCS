// --- 迷你 Python 解释器：词法 → 语法树 → 求值；所有 ndarray 下标读写都会记录一条 IndexTrace 供可视化 ---
// 支持子集：import numpy as np、赋值 / 增量赋值 / 元组解包、表达式（含比较链、and/or/not、三元表达式）、
// 函数调用与关键字参数、属性、下标与切片。不支持 for / if / def 等语句块。

import {
  NDArray, PyError, MAX_SIZE, arrayRepr, arrayStr, argsort, binaryOp, castValue, dot, expandDims, formatScalar, getIndex, reduce,
  reshape, setIndex, sharesMemory, shapeStr, transpose, unaryOp, checkSize, applyOp, cStrides, broadcastShapes, broadcastTo,
} from './ndarray'
import type { AxisNote, BinOp, DType, IndexItem, IndexPlan, ReduceKind } from './ndarray'

// ================= 词法 =================

type TokType = 'name' | 'num' | 'str' | 'op' | 'nl' | 'eof'
interface Tok { t: TokType; v: string; pos: number; end: number; line: number }

const OPS = [
  '**=', '//=', '...', '**', '//', '==', '!=', '<=', '>=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '@=',
  '+', '-', '*', '/', '%', '@', '&', '|', '^', '~', '<', '>', '=', '(', ')', '[', ']', '{', '}', ',', ':', '.', ';',
]

const syntaxError = (msg: string, line: number) => new PyError('SyntaxError', msg, line)

export const tokenize = (src: string): Tok[] => {
  const toks: Tok[] = []
  let i = 0
  let line = 1
  let depth = 0
  let lineStart = true
  const push = (t: TokType, v: string, pos: number, end = pos + v.length) => toks.push({ t, v, pos, end, line })
  while (i < src.length) {
    const ch = src[i]
    if (lineStart && depth === 0) {
      // 行首缩进检查（没有语句块，任何缩进都是 unexpected indent）
      let j = i
      while (src[j] === ' ' || src[j] === '\t') j++
      const c = src[j]
      if (j > i && c !== undefined && c !== '\n' && c !== '#' && c !== '\r') {
        throw new PyError('IndentationError', 'unexpected indent', line)
      }
      i = j
      lineStart = false
      continue
    }
    if (ch === '\n') {
      if (depth === 0 && toks.length && toks[toks.length - 1].t !== 'nl') push('nl', '\n', i)
      line++
      i++
      lineStart = true
      continue
    }
    if (ch === ' ' || ch === '\t' || ch === '\r') { i++; continue }
    if (ch === '\\' && src[i + 1] === '\n') { i += 2; line++; continue }
    if (ch === '#') {
      while (i < src.length && src[i] !== '\n') i++
      continue
    }
    if (/[0-9]/.test(ch) || (ch === '.' && /[0-9]/.test(src[i + 1] ?? ''))) {
      const m = /^(\d[\d_]*\.?[\d_]*|\.\d[\d_]*)([eE][+-]?\d+)?/.exec(src.slice(i))!
      push('num', m[0].replace(/_/g, ''), i)
      i += m[0].length
      continue
    }
    if (/[A-Za-z_]/.test(ch)) {
      const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i))!
      push('name', m[0], i)
      i += m[0].length
      continue
    }
    if (ch === '"' || ch === "'") {
      let j = i + 1
      let s = ''
      while (j < src.length && src[j] !== ch) {
        if (src[j] === '\n') throw syntaxError('unterminated string literal', line)
        if (src[j] === '\\' && j + 1 < src.length) {
          const e = src[j + 1]
          s += e === 'n' ? '\n' : e === 't' ? '\t' : e
          j += 2
        } else s += src[j++]
      }
      if (j >= src.length) throw syntaxError('unterminated string literal', line)
      push('str', s, i, j + 1)
      i = j + 1
      continue
    }
    const op = OPS.find((o) => src.startsWith(o, i))
    if (!op) throw syntaxError(`invalid character '${ch}'`, line)
    if (op === '(' || op === '[' || op === '{') depth++
    if (op === ')' || op === ']' || op === '}') depth = Math.max(0, depth - 1)
    push('op', op, i)
    i += op.length
  }
  if (depth > 0) throw syntaxError("'(' was never closed", line)
  if (toks.length && toks[toks.length - 1].t !== 'nl') push('nl', '\n', src.length)
  push('eof', '', src.length)
  return toks
}

// ================= 语法树 =================

interface Span { s: number; e: number; line: number }

export type Node = Span & (
  | { k: 'num'; v: number; isFloat: boolean }
  | { k: 'str'; v: string }
  | { k: 'const'; v: 'True' | 'False' | 'None' }
  | { k: 'name'; id: string }
  | { k: 'list'; items: Node[] }
  | { k: 'tuple'; items: Node[] }
  | { k: 'ellipsis' }
  | { k: 'unary'; op: '-' | '+' | '~'; x: Node }
  | { k: 'not'; x: Node }
  | { k: 'bin'; op: string; l: Node; r: Node }
  | { k: 'boolop'; op: 'and' | 'or'; l: Node; r: Node }
  | { k: 'compare'; ops: string[]; xs: Node[] }
  | { k: 'ifexp'; cond: Node; a: Node; b: Node }
  | { k: 'call'; fn: Node; args: Node[]; kw: { name: string; v: Node }[] }
  | { k: 'attr'; obj: Node; name: string }
  | { k: 'sub'; obj: Node; idx: Node }
  | { k: 'slice'; start: Node | null; stop: Node | null; step: Node | null }
)

export type Stmt =
  | { k: 'expr'; x: Node; line: number }
  | { k: 'assign'; targets: Node[]; x: Node; line: number }
  | { k: 'aug'; op: string; target: Node; x: Node; line: number; s: number; e: number }
  | { k: 'import'; alias: string; line: number }
  | { k: 'pass'; line: number }

const UNSUPPORTED = new Set(['for', 'while', 'if', 'def', 'class', 'with', 'try', 'return', 'lambda', 'del', 'global', 'yield', 'elif', 'else', 'except', 'finally'])
const COMPARE_OPS = new Set(['<', '>', '==', '!=', '<=', '>='])

class Parser {
  toks: Tok[]
  i = 0
  constructor(toks: Tok[]) {
    this.toks = toks
  }
  get cur() { return this.toks[this.i] }
  peek(n = 1) { return this.toks[Math.min(this.i + n, this.toks.length - 1)] }
  atLineEnd() { return this.cur.t === 'nl' || this.cur.t === 'eof' }
  isOp(v: string) { return this.cur.t === 'op' && this.cur.v === v }
  isName(v: string) { return this.cur.t === 'name' && this.cur.v === v }
  next() { return this.toks[this.i++] }
  expectOp(v: string) {
    if (!this.isOp(v)) throw this.unexpected()
    return this.next()
  }
  unexpected() {
    const tk = this.cur
    if (tk.t === 'eof' || tk.t === 'nl') return syntaxError('unexpected end of line', tk.line)
    return syntaxError(`invalid syntax near '${tk.v}'`, tk.line)
  }
  endOf(): number { return this.toks[this.i - 1].end }
  mk<T extends Omit<Node, keyof Span>>(start: Tok, n: T): Node {
    return { ...n, s: start.pos, e: this.endOf(), line: start.line } as unknown as Node
  }

  program(): Stmt[] {
    const out: Stmt[] = []
    while (this.cur.t !== 'eof') {
      if (this.cur.t === 'nl' || this.isOp(';')) { this.next(); continue }
      out.push(this.statement())
      if (this.isOp(';')) { this.next(); continue }
      if (!this.atLineEnd()) throw this.unexpected()
    }
    return out
  }

  statement(): Stmt {
    const tk = this.cur
    if (tk.t === 'name' && UNSUPPORTED.has(tk.v)) {
      throw new PyError('NotImplementedError', `'${tk.v}' is not supported in this sandbox — write one statement per line`, tk.line)
    }
    if (this.isName('pass')) { this.next(); return { k: 'pass', line: tk.line } }
    if (this.isName('import')) {
      this.next()
      let mod = this.next().v
      while (this.isOp('.')) { this.next(); mod += '.' + this.next().v }
      if (mod !== 'numpy') throw new PyError('ModuleNotFoundError', `No module named '${mod}' (only numpy is available here)`, tk.line)
      let alias = 'numpy'
      if (this.isName('as')) { this.next(); alias = this.next().v }
      return { k: 'import', alias, line: tk.line }
    }
    if (this.isName('from')) {
      throw new PyError('NotImplementedError', "use 'import numpy as np' in this sandbox", tk.line)
    }
    const first = this.testList()
    if (this.cur.t === 'op' && /^(\*\*|\/\/|[-+*/%&|^@])=$/.test(this.cur.v)) {
      const op = this.next().v.slice(0, -1)
      if (first.k !== 'name' && first.k !== 'sub') throw syntaxError("'illegal expression for augmented assignment'", tk.line)
      const x = this.testList()
      return { k: 'aug', op, target: first, x, line: tk.line, s: tk.pos, e: this.endOf() }
    }
    if (this.isOp('=')) {
      const targets = [first]
      let x: Node = first
      while (this.isOp('=')) {
        this.next()
        x = this.testList()
        targets.push(x)
      }
      targets.pop()
      for (const t of targets) this.checkTarget(t)
      return { k: 'assign', targets, x, line: tk.line }
    }
    return { k: 'expr', x: first, line: tk.line }
  }

  checkTarget(t: Node) {
    if (t.k === 'name' || t.k === 'sub') return
    if ((t.k === 'tuple' || t.k === 'list') && t.items.every((x) => x.k === 'name' || x.k === 'sub')) return
    throw syntaxError('cannot assign to expression', t.line)
  }

  testList(): Node {
    const start = this.cur
    const first = this.test()
    if (!this.isOp(',')) return first
    const items = [first]
    while (this.isOp(',')) {
      this.next()
      if (this.cur.t === 'nl' || this.cur.t === 'eof' || this.isOp('=') || this.isOp(';')) break
      items.push(this.test())
    }
    return this.mk(start, { k: 'tuple', items })
  }

  test(): Node {
    const start = this.cur
    const a = this.orTest()
    if (this.isName('if')) {
      this.next()
      const cond = this.orTest()
      if (!this.isName('else')) throw this.unexpected()
      this.next()
      const b = this.test()
      return this.mk(start, { k: 'ifexp', cond, a, b })
    }
    return a
  }

  orTest(): Node {
    const start = this.cur
    let l = this.andTest()
    while (this.isName('or')) { this.next(); l = this.mk(start, { k: 'boolop', op: 'or', l, r: this.andTest() }) }
    return l
  }

  andTest(): Node {
    const start = this.cur
    let l = this.notTest()
    while (this.isName('and')) { this.next(); l = this.mk(start, { k: 'boolop', op: 'and', l, r: this.notTest() }) }
    return l
  }

  notTest(): Node {
    const start = this.cur
    if (this.isName('not')) { this.next(); return this.mk(start, { k: 'not', x: this.notTest() }) }
    return this.comparison()
  }

  comparison(): Node {
    const start = this.cur
    const first = this.bitOr()
    const ops: string[] = []
    const xs = [first]
    for (;;) {
      if (this.cur.t === 'op' && COMPARE_OPS.has(this.cur.v)) ops.push(this.next().v)
      else if (this.isName('in')) { this.next(); ops.push('in') }
      else if (this.isName('not') && this.peek().t === 'name' && this.peek().v === 'in') { this.i += 2; ops.push('not in') }
      else if (this.isName('is')) {
        this.next()
        if (this.isName('not')) { this.next(); ops.push('is not') } else ops.push('is')
      } else break
      xs.push(this.bitOr())
    }
    return ops.length ? this.mk(start, { k: 'compare', ops, xs }) : first
  }

  binLevel(ops: string[], nextLevel: () => Node): Node {
    const start = this.cur
    let l = nextLevel()
    while (this.cur.t === 'op' && ops.includes(this.cur.v)) {
      const op = this.next().v
      l = this.mk(start, { k: 'bin', op, l, r: nextLevel() })
    }
    return l
  }

  bitOr(): Node { return this.binLevel(['|'], () => this.bitXor()) }
  bitXor(): Node { return this.binLevel(['^'], () => this.bitAnd()) }
  bitAnd(): Node { return this.binLevel(['&'], () => this.arith()) }
  arith(): Node { return this.binLevel(['+', '-'], () => this.term()) }
  term(): Node { return this.binLevel(['*', '/', '//', '%', '@'], () => this.factor()) }

  factor(): Node {
    const start = this.cur
    if (this.isOp('-') || this.isOp('+') || this.isOp('~')) {
      const op = this.next().v as '-' | '+' | '~'
      return this.mk(start, { k: 'unary', op, x: this.factor() })
    }
    return this.power()
  }

  power(): Node {
    const start = this.cur
    const base = this.postfix()
    if (this.isOp('**')) {
      this.next()
      return this.mk(start, { k: 'bin', op: '**', l: base, r: this.factor() })
    }
    return base
  }

  postfix(): Node {
    const start = this.cur
    let x = this.atom()
    for (;;) {
      if (this.isOp('(')) {
        this.next()
        const args: Node[] = []
        const kw: { name: string; v: Node }[] = []
        while (!this.isOp(')')) {
          if (this.isOp('*')) throw new PyError('NotImplementedError', '*args is not supported in this sandbox', this.cur.line)
          if (this.cur.t === 'name' && this.peek().t === 'op' && this.peek().v === '=') {
            const name = this.next().v
            this.next()
            kw.push({ name, v: this.test() })
          } else {
            if (kw.length) throw syntaxError('positional argument follows keyword argument', this.cur.line)
            args.push(this.test())
          }
          if (!this.isOp(')')) this.expectOp(',')
        }
        this.next()
        x = this.mk(start, { k: 'call', fn: x, args, kw })
      } else if (this.isOp('[')) {
        this.next()
        const idx = this.subscriptList()
        this.expectOp(']')
        x = this.mk(start, { k: 'sub', obj: x, idx })
      } else if (this.isOp('.')) {
        this.next()
        if (this.cur.t !== 'name') throw this.unexpected()
        x = this.mk(start, { k: 'attr', obj: x, name: this.next().v })
      } else return x
    }
  }

  subscriptList(): Node {
    const start = this.cur
    const first = this.subscript()
    if (!this.isOp(',')) return first
    const items = [first]
    while (this.isOp(',')) {
      this.next()
      if (this.isOp(']')) break
      items.push(this.subscript())
    }
    return this.mk(start, { k: 'tuple', items })
  }

  subscript(): Node {
    const start = this.cur
    const part = () => (this.isOp(':') || this.isOp(']') || this.isOp(',') ? null : this.test())
    const a = part()
    if (!this.isOp(':')) {
      if (!a) throw this.unexpected()
      return a
    }
    this.next()
    const b = part()
    let c: Node | null = null
    if (this.isOp(':')) { this.next(); c = part() }
    return this.mk(start, { k: 'slice', start: a, stop: b, step: c })
  }

  atom(): Node {
    const tk = this.cur
    if (tk.t === 'num') {
      this.next()
      const isFloat = /[.eE]/.test(tk.v)
      return this.mk(tk, { k: 'num', v: Number(tk.v), isFloat })
    }
    if (tk.t === 'str') {
      this.next()
      let v = tk.v
      while (this.cur.t === 'str') v += this.next().v
      return this.mk(tk, { k: 'str', v })
    }
    if (tk.t === 'name') {
      if (UNSUPPORTED.has(tk.v)) throw new PyError('NotImplementedError', `'${tk.v}' is not supported in this sandbox`, tk.line)
      this.next()
      if (tk.v === 'True' || tk.v === 'False' || tk.v === 'None') return this.mk(tk, { k: 'const', v: tk.v })
      return this.mk(tk, { k: 'name', id: tk.v })
    }
    if (this.isOp('...')) { this.next(); return this.mk(tk, { k: 'ellipsis' }) }
    if (this.isOp('(')) {
      this.next()
      if (this.isOp(')')) { this.next(); return this.mk(tk, { k: 'tuple', items: [] }) }
      const first = this.test()
      if (this.isOp(')')) { this.next(); return { ...first, s: tk.pos, e: this.endOf() } }
      const items = [first]
      while (this.isOp(',')) {
        this.next()
        if (this.isOp(')')) break
        items.push(this.test())
      }
      this.expectOp(')')
      return this.mk(tk, { k: 'tuple', items })
    }
    if (this.isOp('[')) {
      this.next()
      const items: Node[] = []
      while (!this.isOp(']')) {
        items.push(this.test())
        if (this.isName('for')) throw new PyError('NotImplementedError', 'list comprehensions are not supported in this sandbox', tk.line)
        if (!this.isOp(']')) this.expectOp(',')
      }
      this.next()
      return this.mk(tk, { k: 'list', items })
    }
    throw this.unexpected()
  }
}

export const parse = (src: string): Stmt[] => new Parser(tokenize(src)).program()

// ================= 运行时值 =================

type Fn = (args: Value[], kw: Record<string, Value>) => Value

export type Value =
  | { k: 'int'; v: number }
  | { k: 'float'; v: number }
  | { k: 'bool'; v: boolean }
  | { k: 'none' }
  | { k: 'str'; v: string }
  | { k: 'list'; items: Value[] }
  | { k: 'tuple'; items: Value[] }
  | { k: 'slice'; start: Value; stop: Value; step: Value }
  | { k: 'ellipsis' }
  | { k: 'array'; a: NDArray }
  | { k: 'fn'; name: string; call: Fn }
  | { k: 'type'; name: string; call: Fn; dtype?: DType }
  | { k: 'module'; name: string; attrs: Record<string, Value> }
  | { k: 'dtype'; d: DType }

const NONE: Value = { k: 'none' }
const int = (v: number): Value => ({ k: 'int', v })
const float = (v: number): Value => ({ k: 'float', v })
const bool = (v: boolean): Value => ({ k: 'bool', v })
const arr = (a: NDArray): Value => ({ k: 'array', a })
const tuple = (items: Value[]): Value => ({ k: 'tuple', items })

const typeName = (v: Value): string => {
  switch (v.k) {
    case 'none': return 'NoneType'
    case 'array': return 'numpy.ndarray'
    case 'fn': return 'builtin_function_or_method'
    case 'type': return 'type'
    case 'dtype': return 'numpy.dtype'
    default: return v.k
  }
}

const fromScalar = (x: number, dtype: DType): Value =>
  dtype === 'bool' ? bool(x !== 0) : dtype === 'int64' ? int(x) : float(x)

type NumValue = Extract<Value, { k: 'int' | 'float' | 'bool' }>
const isNum = (v: Value): v is NumValue => v.k === 'int' || v.k === 'float' || v.k === 'bool'
const numOf = (v: Value): number => (v.k === 'bool' ? (v.v ? 1 : 0) : v.k === 'int' || v.k === 'float' ? v.v : NaN)
const scalarDType = (v: Value): DType => (v.k === 'bool' ? 'bool' : v.k === 'int' ? 'int64' : 'float64')

const reprStr = (s: string) => (s.includes("'") && !s.includes('"') ? `"${s}"` : `'${s.replace(/'/g, "\\'")}'`)

export const repr = (v: Value): string => {
  switch (v.k) {
    case 'int': return String(v.v)
    case 'float': return formatScalar(v.v, 'float64')
    case 'bool': return v.v ? 'True' : 'False'
    case 'none': return 'None'
    case 'str': return reprStr(v.v)
    case 'list': return `[${v.items.map(repr).join(', ')}]`
    case 'tuple': return v.items.length === 1 ? `(${repr(v.items[0])},)` : `(${v.items.map(repr).join(', ')})`
    case 'slice': return `slice(${repr(v.start)}, ${repr(v.stop)}, ${repr(v.step)})`
    case 'ellipsis': return 'Ellipsis'
    case 'array': return arrayRepr(v.a)
    case 'fn': return `<built-in function ${v.name}>`
    case 'type': return `<class '${v.name}'>`
    case 'module': return `<module '${v.name}'>`
    case 'dtype': return `dtype('${v.d}')`
  }
}

export const str = (v: Value): string => {
  if (v.k === 'str') return v.v
  if (v.k === 'array') return arrayStr(v.a)
  if (v.k === 'dtype') return v.d
  return repr(v)
}

/** Python 值（嵌套 list / 标量 / 数组）→ NDArray */
export const toArray = (v: Value, dtype?: DType): NDArray => {
  if (v.k === 'array') return dtype && dtype !== v.a.dtype ? NDArray.create(v.a.values(), v.a.shape, dtype) : v.a
  const flat: number[] = []
  let kind: DType = 'bool'
  const promote = (d: DType) => {
    if (d === 'float64' || (d === 'int64' && kind === 'bool')) kind = d
  }
  const shapeOf = (x: Value, prefix: number[]): number[] => {
    if (x.k === 'list' || x.k === 'tuple') {
      const here = [...prefix, x.items.length]
      const shapes = x.items.map((it) => shapeOf(it, here))
      const s0 = JSON.stringify(shapes[0] ?? [])
      if (shapes.some((s) => JSON.stringify(s) !== s0)) {
        throw new PyError('ValueError', `setting an array element with a sequence. The requested array has an inhomogeneous shape after ${here.length} dimensions. The detected shape was ${shapeStr(here)} + inhomogeneous part.`)
      }
      return [x.items.length, ...(shapes[0] ?? [])]
    }
    if (x.k === 'array') return x.a.shape
    if (isNum(x)) return []
    throw new PyError('TypeError', `cannot convert '${typeName(x)}' to a numeric array`)
  }
  const fill = (x: Value) => {
    if (x.k === 'list' || x.k === 'tuple') x.items.forEach(fill)
    else if (x.k === 'array') {
      promote(x.a.dtype)
      flat.push(...x.a.values())
    } else {
      promote(scalarDType(x))
      flat.push(numOf(x))
    }
  }
  const shape = shapeOf(v, [])
  fill(v)
  if ((v.k === 'list' || v.k === 'tuple') && flat.length === 0) kind = 'float64'
  checkSize(shape)
  return NDArray.create(flat, shape, dtype ?? kind)
}

// ================= 索引追踪 =================

export interface GridSnapshot {
  shape: number[]
  values: number[]
  dtype: DType
}

export interface AliasEffect {
  name: string
  after: GridSnapshot
  /** 被改动的元素（C 顺序扁平下标） */
  changed: number[]
}

export interface MemoryInfo {
  /** 底层缓冲区（写入前） */
  buffer: number[]
  dtype: DType
  /** 源数组覆盖的缓冲区地址 */
  sourceAddrs: number[]
  /** 结果每个元素指向的缓冲区地址 */
  picked: number[]
}

export interface IndexTrace {
  id: number
  line: number
  code: string
  target: string
  mode: 'read' | 'write'
  result: 'view' | 'copy' | 'scalar' | 'write'
  source: GridSnapshot
  after?: GridSnapshot
  out: GridSnapshot
  /** 结果第 i 个元素来自源数组的哪个扁平下标 */
  srcFlat: number[]
  notes: AxisNote[]
  hasMask: boolean
  hasFancy: boolean
  advFront: boolean
  view?: { offset: number; strides: number[] }
  mask?: { snapshot: GridSnapshot; axes: number[] }
  memory: MemoryInfo
  aliases: AliasEffect[]
}

export interface VarInfo {
  name: string
  kind: string
  shape?: number[]
  dtype?: DType
  sharesWith: string[]
  isView: boolean
  preview: string
}

export interface RunResult {
  stdout: string
  out: string | null
  error: { type: string; message: string; line?: number } | null
  traces: IndexTrace[]
  vars: VarInfo[]
}

const snap = (a: NDArray): GridSnapshot => ({ shape: [...a.shape], values: a.values(), dtype: a.dtype })

// ================= 解释器 =================

class Interp {
  src: string
  env = new Map<string, Value>()
  stdout: string[] = []
  traces: IndexTrace[] = []
  line = 0
  quiet = 0
  builtins: Record<string, Value>
  np: Value

  constructor(src: string) {
    this.src = src
    this.np = this.makeNumpy()
    this.builtins = this.makeBuiltins()
  }

  err(type: string, msg: string): PyError {
    return new PyError(type, msg, this.line)
  }

  text(n: Span) {
    return this.src.slice(n.s, n.e)
  }

  // ---------- 基本协议 ----------

  truthy(v: Value): boolean {
    switch (v.k) {
      case 'bool': return v.v
      case 'int': case 'float': return v.v !== 0
      case 'none': return false
      case 'str': return v.v.length > 0
      case 'list': case 'tuple': return v.items.length > 0
      case 'array':
        if (v.a.size === 1) return v.a.values()[0] !== 0
        if (v.a.size === 0) throw this.err('ValueError', 'The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.')
        throw this.err('ValueError', 'The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()')
      default: return true
    }
  }

  toInt(v: Value, what = 'an integer'): number {
    if (v.k === 'int') return v.v
    if (v.k === 'bool') return v.v ? 1 : 0
    if (v.k === 'array' && v.a.ndim === 0 && v.a.dtype !== 'float64') return v.a.values()[0]
    throw this.err('TypeError', `'${typeName(v)}' object cannot be interpreted as ${what}`)
  }

  toShape(args: Value[]): number[] {
    const src = args.length === 1 && (args[0].k === 'tuple' || args[0].k === 'list') ? args[0].items : args
    return src.map((x) => this.toInt(x))
  }

  binop(op: string, l: Value, r: Value): Value {
    if (op === '@') {
      if (l.k !== 'array' && r.k !== 'array') throw this.err('TypeError', `unsupported operand type(s) for @: '${typeName(l)}' and '${typeName(r)}'`)
      if (toArray(l).ndim === 0 || toArray(r).ndim === 0) throw this.err('ValueError', 'matmul: Input operand does not have enough dimensions')
      const res = dot(toArray(l), toArray(r))
      return typeof res === 'number' ? fromScalar(res, toArray(l).dtype === 'float64' || toArray(r).dtype === 'float64' ? 'float64' : 'int64') : arr(res)
    }
    if (l.k === 'array' || r.k === 'array') {
      const la = this.operand(l, op)
      const ra = this.operand(r, op)
      return arr(binaryOp(op as BinOp, la, ra))
    }
    if (isNum(l) && isNum(r)) return this.scalarOp(op, l, r)
    if (op === '+' && l.k === r.k && (l.k === 'str' || l.k === 'list' || l.k === 'tuple')) {
      if (l.k === 'str') return { k: 'str', v: l.v + (r as { v: string }).v }
      return { k: l.k, items: [...l.items, ...(r as { items: Value[] }).items] } as Value
    }
    if (op === '*' && (l.k === 'list' || l.k === 'tuple' || l.k === 'str') && (r.k === 'int' || r.k === 'bool')) {
      const n = Math.max(0, numOf(r))
      if (l.k === 'str') return { k: 'str', v: l.v.repeat(n) }
      return { k: l.k, items: Array.from({ length: n }, () => l.items).flat() } as Value
    }
    if (op === '+' && (l.k === 'list' || l.k === 'tuple' || l.k === 'str')) {
      const name = l.k === 'str' ? 'str' : l.k
      throw this.err('TypeError', `can only concatenate ${name} (not "${typeName(r)}") to ${name}`)
    }
    if (op === '==' || op === '!=') {
      const eq = repr(l) === repr(r)
      return bool(op === '==' ? eq : !eq)
    }
    throw this.err('TypeError', `unsupported operand type(s) for ${op}: '${typeName(l)}' and '${typeName(r)}'`)
  }

  operand(v: Value, op: string): NDArray {
    if (v.k === 'array' || isNum(v) || v.k === 'list' || v.k === 'tuple') return toArray(v)
    throw this.err('TypeError', `unsupported operand type(s) for ${op}: 'numpy.ndarray' and '${typeName(v)}'`)
  }

  scalarOp(op: string, l: Value, r: Value): Value {
    const x = numOf(l)
    const y = numOf(r)
    const isFloat = l.k === 'float' || r.k === 'float'
    if (['<', '>', '==', '!=', '<=', '>='].includes(op)) return bool(applyOp(op as BinOp, x, y, 'float64') === 1)
    if (op === '&' || op === '|' || op === '^') {
      if (isFloat) throw this.err('TypeError', `unsupported operand type(s) for ${op}: '${typeName(l)}' and '${typeName(r)}'`)
      const v = applyOp(op, x, y, 'int64')
      return l.k === 'bool' && r.k === 'bool' ? bool(v !== 0) : int(v)
    }
    if (op === '/' || op === '//' || op === '%') {
      if (y === 0) {
        throw this.err('ZeroDivisionError', op === '/' ? (isFloat ? 'float division by zero' : 'division by zero') : isFloat ? 'float modulo by zero' : 'integer division or modulo by zero')
      }
      if (op === '/') return float(x / y)
    }
    if (op === '**' && !isFloat && y < 0) return float(x ** y)
    const v = applyOp(op as BinOp, x, y, isFloat ? 'float64' : 'int64')
    return isFloat ? float(v) : int(v)
  }

  compare(op: string, l: Value, r: Value): Value {
    if (op === 'is' || op === 'is not') {
      const same =
        l.k === r.k &&
        (l.k === 'array' ? l.a === (r as { a: NDArray }).a : l.k === 'none' || l.k === 'ellipsis' || repr(l) === repr(r))
      return bool(op === 'is' ? same : !same)
    }
    if (op === 'in' || op === 'not in') {
      let found: boolean
      if (r.k === 'list' || r.k === 'tuple') found = r.items.some((x) => this.truthy(this.binop('==', x, l)))
      else if (r.k === 'str' && l.k === 'str') found = r.v.includes(l.v)
      else if (r.k === 'array') found = toArray(this.binop('==', r, l)).values().some((x) => x !== 0)
      else throw this.err('TypeError', `argument of type '${typeName(r)}' is not iterable`)
      return bool(op === 'in' ? found : !found)
    }
    return this.binop(op, l, r)
  }

  // ---------- 下标 ----------

  toIndexItems(v: Value): IndexItem[] {
    const items = v.k === 'tuple' ? v.items : [v]
    return items.map((it): IndexItem => {
      switch (it.k) {
        case 'int': return { kind: 'int', value: it.v }
        case 'none': return { kind: 'newaxis' }
        case 'ellipsis': return { kind: 'ellipsis' }
        case 'slice': {
          const part = (x: Value) => (x.k === 'none' ? null : this.sliceInt(x))
          return { kind: 'slice', start: part(it.start), stop: part(it.stop), step: part(it.step) }
        }
        case 'list': case 'tuple': case 'array': {
          const a = toArray(it.k === 'list' && it.items.length === 0 ? { k: 'array', a: NDArray.create([], [0], 'int64') } : it)
          if (a.dtype === 'float64') throw this.err('IndexError', 'arrays used as indices must be of integer (or boolean) type')
          return { kind: 'array', arr: a }
        }
        case 'bool': return { kind: 'array', arr: NDArray.create([it.v ? 1 : 0], [], 'bool') }
        default:
          throw this.err('IndexError', 'only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices')
      }
    })
  }

  sliceInt(x: Value): number {
    if (x.k === 'int' || x.k === 'bool') return numOf(x)
    if (x.k === 'array' && x.a.ndim === 0 && x.a.dtype !== 'float64') return x.a.values()[0]
    throw this.err('TypeError', 'slice indices must be integers or None or have an __index__ method')
  }

  seqIndex(seq: Value[], idx: Value, kind: string): Value | Value[] {
    if (idx.k === 'int' || idx.k === 'bool') {
      const n = seq.length
      const i = numOf(idx)
      if (i < -n || i >= n) throw this.err('IndexError', `${kind} index out of range`)
      return seq[i < 0 ? i + n : i]
    }
    if (idx.k === 'slice') {
      const a = toArray(arr(NDArray.create(seq.map((_, i) => i), [seq.length], 'int64')))
      const it = this.toIndexItems(idx)
      const { value } = getIndex(a, it)
      return (value as NDArray).values().map((i) => seq[i])
    }
    throw this.err('TypeError', `${kind} indices must be integers or slices, not ${typeName(idx)}`)
  }

  getItem(node: Node & { k: 'sub' }, obj: Value, idx: Value): Value {
    if (obj.k === 'array') {
      const a = obj.a
      const before = snap(a)
      const items = this.toIndexItems(idx)
      const { plan, value } = getIndex(a, items)
      const res: Value = typeof value === 'number' ? fromScalar(value, a.dtype) : arr(value)
      if (!this.quiet) {
        const out = typeof value === 'number' ? { shape: [], values: [value], dtype: a.dtype } : snap(value)
        this.record(this.targetOf(node), this.text(node), a, before, plan, 'read', out)
      }
      return res
    }
    if (obj.k === 'list' || obj.k === 'tuple') {
      const r = this.seqIndex(obj.items, idx, obj.k)
      return Array.isArray(r) ? ({ k: obj.k, items: r } as Value) : r
    }
    if (obj.k === 'str') {
      const r = this.seqIndex([...obj.v].map((c) => ({ k: 'str', v: c }) as Value), idx, 'string')
      return Array.isArray(r) ? { k: 'str', v: r.map((c) => (c as { v: string }).v).join('') } : r
    }
    throw this.err('TypeError', `'${typeName(obj)}' object is not subscriptable`)
  }

  setItem(node: Node & { k: 'sub' }, obj: Value, idx: Value, val: Value, codeNode?: Span) {
    if (obj.k === 'list') {
      if (idx.k !== 'int') throw this.err('TypeError', 'list indices must be integers or slices, not ' + typeName(idx))
      const n = obj.items.length
      if (idx.v < -n || idx.v >= n) throw this.err('IndexError', 'list assignment index out of range')
      obj.items[idx.v < 0 ? idx.v + n : idx.v] = val
      return
    }
    if (obj.k !== 'array') throw this.err('TypeError', `'${typeName(obj)}' object does not support item assignment`)
    const a = obj.a
    const before = snap(a)
    const memBefore = [...a.data]
    const plan = setIndex(a, this.toIndexItems(idx), this.operand(val, '='))
    const written = { shape: plan.shape, values: plan.addresses.map((x) => a.data[x]), dtype: a.dtype }
    this.record(this.targetOf(node), this.text(codeNode ?? node), a, before, plan, 'write', written, memBefore)
  }

  /** 原地增量赋值 b += 1：b 是视图时同样会改到原数组 */
  inplace(name: string, a: NDArray, op: string, rhs: Value, stmt: Span) {
    const res = binaryOp(op as BinOp, a, this.operand(rhs, op + '='))
    if (a.dtype !== 'float64' && res.dtype === 'float64') {
      throw this.err('UFuncTypeError', `Cannot cast ufunc '${op === '/' ? 'divide' : op}' output from dtype('float64') to dtype('${a.dtype}') with casting rule 'same_kind'`)
    }
    if (res.size !== a.size) {
      throw this.err('ValueError', `non-broadcastable output operand with shape ${shapeStr(a.shape)} doesn't match the broadcast shape ${shapeStr(res.shape)}`)
    }
    const before = snap(a)
    const memBefore = [...a.data]
    const addrs = a.addresses()
    const vals = res.values()
    addrs.forEach((x, i) => { a.data[x] = castValue(vals[i], a.dtype) })
    const plan: IndexPlan = {
      shape: [...a.shape], addresses: addrs, advanced: false, scalar: false, advFront: false,
      notes: [], view: { shape: a.shape, strides: a.strides, offset: a.offset },
    }
    this.record(name, this.text(stmt), a, before, plan, 'write', snap(a), memBefore)
  }

  targetOf(node: Node & { k: 'sub' }): string {
    return node.obj.k === 'name' ? node.obj.id : this.text(node.obj)
  }

  record(target: string, code: string, a: NDArray, before: GridSnapshot, plan: IndexPlan, mode: 'read' | 'write', out: GridSnapshot, memBefore?: number[]) {
    // 源数组：地址 → 扁平下标
    const srcAddrs = a.addresses()
    const flatOf = new Map(srcAddrs.map((x, i) => [x, i]))
    const aliases: AliasEffect[] = []
    if (mode === 'write') {
      const written = new Set(plan.addresses)
      for (const [name, v] of this.env) {
        if (v.k !== 'array' || v.a === a || name === target || v.a.data !== a.data) continue
        const addrs = v.a.addresses()
        const changed = addrs.flatMap((x, i) => (written.has(x) && memBefore && memBefore[x] !== a.data[x] ? [i] : []))
        if (changed.length) aliases.push({ name, after: snap(v.a), changed })
      }
    }
    const result: IndexTrace['result'] = mode === 'write' ? 'write' : plan.scalar ? 'scalar' : plan.advanced ? 'copy' : 'view'
    this.traces.push({
      id: this.traces.length,
      line: this.line,
      code,
      target,
      mode,
      result,
      source: before,
      after: mode === 'write' ? snap(a) : undefined,
      out,
      srcFlat: plan.addresses.map((x) => flatOf.get(x)!),
      notes: plan.notes,
      hasMask: plan.notes.some((n) => n.kind === 'mask'),
      hasFancy: plan.notes.some((n) => n.kind === 'fancy'),
      advFront: plan.advFront,
      view: plan.view ? { offset: plan.view.offset, strides: plan.view.strides } : undefined,
      mask: plan.mask ? { snapshot: snap(plan.mask.arr), axes: plan.mask.axes } : undefined,
      memory: { buffer: memBefore ?? [...a.data], dtype: a.dtype, sourceAddrs: srcAddrs, picked: plan.addresses },
      aliases,
    })
  }

  // ---------- 求值 ----------

  eval(n: Node): Value {
    switch (n.k) {
      case 'num': return n.isFloat ? float(n.v) : int(n.v)
      case 'str': return { k: 'str', v: n.v }
      case 'const': return n.v === 'None' ? NONE : bool(n.v === 'True')
      case 'ellipsis': return { k: 'ellipsis' }
      case 'name': {
        const v = this.env.get(n.id) ?? this.builtins[n.id]
        if (!v) {
          if (n.id === 'np' || n.id === 'numpy') throw this.err('NameError', `name '${n.id}' is not defined. Did you forget 'import numpy as np'?`)
          throw this.err('NameError', `name '${n.id}' is not defined`)
        }
        return v
      }
      case 'list': return { k: 'list', items: n.items.map((x) => this.eval(x)) }
      case 'tuple': return tuple(n.items.map((x) => this.eval(x)))
      case 'unary': {
        const x = this.eval(n.x)
        if (x.k === 'array') return arr(unaryOp(n.op, x.a))
        if (!isNum(x)) throw this.err('TypeError', `bad operand type for unary ${n.op}: '${typeName(x)}'`)
        if (n.op === '~') {
          if (x.k === 'float') throw this.err('TypeError', "bad operand type for unary ~: 'float'")
          return int(~numOf(x))
        }
        const v = n.op === '-' ? -numOf(x) : numOf(x)
        return x.k === 'float' ? float(v) : int(v || 0)
      }
      case 'not': return bool(!this.truthy(this.eval(n.x)))
      case 'boolop': {
        const l = this.eval(n.l)
        const t = this.truthy(l)
        if (n.op === 'and') return t ? this.eval(n.r) : l
        return t ? l : this.eval(n.r)
      }
      case 'compare': {
        let l = this.eval(n.xs[0])
        let result: Value = bool(true)
        for (let i = 0; i < n.ops.length; i++) {
          const r = this.eval(n.xs[i + 1])
          result = this.compare(n.ops[i], l, r)
          if (i < n.ops.length - 1 && !this.truthy(result)) return result
          l = r
        }
        return result
      }
      case 'ifexp': return this.truthy(this.eval(n.cond)) ? this.eval(n.a) : this.eval(n.b)
      case 'bin': return this.binop(n.op, this.eval(n.l), this.eval(n.r))
      case 'attr': return this.getAttr(this.eval(n.obj), n.name)
      case 'call': {
        const f = this.eval(n.fn)
        const args = n.args.map((x) => this.eval(x))
        const kw: Record<string, Value> = {}
        for (const { name, v } of n.kw) kw[name] = this.eval(v)
        if (f.k !== 'fn' && f.k !== 'type') throw this.err('TypeError', `'${typeName(f)}' object is not callable`)
        return f.call(args, kw)
      }
      case 'sub': return this.getItem(n, this.eval(n.obj), this.eval(n.idx))
      case 'slice':
        return {
          k: 'slice',
          start: n.start ? this.eval(n.start) : NONE,
          stop: n.stop ? this.eval(n.stop) : NONE,
          step: n.step ? this.eval(n.step) : NONE,
        }
    }
  }

  assign(t: Node, v: Value, stmtSpan?: Span) {
    if (t.k === 'name') {
      this.env.set(t.id, v)
    } else if (t.k === 'sub') {
      this.setItem(t, this.eval(t.obj), this.eval(t.idx), v, stmtSpan)
    } else if (t.k === 'tuple' || t.k === 'list') {
      const items = this.iterate(v)
      if (items.length !== t.items.length) {
        throw this.err('ValueError', items.length > t.items.length ? `too many values to unpack (expected ${t.items.length})` : `not enough values to unpack (expected ${t.items.length}, got ${items.length})`)
      }
      t.items.forEach((x, i) => this.assign(x, items[i]))
    }
  }

  iterate(v: Value): Value[] {
    if (v.k === 'list' || v.k === 'tuple') return v.items
    if (v.k === 'str') return [...v.v].map((c) => ({ k: 'str', v: c }))
    if (v.k === 'array') {
      if (v.a.ndim === 0) throw this.err('TypeError', 'iteration over a 0-d array')
      return Array.from({ length: v.a.shape[0] }, (_, i) => {
        const { value } = getIndex(v.a, [{ kind: 'int', value: i }])
        return typeof value === 'number' ? fromScalar(value, v.a.dtype) : arr(value)
      })
    }
    throw this.err('TypeError', `'${typeName(v)}' object is not iterable`)
  }

  exec(stmts: Stmt[]): string | null {
    let out: string | null = null
    stmts.forEach((st, i) => {
      this.line = st.line
      out = null
      switch (st.k) {
        case 'pass': break
        case 'import': this.env.set(st.alias, this.np); break
        case 'expr': {
          const v = this.eval(st.x)
          if (i === stmts.length - 1 && v.k !== 'none') out = repr(v)
          break
        }
        case 'assign': {
          const v = this.eval(st.x)
          for (const t of st.targets) this.assign(t, v, { s: t.s, e: st.x.e, line: st.line })
          break
        }
        case 'aug': {
          const span = { s: st.s, e: st.e, line: st.line }
          if (st.target.k === 'name') {
            const cur = this.eval(st.target)
            const rhs = this.eval(st.x)
            if (cur.k === 'array') this.inplace(st.target.id, cur.a, st.op, rhs, span)
            else this.env.set(st.target.id, this.binop(st.op, cur, rhs))
          } else if (st.target.k === 'sub') {
            const obj = this.eval(st.target.obj)
            const idx = this.eval(st.target.idx)
            this.quiet++
            const cur = this.getItem(st.target, obj, idx)
            this.quiet--
            const v = this.binop(st.op, cur, this.eval(st.x))
            if (obj.k === 'array' && v.k === 'array' && obj.a.dtype !== 'float64' && v.a.dtype === 'float64') {
              throw this.err('UFuncTypeError', `Cannot cast ufunc output from dtype('float64') to dtype('${obj.a.dtype}') with casting rule 'same_kind'`)
            }
            this.setItem(st.target, obj, idx, v, span)
          }
          break
        }
      }
    })
    return out
  }

  // ---------- 属性与方法 ----------

  kwInt(kw: Record<string, Value>, args: Value[], pos: number, name: string): number | null {
    const v = kw[name] ?? args[pos]
    if (!v || v.k === 'none') return null
    return this.toInt(v)
  }

  /** argsort 的 axis：缺省为 -1，显式 None 表示先拉平 */
  sortAxis(kw: Record<string, Value>, args: Value[]): number | null {
    return (kw.axis ?? args[0]) === undefined ? -1 : this.kwInt(kw, args, 0, 'axis')
  }

  dtypeOf(v: Value | undefined): DType | undefined {
    if (!v || v.k === 'none') return undefined
    if (v.k === 'dtype') return v.d
    if (v.k === 'type' && v.dtype) return v.dtype
    if (v.k === 'str') {
      const m: Record<string, DType> = { int: 'int64', int64: 'int64', int32: 'int64', float: 'float64', float64: 'float64', float32: 'float64', bool: 'bool' }
      if (m[v.v]) return m[v.v]
      throw this.err('TypeError', `data type '${v.v}' not understood`)
    }
    throw this.err('TypeError', `Cannot interpret '${repr(v)}' as a data type`)
  }

  reduceFn(a: NDArray, kind: ReduceKind, args: Value[], kw: Record<string, Value>): Value {
    const axis = this.kwInt(kw, args, 0, 'axis')
    const res = reduce(a, kind, axis)
    if (typeof res === 'number') {
      const d: DType = kind === 'mean' ? 'float64' : kind === 'any' || kind === 'all' ? 'bool' : kind.startsWith('arg') ? 'int64' : kind === 'sum' && a.dtype === 'bool' ? 'int64' : a.dtype
      return fromScalar(res, d)
    }
    return arr(res)
  }

  getAttr(obj: Value, name: string): Value {
    const fn = (call: Fn): Value => ({ k: 'fn', name, call })
    if (obj.k === 'module') {
      const v = obj.attrs[name]
      if (!v) throw this.err('AttributeError', `module 'numpy' has no attribute '${name}' (not available in this sandbox)`)
      return v
    }
    if (obj.k === 'array') {
      const a = obj.a
      switch (name) {
        case 'shape': return tuple(a.shape.map(int))
        case 'ndim': return int(a.ndim)
        case 'size': return int(a.size)
        case 'dtype': return { k: 'dtype', d: a.dtype }
        case 'T': return arr(transpose(a))
        case 'base': return a.base ? arr(a.base) : NONE
        case 'strides': return tuple(a.strides.map((s) => int(s * 8)))
        case 'reshape': return fn((args) => arr(reshape(a, this.toShape(args))))
        case 'transpose': return fn((args) => arr(transpose(a, args.length ? this.toShape(args) : undefined)))
        case 'copy': return fn(() => arr(a.copy()))
        case 'flatten': return fn(() => arr(NDArray.create(a.values(), [a.size], a.dtype)))
        case 'ravel': return fn(() => arr(reshape(a, [-1])))
        case 'astype': return fn((args, kw) => arr(NDArray.create(a.values(), a.shape, this.dtypeOf(kw.dtype ?? args[0]) ?? a.dtype)))
        case 'tolist': return fn(() => this.toList(a))
        case 'item': return fn(() => {
          if (a.size !== 1) throw this.err('ValueError', 'can only convert an array of size 1 to a Python scalar')
          return fromScalar(a.values()[0], a.dtype)
        })
        case 'nonzero': return fn(() => this.nonzero(a))
        case 'sum': case 'mean': case 'max': case 'min': case 'argmax': case 'argmin': case 'any': case 'all':
          return fn((args, kw) => this.reduceFn(a, name, args, kw))
        case 'dot': return fn((args) => this.binop('@', obj, args[0]))
        case 'argsort': return fn((args, kw) => arr(argsort(a, this.sortAxis(kw, args))))
      }
      throw this.err('AttributeError', `'numpy.ndarray' object has no attribute '${name}'`)
    }
    if (obj.k === 'list' && name === 'append') {
      return fn((args) => { obj.items.push(args[0]); return NONE })
    }
    if (obj.k === 'dtype' && name === 'name') return { k: 'str', v: obj.d }
    throw this.err('AttributeError', `'${typeName(obj)}' object has no attribute '${name}'`)
  }

  toList(a: NDArray): Value {
    const vals = a.values()
    if (a.ndim === 0) return fromScalar(vals[0], a.dtype)
    const st = cStrides(a.shape)
    const rec = (d: number, base: number): Value =>
      d === a.ndim - 1
        ? { k: 'list', items: Array.from({ length: a.shape[d] }, (_, i) => fromScalar(vals[base + i], a.dtype)) }
        : { k: 'list', items: Array.from({ length: a.shape[d] }, (_, i) => rec(d + 1, base + i * st[d])) }
    return rec(0, 0)
  }

  nonzero(a: NDArray): Value {
    const coords: number[][] = a.shape.map(() => [])
    const st = cStrides(a.shape)
    a.values().forEach((v, f) => {
      if (!v) return
      let r = f
      a.shape.forEach((_, k) => { coords[k].push(Math.floor(r / st[k])); r %= st[k] })
    })
    return tuple(coords.map((c) => arr(NDArray.create(c, [c.length], 'int64'))))
  }

  // ---------- numpy 命名空间与内置函数 ----------

  makeNumpy(): Value {
    const fn = (name: string, call: Fn): Value => ({ k: 'fn', name, call })
    const asArr = (v: Value | undefined, name: string): NDArray => {
      if (!v) throw this.err('TypeError', `${name}() missing required argument`)
      return toArray(v)
    }
    const filled = (name: string, fill: number) =>
      fn(name, (args, kw) => {
        const shapeV = kw.shape ?? args[0]
        if (!shapeV) throw this.err('TypeError', `${name}() missing required argument 'shape'`)
        const shape = this.toShape([shapeV])
        if (shape.some((d) => d < 0)) throw this.err('ValueError', 'negative dimensions are not allowed')
        checkSize(shape)
        const dtype = this.dtypeOf(kw.dtype ?? args[1]) ?? 'float64'
        return arr(NDArray.create(new Array(shape.reduce((p, x) => p * x, 1)).fill(fill), shape, dtype))
      })
    const reducer = (kind: ReduceKind) => fn(kind, (args, kw) => this.reduceFn(asArr(args[0], kind), kind, args.slice(1), kw))
    const elementwise = (name: string, f: (x: number) => number, keepInt = false) =>
      fn(name, (args) => {
        const a = asArr(args[0], name)
        const dtype: DType = keepInt && a.dtype !== 'float64' ? (a.dtype === 'bool' ? 'int64' : a.dtype) : 'float64'
        const res = NDArray.create(a.values().map(f), a.shape, dtype)
        return args[0].k === 'array' || args[0].k === 'list' ? arr(res) : fromScalar(res.values()[0], dtype)
      })
    const dt = (d: DType, name: string): Value => ({ k: 'type', name: `numpy.${name}`, dtype: d, call: (args) => fromScalar(castValue(numOf(args[0] ?? int(0)), d), d) })

    const attrs: Record<string, Value> = {
      array: fn('array', (args, kw) => {
        if (!args[0]) throw this.err('TypeError', "array() missing required argument 'object'")
        const src = toArray(args[0], this.dtypeOf(kw.dtype ?? args[1]))
        return arr(src === (args[0] as { a?: NDArray }).a ? src.copy() : src)
      }),
      asarray: fn('asarray', (args, kw) => arr(toArray(args[0], this.dtypeOf(kw.dtype)))),
      copy: fn('copy', (args) => arr(asArr(args[0], 'copy').copy())),
      arange: fn('arange', (args, kw) => {
        const nums = args.map((x) => { if (!isNum(x)) throw this.err('TypeError', 'arange() arguments must be numbers'); return x })
        if (!nums.length) throw this.err('TypeError', 'arange() requires stop to be specified.')
        const [start, stop, step] = nums.length === 1 ? [int(0), nums[0], int(1)] : [nums[0], nums[1], nums[2] ?? int(1)]
        const isFloat = [start, stop, step].some((x) => x.k === 'float')
        const s = numOf(step)
        if (s === 0) throw this.err('ZeroDivisionError', 'division by zero')
        const n = Math.max(0, Math.ceil((numOf(stop) - numOf(start)) / s))
        if (n > MAX_SIZE) throw this.err('MemoryError', `array of ${n} elements exceeds the sandbox limit of ${MAX_SIZE}`)
        const vals = Array.from({ length: n }, (_, i) => numOf(start) + i * s)
        return arr(NDArray.create(vals, [n], this.dtypeOf(kw.dtype) ?? (isFloat ? 'float64' : 'int64')))
      }),
      linspace: fn('linspace', (args, kw) => {
        const a = numOf(args[0])
        const b = numOf(args[1])
        const n = this.kwInt(kw, args, 2, 'num') ?? 50
        if (n > MAX_SIZE) throw this.err('MemoryError', `array of ${n} elements exceeds the sandbox limit of ${MAX_SIZE}`)
        return arr(NDArray.create(Array.from({ length: n }, (_, i) => (n === 1 ? a : a + ((b - a) * i) / (n - 1))), [n], 'float64'))
      }),
      zeros: filled('zeros', 0),
      ones: filled('ones', 1),
      full: fn('full', (args, kw) => {
        const shape = this.toShape([kw.shape ?? args[0]])
        checkSize(shape)
        const fill = kw.fill_value ?? args[1]
        const dtype = this.dtypeOf(kw.dtype ?? args[2]) ?? scalarDType(fill)
        return arr(NDArray.create(new Array(shape.reduce((p, x) => p * x, 1)).fill(numOf(fill)), shape, dtype))
      }),
      eye: fn('eye', (args, kw) => {
        const n = this.toInt(args[0])
        checkSize([n, n])
        return arr(NDArray.create(Array.from({ length: n * n }, (_, i) => (i % (n + 1) === 0 ? 1 : 0)), [n, n], this.dtypeOf(kw.dtype) ?? 'float64'))
      }),
      reshape: fn('reshape', (args, kw) => arr(reshape(asArr(args[0], 'reshape'), this.toShape([kw.shape ?? kw.newshape ?? args[1]])))),
      transpose: fn('transpose', (args) => arr(transpose(asArr(args[0], 'transpose')))),
      shares_memory: fn('shares_memory', (args) => bool(sharesMemory(asArr(args[0], 'shares_memory'), asArr(args[1], 'shares_memory')))),
      may_share_memory: fn('may_share_memory', (args) => bool(asArr(args[0], 'may_share_memory').data === asArr(args[1], 'may_share_memory').data)),
      where: fn('where', (args) => {
        const cond = asArr(args[0], 'where')
        if (args.length === 1) return this.nonzero(cond)
        const x = toArray(args[1])
        const y = toArray(args[2])
        const shape = broadcastShapes([cond.shape, x.shape, y.shape])
        const cv = broadcastTo(cond, shape).values()
        const xv = broadcastTo(x, shape).values()
        const yv = broadcastTo(y, shape).values()
        const dtype: DType = x.dtype === 'float64' || y.dtype === 'float64' ? 'float64' : x.dtype === 'bool' && y.dtype === 'bool' ? 'bool' : 'int64'
        return arr(NDArray.create(cv.map((v, i) => (v ? xv[i] : yv[i])), shape, dtype))
      }),
      nonzero: fn('nonzero', (args) => this.nonzero(asArr(args[0], 'nonzero'))),
      dot: fn('dot', (args) => this.binop('@', args[0], args[1])),
      matmul: fn('matmul', (args) => this.binop('@', args[0], args[1])),
      sum: reducer('sum'), mean: reducer('mean'), max: reducer('max'), min: reducer('min'),
      argmax: reducer('argmax'), argmin: reducer('argmin'), any: reducer('any'), all: reducer('all'),
      abs: elementwise('abs', Math.abs, true),
      sqrt: elementwise('sqrt', Math.sqrt),
      square: elementwise('square', (x) => x * x, true),
      expand_dims: fn('expand_dims', (args, kw) => {
        const axis = this.kwInt(kw, args.slice(1), 0, 'axis')
        if (axis === null) throw this.err('TypeError', "expand_dims() missing required argument 'axis'")
        return arr(expandDims(asArr(args[0], 'expand_dims'), axis))
      }),
      argsort: fn('argsort', (args, kw) => {
        const a = asArr(args[0], 'argsort')
        return arr(argsort(a, this.sortAxis(kw, args.slice(1))))
      }),
      exp: elementwise('exp', Math.exp),
      log: elementwise('log', Math.log),
      newaxis: NONE,
      nan: float(NaN),
      inf: float(Infinity),
      pi: float(Math.PI),
      int64: dt('int64', 'int64'),
      int32: dt('int64', 'int32'),
      float64: dt('float64', 'float64'),
      float32: dt('float64', 'float32'),
      bool_: dt('bool', 'bool'),
      ndarray: { k: 'type', name: 'numpy.ndarray', call: () => { throw this.err('TypeError', 'use np.array / np.zeros to create arrays') } },
    }
    return { k: 'module', name: 'numpy', attrs }
  }

  makeBuiltins(): Record<string, Value> {
    const fn = (name: string, call: Fn): Value => ({ k: 'fn', name, call })
    const conv = (name: string, d: DType): Value => ({
      k: 'type', name, dtype: d,
      call: (args) => {
        const x = args[0] ?? int(0)
        if (x.k === 'str') {
          const n = Number(x.v)
          if (x.v.trim() === '' || Number.isNaN(n) || (d === 'int64' && !/^\s*[-+]?\d+\s*$/.test(x.v))) {
            throw this.err('ValueError', `invalid literal for ${name}(): ${reprStr(x.v)}`)
          }
          return fromScalar(n, d)
        }
        if (d === 'bool') return bool(this.truthy(x))
        if (x.k === 'array' && x.a.size !== 1) throw this.err('TypeError', 'only length-1 arrays can be converted to Python scalars')
        const v = x.k === 'array' ? x.a.values()[0] : numOf(x)
        if (Number.isNaN(v) && !isNum(x) && x.k !== 'array') throw this.err('TypeError', `${name}() argument must be a string or a real number, not '${typeName(x)}'`)
        return fromScalar(castValue(v, d), d)
      },
    })
    return {
      print: fn('print', (args, kw) => {
        const sep = kw.sep && kw.sep.k === 'str' ? kw.sep.v : ' '
        const end = kw.end && kw.end.k === 'str' ? kw.end.v : '\n'
        this.stdout.push(args.map(str).join(sep) + end)
        return NONE
      }),
      len: fn('len', (args) => {
        const v = args[0]
        if (v.k === 'array') {
          if (v.a.ndim === 0) throw this.err('TypeError', 'len() of unsized object')
          return int(v.a.shape[0])
        }
        if (v.k === 'list' || v.k === 'tuple') return int(v.items.length)
        if (v.k === 'str') return int(v.v.length)
        throw this.err('TypeError', `object of type '${typeName(v)}' has no len()`)
      }),
      range: fn('range', (args) => {
        const n = args.map((x) => this.toInt(x))
        const [a, b, s] = n.length === 1 ? [0, n[0], 1] : [n[0], n[1], n[2] ?? 1]
        if (s === 0) throw this.err('ValueError', 'range() arg 3 must not be zero')
        const len = Math.max(0, Math.ceil((b - a) / s))
        if (len > MAX_SIZE) throw this.err('MemoryError', 'range too large for this sandbox')
        return { k: 'list', items: Array.from({ length: len }, (_, i) => int(a + i * s)) }
      }),
      list: fn('list', (args) => ({ k: 'list', items: args[0] ? [...this.iterate(args[0])] : [] })),
      tuple: fn('tuple', (args) => tuple(args[0] ? [...this.iterate(args[0])] : [])),
      abs: fn('abs', (args) => {
        const x = args[0]
        if (x.k === 'array') return arr(NDArray.create(x.a.values().map(Math.abs), x.a.shape, x.a.dtype === 'bool' ? 'int64' : x.a.dtype))
        return x.k === 'float' ? float(Math.abs(x.v)) : int(Math.abs(numOf(x)))
      }),
      type: fn('type', (args) => ({ k: 'type', name: typeName(args[0]), call: () => NONE })),
      slice: fn('slice', (args) => {
        const [a, b, c] = args.length === 1 ? [NONE, args[0], NONE] : [args[0], args[1], args[2] ?? NONE]
        return { k: 'slice', start: a, stop: b, step: c }
      }),
      int: conv('int', 'int64'),
      float: conv('float', 'float64'),
      bool: conv('bool', 'bool'),
      Ellipsis: { k: 'ellipsis' },
    }
  }

  vars(): VarInfo[] {
    const out: VarInfo[] = []
    for (const [name, v] of this.env) {
      if (v.k === 'module') continue
      if (v.k === 'array') {
        const sharesWith = [...this.env].filter(([n, o]) => n !== name && o.k === 'array' && sharesMemory(o.a, v.a)).map(([n]) => n)
        out.push({ name, kind: 'ndarray', shape: v.a.shape, dtype: v.a.dtype, sharesWith, isView: v.a.base !== null, preview: '' })
      } else {
        const p = repr(v)
        out.push({ name, kind: typeName(v), sharesWith: [], isView: false, preview: p.length > 40 ? p.slice(0, 39) + '…' : p })
      }
    }
    return out
  }
}

/** 运行一段代码：返回输出、错误、所有下标读写轨迹与变量表 */
export const runPython = (src: string): RunResult => {
  const it = new Interp(src)
  let out: string | null = null
  let error: RunResult['error'] = null
  try {
    out = it.exec(parse(src))
  } catch (e) {
    if (e instanceof PyError) error = { type: e.pyType, message: e.message, line: e.line ?? it.line }
    else if (e instanceof RangeError) error = { type: 'RecursionError', message: 'expression too deeply nested', line: it.line }
    else error = { type: 'InternalError', message: e instanceof Error ? e.message : String(e), line: it.line }
  }
  return { stdout: it.stdout.join(''), out, error, traces: it.traces, vars: it.vars() }
}
