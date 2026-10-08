// --- 迷你 Python 解释器：词法 → 语法树 → 求值；所有 ndarray 下标读写都会记录一条 IndexTrace 供可视化 ---
// 支持子集：import（numpy / math / random 与 RunOptions.libs 提供的库）、赋值 / 增量赋值 / 解包、表达式、f-string、
// 推导式、lambda，以及 if / for / while / def / class / with 语句块。不支持 try / raise / yield 与 *args。

import {
  NDArray, PyError, MAX_SIZE, arrayRepr, arrayStr, binaryOp, castValue, dot, formatScalar, getIndex, reduce,
  reshape, setIndex, sharesMemory, shapeStr, transpose, unaryOp, checkSize, applyOp, cStrides, broadcastShapes, broadcastTo,
  reduceDType, setSizeLimit,
} from './ndarray'
import type { AxisNote, BinOp, DType, IndexItem, IndexPlan, ReduceKind } from './ndarray'
import type { Display, PyEvent, PyEventInput } from './pyEvents'
import { DEFAULT_IV, complexRepr, convert, deriv, integ, lstsq, mapParams, polyRepr, polyStr, polyadd, polymul, polypow, polysub, polyval, roots, trim } from './poly'
import type { Interval } from './poly'
import {
  SandboxRandom, concatenate, cumsum, det, expandDims, flip, hstack, inv, norm, outer, repeat, roundHalfEven, sortAlong,
  squeeze, stack, swapaxes, tile, trace, unique, vstack, commonDType,
} from './ndops'

// ================= 词法 =================

type TokType = 'name' | 'num' | 'str' | 'fstr' | 'op' | 'nl' | 'indent' | 'dedent' | 'eof'
interface Tok { t: TokType; v: string; pos: number; end: number; line: number; body?: number }

const OPS = [
  '**=', '//=', '...', '->', '**', '//', '==', '!=', '<=', '>=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '@=',
  '+', '-', '*', '/', '%', '@', '&', '|', '^', '~', '<', '>', '=', '(', ')', '[', ']', '{', '}', ',', ':', '.', ';',
]

const syntaxError = (msg: string, line: number) => new PyError('SyntaxError', msg, line)

const ESCAPES: Record<string, string> = { n: '\n', t: '\t', r: '\r', '0': '\0', '\\': '\\', "'": "'", '"': '"' }

/** offset / firstLine: an f-string's {expression} is tokenized in place, so its spans still point into the program */
export const tokenize = (src: string, offset = 0, firstLine = 1): Tok[] => {
  const toks: Tok[] = []
  let i = 0
  let line = firstLine
  let depth = 0
  let lineStart = offset === 0
  const indents = [0]
  const push = (t: TokType, v: string, pos: number, end = pos + v.length) => toks.push({ t, v, pos: pos + offset, end: end + offset, line })
  while (i < src.length) {
    const ch = src[i]
    if (lineStart && depth === 0) {
      // 行首缩进：比上一层深 → indent；回到外层 → 一个或多个 dedent。空行与注释行不算
      let j = i
      let width = 0
      while (src[j] === ' ' || src[j] === '\t') width = src[j++] === '\t' ? width + 8 - (width % 8) : width + 1
      const c = src[j]
      lineStart = false
      i = j
      if (c === undefined || c === '\n' || c === '#' || c === '\r') continue
      if (width > indents[indents.length - 1]) {
        indents.push(width)
        push('indent', '', i, i)
      } else {
        while (width < indents[indents.length - 1]) {
          indents.pop()
          push('dedent', '', i, i)
        }
        if (width !== indents[indents.length - 1]) throw new PyError('IndentationError', 'unindent does not match any outer indentation level', line)
      }
      continue
    }
    if (ch === '\n') {
      if (depth === 0 && toks.length && toks[toks.length - 1].t !== 'nl' && toks[toks.length - 1].t !== 'dedent' && toks[toks.length - 1].t !== 'indent') push('nl', '\n', i)
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
    // string prefixes: r'…' b'…' f'…' rf'…'
    const pre = /^([rRbBfFuU]{1,2})(?=['"])/.exec(src.slice(i, i + 3))
    if (/[A-Za-z_]/.test(ch) && !pre) {
      const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i))!
      push('name', m[0], i)
      i += m[0].length
      continue
    }
    if (ch === '"' || ch === "'" || pre) {
      const prefix = pre ? pre[1].toLowerCase() : ''
      const start = i
      let j = i + prefix.length
      const q = src[j]
      const triple = src.startsWith(q.repeat(3), j)
      const close = triple ? q.repeat(3) : q
      j += close.length
      const startLine = line
      let s = ''
      for (;;) {
        if (j >= src.length) throw syntaxError(triple ? 'unterminated triple-quoted string literal' : 'unterminated string literal', startLine)
        if (src.startsWith(close, j)) break
        const c = src[j]
        if (c === '\n') {
          if (!triple) throw syntaxError('unterminated string literal', line)
          line++
        }
        if (c === '\\' && j + 1 < src.length && !prefix.includes('r')) {
          const e = src[j + 1]
          if (e === '\n') line++
          else s += ESCAPES[e] ?? '\\' + e
          j += 2
        } else {
          s += c
          j++
        }
      }
      j += close.length
      const t: TokType = prefix.includes('f') ? 'fstr' : 'str'
      // an f-string keeps its raw source (the {expressions} are parsed later), so record where the body starts
      const bodyAt = start + prefix.length + close.length
      const tok: Tok = { t, v: t === 'fstr' ? src.slice(bodyAt, j - close.length) : s, pos: start + offset, end: j + offset, line: startLine, body: bodyAt + offset }
      toks.push(tok)
      i = j
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
  if (toks.length && toks[toks.length - 1].t !== 'nl' && toks[toks.length - 1].t !== 'dedent') push('nl', '\n', src.length)
  while (indents.length > 1) {
    indents.pop()
    push('dedent', '', src.length, src.length)
  }
  push('eof', '', src.length)
  return toks
}

// ================= 语法树 =================

interface Span { s: number; e: number; line: number }

export interface Param { name: string; def: Node | null }
export interface CompFor { target: Node; iter: Node; conds: Node[] }
export type FPart = string | { x: Node; conv: 'r' | 's' | null; spec: string }

export type Node = Span & (
  | { k: 'num'; v: number; isFloat: boolean }
  | { k: 'str'; v: string }
  | { k: 'fstr'; parts: FPart[] }
  | { k: 'const'; v: 'True' | 'False' | 'None' }
  | { k: 'name'; id: string }
  | { k: 'list'; items: Node[] }
  | { k: 'tuple'; items: Node[] }
  | { k: 'dict'; keys: Node[]; vals: Node[] }
  | { k: 'comp'; kind: 'list' | 'gen' | 'dict'; elt: Node; val: Node | null; gens: CompFor[] }
  | { k: 'lambda'; params: Param[]; body: Node }
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
  | { k: 'import'; mod: string; alias: string | null; line: number }
  | { k: 'from'; mod: string; names: { name: string; alias: string }[]; line: number }
  | { k: 'pass'; line: number }
  | { k: 'if'; branches: { cond: Node | null; body: Stmt[] }[]; line: number }
  | { k: 'for'; target: Node; iter: Node; body: Stmt[]; line: number }
  | { k: 'while'; cond: Node; body: Stmt[]; line: number }
  | { k: 'def'; name: string; params: Param[]; body: Stmt[]; line: number }
  | { k: 'return'; x: Node | null; line: number }
  | { k: 'class'; name: string; bases: Node[]; body: Stmt[]; line: number }
  | { k: 'with'; items: { x: Node; as: Node | null }[]; body: Stmt[]; line: number }
  | { k: 'break' | 'continue'; line: number }
  | { k: 'del'; targets: Node[]; line: number }
  | { k: 'assert'; x: Node; msg: Node | null; line: number }

const UNSUPPORTED = new Set(['try', 'except', 'finally', 'raise', 'global', 'nonlocal', 'yield', 'async', 'await'])
/** keywords that can never start an expression */
const KEYWORDS = new Set([...UNSUPPORTED, 'for', 'while', 'if', 'elif', 'else', 'def', 'class', 'with', 'return', 'del', 'pass', 'break', 'continue', 'import', 'from', 'as', 'assert', 'in', 'is', 'and', 'or'])
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
  expectName(): string {
    if (this.cur.t !== 'name' || KEYWORDS.has(this.cur.v)) throw this.unexpected()
    return this.next().v
  }
  unexpected() {
    const tk = this.cur
    if (tk.t === 'indent') return new PyError('IndentationError', 'unexpected indent', tk.line)
    if (tk.t === 'eof' || tk.t === 'nl' || tk.t === 'dedent') return syntaxError('unexpected end of line', tk.line)
    return syntaxError(`invalid syntax near '${tk.v}'`, tk.line)
  }
  endOf(): number { return this.toks[this.i - 1].end }
  mk<T extends Omit<Node, keyof Span>>(start: Tok, n: T): Node {
    return { ...n, s: start.pos, e: this.endOf(), line: start.line } as unknown as Node
  }

  program(): Stmt[] {
    const out: Stmt[] = []
    while (this.cur.t !== 'eof') {
      if (this.cur.t === 'nl') { this.next(); continue }
      out.push(...this.statement())
    }
    return out
  }

  /** after a header's ':' — an indented block, or simple statements on the same line */
  block(): Stmt[] {
    this.expectOp(':')
    if (this.cur.t !== 'nl') return this.simpleLine()
    this.next()
    if (this.toks[this.i].t !== 'indent') throw new PyError('IndentationError', 'expected an indented block', this.cur.line)
    this.next()
    const out: Stmt[] = []
    const kind = () => this.toks[this.i].t
    while (kind() !== 'dedent' && kind() !== 'eof') {
      if (kind() === 'nl') { this.next(); continue }
      out.push(...this.statement())
    }
    if (kind() === 'dedent') this.next()
    return out
  }

  /** one or more ';'-separated simple statements, up to the end of the line */
  simpleLine(): Stmt[] {
    const out = [this.simple()]
    while (this.isOp(';')) {
      this.next()
      if (this.atLineEnd()) break
      out.push(this.simple())
    }
    if (!this.atLineEnd()) throw this.unexpected()
    if (this.cur.t === 'nl') this.next()
    return out
  }

  statement(): Stmt[] {
    const tk = this.cur
    if (tk.t === 'indent') throw this.unexpected()
    if (tk.t !== 'name') return this.simpleLine()
    switch (tk.v) {
      case 'if': {
        const branches: { cond: Node | null; body: Stmt[] }[] = []
        this.next()
        branches.push({ cond: this.namedTest(), body: this.block() })
        while (this.isName('elif')) {
          this.next()
          branches.push({ cond: this.namedTest(), body: this.block() })
        }
        if (this.isName('else')) {
          this.next()
          branches.push({ cond: null, body: this.block() })
        }
        return [{ k: 'if', branches, line: tk.line }]
      }
      case 'for': {
        this.next()
        const target = this.targetList()
        if (!this.isName('in')) throw this.unexpected()
        this.next()
        const iter = this.testList()
        const body = this.block()
        if (this.isName('else')) throw new PyError('NotImplementedError', "'for … else' is not supported in this sandbox", this.cur.line)
        return [{ k: 'for', target, iter, body, line: tk.line }]
      }
      case 'while': {
        this.next()
        const cond = this.namedTest()
        return [{ k: 'while', cond, body: this.block(), line: tk.line }]
      }
      case 'def': {
        this.next()
        const name = this.expectName()
        this.expectOp('(')
        const params = this.params(')')
        this.expectOp(')')
        if (this.isOp('->')) { this.next(); this.test() }
        return [{ k: 'def', name, params, body: this.block(), line: tk.line }]
      }
      case 'class': {
        this.next()
        const name = this.expectName()
        const bases: Node[] = []
        if (this.isOp('(')) {
          this.next()
          while (!this.isOp(')')) {
            bases.push(this.test())
            if (!this.isOp(')')) this.expectOp(',')
          }
          this.next()
        }
        return [{ k: 'class', name, bases, body: this.block(), line: tk.line }]
      }
      case 'with': {
        this.next()
        const items: { x: Node; as: Node | null }[] = []
        do {
          if (this.isOp(',')) this.next()
          const x = this.test()
          let as: Node | null = null
          if (this.isName('as')) {
            this.next()
            as = this.postfix()
            this.checkTarget(as)
          }
          items.push({ x, as })
        } while (this.isOp(','))
        return [{ k: 'with', items, body: this.block(), line: tk.line }]
      }
    }
    if (tk.v === 'elif' || tk.v === 'else') throw syntaxError('invalid syntax', tk.line)
    return this.simpleLine()
  }

  params(close: string): Param[] {
    const params: Param[] = []
    while (!this.isOp(close)) {
      if (this.isOp('*') || this.isOp('**')) throw new PyError('NotImplementedError', '*args / **kwargs are not supported in this sandbox', this.cur.line)
      const name = this.expectName()
      if (this.isOp(':') && close === ')') { this.next(); this.test() }
      let def: Node | null = null
      if (this.isOp('=')) { this.next(); def = this.test() }
      else if (params.some((p) => p.def)) throw syntaxError('non-default argument follows default argument', this.cur.line)
      params.push({ name, def })
      if (!this.isOp(close)) this.expectOp(',')
    }
    return params
  }

  /** a condition (no assignment expressions in this sandbox) */
  namedTest(): Node {
    return this.test()
  }

  simple(): Stmt {
    const tk = this.cur
    if (tk.t === 'name' && UNSUPPORTED.has(tk.v)) {
      throw new PyError('NotImplementedError', `'${tk.v}' is not supported in this sandbox`, tk.line)
    }
    if (this.isName('pass')) { this.next(); return { k: 'pass', line: tk.line } }
    if (this.isName('break') || this.isName('continue')) return { k: this.next().v as 'break' | 'continue', line: tk.line }
    if (this.isName('return')) {
      this.next()
      return { k: 'return', x: this.atLineEnd() || this.isOp(';') ? null : this.testList(), line: tk.line }
    }
    if (this.isName('del')) {
      this.next()
      const t = this.testList()
      const targets = t.k === 'tuple' ? t.items : [t]
      targets.forEach((x) => this.checkTarget(x))
      return { k: 'del', targets, line: tk.line }
    }
    if (this.isName('assert')) {
      this.next()
      const x = this.test()
      let msg: Node | null = null
      if (this.isOp(',')) { this.next(); msg = this.test() }
      return { k: 'assert', x, msg, line: tk.line }
    }
    if (this.isName('import')) {
      this.next()
      const mod = this.dotted()
      let alias: string | null = null
      if (this.isName('as')) { this.next(); alias = this.expectName() }
      if (this.isOp(',')) throw new PyError('NotImplementedError', 'write one import per line in this sandbox', tk.line)
      return { k: 'import', mod, alias, line: tk.line }
    }
    if (this.isName('from')) {
      this.next()
      const mod = this.dotted()
      if (!this.isName('import')) throw this.unexpected()
      this.next()
      const paren = this.isOp('(')
      if (paren) this.next()
      const names: { name: string; alias: string }[] = []
      do {
        if (this.isOp(',')) this.next()
        if (paren && this.isOp(')')) break
        if (this.isOp('*')) throw new PyError('NotImplementedError', "'from … import *' is not supported in this sandbox", tk.line)
        const name = this.expectName()
        let alias = name
        if (this.isName('as')) { this.next(); alias = this.expectName() }
        names.push({ name, alias })
      } while (this.isOp(','))
      if (paren) this.expectOp(')')
      return { k: 'from', mod, names, line: tk.line }
    }
    const first = this.testList()
    if (this.cur.t === 'op' && /^(\*\*|\/\/|[-+*/%&|^@])=$/.test(this.cur.v)) {
      const op = this.next().v.slice(0, -1)
      if (first.k !== 'name' && first.k !== 'sub' && first.k !== 'attr') throw syntaxError("'illegal expression for augmented assignment'", tk.line)
      const x = this.testList()
      return { k: 'aug', op, target: first, x, line: tk.line, s: tk.pos, e: this.endOf() }
    }
    if (this.isOp(':') && (first.k === 'name' || first.k === 'attr')) {
      // annotated assignment: x: int = 3
      this.next()
      this.test()
      if (!this.isOp('=')) return { k: 'pass', line: tk.line }
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

  dotted(): string {
    let mod = this.expectName()
    while (this.isOp('.')) { this.next(); mod += '.' + this.expectName() }
    return mod
  }

  checkTarget(t: Node) {
    if (t.k === 'name' || t.k === 'sub' || t.k === 'attr') return
    if ((t.k === 'tuple' || t.k === 'list') && t.items.length) {
      t.items.forEach((x) => this.checkTarget(x))
      return
    }
    throw syntaxError('cannot assign to expression', t.line)
  }

  /** for-loop / comprehension target: a, (b, c) — stops before 'in' */
  targetList(): Node {
    const start = this.cur
    const first = this.bitOr()
    if (!this.isOp(',')) {
      this.checkTarget(first)
      return first
    }
    const items = [first]
    while (this.isOp(',')) {
      this.next()
      if (this.isName('in')) break
      items.push(this.bitOr())
    }
    const t = this.mk(start, { k: 'tuple', items })
    this.checkTarget(t)
    return t
  }

  testList(): Node {
    const start = this.cur
    const first = this.test()
    if (!this.isOp(',')) return first
    const items = [first]
    while (this.isOp(',')) {
      this.next()
      if (this.atLineEnd() || this.isOp('=') || this.isOp(';') || this.isOp(':') || this.isOp(')')) break
      items.push(this.test())
    }
    return this.mk(start, { k: 'tuple', items })
  }

  test(): Node {
    const start = this.cur
    if (this.isName('lambda')) {
      this.next()
      const params = this.params(':')
      this.expectOp(':')
      return this.mk(start, { k: 'lambda', params, body: this.test() })
    }
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

  /** the 'for … in … if …' clauses of a comprehension */
  compFors(): CompFor[] {
    const gens: CompFor[] = []
    while (this.isName('for')) {
      this.next()
      const target = this.targetList()
      if (!this.isName('in')) throw this.unexpected()
      this.next()
      const iter = this.orTest()
      const conds: Node[] = []
      while (this.isName('if')) {
        this.next()
        conds.push(this.orTest())
      }
      gens.push({ target, iter, conds })
    }
    return gens
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
          if (this.isOp('*') || this.isOp('**')) throw new PyError('NotImplementedError', '*args / **kwargs are not supported in this sandbox', this.cur.line)
          if (this.cur.t === 'name' && this.peek().t === 'op' && this.peek().v === '=') {
            const name = this.next().v
            this.next()
            kw.push({ name, v: this.test() })
          } else {
            if (kw.length) throw syntaxError('positional argument follows keyword argument', this.cur.line)
            const argStart = this.cur
            const a = this.test()
            // sum(x * x for x in xs)
            args.push(this.isName('for') ? this.mk(argStart, { k: 'comp', kind: 'gen', elt: a, val: null, gens: this.compFors() }) : a)
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

  /** f'…{expr!r:spec}…' → literal text and parsed expressions */
  fstring(tk: Tok): Node {
    const raw = tk.v
    const bodyStart = tk.body ?? tk.pos
    const parts: FPart[] = []
    let lit = ''
    let i = 0
    const flush = () => {
      if (lit) parts.push(lit.replace(/\\(.)/g, (_, e: string) => ESCAPES[e] ?? '\\' + e))
      lit = ''
    }
    while (i < raw.length) {
      const c = raw[i]
      if (c === '{' && raw[i + 1] === '{') { lit += '{'; i += 2; continue }
      if (c === '}' && raw[i + 1] === '}') { lit += '}'; i += 2; continue }
      if (c === '}') throw syntaxError("f-string: single '}' is not allowed", tk.line)
      if (c !== '{') { lit += c; i++; continue }
      flush()
      // find the matching close brace, skipping nested brackets and strings
      let j = i + 1
      let depth = 0
      let exprEnd = -1
      let q: string | null = null
      for (; j < raw.length; j++) {
        const d = raw[j]
        if (q) { if (d === q) q = null; continue }
        if (d === '"' || d === "'") q = d
        else if ('([{'.includes(d)) depth++
        else if (')]'.includes(d) || (d === '}' && depth > 0)) depth--
        else if (depth === 0 && (d === '}' || d === ':' || (d === '!' && raw[j + 1] !== '='))) {
          if (exprEnd < 0) exprEnd = j
          if (d === '}') break
          if (d === '!' || d === ':') {
            // conversion and format spec run to the closing brace
            const close = raw.indexOf('}', j)
            if (close < 0) break
            j = close
            break
          }
        }
      }
      if (j >= raw.length) throw syntaxError("f-string: expecting '}'", tk.line)
      const exprSrc = raw.slice(i + 1, exprEnd)
      if (!exprSrc.trim()) throw syntaxError('f-string: empty expression not allowed', tk.line)
      const tail = raw.slice(exprEnd, j)
      const m = /^(?:!([rs]))?(?::(.*))?$/s.exec(tail)
      if (!m) throw syntaxError('f-string: invalid conversion character', tk.line)
      const sub = new Parser(tokenize(exprSrc, bodyStart + i + 1, tk.line))
      const x = sub.testList()
      if (sub.cur.t !== 'nl' && sub.cur.t !== 'eof') throw syntaxError('f-string: invalid syntax', tk.line)
      parts.push({ x, conv: (m[1] as 'r' | 's' | undefined) ?? null, spec: m[2] ?? '' })
      i = j + 1
    }
    flush()
    return this.mk(tk, { k: 'fstr', parts })
  }

  atom(): Node {
    const tk = this.cur
    if (tk.t === 'num') {
      this.next()
      const isFloat = /[.eE]/.test(tk.v)
      return this.mk(tk, { k: 'num', v: Number(tk.v), isFloat })
    }
    if (tk.t === 'str' || tk.t === 'fstr') {
      // adjacent literals concatenate; any f-string makes the whole run an f-string
      const pieces: Node[] = []
      while (this.cur.t === 'str' || this.cur.t === 'fstr') {
        const p = this.next()
        pieces.push(p.t === 'fstr' ? this.fstring(p) : this.mk(p, { k: 'str', v: p.v }))
      }
      if (pieces.every((p) => p.k === 'str')) return this.mk(tk, { k: 'str', v: pieces.map((p) => (p as { v: string }).v).join('') })
      return this.mk(tk, { k: 'fstr', parts: pieces.flatMap((p) => (p.k === 'str' ? [p.v] : p.k === 'fstr' ? p.parts : [])) })
    }
    if (tk.t === 'name') {
      if (UNSUPPORTED.has(tk.v)) throw new PyError('NotImplementedError', `'${tk.v}' is not supported in this sandbox`, tk.line)
      if (KEYWORDS.has(tk.v)) throw this.unexpected()
      this.next()
      if (tk.v === 'True' || tk.v === 'False' || tk.v === 'None') return this.mk(tk, { k: 'const', v: tk.v })
      return this.mk(tk, { k: 'name', id: tk.v })
    }
    if (this.isOp('...')) { this.next(); return this.mk(tk, { k: 'ellipsis' }) }
    if (this.isOp('(')) {
      this.next()
      if (this.isOp(')')) { this.next(); return this.mk(tk, { k: 'tuple', items: [] }) }
      const first = this.test()
      if (this.isName('for')) {
        const gens = this.compFors()
        this.expectOp(')')
        return this.mk(tk, { k: 'comp', kind: 'gen', elt: first, val: null, gens })
      }
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
        if (items.length === 1 && this.isName('for')) {
          const gens = this.compFors()
          this.expectOp(']')
          return this.mk(tk, { k: 'comp', kind: 'list', elt: items[0], val: null, gens })
        }
        if (!this.isOp(']')) this.expectOp(',')
      }
      this.next()
      return this.mk(tk, { k: 'list', items })
    }
    if (this.isOp('{')) {
      this.next()
      const keys: Node[] = []
      const vals: Node[] = []
      while (!this.isOp('}')) {
        const key = this.test()
        if (!this.isOp(':')) throw new PyError('NotImplementedError', 'sets are not supported in this sandbox — use a list', tk.line)
        this.next()
        const val = this.test()
        if (keys.length === 0 && this.isName('for')) {
          const gens = this.compFors()
          this.expectOp('}')
          return this.mk(tk, { k: 'comp', kind: 'dict', elt: key, val, gens })
        }
        keys.push(key)
        vals.push(val)
        if (!this.isOp('}')) this.expectOp(',')
      }
      this.next()
      return this.mk(tk, { k: 'dict', keys, vals })
    }
    throw this.unexpected()
  }
}

export const parse = (src: string): Stmt[] => new Parser(tokenize(src)).program()

// ================= 运行时值 =================

type Fn = (args: Value[], kw: Record<string, Value>) => Value

/** 可视化用的 API 分类：决定「输出元素来自哪些输入元素」的计算方式 */
export type ApiKind = 'create' | 'move' | 'elementwise' | 'reduce' | 'scan' | 'matmul' | 'sort' | 'unique' | 'linalg' | 'random' | 'info' | 'poly'

interface ApiMeta {
  /** 'np.sum' / 'ndarray.sum' / 'np.linalg.inv' … */
  name: string
  kind: ApiKind
  /** axis 参数在位置参数中的下标（不含 self） */
  axisPos?: number
  /** 未给 axis 时的默认值（None 记为 null） */
  defaultAxis?: number | null
}

export type Kw = Record<string, Value>

/** A user-defined function or lambda; closure = the enclosing function's frame (null at module level) */
export interface FuncValue {
  k: 'func'
  name: string
  params: Param[]
  defaults: (Value | null)[]
  body: Stmt[] | null
  expr: Node | null
  closure: Frame | null
  /** the class whose body defined it (for zero-argument super()) */
  owner: ClassValue | null
}

/**
 * A library base class that user classes can extend (nn.Module, keras.Model, …).
 * init runs for super().__init__(); the other hooks run when an attribute is not found on the instance / class.
 */
export interface HostClass {
  init(inst: InstValue, args: Value[], kw: Kw, h: Host): void
  getAttr?(inst: InstValue, name: string, h: Host): Value | undefined
  setAttr?(inst: InstValue, name: string, v: Value, h: Host): void
  call?(inst: InstValue, args: Value[], kw: Kw, h: Host): Value
  repr?(inst: InstValue): string | undefined
}

export interface ClassValue { k: 'class'; name: string; bases: Value[]; ns: Map<string, Value>; host: HostClass | null }
export interface InstValue { k: 'inst'; cls: ClassValue; attrs: Map<string, Value>; state: Record<string, unknown> }

export type Value =
  | { k: 'int'; v: number }
  | { k: 'float'; v: number }
  | { k: 'bool'; v: boolean }
  | { k: 'none' }
  | { k: 'str'; v: string }
  | { k: 'list'; items: Value[] }
  | { k: 'tuple'; items: Value[] }
  | { k: 'dict'; d: PyDict }
  | { k: 'range'; start: number; stop: number; step: number }
  | { k: 'slice'; start: Value; stop: Value; step: Value }
  | { k: 'ellipsis' }
  | { k: 'array'; a: NDArray }
  | { k: 'fn'; name: string; call: Fn; api?: ApiMeta; self?: NDArray; rebind?: (self: NDArray) => Fn }
  | { k: 'type'; name: string; call: Fn; dtype?: DType; api?: ApiMeta; host?: HostClass }
  | { k: 'poly'; coef: number[]; domain: Interval; window: Interval }
  | { k: 'carray'; re: number[]; im: number[] }
  | { k: 'module'; name: string; attrs: Record<string, Value> }
  | { k: 'dtype'; d: DType }
  | FuncValue
  | { k: 'bound'; self: Value; f: FuncValue }
  | ClassValue
  | InstValue
  | { k: 'super'; self: InstValue; after: ClassValue }
  | { k: 'obj'; o: PyObj }

export interface Frame { vars: Map<string, Value>; parent: Frame | null; isClass?: boolean; owner?: ClassValue | null; self?: Value }

/** Python dict: insertion-ordered, keyed by a hash of the key's value (1, 1.0 and True collide, as in Python) */
export class PyDict {
  m = new Map<string, [Value, Value]>()
  static hash(v: Value): string {
    switch (v.k) {
      case 'int': case 'float': return `n${v.v}`
      case 'bool': return `n${v.v ? 1 : 0}`
      case 'str': return `s${v.v}`
      case 'none': return 'None'
      case 'tuple': return `t(${v.items.map(PyDict.hash).join(',')})`
      case 'list': case 'dict': case 'array':
        throw new PyError('TypeError', `unhashable type: '${v.k === 'array' ? 'numpy.ndarray' : v.k}'`)
      case 'obj': return `o${objId(v.o)}`
      default: return `${v.k}:${repr(v)}`
    }
  }
  get(k: Value): Value | undefined { return this.m.get(PyDict.hash(k))?.[1] }
  set(k: Value, v: Value) {
    const h = PyDict.hash(k)
    const old = this.m.get(h)
    this.m.set(h, [old ? old[0] : k, v])
  }
  has(k: Value) { return this.m.has(PyDict.hash(k)) }
  delete(k: Value) { return this.m.delete(PyDict.hash(k)) }
  keys(): Value[] { return [...this.m.values()].map((e) => e[0]) }
  values(): Value[] { return [...this.m.values()].map((e) => e[1]) }
  items(): [Value, Value][] { return [...this.m.values()] }
  get size() { return this.m.size }
  static from(entries: [Value, Value][]): PyDict {
    const d = new PyDict()
    for (const [k, v] of entries) d.set(k, v)
    return d
  }
}

const OBJ_IDS = new WeakMap<object, number>()
let nextObjId = 1
export const objId = (o: object): number => {
  let id = OBJ_IDS.get(o)
  if (id === undefined) OBJ_IDS.set(o, (id = nextObjId++))
  return id
}

/**
 * A Python object implemented by a sandbox library (DataFrame, Tensor, Figure, …).
 * Every hook is optional: a missing one gives Python's usual TypeError / AttributeError.
 */
export abstract class PyObj {
  /** type(x).__name__, used in error messages */
  abstract readonly cls: string
  abstract repr(): string
  str?(): string
  /** format(x, spec) — f'{x:.3f}' */
  format?(spec: string): string | undefined
  getAttr?(name: string, h: Host): Value | undefined
  setAttr?(name: string, v: Value, h: Host): boolean
  getItem?(idx: Value, h: Host): Value
  setItem?(idx: Value, v: Value, h: Host): void
  /** undefined = NotImplemented (the other operand gets a turn) */
  binop?(op: string, other: Value, reflected: boolean, h: Host): Value | undefined
  /** x += y in place; false = fall back to x = x + y */
  inplace?(op: string, rhs: Value, h: Host): boolean
  unary?(op: '-' | '+' | '~', h: Host): Value
  call?(args: Value[], kw: Kw, h: Host): Value
  iter?(h: Host): Value[]
  len?(h: Host): number
  contains?(v: Value, h: Host): boolean
  truthy?(h: Host): boolean
  enter?(h: Host): Value
  exit?(h: Host): void
  /** np.asarray(x) — lets NumPy functions and plotting take this object */
  toArray?(): NDArray
  /** isinstance(x, T) for a library type named name */
  isa?(name: string): boolean
  /** a richer rendering for the output pane (a table, a figure …) when this is the cell's last value */
  display?(): Display | null
}

/** What a library sees of the interpreter */
export interface Host {
  err(type: string, msg: string): PyError
  call(f: Value, args: Value[], kw?: Kw): Value
  iterate(v: Value): Value[]
  truthy(v: Value): boolean
  toInt(v: Value, what?: string): number
  /** a Python number (or a one-element array / tensor) as a JS number */
  num(v: Value, what?: string): number
  str(v: Value): string
  getAttr(obj: Value, name: string): Value
  module(name: string): Value
  /** current line, and the source text of the call or operator being evaluated */
  readonly line: number
  readonly code: string
  /** false inside a loop after its first iterations, so a 100-epoch loop does not record 100 copies of each step */
  readonly tracing: boolean
  /** the variable name a value is bound to, or the given fallback */
  nameOf(v: Value, fallback: string): string
  emit(ev: PyEventInput): void
  traceCall(api: string, kind: ApiKind, operands: Operand[], result: GridSnapshot | null, resultText: string, axis?: number | null): void
  print(s: string): void
  readonly rng: SandboxRandom
  /** per-run state a library keeps between calls (pyplot's current figure …) */
  readonly state: Map<string, unknown>
  /** the run's array size limit */
  readonly maxSize: number
}

/** A library the sandbox can import: the dotted module names it provides and how to build them (once per run) */
export interface PyLib {
  modules: string[]
  load(h: Host): Record<string, Value>
  /** outputs to show after the run (pyplot's open figures) */
  finish?(h: Host): Display[]
}

const NONE: Value = { k: 'none' }
const int = (v: number): Value => ({ k: 'int', v })
const float = (v: number): Value => ({ k: 'float', v })
const bool = (v: boolean): Value => ({ k: 'bool', v })
const arr = (a: NDArray): Value => ({ k: 'array', a })
const tuple = (items: Value[]): Value => ({ k: 'tuple', items })
const str_ = (v: string): Value => ({ k: 'str', v })

/** constructors and helpers for library code */
export const py = { NONE, int, float, bool, arr, tuple, str: str_, list: (items: Value[]): Value => ({ k: 'list', items }), obj: (o: PyObj): Value => ({ k: 'obj', o }) }

export const typeName = (v: Value): string => {
  switch (v.k) {
    case 'none': return 'NoneType'
    case 'array': return 'numpy.ndarray'
    case 'fn': return 'builtin_function_or_method'
    case 'type': case 'class': return 'type'
    case 'dtype': return 'numpy.dtype'
    case 'poly': return 'Polynomial'
    case 'carray': return 'numpy.ndarray'
    case 'func': return 'function'
    case 'bound': return 'method'
    case 'inst': return v.cls.name
    case 'obj': return v.o.cls
    default: return v.k
  }
}

const fromScalar = (x: number, dtype: DType): Value =>
  dtype === 'bool' ? bool(x !== 0) : dtype === 'int64' ? int(x) : float(x)

type NumValue = Extract<Value, { k: 'int' | 'float' | 'bool' }>
export const isNum = (v: Value): v is NumValue => v.k === 'int' || v.k === 'float' || v.k === 'bool'
export const numOf = (v: Value): number => (v.k === 'bool' ? (v.v ? 1 : 0) : v.k === 'int' || v.k === 'float' ? v.v : NaN)
const scalarDType = (v: Value): DType => (v.k === 'bool' ? 'bool' : v.k === 'int' ? 'int64' : 'float64')

const reprStr = (s: string) => {
  const q = s.includes("'") && !s.includes('"') ? '"' : "'"
  const body = s.replace(/\\/g, '\\\\').replace(/\n/g, '\\n').replace(/\t/g, '\\t')
  return q + (q === "'" ? body.replace(/'/g, "\\'") : body) + q
}

const rangeLen = (r: { start: number; stop: number; step: number }) => Math.max(0, Math.ceil((r.stop - r.start) / r.step))

export const repr = (v: Value): string => {
  switch (v.k) {
    case 'int': return String(v.v)
    case 'float': return formatScalar(v.v, 'float64')
    case 'bool': return v.v ? 'True' : 'False'
    case 'none': return 'None'
    case 'str': return reprStr(v.v)
    case 'list': return `[${v.items.map(repr).join(', ')}]`
    case 'tuple': return v.items.length === 1 ? `(${repr(v.items[0])},)` : `(${v.items.map(repr).join(', ')})`
    case 'dict': return `{${v.d.items().map(([k, x]) => `${repr(k)}: ${repr(x)}`).join(', ')}}`
    case 'range': return v.step === 1 ? `range(${v.start}, ${v.stop})` : `range(${v.start}, ${v.stop}, ${v.step})`
    case 'slice': return `slice(${repr(v.start)}, ${repr(v.stop)}, ${repr(v.step)})`
    case 'ellipsis': return 'Ellipsis'
    case 'array': return arrayRepr(v.a)
    case 'fn': return `<built-in function ${v.name}>`
    case 'type': return `<class '${v.name}'>`
    case 'module': return `<module '${v.name}'>`
    case 'dtype': return `dtype('${v.d}')`
    case 'poly': return polyRepr(v.coef, v.domain, v.window)
    case 'carray': return complexRepr(v.re, v.im)
    case 'func': return `<function ${v.owner ? `${v.owner.name}.` : ''}${v.name}>`
    case 'bound': return `<bound method ${v.f.owner ? `${v.f.owner.name}.` : ''}${v.f.name} of ${repr(v.self)}>`
    case 'class': return `<class '__main__.${v.name}'>`
    case 'inst': return v.cls.host?.repr?.(v) ?? `<__main__.${v.cls.name} object>`
    case 'super': return `<super: <class '${v.after.name}'>, <${v.self.cls.name} object>>`
    case 'obj': return v.o.repr()
  }
}

export const str = (v: Value): string => {
  if (v.k === 'str') return v.v
  if (v.k === 'array') return arrayStr(v.a)
  if (v.k === 'dtype') return v.d
  if (v.k === 'poly') return polyStr(v.coef)
  if (v.k === 'obj') return v.o.str?.() ?? v.o.repr()
  return repr(v)
}

const groupThousands = (s: string, sep: string) => s.replace(/^(-?\d+)/, (m) => m.replace(/\B(?=(\d{3})+(?!\d))/g, sep))

/** Python's format-spec mini-language for numbers and strings: [[fill]align][sign][0][width][,][.precision][type] */
export const formatSpec = (v: Value, spec: string): string => {
  if (!spec) return str(v)
  if (v.k === 'obj') {
    const s = v.o.format?.(spec)
    if (s !== undefined) return s
  }
  const m = /^(?:(.)?([<>^=]))?([+\- ])?(#)?(0)?(\d+)?([,_])?(?:\.(\d+))?([bcdeEfFgGnosxX%])?$/.exec(spec)
  if (!m) throw new PyError('ValueError', `Invalid format specifier '${spec}'`)
  const [, fill0, align0, sign, , zero, widthS, group, precS, type] = m
  const width = widthS ? Number(widthS) : 0
  const prec = precS !== undefined ? Number(precS) : null
  let body: string
  const numeric = isNum(v) || (v.k === 'array' && v.a.size === 1)
  if (numeric) {
    const x = v.k === 'array' ? v.a.values()[0] : numOf(v)
    const isInt = v.k === 'int' || v.k === 'bool' || (v.k === 'array' && v.a.dtype !== 'float64')
    const t = type ?? (isInt && prec === null ? 'd' : prec === null ? '' : 'g')
    const ax = Math.abs(x)
    const fixed = (p: number) => (ax === Infinity ? 'inf' : Number.isNaN(x) ? 'nan' : ax.toFixed(p))
    const expo = (p: number, up = false) => {
      const s = ax.toExponential(p).replace(/e([+-])(\d)$/, 'e$10$2')
      return up ? s.toUpperCase() : s
    }
    switch (t) {
      case 'd': case 'n':
        if (!isInt) throw new PyError('ValueError', `Unknown format code 'd' for object of type 'float'`)
        body = String(ax)
        break
      case 'f': case 'F': body = fixed(prec ?? 6); break
      case 'e': case 'E': body = expo(prec ?? 6, t === 'E'); break
      case '%': body = (ax * 100).toFixed(prec ?? 6) + '%'; break
      case 'g': case 'G': {
        const p = prec === 0 ? 1 : prec ?? 6
        const e = ax === 0 ? 0 : Math.floor(Math.log10(ax))
        body = e < -4 || e >= p ? expo(p - 1, t === 'G').replace(/\.?0+e/, 'e') : ax.toFixed(Math.max(0, p - 1 - e)).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '')
        break
      }
      case 'x': body = ax.toString(16); break
      case 'b': body = ax.toString(2); break
      default: body = repr(isInt ? int(ax) : float(ax))
    }
    if (group) body = groupThousands(body, group)
    const neg = x < 0 || Object.is(x, -0)
    const signStr = neg ? '-' : sign === '+' ? '+' : sign === ' ' ? ' ' : ''
    if (zero && !align0) return signStr + body.padStart(width - signStr.length, '0')
    body = signStr + body
    const align = align0 ?? '>'
    return pad(body, width, fill0 ?? ' ', align)
  }
  if (type && type !== 's') throw new PyError('ValueError', `Unknown format code '${type}' for object of type '${typeName(v)}'`)
  body = str(v)
  if (prec !== null) body = body.slice(0, prec)
  return pad(body, width, fill0 ?? ' ', align0 ?? '<')
}

/** operator → the dunder method a user class can define for it */
const DUNDER: Record<string, string> = {
  '+': '__add__', '-': '__sub__', '*': '__mul__', '/': '__truediv__', '@': '__matmul__', '**': '__pow__',
  '==': '__eq__', '<': '__lt__', '>': '__gt__', '<=': '__le__', '>=': '__ge__',
}

const escRe = (s: string) => s.replace(/[\\^$.*+?()[\]{}|-]/g, '\\$&')

const pad = (s: string, width: number, fill: string, align: string): string => {
  const n = width - s.length
  if (n <= 0) return s
  if (align === '<') return s + fill.repeat(n)
  if (align === '^') return fill.repeat(Math.floor(n / 2)) + s + fill.repeat(Math.ceil(n / 2))
  return fill.repeat(n) + s
}

/** Python 值（嵌套 list / 标量 / 数组）→ NDArray */
/** library objects (tensors, Series …) and ranges as plain arrays / lists */
const asArrayLike = (x: Value): Value => {
  if (x.k === 'obj' && x.o.toArray) return arr(x.o.toArray())
  if (x.k === 'range') {
    const n = rangeLen(x)
    checkSize([n])
    return { k: 'list', items: Array.from({ length: n }, (_, i) => int(x.start + i * x.step)) }
  }
  return x
}

export const toArray = (v0: Value, dtype?: DType): NDArray => {
  const v = asArrayLike(v0)
  if (v.k === 'array') return dtype && dtype !== v.a.dtype ? NDArray.create(v.a.values(), v.a.shape, dtype) : v.a
  const flat: number[] = []
  let kind: DType = 'bool'
  const promote = (d: DType) => {
    if (d === 'float64' || (d === 'int64' && kind === 'bool')) kind = d
  }
  const shapeOf = (x0: Value, prefix: number[]): number[] => {
    const x = asArrayLike(x0)
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
  const fill = (x0: Value) => {
    const x = asArrayLike(x0)
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

export interface Operand {
  label: string
  snap: GridSnapshot
}

/** 一次 API 调用 / 运算符 / 下标读取的记录，供 API 可视化面板使用 */
export interface CallTrace {
  id: number
  line: number
  code: string
  /** 'np.sum'、'ndarray.reshape'、'op:+'、'ndarray.T'、'index' … */
  api: string
  kind: ApiKind
  /** 参与运算的数组（逐元素运算时也包括标量），按出现顺序 */
  operands: Operand[]
  axis: number | null
  /** 数组或标量结果；返回元组等其他类型时为 null */
  result: GridSnapshot | null
  resultText: string
  /** kind = 'move'：结果每个元素拷贝自 [第几个操作数, 其扁平下标] */
  source?: [number, number][]
  /** kind = 'poly'：要画的曲线（普通 x 的系数，低次在前）、数据点与标记点 */
  plot?: PolyPlot
}

export interface PolyPlot {
  curves: { label: string; coef: number[] }[]
  points?: { x: number[]; y: number[] }
  marks?: { x: number; y: number }[]
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
  /** the final value drawn richly (a DataFrame as a table …) */
  outDisplay: Display | null
  error: { type: string; message: string; line?: number } | null
  traces: IndexTrace[]
  calls: CallTrace[]
  /** library events: figures, model shape flows, autograd graphs … */
  events: PyEvent[]
  /** what the libraries show after the run (open figures) */
  displays: Display[]
  vars: VarInfo[]
}

const snap = (a: NDArray): GridSnapshot => ({ shape: [...a.shape], values: a.values(), dtype: a.dtype })

// ================= 解释器 =================

/** how many runs of one source line keep recording traces (a loop body records its first iterations only) */
const TRACE_REPEATS = 2
/** total traces + calls + events kept per run */
const MAX_RECORDS = 600
const MAX_DEPTH = 120

class BreakSig {}
class ContinueSig {}
class ReturnSig {
  v: Value
  constructor(v: Value) { this.v = v }
}

export interface RunOptions {
  libs?: PyLib[]
  /** element limit for one array (default MAX_SIZE); the deep-learning playgrounds raise it */
  maxSize?: number
  /** wall-clock budget in ms before a TimeoutError (default 2000) */
  timeBudget?: number
}

class Interp implements Host {
  src: string
  env = new Map<string, Value>()
  frame: Frame | null = null
  stdout: string[] = []
  traces: IndexTrace[] = []
  calls: CallTrace[] = []
  events: PyEvent[] = []
  line = 0
  quiet = 0
  /** >0 while re-running a call on id arrays to recover element provenance */
  replaying = 0
  rng = new SandboxRandom()
  /** 多项式调用把要画的曲线放在这里，由 pushCall 取走 */
  pendingPlot: PolyPlot | null = null
  builtins: Record<string, Value>
  np: Value
  libs: PyLib[]
  loaded = new Map<PyLib, Record<string, Value>>()
  state = new Map<string, unknown>()
  maxSize: number
  lineRuns = new Map<number, number>()
  callText = ''
  steps = 0
  depth = 0
  deadline: number

  constructor(src: string, opts: RunOptions = {}) {
    this.src = src
    this.libs = opts.libs ?? []
    this.maxSize = opts.maxSize ?? MAX_SIZE
    this.deadline = Date.now() + (opts.timeBudget ?? 2000)
    this.np = this.makeNumpy()
    this.builtins = this.makeBuiltins()
  }

  err(type: string, msg: string): PyError {
    return new PyError(type, msg, this.line)
  }

  text(n: Span) {
    return this.src.slice(n.s, n.e)
  }

  // ---------- 作用域、追踪开关、库接口 ----------

  lookup(id: string): Value | undefined {
    for (let f = this.frame; f; f = f.parent) {
      if (f.isClass && f !== this.frame) continue
      const v = f.vars.get(id)
      if (v) return v
    }
    return this.env.get(id) ?? this.builtins[id]
  }

  bind(id: string, v: Value) {
    ;(this.frame ? this.frame.vars : this.env).set(id, v)
  }

  get tracing(): boolean {
    return !this.quiet && !this.replaying && (this.lineRuns.get(this.line) ?? 0) <= TRACE_REPEATS && this.traces.length + this.calls.length + this.events.length < MAX_RECORDS
  }

  get code(): string {
    return this.callText
  }

  tick() {
    if ((++this.steps & 1023) === 0 && Date.now() > this.deadline) {
      throw this.err('TimeoutError', 'the code ran too long for this sandbox (an endless loop, or too many iterations?)')
    }
  }

  emit(ev: PyEventInput) {
    if (this.events.length + this.calls.length + this.traces.length >= MAX_RECORDS) return
    this.events.push({ ...ev, id: this.events.length, line: this.line, code: ev.code ?? this.callText } as PyEvent)
  }

  traceCall(api: string, kind: ApiKind, operands: Operand[], result: GridSnapshot | null, resultText: string, axis: number | null = null) {
    if (!this.tracing) return
    this.calls.push({ id: this.calls.length, line: this.line, code: this.callText, api, kind, operands, axis, result, resultText })
  }

  print(s: string) {
    this.stdout.push(s)
  }

  num(v: Value, what = 'a number'): number {
    if (isNum(v)) return numOf(v)
    if (v.k === 'array' && v.a.size === 1) return v.a.values()[0]
    if (v.k === 'obj' && v.o.toArray) {
      const a = v.o.toArray()
      if (a.size === 1) return a.values()[0]
    }
    throw this.err('TypeError', `expected ${what}, got '${typeName(v)}'`)
  }

  nameOf(v: Value, fallback: string): string {
    const same = (x: Value) => x === v || (x.k === 'obj' && v.k === 'obj' && x.o === v.o) || (x.k === 'array' && v.k === 'array' && x.a === v.a)
    for (let f = this.frame; f; f = f.parent) for (const [n, x] of f.vars) if (same(x)) return n
    for (const [n, x] of this.env) if (same(x)) return n
    return fallback
  }

  /** import by dotted name: numpy (built in), then the run's libraries */
  module(name: string): Value {
    const np = this.np as Value & { k: 'module' }
    if (name === 'numpy') return np
    if (name.startsWith('numpy.')) {
      const sub = np.attrs[name.slice(6)]
      if (sub?.k === 'module') return sub
    }
    if (name === 'math') return this.builtins.__math__
    if (name === 'random') return this.builtins.__random__
    const lib = this.libs.find((l) => l.modules.includes(name))
    if (!lib) {
      const have = ['numpy', 'math', 'random', ...this.libs.flatMap((l) => l.modules.filter((m) => !m.includes('.')))]
      throw this.err('ModuleNotFoundError', `No module named '${name}' (this sandbox has ${[...new Set(have)].join(', ')})`)
    }
    let mods = this.loaded.get(lib)
    if (!mods) {
      mods = lib.load(this)
      this.loaded.set(lib, mods)
    }
    return mods[name]
  }

  str(v: Value): string {
    if (v.k === 'inst') {
      const f = this.classAttr(v.cls, '__str__') ?? this.classAttr(v.cls, '__repr__')
      if (f && f.k === 'func') return str(this.callFunc(f, [v], {}))
    }
    if (v.k === 'list' || v.k === 'tuple' || v.k === 'dict') return repr(v)
    return str(v)
  }

  // ---------- 基本协议 ----------

  truthy(v: Value): boolean {
    switch (v.k) {
      case 'bool': return v.v
      case 'int': case 'float': return v.v !== 0
      case 'none': return false
      case 'str': return v.v.length > 0
      case 'list': case 'tuple': return v.items.length > 0
      case 'dict': return v.d.size > 0
      case 'range': return rangeLen(v) > 0
      case 'obj': return v.o.truthy ? v.o.truthy(this) : v.o.len ? v.o.len(this) > 0 : true
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
    if (l.k === 'obj' || r.k === 'obj') {
      const res = (l.k === 'obj' ? l.o.binop?.(op, r, false, this) : undefined) ?? (r.k === 'obj' ? r.o.binop?.(op, l, true, this) : undefined)
      if (res) return res
      if (op === '==' || op === '!=') return bool((op === '==') === (l.k === 'obj' && r.k === 'obj' && l.o === r.o))
      throw this.err('TypeError', `unsupported operand type(s) for ${op}: '${typeName(l)}' and '${typeName(r)}'`)
    }
    if (l.k === 'inst' || r.k === 'inst') {
      const dunder = DUNDER[op]
      const f = dunder && l.k === 'inst' ? this.classAttr(l.cls, dunder) : undefined
      if (f && f.k === 'func') return this.callFunc(f, [l, r], {})
      if (op === '==' || op === '!=') return bool((op === '==') === (l === r))
      throw this.err('TypeError', `unsupported operand type(s) for ${op}: '${typeName(l)}' and '${typeName(r)}'`)
    }
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
      if (r.k === 'list' || r.k === 'tuple') found = r.items.some((x) => x === l || this.truthy(this.binop('==', x, l)))
      else if (r.k === 'dict') found = r.d.has(l)
      else if (r.k === 'range') {
        const x = isNum(l) ? numOf(l) : NaN
        found = Number.isInteger(x) && (r.step > 0 ? x >= r.start && x < r.stop : x <= r.start && x > r.stop) && (x - r.start) % r.step === 0
      } else if (r.k === 'obj' && r.o.contains) found = r.o.contains(l, this)
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
      if (this.tracing) {
        const out = typeof value === 'number' ? { shape: [], values: [value], dtype: a.dtype } : snap(value)
        this.record(this.targetOf(node), this.text(node), a, before, plan, 'read', out)
        const flatOf = new Map(a.addresses().map((x, i) => [x, i]))
        this.pushCall('index', 'move', this.text(node), [{ label: this.targetOf(node), v: arr(NDArray.create(before.values, before.shape, before.dtype)) }], res, null, plan.addresses.map((x) => [0, flatOf.get(x)!]))
      }
      return res
    }
    if (obj.k === 'list' || obj.k === 'tuple') {
      const r = this.seqIndex(obj.items, idx, obj.k)
      return Array.isArray(r) ? ({ k: obj.k, items: r } as Value) : r
    }
    if (obj.k === 'dict') {
      const v = obj.d.get(idx)
      if (!v) throw this.err('KeyError', repr(idx))
      return v
    }
    if (obj.k === 'range') {
      const n = rangeLen(obj)
      const items = Array.from({ length: Math.min(n, this.maxSize) }, (_, i) => int(obj.start + i * obj.step))
      const r = this.seqIndex(items, idx, 'range object')
      return Array.isArray(r) ? { k: 'list', items: r } : r
    }
    if (obj.k === 'obj') {
      if (!obj.o.getItem) throw this.err('TypeError', `'${obj.o.cls}' object is not subscriptable`)
      this.callText = this.text(node)
      return obj.o.getItem(idx, this)
    }
    if (obj.k === 'str') {
      const r = this.seqIndex([...obj.v].map((c) => ({ k: 'str', v: c }) as Value), idx, 'string')
      return Array.isArray(r) ? { k: 'str', v: r.map((c) => (c as { v: string }).v).join('') } : r
    }
    throw this.err('TypeError', `'${typeName(obj)}' object is not subscriptable`)
  }

  setItem(node: Node & { k: 'sub' }, obj: Value, idx: Value, val: Value, codeNode?: Span) {
    if (obj.k === 'dict') {
      obj.d.set(idx, val)
      return
    }
    if (obj.k === 'obj') {
      if (!obj.o.setItem) throw this.err('TypeError', `'${obj.o.cls}' object does not support item assignment`)
      this.callText = this.text(codeNode ?? node)
      obj.o.setItem(idx, val, this)
      return
    }
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
    if (!this.tracing) return
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

  // ---------- API 调用追踪 ----------

  /** 收集参与运算的数组（逐元素运算时也收集数值标量），标签取自源码 */
  collectOperands(nodes: Node[], vals: Value[], scalars: boolean): { label: string; v: Value }[] {
    const out: { label: string; v: Value }[] = []
    const seen = new Set<NDArray>()
    const walk = (node: Node | null, v: Value, label: string) => {
      if (v.k === 'array') {
        if (!seen.has(v.a)) {
          seen.add(v.a)
          out.push({ label, v })
        }
      } else if ((v.k === 'list' || v.k === 'tuple') && v.items.some((x) => x.k === 'array')) {
        const lit = node && (node.k === 'list' || node.k === 'tuple') ? node.items : null
        v.items.forEach((x, i) => walk(lit ? lit[i] : null, x, lit ? this.text(lit[i]) : `${label}[${i}]`))
      } else if (scalars && isNum(v)) {
        out.push({ label, v })
      }
    }
    vals.forEach((v, i) => walk(nodes[i] ?? null, v, nodes[i] ? this.text(nodes[i]) : `arg${i}`))
    return out
  }

  resultSnap(v: Value): GridSnapshot | null {
    if (v.k === 'array') return snap(v.a)
    if (isNum(v)) return { shape: [], values: [numOf(v)], dtype: v.k === 'float' ? 'float64' : v.k === 'bool' ? 'bool' : 'int64' }
    return null
  }

  pushCall(api: string, kind: ApiKind, code: string, ops: { label: string; v: Value }[], res: Value, axis: number | null, source?: [number, number][]) {
    const plot = this.pendingPlot ?? undefined
    this.pendingPlot = null
    this.calls.push({
      id: this.calls.length,
      line: this.line,
      code,
      api,
      kind,
      operands: ops.map((o) => ({ label: o.label, snap: o.v.k === 'array' ? snap(o.v.a) : this.resultSnap(o.v)! })),
      axis,
      result: this.resultSnap(res),
      resultText: repr(res),
      source,
      plot,
    })
  }

  /** 在「元素编号」数组上重放一次搬运类调用，得到结果每个元素的来源 */
  replayMove(ops: { v: Value }[], run: (subst: (v: Value) => Value) => Value): [number, number][] | undefined {
    const ID = 1_000_000
    const ids = new Map<NDArray, NDArray>()
    ops.forEach((o, k) => {
      if (o.v.k === 'array') ids.set(o.v.a, NDArray.create(o.v.a.values().map((_, f) => k * ID + f), o.v.a.shape, 'int64'))
    })
    const subst = (v: Value): Value => {
      if (v.k === 'array') return ids.has(v.a) ? arr(ids.get(v.a)!) : v
      if (v.k === 'list' || v.k === 'tuple') return { k: v.k, items: v.items.map(subst) } as Value
      return v
    }
    this.replaying++
    try {
      const r = run(subst)
      if (r.k !== 'array') return undefined
      return r.a.values().map((id) => [Math.floor(id / ID), id % ID] as [number, number])
    } catch {
      return undefined
    } finally {
      this.replaying--
    }
  }

  tracedCall(f: Value & { k: 'fn' | 'type' }, n: Node & { k: 'call' }, args: Value[], kw: Record<string, Value>): Value {
    const meta = f.api!
    const self = f.k === 'fn' ? f.self : undefined
    const selfNode = self && n.fn.k === 'attr' ? n.fn.obj : null
    const nodes = selfNode ? [selfNode, ...n.args] : n.args
    const vals = self ? [arr(self), ...args] : args
    const ops = this.collectOperands(nodes, vals, meta.kind === 'elementwise')
    // 先拍快照：调用本身可能改动输入（如原地操作）
    const opSnaps = ops.map((o) => ({ label: o.label, v: o.v.k === 'array' ? arr(o.v.a.copy()) : o.v }))
    const res = f.call(args, kw)
    let axis: number | null = null
    if (meta.axisPos !== undefined) {
      const av = kw.axis ?? args[meta.axisPos]
      axis = av === undefined || av.k === 'none' ? meta.defaultAxis ?? null : isNum(av) ? numOf(av) : null
    }
    let source: [number, number][] | undefined
    if (meta.kind === 'move') {
      source = this.replayMove(ops, (subst) => {
        const call = f.k === 'fn' && f.self && f.rebind ? f.rebind((subst(arr(f.self)) as { a: NDArray }).a) : f.call
        const skw: Record<string, Value> = {}
        for (const k in kw) skw[k] = subst(kw[k])
        return call(args.map(subst), skw)
      })
    }
    this.pushCall(meta.name, meta.kind, this.text(n), opSnaps, res, axis, source)
    return res
  }

  // ---------- 求值 ----------

  eval(n: Node): Value {
    switch (n.k) {
      case 'num': return n.isFloat ? float(n.v) : int(n.v)
      case 'str': return { k: 'str', v: n.v }
      case 'const': return n.v === 'None' ? NONE : bool(n.v === 'True')
      case 'ellipsis': return { k: 'ellipsis' }
      case 'name': {
        const v = this.lookup(n.id)
        if (!v) {
          if (n.id === 'np' || n.id === 'numpy') throw this.err('NameError', `name '${n.id}' is not defined. Did you forget 'import numpy as np'?`)
          throw this.err('NameError', `name '${n.id}' is not defined`)
        }
        return v
      }
      case 'list': return { k: 'list', items: n.items.map((x) => this.eval(x)) }
      case 'tuple': return tuple(n.items.map((x) => this.eval(x)))
      case 'dict': return { k: 'dict', d: PyDict.from(n.keys.map((key, i) => [this.eval(key), this.eval(n.vals[i])])) }
      case 'fstr':
        return str_(n.parts.map((p) => {
          if (typeof p === 'string') return p
          const v = this.eval(p.x)
          if (p.spec) return formatSpec(v, p.spec)
          return p.conv === 'r' ? repr(v) : this.str(v)
        }).join(''))
      case 'comp': return this.comprehension(n)
      case 'lambda':
        return { k: 'func', name: '<lambda>', params: n.params, defaults: n.params.map((p) => (p.def ? this.eval(p.def) : null)), body: null, expr: n.body, closure: this.closureFrame(), owner: null }
      case 'unary': {
        const x = this.eval(n.x)
        if (x.k === 'obj') {
          if (!x.o.unary) throw this.err('TypeError', `bad operand type for unary ${n.op}: '${x.o.cls}'`)
          this.callText = this.text(n)
          return x.o.unary(n.op, this)
        }
        if (x.k === 'array') {
          const res = arr(unaryOp(n.op, x.a))
          if (this.tracing && n.op !== '+') this.pushCall(`op:${n.op}x`, 'elementwise', this.text(n), [{ label: this.text(n.x), v: arr(x.a.copy()) }], res, null)
          return res
        }
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
          this.callText = this.text(n)
          result = this.compare(n.ops[i], l, r)
          if ((l.k === 'array' || r.k === 'array') && this.tracing) {
            const ops = this.collectOperands([n.xs[i], n.xs[i + 1]], [l, r], true).map((o) => ({ label: o.label, v: o.v.k === 'array' ? arr(o.v.a.copy()) : o.v }))
            this.pushCall(`op:${n.ops[i]}`, 'elementwise', n.ops.length === 1 ? this.text(n) : `${this.text(n.xs[i])} ${n.ops[i]} ${this.text(n.xs[i + 1])}`, ops, result, null)
          }
          if (i < n.ops.length - 1 && !this.truthy(result)) return result
          l = r
        }
        return result
      }
      case 'ifexp': return this.truthy(this.eval(n.cond)) ? this.eval(n.a) : this.eval(n.b)
      case 'bin': {
        const l = this.eval(n.l)
        const r = this.eval(n.r)
        if (l.k === 'poly' || r.k === 'poly') {
          const res = this.polyArith(n.op, l, r)
          if (this.tracing) {
            const curves = [[n.l, l], [n.r, r]].flatMap(([node, v]) => ((v as Value).k === 'poly' ? [this.curve(this.text(node as Node), v as Value & { k: 'poly' })] : []))
            this.pendingPlot = { curves: [...curves, this.curve(this.text(n), res)] }
            this.pushCall(`op:${n.op}`, 'poly', this.text(n), [], res, null)
          }
          return res
        }
        this.callText = this.text(n)
        const res = this.binop(n.op, l, r)
        if ((l.k === 'array' || r.k === 'array') && this.tracing) {
          const ops = this.collectOperands([n.l, n.r], [l, r], n.op !== '@').map((o) => ({ label: o.label, v: o.v.k === 'array' ? arr(o.v.a.copy()) : o.v }))
          this.pushCall(`op:${n.op}`, n.op === '@' ? 'matmul' : 'elementwise', this.text(n), ops, res, null)
        }
        return res
      }
      case 'attr': {
        const obj = this.eval(n.obj)
        const res = this.getAttr(obj, n.name)
        if (obj.k === 'array' && n.name === 'T' && this.tracing) {
          const ops = [{ label: this.text(n.obj), v: arr(obj.a.copy()) }]
          const source = this.replayMove([{ v: obj }], (subst) => this.getAttr(subst(obj), 'T'))
          this.pushCall('ndarray.T', 'move', this.text(n), ops, res, null, source)
        }
        return res
      }
      case 'call': {
        const f = this.eval(n.fn)
        const args = n.args.map((x) => this.eval(x))
        const kw: Record<string, Value> = {}
        for (const { name, v } of n.kw) kw[name] = this.eval(v)
        if (f.k === 'poly') return this.callPoly(f, n, args)
        this.callText = this.text(n)
        if ((f.k === 'fn' || f.k === 'type') && f.api && this.tracing) return this.tracedCall(f, n, args, kw)
        const line = this.line
        try {
          return this.call(f, args, kw)
        } finally {
          this.line = line
        }
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
      this.bind(t.id, v)
    } else if (t.k === 'sub') {
      this.setItem(t, this.eval(t.obj), this.eval(t.idx), v, stmtSpan)
    } else if (t.k === 'attr') {
      this.callText = stmtSpan ? this.src.slice(stmtSpan.s, stmtSpan.e) : this.text(t)
      this.setAttr(this.eval(t.obj), t.name, v)
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
    if (v.k === 'dict') return v.d.keys()
    if (v.k === 'range') {
      const n = rangeLen(v)
      if (n > 1_000_000) throw this.err('MemoryError', 'range too large for this sandbox')
      return Array.from({ length: n }, (_, i) => int(v.start + i * v.step))
    }
    if (v.k === 'obj' && v.o.iter) return v.o.iter(this)
    if (v.k === 'array') {
      if (v.a.ndim === 0) throw this.err('TypeError', 'iteration over a 0-d array')
      return Array.from({ length: v.a.shape[0] }, (_, i) => {
        const { value } = getIndex(v.a, [{ kind: 'int', value: i }])
        return typeof value === 'number' ? fromScalar(value, v.a.dtype) : arr(value)
      })
    }
    throw this.err('TypeError', `'${typeName(v)}' object is not iterable`)
  }

  // ---------- 函数、类与调用 ----------

  /** the frame a new function closes over (class bodies are not visible from methods) */
  closureFrame(): Frame | null {
    let f = this.frame
    while (f && f.isClass) f = f.parent
    return f
  }

  /** look a name up along a class's bases (depth-first, left to right) */
  classAttr(cls: ClassValue, name: string, after?: ClassValue): Value | undefined {
    let skipping = !!after
    const seen = new Set<ClassValue>()
    const walk = (c: ClassValue): Value | undefined => {
      if (seen.has(c)) return undefined
      seen.add(c)
      if (skipping) {
        if (c === after) skipping = false
      } else if (c.ns.has(name)) return c.ns.get(name)
      for (const b of c.bases) {
        if (b.k === 'class') {
          const v = walk(b)
          if (v) return v
        }
      }
      return undefined
    }
    return walk(cls)
  }

  call(f: Value, args: Value[], kw: Kw = {}): Value {
    switch (f.k) {
      case 'fn': case 'type': return f.call(args, kw)
      case 'func': return this.callFunc(f, args, kw)
      case 'bound': return this.callFunc(f.f, [f.self, ...args], kw)
      case 'class': return this.instantiate(f, args, kw)
      case 'inst': {
        const c = this.classAttr(f.cls, '__call__')
        if (c && c.k === 'func') return this.callFunc(c, [f, ...args], kw)
        if (f.cls.host?.call) return f.cls.host.call(f, args, kw, this)
        break
      }
      case 'obj':
        if (f.o.call) return f.o.call(args, kw, this)
        break
    }
    throw this.err('TypeError', `'${typeName(f)}' object is not callable`)
  }

  callFunc(f: FuncValue, args: Value[], kw: Kw): Value {
    const vars = new Map<string, Value>()
    const ps = f.params
    if (args.length > ps.length) throw this.err('TypeError', `${f.name}() takes ${ps.length} positional argument${ps.length === 1 ? '' : 's'} but ${args.length} were given`)
    ps.forEach((p, i) => {
      if (i < args.length) {
        if (p.name in kw) throw this.err('TypeError', `${f.name}() got multiple values for argument '${p.name}'`)
        vars.set(p.name, args[i])
      } else if (p.name in kw) vars.set(p.name, kw[p.name])
      else if (f.defaults[i]) vars.set(p.name, f.defaults[i]!)
    })
    for (const k in kw) if (!ps.some((p) => p.name === k)) throw this.err('TypeError', `${f.name}() got an unexpected keyword argument '${k}'`)
    const missing = ps.filter((p) => !vars.has(p.name)).map((p) => `'${p.name}'`)
    if (missing.length) throw this.err('TypeError', `${f.name}() missing ${missing.length} required positional argument${missing.length > 1 ? 's' : ''}: ${missing.join(' and ')}`)
    if (this.depth >= MAX_DEPTH) throw this.err('RecursionError', 'maximum recursion depth exceeded')
    const saved = this.frame
    this.frame = { vars, parent: f.closure, owner: f.owner, self: ps.length ? vars.get(ps[0].name) : undefined }
    this.depth++
    try {
      if (f.expr) return this.eval(f.expr)
      this.execBlock(f.body!)
      return NONE
    } catch (e) {
      if (e instanceof ReturnSig) return e.v
      throw e
    } finally {
      this.frame = saved
      this.depth--
    }
  }

  instantiate(cls: ClassValue, args: Value[], kw: Kw): Value {
    const inst: InstValue = { k: 'inst', cls, attrs: new Map(), state: {} }
    const init = this.classAttr(cls, '__init__')
    if (init && init.k === 'func') {
      const r = this.callFunc(init, [inst, ...args], kw)
      if (r.k !== 'none') throw this.err('TypeError', '__init__() should return None')
    } else if (cls.host) cls.host.init(inst, args, kw, this)
    else if (args.length || Object.keys(kw).length) throw this.err('TypeError', `${cls.name}() takes no arguments`)
    return inst
  }

  /** the library base class of a user class, if any */
  hostOf(bases: Value[]): HostClass | null {
    for (const b of bases) {
      if (b.k === 'type' && b.host) return b.host
      if (b.k === 'class' && b.host) return b.host
    }
    return null
  }

  setAttr(obj: Value, name: string, v: Value) {
    switch (obj.k) {
      case 'inst':
        obj.cls.host?.setAttr?.(obj, name, v, this)
        obj.attrs.set(name, v)
        return
      case 'class':
        obj.ns.set(name, v)
        return
      case 'obj':
        if (obj.o.setAttr?.(name, v, this)) return
        throw this.err('AttributeError', `'${obj.o.cls}' object attribute '${name}' is read-only (or does not exist in this sandbox)`)
      case 'func':
        return
    }
    throw this.err('AttributeError', `'${typeName(obj)}' object has no attribute '${name}'`)
  }

  comprehension(n: Node & { k: 'comp' }): Value {
    const out: Value[] = []
    const pairs: [Value, Value][] = []
    const saved = this.frame
    // the loop variables live in their own scope, as in Python 3
    this.frame = { vars: new Map(), parent: saved }
    try {
      const loop = (g: number) => {
        if (g === n.gens.length) {
          if (n.kind === 'dict') pairs.push([this.eval(n.elt), this.eval(n.val!)])
          else out.push(this.eval(n.elt))
          return
        }
        const gen = n.gens[g]
        for (const item of this.iterate(this.eval(gen.iter))) {
          this.tick()
          this.assign(gen.target, item)
          if (gen.conds.every((c) => this.truthy(this.eval(c)))) loop(g + 1)
        }
      }
      loop(0)
    } finally {
      this.frame = saved
    }
    return n.kind === 'dict' ? { k: 'dict', d: PyDict.from(pairs) } : { k: 'list', items: out }
  }

  // ---------- 语句 ----------

  /** run a program: the value of a final expression statement is the cell's output */
  run(stmts: Stmt[]): Value | null {
    let out: Value | null = null
    stmts.forEach((st, i) => {
      out = null
      if (st.k === 'expr' && i === stmts.length - 1) {
        this.at(st.line)
        const v = this.eval(st.x)
        if (v.k !== 'none') out = v
      } else this.execStmt(st)
    })
    return out
  }

  /** enter a source line: counts its runs (for trace throttling) and the step budget */
  at(line: number) {
    this.line = line
    this.lineRuns.set(line, (this.lineRuns.get(line) ?? 0) + 1)
    this.tick()
  }

  execBlock(stmts: Stmt[]) {
    for (const st of stmts) this.execStmt(st)
  }

  execStmt(st: Stmt) {
    this.at(st.line)
    switch (st.k) {
      case 'pass': return
      case 'break': throw new BreakSig()
      case 'continue': throw new ContinueSig()
      case 'return':
        if (!this.frame || this.frame.isClass) throw this.err('SyntaxError', "'return' outside function")
        throw new ReturnSig(st.x ? this.eval(st.x) : NONE)
      case 'import': {
        const mod = this.module(st.mod)
        if (st.alias) this.bind(st.alias, mod)
        else {
          const root = st.mod.split('.')[0]
          this.bind(root, this.module(root))
        }
        return
      }
      case 'from': {
        const mod = this.module(st.mod)
        if (mod.k !== 'module') throw this.err('ImportError', `cannot import from '${st.mod}'`)
        for (const { name, alias } of st.names) {
          let v: Value | undefined = mod.attrs[name]
          if (!v) {
            try {
              v = this.module(`${st.mod}.${name}`)
            } catch {
              throw this.err('ImportError', `cannot import name '${name}' from '${st.mod}' (not available in this sandbox)`)
            }
          }
          this.bind(alias, v)
        }
        return
      }
      case 'expr':
        this.eval(st.x)
        return
      case 'assign': {
        const v = this.eval(st.x)
        for (const t of st.targets) this.assign(t, v, { s: t.s, e: st.x.e, line: st.line })
        return
      }
      case 'aug': return this.augAssign(st)
      case 'del':
        for (const t of st.targets) this.del(t)
        return
      case 'assert':
        if (!this.truthy(this.eval(st.x))) throw this.err('AssertionError', st.msg ? this.str(this.eval(st.msg)) : '')
        return
      case 'if':
        for (const b of st.branches) {
          if (b.cond === null || this.truthy(this.eval(b.cond))) {
            this.execBlock(b.body)
            return
          }
        }
        return
      case 'for': {
        const items = this.iterate(this.eval(st.iter))
        for (const item of items) {
          this.line = st.line
          this.tick()
          this.assign(st.target, item)
          try {
            this.execBlock(st.body)
          } catch (e) {
            if (e instanceof BreakSig) break
            if (e instanceof ContinueSig) continue
            throw e
          }
        }
        return
      }
      case 'while':
        while (this.truthy(this.eval(st.cond))) {
          try {
            this.execBlock(st.body)
          } catch (e) {
            if (e instanceof BreakSig) break
            if (e instanceof ContinueSig) continue
            throw e
          }
          this.at(st.line)
        }
        return
      case 'def': {
        const f: FuncValue = {
          k: 'func', name: st.name, params: st.params, defaults: st.params.map((p) => (p.def ? this.eval(p.def) : null)),
          body: st.body, expr: null, closure: this.closureFrame(), owner: null,
        }
        this.bind(st.name, f)
        return
      }
      case 'class': {
        const bases = st.bases.map((b) => this.eval(b))
        for (const b of bases) {
          if (b.k !== 'class' && b.k !== 'type') throw this.err('TypeError', `bases must be classes, not '${typeName(b)}'`)
          if (b.k === 'type' && !b.host && b.name !== 'object') throw this.err('TypeError', `cannot subclass '${b.name}' in this sandbox`)
        }
        const cls: ClassValue = { k: 'class', name: st.name, bases, ns: new Map(), host: this.hostOf(bases) }
        const saved = this.frame
        this.frame = { vars: cls.ns, parent: saved, isClass: true }
        try {
          this.execBlock(st.body)
        } finally {
          this.frame = saved
        }
        for (const v of cls.ns.values()) if (v.k === 'func') v.owner = cls
        this.bind(st.name, cls)
        return
      }
      case 'with': {
        const exits: (() => void)[] = []
        try {
          for (const item of st.items) {
            const ctx = this.eval(item.x)
            this.callText = this.text(item.x)
            let entered: Value
            if (ctx.k === 'obj' && ctx.o.enter) {
              entered = ctx.o.enter(this)
              exits.push(() => ctx.o.exit?.(this))
            } else throw this.err('TypeError', `'${typeName(ctx)}' object does not support the context manager protocol`)
            if (item.as) this.assign(item.as, entered)
          }
          this.execBlock(st.body)
        } finally {
          for (const x of exits.reverse()) x()
        }
        return
      }
    }
  }

  augAssign(st: Stmt & { k: 'aug' }) {
    const span = { s: st.s, e: st.e, line: st.line }
    this.callText = this.src.slice(st.s, st.e)
    const t = st.target
    if (t.k === 'name') {
      const cur = this.eval(t)
      const rhs = this.eval(st.x)
      this.callText = this.src.slice(st.s, st.e)
      if (cur.k === 'array') this.inplace(t.id, cur.a, st.op, rhs, span)
      else if (cur.k === 'obj' && cur.o.inplace?.(st.op, rhs, this)) return
      else if (cur.k === 'list' && st.op === '+') cur.items.push(...this.iterate(rhs))
      else this.bind(t.id, this.binop(st.op, cur, rhs))
    } else if (t.k === 'attr') {
      const obj = this.eval(t.obj)
      const cur = this.getAttr(obj, t.name)
      const rhs = this.eval(st.x)
      this.callText = this.src.slice(st.s, st.e)
      if (cur.k === 'obj' && cur.o.inplace?.(st.op, rhs, this)) return
      if (cur.k === 'array') {
        this.inplace(this.text(t), cur.a, st.op, rhs, span)
        return
      }
      this.setAttr(obj, t.name, this.binop(st.op, cur, rhs))
    } else if (t.k === 'sub') {
      const obj = this.eval(t.obj)
      const idx = this.eval(t.idx)
      this.quiet++
      let cur: Value
      try {
        cur = this.getItem(t, obj, idx)
      } finally {
        this.quiet--
      }
      const v = this.binop(st.op, cur, this.eval(st.x))
      if (obj.k === 'array' && v.k === 'array' && obj.a.dtype !== 'float64' && v.a.dtype === 'float64') {
        throw this.err('UFuncTypeError', `Cannot cast ufunc output from dtype('float64') to dtype('${obj.a.dtype}') with casting rule 'same_kind'`)
      }
      this.setItem(t, obj, idx, v, span)
    }
  }

  del(t: Node) {
    if (t.k === 'name') {
      const scope = this.frame ? this.frame.vars : this.env
      if (!scope.delete(t.id)) throw this.err('NameError', `name '${t.id}' is not defined`)
    } else if (t.k === 'sub') {
      const obj = this.eval(t.obj)
      const idx = this.eval(t.idx)
      if (obj.k === 'dict') {
        if (!obj.d.delete(idx)) throw this.err('KeyError', repr(idx))
      } else if (obj.k === 'list' && (idx.k === 'int' || idx.k === 'bool')) {
        const n = obj.items.length
        const i = numOf(idx)
        if (i < -n || i >= n) throw this.err('IndexError', 'list assignment index out of range')
        obj.items.splice(i < 0 ? i + n : i, 1)
      } else throw this.err('TypeError', `'${typeName(obj)}' object doesn't support item deletion`)
    } else if (t.k === 'attr') {
      const obj = this.eval(t.obj)
      if (obj.k !== 'inst' || !obj.attrs.delete(t.name)) throw this.err('AttributeError', t.name)
    } else for (const x of (t as { items: Node[] }).items ?? []) this.del(x)
  }

  // ---------- 属性与方法 ----------

  kwInt(kw: Record<string, Value>, args: Value[], pos: number, name: string): number | null {
    const v = kw[name] ?? args[pos]
    if (!v || v.k === 'none') return null
    return this.toInt(v)
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
    if (typeof res === 'number') return fromScalar(res, reduceDType(kind, a.dtype))
    return arr(res)
  }

  /** ndarray 方法表：impl(self) 返回可调用对象，便于在元素编号数组上重放 */
  arrayMethod(name: string): { impl: (a: NDArray) => Fn; meta: ApiMeta } | null {
    const M = (kind: ApiKind, impl: (a: NDArray) => Fn, extra: Partial<ApiMeta> = {}) => ({ impl, meta: { name: `ndarray.${name}`, kind, ...extra } })
    switch (name) {
      case 'reshape': return M('move', (a) => (args, kw) => arr(reshape(a, this.toShape(kw.shape ? [kw.shape] : args))))
      case 'transpose': return M('move', (a) => (args) => arr(transpose(a, args.length ? this.toShape(args) : undefined)))
      case 'copy': return M('move', (a) => () => arr(a.copy()))
      case 'flatten': return M('move', (a) => () => arr(NDArray.create(a.values(), [a.size], a.dtype)))
      case 'ravel': return M('move', (a) => () => arr(reshape(a, [-1])))
      case 'squeeze': return M('move', (a) => (args, kw) => arr(squeeze(a, this.kwInt(kw, args, 0, 'axis'))), { axisPos: 0, defaultAxis: null })
      case 'swapaxes': return M('move', (a) => (args) => arr(swapaxes(a, this.toInt(args[0]), this.toInt(args[1]))))
      case 'repeat': return M('move', (a) => (args, kw) => arr(repeat(a, this.toInt(kw.repeats ?? args[0]), this.kwInt(kw, args, 1, 'axis'))), { axisPos: 1, defaultAxis: null })
      case 'astype': return M('elementwise', (a) => (args, kw) => arr(NDArray.create(a.values(), a.shape, this.dtypeOf(kw.dtype ?? args[0]) ?? a.dtype)))
      case 'round': return M('elementwise', (a) => (args, kw) => arr(this.roundArr(a, this.kwInt(kw, args, 0, 'decimals') ?? 0)))
      case 'clip': return M('elementwise', (a) => (args, kw) => this.clip(arr(a), kw.min ?? kw.a_min ?? args[0], kw.max ?? kw.a_max ?? args[1]))
      case 'tolist': return M('info', (a) => () => this.toList(a))
      case 'item': return M('info', (a) => () => {
        if (a.size !== 1) throw this.err('ValueError', 'can only convert an array of size 1 to a Python scalar')
        return fromScalar(a.values()[0], a.dtype)
      })
      case 'nonzero': return M('info', (a) => () => this.nonzero(a))
      case 'sum': case 'mean': case 'max': case 'min': case 'argmax': case 'argmin': case 'any': case 'all': case 'prod': case 'std': case 'var':
        return M('reduce', (a) => (args, kw) => this.reduceFn(a, name, args, kw), { axisPos: 0, defaultAxis: null })
      case 'cumsum': return M('scan', (a) => (args, kw) => arr(cumsum(a, this.kwInt(kw, args, 0, 'axis'))), { axisPos: 0, defaultAxis: null })
      case 'argsort': return M('sort', (a) => (args, kw) => {
        const v = kw.axis ?? args[0]
        return arr(sortAlong(a, v === undefined ? -1 : v.k === 'none' ? null : this.toInt(v)).order)
      }, { axisPos: 0, defaultAxis: -1 })
      case 'dot': return M('matmul', (a) => (args) => this.binop('@', arr(a), args[0]))
    }
    return null
  }

  roundArr(a: NDArray, decimals: number): NDArray {
    return NDArray.create(a.values().map((v) => roundHalfEven(v, decimals)), a.shape, a.dtype === 'bool' ? 'int64' : a.dtype)
  }

  /** np.maximum / np.minimum：逐元素取大 / 取小（NaN 传播），支持广播 */
  extremum(l: Value, r: Value, pick: (x: number, y: number) => number): Value {
    const a = toArray(l)
    const b = toArray(r)
    const shape = broadcastShapes([a.shape, b.shape])
    const av = broadcastTo(a, shape).values()
    const bv = broadcastTo(b, shape).values()
    const res = NDArray.create(av.map((x, i) => (Number.isNaN(x) || Number.isNaN(bv[i]) ? NaN : pick(x, bv[i]))), shape, commonDType([a, b]))
    return l.k === 'array' || r.k === 'array' || l.k === 'list' || r.k === 'list' ? arr(res) : fromScalar(res.values()[0], res.dtype)
  }

  clip(a: Value, lo: Value | undefined, hi: Value | undefined): Value {
    let v = a
    if (lo && lo.k !== 'none') v = this.extremum(v, lo, Math.max)
    if (hi && hi.k !== 'none') v = this.extremum(v, hi, Math.min)
    return v
  }

  // ---------- numpy.polynomial ----------

  makePoly(coef: number[], domain: Interval = DEFAULT_IV, window: Interval = DEFAULT_IV): Value & { k: 'poly' } {
    return { k: 'poly', coef: trim(coef), domain, window }
  }

  /** 画图用：换算成普通 x 的系数 */
  curve(label: string, p: Value): { label: string; coef: number[] } {
    const q = p as Value & { k: 'poly' }
    return { label, coef: convert(q.coef, q.domain, q.window) }
  }

  coefList(v: Value | undefined, name: string): number[] {
    if (!v) throw this.err('TypeError', `${name}() missing required argument 'coef'`)
    if (isNum(v)) return [numOf(v)]
    return toArray(v, 'float64').values()
  }

  /** p(x)：x 可以是数或数组 */
  callPoly(p: Value & { k: 'poly' }, n: Node & { k: 'call' }, args: Value[]): Value {
    const x = args[0]
    if (!x) throw this.err('TypeError', "__call__() missing 1 required positional argument: 'arg'")
    const { off, scl } = mapParams(p.domain, p.window)
    const ev = (t: number) => polyval(p.coef, off + scl * t)
    let res: Value
    let marks: { x: number; y: number }[]
    if (isNum(x)) {
      res = float(ev(numOf(x)))
      marks = [{ x: numOf(x), y: numOf(res) }]
    } else {
      const xa = toArray(x, 'float64')
      const ys = xa.values().map(ev)
      res = arr(NDArray.create(ys, xa.shape, 'float64'))
      marks = xa.values().map((t, i) => ({ x: t, y: ys[i] }))
    }
    if (this.tracing) {
      this.pendingPlot = { curves: [this.curve(n.fn.k === 'name' ? n.fn.id : this.text(n.fn), p)], marks }
      this.pushCall('Polynomial.__call__', 'poly', this.text(n), [], res, null)
    }
    return res
  }

  polyArith(op: string, l: Value, r: Value): Value {
    const asPoly = (v: Value): Value & { k: 'poly' } => {
      if (v.k === 'poly') return v
      if (isNum(v)) return this.makePoly([numOf(v)])
      throw this.err('TypeError', `unsupported operand type(s) for ${op}: '${typeName(l)}' and '${typeName(r)}'`)
    }
    if (op === '**') {
      if (l.k !== 'poly' || (r.k !== 'int' && r.k !== 'bool') || numOf(r) < 0) throw this.err('ValueError', 'Power must be a non-negative integer.')
      return this.makePoly(polypow(l.coef, numOf(r)), l.domain, l.window)
    }
    const a = asPoly(l)
    const b = asPoly(r)
    const base = l.k === 'poly' ? l : a
    if (l.k === 'poly' && r.k === 'poly' && (a.domain.join() !== b.domain.join() || a.window.join() !== b.window.join())) throw this.err('TypeError', 'Domains differ')
    const f = op === '+' ? polyadd : op === '-' ? polysub : op === '*' ? polymul : null
    if (!f) throw this.err('TypeError', `unsupported operand type(s) for ${op}: '${typeName(l)}' and '${typeName(r)}'`)
    return this.makePoly(f(a.coef, b.coef), base.domain, base.window)
  }

  polyAttr(p: Value & { k: 'poly' }, name: string): Value {
    const { scl } = mapParams(p.domain, p.window)
    const label = 'p'
    const method = (call: Fn): Value => ({ k: 'fn', name, call, api: { name: `Polynomial.${name}`, kind: 'poly' } })
    switch (name) {
      case 'coef': return arr(NDArray.create(p.coef, [p.coef.length], 'float64'))
      case 'domain': return arr(NDArray.create(p.domain, [2], 'float64'))
      case 'window': return arr(NDArray.create(p.window, [2], 'float64'))
      case 'degree': return { k: 'fn', name, call: () => int(p.coef.length - 1) }
      case 'deriv': return method((args, kw) => {
        const res = this.makePoly(deriv(p.coef, this.kwInt(kw, args, 0, 'm') ?? 1, scl), p.domain, p.window)
        this.pendingPlot = { curves: [this.curve(label, p), this.curve("p'", res)] }
        return res
      })
      case 'integ': return method((args, kw) => {
        const k = kw.k ?? args[1]
        const res = this.makePoly(integ(p.coef, this.kwInt(kw, args, 0, 'm') ?? 1, k ? numOf(k) : 0, 1 / scl), p.domain, p.window)
        this.pendingPlot = { curves: [this.curve(label, p), this.curve('∫p', res)] }
        return res
      })
      case 'roots': return method(() => {
        const r = roots(convert(p.coef, p.domain, p.window))
        this.pendingPlot = { curves: [this.curve(label, p)], marks: r.real ? r.re.map((x) => ({ x, y: 0 })) : [] }
        return r.real ? arr(NDArray.create(r.re, [r.re.length], 'float64')) : { k: 'carray', re: r.re, im: r.im }
      })
      case 'convert': return method(() => {
        const res = this.makePoly(convert(p.coef, p.domain, p.window))
        this.pendingPlot = { curves: [this.curve(label, res)] }
        return res
      })
    }
    throw this.err('AttributeError', `'Polynomial' object has no attribute '${name}'`)
  }

  /** Polynomial.fit(x, y, deg)：在 window 变量上做最小二乘 */
  polyFit(args: Value[], kw: Record<string, Value>): Value {
    const x = toArray(args[0] ?? NONE, 'float64').values()
    const y = toArray(args[1] ?? NONE, 'float64').values()
    const deg = this.toInt(kw.deg ?? args[2])
    if (x.length !== y.length) throw this.err('TypeError', 'expected x and y to have same length')
    if (x.length <= deg) throw this.err('ValueError', `need at least ${deg + 1} points to fit degree ${deg}`)
    const domain: Interval = [Math.min(...x), Math.max(...x)]
    const { off, scl } = mapParams(domain, DEFAULT_IV)
    const res = this.makePoly(lstsq(x.map((t) => off + scl * t), y, deg), domain, DEFAULT_IV)
    this.pendingPlot = { curves: [this.curve('fit', res)], points: { x, y } }
    return res
  }

  getAttr(obj: Value, name: string): Value {
    const fn = (call: Fn): Value => ({ k: 'fn', name, call })
    switch (obj.k) {
      case 'obj': {
        const v = obj.o.getAttr?.(name, this)
        if (v) return v
        throw this.err('AttributeError', `'${obj.o.cls}' object has no attribute '${name}'${obj.o.getAttr ? ' (or it is not available in this sandbox)' : ''}`)
      }
      case 'inst': {
        const own = obj.attrs.get(name)
        if (own) return own
        const c = this.classAttr(obj.cls, name)
        if (c) return c.k === 'func' ? { k: 'bound', self: obj, f: c } : c
        const h = obj.cls.host?.getAttr?.(obj, name, this)
        if (h) return h
        if (name === '__class__') return obj.cls
        throw this.err('AttributeError', `'${obj.cls.name}' object has no attribute '${name}'`)
      }
      case 'class': {
        if (name === '__name__') return str_(obj.name)
        const c = this.classAttr(obj, name)
        if (c) return c
        throw this.err('AttributeError', `type object '${obj.name}' has no attribute '${name}'`)
      }
      case 'super': {
        const c = this.classAttr(obj.self.cls, name, obj.after)
        if (c) return c.k === 'func' ? { k: 'bound', self: obj.self, f: c } : c
        const host = obj.self.cls.host
        const inst = obj.self
        if (name === '__init__') return fn((args, kw) => { host?.init(inst, args, kw, this); return NONE })
        const h = host?.getAttr?.(inst, name, this)
        if (h) return h
        throw this.err('AttributeError', `'super' object has no attribute '${name}'`)
      }
      case 'func':
        if (name === '__name__') return str_(obj.name)
        break
      case 'type':
        if (name === '__name__') return str_(obj.name.split('.').pop()!)
        break
      case 'range':
        if (name === 'start' || name === 'stop' || name === 'step') return int(obj[name])
        break
      case 'int': case 'float': case 'bool':
        if (name === 'item') return fn(() => obj)
        if (name === 'is_integer') return fn(() => bool(Number.isInteger(numOf(obj))))
        break
    }
    if (obj.k === 'module') {
      const v = obj.attrs[name]
      if (!v) throw this.err('AttributeError', `module '${obj.name}' has no attribute '${name}' (not available in this sandbox)`)
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
      }
      const m = this.arrayMethod(name)
      if (m) return { k: 'fn', name, call: m.impl(a), api: m.meta, self: a, rebind: m.impl }
      throw this.err('AttributeError', `'numpy.ndarray' object has no attribute '${name}'`)
    }
    if (obj.k === 'list') {
      const m = this.listMethod(obj, name)
      if (m) return fn(m)
    }
    if (obj.k === 'dict') {
      const m = this.dictMethod(obj, name)
      if (m) return fn(m)
    }
    if (obj.k === 'str') {
      const m = this.strMethod(obj.v, name)
      if (m) return fn(m)
    }
    if (obj.k === 'dtype' && name === 'name') return { k: 'str', v: obj.d }
    if (obj.k === 'poly') return this.polyAttr(obj, name)
    if (obj.k === 'type' && obj.name === 'Polynomial' && name === 'fit') {
      return { k: 'fn', name: 'fit', call: (args, kw) => this.polyFit(args, kw), api: { name: 'Polynomial.fit', kind: 'poly' } }
    }
    throw this.err('AttributeError', `'${typeName(obj)}' object has no attribute '${name}'`)
  }

  listMethod(l: Value & { k: 'list' }, name: string): Fn | null {
    const it = l.items
    const idxOf = (x: Value) => it.findIndex((y) => y === x || this.truthy(this.binop('==', y, x)))
    switch (name) {
      case 'append': return (args) => { it.push(args[0]); return NONE }
      case 'extend': return (args) => { it.push(...this.iterate(args[0])); return NONE }
      case 'insert': return (args) => { const n = it.length; let i = this.toInt(args[0]); if (i < 0) i = Math.max(0, i + n); it.splice(Math.min(i, n), 0, args[1]); return NONE }
      case 'pop': return (args) => {
        if (!it.length) throw this.err('IndexError', 'pop from empty list')
        const n = it.length
        const i = args[0] ? this.toInt(args[0]) : -1
        if (i < -n || i >= n) throw this.err('IndexError', 'pop index out of range')
        return it.splice(i < 0 ? i + n : i, 1)[0]
      }
      case 'index': return (args) => { const i = idxOf(args[0]); if (i < 0) throw this.err('ValueError', `${repr(args[0])} is not in list`); return int(i) }
      case 'count': return (args) => int(it.filter((y) => this.truthy(this.binop('==', y, args[0]))).length)
      case 'remove': return (args) => { const i = idxOf(args[0]); if (i < 0) throw this.err('ValueError', 'list.remove(x): x not in list'); it.splice(i, 1); return NONE }
      case 'reverse': return () => { it.reverse(); return NONE }
      case 'copy': return () => ({ k: 'list', items: [...it] })
      case 'clear': return () => { it.length = 0; return NONE }
      case 'sort': return (_a, kw) => { const sorted = this.sorted(it, kw.key, kw.reverse); it.splice(0, it.length, ...sorted); return NONE }
    }
    return null
  }

  dictMethod(d: Value & { k: 'dict' }, name: string): Fn | null {
    const m = d.d
    switch (name) {
      case 'keys': return () => ({ k: 'list', items: m.keys() })
      case 'values': return () => ({ k: 'list', items: m.values() })
      case 'items': return () => ({ k: 'list', items: m.items().map(([a, b]) => tuple([a, b])) })
      case 'get': return (args) => m.get(args[0]) ?? args[1] ?? NONE
      case 'pop': return (args) => {
        const v = m.get(args[0])
        if (!v) { if (args[1]) return args[1]; throw this.err('KeyError', repr(args[0])) }
        m.delete(args[0])
        return v
      }
      case 'setdefault': return (args) => { if (!m.has(args[0])) m.set(args[0], args[1] ?? NONE); return m.get(args[0])! }
      case 'update': return (args, kw) => {
        if (args[0]?.k === 'dict') for (const [k, v] of args[0].d.items()) m.set(k, v)
        for (const k in kw) m.set(str_(k), kw[k])
        return NONE
      }
      case 'copy': return () => ({ k: 'dict', d: PyDict.from(m.items()) })
    }
    return null
  }

  strMethod(sv: string, name: string): Fn | null {
    const S = (x: string): Value => str_(x)
    const arg = (v: Value | undefined, what: string) => {
      if (!v || v.k !== 'str') throw this.err('TypeError', `${what} must be str, not ${v ? typeName(v) : 'nothing'}`)
      return v.v
    }
    switch (name) {
      case 'upper': return () => S(sv.toUpperCase())
      case 'lower': return () => S(sv.toLowerCase())
      case 'title': return () => S(sv.replace(/\b\w/g, (c) => c.toUpperCase()))
      case 'strip': return (a) => S(a[0]?.k === 'str' ? sv.replace(new RegExp(`^[${escRe(a[0].v)}]+|[${escRe(a[0].v)}]+$`, 'g'), '') : sv.trim())
      case 'lstrip': return () => S(sv.trimStart())
      case 'rstrip': return () => S(sv.trimEnd())
      case 'split': return (a) => ({ k: 'list', items: (a[0] && a[0].k === 'str' ? sv.split(a[0].v) : sv.trim().split(/\s+/).filter(Boolean)).map(S) })
      case 'join': return (a) => S(this.iterate(a[0]).map((x) => arg(x, 'sequence item')).join(sv))
      case 'replace': return (a) => S(sv.split(arg(a[0], 'replace() argument 1')).join(arg(a[1], 'replace() argument 2')))
      case 'startswith': return (a) => bool(sv.startsWith(arg(a[0], 'startswith arg')))
      case 'endswith': return (a) => bool(sv.endsWith(arg(a[0], 'endswith arg')))
      case 'find': return (a) => int(sv.indexOf(arg(a[0], 'find arg')))
      case 'count': return (a) => int(sv.split(arg(a[0], 'count arg')).length - 1)
      case 'isdigit': return () => bool(/^\d+$/.test(sv))
      case 'center': return (a) => S(pad(sv, this.toInt(a[0]), a[1]?.k === 'str' ? a[1].v : ' ', '^'))
      case 'ljust': return (a) => S(pad(sv, this.toInt(a[0]), ' ', '<'))
      case 'rjust': return (a) => S(pad(sv, this.toInt(a[0]), ' ', '>'))
      case 'format': return (args, kw) => {
        let auto = 0
        return S(sv.replace(/\{\{|\}\}|\{([^{}:!]*)(?:!([rs]))?(?::([^{}]*))?\}/g, (whole, key: string | undefined, conv: string | undefined, spec: string | undefined) => {
          if (whole === '{{') return '{'
          if (whole === '}}') return '}'
          const v = key === undefined || key === '' ? args[auto++] : /^\d+$/.test(key) ? args[Number(key)] : kw[key]
          if (!v) throw this.err('IndexError', 'Replacement index out of range for positional args tuple')
          return spec ? formatSpec(v, spec) : conv === 'r' ? repr(v) : this.str(v)
        }))
      }
    }
    return null
  }

  /** sorted() / list.sort(): numbers and strings, optional key function */
  sorted(items: Value[], key?: Value, reverse?: Value): Value[] {
    const keyed = items.map((v) => ({ v, k: key && key.k !== 'none' ? this.call(key, [v]) : v }))
    const cmp = (a: Value, b: Value): number => {
      if (isNum(a) && isNum(b)) return numOf(a) - numOf(b)
      if (a.k === 'str' && b.k === 'str') return a.v < b.v ? -1 : a.v > b.v ? 1 : 0
      if ((a.k === 'tuple' || a.k === 'list') && (b.k === 'tuple' || b.k === 'list')) {
        for (let i = 0; i < Math.min(a.items.length, b.items.length); i++) {
          const c = cmp(a.items[i], b.items[i])
          if (c) return c
        }
        return a.items.length - b.items.length
      }
      if (a.k === 'obj' || b.k === 'obj' || a.k === 'array' || b.k === 'array') return this.truthy(this.binop('<', a, b)) ? -1 : this.truthy(this.binop('<', b, a)) ? 1 : 0
      throw this.err('TypeError', `'<' not supported between instances of '${typeName(a)}' and '${typeName(b)}'`)
    }
    keyed.sort((a, b) => cmp(a.k, b.k))
    if (reverse && this.truthy(reverse)) keyed.reverse()
    return keyed.map((x) => x.v)
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
    const api = (prefix: string) => (name: string, call: Fn, kind?: ApiKind, extra: Partial<ApiMeta> = {}): Value => ({
      k: 'fn', name, call, api: kind ? { name: `${prefix}.${name}`, kind, ...extra } : undefined,
    })
    const fn = api('np')
    const asArr = (v: Value | undefined, name: string): NDArray => {
      if (!v) throw this.err('TypeError', `${name}() missing required argument`)
      return toArray(v)
    }
    const seq = (v: Value | undefined, name: string): NDArray[] => {
      if (!v || (v.k !== 'list' && v.k !== 'tuple')) throw this.err('TypeError', `${name}() expects a sequence of arrays, e.g. ${name}([a, b])`)
      return v.items.map((x) => toArray(x))
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
      }, 'create')
    const like = (name: string, fill: (args: Value[], kw: Record<string, Value>) => Value) =>
      fn(name, (args, kw) => {
        const a = asArr(args[0], name)
        const f = fill(args, kw)
        const dtype = this.dtypeOf(kw.dtype) ?? (name === 'full_like' ? a.dtype : a.dtype)
        return arr(NDArray.create(new Array(a.size).fill(numOf(f)), a.shape, dtype))
      }, 'create')
    const reducer = (kind: ReduceKind) => fn(kind, (args, kw) => this.reduceFn(asArr(args[0], kind), kind, args.slice(1), kw), 'reduce', { axisPos: 1, defaultAxis: null })
    const elementwise = (name: string, f: (x: number) => number, out: 'float' | 'same' | 'int-keep' = 'float') =>
      fn(name, (args) => {
        const a = asArr(args[0], name)
        const dtype: DType = out === 'float' ? 'float64' : a.dtype === 'bool' ? 'int64' : a.dtype
        const res = NDArray.create(a.values().map(f), a.shape, dtype)
        return args[0].k === 'array' || args[0].k === 'list' ? arr(res) : fromScalar(res.values()[0], dtype)
      }, 'elementwise')
    const ufunc2 = (name: string, op: BinOp) => fn(name, (args) => this.binop(op, args[0], args[1]), 'elementwise')
    const logical = (name: string, f: (x: number, y: number) => number) =>
      fn(name, (args) => {
        const a = toArray(args[0])
        const b = toArray(args[1])
        const shape = broadcastShapes([a.shape, b.shape])
        const bv = broadcastTo(b, shape).values()
        return arr(NDArray.create(broadcastTo(a, shape).values().map((x, i) => f(x ? 1 : 0, bv[i] ? 1 : 0)), shape, 'bool'))
      }, 'elementwise')
    const dt = (d: DType, name: string): Value => ({ k: 'type', name: `numpy.${name}`, dtype: d, call: (args) => fromScalar(castValue(numOf(args[0] ?? int(0)), d), d) })
    const shapeArg = (args: Value[]): number[] => (args.length === 1 && (args[0].k === 'tuple' || args[0].k === 'list') ? this.toShape(args) : args.map((x) => this.toInt(x)))

    const rnd = api('np.random')
    const random: Value = {
      k: 'module',
      name: 'numpy.random',
      attrs: {
        seed: rnd('seed', (args) => { this.rng.seed(args[0] ? this.toInt(args[0]) : 0); return NONE }),
        rand: rnd('rand', (args) => {
          const shape = shapeArg(args)
          if (!shape.length) return float(this.rng.next())
          checkSize(shape)
          return arr(NDArray.create(Array.from({ length: shape.reduce((p, x) => p * x, 1) }, () => this.rng.next()), shape, 'float64'))
        }, 'random'),
        randn: rnd('randn', (args) => {
          const shape = shapeArg(args)
          if (!shape.length) return float(this.rng.normal())
          checkSize(shape)
          return arr(NDArray.create(Array.from({ length: shape.reduce((p, x) => p * x, 1) }, () => this.rng.normal()), shape, 'float64'))
        }, 'random'),
        randint: rnd('randint', (args, kw) => {
          let lo = this.toInt(args[0])
          const hi = kw.high ?? args[1]
          let hiN: number
          if (!hi || hi.k === 'none') { hiN = lo; lo = 0 } else hiN = this.toInt(hi)
          if (hiN <= lo) throw this.err('ValueError', 'high <= low')
          const size = kw.size ?? args[2]
          const draw = () => lo + Math.floor(this.rng.next() * (hiN - lo))
          if (!size || size.k === 'none') return int(draw())
          const shape = this.toShape([size])
          checkSize(shape)
          return arr(NDArray.create(Array.from({ length: shape.reduce((p, x) => p * x, 1) }, draw), shape, 'int64'))
        }, 'random'),
      },
    }
    const la = api('np.linalg')
    const linalg: Value = {
      k: 'module',
      name: 'numpy.linalg',
      attrs: {
        inv: la('inv', (args) => arr(inv(asArr(args[0], 'inv'))), 'linalg'),
        det: la('det', (args) => float(det(asArr(args[0], 'det'))), 'linalg'),
        norm: la('norm', (args, kw) => {
          const r = norm(asArr(args[0], 'norm'), this.kwInt(kw, args, 1, 'axis'))
          return typeof r === 'number' ? float(r) : arr(r)
        }, 'reduce', { axisPos: 1, defaultAxis: null }),
      },
    }

    const attrs: Record<string, Value> = {
      // ---- creation
      array: fn('array', (args, kw) => {
        if (!args[0]) throw this.err('TypeError', "array() missing required argument 'object'")
        const src = toArray(args[0], this.dtypeOf(kw.dtype ?? args[1]))
        return arr(src === (args[0] as { a?: NDArray }).a ? src.copy() : src)
      }, 'create'),
      asarray: fn('asarray', (args, kw) => arr(toArray(args[0], this.dtypeOf(kw.dtype))), 'create'),
      copy: fn('copy', (args) => arr(asArr(args[0], 'copy').copy()), 'move'),
      arange: fn('arange', (args, kw) => {
        const nums = args.map((x) => { if (!isNum(x)) throw this.err('TypeError', 'arange() arguments must be numbers'); return x })
        if (!nums.length) throw this.err('TypeError', 'arange() requires stop to be specified.')
        const [start, stop, step] = nums.length === 1 ? [int(0), nums[0], int(1)] : [nums[0], nums[1], nums[2] ?? int(1)]
        const isFloat = [start, stop, step].some((x) => x.k === 'float')
        const s = numOf(step)
        if (s === 0) throw this.err('ZeroDivisionError', 'division by zero')
        const n = Math.max(0, Math.ceil((numOf(stop) - numOf(start)) / s))
        if (n > this.maxSize) throw this.err('MemoryError', `array of ${n} elements exceeds the sandbox limit of ${this.maxSize}`)
        const vals = Array.from({ length: n }, (_, i) => numOf(start) + i * s)
        return arr(NDArray.create(vals, [n], this.dtypeOf(kw.dtype) ?? (isFloat ? 'float64' : 'int64')))
      }, 'create'),
      linspace: fn('linspace', (args, kw) => {
        const a = numOf(args[0])
        const b = numOf(args[1])
        const n = this.kwInt(kw, args, 2, 'num') ?? 50
        if (n > this.maxSize) throw this.err('MemoryError', `array of ${n} elements exceeds the sandbox limit of ${this.maxSize}`)
        return arr(NDArray.create(Array.from({ length: n }, (_, i) => (n === 1 ? a : a + ((b - a) * i) / (n - 1))), [n], 'float64'))
      }, 'create'),
      zeros: filled('zeros', 0),
      ones: filled('ones', 1),
      full: fn('full', (args, kw) => {
        const shape = this.toShape([kw.shape ?? args[0]])
        checkSize(shape)
        const fill = kw.fill_value ?? args[1]
        const dtype = this.dtypeOf(kw.dtype ?? args[2]) ?? scalarDType(fill)
        return arr(NDArray.create(new Array(shape.reduce((p, x) => p * x, 1)).fill(numOf(fill)), shape, dtype))
      }, 'create'),
      eye: fn('eye', (args, kw) => {
        const n = this.toInt(args[0])
        checkSize([n, n])
        return arr(NDArray.create(Array.from({ length: n * n }, (_, i) => (i % (n + 1) === 0 ? 1 : 0)), [n, n], this.dtypeOf(kw.dtype) ?? 'float64'))
      }, 'create'),
      identity: fn('identity', (args, kw) => {
        const n = this.toInt(args[0])
        checkSize([n, n])
        return arr(NDArray.create(Array.from({ length: n * n }, (_, i) => (i % (n + 1) === 0 ? 1 : 0)), [n, n], this.dtypeOf(kw.dtype) ?? 'float64'))
      }, 'create'),
      zeros_like: like('zeros_like', () => int(0)),
      ones_like: like('ones_like', () => int(1)),
      full_like: like('full_like', (args, kw) => kw.fill_value ?? args[1] ?? int(0)),
      // ---- shape & data movement
      reshape: fn('reshape', (args, kw) => arr(reshape(asArr(args[0], 'reshape'), this.toShape([kw.shape ?? kw.newshape ?? args[1]]))), 'move'),
      transpose: fn('transpose', (args) => arr(transpose(asArr(args[0], 'transpose'), args[1] ? this.toShape([args[1]]) : undefined)), 'move'),
      ravel: fn('ravel', (args) => arr(reshape(asArr(args[0], 'ravel'), [-1])), 'move'),
      swapaxes: fn('swapaxes', (args) => arr(swapaxes(asArr(args[0], 'swapaxes'), this.toInt(args[1]), this.toInt(args[2]))), 'move'),
      expand_dims: fn('expand_dims', (args, kw) => {
        const axis = kw.axis ?? args[1]
        if (!axis || axis.k === 'none') throw this.err('TypeError', "expand_dims() missing required argument 'axis'")
        return arr(expandDims(asArr(args[0], 'expand_dims'), this.toInt(axis)))
      }, 'move', { axisPos: 1 }),
      squeeze: fn('squeeze', (args, kw) => arr(squeeze(asArr(args[0], 'squeeze'), this.kwInt(kw, args, 1, 'axis'))), 'move', { axisPos: 1, defaultAxis: null }),
      concatenate: fn('concatenate', (args, kw) => {
        const ax = kw.axis ?? args[1]
        return arr(concatenate(seq(args[0], 'concatenate'), ax && ax.k === 'none' ? null : ax ? this.toInt(ax) : 0))
      }, 'move', { axisPos: 1, defaultAxis: 0 }),
      stack: fn('stack', (args, kw) => arr(stack(seq(args[0], 'stack'), this.kwInt(kw, args, 1, 'axis') ?? 0)), 'move', { axisPos: 1, defaultAxis: 0 }),
      vstack: fn('vstack', (args) => arr(vstack(seq(args[0], 'vstack'))), 'move'),
      hstack: fn('hstack', (args) => arr(hstack(seq(args[0], 'hstack'))), 'move'),
      tile: fn('tile', (args, kw) => {
        const r = kw.reps ?? args[1]
        return arr(tile(asArr(args[0], 'tile'), r.k === 'tuple' || r.k === 'list' ? this.toShape([r]) : [this.toInt(r)]))
      }, 'move'),
      repeat: fn('repeat', (args, kw) => arr(repeat(asArr(args[0], 'repeat'), this.toInt(kw.repeats ?? args[1]), this.kwInt(kw, args, 2, 'axis'))), 'move', { axisPos: 2, defaultAxis: null }),
      flip: fn('flip', (args, kw) => arr(flip(asArr(args[0], 'flip'), this.kwInt(kw, args, 1, 'axis'))), 'move', { axisPos: 1, defaultAxis: null }),
      shares_memory: fn('shares_memory', (args) => bool(sharesMemory(asArr(args[0], 'shares_memory'), asArr(args[1], 'shares_memory')))),
      may_share_memory: fn('may_share_memory', (args) => bool(asArr(args[0], 'may_share_memory').data === asArr(args[1], 'may_share_memory').data)),
      // ---- selection
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
      }, 'elementwise'),
      nonzero: fn('nonzero', (args) => this.nonzero(asArr(args[0], 'nonzero')), 'info'),
      // ---- linear algebra
      dot: fn('dot', (args) => this.binop('@', args[0], args[1]), 'matmul'),
      matmul: fn('matmul', (args) => this.binop('@', args[0], args[1]), 'matmul'),
      outer: fn('outer', (args) => arr(outer(asArr(args[0], 'outer'), asArr(args[1], 'outer'))), 'matmul'),
      trace: fn('trace', (args) => {
        const a = asArr(args[0], 'trace')
        return fromScalar(trace(a), a.dtype === 'float64' ? 'float64' : 'int64')
      }, 'linalg'),
      linalg,
      random,
      polynomial: {
        k: 'module',
        name: 'numpy.polynomial',
        attrs: {
          Polynomial: {
            k: 'type',
            name: 'Polynomial',
            api: { name: 'Polynomial', kind: 'poly' },
            call: (args, kw) => {
              const dom = kw.domain ?? args[1]
              const domain: Interval = dom && dom.k !== 'none' ? (toArray(dom, 'float64').values() as Interval) : DEFAULT_IV
              const res = this.makePoly(this.coefList(args[0], 'Polynomial'), domain, DEFAULT_IV)
              this.pendingPlot = { curves: [this.curve('p', res)] }
              return res
            },
          },
        },
      },
      polyfit: fn('polyfit', (args, kw) => {
        const x = toArray(args[0], 'float64').values()
        const y = toArray(args[1], 'float64').values()
        const deg = this.toInt(kw.deg ?? args[2])
        if (x.length <= deg) throw this.err('ValueError', `need at least ${deg + 1} points to fit degree ${deg}`)
        const low = lstsq(x, y, deg)
        this.pendingPlot = { curves: [{ label: 'polyfit', coef: low }], points: { x, y } }
        return arr(NDArray.create([...low].reverse(), [low.length], 'float64'))
      }, 'poly'),
      polyval: fn('polyval', (args) => {
        const pa = toArray(args[0] ?? NONE)
        const x = args[1] ?? NONE
        const xa = toArray(x)
        const low = [...pa.values()].reverse()
        const ys = xa.values().map((t) => polyval(low, t))
        this.pendingPlot = { curves: [{ label: 'p', coef: low }], marks: xa.values().map((t, i) => ({ x: t, y: ys[i] })) }
        // integer result only when both the coefficients and x are integers (numpy's type promotion)
        const dtype: DType = pa.dtype === 'float64' || xa.dtype === 'float64' ? 'float64' : 'int64'
        return isNum(x) ? fromScalar(ys[0], dtype) : arr(NDArray.create(ys, xa.shape, dtype))
      }, 'poly'),
      // ---- reductions, scans, sorting
      sum: reducer('sum'), mean: reducer('mean'), max: reducer('max'), min: reducer('min'),
      argmax: reducer('argmax'), argmin: reducer('argmin'), any: reducer('any'), all: reducer('all'),
      prod: reducer('prod'), std: reducer('std'), var: reducer('var'), count_nonzero: reducer('count_nonzero'),
      cumsum: fn('cumsum', (args, kw) => arr(cumsum(asArr(args[0], 'cumsum'), this.kwInt(kw, args, 1, 'axis'))), 'scan', { axisPos: 1, defaultAxis: null }),
      sort: fn('sort', (args, kw) => {
        const ax = kw.axis ?? args[1]
        return arr(sortAlong(asArr(args[0], 'sort'), ax && ax.k === 'none' ? null : ax ? this.toInt(ax) : -1).sorted)
      }, 'sort', { axisPos: 1, defaultAxis: -1 }),
      argsort: fn('argsort', (args, kw) => {
        const ax = kw.axis ?? args[1]
        return arr(sortAlong(asArr(args[0], 'argsort'), ax && ax.k === 'none' ? null : ax ? this.toInt(ax) : -1).order)
      }, 'sort', { axisPos: 1, defaultAxis: -1 }),
      unique: fn('unique', (args) => arr(unique(asArr(args[0], 'unique'))), 'unique'),
      // ---- element-wise math
      abs: elementwise('abs', Math.abs, 'same'),
      sqrt: elementwise('sqrt', Math.sqrt),
      exp: elementwise('exp', Math.exp),
      log: elementwise('log', Math.log),
      sin: elementwise('sin', Math.sin),
      cos: elementwise('cos', Math.cos),
      square: elementwise('square', (x) => x * x, 'same'),
      // numpy 2: floor / ceil keep integer dtypes
      floor: elementwise('floor', Math.floor, 'same'),
      ceil: elementwise('ceil', Math.ceil, 'same'),
      sign: elementwise('sign', (x) => (Number.isNaN(x) ? NaN : Math.sign(x) || 0), 'same'),
      round: fn('round', (args, kw) => {
        const a = asArr(args[0], 'round')
        const res = this.roundArr(a, this.kwInt(kw, args, 1, 'decimals') ?? 0)
        return args[0].k === 'array' || args[0].k === 'list' ? arr(res) : fromScalar(res.values()[0], res.dtype)
      }, 'elementwise'),
      around: fn('around', (args, kw) => {
        const a = asArr(args[0], 'around')
        const res = this.roundArr(a, this.kwInt(kw, args, 1, 'decimals') ?? 0)
        return args[0].k === 'array' || args[0].k === 'list' ? arr(res) : fromScalar(res.values()[0], res.dtype)
      }, 'elementwise'),
      add: ufunc2('add', '+'), subtract: ufunc2('subtract', '-'), multiply: ufunc2('multiply', '*'), divide: ufunc2('divide', '/'), power: ufunc2('power', '**'),
      maximum: fn('maximum', (args) => this.extremum(args[0], args[1], Math.max), 'elementwise'),
      minimum: fn('minimum', (args) => this.extremum(args[0], args[1], Math.min), 'elementwise'),
      clip: fn('clip', (args, kw) => this.clip(args[0], kw.a_min ?? kw.min ?? args[1], kw.a_max ?? kw.max ?? args[2]), 'elementwise'),
      logical_and: logical('logical_and', (x, y) => x & y),
      logical_or: logical('logical_or', (x, y) => x | y),
      logical_not: fn('logical_not', (args) => arr(NDArray.create(asArr(args[0], 'logical_not').values().map((x) => (x ? 0 : 1)), asArr(args[0], 'logical_not').shape, 'bool')), 'elementwise'),
      // ---- constants & dtypes
      newaxis: NONE,
      nan: float(NaN),
      inf: float(Infinity),
      pi: float(Math.PI),
      e: float(Math.E),
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
        if (x.k === 'str' && d === 'float64' && /^\s*[-+]?(inf|infinity|nan)\s*$/i.test(x.v)) {
          return float(/nan/i.test(x.v) ? NaN : x.v.includes('-') ? -Infinity : Infinity)
        }
        if (x.k === 'str') {
          const n = Number(x.v)
          if (x.v.trim() === '' || Number.isNaN(n) || (d === 'int64' && !/^\s*[-+]?\d+\s*$/.test(x.v))) {
            throw this.err('ValueError', `invalid literal for ${name}(): ${reprStr(x.v)}`)
          }
          return fromScalar(n, d)
        }
        if (d === 'bool') return bool(this.truthy(x))
        if (x.k === 'array' && x.a.size !== 1) throw this.err('TypeError', 'only length-1 arrays can be converted to Python scalars')
        if (x.k === 'obj') return fromScalar(castValue(this.num(x, 'a number'), d), d)
        const v = x.k === 'array' ? x.a.values()[0] : numOf(x)
        if (Number.isNaN(v) && !isNum(x) && x.k !== 'array') throw this.err('TypeError', `${name}() argument must be a string or a real number, not '${typeName(x)}'`)
        return fromScalar(castValue(v, d), d)
      },
    })
    return {
      print: fn('print', (args, kw) => {
        const sep = kw.sep && kw.sep.k === 'str' ? kw.sep.v : ' '
        const end = kw.end && kw.end.k === 'str' ? kw.end.v : '\n'
        this.stdout.push(args.map((a) => this.str(a)).join(sep) + end)
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
        if (v.k === 'dict') return int(v.d.size)
        if (v.k === 'range') return int(rangeLen(v))
        if (v.k === 'obj' && v.o.len) return int(v.o.len(this))
        throw this.err('TypeError', `object of type '${typeName(v)}' has no len()`)
      }),
      range: { k: 'type', name: 'range', call: (args) => {
        if (!args.length) throw this.err('TypeError', 'range expected at least 1 argument, got 0')
        const n = args.map((x) => this.toInt(x))
        const [a, b, st] = n.length === 1 ? [0, n[0], 1] : [n[0], n[1], n[2] ?? 1]
        if (st === 0) throw this.err('ValueError', 'range() arg 3 must not be zero')
        return { k: 'range', start: a, stop: b, step: st }
      } },
      list: fn('list', (args) => ({ k: 'list', items: args[0] ? [...this.iterate(args[0])] : [] })),
      tuple: fn('tuple', (args) => tuple(args[0] ? [...this.iterate(args[0])] : [])),
      abs: fn('abs', (args) => {
        const x = args[0]
        if (x.k === 'array') return arr(NDArray.create(x.a.values().map(Math.abs), x.a.shape, x.a.dtype === 'bool' ? 'int64' : x.a.dtype))
        return x.k === 'float' ? float(Math.abs(x.v)) : int(Math.abs(numOf(x)))
      }),
      type: fn('type', (args) => (args[0].k === 'inst' ? args[0].cls : { k: 'type', name: typeName(args[0]), call: () => NONE })),
      str: { k: 'type', name: 'str', call: (args) => str_(args.length ? this.str(args[0]) : '') },
      dict: { k: 'type', name: 'dict', call: (args, kw) => {
        const d = new PyDict()
        if (args[0]) for (const p of this.iterate(args[0])) {
          const kv = this.iterate(p)
          if (kv.length !== 2) throw this.err('ValueError', 'dictionary update sequence element has wrong length')
          d.set(kv[0], kv[1])
        }
        for (const k in kw) d.set(str_(k), kw[k])
        return { k: 'dict', d }
      } },
      object: { k: 'type', name: 'object', call: () => { throw this.err('TypeError', 'object() is not useful in this sandbox') } },
      enumerate: fn('enumerate', (args, kw) => {
        const start = kw.start ?? args[1]
        const s0 = start ? this.toInt(start) : 0
        return { k: 'list', items: this.iterate(args[0]).map((v, i) => tuple([int(i + s0), v])) }
      }),
      zip: fn('zip', (args) => {
        const its = args.map((a) => this.iterate(a))
        const n = its.length ? Math.min(...its.map((x) => x.length)) : 0
        return { k: 'list', items: Array.from({ length: n }, (_, i) => tuple(its.map((x) => x[i]))) }
      }),
      sum: fn('sum', (args, kw) => this.iterate(args[0]).reduce((acc, v) => this.binop('+', acc, v), kw.start ?? args[1] ?? int(0))),
      min: fn('min', (args, kw) => this.extreme(args, kw, -1)),
      max: fn('max', (args, kw) => this.extreme(args, kw, 1)),
      sorted: fn('sorted', (args, kw) => ({ k: 'list', items: this.sorted([...this.iterate(args[0])], kw.key, kw.reverse) })),
      reversed: fn('reversed', (args) => ({ k: 'list', items: [...this.iterate(args[0])].reverse() })),
      any: fn('any', (args) => bool(this.iterate(args[0]).some((v) => this.truthy(v)))),
      all: fn('all', (args) => bool(this.iterate(args[0]).every((v) => this.truthy(v)))),
      map: fn('map', (args) => ({ k: 'list', items: this.iterate(args[1]).map((v) => this.call(args[0], [v])) })),
      filter: fn('filter', (args) => ({ k: 'list', items: this.iterate(args[1]).filter((v) => this.truthy(args[0].k === 'none' ? v : this.call(args[0], [v]))) })),
      round: fn('round', (args) => {
        const x = args[0]
        const nd = args[1] && args[1].k !== 'none' ? this.toInt(args[1]) : null
        if (x.k === 'obj' || x.k === 'array') {
          const r = this.getAttr(x, 'round')
          return this.call(r, nd === null ? [] : [int(nd)])
        }
        const v = this.num(x)
        return nd === null ? int(roundHalfEven(v)) : x.k === 'int' ? x : float(roundHalfEven(v, nd))
      }),
      pow: fn('pow', (args) => this.binop('**', args[0], args[1])),
      divmod: fn('divmod', (args) => tuple([this.binop('//', args[0], args[1]), this.binop('%', args[0], args[1])])),
      isinstance: fn('isinstance', (args) => bool(this.isinstance(args[0], args[1]))),
      hasattr: fn('hasattr', (args) => {
        try {
          this.getAttr(args[0], (args[1] as { v: string }).v)
          return bool(true)
        } catch (e) {
          if (e instanceof PyError) return bool(false)
          throw e
        }
      }),
      getattr: fn('getattr', (args) => {
        try {
          return this.getAttr(args[0], (args[1] as { v: string }).v)
        } catch (e) {
          if (e instanceof PyError && args[2]) return args[2]
          throw e
        }
      }),
      setattr: fn('setattr', (args) => { this.setAttr(args[0], (args[1] as { v: string }).v, args[2]); return NONE }),
      super: fn('super', (args) => {
        if (args.length === 2) {
          if (args[0].k !== 'class' || args[1].k !== 'inst') throw this.err('TypeError', 'super(type, obj): obj must be an instance of type')
          return { k: 'super', self: args[1], after: args[0] }
        }
        const f = this.frame
        if (!f?.owner || f.self?.k !== 'inst') throw this.err('RuntimeError', 'super(): no arguments — call it inside a method')
        return { k: 'super', self: f.self, after: f.owner }
      }),
      __math__: this.makeMath(),
      __random__: this.makeRandom(),
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

  extreme(args: Value[], kw: Kw, sign: 1 | -1): Value {
    const items = args.length === 1 ? this.iterate(args[0]) : args
    if (!items.length) {
      if (kw.default) return kw.default
      throw this.err('ValueError', `${sign > 0 ? 'max' : 'min'}() arg is an empty sequence`)
    }
    const key = kw.key && kw.key.k !== 'none' ? kw.key : null
    let best = items[0]
    let bestK = key ? this.call(key, [best]) : best
    for (const v of items.slice(1)) {
      const k = key ? this.call(key, [v]) : v
      if (this.truthy(this.binop(sign > 0 ? '>' : '<', k, bestK))) {
        best = v
        bestK = k
      }
    }
    return best
  }

  isinstance(v: Value, t: Value): boolean {
    if (t.k === 'tuple') return t.items.some((x) => this.isinstance(v, x))
    if (t.k === 'class') {
      if (v.k !== 'inst') return false
      const walk = (c: ClassValue): boolean => c === t || c.bases.some((b) => b.k === 'class' && walk(b))
      return walk(v.cls)
    }
    if (t.k === 'type') {
      const name = t.name
      if (v.k === 'obj') return v.o.cls === name || !!v.o.isa?.(name)
      if (v.k === 'inst') return v.cls.host !== null && t.host === v.cls.host
      const map: Record<string, Value['k'][]> = { int: ['int', 'bool'], float: ['float'], bool: ['bool'], str: ['str'], list: ['list'], tuple: ['tuple'], dict: ['dict'], range: ['range'], 'numpy.ndarray': ['array'] }
      return (map[name] ?? []).includes(v.k)
    }
    throw this.err('TypeError', 'isinstance() arg 2 must be a type, a tuple of types, or a union')
  }

  makeMath(): Value {
    const f1 = (name: string, g: (x: number) => number): Value => ({ k: 'fn', name, call: (args) => {
      const r = g(this.num(args[0]))
      if (Number.isNaN(r)) throw this.err('ValueError', 'math domain error')
      return float(r)
    } })
    return {
      k: 'module', name: 'math', attrs: {
        sqrt: f1('sqrt', Math.sqrt), exp: f1('exp', Math.exp), log: { k: 'fn', name: 'log', call: (args) => {
          const x = this.num(args[0])
          if (x <= 0) throw this.err('ValueError', 'math domain error')
          return float(args[1] ? Math.log(x) / Math.log(this.num(args[1])) : Math.log(x))
        } },
        log2: f1('log2', Math.log2), log10: f1('log10', Math.log10), sin: f1('sin', Math.sin), cos: f1('cos', Math.cos), tan: f1('tan', Math.tan),
        tanh: f1('tanh', Math.tanh), fabs: f1('fabs', Math.abs),
        floor: { k: 'fn', name: 'floor', call: (args) => int(Math.floor(this.num(args[0]))) },
        ceil: { k: 'fn', name: 'ceil', call: (args) => int(Math.ceil(this.num(args[0]))) },
        isclose: { k: 'fn', name: 'isclose', call: (args) => bool(Math.abs(this.num(args[0]) - this.num(args[1])) <= 1e-9 * Math.max(Math.abs(this.num(args[0])), Math.abs(this.num(args[1])))) },
        pi: float(Math.PI), e: float(Math.E), inf: float(Infinity), nan: float(NaN),
      },
    }
  }

  makeRandom(): Value {
    const r = this.rng
    return {
      k: 'module', name: 'random', attrs: {
        seed: { k: 'fn', name: 'seed', call: (args) => { r.seed(args[0] ? this.toInt(args[0]) : 0); return NONE } },
        random: { k: 'fn', name: 'random', call: () => float(r.next()) },
        uniform: { k: 'fn', name: 'uniform', call: (args) => float(this.num(args[0]) + (this.num(args[1]) - this.num(args[0])) * r.next()) },
        randint: { k: 'fn', name: 'randint', call: (args) => int(this.toInt(args[0]) + Math.floor(r.next() * (this.toInt(args[1]) - this.toInt(args[0]) + 1))) },
        choice: { k: 'fn', name: 'choice', call: (args) => { const it = this.iterate(args[0]); return it[Math.floor(r.next() * it.length)] } },
        shuffle: { k: 'fn', name: 'shuffle', call: (args) => {
          if (args[0].k !== 'list') throw this.err('TypeError', 'shuffle() needs a list')
          const it = args[0].items
          for (let i = it.length - 1; i > 0; i--) {
            const j = Math.floor(r.next() * (i + 1))
            ;[it[i], it[j]] = [it[j], it[i]]
          }
          return NONE
        } },
      },
    }
  }

  vars(): VarInfo[] {
    const out: VarInfo[] = []
    for (const [name, v] of this.env) {
      if (v.k === 'module' || v.k === 'func' || v.k === 'class' || v.k === 'fn' || v.k === 'type') continue
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
export const runPython = (src: string, opts: RunOptions = {}): RunResult => {
  const it = new Interp(src, opts)
  let out: string | null = null
  let outDisplay: Display | null = null
  let error: RunResult['error'] = null
  const prevLimit = setSizeLimit(it.maxSize)
  try {
    const v = it.run(parse(src))
    if (v) {
      out = repr(v)
      outDisplay = v.k === 'obj' ? v.o.display?.() ?? null : null
    }
  } catch (e) {
    if (e instanceof PyError) error = { type: e.pyType, message: e.message, line: e.line ?? it.line }
    else if (e instanceof BreakSig || e instanceof ContinueSig) error = { type: 'SyntaxError', message: `'${e instanceof BreakSig ? 'break' : 'continue'}' outside loop`, line: it.line }
    else if (e instanceof RangeError) error = { type: 'RecursionError', message: 'expression too deeply nested', line: it.line }
    else error = { type: 'InternalError', message: e instanceof Error ? e.message : String(e), line: it.line }
  } finally {
    setSizeLimit(prevLimit)
  }
  const displays: Display[] = []
  for (const lib of it.libs) if (it.loaded.has(lib) && lib.finish) {
    try {
      displays.push(...lib.finish(it))
    } catch {
      // a broken figure must not hide the run's output
    }
  }
  return { stdout: it.stdout.join(''), out, outDisplay, error, traces: it.traces, calls: it.calls, events: it.events, displays, vars: it.vars() }
}
