// --- 迷你 Python 解释器：词法 → 语法树 → 求值；所有 ndarray 下标读写都会记录一条 IndexTrace 供可视化 ---
// 支持子集：import（numpy / math / random 与 RunOptions.libs 提供的库）、赋值 / 增量赋值 / 解包、表达式、f-string、
// 推导式、lambda、生成器、异常，以及 if / for / while / def / class / with / try 语句块。标准库在 pyStdlib.ts。

import {
  NDArray, PyError, MAX_SIZE, arrayRepr, arrayStr, binaryOp, castValue, dot, formatScalar, getIndex, reduce,
  reshape, setIndex, sharesMemory, shapeStr, transpose, unaryOp, checkSize, applyOp, cStrides, broadcastShapes, broadcastTo,
  reduceDType, setSizeLimit, normAxis, resetPrintOptions, isFloatD, promoteD,
} from './ndarray'
import type { AxisNote, BinOp, DType, IndexItem, IndexPlan, ReduceKind } from './ndarray'
import type { Display, PyEvent, PyEventInput } from './pyEvents'
import { DEFAULT_IV, complexRepr, convert, deriv, integ, lstsq, mapParams, polyRepr, polyStr, polyadd, polymul, polypow, polysub, polyval, roots, trim } from './poly'
import type { Interval } from './poly'
import { allclose, average, bincount, diff, histogram, isin, median, meshgrid, pad as padArray, padWidths, percentile } from './ndextra'
import { ON_DEMAND } from './pyLazy'
import { STDLIB, erf, gammaFn, gcdOf } from './pyStdlib'
import { ARRAY_METHODS_VIA_FUNCTION, addNumpy, spacedBetween } from './npMore'
import { gamma as gammaDraw } from './npRandom'
import { LegacyRandom } from './mt19937'
import {
  SandboxRandom, concatenate, cumsum, det, expandDims, flip, hstack, inv, norm, outer, repeat, roundHalfEven, sortAlong,
  squeeze, stack, swapaxes, tile, trace, unique, vstack, commonDType,
} from './ndops'

// ================= 词法 =================

type TokType = 'name' | 'num' | 'str' | 'fstr' | 'op' | 'nl' | 'indent' | 'dedent' | 'eof'
interface Tok { t: TokType; v: string; pos: number; end: number; line: number; imag?: boolean; body?: number }

const OPS = [
  '**=', '//=', '<<=', '>>=', '...', '->', '**', '//', '==', '!=', '<=', '>=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '@=', '<<', '>>', ':=',
  '+', '-', '*', '/', '%', '@', '&', '|', '^', '~', '<', '>', '=', '(', ')', '[', ']', '{', '}', ',', ':', '.', ';',
]

const syntaxError = (msg: string, line: number) => new PyError('SyntaxError', msg, line)

const ESCAPES: Record<string, string> = { n: '\n', t: '\t', r: '\r', a: '\x07', b: '\b', f: '\f', v: '\v', '\\': '\\', "'": "'", '"': '"' }
const NUMERIC_ESCAPE = /^(?:x([0-9a-fA-F]{2})|u([0-9a-fA-F]{4})|U([0-9a-fA-F]{8})|([0-7]{1,3}))/

/** the escape after a backslash: what it means, and how many characters it took */
const escapeAt = (text: string, at: number): [string, number] => {
  const m = NUMERIC_ESCAPE.exec(text.slice(at, at + 9))
  if (m) return [String.fromCodePoint(m[4] ? parseInt(m[4], 8) : parseInt(m[1] ?? m[2] ?? m[3], 16)), m[0].length]
  return [ESCAPES[text[at]] ?? '\\' + text[at], 1]
}

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
      const radix = /^0([xX][\da-fA-F_]+|[bB][01_]+|[oO][0-7_]+)/.exec(src.slice(i))
      const m = radix ?? /^(\d[\d_]*\.?[\d_]*|\.\d[\d_]*)([eE][+-]?\d+)?/.exec(src.slice(i))!
      const text = m[0].replace(/_/g, '')
      push('num', radix ? String(Number(text)) : text, i, i + m[0].length)
      i += m[0].length
      // complex numbers are not modelled, but np.mgrid[0:1:5j] spells "5 points" with one
      if (src[i] === 'j' || src[i] === 'J') {
        toks[toks.length - 1].imag = true
        i++
      }
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
          if (e === '\n') {
            line++
            j += 2
          } else {
            const [text, took] = escapeAt(src, j + 1)
            s += text
            j += 1 + took
          }
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

/** kind: undefined = positional or keyword; 'star' = *args; 'kwstar' = **kwargs; 'kwonly' = after a bare * or *args */
export interface Param { name: string; def: Node | null; kind?: 'star' | 'kwstar' | 'kwonly'; note?: Node; posOnly?: boolean }
export interface CompFor { target: Node; iter: Node; conds: Node[] }
export type FPart = string | { x: Node; conv: 'r' | 's' | 'a' | null; spec: string; specX?: Node }

export type Node = Span & (
  | { k: 'num'; v: number; isFloat: boolean; imag?: boolean }
  | { k: 'str'; v: string }
  | { k: 'fstr'; parts: FPart[] }
  | { k: 'const'; v: 'True' | 'False' | 'None' }
  | { k: 'name'; id: string }
  | { k: 'list'; items: Node[] }
  | { k: 'tuple'; items: Node[] }
  | { k: 'set'; items: Node[] }
  /** a null key is a **mapping merged in */
  | { k: 'dict'; keys: (Node | null)[]; vals: Node[] }
  | { k: 'comp'; kind: 'list' | 'gen' | 'dict' | 'set'; elt: Node; val: Node | null; gens: CompFor[] }
  | { k: 'star'; x: Node }
  | { k: 'walrus'; id: string; x: Node }
  | { k: 'lambda'; params: Param[]; body: Node }
  | { k: 'ellipsis' }
  | { k: 'unary'; op: '-' | '+' | '~'; x: Node }
  | { k: 'not'; x: Node }
  | { k: 'bin'; op: string; l: Node; r: Node }
  | { k: 'boolop'; op: 'and' | 'or'; l: Node; r: Node }
  | { k: 'compare'; ops: string[]; xs: Node[] }
  | { k: 'ifexp'; cond: Node; a: Node; b: Node }
  /** a null name is **mapping */
  | { k: 'call'; fn: Node; args: Node[]; kw: { name: string | null; v: Node }[] }
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
  | { k: 'for'; target: Node; iter: Node; body: Stmt[]; orelse: Stmt[] | null; line: number }
  | { k: 'while'; cond: Node; body: Stmt[]; orelse: Stmt[] | null; line: number }
  | { k: 'def'; name: string; params: Param[]; returns: Node | null; body: Stmt[]; isGen: boolean; decorators: Node[]; line: number }
  | { k: 'return'; x: Node | null; line: number }
  | { k: 'class'; name: string; bases: Node[]; body: Stmt[]; decorators: Node[]; line: number }
  | { k: 'with'; items: { x: Node; as: Node | null }[]; body: Stmt[]; line: number }
  | { k: 'break' | 'continue'; line: number }
  | { k: 'del'; targets: Node[]; line: number }
  | { k: 'assert'; x: Node; msg: Node | null; line: number }
  | { k: 'try'; body: Stmt[]; handlers: { types: Node | null; name: string | null; body: Stmt[]; line: number }[]; orelse: Stmt[] | null; final: Stmt[] | null; line: number }
  | { k: 'raise'; x: Node | null; cause: Node | null; line: number }
  | { k: 'global' | 'nonlocal'; names: string[]; line: number }
  /** yield x, yield from x, or a = yield x */
  | { k: 'yield'; x: Node | null; from: boolean; targets: Node[]; line: number }
  /** x: int = 3 (a class body keeps the names, for dataclasses) */
  | { k: 'annot'; name: string; note: Node | null; x: Node | null; line: number }
  /** several statements written as one: import a, b */
  | { k: 'seq'; body: Stmt[]; line: number }

const UNSUPPORTED = new Set(['async', 'await'])
/** keywords that can never start an expression */
const KEYWORDS = new Set([...UNSUPPORTED, 'try', 'except', 'finally', 'raise', 'global', 'nonlocal', 'yield', 'for', 'while', 'if', 'elif', 'else', 'def', 'class', 'with', 'return', 'del', 'pass', 'break', 'continue', 'import', 'from', 'as', 'assert', 'in', 'is', 'and', 'or'])
const COMPARE_OPS = new Set(['<', '>', '==', '!=', '<=', '>='])

class Parser {
  toks: Tok[]
  i = 0
  /** whether the function being parsed has a yield, which makes it a generator */
  sawYield = false
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
    if (this.isOp('@')) {
      const decorators: Node[] = []
      while (this.isOp('@')) {
        this.next()
        decorators.push(this.test())
        if (this.cur.t !== 'nl') throw this.unexpected()
        this.next()
      }
      if (!this.isName('def') && !this.isName('class')) throw this.unexpected()
      const [st] = this.statement()
      ;(st as { decorators: Node[] }).decorators = decorators
      return [st]
    }
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
        return [{ k: 'for', target, iter, body, orelse: this.orElse(), line: tk.line }]
      }
      case 'while': {
        this.next()
        const cond = this.namedTest()
        const body = this.block()
        return [{ k: 'while', cond, body, orelse: this.orElse(), line: tk.line }]
      }
      case 'def': {
        this.next()
        const name = this.expectName()
        this.expectOp('(')
        const params = this.params(')')
        this.expectOp(')')
        let returns: Node | null = null
        if (this.isOp('->')) { this.next(); returns = this.test() }
        const outer = this.sawYield
        this.sawYield = false
        const body = this.block()
        const isGen = this.sawYield
        this.sawYield = outer
        return [{ k: 'def', name, params, returns, body, isGen, decorators: [], line: tk.line }]
      }
      case 'class': {
        this.next()
        const name = this.expectName()
        const bases: Node[] = []
        if (this.isOp('(')) {
          this.next()
          while (!this.isOp(')')) {
            // class A(B, metaclass=M): the keywords do not change anything here
            if (this.cur.t === 'name' && this.peek().t === 'op' && this.peek().v === '=') {
              this.i += 2
              this.test()
            } else bases.push(this.test())
            if (!this.isOp(')')) this.expectOp(',')
          }
          this.next()
        }
        const outer = this.sawYield
        const body = this.block()
        this.sawYield = outer
        return [{ k: 'class', name, bases, body, decorators: [], line: tk.line }]
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
      case 'try': {
        this.next()
        const body = this.block()
        const handlers: { types: Node | null; name: string | null; body: Stmt[]; line: number }[] = []
        while (this.isName('except')) {
          const line = this.next().line
          let types: Node | null = null
          let name: string | null = null
          if (!this.isOp(':')) {
            types = this.test()
            if (this.isName('as')) {
              this.next()
              name = this.expectName()
            }
          }
          handlers.push({ types, name, body: this.block(), line })
        }
        const orelse = this.orElse()
        let final: Stmt[] | null = null
        if (this.isName('finally')) {
          this.next()
          final = this.block()
        }
        if (!handlers.length && !final) throw syntaxError("expected 'except' or 'finally' block", this.cur.line)
        return [{ k: 'try', body, handlers, orelse, final, line: tk.line }]
      }
    }
    if (tk.v === 'elif' || tk.v === 'else' || tk.v === 'except' || tk.v === 'finally') throw syntaxError('invalid syntax', tk.line)
    return this.simpleLine()
  }

  orElse(): Stmt[] | null {
    if (!this.isName('else')) return null
    this.next()
    return this.block()
  }

  params(close: string): Param[] {
    const params: Param[] = []
    let afterStar = false
    const annotation = (): { note?: Node } => {
      if (!this.isOp(':') || close !== ')') return {}
      this.next()
      return { note: this.test() }
    }
    while (!this.isOp(close)) {
      if (this.isOp('/')) {
        // def f(a, /, b): positional-only markers change nothing a learner would notice, but inspect.signature prints them
        this.next()
        for (const p of params) p.posOnly = true
      } else if (this.isOp('**')) {
        this.next()
        const name = this.expectName()
        params.push({ name, def: null, kind: 'kwstar', ...annotation() })
      } else if (this.isOp('*')) {
        this.next()
        afterStar = true
        if (!this.isOp(',')) {
          const name = this.expectName()
          params.push({ name, def: null, kind: 'star', ...annotation() })
        }
      } else {
        const name = this.expectName()
        const noted = annotation()
        let def: Node | null = null
        if (this.isOp('=')) { this.next(); def = this.test() }
        else if (!afterStar && params.some((p) => p.def)) throw syntaxError('non-default argument follows default argument', this.cur.line)
        params.push(afterStar ? { name, def, kind: 'kwonly', ...noted } : { name, def, ...noted })
      }
      if (!this.isOp(close)) this.expectOp(',')
    }
    return params
  }

  /** a test that may be an assignment expression: (n := 10) */
  namedTest(): Node {
    const start = this.cur
    if (start.t === 'name' && !KEYWORDS.has(start.v) && this.peek().t === 'op' && this.peek().v === ':=') {
      this.i += 2
      return this.mk(start, { k: 'walrus', id: start.v, x: this.test() })
    }
    return this.test()
  }

  /** an element of a display or call: x, or *x */
  element(): Node {
    const start = this.cur
    if (this.isOp('*')) {
      this.next()
      return this.mk(start, { k: 'star', x: this.bitOr() })
    }
    return this.namedTest()
  }

  yieldExpr(targets: Node[], line: number): Stmt {
    this.next()
    this.sawYield = true
    if (this.isName('from')) {
      this.next()
      return { k: 'yield', x: this.testList(), from: true, targets, line }
    }
    return { k: 'yield', x: this.atLineEnd() || this.isOp(';') ? null : this.testList(), from: false, targets, line }
  }

  simple(): Stmt {
    const tk = this.cur
    if (tk.t === 'name' && UNSUPPORTED.has(tk.v)) {
      throw new PyError('NotImplementedError', `'${tk.v}' is not supported in this sandbox`, tk.line)
    }
    if (this.isName('yield')) return this.yieldExpr([], tk.line)
    if (this.isName('raise')) {
      this.next()
      if (this.atLineEnd() || this.isOp(';')) return { k: 'raise', x: null, cause: null, line: tk.line }
      const x = this.test()
      let cause: Node | null = null
      if (this.isName('from')) {
        this.next()
        cause = this.test()
      }
      return { k: 'raise', x, cause, line: tk.line }
    }
    if (this.isName('global') || this.isName('nonlocal')) {
      const k = this.next().v as 'global' | 'nonlocal'
      const names = [this.expectName()]
      while (this.isOp(',')) {
        this.next()
        names.push(this.expectName())
      }
      return { k, names, line: tk.line }
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
      const body: Stmt[] = []
      do {
        this.next()
        const mod = this.dotted()
        let alias: string | null = null
        if (this.isName('as')) { this.next(); alias = this.expectName() }
        body.push({ k: 'import', mod, alias, line: tk.line })
      } while (this.isOp(','))
      return body.length === 1 ? body[0] : { k: 'seq', body, line: tk.line }
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
        if (this.isOp('*')) {
          this.next()
          names.push({ name: '*', alias: '*' })
          break
        }
        const name = this.expectName()
        let alias = name
        if (this.isName('as')) { this.next(); alias = this.expectName() }
        names.push({ name, alias })
      } while (this.isOp(','))
      if (paren) this.expectOp(')')
      return { k: 'from', mod, names, line: tk.line }
    }
    const first = this.testList()
    if (this.cur.t === 'op' && /^(\*\*|\/\/|<<|>>|[-+*/%&|^@])=$/.test(this.cur.v)) {
      const op = this.next().v.slice(0, -1)
      if (first.k !== 'name' && first.k !== 'sub' && first.k !== 'attr') throw syntaxError("'illegal expression for augmented assignment'", tk.line)
      const x = this.testList()
      return { k: 'aug', op, target: first, x, line: tk.line, s: tk.pos, e: this.endOf() }
    }
    if (this.isOp(':') && (first.k === 'name' || first.k === 'attr')) {
      // annotated assignment: x: int = 3
      this.next()
      const note = this.test()
      const name = first.k === 'name' ? first.id : null
      if (!this.isOp('=')) return name ? { k: 'annot', name, note, x: null, line: tk.line } : { k: 'pass', line: tk.line }
      if (name) {
        this.next()
        return { k: 'annot', name, note, x: this.testList(), line: tk.line }
      }
    }
    if (this.isOp('=')) {
      const targets = [first]
      let x: Node = first
      while (this.isOp('=')) {
        this.next()
        if (this.isName('yield')) {
          for (const t of targets) this.checkTarget(t)
          return this.yieldExpr(targets, tk.line)
        }
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

  checkTarget(t: Node): void {
    if (t.k === 'name' || t.k === 'sub' || t.k === 'attr') return
    if (t.k === 'star') return this.checkTarget(t.x)
    if ((t.k === 'tuple' || t.k === 'list') && t.items.length) {
      t.items.forEach((x) => this.checkTarget(x))
      return
    }
    throw syntaxError('cannot assign to expression', t.line)
  }

  /** for-loop / comprehension target: a, (b, c) — stops before 'in' */
  targetList(): Node {
    const start = this.cur
    const one = () => {
      const s = this.cur
      if (!this.isOp('*')) return this.bitOr()
      this.next()
      return this.mk(s, { k: 'star', x: this.bitOr() })
    }
    const first = one()
    if (!this.isOp(',')) {
      this.checkTarget(first)
      return first
    }
    const items = [first]
    while (this.isOp(',')) {
      this.next()
      if (this.isName('in')) break
      items.push(one())
    }
    const t = this.mk(start, { k: 'tuple', items })
    this.checkTarget(t)
    return t
  }

  testList(): Node {
    const start = this.cur
    const first = this.element()
    if (!this.isOp(',')) return first
    const items = [first]
    while (this.isOp(',')) {
      this.next()
      if (this.atLineEnd() || this.isOp('=') || this.isOp(';') || this.isOp(':') || this.isOp(')')) break
      items.push(this.element())
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
  bitAnd(): Node { return this.binLevel(['&'], () => this.shift()) }
  shift(): Node { return this.binLevel(['<<', '>>'], () => this.arith()) }
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
        const kw: { name: string | null; v: Node }[] = []
        while (!this.isOp(')')) {
          if (this.isOp('**')) {
            this.next()
            kw.push({ name: null, v: this.test() })
          } else if (this.isOp('*')) {
            args.push(this.element())
          } else if (this.cur.t === 'name' && this.peek().t === 'op' && this.peek().v === '=') {
            const name = this.next().v
            this.next()
            kw.push({ name, v: this.test() })
          } else {
            if (kw.length) throw syntaxError('positional argument follows keyword argument', this.cur.line)
            const argStart = this.cur
            const a = this.namedTest()
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
      if (lit) {
        let text = ''
        for (let at = 0; at < lit.length;) {
          if (lit[at] === '\\' && at + 1 < lit.length) {
            const [one, took] = escapeAt(lit, at + 1)
            text += one
            at += 1 + took
          } else text += lit[at++]
        }
        parts.push(text)
      }
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
            // conversion and format spec run to the closing brace; the spec may hold {nested} fields
            let nested = 0
            let k = j
            for (; k < raw.length; k++) {
              if (raw[k] === '{') nested++
              else if (raw[k] === '}') {
                if (!nested) break
                nested--
              }
            }
            j = k
            break
          }
        }
      }
      if (j >= raw.length) throw syntaxError("f-string: expecting '}'", tk.line)
      let exprSrc = raw.slice(i + 1, exprEnd)
      if (!exprSrc.trim()) throw syntaxError('f-string: empty expression not allowed', tk.line)
      const tail = raw.slice(exprEnd, j)
      const m = /^(?:!([rsa]))?(?::(.*))?$/s.exec(tail)
      if (!m) throw syntaxError('f-string: invalid conversion character', tk.line)
      // f'{x=}' prints the source text, then the value
      const debug = /[^=!<>]=\s*$/.test(exprSrc)
      if (debug) {
        parts.push(exprSrc)
        exprSrc = exprSrc.replace(/=\s*$/, '')
      }
      const sub = new Parser(tokenize(exprSrc, bodyStart + i + 1, tk.line))
      const x = sub.testList()
      if (sub.cur.t !== 'nl' && sub.cur.t !== 'eof') throw syntaxError('f-string: invalid syntax', tk.line)
      const spec = m[2] ?? ''
      const specX = spec.includes('{') ? this.fstring({ ...tk, v: spec, body: bodyStart + exprEnd + 1 }) : undefined
      parts.push({ x, conv: (m[1] as 'r' | 's' | 'a' | undefined) ?? (debug && !spec ? 'r' : null), spec, specX })
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
      return this.mk(tk, { k: 'num', v: Number(tk.v), isFloat, imag: tk.imag })
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
      if (tk.v === 'yield') throw new PyError('NotImplementedError', "this sandbox supports yield as a statement or as 'x = yield y', not inside a larger expression", tk.line)
      if (KEYWORDS.has(tk.v)) throw this.unexpected()
      this.next()
      if (tk.v === 'True' || tk.v === 'False' || tk.v === 'None') return this.mk(tk, { k: 'const', v: tk.v })
      return this.mk(tk, { k: 'name', id: tk.v })
    }
    if (this.isOp('...')) { this.next(); return this.mk(tk, { k: 'ellipsis' }) }
    if (this.isOp('(')) {
      this.next()
      if (this.isOp(')')) { this.next(); return this.mk(tk, { k: 'tuple', items: [] }) }
      const first = this.element()
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
        items.push(this.element())
      }
      this.expectOp(')')
      return this.mk(tk, { k: 'tuple', items })
    }
    if (this.isOp('[')) {
      this.next()
      const items: Node[] = []
      while (!this.isOp(']')) {
        items.push(this.element())
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
      const keys: (Node | null)[] = []
      const vals: Node[] = []
      if (!this.isOp('}') && !this.isOp('**')) {
        const first = this.element()
        if (!this.isOp(':')) {
          // a set display or comprehension
          if (this.isName('for')) {
            const gens = this.compFors()
            this.expectOp('}')
            return this.mk(tk, { k: 'comp', kind: 'set', elt: first, val: null, gens })
          }
          const items = [first]
          while (this.isOp(',')) {
            this.next()
            if (this.isOp('}')) break
            items.push(this.element())
          }
          this.expectOp('}')
          return this.mk(tk, { k: 'set', items })
        }
        this.next()
        const val = this.test()
        if (this.isName('for')) {
          const gens = this.compFors()
          this.expectOp('}')
          return this.mk(tk, { k: 'comp', kind: 'dict', elt: first, val, gens })
        }
        keys.push(first)
        vals.push(val)
        if (!this.isOp('}')) this.expectOp(',')
      }
      while (!this.isOp('}')) {
        if (this.isOp('**')) {
          this.next()
          keys.push(null)
          vals.push(this.bitOr())
        } else {
          keys.push(this.test())
          this.expectOp(':')
          vals.push(this.test())
        }
        if (!this.isOp('}')) this.expectOp(',')
      }
      this.next()
      return this.mk(tk, { k: 'dict', keys, vals })
    }
    throw this.unexpected()
  }
}

export const parse = (src: string): Stmt[] => new Parser(tokenize(src)).program()

/** the text given to eval() or exec(), every line of it counted as the line that called it */
const parseText = (src: string, mode: 'eval' | 'exec', line: number): Node | Stmt[] => {
  const read = (toks: Tok[]): Node | Stmt[] => {
    const p = new Parser(toks)
    if (mode === 'exec') return p.program()
    const x = p.testList()
    while (p.cur.t === 'nl') p.next()
    if (p.cur.t !== 'eof') throw syntaxError('invalid syntax', p.cur.line)
    return x
  }
  const text = mode === 'eval' ? src.trimStart() : src
  try {
    read(tokenize(text))
  } catch (e) {
    // CPython's wording for most of these; its finer messages ('(' was never closed …) are not reproduced
    if (e instanceof PyError && e.pyType === 'SyntaxError') throw new PyError('SyntaxError', `invalid syntax (<string>, line ${e.line ?? 1})`, line)
    throw e
  }
  return read(tokenize(text).map((t) => ({ ...t, line })))
}

// ================= 运行时值 =================

export type Fn = (args: Value[], kw: Record<string, Value>) => Value

/** 可视化用的 API 分类：决定「输出元素来自哪些输入元素」的计算方式 */
export type ApiKind = 'create' | 'move' | 'elementwise' | 'reduce' | 'scan' | 'matmul' | 'sort' | 'unique' | 'linalg' | 'random' | 'info' | 'poly' | 'group'

export interface ApiMeta {
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
  abstract?: boolean
  k: 'func'
  name: string
  params: Param[]
  defaults: (Value | null)[]
  body: Stmt[] | null
  expr: Node | null
  closure: Frame | null
  /** the class whose body defined it (for zero-argument super()) */
  owner: ClassValue | null
  isGen?: boolean
  doc?: string
  /** attributes set on the function: f.calls = 0 */
  attrs?: Map<string, Value>
  /** the annotations, 'return' last, as f.__annotations__ has them */
  notes?: Map<string, Value>
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

/**
 * What a library metaclass (enum's) adds to the classes it makes: how the class itself behaves when called, indexed
 * or iterated. A subclass inherits it. Every hook is optional; a missing one gives the class its usual behaviour.
 */
export interface ClassMeta {
  /** __prepare__: the namespace the class body runs in */
  prepare?(name: string, bases: Value[], h: Host): Map<string, Value>
  /** the end of __new__: the class body has run */
  created?(cls: ClassValue, h: Host): void
  call?(cls: ClassValue, args: Value[], kw: Kw, h: Host): Value
  getItem?(cls: ClassValue, idx: Value, h: Host): Value
  iter?(cls: ClassValue, h: Host): Value[]
  len?(cls: ClassValue, h: Host): number
  contains?(cls: ClassValue, v: Value, h: Host): boolean
  /** after the class's own attributes; may raise the class's own AttributeError */
  getAttr?(cls: ClassValue, name: string, h: Host): Value | undefined
  /** true = handled */
  setAttr?(cls: ClassValue, name: string, v: Value, h: Host): boolean
  repr?(cls: ClassValue): string
  /** the built-in value an instance also is (class Level(int, Enum)): it finds the same dict entry */
  scalar?(inst: InstValue): Value | undefined
}

/** builtin: part of Python (ValueError …), so it prints without __main__; module: the library module it prints with */
export interface ClassValue { k: 'class'; generic?: boolean; abstractBase?: boolean; abstracts?: string[]; name: string; bases: Value[]; ns: Map<string, Value>; host: HostClass | null; builtin?: boolean; module?: string; meta?: ClassMeta }
export interface InstValue { k: 'inst'; cls: ClassValue; attrs: Map<string, Value>; state: Record<string, unknown> }

export type Value =
  | { k: 'int'; v: number }
  /** f32: a NumPy float32 scalar, which prints its own shortest digits */
  | { k: 'float'; v: number; f32?: boolean }
  | { k: 'bool'; v: boolean }
  | { k: 'none' }
  | { k: 'str'; v: string }
  /** view: d.keys() and friends, which print as dict_keys([...]) */
  | { k: 'list'; items: Value[]; view?: string }
  /** names, tname: a namedtuple */
  | { k: 'tuple'; items: Value[]; names?: string[]; tname?: string }
  /** flavor and factory: collections' Counter, defaultdict and OrderedDict */
  | { k: 'dict'; d: PyDict; flavor?: 'Counter' | 'defaultdict' | 'OrderedDict'; factory?: Value }
  | { k: 'set'; s: PyDict; frozen?: boolean }
  /** an iterator: next() gives undefined when it is exhausted */
  | { k: 'iter'; shown?: () => string; name: string; next: () => Value | undefined; send?: (v: Value) => Value | undefined; throw?: (e: unknown) => Value | undefined; close?: () => void; result?: Value }
  /** @staticmethod, @classmethod, @property */
  | { k: 'descr'; kind: 'static' | 'class' | 'prop'; f: Value; setter?: Value }
  | { k: 'range'; start: number; stop: number; step: number }
  | { k: 'slice'; start: Value; stop: Value; step: Value }
  | { k: 'ellipsis' }
  | { k: 'array'; a: NDArray }
  /** bindable: acts as a method when stored in a class (functools.lru_cache …) */
  | { k: 'fn'; shown?: string; name: string; call: Fn; api?: ApiMeta; self?: NDArray; rebind?: (self: NDArray) => Fn; bindable?: boolean; attrs?: Map<string, Value> }
  | { k: 'type'; name: string; call: Fn; dtype?: DType; api?: ApiMeta; host?: HostClass; attrs?: Map<string, Value>; parents?: string[]; methods?: Map<string, Value> }
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

export interface Frame { vars: Map<string, Value>; parent: Frame | null; isClass?: boolean; owner?: ClassValue | null; self?: Value; globals?: Set<string>; nonlocals?: Set<string> }

/** the common type of two arrays; a Python number beside a uint8 or float32 array keeps its type */
const pairDType = (x: NDArray, y: NDArray): DType => {
  const pythonInt = (t: NDArray) => t.ndim === 0 && t.dtype === 'int64'
  const pythonNumber = (t: NDArray) => t.ndim === 0 && (t.dtype === 'int64' || t.dtype === 'float64')
  if ((x.dtype === 'float32' && pythonNumber(y)) || (y.dtype === 'float32' && pythonNumber(x))) return 'float32'
  return (x.dtype === 'uint8' && pythonInt(y)) || (y.dtype === 'uint8' && pythonInt(x)) ? 'uint8' : commonDType([x, y])
}

/** round() on a float looks at the exact binary value: round(2.675, 2) is 2.67, round(0.125, 2) is 0.12 */
const roundExact = (x: number, nd: number): number => {
  if (!Number.isFinite(x) || Math.abs(x) >= 1e21 || nd > 20) return x
  if (nd < 0) return roundHalfEven(x / 10 ** -nd) * 10 ** -nd
  const [whole, frac] = Math.abs(x).toFixed(Math.min(100, nd + 60)).split('.')
  const rest = frac.slice(nd)
  let n = BigInt(whole + frac.slice(0, nd))
  const tie = rest[0] === '5' && !/[1-9]/.test(rest.slice(1))
  if (tie ? n % 2n === 1n : rest[0] >= '5') n += 1n
  const r = Number(n) / 10 ** nd
  return x < 0 ? -r : r
}

/** the running program's __hash__, so instances that compare equal find the same dict entry */
let instHash: ((v: InstValue) => string | undefined) | null = null

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
      case 'set':
        if (!v.frozen) throw new PyError('TypeError', "unhashable type: 'set'")
        return `f(${v.s.keys().map(PyDict.hash).sort().join(',')})`
      case 'inst': return instHash?.(v) ?? `i${objId(v)}`
      case 'obj': return v.o.hashKey?.() ?? `o${objId(v.o)}`
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

/** obj.__dict__, vars(obj): the object's own attributes, so writing to it writes to them */
class AttrDict extends PyDict {
  attrs: Map<string, Value>
  constructor(attrs: Map<string, Value>) {
    super()
    this.attrs = attrs
  }
  name(k: Value): string {
    if (k.k !== 'str') throw new PyError('TypeError', 'attribute name must be string')
    return k.v
  }
  get(k: Value) { return k.k === 'str' ? this.attrs.get(k.v) : undefined }
  set(k: Value, v: Value) { this.attrs.set(this.name(k), v) }
  has(k: Value) { return k.k === 'str' && this.attrs.has(k.v) }
  delete(k: Value) { return k.k === 'str' && this.attrs.delete(k.v) }
  keys(): Value[] { return [...this.attrs.keys()].map((n) => ({ k: 'str', v: n })) }
  values(): Value[] { return [...this.attrs.values()] }
  items(): [Value, Value][] { return [...this.attrs].map(([n, v]) => [{ k: 'str', v: n }, v]) }
  get size() { return this.attrs.size }
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
/** dir() of the built-in values, as CPython lists them */
const BUILTIN_DIR: Record<string, string> = {
  list: '__add__ __class__ __class_getitem__ __contains__ __delattr__ __delitem__ __dir__ __doc__ __eq__ __format__ __ge__ __getattribute__ __getitem__ __gt__ __hash__ __iadd__ __imul__ __init__ __init_subclass__ __iter__ __le__ __len__ __lt__ __mul__ __ne__ __new__ __reduce__ __reduce_ex__ __repr__ __reversed__ __rmul__ __setattr__ __setitem__ __sizeof__ __str__ __subclasshook__ append clear copy count extend index insert pop remove reverse sort',
  str: '__add__ __class__ __contains__ __delattr__ __dir__ __doc__ __eq__ __format__ __ge__ __getattribute__ __getitem__ __getnewargs__ __gt__ __hash__ __init__ __init_subclass__ __iter__ __le__ __len__ __lt__ __mod__ __mul__ __ne__ __new__ __reduce__ __reduce_ex__ __repr__ __rmod__ __rmul__ __setattr__ __sizeof__ __str__ __subclasshook__ capitalize casefold center count encode endswith expandtabs find format format_map index isalnum isalpha isascii isdecimal isdigit isidentifier islower isnumeric isprintable isspace istitle isupper join ljust lower lstrip maketrans partition removeprefix removesuffix replace rfind rindex rjust rpartition rsplit rstrip split splitlines startswith strip swapcase title translate upper zfill',
  dict: '__class__ __class_getitem__ __contains__ __delattr__ __delitem__ __dir__ __doc__ __eq__ __format__ __ge__ __getattribute__ __getitem__ __gt__ __hash__ __init__ __init_subclass__ __ior__ __iter__ __le__ __len__ __lt__ __ne__ __new__ __or__ __reduce__ __reduce_ex__ __repr__ __reversed__ __ror__ __setattr__ __setitem__ __sizeof__ __str__ __subclasshook__ clear copy fromkeys get items keys pop popitem setdefault update values',
  set: '__and__ __class__ __class_getitem__ __contains__ __delattr__ __dir__ __doc__ __eq__ __format__ __ge__ __getattribute__ __gt__ __hash__ __iand__ __init__ __init_subclass__ __ior__ __isub__ __iter__ __ixor__ __le__ __len__ __lt__ __ne__ __new__ __or__ __rand__ __reduce__ __reduce_ex__ __repr__ __ror__ __rsub__ __rxor__ __setattr__ __sizeof__ __str__ __sub__ __subclasshook__ __xor__ add clear copy difference difference_update discard intersection intersection_update isdisjoint issubset issuperset pop remove symmetric_difference symmetric_difference_update union update',
  tuple: '__add__ __class__ __class_getitem__ __contains__ __delattr__ __dir__ __doc__ __eq__ __format__ __ge__ __getattribute__ __getitem__ __getnewargs__ __gt__ __hash__ __init__ __init_subclass__ __iter__ __le__ __len__ __lt__ __mul__ __ne__ __new__ __reduce__ __reduce_ex__ __repr__ __rmul__ __setattr__ __sizeof__ __str__ __subclasshook__ count index',
  int: '__abs__ __add__ __and__ __bool__ __ceil__ __class__ __delattr__ __dir__ __divmod__ __doc__ __eq__ __float__ __floor__ __floordiv__ __format__ __ge__ __getattribute__ __getnewargs__ __gt__ __hash__ __index__ __init__ __init_subclass__ __int__ __invert__ __le__ __lshift__ __lt__ __mod__ __mul__ __ne__ __neg__ __new__ __or__ __pos__ __pow__ __radd__ __rand__ __rdivmod__ __reduce__ __reduce_ex__ __repr__ __rfloordiv__ __rlshift__ __rmod__ __rmul__ __ror__ __round__ __rpow__ __rrshift__ __rshift__ __rsub__ __rtruediv__ __rxor__ __setattr__ __sizeof__ __str__ __sub__ __subclasshook__ __truediv__ __trunc__ __xor__ as_integer_ratio bit_count bit_length conjugate denominator from_bytes imag numerator real to_bytes',
  float: '__abs__ __add__ __bool__ __ceil__ __class__ __delattr__ __dir__ __divmod__ __doc__ __eq__ __float__ __floor__ __floordiv__ __format__ __ge__ __getattribute__ __getformat__ __getnewargs__ __gt__ __hash__ __init__ __init_subclass__ __int__ __le__ __lt__ __mod__ __mul__ __ne__ __neg__ __new__ __pos__ __pow__ __radd__ __rdivmod__ __reduce__ __reduce_ex__ __repr__ __rfloordiv__ __rmod__ __rmul__ __round__ __rpow__ __rsub__ __rtruediv__ __setattr__ __setformat__ __sizeof__ __str__ __sub__ __subclasshook__ __truediv__ __trunc__ as_integer_ratio conjugate fromhex hex imag is_integer real',
  bool: '__abs__ __add__ __and__ __bool__ __ceil__ __class__ __delattr__ __dir__ __divmod__ __doc__ __eq__ __float__ __floor__ __floordiv__ __format__ __ge__ __getattribute__ __getnewargs__ __gt__ __hash__ __index__ __init__ __init_subclass__ __int__ __invert__ __le__ __lshift__ __lt__ __mod__ __mul__ __ne__ __neg__ __new__ __or__ __pos__ __pow__ __radd__ __rand__ __rdivmod__ __reduce__ __reduce_ex__ __repr__ __rfloordiv__ __rlshift__ __rmod__ __rmul__ __ror__ __round__ __rpow__ __rrshift__ __rshift__ __rsub__ __rtruediv__ __rxor__ __setattr__ __sizeof__ __str__ __sub__ __subclasshook__ __truediv__ __trunc__ __xor__ as_integer_ratio bit_count bit_length conjugate denominator from_bytes imag numerator real to_bytes',
}

export abstract class PyObj {
  /** type(x).__name__, used in error messages */
  abstract readonly cls: string
  abstract repr(): string
  str?(): string
  /** format(x, spec) — f'{x:.3f}'; spec may be empty; undefined = the usual rules */
  format?(spec: string): string | undefined
  /** __hash__: objects with the same key are one dict entry; PyDict.hash of an equal built-in value, if there is one */
  hashKey?(): string
  /** __fspath__: the path this stands for */
  fspath?(): string
  /** type(x), when it is a value the program can also reach another way */
  typeValue?(): Value
  /** __index__: the integer this is when used as a list index or in range() */
  asIndex?(h: Host): number
  /** __float__ / __int__: the number this is, when it has its own rule for not being one */
  asNumber?(h: Host): number
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
  /** an iterator's next item, read as it is asked for; undefined at the end */
  nextItem?(h: Host): Value | undefined
  /** del x[i] */
  delItem?(idx: Value, h: Host): void
  len?(h: Host): number
  contains?(v: Value, h: Host): boolean
  truthy?(h: Host): boolean
  enter?(h: Host): Value
  exit?(h: Host): void
  /** pickle's copy of this object, when copying its fields is not enough; copy deep-copies a part of it */
  pickled?(copy: <T>(part: T) => T): PyObj
  /** np.asarray(x) — lets NumPy functions and plotting take this object */
  toArray?(): NDArray
  /** np.name(…, this, …) handled by the object itself (a Series keeps its index); undefined = treat it as an array */
  npCall?(name: string, args: Value[], kw: Kw, h: Host, impl: Fn): Value | undefined
  /** isinstance(x, T) for a library type named name */
  isa?(name: string): boolean
  /** a richer rendering for the output pane (a table, a figure …) when this is the cell's last value */
  display?(): Display | null
  /** __mro_entries__: what this stands for in a class's bases; [] = nothing at run time (Generic[T], Protocol) */
  mroEntries?(): Value[]
  /** a class statement with this among its bases makes this value instead (typing.NamedTuple) */
  classFromBody?(name: string, ns: Map<string, Value>, h: Host): Value
}

/** 5j: only np.mgrid and friends can use one */
class ImagLiteral extends PyObj {
  readonly cls = 'complex'
  v: number
  constructor(v: number) {
    super()
    this.v = v
  }
  repr() { return `${this.v}j` }
  binop(): Value | undefined {
    throw new PyError('NotImplementedError', 'complex numbers are not supported in this sandbox: arrays here hold real numbers only')
  }
}

/** a.flat: walks the array as one long row, and writes through to it */
class FlatIter extends PyObj {
  readonly cls = 'numpy.flatiter'
  a: NDArray
  constructor(a: NDArray) {
    super()
    this.a = a
  }
  repr() { return '<numpy.flatiter object>' }
  len() { return this.a.size }
  iter() { return this.a.values().map((x) => fromScalar(x, this.a.dtype)) }
  toArray() { return NDArray.create(this.a.values(), [this.a.size], this.a.dtype) }
  at(i: number) {
    const n = this.a.size
    if (i < -n || i >= n) throw new PyError('IndexError', `index ${i} is out of bounds for size ${n}`)
    return i < 0 ? i + n : i
  }
  /** the positions an index picks: an integer, a slice or a list of them */
  picks(idx: Value, h: Host): number[] | null {
    if (isNum(idx)) return null
    const n = this.a.size
    if (idx.k === 'slice') {
      const step = idx.step.k === 'none' ? 1 : h.toInt(idx.step)
      const clamp = (v: number) => (v < 0 ? Math.max(0, v + n) : Math.min(v, n))
      const start = idx.start.k === 'none' ? (step > 0 ? 0 : n - 1) : clamp(h.toInt(idx.start))
      const stop = idx.stop.k === 'none' ? (step > 0 ? n : -1) : clamp(h.toInt(idx.stop))
      const out: number[] = []
      for (let i = start; step > 0 ? i < stop : i > stop; i += step) out.push(i)
      return out
    }
    return toArray(idx).values().map((i) => this.at(i))
  }
  getItem(idx: Value, h: Host): Value {
    const v = this.a.values()
    const many = this.picks(idx, h)
    if (!many) return fromScalar(v[this.at(h.toInt(idx))], this.a.dtype)
    return arr(NDArray.create(many.map((i) => v[i]), [many.length], this.a.dtype))
  }
  setItem(idx: Value, x: Value, h: Host) {
    const ad = this.a.addresses()
    const many = this.picks(idx, h) ?? [this.at(h.toInt(idx))]
    const vals = toArray(x).values()
    many.forEach((i, k) => { this.a.data[ad[i]] = castValue(vals[k % vals.length], this.a.dtype) })
  }
}

/** a.flags */
class ArrayFlags extends PyObj {
  readonly cls = 'numpy.flagsobj'
  a: NDArray
  constructor(a: NDArray) {
    super()
    this.a = a
  }
  table(): Record<string, boolean> {
    const c = this.a.isCContiguous()
    const f = this.a.ndim <= 1 ? c : false
    return { C_CONTIGUOUS: c, F_CONTIGUOUS: f, OWNDATA: !this.a.base, WRITEABLE: true, ALIGNED: true, WRITEBACKIFCOPY: false }
  }
  repr() { return Object.entries(this.table()).map(([k, v]) => `  ${k} : ${v ? 'True' : 'False'}`).join('\n') }
  getItem(idx: Value): Value {
    const v = this.table()[(idx as { v: string }).v]
    if (v === undefined) throw new PyError('KeyError', repr(idx))
    return bool(v)
  }
  getAttr(name: string): Value | undefined {
    const short: Record<string, string> = { c_contiguous: 'C_CONTIGUOUS', contiguous: 'C_CONTIGUOUS', f_contiguous: 'F_CONTIGUOUS', fortran: 'F_CONTIGUOUS', owndata: 'OWNDATA', writeable: 'WRITEABLE', aligned: 'ALIGNED' }
    const key = short[name]
    return key ? bool(this.table()[key]) : undefined
  }
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
  /** a variable visible at the current line, by name */
  lookup(id: string): Value | undefined
  /** current line, and the source text of the call or operator being evaluated */
  readonly line: number
  readonly code: string
  /** false inside a loop after its first iterations, so a 100-epoch loop does not record 100 copies of each step */
  readonly tracing: boolean
  /** the variable name a value is bound to, or the given fallback */
  nameOf(v: Value, fallback: string): string
  emit(ev: PyEventInput): void
  /** groups: for kind 'group', the input cells behind each result cell ([operand, flat index]) */
  traceCall(api: string, kind: ApiKind, operands: Operand[], result: GridSnapshot | null, resultText: string, axis?: number | null, groups?: [number, number][][]): void
  print(s: string): void
  /** what print() writes to while sys.stdout is replaced; null = the run's output */
  stdoutFile: Value | null
  readonly rng: SandboxRandom
  /** np.random's global stream: NumPy's legacy MT19937 RandomState */
  readonly npRng: LegacyRandom
  /** what np.random.seed / RandomState(seed) accept: one 32-bit integer, a list of them, or None for a fresh seed */
  legacySeed(v: Value | undefined): number | number[]
  /** per-run state a library keeps between calls (pyplot's current figure …) */
  readonly state: Map<string, unknown>
  /** the run's array size limit */
  readonly maxSize: number
  binop(op: string, l: Value, r: Value): Value
  compare(op: string, l: Value, r: Value): Value
  repr(v: Value): string
  sorted(items: Value[], key?: Value, reverse?: Value): Value[]
  /** a lazy walk over any iterable: undefined when it ends */
  iterOf(v: Value): () => Value | undefined
  /** an attribute if the value has one, without raising */
  getAttrOr(obj: Value, name: string): Value | undefined
  /** the exception class of this name */
  excClass(name: string): ClassValue
  /** raise a Python exception of this class */
  raise(cls: ClassValue, args: Value[]): never
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

const KEEPS_MATRIX: ApiKind[] = ['move', 'elementwise', 'reduce', 'scan', 'sort', 'matmul', 'unique', 'linalg']
/** these go through np.asarray or ignore subclasses, so they return a plain array even for a matrix */
const DROPS_MATRIX = ['copy', 'ravel', 'expand_dims', 'where', 'norm']

// ---- np.matrix: an ndarray with a flag; these mirror numpy.matrix.__array_finalize__ and friends
const isMatrix = (v: Value | undefined): boolean =>
  !!v && ((v.k === 'array' && !!v.a.matrix) || ((v.k === 'list' || v.k === 'tuple') && v.items.some(isMatrix)))
/** the same memory, seen as a plain ndarray */
const plainView = (a: NDArray): NDArray => new NDArray(a.data, a.shape, a.strides, a.offset, a.dtype, a.base ?? a)
/** the shape a result takes once it is a matrix: 0-d → 1×1, 1-D → a row, extra length-1 axes dropped */
const matrixArray = (a: NDArray): NDArray => {
  const sh = a.ndim > 2 ? a.shape.filter((d) => d > 1) : a.shape
  if (sh.length > 2) throw new PyError('ValueError', 'shape too large to be a matrix.')
  const shape = sh.length === 2 ? sh : sh.length === 1 ? [1, sh[0]] : [1, 1]
  const out = shape.length === a.ndim && shape.every((d, i) => d === a.shape[i]) ? plainView(a) : reshape(a, shape)
  out.matrix = true
  return out
}
const plainValue = (v: Value): Value => (v.k === 'array' ? arr(plainView(v.a)) : v)
const toMatrix = (v: Value): Value => (v.k === 'array' ? arr(matrixArray(v.a)) : v)
/** run an API so that any matrix argument makes the array it returns a matrix too */
const keepMatrix = (call: Fn): Fn => (args, kw) => {
  const res = call(args, kw)
  return args.some(isMatrix) ? toMatrix(res) : res
}
const tuple = (items: Value[]): Value => ({ k: 'tuple', items })
const str_ = (v: string): Value => ({ k: 'str', v })

/** constructors and helpers for library code */
export const py = { NONE, int, float, bool, arr, tuple, str: str_, list: (items: Value[]): Value => ({ k: 'list', items }), obj: (o: PyObj): Value => ({ k: 'obj', o }) }

export const typeName = (v: Value): string => {
  switch (v.k) {
    case 'none': return 'NoneType'
    case 'array': return v.a.matrix ? 'numpy.matrix' : 'numpy.ndarray'
    case 'fn': return 'builtin_function_or_method'
    case 'type': case 'class': return 'type'
    case 'dtype': return 'numpy.dtype'
    case 'set': return v.frozen ? 'frozenset' : 'set'
    case 'iter': return v.name
    case 'descr': return v.kind === 'prop' ? 'property' : v.kind === 'class' ? 'classmethod' : 'staticmethod'
    case 'list': return v.view ?? 'list'
    case 'tuple': return v.tname ?? 'tuple'
    case 'dict': return v.flavor ?? 'dict'
    case 'poly': return 'Polynomial'
    case 'carray': return 'numpy.ndarray'
    case 'func': return 'function'
    case 'bound': return 'method'
    case 'inst': return v.cls.name
    case 'obj': return v.o.cls
    default: return v.k
  }
}

export const fromScalar = (x: number, dtype: DType): Value =>
  dtype === 'bool' ? bool(x !== 0) : dtype === 'float64' ? float(x) : dtype === 'float32' ? { k: 'float', v: Math.fround(x), f32: true } : int(x)

type NumValue = Extract<Value, { k: 'int' | 'float' | 'bool' }>
export const isNum = (v: Value): v is NumValue => v.k === 'int' || v.k === 'float' || v.k === 'bool'
export const numOf = (v: Value): number => (v.k === 'bool' ? (v.v ? 1 : 0) : v.k === 'int' || v.k === 'float' ? v.v : NaN)
const scalarDType = (v: Value): DType => (v.k === 'bool' ? 'bool' : v.k === 'int' ? 'int64' : v.k === 'float' && v.f32 ? 'float32' : 'float64')

const reprStr = (s: string) => {
  const q = s.includes("'") && !s.includes('"') ? '"' : "'"
  const body = s.replace(/\\/g, '\\\\').replace(/\n/g, '\\n').replace(/\t/g, '\\t').replace(/\r/g, '\\r')
    // eslint-disable-next-line no-control-regex -- what Python's repr spells in hex
    .replace(/[\x00-\x1f\x7f]/g, (c) => `\\x${c.charCodeAt(0).toString(16).padStart(2, '0')}`)
  return q + (q === "'" ? body.replace(/'/g, "\\'") : body) + q
}

const rangeLen = (r: { start: number; stop: number; step: number }) => Math.max(0, Math.ceil((r.stop - r.start) / r.step))

/** a set prints its small integers in order, as CPython's does; other items in insertion order */
export const setItems = (v: Value & { k: 'set' }): Value[] => {
  const items = v.s.keys()
  return items.every((x) => x.k === 'int' || x.k === 'bool') ? [...items].sort((a, b) => numOf(a) - numOf(b)) : items
}
export const makeSet = (items: Value[], frozen = false): Value => {
  const s = new PyDict()
  for (const x of items) s.set(x, x)
  return frozen ? { k: 'set', s, frozen } : { k: 'set', s }
}

const inRepr = new Set<Value>()
/** a list that holds itself prints as [[...]] */
const guardedRepr = (v: Value, dots: string, make: () => string): string => {
  if (inRepr.has(v)) return dots
  inRepr.add(v)
  try {
    return make()
  } finally {
    inRepr.delete(v)
  }
}

/** a user class's __repr__, while a program runs (a list of objects prints through this too) */
let instRepr: ((v: InstValue) => string | undefined) | null = null

export const repr = (v: Value): string => {
  switch (v.k) {
    // above 2**53 an integer here is a double: its exact digits, not the shortest that read back
    case 'int': return Number.isSafeInteger(v.v) || !Number.isFinite(v.v) ? String(v.v) : String(BigInt(v.v))
    case 'float': return formatScalar(v.v, v.f32 ? 'float32' : 'float64')
    case 'bool': return v.v ? 'True' : 'False'
    case 'none': return 'None'
    case 'str': return reprStr(v.v)
    case 'list': return guardedRepr(v, '[...]', () => (v.view ? `${v.view}([${v.items.map(repr).join(', ')}])` : `[${v.items.map(repr).join(', ')}]`))
    case 'tuple':
      if (v.names) return `${v.tname}(${v.items.map((x, i) => `${v.names![i]}=${repr(x)}`).join(', ')})`
      return v.items.length === 1 ? `(${repr(v.items[0])},)` : `(${v.items.map(repr).join(', ')})`
    case 'dict': return guardedRepr(v, '{...}', () => {
      const entries = v.flavor === 'Counter' ? [...v.d.items()].sort((a, b) => numOf(b[1]) - numOf(a[1])) : v.d.items()
      const body = entries.length || !v.flavor ? `{${entries.map(([k, x]) => `${repr(k)}: ${repr(x)}`).join(', ')}}` : ''
      if (v.flavor === 'defaultdict') return `defaultdict(${v.factory ? repr(v.factory) : 'None'}, ${body || '{}'})`
      return v.flavor ? `${v.flavor}(${body})` : body
    })
    case 'set': {
      const items = setItems(v).map(repr).join(', ')
      if (v.frozen) return items ? `frozenset({${items}})` : 'frozenset()'
      return items ? `{${items}}` : 'set()'
    }
    case 'iter': return v.shown?.() ?? (v.name === 'generator' ? '<generator object>' : `<${v.name} object>`)
    case 'descr': return `<${typeName(v)} object>`
    case 'range': return v.step === 1 ? `range(${v.start}, ${v.stop})` : `range(${v.start}, ${v.stop}, ${v.step})`
    case 'slice': return `slice(${repr(v.start)}, ${repr(v.stop)}, ${repr(v.step)})`
    case 'ellipsis': return 'Ellipsis'
    case 'array': return arrayRepr(v.a)
    case 'fn': return v.shown ?? `<built-in function ${v.name}>`
    case 'type': return `<class '${v.name}'>`
    case 'module': return `<module '${v.name}'>`
    case 'dtype': return `dtype('${v.d}')`
    case 'poly': return polyRepr(v.coef, v.domain, v.window)
    case 'carray': return complexRepr(v.re, v.im)
    case 'func': return `<function ${v.owner ? `${v.owner.name}.` : ''}${v.name}>`
    case 'bound': return `<bound method ${v.f.owner ? `${v.f.owner.name}.` : ''}${v.f.name} of ${repr(v.self)}>`
    case 'class': return v.meta?.repr?.(v) ?? (v.builtin ? `<class '${v.module ? `${v.module}.` : ''}${v.name}'>` : `<class '__main__.${v.name}'>`)
    case 'inst': return instRepr?.(v) ?? v.cls.host?.repr?.(v) ?? `<${v.cls.module ?? '__main__'}.${v.cls.name} object>`
    case 'super': return `<super: <class '${v.after.name}'>, <${v.self.cls.name} object>>`
    case 'obj': return v.o.repr()
  }
}

export const str = (v: Value): string => {
  if (v.k === 'carray') return complexRepr(v.re, v.im).replace(/^array\(/, '').replace(/\)$/, '').replace(/, /g, ' ')
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
  // a library type decides first, as __format__ does: f'{torch.tensor(1.5)}' is 1.5
  if (v.k === 'obj') {
    const s = v.o.format?.(spec)
    if (s !== undefined) return s
  }
  // format() reads a float32 scalar as a Python float, unlike str()
  if (!spec) return v.k === 'float' && v.f32 ? repr(float(v.v)) : str(v)
  const m = /^(?:(.)?([<>^=]))?([+\- ])?(#)?(0)?(\d+)?([,_])?(?:\.(\d+))?([bcdeEfFgGnosxX%])?$/.exec(spec)
  if (!m) throw new PyError('ValueError', `Invalid format specifier '${spec}'`)
  const [, fill0, align0, sign, alt, zero, widthS, group, precS, type] = m
  const width = widthS ? Number(widthS) : 0
  const prec = precS !== undefined ? Number(precS) : null
  let body: string
  const numeric = isNum(v) || (v.k === 'array' && v.a.size === 1)
  if (numeric) {
    const x = v.k === 'array' ? v.a.values()[0] : numOf(v)
    const isInt = v.k === 'int' || v.k === 'bool' || (v.k === 'array' && !isFloatD(v.a.dtype))
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
      case 'x': body = (alt ? '0x' : '') + ax.toString(16); break
      case 'X': body = (alt ? '0X' : '') + ax.toString(16).toUpperCase(); break
      case 'o': body = (alt ? '0o' : '') + ax.toString(8); break
      case 'b': body = (alt ? '0b' : '') + ax.toString(2); break
      case 'c': body = String.fromCodePoint(ax); break
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
  '+': '__add__', '-': '__sub__', '*': '__mul__', '/': '__truediv__', '//': '__floordiv__', '%': '__mod__', '@': '__matmul__', '**': '__pow__',
  '&': '__and__', '|': '__or__', '^': '__xor__', '<<': '__lshift__', '>>': '__rshift__',
  '==': '__eq__', '!=': '__ne__', '<': '__lt__', '>': '__gt__', '<=': '__le__', '>=': '__ge__',
}
/** what the right operand is asked when the left one has no answer */
const REFLECTED: Record<string, string> = {
  '+': '__radd__', '-': '__rsub__', '*': '__rmul__', '/': '__rtruediv__', '//': '__rfloordiv__', '%': '__rmod__', '@': '__rmatmul__', '**': '__rpow__',
  '&': '__rand__', '|': '__ror__', '^': '__rxor__', '<<': '__rlshift__', '>>': '__rrshift__',
  '==': '__eq__', '!=': '__ne__', '<': '__gt__', '>': '__lt__', '<=': '__ge__', '>=': '__le__',
}
export const NOT_IMPLEMENTED: Value = { k: 'type', name: 'NotImplementedType', call: () => NONE }

/** the builtin exception tree: [name, bases] */
const EXCEPTIONS: [string, string[]][] = [
  ['BaseException', []], ['Exception', ['BaseException']], ['SystemExit', ['BaseException']], ['KeyboardInterrupt', ['BaseException']], ['GeneratorExit', ['BaseException']],
  ['ArithmeticError', ['Exception']], ['ZeroDivisionError', ['ArithmeticError']], ['OverflowError', ['ArithmeticError']], ['FloatingPointError', ['ArithmeticError']],
  ['AssertionError', ['Exception']], ['AttributeError', ['Exception']], ['EOFError', ['Exception']], ['ImportError', ['Exception']], ['ModuleNotFoundError', ['ImportError']],
  ['LookupError', ['Exception']], ['IndexError', ['LookupError']], ['KeyError', ['LookupError']], ['MemoryError', ['Exception']], ['NameError', ['Exception']], ['UnboundLocalError', ['NameError']],
  ['OSError', ['Exception']], ['FileNotFoundError', ['OSError']], ['FileExistsError', ['OSError']], ['IsADirectoryError', ['OSError']], ['NotADirectoryError', ['OSError']], ['PermissionError', ['OSError']], ['TimeoutError', ['OSError']],
  ['RuntimeError', ['Exception']], ['NotImplementedError', ['RuntimeError']], ['RecursionError', ['RuntimeError']], ['StopIteration', ['Exception']],
  ['SyntaxError', ['Exception']], ['IndentationError', ['SyntaxError']], ['TypeError', ['Exception']], ['ValueError', ['Exception']], ['UnicodeError', ['ValueError']], ['UnicodeDecodeError', ['UnicodeError']],
  ['_LinAlgError', ['RuntimeError']], ['PickleError', ['Exception']], ['PicklingError', ['PickleError']], ['UnpicklingError', ['PickleError']], ['Warning', ['Exception']], ['UserWarning', ['Warning']], ['DeprecationWarning', ['Warning']], ['RuntimeWarning', ['Warning']], ['FutureWarning', ['Warning']],
  ['LinAlgError', ['ValueError']], ['AxisError', ['ValueError', 'IndexError']], ['UFuncTypeError', ['TypeError']],
  ['MergeError', ['ValueError']], ['ParserError', ['ValueError']], ['EmptyDataError', ['ValueError']], ['DataError', ['ValueError']], ['IntCastingNaNError', ['ValueError']], ['UndefinedVariableError', ['NameError']], ['OptionError', ['AttributeError', 'KeyError']], ['IndexingError', ['Exception']], ['SpecificationError', ['Exception']],
]
const isOSError = (c: ClassValue): boolean => c.name === 'OSError' || c.bases.some((b) => b.k === 'class' && isOSError(b))
/** every exception keeps its constructor arguments in .args */
const EXC_HOST: HostClass = {
  init(inst, args) {
    // OSError(errno, strerror, filename, winerror, filename2) keeps the file names out of .args
    const named = args.length >= 3 && args.length <= 5 && isOSError(inst.cls)
    if (named) inst.attrs.set('filename', args[2])
    if (named && args[4]) inst.attrs.set('filename2', args[4])
    inst.attrs.set('args', { k: 'tuple', items: named ? args.slice(0, 2) : args })
  },
  repr(inst) {
    const a = inst.attrs.get('args')
    const items = a?.k === 'tuple' ? a.items : []
    return `${inst.cls.name}(${items.map(repr).join(', ')})`
  },
  getAttr(inst, name) {
    if ((name === 'errno' || name === 'strerror' || name === 'filename' || name === 'filename2') && isOSError(inst.cls)) {
      const a = inst.attrs.get('args')
      return (name.startsWith('filename') || a?.k !== 'tuple' || a.items.length !== 2 ? undefined : a.items[name === 'errno' ? 0 : 1]) ?? NONE
    }
    // StopIteration.value and SystemExit.code are the first argument
    if (name !== 'value' && name !== 'code') return undefined
    if ((name === 'value') !== inst.cls.name.endsWith('StopIteration') && (name === 'code') !== inst.cls.name.endsWith('SystemExit')) return undefined
    const a = inst.attrs.get('args')
    return (a?.k === 'tuple' ? a.items[0] : undefined) ?? NONE
  },
}

const stmtHasYield = new WeakMap<Stmt, boolean>()
/** does this statement yield, not counting functions defined inside it? */
const hasYield = (st: Stmt): boolean => {
  const known = stmtHasYield.get(st)
  if (known !== undefined) return known
  const any = (b: Stmt[] | null | undefined) => !!b && b.some(hasYield)
  let r = false
  switch (st.k) {
    case 'yield': r = true; break
    case 'if': r = st.branches.some((b) => any(b.body)); break
    case 'for': case 'while': r = any(st.body) || any(st.orelse); break
    case 'with': case 'seq': r = any(st.body); break
    case 'try': r = any(st.body) || any(st.orelse) || any(st.final) || st.handlers.some((h) => any(h.body)); break
  }
  stmtHasYield.set(st, r)
  return r
}

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
  // np.array([Level.LOW]): a member of class Level(int, Enum) is that int
  const same = x.k === 'inst' ? x.cls.meta?.scalar?.(x) : undefined
  if (same && isNum(same)) return same
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
    kind = promoteD(kind, d)
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
  // NumPy 2 refuses a Python int that does not fit a uint8; floats and arrays wrap instead
  if (dtype === 'uint8' && !isFloatD(kind)) {
    const bad = flat.find((x) => x < 0 || x > 255)
    if (bad !== undefined) throw new PyError('OverflowError', `Python integer ${bad} out of bounds for uint8`)
  }
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
  /** order among calls and library events */
  seq: number
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
  /** kind = 'group'：结果每个元素来自哪一组输入元素（由函数自己算出，如 bincount 的计数、pad 的边框） */
  groups?: [number, number][][]
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
/** arrays past this size run as usual but stay out of the step list: a copy per step, for a grid too big to show */
const TRACE_CELLS = 4096
const tooBigToTrace = (v: Value | undefined) => v?.k === 'array' && v.a.size > TRACE_CELLS

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
  npRng = new LegacyRandom(2211)
  /** 多项式调用把要画的曲线放在这里，由 pushCall 取走 */
  pendingPlot: PolyPlot | null = null
  /** set by a 'group' function during its call: the provenance of each result cell, and the array to show as the result */
  pendingGroups: { groups: [number, number][][]; result: NDArray } | null = null
  builtins: Record<string, Value>
  np: Value
  libs: PyLib[]
  loaded = new Map<PyLib, Record<string, Value>>()
  state = new Map<string, unknown>()
  maxSize: number
  lineRuns = new Map<number, number>()
  callText = ''
  steps = 0
  seqN = 0
  depth = 0
  deadline: number
  excClasses = new Map<string, ClassValue>()
  /** the exceptions being handled, innermost last: what a bare raise re-raises */
  handling: PyError[] = []
  stdoutFile: Value | null = null

  constructor(src: string, opts: RunOptions = {}) {
    this.src = src
    this.libs = [...(opts.libs ?? []), STDLIB]
    this.maxSize = opts.maxSize ?? MAX_SIZE
    this.deadline = Date.now() + (opts.timeBudget ?? 2000)
    this.np = this.makeNumpy()
    this.builtins = this.makeBuiltins()
    // these exceptions live in NumPy's namespace but need the exception tree, which comes second
    const np = (this.np as { attrs: Record<string, Value> }).attrs
    const linalgAttrs = (np.linalg as { attrs: Record<string, Value> }).attrs
    linalgAttrs.LinAlgError = this.excClass('LinAlgError')
    np.exceptions = { k: 'module', name: 'numpy.exceptions', attrs: { AxisError: this.excClass('AxisError') } }
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
    const f = this.frame
    if (!f) {
      this.env.set(id, v)
      return
    }
    if (f.globals?.has(id)) {
      this.env.set(id, v)
      return
    }
    if (f.nonlocals?.has(id)) {
      for (let p = f.parent; p; p = p.parent) {
        if (!p.isClass && p.vars.has(id)) {
          p.vars.set(id, v)
          return
        }
      }
    }
    f.vars.set(id, v)
  }

  get tracing(): boolean {
    return !this.quiet && !this.replaying && (this.lineRuns.get(this.line) ?? 0) <= TRACE_REPEATS && this.traces.length + this.calls.length + this.events.length < MAX_RECORDS
  }

  get code(): string {
    return this.callText
  }

  tick() {
    if ((++this.steps & 1023) === 0 && Date.now() > this.deadline) {
      const e = this.err('TimeoutError', 'the code ran too long for this sandbox (an endless loop, or too many iterations?)')
      e.fatal = true
      throw e
    }
  }

  emit(ev: PyEventInput) {
    if (this.events.length + this.calls.length + this.traces.length >= MAX_RECORDS) return
    this.events.push({ ...ev, id: this.events.length, seq: this.seqN++, line: ev.line ?? this.line, code: ev.code ?? this.callText } as PyEvent)
  }

  traceCall(api: string, kind: ApiKind, operands: Operand[], result: GridSnapshot | null, resultText: string, axis: number | null = null, groups?: [number, number][][]) {
    if (!this.tracing) return
    this.calls.push({ id: this.calls.length, seq: this.seqN++, line: this.line, code: this.callText, api, kind, operands, axis, result, resultText, groups })
  }

  print(s: string) {
    if (!this.stdoutFile) this.stdout.push(s)
    else if (this.stdoutFile.k !== 'none') this.call(this.getAttr(this.stdoutFile, 'write'), [str_(s)])
  }

  num(v: Value, what = 'a number'): number {
    if (isNum(v)) return numOf(v)
    if (v.k === 'obj' && v.o.asNumber) return v.o.asNumber(this)
    if (v.k === 'array' && v.a.size === 1) return v.a.values()[0]
    if (v.k === 'obj' && v.o.toArray) {
      const a = v.o.toArray()
      if (a.size === 1) return a.values()[0]
    }
    if (v.k === 'inst') {
      const f = this.classAttr(v.cls, '__float__') ?? this.classAttr(v.cls, '__index__')
      if (f) return this.num(this.call(f, [v]), what)
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
    if (name === 'builtins') return { k: 'module', name, attrs: this.builtins }
    if (name.startsWith('numpy.')) {
      const sub = np.attrs[name.slice(6)]
      if (sub?.k === 'module') return sub
    }
    if (name === 'math') return this.builtins.__math__
    if (name === 'random') return this.builtins.__random__
    const lib = this.libs.find((l) => l.modules.includes(name))
    if (!lib) {
      const have = ['numpy', 'math', 'random', ...this.libs.flatMap((l) => l.modules.filter((m) => !m.includes('.'))), ...ON_DEMAND]
      // only reached when the program's text did not show the import, or the download failed
      if (ON_DEMAND.includes(name.split('.')[0])) {
        throw this.err('ModuleNotFoundError', `No module named '${name}': this sandbox fetches it when an import line names it, and it has not arrived`)
      }
      if (/^(scipy|PIL|skimage|statsmodels|networkx|sympy|requests)(\.|$)/.test(name)) {
        throw this.err('ModuleNotFoundError', `No module named '${name}' in step-through mode: switch to Real Python to import it (this sandbox has ${[...new Set(have)].join(', ')})`)
      }
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
      const f = this.classAttr(v.cls, '__str__')
      if (f) return str(this.call(f, [v]))
      if (this.isException(v)) {
        const text = v.state.text
        if (typeof text === 'string') return text
        const a = v.attrs.get('args')
        const items = a?.k === 'tuple' ? a.items : []
        if (!items.length) return ''
        if (items.length === 2 && isOSError(v.cls)) {
          const [file, file2] = [v.attrs.get('filename'), v.attrs.get('filename2')]
          return `[Errno ${this.str(items[0])}] ${this.str(items[1])}${file ? `: ${repr(file)}` : ''}${file2 ? ` -> ${repr(file2)}` : ''}`
        }
        if (items.length > 1) return repr(a!)
        return this.isSubclass(v.cls, this.excClass('KeyError')) ? repr(items[0]) : this.str(items[0])
      }
      return repr(v)
    }
    if (v.k === 'list' || v.k === 'tuple' || v.k === 'dict' || v.k === 'set') return repr(v)
    return str(v)
  }

  repr(v: Value): string {
    return repr(v)
  }

  getAttrOr(obj: Value, name: string): Value | undefined {
    try {
      return this.getAttr(obj, name)
    } catch (e) {
      if (e instanceof PyError) return undefined
      throw e
    }
  }

  // ---------- 异常 ----------

  excClass(name: string): ClassValue {
    return this.excClasses.get(name) ?? this.excClasses.get('Exception')!
  }

  isSubclass(c: ClassValue, of: ClassValue): boolean {
    return c === of || c.bases.some((b) => b.k === 'class' && this.isSubclass(b, of))
  }

  isException(v: Value): v is InstValue {
    return v.k === 'inst' && this.isSubclass(v.cls, this.excClass('BaseException'))
  }

  /** what except can catch; null for the interpreter's own control flow, and for the sandbox stopping the run */
  asPyError(e: unknown): PyError | null {
    if (e instanceof RangeError) return new PyError('RecursionError', 'maximum recursion depth exceeded', this.line)
    return e instanceof PyError && !e.fatal ? e : null
  }

  /** the exception object behind an error, whether a program raised it or the sandbox did */
  excOf(e: PyError): InstValue {
    if (e.exc) return e.exc as InstValue
    const inst: InstValue = { k: 'inst', cls: this.excClass(e.pyType), attrs: new Map(), state: { text: e.message } }
    // the interpreter words a KeyError as the key's repr; args holds the key itself
    const quoted = e.pyType === 'KeyError' ? /^'(.*)'$/.exec(e.message) : null
    const number = e.pyType === 'KeyError' && /^-?\d+$/.test(e.message)
    EXC_HOST.init(inst, [quoted ? str_(quoted[1]) : number ? int(Number(e.message)) : str_(e.message)], {}, this)
    e.exc = inst
    return inst
  }

  raise(cls: ClassValue, args: Value[]): never {
    this.throwExc(this.instantiate(cls, args, {}) as InstValue)
  }

  throwExc(inst: InstValue): never {
    const e = new PyError(inst.cls.name, this.str(inst), this.line)
    e.exc = inst
    throw e
  }

  /** does this exception match an except clause's class, or tuple of classes? */
  excMatches(inst: InstValue, types: Value): boolean {
    if (types.k === 'tuple') return types.items.some((t) => this.excMatches(inst, t))
    if (types.k === 'class' && this.isSubclass(types, this.excClass('BaseException'))) return this.isSubclass(inst.cls, types)
    throw this.err('TypeError', 'catching classes that do not inherit from BaseException is not allowed')
  }

  /** the handler for this error, if any, with the error being handled */
  findHandler(st: Stmt & { k: 'try' }, err: PyError) {
    const inst = this.excOf(err)
    return st.handlers.find((h) => {
      this.line = h.line
      return !h.types || this.excMatches(inst, this.eval(h.types))
    })
  }

  /** the setup and teardown around an except block's body */
  enterHandler(h: { name: string | null }, err: PyError) {
    this.handling.push(err)
    if (h.name) this.bind(h.name, this.excOf(err))
  }

  leaveHandler(h: { name: string | null }) {
    this.handling.pop()
    if (h.name) (this.frame ? this.frame.vars : this.env).delete(h.name)
  }

  /** a context manager's enter, and an exit that says whether to swallow the error */
  enterContext(ctx: Value): { entered: Value; exit: (e: PyError | null) => boolean } {
    if (ctx.k === 'obj' && ctx.o.enter) {
      const entered = ctx.o.enter(this)
      return { entered, exit: () => { ctx.o.exit?.(this); return false } }
    }
    if (ctx.k === 'inst') {
      const enter = this.classAttr(ctx.cls, '__enter__')
      const exit = this.classAttr(ctx.cls, '__exit__')
      if (enter && exit) {
        const entered = this.call(enter, [ctx])
        return {
          entered,
          exit: (e) => {
            const inst = e ? this.excOf(e) : null
            return this.truthy(this.call(exit, [ctx, inst ? inst.cls : NONE, inst ?? NONE, NONE]))
          },
        }
      }
    }
    throw this.err('TypeError', `'${typeName(ctx)}' object does not support the context manager protocol`)
  }

  // ---------- 迭代与生成器 ----------

  isStop(e: unknown): boolean {
    return e instanceof PyError && this.isSubclass(this.excOf(e).cls, this.excClass('StopIteration'))
  }

  iterOf(v: Value): () => Value | undefined {
    if (v.k === 'iter') return v.next
    if (v.k === 'obj' && v.o.nextItem) return () => (v as { o: PyObj }).o.nextItem!(this)
    if (v.k === 'inst') {
      const it = this.classAttr(v.cls, '__iter__')
      if (it) {
        const r = this.call(it, [v])
        if (r.k !== 'inst') return this.iterOf(r)
        const nx = this.classAttr(r.cls, '__next__')
        if (!nx) throw this.err('TypeError', `iter() returned non-iterator of type '${r.cls.name}'`)
        return () => {
          try {
            return this.call(nx, [r])
          } catch (e) {
            if (this.isStop(e)) return undefined
            throw e
          }
        }
      }
      const gi = this.classAttr(v.cls, '__getitem__')
      if (gi) {
        let i = 0
        return () => {
          try {
            return this.call(gi, [v, int(i++)])
          } catch (e) {
            if (e instanceof PyError && this.isSubclass(this.excOf(e).cls, this.excClass('IndexError'))) return undefined
            throw e
          }
        }
      }
      throw this.err('TypeError', `'${v.cls.name}' object is not iterable`)
    }
    const items = this.iterate(v)
    let i = 0
    return () => (i < items.length ? items[i++] : undefined)
  }

  /** a call of a generator function: nothing runs until the first next() */
  makeGenerator(f: FuncValue, frame: Frame): Value {
    const g = this.genBlock(f.body!)
    let done = false
    const resume = (step: () => IteratorResult<Value, void>): Value | undefined => {
      if (done) return undefined
      const saved = this.frame
      this.frame = frame
      this.depth++
      try {
        const r = step()
        if (r.done) done = true
        return r.done ? undefined : r.value
      } catch (e) {
        done = true
        if (e instanceof ReturnSig) {
          self.result = e.v
          return undefined
        }
        throw e
      } finally {
        this.frame = saved
        this.depth--
      }
    }
    const self: Value & { k: 'iter' } = {
      k: 'iter', name: 'generator', next: () => resume(() => g.next(NONE)), send: (v) => resume(() => g.next(v)),
      throw: (e) => {
        if (done) throw e
        return resume(() => g.throw(e))
      },
      close: () => {
        if (done) return
        done = true
        const saved = this.frame
        this.frame = frame
        try {
          g.return(undefined)
        } finally {
          this.frame = saved
        }
      },
    }
    return self
  }

  *genBlock(stmts: Stmt[]): Generator<Value, void, Value> {
    for (const st of stmts) {
      if (hasYield(st)) yield* this.genStmt(st)
      else this.execStmt(st)
    }
  }

  /** the compound statements a yield can sit inside, run so that they can pause at it */
  *genStmt(st: Stmt): Generator<Value, void, Value> {
    this.at(st.line)
    switch (st.k) {
      case 'yield': {
        if (st.from) {
          const src = this.eval(st.x!)
          const next = this.iterOf(src)
          for (let v = next(); v !== undefined;) {
            try {
              yield v
              v = next()
            } catch (e) {
              // gen.throw() goes on to the generator being delegated to
              if (src.k !== 'iter' || !src.throw) throw e
              v = src.throw(e)
            }
          }
          this.line = st.line
          // x = yield from g: x is what g returned
          for (const t of st.targets) this.assign(t, src.k === 'iter' ? src.result ?? NONE : NONE)
          return
        }
        const sent = yield st.x ? this.eval(st.x) : NONE
        this.line = st.line
        for (const t of st.targets) this.assign(t, sent)
        return
      }
      case 'seq': yield* this.genBlock(st.body); return
      case 'if':
        for (const b of st.branches) {
          if (b.cond === null || this.truthy(this.eval(b.cond))) {
            yield* this.genBlock(b.body)
            return
          }
        }
        return
      case 'for': {
        const next = this.iterOf(this.eval(st.iter))
        for (let item = next(); item !== undefined; item = next()) {
          this.line = st.line
          this.tick()
          this.assign(st.target, item)
          try {
            yield* this.genBlock(st.body)
          } catch (e) {
            if (e instanceof BreakSig) return
            if (e instanceof ContinueSig) continue
            throw e
          }
        }
        if (st.orelse) yield* this.genBlock(st.orelse)
        return
      }
      case 'while':
        while (this.truthy(this.eval(st.cond))) {
          try {
            yield* this.genBlock(st.body)
          } catch (e) {
            if (e instanceof BreakSig) return
            if (e instanceof ContinueSig) continue
            throw e
          }
          this.at(st.line)
        }
        if (st.orelse) yield* this.genBlock(st.orelse)
        return
      case 'try':
        try {
          try {
            yield* this.genBlock(st.body)
          } catch (e) {
            const err = this.asPyError(e)
            const h = err && this.findHandler(st, err)
            if (!err || !h) throw e
            this.enterHandler(h, err)
            try {
              yield* this.genBlock(h.body)
            } finally {
              this.leaveHandler(h)
            }
            return
          }
          if (st.orelse) yield* this.genBlock(st.orelse)
        } finally {
          if (st.final) this.execBlock(st.final)
        }
        return
      case 'with': {
        const ctxs: ((e: PyError | null) => boolean)[] = []
        for (const item of st.items) {
          const { entered, exit } = this.enterContext(this.eval(item.x))
          ctxs.push(exit)
          if (item.as) this.assign(item.as, entered)
        }
        try {
          yield* this.genBlock(st.body)
        } catch (e) {
          const err = this.asPyError(e)
          if (!ctxs.reverse().map((x) => x(err)).some(Boolean) || !err) throw e
          return
        }
        ctxs.reverse().forEach((x) => x(null))
        return
      }
    }
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
      case 'set': return v.s.size > 0
      case 'inst': {
        const b = this.classAttr(v.cls, '__bool__')
        if (b) return this.truthy(this.call(b, [v]))
        const l = this.classAttr(v.cls, '__len__')
        return l ? this.num(this.call(l, [v])) !== 0 : true
      }
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
    if (v.k === 'obj' && v.o.asIndex) return v.o.asIndex(this)
    if (v.k === 'inst') {
      const f = this.classAttr(v.cls, '__index__')
      if (f) return this.toInt(this.call(f, [v]), what)
    }
    if (v.k === 'array' && v.a.ndim === 0 && !isFloatD(v.a.dtype)) return v.a.values()[0]
    throw this.err('TypeError', `'${typeName(v)}' object cannot be interpreted as ${what}`)
  }

  toShape(args: Value[]): number[] {
    const src = args.length === 1 && (args[0].k === 'tuple' || args[0].k === 'list') ? args[0].items : args
    return src.map((x) => this.toInt(x))
  }

  /** `*` and `**` are the matrix product and power; everything else is element-wise and stays a matrix */
  matrixOp(op: string, l: Value, r: Value): Value {
    if (op === '*' && !isNum(l) && !isNum(r)) {
      // matrix * x reads x as a matrix (a 1-D x is a row); x * matrix goes through np.dot(x, matrix)
      const left = isMatrix(l) ? plainView(toArray(l)) : toArray(l)
      const right = isMatrix(l) ? matrixArray(toArray(r)) : toArray(r)
      return toMatrix(this.binop('@', arr(left), arr(plainView(right))))
    }
    if (op === '**') {
      if (!isMatrix(l)) throw this.err('TypeError', `unsupported operand type(s) for ** or pow(): '${typeName(l)}' and 'matrix'`)
      if (r.k !== 'int' && r.k !== 'bool') throw this.err('TypeError', 'exponent must be an integer')
      return arr(this.matrixPower(toArray(l), numOf(r)))
    }
    return toMatrix(this.binop(op, plainValue(l), plainValue(r)))
  }

  matrixPower(a: NDArray, n: number): NDArray {
    if (a.shape[0] !== a.shape[1]) throw this.err('LinAlgError', 'Last 2 dimensions of the array must be square')
    const base = n < 0 ? inv(plainView(a)) : plainView(a)
    const size = a.shape[0]
    let out = NDArray.create(Array.from({ length: size * size }, (_, i) => (i % (size + 1) === 0 ? 1 : 0)), [size, size], base.dtype)
    for (let i = 0; i < Math.abs(n); i++) out = dot(out, base) as NDArray
    return matrixArray(out)
  }

  binop(op: string, l: Value, r: Value): Value {
    // a.dtype == np.uint8, a.dtype == 'float64', a.dtype == int
    if ((op === '==' || op === '!=') && (l.k === 'dtype' || r.k === 'dtype')) {
      const asDtype = (v: Value) => {
        try {
          return this.dtypeOf(v) ?? null
        } catch {
          return null
        }
      }
      const same = asDtype(l) !== null && asDtype(l) === asDtype(r)
      return bool(op === '==' ? same : !same)
    }
    if (l.k === 'obj' || r.k === 'obj') {
      const res = (l.k === 'obj' ? l.o.binop?.(op, r, false, this) : undefined) ?? (r.k === 'obj' ? r.o.binop?.(op, l, true, this) : undefined)
      if (res) return res
      if (op === '==' || op === '!=') return bool((op === '==') === (l.k === 'obj' && r.k === 'obj' && l.o === r.o))
      // 'text' % obj: the object was asked first, and declined
      if (l.k === 'str' && op === '%') return str_(this.percentFormat(l.v, r))
      if (['<', '>', '<=', '>='].includes(op)) throw this.err('TypeError', `'${op}' not supported between instances of '${typeName(l)}' and '${typeName(r)}'`)
      if (op === '+' && (l.k === 'list' || l.k === 'tuple' || l.k === 'str')) throw this.err('TypeError', `can only concatenate ${l.k} (not "${typeName(r)}") to ${l.k}`)
      throw this.err('TypeError', `unsupported operand type(s) for ${op}: '${typeName(l)}' and '${typeName(r)}'`)
    }
    if (l.k === 'inst' || r.k === 'inst') {
      const ask = (self: Value, name: string | undefined, other: Value): Value | undefined => {
        const f = name && self.k === 'inst' ? this.classAttr(self.cls, name) : undefined
        if (!f) return undefined
        const res = this.call(f, [self, other])
        return res === NOT_IMPLEMENTED ? undefined : res
      }
      const res = ask(l, DUNDER[op], r) ?? ask(r, REFLECTED[op], l)
      if (res) return res
      if (op === '!=') {
        const eq = ask(l, '__eq__', r)
        if (eq) return bool(!this.truthy(eq))
      }
      if (op === '==' || op === '!=') return bool((op === '==') === (l === r))
      // 'text' % obj: the object was asked first, and declined
      if (l.k === 'str' && op === '%') return str_(this.percentFormat(l.v, r))
      if (['<', '>', '<=', '>='].includes(op)) throw this.err('TypeError', `'${op}' not supported between instances of '${typeName(l)}' and '${typeName(r)}'`)
      if (op === '+' && (l.k === 'list' || l.k === 'tuple' || l.k === 'str')) throw this.err('TypeError', `can only concatenate ${l.k} (not "${typeName(r)}") to ${l.k}`)
      throw this.err('TypeError', `unsupported operand type(s) for ${op}: '${typeName(l)}' and '${typeName(r)}'`)
    }
    if (l.k === 'dict' && r.k === 'dict' && l.flavor === 'Counter' && r.flavor === 'Counter' && ['+', '-', '&', '|'].includes(op)) {
      // Counter arithmetic keeps only the keys that end up positive
      const out = new PyDict()
      for (const k of [...l.d.keys(), ...r.d.keys()]) {
        const a = numOf(l.d.get(k) ?? int(0))
        const b = numOf(r.d.get(k) ?? int(0))
        const n = op === '+' ? a + b : op === '-' ? a - b : op === '&' ? Math.min(a, b) : Math.max(a, b)
        if (n > 0) out.set(k, int(n))
      }
      return { k: 'dict', d: out, flavor: 'Counter' }
    }
    if ((op === '<<' || op === '>>') && (l.k === 'array' || r.k === 'array')) {
      return this.call((this.np as { attrs: Record<string, Value> }).attrs[op === '<<' ? 'left_shift' : 'right_shift'], [l, r])
    }
    if (l.k === 'dict' && r.k === 'dict' && op === '|') return { ...l, d: PyDict.from([...l.d.items(), ...r.d.items()]) }
    if (isMatrix(l) || isMatrix(r)) return this.matrixOp(op, l, r)
    if (op === '@') {
      if (l.k !== 'array' && r.k !== 'array') throw this.err('TypeError', `unsupported operand type(s) for @: '${typeName(l)}' and '${typeName(r)}'`)
      if (toArray(l).ndim === 0 || toArray(r).ndim === 0) throw this.err('ValueError', 'matmul: Input operand does not have enough dimensions')
      const res = dot(toArray(l), toArray(r))
      return typeof res === 'number' ? fromScalar(res, promoteD(toArray(l).dtype, toArray(r).dtype) === 'bool' ? 'int64' : promoteD(toArray(l).dtype, toArray(r).dtype)) : arr(res)
    }
    if (l.k === 'array' || r.k === 'array') {
      const la = this.operand(l, op)
      const ra = this.operand(r, op)
      return arr(binaryOp(op as BinOp, la, ra))
    }
    if (isNum(l) && isNum(r)) return this.scalarOp(op, l, r)
    if (l.k === 'set' && r.k === 'set') {
      const res = this.setOp(op, l, r)
      if (res) return res
    }
    if (l.k === 'str' && op === '%') return str_(this.percentFormat(l.v, r))
    if (['<', '>', '<=', '>='].includes(op) && l.k === r.k && (l.k === 'str' || l.k === 'list' || l.k === 'tuple')) {
      const c = this.sequenceOrder(l, r)
      return bool(op === '<' ? c < 0 : op === '>' ? c > 0 : op === '<=' ? c <= 0 : c >= 0)
    }
    if (op === '*' && (r.k === 'list' || r.k === 'tuple' || r.k === 'str') && (l.k === 'int' || l.k === 'bool')) return this.binop('*', r, l)
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
      const eq = this.sameValue(l, r)
      return bool(op === '==' ? eq : !eq)
    }
    if (['<', '>', '<=', '>='].includes(op)) throw this.err('TypeError', `'${op}' not supported between instances of '${typeName(l)}' and '${typeName(r)}'`)
    throw this.err('TypeError', `unsupported operand type(s) for ${op}: '${typeName(l)}' and '${typeName(r)}'`)
  }

  /** == for containers: 1 == 1.0, so [1] == [1.0] */
  sameValue(l: Value, r: Value): boolean {
    if (isNum(l) && isNum(r)) return numOf(l) === numOf(r)
    if ((l.k === 'list' && r.k === 'list') || (l.k === 'tuple' && r.k === 'tuple')) {
      return l.items.length === (r as { items: Value[] }).items.length && l.items.every((x, i) => this.equal(x, (r as { items: Value[] }).items[i]))
    }
    if (l.k === 'dict' && r.k === 'dict') {
      return l.d.size === r.d.size && l.d.items().every(([k, v]) => r.d.has(k) && this.equal(v, r.d.get(k)!))
    }
    if (l.k === 'set' && r.k === 'set') return l.s.size === r.s.size && l.s.keys().every((k) => r.s.has(k))
    if (l.k === 'func' || r.k === 'func' || l.k === 'class' || r.k === 'class') return l === r
    return l.k === r.k && repr(l) === repr(r)
  }

  equal(a: Value, b: Value): boolean {
    // identity first, as Python's containers do: pd.NA or a tensor need not have a truth value
    if (a === b || (a.k === 'obj' && b.k === 'obj' && a.o === b.o)) return true
    return this.truthy(this.binop('==', a, b))
  }

  /** the lexicographic order of two strings, lists or tuples */
  sequenceOrder(a: Value, b: Value): number {
    if (a.k === 'str' && b.k === 'str') return a.v < b.v ? -1 : a.v > b.v ? 1 : 0
    const x = (a as { items: Value[] }).items
    const y = (b as { items: Value[] }).items
    for (let i = 0; i < Math.min(x.length, y.length); i++) {
      if (!this.equal(x[i], y[i])) return this.truthy(this.binop('<', x[i], y[i])) ? -1 : 1
    }
    return x.length - y.length
  }

  setOp(op: string, l: Value & { k: 'set' }, r: Value & { k: 'set' }): Value | null {
    const a = setItems(l)
    const b = setItems(r)
    const frozen = l.frozen
    switch (op) {
      case '|': return makeSet([...a, ...b], frozen)
      case '&': return makeSet(a.filter((x) => r.s.has(x)), frozen)
      case '-': return makeSet(a.filter((x) => !r.s.has(x)), frozen)
      case '^': return makeSet([...a.filter((x) => !r.s.has(x)), ...b.filter((x) => !l.s.has(x))], frozen)
      case '<=': return bool(a.every((x) => r.s.has(x)))
      case '<': return bool(a.length < b.length && a.every((x) => r.s.has(x)))
      case '>=': return bool(b.every((x) => l.s.has(x)))
      case '>': return bool(b.length < a.length && b.every((x) => l.s.has(x)))
    }
    return null
  }

  /** 'x=%d, y=%.2f' % (1, 2.5) */
  percentFormat(fmt: string, args: Value): string {
    const list = args.k === 'tuple' ? args.items : [args]
    let at = 0
    const used = () => {
      if (at >= list.length) throw this.err('TypeError', 'not enough arguments for format string')
      return list[at++]
    }
    const out = fmt.replace(/%(?:\((\w+)\))?([-+ 0#]*)(\*|\d+)?(?:\.(\d+))?([a-zA-Z%])/g, (_m, key: string | undefined, flags: string, width: string | undefined, prec: string | undefined, type: string) => {
      if (type === '%') return '%'
      const v = key !== undefined ? this.getItem({ k: 'sub', obj: { k: 'ellipsis', s: 0, e: 0, line: 0 }, idx: { k: 'ellipsis', s: 0, e: 0, line: 0 }, s: 0, e: 0, line: this.line }, args, str_(key)) : used()
      const w = width === '*' ? String(this.toInt(used())) : width ?? ''
      const left = flags.includes('-')
      if (type === 's' || type === 'r' || type === 'a') {
        let t = type === 's' ? this.str(v) : repr(v)
        if (prec !== undefined) t = t.slice(0, Number(prec))
        return left ? t.padEnd(Number(w)) : t.padStart(Number(w))
      }
      if (type === 'c') return (v.k === 'str' ? v.v : String.fromCharCode(this.toInt(v))).padStart(Number(w))
      if (!'diuoxXeEfFgG'.includes(type)) throw this.err('ValueError', `unsupported format character '${type}'`)
      if (!isNum(v) && !(v.k === 'array' && v.a.size === 1) && !(v.k === 'obj' && v.o.asNumber) && !(v.k === 'inst' && (this.classAttr(v.cls, '__float__') || this.classAttr(v.cls, '__index__')))) throw this.err('TypeError', `%${type} format: a real number is required, not ${typeName(v)}`)
      const integer = 'diuoxX'.includes(type)
      const toInt = integer && v.k === 'inst' ? this.classAttr(v.cls, '__int__') : undefined
      const x = toInt ? this.num(this.call(toInt, [v])) : this.num(v)
      const num: Value = integer ? int(Math.trunc(x)) : float(x)
      const sign = flags.includes('+') ? '+' : flags.includes(' ') ? ' ' : ''
      const spec = `${left ? '<' : ''}${sign}${flags.includes('0') && !left ? '0' : ''}${w}${prec !== undefined && !integer ? `.${prec}` : ''}${type === 'i' || type === 'u' ? 'd' : type}`
      return formatSpec(num, spec)
    })
    if (fmt.replace(/%(?:\((\w+)\))?([-+ 0#]*)(\*|\d+)?(?:\.(\d+))?([a-zA-Z%])/g, '').includes('%')) throw this.err('ValueError', 'incomplete format')
    if (at < list.length && args.k === 'tuple') throw this.err('TypeError', 'not all arguments converted during string formatting')
    if (at === 0 && args.k !== 'tuple' && args.k !== 'dict' && !/%(?!%)/.test(fmt)) throw this.err('TypeError', 'not all arguments converted during string formatting')
    return out
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
    if (op === '<<' || op === '>>') {
      if (isFloat) throw this.err('TypeError', `unsupported operand type(s) for ${op}: '${typeName(l)}' and '${typeName(r)}'`)
      if (y < 0) throw this.err('ValueError', 'negative shift count')
      return int(op === '<<' ? x * 2 ** y : Math.floor(x / 2 ** y))
    }
    // a NumPy float32 scalar stays one beside a Python number
    const f32 = (l.k === 'float' && l.f32) || (r.k === 'float' && r.f32)
    const result = (n: number): Value => (f32 ? fromScalar(n, 'float32') : float(n))
    if (op === '&' || op === '|' || op === '^') {
      if (isFloat) throw this.err('TypeError', `unsupported operand type(s) for ${op}: '${typeName(l)}' and '${typeName(r)}'`)
      const v = applyOp(op, x, y, 'int64')
      return l.k === 'bool' && r.k === 'bool' ? bool(v !== 0) : int(v)
    }
    if (op === '/' || op === '//' || op === '%') {
      if (y === 0) {
        throw this.err('ZeroDivisionError', op === '/' ? (isFloat ? 'float division by zero' : 'division by zero') : isFloat ? 'float modulo by zero' : 'integer division or modulo by zero')
      }
      if (op === '/') return result(x / y)
    }
    if (op === '**' && !isFloat && y < 0) return result(x ** y)
    const v = applyOp(op as BinOp, x, y, isFloat ? 'float64' : 'int64')
    return isFloat ? result(v) : int(v)
  }

  compare(op: string, l: Value, r: Value): Value {
    if (op === 'is' || op === 'is not') {
      const same =
        l.k === r.k &&
        (l.k === 'array' ? l.a === (r as { a: NDArray }).a : l.k === 'obj' ? l.o === (r as { o: PyObj }).o : l.k === 'list' || l.k === 'dict' || l.k === 'set' || l.k === 'inst' || l.k === 'class' || l.k === 'func' ? l === r : l.k === 'none' || l.k === 'ellipsis' || repr(l) === repr(r))
      return bool(op === 'is' ? same : !same)
    }
    if (op === 'in' || op === 'not in') {
      let found: boolean
      if (r.k === 'list' || r.k === 'tuple') found = r.items.some((x) => x === l || this.truthy(this.binop('==', x, l)))
      else if (r.k === 'dict') found = r.d.has(l)
      else if (r.k === 'set') found = r.s.has(l)
      else if (r.k === 'class' && r.meta?.contains) found = r.meta.contains(r, l, this)
      else if (r.k === 'inst') {
        const c = this.classAttr(r.cls, '__contains__')
        found = c ? this.truthy(this.call(c, [r, l])) : this.iterate(r).some((x) => this.equal(x, l))
      } else if (r.k === 'iter') found = this.iterate(r).some((x) => this.equal(x, l))
      else if (r.k === 'range') {
        const x = isNum(l) ? numOf(l) : NaN
        found = Number.isInteger(x) && (r.step > 0 ? x >= r.start && x < r.stop : x <= r.start && x > r.stop) && (x - r.start) % r.step === 0
      } else if (r.k === 'obj' && r.o.contains) found = r.o.contains(l, this)
      else if (r.k === 'str' && this.plainStr(l) !== null) found = r.v.includes(this.plainStr(l)!)
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
          if (isFloatD(a.dtype)) throw this.err('IndexError', 'arrays used as indices must be of integer (or boolean) type')
          return { kind: 'array', arr: a }
        }
        case 'bool': return { kind: 'array', arr: NDArray.create([it.v ? 1 : 0], [], 'bool') }
        case 'obj':
          if (it.o.asIndex) return { kind: 'int', value: it.o.asIndex(this) }
          throw this.err('IndexError', 'only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices')
        case 'inst':
          if (this.classAttr(it.cls, '__index__')) return { kind: 'int', value: this.toInt(it) }
          throw this.err('IndexError', 'only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices')
        default:
          throw this.err('IndexError', 'only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices')
      }
    })
  }

  sliceInt(x: Value): number {
    if (x.k === 'int' || x.k === 'bool') return numOf(x)
    if (x.k === 'obj' && x.o.asIndex) return x.o.asIndex(this)
    if (x.k === 'inst' && this.classAttr(x.cls, '__index__')) return this.toInt(x)
    if (x.k === 'array' && x.a.ndim === 0 && !isFloatD(x.a.dtype)) return x.a.values()[0]
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

  /** np.matrix(data): a list, an array, or text like "1 2; 3 4" */
  makeMatrix(data: Value | undefined, dtype: DType | undefined): Value {
    if (!data) throw this.err('TypeError', "matrix() missing required argument 'data'")
    let src: Value = data
    if (data.k === 'str') {
      const rows = data.v.split(';').map((row) => row.trim().split(/[\s,]+/).filter(Boolean))
      if (rows.some((row) => row.length !== rows[0].length)) throw this.err('ValueError', 'Rows not the same size.')
      const cell = (t: string): Value => {
        const x = Number(t)
        if (t === '' || Number.isNaN(x)) throw this.err('NameError', `name '${t}' is not defined`)
        return /^[-+]?\d+$/.test(t) ? int(x) : float(x)
      }
      src = { k: 'list', items: rows.map((row) => ({ k: 'list', items: row.map(cell) })) }
    }
    const a = toArray(src, dtype)
    if (a.ndim > 2) throw this.err('ValueError', 'shape too large to be a matrix.')
    return arr(matrixArray(a === (src as { a?: NDArray }).a ? a.copy() : a))
  }

  /** m[0] stays a row and m[:, 0] a column, each still 2-D */
  matrixItem(value: NDArray, idx: Value): NDArray {
    const out = value.ndim !== 1 ? plainView(value) : idx.k === 'tuple' && idx.items.length > 1 && isNum(idx.items[1]) ? reshape(value, [value.size, 1]) : reshape(value, [1, value.size])
    out.matrix = true
    return out
  }

  getItem(node: Node & { k: 'sub' }, obj: Value, idx0: Value): Value {
    // anything with __index__ (a 0-d integer tensor, say) indexes a sequence as that integer
    const asIndex = (obj.k === 'list' || obj.k === 'tuple' || obj.k === 'str') && ((idx0.k === 'obj' && idx0.o.asIndex) || (idx0.k === 'inst' && this.classAttr(idx0.cls, '__index__')))
    const idx = asIndex ? int(this.toInt(idx0)) : idx0
    if (obj.k === 'array') {
      const a = obj.a
      const showing = this.tracing && a.size <= TRACE_CELLS
      const before = showing ? snap(a) : null
      const items = this.toIndexItems(idx)
      const { plan, value } = getIndex(a, items)
      const res: Value = typeof value === 'number' ? fromScalar(value, a.dtype) : a.matrix ? arr(this.matrixItem(value, idx)) : arr(value)
      if (showing && before) {
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
      if (v) return v
      if (obj.flavor === 'Counter') return int(0)
      if (obj.flavor === 'defaultdict' && obj.factory && obj.factory.k !== 'none') {
        const made = this.call(obj.factory, [])
        obj.d.set(idx, made)
        return made
      }
      throw this.err('KeyError', repr(idx))
    }
    if (obj.k === 'inst') {
      const f = this.classAttr(obj.cls, '__getitem__')
      if (f) return this.call(f, [obj, idx])
    }
    if (obj.k === 'class' && obj.meta?.getItem) return obj.meta.getItem(obj, idx, this)
    if (obj.k === 'class') {
      const f = this.classAttr(obj, '__class_getitem__')
      if (f) return f.k === 'func' ? this.call(f, [obj, idx]) : this.call(f, [idx])
      if (obj.generic) return obj
      throw this.err('TypeError', `type '${obj.name}' is not subscriptable`)
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
    if (obj.k === 'inst') {
      const f = this.classAttr(obj.cls, '__setitem__')
      if (!f) throw this.err('TypeError', `'${obj.cls.name}' object does not support item assignment`)
      this.call(f, [obj, idx, val])
      return
    }
    if (obj.k === 'list' && idx.k === 'slice') {
      const n = obj.items.length
      const order = this.seqIndex(obj.items.map((_, i) => int(i)), idx, 'list') as Value[]
      const vals = this.iterate(val)
      const step = idx.step.k === 'none' ? 1 : this.toInt(idx.step)
      if (step !== 1) {
        if (vals.length !== order.length) throw this.err('ValueError', `attempt to assign sequence of size ${vals.length} to extended slice of size ${order.length}`)
        order.forEach((p, i) => { obj.items[numOf(p)] = vals[i] })
        return
      }
      const norm = (v: Value, fallback: number) => (v.k === 'none' ? fallback : Math.max(0, Math.min(n, this.toInt(v) < 0 ? this.toInt(v) + n : this.toInt(v))))
      const from = norm(idx.start, 0)
      obj.items.splice(from, Math.max(0, norm(idx.stop, n) - from), ...vals)
      return
    }
    if (obj.k === 'list') {
      const i = idx.k === 'inst' && this.classAttr(idx.cls, '__index__') ? this.toInt(idx) : idx.k === 'int' ? idx.v : null
      if (i === null) throw this.err('TypeError', 'list indices must be integers or slices, not ' + typeName(idx))
      const n = obj.items.length
      if (i < -n || i >= n) throw this.err('IndexError', 'list assignment index out of range')
      obj.items[i < 0 ? i + n : i] = val
      return
    }
    if (obj.k !== 'array') throw this.err('TypeError', `'${typeName(obj)}' object does not support item assignment`)
    const a = obj.a
    if (!this.tracing || a.size > TRACE_CELLS) {
      setIndex(a, this.toIndexItems(idx), this.operand(val, '='))
      return
    }
    const before = snap(a)
    const memBefore = [...a.data]
    const plan = setIndex(a, this.toIndexItems(idx), this.operand(val, '='))
    const written = { shape: plan.shape, values: plan.addresses.map((x) => a.data[x]), dtype: a.dtype }
    this.record(this.targetOf(node), this.text(codeNode ?? node), a, before, plan, 'write', written, memBefore)
  }

  /** 原地增量赋值 b += 1：b 是视图时同样会改到原数组 */
  inplace(name: string, a: NDArray, op: string, rhs: Value, stmt: Span) {
    const res = a.matrix && (op === '**' || (op === '*' && !isNum(rhs))) ? toArray(this.matrixOp(op, arr(a), rhs)) : binaryOp(op as BinOp, plainView(a), this.operand(rhs, op + '='))
    if (!isFloatD(a.dtype) && isFloatD(res.dtype)) {
      throw this.err('UFuncTypeError', `Cannot cast ufunc '${op === '/' ? 'divide' : op}' output from dtype('float64') to dtype('${a.dtype}') with casting rule 'same_kind'`)
    }
    if (res.size !== a.size || (a.matrix && res.shape.some((d, i) => d !== a.shape[i]))) {
      throw this.err('ValueError', `non-broadcastable output operand with shape ${shapeStr(a.shape)} doesn't match the broadcast shape ${shapeStr(res.shape)}`)
    }
    const showing = this.tracing && a.size <= TRACE_CELLS
    const before = showing ? snap(a) : null
    const memBefore = showing ? [...a.data] : undefined
    const addrs = a.addresses()
    const vals = res.values()
    addrs.forEach((x, i) => { a.data[x] = castValue(vals[i], a.dtype) })
    if (!before) return
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
    const grouped = this.pendingGroups
    this.pendingGroups = null
    if (tooBigToTrace(res) || ops.some((o) => tooBigToTrace(o.v))) return
    this.calls.push({
      id: this.calls.length,
      seq: this.seqN++,
      line: this.line,
      code,
      api,
      kind,
      operands: ops.map((o) => ({ label: o.label, snap: o.v.k === 'array' ? snap(o.v.a) : this.resultSnap(o.v)! })),
      axis,
      result: grouped ? snap(grouped.result) : this.resultSnap(res),
      resultText: repr(res),
      source,
      plot,
      groups: grouped?.groups,
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
    if ((self && self.size > TRACE_CELLS) || args.some(tooBigToTrace)) return f.call(args, kw)
    const selfNode = self && n.fn.k === 'attr' ? n.fn.obj : null
    // after a *x the argument nodes no longer line up with the values
    const argNodes = n.args.some((x) => x.k === 'star') ? [] : n.args
    const nodes = selfNode ? [selfNode, ...argNodes] : argNodes
    const vals = self ? [arr(self), ...args] : args
    const ops = this.collectOperands(nodes, vals, meta.kind === 'elementwise')
    // 先拍快照：调用本身可能改动输入（如原地操作）
    const opSnaps = ops.map((o) => ({ label: o.label, v: o.v.k === 'array' ? arr(o.v.a.copy()) : o.v }))
    this.pendingGroups = null
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
      case 'num': return n.imag ? { k: 'obj', o: new ImagLiteral(n.v) } : n.isFloat ? float(n.v) : int(n.v)
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
      case 'list': return { k: 'list', items: this.evalItems(n.items) }
      case 'tuple': return tuple(this.evalItems(n.items))
      case 'set': return makeSet(this.evalItems(n.items))
      case 'star': throw this.err('SyntaxError', "can't use starred expression here")
      case 'walrus': {
        const v = this.eval(n.x)
        this.bind(n.id, v)
        return v
      }
      case 'dict': {
        const d = new PyDict()
        n.keys.forEach((key, i) => {
          if (key) {
            d.set(this.eval(key), this.eval(n.vals[i]))
            return
          }
          const m = this.eval(n.vals[i])
          if (m.k !== 'dict') throw this.err('TypeError', `'${typeName(m)}' object is not a mapping`)
          for (const [a, b] of m.d.items()) d.set(a, b)
        })
        return { k: 'dict', d }
      }
      case 'fstr':
        return str_(n.parts.map((p) => {
          if (typeof p === 'string') return p
          const v = this.eval(p.x)
          const spec = p.specX ? this.str(this.eval(p.specX)) : p.spec
          const shown = p.conv === 'r' || p.conv === 'a' ? str_(repr(v)) : p.conv === 's' ? str_(this.str(v)) : v
          const own = this.instFormat(shown, spec)
          if (own !== undefined) return own
          if (spec) return formatSpec(shown, spec)
          if (shown !== v) return (shown as { v: string }).v
          // a library object or a NumPy scalar has a say in its own format, even with no spec
          return v.k === 'obj' || (v.k === 'float' && v.f32) ? formatSpec(v, '') : this.str(v)
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
          const res = x.a.matrix ? toMatrix(arr(unaryOp(n.op, x.a))) : arr(unaryOp(n.op, x.a))
          if (this.tracing && n.op !== '+') this.pushCall(`op:${n.op}x`, 'elementwise', this.text(n), [{ label: this.text(n.x), v: arr(x.a.copy()) }], res, null)
          return res
        }
        if (x.k === 'inst') {
          const f = this.classAttr(x.cls, n.op === '-' ? '__neg__' : n.op === '+' ? '__pos__' : '__invert__')
          if (f) return this.call(f, [x])
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
          if ((l.k === 'array' || r.k === 'array') && this.tracing && !tooBigToTrace(l) && !tooBigToTrace(r)) {
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
        if ((l.k === 'array' || r.k === 'array') && this.tracing && !tooBigToTrace(l) && !tooBigToTrace(r)) {
          const ops = this.collectOperands([n.l, n.r], [l, r], n.op !== '@' && !(n.op === '*' && isMatrix(l) && isMatrix(r))).map((o) => ({ label: o.label, v: o.v.k === 'array' ? arr(o.v.a.copy()) : o.v }))
          this.pushCall(`op:${n.op}`, n.op === '@' || (n.op === '*' && (isMatrix(l) || isMatrix(r)) && !isNum(l) && !isNum(r)) ? 'matmul' : 'elementwise', this.text(n), ops, res, null)
        }
        return res
      }
      case 'attr': {
        const obj = this.eval(n.obj)
        const res = this.getAttr(obj, n.name)
        if (obj.k === 'array' && n.name === 'T' && this.tracing && !tooBigToTrace(obj)) {
          const ops = [{ label: this.text(n.obj), v: arr(obj.a.copy()) }]
          const source = this.replayMove([{ v: obj }], (subst) => this.getAttr(subst(obj), 'T'))
          this.pushCall('ndarray.T', 'move', this.text(n), ops, res, null, source)
        }
        return res
      }
      case 'call': {
        const f = this.eval(n.fn)
        const args = this.evalItems(n.args)
        const kw: Record<string, Value> = {}
        for (const { name, v } of n.kw) {
          if (name !== null) {
            kw[name] = this.eval(v)
            continue
          }
          const m = this.eval(v)
          if (m.k !== 'dict') throw this.err('TypeError', `argument after ** must be a mapping, not ${typeName(m)}`)
          for (const [a, b] of m.d.items()) {
            if (a.k !== 'str') throw this.err('TypeError', 'keywords must be strings')
            kw[a.v] = b
          }
        }
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

  /** f'{x:spec}' for an instance of a class with __format__ */
  instFormat(v: Value, spec: string): string | undefined {
    const f = v.k === 'inst' ? this.classAttr(v.cls, '__format__') : undefined
    return f ? this.str(this.call(f, [v, str_(spec)])) : undefined
  }

  /** the elements of a display or call, spreading any *x */
  evalItems(nodes: Node[]): Value[] {
    const out: Value[] = []
    for (const x of nodes) {
      if (x.k === 'star') out.push(...this.iterate(this.eval(x.x)))
      else out.push(this.eval(x))
    }
    return out
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
      const star = t.items.findIndex((x) => x.k === 'star')
      if (star >= 0) {
        const after = t.items.length - star - 1
        if (items.length < star + after) throw this.err('ValueError', `not enough values to unpack (expected at least ${star + after}, got ${items.length})`)
        t.items.forEach((x, i) => {
          if (i < star) this.assign(x, items[i])
          else if (i === star) this.assign((x as { x: Node }).x, { k: 'list', items: items.slice(star, items.length - after) })
          else this.assign(x, items[items.length - (t.items.length - i)])
        })
        return
      }
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
    if (v.k === 'set') return setItems(v)
    if (v.k === 'iter' || v.k === 'inst') {
      const next = this.iterOf(v)
      const out: Value[] = []
      for (let x = next(); x !== undefined; x = next()) {
        this.tick()
        out.push(x)
        if (out.length > 5_000_000) throw this.err('MemoryError', 'too many values for this sandbox')
      }
      return out
    }
    if (v.k === 'range') {
      const n = rangeLen(v)
      if (n > 1_000_000) throw this.err('MemoryError', 'range too large for this sandbox')
      return Array.from({ length: n }, (_, i) => int(v.start + i * v.step))
    }
    if (v.k === 'obj' && v.o.iter) return v.o.iter(this)
    if (v.k === 'class' && v.meta?.iter) return v.meta.iter(v, this)
    if (v.k === 'array') {
      if (v.a.ndim === 0) throw this.err('TypeError', 'iteration over a 0-d array')
      return Array.from({ length: v.a.shape[0] }, (_, i) => {
        const { value } = getIndex(v.a, [{ kind: 'int', value: i }])
        if (typeof value === 'number') return fromScalar(value, v.a.dtype)
        return v.a.matrix ? arr(matrixArray(value)) : arr(value)
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
        } else if (b.k === 'type' && !skipping) {
          const v = b.methods?.get(name)
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
      case 'descr': if (f.kind === 'static') return this.call(f.f, args, kw); break
      case 'class': return f.meta?.call ? f.meta.call(f, args, kw, this) : this.instantiate(f, args, kw)
      case 'inst': {
        const c = this.classAttr(f.cls, '__call__')
        if (c) return this.call(c, [f, ...args], kw)
        if (f.cls.host?.call) return f.cls.host.call(f, args, kw, this)
        break
      }
      case 'obj':
        if (f.o.call) return f.o.call(args, kw, this)
        break
    }
    throw this.err('TypeError', `'${typeName(f)}' object is not callable`)
  }

  /** match a call's arguments to the function's parameters, with Python's messages */
  bindArgs(f: FuncValue, args: Value[], kw: Kw): Map<string, Value> {
    const vars = new Map<string, Value>()
    const ps = f.params
    const pos = ps.filter((p) => !p.kind)
    const star = ps.find((p) => p.kind === 'star')
    const kwstar = ps.find((p) => p.kind === 'kwstar')
    if (args.length > pos.length && !star) {
      const required = pos.filter((p) => !p.def).length
      const takes = required === pos.length ? String(pos.length) : `from ${required} to ${pos.length}`
      throw this.err('TypeError', `${f.name}() takes ${takes} positional argument${pos.length === 1 && required === 1 ? '' : 's'} but ${args.length} ${args.length === 1 ? 'was' : 'were'} given`)
    }
    ps.forEach((p, i) => {
      if (p.kind === 'star' || p.kind === 'kwstar') return
      const at = p.kind ? -1 : pos.indexOf(p)
      if (at >= 0 && at < args.length) {
        if (p.name in kw) throw this.err('TypeError', `${f.name}() got multiple values for argument '${p.name}'`)
        vars.set(p.name, args[at])
      } else if (p.name in kw) vars.set(p.name, kw[p.name])
      else if (f.defaults[i]) vars.set(p.name, f.defaults[i]!)
    })
    if (star) vars.set(star.name, tuple(args.slice(pos.length)))
    const named = new Set(ps.filter((p) => p.kind !== 'star' && p.kind !== 'kwstar').map((p) => p.name))
    const extra = Object.keys(kw).filter((k) => !named.has(k))
    if (kwstar) vars.set(kwstar.name, { k: 'dict', d: PyDict.from(extra.map((k) => [str_(k), kw[k]])) })
    else if (extra.length) throw this.err('TypeError', `${f.name}() got an unexpected keyword argument '${extra[0]}'`)
    const list = (names: string[]) => (names.length > 1 ? `${names.slice(0, -1).join(', ')}${names.length > 2 ? ',' : ''} and ${names[names.length - 1]}` : names[0])
    const missing = pos.filter((p) => !vars.has(p.name)).map((p) => `'${p.name}'`)
    if (missing.length) throw this.err('TypeError', `${f.name}() missing ${missing.length} required positional argument${missing.length > 1 ? 's' : ''}: ${list(missing)}`)
    const missingKw = ps.filter((p) => p.kind === 'kwonly' && !vars.has(p.name)).map((p) => `'${p.name}'`)
    if (missingKw.length) throw this.err('TypeError', `${f.name}() missing ${missingKw.length} required keyword-only argument${missingKw.length > 1 ? 's' : ''}: ${list(missingKw)}`)
    return vars
  }

  callFunc(f: FuncValue, args: Value[], kw: Kw): Value {
    const vars = this.bindArgs(f, args, kw)
    const ps = f.params
    if (f.isGen) return this.makeGenerator(f, { vars, parent: f.closure, owner: f.owner, self: ps.length ? vars.get(ps[0].name) : undefined })
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

  /** the methods a class derived from ABC has not written yet */
  abstractsOf(cls: ClassValue): string[] {
    const order: ClassValue[] = []
    const walk = (c: Value) => {
      if (c.k === 'class' && !order.includes(c)) {
        order.push(c)
        c.bases.forEach(walk)
      }
    }
    walk(cls)
    if (!order.some((c) => c.abstractBase)) return []
    const found = new Set<string>()
    const out: string[] = []
    for (const c of order) {
      for (const [n, v] of c.ns) {
        if (found.has(n)) continue
        found.add(n)
        const f = v.k === 'descr' ? v.f : v
        if (f.k === 'func' && f.abstract) out.push(n)
      }
    }
    return out.sort()
  }

  instantiate(cls: ClassValue, args: Value[], kw: Kw): Value {
    const missing = (cls.abstracts ??= this.abstractsOf(cls))
    if (missing.length) throw this.err('TypeError', `Can't instantiate abstract class ${cls.name} with abstract method${missing.length > 1 ? 's' : ''} ${missing.join(', ')}`)
    const inst: InstValue = { k: 'inst', cls, attrs: new Map(), state: {} }
    // an exception keeps its arguments even when its own __init__ does not pass them on
    if (cls.host === EXC_HOST) EXC_HOST.init(inst, args, kw, this)
    const init = this.classAttr(cls, '__init__')
    if (init) {
      const r = this.call(init, [inst, ...args], kw)
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
      case 'inst': {
        const c = this.classAttr(obj.cls, name)
        if (c?.k === 'descr' && c.kind === 'prop') {
          if (!c.setter) throw this.err('AttributeError', `property '${name}' of '${obj.cls.name}' object has no setter`)
          this.call(c.setter, [obj, v])
          return
        }
        obj.cls.host?.setAttr?.(obj, name, v, this)
        obj.attrs.set(name, v)
        return
      }
      case 'class':
        if (!obj.meta?.setAttr?.(obj, name, v, this)) obj.ns.set(name, v)
        return
      case 'obj':
        if (obj.o.setAttr?.(name, v, this)) return
        throw this.err('AttributeError', `'${obj.o.cls}' object attribute '${name}' is read-only (or does not exist in this sandbox)`)
      case 'func': case 'fn':
        ;(obj.attrs ??= new Map()).set(name, v)
        return
      case 'module':
        obj.attrs[name] = v
        return
    }
    throw this.err('AttributeError', `'${typeName(obj)}' object has no attribute '${name}'`)
  }

  /** the items of a generator expression, one at a time */
  *compItems(n: Node & { k: 'comp' }, g: number, first?: () => Value | undefined): Generator<Value, void, undefined> {
    const gen = n.gens[g]
    const next = first ?? this.iterOf(this.eval(gen.iter))
    for (let item = next(); item !== undefined; item = next()) {
      this.tick()
      this.assign(gen.target, item)
      if (!gen.conds.every((c) => this.truthy(this.eval(c)))) continue
      if (g === n.gens.length - 1) yield this.eval(n.elt)
      else yield* this.compItems(n, g + 1)
    }
  }

  comprehension(n: Node & { k: 'comp' }): Value {
    if (n.kind === 'gen') {
      // as in Python, the first iterable is evaluated now and the rest as items are asked for
      const first = this.iterOf(this.eval(n.gens[0].iter))
      const frame: Frame = { vars: new Map(), parent: this.frame }
      const g = this.compItems(n, 0, first)
      return {
        k: 'iter', name: 'generator',
        next: () => {
          const saved = this.frame
          this.frame = frame
          try {
            const r = g.next()
            return r.done ? undefined : r.value
          } finally {
            this.frame = saved
          }
        },
      }
    }
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
    if (n.kind === 'set') return makeSet(out)
    return n.kind === 'dict' ? { k: 'dict', d: PyDict.from(pairs) } : { k: 'list', items: out }
  }

  // ---------- 语句 ----------

  /** run a program: the value of a final expression statement is the cell's output */
  run(stmts: Stmt[]): Value | null {
    const first = stmts[0]
    if (first?.k === 'expr' && first.x.k === 'str') this.builtins.__doc__ = str_(first.x.v)
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
          if (name === '*') {
            for (const a in mod.attrs) if (!a.startsWith('_')) this.bind(a, mod.attrs[a])
            continue
          }
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
        const next = this.iterOf(this.eval(st.iter))
        for (let item = next(); item !== undefined; item = next()) {
          this.line = st.line
          this.tick()
          this.assign(st.target, item)
          try {
            this.execBlock(st.body)
          } catch (e) {
            if (e instanceof BreakSig) return
            if (e instanceof ContinueSig) continue
            throw e
          }
        }
        if (st.orelse) this.execBlock(st.orelse)
        return
      }
      case 'while':
        while (this.truthy(this.eval(st.cond))) {
          try {
            this.execBlock(st.body)
          } catch (e) {
            if (e instanceof BreakSig) return
            if (e instanceof ContinueSig) continue
            throw e
          }
          this.at(st.line)
        }
        if (st.orelse) this.execBlock(st.orelse)
        return
      case 'seq':
        this.execBlock(st.body)
        return
      case 'annot': {
        if (this.frame?.isClass) {
          const ns = this.frame.vars
          let notes = ns.get('__annotations__')
          if (!notes) ns.set('__annotations__', (notes = { k: 'dict', d: new PyDict() }))
          let note: Value = NONE
          try {
            if (st.note) note = this.eval(st.note)
          } catch (e) {
            if (!(e instanceof PyError)) throw e
          }
          ;(notes as { d: PyDict }).d.set(str_(st.name), note)
        }
        if (st.x) this.bind(st.name, this.eval(st.x))
        return
      }
      case 'global':
        if (this.frame) for (const n of st.names) (this.frame.globals ??= new Set()).add(n)
        return
      case 'nonlocal':
        if (!this.frame) throw this.err('SyntaxError', 'nonlocal declaration not allowed at module level')
        for (const n of st.names) (this.frame.nonlocals ??= new Set()).add(n)
        return
      case 'yield':
        throw this.err('SyntaxError', "'yield' outside function")
      case 'raise': {
        if (!st.x) {
          const cur = this.handling[this.handling.length - 1]
          if (!cur) throw this.err('RuntimeError', 'No active exception to reraise')
          throw cur
        }
        const x = this.eval(st.x)
        const inst = x.k === 'class' && this.isSubclass(x, this.excClass('BaseException')) ? this.instantiate(x, [], {}) : x
        if (!this.isException(inst)) throw this.err('TypeError', 'exceptions must derive from BaseException')
        this.line = st.line
        if (st.cause) inst.attrs.set('__cause__', this.eval(st.cause))
        return this.throwExc(inst)
      }
      case 'try':
        try {
          try {
            this.execBlock(st.body)
          } catch (e) {
            const err = this.asPyError(e)
            const h = err && this.findHandler(st, err)
            if (!err || !h) throw e
            this.enterHandler(h, err)
            try {
              this.execBlock(h.body)
            } finally {
              this.leaveHandler(h)
            }
            return
          }
          if (st.orelse) this.execBlock(st.orelse)
        } finally {
          if (st.final) this.execBlock(st.final)
        }
        return
      case 'def': {
        const decorators = st.decorators.map((d) => this.eval(d))
        const doc = st.body[0]?.k === 'expr' && st.body[0].x.k === 'str' ? st.body[0].x.v : undefined
        let f: Value = {
          k: 'func', name: st.name, params: st.params, defaults: st.params.map((p) => (p.def ? this.eval(p.def) : null)),
          body: st.body, expr: null, closure: this.closureFrame(), owner: null, isGen: st.isGen, doc, notes: this.annotations(st.params, st.returns),
        }
        for (const d of decorators.reverse()) {
          this.line = st.line
          f = this.call(d, [f])
        }
        this.bind(st.name, f)
        return
      }
      case 'class': {
        const written = st.bases.map((b) => this.eval(b))
        const maker = written.find((b) => b.k === 'obj' && b.o.classFromBody) as (Value & { k: 'obj' }) | undefined
        const bases = written.flatMap((b) => (b.k === 'obj' && b.o.mroEntries ? b.o.mroEntries() : [b]))
        const meta = bases.flatMap((b) => (b.k === 'class' && b.meta ? [b.meta] : []))[0]
        for (const b of bases) {
          if (b.k !== 'class' && b.k !== 'type') throw this.err('TypeError', `bases must be classes, not '${typeName(b)}'`)
          // a metaclass can mix a built-in type in: class Level(int, Enum)
          if (b.k === 'type' && !b.host && b.name !== 'object' && !meta) throw this.err('TypeError', `cannot subclass '${b.name}' in this sandbox`)
        }
        const cls: ClassValue = { k: 'class', name: st.name, bases, ns: meta?.prepare?.(st.name, bases, this) ?? new Map(), host: this.hostOf(bases), ...(meta ? { meta } : {}) }
        if (st.body[0]?.k === 'expr' && st.body[0].x.k === 'str') cls.ns.set('__doc__', str_(st.body[0].x.v))
        const saved = this.frame
        this.frame = { vars: cls.ns, parent: saved, isClass: true }
        try {
          this.execBlock(st.body)
        } finally {
          this.frame = saved
        }
        for (const v of cls.ns.values()) {
          if (v.k === 'func') v.owner = cls
          else if (v.k === 'descr' && v.f.k === 'func') v.f.owner = cls
        }
        meta?.created?.(cls, this)
        // a base that vanished was Generic[T] or Protocol: X[int] is X
        if (written.length !== bases.length || bases.some((b) => b.k === 'class' && b.generic)) cls.generic = true
        let c: Value = maker ? maker.o.classFromBody!(st.name, cls.ns, this) : cls
        for (const d of st.decorators.map((x) => this.eval(x)).reverse()) c = this.call(d, [c])
        this.bind(st.name, c)
        return
      }
      case 'with': {
        const enter = (i: number) => {
          if (i === st.items.length) {
            this.execBlock(st.body)
            return
          }
          const item = st.items[i]
          const ctx = this.eval(item.x)
          this.callText = this.text(item.x)
          const { entered, exit } = this.enterContext(ctx)
          if (item.as) this.assign(item.as, entered)
          try {
            enter(i + 1)
          } catch (e) {
            const err = this.asPyError(e)
            if (exit(err) && err) return
            throw e
          }
          exit(null)
        }
        enter(0)
        return
      }
    }
  }

  /** def f(x: int) -> str: a note the sandbox cannot evaluate (list[int]) keeps its source text */
  annotations(params: Param[], returns: Node | null): Map<string, Value> | undefined {
    const named: [string, Node][] = params.flatMap((p): [string, Node][] => (p.note ? [[p.name, p.note]] : []))
    if (returns) named.push(['return', returns])
    if (!named.length) return undefined
    this.quiet++
    try {
      return new Map(named.map(([name, n]): [string, Value] => {
        try {
          return [name, this.eval(n)]
        } catch (e) {
          if (!(e instanceof PyError)) throw e
          return [name, { k: 'type', name: this.text(n), call: () => NONE }]
        }
      }))
    } finally {
      this.quiet--
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
      else if (cur.k === 'set' && !cur.frozen && ['|', '&', '-', '^'].includes(st.op) && rhs.k === 'set') {
        const res = this.setOp(st.op, cur, rhs) as Value & { k: 'set' }
        cur.s = res.s
      } else if (cur.k === 'inst' && this.classAttr(cur.cls, `__i${(DUNDER[st.op] ?? '').slice(2)}`)) this.bind(t.id, this.call(this.classAttr(cur.cls, `__i${DUNDER[st.op].slice(2)}`)!, [cur, rhs]))
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
      if (obj.k === 'array' && v.k === 'array' && !isFloatD(obj.a.dtype) && isFloatD(v.a.dtype)) {
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
      } else if (obj.k === 'inst' && this.classAttr(obj.cls, '__delitem__')) {
        this.call(this.classAttr(obj.cls, '__delitem__')!, [obj, idx])
      } else if (obj.k === 'list' && idx.k === 'slice') {
        const gone = new Set((this.seqIndex(obj.items.map((_, i) => int(i)), idx, 'list') as Value[]).map(numOf))
        obj.items.splice(0, obj.items.length, ...obj.items.filter((_, i) => !gone.has(i)))
      } else if (obj.k === 'list' && (idx.k === 'int' || idx.k === 'bool')) {
        const n = obj.items.length
        const i = numOf(idx)
        if (i < -n || i >= n) throw this.err('IndexError', 'list assignment index out of range')
        obj.items.splice(i < 0 ? i + n : i, 1)
      } else if (obj.k === 'obj' && obj.o.delItem) obj.o.delItem(idx, this)
      else throw this.err('TypeError', `'${typeName(obj)}' object doesn't support item deletion`)
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
      const m: Record<string, DType> = { uint8: 'uint8', int: 'int64', int64: 'int64', int32: 'int64', float: 'float64', float64: 'float64', float32: 'float32', single: 'float32', f4: 'float32', f8: 'float64', u1: 'uint8', bool: 'bool' }
      if (m[v.v]) return m[v.v]
      throw this.err('TypeError', `data type '${v.v}' not understood`)
    }
    throw this.err('TypeError', `Cannot interpret '${repr(v)}' as a data type`)
  }

  reduceFn(a: NDArray, kind: ReduceKind, args: Value[], kw: Record<string, Value>): Value {
    const axisV = kw.axis ?? args[0]
    const keep = !!kw.keepdims && this.truthy(kw.keepdims)
    // several axes at once: run them together as one long axis
    if (axisV?.k === 'tuple') {
      const axes = axisV.items.map((x) => normAxis(this.toInt(x), a.ndim))
      const rest = a.shape.map((_, i) => i).filter((i) => !axes.includes(i))
      const merged = reshape(transpose(a, [...rest, ...axes]).copy(), [...rest.map((i) => a.shape[i]), -1])
      const sub: Kw = { ...kw, axis: int(-1) }
      delete sub.keepdims
      if (kw.where) sub.where = arr(reshape(transpose(broadcastTo(toArray(kw.where), a.shape), [...rest, ...axes]).copy(), merged.shape))
      const res = this.reduceFn(merged, kind, [], sub)
      if (!keep) return res
      const shape = a.shape.map((d, i) => (axes.includes(i) ? 1 : d))
      return arr(reshape(toArray(res), shape))
    }
    const axis = this.kwInt(kw, args, 0, 'axis')
    const hasInitial = !!kw.initial && kw.initial.k !== 'none'
    if (kw.where || hasInitial) return this.reduceMasked(a, kind, axis, kw, keep)
    const ddof = kw.ddof ? this.num(kw.ddof) : 0
    let res = reduce(a, kind, axis)
    if (ddof && (kind === 'std' || kind === 'var')) {
      // the variance with a different divisor; std is its root, as in NumPy
      const n = axis === null ? a.size : a.shape[normAxis(axis, a.ndim)]
      const corrected = (v: number) => (kind === 'var' ? (v * n) / (n - ddof) : Math.sqrt((v * n) / (n - ddof)))
      const variance = reduce(a, 'var', axis)
      res = typeof variance === 'number' ? corrected(variance) : NDArray.create(variance.values().map(corrected), variance.shape, reduceDType(kind, a.dtype))
    }
    // keepdims=True leaves the reduced axis in place with length 1, so the result still broadcasts against a.
    // A matrix does this by itself along an axis: m.sum(axis=1) is a column
    if (keep || (a.matrix && axis !== null)) {
      const shape = axis === null ? a.shape.map(() => 1) : a.shape.map((d, i) => (i === normAxis(axis, a.ndim) ? 1 : d))
      const vals = typeof res === 'number' ? [res] : res.values()
      return arr(NDArray.create(vals, shape, typeof res === 'number' ? reduceDType(kind, a.dtype) : res.dtype))
    }
    if (typeof res === 'number') return fromScalar(res, reduceDType(kind, a.dtype))
    return arr(res)
  }

  /** np.sum(a, where=mask) and np.max(a, initial=0): each lane is cut down by the mask, then reduced as usual */
  reduceMasked(a: NDArray, kind: ReduceKind, axis: number | null, kw: Kw, keep: boolean): Value {
    const mask = kw.where ? broadcastTo(toArray(kw.where), a.shape) : null
    const initial = kw.initial && kw.initial.k !== 'none' ? this.num(kw.initial) : null
    const dtype = reduceDType(kind, a.dtype)
    const one = (xs: number[], on: number[] | null): number => {
      const vals = on ? xs.filter((_, i) => on[i]) : [...xs]
      if (initial !== null) vals.push(initial)
      if (!vals.length) {
        if (kind === 'sum') return 0
        if (kind === 'prod') return 1
        if (kind === 'any') return 0
        if (kind === 'all') return 1
        if (kind === 'mean' || kind === 'std' || kind === 'var') return NaN
        throw this.err('ValueError', `zero-size array to reduction operation ${kind} which has no identity`)
      }
      return reduce(NDArray.create(vals, [vals.length], a.dtype), kind, null) as number
    }
    if (axis === null) {
      const r = one(a.values(), mask ? mask.values() : null)
      return keep ? arr(NDArray.create([r], a.shape.map(() => 1), dtype)) : fromScalar(r, dtype)
    }
    const ax = normAxis(axis, a.ndim)
    const order = [...a.shape.map((_, i) => i).filter((i) => i !== ax), ax]
    const moved = transpose(a, order).values()
    const movedMask = mask ? transpose(mask, order).values() : null
    const n = a.shape[ax]
    const outer = order.slice(0, -1).map((i) => a.shape[i])
    const out = Array.from({ length: outer.reduce((p, x) => p * x, 1) }, (_, i) => one(moved.slice(i * n, i * n + n), movedMask ? movedMask.slice(i * n, i * n + n) : null))
    if (keep) return arr(NDArray.create(out, a.shape.map((d, i) => (i === ax ? 1 : d)), dtype))
    return outer.length ? arr(NDArray.create(out, outer, dtype)) : fromScalar(out[0], dtype)
  }

  /** ndarray 方法表：impl(self) 返回可调用对象，便于在元素编号数组上重放 */
  arrayMethod(name: string): { impl: (a: NDArray) => Fn; meta: ApiMeta } | null {
    const M = (kind: ApiKind, impl: (a: NDArray) => Fn, extra: Partial<ApiMeta> = {}) => ({
      impl: (a: NDArray): Fn => (a.matrix ? (args, kw) => toMatrix(impl(a)(args, kw)) : impl(a)),
      meta: { name: `ndarray.${name}`, kind, ...extra },
    })
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
      case 'sort': return M('sort', (a) => (args, kw) => {
        const sorted = sortAlong(a, this.kwInt(kw, args, 0, 'axis') ?? -1).sorted
        const ad = a.addresses()
        sorted.values().forEach((x, i) => { a.data[ad[i]] = x })
        return NONE
      })
      // a full sort is one valid partition; NumPy's arrangement of the two sides differs and is not promised
      case 'partition': return M('sort', (a) => (args, kw) => {
        const sorted = sortAlong(a, kw.axis ? this.toInt(kw.axis) : args[1] ? this.toInt(args[1]) : -1).sorted
        const ad = a.addresses()
        sorted.values().forEach((x, i) => { a.data[ad[i]] = x })
        return NONE
      })
      case 'argpartition': return M('sort', (a) => (args, kw) => arr(sortAlong(a, kw.axis ? this.toInt(kw.axis) : args[1] ? this.toInt(args[1]) : -1).order))
      case 'fill': return M('elementwise', (a) => (args) => {
        const x = castValue(this.num(args[0]), a.dtype)
        for (const at of a.addresses()) a.data[at] = x
        return NONE
      })
      case 'put': return M('move', (a) => (args) => {
        const ind = toArray(args[0]).values()
        const vals = toArray(args[1]).values()
        const ad = a.addresses()
        const n = a.size
        ind.forEach((i, k) => {
          if (i < -n || i >= n) throw this.err('IndexError', `index ${i} is out of bounds for axis 0 with size ${n}`)
          a.data[ad[i < 0 ? i + n : i]] = castValue(vals[k % vals.length], a.dtype)
        })
        return NONE
      })
      case 'resize': return M('move', (a) => (args) => {
        // in place, padding with zeros (np.resize instead repeats the data)
        if (a.base) throw this.err('ValueError', 'cannot resize this array: it does not own its data')
        const shape = this.toShape(args)
        checkSize(shape)
        const old = a.values()
        a.data = Array.from({ length: shape.reduce((p, x) => p * x, 1) }, (_, i) => old[i] ?? 0)
        a.shape = shape
        a.strides = cStrides(shape)
        a.offset = 0
        return NONE
      })
      case 'compress': return M('move', (a) => (args, kw) => this.call((this.np as { attrs: Record<string, Value> }).attrs.compress, [args[0], arr(a), ...args.slice(1)], kw))
      case 'view': return M('move', (a) => () => arr(new NDArray(a.data, a.shape, a.strides, a.offset, a.dtype, a.base ?? a)))
      case 'trace': return M('info', (a) => () => fromScalar(trace(a), a.dtype === 'bool' ? 'int64' : a.dtype))
      case 'item': return M('info', (a) => (args) => {
        // a Python number: a float32 shows all its digits, as float() would
        const asPython = (x: number) => (a.dtype === 'float32' ? float(x) : fromScalar(x, a.dtype))
        if (args[0]) return asPython(a.values()[this.toInt(args[0])])
        if (a.size !== 1) throw this.err('ValueError', 'can only convert an array of size 1 to a Python scalar')
        return asPython(a.values()[0])
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
    const res = NDArray.create(av.map((x, i) => (Number.isNaN(x) || Number.isNaN(bv[i]) ? NaN : pick(x, bv[i]))), shape, pairDType(a, b))
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
        if (name === '__class__') return this.call(this.builtins.type, [obj])
        throw this.err('AttributeError', `'${obj.o.cls}' object has no attribute '${name}'${obj.o.getAttr ? ' (or it is not available in this sandbox)' : ''}`)
      }
      case 'inst': {
        const own = obj.attrs.get(name)
        if (own) return own
        const c = this.classAttr(obj.cls, name)
        if (c) return this.bindAttr(c, obj)
        const h = obj.cls.host?.getAttr?.(obj, name, this)
        if (h) return h
        if (name === '__class__') return obj.cls
        if (name === '__dict__') return { k: 'dict', d: new AttrDict(obj.attrs) }
        if (name === '__doc__') return NONE
        if (name === '__cause__' || name === '__traceback__') return NONE
        if (name === 'with_traceback' && this.isException(obj)) return fn(() => obj)
        const ga = this.classAttr(obj.cls, '__getattr__')
        if (ga) return this.call(ga, [obj, str_(name)])
        throw this.err('AttributeError', `'${obj.cls.name}' object has no attribute '${name}'`)
      }
      case 'class': {
        if (name === '__name__' || name === '__qualname__') return str_(obj.name)
        if (name === '__module__') return str_(obj.module ?? (obj.builtin ? 'builtins' : '__main__'))
        if (name === '__bases__') return tuple(obj.bases)
        if (name === '__doc__') return this.classAttr(obj, '__doc__') ?? NONE
        if (name === '__dict__') return { k: 'dict', d: PyDict.from([...obj.ns].map(([n, v]) => [str_(n), v])) }
        if (name === 'mro' || name === '__mro__') {
          const order: Value[] = []
          const walk = (c: Value) => {
            if (!order.includes(c)) order.push(c)
            if (c.k === 'class') c.bases.forEach(walk)
          }
          walk(obj)
          if (!order.includes(this.builtins.object)) order.push(this.builtins.object)
          return name === 'mro' ? fn(() => ({ k: 'list', items: order })) : tuple(order)
        }
        const c = this.classAttr(obj, name)
        if (c?.k === 'descr') {
          if (c.kind === 'static') return c.f
          if (c.kind === 'class') return this.bindAttr(c, obj)
          return c
        }
        if (c) return c
        const m = obj.meta?.getAttr?.(obj, name, this)
        if (m) return m
        throw this.err('AttributeError', `type object '${obj.name}' has no attribute '${name}'`)
      }
      case 'descr':
        if (obj.kind === 'prop' && (name === 'setter' || name === 'getter' || name === 'deleter')) {
          return fn((args) => (name === 'setter' ? { ...obj, setter: args[0] } : name === 'getter' ? { ...obj, f: args[0] } : obj))
        }
        break
      case 'iter':
        if (name === '__next__') return fn(() => {
          const r = obj.next()
          if (r === undefined) this.raise(this.excClass('StopIteration'), [])
          return r as Value
        })
        if (name === '__iter__') return fn(() => obj)
        if (name === 'send' && obj.send) return fn((args) => {
          const r = obj.send!(args[0])
          if (r === undefined) this.raise(this.excClass('StopIteration'), obj.result ? [obj.result] : [])
          return r as Value
        })
        if (name === 'close' && obj.close) return fn(() => { obj.close!(); return NONE })
        if (name === 'throw' && obj.throw) return fn((args) => {
          const x = args[1] && args[1].k !== 'none' ? args[1] : args[0]
          const inst = x.k === 'class' ? this.instantiate(x, args[1]?.k === 'none' || !args[1] ? [] : [args[1]], {}) : x
          if (!this.isException(inst)) throw this.err('TypeError', `exceptions must be classes or instances deriving from BaseException, not ${typeName(inst)}`)
          let thrown: unknown
          try {
            this.throwExc(inst)
          } catch (e) {
            thrown = e
          }
          const r = obj.throw!(thrown)
          if (r === undefined) this.raise(this.excClass('StopIteration'), obj.result ? [obj.result] : [])
          return r as Value
        })
        break
      case 'super': {
        const c = this.classAttr(obj.self.cls, name, obj.after)
        if (c) return this.bindAttr(c, obj.self)
        const host = obj.self.cls.host
        const inst = obj.self
        if (name === '__init__') return fn((args, kw) => { host?.init(inst, args, kw, this); return NONE })
        const h = host?.getAttr?.(inst, name, this)
        if (h) return h
        throw this.err('AttributeError', `'super' object has no attribute '${name}'`)
      }
      case 'func': {
        const own = obj.attrs?.get(name)
        if (own) return own
        if (name === '__name__' || name === '__qualname__') return str_(obj.name)
        if (name === '__doc__') return obj.doc === undefined ? NONE : str_(obj.doc)
        if (name === '__module__') return str_('__main__')
        if (name === '__annotations__') return { k: 'dict', d: PyDict.from([...(obj.notes ?? [])].map(([n, v]) => [str_(n), v])) }
        if (name === '__defaults__') {
          const ds = obj.defaults.filter((d): d is Value => !!d)
          return ds.length ? tuple(ds) : NONE
        }
        break
      }
      case 'fn': {
        const own = obj.attrs?.get(name)
        if (own) return own
        if (name === '__name__' || name === '__qualname__') return str_(obj.name)
        break
      }
      case 'bound':
        if (name === '__self__') return obj.self
        if (name === '__func__') return obj.f
        return this.getAttr(obj.f, name)
      case 'type': {
        const own = obj.attrs?.get(name)
        if (own) return own
        if (name === '__name__') return str_(obj.name.split('.').pop()!)
        if (obj.name === 'str' && name === 'maketrans') {
          return fn((args) => {
            const d = new PyDict()
            if (args[0]?.k === 'dict') {
              for (const [k, v] of args[0].d.items()) d.set(k.k === 'str' ? int(k.v.codePointAt(0)!) : k, v)
              return { k: 'dict', d }
            }
            const [from, to, drop] = args.map((a) => (a as { v: string }).v)
            if (from.length !== to.length) throw this.err('ValueError', 'the first two maketrans arguments must have equal length')
            for (let i = 0; i < from.length; i++) d.set(int(from.codePointAt(i)!), int(to.codePointAt(i)!))
            for (const c of drop ?? '') d.set(int(c.codePointAt(0)!), NONE)
            return { k: 'dict', d }
          })
        }
        if (obj.name === 'dict' && name === 'fromkeys') {
          return fn((args) => ({ k: 'dict', d: PyDict.from(this.iterate(args[0]).map((x) => [x, args[1] ?? NONE])) }))
        }
        // str.lower, list.append …: the method, taking the object as its first argument
        if (['str', 'list', 'dict', 'tuple', 'set', 'int', 'float'].includes(obj.name)) {
          if (!BUILTIN_DIR[obj.name].split(' ').includes(name)) throw this.err('AttributeError', `type object '${obj.name}' has no attribute '${name}'`)
          return fn((args, kw) => {
            if (!args.length) throw this.err('TypeError', `unbound method ${obj.name}.${name}() needs an argument`)
            return this.call(this.getAttr(args[0], name), args.slice(1), kw)
          })
        }
        break
      }
      case 'range':
        if (name === 'start' || name === 'stop' || name === 'step') return int(obj[name])
        break
      case 'int': case 'float': case 'bool':
        if (name === 'item') return fn(() => obj)
        // NumPy's scalars are plain numbers here, and a.mean().round(2) is how people write it
        if (name === 'round' && obj.k === 'float') return fn((args, kw) => this.call(this.builtins.round, [obj, args[0] ?? kw.decimals ?? int(0)]))
        if (name === 'is_integer') return fn(() => bool(Number.isInteger(numOf(obj))))
        if (name === 'real' || name === 'conjugate') {
          const self = obj.k === 'float' ? obj : int(numOf(obj))
          return name === 'real' ? self : fn(() => self)
        }
        if (name === 'imag') return obj.k === 'float' ? float(0) : int(0)
        if (obj.k !== 'float') {
          if (name === 'numerator') return int(numOf(obj))
          if (name === 'denominator') return int(1)
          if (name === 'bit_length') return fn(() => int(numOf(obj) === 0 ? 0 : Math.floor(Math.log2(Math.abs(numOf(obj)))) + 1))
        }
        if (name === 'as_integer_ratio') return fn(() => {
          let n = numOf(obj)
          let d = 1
          while (!Number.isInteger(n)) {
            n *= 2
            d *= 2
          }
          return tuple([int(n), int(d)])
        })
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
        case 'T': return a.matrix ? toMatrix(arr(transpose(a))) : arr(transpose(a))
        case 'base': return a.base ? arr(a.base) : NONE
        case 'strides': return tuple(a.strides.map((s) => int(s * 8)))
      }
      if (a.matrix) {
        switch (name) {
          case 'A': return arr(plainView(a))
          case 'A1': return arr(reshape(plainView(a), [-1]))
          case 'H': return toMatrix(arr(transpose(a)))
          case 'I': return toMatrix(arr(inv(plainView(a))))
        }
      }
      switch (name) {
        case 'real': return obj
        case 'imag': return arr(NDArray.create(new Array<number>(a.size).fill(0), a.shape, a.dtype))
        case 'itemsize': return int(a.dtype === 'bool' || a.dtype === 'uint8' ? 1 : a.dtype === 'float32' ? 4 : 8)
        case 'nbytes': return int(a.size * (a.dtype === 'bool' || a.dtype === 'uint8' ? 1 : a.dtype === 'float32' ? 4 : 8))
        case 'mT': return arr(transpose(a, a.shape.map((_, i) => (i === a.ndim - 1 ? a.ndim - 2 : i === a.ndim - 2 ? a.ndim - 1 : i))))
        case 'flat': return { k: 'obj', o: new FlatIter(a) }
        case 'flags': return { k: 'obj', o: new ArrayFlags(a) }
      }
      // the array methods that are the NumPy function with the array first
      if (ARRAY_METHODS_VIA_FUNCTION[name]) {
        const f = (this.np as { attrs: Record<string, Value> }).attrs[name]
        return { k: 'fn', name, call: (args, kw) => this.call(f, [obj, ...args], kw), api: { name: `ndarray.${name}`, kind: ARRAY_METHODS_VIA_FUNCTION[name] }, self: a }
      }
      const m = this.arrayMethod(name)
      if (m) return { k: 'fn', name, call: m.impl(a), api: m.meta, self: a, rebind: m.impl }
      throw this.err('AttributeError', `'${typeName(obj)}' object has no attribute '${name}'`)
    }
    if (obj.k === 'list') {
      const m = this.listMethod(obj, name)
      if (m) return fn(m)
    }
    if (obj.k === 'tuple') {
      if (obj.names) {
        const at = obj.names.indexOf(name)
        if (at >= 0) return obj.items[at]
        if (name === '_fields') return tuple(obj.names.map(str_))
        if (name === '_asdict') return fn(() => ({ k: 'dict', d: PyDict.from(obj.names!.map((n, i) => [str_(n), obj.items[i]])) }))
        if (name === '_replace') return fn((_a, kw) => ({ ...obj, items: obj.items.map((x, i) => kw[obj.names![i]] ?? x) }))
      }
      if (name === 'count') return fn((args) => int(obj.items.filter((y) => this.equal(y, args[0])).length))
      if (name === 'index') return fn((args) => {
        const i = obj.items.findIndex((y) => this.equal(y, args[0]))
        if (i < 0) throw this.err('ValueError', 'tuple.index(x): x not in tuple')
        return int(i)
      })
    }
    if (obj.k === 'dict') {
      const m = this.dictMethod(obj, name)
      if (m) return fn(m)
    }
    if (obj.k === 'set') {
      const m = this.setMethod(obj, name)
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
    const dunder = this.builtinDunder(obj, name)
    if (dunder) return dunder
    throw this.err('AttributeError', `'${typeName(obj)}' object has no attribute '${name}'`)
  }

  /** [1].__len__(), (1).__add__(2): the operators a built-in value answers to, spelled as methods */
  builtinDunder(obj: Value, name: string): Value | undefined {
    if (!name.startsWith('__') || !name.endsWith('__')) return undefined
    const fn = (call: Fn): Value => ({ k: 'fn', name, call })
    const op = Object.keys(DUNDER).find((o) => DUNDER[o] === name)
    if (op) return fn((args) => this.binop(op, obj, args[0]))
    const builtin = (n: string, ...before: Value[]) => fn((args) => this.call(this.builtins[n], [...before, obj, ...args]))
    switch (name) {
      case '__len__': return builtin('len')
      case '__iter__': return builtin('iter')
      case '__repr__': return builtin('repr')
      case '__str__': return builtin('str')
      case '__hash__': return builtin('hash')
      case '__bool__': return builtin('bool')
      case '__abs__': return builtin('abs')
      case '__contains__': return fn((args) => this.compare('in', args[0], obj))
      case '__getitem__': return fn((args) => {
        const code = this.code
        const self = { k: 'name', id: code, line: this.line } as Node
        return this.getItem({ k: 'sub', obj: self, idx: self, line: this.line } as unknown as Node & { k: 'sub' }, obj, args[0])
      })
    }
    return undefined
  }

  /** a class attribute reached through an instance or class: functions become methods */
  bindAttr(c: Value, self: Value): Value {
    if (c.k === 'func') return { k: 'bound', self, f: c }
    if (c.k === 'fn' && c.bindable) return { ...c, call: (args, kw) => c.call([self, ...args], kw), bindable: false }
    if (c.k === 'descr') {
      if (c.kind === 'static') return c.f
      if (c.kind === 'prop') return this.call(c.f, [self])
      const cls = self.k === 'inst' ? self.cls : self
      return c.f.k === 'func' ? { k: 'bound', self: cls, f: c.f } : { k: 'fn', name: 'classmethod', call: (args, kw) => this.call(c.f, [cls, ...args], kw) }
    }
    return c
  }

  setMethod(v: Value & { k: 'set' }, name: string): Fn | null {
    const others = (args: Value[]) => args.map((a) => makeSet(this.iterate(a)) as Value & { k: 'set' })
    const fold = (op: string, args: Value[]): Value => others(args).reduce<Value>((acc, o) => this.setOp(op, acc as Value & { k: 'set' }, o)!, v)
    const inplace = (op: string, args: Value[]) => {
      if (v.frozen) throw this.err('AttributeError', `'frozenset' object has no attribute '${name}'`)
      v.s = (fold(op, args) as { s: PyDict }).s
      return NONE
    }
    switch (name) {
      case 'union': return (args) => fold('|', args)
      case 'intersection': return (args) => fold('&', args)
      case 'difference': return (args) => fold('-', args)
      case 'symmetric_difference': return (args) => fold('^', args)
      case 'issubset': return (args) => this.setOp('<=', v, others(args)[0])!
      case 'issuperset': return (args) => this.setOp('>=', v, others(args)[0])!
      case 'isdisjoint': return (args) => bool(!setItems(v).some((x) => others(args)[0].s.has(x)))
      case 'copy': return () => makeSet(setItems(v), v.frozen)
    }
    if (v.frozen) return null
    switch (name) {
      case 'add': return (args) => { v.s.set(args[0], args[0]); return NONE }
      case 'remove': return (args) => { if (!v.s.delete(args[0])) throw this.err('KeyError', repr(args[0])); return NONE }
      case 'discard': return (args) => { v.s.delete(args[0]); return NONE }
      case 'clear': return () => { v.s = new PyDict(); return NONE }
      case 'pop': return () => {
        const first = setItems(v)[0]
        if (!first) throw this.err('KeyError', "'pop from an empty set'")
        v.s.delete(first)
        return first
      }
      case 'update': return (args) => inplace('|', args)
      case 'intersection_update': return (args) => inplace('&', args)
      case 'difference_update': return (args) => inplace('-', args)
      case 'symmetric_difference_update': return (args) => inplace('^', args)
    }
    return null
  }

  listMethod(l: Value & { k: 'list' }, name: string): Fn | null {
    const it = l.items
    const idxOf = (x: Value) => it.findIndex((y) => this.equal(y, x))
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
      case 'count': return (args) => int(it.filter((y) => this.equal(y, args[0])).length)
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
      case 'keys': return () => ({ k: 'list', items: m.keys(), view: 'dict_keys' })
      case 'values': return () => ({ k: 'list', items: m.values(), view: 'dict_values' })
      case 'items': return () => ({ k: 'list', items: m.items().map(([a, b]) => tuple([a, b])), view: 'dict_items' })
      case 'clear': return () => { for (const k of m.keys()) m.delete(k); return NONE }
      case 'fromkeys': return (args) => ({ k: 'dict', d: PyDict.from(this.iterate(args[0]).map((x) => [x, args[1] ?? NONE])) })
      case 'popitem': return (args, kw) => {
        const keys = m.keys()
        if (!keys.length) throw this.err('KeyError', "'popitem(): dictionary is empty'")
        const lastArg = kw.last ?? args[0]
        const k = lastArg && !this.truthy(lastArg) ? keys[0] : keys[keys.length - 1]
        const v = m.get(k)!
        m.delete(k)
        return tuple([k, v])
      }
      case 'move_to_end': if (d.flavor !== 'OrderedDict') break
        return (args, kw) => {
          const v = m.get(args[0])
          if (!v) throw this.err('KeyError', repr(args[0]))
          const lastArg = kw.last ?? args[1]
          if (!lastArg || this.truthy(lastArg)) {
            m.delete(args[0])
            m.set(args[0], v)
          } else {
            const rest = m.items().filter(([k]) => PyDict.hash(k) !== PyDict.hash(args[0]))
            m.m.clear()
            m.set(args[0], v)
            for (const [k, x] of rest) m.set(k, x)
          }
          return NONE
        }
      case 'get': return (args) => m.get(args[0]) ?? args[1] ?? NONE
      case 'pop': return (args) => {
        const v = m.get(args[0])
        if (!v) { if (args[1]) return args[1]; throw this.err('KeyError', repr(args[0])) }
        m.delete(args[0])
        return v
      }
      case 'setdefault': return (args) => { if (!m.has(args[0])) m.set(args[0], args[1] ?? NONE); return m.get(args[0])! }
      case 'update': return (args, kw) => {
        const add = (k: Value, v: Value) => m.set(k, d.flavor === 'Counter' ? this.binop('+', m.get(k) ?? int(0), v) : v)
        if (args[0]?.k === 'dict') for (const [k, v] of args[0].d.items()) add(k, v)
        else if (args[0] && d.flavor === 'Counter') for (const k of this.iterate(args[0])) add(k, int(1))
        else if (args[0]) for (const [at, p] of this.iterate(args[0]).entries()) {
          const kv = this.pairOf(p, at)
          add(kv[0], kv[1])
        }
        for (const k in kw) add(str_(k), kw[k])
        return NONE
      }
      case 'copy': return () => ({ ...d, d: PyDict.from(m.items()) })
    }
    if (d.flavor === 'Counter') {
      const ranked = () => [...m.items()].sort((a, b) => numOf(b[1]) - numOf(a[1]))
      switch (name) {
        case 'most_common': return (args) => {
          const top = ranked()
          return { k: 'list', items: (args[0] && args[0].k !== 'none' ? top.slice(0, this.toInt(args[0])) : top).map((e) => tuple([...e])) }
        }
        case 'elements': return () => ({ k: 'list', items: m.items().flatMap(([k, n]) => new Array<Value>(Math.max(0, numOf(n))).fill(k)) })
        case 'total': return () => int(m.values().reduce((s, n) => s + numOf(n), 0))
        case 'subtract': return (args) => {
          const take = (k: Value, n: number) => m.set(k, int(numOf(m.get(k) ?? int(0)) - n))
          if (args[0]?.k === 'dict') for (const [k, n] of args[0].d.items()) take(k, numOf(n))
          else for (const k of this.iterate(args[0])) take(k, 1)
          return NONE
        }
      }
    }
    return null
  }

  /** a str, or a member of class Mode(str, Enum) */
  plainStr(v: Value | undefined): string | null {
    const s = v?.k === 'inst' ? v.cls.meta?.scalar?.(v) : v
    return s?.k === 'str' ? s.v : null
  }

  strMethod(sv: string, name: string): Fn | null {
    const S = (x: string): Value => str_(x)
    const arg = (v: Value | undefined, what: string) => {
      const s = this.plainStr(v)
      if (s === null) throw this.err('TypeError', `${what} must be str, not ${v ? typeName(v) : 'nothing'}`)
      return s
    }
    switch (name) {
      case 'upper': return () => S(sv.toUpperCase())
      case 'lower': case 'casefold': return () => S(sv.toLowerCase())
      case 'title': return () => S(sv.toLowerCase().replace(/[A-Za-z]+/g, (w) => w[0].toUpperCase() + w.slice(1)))
      case 'capitalize': return () => S(sv.slice(0, 1).toUpperCase() + sv.slice(1).toLowerCase())
      case 'swapcase': return () => S([...sv].map((c) => (c === c.toUpperCase() ? c.toLowerCase() : c.toUpperCase())).join(''))
      case 'strip': case 'lstrip': case 'rstrip': return (a) => {
        const set = a[0]?.k === 'str' ? a[0].v : null
        const drop = (c: string) => (set === null ? /\s/.test(c) : set.includes(c))
        let i = 0
        let j = sv.length
        if (name !== 'rstrip') while (i < j && drop(sv[i])) i++
        if (name !== 'lstrip') while (j > i && drop(sv[j - 1])) j--
        return S(sv.slice(i, j))
      }
      case 'split': case 'rsplit': return (a, kw) => {
        const sep = kw.sep ?? a[0]
        const maxV = kw.maxsplit ?? a[1]
        const max = maxV ? this.toInt(maxV) : -1
        const parts = sep && sep.k === 'str' ? sv.split(sep.v) : sv.trim().split(/\s+/).filter(Boolean)
        const glue = sep && sep.k === 'str' ? sep.v : ' '
        if (max < 0 || parts.length <= max + 1) return { k: 'list', items: parts.map(S) }
        const kept = name === 'split'
          ? [...parts.slice(0, max), parts.slice(max).join(glue)]
          : [parts.slice(0, parts.length - max).join(glue), ...parts.slice(parts.length - max)]
        return { k: 'list', items: kept.map(S) }
      }
      case 'splitlines': return () => ({ k: 'list', items: (sv.endsWith('\n') ? sv.slice(0, -1) : sv).split(/\r\n|\r|\n/).filter((_, i, all) => sv !== '' || all.length > 1 || i > 0).map(S) })
      case 'partition': case 'rpartition': return (a) => {
        const sep = arg(a[0], `${name} arg`)
        const at = name === 'partition' ? sv.indexOf(sep) : sv.lastIndexOf(sep)
        if (at < 0) return name === 'partition' ? tuple([S(sv), S(''), S('')]) : tuple([S(''), S(''), S(sv)])
        return tuple([S(sv.slice(0, at)), S(sep), S(sv.slice(at + sep.length))])
      }
      case 'join': return (a) => S(this.iterateOr(a[0], 'can only join an iterable').map((x, i) => {
        const s = this.plainStr(x)
        if (s === null) throw this.err('TypeError', `sequence item ${i}: expected str instance, ${typeName(x)} found`)
        return s
      }).join(sv))
      case 'replace': return (a) => {
        const from = arg(a[0], 'replace() argument 1')
        const to = arg(a[1], 'replace() argument 2')
        const n = a[2] ? this.toInt(a[2]) : -1
        if (n < 0) return S(sv.split(from).join(to))
        const parts = sv.split(from)
        return S(parts.length <= n + 1 ? parts.join(to) : parts.slice(0, n + 1).join(to) + from + parts.slice(n + 1).join(from))
      }
      case 'startswith': case 'endswith': return (a) => {
        const of = a[0]?.k === 'tuple' ? a[0].items.map((x) => arg(x, `${name} arg`)) : [arg(a[0], `${name} arg`)]
        const s = this.sliceOf(sv, a[1], a[2])
        return bool(of.some((p) => (name === 'startswith' ? s.startsWith(p) : s.endsWith(p))))
      }
      case 'find': case 'index': case 'rfind': case 'rindex': return (a) => {
        const sub = arg(a[0], `${name} arg`)
        const from = a[1] && a[1].k !== 'none' ? this.toInt(a[1]) : 0
        const start = from < 0 ? Math.max(0, sv.length + from) : from
        const s = this.sliceOf(sv, a[1], a[2])
        const at = name.startsWith('r') ? s.lastIndexOf(sub) : s.indexOf(sub)
        if (at < 0 && name.endsWith('index')) throw this.err('ValueError', 'substring not found')
        return int(at < 0 ? -1 : at + start)
      }
      case 'count': return (a) => {
        const sub = arg(a[0], 'count arg')
        const s = this.sliceOf(sv, a[1], a[2])
        return int(sub === '' ? s.length + 1 : s.split(sub).length - 1)
      }
      case 'isdigit': case 'isdecimal': case 'isnumeric': return () => bool(/^\d+$/.test(sv))
      case 'isalpha': return () => bool(/^\p{L}+$/u.test(sv))
      case 'isalnum': return () => bool(/^[\p{L}\p{N}]+$/u.test(sv))
      case 'isspace': return () => bool(/^\s+$/.test(sv))
      case 'islower': return () => bool(/\p{Ll}/u.test(sv) && sv === sv.toLowerCase())
      case 'isupper': return () => bool(/\p{Lu}/u.test(sv) && sv === sv.toUpperCase())
      case 'istitle': return () => bool(/[A-Za-z]/.test(sv) && sv === sv.toLowerCase().replace(/[A-Za-z]+/g, (w) => w[0].toUpperCase() + w.slice(1)))
      case 'isidentifier': return () => bool(/^[A-Za-z_]\w*$/.test(sv))
      case 'center': return (a) => S(pad(sv, this.toInt(a[0]), a[1]?.k === 'str' ? a[1].v : ' ', '^'))
      case 'ljust': return (a) => S(pad(sv, this.toInt(a[0]), a[1]?.k === 'str' ? a[1].v : ' ', '<'))
      case 'rjust': return (a) => S(pad(sv, this.toInt(a[0]), a[1]?.k === 'str' ? a[1].v : ' ', '>'))
      case 'zfill': return (a) => {
        const w = this.toInt(a[0])
        const sign = /^[-+]/.test(sv) ? sv[0] : ''
        return S(sign + sv.slice(sign.length).padStart(w - sign.length, '0'))
      }
      case 'removeprefix': return (a) => S(sv.startsWith(arg(a[0], 'arg')) ? sv.slice((a[0] as { v: string }).v.length) : sv)
      case 'removesuffix': return (a) => { const x = arg(a[0], 'arg'); return S(x && sv.endsWith(x) ? sv.slice(0, -x.length) : sv) }
      case 'translate': return (a) => S([...sv].map((c) => {
        const to = a[0].k === 'dict' ? a[0].d.get(int(c.codePointAt(0)!)) : undefined
        if (!to) return c
        return to.k === 'none' ? '' : to.k === 'int' ? String.fromCodePoint(to.v) : this.str(to)
      }).join(''))
      case 'expandtabs': return (a, kw) => {
        const size = kw.tabsize ? this.toInt(kw.tabsize) : a[0] ? this.toInt(a[0]) : 8
        let col = 0
        let out = ''
        for (const c of sv) {
          if (c === '\t') {
            const n = size > 0 ? size - (col % size) : 0
            out += ' '.repeat(n)
            col += n
          } else {
            out += c
            col = c === '\n' || c === '\r' ? 0 : col + 1
          }
        }
        return S(out)
      }
      case 'format': return (args, kw) => {
        let auto = 0
        return S(sv.replace(/\{\{|\}\}|\{([^{}:!]*)(?:!([rs]))?(?::([^{}]*))?\}/g, (whole, key: string | undefined, conv: string | undefined, spec: string | undefined) => {
          if (whole === '{{') return '{'
          if (whole === '}}') return '}'
          const v = key === undefined || key === '' ? args[auto++] : /^\d+$/.test(key) ? args[Number(key)] : kw[key]
          if (!v) throw this.err('IndexError', 'Replacement index out of range for positional args tuple')
          if (v.k === 'obj' && !conv) {
            const own = v.o.format?.(spec ?? '')
            if (own !== undefined) return own
          }
          const own = conv ? undefined : this.instFormat(v, spec ?? '')
          if (own !== undefined) return own
          return spec ? formatSpec(v, spec) : conv === 'r' ? repr(v) : this.str(v)
        }))
      }
    }
    return null
  }

  /** s[start:end] with Python's negative indices, for find(sub, start, end) and friends */
  sliceOf(s: string, start: Value | undefined, end: Value | undefined): string {
    const at = (v: Value | undefined, d: number) => {
      if (!v || v.k === 'none') return d
      const i = this.toInt(v)
      return i < 0 ? Math.max(0, s.length + i) : Math.min(i, s.length)
    }
    return s.slice(at(start, 0), at(end, s.length))
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
      if (a.k === 'obj' || b.k === 'obj' || a.k === 'array' || b.k === 'array' || a.k === 'inst' || b.k === 'inst' || a.k === 'set' || b.k === 'set') return this.truthy(this.binop('<', a, b)) ? -1 : this.truthy(this.binop('<', b, a)) ? 1 : 0
      throw this.err('TypeError', `'<' not supported between instances of '${typeName(a)}' and '${typeName(b)}'`)
    }
    keyed.sort((a, b) => cmp(a.k, b.k))
    if (reverse && this.truthy(reverse)) keyed.reverse()
    return keyed.map((x) => x.v)
  }

  toList(a: NDArray): Value {
    const vals = a.values()
    // Python numbers: a float32 shows all its digits, as float() would
    const item = (x: number) => (a.dtype === 'float32' ? float(x) : fromScalar(x, a.dtype))
    if (a.ndim === 0) return item(vals[0])
    const st = cStrides(a.shape)
    const rec = (d: number, base: number): Value =>
      d === a.ndim - 1
        ? { k: 'list', items: Array.from({ length: a.shape[d] }, (_, i) => item(vals[base + i])) }
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

  /** a median / percentile / average result: a NumPy float scalar or an array */
  orderStat(r: NDArray | number): Value {
    return typeof r === 'number' ? fromScalar(r, 'float64') : arr(r)
  }

  /** np.percentile with one q, or with a list of them: the q axis comes first */
  percentiles(a: NDArray, q: Value | undefined, scale: number, axis: number | null): Value {
    if (q && (q.k === 'list' || q.k === 'tuple' || q.k === 'array')) {
      const parts = Array.from(toArray(q, 'float64').values(), (v) => percentile(a, this.scalarQ(float(v), scale), axis))
      if (typeof parts[0] === 'number') return arr(NDArray.create(parts as number[], [parts.length], 'float64'))
      return arr(stack(parts as NDArray[], 0))
    }
    return this.orderStat(percentile(a, this.scalarQ(q, scale), axis))
  }

  /** percentile's q (scale 1) or quantile's q (scale 100, so 0.75 → 75) as one number */
  scalarQ(v: Value | undefined, scale: number): number {
    if (!v) throw this.err('TypeError', "missing required argument 'q'")
    if (!isNum(v)) throw this.err('NotImplementedError', 'q must be a single number in this sandbox')
    if (scale === 100 && (numOf(v) < 0 || numOf(v) > 1)) throw this.err('ValueError', 'Quantiles must be in the range [0, 1]')
    return numOf(v) * scale
  }

  /** nested Python numbers (pad_width: 1, (1, 2), ((1, 1), (0, 2))) */
  toPyNumbers(v: Value | undefined): number | number[] | number[][] {
    if (!v) throw this.err('TypeError', "pad() missing required argument 'pad_width'")
    if (isNum(v)) return this.toInt(v)
    if (v.k === 'array') return this.toPyNumbers(this.toList(v.a))
    if (v.k !== 'list' && v.k !== 'tuple') throw this.err('TypeError', '`pad_width` must be of integral type.')
    return v.items.map((x) => (isNum(x) ? this.toInt(x) : this.iterate(x).map((y) => this.toInt(y)))) as number[] | number[][]
  }

  /** np.isnan and friends: element-wise, boolean result */
  predicate(name: string, f: (x: number) => boolean): Value {
    return {
      k: 'fn', name, api: { name: `np.${name}`, kind: 'elementwise' },
      call: keepMatrix((args, kw) => {
        const viaObj = args[0].k === 'obj' ? args[0].o.npCall?.(name, args, kw, this, (a) => arr(NDArray.create(toArray(a[0]).values().map((x) => (f(x) ? 1 : 0)), toArray(a[0]).shape, 'bool'))) : undefined
        if (viaObj) return viaObj
        const a = toArray(args[0])
        const res = NDArray.create(a.values().map((x) => (f(x) ? 1 : 0)), a.shape, 'bool')
        return args[0].k === 'array' || args[0].k === 'list' ? arr(res) : bool(res.values()[0] === 1)
      }),
    }
  }

  // ---------- numpy 命名空间与内置函数 ----------

  makeNumpy(): Value {
    const api = (prefix: string) => (name: string, call: Fn, kind?: ApiKind, extra: Partial<ApiMeta> = {}): Value => ({
      k: 'fn', name, call: kind && KEEPS_MATRIX.includes(kind) && !DROPS_MATRIX.includes(name) ? keepMatrix(call) : call, api: kind ? { name: `${prefix}.${name}`, kind, ...extra } : undefined,
    })
    const npApi = api('np')
    const fn: ReturnType<typeof api> = (name, call, kind, extra) => npApi(name, (args, kw) => {
      for (const x of args) {
        const res = x.k === 'obj' ? x.o.npCall?.(name, args, kw, this, call) : undefined
        if (res) return res
      }
      return call(args, kw)
    }, kind, extra)
    const asArr = (v: Value | undefined, name: string): NDArray => {
      if (!v) throw this.err('TypeError', `${name}() missing required argument`)
      return toArray(v)
    }
    const seq = (v: Value | undefined, name: string): NDArray[] => {
      if (v?.k === 'array') return this.iterate(v).map((x) => toArray(x))
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
        const values = isNum(f) ? new Array<number>(a.size).fill(numOf(f)) : toArray(this.binop('+', arr(NDArray.create(new Array(a.size).fill(0), a.shape, 'float64')), arr(toArray(f)))).values()
        const res = arr(NDArray.create(values, a.shape, dtype))
        return a.matrix ? toMatrix(res) : res
      }, 'create')
    const reducer = (kind: ReduceKind) => fn(kind, (args, kw) => this.reduceFn(asArr(args[0], kind), kind, args.slice(1), kw), 'reduce', { axisPos: 1, defaultAxis: null })
    const elementwise = (name: string, f: (x: number) => number, out: 'float' | 'same' | 'int-keep' = 'float') =>
      fn(name, (args) => {
        const a = asArr(args[0], name)
        const dtype: DType = out === 'float' ? (a.dtype === 'float32' ? 'float32' : 'float64') : a.dtype === 'bool' ? 'int64' : a.dtype
        const res = NDArray.create(a.values().map(f), a.shape, dtype)
        return args[0].k === 'array' || args[0].k === 'list' || args[0].k === 'tuple' || args[0].k === 'obj' ? arr(res) : fromScalar(res.values()[0], dtype)
      }, 'elementwise')
    const ufunc2 = (name: string, op: BinOp) => fn(name, (args) => this.binop(op, plainValue(args[0]), plainValue(args[1])), 'elementwise')
    const logical = (name: string, f: (x: number, y: number) => number) =>
      fn(name, (args) => {
        const a = toArray(args[0])
        const b = toArray(args[1])
        const shape = broadcastShapes([a.shape, b.shape])
        const bv = broadcastTo(b, shape).values()
        return arr(NDArray.create(broadcastTo(a, shape).values().map((x, i) => f(x ? 1 : 0, bv[i] ? 1 : 0)), shape, 'bool'))
      }, 'elementwise')
    const dt = (d: DType, name: string): Value => ({
      k: 'type', name: `numpy.${name}`, dtype: d,
      call: (args) => {
        const v = args[0] ?? int(0)
        if (v.k === 'array' || v.k === 'list' || v.k === 'tuple' || v.k === 'obj') return arr(NDArray.create(toArray(v).values(), toArray(v).shape, d))
        return fromScalar(castValue(numOf(v), d), d)
      },
    })
    const shapeArg = (args: Value[]): number[] => (args.length === 1 && (args[0].k === 'tuple' || args[0].k === 'list') ? this.toShape(args) : args.map((x) => this.toInt(x)))

    const rnd = api('np.random')
    const random: Value = {
      k: 'module',
      name: 'numpy.random',
      attrs: {
        seed: rnd('seed', (args, kw) => { this.npRng.seed(this.legacySeed(kw.seed ?? args[0])); return NONE }),
        rand: rnd('rand', (args) => {
          const shape = shapeArg(args)
          if (!shape.length) return float(this.npRng.next())
          checkSize(shape)
          return arr(NDArray.create(Array.from({ length: shape.reduce((p, x) => p * x, 1) }, () => this.npRng.next()), shape, 'float64'))
        }, 'random'),
        randn: rnd('randn', (args) => {
          const shape = shapeArg(args)
          if (!shape.length) return float(this.npRng.normal())
          checkSize(shape)
          return arr(NDArray.create(Array.from({ length: shape.reduce((p, x) => p * x, 1) }, () => this.npRng.normal()), shape, 'float64'))
        }, 'random'),
        randint: rnd('randint', (args, kw) => {
          // the legacy interface cuts floats down to whole numbers
          const whole = (v: Value) => (v.k === 'float' ? Math.trunc(v.v) : this.toInt(v))
          let lo = whole(args[0])
          const hi = kw.high ?? args[1]
          let hiN: number
          if (!hi || hi.k === 'none') { hiN = lo; lo = 0 } else hiN = whole(hi)
          if (hiN <= lo) throw this.err('ValueError', 'high <= low')
          const size = kw.size ?? args[2]
          const draw = () => this.npRng.randint(lo, hiN)
          if (!size || size.k === 'none') return int(draw())
          const shape = this.toShape([size])
          checkSize(shape)
          return arr(NDArray.create(Array.from({ length: shape.reduce((p, x) => p * x, 1) }, draw), shape, 'int64'))
        }, 'random'),
        normal: rnd('normal', (args, kw) => {
          const loc = numOf(kw.loc ?? args[0] ?? float(0))
          const scale = numOf(kw.scale ?? args[1] ?? float(1))
          const size = kw.size ?? args[2]
          if (!size || size.k === 'none') return float(loc + scale * this.npRng.normal())
          const shape = this.toShape([size])
          checkSize(shape)
          return arr(NDArray.create(Array.from({ length: shape.reduce((p, x) => p * x, 1) }, () => loc + scale * this.npRng.normal()), shape, 'float64'))
        }, 'random'),
        uniform: rnd('uniform', (args, kw) => {
          const lo = numOf(kw.low ?? args[0] ?? float(0))
          const hi = numOf(kw.high ?? args[1] ?? float(1))
          const size = kw.size ?? args[2]
          if (!size || size.k === 'none') return float(lo + (hi - lo) * this.npRng.next())
          const shape = this.toShape([size])
          checkSize(shape)
          return arr(NDArray.create(Array.from({ length: shape.reduce((p, x) => p * x, 1) }, () => lo + (hi - lo) * this.npRng.next()), shape, 'float64'))
        }, 'random'),
        permutation: rnd('permutation', (args) => {
          const a = args[0].k === 'int' ? NDArray.create(Array.from({ length: args[0].v }, (_, i) => i), [args[0].v], 'int64') : toArray(args[0])
          const n = a.shape[0] ?? 0
          const order = Array.from({ length: n }, (_, i) => i)
          for (let i = n - 1; i > 0; i--) {
            const j = this.npRng.interval(i)
            ;[order[i], order[j]] = [order[j], order[i]]
          }
          const { value } = getIndex(a, [{ kind: 'array', arr: NDArray.create(order, [n], 'int64') }])
          return arr(value as NDArray)
        }, 'random'),
        choice: rnd('choice', (args, kw) => {
          const pool = args[0].k === 'int' ? Array.from({ length: args[0].v }, (_, i) => i) : toArray(args[0]).values()
          const size = kw.size ?? args[1]
          const draw = () => pool[this.npRng.interval(pool.length - 1)]
          if (!size || size.k === 'none') return fromScalar(draw(), args[0].k === 'int' ? 'int64' : toArray(args[0]).dtype)
          const shape = this.toShape([size])
          checkSize(shape)
          return arr(NDArray.create(Array.from({ length: shape.reduce((p, x) => p * x, 1) }, draw), shape, args[0].k === 'int' ? 'int64' : toArray(args[0]).dtype))
        }, 'random'),
        shuffle: rnd('shuffle', (args) => {
          const v = args[0]
          if (v.k === 'list') {
            for (let i = v.items.length - 1; i > 0; i--) {
              const j = this.npRng.interval(i)
              ;[v.items[i], v.items[j]] = [v.items[j], v.items[i]]
            }
            return NONE
          }
          throw this.err('NotImplementedError', 'np.random.shuffle works on lists here; use a = a[np.random.permutation(len(a))] for arrays')
        }),
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
      asarray: fn('asarray', (args, kw) => {
        const a = toArray(args[0], this.dtypeOf(kw.dtype))
        return arr(a.matrix ? plainView(a) : a)
      }, 'create'),
      matrix: { k: 'type', name: 'numpy.matrix', call: (args, kw) => this.makeMatrix(args[0], this.dtypeOf(kw.dtype ?? args[1])), api: { name: 'np.matrix', kind: 'create' } },
      asmatrix: fn('asmatrix', (args, kw) => {
        const a = toArray(args[0], this.dtypeOf(kw.dtype ?? args[1]))
        if (a.ndim > 2) throw this.err('ValueError', 'shape too large to be a matrix.')
        return arr(matrixArray(a))
      }, 'create'),
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
        const n = this.kwInt(kw, args, 2, 'num') ?? 50
        if (n > this.maxSize) throw this.err('MemoryError', `array of ${n} elements exceeds the sandbox limit of ${this.maxSize}`)
        const endpoint = !kw.endpoint || this.truthy(kw.endpoint)
        const start = kw.start ?? args[0]
        const stop = kw.stop ?? args[1]
        const axis = this.kwInt(kw, args, 5, 'axis') ?? 0
        const out = spacedBetween(start, stop, n, endpoint, axis, (x) => x, this.dtypeOf(kw.dtype) ?? 'float64')
        if (!kw.retstep || !this.truthy(kw.retstep)) return arr(out)
        const div = endpoint ? n - 1 : n
        const step = this.binop('/', this.binop('-', stop, start), float(div))
        return tuple([arr(out), step])
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
        const dtype = pairDType(x, y)
        return arr(NDArray.create(cv.map((v, i) => (v ? xv[i] : yv[i])), shape, dtype))
      }, 'elementwise'),
      nonzero: fn('nonzero', (args) => this.nonzero(asArr(args[0], 'nonzero')), 'info'),
      // ---- linear algebra
      dot: fn('dot', (args) => {
        const [a, b] = args
        if (isNum(a) || isNum(b)) return this.binop('*', isNum(a) ? a : arr(asArr(a, 'dot')), isNum(b) ? b : arr(asArr(b, 'dot')))
        return this.binop('@', arr(asArr(a, 'dot')), arr(asArr(b, 'dot')))
      }, 'matmul'),
      matmul: fn('matmul', (args) => this.binop('@', arr(asArr(args[0], 'matmul')), arr(asArr(args[1], 'matmul'))), 'matmul'),
      outer: fn('outer', (args) => arr(outer(asArr(args[0], 'outer'), asArr(args[1], 'outer'))), 'matmul'),
      trace: fn('trace', (args) => {
        const a = asArr(args[0], 'trace')
        return fromScalar(trace(a), isFloatD(a.dtype) ? a.dtype : 'int64')
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
        const dtype: DType = isFloatD(pa.dtype) || isFloatD(xa.dtype) ? 'float64' : 'int64'
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
      // ---- counting, histograms and order statistics (lib/ndextra)
      bincount: fn('bincount', (args, kw) => {
        const w = kw.weights ?? args[1]
        const r = bincount(asArr(args[0], 'bincount'), this.kwInt(kw, args, 2, 'minlength') ?? 0, w && w.k !== 'none' ? toArray(w) : null)
        this.pendingGroups = { groups: r.groups, result: r.out }
        return arr(r.out)
      }, 'group'),
      histogram: fn('histogram', (args, kw) => {
        const b = kw.bins ?? args[1]
        const density = (kw.density ?? args[3]) && this.truthy(kw.density ?? args[3])
        const weights = kw.weights ?? args[4]
        if ((b && b.k !== 'none' && !isNum(b)) || density || (weights && weights.k !== 'none')) {
          const v = asArr(args[0], 'histogram').values()
          let edges: number[]
          if (b && b.k !== 'none' && !isNum(b)) edges = toArray(b).values()
          else {
            const rg0 = kw.range ?? args[2]
            const span = rg0 && rg0.k !== 'none' ? toArray(rg0, 'float64').values() : [Math.min(...v), Math.max(...v)]
            const [lo, hi] = span[0] === span[1] ? [span[0] - 0.5, span[1] + 0.5] : span
            const n = b && b.k !== 'none' ? this.toInt(b) : 10
            edges = Array.from({ length: n + 1 }, (_, i) => lo + ((hi - lo) * i) / n)
          }
          const w = weights && weights.k !== 'none' ? toArray(weights).values() : null
          const counts = new Array<number>(edges.length - 1).fill(0)
          v.forEach((x, i) => {
            if (x < edges[0] || x > edges[edges.length - 1]) return
            // each bin is half open, except the last
            let k = edges.findIndex((e) => x < e) - 1
            if (k < 0) k = edges.length - 2
            counts[k] += w ? w[i] : 1
          })
          const total = counts.reduce((t, c) => t + c, 0)
          const out = density ? counts.map((c, i) => c / total / (edges[i + 1] - edges[i])) : counts
          const whole = !density && (!w || !isFloatD(toArray(weights!).dtype))
          const edgeType = b && b.k !== 'none' && !isNum(b) ? toArray(b).dtype : 'float64'
          return tuple([arr(NDArray.create(out, [out.length], whole ? 'int64' : 'float64')), arr(NDArray.create(edges, [edges.length], edgeType))])
        }
        const rg = kw.range ?? args[2]
        const range = rg && rg.k !== 'none' ? (toArray(rg, 'float64').values() as [number, number]) : null
        const r = histogram(asArr(args[0], 'histogram'), b && b.k !== 'none' ? this.toInt(b) : 10, range)
        this.pendingGroups = { groups: r.groups, result: r.out }
        return tuple([arr(r.out), arr(r.edges)])
      }, 'group'),
      median: fn('median', (args, kw) => this.orderStat(median(asArr(args[0], 'median'), this.kwInt(kw, args, 1, 'axis'))), 'reduce', { axisPos: 1, defaultAxis: null }),
      percentile: fn('percentile', (args, kw) => this.percentiles(asArr(args[0], 'percentile'), kw.q ?? args[1], 1, this.kwInt(kw, args, 2, 'axis')), 'reduce', { axisPos: 2, defaultAxis: null }),
      quantile: fn('quantile', (args, kw) => this.percentiles(asArr(args[0], 'quantile'), kw.q ?? args[1], 100, this.kwInt(kw, args, 2, 'axis')), 'reduce', { axisPos: 2, defaultAxis: null }),
      average: fn('average', (args, kw) => {
        const w = kw.weights ?? args[2]
        return this.orderStat(average(asArr(args[0], 'average'), this.kwInt(kw, args, 1, 'axis'), w && w.k !== 'none' ? toArray(w) : null))
      }, 'reduce', { axisPos: 1, defaultAxis: null }),
      // ---- grids, differences, padding, membership
      meshgrid: fn('meshgrid', (args, kw) => {
        const ij = kw.indexing && kw.indexing.k === 'str' && kw.indexing.v === 'ij'
        if (kw.indexing && kw.indexing.k === 'str' && !['xy', 'ij'].includes(kw.indexing.v)) throw this.err('ValueError', "Valid values for `indexing` are 'xy' and 'ij'.")
        const sparse = !!kw.sparse && this.truthy(kw.sparse)
        if (args.length === 2 && !ij && !sparse) {
          const { X, Y } = meshgrid(asArr(args[0], 'meshgrid'), asArr(args[1], 'meshgrid'))
          checkSize(X.out.shape)
          this.pendingGroups = { groups: X.groups, result: X.out }
          return tuple([arr(X.out), arr(Y.out)])
        }
        const axes = args.map((a) => toArray(a))
        const sizes = axes.map((a) => a.size)
        const axisOf = (k: number) => (!ij && axes.length >= 2 && k < 2 ? 1 - k : k)
        const shape = axes.map((_, d) => sizes[axes.findIndex((__, k) => axisOf(k) === d)])
        checkSize(shape)
        return tuple(axes.map((a, k) => {
          const at = axisOf(k)
          const vals = a.values()
          if (sparse) return arr(NDArray.create([...vals], shape.map((n, d) => (d === at ? n : 1)), a.dtype))
          const after = shape.slice(at + 1).reduce((p, x) => p * x, 1)
          const total = shape.reduce((p, x) => p * x, 1)
          return arr(NDArray.create(Array.from({ length: total }, (_, i) => vals[Math.floor(i / after) % sizes[k]]), shape, a.dtype))
        }))
      }, 'group'),
      diff: fn('diff', (args, kw) => {
        const n = kw.n ?? args[1]
        if (n && n.k !== 'none' && this.toInt(n) !== 1) throw this.err('NotImplementedError', 'diff supports n=1 in this sandbox')
        const r = diff(asArr(args[0], 'diff'), this.kwInt(kw, args, 2, 'axis') ?? -1)
        this.pendingGroups = { groups: r.groups, result: r.out }
        return arr(r.out)
      }, 'group'),
      pad: fn('pad', (args, kw) => {
        const a = asArr(args[0], 'pad')
        const mode = kw.mode ?? args[2]
        if (mode && mode.k === 'str' && mode.v !== 'constant') throw this.err('NotImplementedError', `pad mode '${mode.v}' is not available in this sandbox (only 'constant')`)
        const spec = this.toPyNumbers(kw.pad_width ?? args[1])
        const cv = kw.constant_values
        const r = padArray(a, padWidths(spec, a.ndim), cv ? this.num(cv) : 0)
        checkSize(r.out.shape)
        this.pendingGroups = { groups: r.groups, result: r.out }
        return arr(r.out)
      }, 'group'),
      isin: fn('isin', (args, kw) => {
        const r = isin(asArr(args[0], 'isin'), toArray(kw.test_elements ?? args[1]))
        this.pendingGroups = { groups: r.groups, result: r.out }
        return arr(r.out)
      }, 'group'),
      isnan: this.predicate('isnan', Number.isNaN),
      isinf: this.predicate('isinf', (x) => x === Infinity || x === -Infinity),
      isfinite: this.predicate('isfinite', Number.isFinite),
      allclose: fn('allclose', (args, kw) => {
        const a = toArray(args[0])
        const b = toArray(args[1])
        const shape = broadcastShapes([a.shape, b.shape])
        return bool(allclose(broadcastTo(a, shape).values(), broadcastTo(b, shape).values(), kw.rtol ? this.num(kw.rtol) : 1e-5, kw.atol ? this.num(kw.atol) : 1e-8))
      }, 'info'),
      array_equal: fn('array_equal', (args) => {
        const a = toArray(args[0])
        const b = toArray(args[1])
        const other = b.values()
        return bool(a.shape.length === b.shape.length && a.shape.every((d, i) => d === b.shape[i]) && a.values().every((x, i) => x === other[i]))
      }, 'info'),
      // ---- element-wise math
      abs: elementwise('abs', Math.abs, 'same'),
      sqrt: elementwise('sqrt', Math.sqrt),
      exp: elementwise('exp', Math.exp),
      log: elementwise('log', Math.log),
      sin: elementwise('sin', Math.sin),
      cos: elementwise('cos', Math.cos),
      tan: elementwise('tan', Math.tan),
      sinh: elementwise('sinh', Math.sinh),
      cosh: elementwise('cosh', Math.cosh),
      tanh: elementwise('tanh', Math.tanh),
      arcsin: elementwise('arcsin', Math.asin),
      arccos: elementwise('arccos', Math.acos),
      arctan: elementwise('arctan', Math.atan),
      arcsinh: elementwise('arcsinh', Math.asinh),
      arccosh: elementwise('arccosh', Math.acosh),
      arctanh: elementwise('arctanh', Math.atanh),
      log2: elementwise('log2', Math.log2),
      log10: elementwise('log10', Math.log10),
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
      shape: fn('shape', (args) => tuple(toArray(args[0]).shape.map(int)), 'info'),
      ndim: fn('ndim', (args) => int(toArray(args[0]).ndim), 'info'),
      size: fn('size', (args) => int(toArray(args[0]).size), 'info'),
      // ---- constants & dtypes
      newaxis: NONE,
      nan: float(NaN),
      inf: float(Infinity),
      pi: float(Math.PI),
      e: float(Math.E),
      int64: dt('int64', 'int64'),
      int32: dt('int64', 'int32'),
      float64: dt('float64', 'float64'),
      float32: dt('float32', 'float32'),
      bool_: dt('bool', 'bool'),
      // uninitialised in NumPy: whatever was in memory. Here that is always 0
      ndarray: { k: 'type', name: 'numpy.ndarray', call: (args, kw) => {
        const shape = this.toShape([kw.shape ?? args[0]])
        checkSize(shape)
        return arr(NDArray.create(new Array(shape.reduce((p, x) => p * x, 1)).fill(0), shape, this.dtypeOf(kw.dtype ?? args[1]) ?? 'float64'))
      } },
    }
    addNumpy(this, attrs, (random as { attrs: Record<string, Value> }).attrs, (linalg as { attrs: Record<string, Value> }).attrs, { fn, rnd, la })
    return { k: 'module', name: 'numpy', attrs }
  }

  makeBuiltins(): Record<string, Value> {
    const fn = (name: string, call: Fn): Value => ({ k: 'fn', name, call })
    const ty = (name: string, call: Fn): Value => ({ k: 'type', name, call })
    const excs: Record<string, Value> = {}
    for (const [name, bases] of EXCEPTIONS) {
      const cls: ClassValue = { k: 'class', name, bases: bases.map((b) => this.excClasses.get(b)!), ns: new Map(), host: EXC_HOST, builtin: true }
      this.excClasses.set(name, cls)
      if (name !== 'LinAlgError' && name !== 'AxisError' && name !== 'UFuncTypeError') excs[name] = cls
    }
    excs.IOError = excs.OSError
    excs.EnvironmentError = excs.OSError
    const iterFrom = (name: string, next: () => Value | undefined): Value => ({ k: 'iter', name, next })
    const kindOf = (v: Value): Value => {
      if (v.k === 'inst') return v.cls
      if (v.k === 'obj' && v.o.typeValue) return v.o.typeValue()
      const name = typeName(v)
      const known = this.builtins?.[name]
      return known && (known.k === 'type' || known.k === 'class') ? known : { k: 'type', name, call: () => NONE }
    }
    const conv = (name: string, d: DType): Value => ({
      k: 'type', name, dtype: d,
      call: (args, kw) => {
        if (d === 'bool') return bool(args.length > 0 && this.truthy(args[0]))
        const x = args[0] ?? int(0)
        const baseArg = args[1] ?? kw.base
        if (d === 'int64' && baseArg) {
          if (x.k !== 'str') throw this.err('TypeError', "int() can't convert non-string with explicit base")
          const base = this.toInt(baseArg)
          const t = x.v.trim().replace(/_/g, '').toLowerCase()
          const neg = t.startsWith('-')
          let body = t.replace(/^[-+]/, '')
          const prefix = /^0([xbo])/.exec(body)
          const fromPrefix = prefix ? { x: 16, b: 2, o: 8 }[prefix[1] as 'x' | 'b' | 'o'] : 0
          if (prefix && (base === 0 || base === fromPrefix)) body = body.slice(2)
          const radix = base === 0 ? fromPrefix || 10 : base
          const digits = '0123456789abcdefghijklmnopqrstuvwxyz'.slice(0, radix)
          if (!body || [...body].some((c) => !digits.includes(c))) throw this.err('ValueError', `invalid literal for int() with base ${base}: ${reprStr(x.v)}`)
          const n = parseInt(body, radix)
          return int(neg ? -n : n)
        }
        if (x.k === 'str' && d === 'float64' && /^\s*[-+]?(inf|infinity|nan)\s*$/i.test(x.v)) {
          return float(/nan/i.test(x.v) ? NaN : x.v.includes('-') ? -Infinity : Infinity)
        }
        if (x.k === 'str') {
          const n = Number(x.v)
          if (x.v.trim() === '' || Number.isNaN(n) || (d === 'int64' && !/^\s*[-+]?\d+\s*$/.test(x.v))) {
            throw this.err('ValueError', `invalid literal for int() with base 10: ${reprStr(x.v)}`)
          }
          return fromScalar(n, d)
        }
        if (x.k === 'array' && x.a.size !== 1) throw this.err('TypeError', 'only length-1 arrays can be converted to Python scalars')
        if (x.k === 'obj') return fromScalar(castValue(this.num(x, 'a number'), d), d)
        if (x.k === 'inst') {
          const f = this.classAttr(x.cls, d === 'float64' ? '__float__' : '__int__') ?? this.classAttr(x.cls, '__index__')
          if (f) return fromScalar(castValue(numOf(this.call(f, [x])), d), d)
        }
        const v = x.k === 'array' ? x.a.values()[0] : numOf(x)
        if (Number.isNaN(v) && !isNum(x) && x.k !== 'array') throw this.err('TypeError', `${name}() argument must be a string${name === 'int' ? ', a bytes-like object' : ''} or a real number, not '${typeName(x)}'`)
        return fromScalar(castValue(v, d), d)
      },
    })
    return {
      print: fn('print', (args, kw) => {
        for (const name of ['sep', 'end']) if (kw[name] && kw[name].k !== 'str' && kw[name].k !== 'none') throw this.err('TypeError', `${name} must be None or a string, not ${typeName(kw[name])}`)
        const sep = kw.sep && kw.sep.k === 'str' ? kw.sep.v : ' '
        const end = kw.end && kw.end.k === 'str' ? kw.end.v : '\n'
        const shown = args.map((a) => this.str(a))
        if (kw.file && kw.file.k !== 'none' || this.stdoutFile) {
          // a file sees each piece as its own write, as Python does it
          const write = (s: string) => (kw.file && kw.file.k !== 'none' ? this.call(this.getAttr(kw.file, 'write'), [str_(s)]) : this.print(s))
          shown.forEach((s, i) => {
            if (i) write(sep)
            write(s)
          })
          write(end)
        } else this.print(shown.join(sep) + end)
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
        if (v.k === 'set') return int(v.s.size)
        if (v.k === 'range') return int(rangeLen(v))
        if (v.k === 'obj' && v.o.len) return int(v.o.len(this))
        if (v.k === 'inst') {
          const f = this.classAttr(v.cls, '__len__')
          if (f) return this.call(f, [v])
        }
        if (v.k === 'class' && v.meta?.len) return int(v.meta.len(v, this))
        throw this.err('TypeError', `object of type '${typeName(v)}' has no len()`)
      }),
      range: { k: 'type', name: 'range', call: (args) => {
        if (!args.length) throw this.err('TypeError', 'range expected at least 1 argument, got 0')
        const n = args.map((x) => this.toInt(x))
        const [a, b, st] = n.length === 1 ? [0, n[0], 1] : [n[0], n[1], n[2] ?? 1]
        if (st === 0) throw this.err('ValueError', 'range() arg 3 must not be zero')
        return { k: 'range', start: a, stop: b, step: st }
      } },
      list: ty('list', (args) => ({ k: 'list', items: args[0] ? [...this.iterate(args[0])] : [] })),
      tuple: ty('tuple', (args) => tuple(args[0] ? [...this.iterate(args[0])] : [])),
      set: ty('set', (args) => makeSet(args[0] ? this.iterate(args[0]) : [])),
      frozenset: ty('frozenset', (args) => makeSet(args[0] ? this.iterate(args[0]) : [], true)),
      iter: fn('iter', (args) => {
        const x = args[0]
        if (x.k === 'iter') return x
        return iterFrom(x.k === 'list' ? 'list_iterator' : x.k === 'tuple' ? 'tuple_iterator' : x.k === 'str' ? 'str_iterator' : 'iterator', this.iterOf(x))
      }),
      next: fn('next', (args) => {
        const x = args[0]
        let r: Value | undefined
        if (x.k === 'iter') r = x.next()
        else if (x.k === 'obj' && x.o.nextItem) r = x.o.nextItem(this)
        else if (x.k === 'inst' && this.classAttr(x.cls, '__next__')) {
          try {
            r = this.call(this.classAttr(x.cls, '__next__')!, [x])
          } catch (e) {
            if (!args[1] || !this.isStop(e)) throw e
          }
        } else throw this.err('TypeError', `'${typeName(x)}' object is not an iterator`)
        if (r !== undefined) return r
        if (args[1]) return args[1]
        return this.raise(this.excClass('StopIteration'), x.k === 'iter' && x.result ? [x.result] : [])
      }),
      callable: fn('callable', (args) => {
        const x = args[0]
        return bool(['fn', 'func', 'bound', 'type', 'class', 'poly'].includes(x.k) || (x.k === 'inst' && (!!this.classAttr(x.cls, '__call__') || !!x.cls.host?.call)) || (x.k === 'obj' && !!x.o.call))
      }),
      eval: fn('eval', (args, kw) => this.runText('eval', args, kw)),
      exec: fn('exec', (args, kw) => this.runText('exec', args, kw)),
      issubclass: fn('issubclass', (args) => {
        const c = args[0]
        if (c.k !== 'class' && c.k !== 'type') throw this.err('TypeError', 'issubclass() arg 1 must be a class')
        const of = (t: Value): boolean => {
          if (t.k === 'tuple') return t.items.some(of)
          if (t.k !== 'type' && t.k !== 'class' && t.k !== 'obj') throw this.err('TypeError', 'issubclass() arg 2 must be a class, a tuple of classes, or a union')
          return (t.k === 'type' && t.name === 'object') || this.classBelow(c, t)
        }
        return bool(of(args[1]))
      }),
      vars: fn('vars', (args) => (args[0] ? this.getAttr(args[0], '__dict__') : { k: 'dict', d: PyDict.from([...(this.frame?.vars ?? this.env)].map(([n, v]) => [str_(n), v])) })),
      globals: fn('globals', () => ({ k: 'dict', d: PyDict.from([...this.env].map(([n, v]) => [str_(n), v])) })),
      locals: fn('locals', () => ({ k: 'dict', d: PyDict.from([...(this.frame?.vars ?? this.env)].map(([n, v]) => [str_(n), v])) })),
      delattr: fn('delattr', (args) => {
        if (args[0].k !== 'inst' || !args[0].attrs.delete((args[1] as { v: string }).v)) throw this.err('AttributeError', (args[1] as { v: string }).v)
        return NONE
      }),
      dir: fn('dir', (args) => {
        const x = args[0]
        const names = new Set<string>()
        if (!x) for (const n of (this.frame?.vars ?? this.env).keys()) names.add(n)
        else if (x.k === 'module') Object.keys(x.attrs).forEach((n) => names.add(n))
        else if (x.k === 'inst' || x.k === 'class') {
          if (x.k === 'inst') for (const n of x.attrs.keys()) names.add(n)
          const walk = (c: Value) => {
            if (c.k !== 'class') return
            for (const n of c.ns.keys()) names.add(n)
            c.bases.forEach(walk)
          }
          walk(x.k === 'inst' ? x.cls : x)
        } else if (x.k === 'type' && BUILTIN_DIR[x.name]) BUILTIN_DIR[x.name].split(' ').forEach((n) => names.add(n))
        else if (BUILTIN_DIR[x.k]) BUILTIN_DIR[x.k].split(' ').forEach((n) => names.add(n))
        return { k: 'list', items: [...names].sort().map(str_) }
      }),
      id: fn('id', (args) => {
        const x = args[0]
        if (x.k === 'array') return int(objId(x.a))
        if (x.k === 'list' || x.k === 'dict' || x.k === 'set' || x.k === 'inst' || x.k === 'func' || x.k === 'class') return int(objId(x))
        if (x.k === 'obj') return int(objId(x.o))
        return int(Math.abs([...PyDict.hash(x)].reduce((h, c) => (h * 31 + c.charCodeAt(0)) | 0, 7)))
      }),
      hash: fn('hash', (args) => {
        const x = args[0]
        if (x.k === 'inst') {
          const f = this.classAttr(x.cls, '__hash__')
          if (f) return this.call(f, [x])
        }
        const key = PyDict.hash(x)
        // a library number that equals an integer hashes like it
        const n = isNum(x) ? numOf(x) : x.k === 'obj' && key.startsWith('n') ? Number(key.slice(1)) : NaN
        if (Number.isInteger(n)) return int(n === -1 ? -2 : n)
        return int([...key].reduce((h, c) => (h * 31 + c.charCodeAt(0)) | 0, 7))
      }),
      hex: fn('hex', (args) => { const n = this.toInt(args[0]); return str_(`${n < 0 ? '-' : ''}0x${Math.abs(n).toString(16)}`) }),
      bin: fn('bin', (args) => { const n = this.toInt(args[0]); return str_(`${n < 0 ? '-' : ''}0b${Math.abs(n).toString(2)}`) }),
      oct: fn('oct', (args) => { const n = this.toInt(args[0]); return str_(`${n < 0 ? '-' : ''}0o${Math.abs(n).toString(8)}`) }),
      ascii: fn('ascii', (args) => str_(repr(args[0]))),
      staticmethod: ty('staticmethod', (args) => ({ k: 'descr', kind: 'static', f: args[0] })),
      classmethod: ty('classmethod', (args) => ({ k: 'descr', kind: 'class', f: args[0] })),
      property: ty('property', (args, kw) => ({ k: 'descr', kind: 'prop', f: args[0] ?? kw.fget, setter: args[1] ?? kw.fset })),
      NotImplemented: NOT_IMPLEMENTED,
      __name__: str_('__main__'),
      __file__: str_('/sandbox/main.py'),
      __doc__: NONE,
      input: fn('input', () => { throw this.err('EOFError', 'this sandbox has no keyboard input: set the value in the code instead') }),
      open: fn('open', (args, kw) => this.call((this.module('_sandbox_files') as { attrs: Record<string, Value> }).attrs.open, args, kw)),
      help: fn('help', (args) => {
        const x = args[0]
        const doc = x?.k === 'func' ? x.doc : undefined
        this.print(`Help on ${x ? repr(x) : 'Python'}:\n${doc ?? 'The sandbox has no documentation for this; see the library\'s website.'}\n`)
        return NONE
      }),
      ...excs,
      abs: fn('abs', (args) => {
        const x = args[0]
        if (x.k === 'str' || x.k === 'none' || x.k === 'list' || x.k === 'tuple' || x.k === 'dict' || x.k === 'set') throw this.err('TypeError', `bad operand type for abs(): '${typeName(x)}'`)
        if (x.k === 'inst') {
          const f = this.classAttr(x.cls, '__abs__')
          if (f) return this.call(f, [x])
        }
        if (x.k === 'obj') {
          const f = this.getAttrOr(x, '__abs__') ?? this.getAttrOr(x, 'abs')
          if (f) return this.call(f, [])
          throw this.err('TypeError', `bad operand type for abs(): '${x.o.cls}'`)
        }
        if (x.k === 'array') return arr(NDArray.create(x.a.values().map(Math.abs), x.a.shape, x.a.dtype === 'bool' ? 'int64' : x.a.dtype))
        return x.k === 'float' ? float(Math.abs(x.v)) : int(Math.abs(numOf(x)))
      }),
      type: ty('type', (args) => {
        if (args.length !== 3) return kindOf(args[0])
        // type(name, bases, namespace) is the class statement as a call
        if (args[0].k !== 'str' || args[2].k !== 'dict') throw this.err('TypeError', 'type() argument 1 must be str, argument 3 must be dict')
        const bases = this.iterate(args[1])
        const ns = new Map(args[2].d.items().map(([k, v]): [string, Value] => [(k as { v: string }).v, v]))
        return { k: 'class', name: args[0].v, bases, ns, host: this.hostOf(bases) } as Value
      }),
      str: { k: 'type', name: 'str', call: (args) => str_(args.length ? this.str(args[0]) : '') },
      repr: fn('repr', (args) => str_(repr(args[0]))),
      format: fn('format', (args) => {
        const spec = args[1]?.k === 'str' ? args[1].v : ''
        return str_(this.instFormat(args[0], spec) ?? (spec ? formatSpec(args[0], spec) : this.str(args[0])))
      }),
      chr: fn('chr', (args) => {
        const n = this.toInt(args[0])
        if (n < 0 || n >= 0x110000) throw this.err('ValueError', 'chr() arg not in range(0x110000)')
        return str_(String.fromCodePoint(n))
      }),
      ord: fn('ord', (args) => {
        const s = args[0]
        if (s.k !== 'str') throw this.err('TypeError', `ord() expected string of length 1, but ${typeName(s)} found`)
        const chars = [...s.v]
        if (chars.length !== 1) throw this.err('TypeError', `ord() expected a character, but string of length ${chars.length} found`)
        return int(chars[0].codePointAt(0)!)
      }),
      dict: { k: 'type', name: 'dict', call: (args, kw) => {
        const d = new PyDict()
        const keys = args[0]?.k === 'obj' || args[0]?.k === 'inst' ? this.getAttrOr(args[0], 'keys') : undefined
        if (args[0]?.k === 'dict') for (const [k, v] of args[0].d.items()) d.set(k, v)
        else if (keys) {
          const from = args[0]
          for (const k of this.iterate(this.call(keys, []))) d.set(k, from.k === 'obj' && from.o.getItem ? from.o.getItem(k, this) : this.call(this.getAttr(from, '__getitem__'), [k]))
        }
        else if (args[0]) for (const [at, p] of this.iterate(args[0]).entries()) {
          const kv = this.pairOf(p, at)
          d.set(kv[0], kv[1])
        }
        for (const k in kw) d.set(str_(k), kw[k])
        return { k: 'dict', d }
      } },
      object: { k: 'type', name: 'object', call: () => ({ k: 'inst', cls: { k: 'class', name: 'object', bases: [], ns: new Map(), host: null, builtin: true }, attrs: new Map(), state: {} }) },
      enumerate: ty('enumerate', (args, kw) => {
        const start = kw.start ?? args[1]
        let i = start ? this.toInt(start) : 0
        const next = this.iterOf(args[0])
        return iterFrom('enumerate', () => {
          const v = next()
          return v === undefined ? undefined : tuple([int(i++), v])
        })
      }),
      zip: ty('zip', (args, kw) => {
        const strict = kw.strict && this.truthy(kw.strict)
        const nexts = args.map((a) => this.iterOf(a))
        return iterFrom('zip', () => {
          if (!nexts.length) return undefined
          const row: Value[] = []
          for (let i = 0; i < nexts.length; i++) {
            const v = nexts[i]()
            if (v === undefined) {
              if (strict) {
                if (i > 0) throw this.err('ValueError', `zip() argument ${i + 1} is shorter than argument${i > 1 ? `s 1-${i}` : ' 1'}`)
                for (let j = 1; j < nexts.length; j++) if (nexts[j]() !== undefined) throw this.err('ValueError', `zip() argument ${j + 1} is longer than argument${j > 1 ? `s 1-${j}` : ' 1'}`)
              }
              return undefined
            }
            row.push(v)
          }
          return tuple(row)
        })
      }),
      sum: fn('sum', (args, kw) => this.iterate(args[0]).reduce((acc, v) => this.binop('+', acc, v), kw.start ?? args[1] ?? int(0))),
      min: fn('min', (args, kw) => this.extreme(args, kw, -1)),
      max: fn('max', (args, kw) => this.extreme(args, kw, 1)),
      sorted: fn('sorted', (args, kw) => ({ k: 'list', items: this.sorted([...this.iterate(args[0])], kw.key, kw.reverse) })),
      reversed: ty('reversed', (args) => {
        const x = args[0]
        if (x.k === 'inst') {
          const f = this.classAttr(x.cls, '__reversed__')
          if (f) return this.call(f, [x])
        }
        if (x.k === 'iter' || x.k === 'set') throw this.err('TypeError', `'${typeName(x)}' object is not reversible`)
        const items = [...this.iterate(x)]
        let i = items.length
        return iterFrom(x.k === 'list' ? 'list_reverseiterator' : x.k === 'range' ? 'range_iterator' : x.k === 'dict' ? 'dict_reversekeyiterator' : 'reversed', () => (i > 0 ? items[--i] : undefined))
      }),
      any: fn('any', (args) => {
        const next = this.iterOf(args[0])
        for (let v = next(); v !== undefined; v = next()) if (this.truthy(v)) return bool(true)
        return bool(false)
      }),
      all: fn('all', (args) => {
        const next = this.iterOf(args[0])
        for (let v = next(); v !== undefined; v = next()) if (!this.truthy(v)) return bool(false)
        return bool(true)
      }),
      map: ty('map', (args) => {
        const nexts = args.slice(1).map((a) => this.iterOf(a))
        return iterFrom('map', () => {
          if (!nexts.length) return undefined
          const row: Value[] = []
          for (const next of nexts) {
            const v = next()
            if (v === undefined) return undefined
            row.push(v)
          }
          return this.call(args[0], row)
        })
      }),
      filter: ty('filter', (args) => {
        const keep = (v: Value) => this.truthy(args[0].k === 'none' ? v : this.call(args[0], [v]))
        const next = this.iterOf(args[1])
        return iterFrom('filter', () => {
          for (let v = next(); v !== undefined; v = next()) if (keep(v)) return v
          return undefined
        })
      }),
      round: fn('round', (args) => {
        const x = args[0]
        if (x.k === 'str' || x.k === 'none' || x.k === 'list' || x.k === 'tuple' || x.k === 'dict' || x.k === 'set') throw this.err('TypeError', `type ${typeName(x)} doesn't define __round__ method`)
        const nd = args[1] && args[1].k !== 'none' ? this.toInt(args[1]) : null
        if (x.k === 'obj' || x.k === 'array') {
          const r = (x.k === 'obj' ? this.getAttrOr(x, '__round__') : undefined) ?? this.getAttr(x, 'round')
          return this.call(r, nd === null ? [] : [int(nd)])
        }
        const own = x.k === 'inst' ? this.classAttr(x.cls, '__round__') : undefined
        if (own) return this.call(own, nd === null ? [x] : [x, int(nd)])
        const v = this.num(x)
        if (nd === null) return int(roundHalfEven(v))
        if (x.k === 'float') return float(roundExact(v, nd))
        return nd >= 0 ? int(v) : int(roundHalfEven(v / 10 ** -nd) * 10 ** -nd)
      }),
      pow: fn('pow', (args, kw) => {
        const m = args[2] ?? kw.mod
        if (!m || m.k === 'none') return this.binop('**', args[0], args[1] ?? kw.exp)
        let result = 1n
        let base = BigInt(this.toInt(args[0]))
        const mod = BigInt(this.toInt(m))
        if (mod === 0n) throw this.err('ValueError', 'pow() 3rd argument cannot be 0')
        for (let e = BigInt(this.toInt(args[1])); e > 0n; e >>= 1n) {
          if (e & 1n) result = (result * base) % mod
          base = (base * base) % mod
        }
        return int(Number(((result % mod) + mod) % mod))
      }),
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
        if (!f?.owner || f.self?.k !== 'inst') throw this.err('RuntimeError', 'super(): no arguments')
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

  /** eval() and exec(): the text runs here, in the caller's scope or in the dictionaries it was given; it adds no steps of its own */
  runText(mode: 'eval' | 'exec', args: Value[], kw: Kw): Value {
    const src = args[0]
    if (src?.k !== 'str') throw this.err('TypeError', `${mode}() arg 1 must be a string, bytes or code object`)
    const given = (v: Value | undefined, what: string): PyDict | null => {
      if (!v || v.k === 'none') return null
      if (v.k !== 'dict') throw this.err('TypeError', `${mode}() ${what} must be a dict, not ${typeName(v)}`)
      return v.d
    }
    const [g, l] = [given(args[1] ?? kw.globals, 'globals'), given(args[2] ?? kw.locals, 'locals')]
    const code = parseText(src.v, mode, this.line)
    const names = (d: PyDict) => new Map(d.items().flatMap(([n, v]): [string, Value][] => (n.k === 'str' ? [[n.v, v]] : [])))
    const [env, frame] = [this.env, this.frame]
    const scope: Frame | null = l ? { vars: names(l), parent: null } : g ? null : frame
    if (g) this.env = names(g)
    this.frame = scope
    this.quiet++
    try {
      if (mode === 'eval') return this.eval(code as Node)
      this.execBlock(code as Stmt[])
      return NONE
    } finally {
      this.quiet--
      if (g) for (const [n, v] of this.env) g.set(str_(n), v)
      if (l && scope) for (const [n, v] of scope.vars) l.set(str_(n), v)
      this.env = env
      this.frame = frame
    }
  }

  /** is class x class t, or one below it */
  classBelow(x: Value, t: Value): boolean {
    if (x === t) return true
    if (x.k === 'class') return x.bases.some((b) => this.classBelow(b, t))
    if (x.k !== 'type') return false
    // a library class says what it inherits from; a Python class of the same name is not that
    if ((t.k === 'type' || (t.k === 'class' && t.builtin)) && x.parents?.includes(t.name.split('.').pop()!)) return true
    return t.k === 'type' && (x.name === t.name || (x.name === 'bool' && t.name === 'int'))
  }

  /** iterate, with the error this caller gives for what cannot be iterated */
  iterateOr(v: Value, message: string): Value[] {
    if (isNum(v) || v.k === 'none') throw this.err('TypeError', message)
    return this.iterate(v)
  }

  /** an element of what dict() is given: a key and a value */
  pairOf(p: Value, at: number): Value[] {
    const kv = this.iterateOr(p, `cannot convert dictionary update sequence element #${at} to a sequence`)
    if (kv.length !== 2) throw this.err('ValueError', `dictionary update sequence element #${at} has length ${kv.length}; 2 is required`)
    return kv
  }

  isinstance(v: Value, t: Value): boolean {
    if (t.k === 'tuple') return t.items.some((x) => this.isinstance(v, x))
    if (t.k === 'type' && t.name === 'object') return true
    if (t.k === 'class') {
      // a library's own object, of a class Python code can also inherit from
      if (v.k === 'obj') return !!t.builtin && !!v.o.isa?.(t.name.split('.').pop()!)
      if (v.k !== 'inst') return false
      const walk = (c: ClassValue): boolean => c === t || c.bases.some((b) => b.k === 'class' && walk(b))
      return walk(v.cls)
    }
    if (t.k === 'type') {
      const name = t.name
      if (v.k === 'array' && (name === 'numpy.ndarray' || name === 'numpy.matrix')) return name === 'numpy.ndarray' || !!v.a.matrix
      if (v.k === 'obj') return v.o.cls === name || !!v.o.isa?.(name)
      if (v.k === 'inst') return (v.cls.host !== null && t.host === v.cls.host) || this.classBelow(v.cls, t)
      const map: Record<string, Value['k'][]> = { int: ['int', 'bool'], float: ['float'], bool: ['bool'], str: ['str'], list: ['list'], tuple: ['tuple'], dict: ['dict'], set: ['set'], frozenset: ['set'], range: ['range'], 'numpy.ndarray': ['array'] }
      if (v.k === 'set') return (name === 'frozenset') === !!v.frozen && (name === 'set' || name === 'frozenset')
      // NoneType, type, Counter, a namedtuple …: whatever the value says it is
      return (map[name] ?? []).includes(v.k) || typeName(v) === name.split('.').pop()
    }
    if (t.k === 'obj' && t.o.cls === 'typing._GenericAlias') return true
    throw this.err('TypeError', 'isinstance() arg 2 must be a type, a tuple of types, or a union')
  }

  makeMath(): Value {
    const f1 = (name: string, g: (x: number) => number): Value => ({ k: 'fn', name, call: (args) => {
      const r = g(this.num(args[0]))
      if (Number.isNaN(r) && !Number.isNaN(this.num(args[0]))) throw this.err('ValueError', 'math domain error')
      return float(r)
    } })
    const g2 = (name: string, g: (x: number, y: number) => number): Value => ({ k: 'fn', name, call: (args) => float(g(this.num(args[0]), this.num(args[1]))) })
    return {
      k: 'module', name: 'math', attrs: {
        sqrt: f1('sqrt', Math.sqrt), exp: f1('exp', Math.exp), log: { k: 'fn', name: 'log', call: (args) => {
          const x = this.num(args[0])
          if (x <= 0) throw this.err('ValueError', 'math domain error')
          return float(args[1] ? Math.log(x) / Math.log(this.num(args[1])) : Math.log(x))
        } },
        log2: f1('log2', Math.log2), log10: f1('log10', Math.log10), sin: f1('sin', Math.sin), cos: f1('cos', Math.cos), tan: f1('tan', Math.tan),
        tanh: f1('tanh', Math.tanh), fabs: f1('fabs', Math.abs), sinh: f1('sinh', Math.sinh), cosh: f1('cosh', Math.cosh),
        asin: f1('asin', Math.asin), acos: f1('acos', Math.acos), atan: f1('atan', Math.atan), asinh: f1('asinh', Math.asinh), acosh: f1('acosh', Math.acosh), atanh: f1('atanh', Math.atanh),
        degrees: f1('degrees', (x) => (x * 180) / Math.PI), radians: f1('radians', (x) => (x * Math.PI) / 180), expm1: f1('expm1', Math.expm1), log1p: f1('log1p', Math.log1p), cbrt: f1('cbrt', Math.cbrt), exp2: f1('exp2', (x) => 2 ** x),
        erf: f1('erf', erf), erfc: f1('erfc', (x) => 1 - erf(x)), gamma: f1('gamma', gammaFn), lgamma: f1('lgamma', (x) => Math.log(Math.abs(gammaFn(x)))),
        atan2: g2('atan2', Math.atan2), hypot: { k: 'fn', name: 'hypot', call: (args) => float(Math.hypot(...args.map((a) => this.num(a)))) }, copysign: g2('copysign', (x, y) => (y < 0 || Object.is(y, -0) ? -Math.abs(x) : Math.abs(x))),
        fmod: g2('fmod', (x, y) => {
          if (y === 0) throw this.err('ValueError', 'math domain error')
          return x % y
        }),
        pow: g2('pow', (x, y) => x ** y),
        dist: { k: 'fn', name: 'dist', call: (args) => {
          const a = this.iterate(args[0]).map((v) => this.num(v))
          const b = this.iterate(args[1]).map((v) => this.num(v))
          if (a.length !== b.length) throw this.err('ValueError', 'both points must have the same number of dimensions')
          return float(Math.hypot(...a.map((x, i) => x - b[i])))
        } },
        fsum: { k: 'fn', name: 'fsum', call: (args) => {
          // Kahan's sum, close enough to Shewchuk's for what a course adds up
          let sum = 0
          let c = 0
          for (const v of this.iterate(args[0])) {
            const x = this.num(v)
            const t = sum + x
            c += Math.abs(sum) >= Math.abs(x) ? sum - t + x : x - t + sum
            sum = t
          }
          return float(sum + c)
        } },
        prod: { k: 'fn', name: 'prod', call: (args, kw) => this.iterate(args[0]).reduce((acc, v) => this.binop('*', acc, v), kw.start ?? int(1)) },
        modf: { k: 'fn', name: 'modf', call: (args) => {
          const x = this.num(args[0])
          return tuple([float(x - Math.trunc(x)), float(Math.trunc(x))])
        } },
        trunc: { k: 'fn', name: 'trunc', call: (args) => int(Math.trunc(this.num(args[0]))) },
        frexp: { k: 'fn', name: 'frexp', call: (args) => {
          const x = this.num(args[0])
          if (x === 0 || !Number.isFinite(x)) return tuple([float(x), int(0)])
          const e = Math.floor(Math.log2(Math.abs(x))) + 1
          return tuple([float(x / 2 ** e), int(e)])
        } },
        ldexp: { k: 'fn', name: 'ldexp', call: (args) => float(this.num(args[0]) * 2 ** this.toInt(args[1])) },
        isnan: { k: 'fn', name: 'isnan', call: (args) => bool(Number.isNaN(this.num(args[0]))) },
        isinf: { k: 'fn', name: 'isinf', call: (args) => bool(Math.abs(this.num(args[0])) === Infinity) },
        isfinite: { k: 'fn', name: 'isfinite', call: (args) => bool(Number.isFinite(this.num(args[0]))) },
        factorial: { k: 'fn', name: 'factorial', call: (args) => {
          const n = this.toInt(args[0])
          if (n < 0) throw this.err('ValueError', 'factorial() not defined for negative values')
          let r = 1
          for (let i = 2; i <= n; i++) r *= i
          return int(r)
        } },
        gcd: { k: 'fn', name: 'gcd', call: (args) => int(args.map((a) => Math.abs(this.toInt(a))).reduce(gcdOf, 0)) },
        lcm: { k: 'fn', name: 'lcm', call: (args) => int(args.map((a) => Math.abs(this.toInt(a))).reduce((a, b) => (a && b ? (a / gcdOf(a, b)) * b : 0), 1)) },
        comb: { k: 'fn', name: 'comb', call: (args) => {
          const n = this.toInt(args[0])
          const k = this.toInt(args[1])
          if (k > n) return int(0)
          let r = 1
          for (let i = 1; i <= Math.min(k, n - k); i++) r = (r * (n - i + 1)) / i
          return int(Math.round(r))
        } },
        perm: { k: 'fn', name: 'perm', call: (args) => {
          const n = this.toInt(args[0])
          const k = args[1] && args[1].k !== 'none' ? this.toInt(args[1]) : n
          if (k > n) return int(0)
          let r = 1
          for (let i = 0; i < k; i++) r *= n - i
          return int(r)
        } },
        isqrt: { k: 'fn', name: 'isqrt', call: (args) => int(Math.floor(Math.sqrt(this.toInt(args[0])))) },
        tau: float(2 * Math.PI),
        floor: { k: 'fn', name: 'floor', call: (args) => int(Math.floor(this.num(args[0]))) },
        ceil: { k: 'fn', name: 'ceil', call: (args) => int(Math.ceil(this.num(args[0]))) },
        isclose: { k: 'fn', name: 'isclose', call: (args) => bool(Math.abs(this.num(args[0]) - this.num(args[1])) <= 1e-9 * Math.max(Math.abs(this.num(args[0])), Math.abs(this.num(args[1])))) },
        pi: float(Math.PI), e: float(Math.E), inf: float(Infinity), nan: float(NaN),
      },
    }
  }

  legacySeed(v: Value | undefined): number | number[] {
    if (!v || v.k === 'none') return Math.floor(this.rng.next() * 4294967296)
    const one = v.k === 'int' || v.k === 'bool' || (v.k === 'array' && v.a.size === 1 && !isFloatD(v.a.dtype))
    const words = one ? [this.toInt(v.k === 'array' ? fromScalar(v.a.values()[0], 'int64') : v)] : toArray(v).values()
    if (!one && (v.k === 'float' || (v.k === 'array' && isFloatD(v.a.dtype)))) throw this.err('TypeError', "Cannot cast scalar from dtype('float64') to dtype('int64') according to the rule 'safe'")
    if (!words.length) throw this.err('ValueError', 'Seed must be non-empty')
    if (words.some((w) => !Number.isInteger(w) || w < 0 || w > 4294967295)) throw this.err('ValueError', 'Seed must be between 0 and 2**32 - 1')
    return one ? words[0] : words
  }

  makeRandom(): Value {
    const r = this.rng
    const gauss = (name: string): Value => ({ k: 'fn', name, call: (args, kw) => {
      const mu = kw.mu ?? args[0]
      const sigma = kw.sigma ?? args[1]
      const z = Math.sqrt(-2 * Math.log(1 - r.next())) * Math.cos(2 * Math.PI * r.next())
      return float((mu ? this.num(mu) : 0) + (sigma ? this.num(sigma) : 1) * z)
    } })
    return {
      k: 'module', name: 'random', attrs: {
        seed: { k: 'fn', name: 'seed', call: (args) => { r.seed(args[0] ? this.toInt(args[0]) : 0); return NONE } },
        random: { k: 'fn', name: 'random', call: () => float(r.next()) },
        uniform: { k: 'fn', name: 'uniform', call: (args) => float(this.num(args[0]) + (this.num(args[1]) - this.num(args[0])) * r.next()) },
        randint: { k: 'fn', name: 'randint', call: (args) => int(this.toInt(args[0]) + Math.floor(r.next() * (this.toInt(args[1]) - this.toInt(args[0]) + 1))) },
        choice: { k: 'fn', name: 'choice', call: (args) => { const it = this.iterate(args[0]); return it[Math.floor(r.next() * it.length)] } },
        randrange: { k: 'fn', name: 'randrange', call: (args) => {
          const [a, b, st] = args.length === 1 ? [0, this.toInt(args[0]), 1] : [this.toInt(args[0]), this.toInt(args[1]), args[2] ? this.toInt(args[2]) : 1]
          const n = Math.ceil((b - a) / st)
          if (n <= 0) throw this.err('ValueError', `empty range in randrange(${a}, ${b})`)
          return int(a + st * Math.floor(r.next() * n))
        } },
        choices: { k: 'fn', name: 'choices', call: (args, kw) => {
          const it = this.iterate(args[0])
          const n = kw.k ? this.toInt(kw.k) : 1
          const wv = kw.weights ?? args[1]
          const ws = wv && wv.k !== 'none' ? this.iterate(wv).map((x) => this.num(x)) : it.map(() => 1)
          const total = ws.reduce((a, b) => a + b, 0)
          return { k: 'list', items: Array.from({ length: n }, () => {
            let x = r.next() * total
            for (let i = 0; i < it.length; i++) {
              x -= ws[i]
              if (x < 0) return it[i]
            }
            return it[it.length - 1]
          }) }
        } },
        sample: { k: 'fn', name: 'sample', call: (args, kw) => {
          const it = [...this.iterate(args[0])]
          const n = this.toInt(kw.k ?? args[1])
          if (n > it.length || n < 0) throw this.err('ValueError', 'Sample larger than population or is negative')
          for (let i = 0; i < n; i++) {
            const j = i + Math.floor(r.next() * (it.length - i))
            ;[it[i], it[j]] = [it[j], it[i]]
          }
          return { k: 'list', items: it.slice(0, n) }
        } },
        gauss: gauss('gauss'), normalvariate: gauss('normalvariate'),
        betavariate: { k: 'fn', name: 'betavariate', call: (args) => {
          const [x, y] = [gammaDraw(r, this.num(args[0])), gammaDraw(r, this.num(args[1]))]
          return float(x / (x + y))
        } },
        expovariate: { k: 'fn', name: 'expovariate', call: (args) => float(-Math.log(1 - r.next()) / this.num(args[0])) },
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
  instRepr = (v) => {
    const f = it.classAttr(v.cls, '__repr__')
    return f ? str(it.call(f, [v])) : undefined
  }
  instHash = (v) => {
    const same = v.cls.meta?.scalar?.(v)
    if (same) return PyDict.hash(same)
    const f = it.classAttr(v.cls, '__hash__')
    if (f) return `h${it.num(it.call(f, [v]))}`
    // defining __eq__ without __hash__ makes a class unhashable
    if (it.classAttr(v.cls, '__eq__') && !v.cls.builtin) throw new PyError('TypeError', `unhashable type: '${v.cls.name}'`)
    return undefined
  }
  resetPrintOptions()
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
    // sys.exit() is quiet; argparse gives up with a message for a terminal, and a page has no other place to show it
    if (e instanceof PyError && e.pyType === 'SystemExit') {
      const said = (e.exc as InstValue | undefined)?.state.said
      error = typeof said === 'string' ? { type: 'SystemExit', message: said, line: e.line ?? it.line } : null
    } else if (e instanceof PyError) error = { type: e.pyType, message: e.message, line: e.line ?? it.line }
    else if (e instanceof BreakSig || e instanceof ContinueSig) error = { type: 'SyntaxError', message: `'${e instanceof BreakSig ? 'break' : 'continue'}' outside loop`, line: it.line }
    else if (e instanceof RangeError) error = { type: 'RecursionError', message: 'expression too deeply nested', line: it.line }
    else error = { type: 'InternalError', message: e instanceof Error ? e.message : String(e), line: it.line }
  } finally {
    setSizeLimit(prevLimit)
    instRepr = null
    instHash = null
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
