/**
 * The tensor engine behind the sandbox's torch / tensorflow / keras: values in an NDArray (so tensors can share
 * memory with NumPy arrays, as torch.from_numpy does), PyTorch's dtype rules, and reverse-mode autograd.
 *
 * Every op computes on plain C-order number arrays and, when gradients are on and an input requires them, records a
 * Node whose backward maps the output gradient to one gradient per input (already reduced to that input's shape).
 * Layer kernels (linear, conv2d, max_pool2d …) are single ops with PyTorch's grad_fn names, so the recorded graph
 * reads like the real one (AddmmBackward0, ConvolutionBackward0 …).
 */
import { NDArray, PyError, broadcastShapes, cStrides, checkSize, getIndex, prod, transpose as ndTranspose } from './ndarray'
import type { DType, IndexItem } from './ndarray'
import type { SandboxRandom } from './ndops'

export type TDType = 'float16' | 'float32' | 'float64' | 'int64' | 'int32' | 'int16' | 'int8' | 'uint8' | 'bool'

export const isFloatT = (d: TDType) => d === 'float32' || d === 'float64' || d === 'float16'
// .numpy() hands out this array, so it carries the type NumPy would report
const storageOf = (d: TDType): DType => (d === 'float32' || d === 'float64' || d === 'bool' || d === 'uint8' ? d : d === 'float16' ? 'float32' : 'int64')
const err = (type: string, msg: string) => new PyError(type, msg)

/** torch.set_default_dtype: what a Python float and an integer's true division become (reset for each run) */
export const DEFAULTS = { float: 'float32' as TDType }

/** round to IEEE half precision */
const f16round = (v: number): number => {
  if (!Number.isFinite(v) || v === 0) return v
  const a = Math.abs(v)
  if (a >= 65520) return Math.sign(v) * Infinity
  const ulp = 2 ** Math.max(Math.floor(Math.log2(a)) - 10, -24)
  const q = a / ulp
  const r = Math.floor(q)
  const up = q - r > 0.5 || (q - r === 0.5 && r % 2 === 1)
  return Math.sign(v) * (up ? r + 1 : r) * ulp
}

/** round to the dtype: float32 storage keeps float32 precision, as torch.tensor(0.1).item() shows */
export const castT = (v: number, d: TDType): number => {
  switch (d) {
    case 'float32': return Math.fround(v)
    case 'float64': return v
    case 'float16': return f16round(v)
    case 'bool': return v ? 1 : 0
    case 'uint8': return ((Math.trunc(v) % 256) + 256) % 256
    case 'int8': return ((Math.trunc(v) + 128) % 256 + 256) % 256 - 128
    case 'int16': return ((Math.trunc(v) + 32768) % 65536 + 65536) % 65536 - 32768
    case 'int32': return Math.trunc(v) | 0
    default: return Number.isFinite(v) ? Math.trunc(v) : 0
  }
}

export interface GradNode {
  /** PyTorch's grad_fn class name: MulBackward0, AddmmBackward0 … */
  name: string
  inputs: (TT | null)[]
  /** output gradient (C-order, output shape) → one gradient per input (input shape) */
  backward: (g: number[]) => (number[] | null)[]
  freed: boolean
  /** each input's version when this node was recorded: an in-place op later on must not change what this node meant */
  versions?: number[]
  /** the input of a step built from smaller ones (std, var), whose gradient reads it again; with its version then */
  keeps?: [TT, number]
  /** the same step on tensors, itself recorded: what create_graph=True runs. Absent where the sandbox has no such rule */
  dual?: (g: TT) => (TT | null)[]
}

let nextId = 1

/** how many inference_mode blocks are open: tensors made inside are marked */
export const inferenceMode = { depth: 0 }

export class TT {
  a: NDArray
  readonly inference = inferenceMode.depth > 0
  dtype: TDType
  requiresGrad = false
  grad: TT | null = null
  fn: GradNode | null = null
  retain = false
  /** nn.Parameter */
  isParam = false
  /** bumped by every in-place write; past[v] stands in for version v in the graphs recorded before that write */
  version = 0
  past: TT[] | null = null
  /** register_hook: called with this tensor's gradient during backward; a returned value replaces it */
  hooks: ((g: TT) => TT | null)[] | null = null
  /** the tensor whose memory this one views, and that tensor's version when the view was last in step with it */
  base: TT | null = null
  baseVersion = 0
  readonly id = nextId++
  constructor(a: NDArray, dtype: TDType) {
    this.a = a
    this.dtype = dtype
  }
  get shape(): number[] { return this.a.shape }
  get ndim(): number { return this.a.ndim }
  get size(): number { return this.a.size }
  get isLeaf(): boolean { return this.fn === null }
  values(): number[] { return this.a.values() }
  static of(values: number[], shape: number[], dtype: TDType): TT {
    return new TT(NDArray.create(values.map((v) => castT(v, dtype)), [...shape], storageOf(dtype)), dtype)
  }
  static scalar(v: number, dtype: TDType): TT { return TT.of([v], [], dtype) }
  /** values already rounded to the dtype (no copy) */
  static raw(values: number[], shape: number[], dtype: TDType): TT {
    checkSize(shape)
    return new TT(new NDArray(values, [...shape], cStrides(shape), 0, storageOf(dtype), null), dtype)
  }
}

/** grad mode and the random source, one per run */
export class Engine {
  grad = true
  rng: SandboxRandom
  constructor(rng: SandboxRandom) {
    this.rng = rng
    // one engine per run: what a previous run set does not carry over, for torch or for keras
    DEFAULTS.float = 'float32'
  }
}

const needsGrad = (E: Engine, ...xs: (TT | null)[]) => E.grad && xs.some((x) => x?.requiresGrad)

/** which version of this input the node meant (a node recorded before an in-place write means the old value) */
export const inputOf = (n: GradNode, i: number): TT | null => {
  const x = n.inputs[i]
  if (!x) return null
  const v = n.versions?.[i] ?? x.version
  return v === x.version ? x : x.past?.[v] ?? x
}

/** the flat positions of a view's elements inside its base */
const positionsIn = (base: TT, view: TT): number[] => {
  const flatOf = new Map(base.a.addresses().map((x, i) => [x, i]))
  return view.a.addresses().map((x) => flatOf.get(x)!)
}

const viewNode = (E: Engine, x: TT, base: TT): void => {
  const pos = positionsIn(base, x)
  x.fn = { name: 'AsStridedBackward0', inputs: [base], versions: [base.version], freed: false, backward: (g) => {
    const ga = new Array<number>(base.size).fill(0)
    pos.forEach((p, i) => (ga[p] += g[i]))
    return [ga]
  } }
  x.requiresGrad = true
  void E
}

/** a view whose base was written in place since: its history now comes from the base, as PyTorch regenerates grad_fn */
export const refreshView = (E: Engine, x: TT): void => {
  const b = x.base
  if (!b || x.baseVersion === b.version) return
  x.baseVersion = b.version
  if (E.grad && b.requiresGrad) viewNode(E, x, b)
}

/** attach a grad_fn when any input requires grad */
export const record = (E: Engine, out: TT, name: string, inputs: (TT | null)[], backward: GradNode['backward'], dual?: GradNode['dual']): TT => {
  for (const x of inputs) if (x?.base) refreshView(E, x)
  if (needsGrad(E, ...inputs)) {
    out.requiresGrad = true
    out.fn = { name, inputs, backward, freed: false, versions: inputs.map((x) => x?.version ?? 0), dual }
  }
  return out
}

export const renamed = (t: TT, name: string): TT => {
  if (t.fn) t.fn.name = name
  return t
}

/** undo broadcasting on a gradient: summed over the axes that were expanded */
const sumTo = (E: Engine, g: TT, shape: number[]): TT => {
  if (g.shape.join() === shape.join()) return g
  const lead = g.ndim - shape.length
  const axes = g.shape.map((_, k) => k).filter((k) => k < lead || (shape[k - lead] === 1 && g.shape[k] !== 1))
  return reshapeT(E, axes.length ? reduceSum(E, g, axes, true, false) : g, shape, false)
}

/** the transpose of a copy: each output cell's gradient goes back to the cell it came from */
const accumulateBack = (E: Engine, g: TT, src: ArrayLike<number>, shape: number[]): TT => {
  const gv = g.values()
  const out = new Array<number>(prod(shape)).fill(0)
  for (let i = 0; i < src.length; i++) if (src[i] >= 0) out[src[i]] += gv[i]
  return record(E, TT.of(out, shape, g.dtype), 'IndexAddBackward0', [g], (gg) => [Array.from({ length: src.length }, (_, i) => (src[i] >= 0 ? gg[src[i]] : 0))], (gg) => [takeT(E, gg, src, g.shape, 'IndexBackward0')])
}

const isView = (out: TT, a: TT): TT => {
  out.base = a.base ?? a
  out.baseVersion = out.base.version
  return out
}

// ---------------------------------------------------------------- dtype promotion

const RANK: Record<TDType, number> = { bool: 0, uint8: 1, int8: 1, int16: 2, int32: 3, int64: 4, float16: 5, float32: 6, float64: 7 }

/** result dtype of a ⊕ b; weak = a Python scalar (it never widens a tensor of its own kind) */
export const promote = (a: TDType, b: TDType, aWeak = false, bWeak = false): TDType => {
  if (aWeak && !bWeak) return isFloatT(a) && !isFloatT(b) ? DEFAULTS.float : b === 'bool' && a !== 'bool' ? (isFloatT(a) ? DEFAULTS.float : 'int64') : b
  if (bWeak && !aWeak) return promote(b, a, true, false)
  if ((a === 'uint8' && b === 'int8') || (a === 'int8' && b === 'uint8')) return 'int16'
  return RANK[a] >= RANK[b] ? a : b
}

// ---------------------------------------------------------------- broadcasting helpers

/** for each element of the output, the flat index of the broadcast input element */
export const bcastIndex = (inShape: number[], outShape: number[]): Int32Array => {
  const n = prod(outShape)
  const idx = new Int32Array(n)
  const nd = outShape.length
  const off = nd - inShape.length
  const inSt = cStrides(inShape)
  const outSt = cStrides(outShape)
  for (let f = 0; f < n; f++) {
    let r = f
    let src = 0
    for (let d = 0; d < nd; d++) {
      const i = Math.floor(r / outSt[d])
      r -= i * outSt[d]
      const k = d - off
      if (k >= 0 && inShape[k] !== 1) src += i * inSt[k]
    }
    idx[f] = src
  }
  return idx
}

/** sum a gradient of the broadcast shape back to an input's shape */
const unbroadcast = (g: number[], idx: Int32Array, inSize: number): number[] => {
  const out = new Array<number>(inSize).fill(0)
  for (let i = 0; i < g.length; i++) out[idx[i]] += g[i]
  return out
}

const shapeOfBcast = (a: number[], b: number[]): number[] => {
  try {
    return broadcastShapes([a, b])
  } catch {
    // PyTorch's wording
    const nd = Math.max(a.length, b.length)
    for (let d = 1; d <= nd; d++) {
      const x = a[a.length - d] ?? 1
      const y = b[b.length - d] ?? 1
      if (x !== y && x !== 1 && y !== 1) throw err('RuntimeError', `The size of tensor a (${x}) must match the size of tensor b (${y}) at non-singleton dimension ${nd - d}`)
    }
    throw err('RuntimeError', 'shapes cannot be broadcast')
  }
}

// ---------------------------------------------------------------- elementwise

export type BinOpT = '+' | '-' | '*' | '/' | '**' | '//' | '%' | 'maximum' | 'minimum'

const BIN_NAME: Record<BinOpT, string> = { '+': 'AddBackward0', '-': 'SubBackward0', '*': 'MulBackward0', '/': 'DivBackward0', '**': 'PowBackward1', '//': 'FloorDivBackward', '%': 'RemainderBackward', maximum: 'MaximumBackward0', minimum: 'MinimumBackward0' }

const applyBin = (op: BinOpT, x: number, y: number): number => {
  switch (op) {
    case '+': return x + y
    case '-': return x - y
    case '*': return x * y
    case '/': return x / y
    case '**': return x ** y
    case '//': return Math.floor(x / y)
    case '%': return x - Math.floor(x / y) * y
    case 'maximum': return Math.max(x, y)
    case 'minimum': return Math.min(x, y)
  }
}

export function binary(E: Engine, op: BinOpT, a: TT, b: TT, aWeak = false, bWeak = false): TT {
  const shape = shapeOfBcast(a.shape, b.shape)
  let dtype = promote(a.dtype, b.dtype, aWeak, bWeak)
  if (op === '/' && !isFloatT(dtype)) dtype = DEFAULTS.float
  if ((op === '//' || op === '%') && (a.dtype === 'bool' || b.dtype === 'bool')) throw err('RuntimeError', 'not supported for bool tensors')
  const ia = bcastIndex(a.shape, shape)
  const ib = bcastIndex(b.shape, shape)
  const av = a.values()
  const bv = b.values()
  const n = prod(shape)
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) out[i] = applyBin(op, av[ia[i]], bv[ib[i]])
  if (!isFloatT(dtype) && out.some((v) => !Number.isFinite(v))) {
    if (op === '//' || op === '%') throw err('RuntimeError', 'ZeroDivisionError')
  }
  const res = TT.of(out, shape, dtype)
  if (op === '//' || op === '%') return res
  const outV = res.values()
  return record(E, res, op === '**' && b.size === 1 && bWeak ? 'PowBackward0' : BIN_NAME[op], [a, b], (g) => {
    const ga = new Array<number>(n)
    const gb = new Array<number>(n)
    for (let i = 0; i < n; i++) {
      const x = av[ia[i]]
      const y = bv[ib[i]]
      switch (op) {
        case '+': ga[i] = g[i]; gb[i] = g[i]; break
        case '-': ga[i] = g[i]; gb[i] = -g[i]; break
        case '*': ga[i] = g[i] * y; gb[i] = g[i] * x; break
        case '/': ga[i] = g[i] / y; gb[i] = (-g[i] * x) / (y * y); break
        case '**': ga[i] = g[i] * y * x ** (y - 1); gb[i] = x > 0 ? g[i] * outV[i] * Math.log(x) : 0; break
        case 'maximum': ga[i] = x >= y ? g[i] : 0; gb[i] = x >= y ? 0 : g[i]; break
        case 'minimum': ga[i] = x <= y ? g[i] : 0; gb[i] = x <= y ? 0 : g[i]; break
        default: ga[i] = 0; gb[i] = 0
      }
    }
    return [a.requiresGrad ? unbroadcast(ga, ia, a.size) : null, b.requiresGrad ? unbroadcast(gb, ib, b.size) : null]
  }, (g) => {
    const back = (x: TT, v: () => TT) => (x.requiresGrad ? sumTo(E, v(), x.shape) : null)
    const mul = (p: TT, q: TT) => binary(E, '*', p, q)
    switch (op) {
      case '+': return [back(a, () => g), back(b, () => g)]
      case '-': return [back(a, () => g), back(b, () => unary(E, 'neg', g))]
      case '*': return [back(a, () => mul(g, b)), back(b, () => mul(g, a))]
      case '/': return [back(a, () => binary(E, '/', g, b)), back(b, () => unary(E, 'neg', binary(E, '/', mul(g, a), mul(b, b))))]
      case '**': return [
        back(a, () => mul(g, mul(b, binary(E, '**', a, binary(E, '-', b, TT.scalar(1, b.dtype), false, true), false, bWeak)))),
        back(b, () => mul(g, mul(res, TT.of(av.map((x) => (x > 0 ? Math.log(x) : 0)), a.shape, dtype)))),
      ]
      case 'maximum': case 'minimum': {
        const firstWins = TT.of(Array.from({ length: n }, (_, i) => ((op === 'maximum' ? av[ia[i]] >= bv[ib[i]] : av[ia[i]] <= bv[ib[i]]) ? 1 : 0)), shape, dtype)
        return [back(a, () => mul(g, firstWins)), back(b, () => mul(g, binary(E, '-', TT.scalar(1, dtype), firstWins, true, false)))]
      }
      default: return [null, null]
    }
  })
}

export type CmpOpT = '<' | '<=' | '>' | '>=' | '==' | '!='

export function compare(a: TT, b: TT, op: CmpOpT): TT {
  const shape = shapeOfBcast(a.shape, b.shape)
  const ia = bcastIndex(a.shape, shape)
  const ib = bcastIndex(b.shape, shape)
  const av = a.values()
  const bv = b.values()
  const out = Array.from({ length: prod(shape) }, (_, i) => {
    const x = av[ia[i]]
    const y = bv[ib[i]]
    switch (op) {
      case '<': return x < y ? 1 : 0
      case '<=': return x <= y ? 1 : 0
      case '>': return x > y ? 1 : 0
      case '>=': return x >= y ? 1 : 0
      case '==': return x === y ? 1 : 0
      default: return x !== y ? 1 : 0
    }
  })
  return TT.of(out, shape, 'bool')
}

export type UnOpT = 'neg' | 'exp' | 'log' | 'sqrt' | 'abs' | 'relu' | 'sigmoid' | 'tanh' | 'square' | 'sin' | 'cos'

const UN_NAME: Record<UnOpT, string> = { neg: 'NegBackward0', exp: 'ExpBackward0', log: 'LogBackward0', sqrt: 'SqrtBackward0', abs: 'AbsBackward0', relu: 'ReluBackward0', sigmoid: 'SigmoidBackward0', tanh: 'TanhBackward0', square: 'PowBackward0', sin: 'SinBackward0', cos: 'CosBackward0' }

export function unary(E: Engine, op: UnOpT, a: TT): TT {
  const av = a.values()
  const f = (x: number): number => {
    switch (op) {
      case 'neg': return -x
      case 'exp': return Math.exp(x)
      case 'log': return Math.log(x)
      case 'sqrt': return Math.sqrt(x)
      case 'abs': return Math.abs(x)
      case 'relu': return x > 0 ? x : 0
      case 'sigmoid': return 1 / (1 + Math.exp(-x))
      case 'tanh': return Math.tanh(x)
      case 'square': return x * x
      case 'sin': return Math.sin(x)
      case 'cos': return Math.cos(x)
    }
  }
  const keepsInt = op === 'neg' || op === 'abs' || op === 'relu' || op === 'square'
  const dtype = keepsInt || isFloatT(a.dtype) ? (a.dtype === 'bool' ? 'int64' : a.dtype) : DEFAULTS.float
  const out = TT.of(av.map(f), a.shape, dtype)
  const ov = out.values()
  return record(E, out, UN_NAME[op], [a], (g) => [g.map((gi, i) => {
    const x = av[i]
    switch (op) {
      case 'neg': return -gi
      case 'exp': return gi * ov[i]
      case 'log': return gi / x
      case 'sqrt': return gi / (2 * ov[i])
      case 'abs': return gi * Math.sign(x)
      case 'relu': return x > 0 ? gi : 0
      case 'sigmoid': return gi * ov[i] * (1 - ov[i])
      case 'tanh': return gi * (1 - ov[i] * ov[i])
      case 'square': return gi * 2 * x
      case 'sin': return gi * Math.cos(x)
      case 'cos': return -gi * Math.sin(x)
    }
  })], (g) => {
    const mul = (p: TT, q: TT) => binary(E, '*', p, q)
    const num = (v: number) => TT.scalar(v, dtype)
    switch (op) {
      case 'neg': return [unary(E, 'neg', g)]
      case 'exp': return [mul(g, out)]
      case 'log': return [binary(E, '/', g, a)]
      case 'sqrt': return [binary(E, '/', g, mul(num(2), out))]
      case 'abs': return [mul(g, TT.of(av.map(Math.sign), a.shape, dtype))]
      case 'relu': return [mul(g, TT.of(av.map((x) => (x > 0 ? 1 : 0)), a.shape, dtype))]
      case 'sigmoid': return [mul(g, mul(out, binary(E, '-', num(1), out)))]
      case 'tanh': return [mul(g, binary(E, '-', num(1), mul(out, out)))]
      case 'square': return [mul(g, mul(num(2), a))]
      case 'sin': return [mul(g, unary(E, 'cos', a))]
      case 'cos': return [unary(E, 'neg', mul(g, unary(E, 'sin', a)))]
    }
  })
}

export function clamp(E: Engine, a: TT, lo: number | null, hi: number | null): TT {
  const av = a.values()
  const out = TT.of(av.map((x) => Math.min(hi ?? Infinity, Math.max(lo ?? -Infinity, x))), a.shape, a.dtype)
  return record(E, out, 'ClampBackward1', [a], (g) => [g.map((gi, i) => ((lo === null || av[i] >= lo) && (hi === null || av[i] <= hi) ? gi : 0))])
}

// ---------------------------------------------------------------- matmul

const dims2 = (s: number[]) => `${s.join('x')}`

export function matmul(E: Engine, a: TT, b: TT, name = 'MmBackward0'): TT {
  const res = matmulKernel(E, a, b, name)
  if (res.fn) res.fn.dual = (g) => {
    const a2 = a.ndim === 1 ? reshapeT(E, a, [1, a.size], false) : a
    const b2 = b.ndim === 1 ? reshapeT(E, b, [b.size, 1], false) : b
    const gShape = [...shapeOfBcast(a2.shape.slice(0, -2), b2.shape.slice(0, -2)), a2.shape[a2.ndim - 2], b2.shape[b2.ndim - 1]]
    const g2 = reshapeT(E, g, gShape, false)
    return [
      a.requiresGrad ? reshapeT(E, sumTo(E, matmul(E, g2, transposeT(E, b2, -2, -1)), a2.shape), a.shape, false) : null,
      b.requiresGrad ? reshapeT(E, sumTo(E, matmul(E, transposeT(E, a2, -2, -1), g2), b2.shape), b.shape, false) : null,
    ]
  }
  return res
}

function matmulKernel(E: Engine, a: TT, b: TT, name: string): TT {
  if (a.ndim === 0 || b.ndim === 0) throw err('RuntimeError', `both arguments to matmul need to be at least 1D, but they are ${a.ndim}D and ${b.ndim}D`)
  const dtype = promote(a.dtype, b.dtype)
  const av = a.values()
  const bv = b.values()
  if (a.ndim === 1 && b.ndim === 1) {
    if (a.size !== b.size) throw err('RuntimeError', `inconsistent tensor size, expected tensor [${a.size}] and src [${b.size}] to have the same number of elements, but got ${a.size} and ${b.size} elements respectively`)
    const out = TT.of([av.reduce((s, x, i) => s + x * bv[i], 0)], [], dtype)
    return record(E, out, 'DotBackward0', [a, b], (g) => [bv.map((y) => y * g[0]), av.map((x) => x * g[0])])
  }
  // promote 1-D operands, broadcast the batch dimensions
  const as = a.ndim === 1 ? [1, a.shape[0]] : a.shape
  const bs = b.ndim === 1 ? [b.shape[0], 1] : b.shape
  const [n, k] = as.slice(-2)
  const [k2, m] = bs.slice(-2)
  if (k !== k2) {
    if (a.ndim === 2 && b.ndim === 1) throw err('RuntimeError', `size mismatch, got input (${a.shape[0]}), mat (${a.shape[0]}x${a.shape[1]}), vec (${b.shape[0]})`)
    throw err('RuntimeError', a.ndim <= 2 && b.ndim <= 2 ? `mat1 and mat2 shapes cannot be multiplied (${dims2(as)} and ${dims2(bs)})` : `Expected size for first two dimensions of batch2 tensor to be: [${bs.slice(0, -2).join(', ')}, ${k}] but got: [${bs.slice(0, -2).join(', ')}, ${k2}].`)
  }
  const batch = shapeOfBcast(as.slice(0, -2), bs.slice(0, -2))
  const nb = prod(batch)
  const ia = bcastIndex(as.slice(0, -2), batch)
  const ib = bcastIndex(bs.slice(0, -2), batch)
  const out = new Array<number>(nb * n * m).fill(0)
  for (let t = 0; t < nb; t++) {
    const ao = ia[t] * n * k
    const bo = ib[t] * k * m
    const oo = t * n * m
    for (let i = 0; i < n; i++) {
      for (let p = 0; p < k; p++) {
        const x = av[ao + i * k + p]
        if (x === 0) continue
        for (let j = 0; j < m; j++) out[oo + i * m + j] += x * bv[bo + p * m + j]
      }
    }
  }
  let shape = [...batch, n, m]
  if (a.ndim === 1) shape = [...batch, m]
  else if (b.ndim === 1) shape = [...batch, n]
  const res = TT.of(out, shape, dtype)
  return record(E, res, a.ndim > 2 || b.ndim > 2 ? 'BmmBackward0' : a.ndim === 1 || b.ndim === 1 ? 'MvBackward0' : name, [a, b], (g) => {
    const ga = new Array<number>(prod(as.slice(0, -2)) * n * k).fill(0)
    const gb = new Array<number>(prod(bs.slice(0, -2)) * k * m).fill(0)
    for (let t = 0; t < nb; t++) {
      const ao = ia[t] * n * k
      const bo = ib[t] * k * m
      const oo = t * n * m
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) {
          const gij = g[oo + i * m + j]
          if (gij === 0) continue
          for (let p = 0; p < k; p++) {
            ga[ao + i * k + p] += gij * bv[bo + p * m + j]
            gb[bo + p * m + j] += gij * av[ao + i * k + p]
          }
        }
      }
    }
    return [ga, gb]
  })
}

// ---------------------------------------------------------------- reductions

export const normDim = (d: number, nd: number): number => {
  const r = d < 0 ? d + Math.max(nd, 1) : d
  if (r < 0 || r >= Math.max(nd, 1)) throw err('IndexError', `Dimension out of range (expected to be in range of [${-Math.max(nd, 1)}, ${Math.max(nd, 1) - 1}], but got ${d})`)
  return r
}

/** output position of every input element when dims are reduced */
const reduceMap = (shape: number[], dims: number[]): { outShapeKeep: number[]; map: Int32Array; outSize: number } => {
  const keep = shape.map((s, d) => (dims.includes(d) ? 1 : s))
  const st = cStrides(shape)
  const ost = cStrides(keep)
  const n = prod(shape)
  const map = new Int32Array(n)
  for (let f = 0; f < n; f++) {
    let r = f
    let o = 0
    for (let d = 0; d < shape.length; d++) {
      const i = Math.floor(r / st[d])
      r -= i * st[d]
      if (!dims.includes(d)) o += i * ost[d]
    }
    map[f] = o
  }
  return { outShapeKeep: keep, map, outSize: prod(keep) }
}

export function reduceSum(E: Engine, a: TT, dims: number[] | null, keepdim: boolean, mean: boolean): TT {
  const ds = dims === null ? a.shape.map((_, i) => i) : [...new Set(dims.map((d) => normDim(d, a.ndim)))]
  if (mean && !isFloatT(a.dtype)) throw err('RuntimeError', `mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: ${typeLabel(a.dtype)}`)
  const { outShapeKeep, map, outSize } = reduceMap(a.shape, ds)
  const av = a.values()
  const out = new Array<number>(outSize).fill(0)
  av.forEach((v, i) => (out[map[i]] += v))
  const count = av.length / Math.max(1, outSize)
  if (mean) for (let i = 0; i < outSize; i++) out[i] /= count
  const shape = keepdim ? outShapeKeep : a.shape.filter((_, d) => !ds.includes(d))
  const dtype = mean ? a.dtype : a.dtype === 'bool' || a.dtype === 'int32' || a.dtype === 'uint8' ? 'int64' : a.dtype
  const res = TT.of(out, shape, dtype)
  const dualSum = (g: TT): (TT | null)[] => {
    const spread = expandT(E, reshapeT(E, g, outShapeKeep, false), a.shape)
    return [mean ? binary(E, '/', spread, TT.scalar(count, a.dtype), false, true) : spread]
  }
  return record(E, res, mean ? 'MeanBackward' + (dims === null ? '0' : '1') : 'SumBackward' + (dims === null ? '0' : '1'), [a], (g) => [Array.from({ length: av.length }, (_, i) => g[map[i]] / (mean ? count : 1))], dualSum)
}

/** max / min along a dim: values and indices (torch.return_types.max) */
export function maxAlong(E: Engine, a: TT, dim: number, keepdim: boolean, min = false): { values: TT; indices: TT } {
  const d = normDim(dim, a.ndim)
  const { outShapeKeep, map, outSize } = reduceMap(a.shape, [d])
  const av = a.values()
  const best = new Array<number>(outSize).fill(min ? Infinity : -Infinity)
  const arg = new Array<number>(outSize).fill(0)
  const src = new Array<number>(outSize).fill(-1)
  const st = cStrides(a.shape)
  av.forEach((v, i) => {
    const o = map[i]
    if (min ? v < best[o] : v > best[o]) {
      best[o] = v
      src[o] = i
      arg[o] = Math.floor(i / st[d]) % a.shape[d]
    }
  })
  const shape = keepdim ? outShapeKeep : a.shape.filter((_, k) => k !== d)
  const values = record(E, TT.of(best, shape, a.dtype), min ? 'MinBackward0' : 'MaxBackward0', [a], (g) => {
    const ga = new Array<number>(av.length).fill(0)
    src.forEach((s, o) => (ga[s] += g[o]))
    return [ga]
  })
  return { values, indices: TT.of(arg, shape, 'int64') }
}

export function maxAll(E: Engine, a: TT, min = false): TT {
  if (!a.size) throw err('RuntimeError', 'max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the \'dim\' argument.')
  const av = a.values()
  let k = 0
  av.forEach((v, i) => { if (min ? v < av[k] : v > av[k]) k = i })
  return record(E, TT.of([av[k]], [], a.dtype), min ? 'MinBackward1' : 'MaxBackward1', [a], (g) => {
    const ga = new Array<number>(av.length).fill(0)
    ga[k] = g[0]
    return [ga]
  })
}

export function argmax(a: TT, dim: number | null, keepdim: boolean, min = false): TT {
  if (dim === null) {
    const av = a.values()
    let k = 0
    av.forEach((v, i) => { if (min ? v < av[k] : v > av[k]) k = i })
    return TT.of([k], keepdim ? a.shape.map(() => 1) : [], 'int64')
  }
  const E = new Engine(null as unknown as SandboxRandom)
  E.grad = false
  return maxAlong(E, a, dim, keepdim, min).indices
}

/** var / std with Bessel's correction by default (torch's unbiased=True), built from differentiable ops */
export function variance(E: Engine, a: TT, dims: number[] | null, keepdim: boolean, correction: number, sqrt: boolean): TT {
  if (!isFloatT(a.dtype)) throw err('RuntimeError', 'std and var only support floating point and complex dtypes')
  const mu = reduceSum(E, a, dims, true, true)
  const d = binary(E, '-', a, mu)
  const sq = unary(E, 'square', d)
  const n = dims === null ? a.size : dims.reduce((p, x) => p * a.shape[normDim(x, a.ndim)], 1)
  const s = reduceSum(E, sq, dims, keepdim, false)
  const v = binary(E, '/', s, TT.scalar(Math.max(0, n - correction), 'float32'), false, true)
  return sqrt ? unary(E, 'sqrt', v) : v
}

// ---------------------------------------------------------------- shape and memory

/** -1 in a target shape: work it out */
export const resolveShape = (size: number, target: number[], from: number[]): number[] => {
  const neg = target.filter((d) => d === -1).length
  if (neg > 1) throw err('RuntimeError', 'only one dimension can be inferred')
  const known = target.reduce((p, d) => (d === -1 ? p : p * d), 1)
  const out = target.map((d) => (d === -1 ? (known === 0 ? 0 : size / known) : d))
  if (prod(out) !== size || out.some((d) => !Number.isInteger(d) || d < 0)) throw err('RuntimeError', `shape '[${target.join(', ')}]' is invalid for input of size ${size}`)
  void from
  return out
}

/** view (shares memory; the tensor must be contiguous) or reshape (copies when it has to) */
export function reshapeT(E: Engine, a: TT, target: number[], mustView: boolean): TT {
  const shape = resolveShape(a.size, target, a.shape)
  let arr: NDArray
  if (a.a.isCContiguous()) arr = a.a.view(shape, cStrides(shape), a.a.offset)
  else if (mustView) throw err('RuntimeError', "view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.")
  else arr = NDArray.create(a.values(), shape, a.a.dtype)
  const out = new TT(arr, a.dtype)
  const view = a.a.isCContiguous()
  if (view) isView(out, a)
  return record(E, out, view ? 'ViewBackward0' : 'ReshapeAliasBackward0', [a], (g) => [g], (g) => [reshapeT(E, g, a.shape, false)])
}

export function permuteT(E: Engine, a: TT, perm: number[], name = 'PermuteBackward0'): TT {
  const p = perm.map((d) => normDim(d, a.ndim))
  if (p.length === a.ndim && new Set(p).size !== p.length) throw err('RuntimeError', 'permute(): duplicate dims are not allowed.')
  if (new Set(p).size !== p.length || p.length !== a.ndim) throw err('RuntimeError', `permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = ${a.ndim} is not equal to len(dims) = ${p.length}`)
  const out = isView(new TT(ndTranspose(a.a, p), a.dtype), a)
  // where each output element comes from, to send gradients back
  const ids = new TT(ndTranspose(NDArray.create(Array.from({ length: a.size }, (_, i) => i), a.shape, 'int64'), p), 'int64').values()
  return record(E, out, name, [a], (g) => {
    const ga = new Array<number>(a.size)
    ids.forEach((src, i) => (ga[src] = g[i]))
    return [ga]
  }, (g) => [accumulateBack(E, g, ids, a.shape)])
}

export function transposeT(E: Engine, a: TT, d0: number, d1: number): TT {
  const p = a.shape.map((_, i) => i)
  const x = normDim(d0, a.ndim)
  const y = normDim(d1, a.ndim)
  ;[p[x], p[y]] = [p[y], p[x]]
  return permuteT(E, a, p, 'TransposeBackward0')
}

export function unsqueezeT(E: Engine, a: TT, dim: number): TT {
  const d = dim < 0 ? dim + a.ndim + 1 : dim
  if (d < 0 || d > a.ndim) throw err('IndexError', `Dimension out of range (expected to be in range of [${-a.ndim - 1}, ${a.ndim}], but got ${dim})`)
  const shape = [...a.shape.slice(0, d), 1, ...a.shape.slice(d)]
  const out = reshapeT(E, a, shape, false)
  if (out.fn) out.fn.name = 'UnsqueezeBackward0'
  return out
}

export function squeezeT(E: Engine, a: TT, dim: number | null): TT {
  const shape = dim === null ? a.shape.filter((s) => s !== 1) : a.shape.filter((s, i) => !(i === normDim(dim, a.ndim) && s === 1))
  const out = reshapeT(E, a, shape, false)
  if (out.fn) out.fn.name = dim === null ? 'SqueezeBackward0' : 'SqueezeBackward1'
  return out
}

export function flattenT(E: Engine, a: TT, start: number, end: number): TT {
  if (a.ndim === 0) return reshapeT(E, a, [1], false)
  const s = normDim(start, a.ndim)
  const e = normDim(end, a.ndim)
  const shape = [...a.shape.slice(0, s), prod(a.shape.slice(s, e + 1)), ...a.shape.slice(e + 1)]
  const out = reshapeT(E, a, shape, false)
  if (out.fn) out.fn.name = 'ViewBackward0'
  return out
}

export function catT(E: Engine, ts: TT[], dim: number, stack = false): TT {
  if (!ts.length) {
    if (stack) throw err('RuntimeError', 'stack expects a non-empty TensorList')
    throw err('ValueError', 'torch.cat(): expected a non-empty list of Tensors')
  }
  if (stack) {
    const bad = ts.findIndex((t) => t.shape.join() !== ts[0].shape.join())
    if (bad >= 0) throw err('RuntimeError', `stack expects each tensor to be equal size, but got [${ts[0].shape.join(', ')}] at entry 0 and [${ts[bad].shape.join(', ')}] at entry ${bad}`)
  }
  const parts = stack ? ts.map((t) => unsqueezeT({ grad: false } as Engine, t, dim)) : ts
  const nd = parts[0].ndim
  if (!stack && !nd) throw err('RuntimeError', 'zero-dimensional tensor (at position 0) cannot be concatenated')
  const d = normDim(dim, nd)
  parts.forEach((t, k) => {
    if (t.ndim !== nd) throw err('RuntimeError', `Tensors must have same number of dimensions: got ${nd} and ${t.ndim}`)
    const at = t.shape.findIndex((s, i) => i !== d && s !== parts[0].shape[i])
    if (at >= 0) throw err('RuntimeError', `Sizes of tensors must match except in dimension ${d}. Expected size ${parts[0].shape[at]} but got size ${t.shape[at]} for tensor number ${k} in the list.`)
  })
  const shape = [...parts[0].shape]
  shape[d] = parts.reduce((s, t) => s + t.shape[d], 0)
  const outer = prod(shape.slice(0, d))
  const inner = prod(shape.slice(d + 1))
  const vals = parts.map((t) => t.values())
  const out: number[] = []
  const owners: [number, number][] = []
  for (let o = 0; o < outer; o++) {
    parts.forEach((t, k) => {
      const len = t.shape[d] * inner
      for (let i = 0; i < len; i++) {
        out.push(vals[k][o * len + i])
        owners.push([k, o * len + i])
      }
    })
  }
  const dtype = parts.reduce<TDType>((acc, t) => promote(acc, t.dtype), parts[0].dtype)
  return record(E, TT.of(out, shape, dtype), stack ? 'StackBackward0' : 'CatBackward0', ts, (g) => {
    const gs = ts.map((t) => new Array<number>(t.size).fill(0))
    owners.forEach(([k, i], j) => (gs[k][i] += g[j]))
    return gs
  })
}

/** a[idx]: basic indexing gives a view (as in PyTorch), advanced indexing a copy */
export function indexT(E: Engine, a: TT, items: IndexItem[]): TT {
  const { plan, value } = getIndex(a.a, items)
  const arr = typeof value === 'number' ? (plan.view ? a.a.view([], [], plan.addresses[0]) : NDArray.create([value], [], a.a.dtype)) : value
  const out = new TT(arr, a.dtype)
  if (plan.view) isView(out, a)
  if (!needsGrad(E, a)) return out
  const flatOf = new Map(a.a.addresses().map((x, i) => [x, i]))
  const src = plan.addresses.map((x) => flatOf.get(x)!)
  // the grad_fn is the last step PyTorch takes: an int selects, a real slice slices, None unsqueezes, a bare : does nothing
  const last = [...items].reverse().find((it) => it.kind !== 'ellipsis' && !(it.kind === 'slice' && it.start === null && it.stop === null && it.step === null))
  const name = plan.advanced ? 'IndexBackward0' : !last ? 'AliasBackward0' : last.kind === 'int' ? 'SelectBackward0' : last.kind === 'newaxis' ? 'UnsqueezeBackward0' : 'SliceBackward0'
  return record(E, out, name, [a], (g) => {
    const ga = new Array<number>(a.size).fill(0)
    src.forEach((s, i) => (ga[s] += g[i]))
    return [ga]
  }, (g) => [accumulateBack(E, g, src, a.shape)])
}

export function detach(a: TT): TT {
  return new TT(a.a, a.dtype)
}

export function cloneT(E: Engine, a: TT): TT {
  return record(E, TT.of(a.values(), a.shape, a.dtype), 'CloneBackward0', [a], (g) => [g], (g) => [g])
}

export function toDtype(E: Engine, a: TT, dtype: TDType): TT {
  if (dtype === a.dtype) return a
  const out = TT.of(a.values(), a.shape, dtype)
  return isFloatT(dtype) ? record(E, out, 'ToCopyBackward0', [a], (g) => [g], (g) => [toDtype(E, g, a.dtype)]) : out
}

/** write values into a tensor's storage (in-place ops, x[idx] = v, optimiser steps) */
export function writeInto(a: TT, values: number[], addresses = a.a.addresses()) {
  addresses.forEach((x, i) => (a.a.data[x] = castT(values[i], a.dtype)))
}

// ---------------------------------------------------------------- neural-network kernels

export function linear(E: Engine, x: TT, w: TT, b: TT | null): TT {
  const res = linearKernel(E, x, w, b)
  if (res.fn) res.fn.dual = (g) => {
    const x2 = x.ndim === 1 ? reshapeT(E, x, [1, x.size], false) : x
    const g2 = reshapeT(E, g, x.ndim === 1 ? [1, g.size] : g.shape, false)
    return [
      x.requiresGrad ? reshapeT(E, matmul(E, g2, w), x.shape, false) : null,
      w.requiresGrad ? sumTo(E, matmul(E, transposeT(E, g2, -2, -1), x2), w.shape) : null,
      b && b.requiresGrad ? sumTo(E, g2, b.shape) : null,
    ]
  }
  return res
}

function linearKernel(E: Engine, x: TT, w: TT, b: TT | null): TT {
  const inF = w.shape[1]
  const outF = w.shape[0]
  const xin = x.shape[x.ndim - 1]
  const rows = x.ndim === 1 ? 1 : x.size / xin
  if (xin !== inF) throw err('RuntimeError', `mat1 and mat2 shapes cannot be multiplied (${rows}x${xin} and ${inF}x${outF})`)
  const xv = x.values()
  const wv = w.values()
  const bv = b ? b.values() : null
  const out = new Array<number>(rows * outF)
  for (let r = 0; r < rows; r++) {
    for (let o = 0; o < outF; o++) {
      let s = bv ? bv[o] : 0
      const xo = r * inF
      const wo = o * inF
      for (let i = 0; i < inF; i++) s += xv[xo + i] * wv[wo + i]
      out[r * outF + o] = s
    }
  }
  const shape = [...x.shape.slice(0, -1), outF]
  return record(E, TT.of(out, shape, promote(x.dtype, w.dtype)), b ? 'AddmmBackward0' : 'MmBackward0', [x, w, b], (g) => {
    const gx = x.requiresGrad ? new Array<number>(xv.length).fill(0) : null
    const gw = w.requiresGrad ? new Array<number>(wv.length).fill(0) : null
    const gb = b && b.requiresGrad ? new Array<number>(outF).fill(0) : null
    for (let r = 0; r < rows; r++) {
      for (let o = 0; o < outF; o++) {
        const go = g[r * outF + o]
        if (go === 0) continue
        if (gb) gb[o] += go
        const xo = r * inF
        const wo = o * inF
        for (let i = 0; i < inF; i++) {
          if (gx) gx[xo + i] += go * wv[wo + i]
          if (gw) gw[wo + i] += go * xv[xo + i]
        }
      }
    }
    return [gx, gw, gb]
  })
}

export interface Conv2dOpts { stride: [number, number]; padding: [number, number] }

/** out size along one axis: ⌊(n + 2p − k) / s⌋ + 1 */
export const convOut = (n: number, k: number, s: number, p: number) => Math.floor((n + 2 * p - k) / s) + 1

export function conv2d(E: Engine, x0: TT, w: TT, b: TT | null, { stride, padding }: Conv2dOpts): TT {
  const unbatched = x0.ndim === 3
  if (x0.ndim !== 3 && x0.ndim !== 4) throw err('RuntimeError', `Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [${x0.shape.join(', ')}]`)
  const x = unbatched ? reshapeT({ grad: false } as Engine, x0, [1, ...x0.shape], false) : x0
  const [N, C, H, W] = x.shape
  const [O, Cw, KH, KW] = w.shape
  if (C !== Cw) throw err('RuntimeError', `Given groups=1, weight of size [${w.shape.join(', ')}], expected input[${x0.shape.join(', ')}] to have ${Cw} channels, but got ${C} channels instead`)
  const [sh, sw] = stride
  const [ph, pw] = padding
  if (H + 2 * ph < KH || W + 2 * pw < KW) throw err('RuntimeError', `Calculated padded input size per channel: (${H + 2 * ph} x ${W + 2 * pw}). Kernel size: (${KH} x ${KW}). Kernel size can't be greater than actual input size`)
  const Ho = convOut(H, KH, sh, ph)
  const Wo = convOut(W, KW, sw, pw)
  const xv = Float64Array.from(x.values())
  const wv = Float64Array.from(w.values())
  const bv = b ? b.values() : null
  const out = new Float64Array(N * O * Ho * Wo)
  for (let n = 0; n < N; n++) {
    for (let o = 0; o < O; o++) {
      const ob = (n * O + o) * Ho * Wo
      const bias = bv ? bv[o] : 0
      for (let i = 0; i < Ho * Wo; i++) out[ob + i] = bias
      for (let c = 0; c < C; c++) {
        const xb = (n * C + c) * H * W
        const wb = (o * C + c) * KH * KW
        for (let u = 0; u < KH; u++) {
          for (let v = 0; v < KW; v++) {
            const wt = wv[wb + u * KW + v]
            if (wt === 0) continue
            for (let i = 0; i < Ho; i++) {
              const yy = i * sh + u - ph
              if (yy < 0 || yy >= H) continue
              const row = xb + yy * W
              const orow = ob + i * Wo
              for (let j = 0; j < Wo; j++) {
                const xx = j * sw + v - pw
                if (xx >= 0 && xx < W) out[orow + j] += wt * xv[row + xx]
              }
            }
          }
        }
      }
    }
  }
  const shape = unbatched ? [O, Ho, Wo] : [N, O, Ho, Wo]
  const odt = promote(x.dtype, w.dtype)
  const ovals = new Array<number>(out.length)
  for (let i = 0; i < out.length; i++) ovals[i] = odt === 'float32' ? Math.fround(out[i]) : out[i]
  return record(E, TT.raw(ovals, shape, odt), 'ConvolutionBackward0', [x0, w, b], (g) => {
    const gx = x0.requiresGrad ? new Float64Array(xv.length) : null
    const gw = w.requiresGrad ? new Float64Array(wv.length) : null
    const gb = b && b.requiresGrad ? new Array<number>(O).fill(0) : null
    for (let n = 0; n < N; n++) {
      for (let o = 0; o < O; o++) {
        const ob = (n * O + o) * Ho * Wo
        if (gb) for (let i = 0; i < Ho * Wo; i++) gb[o] += g[ob + i]
        for (let c = 0; c < C; c++) {
          const xb = (n * C + c) * H * W
          const wb = (o * C + c) * KH * KW
          for (let u = 0; u < KH; u++) {
            for (let v = 0; v < KW; v++) {
              const wt = wv[wb + u * KW + v]
              let acc = 0
              for (let i = 0; i < Ho; i++) {
                const yy = i * sh + u - ph
                if (yy < 0 || yy >= H) continue
                for (let j = 0; j < Wo; j++) {
                  const xx = j * sw + v - pw
                  if (xx < 0 || xx >= W) continue
                  const gij = g[ob + i * Wo + j]
                  if (gij === 0) continue
                  acc += gij * xv[xb + yy * W + xx]
                  if (gx) gx[xb + yy * W + xx] += gij * wt
                }
              }
              if (gw) gw[wb + u * KW + v] += acc
            }
          }
        }
      }
    }
    return [gx ? Array.from(gx) : null, gw ? Array.from(gw) : null, gb]
  })
}

export function pool2d(E: Engine, x0: TT, k: [number, number], s: [number, number], p: [number, number], kind: 'max' | 'avg'): TT {
  const unbatched = x0.ndim === 3
  if (x0.ndim !== 3 && x0.ndim !== 4) throw err('RuntimeError', `${kind}_pool2d: Expected 3D or 4D (batch mode) tensor for input, but got: [${x0.shape.join(', ')}]`)
  const shape4 = unbatched ? [1, ...x0.shape] : x0.shape
  const [N, C, H, W] = shape4
  if (p[0] > k[0] / 2 || p[1] > k[1] / 2) throw err('RuntimeError', `pad should be at most half of effective kernel size, but got pad=${p[0]}, kernel_size=${k[0]} and dilation=1`)
  const Ho = convOut(H, k[0], s[0], p[0])
  const Wo = convOut(W, k[1], s[1], p[1])
  if (Ho < 1 || Wo < 1) throw err('RuntimeError', `Given input size: (${C}x${H}x${W}). Calculated output size: (${C}x${Ho}x${Wo}). Output size is too small`)
  const xv = x0.values()
  const out = new Array<number>(N * C * Ho * Wo)
  const arg = new Int32Array(out.length)
  for (let nc = 0; nc < N * C; nc++) {
    for (let i = 0; i < Ho; i++) {
      for (let j = 0; j < Wo; j++) {
        let best = kind === 'max' ? -Infinity : 0
        let at = -1
        let cnt = 0
        for (let u = 0; u < k[0]; u++) {
          for (let v = 0; v < k[1]; v++) {
            const y = i * s[0] + u - p[0]
            const xx = j * s[1] + v - p[1]
            cnt++
            if (y < 0 || y >= H || xx < 0 || xx >= W) continue
            const val = xv[nc * H * W + y * W + xx]
            if (kind === 'max') {
              if (val > best || at < 0) { best = val; at = nc * H * W + y * W + xx }
            } else best += val
          }
        }
        const o = nc * Ho * Wo + i * Wo + j
        out[o] = kind === 'max' ? best : best / cnt
        arg[o] = at
      }
    }
  }
  const shape = unbatched ? [C, Ho, Wo] : [N, C, Ho, Wo]
  return record(E, TT.of(out, shape, x0.dtype), kind === 'max' ? 'MaxPool2DWithIndicesBackward0' : 'AvgPool2DBackward0', [x0], (g) => {
    const gx = new Array<number>(xv.length).fill(0)
    for (let nc = 0; nc < N * C; nc++) {
      for (let i = 0; i < Ho; i++) {
        for (let j = 0; j < Wo; j++) {
          const o = nc * Ho * Wo + i * Wo + j
          if (kind === 'max') gx[arg[o]] += g[o]
          else {
            for (let u = 0; u < k[0]; u++) {
              for (let v = 0; v < k[1]; v++) {
                const y = i * s[0] + u - p[0]
                const xx = j * s[1] + v - p[1]
                if (y >= 0 && y < H && xx >= 0 && xx < W) gx[nc * H * W + y * W + xx] += g[o] / (k[0] * k[1])
              }
            }
          }
        }
      }
    }
    return [gx]
  })
}

export function dropout(E: Engine, x: TT, p: number, training: boolean): TT {
  if (p < 0 || p > 1) throw err('ValueError', `dropout probability has to be between 0 and 1, but got ${p}`)
  if (!training || p === 0) return x
  const keep = x.values().map(() => (E.rng.next() >= p ? 1 / (1 - p) : 0))
  const xv = x.values()
  return record(E, TT.of(xv.map((v, i) => v * keep[i]), x.shape, x.dtype), 'NativeDropoutBackward0', [x], (g) => [g.map((gi, i) => gi * keep[i])])
}

/** softmax / log_softmax along a dim */
export function softmax(E: Engine, x: TT, dim: number, log: boolean): TT {
  const d = normDim(dim, x.ndim)
  const xv = x.values()
  const len = x.shape[d] ?? 1
  const st = x.ndim ? cStrides(x.shape)[d] : 1
  const out = new Array<number>(xv.length)
  const groups: number[][] = []
  for (let f = 0; f < xv.length; f++) {
    if (Math.floor(f / st) % len !== 0) continue
    const idx = Array.from({ length: len }, (_, k) => f + k * st)
    groups.push(idx)
    const m = Math.max(...idx.map((i) => xv[i]))
    const sum = idx.reduce((s, i) => s + Math.exp(xv[i] - m), 0)
    for (const i of idx) out[i] = log ? xv[i] - m - Math.log(sum) : Math.exp(xv[i] - m) / sum
  }
  const dtype = isFloatT(x.dtype) ? x.dtype : DEFAULTS.float
  const res = TT.of(out, x.shape, dtype)
  const ov = res.values()
  return record(E, res, log ? 'LogSoftmaxBackward0' : 'SoftmaxBackward0', [x], (g) => {
    const gx = new Array<number>(xv.length).fill(0)
    for (const idx of groups) {
      if (log) {
        const gs = idx.reduce((s, i) => s + g[i], 0)
        for (const i of idx) gx[i] = g[i] - Math.exp(ov[i]) * gs
      } else {
        const dot = idx.reduce((s, i) => s + g[i] * ov[i], 0)
        for (const i of idx) gx[i] = ov[i] * (g[i] - dot)
      }
    }
    return [gx]
  })
}

export type Reduction = 'mean' | 'sum' | 'none'

const reduceLoss = (E: Engine, per: TT, reduction: Reduction): TT => (reduction === 'none' ? per : reduceSum(E, per, null, false, reduction === 'mean'))

/** PyTorch's first checks on class indices, in its order */
export const checkClassTarget = (input: TT, target: TT): void => {
  const batch = target.ndim ? target.shape[0] : 0
  if (input.ndim === 2 && batch !== input.shape[0]) throw err('ValueError', `Expected input batch_size (${input.shape[0]}) to match target batch_size (${batch}).`)
  if (input.ndim === 1 && target.ndim === 1 && target.shape[0] !== 1) throw err('ValueError', `For 1D input, 1D target must have size 1, but got target size: ${target.shape[0]}`)
  if (target.ndim > 1) throw err('RuntimeError', '0D or 1D target tensor expected, multi-target not supported')
}

/** negative log-likelihood of class indices, input = log-probabilities (N, C) or (C) */
export function nllLoss(E: Engine, logp: TT, target: TT, reduction: Reduction = 'mean'): TT {
  checkClassTarget(logp, target)
  if (target.dtype !== 'int64' && target.dtype !== 'uint8') throw err('RuntimeError', `expected target dtype to be Long or Byte, but got ${typeLabel(target.dtype)}`)
  const lp = logp.ndim === 1 ? reshapeT({ grad: false } as Engine, logp, [1, logp.shape[0]], false) : logp
  const [N, C] = lp.shape
  const tv = target.values()
  if (tv.length !== N) throw err('ValueError', `Expected input batch_size (${N}) to match target batch_size (${tv.length}).`)
  for (const t of tv) if (t < 0 || t >= C) throw err('IndexError', `Target ${t} is out of bounds.`)
  const lv = logp.values()
  const per = tv.map((t, n) => -lv[n * C + t])
  const total = reduction === 'mean' ? per.reduce((a, b) => a + b, 0) / N : reduction === 'sum' ? per.reduce((a, b) => a + b, 0) : 0
  const res = reduction === 'none' ? TT.of(per, [N], logp.dtype) : TT.of([total], [], logp.dtype)
  return record(E, res, 'NllLossBackward0', [logp], (g) => {
    const gl = new Array<number>(lv.length).fill(0)
    tv.forEach((t, n) => (gl[n * C + t] = -(reduction === 'none' ? g[n] : reduction === 'mean' ? g[0] / N : g[0])))
    return [gl]
  })
}

export function crossEntropy(E: Engine, logits: TT, target: TT, reduction: Reduction = 'mean'): TT {
  if (isFloatT(target.dtype) && target.shape.join() === logits.shape.join()) {
    // class probabilities: −Σ p log softmax(z)
    const lp = softmax(E, logits, logits.ndim === 1 ? 0 : 1, true)
    const per = reduceSum(E, binary(E, '*', unary(E, 'neg', target), lp), [logits.ndim === 1 ? 0 : 1], false, false)
    return reduceLoss(E, per, reduction)
  }
  if (logits.ndim < 1 || logits.ndim > 2) throw err('RuntimeError', `cross_entropy here takes (N, C) logits; got shape [${logits.shape.join(', ')}]`)
  return nllLoss(E, softmax(E, logits, -1, true), target, reduction)
}

const sizeMsg = (a: TT, b: TT) => `target size (torch.Size([${b.shape.join(', ')}])) that is different to the input size (torch.Size([${a.shape.join(', ')}]))`

/** mean squared error; mismatched shapes broadcast after a warning, exactly the silent bug PyTorch warns about */
export function mseLoss(E: Engine, a: TT, b: TT, reduction: Reduction = 'mean', warn?: (msg: string) => void): TT {
  if (a.shape.join() !== b.shape.join()) {
    warn?.(`UserWarning: Using a ${sizeMsg(a, b)}. This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.`)
    const sq = unary(E, 'square', binary(E, '-', a, b))
    return reduceLoss(E, sq, reduction)
  }
  const av = a.values()
  const bv = b.values()
  const per = av.map((x, i) => (x - bv[i]) ** 2)
  const n = per.length
  const dt = promote(a.dtype, b.dtype)
  if (!isFloatT(dt)) throw err('NotImplementedError', `"mse_cpu" not implemented for '${typeLabel(dt)}'`)
  const res = reduction === 'none' ? TT.of(per, a.shape, dt) : TT.of([per.reduce((s, v) => s + v, 0) / (reduction === 'mean' ? n : 1)], [], dt)
  return record(E, res, 'MseLossBackward0', [a, b], (g) => {
    const scale = (i: number) => (reduction === 'none' ? g[i] : g[0] / (reduction === 'mean' ? n : 1))
    const ga = av.map((x, i) => 2 * (x - bv[i]) * scale(i))
    return [ga, ga.map((v) => -v)]
  })
}

export function bce(E: Engine, p: TT, y: TT, logits: boolean, reduction: Reduction = 'mean'): TT {
  if (p.shape.join() !== y.shape.join()) throw err('ValueError', logits ? `Target size (torch.Size([${y.shape.join(', ')}])) must be the same as input size (torch.Size([${p.shape.join(', ')}]))` : `Using a ${sizeMsg(p, y)} is deprecated. Please ensure they have the same size.`)
  const pv = p.values()
  const yv = y.values()
  if (!logits && y.dtype !== p.dtype) throw err('RuntimeError', `Found dtype ${typeLabel(y.dtype)} but expected ${typeLabel(p.dtype)}`)
  if (logits && !isFloatT(p.dtype)) throw err('NotImplementedError', `"log_sigmoid_cpu" not implemented for '${typeLabel(p.dtype)}'`)
  if (logits && y.dtype === 'bool') throw err('NotImplementedError', 'Subtraction, the `-` operator, with two bool tensors is not supported. Use the `^` or `logical_xor()` operator instead.')
  if (logits && !isFloatT(y.dtype)) throw err('RuntimeError', `result type ${typeLabel(p.dtype)} can't be cast to the desired output type ${typeLabel(y.dtype)}`)
  if (!logits && pv.some((v) => v < 0 || v > 1)) throw err('RuntimeError', 'all elements of input should be between 0 and 1')
  const eps = (v: number) => Math.max(-100, Math.log(v))
  const per = pv.map((v, i) => (logits ? Math.max(v, 0) - v * yv[i] + Math.log1p(Math.exp(-Math.abs(v))) : -(yv[i] * eps(v) + (1 - yv[i]) * eps(1 - v))))
  const n = per.length
  // with logits, the result takes the target's type
  const dt = logits ? y.dtype : p.dtype
  const res = reduction === 'none' ? TT.of(per, p.shape, dt) : TT.of([per.reduce((s, v) => s + v, 0) / (reduction === 'mean' ? n : 1)], [], dt)
  return record(E, res, logits ? 'BinaryCrossEntropyWithLogitsBackward0' : 'BinaryCrossEntropyBackward0', [p, y], (g) => {
    const scale = (i: number) => (reduction === 'none' ? g[i] : g[0] / (reduction === 'mean' ? n : 1))
    return [pv.map((v, i) => (logits ? 1 / (1 + Math.exp(-v)) - yv[i] : (v - yv[i]) / Math.max(v * (1 - v), 1e-12)) * scale(i)), null]
  })
}

// ---------------------------------------------------------------- backward

const FN_LABEL: Record<string, string> = { AsStridedBackward0: 'AsStrided', AddBackward0: 'Add', SubBackward0: 'Sub', MulBackward0: 'Mul', DivBackward0: 'Div', PowBackward0: 'Pow', PowBackward1: 'Pow' }

/**
 * What real PyTorch does when a value is overwritten after one of these used it: their gradients read the input again.
 * exp, sqrt, tan, tanh, sigmoid, relu, softmax and the rest read their output instead, so overwriting the input does no harm.
 */
const KEEPS_INPUT = /^(Pow|Log|Log2|Log10|Log1P|Sin|Cos|Sinh|Cosh|Asinh|Acosh|Abs|Atan|Asin|Acos|Erf|Erfc|Lgamma|Digamma|Logit|Gelu|Elu|Celu|Silu|Mish|Softplus|Hardtanh|Hardswish|Hardsigmoid|LogSigmoid|LeakyRelu|Threshold|Clamp|LinalgVectorNorm|Std|Var|Prod|Cumprod|Amax|Amin|Median|Logsumexp|MseLoss|SmoothL1Loss|HuberLoss|NllLoss|BinaryCrossEntropy|BinaryCrossEntropyWithLogits|Atan2|Maximum|Minimum|Hypot)Backward\d$|^(Max|Min)Backward1$/

/** nodes whose gradient needs an input's value: PyTorch refuses if that value has been overwritten since */
const checkSaved = (n: GradNode): void => {
  const refuse = (x: TT, now: number, was: number, name?: string): never => {
    const output = name ? `, which is output 0 of ${FN_LABEL[name] ?? name.replace(/Backward\d*$/, '')},` : ''
    throw err('RuntimeError', `one of the variables needed for gradient computation has been modified by an inplace operation: [torch.${typeLabel(x.dtype)}Tensor [${x.shape.join(', ')}]]${output} is at version ${now}; expected version ${was} instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True, check_nan=False).`)
  }
  if (n.keeps && n.keeps[0].version !== n.keeps[1]) refuse(n.keeps[0], n.keeps[0].version, n.keeps[1], n.keeps[0].fn?.name)
  const product = /^(Mul|Dot|Mv|Mm|Bmm|Addmm)Backward/.test(n.name)
  const div = /^DivBackward/.test(n.name)
  if (!product && !div && !KEEPS_INPUT.test(n.name)) return
  n.inputs.forEach((x, i) => {
    if (!x || !n.versions) return
    // a view shares its base's memory: writing to the base has overwritten what this input held
    const viaBase = !!x.base && x.baseVersion !== x.base.version
    if (x.version === n.versions[i] && !viaBase) return
    // a product keeps an operand only if the other one needs its gradient; a division always keeps what it divides by
    const other = n.inputs[1 - i]
    if ((product || (div && i === 0)) && !other?.requiresGrad) return
    if (viaBase) refuse(x, x.base!.version, x.baseVersion!, 'AsStridedBackward0')
    refuse(x, x.version, n.versions[i], x.fn?.name ?? inputOf(n, i)?.fn?.name)
  })
}

/**
 * An in-place op has computed res from a stand-in for t's old value: write it back. Graphs recorded earlier keep meaning
 * the old value (past), and writing through a view rewrites the base's history too (PyTorch's CopySlices).
 */
export function commitInplace(E: Engine, t: TT, old: TT, res: TT, reshape: boolean): void {
  if (reshape) t.a = res.a
  else writeInto(t, res.values())
  if (res.fn) {
    (t.past ??= [])[t.version] = old
    t.fn = res.fn
    t.requiresGrad = true
  }
  t.version++
  const b = t.base
  if (!b || reshape) return
  if (!res.fn) {
    b.version++
    t.baseVersion = b.version
    return
  }
  // b's elements under the view now come from res, the others from b's old self
  const pos = positionsIn(b, t)
  const baseOld = new TT(b.a, b.dtype)
  baseOld.fn = b.fn
  baseOld.requiresGrad = b.requiresGrad
  ;(b.past ??= [])[b.version] = baseOld
  b.fn = { name: 'CopySlices', inputs: [baseOld, res], versions: [0, 0], freed: false, backward: (g) => {
    const gv = pos.map((p) => g[p])
    const ga = [...g]
    pos.forEach((p) => (ga[p] = 0))
    return [ga, gv]
  } }
  b.requiresGrad = true
  b.version++
  t.baseVersion = b.version
  viewNode(E, t, b)
}

export function backward(E: Engine, root: TT, gradOut: number[] | null, retainGraph: boolean): TT[] {
  if (!root.requiresGrad) throw err('RuntimeError', 'element 0 of tensors does not require grad and does not have a grad_fn')
  if (!gradOut && root.size !== 1) throw err('RuntimeError', 'grad can be implicitly created only for scalar outputs')
  // topological order (inputs after the tensors that use them)
  const order: TT[] = []
  const seen = new Set<TT>()
  const visit = (t: TT) => {
    if (seen.has(t)) return
    seen.add(t)
    const n = t.fn
    if (n) n.inputs.forEach((_, i) => {
      const x = inputOf(n, i)
      if (x && x.requiresGrad) visit(x)
    })
    order.push(t)
  }
  visit(root)
  for (const t of order) if (t.fn?.freed) {
    throw err('RuntimeError', 'Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.')
  }
  const grads = new Map<TT, number[]>([[root, gradOut ?? [1]]])
  const touched: TT[] = []
  const wasGrad = E.grad
  E.grad = false
  try {
    for (let k = order.length - 1; k >= 0; k--) {
      const t = order[k]
      let g = grads.get(t)
      if (!g) continue
      for (const hook of t.hooks ?? []) g = hook(TT.of(g, t.shape, t.dtype))?.values() ?? g
      if (t.fn) {
        checkSaved(t.fn)
        const gs = t.fn.backward(g)
        const n = t.fn
        n.inputs.forEach((_, i) => {
          const x = inputOf(n, i)
          const gi = gs[i]
          if (!x || !x.requiresGrad || !gi) return
          const prev = grads.get(x)
          grads.set(x, prev ? prev.map((v, j) => v + gi[j]) : gi)
        })
        if (t.retain) t.grad = TT.of(g, t.shape, t.dtype)
      } else {
        // a leaf: gradients accumulate across backward() calls
        const prev = t.grad?.values()
        t.grad = TT.of(prev ? prev.map((v, j) => v + g[j]) : g, t.shape, t.dtype)
        touched.push(t)
      }
    }
  } finally {
    E.grad = wasGrad
  }
  if (!retainGraph) for (const t of order) if (t.fn) t.fn.freed = true
  return touched
}

/** backward with create_graph=True: gradients are tensors with a history of their own. Every step's gradient is kept */
export function backwardGraph(E: Engine, root: TT, gradOut: TT | null): Map<TT, TT> {
  if (!root.requiresGrad) throw err('RuntimeError', 'element 0 of tensors does not require grad and does not have a grad_fn')
  if (!gradOut && root.size !== 1) throw err('RuntimeError', 'grad can be implicitly created only for scalar outputs')
  const order: TT[] = []
  const seen = new Set<TT>()
  const visit = (t: TT) => {
    if (seen.has(t)) return
    seen.add(t)
    const n = t.fn
    if (n) n.inputs.forEach((_, i) => {
      const x = inputOf(n, i)
      if (x && x.requiresGrad) visit(x)
    })
    order.push(t)
  }
  visit(root)
  const grads = new Map<TT, TT>([[root, gradOut ?? TT.of(new Array<number>(root.size).fill(1), root.shape, root.dtype)]])
  const wasGrad = E.grad
  E.grad = true
  try {
    for (let k = order.length - 1; k >= 0; k--) {
      const t = order[k]
      let g = grads.get(t)
      if (!g) continue
      for (const hook of t.hooks ?? []) g = hook(g) ?? g
      grads.set(t, g)
      const n = t.fn
      if (!n) continue
      if (n.freed) throw err('RuntimeError', 'Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.')
      if (!n.dual) throw err('NotImplementedError', `create_graph=True: the gradient through ${n.name} cannot be differentiated again in this sandbox`)
      const gs = n.dual(g)
      n.inputs.forEach((_, i) => {
        const x = inputOf(n, i)
        const gi = gs[i]
        if (!x || !x.requiresGrad || !gi) return
        const prev = grads.get(x)
        grads.set(x, prev ? binary(E, '+', prev, gi) : gi)
      })
    }
  } finally {
    E.grad = wasGrad
  }
  return grads
}

// ---------------------------------------------------------------- the wider torch API (sort, gather, BatchNorm …)

export const typeLabel = (d: TDType) => ({ float16: 'Half', float32: 'Float', float64: 'Double', int64: 'Long', int32: 'Int', int16: 'Short', int8: 'Char', uint8: 'Byte', bool: 'Bool' })[d]

const flatIds = (shape: number[]) => NDArray.create(Array.from({ length: prod(shape) }, (_, i) => i), shape, 'int64')

/** where each element of a view comes from, as flat positions in a */
export const viewSource = (a: TT, f: (x: NDArray) => NDArray): number[] => f(flatIds(a.shape)).values()

/** a view: f rewrites shape, strides and offset without copying, so the result shares a's memory */
export function viewT(E: Engine, a: TT, f: (x: NDArray) => NDArray, name: string): TT {
  const out = isView(new TT(f(a.a), a.dtype), a)
  if (!needsGrad(E, a)) return out
  const src = viewSource(a, f)
  return record(E, out, name, [a], (g) => {
    const ga = new Array<number>(a.size).fill(0)
    src.forEach((s, i) => (ga[s] += g[i]))
    return [ga]
  }, (g) => [accumulateBack(E, g, src, a.shape)])
}

export const narrowT = (E: Engine, a: TT, d: number, start: number, len: number, name = 'NarrowBackward0'): TT =>
  viewT(E, a, (x) => x.view(x.shape.map((s, k) => (k === d ? len : s)), x.strides, x.offset + start * x.strides[d]), name)

/** expand: dimensions of size 1 (and new leading ones) repeat without copying */
export const expandT = (E: Engine, a: TT, shape: number[]): TT =>
  viewT(E, a, (x) => {
    const lead = shape.length - x.ndim
    return x.view([...shape], shape.map((d, k) => (k < lead || (x.shape[k - lead] === 1 && d !== 1) ? 0 : x.strides[k - lead])), x.offset)
  }, 'ExpandBackward0')

/** out[i] = a[src[i]], a copy; src[i] < 0 = the fill value. Gradients go back to the elements they came from */
export function takeT(E: Engine, a: TT, src: ArrayLike<number>, shape: number[], name: string, fill = 0): TT {
  const av = a.values()
  const n = src.length
  const vals = new Array<number>(n)
  for (let i = 0; i < n; i++) vals[i] = src[i] < 0 ? castT(fill, a.dtype) : av[src[i]]
  return record(E, TT.raw(vals, shape, a.dtype), name, [a], (g) => {
    const ga = new Array<number>(av.length).fill(0)
    for (let i = 0; i < n; i++) if (src[i] >= 0) ga[src[i]] += g[i]
    return [ga]
  }, (g) => [accumulateBack(E, g, src, a.shape)])
}

/** output (i, j, …) comes from input (maps[0][i], maps[1][j], …); -1 on any axis = no input element */
export const crossSrc = (inShape: number[], maps: number[][]): Int32Array => {
  const st = cStrides(inShape)
  let src = new Int32Array([0])
  maps.forEach((m, d) => {
    const next = new Int32Array(src.length * m.length)
    let o = 0
    for (let p = 0; p < src.length; p++) for (let i = 0; i < m.length; i++) next[o++] = src[p] < 0 || m[i] < 0 ? -1 : src[p] + m[i] * st[d]
    src = next
  })
  return src
}

/** an elementwise function and its derivative (null = piecewise constant, gradient 0) */
export function mapT(E: Engine, a: TT, f: (x: number) => number, df: ((x: number) => number) | null, name: string, keepInt = false): TT {
  const av = a.values()
  const dtype = isFloatT(a.dtype) || keepInt ? a.dtype : DEFAULTS.float
  return record(E, TT.of(av.map(f), a.shape, dtype), name, [a], (g) => [df ? g.map((gi, i) => gi * df(av[i])) : g.map(() => 0)])
}

/** an elementwise function of two broadcast tensors and its two partial derivatives */
export function map2T(E: Engine, a: TT, b: TT, f: (x: number, y: number) => number, dfx: (x: number, y: number) => number, dfy: (x: number, y: number) => number, dtype: TDType, name: string): TT {
  const shape = shapeOfBcast(a.shape, b.shape)
  const [ia, ib] = [bcastIndex(a.shape, shape), bcastIndex(b.shape, shape)]
  const [av, bv] = [a.values(), b.values()]
  const out = Array.from({ length: prod(shape) }, (_, i) => f(av[ia[i]], bv[ib[i]]))
  return record(E, TT.of(out, shape, dtype), name, [a, b], (g) => [
    unbroadcast(g.map((x, i) => x * dfx(av[ia[i]], bv[ib[i]])), ia, a.size),
    unbroadcast(g.map((x, i) => x * dfy(av[ia[i]], bv[ib[i]])), ib, b.size),
  ])
}

/** the largest or smallest over dims; tied elements share the gradient evenly, as amax, amin and max() do */
export function extremeT(E: Engine, a: TT, dims: number[] | null, keepdim: boolean, min: boolean, name: string): TT {
  const ds = dims === null ? a.shape.map((_, i) => i) : [...new Set(dims.map((d) => normDim(d, a.ndim)))]
  if (!a.size) throw err('RuntimeError', `${min ? 'min' : 'max'}(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.`)
  const { map, outSize } = reduceMap(a.shape, ds)
  const av = a.values()
  const best = new Array<number>(outSize).fill(min ? Infinity : -Infinity)
  av.forEach((v, i) => { best[map[i]] = Number.isNaN(best[map[i]]) || Number.isNaN(v) ? NaN : min ? Math.min(best[map[i]], v) : Math.max(best[map[i]], v) })
  return record(E, TT.of(best, keptShape(a.shape, ds, keepdim), a.dtype), name, [a], (g) => [evenShare(av, map, best, g)])
}

/** each output's gradient split among the inputs equal to it */
const evenShare = (av: number[], map: Int32Array | number[], best: number[], g: number[]): number[] => {
  const ties = new Array<number>(best.length).fill(0)
  av.forEach((v, i) => { if (v === best[map[i]]) ties[map[i]]++ })
  return av.map((v, i) => (v === best[map[i]] ? g[map[i]] / ties[map[i]] : 0))
}

/** the one value a reduction over everything settled on (median), with the gradient shared among its copies */
export function sharedValueT(E: Engine, a: TT, at: number, name: string): TT {
  const av = a.values()
  const map = new Int32Array(av.length)
  return record(E, TT.of([av[at]], [], a.dtype), name, [a], (g) => [evenShare(av, map, [av[at]], g)])
}

export const itemSize = (d: TDType): number => ({ float16: 2, float32: 4, float64: 8, int64: 8, int32: 4, int16: 2, int8: 1, uint8: 1, bool: 1 })[d]

/** an op with no derivative formula: the graph records it, and only backward() complains */
export function noGradient(E: Engine, out: TT, inputs: TT[], op: string, shown = 'NotImplemented'): TT {
  return record(E, out, shown, inputs, () => {
    // real torch has no derivative at all for a NotImplemented node; for the others it is this sandbox that lacks one
    if (shown === 'NotImplemented') throw err('RuntimeError', `derivative for aten::${op} is not implemented`)
    throw err('NotImplementedError', `the gradient through ${op} is not available in this sandbox`)
  })
}

export function whereT(E: Engine, cond: TT, a: TT, b: TT, dtype: TDType, name = 'WhereBackward0'): TT {
  const shape = shapeOfBcast(shapeOfBcast(cond.shape, a.shape), b.shape)
  const [ic, ia, ib] = [cond, a, b].map((t) => bcastIndex(t.shape, shape))
  const [cv, av, bv] = [cond, a, b].map((t) => t.values())
  const n = prod(shape)
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) out[i] = cv[ic[i]] ? av[ia[i]] : bv[ib[i]]
  return record(E, TT.of(out, shape, dtype), name, [a, b], (g) => [
    unbroadcast(g.map((x, i) => (cv[ic[i]] ? x : 0)), ia, a.size),
    unbroadcast(g.map((x, i) => (cv[ic[i]] ? 0 : x)), ib, b.size),
  ])
}

/** the flat indices of each line along dim d (a 0-d tensor is one line of one element) */
const lanesOf = (shape: number[], d: number): number[][] => {
  if (!shape.length) return [[0]]
  const st = cStrides(shape)[d]
  const len = shape[d]
  const lanes: number[][] = []
  for (let f = 0, n = prod(shape); f < n; f++) {
    if (Math.floor(f / st) % len === 0) lanes.push(Array.from({ length: len }, (_, k) => f + k * st))
  }
  return lanes
}

const keptShape = (shape: number[], ds: number[], keepdim: boolean) => (keepdim ? shape.map((s, k) => (ds.includes(k) ? 1 : s)) : shape.filter((_, k) => !ds.includes(k)))

export function prodT(E: Engine, a: TT, dim: number | null, keepdim: boolean): TT {
  const ds = dim === null ? a.shape.map((_, i) => i) : [normDim(dim, a.ndim)]
  const { map, outSize } = reduceMap(a.shape, ds)
  const av = a.values()
  const out = new Array<number>(outSize).fill(1)
  av.forEach((v, i) => (out[map[i]] *= v))
  return record(E, TT.of(out, keptShape(a.shape, ds, keepdim), isFloatT(a.dtype) ? a.dtype : 'int64'), dim === null ? 'ProdBackward0' : 'ProdBackward1', [a], (g) => {
    // the product of the others, without dividing by zero: one zero passes gradient only to itself, two to nobody
    const zeros = new Array<number>(outSize).fill(0)
    const rest = new Array<number>(outSize).fill(1)
    av.forEach((v, i) => { if (v === 0) zeros[map[i]]++; else rest[map[i]] *= v })
    return [av.map((v, i) => {
      const o = map[i]
      if (zeros[o] > 1) return 0
      if (zeros[o] === 1) return v === 0 ? g[o] * rest[o] : 0
      return (g[o] * rest[o]) / v
    })]
  })
}

/** all / any / count_nonzero: f sees how many of how many elements are non-zero */
export function countReduce(a: TT, dims: number[] | null, keepdim: boolean, dtype: TDType, f: (nonzero: number, total: number) => number): TT {
  const ds = dims === null ? a.shape.map((_, i) => i) : [...new Set(dims.map((d) => normDim(d, a.ndim)))]
  const { map, outSize } = reduceMap(a.shape, ds)
  const nz = new Array<number>(outSize).fill(0)
  const count = new Array<number>(outSize).fill(0)
  a.values().forEach((v, i) => { count[map[i]]++; if (v !== 0) nz[map[i]]++ })
  return TT.of(nz.map((x, o) => f(x, count[o])), keptShape(a.shape, ds, keepdim), dtype)
}

export function cumT(E: Engine, a: TT, dim: number, mul: boolean): TT {
  const lanes = lanesOf(a.shape, normDim(dim, a.ndim))
  const av = a.values()
  const out = [...av]
  for (const l of lanes) for (let k = 1; k < l.length; k++) out[l[k]] = mul ? out[l[k - 1]] * av[l[k]] : out[l[k - 1]] + av[l[k]]
  return record(E, TT.of(out, a.shape, isFloatT(a.dtype) ? a.dtype : 'int64'), mul ? 'CumprodBackward0' : 'CumsumBackward0', [a], (g) => {
    const ga = new Array<number>(av.length).fill(0)
    for (const l of lanes) {
      if (!mul) {
        let s = 0
        for (let i = l.length - 1; i >= 0; i--) ga[l[i]] = s += g[l[i]]
        continue
      }
      // d y_j / d x_i = the product of x_0 … x_j without x_i
      let before = 1
      for (let i = 0; i < l.length; i++) {
        let p = before
        for (let j = i; j < l.length; j++) {
          if (j > i) p *= av[l[j]]
          ga[l[i]] += g[l[j]] * p
        }
        before *= av[l[i]]
      }
    }
    return [ga]
  })
}

/** positions of a line in sorted order; NaN counts as the largest, ties keep their order */
const sortedOrder = (vals: number[], descending: boolean): number[] => {
  const cmp = (x: number, y: number) => (x < y ? -1 : x > y ? 1 : Number.isNaN(x) ? (Number.isNaN(y) ? 0 : 1) : Number.isNaN(y) ? -1 : 0)
  return vals.map((_, k) => k).sort((p, q) => (descending ? cmp(vals[q], vals[p]) : cmp(vals[p], vals[q])))
}

export function sortT(E: Engine, a: TT, dim: number, descending: boolean): { values: TT; indices: TT; src: Int32Array } {
  const av = a.values()
  const src = new Int32Array(av.length)
  const idx = new Array<number>(av.length)
  for (const l of lanesOf(a.shape, normDim(dim, a.ndim))) {
    sortedOrder(l.map((f) => av[f]), descending).forEach((k, pos) => {
      src[l[pos]] = l[k]
      idx[l[pos]] = k
    })
  }
  return { values: takeT(E, a, src, a.shape, 'SortBackward0'), indices: TT.raw(idx, a.shape, 'int64'), src }
}

export function topkT(E: Engine, a: TT, k: number, dim: number, largest: boolean): { values: TT; indices: TT; src: Int32Array } {
  const d = normDim(dim, a.ndim)
  if (k < 0 || k > (a.shape[d] ?? 1)) throw err('RuntimeError', 'selected index k out of range')
  const av = a.values()
  const shape = a.shape.map((s, i) => (i === d ? k : s))
  const outLanes = lanesOf(shape, d)
  const src = new Int32Array(prod(shape))
  const idx = new Array<number>(src.length)
  lanesOf(a.shape, d).forEach((l, j) => {
    sortedOrder(l.map((f) => av[f]), largest).slice(0, k).forEach((p, pos) => {
      src[outLanes[j][pos]] = l[p]
      idx[outLanes[j][pos]] = p
    })
  })
  return { values: takeT(E, a, src, shape, 'TopkBackward0'), indices: TT.raw(idx, shape, 'int64'), src }
}

/** one element of every line along dim, as median / mode / kthvalue return it: choose gets a line and names a position */
export function pickAlong(E: Engine, a: TT, dim: number, keepdim: boolean, choose: (vals: number[]) => number, name: string): { values: TT; indices: TT; src: Int32Array } {
  const d = normDim(dim, a.ndim)
  const av = a.values()
  const lanes = lanesOf(a.shape, d)
  const src = new Int32Array(lanes.length)
  const idx = new Array<number>(lanes.length)
  lanes.forEach((l, j) => {
    idx[j] = choose(l.map((f) => av[f]))
    src[j] = l[idx[j]]
  })
  const shape = keptShape(a.shape, [d], keepdim)
  return { values: takeT(E, a, src, shape, name), indices: TT.raw(idx, shape, 'int64'), src }
}

/** torch.median: the lower of the two middle values; a NaN in the line wins */
export const lowerMedianAt = (vals: number[]): number => {
  const nan = vals.findIndex(Number.isNaN)
  return nan >= 0 ? nan : sortedOrder(vals, false)[(vals.length - 1) >> 1]
}

/** torch.mode: the most frequent value, the smallest on a tie, at its last position */
export const modeAt = (vals: number[]): number => {
  const order = sortedOrder(vals, false)
  let best = 0
  let freq = 0
  let run = 0
  order.forEach((p, i) => {
    run++
    if (i === order.length - 1 || vals[p] !== vals[order[i + 1]]) {
      if (run > freq) {
        best = p
        freq = run
      }
      run = 0
    }
  })
  return best
}

export const kthAt = (k: number) => (vals: number[]): number => {
  if (k < 1 || k > vals.length) throw err('RuntimeError', 'kthvalue(): selected number k out of range for dimension')
  return sortedOrder(vals, false)[k - 1]
}

/** the flat position of every index element in a tensor of this shape, with dim replaced by the index value */
const indexedFlat = (shape: number[], d: number, index: TT): Int32Array => {
  const st = cStrides(shape)
  const ist = cStrides(index.shape)
  const iv = index.values()
  const out = new Int32Array(iv.length)
  for (let f = 0; f < iv.length; f++) {
    let r = f
    let pos = 0
    for (let k = 0; k < shape.length; k++) {
      const c = Math.floor(r / ist[k])
      r -= c * ist[k]
      pos += (k === d ? iv[f] : c) * st[k]
    }
    out[f] = pos
  }
  return out
}

const checkIndex = (a: TT, d: number, index: TT, fname: string) => {
  if (index.dtype !== 'int64' && index.dtype !== 'int32') throw err('RuntimeError', `${fname}(): Expected dtype ${fname === 'gather' ? 'int32/int64' : 'int64'} for index`)
  if (index.ndim !== a.ndim) throw err('RuntimeError', 'Index tensor must have the same number of dimensions as input tensor')
  index.values().forEach((v) => {
    if (v < 0 || v >= (a.shape[d] ?? 1)) throw err('RuntimeError', `index ${v} is out of bounds for dimension ${d} with size ${a.shape[d] ?? 1}`)
  })
}

export function gatherT(E: Engine, a: TT, dim: number, index: TT): { out: TT; src: Int32Array } {
  const d = normDim(dim, a.ndim)
  checkIndex(a, d, index, 'gather')
  index.shape.forEach((s, k) => {
    if (k !== d && s > a.shape[k]) throw err('RuntimeError', `Size does not match at dimension ${k} expected index [${index.shape.join(', ')}] to be no larger than self [${a.shape.join(', ')}] apart from dimension ${d}`)
  })
  const src = indexedFlat(a.shape, d, index)
  return { out: takeT(E, a, src, index.shape, 'GatherBackward0'), src }
}

/** scatter / scatter_add: writes src (a tensor, or one number) at the positions the index names */
export function scatterT(E: Engine, a: TT, dim: number, index: TT, src: TT | number, add: boolean): TT {
  const d = normDim(dim, a.ndim)
  checkIndex(a, d, index, 'scatter')
  const srcT = src instanceof TT ? src : null
  if (srcT) {
    if (srcT.ndim !== index.ndim) throw err('RuntimeError', 'Index tensor must have the same number of dimensions as src tensor')
    if (srcT.dtype !== a.dtype) throw err('RuntimeError', 'scatter(): Expected self.dtype to be equal to src.dtype')
  }
  index.shape.forEach((s, k) => {
    if ((k !== d && s > a.shape[k]) || (srcT && s > srcT.shape[k])) throw err('RuntimeError', `Expected index [${index.shape.join(', ')}] to be no larger than self [${a.shape.join(', ')}] apart from dimension ${d}${srcT ? ` and to be no larger size than src [${srcT.shape.join(', ')}]` : ''}`)
  })
  const pos = indexedFlat(a.shape, d, index)
  const ist = cStrides(index.shape)
  const sst = srcT ? cStrides(srcT.shape) : []
  const from = new Int32Array(pos.length)
  for (let f = 0; f < pos.length; f++) {
    let r = f
    for (let k = 0; k < index.ndim; k++) {
      const c = Math.floor(r / ist[k])
      r -= c * ist[k]
      from[f] += c * (sst[k] ?? 0)
    }
  }
  const sv = srcT ? srcT.values() : null
  const out = a.values()
  const last = new Int32Array(out.length).fill(-1)
  pos.forEach((p, f) => {
    const v = sv ? sv[from[f]] : (src as number)
    out[p] = add ? out[p] + v : v
    last[p] = from[f]
  })
  return record(E, TT.of(out, a.shape, a.dtype), add ? 'ScatterAddBackward0' : srcT ? 'ScatterBackward0' : 'ScatterBackward1', [a, srcT], (g) => {
    const gs = srcT ? new Array<number>(srcT.size).fill(0) : null
    // as PyTorch: grad.gather(dim, index), even for a source cell a later one overwrote
    if (gs) pos.forEach((p, f) => (gs[from[f]] += g[p]))
    return [add ? g : g.map((x, p) => (last[p] >= 0 ? 0 : x)), gs]
  })
}

/** x[idx] = v on a tensor in a graph: the selected elements now depend on v, the others on the old x */
export function setItemT(E: Engine, old: TT, addresses: number[], planShape: number[], v: TT): TT {
  const flatOf = new Map(old.a.addresses().map((x, i) => [x, i]))
  const pos = addresses.map((x) => flatOf.get(x)!)
  const iv = bcastIndex(v.shape, planShape)
  const out = old.values()
  const vv = v.values()
  pos.forEach((p, i) => (out[p] = vv[iv[i]]))
  return record(E, TT.of(out, old.shape, old.dtype), 'CopySlices', [old, v], (g) => {
    const gv = new Array<number>(v.size).fill(0)
    const ga = [...g]
    pos.forEach((p, i) => {
      gv[iv[i]] += g[p]
      ga[p] = 0
    })
    return [ga, gv]
  })
}

export function triT(E: Engine, a: TT, k: number, upper: boolean): TT {
  if (a.ndim < 2) throw err('RuntimeError', `${upper ? 'triu' : 'tril'}: input tensor must have at least 2 dimensions`)
  const [r, c] = a.shape.slice(-2)
  const keep = (f: number) => {
    const delta = f % c - Math.floor(f / c) % r
    return upper ? delta >= k : delta <= k
  }
  return record(E, TT.of(a.values().map((v, f) => (keep(f) ? v : 0)), a.shape, a.dtype), upper ? 'TriuBackward0' : 'TrilBackward0', [a], (g) => [g.map((x, f) => (keep(f) ? x : 0))])
}

/** ‖x‖_p over dims, with the subgradients autograd uses at 0 and for the ∞ norms */
export function vectorNorm(E: Engine, a: TT, ord: number, dims: number[] | null, keepdim: boolean): TT {
  if (!isFloatT(a.dtype)) throw err('RuntimeError', `linalg.vector_norm: Expected a floating point or complex tensor as input. Got ${typeLabel(a.dtype)}`)
  const ds = dims === null ? a.shape.map((_, i) => i) : [...new Set(dims.map((d) => normDim(d, a.ndim)))]
  const { map, outSize } = reduceMap(a.shape, ds)
  const av = a.values()
  const inf = ord === Infinity || ord === -Infinity
  const acc = new Array<number>(outSize).fill(ord === -Infinity ? Infinity : 0)
  av.forEach((v, i) => {
    const x = Math.abs(v)
    const o = map[i]
    if (ord === Infinity) acc[o] = Math.max(acc[o], x)
    else if (ord === -Infinity) acc[o] = Math.min(acc[o], x)
    else if (ord === 0) acc[o] += x !== 0 ? 1 : 0
    else acc[o] += x ** ord
  })
  const norms = inf || ord === 0 ? acc : acc.map((s) => s ** (1 / ord))
  return record(E, TT.of(norms, keptShape(a.shape, ds, keepdim), a.dtype), 'LinalgVectorNormBackward0', [a], (g) => {
    if (ord === 0) return [av.map(() => 0)]
    if (inf) {
      const ties = new Array<number>(outSize).fill(0)
      av.forEach((v, i) => { if (Math.abs(v) === norms[map[i]]) ties[map[i]]++ })
      return [av.map((v, i) => (Math.abs(v) === norms[map[i]] ? (Math.sign(v) * g[map[i]]) / ties[map[i]] : 0))]
    }
    return [av.map((v, i) => {
      const o = map[i]
      if (norms[o] === 0 || v === 0) return 0
      return ord === 1 ? Math.sign(v) * g[o] : (Math.sign(v) * Math.abs(v) ** (ord - 1) * g[o]) / norms[o] ** (ord - 1)
    })]
  })
}

// ---- linear algebra, in the tensor's own precision (so a float32 det can print -2.0000, as LAPACK's single precision does)

const squareBatches = (a: TT, fname: string) => {
  if (!isFloatT(a.dtype)) throw err('RuntimeError', `${fname}: Expected a floating point or complex tensor as input. Got ${typeLabel(a.dtype)}`)
  if (a.ndim < 2) throw err('RuntimeError', `${fname}: The input tensor A must have at least 2 dimensions.`)
  const n = a.shape[a.ndim - 1]
  if (a.shape[a.ndim - 2] !== n) throw err('RuntimeError', `${fname}: A must be batches of square matrices, but they are ${a.shape[a.ndim - 2]} by ${n} matrices`)
  return { n, nb: prod(a.shape.slice(0, -2)) }
}

const rounder = (d: TDType): ((x: number) => number) => (d === 'float64' ? (x) => x : (x) => castT(x, d))

/** LU with partial pivoting, column by column and scaling by the pivot's reciprocal, as LAPACK's getrf does */
const luFactor = (m: number[], n: number, R: (x: number) => number) => {
  const lu = [...m]
  const piv = Array.from({ length: n }, (_, i) => i)
  let sign = 1
  let zero = -1
  for (let k = 0; k < n; k++) {
    let p = k
    for (let i = k + 1; i < n; i++) if (Math.abs(lu[i * n + k]) > Math.abs(lu[p * n + k])) p = i
    if (p !== k) {
      for (let j = 0; j < n; j++) [lu[k * n + j], lu[p * n + j]] = [lu[p * n + j], lu[k * n + j]]
      ;[piv[k], piv[p]] = [piv[p], piv[k]]
      sign = -sign
    }
    if (lu[k * n + k] === 0) {
      if (zero < 0) zero = k
      continue
    }
    const r = R(1 / lu[k * n + k])
    for (let i = k + 1; i < n; i++) {
      lu[i * n + k] = R(lu[i * n + k] * r)
      for (let j = k + 1; j < n; j++) lu[i * n + j] = R(lu[i * n + j] - R(lu[i * n + k] * lu[k * n + j]))
    }
  }
  return { lu, piv, sign, zero }
}

/** solve with the LU factors for an n × k right-hand side (BLAS trsm's order of operations) */
const luSolve = (lu: number[], piv: number[], n: number, b: number[], k: number, R: (x: number) => number): number[] => {
  const x = new Array<number>(n * k)
  for (let r = 0; r < n; r++) for (let c = 0; c < k; c++) x[r * k + c] = b[piv[r] * k + c]
  for (let c = 0; c < k; c++) {
    for (let p = 0; p < n; p++) for (let i = p + 1; i < n; i++) x[i * k + c] = R(x[i * k + c] - R(x[p * k + c] * lu[i * n + p]))
    for (let p = n - 1; p >= 0; p--) {
      x[p * k + c] = R(x[p * k + c] / lu[p * n + p])
      for (let i = 0; i < p; i++) x[i * k + c] = R(x[i * k + c] - R(x[p * k + c] * lu[i * n + p]))
    }
  }
  return x
}

const identity = (n: number) => Array.from({ length: n * n }, (_, i) => (i % (n + 1) === 0 ? 1 : 0))

const singular = (fname: string, nb: number, b: number, zero: number) =>
  err('LinAlgError', `${fname}: ${nb > 1 ? `(Batch element ${b}): ` : ''}The diagonal element ${zero + 1} is zero, the ${fname === 'linalg.inv' ? 'inversion' : 'solve'} could not be completed because the input matrix is singular.`)

export function invT(E: Engine, a: TT): TT {
  const { n, nb } = squareBatches(a, 'linalg.inv')
  const av = a.values()
  const R = rounder(a.dtype)
  const X: number[] = []
  for (let b = 0; b < nb; b++) {
    const { lu, piv, zero } = luFactor(av.slice(b * n * n, (b + 1) * n * n), n, R)
    if (zero >= 0) throw singular('linalg.inv', nb, b, zero)
    X.push(...luSolve(lu, piv, n, identity(n), n, R))
  }
  const res = TT.of(X, a.shape, a.dtype)
  const xv = res.values()
  return record(E, res, 'LinalgInvExBackward0', [a], (g) => {
    // d(A⁻¹) = −A⁻¹ dA A⁻¹, so the gradient is −A⁻ᵀ G A⁻ᵀ
    const ga = new Array<number>(av.length).fill(0)
    for (let b = 0; b < nb; b++) {
      const o = b * n * n
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          let s = 0
          for (let p = 0; p < n; p++) for (let q = 0; q < n; q++) s += xv[o + p * n + i] * g[o + p * n + q] * xv[o + j * n + q]
          ga[o + i * n + j] = -s
        }
      }
    }
    return [ga]
  })
}

export function detT(E: Engine, a: TT): TT {
  const { n, nb } = squareBatches(a, 'linalg.det')
  const av = a.values()
  const R = rounder(a.dtype)
  const detOf = (m: number[], size: number) => {
    const { lu, sign } = luFactor(m, size, R)
    let d = sign
    for (let i = 0; i < size; i++) d = R(d * lu[i * size + i])
    return d
  }
  const out = Array.from({ length: nb }, (_, b) => detOf(av.slice(b * n * n, (b + 1) * n * n), n))
  return record(E, TT.of(out, a.shape.slice(0, -2), a.dtype), 'LinalgDetBackward0', [a], (g) => {
    // the gradient of det is the cofactor matrix
    const ga = new Array<number>(av.length).fill(0)
    for (let b = 0; b < nb; b++) {
      const o = b * n * n
      const m = av.slice(o, o + n * n)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const minor = m.filter((_, f) => Math.floor(f / n) !== i && f % n !== j)
          ga[o + i * n + j] = g[b] * ((i + j) % 2 ? -1 : 1) * (n === 1 ? 1 : detOf(minor, n - 1))
        }
      }
    }
    return [ga]
  })
}

/** A X = B; B is a vector (…, n) or a matrix (…, n, k) */
export function solveT(E: Engine, A0: TT, B0: TT): TT {
  const { n } = squareBatches(A0, 'linalg.solve')
  if (A0.dtype !== B0.dtype) throw err('RuntimeError', `linalg.solve: Expected A and B to have the same dtype, but found A of type ${typeLabel(A0.dtype)} and B of type ${typeLabel(B0.dtype)} instead`)
  const vector = B0.ndim === 1 || (B0.ndim === A0.ndim - 1 && B0.shape.every((s, i) => s === A0.shape[i]))
  const B1 = vector ? reshapeT(E, B0, [...B0.shape, 1], false) : B0
  if (B1.shape[B1.ndim - 2] !== n) throw err('RuntimeError', `linalg.solve: Incompatible shapes of A and B for the equation AX = B (${A0.shape.join(', ')} and ${B0.shape.join(', ')})`)
  const k = B1.shape[B1.ndim - 1]
  const batch = shapeOfBcast(A0.shape.slice(0, -2), B1.shape.slice(0, -2))
  const A = expandT(E, A0, [...batch, n, n])
  const B = expandT(E, B1, [...batch, n, k])
  const nb = prod(batch)
  const av = A.values()
  const bv = B.values()
  const R = rounder(A.dtype)
  const X: number[] = []
  const invs: number[] = []
  for (let b = 0; b < nb; b++) {
    const { lu, piv, zero } = luFactor(av.slice(b * n * n, (b + 1) * n * n), n, R)
    if (zero >= 0) throw singular('linalg.solve', nb, b, zero)
    X.push(...luSolve(lu, piv, n, bv.slice(b * n * k, (b + 1) * n * k), k, R))
    if (needsGrad(E, A, B)) invs.push(...luSolve(lu, piv, n, identity(n), n, (x) => x))
  }
  const res = record(E, TT.of(X, B.shape, A.dtype), 'LinalgSolveExBackward0', [A, B], (g) => {
    // with Y = A⁻ᵀ G: dB receives Y and dA receives −Y Xᵀ
    const gA = new Array<number>(av.length).fill(0)
    const gB = new Array<number>(bv.length).fill(0)
    for (let b = 0; b < nb; b++) {
      const oa = b * n * n
      const ox = b * n * k
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < k; c++) {
          let y = 0
          for (let p = 0; p < n; p++) y += invs[oa + p * n + i] * g[ox + p * k + c]
          gB[ox + i * k + c] = y
          for (let j = 0; j < n; j++) gA[oa + i * n + j] -= y * X[ox + j * k + c]
        }
      }
    }
    return [gA, gB]
  })
  return vector ? reshapeT(E, res, res.shape.slice(0, -1), false) : res
}

// ---- layer kernels

/** batch normalisation of (N, C, …): stats = null normalises by this batch (and differentiates through its mean and variance) */
export function batchNormT(E: Engine, x: TT, w: TT | null, b: TT | null, stats: { mean: number[]; vari: number[] } | null, eps: number): { out: TT; mean: number[]; vari: number[] } {
  const Ch = x.shape[1]
  const rest = prod(x.shape.slice(2))
  const xv = x.values()
  const n = xv.length
  const count = n / Math.max(1, Ch)
  const wv = w ? w.values() : null
  const bv = b ? b.values() : null
  const chan = (f: number) => Math.floor(f / rest) % Ch
  let mean = stats?.mean
  let vari = stats?.vari
  if (!mean || !vari) {
    const m = new Array<number>(Ch).fill(0)
    const v = new Array<number>(Ch).fill(0)
    for (let f = 0; f < n; f++) m[chan(f)] += xv[f]
    for (let c = 0; c < Ch; c++) m[c] /= count
    for (let f = 0; f < n; f++) v[chan(f)] += (xv[f] - m[chan(f)]) ** 2
    for (let c = 0; c < Ch; c++) v[c] /= count
    mean = m
    vari = v
  }
  const [mu, sigma2] = [mean, vari]
  const s = sigma2.map((v) => Math.sqrt(v + eps))
  const out = new Array<number>(n)
  for (let f = 0; f < n; f++) {
    const c = chan(f)
    const z = (xv[f] - mu[c]) / s[c]
    out[f] = (wv ? z * wv[c] : z) + (bv ? bv[c] : 0)
  }
  const res = record(E, TT.of(out, x.shape, x.dtype), 'NativeBatchNormBackward0', [x, w, b], (g) => {
    const gSum = new Array<number>(Ch).fill(0)
    const gCov = new Array<number>(Ch).fill(0)
    for (let f = 0; f < n; f++) {
      const c = chan(f)
      gSum[c] += g[f]
      gCov[c] += g[f] * (xv[f] - mu[c])
    }
    const gx = new Array<number>(n)
    for (let f = 0; f < n; f++) {
      const c = chan(f)
      const scale = wv ? wv[c] : 1
      // with stats from the batch: ∂y_i/∂x_j = (δ_ij − 1/n)/s − (x_i − μ)(x_j − μ)/(n s³)
      gx[f] = stats ? (scale * g[f]) / s[c] : scale * ((g[f] - gSum[c] / count) / s[c] - (gCov[c] * (xv[f] - mu[c])) / (count * s[c] ** 3))
    }
    return [gx, gCov.map((v, c) => v / s[c]), gSum]
  })
  return { out: res, mean: mu, vari: sigma2 }
}

/** the older entry point for a 2-D transposed convolution, as keras's Conv2DTranspose calls it */
export function convTranspose2d(E: Engine, x0: TT, w: TT, b: TT | null, stride: [number, number], padding: [number, number], outputPadding: [number, number]): TT {
  if (x0.ndim !== 3 && x0.ndim !== 4) throw err('RuntimeError', `Expected 3D (unbatched) or 4D (batched) input to conv_transpose2d, but got input of size: [${x0.shape.join(', ')}]`)
  const Ch = x0.shape[x0.ndim - 3]
  if (Ch !== w.shape[0]) throw err('RuntimeError', `Given transposed=1, weight of size [${w.shape.join(', ')}], expected input[${x0.shape.join(', ')}] to have ${w.shape[0]} channels, but got ${Ch} channels instead`)
  return convNd(E, x0, w, b, stride, padding, [1, 1], outputPadding, true)
}

/** adaptive pooling: output cell i of n covers input ⌊i·H/n⌋ … ⌈(i+1)·H/n⌉ */
export function adaptivePool2d(E: Engine, x0: TT, [Ho, Wo]: [number, number], kind: 'max' | 'avg'): { out: TT; indices: TT } {
  if (x0.ndim !== 3 && x0.ndim !== 4) throw err('RuntimeError', `adaptive_${kind}_pool2d(): Expected 3D or 4D tensor, but got: [${x0.shape.join(', ')}]`)
  const [N, Ch, H, W] = x0.ndim === 3 ? [1, ...x0.shape] : x0.shape
  const xv = x0.values()
  const cells: number[][] = []
  const out = new Array<number>(N * Ch * Ho * Wo)
  for (let nc = 0; nc < N * Ch; nc++) {
    for (let i = 0; i < Ho; i++) {
      for (let j = 0; j < Wo; j++) {
        const cell: number[] = []
        for (let y = Math.floor((i * H) / Ho); y < Math.ceil(((i + 1) * H) / Ho); y++) {
          for (let x = Math.floor((j * W) / Wo); x < Math.ceil(((j + 1) * W) / Wo); x++) cell.push(nc * H * W + y * W + x)
        }
        const o = nc * Ho * Wo + i * Wo + j
        if (kind === 'avg') {
          out[o] = cell.reduce((s, f) => s + xv[f], 0) / cell.length
          cells[o] = cell
        } else {
          const at = cell.reduce((best, f) => (xv[f] > xv[best] ? f : best), cell[0])
          out[o] = xv[at]
          cells[o] = [at]
        }
      }
    }
  }
  const shape = x0.ndim === 3 ? [Ch, Ho, Wo] : [N, Ch, Ho, Wo]
  const res = record(E, TT.of(out, shape, x0.dtype), kind === 'avg' ? 'AdaptiveAvgPool2DBackward0' : 'AdaptiveMaxPool2DBackward0', [x0], (g) => {
    const gx = new Array<number>(xv.length).fill(0)
    cells.forEach((cell, o) => cell.forEach((f) => (gx[f] += g[o] / cell.length)))
    return [gx]
  })
  return { out: res, indices: TT.of(cells.map((c) => (kind === 'max' ? c[0] % (H * W) : 0)), shape, 'int64') }
}

/** every combination of one offset per axis, as flat offsets */
const crossOffsets = (lists: number[][], strides: number[]): number[] => {
  let out = [0]
  lists.forEach((l, a) => {
    const next: number[] = []
    for (const base of out) for (const v of l) next.push(base + v * strides[a])
    out = next
  })
  return out
}

const poolOutSize = (n: number, k: number, s: number, p: number, d: number, ceil: boolean): number => {
  const out = Math.floor((n + 2 * p - d * (k - 1) - 1 + (ceil ? s - 1 : 0)) / s) + 1
  return ceil && (out - 1) * s >= n + p ? out - 1 : out
}

export interface PoolOpts { k: number[]; s: number[]; p: number[]; d: number[]; ceil: boolean; countPad: boolean; divisor: number | null }

/** max or average pooling over the last k.length dimensions, with every option torch's pools have */
export function poolNd(E: Engine, x: TT, kind: 'max' | 'avg', { k, s, p, d, ceil, countPad, divisor }: PoolOpts): { out: TT; indices: TT } {
  const n = k.length
  const spatial = x.shape.slice(-n)
  const lead = x.shape.slice(0, -n)
  const outS = spatial.map((sz, a) => poolOutSize(sz, k[a], s[a], p[a], d[a], ceil))
  if (outS.some((o) => o < 1)) throw err('RuntimeError', `Given input size: (${lead.slice(-1).concat(spatial).join('x')}). Calculated output size: (${lead.slice(-1).concat(outS).join('x')}). Output size is too small`)
  const st = cStrides(spatial)
  const groups = prod(lead)
  const area = prod(spatial)
  // per axis and output position: the input positions in the window, and how wide the window is once padding is counted
  const windows = outS.map((o, a) => Array.from({ length: o }, (_, i) => {
    const start = i * s[a] - p[a]
    const inside: number[] = []
    for (let u = 0; u < k[a]; u++) {
      const at = start + u * d[a]
      if (at >= 0 && at < spatial[a]) inside.push(at)
    }
    return { inside, span: Math.min(start + k[a], spatial[a] + p[a]) - start }
  }))
  const cells: { offs: number[]; div: number }[] = []
  const walk = (a: number, picked: number[][], span: number) => {
    if (a === n) {
      const offs = crossOffsets(picked, st)
      cells.push({ offs, div: divisor ?? (countPad ? span : offs.length) })
      return
    }
    for (const w of windows[a]) walk(a + 1, [...picked, w.inside], span * w.span)
  }
  walk(0, [], 1)
  const xv = x.values()
  const out = new Array<number>(groups * cells.length)
  const arg = new Array<number>(out.length)
  for (let g = 0; g < groups; g++) {
    cells.forEach(({ offs, div }, c) => {
      const o = g * cells.length + c
      if (kind === 'max') {
        let best = -Infinity
        let at = offs[0] ?? 0
        for (const f of offs) {
          const v = xv[g * area + f]
          if (v > best) {
            best = v
            at = f
          }
        }
        out[o] = best
        arg[o] = at
      } else {
        let sum = 0
        for (const f of offs) sum += xv[g * area + f]
        out[o] = sum / div
        arg[o] = 0
      }
    })
  }
  const shape = [...lead, ...outS]
  const name = kind === 'max' ? `MaxPool${n}DWithIndicesBackward0` : `AvgPool${n}DBackward0`
  const res = record(E, TT.of(out, shape, x.dtype), name, [x], (g) => {
    const gx = new Array<number>(xv.length).fill(0)
    for (let gr = 0; gr < groups; gr++) {
      cells.forEach(({ offs, div }, c) => {
        const o = gr * cells.length + c
        if (kind === 'max') gx[gr * area + arg[o]] += g[o]
        else for (const f of offs) gx[gr * area + f] += g[o] / div
      })
    }
    return [gx]
  })
  return { out: res, indices: TT.of(arg, shape, 'int64') }
}

/** convolution or transposed convolution over any number of spatial dims; weight (out, in, *k), or (in, out, *k) when transposed */
export function convNd(E: Engine, x0: TT, w: TT, b: TT | null, s: number[], p: number[], d: number[], outPad: number[], transposed: boolean): TT {
  const n = s.length
  const unbatched = x0.ndim === n + 1
  const [N, Ch] = unbatched ? [1, x0.shape[0]] : x0.shape
  const S = x0.shape.slice(-n)
  const K = w.shape.slice(2)
  const O = transposed ? w.shape[1] : w.shape[0]
  const So = S.map((sz, a) => (transposed ? (sz - 1) * s[a] - 2 * p[a] + d[a] * (K[a] - 1) + outPad[a] + 1 : Math.floor((sz + 2 * p[a] - d[a] * (K[a] - 1) - 1) / s[a]) + 1))
  const [stS, stO, stK] = [cStrides(S), cStrides(So), cStrides(K)]
  const area = prod(S)
  const areaO = prod(So)
  const areaK = prod(K)
  // the (kernel, input, output) positions that meet, by axis, then all their combinations
  const triples: number[] = []
  const walk = (a: number, kf: number, inf: number, of: number) => {
    if (a === n) {
      triples.push(kf, inf, of)
      return
    }
    for (let u = 0; u < K[a]; u++) {
      const reach = transposed ? S[a] : So[a]
      for (let i = 0; i < reach; i++) {
        const far = i * s[a] + u * d[a] - p[a]
        const [ii, oo] = transposed ? [i, far] : [far, i]
        if (ii >= 0 && ii < S[a] && oo >= 0 && oo < So[a]) walk(a + 1, kf + u * stK[a], inf + ii * stS[a], of + oo * stO[a])
      }
    }
  }
  walk(0, 0, 0, 0)
  const xv = x0.values()
  const wv = w.values()
  const bv = b ? b.values() : null
  const each = (f: (xi: number, wi: number, oi: number) => void) => {
    for (let nn = 0; nn < N; nn++) {
      for (let c = 0; c < Ch; c++) {
        for (let o = 0; o < O; o++) {
          const wb = ((transposed ? c * O + o : o * Ch + c)) * areaK
          const xb = (nn * Ch + c) * area
          const ob = (nn * O + o) * areaO
          for (let t = 0; t < triples.length; t += 3) f(xb + triples[t + 1], wb + triples[t], ob + triples[t + 2])
        }
      }
    }
  }
  const out = new Array<number>(N * O * areaO).fill(0)
  each((xi, wi, oi) => (out[oi] += xv[xi] * wv[wi]))
  if (bv) for (let i = 0; i < out.length; i++) out[i] += bv[Math.floor(i / areaO) % O]
  const shape = unbatched ? [O, ...So] : [N, O, ...So]
  return record(E, TT.of(out, shape, promote(x0.dtype, w.dtype)), 'ConvolutionBackward0', [x0, w, b], (g) => {
    const gx = new Array<number>(xv.length).fill(0)
    const gw = new Array<number>(wv.length).fill(0)
    each((xi, wi, oi) => {
      gx[xi] += g[oi] * wv[wi]
      gw[wi] += g[oi] * xv[xi]
    })
    const gb = bv ? new Array<number>(O).fill(0) : null
    if (gb) g.forEach((x, i) => (gb[Math.floor(i / areaO) % O] += x))
    return [gx, gw, gb]
  })
}

export type PadMode = 'constant' | 'reflect' | 'replicate' | 'circular'

/** pad = [left, right] for the last dimension, then the one before it …; a negative amount crops */
export function padT(E: Engine, a: TT, pad: number[], mode: PadMode, value: number): TT {
  const maps = a.shape.map((n) => Array.from({ length: n }, (_, i) => i))
  for (let k = 0; k < pad.length / 2; k++) {
    const d = a.ndim - 1 - k
    const [l, r] = [pad[2 * k], pad[2 * k + 1]]
    const n = a.shape[d]
    if (mode === 'reflect' && (l >= n || r >= n)) throw err('RuntimeError', `Argument #${a.ndim}: Padding size should be less than the corresponding input dimension, but got: padding (${l}, ${r}) at dimension ${d} of input [${a.shape.join(', ')}]`)
    if (mode === 'circular' && (l > n || r > n)) throw err('RuntimeError', 'Padding value causes wrapping around more than once.')
    if (n + l + r < 0) throw err('RuntimeError', `The input size ${n}, plus negative padding ${l} and ${r}, resulted in a negative output size, which is invalid. Check dimension ${d} of your input.`)
    maps[d] = Array.from({ length: n + l + r }, (_, o) => {
      const i = o - l
      if (i >= 0 && i < n) return i
      if (mode === 'constant') return -1
      if (mode === 'replicate') return i < 0 ? 0 : n - 1
      if (mode === 'circular') return ((i % n) + n) % n
      const period = 2 * (n - 1)
      const m = period ? ((i % period) + period) % period : 0
      return m < n ? m : period - m
    })
  }
  return takeT(E, a, crossSrc(a.shape, maps), maps.map((m) => m.length), 'PadBackward0', value)
}

/** one axis of a resize: output position i is the weighted sum of input positions idx[i] (linear: 2 of them, bicubic: 4) */
export function weightedAxis(E: Engine, a: TT, d: number, idx: number[][], wt: number[][]): TT {
  const full = a.shape.map((n) => Array.from({ length: n }, (_, i) => i))
  const shape = a.shape.map((n, k) => (k === d ? idx.length : n))
  const terms = idx[0]?.length ?? 0
  const srcs = Array.from({ length: terms }, (_, t) => crossSrc(a.shape, full.map((m, k) => (k === d ? idx.map((l) => l[t]) : m))))
  const st = cStrides(shape)[d]
  const av = a.values()
  const n = prod(shape)
  const cell = (o: number) => Math.floor(o / st) % idx.length
  const out = new Array<number>(n)
  for (let o = 0; o < n; o++) {
    let s = 0
    const w = wt[cell(o)]
    for (let t = 0; t < terms; t++) s += w[t] * av[srcs[t][o]]
    out[o] = s
  }
  return record(E, TT.of(out, shape, a.dtype), 'UpsampleBackward0', [a], (g) => {
    const ga = new Array<number>(av.length).fill(0)
    g.forEach((x, o) => {
      const w = wt[cell(o)]
      for (let t = 0; t < terms; t++) ga[srcs[t][o]] += w[t] * x
    })
    return [ga]
  })
}

/** one axis of a linear resize: out[i] = (1 − t) · a[i0] + t · a[i1] */
export function linearAxis(E: Engine, a: TT, d: number, i0: number[], i1: number[], t: number[]): TT {
  return weightedAxis(E, a, d, i0.map((i, k) => [i, i1[k]]), t.map((w) => [1 - w, w]))
}

/** adaptive pooling over the last three dimensions of (C, D, H, W) or (N, C, D, H, W) */
export function adaptivePool3d(E: Engine, x: TT, out: number[], kind: 'max' | 'avg'): { out: TT; indices: TT } {
  const inSizes = x.shape.slice(-3)
  const lead = x.shape.slice(0, -3)
  const plane = prod(inSizes)
  const st = cStrides(inSizes)
  const ranges = out.map((o, k) => Array.from({ length: o }, (_, i) => [Math.floor((i * inSizes[k]) / o), Math.ceil(((i + 1) * inSizes[k]) / o)]))
  const xv = x.values()
  const outShape = [...lead, ...out]
  const n = prod(lead)
  const vals: number[] = []
  const idx: number[] = []
  const cells: number[][] = []
  for (let b = 0; b < n; b++) for (const [d0, d1] of ranges[0]) for (const [h0, h1] of ranges[1]) for (const [w0, w1] of ranges[2]) {
    const cell: number[] = []
    for (let d = d0; d < d1; d++) for (let r = h0; r < h1; r++) for (let c = w0; c < w1; c++) cell.push(b * plane + d * st[0] + r * st[1] + c)
    if (kind === 'avg') vals.push(cell.reduce((s, f) => s + xv[f], 0) / cell.length)
    else {
      const best = cell.reduce((p, q) => (xv[q] > xv[p] ? q : p), cell[0])
      vals.push(xv[best])
      idx.push(best - b * plane)
      cell.length = 0
      cell.push(best)
    }
    cells.push(cell)
  }
  const res = record(E, TT.of(vals, outShape, x.dtype), kind === 'avg' ? 'AdaptiveAvgPool3DBackward0' : 'AdaptiveMaxPool3DBackward0', [x], (g) => {
    const gx = new Array<number>(xv.length).fill(0)
    cells.forEach((cell, o) => cell.forEach((f) => (gx[f] += g[o] / cell.length)))
    return [gx]
  })
  return { out: res, indices: TT.of(kind === 'max' ? idx : new Array<number>(vals.length).fill(0), outShape, 'int64') }
}

/** F.grid_sample on (N, C, H, W): each output cell is up to four input cells, weighted by where the grid points */
export function gridSampleT(E: Engine, x: TT, grid: TT, mode: 'bilinear' | 'nearest', padding: 'zeros' | 'border' | 'reflection', corners: boolean): TT {
  const [N, Ch, H, W] = x.shape
  const [, Ho, Wo] = grid.shape
  const gv = grid.values()
  const xv = x.values()
  const place = (v: number, n: number) => (corners ? ((v + 1) / 2) * (n - 1) : ((v + 1) * n - 1) / 2)
  const reflect = (v: number, n: number) => {
    const [lo, hi] = corners ? [0, n - 1] : [-0.5, n - 0.5]
    const span = hi - lo
    if (span === 0) return 0
    const off = ((v - lo) % (2 * span) + 2 * span) % (2 * span)
    return off > span ? hi - (off - span) : lo + off
  }
  const fix = (v: number, n: number) => (padding === 'border' ? Math.min(n - 1, Math.max(0, v)) : padding === 'reflection' ? Math.min(n - 1, Math.max(0, reflect(v, n))) : v)
  // for each output cell of one channel: [input position in one (H, W) plane, or -1, its weight]
  const terms: [number, number][][] = []
  for (let n = 0; n < N; n++) for (let i = 0; i < Ho * Wo; i++) {
    const at = (n * Ho * Wo + i) * 2
    const px = fix(place(gv[at], W), W)
    const py = fix(place(gv[at + 1], H), H)
    const cell = (r: number, c: number, w: number): [number, number] => (r >= 0 && r < H && c >= 0 && c < W ? [r * W + c, w] : [-1, 0])
    if (mode === 'nearest') terms.push([cell(Math.round(py), Math.round(px), 1)])
    else {
      const [x0, y0] = [Math.floor(px), Math.floor(py)]
      const [fx, fy] = [px - x0, py - y0]
      terms.push([cell(y0, x0, (1 - fx) * (1 - fy)), cell(y0, x0 + 1, fx * (1 - fy)), cell(y0 + 1, x0, (1 - fx) * fy), cell(y0 + 1, x0 + 1, fx * fy)])
    }
  }
  const out = new Array<number>(N * Ch * Ho * Wo)
  const srcAt = (o: number): [number, number] => [Math.floor(o / (Ch * Ho * Wo)), Math.floor(o / (Ho * Wo)) % Ch]
  for (let o = 0; o < out.length; o++) {
    const [n, c] = srcAt(o)
    const base = (n * Ch + c) * H * W
    let s = 0
    for (const [p, w] of terms[n * Ho * Wo + (o % (Ho * Wo))]) if (p >= 0) s += w * xv[base + p]
    out[o] = s
  }
  return record(E, TT.of(out, [N, Ch, Ho, Wo], x.dtype), 'GridSampler2DBackward0', [x], (g) => {
    const gx = new Array<number>(xv.length).fill(0)
    g.forEach((v, o) => {
      const [n, c] = srcAt(o)
      const base = (n * Ch + c) * H * W
      for (const [p, w] of terms[n * Ho * Wo + (o % (Ho * Wo))]) if (p >= 0) gx[base + p] += w * v
    })
    return [gx]
  })
}

export function movedimT(E: Engine, a: TT, from: number, to: number): TT {
  const [f, t] = [normDim(from, a.ndim), normDim(to, a.ndim)]
  const order = a.shape.map((_, i) => i).filter((i) => i !== f)
  order.splice(t, 0, f)
  return permuteT(E, a, order, 'MovedimBackward0')
}

/** whole channels are zeroed: one draw per (N, C), the rest of the shape shares it */
export function dropoutChannels(E: Engine, x: TT, p: number, training: boolean): TT {
  if (p < 0 || p > 1) throw err('ValueError', `dropout probability has to be between 0 and 1, but got ${p}`)
  if (!training || p === 0) return x
  const groups = prod(x.shape.slice(0, 2))
  const per = x.size / Math.max(1, groups)
  const keep = Array.from({ length: groups }, () => (E.rng.next() >= p ? 1 / (1 - p) : 0))
  return record(E, TT.of(x.values().map((v, i) => v * keep[Math.floor(i / per)]), x.shape, x.dtype), 'MulBackward0', [x], (g) => [g.map((gi, i) => gi * keep[Math.floor(i / per)])])
}

export function embeddingT(E: Engine, w: TT, idx: TT, paddingIdx: number | null, scaleFreq = false): TT {
  if (idx.dtype !== 'int64' && idx.dtype !== 'int32') throw err('RuntimeError', `Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.${typeLabel(idx.dtype)}Tensor instead (while checking arguments for embedding)`)
  const [rows, D] = w.shape
  const iv = idx.values()
  if (iv.some((i) => i < 0 || i >= rows)) throw err('IndexError', 'index out of range in self')
  const wv = w.values()
  const out = new Array<number>(iv.length * D)
  iv.forEach((i, n) => { for (let j = 0; j < D; j++) out[n * D + j] = wv[i * D + j] })
  return record(E, TT.raw(out, [...idx.shape, D], w.dtype), 'EmbeddingBackward0', [w], (g) => {
    const gw = new Array<number>(wv.length).fill(0)
    iv.forEach((i, n) => { if (i !== paddingIdx) for (let j = 0; j < D; j++) gw[i * D + j] += g[n * D + j] })
    if (scaleFreq) {
      // scale_grad_by_freq: each row's gradient divided by how often the batch used it
      const count = new Map<number, number>()
      for (const i of iv) count.set(i, (count.get(i) ?? 0) + 1)
      count.forEach((c, i) => { for (let j = 0; j < D; j++) gw[i * D + j] /= c })
    }
    return [gw]
  })
}

/** class indices with class weights, ignore_index and label smoothing, or class probabilities; (N, C) log-probabilities. −Σ_c w_c q_c log p_c per sample */
export function weightedNll(E: Engine, lp: TT, target: TT, weight: number[] | null, ignore: number, smoothing: number, reduction: Reduction): TT {
  const [N, Cl] = lp.ndim === 1 ? [1, lp.shape[0]] : lp.shape
  const tv = target.values()
  const probs = isFloatT(target.dtype) && target.shape.join() === lp.shape.join()
  if (!probs) {
    checkClassTarget(lp, target)
    if (target.dtype !== 'int64' && target.dtype !== 'uint8') throw err('RuntimeError', `expected target dtype to be Long or Byte, but got ${typeLabel(target.dtype)}`)
  }
  const q = new Array<number>(N * Cl).fill(0)
  let denom = probs ? N : 0
  for (let n = 0; n < N; n++) {
    if (!probs && tv[n] === ignore) continue
    if (!probs && (tv[n] < 0 || tv[n] >= Cl)) throw err('IndexError', `Target ${tv[n]} is out of bounds.`)
    if (!probs) denom += weight ? weight[tv[n]] : 1
    for (let c = 0; c < Cl; c++) {
      const p = probs ? tv[n * Cl + c] : c === tv[n] ? 1 : 0
      q[n * Cl + c] = ((1 - smoothing) * p + smoothing / Cl) * (weight ? weight[c] : 1)
    }
  }
  const dim = lp.ndim === 1 ? 0 : 1
  const coef = TT.of(q, lp.shape, lp.dtype)
  const per = reduceSum(E, binary(E, '*', unary(E, 'neg', lp), coef), [dim], false, false)
  if (reduction === 'none') return per
  const total = reduceSum(E, per, null, false, false)
  return reduction === 'sum' ? total : binary(E, '/', total, TT.scalar(denom, lp.dtype), false, true)
}

/** a loss applied pointwise to input − target */
export function pointLoss(E: Engine, a: TT, b: TT, f: (d: number) => number, df: (d: number) => number, name: string, reduction: Reduction): TT {
  const per = mapT(E, binary(E, '-', a, b), f, df, name)
  const out = reduceLoss(E, per, reduction)
  if (out.fn) {
    out.fn.name = name
    out.fn.keeps = [a, a.version]
  }
  return out
}

/** erf with enough digits for float64: a positive series below 3, then 1 − erfc by continued fraction */
export const erf = (x: number): number => {
  const a = Math.abs(x)
  if (Number.isNaN(x)) return x
  if (a >= 3) return Math.sign(x) * (1 - erfc(a))
  let term = a
  let sum = a
  for (let n = 1; term > 1e-17 * sum; n++) {
    term *= (2 * a * a) / (2 * n + 1)
    sum += term
  }
  return Math.sign(x) * (2 / Math.sqrt(Math.PI)) * Math.exp(-a * a) * sum
}

export const erfc = (x: number): number => {
  if (x < 3) return 1 - erf(x)
  let f = x
  for (let k = 60; k >= 1; k--) f = x + k / 2 / f
  return Math.exp(-x * x) / (Math.sqrt(Math.PI) * f)
}

// ---------------------------------------------------------------- printing (a port of torch/_tensor_str.py)

export const PRINT_OPTS = { precision: 4, threshold: 1000, edgeitems: 3, linewidth: 80, sciMode: null as boolean | null }

export const setPrintProfile = (profile: 'default' | 'short' | 'full') => {
  PRINT_OPTS.precision = profile === 'short' ? 2 : 4
  PRINT_OPTS.threshold = profile === 'full' ? Infinity : 1000
  PRINT_OPTS.edgeitems = profile === 'short' ? 2 : 3
  PRINT_OPTS.linewidth = 80
}

/** toFixed / toExponential round an exact tie up; Python's format rounds it to even (f'{0.03125:.4f}' is 0.0312) */
const evenTie = (x: number, digits: number, render: (x: number, d: number) => string, cut: (s: string, d: number) => string): string => {
  const out = render(x, digits)
  if (digits + 40 > 100) return out
  const long = render(x, digits + 40)
  const mant = long.split('e')[0]
  if (!/50{39}$/.test(mant)) return out
  const down = cut(long, digits)
  return Number(down.split('e')[0].slice(-1)) % 2 === 0 ? down : out
}

/** Python's f'{x:.{p}f}' */
export const pyFixed = (x: number, p: number): string => {
  if (Number.isNaN(x)) return 'nan'
  if (!Number.isFinite(x)) return x > 0 ? 'inf' : '-inf'
  const body = evenTie(Math.abs(x), p, (v, d) => v.toFixed(d), (s, d) => (d ? s.slice(0, s.indexOf('.') + d + 1) : s.slice(0, s.indexOf('.'))))
  return (x < 0 || Object.is(x, -0) ? '-' : '') + body
}

/** Python's f'{x:.{p}e}' */
export const pyExp = (x: number, p: number): string => {
  if (Number.isNaN(x)) return 'nan'
  if (!Number.isFinite(x)) return x > 0 ? 'inf' : '-inf'
  const body = evenTie(Math.abs(x), p, (v, d) => v.toExponential(d), (s, d) => {
    const [m, e] = s.split('e')
    return `${m.slice(0, d ? m.indexOf('.') + d + 1 : 1)}e${e}`
  })
  return (x < 0 || Object.is(x, -0) ? '-' : '') + body.replace(/e([+-])(\d)$/, 'e$10$2')
}

/** _Formatter: one width and one notation for all the values shown */
const makeFormatter = (values: number[], dtype: TDType) => {
  const floating = isFloatT(dtype)
  let intMode = true
  let sciMode = false
  let maxWidth = 1
  const plain = (v: number) => (dtype === 'bool' ? (v ? 'True' : 'False') : String(v))
  if (!floating) {
    for (const v of values) maxWidth = Math.max(maxWidth, plain(v).length)
  } else {
    const nz = values.filter((v) => Number.isFinite(v) && v !== 0)
    if (nz.length) {
      let min = Infinity
      let max = 0
      for (const v of nz) {
        const a = Math.abs(v)
        if (a < min) min = a
        if (a > max) max = a
        if (v !== Math.ceil(v)) intMode = false
      }
      sciMode = PRINT_OPTS.sciMode ?? (max / min > 1000 || max > 1e8 || min < 1e-4)
      for (const v of nz) {
        if (sciMode) maxWidth = Math.max(maxWidth, pyExp(v, PRINT_OPTS.precision).length)
        else if (intMode) maxWidth = Math.max(maxWidth, pyFixed(v, 0).length + 1)
        else maxWidth = Math.max(maxWidth, pyFixed(v, PRINT_OPTS.precision).length)
      }
    }
  }
  const format = (v: number): string => {
    let ret: string
    if (!floating) ret = plain(v)
    else if (sciMode) ret = pyExp(v, PRINT_OPTS.precision)
    else if (intMode) ret = pyFixed(v, 0) + (Number.isFinite(v) ? '.' : '')
    else ret = pyFixed(v, PRINT_OPTS.precision)
    return ret.padStart(maxWidth)
  }
  return { format, width: maxWidth }
}

/** get_summarized_data: the flat positions the formatter gets to see */
const summarized = (shape: number[], base: number, stride: number[], out: number[]) => {
  const edge = PRINT_OPTS.edgeitems
  if (!shape.length) {
    out.push(base)
    return
  }
  const n = shape[0]
  if (shape.length === 1) {
    // self[-0:] is all of it, so edgeitems=0 leaves the whole vector here
    if (edge && n > 2 * edge) for (const i of [...Array(edge).keys(), ...Array.from({ length: edge }, (_, k) => n - edge + k)]) out.push(base + i)
    else for (let i = 0; i < n; i++) out.push(base + i)
    return
  }
  if (!edge) return
  const rows = n > 2 * edge ? [...Array(edge).keys(), ...Array.from({ length: edge }, (_, k) => n - edge + k)] : [...Array(n).keys()]
  for (const i of rows) summarized(shape.slice(1), base + i * stride[0], stride.slice(1), out)
}

/** _tensor_str */
export function tensorBody(shape: number[], values: number[], dtype: TDType, indent: number): string {
  if (values.length === 0) return '[]'
  const summarize = values.length > PRINT_OPTS.threshold
  const edge = PRINT_OPTS.edgeitems
  const st = cStrides(shape)
  let seen = values
  if (summarize) {
    const at: number[] = []
    summarized(shape, 0, st, at)
    seen = at.map((f) => values[f])
  }
  const fmt = makeFormatter(seen, dtype)
  const vector = (d: number, base: number, ind: number): string => {
    const n = shape[d]
    const perLine = Math.max(1, Math.floor((PRINT_OPTS.linewidth - ind) / (fmt.width + 2)))
    let data: string[]
    if (summarize && !edge) data = ['...']
    else if (summarize && n > 2 * edge) data = [...Array.from({ length: edge }, (_, i) => fmt.format(values[base + i])), ' ...', ...Array.from({ length: edge }, (_, i) => fmt.format(values[base + n - edge + i]))]
    else data = Array.from({ length: n }, (_, i) => fmt.format(values[base + i]))
    const lines: string[] = []
    for (let i = 0; i < data.length; i += perLine) lines.push(data.slice(i, i + perLine).join(', '))
    return `[${lines.join(`,\n${' '.repeat(ind + 1)}`)}]`
  }
  const rec = (d: number, base: number, ind: number): string => {
    const left = shape.length - d
    if (left === 0) return fmt.format(values[base])
    if (left === 1) return vector(d, base, ind)
    const n = shape[d]
    const sub = (i: number) => rec(d + 1, base + i * st[d], ind + 1)
    const slices = summarize && n > 2 * edge
      ? [...Array.from({ length: edge }, (_, i) => sub(i)), '...', ...Array.from({ length: edge }, (_, i) => sub(n - edge + i))]
      : Array.from({ length: n }, (_, i) => sub(i))
    return `[${slices.join(`,${'\n'.repeat(left - 1)}${' '.repeat(ind + 1)}`)}]`
  }
  return rec(0, 0, indent)
}

const sizeTuple = (shape: number[]) => `(${shape.join(', ')}${shape.length === 1 ? ',' : ''})`

/** _str_intern and _add_suffixes */
export function formatTensor(t: TT, opts: { prefix?: string; forceDtype?: boolean } = {}): string {
  const prefix = opts.prefix ?? 'tensor('
  const indent = prefix.length
  const suffixes: string[] = []
  const hasDefault = t.dtype === DEFAULTS.float || t.dtype === 'int64' || t.dtype === 'bool'
  let body: string
  if (t.size === 0) {
    if (t.ndim !== 1) suffixes.push(`size=${sizeTuple(t.shape)}`)
    if (t.dtype !== DEFAULTS.float) suffixes.push(`dtype=torch.${t.dtype}`)
    body = '[]'
  } else {
    if (!PRINT_OPTS.edgeitems) suffixes.push(`size=${sizeTuple(t.shape)}`)
    if (!hasDefault || opts.forceDtype) suffixes.push(`dtype=torch.${t.dtype}`)
    body = tensorBody(t.shape, t.values(), t.dtype, indent)
  }
  if (t.fn) suffixes.push(`grad_fn=<${t.fn.name}>`)
  else if (t.requiresGrad) suffixes.push('requires_grad=True')
  const text = prefix + body
  let out = text
  let last = text.length - text.lastIndexOf('\n') + 1
  for (const suffix of suffixes) {
    if (last + suffix.length + 2 > PRINT_OPTS.linewidth) {
      out += `,\n${' '.repeat(indent)}${suffix}`
      last = indent + suffix.length
    } else {
      out += `, ${suffix}`
      last += suffix.length + 2
    }
  }
  return `${out})`
}
