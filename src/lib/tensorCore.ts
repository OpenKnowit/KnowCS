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

export type TDType = 'float32' | 'float64' | 'int64' | 'int32' | 'uint8' | 'bool'

export const isFloatT = (d: TDType) => d === 'float32' || d === 'float64'
const storageOf = (d: TDType): DType => (isFloatT(d) ? 'float64' : d === 'bool' ? 'bool' : 'int64')
const err = (type: string, msg: string) => new PyError(type, msg)

/** round to the dtype: float32 storage keeps float32 precision, as torch.tensor(0.1).item() shows */
export const castT = (v: number, d: TDType): number => {
  switch (d) {
    case 'float32': return Math.fround(v)
    case 'float64': return v
    case 'bool': return v ? 1 : 0
    case 'uint8': return ((Math.trunc(v) % 256) + 256) % 256
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
}

let nextId = 1

export class TT {
  a: NDArray
  dtype: TDType
  requiresGrad = false
  grad: TT | null = null
  fn: GradNode | null = null
  retain = false
  /** nn.Parameter */
  isParam = false
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
  constructor(rng: SandboxRandom) { this.rng = rng }
}

const needsGrad = (E: Engine, ...xs: (TT | null)[]) => E.grad && xs.some((x) => x?.requiresGrad)

/** attach a grad_fn when any input requires grad */
const record = (E: Engine, out: TT, name: string, inputs: (TT | null)[], backward: GradNode['backward']): TT => {
  if (needsGrad(E, ...inputs)) {
    out.requiresGrad = true
    out.fn = { name, inputs, backward, freed: false }
  }
  return out
}

// ---------------------------------------------------------------- dtype promotion

const RANK: Record<TDType, number> = { bool: 0, uint8: 1, int32: 2, int64: 3, float32: 4, float64: 5 }

/** result dtype of a ⊕ b; weak = a Python scalar (it never widens a tensor of its own kind) */
export const promote = (a: TDType, b: TDType, aWeak = false, bWeak = false): TDType => {
  if (aWeak && !bWeak) return isFloatT(a) && !isFloatT(b) ? 'float32' : b === 'bool' && a !== 'bool' ? (isFloatT(a) ? 'float32' : 'int64') : b
  if (bWeak && !aWeak) return promote(b, a, true, false)
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
  if (op === '/' && !isFloatT(dtype)) dtype = 'float32'
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
  const dtype = keepsInt || isFloatT(a.dtype) ? (a.dtype === 'bool' ? 'int64' : a.dtype) : 'float32'
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
  })])
}

export function clamp(E: Engine, a: TT, lo: number | null, hi: number | null): TT {
  const av = a.values()
  const out = TT.of(av.map((x) => Math.min(hi ?? Infinity, Math.max(lo ?? -Infinity, x))), a.shape, a.dtype)
  return record(E, out, 'ClampBackward1', [a], (g) => [g.map((gi, i) => ((lo === null || av[i] >= lo) && (hi === null || av[i] <= hi) ? gi : 0))])
}

// ---------------------------------------------------------------- matmul

const dims2 = (s: number[]) => `${s.join('x')}`

export function matmul(E: Engine, a: TT, b: TT, name = 'MmBackward0'): TT {
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
  const r = d < 0 ? d + nd : d
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
  if (mean && !isFloatT(a.dtype)) throw err('RuntimeError', `mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: ${a.dtype === 'int64' ? 'Long' : a.dtype}`)
  const { outShapeKeep, map, outSize } = reduceMap(a.shape, ds)
  const av = a.values()
  const out = new Array<number>(outSize).fill(0)
  av.forEach((v, i) => (out[map[i]] += v))
  const count = av.length / Math.max(1, outSize)
  if (mean) for (let i = 0; i < outSize; i++) out[i] /= count
  const shape = keepdim ? outShapeKeep : a.shape.filter((_, d) => !ds.includes(d))
  const dtype = mean ? a.dtype : a.dtype === 'bool' || a.dtype === 'int32' || a.dtype === 'uint8' ? 'int64' : a.dtype
  const res = TT.of(out, shape, dtype)
  return record(E, res, mean ? 'MeanBackward' + (dims === null ? '0' : '1') : 'SumBackward' + (dims === null ? '0' : '1'), [a], (g) => [Array.from({ length: av.length }, (_, i) => g[map[i]] / (mean ? count : 1))])
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
  return record(E, out, mustView ? 'ViewBackward0' : 'ReshapeAliasBackward0', [a], (g) => [g])
}

export function permuteT(E: Engine, a: TT, perm: number[], name = 'PermuteBackward0'): TT {
  const p = perm.map((d) => normDim(d, a.ndim))
  if (new Set(p).size !== p.length || p.length !== a.ndim) throw err('RuntimeError', `permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = ${a.ndim} is not equal to len(dims) = ${p.length}`)
  const out = new TT(ndTranspose(a.a, p), a.dtype)
  // where each output element comes from, to send gradients back
  const ids = new TT(ndTranspose(NDArray.create(Array.from({ length: a.size }, (_, i) => i), a.shape, 'int64'), p), 'int64').values()
  return record(E, out, name, [a], (g) => {
    const ga = new Array<number>(a.size)
    ids.forEach((src, i) => (ga[src] = g[i]))
    return [ga]
  })
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
  if (out.fn) out.fn.name = 'SqueezeBackward0'
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
  if (!ts.length) throw err('RuntimeError', `torch.${stack ? 'stack' : 'cat'}(): expected a non-empty list of Tensors`)
  const parts = stack ? ts.map((t) => unsqueezeT({ grad: false } as Engine, t, dim)) : ts
  const nd = parts[0].ndim
  const d = normDim(dim, nd)
  for (const t of parts) {
    if (t.ndim !== nd || t.shape.some((s, i) => i !== d && s !== parts[0].shape[i])) {
      throw err('RuntimeError', stack ? `stack expects each tensor to be equal size, but got [${ts[0].shape.join(', ')}] at entry 0 and [${t.shape.join(', ')}] at entry ${parts.indexOf(t)}` : `Sizes of tensors must match except in dimension ${d}. Expected size ${parts[0].shape.find((_, i) => i !== d)} but got size ${t.shape.find((_, i) => i !== d)} for tensor number ${parts.indexOf(t)} in the list.`)
    }
  }
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
  const flatOf = new Map(a.a.addresses().map((x, i) => [x, i]))
  const src = plan.addresses.map((x) => flatOf.get(x)!)
  const arr = typeof value === 'number' ? (plan.view ? a.a.view([], [], plan.addresses[0]) : NDArray.create([value], [], a.a.dtype)) : value
  const out = new TT(arr, a.dtype)
  return record(E, out, plan.advanced ? 'IndexBackward0' : plan.scalar ? 'SelectBackward0' : 'SliceBackward0', [a], (g) => {
    const ga = new Array<number>(a.size).fill(0)
    src.forEach((s, i) => (ga[s] += g[i]))
    return [ga]
  })
}

export function detach(a: TT): TT {
  return new TT(a.a, a.dtype)
}

export function cloneT(E: Engine, a: TT): TT {
  return record(E, TT.of(a.values(), a.shape, a.dtype), 'CloneBackward0', [a], (g) => [g])
}

export function toDtype(E: Engine, a: TT, dtype: TDType): TT {
  if (dtype === a.dtype) return a
  const out = TT.of(a.values(), a.shape, dtype)
  return isFloatT(dtype) ? record(E, out, 'ToCopyBackward0', [a], (g) => [g]) : out
}

/** write values into a tensor's storage (in-place ops, x[idx] = v, optimiser steps) */
export function writeInto(a: TT, values: number[], addresses = a.a.addresses()) {
  addresses.forEach((x, i) => (a.a.data[x] = castT(values[i], a.dtype)))
}

// ---------------------------------------------------------------- neural-network kernels

export function linear(E: Engine, x: TT, w: TT, b: TT | null): TT {
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
  const dtype = isFloatT(x.dtype) ? x.dtype : 'float32'
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

/** negative log-likelihood of class indices, input = log-probabilities (N, C) or (C) */
export function nllLoss(E: Engine, logp: TT, target: TT, reduction: Reduction = 'mean'): TT {
  if (isFloatT(target.dtype)) throw err('RuntimeError', `expected scalar type Long but found ${target.dtype === 'float32' ? 'Float' : 'Double'}`)
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
  const res = reduction === 'none' ? TT.of(per, a.shape, a.dtype) : TT.of([per.reduce((s, v) => s + v, 0) / (reduction === 'mean' ? n : 1)], [], a.dtype)
  return record(E, res, 'MseLossBackward0', [a, b], (g) => {
    const scale = (i: number) => (reduction === 'none' ? g[i] : g[0] / (reduction === 'mean' ? n : 1))
    const ga = av.map((x, i) => 2 * (x - bv[i]) * scale(i))
    return [ga, ga.map((v) => -v)]
  })
}

export function bce(E: Engine, p: TT, y: TT, logits: boolean, reduction: Reduction = 'mean'): TT {
  if (p.shape.join() !== y.shape.join()) throw err('ValueError', `Using a ${sizeMsg(p, y)} is deprecated. Please ensure they have the same size.`)
  const pv = p.values()
  const yv = y.values()
  if (!logits && pv.some((v) => v < 0 || v > 1)) throw err('RuntimeError', 'all elements of input should be between 0 and 1')
  const eps = (v: number) => Math.max(-100, Math.log(v))
  const per = pv.map((v, i) => (logits ? Math.max(v, 0) - v * yv[i] + Math.log1p(Math.exp(-Math.abs(v))) : -(yv[i] * eps(v) + (1 - yv[i]) * eps(1 - v))))
  const n = per.length
  const res = reduction === 'none' ? TT.of(per, p.shape, p.dtype) : TT.of([per.reduce((s, v) => s + v, 0) / (reduction === 'mean' ? n : 1)], [], p.dtype)
  return record(E, res, logits ? 'BinaryCrossEntropyWithLogitsBackward0' : 'BinaryCrossEntropyBackward0', [p, y], (g) => {
    const scale = (i: number) => (reduction === 'none' ? g[i] : g[0] / (reduction === 'mean' ? n : 1))
    return [pv.map((v, i) => (logits ? 1 / (1 + Math.exp(-v)) - yv[i] : (v - yv[i]) / Math.max(v * (1 - v), 1e-12)) * scale(i)), null]
  })
}

// ---------------------------------------------------------------- backward

export function backward(E: Engine, root: TT, gradOut: number[] | null, retainGraph: boolean): TT[] {
  if (!root.requiresGrad) throw err('RuntimeError', 'element 0 of tensors does not require grad and does not have a grad_fn')
  if (!gradOut && root.size !== 1) throw err('RuntimeError', 'grad can be implicitly created only for scalar outputs')
  // topological order (inputs after the tensors that use them)
  const order: TT[] = []
  const seen = new Set<TT>()
  const visit = (t: TT) => {
    if (seen.has(t)) return
    seen.add(t)
    for (const x of t.fn?.inputs ?? []) if (x && x.requiresGrad) visit(x)
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
      const g = grads.get(t)
      if (!g) continue
      if (t.fn) {
        const gs = t.fn.backward(g)
        t.fn.inputs.forEach((x, i) => {
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

// ---------------------------------------------------------------- printing (torch._tensor_str)

const fmtE = (v: number) => v.toExponential(4).replace(/e([+-])(\d)$/, 'e$10$2')

/** element strings, padded to one width, as PyTorch prints them */
export function formatElements(values: number[], dtype: TDType): string[] {
  if (dtype === 'bool') return values.map((v) => (v ? 'True' : 'False').padStart(values.length > 0 && values.some((x) => !x) ? 5 : 4))
  if (!isFloatT(dtype)) {
    const s = values.map((v) => String(v))
    const w = Math.max(...s.map((x) => x.length))
    return s.map((x) => x.padStart(w))
  }
  const finite = values.filter((v) => Number.isFinite(v))
  const nz = finite.filter((v) => v !== 0).map(Math.abs)
  let fmt: (v: number) => string
  if (!nz.length) fmt = (v) => (Number.isFinite(v) ? `${v === 0 ? 0 : v}.` : String(v))
  else {
    const max = Math.max(...nz)
    const min = Math.min(...nz)
    const allInt = finite.every((v) => Number.isInteger(v))
    if (allInt) fmt = max / min > 1000 || max > 1e8 ? fmtE : (v) => `${v.toFixed(0)}.`
    else if (max / min > 1000 || max > 1e8 || min < 1e-4) fmt = fmtE
    else fmt = (v) => v.toFixed(4)
  }
  const s = values.map((v) => (Number.isNaN(v) ? 'nan' : v === Infinity ? 'inf' : v === -Infinity ? '-inf' : fmt(v)))
  const w = Math.max(...s.map((x) => x.length))
  return s.map((x) => x.padStart(w))
}

const SUMMARY_AT = 1000
const EDGE = 3
const LINE = 80

export function tensorBody(shape: number[], values: number[], dtype: TDType, indent: number): string {
  if (!shape.length) return formatElements(values, dtype)[0].trim()
  if (values.length === 0) return '[]'
  const summarize = values.length > SUMMARY_AT
  const strs = formatElements(values, dtype)
  const st = cStrides(shape)
  const rec = (d: number, base: number, ind: number): string => {
    const n = shape[d]
    const idx = summarize && n > 2 * EDGE ? [...Array.from({ length: EDGE }, (_, i) => i), -1, ...Array.from({ length: EDGE }, (_, i) => n - EDGE + i)] : Array.from({ length: n }, (_, i) => i)
    if (d === shape.length - 1) {
      const items = idx.map((i) => (i < 0 ? '...' : strs[base + i]))
      const width = (strs[0]?.length ?? 1) + 2
      const per = Math.max(1, Math.floor((LINE - ind) / width))
      const lines: string[] = []
      for (let i = 0; i < items.length; i += per) lines.push(items.slice(i, i + per).join(', '))
      return `[${lines.join(`,\n${' '.repeat(ind + 1)}`)}]`
    }
    const parts = idx.map((i) => (i < 0 ? '...' : rec(d + 1, base + i * st[d], ind + 1)))
    return `[${parts.join(`,${'\n'.repeat(shape.length - d - 1)}${' '.repeat(ind + 1)}`)}]`
  }
  return rec(0, 0, indent)
}

const DEFAULT_DTYPES: TDType[] = ['float32', 'int64', 'bool']

export function formatTensor(t: TT, opts: { prefix?: string; forceDtype?: boolean } = {}): string {
  const prefix = opts.prefix ?? 'tensor('
  let s = prefix + tensorBody(t.shape, t.values(), t.dtype, prefix.length)
  if (t.size === 0 && t.dtype !== 'float32') s += `, dtype=torch.${t.dtype}`
  else if (!DEFAULT_DTYPES.includes(t.dtype) || opts.forceDtype) s += `, dtype=torch.${t.dtype}`
  if (t.fn) s += `, grad_fn=<${t.fn.name}>`
  else if (t.requiresGrad) s += ', requires_grad=True'
  return s + ')'
}
