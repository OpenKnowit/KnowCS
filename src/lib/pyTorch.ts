/**
 * torch for the sandbox (lecture 9): tensors with PyTorch's printing and dtype rules, autograd (the recorded graph is
 * shown after backward()), nn layers and nn.Module subclasses (each forward pass is traced layer by layer with its
 * shapes and parameter counts), losses, optimisers, TensorDataset / DataLoader and torch.from_numpy's shared memory.
 * Built on tensorCore; random values come from the sandbox generator, so they differ from real PyTorch.
 */
import { NDArray, PyError, broadcastShapes, cStrides, prod, setIndex, sliceIndices } from './ndarray'
import type { DType, IndexItem } from './ndarray'
import { PyDict, PyObj, formatSpec, isNum, numOf, py, repr, toArray } from './minipy'
import type { GridSnapshot, Host, HostClass, InstValue, Kw, PyLib, Value } from './minipy'
import type { FlowRow, GraphNode } from './pyEvents'
import * as C from './tensorCore'
import { Engine, TT, formatTensor, isFloatT } from './tensorCore'
import type { TDType } from './tensorCore'

// ---------------------------------------------------------------- per-run state

interface FlowState {
  /** where the outermost call was made (the event belongs to that line) */
  line: number
  code: string
  rows: FlowRow[]
  depth: number
  names: Map<object, string>
  model: string
  root: object
}

interface TorchState {
  E: Engine
  flow: FlowState | null
  /** torch.save / torch.load keep objects here (no files in the sandbox) */
  files: Map<string, Value>
  notes: Set<string>
  /** the last model called, to name parameters in the autograd graph */
  lastModel: Value | null
}

const stateOf = (h: Host): TorchState => {
  let s = h.state.get('torch') as TorchState | undefined
  if (!s) {
    s = { E: new Engine(h.rng), flow: null, files: new Map(), notes: new Set(), lastModel: null }
    h.state.set('torch', s)
  }
  return s
}

const noteOnce = (h: Host, key: string, tone: 'info' | 'warn', params: Record<string, string | number> = {}) => {
  const s = stateOf(h)
  if (s.notes.has(key) || !h.tracing) return
  s.notes.add(key)
  h.emit({ type: 'note', tone, key, params })
}

// ---------------------------------------------------------------- small objects

export const DTYPES: Record<string, TDType> = {
  float32: 'float32', float: 'float32', float64: 'float64', double: 'float64', int64: 'int64', long: 'int64',
  int32: 'int32', int: 'int32', uint8: 'uint8', bool: 'bool',
}

export class DtypeObj extends PyObj {
  readonly cls = 'dtype'
  d: TDType
  constructor(d: TDType) {
    super()
    this.d = d
  }
  repr() { return `torch.${this.d}` }
  binop(op: string, other: Value): Value | undefined {
    if (op !== '==' && op !== '!=') return undefined
    const same = other.k === 'obj' && other.o instanceof DtypeObj && other.o.d === this.d
    return py.bool(op === '==' ? same : !same)
  }
  getAttr(name: string): Value | undefined {
    if (name === 'is_floating_point') return py.bool(isFloatT(this.d))
    return undefined
  }
}

class DeviceObj extends PyObj {
  readonly cls = 'device'
  type: string
  constructor(type: string) {
    super()
    this.type = type
  }
  repr() { return `device(type='${this.type}')` }
  str() { return this.type }
  getAttr(name: string): Value | undefined {
    return name === 'type' ? py.str(this.type) : undefined
  }
  binop(op: string, other: Value): Value | undefined {
    if (op !== '==' && op !== '!=') return undefined
    const t = other.k === 'obj' && other.o instanceof DeviceObj ? other.o.type : other.k === 'str' ? other.v : null
    return py.bool((t === this.type) === (op === '=='))
  }
}

/** torch.Size: a tuple of ints */
export class SizeObj extends PyObj {
  readonly cls = 'Size'
  dims: number[]
  constructor(dims: number[]) {
    super()
    this.dims = [...dims]
  }
  repr() { return `torch.Size([${this.dims.join(', ')}])` }
  len() { return this.dims.length }
  iter() { return this.dims.map(py.int) }
  isa(name: string) { return name === 'tuple' }
  getItem(idx: Value, h: Host): Value {
    if (idx.k === 'slice') {
      const part = (x: Value) => (x.k === 'none' ? null : h.toInt(x))
      const { start, step, len } = sliceIndices(part(idx.start), part(idx.stop), part(idx.step), this.dims.length)
      return py.obj(new SizeObj(Array.from({ length: len }, (_, k) => this.dims[start + k * step])))
    }
    const i = h.toInt(idx)
    const n = this.dims.length
    if (i < -n || i >= n) throw h.err('IndexError', 'tuple index out of range')
    return py.int(this.dims[i < 0 ? i + n : i])
  }
  binop(op: string, other: Value, _r: boolean, h: Host): Value | undefined {
    if (op === '==' || op === '!=') {
      const o = other.k === 'obj' && other.o instanceof SizeObj ? other.o.dims : other.k === 'tuple' || other.k === 'list' ? other.items.map((v) => (isNum(v) ? numOf(v) : NaN)) : null
      const same = !!o && o.length === this.dims.length && o.every((d, i) => d === this.dims[i])
      return py.bool(op === '==' ? same : !same)
    }
    if (op === '+' && (other.k === 'tuple' || (other.k === 'obj' && other.o instanceof SizeObj))) {
      const more = other.k === 'tuple' ? other.items.map((v) => h.toInt(v)) : (other.o as SizeObj).dims
      return py.obj(new SizeObj([...this.dims, ...more]))
    }
    return undefined
  }
  getAttr(name: string): Value | undefined {
    if (name === 'numel') return { k: 'fn', name, call: () => py.int(prod(this.dims)) }
    return undefined
  }
}

/** torch.return_types.max: (values, indices) */
class ReturnTypes extends PyObj {
  readonly cls = 'return_types'
  name: string
  values: Value
  indices: Value
  constructor(name: string, values: Value, indices: Value) {
    super()
    this.name = name
    this.values = values
    this.indices = indices
  }
  repr() { return `torch.return_types.${this.name}(\nvalues=${repr(this.values)},\nindices=${repr(this.indices)})` }
  iter() { return [this.values, this.indices] }
  len() { return 2 }
  getItem(idx: Value, h: Host) {
    const i = h.toInt(idx)
    if (i === 0 || i === -2) return this.values
    if (i === 1 || i === -1) return this.indices
    throw h.err('IndexError', 'tuple index out of range')
  }
  getAttr(name: string) { return name === 'values' ? this.values : name === 'indices' ? this.indices : undefined }
}

class GradFnObj extends PyObj {
  readonly cls: string
  constructor(name: string) {
    super()
    this.cls = name
  }
  repr() { return `<${this.cls} object>` }
  getAttr(name: string) { return name === 'name' ? { k: 'fn' as const, name, call: () => py.str(this.cls) } : undefined }
}

/** with torch.no_grad(): … */
class GradMode extends PyObj {
  readonly cls: string
  enabled: boolean
  prev = true
  constructor(enabled: boolean) {
    super()
    this.enabled = enabled
    this.cls = enabled ? 'enable_grad' : 'no_grad'
  }
  repr() { return `<torch.autograd.grad_mode.${this.cls} object>` }
  enter(h: Host): Value {
    const E = stateOf(h).E
    this.prev = E.grad
    E.grad = this.enabled
    return py.NONE
  }
  exit(h: Host) {
    stateOf(h).E.grad = this.prev
  }
}

// ---------------------------------------------------------------- tensors

const WRAP = new WeakMap<TT, TensorObj>()
const wrap = (t: TT): Value => {
  let o = WRAP.get(t)
  if (!o) WRAP.set(t, (o = new TensorObj(t)))
  return py.obj(o)
}
export const tensorValue = wrap

export const asTT = (v: Value): TT | null => (v.k === 'obj' && v.o instanceof TensorObj ? v.o.t : null)

/** a Python value as a tensor operand; scalars are "weak" (they do not widen the tensor's dtype) */
const operand = (h: Host, v: Value): { t: TT; weak: boolean } | null => {
  const t = asTT(v)
  if (t) return { t, weak: false }
  if (v.k === 'int' || v.k === 'bool') return { t: TT.scalar(numOf(v), v.k === 'bool' ? 'bool' : 'int64'), weak: true }
  if (v.k === 'float') return { t: TT.scalar(v.v, 'float32'), weak: true }
  if (v.k === 'array') return { t: fromNd(v.a, false), weak: false }
  void h
  return null
}

const ND_TO_T: Record<DType, TDType> = { float64: 'float64', int64: 'int64', bool: 'bool' }

/** a NumPy array as a tensor; share = torch.from_numpy (same memory) */
const fromNd = (a: NDArray, share: boolean): TT => (share ? new TT(a, ND_TO_T[a.dtype]) : TT.of(a.values(), a.shape, ND_TO_T[a.dtype]))

/** torch.tensor(data): Python floats → float32, ints → int64, NumPy keeps its dtype */
const tensorFrom = (h: Host, v: Value, dtype: TDType | null): TT => {
  const t = asTT(v)
  if (t) return TT.of(t.values(), t.shape, dtype ?? t.dtype)
  if (v.k === 'array') return TT.of(v.a.values(), v.a.shape, dtype ?? ND_TO_T[v.a.dtype])
  let a: NDArray
  try {
    a = toArray(v)
  } catch (e) {
    throw h.err('TypeError', `torch.tensor(): could not build a tensor from this data (${(e as Error).message})`)
  }
  const inferred: TDType = a.dtype === 'float64' ? 'float32' : a.dtype === 'bool' ? 'bool' : 'int64'
  return TT.of(a.values(), a.shape, dtype ?? inferred)
}

const SNAP_MAX = 240
const snapOf = (t: TT): GridSnapshot => ({ shape: [...t.shape], values: t.values(), dtype: t.a.dtype })

/** record a small tensor op for the element-provenance view */
const trace = (h: Host, api: string, kind: 'elementwise' | 'matmul' | 'reduce' | 'move' | 'create' | 'random', ops: { label: string; t: TT }[], out: TT, axis: number | null = null) => {
  if (!h.tracing) return
  if (out.size > SNAP_MAX || ops.some((o) => o.t.size > SNAP_MAX)) return
  h.traceCall(api, kind, ops.map((o) => ({ label: o.label, snap: snapOf(o.t) })), snapOf(out), formatTensor(out), axis)
}

const label = (h: Host, v: Value, fallback: string) => h.nameOf(v, fallback)

/** F.conv2d on one image: each output cell comes from a K×K window of the input (every input channel) and the kernel */
const traceConv = (h: Host, x: TT, w: TT, out: TT, [sh, sw]: number[], [ph, pw]: number[], xl: string, wl: string) => {
  if (!h.tracing || x.shape[0] !== 1 || out.size > SNAP_MAX || x.size > SNAP_MAX || w.size > SNAP_MAX) return
  const [, cin, H, W] = x.shape
  const [cout, , kh, kw] = w.shape
  const [, , ho, wo] = out.shape
  const groups: [number, number][][] = []
  for (let co = 0; co < cout; co++) {
    for (let i = 0; i < ho; i++) {
      for (let j = 0; j < wo; j++) {
        const g: [number, number][] = []
        for (let ci = 0; ci < cin; ci++) {
          for (let a = 0; a < kh; a++) {
            for (let c = 0; c < kw; c++) {
              const r = i * sh - ph + a
              const q = j * sw - pw + c
              if (r >= 0 && r < H && q >= 0 && q < W) g.push([0, (ci * H + r) * W + q])
              g.push([1, ((co * cin + ci) * kh + a) * kw + c])
            }
          }
        }
        groups.push(g)
      }
    }
  }
  h.traceCall('F.conv2d', 'group', [{ label: xl, snap: snapOf(x) }, { label: wl, snap: snapOf(w) }], snapOf(out), formatTensor(out), null, groups)
}

const toDims = (h: Host, args: Value[]): number[] => {
  const src = args.length === 1 && (args[0].k === 'tuple' || args[0].k === 'list' || (args[0].k === 'obj' && args[0].o instanceof SizeObj)) ? h.iterate(args[0]) : args
  return src.map((v) => h.toInt(v, 'an integer size'))
}

const dtypeArg = (h: Host, v: Value | undefined): TDType | null => {
  if (!v || v.k === 'none') return null
  if (v.k === 'obj' && v.o instanceof DtypeObj) return v.o.d
  if (v.k === 'type' && v.name === 'float') return 'float32'
  if (v.k === 'type' && v.name === 'int') return 'int64'
  if (v.k === 'type' && v.name === 'bool') return 'bool'
  throw h.err('TypeError', `dtype must be a torch.dtype (e.g. torch.float32), not ${repr(v)}`)
}

const optDim = (h: Host, v: Value | undefined): number | null => (!v || v.k === 'none' ? null : h.toInt(v))
const dimsArg = (h: Host, v: Value | undefined): number[] | null => (!v || v.k === 'none' ? null : v.k === 'tuple' || v.k === 'list' ? v.items.map((x) => h.toInt(x)) : [h.toInt(v)])

/** index Values → IndexItems (ints, slices, None, ..., lists, bool / long tensors, arrays) */
const indexItems = (h: Host, idx: Value): IndexItem[] => {
  const items = idx.k === 'tuple' ? idx.items : [idx]
  return items.map((it): IndexItem => {
    switch (it.k) {
      case 'int': case 'bool': return it.k === 'int' ? { kind: 'int', value: it.v } : { kind: 'array', arr: NDArray.create([it.v ? 1 : 0], [], 'bool') }
      case 'none': return { kind: 'newaxis' }
      case 'ellipsis': return { kind: 'ellipsis' }
      case 'slice': {
        const part = (x: Value) => (x.k === 'none' ? null : h.toInt(x))
        return { kind: 'slice', start: part(it.start), stop: part(it.stop), step: part(it.step) }
      }
      case 'list': case 'tuple': return { kind: 'array', arr: toArray(it) }
      case 'array': return { kind: 'array', arr: it.a }
      case 'obj': {
        const t = asTT(it)
        if (t) {
          if (t.ndim === 0 && t.dtype !== 'bool') return { kind: 'int', value: t.values()[0] }
          if (isFloatT(t.dtype)) throw h.err('IndexError', 'tensors used as indices must be long, int, byte or bool tensors')
          return { kind: 'array', arr: NDArray.create(t.values(), t.shape, t.dtype === 'bool' ? 'bool' : 'int64') }
        }
        break
      }
    }
    throw h.err('IndexError', `only integers, slices (\`:\`), ellipsis (\`...\`), None and long or byte Variables are valid indices (got ${it.k === 'obj' ? it.o.cls : it.k})`)
  })
}

const BIN_API: Record<string, string> = { '+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '**': 'pow', '//': 'floor_divide', '%': 'remainder' }

export class TensorObj extends PyObj {
  readonly cls = 'Tensor'
  t: TT
  constructor(t: TT) {
    super()
    this.t = t
  }
  repr() {
    return this.t.isParam ? `Parameter containing:\n${formatTensor(this.t)}` : formatTensor(this.t)
  }
  format(spec: string) {
    if (this.t.size !== 1) return undefined
    const v = this.t.values()[0]
    return formatSpec(isFloatT(this.t.dtype) ? py.float(v) : py.int(v), spec)
  }
  isa(name: string) { return name === 'Tensor' || (name === 'Parameter' && this.t.isParam) }
  toArray(): NDArray {
    if (this.t.requiresGrad) throw new PyError('RuntimeError', "Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.")
    return this.t.a
  }
  len(h: Host) {
    if (this.t.ndim === 0) throw h.err('TypeError', 'len() of a 0-d tensor')
    return this.t.shape[0]
  }
  truthy(h: Host) {
    if (this.t.size !== 1) throw h.err('RuntimeError', `Boolean value of Tensor with ${this.t.size ? 'more than one value' : 'no values'} is ambiguous`)
    return this.t.values()[0] !== 0
  }
  iter(h: Host): Value[] {
    if (this.t.ndim === 0) throw h.err('TypeError', 'iteration over a 0-d tensor')
    const E = stateOf(h).E
    return Array.from({ length: this.t.shape[0] }, (_, i) => wrap(C.indexT(E, this.t, [{ kind: 'int', value: i }])))
  }
  contains(v: Value, h: Host) {
    const o = operand(h, v)
    if (!o) return false
    return C.compare(this.t, o.t, '==').values().some((x) => x !== 0)
  }
  getItem(idx: Value, h: Host): Value {
    const E = stateOf(h).E
    const out = C.indexT(E, this.t, indexItems(h, idx))
    trace(h, 'Tensor.__getitem__', 'move', [{ label: label(h, py.obj(this), 'x'), t: this.t }], out)
    return wrap(out)
  }
  setItem(idx: Value, v: Value, h: Host) {
    this.checkInplace(h)
    const o = operand(h, v)
    if (!o) throw h.err('TypeError', `can't assign a ${v.k === 'obj' ? v.o.cls : v.k} to a torch tensor`)
    setIndex(this.t.a, indexItems(h, idx), NDArray.create(o.t.values().map((x) => C.castT(x, this.t.dtype)), o.t.shape, this.t.a.dtype))
  }
  checkInplace(h: Host) {
    if (this.t.isLeaf && this.t.requiresGrad && stateOf(h).E.grad) throw h.err('RuntimeError', 'a leaf Variable that requires grad is being used in an in-place operation.')
  }
  binop(op: string, other: Value, reflected: boolean, h: Host): Value | undefined {
    const o = operand(h, other)
    if (!o) return undefined
    const E = stateOf(h).E
    const [a, b, aw, bw] = reflected ? [o.t, this.t, o.weak, false] : [this.t, o.t, false, o.weak]
    let out: TT
    if (op === '@') {
      out = C.matmul(E, a, b)
      trace(h, 'op:@', 'matmul', [{ label: reflected ? 'a' : label(h, py.obj(this), 'a'), t: a }, { label: reflected ? label(h, py.obj(this), 'b') : label(h, other, 'b'), t: b }], out)
      return wrap(out)
    }
    if (['<', '<=', '>', '>=', '==', '!='].includes(op)) {
      out = C.compare(a, b, op as C.CmpOpT)
    } else if (op in BIN_API) {
      out = C.binary(E, op as C.BinOpT, a, b, aw, bw)
    } else if (op === '&' || op === '|' || op === '^') {
      if (a.dtype !== 'bool' && isFloatT(a.dtype)) throw h.err('RuntimeError', `"bitwise_${op === '&' ? 'and' : op === '|' ? 'or' : 'xor'}_cpu" not implemented for 'Float'`)
      const av = a.values()
      const bv = b.values()
      const shape = broadcastShapes([a.shape, b.shape])
      const ia = C.bcastIndex(a.shape, shape)
      const ib = C.bcastIndex(b.shape, shape)
      const f = (x: number, y: number) => (op === '&' ? x & y : op === '|' ? x | y : x ^ y)
      out = TT.of(Array.from({ length: prod(shape) }, (_, i) => f(av[ia[i]], bv[ib[i]])), shape, promote2(a.dtype, b.dtype))
    } else return undefined
    trace(h, `op:${op}`, 'elementwise', [{ label: reflected ? label(h, other, 'a') : label(h, py.obj(this), 'a'), t: a }, { label: reflected ? label(h, py.obj(this), 'b') : label(h, other, 'b'), t: b }], out)
    return wrap(out)
  }
  inplace(op: string, rhs: Value, h: Host): boolean {
    if (!(op in BIN_API)) return false
    const o = operand(h, rhs)
    if (!o) return false
    this.checkInplace(h)
    const E = stateOf(h).E
    const was = E.grad
    E.grad = false
    try {
      const res = C.binary(E, op as C.BinOpT, this.t, o.t, false, o.weak)
      if (res.shape.join() !== this.t.shape.join()) throw h.err('RuntimeError', `output with shape [${this.t.shape.join(', ')}] doesn't match the broadcast shape [${res.shape.join(', ')}]`)
      if (!isFloatT(this.t.dtype) && isFloatT(res.dtype)) throw h.err('RuntimeError', `result type Float can't be cast to the desired output type ${this.t.dtype === 'int64' ? 'Long' : this.t.dtype}`)
      C.writeInto(this.t, res.values())
    } finally {
      E.grad = was
    }
    return true
  }
  unary(op: '-' | '+' | '~', h: Host): Value {
    const E = stateOf(h).E
    if (op === '+') return py.obj(this)
    if (op === '~') {
      if (isFloatT(this.t.dtype)) throw h.err('TypeError', "~ (operator.invert) is only implemented on integer and Boolean-type tensors")
      return wrap(TT.of(this.t.values().map((v) => (this.t.dtype === 'bool' ? (v ? 0 : 1) : ~v)), this.t.shape, this.t.dtype))
    }
    return wrap(C.unary(E, 'neg', this.t))
  }
  setAttr(name: string, v: Value, h: Host): boolean {
    if (name === 'requires_grad') {
      const b = h.truthy(v)
      if (!this.t.isLeaf && !b) throw h.err('RuntimeError', "you can only change requires_grad flags of leaf variables. If you want to use a computed variable in a subgraph that doesn't require differentiation use var_no_grad = var.detach().")
      if (b && !isFloatT(this.t.dtype)) throw h.err('RuntimeError', 'only Tensors of floating point dtype can require gradients')
      this.t.requiresGrad = b
      return true
    }
    if (name === 'grad') {
      this.t.grad = v.k === 'none' ? null : asTT(v)
      return true
    }
    if (name === 'data') {
      const o = asTT(v)
      if (o) this.t.a = o.a
      return true
    }
    return false
  }
  getAttr(name: string, h: Host): Value | undefined {
    const t = this.t
    const S = stateOf(h)
    const E = S.E
    const self = py.obj(this)
    const fn = (call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })
    const me = () => label(h, self, 'x')
    switch (name) {
      case 'shape': return py.obj(new SizeObj(t.shape))
      case 'dtype': return py.obj(new DtypeObj(t.dtype))
      case 'device': return py.obj(new DeviceObj('cpu'))
      case 'is_cuda': return py.bool(false)
      case 'requires_grad': return py.bool(t.requiresGrad)
      case 'grad': return t.grad ? wrap(t.grad) : (t.isLeaf || t.retain ? py.NONE : (noteOnce(h, 'nonleaf_grad', 'warn'), py.NONE))
      case 'is_leaf': return py.bool(t.isLeaf)
      case 'grad_fn': return t.fn ? py.obj(new GradFnObj(t.fn.name)) : py.NONE
      case 'data': return wrap(C.detach(t))
      case 'ndim': return py.int(t.ndim)
      case 'T': {
        if (t.ndim > 2) noteOnce(h, 'T_nd', 'warn')
        const out = C.permuteT(E, t, t.shape.map((_, i) => t.ndim - 1 - i), 'TBackward0')
        trace(h, 'Tensor.T', 'move', [{ label: me(), t }], out)
        return wrap(out)
      }
      case 'size': return fn((args) => (args.length ? py.int(t.shape[C.normDim(h.toInt(args[0]), t.ndim)]) : py.obj(new SizeObj(t.shape))))
      case 'dim': return fn(() => py.int(t.ndim))
      case 'numel': case 'nelement': return fn(() => py.int(t.size))
      case 'item': return fn(() => {
        if (t.size !== 1) throw h.err('RuntimeError', `a Tensor with ${t.size} elements cannot be converted to Scalar`)
        const v = t.values()[0]
        return t.dtype === 'bool' ? py.bool(v !== 0) : isFloatT(t.dtype) ? py.float(v) : py.int(v)
      })
      case 'tolist': return fn(() => {
        const vals = t.values()
        const one = (v: number) => (t.dtype === 'bool' ? py.bool(v !== 0) : isFloatT(t.dtype) ? py.float(v) : py.int(v))
        if (!t.ndim) return one(vals[0])
        const st = cStrides(t.shape)
        const rec = (d: number, base: number): Value => py.list(Array.from({ length: t.shape[d] }, (_, i) => (d === t.ndim - 1 ? one(vals[base + i]) : rec(d + 1, base + i * st[d]))))
        return rec(0, 0)
      })
      case 'numpy': return fn(() => {
        if (t.requiresGrad) throw h.err('RuntimeError', "Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.")
        // shares memory with the tensor, as in PyTorch
        return py.arr(t.a)
      })
      case 'detach': return fn(() => wrap(C.detach(t)))
      case 'clone': return fn(() => wrap(C.cloneT(E, t)))
      case 'contiguous': return fn(() => (t.a.isCContiguous() ? self : wrap(C.reshapeT(E, t, t.shape, false))))
      case 'is_contiguous': return fn(() => py.bool(t.a.isCContiguous()))
      case 'backward': return fn((args, kw) => {
        const g = kw.gradient ?? args[0]
        const gt = g && g.k !== 'none' ? asTT(g) : null
        doBackward(h, t, self, gt ? gt.values() : null, !!(kw.retain_graph && h.truthy(kw.retain_graph)))
        return py.NONE
      })
      case 'retain_grad': return fn(() => { t.retain = true; return py.NONE })
      case 'requires_grad_': return fn((args) => { this.setAttr('requires_grad', args[0] ?? py.bool(true), h); return self })
      case 'zero_': return fn(() => { this.checkInplace(h); C.writeInto(t, t.values().map(() => 0)); return self })
      case 'fill_': return fn((args) => { this.checkInplace(h); const v = h.num(args[0]); C.writeInto(t, t.values().map(() => v)); return self })
      case 'copy_': return fn((args) => {
        this.checkInplace(h)
        const o = asTT(args[0])
        if (!o) throw h.err('TypeError', 'copy_(): argument must be a Tensor')
        const ia = C.bcastIndex(o.shape, t.shape)
        const ov = o.values()
        C.writeInto(t, Array.from({ length: t.size }, (_, i) => ov[ia[i]]))
        return self
      })
      case 'add_': case 'sub_': case 'mul_': case 'div_': return fn((args, kw) => {
        const op = { add_: '+', sub_: '-', mul_: '*', div_: '/' }[name]!
        const alpha = kw.alpha ?? args[1]
        const o = operand(h, args[0])
        if (!o) throw h.err('TypeError', `${name}(): argument 'other' must be a Tensor or a number`)
        const rhs = alpha ? wrap(C.binary({ grad: false } as Engine, '*', o.t, TT.scalar(h.num(alpha), 'float32'), o.weak, true)) : args[0]
        this.inplace(op, rhs, h)
        return self
      })
      case 'view': case 'reshape': return fn((args) => {
        const dims = toDims(h, args)
        const out = C.reshapeT(E, t, dims, name === 'view')
        trace(h, `Tensor.${name}`, 'move', [{ label: me(), t }], out)
        flowStep(h, `${name}(${dims.join(', ')})`, t, out)
        return wrap(out)
      })
      case 'flatten': return fn((args, kw) => {
        const out = C.flattenT(E, t, optDim(h, kw.start_dim ?? args[0]) ?? 0, optDim(h, kw.end_dim ?? args[1]) ?? -1)
        flowStep(h, 'flatten', t, out)
        return wrap(out)
      })
      case 'unsqueeze': return fn((args, kw) => wrap(C.unsqueezeT(E, t, h.toInt(kw.dim ?? args[0]))))
      case 'squeeze': return fn((args, kw) => wrap(C.squeezeT(E, t, optDim(h, kw.dim ?? args[0]))))
      case 'permute': return fn((args) => {
        const out = C.permuteT(E, t, toDims(h, args))
        trace(h, 'Tensor.permute', 'move', [{ label: me(), t }], out)
        return wrap(out)
      })
      case 'transpose': return fn((args) => {
        const out = C.transposeT(E, t, h.toInt(args[0]), h.toInt(args[1]))
        trace(h, 'Tensor.transpose', 'move', [{ label: me(), t }], out)
        return wrap(out)
      })
      case 't': return fn(() => {
        if (t.ndim > 2) throw h.err('RuntimeError', `t() expects a tensor with <= 2 dimensions, but self is ${t.ndim}D`)
        return wrap(t.ndim < 2 ? t : C.transposeT(E, t, 0, 1))
      })
      case 'sum': case 'mean': return fn((args, kw) => {
        const dims = dimsArg(h, kw.dim ?? kw.axis ?? args[0])
        const out = C.reduceSum(E, t, dims, !!(kw.keepdim && h.truthy(kw.keepdim)), name === 'mean')
        trace(h, `Tensor.${name}`, 'reduce', [{ label: me(), t }], out, dims && dims.length === 1 ? C.normDim(dims[0], t.ndim) : null)
        return wrap(out)
      })
      case 'std': case 'var': return fn((args, kw) => {
        const corr = kw.correction ? h.num(kw.correction) : kw.unbiased && !h.truthy(kw.unbiased) ? 0 : 1
        return wrap(C.variance(E, t, dimsArg(h, kw.dim ?? args[0]), !!(kw.keepdim && h.truthy(kw.keepdim)), corr, name === 'std'))
      })
      case 'max': case 'min': return fn((args, kw) => maxMin(h, name, t, kw.dim ?? args[0], kw.keepdim, self))
      case 'argmax': case 'argmin': return fn((args, kw) => {
        const out = C.argmax(t, optDim(h, kw.dim ?? args[0]), !!(kw.keepdim && h.truthy(kw.keepdim)), name === 'argmin')
        trace(h, `Tensor.${name}`, 'reduce', [{ label: me(), t }], out, optDim(h, kw.dim ?? args[0]))
        return wrap(out)
      })
      case 'exp': case 'log': case 'sqrt': case 'abs': case 'relu': case 'sigmoid': case 'tanh': case 'square': case 'sin': case 'cos':
        return fn(() => {
          const out = C.unary(E, name as C.UnOpT, t)
          trace(h, `Tensor.${name}`, 'elementwise', [{ label: me(), t }], out)
          return wrap(out)
        })
      case 'softmax': case 'log_softmax': return fn((args, kw) => wrap(C.softmax(E, t, h.toInt(kw.dim ?? args[0] ?? py.int(-1)), name === 'log_softmax')))
      case 'pow': return fn((args) => this.binop('**', args[0], false, h)!)
      case 'clamp': case 'clip': return fn((args, kw) => wrap(C.clamp(E, t, optNumV(h, kw.min ?? args[0]), optNumV(h, kw.max ?? args[1]))))
      case 'round': return fn((_args, kw) => {
        const d = kw.decimals ? h.toInt(kw.decimals) : 0
        return wrap(TT.of(t.values().map((v) => roundHalfEvenT(v, d)), t.shape, t.dtype))
      })
      case 'matmul': case 'mm': case 'dot': return fn((args) => this.binop('@', args[0], false, h)!)
      case 'add': case 'sub': case 'mul': case 'div': return fn((args) => this.binop({ add: '+', sub: '-', mul: '*', div: '/' }[name]!, args[0], false, h)!)
      case 'eq': case 'ne': case 'gt': case 'lt': case 'ge': case 'le': return fn((args) => this.binop({ eq: '==', ne: '!=', gt: '>', lt: '<', ge: '>=', le: '<=' }[name]!, args[0], false, h)!)
      case 'all': case 'any': return fn(() => py.obj(new TensorObj(TT.scalar(name === 'all' ? (t.values().every((v) => v !== 0) ? 1 : 0) : t.values().some((v) => v !== 0) ? 1 : 0, 'bool'))))
      case 'float': case 'double': case 'long': case 'int': case 'bool': case 'half':
        return fn(() => wrap(C.toDtype(E, t, ({ float: 'float32', double: 'float64', long: 'int64', int: 'int32', bool: 'bool', half: 'float32' } as Record<string, TDType>)[name])))
      case 'type': return fn((args) => {
        const d = dtypeArg(h, args[0])
        return d ? wrap(C.toDtype(E, t, d)) : py.str(`torch.${{ float32: 'Float', float64: 'Double', int64: 'Long', int32: 'Int', uint8: 'Byte', bool: 'Bool' }[t.dtype]}Tensor`)
      })
      case 'to': return fn((args, kw) => {
        const target = kw.dtype ?? args.find((a) => a.k === 'obj' && a.o instanceof DtypeObj)
        const d = target ? dtypeArg(h, target) : null
        const dev = kw.device ?? args.find((a) => a.k === 'str' || (a.k === 'obj' && a.o instanceof DeviceObj))
        if (dev && h.str(dev).startsWith('cuda')) throw h.err('AssertionError', 'Torch not compiled with CUDA enabled')
        return d ? wrap(C.toDtype(E, t, d)) : self
      })
      case 'cpu': return fn(() => self)
      case 'cuda': return fn(() => { throw h.err('AssertionError', 'Torch not compiled with CUDA enabled') })
      case 'masked_fill': return fn((args) => {
        const m = asTT(args[0])
        if (!m) throw h.err('TypeError', 'masked_fill(): mask must be a bool tensor')
        const v = h.num(args[1])
        const ia = C.bcastIndex(m.shape, t.shape)
        const mv = m.values()
        return wrap(TT.of(t.values().map((x, i) => (mv[ia[i]] ? v : x)), t.shape, t.dtype))
      })
      case 'apply_': case 'register_hook': return undefined
    }
    return undefined
  }
}

const promote2 = (a: TDType, b: TDType) => C.promote(a, b)
const optNumV = (h: Host, v: Value | undefined): number | null => (!v || v.k === 'none' ? null : h.num(v))
const roundHalfEvenT = (v: number, d: number) => {
  const f = 10 ** d
  const x = v * f
  const r = Math.round(x)
  return (Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r) / f
}

/** tensor.max() / torch.max(t, dim) */
const maxMin = (h: Host, name: string, t: TT, dimV: Value | undefined, keepV: Value | undefined, self: Value): Value => {
  const E = stateOf(h).E
  const isMin = name.endsWith('min')
  if (dimV && dimV.k === 'obj' && asTT(dimV)) {
    // torch.max(a, b): elementwise
    return wrap(C.binary(E, isMin ? 'minimum' : 'maximum', t, asTT(dimV)!))
  }
  if (!dimV || dimV.k === 'none') {
    const out = C.maxAll(E, t, isMin)
    trace(h, `torch.${isMin ? 'min' : 'max'}`, 'reduce', [{ label: label(h, self, 'x'), t }], out)
    return wrap(out)
  }
  const dim = h.toInt(dimV)
  const r = C.maxAlong(E, t, dim, !!(keepV && h.truthy(keepV)), isMin)
  trace(h, `torch.${isMin ? 'min' : 'max'}`, 'reduce', [{ label: label(h, self, 'x'), t }], r.values, C.normDim(dim, t.ndim))
  return py.obj(new ReturnTypes(isMin ? 'min' : 'max', wrap(r.values), wrap(r.indices)))
}

// ---------------------------------------------------------------- autograd graph event

const shortVal = (t: TT): string => {
  if (t.size === 1) {
    const v = t.values()[0]
    return isFloatT(t.dtype) ? String(+v.toFixed(4)) : String(v)
  }
  if (t.size <= 4) return `[${t.values().map((v) => +v.toFixed(3)).join(', ')}]`
  return `shape (${t.shape.join(', ')})`
}

const OP_SHORT = (name: string) => name.replace(/Backward\d*$/, '').replace(/^Native/, '').toLowerCase()

function doBackward(h: Host, t: TT, self: Value, grad: number[] | null, retain: boolean) {
  const S = stateOf(h)
  // gradients already sitting in leaves will be added to: the reason for zero_grad()
  const before = new Map<TT, boolean>()
  const leaves: TT[] = []
  const walk = (x: TT, seen: Set<TT>) => {
    if (seen.has(x)) return
    seen.add(x)
    if (x.isLeaf && x.requiresGrad) leaves.push(x)
    for (const i of x.fn?.inputs ?? []) if (i && i.requiresGrad) walk(i, seen)
  }
  walk(t, new Set())
  for (const l of leaves) before.set(l, l.grad !== null && l.grad.values().some((v) => v !== 0))
  C.backward(S.E, t, grad, retain)
  if (leaves.some((l) => before.get(l))) noteOnce(h, 'grad_accumulate', 'warn')
  if (!h.tracing) return
  // the graph, for the step view
  const names = new Map<TT, string>()
  const model = S.lastModel
  if (model) for (const [n, p] of namedParams(h, model, '')) names.set(p, n)
  const nodes: GraphNode[] = []
  const ids = new Map<TT, number>()
  const visit = (x: TT): number => {
    const known = ids.get(x)
    if (known !== undefined) return known
    const id = ids.size
    ids.set(x, id)
    const inputs = (x.fn?.inputs ?? []).filter((i): i is TT => !!i && (i.requiresGrad || i.size <= 4))
    const node: GraphNode = {
      id, label: names.get(x) ?? h.nameOf(wrap(x), x.fn ? OP_SHORT(x.fn.name) : 'tensor'), op: x.fn?.name ?? null, shape: [...x.shape],
      value: shortVal(x), grad: x.grad ? shortVal(x.grad) : null, leaf: x.isLeaf, requiresGrad: x.requiresGrad, inputs: [],
    }
    nodes.push(node)
    if (ids.size < 48) node.inputs = inputs.map(visit)
    return id
  }
  const root = visit(t)
  void self
  h.emit({ type: 'graph', nodes, root, title: 'backward()' })
}

// ---------------------------------------------------------------- modules

/** what every module (a built-in layer or an nn.Module subclass) offers */
interface ModuleLike {
  kind: string
  training: boolean
  children(h: Host): [string, Value][]
  ownParams(): [string, TT][]
  extraRepr(): string
}

const MODULE_STATE = 'torch.module'

const moduleOf = (v: Value): ModuleLike | null => {
  if (v.k === 'obj' && v.o instanceof Layer) return v.o
  if (v.k === 'inst' && v.state[MODULE_STATE]) return instModule(v)
  return null
}

const instModule = (inst: InstValue): ModuleLike => {
  const st = inst.state[MODULE_STATE] as { training: boolean }
  return {
    kind: inst.cls.name,
    get training() { return st.training },
    set training(b: boolean) { st.training = b },
    children: () => [...inst.attrs].filter(([, v]) => moduleOf(v) !== null),
    ownParams: () => [...inst.attrs].flatMap(([n, v]): [string, TT][] => {
      const t = asTT(v)
      return t && t.isParam ? [[n, t]] : []
    }),
    extraRepr: () => '',
  }
}

export function namedParams(h: Host, v: Value, prefix: string): [string, TT][] {
  const m = moduleOf(v)
  if (!m) return []
  const own = m.ownParams().map(([n, t]): [string, TT] => [prefix + n, t])
  return [...own, ...m.children(h).flatMap(([n, c]) => namedParams(h, c, `${prefix}${n}.`))]
}

const allModules = (h: Host, v: Value, prefix: string, out: Map<object, string>) => {
  const m = moduleOf(v)
  if (!m) return
  out.set(v.k === 'obj' ? v.o : v, prefix.replace(/\.$/, ''))
  for (const [n, c] of m.children(h)) allModules(h, c, `${prefix}${n}.`, out)
}

const moduleRepr = (h: Host, v: Value, indent = ''): string => {
  const m = moduleOf(v)!
  const kids = m.children(h)
  const head = `${m.kind}(${m.extraRepr()}`
  if (!kids.length) return `${head})`
  const inner = kids.map(([n, c]) => `${indent}  (${n}): ${moduleRepr(h, c, `${indent}  `)}`).join('\n')
  return `${head}\n${inner}\n${indent})`
}

const setTraining = (h: Host, v: Value, b: boolean) => {
  const m = moduleOf(v)
  if (!m) return
  m.training = b
  for (const [, c] of m.children(h)) setTraining(h, c, b)
}

/** methods every module answers to */
const moduleMethod = (h: Host, self: Value, name: string): Value | undefined => {
  const m = moduleOf(self)!
  const fn = (call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })
  switch (name) {
    case 'parameters': return fn(() => py.list(namedParams(h, self, '').map(([, t]) => wrap(t))))
    case 'named_parameters': return fn(() => py.list(namedParams(h, self, '').map(([n, t]) => py.tuple([py.str(n), wrap(t)]))))
    case 'children': return fn(() => py.list(m.children(h).map(([, c]) => c)))
    case 'named_children': return fn(() => py.list(m.children(h).map(([n, c]) => py.tuple([py.str(n), c]))))
    case 'modules': return fn(() => {
      const out = new Map<object, string>()
      allModules(h, self, '', out)
      return py.list([...out.keys()].map((o) => (o instanceof PyObj ? py.obj(o) : (o as Value))))
    })
    case 'train': return fn((args) => { setTraining(h, self, args[0] ? h.truthy(args[0]) : true); return self })
    case 'eval': return fn(() => { setTraining(h, self, false); return self })
    case 'training': return py.bool(m.training)
    case 'zero_grad': return fn(() => {
      for (const [, t] of namedParams(h, self, '')) t.grad = null
      return py.NONE
    })
    case 'to': case 'cpu': case 'float': case 'double': return fn((args, kw) => {
      const dev = kw.device ?? args[0]
      if (dev && h.str(dev).startsWith('cuda')) throw h.err('AssertionError', 'Torch not compiled with CUDA enabled')
      return self
    })
    case 'cuda': return fn(() => { throw h.err('AssertionError', 'Torch not compiled with CUDA enabled') })
    case 'state_dict': return fn(() => ({ k: 'dict', d: PyDict.from(namedParams(h, self, '').map(([n, t]) => [py.str(n), wrap(C.detach(t))])) }))
    case 'load_state_dict': return fn((args) => {
      const d = args[0]
      if (d?.k !== 'dict') throw h.err('TypeError', 'load_state_dict() expects a dict from state_dict()')
      const missing: string[] = []
      for (const [n, t] of namedParams(h, self, '')) {
        const src = asTT(d.d.get(py.str(n)) ?? py.NONE)
        if (!src) missing.push(n)
        else if (src.shape.join() !== t.shape.join()) throw h.err('RuntimeError', `Error(s) in loading state_dict: size mismatch for ${n}: copying a param with shape torch.Size([${src.shape.join(', ')}]) from checkpoint, the shape in current model is torch.Size([${t.shape.join(', ')}]).`)
        else C.writeInto(t, src.values())
      }
      if (missing.length) throw h.err('RuntimeError', `Error(s) in loading state_dict for ${m.kind}: Missing key(s) in state_dict: ${missing.map((x) => `"${x}"`).join(', ')}.`)
      return py.str('<All keys matched successfully>')
    })
    case 'apply': return fn((args) => {
      const out = new Map<object, string>()
      allModules(h, self, '', out)
      for (const o of out.keys()) h.call(args[0], [o instanceof PyObj ? py.obj(o) : (o as Value)])
      return self
    })
  }
  return undefined
}

/** run a module call, tracing shapes when it is the outermost call or a layer inside one */
const callModule = (h: Host, self: Value, run: () => Value, leaf: Layer | null, input: TT | null): Value => {
  const S = stateOf(h)
  let flow = S.flow
  const outer = !flow
  if (!flow) {
    const names = new Map<object, string>()
    allModules(h, self, '', names)
    flow = S.flow = { line: h.line, code: h.code, rows: [], depth: 0, names, model: moduleOf(self)!.kind, root: self.k === 'obj' ? self.o : self }
    // the model whose parameters name the autograd graph (a loss module has none)
    if (namedParams(h, self, '').length) S.lastModel = self
  }
  flow.depth++
  let out: Value
  try {
    out = run()
  } finally {
    flow.depth--
    if (outer) S.flow = null
  }
  const ot = asTT(out)
  if (leaf && input && ot && !(leaf instanceof Sequential) && !(leaf instanceof LossLayer)) {
    flow.rows.push({
      name: flow.names.get(leaf) || leaf.kind, layer: `${leaf.kind}(${leaf.extraRepr()})`, input: [...input.shape], output: [...ot.shape],
      params: leaf.ownParams().reduce((s, [, t]) => s + t.size, 0), note: leaf.flowNote(input, ot),
    })
  }
  if (outer && flow.rows.length && h.tracing) {
    const params = namedParams(h, self, '')
    const total = params.reduce((s, [, t]) => s + t.size, 0)
    h.emit({ type: 'flow', framework: 'torch', model: flow.model, rows: flow.rows, total, trainable: params.filter(([, t]) => t.requiresGrad).reduce((s, [, t]) => s + t.size, 0), title: `${flow.model} forward`, line: flow.line, code: flow.code })
  }
  return out
}

/** functional ops inside a forward pass (F.relu, x.view …) also become rows of the shape trace */
const flowStep = (h: Host, what: string, input: TT, out: TT) => {
  const flow = stateOf(h).flow
  if (!flow) return
  const shapeNote = what.startsWith('view') || what.startsWith('reshape') || what === 'flatten' ? `${input.shape.slice(1).join(' × ')} = ${prod(input.shape.slice(1))}` : null
  flow.rows.push({ name: h.code.length < 40 ? h.code : what, layer: what, input: [...input.shape], output: [...out.shape], params: 0, note: input.ndim > 2 && out.ndim === 2 ? shapeNote : null })
}

const pair = (h: Host, v: Value | undefined, d: number): [number, number] => {
  if (!v || v.k === 'none') return [d, d]
  if (v.k === 'tuple' || v.k === 'list') return [h.toInt(v.items[0]), h.toInt(v.items[1])]
  const n = h.toInt(v)
  return [n, n]
}

const tupleStr = (p: [number, number]) => `(${p[0]}, ${p[1]})`

/** default init: U(−1/√fan_in, 1/√fan_in), as PyTorch's kaiming_uniform_(a=√5) gives */
const initParam = (h: Host, shape: number[], fanIn: number): TT => {
  const bound = 1 / Math.sqrt(fanIn)
  const n = prod(shape)
  const vals = new Array<number>(n)
  for (let i = 0; i < n; i++) vals[i] = Math.fround((h.rng.next() * 2 - 1) * bound)
  const t = TT.raw(vals, shape, 'float32')
  t.requiresGrad = true
  t.isParam = true
  return t
}

abstract class Layer extends PyObj implements ModuleLike {
  abstract readonly kind: string
  get cls() { return this.kind }
  training = true
  extraRepr() { return '' }
  ownParams(): [string, TT][] { return [] }
  children(): [string, Value][] { return [] }
  flowNote(_i: TT, _o: TT): string | null { return null }
  abstract forward(args: Value[], kw: Kw, h: Host): Value
  repr() { return `${this.kind}(${this.extraRepr()})` }
  isa(name: string) { return name === 'Module' || name === this.kind }
  call(args: Value[], kw: Kw, h: Host): Value {
    return callModule(h, py.obj(this), () => this.forward(args, kw, h), this, asTT(args[0] ?? py.NONE))
  }
  getAttr(name: string, h: Host): Value | undefined {
    const p = this.ownParams().find(([n]) => n === name)
    if (p) return wrap(p[1])
    if (name === 'forward') return { k: 'fn', name, call: (args, kw) => this.forward(args, kw, h) }
    return this.attr(name) ?? moduleMethod(h, py.obj(this), name)
  }
  attr(_name: string): Value | undefined { return undefined }
}

const tensorArg = (h: Host, v: Value | undefined, what: string): TT => {
  const t = v ? asTT(v) : null
  if (!t) throw h.err('TypeError', `${what}: expected a Tensor as input, got ${v ? (v.k === 'obj' ? v.o.cls : v.k === 'array' ? 'numpy.ndarray (wrap it with torch.tensor / torch.from_numpy first)' : v.k) : 'nothing'}`)
  return t
}

const floatInput = (h: Host, t: TT, what: string) => {
  if (!isFloatT(t.dtype)) throw h.err('RuntimeError', `${what}: expected input of a floating dtype, but got ${t.dtype === 'int64' ? 'Long' : t.dtype} (call .float() first)`)
}

class Linear extends Layer {
  readonly kind = 'Linear'
  inF: number
  outF: number
  w: TT
  b: TT | null
  constructor(h: Host, inF: number, outF: number, bias: boolean) {
    super()
    this.inF = inF
    this.outF = outF
    this.w = initParam(h, [outF, inF], inF)
    this.b = bias ? initParam(h, [outF], inF) : null
  }
  extraRepr() { return `in_features=${this.inF}, out_features=${this.outF}, bias=${this.b ? 'True' : 'False'}` }
  ownParams(): [string, TT][] { return this.b ? [['weight', this.w], ['bias', this.b]] : [['weight', this.w]] }
  attr(name: string) { return name === 'in_features' ? py.int(this.inF) : name === 'out_features' ? py.int(this.outF) : name === 'bias' ? py.NONE : undefined }
  flowNote(i: TT) { return `${i.shape[i.ndim - 1]} × ${this.outF} + ${this.b ? this.outF : 0} = ${(this.inF + (this.b ? 1 : 0)) * this.outF} params` }
  forward(args: Value[], _kw: Kw, h: Host) {
    const x = tensorArg(h, args[0], 'Linear')
    floatInput(h, x, 'Linear')
    return wrap(C.linear(stateOf(h).E, x, this.w, this.b))
  }
}

class Conv2d extends Layer {
  readonly kind = 'Conv2d'
  inC: number
  outC: number
  k: [number, number]
  stride: [number, number]
  padding: [number, number]
  w: TT
  b: TT | null
  constructor(h: Host, inC: number, outC: number, k: [number, number], stride: [number, number], padding: [number, number], bias: boolean) {
    super()
    this.inC = inC
    this.outC = outC
    this.k = k
    this.stride = stride
    this.padding = padding
    const fan = inC * k[0] * k[1]
    this.w = initParam(h, [outC, inC, k[0], k[1]], fan)
    this.b = bias ? initParam(h, [outC], fan) : null
  }
  extraRepr() {
    return `${this.inC}, ${this.outC}, kernel_size=${tupleStr(this.k)}, stride=${tupleStr(this.stride)}${this.padding[0] || this.padding[1] ? `, padding=${tupleStr(this.padding)}` : ''}${this.b ? '' : ', bias=False'}`
  }
  ownParams(): [string, TT][] { return this.b ? [['weight', this.w], ['bias', this.b]] : [['weight', this.w]] }
  attr(name: string) {
    if (name === 'in_channels') return py.int(this.inC)
    if (name === 'out_channels') return py.int(this.outC)
    if (name === 'kernel_size') return py.tuple(this.k.map(py.int))
    if (name === 'stride') return py.tuple(this.stride.map(py.int))
    if (name === 'padding') return py.tuple(this.padding.map(py.int))
    return undefined
  }
  flowNote(i: TT, o: TT) {
    const H = i.shape[i.ndim - 2]
    return `⌊(${H} + 2·${this.padding[0]} − ${this.k[0]}) / ${this.stride[0]}⌋ + 1 = ${o.shape[o.ndim - 2]} · params (${this.k[0]}·${this.k[1]}·${this.inC} + 1)·${this.outC}`
  }
  forward(args: Value[], _kw: Kw, h: Host) {
    const x = tensorArg(h, args[0], 'Conv2d')
    floatInput(h, x, 'Conv2d')
    return wrap(C.conv2d(stateOf(h).E, x, this.w, this.b, { stride: this.stride, padding: this.padding }))
  }
}

class Pool2d extends Layer {
  readonly kind: string
  k: [number, number]
  s: [number, number]
  p: [number, number]
  mode: 'max' | 'avg'
  constructor(mode: 'max' | 'avg', k: [number, number], s: [number, number], p: [number, number]) {
    super()
    this.mode = mode
    this.kind = mode === 'max' ? 'MaxPool2d' : 'AvgPool2d'
    this.k = k
    this.s = s
    this.p = p
  }
  extraRepr() {
    const one = (x: [number, number]) => (x[0] === x[1] ? String(x[0]) : tupleStr(x))
    return this.mode === 'max' ? `kernel_size=${one(this.k)}, stride=${one(this.s)}, padding=${one(this.p)}, dilation=1, ceil_mode=False` : `kernel_size=${one(this.k)}, stride=${one(this.s)}, padding=${one(this.p)}`
  }
  flowNote(i: TT, o: TT) { return `⌊(${i.shape[i.ndim - 2]} + 2·${this.p[0]} − ${this.k[0]}) / ${this.s[0]}⌋ + 1 = ${o.shape[o.ndim - 2]}` }
  forward(args: Value[], _kw: Kw, h: Host) {
    return wrap(C.pool2d(stateOf(h).E, tensorArg(h, args[0], this.kind), this.k, this.s, this.p, this.mode))
  }
}

class Activation extends Layer {
  readonly kind: string
  fn: (x: TT, h: Host) => TT
  extra: string
  constructor(kind: string, fn: (x: TT, h: Host) => TT, extra = '') {
    super()
    this.kind = kind
    this.fn = fn
    this.extra = extra
  }
  extraRepr() { return this.extra }
  forward(args: Value[], _kw: Kw, h: Host) { return wrap(this.fn(tensorArg(h, args[0], this.kind), h)) }
}

class DropoutLayer extends Layer {
  readonly kind = 'Dropout'
  p: number
  constructor(p: number) {
    super()
    this.p = p
  }
  extraRepr() { return `p=${this.p}, inplace=False` }
  forward(args: Value[], _kw: Kw, h: Host) {
    const S = stateOf(h)
    if (this.training && !S.E.grad) noteOnce(h, 'dropout_eval', 'warn')
    return wrap(C.dropout(S.E, tensorArg(h, args[0], 'Dropout'), this.p, this.training))
  }
}

class Sequential extends Layer {
  readonly kind: string = 'Sequential'
  items: Value[]
  constructor(items: Value[]) {
    super()
    this.items = items
  }
  children(): [string, Value][] { return this.items.map((v, i) => [String(i), v]) }
  len() { return this.items.length }
  iter() { return this.items }
  getItem(idx: Value, h: Host) {
    const i = h.toInt(idx)
    const n = this.items.length
    if (i < -n || i >= n) throw h.err('IndexError', `index ${i} is out of range`)
    return this.items[i < 0 ? i + n : i]
  }
  forward(args: Value[], _kw: Kw, h: Host) {
    let x = args[0]
    for (const m of this.items) x = h.call(m, [x])
    return x
  }
  attr(name: string) {
    if (name === 'append') return { k: 'fn' as const, name, call: (args: Value[]) => { this.items.push(args[0]); return py.obj(this) } }
    return undefined
  }
}

class ModuleList extends Sequential {
  readonly kind = 'ModuleList'
  forward(_a: Value[], _k: Kw, h: Host): Value { throw h.err('NotImplementedError', 'ModuleList has no forward(): loop over it in your own forward') }
}

class LossLayer extends Layer {
  readonly kind: string
  fn: (a: TT, b: TT, h: Host) => TT
  constructor(kind: string, fn: (a: TT, b: TT, h: Host) => TT) {
    super()
    this.kind = kind
    this.fn = fn
  }
  forward(args: Value[], _kw: Kw, h: Host) {
    return wrap(this.fn(tensorArg(h, args[0], this.kind), tensorArg(h, args[1], `${this.kind} target`), h))
  }
}

/** the losses, shared by the nn classes and torch.nn.functional */
const ceLoss = (h: Host, a: TT, b: TT, red: C.Reduction = 'mean') => {
  // softmax already applied? CrossEntropyLoss does log_softmax itself
  if (a.ndim === 2 && a.size <= 4096) {
    const v = a.values()
    const C2 = a.shape[1]
    let probs = true
    for (let n = 0; n < a.shape[0] && probs; n++) {
      const row = v.slice(n * C2, (n + 1) * C2)
      probs = row.every((x) => x >= 0 && x <= 1) && Math.abs(row.reduce((s, x) => s + x, 0) - 1) < 1e-4
    }
    if (probs && C2 > 1) noteOnce(h, 'softmax_ce', 'warn')
  }
  return C.crossEntropy(stateOf(h).E, a, b, red)
}
const mse = (h: Host, a: TT, b: TT, red: C.Reduction = 'mean') => C.mseLoss(stateOf(h).E, a, b, red, (msg) => {
  h.print(`${msg}\n`)
  noteOnce(h, 'mse_shape', 'warn', { a: `(${a.shape.join(', ')})`, b: `(${b.shape.join(', ')})` })
})

// ---------------------------------------------------------------- optimisers

class Optimizer extends PyObj {
  readonly cls: string
  params: TT[]
  lr: number
  momentum: number
  betas: [number, number]
  eps: number
  weightDecay: number
  state = new Map<TT, { m: number[]; v: number[]; buf: number[] | null }>()
  steps = 0
  constructor(kind: string, params: TT[], lr: number, momentum: number, betas: [number, number], eps: number, wd: number) {
    super()
    this.cls = kind
    this.params = params
    this.lr = lr
    this.momentum = momentum
    this.betas = betas
    this.eps = eps
    this.weightDecay = wd
  }
  repr() {
    const groups = this.cls === 'SGD' ? `    dampening: 0\n    lr: ${this.lr}\n    momentum: ${this.momentum}\n    nesterov: False\n    weight_decay: ${this.weightDecay}` : `    betas: (${this.betas.join(', ')})\n    eps: ${this.eps}\n    lr: ${this.lr}\n    weight_decay: ${this.weightDecay}`
    return `${this.cls} (\nParameter Group 0\n${groups}\n)`
  }
  getAttr(name: string, h: Host): Value | undefined {
    const fn = (call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })
    switch (name) {
      case 'zero_grad': return fn(() => {
        for (const p of this.params) p.grad = null
        return py.NONE
      })
      case 'step': return fn(() => {
        this.steps++
        for (const p of this.params) {
          if (!p.grad) continue
          const g0 = p.grad.values()
          const w = p.values()
          const g = this.weightDecay ? g0.map((x, i) => x + this.weightDecay * w[i]) : g0
          let st = this.state.get(p)
          if (!st) this.state.set(p, (st = { m: g.map(() => 0), v: g.map(() => 0), buf: null }))
          let next: number[]
          if (this.cls === 'SGD') {
            let d = g
            if (this.momentum) {
              st.buf = st.buf ? st.buf.map((b, i) => this.momentum * b + g[i]) : [...g]
              d = st.buf
            }
            next = w.map((x, i) => x - this.lr * d[i])
          } else {
            const [b1, b2] = this.betas
            st.m = st.m.map((m, i) => b1 * m + (1 - b1) * g[i])
            st.v = st.v.map((v, i) => b2 * v + (1 - b2) * g[i] * g[i])
            const c1 = 1 - b1 ** this.steps
            const c2 = 1 - b2 ** this.steps
            next = w.map((x, i) => x - (this.lr * st!.m[i]) / c1 / (Math.sqrt(st!.v[i] / c2) + this.eps))
          }
          C.writeInto(p, next)
        }
        void h
        return py.NONE
      })
      case 'param_groups': return py.list([{ k: 'dict', d: PyDict.from([[py.str('lr'), py.float(this.lr)], [py.str('params'), py.list(this.params.map(wrap))]]) }])
    }
    return undefined
  }
}

// ---------------------------------------------------------------- data

class TensorDataset extends PyObj {
  readonly cls = 'TensorDataset'
  tensors: TT[]
  constructor(tensors: TT[]) {
    super()
    this.tensors = tensors
  }
  repr() { return '<torch.utils.data.dataset.TensorDataset object>' }
  len() { return this.tensors[0]?.shape[0] ?? 0 }
  getItem(idx: Value, h: Host): Value {
    const E = stateOf(h).E
    return py.tuple(this.tensors.map((t) => wrap(C.indexT(E, t, indexItems(h, idx)))))
  }
  isa(name: string) { return name === 'Dataset' }
}

class DataLoader extends PyObj {
  readonly cls = 'DataLoader'
  ds: Value
  bs: number
  shuffle: boolean
  constructor(ds: Value, bs: number, shuffle: boolean) {
    super()
    this.ds = ds
    this.bs = bs
    this.shuffle = shuffle
  }
  repr() { return '<torch.utils.data.dataloader.DataLoader object>' }
  size(h: Host) {
    const n = this.ds.k === 'obj' ? this.ds.o.len?.(h) ?? 0 : h.num(h.call(h.getAttr(this.ds, '__len__'), []))
    return n
  }
  len(h: Host) { return Math.ceil(this.size(h) / this.bs) }
  iter(h: Host): Value[] {
    const n = this.size(h)
    const order = Array.from({ length: n }, (_, i) => i)
    if (this.shuffle) {
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(h.rng.next() * (i + 1))
        ;[order[i], order[j]] = [order[j], order[i]]
      }
    }
    const E = stateOf(h).E
    const item = (i: number): Value[] => {
      const v = this.ds.k === 'obj' && this.ds.o.getItem ? this.ds.o.getItem(py.int(i), h) : h.call(h.getAttr(this.ds, '__getitem__'), [py.int(i)])
      return v.k === 'tuple' || v.k === 'list' ? v.items : [v]
    }
    const batches: Value[] = []
    for (let s = 0; s < n; s += this.bs) {
      const rows = order.slice(s, s + this.bs).map(item)
      const cols = rows[0].map((_, c) => {
        const parts = rows.map((r) => r[c])
        const ts = parts.map((p) => asTT(p) ?? tensorFrom(h, p, null))
        return wrap(C.catT(E, ts, 0, true))
      })
      batches.push(cols.length === 1 ? cols[0] : py.list(cols))
    }
    return batches
  }
}

// ---------------------------------------------------------------- the torch module

const fnv = (name: string, call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })

const build = (h: Host): Record<string, Value> => {
  const S = stateOf(h)
  const E = S.E
  const T = (v: Value | undefined, what: string) => tensorArg(h, v, what)
  /** a new tensor: requires_grad=…, and a 'create' step for the step list */
  const finish = (t: TT, kw: Kw, name: string) => {
    if (kw.requires_grad && h.truthy(kw.requires_grad)) {
      if (!isFloatT(t.dtype)) throw h.err('RuntimeError', 'Only Tensors of floating point and complex dtype can require gradients')
      t.requiresGrad = true
    }
    trace(h, `torch.${name}`, name.startsWith('rand') ? 'random' : 'create', [], t)
    return wrap(t)
  }
  const filled = (name: string, value: (i: number) => number, defaultDtype: TDType = 'float32') =>
    fnv(name, (args, kw) => {
      const dims = toDims(h, kw.size ? [kw.size] : args)
      const n = prod(dims)
      if (n > h.maxSize) throw h.err('RuntimeError', `[enforce fail] tensor of ${n} elements exceeds the sandbox limit of ${h.maxSize}`)
      return finish(TT.of(Array.from({ length: n }, (_, i) => value(i)), dims, dtypeArg(h, kw.dtype) ?? defaultDtype), kw, name)
    })
  const like = (name: string, value: () => number, float = false) =>
    fnv(name, (args, kw) => {
      const t = T(args[0], name)
      const dtype = dtypeArg(h, kw.dtype) ?? t.dtype
      if (float && !isFloatT(dtype)) throw h.err('RuntimeError', `"check_uniform_bounds" not implemented for '${dtype === 'int64' ? 'Long' : dtype}'`)
      return finish(TT.of(t.values().map(() => value()), t.shape, dtype), kw, name)
    })
  const uni = (op: C.UnOpT) => fnv(op, (args) => {
    const t = T(args[0], `torch.${op}`)
    const out = C.unary(E, op, t)
    trace(h, `torch.${op}`, 'elementwise', [{ label: label(h, args[0], 'x'), t }], out)
    flowStep(h, `torch.${op}`, t, out)
    return wrap(out)
  })
  const viaMethod = (name: string, method = name) => fnv(name, (args, kw) => {
    const t = asTT(args[0] ?? py.NONE)
    if (!t) throw h.err('TypeError', `${name}(): argument 'input' must be Tensor, not ${args[0] ? (args[0].k === 'obj' ? args[0].o.cls : args[0].k) : 'nothing'}`)
    return h.call(h.getAttr(args[0], method), args.slice(1), kw)
  })
  const twoArg = (name: string, op: string) => fnv(name, (args) => {
    const self = args[0]
    if (self?.k !== 'obj' || !(self.o instanceof TensorObj)) throw h.err('TypeError', `${name}(): argument 'input' must be Tensor`)
    const out = self.o.binop(op, args[1], false, h)
    if (!out) throw h.err('TypeError', `${name}(): argument 'other' must be a Tensor or a number`)
    return out
  })

  // ---- nn
  const layerType = (name: string, make: (args: Value[], kw: Kw) => Layer): Value => ({ k: 'type', name, call: (args, kw) => py.obj(make(args, kw)) })
  const num = (v: Value | undefined, d: number) => (v && v.k !== 'none' ? h.num(v) : d)
  const int = (v: Value | undefined, what: string) => {
    if (!v) throw h.err('TypeError', `missing required argument '${what}'`)
    return h.toInt(v)
  }
  const bool = (v: Value | undefined, d: boolean) => (v ? h.truthy(v) : d)
  const dimOf = (args: Value[], kw: Kw, i: number) => (kw.dim ?? args[i] ? h.toInt((kw.dim ?? args[i])!) : null)
  const reduction = (kw: Kw): C.Reduction => (kw.reduction?.k === 'str' ? (kw.reduction.v as C.Reduction) : 'mean')

  const moduleHost: HostClass = {
    init(inst) {
      inst.state[MODULE_STATE] = { training: true }
    },
    setAttr(inst, name, v, hh) {
      if (inst.state[MODULE_STATE]) return
      const t = asTT(v)
      if (moduleOf(v)) throw hh.err('AttributeError', 'cannot assign module before Module.__init__() call')
      if (t && t.isParam) throw hh.err('AttributeError', 'cannot assign parameters before Module.__init__() call')
      void name
    },
    getAttr(inst, name, hh) {
      if (!inst.state[MODULE_STATE]) {
        if (name === 'parameters' || name === 'forward') throw hh.err('AttributeError', `'${inst.cls.name}' object has no attribute '_parameters' (did you forget super().__init__()?)`)
        return undefined
      }
      return moduleMethod(hh, inst, name)
    },
    call(inst, args, kw, hh) {
      if (!inst.state[MODULE_STATE]) throw hh.err('AttributeError', `'${inst.cls.name}' object has no attribute '_call_impl' (did you forget super().__init__()?)`)
      const fwd = hh.getAttr(inst, 'forward')
      if (fwd.k !== 'bound') throw hh.err('NotImplementedError', `Module [${inst.cls.name}] is missing the required "forward" function`)
      return callModule(hh, inst, () => hh.call(fwd, args, kw), null, asTT(args[0] ?? py.NONE))
    },
    repr(inst) {
      return inst.state[MODULE_STATE] ? moduleRepr(h, inst) : undefined
    },
  }

  const Module: Value = { k: 'type', name: 'Module', host: moduleHost, call: () => { throw h.err('TypeError', 'nn.Module is a base class: define class Net(nn.Module) with __init__ and forward') } }
  const Dataset: Value = { k: 'type', name: 'Dataset', host: { init() {} }, call: () => { throw h.err('TypeError', 'Dataset is a base class: subclass it with __len__ and __getitem__') } }

  const act = (name: string, f: (x: TT) => TT, extra = (_a: Value[], _k: Kw) => '') => layerType(name, (a, k) => new Activation(name, f, extra(a, k)))
  const nnAttrs: Record<string, Value> = {
    Module,
    Linear: layerType('Linear', (a, k) => new Linear(h, int(k.in_features ?? a[0], 'in_features'), int(k.out_features ?? a[1], 'out_features'), bool(k.bias ?? a[2], true))),
    Conv2d: layerType('Conv2d', (a, k) => {
      const pad = k.padding ?? a[4]
      const ks = pair(h, k.kernel_size ?? a[2], 1)
      const p: [number, number] = pad?.k === 'str' ? (pad.v === 'same' ? [Math.floor(ks[0] / 2), Math.floor(ks[1] / 2)] : [0, 0]) : pair(h, pad, 0)
      return new Conv2d(h, int(k.in_channels ?? a[0], 'in_channels'), int(k.out_channels ?? a[1], 'out_channels'), ks, pair(h, k.stride ?? a[3], 1), p, bool(k.bias, true))
    }),
    MaxPool2d: layerType('MaxPool2d', (a, k) => {
      const ks = pair(h, k.kernel_size ?? a[0], 2)
      const st = k.stride ?? a[1]
      return new Pool2d('max', ks, st && st.k !== 'none' ? pair(h, st, 1) : ks, pair(h, k.padding ?? a[2], 0))
    }),
    AvgPool2d: layerType('AvgPool2d', (a, k) => {
      const ks = pair(h, k.kernel_size ?? a[0], 2)
      const st = k.stride ?? a[1]
      return new Pool2d('avg', ks, st && st.k !== 'none' ? pair(h, st, 1) : ks, pair(h, k.padding ?? a[2], 0))
    }),
    ReLU: act('ReLU', (x) => C.unary(E, 'relu', x)),
    Sigmoid: act('Sigmoid', (x) => C.unary(E, 'sigmoid', x)),
    Tanh: act('Tanh', (x) => C.unary(E, 'tanh', x)),
    LeakyReLU: layerType('LeakyReLU', (a, k) => {
      const s = num(k.negative_slope ?? a[0], 0.01)
      return new Activation('LeakyReLU', (x) => C.binary(E, 'maximum', x, C.binary(E, '*', x, TT.scalar(s, 'float32'), false, true)), `negative_slope=${s}`)
    }),
    Softmax: layerType('Softmax', (a, k) => {
      const d = dimOf(a, k, 0)
      return new Activation('Softmax', (x) => C.softmax(E, x, d ?? -1, false), `dim=${d ?? 'None'}`)
    }),
    LogSoftmax: layerType('LogSoftmax', (a, k) => {
      const d = dimOf(a, k, 0)
      return new Activation('LogSoftmax', (x) => C.softmax(E, x, d ?? -1, true), `dim=${d ?? 'None'}`)
    }),
    Flatten: layerType('Flatten', (a, k) => {
      const s = optDim(h, k.start_dim ?? a[0]) ?? 1
      const e = optDim(h, k.end_dim ?? a[1]) ?? -1
      return new Activation('Flatten', (x) => C.flattenT(E, x, s, e), `start_dim=${s}, end_dim=${e}`)
    }),
    Identity: act('Identity', (x) => x),
    Dropout: layerType('Dropout', (a, k) => new DropoutLayer(num(k.p ?? a[0], 0.5))),
    Sequential: layerType('Sequential', (a) => new Sequential(a)),
    ModuleList: layerType('ModuleList', (a) => new ModuleList(a[0] ? h.iterate(a[0]) : [])),
    CrossEntropyLoss: layerType('CrossEntropyLoss', (_a, k) => new LossLayer('CrossEntropyLoss', (x, y, hh) => ceLoss(hh, x, y, reduction(k)))),
    NLLLoss: layerType('NLLLoss', (_a, k) => new LossLayer('NLLLoss', (x, y) => C.nllLoss(E, x, y, reduction(k)))),
    MSELoss: layerType('MSELoss', (_a, k) => new LossLayer('MSELoss', (x, y, hh) => mse(hh, x, y, reduction(k)))),
    BCELoss: layerType('BCELoss', (_a, k) => new LossLayer('BCELoss', (x, y) => C.bce(E, x, y, false, reduction(k)))),
    BCEWithLogitsLoss: layerType('BCEWithLogitsLoss', (_a, k) => new LossLayer('BCEWithLogitsLoss', (x, y) => C.bce(E, x, y, true, reduction(k)))),
    L1Loss: layerType('L1Loss', () => new LossLayer('L1Loss', (x, y) => C.reduceSum(E, C.unary(E, 'abs', C.binary(E, '-', x, y)), null, false, true))),
    Parameter: { k: 'type', name: 'Parameter', call: (args, kw) => {
      const t = TT.of(T(args[0], 'Parameter').values(), T(args[0], 'Parameter').shape, T(args[0], 'Parameter').dtype)
      t.isParam = true
      t.requiresGrad = !kw.requires_grad || h.truthy(kw.requires_grad)
      return wrap(t)
    } },
  }

  // ---- torch.nn.functional
  const Fattrs: Record<string, Value> = {
    relu: fnv('relu', (args) => {
      const t = T(args[0], 'relu')
      const out = C.unary(E, 'relu', t)
      flowStep(h, 'F.relu', t, out)
      return wrap(out)
    }),
    sigmoid: uni('sigmoid'), tanh: uni('tanh'),
    leaky_relu: fnv('leaky_relu', (args, kw) => {
      const t = T(args[0], 'leaky_relu')
      return wrap(C.binary(E, 'maximum', t, C.binary(E, '*', t, TT.scalar(num(kw.negative_slope ?? args[1], 0.01), 'float32'), false, true)))
    }),
    softmax: fnv('softmax', (args, kw) => wrap(C.softmax(E, T(args[0], 'softmax'), dimOf(args, kw, 1) ?? -1, false))),
    log_softmax: fnv('log_softmax', (args, kw) => wrap(C.softmax(E, T(args[0], 'log_softmax'), dimOf(args, kw, 1) ?? -1, true))),
    dropout: fnv('dropout', (args, kw) => wrap(C.dropout(E, T(args[0], 'dropout'), num(kw.p ?? args[1], 0.5), bool(kw.training ?? args[2], true)))),
    max_pool2d: fnv('max_pool2d', (args, kw) => {
      const t = T(args[0], 'max_pool2d')
      const k = pair(h, kw.kernel_size ?? args[1], 2)
      const st = kw.stride ?? args[2]
      const out = C.pool2d(E, t, k, st && st.k !== 'none' ? pair(h, st, 1) : k, pair(h, kw.padding ?? args[3], 0), 'max')
      flowStep(h, 'F.max_pool2d', t, out)
      return wrap(out)
    }),
    avg_pool2d: fnv('avg_pool2d', (args, kw) => {
      const k = pair(h, kw.kernel_size ?? args[1], 2)
      const st = kw.stride ?? args[2]
      return wrap(C.pool2d(E, T(args[0], 'avg_pool2d'), k, st && st.k !== 'none' ? pair(h, st, 1) : k, pair(h, kw.padding ?? args[3], 0), 'avg'))
    }),
    conv2d: fnv('conv2d', (args, kw) => {
      const b = kw.bias ?? args[2]
      const x = T(args[0], 'conv2d')
      const w = T(kw.weight ?? args[1], 'conv2d weight')
      const stride = pair(h, kw.stride ?? args[3], 1)
      const padding = pair(h, kw.padding ?? args[4], 0)
      const out = C.conv2d(E, x, w, b && b.k !== 'none' ? asTT(b) : null, { stride, padding })
      traceConv(h, x, w, out, stride, padding, label(h, args[0], 'input'), label(h, kw.weight ?? args[1], 'weight'))
      return wrap(out)
    }),
    linear: fnv('linear', (args, kw) => {
      const b = kw.bias ?? args[2]
      return wrap(C.linear(E, T(args[0], 'linear'), T(kw.weight ?? args[1], 'linear weight'), b && b.k !== 'none' ? asTT(b) : null))
    }),
    cross_entropy: fnv('cross_entropy', (args, kw) => wrap(ceLoss(h, T(args[0], 'cross_entropy'), T(kw.target ?? args[1], 'cross_entropy target'), reduction(kw)))),
    nll_loss: fnv('nll_loss', (args, kw) => wrap(C.nllLoss(E, T(args[0], 'nll_loss'), T(args[1], 'nll_loss target'), reduction(kw)))),
    mse_loss: fnv('mse_loss', (args, kw) => wrap(mse(h, T(args[0], 'mse_loss'), T(args[1], 'mse_loss target'), reduction(kw)))),
    binary_cross_entropy: fnv('binary_cross_entropy', (args, kw) => wrap(C.bce(E, T(args[0], 'binary_cross_entropy'), T(args[1], 'target'), false, reduction(kw)))),
    binary_cross_entropy_with_logits: fnv('binary_cross_entropy_with_logits', (args, kw) => wrap(C.bce(E, T(args[0], 'binary_cross_entropy_with_logits'), T(args[1], 'target'), true, reduction(kw)))),
    one_hot: fnv('one_hot', (args, kw) => {
      const t = T(args[0], 'one_hot')
      if (isFloatT(t.dtype)) throw h.err('RuntimeError', 'one_hot is only applicable to index tensor of type LongTensor.')
      const v = t.values()
      const n = kw.num_classes ?? args[1] ? h.toInt((kw.num_classes ?? args[1])!) : Math.max(...v) + 1
      return wrap(TT.of(v.flatMap((c) => Array.from({ length: n }, (_, j) => (j === c ? 1 : 0))), [...t.shape, n], 'int64'))
    }),
  }

  // ---- optim
  const paramsOf = (v: Value): TT[] => h.iterate(v).map((p) => {
    const t = asTT(p)
    if (!t) throw h.err('TypeError', `optimizer can only optimize Tensors, but one of the params is ${p.k === 'obj' ? p.o.cls : p.k}`)
    if (!t.isLeaf) throw h.err('ValueError', "can't optimize a non-leaf Tensor")
    return t
  })
  const optim: Record<string, Value> = {
    SGD: { k: 'type', name: 'SGD', call: (args, kw) => {
      const lr = kw.lr ?? args[1]
      if (!lr) throw h.err('TypeError', "SGD.__init__() missing 1 required positional argument: 'lr'")
      return py.obj(new Optimizer('SGD', paramsOf(args[0] ?? kw.params), h.num(lr), num(kw.momentum, 0), [0.9, 0.999], 1e-8, num(kw.weight_decay, 0)))
    } },
    Adam: { k: 'type', name: 'Adam', call: (args, kw) => {
      const b = kw.betas
      return py.obj(new Optimizer('Adam', paramsOf(args[0] ?? kw.params), num(kw.lr ?? args[1], 0.001), 0, b ? (h.iterate(b).map((x) => h.num(x)) as [number, number]) : [0.9, 0.999], num(kw.eps, 1e-8), num(kw.weight_decay, 0)))
    } },
  }

  const dtypeVals = Object.fromEntries(Object.entries(DTYPES).map(([k, d]) => [k, py.obj(new DtypeObj(d))]))
  const cuda: Value = {
    k: 'module', name: 'torch.cuda', attrs: {
      is_available: fnv('is_available', () => py.bool(false)),
      device_count: fnv('device_count', () => py.int(0)),
      current_device: fnv('current_device', () => { throw h.err('AssertionError', 'Torch not compiled with CUDA enabled') }),
    },
  }
  const data: Value = {
    k: 'module', name: 'torch.utils.data', attrs: {
      TensorDataset: { k: 'type', name: 'TensorDataset', call: (args) => {
        const ts = args.map((a) => T(a, 'TensorDataset'))
        if (ts.some((t) => t.shape[0] !== ts[0].shape[0])) throw h.err('AssertionError', 'Size mismatch between tensors')
        return py.obj(new TensorDataset(ts))
      } },
      DataLoader: { k: 'type', name: 'DataLoader', call: (args, kw) => py.obj(new DataLoader(args[0] ?? kw.dataset, kw.batch_size ?? args[1] ? h.toInt((kw.batch_size ?? args[1])!) : 1, bool(kw.shuffle ?? args[2], false))) },
      Dataset,
    },
  }
  const nnFunctional: Value = { k: 'module', name: 'torch.nn.functional', attrs: Fattrs }
  const nn: Value = { k: 'module', name: 'torch.nn', attrs: { ...nnAttrs, functional: nnFunctional } }
  const optimMod: Value = { k: 'module', name: 'torch.optim', attrs: optim }

  const arange = fnv('arange', (args, kw) => {
    const nums = args.map((a) => h.num(a))
    const [start, end, step] = nums.length === 1 ? [0, nums[0], 1] : [nums[0], nums[1], nums[2] ?? 1]
    if (step === 0) throw h.err('RuntimeError', 'step must be nonzero')
    const n = Math.max(0, Math.ceil((end - start) / step))
    if (n > h.maxSize) throw h.err('RuntimeError', `arange of ${n} elements exceeds the sandbox limit`)
    const anyFloat = args.some((a) => a.k === 'float')
    const out = TT.of(Array.from({ length: n }, (_, i) => start + i * step), [n], dtypeArg(h, kw.dtype) ?? (anyFloat ? 'float32' : 'int64'))
    return finish(out, kw, 'arange')
  })

  const torch: Record<string, Value> = {
    ...dtypeVals,
    __version__: py.str('2.12 (sandbox)'),
    Tensor: { k: 'type', name: 'Tensor', call: (args) => wrap(tensorFrom(h, args[0] ?? py.list([]), 'float32')) },
    Size: { k: 'type', name: 'Size', call: (args) => py.obj(new SizeObj(args[0] ? h.iterate(args[0]).map((v) => h.toInt(v)) : [])) },
    tensor: fnv('tensor', (args, kw) => {
      if (!args.length) throw h.err('TypeError', "tensor() missing 1 required positional arguments: 'data'")
      return finish(tensorFrom(h, args[0], dtypeArg(h, kw.dtype)), kw, 'tensor')
    }),
    as_tensor: fnv('as_tensor', (args, kw) => (args[0]?.k === 'array' && !kw.dtype ? wrap(fromNd(args[0].a, true)) : wrap(tensorFrom(h, args[0], dtypeArg(h, kw.dtype))))),
    from_numpy: fnv('from_numpy', (args) => {
      if (args[0]?.k !== 'array') throw h.err('TypeError', `expected np.ndarray (got ${args[0] ? (args[0].k === 'obj' ? args[0].o.cls : args[0].k) : 'nothing'})`)
      return wrap(fromNd(args[0].a, true))
    }),
    zeros: filled('zeros', () => 0), ones: filled('ones', () => 1), empty: filled('empty', () => 0),
    full: fnv('full', (args, kw) => {
      const dims = toDims(h, [args[0]])
      const fill = args[1] ?? kw.fill_value
      return finish(TT.of(new Array(prod(dims)).fill(h.num(fill)), dims, dtypeArg(h, kw.dtype) ?? (fill.k === 'int' ? 'int64' : fill.k === 'bool' ? 'bool' : 'float32')), kw, 'full')
    }),
    rand: filled('rand', () => h.rng.next()),
    randn: filled('randn', () => h.rng.normal()),
    randint: fnv('randint', (args, kw) => {
      const [lo, hi, size] = args.length === 2 ? [0, h.toInt(args[0]), args[1]] : [h.toInt(args[0]), h.toInt(args[1]), args[2] ?? kw.size]
      const dims = toDims(h, [size])
      return wrap(TT.of(Array.from({ length: prod(dims) }, () => lo + Math.floor(h.rng.next() * (hi - lo))), dims, dtypeArg(h, kw.dtype) ?? 'int64'))
    }),
    eye: fnv('eye', (args, kw) => {
      const n = h.toInt(args[0])
      const m = args[1] ? h.toInt(args[1]) : n
      return finish(TT.of(Array.from({ length: n * m }, (_, i) => (Math.floor(i / m) === i % m ? 1 : 0)), [n, m], dtypeArg(h, kw.dtype) ?? 'float32'), kw, 'eye')
    }),
    arange,
    linspace: fnv('linspace', (args, kw) => {
      const a = h.num(args[0])
      const b = h.num(args[1])
      const n = h.toInt(args[2] ?? kw.steps)
      return finish(TT.of(Array.from({ length: n }, (_, i) => (n === 1 ? a : a + ((b - a) * i) / (n - 1))), [n], dtypeArg(h, kw.dtype) ?? 'float32'), kw, 'linspace')
    }),
    zeros_like: like('zeros_like', () => 0), ones_like: like('ones_like', () => 1),
    rand_like: like('rand_like', () => h.rng.next(), true), randn_like: like('randn_like', () => h.rng.normal(), true),
    full_like: fnv('full_like', (args, kw) => {
      const t = T(args[0], 'full_like')
      return wrap(TT.of(t.values().map(() => h.num(args[1])), t.shape, dtypeArg(h, kw.dtype) ?? t.dtype))
    }),
    manual_seed: fnv('manual_seed', (args, kw) => {
      h.rng.seed(h.toInt(kw.seed ?? args[0]))
      return py.obj(new GradFnObj('Generator'))
    }),
    is_tensor: fnv('is_tensor', (args) => py.bool(!!asTT(args[0]))),
    numel: fnv('numel', (args) => py.int(T(args[0], 'numel').size)),
    cat: fnv('cat', (args, kw) => wrap(C.catT(E, h.iterate(args[0]).map((v) => T(v, 'cat')), dimOf(args, kw, 1) ?? 0))),
    concat: fnv('concat', (args, kw) => wrap(C.catT(E, h.iterate(args[0]).map((v) => T(v, 'concat')), dimOf(args, kw, 1) ?? 0))),
    stack: fnv('stack', (args, kw) => wrap(C.catT(E, h.iterate(args[0]).map((v) => T(v, 'stack')), dimOf(args, kw, 1) ?? 0, true))),
    matmul: fnv('matmul', (args) => {
      const a = T(args[0], 'matmul')
      const b = T(args[1], 'matmul')
      const out = C.matmul(E, a, b)
      trace(h, 'torch.matmul', 'matmul', [{ label: label(h, args[0], 'a'), t: a }, { label: label(h, args[1], 'b'), t: b }], out)
      return wrap(out)
    }),
    mm: twoArg('mm', '@'), dot: twoArg('dot', '@'), bmm: twoArg('bmm', '@'),
    mul: twoArg('mul', '*'), add: twoArg('add', '+'), sub: twoArg('sub', '-'), div: twoArg('div', '/'), pow: twoArg('pow', '**'),
    eq: twoArg('eq', '=='), gt: twoArg('gt', '>'), lt: twoArg('lt', '<'),
    maximum: fnv('maximum', (args) => wrap(C.binary(E, 'maximum', T(args[0], 'maximum'), T(args[1], 'maximum')))),
    minimum: fnv('minimum', (args) => wrap(C.binary(E, 'minimum', T(args[0], 'minimum'), T(args[1], 'minimum')))),
    sum: viaMethod('sum'), mean: viaMethod('mean'), std: viaMethod('std'), var: viaMethod('var'),
    argmax: viaMethod('argmax'), argmin: viaMethod('argmin'),
    max: fnv('max', (args, kw) => maxMin(h, 'max', T(args[0], 'max'), kw.dim ?? args[1], kw.keepdim ?? args[2], args[0])),
    min: fnv('min', (args, kw) => maxMin(h, 'min', T(args[0], 'min'), kw.dim ?? args[1], kw.keepdim ?? args[2], args[0])),
    exp: uni('exp'), log: uni('log'), sqrt: uni('sqrt'), abs: uni('abs'), relu: uni('relu'), sigmoid: uni('sigmoid'), tanh: uni('tanh'), square: uni('square'), sin: uni('sin'), cos: uni('cos'),
    softmax: fnv('softmax', (args, kw) => wrap(C.softmax(E, T(args[0], 'softmax'), dimOf(args, kw, 1) ?? -1, false))),
    log_softmax: fnv('log_softmax', (args, kw) => wrap(C.softmax(E, T(args[0], 'log_softmax'), dimOf(args, kw, 1) ?? -1, true))),
    clamp: viaMethod('clamp'), round: viaMethod('round'),
    reshape: fnv('reshape', (args) => h.call(h.getAttr(args[0], 'reshape'), [args[1]])),
    flatten: fnv('flatten', (args, kw) => {
      const t = T(args[0], 'flatten')
      const out = C.flattenT(E, t, optDim(h, kw.start_dim ?? args[1]) ?? 0, optDim(h, kw.end_dim ?? args[2]) ?? -1)
      flowStep(h, 'torch.flatten', t, out)
      return wrap(out)
    }),
    squeeze: viaMethod('squeeze'), unsqueeze: viaMethod('unsqueeze'), transpose: viaMethod('transpose'), permute: viaMethod('permute'),
    t: viaMethod('t'), clone: viaMethod('clone'), detach: viaMethod('detach'), where: fnv('where', (args) => {
      const c = T(args[0], 'where')
      const a = operand(h, args[1])
      const b = operand(h, args[2])
      if (!a || !b) throw h.err('TypeError', 'where(condition, x, y): x and y must be tensors or numbers')
      const shape = broadcastShapes([c.shape, a.t.shape, b.t.shape])
      const [ic, ia, ib] = [c, a.t, b.t].map((t) => C.bcastIndex(t.shape, shape))
      const [cv, av, bv] = [c, a.t, b.t].map((t) => t.values())
      return wrap(TT.of(Array.from({ length: prod(shape) }, (_, i) => (cv[ic[i]] ? av[ia[i]] : bv[ib[i]])), shape, C.promote(a.t.dtype, b.t.dtype, a.weak, b.weak)))
    }),
    no_grad: { k: 'type', name: 'no_grad', call: () => py.obj(new GradMode(false)) },
    enable_grad: { k: 'type', name: 'enable_grad', call: () => py.obj(new GradMode(true)) },
    set_grad_enabled: fnv('set_grad_enabled', (args) => {
      E.grad = h.truthy(args[0])
      return py.obj(new GradMode(E.grad))
    }),
    is_grad_enabled: fnv('is_grad_enabled', () => py.bool(E.grad)),
    device: { k: 'type', name: 'device', call: (args) => {
      const s = h.str(args[0] ?? py.str('cpu'))
      return py.obj(new DeviceObj(s.split(':')[0]))
    } },
    save: fnv('save', (args) => {
      S.files.set(h.str(args[1]), args[0])
      h.print(`(sandbox) saved to ${h.str(args[1])} in memory: there is no disk here\n`)
      return py.NONE
    }),
    load: fnv('load', (args) => {
      const v = S.files.get(h.str(args[0]))
      if (!v) throw h.err('FileNotFoundError', `[Errno 2] No such file or directory: '${h.str(args[0])}' (the sandbox only knows files saved with torch.save in this run)`)
      return v
    }),
    cuda, nn, optim: optimMod,
    utils: { k: 'module', name: 'torch.utils', attrs: { data } },
  }
  const torchMod: Value = { k: 'module', name: 'torch', attrs: torch }

  // torchvision: the lecture's transforms; no datasets to download here
  const transforms: Value = {
    k: 'module', name: 'torchvision.transforms', attrs: {
      Compose: { k: 'type', name: 'Compose', call: (args) => {
        const steps = h.iterate(args[0])
        return fnv('Compose', (a) => steps.reduce((x, f) => h.call(f, [x]), a[0]))
      } },
      ToTensor: { k: 'type', name: 'ToTensor', call: () => fnv('ToTensor', (a) => {
        const arr = toArray(a[0])
        const isInt = arr.dtype !== 'float64'
        // (H, W) or (H, W, C) in 0..255 → (C, H, W) in 0..1
        const hw = arr.ndim === 2 ? [1, ...arr.shape] : [arr.shape[2], arr.shape[0], arr.shape[1]]
        const v = arr.values()
        const out = arr.ndim === 2 ? v : Array.from({ length: v.length }, (_, i) => {
          const c = Math.floor(i / (hw[1] * hw[2]))
          const p = i % (hw[1] * hw[2])
          return v[p * hw[0] + c]
        })
        return wrap(TT.of(out.map((x) => (isInt ? x / 255 : x)), hw, 'float32'))
      }) },
      Normalize: { k: 'type', name: 'Normalize', call: (args) => {
        const mean = h.iterate(args[0]).map((v) => h.num(v))
        const std = h.iterate(args[1]).map((v) => h.num(v))
        return fnv('Normalize', (a) => {
          const t = T(a[0], 'Normalize')
          const c = t.shape[0]
          const per = t.size / c
          return wrap(TT.of(t.values().map((x, i) => (x - mean[Math.floor(i / per) % mean.length]) / std[Math.floor(i / per) % std.length]), t.shape, t.dtype))
        })
      } },
    },
  }
  const datasets: Value = {
    k: 'module', name: 'torchvision.datasets', attrs: {
      MNIST: { k: 'type', name: 'MNIST', call: () => { throw h.err('RuntimeError', 'the sandbox cannot download MNIST: build a small tensor dataset yourself (see the training examples)') } },
    },
  }
  return {
    torch: torchMod, 'torch.nn': nn, 'torch.nn.functional': nnFunctional, 'torch.optim': optimMod, 'torch.utils': torch.utils, 'torch.utils.data': data, 'torch.cuda': cuda,
    torchvision: { k: 'module', name: 'torchvision', attrs: { transforms, datasets } }, 'torchvision.transforms': transforms, 'torchvision.datasets': datasets,
  }
}

export const TORCH: PyLib = {
  modules: ['torch', 'torch.nn', 'torch.nn.functional', 'torch.optim', 'torch.utils', 'torch.utils.data', 'torch.cuda', 'torchvision', 'torchvision.transforms', 'torchvision.datasets'],
  load: build,
}
