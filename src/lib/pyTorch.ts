/**
 * torch for the sandbox (lecture 9): tensors with PyTorch's printing and dtype rules, autograd (the recorded graph is
 * shown after backward()), nn layers and nn.Module subclasses (each forward pass is traced layer by layer with its
 * shapes and parameter counts), losses, optimisers and learning-rate schedulers, torch.utils.data, torch.linalg and
 * torch.from_numpy's shared memory.
 * Built on tensorCore; random values come from the sandbox generator, so they differ from real PyTorch.
 */
import { NDArray, PyError, broadcastShapes, cStrides, planIndex, prod, setIndex, sliceIndices } from './ndarray'
import type { DType, IndexItem } from './ndarray'
import { SandboxRandom } from './ndops'
import { PyDict, PyObj, formatSpec, isNum, numOf, py, repr, str, toArray, typeName } from './minipy'
import type { ApiKind, GridSnapshot, Host, HostClass, InstValue, Kw, PyLib, Value } from './minipy'
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
  /** set_grad_enabled() takes effect when called; its `with` must still restore what was there before */
  applied = false
  constructor(enabled: boolean) {
    super()
    this.enabled = enabled
    this.cls = enabled ? 'enable_grad' : 'no_grad'
  }
  repr() { return `<torch.autograd.grad_mode.${this.cls} object>` }
  enter(h: Host): Value {
    const E = stateOf(h).E
    if (!this.applied) this.prev = E.grad
    E.grad = this.enabled
    return py.NONE
  }
  exit(h: Host) {
    stateOf(h).E.grad = this.prev
    this.applied = false
  }
  /** @torch.no_grad(): the function runs in this mode */
  call(args: Value[], _kw: Kw, h: Host): Value {
    const f = args[0]
    return { k: 'fn', name: 'decorated', call: (a, k) => {
      this.enter(h)
      try {
        return h.call(f, a, k)
      } finally {
        this.exit(h)
      }
    } }
  }
}

/** torch.Generator, and what torch.manual_seed returns: the global one */
class GeneratorObj extends PyObj {
  readonly cls = 'Generator'
  rng: SandboxRandom
  start = 0
  constructor(rng: SandboxRandom) {
    super()
    this.rng = rng
  }
  repr() { return '<torch._C.Generator object>' }
  getAttr(name: string, h: Host): Value | undefined {
    const self = py.obj(this)
    if (name === 'manual_seed') return { k: 'fn', name, call: (args) => { this.start = h.toInt(args[0]); this.rng.seed(this.start); return self } }
    if (name === 'initial_seed') return { k: 'fn', name, call: () => py.int(this.start) }
    return undefined
  }
}

// ---------------------------------------------------------------- tensors

const WRAP = new WeakMap<TT, TensorObj>()
const objOf = (t: TT): TensorObj => {
  let o = WRAP.get(t)
  if (!o) WRAP.set(t, (o = new TensorObj(t)))
  return o
}
const wrap = (t: TT): Value => py.obj(objOf(t))
export const tensorValue = wrap

export const asTT = (v: Value): TT | null => (v.k === 'obj' && v.o instanceof TensorObj ? v.o.t : null)

/**
 * A Python value as a tensor operand; scalars are "weak" (they do not widen the tensor's dtype).
 * like: the tensor it meets, which decides the precision a Python float keeps (0.1 is not float32's 0.1 next to a float64).
 */
const operand = (h: Host, v: Value, like?: TDType): { t: TT; weak: boolean } | null => {
  const t = asTT(v)
  if (t) return { t, weak: false }
  if (v.k === 'int' || v.k === 'bool') return { t: TT.scalar(numOf(v), v.k === 'bool' ? 'bool' : 'int64'), weak: true }
  if (v.k === 'float') return { t: TT.scalar(v.v, like === 'float64' ? 'float64' : 'float32'), weak: true }
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
const trace = (h: Host, api: string, kind: ApiKind, ops: { label: string; t: TT }[], out: TT, axis: number | null = null, src?: ArrayLike<number> | [number, number][][]) => {
  if (!h.tracing) return
  if (out.size > SNAP_MAX || ops.some((o) => o.t.size > SNAP_MAX)) return
  // src: the input element behind each output element, or for a kind that combines several, the whole group
  const groups = src && Array.from(src as ArrayLike<number | [number, number][]>, (x): [number, number][] => (typeof x !== 'number' ? x : x >= 0 ? [[0, x]] : []))
  h.traceCall(api, kind, ops.map((o) => ({ label: o.label, snap: snapOf(o.t) })), snapOf(out), formatTensor(out), axis, groups)
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

/** torch.FloatTensor and friends */
const LEGACY: Record<string, TDType> = { FloatTensor: 'float32', DoubleTensor: 'float64', LongTensor: 'int64', IntTensor: 'int32', ByteTensor: 'uint8', BoolTensor: 'bool' }

const dtypeArg = (h: Host, v: Value | undefined): TDType | null => {
  if (!v || v.k === 'none') return null
  if (v.k === 'obj' && v.o instanceof DtypeObj) return v.o.d
  if (v.k === 'type' && v.name === 'float') return 'float32'
  if (v.k === 'type' && v.name === 'int') return 'int64'
  if (v.k === 'type' && v.name === 'bool') return 'bool'
  if (v.k === 'type' && LEGACY[v.name]) return LEGACY[v.name]
  if (v.k === 'str' && LEGACY[v.v.replace('torch.', '')]) return LEGACY[v.v.replace('torch.', '')]
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

/** torch's argument parser takes NumPy's axis and keepdims for dim and keepdim */
const npKw = (kw: Kw): Kw => {
  if (!kw.axis && !kw.keepdims) return kw
  const { axis, keepdims, ...rest } = kw
  return { ...rest, ...(axis ? { dim: axis } : {}), ...(keepdims ? { keepdim: keepdims } : {}) }
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
  isa(name: string) { return name === 'Tensor' || (name === 'Parameter' && this.t.isParam) || LEGACY[name] === this.t.dtype }
  toArray(): NDArray {
    // a single element: float(loss) reads it, and a Python number never needed the graph
    if (this.t.requiresGrad && this.t.size !== 1) throw new PyError('RuntimeError', "Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.")
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
    const o = operand(h, v, this.t.dtype)
    if (!o) throw h.err('TypeError', `can't assign a ${v.k === 'obj' ? v.o.cls : v.k} to a torch tensor`)
    const E = stateOf(h).E
    const items = indexItems(h, idx)
    const plan = planIndex(this.t.a, items)
    // extra leading dimensions of size 1 do not count, as in PyTorch
    const extra = Math.max(0, o.t.ndim - plan.shape.length)
    const fits = o.t.shape.slice(0, extra).every((d) => d === 1) && o.t.shape.slice(extra).every((d, k) => d === 1 || d === plan.shape[plan.shape.length - (o.t.ndim - extra) + k])
    if (!fits) throw h.err('RuntimeError', `shape mismatch: value tensor of shape [${o.t.shape.join(', ')}] cannot be broadcast to indexing result of shape [${plan.shape.join(', ')}]`)
    const val = extra ? C.reshapeT(E, o.t, o.t.shape.slice(extra), false) : o.t
    if (E.grad && (val.requiresGrad || this.t.requiresGrad)) {
      this.rebase(h, (old) => C.setItemT(E, old, plan.addresses, plan.shape, val))
      return
    }
    setIndex(this.t.a, items, NDArray.create(val.values().map((x) => C.castT(x, this.t.dtype)), val.shape, this.t.a.dtype))
  }
  checkInplace(h: Host) {
    const t = this.t
    if (!stateOf(h).E.grad) return
    if (t.isLeaf && t.requiresGrad) throw h.err('RuntimeError', 'a leaf Variable that requires grad is being used in an in-place operation.')
    // follow the views down to the tensor that owns this memory
    let base = t
    while (base.fn && base.fn.inputs[0]?.a.data === t.a.data) base = base.fn.inputs[0]!
    if (base !== t && base.isLeaf && base.requiresGrad) throw h.err('RuntimeError', 'a view of a leaf Variable that requires grad is being used in an in-place operation.')
  }
  /**
   * An in-place op: the result overwrites this tensor and its history continues from a stand-in for the old value, so
   * y = x * 2; y.mul_(3) still differentiates to 6. reshape: the op only changes the view (t_, unsqueeze_ …).
   */
  rebase(h: Host, compute: (old: TT) => TT, reshape = false): Value {
    this.checkInplace(h)
    const t = this.t
    const old = new TT(t.a, t.dtype)
    old.fn = t.fn
    old.requiresGrad = t.requiresGrad
    const res = compute(old)
    if (!isFloatT(t.dtype) && isFloatT(res.dtype)) throw h.err('RuntimeError', `result type ${C.typeLabel(res.dtype)} can't be cast to the desired output type ${C.typeLabel(t.dtype)}`)
    if (reshape) t.a = res.a
    else {
      if (res.shape.join() !== t.shape.join()) throw h.err('RuntimeError', `output with shape [${t.shape.join(', ')}] doesn't match the broadcast shape [${res.shape.join(', ')}]`)
      if (res.fn && t.a.base) throw h.err('NotImplementedError', 'an in-place operation on a view of a tensor that is part of the autograd graph is not available in this sandbox: use an out-of-place operation')
      C.writeInto(t, res.values())
    }
    if (res.fn) {
      t.fn = res.fn
      t.requiresGrad = true
    }
    return py.obj(this)
  }
  binop(op: string, other: Value, reflected: boolean, h: Host): Value | undefined {
    const o = operand(h, other, this.t.dtype)
    if (!o) return undefined
    const E = stateOf(h).E
    const [a, b, aw, bw] = reflected ? [o.t, this.t, o.weak, false] : [this.t, o.t, false, o.weak]
    let out: TT
    if (op === '@') {
      sameDtype(h, a, b)
      out = C.matmul(E, a, b)
      trace(h, 'op:@', 'matmul', [{ label: reflected ? 'a' : label(h, py.obj(this), 'a'), t: a }, { label: reflected ? label(h, py.obj(this), 'b') : label(h, other, 'b'), t: b }], out)
      return wrap(out)
    }
    if (['<', '<=', '>', '>=', '==', '!='].includes(op)) {
      out = C.compare(a, b, op as C.CmpOpT)
    } else if (op === '%' && (a.requiresGrad || b.requiresGrad)) {
      out = C.map2T(E, a, b, (x, y) => x - Math.floor(x / y) * y, () => 1, (x, y) => -Math.floor(x / y), C.promote(a.dtype, b.dtype, aw, bw), 'RemainderBackward0')
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
    const o = operand(h, rhs, this.t.dtype)
    if (!o) return false
    const E = stateOf(h).E
    this.rebase(h, (old) => C.binary(E, op as C.BinOpT, old, o.t, false, o.weak))
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
      if (o) {
        this.t.a = o.a
        this.t.dtype = o.dtype
      }
      return true
    }
    return false
  }
  getAttr(name: string, h: Host): Value | undefined {
    const t = this.t
    const S = stateOf(h)
    const E = S.E
    const self = py.obj(this)
    const fn = (call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call: (args, kw) => call(args, npKw(kw)) })
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
      case 'size': return fn((args, kw) => {
        const d = kw.dim ?? args[0]
        return d ? py.int(t.shape[C.normDim(h.toInt(d), t.ndim)]) : py.obj(new SizeObj(t.shape))
      })
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
      case 'numpy': return fn((args, kw) => {
        if (flagOf(h, kw.force ?? args[0])) return py.arr(t.a)
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
        if (flagOf(h, kw.create_graph ?? args[2])) throw h.err('NotImplementedError', 'backward(create_graph=True): higher-order gradients are not available in this sandbox')
        if ((kw.inputs ?? args[3]) && (kw.inputs ?? args[3])!.k !== 'none') throw h.err('NotImplementedError', 'backward(inputs=…) is not available in this sandbox')
        const gt = g && g.k !== 'none' ? asTT(g) : null
        doBackward(h, t, self, gt ? gt.values() : null, flagOf(h, kw.retain_graph ?? args[1]))
        return py.NONE
      })
      case 'retain_grad': return fn(() => { t.retain = true; return py.NONE })
      case 'requires_grad_': return fn((args, kw) => { this.setAttr('requires_grad', args[0] ?? kw.requires_grad ?? py.bool(true), h); return self })
      case 'zero_': return fn(() => this.rebase(h, (old) => C.mapT(E, old, () => 0, null, 'ZeroBackward0', true)))
      case 'fill_': return fn((args, kw) => {
        const v = h.num(args[0] ?? kw.value)
        return this.rebase(h, (old) => C.mapT(E, old, () => v, null, 'FillBackward0', true))
      })
      case 'copy_': return fn((args) => {
        const o = asTT(args[0])
        if (!o) throw h.err('TypeError', 'copy_(): argument must be a Tensor')
        const bad = o.shape.findIndex((d, k) => d !== 1 && d !== t.shape[t.ndim - o.ndim + k])
        if (bad >= 0 || o.ndim > t.ndim) throw h.err('RuntimeError', `The size of tensor a (${t.shape[t.ndim - o.ndim + bad]}) must match the size of tensor b (${o.shape[bad]}) at non-singleton dimension ${t.ndim - o.ndim + bad}`)
        return this.rebase(h, () => C.toDtype(E, C.expandT(E, o, t.shape), t.dtype))
      })
      case 'view': case 'reshape': return fn((args, kw) => {
        const dims = toDims(h, kw.shape ? [kw.shape] : args)
        const out = viewOrReshape(E, t, dims, name === 'view')
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
      case 'squeeze': return fn((args, kw) => {
        const dims = dimsArg(h, kw.dim ?? args[0])?.map((d) => C.normDim(d, t.ndim))
        if (!dims || dims.length === 1) return wrap(C.squeezeT(E, t, dims ? dims[0] : null))
        const out = C.reshapeT(E, t, t.shape.filter((s, d) => !(s === 1 && dims.includes(d))), false)
        if (out.fn) out.fn.name = 'SqueezeBackward1'
        return wrap(out)
      })
      case 'permute': return fn((args, kw) => {
        const out = C.permuteT(E, t, toDims(h, kw.dims ? [kw.dims] : args))
        trace(h, 'Tensor.permute', 'move', [{ label: me(), t }], out)
        return wrap(out)
      })
      case 'transpose': return fn((args, kw) => {
        const out = C.transposeT(E, t, h.toInt(kw.dim0 ?? args[0]), h.toInt(kw.dim1 ?? args[1]))
        trace(h, 'Tensor.transpose', 'move', [{ label: me(), t }], out)
        return wrap(out)
      })
      case 't': return fn(() => {
        if (t.ndim > 2) throw h.err('RuntimeError', `t() expects a tensor with <= 2 dimensions, but self is ${t.ndim}D`)
        return wrap(t.ndim < 2 ? t : C.transposeT(E, t, 0, 1))
      })
      case 'sum': case 'mean': return fn((args, kw) => {
        const dims = dimsArg(h, kw.dim ?? args[0])
        const dt = dtypeArg(h, kw.dtype)
        const out = C.reduceSum(E, dt ? C.toDtype(E, t, dt) : t, dims, flagOf(h, kw.keepdim ?? args[1]), name === 'mean')
        trace(h, `Tensor.${name}`, 'reduce', [{ label: me(), t }], out, dims && dims.length === 1 ? C.normDim(dims[0], t.ndim) : null)
        return wrap(out)
      })
      case 'std': case 'var': return fn((args, kw) => {
        // std(unbiased) or std(dim, unbiased, keepdim)
        const first = args[0]?.k === 'bool' ? null : (kw.dim ?? args[0])
        const unbiased = args[0]?.k === 'bool' ? args[0] : (kw.unbiased ?? args[1])
        const corr = kw.correction ? h.num(kw.correction) : unbiased && !h.truthy(unbiased) ? 0 : 1
        const out = C.variance(E, t, dimsArg(h, first ?? undefined), flagOf(h, kw.keepdim ?? args[2]), corr, name === 'std')
        if (out.fn) out.fn.name = name === 'std' ? 'StdBackward0' : 'VarBackward0'
        return wrap(out)
      })
      case 'max': case 'min': return fn((args, kw) => maxMin(h, name, t, kw.dim ?? args[0], kw.keepdim ?? args[1], self))
      case 'argmax': case 'argmin': return fn((args, kw) => {
        const out = C.argmax(t, optDim(h, kw.dim ?? args[0]), flagOf(h, kw.keepdim ?? args[1]), name === 'argmin')
        trace(h, `Tensor.${name}`, 'reduce', [{ label: me(), t }], out, optDim(h, kw.dim ?? args[0]))
        return wrap(out)
      })
      case 'exp': case 'log': case 'sqrt': case 'abs': case 'relu': case 'sigmoid': case 'tanh': case 'square': case 'sin': case 'cos':
        return fn(() => {
          const out = C.unary(E, name as C.UnOpT, t)
          trace(h, `Tensor.${name}`, 'elementwise', [{ label: me(), t }], out)
          return wrap(out)
        })
      case 'softmax': case 'log_softmax': return fn((args, kw) => {
        const dim = kw.dim ?? args[0]
        if (!dim) throw h.err('TypeError', `${name}() missing 1 required positional argument: 'dim'`)
        const dt = dtypeArg(h, kw.dtype ?? args[1])
        return wrap(C.softmax(E, dt ? C.toDtype(E, t, dt) : t, h.toInt(dim), name === 'log_softmax'))
      })
      case 'pow': return fn((args, kw) => this.binop('**', args[0] ?? kw.exponent, false, h)!)
      case 'matmul': return fn((args) => this.binop('@', args[0], false, h)!)
      case 'mm': case 'mv': case 'bmm': case 'dot': return fn((args) => {
        const o = tensorArg(h, args[0], name)
        const [nd1, nd2] = { mm: [2, 2], mv: [2, 1], bmm: [3, 3], dot: [1, 1] }[name]!
        if (name === 'dot' && (t.ndim !== 1 || o.ndim !== 1)) throw h.err('RuntimeError', `1D tensors expected, but got ${t.ndim}D and ${o.ndim}D tensors`)
        if (t.ndim !== nd1) throw h.err('RuntimeError', name === 'bmm' ? `batch1 must be a 3D tensor` : `self must be a matrix`)
        if (o.ndim !== nd2) throw h.err('RuntimeError', name === 'bmm' ? `batch2 must be a 3D tensor` : name === 'mv' ? 'vec must be a vector' : 'mat2 must be a matrix')
        return this.binop('@', args[0], false, h)!
      })
      case 'add': case 'sub': return fn((args, kw) => {
        const alpha = kw.alpha ?? args[1]
        const other = kw.other ?? args[0]
        const o = operand(h, other, t.dtype)
        if (!o) throw h.err('TypeError', `${name}(): argument 'other' must be a Tensor or a number, not ${typeName(other)}`)
        const rhs = alpha ? wrap(C.binary(E, '*', o.t, operand(h, alpha, t.dtype)!.t, o.weak, true)) : other
        return this.binop(name === 'add' ? '+' : '-', rhs, false, h)!
      })
      case 'mul': return fn((args, kw) => this.binop('*', kw.other ?? args[0], false, h)!)
      case 'div': return fn((args, kw) => {
        const mode = kw.rounding_mode ?? args[1]
        const q = this.binop('/', kw.other ?? args[0], false, h)!
        if (!mode || mode.k === 'none') return q
        const o = operand(h, kw.other ?? args[0], t.dtype)!
        const res = C.mapT(E, asTT(q)!, mode.k === 'str' && mode.v === 'floor' ? Math.floor : Math.trunc, null, 'DivBackward0', true)
        return wrap(isFloatT(C.promote(t.dtype, o.t.dtype, false, o.weak)) ? res : C.toDtype(E, res, C.promote(t.dtype, o.t.dtype, false, o.weak)))
      })
      case 'eq': case 'ne': case 'gt': case 'lt': case 'ge': case 'le': return fn((args, kw) => this.binop({ eq: '==', ne: '!=', gt: '>', lt: '<', ge: '>=', le: '<=' }[name]!, kw.other ?? args[0], false, h)!)
      case 'float': case 'double': case 'long': case 'int': case 'bool': case 'byte':
        return fn(() => wrap(C.toDtype(E, t, ({ float: 'float32', double: 'float64', long: 'int64', int: 'int32', bool: 'bool', byte: 'uint8' } as Record<string, TDType>)[name])))
      case 'half': case 'short': case 'char': return fn(() => { throw h.err('NotImplementedError', `Tensor.${name}(): this dtype is not available in this sandbox`) })
      case 'type': return fn((args, kw) => {
        const d = dtypeArg(h, kw.dtype ?? args[0])
        return d ? wrap(C.toDtype(E, t, d)) : py.str(`torch.${C.typeLabel(t.dtype)}Tensor`)
      })
      case 'to': return fn((args, kw) => {
        const other = kw.other ?? args.find((a) => asTT(a))
        const target = kw.dtype ?? args.find((a) => a.k === 'obj' && a.o instanceof DtypeObj)
        const d = other ? asTT(other)!.dtype : target ? dtypeArg(h, target) : null
        const dev = kw.device ?? args.find((a) => a.k === 'str' || (a.k === 'obj' && a.o instanceof DeviceObj))
        if (dev && h.str(dev).startsWith('cuda')) throw h.err('AssertionError', 'Torch not compiled with CUDA enabled')
        return d ? wrap(C.toDtype(E, t, d)) : self
      })
      case 'cpu': return fn(() => self)
      case 'cuda': return fn(() => { throw h.err('AssertionError', 'Torch not compiled with CUDA enabled') })
      case 'apply_': case 'register_hook': return undefined
    }
    const m = METHODS[name]
    if (m) return fn((args, kw) => m[1]({ h, E, t, self, name, a: bind(h, name, m[0], args, kw) }))
    // x.sqrt_(): the out-of-place op, written back
    if (name.endsWith('_')) {
      const base = name.slice(0, -1)
      if (INPLACE.has(base)) return fn((args, kw) => this.rebase(h, (old) => asTT(h.call(h.getAttr(wrap(old), base), args, kw))!))
      if (INPLACE_VIEW.has(base)) return fn((args, kw) => this.rebase(h, (old) => asTT(h.call(h.getAttr(wrap(old), base), args, kw))!, true))
    }
    return undefined
  }
}

const promote2 = (a: TDType, b: TDType) => C.promote(a, b)
const optNumV = (h: Host, v: Value | undefined): number | null => (!v || v.k === 'none' ? null : h.num(v))
const flagOf = (h: Host, v: Value | undefined, d = false): boolean => (v && v.k !== 'none' ? h.truthy(v) : d)
const roundHalfEvenT = (v: number, d: number) => {
  const f = 10 ** d
  const x = v * f
  const r = Math.round(x)
  return (Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r) / f
}

/** the strides of a view of a non-contiguous tensor, when it can be one (PyTorch's computeStride) */
const viewStrides = (shape: number[], strides: number[], target: number[]): number[] | null => {
  if (!shape.length) return target.map(() => 1)
  const out = new Array<number>(target.length).fill(0)
  let v = target.length - 1
  let base = strides[strides.length - 1]
  let inChunk = 1
  let viewChunk = 1
  for (let d = shape.length - 1; d >= 0; d--) {
    inChunk *= shape[d]
    if (d > 0 && (shape[d - 1] === 1 || strides[d - 1] === inChunk * base)) continue
    while (v >= 0 && (viewChunk < inChunk || target[v] === 1)) {
      out[v] = viewChunk * base
      viewChunk *= target[v]
      v--
    }
    if (viewChunk !== inChunk) return null
    if (d > 0) {
      base = strides[d - 1]
      inChunk = 1
      viewChunk = 1
    }
  }
  return v === -1 ? out : null
}

const viewOrReshape = (E: Engine, t: TT, dims: number[], mustView: boolean): TT => {
  if (t.a.isCContiguous() || !mustView) return C.reshapeT(E, t, dims, mustView)
  const shape = C.resolveShape(t.size, dims, t.shape)
  if (!viewStrides(t.shape, t.a.strides, shape)) return C.reshapeT(E, t, dims, true)
  return C.viewT(E, t, (x) => x.view(shape, viewStrides(x.shape, x.strides, shape)!, x.offset), 'ViewBackward0')
}

const sameDtype = (h: Host, a: TT, b: TT) => {
  if (a.dtype === b.dtype) return
  const [x, y] = [C.typeLabel(a.dtype), C.typeLabel(b.dtype)]
  throw h.err('RuntimeError', a.ndim === 1 && b.ndim === 1 ? `dot : expected both vectors to have same dtype, but found ${x} and ${y}` : `mat1 and mat2 must have the same dtype, but got ${x} and ${y}`)
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
    const out = C.extremeT(E, t, null, false, isMin, isMin ? 'MinBackward1' : 'MaxBackward1')
    trace(h, `torch.${isMin ? 'min' : 'max'}`, 'reduce', [{ label: label(h, self, 'x'), t }], out)
    return wrap(out)
  }
  const dim = h.toInt(dimV)
  const r = C.maxAlong(E, t, dim, !!(keepV && h.truthy(keepV)), isMin)
  trace(h, `torch.${isMin ? 'min' : 'max'}`, 'reduce', [{ label: label(h, self, 'x'), t }], r.values, C.normDim(dim, t.ndim))
  return py.obj(new ReturnTypes(isMin ? 'min' : 'max', wrap(r.values), wrap(r.indices)))
}

// ---------------------------------------------------------------- the rest of the tensor methods

interface TCall {
  h: Host
  E: Engine
  t: TT
  self: Value
  name: string
  /** the arguments, by parameter name */
  a: Record<string, Value | undefined>
}
/** [parameter names, implementation]; '*x' collects the remaining positional arguments, names after it are keyword-only */
type TMethod = [string, (c: TCall) => Value]

/** arguments by parameter name. A keyword the function does not list is an error, never silently ignored */
const bind = (h: Host, fname: string, params: string, args: Value[], kw: Kw): Record<string, Value | undefined> => {
  const names = params ? params.split(' ') : []
  const star = names.findIndex((n) => n.startsWith('*'))
  const fixed = star >= 0 ? names.slice(0, star) : names
  if (star < 0 && args.length > names.length) throw h.err('TypeError', `${fname}() takes ${names.length} positional argument${names.length === 1 ? '' : 's'} but ${args.length} were given`)
  const out: Record<string, Value | undefined> = {}
  fixed.forEach((n, i) => (out[n] = args[i]))
  if (star >= 0 && args.length > star) out[names[star].slice(1)] = py.tuple(args.slice(star))
  for (const k in kw) {
    if (!names.includes(k) && !names.includes(`*${k}`)) {
      if (k === 'out') throw h.err('NotImplementedError', `${fname}(): the out= argument is not available in this sandbox`)
      throw h.err('TypeError', `${fname}() got an unexpected keyword argument '${k}'`)
    }
    out[k] = kw[k]
  }
  return out
}

/** the finished result of Tensor.<name>: a step in the step list, and a row of the shape trace if the shape changed */
const done = (c: TCall, out: TT, kind: ApiKind, axis: number | null = null, src?: ArrayLike<number> | [number, number][][]): Value => {
  trace(c.h, `Tensor.${c.name}`, kind, [{ label: label(c.h, c.self, 'x'), t: c.t }], out, axis, src)
  if (out.shape.join() !== c.t.shape.join()) flowStep(c.h, c.name, c.t, out)
  return wrap(out)
}

/** worth working out where each output cell came from: trace() would otherwise drop it */
const shown = (c: TCall) => c.h.tracing && c.t.size <= SNAP_MAX

const need = (c: TCall, key: string): Value => {
  const v = c.a[key]
  if (!v) throw c.h.err('TypeError', `${c.name}() missing 1 required positional argument: '${key}'`)
  return v
}
const tArg = (c: TCall, key: string): TT => tensorArg(c.h, need(c, key), `${c.name}(): argument '${key}'`)
const intArg = (c: TCall, key: string, d?: number): number => {
  const v = c.a[key]
  if (v && v.k !== 'none') return c.h.toInt(v)
  if (d === undefined) throw c.h.err('TypeError', `${c.name}() missing 1 required positional argument: '${key}'`)
  return d
}
const flag = (c: TCall, key: string, d = false) => flagOf(c.h, c.a[key], d)
const sizes = (c: TCall, key: string): number[] => toDims(c.h, c.a[key]?.k === 'tuple' ? c.a[key]!.items : [need(c, key)])
const withDtype = (c: TCall, key = 'dtype'): TT => {
  const d = dtypeArg(c.h, c.a[key])
  return d ? C.toDtype(c.E, c.t, d) : c.t
}

const results = (name: string, r: { values: TT; indices: TT }) => py.obj(new ReturnTypes(name, wrap(r.values), wrap(r.indices)))

/** median, mode, kthvalue along a dim: a step that reduces that dim, and the (values, indices) they return */
const picked = (c: TCall, name: string, dim: number, r: { values: TT; indices: TT }): Value => {
  trace(c.h, `Tensor.${name}`, 'reduce', [{ label: label(c.h, c.self, 'x'), t: c.t }], r.values, C.normDim(dim, c.t.ndim))
  return results(name, r)
}

/** x.expand(…) and torch.broadcast_to: -1 keeps a size, new dimensions go in front */
const expandTo = (h: Host, E: Engine, t: TT, target: number[]): TT => {
  const lead = target.length - t.ndim
  if (lead < 0) throw h.err('RuntimeError', `the number of sizes provided (${target.length}) must be greater or equal to the number of dimensions in the tensor (${t.ndim})`)
  const shape = target.map((d, k) => {
    const cur = k < lead ? null : t.shape[k - lead]
    if (d === -1) {
      if (cur === null) throw h.err('RuntimeError', `The expanded size of the tensor (-1) isn't allowed in a leading, non-existing dimension ${k}`)
      return cur
    }
    if (cur !== null && cur !== 1 && cur !== d) throw h.err('RuntimeError', `The expanded size of the tensor (${d}) must match the existing size (${cur}) at non-singleton dimension ${k}.  Target sizes: [${target.join(', ')}].  Tensor sizes: [${t.shape.join(', ')}]`)
    return d
  })
  return C.expandT(E, t, shape)
}

/** x.repeat(…): reps ≥ dims, a tile of the whole tensor */
const repeated = (c: TCall, reps: number[]): Value => {
  const lead = reps.length - c.t.ndim
  const inShape = [...new Array<number>(Math.max(0, lead)).fill(1), ...c.t.shape]
  const maps = reps.map((r, d) => Array.from({ length: r * inShape[d] }, (_, i) => i % inShape[d]))
  const src = C.crossSrc(inShape, maps)
  return done(c, C.takeT(c.E, c.t, src, maps.map((m) => m.length), 'RepeatBackward0'), 'move', null, src)
}

const identityMaps = (t: TT) => t.shape.map((n) => Array.from({ length: n }, (_, i) => i))

/** a copy built axis by axis: maps[d][i] is the input position for output position i on axis d */
const byMaps = (c: TCall, maps: number[][], gradName: string, axis: number | null = null): Value => {
  const src = C.crossSrc(c.t.shape, maps)
  return done(c, C.takeT(c.E, c.t, src, maps.map((m) => m.length), gradName), 'move', axis, src)
}

const flipped = (c: TCall, dims: number[]): Value => {
  const ds = dims.map((d) => C.normDim(d, c.t.ndim))
  const dup = ds.find((d, i) => ds.indexOf(d) !== i)
  if (dup !== undefined) throw c.h.err('RuntimeError', `dim ${dup} appears multiple times in the list of dims`)
  return byMaps(c, identityMaps(c.t).map((m, d) => (ds.includes(d) ? [...m].reverse() : m)), 'FlipBackward0')
}

const parts = (c: TCall, d: number, lens: number[], gradName: string): Value => {
  let at = 0
  return py.tuple(lens.map((n) => {
    const start = at
    const p = C.narrowT(c.E, c.t, d, start, n, gradName)
    at += n
    return done(c, p, 'move', d, shown(c) ? C.viewSource(c.t, (x) => x.view(x.shape.map((s, k) => (k === d ? n : s)), x.strides, x.offset + start * x.strides[d])) : undefined)
  }))
}

const needDims = (c: TCall, what: string) => {
  if (!c.t.ndim) throw c.h.err('RuntimeError', `${what} expects at least a 1-dimensional tensor`)
}

const boolMap = (c: TCall, f: (v: number) => boolean): Value => done(c, TT.of(c.t.values().map((v) => (f(v) ? 1 : 0)), c.t.shape, 'bool'), 'elementwise')

const reduceAxis = (c: TCall, dims: number[] | null) => (dims && dims.length === 1 ? C.normDim(dims[0], c.t.ndim) : null)

const countMethod = (f: (nz: number, total: number) => number, dtype: (t: TT) => TDType): TMethod => ['dim keepdim', (c) => {
  const dims = dimsArg(c.h, c.a.dim)
  return done(c, C.countReduce(c.t, dims, flag(c, 'keepdim'), dtype(c.t), f), 'reduce', reduceAxis(c, dims))
}]

const clampV = (c: TCall, lo: Value | undefined, hi: Value | undefined): Value => {
  const { h, E, t } = c
  const none = (v: Value | undefined) => !v || v.k === 'none'
  if (none(lo) && none(hi)) throw h.err('RuntimeError', "torch.clamp: At least one of 'min' or 'max' must not be None")
  if (asTT(lo ?? py.NONE) || asTT(hi ?? py.NONE)) {
    // tensor bounds: min(max(x, lo), hi)
    let out = t
    const bound = (v: Value) => operand(h, v, t.dtype)!
    if (!none(lo)) out = C.binary(E, 'maximum', out, bound(lo!).t, false, bound(lo!).weak)
    if (!none(hi)) out = C.binary(E, 'minimum', out, bound(hi!).t, false, bound(hi!).weak)
    if (out.fn) out.fn.name = 'ClampBackward1'
    return done(c, out, 'elementwise')
  }
  const floatBound = lo?.k === 'float' || hi?.k === 'float'
  const x = !isFloatT(t.dtype) && floatBound ? C.toDtype(E, t, 'float32') : t
  return done(c, C.clamp(E, x, optNumV(h, lo), optNumV(h, hi)), 'elementwise')
}

const closeness = (c: TCall): TT => {
  const o = tArg(c, 'other')
  const { t, h } = c
  if (t.dtype !== o.dtype) throw h.err('RuntimeError', `${C.typeLabel(t.dtype)} did not match ${C.typeLabel(o.dtype)}`)
  const rtol = c.a.rtol ? h.num(c.a.rtol) : 1e-5
  const atol = c.a.atol ? h.num(c.a.atol) : 1e-8
  const equalNan = flag(c, 'equal_nan')
  const R = t.dtype === 'float64' ? (x: number) => x : Math.fround
  const shape = broadcastShapes([t.shape, o.shape])
  const [ia, ib] = [C.bcastIndex(t.shape, shape), C.bcastIndex(o.shape, shape)]
  const [av, bv] = [t.values(), o.values()]
  return TT.of(Array.from({ length: prod(shape) }, (_, i) => {
    const [x, y] = [av[ia[i]], bv[ib[i]]]
    if (x === y || (equalNan && Number.isNaN(x) && Number.isNaN(y))) return 1
    if (!rtol && !atol) return 0
    const actual = R(Math.abs(R(x - y)))
    return Number.isFinite(actual) && actual <= R(atol + Math.abs(R(rtol * y))) ? 1 : 0
  }), shape, 'bool')
}

/** the unique values or (with dim) the unique slices, sorted; inverse maps each original back to its unique */
const uniqueOf = (c: TCall): Value => {
  const { t, h, E } = c
  const dimV = c.a.dim
  const withInverse = flag(c, 'return_inverse')
  const withCounts = flag(c, 'return_counts')
  let out: TT
  let inverse: number[]
  let counts: number[] = []
  if (!dimV || dimV.k === 'none') {
    const order = t.values().map((v, i) => [v, i]).sort((p, q) => (Number.isNaN(p[0]) ? 1 : Number.isNaN(q[0]) ? -1 : p[0] - q[0]))
    const vals: number[] = []
    inverse = new Array<number>(order.length)
    for (const [v, i] of order) {
      if (!vals.length || vals[vals.length - 1] !== v) {
        vals.push(v)
        counts.push(0)
      }
      counts[counts.length - 1]++
      inverse[i] = vals.length - 1
    }
    out = TT.of(vals, [vals.length], t.dtype)
    trace(h, 'Tensor.unique', 'unique', [{ label: label(h, c.self, 'x'), t }], out)
    return withInverse || withCounts ? py.tuple([wrap(out), ...(withInverse ? [wrap(TT.of(inverse, t.shape, 'int64'))] : []), ...(withCounts ? [wrap(TT.of(counts, [counts.length], 'int64'))] : [])]) : wrap(out)
  }
  const d = C.normDim(h.toInt(dimV), t.ndim)
  const n = t.shape[d]
  const slices = Array.from({ length: n }, (_, i) => C.narrowT({ grad: false } as Engine, t, d, i, 1).values())
  const cmp = (a: number[], b: number[]) => {
    for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) return a[k] - b[k]
    return 0
  }
  const order = slices.map((_, i) => i).sort((p, q) => cmp(slices[p], slices[q]))
  const keep: number[] = []
  inverse = new Array<number>(n)
  counts = []
  for (const i of order) {
    if (!keep.length || cmp(slices[keep[keep.length - 1]], slices[i]) !== 0) {
      keep.push(i)
      counts.push(0)
    }
    counts[counts.length - 1]++
    inverse[i] = keep.length - 1
  }
  out = C.takeT(E, t, C.crossSrc(t.shape, identityMaps(t).map((m, k) => (k === d ? keep : m))), t.shape.map((s, k) => (k === d ? keep.length : s)), 'UniqueDimBackward0')
  return withInverse || withCounts ? py.tuple([wrap(out), ...(withInverse ? [wrap(TT.of(inverse, [n], 'int64'))] : []), ...(withCounts ? [wrap(TT.of(counts, [counts.length], 'int64'))] : [])]) : wrap(out)
}

/** ‖A‖ for a stack of matrices, dims = the two matrix axes; the norms that need singular values are not available here */
const matrixNorm = (h: Host, E: Engine, a: TT, ord: Value | undefined, dims: [number, number], keepdim: boolean): TT => {
  const [d0, d1] = dims.map((d) => C.normDim(d, a.ndim))
  if (!ord || ord.k === 'none' || (ord.k === 'str' && ord.v === 'fro')) return C.vectorNorm(E, a, 2, [d0, d1], keepdim)
  if (ord.k === 'str' && ord.v === 'nuc') throw h.err('NotImplementedError', "ord='nuc' needs a singular value decomposition, which this sandbox does not have")
  const p = ord.k === 'str' ? NaN : h.num(ord)
  if (p === 2 || p === -2) throw h.err('NotImplementedError', `ord=${p} needs a singular value decomposition, which this sandbox does not have`)
  if (Math.abs(p) !== 1 && Math.abs(p) !== Infinity) throw h.err('RuntimeError', `linalg.matrix_norm: Order ${repr(ord)} not supported.`)
  if (!isFloatT(a.dtype)) throw h.err('RuntimeError', `linalg.matrix_norm: Expected a floating point or complex tensor as input. Got ${C.typeLabel(a.dtype)}`)
  // ±1: sum over rows, then the extreme column; ±∞: the other way round
  const [sumOver, pickOver] = Math.abs(p) === 1 ? [d0, d1] : [d1, d0]
  const best = C.maxAlong(E, C.reduceSum(E, C.unary(E, 'abs', a), [sumOver], true, false), pickOver, true, p < 0).values
  return keepdim ? best : C.reshapeT(E, best, a.shape.filter((_, k) => k !== d0 && k !== d1), false)
}

const UNARY: Record<string, [(x: number) => number, ((x: number) => number) | null, string, boolean?]> = {
  floor: [Math.floor, null, 'FloorBackward0', true],
  ceil: [Math.ceil, null, 'CeilBackward0', true],
  trunc: [Math.trunc, null, 'TruncBackward0', true],
  sign: [Math.sign, null, 'SignBackward0', true],
  frac: [(x) => x - Math.trunc(x), () => 1, 'FracBackward0'],
  log2: [Math.log2, (x) => 1 / (x * Math.LN2), 'Log2Backward0'],
  log10: [Math.log10, (x) => 1 / (x * Math.LN10), 'Log10Backward0'],
  log1p: [Math.log1p, (x) => 1 / (1 + x), 'Log1PBackward0'],
  expm1: [Math.expm1, Math.exp, 'Expm1Backward0'],
  exp2: [(x) => 2 ** x, (x) => 2 ** x * Math.LN2, 'Exp2Backward0'],
  tan: [Math.tan, (x) => 1 / Math.cos(x) ** 2, 'TanBackward0'],
  asin: [Math.asin, (x) => 1 / Math.sqrt(1 - x * x), 'AsinBackward0'],
  acos: [Math.acos, (x) => -1 / Math.sqrt(1 - x * x), 'AcosBackward0'],
  atan: [Math.atan, (x) => 1 / (1 + x * x), 'AtanBackward0'],
  sinh: [Math.sinh, Math.cosh, 'SinhBackward0'],
  cosh: [Math.cosh, Math.sinh, 'CoshBackward0'],
  erf: [C.erf, (x) => (2 / Math.sqrt(Math.PI)) * Math.exp(-x * x), 'ErfBackward0'],
  erfc: [C.erfc, (x) => (-2 / Math.sqrt(Math.PI)) * Math.exp(-x * x), 'ErfcBackward0'],
  reciprocal: [(x) => 1 / x, (x) => -1 / (x * x), 'ReciprocalBackward0'],
  rsqrt: [(x) => 1 / Math.sqrt(x), (x) => -0.5 / x ** 1.5, 'RsqrtBackward0'],
}

const twoOp = (op: string): TMethod => ['other', (c) => {
  const out = objOf(c.t).binop(op, need(c, 'other'), false, c.h)
  if (!out) throw c.h.err('TypeError', `${c.name}(): argument 'other' must be a Tensor or a number`)
  return out
}]

const METHODS: Record<string, TMethod> = {
  ...Object.fromEntries(Object.entries(UNARY).map(([name, [f, df, grad, keepInt]]): [string, TMethod] => [name, ['', (c) => done(c, C.mapT(c.E, c.t, f, df, grad, keepInt), 'elementwise')]])),
  round: ['decimals', (c) => {
    const d = intArg(c, 'decimals', 0)
    return done(c, C.mapT(c.E, c.t, (v) => roundHalfEvenT(v, d), null, 'RoundBackward0', true), 'elementwise')
  }],
  neg: ['', (c) => done(c, C.unary(c.E, 'neg', c.t), 'elementwise')],
  negative: ['', (c) => done(c, C.unary(c.E, 'neg', c.t), 'elementwise')],
  remainder: twoOp('%'),
  // C's fmod: the remainder takes the sign of the dividend
  fmod: ['other', (c) => {
    const o = operand(c.h, need(c, 'other'), c.t.dtype)
    if (!o) throw c.h.err('TypeError', "fmod(): argument 'other' must be a Tensor or a number")
    return done(c, C.map2T(c.E, c.t, o.t, (x, y) => x % y, () => 1, (x, y) => -Math.trunc(x / y), C.promote(c.t.dtype, o.t.dtype, false, o.weak), 'FmodBackward0'), 'elementwise')
  }],
  floor_divide: twoOp('//'), true_divide: twoOp('/'), divide: ['other rounding_mode', (c) => c.h.call(c.h.getAttr(c.self, 'div'), [need(c, 'other'), c.a.rounding_mode ?? py.NONE])],
  multiply: twoOp('*'), subtract: ['other alpha', (c) => c.h.call(c.h.getAttr(c.self, 'sub'), [need(c, 'other')], c.a.alpha ? { alpha: c.a.alpha } : {})],
  atan2: ['other', (c) => {
    const o = tArg(c, 'other')
    const dtype = C.promote(c.t.dtype, o.dtype)
    return done(c, C.map2T(c.E, c.t, o, Math.atan2, (y, x) => x / (x * x + y * y), (y, x) => -y / (x * x + y * y), isFloatT(dtype) ? dtype : 'float32', 'Atan2Backward0'), 'elementwise')
  }],
  clamp: ['min max', (c) => clampV(c, c.a.min, c.a.max)],
  clip: ['min max', (c) => clampV(c, c.a.min, c.a.max)],
  clamp_min: ['min', (c) => clampV(c, need(c, 'min'), undefined)],
  clamp_max: ['max', (c) => clampV(c, undefined, need(c, 'max'))],
  lerp: ['end weight', (c) => {
    const { h, E, t } = c
    const end = tArg(c, 'end')
    if (!isFloatT(t.dtype)) throw h.err('RuntimeError', `"lerp_kernel_scalar" not implemented for '${C.typeLabel(t.dtype)}'`)
    const w = need(c, 'weight')
    const wt = asTT(w)
    const span = C.binary(E, '-', end, t)
    const out = wt ? C.binary(E, '+', t, C.binary(E, '*', wt, span))
      : h.num(w) < 0.5 && h.num(w) > -0.5 ? C.binary(E, '+', t, C.binary(E, '*', span, TT.scalar(h.num(w), t.dtype), false, true))
        : C.binary(E, '-', end, C.binary(E, '*', span, TT.scalar(1 - h.num(w), t.dtype), false, true))
    if (out.fn) out.fn.name = wt ? 'LerpBackward1' : 'LerpBackward0'
    return done(c, out, 'elementwise')
  }],

  // ---- comparison
  equal: ['other', (c) => {
    const o = tArg(c, 'other')
    if (o.shape.join() !== c.t.shape.join()) return py.bool(false)
    const bv = o.values()
    return py.bool(c.t.values().every((v, i) => v === bv[i]))
  }],
  allclose: ['other rtol atol equal_nan', (c) => py.bool(closeness(c).values().every((v) => v !== 0))],
  isclose: ['other rtol atol equal_nan', (c) => done(c, closeness(c), 'elementwise')],
  isnan: ['', (c) => boolMap(c, Number.isNaN)],
  isinf: ['', (c) => boolMap(c, (v) => v === Infinity || v === -Infinity)],
  isfinite: ['', (c) => boolMap(c, Number.isFinite)],
  isposinf: ['', (c) => boolMap(c, (v) => v === Infinity)],
  isneginf: ['', (c) => boolMap(c, (v) => v === -Infinity)],
  logical_not: ['', (c) => boolMap(c, (v) => v === 0)],
  logical_and: ['other', (c) => logicalOp(c, (x, y) => x && y)],
  logical_or: ['other', (c) => logicalOp(c, (x, y) => x || y)],
  logical_xor: ['other', (c) => logicalOp(c, (x, y) => x !== y)],
  is_floating_point: ['', (c) => py.bool(isFloatT(c.t.dtype))],
  is_complex: ['', () => py.bool(false)],

  // ---- reductions
  all: countMethod((nz, total) => (nz === total ? 1 : 0), (t) => (t.dtype === 'uint8' ? 'uint8' : 'bool')),
  any: countMethod((nz) => (nz > 0 ? 1 : 0), (t) => (t.dtype === 'uint8' ? 'uint8' : 'bool')),
  count_nonzero: ['dim', (c) => {
    const dims = dimsArg(c.h, c.a.dim)
    return done(c, C.countReduce(c.t, dims, false, 'int64', (nz) => nz), 'reduce', reduceAxis(c, dims))
  }],
  prod: ['dim keepdim dtype', (c) => {
    const dim = optDim(c.h, c.a.dim)
    return done(c, C.prodT(c.E, withDtype(c), dim, flag(c, 'keepdim')), 'reduce', dim === null ? null : C.normDim(dim, c.t.ndim))
  }],
  median: ['dim keepdim', (c) => {
    const { t, E, h } = c
    const dim = optDim(h, c.a.dim)
    if (dim === null) {
      if (!t.size) return wrap(TT.scalar(NaN, t.dtype))
      return done(c, C.sharedValueT(E, t, C.lowerMedianAt(t.values()), 'MedianBackward0'), 'reduce')
    }
    if (!(t.shape[C.normDim(dim, t.ndim)] ?? 1)) throw h.err('IndexError', 'median() cannot compute median for a dimension of size 0')
    return picked(c, 'median', dim, C.pickAlong(E, t, dim, flag(c, 'keepdim'), C.lowerMedianAt, 'MedianBackward1'))
  }],
  mode: ['dim keepdim', (c) => {
    if (!c.t.size) throw c.h.err('IndexError', 'mode(): Expected reduction dim to be specified for input.numel() == 0.')
    return picked(c, 'mode', intArg(c, 'dim', -1), C.pickAlong(c.E, c.t, intArg(c, 'dim', -1), flag(c, 'keepdim'), C.modeAt, 'ModeBackward0'))
  }],
  kthvalue: ['k dim keepdim', (c) => picked(c, 'kthvalue', intArg(c, 'dim', -1), C.pickAlong(c.E, c.t, intArg(c, 'dim', -1), flag(c, 'keepdim'), C.kthAt(intArg(c, 'k')), 'KthvalueBackward0'))],
  amax: ['dim keepdim', (c) => amaxMin(c, false)],
  amin: ['dim keepdim', (c) => amaxMin(c, true)],
  norm: ['p dim keepdim dtype', (c) => {
    const { h, E } = c
    const p = c.a.p
    const t = withDtype(c)
    const dims = dimsArg(h, c.a.dim)
    if (p?.k === 'str' && p.v === 'nuc') throw h.err('NotImplementedError', "torch.norm(p='nuc') needs a singular value decomposition, which this sandbox does not have")
    if (p?.k === 'str' && p.v !== 'fro') throw h.err('ValueError', `Unknown norm type ${p.v}`)
    const ord = !p || p.k === 'none' || p.k === 'str' ? 2 : h.num(p)
    return done(c, C.vectorNorm(E, t, ord, dims, flag(c, 'keepdim')), 'reduce', reduceAxis(c, dims))
  }],
  trace: ['', (c) => {
    const { h, E, t } = c
    if (t.ndim !== 2) throw h.err('RuntimeError', `expected a matrix, but got tensor with dim ${t.ndim}`)
    const out = C.reduceSum(E, diagonalOf(E, t, 0, 0, 1, 'DiagonalBackward0'), null, false, false)
    if (out.fn) out.fn.name = 'TraceBackward0'
    return done(c, out, 'linalg')
  }],

  inverse: ['', (c) => done(c, C.invT(c.E, c.t), 'linalg')],
  det: ['', (c) => done(c, C.detT(c.E, c.t), 'linalg')],
  matrix_power: ['n', (c) => c.h.call(c.h.getAttr(c.h.module('torch.linalg'), 'matrix_power'), [c.self, need(c, 'n')])],

  // ---- cumulative and sorted
  cumsum: ['dim dtype', (c) => done(c, C.cumT(c.E, withDtype(c), intArg(c, 'dim'), false), 'scan', C.normDim(intArg(c, 'dim'), c.t.ndim))],
  cumprod: ['dim dtype', (c) => done(c, C.cumT(c.E, withDtype(c), intArg(c, 'dim'), true), 'scan', C.normDim(intArg(c, 'dim'), c.t.ndim))],
  sort: ['dim descending stable', (c) => {
    const dim = intArg(c, 'dim', -1)
    const r = C.sortT(c.E, c.t, dim, flag(c, 'descending'))
    trace(c.h, 'Tensor.sort', 'sort', [{ label: label(c.h, c.self, 'x'), t: c.t }], r.values, C.normDim(dim, c.t.ndim), r.src)
    return results('sort', r)
  }],
  argsort: ['dim descending stable', (c) => {
    const dim = intArg(c, 'dim', -1)
    const r = C.sortT({ grad: false } as Engine, c.t, dim, flag(c, 'descending'))
    return done(c, r.indices, 'sort', C.normDim(dim, c.t.ndim), r.src)
  }],
  topk: ['k dim largest sorted', (c) => {
    const dim = intArg(c, 'dim', -1)
    const r = C.topkT(c.E, c.t, intArg(c, 'k'), dim, flag(c, 'largest', true))
    trace(c.h, 'Tensor.topk', 'sort', [{ label: label(c.h, c.self, 'x'), t: c.t }], r.values, C.normDim(dim, c.t.ndim), r.src)
    return results('topk', r)
  }],
  unique: ['sorted return_inverse return_counts dim', uniqueOf],
  bincount: ['weights minlength', (c) => {
    const { t, h } = c
    if (t.ndim > 1 || isFloatT(t.dtype) || t.values().some((v) => v < 0)) throw h.err('RuntimeError', 'bincount only supports 1-d non-negative integral inputs.')
    const iv = t.values()
    const w = c.a.weights && c.a.weights.k !== 'none' ? tArg(c, 'weights') : null
    if (w && w.size !== t.size) throw h.err('RuntimeError', 'the input and weights should have the same length.')
    const wv = w ? w.values() : null
    const n = Math.max(intArg(c, 'minlength', 0), iv.length ? iv.reduce((m, x) => Math.max(m, x), 0) + 1 : 0)
    const out = new Array<number>(n).fill(0)
    iv.forEach((v, i) => (out[v] += wv ? wv[i] : 1))
    const res = TT.of(out, [n], w ? (w.dtype === 'float32' ? 'float32' : 'float64') : 'int64')
    trace(h, 'Tensor.bincount', 'group', [{ label: label(h, c.self, 'x'), t }], res, null, out.map((_, v) => iv.flatMap((x, i): [number, number][] => (x === v ? [[0, i]] : []))))
    return wrap(res)
  }],

  // ---- selection
  gather: ['dim index', (c) => {
    const r = C.gatherT(c.E, c.t, intArg(c, 'dim'), tArg(c, 'index'))
    return done(c, r.out, 'move', null, r.src)
  }],
  scatter: ['dim index src value reduce', (c) => {
    if (c.a.reduce) throw c.h.err('NotImplementedError', 'scatter(): the reduce= argument is not available in this sandbox')
    const src = need(c, c.a.src ? 'src' : 'value')
    return done(c, C.scatterT(c.E, c.t, intArg(c, 'dim'), tArg(c, 'index'), asTT(src) ?? c.h.num(src), false), 'group')
  }],
  scatter_add: ['dim index src', (c) => done(c, C.scatterT(c.E, c.t, intArg(c, 'dim'), tArg(c, 'index'), tArg(c, 'src'), true), 'group')],
  index_select: ['dim index', (c) => {
    const { t, h } = c
    const d = C.normDim(intArg(c, 'dim'), t.ndim)
    const index = tArg(c, 'index')
    if (index.ndim > 1) throw h.err('RuntimeError', 'index_select(): Index is supposed to be a vector')
    if (isFloatT(index.dtype)) throw h.err('RuntimeError', 'index_select(): Expected dtype int32 or int64 for index')
    const iv = index.values()
    if (iv.some((i) => i < 0 || i >= (t.shape[d] ?? 1))) throw h.err('IndexError', 'index out of range in self')
    return byMaps(c, identityMaps(t).map((m, k) => (k === d ? iv : m)), 'IndexSelectBackward0')
  }],
  index_fill: ['dim index value', (c) => {
    const { t, E } = c
    const d = C.normDim(intArg(c, 'dim'), t.ndim)
    const iv = tArg(c, 'index').values()
    const pick = t.shape.map((n, k) => Array.from({ length: n }, (_, i) => (k === d && !iv.includes(i) ? -1 : i)))
    const src = C.crossSrc(t.shape, pick)
    const mask = TT.of(Array.from(src, (s) => (s >= 0 ? 1 : 0)), t.shape, 'bool')
    return done(c, C.whereT(E, mask, TT.scalar(c.h.num(need(c, 'value')), t.dtype), t, t.dtype, 'IndexFillBackward0'), 'elementwise')
  }],
  masked_select: ['mask', (c) => {
    const { t, E, h } = c
    const mask = tArg(c, 'mask')
    if (mask.dtype !== 'bool') throw h.err('RuntimeError', `masked_select: expected BoolTensor for mask`)
    const shape = broadcastShapes([t.shape, mask.shape])
    const it = C.bcastIndex(t.shape, shape)
    const im = C.bcastIndex(mask.shape, shape)
    const mv = mask.values()
    const src = Array.from(it).filter((_, i) => mv[im[i]])
    return done(c, C.takeT(E, t, src, [src.length], 'MaskedSelectBackward0'), 'move', null, src)
  }],
  masked_fill: ['mask value', (c) => {
    const { t, E, h } = c
    const mask = tArg(c, 'mask')
    if (mask.dtype !== 'bool') throw h.err('RuntimeError', `masked_fill_ only supports boolean masks, but got mask with dtype ${mask.dtype}`)
    const v = need(c, 'value')
    const vt = asTT(v)
    if (vt && vt.ndim) throw h.err('RuntimeError', `value tensor of a masked_fill must be 0-dimensional, but got one with ${vt.ndim} dimensions.`)
    return done(c, C.whereT(E, mask, vt ?? TT.scalar(h.num(v), t.dtype), t, t.dtype, 'MaskedFillBackward0'), 'elementwise')
  }],
  where: ['condition other', (c) => {
    const o = operand(c.h, need(c, 'other'), c.t.dtype)
    if (!o) throw c.h.err('TypeError', 'where(): argument other must be a Tensor or a number')
    return done(c, C.whereT(c.E, tArg(c, 'condition'), c.t, o.t, C.promote(c.t.dtype, o.t.dtype, false, o.weak)), 'elementwise')
  }],
  nonzero: ['as_tuple', (c) => {
    const { t, h } = c
    const st = cStrides(t.shape)
    const hits: number[] = []
    t.values().forEach((v, f) => { if (v !== 0) hits.push(f) })
    const coords = hits.map((f) => t.shape.map((_, d) => Math.floor(f / st[d]) % t.shape[d]))
    if (flag(c, 'as_tuple')) return py.tuple((t.ndim ? t.shape : [1]).map((_, d) => wrap(TT.of(coords.map((p) => p[d] ?? 0), [hits.length], 'int64'))))
    const out = TT.of(coords.flat(), [hits.length, t.ndim], 'int64')
    trace(h, 'Tensor.nonzero', 'group', [{ label: label(h, c.self, 'x'), t }], out, null, hits.flatMap((f) => t.shape.map(() => f)))
    return wrap(out)
  }],
  take: ['index', (c) => {
    const index = tArg(c, 'index')
    const n = c.t.size
    const iv = index.values()
    if (iv.some((i) => i < -n || i >= n)) throw c.h.err('RuntimeError', `out of range: tried to access index ${iv.find((i) => i < -n || i >= n)} on a tensor of ${n} elements.`)
    const src = iv.map((i) => (i < 0 ? i + n : i))
    return done(c, C.takeT(c.E, c.t, src, index.shape, 'TakeBackward0'), 'move', null, src)
  }],
  take_along_dim: ['indices dim', (c) => {
    const index = tArg(c, 'indices')
    const dim = optDim(c.h, c.a.dim)
    if (dim === null) return c.h.call(c.h.getAttr(c.self, 'take'), [need(c, 'indices')])
    const r = C.gatherT(c.E, c.t, dim, index)
    return done(c, r.out, 'move', null, r.src)
  }],

  // ---- shape
  view_as: ['other', (c) => done(c, viewOrReshape(c.E, c.t, tArg(c, 'other').shape, true), 'move')],
  expand: ['*size', (c) => done(c, expandTo(c.h, c.E, c.t, sizes(c, 'size')), 'move', null, expandSource(c, sizes(c, 'size')))],
  expand_as: ['other', (c) => done(c, expandTo(c.h, c.E, c.t, tArg(c, 'other').shape), 'move', null, expandSource(c, tArg(c, 'other').shape))],
  broadcast_to: ['size', (c) => done(c, expandTo(c.h, c.E, c.t, sizes(c, 'size')), 'move', null, expandSource(c, sizes(c, 'size')))],
  repeat: ['*repeats', (c) => {
    const reps = sizes(c, 'repeats')
    if (reps.length < c.t.ndim) throw c.h.err('RuntimeError', 'Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor')
    return repeated(c, reps)
  }],
  tile: ['*dims', (c) => {
    const dims = sizes(c, 'dims')
    return repeated(c, [...new Array<number>(Math.max(0, c.t.ndim - dims.length)).fill(1), ...dims])
  }],
  repeat_interleave: ['repeats dim output_size', (c) => {
    const { t, E, h } = c
    const r = need(c, 'repeats')
    const rt = asTT(r)
    const dim = optDim(h, c.a.dim)
    const n = dim === null ? t.size : t.shape[C.normDim(dim, t.ndim)]
    if (rt && rt.ndim > 1) throw h.err('RuntimeError', 'repeats must be 0-dim or 1-dim tensor')
    const counts = rt ? (rt.size === 1 ? new Array<number>(n).fill(rt.values()[0]) : rt.values()) : new Array<number>(n).fill(h.toInt(r))
    if (counts.length !== n) throw h.err('RuntimeError', `repeats must have the same size as input along dim, but got repeats.size(0) = ${counts.length} and input.size(${dim ?? 0}) = ${n}`)
    const positions = counts.flatMap((m, i) => new Array<number>(m).fill(i))
    if (dim === null) return done(c, C.takeT(E, t, positions, [positions.length], 'IndexSelectBackward0'), 'move', null, positions)
    const d = C.normDim(dim, t.ndim)
    return byMaps(c, identityMaps(t).map((m, k) => (k === d ? positions : m)), 'IndexSelectBackward0')
  }],
  narrow: ['dim start length', (c) => {
    const { t, h, E } = c
    const d = C.normDim(intArg(c, 'dim'), t.ndim)
    const len = intArg(c, 'length')
    const n = t.shape[d]
    const start = intArg(c, 'start') < 0 ? intArg(c, 'start') + n : intArg(c, 'start')
    if (start < 0 || start > n) throw h.err('IndexError', `Dimension out of range (expected to be in range of [${-n}, ${n}], but got ${intArg(c, 'start')})`)
    if (start + len > n) throw h.err('RuntimeError', `start (${start}) + length (${len}) exceeds dimension size (${n}).`)
    return done(c, C.narrowT(E, t, d, start, len), 'move')
  }],
  split: ['split_size dim split_size_or_sections', (c) => {
    const { t, h } = c
    needDims(c, 'split')
    const spec = c.a.split_size ?? need(c, 'split_size_or_sections')
    const d = C.normDim(intArg(c, 'dim', 0), t.ndim)
    const n = t.shape[d]
    if (spec.k === 'int') {
      const size = spec.v
      if (size < 0) throw h.err('RuntimeError', `split expects split_size be non-negative, but got split_size=${size}`)
      if (size === 0 && n !== 0) throw h.err('RuntimeError', `split_size can only be 0 if dimension size is 0, but got dimension size of ${n}`)
      const count = size === 0 ? 1 : Math.max(1, Math.ceil(n / size))
      return parts(c, d, Array.from({ length: count }, (_, i) => (i === count - 1 ? n - size * (count - 1) : size)), 'SplitBackward0')
    }
    const lens = h.iterate(spec).map((v) => h.toInt(v))
    if (lens.reduce((s, x) => s + x, 0) !== n) throw h.err('RuntimeError', `split_with_sizes expects split_sizes to sum exactly to ${n} (input tensor's size at dimension ${d}), but got split_sizes=[${lens.join(', ')}]`)
    return parts(c, d, lens, 'SplitWithSizesBackward0')
  }],
  chunk: ['chunks dim', (c) => {
    const { t, h } = c
    needDims(c, 'chunk')
    const chunks = intArg(c, 'chunks')
    if (chunks <= 0) throw h.err('RuntimeError', `chunk expects \`chunks\` to be greater than 0, got: ${chunks}`)
    const d = C.normDim(intArg(c, 'dim', 0), t.ndim)
    const n = t.shape[d]
    const size = Math.ceil(n / chunks)
    const count = size ? Math.ceil(n / size) : chunks
    return parts(c, d, Array.from({ length: count }, (_, i) => (i === count - 1 ? n - size * (count - 1) : size)), 'SplitBackward0')
  }],
  unbind: ['dim', (c) => {
    needDims(c, 'unbind')
    const d = C.normDim(intArg(c, 'dim', 0), c.t.ndim)
    return py.tuple(Array.from({ length: c.t.shape[d] }, (_, i) => {
      const f = (x: NDArray) => x.view(x.shape.filter((_s, k) => k !== d), x.strides.filter((_s, k) => k !== d), x.offset + i * x.strides[d])
      return done(c, C.viewT(c.E, c.t, f, 'UnbindBackward0'), 'move', d, shown(c) ? C.viewSource(c.t, f) : undefined)
    }))
  }],
  flip: ['*dims', (c) => flipped(c, sizes(c, 'dims'))],
  fliplr: ['', (c) => {
    if (c.t.ndim < 2) throw c.h.err('RuntimeError', 'Input must be >= 2-d.')
    return flipped(c, [1])
  }],
  flipud: ['', (c) => {
    if (c.t.ndim < 1) throw c.h.err('RuntimeError', 'Input must be >= 1-d.')
    return flipped(c, [0])
  }],
  roll: ['shifts dims', (c) => {
    const { t, h, E } = c
    const shifts = sizes(c, 'shifts')
    const dv = c.a.dims
    if (!dv || dv.k === 'none' || (dv.k === 'tuple' && !dv.items.length)) {
      if (shifts.length !== 1) throw h.err('RuntimeError', `shifts and dimensions must align. shifts: ${shifts.length}, dims:0`)
      // no dims: the flattened tensor rolls and keeps its shape
      const n = t.size
      const src = Array.from({ length: n }, (_, i) => ((i - shifts[0]) % n + n) % n)
      return done(c, C.takeT(E, t, src, t.shape, 'RollBackward0'), 'move', null, src)
    }
    const dims = dimsArg(h, dv)!.map((d) => C.normDim(d, t.ndim))
    if (dims.length !== shifts.length) throw h.err('RuntimeError', `shifts and dimensions must align. shifts: ${shifts.length}, dims:${dims.length}`)
    const maps = identityMaps(t)
    dims.forEach((d, k) => { maps[d] = maps[d].map((p) => (((maps[d][p] - shifts[k]) % t.shape[d]) + t.shape[d]) % t.shape[d]) })
    return byMaps(c, maps, 'RollBackward0')
  }],
  diag: ['diagonal', (c) => {
    const { t, h, E } = c
    const k = intArg(c, 'diagonal', 0)
    if (t.ndim === 1) {
      const n = t.size + Math.abs(k)
      const src = new Array<number>(n * n).fill(-1)
      for (let i = 0; i < t.size; i++) src[k >= 0 ? i * n + i + k : (i - k) * n + i] = i
      return done(c, C.takeT(E, t, src, [n, n], 'DiagBackward0'), 'move', null, src)
    }
    if (t.ndim !== 2) throw h.err('RuntimeError', `diag(): Supports 1D or 2D tensors. Got ${t.ndim}D`)
    const f = diagonalView(t.shape, k, 0, 1)
    const src = C.viewSource(t, f)
    return done(c, C.takeT(E, t, src, [src.length], 'DiagBackward0'), 'move', null, src)
  }],
  diagonal: ['offset dim1 dim2', (c) => {
    if (c.t.ndim < 2) throw c.h.err('RuntimeError', 'diagonal dimensions cannot be identical 0, 1')
    return done(c, diagonalOf(c.E, c.t, intArg(c, 'offset', 0), intArg(c, 'dim1', 0), intArg(c, 'dim2', 1), 'DiagonalBackward0'), 'move')
  }],
  tril: ['diagonal', (c) => done(c, C.triT(c.E, c.t, intArg(c, 'diagonal', 0), false), 'elementwise')],
  triu: ['diagonal', (c) => done(c, C.triT(c.E, c.t, intArg(c, 'diagonal', 0), true), 'elementwise')],
  swapaxes: ['axis0 axis1', (c) => done(c, C.transposeT(c.E, c.t, intArg(c, 'axis0'), intArg(c, 'axis1')), 'move')],
  swapdims: ['dim0 dim1', (c) => done(c, C.transposeT(c.E, c.t, intArg(c, 'dim0'), intArg(c, 'dim1')), 'move')],
  movedim: ['source destination', (c) => moved(c, need(c, 'source'), need(c, 'destination'))],
  moveaxis: ['source destination', (c) => moved(c, need(c, 'source'), need(c, 'destination'))],
  ravel: ['', (c) => done(c, C.flattenT(c.E, c.t, 0, -1), 'move')],
  unflatten: ['dim sizes', (c) => {
    const d = C.normDim(intArg(c, 'dim'), c.t.ndim)
    const shape = [...c.t.shape.slice(0, d), ...toDims(c.h, [need(c, 'sizes')]), ...c.t.shape.slice(d + 1)]
    const out = C.reshapeT(c.E, c.t, shape, false)
    if (out.fn) out.fn.name = 'UnflattenBackward0'
    return done(c, out, 'move')
  }],
  outer: ['vec2', (c) => {
    const { t, h, E } = c
    const o = tArg(c, 'vec2')
    if (t.ndim !== 1) throw h.err('RuntimeError', `outer: Expected 1-D argument self, but got ${t.ndim}-D`)
    if (o.ndim !== 1) throw h.err('RuntimeError', `outer: Expected 1-D argument vec2, but got ${o.ndim}-D`)
    const out = C.binary(E, '*', C.reshapeT(E, t, [t.size, 1], false), o)
    trace(h, 'Tensor.outer', 'matmul', [{ label: label(h, c.self, 'a'), t }, { label: label(h, c.a.vec2!, 'b'), t: o }], out)
    return wrap(out)
  }],
  inner: ['other', (c) => {
    const { t, h, E } = c
    const o = tArg(c, 'other')
    if (!t.ndim || !o.ndim) return wrap(C.binary(E, '*', t, o))
    sameDtype(h, t, o)
    const k = t.shape[t.ndim - 1]
    if (k !== o.shape[o.ndim - 1]) throw h.err('RuntimeError', `inner() the last dimension must match on both input tensors but got shapes [${t.shape.join(', ')}] and [${o.shape.join(', ')}].`)
    if (t.ndim === 1 && o.ndim === 1) return wrap(C.matmul(E, t, o))
    const product = C.matmul(E, C.reshapeT(E, t, [-1, k], false), C.transposeT(E, C.reshapeT(E, o, [-1, k], false), 0, 1))
    return done(c, C.reshapeT(E, product, [...t.shape.slice(0, -1), ...o.shape.slice(0, -1)], false), 'move')
  }],
  ger: ['vec2', (c) => METHODS.outer[1](c)],

  // ---- new tensors like this one
  type_as: ['other', (c) => wrap(C.toDtype(c.E, c.t, tArg(c, 'other').dtype))],
  new_zeros: ['*size dtype requires_grad device', (c) => newLike(c, () => 0)],
  new_ones: ['*size dtype requires_grad device', (c) => newLike(c, () => 1)],
  new_empty: ['*size dtype requires_grad device', (c) => newLike(c, () => 0)],
  new_full: ['size fill_value dtype requires_grad device', (c) => {
    const v = c.h.num(need(c, 'fill_value'))
    return newLike(c, () => v)
  }],
  new_tensor: ['data dtype requires_grad device', (c) => {
    const out = tensorFrom(c.h, need(c, 'data'), dtypeArg(c.h, c.a.dtype) ?? c.t.dtype)
    out.requiresGrad = flag(c, 'requires_grad')
    return wrap(out)
  }],

  // ---- fill with random numbers, in place
  normal_: ['mean std generator', (c) => {
    if (!isFloatT(c.t.dtype)) throw c.h.err('RuntimeError', `"normal_kernel_cpu" not implemented for '${C.typeLabel(c.t.dtype)}'`)
    const [mean, std] = [numOr(c, 'mean', 0), numOr(c, 'std', 1)]
    if (std < 0) throw c.h.err('RuntimeError', `normal expects std >= 0.0, but found std ${repr(py.float(std))}`)
    const rng = rngOf(c.h, c.a.generator)
    return filledIn(c, () => mean + std * rng.normal(), 'NormalBackward0')
  }],
  uniform_: ['from to generator', (c) => {
    if (!isFloatT(c.t.dtype)) throw c.h.err('RuntimeError', `"check_uniform_bounds" not implemented for '${C.typeLabel(c.t.dtype)}'`)
    const [lo, hi] = [numOr(c, 'from', 0), numOr(c, 'to', 1)]
    const rng = rngOf(c.h, c.a.generator)
    return filledIn(c, () => lo + (hi - lo) * rng.next(), 'UniformBackward0')
  }],
  exponential_: ['lambd generator', (c) => {
    const rate = numOr(c, 'lambd', 1)
    const rng = rngOf(c.h, c.a.generator)
    return filledIn(c, () => -Math.log(1 - rng.next()) / rate, 'ExponentialBackward0')
  }],
  bernoulli_: ['p generator', (c) => {
    const p = numOr(c, 'p', 0.5)
    const rng = rngOf(c.h, c.a.generator)
    return filledIn(c, () => (rng.next() < p ? 1 : 0), 'BernoulliBackward0')
  }],
  random_: ['from to generator', (c) => {
    const rng = rngOf(c.h, c.a.generator)
    const [lo, hi] = c.a.to ? [numOr(c, 'from', 0), c.h.num(c.a.to)] : [0, numOr(c, 'from', 2 ** 31)]
    return filledIn(c, () => lo + Math.floor(rng.next() * (hi - lo)), 'RandomBackward0')
  }],
  fill_diagonal_: ['fill_value wrap', (c) => {
    const { t, h } = c
    if (t.ndim < 2) throw h.err('RuntimeError', 'dimensions must larger than 1')
    if (t.ndim > 2 && new Set(t.shape).size > 1) throw h.err('RuntimeError', 'all dimensions of input must be of equal length')
    if (flag(c, 'wrap')) throw h.err('NotImplementedError', 'fill_diagonal_(): wrap=True is not available in this sandbox')
    const v = h.num(need(c, 'fill_value'))
    const step = cStrides(t.shape).reduce((s, x) => s + x, 0)
    const n = Math.min(...t.shape)
    const vals = t.values()
    for (let i = 0; i < n; i++) vals[i * step] = v
    return objOf(t).rebase(h, () => TT.of(vals, t.shape, t.dtype))
  }],
  detach_: ['', (c) => {
    c.t.fn = null
    c.t.requiresGrad = false
    return c.self
  }],
}

const INPLACE = new Set(['abs', 'acos', 'add', 'asin', 'atan', 'ceil', 'clamp', 'clamp_max', 'clamp_min', 'clip', 'cos', 'cosh', 'div', 'divide', 'erf', 'erfc', 'exp', 'exp2', 'expm1', 'floor', 'floor_divide', 'fmod', 'frac', 'index_fill', 'lerp', 'log', 'log10', 'log1p', 'log2', 'logical_not', 'masked_fill', 'mul', 'multiply', 'neg', 'negative', 'pow', 'reciprocal', 'relu', 'remainder', 'round', 'rsqrt', 'scatter', 'scatter_add', 'sigmoid', 'sign', 'sin', 'sinh', 'sqrt', 'square', 'sub', 'subtract', 'tan', 'tanh', 'tril', 'triu', 'true_divide', 'trunc'])
const INPLACE_VIEW = new Set(['t', 'transpose', 'swapaxes', 'swapdims', 'squeeze', 'unsqueeze'])

const numOr = (c: TCall, key: string, d: number) => {
  const v = c.a[key]
  return v && v.k !== 'none' ? c.h.num(v) : d
}

const filledIn = (c: TCall, value: () => number, gradName: string): Value => objOf(c.t).rebase(c.h, (old) => C.mapT(c.E, old, value, null, gradName, true))

const newLike = (c: TCall, value: () => number): Value => {
  const dims = sizes(c, 'size')
  const out = TT.of(Array.from({ length: prod(dims) }, value), dims, dtypeArg(c.h, c.a.dtype) ?? c.t.dtype)
  out.requiresGrad = flag(c, 'requires_grad')
  return wrap(out)
}

const moved = (c: TCall, source: Value, destination: Value): Value => {
  const { t, h, E } = c
  const src = dimsArg(h, source)!.map((d) => C.normDim(d, t.ndim))
  const dst = dimsArg(h, destination)!.map((d) => C.normDim(d, t.ndim))
  if (src.length !== dst.length) throw h.err('RuntimeError', `movedim: Invalid source or destination dims: source (${src.length} dims) should contain the same number of dims as destination (${dst.length} dims)`)
  const rest = t.shape.map((_, d) => d).filter((d) => !src.includes(d))
  const perm = new Array<number>(t.ndim).fill(-1)
  src.forEach((s, k) => (perm[dst[k]] = s))
  for (let i = 0, r = 0; i < t.ndim; i++) if (perm[i] < 0) perm[i] = rest[r++]
  return done(c, C.permuteT(E, t, perm, 'MovedimBackward0'), 'move')
}

/** which element of x each element of x.expand(shape) repeats, when the step list will show it */
const expandSource = (c: TCall, target: number[]): number[] | undefined => (!shown(c) || prod(target.map((d) => Math.abs(d))) > SNAP_MAX ? undefined : C.viewSource(c.t, (x) => {
  const lead = target.length - x.ndim
  const shape = target.map((d, k) => (d === -1 ? x.shape[k - lead] : d))
  return x.view(shape, shape.map((d, k) => (k < lead || (x.shape[k - lead] === 1 && d !== 1) ? 0 : x.strides[k - lead])), x.offset)
}))

const diagonalView = (shape: number[], offset: number, dim1: number, dim2: number) => (x: NDArray): NDArray => {
  const [a, b] = [dim1, dim2].map((d) => C.normDim(d, shape.length))
  const len = Math.max(0, Math.min(offset >= 0 ? shape[b] - offset : shape[a] + offset, offset >= 0 ? shape[a] : shape[b]))
  const start = offset >= 0 ? offset * x.strides[b] : -offset * x.strides[a]
  const keep = (_: number, k: number) => k !== a && k !== b
  return x.view([...shape.filter(keep), len], [...x.strides.filter(keep), x.strides[a] + x.strides[b]], x.offset + start)
}

const diagonalOf = (E: Engine, t: TT, offset: number, dim1: number, dim2: number, name: string): TT => C.viewT(E, t, diagonalView(t.shape, offset, dim1, dim2), name)

const amaxMin = (c: TCall, min: boolean): Value => {
  const dims = dimsArg(c.h, c.a.dim)
  const out = C.extremeT(c.E, c.t, dims && dims.length ? dims : null, flag(c, 'keepdim'), min, min ? 'AminBackward0' : 'AmaxBackward0')
  return done(c, out, 'reduce', reduceAxis(c, dims))
}

const logicalOp = (c: TCall, f: (x: boolean, y: boolean) => boolean): Value => {
  const o = tArg(c, 'other')
  const shape = broadcastShapes([c.t.shape, o.shape])
  const [ia, ib] = [C.bcastIndex(c.t.shape, shape), C.bcastIndex(o.shape, shape)]
  const [av, bv] = [c.t.values(), o.values()]
  return done(c, TT.of(Array.from({ length: prod(shape) }, (_, i) => (f(av[ia[i]] !== 0, bv[ib[i]] !== 0) ? 1 : 0)), shape, 'bool'), 'elementwise')
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
  if (model) for (const [n, p] of namedParams(model, '')) names.set(p, n)
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
  children(): [string, Value][]
  ownParams(): [string, TT][]
  /** [name, tensor, whether state_dict includes it] */
  ownBuffers(): [string, TT, boolean][]
  extraRepr(): string
}

const MODULE_STATE = 'torch.module'

const moduleOf = (v: Value): ModuleLike | null => {
  if (v.k === 'obj' && v.o instanceof Layer) return v.o
  if (v.k === 'inst' && v.state[MODULE_STATE]) return instModule(v)
  return null
}

const instModule = (inst: InstValue): ModuleLike => {
  const st = inst.state[MODULE_STATE] as { training: boolean; buffers: Map<string, boolean> }
  return {
    kind: inst.cls.name,
    get training() { return st.training },
    set training(b: boolean) { st.training = b },
    children: () => [...inst.attrs].filter(([, v]) => moduleOf(v) !== null),
    ownParams: () => [...inst.attrs].flatMap(([n, v]): [string, TT][] => {
      const t = asTT(v)
      return t && t.isParam ? [[n, t]] : []
    }),
    ownBuffers: () => [...st.buffers].flatMap(([n, keep]): [string, TT, boolean][] => {
      const t = asTT(inst.attrs.get(n) ?? py.NONE)
      return t ? [[n, t, keep]] : []
    }),
    extraRepr: () => '',
  }
}

export function namedParams(v: Value, prefix: string): [string, TT][] {
  const m = moduleOf(v)
  if (!m) return []
  const own = m.ownParams().map(([n, t]): [string, TT] => [prefix + n, t])
  return [...own, ...m.children().flatMap(([n, c]) => namedParams(c, `${prefix}${n}.`))]
}

const namedBuffers = (v: Value, prefix: string): [string, TT, boolean][] => {
  const m = moduleOf(v)
  if (!m) return []
  return [...m.ownBuffers().map(([n, t, keep]): [string, TT, boolean] => [prefix + n, t, keep]), ...m.children().flatMap(([n, c]) => namedBuffers(c, `${prefix}${n}.`))]
}

/** what state_dict holds: each module's parameters, its buffers, then its children */
const stateItems = (v: Value, prefix: string): [string, TT][] => {
  const m = moduleOf(v)
  if (!m) return []
  return [
    ...m.ownParams().map(([n, t]): [string, TT] => [prefix + n, t]),
    ...m.ownBuffers().filter(([, , keep]) => keep).map(([n, t]): [string, TT] => [prefix + n, t]),
    ...m.children().flatMap(([n, c]) => stateItems(c, `${prefix}${n}.`)),
  ]
}

/** parameters() skips a tensor it has already listed (tied weights) */
const distinctParams = (items: [string, TT][]) => {
  const seen = new Set<TT>()
  return items.filter(([, t]) => !seen.has(t) && seen.add(t))
}

const allModules = (v: Value, prefix: string, out: Map<object, string>) => {
  const m = moduleOf(v)
  if (!m) return
  out.set(v.k === 'obj' ? v.o : v, prefix.replace(/\.$/, ''))
  for (const [n, c] of m.children()) allModules(c, `${prefix}${n}.`, out)
}

const moduleRepr = (v: Value): string => {
  const m = moduleOf(v)!
  const indented = (s: string) => s.replace(/\n/g, '\n  ')
  // ModuleList folds runs of identical layers: (0-2): 3 x Linear(…)
  const kids = m.children().map(([n, c]) => [n, moduleRepr(c)])
  const runs: [string, string, number][] = []
  for (const [n, r] of kids) {
    const last = runs[runs.length - 1]
    if (v.k === 'obj' && v.o instanceof ModuleList && last && last[1] === r) last[2]++
    else runs.push([n, r, 1])
  }
  const lines = runs.map(([n, r, count]) => indented(count > 1 ? `(${n}-${Number(n) + count - 1}): ${count} x ${r}` : `(${n}): ${r}`))
  const extra = m.extraRepr()
  if (!lines.length) return `${m.kind}(${extra})`
  return `${m.kind}(\n  ${[...(extra ? [extra] : []), ...lines].join('\n  ')}\n)`
}

const setTraining = (v: Value, b: boolean) => {
  const m = moduleOf(v)
  if (!m) return
  m.training = b
  for (const [, c] of m.children()) setTraining(c, b)
}

/** model.double() and friends: floating-point parameters and buffers change dtype in place */
const castFloats = (v: Value, d: TDType) => {
  const m = moduleOf(v)
  if (!m) return
  for (const [, t] of [...m.ownParams(), ...m.ownBuffers()]) {
    if (!isFloatT(t.dtype) || t.dtype === d) continue
    t.a = TT.of(t.values(), t.shape, d).a
    t.dtype = d
  }
  for (const [, c] of m.children()) castFloats(c, d)
}

/** what load_state_dict returns: its repr is the familiar <All keys matched successfully> */
class IncompatibleKeys extends PyObj {
  readonly cls = '_IncompatibleKeys'
  missing: string[]
  unexpected: string[]
  constructor(missing: string[], unexpected: string[]) {
    super()
    this.missing = missing
    this.unexpected = unexpected
  }
  private names(xs: string[]) { return py.list(xs.map(py.str)) }
  repr() { return this.missing.length || this.unexpected.length ? `_IncompatibleKeys(missing_keys=${repr(this.names(this.missing))}, unexpected_keys=${repr(this.names(this.unexpected))})` : '<All keys matched successfully>' }
  getAttr(name: string) { return name === 'missing_keys' ? this.names(this.missing) : name === 'unexpected_keys' ? this.names(this.unexpected) : undefined }
}

/** methods every module answers to */
const moduleMethod = (h: Host, self: Value, name: string): Value | undefined => {
  const m = moduleOf(self)!
  const fn = (call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })
  switch (name) {
    case 'parameters': return fn((args, kw) => {
      const recurse = flagOf(h, kw.recurse ?? args[0], true)
      return py.list(distinctParams(recurse ? namedParams(self, '') : m.ownParams()).map(([, t]) => wrap(t)))
    })
    case 'named_parameters': return fn((args, kw) => {
      const prefix = kw.prefix ?? args[0]
      const pre = prefix ? `${h.str(prefix)}.` : ''
      const recurse = flagOf(h, kw.recurse ?? args[1], true)
      return py.list(distinctParams(recurse ? namedParams(self, pre) : m.ownParams().map(([n, t]): [string, TT] => [pre + n, t])).map(([n, t]) => py.tuple([py.str(n), wrap(t)])))
    })
    case 'buffers': return fn(() => py.list(namedBuffers(self, '').map(([, t]) => wrap(t))))
    case 'named_buffers': return fn(() => py.list(namedBuffers(self, '').map(([n, t]) => py.tuple([py.str(n), wrap(t)]))))
    case 'register_buffer': return fn((args, kw) => {
      const bname = h.str(args[0] ?? kw.name)
      const tv = args[1] ?? kw.tensor
      const keep = flagOf(h, kw.persistent ?? args[2], true)
      if (self.k === 'inst') {
        ;(self.state[MODULE_STATE] as { buffers: Map<string, boolean> }).buffers.set(bname, keep)
        self.attrs.set(bname, tv)
      } else if (self.k === 'obj' && self.o instanceof Layer) {
        const t = asTT(tv)
        if (!t) throw h.err('TypeError', `cannot assign '${typeName(tv)}' object to buffer '${bname}' (torch Tensor or None required)`)
        self.o.buffers.set(bname, { t, keep })
      }
      return py.NONE
    })
    case 'register_parameter': return fn((args, kw) => {
      if (self.k !== 'inst') throw h.err('NotImplementedError', 'register_parameter() on a built-in layer is not available in this sandbox')
      self.attrs.set(h.str(args[0] ?? kw.name), args[1] ?? kw.param)
      return py.NONE
    })
    case 'add_module': case 'register_module': return fn((args, kw) => {
      const cname = h.str(args[0] ?? kw.name)
      const child = args[1] ?? kw.module
      if (self.k === 'inst') self.attrs.set(cname, child)
      else if (self.k === 'obj' && self.o instanceof Sequential) self.o.add(cname, child)
      else if (self.k === 'obj' && self.o instanceof ModuleDict) self.o.items.set(cname, child)
      else throw h.err('NotImplementedError', `add_module() on ${m.kind} is not available in this sandbox`)
      return py.NONE
    })
    case 'requires_grad_': return fn((args, kw) => {
      const b = flagOf(h, kw.requires_grad ?? args[0], true)
      for (const [, t] of namedParams(self, '')) t.requiresGrad = b
      return self
    })
    case 'children': return fn(() => py.list(m.children().map(([, c]) => c)))
    case 'named_children': return fn(() => py.list(m.children().map(([n, c]) => py.tuple([py.str(n), c]))))
    case 'modules': case 'named_modules': return fn(() => {
      const out = new Map<object, string>()
      allModules(self, '', out)
      const val = (o: object) => (o instanceof PyObj ? py.obj(o) : (o as Value))
      return py.list([...out].map(([o, n]) => (name === 'modules' ? val(o) : py.tuple([py.str(n), val(o)]))))
    })
    case 'train': return fn((args, kw) => { setTraining(self, flagOf(h, kw.mode ?? args[0], true)); return self })
    case 'eval': return fn(() => { setTraining(self, false); return self })
    case 'training': return py.bool(m.training)
    case 'zero_grad': return fn((args, kw) => {
      const toNone = flagOf(h, kw.set_to_none ?? args[0], true)
      for (const [, t] of namedParams(self, '')) t.grad = toNone || !t.grad ? null : TT.of(t.grad.values().map(() => 0), t.shape, t.dtype)
      return py.NONE
    })
    case 'to': return fn((args, kw) => {
      const dev = kw.device ?? args.find((a) => a.k === 'str' || (a.k === 'obj' && a.o instanceof DeviceObj))
      if (dev && h.str(dev).startsWith('cuda')) throw h.err('AssertionError', 'Torch not compiled with CUDA enabled')
      const dt = kw.dtype ?? args.find((a) => a.k === 'obj' && a.o instanceof DtypeObj)
      const d = dtypeArg(h, dt)
      if (d && !isFloatT(d)) throw h.err('TypeError', `nn.Module.to only accepts floating point or complex dtypes, but got desired dtype=torch.${d}`)
      if (d) castFloats(self, d)
      return self
    })
    case 'cpu': return fn(() => self)
    case 'float': case 'double': return fn(() => { castFloats(self, name === 'float' ? 'float32' : 'float64'); return self })
    case 'half': return fn(() => { throw h.err('NotImplementedError', 'Module.half(): float16 is not available in this sandbox') })
    case 'cuda': return fn(() => { throw h.err('AssertionError', 'Torch not compiled with CUDA enabled') })
    case 'state_dict': return fn(() => ({ k: 'dict', d: PyDict.from(stateItems(self, '').map(([n, t]) => [py.str(n), wrap(C.detach(t))])) }))
    case 'load_state_dict': return fn((args, kw) => {
      const d = args[0] ?? kw.state_dict
      if (d?.k !== 'dict') throw h.err('TypeError', `Expected state_dict to be dict-like, got ${d ? typeName(d) : 'nothing'}.`)
      const strict = flagOf(h, kw.strict ?? args[1], true)
      const items = stateItems(self, '')
      const missing: string[] = []
      const errs: string[] = []
      for (const [n, t] of items) {
        const src = asTT(d.d.get(py.str(n)) ?? py.NONE)
        if (!src) missing.push(n)
        else if (src.shape.join() !== t.shape.join()) errs.push(`size mismatch for ${n}: copying a param with shape torch.Size([${src.shape.join(', ')}]) from checkpoint, the shape in current model is torch.Size([${t.shape.join(', ')}]).`)
        else C.writeInto(t, src.values())
      }
      const known = new Set(items.map(([n]) => n))
      const unexpected = d.d.keys().map((k) => h.str(k)).filter((k) => !known.has(k))
      const quoted = (xs: string[]) => xs.map((x) => `"${x}"`).join(', ')
      if (strict && unexpected.length) errs.unshift(`Unexpected key(s) in state_dict: ${quoted(unexpected)}. `)
      if (strict && missing.length) errs.unshift(`Missing key(s) in state_dict: ${quoted(missing)}. `)
      if (errs.length) throw h.err('RuntimeError', `Error(s) in loading state_dict for ${m.kind}:\n\t${errs.join('\n\t')}`)
      return py.obj(new IncompatibleKeys(missing, unexpected))
    })
    case 'apply': return fn((args) => {
      // children first, as Module.apply goes
      const visit = (v: Value) => {
        for (const [, c] of moduleOf(v)!.children()) visit(c)
        h.call(args[0], [v])
      }
      visit(self)
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
    allModules(self, '', names)
    flow = S.flow = { line: h.line, code: h.code, rows: [], depth: 0, names, model: moduleOf(self)!.kind, root: self.k === 'obj' ? self.o : self }
    // the model whose parameters name the autograd graph (a loss module has none)
    if (namedParams(self, '').length) S.lastModel = self
  }
  flow.depth++
  let out: Value
  try {
    out = run()
  } finally {
    flow.depth--
    if (outer) S.flow = null
  }
  // nn.LSTM and friends return (output, state)
  const ot = asTT(out.k === 'tuple' ? out.items[0] ?? py.NONE : out)
  if (leaf && input && ot && !(leaf instanceof Sequential) && !(leaf instanceof LossLayer)) {
    flow.rows.push({
      name: flow.names.get(leaf) || leaf.kind, layer: `${leaf.kind}(${leaf.extraRepr()})`, input: [...input.shape], output: [...ot.shape],
      params: leaf.ownParams().reduce((s, [, t]) => s + t.size, 0), note: leaf.flowNote(input, ot),
    })
  }
  if (outer && flow.rows.length && h.tracing) {
    const params = namedParams(self, '')
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
  buffers = new Map<string, { t: TT; keep: boolean }>()
  ownBuffers(): [string, TT, boolean][] { return [...this.buffers].map(([n, b]) => [n, b.t, b.keep]) }
  children(): [string, Value][] { return [] }
  flowNote(_i: TT, _o: TT): string | null { return null }
  abstract forward(args: Value[], kw: Kw, h: Host): Value
  repr() { return moduleRepr(py.obj(this)) }
  isa(name: string) { return name === 'Module' || name === this.kind }
  /** layer.weight = nn.Parameter(…), layer.bias = None, layer.training = False */
  setAttr(name: string, v: Value, h: Host): boolean {
    if (name === 'training') {
      this.training = h.truthy(v)
      return true
    }
    const self = this as unknown as { w?: TT | null; b?: TT | null }
    if ((name === 'weight' && 'w' in self) || (name === 'bias' && 'b' in self)) {
      const t = asTT(v)
      if (t && !t.isParam) throw h.err('TypeError', `cannot assign 'torch.${C.typeLabel(t.dtype)}Tensor' as parameter '${name}' (torch.nn.Parameter or None expected)`)
      if (!t && v.k !== 'none') throw h.err('TypeError', `cannot assign '${typeName(v)}' as parameter '${name}' (torch.nn.Parameter or None expected)`)
      if (name === 'weight') self.w = t
      else self.b = t
      return true
    }
    const b = this.buffers.get(name)
    const t = asTT(v)
    if (b && t) {
      b.t = t
      return true
    }
    return false
  }
  call(args: Value[], kw: Kw, h: Host): Value {
    return callModule(h, py.obj(this), () => this.forward(args, kw, h), this, asTT(args[0] ?? py.NONE))
  }
  getAttr(name: string, h: Host): Value | undefined {
    const p = this.ownParams().find(([n]) => n === name)
    if (p) return wrap(p[1])
    const b = this.buffers.get(name)
    if (b) return wrap(b.t)
    if (name === 'forward') return { k: 'fn', name, call: (args, kw) => this.forward(args, kw, h) }
    return this.attr(name, h) ?? moduleMethod(h, py.obj(this), name)
  }
  attr(_name: string, _h: Host): Value | undefined { return undefined }
}

const tensorArg = (h: Host, v: Value | undefined, what: string): TT => {
  const t = v ? asTT(v) : null
  if (!t) throw h.err('TypeError', `${what}: expected a Tensor as input, got ${v ? (v.k === 'obj' ? v.o.cls : v.k === 'array' ? 'numpy.ndarray (wrap it with torch.tensor / torch.from_numpy first)' : v.k) : 'nothing'}`)
  return t
}


class Linear extends Layer {
  readonly kind: string = 'Linear'
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
    sameDtype(h, x, this.w)
    return wrap(C.linear(stateOf(h).E, x, this.w, this.b))
  }
}

interface ConvOpts {
  stride: [number, number]
  padding: [number, number] | 'same' | 'valid'
  dilation: [number, number]
  groups: number
  paddingMode: C.PadMode
}

const CONV_TYPE: Record<TDType, string> = { float32: 'float', float64: 'double', int64: 'long int', int32: 'int', uint8: 'unsigned char', bool: 'bool' }

/**
 * conv1d runs as a conv2d of height 1; dilation spreads the kernel out with zeros; each group runs on its own channels
 * and the results are joined. All of it stays differentiable because every step is.
 */
const convolve = (h: Host, x0: TT, w0: TT, b: TT | null, o: ConvOpts, one: boolean): TT => {
  const E = stateOf(h).E
  const nd = one ? 1 : 2
  if (x0.ndim !== nd + 1 && x0.ndim !== nd + 2) throw h.err('RuntimeError', `Expected ${nd + 1}D (unbatched) or ${nd + 2}D (batched) input to conv${nd}d, but got input of size: [${x0.shape.join(', ')}]`)
  if (x0.dtype !== w0.dtype) throw h.err('RuntimeError', b ? `Input type (${CONV_TYPE[x0.dtype]}) and bias type (${CONV_TYPE[w0.dtype]}) should be the same` : `Input type (torch.${C.typeLabel(x0.dtype)}Tensor) and weight type (torch.${C.typeLabel(w0.dtype)}Tensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor`)
  const chanDim = x0.ndim - nd - 1
  if (x0.shape[chanDim] !== w0.shape[1] * o.groups) throw h.err('RuntimeError', `Given groups=${o.groups}, weight of size [${w0.shape.join(', ')}], expected input[${x0.shape.join(', ')}] to have ${w0.shape[1] * o.groups} channels, but got ${x0.shape[chanDim]} channels instead`)
  let x = one ? C.unsqueezeT(E, x0, -2) : x0
  let w = one ? C.unsqueezeT(E, w0, -2) : w0
  const [kh, kw] = w.shape.slice(-2)
  const [dh, dw] = o.dilation
  if (dh !== 1 || dw !== 1) {
    const spread = (k: number, d: number) => Array.from({ length: d * (k - 1) + 1 }, (_, i) => (i % d ? -1 : i / d))
    const maps = [[...Array(w.shape[0]).keys()], [...Array(w.shape[1]).keys()], spread(kh, dh), spread(kw, dw)]
    w = C.takeT(E, w, C.crossSrc(w.shape, maps), maps.map((m) => m.length), 'DilatedKernelBackward0')
  }
  const [ek, el] = w.shape.slice(-2)
  // padding='same' can need one more cell on the far side, and the other padding modes pad the input first
  const sym: [number, number] = o.padding === 'valid' ? [0, 0] : o.padding === 'same' ? [(ek - 1) >> 1, (el - 1) >> 1] : o.padding
  const far = o.padding === 'same' ? [ek - 1 - sym[0], el - 1 - sym[1]] : sym
  let p = sym
  if (o.paddingMode !== 'constant' || far[0] !== sym[0] || far[1] !== sym[1]) {
    x = C.padT(E, x, [sym[1], far[1], sym[0], far[0]], o.paddingMode, 0)
    p = [0, 0]
  }
  if (one && x.shape[x.ndim - 1] + 2 * p[1] < el) throw h.err('RuntimeError', `Calculated padded input size per channel: (${x.shape[x.ndim - 1] + 2 * p[1]}). Kernel size: (${el}). Kernel size can't be greater than actual input size`)
  const run = (xg: TT, wg: TT, bg: TT | null) => C.conv2d(E, xg, wg, bg, { stride: o.stride, padding: p })
  const cg = w0.shape[1]
  const og = w0.shape[0] / o.groups
  const part = (g: number) => run(C.narrowT(E, x, chanDim, g * cg, cg), C.narrowT(E, w, 0, g * og, og), b && C.narrowT(E, b, 0, g * og, og))
  let out = o.groups === 1 ? run(x, w, b) : C.catT(E, Array.from({ length: o.groups }, (_, g) => part(g)), chanDim)
  if (one) out = C.reshapeT(E, out, out.shape.filter((_, k) => k !== out.ndim - 2), false)
  if (out.fn) out.fn.name = 'ConvolutionBackward0'
  return out
}

class ConvLayer extends Layer {
  readonly kind: string
  one: boolean
  inC: number
  outC: number
  k: [number, number]
  o: ConvOpts
  w: TT
  b: TT | null
  constructor(h: Host, kind: string, inC: number, outC: number, k: [number, number], o: ConvOpts, bias: boolean) {
    super()
    this.kind = kind
    this.one = kind === 'Conv1d'
    this.inC = inC
    this.outC = outC
    this.k = k
    this.o = o
    const fan = (inC / o.groups) * k[0] * k[1]
    this.w = initParam(h, this.one ? [outC, inC / o.groups, k[1]] : [outC, inC / o.groups, k[0], k[1]], fan)
    this.b = bias ? initParam(h, [outC], fan) : null
  }
  private dims(p: [number, number]) { return this.one ? `(${p[1]},)` : tupleStr(p) }
  private tuple(p: [number, number]) { return py.tuple((this.one ? p.slice(1) : p).map(py.int)) }
  extraRepr() {
    const { padding, dilation, groups, stride, paddingMode } = this.o
    const pad = typeof padding === 'string' ? `, padding=${padding}` : padding[0] || padding[1] ? `, padding=${this.dims(padding)}` : ''
    const dil = dilation[0] !== 1 || dilation[1] !== 1 ? `, dilation=${this.dims(dilation)}` : ''
    const mode = paddingMode === 'constant' ? '' : `, padding_mode=${paddingMode}`
    return `${this.inC}, ${this.outC}, kernel_size=${this.dims(this.k)}, stride=${this.dims(stride)}${pad}${dil}${groups !== 1 ? `, groups=${groups}` : ''}${this.b ? '' : ', bias=False'}${mode}`
  }
  ownParams(): [string, TT][] { return this.b ? [['weight', this.w], ['bias', this.b]] : [['weight', this.w]] }
  attr(name: string) {
    if (name === 'in_channels') return py.int(this.inC)
    if (name === 'out_channels') return py.int(this.outC)
    if (name === 'kernel_size') return this.tuple(this.k)
    if (name === 'stride') return this.tuple(this.o.stride)
    if (name === 'padding') return typeof this.o.padding === 'string' ? py.str(this.o.padding) : this.tuple(this.o.padding)
    if (name === 'dilation') return this.tuple(this.o.dilation)
    if (name === 'groups') return py.int(this.o.groups)
    if (name === 'bias') return py.NONE
    return undefined
  }
  flowNote(i: TT, o: TT) {
    const a = this.one ? 1 : 0
    const [p, d, s, k] = [typeof this.o.padding === 'string' ? (this.o.padding === 'same' ? (this.k[a] - 1) >> 1 : 0) : this.o.padding[a], this.o.dilation[a], this.o.stride[a], this.k[a]]
    const n = i.shape[i.ndim - (this.one ? 1 : 2)]
    const span = d === 1 ? `${k}` : `${d}·(${k} − 1) − 1`
    const kernel = this.one ? `${k}·${this.inC}` : `${this.k[0]}·${this.k[1]}·${this.inC}`
    return `⌊(${n} + 2·${p} − ${span}) / ${s}⌋ + 1 = ${o.shape[o.ndim - (this.one ? 1 : 2)]} · params (${kernel} + ${this.b ? 1 : 0})·${this.outC}`
  }
  forward(args: Value[], _kw: Kw, h: Host) {
    return wrap(convolve(h, tensorArg(h, args[0], this.kind), this.w, this.b, this.o, this.one))
  }
}

class ConvTranspose2d extends Layer {
  readonly kind = 'ConvTranspose2d'
  inC: number
  outC: number
  k: [number, number]
  stride: [number, number]
  padding: [number, number]
  outPad: [number, number]
  w: TT
  b: TT | null
  constructor(h: Host, inC: number, outC: number, k: [number, number], stride: [number, number], padding: [number, number], outPad: [number, number], bias: boolean) {
    super()
    this.inC = inC
    this.outC = outC
    this.k = k
    this.stride = stride
    this.padding = padding
    this.outPad = outPad
    const fan = outC * k[0] * k[1]
    this.w = initParam(h, [inC, outC, k[0], k[1]], fan)
    this.b = bias ? initParam(h, [outC], fan) : null
  }
  extraRepr() {
    const on = (p: [number, number]) => p[0] !== 0 || p[1] !== 0
    return `${this.inC}, ${this.outC}, kernel_size=${tupleStr(this.k)}, stride=${tupleStr(this.stride)}${on(this.padding) ? `, padding=${tupleStr(this.padding)}` : ''}${on(this.outPad) ? `, output_padding=${tupleStr(this.outPad)}` : ''}${this.b ? '' : ', bias=False'}`
  }
  ownParams(): [string, TT][] { return this.b ? [['weight', this.w], ['bias', this.b]] : [['weight', this.w]] }
  flowNote(i: TT, o: TT) { return `(${i.shape[i.ndim - 2]} − 1)·${this.stride[0]} − 2·${this.padding[0]} + ${this.k[0]} + ${this.outPad[0]} = ${o.shape[o.ndim - 2]}` }
  forward(args: Value[], kw: Kw, h: Host) {
    if (kw.output_size ?? args[1]) throw h.err('NotImplementedError', 'ConvTranspose2d(): output_size= is not available in this sandbox: use output_padding')
    const x = tensorArg(h, args[0], this.kind)
    if (x.dtype !== this.w.dtype) throw h.err('RuntimeError', `Input type (${CONV_TYPE[x.dtype]}) and bias type (${CONV_TYPE[this.w.dtype]}) should be the same`)
    return wrap(C.convTranspose2d(stateOf(h).E, x, this.w, this.b, this.stride, this.padding, this.outPad))
  }
}

class Pool2d extends Layer {
  readonly kind: string
  k: [number, number]
  s: [number, number]
  p: [number, number]
  mode: 'max' | 'avg'
  /** kernel_size, stride and padding as the caller wrote them: repr keeps 2 and (2, 2) apart */
  shown: string
  constructor(mode: 'max' | 'avg', k: [number, number], s: [number, number], p: [number, number], shown: string) {
    super()
    this.mode = mode
    this.kind = mode === 'max' ? 'MaxPool2d' : 'AvgPool2d'
    this.k = k
    this.s = s
    this.p = p
    this.shown = shown
  }
  extraRepr() { return this.mode === 'max' ? `${this.shown}, dilation=1, ceil_mode=False` : this.shown }
  flowNote(i: TT, o: TT) { return `⌊(${i.shape[i.ndim - 2]} + 2·${this.p[0]} − ${this.k[0]}) / ${this.s[0]}⌋ + 1 = ${o.shape[o.ndim - 2]}` }
  forward(args: Value[], _kw: Kw, h: Host) {
    return wrap(C.pool2d(stateOf(h).E, tensorArg(h, args[0], this.kind), this.k, this.s, this.p, this.mode))
  }
}

/** a 1-D pool is a 2-D pool of height 1 */
const pool1d = (h: Host, x: TT, k: number, s: number, p: number, mode: 'max' | 'avg'): TT => {
  const E = stateOf(h).E
  if (x.ndim !== 2 && x.ndim !== 3) throw h.err('RuntimeError', `${mode}_pool1d: Expected 2D or 3D (batch mode) tensor for input, but got: [${x.shape.join(', ')}]`)
  const out = C.pool2d(E, C.unsqueezeT(E, x, -2), [1, k], [1, s], [0, p], mode)
  const res = C.squeezeT(E, out, -2)
  if (res.fn) res.fn.name = mode === 'max' ? 'SqueezeBackward1' : 'AvgPool1DBackward0'
  return res
}

class Pool1d extends Layer {
  readonly kind: string
  k: number
  s: number
  p: number
  mode: 'max' | 'avg'
  constructor(mode: 'max' | 'avg', k: number, s: number, p: number) {
    super()
    this.mode = mode
    this.kind = mode === 'max' ? 'MaxPool1d' : 'AvgPool1d'
    this.k = k
    this.s = s
    this.p = p
  }
  extraRepr() { return this.mode === 'max' ? `kernel_size=${this.k}, stride=${this.s}, padding=${this.p}, dilation=1, ceil_mode=False` : `kernel_size=(${this.k},), stride=(${this.s},), padding=(${this.p},)` }
  flowNote(i: TT, o: TT) { return `⌊(${i.shape[i.ndim - 1]} + 2·${this.p} − ${this.k}) / ${this.s}⌋ + 1 = ${o.shape[o.ndim - 1]}` }
  forward(args: Value[], _kw: Kw, h: Host) { return wrap(pool1d(h, tensorArg(h, args[0], this.kind), this.k, this.s, this.p, this.mode)) }
}

class Activation extends Layer {
  readonly kind: string
  fn: (x: TT, h: Host) => TT
  extra: string
  inPlace: boolean
  constructor(kind: string, fn: (x: TT, h: Host) => TT, extra = '', inPlace = false) {
    super()
    this.kind = kind
    this.fn = fn
    this.extra = extra
    this.inPlace = inPlace
  }
  extraRepr() { return this.extra }
  forward(args: Value[], _kw: Kw, h: Host) {
    const x = tensorArg(h, args[0], this.kind)
    return this.inPlace ? objOf(x).rebase(h, (old) => this.fn(old, h)) : wrap(this.fn(x, h))
  }
}

class DropoutLayer extends Layer {
  readonly kind: string
  p: number
  inPlace: boolean
  constructor(h: Host, kind: string, p: number, inPlace: boolean) {
    super()
    if (p < 0 || p > 1) throw h.err('ValueError', `dropout probability has to be between 0 and 1, but got ${repr(py.float(p))}`)
    this.kind = kind
    this.p = p
    this.inPlace = inPlace
  }
  extraRepr() { return `p=${this.p}, inplace=${this.inPlace ? 'True' : 'False'}` }
  forward(args: Value[], _kw: Kw, h: Host) {
    const S = stateOf(h)
    if (this.training && !S.E.grad) noteOnce(h, 'dropout_eval', 'warn')
    const drop = (x: TT) => (this.kind === 'Dropout2d' ? C.dropoutChannels : C.dropout)(S.E, x, this.p, this.training)
    const x = tensorArg(h, args[0], this.kind)
    return this.inPlace ? objOf(x).rebase(h, drop) : wrap(drop(x))
  }
}

class Sequential extends Layer {
  readonly kind: string = 'Sequential'
  items: Value[]
  /** child names; positions unless the model was built from a dict or add_module */
  names: string[]
  constructor(items: Value[], names?: string[]) {
    super()
    this.items = items
    this.names = names ?? items.map((_, i) => String(i))
  }
  add(name: string, m: Value) {
    this.items.push(m)
    this.names.push(name)
  }
  children(): [string, Value][] { return this.items.map((v, i) => [this.names[i], v]) }
  len() { return this.items.length }
  iter() { return this.items }
  private at(idx: Value, h: Host) {
    const i = h.toInt(idx)
    const n = this.items.length
    if (i < -n || i >= n) throw h.err('IndexError', `index ${i} is out of range`)
    return i < 0 ? i + n : i
  }
  getItem(idx: Value, h: Host): Value {
    if (idx.k === 'slice') {
      const part = (x: Value) => (x.k === 'none' ? null : h.toInt(x))
      const { start, step, len } = sliceIndices(part(idx.start), part(idx.stop), part(idx.step), this.items.length)
      const at = Array.from({ length: len }, (_, k) => start + k * step)
      return py.obj(new (this.constructor as typeof Sequential)(at.map((i) => this.items[i]), this instanceof ModuleList ? undefined : at.map((i) => this.names[i])))
    }
    return this.items[this.at(idx, h)]
  }
  setItem(idx: Value, v: Value, h: Host) { this.items[this.at(idx, h)] = v }
  forward(args: Value[], _kw: Kw, h: Host): Value {
    let x = args[0]
    for (const m of this.items) x = h.call(m, [x])
    return x
  }
  attr(name: string, h: Host): Value | undefined {
    const self = py.obj(this)
    if (name === 'append') return { k: 'fn', name, call: (args) => { this.add(String(this.items.length), args[0]); return self } }
    if (name === 'extend') return { k: 'fn', name, call: (args) => { for (const m of h.iterate(args[0])) this.add(String(this.items.length), m); return self } }
    if (name === 'insert') return { k: 'fn', name, call: (args) => {
      const n = this.items.length
      const i = h.toInt(args[0])
      this.items.splice(i < 0 ? i + n : i, 0, args[1])
      this.names = this.items.map((_, k) => String(k))
      return self
    } }
    const named = this.names.indexOf(name)
    return named >= 0 ? this.items[named] : undefined
  }
}

class ModuleList extends Sequential {
  readonly kind = 'ModuleList'
  forward(_a: Value[], _k: Kw, h: Host): Value { throw h.err('NotImplementedError', 'Module [ModuleList] is missing the required "forward" function') }
}

class ModuleDict extends Layer {
  readonly kind = 'ModuleDict'
  items = new Map<string, Value>()
  children(): [string, Value][] { return [...this.items] }
  len() { return this.items.size }
  iter() { return [...this.items.keys()].map(py.str) }
  contains(v: Value) { return v.k === 'str' && this.items.has(v.v) }
  getItem(idx: Value, h: Host) {
    const m = idx.k === 'str' ? this.items.get(idx.v) : undefined
    if (!m) throw h.err('KeyError', repr(idx))
    return m
  }
  setItem(idx: Value, v: Value, h: Host) { this.items.set(h.str(idx), v) }
  update(v: Value, h: Host) {
    const pairs = v.k === 'dict' ? v.d.items() : v.k === 'obj' && v.o instanceof ModuleDict ? [...v.o.items].map(([n, m]): [Value, Value] => [py.str(n), m]) : h.iterate(v).map((p) => h.iterate(p) as [Value, Value])
    for (const [n, m] of pairs) this.items.set(h.str(n), m)
  }
  forward(_a: Value[], _k: Kw, h: Host): Value { throw h.err('NotImplementedError', 'Module [ModuleDict] is missing the required "forward" function') }
  attr(name: string, h: Host): Value | undefined {
    const fn = (call: (args: Value[]) => Value): Value => ({ k: 'fn', name, call })
    switch (name) {
      case 'keys': return fn(() => py.list([...this.items.keys()].map(py.str)))
      case 'values': return fn(() => py.list([...this.items.values()]))
      case 'items': return fn(() => py.list([...this.items].map(([n, m]) => py.tuple([py.str(n), m]))))
      case 'clear': return fn(() => { this.items.clear(); return py.NONE })
      case 'pop': return fn((args) => {
        const m = this.getItem(args[0], h)
        this.items.delete(h.str(args[0]))
        return m
      })
      case 'update': return fn((args) => { this.update(args[0], h); return py.NONE })
    }
    return this.items.get(name)
  }
}

class BatchNorm extends Layer {
  readonly kind: string
  num: number
  eps: number
  momentum: number | null
  affine: boolean
  track: boolean
  w: TT | null
  b: TT | null
  /** the input's allowed dimensions and how the error names them */
  ndims: number[]
  constructor(kind: string, ndims: number[], num: number, eps: number, momentum: number | null, affine: boolean, track: boolean) {
    super()
    this.kind = kind
    this.ndims = ndims
    this.num = num
    this.eps = eps
    this.momentum = momentum
    this.affine = affine
    this.track = track
    const param = (v: number) => {
      const t = TT.of(new Array<number>(num).fill(v), [num], 'float32')
      t.requiresGrad = true
      t.isParam = true
      return t
    }
    this.w = affine ? param(1) : null
    this.b = affine ? param(0) : null
    if (track) {
      this.buffers.set('running_mean', { t: TT.of(new Array<number>(num).fill(0), [num], 'float32'), keep: true })
      this.buffers.set('running_var', { t: TT.of(new Array<number>(num).fill(1), [num], 'float32'), keep: true })
      this.buffers.set('num_batches_tracked', { t: TT.scalar(0, 'int64'), keep: true })
    }
  }
  extraRepr() { return `${this.num}, eps=${repr(py.float(this.eps))}, momentum=${this.momentum === null ? 'None' : repr(py.float(this.momentum))}, affine=${this.affine ? 'True' : 'False'}, track_running_stats=${this.track ? 'True' : 'False'}` }
  ownParams(): [string, TT][] { return this.w && this.b ? [['weight', this.w], ['bias', this.b]] : [] }
  attr(name: string) {
    if (name === 'num_features') return py.int(this.num)
    if (name === 'eps') return py.float(this.eps)
    if (name === 'momentum') return this.momentum === null ? py.NONE : py.float(this.momentum)
    if (name === 'affine') return py.bool(this.affine)
    if (name === 'track_running_stats') return py.bool(this.track)
    return undefined
  }
  forward(args: Value[], _kw: Kw, h: Host) {
    const x = tensorArg(h, args[0], this.kind)
    const E = stateOf(h).E
    if (!this.ndims.includes(x.ndim)) throw h.err('ValueError', `expected ${this.ndims.map((n) => `${n}D`).join(' or ')} input (got ${x.ndim}D input)`)
    const running = this.buffers.get('running_mean')?.t
    const runVar = this.buffers.get('running_var')?.t
    const count = this.buffers.get('num_batches_tracked')?.t
    if (running && x.shape[1] !== this.num) throw h.err('RuntimeError', `running_mean should contain ${x.shape[1]} elements not ${this.num}`)
    if (this.w && x.shape[1] !== this.num) throw h.err('RuntimeError', `weight should contain ${x.shape[1]} elements not ${this.num}`)
    if (this.w && x.dtype !== this.w.dtype) throw h.err('RuntimeError', `expected scalar type ${C.typeLabel(x.dtype)} but found ${C.typeLabel(this.w.dtype)}`)
    if (this.training && x.size / x.shape[1] === 1) throw h.err('ValueError', `Expected more than 1 value per channel when training, got input of size torch.Size([${x.shape.join(', ')}])`)
    const fromBatch = this.training || !running || !runVar
    const r = C.batchNormT(E, x, this.w, this.b, fromBatch ? null : { mean: running.values(), vari: runVar.values() }, this.eps)
    if (this.training && running && runVar && count) {
      const seen = count.values()[0] + 1
      C.writeInto(count, [seen])
      const m = this.momentum ?? 1 / seen
      const mv = running.values()
      const vv = runVar.values()
      C.writeInto(running, mv.map((v, c) => (1 - m) * v + m * r.mean[c]))
      C.writeInto(runVar, vv.map((v, c) => (1 - m) * v + m * r.vari[c]))
    }
    return wrap(r.out)
  }
}

class LayerNorm extends Layer {
  readonly kind = 'LayerNorm'
  normShape: number[]
  eps: number
  affine: boolean
  w: TT | null
  b: TT | null
  constructor(normShape: number[], eps: number, affine: boolean, bias: boolean) {
    super()
    this.normShape = normShape
    this.eps = eps
    this.affine = affine
    const param = (v: number) => {
      const t = TT.of(new Array<number>(prod(normShape)).fill(v), normShape, 'float32')
      t.requiresGrad = true
      t.isParam = true
      return t
    }
    this.w = affine ? param(1) : null
    this.b = affine && bias ? param(0) : null
  }
  extraRepr() { return `(${this.normShape.join(', ')}${this.normShape.length === 1 ? ',' : ''}), eps=${repr(py.float(this.eps))}, elementwise_affine=${this.affine ? 'True' : 'False'}` }
  ownParams(): [string, TT][] { return [...(this.w ? [['weight', this.w] as [string, TT]] : []), ...(this.b ? [['bias', this.b] as [string, TT]] : [])] }
  forward(args: Value[], _kw: Kw, h: Host) {
    const x = tensorArg(h, args[0], this.kind)
    const E = stateOf(h).E
    const D = this.normShape.length
    if (x.ndim < D || this.normShape.some((s, k) => s !== x.shape[x.ndim - D + k])) throw h.err('RuntimeError', `Given normalized_shape=[${this.normShape.join(', ')}], expected input with shape [*, ${this.normShape.join(', ')}], but got input of size[${x.shape.join(', ')}]`)
    const dims = Array.from({ length: D }, (_, k) => x.ndim - D + k)
    const mean = C.reduceSum(E, x, dims, true, true)
    const centred = C.binary(E, '-', x, mean)
    const vari = C.reduceSum(E, C.unary(E, 'square', centred), dims, true, true)
    let out = C.binary(E, '/', centred, C.unary(E, 'sqrt', C.binary(E, '+', vari, TT.scalar(this.eps, x.dtype), false, true)))
    if (this.w) out = C.binary(E, '*', out, this.w)
    if (this.b) out = C.binary(E, '+', out, this.b)
    if (out.fn) out.fn.name = 'NativeLayerNormBackward0'
    return wrap(out)
  }
}

class GroupNorm extends Layer {
  readonly kind = 'GroupNorm'
  groups: number
  num: number
  eps: number
  affine: boolean
  w: TT | null
  b: TT | null
  constructor(h: Host, groups: number, num: number, eps: number, affine: boolean) {
    super()
    if (num % groups) throw h.err('ValueError', 'num_channels must be divisible by num_groups')
    this.groups = groups
    this.num = num
    this.eps = eps
    this.affine = affine
    const param = (v: number) => {
      const t = TT.of(new Array<number>(num).fill(v), [num], 'float32')
      t.requiresGrad = true
      t.isParam = true
      return t
    }
    this.w = affine ? param(1) : null
    this.b = affine ? param(0) : null
  }
  extraRepr() { return `${this.groups}, ${this.num}, eps=${repr(py.float(this.eps))}, affine=${this.affine ? 'True' : 'False'}` }
  ownParams(): [string, TT][] { return this.w && this.b ? [['weight', this.w], ['bias', this.b]] : [] }
  forward(args: Value[], _kw: Kw, h: Host) {
    const x = tensorArg(h, args[0], this.kind)
    const E = stateOf(h).E
    if (x.ndim < 2) throw h.err('RuntimeError', `Expected at least 2 dimensions for input tensor but received ${x.ndim}`)
    if (x.shape[1] !== this.num) throw h.err('RuntimeError', `Expected number of channels in input to be divisible by num_groups, but got input of shape [${x.shape.join(', ')}] and num_groups=${this.groups}`)
    // (N, G, C/G, …): each group of channels of each sample is normalised over its channels and positions
    const rest = x.shape.slice(2)
    const g = C.reshapeT(E, x, [x.shape[0], this.groups, this.num / this.groups, ...rest], false)
    const dims = Array.from({ length: rest.length + 1 }, (_, k) => k + 2)
    const mean = C.reduceSum(E, g, dims, true, true)
    const centred = C.binary(E, '-', g, mean)
    const vari = C.reduceSum(E, C.unary(E, 'square', centred), dims, true, true)
    let out = C.reshapeT(E, C.binary(E, '/', centred, C.unary(E, 'sqrt', C.binary(E, '+', vari, TT.scalar(this.eps, x.dtype), false, true))), x.shape, false)
    if (this.w && this.b) {
      const across = [1, this.num, ...rest.map(() => 1)]
      out = C.binary(E, '+', C.binary(E, '*', out, C.reshapeT(E, this.w, across, false)), C.reshapeT(E, this.b, across, false))
    }
    if (out.fn) out.fn.name = 'GroupNormBackward0'
    return wrap(out)
  }
}

class PReLU extends Layer {
  readonly kind = 'PReLU'
  w: TT
  constructor(num: number, init: number) {
    super()
    this.w = TT.of(new Array<number>(num).fill(init), [num], 'float32')
    this.w.requiresGrad = true
    this.w.isParam = true
  }
  extraRepr() { return `num_parameters=${this.w.size}` }
  ownParams(): [string, TT][] { return [['weight', this.w]] }
  forward(args: Value[], _kw: Kw, h: Host) {
    const x = tensorArg(h, args[0], this.kind)
    const E = stateOf(h).E
    if (this.w.size > 1 && x.shape[1] !== this.w.size) throw h.err('RuntimeError', `Mismatch of parameter numbers and input channel size. Found parameter numbers = ${this.w.size} and channel size = ${x.shape[1]}.`)
    const a = this.w.size > 1 ? C.reshapeT(E, this.w, x.shape.map((_, k) => (k === 1 ? this.w.size : 1)), false) : this.w
    const out = C.whereT(E, C.compare(x, TT.scalar(0, x.dtype), '>='), x, C.binary(E, '*', x, a), x.dtype, 'PreluKernelBackward0')
    return wrap(out)
  }
}

class Embedding extends Layer {
  readonly kind = 'Embedding'
  rows: number
  dim: number
  paddingIdx: number | null
  w: TT
  constructor(h: Host, rows: number, dim: number, paddingIdx: number | null) {
    super()
    this.rows = rows
    this.dim = dim
    this.paddingIdx = paddingIdx === null ? null : paddingIdx < 0 ? paddingIdx + rows : paddingIdx
    this.w = TT.of(Array.from({ length: rows * dim }, (_, f) => (Math.floor(f / dim) === this.paddingIdx ? 0 : h.rng.normal())), [rows, dim], 'float32')
    this.w.requiresGrad = true
    this.w.isParam = true
  }
  extraRepr() { return `${this.rows}, ${this.dim}${this.paddingIdx === null ? '' : `, padding_idx=${this.paddingIdx}`}` }
  ownParams(): [string, TT][] { return [['weight', this.w]] }
  attr(name: string) {
    if (name === 'num_embeddings') return py.int(this.rows)
    if (name === 'embedding_dim') return py.int(this.dim)
    return undefined
  }
  flowNote() { return `${this.rows} × ${this.dim} = ${this.rows * this.dim} params` }
  forward(args: Value[], _kw: Kw, h: Host) { return wrap(C.embeddingT(stateOf(h).E, this.w, tensorArg(h, args[0], this.kind), this.paddingIdx)) }
}

type RnnMode = 'RNN' | 'LSTM' | 'GRU'

const rnnParam = (shape: number[], hidden: number, h: Host): TT => {
  const bound = hidden > 0 ? 1 / Math.sqrt(hidden) : 0
  const t = TT.raw(Array.from({ length: prod(shape) }, () => Math.fround((h.rng.next() * 2 - 1) * bound)), shape, 'float32')
  t.requiresGrad = true
  t.isParam = true
  return t
}

/** one step of a recurrent cell on (N, in) and (N, hidden); the gate order inside the weights is PyTorch's */
const rnnStep = (E: Engine, mode: RnnMode, relu: boolean, H: number, x: TT, hx: TT, cx: TT | null, [wih, whh, bih, bhh]: (TT | null)[]): { h: TT; c: TT | null } => {
  const fromInput = C.linear(E, x, wih!, bih)
  const fromState = C.linear(E, hx, whh!, bhh)
  const chunk = (t: TT, k: number) => C.narrowT(E, t, 1, k * H, H, 'SplitBackward0')
  if (mode === 'RNN') {
    const sum = C.binary(E, '+', fromInput, fromState)
    return { h: relu ? C.unary(E, 'relu', sum) : C.unary(E, 'tanh', sum), c: null }
  }
  if (mode === 'GRU') {
    const reset = C.unary(E, 'sigmoid', C.binary(E, '+', chunk(fromInput, 0), chunk(fromState, 0)))
    const update = C.unary(E, 'sigmoid', C.binary(E, '+', chunk(fromInput, 1), chunk(fromState, 1)))
    const fresh = C.unary(E, 'tanh', C.binary(E, '+', chunk(fromInput, 2), C.binary(E, '*', reset, chunk(fromState, 2))))
    return { h: C.binary(E, '+', fresh, C.binary(E, '*', update, C.binary(E, '-', hx, fresh))), c: null }
  }
  const gates = C.binary(E, '+', fromInput, fromState)
  const input = C.unary(E, 'sigmoid', chunk(gates, 0))
  const forget = C.unary(E, 'sigmoid', chunk(gates, 1))
  const cell = C.unary(E, 'tanh', chunk(gates, 2))
  const output = C.unary(E, 'sigmoid', chunk(gates, 3))
  const c = C.binary(E, '+', C.binary(E, '*', forget, cx!), C.binary(E, '*', input, cell))
  return { h: C.binary(E, '*', output, C.unary(E, 'tanh', c)), c }
}

const GATES: Record<RnnMode, number> = { RNN: 1, LSTM: 4, GRU: 3 }

/** nn.RNN, nn.LSTM, nn.GRU */
class Recurrent extends Layer {
  readonly kind: string
  mode: RnnMode
  relu: boolean
  inSize: number
  hidden: number
  layers: number
  bias: boolean
  batchFirst: boolean
  dropoutP: number
  bi: boolean
  params: [string, TT][] = []
  weights: (TT | null)[][] = []
  constructor(h: Host, mode: RnnMode, relu: boolean, inSize: number, hidden: number, layers: number, bias: boolean, batchFirst: boolean, dropoutP: number, bi: boolean) {
    super()
    if (layers < 1) throw h.err('ValueError', 'num_layers must be greater than or equal to 1')
    this.kind = mode
    this.mode = mode
    this.relu = relu
    this.inSize = inSize
    this.hidden = hidden
    this.layers = layers
    this.bias = bias
    this.batchFirst = batchFirst
    this.dropoutP = dropoutP
    this.bi = bi
    const gate = GATES[mode] * hidden
    const dirs = bi ? 2 : 1
    for (let l = 0; l < layers; l++) {
      for (let d = 0; d < dirs; d++) {
        const suffix = `l${l}${d ? '_reverse' : ''}`
        const w = [rnnParam([gate, l ? hidden * dirs : inSize], hidden, h), rnnParam([gate, hidden], hidden, h), bias ? rnnParam([gate], hidden, h) : null, bias ? rnnParam([gate], hidden, h) : null]
        this.weights.push(w)
        this.params.push([`weight_ih_${suffix}`, w[0]!], [`weight_hh_${suffix}`, w[1]!])
        if (bias) this.params.push([`bias_ih_${suffix}`, w[2]!], [`bias_hh_${suffix}`, w[3]!])
      }
    }
  }
  extraRepr() {
    const yes = (b: boolean, s: string) => (b ? `, ${s}` : '')
    return `${this.inSize}, ${this.hidden}${yes(this.layers !== 1, `num_layers=${this.layers}`)}${yes(!this.bias, 'bias=False')}${yes(this.batchFirst, 'batch_first=True')}${yes(this.dropoutP !== 0, `dropout=${this.dropoutP}`)}${yes(this.bi, 'bidirectional=True')}`
  }
  ownParams() { return this.params }
  attr(name: string) {
    if (name === 'input_size') return py.int(this.inSize)
    if (name === 'hidden_size') return py.int(this.hidden)
    if (name === 'num_layers') return py.int(this.layers)
    if (name === 'batch_first') return py.bool(this.batchFirst)
    if (name === 'bidirectional') return py.bool(this.bi)
    return undefined
  }
  flowNote() { return `${this.params.reduce((s, [, t]) => s + t.size, 0)} params · state ${this.mode === 'LSTM' ? '(h, c)' : 'h'} of size ${this.hidden}` }
  private checkState(h: Host, t: TT, want: number[], what: string) {
    if (t.shape.join() !== want.join()) throw h.err('RuntimeError', `Expected hidden${what} size (${want.join(', ')}), got [${t.shape.join(', ')}]`)
  }
  forward(args: Value[], kw: Kw, h: Host) {
    const E = stateOf(h).E
    const input = tensorArg(h, args[0], this.kind)
    const state = kw.hx ?? args[1]
    if (input.ndim !== 2 && input.ndim !== 3) throw h.err('ValueError', `${this.kind}: Expected input to be 2D or 3D, got ${input.ndim}D instead`)
    if (input.dtype !== this.params[0][1].dtype) throw h.err('RuntimeError', `input must have the type torch.${this.params[0][1].dtype}, got type torch.${input.dtype}`)
    if (input.shape[input.ndim - 1] !== this.inSize) throw h.err('RuntimeError', `input.size(-1) must be equal to input_size. Expected ${this.inSize}, got ${input.shape[input.ndim - 1]}`)
    const unbatched = input.ndim === 2
    const dirs = this.bi ? 2 : 1
    const seqDim = this.batchFirst && !unbatched ? 1 : 0
    const L = input.shape[seqDim]
    const N = unbatched ? 1 : input.shape[1 - seqDim]
    // one (N, in) tensor per time step
    let xs = Array.from({ length: L }, (_, t) => {
      const step = C.narrowT(E, input, seqDim, t, 1, 'UnbindBackward0')
      return C.reshapeT(E, step, [N, this.inSize], false)
    })
    const want = [this.layers * dirs, N, this.hidden]
    const start = (t: TT | null, what: string): (i: number) => TT => {
      if (!t) return () => TT.of(new Array<number>(N * this.hidden).fill(0), [N, this.hidden], input.dtype)
      const three = unbatched ? C.unsqueezeT(E, t, 1) : t
      this.checkState(h, three, want, what)
      return (i) => C.reshapeT(E, C.narrowT(E, three, 0, i, 1, 'UnbindBackward0'), [N, this.hidden], false)
    }
    const sv = state && state.k !== 'none' ? state : null
    const dimsSeen = (t: TT | null) => {
      if (t && t.ndim !== (unbatched ? 2 : 3)) throw h.err('RuntimeError', `For ${unbatched ? 'unbatched 2-D' : 'batched 3-D'} input, hx should also be ${unbatched ? 2 : 3}-D but got ${t.ndim}-D tensor`)
      return t
    }
    const h0 = start(dimsSeen(this.mode === 'LSTM' ? (sv ? asTT(h.iterate(sv)[0]) : null) : sv ? asTT(sv) : null), this.mode === 'LSTM' ? '[0]' : '')
    const c0 = this.mode === 'LSTM' ? start(dimsSeen(sv ? asTT(h.iterate(sv)[1]) : null), '[1]') : null
    const hs: TT[] = []
    const cs: TT[] = []
    for (let l = 0; l < this.layers; l++) {
      const outs: TT[][] = []
      for (let d = 0; d < dirs; d++) {
        const k = l * dirs + d
        let hx = h0(k)
        let cx = c0 ? c0(k) : null
        const seq = d ? [...xs].reverse() : xs
        const ys = seq.map((x) => {
          const r = rnnStep(E, this.mode, this.relu, this.hidden, x, hx, cx, this.weights[k])
          hx = r.h
          cx = r.c
          return r.h
        })
        outs.push(d ? ys.reverse() : ys)
        hs.push(hx)
        if (cx) cs.push(cx)
      }
      const joined = this.bi ? outs[0].map((y, t) => C.catT(E, [y, outs[1][t]], 1)) : outs[0]
      xs = this.dropoutP > 0 && l < this.layers - 1 ? joined.map((y) => C.dropout(E, y, this.dropoutP, this.training)) : joined
    }
    // (L, N, D·H) back to the shape the caller used
    let output = C.catT(E, xs, 0, true)
    if (unbatched) output = C.squeezeT(E, output, 1)
    else if (this.batchFirst) output = C.transposeT(E, output, 0, 1)
    const packed = (ts: TT[]) => {
      const t = C.catT(E, ts, 0, true)
      return wrap(unbatched ? C.squeezeT(E, t, 1) : t)
    }
    return py.tuple([wrap(output), this.mode === 'LSTM' ? py.tuple([packed(hs), packed(cs)]) : packed(hs)])
  }
}

/** nn.RNNCell, nn.LSTMCell, nn.GRUCell: one time step */
class RecurrentCell extends Layer {
  readonly kind: string
  mode: RnnMode
  relu: boolean
  inSize: number
  hidden: number
  w: (TT | null)[]
  constructor(h: Host, kind: string, mode: RnnMode, relu: boolean, inSize: number, hidden: number, bias: boolean) {
    super()
    this.kind = kind
    this.mode = mode
    this.relu = relu
    this.inSize = inSize
    this.hidden = hidden
    const gate = GATES[mode] * hidden
    this.w = [rnnParam([gate, inSize], hidden, h), rnnParam([gate, hidden], hidden, h), bias ? rnnParam([gate], hidden, h) : null, bias ? rnnParam([gate], hidden, h) : null]
  }
  extraRepr() { return `${this.inSize}, ${this.hidden}${this.w[2] ? '' : ', bias=False'}` }
  ownParams(): [string, TT][] {
    const names = ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']
    return this.w.flatMap((t, i): [string, TT][] => (t ? [[names[i], t]] : []))
  }
  forward(args: Value[], kw: Kw, h: Host): Value {
    const E = stateOf(h).E
    const x0 = tensorArg(h, args[0], this.kind)
    if (x0.ndim !== 1 && x0.ndim !== 2) throw h.err('RuntimeError', `${this.kind}: Expected input to be 1D or 2D, got ${x0.ndim}D instead`)
    const unbatched = x0.ndim === 1
    const x = unbatched ? C.unsqueezeT(E, x0, 0) : x0
    const zeros = () => TT.of(new Array<number>(x.shape[0] * this.hidden).fill(0), [x.shape[0], this.hidden], x.dtype)
    const state = kw.hx ?? args[1]
    const sv = state && state.k !== 'none' ? state : null
    const given = (i: number) => {
      const t = sv ? (this.mode === 'LSTM' ? asTT(h.iterate(sv)[i]) : asTT(sv)) : null
      return t ? (unbatched ? C.unsqueezeT(E, t, 0) : t) : zeros()
    }
    const r = rnnStep(E, this.mode, this.relu, this.hidden, x, given(0), this.mode === 'LSTM' ? given(1) : null, this.w)
    const back = (t: TT) => wrap(unbatched ? C.squeezeT(E, t, 0) : t)
    return this.mode === 'LSTM' ? py.tuple([back(r.h), back(r.c!)]) : back(r.h)
  }
}

/** nn.MultiheadAttention: F.multi_head_attention_forward's steps with one projection per query, key and value */
class MultiheadAttention extends Layer {
  readonly kind = 'MultiheadAttention'
  embed: number
  heads: number
  dropoutP: number
  batchFirst: boolean
  inW: TT
  inB: TT | null
  outProj: Linear
  constructor(h: Host, embed: number, heads: number, dropoutP: number, bias: boolean, batchFirst: boolean) {
    super()
    if (embed % heads) throw h.err('AssertionError', 'embed_dim must be divisible by num_heads')
    this.embed = embed
    this.heads = heads
    this.dropoutP = dropoutP
    this.batchFirst = batchFirst
    // xavier_uniform_ for the projections, zeros for the biases, as _reset_parameters does
    const bound = Math.sqrt(3) * Math.sqrt(2 / (embed + 3 * embed))
    this.inW = TT.raw(Array.from({ length: 3 * embed * embed }, () => Math.fround((h.rng.next() * 2 - 1) * bound)), [3 * embed, embed], 'float32')
    this.inW.requiresGrad = true
    this.inW.isParam = true
    this.inB = bias ? TT.of(new Array<number>(3 * embed).fill(0), [3 * embed], 'float32') : null
    if (this.inB) {
      this.inB.requiresGrad = true
      this.inB.isParam = true
    }
    this.outProj = new Linear(h, embed, embed, bias)
    ;(this.outProj as { kind: string }).kind = 'NonDynamicallyQuantizableLinear'
    if (this.outProj.b) C.writeInto(this.outProj.b, new Array<number>(embed).fill(0))
  }
  ownParams(): [string, TT][] { return [['in_proj_weight', this.inW], ...(this.inB ? [['in_proj_bias', this.inB] as [string, TT]] : [])] }
  children(): [string, Value][] { return [['out_proj', py.obj(this.outProj)]] }
  forward(args: Value[], kw: Kw, h: Host): Value {
    const E = stateOf(h).E
    const a = bind(h, 'MultiheadAttention.forward', 'query key value key_padding_mask need_weights attn_mask average_attn_weights is_causal', args, kw)
    const q0 = tensorArg(h, a.query, 'query')
    const k0 = tensorArg(h, a.key, 'key')
    const v0 = tensorArg(h, a.value, 'value')
    const batched = q0.ndim === 3
    if (q0.ndim !== 2 && !batched) throw h.err('AssertionError', `query should be unbatched 2D or batched 3D tensor but received ${q0.ndim}-D query tensor`)
    if (q0.shape[q0.ndim - 1] !== this.embed) throw h.err('AssertionError', `was expecting embedding dimension of ${this.embed}, but got ${q0.shape[q0.ndim - 1]}`)
    // work on (L, N, E)
    const toSeq = (t: TT) => (!batched ? C.unsqueezeT(E, t, 1) : this.batchFirst ? C.transposeT(E, t, 0, 1) : t)
    const [q, k, v] = [q0, k0, v0].map(toSeq)
    const [tgt, bsz] = q.shape
    const src = k.shape[0]
    const E3 = this.embed
    const headDim = E3 / this.heads
    const project = (x: TT, i: number) => C.linear(E, x, C.narrowT(E, this.inW, 0, i * E3, E3), this.inB && C.narrowT(E, this.inB, 0, i * E3, E3))
    const perHead = (x: TT, i: number) => C.transposeT(E, C.reshapeT(E, project(x, i), [x.shape[0], bsz * this.heads, headDim], false), 0, 1)
    const Q = perHead(q, 0)
    const K = perHead(k, 1)
    const V = perHead(v, 2)
    let scores = C.matmul(E, C.binary(E, '/', Q, TT.scalar(Math.sqrt(headDim), q.dtype), false, true), C.transposeT(E, K, -2, -1))
    const addMask = (m: TT) => {
      if (m.dtype === 'bool') scores = C.whereT(E, m, TT.scalar(-Infinity, scores.dtype), scores, scores.dtype, 'MaskedFillBackward0')
      else scores = C.binary(E, '+', scores, m)
    }
    const attnMask = a.attn_mask && a.attn_mask.k !== 'none' ? tensorArg(h, a.attn_mask, 'attn_mask') : null
    if (attnMask) addMask(attnMask)
    const padMask = a.key_padding_mask && a.key_padding_mask.k !== 'none' ? tensorArg(h, a.key_padding_mask, 'key_padding_mask') : null
    if (padMask) {
      const perBatch = C.reshapeT(E, batched ? padMask : C.unsqueezeT(E, padMask, 0), [bsz, 1, 1, src], false)
      addMask(C.reshapeT(E, C.expandT(E, perBatch, [bsz, this.heads, 1, src]), [bsz * this.heads, 1, src], false))
    }
    const weights = C.softmax(E, scores, -1, false)
    const dropped = this.dropoutP > 0 ? C.dropout(E, weights, this.dropoutP, this.training) : weights
    const merged = C.reshapeT(E, C.transposeT(E, C.matmul(E, dropped, V), 0, 1), [tgt, bsz, E3], false)
    const output = C.linear(E, merged, this.outProj.w, this.outProj.b)
    const out = !batched ? C.squeezeT(E, output, 1) : this.batchFirst ? C.transposeT(E, output, 0, 1) : output
    if (a.need_weights && !h.truthy(a.need_weights)) return py.tuple([wrap(out), py.NONE])
    let w = C.reshapeT(E, weights, [bsz, this.heads, tgt, src], false)
    if (flagOf(h, a.average_attn_weights, true)) w = C.reduceSum(E, w, [1], false, true)
    return py.tuple([wrap(out), wrap(!batched ? C.squeezeT(E, w, 0) : w)])
  }
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

interface CeOpts { weight: TT | null; ignore: number; smoothing: number; red: C.Reduction }

/** cross entropy with its options, and with extra dimensions after the class dimension: (N, C, d…) against (N, d…) */
const ceFull = (h: Host, x: TT, y: TT, o: CeOpts): TT => {
  const E = stateOf(h).E
  if (o.smoothing < 0 || o.smoothing > 1) throw h.err('RuntimeError', `label_smoothing must be between 0.0 and 1.0. Got: ${o.smoothing}`)
  const probs = isFloatT(y.dtype)
  if (x.ndim > 2) {
    const toEnd = (t: TT) => C.reshapeT(E, C.permuteT(E, t, [0, ...t.shape.slice(2).map((_, k) => k + 2), 1]), [-1, x.shape[1]], false)
    const per = ceFull(h, toEnd(x), probs ? toEnd(y) : C.reshapeT(E, y, [-1], false), o)
    return o.red === 'none' ? C.reshapeT(E, per, [x.shape[0], ...x.shape.slice(2)], false) : per
  }
  if (!o.weight && o.ignore === -100 && !o.smoothing) return ceLoss(h, x, y, o.red)
  if (!probs && !y.ndim && x.ndim !== 1) throw h.err('RuntimeError', `Expected target size to match input`)
  if (o.weight && o.weight.size !== x.shape[x.ndim - 1]) throw h.err('RuntimeError', `weight tensor should be defined either for all ${x.shape[x.ndim - 1]} classes or no classes but got weight tensor of shape: [${o.weight.shape.join(', ')}]`)
  return C.weightedNll(E, C.softmax(E, x, x.ndim === 1 ? 0 : 1, true), y, o.weight ? o.weight.values() : null, o.ignore, o.smoothing, o.red)
}

/** nll_loss with the same options, on log-probabilities */
const nllFull = (h: Host, lp: TT, y: TT, o: CeOpts): TT => {
  const E = stateOf(h).E
  if (lp.ndim > 2) {
    const per = nllFull(h, C.reshapeT(E, C.permuteT(E, lp, [0, ...lp.shape.slice(2).map((_, k) => k + 2), 1]), [-1, lp.shape[1]], false), C.reshapeT(E, y, [-1], false), o)
    return o.red === 'none' ? C.reshapeT(E, per, [lp.shape[0], ...lp.shape.slice(2)], false) : per
  }
  if (!o.weight && o.ignore === -100) return C.nllLoss(E, lp, y, o.red)
  if (isFloatT(y.dtype)) throw h.err('RuntimeError', `expected scalar type Long but found ${C.typeLabel(y.dtype)}`)
  return C.weightedNll(E, lp, y, o.weight ? o.weight.values() : null, o.ignore, 0, o.red)
}

/** BCE with weight and pos_weight: l = −w · (p · y · log σ(z) + (1 − y) · log(1 − σ(z))) */
const bceWeighted = (h: Host, z: TT, y: TT, logits: boolean, weight: TT | null, posWeight: TT | null, red: C.Reduction): TT => {
  const E = stateOf(h).E
  if (!weight && !posWeight) return C.bce(E, z, y, logits, red)
  let per = C.bce(E, z, y, logits, 'none')
  if (posWeight) {
    // (1 − y) z + (1 + (p − 1) y) softplus(−z), with softplus(−z) = −log σ(z)
    const softplusNeg = C.mapT(E, z, (v) => Math.log1p(Math.exp(-Math.abs(v))) + Math.max(-v, 0), (v) => -1 / (1 + Math.exp(v)), 'SoftplusBackward0')
    const one = TT.scalar(1, z.dtype)
    const logWeight = C.binary(E, '+', one, C.binary(E, '*', C.binary(E, '-', posWeight, one), y))
    per = C.binary(E, '+', C.binary(E, '*', C.binary(E, '-', one, y), z), C.binary(E, '*', logWeight, softplusNeg))
  }
  if (weight) per = C.binary(E, '*', per, weight)
  const out = red === 'none' ? per : C.reduceSum(E, per, null, false, red === 'mean')
  if (out.fn) out.fn.name = logits ? 'BinaryCrossEntropyWithLogitsBackward0' : 'BinaryCrossEntropyBackward0'
  return out
}

/** the warning PyTorch gives before it broadcasts a target of a different shape */
const warnShape = (h: Host, a: TT, b: TT) => {
  if (a.shape.join() !== b.shape.join()) h.print(`UserWarning: Using a target size (torch.Size([${b.shape.join(', ')}])) that is different to the input size (torch.Size([${a.shape.join(', ')}])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.\n`)
}

const smoothL1 = (h: Host, a: TT, b: TT, beta: number, red: C.Reduction): TT => {
  warnShape(h, a, b)
  if (beta < 0) throw h.err('RuntimeError', 'smooth_l1_loss does not support negative values for beta.')
  if (beta === 0) return C.pointLoss(stateOf(h).E, a, b, Math.abs, Math.sign, 'SmoothL1LossBackward0', red)
  return C.pointLoss(stateOf(h).E, a, b, (d) => (Math.abs(d) < beta ? (0.5 * d * d) / beta : Math.abs(d) - 0.5 * beta), (d) => (Math.abs(d) < beta ? d / beta : Math.sign(d)), 'SmoothL1LossBackward0', red)
}

const huber = (h: Host, a: TT, b: TT, delta: number, red: C.Reduction): TT => {
  warnShape(h, a, b)
  if (delta <= 0) throw h.err('RuntimeError', 'huber_loss does not support non-positive values for delta.')
  return C.pointLoss(stateOf(h).E, a, b, (d) => (Math.abs(d) <= delta ? 0.5 * d * d : delta * (Math.abs(d) - 0.5 * delta)), (d) => (Math.abs(d) <= delta ? d : delta * Math.sign(d)), 'HuberLossBackward0', red)
}

const l1 = (h: Host, a: TT, b: TT, red: C.Reduction): TT => {
  warnShape(h, a, b)
  const E = stateOf(h).E
  const per = C.unary(E, 'abs', C.binary(E, '-', a, b))
  return red === 'none' ? per : C.reduceSum(E, per, null, false, red === 'mean')
}

// ---------------------------------------------------------------- optimisers

/** one parameter's state: its step count and named buffers, as Optimizer.state_dict()['state'] holds them */
interface ParamState { step: number; buf: Record<string, number[]> }

/** a group's settings as plain numbers and flags, read afresh at each step so that changing param_groups takes effect */
type Hyper = (key: string) => number
type UpdateRule = (w: number[], g: number[], st: ParamState, hp: Hyper) => number[]

const zerosLike = (x: number[]) => x.map(() => 0)

/** L2 weight decay added to the gradient, as all of these do except AdamW */
const decayed = (g: number[], w: number[], wd: number) => (wd ? g.map((x, i) => x + wd * w[i]) : g)

const adamStep = (w: number[], g: number[], st: ParamState, hp: Hyper): number[] => {
  const [b1, b2, lr, eps] = [hp('b1'), hp('b2'), hp('lr'), hp('eps')]
  st.buf.exp_avg = (st.buf.exp_avg ?? zerosLike(g)).map((m, i) => b1 * m + (1 - b1) * g[i])
  st.buf.exp_avg_sq = (st.buf.exp_avg_sq ?? zerosLike(g)).map((v, i) => b2 * v + (1 - b2) * g[i] * g[i])
  const c1 = 1 - b1 ** st.step
  const c2 = 1 - b2 ** st.step
  const second = hp('amsgrad') ? (st.buf.max_exp_avg_sq = (st.buf.max_exp_avg_sq ?? zerosLike(g)).map((m, i) => Math.max(m, st.buf.exp_avg_sq[i]))) : st.buf.exp_avg_sq
  return w.map((x, i) => x - (lr / c1) * st.buf.exp_avg[i] / (Math.sqrt(second[i]) / Math.sqrt(c2) + eps))
}

/** the published update rules, with each optimiser's own defaults listed below */
const UPDATE: Record<string, UpdateRule> = {
  SGD: (w, g0, st, hp) => {
    const [lr, mu, damp] = [hp('lr'), hp('momentum'), hp('dampening')]
    let g = decayed(g0, w, hp('weight_decay'))
    if (mu) {
      st.buf.momentum_buffer = st.buf.momentum_buffer ? st.buf.momentum_buffer.map((b, i) => mu * b + (1 - damp) * g[i]) : [...g]
      g = hp('nesterov') ? g.map((x, i) => x + mu * st.buf.momentum_buffer[i]) : st.buf.momentum_buffer
    }
    return w.map((x, i) => x - lr * g[i])
  },
  Adam: (w, g0, st, hp) => (hp('decoupled_weight_decay') ? UPDATE.AdamW(w, g0, st, hp) : adamStep(w, decayed(g0, w, hp('weight_decay')), st, hp)),
  AdamW: (w, g0, st, hp) => {
    const shrink = 1 - hp('lr') * hp('weight_decay')
    return adamStep(w.map((x) => x * shrink), g0, st, hp)
  },
  RMSprop: (w, g0, st, hp) => {
    const [lr, alpha, eps, mu] = [hp('lr'), hp('alpha'), hp('eps'), hp('momentum')]
    const g = decayed(g0, w, hp('weight_decay'))
    st.buf.square_avg = (st.buf.square_avg ?? zerosLike(g)).map((v, i) => alpha * v + (1 - alpha) * g[i] * g[i])
    let avg = st.buf.square_avg
    if (hp('centered')) {
      st.buf.grad_avg = (st.buf.grad_avg ?? zerosLike(g)).map((m, i) => alpha * m + (1 - alpha) * g[i])
      avg = avg.map((v, i) => v - st.buf.grad_avg[i] ** 2)
    }
    const scaled = g.map((x, i) => x / (Math.sqrt(avg[i]) + eps))
    if (!mu) return w.map((x, i) => x - lr * scaled[i])
    st.buf.momentum_buffer = (st.buf.momentum_buffer ?? zerosLike(g)).map((b, i) => mu * b + scaled[i])
    return w.map((x, i) => x - lr * st.buf.momentum_buffer[i])
  },
  Adagrad: (w, g0, st, hp) => {
    const g = decayed(g0, w, hp('weight_decay'))
    const lr = hp('lr') / (1 + (st.step - 1) * hp('lr_decay'))
    st.buf.sum = (st.buf.sum ?? g.map(() => hp('initial_accumulator_value'))).map((s, i) => s + g[i] * g[i])
    return w.map((x, i) => x - (lr * g[i]) / (Math.sqrt(st.buf.sum[i]) + hp('eps')))
  },
}

class Optimizer extends PyObj {
  readonly cls: string
  groups: Value
  defaults: [string, Value][]
  state = new Map<TT, ParamState>()
  constructor(kind: string, h: Host, params: Value, defaults: [string, Value][]) {
    super()
    this.cls = kind
    this.defaults = defaults
    this.groups = py.list([])
    const single = asTT(params)
    if (single) throw h.err('TypeError', `params argument given to the optimizer should be an iterable of Tensors or dicts, but got torch.${C.typeLabel(single.dtype)}Tensor`)
    const items = h.iterate(params)
    if (!items.length) throw h.err('ValueError', 'optimizer got an empty parameter list')
    if (items[0].k === 'dict') for (const g of items) this.addGroup(h, g)
    else this.addGroup(h, { k: 'dict', d: PyDict.from([[py.str('params'), py.list(items)]]) })
  }
  /** param_groups[i] is a real dict: a learning rate written into it is the one the next step uses */
  addGroup(h: Host, g: Value) {
    if (g.k !== 'dict') throw h.err('TypeError', 'param_group must be a dict')
    const raw = g.d.get(py.str('params'))
    if (!raw) throw h.err('KeyError', "'params'")
    const single = asTT(raw)
    const params = (single ? [raw] : h.iterate(raw)).map((p) => {
      // named_parameters() gives (name, tensor) pairs
      const t = asTT(p.k === 'tuple' ? p.items[1] : p)
      if (!t) throw h.err('TypeError', `optimizer can only optimize Tensors, but one of the params is ${typeName(p)}`)
      if (!t.isLeaf) throw h.err('ValueError', "can't optimize a non-leaf Tensor")
      return wrap(t)
    })
    if (!params.length) throw h.err('ValueError', 'optimizer got an empty parameter list')
    const seen = new Set(this.paramsIn().map((t) => t))
    if (params.some((p) => seen.has(asTT(p)!))) throw h.err('ValueError', 'some parameters appear in more than one parameter group')
    const d = PyDict.from([[py.str('params'), py.list(params)], ...g.d.items().filter(([k]) => !(k.k === 'str' && k.v === 'params'))])
    for (const [k, v] of this.defaults) if (!d.has(py.str(k))) d.set(py.str(k), v)
    ;(this.groups as { items: Value[] }).items.push({ k: 'dict', d })
  }
  private groupDicts(): PyDict[] { return (this.groups as { items: Value[] }).items.map((g) => (g as { d: PyDict }).d) }
  private paramsIn(d?: PyDict): TT[] {
    return (d ? [d] : this.groupDicts()).flatMap((x) => (x.get(py.str('params')) as { items: Value[] }).items.map((p) => asTT(p)!))
  }
  repr() {
    const lines = this.groupDicts().map((d, i) => {
      const keys = d.items().filter(([k]) => !(k.k === 'str' && k.v === 'params')).sort(([a], [b]) => (str(a) < str(b) ? -1 : 1))
      return `\nParameter Group ${i}\n${keys.map(([k, v]) => `    ${str(k)}: ${str(v)}\n`).join('')}`
    })
    return `${this.cls} (${lines.join('')})`
  }
  /** optimizer.step(): every parameter with a gradient moves by this optimiser's rule */
  step(h: Host) {
    for (const d of this.groupDicts()) {
      const hp: Hyper = (key) => {
        const v = d.get(py.str(key))
        if (key === 'b1' || key === 'b2') return h.num(h.iterate(d.get(py.str('betas'))!)[key === 'b1' ? 0 : 1])
        return !v || v.k === 'none' ? 0 : h.num(v)
      }
      for (const p of this.paramsIn(d)) {
        if (!p.grad) continue
        let st = this.state.get(p)
        if (!st) this.state.set(p, (st = { step: 0, buf: {} }))
        st.step++
        const g = hp('maximize') ? p.grad.values().map((x) => -x) : p.grad.values()
        const next = UPDATE[this.cls](p.values(), g, st, hp)
        C.writeInto(p, next)
        if (p.dtype === 'float32') for (const k in st.buf) st.buf[k] = st.buf[k].map(Math.fround)
      }
    }
  }
  getAttr(name: string, h: Host): Value | undefined {
    const fn = (call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })
    switch (name) {
      case 'zero_grad': return fn((args, kw) => {
        const toNone = flagOf(h, kw.set_to_none ?? args[0], true)
        for (const p of this.paramsIn()) p.grad = toNone || !p.grad ? null : TT.of(zerosLike(p.grad.values()), p.shape, p.dtype)
        return py.NONE
      })
      case 'step': return fn((args, kw) => {
        const closure = args[0] ?? kw.closure
        const E = stateOf(h).E
        let loss: Value = py.NONE
        if (closure && closure.k !== 'none') {
          const was = E.grad
          E.grad = true
          try {
            loss = h.call(closure, [])
          } finally {
            E.grad = was
          }
        }
        this.step(h)
        return loss
      })
      case 'param_groups': return this.groups
      case 'defaults': return { k: 'dict', d: PyDict.from(this.defaults.map(([k, v]) => [py.str(k), v])) }
      case 'add_param_group': return fn((args) => { this.addGroup(h, args[0]); return py.NONE })
      case 'state_dict': return fn(() => {
        const ids = this.paramsIn()
        const state = new PyDict()
        ids.forEach((p, i) => {
          const st = this.state.get(p)
          // SGD keeps no step count
          const step: [Value, Value][] = this.cls === 'SGD' ? [] : [[py.str('step'), wrap(TT.scalar(st?.step ?? 0, 'float32'))]]
          if (st && (step.length || Object.keys(st.buf).length)) state.set(py.int(i), { k: 'dict', d: PyDict.from([...step, ...Object.entries(st.buf).map(([k, v]): [Value, Value] => [py.str(k), wrap(TT.of(v, p.shape, p.dtype))])]) })
        })
        let at = 0
        const groups = this.groupDicts().map((d): Value => ({ k: 'dict', d: PyDict.from(d.items().map(([k, v]): [Value, Value] => {
          if (!(k.k === 'str' && k.v === 'params')) return [k, v]
          const n = (v as { items: Value[] }).items.length
          at += n
          return [k, py.list(Array.from({ length: n }, (_, i) => py.int(at - n + i)))]
        })) }))
        return { k: 'dict', d: PyDict.from([[py.str('state'), { k: 'dict', d: state }], [py.str('param_groups'), py.list(groups)]]) }
      })
      case 'load_state_dict': return fn((args) => {
        const d = args[0]
        const sd = d?.k === 'dict' ? d.d : null
        const groups = sd?.get(py.str('param_groups'))
        const state = sd?.get(py.str('state'))
        if (!groups || groups.k !== 'list' || !state || state.k !== 'dict') throw h.err('TypeError', 'load_state_dict() expects what Optimizer.state_dict() returned')
        const mine = this.groupDicts()
        if (groups.items.length !== mine.length) throw h.err('ValueError', 'loaded state dict has a different number of parameter groups')
        groups.items.forEach((g, i) => {
          for (const [k, v] of (g as { d: PyDict }).d.items()) if (!(k.k === 'str' && k.v === 'params')) mine[i].set(k, v)
        })
        const ids = this.paramsIn()
        this.state = new Map()
        for (const [k, v] of state.d.items()) {
          const p = ids[h.toInt(k)]
          const st: ParamState = { step: 0, buf: {} }
          for (const [n, val] of (v as { d: PyDict }).d.items()) {
            const t = asTT(val)!
            if (str(n) === 'step') st.step = t.values()[0]
            else st.buf[str(n)] = t.values()
          }
          this.state.set(p, st)
        }
        return py.NONE
      })
    }
    return undefined
  }
}

/** torch.optim.lr_scheduler: each step() writes new learning rates into the optimiser's param_groups */
class Scheduler extends PyObj {
  readonly cls: string
  opt: Optimizer
  lastEpoch = -1
  baseLrs: number[]
  lastLr: number[] = []
  next: (lr: number, base: number, epoch: number) => number
  constructor(kind: string, h: Host, opt: Optimizer, next: (lr: number, base: number, epoch: number) => number) {
    super()
    this.cls = kind
    this.opt = opt
    this.next = next
    const dicts = this.dicts()
    for (const d of dicts) if (!d.has(py.str('initial_lr'))) d.set(py.str('initial_lr'), d.get(py.str('lr'))!)
    this.baseLrs = dicts.map((d) => h.num(d.get(py.str('initial_lr'))!))
    this.advance(h)
  }
  private dicts(): PyDict[] { return (this.opt.groups as { items: Value[] }).items.map((g) => (g as { d: PyDict }).d) }
  advance(h: Host) {
    this.lastEpoch++
    const dicts = this.dicts()
    dicts.forEach((d, i) => d.set(py.str('lr'), py.float(this.next(h.num(d.get(py.str('lr'))!), this.baseLrs[i], this.lastEpoch))))
    this.lastLr = dicts.map((d) => h.num(d.get(py.str('lr'))!))
  }
  repr() { return `<torch.optim.lr_scheduler.${this.cls} object>` }
  isa(name: string) { return name === 'LRScheduler' }
  getAttr(name: string, h: Host): Value | undefined {
    switch (name) {
      case 'step': return { k: 'fn', name, call: (args, kw) => {
        if (args[0] ?? kw.epoch) throw h.err('NotImplementedError', 'scheduler.step(epoch): the deprecated epoch argument is not available in this sandbox')
        this.advance(h)
        return py.NONE
      } }
      case 'get_last_lr': return { k: 'fn', name, call: () => py.list(this.lastLr.map(py.float)) }
      case 'last_epoch': return py.int(this.lastEpoch)
      case 'base_lrs': return py.list(this.baseLrs.map(py.float))
      case 'optimizer': return py.obj(this.opt)
    }
    return undefined
  }
}

class ReduceLROnPlateau extends PyObj {
  readonly cls = 'ReduceLROnPlateau'
  opt: Optimizer
  max: boolean
  factor: number
  patience: number
  threshold: number
  absMode: boolean
  cooldown: number
  minLr: number[]
  eps: number
  best: number
  bad = 0
  cool = 0
  lastEpoch = 0
  lastLr: number[]
  constructor(h: Host, opt: Optimizer, max: boolean, factor: number, patience: number, threshold: number, absMode: boolean, cooldown: number, minLr: Value | undefined, eps: number) {
    super()
    if (factor >= 1) throw h.err('ValueError', 'Factor should be < 1.0.')
    this.opt = opt
    this.max = max
    this.factor = factor
    this.patience = patience
    this.threshold = threshold
    this.absMode = absMode
    this.cooldown = cooldown
    this.eps = eps
    const n = this.dicts().length
    this.minLr = minLr && (minLr.k === 'list' || minLr.k === 'tuple') ? minLr.items.map((v) => h.num(v)) : new Array<number>(n).fill(minLr ? h.num(minLr) : 0)
    this.best = max ? -Infinity : Infinity
    this.lastLr = this.dicts().map((d) => h.num(d.get(py.str('lr'))!))
  }
  private dicts(): PyDict[] { return (this.opt.groups as { items: Value[] }).items.map((g) => (g as { d: PyDict }).d) }
  private better(a: number) {
    if (this.max) return this.absMode ? a > this.best + this.threshold : a > this.best * (1 + this.threshold)
    return this.absMode ? a < this.best - this.threshold : a < this.best * (1 - this.threshold)
  }
  repr() { return '<torch.optim.lr_scheduler.ReduceLROnPlateau object>' }
  isa(name: string) { return name === 'LRScheduler' }
  getAttr(name: string, h: Host): Value | undefined {
    switch (name) {
      case 'step': return { k: 'fn', name, call: (args, kw) => {
        const metric = args[0] ?? kw.metrics
        if (!metric) throw h.err('TypeError', "step() missing 1 required positional argument: 'metrics'")
        const current = h.num(metric)
        this.lastEpoch++
        if (this.better(current)) {
          this.best = current
          this.bad = 0
        } else this.bad++
        if (this.cool > 0) {
          this.cool--
          this.bad = 0
        }
        if (this.bad > this.patience) {
          this.dicts().forEach((d, i) => {
            const old = h.num(d.get(py.str('lr'))!)
            const fresh = Math.max(old * this.factor, this.minLr[i] ?? 0)
            if (old - fresh > this.eps) d.set(py.str('lr'), py.float(fresh))
          })
          this.cool = this.cooldown
          this.bad = 0
        }
        this.lastLr = this.dicts().map((d) => h.num(d.get(py.str('lr'))!))
        return py.NONE
      } }
      case 'get_last_lr': return { k: 'fn', name, call: () => py.list(this.lastLr.map(py.float)) }
      case 'best': return py.float(this.best)
      case 'num_bad_epochs': return py.int(this.bad)
      case 'last_epoch': return py.int(this.lastEpoch)
      case 'optimizer': return py.obj(this.opt)
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
    if (idx.k === 'int') {
      // a DataLoader asks for one row at a time: this has to stay cheap on a large tensor
      const n = this.len()
      const i = idx.v < 0 ? idx.v + n : idx.v
      if (i < 0 || i >= n) throw h.err('IndexError', `index ${idx.v} is out of bounds for dimension 0 with size ${n}`)
      return py.tuple(this.tensors.map((t) => wrap(C.viewT(E, t, (x) => x.view(x.shape.slice(1), x.strides.slice(1), x.offset + i * x.strides[0]), 'SelectBackward0'))))
    }
    return py.tuple(this.tensors.map((t) => wrap(C.indexT(E, t, indexItems(h, idx)))))
  }
  isa(name: string) { return name === 'Dataset' }
  getAttr(name: string): Value | undefined { return name === 'tensors' ? py.tuple(this.tensors.map(wrap)) : undefined }
}

const datasetLen = (h: Host, ds: Value): number => (ds.k === 'obj' && ds.o.len ? ds.o.len(h) : h.num(h.call(h.getAttr(ds, '__len__'), [])))
const datasetItem = (h: Host, ds: Value, i: number): Value => (ds.k === 'obj' && ds.o.getItem ? ds.o.getItem(py.int(i), h) : h.call(h.getAttr(ds, '__getitem__'), [py.int(i)]))

/** random_split's pieces */
class Subset extends PyObj {
  readonly cls = 'Subset'
  ds: Value
  indices: number[]
  constructor(ds: Value, indices: number[]) {
    super()
    this.ds = ds
    this.indices = indices
  }
  repr() { return '<torch.utils.data.dataset.Subset object>' }
  len() { return this.indices.length }
  getItem(idx: Value, h: Host): Value {
    if (idx.k === 'list') return py.list(idx.items.map((v) => this.getItem(v, h)))
    const n = this.indices.length
    const i = h.toInt(idx)
    if (i < -n || i >= n) throw h.err('IndexError', `list index out of range`)
    return datasetItem(h, this.ds, this.indices[i < 0 ? i + n : i])
  }
  isa(name: string) { return name === 'Dataset' }
  getAttr(name: string): Value | undefined { return name === 'dataset' ? this.ds : name === 'indices' ? py.list(this.indices.map(py.int)) : undefined }
}

class ConcatDataset extends PyObj {
  readonly cls = 'ConcatDataset'
  parts: Value[]
  constructor(parts: Value[]) {
    super()
    this.parts = parts
  }
  repr() { return '<torch.utils.data.dataset.ConcatDataset object>' }
  len(h: Host) { return this.parts.reduce((s, d) => s + datasetLen(h, d), 0) }
  getItem(idx: Value, h: Host): Value {
    let i = h.toInt(idx)
    const n = this.len(h)
    if (i < -n || i >= n) throw h.err('ValueError', 'absolute value of index should not exceed dataset length')
    if (i < 0) i += n
    for (const d of this.parts) {
      const size = datasetLen(h, d)
      if (i < size) return datasetItem(h, d, i)
      i -= size
    }
    throw h.err('IndexError', 'dataset index out of range')
  }
  isa(name: string) { return name === 'Dataset' }
}

/** a Sampler: the indices of one pass, drawn afresh each time it is iterated */
class SamplerObj extends PyObj {
  readonly cls: string
  draw: (h: Host) => number[]
  size: (h: Host) => number
  constructor(kind: string, draw: (h: Host) => number[], size: (h: Host) => number) {
    super()
    this.cls = kind
    this.draw = draw
    this.size = size
  }
  repr() { return `<torch.utils.data.sampler.${this.cls} object>` }
  iter(h: Host) { return this.draw(h).map(py.int) }
  len(h: Host) { return this.size(h) }
  isa(name: string) { return name === 'Sampler' }
}

const shuffled = (rng: SandboxRandom, n: number): number[] => {
  const order = Array.from({ length: n }, (_, i) => i)
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1))
    ;[order[i], order[j]] = [order[j], order[i]]
  }
  return order
}

const rngOf = (h: Host, g: Value | undefined): SandboxRandom => {
  if (!g || g.k === 'none') return h.rng
  if (g.k === 'obj' && g.o instanceof GeneratorObj) return g.o.rng
  throw h.err('TypeError', `generator must be a torch.Generator, not ${typeName(g)}`)
}

/** default_collate: a batch of samples → stacked tensors, keeping the sample's structure */
const collate = (h: Host, samples: Value[]): Value => {
  const E = stateOf(h).E
  const first = samples[0]
  if (asTT(first)) return wrap(C.catT(E, samples.map((v) => tensorArg(h, v, 'default_collate')), 0, true))
  if (first.k === 'array') return wrap(C.catT(E, samples.map((v) => fromNd((v as { a: NDArray }).a, false)), 0, true))
  if (first.k === 'float') return wrap(TT.of(samples.map((v) => h.num(v)), [samples.length], 'float64'))
  if (first.k === 'int') return wrap(TT.of(samples.map((v) => h.num(v)), [samples.length], 'int64'))
  if (first.k === 'bool') return wrap(TT.of(samples.map((v) => h.num(v)), [samples.length], 'bool'))
  if (first.k === 'str') return py.list(samples)
  if (first.k === 'dict') return { k: 'dict', d: PyDict.from(first.d.keys().map((key): [Value, Value] => [key, collate(h, samples.map((v) => (v as { d: PyDict }).d.get(key)!))])) }
  if (first.k === 'tuple' || first.k === 'list') {
    const size = first.items.length
    if (samples.some((v) => (v as { items: Value[] }).items.length !== size)) throw h.err('RuntimeError', 'each element in list of batch should be of equal size')
    return py.list(first.items.map((_, c) => collate(h, samples.map((v) => (v as { items: Value[] }).items[c]))))
  }
  throw h.err('TypeError', `default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found ${typeName(first)}`)
}

class DataLoader extends PyObj {
  readonly cls = 'DataLoader'
  ds: Value
  bs: number
  shuffle: boolean
  dropLast: boolean
  sampler: Value | null
  collateFn: Value | null
  rng: SandboxRandom
  constructor(ds: Value, bs: number, shuffle: boolean, dropLast: boolean, sampler: Value | null, collateFn: Value | null, rng: SandboxRandom) {
    super()
    this.ds = ds
    this.bs = bs
    this.shuffle = shuffle
    this.dropLast = dropLast
    this.sampler = sampler
    this.collateFn = collateFn
    this.rng = rng
  }
  repr() { return '<torch.utils.data.dataloader.DataLoader object>' }
  size(h: Host) { return this.sampler && this.sampler.k === 'obj' && this.sampler.o.len ? this.sampler.o.len(h) : this.sampler ? h.iterate(this.sampler).length : datasetLen(h, this.ds) }
  len(h: Host) { return this.dropLast ? Math.floor(this.size(h) / this.bs) : Math.ceil(this.size(h) / this.bs) }
  getAttr(name: string): Value | undefined {
    if (name === 'dataset') return this.ds
    if (name === 'batch_size') return py.int(this.bs)
    if (name === 'drop_last') return py.bool(this.dropLast)
    return undefined
  }
  iter(h: Host): Value[] {
    const n = datasetLen(h, this.ds)
    const order = this.sampler ? h.iterate(this.sampler).map((v) => h.toInt(v)) : this.shuffle ? shuffled(this.rng, n) : Array.from({ length: n }, (_, i) => i)
    const batches: Value[] = []
    const stop = this.dropLast ? order.length - (order.length % this.bs) : order.length
    for (let s = 0; s < stop; s += this.bs) {
      const samples = order.slice(s, s + this.bs).map((i) => datasetItem(h, this.ds, i))
      batches.push(this.collateFn ? h.call(this.collateFn, [py.list(samples)]) : collate(h, samples))
    }
    return batches
  }
}

// ---------------------------------------------------------------- the torch module

const fnv = (name: string, call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })

/** torch.save keeps a snapshot, so that later training does not change what was saved */
const snapshot = (x: unknown, memo: Map<object, unknown>): unknown => {
  if (x === null || typeof x !== 'object') return x
  if (memo.has(x)) return memo.get(x)
  const remember = <T extends object>(o: T): T => {
    memo.set(x, o)
    return o
  }
  if (x instanceof TT) {
    const t = remember(TT.of(x.values(), x.shape, x.dtype))
    t.requiresGrad = x.requiresGrad && x.isLeaf
    t.isParam = x.isParam
    return t
  }
  if (x instanceof TensorObj) return objOf(snapshot(x.t, memo) as TT)
  if (x instanceof NDArray) return x.copy()
  if (x instanceof PyDict) {
    const d = remember(new PyDict())
    for (const [k, v] of x.items()) d.set(k, snapshot(v, memo) as Value)
    return d
  }
  if (x instanceof Map) {
    const m = remember(new Map())
    for (const [k, v] of x) m.set(k, snapshot(v, memo))
    return m
  }
  if (Array.isArray(x)) {
    const a: unknown[] = remember([])
    for (const v of x) a.push(snapshot(v, memo))
    return a
  }
  // a layer, a container Value or a plain record of fields; classes, functions and other library objects stay shared
  const plain = Object.getPrototypeOf(x) === Object.prototype
  const copyable = x instanceof Layer || (plain && (!('k' in x) || ['list', 'tuple', 'dict', 'inst', 'obj', 'array'].includes((x as Value).k)))
  if (!copyable) return x
  const o = remember(Object.create(Object.getPrototypeOf(x)) as Record<string, unknown>)
  for (const [k, v] of Object.entries(x)) o[k] = snapshot(v, memo)
  return o
}

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
    trace(h, `torch.${name}`, name.startsWith('rand') || name === 'bernoulli' || name === 'normal' || name === 'multinomial' ? 'random' : 'create', [], t)
    return wrap(t)
  }
  const filled = (name: string, value: (rng: SandboxRandom) => number, defaultDtype: TDType = 'float32', floatOnly = false) =>
    fnv(name, (args, kw) => {
      const dims = toDims(h, kw.size ? [kw.size] : args)
      const n = prod(dims)
      if (n > h.maxSize) throw h.err('RuntimeError', `[enforce fail] tensor of ${n} elements exceeds the sandbox limit of ${h.maxSize}`)
      const dtype = dtypeArg(h, kw.dtype) ?? defaultDtype
      if (floatOnly && !isFloatT(dtype)) throw h.err('RuntimeError', `"check_uniform_bounds" not implemented for '${C.typeLabel(dtype)}'`)
      const rng = rngOf(h, kw.generator)
      return finish(TT.of(Array.from({ length: n }, () => value(rng)), dims, dtype), kw, name)
    })
  const like = (name: string, value: () => number, float = false) =>
    fnv(name, (args, kw) => {
      const t = T(args[0], name)
      const dtype = dtypeArg(h, kw.dtype) ?? t.dtype
      if (float && !isFloatT(dtype)) throw h.err('RuntimeError', `"check_uniform_bounds" not implemented for '${C.typeLabel(dtype)}'`)
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
    const self = args[0] ?? kw.input
    if (!asTT(self ?? py.NONE)) throw h.err('TypeError', `${name}(): argument 'input' must be Tensor, not ${self ? typeName(self) : 'nothing'}`)
    const { input: _input, ...rest } = kw
    return h.call(h.getAttr(self, method), args.slice(1), rest)
  })
  const twoArg = (name: string, op: string) => fnv(name, (args) => {
    const self = args[0]
    if (self?.k !== 'obj' || !(self.o instanceof TensorObj)) throw h.err('TypeError', `${name}(): argument 'input' must be Tensor`)
    const out = self.o.binop(op, args[1], false, h)
    if (!out) throw h.err('TypeError', `${name}(): argument 'other' must be a Tensor or a number`)
    return out
  })

  // ---- nn
  type Args = Record<string, Value | undefined>
  const num = (v: Value | undefined, d: number) => (v && v.k !== 'none' ? h.num(v) : d)
  const int = (v: Value | undefined, what: string) => {
    if (!v) throw h.err('TypeError', `missing required argument '${what}'`)
    return h.toInt(v)
  }
  const bool = (v: Value | undefined, d: boolean) => flagOf(h, v, d)
  /** a parameter this sandbox does not implement must not be accepted and ignored */
  const refuse = (who: string, a: Args, key: string, isDefault: (v: Value) => boolean) => {
    const v = a[key]
    if (v && !isDefault(v)) throw h.err('NotImplementedError', `${who}: ${key}=${repr(v)} is not available in this sandbox`)
  }
  const isNone = (v: Value) => v.k === 'none'
  const isFalse = (v: Value) => !h.truthy(v)
  const isOne = (v: Value) => (v.k === 'tuple' || v.k === 'list' ? v.items.every((x) => h.num(x) === 1) : h.num(v) === 1)

  /** an nn class: device and dtype are handled here, every other keyword must be one the layer lists */
  const layer = (name: string, params: string, make: (a: Args) => Layer): Value => ({
    k: 'type', name, call: (args, kw) => {
      const { device, dtype, ...rest } = kw
      if (device && device.k !== 'none' && h.str(device).startsWith('cuda')) throw h.err('AssertionError', 'Torch not compiled with CUDA enabled')
      const m = py.obj(make(bind(h, `${name}.__init__`, params, args, rest)))
      const d = dtypeArg(h, dtype)
      if (d) castFloats(m, d)
      return m
    },
  })
  const reduction = (a: Args): C.Reduction => {
    if ((a.size_average && !isNone(a.size_average)) || (a.reduce && !isNone(a.reduce))) return !bool(a.reduce, true) ? 'none' : bool(a.size_average, true) ? 'mean' : 'sum'
    const v = a.reduction
    if (!v || v.k === 'none') return 'mean'
    const r = h.str(v)
    if (r !== 'mean' && r !== 'sum' && r !== 'none') throw h.err('ValueError', `${r} is not a valid value for reduction`)
    return r
  }
  const optT = (v: Value | undefined): TT | null => (v && v.k !== 'none' ? T(v, 'weight') : null)
  const ceOpts = (a: Args): CeOpts => ({ weight: optT(a.weight), ignore: a.ignore_index ? h.toInt(a.ignore_index) : -100, smoothing: num(a.label_smoothing, 0), red: reduction(a) })

  const pairOf = (v: Value | undefined, d: number, one: boolean, rest: number): [number, number] => {
    if (!one) return pair(h, v, d)
    if (!v || v.k === 'none') return [rest, d]
    return [rest, h.toInt(v.k === 'tuple' || v.k === 'list' ? v.items[0] : v)]
  }
  const paddingModes: Record<string, C.PadMode> = { zeros: 'constant', reflect: 'reflect', replicate: 'replicate', circular: 'circular' }
  const padModeOf = (v: Value | undefined): C.PadMode => {
    const s = v ? h.str(v) : 'zeros'
    if (!paddingModes[s]) throw h.err('ValueError', `padding_mode must be one of {'zeros', 'reflect', 'replicate', 'circular'}, but got padding_mode='${s}'`)
    return paddingModes[s]
  }
  const convOpts = (a: Args, one: boolean): ConvOpts => {
    const stride = pairOf(a.stride, 1, one, 1)
    const pad = a.padding
    if (pad?.k === 'str' && pad.v !== 'same' && pad.v !== 'valid') throw h.err('ValueError', `Invalid padding string '${pad.v}', should be one of {'valid', 'same'}`)
    if (pad?.k === 'str' && pad.v === 'same' && (stride[0] !== 1 || stride[1] !== 1)) throw h.err('ValueError', "padding='same' is not supported for strided convolutions")
    return {
      stride, dilation: pairOf(a.dilation, 1, one, 1), groups: a.groups ? h.toInt(a.groups) : 1, paddingMode: padModeOf(a.padding_mode),
      padding: pad?.k === 'str' ? (pad.v as 'same' | 'valid') : pairOf(pad, 0, one, 0),
    }
  }
  const CONV_PARAMS = 'in_channels out_channels kernel_size stride padding dilation groups bias padding_mode'
  const makeConv = (kind: string, a: Args) => {
    const one = kind === 'Conv1d'
    const [inC, outC] = [int(a.in_channels, 'in_channels'), int(a.out_channels, 'out_channels')]
    const o = convOpts(a, one)
    if (o.groups <= 0) throw h.err('ValueError', 'groups must be a positive integer')
    if (inC % o.groups) throw h.err('ValueError', 'in_channels must be divisible by groups')
    if (outC % o.groups) throw h.err('ValueError', 'out_channels must be divisible by groups')
    return new ConvLayer(h, kind, inC, outC, pairOf(a.kernel_size, 1, one, 1), o, bool(a.bias, true))
  }
  const poolRefuse = (who: string, a: Args, max: boolean) => {
    refuse(who, a, 'ceil_mode', isFalse)
    if (max) {
      refuse(who, a, 'dilation', isOne)
      refuse(who, a, 'return_indices', isFalse)
    } else {
      refuse(who, a, 'count_include_pad', (v) => h.truthy(v))
      refuse(who, a, 'divisor_override', isNone)
    }
  }
  const MAXPOOL = 'kernel_size stride padding dilation return_indices ceil_mode'
  const FMAXPOOL = 'kernel_size stride padding dilation ceil_mode return_indices'
  const AVGPOOL = 'kernel_size stride padding ceil_mode count_include_pad divisor_override'
  const pool2Layer = (mode: 'max' | 'avg') => (a: Args) => {
    const name = mode === 'max' ? 'MaxPool2d' : 'AvgPool2d'
    poolRefuse(name, a, mode === 'max')
    const ks = pair(h, a.kernel_size, 2)
    const stride = a.stride && a.stride.k !== 'none' ? a.stride : a.kernel_size
    return new Pool2d(mode, ks, pair(h, stride, 1), pair(h, a.padding, 0), `kernel_size=${repr(a.kernel_size ?? py.int(2))}, stride=${repr(stride ?? py.int(2))}, padding=${repr(a.padding ?? py.int(0))}`)
  }
  const pool1Layer = (mode: 'max' | 'avg') => (a: Args) => {
    poolRefuse(mode === 'max' ? 'MaxPool1d' : 'AvgPool1d', a, mode === 'max')
    const k = pairOf(a.kernel_size, 2, true, 1)[1]
    return new Pool1d(mode, k, a.stride && a.stride.k !== 'none' ? pairOf(a.stride, 1, true, 1)[1] : k, pairOf(a.padding, 0, true, 0)[1])
  }
  /** F.max_pool2d and friends */
  const pool2F = (name: string, mode: 'max' | 'avg') => fnv(name, (args, kw) => {
    const a = bind(h, name, `input ${mode === 'max' ? FMAXPOOL : AVGPOOL}`, args, kw)
    poolRefuse(`F.${name}`, a, mode === 'max')
    const t = T(a.input, name)
    const k = pair(h, a.kernel_size, 2)
    const out = C.pool2d(E, t, k, a.stride && a.stride.k !== 'none' ? pair(h, a.stride, 1) : k, pair(h, a.padding, 0), mode)
    flowStep(h, `F.${name}`, t, out)
    return wrap(out)
  })
  const pool1F = (name: string, mode: 'max' | 'avg') => fnv(name, (args, kw) => {
    const a = bind(h, name, `input ${mode === 'max' ? FMAXPOOL : AVGPOOL}`, args, kw)
    poolRefuse(`F.${name}`, a, mode === 'max')
    const t = T(a.input, name)
    const k = pairOf(a.kernel_size, 2, true, 1)[1]
    const out = pool1d(h, t, k, a.stride && a.stride.k !== 'none' ? pairOf(a.stride, 1, true, 1)[1] : k, pairOf(a.padding, 0, true, 0)[1], mode)
    flowStep(h, `F.${name}`, t, out)
    return wrap(out)
  })

  /** adaptive pool output size: an int, or a pair where None keeps the input's size */
  const adaptiveSize = (v: Value | undefined, x: TT, one: boolean): [number, number] => {
    const hw: [number, number] = [x.shape[x.ndim - 2], x.shape[x.ndim - 1]]
    if (one) return [1, h.toInt(v!.k === 'tuple' ? v!.items[0] : v!)]
    if (v?.k === 'tuple' || v?.k === 'list') return [v.items[0].k === 'none' ? hw[0] : h.toInt(v.items[0]), v.items[1].k === 'none' ? hw[1] : h.toInt(v.items[1])]
    return [h.toInt(v!), h.toInt(v!)]
  }
  const adaptive = (x: TT, size: Value | undefined, mode: 'max' | 'avg', one: boolean): TT => {
    if (!one) return C.adaptivePool2d(E, x, adaptiveSize(size, x, false), mode)
    if (x.ndim !== 2 && x.ndim !== 3) throw h.err('RuntimeError', `adaptive_${mode}_pool1d(): Expected 2D or 3D tensor, but got: [${x.shape.join(', ')}]`)
    const out = C.adaptivePool2d(E, C.unsqueezeT(E, x, -2), adaptiveSize(size, x, true), mode)
    return C.squeezeT(E, out, -2)
  }

  /** the interpolation shared by F.interpolate and nn.Upsample */
  const interpolate = (x: TT, size: Value | undefined, scale: Value | undefined, mode: string, align: Value | undefined): TT => {
    const spatial = x.ndim - 2
    const hasSize = !!size && size.k !== 'none'
    const hasScale = !!scale && scale.k !== 'none'
    if (spatial < 1 || spatial > 3) throw h.err('NotImplementedError', `Input Error: Only 3D, 4D and 5D input Tensors supported (got ${x.ndim}D)`)
    if (hasSize && hasScale) throw h.err('ValueError', 'only one of size or scale_factor should be defined')
    if (!hasSize && !hasScale) throw h.err('ValueError', 'either size or scale_factor should be defined')
    const list = (v: Value) => (v.k === 'tuple' || v.k === 'list' ? v.items.map((i) => h.num(i)) : new Array<number>(spatial).fill(h.num(v)))
    const given = list((hasSize ? size : scale)!)
    if (given.length !== spatial) throw h.err('ValueError', `Input and output must have the same number of spatial dimensions, but got input with spatial dimensions of [${x.shape.slice(2).join(', ')}] and output size of [${given.join(', ')}]. Please provide input tensor in (N, C, d1, d2, ...,dK) format and output size in (o1, o2, ...,oK) format.`)
    const inSizes = x.shape.slice(2)
    const outSizes = hasSize ? given : inSizes.map((n, k) => Math.floor(n * given[k]))
    // the source position of output cell i: in / out, or 1 / scale_factor when that was what the caller gave
    const ratio = inSizes.map((n, k) => Math.fround(hasSize ? n / outSizes[k] : 1 / given[k]))
    if (mode === 'nearest' || mode === 'nearest-exact') {
      if (align && align.k !== 'none') throw h.err('ValueError', 'align_corners option can only be set with the interpolating modes: linear | bilinear | bicubic | trilinear')
      const offset = mode === 'nearest' ? 0 : 0.5
      const maps = [[...Array(x.shape[0]).keys()], [...Array(x.shape[1]).keys()], ...outSizes.map((m, k) => Array.from({ length: m }, (_, i) => Math.min(Math.floor(Math.fround((i + offset) * ratio[k])), inSizes[k] - 1)))]
      const name = `Upsample${mode === 'nearest' ? 'Nearest' : 'NearestExact'}${spatial}DBackward0`
      return C.takeT(E, x, C.crossSrc(x.shape, maps), maps.map((m) => m.length), name)
    }
    const needs = { linear: 1, bilinear: 2, trilinear: 3 }[mode]
    if (!needs) throw h.err('NotImplementedError', `F.interpolate: mode='${mode}' is not available in this sandbox (nearest, nearest-exact, linear, bilinear and trilinear are)`)
    if (needs !== spatial) throw h.err('NotImplementedError', `Got ${x.ndim}D input, but ${mode} mode needs ${needs + 2}D input`)
    const corners = bool(align, false)
    let out = x
    outSizes.forEach((m, k) => {
      const n = inSizes[k]
      const pos = Array.from({ length: m }, (_, i) => (corners ? (m > 1 ? (i * (n - 1)) / (m - 1) : 0) : Math.max(0, (i + 0.5) * ratio[k] - 0.5)))
      const i0 = pos.map((p) => Math.min(Math.floor(p), n - 1))
      out = C.linearAxis(E, out, k + 2, i0, i0.map((i) => Math.min(i + 1, n - 1)), pos.map((p, i) => p - i0[i]))
    })
    if (out.fn) out.fn.name = `Upsample${mode === 'trilinear' ? 'Trilinear3D' : mode === 'bilinear' ? 'Bilinear2D' : 'Linear1D'}Backward0`
    return out
  }

  const padOf = (x: TT, padV: Value | undefined, modeV: Value | undefined, valueV: Value | undefined): TT => {
    const pad = h.iterate(padV!).map((v) => h.toInt(v))
    const mode = modeV ? h.str(modeV) : 'constant'
    if (pad.length % 2) throw h.err('RuntimeError', 'Padding length must be divisible by 2')
    if (pad.length / 2 > x.ndim) throw h.err('RuntimeError', `Padding length should be less than or equal to two times the input dimension but got padding length ${pad.length} and input of dimension ${x.ndim}`)
    if (mode !== 'constant' && !['reflect', 'replicate', 'circular'].includes(mode)) throw h.err('NotImplementedError', `Unrecognised padding mode ${mode}`)
    if (mode !== 'constant') {
      if (valueV && valueV.k !== 'none' && h.num(valueV) !== 0) throw h.err('AssertionError', `Padding mode "${mode}" doesn't take in value argument`)
      const fits = (pad.length === 2 && (x.ndim === 2 || x.ndim === 3)) || (pad.length === 4 && (x.ndim === 3 || x.ndim === 4)) || (pad.length === 6 && (x.ndim === 4 || x.ndim === 5))
      if (!fits) throw h.err('NotImplementedError', 'Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now')
    }
    return C.padT(E, x, pad, mode as C.PadMode, num(valueV, 0))
  }

  // ---- activation functions, shared by nn.X and F.x
  const sig = (v: number) => 1 / (1 + Math.exp(-v))
  const softplusAt = (beta: number, threshold: number) => (v: number) => (v * beta > threshold ? v : Math.log1p(Math.exp(v * beta)) / beta)
  const hardtanh = (lo: number, hi: number) => (x: TT) => C.mapT(E, x, (v) => Math.min(hi, Math.max(lo, v)), (v) => (v <= lo || v >= hi ? 0 : 1), 'HardtanhBackward0')
  const SELU_ALPHA = 1.6732632423543772
  const SELU_SCALE = 1.0507009873554805
  const inplaceNote = (a: Args) => (bool(a.inplace, false) ? 'inplace=True' : '')
  const quiet = () => ''
  const ACTS: [string, string, string, (a: Args) => (x: TT) => TT, (a: Args) => string][] = [
    ['ReLU', 'relu', 'inplace', () => (x) => C.unary(E, 'relu', x), inplaceNote],
    ['ReLU6', 'relu6', 'inplace', () => hardtanh(0, 6), inplaceNote],
    ['LeakyReLU', 'leaky_relu', 'negative_slope inplace', (a) => {
      const s = num(a.negative_slope, 0.01)
      return (x) => C.mapT(E, x, (v) => (v >= 0 ? v : s * v), (v) => (v > 0 ? 1 : s), 'LeakyReluBackward0')
    }, (a) => `negative_slope=${repr(a.negative_slope ?? py.float(0.01))}${bool(a.inplace, false) ? ', inplace=True' : ''}`],
    ['ELU', 'elu', 'alpha inplace', (a) => {
      const al = num(a.alpha, 1)
      return (x) => C.mapT(E, x, (v) => (v > 0 ? v : al * Math.expm1(v)), (v) => (v > 0 ? 1 : al * Math.exp(v)), 'EluBackward0')
    }, (a) => `alpha=${repr(a.alpha ?? py.float(1))}${bool(a.inplace, false) ? ', inplace=True' : ''}`],
    ['SELU', 'selu', 'inplace', () => (x) => C.mapT(E, x, (v) => SELU_SCALE * (v > 0 ? v : SELU_ALPHA * Math.expm1(v)), (v) => SELU_SCALE * (v > 0 ? 1 : SELU_ALPHA * Math.exp(v)), 'EluBackward0'), inplaceNote],
    ['GELU', 'gelu', 'approximate', (a) => {
      const how = a.approximate ? h.str(a.approximate) : 'none'
      if (how !== 'none' && how !== 'tanh') throw h.err('RuntimeError', `approximate argument must be either none or tanh.`)
      const c = Math.sqrt(2 / Math.PI)
      return how === 'none'
        ? (x) => C.mapT(E, x, (v) => v * 0.5 * (1 + C.erf(v / Math.SQRT2)), (v) => 0.5 * (1 + C.erf(v / Math.SQRT2)) + v * Math.exp(-0.5 * v * v) / Math.sqrt(2 * Math.PI), 'GeluBackward0')
        : (x) => C.mapT(E, x, (v) => 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v ** 3))), (v) => {
          const inner = c * (v + 0.044715 * v ** 3)
          return 0.5 * (1 + Math.tanh(inner)) + 0.5 * v * (c * (1 + 3 * 0.044715 * v * v)) / Math.cosh(inner) ** 2
        }, 'GeluBackward0')
    }, (a) => `approximate=${repr(a.approximate ?? py.str('none'))}`],
    ['SiLU', 'silu', 'inplace', () => (x) => C.mapT(E, x, (v) => v * sig(v), (v) => sig(v) + v * sig(v) * (1 - sig(v)), 'SiluBackward0'), inplaceNote],
    ['Mish', 'mish', 'inplace', () => (x) => {
      const sp = (v: number) => Math.log1p(Math.exp(v))
      return C.mapT(E, x, (v) => v * Math.tanh(sp(v)), (v) => Math.tanh(sp(v)) + v * sig(v) / Math.cosh(sp(v)) ** 2, 'MishBackward0')
    }, inplaceNote],
    ['Softplus', 'softplus', 'beta threshold', (a) => {
      const [b, th] = [num(a.beta, 1), num(a.threshold, 20)]
      return (x) => C.mapT(E, x, softplusAt(b, th), (v) => (v * b > th ? 1 : sig(v * b)), 'SoftplusBackward0')
    }, quiet],
    ['Hardtanh', 'hardtanh', 'min_val max_val inplace', (a) => hardtanh(num(a.min_val, -1), num(a.max_val, 1)), (a) => `min_val=${repr(a.min_val ?? py.float(-1))}, max_val=${repr(a.max_val ?? py.float(1))}${bool(a.inplace, false) ? ', inplace=True' : ''}`],
    ['LogSigmoid', 'logsigmoid', '', () => (x) => C.mapT(E, x, (v) => -softplusAt(1, Infinity)(-v), (v) => sig(-v), 'LogSigmoidBackward0'), quiet],
    ['Softsign', 'softsign', '', () => (x) => C.mapT(E, x, (v) => v / (1 + Math.abs(v)), (v) => 1 / (1 + Math.abs(v)) ** 2, 'SoftsignBackward0'), quiet],
    ['Sigmoid', 'sigmoid', '', () => (x) => C.unary(E, 'sigmoid', x), quiet],
    ['Tanh', 'tanh', '', () => (x) => C.unary(E, 'tanh', x), quiet],
  ]
  /** softmax without a dim: PyTorch's implicit choice */
  const softDim = (x: TT, d: Value | undefined) => (d && d.k !== 'none' ? h.toInt(d) : x.ndim === 0 || x.ndim === 1 || x.ndim === 3 ? 0 : 1)

  const Unflatten = layer('Unflatten', 'dim unflattened_size', (a) => {
    const d = int(a.dim, 'dim')
    const sizesV = toDims(h, [a.unflattened_size ?? py.NONE])
    return new Activation('Unflatten', (x) => {
      const axis = C.normDim(d, x.ndim)
      const out = C.reshapeT(E, x, [...x.shape.slice(0, axis), ...sizesV, ...x.shape.slice(axis + 1)], false)
      if (out.fn) out.fn.name = 'UnflattenBackward0'
      return out
    }, `dim=${d}, unflattened_size=(${sizesV.join(', ')}${sizesV.length === 1 ? ',' : ''})`)
  })

  const padLayer = (kind: string, n: number, mode: C.PadMode, withValue: boolean): Value => layer(kind, withValue ? 'padding value' : 'padding', (a) => {
    const pv = a.padding
    const pad = pv?.k === 'tuple' || pv?.k === 'list' ? pv.items.map((v) => h.toInt(v)) : new Array<number>(2 * n).fill(int(pv, 'padding'))
    if (pad.length !== 2 * n) throw h.err('ValueError', `${kind}: padding must be an int or a tuple of ${2 * n} ints`)
    const value = withValue ? num(a.value, 0) : 0
    return new Activation(kind, (x) => C.padT(E, x, pad, mode, value), withValue ? `padding=(${pad.join(', ')}), value=${repr(a.value ?? py.float(0))}` : `(${pad.join(', ')})`)
  })

  const moduleHost: HostClass = {
    init(inst) {
      inst.state[MODULE_STATE] = { training: true, buffers: new Map<string, boolean>() }
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
      return inst.state[MODULE_STATE] ? moduleRepr(inst) : undefined
    },
  }
  const Module: Value = { k: 'type', name: 'Module', host: moduleHost, call: () => { throw h.err('TypeError', 'nn.Module is a base class: define class Net(nn.Module) with __init__ and forward') } }
  const Dataset: Value = { k: 'type', name: 'Dataset', host: { init() {} }, call: () => { throw h.err('TypeError', 'Dataset is a base class: subclass it with __len__ and __getitem__') } }

  const lossLayer = (name: string, params: string, f: (a: Args) => (x: TT, y: TT, hh: Host) => TT) => layer(name, params, (a) => new LossLayer(name, f(a)))
  const LEGACY_LOSS = 'size_average reduce reduction'

  const nnAttrs: Record<string, Value> = {
    Module,
    Linear: layer('Linear', 'in_features out_features bias', (a) => new Linear(h, int(a.in_features, 'in_features'), int(a.out_features, 'out_features'), bool(a.bias, true))),
    Conv2d: layer('Conv2d', CONV_PARAMS, (a) => makeConv('Conv2d', a)),
    Conv1d: layer('Conv1d', CONV_PARAMS, (a) => makeConv('Conv1d', a)),
    ConvTranspose2d: layer('ConvTranspose2d', 'in_channels out_channels kernel_size stride padding output_padding groups bias dilation padding_mode', (a) => {
      refuse('ConvTranspose2d', a, 'groups', isOne)
      refuse('ConvTranspose2d', a, 'dilation', isOne)
      if (a.padding_mode && h.str(a.padding_mode) !== 'zeros') throw h.err('ValueError', 'Only "zeros" padding mode is supported for ConvTranspose2d')
      return new ConvTranspose2d(h, int(a.in_channels, 'in_channels'), int(a.out_channels, 'out_channels'), pair(h, a.kernel_size, 1), pair(h, a.stride, 1), pair(h, a.padding, 0), pair(h, a.output_padding, 0), bool(a.bias, true))
    }),
    MaxPool2d: layer('MaxPool2d', MAXPOOL, pool2Layer('max')),
    AvgPool2d: layer('AvgPool2d', AVGPOOL, pool2Layer('avg')),
    MaxPool1d: layer('MaxPool1d', MAXPOOL, pool1Layer('max')),
    AvgPool1d: layer('AvgPool1d', AVGPOOL, pool1Layer('avg')),
    ...Object.fromEntries(([['AdaptiveAvgPool2d', 'avg', false], ['AdaptiveMaxPool2d', 'max', false], ['AdaptiveAvgPool1d', 'avg', true], ['AdaptiveMaxPool1d', 'max', true]] as const).map(([name, mode, one]) => [name, layer(name, mode === 'max' ? 'output_size return_indices' : 'output_size', (a) => {
      refuse(name, a, 'return_indices', isFalse)
      return new Activation(name, (x) => adaptive(x, a.output_size, mode, one), `output_size=${repr(a.output_size ?? py.NONE)}`)
    })])),
    ...Object.fromEntries(ACTS.map(([name, , params, make, extra]) => [name, layer(name, params, (a) => new Activation(name, make(a), extra(a), bool(a.inplace, false)))])),
    PReLU: layer('PReLU', 'num_parameters init', (a) => new PReLU(a.num_parameters ? h.toInt(a.num_parameters) : 1, num(a.init, 0.25))),
    Softmax: layer('Softmax', 'dim', (a) => new Activation('Softmax', (x) => C.softmax(E, x, softDim(x, a.dim), false), `dim=${a.dim && a.dim.k !== 'none' ? h.toInt(a.dim) : 'None'}`)),
    LogSoftmax: layer('LogSoftmax', 'dim', (a) => new Activation('LogSoftmax', (x) => C.softmax(E, x, softDim(x, a.dim), true), `dim=${a.dim && a.dim.k !== 'none' ? h.toInt(a.dim) : 'None'}`)),
    Softmin: layer('Softmin', 'dim', (a) => new Activation('Softmin', (x) => C.softmax(E, C.unary(E, 'neg', x), softDim(x, a.dim), false), `dim=${a.dim && a.dim.k !== 'none' ? h.toInt(a.dim) : 'None'}`)),
    Flatten: layer('Flatten', 'start_dim end_dim', (a) => {
      const s = optDim(h, a.start_dim) ?? 1
      const e = optDim(h, a.end_dim) ?? -1
      return new Activation('Flatten', (x) => C.flattenT(E, x, s, e), `start_dim=${s}, end_dim=${e}`)
    }),
    Unflatten,
    Identity: { k: 'type', name: 'Identity', call: () => py.obj(new Activation('Identity', (x) => x)) },
    Dropout: layer('Dropout', 'p inplace', (a) => new DropoutLayer(h, 'Dropout', num(a.p, 0.5), bool(a.inplace, false))),
    Dropout2d: layer('Dropout2d', 'p inplace', (a) => new DropoutLayer(h, 'Dropout2d', num(a.p, 0.5), bool(a.inplace, false))),
    BatchNorm1d: layer('BatchNorm1d', 'num_features eps momentum affine track_running_stats', (a) => new BatchNorm('BatchNorm1d', [2, 3], int(a.num_features, 'num_features'), num(a.eps, 1e-5), a.momentum?.k === 'none' ? null : num(a.momentum, 0.1), bool(a.affine, true), bool(a.track_running_stats, true))),
    BatchNorm2d: layer('BatchNorm2d', 'num_features eps momentum affine track_running_stats', (a) => new BatchNorm('BatchNorm2d', [4], int(a.num_features, 'num_features'), num(a.eps, 1e-5), a.momentum?.k === 'none' ? null : num(a.momentum, 0.1), bool(a.affine, true), bool(a.track_running_stats, true))),
    BatchNorm3d: layer('BatchNorm3d', 'num_features eps momentum affine track_running_stats', (a) => new BatchNorm('BatchNorm3d', [5], int(a.num_features, 'num_features'), num(a.eps, 1e-5), a.momentum?.k === 'none' ? null : num(a.momentum, 0.1), bool(a.affine, true), bool(a.track_running_stats, true))),
    LayerNorm: layer('LayerNorm', 'normalized_shape eps elementwise_affine bias', (a) => new LayerNorm(toDims(h, [a.normalized_shape ?? py.NONE]), num(a.eps, 1e-5), bool(a.elementwise_affine, true), bool(a.bias, true))),
    GroupNorm: layer('GroupNorm', 'num_groups num_channels eps affine', (a) => new GroupNorm(h, int(a.num_groups, 'num_groups'), int(a.num_channels, 'num_channels'), num(a.eps, 1e-5), bool(a.affine, true))),
    Embedding: layer('Embedding', 'num_embeddings embedding_dim padding_idx max_norm norm_type scale_grad_by_freq sparse _weight _freeze', (a) => {
      for (const key of ['max_norm', 'scale_grad_by_freq', '_weight']) refuse('Embedding', a, key, (v) => isNone(v) || isFalse(v))
      return new Embedding(h, int(a.num_embeddings, 'num_embeddings'), int(a.embedding_dim, 'embedding_dim'), a.padding_idx && a.padding_idx.k !== 'none' ? h.toInt(a.padding_idx) : null)
    }),
    ...Object.fromEntries((['RNN', 'LSTM', 'GRU'] as const).map((mode) => [mode, layer(mode, `input_size hidden_size num_layers${mode === 'RNN' ? ' nonlinearity' : ''} bias batch_first dropout bidirectional${mode === 'LSTM' ? ' proj_size' : ''}`, (a) => {
      refuse(mode, a, 'proj_size', (v) => h.num(v) === 0)
      const how = a.nonlinearity ? h.str(a.nonlinearity) : 'tanh'
      if (how !== 'tanh' && how !== 'relu') throw h.err('ValueError', `Unknown nonlinearity '${how}'. Select from 'tanh' or 'relu'.`)
      return new Recurrent(h, mode, how === 'relu', int(a.input_size, 'input_size'), int(a.hidden_size, 'hidden_size'), a.num_layers ? h.toInt(a.num_layers) : 1, bool(a.bias, true), bool(a.batch_first, false), num(a.dropout, 0), bool(a.bidirectional, false))
    })])),
    RNNCell: layer('RNNCell', 'input_size hidden_size bias nonlinearity', (a) => new RecurrentCell(h, 'RNNCell', 'RNN', !!a.nonlinearity && h.str(a.nonlinearity) === 'relu', int(a.input_size, 'input_size'), int(a.hidden_size, 'hidden_size'), bool(a.bias, true))),
    LSTMCell: layer('LSTMCell', 'input_size hidden_size bias', (a) => new RecurrentCell(h, 'LSTMCell', 'LSTM', false, int(a.input_size, 'input_size'), int(a.hidden_size, 'hidden_size'), bool(a.bias, true))),
    GRUCell: layer('GRUCell', 'input_size hidden_size bias', (a) => new RecurrentCell(h, 'GRUCell', 'GRU', false, int(a.input_size, 'input_size'), int(a.hidden_size, 'hidden_size'), bool(a.bias, true))),
    MultiheadAttention: layer('MultiheadAttention', 'embed_dim num_heads dropout bias add_bias_kv add_zero_attn kdim vdim batch_first', (a) => {
      for (const key of ['add_bias_kv', 'add_zero_attn', 'kdim', 'vdim']) refuse('MultiheadAttention', a, key, (v) => isNone(v) || isFalse(v))
      return new MultiheadAttention(h, int(a.embed_dim, 'embed_dim'), int(a.num_heads, 'num_heads'), num(a.dropout, 0), bool(a.bias, true), bool(a.batch_first, false))
    }),
    Upsample: layer('Upsample', 'size scale_factor mode align_corners recompute_scale_factor', (a) => {
      refuse('Upsample', a, 'recompute_scale_factor', (v) => isNone(v))
      const mode = a.mode ? h.str(a.mode) : 'nearest'
      const scale = a.scale_factor && a.scale_factor.k !== 'none' ? a.scale_factor : null
      const shown = scale ? (scale.k === 'tuple' ? py.tuple(scale.items.map((v) => py.float(h.num(v)))) : py.float(h.num(scale))) : a.size ?? py.NONE
      return new Activation('Upsample', (x) => interpolate(x, a.size, a.scale_factor, mode, a.align_corners), `${scale ? 'scale_factor' : 'size'}=${repr(shown)}, mode=${repr(py.str(mode))}`)
    }),
    ZeroPad1d: padLayer('ZeroPad1d', 1, 'constant', false),
    ZeroPad2d: padLayer('ZeroPad2d', 2, 'constant', false),
    ConstantPad1d: padLayer('ConstantPad1d', 1, 'constant', true),
    ConstantPad2d: padLayer('ConstantPad2d', 2, 'constant', true),
    ReflectionPad1d: padLayer('ReflectionPad1d', 1, 'reflect', false),
    ReflectionPad2d: padLayer('ReflectionPad2d', 2, 'reflect', false),
    ReplicationPad1d: padLayer('ReplicationPad1d', 1, 'replicate', false),
    ReplicationPad2d: padLayer('ReplicationPad2d', 2, 'replicate', false),
    CircularPad1d: padLayer('CircularPad1d', 1, 'circular', false),
    CircularPad2d: padLayer('CircularPad2d', 2, 'circular', false),
    Sequential: { k: 'type', name: 'Sequential', call: (args) => {
      if (args.length === 1 && args[0].k === 'dict') return py.obj(new Sequential(args[0].d.values(), args[0].d.keys().map((k) => h.str(k))))
      return py.obj(new Sequential([...args]))
    } },
    ModuleList: { k: 'type', name: 'ModuleList', call: (args) => py.obj(new ModuleList(args[0] ? [...h.iterate(args[0])] : [])) },
    ModuleDict: { k: 'type', name: 'ModuleDict', call: (args) => {
      const m = new ModuleDict()
      if (args[0]) m.update(args[0], h)
      return py.obj(m)
    } },
    CrossEntropyLoss: lossLayer('CrossEntropyLoss', `weight size_average ignore_index reduce reduction label_smoothing`, (a) => {
      const o = ceOpts(a)
      return (x, y, hh) => ceFull(hh, x, y, o)
    }),
    NLLLoss: lossLayer('NLLLoss', 'weight size_average ignore_index reduce reduction', (a) => {
      const o = ceOpts(a)
      return (x, y, hh) => nllFull(hh, x, y, o)
    }),
    MSELoss: lossLayer('MSELoss', LEGACY_LOSS, (a) => {
      const r = reduction(a)
      return (x, y, hh) => mse(hh, x, y, r)
    }),
    BCELoss: lossLayer('BCELoss', `weight ${LEGACY_LOSS}`, (a) => {
      const r = reduction(a)
      return (x, y, hh) => bceWeighted(hh, x, y, false, optT(a.weight), null, r)
    }),
    BCEWithLogitsLoss: lossLayer('BCEWithLogitsLoss', `weight ${LEGACY_LOSS} pos_weight`, (a) => {
      const r = reduction(a)
      return (x, y, hh) => bceWeighted(hh, x, y, true, optT(a.weight), a.pos_weight && a.pos_weight.k !== 'none' ? T(a.pos_weight, 'pos_weight') : null, r)
    }),
    L1Loss: lossLayer('L1Loss', LEGACY_LOSS, (a) => {
      const r = reduction(a)
      return (x, y, hh) => l1(hh, x, y, r)
    }),
    SmoothL1Loss: lossLayer('SmoothL1Loss', `${LEGACY_LOSS} beta`, (a) => {
      const r = reduction(a)
      return (x, y, hh) => smoothL1(hh, x, y, num(a.beta, 1), r)
    }),
    HuberLoss: lossLayer('HuberLoss', 'reduction delta', (a) => {
      const r = reduction(a)
      return (x, y, hh) => huber(hh, x, y, num(a.delta, 1), r)
    }),
    CosineSimilarity: layer('CosineSimilarity', 'dim eps', (a) => new LossLayer('CosineSimilarity', (x, y) => cosineSim(x, y, a.dim ? h.toInt(a.dim) : 1, num(a.eps, 1e-8)))),
    Parameter: { k: 'type', name: 'Parameter', call: (args, kw) => {
      const v = args[0] ?? kw.data
      const src = T(v, 'Parameter')
      // shares memory with the tensor it wraps, as in PyTorch
      const t = new TT(src.a, src.dtype)
      t.isParam = true
      t.requiresGrad = flagOf(h, kw.requires_grad ?? args[1], true)
      if (t.requiresGrad && !isFloatT(t.dtype)) throw h.err('RuntimeError', 'Only Tensors of floating point and complex dtype can require gradients')
      return wrap(t)
    } },
  }

  const cosineSim = (x: TT, y: TT, dim: number, eps: number): TT => {
    const shape = broadcastShapes([x.shape, y.shape])
    // each side divided by its norm first, as PyTorch does for numerical stability
    const unit = (t: TT) => {
      const a = C.expandT(E, t, shape)
      return C.binary(E, '/', a, C.clamp(E, C.vectorNorm(E, a, 2, [dim], true), eps, null))
    }
    return C.reduceSum(E, C.binary(E, '*', unit(x), unit(y)), [dim], false, false)
  }

  // ---- torch.nn.functional
  /** F.x: its parameter names after the input; each call is a step and a row of the shape trace */
  const Ff = (name: string, params: string, run: (x: TT, a: Args) => TT, kind: ApiKind = 'elementwise') => fnv(name, (args, kw) => {
    const a = bind(h, name, `input ${params}`, args, kw)
    const t = T(a.input, name)
    const out = run(t, a)
    trace(h, `F.${name}`, kind, [{ label: label(h, a.input!, 'x'), t }], out)
    flowStep(h, `F.${name}`, t, out)
    return wrap(out)
  })
  const Floss = (name: string, params: string, run: (x: TT, y: TT, a: Args) => TT) => fnv(name, (args, kw) => {
    const a = bind(h, name, `input target ${params}`, args, kw)
    return wrap(run(T(a.input, name), T(a.target, `${name} target`), a))
  })
  const Fattrs: Record<string, Value> = {
    ...Object.fromEntries(ACTS.map(([, name, params, make]) => [name, Ff(name, params, (x, a) => (bool(a.inplace, false) ? asTT(objOf(x).rebase(h, make(a)))! : make(a)(x)))])),
    softmax: Ff('softmax', 'dim _stacklevel dtype', (x, a) => C.softmax(E, dtypeArg(h, a.dtype) ? C.toDtype(E, x, dtypeArg(h, a.dtype)!) : x, softDim(x, a.dim), false)),
    log_softmax: Ff('log_softmax', 'dim _stacklevel dtype', (x, a) => C.softmax(E, dtypeArg(h, a.dtype) ? C.toDtype(E, x, dtypeArg(h, a.dtype)!) : x, softDim(x, a.dim), true)),
    softmin: Ff('softmin', 'dim _stacklevel dtype', (x, a) => C.softmax(E, C.unary(E, 'neg', dtypeArg(h, a.dtype) ? C.toDtype(E, x, dtypeArg(h, a.dtype)!) : x), softDim(x, a.dim), false)),
    dropout: Ff('dropout', 'p training inplace', (x, a) => {
      const f = (t: TT) => C.dropout(E, t, num(a.p, 0.5), bool(a.training, true))
      return bool(a.inplace, false) ? asTT(objOf(x).rebase(h, f))! : f(x)
    }, 'random'),
    dropout2d: Ff('dropout2d', 'p training inplace', (x, a) => {
      const f = (t: TT) => C.dropoutChannels(E, t, num(a.p, 0.5), bool(a.training, true))
      return bool(a.inplace, false) ? asTT(objOf(x).rebase(h, f))! : f(x)
    }, 'random'),
    max_pool2d: pool2F('max_pool2d', 'max'),
    avg_pool2d: pool2F('avg_pool2d', 'avg'),
    max_pool1d: pool1F('max_pool1d', 'max'),
    avg_pool1d: pool1F('avg_pool1d', 'avg'),
    adaptive_avg_pool2d: Ff('adaptive_avg_pool2d', 'output_size', (x, a) => adaptive(x, a.output_size, 'avg', false), 'reduce'),
    adaptive_max_pool2d: Ff('adaptive_max_pool2d', 'output_size return_indices', (x, a) => {
      refuse('adaptive_max_pool2d', a, 'return_indices', isFalse)
      return adaptive(x, a.output_size, 'max', false)
    }, 'reduce'),
    adaptive_avg_pool1d: Ff('adaptive_avg_pool1d', 'output_size', (x, a) => adaptive(x, a.output_size, 'avg', true), 'reduce'),
    adaptive_max_pool1d: Ff('adaptive_max_pool1d', 'output_size return_indices', (x, a) => {
      refuse('adaptive_max_pool1d', a, 'return_indices', isFalse)
      return adaptive(x, a.output_size, 'max', true)
    }, 'reduce'),
    conv2d: fnv('conv2d', (args, kw) => {
      const a = bind(h, 'conv2d', 'input weight bias stride padding dilation groups', args, kw)
      const x = T(a.input, 'conv2d')
      const w = T(a.weight, 'conv2d weight')
      const o: ConvOpts = convOpts(a, false)
      const out = convolve(h, x, w, a.bias && a.bias.k !== 'none' ? asTT(a.bias) : null, o, false)
      if (typeof o.padding !== 'string' && o.groups === 1 && o.dilation[0] === 1 && o.dilation[1] === 1) traceConv(h, x, w, out, o.stride, o.padding, label(h, a.input!, 'input'), label(h, a.weight!, 'weight'))
      flowStep(h, 'F.conv2d', x, out)
      return wrap(out)
    }),
    conv1d: fnv('conv1d', (args, kw) => {
      const a = bind(h, 'conv1d', 'input weight bias stride padding dilation groups', args, kw)
      const x = T(a.input, 'conv1d')
      const out = convolve(h, x, T(a.weight, 'conv1d weight'), a.bias && a.bias.k !== 'none' ? asTT(a.bias) : null, convOpts(a, true), true)
      flowStep(h, 'F.conv1d', x, out)
      return wrap(out)
    }),
    conv_transpose2d: fnv('conv_transpose2d', (args, kw) => {
      const a = bind(h, 'conv_transpose2d', 'input weight bias stride padding output_padding groups dilation', args, kw)
      refuse('F.conv_transpose2d', a, 'groups', isOne)
      refuse('F.conv_transpose2d', a, 'dilation', isOne)
      const x = T(a.input, 'conv_transpose2d')
      const out = C.convTranspose2d(E, x, T(a.weight, 'conv_transpose2d weight'), a.bias && a.bias.k !== 'none' ? asTT(a.bias) : null, pair(h, a.stride, 1), pair(h, a.padding, 0), pair(h, a.output_padding, 0))
      flowStep(h, 'F.conv_transpose2d', x, out)
      return wrap(out)
    }),
    linear: fnv('linear', (args, kw) => {
      const a = bind(h, 'linear', 'input weight bias', args, kw)
      const x = T(a.input, 'linear')
      const w = T(a.weight, 'linear weight')
      sameDtype(h, x, w)
      return wrap(C.linear(E, x, w, a.bias && a.bias.k !== 'none' ? asTT(a.bias) : null))
    }),
    cross_entropy: Floss('cross_entropy', 'weight size_average ignore_index reduce reduction label_smoothing', (x, y, a) => ceFull(h, x, y, ceOpts(a))),
    nll_loss: Floss('nll_loss', 'weight size_average ignore_index reduce reduction', (x, y, a) => nllFull(h, x, y, ceOpts(a))),
    mse_loss: Floss('mse_loss', `${LEGACY_LOSS} weight`, (x, y, a) => {
      if (a.weight && a.weight.k !== 'none') throw h.err('NotImplementedError', 'mse_loss(): weight= is not available in this sandbox')
      return mse(h, x, y, reduction(a))
    }),
    l1_loss: Floss('l1_loss', LEGACY_LOSS, (x, y, a) => l1(h, x, y, reduction(a))),
    smooth_l1_loss: Floss('smooth_l1_loss', `${LEGACY_LOSS} beta`, (x, y, a) => smoothL1(h, x, y, num(a.beta, 1), reduction(a))),
    huber_loss: Floss('huber_loss', 'reduction delta', (x, y, a) => huber(h, x, y, num(a.delta, 1), reduction(a))),
    binary_cross_entropy: Floss('binary_cross_entropy', `weight ${LEGACY_LOSS}`, (x, y, a) => bceWeighted(h, x, y, false, optT(a.weight), null, reduction(a))),
    binary_cross_entropy_with_logits: Floss('binary_cross_entropy_with_logits', `weight ${LEGACY_LOSS} pos_weight`, (x, y, a) => bceWeighted(h, x, y, true, optT(a.weight), a.pos_weight && a.pos_weight.k !== 'none' ? T(a.pos_weight, 'pos_weight') : null, reduction(a))),
    one_hot: fnv('one_hot', (args, kw) => {
      const a = bind(h, 'one_hot', 'tensor num_classes', args, kw)
      const t = T(a.tensor, 'one_hot')
      if (t.dtype !== 'int64') throw h.err('RuntimeError', 'one_hot is only applicable to index tensor of type LongTensor.')
      const v = t.values()
      if (v.some((c) => c < 0)) throw h.err('RuntimeError', 'Class values must be non-negative.')
      const given = a.num_classes ? h.toInt(a.num_classes) : -1
      const n = given === -1 ? v.reduce((m, x) => Math.max(m, x), -1) + 1 : given
      if (v.some((c) => c >= n)) throw h.err('RuntimeError', 'Class values must be smaller than num_classes.')
      return wrap(TT.of(v.flatMap((c) => Array.from({ length: n }, (_, j) => (j === c ? 1 : 0))), [...t.shape, n], 'int64'))
    }),
    normalize: Ff('normalize', 'p dim eps', (x, a) => {
      // input / max(‖input‖_p, eps), as PyTorch composes it
      const denom = C.clamp(E, C.vectorNorm(E, x, num(a.p, 2), [a.dim ? h.toInt(a.dim) : 1], true), num(a.eps, 1e-12), null)
      return C.binary(E, '/', x, C.expandT(E, denom, x.shape))
    }),
    cosine_similarity: fnv('cosine_similarity', (args, kw) => {
      const a = bind(h, 'cosine_similarity', 'x1 x2 dim eps', args, kw)
      return wrap(cosineSim(T(a.x1, 'cosine_similarity'), T(a.x2, 'cosine_similarity'), a.dim ? h.toInt(a.dim) : 1, num(a.eps, 1e-8)))
    }),
    pairwise_distance: fnv('pairwise_distance', (args, kw) => {
      const a = bind(h, 'pairwise_distance', 'x1 x2 p eps keepdim', args, kw)
      const x1 = T(a.x1, 'pairwise_distance')
      const d = C.binary(E, '+', C.binary(E, '-', x1, T(a.x2, 'pairwise_distance')), TT.scalar(num(a.eps, 1e-6), x1.dtype), false, true)
      return wrap(C.vectorNorm(E, d, num(a.p, 2), [-1], bool(a.keepdim, false)))
    }),
    pad: Ff('pad', 'pad mode value', (x, a) => padOf(x, a.pad, a.mode, a.value), 'group'),
    interpolate: Ff('interpolate', 'size scale_factor mode align_corners recompute_scale_factor antialias', (x, a) => {
      refuse('F.interpolate', a, 'recompute_scale_factor', isNone)
      refuse('F.interpolate', a, 'antialias', isFalse)
      return interpolate(x, a.size, a.scale_factor, a.mode ? h.str(a.mode) : 'nearest', a.align_corners)
    }, 'group'),
    embedding: fnv('embedding', (args, kw) => {
      const a = bind(h, 'embedding', 'input weight padding_idx max_norm norm_type scale_grad_by_freq sparse', args, kw)
      refuse('F.embedding', a, 'max_norm', isNone)
      refuse('F.embedding', a, 'scale_grad_by_freq', isFalse)
      return wrap(C.embeddingT(E, T(a.weight, 'embedding weight'), T(a.input, 'embedding'), a.padding_idx && a.padding_idx.k !== 'none' ? h.toInt(a.padding_idx) : null))
    }),
  }

  // ---- torch.nn.init
  const fans = (t: TT): [number, number] => {
    if (t.ndim < 2) throw h.err('ValueError', 'Fan in and fan out can not be computed for tensor with fewer than 2 dimensions')
    const rest = prod(t.shape.slice(2))
    return [t.shape[1] * rest, t.shape[0] * rest]
  }
  const gainOf = (nonlinearity: string, param: Value | undefined): number => {
    if (['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d', 'sigmoid'].includes(nonlinearity)) return 1
    if (nonlinearity === 'tanh') return 5 / 3
    if (nonlinearity === 'relu') return Math.SQRT2
    if (nonlinearity === 'selu') return 3 / 4
    if (nonlinearity === 'leaky_relu') return Math.sqrt(2 / (1 + (param && param.k !== 'none' ? h.num(param) : 0.01) ** 2))
    throw h.err('ValueError', `Unsupported nonlinearity ${nonlinearity}`)
  }
  const fanOf = (t: TT, mode: Value | undefined): number => {
    const m = mode ? h.str(mode) : 'fan_in'
    if (m !== 'fan_in' && m !== 'fan_out') throw h.err('ValueError', `Mode ${m} not supported, please use one of ['fan_in', 'fan_out']`)
    return fans(t)[m === 'fan_in' ? 0 : 1]
  }
  /** nn.init.x_: writes under no_grad, so it may touch a parameter; returns the tensor */
  const initFn = (name: string, params: string, value: (t: TT, a: Args, rng: SandboxRandom) => () => number) => fnv(name, (args, kw) => {
    const a = bind(h, name, `tensor ${params} generator`, args, kw)
    const t = T(a.tensor, `nn.init.${name}`)
    const next = value(t, a, rngOf(h, a.generator))
    C.writeInto(t, t.values().map(() => next()))
    trace(h, `nn.init.${name}`, 'random', [], t)
    return wrap(t)
  })
  const uniformBound = (rng: SandboxRandom, b: number) => () => (rng.next() * 2 - 1) * b
  const normalStd = (rng: SandboxRandom, std: number) => () => std * rng.normal()
  const initAttrs: Record<string, Value> = {
    calculate_gain: fnv('calculate_gain', (args, kw) => {
      const g = gainOf(h.str(args[0] ?? kw.nonlinearity), kw.param ?? args[1])
      return g === 1 ? py.int(1) : py.float(g)
    }),
    uniform_: initFn('uniform_', 'a b', (_t, a, rng) => {
      const [lo, hi] = [num(a.a, 0), num(a.b, 1)]
      return () => lo + (hi - lo) * rng.next()
    }),
    normal_: initFn('normal_', 'mean std', (_t, a, rng) => {
      const [mean, std] = [num(a.mean, 0), num(a.std, 1)]
      return () => mean + std * rng.normal()
    }),
    trunc_normal_: initFn('trunc_normal_', 'mean std a b', (_t, a, rng) => {
      const [mean, std, lo, hi] = [num(a.mean, 0), num(a.std, 1), num(a.a, -2), num(a.b, 2)]
      return () => {
        for (let i = 0; i < 1000; i++) {
          const v = mean + std * rng.normal()
          if (v >= lo && v <= hi) return v
        }
        return Math.min(hi, Math.max(lo, mean))
      }
    }),
    constant_: initFn('constant_', 'val', (_t, a) => {
      const v = h.num(a.val ?? py.NONE)
      return () => v
    }),
    zeros_: initFn('zeros_', '', () => () => 0),
    ones_: initFn('ones_', '', () => () => 1),
    eye_: fnv('eye_', (args) => {
      const t = T(args[0], 'nn.init.eye_')
      if (t.ndim !== 2) throw h.err('ValueError', 'Only tensors with 2 dimensions are supported')
      const [r, c] = t.shape
      C.writeInto(t, Array.from({ length: r * c }, (_, f) => (Math.floor(f / c) === f % c ? 1 : 0)))
      return wrap(t)
    }),
    xavier_uniform_: initFn('xavier_uniform_', 'gain', (t, a, rng) => {
      const [fi, fo] = fans(t)
      return uniformBound(rng, Math.sqrt(3) * num(a.gain, 1) * Math.sqrt(2 / (fi + fo)))
    }),
    xavier_normal_: initFn('xavier_normal_', 'gain', (t, a, rng) => {
      const [fi, fo] = fans(t)
      return normalStd(rng, num(a.gain, 1) * Math.sqrt(2 / (fi + fo)))
    }),
    kaiming_uniform_: initFn('kaiming_uniform_', 'a mode nonlinearity', (t, a, rng) => uniformBound(rng, (Math.sqrt(3) * gainOf(a.nonlinearity ? h.str(a.nonlinearity) : 'leaky_relu', a.a ?? py.int(0))) / Math.sqrt(fanOf(t, a.mode)))),
    kaiming_normal_: initFn('kaiming_normal_', 'a mode nonlinearity', (t, a, rng) => normalStd(rng, gainOf(a.nonlinearity ? h.str(a.nonlinearity) : 'leaky_relu', a.a ?? py.int(0)) / Math.sqrt(fanOf(t, a.mode)))),
  }
  const nnInit: Value = { k: 'module', name: 'torch.nn.init', attrs: initAttrs }

  // ---- torch.nn.utils
  const gradsOf = (params: Value): TT[] => (asTT(params) ? [asTT(params)!] : h.iterate(params).map((p) => T(p, 'parameter'))).filter((p) => p.grad)
  const nnUtils: Value = {
    k: 'module', name: 'torch.nn.utils', attrs: {
      clip_grad_norm_: fnv('clip_grad_norm_', (args, kw) => {
        const a = bind(h, 'clip_grad_norm_', 'parameters max_norm norm_type error_if_nonfinite foreach', args, kw)
        const ps = gradsOf(a.parameters!)
        const p = num(a.norm_type, 2)
        const norms = ps.map((t) => C.vectorNorm({ grad: false } as Engine, t.grad!, p, null, false).values()[0])
        const dtype = ps[0]?.dtype ?? 'float32'
        const total = C.vectorNorm({ grad: false } as Engine, TT.of(norms, [norms.length], dtype), p, null, false)
        const value = total.values()[0]
        if (bool(a.error_if_nonfinite, false) && !Number.isFinite(value)) throw h.err('RuntimeError', `The total norm of order ${p} for gradients from \`parameters\` is non-finite, so it cannot be clipped. To disable this error and scale the gradients by the non-finite norm anyway, set \`error_if_nonfinite=False\``)
        const coef = Math.min(1, h.num(a.max_norm ?? py.NONE) / (value + 1e-6))
        for (const t of ps) t.grad = TT.of(t.grad!.values().map((g) => g * coef), t.shape, t.dtype)
        return wrap(total)
      }),
      clip_grad_value_: fnv('clip_grad_value_', (args, kw) => {
        const a = bind(h, 'clip_grad_value_', 'parameters clip_value foreach', args, kw)
        const c = h.num(a.clip_value ?? py.NONE)
        for (const t of gradsOf(a.parameters!)) t.grad = TT.of(t.grad!.values().map((g) => Math.min(c, Math.max(-c, g))), t.shape, t.dtype)
        return py.NONE
      }),
      parameters_to_vector: fnv('parameters_to_vector', (args) => wrap(C.catT(E, h.iterate(args[0]).map((p) => C.reshapeT(E, T(p, 'parameter'), [-1], false)), 0))),
    },
  }

  // ---- optim
  const defaultsOf = (a: Args, spec: [string, Value][]): [string, Value][] => spec.map(([k, d]) => [k, a[k] ?? d])
  const nonNegative = (a: Args, key: string, what: string) => {
    const v = a[key]
    if (v && v.k !== 'tuple' && h.num(v) < 0) throw h.err('ValueError', `Invalid ${what}: ${repr(v)}`)
  }
  const checkBetas = (a: Args) => {
    const b = a.betas ? h.iterate(a.betas).map((x) => h.num(x)) : []
    b.forEach((x, i) => {
      if (x < 0 || x >= 1) throw h.err('ValueError', `Invalid beta parameter at index ${i}: ${repr(py.float(x))}`)
    })
  }
  const noDifferentiable = (who: string, a: Args) => refuse(who, a, 'differentiable', isFalse)
  const optimizer = (name: string, params: string, spec: [string, Value][], check: (a: Args) => void): Value => ({ k: 'type', name, call: (args, kw) => {
    const a = bind(h, name, `params ${params}`, args, kw)
    if (!a.params) throw h.err('TypeError', `${name}.__init__() missing 1 required positional argument: 'params'`)
    nonNegative(a, 'lr', 'learning rate')
    nonNegative(a, 'weight_decay', 'weight_decay value')
    nonNegative(a, 'eps', 'epsilon value')
    check(a)
    noDifferentiable(name, a)
    return py.obj(new Optimizer(name, h, a.params, defaultsOf(a, spec)))
  } })
  const F0 = py.bool(false)
  const betas = py.tuple([py.float(0.9), py.float(0.999)])
  const ADAM_PARAMS = 'lr betas eps weight_decay amsgrad foreach maximize capturable differentiable fused decoupled_weight_decay'
  const adamSpec = (wd: Value, decoupled: boolean): [string, Value][] => [['lr', py.float(0.001)], ['betas', betas], ['eps', py.float(1e-8)], ['weight_decay', wd], ['amsgrad', F0], ['maximize', F0], ['foreach', py.NONE], ['capturable', F0], ['differentiable', F0], ['fused', py.NONE], ['decoupled_weight_decay', py.bool(decoupled)]]
  const optim: Record<string, Value> = {
    SGD: optimizer('SGD', 'lr momentum dampening weight_decay nesterov maximize foreach differentiable fused', [['lr', py.float(0.001)], ['momentum', py.int(0)], ['dampening', py.int(0)], ['weight_decay', py.int(0)], ['nesterov', F0], ['maximize', F0], ['foreach', py.NONE], ['differentiable', F0], ['fused', py.NONE]], (a) => {
      nonNegative(a, 'momentum', 'momentum value')
      if (bool(a.nesterov, false) && (num(a.momentum, 0) <= 0 || num(a.dampening, 0) !== 0)) throw h.err('ValueError', 'Nesterov momentum requires a momentum and zero dampening')
    }),
    Adam: optimizer('Adam', ADAM_PARAMS, adamSpec(py.int(0), false), checkBetas),
    AdamW: optimizer('AdamW', ADAM_PARAMS, adamSpec(py.float(0.01), true), checkBetas),
    RMSprop: optimizer('RMSprop', 'lr alpha eps weight_decay momentum centered capturable foreach maximize differentiable', [['lr', py.float(0.01)], ['momentum', py.int(0)], ['alpha', py.float(0.99)], ['eps', py.float(1e-8)], ['centered', F0], ['weight_decay', py.int(0)], ['capturable', F0], ['foreach', py.NONE], ['maximize', F0], ['differentiable', F0]], (a) => {
      nonNegative(a, 'momentum', 'momentum value')
      nonNegative(a, 'alpha', 'alpha value')
    }),
    Adagrad: optimizer('Adagrad', 'lr lr_decay weight_decay initial_accumulator_value eps foreach maximize differentiable fused', [['lr', py.float(0.01)], ['lr_decay', py.int(0)], ['eps', py.float(1e-10)], ['weight_decay', py.int(0)], ['initial_accumulator_value', py.int(0)], ['foreach', py.NONE], ['maximize', F0], ['differentiable', F0], ['fused', py.NONE]], (a) => {
      nonNegative(a, 'lr_decay', 'lr_decay value')
      nonNegative(a, 'initial_accumulator_value', 'initial_accumulator_value value')
    }),
  }
  // ---- torch.optim.lr_scheduler: each rule works from the current lr, as PyTorch's chainable form does
  const schedType = (name: string, params: string, rule: (a: Args) => (lr: number, base: number, epoch: number) => number): Value => ({ k: 'type', name, call: (args, kw) => {
    const a = bind(h, name, `optimizer ${params} last_epoch verbose`, args, kw)
    refuse(name, a, 'last_epoch', (v) => h.num(v) === -1)
    const o = a.optimizer
    if (o?.k !== 'obj' || !(o.o instanceof Optimizer)) throw h.err('TypeError', `${o ? typeName(o) : 'nothing'} is not an Optimizer`)
    return py.obj(new Scheduler(name, h, o.o, rule(a)))
  } })
  const pyMod = (a: number, b: number) => ((a % b) + b) % b
  const lrScheduler: Value = {
    k: 'module', name: 'torch.optim.lr_scheduler', attrs: {
      LRScheduler: { k: 'type', name: 'LRScheduler', call: () => { throw h.err('TypeError', 'LRScheduler is a base class') } },
      StepLR: schedType('StepLR', 'step_size gamma', (a) => {
        const size = int(a.step_size, 'step_size')
        const gamma = num(a.gamma, 0.1)
        return (lr, _b, e) => (e === 0 || e % size !== 0 ? lr : lr * gamma)
      }),
      MultiStepLR: schedType('MultiStepLR', 'milestones gamma', (a) => {
        const gamma = num(a.gamma, 0.1)
        const counts = new Map<number, number>()
        for (const m of h.iterate(a.milestones ?? py.NONE).map((v) => h.toInt(v))) counts.set(m, (counts.get(m) ?? 0) + 1)
        return (lr, _b, e) => (counts.has(e) ? lr * gamma ** counts.get(e)! : lr)
      }),
      ExponentialLR: schedType('ExponentialLR', 'gamma', (a) => {
        const gamma = h.num(a.gamma ?? py.NONE)
        return (lr, _b, e) => (e === 0 ? lr : lr * gamma)
      }),
      CosineAnnealingLR: schedType('CosineAnnealingLR', 'T_max eta_min', (a) => {
        const T_max = int(a.T_max, 'T_max')
        const eta = num(a.eta_min, 0)
        return (lr, base, e) => {
          if (e === 0) return lr
          if (pyMod(e - 1 - T_max, 2 * T_max) === 0) return lr + ((base - eta) * (1 - Math.cos(Math.PI / T_max))) / 2
          return ((1 + Math.cos((Math.PI * e) / T_max)) / (1 + Math.cos((Math.PI * (e - 1)) / T_max))) * (lr - eta) + eta
        }
      }),
      LambdaLR: schedType('LambdaLR', 'lr_lambda', (a) => (_lr, base, e) => base * h.num(h.call(a.lr_lambda ?? py.NONE, [py.int(e)]))),
      ReduceLROnPlateau: { k: 'type', name: 'ReduceLROnPlateau', call: (args, kw) => {
        const a = bind(h, 'ReduceLROnPlateau', 'optimizer mode factor patience threshold threshold_mode cooldown min_lr eps verbose', args, kw)
        const o = a.optimizer
        if (o?.k !== 'obj' || !(o.o instanceof Optimizer)) throw h.err('TypeError', `${o ? typeName(o) : 'nothing'} is not an Optimizer`)
        const mode = a.mode ? h.str(a.mode) : 'min'
        if (mode !== 'min' && mode !== 'max') throw h.err('ValueError', `mode ${mode} is unknown!`)
        const tm = a.threshold_mode ? h.str(a.threshold_mode) : 'rel'
        if (tm !== 'rel' && tm !== 'abs') throw h.err('ValueError', `threshold mode ${tm} is unknown!`)
        return py.obj(new ReduceLROnPlateau(h, o.o, mode === 'max', num(a.factor, 0.1), a.patience ? h.toInt(a.patience) : 10, num(a.threshold, 1e-4), tm === 'abs', a.cooldown ? h.toInt(a.cooldown) : 0, a.min_lr, num(a.eps, 1e-8)))
      } },
    },
  }

  const dtypeVals = Object.fromEntries(Object.entries(DTYPES).map(([k, d]) => [k, py.obj(new DtypeObj(d))]))
  const cuda: Value = {
    k: 'module', name: 'torch.cuda', attrs: {
      is_available: fnv('is_available', () => py.bool(false)),
      device_count: fnv('device_count', () => py.int(0)),
      current_device: fnv('current_device', () => { throw h.err('AssertionError', 'Torch not compiled with CUDA enabled') }),
      manual_seed: fnv('manual_seed', () => py.NONE),
      manual_seed_all: fnv('manual_seed_all', () => py.NONE),
    },
  }

  // ---- torch.utils.data
  const samplerType = (name: string, params: string, make: (a: Args) => [(hh: Host) => number[], (hh: Host) => number]): Value => ({ k: 'type', name, call: (args, kw) => {
    const [draw, size] = make(bind(h, name, params, args, kw))
    return py.obj(new SamplerObj(name, draw, size))
  } })
  const data: Value = {
    k: 'module', name: 'torch.utils.data', attrs: {
      TensorDataset: { k: 'type', name: 'TensorDataset', call: (args) => {
        const ts = args.map((a) => T(a, 'TensorDataset'))
        if (ts.some((t) => t.shape[0] !== ts[0].shape[0])) throw h.err('AssertionError', 'Size mismatch between tensors')
        return py.obj(new TensorDataset(ts))
      } },
      DataLoader: { k: 'type', name: 'DataLoader', call: (args, kw) => {
        const a = bind(h, 'DataLoader', 'dataset batch_size shuffle sampler batch_sampler num_workers collate_fn pin_memory drop_last timeout worker_init_fn multiprocessing_context generator prefetch_factor persistent_workers pin_memory_device', args, kw)
        if (!a.dataset) throw h.err('TypeError', "DataLoader.__init__() missing 1 required positional argument: 'dataset'")
        refuse('DataLoader', a, 'batch_sampler', isNone)
        if (a.batch_size?.k === 'none') throw h.err('NotImplementedError', 'DataLoader: batch_size=None is not available in this sandbox')
        const bs = a.batch_size ? h.toInt(a.batch_size) : 1
        if (bs <= 0) throw h.err('ValueError', `batch_size should be a positive integer value, but got batch_size=${bs}`)
        const sampler = a.sampler && a.sampler.k !== 'none' ? a.sampler : null
        if (sampler && bool(a.shuffle, false)) throw h.err('ValueError', 'sampler option is mutually exclusive with shuffle')
        return py.obj(new DataLoader(a.dataset, bs, bool(a.shuffle, false), bool(a.drop_last, false), sampler, a.collate_fn && a.collate_fn.k !== 'none' ? a.collate_fn : null, rngOf(h, a.generator)))
      } },
      Dataset,
      Subset: { k: 'type', name: 'Subset', call: (args, kw) => py.obj(new Subset(args[0] ?? kw.dataset, h.iterate(args[1] ?? kw.indices).map((v) => h.toInt(v)))) },
      ConcatDataset: { k: 'type', name: 'ConcatDataset', call: (args, kw) => py.obj(new ConcatDataset([...h.iterate(args[0] ?? kw.datasets)])) },
      random_split: fnv('random_split', (args, kw) => {
        const a = bind(h, 'random_split', 'dataset lengths generator', args, kw)
        const ds = a.dataset!
        const n = datasetLen(h, ds)
        let lens = h.iterate(a.lengths ?? py.NONE).map((v) => h.num(v))
        const total = lens.reduce((s, x) => s + x, 0)
        if (Math.abs(total - 1) <= 1e-9 && total <= 1) {
          lens.forEach((f, i) => {
            if (f < 0 || f > 1) throw h.err('ValueError', `Fraction at index ${i} is not between 0 and 1`)
          })
          const counts = lens.map((f) => Math.floor(n * f))
          const remainder = n - counts.reduce((s, x) => s + x, 0)
          for (let i = 0; i < remainder; i++) counts[i % counts.length]++
          lens = counts
        }
        if (lens.reduce((s, x) => s + x, 0) !== n) throw h.err('ValueError', 'Sum of input lengths does not equal the length of the input dataset!')
        const order = shuffled(rngOf(h, a.generator), n)
        let at = 0
        return py.list(lens.map((len) => {
          at += len
          return py.obj(new Subset(ds, order.slice(at - len, at)))
        }))
      }),
      default_collate: fnv('default_collate', (args) => collate(h, h.iterate(args[0]))),
      Sampler: { k: 'type', name: 'Sampler', call: () => { throw h.err('TypeError', 'Sampler is a base class') } },
      SequentialSampler: samplerType('SequentialSampler', 'data_source', (a) => [(hh) => Array.from({ length: datasetLen(hh, a.data_source!) }, (_, i) => i), (hh) => datasetLen(hh, a.data_source!)]),
      RandomSampler: samplerType('RandomSampler', 'data_source replacement num_samples generator', (a) => {
        const rng = rngOf(h, a.generator)
        const size = (hh: Host) => (a.num_samples && a.num_samples.k !== 'none' ? hh.toInt(a.num_samples) : datasetLen(hh, a.data_source!))
        if (bool(a.replacement, false)) return [(hh) => Array.from({ length: size(hh) }, () => Math.floor(rng.next() * datasetLen(hh, a.data_source!))), size]
        refuse('RandomSampler', a, 'num_samples', isNone)
        return [(hh) => shuffled(rng, datasetLen(hh, a.data_source!)), size]
      }),
      SubsetRandomSampler: samplerType('SubsetRandomSampler', 'indices generator', (a) => {
        const rng = rngOf(h, a.generator)
        const ids = h.iterate(a.indices ?? py.NONE).map((v) => h.toInt(v))
        return [() => shuffled(rng, ids.length).map((i) => ids[i]), () => ids.length]
      }),
      WeightedRandomSampler: samplerType('WeightedRandomSampler', 'weights num_samples replacement generator', (a) => {
        const rng = rngOf(h, a.generator)
        const w = h.iterate(a.weights ?? py.NONE).map((v) => h.num(v))
        const n = int(a.num_samples, 'num_samples')
        return [() => drawIndices(rng, w, n, bool(a.replacement, true)), () => n]
      }),
    },
  }
  const drawIndices = (rng: SandboxRandom, weights: number[], n: number, replacement: boolean): number[] => {
    const w = [...weights]
    return Array.from({ length: n }, () => {
      const total = w.reduce((s, x) => s + x, 0)
      if (!(total > 0)) throw h.err('RuntimeError', 'invalid multinomial distribution (sum of probabilities <= 0)')
      let r = rng.next() * total
      let i = 0
      while (i < w.length - 1 && r >= w[i]) r -= w[i++]
      if (!replacement) w[i] = 0
      return i
    })
  }
  const nnFunctional: Value = { k: 'module', name: 'torch.nn.functional', attrs: Fattrs }
  const nn: Value = { k: 'module', name: 'torch.nn', attrs: { ...nnAttrs, functional: nnFunctional, init: nnInit, utils: nnUtils } }
  const optimMod: Value = { k: 'module', name: 'torch.optim', attrs: { ...optim, lr_scheduler: lrScheduler } }

  // ---- torch.linalg
  const linalgStep = (name: string, out: TT, input: TT, v: Value) => {
    trace(h, `torch.linalg.${name}`, 'linalg', [{ label: label(h, v, 'A'), t: input }], out)
    return wrap(out)
  }
  const normDtype = (x: TT, d: Value | undefined) => {
    const dt = dtypeArg(h, d)
    return dt ? C.toDtype(E, x, dt) : x
  }
  const vectorNormOf = (x: TT, ord: Value | undefined, dim: Value | undefined, keep: Value | undefined): TT => {
    if (ord?.k === 'str') throw h.err('RuntimeError', `linalg.vector_norm: ord must be a number, got '${ord.v}'`)
    return C.vectorNorm(E, x, ord && ord.k !== 'none' ? h.num(ord) : 2, dimsArg(h, dim), bool(keep, false))
  }
  const linalgFns: Record<string, Value> = {
    inv: fnv('inv', (args, kw) => {
      const x = T(args[0] ?? kw.A, 'linalg.inv')
      return linalgStep('inv', C.invT(E, x), x, args[0] ?? kw.A)
    }),
    det: fnv('det', (args, kw) => {
      const x = T(args[0] ?? kw.A, 'linalg.det')
      return linalgStep('det', C.detT(E, x), x, args[0] ?? kw.A)
    }),
    solve: fnv('solve', (args, kw) => {
      const a = bind(h, 'linalg.solve', 'A B left', args, kw)
      const A = T(a.A, 'linalg.solve')
      const B = T(a.B, 'linalg.solve')
      if (!bool(a.left, true)) return wrap(C.transposeT(E, C.solveT(E, C.transposeT(E, A, -2, -1), C.transposeT(E, B, -2, -1)), -2, -1))
      return linalgStep('solve', C.solveT(E, A, B), A, a.A!)
    }),
    norm: fnv('norm', (args, kw) => {
      const a = bind(h, 'linalg.norm', 'A ord dim keepdim dtype', args, kw)
      const x = normDtype(T(a.A, 'linalg.norm'), a.dtype)
      const dims = dimsArg(h, a.dim)
      const keep = bool(a.keepdim, false)
      const noOrd = !a.ord || a.ord.k === 'none'
      if (dims && dims.length === 2) return wrap(matrixNorm(h, E, x, a.ord, [dims[0], dims[1]], keep))
      if (dims && dims.length > 2) throw h.err('RuntimeError', `linalg.norm: If dim is specified, it must be of length 1 or 2. Got [${dims.join(', ')}]`)
      if (dims) return wrap(vectorNormOf(x, a.ord, a.dim, a.keepdim))
      if (noOrd) return wrap(C.vectorNorm(E, x, 2, null, keep))
      if (x.ndim === 1) return wrap(vectorNormOf(x, a.ord, undefined, a.keepdim))
      if (x.ndim === 2) return wrap(matrixNorm(h, E, x, a.ord, [0, 1], keep))
      throw h.err('RuntimeError', `linalg.norm: If dim is not specified but ord is, the input must be 1D or 2D. Got ${x.ndim}D.`)
    }),
    vector_norm: fnv('vector_norm', (args, kw) => {
      const a = bind(h, 'linalg.vector_norm', 'x ord dim keepdim dtype', args, kw)
      return wrap(vectorNormOf(normDtype(T(a.x, 'linalg.vector_norm'), a.dtype), a.ord, a.dim, a.keepdim))
    }),
    matrix_norm: fnv('matrix_norm', (args, kw) => {
      const a = bind(h, 'linalg.matrix_norm', 'A ord dim keepdim dtype', args, kw)
      const dims = dimsArg(h, a.dim) ?? [-2, -1]
      if (dims.length !== 2) throw h.err('RuntimeError', `linalg.matrix_norm: dim must be a 2-tuple. Got [${dims.join(', ')}]`)
      return wrap(matrixNorm(h, E, normDtype(T(a.A, 'linalg.matrix_norm'), a.dtype), a.ord, [dims[0], dims[1]], bool(a.keepdim, false)))
    }),
    matrix_power: fnv('matrix_power', (args, kw) => {
      const a = bind(h, 'linalg.matrix_power', 'A n', args, kw)
      const A0 = T(a.A, 'linalg.matrix_power')
      const n = int(a.n, 'n')
      const k = A0.shape[A0.ndim - 1]
      if (A0.ndim < 2 || A0.shape[A0.ndim - 2] !== k) throw h.err('RuntimeError', `linalg.matrix_power: A must be batches of square matrices, but they are ${A0.shape[A0.ndim - 2]} by ${k} matrices`)
      const base = n < 0 ? C.invT(E, A0) : A0
      let out = TT.of(Array.from({ length: prod(A0.shape) }, (_, f) => (f % (k + 1) === 0 ? 1 : 0)), A0.shape, A0.dtype)
      for (let i = 0; i < Math.abs(n); i++) out = i ? C.matmul(E, out, base) : base
      return wrap(out)
    }),
    multi_dot: fnv('multi_dot', (args) => wrap(h.iterate(args[0]).map((v) => T(v, 'linalg.multi_dot')).reduce((p, q) => C.matmul(E, p, q)))),
  }
  const linalg: Value = { k: 'module', name: 'torch.linalg', attrs: linalgFns }

  // ---- torch.autograd
  const autograd: Value = {
    k: 'module', name: 'torch.autograd', attrs: {
      grad: fnv('grad', (args, kw) => {
        const a = bind(h, 'autograd.grad', 'outputs inputs grad_outputs retain_graph create_graph only_inputs allow_unused materialize_grads', args, kw)
        refuse('autograd.grad', a, 'create_graph', isFalse)
        const outV = a.outputs ?? py.NONE
        const outs = asTT(outV) ? [outV] : h.iterate(outV)
        if (outs.length !== 1) throw h.err('NotImplementedError', 'autograd.grad: more than one output is not available in this sandbox: sum them first')
        const root = T(outs[0], 'autograd.grad outputs')
        const inV = a.inputs ?? py.NONE
        const inputs = (asTT(inV) ? [inV] : h.iterate(inV)).map((v) => T(v, 'autograd.grad inputs'))
        for (const x of inputs) if (!x.requiresGrad) throw h.err('RuntimeError', 'One of the differentiated Tensors does not require grad')
        const gv = a.grad_outputs ?? py.NONE
        const gout = asTT(gv) ?? (gv.k === 'tuple' || gv.k === 'list' ? asTT(gv.items[0] ?? py.NONE) : null)
        // a backward pass with every .grad set aside; the requested ones are read off and the rest put back
        const graph = new Set<TT>()
        const walk = (t: TT) => {
          if (graph.has(t)) return
          graph.add(t)
          for (const i of t.fn?.inputs ?? []) if (i && i.requiresGrad) walk(i)
        }
        walk(root)
        const saved = [...graph].map((t): [TT, TT | null, boolean] => [t, t.grad, t.retain])
        for (const t of graph) {
          t.grad = null
          t.retain = t.retain || inputs.includes(t)
        }
        try {
          C.backward(E, root, gout ? gout.values() : null, bool(a.retain_graph, false))
          const res = inputs.map((x) => (graph.has(x) && x.grad ? C.detach(x.grad) : null))
          if (res.some((g) => !g) && !bool(a.allow_unused, false)) throw h.err('RuntimeError', 'One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.')
          return py.tuple(res.map((g) => (g ? wrap(g) : py.NONE)))
        } finally {
          for (const [t, g, r] of saved) {
            t.grad = g
            t.retain = r
          }
        }
      }),
      backward: fnv('backward', (args, kw) => {
        const a = bind(h, 'autograd.backward', 'tensors grad_tensors retain_graph create_graph inputs', args, kw)
        refuse('autograd.backward', a, 'create_graph', isFalse)
        refuse('autograd.backward', a, 'inputs', isNone)
        const list = asTT(a.tensors ?? py.NONE) ? [a.tensors!] : h.iterate(a.tensors ?? py.NONE)
        const gs = a.grad_tensors && a.grad_tensors.k !== 'none' ? (asTT(a.grad_tensors) ? [a.grad_tensors] : h.iterate(a.grad_tensors)) : []
        list.forEach((v, i) => doBackward(h, T(v, 'backward'), v, gs[i] ? T(gs[i], 'grad_tensors').values() : null, bool(a.retain_graph, false)))
        return py.NONE
      }),
      Variable: { k: 'type', name: 'Variable', call: (args, kw) => {
        const src = T(args[0], 'Variable')
        src.requiresGrad = flagOf(h, kw.requires_grad ?? args[1], false)
        return wrap(src)
      } },
    },
  }

  const arange = fnv('arange', (args, kw) => {
    const given = kw.start || kw.end || kw.step
    const a = given ? [kw.start ?? py.int(0), kw.end ?? args[0], kw.step ?? py.int(1)] : args
    const nums = a.map((v) => h.num(v))
    const [start, end, step] = nums.length === 1 ? [0, nums[0], 1] : [nums[0], nums[1], nums[2] ?? 1]
    if (step === 0) throw h.err('RuntimeError', 'step must be nonzero')
    const n = Math.max(0, Math.ceil((end - start) / step))
    if (n > h.maxSize) throw h.err('RuntimeError', `arange of ${n} elements exceeds the sandbox limit`)
    const anyFloat = a.some((v) => v.k === 'float')
    const out = TT.of(Array.from({ length: n }, (_, i) => start + i * step), [n], dtypeArg(h, kw.dtype) ?? (anyFloat ? 'float32' : 'int64'))
    return finish(out, kw, 'arange')
  })

  /** torch.Tensor(2, 3) is a size, torch.Tensor([1, 2]) is data; the memory of a new size is not initialised, here it is 0 */
  const legacyCtor = (name: string, dtype: TDType): Value => ({ k: 'type', name, call: (args) => {
    if (!args.length) return wrap(TT.of([], [0], dtype))
    if (args[0].k === 'int' || (args[0].k === 'obj' && args[0].o instanceof SizeObj)) {
      const dims = toDims(h, args)
      return wrap(TT.of(new Array<number>(prod(dims)).fill(0), dims, dtype))
    }
    return wrap(tensorFrom(h, args[0], dtype))
  } })

  const dimOf = (args: Value[], kw: Kw): number | null => {
    const d = kw.dim ?? kw.axis ?? args[1]
    return d ? h.toInt(d) : null
  }
  const tensorList = (args: Value[], name: string): TT[] => (args.length === 1 && !asTT(args[0]) ? h.iterate(args[0]) : args).map((v) => T(v, name))
  const atLeast = (t: TT, d: 1 | 2): TT => {
    if (t.ndim >= d) return t
    return C.reshapeT(E, t, d === 1 ? [1] : t.ndim ? [1, t.size] : [1, 1], false)
  }
  /** torch.cat and its relatives: plan turns the tensors and the dim into what catT gets; the step shows which input each cell came from */
  const joined = (name: string, plan: (ts: TT[], dim: number | null) => [TT[], number], stack = false) => fnv(name, (args, kw) => {
    const vals = h.iterate(args[0] ?? kw.tensors)
    const inputs = vals.map((v) => T(v, name))
    const [ts, d] = plan(inputs, dimOf(args, kw))
    const out = C.catT(E, ts, d, stack)
    if (h.tracing && out.size <= SNAP_MAX) {
      // the same join over ids: operand · 2²⁰ + position
      const ids = C.catT({ grad: false } as Engine, ts.map((t, k) => TT.of(Array.from({ length: t.size }, (_, f) => k * 2 ** 20 + f), t.shape, 'int64')), d, stack)
      trace(h, `torch.${name}`, 'move', inputs.map((t, k) => ({ label: label(h, vals[k], `t${k}`), t })), out, null, ids.values().map((id): [number, number][] => [[Math.floor(id / 2 ** 20), id % 2 ** 20]]))
    }
    return wrap(out)
  })

  /** with torch.no_grad():, @torch.no_grad() or @torch.no_grad */
  const gradModeType = (name: string, enabled: boolean): Value => ({ k: 'type', name, call: (args) => {
    const m = py.obj(new GradMode(enabled))
    return args[0] ? h.call(m, [args[0]]) : m
  } })

  const VIA_METHOD = ['add', 'sub', 'div', 'mm', 'dot', 'bmm', 'mv', 'prod', 'sort', 'argsort', 'topk', 'kthvalue', 'trunc', 'frac', 'floor', 'ceil', 'sign', 'log2', 'log10', 'log1p', 'expm1', 'exp2', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'erf', 'erfc', 'reciprocal', 'rsqrt', 'neg', 'negative', 'remainder', 'fmod', 'floor_divide', 'true_divide', 'divide', 'multiply', 'subtract', 'atan2', 'all', 'any', 'count_nonzero', 'nonzero', 'masked_select', 'masked_fill', 'isnan', 'isinf', 'isfinite', 'isposinf', 'isneginf', 'logical_and', 'logical_or', 'logical_xor', 'logical_not', 'allclose', 'isclose', 'equal', 'is_floating_point', 'cumsum', 'cumprod', 'unique', 'bincount', 'tile', 'repeat_interleave', 'gather', 'scatter', 'scatter_add', 'index_select', 'index_fill', 'take', 'take_along_dim', 'narrow', 'split', 'chunk', 'unbind', 'flip', 'fliplr', 'flipud', 'roll', 'diag', 'diagonal', 'tril', 'triu', 'trace', 'mode', 'median', 'amax', 'amin', 'norm', 'broadcast_to', 'swapaxes', 'swapdims', 'moveaxis', 'movedim', 'ravel', 'unflatten', 'outer', 'inner', 'ger', 'lerp', 'clamp', 'clip', 'clamp_min', 'clamp_max', 'inverse', 'det', 'matrix_power', 'sum', 'mean', 'std', 'var', 'argmax', 'argmin', 'round', 'squeeze', 'unsqueeze', 'transpose', 'permute', 't', 'clone', 'detach', 'softmax', 'log_softmax']

  const torch: Record<string, Value> = {
    ...dtypeVals,
    ...Object.fromEntries(VIA_METHOD.map((n) => [n, viaMethod(n)])),
    __version__: py.str('2.12 (sandbox)'),
    pi: py.float(Math.PI), e: py.float(Math.E), inf: py.float(Infinity), nan: py.float(NaN),
    Tensor: legacyCtor('Tensor', 'float32'),
    ...Object.fromEntries(Object.entries(LEGACY).map(([name, d]) => [name, legacyCtor(name, d)])),
    Generator: { k: 'type', name: 'Generator', call: (args, kw) => {
      const device = kw.device ?? args[0]
      if (device && h.str(device).startsWith('cuda')) throw h.err('RuntimeError', 'Device type CUDA is not supported for torch.Generator() api.')
      return py.obj(new GeneratorObj(new SandboxRandom(1)))
    } },
    Size: { k: 'type', name: 'Size', call: (args) => py.obj(new SizeObj(args[0] ? h.iterate(args[0]).map((v) => h.toInt(v)) : [])) },
    tensor: fnv('tensor', (args, kw) => {
      if (!args.length) throw h.err('TypeError', "tensor() missing 1 required positional arguments: 'data'")
      return finish(tensorFrom(h, args[0], dtypeArg(h, kw.dtype)), kw, 'tensor')
    }),
    as_tensor: fnv('as_tensor', (args, kw) => {
      const d = dtypeArg(h, kw.dtype)
      const t = asTT(args[0])
      if (t) return wrap(d ? C.toDtype(E, t, d) : t)
      return !d && args[0]?.k === 'array' ? wrap(fromNd(args[0].a, true)) : wrap(tensorFrom(h, args[0], d))
    }),
    from_numpy: fnv('from_numpy', (args) => {
      if (args[0]?.k !== 'array') throw h.err('TypeError', `expected np.ndarray (got ${args[0] ? typeName(args[0]) : 'nothing'})`)
      return wrap(fromNd(args[0].a, true))
    }),
    zeros: filled('zeros', () => 0), ones: filled('ones', () => 1), empty: filled('empty', () => 0),
    full: fnv('full', (args, kw) => {
      const dims = toDims(h, [args[0] ?? kw.size])
      const fill = args[1] ?? kw.fill_value
      return finish(TT.of(new Array(prod(dims)).fill(h.num(fill)), dims, dtypeArg(h, kw.dtype) ?? (fill.k === 'int' ? 'int64' : fill.k === 'bool' ? 'bool' : 'float32')), kw, 'full')
    }),
    rand: filled('rand', (rng) => rng.next(), 'float32', true),
    randn: filled('randn', (rng) => rng.normal(), 'float32', true),
    randint: fnv('randint', (args, kw) => {
      const positional = kw.size ? args : args.slice(0, -1)
      const size = kw.size ?? args[args.length - 1]
      const [lo, hi] = kw.high ? [kw.low ?? args[0] ?? py.int(0), kw.high] : positional.length === 1 ? [py.int(0), positional[0]] : [positional[0], positional[1]]
      const [l, u] = [h.toInt(lo), h.toInt(hi)]
      if (l >= u) throw h.err('RuntimeError', `random_ expects 'from' to be less than 'to', but got from=${l} >= to=${u}`)
      const dims = toDims(h, [size])
      const rng = rngOf(h, kw.generator)
      return finish(TT.of(Array.from({ length: prod(dims) }, () => l + Math.floor(rng.next() * (u - l))), dims, dtypeArg(h, kw.dtype) ?? 'int64'), kw, 'randint')
    }),
    randint_like: fnv('randint_like', (args, kw) => {
      const t = T(args[0], 'randint_like')
      const [l, u] = args.length === 2 ? [0, h.toInt(args[1])] : [h.toInt(args[1]), h.toInt(args[2])]
      return finish(TT.of(t.values().map(() => l + Math.floor(h.rng.next() * (u - l))), t.shape, dtypeArg(h, kw.dtype) ?? t.dtype), kw, 'randint_like')
    }),
    randperm: fnv('randperm', (args, kw) => {
      const n = h.toInt(args[0] ?? kw.n)
      if (n < 0) throw h.err('RuntimeError', `n must be non-negative, got ${n}`)
      return finish(TT.of(shuffled(rngOf(h, kw.generator), n), [n], dtypeArg(h, kw.dtype) ?? 'int64'), kw, 'randperm')
    }),
    normal: fnv('normal', (args, kw) => {
      const a = bind(h, 'normal', 'mean std size generator dtype device requires_grad', args, kw)
      const rng = rngOf(h, a.generator)
      const mt = asTT(a.mean ?? py.NONE)
      const st = asTT(a.std ?? py.NONE)
      if (!mt && !st) {
        if (!a.size) throw h.err('TypeError', 'normal() received an invalid combination of arguments: a size is needed when mean and std are both numbers')
        const dims = toDims(h, [a.size])
        const [m, s] = [num(a.mean, 0), num(a.std, 1)]
        if (s < 0) throw h.err('RuntimeError', `normal expects std >= 0.0, but found std ${repr(py.float(s))}`)
        return finish(TT.of(Array.from({ length: prod(dims) }, () => m + s * rng.normal()), dims, dtypeArg(h, a.dtype) ?? 'float32'), {}, 'normal')
      }
      const shape = broadcastShapes([mt?.shape ?? [], st?.shape ?? []])
      const [im, is] = [mt ? C.bcastIndex(mt.shape, shape) : null, st ? C.bcastIndex(st.shape, shape) : null]
      const [mv, sv] = [mt?.values(), st?.values()]
      const [m0, s0] = [mt ? 0 : num(a.mean, 0), st ? 0 : num(a.std, 1)]
      if (sv?.some((s) => s < 0)) throw h.err('RuntimeError', 'normal expects all elements of std >= 0.0')
      return finish(TT.of(Array.from({ length: prod(shape) }, (_, i) => (mv ? mv[im![i]] : m0) + (sv ? sv[is![i]] : s0) * rng.normal()), shape, (mt ?? st)!.dtype), {}, 'normal')
    }),
    bernoulli: fnv('bernoulli', (args, kw) => {
      const t = T(args[0] ?? kw.input, 'bernoulli')
      if (t.values().some((p) => !(p >= 0 && p <= 1))) throw h.err('RuntimeError', 'Expected p_in >= 0 && p_in <= 1 to be true')
      const rng = rngOf(h, kw.generator)
      return finish(TT.of(t.values().map((p) => (rng.next() < p ? 1 : 0)), t.shape, isFloatT(t.dtype) ? t.dtype : 'float32'), {}, 'bernoulli')
    }),
    multinomial: fnv('multinomial', (args, kw) => {
      const a = bind(h, 'multinomial', 'input num_samples replacement generator', args, kw)
      const t = T(a.input, 'multinomial')
      if (t.ndim < 1 || t.ndim > 2) throw h.err('RuntimeError', 'prob_dist must be 1 or 2 dim')
      const n = int(a.num_samples, 'num_samples')
      const replacement = bool(a.replacement, false)
      const classes = t.shape[t.ndim - 1]
      if (!replacement && n > classes) throw h.err('RuntimeError', 'cannot sample n_sample > prob_dist.size(-1) samples without replacement')
      const rng = rngOf(h, a.generator)
      const vals = t.values()
      const rows = t.ndim === 2 ? t.shape[0] : 1
      const out = Array.from({ length: rows }, (_, r) => drawIndices(rng, vals.slice(r * classes, (r + 1) * classes), n, replacement)).flat()
      return finish(TT.of(out, t.ndim === 2 ? [rows, n] : [n], 'int64'), {}, 'multinomial')
    }),
    eye: fnv('eye', (args, kw) => {
      const n = h.toInt(args[0] ?? kw.n)
      const m = args[1] ?? kw.m ? h.toInt((args[1] ?? kw.m)!) : n
      return finish(TT.of(Array.from({ length: n * m }, (_, i) => (Math.floor(i / m) === i % m ? 1 : 0)), [n, m], dtypeArg(h, kw.dtype) ?? 'float32'), kw, 'eye')
    }),
    arange,
    linspace: fnv('linspace', (args, kw) => {
      const a = h.num(args[0] ?? kw.start)
      const b = h.num(args[1] ?? kw.end)
      const n = h.toInt(args[2] ?? kw.steps)
      return finish(TT.of(Array.from({ length: n }, (_, i) => (n === 1 ? a : a + ((b - a) * i) / (n - 1))), [n], dtypeArg(h, kw.dtype) ?? 'float32'), kw, 'linspace')
    }),
    logspace: fnv('logspace', (args, kw) => {
      const a = h.num(args[0] ?? kw.start)
      const b = h.num(args[1] ?? kw.end)
      const n = h.toInt(args[2] ?? kw.steps)
      const base = num(args[3] ?? kw.base, 10)
      return finish(TT.of(Array.from({ length: n }, (_, i) => base ** (n === 1 ? a : a + ((b - a) * i) / (n - 1))), [n], dtypeArg(h, kw.dtype) ?? 'float32'), kw, 'logspace')
    }),
    zeros_like: like('zeros_like', () => 0), ones_like: like('ones_like', () => 1), empty_like: like('empty_like', () => 0),
    rand_like: like('rand_like', () => h.rng.next(), true), randn_like: like('randn_like', () => h.rng.normal(), true),
    full_like: fnv('full_like', (args, kw) => {
      const t = T(args[0], 'full_like')
      const v = h.num(args[1] ?? kw.fill_value)
      return finish(TT.of(t.values().map(() => v), t.shape, dtypeArg(h, kw.dtype) ?? t.dtype), kw, 'full_like')
    }),
    manual_seed: fnv('manual_seed', (args, kw) => {
      h.rng.seed(h.toInt(kw.seed ?? args[0]))
      const g = new GeneratorObj(h.rng)
      g.start = h.toInt(kw.seed ?? args[0])
      return py.obj(g)
    }),
    is_tensor: fnv('is_tensor', (args) => py.bool(!!asTT(args[0]))),
    numel: fnv('numel', (args) => py.int(T(args[0], 'numel').size)),
    cat: joined('cat', (ts, d) => [ts, d ?? 0]), concat: joined('concat', (ts, d) => [ts, d ?? 0]), concatenate: joined('concatenate', (ts, d) => [ts, d ?? 0]),
    stack: joined('stack', (ts, d) => [ts, d ?? 0], true),
    hstack: joined('hstack', (ts) => {
      const parts1 = ts.map((t) => atLeast(t, 1))
      return [parts1, parts1[0]?.ndim === 1 ? 0 : 1]
    }),
    vstack: joined('vstack', (ts) => [ts.map((t) => atLeast(t, 2)), 0]),
    row_stack: joined('row_stack', (ts) => [ts.map((t) => atLeast(t, 2)), 0]),
    column_stack: joined('column_stack', (ts) => [ts.map((t) => (t.ndim > 1 ? t : C.reshapeT(E, t, [t.size, 1], false))), 1]),
    meshgrid: fnv('meshgrid', (args, kw) => {
      const how = kw.indexing && kw.indexing.k !== 'none' ? h.str(kw.indexing) : 'ij'
      if (how !== 'ij' && how !== 'xy') throw h.err('RuntimeError', `torch.meshgrid: indexing must be one of "xy" or "ij", but received: ${how}`)
      const ts = tensorList(args, 'meshgrid')
      if (!ts.length) throw h.err('RuntimeError', 'meshgrid expects a non-empty TensorList')
      if (ts.some((t) => t.ndim > 1)) throw h.err('RuntimeError', 'torch.meshgrid expects all tensors to be 0D or 1D')
      if (ts.some((t) => t.dtype !== ts[0].dtype)) throw h.err('RuntimeError', 'torch.meshgrid expects all tensors to have the same dtype')
      // 'xy' puts y first: the first two axes swap
      const swap = how === 'xy' && ts.length > 1
      const lens = ts.map((t) => t.size)
      const shape = swap ? [lens[1], lens[0], ...lens.slice(2)] : lens
      return py.tuple(ts.map((t, i) => {
        const axis = swap && i < 2 ? 1 - i : i
        const out = C.expandT(E, C.reshapeT(E, t, shape.map((_, k) => (k === axis ? t.size : 1)), false), shape)
        trace(h, 'torch.meshgrid', 'move', [{ label: `t${i}`, t }], out, null, h.tracing && out.size <= SNAP_MAX ? Array.from({ length: out.size }, (_, f) => Math.floor(f / cStrides(shape)[axis]) % t.size) : undefined)
        return wrap(out)
      }))
    }),
    broadcast_tensors: fnv('broadcast_tensors', (args) => {
      const ts = tensorList(args, 'broadcast_tensors')
      const shape = broadcastShapes(ts.map((t) => t.shape))
      return py.tuple(ts.map((t) => wrap(C.expandT(E, t, shape))))
    }),
    broadcast_shapes: fnv('broadcast_shapes', (args) => py.obj(new SizeObj(broadcastShapes(args.map((v) => (v.k === 'int' ? [v.v] : h.iterate(v).map((x) => h.toInt(x)))))))),
    matmul: fnv('matmul', (args) => {
      const a = T(args[0], 'matmul')
      const b = T(args[1], 'matmul')
      sameDtype(h, a, b)
      const out = C.matmul(E, a, b)
      trace(h, 'torch.matmul', 'matmul', [{ label: label(h, args[0], 'a'), t: a }, { label: label(h, args[1], 'b'), t: b }], out)
      return wrap(out)
    }),
    mul: twoArg('mul', '*'),
    pow: fnv('pow', (args, kw) => {
      const base = args[0]
      const exponent = args[1] ?? kw.exponent
      const self = asTT(base)
      if (self) return objOf(self).binop('**', exponent, false, h)!
      return objOf(T(exponent, 'pow')).binop('**', base, true, h)!
    }),
    eq: twoArg('eq', '=='), gt: twoArg('gt', '>'), lt: twoArg('lt', '<'), ne: twoArg('ne', '!='), ge: twoArg('ge', '>='), le: twoArg('le', '<='),
    greater: twoArg('greater', '>'), less: twoArg('less', '<'), greater_equal: twoArg('greater_equal', '>='), less_equal: twoArg('less_equal', '<='), not_equal: twoArg('not_equal', '!='),
    maximum: fnv('maximum', (args) => wrap(C.binary(E, 'maximum', T(args[0], 'maximum'), T(args[1], 'maximum')))),
    minimum: fnv('minimum', (args) => wrap(C.binary(E, 'minimum', T(args[0], 'minimum'), T(args[1], 'minimum')))),
    max: viaMethod('max'), min: viaMethod('min'),
    exp: uni('exp'), log: uni('log'), sqrt: uni('sqrt'), abs: uni('abs'), relu: uni('relu'), sigmoid: uni('sigmoid'), tanh: uni('tanh'), square: uni('square'), sin: uni('sin'), cos: uni('cos'),
    absolute: uni('abs'),
    reshape: fnv('reshape', (args, kw) => h.call(h.getAttr(args[0], 'reshape'), [args[1] ?? kw.shape])),
    flatten: fnv('flatten', (args, kw) => {
      const t = T(args[0], 'flatten')
      const out = C.flattenT(E, t, optDim(h, kw.start_dim ?? args[1]) ?? 0, optDim(h, kw.end_dim ?? args[2]) ?? -1)
      flowStep(h, 'torch.flatten', t, out)
      return wrap(out)
    }),
    where: fnv('where', (args, kw) => {
      const c = T(args[0] ?? kw.condition, 'where')
      if (args.length === 1 && !kw.input && !kw.other) return h.call(h.getAttr(wrap(c), 'nonzero'), [], { as_tuple: py.bool(true) })
      const a = operand(h, args[1] ?? kw.input, asTT(args[2] ?? py.NONE)?.dtype)
      const b = operand(h, args[2] ?? kw.other, asTT(args[1] ?? py.NONE)?.dtype)
      if (!a || !b) throw h.err('TypeError', 'where(condition, x, y): x and y must be tensors or numbers')
      const out = C.whereT(E, c, a.t, b.t, C.promote(a.t.dtype, b.t.dtype, a.weak, b.weak))
      trace(h, 'torch.where', 'elementwise', [{ label: label(h, args[0], 'condition'), t: c }, { label: label(h, args[1], 'x'), t: a.t }, { label: label(h, args[2], 'y'), t: b.t }], out)
      return wrap(out)
    }),
    no_grad: gradModeType('no_grad', false),
    enable_grad: gradModeType('enable_grad', true),
    inference_mode: { k: 'type', name: 'inference_mode', call: (args, kw) => {
      const mode = kw.mode ?? args[0]
      if (mode && mode.k !== 'bool') return h.call(py.obj(new GradMode(false)), [mode])
      return py.obj(new GradMode(!flagOf(h, mode, true)))
    } },
    set_grad_enabled: fnv('set_grad_enabled', (args, kw) => {
      const m = new GradMode(h.truthy(args[0] ?? kw.mode))
      m.enter(h)
      m.applied = true
      return py.obj(m)
    }),
    is_grad_enabled: fnv('is_grad_enabled', () => py.bool(E.grad)),
    device: { k: 'type', name: 'device', call: (args) => {
      const s = h.str(args[0] ?? py.str('cpu'))
      return py.obj(new DeviceObj(s.split(':')[0]))
    } },
    save: fnv('save', (args, kw) => {
      S.files.set(h.str(args[1] ?? kw.f), snapshot(args[0] ?? kw.obj, new Map()) as Value)
      h.print(`(sandbox) saved to ${h.str(args[1] ?? kw.f)} in memory: there is no disk here\n`)
      return py.NONE
    }),
    load: fnv('load', (args, kw) => {
      const path = h.str(args[0] ?? kw.f)
      const v = S.files.get(path)
      if (!v) throw h.err('FileNotFoundError', `[Errno 2] No such file or directory: '${path}' (the sandbox only knows files saved with torch.save in this run)`)
      return snapshot(v, new Map()) as Value
    }),
    cuda, nn, optim: optimMod, linalg, autograd,
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
    torch: torchMod, 'torch.nn': nn, 'torch.nn.functional': nnFunctional, 'torch.nn.init': nnInit, 'torch.nn.utils': nnUtils, 'torch.optim': optimMod, 'torch.optim.lr_scheduler': lrScheduler, 'torch.utils': torch.utils, 'torch.utils.data': data, 'torch.cuda': cuda, 'torch.linalg': linalg, 'torch.autograd': autograd,
    torchvision: { k: 'module', name: 'torchvision', attrs: { transforms, datasets } }, 'torchvision.transforms': transforms, 'torchvision.datasets': datasets,
  }
}

export const TORCH: PyLib = {
  modules: ['torch', 'torch.nn', 'torch.nn.functional', 'torch.nn.init', 'torch.nn.utils', 'torch.optim', 'torch.optim.lr_scheduler', 'torch.utils', 'torch.utils.data', 'torch.cuda', 'torch.linalg', 'torch.autograd', 'torchvision', 'torchvision.transforms', 'torchvision.datasets'],
  load: build,
}
