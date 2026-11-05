/**
 * torch for the sandbox (lecture 9): tensors with PyTorch's printing and dtype rules, autograd (the recorded graph is
 * shown after backward()), nn layers and nn.Module subclasses (each forward pass is traced layer by layer with its
 * shapes and parameter counts), losses, optimisers and learning-rate schedulers, torch.utils.data, torch.linalg,
 * torch.special, torch.einsum and torch.from_numpy's shared memory.
 * Built on tensorCore; random values come from the sandbox generator, so they differ from real PyTorch.
 * Checked against real torch by pyTorchCases.test.ts: run scripts/gen-torch-cases.py to refreeze what it printed.
 */
import { NDArray, PyError, broadcastShapes, cStrides, planIndex, prod, setIndex, sliceIndices } from './ndarray'
import type { DType, IndexItem } from './ndarray'
import { SandboxRandom } from './ndops'
import { PyDict, PyObj, formatSpec, isNum, numOf, py, repr, str, toArray, typeName } from './minipy'
import type { ApiKind, ClassValue, GridSnapshot, Host, HostClass, InstValue, Kw, PyLib, Value } from './minipy'
import type { FlowRow, GraphNode } from './pyEvents'
import * as C from './tensorCore'
import * as L from './npLinalg'
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
  /** what torch.initial_seed() reports */
  seed: number
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
    s = { seed: 67280421310721, E: new Engine(h.rng), flow: null, files: new Map(), notes: new Set(), lastModel: null }
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
  float16: 'float16', half: 'float16', float32: 'float32', float: 'float32', float64: 'float64', double: 'float64', int64: 'int64', long: 'int64',
  int32: 'int32', int: 'int32', int16: 'int16', short: 'int16', int8: 'int8', uint8: 'uint8', bool: 'bool',
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

const DEVICE_TYPES = ['cpu', 'cuda', 'ipu', 'xpu', 'mkldnn', 'opengl', 'opencl', 'ideep', 'hip', 've', 'fpga', 'maia', 'xla', 'lazy', 'vulkan', 'mps', 'meta', 'hpu', 'mtia', 'privateuseone']

class DeviceObj extends PyObj {
  readonly cls = 'device'
  type: string
  index: number | null
  constructor(type: string, index: number | null = null) {
    super()
    this.type = type
    this.index = index
  }
  repr() { return `device(type='${this.type}'${this.index === null ? '' : `, index=${this.index}`})` }
  str() { return this.index === null ? this.type : `${this.type}:${this.index}` }
  getAttr(name: string): Value | undefined {
    if (name === 'index') return this.index === null ? py.NONE : py.int(this.index)
    return name === 'type' ? py.str(this.type) : undefined
  }
  binop(op: string, other: Value): Value | undefined {
    if (op !== '==' && op !== '!=') return undefined
    const o = other.k === 'obj' && other.o instanceof DeviceObj ? other.o : null
    return py.bool(!!o && o.type === this.type && o.index === this.index === (op === '=='))
  }
}

/** a device string or object: 'cpu', 'cuda:1' */
const deviceOf = (h: Host, v: Value): DeviceObj => {
  if (v.k === 'obj' && v.o instanceof DeviceObj) return v.o
  // a bare number means the machine's accelerator, and this machine has none
  if (v.k === 'int') return new DeviceObj('cuda', v.v)
  if (v.k !== 'str') throw h.err('TypeError', `device(): argument 'type' must be str, not ${typeName(v)}`)
  const [type, ix] = v.v.split(':')
  if (!DEVICE_TYPES.includes(type)) throw h.err('RuntimeError', `Expected one of ${DEVICE_TYPES.join(', ')} device type at start of device string: ${v.v}`)
  if (ix !== undefined && !/^\d+$/.test(ix)) throw h.err('RuntimeError', `Invalid device string: '${v.v}'`)
  return new DeviceObj(type, ix === undefined ? null : Number(ix))
}

/** only the CPU exists here, as on a machine without CUDA */
const onlyCpu = (h: Host, v: Value | undefined) => {
  if (v && v.k !== 'none' && deviceOf(h, v).type === 'cuda') throw h.err('AssertionError', 'Torch not compiled with CUDA enabled')
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

/** torch.return_types.max: a named tuple such as (values, indices) */
class ReturnTypes extends PyObj {
  readonly cls: string
  fields: [string, Value][]
  constructor(name: string, fields: [string, Value][]) {
    super()
    this.cls = name
    this.fields = fields
  }
  repr() { return `torch.return_types.${this.cls}(\n${this.fields.map(([n, v]) => `${n}=${repr(v)}`).join(',\n')})` }
  iter() { return this.fields.map(([, v]) => v) }
  len() { return this.fields.length }
  isa(name: string) { return name === 'tuple' }
  getItem(idx: Value, h: Host) {
    const i = h.toInt(idx)
    const n = this.fields.length
    if (i < -n || i >= n) throw h.err('IndexError', 'tuple index out of range')
    return this.fields[i < 0 ? i + n : i][1]
  }
  getAttr(name: string) { return this.fields.find(([n]) => n === name)?.[1] }
}

class GradFnObj extends PyObj {
  readonly cls: string
  node: C.GradNode | null
  leaf: TT | null
  constructor(name: string, node: C.GradNode | null = null, leaf: TT | null = null) {
    super()
    this.cls = name
    this.node = node
    this.leaf = leaf
  }
  repr() { return `<${this.cls} object>` }
  getAttr(name: string): Value | undefined {
    if (name === 'name') return { k: 'fn', name, call: () => py.str(this.leaf ? `torch::autograd::${this.cls}` : this.cls) }
    if (name === 'variable' && this.leaf) return wrap(this.leaf)
    if (name === 'next_functions') {
      const n = this.node
      return py.tuple(n ? n.inputs.map((_, i) => {
        const x = C.inputOf(n, i)
        const next = !x || !x.requiresGrad ? py.NONE : x.fn ? py.obj(new GradFnObj(x.fn.name, x.fn)) : py.obj(new GradFnObj('AccumulateGrad', null, x))
        return py.tuple([next, py.int(0)])
      }) : [])
    }
    return undefined
  }
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
const scalarOf = (t: TT, v: number): Value => (t.dtype === 'bool' ? py.bool(v !== 0) : isFloatT(t.dtype) ? py.float(v) : py.int(v))

const operand = (h: Host, v: Value, like?: TDType): { t: TT; weak: boolean } | null => {
  const t = asTT(v)
  if (t) return { t, weak: false }
  if (v.k === 'int' || v.k === 'bool') return { t: TT.scalar(numOf(v), v.k === 'bool' ? 'bool' : 'int64'), weak: true }
  if (v.k === 'float') return { t: TT.scalar(v.v, like === 'float64' ? 'float64' : C.DEFAULTS.float), weak: true }
  if (v.k === 'array') return { t: fromNd(v.a, false), weak: false }
  void h
  return null
}

const ND_TO_T: Record<DType, TDType> = { float64: 'float64', float32: 'float32', int64: 'int64', bool: 'bool', uint8: 'uint8' }

/** a NumPy array as a tensor; share = torch.from_numpy (same memory) */
const fromNd = (a: NDArray, share: boolean): TT => (share ? new TT(a, ND_TO_T[a.dtype]) : TT.of(a.values(), a.shape, ND_TO_T[a.dtype]))

/** torch.tensor(data): Python floats → float32, ints → int64, NumPy keeps its dtype */
/** data with a float64 array or tensor in it keeps that dtype; Python numbers take the default */
const anyTensorDouble = (v: Value): boolean => (v.k === 'list' || v.k === 'tuple' ? v.items.some(anyTensorDouble) : v.k === 'array' ? v.a.dtype === 'float64' : asTT(v)?.dtype === 'float64')

/** walks the data as torch.tensor does, for its own messages */
const checkData = (h: Host, v: Value) => {
  const seq = (x: Value) => x.k === 'list' || x.k === 'tuple' || x.k === 'array' || x.k === 'range' || asTT(x) || (x.k === 'obj' && x.o instanceof SizeObj)
  const scalar = (x: Value) => isNum(x)
  const items = (x: Value): Value[] => (x.k === 'list' || x.k === 'tuple' ? x.items : [])
  if (!seq(v) && !scalar(v)) {
    if (v.k === 'str') throw h.err('TypeError', `new(): invalid data type 'str'`)
    throw h.err('RuntimeError', `Could not infer dtype of ${typeName(v)}`)
  }
  if (v.k !== 'list' && v.k !== 'tuple') return
  // the shape comes from the first element at each depth
  const shape: number[] = []
  let at: Value = v
  while (at.k === 'list' || at.k === 'tuple') {
    shape.push(at.items.length)
    if (!at.items.length) break
    at = at.items[0]
  }
  if ((at as Value).k === 'none') throw h.err('RuntimeError', 'Could not infer dtype of NoneType')
  if ((at as Value).k === 'str') throw h.err('ValueError', "too many dimensions 'str'")
  const walk = (x: Value, d: number) => {
    if (d === shape.length) {
      if (x.k === 'list' || x.k === 'tuple') throw h.err('TypeError', `'${x.k}' object cannot be interpreted as an integer`)
      if (x.k === 'str') throw h.err('TypeError', `new(): invalid data type 'str'`)
      if (x.k === 'none') throw h.err('RuntimeError', 'Could not infer dtype of NoneType')
      return
    }
    if (x.k !== 'list' && x.k !== 'tuple') {
      if (asTT(x) || x.k === 'array') return
      throw h.err('TypeError', 'not a sequence')
    }
    if (x.items.length !== shape[d]) throw h.err('ValueError', `expected sequence of length ${shape[d]} at dim ${d} (got ${x.items.length})`)
    for (const y of items(x)) walk(y, d + 1)
  }
  walk(v, 0)
}

const tensorFrom = (h: Host, v: Value, dtype: TDType | null): TT => {
  checkData(h, v)
  const t = asTT(v)
  if (t) return TT.of(t.values(), t.shape, dtype ?? t.dtype)
  if (v.k === 'array') return TT.of(v.a.values(), v.a.shape, dtype ?? ND_TO_T[v.a.dtype])
  let a: NDArray
  try {
    a = toArray(v)
  } catch (e) {
    throw h.err('TypeError', `torch.tensor(): could not build a tensor from this data (${(e as Error).message})`)
  }
  const inferred: TDType = a.dtype === 'float64' && anyTensorDouble(v) ? 'float64' : a.dtype === 'float32' ? 'float32' : a.dtype === 'float64' ? C.DEFAULTS.float : a.dtype === 'bool' ? 'bool' : a.dtype === 'uint8' ? 'uint8' : 'int64'
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

/** the index errors say tensor and dimension here, not array and axis */
const indexWording = <T>(t: TT, f: () => T): T => {
  try {
    return f()
  } catch (e) {
    if (e instanceof PyError) {
      if (e.message.startsWith('too many indices')) throw new PyError('IndexError', `too many indices for tensor of dimension ${t.ndim}`)
      const m = /^index (-?\d+) is out of bounds for axis (\d+) with size (\d+)/.exec(e.message)
      if (m) throw new PyError('IndexError', `index ${m[1]} is out of bounds for dimension ${m[2]} with size ${m[3]}`)
    }
    throw e
  }
}

const BITWISE: Record<string, string> = { '&': 'bitwise_and', '|': 'bitwise_or', '^': 'bitwise_xor', '<<': 'lshift', '>>': 'rshift' }

/** a @ b, under the grad_fn the real thing ends on: the batched and vector forms finish by reshaping */
const matmulNamed = (E: Engine, a: TT, b: TT): TT => {
  const out = C.matmul(E, a, b)
  if (out.fn && a.ndim === 1 && b.ndim >= 2) out.fn.name = 'SqueezeBackward4'
  else if (out.fn && (a.ndim > 2 || b.ndim > 2)) out.fn.name = b.ndim === 1 ? 'SqueezeBackward1' : 'UnsafeViewBackward0'
  return out
}

const BIN_API: Record<string, string> = { '+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '**': 'pow', '//': 'floor_divide', '%': 'remainder' }

export class TensorObj extends PyObj {
  get cls() { return this.t.isParam ? 'Parameter' : 'Tensor' }
  t: TT
  constructor(t: TT) {
    super()
    this.t = t
  }
  repr() {
    return this.t.isParam ? `Parameter containing:\n${formatTensor(this.t)}` : formatTensor(this.t)
  }
  /** Tensor.__format__: a plain 0-d tensor formats as its number, anything else only with an empty spec */
  format(spec: string) {
    if (this.t.ndim === 0 && !this.t.isParam) return formatSpec(scalarOf(this.t, this.t.values()[0]), spec)
    if (spec) throw new PyError('TypeError', 'unsupported format string passed to Tensor.__format__')
    return undefined
  }
  asIndex(h: Host): number {
    if (this.t.size !== 1 || isFloatT(this.t.dtype)) throw h.err('TypeError', 'only integer tensors of a single element can be converted to an index')
    return this.t.values()[0]
  }
  asNumber(h: Host): number {
    if (this.t.size !== 1) throw h.err('ValueError', 'only one element tensors can be converted to Python scalars')
    return this.t.values()[0]
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
    const out = indexWording(this.t, () => C.indexT(E, this.t, indexItems(h, idx)))
    trace(h, 'Tensor.__getitem__', 'move', [{ label: label(h, py.obj(this), 'x'), t: this.t }], out)
    return wrap(out)
  }
  setItem(idx: Value, v: Value, h: Host) {
    const E = stateOf(h).E
    const t = this.t
    const items = indexItems(h, idx)
    const plan = indexWording(t, () => planIndex(t.a, items))
    // a[i] = v goes through a view of a; a[mask] = v does not
    if (E.grad && t.isLeaf && t.requiresGrad) throw h.err('RuntimeError', plan.advanced ? 'a leaf Variable that requires grad is being used in an in-place operation.' : 'a view of a leaf Variable that requires grad is being used in an in-place operation.')
    this.checkInplace(h)
    const o = operand(h, v, t.dtype)
    if (!o) throw h.err('TypeError', `can't assign a ${typeName(v)} to a torch.${C.typeLabel(t.dtype)}Tensor`)
    // extra leading dimensions of size 1 do not count, as in PyTorch
    const extra = Math.max(0, o.t.ndim - plan.shape.length)
    const lead = plan.shape.length - (o.t.ndim - extra)
    const bad = o.t.shape.findIndex((d, k) => (k < extra ? d !== 1 : d !== 1 && d !== plan.shape[lead + k - extra]))
    if (bad >= 0) {
      if (plan.advanced) throw h.err('RuntimeError', `shape mismatch: value tensor of shape [${o.t.shape.join(', ')}] cannot be broadcast to indexing result of shape [${plan.shape.join(', ')}]`)
      throw h.err('RuntimeError', `The expanded size of the tensor (${plan.shape[lead + bad - extra]}) must match the existing size (${o.t.shape[bad]}) at non-singleton dimension ${lead + bad - extra}.  Target sizes: [${plan.shape.join(', ')}].  Tensor sizes: [${o.t.shape.join(', ')}]`)
    }
    const val = extra ? C.reshapeT(E, o.t, o.t.shape.slice(extra), false) : o.t
    if (E.grad && (val.requiresGrad || t.requiresGrad)) {
      this.rebase(h, (old) => C.setItemT(E, old, plan.addresses, plan.shape, val))
      return
    }
    setIndex(t.a, items, NDArray.create(val.values().map((x) => C.castT(x, t.dtype)), val.shape, t.a.dtype))
    t.version++
    if (t.base) t.base.version++
  }
  checkInplace(h: Host) {
    const t = this.t
    if (!stateOf(h).E.grad) return
    if (t.isLeaf && t.requiresGrad) throw h.err('RuntimeError', 'a leaf Variable that requires grad is being used in an in-place operation.')
    if (t.base && t.base.isLeaf && t.base.requiresGrad) throw h.err('RuntimeError', 'a view of a leaf Variable that requires grad is being used in an in-place operation.')
  }
  /**
   * An in-place op: the result overwrites this tensor and its history continues from a stand-in for the old value, so
   * y = x * 2; y.mul_(3) still differentiates to 6. reshape: the op only changes the view (t_, unsqueeze_ …).
   */
  rebase(h: Host, compute: (old: TT) => TT, reshape = false): Value {
    this.checkInplace(h)
    const t = this.t
    const E = stateOf(h).E
    C.refreshView(E, t)
    const old = new TT(t.a, t.dtype)
    old.fn = t.fn
    old.requiresGrad = t.requiresGrad
    const res = compute(old)
    if (!isFloatT(t.dtype) && isFloatT(res.dtype)) throw h.err('RuntimeError', `result type ${C.typeLabel(res.dtype)} can't be cast to the desired output type ${C.typeLabel(t.dtype)}`)
    if (!reshape && res.shape.join() !== t.shape.join()) throw h.err('RuntimeError', `output with shape [${t.shape.join(', ')}] doesn't match the broadcast shape [${res.shape.join(', ')}]`)
    C.commitInplace(E, t, old, res, reshape)
    return py.obj(this)
  }
  binop(op: string, other: Value, reflected: boolean, h: Host): Value | undefined {
    if (reflected && other.k === 'array' && ['+', '-', '*', '/', '**', '//', '%', '@'].includes(op)) {
      // numpy on the left gives up on a tensor, in its own words
      throw h.err('TypeError', op === '+' ? 'Concatenation operation is not implemented for NumPy arrays, use np.concatenate() instead. Please do not rely on this error; it may not be given on all Python implementations.' : `unsupported operand type(s) for ${op}: 'numpy.ndarray' and 'Tensor'`)
    }
    if (reflected && op === '+' && (other.k === 'list' || other.k === 'tuple' || other.k === 'str')) throw h.err('TypeError', `can only concatenate ${other.k} (not "Tensor") to ${other.k}`)
    // 'text' % tensor is Python's string formatting
    if (reflected && op === '%' && other.k === 'str') return undefined
    const o = operand(h, other, this.t.dtype)
    if (!o) {
      // a string or None next to a tensor is an error here, not a reason to try the other operand's method
      if (['+', '-', '*', '/', '**', '//', '%'].includes(op) && (other.k === 'str' || other.k === 'none')) throw h.err('TypeError', `unsupported operand type(s) for ${op}: '${reflected ? typeName(other) : 'Tensor'}' and '${reflected ? 'Tensor' : typeName(other)}'`)
      return undefined
    }
    const E = stateOf(h).E
    const [a, b, aw, bw] = reflected ? [o.t, this.t, o.weak, false] : [this.t, o.t, false, o.weak]
    let out: TT
    if (op === '@') {
      sameDtype(h, a, b)
      out = matmulNamed(E, a, b)
      trace(h, 'op:@', 'matmul', [{ label: reflected ? 'a' : label(h, py.obj(this), 'a'), t: a }, { label: reflected ? label(h, py.obj(this), 'b') : label(h, other, 'b'), t: b }], out)
      return wrap(out)
    }
    if (['<', '<=', '>', '>=', '==', '!='].includes(op)) {
      out = C.compare(a, b, op as C.CmpOpT)
    } else if (op === '%' && (a.requiresGrad || b.requiresGrad)) {
      out = C.map2T(E, a, b, (x, y) => x - Math.floor(x / y) * y, () => 1, (x, y) => -Math.floor(x / y), C.promote(a.dtype, b.dtype, aw, bw), 'RemainderBackward0')
    } else if (op === '//' && (a.requiresGrad || b.requiresGrad)) {
      out = C.noGradient(E, C.binary({ grad: false } as Engine, '//', a, b, aw, bw), [a, b], 'floor_divide')
    } else if (op in BIN_API) {
      out = C.binary(E, op as C.BinOpT, a, b, aw, bw)
      if (out.fn && reflected && o.weak && op === '-') out.fn.name = 'RsubBackward1'
      if (out.fn && reflected && o.weak && op === '/') out.fn.name = 'MulBackward0'
      if (out.fn && reflected && o.weak && op === '**') out.fn.name = 'PowBackward2'
      if (out.fn && !reflected && o.t.ndim && op === '**') out.fn.name = 'PowBackward1'
    } else if (BITWISE[op]) {
      const dtype = C.promote(a.dtype, b.dtype, aw, bw)
      const shift = op === '<<' || op === '>>'
      if (isFloatT(dtype)) throw h.err('NotImplementedError', `"${BITWISE[op]}_cpu" not implemented for '${C.typeLabel(dtype)}'`)
      if (shift && dtype === 'bool') throw h.err('NotImplementedError', `"${BITWISE[op]}_cpu" not implemented for 'Bool'`)
      const f = (x: number, y: number) => (op === '&' ? x & y : op === '|' ? x | y : op === '^' ? x ^ y : op === '<<' ? x * 2 ** y : Math.floor(x / 2 ** y))
      out = C.map2T({ grad: false } as Engine, a, b, f, () => 0, () => 0, dtype, 'BitwiseBackward0')
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
    if (this.t.dtype === 'bool') throw h.err('NotImplementedError', 'Negation, the `-` operator, on a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.')
    return wrap(C.unary(E, 'neg', this.t))
  }
  setAttr(name: string, v: Value, h: Host): boolean {
    if (name === 'requires_grad') {
      const b = h.truthy(v)
      if (!this.t.isLeaf && !b) throw h.err('RuntimeError', "you can only change requires_grad flags of leaf variables. If you want to use a computed variable in a subgraph that doesn't require differentiation use var_no_grad = var.detach().")
      if (b && !isFloatT(this.t.dtype)) throw h.err('RuntimeError', 'only Tensors of floating point and complex dtype can require gradients')
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
      case 'grad_fn': return t.fn ? py.obj(new GradFnObj(t.fn.name, t.fn)) : py.NONE
      case 'retains_grad': return py.bool(t.retain)
      case 'real': return self
      case 'mT': case 'mH': {
        if (t.ndim < 2) throw h.err('RuntimeError', `tensor.${name} is only supported on matrices or batches of matrices. Got ${t.ndim}-D tensor.`)
        return wrap(C.transposeT(E, t, -2, -1))
      }
      case 'H': {
        if (t.ndim !== 2 && t.ndim !== 0) throw h.err('RuntimeError', `tensor.H is only supported on matrices (2-D tensors). Got ${t.ndim}-D tensor.`)
        return t.ndim ? wrap(C.transposeT(E, t, 0, 1)) : self
      }
      case 'is_cpu': return py.bool(true)
      case 'is_sparse': case 'is_quantized': case 'is_mkldnn': case 'is_meta': case 'is_nested': return py.bool(false)
      case 'itemsize': return py.int(C.itemSize(t.dtype))
      case 'nbytes': return py.int(C.itemSize(t.dtype) * t.size)
      case 'name': return py.NONE
      case '__repr__': return fn(() => py.str(this.repr()))
      case '__str__': return fn(() => py.str(this.repr()))
      case '__float__': return fn(() => py.float(this.asNumber(h)))
      case '__int__': return fn(() => py.int(Math.trunc(this.asNumber(h))))
      case '__index__': return fn(() => py.int(this.asIndex(h)))
      case '__bool__': return fn(() => py.bool(this.truthy(h)))
      case '__len__': return fn(() => py.int(this.len(h)))
      case '__abs__': case 'absolute': return fn(() => wrap(C.unary(E, 'abs', t)))
      case '__round__': return fn(() => { throw h.err('TypeError', "type Tensor doesn't define __round__ method") })
      case 'data': return wrap(C.detach(t))
      case 'ndim': return py.int(t.ndim)
      case 'T': {
        if (t.ndim > 2) noteOnce(h, 'T_nd', 'warn')
        const out = C.permuteT(E, t, t.shape.map((_, i) => t.ndim - 1 - i), 'PermuteBackward0')
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
        return scalarOf(t, t.values()[0])
      })
      case 'tolist': return fn(() => {
        const vals = t.values()
        const one = (v: number) => scalarOf(t, v)
        if (!t.ndim) return one(vals[0])
        const st = cStrides(t.shape)
        const rec = (d: number, base: number): Value => py.list(Array.from({ length: t.shape[d] }, (_, i) => (d === t.ndim - 1 ? one(vals[base + i]) : rec(d + 1, base + i * st[d]))))
        return rec(0, 0)
      })
      case 'numpy': return fn((args, kw) => {
        if (t.dtype === 'int8' || t.dtype === 'int16' || t.dtype === 'float16') throw h.err('NotImplementedError', `Tensor.numpy(): the sandbox's NumPy has no ${t.dtype} arrays`)
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
        if ((kw.inputs ?? args[3]) && (kw.inputs ?? args[3])!.k !== 'none') throw h.err('NotImplementedError', 'backward(inputs=…) is not available in this sandbox')
        const gt = g && g.k !== 'none' ? asTT(g) : null
        if (flagOf(h, kw.create_graph ?? args[2])) {
          const grads = C.backwardGraph(E, t, gt)
          for (const [x, gx] of grads) {
            if (x.fn && !x.retain) continue
            // a leaf's gradient is a copy that remembers where it came from
            x.grad = x.grad ? C.binary(E, '+', x.grad, gx) : C.renamed(C.cloneT(E, gx), 'CopyBackwards')
          }
          return py.NONE
        }
        doBackward(h, t, self, gt ? gt.values() : null, flagOf(h, kw.retain_graph ?? args[1]), gt?.shape)
        return py.NONE
      })
      case 'retain_grad': return fn(() => { t.retain = true; return py.NONE })
      case 'requires_grad_': return fn((args, kw) => {
        const v = args[0] ?? kw.requires_grad ?? py.bool(true)
        if (h.truthy(v) && !isFloatT(t.dtype)) throw h.err('RuntimeError', 'only Tensors of floating point dtype can require gradients')
        this.setAttr('requires_grad', v, h)
        return self
      })
      case 'zero_': return fn(() => this.rebase(h, (old) => C.mapT(E, old, () => 0, null, 'ZeroBackward0', true)))
      case 'fill_': return fn((args, kw) => {
        const given = asTT(args[0] ?? kw.value)
        if (given && given.ndim) throw h.err('RuntimeError', `fill_ only supports 0-dimension value tensor but got tensor with ${given.ndim} dimensions.`)
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
        const given = name === 'view' ? kw.size : kw.shape
        if (!args.length && !given) {
          if (name === 'reshape' && !Object.keys(kw).length) throw h.err('TypeError', 'reshape() missing 1 required positional arguments: "shape"')
          throw invalidArgs(h, name, args, kw)
        }
        if (args.some((a) => a.k === 'str' || a.k === 'float' || a.k === 'none')) throw invalidArgs(h, name, args, kw)
        const dims = toDims(h, given ? [given] : args)
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
        return wrap(t.ndim < 2 ? t : C.permuteT(E, t, [1, 0], 'TBackward0'))
      })
      case 'sum': case 'mean': return fn((args, kw) => {
        for (const k in kw) if (!['dim', 'keepdim', 'dtype', 'axis'].includes(k)) {
          if (args.length) throw h.err('TypeError', `${name}() got an unexpected keyword argument '${k}'`)
          throw invalidArgs(h, name, args, kw)
        }
        if (args.length > 2) throw h.err('TypeError', `${name}() takes from 1 to 2 positional arguments but ${args.length} were given`)
        if (kw.dtype && kw.dtype.k !== 'none' && !(kw.dtype.k === 'obj' && kw.dtype.o instanceof DtypeObj)) throw invalidArgs(h, name, args, kw)
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
        if (!t.size && optDim(h, kw.dim ?? args[0]) === null) throw h.err('IndexError', `${name}(): Expected reduction dim to be specified for input.numel() == 0.`)
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
        if (!dim) throw h.err('TypeError', `${name}() missing 1 required positional arguments: "dim"`)
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
        if (!other) throw invalidArgs(h, name, args, kw)
        const o = operand(h, other, t.dtype)
        if (!o) throw h.err('TypeError', `${name}(): argument 'other' (position 1) must be Tensor, not ${typeName(other)}`)
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
      case 'half': case 'short': case 'char': return fn(() => wrap(C.toDtype(E, t, ({ half: 'float16', short: 'int16', char: 'int8' } as Record<string, TDType>)[name])))
      case 'type': return fn((args, kw) => {
        const v = kw.dtype ?? args[0]
        if (v?.k === 'str') {
          const named = Object.entries(LEGACY).find(([n]) => v.v === `torch.${n}`)
          if (!named) throw h.err('ValueError', `invalid type: '${v.v}'`)
          return wrap(C.toDtype(E, t, named[1]))
        }
        const d = dtypeArg(h, v)
        return d ? wrap(C.toDtype(E, t, d)) : py.str(`torch.${C.typeLabel(t.dtype)}Tensor`)
      })
      case 'to': return fn((args, kw) => {
        const other = kw.other ?? args.find((a) => asTT(a))
        const target = kw.dtype ?? args.find((a) => a.k === 'obj' && a.o instanceof DtypeObj)
        const d = other ? asTT(other)!.dtype : target ? dtypeArg(h, target) : null
        for (const k in kw) if (!['other', 'dtype', 'device', 'non_blocking', 'copy', 'memory_format'].includes(k)) throw invalidArgs(h, 'to', args, kw)
        const dev = kw.device ?? args.find((a) => a.k === 'str' || a.k === 'int' || (a.k === 'obj' && a.o instanceof DeviceObj))
        onlyCpu(h, dev)
        const copied = flagOf(h, kw.copy)
        return d && d !== t.dtype ? wrap(C.toDtype(E, t, d)) : copied ? wrap(C.cloneT(E, t)) : self
      })
      case 'cpu': return fn(() => self)
      case 'cuda': return fn(() => { throw h.err('AssertionError', 'Torch not compiled with CUDA enabled') })
      case 'cholesky': case 'solve': case 'lstsq': case 'eig': case 'symeig': case 'qr': return fn(() => { throw h.err('RuntimeError', REMOVED[name]) })
      case 'logdet': case 'slogdet': case 'svd': case 'pinverse': case 'matrix_exp': case 'lu': case 'bernoulli': case 'multinomial': case 'hardshrink': case 'prelu': case 'mvlgamma':
        return fn((args, kw) => h.call(h.getAttr(h.module('torch'), name), [self, ...args], kw))
      case 'polygamma': return fn((args, kw) => h.call(h.getAttr(h.module('torch'), 'polygamma'), [args[0] ?? kw.n, self]))
      case 'ndimension': return fn(() => py.int(t.ndim))
      case 'element_size': return fn(() => py.int(C.itemSize(t.dtype)))
      case 'is_inference': case 'is_pinned': case 'is_conj': case 'is_neg': case 'is_distributed': return fn(() => py.bool(false))
      case 'pin_memory': return fn(() => { throw h.err('RuntimeError', "Cannot access accelerator device when none is available.") })
      case 'layout': return py.obj(new DtypeLike('torch.strided'))
      case 'output_nr': return py.int(0)
      case 'apply_': return fn((args) => {
        if (t.requiresGrad && E.grad) throw h.err('RuntimeError', 'apply_ is only implemented on CPU tensors')
        C.writeInto(t, t.values().map((v) => h.num(h.call(args[0], [scalarOf(t, v)]))))
        t.version++
        return self
      })
      case 'map_': return fn((args) => {
        const o = tensorArg(h, args[0], 'map_', 'tensor')
        const ov = C.expandT({ grad: false } as Engine, o, t.shape).values()
        C.writeInto(t, t.values().map((v, i) => h.num(h.call(args[1], [scalarOf(t, v), scalarOf(o, ov[i])]))))
        return self
      })
      case 'register_hook': return fn((args, kw) => {
        const f = args[0] ?? kw.hook
        if (!t.requiresGrad) throw h.err('RuntimeError', "cannot register a hook on a tensor that doesn't require gradient")
        const hook = (g: TT) => {
          const r = h.call(f, [wrap(g)])
          return r.k === 'none' ? null : asTT(r)
        }
        ;(t.hooks ??= []).push(hook)
        return handle(() => {
          const at = t.hooks!.indexOf(hook)
          if (at >= 0) t.hooks!.splice(at, 1)
        })
      })
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
  throw h.err('RuntimeError', a.ndim === 1 && b.ndim === 1 ? `dot : expected both vectors to have same dtype, but found ${x} and ${y}` : `expected m1 and m2 to have the same dtype, but got: ${CONV_TYPE[a.dtype]} != ${CONV_TYPE[b.dtype]}`)
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
  if (dimV.k !== 'int' && dimV.k !== 'bool') throw invalidArgs(h, isMin ? 'min' : 'max', [dimV], keepV ? { keepdim: keepV } : {})
  const dim = h.toInt(dimV)
  const r = C.maxAlong(E, t, dim, !!(keepV && h.truthy(keepV)), isMin)
  trace(h, `torch.${isMin ? 'min' : 'max'}`, 'reduce', [{ label: label(h, self, 'x'), t }], r.values, C.normDim(dim, t.ndim))
  return results(isMin ? 'min' : 'max', r)
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
const bind = (h: Host, fname: string, params0: string, args: Value[], kw: Kw): Record<string, Value | undefined> => {
  // name! is a required parameter, reported as PyTorch's argument parser reports it
  const names = params0 ? params0.replace(/!/g, '').split(' ') : []
  const required = params0 ? params0.split(' ').filter((n) => n.endsWith('!')).map((n) => n.slice(0, -1)) : []
  const star = names.findIndex((n) => n.startsWith('*'))
  const fixed = star >= 0 ? names.slice(0, star) : names
  if (star < 0 && args.length > names.length) throw h.err('TypeError', `${fname}() takes ${names.length} positional argument${names.length === 1 ? '' : 's'} but ${args.length} were given`)
  const out: Record<string, Value | undefined> = {}
  fixed.forEach((n, i) => (out[n] = args[i]))
  if (star >= 0 && args.length > star) out[names[star].slice(1)] = py.tuple(args.slice(star))
  for (const k in kw) {
    if (!names.includes(k) && !names.includes(`*${k}`)) {
      throw h.err('TypeError', `${fname}() got an unexpected keyword argument '${k}'`)
    }
    out[k] = kw[k]
  }
  const missing = required.filter((n) => out[n] === undefined)
  if (missing.length) throw h.err('TypeError', `${fname}() missing ${missing.length} required positional argument${missing.length === 1 ? 's' : ''}: ${missing.map((n) => `"${n}"`).join(', ')}`)
  return out
}

/** a default written in a signature: 0.5, True, None, 'mean' */
const parseDefault = (s: string): Value => {
  if (s === 'True' || s === 'False') return py.bool(s === 'True')
  if (s === 'None') return py.NONE
  if (s.startsWith("'")) return py.str(s.slice(1, -1))
  if (s.startsWith('(')) return py.tuple(s.slice(1, -1).split(',').map(parseDefault))
  return /[.e]/.test(s) || s.includes('inf') ? py.float(Number(s)) : py.int(Number(s))
}

/**
 * Arguments for something PyTorch writes in Python, from its signature: 'in_features out_features bias=True'. A name
 * without a default is required; after a lone * they are keyword-only. Missing and surplus arguments read as Python's.
 */
const bindPy = (h: Host, fname: string, params: string, args: Value[], kw: Kw): Record<string, Value | undefined> => {
  const tokens = params ? params.split(' ') : []
  const star = tokens.indexOf('*')
  const fixed = (star >= 0 ? tokens.slice(0, star) : tokens).map((t) => t.split('=')[0])
  const all = tokens.filter((t) => t !== '*')
  const out: Record<string, Value | undefined> = {}
  const required: string[] = []
  for (const t of all) {
    const [n, d] = t.split('=')
    if (d === undefined) required.push(n)
    else out[n] = parseDefault(d)
  }
  if (args.length > fixed.length) {
    const positional = (star >= 0 ? tokens.slice(0, star) : tokens)
    const least = positional.filter((t) => !t.includes('=')).length
    const range = least === fixed.length ? `${fixed.length + 1}` : `from ${least + 1} to ${fixed.length + 1}`
    throw h.err('TypeError', `${fname}() takes ${range} positional argument${fixed.length === 0 ? '' : 's'} but ${args.length + 1} ${args.length === 0 ? 'was' : 'were'} given`)
  }
  fixed.forEach((n, i) => { if (i < args.length) out[n] = args[i] })
  for (const k in kw) {
    if (!all.some((t) => t.split('=')[0] === k)) throw h.err('TypeError', `${fname}() got an unexpected keyword argument '${k}'`)
    out[k] = kw[k]
  }
  const missing = required.filter((n) => out[n] === undefined)
  if (missing.length) {
    const names = missing.map((n) => `'${n}'`)
    throw h.err('TypeError', `${fname}() missing ${missing.length} required positional argument${missing.length > 1 ? 's' : ''}: ${names.length > 1 ? `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}` : names[0]}`)
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
  if (!v) throw c.h.err('TypeError', `${c.name}() missing 1 required positional arguments: "${key}"`)
  return v
}
const tArg = (c: TCall, key: string): TT => tensorArg(c.h, need(c, key), `${c.name}(): argument '${key}'`)
const intArg = (c: TCall, key: string, d?: number): number => {
  const v = c.a[key]
  if (v && v.k !== 'none') return c.h.toInt(v)
  if (d === undefined) throw c.h.err('TypeError', `${c.name}() missing 1 required positional arguments: "${key}"`)
  return d
}
const flag = (c: TCall, key: string, d = false) => flagOf(c.h, c.a[key], d)
const sizes = (c: TCall, key: string): number[] => toDims(c.h, c.a[key]?.k === 'tuple' ? c.a[key]!.items : [need(c, key)])
const withDtype = (c: TCall, key = 'dtype'): TT => {
  const d = dtypeArg(c.h, c.a[key])
  return d ? C.toDtype(c.E, c.t, d) : c.t
}

const results = (name: string, r: { values: TT; indices: TT }) => py.obj(new ReturnTypes(name, [['values', wrap(r.values)], ['indices', wrap(r.indices)]]))

/** median, mode, kthvalue along a dim: a step that reduces that dim, and the (values, indices) they return */
const picked = (c: TCall, name: string, dim: number, r: { values: TT; indices: TT }): Value => {
  trace(c.h, `Tensor.${name}`, 'reduce', [{ label: label(c.h, c.self, 'x'), t: c.t }], r.values, C.normDim(dim, c.t.ndim))
  return results(name, r)
}

/** x.expand(…) and torch.broadcast_to: -1 keeps a size, new dimensions go in front */
const expandTo = (h: Host, E: Engine, t: TT, target: number[]): TT => {
  const lead = target.length - t.ndim
  if (lead < 0) throw h.err('RuntimeError', `expand(torch.${C.typeLabel(t.dtype)}Tensor{[${t.shape.join(', ')}]}, size=[${target.join(', ')}]): the number of sizes provided (${target.length}) must be greater or equal to the number of dimensions in the tensor (${t.ndim})`)
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

/** ‖A‖ for a stack of matrices, dims = the two matrix axes; matrixNormOf in build() handles those that need singular values */
const matrixNorm = (h: Host, E: Engine, a: TT, ord: Value | undefined, dims: [number, number], keepdim: boolean): TT => {
  const [d0, d1] = dims.map((d) => C.normDim(d, a.ndim))
  if (!ord || ord.k === 'none' || (ord.k === 'str' && ord.v === 'fro')) return C.vectorNorm(E, a, 2, [d0, d1], keepdim)
  const p = ord.k === 'str' ? NaN : h.num(ord)
  if (Math.abs(p) !== 1 && Math.abs(p) !== Infinity) throw h.err('RuntimeError', `linalg.matrix_norm: Order ${repr(ord)} not supported.`)
  if (!isFloatT(a.dtype)) throw h.err('RuntimeError', `linalg.matrix_norm: Expected a floating point or complex tensor as input. Got ${C.typeLabel(a.dtype)}`)
  // ±1: sum over rows, then the extreme column; ±∞: the other way round
  const [sumOver, pickOver] = Math.abs(p) === 1 ? [d0, d1] : [d1, d0]
  const best = C.maxAlong(E, C.reduceSum(E, C.unary(E, 'abs', a), [sumOver], true, false), pickOver, true, p < 0).values
  return keepdim ? best : C.reshapeT(E, best, a.shape.filter((_, k) => k !== d0 && k !== d1), false)
}

// ---- special functions, to the precision float32 and float64 printing needs
const LANCZOS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
const lgammaFn = (x: number): number => {
  if (Number.isNaN(x)) return NaN
  if (!Number.isFinite(x)) return Infinity
  if (x <= 0 && Number.isInteger(x)) return Infinity
  if (x === 1 || x === 2) return 0
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgammaFn(1 - x)
  const y = x - 1
  const t = y + 7.5
  let a = LANCZOS[0]
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (y + i)
  return 0.5 * Math.log(2 * Math.PI) + (y + 0.5) * Math.log(t) - t + Math.log(a)
}
/** ψ⁽ⁿ⁾(x): upward recurrence to a large argument, then the asymptotic series */
const polygammaFn = (n: number, x: number): number => {
  if (Number.isNaN(x)) return NaN
  if (n === 0) {
    if (x === 0) return -Infinity
    if (x < 0) return Number.isInteger(x) ? NaN : polygammaFn(0, 1 - x) - Math.PI / Math.tan(Math.PI * x)
    let r = 0
    let y = x
    while (y < 10) {
      r -= 1 / y
      y += 1
    }
    const f = 1 / (y * y)
    return r + Math.log(y) - 0.5 / y - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))
  }
  if (x <= 0 && Number.isInteger(x)) return n % 2 ? Infinity : NaN
  const s = n + 1
  let fact = 1
  for (let k = 2; k <= n; k++) fact *= k
  let sum = 0
  let y = x
  while (y < 20) {
    sum += y ** -s
    y += 1
  }
  sum += y ** (1 - s) / (s - 1) + y ** -s / 2 + (s * y ** (-s - 1)) / 12 - (s * (s + 1) * (s + 2) * y ** (-s - 3)) / 720
  return (n % 2 ? 1 : -1) * fact * sum
}
const erfinvFn = (y: number): number => {
  if (Number.isNaN(y) || Math.abs(y) > 1) return NaN
  if (Math.abs(y) === 1) return y * Infinity
  if (y === 0) return y
  const w = Math.log(1 - y * y)
  const a = 0.147
  const b = 2 / (Math.PI * a) + w / 2
  let x = Math.sign(y) * Math.sqrt(Math.sqrt(b * b - w / a) - b)
  for (let i = 0; i < 3; i++) x -= (C.erf(x) - y) / ((2 / Math.sqrt(Math.PI)) * Math.exp(-x * x))
  return x
}
/** modified Bessel functions: power series while it converges quickly, the large-argument series after */
const besselI = (order: 0 | 1, x: number, scaled: boolean): number => {
  const a = Math.abs(x)
  if (a > 30) {
    const mu = 4 * order * order
    const t = 1 / (8 * a)
    const v = (1 - (mu - 1) * t + ((mu - 1) * (mu - 9) * t * t) / 2 - ((mu - 1) * (mu - 9) * (mu - 25) * t * t * t) / 6) / Math.sqrt(2 * Math.PI * a)
    return (order && x < 0 ? -1 : 1) * v * (scaled ? 1 : Math.exp(a))
  }
  const q = (x * x) / 4
  let term = order ? x / 2 : 1
  let sum = term
  for (let k = 1; k < 200; k++) {
    term *= q / (k * (k + order))
    sum += term
    if (Math.abs(term) < Math.abs(sum) * 1e-17) break
  }
  return scaled ? sum * Math.exp(-a) : sum
}
const ndtrFn = (x: number) => 0.5 * C.erfc(-x / Math.SQRT2)
const erfcxFn = (x: number): number => {
  if (x < 25) return Math.exp(x * x) * C.erfc(x)
  const t = 1 / (2 * x * x)
  return (1 - t + 3 * t * t - 15 * t * t * t) / (x * Math.sqrt(Math.PI))
}
const logNdtrFn = (x: number): number => (x > -5 ? Math.log(ndtrFn(x)) : -0.5 * x * x + Math.log(erfcxFn(-x / Math.SQRT2) / 2))
const entrFn = (x: number) => (Number.isNaN(x) ? NaN : x > 0 ? -x * Math.log(x) : x === 0 ? 0 : -Infinity)
const sincFn = (x: number) => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x))
const sincD = (x: number) => (x === 0 ? 0 : (Math.cos(Math.PI * x) * Math.PI * x - Math.sin(Math.PI * x)) / (Math.PI * x * x))

const UNARY: Record<string, [(x: number) => number, ((x: number) => number) | null, string, boolean?]> = {
  asinh: [Math.asinh, (x) => 1 / Math.sqrt(x * x + 1), 'AsinhBackward0'],
  acosh: [Math.acosh, (x) => 1 / Math.sqrt(x * x - 1), 'AcoshBackward0'],
  atanh: [Math.atanh, (x) => 1 / (1 - x * x), 'AtanhBackward0'],
  arcsin: [Math.asin, (x) => 1 / Math.sqrt(1 - x * x), 'AsinBackward0'],
  arccos: [Math.acos, (x) => -1 / Math.sqrt(1 - x * x), 'AcosBackward0'],
  arctan: [Math.atan, (x) => 1 / (1 + x * x), 'AtanBackward0'],
  arcsinh: [Math.asinh, (x) => 1 / Math.sqrt(x * x + 1), 'AsinhBackward0'],
  arccosh: [Math.acosh, (x) => 1 / Math.sqrt(x * x - 1), 'AcoshBackward0'],
  arctanh: [Math.atanh, (x) => 1 / (1 - x * x), 'AtanhBackward0'],
  deg2rad: [(x) => (x * Math.PI) / 180, () => Math.PI / 180, 'Deg2RadBackward0'],
  rad2deg: [(x) => (x * 180) / Math.PI, () => 180 / Math.PI, 'Rad2DegBackward0'],
  sinc: [sincFn, sincD, 'SincBackward0'],
  sgn: [Math.sign, null, 'SgnBackward0', true],
  fix: [Math.trunc, null, 'TruncBackward0', true],
  lgamma: [lgammaFn, (x) => polygammaFn(0, x), 'LgammaBackward0'],
  digamma: [(x) => polygammaFn(0, x), (x) => polygammaFn(1, x), 'DigammaBackward0'],
  erfinv: [erfinvFn, (x) => (Math.sqrt(Math.PI) / 2) * Math.exp(erfinvFn(x) ** 2), 'ErfinvBackward0'],
  i0: [(x) => besselI(0, x, false), (x) => besselI(1, x, false), 'I0Backward0'],
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

const named2 = (t: TT, name: string): TT => {
  if (t.fn) t.fn.name = name
  return t
}
const gcdOf = (a: number, b: number): number => {
  let [x, y] = [Math.abs(a), Math.abs(b)]
  while (y) [x, y] = [y, x % y]
  return x
}
/** two float tensors, or two integer tensors that become floats */
const floatPair = (f: (x: number, y: number) => number, dfx: (x: number, y: number) => number, dfy: (x: number, y: number) => number, grad: string): TMethod => ['other!', (c) => {
  const o = tensorArg(c.h, c.a.other, c.name, 'other', 2)
  const d = C.promote(c.t.dtype, o.dtype)
  if (!isFloatT(d)) throw c.h.err('RuntimeError', `${c.name}: expected a floating point input, got ${C.typeLabel(d)}`)
  return done(c, C.map2T(c.E, c.t, o, f, dfx, dfy, d, grad), 'elementwise')
}]
/** a tensor or a number on the right: the gradient's name says which */
const scalarPair = (f: (x: number, y: number) => number, dfx: (x: number, y: number) => number, dfy: (x: number, y: number) => number, grad: string): TMethod => ['other!', (c) => {
  const v = c.a.other!
  const o = operand(c.h, v, c.t.dtype)
  if (!o) throw c.h.err('TypeError', `${c.name}(): argument 'other' must be Tensor, not ${typeName(v)}`)
  const d = C.promote(c.t.dtype, o.t.dtype, false, o.weak)
  return done(c, C.map2T(c.E, c.t, o.t, f, dfx, dfy, isFloatT(d) ? d : C.DEFAULTS.float, `${grad}${o.weak ? 1 : 0}`), 'elementwise')
}]
const anyPair = (f: (x: number, y: number) => number, dfx: (x: number, y: number) => number, dfy: (x: number, y: number) => number, grad: string): TMethod => ['other!', (c) => {
  const o = tensorArg(c.h, c.a.other, c.name, 'other', 2)
  return done(c, C.map2T(c.E, c.t, o, f, dfx, dfy, C.promote(c.t.dtype, o.dtype), grad), 'elementwise')
}]
const intPair = (name: string, f: (x: number, y: number) => number): TMethod => ['other!', (c) => {
  const o = tensorArg(c.h, c.a.other, c.name, 'other', 2)
  const d = C.promote(c.t.dtype, o.dtype)
  if (isFloatT(d) || d === 'bool') throw c.h.err('NotImplementedError', `"${name}_cpu" not implemented for '${C.typeLabel(d)}'`)
  return done(c, C.map2T({ grad: false } as Engine, c.t, o, f, () => 0, () => 0, d, 'GcdBackward0'), 'elementwise')
}]
const bitwise = (op: string): TMethod => ['other!', (c) => objOf(c.t).binop(op, c.a.other!, false, c.h) as Value]
const nanMedianAt = (vals: number[]): number => {
  const order = vals.map((x, i): [number, number] => [x, i]).filter(([x]) => !Number.isNaN(x)).sort((p, q) => p[0] - q[0] || p[1] - q[1])
  return order.length ? order[Math.floor((order.length - 1) / 2)][1] : 0
}
/** Tensor.index_add: a vector of positions along d, spread over the other axes for scatterT */
const expandIndexAlong = (index: TT, d: number, shape: number[]): TT => {
  const iv = index.values()
  const st = cStrides(shape)
  return TT.of(Array.from({ length: prod(shape) }, (_, f) => iv[Math.floor(f / st[d]) % shape[d]]), shape, 'int64')
}
const quantileOf = (c: TCall, skipNan: boolean): Value => {
  const { t, h, E } = c
  const fname = skipNan ? 'nanquantile' : 'quantile'
  if (!isFloatT(t.dtype)) throw h.err('RuntimeError', `${fname}() input tensor must be either float or double dtype`)
  const qV = c.a.q!
  const qT = asTT(qV)
  if (qT && qT.dtype !== t.dtype) throw h.err('RuntimeError', `${fname}() q tensor must be same dtype as the input tensor`)
  if (qT && qT.ndim > 1) throw h.err('RuntimeError', `${fname}() q must be a scalar or 1D tensor`)
  const qs = qT ? qT.values() : [h.num(qV)]
  const badQ = qs.find((q) => !(q >= 0 && q <= 1))
  if (badQ !== undefined) throw h.err('RuntimeError', qT ? `${fname}() q values must be in the range [0, 1]` : `${fname}() q must be in the range [0, 1] but got ${badQ}`)
  const how = c.a.interpolation ? h.str(c.a.interpolation) : 'linear'
  if (!['linear', 'lower', 'higher', 'midpoint', 'nearest'].includes(how)) throw h.err('RuntimeError', `${fname}() interpolation must be one of linear, lower, higher, midpoint or nearest, but got ${how}`)
  if (!t.size) throw h.err('RuntimeError', `${fname}() input tensor must be non-empty`)
  const dimV = optDim(h, c.a.dim)
  const keep = flag(c, 'keepdim')
  const x = dimV === null ? C.reshapeT(E, t, [-1], false) : t
  const d = dimV === null ? 0 : C.normDim(dimV, t.ndim)
  const sorted = C.sortT(E, x, d, false).values
  const S = sorted.shape
  const st = cStrides(S)
  const sv = sorted.values()
  const lines: number[] = []
  for (let f = 0; f < sv.length; f++) if (Math.floor(f / st[d]) % S[d] === 0) lines.push(f)
  const R = t.dtype === 'float64' ? (v: number) => v : Math.fround
  const lo: number[] = []
  const hi: number[] = []
  const wt: number[] = []
  for (const q of qs) {
    for (const base of lines) {
      let n = S[d]
      let nan = false
      for (let k = 0; k < S[d]; k++) {
        if (Number.isNaN(sv[base + k * st[d]])) {
          if (skipNan) {
            n = k
            break
          }
          nan = true
          break
        }
      }
      const rank = nan ? S[d] - 1 : n === 0 ? 0 : R(q * (n - 1))
      let r = rank
      if (how === 'lower') r = Math.floor(rank)
      else if (how === 'higher') r = Math.ceil(rank)
      else if (how === 'nearest') r = roundHalfEvenT(rank, 0)
      const below = Math.floor(r)
      const above = Math.ceil(r)
      lo.push(base + below * st[d])
      hi.push(base + above * st[d])
      wt.push(how === 'midpoint' ? (below === above ? 0 : 0.5) : r - below)
    }
  }
  const lead = qT && qT.ndim ? [qs.length] : []
  const rest = S.map((m, k) => (k === d ? 1 : m))
  const reduced = keep ? (dimV === null ? t.shape.map(() => 1) : rest) : S.filter((_, k) => k !== d)
  const outShape = [...lead, ...reduced]
  const a = C.takeT(E, sorted, lo, outShape, 'IndexBackward0')
  const b = C.takeT(E, sorted, hi, outShape, 'IndexBackward0')
  const w = TT.of(wt, outShape, t.dtype)
  const out = C.binary(E, '+', a, C.binary(E, '*', C.binary(E, '-', b, a), w))
  return done(c, named2(out, 'SqueezeBackward4'), 'reduce', dimV === null ? null : d)
}
const cumExtreme = (c: TCall, min: boolean): Value => {
  const { t, E } = c
  const name = min ? 'cummin' : 'cummax'
  if (!t.ndim) return py.obj(new ReturnTypes(name, [['values', wrap(C.cloneT(E, t))], ['indices', wrap(TT.scalar(0, 'int64'))]]))
  const d = C.normDim(intArg(c, 'dim'), t.ndim)
  const st = cStrides(t.shape)
  const vals = t.values()
  const src = new Array<number>(vals.length)
  const idx = new Array<number>(vals.length)
  for (let f = 0; f < vals.length; f++) {
    const i = Math.floor(f / st[d]) % t.shape[d]
    if (i === 0) {
      src[f] = f
      idx[f] = 0
      continue
    }
    const prev = src[f - st[d]]
    const better = Number.isNaN(vals[f]) || (min ? vals[f] <= vals[prev] : vals[f] >= vals[prev])
    src[f] = better && !Number.isNaN(vals[prev]) ? f : prev
    idx[f] = better && !Number.isNaN(vals[prev]) ? i : idx[f - st[d]]
  }
  return py.obj(new ReturnTypes(name, [['values', wrap(C.takeT(E, t, src, t.shape, min ? 'CumminBackward0' : 'CummaxBackward0'))], ['indices', wrap(TT.of(idx, t.shape, 'int64'))]]))
}
const histogramOf = (c: TCall): Value => {
  const { t, h } = c
  if (!isFloatT(t.dtype)) throw h.err('NotImplementedError', `"histogramdd" not implemented for '${C.typeLabel(t.dtype)}'`)
  const binsV = c.a.bins!
  const vals = t.values()
  const R = t.dtype === 'float64' ? (v: number) => v : Math.fround
  let edges: number[]
  if (asTT(binsV)) edges = asTT(binsV)!.values()
  else {
    const n = h.toInt(binsV)
    if (n <= 0) throw h.err('RuntimeError', `histogram(): bins must be > 0, but got ${n} for dimension 0`)
    let lo: number
    let hi: number
    if (c.a.range && c.a.range.k !== 'none') {
      [lo, hi] = h.iterate(c.a.range).map((v) => h.num(v))
      if (lo > hi) throw h.err('RuntimeError', `histogram(): min should not exceed max, but got min = ${lo} max = ${hi}`)
    } else {
      lo = vals.length ? vals.reduce((m, x) => Math.min(m, x), Infinity) : 0
      hi = vals.length ? vals.reduce((m, x) => Math.max(m, x), -Infinity) : 0
    }
    if (lo === hi) {
      lo -= 1
      hi += 1
    }
    edges = Array.from({ length: n + 1 }, (_, i) => (i === n ? R(hi) : R(lo + ((hi - lo) * i) / n)))
  }
  const w = c.a.weight && c.a.weight.k !== 'none' ? tArg(c, 'weight').values() : null
  const n = edges.length - 1
  const out = new Array<number>(n).fill(0)
  vals.forEach((x, f) => {
    if (x < edges[0] || x > edges[n] || Number.isNaN(x)) return
    let i = 0
    while (i < n - 1 && x >= edges[i + 1]) i++
    out[i] += w ? w[f] : 1
  })
  const res = flag(c, 'density') ? (() => {
    const total = out.reduce((p, q) => p + q, 0)
    return out.map((v, i) => v / total / (edges[i + 1] - edges[i]))
  })() : out
  return py.obj(new ReturnTypes('histogram', [['hist', wrap(TT.of(res, [n], t.dtype))], ['bin_edges', wrap(TT.of(edges, [n + 1], t.dtype))]]))
}
const trapezoidOf = (c: TCall, cumulative: boolean): Value => {
  const { t, h, E } = c
  const d = C.normDim(intArg(c, 'dim', -1), t.ndim)
  const n = t.shape[d]
  const y = isFloatT(t.dtype) ? t : C.toDtype(E, t, C.DEFAULTS.float)
  const avg = C.binary(E, '+', C.narrowT(E, y, d, 1, n - 1), C.narrowT(E, y, d, 0, n - 1))
  const xV = c.a.x
  let widths: TT
  if (xV && xV.k !== 'none') {
    const xt = tensorArg(h, xV, c.name, 'x', 2)
    const reshaped = xt.ndim === 1 ? C.reshapeT(E, xt, y.shape.map((_, k) => (k === d ? xt.size : 1)), false) : xt
    const dx = reshaped.shape[d] - 1
    widths = C.binary(E, '-', C.narrowT(E, reshaped, d, 1, dx), C.narrowT(E, reshaped, d, 0, dx))
  } else widths = TT.scalar(numOr(c, 'dx', 1), y.dtype)
  const areas = C.binary(E, '/', C.binary(E, '*', avg, widths), TT.scalar(2, y.dtype), false, true)
  const given = !!xV && xV.k !== 'none'
  return wrap(named2(cumulative ? C.cumT(E, areas, d, false) : C.reduceSum(E, areas, [d], false, false), given ? 'DivBackward1' : cumulative ? 'CumsumBackward0' : 'MulBackward1'))
}
const covOf = (c: TCall): Value => {
  const { t, h, E } = c
  if (t.ndim > 2) throw h.err('RuntimeError', `cov(): expected input to have two or fewer dimensions but got an input with ${t.ndim} dimensions`)
  const x = t.ndim < 2 ? C.reshapeT(E, t, [1, -1], false) : isFloatT(t.dtype) ? t : C.toDtype(E, t, C.DEFAULTS.float)
  const xf = isFloatT(x.dtype) ? x : C.toDtype(E, x, C.DEFAULTS.float)
  const obs = xf.shape[1]
  const corr = numOr(c, 'correction', 1)
  const fw = c.a.fweights && c.a.fweights.k !== 'none' ? tArg(c, 'fweights').values() : null
  const aw = c.a.aweights && c.a.aweights.k !== 'none' ? tArg(c, 'aweights').values() : null
  const w = Array.from({ length: obs }, (_, i) => (fw ? fw[i] : 1) * (aw ? aw[i] : 1))
  const sumW = w.reduce((p, q) => p + q, 0)
  const wT = TT.of(w, [1, obs], xf.dtype)
  const avg = C.binary(E, '/', C.reduceSum(E, C.binary(E, '*', xf, wT), [1], true, false), TT.scalar(sumW, xf.dtype), false, true)
  const m = C.binary(E, '-', xf, avg)
  const norm = corr === 0 ? sumW : aw ? sumW - (corr * w.reduce((p, q, i) => p + q * aw[i], 0)) / sumW : sumW - corr
  const out = C.binary(E, '/', C.matmul(E, C.binary(E, '*', m, wT), C.transposeT(E, m, 0, 1)), TT.scalar(Math.max(norm, 0), xf.dtype), false, true)
  return wrap(named2(xf.shape[0] === 1 ? C.reshapeT(E, out, [], false) : C.cloneT(E, out), 'SqueezeBackward0'))
}
/** a × b along axis d, the three components stacked again */
const crossProduct = (E: Engine, a: TT, b: TT, d: number): TT => {
  const at = (x: TT, i: number) => C.narrowT(E, x, d, i, 1)
  const part = (i: number, j: number) => C.binary(E, '-', C.binary(E, '*', at(a, i), at(b, j)), C.binary(E, '*', at(a, j), at(b, i)))
  return C.catT(E, [part(1, 2), part(2, 0), part(0, 1)], d)
}
const splitAlong = (c: TCall, name: string, d: number, least: number): Value => {
  const { t, h } = c
  if (t.ndim < least) throw h.err('RuntimeError', `torch.${name} requires a tensor with at least ${least} dimension, but got a tensor with ${t.ndim} dimensions!`)
  const v = need(c, 'indices_or_sections')
  if (v.k === 'int' && (v.v === 0 || t.shape[d] % v.v)) throw h.err('RuntimeError', `torch.${name} attempted to split along dimension ${d}, but the size of the dimension ${t.shape[d]} is not divisible by the split_size ${v.v}!`)
  return h.call(h.getAttr(c.self, 'tensor_split'), [v], { dim: py.int(d) })
}
const diagEmbedOf = (c: TCall, x: TT, k: number, dim1: number, dim2: number): Value => {
  const { h, E } = c
  const nd = x.ndim + 1
  const [a, b] = [C.normDim(dim1, nd), C.normDim(dim2, nd)]
  if (a === b) throw h.err('RuntimeError', `diagonal dimensions cannot be identical ${dim1}, ${dim2}`)
  const n = (x.shape[x.ndim - 1] ?? 1) + Math.abs(k)
  const lead = x.shape.slice(0, -1)
  // output position → input position, or -1 away from the diagonal
  const shape: number[] = []
  let leadAt = 0
  for (let q = 0; q < nd; q++) shape.push(q === a || q === b ? n : lead[leadAt++])
  const st = cStrides(shape)
  const leadSt = cStrides(lead)
  const leadAxes = shape.map((_, q) => q).filter((q) => q !== a && q !== b)
  const m = x.shape[x.ndim - 1] ?? 1
  const src = new Array<number>(prod(shape))
  for (let f = 0; f < src.length; f++) {
    const [i, j] = [Math.floor(f / st[a]) % n, Math.floor(f / st[b]) % n]
    if (j - i !== k) {
      src[f] = -1
      continue
    }
    let base = 0
    leadAxes.forEach((q, l) => (base += (Math.floor(f / st[q]) % shape[q]) * leadSt[l]))
    src[f] = base * m + (k >= 0 ? i : j)
  }
  const pick = C.takeT(E, x, src.map((p) => Math.max(p, 0)), shape, 'DiagEmbedBackward0')
  const mask = TT.of(src.map((p) => (p >= 0 ? 1 : 0)), shape, 'bool')
  return wrap(named2(C.whereT(E, mask, pick, TT.scalar(0, x.dtype), x.dtype), 'DiagEmbedBackward0'))
}
const scatterReduced = (c: TCall, dim: number, index: TT, src: TT, how: string, includeSelf: boolean): TT => {
  const { t, h, E } = c
  const d = C.normDim(dim, t.ndim)
  if (isFloatT(index.dtype)) throw h.err('RuntimeError', 'scatter(): Expected dtype int64 for index')
  const iv = index.values()
  const st = cStrides(t.shape)
  const ist = cStrides(index.shape)
  const sv = src.values()
  const sst = cStrides(src.shape)
  const tv = t.values()
  const groups = new Map<number, number[]>()
  for (let f = 0; f < iv.length; f++) {
    let pos = 0
    let from = 0
    index.shape.forEach((_, q) => {
      const c0 = Math.floor(f / ist[q]) % index.shape[q]
      pos += (q === d ? iv[f] : c0) * st[q]
      from += c0 * sst[q]
    })
    const g = groups.get(pos) ?? []
    g.push(sv[from])
    groups.set(pos, g)
  }
  const out = [...tv]
  for (const [pos, g] of groups) {
    const all = includeSelf ? [tv[pos], ...g] : g
    out[pos] = how === 'sum' ? all.reduce((p, q) => p + q, 0) : how === 'prod' ? all.reduce((p, q) => p * q, 1)
      : how === 'mean' ? (isFloatT(t.dtype) ? all.reduce((p, q) => p + q, 0) / all.length : Math.floor(all.reduce((p, q) => p + q, 0) / all.length))
        : how === 'amax' ? all.reduce((p, q) => Math.max(p, q), -Infinity) : all.reduce((p, q) => Math.min(p, q), Infinity)
  }
  return C.noGradient(E, TT.of(out, t.shape, t.dtype), [t, src], c.name, c.name === 'index_reduce' ? 'IndexReduceBackward0' : 'ScatterReduceBackward0')
}

/** regularised incomplete gamma: series below a + 1, continued fraction above */
const gammaIncFn = (a: number, x: number, upper: boolean): number => {
  if (Number.isNaN(a) || Number.isNaN(x) || a < 0 || x < 0) return NaN
  if (a === 0) return x === 0 ? NaN : upper ? 0 : 1
  if (x === 0) return upper ? 1 : 0
  if (x === Infinity) return upper ? 0 : 1
  const front = Math.exp(-x + a * Math.log(x) - lgammaFn(a))
  if (x < a + 1) {
    let [sum, term, n] = [1 / a, 1 / a, a]
    for (let i = 0; i < 500; i++) {
      n += 1
      term *= x / n
      sum += term
      if (Math.abs(term) < Math.abs(sum) * 1e-16) break
    }
    return upper ? 1 - sum * front : sum * front
  }
  let b = x + 1 - a
  let c0 = 1e300
  let d = 1 / b
  let f = d
  for (let i = 1; i < 500; i++) {
    const an = -i * (i - a)
    b += 2
    d = an * d + b
    if (Math.abs(d) < 1e-300) d = 1e-300
    c0 = b + an / c0
    if (Math.abs(c0) < 1e-300) c0 = 1e-300
    d = 1 / d
    const del = d * c0
    f *= del
    if (Math.abs(del - 1) < 1e-16) break
  }
  return upper ? front * f : 1 - front * f
}
/** β·self + α·product, the addmm family */
const matAdd = (c: TCall, product: TT, name: string): Value => {
  const [beta, alpha] = [numOr(c, 'beta', 1), numOr(c, 'alpha', 1)]
  const scaled = (x: TT, k: number) => (k === 1 ? x : C.binary(c.E, '*', x, TT.scalar(k, x.dtype), false, true))
  return done(c, named2(C.binary(c.E, '+', scaled(c.t, beta), scaled(product, alpha)), name), 'matmul')
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
    if (!isFloatT(t.dtype)) throw h.err('NotImplementedError', `"lerp_kernel_scalar" not implemented for '${C.typeLabel(t.dtype)}'`)
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
  kthvalue: ['k! dim keepdim', (c) => picked(c, 'kthvalue', intArg(c, 'dim', -1), C.pickAlong(c.E, c.t, intArg(c, 'dim', -1), flag(c, 'keepdim'), C.kthAt(intArg(c, 'k')), 'KthvalueBackward0'))],
  amax: ['dim keepdim', (c) => amaxMin(c, false)],
  amin: ['dim keepdim', (c) => amaxMin(c, true)],
  norm: ['p dim keepdim dtype', (c) => {
    const { h, E } = c
    const p = c.a.p
    const t = withDtype(c)
    const dims = dimsArg(h, c.a.dim)
    if (p?.k === 'str' && p.v === 'nuc') return h.call(h.getAttr(h.module('torch.linalg'), 'matrix_norm'), [c.self, p], c.a.keepdim ? { keepdim: c.a.keepdim } : {})
    if (p?.k === 'str' && p.v !== 'fro') throw h.err('ValueError', `Unknown norm type ${p.v}`)
    const ord = !p || p.k === 'none' || p.k === 'str' ? 2 : h.num(p)
    return done(c, C.vectorNorm(E, t, ord, dims, flag(c, 'keepdim')), 'reduce', reduceAxis(c, dims))
  }],
  trace: ['', (c) => {
    const { h, E, t } = c
    if (t.ndim !== 2) throw h.err('RuntimeError', `trace: expected a matrix, but got tensor with dim ${t.ndim}`)
    const out = C.reduceSum(E, diagonalOf(E, t, 0, 0, 1, 'DiagonalBackward0'), null, false, false)
    if (out.fn) out.fn.name = 'TraceBackward0'
    return done(c, out, 'linalg')
  }],

  inverse: ['', (c) => done(c, C.invT(c.E, c.t), 'linalg')],
  det: ['', (c) => done(c, C.detT(c.E, c.t), 'linalg')],
  matrix_power: ['n', (c) => c.h.call(c.h.getAttr(c.h.module('torch.linalg'), 'matrix_power'), [c.self, need(c, 'n')])],

  // ---- cumulative and sorted
  cumsum: ['dim! dtype', (c) => done(c, C.cumT(c.E, withDtype(c), intArg(c, 'dim'), false), 'scan', C.normDim(intArg(c, 'dim'), c.t.ndim))],
  cumprod: ['dim! dtype', (c) => done(c, C.cumT(c.E, withDtype(c), intArg(c, 'dim'), true), 'scan', C.normDim(intArg(c, 'dim'), c.t.ndim))],
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
  topk: ['k! dim largest sorted', (c) => {
    const dim = intArg(c, 'dim', -1)
    const r = C.topkT(c.E, c.t, intArg(c, 'k'), dim, flag(c, 'largest', true))
    trace(c.h, 'Tensor.topk', 'sort', [{ label: label(c.h, c.self, 'x'), t: c.t }], r.values, C.normDim(dim, c.t.ndim), r.src)
    return results('topk', r)
  }],
  unique: ['sorted return_inverse return_counts dim', uniqueOf],
  bincount: ['weights minlength', (c) => {
    const { t, h } = c
    if (isFloatT(t.dtype)) throw h.err('NotImplementedError', `"bincount_cpu" not implemented for '${C.typeLabel(t.dtype)}'`)
    if (t.ndim > 1 || t.values().some((v) => v < 0)) throw h.err('RuntimeError', 'bincount only supports 1-d non-negative integral inputs.')
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
  gather: ['dim! index!', (c) => {
    const r = C.gatherT(c.E, c.t, intArg(c, 'dim'), tArg(c, 'index'))
    return done(c, r.out, 'move', null, r.src)
  }],
  scatter: ['dim index src value reduce', (c) => {
    const src = need(c, c.a.src ? 'src' : 'value')
    if (c.a.reduce) {
      const how = c.h.str(c.a.reduce)
      if (how !== 'add' && how !== 'multiply') throw c.h.err('RuntimeError', "reduce argument must be either add or multiply.")
      const index = tArg(c, 'index')
      const srcT = asTT(src) ?? TT.of(new Array<number>(index.size).fill(c.h.num(src)), index.shape, c.t.dtype)
      return done(c, scatterReduced(c, intArg(c, 'dim'), index, srcT, how === 'add' ? 'sum' : 'prod', true), 'group')
    }
    return done(c, C.scatterT(c.E, c.t, intArg(c, 'dim'), tArg(c, 'index'), asTT(src) ?? c.h.num(src), false), 'group')
  }],
  scatter_add: ['dim! index! src!', (c) => done(c, C.scatterT(c.E, c.t, intArg(c, 'dim'), tArg(c, 'index'), tArg(c, 'src'), true), 'group')],
  index_select: ['dim! index!', (c) => {
    const { t, h } = c
    const d = C.normDim(intArg(c, 'dim'), t.ndim)
    const index = tArg(c, 'index')
    if (index.ndim > 1) throw h.err('IndexError', 'index_select(): Index is supposed to be a vector')
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
  masked_select: ['mask!', (c) => {
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
  masked_fill: ['mask! value!', (c) => {
    const { t, E, h } = c
    const mask = tArg(c, 'mask')
    if (mask.dtype !== 'bool') throw h.err('RuntimeError', `masked_fill_ only supports boolean masks, but got mask with dtype ${CONV_TYPE[mask.dtype]}`)
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
  narrow: ['dim! start! length!', (c) => {
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
    if (k !== o.shape[o.ndim - 1]) throw h.err('RuntimeError', `inner() the last dimension must match on both input tensors but got shapes [${t.shape.join(', ')}] and [${o.shape.join(', ')}]`)
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
    if (!isFloatT(c.t.dtype)) throw c.h.err('NotImplementedError', `"normal_kernel_cpu" not implemented for '${C.typeLabel(c.t.dtype)}'`)
    const [mean, std] = [numOr(c, 'mean', 0), numOr(c, 'std', 1)]
    if (std < 0) throw c.h.err('RuntimeError', `normal expects std >= 0.0, but found std ${std}`)
    const rng = rngOf(c.h, c.a.generator)
    return filledIn(c, () => mean + std * rng.normal(), 'NormalBackward0')
  }],
  uniform_: ['from to generator', (c) => {
    if (!isFloatT(c.t.dtype)) throw c.h.err('NotImplementedError', `"check_uniform_bounds" not implemented for '${C.typeLabel(c.t.dtype)}'`)
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
    const v = h.num(need(c, 'fill_value'))
    const step = cStrides(t.shape).reduce((s, x) => s + x, 0)
    const n = Math.min(...t.shape)
    const vals = t.values()
    for (let i = 0; i < n; i++) vals[i * step] = v
    // a tall matrix: the diagonal starts again one row below where it ended
    if (flag(c, 'wrap') && t.ndim === 2 && t.shape[0] > t.shape[1] + 1) {
      const [rows, cols] = t.shape
      for (let r = cols + 1; r < rows; r += cols + 1) for (let i = 0; i < cols && r + i < rows; i++) vals[(r + i) * cols + i] = v
    }
    return objOf(t).rebase(h, () => TT.of(vals, t.shape, t.dtype))
  }],
  detach_: ['', (c) => {
    c.t.fn = null
    c.t.requiresGrad = false
    return c.self
  }],

  // ---- elementwise, more of them
  positive: ['', (c) => {
    if (c.t.dtype === 'bool') throw c.h.err('RuntimeError', 'The `+` operator, on a bool tensor is not supported.')
    return c.self
  }],
  signbit: ['', (c) => boolMap(c, (v) => v < 0 || Object.is(v, -0))],
  logit: ['eps', (c) => {
    const eps = optNumV(c.h, c.a.eps)
    const clip = (x: number) => (eps === null ? x : Math.min(1 - eps, Math.max(eps, x)))
    return done(c, C.mapT(c.E, c.t, (x) => Math.log(clip(x) / (1 - clip(x))), (x) => (eps !== null && (x < eps || x > 1 - eps) ? 0 : 1 / (x * (1 - x))), 'LogitBackward0'), 'elementwise')
  }],
  nan_to_num: ['nan posinf neginf', (c) => {
    const { t, h, E } = c
    if (!isFloatT(t.dtype)) return c.self
    const big = t.dtype === 'float64' ? Number.MAX_VALUE : t.dtype === 'float16' ? 65504 : 3.4028234663852886e38
    const [nan, pos, neg] = [optNumV(h, c.a.nan) ?? 0, optNumV(h, c.a.posinf) ?? big, optNumV(h, c.a.neginf) ?? -big]
    return done(c, C.mapT(E, t, (x) => (Number.isNaN(x) ? nan : x === Infinity ? pos : x === -Infinity ? neg : x), (x) => (Number.isFinite(x) ? 1 : 0), 'NanToNumBackward0'), 'elementwise')
  }],
  hypot: floatPair((x, y) => Math.hypot(x, y), (x, y) => x / Math.hypot(x, y), (x, y) => y / Math.hypot(x, y), 'HypotBackward0'),
  logaddexp: floatPair((x, y) => (x > y ? x + Math.log1p(Math.exp(y - x)) : y + Math.log1p(Math.exp(x - y))), (x, y) => 1 / (1 + Math.exp(y - x)), (x, y) => 1 / (1 + Math.exp(x - y)), 'LogaddexpBackward0'),
  logaddexp2: floatPair((x, y) => (x > y ? x + Math.log2(1 + 2 ** (y - x)) : y + Math.log2(1 + 2 ** (x - y))), (x, y) => 1 / (1 + 2 ** (y - x)), (x, y) => 1 / (1 + 2 ** (x - y)), 'Logaddexp2Backward0'),
  copysign: scalarPair((x, y) => (y < 0 || Object.is(y, -0) ? -Math.abs(x) : Math.abs(x)), (x, y) => ((y < 0 || Object.is(y, -0)) === (x < 0 || Object.is(x, -0)) ? 1 : -1), () => 0, 'CopysignBackward'),
  xlogy: scalarPair((x, y) => (x === 0 && !Number.isNaN(y) ? 0 : x * Math.log(y)), (_x, y) => Math.log(y), (x, y) => x / y, 'XlogyBackward'),
  fmax: anyPair((x, y) => (Number.isNaN(x) ? y : Number.isNaN(y) ? x : Math.max(x, y)), (x, y) => (Number.isNaN(y) || x >= y ? 1 : 0), (x, y) => (Number.isNaN(x) || y > x ? 1 : 0), 'FmaxBackward0'),
  fmin: anyPair((x, y) => (Number.isNaN(x) ? y : Number.isNaN(y) ? x : Math.min(x, y)), (x, y) => (Number.isNaN(y) || x <= y ? 1 : 0), (x, y) => (Number.isNaN(x) || y < x ? 1 : 0), 'FminBackward0'),
  gcd: intPair('gcd', gcdOf),
  lcm: intPair('lcm', (a, b) => (a === 0 || b === 0 ? 0 : Math.abs(a * b) / gcdOf(a, b))),
  heaviside: ['values', (c) => {
    const o = tArg(c, 'values')
    if (c.t.dtype !== o.dtype) throw c.h.err('RuntimeError', 'heaviside is not yet implemented for tensors with different dtypes.')
    return done(c, C.noGradient(c.E, C.map2T({ grad: false } as Engine, c.t, o, (x, y) => (x === 0 ? y : x < 0 ? 0 : 1), () => 0, () => 0, c.t.dtype, 'HeavisideBackward0'), [c.t, o], 'heaviside'), 'elementwise')
  }],
  float_power: ['exponent', (c) => {
    const { t, h, E } = c
    const v = need(c, 'exponent')
    const o = operand(h, v, 'float64')
    if (!o) throw h.err('TypeError', `float_power(): argument 'exponent' must be Tensor, not ${typeName(v)}`)
    return wrap(C.binary(E, '**', C.toDtype(E, t, 'float64'), C.toDtype(E, o.t, 'float64'), false, o.weak))
  }],
  ldexp: ['other', (c) => {
    const o = tArg(c, 'other')
    const two = C.binary(c.E, '**', TT.scalar(2, 'float32'), isFloatT(o.dtype) ? o : C.toDtype(c.E, o, 'float32'), true, false)
    return done(c, named2(C.binary(c.E, '*', c.t, two), 'LdexpBackward0'), 'elementwise')
  }],
  bitwise_and: bitwise('&'), bitwise_or: bitwise('|'), bitwise_xor: bitwise('^'), bitwise_left_shift: bitwise('<<'), bitwise_right_shift: bitwise('>>'),
  bitwise_not: ['', (c) => {
    if (isFloatT(c.t.dtype)) throw c.h.err('NotImplementedError', `"bitwise_not_cpu" not implemented for '${C.typeLabel(c.t.dtype)}'`)
    return objOf(c.t).unary('~', c.h)
  }],
  addcmul: ['tensor1 tensor2 value', (c) => {
    const [a, b] = [tArg(c, 'tensor1'), tArg(c, 'tensor2')]
    const v = numOr(c, 'value', 1)
    const prod2 = C.binary(c.E, '*', a, b)
    return done(c, named2(C.binary(c.E, '+', c.t, v === 1 ? prod2 : C.binary(c.E, '*', prod2, TT.scalar(v, prod2.dtype), false, true)), 'AddcmulBackward0'), 'elementwise')
  }],
  addcdiv: ['tensor1 tensor2 value', (c) => {
    const [a, b] = [tArg(c, 'tensor1'), tArg(c, 'tensor2')]
    if (!isFloatT(a.dtype) && !isFloatT(b.dtype)) throw c.h.err('RuntimeError', 'Integer division with addcdiv is no longer supported, and in a future  release addcdiv will perform a true division of tensor1 and tensor2. The historic addcdiv behavior can be implemented as (input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype) for integer inputs and as (input + value * tensor1 / tensor2) for float inputs. The future addcdiv behavior is just the latter implementation: (input + value * tensor1 / tensor2), for all dtypes.')
    const v = numOr(c, 'value', 1)
    const q = C.binary(c.E, '/', a, b)
    return done(c, named2(C.binary(c.E, '+', c.t, v === 1 ? q : C.binary(c.E, '*', q, TT.scalar(v, q.dtype), false, true)), 'AddcdivBackward0'), 'elementwise')
  }],
  isin: ['test_elements assume_unique invert', (c) => {
    const { h } = c
    const v = need(c, 'test_elements')
    const pool = new Set(asTT(v) ? asTT(v)!.values() : [h.num(v)])
    const inv = flag(c, 'invert')
    return boolMap(c, (x) => pool.has(x) !== inv)
  }],
  is_nonzero: ['', (c) => {
    if (c.t.size === 0) throw c.h.err('RuntimeError', 'Boolean value of Tensor with no values is ambiguous')
    if (c.t.size !== 1) throw c.h.err('RuntimeError', 'Boolean value of Tensor with more than one value is ambiguous')
    return py.bool(c.t.values()[0] !== 0)
  }],
  is_same_size: ['other', (c) => py.bool(tArg(c, 'other').shape.join() === c.t.shape.join())],
  is_signed: ['', (c) => py.bool(c.t.dtype !== 'bool' && c.t.dtype !== 'uint8')],
  is_shared: ['', (c) => py.bool(!!(c.t as { shared?: boolean }).shared)],
  share_memory_: ['', (c) => {
    (c.t as { shared?: boolean }).shared = true
    return c.self
  }],
  stride: ['dim', (c) => {
    const st = c.t.a.strides
    return c.a.dim ? py.int(st[C.normDim(intArg(c, 'dim'), c.t.ndim)]) : py.tuple(st.map(py.int))
  }],
  storage_offset: ['', (c) => py.int(c.t.a.offset)],

  // ---- reductions, more of them
  nansum: ['dim keepdim dtype', (c) => {
    const { E } = c
    const t = withDtype(c)
    const clean = C.mapT(E, t, (x) => (Number.isNaN(x) ? 0 : x), (x) => (Number.isNaN(x) ? 0 : 1), 'NanToNumBackward0', true)
    const dims = dimsArg(c.h, c.a.dim)
    return done(c, named2(C.reduceSum(E, clean, dims, flag(c, 'keepdim'), false), 'NansumBackward0'), 'reduce', reduceAxis(c, dims))
  }],
  nanmean: ['dim keepdim dtype', (c) => {
    const { E, h } = c
    const t = withDtype(c)
    if (!isFloatT(t.dtype)) throw h.err('RuntimeError', `nanmean(): expected input to have floating point or complex dtype but got ${C.typeLabel(t.dtype)}`)
    const clean = C.mapT(E, t, (x) => (Number.isNaN(x) ? 0 : x), (x) => (Number.isNaN(x) ? 0 : 1), 'NanToNumBackward0', true)
    const dims = dimsArg(h, c.a.dim)
    const keep = flag(c, 'keepdim')
    const counts = C.countReduce(TT.of(t.values().map((x) => (Number.isNaN(x) ? 0 : 1)), t.shape, 'bool'), dims, keep, t.dtype, (nz) => nz)
    return done(c, C.binary(E, '/', C.reduceSum(E, clean, dims, keep, false), counts), 'reduce', reduceAxis(c, dims))
  }],
  nanmedian: ['dim keepdim', (c) => {
    const { t, E, h } = c
    const dim = optDim(h, c.a.dim)
    if (dim === null) return done(c, C.sharedValueT(E, t, nanMedianAt(t.values()), 'NanmedianBackward0'), 'reduce')
    return picked(c, 'nanmedian', dim, C.pickAlong(E, t, dim, flag(c, 'keepdim'), nanMedianAt, 'NanmedianBackward1'))
  }],
  quantile: ['q! dim keepdim * interpolation', (c) => quantileOf(c, false)],
  nanquantile: ['q! dim keepdim * interpolation', (c) => quantileOf(c, true)],
  aminmax: ['* dim keepdim', (c) => {
    const { t, E, h } = c
    const dim = optDim(h, c.a.dim)
    if (!t.size && dim === null) throw h.err('RuntimeError', 'aminmax(): cannot compute aminmax over an empty dimension as the operation has no identity.')
    const keep = flag(c, 'keepdim')
    const one = (min: boolean) => named2(dim === null ? (keep ? C.reshapeT(E, C.extremeT(E, t, null, false, min, 'AminmaxBackward0'), t.shape.map(() => 1), false) : C.extremeT(E, t, null, false, min, 'AminmaxBackward0')) : C.extremeT(E, t, [dim], keep, min, 'AminmaxBackward0'), 'AminmaxBackward0')
    return py.obj(new ReturnTypes('aminmax', [['min', wrap(one(true))], ['max', wrap(one(false))]]))
  }],
  logsumexp: ['dim! keepdim', (c) => {
    const { t, E, h } = c
    if (!isFloatT(t.dtype)) return wrap(C.toDtype(E, t, 'float32'))
    const dims = dimsArg(h, c.a.dim) ?? []
    const keep = flag(c, 'keepdim')
    const peak = dims.reduce((x, d) => C.extremeT({ grad: false } as Engine, x, [d], true, false, 'AmaxBackward0'), t)
    const inner = C.reduceSum(E, C.unary(E, 'exp', C.binary(E, '-', t, peak)), dims, keep, false)
    const out = C.binary(E, '+', C.unary(E, 'log', inner), keep ? peak : C.reshapeT({ grad: false } as Engine, peak, inner.shape, false))
    return done(c, named2(out, 'LogsumexpBackward0'), 'reduce', reduceAxis(c, dims))
  }],
  cummax: ['dim!', (c) => cumExtreme(c, false)],
  cummin: ['dim!', (c) => cumExtreme(c, true)],
  logcumsumexp: ['dim!', (c) => {
    const { t, E, h } = c
    const d = C.normDim(intArg(c, 'dim'), t.ndim)
    if (!isFloatT(t.dtype)) throw h.err('RuntimeError', `logcumsumexp(): expected input to have floating point dtype but got ${C.typeLabel(t.dtype)}`)
    return done(c, named2(C.unary(E, 'log', C.cumT(E, C.unary(E, 'exp', t), d, false)), 'LogcumsumexpBackward0'), 'scan', d)
  }],
  diff: ['n dim prepend append', (c) => {
    const { t, E, h } = c
    let x = t
    const glue = (v: Value | undefined, front: boolean) => {
      if (!v || v.k === 'none') return
      x = C.catT(E, front ? [tensorArg(h, v, 'diff', 'prepend'), x] : [x, tensorArg(h, v, 'diff', 'append')], C.normDim(intArg(c, 'dim', -1), t.ndim))
    }
    const n = intArg(c, 'n', 1)
    if (n < 0) throw h.err('RuntimeError', `order must be non-negative but got ${n}`)
    if (!t.ndim) throw h.err('RuntimeError', 'diff expects input to be at least one-dimensional')
    glue(c.a.prepend, true)
    glue(c.a.append, false)
    const d = C.normDim(intArg(c, 'dim', -1), x.ndim)
    for (let i = 0; i < n; i++) {
      const len = x.shape[d]
      if (len <= 1) {
        x = C.narrowT(E, x, d, 0, 0)
        continue
      }
      const [hi, lo] = [C.narrowT(E, x, d, 1, len - 1), C.narrowT(E, x, d, 0, len - 1)]
      x = x.dtype === 'bool' ? C.map2T({ grad: false } as Engine, hi, lo, (p, q) => (p !== q ? 1 : 0), () => 0, () => 0, 'bool', 'XorBackward0') : C.binary(E, '-', hi, lo)
    }
    return done(c, x, 'group')
  }],
  histc: ['bins min max', (c) => {
    const { t, h } = c
    if (!isFloatT(t.dtype)) throw h.err('NotImplementedError', `"histogram_cpu" not implemented for '${C.typeLabel(t.dtype)}'`)
    const bins = intArg(c, 'bins', 100)
    if (bins <= 0) throw h.err('RuntimeError', 'bins must be > 0')
    let [lo, hi] = [numOr(c, 'min', 0), numOr(c, 'max', 0)]
    if (lo > hi) throw h.err('RuntimeError', 'max must be larger than min')
    const vals = t.values()
    if (lo === hi && vals.length) {
      lo = vals.reduce((m, x) => Math.min(m, x), Infinity)
      hi = vals.reduce((m, x) => Math.max(m, x), -Infinity)
    }
    if (lo === hi) {
      lo -= 1
      hi += 1
    }
    const out = new Array<number>(bins).fill(0)
    for (const x of vals) if (x >= lo && x <= hi) out[Math.min(bins - 1, Math.floor(((x - lo) / (hi - lo)) * bins))]++
    return done(c, C.noGradient(c.E, TT.of(out, [bins], t.dtype), [t], 'histc'), 'group')
  }],
  unique_consecutive: ['return_inverse return_counts dim', (c) => {
    const { t, E, h } = c
    const [wi, wc] = [flag(c, 'return_inverse'), flag(c, 'return_counts')]
    const dimV = c.a.dim
    const flat = !dimV || dimV.k === 'none'
    const d = flat ? 0 : C.normDim(h.toInt(dimV), t.ndim)
    const x = flat ? C.reshapeT(E, t, [-1], false) : t
    const n = x.shape[d] ?? 0
    const slices = Array.from({ length: n }, (_, i) => C.narrowT({ grad: false } as Engine, x, d, i, 1).values())
    const same = (p: number[], q: number[]) => p.every((v, k) => v === q[k])
    const keep: number[] = []
    const counts: number[] = []
    const inverse: number[] = []
    slices.forEach((row, i) => {
      if (!keep.length || !same(slices[keep[keep.length - 1]], row)) {
        keep.push(i)
        counts.push(0)
      }
      counts[counts.length - 1]++
      inverse.push(keep.length - 1)
    })
    const out = C.takeT(E, x, C.crossSrc(x.shape, x.shape.map((m, k) => (k === d ? keep : Array.from({ length: m }, (_, i) => i)))), x.shape.map((m, k) => (k === d ? keep.length : m)), 'UniqueConsecutiveBackward0')
    const extra = [...(wi ? [wrap(TT.of(inverse, flat ? t.shape : [n], 'int64'))] : []), ...(wc ? [wrap(TT.of(counts, [counts.length], 'int64'))] : [])]
    return extra.length ? py.tuple([wrap(out), ...extra]) : wrap(out)
  }],
  msort: ['', (c) => {
    if (!c.t.ndim) throw c.h.err('IndexError', 'Dimension specified as 0 but tensor has no dimensions')
    return wrap(C.sortT(c.E, c.t, 0, false).values)
  }],
  argwhere: ['', (c) => c.h.call(c.h.getAttr(c.self, 'nonzero'), [])],
  histogram: ['bins! range weight density', (c) => histogramOf(c)],
  trapezoid: ['x * dx dim', (c) => trapezoidOf(c, false)],
  trapz: ['x * dx dim', (c) => trapezoidOf(c, false)],
  cumulative_trapezoid: ['x * dx dim', (c) => trapezoidOf(c, true)],
  cov: ['* correction fweights aweights', (c) => covOf(c)],
  corrcoef: ['', (c) => {
    const { E } = c
    const cv = asTT(covOf({ ...c, a: {} }))!
    if (!cv.ndim) return wrap(C.binary(E, '/', cv, cv))
    const dv = diagonalOf(E, cv, 0, 0, 1, 'DiagonalBackward0')
    const root = C.unary(E, 'sqrt', dv)
    const scaled = C.binary(E, '/', C.binary(E, '/', cv, C.unsqueezeT(E, root, 0)), C.unsqueezeT(E, root, 1))
    return wrap(named2(C.clamp(E, scaled, -1, 1), 'ClampBackward1'))
  }],

  dist: ['other! p', (c) => {
    const o = tArg(c, 'other')
    return done(c, named2(C.vectorNorm(c.E, C.binary(c.E, '-', c.t, o), numOr(c, 'p', 2), null, false), 'DistBackward0'), 'reduce')
  }],
  renorm: ['p! dim! maxnorm!', (c) => {
    const { t, E, h } = c
    const d = C.normDim(intArg(c, 'dim'), t.ndim)
    const p = h.num(need(c, 'p'))
    const cap = h.num(need(c, 'maxnorm'))
    if (p <= 0) throw h.err('RuntimeError', 'non-positive-norm not supported')
    const others = t.shape.map((_, k) => k).filter((k) => k !== d)
    const norms = C.vectorNorm({ grad: false } as Engine, t, p, others, true)
    const factors = TT.of(norms.values().map((n) => (n > cap ? cap / (n + 1e-7) : 1)), norms.shape, t.dtype)
    return done(c, named2(C.binary(E, '*', t, factors), 'RenormBackward0'), 'elementwise')
  }],

  greater: twoOp('>'), greater_equal: twoOp('>='), less: twoOp('<'), less_equal: twoOp('<='), not_equal: twoOp('!='),
  maximum: ['other!', (c) => wrap(C.binary(c.E, 'maximum', c.t, tArg(c, 'other')))],
  minimum: ['other!', (c) => wrap(C.binary(c.E, 'minimum', c.t, tArg(c, 'other')))],
  arctan2: ['other!', (c) => c.h.call(c.h.getAttr(c.self, 'atan2'), [need(c, 'other')])],
  conj: ['', (c) => c.self],
  isreal: ['', (c) => boolMap(c, () => true)],
  imag: ['', (c) => { throw c.h.err('RuntimeError', 'imag is not implemented for tensors with non-complex dtypes.') }],
  angle: ['', (c) => done(c, C.mapT(c.E, c.t, (x) => (x < 0 ? Math.PI : Number.isNaN(x) ? NaN : 0), () => 0, 'AngleBackward0'), 'elementwise')],
  nextafter: floatPair((x, y) => {
    if (x === y) return y
    const bits = new Float32Array([x])
    const raw = new Uint32Array(bits.buffer)
    if (x === 0) raw[0] = y > 0 ? 1 : 0x80000001
    else raw[0] += (x < y) === (x > 0) ? 1 : -1
    return bits[0]
  }, () => 0, () => 0, 'NextafterBackward0'),
  frexp: ['', (c) => {
    const parts2 = c.t.values().map((x): [number, number] => {
      if (x === 0 || !Number.isFinite(x)) return [x, 0]
      const e = Math.max(-1023, Math.floor(Math.log2(Math.abs(x))) + 1)
      return [x * 2 ** -e, e]
    })
    return py.obj(new ReturnTypes('frexp', [['mantissa', wrap(TT.of(parts2.map((p) => p[0]), c.t.shape, c.t.dtype))], ['exponent', wrap(TT.of(parts2.map((p) => p[1]), c.t.shape, 'int32'))]]))
  }],
  igamma: floatPair((a, x) => gammaIncFn(a, x, false), () => NaN, (a, x) => Math.exp(-x + (a - 1) * Math.log(x) - lgammaFn(a)), 'IgammaBackward0'),
  igammac: floatPair((a, x) => gammaIncFn(a, x, true), () => NaN, (a, x) => -Math.exp(-x + (a - 1) * Math.log(x) - lgammaFn(a)), 'IgammacBackward0'),
  addmm: ['mat1! mat2! * beta alpha', (c) => matAdd(c, C.matmul(c.E, tArg(c, 'mat1'), tArg(c, 'mat2')), 'AddmmBackward0')],
  addmv: ['mat! vec! * beta alpha', (c) => matAdd(c, C.matmul(c.E, tArg(c, 'mat'), tArg(c, 'vec')), 'AddmvBackward0')],
  addr: ['vec1! vec2! * beta alpha', (c) => matAdd(c, C.matmul(c.E, C.unsqueezeT(c.E, tArg(c, 'vec1'), -1), C.unsqueezeT(c.E, tArg(c, 'vec2'), 0)), 'AddrBackward0')],
  addbmm: ['batch1! batch2! * beta alpha', (c) => matAdd(c, C.reduceSum(c.E, C.matmul(c.E, tArg(c, 'batch1'), tArg(c, 'batch2')), [0], false, false), 'AddbmmBackward0')],
  baddbmm: ['batch1! batch2! * beta alpha', (c) => matAdd(c, C.matmul(c.E, tArg(c, 'batch1'), tArg(c, 'batch2')), 'BaddbmmBackward0')],
  split_with_sizes: ['split_sizes! dim', (c) => {
    const d = C.normDim(intArg(c, 'dim', 0), c.t.ndim)
    const sizes2 = c.h.iterate(need(c, 'split_sizes')).map((v) => c.h.toInt(v))
    if (sizes2.reduce((p, q) => p + q, 0) !== c.t.shape[d]) throw c.h.err('RuntimeError', `split_with_sizes expects split_sizes to sum exactly to ${c.t.shape[d]} (input tensor's size at dimension ${d}), but got split_sizes=[${sizes2.join(', ')}]`)
    return parts(c, d, sizes2, 'SplitWithSizesBackward0')
  }],
  put: ['index! source! accumulate', (c) => {
    const { t, E, h } = c
    const index = tArg(c, 'index')
    const src = tArg(c, 'source')
    if (index.size !== src.size && src.size !== 1) throw h.err('RuntimeError', `put_(): Expected source and index to have the same number of elements, but got source.numel() = ${src.size}, index.numel() = ${index.size}`)
    const pos = index.values().map((i) => (i < 0 ? i + t.size : i))
    const flat = C.reshapeT(E, t, [-1], false)
    const vals = C.reshapeT(E, C.expandT(E, src, index.shape), [-1], false)
    return done(c, named2(C.reshapeT(E, C.scatterT(E, flat, 0, TT.of(pos, [pos.length], 'int64'), vals, flag(c, 'accumulate')), t.shape, false), 'PutBackward0'), 'group')
  }],
  index_reduce: ['dim! index! source! reduce! * include_self', (c) => {
    const { t, h } = c
    const d = C.normDim(intArg(c, 'dim'), t.ndim)
    const src = tArg(c, 'source')
    const how = h.str(need(c, 'reduce'))
    if (!['prod', 'mean', 'amax', 'amin'].includes(how)) throw h.err('RuntimeError', 'index_reduce(): reduce argument must be either prod, mean, amax or amin, got ' + how + '.')
    return done(c, scatterReduced(c, d, expandIndexAlong(tArg(c, 'index'), d, src.shape), src, how, flag(c, 'include_self', true)), 'group')
  }],
  resize: ['*size', (c) => {
    const dims = toDims(c.h, c.a.size?.k === 'tuple' ? c.a.size.items : [])
    const old = c.t.values()
    return wrap(TT.of(Array.from({ length: prod(dims) }, (_, i) => old[i] ?? 0), dims, c.t.dtype))
  }],
  resize_as_: ['the_template!', (c) => c.h.call(c.h.getAttr(c.self, 'resize_'), [py.tuple(tArg(c, 'the_template').shape.map(py.int))])],
  new: ['*args', (c) => {
    const v = c.a.args?.k === 'tuple' ? c.a.args.items : []
    if (!v.length) return wrap(TT.of([], [0], c.t.dtype))
    if (v.length === 1 && (v[0].k === 'list' || v[0].k === 'tuple')) return wrap(tensorFrom(c.h, v[0], c.t.dtype))
    const dims = toDims(c.h, v)
    return wrap(TT.of(new Array<number>(prod(dims)).fill(0), dims, c.t.dtype))
  }],
  cauchy_: ['median sigma generator', (c) => {
    const [m, s0] = [numOr(c, 'median', 0), numOr(c, 'sigma', 1)]
    const rng = rngOf(c.h, c.a.generator)
    return filledIn(c, () => m + s0 * Math.tan(Math.PI * (rng.next() - 0.5)), 'CauchyBackward0')
  }],
  log_normal_: ['mean std generator', (c) => {
    const [m, s0] = [numOr(c, 'mean', 1), numOr(c, 'std', 2)]
    const rng = rngOf(c.h, c.a.generator)
    return filledIn(c, () => Math.exp(m + s0 * rng.normal()), 'LogNormalBackward0')
  }],
  geometric_: ['p! generator', (c) => {
    const p = c.h.num(need(c, 'p'))
    const rng = rngOf(c.h, c.a.generator)
    return filledIn(c, () => Math.ceil(Math.log(1 - rng.next()) / Math.log(1 - p)), 'GeometricBackward0')
  }],

  // ---- shape and selection, more of them
  reshape_as: ['other!', (c) => {
    const o = tArg(c, 'other')
    return wrap(viewOrReshape(c.E, c.t, o.shape, false))
  }],
  resize_: ['*size', (c) => {
    const { t, h } = c
    const dims = toDims(h, c.a.size?.k === 'tuple' ? c.a.size.items : [])
    if (t.requiresGrad) throw h.err('RuntimeError', 'cannot resize variables that require grad')
    const n = prod(dims)
    if (n > h.maxSize) throw h.err('RuntimeError', `resize_ to ${n} elements exceeds the sandbox limit`)
    // the storage outlives the shape: elements beyond the new size come back if it grows again
    const box = t as { storage?: number[] }
    const store = box.storage && box.storage.length >= t.size ? box.storage : [...t.values()]
    while (store.length < n) store.push(0)
    box.storage = store
    t.a = NDArray.create(store.slice(0, n), dims, t.a.dtype)
    t.version++
    return c.self
  }],
  tensor_split: ['indices_or_sections! dim', (c) => {
    const { t, h } = c
    const d = C.normDim(intArg(c, 'dim', 0), t.ndim)
    const n = t.shape[d]
    const v = need(c, 'indices_or_sections')
    if (v.k === 'int' || (asTT(v) && asTT(v)!.ndim === 0)) {
      const k = h.toInt(v)
      if (k <= 0) throw h.err('RuntimeError', `number of sections must be larger than 0, got ${k}`)
      const [q, r] = [Math.floor(n / k), n % k]
      return parts(c, d, Array.from({ length: k }, (_, i) => q + (i < r ? 1 : 0)), 'SliceBackward0')
    }
    const cuts = (asTT(v) ? asTT(v)!.values() : h.iterate(v).map((x) => h.toInt(x)))
    const edges = [0, ...cuts, n]
    return py.tuple(edges.slice(1).map((e, i) => {
      const [a, b] = [Math.max(0, Math.min(n, edges[i] < 0 ? edges[i] + n : edges[i])), Math.max(0, Math.min(n, e < 0 ? e + n : e))]
      return wrap(C.narrowT(c.E, c.t, d, a, Math.max(0, b - a), 'SliceBackward0'))
    }))
  }],
  hsplit: ['indices_or_sections!', (c) => splitAlong(c, 'hsplit', c.t.ndim === 1 ? 0 : 1, 1)],
  vsplit: ['indices_or_sections!', (c) => splitAlong(c, 'vsplit', 0, 2)],
  dsplit: ['indices_or_sections!', (c) => splitAlong(c, 'dsplit', 2, 3)],
  rot90: ['k dims', (c) => {
    const { t, E, h } = c
    const ds = c.a.dims ? h.iterate(c.a.dims).map((x) => h.toInt(x)) : [0, 1]
    if (ds.length !== 2) throw h.err('RuntimeError', `expected total rotation dims == 2, but got dims = ${ds.length}`)
    if (t.ndim < 2) throw h.err('RuntimeError', `expected total dims >= 2, but got total dims = ${t.ndim}`)
    const [a, b] = ds.map((x) => C.normDim(x, t.ndim))
    if (a === b) throw h.err('RuntimeError', `expected rotation dims to be different, but got dim0 = ${ds[0]} and dim1 = ${ds[1]}`)
    const k = ((intArg(c, 'k', 1) % 4) + 4) % 4
    const flipAxes = (x: TT, axes: number[]) => {
      const maps = x.shape.map((m, q) => Array.from({ length: m }, (_, i) => (axes.includes(q) ? m - 1 - i : i)))
      return C.takeT(E, x, C.crossSrc(x.shape, maps), x.shape, 'FlipBackward0')
    }
    const rotated = k === 0 ? C.cloneT(E, t) : k === 2 ? flipAxes(t, [a, b]) : k === 1 ? C.transposeT(E, flipAxes(t, [b]), a, b) : flipAxes(C.transposeT(E, t, a, b), [b])
    return wrap(named2(rotated, 'Rot90Backward0'))
  }],
  cross: ['other! dim', (c) => {
    const { t, E, h } = c
    const o = tArg(c, 'other')
    const shape = broadcastShapes([t.shape, o.shape])
    const given = optDim(h, c.a.dim)
    const pick = shape.findIndex((n) => n === 3)
    if (given === null && pick < 0) throw h.err('RuntimeError', `no dimension of size 3 in input`)
    const d = given === null ? pick : C.normDim(given, shape.length)
    if (shape[d] !== 3) throw h.err('RuntimeError', `dimension ${d} does not have size 3`)
    return wrap(named2(crossProduct(E, C.expandT(E, t, shape), C.expandT(E, o, shape), d), 'LinalgCrossBackward0'))
  }],
  kron: ['other!', (c) => {
    const { t, E } = c
    const o = tArg(c, 'other')
    const n = Math.max(t.ndim, o.ndim)
    const [sa, sb] = [[...new Array<number>(n - t.ndim).fill(1), ...t.shape], [...new Array<number>(n - o.ndim).fill(1), ...o.shape]]
    const a = C.reshapeT(E, t, sa.flatMap((m) => [m, 1]), false)
    const b = C.reshapeT(E, o, sb.flatMap((m) => [1, m]), false)
    const out = C.reshapeT(E, C.binary(E, '*', a, b), sa.map((m, k) => m * sb[k]), false)
    return done(c, named2(out, 'UnsafeViewBackward0'), 'matmul')
  }],
  vdot: ['other!', (c) => {
    const out = objOf(c.t).binop('@', need(c, 'other'), false, c.h)!
    named2(asTT(out)!, 'VdotBackward0')
    return out
  }],
  adjoint: ['', (c) => wrap(C.transposeT(c.E, c.t, -2, -1))],
  diagflat: ['offset', (c) => {
    const { t, E } = c
    const k = intArg(c, 'offset', 0)
    return diagEmbedOf(c, C.reshapeT(E, t, [-1], false), k, 0, 1)
  }],
  diag_embed: ['offset dim1 dim2', (c) => diagEmbedOf(c, c.t, intArg(c, 'offset', 0), intArg(c, 'dim1', -2), intArg(c, 'dim2', -1))],
  sum_to_size: ['*size', (c) => {
    const { t, E, h } = c
    const dims = toDims(h, c.a.size?.k === 'tuple' ? c.a.size.items : [])
    const lead = t.ndim - dims.length
    if (lead < 0 || dims.some((m, k) => m !== t.shape[lead + k] && m !== 1)) throw h.err('RuntimeError', `size {[${dims.join(', ')}]} is not expandable to size {[${t.shape.join(', ')}]}.`)
    const axes = t.shape.map((_, k) => k).filter((k) => k < lead || (dims[k - lead] === 1 && t.shape[k] !== 1))
    return wrap(named2(C.reshapeT(E, axes.length ? C.reduceSum(E, t, axes, true, false) : t, dims, false), 'SumBackward1'))
  }],
  select: ['dim! index!', (c) => {
    const { t, E, h } = c
    if (!t.ndim) throw h.err('IndexError', 'select() cannot be applied to a 0-dim tensor.')
    const d = C.normDim(intArg(c, 'dim'), t.ndim)
    const i = intArg(c, 'index')
    if (i < -t.shape[d] || i >= t.shape[d]) throw h.err('IndexError', `select(): index ${i} out of range for tensor of size [${t.shape.join(', ')}] at dimension ${d}`)
    return wrap(C.viewT(E, t, (x) => x.view(x.shape.filter((_, k) => k !== d), x.strides.filter((_, k) => k !== d), x.offset + (i < 0 ? i + t.shape[d] : i) * x.strides[d]), 'SelectBackward0'))
  }],
  unfold: ['dimension! size! step!', (c) => {
    const { t, E, h } = c
    const d = t.ndim ? C.normDim(intArg(c, 'dimension'), t.ndim) : 0
    const size = intArg(c, 'size')
    const step = intArg(c, 'step')
    const n = t.ndim ? t.shape[d] : 1
    if (size > n) throw h.err('RuntimeError', `maximum size for tensor at dimension ${d} is ${n} but size is ${size}`)
    if (step <= 0) throw h.err('RuntimeError', `step is ${step} but must be > 0`)
    const count = Math.floor((n - size) / step) + 1
    const st = t.a.strides
    return wrap(C.viewT(E, t, (x) => x.view([...x.shape.map((m, k) => (k === d ? count : m)), size], [...x.strides.map((v, k) => (k === d ? v * step : v)), st[d]], x.offset), 'UnfoldBackward0'))
  }],
  index_add: ['dim! index! source! * alpha', (c) => {
    const { t, E, h } = c
    const d = C.normDim(intArg(c, 'dim'), t.ndim)
    const index = tArg(c, 'index')
    const src = tArg(c, 'source')
    if (index.ndim > 1) throw h.err('RuntimeError', 'index_add_(): Index is supposed to be a vector')
    if (src.shape[d] !== index.size) throw h.err('RuntimeError', `index_add_(): Number of indices (${index.size}) should be equal to source.size(dim): (${src.shape[d]}), for dim: ${d}`)
    const alpha = numOr(c, 'alpha', 1)
    const expanded = expandIndexAlong(index, d, src.shape)
    const scaled = alpha === 1 ? src : C.binary(E, '*', src, TT.scalar(alpha, src.dtype), false, true)
    return done(c, named2(C.scatterT(E, t, d, expanded, scaled, true), 'IndexAddBackward0'), 'group')
  }],
  index_copy: ['dim! index! tensor!', (c) => {
    const { t, E, h } = c
    const d = C.normDim(intArg(c, 'dim'), t.ndim)
    const index = tArg(c, 'index')
    const src = tArg(c, 'tensor')
    if (index.ndim > 1) throw h.err('RuntimeError', 'index_copy_(): Index should have dimension 1 or 0 (got ' + index.ndim + ')')
    if ((src.shape[d] ?? 1) !== index.size) throw h.err('RuntimeError', `index_copy_(): Number of indices (${index.size}) should be equal to source.size(dim) (${src.shape[d]})`)
    return done(c, named2(C.scatterT(E, t, d, expandIndexAlong(index, d, src.shape), src, false), 'IndexCopyBackward0'), 'group')
  }],
  index_put: ['indices! values! accumulate', (c) => {
    const { t, E, h } = c
    const ids = h.iterate(need(c, 'indices')).map((v) => tensorArg(h, v, 'index_put', 'indices'))
    const shape = broadcastShapes(ids.map((x) => x.shape))
    const st = cStrides(t.shape)
    const offs = ids.map((x) => C.bcastIndex(x.shape, shape))
    const iv = ids.map((x) => x.values())
    const pos = Array.from({ length: prod(shape) }, (_, f) => ids.reduce((sum, _x, k) => {
      const raw = iv[k][offs[k][f]]
      return sum + (raw < 0 ? raw + t.shape[k] : raw) * st[k]
    }, 0))
    const vals = tArg(c, 'values')
    const flatV = C.reshapeT(E, C.expandT(E, vals, shape), [-1], false)
    const flatT = C.reshapeT(E, t, [-1], false)
    const out = C.scatterT(E, flatT, 0, TT.of(pos, [pos.length], 'int64'), flatV, flag(c, 'accumulate'))
    return done(c, named2(C.reshapeT(E, out, t.shape, false), 'IndexPutBackward0'), 'group')
  }],
  scatter_reduce: ['dim! index! src! reduce! * include_self', (c) => {
    const { h } = c
    const how = h.str(need(c, 'reduce'))
    if (!['sum', 'prod', 'mean', 'amax', 'amin'].includes(how)) throw h.err('RuntimeError', 'reduce argument must be either sum, prod, mean, amax or amin, got ' + how)
    const [d, index, src] = [intArg(c, 'dim'), tArg(c, 'index'), tArg(c, 'src')]
    if (how === 'sum' && flag(c, 'include_self', true) && (c.t.requiresGrad || src.requiresGrad)) return done(c, named2(C.scatterT(c.E, c.t, d, index, src, true), 'ScatterReduceBackward0'), 'group')
    return done(c, scatterReduced(c, d, index, src, how, flag(c, 'include_self', true)), 'group')
  }],
  masked_scatter: ['mask! source!', (c) => {
    const { t, E, h } = c
    const mask = tArg(c, 'mask')
    const src = tArg(c, 'source')
    if (mask.dtype !== 'bool') throw h.err('RuntimeError', `masked_scatter_ only supports boolean masks, but got mask with dtype ${C.typeLabel(mask.dtype)}`)
    if (src.dtype !== t.dtype) throw h.err('RuntimeError', `masked_scatter: expected self and source to have same dtypes but got ${C.typeLabel(t.dtype)} and ${C.typeLabel(src.dtype)}`)
    const mv = C.expandT({ grad: false } as Engine, mask, t.shape).values()
    const count = mv.reduce((n, m) => n + (m ? 1 : 0), 0)
    if (src.size < count) throw h.err('RuntimeError', `Number of elements of source < number of ones in mask`)
    let at = 0
    const pick = mv.map((m) => (m ? at++ : -1))
    const filled = C.takeT(E, src, pick.map((p) => Math.max(p, 0)), t.shape, 'MaskedScatterBackward0')
    return done(c, C.whereT(E, TT.of(mv, t.shape, 'bool'), filled, t, t.dtype, 'MaskedScatterBackward0'), 'group')
  }],
}

const INPLACE = new Set(['absolute', 'addmm', 'addmv', 'addr', 'addbmm', 'baddbmm', 'arctan2', 'atan2', 'greater', 'greater_equal', 'less', 'less_equal', 'not_equal', 'igamma', 'igammac', 'nextafter', 'put', 'renorm', 'index_reduce', 'mvlgamma', 'polygamma', 'addcmul', 'addcdiv', 'arccos', 'arcsin', 'arctan', 'arccosh', 'arcsinh', 'arctanh', 'acosh', 'asinh', 'atanh', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not', 'bitwise_left_shift', 'bitwise_right_shift', 'copysign', 'cumsum', 'cumprod', 'deg2rad', 'digamma', 'eq', 'erfinv', 'fix', 'float_power', 'ge', 'gt', 'gcd', 'heaviside', 'hypot', 'i0', 'index_add', 'index_copy', 'index_fill', 'index_put', 'lcm', 'ldexp', 'le', 'lgamma', 'logit', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'lt', 'masked_scatter', 'nan_to_num', 'ne', 'rad2deg', 'scatter_reduce', 'sgn', 'sinc', 'xlogy', 'abs', 'acos', 'add', 'asin', 'atan', 'ceil', 'clamp', 'clamp_max', 'clamp_min', 'clip', 'cos', 'cosh', 'div', 'divide', 'erf', 'erfc', 'exp', 'exp2', 'expm1', 'floor', 'floor_divide', 'fmod', 'frac', 'index_fill', 'lerp', 'log', 'log10', 'log1p', 'log2', 'logical_not', 'masked_fill', 'mul', 'multiply', 'neg', 'negative', 'pow', 'reciprocal', 'relu', 'remainder', 'round', 'rsqrt', 'scatter', 'scatter_add', 'sigmoid', 'sign', 'sin', 'sinh', 'sqrt', 'square', 'sub', 'subtract', 'tan', 'tanh', 'tril', 'triu', 'true_divide', 'trunc'])
const INPLACE_VIEW = new Set(['t', 'transpose', 'swapaxes', 'swapdims', 'squeeze', 'unsqueeze', 'swapdims', 'unflatten'])

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

function doBackward(h: Host, t: TT, self: Value, grad: number[] | null, retain: boolean, gradShape?: number[]) {
  const S = stateOf(h)
  if (gradShape && gradShape.join() !== t.shape.join()) throw h.err('RuntimeError', `Mismatch in shape: grad_output[0] has a shape of torch.Size([${gradShape.join(', ')}]) and output[0] has a shape of torch.Size([${t.shape.join(', ')}]).`)
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

/** set for each run: a user's extra_repr has to be called from where repr has no Host */
const RUN = { host: null as Host | null }

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
    extraRepr: () => {
      const h = RUN.host
      const f = h ? (() => {
        try {
          return h.getAttr(inst, 'extra_repr')
        } catch {
          return null
        }
      })() : null
      if (!h || !f || f.k !== 'bound') return ''
      return h.str(h.call(f, []))
    },
  }
}

export function namedParams(v: Value, prefix: string): [string, TT][] {
  const m = moduleOf(v)
  if (!m) return []
  const own = m.ownParams().map(([n, t]): [string, TT] => [prefix + n, t])
  return [...own, ...m.children().flatMap(([n, c]) => namedParams(c, `${prefix}${n}.`))]
}

/** what model.parameters() and friends return: one-shot iterators, as PyTorch's are generators */
const generator = (items: Value[]): Value => {
  let i = 0
  return { k: 'iter', name: 'generator', next: () => (i < items.length ? items[i++] : undefined) }
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
  // a Python subclass's own extra_repr; its lines come first, and one line with no children stays on the same line
  const own = v.k === 'inst' ? RUN.host?.getAttrOr(v, 'extra_repr') : undefined
  const extra = own && own.k === 'bound' ? RUN.host!.str(RUN.host!.call(own, [])) : m.extraRepr()
  const extraLines = extra ? extra.split('\n') : []
  if (!lines.length) return extraLines.length > 1 ? `${m.kind}(\n  ${extraLines.join('\n  ')}\n)` : `${m.kind}(${extra})`
  return `${m.kind}(\n  ${[...extraLines, ...lines].join('\n  ')}\n)`
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
      return generator(distinctParams(recurse ? namedParams(self, '') : m.ownParams()).map(([, t]) => wrap(t)))
    })
    case 'named_parameters': return fn((args, kw) => {
      const prefix = kw.prefix ?? args[0]
      const pre = prefix ? `${h.str(prefix)}.` : ''
      const recurse = flagOf(h, kw.recurse ?? args[1], true)
      return generator(distinctParams(recurse ? namedParams(self, pre) : m.ownParams().map(([n, t]): [string, TT] => [pre + n, t])).map(([n, t]) => py.tuple([py.str(n), wrap(t)])))
    })
    case 'buffers': return fn(() => generator(namedBuffers(self, '').map(([, t]) => wrap(t))))
    case 'named_buffers': return fn(() => generator(namedBuffers(self, '').map(([n, t]) => py.tuple([py.str(n), wrap(t)]))))
    case 'get_parameter': case 'get_buffer': return fn((args, kw) => {
      const path = h.str(args[0] ?? kw.target)
      const hit = (name === 'get_parameter' ? namedParams(self, '') : namedBuffers(self, '')).find(([n]) => n === path)
      if (!hit) throw h.err('AttributeError', `${m.kind} has no attribute \`${path}\``)
      return wrap(hit[1])
    })
    case 'get_submodule': return fn((args, kw) => {
      const path = h.str(args[0] ?? kw.target)
      let at = self
      if (path) for (const part of path.split('.')) {
        const c = moduleOf(at)!.children().find(([n]) => n === part)
        if (!c) throw h.err('AttributeError', `${moduleOf(at)!.kind} has no attribute \`${part}\``)
        at = c[1]
      }
      return at
    })
    case 'extra_repr': return fn(() => py.str(m.extraRepr()))
    case 'register_forward_pre_hook': case 'register_forward_hook': return fn((args, kw) => {
      const list = hooksOf(self)[name === 'register_forward_hook' ? 'post' : 'pre']
      const f = args[0] ?? kw.hook
      list.push(f)
      return handle(() => { list.splice(list.indexOf(f), 1) })
    })
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
    case 'children': return fn(() => generator(m.children().map(([, c]) => c)))
    case 'named_children': return fn(() => generator(m.children().map(([n, c]) => py.tuple([py.str(n), c]))))
    case 'modules': case 'named_modules': return fn(() => {
      const out = new Map<object, string>()
      allModules(self, '', out)
      const val = (o: object) => (o instanceof PyObj ? py.obj(o) : (o as Value))
      return generator([...out].map(([o, n]) => (name === 'modules' ? val(o) : py.tuple([py.str(n), val(o)]))))
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
    case 'float': case 'double': case 'half': return fn(() => { castFloats(self, name === 'float' ? 'float32' : name === 'half' ? 'float16' : 'float64'); return self })
    case 'type': return fn((args, kw) => {
      const d = dtypeArg(h, kw.dst_type ?? args[0])
      if (d) castFloats(self, d)
      return self
    })
    case 'share_memory': return fn(() => self)
    case 'cuda': return fn(() => { throw h.err('AssertionError', 'Torch not compiled with CUDA enabled') })
    case 'state_dict': return fn(() => ({ k: 'dict', flavor: 'OrderedDict', d: PyDict.from(stateItems(self, '').map(([n, t]) => [py.str(n), wrap(C.detach(t))])) }))
    case 'load_state_dict': return fn((args, kw) => {
      const d = args[0] ?? kw.state_dict
      if (d?.k !== 'dict') throw h.err('TypeError', `Expected state_dict to be dict-like, got <class '${d ? typeName(d) : 'NoneType'}'>.`)
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

const hooksOf = (v: Value): Hooks => (v.k === 'obj' && v.o instanceof Layer ? v.o.hooks : (v as InstValue).state[MODULE_STATE] as Hooks)

/** run a module call, tracing shapes when it is the outermost call or a layer inside one */
const callModule = (h: Host, self: Value, args0: Value[], run: (args: Value[]) => Value, leaf: Layer | null, input: TT | null): Value => {
  const S = stateOf(h)
  const hooks = hooksOf(self)
  let args = args0
  for (const f of hooks.pre) {
    const r = h.call(f, [self, py.tuple(args)])
    if (r.k !== 'none') args = r.k === 'tuple' ? r.items : [r]
  }
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
    out = run(args)
    for (const f of hooks.post) {
      const r = h.call(f, [self, py.tuple(args), out])
      if (r.k !== 'none') out = r
    }
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

const handle = (remove: () => void): Value => py.obj(new HookHandle(remove))

class HookHandle extends PyObj {
  readonly cls = 'RemovableHandle'
  remove: () => void
  constructor(remove: () => void) {
    super()
    this.remove = remove
  }
  repr() { return '<torch.utils.hooks.RemovableHandle object>' }
  getAttr(name: string): Value | undefined {
    return name === 'remove' ? { k: 'fn', name, call: () => { this.remove(); return py.NONE } } : undefined
  }
}

interface Hooks { pre: Value[]; post: Value[] }

abstract class Layer extends PyObj implements ModuleLike {
  abstract readonly kind: string
  get cls() { return this.kind }
  training = true
  extraRepr() { return '' }
  ownParams(): [string, TT][] { return [] }
  buffers = new Map<string, { t: TT; keep: boolean }>()
  ownBuffers(): [string, TT, boolean][] { return [...this.buffers].map(([n, b]) => [n, b.t, b.keep]) }
  /** the constructor's arguments, which are attributes of the module: m.p, m.dim */
  fields: Record<string, Value | undefined> = {}
  /** attributes assigned from Python */
  extras = new Map<string, Value>()
  hooks: Hooks = { pre: [], post: [] }
  /** forward's parameter names, and how many of them at the end are optional */
  fwd: string[] = ['input']
  fwdOptional = 0
  children(): [string, Value][] { return [] }
  flowNote(_i: TT, _o: TT): string | null { return null }
  abstract forward(args: Value[], kw: Kw, h: Host): Value
  repr() { return moduleRepr(py.obj(this)) }
  isa(name: string) { return name === 'Module' || name === this.kind }
  /** layer.weight = nn.Parameter(…), layer.bias = None, layer.training = False, layer.anything = 3 */
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
    this.extras.set(name, v)
    return true
  }
  call(args: Value[], kw: Kw, h: Host): Value {
    const n = this.fwd.length
    const label = `${this.kind}.forward`
    const known = Object.keys(kw).filter((k) => this.fwd.includes(k))
    if (args.length > n) throw h.err('TypeError', `${label}() takes ${this.fwdOptional ? `from ${n - this.fwdOptional + 1} to ${n + 1}` : n + 1} positional arguments but ${args.length + 1} were given`)
    const unknown = Object.keys(kw).find((k) => !this.fwd.includes(k))
    if (unknown) throw h.err('TypeError', `${label}() got an unexpected keyword argument '${unknown}'`)
    const missing = this.fwd.slice(args.length, n - this.fwdOptional).filter((p) => !known.includes(p))
    if (missing.length) throw h.err('TypeError', `${label}() missing ${missing.length} required positional argument${missing.length > 1 ? 's' : ''}: ${missing.map((p) => `'${p}'`).join(' and ')}`)
    const full = [...args]
    for (let i = args.length; i < n; i++) if (kw[this.fwd[i]]) full[i] = kw[this.fwd[i]]
    return callModule(h, py.obj(this), full, (a) => this.forward(a, {}, h), this, asTT(full[0] ?? py.NONE))
  }
  getAttr(name: string, h: Host): Value | undefined {
    const p = this.ownParams().find(([n]) => n === name)
    if (p) return wrap(p[1])
    const b = this.buffers.get(name)
    if (b) return wrap(b.t)
    if (name === 'forward') return { k: 'fn', name, call: (args, kw) => this.forward(args, kw, h) }
    if (name === '__class__') return { k: 'type', name: this.kind, call: () => { throw h.err('TypeError', `${this.kind}() cannot be called from its __class__ in this sandbox`) } }
    return this.attr(name, h) ?? this.extras.get(name) ?? this.fields[name] ?? moduleMethod(h, py.obj(this), name)
  }
  attr(_name: string, _h: Host): Value | undefined { return undefined }
}

/** what torch's argument parser says when something else is passed for a tensor */
/** what PyTorch says about the functions it has removed from torch.* in favour of torch.linalg */
const REMOVED: Record<string, string> = {"cholesky": "This function was deprecated since version 1.9 and is now removed. `torch.cholesky` is deprecated in favor of `torch.linalg.cholesky`.\nL = torch.cholesky(A) should be replaced with:\nL = torch.linalg.cholesky(A)\nU = torch.cholesky(A, upper=True) should be replaced with:\nU = torch.linalg.cholesky(A).mH", "solve": "This function was deprecated since version 1.9 and is now removed. `torch.solve` is deprecated in favor of `torch.linalg.solve`. `torch.linalg.solve` has its arguments reversed and does not return the LU factorization.\n\nTo get the LU factorization see `torch.lu`, which can be used with `torch.lu_solve` or `torch.lu_unpack`.\nX = torch.solve(B, A).solution should be replaced with:\nX = torch.linalg.solve(A, B)", "lstsq": "This function was deprecated since version 1.9 and is now removed. `torch.lstsq` is deprecated in favor of `torch.linalg.lstsq`.\n`torch.linalg.lstsq` has reversed arguments and does not return the QR decomposition in the returned tuple (although it returns other information about the problem).\n\nTo get the QR decomposition consider using `torch.linalg.qr`.\n\nThe returned solution in `torch.lstsq` stored the residuals of the solution in the last m - n columns of the returned value whenever m > n. In torch.linalg.lstsq, the residuals are in the field 'residuals' of the returned named tuple.\n\nThe unpacking of the solution, as in\nX, _ = torch.lstsq(B, A).solution[:A.size(1)]\nshould be replaced with:\nX = torch.linalg.lstsq(A, B).solution", "eig": "This function was deprecated since version 1.9 and is now removed. `torch.linalg.eig` returns complex tensors of dtype `cfloat` or `cdouble` rather than real tensors mimicking complex tensors.\n\nL, _ = torch.eig(A) should be replaced with:\nL_complex = torch.linalg.eigvals(A)\n\nand\n\nL, V = torch.eig(A, eigenvectors=True) should be replaced with:\nL_complex, V_complex = torch.linalg.eig(A)", "symeig": "This function was deprecated since version 1.9 and is now removed. The default behavior has changed from using the upper triangular portion of the matrix by default to using the lower triangular portion.\n\nL, _ = torch.symeig(A, upper=upper) should be replaced with:\nL = torch.linalg.eigvalsh(A, UPLO='U' if upper else 'L')\n\nand\n\nL, V = torch.symeig(A, eigenvectors=True) should be replaced with:\nL, V = torch.linalg.eigh(A, UPLO='U' if upper else 'L')", "matrix_rank": "This function was deprecated since version 1.9 and is now removed.\nPlease use the `torch.linalg.matrix_rank` function instead. The parameter 'symmetric' was renamed in `torch.linalg.matrix_rank()` to 'hermitian'.", "qr": "This function was deprecated since version 1.9 and is now removed. `torch.qr` is deprecated in favor of `torch.linalg.qr` and the boolean parameter `some` has been replaced with a string parameter `mode`.\n\nQ, R = torch.qr(A) should be replaced with:\nQ, R = torch.linalg.qr(A, mode='reduced')\n\nand\n\nQ, R = torch.qr(A, some=False) should be replaced with:\nQ, R = torch.linalg.qr(A, mode='complete')\n\nSince `some` was set to True, use mode='reduced'."}

/** PyTorch's type names in its argument errors */
const argType = (v: Value): string => {
  if (asTT(v)) return 'Tensor'
  if (v.k === 'obj' && v.o instanceof DtypeObj) return 'torch.dtype'
  return typeName(v)
}
/** the head of the message when no overload fits; the list of overloads is left to the documentation */
const invalidArgs = (h: Host, name: string, args: Value[], kw: Kw): PyError => {
  const pos = args.map(argType)
  const named = Object.keys(kw).map((k) => `${k}=${argType(kw[k])}`)
  const got = !pos.length && named.length ? `${named.join(', ')}, ` : [...pos, ...named].join(', ')
  return h.err('TypeError', `${name}() received an invalid combination of arguments - got (${got}), but expected one of:\n * (see the PyTorch documentation for its signatures)`)
}

const tensorArg = (h: Host, v: Value | undefined, what: string, arg = 'input', pos = 1): TT => {
  const t = v ? asTT(v) : null
  if (!t) throw h.err('TypeError', `${what}(): argument '${arg}' (position ${pos}) must be Tensor, not ${v ? typeName(v) : 'NoneType'}`)
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
    const x = tensorArg(h, args[0], 'linear')
    if (x.dtype !== this.w.dtype) throw h.err('RuntimeError', `mat1 and mat2 must have the same dtype, but got ${C.typeLabel(x.dtype)} and ${C.typeLabel(this.w.dtype)}`)
    const out = C.linear(stateOf(h).E, x, this.w, this.b)
    if (out.fn && x.ndim === 1) out.fn.name = this.b ? 'ViewBackward0' : 'SqueezeBackward4'
    return wrap(out)
  }
}

interface ConvOpts {
  stride: number[]
  padding: number[] | 'same' | 'valid'
  dilation: number[]
  groups: number
  paddingMode: C.PadMode
}

const CONV_TYPE: Record<TDType, string> = { float16: 'c10::Half', float32: 'float', float64: 'double', int64: 'long long', int32: 'int', int16: 'short', int8: 'signed char', uint8: 'unsigned char', bool: 'bool' }

const inputDtypeCheck = (h: Host, x: TT, w: TT, b: TT | null) => {
  if (x.dtype === w.dtype) return
  if (b) throw h.err('RuntimeError', `Input type (${CONV_TYPE[x.dtype]}) and bias type (${CONV_TYPE[w.dtype]}) should be the same`)
  throw h.err('RuntimeError', `expected scalar type ${C.typeLabel(x.dtype)} but found ${C.typeLabel(w.dtype)}`)
}

/**
 * conv1d runs as a conv2d of height 1 and conv3d on the general kernel; dilation spreads the kernel out with zeros; each
 * group runs on its own channels and the results are joined. All of it stays differentiable because every step is.
 */
const convolve = (h: Host, x0: TT, w0: TT, b: TT | null, o: ConvOpts, n: number): TT => {
  const E = stateOf(h).E
  if (x0.ndim !== n + 1 && x0.ndim !== n + 2) throw h.err('RuntimeError', `Expected ${n + 1}D (unbatched) or ${n + 2}D (batched) input to conv${n}d, but got input of size: [${x0.shape.join(', ')}]`)
  inputDtypeCheck(h, x0, w0, b)
  const chanDim = x0.ndim - n - 1
  if (x0.shape[chanDim] !== w0.shape[1] * o.groups) throw h.err('RuntimeError', `Given groups=${o.groups}, weight of size [${w0.shape.join(', ')}], expected input[${x0.shape.join(', ')}] to have ${w0.shape[1] * o.groups} channels, but got ${x0.shape[chanDim]} channels instead`)
  const K = w0.shape.slice(2)
  const eff = K.map((k, a) => o.dilation[a] * (k - 1) + 1)
  // padding='same' can need one more cell on the far side, and the other padding modes pad the input first
  const sym = o.padding === 'valid' ? K.map(() => 0) : o.padding === 'same' ? eff.map((e) => (e - 1) >> 1) : o.padding
  const far = o.padding === 'same' ? eff.map((e, a) => e - 1 - sym[a]) : sym
  let x = x0
  let p = sym
  if (o.paddingMode !== 'constant' || far.some((f, a) => f !== sym[a])) {
    x = C.padT(E, x, [...sym].reverse().flatMap((l, r) => [l, [...far].reverse()[r]]), o.paddingMode, 0)
    p = sym.map(() => 0)
  }
  const spatial = x.shape.slice(-n)
  if (spatial.some((sz, a) => sz + 2 * p[a] < eff[a])) {
    const big = (xs: number[]) => xs.join(' x ')
    throw h.err('RuntimeError', `Calculated padded input size per channel: (${big(spatial.map((sz, a) => sz + 2 * p[a]))}). Kernel size: (${big(eff)}). Kernel size can't be greater than actual input size`)
  }
  const run = (xg: TT, wg: TT, bg: TT | null): TT => {
    if (n === 3) return C.convNd(E, xg, wg, bg, o.stride, p, o.dilation, [0, 0, 0], false)
    let w = n === 1 ? C.unsqueezeT(E, wg, -2) : wg
    const dil = n === 1 ? [1, o.dilation[0]] : o.dilation
    if (dil[0] !== 1 || dil[1] !== 1) {
      const spread = (k: number, d: number) => Array.from({ length: d * (k - 1) + 1 }, (_, i) => (i % d ? -1 : i / d))
      const [kh, kw] = w.shape.slice(-2)
      const maps = [[...Array(w.shape[0]).keys()], [...Array(w.shape[1]).keys()], spread(kh, dil[0]), spread(kw, dil[1])]
      w = C.takeT(E, w, C.crossSrc(w.shape, maps), maps.map((m) => m.length), 'DilatedKernelBackward0')
    }
    const out = C.conv2d(E, n === 1 ? C.unsqueezeT(E, xg, -2) : xg, w, bg, n === 1 ? { stride: [1, o.stride[0]], padding: [0, p[0]] } : { stride: [o.stride[0], o.stride[1]], padding: [p[0], p[1]] })
    return n === 1 ? C.squeezeT(E, out, -2) : out
  }
  const cg = w0.shape[1]
  const og = w0.shape[0] / o.groups
  const part = (g: number) => run(C.narrowT(E, x, chanDim, g * cg, cg), C.narrowT(E, w0, 0, g * og, og), b && C.narrowT(E, b, 0, g * og, og))
  const out = o.groups === 1 ? run(x, w0, b) : C.catT(E, Array.from({ length: o.groups }, (_, g) => part(g)), chanDim)
  if (out.fn) out.fn.name = chanDim === 0 ? 'SqueezeBackward1' : 'ConvolutionBackward0'
  return out
}

/** transposed: the weight is (in, out / groups, *k) and each input cell spreads over a kernel-sized patch */
const convolveT = (h: Host, x: TT, w: TT, b: TT | null, s: number[], p: number[], d: number[], op: number[], groups: number, n: number): TT => {
  const E = stateOf(h).E
  if (x.ndim !== n + 1 && x.ndim !== n + 2) throw h.err('RuntimeError', `Expected ${n + 1}D (unbatched) or ${n + 2}D (batched) input to conv_transpose${n}d, but got input of size: [${x.shape.join(', ')}]`)
  inputDtypeCheck(h, x, w, b)
  const chanDim = x.ndim - n - 1
  if (x.shape[chanDim] !== w.shape[0]) throw h.err('RuntimeError', `Given transposed=1, weight of size [${w.shape.join(', ')}], expected input[${x.shape.join(', ')}] to have ${w.shape[0]} channels, but got ${x.shape[chanDim]} channels instead`)
  const out = groups === 1 ? C.convNd(E, x, w, b, s, p, d, op, true) : C.catT(E, Array.from({ length: groups }, (_, g) => {
    const cg = w.shape[0] / groups
    const og = w.shape[1]
    return C.convNd(E, C.narrowT(E, x, chanDim, g * cg, cg), C.narrowT(E, w, 0, g * cg, cg), b && C.narrowT(E, b, 0, g * og, og), s, p, d, op, true)
  }), chanDim)
  if (out.fn) out.fn.name = chanDim === 0 ? 'SqueezeBackward1' : 'ConvolutionBackward0'
  return out
}

const dimsStr = (p: number[]) => `(${p.join(', ')}${p.length === 1 ? ',' : ''})`

/** nn.Conv1d … Conv3d and nn.ConvTranspose1d … 3d */
class ConvLayer extends Layer {
  readonly kind: string
  n: number
  transposed: boolean
  inC: number
  outC: number
  k: number[]
  o: ConvOpts
  outPad: number[]
  w: TT
  b: TT | null
  constructor(h: Host, kind: string, n: number, transposed: boolean, inC: number, outC: number, k: number[], o: ConvOpts, outPad: number[], bias: boolean) {
    super()
    this.kind = kind
    this.n = n
    this.transposed = transposed
    this.inC = inC
    this.outC = outC
    this.k = k
    this.o = o
    this.outPad = outPad
    const lead = transposed ? [inC, outC / o.groups] : [outC, inC / o.groups]
    const fan = lead[1] * prod(k)
    this.w = initParam(h, [...lead, ...k], fan)
    this.b = bias ? initParam(h, [outC], fan) : null
    if (transposed) this.fwd = ['input', 'output_size']
    if (transposed) this.fwdOptional = 1
  }
  extraRepr() {
    const { padding, dilation, groups, stride, paddingMode } = this.o
    const on = (p: number[], v: number) => p.some((x) => x !== v)
    const pad = typeof padding === 'string' ? `, padding=${padding}` : on(padding, 0) ? `, padding=${dimsStr(padding)}` : ''
    const dil = on(dilation, 1) ? `, dilation=${dimsStr(dilation)}` : ''
    const op = on(this.outPad, 0) ? `, output_padding=${dimsStr(this.outPad)}` : ''
    const mode = paddingMode === 'constant' ? '' : `, padding_mode=${paddingMode}`
    return `${this.inC}, ${this.outC}, kernel_size=${dimsStr(this.k)}, stride=${dimsStr(stride)}${pad}${dil}${op}${groups !== 1 ? `, groups=${groups}` : ''}${this.b ? '' : ', bias=False'}${mode}`
  }
  ownParams(): [string, TT][] { return this.b ? [['weight', this.w], ['bias', this.b]] : [['weight', this.w]] }
  attr(name: string) {
    const t = (p: number[]) => py.tuple(p.map(py.int))
    switch (name) {
      case 'in_channels': return py.int(this.inC)
      case 'out_channels': return py.int(this.outC)
      case 'kernel_size': return t(this.k)
      case 'stride': return t(this.o.stride)
      case 'padding': return typeof this.o.padding === 'string' ? py.str(this.o.padding) : t(this.o.padding)
      case 'dilation': return t(this.o.dilation)
      case 'groups': return py.int(this.o.groups)
      case 'padding_mode': return py.str(this.o.paddingMode === 'constant' ? 'zeros' : this.o.paddingMode)
      case 'transposed': return py.bool(this.transposed)
      case 'output_padding': return t(this.outPad)
      case 'bias': return py.NONE
    }
    return undefined
  }
  flowNote(i: TT, o: TT) {
    const n = i.shape[i.ndim - this.n]
    const out = o.shape[o.ndim - this.n]
    const pv = typeof this.o.padding === 'string' ? (this.o.padding === 'same' ? (this.k[0] - 1) >> 1 : 0) : this.o.padding[0]
    if (this.transposed) return `(${n} − 1)·${this.o.stride[0]} − 2·${pv} + ${this.k[0]} + ${this.outPad[0]} = ${out}`
    const d = this.o.dilation[0]
    const span = d === 1 ? `${this.k[0]}` : `${d}·(${this.k[0]} − 1) − 1`
    return `⌊(${n} + 2·${pv} − ${span}) / ${this.o.stride[0]}⌋ + 1 = ${out} · params (${[...this.k, this.inC / this.o.groups].join('·')} + ${this.b ? 1 : 0})·${this.outC}`
  }
  forward(args: Value[], _kw: Kw, h: Host) {
    const x = tensorArg(h, args[0], `conv${this.transposed ? '_transpose' : ''}${this.n}d`)
    if (!this.transposed) return wrap(convolve(h, x, this.w, this.b, this.o, this.n))
    const padding = this.o.padding as number[]
    let op = this.outPad
    const size = args[1]
    if (size && size.k !== 'none') {
      const want = h.iterate(size).map((v) => h.toInt(v)).slice(-this.n)
      const S = x.shape.slice(-this.n)
      const least = S.map((sz, a) => (sz - 1) * this.o.stride[a] - 2 * padding[a] + this.o.dilation[a] * (this.k[a] - 1) + 1)
      const most = least.map((m, a) => m + this.o.stride[a] - 1)
      if (want.some((v, a) => v < least[a] || v > most[a])) throw h.err('ValueError', `requested an output size of [${want.join(', ')}], but valid sizes range from [${least.join(', ')}] to [${most.join(', ')}] (for an input of torch.Size([${S.join(', ')}]))`)
      op = want.map((v, a) => v - least[a])
    }
    return wrap(convolveT(h, x, this.w, this.b, this.o.stride, padding, this.o.dilation, op, this.o.groups, this.n))
  }
}

const POOL_DIMS: Record<string, string> = { 1: '2D or 3D', 2: '3D or 4D', 3: '4D or 5D' }

/** max or average pooling on 1 to 3 spatial dimensions, with torch's checks */
const runPool = (h: Host, x: TT, mode: 'max' | 'avg', n: number, o: C.PoolOpts, withIndices: boolean): Value => {
  const E = stateOf(h).E
  if (x.ndim !== n + 1 && x.ndim !== n + 2) {
    if (n === 1 && mode === 'max') throw h.err('RuntimeError', `max_pool1d: Expected 2D or 3D (batch mode) tensor with optional 0 dim batch size for input, but got:[${x.shape.join(', ')}]`)
    throw h.err('RuntimeError', `non-empty ${POOL_DIMS[n]} (batch mode) tensor expected for input`)
  }
  o.p.forEach((pad, a) => {
    if (pad > (o.d[a] * (o.k[a] - 1) + 1) / 2) throw h.err('RuntimeError', `pad should be at most half of effective kernel size, but got pad=${pad}, kernel_size=${o.k[a]} and dilation=${o.d[a]}`)
  })
  const r = C.poolNd(E, x, mode, o)
  if (r.out.fn) r.out.fn.name = mode === 'avg' ? `AvgPool${n}DBackward0` : withIndices ? `MaxPool${n}DWithIndicesBackward0` : n === 1 ? 'MaxPool1DBackward0' : `MaxPool${n}DWithIndicesBackward0`
  return withIndices ? py.tuple([wrap(r.out), wrap(r.indices)]) : wrap(r.out)
}

/** nn.MaxPool1d … 3d, nn.AvgPool1d … 3d */
class PoolLayer extends Layer {
  readonly kind: string
  mode: 'max' | 'avg'
  n: number
  o: C.PoolOpts
  withIndices: boolean
  /** kernel_size, stride and padding as the caller wrote them: repr keeps 2 and (2, 2) apart */
  shown: string
  constructor(kind: string, mode: 'max' | 'avg', n: number, o: C.PoolOpts, withIndices: boolean, shown: string) {
    super()
    this.kind = kind
    this.mode = mode
    this.n = n
    this.o = o
    this.withIndices = withIndices
    this.shown = shown
  }
  extraRepr() { return this.shown }
  attr(name: string) { return name === 'stride' && this.fields.stride?.k === 'none' ? this.fields.kernel_size : undefined }
  flowNote(i: TT, o: TT) { return `⌊(${i.shape[i.ndim - this.n]} + 2·${this.o.p[0]} − ${this.o.k[0]}) / ${this.o.s[0]}⌋ + 1 = ${o.shape[o.ndim - this.n]}` }
  forward(args: Value[], _kw: Kw, h: Host) { return runPool(h, tensorArg(h, args[0], `${this.mode}_pool${this.n}d`), this.mode, this.n, this.o, this.withIndices) }
}

/** nn.MaxUnpool1d, 2d: puts the values back where max pooling's indices say, zeros elsewhere */
const maxUnpool = (h: Host, x: TT, idx: TT, n: number, k: number[], s: number[], p: number[], size: Value | undefined): TT => {
  const E = stateOf(h).E
  if (idx.shape.join() !== x.shape.join()) throw h.err('RuntimeError', `Expected shape of indices to be same as that of the input tensor (${x.shape.join(', ')}) but got indices tensor with shape [${idx.shape.join(', ')}]`)
  const spatial = x.shape.slice(-n)
  const outS = size && size.k !== 'none' ? h.iterate(size).map((v) => h.toInt(v)).slice(-n) : spatial.map((sz, a) => (sz - 1) * s[a] + k[a] - 2 * p[a])
  const lead = x.shape.slice(0, -n)
  const area = prod(outS)
  const iv = idx.values()
  const per = prod(spatial)
  const shape = [...lead, ...outS]
  const zeros = TT.of(new Array<number>(prod(shape)).fill(0), shape, x.dtype)
  const flat = TT.of(iv.map((v, i) => Math.floor(i / per) * area + v), [iv.length], 'int64')
  const out = C.scatterT(E, C.reshapeT({ grad: false } as Engine, zeros, [-1], false), 0, flat, C.reshapeT(E, x, [-1], false), false)
  const res = C.reshapeT(E, out, shape, false)
  if (res.fn) res.fn.name = `MaxUnpool${n}DBackward0`
  return res
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
    const x = tensorArg(h, args[0], this.kind.toLowerCase())
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
    const drop = (x: TT) => (this.kind === 'Dropout' ? C.dropout : C.dropoutChannels)(S.E, x, this.p, this.training)
    const x = tensorArg(h, args[0], this.kind.toLowerCase())
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
  delItem(idx: Value, h: Host) {
    this.items.splice(this.at(idx, h), 1)
    // positions close up, as nn.Sequential renumbers them
    this.names = this.items.map((_, k) => String(k))
  }
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
  constructor(kind: string, ndims: number[], num: number, eps: number, momentum: number | null, affine: boolean, track: boolean, bias = true) {
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
    this.b = affine && bias ? param(0) : null
    if (track) {
      this.buffers.set('running_mean', { t: TT.of(new Array<number>(num).fill(0), [num], 'float32'), keep: true })
      this.buffers.set('running_var', { t: TT.of(new Array<number>(num).fill(1), [num], 'float32'), keep: true })
      this.buffers.set('num_batches_tracked', { t: TT.scalar(0, 'int64'), keep: true })
    }
  }
  extraRepr() { return `${this.num}, eps=${repr(py.float(this.eps))}, momentum=${this.momentum === null ? 'None' : repr(py.float(this.momentum))}, affine=${this.affine ? 'True' : 'False'}, bias=${this.b ? 'True' : 'False'}, track_running_stats=${this.track ? 'True' : 'False'}` }
  ownParams(): [string, TT][] { return [...(this.w ? [['weight', this.w] as [string, TT]] : []), ...(this.b ? [['bias', this.b] as [string, TT]] : [])] }
  forward(args: Value[], _kw: Kw, h: Host) {
    const x = tensorArg(h, args[0], this.kind)
    const E = stateOf(h).E
    if (!this.ndims.includes(x.ndim)) throw h.err('ValueError', `expected ${this.ndims.map((n) => `${n}D`).join(' or ')} input (got ${x.ndim}D input)`)
    const running = this.buffers.get('running_mean')?.t
    const runVar = this.buffers.get('running_var')?.t
    const count = this.buffers.get('num_batches_tracked')?.t
    if (running && x.shape[1] !== this.num) throw h.err('RuntimeError', `running_mean should contain ${x.shape[1]} elements not ${this.num}`)
    if (this.w && x.shape[1] !== this.num) throw h.err('RuntimeError', `weight should contain ${x.shape[1]} elements not ${this.num}`)
    if (!isFloatT(x.dtype)) throw h.err('NotImplementedError', `"batch_norm" not implemented for '${C.typeLabel(x.dtype)}'`)
    if (this.w && x.dtype !== this.w.dtype) throw h.err('RuntimeError', 'mixed dtype (CPU): all inputs must share same datatype.')
    if (this.training && x.size / x.shape[1] === 1) throw h.err('ValueError', `Expected more than 1 value per channel when training, got input size torch.Size([${x.shape.join(', ')}])`)
    const fromBatch = this.training || !running || !runVar
    const r = C.batchNormT(E, x, this.w, this.b, fromBatch ? null : { mean: running.values(), vari: runVar.values() }, this.eps)
    if (this.training && running && runVar && count) {
      const seen = count.values()[0] + 1
      C.writeInto(count, [seen])
      const m = this.momentum ?? 1 / seen
      const mv = running.values()
      const vv = runVar.values()
      // the running variance uses the unbiased estimate, though the batch itself is normalised by the biased one
      const n = x.size / x.shape[1]
      C.writeInto(running, mv.map((v, c) => (1 - m) * v + m * r.mean[c]))
      C.writeInto(runVar, vv.map((v, c) => (1 - m) * v + m * r.vari[c] * (n / (n - 1))))
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
  extraRepr() { return `(${this.normShape.join(', ')}${this.normShape.length === 1 ? ',' : ''}), eps=${repr(py.float(this.eps))}, elementwise_affine=${this.affine ? 'True' : 'False'}, bias=${this.b ? 'True' : 'False'}` }
  ownParams(): [string, TT][] { return [...(this.w ? [['weight', this.w] as [string, TT]] : []), ...(this.b ? [['bias', this.b] as [string, TT]] : [])] }
  forward(args: Value[], _kw: Kw, h: Host) {
    const x = tensorArg(h, args[0], 'layer_norm')
    const E = stateOf(h).E
    const D = this.normShape.length
    if (this.w && x.dtype !== this.w.dtype) throw h.err('RuntimeError', 'mixed dtype (CPU): all inputs must share same datatype.')
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
  constructor(h: Host, groups: number, num: number, eps: number, affine: boolean, bias: boolean) {
    super()
    if (num % groups) throw h.err('ValueError', `num_channels (${num}) must be divisible by num_groups (${groups})`)
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
    this.b = affine && bias ? param(0) : null
  }
  extraRepr() { return `${this.groups}, ${this.num}, eps=${repr(py.float(this.eps))}, affine=${this.affine ? 'True' : 'False'}, bias=${this.b ? 'True' : 'False'}` }
  ownParams(): [string, TT][] { return [...(this.w ? [['weight', this.w] as [string, TT]] : []), ...(this.b ? [['bias', this.b] as [string, TT]] : [])] }
  forward(args: Value[], _kw: Kw, h: Host) {
    const x = tensorArg(h, args[0], 'group_norm')
    const E = stateOf(h).E
    if (x.ndim < 2) throw h.err('RuntimeError', `Expected at least 2 dimensions for input tensor but received ${x.ndim}`)
    const rest = x.shape.slice(2)
    if (Math.floor((x.shape[0] * x.shape[1]) / this.groups) * prod(rest) === 1) throw h.err('ValueError', `Expected more than 1 value per channel when training, got input size [${[Math.floor((x.shape[0] * x.shape[1]) / this.groups), this.groups, ...rest].join(', ')}]`)
    if (x.shape[1] % this.groups) throw h.err('RuntimeError', `Expected number of channels in input to be divisible by num_groups, but got input of shape [${x.shape.join(', ')}] and num_groups=${this.groups}`)
    // (N, G, C/G, …): each group of channels of each sample is normalised over its channels and positions
    const g = C.reshapeT(E, x, [x.shape[0], this.groups, x.shape[1] / this.groups, ...rest], false)
    const dims = Array.from({ length: rest.length + 1 }, (_, k) => k + 2)
    let out = C.reshapeT(E, normalised(E, g, dims, this.eps), x.shape, false)
    const across = [1, x.shape[1], ...rest.map(() => 1)]
    if (this.w) out = C.binary(E, '*', out, C.reshapeT(E, this.w, across, false))
    if (this.b) out = C.binary(E, '+', out, C.reshapeT(E, this.b, across, false))
    if (out.fn) out.fn.name = 'NativeGroupNormBackward0'
    return wrap(out)
  }
}

/** (x − mean) / √(var + eps) over dims, with the biased variance */
const normalised = (E: Engine, x: TT, dims: number[], eps: number): TT => {
  const mean = C.reduceSum(E, x, dims, true, true)
  const centred = C.binary(E, '-', x, mean)
  const vari = C.reduceSum(E, C.unary(E, 'square', centred), dims, true, true)
  return C.binary(E, '/', centred, C.unary(E, 'sqrt', C.binary(E, '+', vari, TT.scalar(eps, x.dtype), false, true)))
}

/** nn.InstanceNorm1d … 3d: each channel of each sample on its own */
class InstanceNorm extends Layer {
  readonly kind: string
  n: number
  num: number
  eps: number
  momentum: number
  affine: boolean
  track: boolean
  w: TT | null
  b: TT | null
  constructor(kind: string, n: number, num: number, eps: number, momentum: number, affine: boolean, track: boolean, bias: boolean) {
    super()
    this.kind = kind
    this.n = n
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
    this.b = affine && bias ? param(0) : null
  }
  extraRepr() { return `${this.num}, eps=${repr(py.float(this.eps))}, momentum=${repr(py.float(this.momentum))}, affine=${this.affine ? 'True' : 'False'}, bias=${this.b ? 'True' : 'False'}, track_running_stats=${this.track ? 'True' : 'False'}` }
  ownParams(): [string, TT][] { return [...(this.w ? [['weight', this.w] as [string, TT]] : []), ...(this.b ? [['bias', this.b] as [string, TT]] : [])] }
  forward(args: Value[], _kw: Kw, h: Host) {
    const x0 = tensorArg(h, args[0], 'instance_norm')
    const E = stateOf(h).E
    if (x0.ndim !== this.n + 1 && x0.ndim !== this.n + 2) throw h.err('ValueError', `expected ${this.n + 1}D or ${this.n + 2}D input (got ${x0.ndim}D input)`)
    const x = x0.ndim === this.n + 1 ? C.unsqueezeT(E, x0, 0) : x0
    let out = normalised(E, x, Array.from({ length: this.n }, (_, k) => k + 2), this.eps)
    const across = x.shape.map((_, k) => (k === 1 ? x.shape[1] : 1))
    if (this.w) out = C.binary(E, '*', out, C.reshapeT(E, this.w, across, false))
    if (this.b) out = C.binary(E, '+', out, C.reshapeT(E, this.b, across, false))
    return wrap(x0.ndim === this.n + 1 ? C.squeezeT(E, out, 0) : out)
  }
}

/** nn.RMSNorm: x / √(mean(x²) + eps) over the normalised dims */
class RMSNorm extends Layer {
  readonly kind = 'RMSNorm'
  normShape: number[]
  eps: number
  w: TT | null
  constructor(normShape: number[], eps: number, affine: boolean) {
    super()
    this.normShape = normShape
    this.eps = eps
    this.w = affine ? TT.of(new Array<number>(prod(normShape)).fill(1), normShape, 'float32') : null
    if (this.w) {
      this.w.requiresGrad = true
      this.w.isParam = true
    }
  }
  extraRepr() { return `${dimsStr(this.normShape)}, eps=${repr(py.float(this.eps))}, elementwise_affine=${this.w ? 'True' : 'False'}` }
  ownParams(): [string, TT][] { return this.w ? [['weight', this.w]] : [] }
  forward(args: Value[], _kw: Kw, h: Host) {
    const x = tensorArg(h, args[0], 'rms_norm')
    const E = stateOf(h).E
    const dims = this.normShape.map((_, k) => x.ndim - this.normShape.length + k)
    const rms = C.unary(E, 'sqrt', C.binary(E, '+', C.reduceSum(E, C.unary(E, 'square', x), dims, true, true), TT.scalar(this.eps, x.dtype), false, true))
    const out = C.binary(E, '/', x, rms)
    return wrap(this.w ? C.binary(E, '*', out, this.w) : out)
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
  maxNorm: number | null
  normType: number
  scaleFreq: boolean
  sparse: boolean
  w: TT
  constructor(h: Host, rows: number, dim: number, paddingIdx: number | null, maxNorm: number | null, normType: number, scaleFreq: boolean, sparse: boolean, weight: TT | null) {
    super()
    this.rows = rows
    this.dim = dim
    if (paddingIdx !== null && (paddingIdx >= rows || paddingIdx < -rows)) throw h.err('AssertionError', paddingIdx >= rows ? 'Padding_idx must be within num_embeddings' : 'Padding_idx must be within num_embeddings')
    this.paddingIdx = paddingIdx === null ? null : paddingIdx < 0 ? paddingIdx + rows : paddingIdx
    this.maxNorm = maxNorm
    this.normType = normType
    this.scaleFreq = scaleFreq
    this.sparse = sparse
    this.w = weight ?? TT.of(Array.from({ length: rows * dim }, (_, f) => (Math.floor(f / dim) === this.paddingIdx ? 0 : h.rng.normal())), [rows, dim], 'float32')
    this.w.requiresGrad = weight ? weight.requiresGrad : true
    this.w.isParam = true
  }
  extraRepr() {
    const parts = [`${this.rows}, ${this.dim}`]
    if (this.paddingIdx !== null) parts.push(`padding_idx=${this.paddingIdx}`)
    if (this.maxNorm !== null) parts.push(`max_norm=${repr(py.float(this.maxNorm))}`)
    if (this.normType !== 2) parts.push(`norm_type=${repr(py.float(this.normType))}`)
    if (this.scaleFreq) parts.push('scale_grad_by_freq=True')
    if (this.sparse) parts.push('sparse=True')
    return parts.join(', ')
  }
  ownParams(): [string, TT][] { return [['weight', this.w]] }
  attr(name: string) {
    if (name === 'num_embeddings') return py.int(this.rows)
    if (name === 'embedding_dim') return py.int(this.dim)
    if (name === 'padding_idx') return this.paddingIdx === null ? py.NONE : py.int(this.paddingIdx)
    return undefined
  }
  flowNote() { return `${this.rows} × ${this.dim} = ${this.rows * this.dim} params` }
  forward(args: Value[], _kw: Kw, h: Host) { return wrap(embed(h, this.w, tensorArg(h, args[0], 'embedding'), this.paddingIdx, this.maxNorm, this.normType, this.scaleFreq)) }
}

/** embedding lookup; max_norm renormalises the rows it touches, in the weight itself, as PyTorch does */
const embed = (h: Host, w: TT, idx: TT, paddingIdx: number | null, maxNorm: number | null, normType: number, scaleFreq: boolean): TT => {
  const [rows, D] = w.shape
  if (maxNorm !== null && (isFloatT(idx.dtype) || idx.dtype === 'bool')) throw h.err('RuntimeError', `Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.${C.typeLabel(idx.dtype)}Tensor instead (while checking arguments for embedding)`)
  if (maxNorm !== null) {
    const wv = w.values()
    for (const r of new Set(idx.values())) {
      if (r < 0 || r >= rows) continue
      const row = wv.slice(r * D, (r + 1) * D)
      const norm = row.reduce((s, v) => s + Math.abs(v) ** normType, 0) ** (1 / normType)
      if (norm > maxNorm) row.forEach((v, j) => (wv[r * D + j] = Math.fround(v * (maxNorm / (norm + 1e-7)))))
    }
    C.writeInto(w, wv)
  }
  return C.embeddingT(stateOf(h).E, w, idx, paddingIdx, scaleFreq)
}

/** nn.EmbeddingBag: the rows of each bag, summed, averaged or maxed */
class EmbeddingBag extends Layer {
  readonly kind = 'EmbeddingBag'
  rows: number
  dim: number
  mode: string
  paddingIdx: number | null
  w: TT
  constructor(h: Host, rows: number, dim: number, mode: string, paddingIdx: number | null) {
    super()
    this.rows = rows
    this.dim = dim
    this.mode = mode
    this.paddingIdx = paddingIdx
    this.w = TT.of(Array.from({ length: rows * dim }, () => h.rng.normal()), [rows, dim], 'float32')
    this.w.requiresGrad = true
    this.w.isParam = true
    this.fwd = ['input', 'offsets', 'per_sample_weights']
    this.fwdOptional = 2
  }
  extraRepr() { return `${this.rows}, ${this.dim}, mode='${this.mode}'${this.paddingIdx === null ? '' : `, padding_idx=${this.paddingIdx}`}` }
  ownParams(): [string, TT][] { return [['weight', this.w]] }
  forward(args: Value[], _kw: Kw, h: Host) {
    const E = stateOf(h).E
    const idx = tensorArg(h, args[0], 'embedding_bag')
    const offsets = asTT(args[1] ?? py.NONE)
    const psw = asTT(args[2] ?? py.NONE)
    if (idx.ndim === 2 && offsets) throw h.err('ValueError', 'offsets has to be None if input is 2D')
    if (idx.ndim === 1 && !offsets) throw h.err('ValueError', 'offsets has to be a 1D Tensor but got None')
    const flat = C.embeddingT(E, this.w, C.reshapeT(E, idx, [-1], false), this.paddingIdx, false)
    const n = idx.size
    const starts = idx.ndim === 2 ? Array.from({ length: idx.shape[0] }, (_, i) => i * idx.shape[1]) : offsets!.values()
    const weighted = psw ? C.binary(E, '*', flat, C.reshapeT(E, psw, [-1, 1], false)) : flat
    const bags = starts.map((st, i) => {
      const end = i + 1 < starts.length ? starts[i + 1] : n
      const part = C.narrowT(E, weighted, 0, st, end - st)
      if (end === st) return TT.of(new Array<number>(this.dim).fill(0), [this.dim], flat.dtype)
      if (this.mode === 'max') return C.maxAlong(E, part, 0, false).values
      return C.reduceSum(E, part, [0], false, this.mode === 'mean')
    })
    const out = C.catT(E, bags, 0, true)
    if (out.fn) out.fn.name = 'EmbeddingBagBackward0'
    return wrap(out)
  }
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
const rnnStep = (E: Engine, mode: RnnMode, relu: boolean, H: number, x: TT, hx: TT, cx: TT | null, [wih, whh, bih, bhh, whr]: (TT | null)[]): { h: TT; c: TT | null } => {
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
  const h = C.binary(E, '*', output, C.unary(E, 'tanh', c))
  return { h: whr ? C.linear(E, h, whr, null) : h, c }
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
  proj: number
  params: [string, TT][] = []
  weights: (TT | null)[][] = []
  constructor(h: Host, mode: RnnMode, relu: boolean, inSize: number, hidden: number, layers: number, bias: boolean, batchFirst: boolean, dropoutP: number, bi: boolean, proj: number) {
    super()
    if (dropoutP < 0 || dropoutP > 1) throw h.err('ValueError', 'dropout should be a number in range [0, 1] representing the probability of an element being zeroed')
    if (hidden <= 0) throw h.err('ValueError', 'hidden_size must be greater than zero')
    if (layers <= 0) throw h.err('ValueError', 'num_layers must be greater than zero')
    if (proj < 0) throw h.err('ValueError', 'proj_size should be a positive integer or zero to disable projections')
    if (proj >= hidden) throw h.err('ValueError', 'proj_size has to be smaller than hidden_size')
    this.proj = proj
    this.fwd = ['input', 'hx']
    this.fwdOptional = 1
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
        const real = proj || hidden
        const w = [rnnParam([gate, l ? real * dirs : inSize], hidden, h), rnnParam([gate, real], hidden, h), bias ? rnnParam([gate], hidden, h) : null, bias ? rnnParam([gate], hidden, h) : null, proj ? rnnParam([proj, hidden], hidden, h) : null]
        this.weights.push(w)
        this.params.push([`weight_ih_${suffix}`, w[0]!], [`weight_hh_${suffix}`, w[1]!])
        if (bias) this.params.push([`bias_ih_${suffix}`, w[2]!], [`bias_hh_${suffix}`, w[3]!])
        if (proj) this.params.push([`weight_hr_${suffix}`, w[4]!])
      }
    }
  }
  extraRepr() {
    const yes = (b: boolean, s: string) => (b ? `, ${s}` : '')
    return `${this.inSize}, ${this.hidden}${yes(this.proj !== 0, `proj_size=${this.proj}`)}${yes(this.layers !== 1, `num_layers=${this.layers}`)}${yes(!this.bias, 'bias=False')}${yes(this.batchFirst, 'batch_first=True')}${yes(this.dropoutP !== 0, `dropout=${this.dropoutP}`)}${yes(this.bi, 'bidirectional=True')}`
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
    if (input.dtype !== this.params[0][1].dtype) throw h.err('ValueError', `RNN input dtype (torch.${input.dtype}) does not match weight dtype (torch.${this.params[0][1].dtype}). Convert input: input.to(torch.${this.params[0][1].dtype}), or convert model: model.to(torch.${input.dtype})`)
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
    const H0 = this.proj || this.hidden
    const start = (t: TT | null, what: string, size: number): (i: number) => TT => {
      if (!t) return () => TT.of(new Array<number>(N * size).fill(0), [N, size], input.dtype)
      const three = unbatched ? C.unsqueezeT(E, t, 1) : t
      this.checkState(h, three, [this.layers * dirs, N, size], what)
      return (i) => C.reshapeT(E, C.narrowT(E, three, 0, i, 1, 'UnbindBackward0'), [N, size], false)
    }
    const sv = state && state.k !== 'none' ? state : null
    // an LSTM's state is (h, c); a lone tensor is indexed as if it were that pair, as in PyTorch
    const part = (i: number) => (sv ? asTT(sv.k === 'obj' && sv.o.getItem ? sv.o.getItem(py.int(i), h) : h.iterate(sv)[i]) : null)
    const dimsSeen = (t: TT | null) => {
      if (t && t.ndim !== (unbatched ? 2 : 3)) throw h.err('RuntimeError', `For ${unbatched ? 'unbatched 2-D' : 'batched 3-D'} input, hx should also be ${unbatched ? 2 : 3}-D but got ${t.ndim}-D tensor`)
      return t
    }
    const lstmH = this.mode === 'LSTM' ? part(0) : null
    const lstmC = this.mode === 'LSTM' ? part(1) : null
    const h0 = start(dimsSeen(this.mode === 'LSTM' ? lstmH : sv ? asTT(sv) : null), this.mode === 'LSTM' ? '[0]' : '', H0)
    const c0 = this.mode === 'LSTM' ? start(dimsSeen(lstmC), '[1]', this.hidden) : null
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
    if (this.bi && output.fn) output.fn.name = 'CatBackward0'
    if (unbatched) output = C.squeezeT(E, output, 1)
    else if (this.batchFirst) {
      output = C.transposeT(E, output, 0, 1)
      if (output.fn) output.fn.name = this.mode === 'LSTM' ? 'TransposeBackward0' : 'TransposeBackward1'
    }
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
    this.fwd = ['input', 'hx']
    this.fwdOptional = 1
  }
  extraRepr() { return `${this.inSize}, ${this.hidden}${this.w[2] ? '' : ', bias=False'}${this.mode === 'RNN' && this.relu ? ', nonlinearity=relu' : ''}` }
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
    if (x.shape[1] !== this.inSize) throw h.err('RuntimeError', `input has inconsistent input_size: got ${x.shape[1]} expected ${this.inSize}`)
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
    this.fwd = ['query', 'key', 'value', 'key_padding_mask', 'need_weights', 'attn_mask', 'average_attn_weights', 'is_causal']
    this.fwdOptional = 5
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
    if (output.fn) output.fn.name = 'ViewBackward0'
    const out = !batched ? C.squeezeT(E, output, 1) : this.batchFirst ? C.transposeT(E, output, 0, 1) : output
    if (a.need_weights && !h.truthy(a.need_weights)) return py.tuple([wrap(out), py.NONE])
    let w = C.reshapeT(E, weights, [bsz, this.heads, tgt, src], false)
    if (flagOf(h, a.average_attn_weights, true)) w = C.reduceSum(E, w, [1], false, true)
    return py.tuple([wrap(out), wrap(!batched ? C.squeezeT(E, w, 0) : w)])
  }
}

class LossLayer extends Layer {
  readonly kind: string
  fn: (a: TT, b: TT, h: Host, more: TT[]) => TT
  kids: [string, Value][] = []
  children() { return this.kids }
  constructor(kind: string, fn: (a: TT, b: TT, h: Host, more: TT[]) => TT, fwd = ['input', 'target']) {
    super()
    this.kind = kind
    this.fn = fn
    this.fwd = fwd
  }
  forward(args: Value[], _kw: Kw, h: Host) {
    const name = this.kind.replace(/Loss$/, '').toLowerCase()
    return wrap(this.fn(tensorArg(h, args[0], name), tensorArg(h, args[1], name, 'target', 2), h, args.slice(2).map((a, i) => tensorArg(h, a, name, this.fwd[i + 2], i + 3))))
  }
}

/** a layer whose forward returns whatever its function does (a tuple, say) */
class FnLayer extends Layer {
  readonly kind: string
  fn: (args: Value[]) => Value
  extra: string
  constructor(kind: string, fn: (args: Value[]) => Value, extra: string) {
    super()
    this.kind = kind
    this.fn = fn
    this.extra = extra
  }
  extraRepr() { return this.extra }
  forward(args: Value[]) { return this.fn(args) }
}

class BilinearLayer extends Layer {
  readonly kind = 'Bilinear'
  w: TT
  b: TT | null
  run: (x1: TT, x2: TT, w: TT, b: TT | null) => TT
  shape: number[]
  constructor(h: Host, in1: number, in2: number, out: number, bias: boolean, run: (x1: TT, x2: TT, w: TT, b: TT | null) => TT) {
    super()
    this.shape = [in1, in2, out]
    this.run = run
    const bound = 1 / Math.sqrt(in1)
    const param = (shape: number[]) => {
      const t = TT.raw(Array.from({ length: prod(shape) }, () => Math.fround((h.rng.next() * 2 - 1) * bound)), shape, 'float32')
      t.requiresGrad = true
      t.isParam = true
      return t
    }
    this.w = param([out, in1, in2])
    this.b = bias ? param([out]) : null
    this.fwd = ['input1', 'input2']
  }
  extraRepr() { return `in1_features=${this.shape[0]}, in2_features=${this.shape[1]}, out_features=${this.shape[2]}, bias=${this.b ? 'True' : 'False'}` }
  ownParams(): [string, TT][] { return this.b ? [['weight', this.w], ['bias', this.b]] : [['weight', this.w]] }
  attr(name: string) { return name === 'bias' ? py.NONE : undefined }
  forward(args: Value[], _kw: Kw, h: Host) { return wrap(this.run(tensorArg(h, args[0], 'bilinear', 'input1'), tensorArg(h, args[1], 'bilinear', 'input2', 2), this.w, this.b)) }
}

/** nn.Buffer: assigning it to a module's attribute registers it */
class BufferMark extends PyObj {
  readonly cls = 'Buffer'
  t: TT
  persistent: boolean
  constructor(t: TT, persistent: boolean) {
    super()
    this.t = t
    this.persistent = persistent
  }
  repr() { return formatTensor(this.t) }
}

const paramOf = (h: Host, v: Value): TT => {
  const t = asTT(v)
  if (!t) throw h.err('TypeError', `value must be a Parameter, not ${typeName(v)}`)
  if (t.isParam) return t
  const p = new TT(t.a, t.dtype)
  p.isParam = true
  p.requiresGrad = isFloatT(t.dtype)
  return p
}

const paramLine = (t: TT, name: string) => `Parameter containing: [${name} of size ${t.shape.join('x')}]`
/** ParameterList and ParameterDict print through extra_repr, one line of which goes inline */
const paramRepr = (kind: string, lines: string[]) => (lines.length === 1 ? `${kind}(${lines[0]})` : lines.length ? `${kind}(\n  ${lines.join('\n  ')}\n)` : `${kind}()`)

class ParameterList extends Layer {
  readonly kind = 'ParameterList'
  items: TT[]
  constructor(items: TT[]) {
    super()
    this.items = items.map((t) => {
      const p = new TT(t.a, t.dtype)
      p.isParam = true
      p.requiresGrad = isFloatT(t.dtype)
      return t.isParam ? t : p
    })
  }
  ownParams(): [string, TT][] { return this.items.map((t, i) => [String(i), t]) }
  repr() { return paramRepr('ParameterList', this.items.map((t, i) => `  (${i}): ${paramLine(t, `torch.${t.dtype}`)}`)) }
  len() { return this.items.length }
  iter() { return this.items.map(wrap) }
  getItem(idx: Value, h: Host) {
    const i = h.toInt(idx)
    const n = this.items.length
    if (i < -n || i >= n) throw h.err('IndexError', `index ${i} is out of range`)
    return wrap(this.items[i < 0 ? i + n : i])
  }
  setItem(idx: Value, v: Value, h: Host) {
    const i = h.toInt(idx)
    this.items[i < 0 ? i + this.items.length : i] = paramOf(h, v)
  }
  forward(_a: Value[], _k: Kw, h: Host): Value { throw h.err('NotImplementedError', 'Module [ParameterList] is missing the required "forward" function') }
  attr(name: string, h: Host): Value | undefined {
    const self = py.obj(this)
    if (name === 'append') return { k: 'fn', name, call: (args) => { this.items.push(paramOf(h, args[0])); return self } }
    if (name === 'extend') return { k: 'fn', name, call: (args) => { for (const v of h.iterate(args[0])) this.items.push(paramOf(h, v)); return self } }
    return undefined
  }
}

class ParameterDict extends Layer {
  readonly kind = 'ParameterDict'
  items = new Map<string, TT>()
  ownParams(): [string, TT][] { return [...this.items] }
  repr() { return paramRepr('ParameterDict', [...this.items].map(([k, t]) => `  (${k}): ${paramLine(t, `torch.${C.typeLabel(t.dtype)}Tensor`)}`)) }
  len() { return this.items.size }
  iter() { return [...this.items.keys()].map(py.str) }
  contains(v: Value) { return v.k === 'str' && this.items.has(v.v) }
  getItem(idx: Value, h: Host) {
    const t = idx.k === 'str' ? this.items.get(idx.v) : undefined
    if (!t) throw h.err('KeyError', repr(idx))
    return wrap(t)
  }
  setItem(idx: Value, v: Value, h: Host) { this.items.set(h.str(idx), paramOf(h, v)) }
  update(v: Value, h: Host) {
    const pairs = v.k === 'dict' ? v.d.items() : h.iterate(v).map((p) => h.iterate(p) as [Value, Value])
    for (const [n, m] of pairs) this.items.set(h.str(n), paramOf(h, m))
  }
  forward(_a: Value[], _k: Kw, h: Host): Value { throw h.err('NotImplementedError', 'Module [ParameterDict] is missing the required "forward" function') }
  attr(name: string, h: Host): Value | undefined {
    const fn = (call: (args: Value[]) => Value): Value => ({ k: 'fn', name, call })
    switch (name) {
      case 'keys': return fn(() => py.list([...this.items.keys()].map(py.str)))
      case 'values': return fn(() => py.list([...this.items.values()].map(wrap)))
      case 'items': return fn(() => py.list([...this.items].map(([k, t]) => py.tuple([py.str(k), wrap(t)]))))
      case 'update': return fn((args) => { this.update(args[0], h); return py.NONE })
    }
    return undefined
  }
}

/** nn.TransformerEncoderLayer: attention then feed-forward, each with dropout, a residual and a LayerNorm */
class TransformerEncoderLayer extends Layer {
  readonly kind = 'TransformerEncoderLayer'
  kids: [string, Value][]
  normFirst: boolean
  activation: Value
  constructor(h: Host, a: Record<string, Value | undefined>, make: (name: string, args: Value[], kw?: Kw) => Value) {
    super()
    const d = a.d_model!
    const ff = a.dim_feedforward!
    const drop = py.float(h.num(a.dropout!))
    const norm = () => make('LayerNorm', [d], { eps: a.layer_norm_eps!, bias: a.bias! })
    this.kids = [
      ['self_attn', make('MultiheadAttention', [d, a.nhead!], { dropout: drop, bias: a.bias!, batch_first: a.batch_first! })],
      ['linear1', make('Linear', [d, ff], { bias: a.bias! })],
      ['dropout', make('Dropout', [drop])],
      ['linear2', make('Linear', [ff, d], { bias: a.bias! })],
      ['norm1', norm()],
      ['norm2', norm()],
      ['dropout1', make('Dropout', [drop])],
      ['dropout2', make('Dropout', [drop])],
    ]
    this.normFirst = flagOf(h, a.norm_first)
    const act = a.activation!
    this.activation = act.k === 'str' ? h.getAttr(h.module('torch.nn.functional'), act.v) : act
    this.fwd = ['src', 'src_mask', 'src_key_padding_mask', 'is_causal']
    this.fwdOptional = 3
  }
  children() { return this.kids }
  attr(name: string) {
    const k = this.kids.find(([n]) => n === name)
    return k ? k[1] : name === 'activation' ? this.activation : undefined
  }
  forward(args: Value[], _kw: Kw, h: Host): Value {
    const kid = (n: string) => this.kids.find(([m]) => m === n)![1]
    const attn = (x: Value) => {
      const kw: Kw = { need_weights: py.bool(false) }
      if (args[1] && args[1].k !== 'none') kw.attn_mask = args[1]
      if (args[2] && args[2].k !== 'none') kw.key_padding_mask = args[2]
      const r = h.call(kid('self_attn'), [x, x, x], kw) as { items: Value[] }
      return h.call(kid('dropout1'), [r.items[0]])
    }
    const feed = (x: Value) => h.call(kid('dropout2'), [h.call(kid('linear2'), [h.call(kid('dropout'), [h.call(this.activation, [h.call(kid('linear1'), [x])])])])])
    const add = (a: Value, b: Value) => asTT(a) && objOf(asTT(a)!).binop('+', b, false, h)!
    let x = args[0]
    if (this.normFirst) {
      x = add(x, attn(h.call(kid('norm1'), [x])))!
      x = add(x, feed(h.call(kid('norm2'), [x])))!
    } else {
      x = h.call(kid('norm1'), [add(x, attn(x))!])
      x = h.call(kid('norm2'), [add(x, feed(x))!])
    }
    return x
  }
}

/** nn.TransformerEncoder: independent copies of one layer, one after the other */
class TransformerEncoder extends Layer {
  readonly kind = 'TransformerEncoder'
  layers: ModuleList
  norm: Value | null
  constructor(layerV: Value, count: number, norm: Value | null) {
    super()
    this.layers = new ModuleList(Array.from({ length: count }, () => snapshot(layerV, new Map()) as Value))
    this.norm = norm
    this.fwd = ['src', 'mask', 'src_key_padding_mask', 'is_causal']
    this.fwdOptional = 3
  }
  children(): [string, Value][] { return [['layers', py.obj(this.layers)], ...(this.norm ? [['norm', this.norm] as [string, Value]] : [])] }
  attr(name: string) { return name === 'layers' ? py.obj(this.layers) : name === 'norm' ? this.norm ?? py.NONE : name === 'num_layers' ? py.int(this.layers.items.length) : undefined }
  forward(args: Value[], _kw: Kw, h: Host): Value {
    let x = args[0]
    for (const m of this.layers.items) x = h.call(m, [x, args[1] ?? py.NONE, args[2] ?? py.NONE])
    return this.norm ? h.call(this.norm, [x]) : x
  }
}

/** nn.LazyLinear: becomes a Linear at the first call, when in_features is known */
class LazyLinear extends Layer {
  kind = 'LazyLinear'
  out: number
  bias: boolean
  inner: Linear | null = null
  host: Host
  constructor(h: Host, out: number, bias: boolean) {
    super()
    this.host = h
    this.out = out
    this.bias = bias
  }
  extraRepr() { return `in_features=${this.inner ? this.inner.inF : 0}, out_features=${this.out}, bias=${this.bias ? 'True' : 'False'}` }
  ownParams(): [string, TT][] { return this.inner ? this.inner.ownParams() : [] }
  attr(name: string, h: Host) { return this.inner ? this.inner.getAttr(name, h) : undefined }
  forward(args: Value[], kw: Kw, h: Host) {
    if (!this.inner) {
      const x = tensorArg(h, args[0], 'linear')
      this.inner = new Linear(h, x.shape[x.ndim - 1], this.out, this.bias)
      this.kind = 'Linear'
    }
    return this.inner.forward(args, kw, h)
  }
}

/** nn.TransformerDecoderLayer: self-attention on the target, attention to the memory, then feed-forward */
class TransformerDecoderLayer extends Layer {
  readonly kind = 'TransformerDecoderLayer'
  kids: [string, Value][]
  normFirst: boolean
  activation: Value
  constructor(h: Host, a: Record<string, Value | undefined>, make: (name: string, args: Value[], kw?: Kw) => Value) {
    super()
    const d = a.d_model!
    const ff = a.dim_feedforward!
    const drop = py.float(h.num(a.dropout!))
    const norm = () => make('LayerNorm', [d], { eps: a.layer_norm_eps!, bias: a.bias! })
    const attn = () => make('MultiheadAttention', [d, a.nhead!], { dropout: drop, bias: a.bias!, batch_first: a.batch_first! })
    this.kids = [
      ['self_attn', attn()],
      ['multihead_attn', attn()],
      ['linear1', make('Linear', [d, ff], { bias: a.bias! })],
      ['dropout', make('Dropout', [drop])],
      ['linear2', make('Linear', [ff, d], { bias: a.bias! })],
      ['norm1', norm()],
      ['norm2', norm()],
      ['norm3', norm()],
      ['dropout1', make('Dropout', [drop])],
      ['dropout2', make('Dropout', [drop])],
      ['dropout3', make('Dropout', [drop])],
    ]
    this.normFirst = flagOf(h, a.norm_first)
    const act = a.activation!
    this.activation = act.k === 'str' ? h.getAttr(h.module('torch.nn.functional'), act.v) : act
    this.fwd = ['tgt', 'memory', 'tgt_mask', 'memory_mask', 'tgt_key_padding_mask', 'memory_key_padding_mask', 'tgt_is_causal', 'memory_is_causal']
    this.fwdOptional = 6
  }
  children() { return this.kids }
  attr(name: string) {
    const k = this.kids.find(([n]) => n === name)
    return k ? k[1] : name === 'activation' ? this.activation : undefined
  }
  forward(args: Value[], _kw: Kw, h: Host): Value {
    const kid = (n: string) => this.kids.find(([m]) => m === n)![1]
    const has = (v: Value | undefined) => !!v && v.k !== 'none'
    const selfBlock = (x: Value) => {
      const kw: Kw = { need_weights: py.bool(false) }
      if (has(args[2])) kw.attn_mask = args[2]
      if (has(args[4])) kw.key_padding_mask = args[4]
      return h.call(kid('dropout1'), [(h.call(kid('self_attn'), [x, x, x], kw) as { items: Value[] }).items[0]])
    }
    const crossBlock = (x: Value) => {
      const kw: Kw = { need_weights: py.bool(false) }
      if (has(args[3])) kw.attn_mask = args[3]
      if (has(args[5])) kw.key_padding_mask = args[5]
      return h.call(kid('dropout2'), [(h.call(kid('multihead_attn'), [x, args[1], args[1]], kw) as { items: Value[] }).items[0]])
    }
    const feed = (x: Value) => h.call(kid('dropout3'), [h.call(kid('linear2'), [h.call(kid('dropout'), [h.call(this.activation, [h.call(kid('linear1'), [x])])])])])
    const add = (a: Value, b: Value) => h.binop('+', a, b)
    let x = args[0]
    if (this.normFirst) {
      x = add(x, selfBlock(h.call(kid('norm1'), [x])))
      x = add(x, crossBlock(h.call(kid('norm2'), [x])))
      x = add(x, feed(h.call(kid('norm3'), [x])))
    } else {
      x = h.call(kid('norm1'), [add(x, selfBlock(x))])
      x = h.call(kid('norm2'), [add(x, crossBlock(x))])
      x = h.call(kid('norm3'), [add(x, feed(x))])
    }
    return x
  }
}

class TransformerDecoder extends Layer {
  readonly kind = 'TransformerDecoder'
  layers: ModuleList
  norm: Value | null
  constructor(layerV: Value, count: number, norm: Value | null) {
    super()
    this.layers = new ModuleList(Array.from({ length: count }, () => snapshot(layerV, new Map()) as Value))
    this.norm = norm
    this.fwd = ['tgt', 'memory', 'tgt_mask', 'memory_mask', 'tgt_key_padding_mask', 'memory_key_padding_mask', 'tgt_is_causal', 'memory_is_causal']
    this.fwdOptional = 6
  }
  children(): [string, Value][] { return [['layers', py.obj(this.layers)], ...(this.norm ? [['norm', this.norm] as [string, Value]] : [])] }
  attr(name: string) { return name === 'layers' ? py.obj(this.layers) : name === 'norm' ? this.norm ?? py.NONE : name === 'num_layers' ? py.int(this.layers.items.length) : undefined }
  forward(args: Value[], _kw: Kw, h: Host): Value {
    let x = args[0]
    for (const m of this.layers.items) x = h.call(m, [x, ...args.slice(1)])
    return this.norm ? h.call(this.norm, [x]) : x
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
const mse = (h: Host, a: TT, b: TT, red: C.Reduction = 'mean') => C.mseLoss(stateOf(h).E, a, b, red, () => noteOnce(h, 'mse_shape', 'warn', { a: `(${a.shape.join(', ')})`, b: `(${b.shape.join(', ')})` }))

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
  if (!o.weight && o.ignore === -100 && !o.smoothing) {
    const out = ceLoss(h, x, y, o.red)
    if (probs && out.fn) out.fn.name = o.red === 'mean' ? 'DivBackward1' : 'NegBackward0'
    return out
  }
  if (!probs && !y.ndim && x.ndim !== 1) throw h.err('RuntimeError', `Expected target size to match input`)
  if (o.weight && o.weight.size !== x.shape[x.ndim - 1]) throw h.err('RuntimeError', `weight tensor should be defined either for all ${x.shape[x.ndim - 1]} classes or no classes but got weight tensor of shape: [${o.weight.shape.join(', ')}]`)
  const out = C.weightedNll(E, C.softmax(E, x, x.ndim === 1 ? 0 : 1, true), y, o.weight ? o.weight.values() : null, o.ignore, o.smoothing, o.red)
  if (out.fn) out.fn.name = probs ? (o.red === 'mean' ? 'DivBackward1' : 'NegBackward0') : o.smoothing ? 'AddBackward0' : 'NllLossBackward0'
  return out
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
  const out = C.weightedNll(E, lp, y, o.weight ? o.weight.values() : null, o.ignore, 0, o.red)
  if (out.fn) out.fn.name = 'NllLossBackward0'
  return out
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
  if (a.shape.join() !== b.shape.join()) noteOnce(h, 'mse_shape', 'warn', { a: `(${a.shape.join(', ')})`, b: `(${b.shape.join(', ')})` })
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
  Adadelta: (w, g0, st, hp) => {
    const [rho, eps, lr] = [hp('rho'), hp('eps'), hp('lr')]
    const g = decayed(g0, w, hp('weight_decay'))
    st.buf.square_avg = (st.buf.square_avg ?? zerosLike(g)).map((v, i) => rho * v + (1 - rho) * g[i] * g[i])
    const acc = st.buf.acc_delta ?? zerosLike(g)
    const delta = g.map((x, i) => (Math.sqrt(acc[i] + eps) / Math.sqrt(st.buf.square_avg[i] + eps)) * x)
    st.buf.acc_delta = acc.map((v, i) => rho * v + (1 - rho) * delta[i] * delta[i])
    return w.map((x, i) => x - lr * delta[i])
  },
  Adamax: (w, g0, st, hp) => {
    const [b1, b2, eps, lr] = [hp('b1'), hp('b2'), hp('eps'), hp('lr')]
    const g = decayed(g0, w, hp('weight_decay'))
    st.buf.exp_avg = (st.buf.exp_avg ?? zerosLike(g)).map((m, i) => m + (1 - b1) * (g[i] - m))
    st.buf.exp_inf = (st.buf.exp_inf ?? zerosLike(g)).map((v, i) => Math.max(v * b2, Math.abs(g[i]) + eps))
    const clr = lr / (1 - b1 ** st.step)
    return w.map((x, i) => x - (clr * st.buf.exp_avg[i]) / st.buf.exp_inf[i])
  },
  NAdam: (w0, g0, st, hp) => {
    const [b1, b2, eps, lr, wd, md] = [hp('b1'), hp('b2'), hp('eps'), hp('lr'), hp('weight_decay'), hp('momentum_decay')]
    const decoupled = wd && hp('decoupled_weight_decay')
    const w = decoupled ? w0.map((x) => x * (1 - lr * wd)) : w0
    const g = decoupled ? g0 : decayed(g0, w, wd)
    const mu = b1 * (1 - 0.5 * 0.96 ** (st.step * md))
    const next = b1 * (1 - 0.5 * 0.96 ** ((st.step + 1) * md))
    st.buf.mu_product = [Math.fround((st.buf.mu_product?.[0] ?? 1) * mu)]
    const mp = st.buf.mu_product[0]
    st.buf.exp_avg = (st.buf.exp_avg ?? zerosLike(g)).map((m, i) => m + (1 - b1) * (g[i] - m))
    st.buf.exp_avg_sq = (st.buf.exp_avg_sq ?? zerosLike(g)).map((v, i) => b2 * v + (1 - b2) * g[i] * g[i])
    const c2 = 1 - b2 ** st.step
    const denom = st.buf.exp_avg_sq.map((v) => Math.sqrt(v / c2) + eps)
    return w.map((x, i) => x + (g[i] / denom[i]) * ((-lr * (1 - mu)) / (1 - mp)) + (st.buf.exp_avg[i] / denom[i]) * ((-lr * next) / (1 - mp * next)))
  },
  RAdam: (w0, g0, st, hp) => {
    const [b1, b2, eps, lr, wd] = [hp('b1'), hp('b2'), hp('eps'), hp('lr'), hp('weight_decay')]
    const decoupled = wd && hp('decoupled_weight_decay')
    const w = decoupled ? w0.map((x) => x * (1 - lr * wd)) : w0
    const g = decoupled ? g0 : decayed(g0, w, wd)
    st.buf.exp_avg = (st.buf.exp_avg ?? zerosLike(g)).map((m, i) => m + (1 - b1) * (g[i] - m))
    st.buf.exp_avg_sq = (st.buf.exp_avg_sq ?? zerosLike(g)).map((v, i) => b2 * v + (1 - b2) * g[i] * g[i])
    const [c1, c2] = [1 - b1 ** st.step, 1 - b2 ** st.step]
    const rhoInf = 2 / (1 - b2) - 1
    const rho = rhoInf - (2 * st.step * b2 ** st.step) / c2
    const rect = rho > 5 ? Math.sqrt(((rho - 4) * (rho - 2) * rhoInf) / ((rhoInf - 4) * (rhoInf - 2) * rho)) : 0
    return w.map((x, i) => x - (st.buf.exp_avg[i] / c1) * lr * (rho > 5 ? (Math.sqrt(c2) / (Math.sqrt(st.buf.exp_avg_sq[i]) + eps)) * rect : 1))
  },
  ASGD: (w, g0, st, hp) => {
    const [lr, lambd, alpha, t0] = [hp('lr'), hp('lambd'), hp('alpha'), hp('t0')]
    const g = decayed(g0, w, hp('weight_decay'))
    const eta = st.buf.eta?.[0] ?? Math.fround(lr)
    const mu = st.buf.mu?.[0] ?? 1
    if (!st.buf.eta) st.buf.eta = [eta]
    if (!st.buf.mu) st.buf.mu = [mu]
    const next = w.map((x, i) => x * (1 - lambd * eta) - eta * g[i])
    const ax = st.buf.ax ?? zerosLike(g)
    st.buf.ax = mu !== 1 ? ax.map((v, i) => v + (next[i] - v) * mu) : [...next]
    st.buf.eta = [lr / (1 + lambd * lr * st.step) ** alpha]
    st.buf.mu = [1 / Math.max(1, st.step - t0)]
    return next
  },
  Rprop: (w, g, st, hp) => {
    const [minus, plus, lo, hi] = [hp('eta-'), hp('eta+'), hp('size-'), hp('size+')]
    const prev = st.buf.prev ?? zerosLike(g)
    const size = st.buf.step_size ?? g.map(() => hp('lr'))
    const factor = g.map((x, i) => (x * prev[i] > 0 ? plus : x * prev[i] < 0 ? minus : 1))
    st.buf.step_size = size.map((v, i) => Math.min(hi, Math.max(lo, v * factor[i])))
    const kept = g.map((x, i) => (factor[i] === minus ? 0 : x))
    st.buf.prev = kept
    return w.map((x, i) => x - Math.sign(kept[i]) * st.buf.step_size[i])
  },
}

/** the state of a single number rather than one per weight */
const SCALAR_STATE = new Set(['mu_product', 'eta', 'mu'])

/** optimizer.state: per parameter, a dict; the built-in rules keep their buffers as plain numbers and show them here */
class OptState extends PyObj {
  readonly cls = 'defaultdict'
  opt: Optimizer
  constructor(opt: Optimizer) {
    super()
    this.opt = opt
  }
  private dictOf(p: TT): Value {
    const made = this.opt.custom.get(p)
    if (made) return made
    const st = this.opt.state.get(p)
    if (!st) {
      const fresh: Value = { k: 'dict', d: new PyDict() }
      this.opt.custom.set(p, fresh)
      return fresh
    }
    return this.opt.stateDict(p, st)
  }
  private entries(): [TT, Value][] {
    const out: [TT, Value][] = []
    for (const p of this.opt.params()) {
      const st = this.opt.state.get(p)
      if (st && this.opt.hasState(st)) out.push([p, this.opt.stateDict(p, st)])
      else if (this.opt.custom.has(p)) out.push([p, this.opt.custom.get(p)!])
    }
    return out
  }
  repr() { return `defaultdict(<class 'dict'>, {${this.entries().map(([p, d]) => `${repr(wrap(p))}: ${repr(d)}`).join(', ')}})` }
  len() { return this.entries().length }
  truthy() { return this.entries().length > 0 }
  iter() { return this.entries().map(([p]) => wrap(p)) }
  contains(v: Value) { return this.entries().some(([p]) => p === asTT(v)) }
  getItem(idx: Value, h: Host): Value {
    const p = asTT(idx)
    if (!p) throw h.err('KeyError', repr(idx))
    return this.dictOf(p)
  }
  getAttr(name: string): Value | undefined {
    const fn = (call: () => Value): Value => ({ k: 'fn', name, call })
    if (name === 'keys') return fn(() => py.list(this.iter()))
    if (name === 'values') return fn(() => py.list(this.entries().map(([, d]) => d)))
    if (name === 'items') return fn(() => py.list(this.entries().map(([p, d]) => py.tuple([wrap(p), d]))))
    return undefined
  }
}

/** what LBFGS carries from one step() to the next */
interface LbfgsState { funcEvals: number; nIter: number; d: number[]; t: number; dirs: number[][]; stps: number[][]; ro: number[]; hDiag: number; prev: number[] | null; prevLoss: number }

class Optimizer extends PyObj {
  readonly cls: string
  groups: Value
  defaults: [string, Value][]
  state = new Map<TT, ParamState>()
  /** self.state[p] of a Python subclass, which keeps what it likes there */
  custom = new Map<TT, Value>()
  lbfgs: LbfgsState | null = null
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
  params() { return this.paramsIn() }
  hasState(st: ParamState) { return this.cls !== 'SGD' || Object.keys(st.buf).length > 0 }
  /** one parameter's state as state_dict() and optimizer.state show it */
  stateDict(p: TT, st: ParamState): Value {
    // SGD keeps no step count
    const step: [Value, Value][] = this.cls === 'SGD' ? [] : [[py.str('step'), wrap(TT.scalar(st.step, 'float32'))]]
    return { k: 'dict', d: PyDict.from([...step, ...Object.entries(st.buf).map(([k, v]): [Value, Value] => [py.str(k), wrap(SCALAR_STATE.has(k) ? TT.scalar(v[0], 'float32') : TT.of(v, p.shape, p.dtype))])]) }
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
    if (!UPDATE[this.cls]) throw h.err('NotImplementedError', '')
    for (const d of this.groupDicts()) {
      const fromPair = (key: string, i: number) => h.num(h.iterate(d.get(py.str(key))!)[i])
      const hp: Hyper = (key) => {
        const v = d.get(py.str(key))
        if (key === 'b1' || key === 'b2') return fromPair('betas', key === 'b1' ? 0 : 1)
        if (key === 'eta-' || key === 'eta+') return fromPair('etas', key === 'eta-' ? 0 : 1)
        if (key === 'size-' || key === 'size+') return fromPair('step_sizes', key === 'size-' ? 0 : 1)
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
  /** torch.optim.LBFGS.step, without a line search unless it was refused at construction */
  private lbfgsStep(h: Host, closure: Value): Value {
    const E = stateOf(h).E
    const d0 = this.groupDicts()[0]
    const opt = (k: string) => h.num(d0.get(py.str(k))!)
    const [lr, maxIter, maxEval, tolGrad, tolChange, history] = [opt('lr'), opt('max_iter'), opt('max_eval'), opt('tolerance_grad'), opt('tolerance_change'), opt('history_size')]
    const maximize = h.truthy(d0.get(py.str('maximize'))!)
    const ps = this.paramsIn()
    const call = () => {
      const was = E.grad
      E.grad = true
      try {
        return h.call(closure, [])
      } finally {
        E.grad = was
      }
    }
    const R = (v: number) => Math.fround(v)
    const dot = (a: number[], b: number[]) => R(a.reduce((s, x, i) => s + x * b[i], 0))
    const flat = () => ps.flatMap((p) => (p.grad ? (maximize ? p.grad.values().map((x) => -x) : p.grad.values()) : zerosLike(p.values())))
    const peak = (a: number[]) => a.reduce((m, x) => Math.max(m, Math.abs(x)), 0)
    const st = (this.lbfgs ??= { funcEvals: 0, nIter: 0, d: [], t: 0, dirs: [], stps: [], ro: [], hDiag: 1, prev: null, prevLoss: 0 })
    const orig = call()
    const lossOf = (v: Value) => (maximize ? -h.num(v) : h.num(v))
    let loss = lossOf(orig)
    let evals = 1
    st.funcEvals++
    let g = flat()
    let optCond = peak(g) <= tolGrad
    if (optCond) return orig
    let n = 0
    while (n < maxIter) {
      n++
      st.nIter++
      if (st.nIter === 1) {
        st.d = g.map((x) => -x)
        st.dirs = []
        st.stps = []
        st.ro = []
        st.hDiag = 1
      } else {
        const y = g.map((x, i) => R(x - st.prev![i]))
        const step = st.d.map((x) => R(x * st.t))
        const ys = dot(y, step)
        if (ys > 1e-10) {
          if (st.dirs.length === history) {
            st.dirs.shift()
            st.stps.shift()
            st.ro.shift()
          }
          st.dirs.push(y)
          st.stps.push(step)
          st.ro.push(R(1 / ys))
          st.hDiag = R(ys / dot(y, y))
        }
        const al: number[] = new Array<number>(st.dirs.length)
        const q = g.map((x) => -x)
        for (let i = st.dirs.length - 1; i >= 0; i--) {
          al[i] = R(dot(st.stps[i], q) * st.ro[i])
          for (let k = 0; k < q.length; k++) q[k] = R(q[k] - al[i] * st.dirs[i][k])
        }
        const r = q.map((x) => R(x * st.hDiag))
        for (let i = 0; i < st.dirs.length; i++) {
          const be = R(dot(st.dirs[i], r) * st.ro[i])
          for (let k = 0; k < r.length; k++) r[k] = R(r[k] + (al[i] - be) * st.stps[i][k])
        }
        st.d = r
      }
      st.prev = [...g]
      const prevLoss = loss
      st.prevLoss = prevLoss
      st.t = st.nIter === 1 ? R(Math.min(1, R(1 / g.reduce((s, x) => s + Math.abs(x), 0))) * lr) : lr
      if (dot(g, st.d) > -tolChange) break
      let at = 0
      for (const p of ps) {
        const size = p.size
        C.writeInto(p, p.values().map((x, i) => x + st.t * st.d[at + i]))
        at += size
      }
      let lsEvals = 0
      if (n !== maxIter) {
        loss = lossOf(call())
        g = flat()
        optCond = peak(g) <= tolGrad
        lsEvals = 1
      }
      evals += lsEvals
      st.funcEvals += lsEvals
      if (n === maxIter || evals >= maxEval || optCond) break
      if (peak(st.d.map((x) => x * st.t)) <= tolChange) break
      if (Math.abs(loss - prevLoss) < tolChange) break
    }
    return orig
  }
  getAttr(name: string, h: Host): Value | undefined {
    const fn = (call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })
    switch (name) {
      case 'state': return py.obj(new OptState(this))
      case 'zero_grad': return fn((args, kw) => {
        const toNone = flagOf(h, kw.set_to_none ?? args[0], true)
        for (const p of this.paramsIn()) p.grad = toNone || !p.grad ? null : TT.of(zerosLike(p.grad.values()), p.shape, p.dtype)
        return py.NONE
      })
      case 'step': return fn((args, kw) => {
        const closure = args[0] ?? kw.closure
        const E = stateOf(h).E
        if (this.cls === 'LBFGS') {
          if (!closure) throw h.err('TypeError', "LBFGS.step() missing 1 required positional argument: 'closure'")
          return this.lbfgsStep(h, closure)
        }
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
          if (st && this.hasState(st)) state.set(py.int(i), this.stateDict(p, st))
        })
        let at = 0
        // 'params' comes last, as PyTorch packs it
        const groups = this.groupDicts().map((d): Value => {
          const n = (d.get(py.str('params')) as { items: Value[] }).items.length
          at += n
          return { k: 'dict', d: PyDict.from([...d.items().filter(([k]) => !(k.k === 'str' && k.v === 'params')), [py.str('params'), py.list(Array.from({ length: n }, (_, i) => py.int(at - n + i)))]]) }
        })
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
            else st.buf[str(n)] = [...t.values()]
          }
          this.state.set(p, st)
        }
        return py.NONE
      })
    }
    return undefined
  }
}

type LrRule = (s: Scheduler) => number[]

/** torch.optim.lr_scheduler.LRScheduler: each step() writes new learning rates into the optimiser's param_groups */
class Scheduler extends PyObj {
  readonly cls: string
  h: Host
  opt: Optimizer
  lastEpoch: number
  baseLrs: number[] = []
  lastLr: number[] = []
  stepCount = 0
  isInitial = false
  /** the constructor's own attributes, in the order they appear in __dict__ */
  vals: Map<string, Value>
  rule: LrRule
  closed: LrRule | null
  /** a whole step() of its own (CosineAnnealingWarmRestarts) */
  customStep: ((s: Scheduler, epoch: number | null) => void) | null = null
  /** added to state_dict() after the base fields (LambdaLR) */
  tail: [string, Value][] = []
  constructor(kind: string, h: Host, opt: Optimizer, lastEpoch: number, vals: [string, Value][], rule: LrRule, closed: LrRule | null = null) {
    super()
    this.cls = kind
    this.h = h
    this.opt = opt
    this.lastEpoch = lastEpoch
    this.vals = new Map(vals)
    this.rule = rule
    this.closed = closed
    this.dicts().forEach((d, i) => {
      if (lastEpoch === -1) {
        if (!d.has(py.str('initial_lr'))) d.set(py.str('initial_lr'), d.get(py.str('lr'))!)
      } else if (!d.has(py.str('initial_lr'))) {
        throw h.err('KeyError', repr(py.str(`param 'initial_lr' is not specified in param_groups[${i}] when resuming scheduler with last_epoch >= 0.\nThis typically happens when:\n1. You're trying to resume training from a checkpoint but haven't properly loaded the optimizer state\n2. You're using last_epoch >= 0 for a fresh training run (not recommended)`)))
      }
    })
    this.baseLrs = this.dicts().map((d) => h.num(d.get(py.str('initial_lr'))!))
  }
  /** _initial_step: the rule runs once from the constructor, with its first-step flag up */
  start() {
    this.stepCount = 0
    this.isInitial = true
    try {
      this.step(null)
    } finally {
      this.isInitial = false
    }
    return this
  }
  dicts(): PyDict[] { return (this.opt.groups as { items: Value[] }).items.map((g) => (g as { d: PyDict }).d) }
  lrs() { return this.dicts().map((d) => this.h.num(d.get(py.str('lr'))!)) }
  n(key: string) { return this.h.num(this.vals.get(key)!) }
  write(values: number[]) {
    this.dicts().forEach((d, i) => d.set(py.str('lr'), py.float(values[i])))
    this.lastLr = this.lrs()
  }
  update(epoch: number | null) {
    if (epoch === null) {
      this.lastEpoch++
      this.write(this.rule(this))
    } else {
      this.lastEpoch = epoch
      this.write((this.closed ?? this.rule)(this))
    }
  }
  step(epoch: number | null) {
    if (this.customStep) {
      this.customStep(this, epoch)
      return
    }
    this.stepCount++
    this.update(epoch)
  }
  repr() { return `<torch.optim.lr_scheduler.${this.cls} object>` }
  isa(name: string) { return name === 'LRScheduler' }
  stateFields(): [Value, Value][] {
    return [
      ...[...this.vals].map(([k, v]): [Value, Value] => [py.str(k), v]),
      [py.str('base_lrs'), py.list(this.baseLrs.map(py.float))],
      [py.str('last_epoch'), py.int(this.lastEpoch)],
      [py.str('_step_count'), py.int(this.stepCount)],
      [py.str('_is_initial'), py.bool(this.isInitial)],
      [py.str('_get_lr_called_within_step'), py.bool(false)],
      [py.str('_last_lr'), py.list(this.lastLr.map(py.float))],
      ...this.tail.map(([k, v]): [Value, Value] => [py.str(k), v]),
    ]
  }
  getAttr(name: string, h: Host): Value | undefined {
    switch (name) {
      case 'step': return { k: 'fn', name, call: (args, kw) => {
        const e = args[0] ?? kw.epoch
        this.step(e && e.k !== 'none' ? h.num(e) : null)
        return py.NONE
      } }
      case 'get_last_lr': return { k: 'fn', name, call: () => py.list(this.lastLr.map(py.float)) }
      case 'get_lr': return { k: 'fn', name, call: () => py.list(this.rule(this).map(py.float)) }
      case 'last_epoch': return py.int(this.lastEpoch)
      case 'base_lrs': return py.list(this.baseLrs.map(py.float))
      case 'optimizer': return py.obj(this.opt)
      case '_step_count': return py.int(this.stepCount)
      case '_last_lr': return py.list(this.lastLr.map(py.float))
      case 'state_dict': return { k: 'fn', name, call: () => ({ k: 'dict', d: PyDict.from(this.stateFields()) }) }
      case 'load_state_dict': return { k: 'fn', name, call: (args) => {
        for (const [k, v] of (args[0] as { d: PyDict }).d.items()) {
          const key = h.str(k)
          if (key === 'base_lrs') this.baseLrs = h.iterate(v).map((x) => h.num(x))
          else if (key === 'last_epoch') this.lastEpoch = h.toInt(v)
          else if (key === '_step_count') this.stepCount = h.toInt(v)
          else if (key === '_last_lr') this.lastLr = h.iterate(v).map((x) => h.num(x))
          else if (this.vals.has(key)) this.vals.set(key, v)
        }
        return py.NONE
      } }
    }
    return this.vals.get(name)
  }
}

/** SequentialLR and ChainedScheduler: other schedulers, taking turns or all at once */
class SchedulerGroup extends PyObj {
  readonly cls: string
  h: Host
  opt: Optimizer
  scheds: Scheduler[]
  milestones: number[] | null
  lastEpoch: number
  lastLr: number[]
  constructor(kind: string, h: Host, opt: Optimizer, scheds: Scheduler[], milestones: number[] | null, lastEpoch: number) {
    super()
    this.cls = kind
    this.h = h
    this.opt = opt
    this.scheds = scheds
    this.milestones = milestones
    this.lastEpoch = lastEpoch + 1
    if (milestones) {
      // back to the start: every scheduler has already taken its first step
      for (const d of scheds[0].dicts()) d.set(py.str('lr'), d.get(py.str('initial_lr'))!)
      for (const s0 of scheds) s0.lastEpoch--
      const first = scheds[this.pick(0)]
      first.start()
      this.lastLr = first.lastLr
    } else {
      this.lastLr = scheds[scheds.length - 1].lrs()
    }
  }
  private pick(epoch: number) { return this.milestones!.filter((m) => m <= epoch).length }
  repr() { return `<torch.optim.lr_scheduler.${this.cls} object>` }
  isa(name: string) { return name === 'LRScheduler' }
  step() {
    if (!this.milestones) {
      for (const s0 of this.scheds) s0.step(null)
      this.lastLr = this.scheds[this.scheds.length - 1].lrs()
      return
    }
    this.lastEpoch++
    const idx = this.pick(this.lastEpoch)
    const sc = this.scheds[idx]
    if (idx > 0 && this.milestones[idx - 1] === this.lastEpoch) sc.update(0)
    else sc.step(null)
    this.lastLr = sc.lastLr
  }
  getAttr(name: string, h: Host): Value | undefined {
    switch (name) {
      case 'step': return { k: 'fn', name, call: () => { this.step(); return py.NONE } }
      case 'get_last_lr': return { k: 'fn', name, call: () => py.list(this.lastLr.map(py.float)) }
      case 'optimizer': return py.obj(this.opt)
      case '_schedulers': return py.list(this.scheds.map(py.obj))
      case 'last_epoch': return this.milestones ? py.int(this.lastEpoch) : undefined
      case '_milestones': return this.milestones ? py.list(this.milestones.map(py.int)) : undefined
      case 'state_dict': return { k: 'fn', name, call: () => {
        const own: [Value, Value][] = this.milestones ? [[py.str('_milestones'), py.list(this.milestones.map(py.int))], [py.str('last_epoch'), py.int(this.lastEpoch)]] : []
        return { k: 'dict', d: PyDict.from([...own, [py.str('_last_lr'), py.list(this.lastLr.map(py.float))], [py.str('_schedulers'), py.list(this.scheds.map((x) => h.call(h.getAttr(py.obj(x), 'state_dict'), [])))]]) }
      } }
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
  minLrGiven: Value
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
    this.minLrGiven = minLr && minLr.k !== 'list' && minLr.k !== 'tuple' ? minLr : minLr ? py.NONE : py.int(0)
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
      case 'patience': return py.int(this.patience)
      case 'factor': return py.float(this.factor)
      case 'cooldown': return py.int(this.cooldown)
      case 'cooldown_counter': return py.int(this.cool)
      case 'threshold': return py.float(this.threshold)
      case 'threshold_mode': return py.str(this.absMode ? 'abs' : 'rel')
      case 'mode': return py.str(this.max ? 'max' : 'min')
      case 'eps': return py.float(this.eps)
      case 'min_lrs': return py.list(this.minLr.map(numberValue))
      case 'state_dict': return { k: 'fn', name, call: () => ({ k: 'dict', d: PyDict.from(([
        ['factor', py.float(this.factor)], ['default_min_lr', this.minLrGiven], ['min_lrs', py.list(this.minLr.map(numberValue))], ['patience', py.int(this.patience)],
        ['cooldown', py.int(this.cooldown)], ['eps', py.float(this.eps)], ['last_epoch', py.int(this.lastEpoch)], ['_last_lr', py.list(this.lastLr.map(py.float))],
        ['mode_worse', py.float(this.max ? -Infinity : Infinity)], ['mode', py.str(this.max ? 'max' : 'min')], ['threshold', py.float(this.threshold)],
        ['threshold_mode', py.str(this.absMode ? 'abs' : 'rel')], ['best', py.float(this.best)], ['cooldown_counter', py.int(this.cool)], ['num_bad_epochs', py.int(this.bad)],
      ] as [string, Value][]).map(([k, v]): [Value, Value] => [py.str(k), v])) }) }
    }
    return undefined
  }
}

// ---------------------------------------------------------------- data

/** the library's datasets: dataset + dataset is a ConcatDataset */
abstract class DatasetObj extends PyObj {
  isa(name: string) { return name === 'Dataset' }
  binop(op: string, other: Value, reflected: boolean): Value | undefined {
    if (op !== '+' || reflected) return undefined
    return py.obj(new ConcatDataset([py.obj(this), other]))
  }
}

class TensorDataset extends DatasetObj {
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
  getAttr(name: string): Value | undefined { return name === 'tensors' ? py.tuple(this.tensors.map(wrap)) : undefined }
}

const datasetLen = (h: Host, ds: Value): number => {
  if (ds.k === 'obj' && ds.o.len) return ds.o.len(h)
  if (ds.k === 'list' || ds.k === 'tuple') return ds.items.length
  const f = ds.k === 'inst' ? h.getAttrOr(ds, '__len__') : undefined
  if (!f) throw h.err('TypeError', `object of type '${typeName(ds)}' has no len()`)
  return h.num(h.call(f, []))
}
/** dataset[idx], whatever kind of dataset (a list and a tensor will do too) */
const datasetGet = (h: Host, ds: Value, idx: Value): Value => {
  if (ds.k === 'obj' && ds.o.getItem) return ds.o.getItem(idx, h)
  if (ds.k === 'list' || ds.k === 'tuple') {
    const i = h.toInt(idx)
    const n = ds.items.length
    if (i < -n || i >= n) throw h.err('IndexError', `${ds.k} index out of range`)
    return ds.items[i < 0 ? i + n : i]
  }
  return h.call(h.getAttr(ds, '__getitem__'), [idx])
}
const datasetItem = (h: Host, ds: Value, i: number): Value => datasetGet(h, ds, py.int(i))

/** the class a Python subclass of IterableDataset descends from: set when the module is built */
const isIterableDataset = (ds: Value): boolean => {
  if (ds.k === 'obj') return ds.o instanceof ChainDataset
  if (ds.k !== 'inst') return false
  const walk = (c: ClassValue): boolean => c === ITERABLE_DATASET.cls || c.bases.some((b) => b.k === 'class' && walk(b))
  return walk(ds.cls)
}
const ITERABLE_DATASET: { cls: ClassValue | null } = { cls: null }

/** random_split's pieces */
class Subset extends DatasetObj {
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
  private at(h: Host, v: Value) {
    const n = this.indices.length
    const i = h.toInt(v)
    if (i < -n || i >= n) throw h.err('IndexError', 'list index out of range')
    return this.indices[i < 0 ? i + n : i]
  }
  getItem(idx: Value, h: Host): Value {
    // a list goes to the dataset as a list, as PyTorch's Subset does
    if (idx.k === 'list') return datasetGet(h, this.ds, py.list(idx.items.map((v) => py.int(this.at(h, v)))))
    return datasetItem(h, this.ds, this.at(h, idx))
  }
  getAttr(name: string): Value | undefined { return name === 'dataset' ? this.ds : name === 'indices' ? py.list(this.indices.map(py.int)) : undefined }
}

class ConcatDataset extends DatasetObj {
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
    if (i < -n) throw h.err('ValueError', 'absolute value of index should not exceed dataset length')
    if (i >= n) throw h.err('IndexError', 'list index out of range')
    if (i < 0) i += n
    for (const d of this.parts) {
      const size = datasetLen(h, d)
      if (i < size) return datasetItem(h, d, i)
      i -= size
    }
    throw h.err('IndexError', 'list index out of range')
  }
  getAttr(name: string, h: Host): Value | undefined {
    if (name === 'datasets') return py.list(this.parts)
    if (name === 'cumulative_sizes') {
      let at = 0
      return py.list(this.parts.map((d) => py.int((at += datasetLen(h, d)))))
    }
    return undefined
  }
}

/** one after the other, for iterable datasets */
class ChainDataset extends DatasetObj {
  readonly cls = 'ChainDataset'
  parts: Value[]
  constructor(parts: Value[]) {
    super()
    this.parts = parts
  }
  repr() { return '<torch.utils.data.dataset.ChainDataset object>' }
  binop(op: string, other: Value, reflected: boolean): Value | undefined {
    if (op !== '+' || reflected) return undefined
    return py.obj(new ChainDataset([py.obj(this), other]))
  }
  iter(h: Host) { return this.parts.flatMap((d) => h.iterate(d)) }
  len(h: Host) { return this.parts.reduce((s, d) => s + datasetLen(h, d), 0) }
  getAttr(name: string): Value | undefined { return name === 'datasets' ? py.list(this.parts) : undefined }
}

class StackDataset extends DatasetObj {
  readonly cls = 'StackDataset'
  parts: [string, Value][]
  named: boolean
  size: number
  constructor(h: Host, parts: [string, Value][], named: boolean) {
    super()
    this.parts = parts
    this.named = named
    this.size = datasetLen(h, parts[0][1])
    if (parts.some(([, d]) => datasetLen(h, d) !== this.size)) throw h.err('ValueError', 'Size mismatch between datasets')
  }
  repr() { return '<torch.utils.data.dataset.StackDataset object>' }
  len() { return this.size }
  getItem(idx: Value, h: Host): Value {
    const got = this.parts.map(([k, d]): [string, Value] => [k, datasetGet(h, d, idx)])
    return this.named ? { k: 'dict', d: PyDict.from(got.map(([k, v]): [Value, Value] => [py.str(k), v])) } : py.tuple(got.map(([, v]) => v))
  }
  getAttr(name: string): Value | undefined {
    if (name !== 'datasets') return undefined
    return this.named ? { k: 'dict', d: PyDict.from(this.parts.map(([k, v]): [Value, Value] => [py.str(k), v])) } : py.tuple(this.parts.map(([, v]) => v))
  }
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

/** default_collate (torch/utils/data/_utils/collate.py): the batch keeps its own container for strings */
const collate = (h: Host, batch: Value): Value => {
  const E = stateOf(h).E
  const samples = h.iterate(batch)
  const first = samples[0]
  if (asTT(first)) return wrap(C.catT(E, samples.map((v) => tensorArg(h, v, 'default_collate')), 0, true))
  if (first.k === 'array') return wrap(C.catT(E, samples.map((v) => fromNd((v as { a: NDArray }).a, false)), 0, true))
  if (first.k === 'float') return wrap(TT.of(samples.map((v) => h.num(v)), [samples.length], first.f32 ? 'float32' : 'float64'))
  if (first.k === 'int') return wrap(TT.of(samples.map((v) => h.num(v)), [samples.length], 'int64'))
  if (first.k === 'bool') return wrap(TT.of(samples.map((v) => h.num(v)), [samples.length], 'bool'))
  if (first.k === 'str') return batch
  if (first.k === 'dict') return { k: 'dict', d: PyDict.from(first.d.keys().map((key): [Value, Value] => [key, collate(h, py.list(samples.map((v) => {
    const got = (v as { d: PyDict }).d.get(key)
    if (!got) throw h.err('KeyError', repr(key))
    return got
  })))])) }
  if (first.k === 'tuple' || first.k === 'list') {
    const size = first.items.length
    if (samples.some((v) => h.iterate(v).length !== size)) throw h.err('RuntimeError', 'each element in list of batch should be of equal size')
    const rows = samples.map((v) => h.iterate(v))
    const out = first.items.map((_, c) => collate(h, py.tuple(rows.map((r) => r[c]))))
    return first.k === 'tuple' && first.names ? { ...first, items: out } : py.list(out)
  }
  throw h.err('TypeError', `default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class '${first.k === 'inst' ? `__main__.${first.cls.name}` : typeName(first)}'>`)
}

/** default_convert: what a loader with batch_size=None does to each sample */
const defaultConvert = (h: Host, v: Value): Value => {
  if (asTT(v)) return v
  if (v.k === 'array') return wrap(fromNd(v.a, true))
  if (v.k === 'dict') return { ...v, d: PyDict.from(v.d.items().map(([k, x]): [Value, Value] => [k, defaultConvert(h, x)])) }
  if (v.k === 'tuple') return v.names ? { ...v, items: v.items.map((x) => defaultConvert(h, x)) } : py.list(v.items.map((x) => defaultConvert(h, x)))
  if (v.k === 'list') return py.list(v.items.map((x) => defaultConvert(h, x)))
  return v
}

class BatchSampler extends PyObj {
  readonly cls = 'BatchSampler'
  sampler: Value
  size: number
  dropLast: boolean
  constructor(sampler: Value, size: number, dropLast: boolean) {
    super()
    this.sampler = sampler
    this.size = size
    this.dropLast = dropLast
  }
  repr() { return '<torch.utils.data.sampler.BatchSampler object>' }
  isa(name: string) { return name === 'Sampler' }
  iter(h: Host) {
    const order = h.iterate(this.sampler)
    const out: Value[] = []
    const stop = this.dropLast ? order.length - (order.length % this.size) : order.length
    for (let at = 0; at < stop; at += this.size) out.push(py.list(order.slice(at, at + this.size)))
    return out
  }
  len(h: Host) {
    const n = datasetLen(h, this.sampler)
    return this.dropLast ? Math.floor(n / this.size) : Math.ceil(n / this.size)
  }
  getAttr(name: string): Value | undefined {
    if (name === 'sampler') return this.sampler
    if (name === 'batch_size') return py.int(this.size)
    if (name === 'drop_last') return py.bool(this.dropLast)
    return undefined
  }
}

class DataLoader extends PyObj {
  readonly cls = 'DataLoader'
  ds: Value
  bs: number | null
  dropLast: boolean
  sampler: Value
  batchSampler: Value
  collateFn: Value | null
  fields: Record<string, Value | undefined>
  constructor(ds: Value, bs: number | null, dropLast: boolean, sampler: Value, batchSampler: Value, collateFn: Value | null, fields: Record<string, Value | undefined>) {
    super()
    this.ds = ds
    this.bs = bs
    this.dropLast = dropLast
    this.sampler = sampler
    this.batchSampler = batchSampler
    this.collateFn = collateFn
    this.fields = fields
  }
  repr() { return '<torch.utils.data.dataloader.DataLoader object>' }
  len(h: Host) {
    if (isIterableDataset(this.ds)) {
      const n = datasetLen(h, this.ds)
      return this.bs === null ? n : this.dropLast ? Math.floor(n / this.bs) : Math.ceil(n / this.bs)
    }
    return this.batchSampler.k === 'none' ? datasetLen(h, this.sampler) : datasetLen(h, this.batchSampler)
  }
  getAttr(name: string): Value | undefined {
    switch (name) {
      case 'dataset': return this.ds
      case 'batch_size': return this.bs === null ? py.NONE : py.int(this.bs)
      case 'drop_last': return py.bool(this.dropLast)
      case 'sampler': return this.sampler
      case 'batch_sampler': return this.batchSampler
      case 'collate_fn': return this.collateFn ?? undefined
    }
    return this.fields[name]
  }
  iter(h: Host): Value[] {
    const finish = (samples: Value[]) => (this.collateFn ? h.call(this.collateFn, [py.list(samples)]) : collate(h, py.list(samples)))
    if (isIterableDataset(this.ds)) {
      const all = h.iterate(this.ds)
      if (this.bs === null) return all.map((v) => (this.collateFn ? h.call(this.collateFn, [v]) : defaultConvert(h, v)))
      const out: Value[] = []
      const stop = this.dropLast ? all.length - (all.length % this.bs) : all.length
      for (let at = 0; at < stop; at += this.bs) out.push(finish(all.slice(at, at + this.bs)))
      return out
    }
    if (this.batchSampler.k === 'none') return h.iterate(this.sampler).map((i) => {
      const v = datasetGet(h, this.ds, i)
      return this.collateFn ? h.call(this.collateFn, [v]) : defaultConvert(h, v)
    })
    return h.iterate(this.batchSampler).map((b) => finish(h.iterate(b).map((i) => datasetGet(h, this.ds, i))))
  }
}

/** what a custom autograd.Function's forward and backward share */
class FunctionCtx extends PyObj {
  readonly cls = 'BackwardCFunction'
  saved: Value[] = []
  extras = new Map<string, Value>()
  repr() { return '<torch.autograd.function.BackwardCFunction object>' }
  getAttr(name: string): Value | undefined {
    if (name === 'save_for_backward') return { k: 'fn', name, call: (args) => { this.saved = args; return py.NONE } }
    if (name === 'saved_tensors') return py.tuple(this.saved)
    if (name === 'mark_non_differentiable' || name === 'mark_dirty' || name === 'set_materialize_grads') return { k: 'fn', name, call: () => py.NONE }
    return this.extras.get(name)
  }
  setAttr(name: string, v: Value) {
    this.extras.set(name, v)
    return true
  }
}

/** what torch.linalg.eig gives: read-only complex numbers, as this sandbox has no complex tensors */
class ComplexTensor extends PyObj {
  readonly cls = 'Tensor'
  re: number[]
  im: number[]
  shape: number[]
  constructor(re: number[], im: number[], shape: number[]) {
    super()
    this.re = re
    this.im = im
    this.shape = shape
  }
  repr() {
    const fmtR = C.tensorBody(this.shape, this.re, 'float32', 7).split(/[\s,[\]]+/).filter(Boolean)
    const fmtI = C.tensorBody(this.shape, this.im, 'float32', 7).split(/[\s,[\]]+/).filter(Boolean)
    // each part gets its own column width, as _Formatter builds one for the reals and one for the imaginaries
    const widthR = fmtR.reduce((m, x) => Math.max(m, x.length), 0)
    const widthI = fmtI.reduce((m, x) => Math.max(m, x.length), 0)
    const cell = (i: number) => {
      const imag = fmtI[i].padStart(widthI).trimStart()
      return `${fmtR[i].padStart(widthR)}${imag.startsWith('-') ? '' : '+'}${imag}j`
    }
    if (this.shape.length === 1) return `tensor([${this.re.map((_, i) => cell(i)).join(', ')}])`
    const [r, c] = this.shape
    const rows = Array.from({ length: r }, (_, i) => `[${Array.from({ length: c }, (_, j) => cell(i * c + j)).join(', ')}]`)
    return `tensor([${rows.join(',\n        ')}])`
  }
  len() { return this.shape[0] }
  getAttr(name: string): Value | undefined {
    switch (name) {
      case 'shape': return py.obj(new SizeObj(this.shape))
      case 'real': return wrap(TT.of(this.re, this.shape, 'float32'))
      case 'imag': return wrap(TT.of(this.im, this.shape, 'float32'))
      case 'dtype': return { k: 'obj', o: new ComplexDtype() }
      case 'is_complex': return { k: 'fn', name, call: () => py.bool(true) }
      case 'abs': return { k: 'fn', name, call: () => wrap(TT.of(this.re.map((x, i) => Math.hypot(x, this.im[i])), this.shape, 'float32')) }
    }
    return undefined
  }
}
const FLOAT_INFO: Partial<Record<TDType, [number, number, number, number, number, string[]]>> = {
  float16: [16, 0.0009765625, 65504, 6.103515625e-5, 0.001, ['0.001', '-65504', '65504', '0.000976562', '6.10352e-05']],
  float32: [32, 1.1920928955078125e-7, 3.4028234663852886e38, 1.1754943508222875e-38, 1e-6, ['1e-06', '-3.40282e+38', '3.40282e+38', '1.19209e-07', '1.17549e-38']],
  float64: [64, 2.220446049250313e-16, Number.MAX_VALUE, 2.2250738585072014e-308, 1e-15, ['1e-15', '-1.79769e+308', '1.79769e+308', '2.22045e-16', '2.22507e-308']],
}
class FloatInfo extends PyObj {
  readonly cls = 'finfo'
  d: TDType
  constructor(d: TDType) {
    super()
    this.d = d
  }
  repr() {
    const [res, min, max, eps, tiny] = FLOAT_INFO[this.d]![5]
    return `finfo(resolution=${res}, min=${min}, max=${max}, eps=${eps}, smallest_normal=${tiny}, tiny=${tiny}, dtype=${this.d})`
  }
  getAttr(name: string): Value | undefined {
    const [bits, eps, max, tiny, res] = FLOAT_INFO[this.d]!
    switch (name) {
      case 'bits': return py.int(bits)
      case 'eps': return py.float(eps)
      case 'max': return py.float(max)
      case 'min': return py.float(-max)
      case 'tiny': case 'smallest_normal': return py.float(tiny)
      case 'resolution': return py.float(res)
      case 'dtype': return py.str(this.d)
    }
    return undefined
  }
}
const INT_INFO: Partial<Record<TDType, [number, number, number]>> = { uint8: [8, 0, 255], int8: [8, -128, 127], int16: [16, -32768, 32767], int32: [32, -2147483648, 2147483647], int64: [64, -(2 ** 63), 2 ** 63] }
class IntInfo extends PyObj {
  readonly cls = 'iinfo'
  d: TDType
  constructor(d: TDType) {
    super()
    this.d = d
  }
  repr() {
    const [, min, max] = INT_INFO[this.d] ?? [0, 0, 0]
    const big = (x: number) => (Math.abs(x) >= 2 ** 53 ? BigInt(x > 0 ? '9223372036854775807' : '-9223372036854775808').toString() : String(x))
    return `iinfo(min=${big(min)}, max=${big(max)}, dtype=${this.d})`
  }
  getAttr(name: string): Value | undefined {
    const [bits, min, max] = INT_INFO[this.d] ?? [0, 0, 0]
    switch (name) {
      case 'bits': return py.int(bits)
      case 'min': return py.int(min)
      case 'max': return py.int(max)
      case 'dtype': return py.str(this.d)
    }
    return undefined
  }
}

/** a named constant that only prints: torch.strided, torch.preserve_format */
class DtypeLike extends PyObj {
  readonly cls: string
  text: string
  constructor(text: string) {
    super()
    this.text = text
    this.cls = text.split('.').pop()!
  }
  repr() { return this.text }
}

class ComplexDtype extends PyObj {
  readonly cls = 'dtype'
  repr() { return 'torch.complex64' }
}

// ---------------------------------------------------------------- the torch module

const fnv = (name: string, call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })
const numberValue = (x: number): Value => (Number.isInteger(x) ? py.int(x) : py.float(x))
/** a matrix as C++ prints one, in PyTorch's internal error messages */
const cppDump = (t: TT): string => {
  const v = t.values()
  const whole = v.every((x) => Number.isInteger(x))
  const text = v.map((x) => (whole ? String(x) : x.toFixed(4)))
  const w = text.reduce((m, x) => Math.max(m, x.length), 0)
  const cols = t.shape[t.ndim - 1]
  const rows: string[] = []
  for (let i = 0; i < text.length; i += cols) rows.push(text.slice(i, i + cols).map((x) => x.padStart(w + 1)).join(' '))
  return `${rows.join('\n')}\n[ torch.${C.typeLabel(t.dtype)}Tensor{${t.shape.join(',')}} ]`
}

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

/** a module anywhere inside a saved object: where the pickle would find its class */
const MODULE_FILE: Record<string, string> = { Linear: 'linear', Bilinear: 'linear', Identity: 'linear', Sequential: 'container', ModuleList: 'container', ModuleDict: 'container', Conv1d: 'conv', Conv2d: 'conv', Conv3d: 'conv', ReLU: 'activation', Sigmoid: 'activation', Tanh: 'activation', LSTM: 'rnn', GRU: 'rnn', RNN: 'rnn', Embedding: 'sparse', Dropout: 'dropout', BatchNorm1d: 'batchnorm', BatchNorm2d: 'batchnorm', LayerNorm: 'normalization', MaxPool2d: 'pooling', AvgPool2d: 'pooling', Flatten: 'flatten' }
const holdsModule = (v: Value): string | null => {
  if (v.k === 'inst' && moduleOf(v)) return `__main__.${v.cls.name}`
  const m = moduleOf(v)
  if (m) return `torch.nn.modules.${MODULE_FILE[m.kind] ?? 'module'}.${m.kind}`
  const inner = v.k === 'list' || v.k === 'tuple' ? v.items : v.k === 'dict' ? v.d.items().map(([, x]) => x) : []
  for (const x of inner) {
    const got = holdsModule(x)
    if (got) return got
  }
  return null
}

const build = (h: Host): Record<string, Value> => {
  const S = stateOf(h)
  const E = S.E
  RUN.host = h
  // print options and the default dtype belong to one run
  C.setPrintProfile('default')
  C.PRINT_OPTS.sciMode = null
  C.DEFAULTS.float = 'float32'
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
  const FACTORY_KW = ['size', 'dtype', 'device', 'requires_grad', 'layout', 'pin_memory', 'memory_format', 'names', 'generator', 'out']
  const knownKw = (name: string, kw: Kw, more: string[] = []) => {
    for (const k in kw) if (!FACTORY_KW.includes(k) && !more.includes(k)) throw h.err('TypeError', `${name}() got an unexpected keyword argument '${k}'`)
    onlyCpu(h, kw.device)
  }
  /** the size of zeros(2, 3), zeros((2, 3)) or zeros([2, 3]), with the argument parser's own complaints */
  const sizeArg = (name: string, args: Value[], kw: Kw, what = 'size'): number[] => {
    const items = kw.size ? [kw.size] : args
    if (!items.length) throw h.err('TypeError', `${name}() missing 1 required positional arguments: "${what}"`)
    const one = items.length === 1 ? items[0] : null
    const isDim = (x: Value) => x.k === 'int' || asTT(x) !== null
    let list: Value[]
    if (one && !isDim(one)) {
      if (one.k !== 'list' && one.k !== 'tuple' && !(one.k === 'obj' && one.o instanceof SizeObj)) throw h.err('TypeError', `${name}(): argument 'size' (position 1) must be tuple of ints, not ${typeName(one)}`)
      list = h.iterate(one)
    } else list = items
    list.forEach((x, i) => {
      if (!isDim(x)) throw h.err('TypeError', `${name}(): argument 'size' failed to unpack the object at pos ${i + 1} with error "type must be tuple of ints,but got ${typeName(x)}"`)
    })
    const dims = list.map((x) => h.toInt(x))
    const neg = dims.find((d) => d < 0)
    if (neg !== undefined) throw h.err('RuntimeError', name === 'zeros' ? 'zeros: Dimension size must be non-negative.' : `Trying to create tensor with negative dimension ${neg}: [${dims.join(', ')}]`)
    return dims
  }
  const filled = (name: string, value: (rng: SandboxRandom) => number, defaultDtype: TDType | null = null, kernel: string | null = null) =>
    fnv(name, (args, kw) => {
      knownKw(name, kw)
      const dims = sizeArg(name, args, kw)
      const n = prod(dims)
      if (n > h.maxSize) throw h.err('RuntimeError', `[enforce fail] tensor of ${n} elements exceeds the sandbox limit of ${h.maxSize}`)
      const dtype = dtypeArg(h, kw.dtype) ?? defaultDtype ?? C.DEFAULTS.float
      if (kernel && !isFloatT(dtype)) throw h.err('NotImplementedError', `"${kernel}" not implemented for '${C.typeLabel(dtype)}'`)
      const rng = rngOf(h, kw.generator)
      return finish(TT.of(Array.from({ length: n }, () => value(rng)), dims, dtype), kw, name)
    })
  const like = (name: string, value: () => number, float = false) =>
    fnv(name, (args, kw) => {
      const t = T(args[0], name)
      const dtype = dtypeArg(h, kw.dtype) ?? t.dtype
      if (float && !isFloatT(dtype)) throw h.err('NotImplementedError', `"${name === 'randn_like' ? 'normal_kernel_cpu' : 'check_uniform_bounds'}" not implemented for '${C.typeLabel(dtype)}'`)
      return finish(TT.of(t.values().map(() => value()), t.shape, dtype), kw, name)
    })
  const uni = (op: C.UnOpT) => fnv(op, (args) => {
    const t = T(args[0], op)
    const out = C.unary(E, op, t)
    trace(h, `torch.${op}`, 'elementwise', [{ label: label(h, args[0], 'x'), t }], out)
    flowStep(h, `torch.${op}`, t, out)
    return wrap(out)
  })
  const OVERLOADED = new Set(['sort', 'argsort', 'max', 'min', 'sum', 'mean', 'std', 'var', 'median', 'mode', 'prod', 'argmax', 'argmin', 'all', 'any', 'cumsum', 'cumprod'])
  const NUMBERS_FIRST = new Set(['add', 'sub', 'subtract', 'multiply', 'div', 'divide', 'true_divide', 'floor_divide', 'remainder', 'fmod', 'atan2', 'pow', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'copysign', 'xlogy'])
  const viaMethod = (name: string, method = name) => fnv(name, (args, kw) => {
    let self = args[0] ?? kw.input
    if (self && !asTT(self) && NUMBERS_FIRST.has(name) && isNum(self)) self = wrap(TT.scalar(h.num(self), self.k === 'float' ? C.DEFAULTS.float : self.k === 'bool' ? 'bool' : 'int64'))
    if (!self) throw h.err('TypeError', `${name}() missing 1 required positional arguments: "input"`)
    if (!asTT(self)) throw h.err('TypeError', `${name}(): argument 'input' (position 1) must be Tensor, not ${typeName(self)}`)
    const { input: _input, ...rest } = kw
    if (rest.out && rest.out.k !== 'none') throw h.err('NotImplementedError', `${name}(): the out= argument is not available in this sandbox`)
    delete rest.out
    try {
      return h.call(h.getAttr(self, method), args.slice(1), rest)
    } catch (e) {
      // the method saw no tensor among its arguments; the function was given one
      const head = `${name}() received an invalid combination of arguments - got (`
      if (e instanceof PyError && e.message.startsWith(head)) {
        const rest0 = e.message.slice(head.length)
        const given = rest0.slice(0, rest0.indexOf('), but expected')).replace(/, $/, '')
        throw h.err('TypeError', `${head}Tensor${given ? `, ${given}` : ''}${rest0.slice(rest0.indexOf('), but expected'))}`)
      }
      if (e instanceof PyError && OVERLOADED.has(name) && e.message.startsWith(`${name}() got an unexpected keyword argument`)) throw invalidArgs(h, name, [self, ...args.slice(1)], rest)
      throw e
    }
  })
  const twoArg = (name: string, op: string) => fnv(name, (args) => {
    const self = args[0]
    if (self?.k !== 'obj' || !(self.o instanceof TensorObj)) throw h.err('TypeError', `${name}(): argument 'input' (position 1) must be Tensor, not ${self ? typeName(self) : 'NoneType'}`)
    if (args[1] && (args[1].k === 'str' || args[1].k === 'none')) throw h.err('TypeError', `${name}(): argument 'other' (position 2) must be Tensor, not ${typeName(args[1])}`)
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
  const isNone = (v: Value | undefined) => !v || v.k === 'none'
  const isFalse = (v: Value) => v.k === 'bool' && !v.v
  /** never silently wrong: a value this sandbox cannot honour says so */
  const refuse = (who: string, a: Args, key: string, ok: (v: Value) => boolean) => {
    const v = a[key]
    if (v && !ok(v)) throw h.err('NotImplementedError', `${who}: ${key}=${repr(v)} is not available in this sandbox`)
  }
  /** an nn class, from its real signature; device and dtype are handled here, and the arguments become attributes */
  const layer = (name: string, params: string, make: (a: Args) => Layer, attrs?: Map<string, Value>): Value => ({
    k: 'type', name, attrs, call: (args, kw) => {
      const { device, dtype, ...rest } = kw
      onlyCpu(h, device)
      const a = bindPy(h, `${name}.__init__`, params, args, rest)
      const o = make(a)
      o.fields = a
      const m = py.obj(o)
      const d = dtypeArg(h, dtype)
      if (d) castFloats(m, d)
      return m
    },
  })
  const reduction = (a: Args): C.Reduction => {
    if (!isNone(a.size_average) || !isNone(a.reduce)) return !bool(a.reduce, true) ? 'none' : bool(a.size_average, true) ? 'mean' : 'sum'
    const v = a.reduction
    if (!v || v.k === 'none') return 'mean'
    const r = h.str(v)
    if (r !== 'mean' && r !== 'sum' && r !== 'none') throw h.err('ValueError', `${r} is not a valid value for reduction`)
    return r
  }
  const optT = (v: Value | undefined, what = 'weight'): TT | null => (v && v.k !== 'none' ? tensorArg(h, v, what) : null)
  const ceOpts = (a: Args): CeOpts => ({ weight: optT(a.weight), ignore: a.ignore_index ? h.toInt(a.ignore_index) : -100, smoothing: num(a.label_smoothing, 0), red: reduction(a) })

  /** an int or a tuple, n numbers long: _single, _pair, _triple */
  const ntuple = (v: Value | undefined, n: number, d: number): number[] => {
    if (!v || v.k === 'none') return new Array<number>(n).fill(d)
    if (v.k === 'tuple' || v.k === 'list' || (v.k === 'obj' && v.o instanceof SizeObj)) {
      const xs = h.iterate(v).map((x) => h.toInt(x))
      return xs.length === 1 ? new Array<number>(n).fill(xs[0]) : xs
    }
    return new Array<number>(n).fill(h.toInt(v))
  }
  const paddingModes: Record<string, C.PadMode> = { zeros: 'constant', reflect: 'reflect', replicate: 'replicate', circular: 'circular' }
  const padModeOf = (v: Value | undefined): C.PadMode => {
    const s = v ? h.str(v) : 'zeros'
    if (!paddingModes[s]) throw h.err('ValueError', `padding_mode must be one of {'circular', 'zeros', 'replicate', 'reflect'}, but got padding_mode='${s}'`)
    return paddingModes[s]
  }
  const convOpts = (a: Args, n: number): ConvOpts => {
    const stride = ntuple(a.stride, n, 1)
    const pad = a.padding
    if (pad?.k === 'str' && pad.v !== 'same' && pad.v !== 'valid') throw h.err('ValueError', `Invalid padding string '${pad.v}', should be one of {'valid', 'same'}`)
    if (pad?.k === 'str' && pad.v === 'same' && stride.some((x) => x !== 1)) throw h.err('ValueError', "padding='same' is not supported for strided convolutions")
    return {
      stride, dilation: ntuple(a.dilation, n, 1), groups: a.groups ? h.toInt(a.groups) : 1, paddingMode: padModeOf(a.padding_mode),
      padding: pad?.k === 'str' ? (pad.v as 'same' | 'valid') : ntuple(pad, n, 0),
    }
  }
  const convClass = (name: string, n: number): Value => layer(name, "in_channels out_channels kernel_size stride=1 padding=0 dilation=1 groups=1 bias=True padding_mode='zeros'", (a) => {
    const [inC, outC] = [h.toInt(a.in_channels!), h.toInt(a.out_channels!)]
    const o = convOpts(a, n)
    if (o.groups <= 0) throw h.err('ValueError', 'groups must be a positive integer')
    if (inC % o.groups) throw h.err('ValueError', 'in_channels must be divisible by groups')
    if (outC % o.groups) throw h.err('ValueError', 'out_channels must be divisible by groups')
    return new ConvLayer(h, name, n, false, inC, outC, ntuple(a.kernel_size, n, 1), o, new Array<number>(n).fill(0), bool(a.bias, true))
  })
  const convTClass = (name: string, n: number): Value => layer(name, "in_channels out_channels kernel_size stride=1 padding=0 output_padding=0 groups=1 bias=True dilation=1 padding_mode='zeros'", (a) => {
    if (h.str(a.padding_mode!) !== 'zeros') throw h.err('ValueError', `Only "zeros" padding mode is supported for ${name}`)
    const [inC, outC] = [h.toInt(a.in_channels!), h.toInt(a.out_channels!)]
    const groups = h.toInt(a.groups!)
    if (groups <= 0) throw h.err('ValueError', 'groups must be a positive integer')
    if (inC % groups) throw h.err('ValueError', 'in_channels must be divisible by groups')
    if (outC % groups) throw h.err('ValueError', 'out_channels must be divisible by groups')
    const o: ConvOpts = { stride: ntuple(a.stride, n, 1), padding: ntuple(a.padding, n, 0), dilation: ntuple(a.dilation, n, 1), groups, paddingMode: 'constant' }
    return new ConvLayer(h, name, n, true, inC, outC, ntuple(a.kernel_size, n, 1), o, ntuple(a.output_padding, n, 0), bool(a.bias, true))
  })
  const poolOpts = (a: Args, n: number): C.PoolOpts => ({
    k: ntuple(a.kernel_size, n, 2), s: ntuple(isNone(a.stride) ? a.kernel_size : a.stride, n, 1), p: ntuple(a.padding, n, 0), d: ntuple(a.dilation, n, 1),
    ceil: bool(a.ceil_mode, false), countPad: bool(a.count_include_pad, true), divisor: isNone(a.divisor_override) ? null : h.num(a.divisor_override!),
  })
  const MAXPOOL = 'kernel_size stride=None padding=0 dilation=1 return_indices=False ceil_mode=False'
  const AVGPOOL = 'kernel_size stride=None padding=0 ceil_mode=False count_include_pad=True divisor_override=None'
  const maxPoolClass = (name: string, n: number): Value => layer(name, MAXPOOL, (a) => {
    const stride = isNone(a.stride) ? a.kernel_size! : a.stride!
    return new PoolLayer(name, 'max', n, poolOpts(a, n), bool(a.return_indices, false), `kernel_size=${repr(a.kernel_size!)}, stride=${repr(stride)}, padding=${repr(a.padding!)}, dilation=${repr(a.dilation!)}, ceil_mode=${repr(a.ceil_mode!)}`)
  })
  const avgPoolClass = (name: string, n: number): Value => layer(name, AVGPOOL, (a) => {
    const stride = isNone(a.stride) ? a.kernel_size! : a.stride!
    const tupled = (v: Value) => (n === 1 ? dimsStr(ntuple(v, 1, 0)) : repr(v))
    return new PoolLayer(name, 'avg', n, poolOpts(a, n), false, `kernel_size=${tupled(a.kernel_size!)}, stride=${tupled(stride)}, padding=${tupled(a.padding!)}`)
  })
  /** F.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices) */
  const maxPoolF = (name: string, n: number, always = false) => fnv(name, (args, kw) => {
    const a = bindPy(h, name, 'input kernel_size stride=None padding=0 dilation=1 ceil_mode=False return_indices=False', args, kw)
    const t = tensorArg(h, a.input, `max_pool${n}d`)
    const out = runPool(h, t, 'max', n, poolOpts(a, n), always || bool(a.return_indices, false))
    const ot = asTT(out.k === 'tuple' ? out.items[0] : out)
    if (ot) flowStep(h, `F.${name}`, t, ot)
    return out
  })
  const avgPoolF = (name: string, n: number) => fnv(name, (args, kw) => {
    const a = bindPy(h, name, 'input kernel_size stride=None padding=0 ceil_mode=False count_include_pad=True divisor_override=None', args, kw)
    const t = tensorArg(h, a.input, `avg_pool${n}d`)
    const out = runPool(h, t, 'avg', n, poolOpts(a, n), false)
    flowStep(h, `F.${name}`, t, asTT(out)!)
    return out
  })

  /** adaptive pool output size: an int, or a pair where None keeps the input's size */
  const adaptiveSize = (v: Value | undefined, x: TT, one: boolean): [number, number] => {
    const hw: [number, number] = [x.shape[C.normDim(-2, x.ndim)], x.shape[x.ndim - 1]]
    if (one) return [1, h.toInt(v!.k === 'tuple' ? v!.items[0] : v!)]
    if (v?.k === 'tuple' || v?.k === 'list') return [v.items[0].k === 'none' ? hw[0] : h.toInt(v.items[0]), v.items[1].k === 'none' ? hw[1] : h.toInt(v.items[1])]
    return [h.toInt(v!), h.toInt(v!)]
  }
  const adaptive = (x: TT, size: Value | undefined, mode: 'max' | 'avg', one: boolean, withIndices = false): Value => {
    const r = one ? C.adaptivePool2d(E, C.unsqueezeT(E, x, -2), adaptiveSize(size, x, true), mode) : C.adaptivePool2d(E, x, adaptiveSize(size, x, false), mode)
    const out = one ? C.squeezeT(E, r.out, -2) : r.out
    const ind = one ? C.squeezeT({ grad: false } as Engine, r.indices, -2) : r.indices
    return withIndices ? py.tuple([wrap(out), wrap(ind)]) : wrap(out)
  }

  /** adaptive pooling over the last three dimensions */
  const adaptive3 = (x: TT, size: Value, mode: 'max' | 'avg', withIndices: boolean): Value => {
    C.normDim(-3, x.ndim)
    if (x.ndim !== 4 && x.ndim !== 5) throw h.err('RuntimeError', `adaptive_${mode}_pool3d(): Expected 4D or 5D tensor, but got: [${x.shape.join(', ')}]`)
    const given = size.k === 'tuple' || size.k === 'list' ? size.items : [size, size, size]
    const r = C.adaptivePool3d(E, x, given.map((v, k) => (v.k === 'none' ? x.shape[x.ndim - 3 + k] : h.toInt(v))), mode)
    return withIndices ? py.tuple([wrap(r.out), wrap(r.indices)]) : wrap(r.out)
  }

  const interpolateModes = 'nearest | linear | bilinear | bicubic | trilinear | lanczos | area | nearest-exact'
  /** the interpolation shared by F.interpolate and nn.Upsample */
  const interpolate = (x: TT, size: Value | undefined, scale: Value | undefined, mode: string, align: Value | undefined): TT => {
    const spatial = x.ndim - 2
    const hasSize = !!size && size.k !== 'none'
    const hasScale = !!scale && scale.k !== 'none'
    if (spatial < 1 || spatial > 3) throw h.err('NotImplementedError', `Input Error: Only 3D, 4D and 5D input Tensors supported (got ${x.ndim}D) for the modes: ${interpolateModes} (got ${mode})`)
    if (hasSize && hasScale) throw h.err('ValueError', 'only one of size or scale_factor should be defined')
    if (!hasSize && !hasScale) throw h.err('ValueError', 'either size or scale_factor should be defined')
    const list = (v: Value) => (v.k === 'tuple' || v.k === 'list' ? v.items.map((i) => h.num(i)) : new Array<number>(spatial).fill(h.num(v)))
    const given = list((hasSize ? size : scale)!)
    if (given.length !== spatial) throw h.err('ValueError', hasSize ? `Input and output must have the same number of spatial dimensions, but got input with spatial dimensions of [${x.shape.slice(2).join(', ')}] and output size of [${given.join(', ')}]. Please provide input tensor in (N, C, d1, d2, ...,dK) format and output size in (o1, o2, ...,oK) format.` : `scale_factor shape must match input shape. Input is ${spatial}D, scale_factor size is torch.Size([${given.length}])`)
    const inSizes = x.shape.slice(2)
    const outSizes = hasSize ? given : inSizes.map((n, k) => Math.floor(n * given[k]))
    // the source position of output cell i: in / out, or 1 / scale_factor when that was what the caller gave
    const ratio = inSizes.map((n, k) => Math.fround(hasSize ? n / outSizes[k] : 1 / given[k]))
    const interpolating = mode === 'linear' || mode === 'bilinear' || mode === 'bicubic' || mode === 'trilinear'
    if (align && align.k !== 'none' && !interpolating) throw h.err('ValueError', 'align_corners option can only be set with the interpolating modes: linear | bilinear | bicubic | trilinear')
    if (mode === 'nearest' || mode === 'nearest-exact') {
      const offset = mode === 'nearest' ? 0 : 0.5
      const maps = [[...Array(x.shape[0]).keys()], [...Array(x.shape[1]).keys()], ...outSizes.map((m, k) => Array.from({ length: m }, (_, i) => Math.min(Math.floor(Math.fround((i + offset) * ratio[k])), inSizes[k] - 1)))]
      const name = `Upsample${mode === 'nearest' ? 'Nearest' : 'NearestExact'}${spatial}DBackward0`
      return C.takeT(E, x, C.crossSrc(x.shape, maps), maps.map((m) => m.length), name)
    }
    if (mode === 'area') {
      if (spatial > 2) throw h.err('NotImplementedError', "mode='area' on 5D input is not available in this sandbox")
      const r = spatial === 1 ? asTT(adaptive(x, py.int(outSizes[0]), 'avg', true))! : asTT(adaptive(x, py.tuple(outSizes.map(py.int)), 'avg', false))!
      return r
    }
    if (!interpolating) throw h.err('NotImplementedError', `Input Error: Only 3D, 4D and 5D input Tensors supported (got ${x.ndim}D) for the modes: ${interpolateModes} (got ${mode})`)
    const wants = mode === 'linear' ? 1 : mode === 'trilinear' ? 3 : 2
    if (wants !== spatial) throw h.err('NotImplementedError', `Got ${x.ndim}D input, but ${mode} mode needs ${wants + 2}D input`)
    const corners = bool(align, false)
    let out = x
    outSizes.forEach((m, k) => {
      const n = inSizes[k]
      const rawPos = (i: number) => (corners ? (m > 1 ? (i * (n - 1)) / (m - 1) : 0) : (i + 0.5) * ratio[k] - 0.5)
      if (mode === 'bicubic') {
        const A = -0.75
        const cub = (t: number) => {
          const x2 = 1 - t
          return [((A * (t + 1) - 5 * A) * (t + 1) + 8 * A) * (t + 1) - 4 * A, ((A + 2) * t - (A + 3)) * t * t + 1, ((A + 2) * x2 - (A + 3)) * x2 * x2 + 1, ((A * (x2 + 1) - 5 * A) * (x2 + 1) + 8 * A) * (x2 + 1) - 4 * A]
        }
        const clampI = (i: number) => Math.max(0, Math.min(i, n - 1))
        const idx = Array.from({ length: m }, (_, i) => [-1, 0, 1, 2].map((d) => clampI(Math.floor(rawPos(i)) + d)))
        const wt = Array.from({ length: m }, (_, i) => cub(rawPos(i) - Math.floor(rawPos(i))))
        out = C.weightedAxis(E, out, k + 2, idx, wt)
      } else {
        const pos = Array.from({ length: m }, (_, i) => Math.max(0, rawPos(i)))
        const i0 = pos.map((q) => Math.min(Math.floor(q), n - 1))
        out = C.weightedAxis(E, out, k + 2, i0.map((i) => [i, Math.min(i + 1, n - 1)]), pos.map((q, i) => [1 - (q - i0[i]), q - i0[i]]))
      }
    })
    if (out.fn) out.fn.name = `Upsample${mode === 'trilinear' ? 'Trilinear3D' : mode === 'bilinear' ? 'Bilinear2D' : mode === 'bicubic' ? 'Bicubic2D' : 'Linear1D'}Backward0`
    return out
  }

  /** recompute_scale_factor=True: the size the factor gives, then as if that size had been asked for */
  const recomputed = (x: TT, size: Value | undefined, scale: Value | undefined, again: Value | undefined): [Value | undefined, Value | undefined] => {
    if (!again || !h.truthy(again)) return [size, scale]
    if (!isNone(size)) throw h.err('ValueError', 'recompute_scale_factor is not meaningful with an explicit size.')
    const factors = scale!.k === 'tuple' || scale!.k === 'list' ? scale!.items.map((v) => h.num(v)) : new Array<number>(x.ndim - 2).fill(h.num(scale!))
    return [py.tuple(x.shape.slice(2).map((n, k) => py.int(Math.floor(n * factors[k])))), undefined]
  }
  const padNamed = (x: TT, pad: number[], mode: C.PadMode, value: number): TT => {
    const out = C.padT(E, x, pad, mode, value)
    if (out.fn) out.fn.name = mode === 'constant' ? 'ConstantPadNdBackward0' : mode === 'circular' ? 'CopySlices' : `${mode === 'reflect' ? 'Reflection' : 'Replication'}Pad${pad.length / 2}DBackward0`
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
    return padNamed(x, pad, mode as C.PadMode, num(valueV, 0))
  }

  // ---- activation functions, shared by nn.X and F.x: [class, function, parameters, make, extra_repr]
  const sig = (v: number) => 1 / (1 + Math.exp(-v))
  const softplusAt = (beta: number, threshold: number) => (v: number) => (v * beta > threshold ? v : Math.log1p(Math.exp(v * beta)) / beta)
  const hardtanh = (lo: number, hi: number) => (x: TT) => C.mapT(E, x, (v) => Math.min(hi, Math.max(lo, v)), (v) => (v <= lo || v >= hi ? 0 : 1), 'HardtanhBackward0')
  const SELU_ALPHA = 1.6732632423543772
  const SELU_SCALE = 1.0507009873554805
  const inplaceNote = (a: Args) => (bool(a.inplace, false) ? 'inplace=True' : '')
  const withInplace = (a: Args, text: string) => `${text}${bool(a.inplace, false) ? ', inplace=True' : ''}`
  const quiet = () => ''
  const ACTS: [string, string, string, (a: Args) => (x: TT) => TT, (a: Args) => string][] = [
    ['ReLU', 'relu', 'inplace=False', () => (x) => C.unary(E, 'relu', x), inplaceNote],
    ['ReLU6', 'relu6', 'inplace=False', () => hardtanh(0, 6), inplaceNote],
    ['LeakyReLU', 'leaky_relu', 'negative_slope=0.01 inplace=False', (a) => {
      const s = h.num(a.negative_slope!)
      return (x) => C.mapT(E, x, (v) => (v >= 0 ? v : s * v), (v) => (v > 0 ? 1 : s), 'LeakyReluBackward0')
    }, (a) => withInplace(a, `negative_slope=${repr(a.negative_slope!)}`)],
    ['ELU', 'elu', 'alpha=1.0 inplace=False', (a) => {
      const al = h.num(a.alpha!)
      return (x) => C.mapT(E, x, (v) => (v > 0 ? v : al * Math.expm1(v)), (v) => (v > 0 ? 1 : al * Math.exp(v)), 'EluBackward0')
    }, (a) => withInplace(a, `alpha=${repr(a.alpha!)}`)],
    ['CELU', 'celu', 'alpha=1.0 inplace=False', (a) => {
      const al = h.num(a.alpha!)
      return (x) => C.mapT(E, x, (v) => (v > 0 ? v : al * Math.expm1(v / al)), (v) => (v > 0 ? 1 : Math.exp(v / al)), 'CeluBackward0')
    }, (a) => withInplace(a, `alpha=${repr(a.alpha!)}`)],
    ['SELU', 'selu', 'inplace=False', () => (x) => C.mapT(E, x, (v) => SELU_SCALE * (v > 0 ? v : SELU_ALPHA * Math.expm1(v)), (v) => SELU_SCALE * (v > 0 ? 1 : SELU_ALPHA * Math.exp(v)), 'EluBackward0'), inplaceNote],
    ['GELU', 'gelu', "approximate='none'", (a) => {
      const how = h.str(a.approximate!)
      if (how !== 'none' && how !== 'tanh') throw h.err('RuntimeError', 'approximate argument must be either none or tanh.')
      const c = Math.sqrt(2 / Math.PI)
      return how === 'none'
        ? (x) => C.mapT(E, x, (v) => v * 0.5 * (1 + C.erf(v / Math.SQRT2)), (v) => 0.5 * (1 + C.erf(v / Math.SQRT2)) + v * Math.exp(-0.5 * v * v) / Math.sqrt(2 * Math.PI), 'GeluBackward0')
        : (x) => C.mapT(E, x, (v) => 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v ** 3))), (v) => {
          const inner = c * (v + 0.044715 * v ** 3)
          return 0.5 * (1 + Math.tanh(inner)) + 0.5 * v * (c * (1 + 3 * 0.044715 * v * v)) / Math.cosh(inner) ** 2
        }, 'GeluBackward0')
    }, (a) => `approximate=${repr(a.approximate!)}`],
    ['SiLU', 'silu', 'inplace=False', () => (x) => C.mapT(E, x, (v) => v * sig(v), (v) => sig(v) + v * sig(v) * (1 - sig(v)), 'SiluBackward0'), inplaceNote],
    ['Mish', 'mish', 'inplace=False', () => (x) => {
      const sp = (v: number) => Math.log1p(Math.exp(v))
      return C.mapT(E, x, (v) => v * Math.tanh(sp(v)), (v) => Math.tanh(sp(v)) + v * sig(v) / Math.cosh(sp(v)) ** 2, 'MishBackward0')
    }, inplaceNote],
    ['Softplus', 'softplus', 'beta=1.0 threshold=20.0', (a) => {
      const [b, th] = [h.num(a.beta!), h.num(a.threshold!)]
      return (x) => C.mapT(E, x, softplusAt(b, th), (v) => (v * b > th ? 1 : sig(v * b)), 'SoftplusBackward0')
    }, (a) => `beta=${repr(py.float(h.num(a.beta!)))}, threshold=${repr(py.float(h.num(a.threshold!)))}`],
    ['Hardtanh', 'hardtanh', 'min_val=-1.0 max_val=1.0 inplace=False', (a) => hardtanh(h.num(a.min_val!), h.num(a.max_val!)), (a) => withInplace(a, `min_val=${repr(a.min_val!)}, max_val=${repr(a.max_val!)}`)],
    ['Hardsigmoid', 'hardsigmoid', 'inplace=False', () => (x) => C.mapT(E, x, (v) => Math.min(1, Math.max(0, v / 6 + 0.5)), (v) => (v > -3 && v < 3 ? 1 / 6 : 0), 'HardsigmoidBackward0'), inplaceNote],
    ['Hardswish', 'hardswish', 'inplace=False', () => (x) => C.mapT(E, x, (v) => (v <= -3 ? 0 : v >= 3 ? v : (v * (v + 3)) / 6), (v) => (v <= -3 ? 0 : v >= 3 ? 1 : (2 * v + 3) / 6), 'HardswishBackward0'), inplaceNote],
    ['Hardshrink', 'hardshrink', 'lambd=0.5', (a) => {
      const l = h.num(a.lambd!)
      return (x) => C.mapT(E, x, (v) => (Math.abs(v) > l ? v : 0), (v) => (Math.abs(v) > l ? 1 : 0), 'HardshrinkBackward0')
    }, (a) => repr(a.lambd!)],
    ['Softshrink', 'softshrink', 'lambd=0.5', (a) => {
      const l = h.num(a.lambd!)
      return (x) => C.mapT(E, x, (v) => (v > l ? v - l : v < -l ? v + l : v * 0), (v) => (Math.abs(v) > l ? 1 : 0), 'SoftshrinkBackward0')
    }, (a) => repr(a.lambd!)],
    ['Tanhshrink', 'tanhshrink', '', () => (x) => C.mapT(E, x, (v) => v - Math.tanh(v), (v) => Math.tanh(v) ** 2, 'SubBackward0'), quiet],
    ['Threshold', 'threshold', 'threshold value inplace=False', (a) => {
      const [t, val] = [h.num(a.threshold!), h.num(a.value!)]
      return (x) => C.mapT(E, x, (v) => (v > t ? v : val), (v) => (v > t ? 1 : 0), 'ThresholdBackward0')
    }, (a) => withInplace(a, `threshold=${repr(a.threshold!)}, value=${repr(a.value!)}`)],
    ['LogSigmoid', 'logsigmoid', '', () => (x) => C.mapT(E, x, (v) => -softplusAt(1, Infinity)(-v), (v) => sig(-v), 'LogSigmoidBackward0'), quiet],
    ['Softsign', 'softsign', '', () => (x) => C.mapT(E, x, (v) => v / (1 + Math.abs(v)), (v) => 1 / (1 + Math.abs(v)) ** 2, 'DivBackward0'), quiet],
    ['Sigmoid', 'sigmoid', '', () => (x) => C.unary(E, 'sigmoid', x), quiet],
    ['Tanh', 'tanh', '', () => (x) => C.unary(E, 'tanh', x), quiet],
  ]
  /** softmax without a dim: PyTorch's implicit choice */
  const softDim = (x: TT, d: Value | undefined) => (d && d.k !== 'none' ? h.toInt(d) : x.ndim === 0 || x.ndim === 1 || x.ndim === 3 ? 0 : 1)
  const dimRepr = (a: Args) => `dim=${isNone(a.dim) ? 'None' : h.toInt(a.dim!)}`

  const Unflatten = layer('Unflatten', 'dim unflattened_size', (a) => {
    const d = h.toInt(a.dim!)
    const sizesV = toDims(h, [a.unflattened_size!])
    return new Activation('Unflatten', (x) => unflatten(x, d, sizesV), `dim=${d}, unflattened_size=${repr(a.unflattened_size!)}`)
  })
  const unflatten = (x: TT, dim: number, sizes0: number[]): TT => {
    const axis = C.normDim(dim, x.ndim)
    const known = sizes0.reduce((p, q) => (q === -1 ? p : p * q), 1)
    if (sizes0.includes(-1) ? x.shape[axis] % known !== 0 : known !== x.shape[axis]) throw h.err('RuntimeError', `unflatten: Provided sizes [${sizes0.join(', ')}] don't multiply up to the size of dim ${axis} (${x.shape[axis]}) in the input tensor`)
    const out = C.reshapeT(E, x, [...x.shape.slice(0, axis), ...sizes0, ...x.shape.slice(axis + 1)], false)
    if (out.fn) out.fn.name = 'UnflattenBackward0'
    return out
  }

  const padLayer = (kind: string, n: number, mode: C.PadMode, withValue: boolean): Value => layer(kind, withValue ? 'padding value' : 'padding', (a) => {
    const pad = a.padding?.k === 'tuple' || a.padding?.k === 'list' ? a.padding.items.map((v) => h.toInt(v)) : new Array<number>(2 * n).fill(h.toInt(a.padding!))
    const value = withValue ? num(a.value, 0) : 0
    return new Activation(kind, (x) => padNamed(x, pad, mode, value), withValue ? `padding=(${pad.join(', ')}), value=${repr(a.value!)}` : `(${pad.join(', ')})`)
  })

  const moduleHost: HostClass = {
    init(inst) {
      inst.state[MODULE_STATE] = { training: true, buffers: new Map<string, boolean>(), pre: [], post: [] }
    },
    setAttr(inst, name, v, hh) {
      const st = inst.state[MODULE_STATE] as { buffers: Map<string, boolean> } | undefined
      const t = asTT(v)
      if (st) {
        // nn.Buffer: assigning one registers it
        if (v.k === 'obj' && v.o instanceof BufferMark) {
          st.buffers.set(name, v.o.persistent)
          inst.attrs.set(name, wrap(v.o.t))
          return
        }
        return
      }
      if (moduleOf(v)) throw hh.err('AttributeError', 'cannot assign module before Module.__init__() call')
      if (t && t.isParam) throw hh.err('AttributeError', 'cannot assign parameters before Module.__init__() call')
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
      const fwd = (() => {
        try {
          return hh.getAttr(inst, 'forward')
        } catch (e) {
          if (e instanceof PyError) return py.NONE
          throw e
        }
      })()
      if (fwd.k !== 'bound') throw hh.err('NotImplementedError', `Module [${inst.cls.name}] is missing the required "forward" function`)
      return callModule(hh, inst, args, (a) => hh.call(fwd, a, kw), null, asTT(args[0] ?? py.NONE))
    },
    repr(inst) {
      return inst.state[MODULE_STATE] ? moduleRepr(inst) : undefined
    },
  }
  const Module: Value = { k: 'type', name: 'Module', host: moduleHost, call: () => { throw h.err('TypeError', 'nn.Module is a base class: define class Net(nn.Module) with __init__ and forward') } }

  const lossLayer = (name: string, params: string, f: (a: Args) => (x: TT, y: TT, hh: Host, more: TT[]) => TT, fwd?: string[]) => layer(name, params, (a) => new LossLayer(name, f(a), fwd))
  const LEGACY_LOSS = "size_average=None reduce=None reduction='mean'"

  const reduced = (per: TT, red: C.Reduction): TT => (red === 'none' ? per : C.reduceSum(E, per, null, false, red === 'mean'))
  const named = (t: TT, name: string): TT => {
    if (t.fn) t.fn.name = name
    return t
  }
  const one = (x: TT, v: number) => TT.scalar(v, x.dtype)
  const mulT = (a: TT, b: TT) => C.binary(E, '*', a, b)
  const subT = (a: TT, b: TT) => C.binary(E, '-', a, b)
  const addT = (a: TT, b: TT) => C.binary(E, '+', a, b)
  const clamp0 = (x: TT) => C.clamp(E, x, 0, null)
  // xlogy(y, y) - y * log(x) with the same rounding on both sides, so equal distributions give exactly 0
  const xlogy = (y: TT, x: TT) => {
    const R = y.dtype === 'float64' ? (v: number) => v : Math.fround
    return C.map2T(E, y, x, (a, b) => (a === 0 ? 0 : a * R(Math.log(b))), (a, b) => (a === 0 ? 0 : Math.log(b)), (a, b) => a / b, y.dtype, 'XlogyBackward0')
  }
  const klDiv = (x: TT, y: TT, red: string, logTarget: boolean): TT => {
    if (red !== 'mean' && red !== 'sum' && red !== 'none' && red !== 'batchmean') throw h.err('ValueError', `${red} is not a valid value for reduction`)
    const per = logTarget ? mulT(C.unary(E, 'exp', y), subT(y, x)) : subT(xlogy(y, y), mulT(y, x))
    if (red === 'none') return per
    if (red === 'batchmean') return C.binary(E, '/', C.reduceSum(E, per, null, false, false), TT.scalar(x.ndim ? x.shape[0] : 1, 'int64'), false, true)
    return C.reduceSum(E, per, null, false, red === 'mean')
  }
  const poissonNll = (x: TT, y: TT, logInput: boolean, full: boolean, eps: number, red: C.Reduction): TT => {
    let per = logInput ? subT(C.unary(E, 'exp', x), mulT(y, x)) : subT(x, mulT(y, C.unary(E, 'log', addT(x, one(x, eps)))))
    if (full) {
      const yv = y.values()
      per = addT(per, TT.of(yv.map((t) => (t > 1 ? t * Math.log(t) - t + 0.5 * Math.log(2 * Math.PI * t) : 0)), y.shape, y.dtype))
    }
    return reduced(per, red)
  }
  const gaussianNll = (x: TT, y: TT, v0: TT, full: boolean, eps: number, red: C.Reduction): TT => {
    if (v0.size !== x.size && v0.shape.join() !== x.shape.join() && !(v0.ndim === x.ndim && v0.shape.every((d, k) => d === x.shape[k] || d === 1)) && !(v0.ndim === x.ndim - 1 && v0.shape.every((d, k) => d === x.shape[k]))) throw h.err('ValueError', 'var is of incorrect size')
    if (v0.values().some((q) => q < 0)) throw h.err('ValueError', 'var has negative entry/entries')
    const vv = v0.ndim === x.ndim - 1 ? C.unsqueezeT(E, v0, -1) : v0
    // the clamp is not differentiated: a variance below eps is treated as a constant eps
    const clampedVals = vv.values().map((q) => Math.max(q, eps))
    const scale = TT.of(clampedVals.map((q, i) => q / (vv.values()[i] || 1)), vv.shape, vv.dtype)
    const vc = mulT(vv, scale)
    let per = C.binary(E, '*', addT(C.unary(E, 'log', vc), C.binary(E, '/', C.unary(E, 'square', subT(x, y)), vc)), TT.scalar(0.5, x.dtype), false, true)
    if (full) per = addT(per, TT.scalar(0.5 * Math.log(2 * Math.PI), x.dtype))
    return reduced(per, red)
  }
  const marginRanking = (x1: TT, x2: TT, y: TT, m: number, red: C.Reduction) => reduced(clamp0(addT(mulT(C.unary(E, 'neg', y), subT(x1, x2)), one(x1, m))), red)
  const hingeEmbedding = (x: TT, y: TT, m: number, red: C.Reduction): TT => {
    const pos = C.whereT(E, C.compare(y, one(y, 1), '=='), x, one(x, 0), x.dtype)
    const neg = C.whereT(E, C.compare(y, one(y, -1), '=='), clamp0(subT(one(x, m), x)), one(x, 0), x.dtype)
    return reduced(addT(pos, neg), red)
  }
  const cosineEmbedding = (x1: TT, x2: TT, y: TT, m: number, red: C.Reduction): TT => {
    const dim = x1.ndim === 1 ? 0 : 1
    const EPS = 1e-12
    const prodSum = C.reduceSum(E, mulT(x1, x2), [dim], false, false)
    const mag = (t: TT) => addT(C.reduceSum(E, mulT(t, t), [dim], false, false), one(t, EPS))
    const cos = C.binary(E, '/', prodSum, C.unary(E, 'sqrt', mulT(mag(x1), mag(x2))))
    const yy = y.ndim > cos.ndim ? y : y
    const pos = C.whereT(E, C.compare(yy, TT.scalar(1, yy.dtype), '=='), subT(one(cos, 1), cos), one(cos, 0), cos.dtype)
    const neg = C.whereT(E, C.compare(yy, TT.scalar(-1, yy.dtype), '=='), clamp0(subT(cos, one(cos, m))), one(cos, 0), cos.dtype)
    return reduced(addT(pos, neg), red)
  }
  const pairDistance = (x1: TT, x2: TT, p: number, eps: number, keep: boolean): TT => C.vectorNorm(E, addT(subT(x1, x2), TT.scalar(eps, x1.dtype)), p, [-1], keep)
  const tripletMargin = (a: TT, p: TT, n: TT, m: number, q: number, eps: number, swap: boolean, red: C.Reduction): TT => {
    const dPos = pairDistance(a, p, q, eps, false)
    let dNeg = pairDistance(a, n, q, eps, false)
    if (swap) dNeg = C.binary(E, 'minimum', dNeg, pairDistance(p, n, q, eps, false))
    return reduced(clamp0(addT(subT(dPos, dNeg), one(a, m))), red)
  }
  const softMargin = (x: TT, y: TT, red: C.Reduction) => named(reduced(C.mapT(E, mulT(C.unary(E, 'neg', y), x), (v) => Math.log1p(Math.exp(v)), (v) => 1 / (1 + Math.exp(-v)), 'SoftMarginLossBackward0'), red), 'SoftMarginLossBackward0')
  const multiMargin = (x: TT, y: TT, p: number, m: number, weight: TT | null, red: C.Reduction): TT => named(multiMargin0(x, y, p, m, weight, red), 'MultiMarginLossBackward0')
  const multiMargin0 = (x: TT, y: TT, p: number, m: number, weight: TT | null, red: C.Reduction): TT => {
    const unbatched = x.ndim === 1
    const xs = unbatched ? C.unsqueezeT(E, x, 0) : x
    const [N, Cl] = xs.shape
    const tv = y.values()
    const rows = Array.from({ length: N }, (_, n) => {
      const row = C.narrowT(E, C.reshapeT(E, C.narrowT(E, xs, 0, n, 1), [Cl], false), 0, 0, Cl)
      const t = tv[n]
      const yv = C.takeT(E, row, [t], [], 'SelectBackward0')
      let per = C.binary(E, '*', C.mapT(E, row, () => 0, null, 'ZeroBackward0'), one(row, 0))
      const terms = Array.from({ length: Cl }, (_, c) => c).filter((c) => c !== t).map((c) => {
        let term = clamp0(addT(subT(C.takeT(E, row, [c], [], 'SelectBackward0'), yv), one(row, m)))
        if (p !== 1) term = C.binary(E, '**', term, TT.scalar(p, 'int64'), false, true)
        if (weight) term = C.binary(E, '*', term, TT.scalar(weight.values()[t], x.dtype), false, true)
        return term
      })
      per = terms.reduce((s0, q) => addT(s0, q), C.reduceSum(E, per, null, false, false))
      return C.binary(E, '/', per, TT.scalar(Cl, 'int64'), false, true)
    })
    const per = C.catT(E, rows, 0, true)
    return reduced(unbatched ? C.squeezeT(E, per, 0) : per, red)
  }
  const multiLabelSoftMargin = (x: TT, y: TT, weight: TT | null, red: C.Reduction): TT => {
    const logsig = (t: TT) => C.mapT(E, t, (v) => -softplusAt(1, Infinity)(-v), (v) => sig(-v), 'LogSigmoidBackward0')
    let per = C.unary(E, 'neg', addT(mulT(y, logsig(x)), mulT(subT(one(y, 1), y), logsig(C.unary(E, 'neg', x)))))
    if (weight) per = mulT(per, weight)
    const m = C.reduceSum(E, per, [x.ndim - 1], false, true)
    return reduced(m, red)
  }
  const multiLabelMargin = (x: TT, y: TT, red: C.Reduction): TT => named(multiLabelMargin0(x, y, red), 'MultilabelMarginLossBackward0')
  const multiLabelMargin0 = (x: TT, y: TT, red: C.Reduction): TT => {
    const unbatched = x.ndim === 1
    const xs = unbatched ? C.unsqueezeT(E, x, 0) : x
    const ys = (unbatched ? C.unsqueezeT(E, y, 0) : y).values()
    const [N, Cl] = xs.shape
    const rows = Array.from({ length: N }, (_, n) => {
      const row = ys.slice(n * Cl, (n + 1) * Cl)
      const stop = row.indexOf(-1)
      const targets = stop < 0 ? row : row.slice(0, stop)
      const at = (c: number) => C.takeT(E, xs, [n * Cl + c], [], 'SelectBackward0')
      let sum = TT.scalar(0, x.dtype)
      for (const t of targets) for (let c = 0; c < Cl; c++) if (!targets.includes(c)) sum = addT(sum, clamp0(addT(subT(at(c), at(t)), one(x, 1))))
      return C.binary(E, '/', sum, TT.scalar(Cl, 'int64'), false, true)
    })
    const per = C.catT(E, rows, 0, true)
    return reduced(unbatched ? C.squeezeT(E, per, 0) : per, red)
  }

  /** nn.PixelShuffle: (…, C·r², H, W) → (…, C, H·r, W·r); PixelUnshuffle the reverse */
  const pixelShuffle = (x: TT, r: number): TT => {
    if (x.ndim < 3) throw h.err('RuntimeError', `pixel_shuffle expects input to have at least 3 dimensions, but got input with ${x.ndim} dimension(s)`)
    const [Ch, H, W] = x.shape.slice(-3)
    if (Ch % (r * r)) throw h.err('RuntimeError', `pixel_shuffle expects its input's 'channel' dimension to be divisible by the square of upscale_factor, but input.size(-3)=${Ch} is not divisible by ${r * r}`)
    const lead = x.shape.slice(0, -3)
    const n = lead.length
    const y = C.reshapeT(E, x, [...lead, Ch / (r * r), r, r, H, W], false)
    const z = C.permuteT(E, y, [...Array(n).keys(), n, n + 3, n + 1, n + 4, n + 2])
    return named(C.reshapeT(E, z, [...lead, Ch / (r * r), H * r, W * r], false), 'PixelShuffleBackward0')
  }
  const pixelUnshuffle = (x: TT, r: number): TT => {
    if (x.ndim < 3) throw h.err('RuntimeError', `pixel_unshuffle expects input to have at least 3 dimensions, but got input with ${x.ndim} dimension(s)`)
    const [Ch, H, W] = x.shape.slice(-3)
    if (H % r || W % r) throw h.err('RuntimeError', `pixel_unshuffle expects height to be divisible by downscale_factor, but input.size(-2)=${H} is not divisible by ${r}`)
    const lead = x.shape.slice(0, -3)
    const n = lead.length
    const y = C.reshapeT(E, x, [...lead, Ch, H / r, r, W / r, r], false)
    const z = C.permuteT(E, y, [...Array(n).keys(), n, n + 2, n + 4, n + 1, n + 3])
    return named(C.reshapeT(E, z, [...lead, Ch * r * r, H / r, W / r], false), 'PixelUnshuffleBackward0')
  }
  const channelShuffle = (x: TT, g: number): TT => {
    const Ch = x.shape[1]
    if (Ch % g) throw h.err('RuntimeError', `Number of channels must be divisible by groups. Got ${Ch} channels and ${g} groups.`)
    const y = C.reshapeT(E, x, [x.shape[0], g, Ch / g, ...x.shape.slice(2)], false)
    const perm = y.shape.map((_, k) => (k === 1 ? 2 : k === 2 ? 1 : k))
    return C.reshapeT(E, C.permuteT(E, y, perm), x.shape, false)
  }
  /** F.unfold: every sliding block of a (N, C, H, W) as one column */
  const unfold = (x0: TT, kV: Value, dV: Value | undefined, pV: Value | undefined, sV: Value | undefined): TT => {
    if (x0.ndim !== 3 && x0.ndim !== 4) throw h.err('NotImplementedError', `Input Error: Only 3D or 4D input Tensors are supported (got ${x0.ndim}D)`)
    const x = x0.ndim === 3 ? C.unsqueezeT(E, x0, 0) : x0
    const [k, d, p, st] = [ntuple(kV, 2, 1), ntuple(dV, 2, 1), ntuple(pV, 2, 0), ntuple(sV, 2, 1)]
    const [N, Ch, H, W] = x.shape
    const rows = Math.floor((H + 2 * p[0] - d[0] * (k[0] - 1) - 1) / st[0]) + 1
    const cols = Math.floor((W + 2 * p[1] - d[1] * (k[1] - 1) - 1) / st[1]) + 1
    const src: number[] = []
    for (let n = 0; n < N; n++) for (let c = 0; c < Ch; c++) for (let u = 0; u < k[0]; u++) for (let v = 0; v < k[1]; v++) {
      for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
        const [y, xx] = [i * st[0] + u * d[0] - p[0], j * st[1] + v * d[1] - p[1]]
        src.push(y >= 0 && y < H && xx >= 0 && xx < W ? (n * Ch + c) * H * W + y * W + xx : -1)
      }
    }
    const shape = [N, Ch * k[0] * k[1], rows * cols]
    const out = C.takeT(E, x, src, shape, 'UnfoldBackward0')
    return x0.ndim === 3 ? C.squeezeT(E, out, 0) : out
  }
  /** F.fold: sliding blocks back into a picture, overlaps added */
  const foldT = (x0: TT, sizeV: Value, kV: Value, dV: Value | undefined, pV: Value | undefined, sV: Value | undefined): TT => {
    const [size, k, d, p, st] = [ntuple(sizeV, 2, 1), ntuple(kV, 2, 1), ntuple(dV, 2, 1), ntuple(pV, 2, 0), ntuple(sV, 2, 1)]
    if (x0.ndim !== 2 && x0.ndim !== 3) throw h.err('NotImplementedError', `Input Error: Only unbatched (2D) or batched (3D) input Tensors are supported (got ${x0.ndim}D)`)
    const x = x0.ndim === 2 ? C.unsqueezeT(E, x0, 0) : x0
    const [N, chan, L] = x.shape
    const Ch = chan / (k[0] * k[1])
    const [H, W] = size
    const rows = Math.floor((H + 2 * p[0] - d[0] * (k[0] - 1) - 1) / st[0]) + 1
    const cols = Math.floor((W + 2 * p[1] - d[1] * (k[1] - 1) - 1) / st[1]) + 1
    if (rows * cols !== L) throw h.err('RuntimeError', `Given output_size=(${H}, ${W}), kernel_size=(${k.join(', ')}), dilation=(${d.join(', ')}), padding=(${p.join(', ')}), stride=(${st.join(', ')}), expected size of input's dimension 2 to match the calculated number of sliding blocks ${rows} * ${cols} = ${rows * cols}, but got input.size(2)=${L}.`)
    const pos: number[] = []
    const from: number[] = []
    let f = 0
    for (let n = 0; n < N; n++) for (let c = 0; c < Ch; c++) for (let u = 0; u < k[0]; u++) for (let v = 0; v < k[1]; v++) {
      for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
        const [y, xx] = [i * st[0] + u * d[0] - p[0], j * st[1] + v * d[1] - p[1]]
        if (y >= 0 && y < H && xx >= 0 && xx < W) {
          pos.push((n * Ch + c) * H * W + y * W + xx)
          from.push(f)
        }
        f++
      }
    }
    const zeros = TT.of(new Array<number>(N * Ch * H * W).fill(0), [N * Ch * H * W], x.dtype)
    const picked = C.takeT(E, x, from, [from.length], 'TakeBackward0')
    const out = C.reshapeT(E, C.scatterT(E, zeros, 0, TT.of(pos, [pos.length], 'int64'), picked, true), [N, Ch, H, W], false)
    return named(x0.ndim === 2 ? C.squeezeT(E, out, 0) : out, 'FoldBackward0')
  }
  const bilinearT = (x1: TT, x2: TT, w: TT, b: TT | null): TT => {
    if (x1.ndim !== x2.ndim) throw h.err('RuntimeError', `Input1 and Input2 must have the same number of dimensions`)
    const a = C.matmul(E, C.unsqueezeT(E, x1, -2), w)
    const out = named(C.reshapeT(E, C.matmul(E, a, C.unsqueezeT(E, x2, -1)), [...x1.shape.slice(0, -1), w.shape[0]], false), 'BilinearBackward0')
    return b ? addT(out, b) : out
  }
  const lpPool = (x: TT, n: number, a: Args): TT => {
    const p = h.num(a.norm_type!)
    const o = poolOpts({ kernel_size: a.kernel_size, stride: a.stride, ceil_mode: a.ceil_mode, divisor_override: py.int(1) }, n)
    const r = C.poolNd(E, C.binary(E, '**', x, TT.scalar(p, x.dtype), false, true), 'avg', o).out
    return C.binary(E, '**', r, TT.scalar(1 / p, x.dtype), false, true)
  }
  const localResponse = (x: TT, size: number, alpha: number, beta: number, k: number): TT => {
    if (x.ndim < 3) throw h.err('ValueError', `Expected 3D or higher dimensionality input (got ${x.ndim} dimensions)`)
    // the running mean of the squares across channels
    const sq = C.unary(E, 'square', x)
    const moved = C.movedimT(E, sq, 1, -1)
    const padded = C.padT(E, moved, [size >> 1, (size - 1) >> 1], 'constant', 0)
    const lead = padded.shape.slice(0, -1)
    const windows = C.poolNd(E, C.reshapeT(E, padded, [-1, 1, padded.shape[padded.ndim - 1]], false), 'avg', { k: [size], s: [1], p: [0], d: [1], ceil: false, countPad: true, divisor: null }).out
    const div = C.movedimT(E, C.reshapeT(E, windows, [...lead, x.shape[1]], false), -1, 1)
    const denom = C.binary(E, '**', addT(C.binary(E, '*', div, TT.scalar(alpha, x.dtype), false, true), TT.scalar(k, x.dtype)), TT.scalar(beta, x.dtype), false, true)
    return C.binary(E, '/', x, denom)
  }
  const scaledDotProduct = (q: TT, k: TT, v: TT, mask: TT | null, dropP: number, causal: boolean, scale: number | null): TT => {
    const L = q.shape[q.ndim - 2]
    const Sz = k.shape[k.ndim - 2]
    const factor = scale ?? 1 / Math.sqrt(q.shape[q.ndim - 1])
    let sc = C.binary(E, '*', C.matmul(E, q, C.transposeT(E, k, -2, -1)), TT.scalar(factor, q.dtype), false, true)
    if (causal) {
      const m = TT.of(Array.from({ length: L * Sz }, (_, f) => (f % Sz > Math.floor(f / Sz) ? 1 : 0)), [L, Sz], 'bool')
      sc = C.whereT(E, m, TT.scalar(-Infinity, sc.dtype), sc, sc.dtype, 'MaskedFillBackward0')
    }
    if (mask) sc = mask.dtype === 'bool' ? C.whereT(E, mask, sc, TT.scalar(-Infinity, sc.dtype), sc.dtype, 'MaskedFillBackward0') : addT(sc, mask)
    const att = C.dropout(E, C.softmax(E, sc, -1, false), dropP, dropP > 0)
    return C.matmul(E, att, v)
  }

  const normClass = (name: string, ndims: number[]): Value => layer(name, 'num_features eps=1e-05 momentum=0.1 affine=True track_running_stats=True * bias=True', (a) => new BatchNorm(name, ndims, h.toInt(a.num_features!), h.num(a.eps!), a.momentum?.k === 'none' ? null : h.num(a.momentum!), bool(a.affine, true), bool(a.track_running_stats, true), bool(a.bias, true)))
  const instanceClass = (name: string, n: number): Value => layer(name, 'num_features eps=1e-05 momentum=0.1 affine=False track_running_stats=False * bias=True', (a) => new InstanceNorm(name, n, h.toInt(a.num_features!), h.num(a.eps!), h.num(a.momentum!), bool(a.affine, false), bool(a.track_running_stats, false), bool(a.bias, true)))
  const dropClass = (name: string): Value => layer(name, 'p=0.5 inplace=False', (a) => new DropoutLayer(h, name, h.num(a.p!), bool(a.inplace, false)))
  const rnnClass = (mode: RnnMode): Value => layer(mode, `input_size hidden_size num_layers=1${mode === 'RNN' ? " nonlinearity='tanh'" : ''} bias=True batch_first=False dropout=0.0 bidirectional=False${mode === 'LSTM' ? ' proj_size=0' : ''}`, (a) => {
    const how = a.nonlinearity ? h.str(a.nonlinearity) : 'tanh'
    if (how !== 'tanh' && how !== 'relu') throw h.err('ValueError', `Unknown nonlinearity '${how}'. Select from 'tanh' or 'relu'.`)
    return new Recurrent(h, mode, how === 'relu', h.toInt(a.input_size!), h.toInt(a.hidden_size!), h.toInt(a.num_layers!), bool(a.bias, true), bool(a.batch_first, false), h.num(a.dropout!), bool(a.bidirectional, false), a.proj_size ? h.toInt(a.proj_size) : 0)
  })
  const embedFromPretrained = fnv('from_pretrained', (args, kw) => {
    const a = bindPy(h, 'Embedding.from_pretrained', 'embeddings freeze=True padding_idx=None max_norm=None norm_type=2.0 scale_grad_by_freq=False sparse=False', args, kw)
    const src = tensorArg(h, a.embeddings, 'from_pretrained')
    if (src.ndim !== 2) throw h.err('AssertionError', 'Embeddings parameter is expected to be 2-dimensional')
    const w = new TT(src.a, src.dtype)
    w.requiresGrad = !bool(a.freeze, true)
    const m = new Embedding(h, src.shape[0], src.shape[1], isNone(a.padding_idx) ? null : h.toInt(a.padding_idx!), isNone(a.max_norm) ? null : h.num(a.max_norm!), h.num(a.norm_type!), bool(a.scale_grad_by_freq, false), bool(a.sparse, false), w)
    m.fields = a
    return py.obj(m)
  })

  const nnAttrs: Record<string, Value> = {
    Module,
    Linear: layer('Linear', 'in_features out_features bias=True', (a) => {
      const [i, o] = [a.in_features!, a.out_features!]
      if (i.k !== 'int' || o.k !== 'int') throw h.err('TypeError', `empty(): argument 'size' failed to unpack the object at pos 2 with error "type must be tuple of ints,but got ${typeName(i.k !== 'int' ? i : o)}"`)
      if (i.v < 0 || o.v < 0) throw h.err('RuntimeError', `Trying to create tensor with negative dimension ${o.v < 0 ? o.v : i.v}: [${o.v}, ${i.v}]`)
      return new Linear(h, i.v, o.v, bool(a.bias, true))
    }),
    Bilinear: layer('Bilinear', 'in1_features in2_features out_features bias=True', (a) => new BilinearLayer(h, h.toInt(a.in1_features!), h.toInt(a.in2_features!), h.toInt(a.out_features!), bool(a.bias, true), bilinearT)),
    Conv1d: convClass('Conv1d', 1), Conv2d: convClass('Conv2d', 2), Conv3d: convClass('Conv3d', 3),
    ConvTranspose1d: convTClass('ConvTranspose1d', 1), ConvTranspose2d: convTClass('ConvTranspose2d', 2), ConvTranspose3d: convTClass('ConvTranspose3d', 3),
    MaxPool1d: maxPoolClass('MaxPool1d', 1), MaxPool2d: maxPoolClass('MaxPool2d', 2), MaxPool3d: maxPoolClass('MaxPool3d', 3),
    AvgPool1d: avgPoolClass('AvgPool1d', 1), AvgPool2d: avgPoolClass('AvgPool2d', 2), AvgPool3d: avgPoolClass('AvgPool3d', 3),
    ...Object.fromEntries(([['AdaptiveAvgPool2d', 'avg', false], ['AdaptiveMaxPool2d', 'max', false], ['AdaptiveAvgPool1d', 'avg', true], ['AdaptiveMaxPool1d', 'max', true]] as const).map(([name, mode, single]) => [name, layer(name, mode === 'max' ? 'output_size return_indices=False' : 'output_size', (a) => {
      const f = new FnLayer(name, (args) => adaptive(tensorArg(h, args[0], `adaptive_${mode}_pool${single ? 1 : 2}d`), a.output_size, mode, single, bool(a.return_indices, false)), `output_size=${repr(a.output_size!)}`)
      return f
    })])),
    ...Object.fromEntries((['MaxUnpool1d', 'MaxUnpool2d', 'MaxUnpool3d'] as const).map((name, i) => [name, layer(name, 'kernel_size stride=None padding=0', (a) => {
      const n = i + 1
      const [k, st, pd] = [ntuple(a.kernel_size, n, 2), ntuple(isNone(a.stride) ? a.kernel_size : a.stride, n, 1), ntuple(a.padding, n, 0)]
      const f = new FnLayer(name, (args) => wrap(maxUnpool(h, tensorArg(h, args[0], `max_unpool${n}d`), tensorArg(h, args[1], `max_unpool${n}d`, 'indices', 2), n, k, st, pd, args[2])), `kernel_size=${dimsStr(k)}, stride=${dimsStr(st)}, padding=${dimsStr(pd)}`)
      f.fwd = ['input', 'indices', 'output_size']
      f.fwdOptional = 1
      return f
    })])),
    LPPool3d: layer('LPPool3d', 'norm_type kernel_size stride=None ceil_mode=False', (a) => new Activation('LPPool3d', (x) => lpPool(x, 3, a), `norm_type=${repr(a.norm_type!)}, kernel_size=${repr(a.kernel_size!)}, stride=${repr(a.stride!)}, ceil_mode=${repr(a.ceil_mode!)}`)),
    AdaptiveAvgPool3d: layer('AdaptiveAvgPool3d', 'output_size', (a) => new FnLayer('AdaptiveAvgPool3d', (args) => adaptive3(tensorArg(h, args[0], 'adaptive_avg_pool3d'), a.output_size!, 'avg', false), `output_size=${repr(a.output_size!)}`)),
    AdaptiveMaxPool3d: layer('AdaptiveMaxPool3d', 'output_size return_indices=False', (a) => new FnLayer('AdaptiveMaxPool3d', (args) => adaptive3(tensorArg(h, args[0], 'adaptive_max_pool3d'), a.output_size!, 'max', bool(a.return_indices, false)), `output_size=${repr(a.output_size!)}`)),
    LPPool1d: layer('LPPool1d', 'norm_type kernel_size stride=None ceil_mode=False', (a) => new Activation('LPPool1d', (x) => lpPool(x, 1, a), `norm_type=${repr(a.norm_type!)}, kernel_size=${repr(a.kernel_size!)}, stride=${repr(a.stride!)}, ceil_mode=${repr(a.ceil_mode!)}`)),
    LPPool2d: layer('LPPool2d', 'norm_type kernel_size stride=None ceil_mode=False', (a) => new Activation('LPPool2d', (x) => lpPool(x, 2, a), `norm_type=${repr(a.norm_type!)}, kernel_size=${repr(a.kernel_size!)}, stride=${repr(a.stride!)}, ceil_mode=${repr(a.ceil_mode!)}`)),
    ...Object.fromEntries(ACTS.map(([name, , params, make, extra]) => [name, layer(name, params, (a) => new Activation(name, make(a), extra(a), bool(a.inplace, false)))])),
    RReLU: layer('RReLU', 'lower=0.125 upper=0.3333333333333333 inplace=False', (a) => {
      const m: Activation = new Activation('RReLU', (x) => rrelu(x, h.num(a.lower!), h.num(a.upper!), m.training), withInplace(a, `lower=${repr(a.lower!)}, upper=${repr(a.upper!)}`), bool(a.inplace, false))
      return m
    }),
    GLU: layer('GLU', 'dim=-1', (a) => new Activation('GLU', (x) => glu(x, h.toInt(a.dim!)), `dim=${h.toInt(a.dim!)}`)),
    PReLU: layer('PReLU', 'num_parameters=1 init=0.25', (a) => new PReLU(h.toInt(a.num_parameters!), h.num(a.init!))),
    Softmax: layer('Softmax', 'dim=None', (a) => new Activation('Softmax', (x) => C.softmax(E, x, softDim(x, a.dim), false), dimRepr(a))),
    LogSoftmax: layer('LogSoftmax', 'dim=None', (a) => new Activation('LogSoftmax', (x) => C.softmax(E, x, softDim(x, a.dim), true), dimRepr(a))),
    Softmin: layer('Softmin', 'dim=None', (a) => new Activation('Softmin', (x) => C.softmax(E, C.unary(E, 'neg', x), softDim(x, a.dim), false), dimRepr(a))),
    Softmax2d: { k: 'type', name: 'Softmax2d', call: () => py.obj(new Activation('Softmax2d', (x) => {
      if (x.ndim !== 3 && x.ndim !== 4) throw h.err('ValueError', `Softmax2d: expected input to be 3D or 4D, got ${x.ndim}D instead`)
      return C.softmax(E, x, -3, false)
    })) },
    Flatten: layer('Flatten', 'start_dim=1 end_dim=-1', (a) => new Activation('Flatten', (x) => C.flattenT(E, x, h.toInt(a.start_dim!), h.toInt(a.end_dim!)), `start_dim=${h.toInt(a.start_dim!)}, end_dim=${h.toInt(a.end_dim!)}`)),
    Unflatten,
    Identity: { k: 'type', name: 'Identity', call: () => py.obj(new Activation('Identity', (x) => x)) },
    Dropout: dropClass('Dropout'), Dropout1d: dropClass('Dropout1d'), Dropout2d: dropClass('Dropout2d'), Dropout3d: dropClass('Dropout3d'),
    FeatureAlphaDropout: layer('FeatureAlphaDropout', 'p=0.5 inplace=False', (a) => {
      const m: Activation = new Activation('FeatureAlphaDropout', (x) => featureAlphaDropout(x, h.num(a.p!), m.training), `p=${repr(a.p!)}, inplace=${repr(a.inplace!)}`, bool(a.inplace, false))
      return m
    }),
    AlphaDropout: layer('AlphaDropout', 'p=0.5 inplace=False', (a) => {
      const m: Activation = new Activation('AlphaDropout', (x) => alphaDropout(x, h.num(a.p!), m.training), `p=${repr(a.p!)}, inplace=${repr(a.inplace!)}`, bool(a.inplace, false))
      return m
    }),
    BatchNorm1d: normClass('BatchNorm1d', [2, 3]), BatchNorm2d: normClass('BatchNorm2d', [4]), BatchNorm3d: normClass('BatchNorm3d', [5]),
    InstanceNorm1d: instanceClass('InstanceNorm1d', 1), InstanceNorm2d: instanceClass('InstanceNorm2d', 2), InstanceNorm3d: instanceClass('InstanceNorm3d', 3),
    LayerNorm: layer('LayerNorm', 'normalized_shape eps=1e-05 elementwise_affine=True bias=True', (a) => new LayerNorm(toDims(h, [a.normalized_shape!]), h.num(a.eps!), bool(a.elementwise_affine, true), bool(a.bias, true))),
    GroupNorm: layer('GroupNorm', 'num_groups num_channels eps=1e-05 affine=True * bias=True', (a) => new GroupNorm(h, h.toInt(a.num_groups!), h.toInt(a.num_channels!), h.num(a.eps!), bool(a.affine, true), bool(a.bias, true))),
    RMSNorm: layer('RMSNorm', 'normalized_shape eps=None elementwise_affine=True', (a) => new RMSNorm(toDims(h, [a.normalized_shape!]), isNone(a.eps) ? 1.1920928955078125e-07 : h.num(a.eps!), bool(a.elementwise_affine, true))),
    LocalResponseNorm: layer('LocalResponseNorm', 'size alpha=0.0001 beta=0.75 k=1.0', (a) => new Activation('LocalResponseNorm', (x) => localResponse(x, h.toInt(a.size!), h.num(a.alpha!), h.num(a.beta!), h.num(a.k!)), `${h.toInt(a.size!)}, alpha=${repr(a.alpha!)}, beta=${repr(a.beta!)}, k=${repr(a.k!)}`)),
    Embedding: layer('Embedding', 'num_embeddings embedding_dim padding_idx=None max_norm=None norm_type=2.0 scale_grad_by_freq=False sparse=False _weight=None _freeze=False', (a) => {
      const w = isNone(a._weight) ? null : new TT(tensorArg(h, a._weight, 'Embedding').a, tensorArg(h, a._weight, 'Embedding').dtype)
      if (w) w.requiresGrad = !bool(a._freeze, false)
      return new Embedding(h, h.toInt(a.num_embeddings!), h.toInt(a.embedding_dim!), isNone(a.padding_idx) ? null : h.toInt(a.padding_idx!), isNone(a.max_norm) ? null : h.num(a.max_norm!), h.num(a.norm_type!), bool(a.scale_grad_by_freq, false), bool(a.sparse, false), w)
    }, new Map([['from_pretrained', embedFromPretrained]])),
    EmbeddingBag: layer('EmbeddingBag', "num_embeddings embedding_dim max_norm=None norm_type=2.0 scale_grad_by_freq=False mode='mean' sparse=False _weight=None include_last_offset=False padding_idx=None", (a) => {
      if (!isNone(a.max_norm)) throw h.err('NotImplementedError', 'EmbeddingBag: max_norm= is not available in this sandbox')
      if (bool(a.include_last_offset, false)) throw h.err('NotImplementedError', 'EmbeddingBag: include_last_offset=True is not available in this sandbox')
      return new EmbeddingBag(h, h.toInt(a.num_embeddings!), h.toInt(a.embedding_dim!), h.str(a.mode!), isNone(a.padding_idx) ? null : h.toInt(a.padding_idx!))
    }),
    RNN: rnnClass('RNN'), LSTM: rnnClass('LSTM'), GRU: rnnClass('GRU'),
    RNNCell: layer('RNNCell', "input_size hidden_size bias=True nonlinearity='tanh'", (a) => new RecurrentCell(h, 'RNNCell', 'RNN', h.str(a.nonlinearity!) === 'relu', h.toInt(a.input_size!), h.toInt(a.hidden_size!), bool(a.bias, true))),
    LSTMCell: layer('LSTMCell', 'input_size hidden_size bias=True', (a) => new RecurrentCell(h, 'LSTMCell', 'LSTM', false, h.toInt(a.input_size!), h.toInt(a.hidden_size!), bool(a.bias, true))),
    GRUCell: layer('GRUCell', 'input_size hidden_size bias=True', (a) => new RecurrentCell(h, 'GRUCell', 'GRU', false, h.toInt(a.input_size!), h.toInt(a.hidden_size!), bool(a.bias, true))),
    MultiheadAttention: layer('MultiheadAttention', 'embed_dim num_heads dropout=0.0 bias=True add_bias_kv=False add_zero_attn=False kdim=None vdim=None batch_first=False', (a) => {
      for (const key of ['add_bias_kv', 'add_zero_attn']) if (bool(a[key], false)) throw h.err('NotImplementedError', `MultiheadAttention: ${key}=True is not available in this sandbox`)
      if (!isNone(a.kdim) || !isNone(a.vdim)) throw h.err('NotImplementedError', 'MultiheadAttention: kdim and vdim different from embed_dim are not available in this sandbox')
      return new MultiheadAttention(h, h.toInt(a.embed_dim!), h.toInt(a.num_heads!), h.num(a.dropout!), bool(a.bias, true), bool(a.batch_first, false))
    }),
    TransformerEncoderLayer: layer('TransformerEncoderLayer', "d_model nhead dim_feedforward=2048 dropout=0.1 activation='relu' layer_norm_eps=1e-05 batch_first=False norm_first=False bias=True", (a) => new TransformerEncoderLayer(h, a, nnBuilt)),
    TransformerEncoder: layer('TransformerEncoder', 'encoder_layer num_layers norm=None enable_nested_tensor=True mask_check=True', (a) => new TransformerEncoder(a.encoder_layer!, h.toInt(a.num_layers!), isNone(a.norm) ? null : a.norm!)),
    Upsample: layer('Upsample', "size=None scale_factor=None mode='nearest' align_corners=None recompute_scale_factor=None", (a) => {
      const mode = h.str(a.mode!)
      const scale = isNone(a.scale_factor) ? null : a.scale_factor!
      const shown = scale ? (scale.k === 'tuple' ? py.tuple(scale.items.map((v) => py.float(h.num(v)))) : py.float(h.num(scale))) : a.size!
      return new Activation('Upsample', (x) => interpolate(x, ...recomputed(x, a.size, a.scale_factor, a.recompute_scale_factor), mode, a.align_corners), `${scale ? 'scale_factor' : 'size'}=${repr(shown)}, mode=${repr(py.str(mode))}`)
    }),
    UpsamplingNearest2d: layer('UpsamplingNearest2d', 'size=None scale_factor=None', (a) => {
      const scale = isNone(a.scale_factor) ? null : a.scale_factor!
      return new Activation('UpsamplingNearest2d', (x) => interpolate(x, a.size, a.scale_factor, 'nearest', undefined), `${scale ? 'scale_factor' : 'size'}=${repr(scale ? py.float(h.num(scale)) : a.size!)}, mode='nearest'`)
    }),
    UpsamplingBilinear2d: layer('UpsamplingBilinear2d', 'size=None scale_factor=None', (a) => {
      const scale = isNone(a.scale_factor) ? null : a.scale_factor!
      return new Activation('UpsamplingBilinear2d', (x) => interpolate(x, a.size, a.scale_factor, 'bilinear', py.bool(true)), `${scale ? 'scale_factor' : 'size'}=${repr(scale ? py.float(h.num(scale)) : a.size!)}, mode='bilinear'`)
    }),
    PixelShuffle: layer('PixelShuffle', 'upscale_factor', (a) => new Activation('PixelShuffle', (x) => pixelShuffle(x, h.toInt(a.upscale_factor!)), `upscale_factor=${h.toInt(a.upscale_factor!)}`)),
    PixelUnshuffle: layer('PixelUnshuffle', 'downscale_factor', (a) => new Activation('PixelUnshuffle', (x) => pixelUnshuffle(x, h.toInt(a.downscale_factor!)), `downscale_factor=${h.toInt(a.downscale_factor!)}`)),
    ChannelShuffle: layer('ChannelShuffle', 'groups', (a) => new Activation('ChannelShuffle', (x) => channelShuffle(x, h.toInt(a.groups!)), `groups=${h.toInt(a.groups!)}`)),
    Unfold: layer('Unfold', 'kernel_size dilation=1 padding=0 stride=1', (a) => new Activation('Unfold', (x) => unfold(x, a.kernel_size!, a.dilation, a.padding, a.stride), `kernel_size=${repr(a.kernel_size!)}, dilation=${repr(a.dilation!)}, padding=${repr(a.padding!)}, stride=${repr(a.stride!)}`)),
    Fold: layer('Fold', 'output_size kernel_size dilation=1 padding=0 stride=1', (a) => new Activation('Fold', (x) => foldT(x, a.output_size!, a.kernel_size!, a.dilation, a.padding, a.stride), `output_size=${repr(a.output_size!)}, kernel_size=${repr(a.kernel_size!)}, dilation=${repr(a.dilation!)}, padding=${repr(a.padding!)}, stride=${repr(a.stride!)}`)),
    ZeroPad1d: padLayer('ZeroPad1d', 1, 'constant', false), ZeroPad2d: padLayer('ZeroPad2d', 2, 'constant', false), ZeroPad3d: padLayer('ZeroPad3d', 3, 'constant', false),
    ConstantPad1d: padLayer('ConstantPad1d', 1, 'constant', true), ConstantPad2d: padLayer('ConstantPad2d', 2, 'constant', true), ConstantPad3d: padLayer('ConstantPad3d', 3, 'constant', true),
    ReflectionPad1d: padLayer('ReflectionPad1d', 1, 'reflect', false), ReflectionPad2d: padLayer('ReflectionPad2d', 2, 'reflect', false), ReflectionPad3d: padLayer('ReflectionPad3d', 3, 'reflect', false),
    ReplicationPad1d: padLayer('ReplicationPad1d', 1, 'replicate', false), ReplicationPad2d: padLayer('ReplicationPad2d', 2, 'replicate', false), ReplicationPad3d: padLayer('ReplicationPad3d', 3, 'replicate', false),
    CircularPad1d: padLayer('CircularPad1d', 1, 'circular', false), CircularPad2d: padLayer('CircularPad2d', 2, 'circular', false), CircularPad3d: padLayer('CircularPad3d', 3, 'circular', false),
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
    ParameterList: { k: 'type', name: 'ParameterList', call: (args) => py.obj(new ParameterList(args[0] ? h.iterate(args[0]).map((v) => tensorArg(h, v, 'ParameterList')) : [])) },
    ParameterDict: { k: 'type', name: 'ParameterDict', call: (args) => {
      const m = new ParameterDict()
      if (args[0]) m.update(args[0], h)
      return py.obj(m)
    } },
    CrossEntropyLoss: lossLayer('CrossEntropyLoss', `weight=None size_average=None ignore_index=-100 reduce=None reduction='mean' label_smoothing=0.0`, (a) => (x, y, hh) => ceFull(hh, x, y, ceOpts(a))),
    NLLLoss: lossLayer('NLLLoss', `weight=None size_average=None ignore_index=-100 reduce=None reduction='mean'`, (a) => (x, y, hh) => nllFull(hh, x, y, ceOpts(a))),
    NLLLoss2d: lossLayer('NLLLoss2d', `weight=None size_average=None ignore_index=-100 reduce=None reduction='mean'`, (a) => (x, y, hh) => nllFull(hh, x, y, ceOpts(a))),
    MSELoss: lossLayer('MSELoss', LEGACY_LOSS, (a) => (x, y, hh) => mse(hh, x, y, reduction(a))),
    BCELoss: lossLayer('BCELoss', `weight=None ${LEGACY_LOSS}`, (a) => (x, y, hh) => bceWeighted(hh, x, y, false, optT(a.weight), null, reduction(a))),
    BCEWithLogitsLoss: lossLayer('BCEWithLogitsLoss', `weight=None ${LEGACY_LOSS} pos_weight=None`, (a) => (x, y, hh) => bceWeighted(hh, x, y, true, optT(a.weight), optT(a.pos_weight, 'pos_weight'), reduction(a))),
    L1Loss: lossLayer('L1Loss', LEGACY_LOSS, (a) => (x, y, hh) => l1(hh, x, y, reduction(a))),
    SmoothL1Loss: lossLayer('SmoothL1Loss', `${LEGACY_LOSS} beta=1.0`, (a) => (x, y, hh) => smoothL1(hh, x, y, h.num(a.beta!), reduction(a))),
    HuberLoss: lossLayer('HuberLoss', "reduction='mean' delta=1.0", (a) => (x, y, hh) => huber(hh, x, y, h.num(a.delta!), reduction(a))),
    KLDivLoss: lossLayer('KLDivLoss', "size_average=None reduce=None reduction='mean' log_target=False", (a) => (x, y) => klDiv(x, y, h.str(a.reduction!), bool(a.log_target, false))),
    PoissonNLLLoss: lossLayer('PoissonNLLLoss', "log_input=True full=False size_average=None eps=1e-08 reduce=None reduction='mean'", (a) => (x, y) => poissonNll(x, y, bool(a.log_input, true), bool(a.full, false), h.num(a.eps!), reduction(a))),
    GaussianNLLLoss: lossLayer('GaussianNLLLoss', "* full=False eps=1e-06 reduction='mean'", (a) => (x, y, _hh, [v]) => gaussianNll(x, y, v, bool(a.full, false), h.num(a.eps!), reduction(a)), ['input', 'target', 'var']),
    MarginRankingLoss: lossLayer('MarginRankingLoss', `margin=0.0 ${LEGACY_LOSS}`, (a) => (x1, x2, _hh, [y]) => marginRanking(x1, x2, y, h.num(a.margin!), reduction(a)), ['input1', 'input2', 'target']),
    HingeEmbeddingLoss: lossLayer('HingeEmbeddingLoss', `margin=1.0 ${LEGACY_LOSS}`, (a) => (x, y) => hingeEmbedding(x, y, h.num(a.margin!), reduction(a))),
    CosineEmbeddingLoss: lossLayer('CosineEmbeddingLoss', `margin=0.0 ${LEGACY_LOSS}`, (a) => (x1, x2, _hh, [y]) => cosineEmbedding(x1, x2, y, h.num(a.margin!), reduction(a)), ['input1', 'input2', 'target']),
    TripletMarginLoss: lossLayer('TripletMarginLoss', `margin=1.0 p=2.0 eps=1e-06 swap=False ${LEGACY_LOSS}`, (a) => (x, pos, _hh, [neg]) => tripletMargin(x, pos, neg, h.num(a.margin!), h.num(a.p!), h.num(a.eps!), bool(a.swap, false), reduction(a)), ['anchor', 'positive', 'negative']),
    MultiMarginLoss: lossLayer('MultiMarginLoss', `p=1 margin=1.0 weight=None ${LEGACY_LOSS}`, (a) => (x, y) => multiMargin(x, y, h.num(a.p!), h.num(a.margin!), optT(a.weight), reduction(a))),
    SoftMarginLoss: lossLayer('SoftMarginLoss', LEGACY_LOSS, (a) => (x, y) => softMargin(x, y, reduction(a))),
    MultiLabelMarginLoss: lossLayer('MultiLabelMarginLoss', LEGACY_LOSS, (a) => (x, y) => multiLabelMargin(x, y, reduction(a))),
    MultiLabelSoftMarginLoss: lossLayer('MultiLabelSoftMarginLoss', `weight=None ${LEGACY_LOSS}`, (a) => (x, y) => multiLabelSoftMargin(x, y, optT(a.weight), reduction(a))),
    CosineSimilarity: layer('CosineSimilarity', 'dim=1 eps=1e-08', (a) => new LossLayer('CosineSimilarity', (x, y) => cosineSim(x, y, h.toInt(a.dim!), h.num(a.eps!)), ['x1', 'x2'])),
    PairwiseDistance: layer('PairwiseDistance', 'p=2.0 eps=1e-06 keepdim=False', (a) => new LossLayer('PairwiseDistance', (x, y) => pairDistance(x, y, h.num(a.p!), h.num(a.eps!), bool(a.keepdim, false)), ['x1', 'x2'])),
    TripletMarginWithDistanceLoss: layer('TripletMarginWithDistanceLoss', "* distance_function=None margin=1.0 swap=False reduction='mean'", (a) => {
      const fn0 = isNone(a.distance_function) ? nnBuilt('PairwiseDistance', []) : a.distance_function!
      const dist = (p: TT, q: TT) => tensorArg(h, h.call(fn0, [wrap(p), wrap(q)]), 'distance_function')
      const m = new LossLayer('TripletMarginWithDistanceLoss', (x, pos, _hh, [neg]) => {
        let far = dist(x, neg)
        if (bool(a.swap, false)) far = C.binary(E, 'minimum', far, dist(pos, neg))
        return reduced(clamp0(addT(subT(dist(x, pos), far), one(x, h.num(a.margin!)))), reduction(a))
      }, ['anchor', 'positive', 'negative'])
      if (moduleOf(fn0)) m.kids = [['distance_function', fn0]]
      return m
    }),
    LazyLinear: layer('LazyLinear', 'out_features bias=True', (a) => new LazyLinear(h, h.toInt(a.out_features!), bool(a.bias, true))),
    TransformerDecoderLayer: layer('TransformerDecoderLayer', "d_model nhead dim_feedforward=2048 dropout=0.1 activation='relu' layer_norm_eps=1e-05 batch_first=False norm_first=False bias=True", (a) => new TransformerDecoderLayer(h, a, nnBuilt)),
    TransformerDecoder: layer('TransformerDecoder', 'decoder_layer num_layers norm=None', (a) => new TransformerDecoder(a.decoder_layer!, h.toInt(a.num_layers!), isNone(a.norm) ? null : a.norm!)),
    Parameter: { k: 'type', name: 'Parameter', call: (args, kw) => {
      const v = args[0] ?? kw.data
      const src = v ? tensorArg(h, v, 'Parameter', 'data') : TT.of([], [0], C.DEFAULTS.float)
      // shares memory with the tensor it wraps, as in PyTorch
      const t = new TT(src.a, src.dtype)
      t.isParam = true
      t.requiresGrad = flagOf(h, kw.requires_grad ?? args[1], true)
      if (t.requiresGrad && !isFloatT(t.dtype)) throw h.err('RuntimeError', 'Only Tensors of floating point and complex dtype can require gradients')
      return wrap(t)
    } },
    Buffer: { k: 'type', name: 'Buffer', call: (args, kw) => py.obj(new BufferMark(tensorArg(h, args[0] ?? kw.data, 'Buffer', 'data'), flagOf(h, kw.persistent, true))) },
  }
  /** what TransformerEncoderLayer builds its children from */
  const nnBuilt = (name: string, args: Value[], kw: Kw = {}): Value => h.call(nnAttrs[name], args, kw)

  const rrelu = (x: TT, lo: number, hi: number, training: boolean): TT => {
    if (!training) {
      const s = (lo + hi) / 2
      return C.mapT(E, x, (v) => (v >= 0 ? v : s * v), (v) => (v > 0 ? 1 : s), 'RreluWithNoiseBackward0')
    }
    const noise = x.values().map(() => lo + (hi - lo) * E.rng.next())
    return C.whereT(E, C.compare(x, TT.scalar(0, x.dtype), '>='), x, C.binary(E, '*', x, TT.of(noise, x.shape, x.dtype)), x.dtype, 'RreluWithNoiseBackward0')
  }
  const featureAlphaDropout = (x: TT, p: number, training: boolean): TT => {
    if (p < 0 || p > 1) throw h.err('ValueError', `dropout probability has to be between 0 and 1, but got ${repr(py.float(p))}`)
    if (!training || p === 0) return x
    const alpha = 1.7811865554395001
    const a = ((1 - p) * (1 + p * alpha * alpha)) ** -0.5
    const b = -a * alpha * p
    const per = prod(x.shape.slice(2))
    const draws = Array.from({ length: x.size / Math.max(per, 1) }, () => (E.rng.next() < 1 - p ? 1 : 0))
    const keep = Array.from({ length: x.size }, (_, f) => draws[Math.floor(f / Math.max(per, 1))])
    return addT(mulT(addT(mulT(x, TT.of(keep, x.shape, x.dtype)), TT.of(keep.map((k) => (k ? 0 : -alpha)), x.shape, x.dtype)), TT.scalar(a, x.dtype)), TT.scalar(b, x.dtype))
  }
  const glu = (x: TT, dim: number): TT => {
    const d = C.normDim(dim, x.ndim)
    const n = x.shape[d]
    if (n % 2) throw h.err('RuntimeError', `Halving dimension must be even, but dimension ${d} is size ${n}`)
    return named(C.binary(E, '*', C.narrowT(E, x, d, 0, n / 2), C.unary(E, 'sigmoid', C.narrowT(E, x, d, n / 2, n / 2))), 'GluBackward0')
  }
  const alphaDropout = (x: TT, p: number, training: boolean): TT => {
    if (p < 0 || p > 1) throw h.err('ValueError', `dropout probability has to be between 0 and 1, but got ${repr(py.float(p))}`)
    if (!training || p === 0) return x
    const alpha = 1.7811865554395001
    const a = ((1 - p) * (1 + p * alpha * alpha)) ** -0.5
    const b = -a * alpha * p
    const keep = x.values().map(() => (E.rng.next() < 1 - p ? 1 : 0))
    return C.mapT(E, x, (v) => v, () => 1, 'AlphaDropoutBackward0') && addT(mulT(addT(mulT(x, TT.of(keep, x.shape, x.dtype)), TT.of(keep.map((k) => (k ? 0 : -alpha)), x.shape, x.dtype)), TT.scalar(a, x.dtype)), TT.scalar(b, x.dtype))
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
    const a = bindPy(h, name, params ? `input ${params}` : 'input', args, kw)
    const t = tensorArg(h, a.input, name)
    const out = run(t, a)
    trace(h, `F.${name}`, kind, [{ label: label(h, a.input!, 'x'), t }], out)
    flowStep(h, `F.${name}`, t, out)
    return wrap(out)
  })
  const Floss = (name: string, params: string, run: (x: TT, y: TT, a: Args) => TT, first = 'input', second = 'target') => fnv(name, (args, kw) => {
    const a = bindPy(h, name, `${first} ${second} ${params}`, args, kw)
    return wrap(run(tensorArg(h, a[first], name), tensorArg(h, a[second], name, second, 2), a))
  })
  const dropF = (name: string, f: (x: TT, p: number, training: boolean) => TT) => Ff(name, 'p=0.5 training=True inplace=False', (x, a) => {
    const g = (t: TT) => f(t, h.num(a.p!), bool(a.training, true))
    return bool(a.inplace, false) ? asTT(objOf(x).rebase(h, g))! : g(x)
  }, 'random')
  const checkedP = (f: (E: Engine, x: TT, p: number, t: boolean) => TT) => (x: TT, p: number, t: boolean) => {
    if (p < 0 || p > 1) throw h.err('ValueError', `dropout probability has to be between 0 and 1, but got ${repr(py.float(p))}`)
    return f(E, x, p, t)
  }
  const convF = (name: string, n: number) => fnv(name, (args, kw) => {
    const a = bindPy(h, name, 'input weight bias=None stride=1 padding=0 dilation=1 groups=1', args, kw)
    const x = tensorArg(h, a.input, name)
    const w = tensorArg(h, a.weight, name, 'weight', 2)
    const o = convOpts(a, n)
    const out = convolve(h, x, w, isNone(a.bias) ? null : asTT(a.bias!), o, n)
    if (n === 2 && typeof o.padding !== 'string' && o.groups === 1 && o.dilation.every((q) => q === 1)) traceConv(h, x, w, out, o.stride, o.padding, label(h, a.input!, 'input'), label(h, a.weight!, 'weight'))
    flowStep(h, `F.${name}`, x, out)
    return wrap(out)
  })
  const convTF = (name: string, n: number) => fnv(name, (args, kw) => {
    const a = bindPy(h, name, 'input weight bias=None stride=1 padding=0 output_padding=0 groups=1 dilation=1', args, kw)
    const x = tensorArg(h, a.input, name)
    const out = convolveT(h, x, tensorArg(h, a.weight, name, 'weight', 2), isNone(a.bias) ? null : asTT(a.bias!), ntuple(a.stride, n, 1), ntuple(a.padding, n, 0), ntuple(a.dilation, n, 1), ntuple(a.output_padding, n, 0), h.toInt(a.groups!), n)
    flowStep(h, `F.${name}`, x, out)
    return wrap(out)
  })
  const Fattrs: Record<string, Value> = {
    ...Object.fromEntries(ACTS.map(([, name, params, make]) => [name, Ff(name, params, (x, a) => (bool(a.inplace, false) ? asTT(objOf(x).rebase(h, make(a)))! : make(a)(x)))])),
    // relu_, elu_ …: the in-place spellings
    ...Object.fromEntries(ACTS.filter(([, , params]) => params.includes('inplace')).map(([, name, params, make]) => [`${name}_`, Ff(`${name}_`, params.replace(/ ?inplace=False/, ''), (x, a) => asTT(objOf(x).rebase(h, make(a)))!)])),
    prelu: Ff('prelu', 'weight', (x, a) => {
      const w = tensorArg(h, a.weight, 'prelu', 'weight', 2)
      return C.whereT(E, C.compare(x, TT.scalar(0, x.dtype), '>='), x, C.binary(E, '*', x, w.size > 1 ? C.reshapeT(E, w, x.shape.map((_, k) => (k === 1 ? w.size : 1)), false) : w), x.dtype, 'PreluKernelBackward0')
    }),
    rrelu: Ff('rrelu', 'lower=0.125 upper=0.3333333333333333 training=False inplace=False', (x, a) => {
      const f = (t: TT) => rrelu(t, h.num(a.lower!), h.num(a.upper!), bool(a.training, false))
      return bool(a.inplace, false) ? asTT(objOf(x).rebase(h, f))! : f(x)
    }),
    glu: Ff('glu', 'dim=-1', (x, a) => glu(x, h.toInt(a.dim!))),
    softmax: Ff('softmax', 'dim=None _stacklevel=3 dtype=None', (x, a) => C.softmax(E, dtypeArg(h, a.dtype) ? C.toDtype(E, x, dtypeArg(h, a.dtype)!) : x, softDim(x, a.dim), false)),
    log_softmax: Ff('log_softmax', 'dim=None _stacklevel=3 dtype=None', (x, a) => C.softmax(E, dtypeArg(h, a.dtype) ? C.toDtype(E, x, dtypeArg(h, a.dtype)!) : x, softDim(x, a.dim), true)),
    softmin: Ff('softmin', 'dim=None _stacklevel=3 dtype=None', (x, a) => C.softmax(E, C.unary(E, 'neg', dtypeArg(h, a.dtype) ? C.toDtype(E, x, dtypeArg(h, a.dtype)!) : x), softDim(x, a.dim), false)),
    gumbel_softmax: fnv('gumbel_softmax', (args, kw) => {
      const a = bindPy(h, 'gumbel_softmax', 'logits tau=1 hard=False eps=1e-10 dim=-1', args, kw)
      const x = tensorArg(h, a.logits, 'gumbel_softmax', 'logits')
      const noise = TT.of(x.values().map(() => -Math.log(-Math.log(1 - E.rng.next()))), x.shape, x.dtype)
      const soft = C.softmax(E, C.binary(E, '/', addT(x, noise), TT.scalar(h.num(a.tau!), x.dtype), false, true), h.toInt(a.dim!), false)
      if (!bool(a.hard, false)) return wrap(soft)
      const d = C.normDim(h.toInt(a.dim!), x.ndim)
      const hard = C.scatterT({ grad: false } as Engine, TT.of(new Array<number>(x.size).fill(0), x.shape, x.dtype), d, C.maxAlong({ grad: false } as Engine, soft, d, true).indices, 1, false)
      return wrap(addT(subT(hard, C.detach(soft)), soft))
    }),
    dropout: dropF('dropout', checkedP(C.dropout)), alpha_dropout: dropF('alpha_dropout', alphaDropout),
    dropout1d: dropF('dropout1d', checkedP(C.dropoutChannels)), dropout2d: dropF('dropout2d', checkedP(C.dropoutChannels)), dropout3d: dropF('dropout3d', checkedP(C.dropoutChannels)),
    max_pool1d: maxPoolF('max_pool1d', 1), max_pool2d: maxPoolF('max_pool2d', 2), max_pool3d: maxPoolF('max_pool3d', 3),
    max_pool1d_with_indices: maxPoolF('max_pool1d_with_indices', 1, true), max_pool2d_with_indices: maxPoolF('max_pool2d_with_indices', 2, true), max_pool3d_with_indices: maxPoolF('max_pool3d_with_indices', 3, true),
    avg_pool1d: avgPoolF('avg_pool1d', 1), avg_pool2d: avgPoolF('avg_pool2d', 2), avg_pool3d: avgPoolF('avg_pool3d', 3),
    lp_pool1d: Ff('lp_pool1d', 'norm_type kernel_size stride=None ceil_mode=False', (x, a) => lpPool(x, 1, a)),
    lp_pool2d: Ff('lp_pool2d', 'norm_type kernel_size stride=None ceil_mode=False', (x, a) => lpPool(x, 2, a)),
    max_unpool1d: fnv('max_unpool1d', (args, kw) => {
      const a = bindPy(h, 'max_unpool1d', 'input indices kernel_size stride=None padding=0 output_size=None', args, kw)
      return wrap(maxUnpool(h, tensorArg(h, a.input, 'max_unpool1d'), tensorArg(h, a.indices, 'max_unpool1d', 'indices', 2), 1, ntuple(a.kernel_size, 1, 2), ntuple(isNone(a.stride) ? a.kernel_size : a.stride, 1, 1), ntuple(a.padding, 1, 0), a.output_size))
    }),
    max_unpool2d: fnv('max_unpool2d', (args, kw) => {
      const a = bindPy(h, 'max_unpool2d', 'input indices kernel_size stride=None padding=0 output_size=None', args, kw)
      return wrap(maxUnpool(h, tensorArg(h, a.input, 'max_unpool2d'), tensorArg(h, a.indices, 'max_unpool2d', 'indices', 2), 2, ntuple(a.kernel_size, 2, 2), ntuple(isNone(a.stride) ? a.kernel_size : a.stride, 2, 1), ntuple(a.padding, 2, 0), a.output_size))
    }),
    adaptive_avg_pool3d: Ff('adaptive_avg_pool3d', 'output_size', (x, a) => asTT(adaptive3(x, a.output_size!, 'avg', false))!, 'reduce'),
    adaptive_max_pool3d: fnv('adaptive_max_pool3d', (args, kw) => {
      const a = bindPy(h, 'adaptive_max_pool3d', 'input output_size return_indices=False', args, kw)
      return adaptive3(tensorArg(h, a.input, 'adaptive_max_pool3d'), a.output_size!, 'max', bool(a.return_indices, false))
    }),
    adaptive_max_pool3d_with_indices: fnv('adaptive_max_pool3d_with_indices', (args, kw) => {
      const a = bindPy(h, 'adaptive_max_pool3d_with_indices', 'input output_size return_indices=False', args, kw)
      return adaptive3(tensorArg(h, a.input, 'adaptive_max_pool3d'), a.output_size!, 'max', true)
    }),
    adaptive_max_pool1d_with_indices: fnv('adaptive_max_pool1d_with_indices', (args, kw) => {
      const a = bindPy(h, 'adaptive_max_pool1d_with_indices', 'input output_size return_indices=False', args, kw)
      return adaptive(tensorArg(h, a.input, 'adaptive_max_pool1d'), a.output_size, 'max', true, true)
    }),
    adaptive_max_pool2d_with_indices: fnv('adaptive_max_pool2d_with_indices', (args, kw) => {
      const a = bindPy(h, 'adaptive_max_pool2d_with_indices', 'input output_size return_indices=False', args, kw)
      return adaptive(tensorArg(h, a.input, 'adaptive_max_pool2d'), a.output_size, 'max', false, true)
    }),
    lp_pool3d: Ff('lp_pool3d', 'norm_type kernel_size stride=None ceil_mode=False', (x, a) => lpPool(x, 3, a)),
    max_unpool3d: fnv('max_unpool3d', (args, kw) => {
      const a = bindPy(h, 'max_unpool3d', 'input indices kernel_size stride=None padding=0 output_size=None', args, kw)
      return wrap(maxUnpool(h, tensorArg(h, a.input, 'max_unpool3d'), tensorArg(h, a.indices, 'max_unpool3d', 'indices', 2), 3, ntuple(a.kernel_size, 3, 2), ntuple(isNone(a.stride) ? a.kernel_size : a.stride, 3, 1), ntuple(a.padding, 3, 0), a.output_size))
    }),
    feature_alpha_dropout: dropF('feature_alpha_dropout', featureAlphaDropout),
    group_norm: fnv('group_norm', (args, kw) => {
      const a = bindPy(h, 'group_norm', 'input num_groups weight=None bias=None eps=1e-05', args, kw)
      const x = tensorArg(h, a.input, 'group_norm')
      const g = new GroupNorm(h, h.toInt(a.num_groups!), x.shape[1], h.num(a.eps!), false, false)
      g.w = optT(a.weight)
      g.b = optT(a.bias, 'bias')
      return g.forward([a.input!], {}, h)
    }),
    layer_norm: fnv('layer_norm', (args, kw) => {
      const a = bindPy(h, 'layer_norm', 'input normalized_shape weight=None bias=None eps=1e-05', args, kw)
      const m = new LayerNorm(toDims(h, [a.normalized_shape!]), h.num(a.eps!), false, false)
      m.w = optT(a.weight)
      m.b = optT(a.bias, 'bias')
      return m.forward([a.input!], {}, h)
    }),
    instance_norm: fnv('instance_norm', (args, kw) => {
      const a = bindPy(h, 'instance_norm', 'input running_mean=None running_var=None weight=None bias=None use_input_stats=True momentum=0.1 eps=1e-05', args, kw)
      const x = tensorArg(h, a.input, 'instance_norm')
      if (!bool(a.use_input_stats, true)) throw h.err('NotImplementedError', 'F.instance_norm: use_input_stats=False is not available in this sandbox')
      const m = new InstanceNorm('InstanceNorm', x.ndim - 2, x.shape[1], h.num(a.eps!), 0.1, false, false, false)
      m.w = optT(a.weight)
      m.b = optT(a.bias, 'bias')
      return m.forward([a.input!], {}, h)
    }),
    batch_norm: fnv('batch_norm', (args, kw) => {
      const a = bindPy(h, 'batch_norm', 'input running_mean running_var weight=None bias=None training=False momentum=0.1 eps=1e-05', args, kw)
      const x = tensorArg(h, a.input, 'batch_norm')
      const useStats = !bool(a.training, false) && !isNone(a.running_mean)
      const stats = useStats ? { mean: [...T(a.running_mean, 'batch_norm').values()], vari: [...T(a.running_var, 'batch_norm').values()] } : null
      return wrap(C.batchNormT(E, x, optT(a.weight), optT(a.bias, 'bias'), stats, h.num(a.eps!)).out)
    }),
    embedding_bag: fnv('embedding_bag', (args, kw) => {
      const a = bindPy(h, 'embedding_bag', "input weight offsets=None max_norm=None norm_type=2 scale_grad_by_freq=False mode='mean' sparse=False per_sample_weights=None include_last_offset=False padding_idx=None", args, kw)
      const w = tensorArg(h, a.weight, 'embedding_bag', 'weight', 2)
      for (const key of ['max_norm', 'per_sample_weights']) if (!isNone(a[key])) throw h.err('NotImplementedError', `F.embedding_bag: ${key}= is not available in this sandbox`)
      if (bool(a.include_last_offset, false)) throw h.err('NotImplementedError', 'F.embedding_bag: include_last_offset=True is not available in this sandbox')
      const m = new EmbeddingBag(h, w.shape[0], w.shape[1], h.str(a.mode!), isNone(a.padding_idx) ? null : h.toInt(a.padding_idx!))
      m.w = w
      return m.forward(isNone(a.offsets) ? [a.input!] : [a.input!, a.offsets!], {}, h)
    }),
    triplet_margin_with_distance_loss: fnv('triplet_margin_with_distance_loss', (args, kw) => {
      const a = bindPy(h, 'triplet_margin_with_distance_loss', "anchor positive negative * distance_function=None margin=1.0 swap=False reduction='mean'", args, kw)
      const m = h.call(nnAttrs.TripletMarginWithDistanceLoss, [], { distance_function: a.distance_function!, margin: a.margin!, swap: a.swap!, reduction: a.reduction! })
      return h.call(m, [a.anchor!, a.positive!, a.negative!])
    }),
    adaptive_avg_pool2d: Ff('adaptive_avg_pool2d', 'output_size', (x, a) => asTT(adaptive(x, a.output_size, 'avg', false))!, 'reduce'),
    adaptive_max_pool2d: fnv('adaptive_max_pool2d', (args, kw) => {
      const a = bindPy(h, 'adaptive_max_pool2d', 'input output_size return_indices=False', args, kw)
      return adaptive(tensorArg(h, a.input, 'adaptive_max_pool2d'), a.output_size, 'max', false, bool(a.return_indices, false))
    }),
    adaptive_avg_pool1d: Ff('adaptive_avg_pool1d', 'output_size', (x, a) => asTT(adaptive(x, a.output_size, 'avg', true))!, 'reduce'),
    adaptive_max_pool1d: fnv('adaptive_max_pool1d', (args, kw) => {
      const a = bindPy(h, 'adaptive_max_pool1d', 'input output_size return_indices=False', args, kw)
      return adaptive(tensorArg(h, a.input, 'adaptive_max_pool1d'), a.output_size, 'max', true, bool(a.return_indices, false))
    }),
    conv1d: convF('conv1d', 1), conv2d: convF('conv2d', 2), conv3d: convF('conv3d', 3),
    conv_transpose1d: convTF('conv_transpose1d', 1), conv_transpose2d: convTF('conv_transpose2d', 2), conv_transpose3d: convTF('conv_transpose3d', 3),
    linear: fnv('linear', (args, kw) => {
      const a = bindPy(h, 'linear', 'input weight bias=None', args, kw)
      const x = tensorArg(h, a.input, 'linear')
      const w = tensorArg(h, a.weight, 'linear', 'weight', 2)
      if (x.dtype !== w.dtype) throw h.err('RuntimeError', `mat1 and mat2 must have the same dtype, but got ${C.typeLabel(x.dtype)} and ${C.typeLabel(w.dtype)}`)
      return wrap(C.linear(E, x, w, isNone(a.bias) ? null : asTT(a.bias!)))
    }),
    bilinear: fnv('bilinear', (args, kw) => {
      const a = bindPy(h, 'bilinear', 'input1 input2 weight bias=None', args, kw)
      return wrap(bilinearT(tensorArg(h, a.input1, 'bilinear', 'input1'), tensorArg(h, a.input2, 'bilinear', 'input2', 2), tensorArg(h, a.weight, 'bilinear', 'weight', 3), isNone(a.bias) ? null : asTT(a.bias!)))
    }),
    cross_entropy: Floss('cross_entropy', "weight=None size_average=None ignore_index=-100 reduce=None reduction='mean' label_smoothing=0.0", (x, y, a) => ceFull(h, x, y, ceOpts(a))),
    nll_loss: Floss('nll_loss', "weight=None size_average=None ignore_index=-100 reduce=None reduction='mean'", (x, y, a) => nllFull(h, x, y, ceOpts(a))),
    mse_loss: Floss('mse_loss', `${LEGACY_LOSS} weight=None`, (x, y, a) => {
      const w = optT(a.weight)
      if (!w) return mse(h, x, y, reduction(a))
      if (w.shape.join() !== x.shape.join()) throw h.err('ValueError', `Weights and input must have the same size, got ${w.shape.join(', ')} and ${x.shape.join(', ')}`)
      const per = mulT(C.unary(E, 'square', subT(x, y)), w)
      const r = reduction(a)
      if (r === 'none') return per
      return r === 'sum' ? C.reduceSum(E, per, null, false, false) : C.binary(E, '/', C.reduceSum(E, per, null, false, false), C.reduceSum(E, w, null, false, false))
    }),
    l1_loss: Floss('l1_loss', LEGACY_LOSS, (x, y, a) => l1(h, x, y, reduction(a))),
    smooth_l1_loss: Floss('smooth_l1_loss', `${LEGACY_LOSS} beta=1.0`, (x, y, a) => smoothL1(h, x, y, h.num(a.beta!), reduction(a))),
    huber_loss: Floss('huber_loss', "reduction='mean' delta=1.0", (x, y, a) => huber(h, x, y, h.num(a.delta!), reduction(a))),
    binary_cross_entropy: Floss('binary_cross_entropy', `weight=None ${LEGACY_LOSS}`, (x, y, a) => bceWeighted(h, x, y, false, optT(a.weight), null, reduction(a))),
    binary_cross_entropy_with_logits: Floss('binary_cross_entropy_with_logits', `weight=None ${LEGACY_LOSS} pos_weight=None`, (x, y, a) => bceWeighted(h, x, y, true, optT(a.weight), optT(a.pos_weight, 'pos_weight'), reduction(a))),
    kl_div: Floss('kl_div', "size_average=None reduce=None reduction='mean' log_target=False", (x, y, a) => klDiv(x, y, h.str(a.reduction!), bool(a.log_target, false))),
    poisson_nll_loss: Floss('poisson_nll_loss', "log_input=True full=False size_average=None eps=1e-08 reduce=None reduction='mean'", (x, y, a) => poissonNll(x, y, bool(a.log_input, true), bool(a.full, false), h.num(a.eps!), reduction(a))),
    gaussian_nll_loss: fnv('gaussian_nll_loss', (args, kw) => {
      const a = bindPy(h, 'gaussian_nll_loss', "input target var full=False eps=1e-06 reduction='mean'", args, kw)
      return wrap(gaussianNll(tensorArg(h, a.input, 'gaussian_nll_loss'), tensorArg(h, a.target, 'gaussian_nll_loss', 'target', 2), tensorArg(h, a.var, 'gaussian_nll_loss', 'var', 3), bool(a.full, false), h.num(a.eps!), reduction(a)))
    }),
    margin_ranking_loss: fnv('margin_ranking_loss', (args, kw) => {
      const a = bindPy(h, 'margin_ranking_loss', `input1 input2 target margin=0.0 ${LEGACY_LOSS}`, args, kw)
      return wrap(marginRanking(tensorArg(h, a.input1, 'margin_ranking_loss', 'input1'), tensorArg(h, a.input2, 'margin_ranking_loss', 'input2', 2), tensorArg(h, a.target, 'margin_ranking_loss', 'target', 3), h.num(a.margin!), reduction(a)))
    }),
    hinge_embedding_loss: Floss('hinge_embedding_loss', `margin=1.0 ${LEGACY_LOSS}`, (x, y, a) => hingeEmbedding(x, y, h.num(a.margin!), reduction(a))),
    cosine_embedding_loss: fnv('cosine_embedding_loss', (args, kw) => {
      const a = bindPy(h, 'cosine_embedding_loss', `input1 input2 target margin=0.0 ${LEGACY_LOSS}`, args, kw)
      return wrap(cosineEmbedding(tensorArg(h, a.input1, 'cosine_embedding_loss', 'input1'), tensorArg(h, a.input2, 'cosine_embedding_loss', 'input2', 2), tensorArg(h, a.target, 'cosine_embedding_loss', 'target', 3), h.num(a.margin!), reduction(a)))
    }),
    triplet_margin_loss: fnv('triplet_margin_loss', (args, kw) => {
      const a = bindPy(h, 'triplet_margin_loss', `anchor positive negative margin=1.0 p=2 eps=1e-06 swap=False ${LEGACY_LOSS}`, args, kw)
      return wrap(tripletMargin(tensorArg(h, a.anchor, 'triplet_margin_loss', 'anchor'), tensorArg(h, a.positive, 'triplet_margin_loss', 'positive', 2), tensorArg(h, a.negative, 'triplet_margin_loss', 'negative', 3), h.num(a.margin!), h.num(a.p!), h.num(a.eps!), bool(a.swap, false), reduction(a)))
    }),
    multi_margin_loss: Floss('multi_margin_loss', `p=1 margin=1.0 weight=None ${LEGACY_LOSS}`, (x, y, a) => multiMargin(x, y, h.num(a.p!), h.num(a.margin!), optT(a.weight), reduction(a))),
    soft_margin_loss: Floss('soft_margin_loss', LEGACY_LOSS, (x, y, a) => softMargin(x, y, reduction(a))),
    multilabel_margin_loss: Floss('multilabel_margin_loss', LEGACY_LOSS, (x, y, a) => multiLabelMargin(x, y, reduction(a))),
    multilabel_soft_margin_loss: Floss('multilabel_soft_margin_loss', `weight=None ${LEGACY_LOSS}`, (x, y, a) => multiLabelSoftMargin(x, y, optT(a.weight), reduction(a))),
    one_hot: fnv('one_hot', (args, kw) => {
      const a = bindPy(h, 'one_hot', 'tensor num_classes=-1', args, kw)
      const t = tensorArg(h, a.tensor, 'one_hot', 'tensor')
      if (t.dtype !== 'int64') throw h.err('RuntimeError', 'one_hot is only applicable to index tensor of type LongTensor.')
      const v = t.values()
      if (v.some((c) => c < 0)) throw h.err('RuntimeError', 'Class values must be non-negative.')
      const given = h.toInt(a.num_classes!)
      const n = given === -1 ? v.reduce((m, x) => Math.max(m, x), -1) + 1 : given
      if (v.some((c) => c >= n)) throw h.err('RuntimeError', 'Class values must be smaller than num_classes.')
      return wrap(TT.of(v.flatMap((c) => Array.from({ length: n }, (_, j) => (j === c ? 1 : 0))), [...t.shape, n], 'int64'))
    }),
    normalize: Ff('normalize', 'p=2.0 dim=1 eps=1e-12', (x, a) => {
      // input / max(‖input‖_p, eps), as PyTorch composes it
      const denom = C.clamp(E, C.vectorNorm(E, x, h.num(a.p!), [h.toInt(a.dim!)], true), h.num(a.eps!), null)
      return C.binary(E, '/', x, C.expandT(E, denom, x.shape))
    }),
    cosine_similarity: fnv('cosine_similarity', (args, kw) => {
      const a = bindPy(h, 'cosine_similarity', 'x1 x2 dim=1 eps=1e-08', args, kw)
      return wrap(cosineSim(tensorArg(h, a.x1, 'cosine_similarity', 'x1'), tensorArg(h, a.x2, 'cosine_similarity', 'x2', 2), h.toInt(a.dim!), h.num(a.eps!)))
    }),
    pairwise_distance: fnv('pairwise_distance', (args, kw) => {
      const a = bindPy(h, 'pairwise_distance', 'x1 x2 p=2.0 eps=1e-06 keepdim=False', args, kw)
      return wrap(pairDistance(tensorArg(h, a.x1, 'pairwise_distance', 'x1'), tensorArg(h, a.x2, 'pairwise_distance', 'x2', 2), h.num(a.p!), h.num(a.eps!), bool(a.keepdim, false)))
    }),
    pdist: Ff('pdist', 'p=2', (x, a) => pdist(x, h.num(a.p!))),
    pad: Ff('pad', "pad mode='constant' value=None", (x, a) => padOf(x, a.pad, a.mode, a.value), 'group'),
    interpolate: Ff('interpolate', "size=None scale_factor=None mode='nearest' align_corners=None recompute_scale_factor=None antialias=False", (x, a) => {
      const mode = h.str(a.mode!)
      if (bool(a.antialias, false)) {
        if (!(x.ndim === 4 && (mode === 'bilinear' || mode === 'bicubic' || mode === 'lanczos'))) throw h.err('ValueError', 'Anti-alias option is restricted to bilinear, bicubic, and lanczos modes and requires a 4-D tensor as input')
        throw h.err('NotImplementedError', 'F.interpolate: antialias=True is not available in this sandbox')
      }
      return interpolate(x, ...recomputed(x, a.size, a.scale_factor, a.recompute_scale_factor), mode, a.align_corners)
    }, 'group'),
    upsample: Ff('upsample', "size=None scale_factor=None mode='nearest' align_corners=None", (x, a) => interpolate(x, a.size, a.scale_factor, h.str(a.mode!), a.align_corners), 'group'),
    upsample_nearest: Ff('upsample_nearest', 'size=None scale_factor=None', (x, a) => interpolate(x, a.size, a.scale_factor, 'nearest', undefined), 'group'),
    upsample_bilinear: Ff('upsample_bilinear', 'size=None scale_factor=None', (x, a) => interpolate(x, a.size, a.scale_factor, 'bilinear', py.bool(true)), 'group'),
    pixel_shuffle: Ff('pixel_shuffle', 'upscale_factor', (x, a) => pixelShuffle(x, h.toInt(a.upscale_factor!))),
    pixel_unshuffle: Ff('pixel_unshuffle', 'downscale_factor', (x, a) => pixelUnshuffle(x, h.toInt(a.downscale_factor!))),
    channel_shuffle: Ff('channel_shuffle', 'groups', (x, a) => channelShuffle(x, h.toInt(a.groups!))),
    unfold: Ff('unfold', 'kernel_size dilation=1 padding=0 stride=1', (x, a) => unfold(x, a.kernel_size!, a.dilation, a.padding, a.stride), 'group'),
    fold: Ff('fold', 'output_size kernel_size dilation=1 padding=0 stride=1', (x, a) => foldT(x, a.output_size!, a.kernel_size!, a.dilation, a.padding, a.stride), 'group'),
    local_response_norm: Ff('local_response_norm', 'size alpha=0.0001 beta=0.75 k=1.0', (x, a) => localResponse(x, h.toInt(a.size!), h.num(a.alpha!), h.num(a.beta!), h.num(a.k!))),
    embedding: fnv('embedding', (args, kw) => {
      const a = bindPy(h, 'embedding', 'input weight padding_idx=None max_norm=None norm_type=2.0 scale_grad_by_freq=False sparse=False', args, kw)
      const w = tensorArg(h, a.weight, 'embedding', 'weight', 2)
      const pi = isNone(a.padding_idx) ? null : h.toInt(a.padding_idx!)
      return wrap(embed(h, w, tensorArg(h, a.input, 'embedding'), pi === null ? null : pi < 0 ? pi + w.shape[0] : pi, isNone(a.max_norm) ? null : h.num(a.max_norm!), h.num(a.norm_type!), bool(a.scale_grad_by_freq, false)))
    }),
    affine_grid: fnv('affine_grid', (args, kw) => {
      const a = bindPy(h, 'affine_grid', 'theta size align_corners=None', args, kw)
      const theta = tensorArg(h, a.theta, 'affine_grid', 'theta')
      if (!isFloatT(theta.dtype)) throw h.err('TypeError', `Expected theta to have floating point type, but got ${theta.dtype}`)
      const size = h.iterate(a.size!).map((v) => h.toInt(v))
      if (size.length !== 4) throw h.err('NotImplementedError', 'affine_grid only supports 4D and 5D sizes, for 2D and 3D affine transforms, respectively. Got size ' + size.join(', ') + '.')
      if (theta.ndim !== 3 || theta.shape[1] !== 2 || theta.shape[2] !== 3) throw h.err('ValueError', `Expected a batch of 2D affine matrices of shape Nx2x3 for size [${size.join(', ')}]. Got [${theta.shape.join(', ')}].`)
      const [N, , H, W] = size
      const corners = bool(a.align_corners, false)
      const line = (n: number) => Array.from({ length: n }, (_, i) => (n <= 1 ? 0 : corners ? -1 + (2 * i) / (n - 1) : (-1 + (2 * i) / n) + 1 / n))
      const [ys, xs] = [line(H), line(W)]
      const base = TT.of(ys.flatMap((y) => xs.flatMap((x) => [x, y, 1])), [H * W, 3], theta.dtype)
      return wrap(C.reshapeT(E, C.matmul(E, C.expandT(E, base, [N, H * W, 3]), C.transposeT(E, theta, 1, 2)), [N, H, W, 2], false))
    }),
    grid_sample: fnv('grid_sample', (args, kw) => {
      const a = bindPy(h, 'grid_sample', "input grid mode='bilinear' padding_mode='zeros' align_corners=None", args, kw)
      const x = tensorArg(h, a.input, 'grid_sample')
      const grid = tensorArg(h, a.grid, 'grid_sample', 'grid', 2)
      const mode = h.str(a.mode!)
      const padding = h.str(a.padding_mode!)
      if (!['bilinear', 'nearest', 'bicubic'].includes(mode)) throw h.err('ValueError', `nn.functional.grid_sample(): expected mode to be 'bilinear', 'nearest' or 'bicubic', but got: '${mode}'`)
      if (!['zeros', 'border', 'reflection'].includes(padding)) throw h.err('ValueError', `nn.functional.grid_sample(): expected padding_mode to be 'zeros', 'border', or 'reflection', but got: '${padding}'`)
      if (mode === 'bicubic') throw h.err('NotImplementedError', "F.grid_sample: mode='bicubic' is not available in this sandbox")
      if (x.ndim !== 4) throw h.err('NotImplementedError', 'F.grid_sample: only 4-D input is available in this sandbox')
      if (grid.requiresGrad) throw h.err('NotImplementedError', 'F.grid_sample: a gradient through the grid is not available in this sandbox')
      if (grid.ndim !== 4 || grid.shape[3] !== 2) throw h.err('RuntimeError', `grid_sampler(): expected grid to have size 2 in last dimension, but got grid with sizes [${grid.shape.join(', ')}]`)
      return wrap(C.gridSampleT(E, x, grid, mode as 'bilinear' | 'nearest', padding as 'zeros' | 'border' | 'reflection', bool(a.align_corners, false)))
    }),
    scaled_dot_product_attention: fnv('scaled_dot_product_attention', (args, kw) => {
      const a = bindPy(h, 'scaled_dot_product_attention', 'query key value attn_mask=None dropout_p=0.0 is_causal=False scale=None', args, kw)
      return wrap(scaledDotProduct(tensorArg(h, a.query, 'scaled_dot_product_attention', 'query'), tensorArg(h, a.key, 'scaled_dot_product_attention', 'key', 2), tensorArg(h, a.value, 'scaled_dot_product_attention', 'value', 3), isNone(a.attn_mask) ? null : asTT(a.attn_mask!), h.num(a.dropout_p!), bool(a.is_causal, false), isNone(a.scale) ? null : h.num(a.scale!)))
    }),
  }

  const pdist = (x: TT, p: number): TT => {
    if (x.ndim !== 2) throw h.err('RuntimeError', `pdist only supports 2D tensors, got: ${x.ndim}D`)
    const N = x.shape[0]
    const rows = Array.from({ length: N }, (_, i) => C.narrowT(E, x, 0, i, 1))
    const out: TT[] = []
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) out.push(C.vectorNorm(E, subT(rows[i], rows[j]), p, null, false))
    return out.length ? named(C.catT(E, out, 0, true), 'PdistBackward0') : TT.of([], [0], x.dtype)
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
      if (std === 0) throw h.err('ZeroDivisionError', 'float division by zero')
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
    dirac_: fnv('dirac_', (args, kw) => {
      const a = bindPy(h, 'dirac_', 'tensor groups=1', args, kw)
      const t = T(a.tensor, 'nn.init.dirac_')
      if (t.ndim < 3 || t.ndim > 5) throw h.err('ValueError', 'Only tensors with 3, 4, or 5 dimensions are supported')
      const groups = h.toInt(a.groups!)
      if (t.shape[0] % groups) throw h.err('ValueError', 'dim 0 must be divisible by groups')
      const out = new Array<number>(t.size).fill(0)
      const st = cStrides(t.shape)
      const per = Math.floor(t.shape[0] / groups)
      const keep = Math.min(per, t.shape[1])
      for (let g = 0; g < groups; g++) for (let d = 0; d < keep; d++) {
        let f = (g * per + d) * st[0] + d * st[1]
        for (let k = 2; k < t.ndim; k++) f += Math.floor(t.shape[k] / 2) * st[k]
        out[f] = 1
      }
      C.writeInto(t, out)
      return wrap(t)
    }),
    orthogonal_: fnv('orthogonal_', (args, kw) => {
      const a = bindPy(h, 'orthogonal_', 'tensor gain=1 generator=None', args, kw)
      const t = T(a.tensor, 'nn.init.orthogonal_')
      if (t.ndim < 2) throw h.err('ValueError', 'Only tensors with 2 or more dimensions are supported')
      if (!t.size) return wrap(t)
      const rng = rngOf(h, a.generator)
      const rows = t.shape[0]
      const cols = t.size / rows
      const raw: L.Mat = Array.from({ length: rows }, () => Array.from({ length: cols }, () => rng.normal()))
      const m = rows < cols ? L.transposeM(raw) : raw
      const { q, r } = L.qr(m)
      const signed = q.map((row) => row.map((x, j) => x * Math.sign(r[j]?.[j] ?? 1)))
      const qq = (rows < cols ? L.transposeM(signed) : signed).map((row) => row.slice(0, cols))
      const gain = h.num(a.gain!)
      C.writeInto(t, qq.flat().map((x) => x * gain))
      return wrap(t)
    }),
    sparse_: fnv('sparse_', (args, kw) => {
      const a = bindPy(h, 'sparse_', 'tensor sparsity std=0.01 generator=None', args, kw)
      const t = T(a.tensor, 'nn.init.sparse_')
      if (t.ndim !== 2) throw h.err('ValueError', 'Only tensors with 2 dimensions are supported')
      const rng = rngOf(h, a.generator)
      const [rows, cols] = t.shape
      const zeros = Math.ceil(h.num(a.sparsity!) * rows)
      const std = h.num(a.std!)
      const out = Array.from({ length: t.size }, () => std * rng.normal())
      for (let c = 0; c < cols; c++) for (const r of shuffled(rng, rows).slice(0, zeros)) out[r * cols + c] = 0
      C.writeInto(t, out)
      return wrap(t)
    }),
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
  // the old spellings, without the underscore
  for (const n of ['uniform', 'normal', 'constant', 'eye', 'dirac', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'orthogonal', 'sparse']) initAttrs[n] = initAttrs[`${n}_`]
  const nnInit: Value = { k: 'module', name: 'torch.nn.init', attrs: initAttrs }

  // ---- torch.nn.utils
  const rnnUtils: Value = {
    k: 'module', name: 'torch.nn.utils.rnn', attrs: {
      pad_sequence: fnv('pad_sequence', (args, kw) => {
        const a = bindPy(h, 'pad_sequence', "sequences batch_first=False padding_value=0.0 padding_side='right'", args, kw)
        const seqs = h.iterate(a.sequences!).map((v) => T(v, 'pad_sequence'))
        if (!seqs.length) throw h.err('IndexError', 'list index out of range')
        const side = h.str(a.padding_side!)
        const longest = seqs.reduce((m, t) => Math.max(m, t.shape[0]), 0)
        const pv = h.num(a.padding_value!)
        const padded = seqs.map((t) => {
          const lack = longest - t.shape[0]
          if (!lack) return t
          const fill = TT.of(new Array<number>(lack * prod(t.shape.slice(1))).fill(pv), [lack, ...t.shape.slice(1)], t.dtype)
          return C.catT(E, side === 'left' ? [fill, t] : [t, fill], 0)
        })
        return wrap(C.catT(E, padded, bool(a.batch_first, false) ? 0 : 1, true))
      }),
      unpad_sequence: fnv('unpad_sequence', (args, kw) => {
        const a = bindPy(h, 'unpad_sequence', 'padded_sequences lengths batch_first=False', args, kw)
        const t = T(a.padded_sequences, 'unpad_sequence')
        const lens = h.iterate(a.lengths!).map((v) => h.toInt(v))
        const batchFirst = bool(a.batch_first, false)
        return py.list(lens.map((n, i) => {
          const row = C.narrowT(E, t, batchFirst ? 0 : 1, i, 1)
          return wrap(C.narrowT(E, C.squeezeT(E, row, batchFirst ? 0 : 1), 0, 0, n))
        }))
      }),
    },
  }
  const gradsOf = (params: Value): TT[] => (asTT(params) ? [asTT(params)!] : h.iterate(params).map((p) => T(p, 'parameter'))).filter((p) => p.grad)
  const nnUtilsBase: Record<string, Value> = {
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
      rnn: rnnUtils,
      clip_grad_norm: fnv('clip_grad_norm', (args, kw) => h.call((nnUtilsBase.clip_grad_norm_), args, kw)),
      vector_to_parameters: fnv('vector_to_parameters', (args, kw) => {
        const a = bindPy(h, 'vector_to_parameters', 'vec parameters', args, kw)
        const v = asTT(a.vec!)
        if (!v) throw h.err('TypeError', `expected torch.Tensor, but got: ${typeName(a.vec!)}`)
        const vals = v.values()
        let at = 0
        for (const p of h.iterate(a.parameters!)) {
          const t = T(p, 'parameter')
          C.writeInto(t, vals.slice(at, at + t.size))
          at += t.size
        }
        return py.NONE
      }),
      get_total_norm: fnv('get_total_norm', (args, kw) => {
        const a = bindPy(h, 'get_total_norm', 'tensors norm_type=2.0 error_if_nonfinite=False foreach=None', args, kw)
        const ts = (asTT(a.tensors!) ? [asTT(a.tensors!)!] : h.iterate(a.tensors!).map((t) => T(t, 'tensor')))
        const p = h.num(a.norm_type!)
        const norms = ts.map((t) => C.vectorNorm({ grad: false } as Engine, t, p, null, false).values()[0])
        return wrap(C.vectorNorm({ grad: false } as Engine, TT.of(norms, [norms.length], ts[0]?.dtype ?? 'float32'), p, null, false))
      }),
  }
  const nnUtils: Value = { k: 'module', name: 'torch.nn.utils', attrs: nnUtilsBase }

  // ---- optim
  const invalid = (a: Args, key: string, what: string, bad: (x: number) => boolean) => {
    const v = a[key]
    if (v && v.k !== 'tuple' && bad(h.num(v))) throw h.err('ValueError', `Invalid ${what}: ${repr(v)}`)
  }
  const nonNegative = (a: Args, key: string, what: string) => invalid(a, key, what, (x) => x < 0)
  const checkBetas = (a: Args) => {
    h.iterate(a.betas!).forEach((v, i) => {
      if (h.num(v) < 0 || h.num(v) >= 1) throw h.err('ValueError', `Invalid beta parameter at index ${i}: ${repr(v)}`)
    })
  }
  /** an optimiser from its real signature; order: its defaults dict, as PyTorch fills it */
  const optimizer = (name: string, params: string, order: string, check: (a: Args) => void): Value => ({ k: 'type', name, call: (args, kw) => {
    const a = bindPy(h, `${name}.__init__`, `params ${params}`, args, kw)
    nonNegative(a, 'lr', 'learning rate')
    check(a)
    refuse(name, a, 'differentiable', isFalse)
    return py.obj(new Optimizer(name, h, a.params!, order.split(' ').map((k) => [k, a[k]!])))
  } })
  const wdAndEps = (a: Args) => {
    nonNegative(a, 'eps', 'epsilon value')
    nonNegative(a, 'weight_decay', 'weight_decay value')
  }
  const adamCheck = (a: Args) => {
    nonNegative(a, 'lr', 'learning rate')
    nonNegative(a, 'eps', 'epsilon value')
    checkBetas(a)
    nonNegative(a, 'weight_decay', 'weight_decay value')
  }
  const ADAM_ORDER = 'lr betas eps weight_decay amsgrad maximize foreach capturable differentiable fused decoupled_weight_decay'
  const optimizerHost: HostClass = {
    init(inst, args, kw, hh) {
      const a = bindPy(hh, 'Optimizer.__init__', 'params defaults', args, kw)
      const defaults = a.defaults?.k === 'dict' ? a.defaults.d.items().map(([k, v]): [string, Value] => [hh.str(k), v]) : []
      inst.state.optimizer = new Optimizer('Optimizer', hh, a.params!, defaults)
    },
    getAttr(inst, name, hh) {
      const o = inst.state.optimizer as Optimizer | undefined
      if (!o) return undefined
      return o.getAttr(name, hh)
    },
  }
  const optim: Record<string, Value> = {
    Optimizer: { k: 'type', name: 'Optimizer', host: optimizerHost, call: (args, kw) => {
      const a = bindPy(h, 'Optimizer.__init__', 'params defaults', args, kw)
      return py.obj(new Optimizer('Optimizer', h, a.params!, a.defaults?.k === 'dict' ? a.defaults.d.items().map(([k, v]): [string, Value] => [h.str(k), v]) : []))
    } },
    SGD: optimizer('SGD', 'lr=0.001 momentum=0 dampening=0 weight_decay=0 nesterov=False * maximize=False foreach=None differentiable=False fused=None', 'lr momentum dampening weight_decay nesterov maximize foreach differentiable fused', (a) => {
      nonNegative(a, 'momentum', 'momentum value')
      nonNegative(a, 'weight_decay', 'weight_decay value')
      if (bool(a.nesterov, false) && (num(a.momentum, 0) <= 0 || num(a.dampening, 0) !== 0)) throw h.err('ValueError', 'Nesterov momentum requires a momentum and zero dampening')
    }),
    Adam: optimizer('Adam', 'lr=0.001 betas=(0.9,0.999) eps=1e-08 weight_decay=0 amsgrad=False * foreach=None maximize=False capturable=False differentiable=False fused=None decoupled_weight_decay=False', ADAM_ORDER, adamCheck),
    AdamW: ((): Value => {
      const made = optimizer('AdamW', 'lr=0.001 betas=(0.9,0.999) eps=1e-08 weight_decay=0.01 amsgrad=False * maximize=False foreach=None capturable=False differentiable=False fused=None', 'lr betas eps weight_decay amsgrad maximize foreach capturable differentiable fused', adamCheck) as { k: 'type'; call: (args: Value[], kw: Kw) => Value }
      return { ...made, call: (args, kw) => {
        const o = made.call(args, kw)
        // AdamW is Adam with this one flag up
        const opt = (o as unknown as { o: Optimizer }).o
        opt.defaults.push(['decoupled_weight_decay', py.bool(true)])
        for (const g of (opt.groups as { items: Value[] }).items) (g as { d: PyDict }).d.set(py.str('decoupled_weight_decay'), py.bool(true))
        return o
      } } as Value
    })(),
    RMSprop: optimizer('RMSprop', 'lr=0.01 alpha=0.99 eps=1e-08 weight_decay=0 momentum=0 centered=False capturable=False foreach=None maximize=False differentiable=False', 'lr momentum alpha eps centered weight_decay capturable foreach maximize differentiable', (a) => {
      nonNegative(a, 'eps', 'epsilon value')
      nonNegative(a, 'momentum', 'momentum value')
      nonNegative(a, 'weight_decay', 'weight_decay value')
      nonNegative(a, 'alpha', 'alpha value')
    }),
    Adagrad: optimizer('Adagrad', 'lr=0.01 lr_decay=0 weight_decay=0 initial_accumulator_value=0 eps=1e-10 foreach=None * maximize=False differentiable=False fused=None', 'lr lr_decay eps weight_decay initial_accumulator_value foreach maximize differentiable fused', (a) => {
      nonNegative(a, 'lr_decay', 'lr_decay value')
      nonNegative(a, 'weight_decay', 'weight_decay value')
      nonNegative(a, 'initial_accumulator_value', 'initial_accumulator_value value')
      nonNegative(a, 'eps', 'epsilon value')
    }),
    Adadelta: optimizer('Adadelta', 'lr=1.0 rho=0.9 eps=1e-06 weight_decay=0 foreach=None * capturable=False maximize=False differentiable=False', 'lr rho eps weight_decay maximize capturable foreach differentiable', (a) => {
      invalid(a, 'rho', 'rho value', (x) => x < 0 || x > 1)
      wdAndEps(a)
    }),
    Adamax: optimizer('Adamax', 'lr=0.002 betas=(0.9,0.999) eps=1e-08 weight_decay=0 foreach=None * maximize=False differentiable=False capturable=False', 'lr betas eps weight_decay foreach maximize differentiable capturable', (a) => {
      nonNegative(a, 'eps', 'epsilon value')
      checkBetas(a)
      nonNegative(a, 'weight_decay', 'weight_decay value')
    }),
    NAdam: optimizer('NAdam', 'lr=0.002 betas=(0.9,0.999) eps=1e-08 weight_decay=0 momentum_decay=0.004 decoupled_weight_decay=False * foreach=None maximize=False capturable=False differentiable=False', 'lr betas eps weight_decay momentum_decay decoupled_weight_decay maximize foreach capturable differentiable', (a) => {
      nonNegative(a, 'eps', 'epsilon value')
      checkBetas(a)
      nonNegative(a, 'weight_decay', 'weight_decay value')
      nonNegative(a, 'momentum_decay', 'momentum_decay value')
    }),
    RAdam: optimizer('RAdam', 'lr=0.001 betas=(0.9,0.999) eps=1e-08 weight_decay=0 decoupled_weight_decay=False * foreach=None maximize=False capturable=False differentiable=False', 'lr betas eps weight_decay maximize foreach capturable decoupled_weight_decay differentiable', (a) => {
      nonNegative(a, 'eps', 'epsilon value')
      checkBetas(a)
      nonNegative(a, 'weight_decay', 'weight_decay value')
    }),
    ASGD: optimizer('ASGD', 'lr=0.01 lambd=0.0001 alpha=0.75 t0=1000000.0 weight_decay=0 foreach=None maximize=False differentiable=False capturable=False', 'lr lambd alpha t0 weight_decay foreach maximize differentiable capturable', (a) => nonNegative(a, 'weight_decay', 'weight_decay value')),
    Rprop: optimizer('Rprop', 'lr=0.01 etas=(0.5,1.2) step_sizes=(1e-06,50) * capturable=False foreach=None maximize=False differentiable=False', 'lr etas step_sizes foreach maximize differentiable capturable', (a) => {
      const [m, p] = h.iterate(a.etas!).map((v) => h.num(v))
      if (!(m > 0 && m < 1 && 1 < p)) throw h.err('ValueError', `Invalid eta values: ${m}, ${p}`)
    }),
    LBFGS: { k: 'type', name: 'LBFGS', call: (args, kw) => {
      const a = bindPy(h, 'LBFGS.__init__', 'params lr=1 max_iter=20 max_eval=None tolerance_grad=1e-07 tolerance_change=1e-09 history_size=100 line_search_fn=None * maximize=False', args, kw)
      if (h.num(a.lr!) < 0) throw h.err('ValueError', `Invalid learning rate: ${repr(a.lr!)}`)
      if (!isNone(a.line_search_fn)) throw h.err('NotImplementedError', "LBFGS: line_search_fn='strong_wolfe' is not available in this sandbox")
      if (isNone(a.max_eval)) a.max_eval = py.int(Math.floor((h.toInt(a.max_iter!) * 5) / 4))
      const o = new Optimizer('LBFGS', h, a.params!, 'lr max_iter max_eval tolerance_grad tolerance_change history_size line_search_fn maximize'.split(' ').map((k) => [k, a[k]!]))
      if ((o.groups as { items: Value[] }).items.length !== 1) throw h.err('ValueError', "LBFGS doesn't support per-parameter options (parameter groups)")
      return py.obj(o)
    } },
  }
  // ---- torch.optim.lr_scheduler: ported from lr_scheduler.py, chained forms and all
  const optimizerArg = (v: Value | undefined): Optimizer => {
    if (v?.k !== 'obj' || !(v.o instanceof Optimizer)) {
      if (v?.k === 'inst' && v.state.optimizer instanceof Optimizer) return v.state.optimizer
      throw h.err('TypeError', `${v ? typeName(v) : 'NoneType'} is not an Optimizer`)
    }
    return v.o
  }
  const schedType = (name: string, params: string, make: (a: Args, o: Optimizer) => { vals: [string, Value][]; rule: LrRule; closed?: LrRule; tail?: [string, Value][] }): Value => ({ k: 'type', name, call: (args, kw) => {
    const a = bindPy(h, `${name}.__init__`, `optimizer ${params} last_epoch=-1`, args, kw)
    const o = optimizerArg(a.optimizer)
    const { vals, rule, closed, tail } = make(a, o)
    const sc = new Scheduler(name, h, o, h.toInt(a.last_epoch!), vals, rule, closed ?? null)
    if (tail) sc.tail = tail
    return py.obj(sc.start())
  } })
  const lambdas = (a: Args, o: Optimizer): Value[] => {
    const n = (o.groups as { items: Value[] }).items.length
    const v = a.lr_lambda!
    if (v.k !== 'list' && v.k !== 'tuple') return new Array<Value>(n).fill(v)
    if (v.items.length !== n) throw h.err('ValueError', `Expected ${n} lr_lambdas, but got ${v.items.length}`)
    return v.items
  }
  /** _format_param: one value for every group, or a list of them */
  const perGroup = (name: string, o: Optimizer, v: Value): number[] => {
    const n = (o.groups as { items: Value[] }).items.length
    if (v.k !== 'list' && v.k !== 'tuple') return new Array<number>(n).fill(h.num(v))
    if (v.items.length !== n) throw h.err('ValueError', `Expected ${n} values for ${name}, got ${v.items.length}`)
    return v.items.map((x) => h.num(x))
  }
  const pyMod = (a: number, b: number) => ((a % b) + b) % b
  /** cycle_momentum: where the momentum (or Adam's beta1) goes on each step */
  const momentumOf = (o: Optimizer) => {
    const hasBetas = o.defaults.some(([k]) => k === 'betas')
    if (!hasBetas && !o.defaults.some(([k]) => k === 'momentum')) throw h.err('ValueError', 'optimizer must support momentum or beta1 with `cycle_momentum` option enabled')
    return (d: PyDict, m: number) => {
      if (hasBetas) d.set(py.str('betas'), py.tuple([py.float(m), ...h.iterate(d.get(py.str('betas'))!).slice(1)]))
      else d.set(py.str('momentum'), py.float(m))
    }
  }
  const groupDicts = (o: Optimizer) => (o.groups as { items: Value[] }).items.map((g) => (g as { d: PyDict }).d)
  const schedulerHost: HostClass = {
    init(inst, args, kw, hh) {
      const a = bindPy(hh, 'LRScheduler.__init__', 'optimizer last_epoch=-1', args, kw)
      const sc = new Scheduler(inst.cls.name, hh, optimizerArg(a.optimizer), hh.toInt(a.last_epoch!), [], () => hh.iterate(hh.call(hh.getAttr(inst, 'get_lr'), [])).map((v) => hh.num(v)))
      inst.state.scheduler = sc
      sc.start()
    },
    getAttr(inst, name, hh) {
      const sc = inst.state.scheduler as Scheduler | undefined
      return sc?.getAttr(name, hh)
    },
  }
  const lrScheduler: Value = {
    k: 'module', name: 'torch.optim.lr_scheduler', attrs: {
      LRScheduler: { k: 'type', name: 'LRScheduler', host: schedulerHost, call: () => { throw h.err('NotImplementedError', '') } },
      _LRScheduler: { k: 'type', name: 'LRScheduler', host: schedulerHost, call: () => { throw h.err('NotImplementedError', '') } },
      StepLR: schedType('StepLR', 'step_size gamma=0.1', (a) => ({
        vals: [['step_size', a.step_size!], ['gamma', a.gamma!]],
        rule: (s) => (s.lastEpoch === 0 || pyMod(s.lastEpoch, s.n('step_size')) !== 0 ? s.lrs() : s.lrs().map((lr) => lr * s.n('gamma'))),
        closed: (s) => s.baseLrs.map((b) => b * s.n('gamma') ** Math.floor(s.lastEpoch / s.n('step_size'))),
      })),
      MultiStepLR: schedType('MultiStepLR', 'milestones gamma=0.1', (a) => {
        const counts = new PyDict()
        for (const m of h.iterate(a.milestones!)) counts.set(m, py.int(((counts.get(m) as { v: number } | undefined)?.v ?? 0) + 1))
        const list = counts.items().flatMap(([m, c]) => new Array<number>((c as { v: number }).v).fill(h.num(m))).sort((p, q) => p - q)
        return {
          vals: [['milestones', { k: 'dict', d: counts, flavor: 'Counter' }], ['gamma', a.gamma!]],
          rule: (s) => {
            const c = counts.get(py.int(s.lastEpoch))
            return c ? s.lrs().map((lr) => lr * s.n('gamma') ** h.num(c)) : s.lrs()
          },
          closed: (s) => s.baseLrs.map((b) => b * s.n('gamma') ** list.filter((m) => m <= s.lastEpoch).length),
        }
      }),
      ConstantLR: schedType('ConstantLR', 'factor=0.3333333333333333 total_iters=5', (a) => {
        if (h.num(a.factor!) > 1 || h.num(a.factor!) < 0) throw h.err('ValueError', 'Constant multiplicative factor expected to be between 0 and 1.')
        return {
          vals: [['factor', a.factor!], ['total_iters', a.total_iters!]],
          rule: (s) => {
            if (s.lastEpoch === 0) return s.lrs().map((lr) => lr * s.n('factor'))
            if (s.lastEpoch !== s.n('total_iters')) return s.lrs()
            return s.lrs().map((lr) => lr * (1 / s.n('factor')))
          },
          closed: (s) => s.baseLrs.map((b) => b * (s.n('factor') + (s.lastEpoch >= s.n('total_iters') ? 1 : 0) * (1 - s.n('factor')))),
        }
      }),
      LinearLR: schedType('LinearLR', 'start_factor=0.3333333333333333 end_factor=1.0 total_iters=5', (a) => {
        if (h.num(a.start_factor!) > 1 || h.num(a.start_factor!) <= 0) throw h.err('ValueError', 'Starting multiplicative factor expected to be greater than 0 and less or equal to 1.')
        if (h.num(a.end_factor!) > 1 || h.num(a.end_factor!) < 0) throw h.err('ValueError', 'Ending multiplicative factor expected to be between 0 and 1.')
        return {
          vals: [['start_factor', a.start_factor!], ['end_factor', a.end_factor!], ['total_iters', a.total_iters!]],
          rule: (s) => {
            const [sf, ef, ti] = [s.n('start_factor'), s.n('end_factor'), s.n('total_iters')]
            if (s.lastEpoch === 0) return s.lrs().map((lr) => lr * sf)
            if (s.isInitial || s.lastEpoch > ti) return s.lrs()
            return s.lrs().map((lr) => lr * (1 + (ef - sf) / (ti * sf + (s.lastEpoch - 1) * (ef - sf))))
          },
          closed: (s) => s.baseLrs.map((b) => b * (s.n('start_factor') + ((s.n('end_factor') - s.n('start_factor')) * Math.min(s.n('total_iters'), s.lastEpoch)) / s.n('total_iters'))),
        }
      }),
      ExponentialLR: schedType('ExponentialLR', 'gamma', (a) => ({
        vals: [['gamma', a.gamma!]],
        rule: (s) => (s.isInitial ? s.lrs() : s.lrs().map((lr) => lr * s.n('gamma'))),
        closed: (s) => s.baseLrs.map((b) => b * s.n('gamma') ** s.lastEpoch),
      })),
      PolynomialLR: schedType('PolynomialLR', 'total_iters=5 power=1.0', (a) => ({
        vals: [['total_iters', a.total_iters!], ['power', a.power!]],
        rule: (s) => {
          const ti = s.n('total_iters')
          if (s.isInitial || s.lastEpoch > ti) return s.lrs()
          const f = ((1 - s.lastEpoch / ti) / (1 - (s.lastEpoch - 1) / ti)) ** s.n('power')
          return s.lrs().map((lr) => lr * f)
        },
        closed: (s) => s.baseLrs.map((b) => b * (1 - Math.min(s.n('total_iters'), s.lastEpoch) / s.n('total_iters')) ** s.n('power')),
      })),
      CosineAnnealingLR: schedType('CosineAnnealingLR', 'T_max eta_min=0.0', (a) => ({
        vals: [['T_max', a.T_max!], ['eta_min', a.eta_min!]],
        rule: (s) => {
          const [T, eta, e] = [s.n('T_max'), s.n('eta_min'), s.lastEpoch]
          if (s.isInitial) return s.lrs()
          if (s.stepCount === 1 && e > 0) return s.baseLrs.map((b) => eta + ((b - eta) * (1 + Math.cos((e * Math.PI) / T))) / 2)
          if (pyMod(e - 1 - T, 2 * T) === 0) return s.lrs().map((lr, i) => lr + ((s.baseLrs[i] - eta) * (1 - Math.cos(Math.PI / T))) / 2)
          return s.lrs().map((lr) => ((1 + Math.cos((Math.PI * e) / T)) / (1 + Math.cos((Math.PI * (e - 1)) / T))) * (lr - eta) + eta)
        },
        closed: (s) => s.baseLrs.map((b) => s.n('eta_min') + ((b - s.n('eta_min')) * (1 + Math.cos((Math.PI * s.lastEpoch) / s.n('T_max')))) / 2),
      })),
      LambdaLR: schedType('LambdaLR', 'lr_lambda', (a, o) => {
        const fs = lambdas(a, o)
        return { vals: [], rule: (s) => s.baseLrs.map((b, i) => b * h.num(h.call(fs[i], [py.int(s.lastEpoch)]))), tail: [['lr_lambdas', py.list(fs.map(() => py.NONE))]] }
      }),
      MultiplicativeLR: schedType('MultiplicativeLR', 'lr_lambda', (a, o) => {
        const fs = lambdas(a, o)
        return { vals: [], rule: (s) => (s.isInitial ? s.lrs() : s.lrs().map((lr, i) => lr * h.num(h.call(fs[i], [py.int(s.lastEpoch)])))), tail: [['lr_lambdas', py.list(fs.map(() => py.NONE))]] }
      }),
      CosineAnnealingWarmRestarts: { k: 'type', name: 'CosineAnnealingWarmRestarts', call: (args, kw) => {
        const a = bindPy(h, 'CosineAnnealingWarmRestarts.__init__', 'optimizer T_0 T_mult=1 eta_min=0.0 last_epoch=-1', args, kw)
        if (a.T_0!.k !== 'int' || a.T_0!.v <= 0) throw h.err('ValueError', `Expected positive integer T_0, but got ${repr(a.T_0!)}`)
        if (a.T_mult!.k !== 'int' || a.T_mult!.v < 1) throw h.err('ValueError', `Expected integer T_mult >= 1, but got ${repr(a.T_mult!)}`)
        const last = h.toInt(a.last_epoch!)
        const [T0, mult] = [a.T_0!.v, a.T_mult!.v]
        let Ti = T0
        let cur = last
        const rule: LrRule = (s) => s.baseLrs.map((b) => s.n('eta_min') + ((b - s.n('eta_min')) * (1 + Math.cos((Math.PI * cur) / Ti))) / 2)
        const sc = new Scheduler('CosineAnnealingWarmRestarts', h, optimizerArg(a.optimizer), last, [['T_0', a.T_0!], ['T_mult', a.T_mult!], ['eta_min', a.eta_min!]], rule)
        sc.customStep = (s, given) => {
          let epoch = given === null && s.lastEpoch < 0 ? 0 : given
          if (epoch === null) {
            epoch = s.lastEpoch + 1
            cur += 1
            if (cur >= Ti) {
              cur = pyMod(cur, Ti)
              Ti *= mult
            }
          } else {
            if (epoch < 0) throw h.err('ValueError', `Expected non-negative epoch, but got ${epoch}`)
            if (epoch >= T0) {
              if (mult === 1) cur = pyMod(epoch, T0)
              else {
                const n = Math.trunc(Math.log((epoch / T0) * (mult - 1) + 1) / Math.log(mult))
                cur = epoch - (T0 * (mult ** n - 1)) / (mult - 1)
                Ti = T0 * mult ** n
              }
            } else {
              Ti = T0
              cur = epoch
            }
          }
          s.lastEpoch = Math.floor(epoch)
          s.write(rule(s))
        }
        return py.obj(sc.start())
      } },
      CyclicLR: { k: 'type', name: 'CyclicLR', call: (args, kw) => {
        const a = bindPy(h, 'CyclicLR.__init__', "optimizer base_lr max_lr step_size_up=2000 step_size_down=None mode='triangular' gamma=1.0 scale_fn=None scale_mode='cycle' cycle_momentum=True base_momentum=0.8 max_momentum=0.9 last_epoch=-1", args, kw)
        const o = optimizerArg(a.optimizer)
        const last = h.toInt(a.last_epoch!)
        const bases = perGroup('base_lr', o, a.base_lr!)
        if (last === -1) groupDicts(o).forEach((d, i) => d.set(py.str('lr'), py.float(bases[i])))
        const maxLrs = perGroup('max_lr', o, a.max_lr!)
        const up = h.num(a.step_size_up!)
        const down = isNone(a.step_size_down) ? up : h.num(a.step_size_down!)
        const total = up + down
        const ratio = up / total
        const mode = h.str(a.mode!)
        const custom = isNone(a.scale_fn) ? null : a.scale_fn!
        if (!['triangular', 'triangular2', 'exp_range'].includes(mode) && !custom) throw h.err('ValueError', 'mode is invalid and scale_fn is None')
        const gamma = h.num(a.gamma!)
        const byCycle = custom ? h.str(a.scale_mode!) === 'cycle' : mode !== 'exp_range'
        const scale = (x: number) => (custom ? h.num(h.call(custom, [py.float(x)])) : mode === 'triangular' ? 1 : mode === 'triangular2' ? 1 / 2 ** (x - 1) : gamma ** x)
        const cycling = bool(a.cycle_momentum, true)
        const setMomentum = cycling ? momentumOf(o) : null
        const baseM = cycling ? perGroup('base_momentum', o, a.base_momentum!) : []
        const maxM = cycling ? perGroup('max_momentum', o, a.max_momentum!) : []
        if (cycling && last === -1) groupDicts(o).forEach((d, i) => {
          setMomentum!(d, maxM[i])
          d.set(py.str('max_momentum'), py.float(maxM[i]))
          d.set(py.str('base_momentum'), py.float(baseM[i]))
        })
        const sc = new Scheduler('CyclicLR', h, o, last, [['max_lrs', py.list(maxLrs.map(py.float))], ['total_size', py.float(total)], ['step_ratio', py.float(ratio)], ['mode', a.mode!], ['gamma', a.gamma!], ['scale_mode', py.str(byCycle ? 'cycle' : 'iterations')], ['cycle_momentum', py.bool(cycling)]], (s) => {
          const cycle = Math.floor(1 + s.lastEpoch / total)
          const x = 1 + s.lastEpoch / total - cycle
          const f = x <= ratio ? x / ratio : (x - 1) / (ratio - 1)
          const by = byCycle ? cycle : s.lastEpoch
          if (cycling) groupDicts(o).forEach((d, i) => setMomentum!(d, maxM[i] - (maxM[i] - baseM[i]) * f * scale(by)))
          return s.baseLrs.map((b, i) => b + (maxLrs[i] - b) * f * scale(by))
        })
        sc.baseLrs = bases
        return py.obj(sc.start())
      } },
      OneCycleLR: { k: 'type', name: 'OneCycleLR', call: (args, kw) => {
        const a = bindPy(h, 'OneCycleLR.__init__', "optimizer max_lr total_steps=None epochs=None steps_per_epoch=None pct_start=0.3 anneal_strategy='cos' cycle_momentum=True base_momentum=0.85 max_momentum=0.95 div_factor=25.0 final_div_factor=10000.0 three_phase=False last_epoch=-1", args, kw)
        const o = optimizerArg(a.optimizer)
        let total: number
        if (!isNone(a.total_steps)) {
          if (a.total_steps!.k !== 'int' || a.total_steps!.v <= 0) throw h.err('ValueError', `Expected positive integer total_steps, but got ${repr(a.total_steps!)}`)
          total = a.total_steps!.v
        } else if (!isNone(a.epochs) && !isNone(a.steps_per_epoch)) {
          if (a.epochs!.k !== 'int' || a.epochs!.v <= 0) throw h.err('ValueError', `Expected positive integer epochs, but got ${repr(a.epochs!)}`)
          if (a.steps_per_epoch!.k !== 'int' || a.steps_per_epoch!.v <= 0) throw h.err('ValueError', `Expected positive integer steps_per_epoch, but got ${repr(a.steps_per_epoch!)}`)
          total = a.epochs!.v * a.steps_per_epoch!.v
        } else throw h.err('ValueError', 'You must define either total_steps OR (epochs AND steps_per_epoch)')
        const pct = a.pct_start!.k === 'float' ? a.pct_start!.v : -1
        if (pct < 0 || pct > 1) throw h.err('ValueError', `Expected float between 0 and 1 pct_start, but got ${repr(a.pct_start!)}`)
        const how = h.str(a.anneal_strategy!)
        if (how !== 'cos' && how !== 'linear') throw h.err('ValueError', `anneal_strategy must be one of 'cos' or 'linear', instead got ${how}`)
        const anneal = (from: number, to: number, f: number) => (how === 'cos' ? to + ((from - to) / 2) * (Math.cos(Math.PI * f) + 1) : (to - from) * f + from)
        const phases: [number, string, string, string, string][] = bool(a.three_phase, false)
          ? [[pct * total - 1, 'initial_lr', 'max_lr', 'max_momentum', 'base_momentum'], [2 * pct * total - 2, 'max_lr', 'initial_lr', 'base_momentum', 'max_momentum'], [total - 1, 'initial_lr', 'min_lr', 'max_momentum', 'max_momentum']]
          : [[pct * total - 1, 'initial_lr', 'max_lr', 'max_momentum', 'base_momentum'], [total - 1, 'max_lr', 'min_lr', 'base_momentum', 'max_momentum']]
        const last = h.toInt(a.last_epoch!)
        const maxLrs = perGroup('max_lr', o, a.max_lr!)
        if (last === -1) groupDicts(o).forEach((d, i) => {
          const init = maxLrs[i] / h.num(a.div_factor!)
          d.set(py.str('initial_lr'), py.float(init))
          d.set(py.str('max_lr'), py.float(maxLrs[i]))
          d.set(py.str('min_lr'), py.float(init / h.num(a.final_div_factor!)))
        })
        const cycling = bool(a.cycle_momentum, true)
        const setMomentum = cycling ? momentumOf(o) : null
        if (cycling && last === -1) {
          const [maxM, baseM] = [perGroup('max_momentum', o, a.max_momentum!), perGroup('base_momentum', o, a.base_momentum!)]
          groupDicts(o).forEach((d, i) => {
            setMomentum!(d, maxM[i])
            d.set(py.str('max_momentum'), py.float(maxM[i]))
            d.set(py.str('base_momentum'), py.float(baseM[i]))
          })
        }
        const sc = new Scheduler('OneCycleLR', h, o, last, [['total_steps', py.int(total)], ['cycle_momentum', py.bool(cycling)]], (s) => {
          const step = s.lastEpoch
          if (step > total) throw h.err('ValueError', `Tried to step ${step} times. The specified number of total steps is ${total}`)
          return groupDicts(o).map((d) => {
            const g = (k: string) => h.num(d.get(py.str(k))!)
            let start = 0
            for (let i = 0; i < phases.length; i++) {
              const [end, lr0, lr1, m0, m1] = phases[i]
              if (step <= end || i === phases.length - 1) {
                const f = (step - start) / (end - start)
                if (cycling) setMomentum!(d, anneal(g(m0), g(m1), f))
                return anneal(g(lr0), g(lr1), f)
              }
              start = end
            }
            return g('lr')
          })
        })
        return py.obj(sc.start())
      } },
      SequentialLR: { k: 'type', name: 'SequentialLR', call: (args, kw) => {
        const a = bindPy(h, 'SequentialLR.__init__', 'optimizer schedulers milestones last_epoch=-1', args, kw)
        const o = optimizerArg(a.optimizer)
        const list = h.iterate(a.schedulers!)
        if (!list.length) throw h.err('ValueError', 'SequentialLR expects at least one scheduler, but got no scheduler.')
        const scheds = list.map((v, i) => schedulerOf(v, 'SequentialLR', i, o))
        const ms = h.iterate(a.milestones!).map((v) => h.toInt(v))
        if (ms.length !== scheds.length - 1) throw h.err('ValueError', `Sequential Schedulers expects number of schedulers provided to be one more than the number of milestone points, but got number of schedulers ${scheds.length} and the number of milestones to be equal to ${ms.length}`)
        return py.obj(new SchedulerGroup('SequentialLR', h, o, scheds, ms, h.toInt(a.last_epoch!)))
      } },
      ChainedScheduler: { k: 'type', name: 'ChainedScheduler', call: (args, kw) => {
        const a = bindPy(h, 'ChainedScheduler.__init__', 'schedulers optimizer=None', args, kw)
        const list = h.iterate(a.schedulers!)
        if (!list.length) throw h.err('ValueError', 'ChainedScheduler expects at least one scheduler to be chained, but got no scheduler.')
        const o = isNone(a.optimizer) ? (list[0] as unknown as { o: Scheduler }).o.opt : optimizerArg(a.optimizer)
        return py.obj(new SchedulerGroup('ChainedScheduler', h, o, list.map((v, i) => schedulerOf(v, 'ChainedScheduler', i, o)), null, 0))
      } },
      ReduceLROnPlateau: { k: 'type', name: 'ReduceLROnPlateau', call: (args, kw) => {
        const a = bindPy(h, 'ReduceLROnPlateau.__init__', "optimizer mode='min' factor=0.1 patience=10 threshold=0.0001 threshold_mode='rel' cooldown=0 min_lr=0 eps=1e-08", args, kw)
        const o = optimizerArg(a.optimizer)
        const mode = h.str(a.mode!)
        if (mode !== 'min' && mode !== 'max') throw h.err('ValueError', `mode ${mode} is unknown!`)
        const tm = h.str(a.threshold_mode!)
        if (tm !== 'rel' && tm !== 'abs') throw h.err('ValueError', `threshold mode ${tm} is unknown!`)
        return py.obj(new ReduceLROnPlateau(h, o, mode === 'max', h.num(a.factor!), h.toInt(a.patience!), h.num(a.threshold!), tm === 'abs', h.toInt(a.cooldown!), a.min_lr, h.num(a.eps!)))
      } },
    },
  }
  function schedulerOf(v: Value, who: string, i: number, o: Optimizer): Scheduler {
    if (v.k === 'obj' && v.o instanceof ReduceLROnPlateau) throw h.err('ValueError', `${who} does not support \`ReduceLROnPlateau\` scheduler as it requires additional kwargs to be specified when calling \`step\`, but got one at index ${i} in the given schedulers sequence.`)
    if (v.k !== 'obj' || !(v.o instanceof Scheduler)) throw h.err('TypeError', `${who} at index ${i} should have \`optimizer\` as its attribute.`)
    if (v.o.opt !== o) throw h.err('ValueError', `${who} expects all schedulers to belong to the same optimizer, but got scheduler ${v.o.cls} at index ${i} has ${v.o.opt.repr()}, which is different from ${o.cls}.`)
    return v.o
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
  const DATASET_NAME = 'torch.utils.data.dataset.'
  /** a class a Python class can inherit from, with methods it inherits */
  const baseClass = (name: string, methods: [string, (args: Value[]) => Value][]): ClassValue => ({
    k: 'class', name: `${DATASET_NAME}${name}`, bases: [], builtin: true, host: null,
    ns: new Map(methods.map(([m, f]): [string, Value] => [m, { k: 'fn', name: m, bindable: true, call: (args) => f(args) }])),
  })
  const Dataset = baseClass('Dataset', [
    ['__getitem__', () => { throw h.err('NotImplementedError', 'Subclasses of Dataset should implement __getitem__.') }],
    ['__add__', (args) => py.obj(new ConcatDataset([args[0], args[1]]))],
  ])
  const IterableDataset = baseClass('IterableDataset', [['__add__', (args) => py.obj(new ChainDataset([args[0], args[1]]))]])
  IterableDataset.bases = [Dataset]
  ITERABLE_DATASET.cls = IterableDataset
  const sequentialSampler = (ds: Value) => new SamplerObj('SequentialSampler', (hh) => Array.from({ length: datasetLen(hh, ds) }, (_, i) => i), (hh) => datasetLen(hh, ds))
  const samplerType = (name: string, params: string, make: (a: Args) => [(hh: Host) => number[], (hh: Host) => number]): Value => ({ k: 'type', name, call: (args, kw) => {
    const [draw, size] = make(bindPy(h, `${name}.__init__`, params, args, kw))
    return py.obj(new SamplerObj(name, draw, size))
  } })
  const randomSampler = samplerType('RandomSampler', 'data_source replacement=False num_samples=None generator=None', (a) => {
    if (a.replacement!.k !== 'bool') throw h.err('TypeError', `replacement should be a boolean value, but got replacement=${repr(a.replacement!)}`)
    const given = isNone(a.num_samples) ? null : a.num_samples!
    if (given && (given.k !== 'int' || given.v <= 0)) throw h.err('ValueError', `num_samples should be a positive integer value, but got num_samples=${repr(given)}`)
    const rng = rngOf(h, a.generator)
    const size = (hh: Host) => (given ? hh.toInt(given) : datasetLen(hh, a.data_source!))
    if (bool(a.replacement, false)) return [(hh) => Array.from({ length: size(hh) }, () => Math.floor(rng.next() * datasetLen(hh, a.data_source!))), size]
    // without replacement, whole permutations follow one another until enough are drawn
    return [(hh) => {
      const n = datasetLen(hh, a.data_source!)
      const out: number[] = []
      while (out.length < size(hh) && n > 0) out.push(...shuffled(rng, n))
      return out.slice(0, size(hh))
    }, size]
  })
  const LOADER = 'dataset batch_size=1 shuffle=None sampler=None batch_sampler=None num_workers=0 collate_fn=None pin_memory=False drop_last=False timeout=0 worker_init_fn=None multiprocessing_context=None generator=None * prefetch_factor=None persistent_workers=False pin_memory_device=\'\' in_order=True'
  const utilsData: Record<string, Value> = {
    TensorDataset: { k: 'type', name: 'TensorDataset', call: (args) => {
      const ts = args.map((a) => T(a, 'TensorDataset'))
      if (ts.some((t) => t.shape[0] !== ts[0].shape[0])) throw h.err('AssertionError', 'Size mismatch between tensors')
      return py.obj(new TensorDataset(ts))
    } },
    DataLoader: { k: 'type', name: 'DataLoader', call: (args, kw) => {
      const a = bindPy(h, 'DataLoader.__init__', LOADER, args, kw)
      const ds = a.dataset!
      const workers = h.toInt(a.num_workers!)
      if (workers < 0) throw h.err('ValueError', 'num_workers option should be non-negative; use num_workers=0 to disable multiprocessing.')
      if (h.num(a.timeout!) < 0) throw h.err('ValueError', 'timeout option should be non-negative')
      const iterable = isIterableDataset(ds)
      const shuffle = a.shuffle!
      const givenBatch = !(a.batch_size!.k === 'int' && a.batch_size!.v === 1)
      if (iterable) {
        if (shuffle.k !== 'none' && h.truthy(shuffle)) throw h.err('ValueError', `DataLoader with IterableDataset: expected unspecified shuffle option, but got shuffle=${repr(shuffle)}`)
        if (!isNone(a.sampler)) throw h.err('ValueError', `DataLoader with IterableDataset: expected unspecified sampler option, but got sampler=${repr(a.sampler!)}`)
        if (!isNone(a.batch_sampler)) throw h.err('ValueError', `DataLoader with IterableDataset: expected unspecified batch_sampler option, but got batch_sampler=${repr(a.batch_sampler!)}`)
      }
      if (!isNone(a.sampler) && bool(shuffle, false)) throw h.err('ValueError', 'sampler option is mutually exclusive with shuffle')
      if (!isNone(a.batch_sampler) && (givenBatch || bool(shuffle, false) || !isNone(a.sampler) || bool(a.drop_last, false))) throw h.err('ValueError', 'batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last')
      const unbatched = isNone(a.batch_size) && isNone(a.batch_sampler)
      if (unbatched && bool(a.drop_last, false)) throw h.err('ValueError', 'batch_size=None option disables auto-batching and is mutually exclusive with drop_last')
      if (!isNone(a.batch_size) && (a.batch_size!.k !== 'int' || a.batch_size!.v <= 0)) throw h.err('ValueError', `batch_size should be a positive integer value, but got batch_size=${repr(a.batch_size!)}`)
      const batchSize = isNone(a.batch_sampler) && !isNone(a.batch_size) ? h.toInt(a.batch_size!) : null
      const drop = bool(a.drop_last, false)
      const sampler: Value = !isNone(a.sampler) ? a.sampler! : iterable ? py.NONE
        : bool(shuffle, false) ? h.call(randomSampler, [ds], isNone(a.generator) ? {} : { generator: a.generator! }) : py.obj(sequentialSampler(ds))
      const batchSampler: Value = !isNone(a.batch_sampler) ? a.batch_sampler! : batchSize !== null && !iterable ? py.obj(new BatchSampler(sampler, batchSize, drop)) : py.NONE
      return py.obj(new DataLoader(ds, batchSize, drop, sampler, batchSampler, isNone(a.collate_fn) ? null : a.collate_fn!, {
        num_workers: a.num_workers, pin_memory: a.pin_memory, timeout: a.timeout, worker_init_fn: a.worker_init_fn, multiprocessing_context: a.multiprocessing_context,
        generator: a.generator, prefetch_factor: a.prefetch_factor, persistent_workers: a.persistent_workers, pin_memory_device: a.pin_memory_device, in_order: a.in_order,
      }))
    } },
    Dataset,
    IterableDataset,
    Subset: { k: 'type', name: 'Subset', call: (args, kw) => {
      const a = bindPy(h, 'Subset.__init__', 'dataset indices', args, kw)
      return py.obj(new Subset(a.dataset!, h.iterate(a.indices!).map((v) => h.toInt(v))))
    } },
    ConcatDataset: { k: 'type', name: 'ConcatDataset', call: (args, kw) => {
      const a = bindPy(h, 'ConcatDataset.__init__', 'datasets', args, kw)
      const parts = [...h.iterate(a.datasets!)]
      if (!parts.length) throw h.err('AssertionError', 'datasets should not be an empty iterable')
      if (parts.some(isIterableDataset)) throw h.err('AssertionError', 'ConcatDataset does not support IterableDataset')
      return py.obj(new ConcatDataset(parts))
    } },
    ChainDataset: { k: 'type', name: 'ChainDataset', call: (args, kw) => py.obj(new ChainDataset([...h.iterate(bindPy(h, 'ChainDataset.__init__', 'datasets', args, kw).datasets!)])) },
    StackDataset: { k: 'type', name: 'StackDataset', call: (args, kw) => {
      const named = Object.keys(kw)
      if (args.length && named.length) throw h.err('ValueError', 'Supported either ``tuple``- (via ``args``) or ``dict``- (via ``kwargs``) like input/output, but both types are given.')
      if (!args.length && !named.length) throw h.err('ValueError', 'At least one dataset should be passed')
      return py.obj(new StackDataset(h, args.length ? args.map((v, i): [string, Value] => [String(i), v]) : named.map((k): [string, Value] => [k, kw[k]]), !args.length))
    } },
    random_split: fnv('random_split', (args, kw) => {
      const a = bindPy(h, 'random_split', 'dataset lengths generator=None', args, kw)
      const ds = a.dataset!
      const n = datasetLen(h, ds)
      let lens = h.iterate(a.lengths!).map((v) => h.num(v))
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
    default_collate: fnv('default_collate', (args, kw) => collate(h, bindPy(h, 'default_collate', 'batch', args, kw).batch!)),
    default_convert: fnv('default_convert', (args, kw) => defaultConvert(h, bindPy(h, 'default_convert', 'data', args, kw).data!)),
    get_worker_info: fnv('get_worker_info', () => py.NONE),
    Sampler: { k: 'type', name: 'Sampler', call: () => py.obj(new SamplerObj('Sampler', () => { throw h.err('NotImplementedError', '') }, () => { throw h.err('TypeError', "object of type 'Sampler' has no len()") })) },
    SequentialSampler: { k: 'type', name: 'SequentialSampler', call: (args, kw) => py.obj(sequentialSampler(bindPy(h, 'SequentialSampler.__init__', 'data_source', args, kw).data_source!)) },
    RandomSampler: randomSampler,
    SubsetRandomSampler: samplerType('SubsetRandomSampler', 'indices generator=None', (a) => {
      const rng = rngOf(h, a.generator)
      const ids = h.iterate(a.indices!).map((v) => h.toInt(v))
      return [() => shuffled(rng, ids.length).map((i) => ids[i]), () => ids.length]
    }),
    WeightedRandomSampler: samplerType('WeightedRandomSampler', 'weights num_samples replacement=True generator=None', (a) => {
      const n = a.num_samples!
      if (n.k !== 'int' || n.v <= 0) throw h.err('ValueError', `num_samples should be a positive integer value, but got num_samples=${repr(n)}`)
      if (a.replacement!.k !== 'bool') throw h.err('ValueError', `replacement should be a boolean value, but got replacement=${repr(a.replacement!)}`)
      const rng = rngOf(h, a.generator)
      const w = h.iterate(a.weights!).map((v) => h.num(v))
      return [() => drawIndices(rng, w, n.v, bool(a.replacement, true)), () => n.v]
    }),
    BatchSampler: { k: 'type', name: 'BatchSampler', call: (args, kw) => {
      const a = bindPy(h, 'BatchSampler.__init__', 'sampler batch_size drop_last', args, kw)
      const size = a.batch_size!
      if (size.k !== 'int' || size.v <= 0) throw h.err('ValueError', `batch_size should be a positive integer value, but got batch_size=${repr(size)}`)
      if (a.drop_last!.k !== 'bool') throw h.err('ValueError', `drop_last should be a boolean value, but got drop_last=${repr(a.drop_last!)}`)
      return py.obj(new BatchSampler(a.sampler!, size.v, a.drop_last!.v))
    } },
  }
  const dataSub = (name: string, keys: string[]): Value => ({ k: 'module', name: `torch.utils.data.${name}`, attrs: Object.fromEntries(keys.map((k) => [k, utilsData[k]])) })
  const dataDataloader = dataSub('dataloader', ['DataLoader', 'default_collate', 'default_convert', 'get_worker_info', 'Dataset', 'IterableDataset', 'Sampler', 'SequentialSampler', 'RandomSampler', 'BatchSampler'])
  const dataDataset = dataSub('dataset', ['Dataset', 'IterableDataset', 'TensorDataset', 'StackDataset', 'ConcatDataset', 'ChainDataset', 'Subset', 'random_split'])
  const dataSampler = dataSub('sampler', ['Sampler', 'SequentialSampler', 'RandomSampler', 'SubsetRandomSampler', 'WeightedRandomSampler', 'BatchSampler'])
  const data: Value = { k: 'module', name: 'torch.utils.data', attrs: { ...utilsData, dataloader: dataDataloader, dataset: dataDataset, sampler: dataSampler } }
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

  /** with torch.no_grad():, @torch.no_grad() or @torch.no_grad */
  const gradModeType = (name: string, enabled: boolean): Value => ({ k: 'type', name, call: (args) => {
    const m = py.obj(new GradMode(enabled))
    return args[0] ? h.call(m, [args[0]]) : m
  } })

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
    if (ord?.k === 'str') throw h.err('TypeError', "linalg_vector_norm(): argument 'ord' (position 2) must be Number, not str")
    if (!isFloatT(x.dtype)) throw h.err('RuntimeError', `linalg.vector_norm: Expected a floating point or complex tensor as input. Got ${C.typeLabel(x.dtype)}`)
    return C.vectorNorm(E, x, ord && ord.k !== 'none' ? h.num(ord) : 2, dimsArg(h, dim), bool(keep, false))
  }
  /** a stack of matrices, one at a time: f gets rows of numbers and gives tensors' (values, shape after the stack) */
  const eachMatrix = (x: TT, f: (m: L.Mat) => { values: number[]; shape: number[] }[]): TT[] => {
    const [r, c] = x.shape.slice(-2)
    const lead = x.shape.slice(0, -2)
    const count = prod(lead)
    const vals = x.values()
    const parts2: number[][] = []
    let shapes: number[][] = []
    for (let b = 0; b < count; b++) {
      const m = Array.from({ length: r }, (_, i) => Array.from({ length: c }, (_, j) => vals[(b * r + i) * c + j] as number))
      const got = f(m)
      shapes = got.map((g) => g.shape)
      got.forEach((g, i) => (parts2[i] = [...(parts2[i] ?? []), ...g.values]))
    }
    return parts2.map((v, i) => TT.of(v, [...lead, ...shapes[i]], x.dtype))
  }
  const matrixInput = (x: TT, name: string, square = false) => {
    if (x.ndim < 2) throw h.err('RuntimeError', `${name}: The input tensor A must have at least 2 dimensions.`)
    if (!isFloatT(x.dtype)) throw h.err('RuntimeError', `${name}: Expected a floating point or complex tensor as input. Got ${C.typeLabel(x.dtype)}`)
    if (square && x.shape[x.ndim - 1] !== x.shape[x.ndim - 2]) throw h.err('RuntimeError', `${name}: A must be batches of square matrices, but they are ${x.shape[x.ndim - 2]} by ${x.shape[x.ndim - 1]} matrices`)
  }
  const flatRows = (m: L.Mat) => m.flat()
  const singularValues = (x: TT): TT => eachMatrix(x, (m) => {
    const small = Math.min(m.length, m[0]?.length ?? 0)
    return [{ values: L.svdThin(m).s.slice(0, small), shape: [small] }]
  })[0]
  /** ‖A‖ from its singular values, differentiable through dσᵢ = uᵢ vᵢᵀ */
  const fromSingular = (x: TT, name: string, pickS: (s: number[]) => number, grads: (s: number[]) => number[]): TT => {
    const [r, c] = x.shape.slice(-2)
    const lead = x.shape.slice(0, -2)
    const count = prod(lead)
    const vals = x.values()
    const slopes: number[][] = []
    const out: number[] = []
    for (let b = 0; b < count; b++) {
      const m = Array.from({ length: r }, (_, i) => Array.from({ length: c }, (_, j) => vals[(b * r + i) * c + j] as number))
      const { u, s, v } = L.svdThin(m)
      out.push(pickS(s))
      const w = grads(s)
      const g = new Array<number>(r * c).fill(0)
      w.forEach((wi, k) => {
        if (!wi) return
        for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) g[i * c + j] += wi * u[i][k] * v[j][k]
      })
      slopes.push(g)
    }
    return C.record(E, TT.of(out, lead, x.dtype), name, [x], (g) => [slopes.flatMap((sl, b) => sl.map((q) => q * g[b]))])
  }
  const peakAt = (s: number[], min: boolean) => {
    let at = 0
    s.forEach((q, i) => { if (min ? q < s[at] : q > s[at]) at = i })
    return at
  }
  const singularNorm = (x: TT, ord: 'nuc' | 2 | -2, grad: string): TT => {
    if (ord === 'nuc') return fromSingular(x, grad, (s) => s.reduce((p, q) => p + q, 0), (s) => s.map(() => 1))
    const min = ord === -2
    return fromSingular(x, grad, (s) => s[peakAt(s, min)], (s) => s.map((_, i) => (i === peakAt(s, min) ? 1 : 0)))
  }
  const matrixNormOf = (x: TT, ord: Value | undefined, dims: [number, number], keep: boolean): TT => {
    matrixInput(x, 'linalg.matrix_norm')
    const special = ord?.k === 'str' && ord.v === 'nuc' ? 'nuc' : ord && ord.k !== 'none' && ord.k !== 'str' && Math.abs(h.num(ord)) === 2 ? (h.num(ord) as 2 | -2) : null
    if (ord?.k === 'str' && ord.v !== 'fro' && ord.v !== 'nuc') throw h.err('RuntimeError', `linalg.matrix_norm: Order ${ord.v} not supported.`)
    if (!special) return matrixNorm(h, E, x, ord, dims, keep)
    const [d0, d1] = dims.map((d) => C.normDim(d, x.ndim))
    const moved2 = d0 === x.ndim - 2 && d1 === x.ndim - 1 ? x : C.permuteT(E, x, [...x.shape.map((_, k) => k).filter((k) => k !== d0 && k !== d1), d0, d1])
    const out = singularNorm(moved2, special, special === 'nuc' ? 'SumBackward1' : special === 2 ? 'AmaxBackward0' : 'AminBackward0')
    return keep ? C.reshapeT(E, out, x.shape.map((n, k) => (k === d0 || k === d1 ? 1 : n)), false) : out
  }
  const inputA = (args: Value[], kw: Kw, name: string, more = '') => bindPy(h, name, more ? `A ${more}` : 'A', args, kw)
  /** a LAPACK failure: PyTorch's torch.linalg.LinAlgError, whose class calls itself _LinAlgError */
  const LinAlgError = h.excClass('_LinAlgError')
  const linalgErr = (msg: string) => h.err('_LinAlgError', msg)
  const solveChecks = (A: TT, B: TT) => {
    matrixInput(A, 'linalg.solve', false)
    if (A.shape[A.ndim - 1] !== A.shape[A.ndim - 2]) throw h.err('RuntimeError', `linalg.solve: A must be batches of square matrices, but they are ${A.shape[A.ndim - 2]} by ${A.shape[A.ndim - 1]} matrices`)
    if (A.dtype !== B.dtype) throw h.err('RuntimeError', `linalg.solve: Expected A and B to have the same dtype, but found A of type ${C.typeLabel(A.dtype)} and B of type ${C.typeLabel(B.dtype)} instead`)
    const n = A.shape[A.ndim - 1]
    const vector = B.ndim === 1 || (A.ndim - 1 === B.ndim && B.shape.slice(0, -1).join() === A.shape.slice(0, -2).join())
    if ((vector ? B.shape[B.ndim - 1] : B.shape[B.ndim - 2]) !== n) throw h.err('RuntimeError', `linalg.solve: Incompatible shapes of A and B for the equation AX = B (${n}x${n} and ${vector ? `${B.shape[B.ndim - 1]}x1` : B.shape.join('x')})`)
  }
  const singularToSolver = <X,>(f: () => X): X => {
    try {
      return f()
    } catch (e) {
      if (e instanceof PyError && e.pyType === 'LinAlgError') throw linalgErr('torch.linalg.solve: The solver failed because the input matrix is singular.')
      throw e
    }
  }
  const renameErr = <X,>(f: () => X): X => {
    try {
      return f()
    } catch (e) {
      if (e instanceof PyError && e.pyType === 'LinAlgError') throw linalgErr(e.message)
      throw e
    }
  }
  const LINALG_NAMES: Record<string, string> = { linalg_qr: 'LinalgQrBackward0', linalg_svd: 'LinalgSvdBackward0', linalg_svdvals: 'LinalgSvdBackward0', linalg_eigh: 'LinalgEighBackward0', linalg_pinv: 'LinalgPinvBackward0', linalg_cholesky: 'LinalgCholeskyExBackward0', linalg_lu: 'LinalgLuBackward0', linalg_lu_factor: 'LinalgLuFactorExBackward0', matrix_exp: 'LinalgMatrixExpBackward0', triangular_solve: 'LinalgSolveTriangularBackward0', linalg_cond: 'SqueezeBackward1' }
  const noGrad = (out: TT, x: TT, op: string): TT => C.noGradient(E, out, [x], op, LINALG_NAMES[op])
  const lstsqOf = (a: Args): Value => {
    const A = T(a.input, 'linalg.lstsq')
    const B = tensorArg(h, a.b, 'linalg.lstsq', 'b', 2)
    if (A.ndim < 2) throw h.err('RuntimeError', 'torch.linalg.lstsq: input must have at least 2 dimensions.')
    const vector = B.ndim === A.ndim - 1
    const bm = vector ? C.unsqueezeT({ grad: false } as Engine, B, -1) : B
    const [n, cols] = A.shape.slice(-2)
    if (n !== bm.shape[bm.ndim - 2]) throw h.err('RuntimeError', `torch.linalg.lstsq: input.size(-2) should match other.size(-2)`)
    const avals = A.values()
    const bvals = bm.values()
    const k = bm.shape[bm.ndim - 1]
    const am = Array.from({ length: n }, (_, i) => Array.from({ length: cols }, (_, j) => avals[i * cols + j] as number))
    const rhs = Array.from({ length: n }, (_, i) => Array.from({ length: k }, (_, j) => bvals[i * k + j] as number))
    const { u, s, v } = L.svdThin(am)
    const cut = Math.max(n, cols) * (A.dtype === 'float64' ? 2.220446049250313e-16 : 1.1920928955078125e-7) * (s[0] ?? 0)
    const ut = L.mul(L.transposeM(u), rhs)
    const sol = L.mul(v, ut.map((row, i) => row.map((x) => (s[i] > cut ? x / s[i] : 0))))
    const solution = TT.of(sol.flat(), vector ? [cols] : [cols, k], A.dtype)
    const rank = s.filter((x) => x > cut).length
    const residual = n > cols && rank === cols ? TT.of(Array.from({ length: k }, (_, j) => rhs.reduce((sum, row, i) => sum + (row[j] - am[i].reduce((p, q, c0) => p + q * sol[c0][j], 0)) ** 2, 0)), [k], A.dtype) : TT.of([], [0], A.dtype)
    return py.obj(new ReturnTypes('linalg_lstsq', [['solution', wrap(solution)], ['residuals', wrap(residual)], ['rank', wrap(TT.scalar(rank, 'int64'))], ['singular_values', wrap(TT.of([], [0], A.dtype))]]))
  }
  const eigOf = (x: TT, name: string): { re: number[]; im: number[]; vecs: L.Mat | null } => {
    matrixInput(x, name, true)
    if (x.ndim !== 2) throw h.err('NotImplementedError', `${name}: a stack of matrices is not available in this sandbox`)
    const v = x.values()
    const n = x.shape[0]
    const m = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => v[i * n + j] as number))
    const r = L.eigGeneral(m)
    return { re: r.re, im: r.im, vecs: r.vectors }
  }
  const noGradNamed = (out: TT, x: TT, name: string) => C.record(E, out, name, [x], (g) => [new Array<number>(x.size).fill(0 * g.length)])
  const slogdetOf = (x: TT): { sign: TT; log: TT } => {
    matrixInput(x, 'linalg.slogdet', true)
    const [n] = x.shape.slice(-1)
    const lead = x.shape.slice(0, -2)
    const vals = x.values()
    const signs: number[] = []
    const logs: number[] = []
    const invs: number[][] = []
    for (let b = 0; b < prod(lead); b++) {
      const m = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => vals[(b * n + i) * n + j] as number))
      const f = L.lu(m)
      let sign = f.sign
      let lg = 0
      for (let i = 0; i < n; i++) {
        const u = f.lu[i][i]
        sign *= Math.sign(u)
        lg += Math.log(Math.abs(u))
      }
      signs.push(sign)
      logs.push(lg)
      const inv = f.singular ? null : L.solveM(m, L.identity(n))
      // d log|det A| = A⁻ᵀ
      invs.push(inv ? L.transposeM(inv).flat() : new Array<number>(n * n).fill(0))
    }
    return { sign: noGradNamed(TT.of(signs, lead, x.dtype), x, 'LinalgSlogdetBackward0'), log: C.record(E, TT.of(logs, lead, x.dtype), 'LinalgSlogdetBackward0', [x], (g) => [invs.flatMap((iv, b) => iv.map((q) => q * g[b]))]) }
  }
  const pinvOf = (x: TT, rcond: number | null): TT => {
    matrixInput(x, 'linalg.pinv')
    const eps = x.dtype === 'float64' ? 2.220446049250313e-16 : 1.1920928955078125e-7
    const out = eachMatrix(x, (m) => {
      const { u, s, v } = L.svdThin(m)
      const cut = (rcond ?? Math.max(m.length, m[0].length) * eps) * Math.max(...s)
      const scaled = v.map((row) => row.map((q, k) => (s[k] > cut ? q / s[k] : 0)))
      return [{ values: flatRows(L.mul(scaled, L.transposeM(u))), shape: [m[0].length, m.length] }]
    })[0]
    return noGrad(out, x, 'linalg_pinv')
  }
  const matrixRankOf = (x: TT, tol: number | null, hermitian: boolean): TT => {
    matrixInput(x, 'torch.linalg.matrix_rank')
    const eps = x.dtype === 'float64' ? 2.220446049250313e-16 : 1.1920928955078125e-7
    const [r, c] = x.shape.slice(-2)
    const s = (hermitian ? eachMatrix(x, (m) => [{ values: L.eigh(m).values.map(Math.abs), shape: [m.length] }])[0] : singularValues(x)).values()
    const k = Math.min(r, c)
    const out: number[] = []
    for (let b = 0; b < s.length / k; b++) {
      const row = s.slice(b * k, (b + 1) * k)
      const cut = tol ?? Math.max(r, c) * eps * Math.max(...row)
      out.push(row.filter((q) => q > cut).length)
    }
    return TT.of(out, x.shape.slice(0, -2), 'int64')
  }
  const qrOf = (x: TT, mode: string): [TT, TT] => {
    matrixInput(x, 'linalg.qr')
    if (!['reduced', 'complete', 'r'].includes(mode)) throw h.err('RuntimeError', `qr received unrecognized mode '${mode}' but expected one of 'reduced' (default), 'r', or 'complete'`)
    const [r, c] = x.shape.slice(-2)
    const k = Math.min(r, c)
    const [q0, r0] = eachMatrix(x, (m) => {
      const { q, r: rr } = L.qr(m)
      const full = mode === 'complete'
      const qk = full ? q : q.map((row) => row.slice(0, k))
      const rk = full ? rr : rr.slice(0, k)
      return [{ values: flatRows(qk), shape: [r, full ? r : k] }, { values: flatRows(rk), shape: [full ? r : k, c] }]
    })
    return [noGrad(q0, x, 'linalg_qr'), noGrad(r0, x, 'linalg_qr')]
  }
  const svdOf = (x: TT, full: boolean): [TT, TT, TT] => {
    matrixInput(x, 'linalg.svd')
    const [r, c] = x.shape.slice(-2)
    const k = Math.min(r, c)
    const [u, s0, v] = eachMatrix(x, (m) => {
      const res = full ? L.svdFull(m) : L.svdThin(m)
      return [
        { values: flatRows(res.u), shape: [r, full ? r : k] },
        { values: res.s.slice(0, k), shape: [k] },
        { values: flatRows(L.transposeM(res.v)), shape: [full ? c : k, c] },
      ]
    })
    return [noGrad(u, x, 'linalg_svd'), noGrad(s0, x, 'linalg_svd'), noGrad(v, x, 'linalg_svd')]
  }
  const triangularSolve = (A: TT, b: TT, upper: boolean, transpose: boolean, unit: boolean, left = true): TT => {
    if (A.ndim !== 2 || b.ndim < 1) throw h.err('NotImplementedError', 'triangular solve: only one matrix at a time is available in this sandbox')
    const n = A.shape[0]
    const av = A.values()
    const flip = transpose !== !left
    const at = (i: number, j: number) => {
      const [r, c] = flip ? [j, i] : [i, j]
      if (unit && r === c) return 1
      return upper ? (c < r ? 0 : av[r * n + c]) : (c > r ? 0 : av[r * n + c])
    }
    const lowerNow = flip ? upper : !upper
    const vector = b.ndim === 1
    const bb = vector ? C.unsqueezeT({ grad: false } as Engine, b, -1) : left ? b : C.transposeT({ grad: false } as Engine, b, -2, -1)
    const k = bb.shape[1]
    const v = bb.values()
    const x = new Array<number>(n * k).fill(0)
    for (let q = 0; q < k; q++) {
      for (let ii = 0; ii < n; ii++) {
        const i = lowerNow ? ii : n - 1 - ii
        let s0 = v[i * k + q]
        for (let j = 0; j < n; j++) if (j !== i && (lowerNow ? j < i : j > i)) s0 -= at(i, j) * x[j * k + q]
        const d = at(i, i)
        if (d === 0) throw linalgErr('torch.linalg.solve_triangular: The solver failed because the input matrix is singular.')
        x[i * k + q] = s0 / d
      }
    }
    const out = noGrad(TT.of(x, [n, k], A.dtype), A, 'triangular_solve')
    return vector ? C.squeezeT(E, out, 1) : left ? out : C.transposeT(E, out, 0, 1)
  }
  const luSolve = (b: TT, f: TT, piv: TT, leftSide = true): Value => {
    if (f.ndim !== 2) throw h.err('NotImplementedError', 'lu_solve: only one matrix at a time is available in this sandbox')
    if (!leftSide) throw h.err('NotImplementedError', 'lu_solve: left=False is not available in this sandbox')
    const n = f.shape[0]
    const vector = b.ndim === 1
    const bm = vector ? C.unsqueezeT({ grad: false } as Engine, b, -1) : b
    const k = bm.shape[1]
    const rows = Array.from({ length: n }, (_, i) => bm.values().slice(i * k, (i + 1) * k) as number[])
    piv.values().forEach((p, i) => { [rows[i], rows[p - 1]] = [rows[p - 1], rows[i]] })
    const y = triangularSolve(f, TT.of(rows.flat(), [n, k], b.dtype), false, false, true)
    const out = triangularSolve(f, y, true, false, false)
    return wrap(vector ? C.squeezeT(E, out, 1) : out)
  }
  const exInfo = (t: TT) => TT.of(new Array<number>(prod(t.shape.slice(0, -2))).fill(0), t.shape.slice(0, -2), 'int32')
  const linalgFns: Record<string, Value> = {
    LinAlgError,
    matmul: fnv('matmul', (args, kw) => {
      const a = bindPy(h, 'matmul', 'input other * out=None', args, kw)
      return objOf(T(a.input, 'matmul')).binop('@', a.other!, false, h)!
    }),
    vander: fnv('vander', (args, kw) => {
      const a = bindPy(h, 'vander', 'x * N=None', args, kw)
      const t = T(a.x, 'linalg.vander')
      if (!t.ndim) throw h.err('RuntimeError', 'x must be at least 1-dimensional.')
      const N = isNone(a.N) ? t.shape[t.ndim - 1] : h.toInt(a.N!)
      if (N <= 1) throw h.err('RuntimeError', 'N must be greater than 1.')
      const dtype = isFloatT(t.dtype) ? t.dtype : 'int64'
      return wrap(TT.of(t.values().flatMap((v) => Array.from({ length: N }, (_, j) => v ** j)), [...t.shape, N], dtype))
    }),
    solve_triangular: fnv('solve_triangular', (args, kw) => {
      const a = bindPy(h, 'solve_triangular', 'A B * upper left=True unitriangular=False', args, kw)
      if (tensorArg(h, a.B, 'linalg.solve_triangular', 'B', 2).ndim < 2) throw h.err('RuntimeError', 'linalg.solve_triangular: The input tensor B must have at least 2 dimensions.')
      return wrap(triangularSolve(T(a.A, 'linalg.solve_triangular'), tensorArg(h, a.B, 'linalg.solve_triangular', 'B', 2), h.truthy(a.upper!), false, bool(a.unitriangular, false), bool(a.left, true)))
    }),
    lu_solve: fnv('lu_solve', (args, kw) => {
      const a = bindPy(h, 'lu_solve', 'LU pivots B * left=True adjoint=False', args, kw)
      if (bool(a.adjoint, false)) throw h.err('NotImplementedError', 'linalg.lu_solve: adjoint=True is not available in this sandbox')
      return luSolve(tensorArg(h, a.B, 'linalg.lu_solve', 'B', 3), T(a.LU, 'linalg.lu_solve'), tensorArg(h, a.pivots, 'linalg.lu_solve', 'pivots', 2), bool(a.left, true))
    }),
    inv_ex: fnv('inv_ex', (args, kw) => {
      const a = bindPy(h, 'inv_ex', 'A * check_errors=False', args, kw)
      const x = T(a.A, 'linalg.inv_ex')
      const inv = renameErr(() => C.invT(E, x))
      return py.obj(new ReturnTypes('linalg_inv_ex', [['inverse', wrap(inv)], ['info', wrap(exInfo(x))]]))
    }),
    cholesky_ex: fnv('cholesky_ex', (args, kw) => {
      const a = bindPy(h, 'cholesky_ex', 'A * upper=False check_errors=False', args, kw)
      const x = T(a.A, 'linalg.cholesky_ex')
      const L0 = h.call(linalgFns.cholesky, [a.A!], { upper: a.upper! })
      return py.obj(new ReturnTypes('linalg_cholesky_ex', [['L', L0], ['info', wrap(exInfo(x))]]))
    }),
    solve_ex: fnv('solve_ex', (args, kw) => {
      const a = bindPy(h, 'solve_ex', 'A B * left=True check_errors=False', args, kw)
      return py.obj(new ReturnTypes('linalg_solve_ex', [['result', h.call(linalgFns.solve, [a.A!, a.B!], { left: a.left! })], ['info', wrap(exInfo(T(a.A, 'linalg.solve_ex')))]]))
    }),
    lu_factor_ex: fnv('lu_factor_ex', (args, kw) => {
      const a = bindPy(h, 'lu_factor_ex', 'A * pivot=True check_errors=False', args, kw)
      const x = T(a.A, 'linalg.lu_factor_ex')
      const [f, p] = luParts(x)
      return py.obj(new ReturnTypes('linalg_lu_factor_ex', [['LU', wrap(f)], ['pivots', wrap(p)], ['info', wrap(exInfo(x))]]))
    }),
    inv: fnv('inv', (args, kw) => {
      const x = T(args[0] ?? kw.A, 'linalg.inv')
      return linalgStep('inv', renameErr(() => C.invT(E, x)), x, args[0] ?? kw.A)
    }),
    det: fnv('det', (args, kw) => {
      const x = T(args[0] ?? kw.A, 'linalg.det')
      return linalgStep('det', C.detT(E, x), x, args[0] ?? kw.A)
    }),
    solve: fnv('solve', (args, kw) => {
      const a = bindPy(h, 'solve', 'A B * left=True', args, kw)
      const A = T(a.A, 'linalg.solve')
      const B = tensorArg(h, a.B, 'linalg.solve', 'B', 2)
      if (!bool(a.left, true)) return wrap(named(singularToSolver(() => C.transposeT(E, C.solveT(E, C.transposeT(E, A, -2, -1), C.transposeT(E, B, -2, -1)), -2, -1)), 'LinalgSolveExBackward0'))
      solveChecks(A, B)
      return linalgStep('solve', named(singularToSolver(() => C.solveT(E, A, B)), 'LinalgSolveExBackward0'), A, a.A!)
    }),
    norm: fnv('norm', (args, kw) => {
      const a = bindPy(h, 'norm', 'A ord=None dim=None keepdim=False * dtype=None', args, kw)
      const x = normDtype(T(a.A, 'linalg.norm'), a.dtype)
      const dims = dimsArg(h, a.dim)
      const keep = bool(a.keepdim, false)
      const noOrd = isNone(a.ord)
      if (!isFloatT(x.dtype)) throw h.err('RuntimeError', `linalg.vector_norm: Expected a floating point or complex tensor as input. Got ${C.typeLabel(x.dtype)}`)
      if (dims && dims.length === 2) return wrap(matrixNormOf(x, a.ord, [dims[0], dims[1]], keep))
      if (dims && dims.length > 2) throw h.err('RuntimeError', `linalg.norm: If dim is specified, it must be of length 1 or 2. Got [${dims.join(', ')}]`)
      if (dims) return wrap(vectorNormOf(x, a.ord, a.dim, a.keepdim))
      if (noOrd) return wrap(C.vectorNorm(E, x, 2, null, keep))
      if (a.ord!.k === 'str' && x.ndim < 2) throw h.err('RuntimeError', 'linalg.matrix_norm: The input tensor A must have at least 2 dimensions.')
      if (x.ndim === 1) return wrap(vectorNormOf(x, a.ord, undefined, a.keepdim))
      if (x.ndim === 2) return wrap(matrixNormOf(x, a.ord, [0, 1], keep))
      throw h.err('RuntimeError', `linalg.norm: If dim is not specified but ord is, the input must be 1D or 2D. Got ${x.ndim}D.`)
    }),
    vector_norm: fnv('vector_norm', (args, kw) => {
      const a = bindPy(h, 'vector_norm', 'x ord=2 dim=None keepdim=False * dtype=None', args, kw)
      return wrap(vectorNormOf(normDtype(T(a.x, 'linalg.vector_norm'), a.dtype), a.ord, a.dim, a.keepdim))
    }),
    matrix_norm: fnv('matrix_norm', (args, kw) => {
      const a = bindPy(h, 'matrix_norm', "A ord='fro' dim=None keepdim=False * dtype=None", args, kw)
      const dims = dimsArg(h, a.dim) ?? [-2, -1]
      if (dims.length !== 2) throw h.err('RuntimeError', `linalg.matrix_norm: dim must be a 2-tuple. Got [${dims.join(', ')}]`)
      return wrap(matrixNormOf(normDtype(T(a.A, 'linalg.matrix_norm'), a.dtype), a.ord, [dims[0], dims[1]], bool(a.keepdim, false)))
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
    multi_dot: fnv('multi_dot', (args) => wrap(named(h.iterate(args[0]).map((v) => T(v, 'linalg.multi_dot')).reduce((p, q) => C.matmul(E, p, q)), 'ViewBackward0'))),
    matrix_rank: fnv('matrix_rank', (args, kw) => {
      const a = bindPy(h, 'matrix_rank', 'A tol=None hermitian=False', args, kw)
      return wrap(matrixRankOf(T(a.A, 'linalg.matrix_rank'), isNone(a.tol) ? null : h.num(a.tol!), bool(a.hermitian, false)))
    }),
    slogdet: fnv('slogdet', (args, kw) => {
      const r = slogdetOf(T(inputA(args, kw, 'slogdet').A, 'linalg.slogdet'))
      return py.obj(new ReturnTypes('linalg_slogdet', [['sign', wrap(r.sign)], ['logabsdet', wrap(r.log)]]))
    }),
    cholesky: fnv('cholesky', (args, kw) => {
      const a = bindPy(h, 'cholesky', 'A * upper=False', args, kw)
      const x = T(a.A, 'linalg.cholesky')
      matrixInput(x, 'linalg.cholesky', true)
      const out = eachMatrix(x, (m) => {
        const l = L.cholesky(m)
        if (!l) {
          let order = 0
          for (let k = 1; k <= m.length; k++) {
            if (!L.cholesky(m.slice(0, k).map((row) => row.slice(0, k)))) {
              order = k
              break
            }
          }
          throw linalgErr(`linalg.cholesky: The factorization could not be completed because the input is not positive-definite (the leading minor of order ${order} is not positive-definite).`)
        }
        return [{ values: flatRows(bool(a.upper, false) ? L.transposeM(l) : l), shape: [m.length, m.length] }]
      })[0]
      return wrap(noGrad(out, x, 'linalg_cholesky'))
    }),
    eigh: fnv('eigh', (args, kw) => {
      const a = bindPy(h, 'eigh', "A UPLO='L'", args, kw)
      const x = T(a.A, 'linalg.eigh')
      matrixInput(x, 'linalg.eigh', true)
      const [w, v] = eachMatrix(x, (m) => {
        const r = L.eigh(m)
        return [{ values: r.values, shape: [m.length] }, { values: flatRows(r.vectors), shape: [m.length, m.length] }]
      })
      return py.obj(new ReturnTypes('linalg_eigh', [['eigenvalues', wrap(noGrad(w, x, 'linalg_eigh'))], ['eigenvectors', wrap(noGrad(v, x, 'linalg_eigh'))]]))
    }),
    eigvalsh: fnv('eigvalsh', (args, kw) => {
      const a = bindPy(h, 'eigvalsh', "A UPLO='L'", args, kw)
      const x = T(a.A, 'linalg.eigvalsh')
      matrixInput(x, 'linalg.eigvalsh', true)
      const n = x.shape[x.ndim - 1]
      const lead = x.shape.slice(0, -2)
      const vals = x.values()
      const out: number[] = []
      const slopes: number[][][] = []
      for (let b = 0; b < prod(lead); b++) {
        const m = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => vals[(b * n + i) * n + j] as number))
        const r = L.eigh(m)
        out.push(...r.values)
        // dλₖ = vₖ vₖᵀ
        slopes.push(r.values.map((_, k) => Array.from({ length: n * n }, (_, f) => r.vectors[Math.floor(f / n)][k] * r.vectors[f % n][k])))
      }
      return wrap(C.record(E, TT.of(out, [...lead, n], x.dtype), 'LinalgEighBackward0', [x], (g) => [slopes.flatMap((sl, b) => {
        const sum = new Array<number>(n * n).fill(0)
        sl.forEach((row, k) => row.forEach((q, f) => (sum[f] += q * g[b * n + k])))
        return sum
      })]))
    }),
    eig: fnv('eig', (args, kw) => {
      const r = eigOf(T(inputA(args, kw, 'eig').A, 'linalg.eig'), 'linalg.eig')
      if (!r.vecs) throw h.err('NotImplementedError', 'linalg.eig: eigenvectors of a matrix with complex eigenvalues are not available in this sandbox; torch.linalg.eigvals gives the values')
      const n = r.re.length
      return py.obj(new ReturnTypes('linalg_eig', [['eigenvalues', py.obj(new ComplexTensor(r.re, r.im, [n]))], ['eigenvectors', py.obj(new ComplexTensor(flatRows(r.vecs), new Array<number>(n * n).fill(0), [n, n]))]]))
    }),
    eigvals: fnv('eigvals', (args, kw) => {
      const r = eigOf(T(inputA(args, kw, 'eigvals').A, 'linalg.eigvals'), 'linalg.eigvals')
      return py.obj(new ComplexTensor(r.re, r.im, [r.re.length]))
    }),
    svd: fnv('svd', (args, kw) => {
      const a = bindPy(h, 'svd', 'A full_matrices=True * driver=None', args, kw)
      const [u, s0, v] = svdOf(T(a.A, 'linalg.svd'), bool(a.full_matrices, true))
      return py.obj(new ReturnTypes('linalg_svd', [['U', wrap(u)], ['S', wrap(s0)], ['Vh', wrap(v)]]))
    }),
    svdvals: fnv('svdvals', (args, kw) => {
      const x = T(bindPy(h, 'svdvals', 'A * driver=None', args, kw).A, 'linalg.svdvals')
      matrixInput(x, 'linalg.svdvals')
      if (!x.requiresGrad || !E.grad) return wrap(singularValues(x))
      if (x.ndim !== 2) return wrap(noGrad(singularValues(x), x, 'linalg_svdvals'))
      const [r, c] = x.shape
      const v0 = x.values()
      const m = Array.from({ length: r }, (_, i) => Array.from({ length: c }, (_, j) => v0[i * c + j] as number))
      const res = L.svdThin(m)
      const k = Math.min(r, c)
      return wrap(C.record(E, TT.of(res.s.slice(0, k), [k], x.dtype), 'LinalgSvdBackward0', [x], (g) => {
        const out = new Array<number>(r * c).fill(0)
        for (let q = 0; q < k; q++) for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i * c + j] += g[q] * res.u[i][q] * res.v[j][q]
        return [out]
      }))
    }),
    qr: fnv('qr', (args, kw) => {
      const a = bindPy(h, 'qr', "A mode='reduced'", args, kw)
      const mode = h.str(a.mode!)
      const [q, r] = qrOf(T(a.A, 'linalg.qr'), mode)
      return py.obj(new ReturnTypes('linalg_qr', [['Q', mode === 'r' ? wrap(TT.of([], [0], q.dtype)) : wrap(q)], ['R', wrap(r)]]))
    }),
    pinv: fnv('pinv', (args, kw) => {
      const a = bindPy(h, 'pinv', 'A rcond=None hermitian=False * atol=None rtol=None', args, kw)
      return wrap(pinvOf(T(a.A, 'linalg.pinv'), isNone(a.rcond) ? (isNone(a.rtol) ? null : h.num(a.rtol!)) : h.num(a.rcond!)))
    }),
    lstsq: fnv('lstsq', (args, kw) => {
      const a = bindPy(h, 'lstsq', 'input b rcond=None * driver=None', args, kw)
      return lstsqOf(a)
    }),
    cond: fnv('cond', (args, kw) => {
      const a = bindPy(h, 'cond', 'A p=None', args, kw)
      const x = T(a.A, 'linalg.cond')
      matrixInput(x, 'linalg.cond')
      if (isNone(a.p) || (a.p!.k !== 'str' && Math.abs(h.num(a.p!)) === 2)) {
        const s0 = singularValues(x).values()
        const k = Math.min(...x.shape.slice(-2))
        const min = !isNone(a.p) && h.num(a.p!) === -2
        const out: number[] = []
        for (let b = 0; b < s0.length / k; b++) {
          const row = s0.slice(b * k, (b + 1) * k)
          const [big, small] = [Math.max(...row), Math.min(...row)]
          out.push(min ? small / big : big / small)
        }
        return wrap(noGrad(TT.of(out, x.shape.slice(0, -2), x.dtype), x, 'linalg_cond'))
      }
      if (x.shape[x.ndim - 1] !== x.shape[x.ndim - 2]) throw h.err('RuntimeError', `linalg.cond: A must be batches of square matrices, but they are ${x.shape[x.ndim - 2]} by ${x.shape[x.ndim - 1]} matrices`)
      return wrap(C.binary(E, '*', matrixNormOf(x, a.p, [-2, -1], false), matrixNormOf(renameErr(() => C.invT(E, x)), a.p, [-2, -1], false)))
    }),
    cross: fnv('cross', (args, kw) => {
      const a = bindPy(h, 'cross', 'input other * dim=-1', args, kw)
      const [x, y] = [T(a.input, 'linalg.cross'), tensorArg(h, a.other, 'linalg.cross', 'other', 2)]
      const shape = broadcastShapes([x.shape, y.shape])
      const d = C.normDim(h.toInt(a.dim!), shape.length)
      if (x.shape[C.normDim(h.toInt(a.dim!), x.ndim)] !== 3 || y.shape[C.normDim(h.toInt(a.dim!), y.ndim)] !== 3) throw h.err('RuntimeError', `linalg.cross: inputs dimension ${h.toInt(a.dim!)} must have length 3. Got ${x.shape[C.normDim(h.toInt(a.dim!), x.ndim)]} and ${y.shape[C.normDim(h.toInt(a.dim!), y.ndim)]}`)
      return wrap(named(crossProduct(E, C.expandT(E, x, shape), C.expandT(E, y, shape), d), 'LinalgCrossBackward0'))
    }),
    diagonal: fnv('diagonal', (args, kw) => {
      const a = bindPy(h, 'diagonal', 'A * offset=0 dim1=-2 dim2=-1', args, kw)
      return h.call(h.getAttr(a.A!, 'diagonal'), [], { offset: a.offset!, dim1: a.dim1!, dim2: a.dim2! })
    }),
    vecdot: fnv('vecdot', (args, kw) => {
      const a = bindPy(h, 'vecdot', 'x y * dim=-1', args, kw)
      const [x, y] = [T(a.x, 'linalg.vecdot', ), tensorArg(h, a.y, 'linalg.vecdot', 'y', 2)]
      return wrap(named(C.reduceSum(E, C.binary(E, '*', x, y), [h.toInt(a.dim!)], false, false), 'VdotBackward0'))
    }),
    tensorinv: fnv('tensorinv', (args, kw) => {
      const a = bindPy(h, 'tensorinv', 'A ind=2', args, kw)
      const x = T(a.A, 'linalg.tensorinv')
      const ind = h.toInt(a.ind!)
      if (ind <= 0) throw h.err('RuntimeError', `Expected a strictly positive integer for 'ind'`)
      const [front, back] = [x.shape.slice(0, ind), x.shape.slice(ind)]
      const n = prod(front)
      if (n !== prod(back)) throw h.err('RuntimeError', 'Expected self to satisfy the requirement prod(self.shape[ind:]) == prod(self.shape[:ind]), but got ' + prod(back) + ' != ' + n)
      return wrap(C.reshapeT(E, renameErr(() => C.invT(E, C.reshapeT(E, x, [n, n], false))), [...back, ...front], false))
    }),
    tensorsolve: fnv('tensorsolve', (args, kw) => {
      const a = bindPy(h, 'tensorsolve', 'A B dims=None', args, kw)
      const x = T(a.A, 'linalg.tensorsolve')
      const b = tensorArg(h, a.B, 'linalg.tensorsolve', 'B', 2)
      if (!isNone(a.dims)) throw h.err('NotImplementedError', 'linalg.tensorsolve: dims= is not available in this sandbox')
      const n = b.size
      const back = x.shape.slice(b.ndim)
      return wrap(C.reshapeT(E, singularToSolver(() => C.solveT(E, C.reshapeT(E, x, [n, n], false), C.reshapeT(E, b, [n], false))), back, false))
    }),
    lu_factor: fnv('lu_factor', (args, kw) => {
      const a = bindPy(h, 'lu_factor', 'A * pivot=True', args, kw)
      const [f, p] = luParts(T(a.A, 'linalg.lu_factor'))
      return py.obj(new ReturnTypes('linalg_lu_factor', [['LU', wrap(f)], ['pivots', wrap(p)]]))
    }),
    lu: fnv('lu', (args, kw) => {
      const a = bindPy(h, 'lu', 'A * pivot=True', args, kw)
      const x = T(a.A, 'linalg.lu')
      if (x.ndim !== 2) throw h.err('NotImplementedError', 'linalg.lu: a stack of matrices is not available in this sandbox')
      const [r, c] = x.shape
      const k = Math.min(r, c)
      const [f, piv] = luParts(x)
      const fv = f.values()
      const lower = Array.from({ length: r * k }, (_, q) => {
        const [i, j] = [Math.floor(q / k), q % k]
        return i === j ? 1 : i > j ? fv[i * c + j] : 0
      })
      const upper = Array.from({ length: k * c }, (_, q) => {
        const [i, j] = [Math.floor(q / c), q % c]
        return i <= j ? fv[i * c + j] : 0
      })
      const order = Array.from({ length: r }, (_, i) => i)
      piv.values().forEach((p0, i) => { [order[i], order[p0 - 1]] = [order[p0 - 1], order[i]] })
      const perm = new Array<number>(r * r).fill(0)
      order.forEach((row, i) => (perm[row * r + i] = 1))
      return py.obj(new ReturnTypes('linalg_lu', [['P', wrap(TT.of(perm, [r, r], x.dtype))], ['L', wrap(noGrad(TT.of(lower, [r, k], x.dtype), x, 'linalg_lu'))], ['U', wrap(noGrad(TT.of(upper, [k, c], x.dtype), x, 'linalg_lu'))]]))
    }),
    matrix_exp: fnv('matrix_exp', (args, kw) => {
      const x = T(inputA(args, kw, 'matrix_exp').A, 'linalg.matrix_exp')
      matrixInput(x, 'linalg.matrix_exp', true)
      return wrap(noGrad(eachMatrix(x, (m) => [{ values: flatRows(matrixExp(m)), shape: [m.length, m.length] }])[0], x, 'matrix_exp'))
    }),
  }
  /** Tensor LU with LAPACK's 1-based pivots */
  const luParts = (x: TT): [TT, TT] => {
    matrixInput(x, 'linalg.lu_factor')
    if (x.ndim !== 2) throw h.err('NotImplementedError', 'linalg.lu_factor: a stack of matrices is not available in this sandbox')
    const [r, c] = x.shape
    const v = x.values()
    const m = Array.from({ length: r }, (_, i) => Array.from({ length: c }, (_, j) => v[i * c + j] as number))
    const k = Math.min(r, c)
    const piv: number[] = []
    for (let i = 0; i < k; i++) {
      let best = i
      for (let q = i + 1; q < r; q++) if (Math.abs(m[q][i]) > Math.abs(m[best][i])) best = q
      piv.push(best + 1)
      ;[m[i], m[best]] = [m[best], m[i]]
      if (m[i][i] === 0) continue
      for (let q = i + 1; q < r; q++) {
        m[q][i] /= m[i][i]
        for (let j = i + 1; j < c; j++) m[q][j] -= m[q][i] * m[i][j]
      }
    }
    return [noGrad(TT.of(m.flat(), [r, c], x.dtype), x, 'linalg_lu_factor'), TT.of(piv, [k], 'int32')]
  }
  /** scaling and squaring with a Taylor series */
  const matrixExp = (m: L.Mat): L.Mat => {
    const n = m.length
    const norm = Math.max(0, ...m.map((row) => row.reduce((s0, x) => s0 + Math.abs(x), 0)))
    const steps = norm > 0.5 ? Math.ceil(Math.log2(norm / 0.5)) : 0
    const a = m.map((row) => row.map((x) => x / 2 ** steps))
    let out = L.identity(n)
    let term = L.identity(n)
    for (let k = 1; k < 20; k++) {
      term = L.mul(term, a).map((row) => row.map((x) => x / k))
      out = out.map((row, i) => row.map((x, j) => x + term[i][j]))
    }
    for (let i = 0; i < steps; i++) out = L.mul(out, out)
    return out
  }
  const linalg: Value = { k: 'module', name: 'torch.linalg', attrs: linalgFns }

  // ---- torch.einsum: parse, take repeated labels' diagonals, line every operand up on all the labels, multiply, sum
  const einsum = fnv('einsum', (args, kw) => {
    if (!args.length || args[0].k !== 'str') throw invalidArgs(h, 'einsum', args, kw)
    const eq = args[0].v.replace(/\s/g, '')
    const ops = args.length === 2 && (args[1].k === 'list' || args[1].k === 'tuple') ? h.iterate(args[1]) : args.slice(1)
    const ts = ops.map((v, i) => {
      const t = asTT(v)
      if (!t) throw h.err('TypeError', `expected Tensor as element ${i} in argument 1, but got ${typeName(v)}`)
      return t
    })
    if (!ts.length) throw h.err('RuntimeError', 'einsum(): must provide at least one operand')
    const arrow = eq.indexOf('->')
    const lhs = arrow >= 0 ? eq.slice(0, arrow) : eq
    const given = arrow >= 0 ? eq.slice(arrow + 2) : null
    const terms = lhs.split(',')
    if (terms.length !== ts.length) throw h.err('RuntimeError', `einsum(): more operands were provided than specified in the equation`.replace('more', terms.length > ts.length ? 'fewer' : 'more'))
    const isLabel = (c: string) => /[a-zA-Z]/.test(c)
    for (const c of lhs + (given ?? '')) if (!isLabel(c) && !',.'.includes(c)) throw h.err('RuntimeError', `einsum(): invalid subscript given at index ${eq.indexOf(c)} in the equation string, subscripts must be in [a-zA-Z]`)
    // ellipsis dimensions get labels of their own, from the right
    const ELL = (k: number) => String.fromCharCode(0x100 + k)
    let ellDims = 0
    const parsed = terms.map((term, i) => {
      const t = ts[i]
      const dots = term.indexOf('...')
      const plain = term.replace('...', '')
      if (dots < 0) {
        if (plain.length !== t.ndim) throw h.err('RuntimeError', `einsum(): the number of subscripts in the equation (${plain.length}) does not match the number of dimensions (${t.ndim}) for operand ${i} and no ellipsis was given`)
        return [...plain]
      }
      const n = t.ndim - plain.length
      if (n < 0) throw h.err('RuntimeError', `einsum(): the number of subscripts in the equation (${plain.length}) is more than the number of dimensions (${t.ndim}) for operand ${i}`)
      ellDims = Math.max(ellDims, n)
      return [...term.slice(0, dots), ...Array.from({ length: n }, (_, k) => ELL(n - 1 - k)), ...term.slice(dots + 3)]
    })
    const counts = new Map<string, number>()
    for (const p of parsed) for (const l of p) counts.set(l, (counts.get(l) ?? 0) + 1)
    let out: string[]
    if (given !== null) {
      const dots = given.indexOf('...')
      const ell = Array.from({ length: ellDims }, (_, k) => ELL(ellDims - 1 - k))
      out = dots < 0 ? [...given] : [...given.slice(0, dots), ...ell, ...given.slice(dots + 3)]
      for (const l of out) if (!counts.has(l) ) throw h.err('RuntimeError', `einsum(): output subscript ${l} does not appear in the equation for any input operand`)
    } else {
      const ell = Array.from({ length: ellDims }, (_, k) => ELL(ellDims - 1 - k))
      out = [...ell, ...[...counts.keys()].filter((l) => isLabel(l) && counts.get(l) === 1).sort()]
    }
    const order = [...out, ...[...counts.keys()].filter((l) => !out.includes(l)).sort()]
    const sizes2 = new Map<string, number>()
    // each operand on every label, in order; a repeated label within an operand is its diagonal
    const aligned = parsed.map((labels, i) => {
      let t = ts[i]
      let cur = [...labels]
      for (let at = 0; at < cur.length; at++) {
        const twin = cur.indexOf(cur[at], at + 1)
        if (twin < 0) continue
        if (t.shape[at] !== t.shape[twin]) throw h.err('RuntimeError', `einsum(): subscript ${cur[at]} is repeated for operand ${i} but the sizes don't match, ${t.shape[twin]} != ${t.shape[at]}`)
        t = diagonalOf(E, t, 0, at, twin, 'DiagonalBackward0')
        cur = [...cur.filter((_, k) => k !== at && k !== twin), cur[at]]
        at = -1
      }
      cur.forEach((l, k) => {
        const n = t.shape[k]
        const seen = sizes2.get(l)
        if (seen === undefined || seen === 1) sizes2.set(l, n)
        else if (n !== 1 && n !== seen) throw h.err('RuntimeError', `einsum(): subscript ${l} has size ${n} for operand ${i} which does not broadcast with previously seen size ${seen}`)
      })
      const perm = order.filter((l) => cur.includes(l)).map((l) => cur.indexOf(l))
      const moved3 = perm.every((p, k) => p === k) ? t : C.permuteT(E, t, perm)
      return C.reshapeT(E, moved3, order.map((l) => (cur.includes(l) ? t.shape[cur.indexOf(l)] : 1)), false)
    })
    let res = aligned[0]
    for (let i = 1; i < aligned.length; i++) {
      if (aligned[i].dtype !== res.dtype) throw h.err('RuntimeError', `expected scalar type ${C.typeLabel(res.dtype)} but found ${C.typeLabel(aligned[i].dtype)}`)
      res = C.binary(E, '*', res, aligned[i])
    }
    const summed = order.map((l, k) => (out.includes(l) ? -1 : k)).filter((k) => k >= 0)
    if (summed.length) res = C.reduceSum(E, res, summed, false, false)
    else if (ts.length === 1 && res === ts[0]) res = C.viewT(E, res, (x) => x, 'AliasBackward0')
    if (aligned.length > 1 && res.fn && summed.length) res.fn.name = 'ViewBackward0'
    if (aligned.length === 1 && res.fn && !summed.length && res.fn.name !== 'PermuteBackward0') res.fn.name = 'PermuteBackward0'
    trace(h, 'torch.einsum', 'matmul', ts.map((t, i) => ({ label: label(h, ops[i], `t${i}`), t })), res)
    return wrap(res)
  })

  // ---- torch.special: mostly what torch.* has, under other names
  const mapped = (name: string, f: (x: number) => number, df: ((x: number) => number) | null, grad: string, keepInt = false) => fnv(name, (args, kw) => {
    const t = T(bindPy(h, name, 'input * out=None', args, kw).input, name)
    return wrap(C.mapT(E, t, f, df, grad, keepInt))
  })
  const sig2 = (v: number) => 1 / (1 + Math.exp(-v))
  const special: Value = {
    k: 'module', name: 'torch.special', attrs: {
      expit: mapped('expit', sig2, (v) => sig2(v) * (1 - sig2(v)), 'SigmoidBackward0'),
      logit: viaMethod('logit'),
      erf: viaMethod('erf'), erfc: viaMethod('erfc'), erfinv: viaMethod('erfinv'),
      erfcx: mapped('erfcx', erfcxFn, (x) => 2 * x * erfcxFn(x) - 2 / Math.sqrt(Math.PI), 'ErfcxBackward0'),
      entr: mapped('entr', entrFn, (x) => -Math.log(x) - 1, 'EntrBackward0'),
      ndtr: mapped('ndtr', ndtrFn, (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI), 'ErfcBackward0'),
      ndtri: mapped('ndtri', (p) => Math.SQRT2 * erfinvFn(2 * p - 1), (p) => Math.sqrt(2 * Math.PI) * Math.exp((Math.SQRT2 * erfinvFn(2 * p - 1)) ** 2 / 2), 'NdtriBackward0'),
      log_ndtr: mapped('log_ndtr', logNdtrFn, (x) => Math.exp(-0.5 * x * x - logNdtrFn(x)) / Math.sqrt(2 * Math.PI), 'LogNdtrBackward0'),
      gammaln: mapped('gammaln', lgammaFn, (x) => polygammaFn(0, x), 'LgammaBackward0'),
      digamma: viaMethod('digamma'), psi: viaMethod('psi', 'digamma'),
      polygamma: fnv('polygamma', (args, kw) => {
        const a = bindPy(h, 'polygamma', 'n input * out=None', args, kw)
        const n = h.toInt(a.n!)
        if (n < 0) throw h.err('RuntimeError', 'polygamma(n, x) does not support negative n.')
        return wrap(C.mapT(E, T(a.input, 'polygamma'), (x) => polygammaFn(n, x), (x) => polygammaFn(n + 1, x), 'PolygammaBackward0'))
      }),
      multigammaln: fnv('multigammaln', (args, kw) => {
        const a = bindPy(h, 'multigammaln', 'input p * out=None', args, kw)
        const p = h.toInt(a.p!)
        const t = T(a.input, 'multigammaln')
        if (p < 1) throw h.err('RuntimeError', 'p has to be greater than or equal to 1')
        if (t.values().some((v) => v <= 0.5 * (p - 1))) throw h.err('RuntimeError', `All elements must be greater than (p-1)/2`)
        const f = (x: number) => (p * (p - 1) * 0.25 * Math.log(Math.PI)) + Array.from({ length: p }, (_, j) => lgammaFn(x - 0.5 * j)).reduce((u, w) => u + w, 0)
        return wrap(C.mapT(E, t, f, (x) => Array.from({ length: p }, (_, j) => polygammaFn(0, x - 0.5 * j)).reduce((u, w) => u + w, 0), 'MvlgammaBackward0'))
      }),
      gammainc: viaMethod('gammainc', 'igamma'), gammaincc: viaMethod('gammaincc', 'igammac'),
      zeta: fnv('zeta', (args, kw) => {
        const a = bindPy(h, 'zeta', 'input other * out=None', args, kw)
        const [x, q] = [operand(h, a.input!, asTT(a.other!)?.dtype), operand(h, a.other!, asTT(a.input!)?.dtype)]
        if (!x || !q) throw invalidArgs(h, 'zeta', args, kw)
        const d = C.promote(x.t.dtype, q.t.dtype, x.weak, q.weak)
        return wrap(C.map2T({ grad: false } as Engine, x.t, q.t, (s0, q0) => {
          if (s0 === 1) return Infinity
          if (s0 < 1 || q0 <= 0) return NaN
          let sum = 0
          let y = q0
          while (y < 30) {
            sum += y ** -s0
            y += 1
          }
          return sum + y ** (1 - s0) / (s0 - 1) + y ** -s0 / 2 + (s0 * y ** (-s0 - 1)) / 12
        }, () => 0, () => 0, isFloatT(d) ? d : C.DEFAULTS.float, 'ZetaBackward0'))
      }),
      i0: viaMethod('i0'),
      i0e: mapped('i0e', (x) => besselI(0, x, true), (x) => besselI(1, x, true) - Math.sign(x) * besselI(0, x, true), 'SpecialI0EBackward0'),
      i1: mapped('i1', (x) => besselI(1, x, false), (x) => (x === 0 ? 0.5 : besselI(0, x, false) - besselI(1, x, false) / x), 'SpecialI1Backward0'),
      i1e: mapped('i1e', (x) => besselI(1, x, true), (x) => (x === 0 ? 0.5 : besselI(0, x, true) - besselI(1, x, true) * (Math.sign(x) + 1 / x)), 'SpecialI1EBackward0'),
      sinc: viaMethod('sinc'), xlogy: viaMethod('xlogy'),
      xlog1py: fnv('xlog1py', (args, kw) => {
        const a = bindPy(h, 'xlog1py', 'input other * out=None', args, kw)
        const x = operand(h, a.input!, asTT(a.other!)?.dtype)
        const y = operand(h, a.other!, asTT(a.input!)?.dtype)
        if (!x || !y) throw h.err('TypeError', 'xlog1py() received invalid arguments')
        const d = C.promote(x.t.dtype, y.t.dtype, x.weak, y.weak)
        return wrap(C.map2T(E, x.t, y.t, (p, q) => (p === 0 && !Number.isNaN(q) ? 0 : p * Math.log1p(q)), (_p, q) => Math.log1p(q), (p, q) => p / (1 + q), isFloatT(d) ? d : C.DEFAULTS.float, 'SpecialXlog1PyBackward'))
      }),
      softmax: viaMethod('softmax'), log_softmax: viaMethod('log_softmax'),
      round: viaMethod('round'), log1p: viaMethod('log1p'), expm1: viaMethod('expm1'), exp2: viaMethod('exp2'), logsumexp: viaMethod('logsumexp'),
    },
  }

  /** torch.autograd.Function: forward under no_grad, backward called from the graph */
  const functionApply = fnv('apply', (args, kw) => {
    const cls = args[0] as ClassValue
    const inputs = args.slice(1)
    const fnsOf = (name: string) => {
      const m = h.getAttrOr(cls, name)
      if (!m) throw h.err('NotImplementedError', name === 'backward' ? 'You must implement either the backward or vjp method for your custom autograd.Function to use it with backward mode AD.' : 'You must implement the forward function for custom autograd.Function.')
      return m
    }
    const forward = fnsOf('forward')
    const ctx = new FunctionCtx()
    const was = E.grad
    E.grad = false
    let result: Value
    try {
      result = h.call(forward, [py.obj(ctx), ...inputs], kw)
    } finally {
      E.grad = was
    }
    const tracked = inputs.map(asTT).filter((t): t is TT => !!t && t.requiresGrad)
    if (!E.grad || !tracked.length) return result
    const single = result.k !== 'tuple'
    const outs = single ? [result] : (result as { items: Value[] }).items
    const name = `${cls.name}Backward`
    const inTs = inputs.map((v) => asTT(v))
    const wrapped = outs.map((v, q) => {
      const t = asTT(v)
      if (!t) return v
      const fresh = TT.of(t.values(), t.shape, t.dtype)
      const node = C.record(E, fresh, name, inTs, (g) => {
        const gt = outs.map((other, k) => {
          const ot = asTT(other)
          return ot ? wrap(TT.of(k === q ? g : new Array<number>(ot.size).fill(0), ot.shape, ot.dtype)) : py.NONE
        })
        const back = fnsOf('backward')
        const res = h.call(back, [py.obj(ctx), ...gt])
        const list = res.k === 'tuple' ? res.items : [res]
        if (list.length !== inTs.length && inTs.filter(Boolean).length > 1) throw h.err('RuntimeError', `function ${name} returned an incorrect number of gradients (expected ${inTs.length}, got ${list.length})`)
        return inTs.map((_, i) => {
          const r = list[i]
          return r && asTT(r) ? [...asTT(r)!.values()] : null
        })
      })
      return wrap(node)
    })
    return single ? wrapped[0] : py.tuple(wrapped)
  })
  const autogradFunction: ClassValue = {
    k: 'class', name: 'torch.autograd.function.Function', bases: [], builtin: true, host: null,
    ns: new Map<string, Value>([['apply', { k: 'descr', kind: 'class', f: functionApply }]]),
  }

  /** a fresh leaf per input, so the caller's tensors stay as they were */
  const leafCopies = (v: Value): { copies: Value[]; single: boolean } => {
    const single = !!asTT(v)
    const list = single ? [v] : h.iterate(v)
    return {
      single,
      copies: list.map((x) => {
        const t = T(x, 'inputs')
        if (!isFloatT(t.dtype)) throw h.err('RuntimeError', 'Only Tensors of floating point and complex dtype can require gradients')
        const c = TT.of(t.values(), t.shape, t.dtype)
        c.requiresGrad = true
        return wrap(c)
      }),
    }
  }
  const autogradFunctional: Value = {
    k: 'module', name: 'torch.autograd.functional', attrs: {
      jacobian: fnv('jacobian', (args, kw) => {
        const a = bindPy(h, 'jacobian', "func inputs create_graph=False strict=False vectorize=False strategy='reverse-mode'", args, kw)
        if (bool(a.create_graph, false)) throw h.err('NotImplementedError', 'autograd.functional.jacobian: create_graph=True is not available in this sandbox')
        const { copies, single } = leafCopies(a.inputs!)
        const out = T(h.call(a.func!, copies), 'jacobian')
        const was = E.grad
        const rows = copies.map((v) => {
          const x = asTT(v)!
          const cols: number[] = []
          for (let f = 0; f < out.size; f++) {
            const seed = new Array<number>(out.size).fill(0)
            seed[f] = 1
            x.grad = null
            E.grad = true
            if (out.requiresGrad) C.backward(E, out, seed, true)
            E.grad = was
            cols.push(...(x.grad ? (x.grad as TT).values() : new Array<number>(x.size).fill(0)))
          }
          return wrap(TT.of(cols, [...out.shape, ...x.shape], x.dtype))
        })
        return single ? rows[0] : py.tuple(rows)
      }),
      hessian: fnv('hessian', (args, kw) => {
        const a = bindPy(h, 'hessian', "func inputs create_graph=False strict=False vectorize=False outer_jacobian_strategy='reverse-mode'", args, kw)
        if (bool(a.create_graph, false)) throw h.err('NotImplementedError', 'autograd.functional.hessian: create_graph=True is not available in this sandbox')
        const { copies, single } = leafCopies(a.inputs!)
        const out = T(h.call(a.func!, copies), 'hessian')
        if (out.size !== 1) throw h.err('RuntimeError', 'The Tensor returned by the function given to hessian should contain a single element')
        const xs = copies.map((v) => asTT(v)!)
        const first = C.backwardGraph(E, out, null)
        const table = xs.map((xi) => xs.map((xj) => {
          const gi = first.get(xi)
          const cols: number[] = []
          for (let f = 0; f < xi.size; f++) {
            if (gi?.requiresGrad) {
              const seed = new Array<number>(gi.size).fill(0)
              seed[f] = 1
              cols.push(...(C.backwardGraph(E, gi, TT.of(seed, gi.shape, gi.dtype)).get(xj)?.values() ?? new Array<number>(xj.size).fill(0)))
            } else cols.push(...new Array<number>(xj.size).fill(0))
          }
          return wrap(TT.of(cols, [...xi.shape, ...xj.shape], xj.dtype))
        }))
        return single ? table[0][0] : py.tuple(table.map((row) => py.tuple(row)))
      }),
    },
  }
  // ---- torch.autograd
  const autograd: Value = {
    k: 'module', name: 'torch.autograd', attrs: {
      Function: autogradFunction,
      functional: autogradFunctional,
      inference_mode: gradModeType('inference_mode', false),
      set_detect_anomaly: fnv('set_detect_anomaly', () => py.NONE),
      detect_anomaly: { k: 'type', name: 'detect_anomaly', call: () => py.obj(new GradMode(E.grad)) },
      is_tensor_like: fnv('is_tensor_like', (args) => py.bool(!!asTT(args[0]))),
      gradcheck: fnv('gradcheck', (args, kw) => {
        const a = bindPy(h, 'gradcheck', 'func inputs * eps=1e-06 atol=1e-05 rtol=0.001 raise_exception=True nondet_tol=0.0 check_undefined_grad=True check_grad_dtypes=False check_batched_grad=False check_batched_forward_grad=False check_forward_ad=False check_backward_ad=True fast_mode=False masked=None', args, kw)
        const inputs = asTT(a.inputs!) ? [a.inputs!] : h.iterate(a.inputs!)
        const ts = inputs.map((v) => T(v, 'gradcheck'))
        const [eps, atol, rtol] = [h.num(a.eps!), h.num(a.atol!), h.num(a.rtol!)]
        const sumOf = () => h.call(a.func!, inputs)
        const outs = (v: Value) => (asTT(v) ? [asTT(v)!] : h.iterate(v).map((x) => T(x, 'gradcheck')))
        for (const t of ts) if (t.requiresGrad && t.dtype !== 'float64') throw h.err('RuntimeError', 'gradcheck expects the differentiated inputs to be double precision') 
        const base = outs(sumOf())
        let good = true
        for (const [q, o] of base.entries()) for (let f = 0; f < o.size; f++) {
          const seed = new Array<number>(o.size).fill(0)
          seed[f] = 1
          const fresh = outs(sumOf())[q]
          for (const t of ts) t.grad = null
          C.backward(E, fresh, seed, false)
          for (const t of ts) {
            if (!t.requiresGrad) continue
            const ana = t.grad ? t.grad.values() : new Array<number>(t.size).fill(0)
            for (let i = 0; i < t.size; i++) {
              const start = t.values()
              const at = (d: number) => {
                const next = [...start]
                next[i] += d
                C.writeInto(t, next)
                const was = E.grad
                E.grad = false
                try {
                  return outs(sumOf())[q].values()[f]
                } finally {
                  E.grad = was
                  C.writeInto(t, start)
                }
              }
              const num0 = (at(eps) - at(-eps)) / (2 * eps)
              if (Math.abs(num0 - ana[i]) > atol + rtol * Math.abs(num0)) good = false
            }
          }
        }
        for (const t of ts) t.grad = null
        if (!good && bool(a.raise_exception, true)) throw h.err('GradcheckError', `Jacobian mismatch for output 0 with respect to input 0,`)
        return py.bool(good)
      }),
      no_grad: gradModeType('no_grad', false),
      enable_grad: gradModeType('enable_grad', true),
      set_grad_enabled: fnv('set_grad_enabled', (args, kw) => {
        const m = new GradMode(h.truthy(args[0] ?? kw.mode))
        m.enter(h)
        m.applied = true
        return py.obj(m)
      }),
      grad: fnv('grad', (args, kw) => {
        const a = bind(h, 'autograd.grad', 'outputs inputs grad_outputs retain_graph create_graph only_inputs allow_unused materialize_grads', args, kw)
        const outV = a.outputs ?? py.NONE
        const outs = asTT(outV) ? [outV] : h.iterate(outV)
        if (outs.length !== 1) throw h.err('NotImplementedError', 'autograd.grad: more than one output is not available in this sandbox: sum them first')
        const root = T(outs[0], 'autograd.grad outputs')
        const inV = a.inputs ?? py.NONE
        const inputs = (asTT(inV) ? [inV] : h.iterate(inV)).map((v) => T(v, 'autograd.grad inputs'))
        if (!root.requiresGrad) throw h.err('RuntimeError', 'element 0 of tensors does not require grad and does not have a grad_fn')
        for (const x of inputs) if (!x.requiresGrad) throw h.err('RuntimeError', 'One of the differentiated Tensors does not require grad')
        const gv = a.grad_outputs ?? py.NONE
        const gout = asTT(gv) ?? (gv.k === 'tuple' || gv.k === 'list' ? asTT(gv.items[0] ?? py.NONE) : null)
        if (bool(a.create_graph, false)) {
          const grads = C.backwardGraph(E, root, gout)
          const res = inputs.map((x) => grads.get(x) ?? null)
          const unused = res.findIndex((g) => !g)
          if (unused >= 0 && !bool(a.allow_unused, false)) throw h.err('RuntimeError', `The differentiated Tensor at index ${unused} appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.`)
          return py.tuple(res.map((g) => (g ? wrap(g) : py.NONE)))
        }
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
          const unused = res.findIndex((g) => !g)
          if (unused >= 0 && !bool(a.allow_unused, false)) throw h.err('RuntimeError', `The differentiated Tensor at index ${unused} appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.`)
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
        refuse('autograd.backward', a, 'inputs', isNone)
        if (bool(a.create_graph, false)) {
          const list0 = asTT(a.tensors ?? py.NONE) ? [a.tensors!] : h.iterate(a.tensors ?? py.NONE)
          const gs0 = a.grad_tensors && a.grad_tensors.k !== 'none' ? (asTT(a.grad_tensors) ? [a.grad_tensors] : h.iterate(a.grad_tensors)) : []
          list0.forEach((v, i) => h.call(h.getAttr(v, 'backward'), [gs0[i] ?? py.NONE], { create_graph: py.bool(true) }))
          return py.NONE
        }
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
    knownKw('arange', kw, ['start', 'end', 'step'])
    const given = kw.start || kw.end || kw.step
    const a = given ? [kw.start ?? py.int(0), kw.end ?? args[0], kw.step ?? py.int(1)] : args
    if (!a.length || a.includes(undefined as unknown as Value)) throw invalidArgs(h, 'arange', args, kw)
    if (a.length > 3) throw invalidArgs(h, 'arange', args, kw)
    a.forEach((v, i) => {
      if (!isNum(v) && !asTT(v)) throw h.err('TypeError', `arange(): argument '${a.length === 1 ? 'end' : ['start', 'end', 'step'][i]}' (position ${i + 1}) must be Number, not ${typeName(v)}`)
    })
    const nums = a.map((v) => h.num(v))
    const [start, end, step] = nums.length === 1 ? [0, nums[0], 1] : [nums[0], nums[1], nums[2] ?? 1]
    if (step === 0) throw h.err('RuntimeError', 'step must be nonzero')
    if (!Number.isFinite(start) || !Number.isFinite(end)) throw h.err('RuntimeError', `unsupported range: ${pyFloatText(start)} -> ${pyFloatText(end)}`)
    if ((step > 0 && end < start) || (step < 0 && end > start)) throw h.err('RuntimeError', 'upper bound and lower bound inconsistent with step sign')
    const n = Math.max(0, Math.ceil((end - start) / step))
    if (n > h.maxSize) throw h.err('RuntimeError', `arange of ${n} elements exceeds the sandbox limit`)
    const anyFloat = a.some((v) => v.k === 'float' || (asTT(v) && isFloatT(asTT(v)!.dtype)))
    const out = TT.of(Array.from({ length: n }, (_, i) => start + i * step), [n], dtypeArg(h, kw.dtype) ?? (anyFloat ? C.DEFAULTS.float : 'int64'))
    return finish(out, kw, 'arange')
  })
  const pyFloatText = (x: number) => (Number.isNaN(x) ? 'nan' : x === Infinity ? 'inf' : x === -Infinity ? '-inf' : String(x))

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
    const given = args[0] ?? kw.tensors
    if (!given) throw h.err('TypeError', `${name}() missing 1 required positional arguments: "tensors"`)
    if (asTT(given) || (given.k !== 'list' && given.k !== 'tuple')) throw h.err('TypeError', `${name}(): argument 'tensors' (position 1) must be tuple of Tensors, not ${argType(given)}`)
    const vals = h.iterate(given)
    vals.forEach((v, i) => {
      if (!asTT(v)) throw h.err('TypeError', `expected Tensor as element ${i} in argument 0, but got ${typeName(v)}`)
    })
    if (!vals.length) {
      if (name === 'stack') throw h.err('RuntimeError', 'stack expects a non-empty TensorList')
      if (name === 'cat' || name === 'concat' || name === 'concatenate') throw h.err('ValueError', `torch.cat(): expected a non-empty list of Tensors`)
    }
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

  const VIA_METHOD = ['arctan2', 'conj', 'isreal', 'imag', 'angle', 'nextafter', 'frexp', 'igamma', 'igammac', 'addmm', 'addmv', 'addr', 'addbmm', 'baddbmm', 'split_with_sizes', 'put', 'index_reduce', 'is_signed', 'asinh', 'acosh', 'atanh', 'arcsin', 'arccos', 'arctan', 'arcsinh', 'arccosh', 'arctanh', 'deg2rad', 'rad2deg', 'sinc', 'sgn', 'fix', 'lgamma', 'digamma', 'erfinv', 'i0', 'positive', 'signbit', 'logit', 'nan_to_num', 'hypot', 'logaddexp', 'logaddexp2', 'copysign', 'xlogy', 'fmax', 'fmin', 'gcd', 'lcm', 'heaviside', 'float_power', 'ldexp', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_left_shift', 'bitwise_right_shift', 'bitwise_not', 'addcmul', 'addcdiv', 'isin', 'is_nonzero', 'is_same_size', 'nansum', 'nanmean', 'nanmedian', 'quantile', 'nanquantile', 'aminmax', 'logsumexp', 'cummax', 'cummin', 'logcumsumexp', 'diff', 'histc', 'unique_consecutive', 'msort', 'argwhere', 'histogram', 'trapezoid', 'trapz', 'cumulative_trapezoid', 'cov', 'corrcoef', 'tensor_split', 'hsplit', 'vsplit', 'dsplit', 'rot90', 'cross', 'kron', 'vdot', 'adjoint', 'diagflat', 'diag_embed', 'select', 'index_add', 'index_copy', 'index_put', 'scatter_reduce', 'masked_scatter', 'dist', 'renorm', 'add', 'sub', 'div', 'mm', 'dot', 'bmm', 'mv', 'prod', 'sort', 'argsort', 'topk', 'kthvalue', 'trunc', 'frac', 'floor', 'ceil', 'sign', 'log2', 'log10', 'log1p', 'expm1', 'exp2', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'erf', 'erfc', 'reciprocal', 'rsqrt', 'neg', 'negative', 'remainder', 'fmod', 'floor_divide', 'true_divide', 'divide', 'multiply', 'subtract', 'atan2', 'all', 'any', 'count_nonzero', 'nonzero', 'masked_select', 'masked_fill', 'isnan', 'isinf', 'isfinite', 'isposinf', 'isneginf', 'logical_and', 'logical_or', 'logical_xor', 'logical_not', 'allclose', 'isclose', 'equal', 'is_floating_point', 'cumsum', 'cumprod', 'unique', 'bincount', 'tile', 'repeat_interleave', 'gather', 'scatter', 'scatter_add', 'index_select', 'index_fill', 'take', 'take_along_dim', 'narrow', 'split', 'chunk', 'unbind', 'flip', 'fliplr', 'flipud', 'roll', 'diag', 'diagonal', 'tril', 'triu', 'trace', 'mode', 'median', 'amax', 'amin', 'norm', 'broadcast_to', 'swapaxes', 'swapdims', 'moveaxis', 'movedim', 'ravel', 'unflatten', 'outer', 'inner', 'ger', 'lerp', 'clamp', 'clip', 'clamp_min', 'clamp_max', 'inverse', 'det', 'matrix_power', 'sum', 'mean', 'std', 'var', 'argmax', 'argmin', 'round', 'squeeze', 'unsqueeze', 'transpose', 'permute', 't', 'clone', 'detach', 'softmax', 'log_softmax']

  const makeWindow = (a: Args, f: (n: number, R: (x: number) => number) => number[]): Value => {
    const dtype = dtypeArg(h, a.dtype) ?? C.DEFAULTS.float
    const R = dtype === 'float64' ? (x: number) => x : Math.fround
    const N = h.toInt(a.window_length!)
    if (N < 0) throw h.err('RuntimeError', `window_length must be non-negative`)
    if (N <= 1) return wrap(TT.of(new Array<number>(N).fill(1), [N], dtype))
    const periodic = bool(a.periodic, true)
    const v = f(periodic ? N + 1 : N, R)
    return wrap(TT.of(periodic ? v.slice(0, N) : v, [N], dtype))
  }
  const windowFn = (name: string, f: (n: number, R: (x: number) => number) => number[]) => fnv(name, (args, kw) => makeWindow(bindPy(h, name, 'window_length periodic=True * dtype=None layout=None device=None pin_memory=False requires_grad=False', args, kw), f))
  const hamming = (n: number, alpha: number, beta: number, R: (x: number) => number) => Array.from({ length: n }, (_, i) => R(R(R(Math.cos(R(i * ((Math.PI * 2) / (n - 1))))) * -beta) + alpha))
  const triIndices = (name: string, keep: (r: number, c: number, k: number) => boolean) => fnv(name, (args, kw) => {
    const a = bindPy(h, name, 'row col offset=0 * dtype=None device=None layout=None', args, kw)
    const [rows, cols, k] = [h.toInt(a.row!), h.toInt(a.col!), h.toInt(a.offset!)]
    const [rs, cs]: [number[], number[]] = [[], []]
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) if (keep(r, c, k)) {
      rs.push(r)
      cs.push(c)
    }
    return wrap(TT.of([...rs, ...cs], [2, rs.length], dtypeArg(h, a.dtype) ?? 'int64'))
  })
  const stdMean = (which: 'std' | 'var') => fnv(`${which}_mean`, (args, kw) => {
    const self = args[0] ?? kw.input
    const { input: _i, ...rest } = kw
    const more = args.slice(1)
    const onlyDim = (x: Value[]) => x.filter((v) => v.k !== 'bool').slice(0, 2)
    const meanKw: Kw = {}
    if (rest.keepdim) meanKw.keepdim = rest.keepdim
    if (rest.dim) meanKw.dim = rest.dim
    const name = which === 'std' ? 'StdMeanBackward0' : 'VarMeanBackward0'
    const tag = (v: Value) => {
      const t = asTT(v)
      if (t?.fn) t.fn.name = name
      return v
    }
    return py.tuple([tag(h.call(h.getAttr(self, which), more, rest)), tag(h.call(h.getAttr(self, 'mean'), onlyDim(more).slice(0, 1), meanKw))])
  })
  const atLeast3 = (t: TT): TT => (t.ndim >= 3 ? t : C.reshapeT(E, t, t.ndim === 2 ? [...t.shape, 1] : t.ndim === 1 ? [1, t.size, 1] : [1, 1, 1], false))
  const atLeastFn = (name: string, d: 1 | 2 | 3) => fnv(name, (args) => {
    const many = args.length !== 1 || !asTT(args[0])
    const list = args.length === 1 && !asTT(args[0]) ? h.iterate(args[0]) : args
    const out = list.map((v) => {
      const t = T(v, name)
      const shaped = d === 3 ? atLeast3(t) : atLeast(t, d)
      return shaped === t ? v : wrap(named(shaped, 'UnsqueezeBackward0'))
    })
    return many ? py.tuple(out) : out[0]
  })
  /** the insertion points for values in a sorted last dimension */
  const searchSorted = (seq: TT, input: Value, right: boolean, int32: boolean): Value => {
    const scalar = !asTT(input)
    const vals = scalar ? TT.scalar(h.num(input), seq.dtype) : asTT(input)!
    const n = seq.shape[seq.ndim - 1]
    const sv = seq.values()
    const lead = seq.shape.slice(0, -1)
    if (seq.ndim > 1 && !vals.ndim) throw h.err('RuntimeError', 'torch.searchsorted(): input value can be a scalar only when sorted_sequence is 1-D')
    if (seq.ndim > 1 && vals.shape.slice(0, -1).join() !== lead.join()) throw h.err('RuntimeError', `torch.searchsorted(): boundaries tensor should be 1 dimension or the first N-1 dimensions of boundaries tensor and input value tensor must match, but we got boundaries tensor [${seq.shape.join(', ')}] and input value tensor [${vals.shape.join(', ')}]`)
    const per = seq.ndim > 1 ? vals.shape[vals.ndim - 1] : vals.size
    const out = vals.values().map((v, f) => {
      const base = seq.ndim > 1 ? Math.floor(f / per) * n : 0
      let [lo, hi] = [0, n]
      while (lo < hi) {
        const mid = (lo + hi) >> 1
        if (right ? sv[base + mid] <= v : sv[base + mid] < v) lo = mid + 1
        else hi = mid
      }
      return lo
    })
    return wrap(TT.of(out, vals.shape, int32 ? 'int32' : 'int64'))
  }

  const torch: Record<string, Value> = {
    ...dtypeVals,
    ...Object.fromEntries(VIA_METHOD.map((n) => [n, viaMethod(n)])),
    __version__: py.str('2.14.0'),
    pi: py.float(Math.PI), e: py.float(Math.E), inf: py.float(Infinity), nan: py.float(NaN),
    Tensor: legacyCtor('Tensor', C.DEFAULTS.float),
    ...Object.fromEntries(Object.entries(LEGACY).map(([name, d]) => [name, legacyCtor(name, d)])),
    Generator: { k: 'type', name: 'Generator', call: (args, kw) => {
      const device = kw.device ?? args[0]
      if (device && h.str(device).startsWith('cuda')) throw h.err('RuntimeError', 'Device type CUDA is not supported for torch.Generator() api.')
      return py.obj(new GeneratorObj(new SandboxRandom(1)))
    } },
    Size: { k: 'type', name: 'Size', call: (args) => py.obj(new SizeObj(args[0] ? h.iterate(args[0]).map((v) => h.toInt(v)) : [])) },
    tensor: fnv('tensor', (args, kw) => {
      if (!args.length && !kw.data) throw h.err('TypeError', 'tensor() missing 1 required positional arguments: "data"')
      if (args.length > 1) throw h.err('TypeError', `tensor() takes 1 positional argument but ${args.length} were given`)
      for (const k in kw) if (!['data', 'dtype', 'device', 'requires_grad', 'pin_memory', 'names'].includes(k)) throw h.err('TypeError', `tensor() got an unexpected keyword argument '${k}'`)
      onlyCpu(h, kw.device)
      return finish(tensorFrom(h, args[0] ?? kw.data, dtypeArg(h, kw.dtype)), kw, 'tensor')
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
      knownKw('full', kw, ['fill_value'])
      const size = args[0] ?? kw.size
      const fill = args[1] ?? kw.fill_value
      if (!size) throw h.err('TypeError', 'full() missing 1 required positional arguments: "size"')
      if (!fill) throw h.err('TypeError', 'full() missing 1 required positional arguments: "fill_value"')
      if (size.k === 'int') throw h.err('TypeError', "full(): argument 'size' (position 1) must be tuple of ints, not int")
      const dims = sizeArg('full', [size], {})
      if (!isNum(fill) && !asTT(fill)) throw h.err('TypeError', `full(): argument 'fill_value' (position 2) must be Number, not ${typeName(fill)}`)
      return finish(TT.of(new Array(prod(dims)).fill(h.num(fill)), dims, dtypeArg(h, kw.dtype) ?? (fill.k === 'int' ? 'int64' : fill.k === 'bool' ? 'bool' : C.DEFAULTS.float)), kw, 'full')
    }),
    rand: filled('rand', (rng) => rng.next(), null, 'check_uniform_bounds'),
    randn: filled('randn', (rng) => rng.normal(), null, 'normal_kernel_cpu'),
    randint: fnv('randint', (args, kw) => {
      for (const k in kw) if (!FACTORY_KW.includes(k) && k !== 'low' && k !== 'high') throw invalidArgs(h, 'randint', args, kw)
      const size0 = kw.size ?? args[args.length - 1]
      if (args.length + Object.keys(kw).length < 2 || !size0) throw invalidArgs(h, 'randint', args, kw)
      if (size0.k === 'int' && !kw.size) throw h.err('TypeError', `randint(): argument 'size' (position ${args.length}) must be tuple of ints, not int`)
      const positional = kw.size ? args : args.slice(0, -1)
      const size = size0
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
      const nV = args[0] ?? kw.n
      if (nV && nV.k !== 'int') throw h.err('TypeError', `randperm(): argument 'n' (position 1) must be int, not ${typeName(nV)}`)
      const n = h.toInt(nV)
      if (n < 0) throw h.err('RuntimeError', `Trying to create tensor with negative dimension ${n}: [${n}]`)
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
        if (s < 0) throw h.err('RuntimeError', `normal expects std >= 0.0, but found std ${s}`)
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
      knownKw('eye', kw, ['n', 'm'])
      const nV = args[0] ?? kw.n
      if (nV && nV.k !== 'int') throw h.err('TypeError', `eye(): argument 'n' (position 1) must be int, not ${typeName(nV)}`)
      const n = h.toInt(nV)
      const m = args[1] ?? kw.m ? h.toInt((args[1] ?? kw.m)!) : n
      if (n < 0) throw h.err('RuntimeError', `n must be greater or equal to 0, got ${n}`)
      if (m < 0) throw h.err('RuntimeError', `m must be greater or equal to 0, got ${m}`)
      return finish(TT.of(Array.from({ length: n * m }, (_, i) => (Math.floor(i / m) === i % m ? 1 : 0)), [n, m], dtypeArg(h, kw.dtype) ?? C.DEFAULTS.float), kw, 'eye')
    }),
    arange,
    linspace: fnv('linspace', (args, kw) => {
      knownKw('linspace', kw, ['start', 'end', 'steps'])
      const [start, end, steps] = [args[0] ?? kw.start, args[1] ?? kw.end, args[2] ?? kw.steps]
      if (!start || !end || !steps || steps.k !== 'int') throw invalidArgs(h, 'linspace', args, kw)
      const [a, b] = [h.num(start), h.num(end)]
      const n = steps.v
      if (n < 0) throw h.err('RuntimeError', 'number of steps must be non-negative')
      const R = (dtypeArg(h, kw.dtype) ?? C.DEFAULTS.float) === 'float64' ? (x: number) => x : Math.fround
      const stepF = n > 1 ? R(R(b - a) / (n - 1)) : 0
      // from both ends towards the middle, as PyTorch's kernel does
      return finish(TT.of(Array.from({ length: n }, (_, i) => (n === 1 ? a : i < n / 2 ? R(a) + i * stepF : i === n - 1 ? b : R(b) - (n - 1 - i) * stepF)), [n], dtypeArg(h, kw.dtype) ?? C.DEFAULTS.float), kw, 'linspace')
    }),
    logspace: fnv('logspace', (args, kw) => {
      knownKw('logspace', kw, ['start', 'end', 'steps', 'base'])
      const [start, end, steps] = [args[0] ?? kw.start, args[1] ?? kw.end, args[2] ?? kw.steps]
      if (!start || !end || !steps || steps.k !== 'int') throw invalidArgs(h, 'logspace', args, kw)
      const [a, b] = [h.num(start), h.num(end)]
      const n = steps.v
      if (n < 0) throw h.err('RuntimeError', 'number of steps must be non-negative')
      const base = num(args[3] ?? kw.base, 10)
      return finish(TT.of(Array.from({ length: n }, (_, i) => base ** (n === 1 ? a : a + ((b - a) * i) / (n - 1))), [n], dtypeArg(h, kw.dtype) ?? C.DEFAULTS.float), kw, 'logspace')
    }),
    zeros_like: like('zeros_like', () => 0), ones_like: like('ones_like', () => 1), empty_like: like('empty_like', () => 0),
    rand_like: like('rand_like', () => h.rng.next(), true), randn_like: like('randn_like', () => h.rng.normal(), true),
    full_like: fnv('full_like', (args, kw) => {
      const t = T(args[0], 'full_like')
      const v = h.num(args[1] ?? kw.fill_value)
      return finish(TT.of(t.values().map(() => v), t.shape, dtypeArg(h, kw.dtype) ?? t.dtype), kw, 'full_like')
    }),
    manual_seed: fnv('manual_seed', (args, kw) => {
      S.seed = h.toInt(kw.seed ?? args[0])
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
      const tooBig = ts.find((t) => t.ndim > 1)
      if (tooBig) throw h.err('RuntimeError', `torch.meshgrid: Expected 0D or 1D tensor in the tensor list but got: ${cppDump(tooBig)}`)
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
    device: { k: 'type', name: 'device', call: (args, kw) => {
      const first = args[0] ?? kw.type ?? kw.device
      if (!first || (first.k !== 'str' && first.k !== 'int' && !(first.k === 'obj' && first.o instanceof DeviceObj))) throw invalidArgs(h, 'device', args, kw)
      const index = args[1] ?? kw.index
      if (index) {
        const i = h.toInt(index)
        if (i < 0) throw h.err('RuntimeError', 'Device index must not be negative')
        return py.obj(new DeviceObj(deviceOf(h, first).type, i))
      }
      return py.obj(deviceOf(h, first))
    } },
    save: fnv('save', (args, kw) => {
      const a = bindPy(h, 'save', 'obj f pickle_module=None pickle_protocol=2 _use_new_zipfile_serialization=True _disable_byteorder_record=False', args, kw)
      S.files.set(h.str(a.f!), snapshot(a.obj!, new Map()) as Value)
      return py.NONE
    }),
    load: fnv('load', (args, kw) => {
      const a = bindPy(h, 'load', 'f map_location=None pickle_module=None * weights_only=None mmap=None', args, kw)
      const path = h.str(a.f!)
      const v = S.files.get(path)
      if (!v) throw h.err('FileNotFoundError', `[Errno 2] No such file or directory: '${path}'`)
      // the default weights_only=True takes tensors, numbers and containers, not a pickled module
      const held = holdsModule(v)
      if (bool(a.weights_only, true) && held) {
        const G = held
        throw h.err('UnpicklingError', `Weights only load failed. This file can still be loaded, to do so you have two options, \u001b[1mdo those steps only if you trust the source of the checkpoint\u001b[0m. \n\t(1) In PyTorch 2.6, we changed the default value of the \`weights_only\` argument in \`torch.load\` from \`False\` to \`True\`. Re-running \`torch.load\` with \`weights_only\` set to \`False\` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.\n\t(2) Alternatively, to load with \`weights_only=True\` please check the recommended steps in the following error message.\n\tWeightsUnpickler error: Unsupported global: GLOBAL ${G} was not an allowed global by default. Please use \`torch.serialization.add_safe_globals([${G}])\` or the \`torch.serialization.safe_globals([${G}])\` context manager to allowlist this global if you trust this class/function.\n\nCheck the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.`)
      }
      return snapshot(v, new Map()) as Value
    }),
    einsum, special,
    real: fnv('real', (args) => h.getAttr(args[0], 'real')),
    newaxis: py.NONE,
    strided: py.obj(new DtypeLike('torch.strided')), preserve_format: py.obj(new DtypeLike('torch.preserve_format')), contiguous_format: py.obj(new DtypeLike('torch.contiguous_format')),
    channels_last: py.obj(new DtypeLike('torch.channels_last')), channels_last_3d: py.obj(new DtypeLike('torch.channels_last_3d')),
    dtype: { k: 'type', name: 'dtype', call: () => { throw h.err('TypeError', "cannot create 'torch.dtype' instances") } },
    rsub: fnv('rsub', (args, kw) => {
      const a = bindPy(h, 'rsub', 'input other * alpha=1', args, kw)
      const [x, y] = [T(a.input, 'rsub'), tensorArg(h, a.other, 'rsub', 'other', 2)]
      return wrap(C.binary(E, '-', y, h.num(a.alpha!) === 1 ? x : C.binary(E, '*', x, TT.scalar(h.num(a.alpha!), x.dtype), false, true)))
    }),
    typename: fnv('typename', (args) => py.str(asTT(args[0]) ? `torch.${C.typeLabel(asTT(args[0])!.dtype)}Tensor` : typeName(args[0]))),
    asarray: fnv('asarray', (args, kw) => {
      const { copy: _c, ...rest } = kw
      const t = asTT(args[0])
      if (t) return dtypeArg(h, rest.dtype) ? wrap(C.toDtype(E, t, dtypeArg(h, rest.dtype)!)) : args[0]
      return args[0]?.k === 'array' && !rest.dtype ? wrap(fromNd(args[0].a, true)) : wrap(tensorFrom(h, args[0], dtypeArg(h, rest.dtype)))
    }),
    scalar_tensor: fnv('scalar_tensor', (args, kw) => wrap(TT.scalar(h.num(args[0] ?? kw.s), dtypeArg(h, kw.dtype) ?? C.DEFAULTS.float))),
    binomial: fnv('binomial', (args, kw) => {
      const a = bindPy(h, 'binomial', 'count prob generator=None', args, kw)
      const [n, p] = [T(a.count, 'binomial'), tensorArg(h, a.prob, 'binomial', 'prob', 2)]
      const rng = rngOf(h, a.generator)
      const shape = broadcastShapes([n.shape, p.shape])
      const [in0, ip0] = [C.bcastIndex(n.shape, shape), C.bcastIndex(p.shape, shape)]
      const [nv, pv] = [n.values(), p.values()]
      return wrap(TT.of(Array.from({ length: prod(shape) }, (_, f) => {
        let hits = 0
        for (let k = 0; k < nv[in0[f]]; k++) if (rng.next() < pv[ip0[f]]) hits++
        return hits
      }), shape, C.DEFAULTS.float))
    }),
    poisson: fnv('poisson', (args, kw) => {
      const a = bindPy(h, 'poisson', 'input generator=None', args, kw)
      const rng = rngOf(h, a.generator)
      const t = T(a.input, 'poisson')
      return wrap(TT.of(t.values().map((lam) => {
        const limit = Math.exp(-lam)
        let [k, prod0] = [0, rng.next()]
        while (prod0 > limit) {
          k++
          prod0 *= rng.next()
        }
        return k
      }), t.shape, t.dtype))
    }),
    unravel_index: fnv('unravel_index', (args, kw) => {
      const a = bindPy(h, 'unravel_index', 'indices shape', args, kw)
      const t = asTT(a.indices!) ?? TT.scalar(h.toInt(a.indices!), 'int64')
      const shape = toDims(h, [a.shape!])
      const st = cStrides(shape)
      return py.tuple(shape.map((n, d) => wrap(TT.of(t.values().map((f) => Math.floor(f / st[d]) % n), t.shape, t.dtype))))
    }),
    nuclear_norm: fnv('nuclear_norm', (args, kw) => {
      const a = bindPy(h, 'nuclear_norm', 'input dim=None keepdim=False', args, kw)
      return h.call(linalgFns.matrix_norm, [a.input!, py.str('nuc')], { keepdim: a.keepdim! })
    }),
    frobenius_norm: fnv('frobenius_norm', (args, kw) => {
      const a = bind(h, 'frobenius_norm', 'input! dim! keepdim', args, kw)
      return wrap(C.vectorNorm(E, T(a.input, 'frobenius_norm'), 2, dimsArg(h, a.dim), bool(a.keepdim, false)))
    }),
    cholesky_solve: fnv('cholesky_solve', (args, kw) => {
      const a = bindPy(h, 'cholesky_solve', 'B L upper=False', args, kw)
      const [B, L0] = [T(a.B, 'cholesky_solve'), tensorArg(h, a.L, 'cholesky_solve', 'L', 2)]
      const U = bool(a.upper, false) ? C.transposeT(E, L0, -2, -1) : L0
      return wrap(singularToSolver(() => C.solveT(E, C.matmul(E, U, C.transposeT(E, U, -2, -1)), B)))
    }),
    cholesky_inverse: fnv('cholesky_inverse', (args, kw) => {
      const a = bindPy(h, 'cholesky_inverse', 'L upper=False', args, kw)
      const L0 = T(a.L, 'cholesky_inverse')
      const U = bool(a.upper, false) ? C.transposeT(E, L0, -2, -1) : L0
      return wrap(renameErr(() => C.invT(E, C.matmul(E, U, C.transposeT(E, U, -2, -1)))))
    }),
    triangular_solve: fnv('triangular_solve', (args, kw) => {
      const a = bindPy(h, 'triangular_solve', 'b A upper=True transpose=False unitriangular=False', args, kw)
      const [b, A] = [T(a.b, 'triangular_solve'), tensorArg(h, a.A, 'triangular_solve', 'A', 2)]
      const sol = triangularSolve(A, b, bool(a.upper, true), bool(a.transpose, false), bool(a.unitriangular, false))
      return py.obj(new ReturnTypes('triangular_solve', [['solution', wrap(sol)], ['cloned_coefficient', wrap(C.cloneT(E, A))]]))
    }),
    lu_solve: fnv('lu_solve', (args, kw) => {
      const a = bindPy(h, 'lu_solve', 'b LU_data LU_pivots', args, kw)
      return luSolve(T(a.b, 'lu_solve'), T(a.LU_data, 'lu_solve'), T(a.LU_pivots, 'lu_solve'))
    }),
    kaiser_window: fnv('kaiser_window', (args, kw) => {
      const a = bindPy(h, 'kaiser_window', 'window_length periodic=True beta=12.0 * dtype=None layout=None device=None pin_memory=False requires_grad=False', args, kw)
      const beta = h.num(a.beta!)
      return makeWindow(a, (n, R) => Array.from({ length: n }, (_, i) => R(besselI(0, beta * Math.sqrt(1 - ((2 * i) / (n - 1) - 1) ** 2), false) / besselI(0, beta, false))))
    }),
    adaptive_max_pool1d: fnv('adaptive_max_pool1d', (args, kw) => {
      const a = bindPy(h, 'adaptive_max_pool1d', 'input output_size', args, kw)
      return adaptive(T(a.input, 'adaptive_max_pool1d'), a.output_size, 'max', true, true)
    }),
    kl_div: fnv('kl_div', (args, kw) => {
      const a = bindPy(h, 'kl_div', 'input target size_average=None reduce=None * log_target=False', args, kw)
      return wrap(klDiv(T(a.input, 'kl_div'), tensorArg(h, a.target, 'kl_div', 'target', 2), 'none', bool(a.log_target, false)))
    }),
    embedding: fnv('embedding', (args, kw) => {
      const a = bindPy(h, 'embedding', 'weight indices padding_idx=-1 scale_grad_by_freq=False sparse=False', args, kw)
      return h.call(Fattrs.embedding, [a.indices!, a.weight!], { scale_grad_by_freq: a.scale_grad_by_freq! })
    }),
    ...Object.fromEntries(['conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d', 'avg_pool1d', 'max_pool1d', 'max_pool2d', 'max_pool3d', 'max_pool1d_with_indices', 'adaptive_avg_pool1d', 'bilinear', 'pixel_shuffle', 'pixel_unshuffle', 'channel_shuffle', 'dropout', 'alpha_dropout', 'feature_alpha_dropout', 'binary_cross_entropy_with_logits', 'cosine_embedding_loss', 'hinge_embedding_loss', 'margin_ranking_loss', 'poisson_nll_loss', 'triplet_margin_loss', 'layer_norm', 'group_norm', 'instance_norm', 'batch_norm', 'rms_norm', 'embedding_bag', 'relu_', 'selu_', 'celu_', 'rrelu_', 'threshold_'].map((n) => [n, Fattrs[n] ?? fnv(n, () => { throw h.err('NotImplementedError', `torch.${n} is not available in this sandbox`) })])),
    ...Object.fromEntries([...INPLACE].filter((n) => !['eq', 'ne', 'ge', 'gt', 'le', 'lt'].includes(n)).map((n) => [`${n}_`, viaMethod(`${n}_`)])),
    zero_: viaMethod('zero_'), fill_: viaMethod('fill_'), detach_: viaMethod('detach_'),
    is_complex: fnv('is_complex', (args) => py.bool(args[0]?.k === 'obj' && args[0].o instanceof ComplexTensor)),
    polygamma: (special as { attrs: Record<string, Value> }).attrs.polygamma,
    mvlgamma: (special as { attrs: Record<string, Value> }).attrs.multigammaln,
    set_printoptions: fnv('set_printoptions', (args, kw) => {
      const a = bindPy(h, 'set_printoptions', 'precision=None threshold=None edgeitems=None linewidth=None profile=None sci_mode=None', args, kw)
      const p = isNone(a.profile) ? null : h.str(a.profile!)
      if (p === 'default' || p === 'short' || p === 'full') C.setPrintProfile(p)
      if (!isNone(a.precision)) C.PRINT_OPTS.precision = h.num(a.precision!)
      if (!isNone(a.threshold)) C.PRINT_OPTS.threshold = h.num(a.threshold!)
      if (!isNone(a.edgeitems)) C.PRINT_OPTS.edgeitems = h.num(a.edgeitems!)
      if (!isNone(a.linewidth)) C.PRINT_OPTS.linewidth = h.num(a.linewidth!)
      C.PRINT_OPTS.sciMode = isNone(a.sci_mode) ? null : h.truthy(a.sci_mode!)
      return py.NONE
    }),
    get_default_dtype: fnv('get_default_dtype', () => py.obj(new DtypeObj(C.DEFAULTS.float))),
    set_default_dtype: fnv('set_default_dtype', (args, kw) => {
      const d = dtypeArg(h, bindPy(h, 'set_default_dtype', 'd', args, kw).d)
      if (!d || !isFloatT(d)) throw h.err('TypeError', 'only floating-point types are supported as the default type')
      C.DEFAULTS.float = d
      return py.NONE
    }),
    finfo: { k: 'type', name: 'finfo', call: (args) => {
      const d = args.length ? dtypeArg(h, args[0]) : C.DEFAULTS.float
      if (!d || !isFloatT(d)) throw h.err('TypeError', "torch.finfo() requires a floating point input type. Use torch.iinfo to handle 'torch.finfo'")
      return py.obj(new FloatInfo(d))
    } },
    iinfo: { k: 'type', name: 'iinfo', call: (args) => {
      const d = dtypeArg(h, args[0])
      if (!d || isFloatT(d)) throw h.err('TypeError', "torch.iinfo() requires an integer input type. Use torch.finfo to handle 'torch.iinfo'")
      return py.obj(new IntInfo(d))
    } },
    promote_types: fnv('promote_types', (args, kw) => {
      const a = bindPy(h, 'promote_types', 'type1 type2', args, kw)
      return py.obj(new DtypeObj(C.promote(dtypeArg(h, a.type1)!, dtypeArg(h, a.type2)!)))
    }),
    can_cast: fnv('can_cast', (args, kw) => {
      const a = bindPy(h, 'can_cast', 'from_ to', args, kw)
      const [f, t] = [dtypeArg(h, a.from_)!, dtypeArg(h, a.to)!]
      return py.bool(!(isFloatT(f) && !isFloatT(t)) && !(f !== 'bool' && t === 'bool'))
    }),
    result_type: fnv('result_type', (args, kw) => {
      const a = bindPy(h, 'result_type', 'tensor1 tensor2', args, kw)
      const [x, y] = [operand(h, a.tensor1!), operand(h, a.tensor2!)]
      if (!x || !y) throw invalidArgs(h, 'result_type', args, kw)
      return py.obj(new DtypeObj(C.promote(x.t.dtype, y.t.dtype, x.weak, y.weak)))
    }),
    initial_seed: fnv('initial_seed', () => py.int(S.seed)),
    seed: fnv('seed', () => {
      S.seed = Math.floor(h.rng.next() * 2 ** 31)
      h.rng.seed(S.seed)
      return py.int(S.seed)
    }),
    get_rng_state: fnv('get_rng_state', () => {
      const t = TT.of([...new Array<number>(4)].map((_, i) => ((h.rng as unknown as { s: number }).s >>> (8 * i)) & 255), [4], 'uint8')
      return wrap(t)
    }),
    set_rng_state: fnv('set_rng_state', (args) => {
      const v = T(args[0], 'set_rng_state').values()
      h.rng.seed(v.reduce((p, b, i) => p + b * 2 ** (8 * i), 0))
      return py.NONE
    }),
    hann_window: windowFn('hann_window', (n, R) => hamming(n, 0.5, 0.5, R)),
    hamming_window: fnv('hamming_window', (args, kw) => {
      const a = bindPy(h, 'hamming_window', 'window_length periodic=True alpha=0.54 beta=0.46 * dtype=None layout=None device=None pin_memory=False requires_grad=False', args, kw)
      return makeWindow(a, (n, R) => hamming(n, h.num(a.alpha!), h.num(a.beta!), R))
    }),
    blackman_window: windowFn('blackman_window', (n, R) => Array.from({ length: n }, (_, i) => {
      const w = R(i * (Math.PI / (n - 1)))
      return R(R(R(R(Math.cos(R(w * 4))) * 0.08) - R(R(Math.cos(R(w * 2))) * 0.5)) + 0.42)
    })),
    bartlett_window: windowFn('bartlett_window', (n, R) => {
      const half = ((n - 1) >> 1) + 1
      return Array.from({ length: n }, (_, i) => {
        const w = R(i * (2 / (n - 1)))
        return i >= half ? R(2 - w) : w
      })
    }),
    tril_indices: triIndices('tril_indices', (r, c, k) => c - r <= k),
    triu_indices: triIndices('triu_indices', (r, c, k) => c - r >= k),
    combinations: fnv('combinations', (args, kw) => {
      const a = bindPy(h, 'combinations', 'input r=2 with_replacement=False', args, kw)
      const t = T(a.input, 'combinations')
      if (t.ndim !== 1) throw h.err('RuntimeError', `Expect a 1D vector, but got shape [${t.shape.join(', ')}]`)
      const r = h.toInt(a.r!)
      if (r < 0) throw h.err('RuntimeError', `Expect a non-negative number, but got ${r}`)
      const repeat = bool(a.with_replacement, false)
      const picks: number[][] = []
      const walk = (start: number, cur: number[]) => {
        if (cur.length === r) {
          picks.push(cur)
          return
        }
        for (let i = start; i < t.size; i++) walk(repeat ? i : i + 1, [...cur, i])
      }
      if (r > 0) walk(0, [])
      return wrap(C.takeT(E, t, picks.flat(), [picks.length, r], 'IndexBackward0'))
    }),
    cartesian_prod: fnv('cartesian_prod', (args) => {
      const ts = args.map((v) => T(v, 'cartesian_prod'))
      ts.forEach((t) => {
        if (t.ndim !== 1) throw h.err('RuntimeError', 'Expect a 1D vector, but got shape [' + t.shape.join(', ') + ']')
      })
      if (ts.length === 1) return args[0]
      if (ts.some((t) => t.dtype !== ts[0].dtype)) throw h.err('RuntimeError', 'meshgrid expects all tensors to have the same dtype')
      const n = ts.reduce((p, t) => p * t.size, 1)
      const dtype = ts[0].dtype
      const cols = ts.map((t, k) => {
        const inner = ts.slice(k + 1).reduce((p, x) => p * x.size, 1)
        return C.toDtype(E, C.takeT(E, t, Array.from({ length: n }, (_, f) => Math.floor(f / inner) % t.size), [n], 'IndexBackward0'), dtype)
      })
      return wrap(C.catT(E, cols, 1, true))
    }),
    block_diag: fnv('block_diag', (args) => {
      const ts = args.map((v, i) => {
        const t = T(v, 'block_diag')
        if (t.ndim > 2) throw h.err('RuntimeError', `torch.block_diag: Input tensors must have 2 or fewer dimensions. Input ${i} has ${t.ndim} dimensions`)
        return t.ndim === 2 ? t : C.reshapeT(E, t, t.ndim ? [1, t.size] : [1, 1], false)
      })
      if (!ts.length) return wrap(TT.of([], [1, 0], C.DEFAULTS.float))
      const dtype = ts.map((t) => t.dtype).reduce((p, q) => C.promote(p, q))
      const [rows, cols] = [ts.reduce((p, t) => p + t.shape[0], 0), ts.reduce((p, t) => p + t.shape[1], 0)]
      // a copy from all the blocks laid end to end; cells off the blocks stay 0
      const flat = C.catT(E, ts.map((t) => C.toDtype(E, C.reshapeT(E, t, [-1], false), dtype)), 0)
      const src = new Array<number>(rows * cols).fill(-1)
      let [r0, c0, at] = [0, 0, 0]
      for (const t of ts) {
        for (let i = 0; i < t.shape[0]; i++) for (let j = 0; j < t.shape[1]; j++) src[(r0 + i) * cols + c0 + j] = at++
        r0 += t.shape[0]
        c0 += t.shape[1]
      }
      return wrap(C.takeT(E, flat, src, [rows, cols], 'BlockDiagBackward0'))
    }),
    vander: fnv('vander', (args, kw) => {
      const a = bindPy(h, 'vander', 'x N=None increasing=False', args, kw)
      const t = T(a.x, 'vander')
      if (t.ndim !== 1) throw h.err('RuntimeError', 'x must be a one-dimensional tensor.')
      const N = isNone(a.N) ? t.size : h.toInt(a.N!)
      if (N < 0) throw h.err('RuntimeError', 'N must be non-negative.')
      const inc = bool(a.increasing, false)
      const dtype = t.dtype === 'bool' || !isFloatT(t.dtype) ? 'int64' : t.dtype
      const x = C.toDtype(E, t, dtype)
      const powers = Array.from({ length: N }, (_, j) => C.binary(E, '**', x, TT.scalar(inc ? j : N - 1 - j, 'int64'), false, true))
      return wrap(named(powers.length ? C.catT(E, powers, 1, true) : TT.of([], [t.size, 0], dtype), inc ? 'CopySlices' : 'FlipBackward0'))
    }),
    chain_matmul: fnv('chain_matmul', (args) => wrap(named(args.map((v) => T(v, 'chain_matmul')).reduce((p, q) => C.matmul(E, p, q)), 'ViewBackward0'))),
    std_mean: stdMean('std'), var_mean: stdMean('var'),
    atleast_1d: atLeastFn('atleast_1d', 1), atleast_2d: atLeastFn('atleast_2d', 2), atleast_3d: atLeastFn('atleast_3d', 3),
    dstack: joined('dstack', (ts) => [ts.map(atLeast3), 2]),
    tensordot: fnv('tensordot', (args, kw) => {
      const a = bindPy(h, 'tensordot', 'a b dims=2 out=None', args, kw)
      const [x, y] = [T(a.a, 'tensordot'), tensorArg(h, a.b, 'tensordot', 'b', 2)]
      let dx: number[]
      let dy: number[]
      if (a.dims!.k === 'int') {
        const n = a.dims!.v
        if (n < 0) throw h.err('RuntimeError', `tensordot expects dims >= 0, but got dims=${n}`)
        dx = Array.from({ length: n }, (_, i) => x.ndim - n + i)
        dy = Array.from({ length: n }, (_, i) => i)
      } else {
        const [p, q] = h.iterate(a.dims!)
        const list = (v: Value) => (v.k === 'int' ? [v.v] : h.iterate(v).map((u) => h.toInt(u)))
        dx = list(p).map((d) => C.normDim(d, x.ndim))
        dy = list(q).map((d) => C.normDim(d, y.ndim))
      }
      if (dx.length !== dy.length) throw h.err('RuntimeError', `the number of dimensions to contract must match, but got ${dx.length} and ${dy.length}`)
      dx.forEach((d, i) => {
        if (x.shape[d] !== y.shape[dy[i]]) throw h.err('RuntimeError', `contracted dimensions need to match, but first has size ${x.shape[d]} in dim ${d} and second has size ${y.shape[dy[i]]} in dim ${dy[i]}`)
      })
      const [fx, fy] = [x.shape.map((_, k) => k).filter((k) => !dx.includes(k)), y.shape.map((_, k) => k).filter((k) => !dy.includes(k))]
      const n = dx.reduce((p, d) => p * x.shape[d], 1)
      const left = C.reshapeT(E, C.permuteT(E, x, [...fx, ...dx]), [-1, n], false)
      const right = C.reshapeT(E, C.permuteT(E, y, [...dy, ...fy]), [n, -1], false)
      return wrap(named(C.reshapeT(E, C.matmul(E, left, right), [...fx.map((k) => x.shape[k]), ...fy.map((k) => y.shape[k])], false), 'ViewBackward0'))
    }),
    gradient: fnv('gradient', (args, kw) => {
      const a = bindPy(h, 'gradient', 'input * spacing=1 dim=None edge_order=1', args, kw)
      const t = T(a.input, 'gradient')
      const order = h.toInt(a.edge_order!)
      if (order !== 1 && order !== 2) throw h.err('RuntimeError', 'torch.gradient only supports edge_order=1 and edge_order=2.')
      const dims = (dimsArg(h, a.dim) ?? t.shape.map((_, k) => k)).map((d) => C.normDim(d, t.ndim))
      const spacing = a.spacing!
      const listOf = spacing.k === 'list' || spacing.k === 'tuple' ? h.iterate(spacing) : null
      return py.tuple(dims.map((d, q) => {
        const n = t.shape[d]
        if (n < order + 1) throw h.err('RuntimeError', `torch.gradient expected each dimension size to be at least edge_order+1`)
        const sp = listOf ? listOf[listOf.length === 1 ? 0 : q] : spacing
        const coords = asTT(sp) ? asTT(sp)!.values() : null
        const step = coords ? 0 : h.num(sp)
        const at = (i: number) => (coords ? coords[i] : i * step)
        const idx: number[][] = []
        const wt: number[][] = []
        for (let i = 0; i < n; i++) {
          if (i === 0 || i === n - 1) {
            const s0 = i === 0 ? 1 : -1
            if (order === 1) {
              const d1 = at(i + s0) - at(i)
              idx.push([i, i + s0, i])
              wt.push([-1 / d1, 1 / d1, 0])
            } else {
              const [d1, d2] = [at(i + s0) - at(i), at(i + 2 * s0) - at(i + s0)]
              idx.push([i, i + s0, i + 2 * s0])
              wt.push([-(2 * d1 + d2) / (d1 * (d1 + d2)), (d1 + d2) / (d1 * d2), -d1 / (d2 * (d1 + d2))])
            }
          } else {
            const [hs, hd] = [at(i) - at(i - 1), at(i + 1) - at(i)]
            idx.push([i - 1, i, i + 1])
            wt.push([-hd / (hs * (hs + hd)), (hd - hs) / (hs * hd), hs / (hd * (hs + hd))])
          }
        }
        return wrap(named(C.weightedAxis(E, isFloatT(t.dtype) ? t : C.toDtype(E, t, C.DEFAULTS.float), d, idx, wt), 'CatBackward0'))
      }))
    }),
    bucketize: fnv('bucketize', (args, kw) => {
      const a = bindPy(h, 'bucketize', 'input boundaries * out_int32=False right=False', args, kw)
      const b = tensorArg(h, a.boundaries, 'bucketize', 'boundaries', 2)
      if (b.ndim !== 1) throw h.err('RuntimeError', `boundaries tensor must be 1 dimension, but got dim(${b.ndim})`)
      return searchSorted(b, a.input!, bool(a.right, false), bool(a.out_int32, false))
    }),
    searchsorted: fnv('searchsorted', (args, kw) => {
      const a = bindPy(h, 'searchsorted', 'sorted_sequence input * out_int32=False right=False side=None sorter=None', args, kw)
      const seq = tensorArg(h, a.sorted_sequence, 'searchsorted', 'sorted_sequence')
      if (!isNone(a.side) && h.str(a.side!) === 'left' && bool(a.right, false)) throw h.err('RuntimeError', "torch.searchsorted(): side and right can't be set to opposites, got side of left while right was True")
      const right = bool(a.right, false) || (!isNone(a.side) && h.str(a.side!) === 'right')
      const sorted0 = isNone(a.sorter) ? seq : C.gatherT({ grad: false } as Engine, seq, -1, T(a.sorter, 'searchsorted')).out
      return searchSorted(sorted0, a.input!, right, bool(a.out_int32, false))
    }),
    cdist: fnv('cdist', (args, kw) => {
      const a = bindPy(h, 'cdist', "x1 x2 p=2.0 compute_mode='use_mm_for_euclid_dist_if_necessary'", args, kw)
      const [x, y] = [T(a.x1, 'cdist', ), tensorArg(h, a.x2, 'cdist', 'x2', 2)]
      if (x.ndim < 2) throw h.err('RuntimeError', `cdist only supports at least 2D tensors, X1 got: ${x.ndim}D`)
      if (y.ndim < 2) throw h.err('RuntimeError', `cdist only supports at least 2D tensors, X2 got: ${y.ndim}D`)
      if (x.shape[x.ndim - 1] !== y.shape[y.ndim - 1]) throw h.err('RuntimeError', `X1 and X2 must have the same number of columns. X1: ${x.shape[x.ndim - 1]} X2: ${y.shape[y.ndim - 1]}`)
      const p = h.num(a.p!)
      if (p < 0) throw h.err('RuntimeError', `cdist only supports non-negative p values`)
      return wrap(named(C.vectorNorm(E, C.binary(E, '-', C.unsqueezeT(E, x, -2), C.unsqueezeT(E, y, -3)), p, [-1], false), 'CdistBackward0'))
    }),
    cosine_similarity: Fattrs.cosine_similarity, pairwise_distance: Fattrs.pairwise_distance, pdist: Fattrs.pdist,
    selu: Fattrs.selu, celu: Fattrs.celu, prelu: Fattrs.prelu, rrelu: Fattrs.rrelu, threshold: Fattrs.threshold, hardshrink: Fattrs.hardshrink,
    svd: fnv('svd', (args, kw) => {
      const a = bindPy(h, 'svd', 'input some=True compute_uv=True', args, kw)
      const x = T(a.input, 'svd')
      if (!bool(a.compute_uv, true)) {
        const zero = (r: number) => wrap(TT.of(new Array<number>(prod([...x.shape.slice(0, -2), r, r])).fill(0), [...x.shape.slice(0, -2), r, r], x.dtype))
        return py.obj(new ReturnTypes('svd', [['U', zero(x.shape[x.ndim - 2])], ['S', wrap(singularValues(x))], ['V', zero(x.shape[x.ndim - 1])]]))
      }
      const [u, s0, vh] = svdOf(x, !bool(a.some, true))
      return py.obj(new ReturnTypes('svd', [['U', wrap(u)], ['S', wrap(s0)], ['V', wrap(C.transposeT(E, vh, -2, -1))]]))
    }),
    ...Object.fromEntries(Object.entries(REMOVED).map(([n, msg]) => [n, fnv(n, () => { throw h.err('RuntimeError', msg) })])),
    slogdet: fnv('slogdet', (args, kw) => {
      const r = slogdetOf(T(bindPy(h, 'slogdet', 'input', args, kw).input, 'slogdet'))
      return py.obj(new ReturnTypes('slogdet', [['sign', wrap(r.sign)], ['logabsdet', wrap(r.log)]]))
    }),
    logdet: fnv('logdet', (args, kw) => {
      const r = slogdetOf(T(bindPy(h, 'logdet', 'input', args, kw).input, 'logdet'))
      return wrap(C.map2T(E, r.log, r.sign, (l, sg) => (sg <= 0 ? (sg === 0 ? -Infinity : NaN) : l), () => 1, () => 0, r.log.dtype, 'WhereBackward0'))
    }),
    pinverse: fnv('pinverse', (args, kw) => {
      const a = bindPy(h, 'pinverse', 'input rcond=1e-15', args, kw)
      return wrap(pinvOf(T(a.input, 'pinverse'), h.num(a.rcond!)))
    }),
    lu: fnv('lu', (args, kw) => {
      const a = bindPy(h, 'lu', 'A pivot=True get_infos=False out=None', args, kw)
      const [f, p] = luParts(T(a.A, 'lu'))
      return bool(a.get_infos, false) ? py.tuple([wrap(f), wrap(p), wrap(TT.of([0], [], 'int32'))]) : py.tuple([wrap(f), wrap(p)])
    }),
    matrix_exp: linalgFns.matrix_exp,
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
        const isInt = arr.dtype !== 'float64' && arr.dtype !== 'float32'
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
    torch: torchMod, 'torch.nn': nn, 'torch.nn.functional': nnFunctional, 'torch.nn.init': nnInit, 'torch.nn.utils': nnUtils, 'torch.nn.utils.rnn': rnnUtils, 'torch.optim': optimMod, 'torch.optim.lr_scheduler': lrScheduler, 'torch.utils': torch.utils, 'torch.utils.data': data, 'torch.utils.data.dataloader': dataDataloader, 'torch.utils.data.dataset': dataDataset, 'torch.utils.data.sampler': dataSampler, 'torch.cuda': cuda, 'torch.linalg': linalg, 'torch.special': special, 'torch.autograd': autograd, 'torch.autograd.functional': autogradFunctional,
    torchvision: { k: 'module', name: 'torchvision', attrs: { transforms, datasets } }, 'torchvision.transforms': transforms, 'torchvision.datasets': datasets,
  }
}

export const TORCH: PyLib = {
  modules: ['torch', 'torch.nn', 'torch.nn.functional', 'torch.nn.init', 'torch.nn.utils', 'torch.nn.utils.rnn', 'torch.optim', 'torch.optim.lr_scheduler', 'torch.utils', 'torch.utils.data', 'torch.utils.data.dataloader', 'torch.utils.data.dataset', 'torch.utils.data.sampler', 'torch.cuda', 'torch.linalg', 'torch.special', 'torch.autograd', 'torch.autograd.functional', 'torchvision', 'torchvision.transforms', 'torchvision.datasets'],
  load: build,
}
