/**
 * keras and tensorflow for the sandbox: Sequential, Functional and subclassed models, the layers of an intro course,
 * model.summary() (Keras 3's table, plus the shape-flow view), compile / fit / evaluate / predict with real training on
 * the tensorCore engine, optimisers, losses, metrics, initializers, regularizers and callbacks; and a tensorflow
 * (tf.constant, tf.Variable, GradientTape, tf.math, tf.nn, tf.image, tf.data …) with TensorFlow's strict dtypes and printing.
 * Keras is channels-last: images are (batch, height, width, channels).
 */
import { NDArray, PyError, arrayRepr, arrayStr, cStrides, prod } from './ndarray'
import type { DType } from './ndarray'
import { PyDict, PyObj, formatSpec, py, repr, toArray } from './minipy'
import type { ApiKind, ClassValue, FuncValue, Host, HostClass, InstValue, Kw, PyLib, Value } from './minipy'
import { SandboxRandom } from './ndops'
import type { FlowRow } from './pyEvents'
import * as C from './tensorCore'
import { Engine, TT, isFloatT } from './tensorCore'
import type { GradNode, TDType } from './tensorCore'

// ---------------------------------------------------------------- per-run state

interface SavedOptimizer { t: number; lr: number | LRSchedule; uProduct: number; built: boolean; slots: Record<string, number[]>[] }
type SavedFile = { kind: 'weights'; values: number[][] } | { kind: 'model'; model: KModel; values: number[][]; opt: SavedOptimizer | null }

interface KerasState {
  E: Engine
  counters: Map<string, number>
  files: Map<string, SavedFile>
  notes: Set<string>
  /** the training flag of the enclosing model or layer call (null: called from the top level) */
  training: boolean | null
  /** add_loss and activity regularizers, collected during a forward pass */
  losses: TT[]
  /** rebuilds a layer from its class name and constructor arguments (clone_model, Bidirectional) */
  make: (kind: string, kw: Kw) => KLayer
  /** the tf function being run has recorded its own step */
  traced: boolean
  errors: Record<string, ClassValue> | null
}

const stateOf = (h: Host): KerasState => {
  let s = h.state.get('keras') as KerasState | undefined
  if (!s) {
    s = { E: new Engine(h.rng), counters: new Map(), files: new Map(), notes: new Set(), training: null, losses: [], make: () => { throw new PyError('RuntimeError', 'keras is not loaded') }, traced: false, errors: null }
    h.state.set('keras', s)
  }
  return s
}

/** Keras's automatic names: dense, dense_1, dense_2 … (one counter per run) */
const autoName = (h: Host, base: string): string => {
  const c = stateOf(h).counters
  const n = c.get(base) ?? 0
  c.set(base, n + 1)
  return n ? `${base}_${n}` : base
}

/** tf.errors: real exception classes, so that except tf.errors.InvalidArgumentError catches them */
const tfErrorClasses = (h: Host): Record<string, ClassValue> => {
  const S = stateOf(h)
  if (!S.errors) {
    const base = h.excClass('Exception')
    const make = (name: string, parent: ClassValue): ClassValue => ({ k: 'class', name, bases: [parent], ns: new Map(), host: parent.host, builtin: true })
    const op = make('OpError', base)
    S.errors = { OpError: op, ...Object.fromEntries(['InvalidArgumentError', 'NotFoundError', 'OutOfRangeError', 'ResourceExhaustedError', 'FailedPreconditionError', 'UnimplementedError', 'UnknownError'].map((n) => [n, make(n, op)])) }
  }
  return S.errors
}

const tfErr = (h: Host, message: string): PyError => {
  const e = h.err('InvalidArgumentError', message)
  const inst = h.call(tfErrorClasses(h).InvalidArgumentError, [py.str(message)])
  if (inst.k === 'inst') inst.attrs.set('message', py.str(message))
  e.exc = inst
  return e
}

const noteOnce = (h: Host, key: string, tone: 'info' | 'warn', params: Record<string, string | number> = {}) => {
  const s = stateOf(h)
  if (s.notes.has(key) || !h.tracing) return
  s.notes.add(key)
  h.emit({ type: 'note', tone, key, params })
}

const fnv = (name: string, call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })
/** keras.src.utils.naming.to_snake_case */
const snake = (s: string) => s.replace(/(.)([A-Z][a-z]+)/g, '$1_$2').replace(/([a-z])([A-Z])/g, '$1_$2').toLowerCase()
const isSeq = (v: Value | undefined): v is Extract<Value, { k: 'list' | 'tuple' }> => !!v && (v.k === 'list' || v.k === 'tuple')
const isNone = (v: Value | undefined) => !v || v.k === 'none'

type KShape = (number | null)[]
const shapeStr = (s: KShape) => `(${s.map((d) => (d === null ? 'None' : d)).join(', ')}${s.length === 1 ? ',' : ''})`
const shapeTuple = (s: KShape): Value => py.tuple(s.map((d) => (d === null ? py.NONE : py.int(d))))

// ---------------------------------------------------------------- constructor signatures, as in the Keras docs

type Params = Record<string, Value>

interface SigEntry { name: string; def: Value | undefined; unsupported: boolean }

const parseDefault = (s: string): Value => {
  if (s === 'None') return py.NONE
  if (s === 'True') return py.bool(true)
  if (s === 'False') return py.bool(false)
  if (/^-?\d+$/.test(s)) return py.int(Number(s))
  if (/^-?[\d.]+(e-?\d+)?$/.test(s)) return py.float(Number(s))
  if (s.startsWith('(')) return py.tuple(s.slice(1, -1).split(',').map(parseDefault))
  return py.str(s)
}

const SIG_CACHE = new Map<string, SigEntry[]>()
const parseSig = (sig: string): SigEntry[] => {
  let out = SIG_CACHE.get(sig)
  if (!out) {
    out = sig.split(' ').filter(Boolean).map((tok) => {
      const unsupported = tok.startsWith('!')
      const [name, d] = (unsupported ? tok.slice(1) : tok).split('=')
      return { name, def: d === undefined ? undefined : parseDefault(d), unsupported }
    })
    SIG_CACHE.set(sig, out)
  }
  return out
}

/** keywords every layer takes */
const LAYER_COMMON = new Set(['name', 'dtype', 'trainable', 'input_shape', 'input_dim', 'batch_input_shape', 'batch_size', 'input_length'])

/**
 * Bind a call to a signature string ("units activation=None !kernel_constraint=None"). A name marked ! changes results
 * and is not implemented: it raises NotImplementedError unless left at its default. Keywords the real API does not have are
 * refused, as Keras does.
 */
const bindArgs = (h: Host, cls: string, sig: string, args: Value[], kw: Kw, common: Set<string> | null = LAYER_COMMON): Params => {
  const entries = parseSig(sig)
  if (args.length > entries.length) throw h.err('TypeError', `${cls}() takes ${entries.length} positional arguments but ${args.length} were given`)
  const P: Params = {}
  entries.forEach((e, i) => {
    const given = i < args.length ? args[i] : kw[e.name]
    if (i < args.length && e.name in kw) throw h.err('TypeError', `${cls}() got multiple values for argument '${e.name}'`)
    if (given !== undefined) {
      const isDefault = e.def !== undefined && (repr(given) === repr(e.def) || (e.name === 'data_format' && repr(given) === "'channels_last'") || (e.def.k === 'tuple' && e.def.items.every((d) => repr(d) === repr(given))))
      if (e.unsupported && !isDefault) throw h.err('NotImplementedError', `${cls}: ${e.name}=${repr(given)} is not implemented in this sandbox`)
      P[e.name] = given
    } else if (e.def !== undefined) P[e.name] = e.def
    else throw h.err('TypeError', `${cls}() missing 1 required positional argument: '${e.name}'`)
  })
  for (const k of Object.keys(kw)) {
    if (entries.some((e) => e.name === k) || common?.has(k)) continue
    if (common) throw h.err('ValueError', `Unrecognized keyword arguments passed to ${cls}: {'${k}': ${repr(kw[k])}}`)
    throw h.err('TypeError', `${cls}() got an unexpected keyword argument '${k}'`)
  }
  return P
}

// ---------------------------------------------------------------- printing (NumPy style, as TensorFlow prints)

/** the shortest decimal that is the same float32, so 0.1 prints as 0.1 and not 0.10000000149 */
const f32clean = (v: number): number => {
  if (!Number.isFinite(v) || v === 0) return v
  for (let p = 1; p <= 9; p++) {
    const x = Number(v.toPrecision(p))
    if (Math.fround(x) === v) return x
  }
  return v
}

const ndFor = (t: TT): NDArray => NDArray.create(t.values(), t.shape, t.dtype === 'int32' ? 'int64' : t.dtype)

/** array([...], dtype=float32); arrays have no int32 of their own, so its suffix is added here */
const npRepr = (t: TT): string => {
  const base = arrayRepr(ndFor(t))
  return t.dtype === 'int32' && t.ndim ? `${base.slice(0, -1)}, dtype=int32)` : base
}

const scalarStr = (t: TT) => {
  const v = t.values()[0]
  if (t.dtype === 'bool') return v ? 'True' : 'False'
  if (!isFloatT(t.dtype)) return String(v)
  if (!Number.isFinite(v)) return Number.isNaN(v) ? 'nan' : v > 0 ? 'inf' : '-inf'
  const c = t.dtype === 'float32' ? f32clean(v) : v
  return Number.isInteger(c) ? `${c}.0` : String(c)
}

// ---------------------------------------------------------------- tensorflow tensors

const TF_DTYPES: Record<string, TDType> = { float32: 'float32', float64: 'float64', double: 'float64', int32: 'int32', int64: 'int64', bool: 'bool', uint8: 'uint8' }

class TFDtype extends PyObj {
  readonly cls = 'DType'
  d: TDType | 'string'
  constructor(d: TDType | 'string') {
    super()
    this.d = d
  }
  repr() { return `tf.${this.d}` }
  binop(op: string, other: Value) {
    if (op !== '==' && op !== '!=') return undefined
    const same = other.k === 'obj' && other.o instanceof TFDtype ? other.o.d === this.d : other.k === 'str' && other.v === this.d
    return py.bool((op === '==') === same)
  }
  getAttr(name: string) {
    if (name === 'name') return py.str(this.d)
    if (name === 'is_floating') return py.bool(this.d !== 'string' && isFloatT(this.d))
    if (name === 'is_integer') return py.bool(this.d === 'int32' || this.d === 'int64' || this.d === 'uint8')
    return undefined
  }
}

class TensorShape extends PyObj {
  readonly cls = 'TensorShape'
  dims: KShape
  constructor(dims: KShape) {
    super()
    this.dims = dims
  }
  repr() { return `TensorShape([${this.dims.map((d) => (d === null ? 'None' : d)).join(', ')}])` }
  str() { return shapeStr(this.dims) }
  len() { return this.dims.length }
  iter() { return this.dims.map((d) => (d === null ? py.NONE : py.int(d))) }
  getItem(idx: Value, h: Host) {
    if (idx.k === 'slice') {
      const part = (x: Value) => (x.k === 'none' ? undefined : h.toInt(x))
      return py.obj(new TensorShape(this.dims.slice(part(idx.start), part(idx.stop))))
    }
    const i = h.toInt(idx)
    const n = this.dims.length
    const d = this.dims[i < 0 ? i + n : i]
    if (d === undefined) throw h.err('IndexError', 'list index out of range')
    return d === null ? py.NONE : py.int(d)
  }
  getAttr(name: string) {
    if (name === 'as_list') return fnv(name, () => py.list(this.iter()))
    if (name === 'rank' || name === 'ndims') return py.int(this.dims.length)
    if (name === 'num_elements') return fnv(name, () => (this.dims.some((d) => d === null) ? py.NONE : py.int(prod(this.dims as number[]))))
    return undefined
  }
  binop(op: string, other: Value, _r: boolean, h: Host) {
    if (op !== '==' && op !== '!=') return undefined
    const o = other.k === 'obj' && other.o instanceof TensorShape ? other.o.dims : other.k === 'tuple' || other.k === 'list' ? h.iterate(other).map((v) => (v.k === 'none' ? null : h.num(v))) : null
    const same = !!o && o.length === this.dims.length && o.every((d, i) => d === this.dims[i])
    return py.bool((op === '==') === same)
  }
  isa(name: string) { return name === 'TensorShape' }
}

const tfWrap = (t: TT): Value => py.obj(new TFTensor(t))

const traceSnap = (t: TT) => ({ shape: [...t.shape], values: t.dtype === 'float32' ? t.values().map(f32clean) : t.values(), dtype: t.a.dtype })

/** a small TF op for the element-provenance view (the same view as NumPy's) */
const traceTF = (h: Host, api: string, kind: ApiKind, ops: { label: string; t: TT }[], out: TT, axis: number | null = null, groups?: [number, number][][]) => {
  stateOf(h).traced = true
  if (!h.tracing || out.size > 240 || ops.some((o) => o.t.size > 240)) return
  h.traceCall(api, kind, ops.map((o) => ({ label: o.label, snap: traceSnap(o.t) })), traceSnap(out), new TFTensor(out).repr(), axis, groups)
}
const tfOf = (v: Value | undefined): TT | null => (v && v.k === 'obj' && v.o instanceof TFTensor ? v.o.t : null)

const OP_NAME: Record<string, string> = { '+': 'AddV2', '-': 'Sub', '*': 'Mul', '/': 'RealDiv', '@': 'MatMul', '**': 'Pow', '<': 'Less', '>': 'Greater', '<=': 'LessEqual', '>=': 'GreaterEqual', '==': 'Equal', '!=': 'NotEqual', maximum: 'Maximum', minimum: 'Minimum', '//': 'FloorDiv', '%': 'FloorMod' }
/** the tensor dtype NumPy data keeps */
const ND_TF: Record<DType, TDType> = { float64: 'float64', float32: 'float32', int64: 'int64', bool: 'bool', uint8: 'uint8' }
const TF_KIND: Record<TDType, string> = { float32: 'float', float64: 'double', int32: 'int32', int64: 'int64', bool: 'bool', uint8: 'uint8' }

/** a Python value as a TF operand: Python numbers take the tensor's dtype, tensors must match exactly */
const tfOperand = (h: Host, v: Value, like: TT | null): TT | null => {
  const t = tfOf(v)
  if (t) return t
  if (v.k === 'int' || v.k === 'float' || v.k === 'bool') {
    const d: TDType = like ? like.dtype : v.k === 'float' ? 'float32' : v.k === 'bool' ? 'bool' : 'int32'
    if (like && v.k === 'float' && !isFloatT(like.dtype) && !Number.isInteger(v.v)) {
      throw h.err('TypeError', `Cannot convert ${repr(v)} to EagerTensor of dtype ${like.dtype}`)
    }
    return TT.scalar(v.k === 'bool' ? (v.v ? 1 : 0) : v.v, d)
  }
  if (v.k === 'array') return TT.of(v.a.values(), v.a.shape, ND_TF[v.a.dtype])
  if (v.k === 'list' || v.k === 'tuple') return toTF(h, v, like?.dtype ?? null)
  return null
}

/** tf.constant / convert_to_tensor: Python floats → float32, ints → int32, NumPy keeps its dtype */
const toTF = (h: Host, v: Value, dtype: TDType | null): TT => {
  const t = tfOf(v)
  if (t) return dtype && dtype !== t.dtype ? TT.of(t.values(), t.shape, dtype) : t
  if (v.k === 'array') return TT.of(v.a.values(), v.a.shape, dtype ?? ND_TF[v.a.dtype])
  if (isSeq(v) && v.items.some((x) => tfOf(x))) {
    // a list holding tensors stacks them, as tf.constant does
    const parts = v.items.map((x) => toTF(h, x, dtype))
    return C.catT(stateOf(h).E, parts, 0, true)
  }
  let a: NDArray
  try {
    a = toArray(v)
  } catch (e) {
    throw h.err('ValueError', `Can't convert to a Tensor: ${(e as Error).message}`)
  }
  return TT.of(a.values(), a.shape, dtype ?? (a.dtype === 'float64' || a.dtype === 'float32' ? 'float32' : a.dtype === 'bool' ? 'bool' : 'int32'))
}

class TFTensor extends PyObj {
  readonly cls: string = 'EagerTensor'
  t: TT
  constructor(t: TT) {
    super()
    this.t = t
  }
  body() {
    return this.t.ndim ? arrayStr(ndFor(this.t)) : scalarStr(this.t)
  }
  repr() {
    const np = this.t.ndim ? npRepr(this.t) : scalarStr(this.t)
    return `<tf.Tensor: shape=${shapeStr(this.t.shape)}, dtype=${this.t.dtype}, numpy=${np.includes('\n') ? '\n' : ''}${np}>`
  }
  str() {
    const b = this.body()
    return `tf.Tensor(${this.t.ndim > 1 ? '\n' : ''}${b}, shape=${shapeStr(this.t.shape)}, dtype=${this.t.dtype})`
  }
  format(spec: string) {
    if (this.t.size !== 1) return undefined
    const v = this.t.values()[0]
    return formatSpec(isFloatT(this.t.dtype) ? py.float(f32clean(v)) : py.int(v), spec)
  }
  toArray() { return ndFor(this.t) }
  isa(name: string): boolean { return name === 'Tensor' }
  len(h: Host) {
    if (!this.t.ndim) throw h.err('TypeError', 'Scalar tensor has no `len()`')
    return this.t.shape[0]
  }
  iter(h: Host) {
    if (!this.t.ndim) throw h.err('TypeError', 'Cannot iterate over a scalar tensor.')
    return unstackT(stateOf(h).E, this.t, 0).map(tfWrap)
  }
  truthy(h: Host) {
    if (this.t.size !== 1) throw h.err('ValueError', 'The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()')
    return this.t.values()[0] !== 0
  }
  getItem(idx: Value, h: Host) {
    const E = stateOf(h).E
    // x[mask] is tf.boolean_mask
    const maskT = tfOf(idx)
    if (maskT && maskT.dtype === 'bool') return tfWrap(booleanMask(h, E, this.t, maskT, 0))
    const items = (idx.k === 'tuple' ? idx.items : [idx]).map((it) => {
      const scalar = tfOf(it)
      if (scalar && scalar.ndim === 0 && !isFloatT(scalar.dtype) && scalar.dtype !== 'bool') return { kind: 'int' as const, value: scalar.values()[0] }
      if (it.k === 'int') return { kind: 'int' as const, value: it.v }
      if (it.k === 'slice') {
        const part = (x: Value) => (x.k === 'none' ? null : h.toInt(x))
        return { kind: 'slice' as const, start: part(it.start), stop: part(it.stop), step: part(it.step) }
      }
      if (it.k === 'none') return { kind: 'newaxis' as const }
      if (it.k === 'ellipsis') return { kind: 'ellipsis' as const }
      throw h.err('TypeError', 'Only integers, slices (`:`), ellipsis (`...`), tf.newaxis (`None`) and scalar tf.int32/tf.int64 tensors are valid indices')
    })
    return tfWrap(C.indexT(E, this.t, items))
  }
  setItem(_i: Value, _v: Value, h: Host) {
    throw h.err('TypeError', `'tensorflow.python.framework.ops.${this.cls}' object does not support item assignment`)
  }
  binop(op: string, other: Value, reflected: boolean, h: Host): Value | undefined {
    const o = tfOperand(h, other, this.t)
    if (!o) return undefined
    const [a, b] = reflected ? [o, this.t] : [this.t, o]
    if (a.dtype !== b.dtype && OP_NAME[op]) {
      throw tfErr(h, `cannot compute ${OP_NAME[op]} as input #1(zero-based) was expected to be a ${TF_KIND[a.dtype]} tensor but is a ${TF_KIND[b.dtype]} tensor [Op:${OP_NAME[op]}]`)
    }
    const E = stateOf(h).E
    const me = h.nameOf(py.obj(this), 'a')
    const ops = reflected ? [{ label: h.nameOf(other, 'a'), t: a }, { label: me, t: b }] : [{ label: me, t: a }, { label: h.nameOf(other, 'b'), t: b }]
    if (op !== '@' && OP_NAME[op]) bcast(h, a.shape, b.shape, OP_NAME[op])
    let out: TT
    if (op === '@') {
      out = C.matmul(E, a, b, 'MatMul')
      traceTF(h, 'op:@', 'matmul', ops, out)
      return tfWrap(out)
    }
    if (['<', '<=', '>', '>=', '==', '!='].includes(op)) out = C.compare(a, b, op as C.CmpOpT)
    else if ((op === '&' || op === '|' || op === '^') && a.dtype === 'bool') out = logical(h, a, b, op)
    else if (op === '/' && !isFloatT(a.dtype)) out = C.binary(E, '/', C.toDtype(E, a, 'float64'), C.toDtype(E, b, 'float64'))
    else if (['+', '-', '*', '/', '**', '//', '%', 'maximum', 'minimum'].includes(op)) {
      if ((op === '//' || op === '%') && !isFloatT(a.dtype) && b.values().some((v) => v === 0)) throw tfErr(h, `Integer division by zero [Op:${OP_NAME[op]}]`)
      const r = C.binary(E, op as C.BinOpT, a, b)
      out = r.dtype === a.dtype ? r : C.toDtype(E, r, a.dtype)
    } else return undefined
    traceTF(h, op === 'maximum' || op === 'minimum' ? `tf.${op}` : `op:${op}`, 'elementwise', ops, out)
    return tfWrap(out)
  }
  unary(op: '-' | '+' | '~', h: Host): Value {
    if (op === '~' && this.t.dtype === 'bool') return tfWrap(TT.of(this.t.values().map((v) => (v ? 0 : 1)), this.t.shape, 'bool'))
    return op === '-' ? tfWrap(C.unary(stateOf(h).E, 'neg', this.t)) : py.obj(this)
  }
  getAttr(name: string, h: Host): Value | undefined {
    const t = this.t
    switch (name) {
      case 'shape': return py.obj(new TensorShape(t.shape))
      case 'get_shape': return fnv(name, () => py.obj(new TensorShape(t.shape)))
      case 'dtype': return py.obj(new TFDtype(t.dtype))
      case 'ndim': return py.int(t.ndim)
      case 'device': return py.str('/job:localhost/replica:0/task:0/device:CPU:0')
      case 'numpy': return fnv(name, () => (t.ndim ? py.arr(ndFor(t)) : isFloatT(t.dtype) ? py.float(f32clean(t.values()[0])) : t.dtype === 'bool' ? py.bool(t.values()[0] !== 0) : py.int(t.values()[0])))
      case 'T': return tfWrap(C.permuteT(stateOf(h).E, t, t.shape.map((_, i) => t.ndim - 1 - i)))
    }
    return undefined
  }
}

const logical = (h: Host, a: TT, b: TT, op: '&' | '|' | '^'): TT => {
  const shape = bcast(h, a.shape, b.shape)
  const ia = C.bcastIndex(a.shape, shape)
  const ib = C.bcastIndex(b.shape, shape)
  const av = a.values()
  const bv = b.values()
  return TT.of(Array.from({ length: prod(shape) }, (_, i) => {
    const x = av[ia[i]] !== 0
    const y = bv[ib[i]] !== 0
    return (op === '&' ? x && y : op === '|' ? x || y : x !== y) ? 1 : 0
  }), shape, 'bool')
}

const bcast = (h: Host, a: number[], b: number[], op?: string): number[] => {
  const nd = Math.max(a.length, b.length)
  const out = new Array<number>(nd)
  for (let d = 1; d <= nd; d++) {
    const x = a[a.length - d] ?? 1
    const y = b[b.length - d] ?? 1
    if (x !== y && x !== 1 && y !== 1) throw tfErr(h, `Incompatible shapes: [${a.join(',')}] vs. [${b.join(',')}]${op ? ` [Op:${op}]` : ''}`)
    out[nd - d] = x === 1 ? y : x
  }
  return out
}

/** tf.Variable: a trainable tensor you update with assign / assign_sub */
class TFVariable extends TFTensor {
  readonly cls: string = 'ResourceVariable'
  name: string
  trainable: boolean
  constructor(t: TT, name: string, trainable: boolean) {
    super(t)
    this.name = name
    this.trainable = trainable
    t.requiresGrad = trainable && isFloatT(t.dtype)
  }
  repr() {
    const np = this.t.ndim ? npRepr(this.t) : scalarStr(this.t)
    return `<tf.Variable '${this.name}:0' shape=${shapeStr(this.t.shape)} dtype=${this.t.dtype}, numpy=${np.includes('\n') ? '\n' : ''}${np}>`
  }
  str() { return this.repr() }
  isa(name: string): boolean { return name === 'Variable' || name === 'Tensor' }
  getAttr(name: string, h: Host): Value | undefined {
    const assign = (op: (old: number, v: number) => number) => fnv(name, (args, kw) => {
      const v = tfOperand(h, args[0] ?? kw.value, this.t)
      if (!v) throw h.err('TypeError', `${name}() needs a tensor or a number`)
      if (v.dtype !== this.t.dtype) throw tfErr(h, `Cannot update variable with shape [${this.t.shape.join(',')}] using a Tensor of dtype ${v.dtype}`)
      if (v.size !== this.t.size && v.size !== 1) throw h.err('ValueError', `Cannot assign value to variable ' ${this.name}:0': Shape mismatch.The variable shape (${this.t.shape.join(', ')}), and the assigned value shape (${v.shape.join(', ')}) are incompatible.`)
      const old = this.t.values()
      const nv = v.values()
      C.writeInto(this.t, old.map((x, i) => op(x, nv[v.size === 1 ? 0 : i])))
      return py.obj(this)
    })
    switch (name) {
      case 'assign': return assign((_o, v) => v)
      case 'assign_add': return assign((o, v) => o + v)
      case 'assign_sub': return assign((o, v) => o - v)
      case 'name': return py.str(`${this.name}:0`)
      case 'trainable': return py.bool(this.trainable)
      case 'value': case 'read_value': return fnv(name, () => tfWrap(C.detach(this.t)))
    }
    return super.getAttr(name, h)
  }
}

/** keras.Variable and every weight of a layer: <Variable path=dense/kernel, shape=(2, 3), dtype=float32, value=…> */
class KVariable extends TFVariable {
  readonly cls: string = 'Variable'
  path: string
  constructor(t: TT, path: string, trainable: boolean) {
    const keep = t.requiresGrad
    super(t, path.split('/').pop()!, trainable)
    t.requiresGrad = keep || (trainable && isFloatT(t.dtype))
    this.path = path
  }
  repr() { return `<Variable path=${this.path}, shape=${shapeStr(this.t.shape)}, dtype=${this.t.dtype}, value=${this.body()}>` }
  getAttr(name: string, h: Host): Value | undefined {
    if (name === 'name') return py.str(this.name)
    if (name === 'path') return py.str(this.path)
    return super.getAttr(name, h)
  }
}

/** with tf.GradientTape() as tape: … ; tape.gradient(loss, [w, b]) */
class GradientTape extends PyObj {
  readonly cls = 'GradientTape'
  persistent: boolean
  used = false
  watched: TT[] = []
  prev = true
  constructor(persistent: boolean) {
    super()
    this.persistent = persistent
  }
  repr() { return '<tensorflow.python.eager.backprop.GradientTape object>' }
  enter(h: Host) {
    const E = stateOf(h).E
    this.prev = E.grad
    E.grad = true
    return py.obj(this)
  }
  exit(h: Host) {
    stateOf(h).E.grad = this.prev
  }
  getAttr(name: string, h: Host): Value | undefined {
    if (name === 'watch') return fnv(name, (args) => {
      for (const v of isSeq(args[0]) ? h.iterate(args[0]) : [args[0]]) {
        const t = tfOf(v)
        if (t && isFloatT(t.dtype)) {
          t.requiresGrad = true
          this.watched.push(t)
        }
      }
      return py.NONE
    })
    if (name === 'gradient') return fnv(name, (args, kw) => {
      if (this.used && !this.persistent) throw h.err('RuntimeError', 'A non-persistent GradientTape can only be used to compute one set of gradients (or jacobians)')
      this.used = true
      const target = tfOf(args[0] ?? kw.target)
      if (!target) throw h.err('TypeError', 'gradient(target, sources): target must be a tensor')
      const srcV = args[1] ?? kw.sources
      const list = isSeq(srcV)
      const sources = (list ? h.iterate(srcV) : [srcV]).map((v) => tfOf(v))
      const E = stateOf(h).E
      const saved = sources.map((s) => s?.grad ?? null)
      sources.forEach((s) => s && (s.grad = null))
      let grads: (TT | null)[] = sources.map(() => null)
      if (target.requiresGrad) {
        const g = target.size === 1 ? null : target.values().map(() => 1)
        C.backward(E, target, g, this.persistent)
        grads = sources.map((s) => s?.grad ?? null)
      }
      sources.forEach((s, i) => s && (s.grad = saved[i]))
      const out = grads.map((g) => (g ? tfWrap(g) : py.NONE))
      return list ? py.list(out) : out[0]
    })
    return undefined
  }
}

/** tf.constant('text'): enough of a string tensor to print, index and hand to TextVectorization */
class TFString extends PyObj {
  readonly cls = 'EagerTensor'
  strs: string[]
  shape: number[]
  constructor(strs: string[], shape: number[]) {
    super()
    this.strs = strs
    this.shape = shape
  }
  private body() {
    const q = (s: string) => `b'${s}'`
    return this.shape.length ? `[${this.strs.map(q).join(' ')}]` : q(this.strs[0])
  }
  repr() {
    const np = this.shape.length ? `array([${this.strs.map((s) => `b'${s}'`).join(', ')}], dtype=object)` : `b'${this.strs[0]}'`
    return `<tf.Tensor: shape=${shapeStr(this.shape)}, dtype=string, numpy=${np}>`
  }
  str() { return `tf.Tensor(${this.body()}, shape=${shapeStr(this.shape)}, dtype=string)` }
  len() { return this.strs.length }
  iter() { return this.strs.map((s) => py.obj(new TFString([s], []))) }
  isa(name: string) { return name === 'Tensor' }
  getAttr(name: string): Value | undefined {
    if (name === 'shape') return py.obj(new TensorShape(this.shape))
    if (name === 'dtype') return py.obj(new TFDtype('string'))
    if (name === 'numpy') return fnv(name, () => (this.shape.length ? py.list(this.strs.map(py.str)) : py.str(this.strs[0])))
    return undefined
  }
}

/** strings on their way through a model: a text input batch, kept beside an empty numeric tensor */
class StrT extends TT {
  strs: string[]
  constructor(strs: string[]) {
    super(NDArray.create(new Array(strs.length).fill(0), [strs.length, 1], 'int64'), 'int64')
    this.strs = strs
  }
}

// ---------------------------------------------------------------- differentiable helpers the engine does not have

/** attach a grad_fn by hand, as tensorCore's own ops do */
const opT = (E: Engine, out: TT, name: string, inputs: (TT | null)[], backward: GradNode['backward']): TT => {
  if (E.grad && inputs.some((x) => x?.requiresGrad)) {
    out.requiresGrad = true
    out.fn = { name, inputs, backward, freed: false }
  }
  return out
}

const allOf = (n: number): number[] => Array.from({ length: n }, (_, i) => i)

/** a tensor rearranged axis by axis, as a copy: slices, flips, padding, tiling and nearest-neighbour resizing are all this */
const gridT = (E: Engine, a: TT, maps: number[][], fill = 0): TT => C.takeT(E, a, C.crossSrc(a.shape, maps), maps.map((m) => m.length), 'TakeBackward', fill)

const sliceAxis = (E: Engine, a: TT, axis: number, start: number, stop: number): TT => gridT(E, a, a.shape.map((n, d) => (d === axis ? Array.from({ length: stop - start }, (_, i) => start + i) : allOf(n))))

const flipAxes = (E: Engine, a: TT, axes: number[]): TT => gridT(E, a, a.shape.map((n, d) => (axes.includes(d) ? allOf(n).reverse() : allOf(n))))

/** tensors along an axis, with that axis removed */
const unstackT = (E: Engine, a: TT, axis: number): TT[] => {
  const shape = a.shape.filter((_, d) => d !== axis)
  const av = a.values()
  const st = cStrides(a.shape)[axis]
  const base = C.crossSrc(a.shape, a.shape.map((n, d) => (d === axis ? [0] : allOf(n))))
  return allOf(a.shape[axis]).map((i) => opT(E, TT.raw(Array.from(base, (b) => av[b + i * st]), shape, a.dtype), 'UnstackBackward', [a], (g) => {
    const ga = new Array<number>(a.size).fill(0)
    base.forEach((b, k) => { ga[b + i * st] = g[k] })
    return [ga]
  }))
}

const scalarT = (v: number): TT => TT.scalar(v, 'float32')
const mulS = (E: Engine, x: TT, v: number): TT => C.binary(E, '*', x, TT.scalar(v, x.dtype), false, true)

/** the product over several axes, in the tensor's own dtype as TensorFlow keeps it */
const prodAxes = (E: Engine, t: TT, axes: number[] | null, keep: boolean): TT => {
  let y = t
  if (axes === null) y = C.prodT(E, t, null, keep)
  else for (const d of [...new Set(axes.map((a) => C.normDim(a, t.ndim)))].sort((a, b) => b - a)) y = C.prodT(E, y, d, keep)
  return y.dtype === t.dtype ? y : C.toDtype(E, y, t.dtype)
}

/** tf.boolean_mask(t, mask, axis): the mask covers axes axis … axis + mask.ndim − 1 */
const booleanMask = (h: Host, E: Engine, t: TT, mask: TT, axis: number): TT => {
  if (!mask.ndim) throw h.err('ValueError', 'mask cannot be scalar.')
  const a = axis < 0 ? axis + t.ndim : axis
  const covered = t.shape.slice(a, a + mask.ndim)
  if (covered.join() !== mask.shape.join()) throw h.err('ValueError', `Shapes (${covered.join(', ')}) and (${mask.shape.join(', ')}) are incompatible`)
  const before = prod(t.shape.slice(0, a))
  const after = prod(t.shape.slice(a + mask.ndim))
  const mv = mask.values()
  const src: number[] = []
  let kept = 0
  mv.forEach((m) => { if (m) kept++ })
  for (let b = 0; b < before; b++) {
    for (let i = 0; i < mv.length; i++) {
      if (!mv[i]) continue
      for (let k = 0; k < after; k++) src.push((b * mv.length + i) * after + k)
    }
  }
  return C.takeT(E, t, src, [...t.shape.slice(0, a), kept, ...t.shape.slice(a + mask.ndim)], 'TakeBackward')
}

// ---------------------------------------------------------------- activations

const softplus = (v: number) => (v > 30 ? v : Math.log1p(Math.exp(v)))
const sigmoidV = (v: number) => 1 / (1 + Math.exp(-v))
const SELU_ALPHA = 1.6732632423543772
const SELU_SCALE = 1.0507009873554805

/** keras.activations.relu with its three options */
const reluOptT = (E: Engine, x: TT, slope: number, maxV: number | null, thr: number): TT => {
  if (!slope && maxV === null && !thr) return C.unary(E, 'relu', x)
  return C.mapT(E, x, (v) => {
    const pos = v > thr ? v : 0
    return (maxV !== null ? Math.min(pos, maxV) : pos) - slope * Math.max(thr - v, 0)
  }, (v) => (v > thr ? (maxV !== null && v > maxV ? 0 : 1) : v < thr ? slope : 0), 'ReluBackward')
}

const relu6T = (E: Engine, x: TT) => C.clamp(E, C.unary(E, 'relu', x), null, 6)
const leakyT = (E: Engine, x: TT, a: number) => C.binary(E, 'maximum', x, C.binary(E, '*', x, TT.scalar(a, 'float32'), false, true))
const eluT = (E: Engine, x: TT, a = 1) => C.mapT(E, x, (v) => (v > 0 ? v : a * (Math.exp(v) - 1)), (v) => (v > 0 ? 1 : a * Math.exp(v)), 'EluBackward')
const seluT = (E: Engine, x: TT) => C.mapT(E, x, (v) => (v > 0 ? SELU_SCALE * v : SELU_SCALE * SELU_ALPHA * (Math.exp(v) - 1)), (v) => (v > 0 ? SELU_SCALE : SELU_SCALE * SELU_ALPHA * Math.exp(v)), 'SeluBackward')
const softplusT = (E: Engine, x: TT) => C.mapT(E, x, softplus, (v) => sigmoidV(v), 'SoftplusBackward')
const silu = (E: Engine, x: TT) => C.mapT(E, x, (v) => v * sigmoidV(v), (v) => sigmoidV(v) + v * sigmoidV(v) * (1 - sigmoidV(v)), 'SiluBackward')
const geluT = (E: Engine, x: TT, approximate = false) => {
  if (approximate) {
    const k = Math.sqrt(2 / Math.PI)
    const inner = (v: number) => k * (v + 0.044715 * v ** 3)
    return C.mapT(E, x, (v) => 0.5 * v * (1 + Math.tanh(inner(v))), (v) => {
      const s = Math.tanh(inner(v))
      return 0.5 * (1 + s) + 0.5 * v * (1 - s * s) * k * (1 + 3 * 0.044715 * v * v)
    }, 'GeluBackward')
  }
  const phi = (v: number) => Math.exp(-0.5 * v * v) / Math.sqrt(2 * Math.PI)
  const Phi = (v: number) => 0.5 * (1 + C.erf(v / Math.SQRT2))
  return C.mapT(E, x, (v) => v * Phi(v), (v) => Phi(v) + v * phi(v), 'GeluBackward')
}

interface ActSpec { name: string; f: (E: Engine, x: TT) => TT }

const ACTS: Record<string, (E: Engine, x: TT) => TT> = {
  linear: (_E, x) => x,
  relu: (E, x) => C.unary(E, 'relu', x),
  sigmoid: (E, x) => C.unary(E, 'sigmoid', x),
  tanh: (E, x) => C.unary(E, 'tanh', x),
  softmax: (E, x) => C.softmax(E, x, -1, false),
  log_softmax: (E, x) => C.softmax(E, x, -1, true),
  leaky_relu: (E, x) => leakyT(E, x, 0.2),
  relu6: relu6T,
  elu: (E, x) => eluT(E, x),
  selu: seluT,
  softplus: softplusT,
  softsign: (E, x) => C.mapT(E, x, (v) => v / (Math.abs(v) + 1), (v) => 1 / (Math.abs(v) + 1) ** 2, 'SoftsignBackward'),
  swish: silu,
  silu,
  gelu: (E, x) => geluT(E, x),
  exponential: (E, x) => C.unary(E, 'exp', x),
  hard_sigmoid: (E, x) => C.mapT(E, x, (v) => Math.min(1, Math.max(0, v / 6 + 0.5)), (v) => (v / 6 + 0.5 > 0 && v / 6 + 0.5 < 1 ? 1 / 6 : 0), 'HardSigmoidBackward'),
  mish: (E, x) => C.mapT(E, x, (v) => v * Math.tanh(softplus(v)), (v) => {
    const t = Math.tanh(softplus(v))
    return t + v * (1 - t * t) * sigmoidV(v)
  }, 'MishBackward'),
}

const toTT = (h: Host, v: Value, what: string): TT => {
  const t = tfOf(v)
  if (t) return t
  if (v.k === 'array') return TT.of(v.a.values(), v.a.shape, 'float32')
  throw h.err('TypeError', `${what} must return a tensor, got ${repr(v)}`)
}

const actArg = (h: Host, v: Value | undefined): ActSpec => {
  if (isNone(v)) return { name: 'linear', f: ACTS.linear }
  const name = v!.k === 'str' ? v!.v : v!.k === 'fn' ? v!.name : ''
  if (ACTS[name]) return { name, f: ACTS[name] }
  if (v!.k === 'str') throw h.err('ValueError', `Could not interpret activation function identifier: ${name}`)
  if (v!.k === 'fn' || v!.k === 'func' || v!.k === 'bound' || v!.k === 'inst' || (v!.k === 'obj' && v!.o.call)) {
    return { name: v!.k === 'func' ? v!.name : v!.k === 'fn' ? v!.name : 'custom', f: (_E, x) => toTT(h, h.call(v!, [tfWrap(x)]), 'an activation function') }
  }
  throw h.err('ValueError', `Could not interpret activation function identifier: ${repr(v!)}`)
}

// ---------------------------------------------------------------- initializers

const compute_fans = (shape: number[]): [number, number] => {
  if (shape.length < 1) return [1, 1]
  if (shape.length === 1) return [shape[0], shape[0]]
  if (shape.length === 2) return [shape[0], shape[1]]
  const rf = prod(shape.slice(0, -2))
  return [shape[shape.length - 2] * rf, shape[shape.length - 1] * rf]
}

interface VS { scale: number; mode: 'fan_in' | 'fan_out' | 'fan_avg'; dist: 'truncated_normal' | 'untruncated_normal' | 'uniform' }

const VARIANCE: Record<string, VS> = {
  GlorotUniform: { scale: 1, mode: 'fan_avg', dist: 'uniform' },
  GlorotNormal: { scale: 1, mode: 'fan_avg', dist: 'truncated_normal' },
  HeUniform: { scale: 2, mode: 'fan_in', dist: 'uniform' },
  HeNormal: { scale: 2, mode: 'fan_in', dist: 'truncated_normal' },
  LecunUniform: { scale: 1, mode: 'fan_in', dist: 'uniform' },
  LecunNormal: { scale: 1, mode: 'fan_in', dist: 'truncated_normal' },
}

const INIT_SIG: Record<string, string> = {
  GlorotUniform: 'seed=None', GlorotNormal: 'seed=None', HeUniform: 'seed=None', HeNormal: 'seed=None', LecunUniform: 'seed=None', LecunNormal: 'seed=None',
  RandomNormal: 'mean=0.0 stddev=0.05 seed=None', RandomUniform: 'minval=-0.05 maxval=0.05 seed=None', TruncatedNormal: 'mean=0.0 stddev=0.05 seed=None',
  Zeros: '', Ones: '', Constant: 'value=0.0', Identity: 'gain=1.0', Orthogonal: 'gain=1.0 seed=None', VarianceScaling: 'scale=1.0 mode=fan_in distribution=truncated_normal seed=None',
}

const INIT_ALIASES: Record<string, string> = {
  glorot_uniform: 'GlorotUniform', glorot_normal: 'GlorotNormal', he_uniform: 'HeUniform', he_normal: 'HeNormal', lecun_uniform: 'LecunUniform', lecun_normal: 'LecunNormal',
  random_normal: 'RandomNormal', random_uniform: 'RandomUniform', truncated_normal: 'TruncatedNormal', zeros: 'Zeros', ones: 'Ones', constant: 'Constant', identity: 'Identity', orthogonal: 'Orthogonal', variance_scaling: 'VarianceScaling', uniform: 'RandomUniform', normal: 'RandomNormal',
}

/** Keras's truncated normal: redraw beyond two standard deviations */
const truncNormal = (r: SandboxRandom): number => {
  for (;;) {
    const z = r.normal()
    if (Math.abs(z) <= 2) return z
  }
}

class Initializer extends PyObj {
  readonly cls: string
  P: Params
  /** a seeded or user-built initializer repeats itself, as Keras documents; null draws from the run's random stream */
  seed: number | null
  fn: Value | null
  constructor(cls: string, P: Params, seed: number | null, fn: Value | null = null) {
    super()
    this.cls = cls
    this.P = P
    this.seed = seed
    this.fn = fn
  }
  repr() { return `<keras.src.initializers.${this.cls} object>` }
  sample(h: Host, shape: number[]): number[] {
    const r: SandboxRandom = this.seed === null ? h.rng : new SandboxRandom(this.seed)
    const n = prod(shape)
    const num = (k: string) => h.num(this.P[k])
    const fill = (f: () => number) => {
      const out = new Array<number>(n)
      for (let i = 0; i < n; i++) out[i] = f()
      return out
    }
    switch (this.cls) {
      case 'Zeros': return new Array<number>(n).fill(0)
      case 'Ones': return new Array<number>(n).fill(1)
      case 'Constant': return new Array<number>(n).fill(num('value'))
      case 'RandomNormal': return fill(() => num('mean') + num('stddev') * r.normal())
      case 'RandomUniform': return fill(() => num('minval') + (num('maxval') - num('minval')) * r.next())
      case 'TruncatedNormal': return fill(() => num('mean') + num('stddev') * truncNormal(r))
      case 'Identity': {
        if (shape.length !== 2) throw h.err('ValueError', `Identity matrix initializer can only be used for 2D matrices. Received: shape=${shapeStr(shape)} of rank ${shape.length}.`)
        const out = new Array<number>(n).fill(0)
        for (let i = 0; i < Math.min(shape[0], shape[1]); i++) out[i * shape[1] + i] = num('gain')
        return out
      }
      case 'Orthogonal': {
        if (shape.length < 2) throw h.err('ValueError', `The tensor to initialize must be at least two-dimensional. Received: shape=${shapeStr(shape)} of rank ${shape.length}.`)
        return orthogonal(r, prod(shape.slice(0, -1)), shape[shape.length - 1]).map((v) => v * num('gain'))
      }
    }
    const vs: VS = VARIANCE[this.cls] ?? { scale: num('scale'), mode: h.str(this.P.mode) as VS['mode'], dist: h.str(this.P.distribution) as VS['dist'] }
    const [fi, fo] = compute_fans(shape)
    const scale = vs.scale / Math.max(1, vs.mode === 'fan_in' ? fi : vs.mode === 'fan_out' ? fo : (fi + fo) / 2)
    if (vs.dist === 'uniform') {
      const lim = Math.sqrt(3 * scale)
      return fill(() => (r.next() * 2 - 1) * lim)
    }
    if (vs.dist === 'untruncated_normal') {
      const sd = Math.sqrt(scale)
      return fill(() => sd * r.normal())
    }
    const sd = Math.sqrt(scale) / 0.87962566103423978
    return fill(() => sd * truncNormal(r))
  }
  /** a tensor of this shape */
  make(h: Host, shape: number[], dtype: TDType = 'float32'): TT {
    if (this.fn) {
      const v = h.call(this.fn, [shapeTuple(shape)], {})
      const t = tfOf(v) ?? TT.of(toArray(v).values(), shape, dtype)
      if (t.shape.join() !== shape.join()) throw h.err('ValueError', `The initializer returned shape ${shapeStr(t.shape)}, expected ${shapeStr(shape)}`)
      return TT.of(t.values(), shape, dtype)
    }
    return TT.of(this.sample(h, shape), shape, dtype)
  }
  call(args: Value[], kw: Kw, h: Host): Value {
    const sv = kw.shape ?? args[0]
    if (!sv) throw h.err('TypeError', `${this.cls}.__call__() missing 1 required positional argument: 'shape'`)
    const dv = kw.dtype ?? args[1]
    const d = dv && dv.k === 'obj' && dv.o instanceof TFDtype ? dv.o.d : dv && dv.k === 'str' ? dv.v : 'float32'
    return tfWrap(this.make(h, h.iterate(sv).map((x) => h.toInt(x)), (TF_DTYPES[d as string] ?? 'float32')))
  }
  getAttr(name: string, h: Host): Value | undefined {
    if (name === 'get_config') return fnv(name, () => jsToPy(initConfig(this, h).config as Json))
    return this.P[name]
  }
}

/** QR by modified Gram-Schmidt: the same Q as Keras's "qr, then make R's diagonal positive" */
const orthogonal = (r: SandboxRandom, rows: number, cols: number): number[] => {
  const m = Math.max(rows, cols)
  const k = Math.min(rows, cols)
  const q: number[][] = []
  for (let j = 0; j < k; j++) {
    const v = Array.from({ length: m }, () => r.normal())
    for (const u of q) {
      const d = u.reduce((s, x, i) => s + x * v[i], 0)
      for (let i = 0; i < m; i++) v[i] -= d * u[i]
    }
    const nrm = Math.hypot(...v)
    q.push(v.map((x) => x / nrm))
  }
  const out = new Array<number>(rows * cols)
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) out[i * cols + j] = rows >= cols ? q[j][i] : q[i][j]
  return out
}

const defaultInit = (h: Host, cls: string): Initializer => {
  const P = bindArgs(h, cls, INIT_SIG[cls], [], {}, null)
  return new Initializer(cls, P, null)
}

const initArg = (h: Host, v: Value | undefined, dflt: string): Initializer => {
  if (isNone(v)) return defaultInit(h, INIT_ALIASES[dflt] ?? dflt)
  if (v!.k === 'obj' && v!.o instanceof Initializer) return v!.o
  if (v!.k === 'str') {
    const cls = INIT_ALIASES[v!.v] ?? v!.v
    if (INIT_SIG[cls] === undefined) throw h.err('ValueError', `Could not interpret initializer identifier: ${v!.v}`)
    return defaultInit(h, cls)
  }
  if (v!.k === 'dict') {
    const cfg = v!.d
    const cls = h.str(cfg.get(py.str('class_name')) ?? py.str(''))
    const conf = cfg.get(py.str('config'))
    const kw: Kw = {}
    if (conf?.k === 'dict') for (const [k, x] of conf.d.items()) kw[h.str(k)] = x
    if (INIT_SIG[cls] === undefined) throw h.err('ValueError', `Could not interpret initializer identifier: ${cls}`)
    return new Initializer(cls, bindArgs(h, cls, INIT_SIG[cls], [], kw, null), null)
  }
  if (v!.k === 'func' || v!.k === 'fn' || v!.k === 'bound' || v!.k === 'inst') return new Initializer(v!.k === 'func' ? v!.name : 'custom', {}, null, v!)
  throw h.err('ValueError', `Could not interpret initializer identifier: ${repr(v!)}`)
}

// ---------------------------------------------------------------- JSON values and configs

/** a whole-number float in a config, which must print as 1.0 and not 1 */
class JFloat {
  v: number
  constructor(v: number) { this.v = v }
}
/** a tuple in a config: get_config shows (3, 3), to_json writes [3, 3] */
class JTuple {
  items: Json[]
  constructor(items: Json[]) { this.items = items }
}
type Json = null | boolean | number | string | JFloat | JTuple | Json[] | { [k: string]: Json }
const jf = (v: number): Json => (Number.isInteger(v) ? new JFloat(v) : v)

const pyToJs = (h: Host, v: Value): Json => {
  switch (v.k) {
    case 'none': return null
    case 'bool': return v.v
    case 'int': return v.v
    case 'float': return jf(v.v)
    case 'str': return v.v
    case 'list': return v.items.map((x) => pyToJs(h, x))
    case 'tuple': return new JTuple(v.items.map((x) => pyToJs(h, x)))
    case 'dict': return Object.fromEntries(v.d.items().map(([k, x]) => [h.str(k), pyToJs(h, x)]))
    case 'obj':
      if (v.o instanceof Initializer) return initConfig(v.o, h)
      if (v.o instanceof Regularizer) return regConfig(v.o)
      if (v.o instanceof TFDtype) return String(v.o.d)
      if (v.o instanceof KLayer) return v.o.serialize(h)
      break
    case 'inst': {
      const l = layerOf(v)
      if (l) return l.serialize(h)
      break
    }
    case 'fn': case 'func': return v.name
  }
  return repr(v)
}

const jsToPy = (j: Json): Value => {
  if (j === null) return py.NONE
  if (j instanceof JFloat) return py.float(j.v)
  if (j instanceof JTuple) return py.tuple(j.items.map(jsToPy))
  if (typeof j === 'boolean') return py.bool(j)
  if (typeof j === 'number') return Number.isInteger(j) ? py.int(j) : py.float(j)
  if (typeof j === 'string') return py.str(j)
  if (Array.isArray(j)) return py.list(j.map(jsToPy))
  return { k: 'dict', d: PyDict.from(Object.entries(j).map(([k, x]) => [py.str(k), jsToPy(x)])) }
}

/** json.dumps: the separators Python uses */
const pyJson = (j: Json): string => {
  if (j instanceof JFloat) return `${j.v}.0`
  if (j instanceof JTuple) return pyJson(j.items)
  if (Array.isArray(j)) return `[${j.map(pyJson).join(', ')}]`
  if (j !== null && typeof j === 'object') return `{${Object.entries(j).map(([k, x]) => `${JSON.stringify(k)}: ${pyJson(x)}`).join(', ')}}`
  return JSON.stringify(j)
}

const objConfig = (module: string, cls: string, config: Json): { [k: string]: Json } => ({ module, class_name: cls, config, registered_name: null })

const initConfig = (i: Initializer, h: Host): { [k: string]: Json } => objConfig('keras.initializers', i.cls, Object.fromEntries(Object.entries(i.P).map(([k, x]) => [k, pyToJs(h, x)])))

// ---------------------------------------------------------------- regularizers

/** kernel_regularizer=regularizers.l2(0.002): a penalty added to the loss */
class Regularizer extends PyObj {
  readonly cls: string
  l1: number
  l2: number
  fn: Value | null
  constructor(cls: string, l1: number, l2: number, fn: Value | null = null) {
    super()
    this.cls = cls
    this.l1 = l1
    this.l2 = l2
    this.fn = fn
  }
  repr() { return `<keras.src.regularizers.regularizers.${this.cls} object>` }
  penalty(E: Engine, w: TT, h: Host): TT {
    if (this.fn) return toTT(h, h.call(this.fn, [tfWrap(w)]), 'a regularizer')
    const f = (kind: 'square' | 'abs', c: number) => C.binary(E, '*', C.reduceSum(E, C.unary(E, kind, w), null, false, false), TT.scalar(c, 'float32'), false, true)
    if (this.l1 && this.l2) return C.binary(E, '+', f('abs', this.l1), f('square', this.l2))
    return this.l1 ? f('abs', this.l1) : f('square', this.l2)
  }
  call(args: Value[], _kw: Kw, h: Host): Value {
    return tfWrap(this.penalty(stateOf(h).E, asFloat(h, args[0]), h))
  }
  getAttr(name: string): Value | undefined {
    if (name === 'l1' && this.cls !== 'L2') return py.float(this.l1)
    if (name === 'l2' && this.cls !== 'L1') return py.float(this.l2)
    if (name === 'get_config') return fnv(name, () => jsToPy(regConfig(this).config as Json))
    return undefined
  }
}

const regConfig = (r: Regularizer): { [k: string]: Json } => objConfig('keras.regularizers', r.cls, r.cls === 'L1' ? { l1: jf(r.l1) } : r.cls === 'L2' ? { l2: jf(r.l2) } : { l1: jf(r.l1), l2: jf(r.l2) })

const regArg = (h: Host, v: Value | undefined): Regularizer | null => {
  if (isNone(v)) return null
  if (v!.k === 'obj' && v!.o instanceof Regularizer) return v!.o
  if (v!.k === 'str') {
    if (v!.v === 'l1' || v!.v === 'L1') return new Regularizer('L1', 0.01, 0)
    if (v!.v === 'l2' || v!.v === 'L2') return new Regularizer('L2', 0, 0.01)
    if (v!.v === 'l1_l2' || v!.v === 'L1L2') return new Regularizer('L1L2', 0, 0)
  }
  if (v!.k === 'dict') {
    const cls = h.str(v!.d.get(py.str('class_name')) ?? py.str(''))
    const conf = v!.d.get(py.str('config'))
    const n = (k: string) => (conf?.k === 'dict' && conf.d.get(py.str(k)) ? h.num(conf.d.get(py.str(k))!) : 0)
    return new Regularizer(cls, n('l1'), n('l2'))
  }
  if (v!.k === 'func' || v!.k === 'bound' || v!.k === 'inst' || v!.k === 'fn') return new Regularizer('custom', 0, 0, v!)
  throw h.err('ValueError', `Could not interpret regularizer identifier: ${repr(v!)}`)
}

// ---------------------------------------------------------------- keras: symbolic tensors and layers

/** one call of a layer in a model graph */
class KNode {
  layer: KLayer
  inputs: KerasTensor[]
  outputs: KerasTensor[] = []
  index: number
  kw: Kw
  constructor(layer: KLayer, inputs: KerasTensor[], kw: Kw) {
    this.layer = layer
    this.inputs = inputs
    this.index = layer.nodes.length
    this.kw = kw
    layer.nodes.push(this)
  }
}

/** Input(shape=…) and every layer output in the Functional API: a shape and where it came from */
class KerasTensor extends PyObj {
  readonly cls = 'KerasTensor'
  shape: KShape
  node: KNode
  outIndex: number
  name: string
  dtype: string
  constructor(shape: KShape, node: KNode, outIndex: number, name: string, dtype = 'float32') {
    super()
    this.shape = shape
    this.node = node
    this.outIndex = outIndex
    this.name = name
    this.dtype = dtype
  }
  get layer() { return this.node.layer }
  repr() { return `<KerasTensor shape=${shapeStr(this.shape)}, dtype=${this.dtype}, sparse=False, ragged=False, name=${this.name}>` }
  getAttr(name: string) {
    if (name === 'shape') return shapeTuple(this.shape)
    if (name === 'dtype') return py.str(this.dtype)
    if (name === 'name') return py.str(this.name)
    if (name === 'ndim') return py.int(this.shape.length)
    return undefined
  }
  isa(name: string) { return name === 'KerasTensor' }
  /** x + y and friends make a parameter-free operation in the graph, as Keras 3 does */
  binop(op: string, other: Value, reflected: boolean, h: Host): Value | undefined {
    const kinds: Record<string, [string, C.BinOpT]> = { '+': ['Add', '+'], '-': ['Subtract', '-'], '*': ['Multiply', '*'], '/': ['TrueDivide', '/'], '**': ['Power', '**'] }
    const k = kinds[op]
    if (!k) throw h.err('ValueError', `A KerasTensor cannot be used with '${op}': use a layer to combine it`)
    const made = (f: (E: Engine, xs: TT[]) => TT, single: boolean) => {
      const l = new Merge(h, k[0], {}, {}, f, single)
      l.isOp = true
      return l
    }
    if (kerasOf(other)) return made((E, [a, b]) => C.binary(E, k[1], reflected ? b : a, reflected ? a : b), false).symbolic(h, [py.obj(this), other], {})
    if (other.k !== 'int' && other.k !== 'float') return undefined
    const c = TT.scalar(other.v, 'float32')
    return made((E, [a]) => C.binary(E, k[1], reflected ? c : a, reflected ? a : c, reflected, !reflected), true).symbolic(h, [py.obj(this)], {})
  }
}

const kerasOf = (v: Value | undefined): KerasTensor | null => (v && v.k === 'obj' && v.o instanceof KerasTensor ? v.o : null)

interface KWeight { name: string; t: TT; declared: boolean; trainable: boolean; bytes: number; layer: KLayer }

const LAYER_STATE = 'keras.layer'

const layerOf = (v: Value | undefined): KLayer | null => {
  if (!v) return null
  if (v.k === 'obj' && v.o instanceof KLayer) return v.o
  if (v.k === 'inst' && v.state[LAYER_STATE]) return v.state[LAYER_STATE] as KLayer
  return null
}

/** the layers a user class keeps in its attributes, also inside lists and dicts */
const attrLayers = (inst: InstValue): KLayer[] => {
  const out: KLayer[] = []
  const visit = (v: Value) => {
    const l = layerOf(v)
    if (l) out.push(l)
    else if (isSeq(v)) v.items.forEach(visit)
    else if (v.k === 'dict') v.d.values().forEach(visit)
  }
  for (const v of inst.attrs.values()) visit(v)
  return out
}

/** a method a user class (or its user-class bases) defines itself */
const userMethod = (cls: ClassValue, name: string): FuncValue | null => {
  const own = cls.ns.get(name)
  if (own && own.k === 'func') return own
  for (const b of cls.bases) {
    if (b.k !== 'class') continue
    const f = userMethod(b, name)
    if (f) return f
  }
  return null
}

const unique = <T,>(xs: T[]): T[] => [...new Set(xs)]

/** a NumPy array / list / tensor as a float32 tensor (Keras casts inputs to float32) */
const asFloat = (h: Host, v: Value | undefined): TT => {
  if (!v) throw h.err('TypeError', 'expected input data')
  if (kerasOf(v)) throw h.err('ValueError', 'A KerasTensor is symbolic: it can only be passed to a Keras layer, not computed with directly')
  const t = tfOf(v)
  if (t) return isFloatT(t.dtype) ? t : TT.of(t.values(), t.shape, 'float32')
  if (v.k === 'obj' && v.o.toArray) {
    const a = v.o.toArray()
    return TT.of(a.values(), a.shape, 'float32')
  }
  const a = toArray(v)
  return TT.of(a.values(), a.shape, 'float32')
}

const strsOf = (v: Value): string[] | null => {
  if (v.k === 'obj' && v.o instanceof TFString) return v.o.strs
  if (isSeq(v) && v.items.length && v.items.every((x) => x.k === 'str')) return v.items.map((x) => (x as { v: string }).v)
  if (isSeq(v) && v.items.length && v.items.every((x) => isSeq(x) && x.items.length === 1 && x.items[0].k === 'str')) return v.items.map((x) => ((x as { items: { v: string }[] }).items[0].v))
  if (v.k === 'str') return [v.v]
  return null
}

/** a model or layer input: strings stay strings, everything else becomes float32 */
const asInput = (h: Host, v: Value | undefined): TT => {
  if (v) {
    let strs = strsOf(v)
    if (!strs && v.k === 'obj' && v.o.iter && v.o.cls === 'Series') {
      const items = v.o.iter(h)
      if (items.length && items.every((x) => x.k === 'str')) strs = items.map((x) => h.str(x))
    }
    if (strs) return new StrT(strs)
  }
  return asFloat(h, v)
}

abstract class KLayer extends PyObj {
  kind: string
  get cls() { return this.kind }
  name: string
  built = false
  inShape: KShape | null = null
  outShape: KShape | null = null
  /** set by input_shape= on the first layer of a Sequential */
  declaredInput: KShape | null = null
  /** the constructor arguments, for get_config and rebuilding */
  P: Params
  trainableFlag = true
  ws: KWeight[] = []
  regs: [TT, Regularizer][] = []
  activity: Regularizer | null = null
  nodes: KNode[] = []
  /** random layers run in "training" mode when called directly */
  defaultTraining = false
  /** the Python object of a user subclass */
  inst: InstValue | null = null
  /** attributes a program sets on the layer */
  extra = new Map<string, Value>()
  constructor(h: Host, kind: string, kw: Kw, P: Params = {}, base?: string) {
    super()
    this.kind = kind
    this.P = P
    this.name = kw.name && kw.name.k === 'str' ? kw.name.v : autoName(h, base ?? snake(kind))
    const is = kw.input_shape ?? (kw.input_dim && !('input_dim' in P) ? py.tuple([kw.input_dim]) : undefined)
    if (is && is.k !== 'none') this.declaredInput = [null, ...h.iterate(is).map((v) => (v.k === 'none' ? null : h.toInt(v)))]
    if (kw.batch_input_shape && kw.batch_input_shape.k !== 'none') this.declaredInput = h.iterate(kw.batch_input_shape).map((v) => (v.k === 'none' ? null : h.toInt(v)))
    if (kw.trainable && !h.truthy(kw.trainable)) this.trainableFlag = false
    const d = kw.dtype
    if (d && d.k !== 'none' && d.k !== 'dict' && !(d.k === 'str' && d.v === 'float32') && !(d.k === 'obj' && d.o instanceof TFDtype && d.o.d === 'float32')) throw h.err('NotImplementedError', `${kind}: dtype=${repr(d)} is not implemented in this sandbox (layers compute in float32)`)
    if (P.activity_regularizer && P.activity_regularizer.k !== 'none') this.activity = regArg(h, P.activity_regularizer)
  }
  /** the Python value of this layer */
  self(): Value { return this.inst ?? py.obj(this) }
  isa(name: string): boolean { return name === 'Layer' || name === 'Operation' }
  children(): KLayer[] { return [] }
  allLayers(): KLayer[] {
    const seen = new Set<KLayer>()
    const walk = (l: KLayer) => {
      if (seen.has(l)) return
      seen.add(l)
      l.children().forEach(walk)
    }
    walk(this)
    return [...seen]
  }
  /** every weight of this layer and the layers inside it: trainable first, as Keras lists them */
  weights(): KWeight[] {
    const all = this.allLayers().flatMap((l) => l.ws)
    return [...all.filter((w) => w.trainable), ...all.filter((w) => !w.trainable)]
  }
  params() { return this.weights().reduce((s, w) => s + w.t.size, 0) }
  addWeight(h: Host, name: string, shape: number[], init: Initializer, opts: { trainable?: boolean; reg?: Regularizer | null; bytes?: number; dtype?: TDType } = {}): TT {
    const t = init.make(h, shape, opts.dtype ?? 'float32')
    const declared = opts.trainable ?? true
    const trainable = declared && this.trainableFlag
    t.requiresGrad = trainable
    this.ws.push({ name, t, declared, trainable, bytes: opts.bytes ?? 4, layer: this })
    if (opts.reg) this.regs.push([t, opts.reg])
    return t
  }
  setTrainable(flag: boolean) {
    for (const l of this.allLayers()) {
      l.trainableFlag = flag
      for (const w of l.ws) {
        w.trainable = flag && w.declared
        w.t.requiresGrad = w.trainable && isFloatT(w.t.dtype)
      }
    }
  }
  /** this layer's own weight penalties */
  penalties(E: Engine, h: Host): TT[] { return this.regs.map(([t, r]) => r.penalty(E, t, h)) }
  /** output shape for an input shape, creating weights on the first call */
  abstract build(h: Host, s: KShape): KShape
  abstract forward(E: Engine, x: TT, training: boolean, h: Host): TT
  /** the outputs for several inputs: what model graphs call */
  run(E: Engine, xs: TT[], training: boolean, h: Host, _kw: Kw): TT[] { return [this.forward(E, xs[0], training, h)] }
  /** output shapes for input shapes (building if need be) */
  outShapes(h: Host, shapes: KShape[], _kw: Kw): KShape[] { return [this.ensureBuilt(h, shapes[0])] }
  /** run with this layer's training flag in force for what it calls, and collect its activity penalty; null = nobody said, so each layer's default decides */
  invoke(E: Engine, xs: TT[], training: boolean | null, h: Host, kw: Kw = {}): TT[] {
    const S = stateOf(h)
    const prev = S.training
    S.training = training
    try {
      const out = this.run(E, xs, training ?? this.defaultTraining, h, kw)
      if (this.activity) for (const o of out) S.losses.push(this.activity.penalty(E, o, h))
      return out
    } finally {
      S.training = prev
    }
  }
  /** the constructor call, for the shape-flow table */
  describe(): string { return `${this.kind}()` }
  note(_i: KShape, _o: KShape): string | null { return null }
  ensureBuilt(h: Host, s: KShape): KShape {
    if (!this.built) {
      this.inShape = s
      this.outShape = this.build(h, s)
      this.built = true
    } else {
      // what Keras's input_spec checks: the axis the weights were built for; other axes may differ between calls
      const want = this.inShape?.[this.inShape.length - 1]
      const got = s[s.length - 1]
      if (this.fixedLastAxis() && want !== null && want !== undefined && got !== null && got !== want) throw h.err('ValueError', `Input 0 of layer "${this.name}" is incompatible with the layer: expected axis -1 of input shape to have value ${want}, but received input with shape ${shapeStr(s)}`)
      this.outShape = this.outFor(h, s) ?? this.outShape
    }
    return this.outShape!
  }
  fixedLastAxis() { return this.ws.length > 0 }
  /** the output shape on a later call with another input shape; weight-free layers can simply build again */
  outFor(h: Host, s: KShape): KShape | null { return this.ws.length ? null : this.build(h, s) }
  repr() { return `<${this.kind} name=${this.name}, built=${this.built ? 'True' : 'False'}>` }

  symbolic(h: Host, items: Value[], kw: Kw): Value {
    const kts = items.map((v) => {
      const k = kerasOf(v)
      if (!k) throw h.err('ValueError', `All inputs to a layer must be KerasTensors when any of them is. Received: ${repr(v)}`)
      return k
    })
    const state = kw.initial_state
    const extra = isNone(state) ? [] : (isSeq(state) ? state.items : [state!]).map((v) => kerasOf(v)!)
    const shapes = this.outShapes(h, kts.map((k) => k.shape), kw)
    const node = new KNode(this, [...kts, ...extra], Object.fromEntries(Object.entries(kw).filter(([k]) => k !== 'initial_state')))
    node.outputs = shapes.map((s, i) => new KerasTensor(s, node, i, autoName(h, 'keras_tensor'), this.outputDtype()))
    return node.outputs.length === 1 ? py.obj(node.outputs[0]) : py.list(node.outputs.map((t) => py.obj(t)))
  }
  outputDtype() { return 'float32' }

  call(args: Value[], kw: Kw, h: Host): Value {
    const x = args[0] ?? kw.inputs
    if (!x) throw h.err('TypeError', `${this.kind}.call() missing 1 required positional argument: 'inputs'`)
    if (!isNone(kw.mask)) throw h.err('NotImplementedError', 'mask= is not implemented in this sandbox')
    const multi = isSeq(x) && x.items.length > 0 && x.items.every((v) => kerasOf(v) || tfOf(v) || v.k === 'array')
    const items = multi ? h.iterate(x) : [x]
    if (items.some((v) => kerasOf(v))) return this.symbolic(h, items, kw)
    const S = stateOf(h)
    const xs = items.map((v) => asInput(h, v))
    this.eagerBuild(h, xs.map((t) => [null, ...t.shape.slice(1)]), kw)
    const training = !isNone(kw.training) ? h.truthy(kw.training!) : S.training
    const out = this.invoke(S.E, this.withState(h, xs, kw), training, h, kw)
    return out.length === 1 ? tfWrap(out[0]) : py.list(out.map(tfWrap))
  }
  eagerBuild(h: Host, shapes: KShape[], kw: Kw) { this.outShapes(h, shapes, kw) }
  /** eager initial_state= becomes extra inputs */
  withState(h: Host, xs: TT[], kw: Kw): TT[] {
    const st = kw.initial_state
    return isNone(st) ? xs : [...xs, ...(isSeq(st) ? st.items : [st!]).map((v) => asFloat(h, v))]
  }

  kvars(list: KWeight[]): Value {
    return py.list(list.map((w) => py.obj(new KVariable(w.t, `${w.layer.name}/${w.name}`, w.trainable))))
  }
  getWeightsPy(): Value { return py.list(this.weights().map((w) => py.arr(ndFor(w.t)))) }
  setWeightsPy(h: Host, v: Value) {
    const ws = h.iterate(v)
    const mine = this.weights()
    if (ws.length !== mine.length) throw h.err('ValueError', `You called \`set_weights(weights)\` on layer '${this.name}' with a weight list of length ${ws.length}, but the layer was expecting ${mine.length} weights.`)
    mine.forEach((w, i) => {
      const a = toArray(tfOf(ws[i]) ? py.arr(ndFor(tfOf(ws[i])!)) : ws[i])
      if (a.shape.join() !== w.t.shape.join()) throw h.err('ValueError', `Layer ${w.layer.name} weight shape ${shapeStr(w.t.shape)} is not compatible with provided weight shape ${shapeStr(a.shape)}.`)
      C.writeInto(w.t, a.values())
    })
  }
  /** Layer.get_config() */
  getConfig(h: Host): { [k: string]: Json } {
    const out: { [k: string]: Json } = { name: this.name, trainable: this.trainableFlag, dtype: objConfig('keras', 'DTypePolicy', { name: 'float32' }) }
    for (const [k, v] of Object.entries(this.P)) {
      if (k === 'activity_regularizer' && v.k === 'none') continue
      if (k.endsWith('_initializer')) out[k] = initConfig(initArg(h, v, 'Zeros'), h)
      else if (k.endsWith('_regularizer')) out[k] = v.k === 'none' ? null : regConfig(regArg(h, v)!)
      else if (v.k === 'fn') out[k] = v.name
      else out[k] = pyToJs(h, v)
    }
    return out
  }
  /** keras.saving.serialize_keras_object */
  serialize(h: Host): { [k: string]: Json } {
    const out = objConfig('keras.layers', this.kind, this.getConfig(h))
    if (this.inShape) out.build_config = { input_shape: this.inShape }
    return out
  }
  getAttr(name: string, h: Host): Value | undefined {
    const fn = (call: (args: Value[], kw: Kw) => Value) => fnv(name, call)
    switch (name) {
      case 'name': return py.str(this.name)
      case 'built': return py.bool(this.built)
      case 'trainable': return py.bool(this.trainableFlag)
      case 'dtype': return py.str('float32')
      case 'weights': case 'variables': return this.kvars(this.weights())
      case 'trainable_weights': case 'trainable_variables': return this.kvars(this.weights().filter((w) => w.trainable))
      case 'non_trainable_weights': case 'non_trainable_variables': return this.kvars(this.weights().filter((w) => !w.trainable))
      case 'get_weights': return fn(() => this.getWeightsPy())
      case 'set_weights': return fn((args, kw) => { this.setWeightsPy(h, args[0] ?? kw.weights); return py.NONE })
      case 'count_params': return fn(() => {
        if (!this.built) throw h.err('ValueError', `You tried to call \`count_params\` on layer '${this.name}', but the layer isn't built. You can build it manually via: \`layer.build(input_shape)\`.`)
        return py.int(this.params())
      })
      case 'input_shape': case 'output_shape': {
        const s = name === 'input_shape' ? this.inShape : this.outShape
        if (!s) throw h.err('AttributeError', `The layer "${this.name}" has never been called and thus has no defined ${name}.`)
        return shapeTuple(s)
      }
      case 'input': case 'output': {
        const n = this.nodes[this.nodes.length - 1]
        if (!n) throw h.err('AttributeError', `The layer "${this.name}" has never been called and thus has no defined ${name}.`)
        const list = name === 'input' ? n.inputs : n.outputs
        return list.length === 1 ? py.obj(list[0]) : py.list(list.map((t) => py.obj(t)))
      }
      case 'get_config': return fn(() => jsToPy(this.getConfig(h)))
      case 'call': return fn((args, kw) => this.call(args, kw, h))
      case 'compute_output_shape': return fn((args, kw) => {
        const s: KShape = h.iterate(args[0] ?? kw.input_shape).map((v) => (v.k === 'none' ? null : h.toInt(v)))
        return shapeTuple(this.outShapes(h, [s], {})[0])
      })
      case 'losses': return py.list(this.allLayers().flatMap((l) => l.penalties(stateOf(h).E, h)).map(tfWrap))
      case 'add_loss': return fn((args) => { stateOf(h).losses.push(asFloat(h, args[0])); return py.NONE })
      case 'add_weight': return fn((args, kw) => {
        const P = bindArgs(h, 'add_weight', 'shape=None initializer=None dtype=None trainable=True autocast=True regularizer=None !constraint=None aggregation=none name=None', args, kw, null)
        const shape = isNone(P.shape) ? [] : (P.shape.k === 'int' ? [P.shape.v] : h.iterate(P.shape).map((v) => h.toInt(v)))
        const wn = P.name.k === 'none' ? autoName(h, 'variable') : h.str(P.name)
        const dtype = P.dtype.k === 'none' ? 'float32' : (P.dtype.k === 'obj' && P.dtype.o instanceof TFDtype ? P.dtype.o.d : h.str(P.dtype)) as TDType
        const trainable = h.truthy(P.trainable)
        const t = this.addWeight(h, wn, shape, initArg(h, P.initializer, 'GlorotUniform'), { trainable, reg: regArg(h, P.regularizer), dtype })
        return py.obj(new KVariable(t, `${this.name}/${wn}`, trainable && this.trainableFlag))
      })
      case '__class__': return { k: 'type', name: this.kind, call: () => { throw h.err('TypeError', 'call the layer class itself') } }
    }
    const w = this.ws.find((x) => x.name === name)
    if (w) return py.obj(new KVariable(w.t, `${this.name}/${w.name}`, w.trainable))
    return this.extra.get(name) ?? this.attr(name, h)
  }
  attr(name: string, _h: Host): Value | undefined { return this.P[name] }
  setAttr(name: string, v: Value, h: Host): boolean {
    if (name === 'trainable') {
      this.setTrainable(h.truthy(v))
      return true
    }
    if (name === 'name' && v.k === 'str') {
      this.name = v.v
      return true
    }
    this.extra.set(name, v)
    return true
  }
}

class InputLayer extends KLayer {
  shape: KShape
  dtypeName: string
  constructor(h: Host, shape: KShape, kw: Kw, dtypeName = 'float32') {
    super(h, 'InputLayer', { name: kw.name ?? py.NONE }, {}, 'input_layer')
    this.shape = shape
    this.dtypeName = dtypeName
    this.built = true
    this.inShape = shape
    this.outShape = shape
  }
  build(_h: Host, s: KShape) { return s }
  forward(_E: Engine, x: TT) { return x }
  getConfig(): { [k: string]: Json } { return { batch_shape: new JTuple(this.shape), dtype: this.dtypeName, sparse: false, name: this.name } }
}

/** weights and shapes for layers that hold a kernel and a bias */
const pairArg = (h: Host, v: Value | undefined, d: number): [number, number] => {
  if (isNone(v)) return [d, d]
  if (isSeq(v)) {
    if (v.items.length !== 2) throw h.err('ValueError', `Expected a tuple of 2 integers. Received: ${repr(v)}`)
    return [h.toInt(v.items[0]), h.toInt(v.items[1])]
  }
  const n = h.toInt(v!)
  return [n, n]
}

const singleArg = (h: Host, v: Value | undefined, d: number): number => {
  if (isNone(v)) return d
  if (isSeq(v)) return h.toInt(v.items[0])
  return h.toInt(v!)
}

type PadMode = 'valid' | 'same' | 'causal'

const padMode = (h: Host, v: Value | undefined, causal = false): PadMode => {
  const s = v && v.k === 'str' ? v.v.toLowerCase() : 'valid'
  if (s !== 'valid' && s !== 'same' && !(causal && s === 'causal')) throw h.err('ValueError', `The \`padding\` argument must be a string, one of "valid", "same"${causal ? ', "causal"' : ''}. Received: padding=${v ? repr(v) : 'None'}`)
  return s as PadMode
}

/** output size: valid ⌊(n − k) / s⌋ + 1, same ⌈n / s⌉ */
const kOut = (n: number, k: number, s: number, mode: PadMode) => (mode === 'valid' ? Math.floor((n - k) / s) + 1 : Math.ceil(n / s))
const kOutN = (n: number | null, k: number, s: number, mode: PadMode) => (n === null ? null : kOut(n, k, s, mode))
/** the padding TF uses for 'same': total = max((⌈n/s⌉ − 1)·s + k − n, 0), split low / high */
const samePad = (n: number, k: number, s: number): [number, number] => {
  const total = Math.max((Math.ceil(n / s) - 1) * s + k - n, 0)
  return [Math.floor(total / 2), total - Math.floor(total / 2)]
}

/** NHWC → NCHW, run a channels-first kernel, back to NHWC; asymmetric 'same' padding is applied first */
const nhwc = (E: Engine, x: TT, pad: [[number, number], [number, number]], run: (y: TT) => TT): TT => {
  let y = C.permuteT(E, x, [0, 3, 1, 2])
  const [[pt, pb], [pl, pr]] = pad
  if (pt || pb || pl || pr) y = C.padT(E, y, [pl, pr, pt, pb], 'constant', 0)
  return C.permuteT(E, run(y), [0, 2, 3, 1])
}

const dilated = (k: number, d: number) => k + (k - 1) * (d - 1)

/** spread a kernel (kh, kw, in, out) with zeros between its elements */
const dilateKernel = (E: Engine, w: TT, d: [number, number]): TT => {
  if (d[0] === 1 && d[1] === 1) return w
  const m = (k: number, dd: number) => Array.from({ length: dilated(k, dd) }, (_, i) => (i % dd === 0 ? i / dd : -1))
  return gridT(E, w, [m(w.shape[0], d[0]), m(w.shape[1], d[1]), allOf(w.shape[2]), allOf(w.shape[3])])
}

/** x (N, H, W, C), kernel (kh, kw, C, out) */
const convNHWC = (E: Engine, x: TT, w: TT, b: TT | null, stride: [number, number], dil: [number, number], mode: PadMode): TT => {
  const k = dilateKernel(E, w, dil)
  const wOIHW = C.permuteT(E, k, [3, 2, 0, 1])
  const pad: [[number, number], [number, number]] = mode === 'same' ? [samePad(x.shape[1], k.shape[0], stride[0]), samePad(x.shape[2], k.shape[1], stride[1])] : mode === 'causal' ? [[0, 0], [k.shape[1] - 1, 0]] : [[0, 0], [0, 0]]
  return nhwc(E, x, pad, (z) => C.conv2d(E, z, wOIHW, b, { stride, padding: [0, 0] }))
}

const CHANNELS_LAST = '!data_format=None'

/** a kernel layer's initializers and regularizers */
const CONV_COMMON = 'activation=None use_bias=True kernel_initializer=glorot_uniform bias_initializer=zeros kernel_regularizer=None bias_regularizer=None activity_regularizer=None !kernel_constraint=None !bias_constraint=None'

class Dense extends KLayer {
  units: number
  act: ActSpec
  useBias: boolean
  w: TT | null = null
  b: TT | null = null
  constructor(h: Host, P: Params, kw: Kw) {
    super(h, 'Dense', kw, P)
    const units = isNaN(h.num(P.units)) ? 0 : h.toInt(P.units)
    if (!Number.isInteger(units) || units < 1) throw h.err('ValueError', `Received an invalid value for \`units\`, expected a positive integer. Received: units=${units}`)
    this.units = units
    this.act = actArg(h, P.activation)
    this.useBias = h.truthy(P.use_bias)
  }
  build(h: Host, s: KShape) {
    if (s.length < 2) throw h.err('ValueError', `Input 0 of layer "${this.name}" is incompatible with the layer: expected min_ndim=2, found ndim=${s.length}. Full shape received: ${shapeStr(s)}`)
    const inF = s[s.length - 1]
    if (inF === null || inF === undefined) throw h.err('ValueError', 'The last dimension of the inputs to a Dense layer should be defined. Found None.')
    if (s.length > 2) noteOnce(h, 'keras_dense_nd', 'warn', { shape: shapeStr(s), units: this.units })
    this.w = this.addWeight(h, 'kernel', [inF, this.units], initArg(h, this.P.kernel_initializer, 'GlorotUniform'), { reg: regArg(h, this.P.kernel_regularizer) })
    this.b = this.useBias ? this.addWeight(h, 'bias', [this.units], initArg(h, this.P.bias_initializer, 'Zeros'), { reg: regArg(h, this.P.bias_regularizer) }) : null
    return [...s.slice(0, -1), this.units]
  }
  outFor(_h: Host, s: KShape): KShape { return [...s.slice(0, -1), this.units] }
  describe() { return `Dense(${this.units}${this.act.name !== 'linear' ? `, activation='${this.act.name}'` : ''})` }
  note(i: KShape) { return `(${i[i.length - 1]} + ${this.useBias ? 1 : 0}) × ${this.units} = ${this.params().toLocaleString('en-US')}` }
  forward(E: Engine, x: TT) {
    const inF = this.w!.shape[0]
    if (x.shape[x.ndim - 1] !== inF) throw new PyError('ValueError', `Input 0 of layer "${this.name}" is incompatible with the layer: expected axis -1 of input shape to have value ${inF}, but received input with shape ${shapeStr([null, ...x.shape.slice(1)])}`)
    // Dense acts on the last axis: x @ kernel + bias
    let y = C.matmul(E, x, this.w!)
    if (this.b) y = C.binary(E, '+', y, this.b)
    return this.act.f(E, y)
  }
  attr(name: string, h: Host) {
    if (name === 'activation') return actValue(this.act, h)
    return super.attr(name, h)
  }
}

const actValue = (a: ActSpec, h: Host): Value => fnv(a.name, (args) => tfWrap(a.f(stateOf(h).E, asFloat(h, args[0]))))

/** Conv1D and Conv2D: Conv1D is a Conv2D of height 1, with a kernel that has no height axis */
class Conv extends KLayer {
  rank: 1 | 2
  filters: number
  k: [number, number]
  strides: [number, number]
  dil: [number, number]
  padding: PadMode
  act: ActSpec
  useBias: boolean
  w: TT | null = null
  b: TT | null = null
  constructor(h: Host, kind: 'Conv1D' | 'Conv2D', P: Params, kw: Kw) {
    super(h, kind, kw, P)
    this.rank = kind === 'Conv1D' ? 1 : 2
    this.filters = h.toInt(P.filters)
    if (this.rank === 1) {
      this.k = [1, singleArg(h, P.kernel_size, 3)]
      this.strides = [1, singleArg(h, P.strides, 1)]
      this.dil = [1, singleArg(h, P.dilation_rate, 1)]
      P.kernel_size = py.tuple([py.int(this.k[1])])
      P.strides = py.tuple([py.int(this.strides[1])])
    } else {
      this.k = pairArg(h, P.kernel_size, 3)
      this.strides = pairArg(h, P.strides, 1)
      this.dil = pairArg(h, P.dilation_rate, 1)
      P.kernel_size = py.tuple(this.k.map(py.int))
      P.strides = py.tuple(this.strides.map(py.int))
    }
    if ((this.dil[0] > 1 || this.dil[1] > 1) && (this.strides[0] > 1 || this.strides[1] > 1)) throw h.err('ValueError', `\`strides > 1\` not supported in conjunction with \`dilation_rate > 1\`. Received: strides=${repr(P.strides)} and dilation_rate=${repr(P.dilation_rate)}`)
    this.padding = padMode(h, P.padding, this.rank === 1)
    this.act = actArg(h, P.activation)
    this.useBias = h.truthy(P.use_bias)
  }
  build(h: Host, s: KShape) {
    const nd = this.rank + 2
    if (s.length !== nd) throw h.err('ValueError', `Input 0 of layer "${this.name}" is incompatible with the layer: expected min_ndim=${nd}, found ndim=${s.length}. Full shape received: ${shapeStr(s)}`)
    const Cin = s[nd - 1] as number
    if (s[nd - 1] === null) throw h.err('ValueError', 'The channel dimension of the inputs should be defined. Found None.')
    const shape = this.rank === 2 ? [this.k[0], this.k[1], Cin, this.filters] : [this.k[1], Cin, this.filters]
    const out = this.outFor(h, s)
    // Keras keeps the kernel as (kh, kw, in, out); the engine wants (out, in, kh, kw)
    this.w = this.addWeight(h, 'kernel', shape, initArg(h, this.P.kernel_initializer, 'GlorotUniform'), { reg: regArg(h, this.P.kernel_regularizer) })
    this.b = this.useBias ? this.addWeight(h, 'bias', [this.filters], initArg(h, this.P.bias_initializer, 'Zeros'), { reg: regArg(h, this.P.bias_regularizer) }) : null
    return out
  }
  outFor(h: Host, s: KShape): KShape {
    const H = this.rank === 2 ? s[1] : 1
    const W = this.rank === 2 ? s[2] : s[1]
    const ek: [number, number] = [dilated(this.k[0], this.dil[0]), dilated(this.k[1], this.dil[1])]
    const Ho = kOutN(H, ek[0], this.strides[0], this.padding)
    const Wo = kOutN(W, ek[1], this.strides[1], this.padding)
    const shape = this.rank === 2 ? [this.k[0], this.k[1], s[3], this.filters] : [this.k[1], s[2], this.filters]
    if ((Ho !== null && Ho < 1) || (Wo !== null && Wo < 1)) throw h.err('ValueError', `Computed output size would be negative. Received \`inputs shape=${shapeStr(s)}\`, \`kernel shape=(${shape.join(', ')})\`, \`dilation_rate=[${this.rank === 2 ? this.dil.join(' ') : this.dil[1]}]\`.`)
    return this.rank === 2 ? [s[0], Ho, Wo, this.filters] : [s[0], Wo, this.filters]
  }
  describe() {
    const ks = this.rank === 2 ? `(${this.k.join(', ')})` : `${this.k[1]}`
    const st = this.rank === 2 ? (this.strides[0] !== 1 || this.strides[1] !== 1 ? `, strides=(${this.strides.join(', ')})` : '') : this.strides[1] !== 1 ? `, strides=${this.strides[1]}` : ''
    return `${this.kind}(${this.filters}, ${ks}${st}${this.padding !== 'valid' ? `, padding='${this.padding}'` : ''}${this.act.name !== 'linear' ? `, activation='${this.act.name}'` : ''})`
  }
  note(i: KShape, o: KShape) {
    const axis = this.rank === 2 ? 0 : 1
    const n = i[1] as number
    const k = dilated(this.k[axis], this.dil[axis])
    const s = this.strides[axis]
    const f = this.padding === 'valid' ? `⌊(${n} − ${k}) / ${s}⌋ + 1 = ${o[1]}` : `⌈${n} / ${s}⌉ = ${o[1]}`
    const ks = this.rank === 2 ? `${this.k[0]}·${this.k[1]}` : `${this.k[1]}`
    return `${f} · params (${ks}·${i[i.length - 1]} + ${this.useBias ? 1 : 0})·${this.filters}`
  }
  forward(E: Engine, x: TT, _t: boolean, h: Host) {
    const nd = this.rank + 2
    if (x.ndim !== nd) throw h.err('ValueError', `Input 0 of layer "${this.name}" is incompatible with the layer: expected min_ndim=${nd}, found ndim=${x.ndim}. Full shape received: ${shapeStr([null, ...x.shape.slice(1)])}`)
    const x4 = this.rank === 1 ? C.unsqueezeT(E, x, 1) : x
    const w4 = this.rank === 1 ? C.unsqueezeT(E, this.w!, 0) : this.w!
    const y = convNHWC(E, x4, w4, this.b, this.strides, this.dil, this.padding)
    return this.act.f(E, this.rank === 1 ? C.squeezeT(E, y, 1) : y)
  }
  attr(name: string, h: Host) {
    if (name === 'activation') return actValue(this.act, h)
    return super.attr(name, h)
  }
}

class Conv2DTranspose extends KLayer {
  filters: number
  k: [number, number]
  strides: [number, number]
  padding: PadMode
  opad: [number, number] | null
  act: ActSpec
  useBias: boolean
  w: TT | null = null
  b: TT | null = null
  constructor(h: Host, P: Params, kw: Kw) {
    super(h, 'Conv2DTranspose', kw, P)
    this.filters = h.toInt(P.filters)
    this.k = pairArg(h, P.kernel_size, 3)
    this.strides = pairArg(h, P.strides, 1)
    this.padding = padMode(h, P.padding)
    this.opad = isNone(P.output_padding) ? null : pairArg(h, P.output_padding, 0)
    this.act = actArg(h, P.activation)
    this.useBias = h.truthy(P.use_bias)
    P.kernel_size = py.tuple(this.k.map(py.int))
    P.strides = py.tuple(this.strides.map(py.int))
  }
  /** keras.src.utils.conv_utils: deconv_output_length */
  outLen(n: number, axis: 0 | 1) {
    const k = this.k[axis]
    const s = this.strides[axis]
    if (!this.opad) return this.padding === 'valid' ? n * s + Math.max(k - s, 0) : n * s
    const pad = this.padding === 'same' ? Math.floor(k / 2) : 0
    return (n - 1) * s + k - 2 * pad + this.opad[axis]
  }
  build(h: Host, s: KShape) {
    if (s.length !== 4) throw h.err('ValueError', `Input 0 of layer "${this.name}" is incompatible with the layer: expected ndim=4, found ndim=${s.length}. Full shape received: ${shapeStr(s)}`)
    const Cin = s[3] as number
    this.w = this.addWeight(h, 'kernel', [this.k[0], this.k[1], this.filters, Cin], initArg(h, this.P.kernel_initializer, 'GlorotUniform'), { reg: regArg(h, this.P.kernel_regularizer) })
    this.b = this.useBias ? this.addWeight(h, 'bias', [this.filters], initArg(h, this.P.bias_initializer, 'Zeros'), { reg: regArg(h, this.P.bias_regularizer) }) : null
    return this.outFor(h, s)
  }
  outFor(_h: Host, s: KShape): KShape { return [s[0], s[1] === null ? null : this.outLen(s[1], 0), s[2] === null ? null : this.outLen(s[2], 1), this.filters] }
  describe() { return `Conv2DTranspose(${this.filters}, (${this.k.join(', ')})${this.strides[0] !== 1 || this.strides[1] !== 1 ? `, strides=(${this.strides.join(', ')})` : ''}${this.padding === 'same' ? ", padding='same'" : ''}${this.act.name !== 'linear' ? `, activation='${this.act.name}'` : ''})` }
  note(i: KShape, o: KShape) { return `${this.padding === 'same' ? `${i[1]} × ${this.strides[0]} = ${o[1]}` : `${i[1]} × ${this.strides[0]} + max(${this.k[0]} − ${this.strides[0]}, 0) = ${o[1]}`} · params (${this.k[0]}·${this.k[1]}·${i[3]} + ${this.useBias ? 1 : 0})·${this.filters}` }
  forward(E: Engine, x: TT) {
    const Ho = this.outLen(x.shape[1], 0)
    const Wo = this.outLen(x.shape[2], 1)
    const low: [number, number] = this.padding === 'same' ? [samePad(Ho, this.k[0], this.strides[0])[0], samePad(Wo, this.k[1], this.strides[1])[0]] : [0, 0]
    // the engine crops the same amount from both ends: Keras's 'same' may crop one more from the far end
    const base = (n: number, a: 0 | 1) => (n - 1) * this.strides[a] - 2 * low[a] + this.k[a]
    const [bh, bw] = [base(x.shape[1], 0), base(x.shape[2], 1)]
    const wT = C.permuteT(E, this.w!, [3, 2, 0, 1])
    const y = C.convTranspose2d(E, C.permuteT(E, x, [0, 3, 1, 2]), wT, this.b, this.strides, low, [Math.max(Ho - bh, 0), Math.max(Wo - bw, 0)])
    const cut = bh > Ho || bw > Wo ? C.narrowT(E, C.narrowT(E, y, 2, 0, Ho), 3, 0, Wo) : y
    return this.act.f(E, C.permuteT(E, cut, [0, 2, 3, 1]))
  }
}

/** DepthwiseConv2D (one kernel per input channel) and SeparableConv2D (that, then a 1 × 1 convolution) */
class SepConv extends KLayer {
  separable: boolean
  filters: number
  k: [number, number]
  strides: [number, number]
  dil: [number, number]
  padding: PadMode
  dm: number
  act: ActSpec
  useBias: boolean
  dw: TT | null = null
  pw: TT | null = null
  b: TT | null = null
  constructor(h: Host, kind: 'DepthwiseConv2D' | 'SeparableConv2D', P: Params, kw: Kw) {
    super(h, kind, kw, P)
    this.separable = kind === 'SeparableConv2D'
    this.filters = this.separable ? h.toInt(P.filters) : 0
    this.k = pairArg(h, P.kernel_size, 3)
    this.strides = pairArg(h, P.strides, 1)
    this.dil = pairArg(h, P.dilation_rate, 1)
    this.padding = padMode(h, P.padding)
    this.dm = h.toInt(P.depth_multiplier)
    this.act = actArg(h, P.activation)
    this.useBias = h.truthy(P.use_bias)
    P.kernel_size = py.tuple(this.k.map(py.int))
    P.strides = py.tuple(this.strides.map(py.int))
  }
  build(h: Host, s: KShape) {
    if (s.length !== 4) throw h.err('ValueError', `Input 0 of layer "${this.name}" is incompatible with the layer: expected ndim=4, found ndim=${s.length}. Full shape received: ${shapeStr(s)}`)
    const Cin = s[3] as number
    const shape = [this.k[0], this.k[1], Cin, this.dm]
    const out = this.separable ? this.filters : Cin * this.dm
    if (this.separable) {
      this.dw = this.addWeight(h, 'depthwise_kernel', shape, initArg(h, this.P.depthwise_initializer, 'GlorotUniform'), { reg: regArg(h, this.P.depthwise_regularizer) })
      this.pw = this.addWeight(h, 'pointwise_kernel', [1, 1, Cin * this.dm, this.filters], initArg(h, this.P.pointwise_initializer, 'GlorotUniform'), { reg: regArg(h, this.P.pointwise_regularizer) })
    } else this.dw = this.addWeight(h, 'kernel', shape, initArg(h, this.P.depthwise_initializer, 'GlorotUniform'), { reg: regArg(h, this.P.depthwise_regularizer) })
    this.b = this.useBias ? this.addWeight(h, 'bias', [out], initArg(h, this.P.bias_initializer, 'Zeros'), { reg: regArg(h, this.P.bias_regularizer) }) : null
    return this.outFor(h, s)
  }
  outFor(_h: Host, s: KShape): KShape {
    const dim = (n: number | null, a: 0 | 1) => kOutN(n, dilated(this.k[a], this.dil[a]), this.strides[a], this.padding)
    return [s[0], dim(s[1], 0), dim(s[2], 1), this.separable ? this.filters : (s[3] as number) * this.dm]
  }
  describe() { return `${this.kind}(${this.separable ? `${this.filters}, ` : ''}(${this.k.join(', ')})${this.padding === 'same' ? ", padding='same'" : ''}${this.act.name !== 'linear' ? `, activation='${this.act.name}'` : ''})` }
  note(i: KShape) {
    const kk = this.k[0] * this.k[1]
    const Cin = i[3] as number
    return this.separable ? `depthwise ${kk}·${Cin}·${this.dm} + pointwise ${Cin}·${this.dm}·${this.filters} + bias ${this.useBias ? this.filters : 0}` : `${kk}·${Cin}·${this.dm} + bias ${this.useBias ? Cin * this.dm : 0}`
  }
  forward(E: Engine, x: TT) {
    const Cin = x.shape[3]
    const parts = allOf(Cin).map((c) => convNHWC(E, sliceAxis(E, x, 3, c, c + 1), sliceAxis(E, this.dw!, 2, c, c + 1), null, this.strides, this.dil, this.padding))
    let y = parts.length === 1 ? parts[0] : C.catT(E, parts, 3)
    if (this.separable) y = convNHWC(E, y, this.pw!, this.b, [1, 1], [1, 1], 'valid')
    else if (this.b) y = C.binary(E, '+', y, this.b)
    return this.act.f(E, y)
  }
}

/** pooling with TF's 'same': the windows stop at the border, so padding neither wins a max nor dilutes a mean */
const poolSame = (E: Engine, x: TT, k: [number, number], s: [number, number], mode: 'max' | 'avg'): TT => {
  const [N, H, W, Cc] = x.shape
  const Ho = Math.ceil(H / s[0])
  const Wo = Math.ceil(W / s[1])
  const low: [number, number] = [samePad(H, k[0], s[0])[0], samePad(W, k[1], s[1])[0]]
  const xv = x.values()
  const size = N * Ho * Wo * Cc
  const windows: number[][] = new Array(size)
  const out = new Array<number>(size)
  for (let n = 0; n < N; n++) {
    for (let i = 0; i < Ho; i++) {
      for (let j = 0; j < Wo; j++) {
        for (let c = 0; c < Cc; c++) {
          const win: number[] = []
          for (let u = 0; u < k[0]; u++) {
            const p = i * s[0] + u - low[0]
            if (p < 0 || p >= H) continue
            for (let v = 0; v < k[1]; v++) {
              const q = j * s[1] + v - low[1]
              if (q >= 0 && q < W) win.push(((n * H + p) * W + q) * Cc + c)
            }
          }
          const o = ((n * Ho + i) * Wo + j) * Cc + c
          windows[o] = win
          out[o] = mode === 'max' ? win.reduce((m, f) => Math.max(m, xv[f]), -Infinity) : win.reduce((a, f) => a + xv[f], 0) / win.length
        }
      }
    }
  }
  return opT(E, TT.of(out, [N, Ho, Wo, Cc], x.dtype), mode === 'max' ? 'MaxPoolBackward' : 'AvgPoolBackward', [x], (g) => {
    const gx = new Array<number>(xv.length).fill(0)
    windows.forEach((win, o) => {
      if (mode === 'avg') for (const f of win) gx[f] += g[o] / win.length
      else {
        let best = win[0]
        for (const f of win) if (xv[f] > xv[best]) best = f
        gx[best] += g[o]
      }
    })
    return [gx]
  })
}

const poolNHWC = (E: Engine, x: TT, k: [number, number], s: [number, number], padding: PadMode, mode: 'max' | 'avg'): TT =>
  padding === 'same' ? poolSame(E, x, k, s, mode) : nhwc(E, x, [[0, 0], [0, 0]], (z) => C.pool2d(E, z, k, s, [0, 0], mode))

class Pool extends KLayer {
  rank: 1 | 2
  k: [number, number]
  s: [number, number]
  padding: PadMode
  mode: 'max' | 'avg'
  constructor(h: Host, kind: string, rank: 1 | 2, mode: 'max' | 'avg', P: Params, kw: Kw) {
    super(h, kind, kw, P)
    this.rank = rank
    this.mode = mode
    if (rank === 1) {
      this.k = [1, singleArg(h, P.pool_size, 2)]
      this.s = isNone(P.strides) ? this.k : [1, singleArg(h, P.strides, 1)]
      P.pool_size = py.tuple([py.int(this.k[1])])
    } else {
      this.k = pairArg(h, P.pool_size, 2)
      this.s = isNone(P.strides) ? this.k : pairArg(h, P.strides, 1)
      P.pool_size = py.tuple(this.k.map(py.int))
    }
    this.padding = padMode(h, P.padding)
  }
  build(h: Host, s: KShape) {
    const nd = this.rank + 2
    if (s.length !== nd) throw h.err('ValueError', `Input 0 of layer "${this.name}" is incompatible with the layer: expected ndim=${nd}, found ndim=${s.length}. Full shape received: ${shapeStr(s)}`)
    const dim = (n: number | null, a: 0 | 1) => {
      const d = kOutN(n, this.k[a], this.s[a], this.padding)
      if (d !== null && d < 1) throw h.err('ValueError', `Computed output size would be negative. Received: \`inputs.shape=${shapeStr(s)}\` and \`pool_size=(${this.rank === 2 ? this.k.join(', ') : `${this.k[1]},`})\`.`)
      return d
    }
    return this.rank === 2 ? [s[0], dim(s[1], 0), dim(s[2], 1), s[3]] : [s[0], dim(s[1], 1), s[2]]
  }
  describe() { return `${this.kind}((${this.rank === 2 ? this.k.join(', ') : `${this.k[1]},`}))` }
  note(i: KShape, o: KShape) {
    const a = this.rank === 2 ? 0 : 1
    return this.padding === 'same' ? `⌈${i[1]} / ${this.s[a]}⌉ = ${o[1]}` : `⌊(${i[1]} − ${this.k[a]}) / ${this.s[a]}⌋ + 1 = ${o[1]}`
  }
  forward(E: Engine, x: TT) {
    if (this.rank === 2) return poolNHWC(E, x, this.k, this.s, this.padding, this.mode)
    return C.squeezeT(E, poolNHWC(E, C.unsqueezeT(E, x, 1), this.k, this.s, this.padding, this.mode), 1)
  }
}

class Simple extends KLayer {
  shapeFn: (s: KShape, h: Host) => KShape
  fwd: (E: Engine, x: TT, training: boolean) => TT
  desc: string
  constructor(h: Host, kind: string, kw: Kw, P: Params, shapeFn: (s: KShape, h: Host) => KShape, fwd: (E: Engine, x: TT, training: boolean) => TT, desc: string, base?: string) {
    super(h, kind, kw, P, base)
    this.shapeFn = shapeFn
    this.fwd = fwd
    this.desc = desc
  }
  build(h: Host, s: KShape) { return this.shapeFn(s, h) }
  describe() { return this.desc }
  note(i: KShape, o: KShape) { return this.kind === 'Flatten' ? `${i.slice(1).join(' × ')} = ${o[1]}` : null }
  forward(E: Engine, x: TT, training: boolean) { return this.fwd(E, x, training) }
}

const flattenShape = (s: KShape): KShape => {
  const rest = s.slice(1)
  return [s[0], !rest.length ? 1 : rest.some((d) => d === null) ? null : prod(rest as number[])]
}

/** the shape two inputs of a merge layer broadcast to, without the batch axis */
const mergeShape = (h: Host, a: KShape, b: KShape): KShape => {
  if (a.length < b.length) return mergeShape(h, b, a)
  const out = [...a]
  for (let i = 1; i <= b.length; i++) {
    const x = a[a.length - i]
    const y = b[b.length - i]
    if (x === null || y === null) out[a.length - i] = null
    else if (x === 1) out[a.length - i] = y
    else if (y !== 1 && x !== y) throw h.err('ValueError', `Inputs have incompatible shapes. Received shapes ${shapeStr(a)} and ${shapeStr(b)}`)
  }
  return out
}

/** Concatenate, Add, Subtract … : layers that take a list of tensors */
class Merge extends KLayer {
  f: (E: Engine, xs: TT[]) => TT
  shapes: ((h: Host, s: KShape[]) => KShape) | null
  single: boolean
  /** made by x + y on KerasTensors: no constructor to rebuild it from */
  isOp = false
  constructor(h: Host, kind: string, kw: Kw, P: Params, f: (E: Engine, xs: TT[]) => TT, single = false, shapes: ((h: Host, s: KShape[]) => KShape) | null = null) {
    super(h, kind, kw, P)
    this.f = f
    this.single = single
    this.shapes = shapes
  }
  build(_h: Host, s: KShape) { return s }
  forward(E: Engine, x: TT) { return this.f(E, [x]) }
  run(E: Engine, xs: TT[]) { return [this.f(E, xs)] }
  outShapes(h: Host, shapes: KShape[]): KShape[] {
    if (shapes.length < 2 && !this.single) throw h.err('ValueError', `A merge layer should be called on a list of at least 2 inputs. Received ${shapes.length} input${shapes.length === 1 ? '' : 's'}.`)
    const out = this.shapes ? this.shapes(h, shapes) : [shapes[0][0], ...shapes.map((s) => s.slice(1)).reduce((a, b) => mergeShape(h, a, b))]
    this.inShape = shapes[0]
    this.outShape = out
    this.built = true
    return [out]
  }
}

/** tf.keras batch_dot: contract the given axes of two batched tensors */
const batchDot = (E: Engine, a: TT, b: TT, ax: [number, number]): TT => {
  if (a.ndim === 2 && b.ndim === 2) return C.unsqueezeT(E, C.reduceSum(E, C.binary(E, '*', a, b), [1], false, false), -1)
  const moveLast = (t: TT, axis: number) => C.permuteT(E, t, [...t.shape.map((_, i) => i).filter((i) => i !== axis), axis])
  const moveFirst = (t: TT, axis: number) => C.permuteT(E, t, [0, axis, ...t.shape.map((_, i) => i).filter((i) => i !== 0 && i !== axis)])
  const x = a.ndim === 2 ? C.unsqueezeT(E, a, 1) : moveLast(a, ax[0])
  const y = b.ndim === 2 ? C.unsqueezeT(E, b, 2) : moveFirst(b, ax[1])
  const flatX = C.reshapeT(E, x, [x.shape[0], -1, x.shape[x.ndim - 1]], false)
  const flatY = C.reshapeT(E, y, [y.shape[0], y.shape[1], -1], false)
  const out = C.matmul(E, flatX, flatY)
  const shape = [x.shape[0], ...(a.ndim === 2 ? [] : x.shape.slice(1, -1)), ...(b.ndim === 2 ? [] : y.shape.slice(2))]
  return C.reshapeT(E, out, shape, false)
}

const l2Normalize = (E: Engine, x: TT, axis: number): TT => {
  const nrm = C.unary(E, 'sqrt', C.clamp(E, C.reduceSum(E, C.unary(E, 'square', x), [axis], true, false), 1e-12, null))
  return C.binary(E, '/', x, nrm)
}

class BatchNorm extends KLayer {
  axis: number
  dim = 0
  gamma: TT | null = null
  beta: TT | null = null
  mm: TT | null = null
  mv: TT | null = null
  constructor(h: Host, P: Params, kw: Kw) {
    super(h, 'BatchNormalization', kw, P)
    this.axis = h.toInt(P.axis)
  }
  build(h: Host, s: KShape) {
    const a = this.axis < 0 ? this.axis + s.length : this.axis
    if (a < 0 || a >= s.length || s[a] === null) throw h.err('ValueError', `Axis ${this.axis} of input tensor should have a defined dimension but the layer received an input with shape ${shapeStr(s)}.`)
    this.axis = a
    this.dim = s[a] as number
    if (h.truthy(this.P.scale)) this.gamma = this.addWeight(h, 'gamma', [this.dim], initArg(h, this.P.gamma_initializer, 'Ones'), { reg: regArg(h, this.P.gamma_regularizer) })
    if (h.truthy(this.P.center)) this.beta = this.addWeight(h, 'beta', [this.dim], initArg(h, this.P.beta_initializer, 'Zeros'), { reg: regArg(h, this.P.beta_regularizer) })
    this.mm = this.addWeight(h, 'moving_mean', [this.dim], initArg(h, this.P.moving_mean_initializer, 'Zeros'), { trainable: false })
    this.mv = this.addWeight(h, 'moving_variance', [this.dim], initArg(h, this.P.moving_variance_initializer, 'Ones'), { trainable: false })
    return s
  }
  outFor(_h: Host, s: KShape): KShape { return s }
  describe() { return 'BatchNormalization()' }
  note(i: KShape) { return `${i[this.axis]} × 4 (γ, β trainable; moving mean and variance not)` }
  forward(E: Engine, x: TT, training: boolean, h: Host) {
    const eps = h.num(this.P.epsilon)
    const shape = x.shape.map((d, i) => (i === this.axis ? d : 1))
    let mean: TT
    let variance: TT
    if (training && this.trainableFlag) {
      const axes = x.shape.map((_, i) => i).filter((i) => i !== this.axis)
      mean = C.reduceSum(E, x, axes, true, true)
      variance = C.reduceSum(E, C.unary(E, 'square', C.binary(E, '-', x, mean)), axes, true, true)
      const m = h.num(this.P.momentum)
      const mv = mean.values()
      const vv = variance.values()
      C.writeInto(this.mm!, this.mm!.values().map((o, i) => o * m + mv[i] * (1 - m)))
      C.writeInto(this.mv!, this.mv!.values().map((o, i) => o * m + vv[i] * (1 - m)))
    } else {
      mean = TT.of(this.mm!.values(), shape, 'float32')
      variance = TT.of(this.mv!.values(), shape, 'float32')
    }
    const inv = C.unary(E, 'sqrt', C.binary(E, '+', variance, scalarT(eps)))
    let y = C.binary(E, '/', C.binary(E, '-', x, mean), inv)
    if (this.gamma) y = C.binary(E, '*', y, C.reshapeT(E, this.gamma, shape, false))
    if (this.beta) y = C.binary(E, '+', y, C.reshapeT(E, this.beta, shape, false))
    return y
  }
}

class LayerNorm extends KLayer {
  axes: number[] = []
  gamma: TT | null = null
  beta: TT | null = null
  constructor(h: Host, P: Params, kw: Kw) {
    super(h, 'LayerNormalization', kw, P)
  }
  build(h: Host, s: KShape) {
    const raw = isSeq(this.P.axis) ? h.iterate(this.P.axis).map((v) => h.toInt(v)) : [h.toInt(this.P.axis)]
    this.axes = raw.map((a) => (a < 0 ? a + s.length : a))
    const shape = this.axes.map((a) => s[a] as number)
    if (h.truthy(this.P.scale)) this.gamma = this.addWeight(h, 'gamma', shape, initArg(h, this.P.gamma_initializer, 'Ones'), { reg: regArg(h, this.P.gamma_regularizer) })
    if (h.truthy(this.P.center)) this.beta = this.addWeight(h, 'beta', shape, initArg(h, this.P.beta_initializer, 'Zeros'), { reg: regArg(h, this.P.beta_regularizer) })
    return s
  }
  outFor(_h: Host, s: KShape): KShape { return s }
  describe() { return 'LayerNormalization()' }
  forward(E: Engine, x: TT, _t: boolean, h: Host) {
    const eps = h.num(this.P.epsilon)
    const mean = C.reduceSum(E, x, this.axes, true, true)
    const variance = C.reduceSum(E, C.unary(E, 'square', C.binary(E, '-', x, mean)), this.axes, true, true)
    let y = C.binary(E, '/', C.binary(E, '-', x, mean), C.unary(E, 'sqrt', C.binary(E, '+', variance, scalarT(eps))))
    const shape = x.shape.map((d, i) => (this.axes.includes(i) ? d : 1))
    if (this.gamma) y = C.binary(E, '*', y, C.reshapeT(E, this.gamma, shape, false))
    if (this.beta) y = C.binary(E, '+', y, C.reshapeT(E, this.beta, shape, false))
    return y
  }
}

class Embedding extends KLayer {
  inDim: number
  outDim: number
  w: TT | null = null
  constructor(h: Host, P: Params, kw: Kw) {
    super(h, 'Embedding', kw, P)
    this.inDim = h.toInt(P.input_dim)
    this.outDim = h.toInt(P.output_dim)
    if (this.inDim <= 0 || this.outDim <= 0) throw h.err('ValueError', `Both \`input_dim\` and \`output_dim\` should be positive, Received input_dim = ${this.inDim} and output_dim = ${this.outDim}`)
  }
  build(h: Host, s: KShape) {
    this.w = this.addWeight(h, 'embeddings', [this.inDim, this.outDim], initArg(h, this.P.embeddings_initializer, 'RandomUniform'), { reg: regArg(h, this.P.embeddings_regularizer) })
    return [...s, this.outDim]
  }
  fixedLastAxis() { return false }
  outFor(_h: Host, s: KShape): KShape { return [...s, this.outDim] }
  describe() { return `Embedding(${this.inDim}, ${this.outDim})` }
  note() { return `${this.inDim} × ${this.outDim} = ${this.params().toLocaleString('en-US')}` }
  forward(E: Engine, x: TT, _t: boolean, h: Host) {
    const idx = TT.of(x.values().map(Math.trunc), x.shape, 'int64')
    idx.values().forEach((c, n) => {
      if (c < 0 || c >= this.inDim) throw tfErr(h, `indices[${n}] = ${c} is not in [0, ${this.inDim}) [Op:ResourceGather]`)
    })
    return C.embeddingT(E, this.w!, idx, null)
  }
}

class PReLU extends KLayer {
  a: TT | null = null
  constructor(h: Host, P: Params, kw: Kw) {
    super(h, 'PReLU', kw, P)
  }
  build(h: Host, s: KShape) {
    const shared = isNone(this.P.shared_axes) ? [] : isSeq(this.P.shared_axes) ? h.iterate(this.P.shared_axes).map((v) => h.toInt(v)) : [h.toInt(this.P.shared_axes)]
    const shape = s.slice(1).map((d, i) => (shared.includes(i + 1) ? 1 : (d as number)))
    this.a = this.addWeight(h, 'alpha', shape, initArg(h, this.P.alpha_initializer, 'Zeros'), { reg: regArg(h, this.P.alpha_regularizer) })
    return s
  }
  outFor(_h: Host, s: KShape): KShape { return s }
  describe() { return 'PReLU()' }
  forward(E: Engine, x: TT) {
    const neg = C.binary(E, '*', this.a!, C.unary(E, 'relu', C.unary(E, 'neg', x)))
    return C.binary(E, '-', C.unary(E, 'relu', x), neg)
  }
}

/** a batch of zeros standing in for a symbolic tensor, to see what a user function returns */
const probeShapes = (h: Host, l: KLayer, shapes: KShape[], kw: Kw): KShape[] => {
  const S = stateOf(h)
  const was = S.E.grad
  const nLoss = S.losses.length
  S.E.grad = false
  const xs = shapes.map((s) => {
    const dims = s.map((d) => d ?? 2)
    return TT.of(new Array(prod(dims)).fill(0), dims, 'float32')
  })
  try {
    const outs = l.invoke(S.E, xs, false, h, kw)
    return outs.map((o) => o.shape.map((d, i) => (d === 2 && shapes[0][i] === null ? null : d)))
  } finally {
    S.E.grad = was
    S.losses.length = nLoss
  }
}

/** the tensors a user function returned */
const flatTensors = (h: Host, v: Value, what: string): TT[] => {
  if (isSeq(v)) return v.items.flatMap((x) => flatTensors(h, x, what))
  if (v.k === 'dict') return v.d.values().flatMap((x) => flatTensors(h, x, what))
  return [toTT(h, v, what)]
}

const acceptsArg = (f: FuncValue, name: string) => f.params.some((p) => p.name === name)

class Lambda extends KLayer {
  fn: Value
  constructor(h: Host, P: Params, kw: Kw) {
    super(h, 'Lambda', kw, P)
    this.fn = P.function
  }
  build(_h: Host, s: KShape) { return s }
  describe() { return 'Lambda(…)' }
  forward(E: Engine, x: TT, training: boolean, h: Host) { return this.run(E, [x], training, h, {})[0] }
  run(_E: Engine, xs: TT[], _t: boolean, h: Host, _kw: Kw): TT[] {
    const extra: Kw = {}
    if (this.P.arguments.k === 'dict') for (const [k, v] of this.P.arguments.d.items()) extra[h.str(k)] = v
    const arg = xs.length > 1 ? py.list(xs.map(tfWrap)) : tfWrap(xs[0])
    return flatTensors(h, h.call(this.fn, [arg], extra), 'a Lambda function')
  }
  eagerBuild(_h: Host, shapes: KShape[]) {
    this.built = true
    this.inShape = shapes[0]
  }
  outShapes(h: Host, shapes: KShape[], kw: Kw): KShape[] {
    if (!this.built || !this.inShape) {
      this.inShape = shapes[0]
      const os = this.P.output_shape
      this.outShape = !isNone(os) ? [shapes[0][0], ...h.iterate(os).map((v) => (v.k === 'none' ? null : h.toInt(v)))] : null
      this.built = true
    }
    if (this.outShape) return [this.outShape]
    const out = probeShapes(h, this, shapes, kw)
    this.outShape = out[0]
    return out
  }
}

/** a user's class Foo(keras.layers.Layer): build() and call() written in Python */
class InstLayer extends KLayer {
  constructor(h: Host, inst: InstValue, kw: Kw) {
    super(h, inst.cls.name, kw)
    this.inst = inst
  }
  children() { return attrLayers(this.inst!) }
  isa(name: string) { return name === 'Layer' }
  outFor() { return null }
  build(h: Host, s: KShape): KShape {
    userBuild(h, this.inst!, s)
    const cos = userMethod(this.inst!.cls, 'compute_output_shape')
    if (cos) {
      const v = h.call({ k: 'bound', self: this.inst!, f: cos }, [shapeTuple(s)])
      return h.iterate(v).map((d) => (d.k === 'none' ? null : h.toInt(d)))
    }
    this.built = true
    return probeShapes(h, this, [s], {})[0]
  }
  forward(E: Engine, x: TT, training: boolean, h: Host) { return this.run(E, [x], training, h, {})[0] }
  run(_E: Engine, xs: TT[], training: boolean, h: Host, kw: Kw): TT[] {
    return flatTensors(h, callUser(h, this.inst!, xs.length > 1 ? py.list(xs.map(tfWrap)) : tfWrap(xs[0]), training, kw), `${this.kind}.call()`)
  }
  outShapes(h: Host, shapes: KShape[], _kw: Kw): KShape[] { return [this.ensureBuilt(h, shapes[0])] }
  call(args: Value[], kw: Kw, h: Host): Value { return userEagerCall(this, h, args, kw) }
  repr() { return `<${this.kind} name=${this.name}, built=${this.built ? 'True' : 'False'}>` }
}

/** self.call(x, training=…), passing training only if the user's call() takes it */
const callUser = (h: Host, inst: InstValue, x: Value, training: boolean | null, kw: Kw, more: Value[] = []): Value => {
  const f = userMethod(inst.cls, 'call')
  if (!f) throw h.err('NotImplementedError', `Layer ${inst.cls.name} does not have a \`call()\` method implemented.`)
  const pass: Kw = { ...kw }
  if (training !== null && acceptsArg(f, 'training')) pass.training = py.bool(training)
  return h.call({ k: 'bound', self: inst, f }, [x, ...more], pass)
}

/** SimpleRNN, LSTM, GRU: the cells of keras.src.layers.rnn, one time step at a time */
class RNN extends KLayer {
  units: number
  cell: 'simple' | 'lstm' | 'gru'
  act: ActSpec
  recAct: ActSpec
  useBias: boolean
  resetAfter: boolean
  returnSeq: boolean
  returnState: boolean
  goBack: boolean
  w: TT | null = null
  r: TT | null = null
  b: TT | null = null
  constructor(h: Host, kind: 'SimpleRNN' | 'LSTM' | 'GRU', P: Params, kw: Kw) {
    super(h, kind, kw, P)
    this.cell = kind === 'SimpleRNN' ? 'simple' : kind === 'LSTM' ? 'lstm' : 'gru'
    this.units = h.toInt(P.units)
    if (this.units < 1) throw h.err('ValueError', `Received an invalid value for argument \`units\`, expected a positive integer, got ${this.units}.`)
    this.act = actArg(h, P.activation)
    this.recAct = this.cell === 'simple' ? this.act : actArg(h, P.recurrent_activation)
    this.useBias = h.truthy(P.use_bias)
    this.resetAfter = this.cell === 'gru' && h.truthy(P.reset_after)
    this.returnSeq = h.truthy(P.return_sequences)
    this.returnState = h.truthy(P.return_state)
    this.goBack = h.truthy(P.go_backwards)
  }
  gates() { return this.cell === 'simple' ? 1 : this.cell === 'lstm' ? 4 : 3 }
  nStates() { return this.cell === 'lstm' ? 2 : 1 }
  build(h: Host, s: KShape): KShape {
    if (s.length !== 3) throw h.err('ValueError', `Input 0 of layer "${this.name}" is incompatible with the layer: expected ndim=3, found ndim=${s.length}. Full shape received: ${shapeStr(s)}`)
    const F = s[2]
    if (F === null) throw h.err('ValueError', 'The last dimension of the inputs to a RNN layer should be defined. Found None.')
    const n = this.units * this.gates()
    this.w = this.addWeight(h, 'kernel', [F, n], initArg(h, this.P.kernel_initializer, 'GlorotUniform'), { reg: regArg(h, this.P.kernel_regularizer) })
    this.r = this.addWeight(h, 'recurrent_kernel', [this.units, n], initArg(h, this.P.recurrent_initializer, 'Orthogonal'), { reg: regArg(h, this.P.recurrent_regularizer) })
    if (this.useBias) {
      const size = this.resetAfter ? 2 * n : n
      this.b = this.addWeight(h, 'bias', [size], initArg(h, this.P.bias_initializer, 'Zeros'), { reg: regArg(h, this.P.bias_regularizer) })
      if (this.cell === 'lstm' && h.truthy(this.P.unit_forget_bias)) {
        const v = this.b.values()
        C.writeInto(this.b, v.map((x, i) => (i >= this.units && i < 2 * this.units ? 1 : x)))
      }
    }
    return this.outFor(h, s)
  }
  outFor(_h: Host, s: KShape): KShape { return this.returnSeq ? [s[0], s[1], this.units] : [s[0], this.units] }
  outShapes(h: Host, shapes: KShape[], _kw: Kw): KShape[] {
    const main = this.ensureBuilt(h, shapes[0])
    return this.returnState ? [main, ...allOf(this.nStates()).map(() => [shapes[0][0], this.units])] : [main]
  }
  describe() { return `${this.kind}(${this.units}${this.returnSeq ? ', return_sequences=True' : ''})` }
  note(i: KShape) {
    const F = i[2]
    const u = this.units
    if (this.cell === 'gru') return `3 × ${u} × (${F} + ${u} + ${this.resetAfter ? 2 : 1}) = ${this.params().toLocaleString('en-US')}`
    return `${this.gates()} × ${u} × (${F} + ${u} + 1) = ${this.params().toLocaleString('en-US')}`
  }
  forward(E: Engine, x: TT, training: boolean, h: Host) { return this.run(E, [x], training, h, {})[0] }
  run(E: Engine, xs: TT[], training: boolean, h: Host, _kw: Kw): TT[] {
    const x = xs[0]
    if (x.ndim !== 3) throw h.err('ValueError', `Input 0 of layer "${this.name}" is incompatible with the layer: expected ndim=3, found ndim=${x.ndim}. Full shape received: ${shapeStr([null, ...x.shape.slice(1)])}`)
    const [N, T] = x.shape
    const U = this.units
    const zero = () => TT.of(new Array(N * U).fill(0), [N, U], 'float32')
    let hState = xs[1] ?? zero()
    let cState = xs[2] ?? zero()
    const rate = (k: string) => (training && this.P[k] ? h.num(this.P[k]) : 0)
    const mask = (shape: number[], p: number) => (p > 0 ? C.dropout(E, TT.of(new Array(prod(shape)).fill(1), shape, 'float32'), p, true) : null)
    const inMask = mask([N, x.shape[2]], rate('dropout'))
    const recMask = mask([N, U], rate('recurrent_dropout'))
    const seq = this.goBack ? allOf(T).reverse() : allOf(T)
    const steps = unstackT(E, x, 1)
    const outs: TT[] = []
    const add = (a: TT, b: TT) => C.binary(E, '+', a, b)
    const mul = (a: TT, b: TT) => C.binary(E, '*', a, b)
    const part = (z: TT, i: number) => sliceAxis(E, z, 1, i * U, (i + 1) * U)
    const one = scalarT(1)
    for (const t of seq) {
      const xt = inMask ? mul(steps[t], inMask) : steps[t]
      const hPrev = recMask ? mul(hState, recMask) : hState
      if (this.cell === 'simple') {
        let z = add(C.matmul(E, xt, this.w!), C.matmul(E, hPrev, this.r!))
        if (this.b) z = add(z, this.b)
        hState = this.act.f(E, z)
        outs.push(hState)
      } else if (this.cell === 'lstm') {
        let z = add(C.matmul(E, xt, this.w!), C.matmul(E, hPrev, this.r!))
        if (this.b) z = add(z, this.b)
        const i = this.recAct.f(E, part(z, 0))
        const f = this.recAct.f(E, part(z, 1))
        cState = add(mul(f, cState), mul(i, this.act.f(E, part(z, 2))))
        const o = this.recAct.f(E, part(z, 3))
        hState = mul(o, this.act.f(E, cState))
        outs.push(hState)
      } else {
        let mx = C.matmul(E, xt, this.w!)
        if (this.b) mx = add(mx, this.resetAfter ? sliceAxis(E, this.b, 0, 0, 3 * U) : this.b)
        let inner = C.matmul(E, hPrev, this.r!)
        if (this.resetAfter && this.b) inner = add(inner, sliceAxis(E, this.b, 0, 3 * U, 6 * U))
        const z = this.recAct.f(E, add(part(mx, 0), part(inner, 0)))
        const r = this.recAct.f(E, add(part(mx, 1), part(inner, 1)))
        // reset_after applies the reset gate to the recurrent term, not to the state before it
        const hh = this.resetAfter
          ? this.act.f(E, add(part(mx, 2), mul(r, part(inner, 2))))
          : this.act.f(E, add(part(mx, 2), part(C.matmul(E, mul(r, hPrev), this.r!), 2)))
        hState = add(mul(z, hPrev), mul(C.binary(E, '-', one, z), hh))
        outs.push(hState)
      }
    }
    const main = this.returnSeq ? C.catT(E, outs, 1, true) : outs[outs.length - 1]
    return this.returnState ? [main, ...(this.cell === 'lstm' ? [hState, cState] : [hState])] : [main]
  }
}

class Bidirectional extends KLayer {
  fw: KLayer
  bw: KLayer
  mode: string | null
  constructor(h: Host, P: Params, kw: Kw) {
    super(h, 'Bidirectional', kw, P)
    const inner = layerOf(P.layer)
    if (!(inner instanceof RNN)) throw h.err('ValueError', `Please initialize \`Bidirectional\` layer with a \`keras.layers.Layer\` instance. Received: ${repr(P.layer)}`)
    if (inner.returnState) throw h.err('NotImplementedError', 'Bidirectional with return_state=True is not implemented in this sandbox')
    this.mode = P.merge_mode.k === 'none' ? null : h.str(P.merge_mode)
    if (this.mode && !['concat', 'sum', 'ave', 'mul'].includes(this.mode)) throw h.err('ValueError', `Invalid merge mode. Merge mode should be one of {"sum", "mul", "ave", "concat", None}. Received: merge_mode=${this.mode}`)
    this.fw = inner
    this.fw.name = `forward_${inner.name}`
    this.bw = stateOf(h).make(inner.kind, { ...inner.P, name: py.str(`backward_${inner.name}`), go_backwards: py.bool(!inner.goBack) })
  }
  children() { return [this.fw, this.bw] }
  merged(s: KShape): KShape {
    return this.mode === 'concat' ? [...s.slice(0, -1), (s[s.length - 1] as number) * 2] : s
  }
  build(h: Host, s: KShape): KShape {
    const f = this.fw.ensureBuilt(h, s)
    this.bw.ensureBuilt(h, s)
    return this.merged(f)
  }
  outShapes(h: Host, shapes: KShape[], _kw: Kw): KShape[] {
    const out = this.ensureBuilt(h, shapes[0])
    return this.mode ? [out] : [out, out]
  }
  describe() { return `Bidirectional(${this.fw.describe()})` }
  note() { return `2 × ${(this.params() / 2).toLocaleString('en-US')} = ${this.params().toLocaleString('en-US')}` }
  forward(E: Engine, x: TT, training: boolean, h: Host) { return this.run(E, [x], training, h, {})[0] }
  run(E: Engine, xs: TT[], training: boolean, h: Host, kw: Kw): TT[] {
    const a = this.fw.run(E, [xs[0]], training, h, kw)[0]
    let b = this.bw.run(E, [xs[0]], training, h, kw)[0]
    if ((this.fw as RNN).returnSeq) b = flipAxes(E, b, [1])
    switch (this.mode) {
      case 'concat': return [C.catT(E, [a, b], -1)]
      case 'sum': return [C.binary(E, '+', a, b)]
      case 'ave': return [mulS(E, C.binary(E, '+', a, b), 0.5)]
      case 'mul': return [C.binary(E, '*', a, b)]
    }
    return [a, b]
  }
}

class TimeDistributed extends KLayer {
  inner: KLayer
  constructor(h: Host, P: Params, kw: Kw) {
    super(h, 'TimeDistributed', kw, P)
    const l = layerOf(P.layer)
    if (!l) throw h.err('ValueError', `Please initialize \`TimeDistributed\` layer with a \`keras.layers.Layer\` instance. Received: ${repr(P.layer)}`)
    this.inner = l
  }
  children() { return [this.inner] }
  build(h: Host, s: KShape): KShape {
    if (s.length < 3) throw h.err('ValueError', `\`TimeDistributed\` Layer should be passed an \`input_shape\` with at least 3 dimensions, received: ${shapeStr(s)}`)
    const o = this.inner.ensureBuilt(h, [s[0], ...s.slice(2)])
    return [s[0], s[1], ...o.slice(1)]
  }
  describe() { return `TimeDistributed(${this.inner.describe()})` }
  forward(E: Engine, x: TT, training: boolean, h: Host) {
    const [N, T] = x.shape
    const flat = C.reshapeT(E, x, [N * T, ...x.shape.slice(2)], false)
    const y = this.inner.invoke(E, [flat], training, h)[0]
    return C.reshapeT(E, y, [N, T, ...y.shape.slice(1)], false)
  }
}

class Normalization extends KLayer {
  axes: number[] | null
  mean: number[] | null = null
  variance: number[] | null = null
  adapted = false
  given: boolean
  keep: number[] = []
  bshape: number[] = []
  constructor(h: Host, P: Params, kw: Kw) {
    super(h, 'Normalization', kw, P)
    this.axes = P.axis.k === 'none' ? [] : isSeq(P.axis) ? h.iterate(P.axis).map((v) => h.toInt(v)) : [h.toInt(P.axis)]
    this.given = !isNone(P.mean)
    if (this.given !== !isNone(P.variance)) throw h.err('ValueError', `When setting values directly, both \`mean\` and \`variance\` must be set. Received: mean=${repr(P.mean)} and variance=${repr(P.variance)}`)
  }
  build(h: Host, s: KShape) {
    const nd = s.length
    this.keep = this.axes!.map((a) => (a < 0 ? a + nd : a)).sort((a, b) => a - b)
    for (const a of this.keep) if (a < 0 || a >= nd || s[a] === null) throw h.err('ValueError', `All \`axis\` values to be kept must have a known shape. Received axis=${repr(this.P.axis)}, inputs.shape=${shapeStr(s)}, with unknown axis at index ${a}`)
    this.bshape = s.map((d, i) => (this.keep.includes(i) ? (d as number) : 1))
    const shape = this.keep.map((a) => s[a] as number)
    if (this.given) {
      const size = prod(shape)
      const grow = (a: number[]) => (a.length === 1 ? new Array<number>(size).fill(a[0]) : a)
      this.mean = grow(toArray(this.P.mean).values())
      this.variance = grow(toArray(this.P.variance).values())
    } else {
      this.addWeight(h, 'mean', shape, defaultInit(h, 'Zeros'), { trainable: false })
      this.addWeight(h, 'variance', shape, defaultInit(h, 'Ones'), { trainable: false })
      this.addWeight(h, 'count', [], defaultInit(h, 'Zeros'), { trainable: false, bytes: 8, dtype: 'int64' })
    }
    return s
  }
  /** adapt() builds it on the data's shape: later calls may come with another batch size or rank */
  ensureBuilt(h: Host, s: KShape): KShape {
    if (!this.built) super.ensureBuilt(h, s)
    this.inShape = s
    this.outShape = s
    return s
  }
  stats(): [number[], number[]] { return this.given ? [this.mean!, this.variance!] : [this.ws[0].t.values(), this.ws[1].t.values()] }
  describe() { return 'Normalization()' }
  note(i: KShape) { return this.given ? null : `mean ${i[i.length - 1]} + variance ${i[i.length - 1]} + count 1, all non-trainable` }
  adapt(h: Host, data: Value) {
    const parts = data.k === 'obj' && data.o instanceof TFDataset ? data.o.batches(h).map((b) => asFloat(h, b)) : [asFloat(h, data)]
    const x = parts.length === 1 ? parts[0] : C.catT({ grad: false } as Engine, parts, 0)
    const shape: KShape = [null, ...x.shape.slice(1)]
    if (!this.built) this.ensureBuilt(h, x.ndim === 1 ? [x.shape[0]] : shape)
    const axes = x.shape.map((_, i) => i).filter((i) => !this.keep.includes(i))
    const E = new Engine(h.rng)
    E.grad = false
    const f64 = TT.of(x.values(), x.shape, 'float64')
    const mean = C.reduceSum(E, f64, axes, true, true)
    const variance = C.reduceSum(E, C.unary(E, 'square', C.binary(E, '-', f64, mean)), axes, true, true)
    if (this.given) {
      this.mean = mean.values()
      this.variance = variance.values()
    } else {
      C.writeInto(this.ws[0].t, mean.values())
      C.writeInto(this.ws[1].t, variance.values())
      C.writeInto(this.ws[2].t, [x.shape[0]])
    }
    this.adapted = true
  }
  forward(E: Engine, x: TT, _t: boolean, h: Host) {
    const invert = h.truthy(this.P.invert)
    const bs = this.bshape.length === x.ndim ? this.bshape : x.shape.map((d, i) => (this.keep.includes(i) ? d : 1))
    const [mv, vv] = this.stats()
    const m = TT.of(mv, bs, 'float32')
    const sd = TT.of(vv.map((v) => Math.max(Math.sqrt(Math.fround(v)), 1e-7)), bs, 'float32')
    return invert ? C.binary(E, '+', m, C.binary(E, '*', x, sd)) : C.binary(E, '/', C.binary(E, '-', x, m), sd)
  }
  getAttr(name: string, h: Host): Value | undefined {
    if ((name === 'mean' || name === 'variance') && this.built) return tfWrap(TT.of(this.stats()[name === 'mean' ? 0 : 1], this.bshape, 'float32'))
    return super.getAttr(name, h)
  }
  attr(name: string, h: Host): Value | undefined {
    if (name === 'adapt') return fnv(name, (args, kw) => { this.adapt(h, args[0] ?? kw.data); return py.NONE })
    if (name === 'is_adapted') return py.bool(this.adapted)
    return super.attr(name, h)
  }
}

// keras.layers.TextVectorization's default strip set
const STRIP = /[!"#$%&()*+,\-./:;<=>?@[\\\]^_`{|}~']/g

class TextVectorization extends KLayer {
  /** index_lookup._oov_start_index: only 'int' output reserves index 0 for the mask */
  specials: string[]
  vocab: string[]
  index = new Map<string, number>()
  maxTokens: number | null
  seqLen: number | null
  mode: string
  constructor(h: Host, P: Params, kw: Kw) {
    super(h, 'TextVectorization', kw, P)
    this.maxTokens = isNone(P.max_tokens) ? null : h.toInt(P.max_tokens)
    this.seqLen = isNone(P.output_sequence_length) ? null : h.toInt(P.output_sequence_length)
    this.mode = h.str(P.output_mode)
    if (!['int', 'multi_hot', 'count'].includes(this.mode)) throw h.err('NotImplementedError', `TextVectorization: output_mode='${this.mode}' is not implemented in this sandbox`)
    if (this.seqLen !== null && this.mode !== 'int') throw h.err('ValueError', "`output_sequence_length` must not be set if `output_mode` is not 'int'. ")
    this.specials = this.mode === 'int' ? ['', '[UNK]'] : ['[UNK]']
    this.vocab = [...this.specials]
    if (!isNone(P.vocabulary)) this.setVocab(h.iterate(P.vocabulary).map((v) => h.str(v)))
  }
  setVocab(tokens: string[]) {
    this.vocab = [...this.specials, ...tokens.filter((t) => !this.specials.includes(t))]
    this.index = new Map(this.vocab.map((t, i) => [t, i]))
  }
  tokens(s: string, h: Host): string[] {
    const std = this.P.standardize
    let t = s
    const mode = std.k === 'none' ? '' : h.str(std)
    if (mode === 'lower' || mode === 'lower_and_strip_punctuation') t = t.toLowerCase()
    if (mode === 'strip_punctuation' || mode === 'lower_and_strip_punctuation') t = t.replace(STRIP, '')
    const split = this.P.split
    if (split.k === 'none') return [t]
    return h.str(split) === 'character' ? [...t] : t.split(/\s+/).filter(Boolean)
  }
  adapt(h: Host, data: Value) {
    const strs = data.k === 'obj' && data.o instanceof TFDataset ? data.o.batches(h).flatMap((b) => strsOf(b) ?? []) : asInput(h, data)
    const list = Array.isArray(strs) ? strs : strs instanceof StrT ? strs.strs : null
    if (!list) throw h.err('ValueError', 'TextVectorization.adapt() expects a list of strings')
    const counts = new Map<string, number>()
    for (const s of list) for (const t of this.tokens(s, h)) counts.set(t, (counts.get(t) ?? 0) + 1)
    // count, then token, both descending: np.lexsort(...)[::-1] in keras.src.layers.preprocessing.index_lookup
    const sorted = [...counts].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? 1 : a[0] > b[0] ? -1 : 0)).map(([t]) => t)
    this.setVocab(this.maxTokens === null ? sorted : sorted.slice(0, Math.max(0, this.maxTokens - this.specials.length)))
  }
  build(_h: Host, s: KShape): KShape {
    if (this.mode !== 'int') return [s[0], this.maxTokens ?? this.vocab.length]
    return [s[0], this.seqLen]
  }
  outputDtype() { return 'int64' }
  describe() { return 'TextVectorization()' }
  forward(_E: Engine, x: TT, _t: boolean, h: Host) {
    if (!(x instanceof StrT)) throw h.err('ValueError', 'TextVectorization expects strings as input')
    const oov = this.specials.length - 1
    const rows = x.strs.map((s) => this.tokens(s, h).map((t) => this.index.get(t) ?? oov))
    if (this.mode === 'int') {
      const L = this.seqLen ?? rows.reduce((m, r) => Math.max(m, r.length), 0)
      const out: number[] = []
      for (const r of rows) for (let i = 0; i < L; i++) out.push(r[i] ?? 0)
      return TT.of(out, [rows.length, L], 'int64')
    }
    const V = this.maxTokens ?? this.vocab.length
    const out = new Array<number>(rows.length * V).fill(0)
    rows.forEach((r, n) => { for (const c of r) if (c < V) out[n * V + c] = this.mode === 'count' ? out[n * V + c] + 1 : 1 })
    return TT.of(out, [rows.length, V], 'float32')
  }
  attr(name: string, h: Host): Value | undefined {
    if (name === 'adapt') return fnv(name, (args, kw) => { this.adapt(h, args[0] ?? kw.data); return py.NONE })
    if (name === 'get_vocabulary') return fnv(name, (args, kw) => {
      const inc = kw.include_special_tokens ?? args[0]
      return py.list((inc && !h.truthy(inc) ? this.vocab.slice(this.specials.length) : this.vocab).map(py.str))
    })
    if (name === 'vocabulary_size') return fnv(name, () => py.int(this.vocab.length))
    if (name === 'set_vocabulary') return fnv(name, (args) => { this.setVocab(h.iterate(args[0]).map((v) => h.str(v))); return py.NONE })
    return super.attr(name, h)
  }
}

// ---------------------------------------------------------------- images: resizing and the random layers

/** the bilinear source of an output coordinate with half-pixel centres: (i + 0.5) · in / out − 0.5, clamped */
const bilinearAxis = (nIn: number, nOut: number) => Array.from({ length: nOut }, (_, i) => {
  const c = ((i + 0.5) * nIn) / nOut - 0.5
  const lo = Math.floor(c)
  return { a: Math.max(lo, 0), b: Math.min(lo + 1, nIn - 1), f: c - lo }
})

/** images (…, H, W, C) to a new height and width */
const resizeT = (E: Engine, x: TT, Ho: number, Wo: number, method: 'nearest' | 'bilinear'): TT => {
  const H = x.shape[x.ndim - 3]
  const W = x.shape[x.ndim - 2]
  const lead = x.shape.slice(0, -3)
  const rest = (m: number[]): number[][] => [...lead.map(allOf), m]
  const Cc = x.shape[x.ndim - 1]
  if (method === 'nearest') {
    const near = (nIn: number, nOut: number) => Array.from({ length: nOut }, (_, i) => Math.min(Math.floor(((i + 0.5) * nIn) / nOut), nIn - 1))
    return gridT(E, x, [...rest(near(H, Ho)), near(W, Wo), allOf(Cc)])
  }
  const f = isFloatT(x.dtype) ? x : TT.of(x.values(), x.shape, 'float32')
  const ya = bilinearAxis(H, Ho)
  const xa = bilinearAxis(W, Wo)
  const rows = C.linearAxis(E, f, x.ndim - 3, ya.map((p) => p.a), ya.map((p) => p.b), ya.map((p) => p.f))
  return C.linearAxis(E, rows, x.ndim - 2, xa.map((p) => p.a), xa.map((p) => p.b), xa.map((p) => p.f))
}

/** tf.image.resize's crop_to_aspect_ratio: the largest centred window of the target's aspect ratio */
const centerCropAspect = (E: Engine, x: TT, Ho: number, Wo: number): TT => {
  const H = x.shape[x.ndim - 3]
  const W = x.shape[x.ndim - 2]
  const ch = Math.max(Math.min(H, Math.trunc(Math.fround(W * Ho) / Wo)), 1)
  const cw = Math.max(Math.min(W, Math.trunc(Math.fround(H * Wo) / Ho)), 1)
  const top = Math.trunc((H - ch) / 2)
  const left = Math.trunc((W - cw) / 2)
  const nd = x.ndim
  return gridT(E, x, x.shape.map((n, d) => (d === nd - 3 ? Array.from({ length: ch }, (_, i) => top + i) : d === nd - 2 ? Array.from({ length: cw }, (_, i) => left + i) : allOf(n))))
}

const fillCoord = (c: number, n: number, mode: string): number => {
  switch (mode) {
    case 'nearest': return Math.min(Math.max(c, 0), n - 1)
    case 'wrap': return ((c % n) + n) % n
    case 'reflect': {
      // d c b a | a b c d | d c b a
      const p = 2 * n
      const m = ((c % p) + p) % p
      return m < n ? m : p - 1 - m
    }
    case 'mirror': {
      if (n === 1) return 0
      const p = 2 * (n - 1)
      const m = ((c % p) + p) % p
      return m < n ? m : p - m
    }
  }
  return c < 0 || c >= n ? -1 : c
}

/** one image (H, W, C), resampled through a map from output to input coordinates */
const resample = (v: number[], H: number, W: number, Cc: number, at: (y: number, x: number) => [number, number], mode: string, fill: number, bilinear: boolean): number[] => {
  const out = new Array<number>(H * W * Cc)
  const read = (y: number, x: number, c: number) => {
    const yy = fillCoord(y, H, mode)
    const xx = fillCoord(x, W, mode)
    return yy < 0 || xx < 0 ? fill : v[(yy * W + xx) * Cc + c]
  }
  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      const [y, x] = at(i, j)
      for (let c = 0; c < Cc; c++) {
        const o = (i * W + j) * Cc + c
        if (!bilinear) out[o] = read(Math.round(y), Math.round(x), c)
        else {
          const y0 = Math.floor(y)
          const x0 = Math.floor(x)
          const fy = y - y0
          const fx = x - x0
          out[o] = (1 - fy) * ((1 - fx) * read(y0, x0, c) + fx * read(y0, x0 + 1, c)) + fy * ((1 - fx) * read(y0 + 1, x0, c) + fx * read(y0 + 1, x0 + 1, c))
        }
      }
    }
  }
  return out
}

/** run a per-image transform over a batch of images or a single image */
const perImage = (x: TT, f: (img: number[], H: number, W: number, Cc: number) => number[]): TT => {
  const v = x.values()
  const H = x.shape[x.ndim - 3]
  const W = x.shape[x.ndim - 2]
  const Cc = x.shape[x.ndim - 1]
  const size = H * W * Cc
  const N = v.length / size
  const out: number[] = []
  for (let n = 0; n < N; n++) for (const p of f(v.slice(n * size, (n + 1) * size), H, W, Cc)) out.push(p)
  return TT.of(out, x.shape, x.dtype)
}

/** a factor for the random layers: f → [−f, f], (a, b) → [a, b] */
const factorRange = (h: Host, v: Value): [number, number] => {
  if (isSeq(v)) return [h.num(v.items[0]), h.num(v.items[1])]
  const f = h.num(v)
  return [-Math.abs(f), Math.abs(f)]
}

// ---------------------------------------------------------------- losses

interface LossSpec {
  name: string
  fromLogits: boolean
  delta: number
  smoothing: number
  /** a Python function or an instance of a user's keras.losses.Loss */
  fn: Value | null
  reduction: ReductionK
}

const LOSS_ALIASES: Record<string, string> = {
  categorical_crossentropy: 'categorical_crossentropy', CategoricalCrossentropy: 'categorical_crossentropy',
  sparse_categorical_crossentropy: 'sparse_categorical_crossentropy', SparseCategoricalCrossentropy: 'sparse_categorical_crossentropy',
  binary_crossentropy: 'binary_crossentropy', BinaryCrossentropy: 'binary_crossentropy',
  mse: 'mean_squared_error', MSE: 'mean_squared_error', mean_squared_error: 'mean_squared_error', MeanSquaredError: 'mean_squared_error',
  mae: 'mean_absolute_error', MAE: 'mean_absolute_error', mean_absolute_error: 'mean_absolute_error', MeanAbsoluteError: 'mean_absolute_error',
  mape: 'mean_absolute_percentage_error', MAPE: 'mean_absolute_percentage_error', mean_absolute_percentage_error: 'mean_absolute_percentage_error', MeanAbsolutePercentageError: 'mean_absolute_percentage_error',
  msle: 'mean_squared_logarithmic_error', MSLE: 'mean_squared_logarithmic_error', mean_squared_logarithmic_error: 'mean_squared_logarithmic_error', MeanSquaredLogarithmicError: 'mean_squared_logarithmic_error',
  huber: 'huber', Huber: 'huber', hinge: 'hinge', Hinge: 'hinge', squared_hinge: 'squared_hinge', SquaredHinge: 'squared_hinge', categorical_hinge: 'categorical_hinge', CategoricalHinge: 'categorical_hinge',
  kl_divergence: 'kl_divergence', kld: 'kl_divergence', KLD: 'kl_divergence', KLDivergence: 'kl_divergence', poisson: 'poisson', Poisson: 'poisson',
  log_cosh: 'log_cosh', logcosh: 'log_cosh', LogCosh: 'log_cosh', cosine_similarity: 'cosine_similarity', CosineSimilarity: 'cosine_similarity',
}

const shapeMismatch = (y: TT, p: TT) => new PyError('ValueError', `Arguments \`target\` and \`output\` must have the same shape. Received: target.shape=${shapeStr([null, ...y.shape.slice(1)])}, output.shape=${shapeStr([null, ...p.shape.slice(1)])}`)

/** keras.src.losses.loss.squeeze_or_expand_to_same_rank */
const sameRank = (E: Engine, y: TT, p: TT): TT => (y.ndim === p.ndim - 1 && p.shape[p.ndim - 1] === 1 ? C.unsqueezeT(E, y, -1) : y.ndim === p.ndim + 1 && y.shape[y.ndim - 1] === 1 ? C.squeezeT(E, y, -1) : y)

const meanLast = (E: Engine, x: TT) => (x.ndim ? C.reduceSum(E, x, [-1], false, true) : x)
const sumLast = (E: Engine, x: TT) => (x.ndim ? C.reduceSum(E, x, [-1], false, false) : x)
const EPS = 1e-7
const clipLog = (E: Engine, p: TT) => C.unary(E, 'log', C.clamp(E, p, EPS, 1 - EPS))
/** Keras rescales "probabilities" that do not sum to 1 before taking the log */
const normalise = (E: Engine, p: TT) => C.binary(E, '/', p, C.reduceSum(E, p, [-1], true, false))
const sub1 = (E: Engine, x: TT) => C.binary(E, '-', scalarT(1), x, true, false)

/** the loss of each sample (the last axis is already reduced), from the definitions in keras.src.losses.losses */
function perSample(E: Engine, h: Host, spec: LossSpec, yTrue: TT, yPred: TT, train: boolean): TT {
  if (spec.fn) return toTT(h, h.call(spec.fn, [tfWrap(yTrue), tfWrap(yPred)]), 'a loss function')
  const add = (a: TT, b: TT) => C.binary(E, '+', a, b)
  const mul = (a: TT, b: TT) => C.binary(E, '*', a, b)
  switch (spec.name) {
    case 'sparse_categorical_crossentropy': {
      const C2 = yPred.shape[yPred.ndim - 1]
      const labels = yTrue.values().map(Math.round)
      if (labels.length !== yPred.size / C2) throw new PyError('ValueError', `Arguments \`target\` and \`output\` must have the same rank (ndim) or the target must have one dimension less. Received: target.shape=${shapeStr([null, ...yTrue.shape.slice(1)])}, output.shape=${shapeStr([null, ...yPred.shape.slice(1)])}`)
      for (const l of labels) if (l < 0 || l >= C2) throw tfErr(h, `Received a label value of ${l} which is outside the valid range of [0, ${C2}). (One-hot targets need loss='categorical_crossentropy'.)`)
      const target = TT.of(labels, [labels.length], 'int64')
      const logp = spec.fromLogits ? C.softmax(E, yPred, -1, true) : clipLog(E, normalise(E, yPred))
      const per = C.nllLoss(E, C.reshapeT(E, logp, [labels.length, C2], false), target, 'none')
      return yPred.ndim > 2 ? C.reshapeT(E, per, yPred.shape.slice(0, -1), false) : per
    }
    case 'categorical_crossentropy': {
      if (yTrue.shape.join() !== yPred.shape.join()) {
        if (train) noteOnce(h, 'keras_sparse_hint', 'info')
        throw shapeMismatch(yTrue, yPred)
      }
      let y = yTrue
      if (spec.smoothing) {
        const n = yPred.shape[yPred.ndim - 1]
        y = add(mulS(E, y, 1 - spec.smoothing), scalarT(spec.smoothing / n))
      }
      const logp = spec.fromLogits ? C.softmax(E, yPred, -1, true) : clipLog(E, normalise(E, yPred))
      return C.reduceSum(E, mul(C.unary(E, 'neg', y), logp), [-1], false, false)
    }
    case 'binary_crossentropy': {
      let y = sameRank(E, yTrue, yPred)
      if (y.shape.join() !== yPred.shape.join()) throw shapeMismatch(yTrue, yPred)
      if (spec.smoothing) y = add(mulS(E, y, 1 - spec.smoothing), scalarT(0.5 * spec.smoothing))
      if (spec.fromLogits) return meanLast(E, C.bce(E, yPred, y, true, 'none'))
      const p = C.clamp(E, yPred, EPS, 1 - EPS)
      const t1 = mul(y, C.unary(E, 'log', p))
      const t2 = mul(sub1(E, y), C.unary(E, 'log', sub1(E, p)))
      return C.unary(E, 'neg', meanLast(E, add(t1, t2)))
    }
  }
  const y = sameRank(E, yTrue, yPred)
  for (let i = 1; i <= Math.min(y.ndim, yPred.ndim); i++) {
    const [a, b] = [yPred.shape[yPred.ndim - i], y.shape[y.ndim - i]]
    if (a !== b && a !== 1 && b !== 1) throw h.err('ValueError', `Dimensions must be equal, but are ${a} and ${b} for 'sub' with input shapes: [?,${yPred.shape.slice(1).join(',')}], [?${y.shape.slice(1).map((v) => `,${v}`).join('')}].`)
  }
  const d = C.binary(E, '-', yPred, y)
  switch (spec.name) {
    case 'mean_squared_error': return meanLast(E, C.unary(E, 'square', d))
    case 'mean_absolute_error': return meanLast(E, C.unary(E, 'abs', d))
    case 'mean_absolute_percentage_error': {
      const denom = TT.of(y.values().map((v) => Math.max(Math.abs(v), EPS)), y.shape, y.dtype)
      return mulS(E, meanLast(E, C.unary(E, 'abs', C.binary(E, '/', d, denom))), 100)
    }
    case 'mean_squared_logarithmic_error': {
      const lg = (x: TT) => C.unary(E, 'log', add(C.clamp(E, x, EPS, null), scalarT(1)))
      return meanLast(E, C.unary(E, 'square', C.binary(E, '-', lg(yPred), lg(y))))
    }
    case 'huber': {
      const dl = spec.delta
      return meanLast(E, C.mapT(E, d, (v) => (Math.abs(v) <= dl ? 0.5 * v * v : dl * Math.abs(v) - 0.5 * dl * dl), (v) => (Math.abs(v) <= dl ? v : dl * Math.sign(v)), 'HuberBackward'))
    }
    case 'log_cosh': return meanLast(E, C.mapT(E, d, (v) => v + softplus(-2 * v) - Math.LN2, (v) => Math.tanh(v), 'LogCoshBackward'))
    case 'hinge': case 'squared_hinge': {
      // labels in {0, 1} are converted to {−1, 1}
      const vals = y.values()
      const signed = vals.every((v) => v === 0 || v === 1) ? TT.of(vals.map((v) => 2 * v - 1), y.shape, y.dtype) : y
      const margin = C.unary(E, 'relu', sub1(E, mul(signed, yPred)))
      return meanLast(E, spec.name === 'hinge' ? margin : C.unary(E, 'square', margin))
    }
    case 'categorical_hinge': {
      const pos = sumLast(E, mul(y, yPred))
      const neg = C.maxAlong(E, mul(sub1(E, y), yPred), -1, false).values
      return C.unary(E, 'relu', add(C.binary(E, '-', neg, pos), scalarT(1)))
    }
    case 'kl_divergence': {
      const t = C.clamp(E, y, EPS, 1)
      const q = C.clamp(E, yPred, EPS, 1)
      return sumLast(E, mul(t, C.unary(E, 'log', C.binary(E, '/', t, q))))
    }
    case 'poisson': return meanLast(E, C.binary(E, '-', yPred, mul(y, C.unary(E, 'log', add(yPred, scalarT(EPS))))))
    case 'cosine_similarity': return C.unary(E, 'neg', sumLast(E, mul(l2Normalize(E, y, -1), l2Normalize(E, yPred, -1))))
  }
  throw new PyError('ValueError', `Could not interpret loss identifier: ${spec.name}`)
}

type ReductionK = 'sum_over_batch_size' | 'mean' | 'sum' | 'none'

/** keras.src.losses.loss.reduce_values: the weights multiply each sample's loss */
const reduceLossT = (E: Engine, per: TT, w: TT | null, reduction: ReductionK): TT => {
  let v = per
  if (w) {
    const ww = w.ndim < v.ndim ? C.reshapeT(E, w, [...w.shape, ...new Array(v.ndim - w.ndim).fill(1)], false) : w
    v = C.binary(E, '*', v, ww)
  }
  if (reduction === 'none') return v
  if (reduction === 'sum') return C.reduceSum(E, v, null, false, false)
  if (reduction === 'mean' && w) return C.binary(E, '/', C.reduceSum(E, v, null, false, false), C.reduceSum(E, w, null, false, false))
  return C.reduceSum(E, v, null, false, true)
}

const specDefaults = (name: string, fn: Value | null = null): LossSpec => ({ name, fromLogits: false, delta: 1, smoothing: 0, fn, reduction: 'sum_over_batch_size' })

const LOSS_SIG = 'reduction=sum_over_batch_size name=None dtype=None'
const LOSS_CLASS_SIG: Record<string, string> = {
  CategoricalCrossentropy: `from_logits=False label_smoothing=0.0 !axis=-1 ${LOSS_SIG}`,
  BinaryCrossentropy: `from_logits=False label_smoothing=0.0 !axis=-1 ${LOSS_SIG}`,
  SparseCategoricalCrossentropy: `from_logits=False !ignore_class=None ${LOSS_SIG}`,
  Huber: `delta=1.0 ${LOSS_SIG}`,
  CosineSimilarity: `!axis=-1 ${LOSS_SIG}`,
}

class LossObj extends PyObj {
  readonly cls: string
  spec: LossSpec
  reduction: ReductionK
  name: string
  constructor(cls: string, spec: LossSpec, reduction: ReductionK = 'sum_over_batch_size', name?: string) {
    super()
    this.cls = cls
    this.spec = spec
    this.reduction = reduction
    this.name = name ?? spec.name
  }
  repr() { return `<LossFunctionWrapper(<function ${this.spec.name}>, kwargs={'from_logits': ${this.spec.fromLogits ? 'True' : 'False'}})>` }
  call(args: Value[], kw: Kw, h: Host): Value {
    const E = stateOf(h).E
    const w = kw.sample_weight ?? args[2]
    return tfWrap(reduceLossT(E, perSample(E, h, this.spec, asFloat(h, args[0] ?? kw.y_true), asFloat(h, args[1] ?? kw.y_pred), false), isNone(w) ? null : asFloat(h, w), this.reduction))
  }
  getAttr(name: string): Value | undefined {
    if (name === 'name') return py.str(this.name)
    if (name === 'reduction') return py.str(this.reduction)
    return undefined
  }
}

const LOSS_HOST = 'keras.loss'

const lossSpec = (h: Host, v: Value | undefined): LossSpec => {
  if (isNone(v)) throw h.err('ValueError', 'No loss to compute. Provide a `loss` argument in `compile()`.')
  if (v!.k === 'obj' && v!.o instanceof LossObj) {
    if (v!.o.reduction === 'none') throw h.err('NotImplementedError', 'compile: a loss with reduction=None is not implemented in this sandbox')
    return { ...v!.o.spec, reduction: v!.o.reduction }
  }
  if (v!.k === 'func' || v!.k === 'bound') return specDefaults(v!.k === 'func' ? v!.name : v!.f.name, v!)
  if (v!.k === 'inst') {
    const f = userMethod(v!.cls, 'call')
    if (f) return specDefaults((v!.state[LOSS_HOST] as string | undefined) ?? snake(v!.cls.name), { k: 'bound', self: v!, f })
    return specDefaults(snake(v!.cls.name), v!)
  }
  const name = v!.k === 'str' ? v!.v : v!.k === 'fn' ? v!.name : ''
  const n = LOSS_ALIASES[name]
  if (!n) throw h.err('ValueError', `Could not interpret loss identifier: ${name || repr(v!)}`)
  return specDefaults(n)
}

// ---------------------------------------------------------------- metrics

const METRIC_ALIASES: Record<string, string> = {
  binary_accuracy: 'BinaryAccuracy', categorical_accuracy: 'CategoricalAccuracy', sparse_categorical_accuracy: 'SparseCategoricalAccuracy',
  top_k_categorical_accuracy: 'TopKCategoricalAccuracy', sparse_top_k_categorical_accuracy: 'SparseTopKCategoricalAccuracy',
  mse: 'MeanSquaredError', mean_squared_error: 'MeanSquaredError', mae: 'MeanAbsoluteError', mean_absolute_error: 'MeanAbsoluteError',
  mape: 'MeanAbsolutePercentageError', mean_absolute_percentage_error: 'MeanAbsolutePercentageError', msle: 'MeanSquaredLogarithmicError', mean_squared_logarithmic_error: 'MeanSquaredLogarithmicError',
  binary_crossentropy: 'BinaryCrossentropy', categorical_crossentropy: 'CategoricalCrossentropy', sparse_categorical_crossentropy: 'SparseCategoricalCrossentropy',
  hinge: 'Hinge', squared_hinge: 'SquaredHinge', categorical_hinge: 'CategoricalHinge', kl_divergence: 'KLDivergence', kld: 'KLDivergence', poisson: 'Poisson', log_cosh: 'LogCoshError', logcosh: 'LogCoshError',
  root_mean_squared_error: 'RootMeanSquaredError', rmse: 'RootMeanSquaredError', cosine_similarity: 'CosineSimilarity', auc: 'AUC', precision: 'Precision', recall: 'Recall',
  Accuracy: 'Accuracy',
}

/** classes that are the mean of a loss function */
const METRIC_LOSS: Record<string, string> = {
  MeanSquaredError: 'mean_squared_error', MeanAbsoluteError: 'mean_absolute_error', MeanAbsolutePercentageError: 'mean_absolute_percentage_error', MeanSquaredLogarithmicError: 'mean_squared_logarithmic_error',
  BinaryCrossentropy: 'binary_crossentropy', CategoricalCrossentropy: 'categorical_crossentropy', SparseCategoricalCrossentropy: 'sparse_categorical_crossentropy',
  Hinge: 'hinge', SquaredHinge: 'squared_hinge', CategoricalHinge: 'categorical_hinge', KLDivergence: 'kl_divergence', Poisson: 'poisson', LogCoshError: 'log_cosh',
}

/** to_snake_case of the class names Keras uses as a metric's default name */
const metricName = (cls: string): string => ({ AUC: 'auc', RootMeanSquaredError: 'root_mean_squared_error', TopKCategoricalAccuracy: 'top_k_categorical_accuracy', SparseTopKCategoricalAccuracy: 'sparse_top_k_categorical_accuracy', KLDivergence: 'kl_divergence', LogCoshError: 'logcosh', Mean: 'mean', Sum: 'sum' } as Record<string, string>)[cls] ?? snake(cls)

const METRIC_SIG: Record<string, string> = {
  Accuracy: 'name=None dtype=None',
  BinaryAccuracy: 'name=None dtype=None threshold=0.5',
  CategoricalAccuracy: 'name=None dtype=None',
  SparseCategoricalAccuracy: 'name=None dtype=None',
  TopKCategoricalAccuracy: 'k=5 name=None dtype=None',
  SparseTopKCategoricalAccuracy: 'k=5 name=None dtype=None !from_sorted_ids=False',
  Precision: '!thresholds=None !top_k=None class_id=None name=None dtype=None',
  Recall: '!thresholds=None !top_k=None class_id=None name=None dtype=None',
  AUC: 'num_thresholds=200 !curve=ROC !summation_method=interpolation name=None dtype=None !thresholds=None !multi_label=False !num_labels=None !label_weights=None from_logits=False',
  TruePositives: '!thresholds=None name=None dtype=None', FalsePositives: '!thresholds=None name=None dtype=None', TrueNegatives: '!thresholds=None name=None dtype=None', FalseNegatives: '!thresholds=None name=None dtype=None',
  Mean: 'name=None dtype=None', Sum: 'name=None dtype=None',
  BinaryCrossentropy: 'name=None dtype=None from_logits=False label_smoothing=0.0',
  CategoricalCrossentropy: 'name=None dtype=None from_logits=False label_smoothing=0.0 !axis=-1',
  SparseCategoricalCrossentropy: 'name=None dtype=None from_logits=False !axis=-1',
  CosineSimilarity: 'name=None dtype=None !axis=-1',
}

const THRESHOLD = 0.5

/** keras.metrics.*: state that update_state accumulates and result reads */
class KMetric extends PyObj {
  readonly cls: string
  name: string
  P: Params
  total = 0
  count = 0
  tp: number[] = []
  fp: number[] = []
  tn: number[] = []
  fn: number[] = []
  thresholds: number[] = []
  /** 'accuracy' chooses among the accuracies when it first sees the data */
  auto: boolean
  impl: string
  /** a Python function given to metrics=[...] */
  pyFn: Value | null = null
  /** a user's keras.metrics.Metric */
  inst: InstValue | null = null
  constructor(cls: string, name: string, P: Params, auto = false) {
    super()
    this.cls = cls
    this.name = name
    this.P = P
    this.auto = auto
    this.impl = cls
    this.reset()
  }
  repr() { return `<${this.cls} name=${this.name}>` }
  /** higher is better? (what EarlyStopping and ModelCheckpoint use on mode='auto') */
  direction(): 'up' | 'down' {
    return ['Accuracy', 'BinaryAccuracy', 'CategoricalAccuracy', 'SparseCategoricalAccuracy', 'TopKCategoricalAccuracy', 'SparseTopKCategoricalAccuracy', 'Precision', 'Recall', 'AUC', 'TruePositives', 'TrueNegatives', 'CosineSimilarity'].includes(this.impl) ? 'up' : 'down'
  }
  reset() {
    this.total = 0
    this.count = 0
    if (this.impl === 'AUC') {
      const n = this.P.num_thresholds.k === 'int' ? this.P.num_thresholds.v : 200
      this.thresholds = [-EPS, ...Array.from({ length: n - 2 }, (_, i) => (i + 1) / (n - 1)), 1 + EPS]
    } else this.thresholds = [THRESHOLD]
    const zeros = () => new Array<number>(this.thresholds.length).fill(0)
    this.tp = zeros()
    this.fp = zeros()
    this.tn = zeros()
    this.fn = zeros()
  }
  addMean(v: number[], w: number[] | null) {
    v.forEach((x, i) => {
      const wi = w ? w[i] : 1
      this.total += x * wi
      this.count += wi
    })
  }
  /** per-sample values, reduced to the samples, each with its weight */
  update(h: Host, y0: TT, p0: TT, w: TT | null) {
    const E = new Engine(h.rng)
    E.grad = false
    const wv = w ? w.values() : null
    if (this.pyFn) {
      const r = h.call(this.pyFn, [tfWrap(y0), tfWrap(p0)])
      this.addMean(toSamples(E, tfOf(r) ?? scalarT(h.num(r))).values(), wv)
      return
    }
    if (this.auto) this.resolveAuto(y0, p0)
    const y = sameRank(E, y0, p0)
    switch (this.impl) {
      case 'Mean': this.addMean(toSamples(E, p0).values(), wv); return
      case 'Sum': {
        const v = p0.values()
        v.forEach((x, i) => { this.total += x * (wv ? wv[i] : 1) })
        return
      }
      case 'Accuracy': {
        const yv = y.values()
        const pv = p0.values()
        const eq = TT.of(yv.map((v, i) => (Math.fround(v) === Math.fround(pv[i]) ? 1 : 0)), y.shape, 'float32')
        this.addMean(toSamples(E, eq).values(), wv)
        return
      }
      case 'BinaryAccuracy': {
        const thr = this.P.threshold && this.P.threshold.k !== 'none' ? h.num(this.P.threshold) : THRESHOLD
        const yv = y.values()
        const pv = p0.values()
        if (yv.length !== pv.length) throw h.err('ValueError', `Shapes ${shapeStr(y.shape)} and ${shapeStr(p0.shape)} are incompatible`)
        const eq = TT.of(pv.map((v, i) => ((v > thr ? 1 : 0) === yv[i] ? 1 : 0)), p0.shape, 'float32')
        this.addMean(toSamples(E, eq).values(), wv)
        return
      }
      case 'CategoricalAccuracy': case 'SparseCategoricalAccuracy': {
        const labels = this.impl === 'CategoricalAccuracy' ? rowArgmax(y0) : y0.values()
        const preds = rowArgmax(p0)
        this.addMean(preds.map((v, i) => (v === Math.round(labels[i]) ? 1 : 0)), wv)
        return
      }
      case 'TopKCategoricalAccuracy': case 'SparseTopKCategoricalAccuracy': {
        const k = h.toInt(this.P.k)
        const labels = this.impl === 'TopKCategoricalAccuracy' ? rowArgmax(y0) : y0.values()
        const d = p0.shape[p0.ndim - 1]
        const pv = p0.values()
        this.addMean(labels.map((l, n) => {
          const row = pv.slice(n * d, (n + 1) * d)
          const better = row.filter((v) => v > row[Math.round(l)]).length
          return better < k ? 1 : 0
        }), wv)
        return
      }
      case 'RootMeanSquaredError': {
        this.addMean(toSamples(E, C.unary(E, 'square', C.binary(E, '-', p0, y))).values(), wv)
        return
      }
      case 'CosineSimilarity': {
        const s = C.reduceSum(E, C.binary(E, '*', l2Normalize(E, y, -1), l2Normalize(E, p0, -1)), [-1], false, false)
        this.addMean(toSamples(E, s).values(), wv)
        return
      }
      case 'Precision': case 'Recall': case 'AUC': case 'TruePositives': case 'FalsePositives': case 'TrueNegatives': case 'FalseNegatives': {
        let yv = y.values()
        let pv = p0.values()
        if (this.impl === 'AUC' && this.P.from_logits && h.truthy(this.P.from_logits)) pv = pv.map(sigmoidV)
        if (this.P.class_id && this.P.class_id.k !== 'none') {
          const c = h.toInt(this.P.class_id)
          const d = p0.shape[p0.ndim - 1]
          yv = yv.filter((_, i) => i % d === c)
          pv = pv.filter((_, i) => i % d === c)
        }
        if (yv.length !== pv.length) throw h.err('ValueError', `Shapes ${shapeStr(y.shape)} and ${shapeStr(p0.shape)} are incompatible`)
        const per = pv.length / Math.max(1, wv?.length ?? pv.length)
        this.thresholds.forEach((thr, t) => {
          pv.forEach((v, i) => {
            const wi = wv ? wv[Math.floor(i / per)] : 1
            const pos = yv[i] !== 0
            if (v > thr) { if (pos) this.tp[t] += wi; else this.fp[t] += wi } else if (pos) this.fn[t] += wi
            else this.tn[t] += wi
          })
        })
        return
      }
    }
    const lossName = METRIC_LOSS[this.impl]
    if (lossName) {
      const spec = specDefaults(lossName)
      if (this.P.from_logits) spec.fromLogits = h.truthy(this.P.from_logits)
      if (this.P.label_smoothing) spec.smoothing = h.num(this.P.label_smoothing)
      this.addMean(toSamples(E, perSample(E, h, spec, y0, p0, false)).values(), wv)
      return
    }
    throw h.err('NotImplementedError', `metric ${this.cls} is not implemented in this sandbox`)
  }
  /** keras.src.trainers.compile_utils.get_metric */
  resolveAuto(y: TT, p: TT) {
    this.auto = false
    const last = p.shape[p.ndim - 1]
    if (last === 1) this.impl = 'BinaryAccuracy'
    else if (y.ndim < p.ndim || (y.ndim === p.ndim && y.shape[y.ndim - 1] === 1)) this.impl = 'SparseCategoricalAccuracy'
    else this.impl = 'CategoricalAccuracy'
  }
  value(h: Host): number {
    if (this.inst) {
      const f = h.getAttr(this.inst, 'result')
      return h.num(h.call(f, []))
    }
    const nan0 = (a: number, b: number) => (b ? a / b : 0)
    switch (this.impl) {
      case 'Sum': return this.total
      case 'RootMeanSquaredError': return Math.sqrt(nan0(this.total, this.count))
      case 'Precision': return nan0(this.tp[0], this.tp[0] + this.fp[0])
      case 'Recall': return nan0(this.tp[0], this.tp[0] + this.fn[0])
      case 'TruePositives': return this.tp[0]
      case 'FalsePositives': return this.fp[0]
      case 'TrueNegatives': return this.tn[0]
      case 'FalseNegatives': return this.fn[0]
      case 'AUC': {
        const recall = this.tp.map((t, i) => nan0(t, t + this.fn[i]))
        const fpr = this.fp.map((f, i) => nan0(f, f + this.tn[i]))
        let area = 0
        for (let i = 0; i + 1 < recall.length; i++) area += (fpr[i] - fpr[i + 1]) * ((recall[i] + recall[i + 1]) / 2)
        return area
      }
    }
    return nan0(this.total, this.count)
  }
  doUpdate(h: Host, y: TT, p: TT, w: TT | null) {
    if (this.inst) {
      const f = h.getAttr(this.inst, 'update_state')
      h.call(f, [tfWrap(y), tfWrap(p)], w ? { sample_weight: tfWrap(w) } : {})
    } else this.update(h, y, p, w)
  }
  doReset(h: Host) {
    if (this.inst) h.call(h.getAttr(this.inst, 'reset_state'), [])
    else this.reset()
  }
  call(args: Value[], kw: Kw, h: Host): Value {
    this.update(h, asFloat(h, args[0] ?? kw.y_true), asFloat(h, args[1] ?? kw.y_pred), isNone(kw.sample_weight ?? args[2]) ? null : asFloat(h, kw.sample_weight ?? args[2]))
    return tfWrap(scalarT(this.value(h)))
  }
  getAttr(name: string, h: Host): Value | undefined {
    switch (name) {
      case 'name': return py.str(this.name)
      case 'result': return fnv(name, () => tfWrap(scalarT(this.value(h))))
      case 'reset_state': case 'reset_states': return fnv(name, () => { this.reset(); return py.NONE })
      case 'update_state': return fnv(name, (args, kw) => {
        const single = this.impl === 'Mean' || this.impl === 'Sum'
        const a0 = args[0] ?? kw.y_true ?? kw.values
        const a1 = single ? a0 : args[1] ?? kw.y_pred
        const wv = single ? kw.sample_weight ?? args[1] : kw.sample_weight ?? args[2]
        this.update(h, asFloat(h, a0), asFloat(h, a1), isNone(wv) ? null : asFloat(h, wv))
        return py.NONE
      })
      case 'get_config': return fnv(name, () => ({ k: 'dict', d: PyDict.from([[py.str('name'), py.str(this.name)], [py.str('dtype'), py.str('float32')]]) }))
    }
    return undefined
  }
}

const rowArgmax = (t: TT): number[] => {
  const d = t.shape[t.ndim - 1]
  const v = t.values()
  const n = v.length / d
  const out = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    let best = 0
    for (let j = 1; j < d; j++) if (v[i * d + j] > v[i * d + best]) best = j
    out[i] = best
  }
  return out
}

/** reduce to one value per sample */
const toSamples = (E: Engine, x: TT): TT => (x.ndim <= 1 ? x : C.reduceSum(E, x, x.shape.map((_, i) => i).slice(1), false, true))

// ---------------------------------------------------------------- learning-rate schedules and optimisers

const SCHEDULE_SIG: Record<string, string> = {
  ExponentialDecay: 'initial_learning_rate decay_steps decay_rate staircase=False name=ExponentialDecay',
  PiecewiseConstantDecay: 'boundaries values name=PiecewiseConstant',
  CosineDecay: 'initial_learning_rate decay_steps alpha=0.0 name=CosineDecay warmup_target=None warmup_steps=0',
  InverseTimeDecay: 'initial_learning_rate decay_steps decay_rate staircase=False name=InverseTimeDecay',
  PolynomialDecay: 'initial_learning_rate decay_steps end_learning_rate=0.0001 power=1.0 cycle=False name=PolynomialDecay',
}

class LRSchedule extends PyObj {
  readonly cls: string
  P: Params
  constructor(cls: string, P: Params) {
    super()
    this.cls = cls
    this.P = P
  }
  repr() { return `<keras.src.optimizers.schedules.learning_rate_schedule.${this.cls} object>` }
  at(h: Host, step: number): number {
    const n = (k: string) => h.num(this.P[k])
    switch (this.cls) {
      case 'ExponentialDecay': {
        const p = step / n('decay_steps')
        return n('initial_learning_rate') * n('decay_rate') ** (h.truthy(this.P.staircase) ? Math.floor(p) : p)
      }
      case 'InverseTimeDecay': {
        const p = step / n('decay_steps')
        return n('initial_learning_rate') / (1 + n('decay_rate') * (h.truthy(this.P.staircase) ? Math.floor(p) : p))
      }
      case 'PiecewiseConstantDecay': {
        const bs = h.iterate(this.P.boundaries).map((v) => h.num(v))
        const vs = h.iterate(this.P.values).map((v) => h.num(v))
        for (let i = 0; i < bs.length; i++) if (step <= bs[i]) return vs[i]
        return vs[vs.length - 1]
      }
      case 'PolynomialDecay': {
        const lr = n('initial_learning_rate')
        const end = n('end_learning_rate')
        let ds = n('decay_steps')
        let s = step
        if (h.truthy(this.P.cycle)) ds *= s === 0 ? 1 : Math.ceil(s / ds)
        else s = Math.min(s, ds)
        return (lr - end) * (1 - s / ds) ** n('power') + end
      }
    }
    // CosineDecay, with its optional warm-up
    const alpha = n('alpha')
    const lr0 = n('initial_learning_rate')
    const ds = n('decay_steps')
    const decay = (s: number, from: number) => from * ((1 - alpha) * (0.5 * (1 + Math.cos((Math.PI * s) / ds))) + alpha)
    if (this.P.warmup_target.k === 'none') return decay(Math.min(step, ds), lr0)
    const target = n('warmup_target')
    const ws = n('warmup_steps')
    const s = Math.min(step, ds + ws)
    return s < ws ? (target - lr0) * (s / ws) + lr0 : decay(s - ws, target)
  }
  call(args: Value[], kw: Kw, h: Host): Value { return tfWrap(scalarT(this.at(h, h.num(args[0] ?? kw.step)))) }
  getAttr(name: string): Value | undefined {
    if (name === 'get_config') return fnv(name, () => ({ k: 'dict', d: PyDict.from(Object.entries(this.P).map(([k, v]) => [py.str(k), v])) }))
    return this.P[name]
  }
}

const OPT_COMMON = 'weight_decay=None clipnorm=None clipvalue=None global_clipnorm=None !use_ema=False !ema_momentum=0.99 !ema_overwrite_frequency=None !loss_scale_factor=None !gradient_accumulation_steps=None name=None'
const OPT_SIG: Record<string, string> = {
  SGD: `learning_rate=0.01 momentum=0.0 nesterov=False ${OPT_COMMON}`,
  Adam: `learning_rate=0.001 beta_1=0.9 beta_2=0.999 epsilon=1e-7 amsgrad=False ${OPT_COMMON}`,
  AdamW: `learning_rate=0.001 weight_decay=0.004 beta_1=0.9 beta_2=0.999 epsilon=1e-7 amsgrad=False ${OPT_COMMON.replace('weight_decay=None ', '')}`,
  RMSprop: `learning_rate=0.001 rho=0.9 momentum=0.0 epsilon=1e-7 centered=False ${OPT_COMMON}`,
  Adagrad: `learning_rate=0.001 initial_accumulator_value=0.1 epsilon=1e-7 ${OPT_COMMON}`,
  Adadelta: `learning_rate=0.001 rho=0.95 epsilon=1e-7 ${OPT_COMMON}`,
  Adamax: `learning_rate=0.001 beta_1=0.9 beta_2=0.999 epsilon=1e-7 ${OPT_COMMON}`,
  Nadam: `learning_rate=0.001 beta_1=0.9 beta_2=0.999 epsilon=1e-7 ${OPT_COMMON}`,
}

/** the update rules of keras.src.optimizers.* */
class KOptimizer extends PyObj {
  readonly cls: string
  P: Params
  lr: number | LRSchedule
  /** iterations */
  t = 0
  built = false
  slots = new Map<TT, Record<string, number[]>>()
  uProduct = 1
  private num: Record<string, number>
  constructor(h: Host, cls: string, P: Params) {
    super()
    this.cls = cls
    this.P = P
    const lv = P.learning_rate
    if (lv.k === 'obj' && lv.o instanceof LRSchedule) this.lr = lv.o
    else this.lr = h.num(lv)
    this.num = {}
    for (const [k, v] of Object.entries(P)) if (v.k === 'int' || v.k === 'float') this.num[k] = v.v
    if ((this.num.clipnorm ? 1 : 0) + (this.num.clipvalue ? 1 : 0) + (this.num.global_clipnorm ? 1 : 0) > 1) throw h.err('ValueError', `Only one of \`clipnorm\`, \`clipvalue\` and \`global_clipnorm\` can be set. Received: clipnorm=${repr(P.clipnorm)}, clipvalue=${repr(P.clipvalue)}, global_clipnorm=${repr(P.global_clipnorm)}`)
  }
  repr() { return `<keras.src.optimizers.${this.cls.toLowerCase()}.${this.cls} object>` }
  currentLr(h: Host): number { return typeof this.lr === 'number' ? this.lr : this.lr.at(h, this.t) }
  /** how many state variables the optimizer keeps per weight (for model.summary) */
  slotCount(): number {
    switch (this.cls) {
      case 'SGD': return this.num.momentum ? 1 : 0
      case 'RMSprop': return 1 + (this.num.momentum > 0 ? 1 : 0) + (this.P.centered.k === 'bool' && this.P.centered.v ? 1 : 0)
      case 'Adagrad': return 1
      case 'Adam': case 'AdamW': return 2 + (this.P.amsgrad.k === 'bool' && this.P.amsgrad.v ? 1 : 0)
    }
    return 2
  }
  /** optimizer.variables: iterations and learning_rate, then the slots (Nadam also keeps u_product) */
  counts(weights: number): { n: number; bytes: number } {
    const extra = this.cls === 'Nadam' ? 1 : 0
    const slot = this.slotCount() * weights
    return { n: 2 + extra + slot, bytes: 8 + 4 + 4 * extra + 4 * slot }
  }
  slot(p: TT, name: string, init = 0): number[] {
    const s = this.slots.get(p) ?? {}
    this.slots.set(p, s)
    return (s[name] ??= new Array<number>(p.size).fill(init))
  }
  clip(grads: number[][]): number[][] {
    const norm = (g: number[]) => Math.sqrt(g.reduce((a, x) => a + x * x, 0))
    if (this.num.clipnorm > 0) {
      const c = this.num.clipnorm
      return grads.map((g) => {
        const n = norm(g)
        return n > 0 ? g.map((x) => (x * c) / Math.max(n, c)) : g
      })
    }
    if (this.num.global_clipnorm > 0) {
      const c = this.num.global_clipnorm
      const n = Math.sqrt(grads.reduce((a, g) => a + g.reduce((b, x) => b + x * x, 0), 0))
      return n > c ? grads.map((g) => g.map((x) => (x * c) / n)) : grads
    }
    if (this.num.clipvalue > 0) {
      const v = this.num.clipvalue
      return grads.map((g) => g.map((x) => Math.min(v, Math.max(-v, x))))
    }
    return grads
  }
  step(h: Host, params: TT[]) {
    const live = params.filter((p) => p.grad)
    this.apply(h, live, live.map((p) => p.grad!.values()))
  }
  apply(h: Host, ps: TT[], grads0: number[][]) {
    this.built = true
    const grads = this.clip(grads0)
    const lr = this.currentLr(h)
    const wd = this.num.weight_decay ?? 0
    if (this.cls === 'Nadam') this.uProduct *= this.num.beta_1 * (1 - 0.5 * 0.96 ** (this.t + 1))
    ps.forEach((p, k) => {
      const g = grads[k]
      const w0 = p.values()
      const w = wd ? w0.map((x) => x - x * wd * lr) : w0
      C.writeInto(p, this.update(p, g, w, lr))
    })
    this.t++
  }
  update(p: TT, g: number[], w: number[], lr: number): number[] {
    const { beta_1: b1, beta_2: b2, epsilon: eps, rho, momentum } = this.num
    const step = this.t + 1
    switch (this.cls) {
      case 'SGD': {
        if (!momentum) return w.map((x, i) => x - lr * g[i])
        const m = this.slot(p, 'momentum').map((x, i) => momentum * x - lr * g[i])
        this.slots.get(p)!.momentum = m
        return this.P.nesterov.k === 'bool' && this.P.nesterov.v ? w.map((x, i) => x + momentum * m[i] - lr * g[i]) : w.map((x, i) => x + m[i])
      }
      case 'RMSprop': {
        const s = this.slots.get(p) ?? {}
        this.slots.set(p, s)
        const v = this.slot(p, 'velocity').map((x, i) => rho * x + (1 - rho) * g[i] * g[i])
        s.velocity = v
        const centered = this.P.centered.k === 'bool' && this.P.centered.v
        let denom = v.map((x) => x + eps)
        if (centered) {
          const a = this.slot(p, 'average').map((x, i) => rho * x + (1 - rho) * g[i])
          s.average = a
          denom = v.map((x, i) => x - a[i] * a[i] + eps)
        }
        const inc = g.map((x, i) => (lr * x) / Math.sqrt(denom[i]))
        if (momentum > 0) {
          const m = this.slot(p, 'momentum').map((x, i) => momentum * x + inc[i])
          s.momentum = m
          return w.map((x, i) => x - m[i])
        }
        return w.map((x, i) => x - inc[i])
      }
      case 'Adagrad': {
        const a = this.slot(p, 'accumulator', this.num.initial_accumulator_value).map((x, i) => x + g[i] * g[i])
        this.slots.get(p)!.accumulator = a
        return w.map((x, i) => x - (lr * g[i]) / Math.sqrt(a[i] + eps))
      }
      case 'Adadelta': {
        const rms = (x: number) => Math.sqrt(x + eps)
        const ag = this.slot(p, 'grad').map((x, i) => rho * x + (1 - rho) * g[i] * g[i])
        const ad = this.slot(p, 'delta')
        const delta = g.map((x, i) => (-rms(ad[i]) * x) / rms(ag[i]))
        const s = this.slots.get(p)!
        s.grad = ag
        s.delta = ad.map((x, i) => rho * x + (1 - rho) * delta[i] * delta[i])
        return w.map((x, i) => x + lr * delta[i])
      }
      case 'Adamax': {
        const m = this.slot(p, 'm').map((x, i) => b1 * x + (1 - b1) * g[i])
        const u = this.slot(p, 'u').map((x, i) => Math.max(b2 * x, Math.abs(g[i])))
        const s = this.slots.get(p)!
        s.m = m
        s.u = u
        const pow = 1 - b1 ** step
        return w.map((x, i) => x - (lr * m[i]) / (pow * (u[i] + eps)))
      }
      case 'Nadam': {
        const uT = b1 * (1 - 0.5 * 0.96 ** step)
        const uT1 = b1 * (1 - 0.5 * 0.96 ** (step + 1))
        const prodT1 = this.uProduct * uT1
        const m = this.slot(p, 'm').map((x, i) => b1 * x + (1 - b1) * g[i])
        const v = this.slot(p, 'v').map((x, i) => b2 * x + (1 - b2) * g[i] * g[i])
        const s = this.slots.get(p)!
        s.m = m
        s.v = v
        const b2p = b2 ** step
        return w.map((x, i) => {
          const mHat = (uT1 * m[i]) / (1 - prodT1) + ((1 - uT) * g[i]) / (1 - this.uProduct)
          return x - (mHat * lr) / (Math.sqrt(v[i] / (1 - b2p)) + eps)
        })
      }
    }
    // Adam and AdamW
    const m = this.slot(p, 'm').map((x, i) => b1 * x + (1 - b1) * g[i])
    const v = this.slot(p, 'v').map((x, i) => b2 * x + (1 - b2) * g[i] * g[i])
    const s = this.slots.get(p)!
    s.m = m
    s.v = v
    let vUse = v
    if (this.P.amsgrad.k === 'bool' && this.P.amsgrad.v) {
      vUse = this.slot(p, 'vhat').map((x, i) => Math.max(x, v[i]))
      s.vhat = vUse
    }
    const a = (lr * Math.sqrt(1 - b2 ** step)) / (1 - b1 ** step)
    return w.map((x, i) => x - (a * m[i]) / (Math.sqrt(vUse[i]) + eps))
  }
  getAttr(name: string, h: Host): Value | undefined {
    switch (name) {
      case 'learning_rate': case 'lr': return py.obj(new LRVariable(this, TT.scalar(this.currentLr(h), 'float32')))
      case 'iterations': return py.obj(new KVariable(TT.scalar(this.t, 'int64'), `${this.cls.toLowerCase()}/iteration`, false))
      case 'name': return py.str(this.P.name.k === 'str' ? this.P.name.v : snake(this.cls))
      case 'apply_gradients': return fnv(name, (args, kw) => {
        const pairs = h.iterate(args[0] ?? kw.grads_and_vars).map((x) => h.iterate(x))
        const live = pairs.filter(([g]) => g.k !== 'none')
        if (!live.length) throw h.err('ValueError', `No gradients provided for any variable.`)
        this.apply(h, live.map(([, v]) => tfOf(v)!), live.map(([g]) => toTF(h, g, null).values()))
        return py.NONE
      })
      case 'apply': return fnv(name, (args, kw) => {
        const gs = h.iterate(args[0] ?? kw.grads)
        const vs = h.iterate(args[1] ?? kw.trainable_variables)
        const live = gs.map((g, i) => [g, vs[i]]).filter(([g]) => g.k !== 'none')
        this.apply(h, live.map(([, v]) => tfOf(v)!), live.map(([g]) => toTF(h, g, null).values()))
        return py.NONE
      })
      case 'get_config': return fnv(name, () => jsToPy(this.config(h)))
    }
    if (name in this.P && name !== 'name') return this.P[name]
    return undefined
  }
  config(h: Host): { [k: string]: Json } {
    const out: { [k: string]: Json } = { name: this.P.name.k === 'str' ? this.P.name.v : snake(this.cls), learning_rate: typeof this.lr === 'number' ? jf(f32clean(Math.fround(this.lr))) : objConfig('keras.optimizers.schedules', this.lr.cls, Object.fromEntries(Object.entries(this.lr.P).map(([k, v]) => [k, pyToJs(h, v)]))) }
    for (const [k, v] of Object.entries(this.P)) if (k !== 'name' && k !== 'learning_rate') out[k] = pyToJs(h, v)
    return out
  }
  setAttr(name: string, v: Value, h: Host): boolean {
    if (name === 'learning_rate' || name === 'lr') {
      if (v.k === 'obj' && v.o instanceof LRSchedule) this.lr = v.o
      else {
        if (typeof this.lr !== 'number') throw h.err('TypeError', 'This optimizer was created with a `LearningRateSchedule` object as its `learning_rate` constructor argument, hence its learning rate is not settable. If you need the learning rate to be settable, you should instantiate the optimizer with a float `learning_rate` argument.')
        this.lr = h.num(v)
      }
      return true
    }
    return false
  }
}

/** optimizer.learning_rate: assigning to it sets the optimizer's rate */
class LRVariable extends KVariable {
  opt: KOptimizer
  constructor(opt: KOptimizer, t: TT) {
    super(t, `${opt.cls.toLowerCase()}/learning_rate`, false)
    this.opt = opt
  }
  getAttr(name: string, h: Host): Value | undefined {
    if (name === 'assign') return fnv(name, (args) => {
      this.opt.setAttr('learning_rate', py.float(h.num(args[0])), h)
      C.writeInto(this.t, [h.num(args[0])])
      return py.obj(this)
    })
    return super.getAttr(name, h)
  }
}

const optimizerArg = (h: Host, v: Value | undefined): KOptimizer => {
  if (v && v.k === 'obj' && v.o instanceof KOptimizer) return v.o
  const name = v && v.k === 'str' ? v.v.toLowerCase() : isNone(v) ? 'rmsprop' : ''
  const cls = Object.keys(OPT_SIG).find((c) => c.toLowerCase() === name)
  if (!cls) throw h.err('ValueError', `Could not interpret optimizer identifier: ${v ? repr(v) : 'None'}`)
  return new KOptimizer(h, cls, bindArgs(h, cls, OPT_SIG[cls], [], {}, null))
}

// ---------------------------------------------------------------- callbacks

type Logs = Record<string, number>

/** what a callback sees during fit */
interface CallbackHooks {
  setup?(model: KModel, params: { epochs: number; steps: number; verbose: number }, h: Host): void
  onTrainBegin?(h: Host, logs: Logs): void
  onTrainEnd?(h: Host, logs: Logs): void
  onEpochBegin?(h: Host, epoch: number, logs: Logs): void
  onEpochEnd?(h: Host, epoch: number, logs: Logs): void
  onBatchBegin?(h: Host, batch: number, logs: Logs): void
  onBatchEnd?(h: Host, batch: number, logs: Logs): void
  onTestBegin?(h: Host, logs: Logs): void
  onTestEnd?(h: Host, logs: Logs): void
  onPredictBegin?(h: Host, logs: Logs): void
  onPredictEnd?(h: Host, logs: Logs): void
}

const logsDict = (logs: Logs): Value => ({ k: 'dict', d: PyDict.from(Object.entries(logs).map(([k, v]) => [py.str(k), py.float(f32clean(Math.fround(v)))])) })

class KCallback extends PyObj implements CallbackHooks {
  readonly cls: string
  model: KModel | null = null
  constructor(cls: string) {
    super()
    this.cls = cls
  }
  repr() { return `<keras.src.callbacks.${snake(this.cls)}.${this.cls} object>` }
  setup(model: KModel, _p: { epochs: number; steps: number; verbose: number }, _h: Host) { this.model = model }
  getAttr(name: string): Value | undefined {
    if (name === 'model') return this.model ? this.model.self() : py.NONE
    return undefined
  }
}

/** the "is this better?" of EarlyStopping, ReduceLROnPlateau and ModelCheckpoint */
class MonitorCallback extends KCallback {
  monitor: string
  mode: string
  minDelta: number
  best: number | null = null
  up: boolean | null = null
  constructor(h: Host, cls: string, monitor: string, mode: string, minDelta: number) {
    super(cls)
    this.monitor = monitor
    this.mode = mode
    this.minDelta = Math.abs(minDelta)
    if (!['auto', 'min', 'max'].includes(mode)) {
      h.print(`UserWarning: ${cls} mode '${mode}' is unknown, fallback to auto mode.\n`)
      this.mode = 'auto'
    }
  }
  direction(h: Host): boolean {
    if (this.up !== null) return this.up
    if (this.mode === 'min') return (this.up = false)
    if (this.mode === 'max') return (this.up = true)
    const name = this.monitor.replace(/^val_/, '')
    if (name === 'loss' || name.endsWith('_loss')) return (this.up = false)
    const m = this.model?.metricSlots().find((s) => s.m.name === name || s.key === name)
    if (m) return (this.up = m.m.direction() === 'up')
    throw h.err('ValueError', `${this.cls} callback received monitor=${this.monitor}, but Keras isn't able to automatically determine whether that metric should be maximized or minimized. Pass \`mode='max'\` in order to monitor based on the highest metric value, or pass \`mode='min'\` in order to use the lowest value.`)
  }
  improved(h: Host, cur: number, ref: number | null): boolean {
    if (ref === null) return true
    return this.direction(h) ? cur - this.minDelta > ref : cur + this.minDelta < ref
  }
}

class EarlyStopping extends MonitorCallback {
  patience: number
  verbose: number
  baseline: number | null
  restore: boolean
  startFrom: number
  wait = 0
  stopped = 0
  bestWeights: number[][] | null = null
  bestEpoch = 0
  constructor(h: Host, P: Params) {
    super(h, 'EarlyStopping', h.str(P.monitor), h.str(P.mode), h.num(P.min_delta))
    this.patience = h.toInt(P.patience)
    this.verbose = h.toInt(P.verbose)
    this.baseline = isNone(P.baseline) ? null : h.num(P.baseline)
    this.restore = h.truthy(P.restore_best_weights)
    this.startFrom = h.toInt(P.start_from_epoch)
  }
  onTrainBegin() {
    this.wait = 0
    this.stopped = 0
    this.best = null
    this.bestWeights = null
    this.bestEpoch = 0
  }
  onEpochEnd(h: Host, epoch: number, logs: Logs) {
    const cur = logs[this.monitor]
    if (cur === undefined) {
      h.print(`UserWarning: Early stopping conditioned on metric \`${this.monitor}\` which is not available. Available metrics are: ${Object.keys(logs).join(',')}\n`)
      return
    }
    if (epoch < this.startFrom) return
    const model = this.model!
    if (this.restore && !this.bestWeights) {
      this.bestWeights = model.weightValues()
      this.bestEpoch = epoch
    }
    this.wait++
    if (this.improved(h, cur, this.best)) {
      this.best = cur
      this.bestEpoch = epoch
      if (this.restore) this.bestWeights = model.weightValues()
      if (this.baseline === null || this.improved(h, cur, this.baseline)) this.wait = 0
      return
    }
    if (this.wait >= this.patience && epoch > 0) {
      this.stopped = epoch
      model.stopTraining = true
    }
  }
  onTrainEnd(h: Host) {
    if (this.stopped > 0 && this.verbose > 0) h.print(`Epoch ${this.stopped + 1}: early stopping\n`)
    if (this.restore && this.bestWeights) {
      if (this.verbose > 0) h.print(`Restoring model weights from the end of the best epoch: ${this.bestEpoch + 1}.\n`)
      this.model!.setWeightValues(this.bestWeights)
    }
  }
  getAttr(name: string): Value | undefined {
    if (name === 'stopped_epoch') return py.int(this.stopped)
    if (name === 'best_epoch') return py.int(this.bestEpoch)
    if (name === 'wait') return py.int(this.wait)
    if (name === 'best') return this.best === null ? py.NONE : py.float(this.best)
    return super.getAttr(name)
  }
}

class ReduceLROnPlateau extends MonitorCallback {
  factor: number
  patience: number
  verbose: number
  cooldown: number
  minLr: number
  cooldownCounter = 0
  wait = 0
  constructor(h: Host, P: Params) {
    super(h, 'ReduceLROnPlateau', h.str(P.monitor), h.str(P.mode), h.num(P.min_delta))
    this.factor = h.num(P.factor)
    if (this.factor >= 1) throw h.err('ValueError', `ReduceLROnPlateau does not support a factor >= 1.0. Received factor=${this.factor}`)
    this.patience = h.toInt(P.patience)
    this.verbose = h.toInt(P.verbose)
    this.cooldown = h.toInt(P.cooldown)
    this.minLr = h.num(P.min_lr)
  }
  onTrainBegin() {
    this.cooldownCounter = 0
    this.wait = 0
  }
  onEpochEnd(h: Host, epoch: number, logs: Logs) {
    const opt = this.model!.compiled!.opt
    const old = Math.fround(opt.currentLr(h))
    logs.learning_rate = old
    const cur = logs[this.monitor]
    if (cur === undefined) {
      h.print(`UserWarning: Learning rate reduction is conditioned on metric \`${this.monitor}\` which is not available. Available metrics are: ${Object.keys(logs).join(',')}.\n`)
      return
    }
    if (this.cooldownCounter > 0) {
      this.cooldownCounter--
      this.wait = 0
    }
    if (this.improved(h, cur, this.best)) {
      this.best = cur
      this.wait = 0
    } else if (this.cooldownCounter <= 0) {
      this.wait++
      if (this.wait >= this.patience && old > Math.fround(this.minLr)) {
        const lr = Math.max(old * this.factor, this.minLr)
        opt.setAttr('learning_rate', py.float(lr), h)
        if (this.verbose > 0) h.print(`\nEpoch ${epoch + 1}: ReduceLROnPlateau reducing learning rate to ${lr}.\n`)
        this.cooldownCounter = this.cooldown
        this.wait = 0
      }
    }
  }
}

class LearningRateScheduler extends KCallback {
  schedule: Value
  verbose: number
  constructor(schedule: Value, verbose: number) {
    super('LearningRateScheduler')
    this.schedule = schedule
    this.verbose = verbose
  }
  onEpochBegin(h: Host, epoch: number) {
    const opt = this.model!.compiled!.opt
    const lr = Math.fround(opt.currentLr(h))
    let out: Value
    try {
      out = h.call(this.schedule, [py.int(epoch), py.float(lr)])
    } catch (e) {
      // the old one-argument form
      if (!(e instanceof PyError) || e.pyType !== 'TypeError') throw e
      out = h.call(this.schedule, [py.int(epoch)])
    }
    if (out.k !== 'float') throw h.err('ValueError', `The output of the \`schedule\` function should be a float. Got: ${repr(out)}`)
    opt.setAttr('learning_rate', out, h)
    if (this.verbose > 0) h.print(`\nEpoch ${epoch + 1}: LearningRateScheduler setting learning rate to ${out.v}.\n`)
  }
  onEpochEnd(h: Host, _e: number, logs: Logs) {
    logs.learning_rate = Math.fround(this.model!.compiled!.opt.currentLr(h))
  }
}

/** keeps the weights in memory under the file's name, so load_model / load_weights find them */
class ModelCheckpoint extends MonitorCallback {
  filepath: string
  verbose: number
  bestOnly: boolean
  weightsOnly: boolean
  constructor(h: Host, P: Params) {
    super(h, 'ModelCheckpoint', h.str(P.monitor), h.str(P.mode), 0)
    this.filepath = h.str(P.filepath)
    this.verbose = h.toInt(P.verbose)
    this.bestOnly = h.truthy(P.save_best_only)
    this.weightsOnly = h.truthy(P.save_weights_only)
    this.best = isNone(P.initial_value_threshold) ? null : h.num(P.initial_value_threshold)
    if (this.weightsOnly && !this.filepath.endsWith('.weights.h5')) throw h.err('ValueError', `When using \`save_weights_only=True\` in \`ModelCheckpoint\`, the filepath provided must end in \`.weights.h5\` (Keras weights format). Received: filepath=${this.filepath}`)
    if (!this.weightsOnly && !/\.(keras|h5)$/.test(this.filepath)) throw h.err('ValueError', `The filepath provided must end in \`.keras\` (Keras model format). Received: filepath=${this.filepath}`)
  }
  onEpochEnd(h: Host, epoch: number, logs: Logs) {
    const path = this.filepath.replace(/\{(\w+)(?::([^}]+))?\}/g, (_m, k: string, spec: string | undefined) => {
      const v = k === 'epoch' ? py.int(epoch + 1) : py.float(logs[k])
      return spec ? formatSpec(v, spec) : h.str(v)
    })
    if (this.bestOnly) {
      const cur = logs[this.monitor]
      if (cur === undefined) h.print(`UserWarning: Can save best model only with ${this.monitor} available.\n`)
      else {
        const bestStr = this.best === null ? 'None' : this.best.toFixed(5)
        if (!this.improved(h, cur, this.best)) {
          if (this.verbose > 0) h.print(`\nEpoch ${epoch + 1}: ${this.monitor} did not improve from ${bestStr}\n`)
          return
        }
        if (this.verbose > 0) h.print(`\nEpoch ${epoch + 1}: ${this.monitor} improved from ${bestStr} to ${cur.toFixed(5)}, saving model to ${path}\n`)
        this.best = cur
      }
    } else if (this.verbose > 0) h.print(`\nEpoch ${epoch + 1}: saving model to ${path}\n`)
    const m = this.model!
    stateOf(h).files.set(path, this.weightsOnly ? { kind: 'weights', values: m.weightValues() } : m.saved())
  }
}

class TerminateOnNaN extends KCallback {
  constructor() { super('TerminateOnNaN') }
  onBatchEnd(h: Host, batch: number, logs: Logs) {
    const l = logs.loss
    if (l !== undefined && !Number.isFinite(l)) {
      h.print(`Batch ${batch}: Invalid loss, terminating training\n`)
      this.model!.stopTraining = true
    }
  }
}

class LambdaCallback extends KCallback {
  fns: Params
  constructor(fns: Params) {
    super('LambdaCallback')
    this.fns = fns
  }
  private fire(h: Host, name: string, args: Value[]) {
    const f = this.fns[name]
    if (!isNone(f)) h.call(f, args)
  }
  onTrainBegin(h: Host, logs: Logs) { this.fire(h, 'on_train_begin', [logsDict(logs)]) }
  onTrainEnd(h: Host, logs: Logs) { this.fire(h, 'on_train_end', [logsDict(logs)]) }
  onEpochBegin(h: Host, e: number, logs: Logs) { this.fire(h, 'on_epoch_begin', [py.int(e), logsDict(logs)]) }
  onEpochEnd(h: Host, e: number, logs: Logs) { this.fire(h, 'on_epoch_end', [py.int(e), logsDict(logs)]) }
  onBatchBegin(h: Host, b: number, logs: Logs) { this.fire(h, 'on_train_batch_begin', [py.int(b), logsDict(logs)]) }
  onBatchEnd(h: Host, b: number, logs: Logs) { this.fire(h, 'on_train_batch_end', [py.int(b), logsDict(logs)]) }
}

const CALLBACK_STATE = 'keras.callback'

const USER_HOOKS: Record<string, string[]> = {
  onTrainBegin: ['on_train_begin'], onTrainEnd: ['on_train_end'], onEpochBegin: ['on_epoch_begin'], onEpochEnd: ['on_epoch_end'],
  onBatchBegin: ['on_train_batch_begin', 'on_batch_begin'], onBatchEnd: ['on_train_batch_end', 'on_batch_end'],
  onTestBegin: ['on_test_begin'], onTestEnd: ['on_test_end'], onPredictBegin: ['on_predict_begin'], onPredictEnd: ['on_predict_end'],
}

/** a user's class Foo(keras.callbacks.Callback): its logs go through a Python dict, and what it adds to logs is kept */
const userCallback = (h: Host, inst: InstValue): CallbackHooks => {
  for (const n of ['on_test_batch_begin', 'on_test_batch_end', 'on_predict_batch_begin', 'on_predict_batch_end']) {
    if (userMethod(inst.cls, n)) throw h.err('NotImplementedError', `Callback.${n} is not implemented in this sandbox: evaluation and prediction run in one pass`)
  }
  const fire = (hh: Host, names: string[], before: Value[], logs: Logs) => {
    for (const n of names) {
      const f = userMethod(inst.cls, n)
      if (!f) continue
      const d = logsDict(logs)
      hh.call({ k: 'bound', self: inst, f }, [...before, d])
      if (d.k === 'dict') for (const [k, v] of d.d.items()) if (v.k === 'int' || v.k === 'float') logs[hh.str(k)] = v.v
      return
    }
  }
  const hooks: CallbackHooks = {
    setup(model, p) {
      inst.attrs.set('model', model.self())
      inst.attrs.set('params', { k: 'dict', d: PyDict.from([[py.str('verbose'), py.int(p.verbose)], [py.str('epochs'), py.int(p.epochs)], [py.str('steps'), py.int(p.steps)]]) })
    },
  }
  const set = hooks as unknown as Record<string, unknown>
  for (const [hook, names] of Object.entries(USER_HOOKS)) {
    set[hook] = hook === 'onEpochBegin' || hook === 'onEpochEnd' || hook === 'onBatchBegin' || hook === 'onBatchEnd'
      ? (hh: Host, i: number, logs: Logs) => fire(hh, names, [py.int(i)], logs)
      : (hh: Host, logs: Logs) => fire(hh, names, [], logs)
  }
  return hooks
}

const callbackHooks = (h: Host, v: Value): CallbackHooks => {
  if (v.k === 'obj' && v.o instanceof KCallback) return v.o
  if (v.k === 'inst' && v.state[CALLBACK_STATE]) return userCallback(h, v)
  throw h.err('TypeError', `Callback must be an instance of keras.callbacks.Callback, got ${repr(v)}`)
}

const hooksOf = (h: Host, v: Value): CallbackHooks[] => (isNone(v) ? [] : (isSeq(v) ? h.iterate(v) : [v]).map((c) => callbackHooks(h, c)))

class History extends PyObj {
  readonly cls = 'History'
  history: Record<string, number[]>
  epochs: number[]
  params: { epochs: number; steps: number; verbose: number }
  model: KModel
  constructor(history: Record<string, number[]>, epochs: number[], params: { epochs: number; steps: number; verbose: number }, model: KModel) {
    super()
    this.history = history
    this.epochs = epochs
    this.params = params
    this.model = model
  }
  repr() { return '<keras.src.callbacks.history.History object>' }
  getAttr(name: string) {
    if (name === 'history') return { k: 'dict' as const, d: PyDict.from(Object.entries(this.history).map(([k, v]) => [py.str(k), py.list(v.map((x) => py.float(f32clean(Math.fround(x)))))])) }
    if (name === 'epoch') return py.list(this.epochs.map((i) => py.int(i)))
    if (name === 'params') return { k: 'dict' as const, d: PyDict.from([[py.str('verbose'), py.int(this.params.verbose)], [py.str('epochs'), py.int(this.params.epochs)], [py.str('steps'), py.int(this.params.steps)]]) }
    if (name === 'model') return this.model.self()
    return undefined
  }
}

/** Keras's log numbers: 4 decimals, or scientific below 1e-3 */
const kfmt = (v: number) => (Math.abs(v) > 1e-3 ? v.toFixed(4) : v.toExponential(4).replace(/e([+-])(\d)$/, 'e$10$2'))

// ---------------------------------------------------------------- tf.data

const specOf = (v: Value): string => {
  if (isSeq(v)) {
    const inner = v.items.map(specOf).join(', ')
    return v.k === 'tuple' ? `(${inner}${v.items.length === 1 ? ',' : ''})` : `[${inner}]`
  }
  if (v.k === 'dict') return `{${v.d.items().map(([k, x]) => `${repr(k)}: ${specOf(x)}`).join(', ')}}`
  const t = tfOf(v)
  const dt = v.k === 'obj' && v.o instanceof TFString ? 'string' : t?.dtype ?? 'float32'
  const base: KShape = t ? t.shape : v.k === 'obj' && v.o instanceof TFString ? v.o.shape : []
  return `TensorSpec(shape=${shapeStr(base)}, dtype=tf.${dt}, name=None)`
}

const mapStruct = (v: Value, f: (x: Value) => Value): Value => {
  if (v.k === 'tuple') return py.tuple(v.items.map((x) => mapStruct(x, f)))
  if (v.k === 'dict') return { k: 'dict', d: PyDict.from(v.d.items().map(([k, x]) => [k, mapStruct(x, f)])) }
  return f(v)
}

/** Stack dataset elements into a batch */
const stackStruct = (h: Host, items: Value[]): Value => {
  const first = items[0]
  if (first.k === 'tuple') return py.tuple(first.items.map((_, i) => stackStruct(h, items.map((it) => (it as { items: Value[] }).items[i]))))
  if (first.k === 'dict') return { k: 'dict', d: PyDict.from(first.d.keys().map((k) => [k, stackStruct(h, items.map((it) => (it as { d: PyDict }).d.get(k)!))])) }
  if (first.k === 'obj' && first.o instanceof TFString) return py.obj(new TFString(items.flatMap((it) => (it.k === 'obj' && it.o instanceof TFString ? it.o.strs : [])), [items.length]))
  const ts = items.map((x) => tfOf(x)!)
  for (const t of ts) if (t.shape.join() !== ts[0].shape.join()) throw tfErr(h, `Cannot batch tensors with different shapes in component 0. First element had shape [${ts[0].shape.join(',')}] and element ${ts.indexOf(t)} had shape [${t.shape.join(',')}].`)
  return tfWrap(TT.of(ts.flatMap((t) => t.values()), [ts.length, ...ts[0].shape], ts[0].dtype))
}

/** element_spec with a batch dimension in front of every shape */
const batchedSpec = (spec: string, d: number | null): string => spec.replace(/shape=\(([^)]*)\)/g, (_m, inner: string) => {
  const parts = [d === null ? 'None' : String(d), ...inner.split(',').map((x) => x.trim()).filter(Boolean)]
  return `shape=(${parts.join(', ')}${parts.length === 1 ? ',' : ''})`
})

/** tf.data.Dataset: a generator of elements, its element_spec, and how many there are */
class TFDataset extends PyObj {
  readonly cls: string
  gen: (h: Host) => Generator<Value>
  spec: () => string
  infinite: boolean
  size: number | null
  constructor(cls: string, gen: (h: Host) => Generator<Value>, spec: () => string, infinite: boolean, size: number | null) {
    super()
    this.cls = cls
    this.gen = gen
    this.spec = spec
    this.infinite = infinite
    this.size = size
  }
  repr() { return `<${this.cls} element_spec=${this.spec()}>` }
  items(h: Host): Value[] {
    if (this.infinite) throw h.err('ValueError', 'This dataset repeats forever: use .take(n) before iterating it in this sandbox')
    return [...this.gen(h)]
  }
  iter(h: Host) { return this.items(h) }
  len(h: Host) {
    if (this.infinite) throw h.err('TypeError', 'The dataset is infinite.')
    if (this.size === null) throw h.err('TypeError', 'The dataset length is unknown.')
    return this.size
  }
  /** the first component of each element, for layers' adapt(): a batch is a tensor or a tuple */
  batches(h: Host): Value[] { return this.items(h).map((b) => (b.k === 'tuple' ? b.items[0] : b)) }
  getAttr(name: string, h: Host): Value | undefined {
    const parent = this.gen
    const fn = (call: (args: Value[], kw: Kw) => Value) => fnv(name, call)
    switch (name) {
      case 'batch': return fn((args, kw) => {
        const P = bindArgs(h, 'batch', 'batch_size drop_remainder=False num_parallel_calls=None deterministic=None name=None', args, kw, null)
        const n = h.toInt(P.batch_size)
        if (n <= 0) throw tfErr(h, 'batch_size must be greater than zero.')
        const drop = h.truthy(P.drop_remainder)
        return py.obj(new TFDataset('_BatchDataset', function* (hh) {
          let buf: Value[] = []
          for (const x of parent(hh)) {
            buf.push(x)
            if (buf.length === n) {
              yield stackStruct(hh, buf)
              buf = []
            }
          }
          if (buf.length && !drop) yield stackStruct(hh, buf)
        }, () => batchedSpec(this.spec(), drop ? n : null), this.infinite, this.size === null ? null : drop ? Math.floor(this.size / n) : Math.ceil(this.size / n)))
      })
      case 'shuffle': return fn((args, kw) => {
        const P = bindArgs(h, 'shuffle', 'buffer_size seed=None reshuffle_each_iteration=True name=None', args, kw, null)
        const buffer = h.toInt(P.buffer_size)
        if (buffer < 1) throw tfErr(h, 'buffer_size must be greater than zero.')
        const again = h.truthy(P.reshuffle_each_iteration)
        // reshuffling draws afresh each pass; otherwise every pass replays one seed
        const seed = P.seed.k === 'none' ? (again ? null : Math.floor(h.rng.next() * 2 ** 31)) : h.toInt(P.seed)
        const stream = seed !== null && again ? new SandboxRandom(seed) : null
        return py.obj(new TFDataset('_ShuffleDataset', function* (hh) {
          const r: SandboxRandom = stream ?? (seed !== null ? new SandboxRandom(seed) : hh.rng)
          // a fill-and-draw buffer, as tf.data's: buffer_size = 1 keeps the order
          const buf: Value[] = []
          const draw = () => {
            const j = Math.floor(r.next() * buf.length)
            const x = buf[j]
            buf[j] = buf[buf.length - 1]
            buf.pop()
            return x
          }
          for (const x of parent(hh)) {
            buf.push(x)
            if (buf.length >= buffer) yield draw()
          }
          while (buf.length) yield draw()
        }, this.spec, this.infinite, this.size))
      })
      case 'map': return fn((args, kw) => {
        const f = args[0] ?? kw.map_func
        const apply = (hh: Host, x: Value) => hh.call(f, x.k === 'tuple' ? x.items : [x])
        return py.obj(new TFDataset('_MapDataset', function* (hh) {
          for (const x of parent(hh)) yield apply(hh, x)
        }, () => {
          const it = parent(h).next()
          return it.done ? 'TensorSpec(shape=<unknown>, dtype=tf.float32, name=None)' : specOf(apply(h, it.value))
        }, this.infinite, this.size))
      })
      case 'filter': return fn((args, kw) => {
        const f = args[0] ?? kw.predicate
        return py.obj(new TFDataset('_FilterDataset', function* (hh) {
          for (const x of parent(hh)) if (hh.truthy(hh.call(f, x.k === 'tuple' ? x.items : [x]))) yield x
        }, this.spec, this.infinite, null))
      })
      case 'take': case 'skip': return fn((args, kw) => {
        const n = h.toInt(args[0] ?? kw.count)
        const take = name === 'take'
        const size = this.size === null ? null : take ? (n < 0 ? this.size : Math.min(n, this.size)) : Math.max(0, this.size - n)
        return py.obj(new TFDataset(take ? '_TakeDataset' : '_SkipDataset', function* (hh) {
          let i = 0
          if (take && n === 0) return
          for (const x of parent(hh)) {
            if (take || i >= n) yield x
            i++
            if (take && n >= 0 && i >= n) return
          }
        }, this.spec, this.infinite && !(take && n >= 0), size))
      })
      case 'repeat': return fn((args, kw) => {
        const cv = args[0] ?? kw.count
        const count = isNone(cv) ? -1 : h.toInt(cv!)
        return py.obj(new TFDataset('_RepeatDataset', function* (hh) {
          for (let i = 0; count < 0 || i < count; i++) {
            let any = false
            for (const x of parent(hh)) {
              any = true
              yield x
            }
            if (!any) return
          }
        }, this.spec, count < 0 || this.infinite, count < 0 || this.size === null ? null : this.size * count))
      })
      case 'prefetch': case 'cache': return fn(() => py.obj(new TFDataset(name === 'cache' ? '_CacheDataset' : '_PrefetchDataset', parent, this.spec, this.infinite, this.size)))
      case 'unbatch': return fn(() => py.obj(new TFDataset('_UnbatchDataset', function* (hh) {
        const E = stateOf(hh).E
        for (const x of parent(hh)) {
          const parts = x.k === 'tuple' ? x.items : [x]
          const un = parts.map((p) => unstackT(E, tfOf(p)!, 0).map(tfWrap))
          for (let i = 0; i < un[0].length; i++) yield x.k === 'tuple' ? py.tuple(un.map((u) => u[i])) : un[0][i]
        }
      }, () => this.spec().replace(/shape=\(([^)]*)\)/g, (_m, inner: string) => {
        const parts = inner.split(',').map((x) => x.trim()).filter(Boolean).slice(1)
        return `shape=(${parts.join(', ')}${parts.length === 1 ? ',' : ''})`
      }), this.infinite, null)))
      case 'as_numpy_iterator': return fn(() => py.list(this.items(h).map((x) => mapStruct(x, (t) => {
        const tt = tfOf(t)
        return tt ? (tt.ndim ? py.arr(ndFor(tt)) : tfNumpyScalar(tt)) : t
      }))))
      case 'cardinality': return fn(() => tfWrap(TT.scalar(this.infinite ? -1 : this.size ?? -2, 'int64')))
      case 'element_spec': return py.str(this.spec())
    }
    return undefined
  }
}

/** tf.TensorSpec(shape, dtype): what from_generator declares */
class TensorSpec extends PyObj {
  readonly cls = 'TensorSpec'
  shape: KShape | null
  dtype: TDType | 'string'
  name: Value
  constructor(shape: KShape | null, dtype: TDType | 'string', name: Value) {
    super()
    this.shape = shape
    this.dtype = dtype
    this.name = name
  }
  repr() { return `TensorSpec(shape=${this.shape ? shapeStr(this.shape) : '<unknown>'}, dtype=tf.${this.dtype}, name=${repr(this.name)})` }
  getAttr(name: string): Value | undefined {
    if (name === 'shape') return py.obj(new TensorShape(this.shape ?? []))
    if (name === 'dtype') return py.obj(new TFDtype(this.dtype))
    if (name === 'name') return this.name
    return undefined
  }
}

const tfNumpyScalar = (t: TT): Value => (isFloatT(t.dtype) ? py.float(f32clean(t.values()[0])) : t.dtype === 'bool' ? py.bool(t.values()[0] !== 0) : py.int(t.values()[0]))

// ---------------------------------------------------------------- models

type ModelMode = 'sequential' | 'functional' | 'subclass'

interface MetricSlot { output: number; key: string; m: KMetric }

interface Compiled {
  opt: KOptimizer
  lossArg: Value
  metricsArg: Value
  lossWeights: Value
  specs: (LossSpec | null)[] | null
  slots: MetricSlot[] | null
  weights: number[]
}

/** the values fit reads from an array: flattened once, so a batch only copies rows */
interface Column { t: TT; v: number[] }
const column = (t: TT): Column => ({ t, v: t instanceof StrT ? [] : t.values() })
const rowsOf = (c: Column, idx: number[]): TT => {
  const t = c.t
  if (t instanceof StrT) return new StrT(idx.map((i) => t.strs[i]))
  const rest = t.shape.slice(1)
  const w = prod(rest)
  const out = new Array<number>(idx.length * w)
  idx.forEach((r, k) => {
    for (let j = 0; j < w; j++) out[k * w + j] = c.v[r * w + j]
  })
  return TT.raw(out, [idx.length, ...rest], t.dtype)
}
const sliceRows = (t: TT, a: number, b: number) => rowsOf(column(t), allOf(b - a).map((i) => a + i))

interface Batch { xs: TT[]; ys: TT[]; sw: TT | null }

/** the data fit and evaluate go through */
interface Source {
  /** the batches of one epoch */
  epoch(shuffle: boolean): Iterable<Batch>
  steps: number
  /** arrays can be scored in one forward pass */
  whole: Batch | null
}

const sortedLogs = (logs: Logs): Logs => Object.fromEntries(Object.entries(logs).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)))

const fmtCount = (n: number) => n.toLocaleString('en-US')

/** rich's table: greedy word wrap, and a word too long for the column ends in an ellipsis */
const wrapCell = (s: string, w: number): string[] => {
  const words = s.match(/\S+\s*/g) ?? ['']
  const lines: string[] = []
  let cur = ''
  for (const word of words) {
    if (cur.length + word.trimEnd().length <= w) cur += word
    else {
      if (cur) lines.push(cur)
      cur = word
    }
  }
  lines.push(cur)
  return lines.map((l) => {
    const t = l.trimEnd()
    return t.length > w ? `${t.slice(0, w - 1)}…` : t
  })
}

/** keras.src.utils.summary_utils.readable_memory_size */
const readableBytes = (n: number): string => {
  let b = n
  for (const u of ['B', 'KB', 'MB', 'GB', 'TB']) {
    if (b / 1024 < 1) return `${b.toFixed(2)} ${u}`
    b /= 1024
  }
  return `${b.toFixed(2)} PB`
}

class KModel extends KLayer {
  mode: ModelMode
  layers: KLayer[] = []
  inputsK: KerasTensor[] = []
  outputsK: KerasTensor[] = []
  /** layers already wired into a Sequential's graph */
  attached = 0
  /** Functional: its nodes, by depth, and whether the summary needs a Connected to column */
  relevant = new Set<KNode>()
  sequentialLike = true
  compiled: Compiled | null = null
  stopTraining = false
  historyObj: History | null = null
  lossT = { total: 0, count: 0 }
  outLossT: { total: number; count: number }[] = []
  /** a subclassed model's number of outputs, once it has run */
  nOut = 1
  constructor(h: Host, mode: ModelMode, kw: Kw, kind?: string) {
    super(h, kind ?? (mode === 'sequential' ? 'Sequential' : mode === 'functional' ? 'Functional' : 'Model'), kw)
    this.mode = mode
  }
  isa(name: string) { return name === 'Model' || name === 'Layer' || name === this.kind }
  children(): KLayer[] { return this.mode === 'subclass' ? attrLayers(this.inst!) : this.layers }
  /** the layers Keras lists: Functional models include their InputLayers */
  listed(): KLayer[] { return this.mode === 'subclass' ? attrLayers(this.inst!) : this.layers }

  // ---- building
  setInput(h: Host, s: KShape, name?: Value) {
    const inp = new InputLayer(h, s, { name: name ?? py.str('input_layer') })
    const node = new KNode(inp, [], {})
    node.outputs = [new KerasTensor(s, node, 0, 'keras_tensor')]
    this.inputsK = [node.outputs[0]]
    this.outputsK = [node.outputs[0]]
    this.inShape = s
  }
  add(h: Host, v: Value) {
    const kt = kerasOf(v)
    const given = kt && kt.layer instanceof InputLayer ? kt.layer : v.k === 'obj' && v.o instanceof InputLayer ? v.o : null
    if (given) {
      if (this.layers.length) throw h.err('ValueError', 'Sequential model has already been configured to use input shape; a further Input cannot be added.')
      this.setInput(h, given.shape, py.str(given.name))
      this.outShape = given.shape
      this.built = true
      return
    }
    const l = layerOf(v)
    if (!l) throw h.err('TypeError', `The added layer must be an instance of class Layer. Received: layer=${repr(v)} of type ${v.k === 'obj' ? v.o.cls : v.k}.`)
    if (this.layers.some((x) => x.name === l.name)) throw h.err('ValueError', `All layers added to a Sequential model should have unique names. Name '${l.name}' is already the name of a layer in this model. Update the \`name\` argument to pass a unique name.`)
    if (!this.inShape && !this.layers.length && l.declaredInput) this.setInput(h, l.declaredInput)
    this.layers.push(l)
    if (this.inShape) this.wire(h)
  }
  /** propagate shapes through the stack, creating weights and the graph behind .input and .output */
  wire(h: Host) {
    let cur = this.outputsK[0]
    for (; this.attached < this.layers.length; this.attached++) {
      const l = this.layers[this.attached]
      const shapes = l.outShapes(h, [cur.shape], {})
      const node = new KNode(l, [cur], {})
      node.outputs = shapes.map((s, i) => new KerasTensor(s, node, i, autoName(h, 'keras_tensor'), l.outputDtype()))
      cur = node.outputs[0]
    }
    this.outputsK = [cur]
    this.outShape = cur.shape
    this.built = true
  }
  build(h: Host, s: KShape): KShape {
    if (this.mode === 'sequential') {
      if (!this.inShape) this.setInput(h, s)
      this.wire(h)
      return this.outShape!
    }
    return this.outShapes(h, [s], {})[0]
  }
  ensureBuilt(h: Host, s: KShape): KShape {
    this.outShapes(h, [s], {})
    return this.outShape!
  }
  outShapes(h: Host, shapes: KShape[], kw: Kw): KShape[] {
    if (this.mode === 'sequential') {
      if (!this.built || !this.inShape) this.build(h, shapes[0])
      else this.checkShape(h, shapes[0], this.inShape, 0)
      return [this.outShape!]
    }
    if (this.mode === 'functional') {
      if (shapes.length !== this.inputsK.length) throw h.err('ValueError', `Layer "${this.name}" expects ${this.inputsK.length} input(s), but it received ${shapes.length} input tensors.`)
      shapes.forEach((s, i) => this.checkShape(h, s, this.inputsK[i].shape, i))
      return this.outputsK.map((t) => t.shape)
    }
    if (!this.built) userBuild(h, this.inst!, shapes[0])
    this.built = true
    this.inShape = shapes[0]
    const cos = userMethod(this.inst!.cls, 'compute_output_shape')
    const out = cos ? [h.iterate(h.call({ k: 'bound', self: this.inst!, f: cos }, [shapeTuple(shapes[0])])).map((d) => (d.k === 'none' ? null : h.toInt(d)))] : probeShapes(h, this, shapes, kw)
    this.outShape = out[0]
    return out
  }
  checkShape(h: Host, got: KShape, want: KShape, i: number) {
    if (got.length !== want.length || want.some((d, k) => k > 0 && d !== null && got[k] !== null && d !== got[k])) {
      const who = this.mode === 'sequential' ? this.layers[0]?.name ?? this.name : this.name
      throw h.err('ValueError', `Input ${i} of layer "${who}" is incompatible with the layer: expected shape=${shapeStr(want)}, found shape=${shapeStr(got)}`)
    }
  }
  /** a batch of data: build if needed, line up its rank with the model's inputs, as Functional._adjust_input_rank does */
  prepare(h: Host, xs: TT[]): TT[] {
    const E = stateOf(h).E
    if (this.mode === 'subclass') {
      if (!this.built) this.dryRun(h, xs)
      return xs
    }
    if (this.mode === 'sequential' && !this.built) this.build(h, [null, ...xs[0].shape.slice(1)])
    if (this.mode === 'functional' && xs.length !== this.inputsK.length) throw h.err('ValueError', `Layer "${this.name}" expects ${this.inputsK.length} input(s), but it received ${xs.length} input tensors.`)
    const wants = this.mode === 'sequential' ? [this.inShape!] : this.inputsK.map((t) => t.shape)
    return xs.map((x, i) => {
      const want = wants[i]
      const fixed = x.ndim === want.length - 1 && want[want.length - 1] === 1 ? C.unsqueezeT(E, x, -1) : x.ndim === want.length + 1 && x.shape[x.ndim - 1] === 1 ? C.squeezeT(E, x, -1) : x
      this.checkShape(h, [null, ...fixed.shape.slice(1)], want, i)
      return fixed
    })
  }
  /** a subclassed model creates its weights by running once */
  dryRun(h: Host, xs: TT[]) {
    const S = stateOf(h)
    const was = S.E.grad
    const nLoss = S.losses.length
    S.E.grad = false
    this.inShape = [null, ...xs[0].shape.slice(1)]
    userBuild(h, this.inst!, this.inShape)
    this.built = true
    try {
      const one = xs.map((x) => (x instanceof StrT ? new StrT(x.strs.slice(0, 1)) : sliceRows(x, 0, Math.min(1, x.shape[0]))))
      const out = this.invoke(S.E, one, false, h)
      this.nOut = out.length
      this.outShape = [null, ...out[0].shape.slice(1)]
    } finally {
      S.E.grad = was
      S.losses.length = nLoss
    }
  }

  // ---- the Functional graph
  /** keras.src.ops.function.map_graph: the operations by depth, in traversal order within a depth */
  mapGraph(h: Host) {
    const finished = new Set<KNode>()
    const progress = new Set<KNode>()
    const order: KNode[] = []
    const opIndex = new Map<KLayer, number>()
    const inputs = new Set(this.inputsK)
    const visit = (t: KerasTensor) => {
      const node = t.node
      if (finished.has(node)) return
      if (progress.has(node)) throw h.err('ValueError', `Tensor ${t.name} from operation '${node.layer.name}' is part of a cycle.`)
      if (!opIndex.has(node.layer)) opIndex.set(node.layer, opIndex.size)
      progress.add(node)
      if (!inputs.has(t)) {
        if (node.layer instanceof InputLayer) throw h.err('ValueError', `Graph disconnected: cannot find parent for tensor ${t.name} (the output of the InputLayer '${node.layer.name}'), which is not among the model's inputs.`)
        node.inputs.forEach(visit)
      }
      finished.add(node)
      progress.delete(node)
      order.push(node)
    }
    this.outputsK.forEach(visit)
    const nodeDepth = new Map<KNode, number>()
    const opDepth = new Map<KLayer, number>()
    for (let i = order.length - 1; i >= 0; i--) {
      const node = order[i]
      const depth = Math.max(nodeDepth.get(node) ?? 0, opDepth.get(node.layer) ?? 0)
      opDepth.set(node.layer, depth)
      nodeDepth.set(node, depth)
      for (const t of node.inputs) if (finished.has(t.node)) nodeDepth.set(t.node, Math.max(depth + 1, nodeDepth.get(t.node) ?? 0))
    }
    for (const t of this.inputsK) {
      if (!opDepth.has(t.layer)) {
        opDepth.set(t.layer, 0)
        opIndex.set(t.layer, -1)
        nodeDepth.set(t.node, 0)
      }
    }
    this.relevant = new Set(nodeDepth.keys())
    const depths = [...new Set(opDepth.values())].sort((a, b) => b - a)
    this.layers = depths.flatMap((d) => [...opDepth].filter(([, x]) => x === d).map(([l]) => l).sort((a, b) => opIndex.get(a)! - opIndex.get(b)!))
    // keras.src.utils.summary_utils: a chain gets three columns, anything else a Connected to column
    const byDepth = new Map<number, KNode[]>()
    for (const [n, d] of nodeDepth) byDepth.set(d, [...(byDepth.get(d) ?? []), n])
    this.sequentialLike = [...byDepth.values()].every((v) => v.length <= 1 && (v.length !== 1 || v[0].inputs.length <= 1))
    if (this.sequentialLike) for (const l of this.layers) if (l.nodes.filter((n) => this.relevant.has(n)).length > 1) this.sequentialLike = false
    this.built = true
    this.inShape = this.inputsK[0].shape
    this.outShape = this.outputsK[0].shape
  }
  runGraph(E: Engine, xs: TT[], training: boolean | null, h: Host): TT[] {
    const known = new Map<KerasTensor, TT>(this.inputsK.map((t, i) => [t, xs[i]]))
    const done = new Map<KNode, TT[]>()
    const evalNode = (n: KNode): TT[] => {
      const seen = done.get(n)
      if (seen) return seen
      const ins = n.inputs.map(value)
      const out = n.layer.invoke(E, ins, !isNone(n.kw.training) ? h.truthy(n.kw.training!) : training, h, n.kw)
      done.set(n, out)
      return out
    }
    const value = (t: KerasTensor): TT => known.get(t) ?? evalNode(t.node)[t.outIndex]
    return this.outputsK.map(value)
  }

  // ---- forward
  run(E: Engine, xs: TT[], _training: boolean, h: Host, kw: Kw): TT[] {
    // what the caller said, which may be nothing: the flag invoke left in force
    const training = stateOf(h).training
    if (this.mode === 'sequential') {
      let y = xs[0]
      for (const l of this.layers) y = l.invoke(E, [y], training, h)[0]
      return [y]
    }
    if (this.mode === 'functional') return this.runGraph(E, xs, training, h)
    const rest = Object.fromEntries(Object.entries(kw).filter(([k]) => k !== 'training'))
    return flatTensors(h, callUser(h, this.inst!, xs.length > 1 ? py.list(xs.map(tfWrap)) : tfWrap(xs[0]), training, rest), `${this.kind}.call()`)
  }
  forward(E: Engine, x: TT, training: boolean, h: Host): TT { return this.run(E, [x], training, h, {})[0] }

  call(args: Value[], kw: Kw, h: Host): Value {
    if (this.mode === 'subclass') return userEagerCall(this, h, args, kw)
    const x = args[0] ?? kw.inputs
    if (!x) throw h.err('TypeError', `${this.kind}.call() missing 1 required positional argument: 'inputs'`)
    if (kerasOf(x) || (isSeq(x) && x.items.some((v) => kerasOf(v)))) return super.call(args, kw, h)
    const S = stateOf(h)
    const training = !isNone(kw.training) ? h.truthy(kw.training!) : S.training
    const out = this.invoke(S.E, this.prepare(h, this.inputsOf(h, x)), training, h)
    return out.length === 1 ? tfWrap(out[0]) : py.list(out.map(tfWrap))
  }

  // ---- saving and restoring weights
  weightValues(): number[][] { return this.weights().map((w) => w.t.values()) }
  setWeightValues(vs: number[][]) { this.weights().forEach((w, i) => C.writeInto(w.t, vs[i])) }
  trainableTs(): TT[] { return unique(this.weights().filter((w) => w.trainable).map((w) => w.t)) }

  // ---- summary
  rowShape(l: KLayer): string {
    const n = l.nodes[l.nodes.length - 1]
    if (n && n.outputs.length > 1) return `[${n.outputs.map((t) => shapeStr(t.shape)).join(', ')}]`
    return l.outShape ? shapeStr(l.outShape) : '?'
  }
  connected(l: KLayer): string {
    const parts = l.nodes.filter((n) => this.relevant.has(n)).flatMap((n) => n.inputs.map((t) => `${t.layer.name}[${t.node.index}][${t.outIndex}]`))
    return parts.length ? parts.join(', ') : '-'
  }
  summary(h: Host) {
    const rows = this.listed()
    const fourCols = this.mode === 'functional' && !this.sequentialLike
    // the widths rich's table reaches in an 80-column terminal, as Keras asks for it
    const W = fourCols ? [21, 19, 12, 19] : [33, 24, 15]
    const heads = fourCols ? ['Layer (type)', 'Output Shape', 'Param #', 'Connected to'] : ['Layer (type)', 'Output Shape', 'Param #']
    const pad = (s: string, w: number, right: boolean) => ` ${right ? s.padStart(w - 2) : s.padEnd(w - 2)} `
    const line = (l: string, m: string, r: string, fill: string) => l + W.map((w) => fill.repeat(w)).join(m) + r
    const out = [`Model: "${this.name}"`, line('┏', '┳', '┓', '━'), `┃${heads.map((c, i) => pad(c, W[i], i === 2)).join('┃')}┃`, line('┡', '╇', '┩', '━')]
    rows.forEach((l, i) => {
      const cells = [`${l.name} (${l.kind})`, this.rowShape(l), l.built ? fmtCount(l.params()) : '0 (unbuilt)', ...(fourCols ? [this.connected(l)] : [])].map((c, k) => wrapCell(c, W[k] - 2))
      const height = Math.max(...cells.map((c) => c.length))
      for (let k = 0; k < height; k++) out.push(`│${cells.map((c, j) => pad(c[k] ?? '', W[j], j === 2)).join('│')}│`)
      out.push(i === rows.length - 1 ? line('└', '┴', '┘', '─') : line('├', '┼', '┤', '─'))
    })
    if (!rows.length) out.push(line('└', '┴', '┘', '─'))
    const ws = this.weights()
    const tr = ws.filter((w) => w.trainable)
    const non = ws.filter((w) => !w.trainable)
    const count = (l: KWeight[]) => l.reduce((s, w) => s + w.t.size, 0)
    const bytes = (l: KWeight[]) => l.reduce((s, w) => s + w.t.size * w.bytes, 0)
    const opt = this.compiled?.opt.built ? this.compiled.opt.counts(count(ws.filter((w) => w.trainable))) : null
    const total = count(ws) + (opt?.n ?? 0)
    const totalBytes = bytes(ws) + (opt?.bytes ?? 0)
    out.push(` Total params: ${fmtCount(total)} (${readableBytes(totalBytes)})`, ` Trainable params: ${fmtCount(count(tr))} (${readableBytes(bytes(tr))})`, ` Non-trainable params: ${fmtCount(count(non))} (${readableBytes(bytes(non))})`)
    if (opt) out.push(` Optimizer params: ${fmtCount(opt.n)} (${readableBytes(opt.bytes)})`)
    h.print(out.join('\n') + '\n')
    if (h.tracing && rows.length) {
      const flow: FlowRow[] = rows.map((l) => ({
        name: `${l.name} (${l.kind})`, layer: l.describe(), input: l.inShape ?? [], output: l.outShape ?? [], params: l.params(),
        note: l.inShape && l.outShape ? l.note(l.inShape, l.outShape) : null,
      }))
      h.emit({ type: 'flow', framework: 'keras', model: this.name, rows: flow, total: count(ws), trainable: count(tr), title: 'model.summary()' })
    }
  }
  describe() { return this.mode === 'sequential' ? `Sequential(${this.layers.length} layers)` : `${this.kind}(…)` }

  // ---- compile: losses and metrics per output
  outputNames(nOut: number): string[] {
    if (this.mode !== 'subclass' && this.outputsK.length) return this.outputsK.map((t) => t.layer.name)
    return allOf(nOut).map((i) => `output_${i + 1}`)
  }
  resolve(h: Host, nOut: number) {
    const c = this.compiled!
    if (c.specs && c.slots) return
    const names = this.outputNames(nOut)
    const pick = (v: Value, what: string): (Value | null)[] => {
      if (isNone(v)) return names.map(() => null)
      if (v.k === 'dict') return names.map((n) => v.d.get(py.str(n)) ?? null)
      if (isSeq(v)) {
        if (nOut > 1 && v.items.length !== nOut) throw h.err('ValueError', `For a model with multiple outputs, when providing the \`${what}\` argument as a list, it should have as many entries as the model has outputs. Received:\n${what}=${repr(v)}\nof length ${v.items.length} whereas the model has ${nOut} outputs.`)
        return nOut === 1 ? [what === 'metrics' ? v : v.items[0]] : v.items
      }
      return names.map(() => v)
    }
    c.specs = pick(c.lossArg, 'loss').map((v) => (v === null || v.k === 'none' ? null : lossSpec(h, v)))
    c.weights = pick(c.lossWeights, 'loss_weights').map((v) => (v === null ? 1 : h.num(v)))
    const slots: MetricSlot[] = []
    pick(c.metricsArg, 'metrics').forEach((v, i) => {
      if (v === null) return
      for (const m of isSeq(v) ? h.iterate(v) : [v]) {
        const km = metricArg(h, m)
        slots.push({ output: i, key: nOut > 1 ? `${names[i]}_${km.name}` : km.name, m: km })
      }
    })
    c.slots = slots
    this.outLossT = names.map(() => ({ total: 0, count: 0 }))
  }
  metricSlots(): MetricSlot[] { return this.compiled?.slots ?? [] }

  /** the loss Keras trains on and reports: compiled losses plus the layers' regularization and add_loss terms */
  computeLoss(E: Engine, h: Host, ys: TT[], outs: TT[], sw: TT | null, train: boolean): { total: TT; parts: number[] } {
    const c = this.compiled!
    this.resolve(h, outs.length)
    let total: TT | null = null
    const parts: number[] = []
    c.specs!.forEach((spec, i) => {
      if (!spec) {
        parts.push(0)
        return
      }
      if (ys[i] === undefined) throw h.err('ValueError', 'No loss found: the model has a compiled loss but fit() received no targets (y).')
      let l = reduceLossT(E, perSample(E, h, spec, ys[i], outs[i], train), sw, spec.reduction)
      if (c.weights[i] !== 1) l = mulS(E, l, c.weights[i])
      parts.push(l.values()[0])
      total = total ? C.binary(E, '+', total, l) : l
    })
    const S = stateOf(h)
    const extras = [...this.allLayers().flatMap((l) => l.penalties(E, h)), ...S.losses]
    for (const p of extras) total = total ? C.binary(E, '+', total, p) : p
    if (!total) throw h.err('ValueError', 'No loss found. You may have forgotten to provide a `loss` argument in the `compile()` method.')
    return { total, parts }
  }
  updateTrackers(h: Host, total: TT, parts: number[], n: number, ys: TT[], outs: TT[]) {
    this.lossT.total += total.values()[0] * n
    this.lossT.count += n
    parts.forEach((p, i) => {
      this.outLossT[i].total += p * n
      this.outLossT[i].count += n
    })
    for (const s of this.metricSlots()) if (ys[s.output] !== undefined) s.m.doUpdate(h, ys[s.output], outs[s.output], null)
  }
  resetMetrics(h: Host) {
    this.lossT = { total: 0, count: 0 }
    this.outLossT = this.outLossT.map(() => ({ total: 0, count: 0 }))
    for (const s of this.metricSlots()) s.m.doReset(h)
  }
  /** Keras sorts its logs: accuracy before loss */
  logs(h: Host): Logs {
    const out: Logs = { loss: this.lossT.count ? this.lossT.total / this.lossT.count : 0 }
    if (this.outLossT.length > 1) this.outputNames(this.outLossT.length).forEach((n, i) => { out[`${n}_loss`] = this.outLossT[i].count ? this.outLossT[i].total / this.outLossT[i].count : 0 })
    for (const s of this.metricSlots()) out[s.key] = s.m.value(h)
    return sortedLogs(out)
  }
  /** one training step */
  trainStep(h: Host, b: Batch, params: TT[]) {
    const S = stateOf(h)
    const E = S.E
    const c = this.compiled!
    for (const p of params) p.grad = null
    S.losses = []
    E.grad = true
    const outs = this.invoke(E, b.xs, true, h)
    const { total, parts } = this.computeLoss(E, h, b.ys, outs, b.sw, true)
    if (total.requiresGrad) {
      C.backward(E, total, null, false)
      c.opt.step(h, params)
    }
    this.updateTrackers(h, total, parts, b.xs[0].shape[0], b.ys, outs)
  }
  /** one evaluation pass over a batch */
  testStep(h: Host, b: Batch) {
    const S = stateOf(h)
    const E = S.E
    const was = E.grad
    E.grad = false
    S.losses = []
    try {
      const outs = this.invoke(E, b.xs, false, h)
      const { total, parts } = this.computeLoss(E, h, b.ys, outs, b.sw, false)
      this.updateTrackers(h, total, parts, b.xs[0].shape[0], b.ys, outs)
    } finally {
      E.grad = was
    }
  }

  // ---- data
  inputsOf(h: Host, v: Value | undefined): TT[] {
    if (isNone(v)) throw h.err('TypeError', 'expected input data')
    if (v!.k === 'dict') {
      const d = v!.d
      const names = this.inputsK.map((t) => t.layer.name)
      return names.map((n) => {
        const x = d.get(py.str(n))
        if (!x) throw h.err('ValueError', `Missing data for input "${n}". You passed a data dictionary with keys ${repr(py.list(d.keys()))}. Expected the following keys: ${repr(py.list(names.map(py.str)))}`)
        return asInput(h, x)
      })
    }
    if (isSeq(v) && v.items.length && (this.inputsK.length > 1 || v.items.every((x) => x.k === 'array' || tfOf(x)))) return v.items.map((x) => asInput(h, x))
    return [asInput(h, v)]
  }
  targetsOf(h: Host, v: Value | undefined, nOut: number): TT[] {
    if (isNone(v)) return []
    if (v!.k === 'dict') {
      const d = v!.d
      return this.outputNames(nOut).map((n) => asFloat(h, d.get(py.str(n))))
    }
    if (isSeq(v) && nOut > 1) return v.items.map((x) => asFloat(h, x))
    return [asFloat(h, v)]
  }
  nOutputs(h: Host, xs: TT[]): number {
    if (this.mode !== 'subclass') return Math.max(1, this.outputsK.length)
    if (!this.built) this.dryRun(h, xs)
    return this.nOut
  }
  /** a batch from an element of a dataset: x, (x, y) or (x, y, sample_weight) */
  batchFrom(h: Host, el: Value): Batch {
    const parts = el.k === 'tuple' ? el.items : [el]
    const xs = this.prepare(h, this.inputsOf(h, parts[0]))
    const ys = parts[1] ? this.targetsOf(h, parts[1], this.nOutputs(h, xs)) : []
    return { xs, ys, sw: parts[2] && parts[2].k !== 'none' ? asFloat(h, parts[2]) : null }
  }
  arraySource(h: Host, xs: TT[], ys: TT[], sw: TT | null, bs: number): Source {
    const xc = xs.map(column)
    const yc = ys.map(column)
    const wc = sw ? column(sw) : null
    const n = xs[0].shape[0]
    const whole: Batch = { xs, ys, sw }
    return {
      steps: Math.ceil(n / bs),
      whole,
      epoch: (shuffle: boolean) => ({
        * [Symbol.iterator]() {
          const order = allOf(n)
          if (shuffle) for (let i = n - 1; i > 0; i--) {
            const j = Math.floor(h.rng.next() * (i + 1))
            ;[order[i], order[j]] = [order[j], order[i]]
          }
          for (let s = 0; s < n; s += bs) {
            const idx = order.slice(s, s + bs)
            yield { xs: xc.map((c) => rowsOf(c, idx)), ys: yc.map((c) => rowsOf(c, idx)), sw: wc ? rowsOf(wc, idx) : null }
          }
        },
      }),
    }
  }
  datasetSource(h: Host, ds: TFDataset, stepsLimit: number | null): Source {
    let live: Generator<Value> | null = null
    const batch = (el: Value) => this.batchFrom(h, el)
    const ranOut = () => {
      this.stopTraining = true
      h.print('UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches.\n')
    }
    const steps = stepsLimit ?? ds.size ?? 0
    return {
      steps,
      whole: null,
      epoch: () => ({
        * [Symbol.iterator]() {
          // with steps_per_epoch the iterator carries on across epochs, as Keras's does
          const it = stepsLimit !== null ? (live ??= ds.gen(h)) : ds.gen(h)
          for (let i = 0; stepsLimit === null || i < stepsLimit; i++) {
            const next = it.next()
            if (next.done) {
              if (stepsLimit !== null) ranOut()
              return
            }
            yield batch(next.value)
          }
        },
      }),
    }
  }

  /** evaluate one source: metrics reset, logs back */
  testLogs(h: Host, src: Source): Logs {
    this.resetMetrics(h)
    // activity penalties and add_loss sum over a batch, so those models are scored batch by batch, as Keras does
    const perBatch = this.allLayers().some((l) => l.activity || l.inst)
    if (src.whole && !perBatch) this.testStep(h, src.whole)
    else for (const b of src.epoch(false)) this.testStep(h, b)
    return this.logs(h)
  }

  fit(h: Host, args: Value[], kw: Kw): Value {
    const P = bindArgs(h, 'fit', 'x=None y=None batch_size=None epochs=1 verbose=auto callbacks=None validation_split=0.0 validation_data=None shuffle=True class_weight=None sample_weight=None initial_epoch=0 steps_per_epoch=None validation_steps=None validation_batch_size=None validation_freq=1', args, kw, null)
    if (!this.compiled) throw h.err('RuntimeError', 'You must call `compile()` before using the model.')
    if (this.inst && userMethod(this.inst.cls, 'train_step')) throw h.err('NotImplementedError', 'a custom train_step() is not implemented in this sandbox: write the loop yourself with tf.GradientTape')
    const S = stateOf(h)
    const E = S.E
    const epochs = h.toInt(P.epochs)
    const start = h.toInt(P.initial_epoch)
    const verbose = P.verbose.k === 'str' ? 1 : h.toInt(P.verbose)
    if (P.shuffle.k === 'str') throw h.err('NotImplementedError', `shuffle='${P.shuffle.v}' is not implemented in this sandbox`)
    const shuffle = h.truthy(P.shuffle)
    const stepsLimit = isNone(P.steps_per_epoch) ? null : h.toInt(P.steps_per_epoch)
    const split = h.num(P.validation_split)
    let src: Source
    let val: Source | null = null
    const xv = P.x
    let dataset = xv.k === 'obj' && xv.o instanceof TFDataset ? xv.o : null
    if (xv.k === 'iter') {
      const next = h.iterOf(xv)
      dataset = new TFDataset('generator', function* () {
        for (let el = next(); el !== undefined; el = next()) yield el
      }, () => 'generator', true, null)
    }
    const vbs = !isNone(P.validation_batch_size) ? h.toInt(P.validation_batch_size) : !isNone(P.batch_size) ? h.toInt(P.batch_size) : 32
    const valFrom = (v: Value): Source => {
      if (v.k === 'obj' && v.o instanceof TFDataset) return this.datasetSource(h, v.o, isNone(P.validation_steps) ? null : h.toInt(P.validation_steps))
      if (!isNone(P.validation_steps)) throw h.err('NotImplementedError', 'fit: validation_steps with arrays is not implemented in this sandbox: pass a tf.data.Dataset')
      const parts = h.iterate(v)
      if (parts.length < 2) throw h.err('ValueError', `Data is expected to be in format \`x\`, \`(x,)\`, \`(x, y)\`, or \`(x, y, sample_weight)\`, found: ${repr(v)}`)
      const vx = this.prepare(h, this.inputsOf(h, parts[0]))
      const vy = this.targetsOf(h, parts[1], this.nOutputs(h, vx))
      return this.arraySource(h, vx, vy, parts[2] && parts[2].k !== 'none' ? asFloat(h, parts[2]) : null, vbs)
    }
    if (dataset) {
      if (!isNone(P.batch_size)) throw h.err('ValueError', `The \`batch_size\` argument must not be specified for the given input type. Received input: ${repr(xv)}, batch_size: ${repr(P.batch_size)}`)
      if (xv.k === 'iter' && stepsLimit === null) throw h.err('NotImplementedError', 'model.fit(generator) without steps_per_epoch is not implemented in this sandbox: a generator may never end')
      if (split > 0) throw h.err('ValueError', '`validation_split` is only supported for Tensors or NumPy arrays')
      if (!isNone(P.y)) throw h.err('ValueError', 'When providing `x` as a tf.data.Dataset, `y` should not be passed. Instead, the targets should be included as part of the tf.data.Dataset.')
      if (!isNone(P.sample_weight)) throw h.err('ValueError', '`sample_weight` argument is not supported when using a tf.data.Dataset as input.')
      // a generator is walked once: its first batch must not be lost to building the model, so only a dataset is peeked at
      if (xv.k !== 'iter') {
        const first = dataset.gen(h).next()
        if (first.done) throw h.err('ValueError', 'Unexpected result of `train_function` (Empty logs). This could be due to issues in input pipeline that resulted in an empty dataset.')
        this.batchFrom(h, first.value)
      }
      src = this.datasetSource(h, dataset, stepsLimit)
      if (!src.steps) src.steps = dataset.infinite ? 0 : dataset.items(h).length
    } else {
      if (stepsLimit !== null) throw h.err('NotImplementedError', 'steps_per_epoch with arrays is not implemented in this sandbox: pass a tf.data.Dataset with .repeat()')
      let xs = this.prepare(h, this.inputsOf(h, xv))
      const nOut = this.nOutputs(h, xs)
      let ys = this.targetsOf(h, P.y, nOut)
      for (const c of ys) if (xs[0].shape[0] !== c.shape[0]) throw h.err('ValueError', `Data cardinality is ambiguous. Make sure all arrays contain the same number of samples.'x' sizes: ${xs[0].shape[0]}\n'y' sizes: ${c.shape[0]}`)
      let sw = isNone(P.sample_weight) ? null : asFloat(h, P.sample_weight)
      if (!isNone(P.class_weight)) {
        if (P.class_weight.k !== 'dict') throw h.err('TypeError', 'class_weight must be a dict {class index: weight}')
        if (ys.length !== 1) throw h.err('ValueError', '`class_weight` is only supported for Models with a single output.')
        if (sw) throw h.err('ValueError', 'You cannot `class_weight` and `sample_weight` at the same time. If you want to use both, you must convert `class_weight` to `sample_weight` first.')
        const cw = P.class_weight.d
        const y = ys[0]
        const labels = y.ndim > 1 && y.shape[y.ndim - 1] > 1 ? rowArgmax(y) : y.values().map(Math.round)
        sw = TT.of(labels.map((l) => {
          const w = cw.get(py.int(l))
          return w ? h.num(w) : 1
        }), [labels.length], 'float32')
      }
      const bs = isNone(P.batch_size) ? 32 : h.toInt(P.batch_size)
      if (isNone(P.validation_data) && split > 0) {
        // Keras holds out the LAST fraction of the data, before shuffling
        const n = xs[0].shape[0]
        const cut = Math.floor(n * (1 - split))
        if (cut === 0 || cut === n) throw h.err('ValueError', `The model received too little data for validation_split=${split}: nothing left on one side of the split.`)
        val = this.arraySource(h, xs.map((t) => sliceRows(t, cut, n)), ys.map((t) => sliceRows(t, cut, n)), sw ? sliceRows(sw, cut, n) : null, vbs)
        xs = xs.map((t) => sliceRows(t, 0, cut))
        ys = ys.map((t) => sliceRows(t, 0, cut))
        sw = sw ? sliceRows(sw, 0, cut) : null
      }
      src = this.arraySource(h, xs, ys, sw, bs)
    }
    if (!isNone(P.validation_data)) val = valFrom(P.validation_data)
    const freq = h.toInt(P.validation_freq)
    const params = { epochs, steps: src.steps, verbose }
    const cbs = hooksOf(h, P.callbacks)
    for (const c of cbs) c.setup?.(this, params, h)
    this.stopTraining = false
    this.inst?.attrs.delete('stop_training')
    const hist: Record<string, number[]> = {}
    const epochsRun: number[] = []
    let last: Logs = {}
    for (const c of cbs) c.onTrainBegin?.(h, {})
    for (let ep = start; ep < epochs; ep++) {
      this.resetMetrics(h)
      for (const c of cbs) c.onEpochBegin?.(h, ep, {})
      if (verbose && epochs > 1) h.print(`Epoch ${ep + 1}/${epochs}\n`)
      const trainable = this.trainableTs()
      if (!trainable.length && ep === start) h.print('UserWarning: The model does not have any trainable weights.\n')
      let step = 0
      for (const b of src.epoch(shuffle)) {
        for (const c of cbs) c.onBatchBegin?.(h, step, {})
        this.trainStep(h, b, trainable)
        const bl = this.logs(h)
        for (const c of cbs) c.onBatchEnd?.(h, step, bl)
        step++
        if (this.stopTraining) break
      }
      if (!params.steps) params.steps = step
      const logs: Logs = { ...this.logs(h) }
      if (val && (ep + 1) % freq === 0) {
        for (const c of cbs) c.onTestBegin?.(h, {})
        const vl = this.testLogs(h, val)
        for (const c of cbs) c.onTestEnd?.(h, vl)
        for (const [k, v] of Object.entries(vl)) logs[`val_${k}`] = v
      }
      for (const c of cbs) c.onEpochEnd?.(h, ep, logs)
      for (const [k, v] of Object.entries(logs)) (hist[k] ??= []).push(v)
      epochsRun.push(ep)
      last = logs
      if (verbose) {
        const text = Object.entries(logs).map(([k, v]) => `${k}: ${kfmt(v)}`).join(' - ')
        h.print(`${params.steps}/${params.steps} ${verbose === 1 ? '━━━━━━━━━━━━━━━━━━━━ ' : ''}- ${text}\n`)
      }
      if (this.stopTraining) break
    }
    for (const c of cbs) c.onTrainEnd?.(h, last)
    E.grad = true
    // overfitting: validation loss climbing while training loss keeps falling
    const vl = hist.val_loss
    if (vl && vl.length >= 4) {
      const lo = vl.indexOf(Math.min(...vl))
      if (lo < vl.length - 2 && hist.loss[hist.loss.length - 1] < hist.loss[lo]) noteOnce(h, 'keras_overfit', 'info', { epoch: lo + 1 })
    }
    if (h.tracing) h.emit({ type: 'history', metrics: hist, title: `fit · ${epochsRun.length} epochs` })
    this.historyObj = new History(hist, epochsRun, params, this)
    return py.obj(this.historyObj)
  }

  /** the numbers evaluate and *_on_batch return: loss, then each output's loss, then the metrics in compile order */
  flatLogs(logs: Logs, dict: boolean): Value {
    const f = (x: number) => py.float(f32clean(Math.fround(x)))
    if (dict) return { k: 'dict', d: PyDict.from(Object.entries(logs).map(([k, v]) => [py.str(k), f(v)])) }
    const names = ['loss', ...(this.outLossT.length > 1 ? this.outputNames(this.outLossT.length).map((n) => `${n}_loss`) : []), ...this.metricSlots().map((s) => s.key)]
    const vals = names.filter((n) => n in logs).map((n) => f(logs[n]))
    return vals.length === 1 ? vals[0] : py.list(vals)
  }
  evaluate(h: Host, args: Value[], kw: Kw): Value {
    if (!this.compiled) throw h.err('RuntimeError', 'You must call `compile()` before using the model.')
    const P = bindArgs(h, 'evaluate', 'x=None y=None batch_size=None verbose=auto sample_weight=None steps=None callbacks=None return_dict=False', args, kw, null)
    const verbose = P.verbose.k === 'str' ? 1 : h.toInt(P.verbose)
    let src: Source
    if (P.x.k === 'obj' && P.x.o instanceof TFDataset) {
      src = this.datasetSource(h, P.x.o, isNone(P.steps) ? null : h.toInt(P.steps))
      if (!src.steps) src.steps = P.x.o.items(h).length
    } else {
      if (!isNone(P.steps)) throw h.err('NotImplementedError', 'evaluate: steps= with arrays is not implemented in this sandbox: pass a tf.data.Dataset')
      const xs = this.prepare(h, this.inputsOf(h, P.x))
      const ys = this.targetsOf(h, P.y, this.nOutputs(h, xs))
      if (!ys.length) throw h.err('ValueError', 'evaluate() needs targets: pass y')
      for (const c of ys) if (xs[0].shape[0] !== c.shape[0]) throw h.err('ValueError', `Data cardinality is ambiguous. Make sure all arrays contain the same number of samples.'x' sizes: ${xs[0].shape[0]}\n'y' sizes: ${c.shape[0]}`)
      src = this.arraySource(h, xs, ys, isNone(P.sample_weight) ? null : asFloat(h, P.sample_weight), isNone(P.batch_size) ? 32 : h.toInt(P.batch_size))
    }
    const cbs = hooksOf(h, P.callbacks)
    for (const c of cbs) {
      c.setup?.(this, { epochs: 1, steps: src.steps, verbose }, h)
      c.onTestBegin?.(h, {})
    }
    const logs = this.testLogs(h, src)
    for (const c of cbs) c.onTestEnd?.(h, logs)
    if (verbose) h.print(`${src.steps}/${src.steps} ━━━━━━━━━━━━━━━━━━━━ - ${Object.entries(logs).map(([k, v]) => `${k}: ${kfmt(v)}`).join(' - ')}\n`)
    return this.flatLogs(logs, h.truthy(P.return_dict))
  }
  outsToPy(outs: TT[]): Value {
    const vals = outs.map((t) => py.arr(ndFor(t)))
    return vals.length === 1 ? vals[0] : py.list(vals)
  }
  /** a forward pass with no gradient */
  infer(h: Host, xs: TT[]): TT[] {
    const S = stateOf(h)
    const was = S.E.grad
    S.E.grad = false
    S.losses = []
    try {
      return this.invoke(S.E, xs, false, h)
    } finally {
      S.E.grad = was
    }
  }
  predict(h: Host, args: Value[], kw: Kw): Value {
    const P = bindArgs(h, 'predict', 'x batch_size=None verbose=auto steps=None callbacks=None', args, kw, null)
    const verbose = P.verbose.k === 'str' ? 1 : h.toInt(P.verbose)
    const cbs = hooksOf(h, P.callbacks)
    for (const c of cbs) c.onPredictBegin?.(h, {})
    let outs: TT[]
    let steps: number
    if (P.x.k === 'obj' && P.x.o instanceof TFDataset) {
      const limit = isNone(P.steps) ? Infinity : h.toInt(P.steps)
      const parts = P.x.o.items(h).slice(0, limit).map((el) => {
        return this.infer(h, this.batchFrom(h, el).xs)
      })
      if (!parts.length) throw h.err('ValueError', 'Unexpected result of `predict_function` (Empty batch_outputs). Please use `Model.compile(..., run_eagerly=True)`, or `tf.config.run_functions_eagerly(True)` for more information of where went wrong, or file a issue/bug to `tf.keras`.')
      outs = parts[0].map((_, i) => C.catT({ grad: false } as Engine, parts.map((p) => p[i]), 0))
      steps = parts.length
    } else {
      if (!isNone(P.steps)) throw h.err('NotImplementedError', 'predict: steps= with arrays is not implemented in this sandbox: pass a tf.data.Dataset')
      const xs = this.prepare(h, this.inputsOf(h, P.x))
      outs = this.infer(h, xs)
      steps = Math.ceil(xs[0].shape[0] / (isNone(P.batch_size) ? 32 : h.toInt(P.batch_size)))
    }
    for (const c of cbs) c.onPredictEnd?.(h, {})
    if (verbose) h.print(`${steps}/${steps} ━━━━━━━━━━━━━━━━━━━━\n`)
    return this.outsToPy(outs)
  }

  // ---- config
  getConfig(h: Host): { [k: string]: Json } {
    if (this.mode === 'subclass') throw h.err('NotImplementedError', `Layer ${this.kind} has arguments in \`__init__\` and therefore must override \`get_config()\`.`)
    const base = super.getConfig(h)
    if (this.mode === 'sequential') {
      const input = this.inShape ? [objConfig('keras.layers', 'InputLayer', { batch_shape: new JTuple(this.inShape), dtype: 'float32', sparse: false, name: this.inputsK[0]?.layer.name ?? 'input_layer' })] : []
      return { ...base, layers: [...input, ...this.layers.map((l) => l.serialize(h))], ...(this.inShape ? { build_input_shape: new JTuple(this.inShape) } : {}) }
    }
    const history = (t: KerasTensor): Json => [t.layer.name, t.node.index, t.outIndex]
    const tensorCfg = (t: KerasTensor): Json => ({ class_name: '__keras_tensor__', config: { shape: t.shape, dtype: t.dtype, keras_history: history(t) } })
    const layers = this.layers.map((l) => ({
      ...l.serialize(h),
      name: l.name,
      inbound_nodes: l.nodes.filter((n) => this.relevant.has(n) && n.inputs.length).map((n): Json => ({ args: n.inputs.length === 1 ? [tensorCfg(n.inputs[0])] : [n.inputs.map(tensorCfg)], kwargs: {} })),
    }))
    return { ...base, layers, input_layers: this.inputsK.map(history), output_layers: this.outputsK.map(history) }
  }
  compileConfig(h: Host): { [k: string]: Json } | null {
    const c = this.compiled
    if (!c) return null
    const asCfg = (v: Value): Json => (v.k === 'obj' && v.o instanceof LossObj ? objConfig('keras.losses', v.o.cls, { name: v.o.name, reduction: v.o.reduction }) : v.k === 'obj' && v.o instanceof KMetric ? objConfig('keras.metrics', v.o.cls, { name: v.o.name, dtype: 'float32' }) : v.k === 'func' ? v.name : v.k === 'fn' ? v.name : isSeq(v) ? v.items.map(asCfg) : pyToJs(h, v))
    return { optimizer: objConfig('keras.optimizers', c.opt.cls, c.opt.config(h)), loss: asCfg(c.lossArg), loss_weights: pyToJs(h, c.lossWeights), metrics: asCfg(c.metricsArg), weighted_metrics: null, run_eagerly: false, steps_per_execution: 1, jit_compile: false }
  }
  serialize(h: Host): { [k: string]: Json } {
    const out = objConfig('keras', this.mode === 'functional' ? 'Functional' : 'Sequential', this.getConfig(h))
    if (this.inShape) out.build_config = { input_shape: this.inShape }
    const cc = this.compileConfig(h)
    if (cc) out.compile_config = cc
    return out
  }

  /** what a .keras file holds: the weights and the optimizer's state, as they are now */
  saved(): SavedFile {
    const opt = this.compiled?.opt
    const slots = opt ? this.trainableTs().map((p) => Object.fromEntries(Object.entries(opt.slots.get(p) ?? {}).map(([k, v]) => [k, [...v]]))) : []
    return { kind: 'model', model: this, values: this.weightValues(), opt: opt ? { t: opt.t, lr: opt.lr, uProduct: opt.uProduct, built: opt.built, slots } : null }
  }

  // ---- attributes
  getAttr(name: string, h: Host): Value | undefined {
    const fn = (call: (args: Value[], kw: Kw) => Value): Value => fnv(name, call)
    switch (name) {
      case 'layers': return py.list(this.listed().map((l) => l.self()))
      case 'add': return fn((args, kw) => {
        if (this.mode !== 'sequential') throw h.err('AttributeError', `'${this.kind}' object has no attribute 'add'`)
        this.add(h, args[0] ?? kw.layer)
        return py.NONE
      })
      case 'pop': return fn(() => {
        if (this.mode !== 'sequential') throw h.err('AttributeError', `'${this.kind}' object has no attribute 'pop'`)
        const l = this.layers.pop()
        if (!l) throw h.err('ValueError', 'There are no layers in the model.')
        l.nodes.pop()
        if (this.attached > this.layers.length) this.attached = this.layers.length
        const prev = this.layers[this.layers.length - 1]
        if (this.inShape) {
          this.outputsK = prev ? [prev.nodes[prev.nodes.length - 1].outputs[0]] : this.inputsK
          this.outShape = this.outputsK[0].shape
        }
        return l.self()
      })
      case 'summary': return fn(() => { this.summary(h); return py.NONE })
      case 'build': return fn((args, kw) => {
        const s: KShape = h.iterate(args[0] ?? kw.input_shape).map((v) => (v.k === 'none' ? null : h.toInt(v)))
        if (this.mode === 'subclass') {
          const dims = s.map((d) => d ?? 1)
          this.dryRun(h, [TT.of(new Array(prod(dims)).fill(0), dims, 'float32')])
        } else this.build(h, s)
        return py.NONE
      })
      case 'compile': return fn((args, kw) => {
        const P = bindArgs(h, 'compile', 'optimizer=rmsprop loss=None loss_weights=None metrics=None !weighted_metrics=None run_eagerly=False steps_per_execution=1 jit_compile=auto auto_scale_loss=True', args, kw, null)
        this.compiled = { opt: optimizerArg(h, P.optimizer), lossArg: P.loss, metricsArg: P.metrics, lossWeights: P.loss_weights, specs: null, slots: null, weights: [] }
        // an unknown loss or metric name fails at compile time, as in Keras
        if (P.loss.k === 'str') lossSpec(h, P.loss)
        this.outLossT = []
        return py.NONE
      })
      case 'get_compile_config': return fn(() => { const c = this.compileConfig(h); return c ? jsToPy(c) : py.NONE })
      case 'fit': return fn((args, kw) => this.fit(h, args, kw))
      case 'evaluate': return fn((args, kw) => this.evaluate(h, args, kw))
      case 'predict': return fn((args, kw) => this.predict(h, args, kw))
      case 'predict_on_batch': return fn((args, kw) => this.outsToPy(this.infer(h, this.prepare(h, this.inputsOf(h, args[0] ?? kw.x)))))
      case 'train_on_batch': case 'test_on_batch': return fn((args, kw) => {
        if (!this.compiled) throw h.err('RuntimeError', 'You must call `compile()` before using the model.')
        const P = bindArgs(h, name, name === 'train_on_batch' ? 'x y=None sample_weight=None !class_weight=None return_dict=False' : 'x y=None sample_weight=None return_dict=False', args, kw, null)
        const xs = this.prepare(h, this.inputsOf(h, P.x))
        const ys = this.targetsOf(h, P.y, this.nOutputs(h, xs))
        const b: Batch = { xs, ys, sw: isNone(P.sample_weight) ? null : asFloat(h, P.sample_weight) }
        if (name === 'train_on_batch') this.trainStep(h, b, this.trainableTs())
        else this.testStep(h, b)
        return this.flatLogs(this.logs(h), h.truthy(P.return_dict))
      })
      case 'reset_metrics': return fn(() => { this.resetMetrics(h); return py.NONE })
      case 'get_layer': return fn((args, kw) => {
        const P = bindArgs(h, 'get_layer', 'name=None index=None', args, kw, null)
        const ls = this.listed()
        if (P.name.k === 'int') {
          const byPos = ls[P.name.v]
          if (!byPos) throw h.err('ValueError', `Was asked to retrieve layer at index ${P.name.v} but model only has ${ls.length} layers.`)
          return byPos.self()
        }
        if (P.index.k !== 'none') {
          const i = h.toInt(P.index)
          const byIndex = ls[i]
          if (!byIndex) throw h.err('ValueError', `Was asked to retrieve layer at index ${i} but model only has ${ls.length} layers.`)
          return byIndex.self()
        }
        if (P.name.k === 'none') throw h.err('ValueError', 'Provide either a layer name or layer index at `get_layer`.')
        const want = h.str(P.name)
        const l = ls.find((x) => x.name === want)
        if (!l) throw h.err('ValueError', `No such layer: ${want}. Existing layers are: [${ls.map((x) => `'${x.name}'`).join(', ')}].`)
        return l.self()
      })
      case 'input_shape': {
        if (!this.inShape) throw h.err('AttributeError', `Sequential model '${this.name}' has no defined input shape yet.`)
        return this.inputsK.length > 1 ? py.list(this.inputsK.map((t) => shapeTuple(t.shape))) : shapeTuple(this.inShape)
      }
      case 'output_shape': {
        if (!this.outShape) throw h.err('AttributeError', `The model '${this.name}' has never been called and thus has no defined output shape.`)
        return this.outputsK.length > 1 ? py.list(this.outputsK.map((t) => shapeTuple(t.shape))) : shapeTuple(this.outShape)
      }
      case 'inputs': case 'outputs': {
        const list = name === 'inputs' ? this.inputsK : this.outputsK
        if (this.mode === 'subclass' || !this.built) return py.NONE
        return py.list(list.map((t) => py.obj(t)))
      }
      case 'input': case 'output': {
        const list = name === 'input' ? this.inputsK : this.outputsK
        if (!this.built || !list.length) throw h.err('AttributeError', `Sequential model '${this.name}' has no defined ${name} yet.`)
        return list.length === 1 ? py.obj(list[0]) : py.list(list.map((t) => py.obj(t)))
      }
      case 'optimizer': return this.compiled ? py.obj(this.compiled.opt) : py.NONE
      case 'loss': return this.compiled ? this.compiled.lossArg : py.NONE
      case 'compiled': return py.bool(!!this.compiled)
      case 'metrics': {
        if (!this.compiled) return py.list([])
        const mean = new KMetric('Mean', 'loss', {})
        const cm = this.metricSlots().length ? [new KMetric('CompileMetrics', 'compile_metrics', {})] : []
        return py.list([mean, ...cm].map((m) => py.obj(m)))
      }
      case 'metrics_names': {
        if (!this.compiled) return py.list([])
        const hasMetrics = !isNone(this.compiled.metricsArg) && !(isSeq(this.compiled.metricsArg) && !this.compiled.metricsArg.items.length)
        return py.list(['loss', ...(hasMetrics ? ['compile_metrics'] : [])].map(py.str))
      }
      case 'stop_training': return py.bool(this.stopTraining)
      case 'history': return this.historyObj ? py.obj(this.historyObj) : undefined
      case 'save': return fn((args, kw) => {
        const path = h.str(args[0] ?? kw.filepath)
        if (!/\.(keras|h5)$/.test(path)) throw h.err('ValueError', `Invalid filepath extension for saving. Please add either a \`.keras\` extension for the native Keras format (recommended) or a \`.h5\` extension. Use \`model.export(filepath)\` if you want to export a SavedModel for use with TFLite/TFServing/etc. Received: filepath=${path}.`)
        stateOf(h).files.set(path, this.saved())
        h.print(`(sandbox) model kept in memory as ${path}: there is no disk here\n`)
        return py.NONE
      })
      case 'save_weights': return fn((args, kw) => {
        const path = h.str(args[0] ?? kw.filepath)
        if (!path.endsWith('.weights.h5')) throw h.err('ValueError', `The filename must end in \`.weights.h5\`. Received: filepath=${path}`)
        stateOf(h).files.set(path, { kind: 'weights', values: this.weightValues() })
        h.print(`(sandbox) weights kept in memory as ${path}: there is no disk here\n`)
        return py.NONE
      })
      case 'load_weights': return fn((args, kw) => {
        const path = h.str(args[0] ?? kw.filepath)
        const f = stateOf(h).files.get(path)
        if (!f) throw h.err('FileNotFoundError', `[Errno 2] Unable to synchronously open file (unable to open file: name = '${path}', errno = 2, error message = 'No such file or directory') (the sandbox only knows files saved in this run)`)
        const ws = this.weights()
        if (f.values.length !== ws.length) throw h.err('ValueError', `A total of ${f.values.length} objects could not be loaded: the file holds ${f.values.length} weights, this model has ${ws.length}.`)
        ws.forEach((w, i) => {
          if (f.values[i].length !== w.t.size) throw h.err('ValueError', `Shape mismatch in layer ${w.layer.name} for weight ${w.name}: the weight has ${w.t.size} values, the file ${f.values[i].length}.`)
          C.writeInto(w.t, f.values[i])
        })
        return py.NONE
      })
      case 'to_json': return fn(() => py.str(pyJson(this.serialize(h))))
      case '__call__': return fn((args, kw) => this.call(args, kw, h))
    }
    return super.getAttr(name, h)
  }
  setAttr(name: string, v: Value, h: Host): boolean {
    switch (name) {
      case 'stop_training': this.stopTraining = h.truthy(v); return true
      case 'optimizer': if (this.compiled) this.compiled.opt = optimizerArg(h, v); return true
    }
    return super.setAttr(name, v, h)
  }
}

/** build(input_shape) as the user wrote it */
const userBuild = (h: Host, inst: InstValue, s: KShape) => {
  const f = userMethod(inst.cls, 'build')
  if (f) h.call({ k: 'bound', self: inst, f }, [shapeTuple(s)])
}

/** eager calls of a user's layer or model hand the arguments straight to its call() and return what it returns */
const userEagerCall = (l: KLayer, h: Host, args: Value[], kw: Kw): Value => {
  const x = args[0] ?? kw.inputs
  if (!x) throw h.err('TypeError', `${l.kind}.call() missing 1 required positional argument: 'inputs'`)
  const items = isSeq(x) ? x.items : [x]
  if (items.some((v) => kerasOf(v))) return l.symbolic(h, items, kw)
  const S = stateOf(h)
  const asT = (v: Value) => (tfOf(v) ? v : tfWrap(asFloat(h, v)))
  const input = isSeq(x) && x.items.every((v) => tfOf(v) || v.k === 'array') ? py.list(x.items.map(asT)) : asT(x)
  const first = tfOf(input) ?? tfOf((input as { items: Value[] }).items[0])!
  if (!l.built) {
    l.inShape = [null, ...first.shape.slice(1)]
    userBuild(h, l.inst!, l.inShape)
    l.built = true
  }
  const training = !isNone(kw.training) ? h.truthy(kw.training!) : S.training
  const prev = S.training
  S.training = training
  try {
    const rest: Kw = Object.fromEntries(Object.entries(kw).filter(([k]) => k !== 'training' && k !== 'inputs'))
    const out = callUser(h, l.inst!, input, training, rest, args.slice(1))
    const t = tfOf(out)
    if (t) l.outShape = [null, ...t.shape.slice(1)]
    return out
  } finally {
    S.training = prev
  }
}

const metricArg = (h: Host, v: Value): KMetric => {
  if (v.k === 'obj' && v.o instanceof KMetric) return v.o
  if (v.k === 'str') {
    if (v.v.toLowerCase() === 'accuracy' || v.v.toLowerCase() === 'acc') return new KMetric('MeanMetricWrapper', v.v, {}, true)
    const cls = METRIC_ALIASES[v.v] ?? (METRIC_SIG[v.v] !== undefined || METRIC_LOSS[v.v] || v.v === 'RootMeanSquaredError' ? v.v : null)
    if (!cls) throw h.err('ValueError', `Could not interpret metric identifier: ${v.v}`)
    return new KMetric(cls, v.v, bindArgs(h, cls, METRIC_SIG[cls] ?? 'name=None dtype=None', [], {}, null))
  }
  if (v.k === 'fn') {
    const cls = METRIC_ALIASES[v.name]
    if (cls) return new KMetric(cls, v.name, bindArgs(h, cls, METRIC_SIG[cls] ?? 'name=None dtype=None', [], {}, null))
  }
  if (v.k === 'inst' && v.state[METRIC_STATE]) {
    const m = new KMetric('Metric', (v.state[METRIC_STATE] as { name: string }).name, {})
    m.inst = v
    return m
  }
  if (v.k === 'func' || v.k === 'bound' || v.k === 'fn' || v.k === 'inst') {
    const m = new KMetric('MeanMetricWrapper', v.k === 'func' ? v.name : v.k === 'bound' ? v.f.name : v.k === 'fn' ? v.name : snake(v.cls.name), {})
    m.pyFn = v
    return m
  }
  throw h.err('ValueError', `Could not interpret metric identifier: ${repr(v)}`)
}

const METRIC_STATE = 'keras.metric'

/** clone_model: fresh weights, same architecture */
const cloneLayer = (h: Host, l: KLayer, seen: Map<KLayer, KLayer>): KLayer => {
  const known = seen.get(l)
  if (known) return known
  let out: KLayer
  if (l instanceof KModel) out = cloneModel(h, l, seen)
  else if (l instanceof Merge && l.isOp) out = l
  else if (l instanceof InstLayer) throw h.err('ValueError', `Unable to clone model: layer ${l.name} is a user class, which needs get_config() and from_config() (not implemented in this sandbox).`)
  else {
    const kw: Kw = { ...l.P, name: py.str(l.name), trainable: py.bool(l.trainableFlag) }
    if (l instanceof Bidirectional) {
      const inner = cloneLayer(h, l.fw, seen)
      inner.name = inner.name.replace(/^forward_/, '')
      kw.layer = inner.self()
    } else if (l instanceof TimeDistributed) kw.layer = cloneLayer(h, l.inner, seen).self()
    out = stateOf(h).make(l.kind, kw)
    if (l instanceof TextVectorization && out instanceof TextVectorization) out.setVocab(l.vocab.slice(l.specials.length))
  }
  seen.set(l, out)
  return out
}

const cloneModel = (h: Host, m: KModel, seen: Map<KLayer, KLayer> = new Map()): KModel => {
  if (m.mode === 'subclass') throw h.err('ValueError', `Expected \`model\` argument to be a functional \`Model\` instance, but got a subclassed model instead: ${m.repr()}`)
  const kw: Kw = { name: py.str(m.name), trainable: py.bool(m.trainableFlag) }
  if (m.mode === 'sequential') {
    const c = new KModel(h, 'sequential', kw)
    if (m.inShape) {
      c.setInput(h, m.inShape, py.str(m.inputsK[0].layer.name))
      c.built = true
      c.outShape = m.inShape
    }
    for (const l of m.layers) c.add(h, cloneLayer(h, l, seen).self())
    return c
  }
  const c = new KModel(h, 'functional', kw)
  const tensors = new Map<KerasTensor, KerasTensor>()
  const inNode = (t: KerasTensor) => {
    const inp = new InputLayer(h, t.shape, { name: py.str(t.layer.name) }, t.dtype)
    const node = new KNode(inp, [], {})
    node.outputs = [new KerasTensor(t.shape, node, 0, autoName(h, 'keras_tensor'), t.dtype)]
    return node.outputs[0]
  }
  m.inputsK.forEach((t) => tensors.set(t, inNode(t)))
  const copy = (t: KerasTensor): KerasTensor => {
    const have = tensors.get(t)
    if (have) return have
    const n = t.node
    const nl = cloneLayer(h, n.layer, seen)
    const made = nl.symbolic(h, n.inputs.map((i) => py.obj(copy(i))), n.kw)
    const outs = isSeq(made) ? made.items : [made]
    n.outputs.forEach((o, i) => tensors.set(o, kerasOf(outs[i])!))
    return tensors.get(t)!
  }
  c.inputsK = m.inputsK.map((t) => tensors.get(t)!)
  c.outputsK = m.outputsK.map(copy)
  c.mapGraph(h)
  return c
}

/** restore a compiled model from what save() kept */
const loadSaved = (h: Host, path: string, f: Extract<SavedFile, { kind: 'model' }>): KModel => {
  if (f.model.mode === 'subclass') return f.model
  const c = cloneModel(h, f.model)
  if (f.model.compiled) {
    const old = f.model.compiled
    const opt = new KOptimizer(h, old.opt.cls, old.opt.P)
    c.compiled = { ...old, opt, specs: null, slots: null }
    c.resolve(h, Math.max(1, c.outputsK.length))
    if (f.opt) {
      opt.t = f.opt.t
      opt.lr = f.opt.lr
      opt.uProduct = f.opt.uProduct
      opt.built = f.opt.built
      const saved = f.opt.slots
      c.trainableTs().forEach((p, i) => { if (saved[i]) opt.slots.set(p, saved[i]) })
    }
  }
  const ws = c.weights()
  if (ws.length !== f.values.length) throw h.err('ValueError', `The file ${path} holds ${f.values.length} weights, the rebuilt model has ${ws.length}.`)
  c.setWeightValues(f.values)
  return c
}

// ---------------------------------------------------------------- the modules

const RNN_COMMON = 'use_bias=True kernel_initializer=glorot_uniform recurrent_initializer=orthogonal bias_initializer=zeros'
const RNN_REGS = 'kernel_regularizer=None recurrent_regularizer=None bias_regularizer=None activity_regularizer=None !kernel_constraint=None !recurrent_constraint=None !bias_constraint=None dropout=0.0 recurrent_dropout=0.0 seed=None return_sequences=False return_state=False go_backwards=False !stateful=False !unroll=False use_cudnn=auto'
const GLOBAL_POOL = '!data_format=None keepdims=False'

/** constructor signatures, in the order of the Keras docs; ! marks what the sandbox refuses rather than ignores */
const LAYER_SIG: Record<string, string> = {
  Dense: `units ${CONV_COMMON}`,
  Conv2D: `filters kernel_size strides=(1,1) padding=valid ${CHANNELS_LAST} dilation_rate=(1,1) !groups=1 ${CONV_COMMON}`,
  Conv1D: `filters kernel_size strides=1 padding=valid ${CHANNELS_LAST} dilation_rate=1 !groups=1 ${CONV_COMMON}`,
  Conv2DTranspose: `filters kernel_size strides=(1,1) padding=valid output_padding=None ${CHANNELS_LAST} !dilation_rate=(1,1) ${CONV_COMMON}`,
  DepthwiseConv2D: `kernel_size strides=(1,1) padding=valid depth_multiplier=1 ${CHANNELS_LAST} dilation_rate=(1,1) activation=None use_bias=True depthwise_initializer=glorot_uniform bias_initializer=zeros depthwise_regularizer=None bias_regularizer=None activity_regularizer=None !depthwise_constraint=None !bias_constraint=None`,
  SeparableConv2D: `filters kernel_size strides=(1,1) padding=valid ${CHANNELS_LAST} dilation_rate=(1,1) depth_multiplier=1 activation=None use_bias=True depthwise_initializer=glorot_uniform pointwise_initializer=glorot_uniform bias_initializer=zeros depthwise_regularizer=None pointwise_regularizer=None bias_regularizer=None activity_regularizer=None !depthwise_constraint=None !pointwise_constraint=None !bias_constraint=None`,
  MaxPooling2D: `pool_size=(2,2) strides=None padding=valid ${CHANNELS_LAST}`,
  AveragePooling2D: `pool_size=(2,2) strides=None padding=valid ${CHANNELS_LAST}`,
  MaxPooling1D: `pool_size=2 strides=None padding=valid ${CHANNELS_LAST}`,
  AveragePooling1D: `pool_size=2 strides=None padding=valid ${CHANNELS_LAST}`,
  GlobalAveragePooling2D: GLOBAL_POOL, GlobalMaxPooling2D: GLOBAL_POOL, GlobalAveragePooling1D: GLOBAL_POOL, GlobalMaxPooling1D: GLOBAL_POOL,
  Flatten: CHANNELS_LAST,
  Dropout: 'rate !noise_shape=None seed=None',
  SpatialDropout1D: 'rate seed=None',
  SpatialDropout2D: `rate seed=None ${CHANNELS_LAST}`,
  GaussianNoise: 'stddev seed=None',
  GaussianDropout: 'rate seed=None',
  Activation: 'activation',
  ReLU: 'max_value=None negative_slope=0.0 threshold=0.0',
  LeakyReLU: 'negative_slope=0.3 alpha=None',
  ELU: 'alpha=1.0',
  PReLU: 'alpha_initializer=zeros alpha_regularizer=None !alpha_constraint=None shared_axes=None',
  Softmax: 'axis=-1',
  Reshape: 'target_shape',
  Permute: 'dims',
  RepeatVector: 'n',
  Cropping1D: 'cropping=(1,1)',
  Cropping2D: `cropping=None ${CHANNELS_LAST}`,
  ZeroPadding1D: 'padding=1',
  ZeroPadding2D: `padding=(1,1) ${CHANNELS_LAST}`,
  UpSampling1D: 'size=2',
  UpSampling2D: `size=(2,2) ${CHANNELS_LAST} interpolation=nearest`,
  Add: '', Subtract: '', Multiply: '', Average: '', Maximum: '', Minimum: '',
  Concatenate: 'axis=-1',
  Dot: 'axes normalize=False',
  Lambda: 'function output_shape=None !mask=None arguments=None',
  Normalization: 'axis=-1 mean=None variance=None invert=False',
  Rescaling: 'scale offset=0.0',
  Resizing: `height width interpolation=bilinear crop_to_aspect_ratio=False !pad_to_aspect_ratio=False !fill_mode=constant !fill_value=0.0 !antialias=False ${CHANNELS_LAST}`,
  RandomFlip: `mode=horizontal_and_vertical seed=None ${CHANNELS_LAST}`,
  RandomRotation: `factor fill_mode=reflect interpolation=bilinear seed=None fill_value=0.0 ${CHANNELS_LAST}`,
  RandomZoom: `height_factor width_factor=None fill_mode=reflect interpolation=bilinear seed=None fill_value=0.0 ${CHANNELS_LAST}`,
  RandomCrop: `height width seed=None ${CHANNELS_LAST}`,
  BatchNormalization: 'axis=-1 momentum=0.99 epsilon=0.001 center=True scale=True beta_initializer=zeros gamma_initializer=ones moving_mean_initializer=zeros moving_variance_initializer=ones beta_regularizer=None gamma_regularizer=None !beta_constraint=None !gamma_constraint=None !synchronized=False',
  LayerNormalization: 'axis=-1 epsilon=0.001 center=True scale=True !rms_scaling=False beta_initializer=zeros gamma_initializer=ones beta_regularizer=None gamma_regularizer=None !beta_constraint=None !gamma_constraint=None',
  Embedding: 'input_dim output_dim embeddings_initializer=uniform embeddings_regularizer=None !embeddings_constraint=None !mask_zero=False !weights=None',
  SimpleRNN: `units activation=tanh ${RNN_COMMON} ${RNN_REGS}`,
  LSTM: `units activation=tanh recurrent_activation=sigmoid ${RNN_COMMON} unit_forget_bias=True ${RNN_REGS}`,
  GRU: `units activation=tanh recurrent_activation=sigmoid ${RNN_COMMON} ${RNN_REGS} reset_after=True`,
  Bidirectional: 'layer merge_mode=concat !weights=None !backward_layer=None',
  TimeDistributed: 'layer',
  TextVectorization: 'max_tokens=None standardize=lower_and_strip_punctuation split=whitespace !ngrams=None output_mode=int output_sequence_length=None !pad_to_max_tokens=False vocabulary=None !idf_weights=None !sparse=False !ragged=False !encoding=utf-8',
}

const LAYER_ALIASES: Record<string, string> = {
  MaxPool2D: 'MaxPooling2D', AvgPool2D: 'AveragePooling2D', MaxPool1D: 'MaxPooling1D', AvgPool1D: 'AveragePooling1D',
  GlobalAvgPool2D: 'GlobalAveragePooling2D', GlobalMaxPool2D: 'GlobalMaxPooling2D', GlobalAvgPool1D: 'GlobalAveragePooling1D', GlobalMaxPool1D: 'GlobalMaxPooling1D',
  Convolution2D: 'Conv2D', Convolution1D: 'Conv1D', Convolution2DTranspose: 'Conv2DTranspose', Conv2DTranspose: 'Conv2DTranspose',
}

const build = (h: Host): Record<string, Value> => {
  const S = stateOf(h)
  const E = S.E
  const num = (v: Value) => h.num(v)
  const shapeOf = (v: Value): number[] => {
    const t = tfOf(v)
    if (t) return t.values()
    if (v.k === 'int') return [v.v]
    return h.iterate(v).map((x) => h.toInt(x))
  }

  // ---- keras.layers
  const seeded = (P: Params) => (isNone(P.seed) ? h.rng : new SandboxRandom(h.toInt(P.seed)))
  const same = (s: KShape): KShape => s
  const makers: Record<string, (P: Params, kw: Kw) => KLayer> = {
    Dense: (P, kw) => new Dense(h, P, kw),
    Conv2D: (P, kw) => new Conv(h, 'Conv2D', P, kw),
    Conv1D: (P, kw) => new Conv(h, 'Conv1D', P, kw),
    Conv2DTranspose: (P, kw) => new Conv2DTranspose(h, P, kw),
    DepthwiseConv2D: (P, kw) => new SepConv(h, 'DepthwiseConv2D', P, kw),
    SeparableConv2D: (P, kw) => new SepConv(h, 'SeparableConv2D', P, kw),
    MaxPooling2D: (P, kw) => new Pool(h, 'MaxPooling2D', 2, 'max', P, kw),
    AveragePooling2D: (P, kw) => new Pool(h, 'AveragePooling2D', 2, 'avg', P, kw),
    MaxPooling1D: (P, kw) => new Pool(h, 'MaxPooling1D', 1, 'max', P, kw),
    AveragePooling1D: (P, kw) => new Pool(h, 'AveragePooling1D', 1, 'avg', P, kw),
    Flatten: (P, kw) => new Simple(h, 'Flatten', kw, P, flattenShape, (e, x) => (x.ndim === 1 ? C.unsqueezeT(e, x, -1) : C.flattenT(e, x, 1, -1)), 'Flatten()'),
    Dropout: (P, kw) => {
      const rate = num(P.rate)
      if (rate < 0 || rate > 1) throw h.err('ValueError', `Invalid value received for argument \`rate\`. Expected a float value between 0 and 1. Received: rate=${rate}`)
      return new Simple(h, 'Dropout', kw, P, same, (e, x, training) => C.dropout(e, x, rate, training), `Dropout(${rate})`)
    },
    SpatialDropout1D: (P, kw) => spatialDropout(P, kw, 'SpatialDropout1D', 3),
    SpatialDropout2D: (P, kw) => spatialDropout(P, kw, 'SpatialDropout2D', 4),
    GaussianNoise: (P, kw) => {
      const sd = num(P.stddev)
      const r = seeded(P)
      return new Simple(h, 'GaussianNoise', kw, P, same, (e, x, training) => (training ? C.binary(e, '+', x, TT.of(x.values().map(() => sd * r.normal()), x.shape, x.dtype)) : x), `GaussianNoise(${sd})`)
    },
    GaussianDropout: (P, kw) => {
      const rate = num(P.rate)
      const r = seeded(P)
      const sd = Math.sqrt(rate / (1 - rate))
      return new Simple(h, 'GaussianDropout', kw, P, same, (e, x, training) => (training && rate > 0 ? C.binary(e, '*', x, TT.of(x.values().map(() => 1 + sd * r.normal()), x.shape, x.dtype)) : x), `GaussianDropout(${rate})`)
    },
    Activation: (P, kw) => {
      const a = actArg(h, P.activation)
      return new Simple(h, 'Activation', kw, P, same, (e, x) => a.f(e, x), `Activation('${a.name}')`)
    },
    ReLU: (P, kw) => {
      const maxV = isNone(P.max_value) ? null : num(P.max_value)
      const slope = num(P.negative_slope)
      const thr = num(P.threshold)
      if (maxV !== null && maxV < 0) throw h.err('ValueError', `max_value of a ReLU layer cannot be a negative value. Received: max_value=${maxV}`)
      return new Simple(h, 'ReLU', kw, P, same, (e, x) => reluOptT(e, x, slope, maxV, thr), 'ReLU()')
    },
    LeakyReLU: (P, kw) => {
      const a = isNone(P.alpha) ? num(P.negative_slope) : num(P.alpha)
      return new Simple(h, 'LeakyReLU', kw, P, same, (e, x) => leakyT(e, x, a), `LeakyReLU(${a})`)
    },
    ELU: (P, kw) => {
      const a = num(P.alpha)
      return new Simple(h, 'ELU', kw, P, same, (e, x) => eluT(e, x, a), 'ELU()')
    },
    PReLU: (P, kw) => new PReLU(h, P, kw),
    Softmax: (P, kw) => {
      const axis = h.toInt(P.axis)
      return new Simple(h, 'Softmax', kw, P, same, (e, x) => C.softmax(e, x, axis, false), 'Softmax()')
    },
    Rescaling: (P, kw) => {
      const sc = num(P.scale)
      const off = num(P.offset)
      return new Simple(h, 'Rescaling', kw, P, same, (e, x) => C.binary(e, '+', C.binary(e, '*', x, TT.scalar(sc, 'float32'), false, true), TT.scalar(off, 'float32'), false, true), `Rescaling(${sc})`)
    },
    ZeroPadding1D: (P, kw) => {
      const p = isSeq(P.padding) ? [h.toInt(P.padding.items[0]), h.toInt(P.padding.items[1])] : [h.toInt(P.padding), h.toInt(P.padding)]
      return new Simple(h, 'ZeroPadding1D', kw, P, (s) => [s[0], s[1] === null ? null : s[1] + p[0] + p[1], s[2]], (e, x) => gridT(e, x, [allOf(x.shape[0]), padMap(x.shape[1], p[0], p[1]), allOf(x.shape[2])]), `ZeroPadding1D(${p.join(', ')})`)
    },
    ZeroPadding2D: (P, kw) => {
      const p = tuple2x2(P.padding, 1)
      return new Simple(h, 'ZeroPadding2D', kw, P, (s) => [s[0], s[1] === null ? null : s[1] + p[0][0] + p[0][1], s[2] === null ? null : s[2] + p[1][0] + p[1][1], s[3]], (e, x) => nhwc(e, x, p, (z) => z), `ZeroPadding2D(${repr(P.padding)})`)
    },
    Cropping1D: (P, kw) => {
      const c = isSeq(P.cropping) ? [h.toInt(P.cropping.items[0]), h.toInt(P.cropping.items[1])] : [h.toInt(P.cropping), h.toInt(P.cropping)]
      return new Simple(h, 'Cropping1D', kw, P, (s) => [s[0], s[1] === null ? null : s[1] - c[0] - c[1], s[2]], (e, x) => sliceAxis(e, x, 1, c[0], x.shape[1] - c[1]), `Cropping1D((${c.join(', ')}))`)
    },
    Cropping2D: (P, kw) => {
      const c = tuple2x2(isNone(P.cropping) ? py.int(0) : P.cropping, 0)
      const shapeFn = (s: KShape, hh: Host): KShape => {
        if (s.length !== 4) throw hh.err('ValueError', `Input 0 of layer "cropping2d" is incompatible with the layer: expected ndim=4, found ndim=${s.length}. Full shape received: ${shapeStr(s)}`)
        const H = s[1] === null ? null : s[1] - c[0][0] - c[0][1]
        const W = s[2] === null ? null : s[2] - c[1][0] - c[1][1]
        if ((H !== null && H <= 0) || (W !== null && W <= 0)) throw hh.err('ValueError', `Values in \`cropping\` argument should be smaller than the corresponding spatial dimension of the input. Received: input_shape=${shapeStr(s)}, cropping=${repr(P.cropping)}`)
        return [s[0], H, W, s[3]]
      }
      return new Simple(h, 'Cropping2D', kw, P, shapeFn, (e, x) => sliceAxis(e, sliceAxis(e, x, 1, c[0][0], x.shape[1] - c[0][1]), 2, c[1][0], x.shape[2] - c[1][1]), `Cropping2D(${repr(P.cropping)})`)
    },
    UpSampling1D: (P, kw) => {
      const n = h.toInt(P.size)
      return new Simple(h, 'UpSampling1D', kw, P, (s) => [s[0], s[1] === null ? null : s[1] * n, s[2]], (e, x) => gridT(e, x, [allOf(x.shape[0]), allOf(x.shape[1]).flatMap((i) => new Array<number>(n).fill(i)), allOf(x.shape[2])]), `UpSampling1D(${n})`)
    },
    UpSampling2D: (P, kw) => {
      const sz = pairArg(h, P.size, 2)
      const mode = h.str(P.interpolation)
      if (mode !== 'nearest' && mode !== 'bilinear') throw h.err('NotImplementedError', `UpSampling2D: interpolation='${mode}' is not implemented in this sandbox`)
      P.size = py.tuple(sz.map(py.int))
      return new Simple(h, 'UpSampling2D', kw, P, (s, hh) => {
        if (s.length !== 4) throw hh.err('ValueError', `Input 0 of layer "up_sampling2d" is incompatible with the layer: expected ndim=4, found ndim=${s.length}. Full shape received: ${shapeStr(s)}`)
        return [s[0], s[1] === null ? null : s[1] * sz[0], s[2] === null ? null : s[2] * sz[1], s[3]]
      }, (e, x) => {
        if (mode === 'bilinear') return resizeT(e, x, x.shape[1] * sz[0], x.shape[2] * sz[1], 'bilinear')
        const rep = (n: number, k: number) => allOf(n).flatMap((i) => new Array<number>(k).fill(i))
        return gridT(e, x, [allOf(x.shape[0]), rep(x.shape[1], sz[0]), rep(x.shape[2], sz[1]), allOf(x.shape[3])])
      }, `UpSampling2D((${sz.join(', ')}))`)
    },
    Reshape: (P, kw) => {
      const target = h.iterate(P.target_shape).map((v) => h.toInt(v))
      const shapeFn = (s: KShape, hh: Host): KShape => {
        const rest = s.slice(1)
        if (rest.some((d) => d === null)) return [s[0], ...target.map((d) => (d === -1 ? null : d))]
        const size = prod(rest as number[])
        const known = target.reduce((a, d) => (d === -1 ? a : a * d), 1)
        if (target.filter((d) => d === -1).length > 1) throw hh.err('ValueError', `There must be at most one unknown dimension in output_shape. Received: output_shape=${shapeStr(target)}.`)
        const out = target.map((d) => (d === -1 ? size / known : d))
        if (prod(out) !== size || out.some((d) => !Number.isInteger(d))) throw hh.err('ValueError', `The total size of the tensor must be unchanged. Received: input_shape=${shapeStr(rest)}, target_shape=${shapeStr(target)}`)
        return [s[0], ...out]
      }
      return new Simple(h, 'Reshape', kw, P, shapeFn, (e, x) => C.reshapeT(e, x, [x.shape[0], ...target], false), `Reshape((${target.join(', ')}))`)
    },
    Permute: (P, kw) => {
      const dims = h.iterate(P.dims).map((v) => h.toInt(v))
      const shapeFn = (s: KShape, hh: Host): KShape => {
        if ([...dims].sort((a, b) => a - b).join() !== allOf(s.length - 1).map((i) => i + 1).join()) throw hh.err('ValueError', `Invalid permutation argument \`dims\` for Permute Layer. The set of indices in \`dims\` must be consecutive and start from 1. Received dims=${repr(P.dims)}`)
        return [s[0], ...dims.map((d) => s[d])]
      }
      return new Simple(h, 'Permute', kw, P, shapeFn, (e, x) => C.permuteT(e, x, [0, ...dims]), `Permute((${dims.join(', ')}))`)
    },
    RepeatVector: (P, kw) => {
      const n = h.toInt(P.n)
      return new Simple(h, 'RepeatVector', kw, P, (s, hh) => {
        if (s.length !== 2) throw hh.err('ValueError', `Input 0 of layer "repeat_vector" is incompatible with the layer: expected ndim=2, found ndim=${s.length}. Full shape received: ${shapeStr(s)}`)
        return [s[0], n, s[1]]
      }, (e, x) => repeatV(e, x, n), `RepeatVector(${n})`)
    },
    GlobalAveragePooling2D: (P, kw) => globalPool(P, kw, 'GlobalAveragePooling2D', 2, 'avg'),
    GlobalMaxPooling2D: (P, kw) => globalPool(P, kw, 'GlobalMaxPooling2D', 2, 'max'),
    GlobalAveragePooling1D: (P, kw) => globalPool(P, kw, 'GlobalAveragePooling1D', 1, 'avg'),
    GlobalMaxPooling1D: (P, kw) => globalPool(P, kw, 'GlobalMaxPooling1D', 1, 'max'),
    Add: (P, kw) => new Merge(h, 'Add', kw, P, (e, xs) => xs.reduce((a, b) => C.binary(e, '+', a, b))),
    Subtract: (P, kw) => new Merge(h, 'Subtract', kw, P, (e, xs) => {
      if (xs.length !== 2) throw h.err('ValueError', `A \`Subtract\` layer should be called on exactly 2 inputs. Received: inputs=${xs.length}`)
      return C.binary(e, '-', xs[0], xs[1])
    }, false, (hh, s) => {
      if (s.length !== 2) throw hh.err('ValueError', `A \`Subtract\` layer should be called on exactly 2 inputs. Received: input_shape=[${s.map(shapeStr).join(', ')}]`)
      return [s[0][0], ...mergeShape(hh, s[0].slice(1), s[1].slice(1))]
    }),
    Multiply: (P, kw) => new Merge(h, 'Multiply', kw, P, (e, xs) => xs.reduce((a, b) => C.binary(e, '*', a, b))),
    Average: (P, kw) => new Merge(h, 'Average', kw, P, (e, xs) => mulS(e, xs.reduce((a, b) => C.binary(e, '+', a, b)), 1 / xs.length)),
    Maximum: (P, kw) => new Merge(h, 'Maximum', kw, P, (e, xs) => xs.reduce((a, b) => C.binary(e, 'maximum', a, b))),
    Minimum: (P, kw) => new Merge(h, 'Minimum', kw, P, (e, xs) => xs.reduce((a, b) => C.binary(e, 'minimum', a, b))),
    Concatenate: (P, kw) => {
      const axis = h.toInt(P.axis)
      return new Merge(h, 'Concatenate', kw, P, (e, xs) => C.catT(e, xs, axis), false, (hh, s) => {
        const ax = axis < 0 ? axis + s[0].length : axis
        const bad = s.some((x) => x.length !== s[0].length || x.some((d, i) => i !== ax && i !== 0 && d !== null && s[0][i] !== null && d !== s[0][i]))
        if (bad) throw hh.err('ValueError', `A \`Concatenate\` layer requires inputs with matching shapes except for the concatenation axis. Received: input_shape=[${s.map(shapeStr).join(', ')}]`)
        return s[0].map((d, i) => (i === ax ? (s.some((x) => x[i] === null) ? null : s.reduce((t, x) => t + (x[i] as number), 0)) : d))
      })
    },
    Dot: (P, kw) => {
      const axes = pairArg(h, P.axes, 1)
      const normalize = h.truthy(P.normalize)
      const rel = (a: number, nd: number) => (a < 0 ? a + nd : a)
      return new Merge(h, 'Dot', kw, P, (e, [a, b]) => {
        const ax: [number, number] = [rel(axes[0], a.ndim), rel(axes[1], b.ndim)]
        return batchDot(e, normalize ? l2Normalize(e, a, ax[0]) : a, normalize ? l2Normalize(e, b, ax[1]) : b, ax)
      }, false, (hh, s) => {
        if (s.length !== 2) throw hh.err('ValueError', `A \`Dot\` layer should be called on a list of 2 inputs. Received: input_shape=[${s.map(shapeStr).join(', ')}]`)
        const [a, b] = s
        const ax = [rel(axes[0], a.length), rel(axes[1], b.length)]
        if (a[ax[0]] !== b[ax[1]]) throw hh.err('ValueError', `Incompatible input shapes: axis values ${a[ax[0]]} (at axis ${ax[0]}) != ${b[ax[1]]} (at axis ${ax[1]}). Full input shapes: ${shapeStr(a)}, ${shapeStr(b)}`)
        const rest = [...a.filter((_, i) => i !== ax[0]), ...b.filter((_, i) => i !== ax[1] && i !== 0)]
        return rest.length === 1 ? [rest[0], 1] : rest
      })
    },
    Lambda: (P, kw) => new Lambda(h, P, kw),
    BatchNormalization: (P, kw) => new BatchNorm(h, P, kw),
    LayerNormalization: (P, kw) => new LayerNorm(h, P, kw),
    Embedding: (P, kw) => new Embedding(h, P, kw),
    SimpleRNN: (P, kw) => new RNN(h, 'SimpleRNN', P, kw),
    LSTM: (P, kw) => new RNN(h, 'LSTM', P, kw),
    GRU: (P, kw) => new RNN(h, 'GRU', P, kw),
    Bidirectional: (P, kw) => new Bidirectional(h, P, kw),
    TimeDistributed: (P, kw) => new TimeDistributed(h, P, kw),
    Normalization: (P, kw) => new Normalization(h, P, kw),
    TextVectorization: (P, kw) => new TextVectorization(h, P, kw),
    Resizing: (P, kw) => {
      const Ho = h.toInt(P.height)
      const Wo = h.toInt(P.width)
      const mode = h.str(P.interpolation)
      if (mode !== 'nearest' && mode !== 'bilinear') throw h.err('NotImplementedError', `Resizing: interpolation='${mode}' is not implemented in this sandbox`)
      const crop = h.truthy(P.crop_to_aspect_ratio)
      return new Simple(h, 'Resizing', kw, P, (s) => [s[0], Ho, Wo, s[3]], (e, x) => resizeT(e, crop ? centerCropAspect(e, x, Ho, Wo) : x, Ho, Wo, mode), `Resizing(${Ho}, ${Wo})`)
    },
    RandomFlip: (P, kw) => {
      const mode = h.str(P.mode)
      if (!['horizontal', 'vertical', 'horizontal_and_vertical'].includes(mode)) throw h.err('ValueError', `RandomFlip layer ${kw.name ? h.str(kw.name) : 'random_flip'} received an unknown mode=${mode}`)
      const r = seeded(P)
      const l = new Simple(h, 'RandomFlip', kw, P, same, (_e, x, training) => {
        if (!training) return x
        return perImage(x, (img, H, W, Cc) => {
          const fh = mode !== 'vertical' && r.next() < 0.5
          const fv = mode !== 'horizontal' && r.next() < 0.5
          return resample(img, H, W, Cc, (y, q) => [fv ? H - 1 - y : y, fh ? W - 1 - q : q], 'nearest', 0, false)
        })
      }, `RandomFlip('${mode}')`)
      l.defaultTraining = true
      return l
    },
    RandomRotation: (P, kw) => {
      const [lo, hi] = factorRange(h, P.factor)
      const fill = h.str(P.fill_mode)
      const bil = imageInterp(P.interpolation, 'RandomRotation')
      const value = num(P.fill_value)
      const r = seeded(P)
      const l = new Simple(h, 'RandomRotation', kw, P, same, (_e, x, training) => {
        if (!training) return x
        return perImage(x, (img, H, W, Cc) => {
          const a = (lo + (hi - lo) * r.next()) * 2 * Math.PI
          const [cy, cx] = [(H - 1) / 2, (W - 1) / 2]
          const [c, s] = [Math.cos(a), Math.sin(a)]
          return resample(img, H, W, Cc, (y, q) => [cy + s * (q - cx) + c * (y - cy), cx + c * (q - cx) - s * (y - cy)], fill, value, bil)
        })
      }, 'RandomRotation(…)')
      l.defaultTraining = true
      return l
    },
    RandomZoom: (P, kw) => {
      const [lo, hi] = factorRange(h, P.height_factor)
      const widthR = isNone(P.width_factor) ? null : factorRange(h, P.width_factor)
      const fill = h.str(P.fill_mode)
      const bil = imageInterp(P.interpolation, 'RandomZoom')
      const value = num(P.fill_value)
      const r = seeded(P)
      const l = new Simple(h, 'RandomZoom', kw, P, same, (_e, x, training) => {
        if (!training) return x
        return perImage(x, (img, H, W, Cc) => {
          const zy = 1 + lo + (hi - lo) * r.next()
          const zx = widthR ? 1 + widthR[0] + (widthR[1] - widthR[0]) * r.next() : zy
          const [cy, cx] = [(H - 1) / 2, (W - 1) / 2]
          return resample(img, H, W, Cc, (y, q) => [cy + (y - cy) * zy, cx + (q - cx) * zx], fill, value, bil)
        })
      }, 'RandomZoom(…)')
      l.defaultTraining = true
      return l
    },
    RandomCrop: (P, kw) => {
      const Ho = h.toInt(P.height)
      const Wo = h.toInt(P.width)
      const r = seeded(P)
      const l = new Simple(h, 'RandomCrop', kw, P, (s) => [s[0], Ho, Wo, s[3]], (e, x, training) => {
        const H = x.shape[x.ndim - 3]
        const W = x.shape[x.ndim - 2]
        if (!training || H < Ho || W < Wo) return resizeT(e, centerCropAspect(e, x, Ho, Wo), Ho, Wo, 'bilinear')
        const top = Math.floor(r.next() * (H - Ho + 1))
        const left = Math.floor(r.next() * (W - Wo + 1))
        const nd = x.ndim
        return gridT(e, x, x.shape.map((n, d) => (d === nd - 3 ? allOf(Ho).map((i) => top + i) : d === nd - 2 ? allOf(Wo).map((i) => left + i) : allOf(n))))
      }, `RandomCrop(${Ho}, ${Wo})`)
      l.defaultTraining = true
      return l
    },
  }
  function imageInterp(v: Value, cls: string): boolean {
    const m = h.str(v)
    if (m !== 'nearest' && m !== 'bilinear') throw h.err('NotImplementedError', `${cls}: interpolation='${m}' is not implemented in this sandbox`)
    return m === 'bilinear'
  }
  function padMap(n: number, lo: number, hi: number): number[] {
    return [...new Array<number>(lo).fill(-1), ...allOf(n), ...new Array<number>(hi).fill(-1)]
  }
  function repeatV(e: Engine, x: TT, n: number): TT {
    return gridT(e, C.unsqueezeT(e, x, 1), [allOf(x.shape[0]), new Array<number>(n).fill(0), allOf(x.shape[1])])
  }
  /** an int, (a, b) for both sides, or ((top, bottom), (left, right)) */
  function tuple2x2(v: Value, d: number): [[number, number], [number, number]] {
    if (isSeq(v) && v.items.length === 2 && isSeq(v.items[0]) && isSeq(v.items[1])) {
      const [a, b] = v.items as { items: Value[] }[]
      return [[h.toInt(a.items[0]), h.toInt(a.items[1])], [h.toInt(b.items[0]), h.toInt(b.items[1])]]
    }
    const [x, y] = pairArg(h, v, d)
    return [[x, x], [y, y]]
  }
  function spatialDropout(P: Params, kw: Kw, kind: string, nd: number): KLayer {
    const rate = num(P.rate)
    return new Simple(h, kind, kw, P, (s, hh) => {
      if (s.length !== nd) throw hh.err('ValueError', `Input 0 of layer "${snake(kind)}" is incompatible with the layer: expected ndim=${nd}, found ndim=${s.length}. Full shape received: ${shapeStr(s)}`)
      return s
    }, (e, x, training) => {
      // whole channels drop together: the engine's channels are on axis 1
      const last = x.ndim - 1
      const toFront = [0, last, ...allOf(last - 1).map((i) => i + 1)]
      const back = toFront.map((_, i) => toFront.indexOf(i))
      return C.permuteT(e, C.dropoutChannels(e, C.permuteT(e, x, toFront), rate, training), back)
    }, `${kind}(${rate})`)
  }
  function globalPool(P: Params, kw: Kw, kind: string, rank: 1 | 2, mode: 'avg' | 'max'): KLayer {
    const keep = h.truthy(P.keepdims)
    const axes = rank === 2 ? [1, 2] : [1]
    return new Simple(h, kind, kw, P, (s, hh) => {
      if (s.length !== rank + 2) throw hh.err('ValueError', `Input 0 of layer "${snake(kind)}" is incompatible with the layer: expected ndim=${rank + 2}, found ndim=${s.length}. Full shape received: ${shapeStr(s)}`)
      return keep ? s.map((d, i) => (axes.includes(i) ? 1 : d)) : [s[0], s[s.length - 1]]
    }, (e, x) => (mode === 'avg' ? C.reduceSum(e, x, axes, keep, true) : C.extremeT(e, x, axes, keep, false, 'MaxBackward')), `${kind}()`)
  }
  const layerType = (name: string, cls = name): Value => ({
    k: 'type', name: cls, call: (args, kw) => makers[cls](bindArgs(h, cls, LAYER_SIG[cls], args, kw), kw).self(),
  })
  S.make = (kind, kw) => makers[kind](bindArgs(h, kind, LAYER_SIG[kind], [], kw), kw)

  const Input = fnv('Input', (args, kw) => {
    const P = bindArgs(h, 'Input', 'shape=None batch_size=None dtype=None !sparse=None batch_shape=None name=None !tensor=None !optional=False', args, kw, null)
    let shape: KShape
    if (!isNone(P.batch_shape)) shape = h.iterate(P.batch_shape).map((v) => (v.k === 'none' ? null : h.toInt(v)))
    else {
      if (isNone(P.shape)) throw h.err('ValueError', 'You must pass a `shape` argument.')
      shape = [isNone(P.batch_size) ? null : h.toInt(P.batch_size), ...h.iterate(P.shape).map((v) => (v.k === 'none' ? null : h.toInt(v)))]
    }
    const dt = P.dtype.k === 'none' ? 'float32' : P.dtype.k === 'obj' && P.dtype.o instanceof TFDtype ? String(P.dtype.o.d) : h.str(P.dtype)
    const layer = new InputLayer(h, shape, kw, dt)
    const node = new KNode(layer, [], {})
    node.outputs = [new KerasTensor(shape, node, 0, autoName(h, 'keras_tensor'), dt)]
    return py.obj(node.outputs[0])
  })
  const layers: Record<string, Value> = {
    Input,
    InputLayer: { k: 'type', name: 'InputLayer', call: (args, kw) => {
      const sv = kw.shape ?? kw.input_shape ?? args[0]
      const batch = kw.batch_shape ?? kw.batch_input_shape
      const shape: KShape = !isNone(batch) ? h.iterate(batch!).map((v) => (v.k === 'none' ? null : h.toInt(v))) : [null, ...h.iterate(sv!).map((v) => (v.k === 'none' ? null : h.toInt(v)))]
      return py.obj(new InputLayer(h, shape, kw))
    } },
  }
  for (const cls of Object.keys(makers)) layers[cls] = layerType(cls)
  for (const [alias, cls] of Object.entries(LAYER_ALIASES)) layers[alias] = layerType(alias, cls)
  // functional forms: keras.layers.concatenate([a, b], axis=1)
  for (const [fn, cls] of [['add', 'Add'], ['subtract', 'Subtract'], ['multiply', 'Multiply'], ['average', 'Average'], ['maximum', 'Maximum'], ['minimum', 'Minimum'], ['concatenate', 'Concatenate'], ['dot', 'Dot']]) {
    layers[fn] = fnv(fn, (args, kw) => {
      const l = S.make(cls, fn === 'dot' ? { axes: args[1] ?? kw.axes, ...(kw.normalize ? { normalize: kw.normalize } : {}) } : fn === 'concatenate' ? { axis: kw.axis ?? args[1] ?? py.int(-1) } : {})
      return l.call([args[0] ?? kw.inputs], {}, h)
    })
  }
  layers.MultiHeadAttention = { k: 'type', name: 'MultiHeadAttention', call: () => { throw h.err('NotImplementedError', 'MultiHeadAttention is not available in this sandbox') } }

  const hostInit = (inst: InstValue, l: KLayer) => {
    l.inst = inst
    inst.state[LAYER_STATE] = l
  }
  const hostCommon = (what: string): Omit<HostClass, 'init'> => ({
    setAttr(inst, name, v, hh) {
      const l = inst.state[LAYER_STATE] as KLayer | undefined
      if (!l) {
        if (layerOf(v)) throw hh.err('RuntimeError', `It looks like you are subclassing \`${what}\` and you forgot to call \`super().__init__()\`. Always start with this line.`)
        return
      }
      if (name === 'trainable') l.setTrainable(hh.truthy(v))
      if (name === 'stop_training' && l instanceof KModel) l.stopTraining = hh.truthy(v)
      if (name === 'name' && v.k === 'str') l.name = v.v
    },
    getAttr(inst, name, hh) {
      const l = inst.state[LAYER_STATE] as KLayer | undefined
      if (!l) {
        if (name === 'add_weight' || name === 'compile' || name === 'fit') throw hh.err('RuntimeError', `It looks like you are subclassing \`${what}\` and you forgot to call \`super().__init__()\`. Always start with this line.`)
        return undefined
      }
      return l.getAttr(name, hh)
    },
    call(inst, args, kw, hh) {
      const l = inst.state[LAYER_STATE] as KLayer | undefined
      if (!l) throw hh.err('RuntimeError', `It looks like you are subclassing \`${what}\` and you forgot to call \`super().__init__()\`. Always start with this line.`)
      return l.call(args, kw, hh)
    },
    repr(inst) {
      const l = inst.state[LAYER_STATE] as KLayer | undefined
      return l?.repr()
    },
  })
  const layerHost: HostClass = {
    ...hostCommon('Layer'),
    init(inst, _args, kw, hh) {
      hostInit(inst, new InstLayer(hh, inst, kw))
    },
  }
  layers.Layer = { k: 'type', name: 'Layer', host: layerHost, call: () => { throw h.err('TypeError', 'Layer is a base class: define class MyLayer(keras.layers.Layer) with build and call') } }

  const makeFunctional = (inV: Value, outV: Value, kw: Kw, kind?: string): KModel => {
    const flat = (v: Value): Value[] => (isSeq(v) ? h.iterate(v) : v.k === 'dict' ? v.d.values() : [v])
    const ins = flat(inV).map(kerasOf)
    const outs = flat(outV).map(kerasOf)
    if (ins.some((i) => !i) || outs.some((o) => !o)) throw h.err('ValueError', 'Model(inputs, outputs) expects KerasTensors from keras.Input(...) and layer calls')
    const m = new KModel(h, 'functional', kw, kind)
    m.inputsK = ins as KerasTensor[]
    m.outputsK = outs as KerasTensor[]
    m.mapGraph(h)
    return m
  }
  const modelHost: HostClass = {
    ...hostCommon('Model'),
    init(inst, args, kw, hh) {
      const inV = kw.inputs ?? args[0]
      const outV = kw.outputs ?? args[1]
      if (inV && outV) {
        const rest = Object.fromEntries(Object.entries(kw).filter(([k]) => k !== 'inputs' && k !== 'outputs'))
        hostInit(inst, makeFunctional(inV, outV, rest, inst.cls.name))
      } else hostInit(inst, new KModel(hh, 'subclass', kw, inst.cls.name))
    },
  }
  const sequentialHost: HostClass = {
    ...hostCommon('Sequential'),
    init(inst, args, kw, hh) {
      const m = new KModel(hh, 'sequential', kw, inst.cls.name)
      hostInit(inst, m)
      const ls = args[0] ?? kw.layers
      if (!isNone(ls)) for (const l of hh.iterate(ls!)) m.add(hh, l)
    },
  }
  const Sequential: Value = {
    k: 'type', name: 'Sequential', host: sequentialHost, call: (args, kw) => {
      const P = bindArgs(h, 'Sequential', 'layers=None trainable=True name=None', args, kw, null)
      const m = new KModel(h, 'sequential', { name: P.name, trainable: P.trainable })
      if (!isNone(P.layers)) for (const l of isSeq(P.layers) ? h.iterate(P.layers) : [P.layers]) m.add(h, l)
      return py.obj(m)
    },
  }
  const Model: Value = {
    k: 'type', name: 'Model', host: modelHost, call: (args, kw) => {
      const inV = kw.inputs ?? args[0]
      const outV = kw.outputs ?? args[1]
      if (!inV || !outV) throw h.err('TypeError', 'Model(inputs, outputs): both are needed for the Functional API; or subclass keras.Model and define call()')
      const rest = Object.fromEntries(Object.entries(kw).filter(([k]) => k !== 'inputs' && k !== 'outputs'))
      return py.obj(makeFunctional(inV, outV, rest))
    },
  }

  // ---- keras.losses
  const lossFn = (name: string, spec: string, extra = ''): Value => fnv(name, (args, kw) => {
    const P = bindArgs(h, name, `y_true y_pred ${extra}`, args, kw, null)
    const s = specDefaults(spec)
    if (P.from_logits) s.fromLogits = h.truthy(P.from_logits)
    if (P.label_smoothing) s.smoothing = num(P.label_smoothing)
    if (P.delta) s.delta = num(P.delta)
    return tfWrap(perSample(E, h, s, asFloat(h, P.y_true), asFloat(h, P.y_pred), false))
  })
  const lossClass = (cls: string, spec: string): Value => ({
    k: 'type', name: cls, call: (args, kw) => {
      const P = bindArgs(h, cls, LOSS_CLASS_SIG[cls] ?? LOSS_SIG, args, kw, null)
      const s = specDefaults(spec)
      if (P.from_logits) s.fromLogits = h.truthy(P.from_logits)
      if (P.label_smoothing) s.smoothing = num(P.label_smoothing)
      if (P.delta) s.delta = num(P.delta)
      const red = P.reduction.k === 'none' ? 'none' : h.str(P.reduction)
      if (!['sum_over_batch_size', 'mean', 'sum', 'none', 'mean_with_sample_weight'].includes(red)) throw h.err('ValueError', `Invalid value for argument \`reduction\`. Expected one of ['sum', 'sum_over_batch_size', 'mean', 'mean_with_sample_weight', 'none', None]. Received: reduction=${repr(P.reduction)}`)
      return py.obj(new LossObj(cls, s, (red === 'mean_with_sample_weight' ? 'mean' : red) as ReductionK, P.name.k === 'none' ? spec : h.str(P.name)))
    },
  })
  const CE = 'from_logits=False label_smoothing=0.0 !axis=-1'
  const lossFns: Record<string, Value> = {
    categorical_crossentropy: lossFn('categorical_crossentropy', 'categorical_crossentropy', CE),
    sparse_categorical_crossentropy: lossFn('sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'from_logits=False !ignore_class=None !axis=-1'),
    binary_crossentropy: lossFn('binary_crossentropy', 'binary_crossentropy', CE),
    huber: lossFn('huber', 'huber', 'delta=1.0'),
    cosine_similarity: lossFn('cosine_similarity', 'cosine_similarity', '!axis=-1'),
  }
  for (const [n, spec] of Object.entries({ mean_squared_error: 'mean_squared_error', mse: 'mean_squared_error', mean_absolute_error: 'mean_absolute_error', mae: 'mean_absolute_error', mean_absolute_percentage_error: 'mean_absolute_percentage_error', mape: 'mean_absolute_percentage_error', mean_squared_logarithmic_error: 'mean_squared_logarithmic_error', msle: 'mean_squared_logarithmic_error', hinge: 'hinge', squared_hinge: 'squared_hinge', categorical_hinge: 'categorical_hinge', kl_divergence: 'kl_divergence', kld: 'kl_divergence', poisson: 'poisson', log_cosh: 'log_cosh', logcosh: 'log_cosh' })) lossFns[n] = lossFn(spec, spec)
  const losses: Record<string, Value> = { ...lossFns }
  for (const [cls, spec] of Object.entries({ CategoricalCrossentropy: 'categorical_crossentropy', SparseCategoricalCrossentropy: 'sparse_categorical_crossentropy', BinaryCrossentropy: 'binary_crossentropy', MeanSquaredError: 'mean_squared_error', MeanAbsoluteError: 'mean_absolute_error', MeanAbsolutePercentageError: 'mean_absolute_percentage_error', MeanSquaredLogarithmicError: 'mean_squared_logarithmic_error', Huber: 'huber', Hinge: 'hinge', SquaredHinge: 'squared_hinge', CategoricalHinge: 'categorical_hinge', KLDivergence: 'kl_divergence', Poisson: 'poisson', LogCosh: 'log_cosh', CosineSimilarity: 'cosine_similarity' })) losses[cls] = lossClass(cls, spec)
  losses.get = fnv('get', (args) => (args[0].k === 'str' && losses[args[0].v] ? (LOSS_ALIASES[args[0].v] && /^[A-Z]/.test(args[0].v) ? h.call(losses[args[0].v], []) : losses[args[0].v]) : args[0]))
  const lossHost: HostClass = {
    init(inst, args, kw, hh) {
      const P = bindArgs(hh, 'Loss', 'name=None reduction=sum_over_batch_size dtype=None', args, kw, null)
      inst.state[LOSS_HOST] = P.name.k === 'none' ? snake(inst.cls.name) : hh.str(P.name)
      inst.attrs.set('reduction', P.reduction)
    },
    getAttr(inst, name) {
      if (name === 'name') return py.str((inst.state[LOSS_HOST] as string | undefined) ?? snake(inst.cls.name))
      return undefined
    },
    call(inst, args, kw, hh) {
      const y = asFloat(hh, args[0] ?? kw.y_true)
      const p = asFloat(hh, args[1] ?? kw.y_pred)
      const s = lossSpec(hh, inst)
      const w = kw.sample_weight ?? args[2]
      const red = inst.attrs.get('reduction')
      return tfWrap(reduceLossT(E, perSample(E, hh, s, y, p, false), isNone(w) ? null : asFloat(hh, w), (red && red.k === 'str' ? red.v : 'none') as ReductionK))
    },
  }
  losses.Loss = { k: 'type', name: 'Loss', host: lossHost, call: () => { throw h.err('TypeError', 'Loss is a base class: define class MyLoss(keras.losses.Loss) with call(self, y_true, y_pred)') } }

  // ---- keras.metrics
  const metricClass = (cls: string): Value => ({
    k: 'type', name: cls, call: (args, kw) => {
      const P = bindArgs(h, cls, METRIC_SIG[cls] ?? 'name=None dtype=None', args, kw, null)
      return py.obj(new KMetric(cls, P.name.k === 'none' ? metricName(cls) : h.str(P.name), P))
    },
  })
  const metricAttrs: Record<string, Value> = { ...lossFns }
  for (const cls of ['Accuracy', 'BinaryAccuracy', 'CategoricalAccuracy', 'SparseCategoricalAccuracy', 'TopKCategoricalAccuracy', 'SparseTopKCategoricalAccuracy', 'MeanSquaredError', 'MeanAbsoluteError', 'MeanAbsolutePercentageError', 'MeanSquaredLogarithmicError', 'RootMeanSquaredError', 'Precision', 'Recall', 'AUC', 'TruePositives', 'FalsePositives', 'TrueNegatives', 'FalseNegatives', 'Mean', 'Sum', 'BinaryCrossentropy', 'CategoricalCrossentropy', 'SparseCategoricalCrossentropy', 'Hinge', 'SquaredHinge', 'CategoricalHinge', 'KLDivergence', 'Poisson', 'LogCoshError', 'CosineSimilarity']) metricAttrs[cls] = metricClass(cls)
  /** binary_accuracy(y_true, y_pred) and friends: the per-sample function behind each accuracy */
  const accuracyFn = (name: string, cls: string, extra = ''): Value => fnv(name, (args, kw) => {
    const P = bindArgs(h, name, `y_true y_pred ${extra}`, args, kw, null)
    const y = asFloat(h, P.y_true)
    const p = asFloat(h, P.y_pred)
    const e2 = new Engine(h.rng)
    e2.grad = false
    const row = cls === 'BinaryAccuracy' ? null : rowArgmax(p)
    if (cls === 'BinaryAccuracy') {
      const yy = sameRank(e2, y, p)
      const yv = yy.values()
      const eq = TT.of(p.values().map((v, i) => ((v > num(P.threshold) ? 1 : 0) === yv[i] ? 1 : 0)), p.shape, 'float32')
      return tfWrap(meanLast(e2, eq))
    }
    const labels = cls.startsWith('Sparse') ? y.values().map(Math.round) : rowArgmax(y)
    if (P.k) {
      const k = h.toInt(P.k)
      const d = p.shape[p.ndim - 1]
      const pv = p.values()
      return tfWrap(TT.of(labels.map((l, n) => (pv.slice(n * d, (n + 1) * d).filter((v, _i, row2) => v > row2[l]).length < k ? 1 : 0)), [labels.length], 'float32'))
    }
    return tfWrap(TT.of(row!.map((v, i) => (v === labels[i] ? 1 : 0)), [labels.length], 'float32'))
  })
  metricAttrs.binary_accuracy = accuracyFn('binary_accuracy', 'BinaryAccuracy', 'threshold=0.5')
  metricAttrs.categorical_accuracy = accuracyFn('categorical_accuracy', 'CategoricalAccuracy')
  metricAttrs.sparse_categorical_accuracy = accuracyFn('sparse_categorical_accuracy', 'SparseCategoricalAccuracy')
  metricAttrs.top_k_categorical_accuracy = accuracyFn('top_k_categorical_accuracy', 'TopKCategoricalAccuracy', 'k=5')
  metricAttrs.sparse_top_k_categorical_accuracy = accuracyFn('sparse_top_k_categorical_accuracy', 'SparseTopKCategoricalAccuracy', 'k=5')
  const metricHost: HostClass = {
    init(inst, args, kw, hh) {
      const P = bindArgs(hh, 'Metric', 'name=None dtype=None', args, kw, null)
      inst.state[METRIC_STATE] = { name: P.name.k === 'none' ? snake(inst.cls.name) : hh.str(P.name), vars: [] as TT[] }
    },
    getAttr(inst, name, hh) {
      const st = inst.state[METRIC_STATE] as { name: string; vars: TT[] } | undefined
      if (!st) return undefined
      if (name === 'name') return py.str(st.name)
      if (name === 'add_variable' || name === 'add_weight') return fnv(name, (args, kw) => {
        const P = bindArgs(hh, name, name === 'add_weight' ? 'name=None shape=() initializer=zeros dtype=None' : 'shape initializer dtype=None aggregation=sum name=None', args, kw, null)
        const shape = isSeq(P.shape) ? hh.iterate(P.shape).map((v) => hh.toInt(v)) : []
        const t = initArg(hh, P.initializer, 'Zeros').make(hh, shape)
        st.vars.push(t)
        return py.obj(new KVariable(t, `${st.name}/${P.name.k === 'none' ? 'variable' : hh.str(P.name)}`, false))
      })
      if (name === 'reset_state' || name === 'reset_states') return fnv(name, () => {
        for (const t of st.vars) C.writeInto(t, new Array<number>(t.size).fill(0))
        return py.NONE
      })
      return undefined
    },
    call(inst, args, kw, hh) {
      hh.call(hh.getAttr(inst, 'update_state'), args, kw)
      return hh.call(hh.getAttr(inst, 'result'), [])
    },
  }
  metricAttrs.Metric = { k: 'type', name: 'Metric', host: metricHost, call: () => { throw h.err('TypeError', 'Metric is a base class: define class MyMetric(keras.metrics.Metric) with update_state and result') } }

  // ---- keras.optimizers
  const optim = (cls: string): Value => ({
    k: 'type', name: cls, call: (args, kw) => py.obj(new KOptimizer(h, cls, bindArgs(h, cls, OPT_SIG[cls], args, kw, null))),
  })
  const schedules: Value = {
    k: 'module', name: 'keras.optimizers.schedules', attrs: Object.fromEntries(Object.keys(SCHEDULE_SIG).map((c): [string, Value] => [c, { k: 'type', name: c, call: (args, kw) => py.obj(new LRSchedule(c, bindArgs(h, c, SCHEDULE_SIG[c], args, kw, null))) }])),
  }
  ;(schedules as { attrs: Record<string, Value> }).attrs.LearningRateSchedule = { k: 'type', name: 'LearningRateSchedule', call: () => { throw h.err('NotImplementedError', 'custom LearningRateSchedule subclasses are not implemented in this sandbox') } }
  const optimizers: Value = { k: 'module', name: 'keras.optimizers', attrs: { ...Object.fromEntries(Object.keys(OPT_SIG).map((c) => [c, optim(c)])), schedules, get: fnv('get', (args) => py.obj(optimizerArg(h, args[0]))) } }

  // ---- keras.callbacks
  const cbSig = (cls: string, sig: string, make: (P: Params) => Value): Value => ({ k: 'type', name: cls, call: (args, kw) => make(bindArgs(h, cls, sig, args, kw, null)) })
  const callbackHost: HostClass = {
    init(inst) {
      inst.state[CALLBACK_STATE] = true
      inst.attrs.set('model', py.NONE)
      inst.attrs.set('params', py.NONE)
    },
    getAttr(_inst, name) {
      // the hooks a subclass did not write do nothing
      if (name.startsWith('on_')) return fnv(name, () => py.NONE)
      if (name === 'set_model' || name === 'set_params') return fnv(name, () => py.NONE)
      return undefined
    },
  }
  const callbacks: Value = {
    k: 'module', name: 'keras.callbacks', attrs: {
      Callback: { k: 'type', name: 'Callback', host: callbackHost, call: () => py.obj(new KCallback('Callback')) },
      EarlyStopping: cbSig('EarlyStopping', 'monitor=val_loss min_delta=0 patience=0 verbose=0 mode=auto baseline=None restore_best_weights=False start_from_epoch=0', (P) => py.obj(new EarlyStopping(h, P))),
      ReduceLROnPlateau: cbSig('ReduceLROnPlateau', 'monitor=val_loss factor=0.1 patience=10 verbose=0 mode=auto min_delta=0.0001 cooldown=0 min_lr=0.0', (P) => py.obj(new ReduceLROnPlateau(h, P))),
      LearningRateScheduler: cbSig('LearningRateScheduler', 'schedule verbose=0', (P) => py.obj(new LearningRateScheduler(P.schedule, h.toInt(P.verbose)))),
      ModelCheckpoint: cbSig('ModelCheckpoint', 'filepath monitor=val_loss verbose=0 save_best_only=False save_weights_only=False mode=auto !save_freq=epoch initial_value_threshold=None', (P) => py.obj(new ModelCheckpoint(h, P))),
      TerminateOnNaN: cbSig('TerminateOnNaN', '', () => py.obj(new TerminateOnNaN())),
      LambdaCallback: cbSig('LambdaCallback', 'on_epoch_begin=None on_epoch_end=None on_train_begin=None on_train_end=None on_train_batch_begin=None on_train_batch_end=None', (P) => py.obj(new LambdaCallback(P))),
      CSVLogger: cbSig('CSVLogger', 'filename separator=, append=False', (P) => {
        h.print(`(sandbox) CSVLogger writes nothing here, there is no disk (${h.str(P.filename)}); history.history holds the same numbers\n`)
        return py.obj(new KCallback('CSVLogger'))
      }),
      TensorBoard: { k: 'type', name: 'TensorBoard', call: () => {
        h.print('(sandbox) TensorBoard logs are not written here; the training curves appear in the history step instead\n')
        return py.obj(new KCallback('TensorBoard'))
      } },
      History: { k: 'type', name: 'History', call: () => py.obj(new KCallback('History')) },
      ProgbarLogger: { k: 'type', name: 'ProgbarLogger', call: () => py.obj(new KCallback('ProgbarLogger')) },
    },
  }

  // ---- keras.utils
  const toCategorical = fnv('to_categorical', (args, kw) => {
    const a = toArray(args[0] ?? kw.x)
    const v = a.values().map(Math.round)
    const n = kw.num_classes ?? args[1] ? h.toInt((kw.num_classes ?? args[1])!) : Math.max(...v) + 1
    for (const c of v) if (c >= n || c < 0) throw h.err('IndexError', `index ${c} is out of bounds for axis 1 with size ${n}`)
    const shape = a.shape.length && a.shape[a.shape.length - 1] === 1 && a.ndim > 1 ? a.shape.slice(0, -1) : a.shape
    return py.arr(NDArray.create(v.flatMap((c) => Array.from({ length: n }, (_, j) => (j === c ? 1 : 0))), [...shape, n], 'float64'))
  })
  const padSequences = fnv('pad_sequences', (args, kw) => {
    const P = bindArgs(h, 'pad_sequences', 'sequences maxlen=None dtype=int32 padding=pre truncating=pre value=0.0', args, kw, null)
    const seqs = h.iterate(P.sequences).map((s) => h.iterate(s).map((x) => num(x)))
    const pads = h.str(P.padding)
    const trunc = h.str(P.truncating)
    if (pads !== 'pre' && pads !== 'post') throw h.err('ValueError', `Padding type "${pads}" not understood`)
    if (trunc !== 'pre' && trunc !== 'post') throw h.err('ValueError', `Truncating type "${trunc}" not understood`)
    const L = isNone(P.maxlen) ? seqs.reduce((m, s) => Math.max(m, s.length), 0) : h.toInt(P.maxlen)
    const value = num(P.value)
    const out: number[] = []
    for (const s of seqs) {
      const cut = s.length > L ? (trunc === 'pre' ? s.slice(s.length - L) : s.slice(0, L)) : s
      const fill = new Array<number>(L - cut.length).fill(value)
      out.push(...(pads === 'post' ? [...cut, ...fill] : [...fill, ...cut]))
    }
    const dt = h.str(P.dtype)
    return py.arr(NDArray.create(out, [seqs.length, L], dt.startsWith('float') ? 'float64' : 'int64'))
  })
  /** keras.utils.split_dataset: arrays, a tuple of arrays or a dataset, into two datasets */
  const splitDataset = fnv('split_dataset', (args, kw) => {
    const P = bindArgs(h, 'split_dataset', 'dataset left_size=None right_size=None shuffle=False seed=None', args, kw, null)
    const dsv = P.dataset
    let elements: Value[]
    if (dsv.k === 'obj' && dsv.o instanceof TFDataset) elements = dsv.o.items(h)
    else elements = sliceElements(h, dsv)
    const total = elements.length
    if (isNone(P.left_size) && isNone(P.right_size)) throw h.err('ValueError', 'At least one of the `left_size` or `right_size` must be specified. Received: left_size=None and right_size=None')
    const size = (v: Value, side: string): number | null => {
      if (isNone(v)) return null
      if (v.k === 'float') {
        if (v.v <= 0 || v.v >= 1) throw h.err('ValueError', `\`${side}_size\` should be either a positive integer smaller than ${total}, or a float within the range \`[0, 1]\`. Received: ${side}_size=${v.v}`)
        return pyRound(v.v * total)
      }
      const n = h.toInt(v)
      if (n <= 0 || n >= total + 1) throw h.err('ValueError', `\`${side}_size\` should be either a positive integer smaller than ${total}, or a float within the range \`[0, 1]\`. Received: ${side}_size=${n}`)
      return n
    }
    let left = size(P.left_size, 'left')
    let right = size(P.right_size, 'right')
    if (left === null) left = total - right!
    if (right === null) right = total - left
    if (left + right > total) throw h.err('ValueError', `The sum of \`left_size\` and \`right_size\` is greater than the total number of samples. Received: left_size=${left}, right_size=${right}, total_size=${total}`)
    if (left === 0 || right === 0) throw h.err('ValueError', `Both \`left_size\` and \`right_size\` must produce a non-empty split. With dataset of length ${total}, left_size=${left} and right_size=${right}.`)
    const order = allOf(total)
    if (h.truthy(P.shuffle)) {
      const r = isNone(P.seed) ? h.rng : new SandboxRandom(h.toInt(P.seed))
      for (let i = total - 1; i > 0; i--) {
        const j = Math.floor(r.next() * (i + 1))
        ;[order[i], order[j]] = [order[j], order[i]]
      }
    }
    const of = (idx: number[]) => datasetOf('_TensorSliceDataset', idx.map((i) => elements[i]))
    return py.tuple([of(order.slice(0, left)), of(order.slice(total - right))])
  })
  const utils: Value = {
    k: 'module', name: 'keras.utils', attrs: {
      to_categorical: toCategorical,
      pad_sequences: padSequences,
      split_dataset: splitDataset,
      normalize: fnv('normalize', (args, kw) => {
        const P = bindArgs(h, 'normalize', 'x axis=-1 order=2', args, kw, null)
        const axis = h.toInt(P.axis)
        const x = tfOf(P.x) ?? TT.of(toArray(P.x).values(), toArray(P.x).shape, 'float64')
        const order = num(P.order)
        const e2 = new Engine(h.rng)
        e2.grad = false
        const norms = C.vectorNorm(e2, isFloatT(x.dtype) ? x : TT.of(x.values(), x.shape, 'float32'), order, [axis], true)
        const out = C.binary(e2, '/', x, TT.of(norms.values().map((n) => (n === 0 ? 1 : n)), norms.shape, x.dtype))
        return tfOf(P.x) ? tfWrap(out) : py.arr(ndFor(out))
      }),
      img_to_array: fnv('img_to_array', (args) => {
        const a = toArray(args[0])
        if (a.ndim !== 2 && a.ndim !== 3) throw h.err('ValueError', `Unsupported image shape: (${a.shape.join(', ')})`)
        return py.arr(NDArray.create(a.values(), a.ndim === 2 ? [...a.shape, 1] : a.shape, 'float64'))
      }),
      set_random_seed: fnv('set_random_seed', (args) => { h.rng.seed(h.toInt(args[0])); return py.NONE }),
      plot_model: fnv('plot_model', () => {
        h.print('(sandbox) plot_model draws a diagram with graphviz; here, model.summary() opens the same layers as a shape table\n')
        return py.NONE
      }),
    },
  }
  const clearSession = fnv('clear_session', () => {
    S.counters.clear()
    return py.NONE
  })
  ;(utils as { attrs: Record<string, Value> }).attrs.clear_session = clearSession

  // ---- keras.models
  const parseLayerKw = (cfg: Json): Kw => {
    const kw: Kw = {}
    for (const [k, v] of Object.entries(cfg as Record<string, Json>)) {
      if (k === 'dtype') continue
      if (k === 'layer' && v && typeof v === 'object' && !Array.isArray(v) && !(v instanceof JFloat) && !(v instanceof JTuple)) kw[k] = layerFromJson(v).self()
      else kw[k] = jsToPy(v)
    }
    return kw
  }
  const layerFromJson = (o: Record<string, Json>): KLayer => {
    const cls = String(o.class_name)
    if (cls === 'Sequential' || cls === 'Functional') return modelFromConfig(o)
    if (!makers[cls]) throw h.err('ValueError', `Could not locate class '${cls}'. Make sure custom classes are decorated with \`@keras.saving.register_keras_serializable()\`.`)
    return S.make(cls, parseLayerKw(o.config))
  }
  const modelFromConfig = (o: Record<string, Json>): KModel => {
    const cls = String(o.class_name)
    const config = o.config as Record<string, Json>
    const kw: Kw = { name: py.str(String(config.name)) }
    if (config.trainable === false) kw.trainable = py.bool(false)
    const ls = (config.layers ?? []) as Record<string, Json>[]
    if (cls === 'Sequential') {
      const m = new KModel(h, 'sequential', kw)
      for (const l of ls) {
        if (l.class_name === 'InputLayer') {
          const c = l.config as Record<string, Json>
          m.setInput(h, c.batch_shape as KShape, py.str(String(c.name)))
          m.outShape = m.inShape
          m.built = true
        } else m.add(h, layerFromJson(l).self())
      }
      if (config.build_input_shape && !m.inShape) m.build(h, config.build_input_shape as KShape)
      return m
    }
    if (cls !== 'Functional') throw h.err('ValueError', `Unknown model class: ${cls}`)
    const made = new Map<string, KLayer>()
    const tensors = new Map<string, KerasTensor>()
    const key = (a: Json[]) => `${a[0]}/${a[1]}/${a[2]}`
    const m = new KModel(h, 'functional', kw)
    for (const l of ls) {
      if (l.class_name !== 'InputLayer') continue
      const c = l.config as Record<string, Json>
      const layer = new InputLayer(h, c.batch_shape as KShape, { name: py.str(String(c.name)) }, String(c.dtype ?? 'float32'))
      const node = new KNode(layer, [], {})
      node.outputs = [new KerasTensor(c.batch_shape as KShape, node, 0, autoName(h, 'keras_tensor'), layer.dtypeName)]
      made.set(layer.name, layer)
      tensors.set(key([layer.name, 0, 0]), node.outputs[0])
    }
    const pending = ls.filter((l) => l.class_name !== 'InputLayer').flatMap((l) => {
      const layer = layerFromJson(l)
      made.set(layer.name, layer)
      return ((l.inbound_nodes ?? []) as Record<string, Json>[]).map((n, i) => ({ layer, args: n.args as Json[], index: i }))
    })
    const historyOf = (a: Json): Json[][] => (Array.isArray(a) ? a.flatMap(historyOf) : [((a as Record<string, Record<string, Json>>).config.keras_history as Json[])])
    while (pending.length) {
      const i = pending.findIndex((p) => historyOf(p.args).every((x) => tensors.has(key(x))))
      if (i < 0) throw h.err('ValueError', 'An error occurred during deserialization of the Functional model: a layer waits for inputs that never appear.')
      const [p] = pending.splice(i, 1)
      const out = p.layer.symbolic(h, historyOf(p.args).map((x) => py.obj(tensors.get(key(x))!)), {})
      ;(isSeq(out) ? out.items : [out]).forEach((v, j) => tensors.set(key([p.layer.name, p.index, j]), kerasOf(v)!))
    }
    m.inputsK = (config.input_layers as Json[][]).map((x) => tensors.get(key(x))!)
    m.outputsK = (config.output_layers as Json[][]).map((x) => tensors.get(key(x))!)
    m.mapGraph(h)
    return m
  }
  const modelOf = (v: Value): KModel => {
    const l = layerOf(v)
    if (!(l instanceof KModel)) throw h.err('ValueError', `Expected \`model\` argument to be a \`Model\` instance, got ${repr(v)}`)
    return l
  }
  const models: Value = {
    k: 'module', name: 'keras.models', attrs: {
      Sequential, Model,
      load_model: fnv('load_model', (args, kw) => {
        const P = bindArgs(h, 'load_model', 'filepath custom_objects=None compile=True safe_mode=True', args, kw, null)
        const path = h.str(P.filepath)
        const f = S.files.get(path)
        if (!f || f.kind !== 'model') throw h.err('ValueError', `File not found: filepath=${path}. Please ensure the file is an accessible \`.keras\` zip file. (The sandbox only knows models saved with model.save in this run.)`)
        const m = loadSaved(h, path, f)
        if (!h.truthy(P.compile)) m.compiled = null
        return m.self()
      }),
      save_model: fnv('save_model', (args, kw) => {
        const P = bindArgs(h, 'save_model', 'model filepath overwrite=True zipped=None', args, kw, null)
        return h.call(h.getAttr(P.model, 'save'), [P.filepath])
      }),
      clone_model: fnv('clone_model', (args, kw) => {
        const P = bindArgs(h, 'clone_model', 'model !input_tensors=None !clone_function=None !call_function=None !recursive=False', args, kw, null)
        return cloneModel(h, modelOf(P.model)).self()
      }),
      model_from_json: fnv('model_from_json', (args, kw) => {
        let cfg: Json
        try {
          cfg = JSON.parse(h.str(args[0] ?? kw.json_string)) as Json
        } catch (e) {
          throw h.err('JSONDecodeError', (e as Error).message)
        }
        return modelFromConfig(cfg as Record<string, Json>).self()
      }),
    },
  }
  const datasets: Value = {
    k: 'module', name: 'keras.datasets', attrs: Object.fromEntries(['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'reuters', 'boston_housing'].map((d) => [d, { k: 'module', name: `keras.datasets.${d}`, attrs: { load_data: fnv('load_data', () => { throw h.err('RuntimeError', `the sandbox cannot download ${d}: build a small NumPy dataset yourself (see the training examples)`) }) } }])),
  }

  // ---- keras.initializers
  const initType = (cls: string): Value => ({
    k: 'type', name: cls, call: (args, kw) => {
      const P = bindArgs(h, cls, INIT_SIG[cls], args, kw, null)
      // an unseeded initializer still repeats itself call after call: it draws its seed once, here
      const seed = P.seed && P.seed.k !== 'none' ? h.toInt(P.seed) : Math.floor(h.rng.next() * 2 ** 31)
      return py.obj(new Initializer(cls, P, seed))
    },
  })
  const initializers: Value = {
    k: 'module', name: 'keras.initializers', attrs: {
      ...Object.fromEntries(Object.keys(INIT_SIG).map((c) => [c, initType(c)])),
      ...Object.fromEntries(Object.entries(INIT_ALIASES).map(([a, c]) => [a, initType(c)])),
      get: fnv('get', (args) => py.obj(initArg(h, args[0], 'Zeros'))),
    },
  }

  // ---- keras.regularizers
  const regType = (name: string, cls: 'L1' | 'L2' | 'L1L2'): Value => ({
    k: 'type', name, call: (args, kw) => {
      const P = bindArgs(h, name, cls === 'L1' ? 'l1=0.01' : cls === 'L2' ? 'l2=0.01' : 'l1=0.0 l2=0.0', args, kw, null)
      return py.obj(new Regularizer(cls, P.l1 ? num(P.l1) : 0, P.l2 ? num(P.l2) : 0))
    },
  })
  const regularizers: Value = {
    k: 'module', name: 'keras.regularizers', attrs: {
      l1: regType('L1', 'L1'), L1: regType('L1', 'L1'), l2: regType('L2', 'L2'), L2: regType('L2', 'L2'), l1_l2: regType('L1L2', 'L1L2'), L1L2: regType('L1L2', 'L1L2'),
      get: fnv('get', (args) => {
        const r = regArg(h, args[0])
        return r ? py.obj(r) : py.NONE
      }),
    },
  }

  // ---- keras.activations
  const actFn = (name: string, sig: string, f: (x: TT, P: Params) => TT): Value => fnv(name, (args, kw) => {
    const P = bindArgs(h, name, `x ${sig}`, args, kw, null)
    return tfWrap(f(asFloat(h, P.x), P))
  })
  const activationAttrs: Record<string, Value> = {}
  for (const n of Object.keys(ACTS)) activationAttrs[n] = actFn(n, '', (x) => ACTS[n](E, x))
  activationAttrs.relu = actFn('relu', 'negative_slope=0.0 max_value=None threshold=0.0', (x, P) => reluOptT(E, x, num(P.negative_slope), isNone(P.max_value) ? null : num(P.max_value), num(P.threshold)))
  activationAttrs.leaky_relu = actFn('leaky_relu', 'negative_slope=0.2', (x, P) => leakyT(E, x, num(P.negative_slope)))
  activationAttrs.elu = actFn('elu', 'alpha=1.0', (x, P) => eluT(E, x, num(P.alpha)))
  activationAttrs.gelu = actFn('gelu', 'approximate=False', (x, P) => geluT(E, x, h.truthy(P.approximate)))
  activationAttrs.softmax = actFn('softmax', 'axis=-1', (x, P) => C.softmax(E, x, h.toInt(P.axis), false))
  activationAttrs.log_softmax = actFn('log_softmax', 'axis=-1', (x, P) => C.softmax(E, x, h.toInt(P.axis), true))
  activationAttrs.get = fnv('get', (args) => {
    if (isNone(args[0])) return activationAttrs.linear
    if (args[0].k === 'str') {
      const f = activationAttrs[args[0].v]
      if (!f) throw h.err('ValueError', `Could not interpret activation function identifier: ${args[0].v}`)
      return f
    }
    return args[0]
  })
  const activations: Value = { k: 'module', name: 'keras.activations', attrs: activationAttrs }

  // ---- keras.backend
  const backend: Value = {
    k: 'module', name: 'keras.backend', attrs: {
      clear_session: clearSession,
      epsilon: fnv('epsilon', () => py.float(1e-7)),
      floatx: fnv('floatx', () => py.str('float32')),
      image_data_format: fnv('image_data_format', () => py.str('channels_last')),
      backend: fnv('backend', () => py.str('tensorflow')),
      get_value: fnv('get_value', (args) => {
        const t = tfOf(args[0])
        if (!t) return args[0]
        return t.ndim ? py.arr(ndFor(t)) : tfNumpyScalar(t)
      }),
      set_value: fnv('set_value', (args) => {
        const t = tfOf(args[0])
        if (t) C.writeInto(t, toTF(h, args[1], t.dtype).values())
        return py.NONE
      }),
    },
  }

  const kerasVariable: Value = { k: 'type', name: 'Variable', call: (args, kw) => {
    const P = bindArgs(h, 'Variable', 'initializer shape=None dtype=None trainable=True autocast=True aggregation=none synchronization=auto name=None', args, kw, null)
    const init = P.initializer
    const t = init.k === 'str' || (init.k === 'obj' && init.o instanceof Initializer) ? initArg(h, init, 'Zeros').make(h, shapeOf(P.shape)) : TT.of(asFloat(h, init).values(), asFloat(h, init).shape, 'float32')
    return py.obj(new KVariable(t, P.name.k === 'none' ? autoName(h, 'variable') : h.str(P.name), h.truthy(P.trainable)))
  } }

  const preprocessing: Value = { k: 'module', name: 'keras.preprocessing', attrs: { sequence: { k: 'module', name: 'keras.preprocessing.sequence', attrs: { pad_sequences: padSequences } } } }
  const keras: Value = {
    k: 'module', name: 'keras', attrs: {
      __version__: py.str('3.12 (sandbox)'),
      Sequential, Model, Input, Layer: layers.Layer, Variable: kerasVariable,
      layers: { k: 'module', name: 'keras.layers', attrs: layers },
      models, optimizers, losses: { k: 'module', name: 'keras.losses', attrs: losses }, callbacks, utils, datasets, activations, metrics: { k: 'module', name: 'keras.metrics', attrs: metricAttrs }, regularizers, initializers, backend, preprocessing,
    },
  }
  const kerasAttrs = (keras as { attrs: Record<string, Value> }).attrs

  return tensorflowModules(h, keras, kerasAttrs, { Sequential, Model, shapeOf, models, optimizers, callbacks, utils, datasets, activations, regularizers, initializers, backend, preprocessing })
}

/** Python's round(): halves go to even */
const pyRound = (v: number): number => {
  const f = Math.floor(v)
  const d = v - f
  return d < 0.5 ? f : d > 0.5 ? f + 1 : f % 2 === 0 ? f : f + 1
}

/** the elements of tf.data.Dataset.from_tensor_slices(value) */
const sliceElements = (h: Host, v: Value): Value[] => {
  const E = stateOf(h).E
  if (v.k === 'tuple' || v.k === 'dict') {
    const parts = v.k === 'tuple' ? v.items : v.d.values()
    const cols = parts.map((p) => sliceElements(h, p))
    const n = cols[0]?.length ?? 0
    for (const c of cols) if (c.length !== n) throw h.err('ValueError', `Dimensions ${n} and ${c.length} are not compatible`)
    return allOf(n).map((i) => (v.k === 'tuple' ? py.tuple(cols.map((c) => c[i])) : { k: 'dict', d: PyDict.from(v.d.keys().map((k, j) => [k, cols[j][i]])) }))
  }
  const strs = strsOf(v)
  if (strs) return strs.map((s) => py.obj(new TFString([s], [])))
  const t = toTF(h, v, null)
  if (!t.ndim) throw h.err('ValueError', `Unbatching a tensor is only supported for rank >= 1 tensors, got a tensor of shape ${shapeStr(t.shape)}`)
  return unstackT(E, t, 0).map(tfWrap)
}

const datasetOf = (cls: string, elements: Value[]): Value => py.obj(new TFDataset(cls, function* () {
  yield* elements
}, () => (elements.length ? specOf(elements[0]) : 'TensorSpec(shape=<unknown>, dtype=tf.float32, name=None)'), false, elements.length))

// ---------------------------------------------------------------- tensorflow

/** the namedtuples TF ops return: TopKV2(values=…, indices=…) */
const named = (tname: string, names: string[], items: Value[]): Value => ({ k: 'tuple', items, names, tname })

/** with tf.device('/CPU:0'): / with tf.name_scope('x'): nothing changes here */
class NoScope extends PyObj {
  readonly cls: string
  constructor(cls: string) {
    super()
    this.cls = cls
  }
  repr() { return `<${this.cls}>` }
  enter() { return py.NONE }
  exit() {}
}

/** tf.print's number format: absl::StrCat's six significant digits */
const g6 = (v: number): string => {
  if (!Number.isFinite(v)) return Number.isNaN(v) ? 'nan' : v > 0 ? 'inf' : '-inf'
  if (v === 0) return '0'
  const e = Math.floor(Math.log10(Math.abs(Number(v.toPrecision(6)))))
  if (e < -4 || e >= 6) return v.toExponential(5).replace(/\.?0+e/, 'e').replace(/e([+-])(\d)$/, 'e$10$2')
  return String(Number(v.toPrecision(6)))
}

const tfPrintTensor = (t: TT, summarize: number): string => {
  const elem = (v: number) => (t.dtype === 'bool' ? (v ? 'True' : 'False') : isFloatT(t.dtype) ? g6(v) : String(v))
  const vals = t.values()
  if (!t.ndim) return elem(vals[0])
  const st = cStrides(t.shape)
  const rec = (d: number, base: number, ind: number): string => {
    const n = t.shape[d]
    const cut = summarize >= 0 && n > 2 * summarize
    const idx = cut ? [...allOf(summarize), -1, ...allOf(summarize).map((i) => n - summarize + i)] : allOf(n)
    if (d === t.ndim - 1) return `[${idx.map((i) => (i < 0 ? '...' : elem(vals[base + i]))).join(' ')}]`
    return `[${idx.map((i) => (i < 0 ? '...' : rec(d + 1, base + i * st[d], ind + 1))).join(`\n${' '.repeat(ind + 1)}`)}]`
  }
  return rec(0, 0, 0)
}

const tensorflowModules = (h: Host, keras: Value, kerasAttrs: Record<string, Value>, K: { Sequential: Value; Model: Value; shapeOf: (v: Value) => number[]; models: Value; optimizers: Value; callbacks: Value; utils: Value; datasets: Value; activations: Value; regularizers: Value; initializers: Value; backend: Value; preprocessing: Value }): Record<string, Value> => {
  const S = stateOf(h)
  const E = S.E
  const { shapeOf } = K
  const tfd: Record<string, Value> = { ...Object.fromEntries(Object.entries(TF_DTYPES).map(([k, d]) => [k, py.obj(new TFDtype(d))])), string: py.obj(new TFDtype('string')) }
  const dt = (v: Value | undefined): TDType | null => {
    if (isNone(v)) return null
    if (v!.k === 'obj' && v!.o instanceof TFDtype && v!.o.d !== 'string') return v!.o.d
    if (v!.k === 'str' && TF_DTYPES[v!.v]) return TF_DTYPES[v!.v]
    if (v!.k === 'type' && v!.name === 'float') return 'float32'
    if (v!.k === 'type' && v!.name === 'int') return 'int32'
    throw h.err('TypeError', `Cannot convert ${repr(v!)} to a TensorFlow DType.`)
  }
  const T = (v: Value | undefined, what: string): TT => {
    if (!v) throw h.err('TypeError', `${what}() missing a tensor argument`)
    if (kerasOf(v)) throw h.err('ValueError', 'A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.ops`).')
    return tfOf(v) ?? toTF(h, v, null)
  }
  const label = (v: Value | undefined, d = 'x') => (v ? h.nameOf(v, d) : d)
  const traced = (api: string, kind: ApiKind, ops: { label: string; t: TT }[], out: TT, axis: number | null = null) => {
    traceTF(h, `tf.${api}`, kind, ops, out, axis)
    return tfWrap(out)
  }
  /** a shape op: every output element is one input element */
  const moved = (api: string, v: Value | undefined, t: TT, src: ArrayLike<number>, shape: number[], fill = 0): Value => {
    const out = C.takeT(E, t, src, shape, 'TakeBackward', fill)
    if (h.tracing && out.size <= 240 && t.size <= 240) traceTF(h, `tf.${api}`, 'group', [{ label: label(v), t }], out, null, Array.from(src, (s) => (s < 0 ? [] : [[0, s] as [number, number]])))
    return tfWrap(out)
  }
  const axisArg = (v: Value | undefined): number[] | null => {
    if (isNone(v)) return null
    const t = tfOf(v)
    if (t) return t.values()
    return isSeq(v) ? h.iterate(v).map((x) => h.toInt(x)) : [h.toInt(v!)]
  }
  const REDUCE_SIG = 'input_tensor axis=None keepdims=False name=None'
  const reduce = (name: string, mean: boolean, f?: 'max' | 'min') => fnv(name, (args, kw) => {
    const P = bindArgs(h, name, REDUCE_SIG, args, kw, null)
    const t = T(P.input_tensor, name)
    const axes = axisArg(P.axis)
    const keep = h.truthy(P.keepdims)
    let out: TT
    if (f) out = C.extremeT(E, t, axes, keep, f === 'min', f === 'min' ? 'MinBackward' : 'MaxBackward')
    else if (mean && !isFloatT(t.dtype)) {
      // TensorFlow keeps the integer dtype: the mean is truncated (a classic surprise)
      const m = C.reduceSum(E, C.toDtype(E, t, 'float64'), axes, keep, true)
      out = TT.of(m.values().map(Math.trunc), m.shape, t.dtype)
    } else {
      const r = C.reduceSum(E, t, axes, keep, mean)
      out = r.dtype === t.dtype || t.dtype === 'bool' ? r : C.toDtype(E, r, t.dtype)
    }
    return traced(name, 'reduce', [{ label: label(P.input_tensor), t }], out, axes && axes.length === 1 ? C.normDim(axes[0], t.ndim) : null)
  })
  const reduceProd = fnv('reduce_prod', (args, kw) => {
    const P = bindArgs(h, 'reduce_prod', REDUCE_SIG, args, kw, null)
    const t = T(P.input_tensor, 'reduce_prod')
    const axes = axisArg(P.axis)
    return traced('reduce_prod', 'reduce', [{ label: label(P.input_tensor), t }], prodAxes(E, t, axes, h.truthy(P.keepdims)), axes && axes.length === 1 ? C.normDim(axes[0], t.ndim) : null)
  })
  const reduceSpread = (name: string, sqrt: boolean) => fnv(name, (args, kw) => {
    const P = bindArgs(h, name, REDUCE_SIG, args, kw, null)
    const t = T(P.input_tensor, name)
    if (!isFloatT(t.dtype)) throw h.err('TypeError', `Input must be either real or complex. Received integer type ${t.dtype}.`)
    const axes = axisArg(P.axis)
    return traced(name, 'reduce', [{ label: label(P.input_tensor), t }], C.variance(E, t, axes, h.truthy(P.keepdims), 0, sqrt), axes && axes.length === 1 ? C.normDim(axes[0], t.ndim) : null)
  })
  const reduceBool = (name: string, op: 'Any' | 'All') => fnv(name, (args, kw) => {
    const P = bindArgs(h, name, REDUCE_SIG, args, kw, null)
    const t = T(P.input_tensor, name)
    if (t.dtype !== 'bool') throw tfErr(h, `cannot compute ${op} as input #0(zero-based) was expected to be a bool tensor but is a ${TF_KIND[t.dtype]} tensor [Op:${op}]`)
    const axes = axisArg(P.axis)
    const out = C.countReduce(t, axes, h.truthy(P.keepdims), 'bool', op === 'Any' ? (nz) => (nz > 0 ? 1 : 0) : (nz, total) => (nz === total ? 1 : 0))
    return traced(name, 'reduce', [{ label: label(P.input_tensor), t }], out, axes && axes.length === 1 ? C.normDim(axes[0], t.ndim) : null)
  })
  const FLOAT_ALLOWED = 'bfloat16, half, float, double, complex64, complex128'
  const floatOnly = (t: TT, op: string) => {
    if (!isFloatT(t.dtype)) throw tfErr(h, `Value for attr 'T' of ${t.dtype} is not in the list of allowed values: ${FLOAT_ALLOWED}\n\t; NodeDef: {{node ${op}}}; Op<name=${op}; signature=x:T -> y:T; attr=T:type,allowed=[DT_BFLOAT16, DT_HALF, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128]> [Op:${op}] name: `)
  }
  const elem = (name: string, op: C.UnOpT, tfOp: string | null) => fnv(name, (args, kw) => {
    const v = args[0] ?? kw.x
    const t = T(v, name)
    if (tfOp) floatOnly(t, tfOp)
    return traced(name, 'elementwise', [{ label: label(v), t }], C.unary(E, op, t))
  })
  /** elementwise with no gradient (TF has none for round, floor, ceil) */
  const plain = (name: string, tfOp: string | null, f: (v: number) => number) => fnv(name, (args, kw) => {
    const v = args[0] ?? kw.x
    const t = T(v, name)
    if (tfOp) floatOnly(t, tfOp)
    return traced(name, 'elementwise', [{ label: label(v), t }], TT.of(t.values().map(f), t.shape, t.dtype))
  })
  const smooth = (name: string, tfOp: string, make: (t: TT) => TT) => fnv(name, (args, kw) => {
    const v = args[0] ?? kw.features ?? kw.x
    const t = T(v, name)
    floatOnly(t, tfOp)
    return traced(name, 'elementwise', [{ label: label(v), t }], make(t))
  })
  const two = (name: string, op: string) => fnv(name, (args, kw) => {
    const a0 = args[0] ?? kw.x
    const b0 = args[1] ?? kw.y
    if (!a0 || !b0) throw h.err('TypeError', `${name}() missing a tensor argument`)
    const like = tfOf(b0)
    const a = tfOf(a0) ?? tfOperand(h, a0, like) ?? T(a0, name)
    return new TFTensor(a).binop(op, b0, false, h) ?? py.NONE
  })
  const logicalFn = (name: string, op: '&' | '|' | '^') => fnv(name, (args, kw) => {
    const a = T(args[0] ?? kw.x, name)
    const b = T(args[1] ?? kw.y, name)
    const tfOp = name === 'logical_and' ? 'LogicalAnd' : name === 'logical_or' ? 'LogicalOr' : 'LogicalXor'
    for (const t of [a, b]) if (t.dtype !== 'bool') throw tfErr(h, `cannot compute ${tfOp} as input #${t === a ? 0 : 1}(zero-based) was expected to be a bool tensor but is a ${TF_KIND[t.dtype]} tensor [Op:${tfOp}]`)
    return traced(name, 'elementwise', [{ label: label(args[0]), t: a }, { label: label(args[1], 'y'), t: b }], logical(h, a, b, op))
  })
  const logicalNot = fnv('logical_not', (args, kw) => {
    const t = T(args[0] ?? kw.x, 'logical_not')
    if (t.dtype !== 'bool') throw tfErr(h, `cannot compute LogicalNot as input #0(zero-based) was expected to be a bool tensor but is a ${TF_KIND[t.dtype]} tensor [Op:LogicalNot]`)
    return traced('logical_not', 'elementwise', [{ label: label(args[0]), t }], TT.of(t.values().map((v) => (v ? 0 : 1)), t.shape, 'bool'))
  })
  const argExtreme = (name: string, min: boolean) => fnv(name, (args, kw) => {
    const P = bindArgs(h, name, 'input axis=None output_type=int64 name=None', args, kw, null)
    const t = T(P.input, name)
    const out = C.argmax(t, isNone(P.axis) ? 0 : h.toInt(P.axis), false, min)
    const d = dt(P.output_type) ?? 'int64'
    return traced(name, 'reduce', [{ label: label(P.input), t }], d === 'int64' ? out : TT.of(out.values(), out.shape, d), isNone(P.axis) ? 0 : C.normDim(h.toInt(P.axis), t.ndim))
  })
  const confusion = fnv('confusion_matrix', (args, kw) => {
    const lab = toArray(args[0] ?? kw.labels).values().map(Math.round)
    const pred = toArray(args[1] ?? kw.predictions).values().map(Math.round)
    const n = kw.num_classes ? h.toInt(kw.num_classes) : Math.max(...lab, ...pred) + 1
    const m = new Array<number>(n * n).fill(0)
    lab.forEach((l, i) => (m[l * n + pred[i]] += 1))
    return tfWrap(TT.of(m, [n, n], 'int32'))
  })
  const clipByValue = fnv('clip_by_value', (args, kw) => {
    const P = bindArgs(h, 'clip_by_value', 't clip_value_min clip_value_max name=None', args, kw, null)
    const t = T(P.t, 'clip_by_value')
    const lo = tfOperand(h, P.clip_value_min, t)
    const hi = tfOperand(h, P.clip_value_max, t)
    if (!lo || !hi) throw h.err('TypeError', 'clip_by_value(t, clip_value_min, clip_value_max): the limits must be numbers or tensors')
    const out = lo.size === 1 && hi.size === 1 ? C.clamp(E, t, lo.values()[0], hi.values()[0]) : C.binary(E, 'minimum', C.binary(E, 'maximum', t, lo), hi)
    return traced('clip_by_value', 'elementwise', [{ label: label(P.t), t }], out)
  })
  const direction = (v: Value): boolean => {
    const d = h.str(v)
    if (d !== 'ASCENDING' && d !== 'DESCENDING') throw h.err('ValueError', `\`direction\` must be one of ['ASCENDING', 'DESCENDING']. Received: direction=${repr(v)}`)
    return d === 'DESCENDING'
  }
  const sortFn = fnv('sort', (args, kw) => {
    const P = bindArgs(h, 'sort', 'values axis=-1 direction=ASCENDING name=None', args, kw, null)
    const t = T(P.values, 'sort')
    const { src } = C.sortT(E, t, h.toInt(P.axis), direction(P.direction))
    return moved('sort', P.values, t, src, t.shape)
  })
  const argsortFn = fnv('argsort', (args, kw) => {
    const P = bindArgs(h, 'argsort', 'values axis=-1 direction=ASCENDING stable=False name=None', args, kw, null)
    const t = T(P.values, 'argsort')
    const { indices } = C.sortT(E, t, h.toInt(P.axis), direction(P.direction))
    return traced('argsort', 'sort', [{ label: label(P.values), t }], TT.of(indices.values(), t.shape, 'int32'), h.toInt(P.axis))
  })
  const topK = fnv('top_k', (args, kw) => {
    const P = bindArgs(h, 'top_k', 'input k=1 sorted=True index_type=None name=None', args, kw, null)
    const t = T(P.input, 'top_k')
    const k = h.toInt(P.k)
    if (!t.ndim) throw h.err('ValueError', 'Shape must be at least rank 1 but is rank 0')
    const d = t.shape[t.ndim - 1]
    if (k > d) throw tfErr(h, `input must have at least k columns. had ${d}, needed ${k} [Op:TopKV2]`)
    const { values, indices } = C.topkT(E, t, k, -1, true)
    return named('TopKV2', ['values', 'indices'], [tfWrap(values), tfWrap(TT.of(indices.values(), indices.shape, 'int32'))])
  })
  const countNonzero = fnv('count_nonzero', (args, kw) => {
    const P = bindArgs(h, 'count_nonzero', 'input axis=None keepdims=None dtype=int64 name=None', args, kw, null)
    const t = T(P.input, 'count_nonzero')
    const axes = axisArg(P.axis)
    return traced('count_nonzero', 'reduce', [{ label: label(P.input), t }], C.countReduce(t, axes, !isNone(P.keepdims) && h.truthy(P.keepdims), dt(P.dtype) ?? 'int64', (nz) => nz), axes && axes.length === 1 ? C.normDim(axes[0], t.ndim) : null)
  })
  const cumsum = fnv('cumsum', (args, kw) => {
    const P = bindArgs(h, 'cumsum', 'x axis=0 exclusive=False reverse=False name=None', args, kw, null)
    const t = T(P.x, 'cumsum')
    const axis = h.toInt(P.axis)
    const excl = h.truthy(P.exclusive)
    const rev = h.truthy(P.reverse)
    const d = C.normDim(axis, t.ndim)
    const n = t.shape[d]
    // reverse runs the sum from the far end; exclusive shifts it one place along, starting from zero
    const flip = (x: TT) => (rev ? flipAxes(E, x, [d]) : x)
    const sum = flip(C.cumT(E, flip(t), d, false))
    const shifted = excl ? gridT(E, sum, sum.shape.map((m, k) => (k === d ? allOf(n).map((i) => (rev ? (i + 1 < n ? i + 1 : -1) : i - 1)) : allOf(m)))) : sum
    const res = shifted.dtype === t.dtype ? shifted : C.toDtype(E, shifted, t.dtype)
    return traced('cumsum', excl || rev ? 'info' : 'scan', [{ label: label(P.x), t }], res, C.normDim(axis, t.ndim))
  })

  // ---- shapes
  const expandDims = fnv('expand_dims', (args, kw) => {
    const P = bindArgs(h, 'expand_dims', 'input axis name=None', args, kw, null)
    const t = T(P.input, 'expand_dims')
    const axis = h.toInt(P.axis)
    if (axis > t.ndim || axis < -t.ndim - 1) throw tfErr(h, `Tried to expand dim index ${axis} for tensor with ${t.ndim} dimensions. [Op:ExpandDims]`)
    return tfWrap(C.unsqueezeT(E, t, axis))
  })
  const squeeze = fnv('squeeze', (args, kw) => {
    const P = bindArgs(h, 'squeeze', 'input axis=None name=None', args, kw, null)
    const t = T(P.input, 'squeeze')
    const axes = axisArg(P.axis)
    if (axes === null) return tfWrap(C.squeezeT(E, t, null))
    const ds = axes.map((a) => {
      const d = a < 0 ? a + t.ndim : a
      if (d < 0 || d >= t.ndim) throw tfErr(h, `Tried to squeeze dim index ${a} for tensor with ${t.ndim} dimensions. [Op:Squeeze]`)
      if (t.shape[d] !== 1) throw tfErr(h, `Can not squeeze dim[${a}], expected a dimension of 1, got ${t.shape[d]} [Op:Squeeze]`)
      return d
    })
    return tfWrap(C.reshapeT(E, t, t.shape.filter((_, d) => !ds.includes(d)), false))
  })
  const splitFn = fnv('split', (args, kw) => {
    const P = bindArgs(h, 'split', 'value num_or_size_splits axis=0 num=None name=None', args, kw, null)
    const t = T(P.value, 'split')
    const axis = C.normDim(h.toInt(P.axis), t.ndim)
    const n = t.shape[axis]
    let sizes: number[]
    const spec = tfOf(P.num_or_size_splits)?.values() ?? (isSeq(P.num_or_size_splits) ? h.iterate(P.num_or_size_splits).map((v) => h.toInt(v)) : null)
    if (spec) {
      const known = spec.filter((s) => s !== -1).reduce((a, b) => a + b, 0)
      sizes = spec.map((s) => (s === -1 ? n - known : s))
      if (sizes.reduce((a, b) => a + b, 0) !== n) throw tfErr(h, `Determined shape must either match input shape along split_dim exactly if fully specified, or be less than the size of the input along split_dim if not fully specified.  Got: ${known} [Op:SplitV]`)
    } else {
      const k = h.toInt(P.num_or_size_splits)
      if (k <= 0 || n % k !== 0) throw tfErr(h, `Number of ways to split should evenly divide the split dimension, but got split_dim ${axis} (size = ${n}) and num_split ${k} [Op:Split]`)
      sizes = new Array<number>(k).fill(n / k)
    }
    let at = 0
    return py.list(sizes.map((s) => {
      const part = sliceAxis(E, t, axis, at, at + s)
      at += s
      return tfWrap(part)
    }))
  })
  const tile = fnv('tile', (args, kw) => {
    const P = bindArgs(h, 'tile', 'input multiples name=None', args, kw, null)
    const t = T(P.input, 'tile')
    const m = shapeOf(P.multiples)
    if (m.length !== t.ndim) throw tfErr(h, `Expected multiples argument to be a vector of length ${t.ndim} but got length ${m.length} [Op:Tile]`)
    const maps = t.shape.map((n, d) => Array.from({ length: n * m[d] }, (_, i) => i % n))
    return moved('tile', P.input, t, C.crossSrc(t.shape, maps), maps.map((x) => x.length))
  })
  const broadcastTo = fnv('broadcast_to', (args, kw) => {
    const P = bindArgs(h, 'broadcast_to', 'input shape name=None', args, kw, null)
    const t = T(P.input, 'broadcast_to')
    const target = shapeOf(P.shape)
    const off = target.length - t.ndim
    if (off < 0 || t.shape.some((d, i) => d !== 1 && d !== target[i + off])) throw tfErr(h, `Incompatible shapes: [${t.shape.join(',')}] vs. [${target.join(',')}] [Op:BroadcastTo]`)
    const full = [...new Array<number>(off).fill(1), ...t.shape]
    const maps = target.map((n, d) => (full[d] === 1 ? new Array<number>(n).fill(0) : allOf(n)))
    return moved('broadcast_to', P.input, t, C.crossSrc(full, maps), target)
  })
  const reverse = fnv('reverse', (args, kw) => {
    const P = bindArgs(h, 'reverse', 'tensor axis name=None', args, kw, null)
    const t = T(P.tensor, 'reverse')
    const axes = (axisArg(P.axis) ?? []).map((a) => C.normDim(a, t.ndim))
    const maps = t.shape.map((n, d) => (axes.includes(d) ? allOf(n).reverse() : allOf(n)))
    return moved('reverse', P.tensor, t, C.crossSrc(t.shape, maps), t.shape)
  })
  const gather = fnv('gather', (args, kw) => {
    const P = bindArgs(h, 'gather', 'params indices validate_indices=None axis=None !batch_dims=0 name=None', args, kw, null)
    const t = T(P.params, 'gather')
    const iv = T(P.indices, 'gather')
    if (isFloatT(iv.dtype)) throw h.err('TypeError', `Value for attr 'Tindices' of ${iv.dtype} is not in the list of allowed values: int16, int32, int64`)
    const axis = isNone(P.axis) ? 0 : C.normDim(h.toInt(P.axis), t.ndim)
    const n = t.shape[axis]
    for (const [i, c] of iv.values().entries()) if (c < 0 || c >= n) throw tfErr(h, `indices[${i}] = ${c} is not in [0, ${n}) [Op:GatherV2]`)
    const st = cStrides(t.shape)[axis]
    const rest = C.crossSrc(t.shape, t.shape.map((d, k) => (k === axis ? [0] : allOf(d))))
    const inner = prod(t.shape.slice(axis + 1))
    const before = rest.length / Math.max(1, inner)
    const src: number[] = []
    for (let b = 0; b < before; b++) for (const c of iv.values()) for (let k = 0; k < inner; k++) src.push(rest[b * inner + k] + c * st)
    return moved('gather', P.params, t, src, [...t.shape.slice(0, axis), ...iv.shape, ...t.shape.slice(axis + 1)])
  })
  const gatherNd = fnv('gather_nd', (args, kw) => {
    const P = bindArgs(h, 'gather_nd', 'params indices !batch_dims=0 name=None', args, kw, null)
    const t = T(P.params, 'gather_nd')
    const iv = T(P.indices, 'gather_nd')
    const depth = iv.shape[iv.ndim - 1]
    const st = cStrides(t.shape)
    const inner = prod(t.shape.slice(depth))
    const v = iv.values()
    const src: number[] = []
    for (let n = 0; n < v.length / depth; n++) {
      let base = 0
      for (let d = 0; d < depth; d++) {
        const c = v[n * depth + d]
        if (c < 0 || c >= t.shape[d]) throw tfErr(h, `indices[${n},${d}] = ${c} does not index into param shape [${t.shape.join(',')}] [Op:GatherNd]`)
        base += c * st[d]
      }
      for (let k = 0; k < inner; k++) src.push(base + k)
    }
    return moved('gather_nd', P.params, t, src, [...iv.shape.slice(0, -1), ...t.shape.slice(depth)])
  })
  const boolMask = fnv('boolean_mask', (args, kw) => {
    const P = bindArgs(h, 'boolean_mask', 'tensor mask axis=None name=None', args, kw, null)
    const t = T(P.tensor, 'boolean_mask')
    const m = T(P.mask, 'boolean_mask')
    if (m.dtype !== 'bool') throw h.err('TypeError', `Value for attr 'T' of ${m.dtype} is not in the list of allowed values: bool`)
    return tfWrap(booleanMask(h, E, t, m, isNone(P.axis) ? 0 : h.toInt(P.axis)))
  })
  const whereFn = fnv('where', (args, kw) => {
    const P = bindArgs(h, 'where', 'condition x=None y=None name=None', args, kw, null)
    const c = T(P.condition, 'where')
    if (isNone(P.x) !== isNone(P.y)) throw h.err('ValueError', 'x and y must both be non-None or both be None.')
    if (isNone(P.x)) {
      const st = cStrides(c.shape)
      const hits: number[] = []
      c.values().forEach((v, f) => {
        if (!v) return
        let r = f
        for (let d = 0; d < c.ndim; d++) {
          hits.push(Math.floor(r / st[d]))
          r %= st[d]
        }
      })
      return tfWrap(TT.of(hits, [hits.length / Math.max(1, c.ndim), c.ndim], 'int64'))
    }
    if (c.dtype !== 'bool') throw tfErr(h, `cannot compute SelectV2 as input #0(zero-based) was expected to be a bool tensor but is a ${TF_KIND[c.dtype]} tensor [Op:SelectV2]`)
    const xT = tfOf(P.x)
    const yT = tfOf(P.y)
    const x = xT ?? tfOperand(h, P.x, yT) ?? T(P.x, 'where')
    const y = yT ?? tfOperand(h, P.y, x) ?? T(P.y, 'where')
    if (x.dtype !== y.dtype) throw tfErr(h, `cannot compute SelectV2 as input #2(zero-based) was expected to be a ${TF_KIND[x.dtype]} tensor but is a ${TF_KIND[y.dtype]} tensor [Op:SelectV2]`)
    bcast(h, bcast(h, c.shape, x.shape), y.shape)
    return tfWrap(C.whereT(E, c, x, y, x.dtype, 'SelectBackward'))
  })
  const padFn = fnv('pad', (args, kw) => {
    const P = bindArgs(h, 'pad', 'tensor paddings mode=CONSTANT constant_values=0 name=None', args, kw, null)
    const t = T(P.tensor, 'pad')
    const pads = T(P.paddings, 'pad')
    if (pads.shape.join() !== [t.ndim, 2].join()) throw tfErr(h, `The first dimension of paddings must be the rank of inputs${repr(py.int(t.ndim))} and the second dimension of paddings must be 2 ${shapeStr(pads.shape)} [Op:Pad]`)
    const pv = pads.values()
    const mode = h.str(P.mode).toUpperCase()
    if (!['CONSTANT', 'REFLECT', 'SYMMETRIC'].includes(mode)) throw h.err('ValueError', `Value of argument \`mode\` expected to be one of "CONSTANT", "REFLECT", or "SYMMETRIC". Received \`mode\` = ${h.str(P.mode)}`)
    const maps = t.shape.map((n, d) => {
      const lo = pv[d * 2]
      const hi = pv[d * 2 + 1]
      if (mode === 'CONSTANT') return [...new Array<number>(lo).fill(-1), ...allOf(n), ...new Array<number>(hi).fill(-1)]
      const limit = mode === 'REFLECT' ? n - 1 : n
      if (lo > limit || hi > limit) throw tfErr(h, `paddings must be no greater than ${limit === n ? 'tensor dim' : 'tensor dim - 1'}, got ${Math.max(lo, hi)} for dim ${n} [Op:MirrorPad]`)
      const shift = mode === 'REFLECT' ? 0 : 1
      const before = Array.from({ length: lo }, (_, i) => lo - i - shift)
      const after = Array.from({ length: hi }, (_, i) => n - 1 - (i + 1) + shift)
      return [...before, ...allOf(n), ...after]
    })
    return moved('pad', P.tensor, t, C.crossSrc(t.shape, maps), maps.map((m) => m.length), mode === 'CONSTANT' ? h.num(P.constant_values) : 0)
  })
  const tensordot = fnv('tensordot', (args, kw) => {
    const P = bindArgs(h, 'tensordot', 'a b axes name=None', args, kw, null)
    const a = T(P.a, 'tensordot')
    const b = T(P.b, 'tensordot')
    if (a.dtype !== b.dtype) throw tfErr(h, `cannot compute MatMul as input #1(zero-based) was expected to be a ${TF_KIND[a.dtype]} tensor but is a ${TF_KIND[b.dtype]} tensor [Op:MatMul]`)
    let axA: number[]
    let axB: number[]
    if (P.axes.k === 'int') {
      const n = P.axes.v
      if (n < 0 || n > a.ndim || n > b.ndim) throw h.err('ValueError', `'axes' must be an integer or have length 2.`)
      axA = allOf(n).map((i) => a.ndim - n + i)
      axB = allOf(n)
    } else {
      const parts = isSeq(P.axes) ? P.axes.items : []
      if (parts.length !== 2) throw h.err('ValueError', `'axes' must be an integer or have length 2.`)
      axA = (axisArg(parts[0]) ?? []).map((x) => C.normDim(x, a.ndim))
      axB = (axisArg(parts[1]) ?? []).map((x) => C.normDim(x, b.ndim))
    }
    if (axA.length !== axB.length || axA.some((x, i) => a.shape[x] !== b.shape[axB[i]])) throw tfErr(h, `Matrix size-incompatible: In[0]: [${a.shape.join(',')}], In[1]: [${b.shape.join(',')}] [Op:MatMul]`)
    const freeA = a.shape.map((_, i) => i).filter((i) => !axA.includes(i))
    const freeB = b.shape.map((_, i) => i).filter((i) => !axB.includes(i))
    const c = prod(axA.map((x) => a.shape[x]))
    const pa = C.reshapeT(E, C.permuteT(E, a, [...freeA, ...axA]), [prod(freeA.map((x) => a.shape[x])), c], false)
    const pb = C.reshapeT(E, C.permuteT(E, b, [...axB, ...freeB]), [c, prod(freeB.map((x) => b.shape[x]))], false)
    const out = C.reshapeT(E, C.matmul(E, pa, pb), [...freeA.map((x) => a.shape[x]), ...freeB.map((x) => b.shape[x])], false)
    return traced('tensordot', 'linalg', [{ label: label(P.a, 'a'), t: a }, { label: label(P.b, 'b'), t: b }], out)
  })
  const norm = fnv('norm', (args, kw) => {
    const P = bindArgs(h, 'norm', 'tensor ord=euclidean axis=None keepdims=None name=None', args, kw, null)
    const t = T(P.tensor, 'norm')
    if (!isFloatT(t.dtype)) throw tfErr(h, `Value for attr 'T' of ${t.dtype} is not in the list of allowed values: float, double`)
    const axes = axisArg(P.axis)
    const keep = !isNone(P.keepdims) && h.truthy(P.keepdims)
    const ord = P.ord
    const isEuclid = (ord.k === 'str' && (ord.v === 'euclidean' || ord.v === 'fro')) || (ord.k !== 'str' && num2(ord) === 2)
    if (ord.k === 'str' && !isEuclid) throw h.err('ValueError', `'ord' must be a supported vector norm, got ${ord.v}.`)
    if (axes && axes.length === 2 && !(ord.k === 'str')) throw h.err('NotImplementedError', `tf.norm: a matrix norm with ord=${repr(ord)} is not implemented in this sandbox`)
    if (!isEuclid && num2(ord) <= 0) throw h.err('ValueError', `'ord' must be a supported vector norm, got ${num2(ord)}.`)
    // the Euclidean norm is built from sqrt and sum, so at zero its gradient is NaN, as TensorFlow's is
    const out = isEuclid ? C.unary(E, 'sqrt', C.reduceSum(E, C.unary(E, 'square', t), axes, keep, false)) : C.vectorNorm(E, t, num2(ord), axes, keep)
    return traced('norm', 'reduce', [{ label: label(P.tensor), t }], out, axes && axes.length === 1 ? C.normDim(axes[0], t.ndim) : null)
  })
  const num2 = (v: Value) => h.num(v)
  const matmul = fnv('matmul', (args, kw) => {
    const P = bindArgs(h, 'matmul', 'a b transpose_a=False transpose_b=False !adjoint_a=False !adjoint_b=False a_is_sparse=False b_is_sparse=False output_type=None grad_a=False grad_b=False name=None', args, kw, null)
    let a = T(P.a, 'matmul')
    let b = T(P.b, 'matmul')
    if (a.dtype !== b.dtype) throw tfErr(h, `cannot compute MatMul as input #1(zero-based) was expected to be a ${TF_KIND[a.dtype]} tensor but is a ${TF_KIND[b.dtype]} tensor [Op:MatMul]`)
    if (a.ndim < 2 || b.ndim < 2) throw tfErr(h, `In[0] and In[1] has different ndims: [${a.shape.join(',')}] vs. [${b.shape.join(',')}] [Op:MatMul]`)
    if (h.truthy(P.transpose_a)) a = C.transposeT(E, a, -1, -2)
    if (h.truthy(P.transpose_b)) b = C.transposeT(E, b, -1, -2)
    if (a.shape[a.ndim - 1] !== b.shape[b.ndim - 2]) throw tfErr(h, `Matrix size-incompatible: In[0]: [${a.shape.join(',')}], In[1]: [${b.shape.join(',')}] [Op:MatMul]`)
    return traced('matmul', 'matmul', [{ label: label(P.a, 'a'), t: a }, { label: label(P.b, 'b'), t: b }], C.matmul(E, a, b, 'MatMul'))
  })

  // ---- tf.math
  const round = plain('round', null, (v) => {
    if (!Number.isFinite(v)) return v
    const f = Math.floor(v)
    const d = v - f
    return d < 0.5 ? f : d > 0.5 ? f + 1 : f % 2 === 0 ? f : f + 1
  })
  const floorFn = plain('floor', 'Floor', Math.floor)
  const ceilFn = plain('ceil', 'Ceil', Math.ceil)
  const signFn = plain('sign', null, Math.sign)
  const mathFns: Record<string, Value> = {
    confusion_matrix: confusion, reduce_sum: reduce('reduce_sum', false), reduce_mean: reduce('reduce_mean', true), reduce_max: reduce('reduce_max', false, 'max'), reduce_min: reduce('reduce_min', false, 'min'),
    reduce_prod: reduceProd, reduce_std: reduceSpread('reduce_std', true), reduce_variance: reduceSpread('reduce_variance', false), reduce_any: reduceBool('reduce_any', 'Any'), reduce_all: reduceBool('reduce_all', 'All'),
    exp: elem('exp', 'exp', 'Exp'), log: elem('log', 'log', 'Log'), sqrt: elem('sqrt', 'sqrt', 'Sqrt'), square: elem('square', 'square', null), abs: elem('abs', 'abs', null), negative: elem('negative', 'neg', null),
    sin: elem('sin', 'sin', 'Sin'), cos: elem('cos', 'cos', 'Cos'), tanh: elem('tanh', 'tanh', 'Tanh'), sigmoid: elem('sigmoid', 'sigmoid', 'Sigmoid'),
    tan: smooth('tan', 'Tan', (t) => C.mapT(E, t, Math.tan, (v) => 1 / Math.cos(v) ** 2, 'TanBackward')),
    log1p: smooth('log1p', 'Log1p', (t) => C.mapT(E, t, Math.log1p, (v) => 1 / (1 + v), 'Log1pBackward')),
    expm1: smooth('expm1', 'Expm1', (t) => C.mapT(E, t, Math.expm1, (v) => Math.exp(v), 'Expm1Backward')),
    softplus: smooth('softplus', 'Softplus', (t) => softplusT(E, t)),
    round, floor: floorFn, ceil: ceilFn, sign: signFn,
    is_nan: plain('is_nan', null, (v) => (Number.isNaN(v) ? 1 : 0)), is_inf: plain('is_inf', null, (v) => (Math.abs(v) === Infinity ? 1 : 0)), is_finite: plain('is_finite', null, (v) => (Number.isFinite(v) ? 1 : 0)),
    add: two('add', '+'), multiply: two('multiply', '*'), subtract: two('subtract', '-'), divide: two('divide', '/'), pow: two('pow', '**'), floordiv: two('floordiv', '//'), mod: two('mod', '%'), maximum: two('maximum', 'maximum'), minimum: two('minimum', 'minimum'),
    equal: two('equal', '=='), not_equal: two('not_equal', '!='), greater: two('greater', '>'), greater_equal: two('greater_equal', '>='), less: two('less', '<'), less_equal: two('less_equal', '<='),
    logical_and: logicalFn('logical_and', '&'), logical_or: logicalFn('logical_or', '|'), logical_xor: logicalFn('logical_xor', '^'), logical_not: logicalNot,
    argmax: argExtreme('argmax', false), argmin: argExtreme('argmin', true), count_nonzero: countNonzero, top_k: topK, cumsum,
    divide_no_nan: fnv('divide_no_nan', (args) => {
      const a = T(args[0], 'divide_no_nan')
      const b = tfOperand(h, args[1], a) ?? T(args[1], 'divide_no_nan')
      const shape = bcast(h, a.shape, b.shape)
      const [ia, ib] = [C.bcastIndex(a.shape, shape), C.bcastIndex(b.shape, shape)]
      const [av, bv] = [a.values(), b.values()]
      return tfWrap(TT.of(Array.from({ length: prod(shape) }, (_, i) => (bv[ib[i]] ? av[ia[i]] / bv[ib[i]] : 0)), shape, a.dtype))
    }),
  }
  // the marker dtype for the three is_* predicates
  for (const n of ['is_nan', 'is_inf', 'is_finite']) {
    const f = mathFns[n] as { k: 'fn'; call: (a: Value[], k: Kw) => Value }
    const inner = f.call
    mathFns[n] = fnv(n, (args, kw) => {
      const out = tfOf(inner(args, kw))!
      return tfWrap(TT.of(out.values(), out.shape, 'bool'))
    })
  }
  const tfMath: Value = { k: 'module', name: 'tensorflow.math', attrs: mathFns }

  // ---- tf.nn
  const hw = (v: Value, what: string): [number, number] => {
    if (v.k === 'int') return [v.v, v.v]
    const xs = h.iterate(v).map((x) => h.toInt(x))
    if (xs.length === 1) return [xs[0], xs[0]]
    if (xs.length === 2) return [xs[0], xs[1]]
    if (xs.length === 4) return [xs[1], xs[2]]
    throw h.err('ValueError', `${what} should be of length 1, 2 or 4 but was ${xs.length}`)
  }
  const tfPad = (v: Value, what: string): PadMode => {
    const s = h.str(v).toUpperCase()
    if (s !== 'SAME' && s !== 'VALID') throw h.err('NotImplementedError', `${what}: padding=${repr(v)} is not implemented in this sandbox`)
    return s === 'SAME' ? 'same' : 'valid'
  }
  const poolFn = (name: string, mode: 'max' | 'avg') => fnv(name, (args, kw) => {
    const P = bindArgs(h, name, 'input ksize strides padding !data_format=NHWC name=None', args, kw, null)
    const t = T(P.input, name)
    if (t.ndim !== 4) throw h.err('ValueError', `Input tensor must be of rank 4, got shape ${shapeStr(t.shape)}`)
    return tfWrap(poolNHWC(E, t, hw(P.ksize, 'ksize'), hw(P.strides, 'strides'), tfPad(P.padding, name), mode))
  })
  const smCE = fnv('softmax_cross_entropy_with_logits', (args, kw) => {
    const P = bindArgs(h, 'softmax_cross_entropy_with_logits', 'labels logits axis=-1 name=None', args, kw, null)
    const z = T(P.logits, 'softmax_cross_entropy_with_logits')
    const y = toTF(h, P.labels, z.dtype)
    if (y.shape.join() !== z.shape.join()) throw tfErr(h, `logits and labels must be broadcastable: logits_size=[${z.shape.join(',')}] labels_size=[${y.shape.join(',')}] [Op:SoftmaxCrossEntropyWithLogits]`)
    const axis = h.toInt(P.axis)
    return tfWrap(C.reduceSum(E, C.binary(E, '*', C.unary(E, 'neg', y), C.softmax(E, z, axis, true)), [axis], false, false))
  })
  const sparseCE = fnv('sparse_softmax_cross_entropy_with_logits', (args, kw) => {
    const P = bindArgs(h, 'sparse_softmax_cross_entropy_with_logits', 'labels logits name=None', args, kw, null)
    const z = T(P.logits, 'sparse_softmax_cross_entropy_with_logits')
    const y = T(P.labels, 'sparse_softmax_cross_entropy_with_logits')
    if (isFloatT(y.dtype)) throw tfErr(h, `Value for attr 'Tlabels' of ${y.dtype} is not in the list of allowed values: int32, int64`)
    if (y.shape.join() !== z.shape.slice(0, -1).join()) throw h.err('ValueError', `\`labels.shape.rank\` must equal \`logits.shape.rank - 1\`. Received: labels.shape=${shapeStr(y.shape)} and logits.shape=${shapeStr(z.shape)}`)
    const d = z.shape[z.ndim - 1]
    const labels = y.values()
    for (const l of labels) if (l < 0 || l >= d) throw tfErr(h, `Received a label value of ${l} which is outside the valid range of [0, ${d}).  Label values: ${labels.join(' ')} [Op:SparseSoftmaxCrossEntropyWithLogits]`)
    const per = C.nllLoss(E, C.reshapeT(E, C.softmax(E, z, -1, true), [labels.length, d], false), TT.of(labels, [labels.length], 'int64'), 'none')
    return tfWrap(C.reshapeT(E, per, y.shape, false))
  })
  const sigCE = fnv('sigmoid_cross_entropy_with_logits', (args, kw) => {
    const P = bindArgs(h, 'sigmoid_cross_entropy_with_logits', 'labels=None logits=None name=None', args, kw, null)
    const z = T(P.logits, 'sigmoid_cross_entropy_with_logits')
    const y = toTF(h, P.labels, z.dtype)
    if (y.shape.join() !== z.shape.join()) throw h.err('ValueError', `\`logits\` and \`labels\` must have the same shape, received (${shapeStr(z.shape)} vs ${shapeStr(y.shape)}).`)
    return tfWrap(C.bce(E, z, y, true, 'none'))
  })
  const conv2dFn = fnv('conv2d', (args, kw) => {
    const P = bindArgs(h, 'conv2d', 'input filters strides padding !data_format=NHWC dilations=None name=None', args, kw, null)
    const x = T(P.input, 'conv2d')
    const w = T(P.filters, 'conv2d')
    if (x.dtype !== w.dtype) throw tfErr(h, `cannot compute Conv2D as input #1(zero-based) was expected to be a ${TF_KIND[x.dtype]} tensor but is a ${TF_KIND[w.dtype]} tensor [Op:Conv2D]`)
    if (x.ndim !== 4 || w.ndim !== 4) throw h.err('ValueError', `Shape must be rank 4 but is rank ${x.ndim !== 4 ? x.ndim : w.ndim}`)
    if (x.shape[3] !== w.shape[2]) throw tfErr(h, `input and filter must have the same depth: ${x.shape[3]} vs ${w.shape[2]} [Op:Conv2D]`)
    return tfWrap(convNHWC(E, x, w, null, hw(P.strides, 'strides'), isNone(P.dilations) ? [1, 1] : hw(P.dilations, 'dilations'), tfPad(P.padding, 'conv2d')))
  })
  const nnFns: Record<string, Value> = {
    relu: elem('relu', 'relu', null), sigmoid: mathFns.sigmoid, tanh: mathFns.tanh, softplus: mathFns.softplus,
    relu6: smooth('relu6', 'Relu6', (t) => relu6T(E, t)),
    elu: smooth('elu', 'Elu', (t) => eluT(E, t)),
    selu: smooth('selu', 'Selu', (t) => seluT(E, t)),
    silu: smooth('silu', 'Silu', (t) => silu(E, t)),
    swish: smooth('swish', 'Silu', (t) => silu(E, t)),
    softsign: smooth('softsign', 'Softsign', (t) => ACTS.softsign(E, t)),
    gelu: fnv('gelu', (args, kw) => {
      const P = bindArgs(h, 'gelu', 'features approximate=False name=None', args, kw, null)
      return tfWrap(geluT(E, T(P.features, 'gelu'), h.truthy(P.approximate)))
    }),
    leaky_relu: fnv('leaky_relu', (args, kw) => {
      const P = bindArgs(h, 'leaky_relu', 'features alpha=0.2 name=None', args, kw, null)
      const t = T(P.features, 'leaky_relu')
      return traced('nn.leaky_relu', 'elementwise', [{ label: label(P.features), t }], leakyT(E, t, h.num(P.alpha)))
    }),
    softmax: fnv('softmax', (args, kw) => {
      const P = bindArgs(h, 'softmax', 'logits axis=None name=None', args, kw, null)
      return tfWrap(C.softmax(E, T(P.logits, 'softmax'), isNone(P.axis) ? -1 : h.toInt(P.axis), false))
    }),
    log_softmax: fnv('log_softmax', (args, kw) => {
      const P = bindArgs(h, 'log_softmax', 'logits axis=None name=None', args, kw, null)
      return tfWrap(C.softmax(E, T(P.logits, 'log_softmax'), isNone(P.axis) ? -1 : h.toInt(P.axis), true))
    }),
    dropout: fnv('dropout', (args, kw) => {
      const P = bindArgs(h, 'dropout', 'x rate !noise_shape=None seed=None name=None', args, kw, null)
      const rate = h.num(P.rate)
      if (rate < 0 || rate >= 1) throw h.err('ValueError', `\`rate\` must be a scalar tensor or a float in the range [0, 1). Received: rate=${rate}`)
      return tfWrap(C.dropout(E, T(P.x, 'dropout'), rate, true))
    }),
    conv2d: conv2dFn,
    max_pool2d: poolFn('max_pool2d', 'max'), avg_pool2d: poolFn('avg_pool2d', 'avg'), max_pool: poolFn('max_pool', 'max'), avg_pool: poolFn('avg_pool', 'avg'),
    softmax_cross_entropy_with_logits: smCE, sparse_softmax_cross_entropy_with_logits: sparseCE, sigmoid_cross_entropy_with_logits: sigCE,
    l2_loss: fnv('l2_loss', (args, kw) => {
      const t = T(args[0] ?? kw.t, 'l2_loss')
      return tfWrap(mulS(E, C.reduceSum(E, C.unary(E, 'square', t), null, false, false), 0.5))
    }),
    l2_normalize: fnv('l2_normalize', (args, kw) => {
      const P = bindArgs(h, 'l2_normalize', 'x axis=None epsilon=1e-12 name=None', args, kw, null)
      const t = T(P.x, 'l2_normalize')
      const axes = axisArg(P.axis)
      const s = C.reduceSum(E, C.unary(E, 'square', t), axes, true, false)
      return tfWrap(C.binary(E, '/', t, C.unary(E, 'sqrt', C.clamp(E, s, h.num(P.epsilon), null))))
    }),
    moments: fnv('moments', (args, kw) => {
      const P = bindArgs(h, 'moments', 'x axes !shift=None keepdims=False name=None', args, kw, null)
      const t = T(P.x, 'moments')
      const axes = axisArg(P.axes)
      const keep = h.truthy(P.keepdims)
      const mean = C.reduceSum(E, t, axes, keep, true)
      const m2 = C.reduceSum(E, t, axes, true, true)
      return py.tuple([tfWrap(mean), tfWrap(C.reduceSum(E, C.unary(E, 'square', C.binary(E, '-', t, m2)), axes, keep, true))])
    }),
    top_k: topK,
    bias_add: fnv('bias_add', (args) => tfWrap(C.binary(E, '+', T(args[0], 'bias_add'), T(args[1], 'bias_add')))),
    embedding_lookup: fnv('embedding_lookup', (args, kw) => {
      const P = bindArgs(h, 'embedding_lookup', 'params ids !max_norm=None name=None', args, kw, null)
      return h.call(gather, [P.params, P.ids])
    }),
  }
  const tfNN: Value = { k: 'module', name: 'tensorflow.nn', attrs: nnFns }

  // ---- tf.random
  const randomValues = (name: string, sig: string, draw: (P: Params) => () => number) => fnv(name, (args, kw) => {
    const P = bindArgs(h, name, sig, args, kw, null)
    const shape = shapeOf(P.shape)
    const f = draw(P)
    const n = prod(shape)
    const vals = new Array<number>(n)
    for (let i = 0; i < n; i++) vals[i] = f()
    return traced(`random.${name}`, 'create', [], TT.of(vals, shape, dt(P.dtype) ?? 'float32'))
  })
  const random: Value = {
    k: 'module', name: 'tensorflow.random', attrs: {
      set_seed: fnv('set_seed', (args) => { h.rng.seed(h.toInt(args[0])); return py.NONE }),
      normal: randomValues('normal', 'shape mean=0.0 stddev=1.0 dtype=float32 seed=None name=None', (P) => {
        const [m, s] = [h.num(P.mean), h.num(P.stddev)]
        return () => m + s * h.rng.normal()
      }),
      truncated_normal: randomValues('truncated_normal', 'shape mean=0.0 stddev=1.0 dtype=float32 seed=None name=None', (P) => {
        const [m, s] = [h.num(P.mean), h.num(P.stddev)]
        return () => m + s * truncNormal(h.rng)
      }),
      uniform: randomValues('uniform', 'shape minval=0 maxval=None dtype=float32 seed=None name=None', (P) => {
        const d = dt(P.dtype) ?? 'float32'
        if (!isFloatT(d) && isNone(P.maxval)) throw h.err('ValueError', `Must specify maxval for integer dtype ${d}`)
        const lo = h.num(P.minval)
        const hi = isNone(P.maxval) ? 1 : h.num(P.maxval)
        return isFloatT(d) ? () => lo + (hi - lo) * h.rng.next() : () => Math.floor(lo + (hi - lo) * h.rng.next())
      }),
      shuffle: fnv('shuffle', (args, kw) => {
        const P = bindArgs(h, 'shuffle', 'value seed=None name=None', args, kw, null)
        const t = T(P.value, 'shuffle')
        if (!t.ndim) throw tfErr(h, 'input must be at least 1-D, got shape [] [Op:RandomShuffle]')
        const order = allOf(t.shape[0])
        for (let i = order.length - 1; i > 0; i--) {
          const j = Math.floor(h.rng.next() * (i + 1))
          ;[order[i], order[j]] = [order[j], order[i]]
        }
        return moved('random.shuffle', P.value, t, C.crossSrc(t.shape, [order, ...t.shape.slice(1).map(allOf)]), t.shape)
      }),
      categorical: fnv('categorical', (args, kw) => {
        const P = bindArgs(h, 'categorical', 'logits num_samples dtype=int64 seed=None name=None', args, kw, null)
        const z = T(P.logits, 'categorical')
        if (z.ndim !== 2) throw tfErr(h, `logits should be a matrix, got shape ${shapeStr(z.shape)} [Op:Multinomial]`)
        const [B, Cc] = z.shape
        const k = h.toInt(P.num_samples)
        const v = z.values()
        const out: number[] = []
        for (let b = 0; b < B; b++) {
          const p = allOf(Cc).map((c) => Math.exp(v[b * Cc + c]))
          const tot = p.reduce((a, x) => a + x, 0)
          for (let s = 0; s < k; s++) {
            let u = h.rng.next() * tot
            let c = 0
            while (c < Cc - 1 && u >= p[c]) u -= p[c++]
            out.push(c)
          }
        }
        return tfWrap(TT.of(out, [B, k], dt(P.dtype) ?? 'int64'))
      }),
    },
  }

  // ---- tf.linalg
  const square = (t: TT, api: string) => {
    if (t.ndim < 2 || t.shape[t.ndim - 2] !== t.shape[t.ndim - 1]) throw tfErr(h, `Input matrix must be square. [Op:${api}]`)
  }
  /** the engine's factorisation fails only on a singular matrix, once squareness is checked */
  const invertible = (run: () => TT, message: string): TT => {
    try {
      return run()
    } catch (e) {
      if (e instanceof PyError) throw tfErr(h, message)
      throw e
    }
  }
  const inverse = fnv('inv', (args, kw) => {
    const P = bindArgs(h, 'inv', 'input adjoint=False name=None', args, kw, null)
    const t = T(P.input, 'inv')
    floatOnly(t, 'MatrixInverse')
    square(t, 'MatrixInverse')
    const a = h.truthy(P.adjoint) ? C.transposeT(E, t, -1, -2) : t
    return traced('linalg.inv', 'linalg', [{ label: label(P.input), t }], invertible(() => C.invT(E, a), 'Input is not invertible. [Op:MatrixInverse]'))
  })
  const determinant = fnv('det', (args, kw) => {
    const t = T(args[0] ?? kw.input, 'det')
    floatOnly(t, 'MatrixDeterminant')
    square(t, 'MatrixDeterminant')
    return traced('linalg.det', 'linalg', [{ label: label(args[0]), t }], C.detT(E, t))
  })
  const traceFn = fnv('trace', (args, kw) => {
    const t = T(args[0] ?? kw.x, 'trace')
    if (t.ndim < 2) throw h.err('ValueError', `Shape must be at least rank 2 but is rank ${t.ndim}`)
    const [r, c] = t.shape.slice(-2)
    const m = Math.min(r, c)
    const batches = t.size / (r * c)
    const v = t.values()
    const out = TT.of(allOf(batches).map((b) => allOf(m).reduce((s, i) => s + v[b * r * c + i * c + i], 0)), t.shape.slice(0, -2), t.dtype)
    return traced('linalg.trace', 'linalg', [{ label: label(args[0]), t }], opT(E, out, 'TraceBackward', [t], (g) => {
      const ga = new Array<number>(t.size).fill(0)
      for (let b = 0; b < batches; b++) for (let i = 0; i < m; i++) ga[b * r * c + i * c + i] = g[b]
      return [ga]
    }))
  })
  const eye = fnv('eye', (args, kw) => {
    const P = bindArgs(h, 'eye', 'num_rows num_columns=None batch_shape=None dtype=float32 name=None', args, kw, null)
    const r = h.toInt(P.num_rows)
    const c = isNone(P.num_columns) ? r : h.toInt(P.num_columns)
    const batch = isNone(P.batch_shape) ? [] : shapeOf(P.batch_shape)
    const vals = new Array<number>(r * c * prod(batch)).fill(0)
    for (let b = 0; b < prod(batch); b++) for (let i = 0; i < Math.min(r, c); i++) vals[b * r * c + i * c + i] = 1
    return traced('eye', 'create', [], TT.of(vals, [...batch, r, c], dt(P.dtype) ?? 'float32'))
  })
  const linalg: Value = {
    k: 'module', name: 'tensorflow.linalg', attrs: {
      inv: inverse, det: determinant, matmul, trace: traceFn, norm, eye,
      diag: fnv('diag', (args) => {
        const t = T(args[0], 'diag')
        const n = t.shape[t.ndim - 1]
        const b = t.size / n
        const src = new Array<number>(b * n * n).fill(-1)
        for (let k = 0; k < b; k++) for (let i = 0; i < n; i++) src[k * n * n + i * n + i] = k * n + i
        return moved('linalg.diag', args[0], t, src, [...t.shape.slice(0, -1), n, n])
      }),
      diag_part: fnv('diag_part', (args) => {
        const t = T(args[0], 'diag_part')
        const [r, c] = t.shape.slice(-2)
        const m = Math.min(r, c)
        const b = t.size / (r * c)
        const src: number[] = []
        for (let k = 0; k < b; k++) for (let i = 0; i < m; i++) src.push(k * r * c + i * c + i)
        return moved('linalg.diag_part', args[0], t, src, [...t.shape.slice(0, -2), m])
      }),
      matrix_transpose: fnv('matrix_transpose', (args) => tfWrap(C.transposeT(E, T(args[0], 'matrix_transpose'), -1, -2))),
      solve: fnv('solve', (args, kw) => {
        const P = bindArgs(h, 'solve', 'matrix rhs !adjoint=False name=None', args, kw, null)
        const a = T(P.matrix, 'solve')
        const b = T(P.rhs, 'solve')
        square(a, 'MatrixSolve')
        if (b.ndim !== a.ndim) throw tfErr(h, `Input matrix and rhs must have the same rank, got ${a.ndim} != ${b.ndim} [Op:MatrixSolve]`)
        return tfWrap(invertible(() => C.solveT(E, a, b), 'Input matrix is not invertible. [Op:MatrixSolve]'))
      }),
    },
  }

  // ---- tf.image
  const imageRank = (t: TT) => {
    if (t.ndim !== 3 && t.ndim !== 4) throw h.err('ValueError', `'image' (shape ${shapeStr(t.shape)}) must have either 3 or 4 dimensions.`)
  }
  const toFloatImage = (t: TT): TT => (isFloatT(t.dtype) ? t : TT.of(t.dtype === 'uint8' ? t.values().map((v) => v / 255) : t.values(), t.shape, 'float32'))
  const flip = (name: string, axisFromEnd: number) => fnv(name, (args, kw) => {
    const v = args[0] ?? kw.image
    const t = T(v, name)
    imageRank(t)
    const axis = t.ndim - axisFromEnd
    const maps = t.shape.map((n, d) => (d === axis ? allOf(n).reverse() : allOf(n)))
    return moved(`image.${name}`, v, t, C.crossSrc(t.shape, maps), t.shape)
  })
  const image: Value = {
    k: 'module', name: 'tensorflow.image', attrs: {
      flip_left_right: flip('flip_left_right', 2),
      flip_up_down: flip('flip_up_down', 3),
      resize: fnv('resize', (args, kw) => {
        const P = bindArgs(h, 'resize', 'images size method=bilinear !preserve_aspect_ratio=False !antialias=False name=None', args, kw, null)
        const t = T(P.images, 'resize')
        if (t.ndim !== 3 && t.ndim !== 4) throw h.err('ValueError', "'images' must have either 3 or 4 dimensions.")
        const [Ho, Wo] = shapeOf(P.size)
        const m = h.str(P.method)
        if (m !== 'nearest' && m !== 'bilinear') throw h.err('NotImplementedError', `tf.image.resize: method='${m}' is not implemented in this sandbox`)
        return traced('image.resize', 'info', [{ label: label(P.images), t }], resizeT(E, t, Ho, Wo, m))
      }),
      rgb_to_grayscale: fnv('rgb_to_grayscale', (args, kw) => {
        const v = args[0] ?? kw.images
        const t = T(v, 'rgb_to_grayscale')
        if (t.shape[t.ndim - 1] !== 3) throw tfErr(h, `Dimensions must be equal, but are ${t.shape[t.ndim - 1]} and 3`)
        const f = toFloatImage(t)
        const w = TT.of([0.2989, 0.587, 0.114], [3], f.dtype)
        const gray = C.unsqueezeT(E, C.reduceSum(E, C.binary(E, '*', f, w), [-1], false, false), -1)
        // convert_image_dtype back: a uint8 result is scaled by 255.5 and truncated
        return tfWrap(t.dtype === 'uint8' ? TT.of(gray.values().map((x) => Math.trunc(x * 255.5)), gray.shape, 'uint8') : gray)
      }),
      grayscale_to_rgb: fnv('grayscale_to_rgb', (args, kw) => {
        const v = args[0] ?? kw.images
        const t = T(v, 'grayscale_to_rgb')
        if (t.shape[t.ndim - 1] !== 1) throw h.err('ValueError', `Dimension -1 in both shapes must be equal, but are ${t.shape[t.ndim - 1]} and 1.`)
        const maps = t.shape.map((n, d) => (d === t.ndim - 1 ? [0, 0, 0] : allOf(n)))
        return moved('image.grayscale_to_rgb', v, t, C.crossSrc(t.shape, maps), [...t.shape.slice(0, -1), 3])
      }),
      convert_image_dtype: fnv('convert_image_dtype', (args, kw) => {
        const P = bindArgs(h, 'convert_image_dtype', 'image dtype saturate=False name=None', args, kw, null)
        const t = T(P.image, 'convert_image_dtype')
        const d = dt(P.dtype)!
        if (d === t.dtype) return tfWrap(t)
        if (isFloatT(d)) return tfWrap(TT.of(t.dtype === 'uint8' ? t.values().map((v) => v / 255) : t.values(), t.shape, d))
        if (d === 'uint8' && isFloatT(t.dtype)) return tfWrap(TT.of(t.values().map((v) => Math.trunc(v * 255.5)), t.shape, 'uint8'))
        throw h.err('NotImplementedError', `convert_image_dtype from ${t.dtype} to ${d} is not implemented in this sandbox`)
      }),
      rot90: fnv('rot90', (args, kw) => {
        const P = bindArgs(h, 'rot90', 'image k=1 name=None', args, kw, null)
        let t = T(P.image, 'rot90')
        imageRank(t)
        const k = ((h.toInt(P.k) % 4) + 4) % 4
        const hAx = t.ndim - 3
        const wAx = t.ndim - 2
        for (let i = 0; i < k; i++) t = flipAxes(E, C.transposeT(E, t, hAx, wAx), [hAx])
        return tfWrap(t)
      }),
      random_flip_left_right: fnv('random_flip_left_right', (args, kw) => {
        const t = T(args[0] ?? kw.image, 'random_flip_left_right')
        return tfWrap(perImage(t, (img, H, W, Cc) => resample(img, H, W, Cc, h.rng.next() < 0.5 ? (y, x) => [y, W - 1 - x] : (y, x) => [y, x], 'nearest', 0, false)))
      }),
      random_flip_up_down: fnv('random_flip_up_down', (args, kw) => {
        const t = T(args[0] ?? kw.image, 'random_flip_up_down')
        return tfWrap(perImage(t, (img, H, W, Cc) => resample(img, H, W, Cc, h.rng.next() < 0.5 ? (y, x) => [H - 1 - y, x] : (y, x) => [y, x], 'nearest', 0, false)))
      }),
    },
  }

  // ---- tf.data
  const Dataset: Value = {
    k: 'module', name: 'tensorflow.data.Dataset', attrs: {
      from_tensor_slices: fnv('from_tensor_slices', (args, kw) => datasetOf('_TensorSliceDataset', sliceElements(h, args[0] ?? kw.tensors))),
      from_tensors: fnv('from_tensors', (args, kw) => {
        const v = args[0] ?? kw.tensors
        return datasetOf('_TensorDataset', [v.k === 'tuple' ? py.tuple(v.items.map((x) => tfWrap(T(x, 'from_tensors')))) : tfWrap(T(v, 'from_tensors'))])
      }),
      from_generator: fnv('from_generator', (args, kw) => {
        const P = bindArgs(h, 'from_generator', 'generator output_types=None output_shapes=None args=None output_signature=None name=None', args, kw, null)
        if (isNone(P.output_signature) && isNone(P.output_types)) throw h.err('TypeError', 'To specify the output signature you need to provide either the `output_signature` argument or the `output_types` argument.')
        const f = P.generator
        const given = isNone(P.args) ? [] : h.iterate(P.args)
        // each element is cast to the dtype the signature declares
        const typeOf = (sig: Value): TDType | null => (sig.k === 'obj' && sig.o instanceof TensorSpec ? (sig.o.dtype === 'string' ? null : sig.o.dtype) : dt(sig))
        const conv = (x: Value, sig: Value): Value => {
          if (isSeq(sig)) return py.tuple(h.iterate(x).map((p, i) => conv(p, sig.items[i])))
          const strs = strsOf(x)
          if (strs) return py.obj(new TFString(strs, x.k === 'str' ? [] : [strs.length]))
          return tfWrap(toTF(h, x, typeOf(sig)))
        }
        const sig = isNone(P.output_signature) ? P.output_types : P.output_signature
        const specText = (v: Value): string => (isSeq(v) ? `(${v.items.map(specText).join(', ')}${v.items.length === 1 ? ',' : ''})` : v.k === 'obj' && v.o instanceof TensorSpec ? v.o.repr() : new TensorSpec(null, typeOf(v) ?? 'string', py.NONE).repr())
        return py.obj(new TFDataset('_FlatMapDataset', function* (hh) {
          const next = hh.iterOf(hh.call(f, given))
          for (let x = next(); x !== undefined; x = next()) yield conv(x, sig)
        }, () => specText(sig), false, null))
      }),
      range: fnv('range', (args) => {
        const nums = args.map((a) => h.toInt(a))
        const [a, b, s] = nums.length === 1 ? [0, nums[0], 1] : [nums[0], nums[1], nums[2] ?? 1]
        const n = Math.max(0, Math.ceil((b - a) / s))
        return datasetOf('_RangeDataset', allOf(n).map((i) => tfWrap(TT.scalar(a + i * s, 'int64'))))
      }),
    },
  }
  const data: Value = { k: 'module', name: 'tensorflow.data', attrs: { Dataset, AUTOTUNE: py.int(-1), experimental: { k: 'module', name: 'tensorflow.data.experimental', attrs: { AUTOTUNE: py.int(-1) } } } }

  // ---- tf.config, tf.test, tf.strings
  const none = (name: string) => fnv(name, () => py.NONE)
  const emptyList = (name: string) => fnv(name, () => py.list([]))
  const config: Value = {
    k: 'module', name: 'tensorflow.config', attrs: {
      list_physical_devices: emptyList('list_physical_devices'),
      list_logical_devices: emptyList('list_logical_devices'),
      get_visible_devices: emptyList('get_visible_devices'),
      set_visible_devices: none('set_visible_devices'),
      run_functions_eagerly: none('run_functions_eagerly'),
      experimental: { k: 'module', name: 'tensorflow.config.experimental', attrs: { set_memory_growth: none('set_memory_growth'), list_physical_devices: emptyList('list_physical_devices') } },
    },
  }
  const testMod: Value = {
    k: 'module', name: 'tensorflow.test', attrs: {
      is_gpu_available: fnv('is_gpu_available', () => py.bool(false)),
      gpu_device_name: fnv('gpu_device_name', () => py.str('')),
      is_built_with_cuda: fnv('is_built_with_cuda', () => py.bool(false)),
    },
  }
  const strT = (v: Value, what: string): TFString => {
    if (v.k === 'obj' && v.o instanceof TFString) return v.o
    const strs = strsOf(v)
    if (!strs) throw h.err('TypeError', `${what}() expects a string tensor`)
    return new TFString(strs, v.k === 'str' ? [] : [strs.length])
  }
  const strMap = (name: string, f: (s: string) => string) => fnv(name, (args, kw) => {
    const t = strT(args[0] ?? kw.input, name)
    return py.obj(new TFString(t.strs.map(f), t.shape))
  })
  const strings: Value = {
    k: 'module', name: 'tensorflow.strings', attrs: {
      lower: strMap('lower', (s) => s.toLowerCase()),
      upper: strMap('upper', (s) => s.toUpperCase()),
      strip: strMap('strip', (s) => s.trim()),
      length: fnv('length', (args, kw) => {
        const t = strT(args[0] ?? kw.input, 'length')
        return tfWrap(TT.of(t.strs.map((s) => new TextEncoder().encode(s).length), t.shape, 'int32'))
      }),
      join: fnv('join', (args, kw) => {
        const sep = kw.separator ?? args[1]
        const parts = h.iterate(args[0] ?? kw.inputs).map((x) => strT(x, 'join').strs[0])
        return py.obj(new TFString([parts.join(sep ? h.str(sep) : '')], []))
      }),
    },
  }

  const errors: Value = { k: 'module', name: 'tensorflow.errors', attrs: { ...tfErrorClasses(h) } }

  // ---- tf itself
  const fill = (name: string, v: number) => fnv(name, (args, kw) => {
    const P = bindArgs(h, name, 'shape dtype=float32 name=None', args, kw, null)
    const shape = shapeOf(P.shape)
    return traced(name, 'create', [], TT.of(new Array(prod(shape)).fill(v), shape, dt(P.dtype) ?? 'float32'))
  })
  const fillLike = (name: string, v: number) => fnv(name, (args, kw) => {
    const P = bindArgs(h, name, 'input dtype=None name=None', args, kw, null)
    const t = T(P.input, name)
    return traced(name, 'create', [], TT.of(new Array(t.size).fill(v), t.shape, dt(P.dtype) ?? t.dtype))
  })
  const tfPrint = fnv('print', (args, kw) => {
    const summarize = isNone(kw.summarize) ? 3 : h.toInt(kw.summarize!)
    const sep = kw.sep ? h.str(kw.sep) : ' '
    const end = kw.end ? h.str(kw.end) : '\n'
    h.print(args.map((a) => {
      const t = tfOf(a)
      return t ? tfPrintTensor(t, summarize) : h.str(a)
    }).join(sep) + end)
    return py.NONE
  })
  const tf: Record<string, Value> = {
    ...tfd,
    ...mathFns,
    __version__: py.str('2.20 (sandbox)'),
    keras, math: tfMath, nn: tfNN, random, linalg, image, data, config, test: testMod, strings, errors,
    losses: kerasAttrs.losses, metrics: kerasAttrs.metrics, optimizers: K.optimizers, initializers: K.initializers, layers: kerasAttrs.layers,
    newaxis: py.NONE,
    constant: fnv('constant', (args, kw) => {
      const v = args[0] ?? kw.value
      const dv = kw.dtype ?? args[1]
      const shapeV = kw.shape ?? args[2]
      const strs = v ? strsOf(v) : null
      if (strs || (dv && dv.k === 'obj' && dv.o instanceof TFDtype && dv.o.d === 'string')) {
        if (!strs) throw h.err('TypeError', `Failed to convert elements of ${repr(v)} to Tensor.`)
        return py.obj(new TFString(strs, v.k === 'str' ? [] : [strs.length]))
      }
      let t = toTF(h, v, dt(dv))
      if (!isNone(shapeV)) {
        const shape = shapeOf(shapeV!)
        if (t.size === 1) t = TT.of(new Array(prod(shape)).fill(t.values()[0]), shape, t.dtype)
        else if (t.size === prod(shape)) t = TT.of(t.values(), shape, t.dtype)
        else throw h.err('TypeError', `Eager execution of tf.constant with unsupported shape. Tensor ${new TFTensor(t).repr()} (converted from ${repr(v)}) has ${t.size} elements, but got \`shape\` ${shapeStr(shape)} with ${prod(shape)} elements).`)
      }
      traceTF(h, 'tf.constant', 'create', [], t)
      return tfWrap(t)
    }),
    convert_to_tensor: fnv('convert_to_tensor', (args, kw) => tfWrap(toTF(h, args[0] ?? kw.value, dt(kw.dtype ?? args[1])))),
    Variable: { k: 'type', name: 'Variable', call: (args, kw) => {
      const base = T(args[0] ?? kw.initial_value, 'Variable')
      const t = TT.of(base.values(), base.shape, dt(kw.dtype) ?? base.dtype)
      return py.obj(new TFVariable(t, kw.name ? h.str(kw.name) : 'Variable', !kw.trainable || h.truthy(kw.trainable)))
    } },
    GradientTape: { k: 'type', name: 'GradientTape', call: (_a, kw) => py.obj(new GradientTape(!!(kw.persistent && h.truthy(kw.persistent)))) },
    zeros: fill('zeros', 0), ones: fill('ones', 1), zeros_like: fillLike('zeros_like', 0), ones_like: fillLike('ones_like', 1),
    fill: fnv('fill', (args, kw) => {
      const shape = shapeOf(args[0] ?? kw.dims)
      const v = args[1] ?? kw.value
      const t = tfOf(v)
      return traced('fill', 'create', [], TT.of(new Array(prod(shape)).fill(h.num(v)), shape, t ? t.dtype : v.k === 'float' ? 'float32' : v.k === 'bool' ? 'bool' : 'int32'))
    }),
    range: fnv('range', (args, kw) => {
      const nums = args.map((a) => h.num(a))
      const [a, b, s] = nums.length === 1 ? [0, nums[0], 1] : [nums[0], nums[1], nums[2] ?? 1]
      const n = Math.max(0, Math.ceil((b - a) / s))
      return traced('range', 'create', [], TT.of(Array.from({ length: n }, (_, i) => a + i * s), [n], dt(kw.dtype) ?? (args.some((x) => x.k === 'float') ? 'float32' : 'int32')))
    }),
    linspace: fnv('linspace', (args, kw) => {
      const P = bindArgs(h, 'linspace', 'start stop num name=None axis=0', args, kw, null)
      const startT = T(P.start, 'linspace')
      if (!isFloatT(startT.dtype)) throw tfErr(h, `cannot compute AddV2 as input #1(zero-based) was expected to be a ${TF_KIND[startT.dtype]} tensor but is a double tensor [Op:AddV2]`)
      if (startT.size !== 1) throw h.err('NotImplementedError', 'tf.linspace with non-scalar start and stop is not implemented in this sandbox')
      const lo = startT.values()[0]
      const hi = h.num(P.stop)
      const n = h.toInt(P.num)
      if (n < 1) throw tfErr(h, `Requires num > 0: ${n} [Op:LinSpace]`)
      const delta = n > 1 ? (hi - lo) / (n - 1) : 0
      return traced('linspace', 'create', [], TT.of(Array.from({ length: n }, (_, i) => (i === n - 1 && n > 1 ? hi : lo + i * delta)), [n], startT.dtype))
    }),
    eye, matmul, norm, tensordot, sort: sortFn, argsort: argsortFn, clip_by_value: clipByValue, unique: fnv('unique', (args, kw) => {
      const P = bindArgs(h, 'unique', 'x out_idx=int32 name=None', args, kw, null)
      const t = T(P.x, 'unique')
      if (t.ndim !== 1) throw tfErr(h, 'unique expects a 1D vector. [Op:UniqueV2]')
      const seen = new Map<number, number>()
      const idx = t.values().map((v) => {
        let i = seen.get(v)
        if (i === undefined) seen.set(v, (i = seen.size))
        return i
      })
      return named('Unique', ['y', 'idx'], [tfWrap(TT.of([...seen.keys()], [seen.size], t.dtype)), tfWrap(TT.of(idx, t.shape, dt(P.out_idx) ?? 'int32'))])
    }),
    reshape: fnv('reshape', (args, kw) => {
      const P = bindArgs(h, 'reshape', 'tensor shape name=None', args, kw, null)
      const t = T(P.tensor, 'reshape')
      const shape = shapeOf(P.shape)
      if (shape.filter((d) => d === -1).length > 1) throw tfErr(h, 'only one input size may be -1, not both 0 and 1 [Op:Reshape]')
      const known = shape.reduce((a, d) => (d === -1 ? a : a * d), 1)
      const size = shape.includes(-1) ? (known ? t.size / known : 0) : known
      if ((shape.includes(-1) && !Number.isInteger(size)) || (!shape.includes(-1) && size !== t.size)) throw tfErr(h, `Input to reshape is a tensor with ${t.size} values, but the requested shape has ${known} [Op:Reshape]`)
      return tfWrap(C.reshapeT(E, t, shape, false))
    }),
    transpose: fnv('transpose', (args, kw) => {
      const t = T(args[0] ?? kw.a, 'transpose')
      const perm = kw.perm ?? args[1]
      return tfWrap(C.permuteT(E, t, !isNone(perm) ? shapeOf(perm!) : t.shape.map((_, i) => t.ndim - 1 - i)))
    }),
    cast: fnv('cast', (args, kw) => tfWrap(C.toDtype(E, T(args[0] ?? kw.x, 'cast'), dt(kw.dtype ?? args[1]) ?? 'float32'))),
    shape: fnv('shape', (args, kw) => {
      const v = args[0] ?? kw.input
      const k = kerasOf(v)
      if (k) throw h.err('ValueError', 'A KerasTensor cannot be used as input to a TensorFlow function.')
      const dims = v.k === 'obj' && v.o instanceof TFString ? v.o.shape : T(v, 'shape').shape
      return tfWrap(TT.of(dims, [dims.length], dt(kw.out_type) ?? 'int32'))
    }),
    rank: fnv('rank', (args, kw) => tfWrap(TT.scalar(T(args[0] ?? kw.input, 'rank').ndim, 'int32'))),
    size: fnv('size', (args, kw) => tfWrap(TT.scalar(T(args[0] ?? kw.input, 'size').size, dt(kw.out_type) ?? 'int32'))),
    concat: fnv('concat', (args, kw) => {
      const parts = h.iterate(args[0] ?? kw.values).map((v) => T(v, 'concat'))
      for (const t of parts) if (t.dtype !== parts[0].dtype) throw tfErr(h, `cannot compute ConcatV2 as input #${parts.indexOf(t)}(zero-based) was expected to be a ${TF_KIND[parts[0].dtype]} tensor but is a ${TF_KIND[t.dtype]} tensor [Op:ConcatV2]`)
      return tfWrap(C.catT(E, parts, h.toInt(kw.axis ?? args[1] ?? py.int(0))))
    }),
    stack: fnv('stack', (args, kw) => tfWrap(C.catT(E, h.iterate(args[0] ?? kw.values).map((v) => T(v, 'stack')), h.toInt(kw.axis ?? args[1] ?? py.int(0)), true))),
    unstack: fnv('unstack', (args, kw) => {
      const P = bindArgs(h, 'unstack', 'value num=None axis=0 name=None', args, kw, null)
      const t = T(P.value, 'unstack')
      return py.list(unstackT(E, t, C.normDim(h.toInt(P.axis), t.ndim)).map(tfWrap))
    }),
    one_hot: fnv('one_hot', (args, kw) => {
      const P = bindArgs(h, 'one_hot', 'indices depth on_value=None off_value=None axis=None dtype=None name=None', args, kw, null)
      const t = T(P.indices, 'one_hot')
      const n = h.toInt(P.depth)
      const on = isNone(P.on_value) ? 1 : h.num(P.on_value)
      const off = isNone(P.off_value) ? 0 : h.num(P.off_value)
      const d = dt(P.dtype) ?? (P.on_value.k === 'int' ? 'int32' : 'float32')
      const axis = isNone(P.axis) ? -1 : h.toInt(P.axis)
      const out = TT.of(t.values().flatMap((c) => Array.from({ length: n }, (_, j) => (j === c ? on : off))), [...t.shape, n], d)
      if (axis === -1 || axis === t.ndim) return tfWrap(out)
      const a = axis < 0 ? axis + t.ndim + 1 : axis
      const perm = [...allOf(a), t.ndim, ...allOf(t.ndim - a).map((i) => a + i)]
      return tfWrap(C.permuteT(E, out, perm))
    }),
    expand_dims: expandDims, squeeze, split: splitFn, tile, broadcast_to: broadcastTo, reverse, gather, gather_nd: gatherNd, boolean_mask: boolMask, where: whereFn, pad: padFn,
    identity: fnv('identity', (args, kw) => tfWrap(C.cloneT(E, T(args[0] ?? kw.input, 'identity')))),
    stop_gradient: fnv('stop_gradient', (args, kw) => {
      const t = T(args[0] ?? kw.input, 'stop_gradient')
      return tfWrap(TT.of(t.values(), t.shape, t.dtype))
    }),
    add_n: fnv('add_n', (args, kw) => {
      const parts = h.iterate(args[0] ?? kw.inputs).map((v) => T(v, 'add_n'))
      if (!parts.length) throw h.err('ValueError', 'Inputs must be an iterable of at least one Tensor/IndexedSlices with the same dtype and shape.')
      for (const t of parts) if (t.shape.join() !== parts[0].shape.join()) throw h.err('ValueError', `Shapes must be equal rank, but are ${parts[0].ndim} and ${t.ndim}`)
      return tfWrap(parts.reduce((a, b) => C.binary(E, '+', a, b)))
    }),
    meshgrid: fnv('meshgrid', (args, kw) => {
      const ix = kw.indexing ? h.str(kw.indexing) : 'xy'
      if (ix !== 'xy' && ix !== 'ij') throw h.err('ValueError', `Value of argument \`indexing\` must be one of 'xy' or 'ij'. Received: indexing=${ix}`)
      const ts = args.map((a) => T(a, 'meshgrid'))
      const sizes = ts.map((t) => t.size)
      const shape = ix === 'xy' && sizes.length >= 2 ? [sizes[1], sizes[0], ...sizes.slice(2)] : sizes
      return py.list(ts.map((t, k) => {
        const flat = C.reshapeT(E, t, [t.size], false)
        const axis = ix === 'xy' && k < 2 ? 1 - k : k
        const src = C.crossSrc(shape.map((n, d) => (d === axis ? n : 1)), shape.map((n, d) => (d === axis ? allOf(n) : new Array<number>(n).fill(0))))
        return tfWrap(C.takeT(E, flat, src, shape, 'TakeBackward'))
      }))
    }),
    is_tensor: fnv('is_tensor', (args) => py.bool(!!tfOf(args[0]) || !!kerasOf(args[0]) || (args[0].k === 'obj' && args[0].o instanceof TFString))),
    executing_eagerly: fnv('executing_eagerly', () => py.bool(true)),
    print: tfPrint,
    // a pass-through: the sandbox runs everything eagerly, as tf.config.run_functions_eagerly(True) would
    function: fnv('function', (args, kw) => {
      const f = args[0] ?? kw.func
      return isNone(f) ? fnv('function', (a) => a[0]) : f!
    }),
    device: fnv('device', () => py.obj(new NoScope('device'))),
    name_scope: fnv('name_scope', () => py.obj(new NoScope('name_scope'))),
    TensorShape: { k: 'type', name: 'TensorShape', call: (args) => py.obj(new TensorShape(h.iterate(args[0]).map((v) => (v.k === 'none' ? null : h.toInt(v))))) },
    TensorSpec: { k: 'type', name: 'TensorSpec', call: (args, kw) => {
      const P = bindArgs(h, 'TensorSpec', 'shape dtype=float32 name=None', args, kw, null)
      const d = P.dtype.k === 'obj' && P.dtype.o instanceof TFDtype ? P.dtype.o.d : h.str(P.dtype)
      return py.obj(new TensorSpec(isNone(P.shape) ? null : h.iterate(P.shape).map((v) => (v.k === 'none' ? null : h.toInt(v))), d as TDType | 'string', P.name))
    } },
    Tensor: { k: 'type', name: 'Tensor', call: () => { throw h.err('TypeError', 'use tf.constant(...) to make a tensor') } },
  }
  // every op shows in the step-through: those that did not record a finer view appear as a plain step
  const SILENT = new Set(['print', 'function', 'device', 'name_scope', 'is_tensor', 'executing_eagerly', 'confusion_matrix'])
  const stepped = new WeakMap<object, Value>()
  const withStep = (prefix: string, attrs: Record<string, Value>) => {
    for (const [name, v] of Object.entries(attrs)) {
      if (v.k !== 'fn' || SILENT.has(name)) continue
      const known = stepped.get(v)
      if (known) {
        attrs[name] = known
        continue
      }
      const inner = v.call
      const wrapped: Value = { ...v, call: (args, kw) => {
        S.traced = false
        const res = inner(args, kw)
        if (!S.traced && h.tracing) {
          const ops = [...args, ...Object.values(kw)].flatMap((a, i) => {
            const t = tfOf(a)
            return t && t.size <= 240 ? [{ label: label(a, `x${i}`), snap: traceSnap(t) }] : []
          })
          const t = tfOf(res)
          h.traceCall(`${prefix}${name}`, 'info', ops, t && t.size <= 240 ? traceSnap(t) : null, h.str(res))
        }
        return res
      } }
      stepped.set(v, wrapped)
      stepped.set(wrapped, wrapped)
      attrs[name] = wrapped
    }
  }
  withStep('tf.', tf)
  withStep('tf.math.', mathFns)
  withStep('tf.nn.', nnFns)
  withStep('tf.random.', (random as { attrs: Record<string, Value> }).attrs)
  withStep('tf.linalg.', (linalg as { attrs: Record<string, Value> }).attrs)
  withStep('tf.image.', (image as { attrs: Record<string, Value> }).attrs)
  const tfMod: Value = { k: 'module', name: 'tensorflow', attrs: tf }

  // keras.ops: the same operations under NumPy-style names
  const aliases: Record<string, string> = { add: 'add', subtract: 'subtract', multiply: 'multiply', divide: 'divide', matmul: 'matmul', sum: 'reduce_sum', mean: 'reduce_mean', max: 'reduce_max', min: 'reduce_min', prod: 'reduce_prod', std: 'reduce_std', var: 'reduce_variance', sqrt: 'sqrt', square: 'square', exp: 'exp', log: 'log', abs: 'abs', absolute: 'abs', sin: 'sin', cos: 'cos', tanh: 'tanh', sigmoid: 'sigmoid', reshape: 'reshape', transpose: 'transpose', concatenate: 'concat', stack: 'stack', expand_dims: 'expand_dims', squeeze: 'squeeze', zeros: 'zeros', ones: 'ones', zeros_like: 'zeros_like', ones_like: 'ones_like', cast: 'cast', convert_to_tensor: 'convert_to_tensor', argmax: 'argmax', argmin: 'argmin', where: 'where', power: 'pow', maximum: 'maximum', minimum: 'minimum', sign: 'sign', floor: 'floor', ceil: 'ceil', round: 'round', one_hot: 'one_hot', tile: 'tile', equal: 'equal', not_equal: 'not_equal', greater: 'greater', less: 'less', tensordot: 'tensordot', sort: 'sort', argsort: 'argsort', eye: 'eye', linspace: 'linspace', arange: 'range', stop_gradient: 'stop_gradient', norm: 'norm' }
  const ops: Value = {
    k: 'module', name: 'keras.ops', attrs: {
      ...Object.fromEntries(Object.entries(aliases).map(([k, v]) => [k, tf[v]])),
      relu: nnFns.relu, softmax: nnFns.softmax, leaky_relu: nnFns.leaky_relu, gelu: nnFns.gelu, elu: nnFns.elu,
      clip: clipByValue,
      shape: fnv('shape', (args) => {
        const k = kerasOf(args[0])
        return shapeTuple(k ? k.shape : T(args[0], 'shape').shape)
      }),
    },
  }
  kerasAttrs.ops = ops
  const schedules = (K.optimizers as { attrs: Record<string, Value> }).attrs.schedules
  const sequence = (K.preprocessing as { attrs: Record<string, Value> }).attrs.sequence
  const kerasMods: Record<string, Value> = {
    '': keras, layers: kerasAttrs.layers, models: K.models, optimizers: K.optimizers, 'optimizers.schedules': schedules, losses: kerasAttrs.losses, callbacks: K.callbacks, utils: K.utils, datasets: K.datasets,
    activations: K.activations, metrics: kerasAttrs.metrics, regularizers: K.regularizers, initializers: K.initializers, backend: K.backend, preprocessing: K.preprocessing, 'preprocessing.sequence': sequence, ops,
  }
  const mods: Record<string, Value> = { tensorflow: tfMod, 'tensorflow.math': tfMath, 'tensorflow.nn': tfNN, 'tensorflow.random': random, 'tensorflow.linalg': linalg, 'tensorflow.image': image, 'tensorflow.data': data, 'tensorflow.config': config, 'tensorflow.errors': errors }
  for (const [k, v] of Object.entries(kerasMods)) {
    mods[k ? `keras.${k}` : 'keras'] = v
    mods[k ? `tensorflow.keras.${k}` : 'tensorflow.keras'] = v
  }
  return mods
}

const KERAS_SUBMODULES = ['layers', 'models', 'optimizers', 'optimizers.schedules', 'losses', 'callbacks', 'utils', 'datasets', 'activations', 'metrics', 'regularizers', 'initializers', 'backend', 'preprocessing', 'preprocessing.sequence', 'ops']

export const KERAS: PyLib = {
  modules: ['keras', 'tensorflow', 'tensorflow.keras', ...KERAS_SUBMODULES.flatMap((m) => [`keras.${m}`, `tensorflow.keras.${m}`]), 'tensorflow.math', 'tensorflow.nn', 'tensorflow.random', 'tensorflow.linalg', 'tensorflow.image', 'tensorflow.data', 'tensorflow.config', 'tensorflow.errors'],
  load: build,
}
