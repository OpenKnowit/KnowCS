/**
 * keras and tensorflow for the sandbox (lectures 6 and 8): Sequential and Functional models, the layers the course
 * uses, model.summary() (Keras 3's table, plus the shape-flow view), compile / fit / evaluate / predict with real
 * training on the tensorCore engine, History and EarlyStopping; and a small tensorflow (tf.constant, tf.Variable,
 * GradientTape, tf.math …) with TensorFlow's strict dtypes and printing.
 * Keras is channels-last: images are (batch, height, width, channels).
 */
import { NDArray, PyError, arrayRepr, arrayStr, prod } from './ndarray'
import { PyDict, PyObj, formatSpec, py, repr, toArray } from './minipy'
import type { Host, Kw, PyLib, Value } from './minipy'
import type { FlowRow } from './pyEvents'
import * as C from './tensorCore'
import { Engine, TT, isFloatT } from './tensorCore'
import type { TDType } from './tensorCore'

// ---------------------------------------------------------------- per-run state

interface KerasState {
  E: Engine
  counters: Map<string, number>
  files: Map<string, Value>
  notes: Set<string>
}

const stateOf = (h: Host): KerasState => {
  let s = h.state.get('keras') as KerasState | undefined
  if (!s) {
    s = { E: new Engine(h.rng), counters: new Map(), files: new Map(), notes: new Set() }
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

const noteOnce = (h: Host, key: string, tone: 'info' | 'warn', params: Record<string, string | number> = {}) => {
  const s = stateOf(h)
  if (s.notes.has(key) || !h.tracing) return
  s.notes.add(key)
  h.emit({ type: 'note', tone, key, params })
}

const fnv = (name: string, call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })
const snake = (s: string) => s.replace(/([a-z0-9])([A-Z])/g, '$1_$2').replace(/([A-Z])([A-Z][a-z])/g, '$1_$2').toLowerCase()

type KShape = (number | null)[]
const shapeStr = (s: KShape) => `(${s.map((d) => (d === null ? 'None' : d)).join(', ')}${s.length === 1 ? ',' : ''})`

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

const ndFor = (t: TT): NDArray => {
  const vals = t.dtype === 'float32' ? t.values().map(f32clean) : t.values()
  return NDArray.create(vals, t.shape, t.dtype === 'bool' ? 'bool' : isFloatT(t.dtype) ? 'float64' : 'int64')
}

/** array([...], dtype=float32) */
const npRepr = (t: TT): string => {
  const base = arrayRepr(ndFor(t))
  if (t.dtype === 'float64' || t.dtype === 'int64' || t.dtype === 'bool') return base
  if (!t.ndim) return base
  return `${base.slice(0, -1)}, dtype=${t.dtype})`
}

const scalarStr = (t: TT) => {
  const v = t.values()[0]
  if (t.dtype === 'bool') return v ? 'True' : 'False'
  if (!isFloatT(t.dtype)) return String(v)
  const c = t.dtype === 'float32' ? f32clean(v) : v
  return Number.isInteger(c) ? `${c}.0` : String(c)
}

// ---------------------------------------------------------------- tensorflow tensors

const TF_DTYPES: Record<string, TDType> = { float32: 'float32', float64: 'float64', double: 'float64', int32: 'int32', int64: 'int64', bool: 'bool', uint8: 'uint8' }

class TFDtype extends PyObj {
  readonly cls = 'DType'
  d: TDType
  constructor(d: TDType) {
    super()
    this.d = d
  }
  repr() { return `tf.${this.d}` }
  binop(op: string, other: Value) {
    if (op !== '==' && op !== '!=') return undefined
    const same = other.k === 'obj' && other.o instanceof TFDtype && other.o.d === this.d
    return py.bool((op === '==') === same)
  }
  getAttr(name: string) { return name === 'name' ? py.str(this.d) : name === 'is_floating' ? py.bool(isFloatT(this.d)) : undefined }
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
    const i = h.toInt(idx)
    const n = this.dims.length
    const d = this.dims[i < 0 ? i + n : i]
    if (d === undefined) throw h.err('IndexError', 'list index out of range')
    return d === null ? py.NONE : py.int(d)
  }
  getAttr(name: string) {
    if (name === 'as_list') return fnv(name, () => py.list(this.iter()))
    if (name === 'rank' || name === 'ndims') return py.int(this.dims.length)
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

/** a small TF op for the element-provenance view (the same view as NumPy's) */
const traceTF = (h: Host, api: string, kind: 'elementwise' | 'matmul' | 'reduce' | 'create', ops: { label: string; t: TT }[], out: TT, axis: number | null = null) => {
  if (!h.tracing || out.size > 240 || ops.some((o) => o.t.size > 240)) return
  const snap = (t: TT) => ({ shape: [...t.shape], values: t.dtype === 'float32' ? t.values().map(f32clean) : t.values(), dtype: t.a.dtype })
  h.traceCall(api, kind, ops.map((o) => ({ label: o.label, snap: snap(o.t) })), snap(out), new TFTensor(out).repr(), axis)
}
const tfOf = (v: Value): TT | null => (v.k === 'obj' && v.o instanceof TFTensor ? v.o.t : null)

const OP_NAME: Record<string, string> = { '+': 'AddV2', '-': 'Sub', '*': 'Mul', '/': 'RealDiv', '@': 'MatMul', '**': 'Pow', '<': 'Less', '>': 'Greater', '<=': 'LessEqual', '>=': 'GreaterEqual', '==': 'Equal', '!=': 'NotEqual' }
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
  if (v.k === 'array') return TT.of(v.a.values(), v.a.shape, v.a.dtype === 'float64' ? 'float64' : v.a.dtype === 'bool' ? 'bool' : 'int64')
  if (v.k === 'list' || v.k === 'tuple') return toTF(h, v, like?.dtype ?? null)
  return null
}

/** tf.constant / convert_to_tensor: Python floats → float32, ints → int32, NumPy keeps its dtype */
const toTF = (h: Host, v: Value, dtype: TDType | null): TT => {
  const t = tfOf(v)
  if (t) return dtype && dtype !== t.dtype ? TT.of(t.values(), t.shape, dtype) : t
  if (v.k === 'array') return TT.of(v.a.values(), v.a.shape, dtype ?? (v.a.dtype === 'float64' ? 'float64' : v.a.dtype === 'bool' ? 'bool' : 'int64'))
  let a: NDArray
  try {
    a = toArray(v)
  } catch (e) {
    throw h.err('ValueError', `Can't convert to a Tensor: ${(e as Error).message}`)
  }
  return TT.of(a.values(), a.shape, dtype ?? (a.dtype === 'float64' ? 'float32' : a.dtype === 'bool' ? 'bool' : 'int32'))
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
    const E = stateOf(h).E
    return Array.from({ length: this.len(h) }, (_, i) => tfWrap(C.indexT(E, this.t, [{ kind: 'int', value: i }])))
  }
  truthy(h: Host) {
    if (this.t.size !== 1) throw h.err('ValueError', 'The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()')
    return this.t.values()[0] !== 0
  }
  getItem(idx: Value, h: Host) {
    const items = (idx.k === 'tuple' ? idx.items : [idx]).map((it) => {
      if (it.k === 'int') return { kind: 'int' as const, value: it.v }
      if (it.k === 'slice') {
        const part = (x: Value) => (x.k === 'none' ? null : h.toInt(x))
        return { kind: 'slice' as const, start: part(it.start), stop: part(it.stop), step: part(it.step) }
      }
      if (it.k === 'none') return { kind: 'newaxis' as const }
      if (it.k === 'ellipsis') return { kind: 'ellipsis' as const }
      throw h.err('TypeError', 'Only integers, slices (`:`), ellipsis (`...`), tf.newaxis (`None`) and scalar tf.int32/tf.int64 tensors are valid indices')
    })
    return tfWrap(C.indexT(stateOf(h).E, this.t, items))
  }
  binop(op: string, other: Value, reflected: boolean, h: Host): Value | undefined {
    const o = tfOperand(h, other, this.t)
    if (!o) return undefined
    const [a, b] = reflected ? [o, this.t] : [this.t, o]
    if (a.dtype !== b.dtype && OP_NAME[op]) {
      throw h.err('InvalidArgumentError', `cannot compute ${OP_NAME[op]} as input #1(zero-based) was expected to be a ${TF_KIND[a.dtype]} tensor but is a ${TF_KIND[b.dtype]} tensor [Op:${OP_NAME[op]}]`)
    }
    const E = stateOf(h).E
    const me = h.nameOf(py.obj(this), 'a')
    const ops = reflected ? [{ label: h.nameOf(other, 'a'), t: a }, { label: me, t: b }] : [{ label: me, t: a }, { label: h.nameOf(other, 'b'), t: b }]
    let out: TT
    if (op === '@') {
      out = C.matmul(E, a, b, 'MatMul')
      traceTF(h, 'op:@', 'matmul', ops, out)
      return tfWrap(out)
    }
    if (['<', '<=', '>', '>=', '==', '!='].includes(op)) out = C.compare(a, b, op as C.CmpOpT)
    else if (op === '/' && !isFloatT(a.dtype)) out = C.binary(E, '/', C.toDtype(E, a, 'float64'), C.toDtype(E, b, 'float64'))
    else if (['+', '-', '*', '/', '**', '//', '%'].includes(op)) {
      const r = C.binary(E, op as C.BinOpT, a, b)
      out = r.dtype === a.dtype ? r : C.toDtype(E, r, a.dtype)
    } else return undefined
    traceTF(h, `op:${op}`, 'elementwise', ops, out)
    return tfWrap(out)
  }
  unary(op: '-' | '+' | '~', h: Host): Value {
    return op === '-' ? tfWrap(C.unary(stateOf(h).E, 'neg', this.t)) : py.obj(this)
  }
  getAttr(name: string, h: Host): Value | undefined {
    const t = this.t
    switch (name) {
      case 'shape': return py.obj(new TensorShape(t.shape))
      case 'dtype': return py.obj(new TFDtype(t.dtype))
      case 'ndim': return py.int(t.ndim)
      case 'device': return py.str('/job:localhost/replica:0/task:0/device:CPU:0')
      case 'numpy': return fnv(name, () => (t.ndim ? py.arr(ndFor(t)) : isFloatT(t.dtype) ? py.float(f32clean(t.values()[0])) : t.dtype === 'bool' ? py.bool(t.values()[0] !== 0) : py.int(t.values()[0])))
      case 'T': return tfWrap(C.permuteT(stateOf(h).E, t, t.shape.map((_, i) => t.ndim - 1 - i)))
    }
    return undefined
  }
}

/** tf.Variable: a trainable tensor you update with assign / assign_sub */
class TFVariable extends TFTensor {
  readonly cls = 'ResourceVariable'
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
    const assign = (op: (old: number, v: number) => number) => fnv(name, (args) => {
      const v = tfOperand(h, args[0], this.t)
      if (!v) throw h.err('TypeError', `${name}() needs a tensor or a number`)
      if (v.dtype !== this.t.dtype) throw h.err('InvalidArgumentError', `Cannot update variable with shape [${this.t.shape.join(',')}] using a Tensor of dtype ${v.dtype}`)
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
      for (const v of args[0].k === 'list' || args[0].k === 'tuple' ? h.iterate(args[0]) : [args[0]]) {
        const t = tfOf(v)
        if (t && isFloatT(t.dtype)) {
          t.requiresGrad = true
          this.watched.push(t)
        }
      }
      return py.NONE
    })
    if (name === 'gradient') return fnv(name, (args) => {
      if (this.used && !this.persistent) throw h.err('RuntimeError', 'A non-persistent GradientTape can only be used to compute one set of gradients (or jacobians)')
      this.used = true
      const target = tfOf(args[0])
      if (!target) throw h.err('TypeError', 'gradient(target, sources): target must be a tensor')
      const srcV = args[1]
      const list = srcV.k === 'list' || srcV.k === 'tuple'
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

// ---------------------------------------------------------------- keras: symbolic tensors and layers

/** Input(shape=…) and every layer output in the Functional API: a shape and where it came from */
class KerasTensor extends PyObj {
  readonly cls = 'KerasTensor'
  shape: KShape
  layer: KLayer
  inputs: KerasTensor[]
  name: string
  constructor(shape: KShape, layer: KLayer, inputs: KerasTensor[], name: string) {
    super()
    this.shape = shape
    this.layer = layer
    this.inputs = inputs
    this.name = name
  }
  repr() { return `<KerasTensor shape=${shapeStr(this.shape)}, dtype=float32, sparse=False, ragged=False, name=${this.name}>` }
  getAttr(name: string) {
    if (name === 'shape') return py.tuple(this.shape.map((d) => (d === null ? py.NONE : py.int(d))))
    if (name === 'dtype') return py.str('float32')
    if (name === 'name') return py.str(this.name)
    return undefined
  }
  binop(op: string, _o: Value, _r: boolean, h: Host): Value | undefined {
    throw h.err('ValueError', `A KerasTensor cannot be used with '${op}' in this sandbox: use layers such as layers.Add() to combine them`)
  }
}

type Act = 'relu' | 'sigmoid' | 'softmax' | 'tanh' | 'linear' | 'leaky_relu' | 'elu'

const activate = (E: Engine, x: TT, a: Act): TT => {
  switch (a) {
    case 'relu': return C.unary(E, 'relu', x)
    case 'sigmoid': return C.unary(E, 'sigmoid', x)
    case 'tanh': return C.unary(E, 'tanh', x)
    case 'softmax': return C.softmax(E, x, -1, false)
    case 'leaky_relu': return C.binary(E, 'maximum', x, C.binary(E, '*', x, TT.scalar(0.2, 'float32'), false, true))
    default: return x
  }
}

const actArg = (h: Host, v: Value | undefined): Act => {
  if (!v || v.k === 'none') return 'linear'
  const name = v.k === 'str' ? v.v : v.k === 'fn' ? v.name : v.k === 'obj' ? v.o.repr() : ''
  if (!['relu', 'sigmoid', 'softmax', 'tanh', 'linear', 'leaky_relu'].includes(name)) throw h.err('ValueError', `Could not interpret activation function identifier: ${name}`)
  return name as Act
}

/** Glorot uniform, Keras's default kernel initializer */
const glorot = (h: Host, shape: number[], fanIn: number, fanOut: number): TT => {
  const lim = Math.sqrt(6 / (fanIn + fanOut))
  const n = prod(shape)
  const vals = new Array<number>(n)
  for (let i = 0; i < n; i++) vals[i] = Math.fround((h.rng.next() * 2 - 1) * lim)
  const t = TT.raw(vals, shape, 'float32')
  t.requiresGrad = true
  return t
}
const zerosP = (shape: number[]): TT => {
  const t = TT.of(new Array(prod(shape)).fill(0), shape, 'float32')
  t.requiresGrad = true
  return t
}

/** kernel_regularizer=regularizers.l2(0.002): a penalty added to the loss */
class Regularizer extends PyObj {
  readonly cls: string
  kind: 'l1' | 'l2'
  f: number
  constructor(kind: 'l1' | 'l2', f: number) {
    super()
    this.kind = kind
    this.f = f
    this.cls = kind === 'l1' ? 'L1' : 'L2'
  }
  repr() { return `<keras.src.regularizers.regularizers.${this.cls} object>` }
  penalty(E: Engine, w: TT): TT {
    const r = C.reduceSum(E, this.kind === 'l2' ? C.unary(E, 'square', w) : C.unary(E, 'abs', w), null, false, false)
    return C.binary(E, '*', r, TT.scalar(this.f, 'float32'), false, true)
  }
}

const regArg = (h: Host, v: Value | undefined): Regularizer | null => {
  if (!v || v.k === 'none') return null
  if (v.k === 'obj' && v.o instanceof Regularizer) return v.o
  if (v.k === 'str' && (v.v === 'l1' || v.v === 'l2')) return new Regularizer(v.v, 0.01)
  throw h.err('ValueError', `Could not interpret regularizer identifier: ${repr(v)}`)
}

abstract class KLayer extends PyObj {
  abstract readonly kind: string
  get cls() { return this.kind }
  name: string
  built = false
  inShape: KShape | null = null
  outShape: KShape | null = null
  /** set by input_shape= on the first layer of a Sequential */
  declaredInput: KShape | null = null
  constructor(h: Host, kw: Kw, base?: string) {
    super()
    this.name = kw.name && kw.name.k === 'str' ? kw.name.v : autoName(h, base ?? snake(this.constructor.name.replace(/^K/, '')))
    const is = kw.input_shape ?? (kw.input_dim ? py.tuple([kw.input_dim]) : undefined)
    if (is) this.declaredInput = [null, ...h.iterate(is).map((v) => (v.k === 'none' ? null : h.toInt(v)))]
  }
  weights(): [string, TT][] { return [] }
  params() { return this.weights().reduce((s, [, t]) => s + t.size, 0) }
  reg: Regularizer | null = null
  /** the regularisation loss of this layer's kernel */
  penalty(E: Engine): TT | null {
    const k = this.weights().find(([n]) => n === 'kernel')
    return this.reg && k ? this.reg.penalty(E, k[1]) : null
  }
  /** output shape for an input shape, creating weights on the first call */
  abstract build(h: Host, s: KShape): KShape
  abstract forward(E: Engine, x: TT, training: boolean, h: Host): TT
  /** the constructor call, for the shape-flow table */
  describe(): string { return `${this.kind}()` }
  note(_i: KShape, _o: KShape): string | null { return null }
  ensureBuilt(h: Host, s: KShape): KShape {
    if (!this.built) {
      this.inShape = s
      this.outShape = this.build(h, s)
      this.built = true
    } else if (this.inShape && s.length === this.inShape.length && s.some((d, i) => i > 0 && d !== this.inShape![i] && this.inShape![i] !== null)) {
      throw h.err('ValueError', `Input 0 of layer "${this.name}" is incompatible with the layer: expected shape=${shapeStr(this.inShape)}, found shape=${shapeStr(s)}`)
    }
    return this.outShape!
  }
  repr() { return `<${this.kind} name=${this.name}, built=${this.built ? 'True' : 'False'}>` }
  call(args: Value[], kw: Kw, h: Host): Value {
    const x = args[0]
    if (x && x.k === 'obj' && x.o instanceof KerasTensor) {
      const out = this.ensureBuilt(h, x.o.shape)
      return py.obj(new KerasTensor(out, this, [x.o], autoName(h, 'keras_tensor')))
    }
    // eager: a NumPy array or tensor in, a tensor out
    const t = asFloat(h, x)
    this.ensureBuilt(h, [null, ...t.shape.slice(1)])
    const training = !!(kw.training && h.truthy(kw.training))
    return tfWrap(this.forward(stateOf(h).E, t, training, h))
  }
  getAttr(name: string, h: Host): Value | undefined {
    switch (name) {
      case 'name': return py.str(this.name)
      case 'built': return py.bool(this.built)
      case 'trainable': return py.bool(true)
      case 'weights': case 'trainable_weights': case 'trainable_variables': return py.list(this.weights().map(([n, t]) => py.obj(new TFVariable(t, `${this.name}/${n}`, true))))
      case 'get_weights': return fnv(name, () => py.list(this.weights().map(([, t]) => py.arr(ndFor(t)))))
      case 'set_weights': return fnv(name, (args) => {
        const ws = h.iterate(args[0])
        this.weights().forEach(([n, t], i) => {
          const a = toArray(ws[i])
          if (a.shape.join() !== t.shape.join()) throw h.err('ValueError', `Layer ${this.name} weight shape (${t.shape.join(', ')}) is not compatible with provided weight shape (${a.shape.join(', ')}) for ${n}.`)
          C.writeInto(t, a.values())
        })
        return py.NONE
      })
      case 'count_params': return fnv(name, () => py.int(this.params()))
      case 'input_shape': case 'output_shape': {
        const s = name === 'input_shape' ? this.inShape : this.outShape
        if (!s) throw h.err('AttributeError', `The layer "${this.name}" has never been called and thus has no defined ${name}.`)
        return py.tuple(s.map((d) => (d === null ? py.NONE : py.int(d))))
      }
    }
    const w = this.weights().find(([n]) => n === name)
    if (w) return py.obj(new TFVariable(w[1], `${this.name}/${w[0]}`, true))
    return this.attr(name)
  }
  attr(_name: string): Value | undefined { return undefined }
}

/** a NumPy array / list / tensor as a float32 tensor (Keras casts inputs to float32) */
const asFloat = (h: Host, v: Value | undefined): TT => {
  if (!v) throw h.err('TypeError', 'expected input data')
  const t = tfOf(v)
  if (t) return isFloatT(t.dtype) ? t : TT.of(t.values(), t.shape, 'float32')
  if (v.k === 'obj' && v.o.toArray) {
    const a = v.o.toArray()
    return TT.of(a.values(), a.shape, 'float32')
  }
  const a = toArray(v)
  return TT.of(a.values(), a.shape, 'float32')
}

class InputLayer extends KLayer {
  readonly kind = 'InputLayer'
  shape: KShape
  constructor(h: Host, shape: KShape, kw: Kw) {
    super(h, kw, 'input_layer')
    this.shape = shape
    this.built = true
    this.inShape = shape
    this.outShape = shape
  }
  build(_h: Host, s: KShape) { return s }
  forward(_E: Engine, x: TT) { return x }
}

class Dense extends KLayer {
  readonly kind = 'Dense'
  units: number
  act: Act
  useBias: boolean
  w: TT | null = null
  b: TT | null = null
  constructor(h: Host, units: number, kw: Kw) {
    super(h, kw, 'dense')
    if (!Number.isInteger(units) || units < 1) throw h.err('ValueError', `Received an invalid value for \`units\`, expected a positive integer. Received: units=${units}`)
    this.units = units
    this.act = actArg(h, kw.activation)
    this.useBias = !kw.use_bias || h.truthy(kw.use_bias)
    this.reg = regArg(h, kw.kernel_regularizer)
  }
  build(h: Host, s: KShape) {
    const inF = s[s.length - 1]
    if (inF === null || inF === undefined) throw h.err('ValueError', 'The last dimension of the inputs to a Dense layer should be defined. Found None.')
    if (s.length > 2) noteOnce(h, 'keras_dense_nd', 'warn', { shape: shapeStr(s), units: this.units })
    this.w = glorot(h, [inF, this.units], inF, this.units)
    this.b = this.useBias ? zerosP([this.units]) : null
    return [...s.slice(0, -1), this.units]
  }
  weights(): [string, TT][] { return this.w ? (this.b ? [['kernel', this.w], ['bias', this.b]] : [['kernel', this.w]]) : [] }
  describe() { return `Dense(${this.units}${this.act !== 'linear' ? `, activation='${this.act}'` : ''})` }
  note(i: KShape) { return `(${i[i.length - 1]} + ${this.useBias ? 1 : 0}) × ${this.units} = ${this.params().toLocaleString('en-US')}` }
  forward(E: Engine, x: TT) {
    const inF = this.w!.shape[0]
    if (x.shape[x.ndim - 1] !== inF) throw new PyError('ValueError', `Input 0 of layer "${this.name}" is incompatible with the layer: expected axis -1 of input shape to have value ${inF}, but received input with shape ${shapeStr([null, ...x.shape.slice(1)])}`)
    // Dense acts on the last axis: x @ kernel + bias
    let y = C.matmul(E, x, this.w!)
    if (this.b) y = C.binary(E, '+', y, this.b)
    return activate(E, y, this.act)
  }
  attr(name: string) {
    if (name === 'units') return py.int(this.units)
    if (name === 'activation') return py.str(this.act)
    return undefined
  }
}

const pairArg = (h: Host, v: Value | undefined, d: number): [number, number] => {
  if (!v || v.k === 'none') return [d, d]
  if (v.k === 'tuple' || v.k === 'list') return [h.toInt(v.items[0]), h.toInt(v.items[1])]
  const n = h.toInt(v)
  return [n, n]
}

const padMode = (h: Host, v: Value | undefined): 'valid' | 'same' => {
  const s = v && v.k === 'str' ? v.v.toLowerCase() : 'valid'
  if (s !== 'valid' && s !== 'same') throw h.err('ValueError', `The \`padding\` argument must be a string, one of "valid", "same". Received: padding=${v ? repr(v) : 'None'}`)
  return s
}

/** output size: valid ⌊(n − k) / s⌋ + 1, same ⌈n / s⌉ */
const kOut = (n: number, k: number, s: number, mode: 'valid' | 'same') => (mode === 'same' ? Math.ceil(n / s) : Math.floor((n - k) / s) + 1)
/** the padding TF uses for 'same': total = max((⌈n/s⌉ − 1)·s + k − n, 0), split low / high */
const samePad = (n: number, k: number, s: number): [number, number] => {
  const total = Math.max((Math.ceil(n / s) - 1) * s + k - n, 0)
  return [Math.floor(total / 2), total - Math.floor(total / 2)]
}

/** NHWC → NCHW, run a channels-first kernel, back to NHWC; asymmetric 'same' padding is applied first */
const nhwc = (E: Engine, x: TT, pad: [[number, number], [number, number]], run: (y: TT) => TT): TT => {
  let y = C.permuteT(E, x, [0, 3, 1, 2])
  const [[pt, pb], [pl, pr]] = pad
  if (pt || pb || pl || pr) {
    const [N, Cc, H, W] = y.shape
    const H2 = H + pt + pb
    const W2 = W + pl + pr
    const vals = y.values()
    const idx: number[] = []
    const out = new Array<number>(N * Cc * H2 * W2).fill(0)
    for (let n = 0; n < N * Cc; n++) {
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < W; j++) {
          const o = n * H2 * W2 + (i + pt) * W2 + (j + pl)
          out[o] = vals[n * H * W + i * W + j]
          idx.push(o)
        }
      }
    }
    const src = y
    const padded = TT.of(out, [N, Cc, H2, W2], y.dtype)
    if (E.grad && src.requiresGrad) {
      padded.requiresGrad = true
      padded.fn = { name: 'PadBackward', inputs: [src], backward: (g) => [idx.map((o) => g[o])], freed: false }
    }
    y = padded
  }
  return C.permuteT(E, run(y), [0, 2, 3, 1])
}

class Conv2D extends KLayer {
  readonly kind = 'Conv2D'
  filters: number
  k: [number, number]
  strides: [number, number]
  padding: 'valid' | 'same'
  act: Act
  useBias: boolean
  w: TT | null = null
  wT: TT | null = null
  b: TT | null = null
  constructor(h: Host, filters: number, kernel: Value | undefined, kw: Kw) {
    super(h, kw, 'conv2d')
    this.filters = filters
    this.k = pairArg(h, kernel ?? kw.kernel_size, 3)
    this.strides = pairArg(h, kw.strides, 1)
    this.padding = padMode(h, kw.padding)
    this.act = actArg(h, kw.activation)
    this.useBias = !kw.use_bias || h.truthy(kw.use_bias)
    this.reg = regArg(h, kw.kernel_regularizer)
  }
  build(h: Host, s: KShape) {
    if (s.length !== 4) throw h.err('ValueError', `Input 0 of layer "${this.name}" is incompatible with the layer: expected min_ndim=4, found ndim=${s.length}. Full shape received: ${shapeStr(s)}`)
    const [, H, W, Cin] = s as number[]
    const Ho = kOut(H, this.k[0], this.strides[0], this.padding)
    const Wo = kOut(W, this.k[1], this.strides[1], this.padding)
    if (Ho < 1 || Wo < 1) throw h.err('ValueError', `Computed output size would be negative. Received \`inputs shape=${shapeStr(s)}\`, \`kernel shape=(${this.k[0]}, ${this.k[1]}, ${Cin}, ${this.filters})\`, \`dilation_rate=[1 1]\`.`)
    // Keras keeps the kernel as (kh, kw, in, out); the engine wants (out, in, kh, kw)
    this.w = glorot(h, [this.k[0], this.k[1], Cin, this.filters], this.k[0] * this.k[1] * Cin, this.k[0] * this.k[1] * this.filters)
    this.b = this.useBias ? zerosP([this.filters]) : null
    return [null, Ho, Wo, this.filters]
  }
  weights(): [string, TT][] { return this.w ? (this.b ? [['kernel', this.w], ['bias', this.b]] : [['kernel', this.w]]) : [] }
  describe() {
    return `Conv2D(${this.filters}, (${this.k.join(', ')})${this.strides[0] !== 1 || this.strides[1] !== 1 ? `, strides=(${this.strides.join(', ')})` : ''}${this.padding === 'same' ? ", padding='same'" : ''}${this.act !== 'linear' ? `, activation='${this.act}'` : ''})`
  }
  note(i: KShape, o: KShape) {
    const H = i[1] as number
    const f = this.padding === 'same' ? `⌈${H} / ${this.strides[0]}⌉ = ${o[1]}` : `⌊(${H} − ${this.k[0]}) / ${this.strides[0]}⌋ + 1 = ${o[1]}`
    return `${f} · params (${this.k[0]}·${this.k[1]}·${i[3]} + 1)·${this.filters}`
  }
  forward(E: Engine, x: TT, _t: boolean, h: Host) {
    if (x.ndim !== 4) throw h.err('ValueError', `Input 0 of layer "${this.name}" is incompatible with the layer: expected min_ndim=4, found ndim=${x.ndim}. Full shape received: ${shapeStr([null, ...x.shape.slice(1)])}`)
    const wOIHW = C.permuteT(E, this.w!, [3, 2, 0, 1])
    const pad: [[number, number], [number, number]] = this.padding === 'same' ? [samePad(x.shape[1], this.k[0], this.strides[0]), samePad(x.shape[2], this.k[1], this.strides[1])] : [[0, 0], [0, 0]]
    const y = nhwc(E, x, pad, (z) => C.conv2d(E, z, wOIHW, this.b, { stride: this.strides, padding: [0, 0] }))
    return activate(E, y, this.act)
  }
  attr(name: string) { return name === 'filters' ? py.int(this.filters) : name === 'kernel_size' ? py.tuple(this.k.map(py.int)) : undefined }
}

class Pool2D extends KLayer {
  readonly kind: string
  k: [number, number]
  s: [number, number]
  padding: 'valid' | 'same'
  mode: 'max' | 'avg'
  constructor(h: Host, mode: 'max' | 'avg', args: Value[], kw: Kw) {
    super(h, kw, mode === 'max' ? 'max_pooling2d' : 'average_pooling2d')
    this.mode = mode
    this.kind = mode === 'max' ? 'MaxPooling2D' : 'AveragePooling2D'
    this.k = pairArg(h, kw.pool_size ?? args[0], 2)
    const st = kw.strides ?? args[1]
    this.s = st && st.k !== 'none' ? pairArg(h, st, 1) : this.k
    this.padding = padMode(h, kw.padding ?? args[2])
  }
  build(h: Host, s: KShape) {
    if (s.length !== 4) throw h.err('ValueError', `Input 0 of layer "${this.name}" is incompatible with the layer: expected ndim=4, found ndim=${s.length}. Full shape received: ${shapeStr(s)}`)
    return [null, kOut(s[1] as number, this.k[0], this.s[0], this.padding), kOut(s[2] as number, this.k[1], this.s[1], this.padding), s[3]]
  }
  describe() { return `${this.kind}((${this.k.join(', ')}))` }
  note(i: KShape, o: KShape) { return this.padding === 'same' ? `⌈${i[1]} / ${this.s[0]}⌉ = ${o[1]}` : `⌊(${i[1]} − ${this.k[0]}) / ${this.s[0]}⌋ + 1 = ${o[1]}` }
  forward(E: Engine, x: TT) {
    const pad: [[number, number], [number, number]] = this.padding === 'same' ? [samePad(x.shape[1], this.k[0], this.s[0]), samePad(x.shape[2], this.k[1], this.s[1])] : [[0, 0], [0, 0]]
    return nhwc(E, x, pad, (z) => C.pool2d(E, z, this.k, this.s, [0, 0], this.mode))
  }
}

class Simple extends KLayer {
  readonly kind: string
  shapeFn: (s: KShape, h: Host) => KShape
  fwd: (E: Engine, x: TT, training: boolean) => TT
  desc: string
  constructor(h: Host, kind: string, base: string, kw: Kw, shapeFn: (s: KShape, h: Host) => KShape, fwd: (E: Engine, x: TT, training: boolean) => TT, desc: string) {
    super(h, kw, base)
    this.kind = kind
    this.shapeFn = shapeFn
    this.fwd = fwd
    this.desc = desc
  }
  build(h: Host, s: KShape) { return this.shapeFn(s, h) }
  describe() { return this.desc }
  note(i: KShape, o: KShape) { return this.kind === 'Flatten' ? `${i.slice(1).join(' × ')} = ${o[1]}` : null }
  forward(E: Engine, x: TT, training: boolean) { return this.fwd(E, x, training) }
}

const flattenShape = (s: KShape, h: Host): KShape => {
  const rest = s.slice(1)
  if (rest.some((d) => d === null)) throw h.err('ValueError', 'The last dimensions of the inputs to Flatten should be defined.')
  return [null, prod(rest as number[])]
}

// ---------------------------------------------------------------- losses, metrics, optimisers

type LossName = 'categorical_crossentropy' | 'sparse_categorical_crossentropy' | 'binary_crossentropy' | 'mean_squared_error' | 'mean_absolute_error'

interface LossSpec { name: LossName; fromLogits: boolean }

const LOSS_ALIASES: Record<string, LossName> = {
  categorical_crossentropy: 'categorical_crossentropy', CategoricalCrossentropy: 'categorical_crossentropy',
  sparse_categorical_crossentropy: 'sparse_categorical_crossentropy', SparseCategoricalCrossentropy: 'sparse_categorical_crossentropy',
  binary_crossentropy: 'binary_crossentropy', BinaryCrossentropy: 'binary_crossentropy',
  mse: 'mean_squared_error', mean_squared_error: 'mean_squared_error', MeanSquaredError: 'mean_squared_error',
  mae: 'mean_absolute_error', mean_absolute_error: 'mean_absolute_error', MeanAbsoluteError: 'mean_absolute_error',
}

class LossObj extends PyObj {
  readonly cls: string
  spec: LossSpec
  constructor(cls: string, spec: LossSpec) {
    super()
    this.cls = cls
    this.spec = spec
  }
  repr() { return `<LossFunctionWrapper(<function ${this.spec.name}>, kwargs={'from_logits': ${this.spec.fromLogits ? 'True' : 'False'}})>` }
  call(args: Value[], _kw: Kw, h: Host): Value {
    const E = stateOf(h).E
    return tfWrap(lossValue(E, h, this.spec, asFloat(h, args[0]), asFloat(h, args[1]), false))
  }
}

const lossSpec = (h: Host, v: Value | undefined): LossSpec => {
  if (!v || v.k === 'none') throw h.err('ValueError', 'No loss to compute. Provide a `loss` argument in `compile()`.')
  if (v.k === 'obj' && v.o instanceof LossObj) return v.o.spec
  const name = v.k === 'str' ? v.v : v.k === 'fn' ? v.name : ''
  const n = LOSS_ALIASES[name]
  if (!n) throw h.err('ValueError', `Could not interpret loss identifier: ${name || repr(v)}`)
  return { name: n, fromLogits: false }
}

const shapeMismatch = (y: TT, p: TT) => new PyError('ValueError', `Arguments \`target\` and \`output\` must have the same shape. Received: target.shape=${shapeStr([null, ...y.shape.slice(1)])}, output.shape=${shapeStr([null, ...p.shape.slice(1)])}`)

/** the loss per batch (mean over samples), as Keras computes it on probabilities unless from_logits */
function lossValue(E: Engine, h: Host, spec: LossSpec, yTrue: TT, yPred: TT, train: boolean): TT {
  const eps = 1e-7
  const clipLog = (p: TT) => C.unary(E, 'log', C.clamp(E, p, eps, 1 - eps))
  switch (spec.name) {
    case 'sparse_categorical_crossentropy': {
      const C2 = yPred.shape[yPred.ndim - 1]
      const labels = yTrue.values().map(Math.round)
      if (labels.length !== yPred.size / C2) throw new PyError('ValueError', `Arguments \`target\` and \`output\` must have the same rank (ndim) or the target must have one dimension less. Received: target.shape=${shapeStr([null, ...yTrue.shape.slice(1)])}, output.shape=${shapeStr([null, ...yPred.shape.slice(1)])}`)
      for (const l of labels) if (l < 0 || l >= C2) throw new PyError('InvalidArgumentError', `Received a label value of ${l} which is outside the valid range of [0, ${C2}). (One-hot targets need loss='categorical_crossentropy'.)`)
      const target = TT.of(labels, [labels.length], 'int64')
      const logp = spec.fromLogits ? C.softmax(E, yPred, -1, true) : clipLog(normalise(E, yPred))
      return C.nllLoss(E, C.reshapeT(E, logp, [labels.length, C2], false), target)
    }
    case 'categorical_crossentropy': {
      if (yTrue.shape.join() !== yPred.shape.join()) {
        if (train) noteOnce(h, 'keras_sparse_hint', 'info')
        throw shapeMismatch(yTrue, yPred)
      }
      const logp = spec.fromLogits ? C.softmax(E, yPred, -1, true) : clipLog(normalise(E, yPred))
      const per = C.reduceSum(E, C.binary(E, '*', C.unary(E, 'neg', yTrue), logp), [-1], false, false)
      return C.reduceSum(E, per, null, false, true)
    }
    case 'binary_crossentropy': {
      const y = yTrue.ndim === yPred.ndim - 1 ? C.unsqueezeT(E, yTrue, -1) : yTrue
      if (y.shape.join() !== yPred.shape.join()) throw shapeMismatch(yTrue, yPred)
      if (spec.fromLogits) return C.bce(E, yPred, y, true)
      const p = C.clamp(E, yPred, eps, 1 - eps)
      const one = TT.scalar(1, 'float32')
      const t1 = C.binary(E, '*', y, C.unary(E, 'log', p))
      const t2 = C.binary(E, '*', C.binary(E, '-', one, y, true, false), C.unary(E, 'log', C.binary(E, '-', one, p, true, false)))
      return C.unary(E, 'neg', C.reduceSum(E, C.binary(E, '+', t1, t2), null, false, true))
    }
    case 'mean_squared_error': case 'mean_absolute_error': {
      const y = yTrue.ndim === yPred.ndim - 1 ? C.unsqueezeT(E, yTrue, -1) : yTrue
      const d = C.binary(E, '-', yPred, y)
      return C.reduceSum(E, spec.name === 'mean_squared_error' ? C.unary(E, 'square', d) : C.unary(E, 'abs', d), null, false, true)
    }
  }
}

/** Keras rescales "probabilities" that do not sum to 1 before taking the log */
const normalise = (E: Engine, p: TT) => C.binary(E, '/', p, C.reduceSum(E, p, [-1], true, false))

/** fraction correct, the way Keras's 'accuracy' picks it for each loss */
const accuracy = (spec: LossSpec, yTrue: TT, yPred: TT): number => {
  const pv = yPred.values()
  const tv = yTrue.values()
  const C2 = yPred.shape[yPred.ndim - 1]
  const n = yPred.size / C2
  let ok = 0
  for (let i = 0; i < n; i++) {
    if (spec.name === 'binary_crossentropy' || C2 === 1) {
      const p = spec.fromLogits ? 1 / (1 + Math.exp(-pv[i])) : pv[i]
      ok += (p > 0.5 ? 1 : 0) === Math.round(tv[i]) ? 1 : 0
    } else {
      const row = pv.slice(i * C2, (i + 1) * C2)
      const arg = row.indexOf(Math.max(...row))
      const label = spec.name === 'categorical_crossentropy' ? tv.slice(i * C2, (i + 1) * C2).indexOf(Math.max(...tv.slice(i * C2, (i + 1) * C2))) : Math.round(tv[i])
      ok += arg === label ? 1 : 0
    }
  }
  return ok / n
}

class KOptimizer extends PyObj {
  readonly cls: string
  kind: 'adam' | 'sgd' | 'rmsprop'
  lr: number
  momentum: number
  t = 0
  m = new Map<TT, number[]>()
  v = new Map<TT, number[]>()
  constructor(kind: 'adam' | 'sgd' | 'rmsprop', lr: number, momentum = 0) {
    super()
    this.kind = kind
    this.cls = kind === 'adam' ? 'Adam' : kind === 'sgd' ? 'SGD' : 'RMSprop'
    this.lr = lr
    this.momentum = momentum
  }
  repr() { return `<keras.src.optimizers.${this.kind}.${this.cls} object>` }
  getAttr(name: string) {
    if (name === 'learning_rate' || name === 'lr') return py.float(f32clean(Math.fround(this.lr)))
    return undefined
  }
  step(params: TT[]) {
    this.t++
    for (const p of params) {
      if (!p.grad) continue
      const g = p.grad.values()
      const w = p.values()
      let next: number[]
      if (this.kind === 'sgd') {
        if (this.momentum) {
          const m = (this.m.get(p) ?? g.map(() => 0)).map((x, i) => this.momentum * x - this.lr * g[i])
          this.m.set(p, m)
          next = w.map((x, i) => x + m[i])
        } else next = w.map((x, i) => x - this.lr * g[i])
      } else if (this.kind === 'rmsprop') {
        const v = (this.v.get(p) ?? g.map(() => 0)).map((x, i) => 0.9 * x + 0.1 * g[i] * g[i])
        this.v.set(p, v)
        next = w.map((x, i) => x - (this.lr * g[i]) / (Math.sqrt(v[i]) + 1e-7))
      } else {
        const m = (this.m.get(p) ?? g.map(() => 0)).map((x, i) => 0.9 * x + 0.1 * g[i])
        const v = (this.v.get(p) ?? g.map(() => 0)).map((x, i) => 0.999 * x + 0.001 * g[i] * g[i])
        this.m.set(p, m)
        this.v.set(p, v)
        const a = (this.lr * Math.sqrt(1 - 0.999 ** this.t)) / (1 - 0.9 ** this.t)
        next = w.map((x, i) => x - (a * m[i]) / (Math.sqrt(v[i]) + 1e-7))
      }
      C.writeInto(p, next)
    }
  }
}

const optimizerArg = (h: Host, v: Value | undefined): KOptimizer => {
  if (v && v.k === 'obj' && v.o instanceof KOptimizer) return v.o
  const name = v && v.k === 'str' ? v.v.toLowerCase() : 'rmsprop'
  if (name === 'adam') return new KOptimizer('adam', 0.001)
  if (name === 'sgd') return new KOptimizer('sgd', 0.01)
  if (name === 'rmsprop') return new KOptimizer('rmsprop', 0.001)
  throw h.err('ValueError', `Could not interpret optimizer identifier: ${v ? repr(v) : 'None'}`)
}

class Callback extends PyObj {
  readonly cls: string
  monitor: string
  patience: number
  restore: boolean
  constructor(cls: string, monitor = 'val_loss', patience = 0, restore = false) {
    super()
    this.cls = cls
    this.monitor = monitor
    this.patience = patience
    this.restore = restore
  }
  repr() { return `<keras.src.callbacks.${snake(this.cls)}.${this.cls} object>` }
}

class History extends PyObj {
  readonly cls = 'History'
  history: Record<string, number[]>
  epochs: number
  constructor(history: Record<string, number[]>, epochs: number) {
    super()
    this.history = history
    this.epochs = epochs
  }
  repr() { return '<keras.src.callbacks.history.History object>' }
  getAttr(name: string) {
    if (name === 'history') return { k: 'dict' as const, d: PyDict.from(Object.entries(this.history).map(([k, v]) => [py.str(k), py.list(v.map((x) => py.float(f32clean(Math.fround(x)))))])) }
    if (name === 'epoch') return py.list(Array.from({ length: this.epochs }, (_, i) => py.int(i)))
    return undefined
  }
}

/** Keras's log numbers: 4 decimals, or scientific below 1e-3 */
const kfmt = (v: number) => (Math.abs(v) > 1e-3 ? v.toFixed(4) : v.toExponential(4).replace(/e([+-])(\d)$/, 'e$10$2'))

// ---------------------------------------------------------------- models

class KModel extends PyObj {
  readonly cls: string
  name: string
  layers: KLayer[] = []
  /** Functional: the input tensors and the output tensor */
  inputs: KerasTensor[] | null = null
  output: KerasTensor | null = null
  inputShape: KShape | null = null
  compiled: { loss: LossSpec; opt: KOptimizer; metrics: string[] } | null = null
  constructor(h: Host, functional: boolean, name: string | null) {
    super()
    this.cls = functional ? 'Functional' : 'Sequential'
    this.name = name ?? autoName(h, functional ? 'functional' : 'sequential')
  }
  repr() { return `<${this.cls} name=${this.name}, built=${this.inputShape ? 'True' : 'False'}>` }

  add(h: Host, l: Value) {
    if (l.k === 'obj' && l.o instanceof InputLayer) {
      this.inputShape = l.o.shape
      return
    }
    if (l.k !== 'obj' || !(l.o instanceof KLayer)) throw h.err('TypeError', `The added layer must be an instance of class Layer. Received: layer=${repr(l)} of type ${l.k === 'obj' ? l.o.cls : l.k}.`)
    if (!this.inputShape && !this.layers.length && l.o.declaredInput) this.inputShape = l.o.declaredInput
    this.layers.push(l.o)
    if (this.inputShape) this.build(h, this.inputShape)
  }

  /** propagate shapes through the stack, creating weights */
  build(h: Host, s: KShape) {
    this.inputShape = s
    if (this.output) return
    let cur = s
    for (const l of this.layers) cur = l.ensureBuilt(h, cur)
  }

  /** Functional graph in topological order */
  nodes(): KerasTensor[] {
    const order: KerasTensor[] = []
    const seen = new Set<KerasTensor>()
    const visit = (t: KerasTensor) => {
      if (seen.has(t)) return
      seen.add(t)
      t.inputs.forEach(visit)
      order.push(t)
    }
    if (this.output) visit(this.output)
    return order
  }

  forward(E: Engine, x: TT, training: boolean, h: Host): TT {
    if (this.output) {
      const vals = new Map<KerasTensor, TT>()
      for (const n of this.nodes()) {
        if (n.layer instanceof InputLayer) vals.set(n, x)
        else if (n.layer instanceof Merge) vals.set(n, n.layer.merge(E, n.inputs.map((i) => vals.get(i)!)))
        else vals.set(n, n.layer.forward(E, vals.get(n.inputs[0])!, training, h))
      }
      return vals.get(this.output)!
    }
    let y = x
    for (const l of this.layers) y = l.forward(E, y, training, h)
    return y
  }

  ensureBuilt(h: Host, x: TT) {
    if (!this.inputShape) this.build(h, [null, ...x.shape.slice(1)])
    const want = this.inputShape!
    if (x.ndim !== want.length || want.some((d, i) => i > 0 && d !== null && d !== x.shape[i])) {
      throw h.err('ValueError', `Input 0 of layer "${this.layers[0]?.name ?? this.name}" is incompatible with the layer: expected shape=${shapeStr(want)}, found shape=${shapeStr([null, ...x.shape.slice(1)])}`)
    }
  }

  params(): TT[] {
    return [...new Set(this.layers.flatMap((l) => l.weights().map(([, t]) => t)))]
  }

  /** model.summary() rows (and the InputLayer for Functional models) */
  rows(): { name: string; shape: KShape; params: number; layer: KLayer }[] {
    const ls = this.output ? this.nodes().map((n) => n.layer) : this.layers
    return [...new Set(ls)].map((l) => ({ name: `${l.name} (${l.kind})`, shape: l.outShape ?? [], params: l.params(), layer: l }))
  }

  summary(h: Host) {
    const rows = this.rows()
    const total = rows.reduce((s, r) => s + r.params, 0)
    const W = [33, 24, 15]
    const cell = (s: string, w: number, right = false) => ` ${right ? s.padStart(w - 2) : s.padEnd(w - 2)} `
    const line = (l: string, m: string, r: string, fill: string) => l + W.map((w) => fill.repeat(w)).join(m) + r
    const out = [`Model: "${this.name}"`, line('┏', '┳', '┓', '━'), `┃${cell('Layer (type)', W[0])}┃${cell('Output Shape', W[1])}┃${cell('Param #', W[2], true)}┃`, line('┡', '╇', '┩', '━')]
    rows.forEach((r, i) => {
      out.push(`│${cell(r.name, W[0])}│${cell(r.shape.length ? shapeStr(r.shape) : '?', W[1])}│${cell(r.params.toLocaleString('en-US'), W[2], true)}│`)
      out.push(i === rows.length - 1 ? line('└', '┴', '┘', '─') : line('├', '┼', '┤', '─'))
    })
    if (!rows.length) out.push(line('└', '┴', '┘', '─'))
    const bytes = (n: number) => {
      const b = n * 4
      return b >= 1024 * 1024 ? `${(b / 1024 / 1024).toFixed(2)} MB` : b >= 1024 ? `${(b / 1024).toFixed(2)} KB` : `${b.toFixed(2)} B`
    }
    out.push(` Total params: ${total.toLocaleString('en-US')} (${bytes(total)})`, ` Trainable params: ${total.toLocaleString('en-US')} (${bytes(total)})`, ' Non-trainable params: 0 (0.00 B)')
    h.print(out.join('\n') + '\n')
    if (h.tracing && rows.length) {
      const flow: FlowRow[] = rows.map((r) => ({
        name: r.name, layer: r.layer.describe(), input: r.layer.inShape ?? [], output: r.shape, params: r.params,
        note: r.layer.inShape && r.layer.outShape ? r.layer.note(r.layer.inShape, r.layer.outShape) : null,
      }))
      h.emit({ type: 'flow', framework: 'keras', model: this.name, rows: flow, total, trainable: total, title: 'model.summary()' })
    }
  }

  fit(h: Host, args: Value[], kw: Kw): Value {
    if (!this.compiled) throw h.err('RuntimeError', 'You must call `compile()` before using the model.')
    const E = stateOf(h).E
    let X = asFloat(h, args[0] ?? kw.x)
    let Y = asFloat(h, args[1] ?? kw.y)
    this.ensureBuilt(h, X)
    const epochs = kw.epochs ?? args[3] ? h.toInt((kw.epochs ?? args[3])!) : 1
    const bs = kw.batch_size ?? args[2] ? h.toInt((kw.batch_size ?? args[2])!) : 32
    const verbose = kw.verbose ? (kw.verbose.k === 'str' ? 1 : h.toInt(kw.verbose)) : 1
    const shuffle = !kw.shuffle || h.truthy(kw.shuffle)
    if (X.shape[0] !== Y.shape[0]) throw h.err('ValueError', `Data cardinality is ambiguous. Make sure all arrays contain the same number of samples.'x' sizes: ${X.shape[0]}\n'y' sizes: ${Y.shape[0]}`)
    let Xv: TT | null = null
    let Yv: TT | null = null
    if (kw.validation_data && kw.validation_data.k !== 'none') {
      const [vx, vy] = h.iterate(kw.validation_data)
      Xv = asFloat(h, vx)
      Yv = asFloat(h, vy)
    } else if (kw.validation_split && h.num(kw.validation_split) > 0) {
      // Keras holds out the LAST fraction of the data, before shuffling
      const n = X.shape[0]
      const cut = Math.floor(n * (1 - h.num(kw.validation_split)))
      const sl = (t: TT, a: number, b: number) => C.indexT({ grad: false } as Engine, t, [{ kind: 'slice', start: a, stop: b, step: 1 }])
      Xv = sl(X, cut, n)
      Yv = sl(Y, cut, n)
      X = sl(X, 0, cut)
      Y = sl(Y, 0, cut)
    }
    const stoppers = kw.callbacks ? h.iterate(kw.callbacks).flatMap((c) => (c.k === 'obj' && c.o instanceof Callback ? [c.o] : [])) : []
    const early = stoppers.find((c) => c.cls === 'EarlyStopping')
    const { loss: spec, opt, metrics } = this.compiled
    const wantAcc = metrics.includes('accuracy') || metrics.includes('acc')
    const params = this.params()
    const n = X.shape[0]
    const steps = Math.ceil(n / bs)
    const hist: Record<string, number[]> = {}
    const push = (k: string, v: number) => (hist[k] ??= []).push(v)
    let best = Infinity
    let wait = 0
    let bestWeights: number[][] | null = null
    let ran = 0
    for (let ep = 0; ep < epochs; ep++) {
      const order = Array.from({ length: n }, (_, i) => i)
      if (shuffle) for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(h.rng.next() * (i + 1))
        ;[order[i], order[j]] = [order[j], order[i]]
      }
      let lossSum = 0
      let correct = 0
      for (let s = 0; s < n; s += bs) {
        const idx = order.slice(s, s + bs)
        const pick = (t: TT) => C.indexT({ grad: false } as Engine, t, [{ kind: 'array', arr: NDArray.create(idx, [idx.length], 'int64') }])
        const xb = pick(X)
        const yb = pick(Y)
        for (const p of params) p.grad = null
        E.grad = true
        const out = this.forward(E, xb, true, h)
        const loss = this.withPenalty(E, lossValue(E, h, spec, yb, out, true))
        C.backward(E, loss, null, false)
        opt.step(params)
        lossSum += loss.values()[0] * idx.length
        if (wantAcc) correct += accuracy(spec, yb, out) * idx.length
      }
      ran++
      push('loss', lossSum / n)
      if (wantAcc) push('accuracy', correct / n)
      if (Xv && Yv) {
        const [vl, va] = this.score(h, Xv, Yv)
        if (wantAcc) push('val_accuracy', va)
        push('val_loss', vl)
      }
      if (verbose) {
        const order2 = [...(wantAcc ? ['accuracy'] : []), 'loss', ...(Xv ? [...(wantAcc ? ['val_accuracy'] : []), 'val_loss'] : [])]
        const logs = order2.map((k) => `${k}: ${kfmt(hist[k][hist[k].length - 1])}`).join(' - ')
        h.print(`Epoch ${ep + 1}/${epochs}\n${steps}/${steps} ${verbose === 1 ? '━━━━━━━━━━━━━━━━━━━━ ' : ''}- ${logs}\n`)
      }
      if (early) {
        const key = early.monitor
        const cur = hist[key]?.[hist[key].length - 1]
        if (cur !== undefined) {
          if (cur < best) {
            best = cur
            wait = 0
            if (early.restore) bestWeights = params.map((p) => p.values())
          } else if (++wait > early.patience) {
            if (early.restore && bestWeights) params.forEach((p, i) => C.writeInto(p, bestWeights![i]))
            if (verbose) h.print(`Epoch ${ep + 1}: early stopping${early.restore ? '\nRestoring model weights from the end of the best epoch: ' + (ep + 1 - wait) + '.' : ''}\n`)
            break
          }
        }
      }
    }
    E.grad = true
    // overfitting: validation loss climbing while training loss keeps falling
    const vl = hist.val_loss
    if (vl && vl.length >= 4) {
      const lo = vl.indexOf(Math.min(...vl))
      if (lo < vl.length - 2 && hist.loss[hist.loss.length - 1] < hist.loss[lo]) noteOnce(h, 'keras_overfit', 'info', { epoch: lo + 1 })
    }
    if (h.tracing) h.emit({ type: 'history', metrics: hist, title: `fit · ${ran} epochs` })
    return py.obj(new History(hist, ran))
  }

  /** Keras adds the layers' regularisation losses to the loss it trains on and reports */
  withPenalty(E: Engine, loss: TT): TT {
    return this.layers.reduce((acc, l) => {
      const p = l.penalty(E)
      return p ? C.binary(E, '+', acc, p) : acc
    }, loss)
  }

  /** [loss, accuracy] on data, no training */
  score(h: Host, X: TT, Y: TT): [number, number] {
    const E = stateOf(h).E
    const was = E.grad
    E.grad = false
    try {
      const out = this.forward(E, X, false, h)
      const l = this.withPenalty(E, lossValue(E, h, this.compiled!.loss, Y, out, false)).values()[0]
      return [l, accuracy(this.compiled!.loss, Y, out)]
    } finally {
      E.grad = was
    }
  }

  getAttr(name: string, h: Host): Value | undefined {
    const fn = (call: (args: Value[], kw: Kw) => Value): Value => fnv(name, call)
    switch (name) {
      case 'name': return py.str(this.name)
      case 'layers': return py.list((this.output ? [...new Set(this.nodes().map((n) => n.layer))].filter((l) => !(l instanceof InputLayer)) : this.layers).map((l) => py.obj(l)))
      case 'add': return fn((args) => { this.add(h, args[0]); return py.NONE })
      case 'pop': return fn(() => { const l = this.layers.pop(); return l ? py.obj(l) : py.NONE })
      case 'summary': return fn(() => {
        if (!this.inputShape && !this.output) throw h.err('ValueError', 'This model has not yet been built. Build the model first by calling `build()` or by calling the model on a batch of data.')
        this.summary(h)
        return py.NONE
      })
      case 'build': return fn((args) => { this.build(h, h.iterate(args[0]).map((v) => (v.k === 'none' ? null : h.toInt(v)))); return py.NONE })
      case 'compile': return fn((_args, kw) => {
        const metrics = kw.metrics ? h.iterate(kw.metrics).map((m) => (m.k === 'str' ? m.v : m.k === 'fn' ? m.name : 'metric')) : []
        this.compiled = { loss: lossSpec(h, kw.loss ?? _args[1]), opt: optimizerArg(h, kw.optimizer ?? _args[0]), metrics }
        return py.NONE
      })
      case 'fit': return fn((args, kw) => this.fit(h, args, kw))
      case 'evaluate': return fn((args, kw) => {
        if (!this.compiled) throw h.err('RuntimeError', 'You must call `compile()` before using the model.')
        const X = asFloat(h, args[0] ?? kw.x)
        const Y = asFloat(h, args[1] ?? kw.y)
        this.ensureBuilt(h, X)
        const [l, a] = this.score(h, X, Y)
        const wantAcc = this.compiled.metrics.includes('accuracy') || this.compiled.metrics.includes('acc')
        const verbose = kw.verbose ? h.toInt(kw.verbose) : 1
        const steps = Math.ceil(X.shape[0] / 32)
        if (verbose) h.print(`${steps}/${steps} ━━━━━━━━━━━━━━━━━━━━ - ${wantAcc ? `accuracy: ${kfmt(a)} - ` : ''}loss: ${kfmt(l)}\n`)
        const f = (x: number) => py.float(f32clean(Math.fround(x)))
        return wantAcc ? py.list([f(l), f(a)]) : f(l)
      })
      case 'predict': case '__call__': return fn((args, kw) => {
        const X = asFloat(h, args[0] ?? kw.x)
        this.ensureBuilt(h, X)
        const E = stateOf(h).E
        const was = E.grad
        E.grad = false
        try {
          const out = this.forward(E, X, false, h)
          const verbose = kw.verbose ? h.toInt(kw.verbose) : 1
          if (verbose && name === 'predict') {
            const steps = Math.ceil(X.shape[0] / 32)
            h.print(`${steps}/${steps} ━━━━━━━━━━━━━━━━━━━━\n`)
          }
          return py.arr(ndFor(out))
        } finally {
          E.grad = was
        }
      })
      case 'count_params': return fn(() => py.int(this.params().reduce((s, t) => s + t.size, 0)))
      case 'get_weights': return fn(() => py.list(this.layers.flatMap((l) => l.weights().map(([, t]) => py.arr(ndFor(t))))))
      case 'get_layer': return fn((args, kw) => {
        const want = kw.name ?? args[0]
        const l = this.layers.find((x) => (want.k === 'str' ? x.name === want.v : false)) ?? (want.k === 'int' ? this.layers[want.v] : undefined)
        if (!l) throw h.err('ValueError', `No such layer: ${h.str(want)}. Existing layers are: [${this.layers.map((x) => `'${x.name}'`).join(', ')}].`)
        return py.obj(l)
      })
      case 'input_shape': return this.inputShape ? py.tuple(this.inputShape.map((d) => (d === null ? py.NONE : py.int(d)))) : py.NONE
      case 'output_shape': {
        const last = this.output?.shape ?? this.layers[this.layers.length - 1]?.outShape
        return last ? py.tuple(last.map((d) => (d === null ? py.NONE : py.int(d)))) : py.NONE
      }
      case 'save': return fn((args) => {
        stateOf(h).files.set(h.str(args[0]), py.obj(this))
        h.print(`(sandbox) model kept in memory as ${h.str(args[0])}: there is no disk here\n`)
        return py.NONE
      })
      case 'trainable_variables': case 'trainable_weights': case 'weights': return py.list(this.layers.flatMap((l) => l.weights().map(([n, t]) => py.obj(new TFVariable(t, `${l.name}/${n}`, true)))))
    }
    return undefined
  }

  call(args: Value[], kw: Kw, h: Host): Value {
    const x = args[0]
    if (x && x.k === 'obj' && x.o instanceof KerasTensor) throw h.err('NotImplementedError', 'nesting models inside the Functional API is not supported in this sandbox')
    const X = asFloat(h, x)
    this.ensureBuilt(h, X)
    const training = !!(kw.training && h.truthy(kw.training))
    return tfWrap(this.forward(stateOf(h).E, X, training, h))
  }
}

/** Concatenate / Add: layers that take a list of tensors */
class Merge extends KLayer {
  readonly kind: string
  axis: number
  constructor(h: Host, kind: string, kw: Kw) {
    super(h, kw, snake(kind))
    this.kind = kind
    this.axis = kw.axis ? h.toInt(kw.axis) : -1
  }
  build(_h: Host, s: KShape) { return s }
  forward(_E: Engine, x: TT) { return x }
  merge(E: Engine, xs: TT[]): TT {
    if (this.kind === 'Add') return xs.reduce((a, b) => C.binary(E, '+', a, b))
    return C.catT(E, xs, this.axis)
  }
  call(args: Value[], _kw: Kw, h: Host): Value {
    const items = h.iterate(args[0])
    const kts = items.flatMap((v) => (v.k === 'obj' && v.o instanceof KerasTensor ? [v.o] : []))
    if (kts.length !== items.length) {
      return tfWrap(this.merge(stateOf(h).E, items.map((v) => asFloat(h, v))))
    }
    const shapes = kts.map((k) => k.shape)
    let out: KShape
    if (this.kind === 'Add') {
      if (shapes.some((s) => s.join() !== shapes[0].join())) throw h.err('ValueError', `Inputs have incompatible shapes. Received shapes ${shapes.map(shapeStr).join(' and ')}`)
      out = shapes[0]
    } else {
      const ax = this.axis < 0 ? this.axis + shapes[0].length : this.axis
      out = shapes[0].map((d, i) => (i === ax ? shapes.reduce((s, x) => s + (x[i] ?? 0), 0) : d))
    }
    this.inShape = shapes[0]
    this.outShape = out
    this.built = true
    return py.obj(new KerasTensor(out, this, kts, autoName(h, 'keras_tensor')))
  }
}

// ---------------------------------------------------------------- the modules

const build = (h: Host): Record<string, Value> => {
  const S = stateOf(h)
  const E = S.E
  const layerType = (name: string, make: (args: Value[], kw: Kw) => KLayer | Merge): Value => ({ k: 'type', name, call: (args, kw) => py.obj(make(args, kw)) })
  const intArg = (v: Value | undefined, what: string) => {
    if (!v) throw h.err('TypeError', `missing required argument '${what}'`)
    return h.toInt(v)
  }
  const Input = fnv('Input', (args, kw) => {
    const sv = kw.shape ?? args[0]
    if (!sv) throw h.err('ValueError', 'Input() needs a shape, e.g. Input(shape=(784,))')
    const shape: KShape = [null, ...h.iterate(sv).map((v) => (v.k === 'none' ? null : h.toInt(v)))]
    const layer = new InputLayer(h, shape, kw)
    return py.obj(new KerasTensor(shape, layer, [], autoName(h, 'keras_tensor')))
  })
  const layers: Record<string, Value> = {
    Input,
    InputLayer: layerType('InputLayer', (args, kw) => new InputLayer(h, [null, ...h.iterate(kw.shape ?? kw.input_shape ?? args[0]).map((v) => h.toInt(v))], kw)),
    Dense: layerType('Dense', (args, kw) => new Dense(h, intArg(kw.units ?? args[0], 'units'), { ...kw, activation: kw.activation ?? args[1] })),
    Conv2D: layerType('Conv2D', (args, kw) => new Conv2D(h, intArg(kw.filters ?? args[0], 'filters'), kw.kernel_size ?? args[1], { ...kw, strides: kw.strides ?? args[2], padding: kw.padding ?? args[3] })),
    MaxPooling2D: layerType('MaxPooling2D', (args, kw) => new Pool2D(h, 'max', args, kw)),
    MaxPool2D: layerType('MaxPooling2D', (args, kw) => new Pool2D(h, 'max', args, kw)),
    AveragePooling2D: layerType('AveragePooling2D', (args, kw) => new Pool2D(h, 'avg', args, kw)),
    Flatten: layerType('Flatten', (_a, kw) => new Simple(h, 'Flatten', 'flatten', kw, flattenShape, (e, x) => C.flattenT(e, x, 1, -1), 'Flatten()')),
    Dropout: layerType('Dropout', (args, kw) => {
      const rate = h.num(kw.rate ?? args[0])
      return new Simple(h, 'Dropout', 'dropout', kw, (s) => s, (e, x, training) => C.dropout(e, x, rate, training), `Dropout(${rate})`)
    }),
    Activation: layerType('Activation', (args, kw) => {
      const a = actArg(h, kw.activation ?? args[0])
      return new Simple(h, 'Activation', 'activation', kw, (s) => s, (e, x) => activate(e, x, a), `Activation('${a}')`)
    }),
    ReLU: layerType('ReLU', (_a, kw) => new Simple(h, 'ReLU', 're_lu', kw, (s) => s, (e, x) => C.unary(e, 'relu', x), 'ReLU()')),
    Softmax: layerType('Softmax', (_a, kw) => new Simple(h, 'Softmax', 'softmax', kw, (s) => s, (e, x) => C.softmax(e, x, -1, false), 'Softmax()')),
    Rescaling: layerType('Rescaling', (args, kw) => {
      const sc = h.num(kw.scale ?? args[0])
      const off = kw.offset ? h.num(kw.offset) : 0
      return new Simple(h, 'Rescaling', 'rescaling', kw, (s) => s, (e, x) => C.binary(e, '+', C.binary(e, '*', x, TT.scalar(sc, 'float32'), false, true), TT.scalar(off, 'float32'), false, true), `Rescaling(${sc})`)
    }),
    ZeroPadding2D: layerType('ZeroPadding2D', (args, kw) => {
      const p = pairArg(h, kw.padding ?? args[0], 1)
      return new Simple(h, 'ZeroPadding2D', 'zero_padding2d', kw, (s) => [null, (s[1] as number) + 2 * p[0], (s[2] as number) + 2 * p[1], s[3]], (e, x) => nhwc(e, x, [[p[0], p[0]], [p[1], p[1]]], (z) => z), `ZeroPadding2D((${p.join(', ')}))`)
    }),
    GlobalAveragePooling2D: layerType('GlobalAveragePooling2D', (_a, kw) => new Simple(h, 'GlobalAveragePooling2D', 'global_average_pooling2d', kw, (s) => [null, s[3]], (e, x) => C.reduceSum(e, x, [1, 2], false, true), 'GlobalAveragePooling2D()')),
    Reshape: layerType('Reshape', (args, kw) => {
      const target = h.iterate(kw.target_shape ?? args[0]).map((v) => h.toInt(v))
      return new Simple(h, 'Reshape', 'reshape', kw, (s) => [null, ...C.resolveShape(prod(s.slice(1) as number[]), target, [])], (e, x) => C.reshapeT(e, x, [x.shape[0], ...target], false), `Reshape((${target.join(', ')}))`)
    }),
    Concatenate: layerType('Concatenate', (_a, kw) => new Merge(h, 'Concatenate', kw)),
    Add: layerType('Add', (_a, kw) => new Merge(h, 'Add', kw)),
  }
  for (const missing of ['LSTM', 'GRU', 'SimpleRNN', 'Embedding', 'TextVectorization', 'MultiHeadAttention', 'BatchNormalization', 'LayerNormalization']) {
    layers[missing] = { k: 'type', name: missing, call: () => { throw h.err('NotImplementedError', `${missing} is not available in this sandbox (it covers the layers of lectures 6 and 8)`) } }
  }

  const Sequential: Value = {
    k: 'type', name: 'Sequential', call: (args, kw) => {
      const m = new KModel(h, false, kw.name && kw.name.k === 'str' ? kw.name.v : null)
      const ls = args[0] ?? kw.layers
      if (ls && ls.k !== 'none') for (const l of h.iterate(ls)) {
        if (l.k === 'obj' && l.o instanceof KerasTensor && l.o.layer instanceof InputLayer) m.inputShape = l.o.shape
        else m.add(h, l)
      }
      if (m.inputShape) m.build(h, m.inputShape)
      return py.obj(m)
    },
  }
  const Model: Value = {
    k: 'type', name: 'Model', call: (args, kw) => {
      const inV = kw.inputs ?? args[0]
      const outV = kw.outputs ?? args[1]
      if (!inV || !outV) throw h.err('TypeError', 'Model(inputs, outputs): both are needed for the Functional API')
      const ins = (inV.k === 'list' || inV.k === 'tuple' ? h.iterate(inV) : [inV]).map((v) => (v.k === 'obj' && v.o instanceof KerasTensor ? v.o : null))
      const out = outV.k === 'obj' && outV.o instanceof KerasTensor ? outV.o : null
      if (!out || ins.some((i) => !i)) throw h.err('ValueError', 'Model(inputs, outputs) expects KerasTensors from keras.Input(...) and layer calls')
      if (ins.length !== 1) throw h.err('NotImplementedError', 'models with several inputs are not supported in this sandbox')
      const m = new KModel(h, true, kw.name && kw.name.k === 'str' ? kw.name.v : null)
      m.inputs = ins as KerasTensor[]
      m.output = out
      m.inputShape = ins[0]!.shape
      m.layers = [...new Set(m.nodes().map((n) => n.layer))].filter((l) => !(l instanceof InputLayer))
      return py.obj(m)
    },
  }
  const lossFn = (name: string, spec: LossName): Value => ({ k: 'fn', name, call: (args) => tfWrap(lossValue(E, h, { name: spec, fromLogits: false }, asFloat(h, args[0]), asFloat(h, args[1]), false)) })
  const lossClass = (cls: string, spec: LossName): Value => ({ k: 'type', name: cls, call: (_a, kw) => py.obj(new LossObj(cls, { name: spec, fromLogits: !!(kw.from_logits && h.truthy(kw.from_logits)) })) })
  const losses: Record<string, Value> = {
    categorical_crossentropy: lossFn('categorical_crossentropy', 'categorical_crossentropy'),
    sparse_categorical_crossentropy: lossFn('sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'),
    binary_crossentropy: lossFn('binary_crossentropy', 'binary_crossentropy'),
    mean_squared_error: lossFn('mean_squared_error', 'mean_squared_error'),
    CategoricalCrossentropy: lossClass('CategoricalCrossentropy', 'categorical_crossentropy'),
    SparseCategoricalCrossentropy: lossClass('SparseCategoricalCrossentropy', 'sparse_categorical_crossentropy'),
    BinaryCrossentropy: lossClass('BinaryCrossentropy', 'binary_crossentropy'),
    MeanSquaredError: lossClass('MeanSquaredError', 'mean_squared_error'),
  }
  const optim = (cls: string, kind: 'adam' | 'sgd' | 'rmsprop', lr: number): Value => ({
    k: 'type', name: cls, call: (args, kw) => py.obj(new KOptimizer(kind, kw.learning_rate ?? kw.lr ?? args[0] ? h.num((kw.learning_rate ?? kw.lr ?? args[0])!) : lr, kw.momentum ? h.num(kw.momentum) : 0)),
  })
  const optimizers: Value = { k: 'module', name: 'keras.optimizers', attrs: { Adam: optim('Adam', 'adam', 0.001), SGD: optim('SGD', 'sgd', 0.01), RMSprop: optim('RMSprop', 'rmsprop', 0.001) } }
  const callbacks: Value = {
    k: 'module', name: 'keras.callbacks', attrs: {
      EarlyStopping: { k: 'type', name: 'EarlyStopping', call: (_a, kw) => py.obj(new Callback('EarlyStopping', kw.monitor ? h.str(kw.monitor) : 'val_loss', kw.patience ? h.toInt(kw.patience) : 0, !!(kw.restore_best_weights && h.truthy(kw.restore_best_weights)))) },
      TensorBoard: { k: 'type', name: 'TensorBoard', call: () => {
        h.print('(sandbox) TensorBoard logs are not written here; the training curves appear in the history step instead\n')
        return py.obj(new Callback('TensorBoard'))
      } },
      ModelCheckpoint: { k: 'type', name: 'ModelCheckpoint', call: () => py.obj(new Callback('ModelCheckpoint')) },
      History: { k: 'type', name: 'History', call: () => py.obj(new Callback('History')) },
    },
  }
  const toCategorical = fnv('to_categorical', (args, kw) => {
    const a = toArray(args[0] ?? kw.x)
    const v = a.values().map(Math.round)
    const n = kw.num_classes ?? args[1] ? h.toInt((kw.num_classes ?? args[1])!) : Math.max(...v) + 1
    for (const c of v) if (c >= n || c < 0) throw h.err('IndexError', `index ${c} is out of bounds for axis 1 with size ${n}`)
    const shape = a.shape.length && a.shape[a.shape.length - 1] === 1 && a.ndim > 1 ? a.shape.slice(0, -1) : a.shape
    return py.arr(NDArray.create(v.flatMap((c) => Array.from({ length: n }, (_, j) => (j === c ? 1 : 0))), [...shape, n], 'float64'))
  })
  const utils: Value = {
    k: 'module', name: 'keras.utils', attrs: {
      to_categorical: toCategorical,
      set_random_seed: fnv('set_random_seed', (args) => { h.rng.seed(h.toInt(args[0])); return py.NONE }),
      plot_model: fnv('plot_model', () => {
        h.print('(sandbox) plot_model draws a diagram with graphviz; here, model.summary() opens the same layers as a shape table\n')
        return py.NONE
      }),
    },
  }
  const models: Value = {
    k: 'module', name: 'keras.models', attrs: {
      Sequential, Model,
      load_model: fnv('load_model', (args) => {
        const v = S.files.get(h.str(args[0]))
        if (!v) throw h.err('ValueError', `File not found: filepath=${h.str(args[0])}. Please ensure the file is an accessible \`.keras\` zip file. (The sandbox only knows models saved with model.save in this run.)`)
        return v
      }),
    },
  }
  const datasets: Value = {
    k: 'module', name: 'keras.datasets', attrs: Object.fromEntries(['mnist', 'fashion_mnist', 'cifar10'].map((d) => [d, { k: 'module', name: `keras.datasets.${d}`, attrs: { load_data: fnv('load_data', () => { throw h.err('RuntimeError', `the sandbox cannot download ${d}: build a small NumPy dataset yourself (see the training examples)`) }) } }])),
  }
  const regularizers: Value = {
    k: 'module', name: 'keras.regularizers', attrs: {
      l2: fnv('l2', (args, kw) => py.obj(new Regularizer('l2', h.num(kw.l2 ?? args[0] ?? py.float(0.01))))),
      l1: fnv('l1', (args, kw) => py.obj(new Regularizer('l1', h.num(kw.l1 ?? args[0] ?? py.float(0.01))))),
      L2: fnv('L2', (args, kw) => py.obj(new Regularizer('l2', h.num(kw.l2 ?? args[0] ?? py.float(0.01))))),
      L1: fnv('L1', (args, kw) => py.obj(new Regularizer('l1', h.num(kw.l1 ?? args[0] ?? py.float(0.01))))),
    },
  }
  const activations: Value = { k: 'module', name: 'keras.activations', attrs: Object.fromEntries(['relu', 'sigmoid', 'softmax', 'tanh', 'linear'].map((a) => [a, fnv(a, (args) => tfWrap(activate(E, asFloat(h, args[0]), a as Act)))])) }
  const metrics: Value = { k: 'module', name: 'keras.metrics', attrs: { ...losses } }
  const keras: Value = {
    k: 'module', name: 'keras', attrs: {
      __version__: py.str('3.12 (sandbox)'),
      Sequential, Model, Input,
      layers: { k: 'module', name: 'keras.layers', attrs: layers },
      models, optimizers, losses: { k: 'module', name: 'keras.losses', attrs: losses }, callbacks, utils, datasets, activations, metrics, regularizers,
    },
  }

  // ---- tensorflow
  const tfd = Object.fromEntries(Object.entries(TF_DTYPES).map(([k, d]) => [k, py.obj(new TFDtype(d))]))
  const dt = (v: Value | undefined): TDType | null => (!v || v.k === 'none' ? null : v.k === 'obj' && v.o instanceof TFDtype ? v.o.d : v.k === 'str' && TF_DTYPES[v.v] ? TF_DTYPES[v.v] : (() => { throw h.err('TypeError', `Cannot convert ${repr(v)} to a TensorFlow DType.`) })())
  const T = (v: Value | undefined, what: string): TT => {
    if (!v) throw h.err('TypeError', `${what}() missing a tensor argument`)
    return tfOf(v) ?? toTF(h, v, null)
  }
  const axisArg = (v: Value | undefined): number[] | null => (!v || v.k === 'none' ? null : v.k === 'list' || v.k === 'tuple' ? h.iterate(v).map((x) => h.toInt(x)) : [h.toInt(v)])
  const reduce = (name: string, mean: boolean, f?: 'max' | 'min') => fnv(name, (args, kw) => {
    const t = T(args[0], name)
    const axes = axisArg(kw.axis ?? args[1])
    const keep = !!(kw.keepdims && h.truthy(kw.keepdims))
    if (f) {
      if (!axes) return tfWrap(C.maxAll(E, t, f === 'min'))
      return tfWrap(C.maxAlong(E, t, axes[0], keep, f === 'min').values)
    }
    let out: TT
    if (mean && !isFloatT(t.dtype)) {
      // TensorFlow keeps the integer dtype: the mean is truncated (a classic surprise)
      const m = C.reduceSum(E, C.toDtype(E, t, 'float64'), axes, keep, true)
      out = TT.of(m.values().map(Math.trunc), m.shape, t.dtype)
    } else out = C.reduceSum(E, t, axes, keep, mean)
    traceTF(h, `tf.${name}`, 'reduce', [{ label: h.nameOf(args[0], 'x'), t }], out, axes && axes.length === 1 ? C.normDim(axes[0], t.ndim) : null)
    return tfWrap(out)
  })
  const elem = (name: string, op: C.UnOpT) => fnv(name, (args) => tfWrap(C.unary(E, op, T(args[0], name))))
  const two = (name: string, op: string) => fnv(name, (args) => {
    const a = T(args[0], name)
    return new TFTensor(a).binop(op, args[1], false, h) ?? py.NONE
  })
  const confusion = fnv('confusion_matrix', (args, kw) => {
    const lab = toArray(args[0] ?? kw.labels).values().map(Math.round)
    const pred = toArray(args[1] ?? kw.predictions).values().map(Math.round)
    const n = kw.num_classes ? h.toInt(kw.num_classes) : Math.max(...lab, ...pred) + 1
    const m = new Array<number>(n * n).fill(0)
    lab.forEach((l, i) => (m[l * n + pred[i]] += 1))
    return tfWrap(TT.of(m, [n, n], 'int32'))
  })
  const tfMath: Value = {
    k: 'module', name: 'tensorflow.math', attrs: {
      confusion_matrix: confusion, reduce_sum: reduce('reduce_sum', false), reduce_mean: reduce('reduce_mean', true),
      reduce_max: reduce('reduce_max', false, 'max'), reduce_min: reduce('reduce_min', false, 'min'),
      exp: elem('exp', 'exp'), log: elem('log', 'log'), sqrt: elem('sqrt', 'sqrt'), square: elem('square', 'square'), abs: elem('abs', 'abs'),
      sigmoid: elem('sigmoid', 'sigmoid'), tanh: elem('tanh', 'tanh'), add: two('add', '+'), multiply: two('multiply', '*'), subtract: two('subtract', '-'), divide: two('divide', '/'),
      argmax: fnv('argmax', (args, kw) => tfWrap(C.argmax(T(args[0], 'argmax'), h.toInt(kw.axis ?? args[1] ?? py.int(0)), false))),
    },
  }
  const tfNN: Value = {
    k: 'module', name: 'tensorflow.nn', attrs: {
      relu: elem('relu', 'relu'), sigmoid: elem('sigmoid', 'sigmoid'), tanh: elem('tanh', 'tanh'),
      softmax: fnv('softmax', (args, kw) => tfWrap(C.softmax(E, T(args[0], 'softmax'), kw.axis ? h.toInt(kw.axis) : -1, false))),
    },
  }
  const random: Value = {
    k: 'module', name: 'tensorflow.random', attrs: {
      set_seed: fnv('set_seed', (args) => { h.rng.seed(h.toInt(args[0])); return py.NONE }),
      normal: fnv('normal', (args, kw) => {
        const shape = h.iterate(kw.shape ?? args[0]).map((v) => h.toInt(v))
        const mean = kw.mean ? h.num(kw.mean) : 0
        const sd = kw.stddev ? h.num(kw.stddev) : 1
        return tfWrap(TT.of(Array.from({ length: prod(shape) }, () => mean + sd * h.rng.normal()), shape, dt(kw.dtype) ?? 'float32'))
      }),
      uniform: fnv('uniform', (args, kw) => {
        const shape = h.iterate(kw.shape ?? args[0]).map((v) => h.toInt(v))
        const lo = kw.minval ? h.num(kw.minval) : 0
        const hi = kw.maxval ? h.num(kw.maxval) : 1
        return tfWrap(TT.of(Array.from({ length: prod(shape) }, () => lo + (hi - lo) * h.rng.next()), shape, dt(kw.dtype) ?? 'float32'))
      }),
    },
  }
  const fill = (name: string, v: number) => fnv(name, (args, kw) => {
    const shape = h.iterate(kw.shape ?? args[0]).map((x) => h.toInt(x))
    return tfWrap(TT.of(new Array(prod(shape)).fill(v), shape, dt(kw.dtype ?? args[1]) ?? 'float32'))
  })
  const tf: Record<string, Value> = {
    ...tfd,
    __version__: py.str('2.20 (sandbox)'),
    keras, math: tfMath, nn: tfNN, random,
    newaxis: py.NONE,
    constant: fnv('constant', (args, kw) => {
      const t = toTF(h, args[0] ?? kw.value, dt(kw.dtype ?? args[1]))
      traceTF(h, 'tf.constant', 'create', [], t)
      return tfWrap(t)
    }),
    convert_to_tensor: fnv('convert_to_tensor', (args, kw) => tfWrap(toTF(h, args[0], dt(kw.dtype)))),
    Variable: { k: 'type', name: 'Variable', call: (args, kw) => {
      const t = TT.of(T(args[0] ?? kw.initial_value, 'Variable').values(), T(args[0] ?? kw.initial_value, 'Variable').shape, dt(kw.dtype) ?? T(args[0] ?? kw.initial_value, 'Variable').dtype)
      return py.obj(new TFVariable(t, kw.name ? h.str(kw.name) : 'Variable', !kw.trainable || h.truthy(kw.trainable)))
    } },
    GradientTape: { k: 'type', name: 'GradientTape', call: (_a, kw) => py.obj(new GradientTape(!!(kw.persistent && h.truthy(kw.persistent)))) },
    zeros: fill('zeros', 0), ones: fill('ones', 1),
    fill: fnv('fill', (args) => {
      const shape = h.iterate(args[0]).map((x) => h.toInt(x))
      const v = args[1]
      return tfWrap(TT.of(new Array(prod(shape)).fill(h.num(v)), shape, v.k === 'float' ? 'float32' : 'int32'))
    }),
    range: fnv('range', (args, kw) => {
      const nums = args.map((a) => h.num(a))
      const [a, b, s] = nums.length === 1 ? [0, nums[0], 1] : [nums[0], nums[1], nums[2] ?? 1]
      const n = Math.max(0, Math.ceil((b - a) / s))
      return tfWrap(TT.of(Array.from({ length: n }, (_, i) => a + i * s), [n], dt(kw.dtype) ?? (args.some((x) => x.k === 'float') ? 'float32' : 'int32')))
    }),
    matmul: fnv('matmul', (args) => {
      const a = T(args[0], 'matmul')
      const b = T(args[1], 'matmul')
      if (a.dtype !== b.dtype) throw h.err('InvalidArgumentError', `cannot compute MatMul as input #1(zero-based) was expected to be a ${TF_KIND[a.dtype]} tensor but is a ${TF_KIND[b.dtype]} tensor [Op:MatMul]`)
      if (a.ndim < 2 || b.ndim < 2) throw h.err('InvalidArgumentError', `In[0] and In[1] has different ndims: [${a.shape.join(',')}] vs. [${b.shape.join(',')}] [Op:MatMul]`)
      if (a.shape[a.ndim - 1] !== b.shape[b.ndim - 2]) throw h.err('InvalidArgumentError', `Matrix size-incompatible: In[0]: [${a.shape.join(',')}], In[1]: [${b.shape.join(',')}] [Op:MatMul]`)
      const out = C.matmul(E, a, b, 'MatMul')
      traceTF(h, 'tf.matmul', 'matmul', [{ label: h.nameOf(args[0], 'a'), t: a }, { label: h.nameOf(args[1], 'b'), t: b }], out)
      return tfWrap(out)
    }),
    add: two('add', '+'), multiply: two('multiply', '*'), subtract: two('subtract', '-'), divide: two('divide', '/'),
    square: elem('square', 'square'), sqrt: elem('sqrt', 'sqrt'), exp: elem('exp', 'exp'), sigmoid: elem('sigmoid', 'sigmoid'), abs: elem('abs', 'abs'),
    reduce_sum: reduce('reduce_sum', false), reduce_mean: reduce('reduce_mean', true), reduce_max: reduce('reduce_max', false, 'max'), reduce_min: reduce('reduce_min', false, 'min'),
    argmax: fnv('argmax', (args, kw) => tfWrap(C.argmax(T(args[0], 'argmax'), h.toInt(kw.axis ?? args[1] ?? py.int(0)), false))),
    reshape: fnv('reshape', (args, kw) => tfWrap(C.reshapeT(E, T(args[0], 'reshape'), h.iterate(kw.shape ?? args[1]).map((v) => h.toInt(v)), false))),
    transpose: fnv('transpose', (args, kw) => {
      const t = T(args[0], 'transpose')
      const perm = kw.perm ?? args[1]
      return tfWrap(C.permuteT(E, t, perm ? h.iterate(perm).map((v) => h.toInt(v)) : t.shape.map((_, i) => t.ndim - 1 - i)))
    }),
    cast: fnv('cast', (args, kw) => tfWrap(C.toDtype(E, T(args[0], 'cast'), dt(kw.dtype ?? args[1]) ?? 'float32'))),
    shape: fnv('shape', (args) => tfWrap(TT.of(T(args[0], 'shape').shape, [T(args[0], 'shape').ndim], 'int32'))),
    concat: fnv('concat', (args, kw) => tfWrap(C.catT(E, h.iterate(args[0]).map((v) => T(v, 'concat')), h.toInt(kw.axis ?? args[1] ?? py.int(0))))),
    stack: fnv('stack', (args, kw) => tfWrap(C.catT(E, h.iterate(args[0]).map((v) => T(v, 'stack')), h.toInt(kw.axis ?? args[1] ?? py.int(0)), true))),
    one_hot: fnv('one_hot', (args, kw) => {
      const v = T(args[0], 'one_hot').values()
      const n = h.toInt(kw.depth ?? args[1])
      return tfWrap(TT.of(v.flatMap((c) => Array.from({ length: n }, (_, j) => (j === c ? 1 : 0))), [v.length, n], 'float32'))
    }),
    TensorShape: { k: 'type', name: 'TensorShape', call: (args) => py.obj(new TensorShape(h.iterate(args[0]).map((v) => (v.k === 'none' ? null : h.toInt(v))))) },
    Tensor: { k: 'type', name: 'Tensor', call: () => { throw h.err('TypeError', 'use tf.constant(...) to make a tensor') } },
  }
  const tfMod: Value = { k: 'module', name: 'tensorflow', attrs: tf }
  const kerasAttrs = (keras as { attrs: Record<string, Value> }).attrs
  const now = new Date()
  const datetime: Value = {
    k: 'module', name: 'datetime', attrs: {
      datetime: { k: 'module', name: 'datetime.datetime', attrs: {
        now: fnv('now', () => ({ k: 'module', name: 'datetime', attrs: {
          strftime: fnv('strftime', (args) => {
            const p = (n: number) => String(n).padStart(2, '0')
            const map: Record<string, string> = { Y: String(now.getFullYear()), m: p(now.getMonth() + 1), d: p(now.getDate()), H: p(now.getHours()), M: p(now.getMinutes()), S: p(now.getSeconds()) }
            return py.str(h.str(args[0]).replace(/%([YmdHMS])/g, (_, c: string) => map[c]))
          }),
        } })),
      } },
    },
  }
  return {
    keras, 'keras.layers': kerasAttrs.layers, 'keras.models': models, 'keras.optimizers': optimizers, 'keras.losses': kerasAttrs.losses, 'keras.callbacks': callbacks,
    'keras.utils': utils, 'keras.datasets': datasets, 'keras.activations': activations, 'keras.metrics': metrics, 'keras.regularizers': regularizers, 'tensorflow.keras.regularizers': regularizers,
    tensorflow: tfMod, 'tensorflow.keras': keras, 'tensorflow.keras.layers': kerasAttrs.layers, 'tensorflow.keras.models': models, 'tensorflow.keras.optimizers': optimizers,
    'tensorflow.keras.losses': kerasAttrs.losses, 'tensorflow.keras.callbacks': callbacks, 'tensorflow.keras.utils': utils, 'tensorflow.keras.datasets': datasets,
    'tensorflow.keras.activations': activations, 'tensorflow.keras.metrics': metrics, 'tensorflow.math': tfMath, 'tensorflow.nn': tfNN, 'tensorflow.random': random,
    datetime,
  }
}

export const KERAS: PyLib = {
  modules: [
    'keras', 'keras.layers', 'keras.models', 'keras.optimizers', 'keras.losses', 'keras.callbacks', 'keras.utils', 'keras.datasets', 'keras.activations', 'keras.metrics', 'keras.regularizers', 'tensorflow.keras.regularizers',
    'tensorflow', 'tensorflow.keras', 'tensorflow.keras.layers', 'tensorflow.keras.models', 'tensorflow.keras.optimizers', 'tensorflow.keras.losses', 'tensorflow.keras.callbacks',
    'tensorflow.keras.utils', 'tensorflow.keras.datasets', 'tensorflow.keras.activations', 'tensorflow.keras.metrics', 'tensorflow.math', 'tensorflow.nn', 'tensorflow.random', 'datetime',
  ],
  load: build,
}

