// --- 迷你 NumPy 内核：共享缓冲区 + shape/strides/offset，忠实复刻索引语义（View / Copy） ---
// 只覆盖教学所需子集：int64 / float64 / bool 三种 dtype；每个结果都能追溯到源缓冲区地址，供可视化使用。

export type DType = 'int64' | 'float64' | 'bool'

/** 带 Python 异常类型名的错误（IndexError / ValueError / TypeError …） */
export class PyError extends Error {
  pyType: string
  line?: number
  constructor(pyType: string, message: string, line?: number) {
    super(message)
    this.pyType = pyType
    this.line = line
  }
}

/** 沙盒内单个数组的元素上限，防止误操作卡死页面 */
export const MAX_SIZE = 4096

export const prod = (shape: readonly number[]): number => shape.reduce((p, x) => p * x, 1)

export const cStrides = (shape: readonly number[]): number[] => {
  const st = new Array<number>(shape.length)
  let acc = 1
  for (let i = shape.length - 1; i >= 0; i--) {
    st[i] = acc
    acc *= shape[i]
  }
  return st
}

export const shapeStr = (shape: readonly number[]): string =>
  shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(', ')})`

/** numpy 报错里的紧凑写法：(3,4) */
const shapeTight = (shape: readonly number[]): string =>
  shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(',')})`

export const checkSize = (shape: readonly number[]): void => {
  const n = prod(shape)
  if (n > MAX_SIZE) {
    throw new PyError('MemoryError', `array of ${n} elements exceeds the sandbox limit of ${MAX_SIZE}`)
  }
}

export const castValue = (v: number, dtype: DType): number => {
  if (dtype === 'bool') return v !== 0 ? 1 : 0
  if (dtype === 'int64') {
    if (!Number.isFinite(v)) throw new PyError('ValueError', `cannot convert float ${Number.isNaN(v) ? 'NaN' : 'infinity'} to integer`)
    return Math.trunc(v) || 0
  }
  return v
}

export class NDArray {
  data: number[]
  shape: number[]
  strides: number[]
  offset: number
  dtype: DType
  /** 拥有内存的数组；自身拥有内存时为 null（同 numpy 的 .base） */
  base: NDArray | null

  constructor(data: number[], shape: number[], strides: number[], offset: number, dtype: DType, base: NDArray | null) {
    this.data = data
    this.shape = shape
    this.strides = strides
    this.offset = offset
    this.dtype = dtype
    this.base = base
  }

  /** 新建拥有独立内存的 C 连续数组 */
  static create(values: number[], shape: number[], dtype: DType): NDArray {
    checkSize(shape)
    return new NDArray(values.map((v) => castValue(v, dtype)), [...shape], cStrides(shape), 0, dtype, null)
  }

  get ndim(): number {
    return this.shape.length
  }

  get size(): number {
    return prod(this.shape)
  }

  /** 以 C（行优先）顺序列出每个元素在缓冲区中的地址 */
  addresses(): number[] {
    const out: number[] = []
    if (this.size === 0) return out
    const n = this.ndim
    const idx = new Array<number>(n).fill(0)
    for (;;) {
      let a = this.offset
      for (let k = 0; k < n; k++) a += idx[k] * this.strides[k]
      out.push(a)
      let k = n - 1
      while (k >= 0) {
        idx[k]++
        if (idx[k] < this.shape[k]) break
        idx[k] = 0
        k--
      }
      if (k < 0) break
    }
    return out
  }

  values(): number[] {
    return this.addresses().map((a) => this.data[a])
  }

  copy(): NDArray {
    return NDArray.create(this.values(), this.shape, this.dtype)
  }

  isCContiguous(): boolean {
    return this.addresses().every((a, i) => a === this.offset + i)
  }

  /** 拥有内存的那个数组（视图链最终指向它） */
  owner(): NDArray {
    return this.base ?? this
  }

  /** 生成共享同一缓冲区的视图 */
  view(shape: number[], strides: number[], offset: number): NDArray {
    return new NDArray(this.data, shape, strides, offset, this.dtype, this.owner())
  }
}

export const scalarArray = (v: number, dtype: DType): NDArray => NDArray.create([v], [], dtype)

export const unravel = (flat: number, shape: readonly number[]): number[] => {
  const idx = new Array<number>(shape.length)
  for (let k = shape.length - 1; k >= 0; k--) {
    idx[k] = flat % shape[k]
    flat = Math.floor(flat / shape[k])
  }
  return idx
}

export const sharesMemory = (a: NDArray, b: NDArray): boolean => {
  if (a.data !== b.data) return false
  const set = new Set(a.addresses())
  return b.addresses().some((x) => set.has(x))
}

// ---------------- 广播 ----------------

export const broadcastShapes = (shapes: number[][], what = 'operands could not be broadcast together with shapes'): number[] => {
  const nd = Math.max(0, ...shapes.map((s) => s.length))
  const out = new Array<number>(nd).fill(1)
  for (const s of shapes) {
    for (let i = 0; i < s.length; i++) {
      const k = nd - s.length + i
      const d = s[i]
      if (out[k] === 1) out[k] = d
      else if (d !== 1 && d !== out[k]) {
        throw new PyError(what.startsWith('shape mismatch') ? 'IndexError' : 'ValueError', `${what} ${shapes.map(shapeTight).join(' ')} `)
      }
    }
  }
  return out
}

/** 把数组以 stride 0 的方式虚拟扩展到目标形状（不复制） */
export const broadcastTo = (a: NDArray, shape: number[]): NDArray => {
  const lead = shape.length - a.ndim
  const strides = shape.map((d, k) => {
    if (k < lead) return 0
    return a.shape[k - lead] === 1 && d !== 1 ? 0 : a.strides[k - lead]
  })
  return new NDArray(a.data, [...shape], strides, a.offset, a.dtype, a.owner())
}

// ---------------- 逐元素运算 ----------------

export type BinOp = '+' | '-' | '*' | '/' | '//' | '%' | '**' | '==' | '!=' | '<' | '<=' | '>' | '>=' | '&' | '|' | '^'

const COMPARE = new Set<BinOp>(['==', '!=', '<', '<=', '>', '>='])

const UFUNC_NAME: Record<BinOp, string> = {
  '+': 'add', '-': 'subtract', '*': 'multiply', '/': 'divide', '//': 'floor_divide', '%': 'remainder',
  '**': 'power', '==': 'equal', '!=': 'not_equal', '<': 'less', '<=': 'less_equal', '>': 'greater',
  '>=': 'greater_equal', '&': 'bitwise_and', '|': 'bitwise_or', '^': 'bitwise_xor',
}

export const resultDType = (op: BinOp, a: DType, b: DType): DType => {
  if (COMPARE.has(op)) return 'bool'
  if (op === '&' || op === '|' || op === '^') {
    if (a === 'float64' || b === 'float64') {
      throw new PyError('TypeError', `ufunc '${UFUNC_NAME[op]}' not supported for the input types`)
    }
    return a === 'bool' && b === 'bool' ? 'bool' : 'int64'
  }
  if (op === '/') return 'float64'
  return a === 'float64' || b === 'float64' ? 'float64' : 'int64'
}

/** 标量运算（Python 语义：// 向下取整，% 结果与除数同号） */
export const applyOp = (op: BinOp, x: number, y: number, dtype: DType): number => {
  const isInt = dtype !== 'float64'
  switch (op) {
    case '+': return x + y
    case '-': return x - y
    case '*': return x * y
    case '/': return x / y
    case '//':
      if (y === 0) return isInt ? 0 : x / y
      return Math.floor(x / y)
    case '%':
      if (y === 0) return isInt ? 0 : NaN
      return x - y * Math.floor(x / y)
    case '**':
      if (isInt && y < 0) throw new PyError('ValueError', 'Integers to negative integer powers are not allowed.')
      return x ** y
    case '==': return x === y ? 1 : 0
    case '!=': return x !== y ? 1 : 0
    case '<': return x < y ? 1 : 0
    case '<=': return x <= y ? 1 : 0
    case '>': return x > y ? 1 : 0
    case '>=': return x >= y ? 1 : 0
    case '&': return x & y
    case '|': return x | y
    case '^': return x ^ y
  }
}

export const binaryOp = (op: BinOp, a: NDArray, b: NDArray): NDArray => {
  const shape = broadcastShapes([a.shape, b.shape])
  checkSize(shape)
  const dtype = resultDType(op, a.dtype, b.dtype)
  const opType = COMPARE.has(op) ? (a.dtype === 'float64' || b.dtype === 'float64' ? 'float64' : 'int64') : dtype
  const av = broadcastTo(a, shape).values()
  const bv = broadcastTo(b, shape).values()
  return NDArray.create(av.map((x, i) => applyOp(op, x, bv[i], opType)), shape, dtype)
}

export type UnaryOp = '-' | '+' | '~'

export const unaryOp = (op: UnaryOp, a: NDArray): NDArray => {
  if (op === '+') return a.copy()
  if (op === '-') {
    if (a.dtype === 'bool') {
      throw new PyError('TypeError', 'The numpy boolean negative, the `-` operator, is not supported, use the `~` operator or the logical_not function instead.')
    }
    return NDArray.create(a.values().map((x) => -x || 0), a.shape, a.dtype)
  }
  if (a.dtype === 'float64') throw new PyError('TypeError', "ufunc 'invert' not supported for the input types")
  return NDArray.create(a.values().map((x) => (a.dtype === 'bool' ? 1 - x : ~x)), a.shape, a.dtype)
}

// ---------------- 形状变换 ----------------

export const transpose = (a: NDArray, axes?: number[]): NDArray => {
  const order = axes ?? a.shape.map((_, i) => a.ndim - 1 - i)
  if (order.length !== a.ndim || new Set(order.map((x) => normAxis(x, a.ndim))).size !== a.ndim) {
    throw new PyError('ValueError', "axes don't match array")
  }
  const ax = order.map((x) => normAxis(x, a.ndim))
  return a.view(ax.map((k) => a.shape[k]), ax.map((k) => a.strides[k]), a.offset)
}

export const reshape = (a: NDArray, target: number[]): NDArray => {
  const unknown = target.filter((d) => d === -1).length
  if (unknown > 1) throw new PyError('ValueError', 'can only specify one unknown dimension')
  const known = prod(target.filter((d) => d !== -1))
  let shape = target
  if (unknown === 1) {
    if (known === 0 || a.size % known !== 0) {
      throw new PyError('ValueError', `cannot reshape array of size ${a.size} into shape ${shapeStr(target)}`)
    }
    shape = target.map((d) => (d === -1 ? a.size / known : d))
  }
  if (shape.some((d) => d < 0) || prod(shape) !== a.size) {
    throw new PyError('ValueError', `cannot reshape array of size ${a.size} into shape ${shapeStr(target)}`)
  }
  // C 连续 → 视图（与 numpy 一致：reshape 尽量不复制）；否则先复制
  if (a.isCContiguous()) return a.view(shape, cStrides(shape), a.offset)
  return NDArray.create(a.values(), shape, a.dtype)
}

export const normAxis = (axis: number, ndim: number): number => {
  if (axis < -ndim || axis >= ndim) {
    throw new PyError('AxisError', `axis ${axis} is out of bounds for array of dimension ${ndim}`)
  }
  return axis < 0 ? axis + ndim : axis
}

// ---------------- 规约 ----------------

export type ReduceKind = 'sum' | 'mean' | 'max' | 'min' | 'argmax' | 'argmin' | 'any' | 'all'

const REDUCE_NAME: Record<ReduceKind, string> = {
  sum: 'add', mean: 'mean', max: 'maximum', min: 'minimum', argmax: 'argmax', argmin: 'argmin', any: 'logical_or', all: 'logical_and',
}

const reduceValues = (kind: ReduceKind, xs: number[]): number => {
  if (xs.length === 0 && ['max', 'min', 'argmax', 'argmin'].includes(kind)) {
    throw new PyError('ValueError', `zero-size array to reduction operation ${REDUCE_NAME[kind]} which has no identity`)
  }
  switch (kind) {
    case 'sum': return xs.reduce((s, x) => s + x, 0)
    case 'mean': return xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : NaN
    case 'max': return xs.reduce((m, x) => (x > m || Number.isNaN(x) ? x : m))
    case 'min': return xs.reduce((m, x) => (x < m || Number.isNaN(x) ? x : m))
    case 'argmax': return xs.reduce((bi, x, i) => (x > xs[bi] ? i : bi), 0)
    case 'argmin': return xs.reduce((bi, x, i) => (x < xs[bi] ? i : bi), 0)
    case 'any': return xs.some((x) => x !== 0) ? 1 : 0
    case 'all': return xs.every((x) => x !== 0) ? 1 : 0
  }
}

export const reduceDType = (kind: ReduceKind, dtype: DType): DType => {
  if (kind === 'mean') return 'float64'
  if (kind === 'any' || kind === 'all') return 'bool'
  if (kind === 'argmax' || kind === 'argmin') return 'int64'
  if (kind === 'sum' && dtype === 'bool') return 'int64'
  return dtype
}

/** axis 为 null 时返回标量；否则返回去掉该轴后的新数组 */
export const reduce = (a: NDArray, kind: ReduceKind, axis: number | null): NDArray | number => {
  if (axis === null) return reduceValues(kind, a.values())
  const ax = normAxis(axis, a.ndim)
  const rest = a.shape.map((_, i) => i).filter((i) => i !== ax)
  const moved = transpose(a, [...rest, ax])
  const vals = moved.values()
  const n = a.shape[ax]
  const outShape = rest.map((i) => a.shape[i])
  const out: number[] = []
  for (let i = 0; i < prod(outShape); i++) out.push(reduceValues(kind, vals.slice(i * n, i * n + n)))
  return NDArray.create(out, outShape, reduceDType(kind, a.dtype))
}

export const dot = (a: NDArray, b: NDArray): NDArray | number => {
  if (a.ndim === 0 || b.ndim === 0) return binaryOp('*', a, b)
  const A = a.ndim === 1 ? reshape(a, [1, a.shape[0]]) : a
  const B = b.ndim === 1 ? reshape(b, [b.shape[0], 1]) : b
  if (A.ndim !== 2 || B.ndim !== 2) throw new PyError('NotImplementedError', 'dot/@ supports 1-D and 2-D arrays only in this sandbox')
  const [n, k] = A.shape
  const [k2, mm] = B.shape
  if (k !== k2) {
    throw new PyError('ValueError', `shapes ${shapeTight(a.shape)} and ${shapeTight(b.shape)} not aligned: ${k} (dim ${a.ndim - 1}) != ${k2} (dim 0)`)
  }
  const av = A.values()
  const bv = B.values()
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < mm; j++) {
      let s = 0
      for (let t = 0; t < k; t++) s += av[i * k + t] * bv[t * mm + j]
      out.push(s)
    }
  }
  const dtype: DType = a.dtype === 'float64' || b.dtype === 'float64' ? 'float64' : 'int64'
  if (a.ndim === 1 && b.ndim === 1) return out[0]
  const shape = a.ndim === 1 ? [mm] : b.ndim === 1 ? [n] : [n, mm]
  return NDArray.create(out, shape, dtype)
}

// ---------------- 索引（核心） ----------------

export type IndexItem =
  | { kind: 'int'; value: number }
  | { kind: 'slice'; start: number | null; stop: number | null; step: number | null }
  | { kind: 'newaxis' }
  | { kind: 'ellipsis' }
  | { kind: 'array'; arr: NDArray }

export type NoteKind = 'int' | 'slice' | 'fancy' | 'mask' | 'newaxis' | 'ellipsis'

/** 每个索引项的解释：作用在哪些源轴上、选中了哪些下标 */
export interface AxisNote {
  label: string
  kind: NoteKind
  axes: number[]
  picks: number[]
  implicit?: boolean
}

export interface IndexPlan {
  shape: number[]
  /** 结果每个元素（C 顺序）对应的源缓冲区地址 */
  addresses: number[]
  advanced: boolean
  /** 全部轴都被整数索引 → numpy 返回标量 */
  scalar: boolean
  /** 基本索引的结果视图参数 */
  view?: { shape: number[]; strides: number[]; offset: number }
  /** 高级索引被切片隔开 → 广播维度被移到最前 */
  advFront: boolean
  notes: AxisNote[]
  mask?: { arr: NDArray; axes: number[] }
}

const fmtNum = (x: number | null): string => (x === null ? '' : String(x))

const fmtList = (arr: NDArray): string => {
  const vals = arr.values()
  if (arr.ndim === 0) return String(vals[0])
  const head = vals.slice(0, 8).join(', ')
  const body = vals.length > 8 ? `${head}, …` : head
  return arr.ndim === 1 ? `[${body}]` : `array${shapeStr(arr.shape)}`
}

export const indexLabel = (item: IndexItem): string => {
  switch (item.kind) {
    case 'int': return String(item.value)
    case 'slice': return `${fmtNum(item.start)}:${fmtNum(item.stop)}${item.step === null ? '' : `:${item.step}`}`
    case 'newaxis': return 'None'
    case 'ellipsis': return '...'
    case 'array': return item.arr.dtype === 'bool' ? `mask${shapeStr(item.arr.shape)}` : fmtList(item.arr)
  }
}

/** Python slice.indices() */
export const sliceIndices = (start: number | null, stop: number | null, step: number | null, n: number) => {
  const st = step ?? 1
  if (st === 0) throw new PyError('ValueError', 'slice step cannot be zero')
  const lower = st > 0 ? 0 : -1
  const upper = st > 0 ? n : n - 1
  const norm = (v: number | null, dflt: number) => {
    if (v === null) return dflt
    return v < 0 ? Math.max(v + n, lower) : Math.min(v, upper)
  }
  const s = norm(start, st > 0 ? lower : upper)
  const e = norm(stop, st > 0 ? upper : lower)
  const len = st > 0 ? Math.max(0, Math.ceil((e - s) / st)) : Math.max(0, Math.ceil((s - e) / -st))
  return { start: s, step: st, len }
}

const outOfBounds = (i: number, axis: number, n: number) =>
  new PyError('IndexError', `index ${i} is out of bounds for axis ${axis} with size ${n}`)

const normIndex = (i: number, axis: number, n: number): number => {
  if (i < -n || i >= n) throw outOfBounds(i, axis, n)
  return i < 0 ? i + n : i
}

type Part =
  | { t: 'slice'; ax: number; start: number; step: number; len: number }
  | { t: 'int'; ax: number; idx: number }
  | { t: 'new' }
  | { t: 'adv'; axes: number[]; arrays: NDArray[] }

export const planIndex = (a: NDArray, rawItems: IndexItem[]): IndexPlan => {
  const ellipses = rawItems.filter((it) => it.kind === 'ellipsis').length
  if (ellipses > 1) throw new PyError('IndexError', "an index can only have a single ellipsis ('...')")
  const consumes = (it: IndexItem) =>
    it.kind === 'int' || it.kind === 'slice' ? 1 : it.kind === 'array' ? (it.arr.dtype === 'bool' ? Math.max(1, it.arr.ndim) : 1) : 0
  const used = rawItems.reduce((s, it) => s + consumes(it), 0)
  if (used > a.ndim) {
    throw new PyError('IndexError', `too many indices for array: array is ${a.ndim}-dimensional, but ${used} were indexed`)
  }

  const notes: AxisNote[] = []
  const parts: Part[] = []
  let mask: IndexPlan['mask']
  let ax = 0
  const full = (implicit: boolean) => {
    const n = a.shape[ax]
    parts.push({ t: 'slice', ax, start: 0, step: 1, len: n })
    if (implicit) notes.push({ label: ':', kind: 'slice', axes: [ax], picks: Array.from({ length: n }, (_, i) => i), implicit: true })
    ax++
  }

  for (const it of rawItems) {
    if (it.kind === 'ellipsis') {
      const axes: number[] = []
      for (let k = 0; k < a.ndim - used; k++) {
        axes.push(ax)
        full(false)
      }
      notes.push({ label: '...', kind: 'ellipsis', axes, picks: [] })
    } else if (it.kind === 'newaxis') {
      parts.push({ t: 'new' })
      notes.push({ label: 'None', kind: 'newaxis', axes: [], picks: [] })
    } else if (it.kind === 'int') {
      const idx = normIndex(it.value, ax, a.shape[ax])
      parts.push({ t: 'int', ax, idx })
      notes.push({ label: indexLabel(it), kind: 'int', axes: [ax], picks: [idx] })
      ax++
    } else if (it.kind === 'slice') {
      const s = sliceIndices(it.start, it.stop, it.step, a.shape[ax])
      parts.push({ t: 'slice', ax, ...s })
      notes.push({ label: indexLabel(it), kind: 'slice', axes: [ax], picks: Array.from({ length: s.len }, (_, i) => s.start + i * s.step) })
      ax++
    } else if (it.arr.dtype === 'bool') {
      if (it.arr.ndim === 0) throw new PyError('NotImplementedError', 'a scalar boolean index (a[True]) is not supported in this sandbox')
      const m = it.arr
      const axes = m.shape.map((_, k) => ax + k)
      m.shape.forEach((d, k) => {
        if (d !== a.shape[ax + k]) {
          throw new PyError('IndexError', `boolean index did not match indexed array along axis ${ax + k}; size of axis is ${a.shape[ax + k]} but size of corresponding boolean axis is ${d}`)
        }
      })
      // nonzero：按行优先顺序收集 True 的坐标，每个轴一条整数数组
      const coords: number[][] = m.shape.map(() => [])
      m.values().forEach((v, flat) => {
        if (v) unravel(flat, m.shape).forEach((c, k) => coords[k].push(c))
      })
      parts.push({ t: 'adv', axes, arrays: coords.map((c) => NDArray.create(c, [c.length], 'int64')) })
      notes.push({ label: indexLabel(it), kind: 'mask', axes, picks: coords[0] ?? [] })
      mask ??= { arr: m, axes }
      ax += m.ndim
    } else {
      if (it.arr.dtype === 'float64') {
        throw new PyError('IndexError', 'arrays used as indices must be of integer (or boolean) type')
      }
      const n = a.shape[ax]
      const axis = ax
      const norm = NDArray.create(it.arr.values().map((v) => normIndex(v, axis, n)), it.arr.shape, 'int64')
      parts.push({ t: 'adv', axes: [ax], arrays: [norm] })
      notes.push({ label: indexLabel(it), kind: 'fancy', axes: [ax], picks: norm.values() })
      ax++
    }
  }
  while (ax < a.ndim) full(true)

  const advanced = parts.some((p) => p.t === 'adv')

  // ---- 基本索引：只改 shape/strides/offset → 视图 ----
  if (!advanced) {
    let offset = a.offset
    const shape: number[] = []
    const strides: number[] = []
    for (const p of parts) {
      if (p.t === 'int') offset += p.idx * a.strides[p.ax]
      else if (p.t === 'slice') {
        offset += p.start * a.strides[p.ax]
        shape.push(p.len)
        strides.push(p.step * a.strides[p.ax])
      } else if (p.t === 'new') {
        shape.push(1)
        strides.push(0)
      }
    }
    const v = new NDArray(a.data, shape, strides, offset, a.dtype, null)
    const scalar = shape.length === 0 && !parts.some((p) => p.t === 'new')
    return { shape, addresses: v.addresses(), advanced, scalar, view: { shape, strides, offset }, advFront: false, notes, mask }
  }

  // ---- 高级索引：整数也视为高级索引参与广播 → 复制 ----
  const advParts = parts.map((p, i) => ({ p, i })).filter(({ p }) => p.t === 'adv' || p.t === 'int')
  const advArrays: { axis: number; arr: NDArray }[] = []
  for (const { p } of advParts) {
    if (p.t === 'adv') p.axes.forEach((axis, k) => advArrays.push({ axis, arr: p.arrays[k] }))
    else if (p.t === 'int') advArrays.push({ axis: p.ax, arr: scalarArray(p.idx, 'int64') })
  }
  const bShape = broadcastShapes(advArrays.map((x) => x.arr.shape), 'shape mismatch: indexing arrays could not be broadcast together with shapes')
  const positions = advParts.map(({ i }) => i)
  const adjacent = positions.every((pos, k) => k === 0 || pos === positions[k - 1] + 1)

  const shape: number[] = []
  const sliceDim = new Map<number, number>() // part index → 结果维度位置
  let bStart = adjacent ? -1 : 0
  if (!adjacent) shape.push(...bShape)
  parts.forEach((p, i) => {
    if (p.t === 'slice') {
      sliceDim.set(i, shape.length)
      shape.push(p.len)
    } else if (p.t === 'new') shape.push(1)
    else if (adjacent && bStart === -1) {
      bStart = shape.length
      shape.push(...bShape)
    }
  })
  checkSize(shape)
  const bVals = advArrays.map((x) => ({ axis: x.axis, vals: broadcastTo(x.arr, bShape).values() }))
  const bStrides = cStrides(bShape)

  const addresses: number[] = []
  const total = prod(shape)
  for (let f = 0; f < total; f++) {
    const o = unravel(f, shape)
    let addr = a.offset
    let bFlat = 0
    for (let k = 0; k < bShape.length; k++) bFlat += o[bStart + k] * bStrides[k]
    parts.forEach((p, i) => {
      if (p.t === 'slice') addr += (p.start + p.step * o[sliceDim.get(i)!]) * a.strides[p.ax]
    })
    for (const b of bVals) addr += b.vals[bFlat] * a.strides[b.axis]
    addresses.push(addr)
  }
  return { shape, addresses, advanced, scalar: false, advFront: !adjacent && positions.length > 1, notes, mask }
}

/** 读取：基本索引返回视图（或标量），高级索引返回新数组 */
export const getIndex = (a: NDArray, items: IndexItem[]): { plan: IndexPlan; value: NDArray | number } => {
  const plan = planIndex(a, items)
  if (plan.scalar) return { plan, value: a.data[plan.addresses[0]] }
  if (plan.view) return { plan, value: a.view(plan.view.shape, plan.view.strides, plan.view.offset) }
  return { plan, value: NDArray.create(plan.addresses.map((x) => a.data[x]), plan.shape, a.dtype) }
}

/** 写入：无论视图还是 fancy/mask，都直接写回 a 的缓冲区 */
export const setIndex = (a: NDArray, items: IndexItem[], value: NDArray): IndexPlan => {
  const plan = planIndex(a, items)
  const lead = plan.shape.length - value.ndim
  const ok = lead >= 0 && value.shape.every((d, k) => d === 1 || d === plan.shape[lead + k])
  if (!ok) {
    throw new PyError('ValueError', `could not broadcast input array from shape ${shapeTight(value.shape)} into shape ${shapeTight(plan.shape)}`)
  }
  const vals = broadcastTo(value, plan.shape).values()
  plan.addresses.forEach((addr, i) => {
    a.data[addr] = castValue(vals[i], a.dtype)
  })
  return plan
}

// ---------------- repr / str 格式化（对齐 numpy 默认打印风格） ----------------

const fmtFloatRaw = (v: number): string => {
  if (Number.isNaN(v)) return 'nan'
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf'
  return v.toFixed(8).replace(/0+$/, '')
}

/** numpy 规则：最大值 ≥ 1e8，或最小非零绝对值 < 1e-4，或二者之比 > 1000 时改用科学计数法 */
const needsScientific = (values: number[]): boolean => {
  const abs = values.filter((v) => Number.isFinite(v) && v !== 0).map(Math.abs)
  if (!abs.length) return false
  const max = Math.max(...abs)
  const min = Math.min(...abs)
  return max >= 1e8 || min < 1e-4 || max / min > 1000
}

const formatScientific = (values: number[]): string[] => {
  const parts = values.map((v) => {
    if (!Number.isFinite(v)) return null
    const [mant, exp] = v.toExponential().split('e')
    const [i, f = ''] = mant.split('.')
    const e = Number(exp)
    return { i, f, e: `e${e < 0 ? '-' : '+'}${String(Math.abs(e)).padStart(2, '0')}` }
  })
  const fracW = Math.max(0, ...parts.map((p) => (p ? p.f.length : 0)))
  return parts.map((p, k) => (p ? `${p.i}.${p.f.padEnd(fracW, '0')}${p.e}` : fmtFloatRaw(values[k])))
}

export const formatElements = (values: number[], dtype: DType): string[] => {
  let words: string[]
  if (dtype === 'bool') words = values.map((v) => (v ? 'True' : 'False'))
  else if (dtype === 'int64') words = values.map(String)
  else if (needsScientific(values)) words = formatScientific(values)
  else {
    const raw = values.map(fmtFloatRaw)
    const parts = raw.map((s) => (s.includes('.') ? s.split('.') : [s, null]))
    const intW = Math.max(0, ...parts.map(([i]) => (i as string).length))
    const fracW = Math.max(0, ...parts.map(([, f]) => (f === null ? 0 : f.length)))
    words = parts.map(([i, f]) =>
      f === null ? (i as string).padStart(intW + 1 + fracW) : `${(i as string).padStart(intW)}.${f.padEnd(fracW)}`,
    )
  }
  const w = Math.max(0, ...words.map((s) => s.length))
  return words.map((s) => s.padStart(w))
}

const SUMMARY_THRESHOLD = 1000
const EDGE = 3
const LINE_WIDTH = 75

const shownIndices = (n: number, summarize: boolean): (number | null)[] =>
  summarize && n > 2 * EDGE
    ? [...Array.from({ length: EDGE }, (_, i) => i), null, ...Array.from({ length: EDGE }, (_, i) => n - EDGE + i)]
    : Array.from({ length: n }, (_, i) => i)

const formatNested = (a: NDArray, repr: boolean, indent: number): string => {
  const vals = a.values()
  const summarize = a.size > SUMMARY_THRESHOLD
  const st = cStrides(a.shape)
  // 只用显示出来的元素计算列宽（与 numpy 一致）
  const shownFlat: number[] = []
  const collect = (d: number, base: number) => {
    if (d === a.ndim) {
      shownFlat.push(base)
      return
    }
    for (const i of shownIndices(a.shape[d], summarize)) if (i !== null) collect(d + 1, base + i * st[d])
  }
  collect(0, 0)
  const words = formatElements(shownFlat.map((f) => vals[f]), a.dtype)
  const wordOf = new Map(shownFlat.map((f, i) => [f, words[i]]))
  const sep = repr ? ',' : ''

  const rec = (d: number, base: number): string => {
    const idx = shownIndices(a.shape[d], summarize)
    if (d === a.ndim - 1) {
      const items = idx.map((i) => (i === null ? '...' : wordOf.get(base + i)!))
      const lines: string[] = []
      let cur = ''
      const lead = indent + d + 1
      items.forEach((w, k) => {
        const word = k < items.length - 1 ? w + sep : w
        const next = cur ? `${cur} ${word}` : word
        if (cur && lead + next.length + (k === items.length - 1 ? a.ndim - d : 0) > LINE_WIDTH) {
          lines.push(cur)
          cur = word
        } else cur = next
      })
      lines.push(cur)
      return `[${lines.join('\n' + ' '.repeat(lead))}]`
    }
    const glue = `${sep}${'\n'.repeat(a.ndim - d - 1)}${' '.repeat(indent + d + 1)}`
    return `[${idx.map((i) => (i === null ? '...' : rec(d + 1, base + i * st[d]))).join(glue)}]`
  }
  return rec(0, 0)
}

export const formatScalar = (v: number, dtype: DType): string => {
  if (dtype === 'bool') return v ? 'True' : 'False'
  if (dtype === 'int64') return String(v)
  if (Number.isNaN(v)) return 'nan'
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf'
  return Number.isInteger(v) && Math.abs(v) < 1e16 ? `${v}.0` : String(v)
}

export const arrayRepr = (a: NDArray): string => {
  if (a.ndim === 0) return `array(${formatScalar(a.values()[0], a.dtype)})`
  if (a.size === 0) {
    return a.ndim === 1 ? `array([], dtype=${a.dtype})` : `array([], shape=${shapeStr(a.shape)}, dtype=${a.dtype})`
  }
  const suffix = a.size > SUMMARY_THRESHOLD ? `, shape=${shapeStr(a.shape)}` : ''
  return `array(${formatNested(a, true, 6)}${suffix})`
}

export const arrayStr = (a: NDArray): string => {
  if (a.ndim === 0) return formatScalar(a.values()[0], a.dtype)
  if (a.size === 0) return '[]'
  return formatNested(a, false, 0)
}
