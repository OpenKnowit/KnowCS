// --- 迷你 NumPy 扩展算子：拼接 / 平铺 / 翻转 / 排序 / 累加 / 线性代数 —— 语义与报错对齐真 numpy ---
import { NDArray, PyError, cStrides, checkSize, normAxis, prod, transpose, unravel } from './ndarray'
import type { DType } from './ndarray'

/** 多个数组合并后的 dtype：有 float 即 float64，否则有 int 即 int64，全 bool 才 bool */
export const commonDType = (arrs: NDArray[]): DType =>
  arrs.some((a) => a.dtype === 'float64') ? 'float64' : arrs.some((a) => a.dtype === 'int64') ? 'int64' : 'bool'

const flatIndex = (idx: number[], shape: readonly number[]): number => {
  const st = cStrides(shape)
  return idx.reduce((s, v, k) => s + v * st[k], 0)
}

// ---------------- 拼接 ----------------

export const concatenate = (arrs: NDArray[], axis: number | null = 0): NDArray => {
  if (!arrs.length) throw new PyError('ValueError', 'need at least one array to concatenate')
  if (axis === null) {
    const vals = arrs.flatMap((a) => a.values())
    return NDArray.create(vals, [vals.length], commonDType(arrs))
  }
  const nd = arrs[0].ndim
  if (nd === 0) throw new PyError('ValueError', 'zero-dimensional arrays cannot be concatenated')
  arrs.forEach((a, i) => {
    if (a.ndim !== nd) {
      throw new PyError('ValueError', `all the input arrays must have same number of dimensions, but the array at index 0 has ${nd} dimension(s) and the array at index ${i} has ${a.ndim} dimension(s)`)
    }
  })
  const ax = normAxis(axis, nd)
  for (let k = 0; k < nd; k++) {
    if (k === ax) continue
    arrs.forEach((a, i) => {
      if (a.shape[k] !== arrs[0].shape[k]) {
        throw new PyError('ValueError', `all the input array dimensions except for the concatenation axis must match exactly, but along dimension ${k}, the array at index 0 has size ${arrs[0].shape[k]} and the array at index ${i} has size ${a.shape[k]}`)
      }
    })
  }
  const shape = [...arrs[0].shape]
  shape[ax] = arrs.reduce((s, a) => s + a.shape[ax], 0)
  checkSize(shape)
  const vals = arrs.map((a) => a.values())
  const starts = arrs.map((_, i) => arrs.slice(0, i).reduce((s, a) => s + a.shape[ax], 0))
  const out: number[] = []
  for (let f = 0; f < prod(shape); f++) {
    const idx = unravel(f, shape)
    let i = arrs.length - 1
    while (starts[i] > idx[ax]) i--
    const local = [...idx]
    local[ax] -= starts[i]
    out.push(vals[i][flatIndex(local, arrs[i].shape)])
  }
  return NDArray.create(out, shape, commonDType(arrs))
}

export const expandDims = (a: NDArray, axis: number): NDArray => {
  const ax = normAxis(axis, a.ndim + 1)
  const shape = [...a.shape]
  const strides = [...a.strides]
  shape.splice(ax, 0, 1)
  strides.splice(ax, 0, 0)
  return a.view(shape, strides, a.offset)
}

export const stack = (arrs: NDArray[], axis = 0): NDArray => {
  if (!arrs.length) throw new PyError('ValueError', 'need at least one array to stack')
  if (arrs.some((a) => a.shape.join() !== arrs[0].shape.join())) throw new PyError('ValueError', 'all input arrays must have the same shape')
  const ax = normAxis(axis, arrs[0].ndim + 1)
  return concatenate(arrs.map((a) => expandDims(a, ax)), ax)
}

const atLeast2d = (a: NDArray): NDArray => (a.ndim === 0 ? a.view([1, 1], [0, 0], a.offset) : a.ndim === 1 ? a.view([1, a.shape[0]], [0, a.strides[0]], a.offset) : a)

export const vstack = (arrs: NDArray[]): NDArray => concatenate(arrs.map(atLeast2d), 0)

export const hstack = (arrs: NDArray[]): NDArray => {
  const ones = arrs.map((a) => (a.ndim === 0 ? a.view([1], [0], a.offset) : a))
  return concatenate(ones, ones[0].ndim === 1 ? 0 : 1)
}

// ---------------- 平铺 / 重复 / 翻转 / 轴变换 ----------------

export const tile = (a: NDArray, reps: number[]): NDArray => {
  const nd = Math.max(a.ndim, reps.length)
  const src = [...new Array(nd - a.ndim).fill(1), ...a.shape]
  const r = [...new Array(nd - reps.length).fill(1), ...reps]
  const shape = src.map((d, k) => d * r[k])
  checkSize(shape)
  const vals = a.values()
  return NDArray.create(Array.from({ length: prod(shape) }, (_, f) => vals[flatIndex(unravel(f, shape).map((v, k) => v % src[k]), src)]), shape, a.dtype)
}

export const repeat = (a: NDArray, n: number, axis: number | null): NDArray => {
  if (n < 0) throw new PyError('ValueError', 'negative dimensions are not allowed')
  if (axis === null) {
    const vals = a.values().flatMap((v) => new Array(n).fill(v))
    return NDArray.create(vals, [vals.length], a.dtype)
  }
  const ax = normAxis(axis, a.ndim)
  const shape = [...a.shape]
  shape[ax] *= n
  checkSize(shape)
  const vals = a.values()
  return NDArray.create(Array.from({ length: prod(shape) }, (_, f) => {
    const idx = unravel(f, shape)
    idx[ax] = Math.floor(idx[ax] / n)
    return vals[flatIndex(idx, a.shape)]
  }), shape, a.dtype)
}

/** np.flip 返回视图（负 stride） */
export const flip = (a: NDArray, axis: number | null): NDArray => {
  const axes = axis === null ? a.shape.map((_, k) => k) : [normAxis(axis, a.ndim)]
  let offset = a.offset
  const strides = [...a.strides]
  for (const k of axes) {
    if (a.shape[k] > 0) offset += (a.shape[k] - 1) * a.strides[k]
    strides[k] = -a.strides[k]
  }
  return a.view([...a.shape], strides, offset)
}

export const swapaxes = (a: NDArray, i: number, j: number): NDArray => {
  const order = a.shape.map((_, k) => k)
  const x = normAxis(i, a.ndim)
  const y = normAxis(j, a.ndim)
  ;[order[x], order[y]] = [order[y], order[x]]
  return transpose(a, order)
}

export const squeeze = (a: NDArray, axis: number | null = null): NDArray => {
  const drop = axis === null ? a.shape.map((d, k) => (d === 1 ? k : -1)).filter((k) => k >= 0) : [normAxis(axis, a.ndim)]
  for (const k of drop) if (a.shape[k] !== 1) throw new PyError('ValueError', 'cannot select an axis to squeeze out which has size not equal to one')
  const keep = a.shape.map((_, k) => k).filter((k) => !drop.includes(k))
  return a.view(keep.map((k) => a.shape[k]), keep.map((k) => a.strides[k]), a.offset)
}

// ---------------- 排序 / 去重 / 累加 ----------------

/** 升序比较：NaN 排在最后（与 numpy 一致） */
const asc = (x: number, y: number): number => (Number.isNaN(x) ? (Number.isNaN(y) ? 0 : 1) : Number.isNaN(y) ? -1 : x - y)

/**
 * 沿 axis 排序（axis = null 时先展平）。返回排好的数组、沿轴的 argsort 下标，
 * 以及 source[f]：输出第 f 个元素来自输入（C 顺序）第几个元素。
 */
export const sortAlong = (a: NDArray, axis: number | null): { sorted: NDArray; order: NDArray; source: number[] } => {
  const vals = a.values()
  if (axis === null || a.ndim <= 1) {
    if (a.ndim === 0) throw new PyError('AxisError', 'axis -1 is out of bounds for array of dimension 0')
    const idx = vals.map((_, i) => i).sort((i, j) => asc(vals[i], vals[j]) || i - j)
    return { sorted: NDArray.create(idx.map((i) => vals[i]), [vals.length], a.dtype), order: NDArray.create(idx, [idx.length], 'int64'), source: idx }
  }
  const ax = normAxis(axis, a.ndim)
  const n = a.shape[ax]
  const source = new Array<number>(vals.length)
  const order = new Array<number>(vals.length)
  const st = cStrides(a.shape)
  for (let f = 0; f < vals.length; f++) {
    const idx = unravel(f, a.shape)
    if (idx[ax] !== 0) continue
    const lane = Array.from({ length: n }, (_, t) => f + t * st[ax])
    const sortedLane = lane.map((_, t) => t).sort((p, q) => asc(vals[lane[p]], vals[lane[q]]) || p - q)
    sortedLane.forEach((t, pos) => {
      source[lane[pos]] = lane[t]
      order[lane[pos]] = t
    })
  }
  return { sorted: NDArray.create(source.map((s) => vals[s]), a.shape, a.dtype), order: NDArray.create(order, a.shape, 'int64'), source }
}

export const unique = (a: NDArray): NDArray => {
  const vals = [...new Set(a.values())].sort(asc)
  return NDArray.create(vals, [vals.length], a.dtype)
}

export const cumsum = (a: NDArray, axis: number | null): NDArray => {
  const dtype: DType = a.dtype === 'float64' ? 'float64' : 'int64'
  if (axis === null || a.ndim <= 1) {
    let s = 0
    const vals = a.values().map((v) => (s += v))
    return NDArray.create(vals, [vals.length], dtype)
  }
  const ax = normAxis(axis, a.ndim)
  const vals = a.values()
  const st = cStrides(a.shape)
  const out = [...vals]
  for (let f = 0; f < vals.length; f++) {
    const i = unravel(f, a.shape)[ax]
    if (i > 0) out[f] = out[f - st[ax]] + vals[f]
  }
  return NDArray.create(out, a.shape, dtype)
}

/** numpy 的 round：银行家舍入（四舍六入五成双） */
export const roundHalfEven = (x: number, decimals = 0): number => {
  if (!Number.isFinite(x)) return x
  const f = 10 ** decimals
  const y = x * f
  const r = Math.round(y)
  const tie = Math.abs(y - Math.trunc(y)) === 0.5
  return (tie ? 2 * Math.round(y / 2) : r) / f
}

// ---------------- 线性代数 ----------------

const square = (a: NDArray): number[][] => {
  if (a.ndim !== 2) throw new PyError('LinAlgError', `${a.ndim}-dimensional array given. Array must be at least two-dimensional`)
  if (a.shape[0] !== a.shape[1]) throw new PyError('LinAlgError', `Last 2 dimensions of the array must be square`)
  const v = a.values()
  const n = a.shape[0]
  return Array.from({ length: n }, (_, i) => v.slice(i * n, i * n + n))
}

export const det = (a: NDArray): number => {
  const m = square(a).map((r) => [...r])
  const n = m.length
  let d = 1
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r
    if (m[p][c] === 0) return 0
    if (p !== c) {
      ;[m[p], m[c]] = [m[c], m[p]]
      d = -d
    }
    d *= m[c][c]
    for (let r = c + 1; r < n; r++) {
      const f = m[r][c] / m[c][c]
      for (let k = c; k < n; k++) m[r][k] -= f * m[c][k]
    }
  }
  return d
}

export const inv = (a: NDArray): NDArray => {
  const m = square(a)
  const n = m.length
  const aug = m.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(aug[r][c]) > Math.abs(aug[p][c])) p = r
    if (Math.abs(aug[p][c]) < 1e-12) throw new PyError('LinAlgError', 'Singular matrix')
    ;[aug[p], aug[c]] = [aug[c], aug[p]]
    const piv = aug[c][c]
    for (let k = 0; k < 2 * n; k++) aug[c][k] /= piv
    for (let r = 0; r < n; r++) {
      if (r === c) continue
      const f = aug[r][c]
      for (let k = 0; k < 2 * n; k++) aug[r][k] -= f * aug[c][k]
    }
  }
  return NDArray.create(aug.flatMap((r) => r.slice(n).map((v) => (Math.abs(v) < 1e-15 ? 0 : v))), [n, n], 'float64')
}

/** 2-范数（axis = null 时为 Frobenius 范数） */
export const norm = (a: NDArray, axis: number | null): NDArray | number => {
  if (axis === null) return Math.sqrt(a.values().reduce((s, v) => s + v * v, 0))
  const ax = normAxis(axis, a.ndim)
  const rest = a.shape.map((_, k) => k).filter((k) => k !== ax)
  const moved = transpose(a, [...rest, ax]).values()
  const n = a.shape[ax]
  const outShape = rest.map((k) => a.shape[k])
  return NDArray.create(Array.from({ length: prod(outShape) }, (_, i) => Math.sqrt(moved.slice(i * n, i * n + n).reduce((s, v) => s + v * v, 0))), outShape, 'float64')
}

export const outer = (a: NDArray, b: NDArray): NDArray => {
  const x = a.values()
  const y = b.values()
  return NDArray.create(x.flatMap((u) => y.map((v) => u * v)), [x.length, y.length], commonDType([a, b]) === 'bool' ? 'bool' : commonDType([a, b]))
}

export const trace = (a: NDArray): number => {
  if (a.ndim < 2) throw new PyError('ValueError', 'diag requires an array of at least two dimensions')
  const v = a.values()
  const [r, c] = a.shape
  let s = 0
  for (let i = 0; i < Math.min(r, c); i++) s += v[i * c + i]
  return s
}

// ---------------- 可复现的伪随机数（与真 numpy 数值不同） ----------------

export class SandboxRandom {
  private s: number
  constructor(seed = 2211) {
    this.s = seed >>> 0
  }
  seed(n: number) {
    this.s = n >>> 0
  }
  next(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0
    let t = this.s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  normal(): number {
    const u = 1 - this.next()
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * this.next())
  }
}
