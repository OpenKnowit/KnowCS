// The rest of NumPy that coursework reaches for: more ufuncs, shape and set routines, np.linalg's solvers,
// np.random's distributions and generators. minipy.ts builds the core table and hands it here to extend.
import { NDArray, arrayRepr, arrayStr, binaryOp, broadcastShapes, broadcastTo, cStrides, castValue, checkSize, dot, isFloatD, normAxis, printOptions, prod, reshape, transpose, unravel } from './ndarray'
import type { BinOp, DType } from './ndarray'
import { commonDType, concatenate, flip, roundHalfEven, sortAlong } from './ndops'
import { PyDict, PyObj, formatSpec, fromScalar, isNum, py, repr, toArray } from './minipy'
import type { ApiKind, ApiMeta, Fn, Host, Kw, Value } from './minipy'
import { cholesky, eigGeneral, eigh, identity, lu, mul, qr, solveM, svdFull, svdThin, transposeM } from './npLinalg'
import type { Mat } from './npLinalg'
import * as R from './npRandom'
import { fullPath } from './pyStdlib'
import type { SandboxRandom } from './ndops'
import { LegacyRandom } from './mt19937'

type Api = (name: string, call: Fn, kind?: ApiKind, extra?: Partial<ApiMeta>) => Value

const LINE = '---------------------------------------------------------------'
const FINFO_64 = `Machine parameters for float64\n${LINE}\nprecision =  15   resolution = 1.0000000000000001e-15\nmachep =    -52   eps =        2.2204460492503131e-16\nnegep =     -53   epsneg =     1.1102230246251565e-16\nminexp =  -1022   tiny =       2.2250738585072014e-308\nmaxexp =   1024   max =        1.7976931348623157e+308\nnexp =       11   min =        -max\nsmallest_normal = 2.2250738585072014e-308   smallest_subnormal = 4.9406564584124654e-324\n${LINE}\n`
const FINFO_32 = `Machine parameters for float32\n${LINE}\nprecision =   6   resolution = 1.0000000e-06\nmachep =    -23   eps =        1.1920929e-07\nnegep =     -24   epsneg =     5.9604645e-08\nminexp =   -126   tiny =       1.1754944e-38\nmaxexp =    128   max =        3.4028235e+38\nnexp =        8   min =        -max\nsmallest_normal = 1.1754944e-38   smallest_subnormal = 1.4012985e-45\n${LINE}\n`

/** np.linspace's points, for scalar or array ends: the new axis comes first unless axis says otherwise */
export const spacedBetween = (start: Value, stop: Value, n: number, endpoint: boolean, axis: number, f: (x: number) => number, dtype: DType): NDArray => {
  const a0 = toArray(start, 'float64')
  const b0 = toArray(stop, 'float64')
  const shape = broadcastShapes([a0.shape, b0.shape])
  const a = broadcastTo(a0, shape).values()
  const b = broadcastTo(b0, shape).values()
  const size = prod(shape)
  const div = endpoint ? n - 1 : n
  const out = new Array<number>(n * size)
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < size; k++) {
      const v = div > 0 ? a[k] + i * ((b[k] - a[k]) / div) : a[k]
      out[i * size + k] = f(endpoint && n > 1 && i === n - 1 ? b[k] : v)
    }
  }
  const res = NDArray.create(out, [n, ...shape], dtype)
  const at = axis < 0 ? axis + shape.length + 1 : axis
  if (at === 0) return res
  const order = shape.map((_, d) => d + 1)
  order.splice(at, 0, 0)
  return transpose(res, order)
}

export const addNumpy = (h: Host, np: Record<string, Value>, random: Record<string, Value>, linalg: Record<string, Value>, api: { fn: Api; rnd: Api; la: Api }): void => {
  const { NONE, int, float, bool, tuple, str: S, list: L } = py
  // (a local name: vitest's module transform loses an imported base class inside a function)
  const Base = PyObj
  const { fn, rnd, la } = api
  const wrap = (a: NDArray): Value => ({ k: 'array', a })
  const mk = (vals: number[], shape: number[], d: DType = 'float64') => wrap(NDArray.create(vals, shape, d))
  const scalar = fromScalar
  const A = (v: Value | undefined, what = 'a'): NDArray => {
    if (!v) throw h.err('TypeError', `missing required argument '${what}'`)
    return toArray(v)
  }
  const opt = (args: Value[], kw: Kw, i: number, name: string): Value | undefined => {
    const v = kw[name] ?? args[i]
    return v && v.k !== 'none' ? v : undefined
  }
  const optInt = (args: Value[], kw: Kw, i: number, name: string): number | null => {
    const v = opt(args, kw, i, name)
    return v ? h.toInt(v) : null
  }
  const optNum = (args: Value[], kw: Kw, i: number, name: string, dflt: number): number => {
    const v = opt(args, kw, i, name)
    return v ? h.num(v) : dflt
  }
  const flag = (kw: Kw, name: string) => !!kw[name] && h.truthy(kw[name])
  const shapeOf = (v: Value): number[] => (isNum(v) ? [h.toInt(v)] : h.iterate(v).map((x) => h.toInt(x)))
  const dtypeArg = (v: Value | undefined): DType | undefined => {
    if (!v || v.k === 'none') return undefined
    if (v.k === 'dtype') return v.d
    if (v.k === 'type' && v.dtype) return v.dtype
    if (v.k === 'str') return ({ uint8: 'uint8', float32: 'float32', float64: 'float64', int: 'int64', int64: 'int64', int32: 'int64', float: 'float64', bool: 'bool' } as Record<string, DType>)[v.v]
    return undefined
  }
  const rows = (a: NDArray): Mat => {
    const v = a.values()
    const [r, c] = a.shape
    return Array.from({ length: r }, (_, i) => v.slice(i * c, i * c + c))
  }
  const matrix = (m: Mat, cols = m[0]?.length ?? 0) => mk(m.flat(), [m.length, cols])
  const square = (a: NDArray, name: string) => {
    if (a.ndim !== 2) throw h.err('LinAlgError', `${a.ndim}-dimensional array given. Array must be at least two-dimensional`)
    if (a.shape[0] !== a.shape[1]) throw h.err('LinAlgError', 'Last 2 dimensions of the array must be square')
    return name ? rows(a) : []
  }
  /** a NumPy 2 result such as EigResult(eigenvalues=…, eigenvectors=…) */
  const named = (tname: string, fields: [string, Value][]): Value => ({ k: 'tuple', items: fields.map((f) => f[1]), names: fields.map((f) => f[0]), tname })

  // ---------- axes ----------

  const moveAxes = (a: NDArray, from: number[], to: number[]): NDArray => {
    const src = from.map((x) => normAxis(x, a.ndim))
    const dst = to.map((x) => normAxis(x, a.ndim))
    const order = a.shape.map((_, i) => i).filter((i) => !src.includes(i))
    dst.map((d, i) => [d, src[i]]).sort((x, y) => x[0] - y[0]).forEach(([d, s]) => order.splice(d, 0, s))
    return transpose(a, order)
  }
  /** run f over each 1-D lane along an axis; its output replaces the lane */
  const lanes = (a: NDArray, axis: number, f: (lane: number[]) => number[], d: DType): NDArray => {
    const ax = normAxis(axis, a.ndim)
    const moved = moveAxes(a, [ax], [-1])
    const vals = moved.values()
    const n = a.shape[ax]
    const outer = moved.shape.slice(0, -1)
    const out: number[] = []
    let m = 0
    for (let i = 0; i < prod(outer); i++) {
      const r = f(vals.slice(i * n, i * n + n))
      m = r.length
      out.push(...r)
    }
    return moveAxes(NDArray.create(out, [...outer, m], d), [-1], [ax]).copy()
  }
  /** a reduction written on plain numbers, with NumPy's axis and keepdims */
  const reduceWith = (name: string, f: (xs: number[], kw: Kw) => number, outType: (a: NDArray) => DType) =>
    fn(name, (args, kw) => {
      const a = A(args[0])
      const axis = opt(args, kw, 1, 'axis')
      const d = outType(a)
      if (!axis) {
        const x = f(a.values(), kw)
        return flag(kw, 'keepdims') ? mk([x], a.shape.map(() => 1), d) : scalar(x, d)
      }
      const axes = isNum(axis) ? [h.toInt(axis)] : h.iterate(axis).map((x) => h.toInt(x))
      if (axes.length !== 1) {
        const norm = axes.map((x) => normAxis(x, a.ndim))
        const flat = moveAxes(a, norm, norm.map((_, i) => i - norm.length))
        const rest = flat.shape.slice(0, -norm.length)
        const res = lanes(reshape(flat.copy(), [...rest, -1]), -1, (lane) => [f(lane, kw)], d)
        return wrap(flag(kw, 'keepdims') ? reshape(res, a.shape.map((s, i) => (norm.includes(i) ? 1 : s))) : reshape(res, rest))
      }
      const res = lanes(a, axes[0], (lane) => [f(lane, kw)], d)
      if (flag(kw, 'keepdims')) return wrap(res)
      const shape = a.shape.filter((_, i) => i !== normAxis(axes[0], a.ndim))
      return shape.length ? wrap(reshape(res, shape)) : scalar(res.values()[0], d)
    }, 'reduce', { axisPos: 1, defaultAxis: null })

  // ---------- element-wise ----------

  const ufunc1 = (name: string, f: (x: number) => number, out: 'float' | 'same' | 'bool' = 'float') =>
    fn(name, (args) => {
      const a = A(args[0], 'x')
      const d: DType = out === 'bool' ? 'bool' : out === 'float' ? (a.dtype === 'float32' ? 'float32' : 'float64') : a.dtype === 'bool' ? 'int64' : a.dtype
      const res = a.values().map(f)
      return isNum(args[0]) ? scalar(res[0], d) : mk(res, a.shape, d)
    }, 'elementwise')
  const ufunc2 = (name: string, f: (x: number, y: number) => number, out: 'float' | 'same' | 'bool' = 'float') =>
    fn(name, (args) => {
      const a = A(args[0], 'x1')
      const b = A(args[1], 'x2')
      const shape = broadcastShapes([a.shape, b.shape])
      checkSize(shape)
      const y = broadcastTo(b, shape).values()
      const d: DType = out === 'bool' ? 'bool' : out === 'float' ? (isFloatD(commonDType([a, b])) ? commonDType([a, b]) : 'float64') : commonDType([a, b])
      const res = broadcastTo(a, shape).values().map((x, i) => f(x, y[i]))
      return isNum(args[0]) && isNum(args[1]) ? scalar(res[0], d) : mk(res, shape, d)
    }, 'elementwise')
  const gcd = (x: number, y: number): number => {
    let a = Math.abs(x)
    let b = Math.abs(y)
    while (b) [a, b] = [b, a % b]
    return a
  }
  const op2 = (name: string, op: string) => fn(name, (args) => h.binop(op, args[0], args[1]), 'elementwise')
  const shift = (x: number, by: number) => Math.trunc(x * 2 ** by)
  const sinc = (x: number) => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x))
  // numbers that are whole stay integers: np.floor of ints is ints since NumPy 2.1
  const keepInts = (name: string, f: (x: number) => number) => fn(name, (args) => {
    const a = A(args[0], 'x')
    const d: DType = a.dtype === 'bool' ? 'bool' : a.dtype
    const res = isFloatD(a.dtype) ? a.values().map(f) : a.values()
    return isNum(args[0]) ? scalar(res[0], d) : mk(res, a.shape, d)
  }, 'elementwise')
  Object.assign(np, {
    cbrt: ufunc1('cbrt', Math.cbrt), expm1: ufunc1('expm1', Math.expm1), log1p: ufunc1('log1p', Math.log1p), exp2: ufunc1('exp2', (x) => 2 ** x),
    deg2rad: ufunc1('deg2rad', (x) => (x * Math.PI) / 180), radians: ufunc1('radians', (x) => (x * Math.PI) / 180),
    rad2deg: ufunc1('rad2deg', (x) => (x * 180) / Math.PI), degrees: ufunc1('degrees', (x) => (x * 180) / Math.PI),
    sinc: ufunc1('sinc', sinc), fabs: ufunc1('fabs', Math.abs), rint: ufunc1('rint', (x) => roundHalfEven(x)),
    fix: keepInts('fix', Math.trunc), trunc: keepInts('trunc', Math.trunc), floor: np.floor, ceil: np.ceil,
    signbit: ufunc1('signbit', (x) => (x < 0 || Object.is(x, -0) ? 1 : 0), 'bool'),
    negative: fn('negative', (args) => {
      const a = A(args[0], 'x')
      const res = a.values().map((x) => (isFloatD(a.dtype) ? -x : -x || 0))
      return isNum(args[0]) ? scalar(res[0], a.dtype) : mk(res, a.shape, a.dtype)
    }, 'elementwise'), positive: ufunc1('positive', (x) => x, 'same'),
    reciprocal: fn('reciprocal', (args) => {
      const a = A(args[0], 'x')
      const f = isFloatD(a.dtype) ? (x: number) => 1 / x : (x: number) => Math.trunc(1 / x)
      return isNum(args[0]) ? scalar(f(a.values()[0]), a.dtype) : mk(a.values().map(f), a.shape, a.dtype)
    }, 'elementwise'),
    conj: ufunc1('conj', (x) => x, 'same'), conjugate: ufunc1('conjugate', (x) => x, 'same'), real: ufunc1('real', (x) => x, 'same'),
    imag: ufunc1('imag', () => 0, 'same'), angle: ufunc1('angle', (x) => (x < 0 ? Math.PI : 0)),
    isneginf: ufunc1('isneginf', (x) => (x === -Infinity ? 1 : 0), 'bool'), isposinf: ufunc1('isposinf', (x) => (x === Infinity ? 1 : 0), 'bool'),
    isreal: ufunc1('isreal', () => 1, 'bool'), iscomplex: ufunc1('iscomplex', () => 0, 'bool'),
    invert: ufunc1('invert', (x) => ~x, 'same'), bitwise_not: ufunc1('bitwise_not', (x) => ~x, 'same'), bitwise_invert: ufunc1('bitwise_invert', (x) => ~x, 'same'),
    hypot: ufunc2('hypot', Math.hypot), arctan2: ufunc2('arctan2', Math.atan2), atan2: ufunc2('atan2', Math.atan2),
    float_power: ufunc2('float_power', (x, y) => x ** y), copysign: ufunc2('copysign', (x, y) => (y < 0 || Object.is(y, -0) ? -Math.abs(x) : Math.abs(x))),
    fmod: ufunc2('fmod', (x, y) => x % y, 'same'),
    fmax: ufunc2('fmax', (x, y) => (Number.isNaN(x) ? y : Number.isNaN(y) ? x : Math.max(x, y)), 'same'),
    fmin: ufunc2('fmin', (x, y) => (Number.isNaN(x) ? y : Number.isNaN(y) ? x : Math.min(x, y)), 'same'),
    gcd: ufunc2('gcd', gcd, 'same'), lcm: ufunc2('lcm', (x, y) => (x && y ? Math.abs(x * y) / gcd(x, y) : 0), 'same'),
    heaviside: ufunc2('heaviside', (x, y) => (Number.isNaN(x) ? NaN : x < 0 ? 0 : x > 0 ? 1 : y)),
    logaddexp: ufunc2('logaddexp', (x, y) => Math.max(x, y) + Math.log1p(Math.exp(-Math.abs(x - y)))),
    logaddexp2: ufunc2('logaddexp2', (x, y) => Math.max(x, y) + Math.log2(1 + 2 ** -Math.abs(x - y))),
    ldexp: ufunc2('ldexp', (x, y) => x * 2 ** y), nextafter: ufunc2('nextafter', (x, y) => (x === y ? y : x + Math.sign(y - x) * Math.max(Math.abs(x) * 2 ** -52, Number.MIN_VALUE))),
    logical_xor: ufunc2('logical_xor', (x, y) => (!!x !== !!y ? 1 : 0), 'bool'),
    left_shift: ufunc2('left_shift', shift, 'same'), bitwise_left_shift: ufunc2('bitwise_left_shift', shift, 'same'),
    right_shift: ufunc2('right_shift', (x, y) => Math.floor(x / 2 ** y), 'same'), bitwise_right_shift: ufunc2('bitwise_right_shift', (x, y) => Math.floor(x / 2 ** y), 'same'),
    bitwise_and: op2('bitwise_and', '&'), bitwise_or: op2('bitwise_or', '|'), bitwise_xor: op2('bitwise_xor', '^'),
    equal: op2('equal', '=='), not_equal: op2('not_equal', '!='), less: op2('less', '<'), less_equal: op2('less_equal', '<='), greater: op2('greater', '>'), greater_equal: op2('greater_equal', '>='),
    floor_divide: op2('floor_divide', '//'), mod: op2('mod', '%'), remainder: op2('remainder', '%'), true_divide: np.divide,
    divmod: fn('divmod', (args) => tuple([h.binop('//', args[0], args[1]), h.binop('%', args[0], args[1])]), 'elementwise'),
    modf: fn('modf', (args) => {
      const a = A(args[0], 'x')
      const whole = a.values().map(Math.trunc)
      const frac = a.values().map((x, i) => x - whole[i])
      return isNum(args[0]) ? tuple([float(frac[0]), float(whole[0])]) : tuple([mk(frac, a.shape), mk(whole, a.shape)])
    }, 'elementwise'),
    frexp: fn('frexp', (args) => {
      const a = A(args[0], 'x')
      const exps = a.values().map((x) => (x === 0 || !Number.isFinite(x) ? 0 : Math.floor(Math.log2(Math.abs(x))) + 1))
      const mant = a.values().map((x, i) => x / 2 ** exps[i])
      return isNum(args[0]) ? tuple([float(mant[0]), int(exps[0])]) : tuple([mk(mant, a.shape), mk(exps, a.shape, 'int64')])
    }, 'elementwise'),
    isclose: fn('isclose', (args, kw) => {
      const a = A(args[0])
      const b = A(args[1], 'b')
      const rtol = optNum(args, kw, 2, 'rtol', 1e-5)
      const atol = optNum(args, kw, 3, 'atol', 1e-8)
      const nanEq = flag(kw, 'equal_nan')
      const shape = broadcastShapes([a.shape, b.shape])
      const y = broadcastTo(b, shape).values()
      const res = broadcastTo(a, shape).values().map((x, i) => {
        if (Number.isNaN(x) || Number.isNaN(y[i])) return nanEq && Number.isNaN(x) && Number.isNaN(y[i]) ? 1 : 0
        if (!Number.isFinite(x) || !Number.isFinite(y[i])) return x === y[i] ? 1 : 0
        return Math.abs(x - y[i]) <= atol + rtol * Math.abs(y[i]) ? 1 : 0
      })
      return isNum(args[0]) && isNum(args[1]) ? bool(res[0] === 1) : mk(res, shape, 'bool')
    }, 'elementwise'),
    nan_to_num: fn('nan_to_num', (args, kw) => {
      const a = A(args[0], 'x')
      const nan = kw.nan ? h.num(kw.nan) : 0
      const hi = kw.posinf && kw.posinf.k !== 'none' ? h.num(kw.posinf) : Number.MAX_VALUE
      const lo = kw.neginf && kw.neginf.k !== 'none' ? h.num(kw.neginf) : -Number.MAX_VALUE
      const res = a.values().map((x) => (Number.isNaN(x) ? nan : x === Infinity ? hi : x === -Infinity ? lo : x))
      return isNum(args[0]) ? scalar(res[0], a.dtype) : mk(res, a.shape, a.dtype)
    }, 'elementwise'),
  })

  // NumPy's own second names, and the array-API spellings it added in 2.0
  for (const [alias, name] of Object.entries({
    absolute: 'abs', amax: 'max', amin: 'min', acos: 'arccos', asin: 'arcsin', atan: 'arctan', acosh: 'arccosh', asinh: 'arcsinh', atanh: 'arctanh', pow: 'power', concat: 'concatenate',
  })) np[alias] ??= np[name]

  // ---------- creating arrays ----------

  const lineFrom = (start: number, stop: number, num: number, endpoint: boolean) =>
    Array.from({ length: num }, (_, i) => start + ((stop - start) * i) / (endpoint ? Math.max(1, num - 1) : num))
  const windowFn = (name: string, f: (i: number, m: number) => number) => fn(name, (args) => {
    const m = h.toInt(args[0])
    return mk(Array.from({ length: Math.max(0, m) }, (_, i) => (m === 1 ? 1 : f(i, m))), [Math.max(0, m)])
  }, 'create')
  const triPart = (name: string, keep: (i: number, j: number, k: number) => boolean) => fn(name, (args, kw) => {
    const a = A(args[0], 'm')
    const k = optInt(args, kw, 1, 'k') ?? 0
    const [r, c] = a.ndim === 1 ? [a.size, a.size] : a.shape.slice(-2)
    // a 1-D input is repeated into rows first
    const src = a.ndim === 1 ? broadcastTo(a, [r, c]) : a
    return mk(src.values().map((x, p) => {
      const q = p % (r * c)
      return keep(Math.floor(q / c), q % c, k) ? x : 0
    }), src.shape, a.dtype)
  }, 'move')
  const indicesPair = (name: string, keep: (i: number, j: number, k: number) => boolean, fromArray: boolean) => fn(name, (args, kw) => {
    const a = fromArray ? A(args[0], 'arr') : null
    const n = a ? a.shape[0] : h.toInt(args[0])
    const k = optInt(args, kw, 1, 'k') ?? 0
    const m = a ? a.shape[1] : optInt(args, kw, 2, 'm') ?? n
    const ii: number[] = []
    const jj: number[] = []
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) if (keep(i, j, k)) {
      ii.push(i)
      jj.push(j)
    }
    return tuple([mk(ii, [ii.length], 'int64'), mk(jj, [jj.length], 'int64')])
  }, 'create')
  const lower = (i: number, j: number, k: number) => j - i <= k
  const upper = (i: number, j: number, k: number) => j - i >= k
  const i0 = (x: number) => {
    let sum = 1
    let term = 1
    for (let k = 1; k < 60; k++) {
      term *= (x / (2 * k)) ** 2
      sum += term
    }
    return sum
  }
  Object.assign(np, {
    empty: np.zeros, empty_like: np.zeros_like, asanyarray: np.asarray, asfortranarray: np.asarray, ascontiguousarray: np.asarray, require: np.asarray, asarray_chkfinite: np.asarray,
    geomspace: fn('geomspace', (args, kw) => {
      const num = optInt(args, kw, 2, 'num') ?? 50
      const end = !kw.endpoint || h.truthy(kw.endpoint)
      if ((args[0].k !== 'array' && args[0].k !== 'list') && (args[1].k !== 'array' && args[1].k !== 'list')) {
        const a = h.num(args[0])
        const b = h.num(args[1])
        const res = lineFrom(Math.log10(Math.abs(a)), Math.log10(Math.abs(b)), num, end).map((x) => Math.sign(a) * 10 ** x)
        if (num) res[0] = a
        if (num > 1 && end) res[num - 1] = b
        return mk(res, [num])
      }
      const logged = (v: Value) => wrap(NDArray.create(toArray(v).values().map((x) => Math.log10(x)), toArray(v).shape, 'float64'))
      return wrap(spacedBetween(logged(args[0]), logged(args[1]), num, end, 0, (x) => 10 ** x, 'float64'))
    }, 'create'),
    logspace: fn('logspace', (args, kw) => {
      const num = optInt(args, kw, 2, 'num') ?? 50
      const base = kw.base ? h.num(kw.base) : 10
      return wrap(spacedBetween(args[0], args[1], num, !kw.endpoint || h.truthy(kw.endpoint), 0, (x) => base ** x, 'float64'))
    }, 'create'),
    tri: fn('tri', (args, kw) => {
      const n = h.toInt(args[0])
      const m = optInt(args, kw, 1, 'M') ?? n
      const k = optInt(args, kw, 2, 'k') ?? 0
      return mk(Array.from({ length: n * m }, (_, p) => (lower(Math.floor(p / m), p % m, k) ? 1 : 0)), [n, m], dtypeArg(kw.dtype) ?? 'float64')
    }, 'create'),
    tril: triPart('tril', lower), triu: triPart('triu', upper),
    tril_indices: indicesPair('tril_indices', lower, false), triu_indices: indicesPair('triu_indices', upper, false),
    tril_indices_from: indicesPair('tril_indices_from', lower, true), triu_indices_from: indicesPair('triu_indices_from', upper, true),
    diag_indices: fn('diag_indices', (args, kw) => {
      const n = h.toInt(args[0])
      const nd = optInt(args, kw, 1, 'ndim') ?? 2
      const ix = Array.from({ length: n }, (_, i) => i)
      return tuple(Array.from({ length: nd }, () => mk(ix, [n], 'int64')))
    }, 'create'),
    diag_indices_from: fn('diag_indices_from', (args) => {
      const a = A(args[0], 'arr')
      const ix = Array.from({ length: a.shape[0] }, (_, i) => i)
      return tuple(a.shape.map(() => mk(ix, [ix.length], 'int64')))
    }, 'create'),
    vander: fn('vander', (args, kw) => {
      const x = A(args[0], 'x').values()
      const n = optInt(args, kw, 1, 'N') ?? x.length
      const inc = (kw.increasing ?? args[2]) && h.truthy(kw.increasing ?? args[2])
      return mk(x.flatMap((v) => Array.from({ length: n }, (_, j) => v ** (inc ? j : n - 1 - j))), [x.length, n], commonDType([A(args[0], 'x')]))
    }, 'create'),
    diag: fn('diag', (args, kw) => {
      const a = A(args[0], 'v')
      const k = optInt(args, kw, 1, 'k') ?? 0
      if (a.ndim === 1) {
        const n = a.size + Math.abs(k)
        const out = new Array<number>(n * n).fill(0)
        a.values().forEach((x, i) => { out[(i + Math.max(0, -k)) * n + i + Math.max(0, k)] = x })
        return mk(out, [n, n], a.dtype)
      }
      if (a.ndim !== 2) throw h.err('ValueError', 'Input must be 1- or 2-d.')
      const [r, c] = a.shape
      const v = a.values()
      const out: number[] = []
      for (let i = Math.max(0, -k), j = Math.max(0, k); i < r && j < c; i++, j++) out.push(v[i * c + j])
      return mk(out, [out.length], a.dtype)
    }, 'move'),
    diagflat: fn('diagflat', (args, kw) => {
      const v = A(args[0], 'v').values()
      const k = optInt(args, kw, 1, 'k') ?? 0
      const n = v.length + Math.abs(k)
      const out = new Array<number>(n * n).fill(0)
      v.forEach((x, i) => { out[(i + Math.max(0, -k)) * n + i + Math.max(0, k)] = x })
      return mk(out, [n, n], A(args[0], 'v').dtype)
    }, 'move'),
    diagonal: fn('diagonal', (args, kw) => {
      const a = A(args[0])
      if (a.ndim < 2) throw h.err('ValueError', 'diag requires an array of at least two dimensions')
      return diagonalOf(a, optInt(args, kw, 1, 'offset') ?? 0, optInt(args, kw, 2, 'axis1') ?? 0, optInt(args, kw, 3, 'axis2') ?? 1)
    }, 'move'),
    indices: fn('indices', (args, kw) => {
      const dims = shapeOf(args[0])
      const total = prod(dims)
      const st = cStrides(dims)
      const parts = dims.map((_, k) => Array.from({ length: total }, (_x, p) => Math.floor(p / st[k]) % dims[k]))
      return mk(parts.flat(), [dims.length, ...dims], dtypeArg(kw.dtype) ?? 'int64')
    }, 'create'),
    fromfunction: fn('fromfunction', (args, kw) => {
      const dims = shapeOf(args[1])
      const total = prod(dims)
      checkSize(dims)
      const st = cStrides(dims)
      const d = dtypeArg(kw.dtype) ?? 'float64'
      const grids = dims.map((_, k) => mk(Array.from({ length: total }, (_x, p) => Math.floor(p / st[k]) % dims[k]), dims, d))
      const rest: Kw = { ...kw }
      delete rest.dtype
      return h.call(args[0], grids, rest)
    }, 'create'),
    fromiter: fn('fromiter', (args, kw) => {
      const count = optInt(args, kw, 2, 'count') ?? -1
      const next = h.iterOf(args[0])
      const out: number[] = []
      for (let x = next(); x !== undefined && (count < 0 || out.length < count); x = next()) out.push(h.num(x))
      return mk(out, [out.length], dtypeArg(kw.dtype ?? args[1]) ?? 'float64')
    }, 'create'),
    fromstring: fn('fromstring', (args, kw) => {
      if (args[0]?.k !== 'str') throw h.err('NotImplementedError', 'np.fromstring reads text here: give it a str and sep=')
      const sep = kw.sep?.k === 'str' ? kw.sep.v : ' '
      const parts = args[0].v.split(sep === ' ' ? /\s+/ : sep).filter((p) => p.trim() !== '')
      return mk(parts.map(Number), [parts.length], dtypeArg(kw.dtype ?? args[1]) ?? 'float64')
    }, 'create'),
    hamming: windowFn('hamming', (i, m) => 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (m - 1))),
    hanning: windowFn('hanning', (i, m) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (m - 1))),
    blackman: windowFn('blackman', (i, m) => 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (m - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (m - 1))),
    bartlett: windowFn('bartlett', (i, m) => 1 - Math.abs((2 * i) / (m - 1) - 1)),
    kaiser: fn('kaiser', (args) => {
      const m = h.toInt(args[0])
      const beta = h.num(args[1])
      return mk(Array.from({ length: m }, (_, i) => (m === 1 ? 1 : i0(beta * Math.sqrt(1 - ((2 * i) / (m - 1) - 1) ** 2)) / i0(beta))), [m])
    }, 'create'),
    i0: ufunc1('i0', i0),
  })
  function diagonalOf(a: NDArray, offset: number, axis1: number, axis2: number): Value {
    const moved = moveAxes(a, [axis1, axis2], [-2, -1])
    const [r, c] = moved.shape.slice(-2)
    const outer = moved.shape.slice(0, -2)
    const v = moved.values()
    const out: number[] = []
    let len = 0
    for (let b = 0; b < prod(outer); b++) {
      len = 0
      for (let i = Math.max(0, -offset), j = Math.max(0, offset); i < r && j < c; i++, j++) {
        out.push(v[b * r * c + i * c + j])
        len++
      }
    }
    return mk(out, [...outer, len], a.dtype)
  }

  // np.mgrid[0:3, 0:2], np.r_[1:4, 7], np.s_[1:3] … all read a slice as a range or, with a 5j step, a line of 5 points
  class IndexMaker extends Base {
    readonly cls = 'numpy.lib._index_tricks_impl.IndexClass'
    name: string
    make: (items: Value[]) => Value
    constructor(name: string, make: (items: Value[]) => Value) {
      super()
      this.name = name
      this.make = make
    }
    repr() { return `<numpy.${this.name}>` }
    getItem(idx: Value): Value { return this.make(idx.k === 'tuple' ? idx.items : [idx]) }
  }
  const imagOf = (v: Value) => (v.k === 'obj' && v.o.cls === 'complex' ? (v.o as unknown as { v: number }).v : null)
  const rangeOf = (s: Value & { k: 'slice' }): { vals: number[]; d: DType } => {
    const start = s.start.k === 'none' ? 0 : h.num(s.start)
    const stop = h.num(s.stop)
    const pts = imagOf(s.step)
    if (pts !== null) return { vals: lineFrom(start, stop, pts, true), d: 'float64' }
    const step = s.step.k === 'none' ? 1 : h.num(s.step)
    const n = Math.max(0, Math.ceil((stop - start) / step))
    const isFloat = [s.start, s.stop, s.step].some((v) => v.k === 'float')
    return { vals: Array.from({ length: n }, (_, i) => start + i * step), d: isFloat ? 'float64' : 'int64' }
  }
  const asSlice = (v: Value): Value & { k: 'slice' } => {
    if (v.k !== 'slice') throw h.err('TypeError', 'expected a slice such as 0:3')
    return v
  }
  const gridFrom = (items: Value[], sparse: boolean): Value => {
    const axes = items.map((s) => rangeOf(asSlice(s)))
    const d: DType = axes.some((x) => x.d === 'float64') ? 'float64' : 'int64'
    const dims = axes.map((x) => x.vals.length)
    if (sparse) return tuple(axes.map((x, k) => mk(x.vals, dims.map((n, j) => (j === k ? n : 1)), d)))
    const total = prod(dims)
    checkSize([total * dims.length])
    const st = cStrides(dims)
    if (dims.length === 1) return mk(axes[0].vals, dims, d)
    return mk(axes.flatMap((x, k) => Array.from({ length: total }, (_, p) => x.vals[Math.floor(p / st[k]) % dims[k]])), [dims.length, ...dims], d)
  }
  /** np.r_ and np.c_: slices become ranges, everything else an array; 'c' reads them as columns */
  const concatMaker = (items: Value[], columns: boolean): Value => {
    const parts = items.map((v) => {
      if (v.k === 'slice') {
        const r = rangeOf(v)
        return NDArray.create(r.vals, [r.vals.length], r.d)
      }
      return toArray(v)
    })
    if (columns) {
      const cols = parts.map((p) => (p.ndim <= 1 ? reshape(p.ndim ? p : reshape(p, [1]), [-1, 1]) : p))
      return wrap(concatenate(cols, 1))
    }
    return wrap(concatenate(parts.map((p) => (p.ndim ? p : reshape(p, [1]))), 0))
  }
  Object.assign(np, {
    mgrid: py.obj(new IndexMaker('mgrid', (it) => gridFrom(it, false))),
    ogrid: py.obj(new IndexMaker('ogrid', (it) => gridFrom(it, true))),
    r_: py.obj(new IndexMaker('r_', (it) => concatMaker(it, false))),
    c_: py.obj(new IndexMaker('c_', (it) => concatMaker(it, true))),
    s_: py.obj(new IndexMaker('s_', (it) => (it.length === 1 ? it[0] : tuple(it)))),
    index_exp: py.obj(new IndexMaker('index_exp', (it) => tuple(it))),
  })

  // ---------- reshaping and joining ----------

  const atLeast = (name: string, shaped: (a: NDArray) => number[]) => fn(name, (args) => {
    const out = args.map((v) => {
      const a = toArray(v)
      const s = shaped(a)
      return wrap(s.length === a.ndim ? a : reshape(a, s))
    })
    return out.length === 1 ? out[0] : tuple(out)
  }, 'move')
  const splitAt = (a: NDArray, cuts: number[], axis: number): Value => {
    const ax = normAxis(axis, a.ndim)
    const moved = moveAxes(a, [ax], [0])
    const rest = moved.shape.slice(1)
    const unit = prod(rest)
    const v = moved.values()
    const n = a.shape[ax]
    const edges = [0, ...cuts.map((c) => Math.min(n, c < 0 ? Math.max(0, n + c) : c)), n]
    return L(edges.slice(1).map((end, i) => {
      const from = edges[i]
      const piece = NDArray.create(from < end ? v.slice(from * unit, end * unit) : [], [Math.max(0, end - from), ...rest], a.dtype)
      return wrap(moveAxes(piece, [0], [ax]).copy())
    }))
  }
  const splitter = (name: string, fixedAxis: number | null, strict: boolean) => fn(name, (args, kw) => {
    const a = A(args[0], 'ary')
    const how = kw.indices_or_sections ?? args[1]
    const axis = fixedAxis ?? optInt(args, kw, 2, 'axis') ?? 0
    if (!a.ndim || normAxis(axis, a.ndim) >= a.ndim) throw h.err('ValueError', `${name} only works on arrays of ${axis + 1} or more dimensions`)
    const n = a.shape[normAxis(axis, a.ndim)]
    if (isNum(how)) {
      const parts = h.toInt(how)
      if (parts <= 0) throw h.err('ValueError', 'number sections must be larger than 0.')
      if (strict && n % parts) throw h.err('ValueError', 'array split does not result in an equal division')
      const size = Math.floor(n / parts)
      const big = n % parts
      const cuts: number[] = []
      for (let i = 1, at = 0; i < parts; i++) {
        at += size + (i <= big ? 1 : 0)
        cuts.push(at)
      }
      return splitAt(a, cuts, axis)
    }
    return splitAt(a, h.iterate(how).map((x) => h.toInt(x)), axis)
  }, 'move')
  const seqOf = (v: Value | undefined, name: string): NDArray[] => {
    if (!v) throw h.err('TypeError', `${name}() expects a sequence of arrays`)
    return h.iterate(v).map((x) => toArray(x))
  }
  const blockOf = (v: Value, depth: number, maxDepth: number): NDArray => {
    if (v.k !== 'list') return toArray(v)
    const parts = v.items.map((x) => blockOf(x, depth + 1, maxDepth))
    const nd = Math.max(...parts.map((p) => p.ndim), maxDepth - depth)
    const lifted = parts.map((p) => (p.ndim < nd ? reshape(p, [...new Array<number>(nd - p.ndim).fill(1), ...p.shape]) : p))
    return concatenate(lifted, nd - (maxDepth - depth))
  }
  const depthOf = (v: Value): number => (v.k === 'list' ? 1 + Math.max(0, ...v.items.map(depthOf)) : 0)
  const rollFlat = (vals: number[], by: number) => {
    const n = vals.length
    if (!n) return vals
    const k = ((by % n) + n) % n
    return [...vals.slice(n - k), ...vals.slice(0, n - k)]
  }
  Object.assign(np, {
    atleast_1d: atLeast('atleast_1d', (a) => (a.ndim ? a.shape : [1])),
    atleast_2d: atLeast('atleast_2d', (a) => (a.ndim === 0 ? [1, 1] : a.ndim === 1 ? [1, a.size] : a.shape)),
    atleast_3d: atLeast('atleast_3d', (a) => (a.ndim === 0 ? [1, 1, 1] : a.ndim === 1 ? [1, a.size, 1] : a.ndim === 2 ? [...a.shape, 1] : a.shape)),
    split: splitter('split', null, true), array_split: splitter('array_split', null, false),
    vsplit: splitter('vsplit', 0, true), hsplit: fn('hsplit', (args) => h.call(splitter('hsplit', A(args[0], 'ary').ndim > 1 ? 1 : 0, true), args), 'move'), dsplit: splitter('dsplit', 2, true),
    column_stack: fn('column_stack', (args) => wrap(concatenate(seqOf(args[0], 'column_stack').map((a) => (a.ndim < 2 ? reshape(a, [-1, 1]) : a)), 1)), 'move'),
    row_stack: np.vstack,
    dstack: fn('dstack', (args) => wrap(concatenate(seqOf(args[0], 'dstack').map((a) => (a.ndim === 0 ? reshape(a, [1, 1, 1]) : a.ndim === 1 ? reshape(a, [1, a.size, 1]) : a.ndim === 2 ? reshape(a, [...a.shape, 1]) : a)), 2)), 'move'),
    block: fn('block', (args) => wrap(blockOf(args[0], 0, depthOf(args[0])).copy()), 'move'),
    append: fn('append', (args, kw) => {
      const a = A(args[0], 'arr')
      const b = A(args[1], 'values')
      const axis = optInt(args, kw, 2, 'axis')
      if (axis === null) return wrap(concatenate([reshape(a.copy(), [-1]), reshape(b.copy(), [-1])], 0))
      return wrap(concatenate([a, b], axis))
    }, 'move'),
    insert: fn('insert', (args, kw) => {
      const a0 = A(args[0], 'arr')
      const axisV = optInt(args, kw, 3, 'axis')
      const a = axisV === null ? reshape(a0.copy(), [-1]) : a0
      const ax = axisV === null ? 0 : normAxis(axisV, a.ndim)
      const moved = moveAxes(a, [ax], [0])
      const unit = prod(moved.shape.slice(1))
      const n = moved.shape[0]
      const rowsOf = moved.values()
      const objV = args[1]
      const given = A(args[2], 'values')
      const single = isNum(objV)
      const positions = (objV.k === 'slice' ? (() => {
        const r = rangeOf({ ...objV, stop: objV.stop.k === 'none' ? int(n) : objV.stop })
        return r.vals
      })() : A(objV, 'obj').values()).map((p) => (p < 0 ? p + n + (single ? 0 : 0) : p))
      for (const p of positions) if (p < 0 || p > n) throw h.err('IndexError', `index ${p} is out of bounds for axis ${ax} with size ${n}`)
      const count = positions.length
      const fill = given.size === 1 ? new Array<number>(count * unit).fill(given.values()[0]) : broadcastTo(given.ndim === a.ndim && single ? given : given, [count, ...moved.shape.slice(1)].slice(unit === 1 ? 0 : 0)).values()
      const order = positions.map((p, i) => [p, i]).sort((x, y) => x[0] - y[0] || x[1] - y[1])
      const out: number[] = []
      let oi = 0
      for (let i = 0; i <= n; i++) {
        while (oi < count && order[oi][0] === i) {
          const k = order[oi][1]
          out.push(...fill.slice(k * unit, k * unit + unit))
          oi++
        }
        if (i < n) out.push(...rowsOf.slice(i * unit, i * unit + unit))
      }
      return wrap(moveAxes(NDArray.create(out, [n + count, ...moved.shape.slice(1)], a.dtype), [0], [ax]).copy())
    }, 'move'),
    delete: fn('delete', (args, kw) => {
      const a0 = A(args[0], 'arr')
      const axisV = optInt(args, kw, 2, 'axis')
      const a = axisV === null ? reshape(a0.copy(), [-1]) : a0
      const ax = axisV === null ? 0 : normAxis(axisV, a.ndim)
      const moved = moveAxes(a, [ax], [0])
      const unit = prod(moved.shape.slice(1))
      const n = moved.shape[0]
      const objV = args[1]
      let gone: number[]
      if (objV.k === 'slice') {
        const step = objV.step.k === 'none' ? 1 : h.toInt(objV.step)
        const start = objV.start.k === 'none' ? (step > 0 ? 0 : n - 1) : h.toInt(objV.start)
        const stop = objV.stop.k === 'none' ? (step > 0 ? n : -1) : h.toInt(objV.stop)
        gone = []
        for (let i = start < 0 ? start + n : start; step > 0 ? i < (stop < 0 ? stop + n : stop) : i > stop; i += step) gone.push(i)
      } else {
        const o = A(objV, 'obj')
        gone = o.dtype === 'bool' ? o.values().flatMap((x, i) => (x ? [i] : [])) : o.values().map((p) => {
          if (p < -n || p >= n) throw h.err('IndexError', `index ${p} is out of bounds for axis ${ax} with size ${n}`)
          return p < 0 ? p + n : p
        })
      }
      const drop = new Set(gone)
      const v = moved.values()
      const out: number[] = []
      for (let i = 0; i < n; i++) if (!drop.has(i)) out.push(...v.slice(i * unit, i * unit + unit))
      return wrap(moveAxes(NDArray.create(out, [n - drop.size, ...moved.shape.slice(1)], a.dtype), [0], [ax]).copy())
    }, 'move'),
    resize: fn('resize', (args) => {
      const a = A(args[0])
      const shape = shapeOf(kwOr(args, 1))
      const total = prod(shape)
      checkSize(shape)
      const v = a.values()
      return mk(Array.from({ length: total }, (_, i) => (v.length ? v[i % v.length] : 0)), shape, a.dtype)
    }, 'move'),
    roll: fn('roll', (args, kw) => {
      const a = A(args[0])
      const by = kw.shift ?? args[1]
      const axisV = opt(args, kw, 2, 'axis')
      if (!axisV) {
        if (!isNum(by)) throw h.err('NotImplementedError', 'np.roll with several shifts needs axis= in this sandbox')
        return mk(rollFlat(a.values(), h.toInt(by)), a.shape, a.dtype)
      }
      const axes = isNum(axisV) ? [h.toInt(axisV)] : h.iterate(axisV).map((x) => h.toInt(x))
      const shifts = isNum(by) ? axes.map(() => h.toInt(by)) : h.iterate(by).map((x) => h.toInt(x))
      return wrap(axes.reduce((cur, ax, i) => lanes(cur, ax, (lane) => rollFlat(lane, shifts[i]), a.dtype), a))
    }, 'move'),
    moveaxis: fn('moveaxis', (args) => {
      const a = A(args[0])
      const list = (v: Value) => (isNum(v) ? [h.toInt(v)] : h.iterate(v).map((x) => h.toInt(x)))
      return wrap(moveAxes(a, list(args[1]), list(args[2])))
    }, 'move'),
    rollaxis: fn('rollaxis', (args, kw) => {
      const a = A(args[0])
      const axis = normAxis(h.toInt(args[1]), a.ndim)
      let start = optInt(args, kw, 2, 'start') ?? 0
      if (start < 0) start += a.ndim
      if (axis < start) start--
      return wrap(moveAxes(a, [axis], [start]))
    }, 'move'),
    rot90: fn('rot90', (args, kw) => {
      const a = A(args[0], 'm')
      const k = (((optInt(args, kw, 1, 'k') ?? 1) % 4) + 4) % 4
      const axesV = kw.axes ?? args[2]
      const [p, q] = axesV ? h.iterate(axesV).map((x) => normAxis(h.toInt(x), a.ndim)) : [0, 1]
      const swap = (x: NDArray) => {
        const order = x.shape.map((_, i) => i)
        ;[order[p], order[q]] = [order[q], order[p]]
        return transpose(x, order)
      }
      if (k === 0) return wrap(a.copy())
      if (k === 2) return wrap(flip(flip(a, p), q).copy())
      return wrap((k === 1 ? swap(flip(a, q)) : flip(swap(a), q)).copy())
    }, 'move'),
    fliplr: fn('fliplr', (args) => {
      const a = A(args[0], 'm')
      if (a.ndim < 2) throw h.err('ValueError', 'Input must be >= 2-d.')
      return wrap(flip(a, 1))
    }, 'move'),
    flipud: fn('flipud', (args) => {
      const a = A(args[0], 'm')
      if (a.ndim < 1) throw h.err('ValueError', 'Input must be >= 1-d.')
      return wrap(flip(a, 0))
    }, 'move'),
    broadcast_to: fn('broadcast_to', (args, kw) => wrap(broadcastTo(A(args[0], 'array'), shapeOf(kw.shape ?? args[1]))), 'move'),
    broadcast_shapes: fn('broadcast_shapes', (args) => tuple(broadcastShapes(args.map(shapeOf)).map(int)), 'info'),
    broadcast_arrays: fn('broadcast_arrays', (args) => {
      const arrs = args.map((v) => toArray(v))
      const shape = broadcastShapes(arrs.map((a) => a.shape))
      return tuple(arrs.map((a) => wrap(broadcastTo(a, shape))))
    }, 'move'),
    unstack: fn('unstack', (args, kw) => {
      const a = A(args[0], 'x')
      const ax = normAxis(optInt(args, kw, 1, 'axis') ?? 0, a.ndim)
      const moved = moveAxes(a, [ax], [0])
      const rest = moved.shape.slice(1)
      const unit = prod(rest)
      const v = moved.values()
      return tuple(Array.from({ length: moved.shape[0] }, (_, i) => mk(v.slice(i * unit, i * unit + unit), rest, a.dtype)))
    }, 'move'),
    matrix_transpose: fn('matrix_transpose', (args) => {
      const a = A(args[0], 'x')
      if (a.ndim < 2) throw h.err('ValueError', `Input array must be at least 2-dimensional, but it is ${a.ndim}`)
      return wrap(moveAxes(a, [-1], [-2]))
    }, 'move'),
    permute_dims: fn('permute_dims', (args, kw) => {
      const given = kw.axes ?? args[1]
      return wrap(transpose(A(args[0]), given ? h.iterate(given).map((x) => h.toInt(x)) : undefined as unknown as number[]))
    }, 'move'),
    trim_zeros: fn('trim_zeros', (args, kw) => {
      const a = A(args[0], 'filt')
      const trim = (kw.trim ?? args[1])?.k === 'str' ? ((kw.trim ?? args[1]) as { v: string }).v.toLowerCase() : 'fb'
      const v = a.values()
      let from = 0
      let to = v.length
      if (trim.includes('f')) while (from < to && v[from] === 0) from++
      if (trim.includes('b')) while (to > from && v[to - 1] === 0) to--
      return mk(v.slice(from, to), [to - from], a.dtype)
    }, 'move'),
  })
  function kwOr(args: Value[], i: number): Value {
    if (!args[i]) throw h.err('TypeError', 'missing required argument')
    return args[i]
  }

  // ---------- picking and writing elements ----------

  const takeAxis = (a: NDArray, idx: NDArray, axis: number | null, mode: string): Value => {
    const src = axis === null ? reshape(a.copy(), [-1]) : a
    const ax = axis === null ? 0 : normAxis(axis, src.ndim)
    const v = src.values()
    const n = src.shape[ax]
    const before = prod(src.shape.slice(0, ax))
    const after = prod(src.shape.slice(ax + 1))
    const fit = (i: number) => {
      if (mode === 'wrap') return ((i % n) + n) % n
      if (mode === 'clip') return Math.min(n - 1, Math.max(0, i))
      if (i < -n || i >= n) throw h.err('IndexError', `index ${i} is out of bounds for axis ${ax} with size ${n}`)
      return i < 0 ? i + n : i
    }
    const ids = idx.values().map(fit)
    const out: number[] = []
    for (let b = 0; b < before; b++) for (const i of ids) for (let c = 0; c < after; c++) out.push(v[(b * n + i) * after + c])
    const shape = [...src.shape.slice(0, ax), ...idx.shape, ...src.shape.slice(ax + 1)]
    if (!shape.length) return scalar(out[0], a.dtype)
    return mk(out, shape, a.dtype)
  }
  /** position of each element of indices in the array it is taken along */
  const alongPositions = (a: NDArray, idx: NDArray, ax: number) => {
    const shape = idx.shape.map((d, i) => (i === ax ? d : Math.max(d, a.shape[i])))
    const st = cStrides(a.shape)
    const ids = broadcastTo(idx, shape).values()
    return ids.map((i, p) => {
      const at = unravel(p, shape)
      let flat = 0
      for (let k = 0; k < a.ndim; k++) flat += (k === ax ? (i < 0 ? i + a.shape[ax] : i) : at[k]) * st[k]
      return flat
    })
  }
  const write = (a: NDArray, values: number[]) => {
    const ad = a.addresses()
    values.forEach((x, i) => { a.data[ad[i]] = castValue(x, a.dtype) })
  }
  const nonzeroFlat = (a: NDArray) => a.values().flatMap((x, i) => (x ? [i] : []))
  Object.assign(np, {
    take: fn('take', (args, kw) => {
      const mode = kw.mode?.k === 'str' ? kw.mode.v : 'raise'
      return takeAxis(A(args[0]), A(kw.indices ?? args[1], 'indices'), optInt(args, kw, 2, 'axis'), mode)
    }, 'move'),
    take_along_axis: fn('take_along_axis', (args, kw) => {
      const a = A(args[0], 'arr')
      const idx = A(kw.indices ?? args[1], 'indices')
      const v = a.values()
      const ax = normAxis(h.toInt(kw.axis ?? args[2]), a.ndim)
      const shape = idx.shape.map((d, i) => (i === ax ? d : Math.max(d, a.shape[i])))
      return mk(alongPositions(a, idx, ax).map((p) => v[p]), shape, a.dtype)
    }, 'move'),
    put_along_axis: fn('put_along_axis', (args, kw) => {
      const a = A(args[0], 'arr')
      const idx = A(kw.indices ?? args[1], 'indices')
      const ax = normAxis(h.toInt(kw.axis ?? args[3]), a.ndim)
      const pos = alongPositions(a, idx, ax)
      const shape = idx.shape.map((d, i) => (i === ax ? d : Math.max(d, a.shape[i])))
      const vals = broadcastTo(A(kw.values ?? args[2], 'values'), shape).values()
      const ad = a.addresses()
      pos.forEach((p, i) => { a.data[ad[p]] = castValue(vals[i], a.dtype) })
      return NONE
    }, 'move'),
    put: fn('put', (args) => {
      putInto(A(args[0]), A(args[1], 'ind'), A(args[2], 'v'))
      return NONE
    }, 'move'),
    putmask: fn('putmask', (args, kw) => {
      const a = A(args[0])
      const mask = broadcastTo(A(kw.mask ?? args[1], 'mask'), a.shape).values()
      const vals = A(kw.values ?? args[2], 'values').values()
      write(a, a.values().map((x, i) => (mask[i] ? vals[i % vals.length] : x)))
      return NONE
    }, 'move'),
    place: fn('place', (args, kw) => {
      const a = A(args[0], 'arr')
      const mask = A(kw.mask ?? args[1], 'mask').values()
      const vals = A(kw.vals ?? args[2], 'vals').values()
      let k = 0
      write(a, a.values().map((x, i) => (mask[i] ? vals[k++ % vals.length] : x)))
      return NONE
    }, 'move'),
    copyto: fn('copyto', (args, kw) => {
      const dst = A(args[0], 'dst')
      const src = broadcastTo(A(args[1], 'src'), dst.shape).values()
      const where = kw.where ? broadcastTo(toArray(kw.where), dst.shape).values() : null
      write(dst, dst.values().map((x, i) => (!where || where[i] ? src[i] : x)))
      return NONE
    }, 'move'),
    fill_diagonal: fn('fill_diagonal', (args) => {
      const a = A(args[0])
      const vals = A(args[1], 'val').values()
      if (a.ndim < 2) throw h.err('ValueError', 'array must be at least 2-d')
      if (a.ndim > 2 && new Set(a.shape).size > 1) throw h.err('ValueError', 'All dimensions of input must be of equal length')
      const n = Math.min(...a.shape)
      const st = cStrides(a.shape)
      const step = st.reduce((s, x) => s + x, 0)
      // a tall matrix wraps nothing here: the diagonal stops at the shorter side
      const v = a.values()
      for (let i = 0; i < n; i++) v[i * step] = vals[i % vals.length]
      write(a, v)
      return NONE
    }, 'move'),
    choose: fn('choose', (args) => {
      const pick = A(args[0], 'a')
      const choices = h.iterate(args[1]).map((x) => toArray(x))
      const shape = broadcastShapes([pick.shape, ...choices.map((c) => c.shape)])
      const ids = broadcastTo(pick, shape).values()
      const cols = choices.map((c) => broadcastTo(c, shape).values())
      return mk(ids.map((c, i) => {
        if (c < 0 || c >= cols.length) throw h.err('ValueError', 'invalid entry in choice array')
        return cols[c][i]
      }), shape, commonDType(choices))
    }, 'move'),
    compress: fn('compress', (args, kw) => {
      const cond = nonzeroFlat(A(args[0], 'condition'))
      return takeAxis(A(args[1]), NDArray.create(cond, [cond.length], 'int64'), optInt(args, kw, 2, 'axis'), 'raise')
    }, 'move'),
    extract: fn('extract', (args) => {
      const a = A(args[1], 'arr')
      const v = a.values()
      const at = nonzeroFlat(A(args[0], 'condition'))
      return mk(at.map((i) => v[i]), [at.length], a.dtype)
    }, 'move'),
    select: fn('select', (args, kw) => {
      const conds = h.iterate(args[0]).map((x) => toArray(x))
      const choices = h.iterate(args[1]).map((x) => toArray(x))
      if (conds.length !== choices.length) throw h.err('ValueError', 'list of cases must be same length as list of conditions')
      const dflt = toArray(opt(args, kw, 2, 'default') ?? int(0))
      const shape = broadcastShapes([...conds.map((c) => c.shape), ...choices.map((c) => c.shape), dflt.shape])
      const cv = conds.map((c) => broadcastTo(c, shape).values())
      const xv = choices.map((c) => broadcastTo(c, shape).values())
      const dv = broadcastTo(dflt, shape).values()
      return mk(dv.map((d, i) => {
        const k = cv.findIndex((c) => c[i])
        return k < 0 ? d : xv[k][i]
      }), shape, commonDType([...choices, dflt]))
    }, 'move'),
    piecewise: fn('piecewise', (args, kw) => {
      const x = A(args[0], 'x')
      const conds = h.iterate(args[1]).map((c) => broadcastTo(toArray(c), x.shape).values())
      const funcs = h.iterate(args[2])
      const v = x.values()
      const out = new Array<number>(v.length).fill(0)
      const rest = args.slice(3)
      // one extra function is the default for everything no condition picked
      const claimed = v.map((_, i) => conds.some((c) => c[i]))
      const apply = (f: Value, chosen: number[]) => {
        if (!chosen.length) return
        const res = f.k === 'func' || f.k === 'fn' || f.k === 'bound' ? toArray(h.call(f, [mk(chosen.map((i) => v[i]), [chosen.length], x.dtype), ...rest], kw)).values() : [h.num(f)]
        chosen.forEach((i, k) => { out[i] = res.length === 1 ? res[0] : res[k] })
      }
      conds.forEach((c, k) => apply(funcs[k], c.flatMap((on, i) => (on ? [i] : []))))
      if (funcs.length > conds.length) apply(funcs[conds.length], claimed.flatMap((on, i) => (on ? [] : [i])))
      return mk(out, x.shape, x.dtype)
    }, 'move'),
    argwhere: fn('argwhere', (args) => {
      const a = A(args[0])
      const nz = nonzeroFlat(a)
      if (a.ndim === 0) return mk([], [nz.length, 0], 'int64')
      return mk(nz.flatMap((p) => unravel(p, a.shape)), [nz.length, a.ndim], 'int64')
    }, 'scan'),
    flatnonzero: fn('flatnonzero', (args) => {
      const nz = nonzeroFlat(A(args[0]))
      return mk(nz, [nz.length], 'int64')
    }, 'scan'),
    unravel_index: fn('unravel_index', (args, kw) => {
      const shape = shapeOf(kw.shape ?? args[1])
      const idx = A(kw.indices ?? args[0], 'indices')
      const total = prod(shape)
      for (const i of idx.values()) if (i < 0 || i >= total) throw h.err('ValueError', `index ${i} is out of bounds for array with size ${total}`)
      if (isNum(kw.indices ?? args[0])) return tuple(unravel(idx.values()[0], shape).map(int))
      const parts = idx.values().map((i) => unravel(i, shape))
      return tuple(shape.map((_, k) => mk(parts.map((p) => p[k]), idx.shape, 'int64')))
    }, 'info'),
    ravel_multi_index: fn('ravel_multi_index', (args, kw) => {
      const multi = h.iterate(args[0]).map((x) => toArray(x))
      const shape = shapeOf(kw.dims ?? args[1])
      const st = cStrides(shape)
      const common = broadcastShapes(multi.map((m) => m.shape))
      const cols = multi.map((m) => broadcastTo(m, common).values())
      const out = Array.from({ length: prod(common) }, (_, i) => cols.reduce((t, c, k) => {
        if (c[i] < 0 || c[i] >= shape[k]) throw h.err('ValueError', 'invalid entry in coordinates array')
        return t + c[i] * st[k]
      }, 0))
      return common.length ? mk(out, common, 'int64') : int(out[0])
    }, 'info'),
    ix_: fn('ix_', (args) => {
      const n = args.length
      return tuple(args.map((v, k) => {
        const a = toArray(v)
        const vals = a.dtype === 'bool' ? nonzeroFlat(a) : a.values()
        return mk(vals, Array.from({ length: n }, (_, j) => (j === k ? vals.length : 1)), a.dtype === 'bool' ? 'int64' : a.dtype)
      }))
    }, 'create'),
    ndenumerate: fn('ndenumerate', (args) => {
      const a = A(args[0], 'arr')
      const v = a.values()
      let i = 0
      return { k: 'iter', name: 'ndenumerate', next: () => {
        if (i >= v.length) return undefined
        const at = tuple(unravel(i, a.shape).map(int))
        return tuple([at, scalar(v[i++], a.dtype)])
      } }
    }),
    ndindex: fn('ndindex', (args) => {
      const shape = args.length === 1 && !isNum(args[0]) ? shapeOf(args[0]) : args.map((x) => h.toInt(x))
      const total = prod(shape)
      let i = 0
      return { k: 'iter', name: 'ndindex', next: () => (i < total ? tuple(unravel(i++, shape).map(int)) : undefined) }
    }),
  })
  function putInto(a: NDArray, ind: NDArray, vals: NDArray) {
    const n = a.size
    const v = vals.values()
    const ad = a.addresses()
    ind.values().forEach((i, k) => {
      if (i < -n || i >= n) throw h.err('IndexError', `index ${i} is out of bounds for axis 0 with size ${n}`)
      a.data[ad[i < 0 ? i + n : i]] = castValue(v[k % v.length], a.dtype)
    })
  }

  // ---------- sorting and searching ----------

  const searchIn = (sorted: number[], x: number, right: boolean) => {
    let lo = 0
    let hi = sorted.length
    while (lo < hi) {
      const m = (lo + hi) >> 1
      if (right ? sorted[m] <= x : sorted[m] < x) lo = m + 1
      else hi = m
    }
    return lo
  }
  const ascending = (x: number, y: number) => (Number.isNaN(x) ? (Number.isNaN(y) ? 0 : 1) : Number.isNaN(y) ? -1 : x - y)
  const uniqSorted = (v: number[]) => [...new Set(v)].sort(ascending)
  Object.assign(np, {
    searchsorted: fn('searchsorted', (args, kw) => {
      const a = A(args[0]).values()
      const side = (kw.side ?? args[2]) as { v?: string } | undefined
      const right = side?.v === 'right'
      const sorter = opt(args, kw, 3, 'sorter')
      const base = sorter ? A(sorter).values().map((i) => a[i]) : a
      const x = A(kw.v ?? args[1], 'v')
      const res = x.values().map((y) => searchIn(base, y, right))
      return isNum(kw.v ?? args[1]) ? int(res[0]) : mk(res, x.shape, 'int64')
    }, 'scan'),
    digitize: fn('digitize', (args, kw) => {
      const x = A(args[0], 'x')
      const bins = A(args[1], 'bins').values()
      const right = (kw.right ?? args[2]) && h.truthy(kw.right ?? args[2])
      const rising = bins.length < 2 || bins[1] >= bins[0]
      const sorted = rising ? bins : [...bins].reverse()
      const res = x.values().map((v) => {
        const at = searchIn(sorted, v, !right)
        return rising ? at : bins.length - at
      })
      return isNum(args[0]) ? int(res[0]) : mk(res, x.shape, 'int64')
    }, 'scan'),
    // a full sort is one valid partition: NumPy's own arrangement of the two sides differs and is not promised
    partition: fn('partition', (args, kw) => wrap(sortAlong(A(args[0]), optInt(args, kw, 2, 'axis') ?? -1).sorted), 'sort'),
    argpartition: fn('argpartition', (args, kw) => wrap(sortAlong(A(args[0]), optInt(args, kw, 2, 'axis') ?? -1).order), 'sort'),
    lexsort: fn('lexsort', (args, kw) => {
      const keys = h.iterate(args[0]).map((k) => toArray(k).values())
      void kw
      const n = keys[0]?.length ?? 0
      const order = Array.from({ length: n }, (_, i) => i).sort((x, y) => {
        for (let k = keys.length - 1; k >= 0; k--) {
          const c = ascending(keys[k][x], keys[k][y])
          if (c) return c
        }
        return x - y
      })
      return mk(order, [n], 'int64')
    }, 'sort'),
    unique: fn('unique', (args, kw) => {
      const a = A(args[0], 'ar')
      const want = (name: string, i: number) => {
        const v = kw[name] ?? args[i]
        return !!v && h.truthy(v)
      }
      const [wantIndex, wantInverse, wantCounts] = [want('return_index', 1), want('return_inverse', 2), want('return_counts', 3)]
      const axisV = opt(args, kw, 4, 'axis')
      const flat = !axisV ? NDArray.create(a.values(), [a.size], a.dtype) : moveAxes(a, [h.toInt(axisV)], [0])
      const n = flat.shape[0]
      const unit = n ? flat.size / n : 1
      const v = flat.values()
      const row = (i: number) => v.slice(i * unit, i * unit + unit)
      const compareRows = (x: number, y: number) => {
        for (let k = 0; k < unit; k++) {
          const c = ascending(v[x * unit + k], v[y * unit + k])
          if (c) return c
        }
        return 0
      }
      const order = Array.from({ length: n }, (_, i) => i).sort((x, y) => compareRows(x, y) || x - y)
      const firsts: number[] = []
      const counts: number[] = []
      const inverse = new Array<number>(n)
      order.forEach((i, k) => {
        const same = k > 0 && compareRows(order[k - 1], i) === 0
        if (!same) {
          firsts.push(i)
          counts.push(0)
        }
        counts[counts.length - 1]++
        inverse[i] = firsts.length - 1
      })
      const uniqueRows = NDArray.create(firsts.flatMap(row), [firsts.length, ...flat.shape.slice(1)], a.dtype)
      const values = wrap(axisV ? moveAxes(uniqueRows, [0], [h.toInt(axisV)]).copy() : uniqueRows)
      if (!wantIndex && !wantInverse && !wantCounts) return values
      const out = [values]
      if (wantIndex) out.push(mk(firsts, [firsts.length], 'int64'))
      if (wantInverse) out.push(mk(inverse, axisV ? [n] : a.shape, 'int64'))
      if (wantCounts) out.push(mk(counts, [counts.length], 'int64'))
      return tuple(out)
    }, 'unique'),
    union1d: fn('union1d', (args) => {
      const [a, b] = [A(args[0], 'ar1'), A(args[1], 'ar2')]
      const u = uniqSorted([...a.values(), ...b.values()])
      return mk(u, [u.length], commonDType([a, b]))
    }, 'unique'),
    intersect1d: fn('intersect1d', (args, kw) => {
      const [a, b] = [A(args[0], 'ar1'), A(args[1], 'ar2')]
      const inB = new Set(b.values())
      const u = uniqSorted(a.values().filter((x) => inB.has(x)))
      const res = mk(u, [u.length], commonDType([a, b]))
      if (!flag(kw, 'return_indices')) return res
      return tuple([res, mk(u.map((x) => a.values().indexOf(x)), [u.length], 'int64'), mk(u.map((x) => b.values().indexOf(x)), [u.length], 'int64')])
    }, 'unique'),
    setdiff1d: fn('setdiff1d', (args) => {
      const a = A(args[0], 'ar1')
      const inB = new Set(A(args[1], 'ar2').values())
      const u = uniqSorted(a.values().filter((x) => !inB.has(x)))
      return mk(u, [u.length], a.dtype)
    }, 'unique'),
    setxor1d: fn('setxor1d', (args) => {
      const [a, b] = [A(args[0], 'ar1'), A(args[1], 'ar2')]
      const [x, y] = [new Set(a.values()), new Set(b.values())]
      const u = uniqSorted([...[...x].filter((v) => !y.has(v)), ...[...y].filter((v) => !x.has(v))])
      return mk(u, [u.length], commonDType([a, b]))
    }, 'unique'),
    in1d: fn('in1d', (args, kw) => {
      const set = new Set(A(args[1], 'ar2').values())
      const inv = flag(kw, 'invert')
      const a = A(args[0], 'ar1').values()
      return mk(a.map((x) => (set.has(x) !== inv ? 1 : 0)), [a.length], 'bool')
    }, 'unique'),
    unique_values: fn('unique_values', (args) => {
      const a = A(args[0], 'x')
      const u = uniqSorted(a.values())
      return mk(u, [u.length], a.dtype)
    }, 'unique'),
    unique_counts: fn('unique_counts', (args) => {
      const a = A(args[0], 'x')
      const u = uniqSorted(a.values())
      return named('UniqueCountsResult', [['values', mk(u, [u.length], a.dtype)], ['counts', mk(u.map((x) => a.values().filter((y) => y === x).length), [u.length], 'int64')]])
    }, 'unique'),
    unique_inverse: fn('unique_inverse', (args) => {
      const a = A(args[0], 'x')
      const u = uniqSorted(a.values())
      return named('UniqueInverseResult', [['values', mk(u, [u.length], a.dtype)], ['inverse_indices', mk(a.values().map((x) => u.indexOf(x)), a.shape, 'int64')]])
    }, 'unique'),
    unique_all: fn('unique_all', (args) => {
      const a = A(args[0], 'x')
      const v = a.values()
      const u = uniqSorted(v)
      return named('UniqueAllResult', [
        ['values', mk(u, [u.length], a.dtype)],
        ['indices', mk(u.map((x) => v.indexOf(x)), [u.length], 'int64')],
        ['inverse_indices', mk(v.map((x) => u.indexOf(x)), a.shape, 'int64')],
        ['counts', mk(u.map((x) => v.filter((y) => y === x).length), [u.length], 'int64')],
      ])
    }, 'unique'),
  })

  // ---------- statistics and calculus ----------

  const nanSafe = (xs: number[]) => xs.filter((x) => !Number.isNaN(x))
  /** np.nanpercentile and np.nanquantile: one q or several, over everything or along an axis */
  const nanQuantiles = (args: Value[], kw: Kw, scale: number): Value => {
    const a = A(args[0])
    const qv = kw.q ?? args[1]
    const axisV = kw.axis ?? args[2]
    const axis = axisV && axisV.k !== 'none' ? normAxis(h.toInt(axisV), a.ndim) : null
    const one = (q: number): Value => {
      if (axis === null) return float(quantileOf(nanSafe(a.values()), q))
      const rest = a.shape.filter((_, d) => d !== axis)
      const after = a.shape.slice(axis + 1).reduce((p, x) => p * x, 1)
      const groups = new Map<number, number[]>()
      a.values().forEach((v, i) => {
        const key = Math.floor(i / (after * a.shape[axis])) * after + (i % after)
        const g = groups.get(key)
        if (g) g.push(v)
        else groups.set(key, [v])
      })
      return wrap(NDArray.create([...groups.keys()].sort((p, q2) => p - q2).map((k) => quantileOf(nanSafe(groups.get(k)!), q)), rest, 'float64'))
    }
    if (qv && (qv.k === 'list' || qv.k === 'tuple' || qv.k === 'array')) {
      const parts = Array.from(toArray(qv, 'float64').values(), (q) => one(q / scale))
      return axis === null ? wrap(NDArray.create(parts.map((p) => (p as { v: number }).v), [parts.length], 'float64')) : h.call(np.stack, [py.list(parts)])
    }
    return one(h.num(qv) / scale)
  }
  const toFloat = (a: NDArray): DType => (a.dtype === 'float32' ? 'float32' : 'float64')
  const sameOr = (a: NDArray) => (a.dtype === 'bool' ? 'int64' : a.dtype) as DType
  const meanOf = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / xs.length
  const varOf = (xs: number[], ddof: number) => {
    const m = meanOf(xs)
    return xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - ddof)
  }
  const medianOf = (xs: number[]) => {
    if (!xs.length) return NaN
    const s = [...xs].sort(ascending)
    const n = s.length
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2
  }
  const quantileOf = (xs: number[], q: number) => {
    if (!xs.length) return NaN
    const s = [...xs].sort(ascending)
    const at = q * (s.length - 1)
    const lo = Math.floor(at)
    const hi = Math.min(s.length - 1, lo + 1)
    return s[lo] + (s[hi] - s[lo]) * (at - lo)
  }
  const ddofOf = (kw: Kw) => (kw.ddof ? h.num(kw.ddof) : 0)
  const argBest = (xs: number[], better: (a: number, b: number) => boolean, name: string) => {
    let at = -1
    xs.forEach((x, i) => {
      if (!Number.isNaN(x) && (at < 0 || better(x, xs[at]))) at = i
    })
    if (at < 0) throw h.err('ValueError', `All-NaN slice encountered in ${name}`)
    return at
  }
  const convolve = (a: number[], b: number[]) => {
    const out = new Array<number>(a.length + b.length - 1).fill(0)
    a.forEach((x, i) => b.forEach((y, j) => { out[i + j] += x * y }))
    return out
  }
  const convolved = (_name: string, a: NDArray, b: NDArray, mode: string, reverse: boolean) => {
    if (!a.size || !b.size) throw h.err('ValueError', `${reverse ? 'v' : 'a'} cannot be empty`)
    if (a.ndim > 1 || b.ndim > 1) throw h.err('ValueError', 'object too deep for desired array')
    const av = a.values()
    const bv = reverse ? [...b.values()].reverse() : b.values()
    const full = convolve(av, bv)
    const [n, m] = [Math.max(av.length, bv.length), Math.min(av.length, bv.length)]
    const d = commonDType([a, b]) === 'bool' ? 'bool' : commonDType([a, b])
    if (mode === 'full') return mk(full, [full.length], d)
    if (mode === 'same') {
      const at = Math.floor((m - 1) / 2)
      return mk(full.slice(at, at + n), [n], d)
    }
    if (mode !== 'valid') throw h.err('ValueError', `mode must be one of 'valid', 'same', or 'full' (got '${mode}')`)
    return mk(full.slice(m - 1, n), [n - m + 1], d)
  }
  const covMatrix = (args: Value[], kw: Kw): { m: number[][]; vars: number; obs: number } => {
    const rowvar = !(kw.rowvar ?? args[2]) || h.truthy(kw.rowvar ?? args[2])
    const bias = (kw.bias ?? args[3]) && h.truthy(kw.bias ?? args[3])
    const x = A(args[0], 'm')
    const y = opt(args, kw, 1, 'y')
    const asRows = (a: NDArray): number[][] => {
      if (a.ndim <= 1) return [a.values()]
      const r = rows(a)
      return rowvar ? r : transposeM(r)
    }
    const data = [...asRows(x), ...(y ? asRows(toArray(y)) : [])]
    const obs = data[0]?.length ?? 0
    const dv = kw.ddof && kw.ddof.k !== 'none' ? h.num(kw.ddof) : bias ? 0 : 1
    const means = data.map(meanOf)
    const m = data.map((u, i) => data.map((w, j) => u.reduce((s, x0, k) => s + (x0 - means[i]) * (w[k] - means[j]), 0) / (obs - dv)))
    return { m, vars: data.length, obs }
  }
  const gradient1 = (y: number[], spacing: number | number[], edge: number): number[] => {
    const n = y.length
    if (n < edge + 1) throw h.err('ValueError', 'Shape of array too small to calculate a numerical gradient, at least (edge_order + 1) elements are required.')
    const dx = (i: number) => (typeof spacing === 'number' ? spacing : spacing[i + 1] - spacing[i])
    const g = new Array<number>(n)
    for (let i = 1; i < n - 1; i++) {
      const [a, b] = [dx(i - 1), dx(i)]
      g[i] = (a * a * y[i + 1] - b * b * y[i - 1] + (b * b - a * a) * y[i]) / (a * b * (a + b))
    }
    if (edge === 1) {
      g[0] = (y[1] - y[0]) / dx(0)
      g[n - 1] = (y[n - 1] - y[n - 2]) / dx(n - 2)
    } else {
      const [a, b] = [dx(0), dx(1)]
      g[0] = (-(2 * a + b) / (a * (a + b))) * y[0] + ((a + b) / (a * b)) * y[1] - (a / (b * (a + b))) * y[2]
      const [c, d] = [dx(n - 3), dx(n - 2)]
      g[n - 1] = (d / (c * (c + d))) * y[n - 3] - ((c + d) / (c * d)) * y[n - 2] + ((c + 2 * d) / (d * (c + d))) * y[n - 1]
    }
    return g
  }
  const trapezoidOf = (y: number[], x: number[] | null, dx: number) => {
    let s = 0
    for (let i = 1; i < y.length; i++) s += ((x ? x[i] - x[i - 1] : dx) * (y[i] + y[i - 1])) / 2
    return s
  }
  const trapezoid = (name: string) => fn(name, (args, kw) => {
    const y = A(args[0], 'y')
    const x = opt(args, kw, 1, 'x')
    const dx = optNum(args, kw, 2, 'dx', 1)
    const axis = optInt(args, kw, 3, 'axis') ?? -1
    const xs = x ? toArray(x).values() : null
    if (y.ndim === 1) return float(trapezoidOf(y.values(), xs, dx))
    const res = lanes(y, axis, (lane) => [trapezoidOf(lane, xs, dx)], 'float64')
    return wrap(reshape(res, y.shape.filter((_, i) => i !== normAxis(axis, y.ndim))))
  }, 'reduce')
  const scan = (name: string, f: (acc: number, x: number) => number, first: (x: number) => number, skipNan: boolean) => fn(name, (args, kw) => {
    const a = A(args[0])
    const axis = optInt(args, kw, 1, 'axis')
    const run = (lane: number[]) => {
      const out: number[] = []
      let acc: number | null = null
      for (const x0 of lane) {
        const x = skipNan && Number.isNaN(x0) ? first(NaN) : x0
        acc = acc === null ? x : f(acc, x)
        out.push(acc)
      }
      return out
    }
    const d = sameOr(a)
    return axis === null ? mk(run(a.values()), [a.size], d) : wrap(lanes(a, axis, run, d))
  }, 'scan')
  const hist2 = (args: Value[], kw: Kw) => {
    const x = A(args[0], 'x').values()
    const y = A(args[1], 'y').values()
    const bins = kw.bins ?? args[2] ?? int(10)
    const edgesOf = (v: number[], b: Value) => {
      if (!isNum(b)) return toArray(b).values()
      const n = h.toInt(b)
      let lo = Math.min(...v)
      let hi = Math.max(...v)
      if (lo === hi) {
        lo -= 0.5
        hi += 0.5
      }
      return lineFrom(lo, hi, n + 1, true)
    }
    const pair = !isNum(bins) && h.iterate(bins).length === 2
    const [bx, by] = pair ? h.iterate(bins) : [bins, bins]
    const ex = edgesOf(x, bx)
    const ey = edgesOf(y, by)
    const cell = (e: number[], v: number) => {
      if (v < e[0] || v > e[e.length - 1]) return -1
      if (v === e[e.length - 1]) return e.length - 2
      return searchIn(e, v, true) - 1
    }
    const out = new Array<number>((ex.length - 1) * (ey.length - 1)).fill(0)
    x.forEach((v, i) => {
      const [a, b] = [cell(ex, v), cell(ey, y[i])]
      if (a >= 0 && b >= 0) out[a * (ey.length - 1) + b] += 1
    })
    return tuple([mk(out, [ex.length - 1, ey.length - 1]), mk(ex, [ex.length]), mk(ey, [ey.length])])
  }
  Object.assign(np, {
    ptp: reduceWith('ptp', (xs) => Math.max(...xs) - Math.min(...xs), sameOr),
    nansum: reduceWith('nansum', (xs) => nanSafe(xs).reduce((s, x) => s + x, 0), sameOr),
    nanprod: reduceWith('nanprod', (xs) => nanSafe(xs).reduce((s, x) => s * x, 1), sameOr),
    nanmean: reduceWith('nanmean', (xs) => meanOf(nanSafe(xs)), toFloat),
    nanmax: reduceWith('nanmax', (xs) => (nanSafe(xs).length ? nanSafe(xs).reduce((m, x) => Math.max(m, x)) : NaN), sameOr),
    nanmin: reduceWith('nanmin', (xs) => (nanSafe(xs).length ? nanSafe(xs).reduce((m, x) => Math.min(m, x)) : NaN), sameOr),
    nanstd: reduceWith('nanstd', (xs, kw) => Math.sqrt(varOf(nanSafe(xs), ddofOf(kw))), toFloat),
    nanvar: reduceWith('nanvar', (xs, kw) => varOf(nanSafe(xs), ddofOf(kw)), toFloat),
    nanmedian: reduceWith('nanmedian', (xs) => medianOf(nanSafe(xs)), toFloat),
    nanargmax: reduceWith('nanargmax', (xs) => argBest(xs, (a, b) => a > b, 'nanargmax'), () => 'int64'),
    nanargmin: reduceWith('nanargmin', (xs) => argBest(xs, (a, b) => a < b, 'nanargmin'), () => 'int64'),
    nanpercentile: fn('nanpercentile', (args, kw) => nanQuantiles(args, kw, 100), 'reduce'),
    nanquantile: fn('nanquantile', (args, kw) => nanQuantiles(args, kw, 1), 'reduce'),
    cumprod: scan('cumprod', (acc, x) => acc * x, (x) => x, false), cumulative_prod: scan('cumulative_prod', (acc, x) => acc * x, (x) => x, false),
    cumulative_sum: scan('cumulative_sum', (acc, x) => acc + x, (x) => x, false),
    nancumsum: scan('nancumsum', (acc, x) => acc + x, () => 0, true), nancumprod: scan('nancumprod', (acc, x) => acc * x, () => 1, true),
    ediff1d: fn('ediff1d', (args) => {
      const a = A(args[0], 'ary')
      const v = a.values()
      return mk(v.slice(1).map((x, i) => x - v[i]), [Math.max(0, v.length - 1)], a.dtype === 'bool' ? 'int64' : a.dtype)
    }, 'scan'),
    gradient: fn('gradient', (args, kw) => {
      const f = A(args[0], 'f')
      const edge = kw.edge_order ? h.toInt(kw.edge_order) : 1
      const axisV = kw.axis && kw.axis.k !== 'none' ? kw.axis : null
      const axes = axisV ? (isNum(axisV) ? [h.toInt(axisV)] : h.iterate(axisV).map((x) => h.toInt(x))) : f.shape.map((_, i) => i)
      const steps = args.slice(1)
      const spacing = (k: number): number | number[] => {
        const s = steps.length === 1 ? steps[0] : steps[k]
        return !s ? 1 : isNum(s) ? h.num(s) : toArray(s).values()
      }
      const parts = axes.map((ax, k) => wrap(lanes(f, ax, (lane) => gradient1(lane, spacing(k), edge), 'float64')))
      return parts.length === 1 ? parts[0] : tuple(parts)
    }, 'scan'),
    interp: fn('interp', (args, kw) => {
      const x = A(args[0], 'x')
      const xp = A(args[1], 'xp').values()
      const fp = A(args[2], 'fp').values()
      if (xp.length !== fp.length) throw h.err('ValueError', 'fp and xp are not of the same length.')
      if (!xp.length) throw h.err('ValueError', 'array of sample points is empty')
      const left = opt(args, kw, 3, 'left')
      const right = opt(args, kw, 4, 'right')
      const res = x.values().map((v) => {
        if (Number.isNaN(v)) return NaN
        if (v < xp[0]) return left ? h.num(left) : fp[0]
        if (v > xp[xp.length - 1]) return right ? h.num(right) : fp[fp.length - 1]
        const i = Math.min(xp.length - 1, searchIn(xp, v, true))
        if (xp[i - 1] === v || i === 0) return fp[Math.max(0, i - 1)]
        return fp[i - 1] + ((fp[i] - fp[i - 1]) * (v - xp[i - 1])) / (xp[i] - xp[i - 1])
      })
      return isNum(args[0]) ? float(res[0]) : mk(res, x.shape)
    }, 'scan'),
    trapezoid: trapezoid('trapezoid'), trapz: trapezoid('trapz'),
    convolve: fn('convolve', (args, kw) => convolved('convolve', A(args[0]), A(args[1], 'v'), (opt(args, kw, 2, 'mode') as { v?: string } | undefined)?.v ?? 'full', false), 'matmul'),
    correlate: fn('correlate', (args, kw) => convolved('correlate', A(args[0]), A(args[1], 'v'), (opt(args, kw, 2, 'mode') as { v?: string } | undefined)?.v ?? 'valid', true), 'matmul'),
    cov: fn('cov', (args, kw) => {
      const { m, vars } = covMatrix(args, kw)
      return vars === 1 ? wrap(NDArray.create([m[0][0]], [], 'float64')) : matrix(m)
    }, 'linalg'),
    corrcoef: fn('corrcoef', (args, kw) => {
      const { m, vars } = covMatrix(args, kw)
      if (vars === 1) return wrap(NDArray.create([m[0][0] / m[0][0]], [], 'float64'))
      return matrix(m.map((row, i) => row.map((x, j) => Math.max(-1, Math.min(1, x / Math.sqrt(m[i][i] * m[j][j]))))))
    }, 'linalg'),
    histogram2d: fn('histogram2d', hist2, 'group'),
    histogram_bin_edges: fn('histogram_bin_edges', (args, kw) => {
      const v = A(args[0]).values()
      const bins = opt(args, kw, 1, 'bins') ?? int(10)
      if (!isNum(bins)) return wrap(toArray(bins))
      let [lo, hi] = v.length ? [Math.min(...v), Math.max(...v)] : [0, 1]
      if (lo === hi) {
        lo -= 0.5
        hi += 0.5
      }
      const n = h.toInt(bins)
      return mk(lineFrom(lo, hi, n + 1, true), [n + 1])
    }, 'group'),
  })

  // ---------- products ----------

  /** np.einsum for explicit or implicit subscripts without an ellipsis: a nested loop over the labels */
  const einsum = (spec: string, ops: NDArray[]): Value => {
    if (spec.includes('.')) throw h.err('NotImplementedError', 'np.einsum with ... is not available in this sandbox: name the axes')
    const clean = spec.replace(/\s+/g, '')
    const [lhs, given] = clean.split('->')
    const terms = lhs.split(',')
    if (terms.length !== ops.length) throw h.err('ValueError', 'more operands provided to einstein sum function than specified in the subscripts string')
    const sizes = new Map<string, number>()
    const counts = new Map<string, number>()
    terms.forEach((t, k) => {
      if (t.length !== ops[k].ndim) throw h.err('ValueError', `einstein sum subscripts string contains too many subscripts for operand ${k}`)
      ;[...t].forEach((c, i) => {
        counts.set(c, (counts.get(c) ?? 0) + 1)
        const d = ops[k].shape[i]
        const was = sizes.get(c)
        if (was !== undefined && was !== d && was !== 1 && d !== 1) throw h.err('ValueError', `operands could not be broadcast together with remapped shapes [original->remapped]: ${d} ${was}`)
        sizes.set(c, Math.max(was ?? 1, d))
      })
    })
    const out = given ?? [...counts.keys()].filter((c) => counts.get(c) === 1).sort().join('')
    for (const c of out) if (!sizes.has(c)) throw h.err('ValueError', `einstein sum subscripts string included output subscript '${c}' which never appeared in an input`)
    const labels = [...sizes.keys()]
    const dims = labels.map((c) => sizes.get(c)!)
    const outShape = [...out].map((c) => sizes.get(c)!)
    checkSize(outShape)
    const res = new Array<number>(prod(outShape)).fill(0)
    const outSt = cStrides(outShape)
    const vals = ops.map((o) => o.values())
    const steps = ops.map((o) => cStrides(o.shape))
    const total = prod(dims)
    if (total > h.maxSize * 8) throw h.err('ValueError', `np.einsum would loop ${total} times, past this sandbox's limit`)
    const at = new Array<number>(labels.length).fill(0)
    const posOf = new Map(labels.map((c, i) => [c, i]))
    for (let p = 0; p < total; p++) {
      let term = 1
      for (let k = 0; k < ops.length; k++) {
        let flat = 0
        const t = terms[k]
        for (let i = 0; i < t.length; i++) {
          const d = ops[k].shape[i]
          flat += (d === 1 ? 0 : at[posOf.get(t[i])!]) * steps[k][i]
        }
        term *= vals[k][flat]
      }
      let o = 0
      for (let i = 0; i < out.length; i++) o += at[posOf.get(out[i])!] * outSt[i]
      res[o] += term
      for (let i = labels.length - 1; i >= 0; i--) {
        if (++at[i] < dims[i]) break
        at[i] = 0
      }
    }
    const d = commonDType(ops)
    return outShape.length ? mk(res, outShape, d) : scalar(res[0], d)
  }
  const LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
  const tensordot = (a: NDArray, b: NDArray, axesV: Value | undefined): Value => {
    let [xs, ys]: number[][] = [[], []]
    if (!axesV || isNum(axesV)) {
      const n = axesV ? h.toInt(axesV) : 2
      xs = Array.from({ length: n }, (_, i) => a.ndim - n + i)
      ys = Array.from({ length: n }, (_, i) => i)
    } else {
      const [p, q] = h.iterate(axesV)
      const list = (v: Value, nd: number) => (isNum(v) ? [normAxis(h.toInt(v), nd)] : h.iterate(v).map((x) => normAxis(h.toInt(x), nd)))
      xs = list(p, a.ndim)
      ys = list(q, b.ndim)
    }
    const la = a.shape.map((_, i) => LETTERS[i])
    const lb = b.shape.map((_, i) => LETTERS[a.ndim + i])
    xs.forEach((x, i) => { lb[ys[i]] = la[x] })
    const out = [...la.filter((_, i) => !xs.includes(i)), ...lb.filter((_, i) => !ys.includes(i))].join('')
    return einsum(`${la.join('')},${lb.join('')}->${out}`, [a, b])
  }
  Object.assign(np, {
    einsum: fn('einsum', (args) => {
      if (args[0]?.k !== 'str') throw h.err('TypeError', 'np.einsum needs its subscripts as a string first')
      return einsum(args[0].v, args.slice(1).map((v) => toArray(v)))
    }, 'matmul'),
    tensordot: fn('tensordot', (args, kw) => tensordot(A(args[0]), A(args[1], 'b'), kw.axes ?? args[2]), 'matmul'),
    inner: fn('inner', (args) => {
      const [a, b] = [A(args[0]), A(args[1], 'b')]
      if (a.ndim === 0 || b.ndim === 0) return h.binop('*', args[0], args[1])
      if (a.shape[a.ndim - 1] !== b.shape[b.ndim - 1]) throw h.err('ValueError', `shapes ${JSON.stringify(a.shape)} and ${JSON.stringify(b.shape)} not aligned`)
      return tensordot(a, b, tuple([int(-1), int(-1)]))
    }, 'matmul'),
    vdot: fn('vdot', (args) => {
      const [a, b] = [A(args[0]), A(args[1], 'b')]
      if (a.size !== b.size) throw h.err('ValueError', 'cannot reshape: vdot needs the same number of elements')
      const y = b.values()
      return scalar(a.values().reduce((s, x, i) => s + x * y[i], 0), commonDType([a, b]))
    }, 'matmul'),
    kron: fn('kron', (args) => {
      const [a, b] = [A(args[0]), A(args[1], 'b')]
      const nd = Math.max(a.ndim, b.ndim)
      const fill = (s: number[]) => [...new Array<number>(nd - s.length).fill(1), ...s]
      const [sa, sb] = [fill(a.shape), fill(b.shape)]
      const shape = sa.map((d, i) => d * sb[i])
      const st = cStrides(shape)
      const out = new Array<number>(prod(shape)).fill(0)
      const [va, vb] = [a.values(), b.values()]
      va.forEach((x, i) => {
        const ia = unravel(i, sa)
        vb.forEach((y, j) => {
          const ib = unravel(j, sb)
          out[ia.reduce((t, c, k) => t + (c * sb[k] + ib[k]) * st[k], 0)] = x * y
        })
      })
      return mk(out, shape, commonDType([a, b]))
    }, 'matmul'),
    cross: fn('cross', (args, kw) => {
      const [a, b] = [A(args[0]), A(args[1], 'b')]
      if (kw.axis || kw.axisa || kw.axisb || kw.axisc) throw h.err('NotImplementedError', 'np.cross takes vectors along the last axis in this sandbox')
      const na = a.shape[a.ndim - 1]
      const nb = b.shape[b.ndim - 1]
      if (![2, 3].includes(na) || ![2, 3].includes(nb)) throw h.err('ValueError', 'incompatible dimensions for cross product\n(dimension must be 2 or 3)')
      const outer = broadcastShapes([a.shape.slice(0, -1), b.shape.slice(0, -1)])
      const [va, vb] = [broadcastTo(a, [...outer, na]).values(), broadcastTo(b, [...outer, nb]).values()]
      const out: number[] = []
      const both2 = na === 2 && nb === 2
      for (let i = 0; i < prod(outer); i++) {
        const [a0, a1, a2 = 0] = va.slice(i * na, i * na + na)
        const [b0, b1, b2 = 0] = vb.slice(i * nb, i * nb + nb)
        if (both2) out.push(a0 * b1 - a1 * b0)
        else out.push(a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0)
      }
      const d = commonDType([a, b])
      const shape = both2 ? outer : [...outer, 3]
      return shape.length ? mk(out, shape, d) : scalar(out[0], d)
    }, 'matmul'),
    vecdot: fn('vecdot', (args) => tensordotLast(A(args[0], 'x1'), A(args[1], 'x2')), 'matmul'),
  })
  function tensordotLast(a: NDArray, b: NDArray): Value {
    const res = binaryOp('*' as BinOp, a, b)
    const out = lanes(res, -1, (lane) => [lane.reduce((s, x) => s + x, 0)], res.dtype)
    const shape = res.shape.slice(0, -1)
    return shape.length ? wrap(reshape(out, shape)) : scalar(out.values()[0], res.dtype)
  }

  // ---------- functions over arrays ----------

  const callableShape = (results: Value[], shape: number[]): Value => {
    const vals = results.map((r) => h.num(r))
    const d: DType = results.every((r) => r.k === 'bool') ? 'bool' : results.every((r) => r.k === 'int' || r.k === 'bool') ? 'int64' : 'float64'
    return shape.length ? mk(vals, shape, d) : results[0]
  }
  const vectorized = (f: Value, otype?: DType): Value => ({
    k: 'fn', name: 'vectorize',
    call: (args, kw) => {
      const arrs = args.map((v) => toArray(v))
      const shape = broadcastShapes(arrs.map((a) => a.shape))
      checkSize(shape)
      const cols = arrs.map((a) => broadcastTo(a, shape).values())
      const dts = arrs.map((a) => a.dtype)
      const results = Array.from({ length: prod(shape) }, (_, i) => h.call(f, cols.map((c, k) => scalar(c[i], dts[k])), kw))
      if (otype) return shape.length ? mk(results.map((r) => h.num(r)), shape, otype) : results[0]
      return callableShape(results, shape)
    },
  })
  Object.assign(np, {
    vectorize: fn('vectorize', (args, kw) => {
      const otypes = kw.otypes && kw.otypes.k !== 'none' ? h.iterate(kw.otypes)[0] : undefined
      const dec = (f: Value) => vectorized(f, dtypeArg(otypes))
      return args[0] ? dec(args[0]) : ({ k: 'fn', name: 'vectorize', call: (a: Value[]) => dec(a[0]) } as Value)
    }),
    frompyfunc: fn('frompyfunc', (args) => vectorized(args[0])),
    apply_along_axis: fn('apply_along_axis', (args, kw) => {
      const f = args[0]
      const a = A(args[2], 'arr')
      const ax = normAxis(h.toInt(args[1]), a.ndim)
      const rest = args.slice(3)
      const results: Value[] = []
      const moved = moveAxes(a, [ax], [-1])
      const v = moved.values()
      const n = a.shape[ax]
      const outer = moved.shape.slice(0, -1)
      const total = prod(outer)
      for (let i = 0; i < total; i++) results.push(h.call(f, [mk(v.slice(i * n, i * n + n), [n], a.dtype), ...rest], kw))
      const arrs = results.map((r) => toArray(r))
      const inner = arrs[0]?.shape ?? []
      const d: DType = commonDType(arrs)
      const res = NDArray.create(arrs.flatMap((x) => x.values()), [...outer, ...inner], d)
      if (!inner.length) return outer.length ? wrap(res) : results[0]
      // the function's axes take the place of the one it consumed
      const order = [...outer.keys()].slice(0, ax)
      const shift2 = inner.map((_, i) => outer.length + i)
      const tail = [...outer.keys()].slice(ax)
      return wrap(transpose(res, [...order, ...shift2, ...tail]).copy())
    }),
    apply_over_axes: fn('apply_over_axes', (args) => {
      const list = isNum(args[2]) ? [h.toInt(args[2])] : h.iterate(args[2]).map((x) => h.toInt(x))
      return list.reduce<Value>((cur, ax) => {
        const res = h.call(args[0], [cur, int(ax)])
        const c = toArray(cur)
        const r = toArray(res)
        return r.ndim === c.ndim ? res : wrap(reshape(r, c.shape.map((d, i) => (i === normAxis(ax, c.ndim) ? 1 : d))))
      }, args[1])
    }),
  })

  // ---------- information about arrays and types ----------

  class Limits extends Base {
    readonly cls: string
    attrs: Record<string, Value>
    text: string
    /** what print() shows, where that is not the repr */
    printed: string | undefined
    constructor(cls: string, attrs: Record<string, Value>, text: string, printed?: string) {
      super()
      this.cls = cls
      this.attrs = attrs
      this.text = text
      this.printed = printed
    }
    repr() { return this.text }
    str() { return this.printed ?? this.text }
    getAttr(name: string) { return this.attrs[name] }
  }
  class Opts extends Base {
    readonly cls = 'printoptions'
    saved: typeof printOptions | null = null
    next: Partial<typeof printOptions>
    constructor(next: Partial<typeof printOptions>) {
      super()
      this.next = next
    }
    repr() { return '<printoptions>' }
    enter() {
      this.saved = { ...printOptions }
      Object.assign(printOptions, this.next)
      return NONE
    }
    exit() {
      if (this.saved) Object.assign(printOptions, this.saved)
    }
  }
  class NoErrors extends Base {
    readonly cls = 'errstate'
    repr() { return '<errstate>' }
    enter() { return NONE }
    exit() { /* the sandbox does not warn on floating point errors */ }
  }
  const printKeys = (kw: Kw): Partial<typeof printOptions> => {
    const out: Partial<typeof printOptions> = {}
    if (kw.precision && kw.precision.k !== 'none') out.precision = h.toInt(kw.precision)
    if (kw.threshold && kw.threshold.k !== 'none') out.threshold = h.num(kw.threshold)
    if (kw.edgeitems && kw.edgeitems.k !== 'none') out.edgeitems = h.toInt(kw.edgeitems)
    if (kw.linewidth && kw.linewidth.k !== 'none') out.linewidth = h.toInt(kw.linewidth)
    if (kw.suppress && kw.suppress.k !== 'none') out.suppress = h.truthy(kw.suppress)
    return out
  }
  const scalarKinds: Record<string, DType> = {
    int8: 'int64', int16: 'int64', intp: 'int64', intc: 'int64', int_: 'int64', short: 'int64', long: 'int64', longlong: 'int64', byte: 'int64',
    ubyte: 'uint8', ushort: 'int64', uint8: 'uint8', uint16: 'int64', uint32: 'int64', uint64: 'int64', uint: 'int64', uintc: 'int64', uintp: 'int64', ulong: 'int64', ulonglong: 'int64',
    float16: 'float64', half: 'float64', single: 'float32', double: 'float64', longdouble: 'float64',
  }
  /** the width is honoured when converting; an array then holds its values as int64 */
  const fitWidth = (name: string) => {
    const m = /^(u?)int(\d+)$/.exec(name)
    if (!m || Number(m[2]) >= 53) return (x: number) => x
    const span = 2 ** Number(m[2])
    const low = m[1] ? 0 : -span / 2
    return (x: number) => ((((x - low) % span) + span) % span) + low
  }
  const typeLike = (name: string, d: DType): Value => {
    const fit = fitWidth(name)
    return { k: 'type', name: `numpy.${name}`, dtype: d, call: (args) => {
      const v = args[0] ?? int(0)
      if (v.k === 'array' || v.k === 'list') return mk(toArray(v).values().map((x) => fit(castValue(x, d))), toArray(v).shape, d)
      // NumPy 2 refuses a Python int that does not fit, and wraps anything else
      if (d === 'uint8' && v.k === 'int' && (v.v < 0 || v.v > 255)) throw h.err('OverflowError', `Python integer ${v.v} out of bounds for ${name}`)
      return scalar(fit(castValue(h.num(v), d)), d)
    } }
  }
  for (const [name, d] of Object.entries(scalarKinds)) np[name] ??= typeLike(name, d)
  // the abstract types are only asked about
  const FAMILY: Record<string, DType[]> = {
    generic: ['int64', 'float64', 'float32', 'bool', 'uint8'], number: ['int64', 'float64', 'float32', 'uint8'], integer: ['int64', 'uint8'], signedinteger: ['int64'], unsignedinteger: ['uint8'],
    inexact: ['float64', 'float32'], floating: ['float64', 'float32'], complexfloating: [], flexible: [], character: [],
  }
  for (const name of Object.keys(FAMILY)) np[name] ??= { k: 'type', name: `numpy.${name}`, call: () => { throw h.err('TypeError', `cannot create 'numpy.${name}' instances`) } }
  Object.assign(np, {
    True_: bool(true), False_: bool(false), bool: np.bool_, euler_gamma: float(0.5772156649015329), little_endian: bool(true), PINF: float(Infinity), NINF: float(-Infinity), PZERO: float(0), NZERO: float(-0),
    newaxis: np.newaxis, str_: { k: 'type', name: 'numpy.str_', call: (args: Value[]) => S(args.length ? h.str(args[0]) : '') },
    dtype: { k: 'type', name: 'numpy.dtype', call: (args: Value[]) => {
      const d = dtypeArg(args[0])
      if (!d) throw h.err('TypeError', `data type ${repr(args[0])} not understood`)
      return { k: 'dtype', d }
    } },
    // numpy's own table (numpy/lib/_type_check_impl.py)
    typename: fn('typename', (args) => S({ S1: 'character', '?': 'bool', b: 'signed char', B: 'unsigned char', h: 'short', H: 'unsigned short', i: 'integer', I: 'unsigned integer', l: 'long integer', L: 'unsigned long integer', q: 'long long integer', Q: 'unsigned long long integer', f: 'single precision', d: 'double precision', g: 'long precision', F: 'complex single precision', D: 'complex double precision', G: 'complex long double precision', S: 'string', U: 'unicode', V: 'void', O: 'object' }[(args[0] as { v: string }).v] ?? 'unknown')),
    isscalar: fn('isscalar', (args) => bool(isNum(args[0]) || args[0].k === 'str')),
    iterable: fn('iterable', (args) => {
      try {
        h.iterOf(args[0])
        return bool(true)
      } catch {
        return bool(false)
      }
    }),
    isrealobj: fn('isrealobj', () => bool(true)), iscomplexobj: fn('iscomplexobj', () => bool(false)), isfortran: fn('isfortran', () => bool(false)),
    array_equiv: fn('array_equiv', (args) => {
      const [a, b] = [A(args[0], 'a1'), A(args[1], 'a2')]
      try {
        const shape = broadcastShapes([a.shape, b.shape])
        const y = broadcastTo(b, shape).values()
        return bool(broadcastTo(a, shape).values().every((x, i) => x === y[i]))
      } catch {
        return bool(false)
      }
    }),
    array2string: fn('array2string', (args, kw) => {
      const next = printKeys(kw)
      const saved = { ...printOptions }
      Object.assign(printOptions, next)
      try {
        const sep = kw.separator?.k === 'str' ? kw.separator.v : ' '
        const s = arrayStr(A(args[0]))
        return S(sep === ' ' ? s : s.replace(/ +/g, (m) => (m.length === 1 ? sep : sep + m.slice(1))))
      } finally {
        Object.assign(printOptions, saved)
      }
    }),
    array_str: fn('array_str', (args) => S(arrayStr(A(args[0])))),
    array_repr: fn('array_repr', (args) => S(arrayRepr(A(args[0])))),
    set_printoptions: fn('set_printoptions', (_args, kw) => {
      Object.assign(printOptions, printKeys(kw))
      return NONE
    }),
    get_printoptions: fn('get_printoptions', () => ({ k: 'dict', d: PyDict.from([
      [S('edgeitems'), int(printOptions.edgeitems)], [S('threshold'), int(printOptions.threshold)], [S('precision'), int(printOptions.precision)],
      [S('linewidth'), int(printOptions.linewidth)], [S('suppress'), bool(printOptions.suppress)],
    ]) })),
    printoptions: fn('printoptions', (_args, kw) => py.obj(new Opts(printKeys(kw)))),
    seterr: fn('seterr', () => ({ k: 'dict', d: PyDict.from([[S('divide'), S('warn')], [S('over'), S('warn')], [S('under'), S('ignore')], [S('invalid'), S('warn')]]) })),
    geterr: fn('geterr', () => ({ k: 'dict', d: PyDict.from([[S('divide'), S('warn')], [S('over'), S('warn')], [S('under'), S('ignore')], [S('invalid'), S('warn')]]) })),
    errstate: fn('errstate', () => py.obj(new NoErrors())),
    finfo: fn('finfo', (args) => {
      const isSingle = /float32|single|float16|half/.test(h.repr(args[0]))
      const [eps, max, tiny, digits] = isSingle ? [1.1920929e-7, 3.4028235e38, 1.1754944e-38, 6] : [2 ** -52, Number.MAX_VALUE, 2.2250738585072014e-308, 15]
      return py.obj(new Limits('finfo', { eps: float(eps), max: float(max), min: float(-max), tiny: float(tiny), smallest_normal: float(tiny), precision: int(digits), resolution: float(10 ** -digits), bits: int(isSingle ? 32 : 64), dtype: { k: 'dtype', d: 'float64' } }, `finfo(resolution=${10 ** -digits}, min=${-max}, max=${max}, dtype=float64)`, isSingle ? FINFO_32 : FINFO_64))
    }),
    iinfo: fn('iinfo', (args) => {
      const name = /(u?int\d+)/.exec(h.repr(args[0]))?.[1] ?? ((args[0].k === 'type' && args[0].name === 'int') || args[0].k === 'int' ? 'int64' : null)
      if (!name) throw h.err('ValueError', `Invalid integer data type ${repr(args[0])}`)
      const bits = Number(/\d+/.exec(name)![0])
      const unsigned = name.startsWith('u')
      const min = unsigned ? 0 : -(2 ** (bits - 1))
      const max = unsigned ? 2 ** bits - 1 : 2 ** (bits - 1) - 1
      return py.obj(new Limits('iinfo', { min: int(min), max: int(max), bits: int(bits) }, `iinfo(min=${min}, max=${max}, dtype=${name})`, `Machine parameters for ${name}\n${LINE}\nmin = ${name === 'int64' ? '-9223372036854775808' : min}\nmax = ${name === 'int64' ? '9223372036854775807' : name === 'uint64' ? '18446744073709551615' : max}\n${LINE}\n`))
    }),
    result_type: fn('result_type', (args) => ({ k: 'dtype', d: commonDType(args.map((v) => {
      const d = dtypeArg(v)
      return d ? NDArray.create([], [0], d) : toArray(v)
    })) })),
    promote_types: fn('promote_types', (args) => ({ k: 'dtype', d: commonDType(args.map((v) => NDArray.create([], [0], dtypeArg(v) ?? 'float64'))) })),
    can_cast: fn('can_cast', (args) => {
      const rank: Record<DType, number> = { bool: 0, uint8: 1, int64: 2, float32: 3, float64: 4 }
      const from = dtypeArg(args[0]) ?? toArray(args[0]).dtype
      const to = dtypeArg(args[1]) ?? 'float64'
      return bool(rank[from] <= rank[to])
    }),
    issubdtype: fn('issubdtype', (args) => {
      const d = dtypeArg(args[0])
      const t = args[1]
      const fam = t.k === 'type' ? FAMILY[t.name.replace('numpy.', '')] : undefined
      if (fam) return bool(!!d && fam.includes(d))
      return bool(d === dtypeArg(t))
    }),
    binary_repr: fn('binary_repr', (args, kw) => {
      const n = h.toInt(args[0])
      const w = optInt(args, kw, 1, 'width')
      if (n >= 0) return S(w === null ? n.toString(2) : n.toString(2).padStart(w, '0'))
      if (w === null) return S(`-${(-n).toString(2)}`)
      return S((2 ** w + n).toString(2).padStart(w, '0'))
    }),
    base_repr: fn('base_repr', (args, kw) => {
      const n = h.toInt(args[0])
      const b = optInt(args, kw, 1, 'base') ?? 2
      const pad = optInt(args, kw, 2, 'padding') ?? 0
      return S(`${n < 0 ? '-' : ''}${'0'.repeat(pad)}${Math.abs(n).toString(b).toUpperCase()}`)
    }),
    packbits: fn('packbits', (args) => {
      const v = A(args[0]).values()
      const out: number[] = []
      for (let i = 0; i < v.length; i += 8) {
        let byte = 0
        for (let j = 0; j < 8; j++) byte = byte * 2 + (v[i + j] ? 1 : 0)
        out.push(byte)
      }
      return mk(out, [out.length], 'int64')
    }),
    unpackbits: fn('unpackbits', (args) => {
      const v = A(args[0]).values()
      return mk(v.flatMap((x) => Array.from({ length: 8 }, (_, j) => (x >> (7 - j)) & 1)), [v.length * 8], 'int64')
    }),
    info: fn('info', () => {
      h.print('The sandbox has no documentation for this; see numpy.org.\n')
      return NONE
    }),
  })
  // files, in the run's in-memory folder: a .npy or .npz holds the arrays themselves (see pickle), a text file holds text
  const builtin = (name: string) => (h.module('builtins') as { attrs: Record<string, Value> }).attrs[name]
  const pathOf = (v: Value): string | null => (v.k === 'str' ? v.v : v.k === 'obj' && v.o.fspath ? v.o.fspath() : null)
  const openFile = (path: string, mode: string) => h.call(builtin('open'), [S(path), S(mode)])
  const closeFile = (f: Value) => h.call(h.getAttr(f, 'close'), [])
  const pickled = (name: 'dump' | 'load', args: Value[]) => h.call(h.getAttr(h.module('pickle'), name), args)
  const NOT_NPY = 'This file contains pickled (object) data. If you trust the file you can load it unsafely using the `allow_pickle=` keyword argument or `pickle.load()`.'
  const fileText = (path: string) => (h.state.get('files') as Map<string, string> | undefined)?.get(fullPath(path))
  /** what np.load gives for a .npz */
  class NpzFile extends Base {
    readonly cls = 'NpzFile'
    readonly arrays: Map<string, NDArray>
    readonly name: string | null
    constructor(arrays: Map<string, NDArray>, name: string | null) {
      super()
      this.arrays = arrays
      this.name = name
    }
    pickled() { return this }
    repr() { return `NpzFile ${this.name === null ? 'object' : repr(S(this.name))} with keys: ${[...this.arrays.keys()].join(', ')}` }
    len() { return this.arrays.size }
    iter() { return [...this.arrays.keys()].map(S) }
    contains(v: Value) { return v.k === 'str' && this.arrays.has(v.v) }
    getItem(idx: Value): Value {
      const a = idx.k === 'str' ? this.arrays.get(idx.v) : undefined
      if (!a) throw h.err('KeyError', repr(S(`${h.str(idx)} is not a file in the archive`)))
      return wrap(a.copy())
    }
    enter() { return { k: 'obj', o: this } as Value }
    exit() {}
    getAttr(name: string): Value | undefined {
      const names = () => [...this.arrays.keys()].map(S)
      switch (name) {
        case 'files': return L(names())
        case 'keys': return fn(name, () => L(names()))
        case 'values': return fn(name, () => L([...this.arrays.values()].map((a) => wrap(a.copy()))))
        case 'items': return fn(name, () => L([...this.arrays].map(([n, a]) => tuple([S(n), wrap(a.copy())]))))
        case 'get': return fn(name, (args) => (this.contains(args[0]) ? this.getItem(args[0]) : args[1] ?? NONE))
        case 'close': return fn(name, () => NONE)
      }
      return undefined
    }
  }
  const toFile = (file: Value, ext: string, obj: Value) => {
    const path = pathOf(file)
    if (path === null) {
      pickled('dump', [obj, file])
      return NONE
    }
    const f = openFile(path.endsWith(ext) ? path : path + ext, 'wb')
    pickled('dump', [obj, f])
    closeFile(f)
    return NONE
  }
  np.save = fn('save', (args, kw) => {
    const arr = kw.arr ?? args[1]
    if (!arr) throw h.err('TypeError', "save() missing 1 required positional argument: 'arr'")
    if (arr.k === 'dict' || arr.k === 'inst' || arr.k === 'set') throw h.err('NotImplementedError', 'np.save of an object array is not available in this sandbox: use pickle.dump for this')
    return toFile(kw.file ?? args[0], '.npy', wrap(A(arr).copy()))
  })
  const savez = (name: string) => fn(name, (args, kw) => {
    const { file: fileKw, allow_pickle: _, ...named } = kw
    const file = fileKw ?? args[0]
    const rest = fileKw ? args : args.slice(1)
    const arrays = new Map<string, NDArray>(rest.map((v, i): [string, NDArray] => [`arr_${i}`, A(v).copy()]))
    for (const [k, v] of Object.entries(named)) {
      if (arrays.has(k)) throw h.err('ValueError', `Cannot use un-named variables and keyword ${k}`)
      arrays.set(k, A(v).copy())
    }
    const path = pathOf(file)
    return toFile(file, '.npz', { k: 'obj', o: new NpzFile(arrays, path === null ? null : path.endsWith('.npz') ? path : `${path}.npz`) })
  })
  np.savez = savez('savez')
  np.savez_compressed = savez('savez_compressed')
  np.load = fn('load', (args, kw) => {
    const file = kw.file ?? args[0]
    const allow = !!(kw.allow_pickle ?? args[2]) && h.truthy(kw.allow_pickle ?? args[2])
    const mmap = kw.mmap_mode ?? args[1]
    if (mmap && mmap.k !== 'none') throw h.err('NotImplementedError', 'np.load: mmap_mode is not available in this sandbox')
    const path = pathOf(file)
    let out: Value
    if (path !== null) {
      const text = fileText(path)
      if (text === '') throw h.err('EOFError', 'No data left in file')
      if (text !== undefined && !text.startsWith('\0pickle:')) throw h.err(allow ? 'UnpicklingError' : 'ValueError', allow ? `Failed to interpret file ${repr(S(path))} as a pickle` : NOT_NPY)
      const f = openFile(path, 'rb')
      out = pickled('load', [f])
      closeFile(f)
    } else out = pickled('load', [file])
    if (allow || out.k === 'array' || (out.k === 'obj' && out.o instanceof NpzFile)) return out
    throw h.err('ValueError', NOT_NPY)
  })
  np.savetxt = fn('savetxt', (args, kw) => {
    const named = ['fname', 'X', 'fmt', 'delimiter', 'newline', 'header', 'footer', 'comments', 'encoding']
    const a = Object.fromEntries(named.map((n, i) => [n, kw[n] ?? args[i]])) as Record<string, Value | undefined>
    const text = (v: Value | undefined, d: string) => (v && v.k !== 'none' ? h.str(v) : d)
    const x = A(a.X, 'X')
    if (x.ndim === 0 || x.ndim > 2) throw h.err('ValueError', `Expected 1D or 2D array, got ${x.ndim}D array instead`)
    const rows = x.ndim === 1 ? x.size : x.shape[0]
    const cols = x.ndim === 1 ? 1 : x.shape[1]
    const delimiter = text(a.delimiter, ' ')
    const newline = text(a.newline, '\n')
    const comments = text(a.comments, '# ')
    const given = a.fmt ?? S('%.18e')
    let fmt: string
    if (given.k === 'str') {
      const count = (given.v.match(/%(?!%)/g) ?? []).length
      if (count === 1) fmt = new Array<string>(cols).fill(given.v).join(delimiter)
      else if (count === cols) fmt = given.v
      else throw h.err('ValueError', `fmt has wrong number of % formats:  ${given.v}`)
    } else {
      const each = h.iterate(given).map((v) => h.str(v))
      if (each.length !== cols) throw h.err('AttributeError', `fmt has wrong shape.  ${repr(given)}`)
      fmt = each.join(delimiter)
    }
    const vals = x.values()
    const cell = (v: number) => (x.dtype === 'int64' || x.dtype === 'uint8' ? int(v) : x.dtype === 'bool' ? bool(!!v) : float(v))
    let out = ''
    const header = text(a.header, '')
    if (header) out += comments + header.replace(/\n/g, `\n${comments}`) + newline
    for (let r = 0; r < rows; r++) out += h.str(h.binop('%', S(fmt), tuple(Array.from({ length: cols }, (_, c) => cell(vals[r * cols + c]))))) + newline
    const footer = text(a.footer, '')
    if (footer) out += comments + footer.replace(/\n/g, `\n${comments}`) + newline
    const target = a.fname!
    const path = pathOf(target)
    if (path === null) {
      h.call(h.getAttr(target, 'write'), [S(out)])
      return NONE
    }
    const f = openFile(path, 'w')
    h.call(h.getAttr(f, 'write'), [S(out)])
    closeFile(f)
    return NONE
  })
  /** the lines of a path, a text file or a list of strings */
  const linesFrom = (src: Value): string[] => {
    const path = pathOf(src)
    if (path !== null) {
      const text = fileText(path)
      if (text === undefined) throw h.err('FileNotFoundError', `${path} not found.`)
      return text.split('\n')
    }
    return h.iterate(src).map((v) => h.str(v).replace(/\r?\n$/, ''))
  }
  const fieldsOf = (lines: string[], comments: Value | undefined, delimiter: Value | undefined): string[][] => {
    const marks = !comments ? ['#'] : comments.k === 'none' ? [] : comments.k === 'str' ? [comments.v] : h.iterate(comments).map((v) => h.str(v))
    const sep = delimiter && delimiter.k !== 'none' ? h.str(delimiter) : null
    const rows: string[][] = []
    for (const line of lines) {
      let body = line
      for (const m of marks) {
        const at = body.indexOf(m)
        if (at >= 0) body = body.slice(0, at)
      }
      if (!body.trim()) continue
      rows.push(sep === null ? body.trim().split(/\s+/) : body.split(sep).map((x) => x.trim()))
    }
    return rows
  }
  const NUMBER_TEXT = /^[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|inf(?:inity)?|nan)$/i
  const textNumber = (s: string): number | null => {
    if (!NUMBER_TEXT.test(s)) return null
    const low = s.toLowerCase().replace(/^\+/, '')
    if (low.endsWith('nan')) return NaN
    if (low.includes('inf')) return low.startsWith('-') ? -Infinity : Infinity
    return Number(low)
  }
  const columnsOf = (u: Value | undefined): number[] | null => (!u || u.k === 'none' ? null : isNum(u) ? [h.toInt(u)] : h.iterate(u).map((c) => h.toInt(c)))
  const pick = (row: string[], cols: number[] | null) => (cols ? cols.map((c) => row[c < 0 ? c + row.length : c]) : row)
  /** the shape the two readers give: one row or one column is a vector, and a single value a scalar array */
  const tableResult = (vals: number[], rows: number, cols: number, d: DType, unpack: boolean, ndmin: number): Value => {
    if (!rows) return wrap(NDArray.create([], [0], d))
    let shape = [rows, cols]
    if (ndmin < 2) shape = rows === 1 && cols === 1 ? (ndmin === 1 ? [1] : []) : rows === 1 || cols === 1 ? [rows * cols] : shape
    const t = NDArray.create(vals, shape, d)
    return wrap(unpack && shape.length === 2 ? transpose(t) : t)
  }
  np.loadtxt = fn('loadtxt', (args, kw) => {
    const named = ['fname', 'dtype', 'comments', 'delimiter', 'converters', 'skiprows', 'usecols', 'unpack', 'ndmin', 'encoding', 'max_rows', 'quotechar', 'like']
    const a = Object.fromEntries(named.map((n, i) => [n, kw[n] ?? args[i]])) as Record<string, Value | undefined>
    if (a.converters && a.converters.k !== 'none') throw h.err('NotImplementedError', 'np.loadtxt: converters is not available in this sandbox')
    const d = dtypeArg(a.dtype) ?? 'float64'
    const skip = a.skiprows ? h.toInt(a.skiprows) : 0
    const rows = fieldsOf(linesFrom(a.fname!).slice(skip), a.comments, a.delimiter)
    const kept = a.max_rows && a.max_rows.k !== 'none' ? rows.slice(0, h.toInt(a.max_rows)) : rows
    const cols = columnsOf(a.usecols)
    const width = kept[0] ? (cols ?? kept[0]).length : 0
    const vals: number[] = []
    kept.forEach((row, r) => {
      if (!cols && row.length !== width) throw h.err('ValueError', `the number of columns changed from ${width} to ${row.length} at row ${r + 1}; use \`usecols\` to select a subset and avoid this error`)
      pick(row, cols).forEach((s, c) => {
        const n = textNumber(s ?? '')
        if (n === null || (!isFloatD(d) && !Number.isInteger(n))) throw h.err('ValueError', `could not convert string ${repr(S(s ?? ''))} to ${d} at row ${r}, column ${(cols ? cols[c] : c) + 1}.`)
        vals.push(n)
      })
    })
    return tableResult(vals, kept.length, width, d, !!a.unpack && h.truthy(a.unpack), a.ndmin ? h.toInt(a.ndmin) : 0)
  })
  np.genfromtxt = fn('genfromtxt', (args, kw) => {
    const named = ['fname', 'dtype', 'comments', 'delimiter', 'skip_header', 'skip_footer', 'converters', 'missing_values', 'filling_values', 'usecols', 'names']
    const a = Object.fromEntries(named.map((n, i) => [n, kw[n] ?? args[i]])) as Record<string, Value | undefined>
    for (const n of ['names', 'converters', 'missing_values']) if (a[n] && a[n]!.k !== 'none') throw h.err('NotImplementedError', `np.genfromtxt: ${n} is not available in this sandbox`)
    if (a.dtype?.k === 'none') throw h.err('NotImplementedError', 'np.genfromtxt: dtype=None gives a structured array, which is not available in this sandbox')
    const d = dtypeArg(a.dtype) ?? 'float64'
    const fill = a.filling_values && a.filling_values.k !== 'none' ? h.num(a.filling_values) : isFloatD(d) ? NaN : -1
    const lines = linesFrom(a.fname!).slice(a.skip_header ? h.toInt(a.skip_header) : 0)
    const foot = a.skip_footer ? h.toInt(a.skip_footer) : 0
    const rows = fieldsOf(foot ? lines.slice(0, -foot) : lines, a.comments, a.delimiter)
    const cols = columnsOf(a.usecols)
    const width = rows[0]?.length ?? 0
    const bad = rows.flatMap((row, r) => (row.length === width ? [] : [`    Line #${r + 1 + (a.skip_header ? h.toInt(a.skip_header) : 0)} (got ${row.length} columns instead of ${width})`]))
    if (bad.length) throw h.err('ValueError', `Some errors were detected !\n${bad.join('\n')}`)
    const vals = rows.flatMap((row) => pick(row, cols).map((s) => textNumber(s ?? '') ?? fill))
    return tableResult(vals, rows.length, cols ? cols.length : width, d, !!kw.unpack && h.truthy(kw.unpack), 0)
  })
  // the rest cannot work on what these files hold
  for (const name of ['fromfile', 'memmap', 'fromregex']) {
    np[name] ??= fn(name, () => {
      throw h.err('NotImplementedError', `np.${name} reads raw bytes, which this sandbox's files do not hold: use np.load or np.loadtxt`)
    })
  }

  // ---------- polynomials (highest power first, as np.polyfit returns them) ----------

  const trimLead = (c: number[]) => {
    let i = 0
    while (i < c.length - 1 && c[i] === 0) i++
    return c.slice(i)
  }
  const pad2 = (a: number[], b: number[]) => {
    const n = Math.max(a.length, b.length)
    return [[...new Array<number>(n - a.length).fill(0), ...a], [...new Array<number>(n - b.length).fill(0), ...b]]
  }
  const polyAdd = (a: number[], b: number[]) => {
    const [x, y] = pad2(a, b)
    return trimLead(x.map((v, i) => v + y[i]))
  }
  const polySub = (a: number[], b: number[]) => {
    const [x, y] = pad2(a, b)
    return trimLead(x.map((v, i) => v - y[i]))
  }
  const polyMul = (a: number[], b: number[]) => trimLead(convolve(a, b))
  const polyDer = (c: number[], m = 1): number[] => {
    let cur = c
    for (let k = 0; k < m; k++) cur = cur.length <= 1 ? [0] : cur.slice(0, -1).map((v, i) => v * (cur.length - 1 - i))
    return cur
  }
  const polyInt = (c: number[], m = 1, k = 0): number[] => {
    let cur = c
    for (let i = 0; i < m; i++) cur = [...cur.map((v, j) => v / (cur.length - j)), k]
    return cur
  }
  const polyDiv = (u: number[], v: number[]): [number[], number[]] => {
    if (v.every((x) => x === 0)) throw h.err('ZeroDivisionError', 'polynomial division by zero')
    const d = trimLead(v)
    const rem = [...u]
    const q: number[] = []
    for (let i = 0; i <= rem.length - d.length; i++) {
      const c = rem[i] / d[0]
      q.push(c)
      d.forEach((x, j) => { rem[i + j] -= c * x })
    }
    return [q.length ? q : [0], trimLead(rem.slice(Math.max(0, rem.length - d.length + 1)).length ? rem.slice(Math.max(0, rem.length - d.length + 1)) : [0])]
  }
  const rootsVal = (c: number[]): Value => h.call(np.roots, [mk(c, [c.length])])
  class Poly1d extends Base {
    readonly cls = 'poly1d'
    c: number[]
    d: DType
    constructor(c: number[], d: DType = allInt(c) ? 'int64' : 'float64') {
      super()
      this.c = trimLead(c.length ? c : [0])
      this.d = d
    }
    repr() { return arrayRepr(this.toArray()).replace(/^array\(/, 'poly1d(').replace(/\n {6}/g, '\n       ') }
    str() {
      const n = this.c.length - 1
      let text = '0'
      this.c.forEach((v, k) => {
        const coef = formatSpec(float(v), '.4g').replace(/\.0000$/, '')
        const power = n - k
        let term: string
        if (power === 0) term = coef !== '0' ? coef : k === 0 ? '0' : ''
        else if (coef === '0') term = ''
        else term = power === 1 ? `${coef} x` : `${coef} x**${power}`
        if (k === 0) text = term
        else if (term) text = term.startsWith('-') ? `${text} - ${term.slice(1)}` : `${text} + ${term}`
      })
      // the powers go on a line of their own above the terms
      let [top, bottom, output, at] = ['', '', ' ', 0]
      for (const m of text.matchAll(/\*\*(\d*)/g)) {
        const part = text.slice(at, m.index)
        at = m.index + m[0].length
        const under = part + ' '.repeat(m[1].length - 1)
        const over = ' '.repeat(part.length - 1) + m[1]
        if (bottom.length + under.length > 70 || top.length + over.length > 70) {
          output += `${top}\n${bottom}\n `
          top = over
          bottom = under
        } else {
          bottom += under
          top += over
        }
      }
      return `${output}${top}\n${bottom}${text.slice(at)}`
    }
    toArray() { return NDArray.create(this.c, [this.c.length], this.d) }
    len() { return this.c.length - 1 }
    iter() { return this.c.map((x) => scalar(x, this.d)) }
    getItem(idx: Value) {
      const p = h.toInt(idx)
      return scalar(this.c[this.c.length - 1 - p] ?? 0, this.d)
    }
    call(args: Value[]) {
      const x = toArray(args[0])
      const res = x.values().map((v) => this.c.reduce((s, c) => s * v + c, 0))
      const d: DType = this.d === 'float64' || isFloatD(x.dtype) ? 'float64' : 'int64'
      return isNum(args[0]) ? scalar(res[0], d) : mk(res, x.shape, d)
    }
    binop(op: string, other: Value, reflected: boolean): Value | undefined {
      const o = other.k === 'obj' && other.o instanceof Poly1d ? other.o.c : isNum(other) ? [h.num(other)] : toArray(other).values()
      const [a, b] = reflected ? [o, this.c] : [this.c, o]
      const d: DType = this.d === 'float64' || other.k === 'float' || (other.k === 'obj' && other.o instanceof Poly1d && other.o.d === 'float64') ? 'float64' : 'int64'
      if (op === '+') return py.obj(new Poly1d(polyAdd(a, b), d))
      if (op === '-') return py.obj(new Poly1d(polySub(a, b), d))
      if (op === '*') return py.obj(new Poly1d(polyMul(a, b), d))
      if (op === '==') return bool(a.length === b.length && a.every((x, i) => x === b[i]))
      if (op === '!=') return bool(!(a.length === b.length && a.every((x, i) => x === b[i])))
      if (op === '**' && !reflected) {
        let out = [1]
        for (let i = 0; i < h.toInt(other); i++) out = polyMul(out, this.c)
        return py.obj(new Poly1d(out, this.d))
      }
      return undefined
    }
    getAttr(name: string): Value | undefined {
      switch (name) {
        case 'c': case 'coef': case 'coefficients': case 'coeffs': return wrap(this.toArray())
        case 'order': case 'o': return int(this.c.length - 1)
        case 'r': case 'roots': return rootsVal(this.c)
        case 'deriv': return fn('deriv', (a, kw) => py.obj(new Poly1d(polyDer(this.c, optInt(a, kw, 0, 'm') ?? 1), this.d)))
        case 'integ': return fn('integ', (a, kw) => py.obj(new Poly1d(polyInt(this.c, optInt(a, kw, 0, 'm') ?? 1, a[1] ? h.num(a[1]) : kw.k ? h.num(kw.k) : 0), 'float64')))
      }
      return undefined
    }
  }
  const allInt = (c: number[]) => c.every(Number.isInteger)
  const coefOf = (v: Value) => (v.k === 'obj' && v.o instanceof Poly1d ? v.o.c : toArray(v).values())
  const polyResult = (args: Value[], c: number[]): Value => {
    if (args.some((v) => v.k === 'obj' && v.o instanceof Poly1d)) return py.obj(new Poly1d(c))
    const d: DType = args.some((v) => (v.k === 'obj' ? false : isFloatD(toArray(v).dtype))) ? 'float64' : 'int64'
    return mk(c, [c.length], d)
  }
  Object.assign(np, {
    poly1d: { k: 'type', name: 'numpy.poly1d', call: (args: Value[], kw: Kw) => {
      if ((kw.r ?? args[1]) && h.truthy(kw.r ?? args[1])) {
        const out = A(args[0]).values().reduce((p, r) => polyMul(p, [1, -r]), [1])
        return py.obj(new Poly1d(out, 'float64'))
      }
      if (args[0].k === 'obj' && args[0].o instanceof Poly1d) return py.obj(new Poly1d(args[0].o.c, args[0].o.d))
      return py.obj(new Poly1d(coefOf(args[0]), isFloatD(toArray(args[0]).dtype) ? 'float64' : 'int64'))
    } },
    poly: fn('poly', (args) => {
      const a = A(args[0], 'seq_of_zeros')
      const vals = a.ndim === 2 ? (eigGeneral(rows(a)).re) : a.values()
      const out = vals.reduce((p, r) => polyMul(p, [1, -r]), [1])
      return mk(out, [out.length])
    }, 'poly'),
    polyadd: fn('polyadd', (args) => polyResult(args, polyAdd(coefOf(args[0]), coefOf(args[1]))), 'poly'),
    polysub: fn('polysub', (args) => polyResult(args, polySub(coefOf(args[0]), coefOf(args[1]))), 'poly'),
    polymul: fn('polymul', (args) => polyResult(args, polyMul(coefOf(args[0]), coefOf(args[1]))), 'poly'),
    polydiv: fn('polydiv', (args) => {
      const [q, r] = polyDiv(coefOf(args[0]), coefOf(args[1]))
      if (args.some((v) => v.k === 'obj')) return tuple([py.obj(new Poly1d(q)), py.obj(new Poly1d(r))])
      return tuple([mk(q, [q.length]), mk(r, [r.length])])
    }, 'poly'),
    polyder: fn('polyder', (args, kw) => polyResult(args.slice(0, 1), polyDer(coefOf(args[0]), optInt(args, kw, 1, 'm') ?? 1)), 'poly'),
    polyint: fn('polyint', (args, kw) => {
      const c = polyInt(coefOf(args[0]), optInt(args, kw, 1, 'm') ?? 1, optNum(args, kw, 2, 'k', 0))
      return args[0].k === 'obj' ? py.obj(new Poly1d(c)) : mk(c, [c.length])
    }, 'poly'),
    roots: fn('roots', (args) => {
      // the eigenvalues of the companion matrix, which is what np.roots computes
      const raw = A(args[0], 'p').values()
      let start = 0
      while (start < raw.length && raw[start] === 0) start++
      let end = raw.length
      while (end > start && raw[end - 1] === 0) end--
      const c = raw.slice(start, end)
      const zeros = raw.length - end
      const n = c.length - 1
      if (n < 1) return mk(new Array<number>(zeros).fill(0), [zeros])
      const comp = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_x, j) => (i === 0 ? -c[j + 1] / c[0] : j === i - 1 ? 1 : 0)))
      const { re, im } = eigGeneral(comp)
      const allRe = [...re, ...new Array<number>(zeros).fill(0)]
      const allIm = [...im, ...new Array<number>(zeros).fill(0)]
      return allIm.some((x) => x !== 0) ? { k: 'carray', re: allRe, im: allIm } : mk(allRe, [allRe.length])
    }, 'poly'),
  })

  // ---------- np.linalg ----------

  const singular = () => h.err('LinAlgError', 'Singular matrix')
  const solveArr = (a: NDArray, b: NDArray): Value => {
    const m = square(a, 'solve')
    const n = m.length
    if (b.ndim === 1) {
      if (b.size !== n) throw h.err('ValueError', `solve: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (m,m),(m)->(m) (size ${b.size} is different from ${n})`)
      const x = solveM(m, b.values().map((v) => [v]))
      if (!x) throw singular()
      return mk(x.flat(), [n])
    }
    if (b.shape[0] !== n) throw h.err('ValueError', `solve: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (m,m),(m,n)->(m,n) (size ${b.shape[0]} is different from ${n})`)
    const x = solveM(m, rows(b))
    if (!x) throw singular()
    return matrix(x, b.shape[1])
  }
  const needs2d = (a: NDArray) => {
    if (a.ndim !== 2) throw h.err('LinAlgError', `${a.ndim}-dimensional array given. Array must be at least two-dimensional`)
    return rows(a)
  }
  const singularValues = (a: NDArray) => svdThin(needs2d(a)).s
  const invM = (m: Mat): Mat => {
    const x = solveM(m, identity(m.length))
    if (!x) throw singular()
    return x
  }
  const pinvOf = (a: NDArray, rcond: number): Value => {
    const m = needs2d(a)
    const { u, s, v } = svdThin(m)
    const cut = rcond * Math.max(0, ...s)
    const scaled = v.map((row) => row.map((x, j) => (s[j] > cut ? x / s[j] : 0)))
    return matrix(mul(scaled, transposeM(u)), m.length)
  }
  const normOf = (a: NDArray, ord: Value | undefined, axis: Value | undefined, keep: boolean): Value => {
    // float32 data gives a float32 norm
    const d: DType = a.dtype === 'float32' ? 'float32' : 'float64'
    const one = (x: number) => fromScalar(x, d)
    const vectorNorm = (xs: number[]): number => {
      if (!ord || ord.k === 'none') return Math.sqrt(xs.reduce((s, x) => s + x * x, 0))
      if (ord.k === 'str') throw h.err('ValueError', `Invalid norm order '${ord.v}' for vectors`)
      const p = h.num(ord)
      if (p === Infinity) return Math.max(...xs.map(Math.abs))
      if (p === -Infinity) return Math.min(...xs.map(Math.abs))
      if (p === 0) return xs.filter((x) => x !== 0).length
      return xs.reduce((s, x) => s + Math.abs(x) ** p, 0) ** (1 / p)
    }
    const matrixNorm = (m: Mat): number => {
      if (!ord || ord.k === 'none' || (ord.k === 'str' && ord.v === 'fro')) return Math.sqrt(m.flat().reduce((s, x) => s + x * x, 0))
      if (ord.k === 'str' && ord.v === 'nuc') return svdThin(m).s.reduce((s, x) => s + x, 0)
      if (ord.k === 'str') throw h.err('ValueError', 'Invalid norm order for matrices.')
      const p = h.num(ord)
      const colSums = () => transposeM(m).map((c) => c.reduce((s, x) => s + Math.abs(x), 0))
      const rowSums = () => m.map((r) => r.reduce((s, x) => s + Math.abs(x), 0))
      if (p === 1) return Math.max(...colSums())
      if (p === -1) return Math.min(...colSums())
      if (p === Infinity) return Math.max(...rowSums())
      if (p === -Infinity) return Math.min(...rowSums())
      if (p === 2) return svdThin(m).s[0]
      if (p === -2) return Math.min(...svdThin(m).s)
      throw h.err('ValueError', 'Invalid norm order for matrices.')
    }
    if (!axis || axis.k === 'none') {
      if (!ord || ord.k === 'none') {
        const x = Math.sqrt(a.values().reduce((s, y) => s + y * y, 0))
        return keep ? mk([x], a.shape.map(() => 1), d) : one(x)
      }
      if (a.ndim === 1) return keep ? mk([vectorNorm(a.values())], [1], d) : one(vectorNorm(a.values()))
      if (a.ndim === 2) return keep ? mk([matrixNorm(rows(a))], [1, 1], d) : one(matrixNorm(rows(a)))
      throw h.err('ValueError', 'Improper number of dimensions to norm.')
    }
    if (isNum(axis)) {
      const ax = normAxis(h.toInt(axis), a.ndim)
      const res = lanes(a, ax, (lane) => [vectorNorm(lane)], d)
      if (keep) return wrap(res)
      const shape = a.shape.filter((_, i) => i !== ax)
      return shape.length ? wrap(reshape(res, shape)) : one(res.values()[0])
    }
    const [p, q] = h.iterate(axis).map((x) => normAxis(h.toInt(x), a.ndim))
    const moved = moveAxes(a, [p, q], [-2, -1])
    const [r, c] = moved.shape.slice(-2)
    const outer = moved.shape.slice(0, -2)
    const v = moved.values()
    const out = Array.from({ length: prod(outer) }, (_, b) => {
      const m = Array.from({ length: r }, (_x, i) => v.slice(b * r * c + i * c, b * r * c + i * c + c))
      return matrixNorm(m)
    })
    if (keep) return mk(out, a.shape.map((n, i) => (i === p || i === q ? 1 : n)), d)
    return outer.length ? mk(out, outer, d) : one(out[0])
  }
  const eigResult = (a: NDArray) => {
    const m = square(a, 'eig')
    for (const x of m.flat()) if (!Number.isFinite(x)) throw h.err('LinAlgError', 'Array must not contain infs or NaNs')
    return eigGeneral(m)
  }
  const complexOr = (re: number[], im: number[]): Value => (im.some((x) => x !== 0) ? { k: 'carray', re, im } : mk(re, [re.length]))
  Object.assign(linalg, {
    solve: la('solve', (args) => solveArr(A(args[0]), A(args[1], 'b')), 'linalg'),
    eig: la('eig', (args) => {
      const { re, im, vectors } = eigResult(A(args[0]))
      if (!vectors) throw h.err('NotImplementedError', 'this matrix has complex eigenvalues, and arrays here hold only real numbers: np.linalg.eigvals gives the values, or switch to Real Python for the vectors')
      return named('EigResult', [['eigenvalues', complexOr(re, im)], ['eigenvectors', matrix(vectors)]])
    }, 'linalg'),
    eigvals: la('eigvals', (args) => {
      const { re, im } = eigResult(A(args[0]))
      return complexOr(re, im)
    }, 'linalg'),
    eigh: la('eigh', (args) => {
      const { values, vectors } = eigh(square(A(args[0]), 'eigh'))
      return named('EighResult', [['eigenvalues', mk(values, [values.length])], ['eigenvectors', matrix(vectors)]])
    }, 'linalg'),
    eigvalsh: la('eigvalsh', (args) => {
      const { values } = eigh(square(A(args[0]), 'eigvalsh'))
      return mk(values, [values.length])
    }, 'linalg'),
    svd: la('svd', (args, kw) => {
      const a = A(args[0])
      const m = needs2d(a)
      const full = !(kw.full_matrices ?? args[1]) || h.truthy(kw.full_matrices ?? args[1])
      const wantUV = !(kw.compute_uv ?? args[2]) || h.truthy(kw.compute_uv ?? args[2])
      if (!wantUV) {
        const s = svdThin(m).s
        return mk(s, [s.length])
      }
      const { u, s, v } = full ? svdFull(m) : svdThin(m)
      return named('SVDResult', [['U', matrix(u)], ['S', mk(s, [s.length])], ['Vh', matrix(transposeM(v))]])
    }, 'linalg'),
    svdvals: la('svdvals', (args) => {
      const s = singularValues(A(args[0]))
      return mk(s, [s.length])
    }, 'linalg'),
    qr: la('qr', (args, kw) => {
      const m = needs2d(A(args[0]))
      const mode = (kw.mode ?? args[1]) as { v?: string } | undefined
      const { q, r } = qr(m)
      const [rowsN, colsN] = [m.length, m[0]?.length ?? 0]
      const k = Math.min(rowsN, colsN)
      if (mode?.v === 'r') return named('QRResult', [['Q', mk([], [0, 0])], ['R', matrix(r.slice(0, k), colsN)]].slice(1) as [string, Value][])
      if (mode?.v === 'complete') return named('QRResult', [['Q', matrix(q)], ['R', matrix(r, colsN)]])
      return named('QRResult', [['Q', matrix(q.map((row) => row.slice(0, k)), k)], ['R', matrix(r.slice(0, k), colsN)]])
    }, 'linalg'),
    cholesky: la('cholesky', (args, kw) => {
      const m = square(A(args[0]), 'cholesky')
      const l = cholesky(m)
      if (!l) throw h.err('LinAlgError', 'Matrix is not positive definite')
      return matrix(kw.upper && h.truthy(kw.upper) ? transposeM(l) : l)
    }, 'linalg'),
    lstsq: la('lstsq', (args, kw) => {
      const a = A(args[0])
      const b = A(args[1], 'b')
      const m = needs2d(a)
      const [rowsN, colsN] = [m.length, m[0]?.length ?? 0]
      const rc = opt(args, kw, 2, 'rcond')
      const { u, s, v } = svdThin(m)
      const cut = (rc ? h.num(rc) : 2 ** -52 * Math.max(rowsN, colsN)) * Math.max(0, ...s)
      const rank = s.filter((x) => x > cut).length
      const rhs = b.ndim === 1 ? b.values().map((x) => [x]) : rows(b)
      if (rhs.length !== rowsN) throw h.err('LinAlgError', 'Incompatible dimensions')
      const utb = mul(transposeM(u), rhs)
      const x = mul(v.map((row) => row.map((y, j) => (s[j] > cut ? y / s[j] : 0))), utb)
      const fit = mul(m, x)
      const resid = rank === colsN && rowsN > colsN
        ? rhs[0].map((_, c) => rhs.reduce((t, row, i) => t + (row[c] - fit[i][c]) ** 2, 0))
        : []
      return tuple([b.ndim === 1 ? mk(x.flat(), [colsN]) : matrix(x, b.shape[1]), mk(resid, [resid.length]), int(rank), mk(s, [s.length])])
    }, 'linalg'),
    pinv: la('pinv', (args, kw) => pinvOf(A(args[0]), optNum(args, kw, 1, 'rcond', 1e-15)), 'linalg'),
    matrix_rank: la('matrix_rank', (args, kw) => {
      const a = A(args[0])
      if (a.ndim < 2) return int(a.values().some((x) => x !== 0) ? 1 : 0)
      const s = singularValues(a)
      const tol = opt(args, kw, 1, 'tol')
      const cut = tol ? h.num(tol) : Math.max(0, ...s) * Math.max(...a.shape) * 2 ** -52
      return int(s.filter((x) => x > cut).length)
    }, 'linalg'),
    matrix_power: la('matrix_power', (args, kw) => {
      const a = A(args[0])
      let e = h.toInt(kw.n ?? args[1])
      const m = square(a, 'matrix_power')
      let base = e < 0 ? invM(m) : m
      e = Math.abs(e)
      let out = identity(m.length)
      while (e > 0) {
        if (e & 1) out = mul(out, base)
        base = mul(base, base)
        e >>= 1
      }
      return mk(out.flat(), a.shape, a.dtype === 'bool' ? 'bool' : a.dtype === 'int64' && h.toInt(kw.n ?? args[1]) >= 0 ? 'int64' : 'float64')
    }, 'linalg'),
    cond: la('cond', (args, kw) => {
      const a = A(args[0])
      const p = opt(args, kw, 1, 'p')
      if (!p || (isNum(p) && Math.abs(h.num(p)) === 2)) {
        const s = singularValues(a)
        const [hi, lo] = [Math.max(...s), Math.min(...s)]
        return float(p && h.num(p) < 0 ? lo / hi : hi / lo)
      }
      const m = square(a, 'cond')
      const once = (x: NDArray) => h.num(normOf(x, p, undefined, false))
      return float(once(a) * once(NDArray.create(invM(m).flat(), a.shape, 'float64')))
    }, 'linalg'),
    slogdet: la('slogdet', (args) => {
      const m = square(A(args[0]), 'slogdet')
      const { lu: f, sign, singular: sing } = lu(m)
      if (sing) return named('SlogdetResult', [['sign', float(0)], ['logabsdet', float(-Infinity)]])
      let s = sign
      let log = 0
      f.forEach((row, i) => {
        if (row[i] < 0) s = -s
        log += Math.log(Math.abs(row[i]))
      })
      return named('SlogdetResult', [['sign', float(s)], ['logabsdet', float(log)]])
    }, 'linalg'),
    multi_dot: la('multi_dot', (args, kw) => {
      void kw
      const arrs = h.iterate(args[0]).map((x) => toArray(x))
      if (arrs.length < 2) throw h.err('ValueError', 'Expecting at least two arrays.')
      const res = arrs.slice(1).reduce<NDArray | number>((acc, b) => {
        const left = typeof acc === 'number' ? NDArray.create([acc], [], 'float64') : acc
        return dot(left, b)
      }, arrs[0])
      return typeof res === 'number' ? scalar(res, commonDType(arrs)) : wrap(res)
    }, 'linalg'),
    norm: la('norm', (args, kw) => normOf(A(args[0], 'x'), opt(args, kw, 1, 'ord'), opt(args, kw, 2, 'axis'), flag(kw, 'keepdims') || (args[3] ? h.truthy(args[3]) : false)), 'linalg'),
    vector_norm: la('vector_norm', (args, kw) => {
      const a = A(args[0], 'x')
      const axis = opt(args, kw, 1, 'axis')
      const ordV = kw.ord ?? args[3] ?? float(2)
      if (!axis) return normOf(NDArray.create(a.values(), [a.size], a.dtype), ordV, undefined, false)
      return normOf(a, ordV, axis, flag(kw, 'keepdims'))
    }, 'linalg'),
    matrix_norm: la('matrix_norm', (args, kw) => normOf(A(args[0], 'x'), opt(args, kw, 1, 'ord') ?? S('fro'), tuple([int(-2), int(-1)]), flag(kw, 'keepdims')), 'linalg'),
    matmul: np.matmul, outer: np.outer, trace: np.trace, diagonal: np.diagonal, cross: np.cross, tensordot: np.tensordot, vecdot: np.vecdot, matrix_transpose: np.matrix_transpose,
  })

  // ---------- np.random ----------

  const sized = (size: Value | undefined, draw: () => number, d: DType = 'float64'): Value => {
    if (!size || size.k === 'none') return scalar(draw(), d)
    const shape = shapeOf(size)
    checkSize(shape)
    return mk(Array.from({ length: prod(shape) }, draw), shape, d)
  }
  /** a distribution taking scalar or array parameters, with size last */
  const dist = (r: () => SandboxRandom, api2: Api, name: string, params: [string, number?][], f: (rng: SandboxRandom, ...p: number[]) => number, d: DType = 'float64') =>
    api2(name, (args, kw) => {
      const vals = params.map(([p, dflt], i) => {
        const v = kw[p] ?? args[i]
        if (v && v.k !== 'none') return toArray(v)
        if (dflt === undefined) throw h.err('TypeError', `${name}() missing required argument '${p}'`)
        return NDArray.create([dflt], [], 'float64')
      })
      const size = kw.size ?? args[params.length]
      const common = broadcastShapes(vals.map((v) => v.shape))
      const shape = size && size.k !== 'none' ? shapeOf(size) : common
      checkSize(shape)
      const cols = vals.map((v) => broadcastTo(v, shape).values())
      const out = Array.from({ length: prod(shape) }, (_, i) => f(r(), ...cols.map((c) => c[i])))
      return shape.length ? mk(out, shape, d) : scalar(out[0], d)
    }, 'random')
  /** np.random.choice and Generator.choice */
  const choose = (r: SandboxRandom, args: Value[], kw: Kw, generator: boolean): Value => {
    const src = args[0]
    const pool = isNum(src) ? null : toArray(src)
    const n = pool ? (pool.ndim ? pool.shape[0] : 0) : h.toInt(src)
    if (n <= 0) throw h.err('ValueError', 'a must be a positive integer or a non-empty sequence')
    const size = kw.size ?? args[1]
    const repl = kw.replace ?? args[2]
    const replace = !repl || h.truthy(repl)
    const pv = kw.p ?? args[3]
    const p = pv && pv.k !== 'none' ? toArray(pv).values() : null
    if (p) {
      if (p.length !== n) throw h.err('ValueError', "'a' and 'p' must have same size")
      if (p.some((x) => x < 0)) throw h.err('ValueError', 'probabilities are not non-negative')
      if (Math.abs(p.reduce((s, x) => s + x, 0) - 1) > 1e-8) throw h.err('ValueError', 'probabilities do not sum to 1')
    }
    const shape = size && size.k !== 'none' ? shapeOf(size) : []
    const count = prod(shape)
    checkSize(shape)
    const pickWeighted = (weights: number[]) => {
      const total = weights.reduce((s, x) => s + x, 0)
      let u = r.next() * total
      for (let i = 0; i < weights.length; i++) {
        u -= weights[i]
        if (u < 0) return i
      }
      return weights.length - 1
    }
    let picks: number[]
    if (r instanceof LegacyRandom) {
      if (!replace && count > n) throw h.err('ValueError', "Cannot take a larger sample than population when 'replace=False'")
      if (!replace && p && p.filter((x) => x > 0).length < count) throw h.err('ValueError', 'Fewer non-zero entries in p than size')
      picks = r.choice(n, count, replace, p)
    } else if (replace) picks = Array.from({ length: count }, () => (p ? pickWeighted(p) : Math.floor(r.next() * n)))
    else {
      if (count > n) throw h.err('ValueError', "Cannot take a larger sample than population when 'replace=False'")
      if (p) {
        const left = [...p]
        picks = Array.from({ length: count }, () => {
          const i = pickWeighted(left)
          left[i] = 0
          return i
        })
      } else picks = R.permutationOf(r, n).slice(0, count)
    }
    if (!generator && pool && pool.ndim > 1) throw h.err('ValueError', 'a must be 1-dimensional')
    if (!pool) return shape.length ? mk(picks, shape, 'int64') : int(picks[0])
    if (pool.ndim > 1) {
      const res = takeAxis(pool, NDArray.create(picks, shape.length ? shape : [], 'int64'), 0, 'raise')
      return res
    }
    const v = pool.values()
    return shape.length ? mk(picks.map((i) => v[i]), shape, pool.dtype) : scalar(v[picks[0]], pool.dtype)
  }
  const shuffleIn = (r: SandboxRandom, v: Value) => {
    if (v.k === 'list') {
      const order = R.permutationOf(r, v.items.length)
      const old = [...v.items]
      order.forEach((from, to) => { v.items[to] = old[from] })
      return NONE
    }
    const a = toArray(v)
    if (!a.ndim) throw h.err('TypeError', 'object of type numpy.ndarray has no len() (0-d array)')
    const n = a.shape[0]
    const unit = a.size / Math.max(n, 1)
    const old = a.values()
    const out: number[] = []
    for (const i of R.permutationOf(r, n)) out.push(...old.slice(i * unit, i * unit + unit))
    write(a, out)
    return NONE
  }
  const permuted = (r: SandboxRandom, v: Value, axisV?: Value): Value => {
    if (isNum(v)) return mk(R.permutationOf(r, h.toInt(v)), [h.toInt(v)], 'int64')
    const a = toArray(v)
    const ax = axisV ? normAxis(h.toInt(axisV), a.ndim) : 0
    return takeAxis(a, NDArray.create(R.permutationOf(r, a.shape[ax]), [a.shape[ax]], 'int64'), ax, 'raise')
  }
  const integers = (r: SandboxRandom, lo: number, hi: number, size: Value | undefined, d: DType): Value => {
    if (hi <= lo) throw h.err('ValueError', 'high <= low')
    return sized(size, () => lo + r.interval(hi - lo - 1), d)
  }
  const mvn = (r: SandboxRandom, args: Value[], kw: Kw): Value => {
    const mean = A(kw.mean ?? args[0], 'mean').values()
    const cov = rows(A(kw.cov ?? args[1], 'cov'))
    const size = kw.size ?? args[2]
    const n = mean.length
    if (cov.length !== n) throw h.err('ValueError', 'mean and cov must have same length')
    // the root of a positive semi-definite cov, from its eigenvalues
    const { values, vectors } = eigh(cov)
    const root = vectors.map((row) => row.map((x, j) => x * Math.sqrt(Math.max(0, values[j]))))
    const draws = size && size.k !== 'none' ? shapeOf(size) : []
    const total = prod(draws)
    checkSize([...draws, n])
    const out: number[] = []
    for (let t = 0; t < total; t++) {
      const z = mean.map(() => r.normal())
      for (let i = 0; i < n; i++) out.push(mean[i] + root[i].reduce((s, x, j) => s + x * z[j], 0))
    }
    return mk(out, [...draws, n])
  }
  const multinom = (r: SandboxRandom, args: Value[], kw: Kw): Value => {
    const n = h.toInt(kw.n ?? args[0])
    const ps = A(kw.pvals ?? args[1], 'pvals').values()
    const size = kw.size ?? args[2]
    const draws = size && size.k !== 'none' ? shapeOf(size) : []
    const total = prod(draws)
    const out: number[] = []
    for (let t = 0; t < total; t++) out.push(...R.multinomial(r, n, ps))
    return mk(out, [...draws, ps.length], 'int64')
  }
  const dirich = (r: SandboxRandom, args: Value[], kw: Kw): Value => {
    const al = A(kw.alpha ?? args[0], 'alpha').values()
    const size = kw.size ?? args[1]
    const draws = size && size.k !== 'none' ? shapeOf(size) : []
    const total = prod(draws)
    const out: number[] = []
    for (let t = 0; t < total; t++) out.push(...R.dirichlet(r, al))
    return mk(out, [...draws, al.length])
  }
  const distributions = (r: () => SandboxRandom, a: Api): Record<string, Value> => ({
    exponential: dist(r, a, 'exponential', [['scale', 1]], (g, s) => s * R.exponential(g)),
    standard_exponential: a('standard_exponential', (args, kw) => sized(kw.size ?? args[0], () => R.exponential(r())), 'random'),
    gamma: dist(r, a, 'gamma', [['shape'], ['scale', 1]], (g, k, s) => s * R.gamma(g, k)),
    standard_gamma: dist(r, a, 'standard_gamma', [['shape']], (g, k) => R.gamma(g, k)),
    beta: dist(r, a, 'beta', [['a'], ['b']], (g, x, y) => R.beta(g, x, y)),
    chisquare: dist(r, a, 'chisquare', [['df']], (g, df) => R.chisquare(g, df)),
    f: dist(r, a, 'f', [['dfnum'], ['dfden']], (g, n, m) => R.chisquare(g, n) / n / (R.chisquare(g, m) / m)),
    standard_t: dist(r, a, 'standard_t', [['df']], (g, df) => R.studentT(g, df)),
    standard_cauchy: a('standard_cauchy', (args, kw) => sized(kw.size ?? args[0], () => Math.tan(Math.PI * (r().next() - 0.5))), 'random'),
    laplace: dist(r, a, 'laplace', [['loc', 0], ['scale', 1]], (g, m, s) => {
      const u = g.next() - 0.5
      return m - s * Math.sign(u) * Math.log(1 - 2 * Math.abs(u))
    }),
    logistic: dist(r, a, 'logistic', [['loc', 0], ['scale', 1]], (g, m, s) => {
      const u = g.next()
      return m + s * Math.log(u / (1 - u))
    }),
    lognormal: dist(r, a, 'lognormal', [['mean', 0], ['sigma', 1]], (g, m, s) => Math.exp(m + s * g.normal())),
    gumbel: dist(r, a, 'gumbel', [['loc', 0], ['scale', 1]], (g, m, s) => m - s * Math.log(-Math.log(1 - g.next()))),
    rayleigh: dist(r, a, 'rayleigh', [['scale', 1]], (g, s) => s * Math.sqrt(-2 * Math.log(1 - g.next()))),
    pareto: dist(r, a, 'pareto', [['a']], (g, x) => (1 - g.next()) ** (-1 / x) - 1),
    power: dist(r, a, 'power', [['a']], (g, x) => (1 - (1 - g.next())) ** (1 / x)),
    weibull: dist(r, a, 'weibull', [['a']], (g, x) => (-Math.log(1 - g.next())) ** (1 / x)),
    triangular: dist(r, a, 'triangular', [['left'], ['mode'], ['right']], (g, x, m, y) => R.triangular(g, x, m, y)),
    poisson: dist(r, a, 'poisson', [['lam', 1]], (g, l) => R.poisson(g, l), 'int64'),
    binomial: dist(r, a, 'binomial', [['n'], ['p']], (g, n, p) => R.binomial(g, n, p), 'int64'),
    negative_binomial: dist(r, a, 'negative_binomial', [['n'], ['p']], (g, n, p) => R.poisson(g, R.gamma(g, n) * ((1 - p) / p)), 'int64'),
    geometric: dist(r, a, 'geometric', [['p']], (g, p) => R.geometric(g, p), 'int64'),
    multinomial: a('multinomial', (args, kw) => multinom(r(), args, kw), 'random'),
    multivariate_normal: a('multivariate_normal', (args, kw) => mvn(r(), args, kw), 'random'),
    dirichlet: a('dirichlet', (args, kw) => dirich(r(), args, kw), 'random'),
  })
  const legacyDraw = (name: string) => rnd(name, (args, kw) => sized(kw.size ?? args[0], () => h.npRng.next()), 'random')
  /** get_state() as NumPy gives it: ('MT19937', the 624 words, position, whether a normal is cached, the cached normal) */
  const stateOf = (r: LegacyRandom): Value => {
    const st = r.getState()
    return tuple([S('MT19937'), mk(st.key, [st.key.length], 'int64'), int(st.pos), int(st.gauss === null ? 0 : 1), py.float(st.gauss ?? 0)])
  }
  const setStateOf = (r: LegacyRandom, v: Value): Value => {
    const parts = h.iterate(v)
    if (parts.length < 3 || h.str(parts[0]) !== 'MT19937') throw h.err('ValueError', 'state must be a 5-tuple (MT19937, key, pos, has_gauss, cached_gaussian)')
    const key = toArray(parts[1]).values()
    if (key.length !== 624) throw h.err('ValueError', 'state must be 624 elements')
    const has = parts[3] ? h.truthy(parts[3]) : false
    r.setState({ key, pos: h.toInt(parts[2]), gauss: has && parts[4] ? h.num(parts[4]) : null })
    return NONE
  }
  Object.assign(random, distributions(() => h.npRng, rnd), {
    random: legacyDraw('random'), random_sample: legacyDraw('random_sample'), ranf: legacyDraw('ranf'), sample: legacyDraw('sample'),
    standard_normal: rnd('standard_normal', (args, kw) => sized(kw.size ?? args[0], () => h.npRng.normal()), 'random'),
    choice: rnd('choice', (args, kw) => choose(h.npRng, args, kw, false), 'random'),
    shuffle: rnd('shuffle', (args) => shuffleIn(h.npRng, args[0])),
    permutation: rnd('permutation', (args) => permuted(h.npRng, args[0]), 'random'),
    random_integers: rnd('random_integers', (args, kw) => {
      // (the old function truncated floats)
      const lo = Math.trunc(h.num(args[0]))
      const hi = opt(args, kw, 1, 'high')
      return hi ? integers(h.npRng, lo, Math.trunc(h.num(hi)) + 1, kw.size ?? args[2], 'int64') : integers(h.npRng, 1, lo + 1, kw.size ?? args[2], 'int64')
    }, 'random'),
    get_state: rnd('get_state', () => stateOf(h.npRng)),
    set_state: rnd('set_state', (args, kw) => setStateOf(h.npRng, kw.state ?? args[0])),
  })

  /** np.random.default_rng(seed) and np.random.RandomState(seed): each has its own stream */
  class RandomObj extends Base {
    readonly cls: string
    r: SandboxRandom
    table: Record<string, Value>
    constructor(cls: string, r: SandboxRandom, table: (self: RandomObj) => Record<string, Value>) {
      super()
      this.cls = cls
      this.r = r
      this.table = table(this)
    }
    repr() { return this.cls === 'Generator' ? 'Generator(PCG64)' : '<RandomState>' }
    getAttr(name: string) { return this.table[name] }
    /** isinstance(rng, np.random.Generator) */
    isa(name: string) { return name === `numpy.random.${this.cls}` }
  }
  const Rng = h.rng.constructor as new (seed?: number) => SandboxRandom
  /** PCG64(5), MT19937(5), SeedSequence(5): only the seed matters here, as the sandbox has one generator */
  class SeedHolder extends Base {
    readonly cls: string
    seed: Value | undefined
    constructor(cls: string, seed: Value | undefined) {
      super()
      this.cls = cls
      this.seed = seed
    }
    repr() { return this.cls === 'SeedSequence' ? `SeedSequence(\n    entropy=${this.seed ? repr(this.seed) : 'None'},\n)` : `<numpy.random._${this.cls.toLowerCase()}.${this.cls} object>` }
  }
  const seedOf = (v0: Value | undefined) => {
    const v = v0?.k === 'obj' && v0.o instanceof SeedHolder ? v0.o.seed : v0
    if (!v || v.k === 'none') return Math.floor(h.rng.next() * 2 ** 32)
    return isNum(v) ? h.toInt(v) : toArray(v).values().reduce((s, x) => (s * 31 + x) >>> 0, 7)
  }
  const genApi: Api = (name, call) => ({ k: 'fn', name, call })
  const sharedDraws = (self: RandomObj): Record<string, Value> => ({
    ...distributions(() => self.r, genApi),
    choice: genApi('choice', (a, kw) => choose(self.r, a, kw, true)),
    shuffle: genApi('shuffle', (a) => shuffleIn(self.r, a[0])),
    permutation: genApi('permutation', (a, kw) => permuted(self.r, a[0], kw.axis)),
    standard_normal: genApi('standard_normal', (a, kw) => sized(kw.size ?? a[0], () => self.r.normal())),
    normal: dist(() => self.r, genApi, 'normal', [['loc', 0], ['scale', 1]], (g, m, s) => m + s * g.normal()),
    uniform: dist(() => self.r, genApi, 'uniform', [['low', 0], ['high', 1]], (g, x, y) => x + (y - x) * g.next()),
  })
  const generator = (seed: Value | undefined): Value => py.obj(new RandomObj('Generator', new Rng(seedOf(seed)), (self) => ({
    ...sharedDraws(self),
    random: genApi('random', (a, kw) => sized(kw.size ?? a[0], () => self.r.next())),
    integers: genApi('integers', (a, kw) => {
      const lo = h.toInt(kw.low ?? a[0])
      const hiV = opt(a, kw, 1, 'high')
      const end = kw.endpoint && h.truthy(kw.endpoint) ? 1 : 0
      const [from, to] = hiV ? [lo, h.toInt(hiV) + end] : [0, lo + end]
      return integers(self.r, from, to, kw.size ?? a[2], 'int64')
    }),
    permuted: genApi('permuted', (a, kw) => {
      const x = toArray(a[0])
      const axis = opt(a, kw, 1, 'axis')
      if (!axis) {
        const v = x.values()
        return mk(R.permutationOf(self.r, v.length).map((i) => v[i]), x.shape, x.dtype)
      }
      return wrap(lanes(x, h.toInt(axis), (lane) => R.permutationOf(self.r, lane.length).map((i) => lane[i]), x.dtype))
    }),
  })))
  const randomState = (seed: Value | undefined): Value => py.obj(new RandomObj('RandomState', new LegacyRandom(h.legacySeed(seed?.k === 'obj' && seed.o instanceof SeedHolder ? seed.o.seed : seed)), (self) => {
    const mt = self.r as LegacyRandom
    const floats = (a: Value[]) => {
      const shape = a.map((x) => h.toInt(x))
      return sized(shape.length ? L(shape.map(int)) : undefined, () => self.r.next())
    }
    return {
      ...sharedDraws(self),
      seed: genApi('seed', (a, kw) => { mt.seed(h.legacySeed(kw.seed ?? a[0])); return NONE }),
      get_state: genApi('get_state', () => stateOf(mt)),
      set_state: genApi('set_state', (a, kw) => setStateOf(mt, kw.state ?? a[0])),
      rand: genApi('rand', floats),
      randn: genApi('randn', (a) => {
        const shape = a.map((x) => h.toInt(x))
        return sized(shape.length ? L(shape.map(int)) : undefined, () => self.r.normal())
      }),
      random: genApi('random', (a, kw) => sized(kw.size ?? a[0], () => self.r.next())),
      random_sample: genApi('random_sample', (a, kw) => sized(kw.size ?? a[0], () => self.r.next())),
      randint: genApi('randint', (a, kw) => {
        const lo = h.toInt(kw.low ?? a[0])
        const hiV = opt(a, kw, 1, 'high')
        return hiV ? integers(self.r, lo, h.toInt(hiV), kw.size ?? a[2], 'int64') : integers(self.r, 0, lo, kw.size ?? a[2], 'int64')
      }),
    }
  }))
  Object.assign(random, {
    default_rng: rnd('default_rng', (args, kw) => generator(kw.seed ?? args[0])),
    Generator: { k: 'type', name: 'numpy.random.Generator', call: (args: Value[]) => generator(args[0]) },
    RandomState: { k: 'type', name: 'numpy.random.RandomState', call: (args: Value[], kw: Kw) => randomState(kw.seed ?? args[0]) },
    PCG64: { k: 'type', name: 'PCG64', call: (args: Value[], kw: Kw) => py.obj(new SeedHolder('PCG64', kw.seed ?? args[0])) },
    MT19937: { k: 'type', name: 'MT19937', call: (args: Value[], kw: Kw) => py.obj(new SeedHolder('MT19937', kw.seed ?? args[0])) },
    SeedSequence: { k: 'type', name: 'SeedSequence', call: (args: Value[], kw: Kw) => py.obj(new SeedHolder('SeedSequence', kw.entropy ?? args[0])) },
  })
}

/** a.partition(), a.take() …: the array methods that are the NumPy function with the array first */
export const ARRAY_METHODS_VIA_FUNCTION: Record<string, ApiKind> = {
  take: 'move', choose: 'move', diagonal: 'move', searchsorted: 'scan', cumprod: 'scan', conj: 'elementwise', conjugate: 'elementwise',
}
