/**
 * OpenCV for the sandbox: colour conversion, resizing, geometric warps, filtering, thresholding, histograms, morphology
 * and saturating arithmetic on NumPy arrays, following OpenCV's own fixed-point formulas so uint8 results match bit for bit.
 *
 * NDArray knows uint8, int64 and float64. uint8 is CV_8U. int64 becomes CV_32S, as it does in the real binding, and
 * float64 stands for both CV_32F and CV_64F. A CV_16S result from Sobel and the like is tagged, so saturation stays right.
 * There are no files and no windows here, so imread, imshow and friends raise an error that says so.
 */
import { NDArray, cStrides } from './ndarray'
import { isNum, numOf, py, toArray } from './minipy'
import type { ApiKind, ClassValue, Host, Kw, PyLib, Value } from './minipy'

type Depth = '8u' | '8s' | '16u' | '16s' | '32s' | '32f' | '64f'
interface Img { h: number; w: number; c: number; v: number[]; dp: Depth }
type Args = Record<string, Value | undefined>

const DEPTH_CODES: Depth[] = ['8u', '8s', '16u', '16s', '32s', '32f', '64f']
const INT_RANGE: Partial<Record<Depth, [number, number]>> = { '8u': [0, 255], '8s': [-128, 127], '16u': [0, 65535], '16s': [-32768, 32767], '32s': [-2147483648, 2147483647] }
const isFloatDepth = (d: Depth) => d === '32f' || d === '64f'
const { fround } = Math

/** cvRound: to the nearest integer, ties to even */
const cvRound = (x: number): number => {
  const r = Math.round(x)
  return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r
}

/** saturate_cast<dp> */
const sat = (x: number, dp: Depth): number => {
  if (dp === '64f') return x
  if (dp === '32f') return fround(x)
  const [lo, hi] = INT_RANGE[dp]!
  if (Number.isNaN(x)) return 0
  const r = cvRound(x)
  return r < lo ? lo : r > hi ? hi : r
}

/** the depth a cv2 result was made with, when the dtype alone cannot say (CV_16S and CV_32F results) */
const TAGS = new WeakMap<NDArray, Depth>()

const depthOf = (a: NDArray): Depth => TAGS.get(a) ?? (a.dtype === 'float64' ? '64f' : a.dtype === 'uint8' ? '8u' : '32s')

const errorClass = (h: Host): ClassValue => {
  let c = h.state.get('cv2.error') as ClassValue | undefined
  if (!c) {
    const base = h.excClass('Exception')
    c = { k: 'class', name: 'error', bases: [base], ns: new Map(), host: base.host, builtin: true }
    h.state.set('cv2.error', c)
  }
  return c
}

const cvErr = (h: Host, kind: string, msg: string): never => h.raise(errorClass(h), [py.str(`OpenCV(4.13.0) (${kind}) ${msg}`)])

/** what the real library refuses on CV_32S, which is where a plain int64 array ends up */
const no32s = (h: Host, im: Img, fn: string) => {
  if (im.dp === '32s') cvErr(h, '-210:Unsupported format or combination of formats', `in function '${fn}': the array has dtype int64, which OpenCV reads as CV_32S. An image needs dtype=np.uint8`)
}

const asMat = (h: Host, v: Value | undefined, what: string): NDArray => {
  if (v && v.k === 'array') return v.a
  if (v && v.k === 'obj' && v.o.toArray) return v.o.toArray()
  throw cvErr(h, 'Bad argument', `'${what}' is not a numpy array, neither a scalar`)
}

const readImg = (h: Host, v: Value | undefined, what = 'src'): Img => {
  const a = asMat(h, v, what)
  if (a.dtype === 'bool') throw cvErr(h, 'Bad argument', `${what} data type = bool is not supported`)
  if (a.ndim !== 2 && a.ndim !== 3) throw cvErr(h, 'Bad argument', `${what} must be a 2-D or 3-D array, got ${a.ndim}-D`)
  if (a.size === 0) throw cvErr(h, '-215:Assertion failed', '!_src.empty()')
  const dp = depthOf(a)
  // the binding casts int64 to int32, wrapping what does not fit
  return { h: a.shape[0], w: a.shape[1], c: a.ndim === 3 ? a.shape[2] : 1, v: dp === '32s' && !TAGS.has(a) ? a.values().map((x) => x | 0) : a.values(), dp }
}

/** a Mat with one channel comes back as 2-D, whatever the input looked like */
const toValue = (im: { h: number; w: number; c: number; v: number[] }, dp: Depth): Value => {
  const shape = im.c === 1 ? [im.h, im.w] : [im.h, im.w, im.c]
  const a = NDArray.create(im.v, shape, isFloatDepth(dp) ? 'float64' : dp === '8u' ? 'uint8' : 'int64')
  if (dp !== '8u' && dp !== '64f') TAGS.set(a, dp)
  return py.arr(a)
}

/** the same depth as the source, for results that only move its values around */
const sameKind = (src: NDArray, values: number[], shape: number[]): Value => {
  const out = NDArray.create(values, shape, src.dtype)
  const tag = TAGS.get(src)
  if (tag) TAGS.set(out, tag)
  return py.arr(out)
}

const blank = (im: { h: number; w: number; c: number }, fill = 0): number[] => new Array(im.h * im.w * im.c).fill(fill)

const need8u = (h: Host, im: Img, fn: string) => {
  if (im.dp !== '8u') throw cvErr(h, '-215:Assertion failed', `_src.depth() == CV_8U in function '${fn}'${im.dp === '32s' ? ': an int64 array reads as CV_32S, an image needs dtype=np.uint8' : ''}`)
}

// ---------------------------------------------------------------- borders

const BORDER = { CONSTANT: 0, REPLICATE: 1, REFLECT: 2, WRAP: 3, REFLECT_101: 4, TRANSPARENT: 5, ISOLATED: 16 }

/** cv::borderInterpolate: which row or column stands in for p; -1 = the constant border */
const borderIdx = (p0: number, len: number, type: number): number => {
  let p = p0
  if (p >= 0 && p < len) return p
  switch (type) {
    case BORDER.REPLICATE: return p < 0 ? 0 : len - 1
    case BORDER.REFLECT: case BORDER.REFLECT_101: {
      const delta = type === BORDER.REFLECT_101 ? 1 : 0
      if (len === 1) return 0
      do {
        p = p < 0 ? -p - 1 + delta : len - 1 - (p - len) - delta
      } while (p < 0 || p >= len)
      return p
    }
    case BORDER.WRAP:
      if (p < 0) p -= Math.trunc((p - len + 1) / len) * len
      return p >= len ? p % len : p
    default: return -1
  }
}

const borderType = (h: Host, v: Value | undefined, dflt: number): number => {
  const t = (v && v.k !== 'none' ? h.toInt(v) : dflt) & ~BORDER.ISOLATED
  if (t < 0 || t > 4) throw cvErr(h, '-5:Bad argument', `Unknown/unsupported border type ${t}`)
  return t
}

// ---------------------------------------------------------------- arguments

const intPair = (h: Host, v: Value | undefined, what: string): [number, number] => {
  if (v && (v.k === 'tuple' || v.k === 'list') && v.items.length === 2 && v.items.every((x) => x.k === 'int' || x.k === 'bool' || (x.k === 'array' && x.a.dtype !== 'float64' && x.a.size === 1))) {
    return [h.num(v.items[0]), h.num(v.items[1])]
  }
  throw cvErr(h, '-5:Bad argument', `Can't parse '${what}'. ${v && (v.k === 'tuple' || v.k === 'list') && v.items.length === 2 ? 'Sequence item with index 0 has a wrong type' : `Expected sequence length 2, got ${v && (v.k === 'tuple' || v.k === 'list') ? v.items.length : 1}`}`)
}

const numArg = (h: Host, v: Value | undefined, dflt: number): number => (v && v.k !== 'none' ? h.num(v) : dflt)
const intArg = (h: Host, v: Value | undefined, dflt: number): number => (v && v.k !== 'none' ? h.toInt(v) : dflt)

/** an output depth argument: -1 or absent = the input's; CV_8UC1 etc. carry the depth in their low bits */
const depthArg = (h: Host, v: Value | undefined, src: Depth): Depth => {
  const d = intArg(h, v, -1)
  return d < 0 ? src : DEPTH_CODES[d & 7] ?? src
}

/** a Scalar: up to 4 numbers, the missing ones 0 */
const scalarOf = (h: Host, v: Value | undefined): number[] => {
  if (!v || v.k === 'none') return [0, 0, 0, 0]
  if (isNum(v)) return [numOf(v), 0, 0, 0]
  const xs = v.k === 'array' ? v.a.values() : h.iterate(v).map((x) => h.num(x))
  if (xs.length > 4) throw cvErr(h, '-5:Bad argument', 'Scalar value must have at most 4 elements')
  return [...xs, 0, 0, 0, 0].slice(0, 4)
}

const kernelOf = (h: Host, v: Value | undefined, what = 'kernel'): { kh: number; kw: number; k: number[] } => {
  const a = v && v.k !== 'array' && v.k !== 'none' ? toArray(v) : asMat(h, v, what)
  if (a.ndim === 1) return { kh: a.shape[0], kw: 1, k: a.values() }
  if (a.ndim !== 2) throw cvErr(h, '-215:Assertion failed', `${what} must be 2-D`)
  return { kh: a.shape[0], kw: a.shape[1], k: a.values() }
}

const anchorOf = (h: Host, v: Value | undefined, kw: number, kh: number): [number, number] => {
  const [x, y] = v && v.k !== 'none' ? intPair(h, v, 'anchor') : [-1, -1]
  const ax = x === -1 ? kw >> 1 : x
  const ay = y === -1 ? kh >> 1 : y
  if (ax < 0 || ax >= kw || ay < 0 || ay >= kh) throw cvErr(h, '-215:Assertion failed', 'anchor is outside the kernel')
  return [ax, ay]
}

// ---------------------------------------------------------------- filtering

/** look up pixels through a border: px(row, col, channel) */
const sampler = (im: { h: number; w: number; c: number; v: number[] }, border: number, cval: number[] = [0, 0, 0, 0]) => {
  const { h: H, w: W, c: C, v } = im
  return (y: number, x: number, c: number): number => {
    const yy = y >= 0 && y < H ? y : borderIdx(y, H, border)
    const xx = x >= 0 && x < W ? x : borderIdx(x, W, border)
    return yy < 0 || xx < 0 ? cval[c] ?? 0 : v[(yy * W + xx) * C + c]
  }
}

/** Filter2D: delta plus each non-zero coefficient times its pixel, in float32 unless a side is CV_64F */
const correlate = (im: Img, kw: number, k0: number[], ax: number, ay: number, delta: number, border: number, dp: Depth): Img => {
  const wide = im.dp === '64f' || dp === '64f'
  const coef = k0.flatMap((val, i) => (val !== 0 ? [{ dy: Math.floor(i / kw) - ay, dx: i % kw - ax, val: wide ? val : fround(val) }] : []))
  const px = sampler(im, border)
  const out = blank(im)
  const start = wide ? delta : fround(delta)
  let at = 0
  for (let y = 0; y < im.h; y++) {
    for (let x = 0; x < im.w; x++) {
      for (let c = 0; c < im.c; c++) {
        let s = start
        if (wide) for (const q of coef) s += q.val * px(y + q.dy, x + q.dx, c)
        else for (const q of coef) s = fround(s + fround(q.val * px(y + q.dy, x + q.dx, c)))
        out[at++] = sat(s, dp)
      }
    }
  }
  return { ...im, v: out, dp }
}

/** separable filtering in double precision; kx runs along the row, ky along the column */
const sepFilter = (im: Img, kx: number[], ky: number[], delta: number, border: number, dp: Depth, ax = kx.length >> 1, ay = ky.length >> 1): Img => {
  const px = sampler(im, border)
  const rows = im.h + ky.length - 1
  const tmp = new Array<number>(rows * im.w * im.c)
  const single = dp === '32f'
  let at = 0
  for (let r = 0; r < rows; r++) {
    const y = r - ay
    for (let x = 0; x < im.w; x++) {
      for (let c = 0; c < im.c; c++) {
        let s = 0
        for (let i = 0; i < kx.length; i++) s += kx[i] * px(y, x + i - ax, c)
        tmp[at++] = single ? fround(s) : s
      }
    }
  }
  const out = blank(im)
  const line = im.w * im.c
  at = 0
  for (let y = 0; y < im.h; y++) {
    for (let j = 0; j < line; j++) {
      let s = delta
      for (let i = 0; i < ky.length; i++) s += ky[i] * tmp[(y + i) * line + j]
      out[at++] = sat(s, dp)
    }
  }
  return { ...im, v: out, dp }
}

const SMALL_GAUSS: Record<number, number[]> = { 1: [1], 3: [0.25, 0.5, 0.25], 5: [0.0625, 0.25, 0.375, 0.25, 0.0625], 7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125], 9: [4, 13, 30, 51, 60, 51, 30, 13, 4].map((x) => x / 256) }

/** getGaussianKernel */
const gaussKernel = (n: number, sigma: number): number[] => {
  if (sigma <= 0 && SMALL_GAUSS[n]) return SMALL_GAUSS[n]
  const sx = sigma > 0 ? sigma : ((n - 1) * 0.5 - 1) * 0.3 + 0.8
  const scale2 = -0.5 / (sx * sx)
  const t = Array.from({ length: n }, (_, i) => Math.exp(scale2 * (i - (n - 1) * 0.5) ** 2))
  const sum = 1 / t.reduce((a, b) => a + b, 0)
  return t.map((x) => x * sum)
}

/** the 8.8 fixed-point kernel GaussianBlur uses on uint8: rounded with error diffusion, the centre takes what is left */
const gaussFixed = (n: number, sigma: number): number[] => {
  const k = gaussKernel(n, sigma)
  const out = new Array<number>(n)
  let err = 0
  let sum = 0
  const half = n >> 1
  for (let i = 0; i < half; i++) {
    const adj = k[i] * 256 + err
    const v = cvRound(adj)
    err = adj - v
    out[i] = out[n - 1 - i] = v
    sum += v
  }
  out[half] = 256 - 2 * sum
  return out
}

const gaussianSize = (h: Host, ksize: [number, number], sx: number, sy0: number, dp: Depth): [number, number, number] => {
  const sy = sy0 > 0 ? sy0 : sx
  const scale = dp === '8u' ? 3 : 4
  let [kw, kh] = ksize
  if (kw <= 0 && sx > 0) kw = cvRound(sx * scale * 2 + 1) | 1
  if (kh <= 0 && sy > 0) kh = cvRound(sy * scale * 2 + 1) | 1
  if (kw <= 0 || kw % 2 === 0 || kh <= 0 || kh % 2 === 0) throw cvErr(h, '-215:Assertion failed', 'ksize.width > 0 && ksize.width % 2 == 1 && ksize.height > 0 && ksize.height % 2 == 1')
  return [kw, kh, sy]
}

const gaussianBlur = (h: Host, im: Img, ksize: [number, number], sx: number, sy0: number, border: number): Img => {
  const [kw, kh, sy] = gaussianSize(h, ksize, sx, sy0, im.dp)
  if (im.dp !== '8u') return sepFilter(im, gaussKernel(kw, Math.max(sx, 0)), gaussKernel(kh, Math.max(sy, 0)), 0, border, im.dp)
  if (kw === 1 && kh === 1) return { ...im, v: [...im.v] }
  const fx = gaussFixed(kw, Math.max(sx, 0))
  const fy = gaussFixed(kh, Math.max(sy, 0))
  const px = sampler(im, border)
  const ax = kw >> 1
  const ay = kh >> 1
  const out = blank(im)
  let at = 0
  for (let y = 0; y < im.h; y++) {
    for (let x = 0; x < im.w; x++) {
      for (let c = 0; c < im.c; c++) {
        let s = 0
        for (let i = 0; i < kh; i++) {
          let row = 0
          for (let j = 0; j < kw; j++) row += fx[j] * px(y + i - ay, x + j - ax, c)
          s += fy[i] * row
        }
        out[at++] = Math.floor((s + 32768) / 65536)
      }
    }
  }
  return { ...im, v: out }
}

/** boxFilter: the window's sum, divided by its area when normalize is set. A small uint8 window divides in 23-bit fixed point (ColumnSum<ushort, uchar>) */
const boxFilter = (im: Img, kw: number, kh: number, ax: number, ay: number, border: number, normalize: boolean, dp: Depth): Img => {
  const px = sampler(im, border)
  const area = kw * kh
  const scale = normalize ? 1 / area : 1
  const fixed = normalize && area > 1 && area <= 256 && im.dp === '8u' && dp === '8u'
  const exact = 8388608 / area
  const up = exact - Math.floor(exact) >= 0.5
  const ds = up ? Math.floor(exact) + 1 : Math.floor(exact)
  const dd = (area >> 1) + (up ? 0 : 1)
  const out = blank(im)
  let at = 0
  for (let y = 0; y < im.h; y++) {
    for (let x = 0; x < im.w; x++) {
      for (let c = 0; c < im.c; c++) {
        let s = 0
        for (let i = 0; i < kh; i++) for (let j = 0; j < kw; j++) s += px(y + i - ay, x + j - ax, c)
        out[at++] = fixed ? Math.floor(((s + dd) * ds) / 8388608) & 255 : sat(s * scale, dp)
      }
    }
  }
  return { ...im, v: out, dp }
}

const medianBlur = (im: Img, k: number): Img => {
  if (k === 1) return { ...im, v: [...im.v] }
  const px = sampler(im, BORDER.REPLICATE)
  const r = k >> 1
  const out = blank(im)
  const win = new Array<number>(k * k)
  let at = 0
  for (let y = 0; y < im.h; y++) {
    for (let x = 0; x < im.w; x++) {
      for (let c = 0; c < im.c; c++) {
        let n = 0
        for (let i = -r; i <= r; i++) for (let j = -r; j <= r; j++) win[n++] = px(y + i, x + j, c)
        win.sort((a, b) => a - b)
        out[at++] = win[(k * k) >> 1]
      }
    }
  }
  return { ...im, v: out }
}

/** getSobelKernels: the derivative kernel of the given order, or the smoothing one for order 0 */
const sobelKernel = (ksize: number, order: number): number[] => {
  if (ksize === 1) return [1]
  if (ksize === 3) return order === 0 ? [1, 2, 1] : order === 1 ? [-1, 0, 1] : [1, -2, 1]
  const k = new Array<number>(ksize + 1).fill(0)
  k[0] = 1
  for (let i = 0; i < ksize - order - 1; i++) {
    let old = k[0]
    for (let j = 1; j <= ksize; j++) {
      const nv = k[j] + k[j - 1]
      k[j - 1] = old
      old = nv
    }
  }
  for (let i = 0; i < order; i++) {
    let old = -k[0]
    for (let j = 1; j <= ksize; j++) {
      const nv = k[j - 1] - k[j]
      k[j - 1] = old
      old = nv
    }
  }
  return k.slice(0, ksize)
}

const sobel = (h: Host, im: Img, dp: Depth, dx: number, dy: number, ksize: number, scale: number, delta: number, border: number): Img => {
  if (dx < 0 || dy < 0 || dx + dy <= 0) throw cvErr(h, '-215:Assertion failed', 'dx >= 0 && dy >= 0 && dx+dy > 0')
  if (ksize % 2 === 0 || ksize > 31) throw cvErr(h, '-211:One of the arguments\' values is out of range', 'The kernel size must be odd and not larger than 31')
  const sizeX = ksize === 1 && dx > 0 ? 3 : ksize
  const sizeY = ksize === 1 && dy > 0 ? 3 : ksize
  if (sizeX <= dx || sizeY <= dy) throw cvErr(h, '-215:Assertion failed', 'ksize > order')
  const kx = sobelKernel(sizeX, dx)
  const ky = sobelKernel(sizeY, dy)
  return sepFilter(im, kx, ky.map((v) => v * scale), delta, border, dp)
}

const SCHARR_D = [-1, 0, 1]
const SCHARR_S = [3, 10, 3]

// ---------------------------------------------------------------- morphology

/** the kernel's cells as offsets from the anchor; a rectangle repeated n times is one larger rectangle, as in OpenCV */
const morphKernel = (kw: number, k: number[], ax: number, ay: number): { dy: number; dx: number }[] =>
  k.flatMap((v, i) => (v !== 0 ? [{ dy: Math.floor(i / kw) - ay, dx: i % kw - ax }] : []))

const morphOnce = (im: Img, cells: { dy: number; dx: number }[], isMax: boolean, border: number, cval: number[]): Img => {
  const px = sampler(im, border, cval)
  const out = blank(im)
  let at = 0
  for (let y = 0; y < im.h; y++) {
    for (let x = 0; x < im.w; x++) {
      for (let c = 0; c < im.c; c++) {
        let m = isMax ? -Infinity : Infinity
        for (const q of cells) {
          const p = px(y + q.dy, x + q.dx, c)
          if (isMax ? p > m : p < m) m = p
        }
        out[at++] = m
      }
    }
  }
  return { ...im, v: out }
}

interface MorphOpts { kh: number; kw: number; k: number[]; ax: number; ay: number; iter: number; border: number; value: number[] | null }

const morph = (im: Img, isMax: boolean, o: MorphOpts): Img => {
  let { kh, kw, k, ax, ay, iter } = o
  if (!k.length) {
    kh = kw = 1 + iter * 2
    k = new Array<number>(kh * kw).fill(1)
    ax = ay = iter
    iter = 1
  } else if (iter > 1 && k.every((v) => v !== 0)) {
    ax *= iter
    ay *= iter
    kw = kw + (iter - 1) * (kw - 1)
    kh = kh + (iter - 1) * (kh - 1)
    k = new Array<number>(kh * kw).fill(1)
    iter = 1
  }
  if (iter <= 0 || k.every((v) => v === 0)) return { ...im, v: [...im.v] }
  const cells = morphKernel(kw, k, ax, ay)
  const fill = isMax ? -Infinity : Infinity
  const cval = o.value ?? [fill, fill, fill, fill]
  let cur = im
  for (let i = 0; i < iter; i++) cur = morphOnce(cur, cells, isMax, o.border, cval)
  return cur
}

// ---------------------------------------------------------------- colour

const COLOR = {
  COLOR_BGR2BGRA: 0, COLOR_RGB2RGBA: 0, COLOR_BGRA2BGR: 1, COLOR_RGBA2RGB: 1, COLOR_BGR2RGBA: 2, COLOR_RGB2BGRA: 2, COLOR_RGBA2BGR: 3, COLOR_BGRA2RGB: 3,
  COLOR_BGR2RGB: 4, COLOR_RGB2BGR: 4, COLOR_BGRA2RGBA: 5, COLOR_RGBA2BGRA: 5, COLOR_BGR2GRAY: 6, COLOR_RGB2GRAY: 7, COLOR_GRAY2BGR: 8, COLOR_GRAY2RGB: 8,
  COLOR_GRAY2BGRA: 9, COLOR_GRAY2RGBA: 9, COLOR_BGRA2GRAY: 10, COLOR_RGBA2GRAY: 11, COLOR_BGR2HSV: 40, COLOR_RGB2HSV: 41, COLOR_HSV2BGR: 54, COLOR_HSV2RGB: 55,
  COLOR_BGR2HSV_FULL: 66, COLOR_RGB2HSV_FULL: 67, COLOR_HSV2BGR_FULL: 70, COLOR_HSV2RGB_FULL: 71,
}

const HSV_SHIFT = 12
const buildHsvTables = (hrange: number) => {
  const sdiv = new Array<number>(256).fill(0)
  const hdiv = new Array<number>(256).fill(0)
  for (let i = 1; i < 256; i++) {
    sdiv[i] = cvRound((255 << HSV_SHIFT) / i)
    hdiv[i] = cvRound((hrange << HSV_SHIFT) / (6 * i))
  }
  return { sdiv, hdiv }
}
const HSV_180 = buildHsvTables(180)
const HSV_256 = buildHsvTables(256)

const rgb2hsv8 = (r: number, g: number, b: number, hr: 180 | 256): [number, number, number] => {
  const { sdiv, hdiv } = hr === 180 ? HSV_180 : HSV_256
  const v = Math.max(r, g, b)
  const diff = v - Math.min(r, g, b)
  const s = (diff * sdiv[v] + (1 << (HSV_SHIFT - 1))) >> HSV_SHIFT
  const base = v === r ? g - b : v === g ? b - r + 2 * diff : r - g + 4 * diff
  let hue = (base * hdiv[diff] + (1 << (HSV_SHIFT - 1))) >> HSV_SHIFT
  if (hue < 0) hue += hr
  return [hue > 255 ? 255 : hue, s, v]
}

/** HSV2RGB_native, in float32 */
const hsv2rgbFloat = (h0: number, s: number, v: number, hscale: number): [number, number, number] => {
  if (s === 0) return [v, v, v]
  const SECTORS = [[1, 3, 0], [1, 0, 2], [3, 0, 1], [0, 2, 1], [0, 1, 3], [2, 1, 0]]
  let hue = fround(h0 * hscale) % 6
  let sector = Math.floor(hue)
  hue = fround(hue - sector)
  if (sector < 0 || sector >= 6) {
    sector = 0
    hue = 0
  }
  const tab = [v, fround(v * fround(1 - s)), fround(v * fround(1 - fround(s * hue))), fround(v * fround(1 - fround(s * fround(1 - hue))))]
  const [bi, gi, ri] = SECTORS[sector]
  return [tab[ri], tab[gi], tab[bi]]
}

const cvtColor = (h: Host, im: Img, code: number): { c: number; v: number[]; dp: Depth } => {
  const n = im.h * im.w
  const needCn = (...ok: number[]) => {
    if (!ok.includes(im.c)) throw cvErr(h, '-2:Unspecified error', `Invalid number of channels in input image: 'VScn::contains(scn)' where 'scn' is ${im.c}`)
  }
  const dp = im.dp
  const is8 = dp === '8u'
  if (!is8 && !isFloatDepth(dp) && dp !== '16u') throw cvErr(h, '-2:Unspecified error', `Unsupported depth of input image: 'VDepth::contains(depth)' where 'depth' is ${dp}${dp === '32s' ? ' (an int64 array reads as CV_32S: an image needs dtype=np.uint8)' : ''}`)
  const alpha = is8 ? 255 : dp === '16u' ? 65535 : 1
  const v = im.v
  const swapRB = (src: number, dst: number, swap: boolean) => {
    const out = new Array<number>(n * dst)
    for (let i = 0; i < n; i++) {
      out[i * dst] = v[i * src + (swap ? 2 : 0)]
      out[i * dst + 1] = v[i * src + 1]
      out[i * dst + 2] = v[i * src + (swap ? 0 : 2)]
      if (dst === 4) out[i * dst + 3] = src === 4 ? v[i * src + 3] : alpha
    }
    return { c: dst, v: out, dp }
  }
  switch (code) {
    case 0: needCn(3, 4); return swapRB(im.c, 4, false)
    case 1: needCn(3, 4); return swapRB(im.c, 3, false)
    case 2: needCn(3, 4); return swapRB(im.c, 4, true)
    case 3: needCn(3, 4); return swapRB(im.c, 3, true)
    case 4: needCn(3, 4); return swapRB(im.c, 3, true)
    case 5: needCn(4); return swapRB(4, 4, true)
    case 6: case 7: case 10: case 11: {
      needCn(3, 4)
      const bi = code === 6 || code === 10 ? 0 : 2
      const ri = 2 - bi
      const out = new Array<number>(n)
      for (let i = 0; i < n; i++) {
        const b = v[i * im.c + bi]
        const g = v[i * im.c + 1]
        const r = v[i * im.c + ri]
        if (is8) out[i] = (b * 3735 + g * 19235 + r * 9798 + 16384) >> 15
        else if (dp === '16u') out[i] = (b * 3735 + g * 19235 + r * 9798 + 16384) / 32768 | 0
        else out[i] = fround(fround(fround(fround(b * fround(0.114)) + fround(g * fround(0.587))) + fround(r * fround(0.299))))
      }
      return { c: 1, v: out, dp }
    }
    case 8: case 9: {
      needCn(1)
      const dst = code === 8 ? 3 : 4
      const out = new Array<number>(n * dst)
      for (let i = 0; i < n; i++) {
        out[i * dst] = out[i * dst + 1] = out[i * dst + 2] = v[i]
        if (dst === 4) out[i * dst + 3] = alpha
      }
      return { c: dst, v: out, dp }
    }
    case 40: case 41: case 66: case 67: {
      needCn(3, 4)
      if (!is8 && !isFloatDepth(dp)) throw cvErr(h, '-2:Unspecified error', 'Unsupported depth of input image')
      const bi = code === 40 || code === 66 ? 0 : 2
      const out = new Array<number>(n * 3)
      const hr = code >= 66 ? 256 : 180
      for (let i = 0; i < n; i++) {
        const b = v[i * im.c + bi]
        const g = v[i * im.c + 1]
        const r = v[i * im.c + 2 - bi]
        if (is8) {
          ;[out[i * 3], out[i * 3 + 1], out[i * 3 + 2]] = rgb2hsv8(r, g, b, hr)
        } else {
          const mx = Math.max(r, g, b)
          const diff = fround(mx - Math.min(r, g, b))
          const s = fround(diff / fround(Math.abs(mx) + 2.2204460492503131e-16))
          const q = fround(60 / fround(diff + 2.2204460492503131e-16))
          let hue = mx === r ? fround(fround(g - b) * q) : mx === g ? fround(fround(fround(b - r) * q) + 120) : fround(fround(fround(r - g) * q) + 240)
          if (hue < 0) hue = fround(hue + 360)
          out[i * 3] = hue
          out[i * 3 + 1] = s
          out[i * 3 + 2] = mx
        }
      }
      return { c: 3, v: out, dp }
    }
    case 54: case 55: case 70: case 71: {
      needCn(3)
      if (!is8 && !isFloatDepth(dp)) throw cvErr(h, '-2:Unspecified error', 'Unsupported depth of input image')
      const toBgr = code === 54 || code === 70
      const hrange = !is8 ? 360 : code >= 70 ? 255 : 180
      const hscale = fround(6 / hrange)
      const out = new Array<number>(n * 3)
      const inv = fround(1 / 255)
      for (let i = 0; i < n; i++) {
        const s = is8 ? fround(v[i * 3 + 1] * inv) : v[i * 3 + 1]
        const val = is8 ? fround(v[i * 3 + 2] * inv) : v[i * 3 + 2]
        const [r, g, b] = hsv2rgbFloat(v[i * 3], s, val, hscale)
        const put = (k: number, x: number) => { out[i * 3 + k] = is8 ? sat(fround(x * 255), '8u') : x }
        put(0, toBgr ? b : r)
        put(1, g)
        put(2, toBgr ? r : b)
      }
      return { c: 3, v: out, dp }
    }
  }
  throw h.err('NotImplementedError', `cv2.cvtColor: colour conversion code ${code} is not available in this sandbox (BGR, RGB, GRAY, BGRA, RGBA and HSV are)`)
}

// ---------------------------------------------------------------- resize

const INTER = { NEAREST: 0, LINEAR: 1, CUBIC: 2, AREA: 3, LANCZOS4: 4, LINEAR_EXACT: 5, NEAREST_EXACT: 6, MAX: 7, WARP_FILL_OUTLIERS: 8, WARP_INVERSE_MAP: 16 }

const cubicCoefs = (x: number): number[] => {
  const A = fround(-0.75)
  const c0 = fround(fround(fround(fround(fround(A * fround(x + 1)) - fround(5 * A)) * fround(x + 1)) + fround(8 * A)) * fround(x + 1)) - fround(4 * A)
  const c1 = fround(fround(fround(fround(A + 2) * x) - fround(A + 3)) * fround(x * x)) + 1
  const y = fround(1 - x)
  const c2 = fround(fround(fround(fround(A + 2) * y) - fround(A + 3)) * fround(y * y)) + 1
  const a = fround(c0)
  const b = fround(c1)
  const c = fround(c2)
  return [a, b, c, fround(fround(fround(1 - a) - b) - c)]
}

interface Axis { ofs: number[]; coef: number[][] }

/** the source position and interpolation weights of each destination column or row */
const resizeAxis = (dn: number, sn: number, scale: number, inv: number, ksize: number, area: boolean, isColumn: boolean): Axis => {
  const half = ksize >> 1
  const ofs: number[] = []
  const coef: number[][] = []
  for (let d = 0; d < dn; d++) {
    let s: number
    let f: number
    if (!area) {
      f = fround((d + 0.5) * scale - 0.5)
      s = Math.floor(f)
      f = fround(f - s)
    } else {
      s = Math.floor(d * scale)
      f = fround(d + 1 - (s + 1) * inv)
      f = f <= 0 ? 0 : fround(f - Math.floor(f))
    }
    if (ksize === 2 && isColumn) {
      if (s < 0) {
        f = 0
        s = 0
      }
      if (s + half >= sn && s >= sn - 1) {
        f = 0
        s = sn - 1
      }
    }
    ofs.push(s)
    coef.push(ksize === 2 ? [fround(1 - f), f] : cubicCoefs(f))
  }
  return { ofs, coef }
}

const resizeLinearCubic = (im: Img, dw: number, dh: number, sx: number, sy: number, ix: number, iy: number, ksize: number, area: boolean): number[] => {
  const xa = resizeAxis(dw, im.w, sx, ix, ksize, area, true)
  const ya = resizeAxis(dh, im.h, sy, iy, ksize, area, false)
  const fixed = im.dp === '8u'
  const single = im.dp !== '64f'
  const half = ksize >> 1
  const alpha = xa.coef.map((k) => (fixed ? k.map((v) => sat(v * 2048, '16s')) : k))
  const beta = ya.coef.map((k) => (fixed ? k.map((v) => sat(v * 2048, '16s')) : k))
  const cache = new Map<number, number[]>()
  const hrow = (sy0: number): number[] => {
    let r = cache.get(sy0)
    if (r) return r
    r = new Array<number>(dw * im.c)
    const base = sy0 * im.w * im.c
    for (let d = 0; d < dw; d++) {
      const a = alpha[d]
      const s0 = xa.ofs[d] - half + 1
      for (let c = 0; c < im.c; c++) {
        let val = 0
        for (let k = 0; k < ksize; k++) {
          const p = s0 + k
          const q = p < 0 ? 0 : p >= im.w ? im.w - 1 : p
          val += a[k] * im.v[base + q * im.c + c]
        }
        r[d * im.c + c] = fixed || !single ? val : fround(val)
      }
    }
    cache.set(sy0, r)
    return r
  }
  const out = new Array<number>(dw * dh * im.c)
  let at = 0
  const line = dw * im.c
  // VResizeCubicVec_32s8u: eight values at a time in float32, rounding ties to even; what is left over stays in fixed point
  const simdEnd = line - line % 8
  const toUnit = fround(1 / (2048 * 2048))
  for (let d = 0; d < dh; d++) {
    const b = beta[d]
    const bf = fixed ? b.map((x) => fround(x * toUnit)) : b
    const rows = Array.from({ length: ksize }, (_, k) => hrow(Math.min(Math.max(ya.ofs[d] - half + 1 + k, 0), im.h - 1)))
    for (let j = 0; j < line; j++) {
      if (fixed && ksize === 2) {
        const v = (((b[0] * (rows[0][j] >> 4)) >> 16) + ((b[1] * (rows[1][j] >> 4)) >> 16) + 2) >> 2
        out[at++] = v < 0 ? 0 : v > 255 ? 255 : v
      } else if (fixed && j < simdEnd) {
        const t3 = fround(fround(rows[3][j]) * bf[3])
        const t2 = fround(fround(fround(rows[2][j]) * bf[2]) + t3)
        const t1 = fround(fround(fround(rows[1][j]) * bf[1]) + t2)
        out[at++] = sat(fround(fround(fround(rows[0][j]) * bf[0]) + t1), '8u')
      } else if (fixed) {
        const v = (rows[0][j] * b[0] + rows[1][j] * b[1] + rows[2][j] * b[2] + rows[3][j] * b[3] + (1 << 21)) >> 22
        out[at++] = v < 0 ? 0 : v > 255 ? 255 : v
      } else {
        let s = 0
        for (let k = 0; k < ksize; k++) s += rows[k][j] * b[k]
        out[at++] = sat(s, im.dp)
      }
    }
  }
  return out
}

const resizeNearest = (im: Img, dw: number, dh: number, ix: number, iy: number): number[] => {
  const fx = 1 / ix
  const fy = 1 / iy
  const out = new Array<number>(dw * dh * im.c)
  let at = 0
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(Math.floor(y * fy), im.h - 1)
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(Math.floor(x * fx), im.w - 1)
      for (let c = 0; c < im.c; c++) out[at++] = im.v[(sy * im.w + sx) * im.c + c]
    }
  }
  return out
}

/** resizeAreaFast_: an exact integer shrink averages whole blocks */
const resizeAreaFast = (im: Img, dw: number, dh: number, isx: number, isy: number): number[] => {
  const area = isx * isy
  const scale = fround(1 / area)
  const out = new Array<number>(dw * dh * im.c)
  const avg2 = isx === 2 && isy === 2 && (im.c === 1 || im.c === 3 || im.c === 4) && im.dp === '8u'
  const wFull = Math.floor(im.w / isx)
  let at = 0
  for (let dy = 0; dy < dh; dy++) {
    const sy0 = dy * isy
    const full = sy0 + isy <= im.h
    for (let dx = 0; dx < dw; dx++) {
      for (let c = 0; c < im.c; c++, at++) {
        if (sy0 >= im.h) { out[at] = 0; continue }
        const sx0 = dx * isx
        let sum = 0
        let count = 0
        for (let i = 0; i < isy && sy0 + i < im.h; i++) {
          for (let j = 0; j < isx && sx0 + j < im.w; j++) {
            sum += im.v[((sy0 + i) * im.w + sx0 + j) * im.c + c]
            count++
          }
        }
        if (full && dx < wFull) out[at] = avg2 ? (sum + 2) >> 2 : sat(im.dp === '8u' || im.dp === '32f' ? fround(fround(sum) * scale) : sum * scale, im.dp)
        else out[at] = sat(im.dp === '8u' ? Math.trunc(sum / count) : sum / count, im.dp)
      }
    }
  }
  return out
}

/** computeResizeAreaTab: which source cells make up each destination cell, and in what share */
const areaTab = (sn: number, dn: number, scale: number): { d: number; s: number; a: number }[] => {
  const tab: { d: number; s: number; a: number }[] = []
  for (let d = 0; d < dn; d++) {
    const f1 = d * scale
    const f2 = f1 + scale
    const cell = Math.min(scale, sn - f1)
    let s1 = Math.ceil(f1)
    let s2 = Math.floor(f2)
    s2 = Math.min(s2, sn - 1)
    s1 = Math.min(s1, s2)
    if (s1 - f1 > 1e-3) tab.push({ d, s: s1 - 1, a: fround((s1 - f1) / cell) })
    for (let s = s1; s < s2; s++) tab.push({ d, s, a: fround(1 / cell) })
    if (f2 - s2 > 1e-3) tab.push({ d, s: s2, a: fround(Math.min(Math.min(f2 - s2, 1), cell) / cell) })
  }
  return tab
}

const resizeArea = (im: Img, dw: number, dh: number, sx: number, sy: number): number[] => {
  const xt = areaTab(im.w, dw, sx)
  const yt = areaTab(im.h, dh, sy)
  const wide = im.dp === '64f'
  const line = dw * im.c
  const out = new Array<number>(dh * line).fill(0)
  const sum = new Array<number>(line).fill(0)
  let k = 0
  for (let dy = 0; dy < dh; dy++) {
    sum.fill(0)
    for (; k < yt.length && yt[k].d === dy; k++) {
      const beta = yt[k].a
      const base = yt[k].s * im.w * im.c
      const buf = new Array<number>(line).fill(0)
      for (const q of xt) {
        for (let c = 0; c < im.c; c++) {
          const at = q.d * im.c + c
          const t = q.a * im.v[base + q.s * im.c + c]
          buf[at] = wide ? buf[at] + t : fround(buf[at] + fround(t))
        }
      }
      for (let j = 0; j < line; j++) sum[j] = wide ? sum[j] + beta * buf[j] : fround(sum[j] + fround(beta * buf[j]))
    }
    for (let j = 0; j < line; j++) out[dy * line + j] = sat(sum[j], im.dp)
  }
  return out
}

const resize = (h: Host, im: Img, dsize: [number, number] | null, fx: number, fy: number, interp0: number): Img => {
  let ix = fx
  let iy = fy
  let dw: number
  let dh: number
  if (!dsize || dsize[0] <= 0 || dsize[1] <= 0) {
    if (ix <= 0) throw cvErr(h, '-215:Assertion failed', 'inv_scale_x > 0')
    if (iy <= 0) throw cvErr(h, '-215:Assertion failed', 'inv_scale_y > 0')
    dw = cvRound(im.w * ix)
    dh = cvRound(im.h * iy)
    if (dw <= 0 || dh <= 0) throw cvErr(h, '-215:Assertion failed', '!dsize.empty()')
  } else {
    ;[dw, dh] = dsize
    ix = dw / im.w
    iy = dh / im.h
  }
  if (dw === im.w && dh === im.h) return { ...im, v: [...im.v] }
  const sx = 1 / ix
  const sy = 1 / iy
  let interp = interp0 & INTER.MAX
  if (interp !== INTER.NEAREST && interp !== INTER.LINEAR && interp !== INTER.CUBIC && interp !== INTER.AREA) {
    throw h.err('NotImplementedError', 'cv2.resize: only INTER_NEAREST, INTER_LINEAR, INTER_CUBIC and INTER_AREA are available in this sandbox')
  }
  const dst = { h: dh, w: dw, c: im.c, dp: im.dp }
  if (interp === INTER.NEAREST) return { ...dst, v: resizeNearest(im, dw, dh, ix, iy) }
  no32s(h, im, 'resize')
  const isx = cvRound(sx)
  const isy = cvRound(sy)
  const fast = Math.abs(sx - isx) < Number.EPSILON && Math.abs(sy - isy) < Number.EPSILON
  if (interp === INTER.LINEAR && fast && isx === 2 && isy === 2) interp = INTER.AREA
  if (interp === INTER.AREA && sx >= 1 && sy >= 1) return { ...dst, v: fast ? resizeAreaFast(im, dw, dh, isx, isy) : resizeArea(im, dw, dh, sx, sy) }
  return { ...dst, v: resizeLinearCubic(im, dw, dh, sx, sy, ix, iy, interp === INTER.CUBIC ? 4 : 2, interp === INTER.AREA) }
}

// ---------------------------------------------------------------- warps

const AB_SCALE = 1024
const TAB = 32

/** one output pixel of remapBilinear / remapNearest: integer source position plus a 1/32 fraction */
const remapPixel = (im: Img, border: number, cval: number[], sx: number, sy: number, ax: number, ay: number, nearest: boolean, out: number[], at: number) => {
  const { h: H, w: W, c: C, v } = im
  if (nearest) {
    if (sx >= 0 && sx < W && sy >= 0 && sy < H) {
      for (let c = 0; c < C; c++) out[at + c] = v[(sy * W + sx) * C + c]
    } else if (border === BORDER.CONSTANT) {
      for (let c = 0; c < C; c++) out[at + c] = sat(cval[c] ?? 0, im.dp)
    } else {
      const yy = borderIdx(sy, H, border)
      const xx = borderIdx(sx, W, border)
      for (let c = 0; c < C; c++) out[at + c] = v[(yy * W + xx) * C + c]
    }
    return
  }
  const fixed = im.dp === '8u'
  const w00 = (TAB - ay) * (TAB - ax)
  const w01 = (TAB - ay) * ax
  const w10 = ay * (TAB - ax)
  const w11 = ay * ax
  const put = (c: number, a: number, b: number, d: number, e: number) => {
    const s = a * w00 + b * w01 + d * w10 + e * w11
    if (fixed) {
      const r = ((s * 32) + (1 << 14)) >> 15
      out[at + c] = r > 255 ? 255 : r < 0 ? 0 : r
    } else out[at + c] = sat(s / 1024, im.dp)
  }
  if (sx >= 0 && sy >= 0 && sx < W - 1 && sy < H - 1) {
    for (let c = 0; c < C; c++) {
      const i = (sy * W + sx) * C + c
      put(c, v[i], v[i + C], v[i + W * C], v[i + W * C + C])
    }
    return
  }
  if (border === BORDER.CONSTANT && (sx >= W || sx + 1 < 0 || sy >= H || sy + 1 < 0)) {
    for (let c = 0; c < C; c++) out[at + c] = sat(cval[c] ?? 0, im.dp)
    return
  }
  const x0 = borderIdx(sx, W, border)
  const x1 = borderIdx(sx + 1, W, border)
  const y0 = borderIdx(sy, H, border)
  const y1 = borderIdx(sy + 1, H, border)
  for (let c = 0; c < C; c++) {
    const cv = cval[c] ?? 0
    const at2 = (y: number, x: number) => (y >= 0 && x >= 0 ? v[(y * W + x) * C + c] : cv)
    put(c, at2(y0, x0), at2(y0, x1), at2(y1, x0), at2(y1, x1))
  }
}

const warpInterp = (h: Host, flags: number, fn: string): 0 | 1 => {
  const interp = flags & INTER.MAX
  if (interp === INTER.NEAREST) return 0
  if (interp === INTER.LINEAR || interp === INTER.AREA) return 1
  throw h.err('NotImplementedError', `cv2.${fn}: only INTER_NEAREST and INTER_LINEAR are available in this sandbox`)
}

const warpAffine = (h: Host, im: Img, M0: number[], dw: number, dh: number, flags: number, border: number, cval: number[]): Img => {
  const interp = warpInterp(h, flags, 'warpAffine')
  if (interp !== 0) no32s(h, im, 'remap')
  const M = [...M0]
  if (!(flags & INTER.WARP_INVERSE_MAP)) {
    let D = M[0] * M[4] - M[1] * M[3]
    D = D !== 0 ? 1 / D : 0
    const a11 = M[4] * D
    const a22 = M[0] * D
    M[0] = a11
    M[1] *= -D
    M[3] *= -D
    M[4] = a22
    const b1 = -M[0] * M[2] - M[1] * M[5]
    const b2 = -M[3] * M[2] - M[4] * M[5]
    M[2] = b1
    M[5] = b2
  }
  const out = new Array<number>(dw * dh * im.c)
  const rd = interp === 0 ? AB_SCALE / 2 : AB_SCALE / TAB / 2
  for (let y = 0; y < dh; y++) {
    const X0 = cvRound((M[1] * y + M[2]) * AB_SCALE) + rd
    const Y0 = cvRound((M[4] * y + M[5]) * AB_SCALE) + rd
    for (let x = 0; x < dw; x++) {
      const at = (y * dw + x) * im.c
      const X = X0 + cvRound(M[0] * x * AB_SCALE)
      const Y = Y0 + cvRound(M[3] * x * AB_SCALE)
      if (interp === 0) remapPixel(im, border, cval, X >> 10, Y >> 10, 0, 0, true, out, at)
      else {
        const fx = X >> 5
        const fy = Y >> 5
        remapPixel(im, border, cval, fx >> 5, fy >> 5, fx & 31, fy & 31, false, out, at)
      }
    }
  }
  return { h: dh, w: dw, c: im.c, v: out, dp: im.dp }
}

/** Gaussian elimination with partial pivoting, the way cv::solve(DECOMP_LU) goes about it */
const solve = (h: Host, A0: number[][], b0: number[]): number[] => {
  const A = A0.map((r) => [...r])
  const b = [...b0]
  const n = b.length
  for (let i = 0; i < n; i++) {
    let k = i
    for (let j = i + 1; j < n; j++) if (Math.abs(A[j][i]) > Math.abs(A[k][i])) k = j
    if (Math.abs(A[k][i]) < 2.220446049250313e-15) throw cvErr(h, '-215:Assertion failed', 'the points are degenerate: the transform has no unique solution')
    if (k !== i) {
      ;[A[i], A[k]] = [A[k], A[i]]
      ;[b[i], b[k]] = [b[k], b[i]]
    }
    for (let j = i + 1; j < n; j++) {
      const f = -A[j][i] / A[i][i]
      for (let c = i + 1; c < n; c++) A[j][c] += f * A[i][c]
      b[j] += f * b[i]
    }
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i]
    for (let c = i + 1; c < n; c++) s -= A[i][c] * b[c]
    b[i] = s / A[i][i]
  }
  return b
}

const invert3 = (h: Host, M: number[]): number[] => {
  const cols = [0, 1, 2].map((j) => solve(h, [M.slice(0, 3), M.slice(3, 6), M.slice(6, 9)], [0, 1, 2].map((i) => (i === j ? 1 : 0))))
  return [0, 1, 2].flatMap((i) => [0, 1, 2].map((j) => cols[j][i]))
}

const warpPerspective = (h: Host, im: Img, M0: number[], dw: number, dh: number, flags: number, border: number, cval: number[]): Img => {
  const interp = warpInterp(h, flags, 'warpPerspective')
  if (interp !== 0) no32s(h, im, 'remap')
  const M = flags & INTER.WARP_INVERSE_MAP ? M0 : invert3(h, M0)
  const out = new Array<number>(dw * dh * im.c)
  const mult = interp === 0 ? 1 : TAB
  const clip = (v: number) => Math.max(-2147483648, Math.min(2147483647, v))
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      const W0 = M[6] * x + M[7] * y + M[8]
      const W = W0 ? mult / W0 : 0
      const X = cvRound(clip((M[0] * x + M[1] * y + M[2]) * W))
      const Y = cvRound(clip((M[3] * x + M[4] * y + M[5]) * W))
      const at = (y * dw + x) * im.c
      if (interp === 0) remapPixel(im, border, cval, X, Y, 0, 0, true, out, at)
      else remapPixel(im, border, cval, X >> 5, Y >> 5, X & 31, Y & 31, false, out, at)
    }
  }
  return { h: dh, w: dw, c: im.c, v: out, dp: im.dp }
}

const points = (h: Host, v: Value | undefined, n: number, what: string): [number, number][] => {
  const a = asMat(h, v, what)
  if (a.size !== n * 2) throw cvErr(h, '-215:Assertion failed', `${what}.checkVector(2, CV_32F) == ${n}`)
  const f = a.values().map(fround)
  return Array.from({ length: n }, (_, i) => [f[i * 2], f[i * 2 + 1]])
}

// ---------------------------------------------------------------- thresholding

const THRESH = { BINARY: 0, BINARY_INV: 1, TRUNC: 2, TOZERO: 3, TOZERO_INV: 4, MASK: 7, OTSU: 8, TRIANGLE: 16 }

const otsu = (v: number[]): number => {
  const hist = new Array<number>(256).fill(0)
  for (const x of v) hist[x]++
  const scale = 1 / v.length
  let mu = 0
  for (let i = 0; i < 256; i++) mu += i * hist[i]
  mu *= scale
  let mu1 = 0
  let q1 = 0
  let best = 0
  let val = 0
  const eps = 1.1920928955078125e-7
  for (let i = 0; i < 256; i++) {
    const p = hist[i] * scale
    mu1 *= q1
    q1 += p
    const q2 = 1 - q1
    if (Math.min(q1, q2) < eps || Math.max(q1, q2) > 1 - eps) continue
    mu1 = (mu1 + i * p) / q1
    const mu2 = (mu - q1 * mu1) / q2
    const sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2)
    if (sigma > best) {
      best = sigma
      val = i
    }
  }
  return val
}

const thresholdCore = (im: Img, thresh0: number, maxval: number, type: number): { thresh: number; v: number[] } => {
  const isInt = !isFloatDepth(im.dp)
  const t = isInt ? Math.floor(thresh0) : thresh0
  const mx = isInt ? sat(maxval, im.dp) : maxval
  const fn = (x: number): number => {
    switch (type) {
      case THRESH.BINARY: return x > t ? mx : 0
      case THRESH.BINARY_INV: return x > t ? 0 : mx
      case THRESH.TRUNC: return x > t ? t : x
      case THRESH.TOZERO: return x > t ? x : 0
      default: return x > t ? 0 : x
    }
  }
  return { thresh: t, v: im.v.map(fn) }
}

// ---------------------------------------------------------------- arithmetic

type Operand = { kind: 'img'; im: Img } | { kind: 'scalar'; s: number[] }

const operand = (h: Host, v: Value | undefined, other: Value | undefined, what: string): Operand => {
  if (!v) throw h.err('TypeError', `missing required argument '${what}'`)
  if (isNum(v)) return { kind: 'scalar', s: [numOf(v), numOf(v), numOf(v), numOf(v)] }
  if (v.k === 'tuple' || v.k === 'list') return { kind: 'scalar', s: scalarOf(h, v) }
  if (v.k === 'array') {
    // checkScalar: a small array that is not the image's size stands for a Scalar when it has one value or one per channel
    const o = other && other.k === 'array' ? other.a : null
    const same = o && o.shape.length === v.a.shape.length && o.shape.every((d, i) => d === v.a.shape[i])
    const cn = o?.ndim === 3 ? o.shape[2] : 1
    const [r, c] = [v.a.shape[0] ?? 1, v.a.ndim >= 2 ? v.a.shape[1] : 1]
    const fits = v.a.ndim <= 2 && ((r === 1 && c === 1) || (r === cn && c === 1) || (r === 1 && c === cn) || (r === 4 && c === 1 && v.a.dtype === 'float64' && cn <= 4))
    if (!same && o && o.ndim >= 2 && fits) {
      const xs = v.a.values()
      return { kind: 'scalar', s: [...xs, 0, 0, 0, 0].slice(0, 4) }
    }
  }
  return { kind: 'img', im: readImg(h, v, what) }
}

/** arithm_op: array op array or array op scalar, into the output depth, with an optional mask */
const arith = (h: Host, a0: Value | undefined, b0: Value | undefined, maskV: Value | undefined, dtypeV: Value | undefined, name: string, f: (x: number, y: number, dp: Depth, i: number, total: number) => number): Value => {
  const a = operand(h, a0, b0, 'src1')
  const b = operand(h, b0, a0, 'src2')
  if (a.kind === 'scalar' && b.kind === 'scalar') throw cvErr(h, '-209:Sizes of input arguments do not match', `The operation is neither 'array op array' (where arrays have the same size and the same number of channels), nor 'array op scalar', nor 'scalar op array' in function 'arithm_op'`)
  const base = a.kind === 'img' ? a.im : (b as { im: Img }).im
  if (a.kind === 'img' && b.kind === 'img') {
    if (a.im.h !== b.im.h || a.im.w !== b.im.w || a.im.c !== b.im.c) throw cvErr(h, '-209:Sizes of input arguments do not match', `The operation is neither 'array op array' (where arrays have the same size and the same number of channels), nor 'array op scalar', nor 'scalar op array' in function 'arithm_op'`)
    if (a.im.dp !== b.im.dp && (!dtypeV || dtypeV.k === 'none')) throw cvErr(h, '-5:Bad argument', `When the input arrays in add/subtract/multiply/divide functions have different types, the output array type must be explicitly specified in function 'arithm_op'`)
  }
  // a 4-element Scalar arrives as a 4x1 double Mat, so a 4x1 image reads it as an array of another type
  if (a.kind !== b.kind && base.h === 4 && base.w === 1 && base.c === 1 && (!dtypeV || dtypeV.k === 'none')) throw cvErr(h, '-5:Bad argument', `When the input arrays in add/subtract/multiply/divide functions have different types, the output array type must be explicitly specified in function 'arithm_op'`)
  const dp = depthArg(h, dtypeV, base.dp)
  const mask = maskV && maskV.k !== 'none' ? readImg(h, maskV, 'mask') : null
  if (mask && (mask.h !== base.h || mask.w !== base.w || mask.c !== 1 || (mask.dp !== '8u' && mask.dp !== '8s'))) throw cvErr(h, '-215:Assertion failed', `(mtype == CV_8U || mtype == CV_8S) && _mask.sameSize(*psrc1) in function '${name}'`)
  const out = blank(base)
  const C = base.c
  for (let i = 0; i < base.h * base.w; i++) {
    if (mask && !mask.v[i]) continue
    for (let c = 0; c < C; c++) {
      const x = a.kind === 'img' ? a.im.v[i * C + c] : a.s[c] ?? 0
      const y = b.kind === 'img' ? b.im.v[i * C + c] : b.s[c] ?? 0
      out[i * C + c] = f(x, y, dp, i * C + c, out.length)
    }
  }
  return toValue({ ...base, v: out }, dp)
}

const compareBound = (h: Host, v: Value | undefined, im: Img, what: string): ((i: number, c: number) => number) => {
  if (v && v.k === 'array' && v.a.ndim >= 2 && v.a.shape[0] === im.h && v.a.shape[1] === im.w) {
    const vals = v.a.values()
    return (i, c) => vals[i * im.c + c]
  }
  const s = v && v.k === 'array' ? v.a.values() : v && isNum(v) ? [numOf(v)] : v ? h.iterate(v).map((x) => h.num(x)) : null
  if (!s) throw h.err('TypeError', `missing required argument '${what}'`)
  if (s.length !== im.c && s.length !== 1 && s.length !== 4) throw cvErr(h, '-209:Sizes of input arguments do not match', `The lower boundary is neither an array of the same size and same type as src, nor a scalar in function 'inRange'`)
  return (_i, c) => s[c] ?? 0
}

// ---------------------------------------------------------------- the module

const NO_FILES = (what: string) => `there are no files in this sandbox, so cv2.${what} has nothing to read or write. Make the image as a NumPy array instead, e.g. img = np.zeros((4, 4, 3), dtype=np.uint8)`
const NO_WINDOWS = (what: string) => `there are no windows in this sandbox, so cv2.${what} cannot show anything. Print the NumPy array, or draw it with plt.imshow(img)`

/** Run one cv2 call's parameters through its Python signature */
const bind = (h: Host, fnName: string, params: string[], args: Value[], kw: Kw): Args => {
  if (args.length > params.length) throw cvErr(h, '-5:Bad argument', `function takes at most ${params.length} arguments (${args.length} given) in function '${fnName}'`)
  const out: Args = {}
  params.forEach((p, i) => { out[p] = args[i] })
  for (const k of Object.keys(kw)) {
    if (!params.includes(k)) throw cvErr(h, '-5:Bad argument', `'${k}' is an invalid keyword argument for ${fnName}()`)
    if (out[k] !== undefined) throw h.err('TypeError', `Argument given by name ('${k}') and position (${params.indexOf(k) + 1})`)
    out[k] = kw[k]
  }
  return out
}

const required = (h: Host, a: Args, fn: string, ...names: string[]) => {
  for (const n of names) if (a[n] === undefined) throw cvErr(h, '-5:Bad argument', `Missing required argument '${n}' (pos ${names.indexOf(n) + 1}) in function '${fn}'`)
}

const build = (h: Host): Record<string, Value> => {
  const attrs: Record<string, Value> = { __version__: py.str('4.13.0'), error: errorClass(h) }

  /** a cv2 function: its Python parameters and how the step-through panel should show it */
  const def = (name: string, params: string, kind: ApiKind | null, body: (a: Args, raw: { args: Value[]; kw: Kw }) => Value) => {
    const names = params.split(' ').filter(Boolean)
    attrs[name] = {
      k: 'fn',
      name,
      call: (args, kw) => {
        const a = bind(h, name, names, args, kw)
        const res = body(a, { args, kw })
        const dst = a.dst
        if (dst && dst.k === 'array' && res.k === 'array' && dst.a.shape.length === res.a.shape.length && dst.a.shape.every((d, i) => d === res.a.shape[i])) {
          const vals = res.a.values()
          dst.a.addresses().forEach((addr, i) => { dst.a.data[addr] = vals[i] })
          return dst
        }
        return res
      },
      api: kind ? { name: `cv2.${name}`, kind } : undefined,
    } as Value
  }

  /** a filter's step: each result cell comes from the window of input cells under the kernel */
  const traceWindow = (api: string, srcV: Value, im: Img, res: Value, kh: number, kw: number, ax: number, ay: number, border: number) => {
    if (!h.tracing || res.k !== 'array' || im.v.length > 240) return
    const groups: [number, number][][] = []
    for (let y = 0; y < im.h; y++) {
      for (let x = 0; x < im.w; x++) {
        for (let c = 0; c < im.c; c++) {
          const g: [number, number][] = []
          for (let i = 0; i < kh; i++) {
            for (let j = 0; j < kw; j++) {
              const yy = borderIdx(y + i - ay, im.h, border)
              const xx = borderIdx(x + j - ax, im.w, border)
              if (yy >= 0 && xx >= 0) g.push([0, (yy * im.w + xx) * im.c + c])
            }
          }
          groups.push(g)
        }
      }
    }
    const a = srcV.k === 'array' ? srcV.a : res.a
    h.traceCall(api, 'group', [{ label: h.nameOf(srcV, 'src'), snap: { shape: [...a.shape], values: a.values(), dtype: a.dtype } }], { shape: [...res.a.shape], values: res.a.values(), dtype: res.a.dtype }, h.str(res), null, groups)
  }

  // ---- files and windows
  const refuse = (name: string, type: string, msg: (what: string) => string) => {
    attrs[name] = { k: 'fn', name, call: () => { throw h.err(type, msg(name)) } }
  }
  refuse('imread', 'FileNotFoundError', NO_FILES)
  refuse('imwrite', 'NotImplementedError', NO_FILES)
  refuse('VideoCapture', 'FileNotFoundError', NO_FILES)
  refuse('VideoWriter', 'NotImplementedError', NO_FILES)
  for (const name of ['imshow', 'waitKey', 'destroyAllWindows', 'destroyWindow', 'namedWindow', 'resizeWindow', 'moveWindow']) refuse(name, 'NotImplementedError', NO_WINDOWS)

  // ---- colour
  def('cvtColor', 'src code dst dstCn hint', 'create', (a) => {
    required(h, a, 'cvtColor', 'src', 'code')
    const im = readImg(h, a.src)
    if (a.dstCn && a.dstCn.k !== 'none' && h.toInt(a.dstCn) !== 0) throw h.err('NotImplementedError', 'cv2.cvtColor: dstCn is not available in this sandbox')
    return toValue({ h: im.h, w: im.w, ...cvtColor(h, im, h.toInt(a.code!)) }, im.dp)
  })

  // ---- geometry
  def('resize', 'src dsize dst fx fy interpolation', 'create', (a) => {
    required(h, a, 'resize', 'src', 'dsize')
    const im = readImg(h, a.src)
    const ds = a.dsize!.k === 'none' ? null : intPair(h, a.dsize, 'dsize')
    const r = resize(h, im, ds, numArg(h, a.fx, 0), numArg(h, a.fy, 0), intArg(h, a.interpolation, INTER.LINEAR))
    return toValue(r, r.dp)
  })
  def('warpAffine', 'src M dsize dst flags borderMode borderValue', 'create', (a) => {
    required(h, a, 'warpAffine', 'src', 'M', 'dsize')
    const im = readImg(h, a.src)
    const M = asMat(h, a.M, 'M')
    if (M.shape.length !== 2 || M.shape[0] !== 2 || M.shape[1] !== 3) throw cvErr(h, '-215:Assertion failed', '(M0.type() == CV_32F || M0.type() == CV_64F) && M0.rows == 2 && M0.cols == 3')
    const [dw, dh] = intPair(h, a.dsize, 'dsize')
    const r = warpAffine(h, im, M.values(), dw, dh, intArg(h, a.flags, INTER.LINEAR), borderType(h, a.borderMode, BORDER.CONSTANT), scalarOf(h, a.borderValue))
    return toValue(r, r.dp)
  })
  def('warpPerspective', 'src M dsize dst flags borderMode borderValue', 'create', (a) => {
    required(h, a, 'warpPerspective', 'src', 'M', 'dsize')
    const im = readImg(h, a.src)
    const M = asMat(h, a.M, 'M')
    if (M.shape.length !== 2 || M.shape[0] !== 3 || M.shape[1] !== 3) throw cvErr(h, '-215:Assertion failed', '(M0.type() == CV_32F || M0.type() == CV_64F) && M0.rows == 3 && M0.cols == 3')
    const [dw, dh] = intPair(h, a.dsize, 'dsize')
    const r = warpPerspective(h, im, M.values(), dw, dh, intArg(h, a.flags, INTER.LINEAR), borderType(h, a.borderMode, BORDER.CONSTANT), scalarOf(h, a.borderValue))
    return toValue(r, r.dp)
  })
  def('getRotationMatrix2D', 'center angle scale', 'create', (a) => {
    required(h, a, 'getRotationMatrix2D', 'center', 'angle', 'scale')
    const c = h.iterate(a.center!).map((x) => fround(h.num(x)))
    if (c.length !== 2) throw cvErr(h, '-5:Bad argument', "Can't parse 'center'. Expected sequence length 2")
    const ang = (h.num(a.angle!) * Math.PI) / 180
    const sc = h.num(a.scale!)
    const al = Math.cos(ang) * sc
    const be = Math.sin(ang) * sc
    return py.arr(NDArray.create([al, be, (1 - al) * c[0] - be * c[1], -be, al, be * c[0] + (1 - al) * c[1]], [2, 3], 'float64'))
  })
  def('getAffineTransform', 'src dst', 'create', (a) => {
    required(h, a, 'getAffineTransform', 'src', 'dst')
    const s = points(h, a.src, 3, 'src')
    const d = points(h, a.dst, 3, 'dst')
    const A: number[][] = []
    const b: number[] = []
    for (let i = 0; i < 3; i++) {
      A.push([s[i][0], s[i][1], 1, 0, 0, 0], [0, 0, 0, s[i][0], s[i][1], 1])
      b.push(d[i][0], d[i][1])
    }
    return py.arr(NDArray.create(solve(h, A, b), [2, 3], 'float64'))
  })
  def('getPerspectiveTransform', 'src dst solveMethod', 'create', (a) => {
    required(h, a, 'getPerspectiveTransform', 'src', 'dst')
    const s = points(h, a.src, 4, 'src')
    const d = points(h, a.dst, 4, 'dst')
    const A: number[][] = Array.from({ length: 8 }, () => new Array<number>(8).fill(0))
    const b = new Array<number>(8).fill(0)
    for (let i = 0; i < 4; i++) {
      A[i][0] = A[i + 4][3] = s[i][0]
      A[i][1] = A[i + 4][4] = s[i][1]
      A[i][2] = A[i + 4][5] = 1
      A[i][6] = -s[i][0] * d[i][0]
      A[i][7] = -s[i][1] * d[i][0]
      A[i + 4][6] = -s[i][0] * d[i][1]
      A[i + 4][7] = -s[i][1] * d[i][1]
      b[i] = d[i][0]
      b[i + 4] = d[i][1]
    }
    return py.arr(NDArray.create([...solve(h, A, b), 1], [3, 3], 'float64'))
  })
  def('flip', 'src flipCode dst', 'move', (a) => {
    required(h, a, 'flip', 'src', 'flipCode')
    const m = asMat(h, a.src, 'src')
    const code = h.toInt(a.flipCode!)
    const C = m.ndim === 3 ? m.shape[2] : 1
    const [H, W] = m.shape
    const v = m.values()
    const out = new Array<number>(v.length)
    for (let y = 0; y < H; y++) {
      const yy = code > 0 ? y : H - 1 - y
      for (let x = 0; x < W; x++) {
        const xx = code === 0 ? x : W - 1 - x
        for (let c = 0; c < C; c++) out[(y * W + x) * C + c] = v[(yy * W + xx) * C + c]
      }
    }
    return sameKind(m, out, C === 1 ? [H, W] : [H, W, C])
  })
  def('rotate', 'src rotateCode dst', 'move', (a) => {
    required(h, a, 'rotate', 'src', 'rotateCode')
    const m = asMat(h, a.src, 'src')
    const code = h.toInt(a.rotateCode!)
    if (code < 0 || code > 2) throw cvErr(h, '-215:Assertion failed', 'rotateCode must be ROTATE_90_CLOCKWISE, ROTATE_180 or ROTATE_90_COUNTERCLOCKWISE')
    const C = m.ndim === 3 ? m.shape[2] : 1
    const [H, W] = m.shape
    const [oh, ow] = code === 1 ? [H, W] : [W, H]
    const v = m.values()
    const out = new Array<number>(v.length)
    for (let y = 0; y < oh; y++) {
      for (let x = 0; x < ow; x++) {
        const [sy, sx] = code === 0 ? [H - 1 - x, y] : code === 1 ? [H - 1 - y, W - 1 - x] : [x, W - 1 - y]
        for (let c = 0; c < C; c++) out[(y * ow + x) * C + c] = v[(sy * W + sx) * C + c]
      }
    }
    return sameKind(m, out, C === 1 ? [oh, ow] : [oh, ow, C])
  })
  def('transpose', 'src dst', 'move', (a) => {
    required(h, a, 'transpose', 'src')
    const m = asMat(h, a.src, 'src')
    const C = m.ndim === 3 ? m.shape[2] : 1
    const [H, W] = m.shape
    const v = m.values()
    const out = new Array<number>(v.length)
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let c = 0; c < C; c++) out[(x * H + y) * C + c] = v[(y * W + x) * C + c]
    return sameKind(m, out, C === 1 ? [W, H] : [W, H, C])
  })
  def('copyMakeBorder', 'src top bottom left right borderType dst value', 'create', (a) => {
    required(h, a, 'copyMakeBorder', 'src', 'top', 'bottom', 'left', 'right', 'borderType')
    const im = readImg(h, a.src)
    const [t, b, l, r] = [a.top!, a.bottom!, a.left!, a.right!].map((x) => h.toInt(x))
    if (t < 0 || b < 0 || l < 0 || r < 0) throw cvErr(h, '-215:Assertion failed', 'top >= 0 && bottom >= 0 && left >= 0 && right >= 0 && _src.dims() <= 2')
    const bt = h.toInt(a.borderType!) & ~BORDER.ISOLATED
    if (bt === BORDER.TRANSPARENT || bt < 0 || bt > 4) throw cvErr(h, '-5:Bad argument', 'Unknown/unsupported border type')
    const value = scalarOf(h, a.value)
    const px = sampler(im, bt, value)
    const oh = im.h + t + b
    const ow = im.w + l + r
    const out = new Array<number>(oh * ow * im.c)
    let at = 0
    for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) for (let c = 0; c < im.c; c++) out[at++] = sat(px(y - t, x - l, c), im.dp)
    return toValue({ h: oh, w: ow, c: im.c, v: out }, im.dp)
  })
  def('split', 'm mv', 'create', (a) => {
    required(h, a, 'split', 'm')
    const im = readImg(h, a.m, 'm')
    return py.tuple(Array.from({ length: im.c }, (_, c) => toValue({ h: im.h, w: im.w, c: 1, v: im.v.filter((_v, i) => i % im.c === c) }, im.dp)))
  })
  def('merge', 'mv dst', 'move', (a) => {
    required(h, a, 'merge', 'mv')
    const parts = h.iterate(a.mv!).map((v) => asMat(h, v, 'mv'))
    if (!parts.length) throw cvErr(h, '-215:Assertion failed', 'mv && n > 0')
    const [H, W] = parts[0].shape
    if (parts.some((p) => p.shape[0] !== H || p.shape[1] !== W)) throw cvErr(h, '-215:Assertion failed', 'mv[i].size == mv[0].size && mv[i].depth() == depth')
    const views = parts.map((p) => ({ c: p.ndim === 3 ? p.shape[2] : 1, v: p.values() }))
    const C = views.reduce((n, p) => n + p.c, 0)
    const out = new Array<number>(H * W * C)
    let at = 0
    for (let i = 0; i < H * W; i++) for (const p of views) for (let c = 0; c < p.c; c++) out[at++] = p.v[i * p.c + c]
    return sameKind(parts[0], out, C === 1 ? [H, W] : [H, W, C])
  })
  const concat = (name: string, vertical: boolean) => def(name, 'src dst', 'move', (a) => {
    required(h, a, name, 'src')
    const parts = h.iterate(a.src!).map((v) => asMat(h, v, 'src'))
    const [H0, W0] = parts[0].shape
    const C = parts[0].ndim === 3 ? parts[0].shape[2] : 1
    if (parts.some((p) => (vertical ? p.shape[1] !== W0 : p.shape[0] !== H0) || (p.ndim === 3 ? p.shape[2] : 1) !== C)) throw cvErr(h, '-215:Assertion failed', vertical ? 'src[i].dims <= 2 && src[i].cols == src[0].cols && src[i].type() == src[0].type()' : 'src[i].dims <= 2 && src[i].rows == src[0].rows && src[i].type() == src[0].type()')
    if (vertical) {
      const H = parts.reduce((n, p) => n + p.shape[0], 0)
      return sameKind(parts[0], parts.flatMap((p) => p.values()), C === 1 ? [H, W0] : [H, W0, C])
    }
    const W = parts.reduce((n, p) => n + p.shape[1], 0)
    const vals = parts.map((p) => p.values())
    const out: number[] = []
    for (let y = 0; y < H0; y++) parts.forEach((p, k) => { for (let j = 0; j < p.shape[1] * C; j++) out.push(vals[k][y * p.shape[1] * C + j]) })
    return sameKind(parts[0], out, C === 1 ? [H0, W] : [H0, W, C])
  })
  concat('hconcat', false)
  concat('vconcat', true)

  // ---- filters
  def('filter2D', 'src ddepth kernel dst anchor delta borderType', null, (a) => {
    required(h, a, 'filter2D', 'src', 'ddepth', 'kernel')
    const im = readImg(h, a.src)
    no32s(h, im, 'getLinearFilter')
    const { kh, kw, k } = kernelOf(h, a.kernel)
    const [ax, ay] = anchorOf(h, a.anchor, kw, kh)
    const border = borderType(h, a.borderType, BORDER.REFLECT_101)
    if (border === BORDER.WRAP) throw cvErr(h, '-215:Assertion failed', '(borderType & ~BORDER_ISOLATED) != BORDER_WRAP')
    const r = correlate(im, kw, k, ax, ay, numArg(h, a.delta, 0), border, depthArg(h, a.ddepth, im.dp))
    const res = toValue(r, r.dp)
    traceWindow('cv2.filter2D', a.src!, im, res, kh, kw, ax, ay, border)
    return res
  })
  def('sepFilter2D', 'src ddepth kernelX kernelY dst anchor delta borderType', 'create', (a) => {
    required(h, a, 'sepFilter2D', 'src', 'ddepth', 'kernelX', 'kernelY')
    const im = readImg(h, a.src)
    no32s(h, im, 'getLinearRowFilter')
    const kx = kernelOf(h, a.kernelX).k
    const ky = kernelOf(h, a.kernelY).k
    const [ax, ay] = anchorOf(h, a.anchor, kx.length, ky.length)
    const r = sepFilter(im, kx, ky, numArg(h, a.delta, 0), borderType(h, a.borderType, BORDER.REFLECT_101), depthArg(h, a.ddepth, im.dp), ax, ay)
    return toValue(r, r.dp)
  })
  def('blur', 'src ksize dst anchor borderType', null, (a) => {
    required(h, a, 'blur', 'src', 'ksize')
    const im = readImg(h, a.src)
    const [kw, kh] = intPair(h, a.ksize, 'ksize')
    if (kw <= 0 || kh <= 0) throw cvErr(h, '-215:Assertion failed', 'ksize.width > 0 && ksize.height > 0')
    const [ax, ay] = anchorOf(h, a.anchor, kw, kh)
    const border = borderType(h, a.borderType, BORDER.REFLECT_101)
    const r = boxFilter(im, kw, kh, ax, ay, border, true, im.dp)
    const res = toValue(r, r.dp)
    traceWindow('cv2.blur', a.src!, im, res, kh, kw, ax, ay, border)
    return res
  })
  def('boxFilter', 'src ddepth ksize dst anchor normalize borderType', 'create', (a) => {
    required(h, a, 'boxFilter', 'src', 'ddepth', 'ksize')
    const im = readImg(h, a.src)
    const [kw, kh] = intPair(h, a.ksize, 'ksize')
    const [ax, ay] = anchorOf(h, a.anchor, kw, kh)
    const r = boxFilter(im, kw, kh, ax, ay, borderType(h, a.borderType, BORDER.REFLECT_101), !a.normalize || h.truthy(a.normalize), depthArg(h, a.ddepth, im.dp))
    return toValue(r, r.dp)
  })
  def('GaussianBlur', 'src ksize sigmaX dst sigmaY borderType hint', null, (a) => {
    required(h, a, 'GaussianBlur', 'src', 'ksize', 'sigmaX')
    const im = readImg(h, a.src)
    no32s(h, im, 'getLinearRowFilter')
    const ks = intPair(h, a.ksize, 'ksize')
    const sx = h.num(a.sigmaX!)
    const sy = numArg(h, a.sigmaY, 0)
    const border = borderType(h, a.borderType, BORDER.REFLECT_101)
    if (border === BORDER.WRAP) throw cvErr(h, '-215:Assertion failed', '(borderType & ~BORDER_ISOLATED) != BORDER_WRAP')
    const r = gaussianBlur(h, im, ks, sx, sy, border)
    const res = toValue(r, r.dp)
    const [kw, kh] = gaussianSize(h, ks, sx, sy, im.dp)
    traceWindow('cv2.GaussianBlur', a.src!, im, res, kh, kw, kw >> 1, kh >> 1, border)
    return res
  })
  def('getGaussianKernel', 'ksize sigma ktype', 'create', (a) => {
    required(h, a, 'getGaussianKernel', 'ksize', 'sigma')
    const n = h.toInt(a.ksize!)
    return py.arr(NDArray.create(gaussKernel(n, h.num(a.sigma!)), [n, 1], 'float64'))
  })
  def('medianBlur', 'src ksize dst', null, (a) => {
    required(h, a, 'medianBlur', 'src', 'ksize')
    const im = readImg(h, a.src)
    no32s(h, im, 'medianBlur')
    const k = h.toInt(a.ksize!)
    if (k % 2 !== 1) throw cvErr(h, '-215:Assertion failed', 'ksize % 2 == 1')
    if (im.dp !== '8u' && k > 5) throw cvErr(h, '-210:Unsupported format or combination of formats', 'medianBlur with ksize > 5 needs an 8-bit image')
    const r = medianBlur(im, k)
    const res = toValue(r, r.dp)
    traceWindow('cv2.medianBlur', a.src!, im, res, k, k, k >> 1, k >> 1, BORDER.REPLICATE)
    return res
  })
  def('Sobel', 'src ddepth dx dy dst ksize scale delta borderType', null, (a) => {
    required(h, a, 'Sobel', 'src', 'ddepth', 'dx', 'dy')
    const im = readImg(h, a.src)
    no32s(h, im, 'getLinearRowFilter')
    const ks = intArg(h, a.ksize, 3)
    const border = borderType(h, a.borderType, BORDER.REFLECT_101)
    const r = sobel(h, im, depthArg(h, a.ddepth, im.dp), h.toInt(a.dx!), h.toInt(a.dy!), ks, numArg(h, a.scale, 1), numArg(h, a.delta, 0), border)
    const res = toValue(r, r.dp)
    traceWindow('cv2.Sobel', a.src!, im, res, Math.max(ks, 3), Math.max(ks, 3), Math.max(ks, 3) >> 1, Math.max(ks, 3) >> 1, border)
    return res
  })
  def('Scharr', 'src ddepth dx dy dst scale delta borderType', 'create', (a) => {
    required(h, a, 'Scharr', 'src', 'ddepth', 'dx', 'dy')
    const im = readImg(h, a.src)
    no32s(h, im, 'getLinearRowFilter')
    const dx = h.toInt(a.dx!)
    const dy = h.toInt(a.dy!)
    if (dx < 0 || dy < 0 || dx + dy !== 1) throw cvErr(h, '-215:Assertion failed', 'dx >= 0 && dy >= 0 && dx+dy == 1')
    const scale = numArg(h, a.scale, 1)
    const r = sepFilter(im, dx ? SCHARR_D : SCHARR_S, (dy ? SCHARR_D : SCHARR_S).map((v) => v * scale), numArg(h, a.delta, 0), borderType(h, a.borderType, BORDER.REFLECT_101), depthArg(h, a.ddepth, im.dp))
    return toValue(r, r.dp)
  })
  def('Laplacian', 'src ddepth dst ksize scale delta borderType', null, (a) => {
    required(h, a, 'Laplacian', 'src', 'ddepth')
    const im = readImg(h, a.src)
    no32s(h, im, 'getLinearFilter')
    const dp = depthArg(h, a.ddepth, im.dp)
    const ks = intArg(h, a.ksize, 1)
    const scale = numArg(h, a.scale, 1)
    const delta = numArg(h, a.delta, 0)
    const border = borderType(h, a.borderType, BORDER.REFLECT_101)
    let r: Img
    if (ks === 1 || ks === 3) {
      const K = ks === 1 ? [0, 1, 0, 1, -4, 1, 0, 1, 0] : [2, 0, 2, 0, -8, 0, 2, 0, 2]
      r = correlate(im, 3, K.map((v) => v * scale), 1, 1, delta, border, dp)
    } else {
      if (ks % 2 === 0 || ks > 31) throw cvErr(h, '-211:One of the arguments\' values is out of range', 'The kernel size must be odd and not larger than 31')
      const kd = sobelKernel(ks, 2)
      const ks0 = sobelKernel(ks, 0)
      const wide: Depth = dp === '64f' ? '64f' : '32f'
      const d2x = sepFilter(im, kd, ks0, 0, border, wide)
      const d2y = sepFilter(im, ks0, kd, 0, border, wide)
      r = { ...im, dp, v: d2x.v.map((x, i) => sat((x + d2y.v[i]) * scale + delta, dp)) }
    }
    const res = toValue(r, r.dp)
    traceWindow('cv2.Laplacian', a.src!, im, res, Math.max(ks, 3), Math.max(ks, 3), Math.max(ks, 3) >> 1, Math.max(ks, 3) >> 1, border)
    return res
  })
  def('Canny', 'image threshold1 threshold2 edges apertureSize L2gradient', 'create', (a) => {
    required(h, a, 'Canny', 'image', 'threshold1', 'threshold2')
    const im = readImg(h, a.image, 'image')
    need8u(h, im, 'Canny')
    const aperture = intArg(h, a.apertureSize, 3)
    if ((aperture & 1) === 0 || (aperture !== -1 && (aperture < 3 || aperture > 7))) throw cvErr(h, '-206:Bad flag', 'Aperture size should be odd between 3 and 7')
    const L2 = !!(a.L2gradient && h.truthy(a.L2gradient))
    let lo = h.num(a.threshold1!)
    let hi = h.num(a.threshold2!)
    if (lo > hi) [lo, hi] = [hi, lo]
    if (L2) {
      lo = Math.min(32767, lo)
      hi = Math.min(32767, hi)
      if (lo > 0) lo *= lo
      if (hi > 0) hi *= hi
    }
    const low = Math.floor(lo)
    const high = Math.floor(hi)
    const dx = sobel(h, im, '16s', 1, 0, aperture, 1, 0, BORDER.REPLICATE).v
    const dy = sobel(h, im, '16s', 0, 1, aperture, 1, 0, BORDER.REPLICATE).v
    const { h: H, w: W, c: C } = im
    const G = (W + 2)
    const mag = new Array<number>((H + 2) * G).fill(0)
    const gx = new Array<number>(H * W)
    const gy = new Array<number>(H * W)
    for (let i = 0; i < H * W; i++) {
      let best = -1
      for (let c = 0; c < C; c++) {
        const x = dx[i * C + c]
        const y = dy[i * C + c]
        const m = L2 ? x * x + y * y : Math.abs(x) + Math.abs(y)
        if (m > best) {
          best = m
          gx[i] = x
          gy[i] = y
        }
      }
      mag[(Math.floor(i / W) + 1) * G + i % W + 1] = best
    }
    const map = new Array<number>((H + 2) * G).fill(1)
    const stack: number[] = []
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const at = (y + 1) * G + x + 1
        const m = mag[at]
        let isEdge = false
        if (m > low) {
          const xs = gx[y * W + x]
          const ys = gy[y * W + x]
          const ax = Math.abs(xs)
          const ay = Math.abs(ys) * 32768
          const tg22 = ax * 13573
          if (ay < tg22) isEdge = m > mag[at - 1] && m >= mag[at + 1]
          else {
            const tg67 = tg22 + ax * 65536
            if (ay > tg67) isEdge = m > mag[at - G] && m >= mag[at + G]
            else {
              const s = (xs < 0) !== (ys < 0) ? -1 : 1
              isEdge = m > mag[at - G - s] && m > mag[at + G + s]
            }
          }
        }
        if (isEdge) {
          if (m > high) {
            map[at] = 2
            stack.push(at)
          } else map[at] = 0
        }
      }
    }
    while (stack.length) {
      const at = stack.pop()!
      for (const d of [-G - 1, -G, -G + 1, -1, 1, G - 1, G, G + 1]) {
        if (map[at + d] === 0) {
          map[at + d] = 2
          stack.push(at + d)
        }
      }
    }
    const out = new Array<number>(H * W)
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) out[y * W + x] = map[(y + 1) * G + x + 1] === 2 ? 255 : 0
    return toValue({ h: H, w: W, c: 1, v: out }, '8u')
  })

  // ---- thresholds
  def('threshold', 'src thresh maxval type dst', null, (a) => {
    required(h, a, 'threshold', 'src', 'thresh', 'maxval', 'type')
    const im = readImg(h, a.src)
    no32s(h, im, 'threshold')
    const flag = h.toInt(a.type!)
    let t = h.num(a.thresh!)
    const auto = flag & (THRESH.OTSU | THRESH.TRIANGLE)
    if (auto === THRESH.TRIANGLE) throw h.err('NotImplementedError', 'cv2.threshold: THRESH_TRIANGLE is not available in this sandbox')
    if (auto === THRESH.OTSU) {
      if (im.dp !== '8u' || im.c !== 1) throw cvErr(h, '-2:Unspecified error', `THRESH_OTSU mode: 'src_type == CV_8UC1 || src_type == CV_16UC1' where 'src_type' is not 8UC1`)
      t = otsu(im.v)
    }
    const kind = flag & THRESH.MASK
    if (kind > 4) throw cvErr(h, '-5:Bad argument', 'Unknown threshold type')
    const r = thresholdCore(im, t, h.num(a.maxval!), kind)
    const res = toValue({ ...im, v: r.v.map((x) => sat(x, im.dp)) }, im.dp)
    if (h.tracing && res.k === 'array' && im.v.length <= 240) {
      const sa = asMat(h, a.src, 'src')
      h.traceCall('cv2.threshold', 'elementwise', [{ label: h.nameOf(a.src!, 'src'), snap: { shape: [...sa.shape], values: sa.values(), dtype: sa.dtype } }], { shape: [...res.a.shape], values: res.a.values(), dtype: res.a.dtype }, h.str(res))
    }
    return py.tuple([py.float(r.thresh), res])
  })
  def('adaptiveThreshold', 'src maxValue adaptiveMethod thresholdType blockSize C dst', 'create', (a) => {
    required(h, a, 'adaptiveThreshold', 'src', 'maxValue', 'adaptiveMethod', 'thresholdType', 'blockSize', 'C')
    const im = readImg(h, a.src)
    if (im.dp !== '8u' || im.c !== 1) throw cvErr(h, '-215:Assertion failed', 'src.type() == CV_8UC1')
    const bs = h.toInt(a.blockSize!)
    if (bs % 2 !== 1 || bs <= 1) throw cvErr(h, '-215:Assertion failed', 'blockSize % 2 == 1 && blockSize > 1')
    const maxV = h.num(a.maxValue!)
    const method = h.toInt(a.adaptiveMethod!)
    const type = h.toInt(a.thresholdType!)
    const C = h.num(a.C!)
    if (maxV < 0) return toValue({ ...im, v: blank(im) }, '8u')
    if (type !== THRESH.BINARY && type !== THRESH.BINARY_INV) throw cvErr(h, '-206:Bad flag', 'Unknown/unsupported threshold type')
    let mean: number[]
    if (method === 0) mean = boxFilter(im, bs, bs, bs >> 1, bs >> 1, BORDER.REPLICATE, true, '8u').v
    else if (method === 1) mean = sepFilter({ ...im, dp: '32f' }, gaussKernel(bs, 0).map(fround), gaussKernel(bs, 0).map(fround), 0, BORDER.REPLICATE, '32f').v.map((x) => sat(x, '8u'))
    else throw cvErr(h, '-5:Bad argument', 'Unknown/unsupported adaptive threshold method')
    const imax = sat(maxV, '8u')
    const delta = type === THRESH.BINARY ? Math.ceil(C) : Math.floor(C)
    return toValue({ ...im, v: im.v.map((x, i) => {
      const d = x - mean[i]
      return (type === THRESH.BINARY ? d > -delta : d <= -delta) ? imax : 0
    }) }, '8u')
  })
  def('inRange', 'src lowerb upperb dst', 'create', (a) => {
    required(h, a, 'inRange', 'src', 'lowerb', 'upperb')
    const im = readImg(h, a.src)
    const lo = compareBound(h, a.lowerb, im, 'lowerb')
    const hi = compareBound(h, a.upperb, im, 'upperb')
    const out = new Array<number>(im.h * im.w)
    for (let i = 0; i < im.h * im.w; i++) {
      let ok = 255
      for (let c = 0; c < im.c; c++) {
        const x = im.v[i * im.c + c]
        if (x < lo(i, c) || x > hi(i, c)) ok = 0
      }
      out[i] = ok
    }
    return toValue({ h: im.h, w: im.w, c: 1, v: out }, '8u')
  })

  // ---- arithmetic
  const satFn = (op: (x: number, y: number) => number) => (x: number, y: number, dp: Depth) => sat(op(x, y), dp)
  def('add', 'src1 src2 dst mask dtype', 'elementwise', (a) => arith(h, a.src1, a.src2, a.mask, a.dtype, 'add', satFn((x, y) => x + y)))
  def('subtract', 'src1 src2 dst mask dtype', 'elementwise', (a) => arith(h, a.src1, a.src2, a.mask, a.dtype, 'subtract', satFn((x, y) => x - y)))
  def('absdiff', 'src1 src2 dst', 'elementwise', (a) => arith(h, a.src1, a.src2, undefined, undefined, 'absdiff', satFn((x, y) => Math.abs(x - y))))
  def('max', 'src1 src2 dst', 'elementwise', (a) => arith(h, a.src1, a.src2, undefined, undefined, 'max', satFn(Math.max)))
  def('min', 'src1 src2 dst', 'elementwise', (a) => arith(h, a.src1, a.src2, undefined, undefined, 'min', satFn(Math.min)))
  def('multiply', 'src1 src2 dst scale dtype', 'elementwise', (a) => {
    const s = numArg(h, a.scale, 1)
    return arith(h, a.src1, a.src2, undefined, a.dtype, 'multiply', (x, y, dp) => sat(isFloatDepth(dp) || s !== 1 ? fround(fround(fround(s) * x) * y) : x * y, dp))
  })
  def('divide', 'src1 src2 dst scale dtype', 'elementwise', (a) => {
    const s = numArg(h, a.scale, 1)
    return arith(h, a.src1, a.src2, undefined, a.dtype, 'divide', (x, y, dp) => (isFloatDepth(dp) ? sat((x * s) / y, dp) : y === 0 ? 0 : sat(fround(fround(fround(x) * fround(s)) / fround(y)), dp)))
  })
  const bitwise = (name: string, f: (x: number, y: number) => number) => def(name, 'src1 src2 dst mask', 'elementwise', (a) => {
    required(h, a, name, 'src1', 'src2')
    for (const v of [a.src1, a.src2]) if (v && v.k === 'array' && v.a.dtype === 'float64') throw h.err('NotImplementedError', `cv2.${name} works on the raw bits of a float image; this sandbox only offers it on integer images`)
    return arith(h, a.src1, a.src2, a.mask, undefined, name, (x, y) => f(x, y))
  })
  bitwise('bitwise_and', (x, y) => x & y)
  bitwise('bitwise_or', (x, y) => x | y)
  bitwise('bitwise_xor', (x, y) => x ^ y)
  def('bitwise_not', 'src dst mask', 'elementwise', (a) => {
    required(h, a, 'bitwise_not', 'src')
    const im = readImg(h, a.src)
    if (isFloatDepth(im.dp)) throw h.err('NotImplementedError', 'cv2.bitwise_not works on the raw bits of a float image; this sandbox only offers it on integer images')
    const mask = a.mask && a.mask.k !== 'none' ? readImg(h, a.mask, 'mask') : null
    const out = im.v.map((x, i) => (mask && !mask.v[Math.floor(i / im.c)] ? 0 : im.dp === '8u' ? 255 - x : im.dp === '16u' ? 65535 - x : ~x))
    return toValue({ ...im, v: out }, im.dp)
  })
  def('addWeighted', 'src1 alpha src2 beta gamma dst dtype', 'elementwise', (a) => {
    required(h, a, 'addWeighted', 'src1', 'alpha', 'src2', 'beta', 'gamma')
    const al = h.num(a.alpha!)
    const be = h.num(a.beta!)
    const ga = h.num(a.gamma!)
    const fa = fround(al)
    const fb = fround(be)
    const fg = fround(ga)
    // the vector loop, 16 values at a time, does a*alpha + (b*beta + gamma); what is left does (a*alpha + b*beta) + gamma
    return arith(h, a.src1, a.src2, undefined, a.dtype, 'addWeighted', (x, y, dp, i, total) => {
      if (dp === '64f') return x * al + y * be + ga
      const simd = dp === '8u' && i < total - total % 16
      return sat(simd ? fround(fround(x * fa) + fround(fround(y * fb) + fg)) : fround(fround(fround(x * fa) + fround(y * fb)) + fg), dp)
    })
  })
  def('convertScaleAbs', 'src dst alpha beta', 'elementwise', (a) => {
    required(h, a, 'convertScaleAbs', 'src')
    const im = readImg(h, a.src)
    const al = numArg(h, a.alpha, 1)
    const be = numArg(h, a.beta, 0)
    if (im.dp === '64f') return toValue({ ...im, v: im.v.map((x) => sat(Math.abs(x * al + be), '8u')) }, '8u')
    const fa = fround(al)
    const fb = fround(be)
    return toValue({ ...im, v: im.v.map((x) => sat(Math.abs(fround(fround(x * fa) + fb)), '8u')) }, '8u')
  })
  def('normalize', 'src dst alpha beta norm_type dtype mask', 'create', (a) => {
    required(h, a, 'normalize', 'src', 'dst')
    const im = readImg(h, a.src)
    const al = numArg(h, a.alpha, 1)
    const be = numArg(h, a.beta, 0)
    const norm = intArg(h, a.norm_type, 4)
    const dp = depthArg(h, a.dtype, im.dp)
    const mask = a.mask && a.mask.k !== 'none' ? readImg(h, a.mask, 'mask') : null
    const live = (i: number) => !mask || !!mask.v[Math.floor(i / im.c)]
    const vals = im.v.filter((_v, i) => live(i))
    let scale: number
    let shift = 0
    if (norm === 32) {
      const smin = Math.min(...vals)
      const smax = Math.max(...vals)
      const dmin = Math.min(al, be)
      const dmax = Math.max(al, be)
      scale = (dmax - dmin) * (smax - smin > Number.EPSILON ? 1 / (smax - smin) : 0)
      if (dp === '32f') {
        scale = fround(scale)
        shift = fround(dmin) - fround(smin * scale)
      } else shift = dmin - smin * scale
    } else if (norm === 1 || norm === 2 || norm === 4) {
      const n = norm === 1 ? Math.max(...vals.map(Math.abs)) : norm === 2 ? vals.reduce((s, x) => s + Math.abs(x), 0) : Math.sqrt(vals.reduce((s, x) => s + x * x, 0))
      scale = n > Number.EPSILON ? al / n : 0
    } else throw cvErr(h, '-5:Bad argument', 'Unknown/unsupported norm type')
    const wide = im.dp === '64f' || dp === '64f'
    const fa = fround(scale)
    const fb = fround(shift)
    return toValue({ ...im, v: im.v.map((x, i) => (live(i) ? sat(wide ? x * scale + shift : fround(fround(x * fa) + fb), dp) : 0)) }, dp)
  })
  def('norm', 'src1 normType mask', 'reduce', (a) => {
    required(h, a, 'norm', 'src1')
    const v = readImg(h, a.src1, 'src1').v
    const t = intArg(h, a.normType, 4)
    if (t === 1) return py.float(Math.max(...v.map(Math.abs)))
    if (t === 2) return py.float(v.reduce((s, x) => s + Math.abs(x), 0))
    if (t === 4) return py.float(Math.sqrt(v.reduce((s, x) => s + x * x, 0)))
    throw h.err('NotImplementedError', 'cv2.norm: only NORM_INF, NORM_L1 and NORM_L2 are available in this sandbox')
  })
  def('magnitude', 'x y magnitude', 'elementwise', (a) => {
    required(h, a, 'magnitude', 'x', 'y')
    const x = readImg(h, a.x, 'x')
    const y = readImg(h, a.y, 'y')
    if (!isFloatDepth(x.dp) || !isFloatDepth(y.dp)) throw cvErr(h, '-215:Assertion failed', "src1.getObj() != dst1.getObj() && src1.depth() == CV_32F || CV_64F: magnitude needs float images")
    if (x.h !== y.h || x.w !== y.w || x.c !== y.c) throw cvErr(h, '-215:Assertion failed', 'X.size == Y.size && type == Y.type() && (depth == CV_32F || depth == CV_64F)')
    return toValue({ ...x, v: x.v.map((p, i) => sat(Math.sqrt(p * p + y.v[i] * y.v[i]), x.dp)) }, x.dp)
  })
  def('countNonZero', 'src', 'reduce', (a) => {
    required(h, a, 'countNonZero', 'src')
    const im = readImg(h, a.src)
    if (im.c !== 1) throw cvErr(h, '-215:Assertion failed', "cn == 1 in function 'countNonZero'")
    return py.int(im.v.filter((x) => x !== 0).length)
  })
  def('minMaxLoc', 'src mask', 'reduce', (a) => {
    required(h, a, 'minMaxLoc', 'src')
    const im = readImg(h, a.src)
    if (im.c !== 1) throw cvErr(h, '-215:Assertion failed', "(cn == 1 && (_mask.empty() || _mask.type() == CV_8U)) || (cn > 1 && _mask.empty() && !minIdx && !maxIdx) in function 'minMaxIdx'")
    const mask = a.mask && a.mask.k !== 'none' ? readImg(h, a.mask, 'mask') : null
    let lo = -1
    let hi = -1
    im.v.forEach((x, i) => {
      if (mask && !mask.v[i]) return
      if (lo < 0 || x < im.v[lo]) lo = i
      if (hi < 0 || x > im.v[hi]) hi = i
    })
    const loc = (i: number) => py.tuple([py.int(i < 0 ? -1 : i % im.w), py.int(i < 0 ? -1 : Math.floor(i / im.w))])
    return py.tuple([py.float(lo < 0 ? 0 : im.v[lo]), py.float(hi < 0 ? 0 : im.v[hi]), loc(lo), loc(hi)])
  })
  def('mean', 'src mask', 'reduce', (a) => {
    required(h, a, 'mean', 'src')
    const im = readImg(h, a.src)
    const mask = a.mask && a.mask.k !== 'none' ? readImg(h, a.mask, 'mask') : null
    const out = [0, 0, 0, 0]
    let n = 0
    for (let i = 0; i < im.h * im.w; i++) {
      if (mask && !mask.v[i]) continue
      n++
      for (let c = 0; c < Math.min(im.c, 4); c++) out[c] += im.v[i * im.c + c]
    }
    return py.tuple(out.map((x) => py.float(n ? x * (1 / n) : 0)))
  })
  def('sumElems', 'src', 'reduce', (a) => {
    required(h, a, 'sumElems', 'src')
    const im = readImg(h, a.src)
    const out = [0, 0, 0, 0]
    for (let i = 0; i < im.h * im.w; i++) for (let c = 0; c < Math.min(im.c, 4); c++) out[c] += im.v[i * im.c + c]
    return py.tuple(out.map((x) => py.float(x)))
  })
  def('LUT', 'src lut dst', 'create', (a) => {
    required(h, a, 'LUT', 'src', 'lut')
    const im = readImg(h, a.src)
    need8u(h, im, 'LUT')
    const lut = asMat(h, a.lut, 'lut')
    if (lut.size !== 256) throw cvErr(h, '-215:Assertion failed', '(lutcn == cn || lutcn == 1) && _lut.total() == 256 && _lut.isContinuous() && (depth == CV_8U || depth == CV_8S)')
    const t = lut.values()
    return toValue({ ...im, v: im.v.map((x) => t[x]) }, depthOf(lut))
  })

  // ---- histograms
  def('calcHist', 'images channels mask histSize ranges hist accumulate', 'create', (a) => {
    required(h, a, 'calcHist', 'images', 'channels', 'mask', 'histSize', 'ranges')
    if ((a.hist && a.hist.k !== 'none') || (a.accumulate && h.truthy(a.accumulate))) throw h.err('NotImplementedError', 'cv2.calcHist: hist and accumulate are not available in this sandbox')
    const imgs = h.iterate(a.images!).map((v) => readImg(h, v, 'images'))
    for (const im of imgs) if (im.dp !== '8u' && im.dp !== '16u' && !isFloatDepth(im.dp)) cvErr(h, '-210:Unsupported format or combination of formats', `in function 'calcHist': ${im.dp === '32s' ? 'the array has dtype int64, which OpenCV reads as CV_32S. An image needs dtype=np.uint8' : `depth ${im.dp}`}`)
    const chans = h.iterate(a.channels!).map((x) => h.toInt(x))
    const sizes = h.iterate(a.histSize!).map((x) => h.toInt(x))
    const rng = h.iterate(a.ranges!).map((x) => h.num(x))
    if (chans.length !== sizes.length || rng.length !== sizes.length * 2) throw cvErr(h, '-215:Assertion failed', 'rsz == dims*2 || rsz == 0')
    const first = imgs[0]
    const mask = a.mask!.k !== 'none' ? readImg(h, a.mask, 'mask') : null
    if (mask && (mask.h !== first.h || mask.w !== first.w)) throw cvErr(h, '-215:Assertion failed', 'mask.size() == images[0].size()')
    const where = chans.map((ch) => {
      let k = ch
      for (const im of imgs) {
        if (k < im.c) return { im, k }
        k -= im.c
      }
      throw cvErr(h, '-215:Assertion failed', 'channels[i] is out of range')
    })
    const st = cStrides(sizes)
    const out = new Array<number>(sizes.reduce((p, x) => p * x, 1)).fill(0)
    const a1 = sizes.map((s, d) => s / (rng[d * 2 + 1] - rng[d * 2]))
    const b1 = sizes.map((_s, d) => -a1[d] * rng[d * 2])
    for (let i = 0; i < first.h * first.w; i++) {
      if (mask && !mask.v[i]) continue
      let at = 0
      let ok = true
      for (let d = 0; d < sizes.length; d++) {
        const { im, k } = where[d]
        const x = im.v[i * im.c + k]
        if (x < rng[d * 2] || x >= rng[d * 2 + 1]) { ok = false; break }
        const idx = Math.floor(x * a1[d] + b1[d])
        if (idx < 0 || idx >= sizes[d]) { ok = false; break }
        at += idx * st[d]
      }
      if (ok) out[at]++
    }
    return py.arr(NDArray.create(out, sizes.length === 1 ? [sizes[0], 1] : sizes, 'float64'))
  })
  def('equalizeHist', 'src dst', 'create', (a) => {
    required(h, a, 'equalizeHist', 'src')
    const im = readImg(h, a.src)
    if (im.dp !== '8u' || im.c !== 1) throw cvErr(h, '-215:Assertion failed', '_src.type() == CV_8UC1')
    const hist = new Array<number>(256).fill(0)
    for (const x of im.v) hist[x]++
    let i = 0
    while (!hist[i]) i++
    const total = im.v.length
    if (hist[i] === total) return toValue({ ...im, v: [...im.v] }, '8u')
    const scale = fround(255 / (total - hist[i]))
    const lut = new Array<number>(256).fill(0)
    let sum = 0
    for (lut[i++] = 0; i < 256; i++) {
      sum += hist[i]
      lut[i] = sat(fround(sum * scale), '8u')
    }
    return toValue({ ...im, v: im.v.map((x) => lut[x]) }, '8u')
  })

  // ---- morphology
  const morphArgs = (a: Args): MorphOpts => {
    const kv = a.kernel
    const { kh, kw, k } = !kv || kv.k === 'none' ? { kh: 0, kw: 0, k: [] as number[] } : kernelOf(h, kv)
    const [ax, ay] = k.length ? anchorOf(h, a.anchor, kw, kh) : [0, 0]
    const border = borderType(h, a.borderType, BORDER.CONSTANT)
    return { kh, kw, k, ax, ay, iter: intArg(h, a.iterations, 1), border, value: a.borderValue && a.borderValue.k !== 'none' ? scalarOf(h, a.borderValue) : null }
  }
  const morphFn = (name: string, isMax: boolean) => def(name, 'src kernel dst anchor iterations borderType borderValue', null, (a) => {
    required(h, a, name, 'src', 'kernel')
    const im = readImg(h, a.src)
    no32s(h, im, 'getMorphologyRowFilter')
    const o = morphArgs(a)
    const r = morph(im, isMax, o)
    const res = toValue(r, r.dp)
    if (o.k.length && o.iter === 1) traceWindow(`cv2.${name}`, a.src!, im, res, o.kh, o.kw, o.ax, o.ay, o.border)
    return res
  })
  morphFn('erode', false)
  morphFn('dilate', true)
  def('morphologyEx', 'src op kernel dst anchor iterations borderType borderValue', 'create', (a) => {
    required(h, a, 'morphologyEx', 'src', 'op', 'kernel')
    const im = readImg(h, a.src)
    no32s(h, im, 'getMorphologyRowFilter')
    const op = h.toInt(a.op!)
    const o = morphArgs(a)
    if (!o.k.length) {
      o.kh = o.kw = 3
      o.k = new Array<number>(9).fill(1)
      o.ax = o.ay = 1
    }
    const er = (x: Img) => morph(x, false, o)
    const di = (x: Img) => morph(x, true, o)
    const diff = (x: Img, y: Img) => ({ ...x, v: x.v.map((p, i) => sat(p - y.v[i], x.dp)) })
    let r: Img
    switch (op) {
      case 0: r = er(im); break
      case 1: r = di(im); break
      case 2: r = di(er(im)); break
      case 3: r = er(di(im)); break
      case 4: r = diff(di(im), er(im)); break
      case 5: r = diff(im, di(er(im))); break
      case 6: r = diff(er(di(im)), im); break
      case 7: {
        if (im.dp !== '8u' || im.c !== 1) throw cvErr(h, '-215:Assertion failed', 'src.type() == CV_8UC1')
        const k1 = o.k.map((v) => (v === 1 ? 1 : 0))
        const k2 = o.k.map((v) => (v === -1 ? 1 : 0))
        const e1 = k1.some(Boolean) ? morph(im, false, { ...o, k: k1 }).v : im.v.map(() => 255)
        const e2 = k2.some(Boolean) ? morph({ ...im, v: im.v.map((x) => 255 - x) }, false, { ...o, k: k2 }).v : im.v.map(() => 255)
        r = { ...im, v: e1.map((x, i) => x & e2[i]) }
        break
      }
      default: throw cvErr(h, '-206:Bad flag', 'unknown morphological operation')
    }
    return toValue(r, r.dp)
  })
  def('getStructuringElement', 'shape ksize anchor', 'create', (a) => {
    required(h, a, 'getStructuringElement', 'shape', 'ksize')
    const shape0 = h.toInt(a.shape!)
    const [kw, kh] = intPair(h, a.ksize, 'ksize')
    const [ax, ay] = anchorOf(h, a.anchor, kw, kh)
    if (shape0 < 0 || shape0 > 2) throw cvErr(h, '-215:Assertion failed', 'shape == MORPH_RECT || shape == MORPH_CROSS || shape == MORPH_ELLIPSE')
    const shape = kw === 1 && kh === 1 ? 0 : shape0
    const r = shape === 2 ? kh >> 1 : 0
    const c = shape === 2 ? kw >> 1 : 0
    const inv = r ? 1 / (r * r) : 0
    const out = new Array<number>(kh * kw).fill(0)
    for (let i = 0; i < kh; i++) {
      let j1 = 0
      let j2 = 0
      if (shape === 0 || (shape === 1 && i === ay)) j2 = kw
      else if (shape === 1) {
        j1 = ax
        j2 = j1 + 1
      } else {
        const dy = i - r
        if (Math.abs(dy) <= r) {
          const dx = cvRound(c * Math.sqrt((r * r - dy * dy) * inv))
          j1 = Math.max(c - dx, 0)
          j2 = Math.min(c + dx + 1, kw)
        }
      }
      for (let j = j1; j < j2; j++) out[i * kw + j] = 1
    }
    return py.arr(NDArray.create(out, [kh, kw], 'uint8'))
  })

  // ---- constants
  const ints: Record<string, number> = {
    ...COLOR,
    INTER_NEAREST: INTER.NEAREST, INTER_LINEAR: INTER.LINEAR, INTER_CUBIC: INTER.CUBIC, INTER_AREA: INTER.AREA, INTER_LANCZOS4: INTER.LANCZOS4, INTER_LINEAR_EXACT: INTER.LINEAR_EXACT, INTER_NEAREST_EXACT: INTER.NEAREST_EXACT, INTER_MAX: INTER.MAX,
    WARP_FILL_OUTLIERS: INTER.WARP_FILL_OUTLIERS, WARP_INVERSE_MAP: INTER.WARP_INVERSE_MAP,
    BORDER_CONSTANT: 0, BORDER_REPLICATE: 1, BORDER_REFLECT: 2, BORDER_WRAP: 3, BORDER_REFLECT_101: 4, BORDER_REFLECT101: 4, BORDER_DEFAULT: 4, BORDER_TRANSPARENT: 5, BORDER_ISOLATED: 16,
    THRESH_BINARY: 0, THRESH_BINARY_INV: 1, THRESH_TRUNC: 2, THRESH_TOZERO: 3, THRESH_TOZERO_INV: 4, THRESH_MASK: 7, THRESH_OTSU: 8, THRESH_TRIANGLE: 16,
    ADAPTIVE_THRESH_MEAN_C: 0, ADAPTIVE_THRESH_GAUSSIAN_C: 1,
    CV_8U: 0, CV_8S: 1, CV_16U: 2, CV_16S: 3, CV_32S: 4, CV_32F: 5, CV_64F: 6, CV_8UC1: 0, CV_8UC3: 16, CV_8UC4: 24, CV_32FC1: 5, CV_32FC3: 21, CV_64FC1: 6, CV_64FC3: 22,
    ROTATE_90_CLOCKWISE: 0, ROTATE_180: 1, ROTATE_90_COUNTERCLOCKWISE: 2,
    NORM_INF: 1, NORM_L1: 2, NORM_L2: 4, NORM_MINMAX: 32,
    MORPH_ERODE: 0, MORPH_DILATE: 1, MORPH_OPEN: 2, MORPH_CLOSE: 3, MORPH_GRADIENT: 4, MORPH_TOPHAT: 5, MORPH_BLACKHAT: 6, MORPH_HITMISS: 7, MORPH_RECT: 0, MORPH_CROSS: 1, MORPH_ELLIPSE: 2,
    HIST_CORREL: 0, IMREAD_COLOR: 1, IMREAD_GRAYSCALE: 0, IMREAD_UNCHANGED: -1,
  }
  for (const [k, v] of Object.entries(ints)) attrs[k] = py.int(v)

  return { cv2: { k: 'module', name: 'cv2', attrs } }
}

export const CV2: PyLib = { modules: ['cv2'], load: build }
