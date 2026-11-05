/**
 * matplotlib for the sandbox: figures are plain data (FigureSpec) that the playground draws as SVG. Limits, ticks and
 * layout follow matplotlib's own algorithms (autoscale with margins and sticky edges, AutoLocator, ScalarFormatter,
 * SubplotParams), so what get_xlim() prints is what gets drawn. Text widths are estimated, since nothing is rendered here.
 */
import { NDArray } from './ndarray'
import type { DType } from './ndarray'
import { PyDict, PyObj, formatSpec, isNum, objId, py, repr as pyRepr, toArray } from './minipy'
import type { Host, Kw, PyLib, Value } from './minipy'
import type { Artist, AxesSpec, AxisSpec, BarArtist, Coord, Dash, Display, FigureSpec, ImageArtist, LegendEntry, LegendSpec, LineArtist, Marker, MeshArtist, PolyArtist, ScatterArtist, SegsArtist, SpineSpec, TextArrow, TextArtist } from './pyEvents'
import { CSS4, LISTED, LUT6, LUT8, RAMPS, RC_DEFAULTS, SAMPLED8, SEGMENTS } from './pyPlotCmaps'
import { delaunay, gaussianKde, gridBands, gridLines, triBands, triLines } from './pyPlotGeom'
import type { Path } from './pyPlotGeom'
import { AUTO, MAXN_STEPS, autoMinor, formatG, linearTicks, logLabels, logNonsingular, logTicks, maxNTicks, multipleTicks, nonsingular, pyRound, scalarLabels, validateSteps } from './pyPlotTicks'
import type { MaxN } from './pyPlotTicks'

const fnv = (name: string, call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })
const typ = (name: string, call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'type', name, call })
const addr = (o: object) => `0x${(0x10a000000 + objId(o) * 48).toString(16)}`
const floats = (xs: number[], shape: number[] = [xs.length]): Value => py.arr(NDArray.create(xs, shape, 'float64'))
const isNone = (v: Value | undefined): boolean => !v || v.k === 'none'
const pop = (kw: Kw, ...names: string[]): Value | undefined => {
  let out: Value | undefined
  for (const n of names) {
    if (n in kw) {
      out ??= kw[n]
      delete kw[n]
    }
  }
  return out
}
const dictOf = (entries: [string, Value][]): Value => ({ k: 'dict', d: PyDict.from(entries.map(([a, b]) => [py.str(a), b])) })
/** the keyword arguments a dict= parameter holds */
const kwOfDict = (h: Host, v: Value | undefined): Kw => {
  if (isNone(v)) return {}
  if (v!.k !== 'dict') throw h.err('TypeError', 'expected a dict')
  return Object.fromEntries(v!.d.items().map(([a, b]) => [h.str(a), b]))
}
const oneOf = (h: Host, v: string, name: string, allowed: string[]) => {
  if (!allowed.includes(v)) throw h.err('ValueError', `'${v}' is not a valid value for ${name}; supported values are ${allowed.map((a) => `'${a}'`).join(', ')}`)
  return v
}

// ---------------------------------------------------------------- colours

/** matplotlib's default property cycle (tab10) */
export const CYCLE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

const NAMED: Record<string, string> = {
  ...Object.fromEntries(CSS4.split(' ').map((e) => { const [n, c] = e.split(':'); return [n, `#${c.toLowerCase()}`] })),
  b: '#0000ff', g: '#008000', r: '#ff0000', c: '#00bfbf', m: '#bf00bf', y: '#bfbf00', k: '#000000', w: '#ffffff',
  'tab:blue': CYCLE[0], 'tab:orange': CYCLE[1], 'tab:green': CYCLE[2], 'tab:red': CYCLE[3], 'tab:purple': CYCLE[4],
  'tab:brown': CYCLE[5], 'tab:pink': CYCLE[6], 'tab:gray': CYCLE[7], 'tab:olive': CYCLE[8], 'tab:cyan': CYCLE[9],
}

const hex2 = (x: number) => Math.round(Math.max(0, Math.min(1, x)) * 255).toString(16).padStart(2, '0')
const rgbHex = (r: number, g: number, b: number, a = 1) => `#${hex2(r)}${hex2(g)}${hex2(b)}${a < 1 ? hex2(a) : ''}`

/**
 * Colours travel as 8-bit strings, which the figure needs; what matplotlib would hand back for one that is not a whole
 * byte ('g' is 0.5, a tuple keeps its floats) is remembered here. intAlpha: names and tuples give alpha as the integer 1.
 */
const EXACT = new Map<string, { rgba: number[]; intAlpha: boolean }>()
const BASE_RGB: Record<string, number[]> = { b: [0, 0, 1], g: [0, 0.5, 0], r: [1, 0, 0], c: [0, 0.75, 0.75], m: [0.75, 0, 0.75], y: [0.75, 0.75, 0], k: [0, 0, 0], w: [1, 1, 1] }

const BAD_COLOR = "supported inputs are (r, g, b) and (r, g, b, a) 0-1 float tuples; '#rrggbb', '#rrggbbaa', '#rgb', '#rgba' strings; named color strings; string reprs of 0-1 floats for grayscale values; 'C0', 'C1', ... strings for colors of the color cycle; and pairs combining one of the above with an alpha value"

const parseColor = (s0: string): string | null => {
  const s = s0.trim()
  if (s.toLowerCase() === 'none') return 'none'
  if (/^#[0-9a-fA-F]{3,4}$/.test(s)) return `#${[...s.slice(1)].map((c) => c + c).join('')}`.toLowerCase().replace(/^(#.{6})ff$/, '$1')
  if (/^#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?$/.test(s)) return s.toLowerCase().replace(/^(#.{6})ff$/, '$1')
  if (/^C\d+$/.test(s)) return CYCLE[Number(s.slice(1)) % 10]
  if (/^(\d*\.?\d+)$/.test(s) && Number(s) >= 0 && Number(s) <= 1) {
    const g = Number(s)
    const hex = rgbHex(g, g, g)
    EXACT.set(hex, { rgba: [g, g, g, 1], intAlpha: true })
    return hex
  }
  const named = NAMED[s] ?? NAMED[s.toLowerCase()] ?? null
  if (named) {
    const base = BASE_RGB[s]
    if (base) EXACT.set(named, { rgba: [...base, 1], intAlpha: true })
    else EXACT.delete(named)
  }
  return named
}

/** a matplotlib colour spec → CSS colour ('#rrggbb', '#rrggbbaa' or 'none') */
export const toColor = (h: Host, v: Value, what = 'color'): string => {
  const bad = () => h.err('ValueError', `${pyRepr(v)} is not a valid value for ${what}: ${BAD_COLOR}`)
  if (v.k === 'str') {
    const c = parseColor(v.v)
    if (!c) throw bad()
    return c
  }
  if (v.k === 'tuple' && v.items.length === 2 && v.items[0].k !== 'int' && v.items[0].k !== 'float') {
    const a = h.num(v.items[1])
    const base = toColor(h, v.items[0], what)
    return base === 'none' ? base : rgbHex(...rgbaOf(base).slice(0, 3) as [number, number, number], a)
  }
  if (v.k === 'tuple' || v.k === 'list' || v.k === 'array') {
    const xs = v.k === 'array' ? v.a.values() : v.items.map((x) => (isNum(x) ? h.num(x) : NaN))
    if (xs.length !== 3 && xs.length !== 4) throw bad()
    if (xs.some((x) => !(x >= 0 && x <= 1))) throw h.err('ValueError', 'RGBA values should be within 0-1 range')
    const hex = rgbHex(xs[0], xs[1], xs[2], xs[3] ?? 1)
    EXACT.set(hex, { rgba: [xs[0], xs[1], xs[2], xs[3] ?? 1], intAlpha: xs.length === 3 })
    return hex
  }
  throw bad()
}

/** '#rrggbb[aa]' → floats; 'none' → all zero */
const rgbaOf = (c: string): [number, number, number, number] => {
  if (!c.startsWith('#')) return [0, 0, 0, 0]
  const exact = EXACT.get(c)
  if (exact) return exact.rgba as [number, number, number, number]
  return [parseInt(c.slice(1, 3), 16) / 255, parseInt(c.slice(3, 5), 16) / 255, parseInt(c.slice(5, 7), 16) / 255, c.length > 7 ? parseInt(c.slice(7, 9), 16) / 255 : 1]
}
/** a colour as matplotlib returns it, with an artist's alpha in place of its own */
const rgbaValue = (c: string, alpha: number | null = null): Value => {
  const [r, g, b, a] = rgbaOf(c)
  return py.tuple([py.float(r), py.float(g), py.float(b), alpha !== null ? py.float(alpha) : EXACT.get(c)?.intAlpha ? py.int(1) : py.float(a)])
}
/** several colours from one argument: a single colour, a list of colours, or an N×3 / N×4 array */
const toColors = (h: Host, v: Value, what = 'color'): string[] => {
  if (v.k === 'array' && v.a.ndim === 2) {
    const [n, m] = v.a.shape
    const xs = v.a.values()
    return Array.from({ length: n }, (_, i) => rgbHex(xs[i * m], xs[i * m + 1], xs[i * m + 2], m > 3 ? xs[i * m + 3] : 1))
  }
  if (v.k === 'str') return [toColor(h, v, what)]
  if ((v.k === 'list' || v.k === 'tuple') && (v.items.length === 0 || !isNum(v.items[0]))) return v.items.map((c) => toColor(h, c, what))
  return [toColor(h, v, what)]
}

// ---------------------------------------------------------------- colormaps

export interface Lut {
  name: string
  n: number
  /** 3 × n floats */
  rgb: number[]
  listed: boolean
  under: string | null
  over: string | null
  bad: string | null
  hex?: string[]
  /** a ListedColormap's colours as they were given */
  raw?: Value
}

const b64 = (s: string) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0))
const hexList = (s: string) => Array.from({ length: s.length / 6 }, (_, i) => [0, 2, 4].map((k) => parseInt(s.slice(i * 6 + k, i * 6 + k + 2), 16) / 255))

/** matplotlib's _create_lookup_table: one channel, segments as flat [x, y0, y1, …] */
const channelLut = (n: number, seg: number[]): number[] => {
  const m = seg.length / 3
  const xs = Array.from({ length: m }, (_, i) => seg[i * 3] * (n - 1))
  const y0 = (i: number) => seg[i * 3 + 1]
  const y1 = (i: number) => seg[i * 3 + 2]
  const clip = (v: number) => Math.max(0, Math.min(1, v))
  if (n === 1) return [clip(y0(m - 1))]
  const step = 1 / (n - 1)
  const out = new Array<number>(n)
  out[0] = clip(y1(0))
  out[n - 1] = clip(y0(m - 1))
  for (let i = 1; i < n - 1; i++) {
    const xi = (n - 1) * (i * step)
    let k = xs.findIndex((x) => x >= xi)
    if (k < 1) k = 1
    const d = (xi - xs[k - 1]) / (xs[k] - xs[k - 1])
    out[i] = clip(d * (y0(k) - y1(k - 1)) + y1(k - 1))
  }
  return out
}

const ALIASES: Record<string, string> = { grey: 'gray', gist_grey: 'gist_gray', gist_yerg: 'gist_yarg', Grays: 'Greys' }
const SEABORN = new Set(['rocket', 'mako', 'icefire', 'vlag', 'flare', 'crest'])
const BASE_NAMES = [...Object.keys(LUT6), ...Object.keys(RAMPS), ...Object.keys(LISTED), ...Object.keys(SEGMENTS), ...Object.keys(LUT8), ...Object.keys(SAMPLED8), ...Object.keys(ALIASES)]
export const CMAP_NAMES = BASE_NAMES.filter((n) => !SEABORN.has(n)).sort()
const ALL_CMAPS = CMAP_NAMES.flatMap((n) => [n, `${n}_r`]).sort()

const resampleListed = (src: Lut, n: number): number[] => {
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    const x = n === 1 ? 0 : i / (n - 1)
    const k = x >= 1 ? src.n - 1 : Math.floor(x * src.n)
    out.push(src.rgb[k * 3], src.rgb[k * 3 + 1], src.rgb[k * 3 + 2])
  }
  return out
}

const makeLut = (base: string, n?: number): Lut | null => {
  const name = ALIASES[base] ?? base
  const fromChannels = (ch: number[][], size: number) => Array.from({ length: size * 3 }, (_, i) => ch[i % 3][Math.floor(i / 3)])
  const mk = (rgb: number[], size: number, listed: boolean): Lut => ({ name: base, n: size, rgb, listed, under: null, over: null, bad: null })
  if (LUT6[name]) {
    const bytes = b64(LUT6[name])
    const rgb = Array.from({ length: 768 }, (_, i) => ((bytes[i * 3] << 16) | (bytes[i * 3 + 1] << 8) | bytes[i * 3 + 2]) / 1e6)
    const lut = mk(rgb, 256, true)
    return n ? mk(resampleListed(lut, n), n, true) : lut
  }
  if (LISTED[name]) {
    const cols = hexList(LISTED[name])
    const lut = mk(cols.flat(), cols.length, true)
    return n ? mk(resampleListed(lut, n), n, true) : lut
  }
  const size = n ?? 256
  if (RAMPS[name]) {
    const cols = hexList(RAMPS[name])
    const seg = (c: number) => cols.flatMap((p, i) => [i / (cols.length - 1), p[c], p[c]])
    return mk(fromChannels([0, 1, 2].map((c) => channelLut(size, seg(c))), size), size, false)
  }
  if (SEGMENTS[name]) return mk(fromChannels(SEGMENTS[name].map((s) => channelLut(size, s)), size), size, false)
  if (LUT8[name]) {
    const bytes = b64(LUT8[name])
    const lut = mk(Array.from(bytes, (b) => b / 255), 256, false)
    return n ? mk(resampleListed(lut, n), n, false) : lut
  }
  if (SAMPLED8[name]) {
    const cols = hexList(SAMPLED8[name])
    const seg = (c: number) => cols.flatMap((p, i) => [i / (cols.length - 1), p[c], p[c]])
    return mk(fromChannels([0, 1, 2].map((c) => channelLut(size, seg(c))), size), size, false)
  }
  return null
}

const LUTS = new Map<string, Lut | null>()
const getLut = (name: string): Lut | null => {
  if (LUTS.has(name)) return LUTS.get(name)!
  let lut: Lut | null
  if (name.endsWith('_r')) {
    const b = makeLut(name.slice(0, -2))
    lut = b && reverseLut(b, name)
  } else lut = makeLut(name)
  LUTS.set(name, lut)
  return lut
}
const reverseLut = (b: Lut, name: string): Lut => {
  const rgb: number[] = []
  for (let i = b.n - 1; i >= 0; i--) rgb.push(b.rgb[i * 3], b.rgb[i * 3 + 1], b.rgb[i * 3 + 2])
  return { ...b, name, rgb, under: b.over, over: b.under, hex: undefined }
}
const lutHex = (l: Lut): string[] => (l.hex ??= Array.from({ length: l.n }, (_, i) => rgbHex(l.rgb[i * 3], l.rgb[i * 3 + 1], l.rgb[i * 3 + 2])))
/** Colormap.__call__ for a float: under / over / bad included */
const lutColor = (l: Lut, t: number): string => {
  if (Number.isNaN(t)) return l.bad ?? '#00000000'
  if (t < 0) return l.under ?? lutHex(l)[0]
  if (t > 1) return l.over ?? lutHex(l)[l.n - 1]
  return lutHex(l)[t === 1 ? l.n - 1 : Math.floor(t * l.n)]
}

/** colour of t ∈ [0, 1] under a colormap ('gray_r' reverses) */
export const cmapColor = (name: string, t: number): string => lutColor(getLut(name) ?? getLut('viridis')!, Number.isFinite(t) ? Math.max(0, Math.min(1, t)) : 0)

export const isCmap = (name: string) => getLut(name) !== null

class ColormapObj extends PyObj {
  readonly cls: string
  lut: Lut
  constructor(lut: Lut) {
    super()
    this.lut = lut
    this.cls = lut.listed ? 'ListedColormap' : 'LinearSegmentedColormap'
  }
  isa(n: string) { return n === 'Colormap' }
  repr() { return `<matplotlib.colors.${this.cls} object at ${addr(this)}>` }
  call(args: Value[], kw: Kw, h: Host): Value {
    const X = args[0] ?? kw.X
    const alpha = optNum(h, kw.alpha)
    const bytes = !!kw.bytes && h.truthy(kw.bytes)
    const one = (x: number, isInt: boolean): number[] => {
      const l = this.lut
      if (Number.isNaN(x)) return rgbaOf(l.bad ?? '#00000000')
      let i = isInt ? Math.trunc(x) : x * l.n
      if (!isInt && i === l.n) i = l.n - 1
      let rgba: number[]
      if (i < 0) rgba = l.under ? rgbaOf(l.under) : [l.rgb[0], l.rgb[1], l.rgb[2], 1]
      else if (i >= l.n) rgba = l.over ? rgbaOf(l.over) : [l.rgb[l.n * 3 - 3], l.rgb[l.n * 3 - 2], l.rgb[l.n * 3 - 1], 1]
      else {
        const k = Math.trunc(i)
        rgba = [l.rgb[k * 3], l.rgb[k * 3 + 1], l.rgb[k * 3 + 2], 1]
      }
      if (alpha !== null) rgba = [rgba[0], rgba[1], rgba[2], alpha]
      return bytes ? rgba.map((c) => Math.floor(c * 255)) : rgba
    }
    if (X && isNum(X)) {
      const c = one(h.num(X), X.k === 'int')
      return py.tuple(c.map((v) => (bytes ? py.int(v) : py.float(v))))
    }
    const a = toArray(X)
    const out = a.values().flatMap((x) => one(x, a.dtype === 'int64'))
    return py.arr(NDArray.create(out, [...a.shape, 4], bytes ? 'int64' : 'float64'))
  }
  getAttr(name: string, h: Host): Value | undefined {
    const l = this.lut
    switch (name) {
      case 'N': return py.int(l.n)
      case 'name': return py.str(l.name)
      case 'colors':
        if (!l.listed) return undefined
        if (l.raw) return l.raw
        return py.list(Array.from({ length: l.n }, (_, i) => py.tuple([l.rgb[i * 3], l.rgb[i * 3 + 1], l.rgb[i * 3 + 2]].map(py.float))))
      case 'reversed': return fnv(name, () => py.obj(new ColormapObj(reverseLut(l, l.name.endsWith('_r') ? l.name.slice(0, -2) : `${l.name}_r`))))
      case 'copy': return fnv(name, () => py.obj(new ColormapObj({ ...l })))
      case 'resampled': return fnv(name, (args) => {
        const n = h.toInt(args[0])
        const made = !l.listed && l.name in { ...RAMPS, ...SEGMENTS, ...SAMPLED8 } ? makeLut(l.name, n) : { ...l, n, rgb: resampleListed(l, n), hex: undefined }
        return py.obj(new ColormapObj(made!))
      })
      case 'set_bad': case 'set_under': case 'set_over': return fnv(name, (args, kw) => {
        const c = args[0] ?? kw.color ?? py.str('k')
        l[name.slice(4) as 'bad' | 'under' | 'over'] = toColor(h, c)
        return py.NONE
      })
      case 'with_extremes': return fnv(name, (_a, kw) => {
        const c: Lut = { ...l, hex: undefined }
        if (kw.bad) c.bad = toColor(h, kw.bad)
        if (kw.under) c.under = toColor(h, kw.under)
        if (kw.over) c.over = toColor(h, kw.over)
        return py.obj(new ColormapObj(c))
      })
    }
    return undefined
  }
}

/** Normalize and friends; also what every colour-mapped artist holds */
class NormObj extends PyObj {
  readonly cls: string
  vmin: number | null
  vmax: number | null
  clip: boolean
  center = 0
  bounds: number[] = []
  ncolors = 256
  extend = 'neither'
  constructor(cls: string, vmin: number | null, vmax: number | null, clip = false) {
    super()
    this.cls = cls
    this.vmin = vmin
    this.vmax = vmax
    this.clip = clip
  }
  isa(n: string) { return n === 'Normalize' }
  repr() { return `<matplotlib.colors.${this.cls} object at ${addr(this)}>` }
  autoscaleNone(values: number[]) {
    const f = values.filter((v) => Number.isFinite(v) && (this.cls !== 'LogNorm' || v > 0))
    if (this.cls === 'NoNorm' || this.cls === 'BoundaryNorm') return
    if (this.vmin === null && f.length) this.vmin = Math.min(...sample(f))
    if (this.vmax === null && f.length) this.vmax = Math.max(...sample(f))
  }
  /** value → position along the colormap (0..1, beyond for under / over); BoundaryNorm gives colour indices through indexOf */
  t(v: number): number {
    if (Number.isNaN(v)) return NaN
    const lo = this.vmin ?? 0
    const hi = this.vmax ?? 1
    const c = this.clip ? Math.max(lo, Math.min(hi, v)) : v
    if (lo === hi) return 0
    switch (this.cls) {
      case 'LogNorm': return c <= 0 ? NaN : (Math.log10(c) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))
      case 'TwoSlopeNorm': return c < this.center ? 0.5 * ((c - lo) / (this.center - lo)) : 0.5 + 0.5 * ((c - this.center) / (hi - this.center))
      default: return (c - lo) / (hi - lo)
    }
  }
  /** BoundaryNorm: the colour index of a value */
  indexOf(v: number): number {
    const b = this.bounds
    let i = b.findIndex((x) => v < x)
    i = i < 0 ? b.length : i
    i -= 1
    const regions = b.length - 1
    if (i < 0) return -1
    if (i >= regions) return this.ncolors
    if (this.ncolors > regions) return regions === 1 ? Math.floor((this.ncolors - 1) / 2) : Math.trunc(((this.ncolors - 1) / (regions - 1)) * i)
    return i
  }
  color(l: Lut, v: number): string {
    if (this.cls === 'BoundaryNorm') {
      if (Number.isNaN(v)) return l.bad ?? '#00000000'
      const i = this.indexOf(v)
      return i < 0 ? l.under ?? lutHex(l)[0] : i >= l.n ? l.over ?? lutHex(l)[l.n - 1] : lutHex(l)[i]
    }
    return lutColor(l, this.t(v))
  }
  call(args: Value[], _kw: Kw, h: Host): Value {
    const v = args[0]
    if (this.vmin !== null && this.vmax !== null && this.vmin > this.vmax) throw h.err('ValueError', 'minvalue must be less than or equal to maxvalue')
    if (isNum(v)) {
      if (this.vmin === null || this.vmax === null) this.autoscaleNone([h.num(v)])
      return py.float(this.cls === 'BoundaryNorm' ? this.indexOf(h.num(v)) : this.t(h.num(v)))
    }
    const a = toArray(v, 'float64')
    const vals = a.values()
    this.autoscaleNone(vals)
    return floats(vals.map((x) => (this.cls === 'BoundaryNorm' ? this.indexOf(x) : this.t(x))), a.shape)
  }
  getAttr(name: string): Value | undefined {
    switch (name) {
      case 'vmin': return this.vmin === null ? py.NONE : py.float(this.vmin)
      case 'vmax': return this.vmax === null ? py.NONE : py.float(this.vmax)
      case 'clip': return py.bool(this.clip)
      case 'autoscale': case 'autoscale_None': return fnv(name, (args) => {
        const vals = toArray(args[0], 'float64').values()
        if (name === 'autoscale') this.vmin = this.vmax = null
        this.autoscaleNone(vals)
        return py.NONE
      })
      case 'inverse': return fnv(name, (args) => {
        const lo = this.vmin ?? 0
        const hi = this.vmax ?? 1
        const inv = (t: number) => (this.cls === 'LogNorm' ? lo * (hi / lo) ** t : lo + t * (hi - lo))
        const v = args[0]
        if (isNum(v)) return py.float(inv(numOf(v)))
        const a = toArray(v, 'float64')
        return floats(a.values().map(inv), a.shape)
      })
    }
    return undefined
  }
  setAttr(name: string, v: Value, h: Host): boolean {
    if (name === 'vmin') this.vmin = optNum(h, v)
    else if (name === 'vmax') this.vmax = optNum(h, v)
    else if (name === 'clip') this.clip = h.truthy(v)
    else return false
    return true
  }
}

/** bounded sample for Math.min / Math.max spreads on large images */
const sample = (xs: number[]): number[] => (xs.length < 60000 ? xs : [xs.reduce((a, b) => Math.min(a, b)), xs.reduce((a, b) => Math.max(a, b))])
const minOf = (xs: number[]) => xs.reduce((a, b) => (b < a ? b : a), Infinity)
const maxOf = (xs: number[]) => xs.reduce((a, b) => (b > a ? b : a), -Infinity)
const finiteOf = (xs: number[]) => xs.filter(Number.isFinite)
const numOf = (v: Value) => (v.k === 'int' || v.k === 'float' ? v.v : v.k === 'bool' ? (v.v ? 1 : 0) : NaN)

const optNum = (h: Host, v: Value | undefined): number | null => (v && v.k !== 'none' ? h.num(v) : null)
const optStr = (h: Host, v: Value | undefined): string | null => (v && v.k !== 'none' ? h.str(v) : null)
const flag = (h: Host, v: Value | undefined, dflt: boolean): boolean => (v === undefined ? dflt : h.truthy(v))
const nums = (h: Host, v: Value, what: string): number[] => {
  try {
    return toArray(v, 'float64').values()
  } catch {
    throw h.err('TypeError', `${what}: expected numbers, got '${v.k === 'obj' ? v.o.cls : v.k}'`)
  }
}
const isStrList = (v: Value) => (v.k === 'list' || v.k === 'tuple') && v.items.length > 0 && v.items.every((x) => x.k === 'str')
/** a scalar repeated, or the array itself */
const broadcast = (xs: number[], n: number) => (xs.length === 1 ? new Array<number>(n).fill(xs[0]) : xs)

// ---------------------------------------------------------------- rcParams

interface PlotState {
  figs: Figure[]
  cur: Figure | null
  rc: Map<string, Value>
  interactive: boolean
  image: ArtistObj | null
}

const toValue = (x: unknown): Value => {
  if (x === null || x === undefined) return py.NONE
  if (typeof x === 'number') return Number.isInteger(x) && Math.abs(x) < 1e15 ? py.int(x) : py.float(x)
  if (typeof x === 'boolean') return py.bool(x)
  if (typeof x === 'string') return py.str(x)
  return py.list((x as unknown[]).map((e) => (typeof e === 'number' ? py.float(e) : toValue(e))))
}

const stateOf = (h: Host): PlotState => {
  let s = h.state.get('pyplot') as PlotState | undefined
  if (!s) {
    s = { figs: [], cur: null, rc: new Map(), interactive: false, image: null }
    EXACT.clear()
    h.state.set('pyplot', s)
  }
  return s
}
const rcv = (h: Host, key: string): Value => stateOf(h).rc.get(key) ?? toValue(RC_DEFAULTS[key])
const rcn = (h: Host, key: string): number => h.num(rcv(h, key))
const rcs = (h: Host, key: string): string => h.str(rcv(h, key))
const rcb = (h: Host, key: string): boolean => h.truthy(rcv(h, key))
const rcc = (h: Host, key: string): string => toColor(h, rcv(h, key))

const FONT_SCALE: Record<string, number> = { 'xx-small': 0.579, 'x-small': 0.694, small: 0.833, medium: 1, large: 1.2, 'x-large': 1.44, 'xx-large': 1.728, larger: 1.2, smaller: 0.833 }
/** a font size in points; names scale with rcParams['font.size'] */
const fontPt = (h: Host, v: Value): number => {
  if (v.k === 'str') {
    const f = FONT_SCALE[v.v]
    if (f !== undefined) return f * rcn(h, 'font.size')
    const n = Number(v.v)
    if (!Number.isNaN(n)) return n
    throw h.err('ValueError', `Size is invalid. Valid font size are ${Object.keys(FONT_SCALE).join(', ')}`)
  }
  return h.num(v)
}
const rcFont = (h: Host, key: string): number => fontPt(h, rcv(h, key))

class CyclerObj extends PyObj {
  readonly cls = 'Cycler'
  colors: string[]
  raw: Value[]
  constructor(raw: Value[], colors: string[]) {
    super()
    this.raw = raw
    this.colors = colors
  }
  repr() { return `cycler('color', ${pyRepr(py.list(this.raw))})` }
  getAttr(name: string): Value | undefined {
    if (name === 'by_key') return fnv(name, () => dictOf([['color', py.list(this.raw)]]))
    if (name === 'keys') return py.list([py.str('color')])
    return undefined
  }
  len() { return this.raw.length }
}

const defaultCycle = (h: Host): string[] => {
  const v = stateOf(h).rc.get('axes.prop_cycle')
  return v && v.k === 'obj' && v.o instanceof CyclerObj ? v.o.colors : CYCLE
}

const cyclerOf = (h: Host, args: Value[], kw: Kw): CyclerObj => {
  const key = args.length === 2 ? h.str(args[0]) : Object.keys(kw)[0]
  const vals = args.length === 2 ? args[1] : kw[key]
  if (key !== 'color' && key !== 'c') throw h.err('NotImplementedError', `cycler: only 'color' is available in this sandbox, not '${key}'`)
  const items = h.iterate(vals)
  return new CyclerObj(items, items.map((c) => toColor(h, c)))
}

/** text sizes: nothing is rendered in Python, so widths use DejaVu Sans's average advance */
export const PT = 100 / 72
export const stripMath = (s: string) => {
  if (!s.includes('$')) return s
  return s.replace(/\$([^$]*)\$/g, (_m, b: string) => b.replace(/\\mathdefault|\\mathrm|\\[a-zA-Z]+|[{}^_]/g, (t) => (t.length > 1 && t !== '\\mathdefault' && t !== '\\mathrm' ? 'x' : '')))
}
export const textWidthPx = (s: string, sizePt: number) => Math.max(...stripMath(s).split('\n').map((l) => l.length)) * 0.6 * sizePt * PT
export const textHeightPx = (s: string, sizePt: number) => stripMath(s).split('\n').length * 1.2 * sizePt * PT

// ---------------------------------------------------------------- artists: Python handles over plain specs

type PS<T> = (o: T, a: Value[], h: Host, kw: Kw) => void
type PG<T> = (o: T, h: Host) => Value
interface Prop<T> { g?: PG<T>; s?: PS<T> }
type Props<T> = Record<string, Prop<T>>

const alias = <T>(p: Props<T>, names: Record<string, string>): Props<T> => {
  for (const [a, b] of Object.entries(names)) p[a] = p[b]
  return p
}
/** accepted as keyword arguments and setters, with no effect on the drawing */
const COSMETIC = new Set(['antialiased', 'aa', 'antialiaseds', 'rasterized', 'snap', 'gid', 'url', 'urls', 'picker', 'pickradius', 'animated', 'in_layout', 'mouseover', 'path_effects', 'sketch_params', 'agg_filter', 'clip_box', 'clip_path', 'solid_capstyle', 'solid_joinstyle', 'dash_capstyle', 'dash_joinstyle', 'capstyle', 'joinstyle', 'hatch', 'hatch_linewidth', 'gapcolor', 'figure', 'fontproperties', 'font_properties', 'font', 'family', 'fontfamily', 'fontname', 'name', 'stretch', 'fontstretch', 'variant', 'fontvariant', 'math_fontfamily', 'wrap', 'linespacing', 'multialignment', 'ma', 'rotation_mode', 'usetex', 'parse_math', 'transform_rotates_text', 'offset_transform', 'pickable', 'dash_offset', 'markerfacecoloralt', 'mfcalt'])

const LS_NAMES: Record<string, string> = { solid: '-', dashed: '--', dashdot: '-.', dotted: ':', '-': '-', '--': '--', '-.': '-.', ':': ':', None: 'None', none: 'None', ' ': 'None', '': 'None' }
const LS_DASH: Record<string, Dash> = { '-': null, '--': [3.7, 1.6], '-.': [6.4, 1.6, 1, 1.6], ':': [1, 1.65], None: null }
/** a linestyle: its name, whether a line shows at all, and its dash pattern */
const lineStyle = (h: Host, v: Value, what = 'ls'): { name: string; line: boolean; dash: Dash } => {
  if (v.k === 'none') return { name: 'None', line: false, dash: null }
  if (v.k === 'str') {
    const n = LS_NAMES[v.v]
    if (n === undefined) throw h.err('ValueError', `'${v.v}' is not a valid value for ${what}; supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'`)
    return { name: n, line: n !== 'None', dash: LS_DASH[n] }
  }
  if (v.k === 'tuple' && v.items.length === 2) {
    const pat = isNone(v.items[1]) ? [] : h.iterate(v.items[1]).map((x) => h.num(x))
    return { name: '--', line: true, dash: pat.length ? pat : null }
  }
  throw h.err('ValueError', `linestyle ${pyRepr(v)} is not a valid on-off ink sequence`)
}

const MARKERS = new Set(['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_'])
const markerOf = (h: Host, v: Value): Marker => {
  if (v.k === 'none') return null
  if (v.k === 'str') {
    if (v.v === 'None' || v.v === 'none' || v.v === '' || v.v === ' ') return null
    if (MARKERS.has(v.v)) return v.v
    if (/^\$.*\$$/.test(v.v)) throw h.err('NotImplementedError', `marker ${pyRepr(v)}: mathtext markers are not drawn in this sandbox`)
  }
  throw h.err('ValueError', `Unrecognized marker style ${pyRepr(v)}`)
}
const UNFILLED = new Set(['x', '+', '1', '2', '3', '4', '|', '_', ','])

class TransformObj extends PyObj {
  readonly cls = 'Transform'
  tx: Coord
  ty: Coord
  constructor(tx: Coord, ty: Coord) {
    super()
    this.tx = tx
    this.ty = ty
  }
  repr() { return `<matplotlib.transforms.Transform object at ${addr(this)}>` }
}

const setTransform = (s: { tx: Coord; ty: Coord }, h: Host, v: Value) => {
  if (v.k === 'obj' && v.o instanceof TransformObj) {
    s.tx = v.o.tx
    s.ty = v.o.ty
  } else throw h.err('NotImplementedError', 'transform: only ax.transData, ax.transAxes, fig.transFigure, ax.get_xaxis_transform() and ax.get_yaxis_transform() work in this sandbox')
}

const BASE: Props<ArtistObj> = {
  alpha: {
    g: (o) => (o.alphaSet === null ? py.NONE : py.float(o.alphaSet)),
    s: (o, [v], h) => {
      const a = optNum(h, v)
      if (a !== null && !(a >= 0 && a <= 1)) throw h.err('ValueError', `alpha (${a}) is outside 0-1 range`)
      o.alphaSet = a
      if ('alpha' in o.spec) o.spec.alpha = a ?? 1
    },
  },
  label: {
    g: (o) => py.str(o.labelOf() ?? ''),
    s: (o, [v], h) => o.setLabel(isNone(v) ? null : h.str(v)),
  },
  visible: { g: (o) => py.bool(o.spec.visible), s: (o, [v], h) => { o.spec.visible = h.truthy(v) } },
  zorder: { g: (o) => py.float(o.spec.z), s: (o, [v], h) => { o.spec.z = h.num(v) } },
  clip_on: { g: (o) => py.bool(o.spec.clip), s: (o, [v], h) => { o.spec.clip = h.truthy(v) } },
  transform: { s: (o, [v], h) => setTransform(o.spec, h, v), g: (o) => py.obj(new TransformObj(o.spec.tx, o.spec.ty)) },
}

abstract class ArtistObj extends PyObj {
  abstract readonly cls: string
  abstract spec: Artist
  ax: Axes | null = null
  fig: Figure | null = null
  alphaSet: number | null = null
  sticky: { x: number[]; y: number[] } = { x: [], y: [] }
  abstract props(): Props<ArtistObj>
  labelOf(): string | null { return 'label' in this.spec ? this.spec.label : null }
  setLabel(l: string | null) { if ('label' in this.spec) this.spec.label = l }
  legendEntry(): LegendEntry | null { return null }
  figure(): Figure | null { return this.ax?.fig ?? this.fig }
  touch(h: Host) {
    const f = this.figure()
    if (f) emitFig(h, f, this.ax, this.ax ? this.ax.children.indexOf(this) : null)
  }
  /** update: the wording of Artist.update, which Text's constructor goes through */
  applyKw(kw: Kw, h: Host, update = false) {
    const table = this.props()
    for (const [k, v] of Object.entries(kw)) {
      const p = table[k] ?? BASE[k]
      if (p?.s) p.s(this, [v], h, {})
      else if (!COSMETIC.has(k)) throw h.err('AttributeError', update ? `'${this.cls}' object has no property '${k}'` : `${this.cls}.set() got an unexpected keyword argument '${k}'`)
    }
  }
  /** the artist's python-level position bounds, for relim */
  bounds(): [number[], number[]] | null { return null }
  getAttr(name: string, h: Host): Value | undefined {
    const table = this.props()
    if (name.startsWith('set_')) {
      const key = name.slice(4)
      const p = table[key] ?? BASE[key]
      if (p?.s) return fnv(name, (args, kw) => {
        p.s!(this, args, h, kw)
        this.touch(h)
        return py.NONE
      })
      if (COSMETIC.has(key)) return fnv(name, () => py.NONE)
    } else if (name.startsWith('get_')) {
      const key = name.slice(4)
      const p = table[key] ?? BASE[key]
      if (p?.g) return fnv(name, () => p.g!(this, h))
    }
    switch (name) {
      case 'set': return fnv(name, (_a, kw) => {
        this.applyKw(kw, h)
        this.touch(h)
        return py.NONE
      })
      case 'update': return fnv(name, (args) => {
        this.applyKw(kwOfDict(h, args[0]), h)
        return py.NONE
      })
      case 'remove': return fnv(name, () => {
        if (this.ax) {
          const i = this.ax.children.indexOf(this)
          if (i >= 0) this.ax.children.splice(i, 1)
          this.ax.dynamic.delete(this)
          emitFig(h, this.ax.fig, this.ax)
          this.ax = null
        } else if (this.fig) this.fig.removeText(this)
        return py.NONE
      })
      case 'axes': return this.ax ? py.obj(this.ax) : py.NONE
      case 'figure': { const f = this.figure(); return f ? py.obj(f) : py.NONE }
      case 'get_figure': return fnv(name, () => { const f = this.figure(); return f ? py.obj(f) : py.NONE })
      case 'get_axes': return fnv(name, () => (this.ax ? py.obj(this.ax) : py.NONE))
      case 'stale': return py.bool(false)
    }
    return undefined
  }
}

const artistBase = (z: number) => ({ z, visible: true, clip: true, tx: 'data' as Coord, ty: 'data' as Coord })

// ---- Line2D

const newLine = (h: Host, x: number[], y: number[], color: string): LineArtist => ({
  kind: 'line', ...artistBase(2), x, y, color, width: rcn(h, 'lines.linewidth'), line: true, dash: null, marker: null, ms: rcn(h, 'lines.markersize'),
  mfc: 'auto', mec: 'auto', mew: rcn(h, 'lines.markeredgewidth'), steps: null, markevery: null, label: null, alpha: 1,
})

const dtypeArr = (xs: number[], dtype: DType): Value => py.arr(NDArray.create(xs, [xs.length], dtype))

class LineObj extends ArtistObj {
  cls = 'Line2D'
  spec: LineArtist
  raw: Value | null = null
  lsName = '-'
  dtypes: [DType, DType] = ['float64', 'float64']
  /** set_data and the constructor keep a list as given, and get_xdata returns it */
  orig: [Value | null, Value | null] = [null, null]
  constructor(spec: LineArtist) {
    super()
    this.spec = spec
  }
  setCoord(i: 0 | 1, v: Value) {
    const a = toArray(v)
    if (i === 0) this.spec.x = a.values()
    else this.spec.y = a.values()
    this.dtypes[i] = a.dtype
    this.orig[i] = v.k === 'list' ? py.list([...v.items]) : null
  }
  props() { return LINE_PROPS as unknown as Props<ArtistObj> }
  repr() { return `<matplotlib.lines.${this.cls} object at ${addr(this)}>` }
  str() { return `Line2D(${this.spec.label ?? ''})` }
  bounds(): [number[], number[]] { return [this.spec.x, this.spec.y] }
  legendEntry(): LegendEntry {
    const a = this.spec
    return { label: a.label ?? '', kind: 'line', color: a.color, width: a.width, dash: a.dash, line: a.line, marker: a.marker, ms: a.ms, mfc: a.mfc, mec: a.mec, mew: a.mew, fc: 'none', ec: 'none', lw: 0, size: 0, alpha: a.alpha }
  }
}

const STEPS: Record<string, LineArtist['steps']> = { default: null, steps: 'pre', 'steps-pre': 'pre', 'steps-mid': 'mid', 'steps-post': 'post' }
const LINE_PROPS: Props<LineObj> = alias({
  color: {
    g: (o) => o.raw ?? py.str(o.spec.color),
    s: (o, [v], h) => {
      const c = isNone(v) ? rcv(h, 'lines.color') : v
      o.spec.color = toColor(h, c)
      o.raw = c
    },
  },
  linewidth: { g: (o) => py.float(o.spec.width), s: (o, [v], h) => { o.spec.width = isNone(v) ? rcn(h, 'lines.linewidth') : h.num(v) } },
  linestyle: {
    g: (o) => py.str(o.lsName),
    s: (o, a, h) => {
      const st = lineStyle(h, a.length === 2 ? py.tuple(a) : isNone(a[0]) ? rcv(h, 'lines.linestyle') : a[0], 'ls')
      o.lsName = st.name
      o.spec.line = st.line
      o.spec.dash = st.dash
    },
  },
  dashes: { s: (o, [v], h) => { o.spec.dash = h.iterate(v).map((x) => h.num(x)) } },
  marker: { g: (o) => py.str(o.spec.marker ?? 'None'), s: (o, [v], h) => { o.spec.marker = markerOf(h, v) } },
  markersize: { g: (o) => py.float(o.spec.ms), s: (o, [v], h) => { o.spec.ms = isNone(v) ? rcn(h, 'lines.markersize') : h.num(v) } },
  markerfacecolor: { g: (o) => py.str(o.spec.mfc === 'auto' ? o.spec.color : o.spec.mfc), s: (o, [v], h) => { o.spec.mfc = isNone(v) || (v.k === 'str' && v.v === 'auto') ? 'auto' : toColor(h, v, 'markerfacecolor') } },
  markeredgecolor: { g: (o) => py.str(o.spec.mec === 'auto' ? o.spec.color : o.spec.mec), s: (o, [v], h) => { o.spec.mec = isNone(v) || (v.k === 'str' && v.v === 'auto') ? 'auto' : toColor(h, v, 'markeredgecolor') } },
  markeredgewidth: { g: (o) => py.float(o.spec.mew), s: (o, [v], h) => { o.spec.mew = isNone(v) ? rcn(h, 'lines.markeredgewidth') : h.num(v) } },
  fillstyle: {
    s: (o, [v], h) => {
      const f = h.str(v)
      if (f === 'none') o.spec.mfc = 'none'
      else if (f !== 'full') throw h.err('NotImplementedError', `fillstyle='${f}' is not drawn in this sandbox (use 'full' or 'none')`)
    },
  },
  drawstyle: {
    g: (o) => py.str(o.spec.steps ? `steps-${o.spec.steps}` : 'default'),
    s: (o, [v], h) => {
      const k = h.str(v)
      if (!(k in STEPS)) throw h.err('ValueError', `'${k}' is not a valid value for drawstyle; supported values are 'default', 'steps-mid', 'steps-pre', 'steps-post', 'steps'`)
      o.spec.steps = STEPS[k]
    },
  },
  markevery: {
    g: (o) => (o.spec.markevery === null ? py.NONE : py.int(o.spec.markevery)),
    s: (o, [v], h) => {
      if (isNone(v)) o.spec.markevery = null
      else if (v.k === 'int') o.spec.markevery = v.v
      else throw h.err('NotImplementedError', 'markevery: only an integer step is drawn in this sandbox')
    },
  },
  xdata: { g: (o) => o.orig[0] ?? dtypeArr(o.spec.x, o.dtypes[0]), s: (o, [v]) => { o.setCoord(0, v) } },
  ydata: { g: (o) => o.orig[1] ?? dtypeArr(o.spec.y, o.dtypes[1]), s: (o, [v]) => { o.setCoord(1, v) } },
  data: {
    g: (o) => py.tuple([o.orig[0] ?? dtypeArr(o.spec.x, o.dtypes[0]), o.orig[1] ?? dtypeArr(o.spec.y, o.dtypes[1])]),
    s: (o, a, h) => {
      const [xv, yv] = a.length === 2 ? a : h.iterate(a[0])
      o.setCoord(0, xv)
      o.setCoord(1, yv)
    },
  },
  xydata: { g: (o) => floats(o.spec.x.flatMap((x, i) => [x, o.spec.y[i]]), [o.spec.x.length, 2]) },
}, { c: 'color', lw: 'linewidth', ls: 'linestyle', ms: 'markersize', mfc: 'markerfacecolor', mec: 'markeredgecolor', mew: 'markeredgewidth', ds: 'drawstyle' })

// ---- patches

type Shape =
  | { t: 'rect'; x: number; y: number; w: number; h: number; angle: number }
  | { t: 'ellipse'; cls: 'Circle' | 'Ellipse'; x: number; y: number; w: number; h: number; angle: number }
  | { t: 'polygon'; pts: number[]; closed: boolean }
  | { t: 'wedge'; x: number; y: number; r: number; t1: number; t2: number; width: number | null }
  | { t: 'arc'; x: number; y: number; w: number; h: number; angle: number; t1: number; t2: number }
  | { t: 'path'; pts: number[] }
  | { t: 'none' }

const CIRCLE_N = 96
const shapePath = (s: Shape): number[] => {
  switch (s.t) {
    case 'rect': {
      const c = Math.cos((s.angle * Math.PI) / 180)
      const n = Math.sin((s.angle * Math.PI) / 180)
      const at = (u: number, v: number) => [s.x + u * c - v * n, s.y + u * n + v * c]
      return [...at(0, 0), ...at(s.w, 0), ...at(s.w, s.h), ...at(0, s.h)]
    }
    case 'ellipse': {
      const out: number[] = []
      const c = Math.cos((s.angle * Math.PI) / 180)
      const n = Math.sin((s.angle * Math.PI) / 180)
      for (let i = 0; i < CIRCLE_N; i++) {
        const t = (2 * Math.PI * i) / CIRCLE_N
        const u = (s.w / 2) * Math.cos(t)
        const v = (s.h / 2) * Math.sin(t)
        out.push(s.x + u * c - v * n, s.y + u * n + v * c)
      }
      return out
    }
    case 'polygon': case 'path': return s.pts
    case 'wedge': {
      const span = s.t2 - s.t1
      const steps = Math.max(2, Math.ceil(Math.abs(span) / 3))
      const arc = (r: number) => Array.from({ length: steps + 1 }, (_, i) => {
        const a = ((s.t1 + (span * i) / steps) * Math.PI) / 180
        return [s.x + r * Math.cos(a), s.y + r * Math.sin(a)]
      })
      const outer = arc(s.r).flat()
      if (s.width === null) return Math.abs(span) >= 360 ? outer : [s.x, s.y, ...outer]
      return [...outer, ...arc(s.r - s.width).reverse().flat()]
    }
    case 'arc': {
      const span = s.t2 - s.t1
      const steps = Math.max(2, Math.ceil(Math.abs(span) / 3))
      const c = Math.cos((s.angle * Math.PI) / 180)
      const n = Math.sin((s.angle * Math.PI) / 180)
      return Array.from({ length: steps + 1 }, (_, i) => {
        const a = ((s.t1 + (span * i) / steps) * Math.PI) / 180
        const u = (s.w / 2) * Math.cos(a)
        const v = (s.h / 2) * Math.sin(a)
        return [s.x + u * c - v * n, s.y + u * n + v * c]
      }).flat()
    }
    case 'none': return []
  }
}

/** extent of a shape: curves' true extrema rather than their polygon's */
const shapeBounds = (s: Shape): [number[], number[]] => {
  if (s.t === 'ellipse') {
    // matplotlib looks for extrema on the unit circle, before rotating: its eight Bezier end points
    const c = Math.cos((s.angle * Math.PI) / 180)
    const n = Math.sin((s.angle * Math.PI) / 180)
    const r = Math.SQRT1_2
    const pts = [[1, 0], [r, r], [0, 1], [-r, r], [-1, 0], [-r, -r], [0, -1], [r, -r]].map(([u, v]) => [(u * s.w) / 2, (v * s.h) / 2])
    return [pts.map(([u, v]) => s.x + u * c - v * n), pts.map(([u, v]) => s.y + u * n + v * c)]
  }
  const p = shapePath(s)
  return [p.filter((_, i) => i % 2 === 0), p.filter((_, i) => i % 2 === 1)]
}

const newPoly = (fc: string, ec: string, lw: number, z = 1): PolyArtist => ({
  kind: 'poly', ...artistBase(z), paths: [], closed: true, fcs: [fc], ecs: [ec], lw, dash: null, alpha: 1, label: null, seamless: false,
})

class PatchObj extends ArtistObj {
  cls: string
  spec: PolyArtist
  shape: Shape
  fill = true
  face: string
  fcSet = false
  ecSet = false
  /** a named constructor's str(): 'Rectangle(xy=(0, 0), width=1, height=2, angle=0)' */
  text = ''
  constructor(cls: string, shape: Shape, h: Host) {
    super()
    this.cls = cls
    this.shape = shape
    this.face = rcc(h, 'patch.facecolor')
    this.spec = newPoly(this.face, 'none', rcn(h, 'patch.linewidth'))
    this.rebuild()
  }
  rebuild() {
    this.spec.paths = [shapePath(this.shape)]
    this.spec.closed = this.shape.t === 'polygon' ? this.shape.closed : this.shape.t !== 'arc' && this.shape.t !== 'path'
  }
  props() { return PATCH_PROPS as unknown as Props<ArtistObj> }
  isa(n: string) { return n === 'Patch' || n === 'Artist' }
  repr() { return `<matplotlib.patches.${this.cls} object at ${addr(this)}>` }
  str() {
    const s = this.shape
    const g = (v: number) => formatG(v)
    switch (s.t) {
      case 'rect': return `Rectangle(xy=(${g(s.x)}, ${g(s.y)}), width=${g(s.w)}, height=${g(s.h)}, angle=${g(s.angle)})`
      case 'ellipse': return s.cls === 'Circle' ? `Circle(xy=(${g(s.x)}, ${g(s.y)}), radius=${g(s.w / 2)})` : `Ellipse(xy=(${g(s.x)}, ${g(s.y)}), width=${g(s.w)}, height=${g(s.h)}, angle=${g(s.angle)})`
      case 'polygon': return s.pts.length ? `Polygon${s.pts.length / 2 + (s.closed ? 1 : 0)}((${g(s.pts[0])}, ${g(s.pts[1])}) ...)` : 'Polygon0()'
      case 'wedge': return `Wedge(center=(${g(s.x)}, ${g(s.y)}), r=${g(s.r)}, theta1=${g(s.t1)}, theta2=${g(s.t2)}, width=${s.width === null ? 'None' : g(s.width)})`
      default: return `${this.cls}()`
    }
  }
  bounds(): [number[], number[]] { return shapeBounds(this.shape) }
  legendEntry(): LegendEntry {
    const a = this.spec
    if (this.cls === 'StepPatch' && !this.fill) return { label: a.label ?? '', kind: 'line', color: a.ecs[0], width: a.lw, dash: a.dash, line: true, marker: null, ms: 0, mfc: 'none', mec: 'none', mew: 0, fc: 'none', ec: 'none', lw: 0, size: 0, alpha: a.alpha }
    return { label: a.label ?? '', kind: 'patch', color: a.fcs[0], width: 0, dash: a.dash, line: false, marker: null, ms: 0, mfc: 'none', mec: 'none', mew: 0, fc: a.fcs[0], ec: a.ecs[0], lw: a.lw, size: 0, alpha: a.alpha }
  }
  /** the rules of Patch._set_edgecolor: no edge unless asked for or unfilled */
  refreshEdge(h: Host) {
    if (!this.ecSet) this.spec.ecs = [this.fill ? 'none' : rcc(h, 'patch.edgecolor')]
    this.spec.fcs = [this.fill ? this.face : 'none']
  }
}

const geomNum = (read: (s: Shape) => number | undefined, write: (s: Shape, v: number) => void): Prop<PatchObj> => ({
  g: (o) => py.float(read(o.shape) ?? 0),
  s: (o, [v], h) => { write(o.shape, h.num(v)); o.rebuild()  },
})

const PATCH_PROPS: Props<PatchObj> = alias({
  facecolor: {
    g: (o) => (o.fill ? rgbaValue(o.face, o.alphaSet) : rgbaValue(o.face, 0)),
    s: (o, [v], h) => {
      o.face = toColor(h, v, 'facecolor')
      o.fcSet = true
      if (o.fill) o.spec.fcs = [o.face]
    },
  },
  edgecolor: { g: (o) => rgbaValue(o.spec.ecs[0], o.alphaSet), s: (o, [v], h) => { if (isNone(v)) { o.ecSet = false; o.refreshEdge(h) } else { o.spec.ecs = [toColor(h, v, 'edgecolor')]; o.ecSet = true } } },
  color: { s: (o, [v], h) => { const c = toColor(h, v); o.face = c; o.spec.fcs = [c]; o.spec.ecs = [c]; o.ecSet = true; o.fcSet = true } },
  linewidth: { g: (o) => py.float(o.spec.lw), s: (o, [v], h) => { o.spec.lw = optNum(h, v) ?? rcn(h, 'patch.linewidth') } },
  linestyle: { s: (o, [v], h) => { const st = lineStyle(h, v); o.spec.dash = st.dash; if (!st.line) o.spec.lw = 0 } },
  fill: { g: (o) => py.bool(o.fill), s: (o, [v], h) => { o.fill = h.truthy(v); o.refreshEdge(h) } },
  x: geomNum((s) => (s.t === 'rect' ? s.x : undefined), (s, v) => { if (s.t === 'rect') s.x = v }),
  y: geomNum((s) => (s.t === 'rect' ? s.y : undefined), (s, v) => { if (s.t === 'rect') s.y = v }),
  width: geomNum((s) => (s.t === 'rect' || s.t === 'ellipse' ? s.w : s.t === 'wedge' ? s.width ?? 0 : undefined), (s, v) => { if (s.t === 'rect' || s.t === 'ellipse') s.w = v; else if (s.t === 'wedge') s.width = v }),
  height: geomNum((s) => (s.t === 'rect' || s.t === 'ellipse' ? s.h : undefined), (s, v) => { if (s.t === 'rect' || s.t === 'ellipse') s.h = v }),
  angle: geomNum((s) => (s.t === 'rect' || s.t === 'ellipse' ? s.angle : undefined), (s, v) => { if (s.t === 'rect' || s.t === 'ellipse') s.angle = v }),
  radius: geomNum((s) => (s.t === 'ellipse' ? s.w / 2 : s.t === 'wedge' ? s.r : undefined), (s, v) => { if (s.t === 'ellipse') { s.w = 2 * v; s.h = 2 * v } else if (s.t === 'wedge') s.r = v }),
  xy: {
    g: (o) => {
      const s = o.shape
      if (s.t === 'rect') return py.tuple([py.float(s.x), py.float(s.y)])
      if (s.t === 'polygon') return floats(s.closed ? [...s.pts, s.pts[0], s.pts[1]] : s.pts, [(s.pts.length + (s.closed ? 2 : 0)) / 2, 2])
      return py.NONE
    },
    s: (o, [v], h) => {
      const s = o.shape
      const p = nums(h, v, 'xy')
      if (s.t === 'rect') { s.x = p[0]; s.y = p[1] } else if (s.t === 'polygon') s.pts = p
      o.rebuild()
    },
  },
  center: {
    g: (o) => {
      const s = o.shape
      if (s.t === 'ellipse' || s.t === 'wedge') return py.tuple([py.float(s.x), py.float(s.y)])
      if (s.t === 'rect') return py.tuple([py.float(s.x + s.w / 2), py.float(s.y + s.h / 2)])
      return py.NONE
    },
    s: (o, [v], h) => {
      const s = o.shape
      const p = nums(h, v, 'center')
      if (s.t === 'ellipse' || s.t === 'wedge') { s.x = p[0]; s.y = p[1] }
      o.rebuild()
    },
  },
  bounds: {
    s: (o, a, h) => {
      const s = o.shape
      const p = a.length === 4 ? a.map((v) => h.num(v)) : nums(h, a[0], 'bounds')
      if (s.t === 'rect') [s.x, s.y, s.w, s.h] = p
      o.rebuild()
    },
  },
}, { fc: 'facecolor', ec: 'edgecolor', lw: 'linewidth', ls: 'linestyle', linewidths: 'linewidth', facecolors: 'facecolor', edgecolors: 'edgecolor', colors: 'color' })

// ---- a colour-mapped artist's shared state

interface Mapped {
  lut: Lut
  norm: NormObj
  values: number[] | null
  shape: number[]
  recolor(): void
}

const MAPPED_PROPS: Props<ArtistObj & Mapped> = {
  cmap: { g: (o) => py.obj(new ColormapObj(o.lut)), s: (o, [v], h) => { o.lut = cmapArg(h, v); o.recolor() } },
  norm: { g: (o) => py.obj(o.norm), s: (o, [v], h) => { if (v.k === 'obj' && v.o instanceof NormObj) { o.norm = v.o; o.norm.autoscaleNone(o.values ?? []); o.recolor() } else throw h.err('ValueError', 'norm must be a Normalize instance') } },
  clim: {
    g: (o) => py.tuple([o.norm.vmin === null ? py.NONE : py.float(o.norm.vmin), o.norm.vmax === null ? py.NONE : py.float(o.norm.vmax)]),
    s: (o, a, h, kw) => {
      let lo = a[0] ?? kw.vmin
      let hi = a[1] ?? kw.vmax
      if (a.length === 1 && !isNum(a[0]) && !isNone(a[0])) [lo, hi] = h.iterate(a[0])
      if (lo && !isNone(lo)) o.norm.vmin = h.num(lo)
      if (hi && !isNone(hi)) o.norm.vmax = h.num(hi)
      o.recolor()
    },
  },
  array: {
    g: (o) => (o.values ? floats(o.values, o.shape) : py.NONE),
    s: (o, [v]) => {
      if (isNone(v)) { o.values = null; return }
      const a = toArray(v, 'float64')
      o.values = a.values()
      o.shape = a.shape
      o.recolor()
    },
  },
}
const mappedProps = <T>(p: Props<T>): Props<T> => ({ ...(MAPPED_PROPS as unknown as Props<T>), ...p })

const cmapArg = (h: Host, v: Value | undefined, what = 'cmap'): Lut => {
  if (isNone(v)) return getLut(rcs(h, 'image.cmap')) ?? getLut('viridis')!
  if (v!.k === 'obj' && v!.o instanceof ColormapObj) return v!.o.lut
  if (v!.k === 'str') {
    const l = getLut(v!.v)
    if (l) return l
    throw h.err('ValueError', `'${v!.v}' is not a valid value for ${what}; supported values are ${ALL_CMAPS.map((c) => `'${c}'`).join(', ')}`)
  }
  throw h.err('TypeError', `${what} must be a str or Colormap, not ${v!.k}`)
}

const normArg = (h: Host, kw: Kw): NormObj => {
  const nv = pop(kw, 'norm')
  const vmin = optNum(h, pop(kw, 'vmin'))
  const vmax = optNum(h, pop(kw, 'vmax'))
  if (nv && !isNone(nv)) {
    if (nv.k === 'str') {
      if (nv.v === 'linear') return new NormObj('Normalize', vmin, vmax)
      if (nv.v === 'log') return new NormObj('LogNorm', vmin, vmax)
      throw h.err('NotImplementedError', `norm='${nv.v}' is not available in this sandbox`)
    }
    if (nv.k !== 'obj' || !(nv.o instanceof NormObj)) throw h.err('ValueError', 'norm must be a Normalize instance or a scale name')
    if (vmin !== null || vmax !== null) throw h.err('ValueError', 'Passing a Normalize instance simultaneously with vmin/vmax is not supported.  Please pass vmin/vmax directly to the norm when creating it.')
    return nv.o
  }
  return new NormObj('Normalize', vmin, vmax)
}

// ---- scatter

class ScatterObj extends ArtistObj implements Mapped {
  cls = 'PathCollection'
  spec: ScatterArtist
  lut: Lut
  norm: NormObj
  values: number[] | null = null
  shape: number[] = []
  sizeType: DType = 'float64'
  lwType: DType = 'float64'
  hollow = false
  constructor(spec: ScatterArtist, lut: Lut, norm: NormObj) {
    super()
    this.spec = spec
    this.lut = lut
    this.norm = norm
  }
  recolor() {
    if (this.values) this.spec.colors = this.values.map((v) => this.norm.color(this.lut, v))
  }
  props() { return SCATTER_PROPS as unknown as Props<ArtistObj> }
  repr() { return `<matplotlib.collections.PathCollection object at ${addr(this)}>` }
  bounds(): [number[], number[]] { return [this.spec.x, this.spec.y] }
  isa(n: string) { return n === 'Collection' }
  legendEntry(): LegendEntry {
    const a = this.spec
    return { label: a.label ?? '', kind: 'scatter', color: a.colors[0] ?? '#000000', width: 0, dash: null, line: false, marker: a.marker, ms: 0, mfc: a.colors[0] ?? 'none', mec: a.edges?.[0] ?? a.colors[0] ?? 'none', mew: a.lw[0] ?? 1, fc: a.colors[0] ?? 'none', ec: a.edges?.[0] ?? 'none', lw: a.lw[0] ?? 1, size: a.sizes[0] ?? 36, alpha: a.alpha }
  }
}
/** an N×4 array with the artist's alpha; 'none' leaves it empty, as to_rgba_array does */
const rgbaArr = (cs: string[], alpha: number | null = null): Value => {
  const list = cs.filter((c) => c !== 'none')
  return floats(list.flatMap((c) => { const [r, g, b, a] = rgbaOf(c); return [r, g, b, alpha ?? a] }), [list.length, 4])
}
/** before a drawing, a colour-mapped collection still holds the default face: matplotlib maps colours at draw time */
const faceNow = (o: ArtistObj & Mapped, cs: string[], h: Host): Value => (o.values && !o.figure()?.drew ? rgbaArr([rcc(h, 'patch.facecolor')], o.alphaSet) : rgbaArr(cs, o.alphaSet))
const SCATTER_PROPS: Props<ScatterObj> = alias(mappedProps<ScatterObj>({
  offsets: {
    g: (o) => floats(o.spec.x.flatMap((x, i) => [x, o.spec.y[i]]), [o.spec.x.length, 2]),
    s: (o, [v]) => {
      const a = toArray(v, 'float64')
      const n = a.size / 2
      const xs = a.values()
      o.spec.x = Array.from({ length: n }, (_, i) => xs[i * 2])
      o.spec.y = Array.from({ length: n }, (_, i) => xs[i * 2 + 1])
    },
  },
  sizes: { g: (o) => dtypeArr(o.spec.sizes, o.sizeType), s: (o, [v], h) => { o.spec.sizes = nums(h, v, 'sizes'); o.sizeType = toArray(v).dtype } },
  facecolor: { g: (o, h) => (o.hollow ? rgbaArr(['none']) : faceNow(o, o.spec.colors, h)), s: (o, [v], h) => { o.values = null; o.spec.colors = toColors(h, v, 'facecolor') } },
  edgecolor: {
    g: (o, h) => (o.spec.edges ? rgbaArr(o.spec.edges, o.alphaSet) : faceNow(o, o.spec.colors, h)),
    s: (o, [v], h) => {
      if (v.k === 'str' && v.v === 'face') o.spec.edges = null
      else if (isNone(v)) o.spec.edges = null
      else o.spec.edges = toColors(h, v, 'edgecolor')
    },
  },
  color: { s: (o, [v], h) => { const c = toColors(h, v); o.values = null; o.spec.colors = c; o.spec.edges = c } },
  linewidth: { g: (o) => dtypeArr(o.spec.lw, o.lwType), s: (o, [v], h) => { o.spec.lw = nums(h, v, 'linewidth'); o.lwType = toArray(v).dtype } },
  linestyle: { s: () => {} },
  paths: {},
}), { fc: 'facecolor', facecolors: 'facecolor', ec: 'edgecolor', edgecolors: 'edgecolor', lw: 'linewidth', linewidths: 'linewidth', ls: 'linestyle', linestyles: 'linestyle' })

// ---- polygons, line collections, meshes

class PolyObj extends ArtistObj implements Mapped {
  cls: string
  spec: PolyArtist
  lut: Lut
  norm: NormObj
  values: number[] | null = null
  shape: number[] = []
  module = 'collections'
  constructor(cls: string, spec: PolyArtist, lut?: Lut, norm?: NormObj) {
    super()
    this.cls = cls
    this.spec = spec
    this.lut = lut ?? getLut('viridis')!
    this.norm = norm ?? new NormObj('Normalize', null, null)
  }
  recolor() {
    if (this.values) this.spec.fcs = this.values.map((v) => this.norm.color(this.lut, v))
  }
  props() { return POLY_PROPS as unknown as Props<ArtistObj> }
  repr() { return `<matplotlib.${this.module}.${this.cls} object at ${addr(this)}>` }
  isa(n: string) { return n === 'Collection' }
  legendEntry(): LegendEntry {
    const a = this.spec
    return { label: a.label ?? '', kind: 'patch', color: a.fcs[0], width: 0, dash: a.dash, line: false, marker: null, ms: 0, mfc: 'none', mec: 'none', mew: 0, fc: a.fcs[0], ec: a.ecs[0], lw: a.lw, size: 0, alpha: a.alpha }
  }
}
const POLY_PROPS: Props<PolyObj> = alias(mappedProps<PolyObj>({
  facecolor: { g: (o, h) => faceNow(o, o.spec.fcs, h), s: (o, [v], h) => { o.values = null; o.spec.fcs = toColors(h, v, 'facecolor') } },
  edgecolor: { g: (o) => rgbaArr(o.spec.ecs, o.alphaSet), s: (o, [v], h) => { o.spec.ecs = v.k === 'str' && v.v === 'face' ? o.spec.fcs : isNone(v) ? ['none'] : toColors(h, v, 'edgecolor') } },
  color: { s: (o, [v], h) => { const c = toColors(h, v); o.values = null; o.spec.fcs = c; o.spec.ecs = c } },
  linewidth: { g: (o) => floats([o.spec.lw]), s: (o, [v], h) => { o.spec.lw = nums(h, v, 'linewidth')[0] } },
  linestyle: { s: (o, [v], h) => { o.spec.dash = lineStyle(h, v).dash } },
  verts: { s: (o, [v], h) => { o.spec.paths = h.iterate(v).map((p) => nums(h, p, 'verts')) } },
  paths: { g: (o) => py.list(o.spec.paths.map((p) => floats(p, [p.length / 2, 2]))) },
}), { fc: 'facecolor', facecolors: 'facecolor', ec: 'edgecolor', edgecolors: 'edgecolor', lw: 'linewidth', linewidths: 'linewidth', ls: 'linestyle', linestyles: 'linestyle' })

const newSegs = (colors: string[], width: number): SegsArtist => ({ kind: 'segs', ...artistBase(2), paths: [], colors, widths: [width], dash: null, alpha: 1, label: null })

class SegsObj extends ArtistObj implements Mapped {
  cls: string
  spec: SegsArtist
  module = 'collections'
  lut: Lut
  norm: NormObj
  values: number[] | null = null
  shape: number[] = []
  constructor(cls: string, spec: SegsArtist) {
    super()
    this.cls = cls
    this.spec = spec
    this.lut = getLut('viridis')!
    this.norm = new NormObj('Normalize', null, null)
  }
  recolor() {
    if (this.values) this.spec.colors = this.values.map((v) => this.norm.color(this.lut, v))
  }
  props() { return SEGS_PROPS as unknown as Props<ArtistObj> }
  repr() { return `<matplotlib.${this.module}.${this.cls} object at ${addr(this)}>` }
  isa(n: string) { return n === 'Collection' }
  legendEntry(): LegendEntry {
    const a = this.spec
    return { label: a.label ?? '', kind: 'line', color: a.colors[0] ?? '#000000', width: a.widths[0], dash: a.dash, line: true, marker: null, ms: 0, mfc: 'none', mec: 'none', mew: 0, fc: 'none', ec: 'none', lw: 0, size: 0, alpha: a.alpha }
  }
}
const SEGS_PROPS: Props<SegsObj> = alias(mappedProps<SegsObj>({
  color: { g: (o) => rgbaArr(o.spec.colors), s: (o, [v], h) => { o.values = null; o.spec.colors = toColors(h, v) } },
  linewidth: { g: (o) => floats(o.spec.widths), s: (o, [v], h) => { o.spec.widths = nums(h, v, 'linewidth') } },
  linestyle: { s: (o, [v], h) => { o.spec.dash = lineStyle(h, v).dash } },
  segments: {
    g: (o) => py.list(o.spec.paths.map((p) => floats(p, [p.length / 2, 2]))),
    s: (o, [v], h) => { o.spec.paths = h.iterate(v).map((p) => nums(h, p, 'segments')) },
  },
}), { colors: 'color', edgecolor: 'color', edgecolors: 'color', ec: 'color', lw: 'linewidth', linewidths: 'linewidth', ls: 'linestyle', linestyles: 'linestyle' })

class MeshObj extends ArtistObj implements Mapped {
  cls = 'QuadMesh'
  spec: MeshArtist
  lut: Lut
  norm: NormObj
  values: number[] | null
  shape: number[]
  constructor(spec: MeshArtist, values: number[], shape: number[], lut: Lut, norm: NormObj) {
    super()
    this.spec = spec
    this.values = values
    this.shape = shape
    this.lut = lut
    this.norm = norm
  }
  recolor() {
    if (this.values) this.spec.colors = this.values.map((v) => this.norm.color(this.lut, v))
  }
  props() { return MESH_PROPS as unknown as Props<ArtistObj> }
  repr() { return `<matplotlib.collections.QuadMesh object at ${addr(this)}>` }
  isa(n: string) { return n === 'Collection' }
}
const MESH_PROPS: Props<MeshObj> = alias(mappedProps<MeshObj>({
  edgecolor: { s: (o, [v], h) => { o.spec.edge = isNone(v) || (v.k === 'str' && v.v === 'none') ? null : toColor(h, v, 'edgecolor') } },
  linewidth: { s: (o, [v], h) => { o.spec.lw = h.num(v) } },
  facecolor: { s: () => {} },
  linestyle: { s: () => {} },
}), { ec: 'edgecolor', edgecolors: 'edgecolor', lw: 'linewidth', linewidths: 'linewidth', ls: 'linestyle', linestyles: 'linestyle', fc: 'facecolor', facecolors: 'facecolor' })

// ---- images

class ImageObj extends ArtistObj implements Mapped {
  cls = 'AxesImage'
  spec: ImageArtist
  lut: Lut
  norm: NormObj
  values: number[] | null
  shape: number[]
  source: NDArray | null = null
  extentRaw: Value | null = null
  origin = 'upper'
  constructor(spec: ImageArtist, lut: Lut, norm: NormObj, values: number[] | null, shape: number[]) {
    super()
    this.spec = spec
    this.lut = lut
    this.norm = norm
    this.values = values
    this.shape = shape
  }
  /** pixels from the current data, colormap and norm */
  recolor() {
    const s = this.spec
    if (s.rgb) return
    s.cmap = this.lut.name
    s.vmin = this.norm.vmin ?? 0
    s.vmax = this.norm.vmax ?? 1
    s.values = this.values ?? []
    // origin 'lower' keeps row 0 at the bottom of the picture
    const px = s.values.map((v) => this.norm.color(this.lut, v))
    s.pixels = this.origin === 'lower' ? flipRows(px, s.rows, s.cols) : px
    if (this.origin === 'lower') s.values = flipRows(s.values, s.rows, s.cols)
  }
  props() { return IMAGE_PROPS as unknown as Props<ArtistObj> }
  repr() { return `<matplotlib.image.AxesImage object at ${addr(this)}>` }
  str() { return `AxesImage(shape=(${this.spec.rows}, ${this.spec.cols}))` }
  bounds(): [number[], number[]] { const e = this.spec.extent; return [[e[0], e[1]], [e[2], e[3]]] }
}
const flipRows = <T>(xs: T[], rows: number, cols: number): T[] => {
  const out: T[] = []
  for (let r = rows - 1; r >= 0; r--) for (let c = 0; c < cols; c++) out.push(xs[r * cols + c])
  return out
}
const IMAGE_PROPS: Props<ImageObj> = mappedProps<ImageObj>({
  data: {
    s: (o, [v], h) => {
      const a = toArray(v)
      const s = o.spec
      if (a.ndim === 2 && !s.rgb) {
        if (a.shape[0] !== s.rows || a.shape[1] !== s.cols) {
          s.rows = a.shape[0]
          s.cols = a.shape[1]
        }
        o.values = a.values()
        o.shape = a.shape
        o.source = a.copy()
        o.recolor()
      } else {
        const pix = imagePixels(h, a)
        s.rows = a.shape[0]
        s.cols = a.shape[1]
        s.pixels = pix.pixels
        s.values = pix.values
        s.rgb = pix.rgb
        o.source = pix.a.copy()
        if (!pix.rgb) { o.values = pix.values; o.recolor() }
      }
    },
  },
  array: { g: (o) => (o.source ? py.arr(o.source.copy()) : py.NONE), s: (o, [v], h) => { IMAGE_PROPS.data.s!(o, [v], h, {}) } },
  extent: {
    g: (o) => o.extentRaw ?? py.list(o.spec.extent.map(py.float)),
    s: (o, [v], h) => {
      const e = nums(h, v, 'extent')
      if (e.length !== 4) throw h.err('ValueError', 'extent must be a 4-tuple')
      o.extentRaw = py.list(h.iterate(v))
      o.spec.extent = e as [number, number, number, number]
      o.ax?.imageExtent(o, e)
    },
  },
  interpolation: { s: (o, [v], h) => { const s = optStr(h, v); o.spec.smooth = !!s && !['nearest', 'none', 'antialiased', 'auto', 'data'].includes(s) } },
  resample: { s: () => {} },
  filternorm: { s: () => {} },
  filterrad: { s: () => {} },
  interpolation_stage: { s: () => {} },
  url: { s: () => {} },
  size: { g: (o) => py.tuple([py.int(o.spec.rows), py.int(o.spec.cols)]) },
  shape: { g: (o) => py.tuple([py.int(o.spec.rows), py.int(o.spec.cols)]) },
})

// ---- text

const newText = (h: Host, x: number, y: number, s: string): TextArtist => ({
  kind: 'text', ...artistBase(3), clip: false, x, y, text: s, color: rcc(h, 'text.color'), size: rcn(h, 'font.size'), ha: 'left', va: 'baseline', rotation: 0, weight: 'normal', italic: false, alpha: 1, dx: 0, dy: 0, bbox: null, arrow: null,
})

const HA: Record<string, TextArtist['ha']> = { left: 'left', center: 'center', right: 'right' }
const VA: Record<string, TextArtist['va']> = { top: 'top', center: 'center', bottom: 'bottom', baseline: 'baseline', center_baseline: 'center' }

class TextObj extends ArtistObj {
  cls = 'Text'
  spec: TextArtist
  /** how x and y print in repr: as they were given */
  xs: string
  ys: string
  /** drawn by its owner (a title, an axis label), not as a child of the Axes */
  owned = false
  xy: [number, number] | null = null
  xyValue: Value | null = null
  colorRaw: Value | null = null
  constructor(spec: TextArtist, xs?: string, ys?: string) {
    super()
    this.spec = spec
    this.xs = xs ?? String(spec.x)
    this.ys = ys ?? String(spec.y)
  }
  props() { return TEXT_PROPS as unknown as Props<ArtistObj> }
  repr() { return `Text(${this.xs}, ${this.ys}, ${pyRepr(py.str(this.spec.text))})` }
  str() { return this.cls === 'Annotation' ? `Annotation(${formatG(this.xy![0])}, ${formatG(this.xy![1])}, ${pyRepr(py.str(this.spec.text))})` : this.repr() }
  getAttr(name: string, h: Host): Value | undefined {
    if (name === 'xy' && this.xyValue) return this.xyValue
    return super.getAttr(name, h)
  }
  labelOf() { return this.textLabel }
  setLabel(l: string | null) { this.textLabel = l ?? '' }
  textLabel = ''
}

/** a number as it was written ('3' stays an int) */
const numText = (t: string): Value => (/^-?\d+$/.test(t) ? py.int(Number(t)) : py.float(Number(t)))

const TEXT_PROPS: Props<TextObj> = alias({
  text: { g: (o) => py.str(o.spec.text), s: (o, [v], h) => { o.spec.text = isNone(v) ? '' : h.str(v) } },
  color: { g: (o) => o.colorRaw ?? py.str(o.spec.color), s: (o, [v], h) => { o.spec.color = toColor(h, v); o.colorRaw = v } },
  fontsize: { g: (o) => py.float(o.spec.size), s: (o, [v], h) => { o.spec.size = fontPt(h, v) } },
  fontweight: {
    g: (o) => py.str(o.spec.weight),
    s: (o, [v], h) => { o.spec.weight = v.k === 'str' ? v.v : h.num(v) >= 600 ? 'bold' : 'normal' },
  },
  fontstyle: { g: (o) => py.str(o.spec.italic ? 'italic' : 'normal'), s: (o, [v], h) => { o.spec.italic = h.str(v) !== 'normal' } },
  horizontalalignment: {
    g: (o) => py.str(o.spec.ha),
    s: (o, [v], h) => {
      const a = h.str(v)
      if (!HA[a]) throw h.err('ValueError', `'${a}' is not a valid value for horizontalalignment; supported values are 'left', 'center', 'right'`)
      o.spec.ha = HA[a]
    },
  },
  verticalalignment: {
    g: (o) => py.str(o.spec.va),
    s: (o, [v], h) => {
      const a = h.str(v)
      if (!VA[a]) throw h.err('ValueError', `'${a}' is not a valid value for verticalalignment; supported values are 'baseline', 'bottom', 'center', 'center_baseline', 'top'`)
      o.spec.va = VA[a]
    },
  },
  rotation: {
    g: (o) => py.float(o.spec.rotation),
    s: (o, [v], h) => {
      if (v.k === 'str') {
        if (v.v === 'horizontal') o.spec.rotation = 0
        else if (v.v === 'vertical') o.spec.rotation = 90
        else throw h.err('ValueError', `rotation must be 'vertical', 'horizontal' or a number, not ${v.v}`)
      } else o.spec.rotation = isNone(v) ? 0 : h.num(v)
    },
  },
  position: {
    g: (o) => py.tuple([numText(o.xs), numText(o.ys)]),
    s: (o, [v], h) => {
      const [xv, yv] = h.iterate(v)
      o.spec.x = h.num(xv)
      o.spec.y = h.num(yv)
      o.xs = h.str(xv)
      o.ys = h.str(yv)
    },
  },
  x: { s: (o, [v], h) => { o.spec.x = h.num(v); o.xs = h.str(v) } },
  y: { s: (o, [v], h) => { o.spec.y = h.num(v); o.ys = h.str(v) } },
  bbox: {
    s: (o, [v], h) => {
      if (isNone(v)) { o.spec.bbox = null; return }
      const kw = kwOfDict(h, v)
      const style = optStr(h, pop(kw, 'boxstyle')) ?? 'square'
      const fc = pop(kw, 'facecolor', 'fc', 'color')
      const ec = pop(kw, 'edgecolor', 'ec')
      const alpha = optNum(h, pop(kw, 'alpha'))
      const lw = optNum(h, pop(kw, 'linewidth', 'lw')) ?? 1
      const pad = optNum(h, pop(kw, 'pad'))
      const m = /pad=([\d.]+)/.exec(style)
      o.spec.bbox = {
        fc: fc ? toColor(h, fc) : rcc(h, 'patch.facecolor'), ec: ec ? toColor(h, ec) : '#000000', lw, alpha: alpha ?? 1,
        pad: m ? Number(m[1]) * o.spec.size : pad ?? 0.3 * o.spec.size, round: /round/.test(style),
      }
    },
  },
  backgroundcolor: { s: (o, [v], h) => { o.spec.bbox = { fc: toColor(h, v), ec: toColor(h, v), lw: 1, alpha: 1, pad: 0.3 * o.spec.size, round: false } } },
  fontdict: { s: (o, [v], h) => o.applyKw(kwOfDict(h, v), h) },
}, { c: 'color', size: 'fontsize', weight: 'fontweight', style: 'fontstyle', ha: 'horizontalalignment', va: 'verticalalignment' })

// ---- containers

class BarRect extends PyObj {
  readonly cls = 'Rectangle'
  bar: BarObj
  k: number
  constructor(bar: BarObj, k: number) {
    super()
    this.bar = bar
    this.k = k
  }
  isa(n: string) { return n === 'Patch' }
  repr() { return `<matplotlib.patches.Rectangle object at ${addr(this)}>` }
  str() {
    const [x, y, w, hh] = this.box()
    return `Rectangle(xy=(${formatG(x)}, ${formatG(y)}), width=${formatG(w)}, height=${formatG(hh)}, angle=0)`
  }
  box(): [number, number, number, number] {
    const a = this.bar.spec
    const k = this.k
    return a.horizontal ? [a.bottoms[k], a.x[k], a.heights[k], a.widths[k]] : [a.x[k], a.bottoms[k], a.widths[k], a.heights[k]]
  }
  getAttr(name: string, h: Host): Value | undefined {
    const a = this.bar.spec
    const k = this.k
    const setList = (field: 'colors' | 'edges', c: string) => {
      const list = Array.from({ length: a.x.length }, (_, i) => a[field][i % a[field].length])
      list[k] = c
      a[field] = list
    }
    const [x, y, w, hh] = this.box()
    const setBox = (nx: number, ny: number, nw: number, nh: number) => {
      const up = (f: 'x' | 'bottoms' | 'widths' | 'heights', v: number) => { const c = [...a[f]]; c[k] = v; a[f] = c }
      if (a.horizontal) { up('bottoms', nx); up('x', ny); up('heights', nw); up('widths', nh) } else { up('x', nx); up('bottoms', ny); up('widths', nw); up('heights', nh) }
      this.bar.touch(h)
    }
    const num1 = (f: (v: number) => void): Value => fnv(name, (args) => { f(h.num(args[0])); return py.NONE })
    switch (name) {
      case 'get_x': return fnv(name, () => py.float(x))
      case 'get_y': return fnv(name, () => py.float(y))
      case 'get_width': return fnv(name, () => py.float(w))
      case 'get_height': return fnv(name, () => py.float(hh))
      case 'get_xy': return fnv(name, () => py.tuple([py.float(x), py.float(y)]))
      case 'get_bbox': return fnv(name, () => py.obj(new Bbox(x, y, x + w, y + hh)))
      case 'set_x': return num1((v) => setBox(v, y, w, hh))
      case 'set_y': return num1((v) => setBox(x, v, w, hh))
      case 'set_width': return num1((v) => setBox(x, y, v, hh))
      case 'set_height': return num1((v) => setBox(x, y, w, v))
      case 'set_xy': return fnv(name, (args) => { const [nx, ny] = nums(h, args[0], 'xy'); setBox(nx, ny, w, hh); return py.NONE })
      case 'get_facecolor': case 'get_fc': return fnv(name, () => rgbaValue(a.colors[k % a.colors.length]))
      case 'get_edgecolor': case 'get_ec': return fnv(name, () => rgbaValue(a.edges[k % a.edges.length]))
      case 'set_facecolor': case 'set_fc': return fnv(name, (args) => { setList('colors', toColor(h, args[0], 'facecolor')); this.bar.touch(h); return py.NONE })
      case 'set_edgecolor': case 'set_ec': return fnv(name, (args) => { setList('edges', toColor(h, args[0], 'edgecolor')); this.bar.touch(h); return py.NONE })
      case 'set_color': return fnv(name, (args) => {
        const c = toColor(h, args[0])
        setList('colors', c)
        setList('edges', c)
        this.bar.touch(h)
        return py.NONE
      })
      case 'set_alpha': return fnv(name, (args) => {
        const al = optNum(h, args[0]) ?? 1
        const c = rgbaOf(a.colors[k % a.colors.length])
        setList('colors', rgbHex(c[0], c[1], c[2], al))
        this.bar.touch(h)
        return py.NONE
      })
      case 'get_alpha': return fnv(name, () => { const al = rgbaOf(a.colors[k % a.colors.length])[3]; return al < 1 ? py.float(al) : py.NONE })
      case 'set_linewidth': case 'set_lw': return num1((v) => { a.lw = v; this.bar.touch(h) })
      case 'get_linewidth': return fnv(name, () => py.float(a.lw))
      case 'set_hatch': case 'set_capstyle': case 'set_joinstyle': return fnv(name, () => py.NONE)
      case 'set_label': return fnv(name, (args) => { if (k === 0) a.label = h.str(args[0]); return py.NONE })
      case 'get_label': return fnv(name, () => py.str(a.label ?? ''))
      case 'set_visible': return fnv(name, (args) => {
        if (!h.truthy(args[0])) { const c = [...a.heights]; c[k] = 0; a.heights = c }
        this.bar.touch(h)
        return py.NONE
      })
      case 'get_visible': return fnv(name, () => py.bool(a.visible))
      case 'axes': return this.bar.ax ? py.obj(this.bar.ax) : py.NONE
      case 'set': return fnv(name, (_a, kw) => {
        for (const [kk, v] of Object.entries(kw)) {
          const f = this.getAttr(`set_${kk}`, h)
          if (!f || f.k !== 'fn') throw h.err('AttributeError', `Rectangle.set() got an unexpected keyword argument '${kk}'`)
          f.call([v], {})
        }
        return py.NONE
      })
    }
    return undefined
  }
}

class BarObj extends ArtistObj {
  cls = 'BarContainer'
  spec: BarArtist
  fromHist = false
  rects: BarRect[]
  errorbar: Value = py.NONE
  datavalues: number[]
  constructor(spec: BarArtist) {
    super()
    this.spec = spec
    this.datavalues = spec.heights
    this.rects = spec.x.map((_, i) => new BarRect(this, i))
  }
  props() { return BAR_PROPS as unknown as Props<ArtistObj> }
  repr() { return `<BarContainer object of ${this.spec.x.length} artists>` }
  len() { return this.spec.x.length }
  iter(): Value[] { return this.rects.map((r) => py.obj(r)) }
  getItem(idx: Value, h: Host): Value {
    const n = this.rects.length
    const i = h.toInt(idx, 'an index')
    if (i < -n || i >= n) throw h.err('IndexError', 'list index out of range')
    return py.obj(this.rects[i < 0 ? i + n : i])
  }
  getAttr(name: string, h: Host): Value | undefined {
    switch (name) {
      case 'patches': return py.list(this.iter())
      case 'errorbar': return this.errorbar
      case 'datavalues': return floats(this.datavalues)
      case 'orientation': return py.str(this.spec.horizontal ? 'horizontal' : 'vertical')
    }
    return super.getAttr(name, h)
  }
  legendEntry(): LegendEntry {
    const a = this.spec
    return { label: a.label ?? '', kind: 'patch', color: a.colors[0], width: 0, dash: a.dash, line: false, marker: null, ms: 0, mfc: 'none', mec: 'none', mew: 0, fc: a.colors[0], ec: a.edges[0], lw: a.lw, size: 0, alpha: a.alpha }
  }
  bounds(): [number[], number[]] {
    const a = this.spec
    const along = a.x.flatMap((x, i) => [x, x + a.widths[i]])
    const across = a.bottoms.flatMap((b, i) => [b, b + a.heights[i]])
    return a.horizontal ? [across, along] : [along, across]
  }
}
const BAR_PROPS: Props<BarObj> = {
  color: { s: (o, [v], h) => { const c = toColors(h, v); o.spec.colors = c; o.spec.edges = c } },
  facecolor: { s: (o, [v], h) => { o.spec.colors = toColors(h, v, 'facecolor') } },
  edgecolor: { s: (o, [v], h) => { o.spec.edges = isNone(v) ? ['none'] : toColors(h, v, 'edgecolor') } },
  linewidth: { s: (o, [v], h) => { o.spec.lw = h.num(v) } },
  linestyle: { s: (o, [v], h) => { o.spec.dash = lineStyle(h, v).dash } },
}

class ContainerObj extends PyObj {
  readonly cls: string
  items: Value[]
  attrs: Record<string, Value>
  label: string | null
  ax: Axes | null = null
  entry: LegendEntry | null = null
  constructor(cls: string, items: Value[], attrs: Record<string, Value> = {}, label: string | null = null) {
    super()
    this.cls = cls
    this.items = items
    this.attrs = attrs
    this.label = label
  }
  repr() { return `<${this.cls} object of ${this.items.length} artists>` }
  len() { return this.items.length }
  iter(): Value[] { return this.items }
  getItem(idx: Value, h: Host): Value {
    const i = h.toInt(idx, 'an index')
    const n = this.items.length
    if (i < -n || i >= n) throw h.err('IndexError', 'tuple index out of range')
    return this.items[i < 0 ? i + n : i]
  }
  getAttr(name: string, h: Host): Value | undefined {
    if (name in this.attrs) return this.attrs[name]
    if (name === 'get_label') return fnv(name, () => (this.label === null ? py.NONE : py.str(this.label)))
    if (name === 'set_label') return fnv(name, (args) => { this.label = h.str(args[0]); return py.NONE })
    if (name === 'get_children') return fnv(name, () => py.list(this.items.flatMap((v) => (v.k === 'tuple' ? v.items : [v]))))
    if (name === 'remove') return fnv(name, () => {
      for (const v of this.items) {
        for (const it of v.k === 'tuple' ? v.items : [v]) {
          if (it.k === 'obj' && it.o instanceof ArtistObj && it.o.ax) {
            const i = it.o.ax.children.indexOf(it.o)
            if (i >= 0) it.o.ax.children.splice(i, 1)
          }
        }
      }
      if (this.ax) this.ax.containers = this.ax.containers.filter((c) => c !== this)
      return py.NONE
    })
    return undefined
  }
}

/** matplotlib's silent_list: a list that prints as a summary */
class SilentList extends PyObj {
  readonly cls = 'list'
  items: Value[]
  what: string
  constructor(what: string, items: Value[]) {
    super()
    this.what = what
    this.items = items
  }
  repr() { return `<a list of ${this.items.length} ${this.what} objects>` }
  len() { return this.items.length }
  iter(): Value[] { return this.items }
  getItem(idx: Value, h: Host): Value {
    const i = h.toInt(idx, 'an index')
    return this.items[i < 0 ? i + this.items.length : i]
  }
}

class Bbox extends PyObj {
  readonly cls = 'Bbox'
  x0: number
  y0: number
  x1: number
  y1: number
  constructor(x0: number, y0: number, x1: number, y1: number) {
    super()
    this.x0 = x0
    this.y0 = y0
    this.x1 = x1
    this.y1 = y1
  }
  repr() { return `Bbox([[${pyRepr(py.float(this.x0))}, ${pyRepr(py.float(this.y0))}], [${pyRepr(py.float(this.x1))}, ${pyRepr(py.float(this.y1))}]])` }
  str() { return `Bbox(x0=${pyRepr(py.float(this.x0))}, y0=${pyRepr(py.float(this.y0))}, x1=${pyRepr(py.float(this.x1))}, y1=${pyRepr(py.float(this.y1))})` }
  getAttr(name: string): Value | undefined {
    const { x0, y0, x1, y1 } = this
    switch (name) {
      case 'x0': return py.float(x0)
      case 'y0': return py.float(y0)
      case 'x1': return py.float(x1)
      case 'y1': return py.float(y1)
      case 'xmin': return py.float(Math.min(x0, x1))
      case 'ymin': return py.float(Math.min(y0, y1))
      case 'xmax': return py.float(Math.max(x0, x1))
      case 'ymax': return py.float(Math.max(y0, y1))
      case 'width': return py.float(x1 - x0)
      case 'height': return py.float(y1 - y0)
      case 'bounds': return py.tuple([x0, y0, x1 - x0, y1 - y0].map(py.float))
      case 'extents': return floats([x0, y0, x1, y1])
      case 'p0': return floats([x0, y0])
      case 'p1': return floats([x1, y1])
      case 'size': return floats([x1 - x0, y1 - y0])
      case 'intervalx': return floats([x0, x1])
      case 'intervaly': return floats([y0, y1])
    }
    return undefined
  }
}

// ---------------------------------------------------------------- tickers

type Locator =
  | { k: 'auto' }
  | { k: 'maxn'; m: MaxN }
  | { k: 'fixed'; at: number[]; nbins?: number; dtype?: DType }
  | { k: 'null' }
  | { k: 'multiple'; base: number; offset: number }
  | { k: 'log'; base: number; subs: number[] | 'auto' | null }
  | { k: 'linear'; n: number }
  | { k: 'cats' }
  | { k: 'minor'; ndivs: number | null }

type Formatter =
  | { k: 'scalar'; useOffset: boolean; limits: [number, number] | null; sci: boolean }
  | { k: 'log'; base: number }
  | { k: 'fixed'; labels: string[] }
  | { k: 'dict'; map: Map<number, string> }
  | { k: 'func'; fn: Value }
  | { k: 'fmt'; fmt: string }
  | { k: 'str'; fmt: string }
  | { k: 'percent'; xmax: number; decimals: number | null; symbol: string }
  | { k: 'null' }
  | { k: 'cats' }

class TickerObj extends PyObj {
  readonly cls: string
  loc: Locator | null
  fmt: Formatter | null
  constructor(cls: string, loc: Locator | null, fmt: Formatter | null) {
    super()
    this.cls = cls
    this.loc = loc
    this.fmt = fmt
  }
  repr() { return `<matplotlib.ticker.${this.cls} object at ${addr(this)}>` }
  getAttr(name: string, h: Host): Value | undefined {
    const loc = this.loc
    if (name === 'tick_values' && loc) return fnv(name, (args) => floats(locatorTicks(loc, h.num(args[0]), h.num(args[1]), 9, Infinity)))
    if (name === 'set_powerlimits' && this.fmt?.k === 'scalar') {
      const f = this.fmt
      return fnv(name, (args) => { const [a, b] = nums(h, args[0], 'lims'); f.limits = [a, b]; return py.NONE })
    }
    if (name === 'set_scientific' && this.fmt?.k === 'scalar') {
      const f = this.fmt
      return fnv(name, (args) => { f.sci = h.truthy(args[0]); return py.NONE })
    }
    if (name === 'set_useOffset' && this.fmt?.k === 'scalar') {
      const f = this.fmt
      return fnv(name, (args) => { f.useOffset = h.truthy(args[0]); return py.NONE })
    }
    return undefined
  }
}

const locatorTicks = (loc: Locator, a: number, b: number, space: number, minpos: number, major: number[] = []): number[] => {
  switch (loc.k) {
    case 'auto': return maxNTicks(a, b, AUTO, space)
    case 'maxn': return maxNTicks(a, b, loc.m, space)
    case 'fixed': {
      if (!loc.nbins) return loc.at
      const stride = Math.max(Math.ceil(loc.at.length / loc.nbins), 1)
      const every = (from: number) => loc.at.filter((_, i) => i % stride === from)
      let ticks = every(0)
      for (let i = 1; i < stride; i++) {
        const other = every(i)
        if (minOf(other.map(Math.abs)) < minOf(ticks.map(Math.abs))) ticks = other
      }
      return ticks
    }
    case 'null': return []
    case 'multiple': return multipleTicks(a, b, loc.base, loc.offset)
    case 'log': return logTicks(Math.min(a, b), Math.max(a, b), loc.base, loc.subs, space, minpos)
    case 'linear': return linearTicks(a, b, loc.n)
    case 'minor': return autoMinor(major, a, b, loc.ndivs)
    case 'cats': return []
  }
}

const SCALAR: Formatter = { k: 'scalar', useOffset: true, limits: null, sci: true }

/** printf for one number: '%1.1f%%', '%d', '%g', '%.2e', '%s' */
const printf = (fmt: string, v: number): string =>
  fmt.replace(/%([-+ #0]*)(\d*)(?:\.(\d+))?([dfiegEGFs%])/g, (_m, flags: string, width: string, prec: string | undefined, conv: string) => {
    if (conv === '%') return '%'
    let s: string
    const p = prec === undefined ? null : Number(prec)
    switch (conv) {
      case 'd': case 'i': s = String(Math.trunc(v)); break
      case 'f': case 'F': s = v.toFixed(p ?? 6); break
      case 'e': case 'E': {
        const [m, e] = v.toExponential(p ?? 6).split('e')
        s = `${m}e${Number(e) < 0 ? '-' : '+'}${String(Math.abs(Number(e))).padStart(2, '0')}`
        if (conv === 'E') s = s.toUpperCase()
        break
      }
      case 'g': case 'G': s = p === null ? formatG(v) : gWith(v, p); break
      default: s = Number.isInteger(v) ? `${v}.0` : String(v)
    }
    if (flags.includes('+') && v >= 0) s = `+${s}`
    const w = Number(width || 0)
    if (s.length < w) s = flags.includes('-') ? s.padEnd(w) : flags.includes('0') ? s.padStart(w, '0') : s.padStart(w)
    return s
  })
const gWith = (x: number, p0: number): string => {
  const p = p0 === 0 ? 1 : p0
  if (x === 0) return '0'
  const e = Math.floor(Math.log10(Math.abs(x)))
  if (e < -4 || e >= p) {
    const [m, ex] = x.toExponential(p - 1).split('e')
    return `${String(Number(m))}e${Number(ex) < 0 ? '-' : '+'}${String(Math.abs(Number(ex))).padStart(2, '0')}`
  }
  return String(Number(x.toPrecision(p)))
}

// ---------------------------------------------------------------- axis

const TICK_KEYS = ['size', 'width', 'color', 'tickdir', 'pad', 'labelsize', 'labelcolor', 'labelfontfamily', 'zorder', 'gridOn', 'tick1On', 'tick2On', 'label1On', 'label2On', 'length', 'direction', 'left', 'bottom', 'right', 'top', 'labelleft', 'labelbottom', 'labelright', 'labeltop', 'labelrotation', 'grid_agg_filter', 'grid_alpha', 'grid_animated', 'grid_antialiased', 'grid_clip_box', 'grid_clip_on', 'grid_clip_path', 'grid_color', 'grid_dash_capstyle', 'grid_dash_joinstyle', 'grid_dashes', 'grid_data', 'grid_drawstyle', 'grid_figure', 'grid_fillstyle', 'grid_gapcolor', 'grid_gid', 'grid_in_layout', 'grid_label', 'grid_linestyle', 'grid_linewidth', 'grid_marker', 'grid_markeredgecolor', 'grid_markeredgewidth', 'grid_markerfacecolor', 'grid_markerfacecoloralt', 'grid_markersize', 'grid_markevery', 'grid_mouseover', 'grid_path', 'grid_path_effects', 'grid_picker', 'grid_pickradius', 'grid_rasterized', 'grid_sketch_params', 'grid_snap', 'grid_solid_capstyle', 'grid_solid_joinstyle', 'grid_transform', 'grid_url', 'grid_visible', 'grid_xdata', 'grid_ydata', 'grid_zorder', 'grid_aa', 'grid_c', 'grid_ds', 'grid_ls', 'grid_lw', 'grid_mec', 'grid_mew', 'grid_mfc', 'grid_mfcalt', 'grid_ms']

type AxisName = 'x' | 'y' | 'z'

class Axis extends PyObj {
  readonly cls: string
  name: AxisName
  ax: Axes
  lim: [number, number] = [0, 1]
  auto = true
  stale = false
  lo = Infinity
  hi = -Infinity
  minpos = Infinity
  margin: number
  marginValue: Value | null = null
  scale = 'linear'
  base = 10
  ticker: { loc: Locator; fmt: Formatter; minorLoc: Locator; minorFmt: Formatter }
  cats: Map<string, number> | null = null
  siblings: Axis[]
  visible = true
  style: Omit<AxisSpec, 'at' | 'labels' | 'minor' | 'minorLabels' | 'offset' | 'visible' | 'label'>
  labelObj: TextObj
  labelLoc: 'left' | 'center' | 'right' = 'center'
  grid = false
  gridMinor = false
  offsetVisible = true
  constructor(name: AxisName, ax: Axes, h: Host) {
    super()
    this.name = name
    this.cls = `${name.toUpperCase()}Axis`
    this.ax = ax
    this.margin = name === 'z' ? 0 : rcn(h, `axes.${name}margin`)
    this.ticker = { loc: { k: 'auto' }, fmt: { ...SCALAR }, minorLoc: { k: 'null' }, minorFmt: { k: 'null' } }
    this.siblings = [this]
    const t = name === 'y' ? 'ytick' : 'xtick'
    const lblSize = rcFont(h, `${t}.labelsize`)
    const c = rcc(h, `${t}.color`)
    const lc = rcv(h, `${t}.labelcolor`)
    this.style = {
      tick1: name === 'y' ? rcb(h, 'ytick.left') : rcb(h, 'xtick.bottom'), tick2: name === 'y' ? rcb(h, 'ytick.right') : rcb(h, 'xtick.top'),
      label1: name === 'y' ? rcb(h, 'ytick.labelleft') : rcb(h, 'xtick.labelbottom'), label2: name === 'y' ? rcb(h, 'ytick.labelright') : rcb(h, 'xtick.labeltop'),
      length: rcn(h, `${t}.major.size`), width: rcn(h, `${t}.major.width`), minorLength: rcn(h, `${t}.minor.size`), minorWidth: rcn(h, `${t}.minor.width`),
      color: c, dir: rcs(h, `${t}.direction`) as 'out', pad: rcn(h, `${t}.major.pad`), labelSize: lblSize,
      labelColor: lc.k === 'str' && lc.v === 'inherit' ? c : toColor(h, lc), rotation: 0, ha: null, va: null, styles: {}, labelSide: 'low', labelPad: rcn(h, 'axes.labelpad'),
    }
    const lab = newText(h, name === 'y' ? 0 : 0.5, name === 'y' ? 0.5 : 0, '')
    lab.size = rcFont(h, 'axes.labelsize')
    lab.color = rcc(h, 'axes.labelcolor')
    lab.weight = rcs(h, 'axes.labelweight')
    this.labelObj = new TextObj(lab, name === 'y' ? '0' : '0.5', name === 'y' ? '0.5' : '0')
    this.labelObj.owned = true
    this.labelObj.ax = ax
  }
  repr() { return `<matplotlib.axis.${this.cls} object at ${addr(this)}>` }
  isLog() { return this.scale === 'log' }

  /** the view limits, autoscaling first if something asked for it */
  view(): [number, number] {
    this.ax.unstale()
    return this.lim
  }
  /** Bbox.update_from_data */
  update(xs: number[]) {
    for (const v of xs) {
      if (!Number.isFinite(v)) continue
      if (v < this.lo) this.lo = v
      if (v > this.hi) this.hi = v
      if (v > 0 && v < this.minpos) this.minpos = v
    }
  }
  minPos() { return Math.min(...this.siblings.map((a) => a.minpos)) }
  nonsingular(a: number, b: number): [number, number] {
    return this.isLog() ? logNonsingular(a, b, this.base, this.minPos()) : nonsingular(a, b, 0.05)
  }
  /** Axis._set_lim; auto null leaves autoscale as it is */
  setLim(h: Host, v0: number | null, v1: number | null, auto: boolean | null) {
    let a = v0
    let b = v1
    if (a === null || b === null) {
      const old = this.view()
      a ??= old[0]
      b ??= old[1]
    }
    if (!Number.isFinite(a) || !Number.isFinite(b)) throw h.err('ValueError', `Axis limits cannot be NaN or Inf`)
    if (this.isLog() && (a <= 0 || b <= 0)) {
      const old = this.view()
      if (a <= 0) a = old[0]
      if (b <= 0) b = old[1]
    }
    const reverse = a > b
    let [lo, hi] = this.nonsingular(a, b)
    if (this.isLog()) {
      const mp = Number.isFinite(this.minPos()) ? this.minPos() : 1e-300
      if (lo <= 0) lo = mp
      if (hi <= 0) hi = mp
    }
    this.lim = reverse ? [Math.max(lo, hi), Math.min(lo, hi)] : [Math.min(lo, hi), Math.max(lo, hi)]
    for (const s of this.siblings) {
      s.stale = false
      s.lim = this.lim
      if (auto !== null) s.auto = auto
    }
    return this.lim
  }
  /** set_view_interval without ignore: only ever grows */
  grow(vmin: number, vmax: number) {
    const [o0, o1] = this.view()
    this.lim = o0 < o1 ? [Math.min(vmin, vmax, o0), Math.max(vmin, vmax, o1)] : [Math.max(vmin, vmax, o0), Math.min(vmin, vmax, o1)]
    for (const s of this.siblings) s.lim = this.lim
  }
  inverted() { return this.view()[0] > this.view()[1] }
  setBound(h: Host, lower: number, upper: number) {
    const inv = this.inverted()
    this.setLim(h, inv ? Math.max(lower, upper) : Math.min(lower, upper), inv ? Math.min(lower, upper) : Math.max(lower, upper), null)
  }
  setScale(h: Host, name: string, kw: Kw) {
    oneOf(h, name, 'scale', ['linear', 'log', 'symlog', 'asinh', 'logit', 'function', 'functionlog'])
    if (name !== 'linear' && name !== 'log') throw h.err('NotImplementedError', `${this.name}scale('${name}') is not drawn in this sandbox (use 'linear' or 'log')`)
    const base = optNum(h, pop(kw, 'base', `base${this.name}`)) ?? 10
    const subs = pop(kw, 'subs', `subs${this.name}`)
    const np = optStr(h, pop(kw, 'nonpositive', `nonpos${this.name}`)) ?? 'clip'
    if (np !== 'clip' && np !== 'mask') throw h.err('ValueError', `nonpositive must be 'mask' or 'clip'`)
    if (base <= 0 || base === 1) throw h.err('ValueError', 'The log base cannot be <= 0 or == 1')
    const keys = Object.keys(kw)
    if (keys.length) throw h.err('TypeError', `LogScale.__init__() got an unexpected keyword argument '${keys[0]}'`)
    const was = this.scale
    for (const s of this.siblings) {
      s.scale = name
      s.base = base
    }
    this.ticker.loc = name === 'log' ? { k: 'log', base, subs: null } : { k: 'auto' }
    this.ticker.fmt = name === 'log' ? { k: 'log', base } : { ...SCALAR }
    this.ticker.minorLoc = name === 'log' ? { k: 'log', base, subs: subs ? nums(h, subs, 'subs') : 'auto' } : { k: 'null' }
    this.ticker.minorFmt = name === 'log' ? { k: 'log', base } : { k: 'null' }
    if (was !== name) {
      if (this.name === 'x') this.ax.autoscaleView(h, null, true, false)
      else if (this.name === 'y') this.ax.autoscaleView(h, null, false, true)
    }
  }
  /** the distance across the axis in points, as get_tick_space uses it */
  tickSpace(rect: Rect): number {
    const [W, H] = this.ax.fig.spec.size
    // a 3D Axes's three axes all count as x axes here
    const vertical = this.name === 'y' && !(this.ax instanceof Axes3D)
    const len = (vertical ? rect[3] * H : rect[2] * W) * 72
    const size = this.style.labelSize * (vertical ? 2 : 3)
    return size > 0 ? Math.floor(len / size) : 2147483647
  }
  majorLocs(view: [number, number], space: number): number[] {
    const l = this.ticker.loc
    if (l.k === 'cats') return [...(this.cats?.values() ?? [])]
    if (l.k === 'auto' && this.ax instanceof Axes3D) return maxNTicks(view[0], view[1], { ...AUTO, rawScale: 23 / 24 }, space)
    return locatorTicks(l, view[0], view[1], space, this.minPos())
  }
  /** Axis.get_minorticklocs: minor ticks that fall on a major tick are dropped */
  minorLocs(view: [number, number], space: number, major: number[]): number[] {
    const minor = locatorTicks(this.ticker.minorLoc, view[0], view[1], space, this.minPos(), major)
    const t = (v: number) => (this.isLog() ? Math.log10(v) : v)
    const tol = Math.abs(t(view[1]) - t(view[0])) * 1e-5
    return minor.filter((m) => !major.some((v) => Math.abs(t(m) - t(v)) <= tol))
  }
  formatTicks(h: Host, fmt: Formatter, locs: number[], view: [number, number]): { labels: string[]; offset: string } {
    switch (fmt.k) {
      case 'scalar': return scalarLabels(locs, view, { useOffset: fmt.useOffset, limits: fmt.limits, sci: fmt.sci })
      case 'log': return { labels: logLabels(locs, view, fmt.base), offset: '' }
      case 'fixed': return { labels: locs.map((_, i) => fmt.labels[i] ?? ''), offset: '' }
      case 'dict': return { labels: locs.map((l) => fmt.map.get(l) ?? ''), offset: '' }
      case 'func': return { labels: locs.map((l, i) => h.str(h.call(fmt.fn, [py.float(l), py.int(i)]))), offset: '' }
      case 'fmt': return { labels: locs.map((l) => printf(fmt.fmt, l)), offset: '' }
      case 'str': return {
        labels: locs.map((l, i) => fmt.fmt.replace(/\{(x|pos)(?::([^}]*))?\}/g, (_m, key: string, spec?: string) => {
          const v = key === 'x' ? py.float(l) : py.int(i)
          return (spec ? formatSpec(v, spec) : h.str(v)).replace(/-/g, '\u2212')
        })),
        offset: '',
      }
      case 'percent': return {
        labels: locs.map((l) => {
          const pct = (x: number) => (100 * x) / fmt.xmax
          const range = pct(Math.abs(view[1] - view[0]))
          const d = fmt.decimals ?? (range <= 0 ? 0 : Math.min(5, Math.max(0, Math.ceil(2 - Math.log10(2 * range)))))
          return `${pct(l).toFixed(d)}${fmt.symbol}`.replace(/-/g, '\u2212')
        }),
        offset: '',
      }
      case 'null': return { labels: locs.map(() => ''), offset: '' }
      case 'cats': {
        const names = [...(this.cats?.entries() ?? [])]
        return { labels: locs.map((l) => names.find(([, v]) => v === l)?.[0] ?? ''), offset: '' }
      }
    }
  }
  /** ticks and labels for a view: positions outside it are dropped, as Axis._update_ticks does */
  resolve(h: Host, view: [number, number], space: number): AxisSpec {
    const major = this.majorLocs(view, space)
    const { labels, offset } = this.formatTicks(h, this.ticker.fmt, major, view)
    const minor = this.minorLocs(view, space, major)
    const minorText = this.formatTicks(h, this.ticker.minorFmt, minor, view).labels
    const vlo = Math.min(...view)
    const vhi = Math.max(...view)
    const tol = (vhi - vlo) * 1e-10
    const logView = this.isLog()
    const within = (v: number) => (logView ? Math.log10(Math.max(v, 1e-300)) >= Math.log10(vlo) - 1e-10 && Math.log10(Math.max(v, 1e-300)) <= Math.log10(vhi) + 1e-10 : v >= vlo - tol && v <= vhi + tol)
    const at: number[] = []
    const out: string[] = []
    const styles: AxisSpec['styles'] = {}
    major.forEach((v, i) => {
      if (!within(v)) return
      if (this.style.styles[i]) styles[at.length] = this.style.styles[i]
      at.push(v)
      out.push(labels[i])
    })
    const mAt: number[] = []
    const mOut: string[] = []
    minor.forEach((v, i) => {
      if (!within(v)) return
      mAt.push(v)
      mOut.push(minorText[i])
    })
    const lab = this.labelObj.spec
    return {
      ...this.style, styles, at, labels: out, minor: mAt, minorLabels: mOut, offset: this.offsetVisible ? offset : '', visible: this.visible,
      label: lab.text ? { text: lab.text, size: lab.size, color: lab.color, weight: lab.weight, italic: lab.italic, loc: this.labelLoc, rotation: lab.rotation } : null,
    }
  }
  /** get_xticks(): an array of the locator's dtype; category axes give a plain list, as matplotlib's locator does */
  tickValue(h: Host): Value {
    const { locs } = this.current(h)
    const l = this.ticker.loc
    if (l.k === 'cats') return py.list(locs.map(py.int))
    if (l.k === 'null') return py.list([])
    return py.arr(NDArray.create(locs, [locs.length], l.k === 'fixed' && l.dtype ? l.dtype : 'float64'))
  }
  /** current tick positions and labels, before any drawing has applied the aspect */
  current(h: Host): { locs: number[]; labels: string[] } {
    const view = this.view()
    const locs = this.majorLocs(view, this.tickSpace(this.ax.position()))
    return { locs, labels: this.formatTicks(h, this.ticker.fmt, locs, view).labels }
  }
  setTicks(h: Host, tv: Value, labels: Value | undefined, minor: boolean, kw: Kw = {}) {
    const at = nums(h, tv, `${this.name}ticks`)
    const dtype = at.length ? toArray(tv).dtype : 'float64'
    if (at.length) this.grow(minOf(at), maxOf(at))
    if (minor) this.ticker.minorLoc = { k: 'fixed', at, dtype }
    else this.ticker.loc = { k: 'fixed', at, dtype }
    if (!isNone(labels)) this.setTickLabels(h, labels!, minor, kw)
    return at
  }
  setTickLabels(h: Host, lv: Value, minor: boolean, kw: Kw = {}) {
    const raw = h.iterate(lv).map((v) => (v.k === 'obj' && v.o instanceof TickLabel ? v.o.text() : h.str(v)))
    const loc = minor ? this.ticker.minorLoc : this.ticker.loc
    let fmt: Formatter
    if (!raw.length) fmt = { k: 'null' }
    else if (loc.k === 'fixed') {
      if (loc.at.length !== raw.length) throw h.err('ValueError', `The number of FixedLocator locations (${loc.at.length}), usually from a call to set_ticks, does not match the number of labels (${raw.length}).`)
      fmt = { k: 'dict', map: new Map(loc.at.map((l, i) => [l, raw[i]])) }
    } else fmt = { k: 'fixed', labels: raw }
    if (minor) this.ticker.minorFmt = fmt
    else this.ticker.fmt = fmt
    this.applyLabelKw(h, kw)
  }
  /** Text keywords on tick labels: the whole axis gets them */
  applyLabelKw(h: Host, kw0: Kw) {
    const kw = { ...kw0 }
    const st = this.style
    for (const [k, v] of Object.entries(kw)) {
      switch (k) {
        case 'rotation': st.rotation = v.k === 'str' ? (v.v === 'vertical' ? 90 : 0) : h.num(v); break
        case 'fontsize': case 'size': st.labelSize = fontPt(h, v); break
        case 'color': case 'c': st.labelColor = toColor(h, v); break
        case 'ha': case 'horizontalalignment': st.ha = h.str(v); break
        case 'va': case 'verticalalignment': st.va = h.str(v); break
        case 'visible': { if (!h.truthy(v)) { st.label1 = false; st.label2 = false } break }
        default:
          if (!COSMETIC.has(k) && !['fontweight', 'weight', 'fontstyle', 'style', 'alpha'].includes(k)) throw h.err('AttributeError', `Text.set() got an unexpected keyword argument '${k}'`)
      }
    }
  }
  tickParams(h: Host, which: string, kw: Kw) {
    const st = this.style
    const major = which !== 'minor'
    const minor = which !== 'major'
    const low = this.name === 'y' ? 'left' : 'bottom'
    const high = this.name === 'y' ? 'right' : 'top'
    for (const [k, v] of Object.entries(kw)) {
      switch (k) {
        case 'size': case 'length': if (major) st.length = h.num(v); if (minor) st.minorLength = h.num(v); break
        case 'width': if (major) st.width = h.num(v); if (minor) st.minorWidth = h.num(v); break
        case 'color': st.color = toColor(h, v); break
        case 'labelcolor': st.labelColor = toColor(h, v); break
        case 'colors': st.color = toColor(h, v); st.labelColor = st.color; break
        case 'pad': st.pad = h.num(v); break
        case 'labelsize': st.labelSize = fontPt(h, v); break
        case 'direction': case 'tickdir': st.dir = oneOf(h, h.str(v), 'direction', ['in', 'out', 'inout']) as 'in'; break
        case 'labelrotation': st.rotation = h.num(v); break
        case 'gridOn': this.grid = h.truthy(v); break
        case 'tick1On': st.tick1 = h.truthy(v); break
        case 'tick2On': st.tick2 = h.truthy(v); break
        case 'label1On': st.label1 = h.truthy(v); break
        case 'label2On': st.label2 = h.truthy(v); break
        case 'grid_color': case 'grid_alpha': case 'grid_linewidth': case 'grid_linestyle': case 'grid_lw': case 'grid_ls': case 'grid_c':
          this.ax.gridStyle(h, { [k.slice(5)]: v })
          break
        default:
          if (k === low) st.tick1 = h.truthy(v)
          else if (k === high) st.tick2 = h.truthy(v)
          else if (k === `label${low}`) st.label1 = h.truthy(v)
          else if (k === `label${high}`) st.label2 = h.truthy(v)
          else if (k === 'left' || k === 'right' || k === 'top' || k === 'bottom' || k.startsWith('label')) { /* the other axis's side */ } else if (!TICK_KEYS.includes(k)) {
            throw h.err('ValueError', `keyword ${k} is not recognized; valid keywords are [${TICK_KEYS.map((n) => `'${n}'`).join(', ')}]`)
          }
      }
    }
  }
  tickLabels(h: Host, minor = false): Value[] {
    const view = this.view()
    const space = this.tickSpace(this.ax.position())
    const major = this.majorLocs(view, space)
    const locs = minor ? this.minorLocs(view, space, major) : major
    const labels = this.formatTicks(h, minor ? this.ticker.minorFmt : this.ticker.fmt, locs, view).labels
    // Axis.get_majorticklabels: only the sides that show labels, the low side first
    const one = () => locs.map((l, i) => py.obj(new TickLabel(this, i, l, labels[i], minor)))
    return [...(this.style.label1 ? one() : []), ...(this.style.label2 ? one() : [])]
  }
  getAttr(name: string, h: Host): Value | undefined {
    const st = this.style
    switch (name) {
      case 'axes': return py.obj(this.ax)
      case 'label': return py.obj(this.labelObj)
      case 'labelpad': return py.float(this.style.labelPad)
      case 'axis_name': return py.str(this.name)
      case 'set_visible': return fnv(name, (args) => { this.visible = h.truthy(args[0]); emitFig(h, this.ax.fig, this.ax); return py.NONE })
      case 'get_visible': return fnv(name, () => py.bool(this.visible))
      case 'set_ticks': return fnv(name, (args, kw) => {
        const minor = flag(h, pop(kw, 'minor'), false)
        const labels = args[1] ?? pop(kw, 'labels')
        const at = this.setTicks(h, args[0] ?? pop(kw, 'ticks')!, labels, minor, kw)
        emitFig(h, this.ax.fig, this.ax)
        return py.list(at.map((_, i) => py.obj(new TickObj(this, i))))
      })
      case 'get_ticklocs': case 'get_majorticklocs': return fnv(name, () => this.tickValue(h))
      case 'get_minorticklocs': return fnv(name, () => { const v = this.view(); const sp = this.tickSpace(this.ax.position()); return floats(this.minorLocs(v, sp, this.majorLocs(v, sp))) })
      case 'set_ticklabels': return fnv(name, (args, kw) => {
        const minor = flag(h, pop(kw, 'minor'), false)
        this.setTickLabels(h, args[0] ?? pop(kw, 'labels')!, minor, kw)
        emitFig(h, this.ax.fig, this.ax)
        return py.list(this.tickLabels(h, minor))
      })
      case 'get_ticklabels': case 'get_majorticklabels': return fnv(name, (_a, kw) => py.list(this.tickLabels(h, flag(h, kw.minor, false))))
      case 'get_minorticklabels': return fnv(name, () => py.list(this.tickLabels(h, true)))
      case 'get_major_ticks': return fnv(name, () => py.list(this.current(h).locs.map((_, i) => py.obj(new TickObj(this, i)))))
      case 'set_major_locator': case 'set_minor_locator': return fnv(name, (args) => {
        const v = args[0]
        if (v.k !== 'obj' || !(v.o instanceof TickerObj) || !v.o.loc) throw h.err('TypeError', 'locator must be a subclass of matplotlib.ticker.Locator')
        if (name === 'set_major_locator') this.ticker.loc = v.o.loc
        else this.ticker.minorLoc = v.o.loc
        emitFig(h, this.ax.fig, this.ax)
        return py.NONE
      })
      case 'set_major_formatter': case 'set_minor_formatter': return fnv(name, (args) => {
        const v = args[0]
        let fmt: Formatter
        if (v.k === 'str') fmt = { k: 'str', fmt: v.v }
        else if (v.k === 'obj' && v.o instanceof TickerObj && v.o.fmt) fmt = v.o.fmt
        else if (v.k === 'func' || v.k === 'fn' || v.k === 'bound') fmt = { k: 'func', fn: v }
        else throw h.err('TypeError', 'formatter must be a subclass of matplotlib.ticker.Formatter')
        if (name === 'set_major_formatter') this.ticker.fmt = fmt
        else this.ticker.minorFmt = fmt
        emitFig(h, this.ax.fig, this.ax)
        return py.NONE
      })
      case 'get_major_locator': return fnv(name, () => py.obj(new TickerObj('Locator', this.ticker.loc, null)))
      case 'get_major_formatter': return fnv(name, () => py.obj(new TickerObj('Formatter', null, this.ticker.fmt)))
      case 'set_tick_params': return fnv(name, (_a, kw) => {
        const k = { ...kw }
        const which = optStr(h, pop(k, 'which')) ?? 'major'
        pop(k, 'reset')
        this.tickParams(h, which, k)
        return py.NONE
      })
      case 'tick_top': case 'tick_bottom': case 'tick_left': case 'tick_right': return fnv(name, () => {
        const high = name === 'tick_top' || name === 'tick_right'
        st.tick1 = !high
        st.tick2 = high
        st.label1 = !high
        st.label2 = high
        emitFig(h, this.ax.fig, this.ax)
        return py.NONE
      })
      case 'set_ticks_position': return fnv(name, (args) => {
        const p = h.str(args[0])
        if (p === 'top' || p === 'right') { st.tick1 = false; st.tick2 = true; st.label1 = false; st.label2 = true } else if (p === 'bottom' || p === 'left') { st.tick1 = true; st.tick2 = false; st.label1 = true; st.label2 = false } else if (p === 'both') { st.tick1 = true; st.tick2 = true } else if (p === 'none') { st.tick1 = false; st.tick2 = false } else if (p !== 'default') throw h.err('ValueError', `'${p}' is not a valid value for position`)
        return py.NONE
      })
      case 'set_label_position': return fnv(name, (args) => {
        const p = h.str(args[0])
        st.labelSide = p === 'top' || p === 'right' ? 'high' : 'low'
        return py.NONE
      })
      case 'set_label_text': return fnv(name, (args) => { this.labelObj.spec.text = h.str(args[0]); return py.obj(this.labelObj) })
      case 'get_label_text': return fnv(name, () => py.str(this.labelObj.spec.text))
      case 'get_label': return fnv(name, () => py.obj(this.labelObj))
      case 'get_label_position': return fnv(name, () => py.str(st.labelSide === 'high' ? (this.name === 'y' ? 'right' : 'top') : (this.name === 'y' ? 'left' : 'bottom')))
      case 'get_offset_text': return fnv(name, () => {
        const hide = (a: Value[]) => { this.offsetVisible = h.truthy(a[0]); return py.NONE }
        return py.obj(new (class extends PyObj {
          readonly cls = 'Text'
          repr() { return "Text(0, 0.5, '')" }
          getAttr(n: string): Value | undefined {
            if (n === 'set_visible') return fnv(n, hide)
            if (n.startsWith('set_')) return fnv(n, () => py.NONE)
            return undefined
          }
        })())
      })
      case 'grid': return fnv(name, (args, kw) => {
        this.ax.setGrid(h, args, { ...kw, axis: py.str(this.name) })
        return py.NONE
      })
      case 'get_scale': return fnv(name, () => py.str(this.scale))
      case 'get_view_interval': return fnv(name, () => floats(this.view()))
      case 'get_data_interval': return fnv(name, () => floats([this.lo, this.hi]))
      case 'get_inverted': return fnv(name, () => py.bool(this.inverted()))
      case 'set_inverted': return fnv(name, (args) => {
        const [a, b] = this.view()
        const want = h.truthy(args[0])
        this.setLim(h, want ? Math.max(a, b) : Math.min(a, b), want ? Math.min(a, b) : Math.max(a, b), null)
        return py.NONE
      })
      case 'minorticks_on': return fnv(name, () => { this.minorOn(); return py.NONE })
      case 'minorticks_off': return fnv(name, () => { this.ticker.minorLoc = { k: 'null' }; return py.NONE })
    }
    return undefined
  }
  setAttr(name: string, v: Value, h: Host): boolean {
    if (name !== 'labelpad') return false
    this.style.labelPad = h.num(v)
    return true
  }
  minorOn() {
    if (this.isLog()) this.ticker.minorLoc = { k: 'log', base: this.base, subs: 'auto' }
    else this.ticker.minorLoc = { k: 'minor', ndivs: null }
  }
}

/** a tick's label: its setters act on that one tick */
class TickLabel extends PyObj {
  readonly cls = 'Text'
  axis: Axis
  i: number
  loc: number
  content: string
  minor: boolean
  constructor(axis: Axis, i: number, loc: number, content: string, minor: boolean) {
    super()
    this.axis = axis
    this.i = i
    this.loc = loc
    this.content = content
    this.minor = minor
  }
  text() { return this.content }
  repr() {
    const pos = this.axis.name === 'y' ? `0, ${pyRepr(py.float(this.loc))}` : `${pyRepr(py.float(this.loc))}, 0`
    return `Text(${pos}, ${pyRepr(py.str(this.content))})`
  }
  getAttr(name: string, h: Host): Value | undefined {
    const st = () => (this.axis.style.styles[this.i] ??= {})
    const done = () => { emitFig(h, this.axis.ax.fig, this.axis.ax); return py.NONE }
    switch (name) {
      case 'get_text': return fnv(name, () => py.str(this.content))
      case 'get_position': return fnv(name, () => py.tuple(this.axis.name === 'y' ? [py.int(0), py.float(this.loc)] : [py.float(this.loc), py.int(0)]))
      case 'get_rotation': return fnv(name, () => py.float(this.axis.style.styles[this.i]?.rotation ?? this.axis.style.rotation))
      case 'get_fontsize': case 'get_size': return fnv(name, () => py.float(this.axis.style.styles[this.i]?.size ?? this.axis.style.labelSize))
      case 'get_color': return fnv(name, () => py.str(this.axis.style.styles[this.i]?.color ?? this.axis.style.labelColor))
      case 'get_visible': return fnv(name, () => py.bool(this.axis.style.styles[this.i]?.visible ?? true))
      case 'set_rotation': return fnv(name, (a) => { st().rotation = a[0].k === 'str' ? (a[0].v === 'vertical' ? 90 : 0) : h.num(a[0]); return done() })
      case 'set_fontsize': case 'set_size': return fnv(name, (a) => { st().size = fontPt(h, a[0]); return done() })
      case 'set_color': case 'set_c': return fnv(name, (a) => { st().color = toColor(h, a[0]); return done() })
      case 'set_visible': return fnv(name, (a) => { st().visible = h.truthy(a[0]); return done() })
      case 'set_fontweight': case 'set_weight': return fnv(name, (a) => { st().weight = h.str(a[0]); return done() })
      case 'set_ha': case 'set_horizontalalignment': return fnv(name, (a) => { this.axis.style.ha = h.str(a[0]); return done() })
      case 'set_va': case 'set_verticalalignment': return fnv(name, (a) => { this.axis.style.va = h.str(a[0]); return done() })
      case 'set_text': return fnv(name, () => { throw h.err('NotImplementedError', 'a tick label\'s set_text does not stick in matplotlib either; use set_xticklabels') })
      case 'set': return fnv(name, (_a, kw) => {
        for (const [k, v] of Object.entries(kw)) {
          const f = this.getAttr(`set_${k}`, h)
          if (f && f.k === 'fn') f.call([v], {})
          else if (!COSMETIC.has(k)) throw h.err('AttributeError', `Text.set() got an unexpected keyword argument '${k}'`)
        }
        return py.NONE
      })
    }
    if (name.startsWith('set_') && COSMETIC.has(name.slice(4))) return fnv(name, () => py.NONE)
    return undefined
  }
}

class TickObj extends PyObj {
  readonly cls: string
  axis: Axis
  i: number
  constructor(axis: Axis, i: number) {
    super()
    this.axis = axis
    this.i = i
    this.cls = `${axis.name.toUpperCase()}Tick`
  }
  repr() { return `<matplotlib.axis.${this.cls} object at ${addr(this)}>` }
  getAttr(name: string, h: Host): Value | undefined {
    if (name === 'label1' || name === 'label2') {
      const lab = this.axis.tickLabels(h)[this.i]
      return lab
    }
    if (name === 'set_visible') return fnv(name, (a) => { (this.axis.style.styles[this.i] ??= {}).visible = h.truthy(a[0]); return py.NONE })
    if (name === 'get_loc') return fnv(name, () => py.float(this.axis.current(h).locs[this.i]))
    return undefined
  }
}

// ---------------------------------------------------------------- layout

type Rect = [number, number, number, number]

interface Params { left: number; right: number; bottom: number; top: number; wspace: number; hspace: number }

class GridSpec extends PyObj {
  readonly cls = 'GridSpec'
  fig: Figure | null
  nrows: number
  ncols: number
  own: Partial<Params> = {}
  wr: number[]
  hr: number[]
  parent: SubplotSpec | null = null
  constructor(fig: Figure | null, nrows: number, ncols: number, wr?: number[], hr?: number[]) {
    super()
    this.fig = fig
    this.nrows = nrows
    this.ncols = ncols
    this.wr = wr ?? new Array<number>(ncols).fill(1)
    this.hr = hr ?? new Array<number>(nrows).fill(1)
  }
  repr() { return `GridSpec(${this.nrows}, ${this.ncols})` }
  params(fig: Figure): Params {
    const base = fig.params
    if (this.parent) {
      const [l, b, w, hh] = this.parent.rect(fig)
      return { left: l, right: l + w, bottom: b, top: b + hh, wspace: this.own.wspace ?? base.wspace, hspace: this.own.hspace ?? base.hspace }
    }
    return { ...base, ...this.own }
  }
  /** GridSpecBase.get_grid_positions */
  grid(fig: Figure) {
    const p = this.params(fig)
    const cum = (seps: number[], cells: number[]) => {
      const out: number[] = []
      let t = 0
      cells.forEach((c, i) => {
        t += seps[i]
        out.push(t)
        t += c
        out.push(t)
      })
      return out
    }
    const cellH = (p.top - p.bottom) / (this.nrows + p.hspace * (this.nrows - 1))
    const normH = (cellH * this.nrows) / this.hr.reduce((a, b) => a + b, 0)
    const hs = cum(this.hr.map((_, i) => (i ? p.hspace * cellH : 0)), this.hr.map((r) => r * normH))
    const cellW = (p.right - p.left) / (this.ncols + p.wspace * (this.ncols - 1))
    const normW = (cellW * this.ncols) / this.wr.reduce((a, b) => a + b, 0)
    const ws = cum(this.wr.map((_, i) => (i ? p.wspace * cellW : 0)), this.wr.map((r) => r * normW))
    return {
      tops: this.hr.map((_, i) => p.top - hs[i * 2]), bottoms: this.hr.map((_, i) => p.top - hs[i * 2 + 1]),
      lefts: this.wr.map((_, i) => p.left + ws[i * 2]), rights: this.wr.map((_, i) => p.left + ws[i * 2 + 1]),
    }
  }
  getItem(idx: Value, h: Host): Value {
    const parts = idx.k === 'tuple' ? idx.items : [idx]
    const span = (v: Value, n: number): [number, number] => {
      if (v.k === 'slice') {
        const s = isNone(v.start) ? 0 : h.toInt(v.start)
        const e = isNone(v.stop) ? n : h.toInt(v.stop)
        const a = s < 0 ? s + n : s
        const b = e < 0 ? e + n : e
        if (a >= b) throw h.err('IndexError', 'GridSpec slice would result in no space allocated for subplot')
        return [a, Math.min(b, n)]
      }
      const i = h.toInt(v, 'an index')
      const k = i < 0 ? i + n : i
      if (k < 0 || k >= n) throw h.err('IndexError', 'index out of bounds')
      return [k, k + 1]
    }
    if (parts.length === 2) {
      const [r0, r1] = span(parts[0], this.nrows)
      const [c0, c1] = span(parts[1], this.ncols)
      return py.obj(new SubplotSpec(this, r0, r1, c0, c1))
    }
    if (this.nrows === 1) {
      const [c0, c1] = span(parts[0], this.ncols)
      return py.obj(new SubplotSpec(this, 0, 1, c0, c1))
    }
    if (this.ncols === 1) {
      const [r0, r1] = span(parts[0], this.nrows)
      return py.obj(new SubplotSpec(this, r0, r1, 0, 1))
    }
    const [a, b] = span(parts[0], this.nrows * this.ncols)
    return py.obj(new SubplotSpec(this, Math.floor(a / this.ncols), Math.floor((b - 1) / this.ncols) + 1, a % this.ncols, ((b - 1) % this.ncols) + 1))
  }
  getAttr(name: string, h: Host): Value | undefined {
    switch (name) {
      case 'nrows': return py.int(this.nrows)
      case 'ncols': return py.int(this.ncols)
      case 'get_geometry': return fnv(name, () => py.tuple([py.int(this.nrows), py.int(this.ncols)]))
      case 'update': return fnv(name, (_a, kw) => {
        for (const k of ['left', 'right', 'bottom', 'top', 'wspace', 'hspace'] as const) if (kw[k]) this.own[k] = h.num(kw[k])
        return py.NONE
      })
      case 'tight_layout': return fnv(name, () => py.NONE)
    }
    return undefined
  }
}

class SubplotSpec extends PyObj {
  readonly cls = 'SubplotSpec'
  gs: GridSpec
  r0: number
  r1: number
  c0: number
  c1: number
  constructor(gs: GridSpec, r0: number, r1: number, c0: number, c1: number) {
    super()
    this.gs = gs
    this.r0 = r0
    this.r1 = r1
    this.c0 = c0
    this.c1 = c1
  }
  repr() { return `GridSpec(${this.gs.nrows}, ${this.gs.ncols})[${this.r0}:${this.r1}, ${this.c0}:${this.c1}]` }
  rect(fig: Figure): Rect {
    const g = this.gs.grid(fig)
    const left = g.lefts[this.c0]
    const right = g.rights[this.c1 - 1]
    const top = g.tops[this.r0]
    const bottom = g.bottoms[this.r1 - 1]
    return [left, bottom, right - left, top - bottom]
  }
  topmost(): SubplotSpec { return this.gs.parent ? this.gs.parent.topmost() : this }
  same(o: SubplotSpec) { return this.gs.nrows === o.gs.nrows && this.gs.ncols === o.gs.ncols && !this.gs.parent && !o.gs.parent && this.r0 === o.r0 && this.r1 === o.r1 && this.c0 === o.c0 && this.c1 === o.c1 }
  getAttr(name: string, h: Host): Value | undefined {
    switch (name) {
      case 'get_gridspec': return fnv(name, () => py.obj(this.gs))
      case 'is_first_row': return fnv(name, () => py.bool(this.r0 === 0))
      case 'is_last_row': return fnv(name, () => py.bool(this.r1 === this.gs.nrows))
      case 'is_first_col': return fnv(name, () => py.bool(this.c0 === 0))
      case 'is_last_col': return fnv(name, () => py.bool(this.c1 === this.gs.ncols))
      case 'subgridspec': return fnv(name, (args, kw) => {
        const g = new GridSpec(this.gs.fig, h.toInt(args[0] ?? kw.nrows), h.toInt(args[1] ?? kw.ncols), kw.width_ratios ? nums(h, kw.width_ratios, 'width_ratios') : undefined, kw.height_ratios ? nums(h, kw.height_ratios, 'height_ratios') : undefined)
        g.parent = this
        if (kw.wspace) g.own.wspace = h.num(kw.wspace)
        if (kw.hspace) g.own.hspace = h.num(kw.hspace)
        return py.obj(g)
      })
    }
    return undefined
  }
}

const ANCHORS: Record<string, [number, number]> = { C: [0.5, 0.5], SW: [0, 0], S: [0.5, 0], SE: [1, 0], E: [1, 0.5], NE: [1, 1], N: [0.5, 1], NW: [0, 1], W: [0, 0.5] }

/** Bbox.shrunk_to_aspect + anchored */
const fitAspect = (pb: Rect, boxAspect: number, figAspect: number, anchor: [number, number]): Rect => {
  const [x, y, w, hh] = pb
  let H = (w * boxAspect) / figAspect
  let W = w
  if (H > hh) {
    W = (hh * figAspect) / boxAspect
    H = hh
  }
  return [x + anchor[0] * (w - W), y + anchor[1] * (hh - H), W, H]
}

/** tick labels' depth beyond the spine, in pixels (estimated text sizes) */
export const tickLabelDepth = (a: AxisSpec, horizontal: boolean, high: boolean): number => {
  const out = a.dir === 'out' ? a.length : a.dir === 'inout' ? a.length / 2 : 0
  const showing = (high ? a.label2 : a.label1) && a.visible
  const shown = showing ? [...a.labels, ...a.minorLabels].filter((l, i) => l && a.styles[i]?.visible !== false) : []
  const ticks = high ? a.tick2 : a.tick1
  let ext = 0
  for (const l of shown) {
    const w = textWidthPx(l, a.labelSize)
    const hh = textHeightPx(l, a.labelSize) * 0.97
    const r = (a.rotation * Math.PI) / 180
    const e = horizontal ? Math.abs(w * Math.sin(r)) + Math.abs(hh * Math.cos(r)) : Math.abs(w * Math.cos(r)) + Math.abs(hh * Math.sin(r))
    if (e > ext) ext = e
  }
  return ((ticks ? out : 0) + a.pad) * PT + ext
}

// ---------------------------------------------------------------- legends and spines

class LegendObj extends PyObj {
  readonly cls = 'Legend'
  spec: LegendSpec
  ax: Axes | null
  fig: Figure | null
  constructor(spec: LegendSpec, ax: Axes | null, fig: Figure | null) {
    super()
    this.spec = spec
    this.ax = ax
    this.fig = fig
  }
  repr() { return `<matplotlib.legend.Legend object at ${addr(this)}>` }
  getAttr(name: string, h: Host): Value | undefined {
    const s = this.spec
    const redraw = () => { const f = this.ax?.fig ?? this.fig; if (f) emitFig(h, f, this.ax) }
    switch (name) {
      case 'set_title': return fnv(name, (args, kw) => { s.title = h.str(args[0]); if (kw.prop) s.titleSize = s.size; redraw(); return py.NONE })
      case 'get_title': return fnv(name, () => py.obj(new TextObj(newText(h, 0, 0, s.title ?? ''))))
      case 'get_texts': return fnv(name, () => py.list(s.entries.map((e) => py.obj(new TextObj(newText(h, 0, 0, e.label))))))
      case 'set_visible': return fnv(name, (args) => { if (!h.truthy(args[0])) s.entries = []; s.frame = s.frame && h.truthy(args[0]); redraw(); return py.NONE })
      case 'remove': return fnv(name, () => {
        if (this.ax?.legendObj === this) this.ax.legendObj = null
        if (this.ax) this.ax.extraLegends = this.ax.extraLegends.filter((l) => l !== this)
        if (this.fig) this.fig.legends = this.fig.legends.filter((l) => l !== this)
        redraw()
        return py.NONE
      })
      case 'set_draggable': case 'set_zorder': case 'set_bbox_to_anchor': return fnv(name, () => py.NONE)
      case 'legend_handles': case 'legendHandles': return py.list([])
      case 'get_frame': return fnv(name, () => py.obj(new (class extends PyObj {
        readonly cls = 'FancyBboxPatch'
        repr() { return '<matplotlib.patches.FancyBboxPatch object>' }
        getAttr(n: string): Value | undefined {
          if (n === 'set_facecolor') return fnv(n, (a) => { s.fc = toColor(h, a[0]); redraw(); return py.NONE })
          if (n === 'set_edgecolor') return fnv(n, (a) => { s.ec = toColor(h, a[0]); redraw(); return py.NONE })
          if (n === 'set_alpha') return fnv(n, (a) => { s.alpha = h.num(a[0]); redraw(); return py.NONE })
          if (n.startsWith('set_')) return fnv(n, () => py.NONE)
          return undefined
        }
      })()))
    }
    return undefined
  }
}

const LOC_NAMES = ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center']

const entryOf = (v: Value): LegendEntry | null => {
  if (v.k !== 'obj') return null
  if (v.o instanceof ArtistObj) return v.o.legendEntry()
  if (v.o instanceof ContainerObj) return v.o.entry
  return null
}
const labelOfHandle = (v: Value): string => {
  if (v.k !== 'obj') return ''
  if (v.o instanceof ArtistObj) return v.o.labelOf() ?? ''
  if (v.o instanceof ContainerObj) return v.o.label ?? ''
  return ''
}

const buildLegend = (h: Host, candidates: Value[], args: Value[], kw0: Kw, owner: { ax: Axes | null; fig: Figure | null }): LegendObj => {
  const kw = { ...kw0 }
  const handlesKw = pop(kw, 'handles')
  const labelsKw = pop(kw, 'labels')
  let handles: Value[]
  let labels: string[]
  const strs = (v: Value) => h.iterate(v).map((x) => h.str(x))
  if (args.length === 2) {
    handles = h.iterate(args[0])
    labels = strs(args[1])
  } else if (args.length === 1) {
    labels = strs(args[0])
    handles = candidates.filter((c) => labelOfHandle(c) !== '_nolegend_').slice(0, labels.length)
    labels = labels.slice(0, handles.length)
  } else if (args.length > 2) throw h.err('TypeError', `legend only accepts two non-keyword arguments`)
  else if (handlesKw) {
    handles = h.iterate(handlesKw)
    labels = labelsKw ? strs(labelsKw) : handles.map(labelOfHandle)
  } else if (labelsKw) {
    labels = strs(labelsKw)
    handles = candidates.filter((c) => labelOfHandle(c) !== '_nolegend_').slice(0, labels.length)
  } else {
    handles = candidates.filter((c) => { const l = labelOfHandle(c); return l && !l.startsWith('_') })
    labels = handles.map(labelOfHandle)
  }
  const entries: LegendEntry[] = []
  handles.forEach((handle, i) => {
    const e = entryOf(handle)
    if (e && labels[i] !== undefined) entries.push({ ...e, label: labels[i] })
  })
  const locV = pop(kw, 'loc') ?? rcv(h, 'legend.loc')
  let loc: LegendSpec['loc']
  if (locV.k === 'str') loc = oneOf(h, locV.v, 'loc', LOC_NAMES)
  else if (locV.k === 'int') {
    if (locV.v < 0 || locV.v > 10) throw h.err('ValueError', `loc=${locV.v} is invalid. Valid loc values are 0-10`)
    loc = LOC_NAMES[locV.v]
  } else {
    const p = nums(h, locV, 'loc')
    if (p.length !== 2) throw h.err('ValueError', `loc must be string, coordinate tuple, or an integer 0-10, not ${pyRepr(locV)}`)
    loc = [p[0], p[1]]
  }
  const anchorV = pop(kw, 'bbox_to_anchor')
  const transV = pop(kw, 'bbox_transform')
  const anchor = isNone(anchorV) ? null : nums(h, anchorV!, 'bbox_to_anchor')
  const propV = pop(kw, 'prop')
  const propSize = propV && propV.k === 'dict' ? propV.d.get(py.str('size')) : undefined
  const sizeV = pop(kw, 'fontsize') ?? propSize
  const size = sizeV ? fontPt(h, sizeV) : rcFont(h, 'legend.fontsize')
  const n = (k: string, rc: string) => optNum(h, pop(kw, k)) ?? rcn(h, `legend.${rc}`)
  const fcV = pop(kw, 'facecolor')
  const ecV = pop(kw, 'edgecolor')
  const rcFc = rcv(h, 'legend.facecolor')
  const titleV = pop(kw, 'title')
  const titleFs = pop(kw, 'title_fontsize')
  const lc = pop(kw, 'labelcolor')
  const spec: LegendSpec = {
    loc, anchor, anchorCoord: transV && transV.k === 'obj' && transV.o instanceof TransformObj ? transV.o.tx : owner.ax ? 'axes' : 'figure',
    entries, ncol: h.toInt(pop(kw, 'ncol', 'ncols') ?? py.int(1)), size, title: isNone(titleV) ? null : h.str(titleV!), titleSize: titleFs ? fontPt(h, titleFs) : size,
    frame: flag(h, pop(kw, 'frameon'), rcb(h, 'legend.frameon')), alpha: n('framealpha', 'framealpha'),
    fc: fcV ? (fcV.k === 'str' && fcV.v === 'inherit' ? rcc(h, 'axes.facecolor') : toColor(h, fcV)) : rcFc.k === 'str' && rcFc.v === 'inherit' ? rcc(h, 'axes.facecolor') : toColor(h, rcFc),
    ec: toColor(h, ecV ?? rcv(h, 'legend.edgecolor')), markerscale: n('markerscale', 'markerscale'),
    borderpad: n('borderpad', 'borderpad'), labelspacing: n('labelspacing', 'labelspacing'), handlelength: n('handlelength', 'handlelength'), handletextpad: n('handletextpad', 'handletextpad'), borderaxespad: n('borderaxespad', 'borderaxespad'), columnspacing: n('columnspacing', 'columnspacing'),
    labelcolor: lc && !isNone(lc) && !(lc.k === 'str' && ['linecolor', 'markerfacecolor', 'markeredgecolor', 'mfc', 'mec'].includes(lc.v)) ? toColor(h, lc) : null, z: 5,
  }
  for (const k of ['fancybox', 'shadow', 'numpoints', 'scatterpoints', 'scatteryoffsets', 'markerfirst', 'reverse', 'mode', 'alignment', 'draggable', 'handleheight', 'handler_map', 'ncol', 'ncols']) pop(kw, k)
  const left = Object.keys(kw)
  if (left.length) throw h.err('TypeError', `Legend.__init__() got an unexpected keyword argument '${left[0]}'`)
  if (!args.length && !handlesKw && !labelsKw && !handles.length) h.print('No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.\n')
  return new LegendObj(spec, owner.ax, owner.fig)
}

class SpineObj extends PyObj {
  readonly cls = 'Spine'
  spec: SpineSpec
  ax: Axes
  constructor(spec: SpineSpec, ax: Axes) {
    super()
    this.spec = spec
    this.ax = ax
  }
  repr() { return `<matplotlib.spines.Spine object at ${addr(this)}>` }
  getAttr(name: string, h: Host): Value | undefined {
    const s = this.spec
    const done = () => { emitFig(h, this.ax.fig, this.ax); return py.NONE }
    switch (name) {
      case 'set_visible': return fnv(name, (a) => { s.visible = h.truthy(a[0]); return done() })
      case 'get_visible': return fnv(name, () => py.bool(s.visible))
      case 'set_color': case 'set_edgecolor': case 'set_ec': return fnv(name, (a) => { s.color = toColor(h, a[0]); return done() })
      case 'get_edgecolor': case 'get_ec': return fnv(name, () => rgbaValue(s.color))
      case 'set_linewidth': case 'set_lw': return fnv(name, (a) => { s.width = h.num(a[0]); return done() })
      case 'get_linewidth': case 'get_lw': return fnv(name, () => py.float(s.width))
      case 'set_linestyle': case 'set_ls': return fnv(name, (a) => { s.dash = lineStyle(h, a[0]).dash; return done() })
      case 'set_position': return fnv(name, (a) => {
        const p = a[0]
        if (p.k === 'str') {
          if (p.v === 'center') s.at = ['axes', 0.5]
          else if (p.v === 'zero') s.at = ['data', 0]
          else throw h.err('ValueError', "position should be 'center' or 2-tuple")
        } else {
          const [kind, val] = h.iterate(p)
          const k = h.str(kind)
          if (k === 'axes' || k === 'data') s.at = [k, h.num(val)]
          else if (k === 'outward') throw h.err('NotImplementedError', "spine position ('outward', n) is not drawn in this sandbox")
          else throw h.err('ValueError', 'position[0] should be one of "outward", "axes", or "data" ')
        }
        return done()
      })
      case 'set_bounds': return fnv(name, () => { throw h.err('NotImplementedError', 'spine.set_bounds is not drawn in this sandbox') })
      case 'set_zorder': case 'set_alpha': case 'set_capstyle': case 'set_smart_bounds': return fnv(name, () => py.NONE)
      case 'set': return fnv(name, (_a, kw) => {
        for (const [k, v] of Object.entries(kw)) {
          const f = this.getAttr(`set_${k}`, h)
          if (f && f.k === 'fn') f.call([v], {})
          else throw h.err('AttributeError', `Spine.set() got an unexpected keyword argument '${k}'`)
        }
        return py.NONE
      })
    }
    return undefined
  }
}

const SPINE_NAMES = ['left', 'right', 'bottom', 'top'] as const

/** ax.spines: a mapping that also takes a list or slice, whose setters apply to all */
class SpinesObj extends PyObj {
  readonly cls = 'Spines'
  ax: Axes
  map: Record<string, SpineObj>
  constructor(ax: Axes, map: Record<string, SpineObj>) {
    super()
    this.ax = ax
    this.map = map
  }
  repr() { return `<matplotlib.spines.Spines object at ${addr(this)}>` }
  len() { return 4 }
  iter(): Value[] { return SPINE_NAMES.map((n) => py.str(n)) }
  contains(v: Value) { return v.k === 'str' && v.v in this.map }
  group(list: SpineObj[]): Value {
    const obj = new (class extends PyObj {
      readonly cls = 'SpinesProxy'
      repr() { return '<matplotlib.spines.SpinesProxy object>' }
      getAttr(name: string, h: Host): Value | undefined {
        if (!name.startsWith('set')) return undefined
        return fnv(name, (a, kw) => {
          for (const s of list) {
            const f = s.getAttr(name, h)
            if (f && f.k === 'fn') f.call(a, kw)
          }
          return py.NONE
        })
      }
    })()
    return py.obj(obj)
  }
  getItem(idx: Value, h: Host): Value {
    if (idx.k === 'str') {
      const s = this.map[idx.v]
      if (!s) throw h.err('KeyError', pyRepr(idx))
      return py.obj(s)
    }
    if (idx.k === 'slice') return this.group(Object.values(this.map))
    if (idx.k === 'list' || idx.k === 'tuple') {
      return this.group(idx.items.map((k) => {
        const s = this.map[h.str(k)]
        if (!s) throw h.err('ValueError', `Spines does not contain ${pyRepr(k)}`)
        return s
      }))
    }
    throw h.err('TypeError', 'spines indices must be strings')
  }
  getAttr(name: string): Value | undefined {
    if (name in this.map) return py.obj(this.map[name])
    if (name === 'keys') return fnv(name, () => py.list(this.iter()))
    if (name === 'values') return fnv(name, () => py.list(SPINE_NAMES.map((n) => py.obj(this.map[n]))))
    if (name === 'items') return fnv(name, () => py.list(SPINE_NAMES.map((n) => py.tuple([py.str(n), py.obj(this.map[n])]))))
    return undefined
  }
}

// ---------------------------------------------------------------- Axes

class Axes extends PyObj {
  cls = 'Axes'
  fig: Figure
  x: Axis
  y: Axis
  h: Host
  ss: SubplotSpec | null
  fixed: Rect | null
  anchor: [number, number] = [0.5, 0.5]
  aspect: number | 'auto' = 'auto'
  adjustable: 'box' | 'datalim' = 'box'
  boxAspect: number | null = null
  children: ArtistObj[] = []
  containers: ContainerObj[] = []
  dynamic = new Map<ArtistObj, () => void>()
  /** runs before the limits are read (a colorbar follows its mappable's clim) */
  before: (() => void)[] = []
  lineCycle = 0
  patchCycle = 0
  cycle: string[]
  cycleRaw: Value[] | null = null
  tight = false
  useSticky = true
  titles: Record<'center' | 'left' | 'right', TextObj>
  titlePad: number
  titleY: number | null = null
  frame = true
  axisOff = false
  facecolor: string
  patchVisible = true
  visible = true
  label = ''
  grid: { color: string; width: number; dash: Dash; alpha: number; below: boolean }
  spines: Record<string, SpineObj>
  spinesObj: SpinesObj
  legendObj: LegendObj | null = null
  extraLegends: LegendObj[] = []
  current: Mapped & ArtistObj | null = null
  zorder = 0
  twinned = false
  /** what the last drawing used, for artists whose shape depends on it */
  drawn: { rect: Rect; xlim: [number, number]; ylim: [number, number] } | null = null
  constructor(fig: Figure, ss: SubplotSpec | null, fixed: Rect | null) {
    super()
    const h = fig.h
    this.h = h
    this.fig = fig
    this.ss = ss
    this.fixed = fixed
    this.x = new Axis('x', this, h)
    this.y = new Axis('y', this, h)
    this.cycle = defaultCycle(h)
    const rcCycle = stateOf(h).rc.get('axes.prop_cycle')
    if (rcCycle && rcCycle.k === 'obj' && rcCycle.o instanceof CyclerObj) this.cycleRaw = rcCycle.o.raw
    this.facecolor = rcc(h, 'axes.facecolor')
    this.titlePad = rcn(h, 'axes.titlepad')
    this.titles = {
      center: this.makeTitle(h, 0.5, 'center'), left: this.makeTitle(h, 0, 'left'), right: this.makeTitle(h, 1, 'right'),
    }
    this.grid = { color: rcc(h, 'grid.color'), width: rcn(h, 'grid.linewidth'), dash: lineStyle(h, rcv(h, 'grid.linestyle')).dash, alpha: rcn(h, 'grid.alpha'), below: rcs(h, 'axes.axisbelow') !== 'false' }
    const lw = rcn(h, 'axes.linewidth')
    const ec = rcc(h, 'axes.edgecolor')
    this.spines = Object.fromEntries(SPINE_NAMES.map((n) => [n, new SpineObj({ visible: rcb(h, `axes.spines.${n}`), color: ec, width: lw, at: null, dash: null }, this)]))
    this.spinesObj = new SpinesObj(this, this.spines)
    if (rcb(h, 'axes.grid')) {
      const which = rcs(h, 'axes.grid.which')
      const ga = rcs(h, 'axes.grid.axis')
      if (which !== 'minor') {
        if (ga !== 'y') this.x.grid = true
        if (ga !== 'x') this.y.grid = true
      }
    }
  }
  makeTitle(h: Host, x: number, ha: TextArtist['ha']): TextObj {
    const t = newText(h, x, 1, '')
    t.size = rcFont(h, 'axes.titlesize')
    t.weight = rcs(h, 'axes.titleweight')
    const c = rcv(h, 'axes.titlecolor')
    t.color = c.k === 'str' && c.v === 'auto' ? rcc(h, 'text.color') : toColor(h, c)
    t.ha = ha
    const o = new TextObj(t, String(x) === '1' ? '1.0' : String(x) === '0' ? '0.0' : String(x), '1.0')
    o.owned = true
    o.ax = this
    return o
  }
  str() {
    const [l, b, w, hh] = this.position()
    const g = (v: number) => gWith(v, 6)
    return `${this.cls}(${g(l)},${g(b)};${g(w)}x${g(hh)})`
  }
  repr() {
    const f: string[] = []
    if (this.label) f.push(`label=${pyRepr(py.str(this.label))}`)
    const t = (['left', 'center', 'right'] as const).filter((k) => this.titles[k].spec.text)
    if (t.length) f.push(`title={${t.map((k) => `'${k}': ${pyRepr(py.str(this.titles[k].spec.text))}`).join(', ')}}`)
    for (const a of this.axisList()) if (a.labelObj.spec.text) f.push(`${a.name}label=${pyRepr(py.str(a.labelObj.spec.text))}`)
    return `<${this.cls}: ${f.join(', ')}>`
  }
  axisList(): Axis[] { return [this.x, this.y] }
  axisOf(n: AxisName): Axis { return n === 'y' ? this.y : this.x }
  requestAxis(n: AxisName) { this.axisOf(n).stale = true }
  setAuto(on: boolean) {
    this.x.auto = on
    this.y.auto = on
  }
  /** what legend() can pick up, in matplotlib's order: artists first, containers last */
  legendHandles(): Value[] {
    const kids = this.children.filter((c) => !(c instanceof TextObj) && !(c instanceof ImageObj) && !(c instanceof MeshObj) && !(c instanceof ContourObj) && !(c instanceof BarObj && !c.fromHist))
    const bars = this.children.filter((c) => c instanceof BarObj && !c.fromHist)
    return [...kids, ...bars, ...this.containers].map((c) => py.obj(c))
  }
  /** the Axes with its decorations, in figure fractions: x0, y0, x1, y1 */
  tightBox(h: Host): [number, number, number, number] {
    const spec = this.resolve(h)
    const [W, H] = this.fig.spec.size
    const [wp, hp] = [W * 100, H * 100]
    const [l, b, w, hh] = spec.rect
    let x0 = l * wp
    let x1 = (l + w) * wp
    let y0 = b * hp
    let y1 = (b + hh) * hp
    if (this instanceof Axes3D) return [x0 / wp, y0 / hp, x1 / wp, y1 / hp]
    if (!spec.axisOff) {
      const xa = spec.xticks
      const ya = spec.yticks
      const label = (a: AxisSpec, horizontal: boolean) => (a.label ? a.labelPad * PT + (horizontal ? textHeightPx(a.label.text, a.label.size) : textHeightPx(a.label.text, a.label.size)) : 0)
      if (xa.visible) {
        const low = tickLabelDepth(xa, true, false)
        const high = tickLabelDepth(xa, true, true)
        const hasLow = xa.label1 && xa.labels.some((t) => t)
        const hasHigh = xa.label2 && xa.labels.some((t) => t)
        if (hasLow || xa.labelSide === 'low') y0 -= (hasLow ? low : 0) + (xa.labelSide === 'low' ? label(xa, true) : 0)
        if (hasHigh || xa.labelSide === 'high') y1 += (hasHigh ? high : 0) + (xa.labelSide === 'high' ? label(xa, true) : 0)
        const end = xa.labels.filter((t) => t)
        if ((hasLow || hasHigh) && end.length) {
          const half = (t: string) => textWidthPx(t, xa.labelSize) / 2
          const first = xa.at[xa.labels.indexOf(end[0])]
          const last = xa.at[xa.labels.lastIndexOf(end[end.length - 1])]
          const toPx = (v: number) => {
            const t = (u: number) => (spec.xscale === 'log' ? Math.log10(u) : u)
            return l * wp + ((t(v) - t(spec.xlim[0])) / (t(spec.xlim[1]) - t(spec.xlim[0]))) * w * wp
          }
          x0 = Math.min(x0, toPx(first) - half(end[0]))
          x1 = Math.max(x1, toPx(last) + half(end[end.length - 1]))
        }
      }
      if (ya.visible) {
        const low = tickLabelDepth(ya, false, false)
        const high = tickLabelDepth(ya, false, true)
        const hasLow = ya.label1 && ya.labels.some((t) => t)
        const hasHigh = ya.label2 && ya.labels.some((t) => t)
        if (hasLow || ya.labelSide === 'low') x0 -= (hasLow ? low : 0) + (ya.labelSide === 'low' ? label(ya, false) : 0)
        if (hasHigh || ya.labelSide === 'high') x1 += (hasHigh ? high : 0) + (ya.labelSide === 'high' ? label(ya, false) : 0)
      }
    }
    for (const t of spec.titles) y1 = Math.max(y1, (b + hh * (t.y ?? 1)) * hp + t.pad * PT + textHeightPx(t.text, t.size))
    return [x0 / wp, y0 / hp, x1 / wp, y1 / hp]
  }
  /** the position before any aspect is applied */
  position(): Rect { return this.fixed ?? this.ss!.rect(this.fig) }

  nextLineColor() { return this.cycle[this.lineCycle++ % this.cycle.length] }
  nextPatchColor() { return this.cycle[this.patchCycle++ % this.cycle.length] }

  add<T extends ArtistObj>(o: T): T {
    o.ax = this
    if ('label' in o.spec && o.spec.label === null) o.spec.label = `_child${this.children.length}`
    this.children.push(o)
    return o
  }
  request(x = true, y = true) {
    if (x) this.x.stale = true
    if (y) this.y.stale = true
  }
  unstale() {
    const nx = this.x.siblings.some((a) => a.stale)
    const ny = this.y.siblings.some((a) => a.stale)
    if (!nx && !ny) return
    for (const a of this.x.siblings) a.stale = false
    for (const a of this.y.siblings) a.stale = false
    this.autoscaleView(this.h, null, nx, ny)
  }
  /** _AxesBase.autoscale_view */
  autoscaleView(h: Host, tight: boolean | null, scalex = true, scaley = true) {
    if (tight !== null) this.tight = tight
    const one = (axis: Axis, which: 'x' | 'y') => {
      if (!axis.auto) return
      let stickies: number[] = []
      if (this.useSticky && axis.margin) {
        stickies = axis.siblings.flatMap((s) => s.ax.children.flatMap((c) => c.sticky[which]))
        stickies.sort((a, b) => a - b)
        if (axis.isLog()) stickies = stickies.filter((v) => v > 0)
      }
      const vals = axis.siblings.flatMap((a) => [a.lo, a.hi]).filter(Number.isFinite)
      let x0: number
      let x1: number
      if (vals.length) {
        x0 = minOf(vals)
        x1 = maxOf(vals)
      } else if (axis.lim[0] !== 0 || axis.lim[1] !== 1) return
      else {
        x0 = -Infinity
        x1 = Infinity
      }
      ;[x0, x1] = axis.nonsingular(x0, x1)
      const tol = 1e-5 * Math.abs(x1 - x0)
      let i0 = -1
      for (let i = 0; i < stickies.length; i++) if (stickies[i] < x0 + tol) i0 = i
      const lowBound = i0 !== -1 ? stickies[i0] : null
      const i1 = stickies.findIndex((v) => v >= x1 - tol)
      const highBound = i1 !== -1 ? stickies[i1] : null
      if (axis.isLog()) {
        const mp = axis.minPos()
        const m = Number.isFinite(mp) ? mp : 1e-300
        if (x0 <= 0) x0 = m
        if (x1 <= 0) x1 = m
        const lb = Math.log(axis.base)
        const t0 = Math.log(x0) / lb
        const t1 = Math.log(x1) / lb
        const delta = (t1 - t0) * axis.margin
        x0 = axis.base ** (t0 - delta)
        x1 = axis.base ** (t1 + delta)
      } else {
        const delta = (x1 - x0) * axis.margin
        x0 -= Number.isFinite(delta) ? delta : 0
        x1 += Number.isFinite(delta) ? delta : 0
      }
      if (lowBound !== null) x0 = Math.max(x0, lowBound)
      if (highBound !== null) x1 = Math.min(x1, highBound)
      if (!this.tight && !axis.isLog()) [x0, x1] = nonsingular(x0, x1, 1e-12, 1e-13)
      else if (!this.tight) [x0, x1] = logNonsingular(x0, x1, axis.base, axis.minPos())
      axis.stale = false
      axis.setBound(h, x0, x1)
    }
    if (scalex) one(this.x, 'x')
    if (scaley) one(this.y, 'y')
  }
  /** set_extent: an image's data limits and sticky edges, and the view when autoscaling */
  imageExtent(o: ArtistObj, e: number[]) {
    const h = this.h
    this.x.update([e[0], e[1]])
    this.y.update([e[2], e[3]])
    o.sticky = { x: [e[0], e[1]], y: [e[2], e[3]] }
    if (this.x.auto) this.x.setLim(h, e[0], e[1], null)
    if (this.y.auto) this.y.setLim(h, e[2], e[3], null)
  }
  gridStyle(h: Host, kw: Kw) {
    for (const [k, v] of Object.entries(kw)) {
      switch (k) {
        case 'color': case 'c': this.grid.color = toColor(h, v); break
        case 'linewidth': case 'lw': this.grid.width = h.num(v); break
        case 'linestyle': case 'ls': this.grid.dash = lineStyle(h, v).dash; break
        case 'alpha': this.grid.alpha = optNum(h, v) ?? 1; break
        default:
          if (!COSMETIC.has(k) && k !== 'zorder' && k !== 'marker' && k !== 'markersize') throw h.err('AttributeError', `Line2D.set() got an unexpected keyword argument '${k}'`)
      }
    }
  }
  setGrid(h: Host, args: Value[], kw0: Kw) {
    const kw = { ...kw0 }
    const v = args[0] ?? pop(kw, 'visible', 'b')
    const which = oneOf(h, optStr(h, args[1] ?? pop(kw, 'which')) ?? 'major', 'which', ['major', 'minor', 'both'])
    const axis = oneOf(h, optStr(h, args[2] ?? pop(kw, 'axis')) ?? 'both', 'axis', ['x', 'y', 'both'])
    const styled = Object.keys(kw).length > 0
    this.gridStyle(h, kw)
    const list = axis === 'both' ? [this.x, this.y] : [axis === 'x' ? this.x : this.y]
    for (const a of list) {
      const on = (cur: boolean) => (isNone(v) ? (styled ? true : !cur) : h.truthy(v!))
      if (which !== 'minor') a.grid = on(a.grid)
      if (which !== 'major') a.gridMinor = on(a.gridMinor)
    }
    emitFig(h, this.fig, this)
  }
  /** apply_aspect, on copies: drawing must not change what get_xlim() reports */
  geometry(): { rect: Rect; xlim: [number, number]; ylim: [number, number] } {
    const xlim = this.x.view()
    const ylim = this.y.view()
    let position = this.position()
    const [W, H] = this.fig.spec.size
    const figAspect = H / W
    const size = (a: Axis, lim: [number, number]) => {
      const t = (v: number) => (a.isLog() ? Math.log10(v) : v)
      return Math.max(Math.abs(t(lim[1]) - t(lim[0])), 1e-30)
    }
    if (this.aspect === 'auto' && this.boxAspect === null) return { rect: position, xlim, ylim }
    if (this.adjustable === 'box' && this.aspect !== 'auto') {
      const ratio = size(this.y, ylim) / size(this.x, xlim)
      return { rect: fitAspect(position, this.aspect * ratio, figAspect, this.anchor), xlim, ylim }
    }
    let rect = position
    if (this.boxAspect !== null) {
      rect = fitAspect(position, this.boxAspect, figAspect, this.anchor)
      if (this.aspect === 'auto') return { rect, xlim, ylim }
      position = rect
    }
    if (this.aspect === 'auto') return { rect, xlim, ylim }
    // adjustable='datalim': change the limits so the data keep their shape
    const t = (a: Axis, v: number) => (a.isLog() ? Math.log10(v) : v)
    const inv = (a: Axis, v: number) => (a.isLog() ? 10 ** v : v)
    const xb = [Math.min(...xlim), Math.max(...xlim)]
    const yb = [Math.min(...ylim), Math.max(...ylim)]
    const [xmin, xmax] = [t(this.x, xb[0]), t(this.x, xb[1])]
    const [ymin, ymax] = [t(this.y, yb[0]), t(this.y, yb[1])]
    const xsize = Math.max(Math.abs(xmax - xmin), 1e-30)
    const ysize = Math.max(Math.abs(ymax - ymin), 1e-30)
    const boxA = figAspect * (position[3] / position[2])
    const ratio = boxA / this.aspect
    const yExpander = (ratio * xsize) / ysize - 1
    if (Math.abs(yExpander) < 0.005) return { rect, xlim, ylim }
    const xr = 1.05 * (t(this.x, this.x.hi) - t(this.x, this.x.lo))
    const yr = 1.05 * (t(this.y, this.y.hi) - t(this.y, this.y.lo))
    const xmarg = xsize - xr
    const ymarg = ysize - yr
    const Ysize = ratio * xsize
    const Xsize = ysize / ratio
    const Xmarg = Xsize - xr
    const Ymarg = Ysize - yr
    const sharedX = this.x.siblings.length > 1
    const sharedY = this.y.siblings.length > 1
    let adjustY: boolean
    if (sharedY) adjustY = false
    else {
      const adjy = xmarg > 0 && ymarg > 0 ? (Ymarg > 0 && yExpander < 0) || (Xmarg < 0 && yExpander > 0) : yExpander > 0
      adjustY = sharedX || adjy
    }
    const put = (lim: [number, number], lo: number, hi: number): [number, number] => (lim[0] > lim[1] ? [hi, lo] : [lo, hi])
    if (adjustY) {
      const yc = 0.5 * (ymin + ymax)
      return { rect, xlim, ylim: put(ylim, inv(this.y, yc - Ysize / 2), inv(this.y, yc + Ysize / 2)) }
    }
    const xc = 0.5 * (xmin + xmax)
    return { rect, xlim: put(xlim, inv(this.x, xc - Xsize / 2), inv(this.x, xc + Xsize / 2)), ylim }
  }
  /** what the view adds for its own decoration (3D panes …) */
  extraArtists(): Artist[] { return [] }
  resolve(h: Host): AxesSpec {
    for (const fn of this.before) fn()
    const g = this.geometry()
    this.drawn = g
    for (const fn of this.dynamic.values()) fn()
    const titles = (['center', 'left', 'right'] as const).filter((k) => this.titles[k].spec.text).map((k) => {
      const t = this.titles[k].spec
      return { text: t.text, size: t.size, color: t.color, weight: t.weight, italic: t.italic, loc: k, rotation: 0, pad: this.titlePad, y: this.titleY }
    })
    const sp = (n: string) => this.spines[n].spec
    return {
      rect: g.rect, title: this.titles.center.spec.text || null, titles, xlim: g.xlim, ylim: g.ylim, xscale: this.x.scale, yscale: this.y.scale,
      xticks: this.x.resolve(h, g.xlim, this.x.tickSpace(g.rect)), yticks: this.y.resolve(h, g.ylim, this.y.tickSpace(g.rect)),
      axisOff: this.axisOff, frame: this.frame, facecolor: this.facecolor, patchVisible: this.patchVisible, visible: this.visible,
      grid: { x: this.x.grid, y: this.y.grid, xminor: this.x.gridMinor, yminor: this.y.gridMinor, ...this.grid },
      spines: { left: { ...sp('left') }, right: { ...sp('right') }, top: { ...sp('top') }, bottom: { ...sp('bottom') } },
      legends: [...this.extraLegends, ...(this.legendObj ? [this.legendObj] : [])].map((l) => l.spec),
      artists: [...this.children.map((c) => ({ ...c.spec })), ...this.extraArtists()],
    }
  }
  clear(h: Host) {
    this.children = []
    this.containers = []
    this.dynamic.clear()
    this.before = []
    this.legendObj = null
    this.extraLegends = []
    this.current = null
    this.lineCycle = 0
    this.patchCycle = 0
    this.tight = false
    this.aspect = 'auto'
    this.adjustable = 'box'
    this.frame = true
    this.axisOff = false
    this.titleY = null
    for (const k of ['center', 'left', 'right'] as const) this.titles[k].spec.text = ''
    for (const a of this.axisList()) {
      a.lim = [0, 1]
      a.auto = true
      a.stale = false
      a.lo = Infinity
      a.hi = -Infinity
      a.minpos = Infinity
      a.margin = a.name === 'z' ? 0 : rcn(h, `axes.${a.name}margin`)
      a.scale = 'linear'
      a.ticker = { loc: { k: 'auto' }, fmt: { ...SCALAR }, minorLoc: { k: 'null' }, minorFmt: { k: 'null' } }
      a.cats = null
      a.labelObj.spec.text = ''
      a.grid = false
      a.gridMinor = false
      a.style.styles = {}
      a.style.rotation = 0
    }
  }
  /** _AxesBase.relim: lines, patches and images count, collections do not */
  relim() {
    for (const a of this.axisList()) {
      a.lo = Infinity
      a.hi = -Infinity
      a.minpos = Infinity
    }
    for (const c of this.children) {
      const b = c.bounds()
      if (!b || c instanceof ScatterObj) continue
      if (c.spec.tx === 'data') this.x.update(b[0])
      if (c.spec.ty === 'data') this.y.update(b[1])
    }
  }
  getAttr(name: string, h: Host): Value | undefined {
    const m = this.method(name)
    if (m) return { k: 'fn', name, call: (args, kw) => m(this, args, { ...kw }, h) }
    const list = (pred: (c: ArtistObj) => boolean) => py.list(this.children.filter(pred).map((c) => py.obj(c)))
    switch (name) {
      case 'figure': return py.obj(this.fig)
      case 'xaxis': return py.obj(this.x)
      case 'yaxis': return py.obj(this.y)
      case 'spines': return py.obj(this.spinesObj)
      case 'title': return py.obj(this.titles.center)
      case '_left_title': return py.obj(this.titles.left)
      case '_right_title': return py.obj(this.titles.right)
      case 'transAxes': return py.obj(new TransformObj('axes', 'axes'))
      case 'transData': return py.obj(new TransformObj('data', 'data'))
      case 'lines': return list((c) => c instanceof LineObj)
      case 'patches': return py.list(this.children.flatMap((c) => (c instanceof BarObj ? c.iter() : c instanceof PatchObj ? [py.obj(c)] : [])))
      case 'collections': return list((c) => c instanceof ScatterObj || c instanceof PolyObj || c instanceof SegsObj || c instanceof MeshObj || c instanceof ContourObj)
      case 'images': return list((c) => c instanceof ImageObj)
      case 'texts': return list((c) => c instanceof TextObj)
      case 'artists': return list(() => false)
      case 'containers': return py.list([...this.children.filter((c) => c instanceof BarObj).map((c) => py.obj(c)), ...this.containers.map((c) => py.obj(c))])
      case 'legend_': return this.legendObj ? py.obj(this.legendObj) : py.NONE
      case 'dataLim': return py.obj(new Bbox(this.x.lo, this.y.lo, this.x.hi, this.y.hi))
      case 'viewLim': { const [a, b] = this.x.view(); const [c, d] = this.y.view(); return py.obj(new Bbox(a, c, b, d)) }
      case 'name': return py.str('rectilinear')
      case 'patch': return py.obj(new AxesPatch(this))
    }
    return undefined
  }
  method(name: string): AxMethod | undefined { return AXES_METHODS[name] }
}

/** ax.patch: the background rectangle */
class AxesPatch extends PyObj {
  readonly cls = 'Rectangle'
  ax: Axes
  constructor(ax: Axes) {
    super()
    this.ax = ax
  }
  repr() { return `<matplotlib.patches.Rectangle object at ${addr(this)}>` }
  getAttr(name: string, h: Host): Value | undefined {
    if (name === 'set_facecolor' || name === 'set_color' || name === 'set_fc') return fnv(name, (a) => { this.ax.facecolor = toColor(h, a[0]); emitFig(h, this.ax.fig, this.ax); return py.NONE })
    if (name === 'set_visible') return fnv(name, (a) => { this.ax.patchVisible = h.truthy(a[0]); emitFig(h, this.ax.fig, this.ax); return py.NONE })
    if (name === 'set_alpha') return fnv(name, (a) => { const c = rgbaOf(this.ax.facecolor); this.ax.facecolor = rgbHex(c[0], c[1], c[2], h.num(a[0])); return py.NONE })
    if (name === 'get_facecolor') return fnv(name, () => rgbaValue(this.ax.facecolor))
    if (name.startsWith('set_')) return fnv(name, () => py.NONE)
    return undefined
  }
}

/** what plt.subplots returns for a grid: an object array of Axes */
class AxesGrid extends PyObj {
  readonly cls = 'ndarray'
  items: Axes[]
  shape: number[]
  constructor(items: Axes[], shape: number[]) {
    super()
    this.items = items
    this.shape = shape
  }
  repr() {
    const row = (xs: Axes[]) => `[${xs.map((a) => a.repr()).join(', ')}]`
    if (this.shape.length === 1) return `array(${row(this.items)}, dtype=object)`
    const rows = Array.from({ length: this.shape[0] }, (_, r) => row(this.items.slice(r * this.shape[1], (r + 1) * this.shape[1])))
    return `array([${rows.join(',\n       ')}], dtype=object)`
  }
  len() { return this.shape[0] }
  iter(): Value[] {
    if (this.shape.length === 1) return this.items.map((a) => py.obj(a))
    return Array.from({ length: this.shape[0] }, (_, r) => py.obj(new AxesGrid(this.items.slice(r * this.shape[1], (r + 1) * this.shape[1]), [this.shape[1]])))
  }
  getItem(idx: Value, h: Host): Value {
    const ints = idx.k === 'tuple' ? idx.items : [idx]
    const at = (v: Value, n: number) => {
      const i = h.toInt(v, 'an index')
      if (i < -n || i >= n) throw h.err('IndexError', `index ${i} is out of bounds for axis with size ${n}`)
      return i < 0 ? i + n : i
    }
    if (this.shape.length === 1) {
      if (ints.length !== 1) throw h.err('IndexError', 'too many indices for array: array is 1-dimensional')
      if (ints[0].k === 'slice') {
        const n = this.items.length
        const s = isNone(ints[0].start) ? 0 : h.toInt(ints[0].start)
        const e = isNone(ints[0].stop) ? n : h.toInt(ints[0].stop)
        const part = this.items.slice(s, e)
        return py.obj(new AxesGrid(part, [part.length]))
      }
      return py.obj(this.items[at(ints[0], this.shape[0])])
    }
    if (ints.length === 1) {
      if (ints[0].k === 'slice') throw h.err('NotImplementedError', 'slices of an Axes grid: index rows and columns one at a time in this sandbox')
      return this.iter()[at(ints[0], this.shape[0])]
    }
    const [rv, cv] = ints
    const [nr, nc] = this.shape
    if (rv.k === 'slice' && !isNone(rv.start) === false && isNone(rv.stop)) {
      const c = at(cv, nc)
      return py.obj(new AxesGrid(Array.from({ length: nr }, (_, r) => this.items[r * nc + c]), [nr]))
    }
    if (cv.k === 'slice' && isNone(cv.start) && isNone(cv.stop)) {
      const r = at(rv, nr)
      return py.obj(new AxesGrid(this.items.slice(r * nc, (r + 1) * nc), [nc]))
    }
    return py.obj(this.items[at(rv, nr) * nc + at(cv, nc)])
  }
  getAttr(name: string): Value | undefined {
    if (name === 'shape') return py.tuple(this.shape.map(py.int))
    if (name === 'ndim') return py.int(this.shape.length)
    if (name === 'flat') return py.list(this.items.map((a) => py.obj(a)))
    if (name === 'flatten' || name === 'ravel') return fnv(name, () => py.obj(new AxesGrid(this.items, [this.items.length])))
    if (name === 'tolist') return fnv(name, () => (this.shape.length === 1 ? py.list(this.items.map((a) => py.obj(a))) : py.list(Array.from({ length: this.shape[0] }, (_, r) => py.list(this.items.slice(r * this.shape[1], (r + 1) * this.shape[1]).map((a) => py.obj(a)))))))
    if (name === 'size') return py.int(this.items.length)
    return undefined
  }
}

// ---------------------------------------------------------------- Figure

class Canvas extends PyObj {
  readonly cls = 'FigureCanvasAgg'
  fig: Figure
  constructor(fig: Figure) {
    super()
    this.fig = fig
  }
  repr() { return `<matplotlib.backends.backend_agg.FigureCanvasAgg object at ${addr(this)}>` }
  getAttr(name: string): Value | undefined {
    if (name === 'draw') return fnv(name, () => { this.fig.drew = true; return py.NONE })
    if (['draw_idle', 'flush_events', 'start_event_loop', 'stop_event_loop'].includes(name)) return fnv(name, () => py.NONE)
    if (name === 'get_width_height') return fnv(name, () => py.tuple([py.int(Math.round(this.fig.spec.size[0] * 100)), py.int(Math.round(this.fig.spec.size[1] * 100))]))
    if (name === 'figure') return py.obj(this.fig)
    if (name === 'mpl_connect' || name === 'manager') return name === 'manager' ? py.NONE : fnv(name, () => py.int(0))
    return undefined
  }
}

class Figure extends PyObj {
  readonly cls = 'Figure'
  spec: { num: number; size: [number, number]; facecolor: string }
  h: Host
  axes: Axes[] = []
  cur: Axes | null = null
  closed = false
  shown = false
  params: Params
  texts: TextObj[] = []
  legends: LegendObj[] = []
  sup: Record<'title' | 'xlabel' | 'ylabel', TextObj | null> = { title: null, xlabel: null, ylabel: null }
  engine: 'tight' | 'constrained' | null = null
  label = ''
  /** canvas.draw() or savefig has run: colour-mapped artists report their mapped colours */
  drew = false
  constructor(h: Host, num: number, size: [number, number], facecolor: string) {
    super()
    this.h = h
    this.spec = { num, size, facecolor }
    this.params = {
      left: rcn(h, 'figure.subplot.left'), right: rcn(h, 'figure.subplot.right'), bottom: rcn(h, 'figure.subplot.bottom'), top: rcn(h, 'figure.subplot.top'),
      wspace: rcn(h, 'figure.subplot.wspace'), hspace: rcn(h, 'figure.subplot.hspace'),
    }
    const lay = rcv(h, 'figure.autolayout')
    if (h.truthy(lay)) this.engine = 'tight'
    if (rcb(h, 'figure.constrained_layout.use')) this.engine = 'constrained'
  }
  repr() { return `<Figure size ${Math.round(this.spec.size[0] * 100)}x${Math.round(this.spec.size[1] * 100)} with ${this.axes.length} Axes>` }
  str() { return `Figure(${Math.round(this.spec.size[0] * 100)}x${Math.round(this.spec.size[1] * 100)})` }
  display(): Display { return { type: 'figure', fig: snapshot(this) } }
  removeText(o: ArtistObj) {
    this.texts = this.texts.filter((t) => t !== o)
    for (const k of ['title', 'xlabel', 'ylabel'] as const) if (this.sup[k] === o) this.sup[k] = null
  }
  /** a grid position, reused by plt.subplot when one with the same spec exists */
  gridAxes(h: Host, args: Value[], kw: Kw, reuse: boolean): Axes {
    const projection = pop(kw, 'projection')
    const polar = flag(h, pop(kw, 'polar'), false)
    const sharex = pop(kw, 'sharex')
    const sharey = pop(kw, 'sharey')
    const label = optStr(h, pop(kw, 'label')) ?? ''
    const anyKw = Object.keys(kw).length > 0 || projection || sharex || sharey
    const ss = subplotSpecOf(h, this, args)
    if (reuse) {
      const found = this.axes.find((a) => a.ss && a.ss.same(ss))
      if (found && !anyKw) {
        this.cur = found
        return found
      }
    }
    const ax = makeAxes(h, this, ss, null, projection, polar)
    ax.label = label
    if (sharex && sharex.k === 'obj' && sharex.o instanceof Axes) shareAxis(ax.x, sharex.o.x)
    if (sharey && sharey.k === 'obj' && sharey.o instanceof Axes) shareAxis(ax.y, sharey.o.y)
    setKw(ax, kw, h)
    this.cur = ax
    return ax
  }
  gca(): Axes {
    if (this.cur) return this.cur
    return this.gridAxes(this.h, [], {}, false)
  }
  clear() {
    this.axes = []
    this.cur = null
    this.texts = []
    this.legends = []
    this.sup = { title: null, xlabel: null, ylabel: null }
  }
  supText(h: Host, which: 'title' | 'xlabel' | 'ylabel', args: Value[], kw0: Kw): Value {
    const kw = { ...kw0 }
    const s = h.str(args[0] ?? pop(kw, 't', which) ?? py.str(''))
    const rcTitle = which === 'title'
    const sx = pop(kw, 'x')
    const sy = pop(kw, 'y')
    const x = which === 'ylabel' ? optNum(h, sx) ?? 0.02 : optNum(h, sx) ?? 0.5
    const y = which === 'title' ? optNum(h, sy) ?? 0.98 : which === 'xlabel' ? optNum(h, sy) ?? 0.01 : optNum(h, sy) ?? 0.5
    const t = newText(h, x, y, s)
    t.tx = 'figure'
    t.ty = 'figure'
    t.ha = which === 'ylabel' ? 'left' : 'center'
    t.va = which === 'title' ? 'top' : which === 'xlabel' ? 'bottom' : 'center'
    t.rotation = which === 'ylabel' ? 90 : 0
    t.size = rcTitle ? rcFont(h, 'figure.titlesize') : rcFont(h, 'figure.labelsize')
    t.weight = rcTitle ? rcs(h, 'figure.titleweight') : rcs(h, 'figure.labelweight')
    const o = new TextObj(t, String(x), String(y))
    o.fig = this
    if (kw.fontdict) o.applyKw(kwOfDict(h, pop(kw, 'fontdict')), h)
    o.applyKw(kw, h)
    this.sup[which] = o
    emitFig(h, this, null)
    return py.obj(o)
  }
  getAttr(name: string, h: Host): Value | undefined {
    const size = this.spec.size
    switch (name) {
      case 'add_subplot': return fnv(name, (args, kw) => py.obj(this.gridAxes(h, args, { ...kw }, false)))
      case 'add_axes': return fnv(name, (args, kw0) => {
        const kw = { ...kw0 }
        const rectV = args[0] ?? pop(kw, 'rect')
        if (!rectV) throw h.err('TypeError', 'add_axes() missing 1 required positional argument: \'rect\'')
        if (rectV.k === 'obj' && rectV.o instanceof Axes) {
          this.cur = rectV.o
          return rectV
        }
        const r = nums(h, rectV, 'rect')
        if (r.length !== 4) throw h.err('ValueError', `rect must be a sequence of 4 numbers, not ${r.length}`)
        const ax = makeAxes(h, this, null, r as Rect, pop(kw, 'projection'), flag(h, pop(kw, 'polar'), false))
        const sx = pop(kw, 'sharex')
        const sy = pop(kw, 'sharey')
        if (sx && sx.k === 'obj' && sx.o instanceof Axes) shareAxis(ax.x, sx.o.x)
        if (sy && sy.k === 'obj' && sy.o instanceof Axes) shareAxis(ax.y, sy.o.y)
        ax.label = optStr(h, pop(kw, 'label')) ?? ''
        setKw(ax, kw, h)
        this.cur = ax
        emitFig(h, this, null)
        return py.obj(ax)
      })
      case 'subplots': return fnv(name, (args, kw) => (subplotsOn(h, this, args, { ...kw }, false) as { items: Value[] }).items[1])
      case 'subplot_mosaic': return fnv(name, (args, kw0) => { const kw = { ...kw0 }; return mosaicOn(h, this, args[0] ?? pop(kw, 'mosaic')!, kw) })
      case 'add_gridspec': case 'gridspec': return fnv(name, (args, kw) => py.obj(gridSpecOf(h, this, args, kw)))
      case 'suptitle': return fnv(name, (args, kw) => this.supText(h, 'title', args, kw))
      case 'supxlabel': return fnv(name, (args, kw) => this.supText(h, 'xlabel', args, kw))
      case 'supylabel': return fnv(name, (args, kw) => this.supText(h, 'ylabel', args, kw))
      case 'get_suptitle': return fnv(name, () => py.str(this.sup.title?.spec.text ?? ''))
      case 'get_supxlabel': return fnv(name, () => py.str(this.sup.xlabel?.spec.text ?? ''))
      case 'get_supylabel': return fnv(name, () => py.str(this.sup.ylabel?.spec.text ?? ''))
      case 'axes': return py.list(this.axes.map((a) => py.obj(a)))
      case 'get_axes': return fnv(name, () => py.list(this.axes.map((a) => py.obj(a))))
      case 'gca': return fnv(name, () => py.obj(this.gca()))
      case 'sca': return fnv(name, (args) => { const a = args[0]; if (a.k === 'obj' && a.o instanceof Axes) this.cur = a.o; return a })
      case 'delaxes': return fnv(name, (args) => {
        const a = args[0]
        if (a.k === 'obj' && a.o instanceof Axes) {
          this.axes = this.axes.filter((x) => x !== a.o)
          if (this.cur === a.o) this.cur = this.axes[this.axes.length - 1] ?? null
        }
        emitFig(h, this, null)
        return py.NONE
      })
      case 'clear': case 'clf': return fnv(name, () => { this.clear(); emitFig(h, this, null); return py.obj(this) })
      case 'tight_layout': return fnv(name, (_a, kw) => { applyTight(h, this, kw); emitFig(h, this, null); return py.NONE })
      case 'set_tight_layout': return fnv(name, (a) => { this.engine = a[0] && h.truthy(a[0]) ? 'tight' : null; return py.NONE })
      case 'get_tight_layout': return fnv(name, () => py.bool(this.engine === 'tight'))
      case 'set_constrained_layout': return fnv(name, (a) => { this.engine = a[0] && h.truthy(a[0]) ? 'constrained' : null; return py.NONE })
      case 'get_constrained_layout': return fnv(name, () => py.bool(this.engine === 'constrained'))
      case 'set_layout_engine': return fnv(name, (a, kw) => { this.engine = layoutEngine(h, a[0] ?? kw.layout); return py.NONE })
      case 'subplots_adjust': return fnv(name, (args, kw) => {
        const keys = ['left', 'bottom', 'right', 'top', 'wspace', 'hspace'] as const
        keys.forEach((k, i) => {
          const v = args[i] ?? kw[k]
          if (!isNone(v)) this.params[k] = h.num(v!)
        })
        if (this.params.left >= this.params.right) throw h.err('ValueError', 'left cannot be >= right')
        if (this.params.bottom >= this.params.top) throw h.err('ValueError', 'bottom cannot be >= top')
        emitFig(h, this, null)
        return py.NONE
      })
      case 'subplotpars': return dictLike(this.params)
      case 'set_size_inches': return fnv(name, (args, kw) => {
        const a = args[0] ?? kw.w
        const s = args.length >= 2 ? [h.num(args[0]), h.num(args[1])] : kw.h ? [h.num(a), h.num(kw.h)] : nums(h, a, 'size')
        if (s.length !== 2 || s.some((v) => !(v > 0))) throw h.err('ValueError', `figure size must be positive finite not ${pyRepr(py.tuple(s.map(py.float)))}`)
        this.spec.size = [s[0], s[1]]
        emitFig(h, this, null)
        return py.NONE
      })
      case 'get_size_inches': return fnv(name, () => floats([...size]))
      case 'set_figwidth': return fnv(name, (a) => { this.spec.size = [h.num(a[0]), size[1]]; emitFig(h, this, null); return py.NONE })
      case 'set_figheight': return fnv(name, (a) => { this.spec.size = [size[0], h.num(a[0])]; emitFig(h, this, null); return py.NONE })
      case 'get_figwidth': return fnv(name, () => py.float(size[0]))
      case 'get_figheight': return fnv(name, () => py.float(size[1]))
      case 'dpi': return py.float(rcn(h, 'figure.dpi'))
      case 'get_dpi': return fnv(name, () => py.float(rcn(h, 'figure.dpi')))
      case 'set_dpi': case 'show': case 'draw': case 'draw_without_rendering': case 'align_labels': case 'align_xlabels': case 'align_ylabels': case 'autofmt_xdate': case 'set_frameon': case 'set_edgecolor': case 'set_linewidth': return fnv(name, () => py.NONE)
      case 'savefig': return fnv(name, (args, kw) => {
        const f = args[0] ?? kw.fname
        if (!f) throw h.err('TypeError', "savefig() missing 1 required positional argument: 'fname'")
        this.drew = true
        h.print(`(sandbox) figure would be saved as ${h.str(f)}\n`)
        return py.NONE
      })
      case 'colorbar': return fnv(name, (args, kw0) => {
        const kw = { ...kw0 }
        const m = args[0] ?? pop(kw, 'mappable')
        if (!m) throw h.err('TypeError', "colorbar() missing 1 required positional argument: 'mappable'")
        return makeColorbar(h, this, m, args[1] ?? pop(kw, 'cax'), args[2] ?? pop(kw, 'ax'), kw)
      })
      case 'legend': return fnv(name, (args, kw) => {
        const cands = this.axes.flatMap((a) => a.legendHandles())
        const l = buildLegend(h, cands, args, kw, { ax: null, fig: this })
        this.legends.push(l)
        emitFig(h, this, null)
        return py.obj(l)
      })
      case 'text': return fnv(name, (args, kw0) => {
        const kw = { ...kw0 }
        if (args.length < 3) throw h.err('TypeError', "text() missing required positional arguments: 'x', 'y' and 's'")
        const t = newText(h, h.num(args[0]), h.num(args[1]), h.str(args[2]))
        t.tx = 'figure'
        t.ty = 'figure'
        const o = new TextObj(t, h.str(args[0]), h.str(args[1]))
        o.fig = this
        o.applyKw(kw, h)
        this.texts.push(o)
        emitFig(h, this, null)
        return py.obj(o)
      })
      case 'add_artist': return fnv(name, (args) => {
        const a = args[0]
        if (a.k === 'obj' && a.o instanceof TextObj) { a.o.fig = this; a.o.spec.tx = 'figure'; a.o.spec.ty = 'figure'; this.texts.push(a.o) } else if (a.k === 'obj' && a.o instanceof LegendObj) this.legends.push(a.o)
        else throw h.err('NotImplementedError', 'fig.add_artist: only Text and Legend are drawn in this sandbox')
        emitFig(h, this, null)
        return a
      })
      case 'set_facecolor': return fnv(name, (a) => { this.spec.facecolor = toColor(h, a[0]); emitFig(h, this, null); return py.NONE })
      case 'get_facecolor': return fnv(name, () => rgbaValue(this.spec.facecolor))
      case 'patch': return py.obj(new (class extends PyObj {
        readonly cls = 'Rectangle'
        fig: Figure
        constructor(f: Figure) { super(); this.fig = f }
        repr() { return '<matplotlib.patches.Rectangle object>' }
        getAttr(n: string): Value | undefined {
          if (n === 'set_facecolor' || n === 'set_color') return fnv(n, (a) => { this.fig.spec.facecolor = toColor(h, a[0]); return py.NONE })
          if (n.startsWith('set_')) return fnv(n, () => py.NONE)
          return undefined
        }
      })(this))
      case 'canvas': return py.obj(new Canvas(this))
      case 'number': return py.int(this.spec.num)
      case 'transFigure': return py.obj(new TransformObj('figure', 'figure'))
      case 'get_label': return fnv(name, () => py.str(this.label))
      case 'set_label': return fnv(name, (a) => { this.label = h.str(a[0]); return py.NONE })
      case 'set': return fnv(name, (_a, kw) => {
        for (const [k, v] of Object.entries(kw)) {
          const f = this.getAttr(`set_${k}`, h)
          if (f && f.k === 'fn') f.call([v], {})
          else throw h.err('AttributeError', `Figure.set() got an unexpected keyword argument '${k}'`)
        }
        return py.NONE
      })
    }
    return undefined
  }
}

const dictLike = (p: Params): Value => py.obj(new (class extends PyObj {
  readonly cls = 'SubplotParams'
  repr() { return '<matplotlib.gridspec.SubplotParams object>' }
  getAttr(n: string): Value | undefined { return n in p ? py.float(p[n as keyof Params]) : undefined }
})())

const layoutEngine = (h: Host, v: Value | undefined): Figure['engine'] => {
  const s = optStr(h, v)
  if (s === null || s === 'none') return null
  if (s === 'tight') return 'tight'
  if (s === 'constrained' || s === 'compressed') return 'constrained'
  throw h.err('ValueError', `Invalid value for 'layout': ${pyRepr(v!)}`)
}

const gridSpecOf = (h: Host, fig: Figure | null, args: Value[], kw: Kw): GridSpec => {
  const r = h.toInt(args[0] ?? kw.nrows ?? py.int(1))
  const c = h.toInt(args[1] ?? kw.ncols ?? py.int(1))
  if (r <= 0) throw h.err('ValueError', `Number of rows must be a positive integer, not ${r}`)
  if (c <= 0) throw h.err('ValueError', `Number of columns must be a positive integer, not ${c}`)
  const g = new GridSpec(fig, r, c, kw.width_ratios && !isNone(kw.width_ratios) ? nums(h, kw.width_ratios, 'width_ratios') : undefined, kw.height_ratios && !isNone(kw.height_ratios) ? nums(h, kw.height_ratios, 'height_ratios') : undefined)
  for (const k of ['left', 'right', 'bottom', 'top', 'wspace', 'hspace'] as const) if (kw[k] && !isNone(kw[k])) g.own[k] = h.num(kw[k])
  return g
}

/** SubplotSpec._from_subplot_args */
const subplotSpecOf = (h: Host, fig: Figure, args: Value[]): SubplotSpec => {
  const a = args.length ? args : [py.int(111)]
  if (a.length === 1) {
    const v = a[0]
    if (v.k === 'obj' && v.o instanceof SubplotSpec) return v.o
    const n = h.toInt(v, 'an integer')
    if (n < 111 || n > 999) throw h.err('ValueError', `Single argument to subplot must be a three-digit integer, not ${n}`)
    const r = Math.floor(n / 100)
    const c = Math.floor(n / 10) % 10
    const i = n % 10
    if (i < 1 || i > r * c) throw h.err('ValueError', `num must be an integer with 1 <= num <= ${r * c}, not ${i}`)
    return cellSpec(fig, r, c, i, i)
  }
  if (a.length !== 3) throw h.err('TypeError', `subplot() takes 1 or 3 positional arguments but ${a.length} were given`)
  const r = h.toInt(a[0])
  const c = h.toInt(a[1])
  if (r < 1) throw h.err('ValueError', `Number of rows must be a positive integer, not ${r}`)
  if (c < 1) throw h.err('ValueError', `Number of columns must be a positive integer, not ${c}`)
  const nv = a[2]
  if (nv.k === 'tuple' && nv.items.length === 2) {
    const [i, j] = nv.items.map((x) => h.toInt(x))
    if (i < 1 || j > r * c) throw h.err('ValueError', `num must be 1 <= num <= ${r * c}, not ${pyRepr(nv)}`)
    return cellSpec(fig, r, c, i, j)
  }
  const i = h.toInt(nv)
  if (i < 1 || i > r * c) throw h.err('ValueError', `num must be an integer with 1 <= num <= ${r * c}, not ${i}`)
  return cellSpec(fig, r, c, i, i)
}
const cellSpec = (fig: Figure, r: number, c: number, i: number, j: number): SubplotSpec => {
  const g = new GridSpec(fig, r, c)
  const a = i - 1
  const b = j - 1
  return new SubplotSpec(g, Math.floor(a / c), Math.floor(b / c) + 1, Math.min(a % c, b % c), Math.max(a % c, b % c) + 1)
}

const shareAxis = (a: Axis, b: Axis) => {
  const all = [...new Set([...b.siblings, ...a.siblings])]
  for (const s of all) s.siblings = all
  a.ticker = b.ticker
  a.cats = b.cats
  a.scale = b.scale
  a.base = b.base
  a.lim = b.view()
  a.auto = b.auto
}

/** Axes.set(**kw) and Axes constructor keywords */
const setKw = (ax: Axes, kw: Kw, h: Host) => {
  for (const [k, v] of Object.entries(kw)) {
    const m = ax.method(`set_${k}`)
    if (m) m(ax, [v], {}, h)
    else if (!COSMETIC.has(k)) throw h.err('AttributeError', `${ax.cls}.set() got an unexpected keyword argument '${k}'`)
  }
}

// ---------------------------------------------------------------- pyplot state

const newFigure = (h: Host, size?: [number, number], num?: number, facecolor?: string): Figure => {
  const s = stateOf(h)
  const open = s.figs.filter((f) => !f.closed)
  const n = num ?? (open.length ? Math.max(...open.map((f) => f.spec.num)) + 1 : 1)
  const dflt = nums(h, rcv(h, 'figure.figsize'), 'figure.figsize') as [number, number]
  const f = new Figure(h, n, size ?? dflt, facecolor ?? rcc(h, 'figure.facecolor'))
  s.figs.push(f)
  s.cur = f
  return f
}

const gcf = (h: Host): Figure => {
  const s = stateOf(h)
  return s.cur && !s.cur.closed && !s.cur.shown ? s.cur : newFigure(h)
}

const gca = (h: Host): Axes => gcf(h).gca()

const axesSpec = (f: Figure, h: Host): AxesSpec[] => f.axes.map((a) => a.resolve(h))

/** run fn as a look at the figure: pending autoscales stay pending, as they do in matplotlib until it really draws */
const untouched = <T>(f: Figure, fn: () => T): T => {
  const saved = f.axes.flatMap((a) => a.axisList().map((x) => ({ x, lim: x.lim, stale: x.stale })))
  const tight = f.axes.map((a) => a.tight)
  const params = { ...f.params }
  try {
    return fn()
  } finally {
    for (const s of saved) {
      s.x.lim = s.lim
      s.x.stale = s.stale
    }
    f.axes.forEach((a, i) => { a.tight = tight[i] })
    f.params = params
  }
}

const snapshot = (f: Figure): FigureSpec => untouched(f, () => {
  const h = f.h
  if (f.engine) applyTight(h, f, {})
  const texts = [...(f.sup.title ? [f.sup.title] : []), ...(f.sup.xlabel ? [f.sup.xlabel] : []), ...(f.sup.ylabel ? [f.sup.ylabel] : []), ...f.texts]
  return {
    num: f.spec.num, size: f.spec.size, facecolor: f.spec.facecolor, suptitle: f.sup.title?.spec.text ?? null, texts: texts.map((t) => ({ ...t.spec })),
    legends: f.legends.map((l) => l.spec), axes: axesSpec(f, h),
  }
})

/** record the figure after a call; focus = the axes and the artists the call added */
const emitFig = (h: Host, f: Figure, ax: Axes | null, artist: number | null = null, last?: number) => {
  if (!h.tracing) return
  h.emit({ type: 'figure', fig: snapshot(f), focus: ax ? { ax: f.axes.indexOf(ax), artist, ...(last !== undefined && last !== artist ? { last } : {}) } : null })
}

const note = (h: Host, tone: 'info' | 'warn', key: string, params: Record<string, string | number> = {}) => {
  if (h.tracing) h.emit({ type: 'note', tone, key, params })
}

/** tight_layout: _auto_adjust_subplotpars with estimated text extents */
const applyTight = (h: Host, f: Figure, kw: Kw) => {
  const size = rcn(h, 'font.size')
  const pad = (optNum(h, kw.pad) ?? 1.08) * size / 72
  const hp = optNum(h, kw.h_pad)
  const wp = optNum(h, kw.w_pad)
  const vpad = hp !== null ? (hp * size) / 72 : pad
  const hpad = wp !== null ? (wp * size) / 72 : pad
  const rectV = isNone(kw.rect) ? null : nums(h, kw.rect!, 'rect')
  const [W, H] = f.spec.size
  const list = f.axes.filter((a) => a.ss && a.visible)
  if (!list.length) return
  const groups = new Map<string, { axes: Axes[]; ss: SubplotSpec }>()
  for (const a of list) {
    const ss = a.ss!.topmost()
    const key = `${ss.gs.nrows}_${ss.gs.ncols}_${ss.r0}_${ss.r1}_${ss.c0}_${ss.c1}`
    const g = groups.get(key)
    if (g) g.axes.push(a)
    else groups.set(key, { axes: [a], ss })
  }
  const all = [...groups.values()]
  const rows = Math.max(...all.map((g) => g.ss.gs.nrows))
  const cols = Math.max(...all.map((g) => g.ss.gs.ncols))
  if (all.some((g) => rows % g.ss.gs.nrows || cols % g.ss.gs.ncols)) return
  const compute = (rect: (number | null)[] | null): Partial<Params> | null => {
    const vs = Array.from({ length: rows + 1 }, () => new Array<number>(cols).fill(0))
    const hs = Array.from({ length: rows }, () => new Array<number>(cols + 1).fill(0))
    for (const { axes, ss } of all) {
      const box = ss.rect(f)
      const dr = rows / ss.gs.nrows
      const dc = cols / ss.gs.ncols
      const [r0, r1, c0, c1] = [ss.r0 * dr, ss.r1 * dr, ss.c0 * dc, ss.c1 * dc]
      let x0 = Infinity
      let x1 = -Infinity
      let y0 = Infinity
      let y1 = -Infinity
      for (const a of axes) {
        const t = a.tightBox(h)
        x0 = Math.min(x0, t[0])
        y0 = Math.min(y0, t[1])
        x1 = Math.max(x1, t[2])
        y1 = Math.max(y1, t[3])
      }
      for (let r = r0; r < r1; r++) {
        hs[r][c0] += box[0] - x0
        hs[r][c1] += x1 - (box[0] + box[2])
      }
      for (let c = c0; c < c1; c++) {
        vs[r0][c] += y1 - (box[1] + box[3])
        vs[r1][c] += box[1] - y0
      }
    }
    const textH = (o: TextObj | null) => (o ? textHeightPx(o.spec.text, o.spec.size) / (H * 100) : 0)
    const textW = (o: TextObj | null) => (o ? textHeightPx(o.spec.text, o.spec.size) / (W * 100) : 0)
    let mL = rect?.[0] ?? null
    let mB = rect?.[1] ?? null
    let mR = rect?.[2] ? 1 - rect[2] : null
    let mT = rect?.[3] ? 1 - rect[3] : null
    if (!mL) {
      mL = Math.max(Math.max(...hs.map((r) => r[0])), 0) + pad / W
      if (f.sup.ylabel) mL += textW(f.sup.ylabel) + pad / W
    }
    if (!mR) mR = Math.max(Math.max(...hs.map((r) => r[cols])), 0) + pad / W
    if (!mT) {
      mT = Math.max(Math.max(...vs[0]), 0) + pad / H
      if (f.sup.title) mT += textH(f.sup.title) + pad / H
    }
    if (!mB) {
      mB = Math.max(Math.max(...vs[rows]), 0) + pad / H
      if (f.sup.xlabel) mB += textH(f.sup.xlabel) + pad / H
    }
    if (mL + mR >= 1 || mB + mT >= 1) return null
    const out: Partial<Params> = { left: mL, right: 1 - mR, bottom: mB, top: 1 - mT }
    if (cols > 1) {
      const hspace = Math.max(...hs.flatMap((r) => r.slice(1, -1))) + hpad / W
      const axW = (1 - mR - mL - hspace * (cols - 1)) / cols
      if (axW < 0) return null
      out.wspace = hspace / axW
    }
    if (rows > 1) {
      const vspace = Math.max(...vs.slice(1, -1).flat()) + vpad / H
      const axH = (1 - mT - mB - vspace * (rows - 1)) / rows
      if (axH < 0) return null
      out.hspace = vspace / axH
    }
    return out
  }
  let out = compute(null)
  if (out && rectV) {
    const [l, b, r, t] = rectV
    out = compute([l + out.left!, b + out.bottom!, r - (1 - out.right!), t - (1 - out.top!)])
  }
  if (out) Object.assign(f.params, out)
}

// ---------------------------------------------------------------- argument helpers

type AxMethod = (ax: Axes, args: Value[], kw: Kw, h: Host) => Value

/** the emit at the end of a plotting call, focused on what it added */
const added = (h: Host, ax: Axes, first: number) => emitFig(h, ax.fig, ax, first, ax.children.length - 1)

/** an axis's numbers from an argument: strings become categories, as matplotlib's category units do */
const axisNums = (h: Host, axis: Axis, v: Value, what: string): number[] => {
  const asStrs = (items: Value[]) => {
    axis.cats ??= new Map()
    if (axis.ticker.loc.k === 'auto') {
      axis.ticker.loc = { k: 'cats' }
      axis.ticker.fmt = { k: 'cats' }
    }
    return items.map((s) => {
      const k = h.str(s)
      if (!axis.cats!.has(k)) axis.cats!.set(k, axis.cats!.size)
      return axis.cats!.get(k)!
    })
  }
  if (v.k === 'str') return asStrs([v])
  if (isStrList(v)) return asStrs((v as { items: Value[] }).items)
  if (v.k === 'obj' && !v.o.toArray) {
    const items = h.iterate(v)
    if (items.length && items.every((x) => x.k === 'str')) return asStrs(items)
  }
  try {
    return toArray(v, 'float64').values()
  } catch {
    try {
      const items = h.iterate(v)
      if (items.length && items.every((x) => x.k === 'str')) return asStrs(items)
    } catch { /* falls through to the error below */ }
    throw h.err('TypeError', `${what}: expected numbers, got '${v.k === 'obj' ? v.o.cls : v.k}'`)
  }
}

const parseFmt = (h: Host, fmt: string): { ls: string | null; marker: string | null; color: Value | null } => {
  if (fmt !== '0' && fmt !== '1') {
    const whole = parseColor(fmt)
    if (whole) return { ls: null, marker: null, color: rgbaValue(whole) }
  }
  const bad = (why: string) => h.err('ValueError', `${pyRepr(py.str(fmt))} is not a valid format string (${why})`)
  let ls: string | null = null
  let marker: string | null = null
  let color: Value | null = null
  let i = 0
  while (i < fmt.length) {
    const two = fmt.slice(i, i + 2)
    const c = fmt[i]
    if (two === '--' || two === '-.') {
      if (ls !== null) throw bad('two linestyle symbols')
      ls = two
      i += 2
    } else if (c === '-' || c === ':') {
      if (ls !== null) throw bad('two linestyle symbols')
      ls = c
      i += 1
    } else if (MARKERS.has(c)) {
      if (marker !== null) throw bad('two marker symbols')
      marker = c
      i += 1
    } else if ('bgrcmykw'.includes(c)) {
      if (color !== null) throw bad('two color symbols')
      color = py.str(c)
      i += 1
    } else if (c === 'C') {
      const m = /^C\d+/.exec(fmt.slice(i))
      if (!m) throw bad("'C' must be followed by a number")
      color = rgbaValue(parseColor(m[0])!)
      i += m[0].length
    } else throw bad(`unrecognized character ${pyRepr(py.str(c))}`)
  }
  if (ls === null && marker === null) return { ls: null, marker: null, color }
  return { ls: ls ?? 'None', marker: marker ?? 'None', color }
}

// ---------------------------------------------------------------- lines

const plotLines = (ax: Axes, args0: Value[], kw0: Kw, h: Host): LineObj[] => {
  const kw = { ...kw0 }
  const scalex = flag(h, pop(kw, 'scalex'), true)
  const scaley = flag(h, pop(kw, 'scaley'), true)
  if (!isNone(pop(kw, 'data'))) throw h.err('NotImplementedError', "plot(data=...) is not available in this sandbox: pass the columns themselves")
  if ('c' in kw && 'color' in kw) throw h.err('TypeError', "Got both 'color' and 'c', which are aliases of one another")
  const out: LineObj[] = []
  const args = [...args0]
  while (args.length) {
    const group = args.splice(0, 2)
    if (args.length && args[0].k === 'str') group.push(args.shift()!)
    let fmt: string | null = null
    if (group.length > 1 && group[group.length - 1].k === 'str') fmt = (group.pop() as { v: string }).v
    let xv: Value | null = group.length === 2 ? group[0] : null
    const yv = group[group.length - 1]
    if (!xv && yv.k === 'obj' && yv.o.cls === 'Series') {
      try {
        const idx = yv.o.getAttr?.('index', h)
        if (idx) xv = py.arr(toArray(idx))
      } catch { /* a non-numeric index plots against positions */ }
    }
    const isCat = (v: Value) => v.k === 'str' || isStrList(v)
    const yCat = isCat(yv)
    const yArr = yCat ? null : toArray(yv)
    const yShape = yCat ? [h.iterate(yv).length] : yArr!.ndim === 0 ? [1] : yArr!.shape
    const n = yShape[0]
    const xCat = xv !== null && isCat(xv)
    const xArr = xv && !xCat ? toArray(xv) : null
    const xShape = xv ? (xCat ? [h.iterate(xv).length] : xArr!.ndim === 0 ? [1] : xArr!.shape) : [n]
    const shapeStr = (s: number[]) => `(${s.join(', ')}${s.length === 1 ? ',' : ''})`
    if (xShape[0] !== n) throw h.err('ValueError', `x and y must have same first dimension, but have shapes ${shapeStr(xShape)} and ${shapeStr(yShape)}`)
    if (xShape.length > 2 || yShape.length > 2) throw h.err('ValueError', `x and y can be no greater than 2D, but have shapes ${shapeStr(xShape)} and ${shapeStr(yShape)}`)
    const ncx = xShape.length === 2 ? xShape[1] : 1
    const ncy = yShape.length === 2 ? yShape[1] : 1
    if (ncx > 1 && ncy > 1 && ncx !== ncy) throw h.err('ValueError', `x has ${ncx} columns but y has ${ncy} columns`)
    const cols = Math.max(ncx, ncy)
    const xs = xv ? (xCat ? axisNums(h, ax.x, xv, 'x') : xArr!.values()) : Array.from({ length: n }, (_, i) => i)
    const ys = yCat ? axisNums(h, ax.y, yv, 'y') : yArr!.values()
    const f = fmt !== null ? parseFmt(h, fmt) : { ls: null, marker: null, color: null }
    const kwc0 = kw.color ?? kw.c
    const kwc = isNone(kwc0) ? undefined : kwc0
    let labels: (string | null)[] = new Array<string | null>(cols).fill(null)
    const lv = kw.label
    if (lv && !isNone(lv)) {
      if (lv.k === 'list' || lv.k === 'tuple') {
        if (lv.items.length !== cols) throw h.err('ValueError', `label must be scalar or have the same length as the input data, but found ${lv.items.length} for ${cols} datasets.`)
        labels = lv.items.map((x) => h.str(x))
      } else labels = new Array<string>(cols).fill(h.str(lv))
    }
    for (let c = 0; c < cols; c++) {
      const colX = ncx === 1 ? xs : Array.from({ length: n }, (_, r) => xs[r * ncx + c])
      const colY = ncy === 1 ? ys : Array.from({ length: n }, (_, r) => ys[r * ncy + c])
      const cycled = kwc || f.color ? null : ax.lineCycle
      const color = kwc ? toColor(h, kwc) : f.color ? toColor(h, f.color) : ax.nextLineColor()
      const o = new LineObj(newLine(h, colX, colY, color))
      if (xArr) o.dtypes[0] = xArr.dtype
      else o.dtypes[0] = 'int64'
      if (yArr) o.dtypes[1] = yArr.dtype
      o.raw = kwc ?? f.color ?? (cycled !== null && ax.cycleRaw ? ax.cycleRaw[cycled % ax.cycleRaw.length] : null)
      const rcLs = rcv(h, 'lines.linestyle')
      LINE_PROPS.linestyle.s!(o, [f.ls !== null ? py.str(f.ls) : rcLs], h, {})
      if (f.marker !== null) o.spec.marker = markerOf(h, py.str(f.marker))
      else o.spec.marker = markerOf(h, rcv(h, 'lines.marker'))
      const rest = { ...kw }
      pop(rest, 'color', 'c', 'label')
      o.applyKw(rest, h)
      if (labels[c] !== null) o.spec.label = labels[c]
      ax.add(o)
      if (o.spec.tx === 'data') ax.x.update(o.spec.x)
      if (o.spec.ty === 'data') ax.y.update(o.spec.y)
      out.push(o)
    }
  }
  ax.request(scalex, scaley)
  return out
}

const plot: AxMethod = (ax, args, kw, h) => {
  const first = ax.children.length
  const lines = plotLines(ax, args, kw, h)
  added(h, ax, first)
  return py.list(lines.map((l) => py.obj(l)))
}

const step: AxMethod = (ax, args, kw, h) => {
  const where = oneOf(h, optStr(h, pop(kw, 'where')) ?? 'pre', 'where', ['pre', 'post', 'mid'])
  return plot(ax, args, { ...kw, drawstyle: py.str(`steps-${where}`) }, h)
}

const scaleThen = (xs: boolean, ys: boolean): AxMethod => (ax, args, kw, h) => {
  const pick = (name: 'x' | 'y') => {
    const d: Kw = {}
    for (const k of ['base', 'subs', 'nonpositive', `base${name}`, `subs${name}`, `nonpos${name}`]) if (k in kw) d[k] = kw[k]
    return d
  }
  const dx = xs ? pick('x') : {}
  const dy = ys ? pick('y') : {}
  for (const k of ['base', 'subs', 'nonpositive', 'basex', 'basey', 'subsx', 'subsy', 'nonposx', 'nonposy']) delete kw[k]
  if (xs) ax.x.setScale(h, 'log', dx)
  if (ys) ax.y.setScale(h, 'log', dy)
  return plot(ax, args, kw, h)
}

/** axhline / axvline: a Line2D in data along one direction and Axes fractions along the other */
const refLine = (vertical: boolean): AxMethod => (ax, args, kw, h) => {
  const at = h.num(args[0] ?? pop(kw, vertical ? 'x' : 'y') ?? py.int(0))
  const lo = optNum(h, args[1] ?? pop(kw, vertical ? 'ymin' : 'xmin')) ?? 0
  const hi = optNum(h, args[2] ?? pop(kw, vertical ? 'ymax' : 'xmax')) ?? 1
  if ('transform' in kw) throw h.err('ValueError', `'transform' is not allowed as a keyword argument; ${vertical ? 'axvline' : 'axhline'} generates its own transform.`)
  const axis = vertical ? ax.x : ax.y
  const [a, b] = axis.view()
  const scale = at < Math.min(a, b) || at > Math.max(a, b)
  const spec = vertical ? newLine(h, [at, at], [lo, hi], rcc(h, 'lines.color')) : newLine(h, [lo, hi], [at, at], rcc(h, 'lines.color'))
  if (vertical) spec.ty = 'axes'
  else spec.tx = 'axes'
  const o = new LineObj(spec)
  o.applyKw(kw, h)
  ax.add(o)
  axis.update([at])
  if (scale) ax.request(vertical, !vertical)
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

const axline: AxMethod = (ax, args, kw, h) => {
  const xy1 = nums(h, args[0] ?? pop(kw, 'xy1')!, 'xy1')
  const xy2v = args[1] ?? pop(kw, 'xy2')
  const slopeV = pop(kw, 'slope')
  const hasXy2 = !isNone(xy2v)
  const hasSlope = !isNone(slopeV)
  if (hasXy2 === hasSlope) throw h.err('TypeError', "Exactly one of 'xy2' and 'slope' must be given")
  if ((ax.x.isLog() || ax.y.isLog()) && hasSlope) throw h.err('TypeError', "'slope' cannot be used with non-linear scales")
  const xy2 = hasXy2 ? nums(h, xy2v!, 'xy2') : null
  const slope = hasSlope ? h.num(slopeV!) : null
  if (xy2 && xy1[0] === xy2[0] && xy1[1] === xy2[1]) throw h.err('ValueError', `Cannot draw a line through two identical points (x=${pyRepr(py.tuple([py.float(xy1[0]), py.float(xy2[0])]))}, y=${pyRepr(py.tuple([py.float(xy1[1]), py.float(xy2[1])]))})`)
  const o = new LineObj(newLine(h, [], [], rcc(h, 'lines.color')))
  o.cls = 'AxLine'
  o.applyKw(kw, h)
  ax.add(o)
  ax.x.update(xy2 ? [xy1[0], xy2[0]] : [xy1[0]])
  ax.y.update(xy2 ? [xy1[1], xy2[1]] : [xy1[1]])
  ax.dynamic.set(o, () => {
    const g = ax.drawn!
    const dx = xy2 ? xy2[0] - xy1[0] : 1
    const dy = xy2 ? xy2[1] - xy1[1] : slope!
    const [x0, x1] = [Math.min(...g.xlim), Math.max(...g.xlim)]
    const [y0, y1] = [Math.min(...g.ylim), Math.max(...g.ylim)]
    if (dx === 0) { o.spec.x = [xy1[0], xy1[0]]; o.spec.y = [y0, y1]; return }
    if (dy === 0) { o.spec.x = [x0, x1]; o.spec.y = [xy1[1], xy1[1]]; return }
    const m = dy / dx
    // far enough beyond the view on both sides; the Axes clips it
    const span = (x1 - x0) + Math.abs((y1 - y0) / m)
    o.spec.x = [x0 - span, x1 + span]
    o.spec.y = o.spec.x.map((x) => xy1[1] + m * (x - xy1[0]))
  })
  ax.request()
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

// ---------------------------------------------------------------- patches and spans

const patchKw = (o: PatchObj, kw0: Kw, h: Host) => {
  const kw = { ...kw0 }
  const color = pop(kw, 'color')
  const fc = pop(kw, 'facecolor')
  const ec = pop(kw, 'edgecolor')
  const fill = pop(kw, 'fill')
  if (fill !== undefined) o.fill = h.truthy(fill)
  if (color && !isNone(color)) PATCH_PROPS.color.s!(o, [color], h, {})
  else {
    if (ec && !isNone(ec)) PATCH_PROPS.edgecolor.s!(o, [ec], h, {})
    if (fc && !isNone(fc)) PATCH_PROPS.facecolor.s!(o, [fc], h, {})
  }
  o.refreshEdge(h)
  o.applyKw(kw, h)
}

const addPatch = (ax: Axes, o: PatchObj) => {
  ax.add(o)
  const skip = o.shape.t === 'rect' && !o.shape.w && !o.shape.h
  if (!skip) {
    const [xs, ys] = o.bounds()
    if (o.spec.tx === 'data') ax.x.update(xs)
    if (o.spec.ty === 'data') ax.y.update(ys)
  }
  return o
}

const spanOf = (vertical: boolean): AxMethod => (ax, args, kw, h) => {
  const name = vertical ? 'axvspan' : 'axhspan'
  const [lo, hi] = vertical ? ['xmin', 'xmax'] : ['ymin', 'ymax']
  const a = args[0] ?? pop(kw, lo)
  const b = args[1] ?? pop(kw, hi)
  if (!a) throw h.err('TypeError', `${name}() missing 2 required positional arguments: '${lo}' and '${hi}'`)
  if (!b) throw h.err('TypeError', `${name}() missing 1 required positional argument: '${hi}'`)
  const u = optNum(h, args[2] ?? pop(kw, vertical ? 'ymin' : 'xmin')) ?? 0
  const w = optNum(h, args[3] ?? pop(kw, vertical ? 'ymax' : 'xmax')) ?? 1
  const v0 = h.num(a)
  const v1 = h.num(b)
  const shape: Shape = vertical ? { t: 'rect', x: v0, y: u, w: v1 - v0, h: w - u, angle: 0 } : { t: 'rect', x: u, y: v0, w: w - u, h: v1 - v0, angle: 0 }
  const o = new PatchObj('Rectangle', shape, h)
  if (vertical) o.spec.ty = 'axes'
  else o.spec.tx = 'axes'
  patchKw(o, kw, h)
  ax.add(o)
  if (vertical) ax.x.update([v0, v1])
  else ax.y.update([v0, v1])
  ax.request(vertical, !vertical)
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

const arrowVerts = (h: Host, x: number, y: number, dx: number, dy: number, kw: Kw): number[] => {
  const width = optNum(h, pop(kw, 'width')) ?? 0.001
  const hw = optNum(h, pop(kw, 'head_width')) ?? 3 * width
  const hl = optNum(h, pop(kw, 'head_length')) ?? 1.5 * hw
  const incl = flag(h, pop(kw, 'length_includes_head'), false)
  const shape = optStr(h, pop(kw, 'shape')) ?? 'full'
  const hs = optNum(h, pop(kw, 'overhang')) ?? 0
  const hz = flag(h, pop(kw, 'head_starts_at_zero'), false)
  const dist = Math.hypot(dx, dy)
  const length = incl ? dist : dist + hl
  if (!length) return []
  let left: [number, number][] = [[0, 0], [-hl, -hw / 2], [-hl * (1 - hs), -width / 2], [-length, -width / 2], [-length, 0]]
  if (!incl) left = left.map(([u, v]) => [u + hl, v])
  if (hz) left = left.map(([u, v]) => [u + hl / 2, v])
  const right = left.map(([u, v]) => [u, -v] as [number, number])
  let coords: [number, number][]
  if (shape === 'left') coords = left
  else if (shape === 'right') coords = right
  else if (shape === 'full') coords = [...left.slice(0, -1), ...right.slice(0, -1).reverse()]
  else throw h.err('ValueError', `Got unknown shape: ${pyRepr(py.str(shape))}`)
  const cx = dist ? dx / dist : 0
  const sx = dist ? dy / dist : 1
  return coords.flatMap(([u, v]) => [u * cx - v * sx + x + dx, u * sx + v * cx + y + dy])
}

const arrow: AxMethod = (ax, args, kw, h) => {
  if (args.length < 4) throw h.err('TypeError', `arrow() missing ${4 - args.length} required positional arguments`)
  const [x, y, dx, dy] = args.map((v) => h.num(v))
  const pts = arrowVerts(h, x, y, dx, dy, kw)
  const o = new PatchObj('FancyArrow', { t: 'polygon', pts, closed: true }, h)
  patchKw(o, kw, h)
  addPatch(ax, o)
  ax.request()
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

/** patch constructors, for matplotlib.patches and plt.Circle … */
const patchTypes = (h: Host): Record<string, Value> => {
  const make = (cls: string, shape: Shape, kw: Kw) => {
    const o = new PatchObj(cls, shape, h)
    patchKw(o, kw, h)
    return py.obj(o)
  }
  const xyOf = (v: Value | undefined, what = 'xy'): [number, number] => {
    if (!v) throw h.err('TypeError', `missing 1 required positional argument: '${what}'`)
    const p = nums(h, v, what)
    return [p[0], p[1]]
  }
  const req = (v: Value | undefined, cls: string, what: string): number => {
    if (!v) throw h.err('TypeError', `${cls}.__init__() missing 1 required positional argument: '${what}'`)
    return h.num(v)
  }
  const pre = (kw0: Kw): Kw => ({ ...kw0 })
  return {
    Rectangle: typ('Rectangle', (args, kw0) => {
      const kw = pre(kw0)
      const [x, y] = xyOf(args[0] ?? pop(kw, 'xy'))
      const w = req(args[1] ?? pop(kw, 'width'), 'Rectangle', 'width')
      const hh = req(args[2] ?? pop(kw, 'height'), 'Rectangle', 'height')
      const angle = optNum(h, args[3] ?? pop(kw, 'angle')) ?? 0
      const rp = pop(kw, 'rotation_point')
      if (rp && !(rp.k === 'str' && rp.v === 'xy')) throw h.err('NotImplementedError', "Rectangle(rotation_point=...) is not available in this sandbox")
      return make('Rectangle', { t: 'rect', x, y, w, h: hh, angle }, kw)
    }),
    Circle: typ('Circle', (args, kw0) => {
      const kw = pre(kw0)
      const [x, y] = xyOf(args[0] ?? pop(kw, 'xy'))
      const r = optNum(h, args[1] ?? pop(kw, 'radius')) ?? 5
      return make('Circle', { t: 'ellipse', cls: 'Circle', x, y, w: 2 * r, h: 2 * r, angle: 0 }, kw)
    }),
    Ellipse: typ('Ellipse', (args, kw0) => {
      const kw = pre(kw0)
      const [x, y] = xyOf(args[0] ?? pop(kw, 'xy'))
      const w = req(args[1] ?? pop(kw, 'width'), 'Ellipse', 'width')
      const hh = req(args[2] ?? pop(kw, 'height'), 'Ellipse', 'height')
      const angle = optNum(h, args[3] ?? pop(kw, 'angle')) ?? 0
      return make('Ellipse', { t: 'ellipse', cls: 'Ellipse', x, y, w, h: hh, angle }, kw)
    }),
    Polygon: typ('Polygon', (args, kw0) => {
      const kw = pre(kw0)
      const v = args[0] ?? pop(kw, 'xy')
      if (!v) throw h.err('TypeError', "Polygon.__init__() missing 1 required positional argument: 'xy'")
      const a = toArray(v, 'float64')
      if (a.ndim !== 2) throw h.err('ValueError', `not enough values to unpack (expected 2, got ${a.ndim})`)
      if (a.shape[1] !== 2) throw h.err('ValueError', `'vertices' must be 2D with shape (N, 2), but your input has shape (${a.shape.join(', ')})`)
      return make('Polygon', { t: 'polygon', pts: a.values(), closed: flag(h, args[1] ?? pop(kw, 'closed'), true) }, kw)
    }),
    Wedge: typ('Wedge', (args, kw0) => {
      const kw = pre(kw0)
      const [x, y] = xyOf(args[0] ?? pop(kw, 'center'), 'center')
      const r = req(args[1] ?? pop(kw, 'r'), 'Wedge', 'r')
      const t1 = req(args[2] ?? pop(kw, 'theta1'), 'Wedge', 'theta1')
      const t2 = req(args[3] ?? pop(kw, 'theta2'), 'Wedge', 'theta2')
      return make('Wedge', { t: 'wedge', x, y, r, t1, t2, width: optNum(h, args[4] ?? pop(kw, 'width')) }, kw)
    }),
    Arc: typ('Arc', (args, kw0) => {
      const kw = pre(kw0)
      const [x, y] = xyOf(args[0] ?? pop(kw, 'xy'))
      const w = req(args[1] ?? pop(kw, 'width'), 'Arc', 'width')
      const hh = req(args[2] ?? pop(kw, 'height'), 'Arc', 'height')
      const angle = optNum(h, pop(kw, 'angle')) ?? 0
      const t1 = optNum(h, pop(kw, 'theta1')) ?? 0
      const t2 = optNum(h, pop(kw, 'theta2')) ?? 360
      const o = new PatchObj('Arc', { t: 'arc', x, y, w, h: hh, angle, t1, t2 }, h)
      patchKw(o, kw, h)
      o.spec.fcs = ['none']
      if (!o.ecSet) o.spec.ecs = [rcc(h, 'patch.edgecolor')]
      return py.obj(o)
    }),
    FancyArrow: typ('FancyArrow', (args, kw0) => {
      const kw = pre(kw0)
      const [x, y, dx, dy] = [args[0] ?? pop(kw, 'x'), args[1] ?? pop(kw, 'y'), args[2] ?? pop(kw, 'dx'), args[3] ?? pop(kw, 'dy')].map((v) => h.num(v!))
      const pts = arrowVerts(h, x, y, dx, dy, kw)
      return make('FancyArrow', { t: 'polygon', pts, closed: true }, kw)
    }),
    Arrow: typ('Arrow', (args, kw0) => {
      const kw = pre(kw0)
      const [x, y, dx, dy] = args.slice(0, 4).map((v) => h.num(v))
      const width = optNum(h, args[4] ?? pop(kw, 'width')) ?? 1
      const L = Math.hypot(dx, dy)
      const a = Math.atan2(dy, dx)
      const base = [[0, 0.1], [0, -0.1], [0.8, -0.1], [0.8, -0.3], [1, 0], [0.8, 0.3], [0.8, 0.1]]
      const pts = base.flatMap(([u, v]) => [(u * L) * Math.cos(a) - (v * width) * Math.sin(a) + x, (u * L) * Math.sin(a) + (v * width) * Math.cos(a) + y])
      return make('Arrow', { t: 'polygon', pts, closed: true }, kw)
    }),
    RegularPolygon: typ('RegularPolygon', (args, kw0) => {
      const kw = pre(kw0)
      const [x, y] = xyOf(args[0] ?? pop(kw, 'xy'))
      const n = h.toInt(args[1] ?? pop(kw, 'numVertices')!)
      const r = optNum(h, pop(kw, 'radius') ?? args[2]) ?? 5
      const ori = optNum(h, pop(kw, 'orientation')) ?? 0
      const pts = Array.from({ length: n }, (_, i) => { const t = (2 * Math.PI * i) / n + Math.PI / 2 + ori; return [x + r * Math.cos(t), y + r * Math.sin(t)] }).flat()
      return make('RegularPolygon', { t: 'polygon', pts, closed: true }, kw)
    }),
    Patch: typ('Patch', (_a, kw0) => make('Patch', { t: 'none' }, pre(kw0))),
  }
}

// ---------------------------------------------------------------- scatter

const scatter: AxMethod = (ax, args, kw, h) => {
  const xv = args[0] ?? pop(kw, 'x')
  const yv = args[1] ?? pop(kw, 'y')
  if (!xv || !yv) throw h.err('TypeError', `scatter() missing 1 required positional argument: '${xv ? 'y' : 'x'}'`)
  const x = axisNums(h, ax.x, xv, 'scatter x')
  const y = axisNums(h, ax.y, yv, 'scatter y')
  if (x.length !== y.length) throw h.err('ValueError', 'x and y must be the same size')
  const sv = args[2] ?? pop(kw, 's')
  const cv = args[3] ?? pop(kw, 'c')
  const mv = args[4] ?? pop(kw, 'marker')
  const cmapV = args[5] ?? pop(kw, 'cmap')
  if (args[6]) kw.norm = args[6]
  if (args[7]) kw.vmin = args[7]
  if (args[8]) kw.vmax = args[8]
  const alpha = optNum(h, args[9] ?? pop(kw, 'alpha'))
  const lwV = args[10] ?? pop(kw, 'linewidths', 'linewidth', 'lw')
  const edgeV = pop(kw, 'edgecolors', 'edgecolor', 'ec')
  const faceKw = pop(kw, 'facecolors', 'facecolor', 'fc')
  const colorKw = pop(kw, 'color')
  const label = optStr(h, pop(kw, 'label'))
  pop(kw, 'plotnonfinite', 'colorizer')
  if (!isNone(pop(kw, 'data'))) throw h.err('NotImplementedError', 'scatter(data=...) is not available in this sandbox: pass the columns themselves')
  if (cv && !isNone(cv) && colorKw) throw h.err('ValueError', "Supply a 'c' argument or a 'color' kwarg but not both; they differ but their functionalities overlap.")
  const norm = normArg(h, kw)
  const lut = cmapArg(h, cmapV)
  const hollow = !!faceKw && faceKw.k === 'str' && faceKw.v.toLowerCase() === 'none'
  const marker = isNone(mv) ? rcs(h, 'scatter.marker') : markerOf(h, mv!) ?? 'None'
  const sizes = isNone(sv) ? [rcn(h, 'lines.markersize') ** 2] : toArray(sv!, 'float64').values()
  if (sizes.length !== 1 && sizes.length !== x.length) throw h.err('ValueError', 's must be a scalar, or float array-like with the same size as x and y')
  let colors: string[]
  let mapped: number[] | null = null
  const sources = cv && !isNone(cv) ? cv : colorKw ?? faceKw
  if (!sources || isNone(sources)) colors = [ax.nextPatchColor()]
  else if (cv && !isNone(cv)) {
    if (cv.k === 'str' || isStrList(cv) || (cv.k === 'list' && cv.items.length > 0 && !isNum(cv.items[0]) && cv.items[0].k !== 'list' && cv.items[0].k !== 'array' && cv.items[0].k !== 'tuple')) {
      try {
        colors = toColors(h, cv, 'c')
      } catch {
        throw h.err('ValueError', `'c' argument must be a color, a sequence of colors, or a sequence of numbers, not ${pyRepr(cv)}`)
      }
      if (colors.length !== 1 && colors.length !== x.length) throw h.err('ValueError', `'c' argument has ${colors.length} elements, which is inconsistent with 'x' and 'y' with size ${x.length}.`)
    } else {
      const a = toArray(cv, 'float64')
      if (a.ndim === 2 && (a.shape[1] === 3 || a.shape[1] === 4) && (a.shape[0] === 1 || a.shape[0] === x.length)) colors = toColors(h, py.arr(a))
      else if (a.size === x.length) {
        mapped = a.values()
        norm.autoscaleNone(mapped)
        colors = mapped.map((v) => norm.color(lut, v))
      } else if (a.ndim === 1 && (a.size === 3 || a.size === 4) && a.values().every((v) => v >= 0 && v <= 1)) colors = [toColor(h, py.arr(a))]
      else throw h.err('ValueError', `'c' argument has ${a.size} elements, which is inconsistent with 'x' and 'y' with size ${x.length}.`)
    }
  } else colors = toColors(h, sources)
  const unfilled = UNFILLED.has(marker ?? '')
  let lw = isNone(lwV) ? [unfilled ? rcn(h, 'lines.linewidth') : rcn(h, 'patch.linewidth')] : nums(h, lwV!, 'linewidths')
  if (lw.some((v) => Number.isNaN(v))) lw = [rcn(h, 'patch.linewidth')]
  let edges: string[] | null = null
  if (edgeV && !isNone(edgeV) && !(edgeV.k === 'str' && edgeV.v === 'face')) edges = toColors(h, edgeV, 'edgecolors')
  const spec: ScatterArtist = { kind: 'scatter', ...artistBase(1), x, y, colors, sizes, marker, edges, lw, label, alpha: alpha ?? 1 }
  const o = new ScatterObj(spec, lut, norm)
  o.alphaSet = alpha
  if (mapped) o.values = mapped
  o.shape = [x.length]
  o.hollow = hollow
  if (!isNone(sv)) o.sizeType = toArray(sv!).dtype
  if (!isNone(lwV)) o.lwType = toArray(lwV!).dtype
  o.applyKw(kw, h)
  ax.add(o)
  const keep = x.map((vx, i) => Number.isFinite(vx) && Number.isFinite(y[i]))
  ax.x.update(x.filter((_, i) => keep[i]))
  ax.y.update(y.filter((_, i) => keep[i]))
  ax.request()
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

// ---------------------------------------------------------------- bars

const shapeMismatch = (h: Host, sizes: number[]) => {
  const big = sizes.map((s, i) => [s, i]).filter(([s]) => s !== 1)
  if (big.length > 1) {
    const other = big.find(([s]) => s !== big[0][0])
    if (other) throw h.err('ValueError', `shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg ${big[0][1]} with shape (${big[0][0]},) and arg ${other[1]} with shape (${other[0]},).`)
  }
}

const barImpl = (ax: Axes, horizontal: boolean, posV: Value, hv: Value, wv: Value | undefined, bv: Value | undefined, kw: Kw, h: Host, color?: string[]): BarObj => {
  const posAxis = horizontal ? ax.y : ax.x
  const valAxis = horizontal ? ax.x : ax.y
  const pos0 = axisNums(h, posAxis, posV, 'bar x')
  const heights0 = axisNums(h, valAxis, hv, 'bar height')
  const w0 = isNone(wv) ? [0.8] : nums(h, wv!, 'width')
  const b0 = isNone(bv) ? [0] : nums(h, bv!, 'bottom')
  if (horizontal) shapeMismatch(h, [b0.length, w0.length, heights0.length, pos0.length])
  else shapeMismatch(h, [pos0.length, heights0.length, w0.length, b0.length])
  const n = Math.max(pos0.length, heights0.length, w0.length, b0.length)
  const pos = broadcast(pos0, n)
  const heights = broadcast(heights0, n)
  const widths = broadcast(w0, n)
  const bottoms = broadcast(b0, n)
  const align = oneOf(h, optStr(h, pop(kw, 'align')) ?? 'center', 'align', ['center', 'edge'])
  const tickLabel = pop(kw, 'tick_label')
  const log = flag(h, pop(kw, 'log'), false)
  const errX = pop(kw, 'xerr')
  const errY = pop(kw, 'yerr')
  const ecolor = pop(kw, 'ecolor')
  const capsize = pop(kw, 'capsize')
  const errorKw = kwOfDict(h, pop(kw, 'error_kw'))
  const cv = pop(kw, 'color')
  const fcv = pop(kw, 'facecolor', 'fc')
  const edgeV = pop(kw, 'edgecolor', 'ec')
  const lwV = pop(kw, 'linewidth', 'lw')
  const lsV = pop(kw, 'linestyle', 'ls')
  const alpha = optNum(h, pop(kw, 'alpha'))
  const labelV = pop(kw, 'label')
  const zV = optNum(h, pop(kw, 'zorder'))
  const colorOf = (v: Value) => {
    try {
      return toColors(h, v)
    } catch {
      throw h.err('ValueError', "'facecolor' or 'color' argument must be a valid color or sequence of colors.")
    }
  }
  const colors = color ?? (cv && !isNone(cv) ? colorOf(cv) : fcv && !isNone(fcv) ? colorOf(fcv) : [ax.nextPatchColor()])
  const edges = edgeV && !isNone(edgeV) ? toColors(h, edgeV, 'edgecolor') : ['none']
  if (log) valAxis.setScale(h, 'log', {})
  const x = pos.map((p, i) => (align === 'center' ? p - widths[i] / 2 : p))
  const spec: BarArtist = {
    kind: 'bar', ...artistBase(1), x, heights, bottoms, widths, colors, edges, lw: optNum(h, lwV) ?? rcn(h, 'patch.linewidth'), dash: lsV ? lineStyle(h, lsV).dash : null,
    label: null, horizontal, alpha: alpha ?? 1,
  }
  if (zV !== null) spec.z = zV
  const o = new BarObj(spec)
  o.alphaSet = alpha
  for (const k of Object.keys(kw)) if (!COSMETIC.has(k) && !(k in BASE)) throw h.err('AttributeError', `Rectangle.set() got an unexpected keyword argument '${k}'`)
  o.applyKw(kw, h)
  if (labelV && !isNone(labelV) && labelV.k !== 'list' && labelV.k !== 'tuple') spec.label = h.str(labelV)
  ax.add(o)
  const far = (start: number, size: number) => start + size - start + start
  const along = x.flatMap((l, i) => [l, far(l, widths[i])])
  const across = bottoms.flatMap((b, i) => [b, far(b, heights[i])])
  const live = heights.map((v, i) => v !== 0 || widths[i] !== 0)
  posAxis.update(along.filter((_, i) => live[Math.floor(i / 2)]))
  valAxis.update(across.filter((_, i) => live[Math.floor(i / 2)]))
  if (horizontal) o.sticky.x = [...bottoms]
  else o.sticky.y = [...bottoms]
  if (!isNone(tickLabel)) {
    posAxis.setTicks(h, floats(pos), undefined, false)
    const labs = tickLabel!.k === 'str' ? pos.map(() => (tickLabel as { v: string }).v) : h.iterate(tickLabel!).map((v) => h.str(v))
    if (labs.length !== pos.length) throw h.err('ValueError', `shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (${labs.length},) and requested shape (${pos.length},)`)
    posAxis.setTickLabels(h, py.list(labs.map(py.str)), false)
  }
  ax.request()
  if ((errX && !isNone(errX)) || (errY && !isNone(errY))) {
    const center = pos.map((p, i) => (align === 'center' ? p : p + widths[i] / 2))
    const ends = heights.map((v, i) => bottoms[i] + v)
    const ek: Kw = { fmt: py.str('none'), ...errorKw }
    if (errX && !isNone(errX)) ek.xerr = errX
    if (errY && !isNone(errY)) ek.yerr = errY
    ek.ecolor = ecolor ?? errorKw.ecolor ?? py.str('k')
    if (capsize) ek.capsize = capsize
    const ev = errorbar(ax, horizontal ? [floats(ends), floats(center)] : [floats(center), floats(ends)], ek, h)
    o.errorbar = ev
  }
  return o
}

const bar: AxMethod = (ax, args, kw, h) => {
  const posV = args[0] ?? pop(kw, 'x')
  const hv = args[1] ?? pop(kw, 'height')
  if (!posV) throw h.err('TypeError', "bar() missing 2 required positional arguments: 'x' and 'height'")
  if (!hv) throw h.err('TypeError', "bar() missing 1 required positional argument: 'height'")
  const first = ax.children.length
  const o = barImpl(ax, false, posV, hv, args[2] ?? pop(kw, 'width'), args[3] ?? pop(kw, 'bottom'), kw, h)
  added(h, ax, first)
  return py.obj(o)
}

const barh: AxMethod = (ax, args, kw, h) => {
  const posV = args[0] ?? pop(kw, 'y')
  const wv = args[1] ?? pop(kw, 'width')
  if (!posV) throw h.err('TypeError', "barh() missing 2 required positional arguments: 'y' and 'width'")
  if (!wv) throw h.err('TypeError', "barh() missing 1 required positional argument: 'width'")
  const first = ax.children.length
  const o = barImpl(ax, true, posV, wv, args[2] ?? pop(kw, 'height'), args[3] ?? pop(kw, 'left'), kw, h)
  added(h, ax, first)
  return py.obj(o)
}

const barLabel: AxMethod = (ax, args, kw, h) => {
  const c = args[0] ?? pop(kw, 'container')
  if (!c || c.k !== 'obj' || !(c.o instanceof BarObj)) throw h.err('AttributeError', "bar_label() needs the BarContainer that bar() returned")
  for (const k of ['horizontalalignment', 'ha', 'verticalalignment', 'va']) if (k in kw) throw h.err('ValueError', `Passing ${pyRepr(py.str(k))} to bar_label() is not supported.`)
  const lv = args[1] ?? pop(kw, 'labels')
  const fmtV = pop(kw, 'fmt') ?? py.str('%g')
  const type = oneOf(h, optStr(h, pop(kw, 'label_type')) ?? 'edge', 'label_type', ['edge', 'center'])
  const padding = optNum(h, pop(kw, 'padding')) ?? 0
  const a = c.o.spec
  const labels = isNone(lv) ? [] : h.iterate(lv!).map((v) => h.str(v))
  const first = ax.children.length
  const out: Value[] = []
  const inverted = (a.horizontal ? ax.x : ax.y).inverted()
  const dirSign = inverted ? -1 : 1
  const eSegs = c.o.errorbar.k === 'obj' && c.o.errorbar.o instanceof ContainerObj ? (() => {
    const bars = c.o.errorbar.o.items[2]
    return bars && bars.k === 'tuple' && bars.items[0]?.k === 'obj' && bars.items[0].o instanceof SegsObj ? bars.items[0].o.spec.paths : null
  })() : null
  a.x.forEach((l, i) => {
    const dat = c.o instanceof BarObj ? c.o.datavalues[i] : a.heights[i]
    const lo = a.bottoms[i]
    const hi = lo + a.heights[i]
    const center = l + a.widths[i] / 2
    const extrema = dat >= 0 ? Math.max(lo, hi) : Math.min(lo, hi)
    const seg = eSegs?.[i]
    const endpt = seg ? (a.horizontal ? (dat >= 0 ? Math.max(seg[0], seg[2]) : Math.min(seg[0], seg[2])) : (dat >= 0 ? Math.max(seg[1], seg[3]) : Math.min(seg[1], seg[3]))) : extrema
    const sign = dat >= 0 ? 1 : -1
    const value = type === 'center' ? sign * Math.abs(hi - lo) : extrema
    let text = labels[i]
    if (Number.isNaN(dat)) text = ''
    if (text === undefined) text = fmtV.k === 'str' ? (fmtV.v.includes('{') ? fmtV.v.replace(/\{:([^}]*)\}/g, (_m, sp: string) => formatSpec(py.float(value), sp)).replace(/\{\}/g, formatG(value)) : printf(fmtV.v, value)) : h.str(h.call(fmtV, [py.float(value)]))
    const pos = type === 'center' ? (lo + hi) / 2 : endpt
    const t = newText(h, a.horizontal ? pos : center, a.horizontal ? center : pos, text)
    if (type === 'center') {
      t.ha = 'center'
      t.va = 'center'
    } else if (a.horizontal) {
      t.dx = dirSign * sign * padding
      t.ha = inverted ? (dat > 0 ? 'right' : 'left') : dat < 0 ? 'right' : 'left'
      t.va = 'center'
    } else {
      t.dy = dirSign * sign * padding
      t.ha = 'center'
      t.va = inverted ? (dat > 0 ? 'top' : 'bottom') : dat < 0 ? 'top' : 'bottom'
    }
    const o = new TextObj(t, String(t.x), String(t.y))
    o.cls = 'Annotation'
    o.xy = [t.x, t.y]
    o.applyKw(kw, h)
    ax.add(o)
    out.push(py.obj(o))
  })
  added(h, ax, first)
  return py.list(out)
}

/** numpy.histogram: equal-width bins over [min, max] (or range=), the last bin closed */
export const histogram = (data: number[], bins: number | number[], range?: [number, number], weights?: number[]): { counts: number[]; edges: number[] } => {
  let edges: number[]
  if (Array.isArray(bins)) edges = bins
  else {
    let [lo, hi] = range ?? [minOf(data), maxOf(data)]
    if (!data.length && !range) [lo, hi] = [0, 1]
    if (lo === hi) [lo, hi] = [lo - 0.5, hi + 0.5]
    const step = (hi - lo) / bins
    edges = Array.from({ length: bins + 1 }, (_, i) => (i === bins ? hi : lo + i * step))
  }
  const counts = new Array<number>(edges.length - 1).fill(0)
  const last = edges.length - 1
  data.forEach((v, n) => {
    if (!(v >= edges[0] && v <= edges[last])) return
    let lo = 0
    let hi = last
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1
      if (v >= edges[mid]) lo = mid
      else hi = mid
    }
    counts[lo] += weights ? weights[n] : 1
  })
  return { counts, edges }
}

/** np.histogram_bin_edges for a string rule: the number of equal-width bins */
const autoBins = (h: Host, rule: string, data: number[], lo: number, hi: number): number => {
  const n = data.length
  const span = hi - lo
  if (!n || span === 0) return 1
  const s = [...data].sort((a, b) => a - b)
  const q = (p: number) => {
    const i = (n - 1) * p
    const f = Math.floor(i)
    return f + 1 < n ? s[f] + (s[f + 1] - s[f]) * (i - f) : s[f]
  }
  const mean = data.reduce((a, b) => a + b, 0) / n
  const std = Math.sqrt(data.reduce((a, b) => a + (b - mean) ** 2, 0) / n)
  const sturges = span / (Math.log2(n) + 1)
  const fd = (2 * (q(0.75) - q(0.25))) / n ** (1 / 3)
  let width: number
  switch (rule) {
    case 'sturges': width = sturges; break
    case 'sqrt': width = span / Math.sqrt(n); break
    case 'rice': width = span / (2 * n ** (1 / 3)); break
    case 'scott': width = ((24 * Math.sqrt(Math.PI)) / n) ** (1 / 3) * std; break
    case 'fd': width = fd; break
    case 'auto': width = fd ? Math.min(fd, sturges) : sturges; break
    default: throw h.err('ValueError', `${pyRepr(py.str(rule))} is not a valid estimator for \`bins\``)
  }
  return width ? Math.ceil(span / width) : 1
}

const datasets = (h: Host, v: Value, what: string): number[][] => {
  if (isNum(v)) return [[h.num(v)]]
  if ((v.k === 'list' || v.k === 'tuple') && v.items.some((x) => x.k === 'list' || x.k === 'tuple' || x.k === 'array' || x.k === 'obj')) return v.items.map((x) => (isNum(x) ? [h.num(x)] : nums(h, x, what)))
  const a = toArray(v, 'float64')
  if (a.ndim === 0) return [a.values()]
  if (a.ndim === 1) return [a.values()]
  if (a.ndim === 2) {
    const [r, c] = a.shape
    const vals = a.values()
    return Array.from({ length: c }, (_, j) => Array.from({ length: r }, (_, i) => vals[i * c + j]))
  }
  throw h.err('ValueError', `${what} must have 2 or fewer dimensions`)
}

const hist: AxMethod = (ax, args, kw, h) => {
  const xv = args[0] ?? pop(kw, 'x')
  if (!xv) throw h.err('TypeError', "hist() missing 1 required positional argument: 'x'")
  const xs = datasets(h, xv, 'x')
  const bv = args[1] ?? pop(kw, 'bins') ?? rcv(h, 'hist.bins')
  const rv = args[2] ?? pop(kw, 'range')
  const density = flag(h, args[3] ?? pop(kw, 'density'), false)
  const wV = args[4] ?? pop(kw, 'weights')
  const cumV = args[5] ?? pop(kw, 'cumulative')
  const bottomV = args[6] ?? pop(kw, 'bottom')
  const histtype = oneOf(h, optStr(h, args[7] ?? pop(kw, 'histtype')) ?? 'bar', 'histtype', ['bar', 'barstacked', 'step', 'stepfilled'])
  const align = oneOf(h, optStr(h, args[8] ?? pop(kw, 'align')) ?? 'mid', 'align', ['left', 'mid', 'right'])
  const orientation = oneOf(h, optStr(h, args[9] ?? pop(kw, 'orientation')) ?? 'vertical', 'orientation', ['horizontal', 'vertical'])
  const rwidth = optNum(h, args[10] ?? pop(kw, 'rwidth'))
  const log = flag(h, args[11] ?? pop(kw, 'log'), false)
  const colorV = args[12] ?? pop(kw, 'color')
  const labelV = args[13] ?? pop(kw, 'label')
  const stacked = flag(h, args[14] ?? pop(kw, 'stacked'), false) || histtype === 'barstacked'
  const nx = xs.length
  const w = isNone(wV) ? null : datasets(h, wV!, 'weights')
  if (w && (w.length !== nx || w.some((wi, i) => wi.length !== xs[i].length))) throw h.err('ValueError', 'weights should have the same shape as x')
  let colors: string[]
  if (isNone(colorV)) colors = xs.map(() => ax.nextLineColor())
  else {
    colors = toColors(h, colorV!)
    if (colors.length !== nx) throw h.err('ValueError', `The 'color' keyword argument must have one color per dataset, but ${nx} datasets and ${colors.length} colors were provided`)
  }
  const clean = xs.map((d) => d.filter((v) => !Number.isNaN(v)))
  const pooled = clean.flat()
  let range: [number, number] | undefined
  if (!isNone(rv)) {
    const r = nums(h, rv!, 'range')
    if (r[0] > r[1]) throw h.err('ValueError', `max must be larger than min in range parameter.`)
    range = [r[0], r[1]]
  } else if (pooled.length) range = [minOf(pooled), maxOf(pooled)]
  let bins: number | number[]
  if (bv.k === 'str') {
    const [lo, hi] = range ?? [0, 1]
    bins = autoBins(h, bv.v, range && !isNone(rv) ? pooled.filter((v) => v >= lo && v <= hi) : pooled, lo, hi)
  } else if (isNum(bv)) {
    bins = h.toInt(bv)
    if (bins < 1) throw h.err('ValueError', '`bins` must be positive, when an integer')
  } else {
    bins = nums(h, bv, 'bins')
    if (bins.some((b, i) => i > 0 && b < (bins as number[])[i - 1])) throw h.err('ValueError', '`bins` must increase monotonically, when an array')
  }
  if (typeof bins === 'number' && bins > 4096) throw h.err('ValueError', '`bins`: at most 4096 in this sandbox')
  let edges: number[] = []
  let tops: number[][] = clean.map((d, i) => {
    const r = histogram(w ? xs[i] : d, bins, range, w ? w[i] : undefined)
    edges = r.edges
    return r.counts
  })
  const diffs = edges.slice(1).map((e, i) => e - edges[i])
  if (density && !stacked) tops = tops.map((t) => { const total = t.reduce((a, b) => a + b, 0); return t.map((c, i) => c / diffs[i] / total) })
  if (stacked) {
    tops = tops.map((_, k) => tops[0].map((__, i) => tops.slice(0, k + 1).reduce((s, t) => s + t[i], 0)))
    if (density) {
      const total = tops[tops.length - 1].reduce((a, b) => a + b, 0)
      tops = tops.map((t) => t.map((c, i) => c / diffs[i] / total))
    }
  }
  if (cumV && h.truthy(cumV)) {
    const reverse = isNum(cumV) && h.num(cumV) < 0
    tops = tops.map((t) => {
      const base = density ? t.map((c, i) => c * diffs[i]) : t
      const seq = reverse ? [...base].reverse() : base
      let s = 0
      const cum = seq.map((c) => (s += c))
      return reverse ? cum.reverse() : cum
    })
  }
  const first = ax.children.length
  const labels = isNone(labelV) ? [] : labelV!.k === 'str' ? [labelV!.v] : h.iterate(labelV!).map((v) => h.str(v))
  const horizontal = orientation === 'horizontal'
  const patches: Value[] = []
  if (histtype.startsWith('bar')) {
    const dr = rwidth !== null ? Math.max(0, Math.min(1, rwidth)) : tops.length > 1 && !stacked ? 0.8 : 1
    let width: number[]
    let dw: number[]
    let boffset: number[]
    if (histtype === 'bar' && !stacked) {
      width = diffs.map((t) => (dr * t) / nx)
      dw = width
      boffset = diffs.map((t) => -0.5 * dr * t * (1 - 1 / nx))
    } else {
      width = diffs.map((t) => dr * t)
      boffset = diffs.map(() => 0)
      dw = diffs.map(() => 0)
    }
    if (align === 'mid') boffset = boffset.map((b, i) => b + 0.5 * diffs[i])
    else if (align === 'right') boffset = boffset.map((b, i) => b + diffs[i])
    let bottom = isNone(bottomV) ? diffs.map(() => 0) : broadcast(nums(h, bottomV!, 'bottom'), diffs.length)
    tops.forEach((top, k) => {
      const height = stacked ? top.map((t, i) => t - bottom[i]) : top
      const pos = edges.slice(0, -1).map((e, i) => e + boffset[i])
      const rest = { ...kw }
      const o = barImpl(ax, horizontal, floats(pos), floats(height), floats(width), floats(bottom), rest, h, [colors[k]])
      o.fromHist = true
      o.spec.label = labels[k] ?? (k === 0 ? o.spec.label : '_nolegend_')
      if (log) (horizontal ? ax.x : ax.y).setScale(h, 'log', {})
      if (k > 0) o.sticky = { x: [], y: [] }
      patches.push(py.obj(o))
      if (stacked) bottom = top
      boffset = boffset.map((b, i) => b + dw[i])
    })
  } else {
    const fill = histtype === 'stepfilled'
    const n = edges.length
    const bv0 = isNone(bottomV) ? 0 : h.num(bottomV!)
    const shift = align === 'left' ? -0.5 * (edges[1] - edges[0]) : align === 'right' ? 0.5 * (edges[1] - edges[0]) : 0
    if (log) (horizontal ? ax.x : ax.y).setScale(h, 'log', {})
    const outlines = tops.map((top, k) => {
      const prev = stacked && k > 0 ? tops[k - 1] : null
      const up: number[] = []
      const px: number[] = []
      for (let i = 0; i < n - 1; i++) {
        px.push(edges[i] + shift, edges[i + 1] + shift)
        up.push(top[i] + bv0, top[i] + bv0)
      }
      const base = [...px].reverse().map((_, i) => (prev ? prev[Math.floor((px.length - 1 - i) / 2)] + bv0 : bv0))
      const allX = fill ? [px[0], ...px, ...[...px].reverse()] : [px[0], ...px, px[px.length - 1]]
      const allY = fill ? [base[base.length - 1], ...up, ...base] : [bv0, ...up, bv0]
      return { allX, allY }
    })
    for (let k = tops.length - 1; k >= 0; k--) {
      const { allX, allY } = outlines[k]
      const xy = horizontal ? allY.flatMap((v, i) => [v, allX[i]]) : allX.flatMap((v, i) => [v, allY[i]])
      const o = new PatchObj('Polygon', { t: 'polygon', pts: xy, closed: fill }, h)
      const rest = { ...kw }
      if (fill) patchKw(o, { facecolor: py.str(colors[k]), ...rest }, h)
      else {
        o.spec.z = 2
        patchKw(o, { facecolor: py.str('none'), edgecolor: py.str(colors[k]), ...rest }, h)
        o.spec.fcs = ['none']
      }
      o.spec.label = labels[k] ?? '_nolegend_'
      addPatch(ax, o)
      if (horizontal) o.sticky.x = [0]
      else o.sticky.y = [0]
      patches.unshift(py.list([py.obj(o)]))
    }
    ax.request()
  }
  added(h, ax, first)
  if (nx === 1) return py.tuple([floats(tops[0]), floats(edges), patches[0].k === 'list' ? py.obj(new SilentList('Polygon', patches[0].items)) : patches[0]])
  return py.tuple([floats(tops.flat(), [nx, edges.length - 1]), floats(edges), py.obj(new SilentList(histtype.startsWith('bar') ? 'BarContainer' : 'list[Polygon]', patches))])
}

// ---------------------------------------------------------------- images

/** a colour array (M, N, 3 or 4) as pixels, or a 2-D array's values */
const imagePixels = (h: Host, a0: NDArray): { pixels: string[]; values: number[]; rgb: boolean; rows: number; cols: number; a: NDArray } => {
  let a = a0
  if (a.ndim === 3 && a.shape[2] === 1) a = NDArray.create(a.values(), [a.shape[0], a.shape[1]], a.dtype)
  if (!(a.ndim === 2 || (a.ndim === 3 && (a.shape[2] === 3 || a.shape[2] === 4)))) throw h.err('TypeError', `Invalid shape (${a.shape.join(', ')}${a.ndim === 1 ? ',' : ''}) for image data`)
  const rows = a.shape[0]
  const cols = a.shape[1]
  const vals = a.values()
  if (a.ndim === 2) return { pixels: [], values: vals, rgb: false, rows, cols, a }
  const isInt = a.dtype === 'int64' || a.dtype === 'uint8'
  const top = isInt ? 255 : 1
  const ch = a.shape[2]
  const lo = minOf(vals)
  const hi = maxOf(vals)
  if (lo < 0 || hi > top) h.print(`Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [${pyRepr(isInt ? py.int(lo) : py.float(lo))}..${pyRepr(isInt ? py.int(hi) : py.float(hi))}].\n`)
  const clip = (v: number) => (Number.isNaN(v) ? 0 : Math.max(0, Math.min(top, v))) / top
  const pixels = Array.from({ length: rows * cols }, (_, p) => {
    const i = p * ch
    return rgbHex(clip(vals[i]), clip(vals[i + 1]), clip(vals[i + 2]), ch === 4 ? clip(vals[i + 3]) : 1)
  })
  return { pixels, values: [], rgb: true, rows, cols, a }
}

const imshow: AxMethod = (ax, args, kw, h) => {
  const src = args[0] ?? pop(kw, 'X')
  if (!src) throw h.err('TypeError', "imshow() missing 1 required positional argument: 'X'")
  const cmapV = args[1] ?? pop(kw, 'cmap')
  if (args[2]) kw.norm = args[2]
  const pix = imagePixels(h, toArray(src))
  const hasNorm = !isNone(kw.norm)
  const given = !isNone(kw.vmin) || !isNone(kw.vmax) || hasNorm
  const norm = normArg(h, kw)
  const lut = cmapArg(h, cmapV)
  const aspectV = pop(kw, 'aspect')
  const interp = optStr(h, pop(kw, 'interpolation'))
  const alpha = optNum(h, pop(kw, 'alpha'))
  const origin = oneOf(h, optStr(h, pop(kw, 'origin')) ?? rcs(h, 'image.origin'), 'origin', ['upper', 'lower'])
  const extV = pop(kw, 'extent')
  for (const k of ['interpolation_stage', 'filternorm', 'filterrad', 'resample', 'url', 'colorizer']) pop(kw, k)
  const { rows, cols } = pix
  let extent: [number, number, number, number] = origin === 'upper' ? [-0.5, cols - 0.5, rows - 0.5, -0.5] : [-0.5, cols - 0.5, -0.5, rows - 0.5]
  if (!isNone(extV)) {
    const e = nums(h, extV!, 'extent')
    if (e.length !== 4) throw h.err('ValueError', 'extent must be a 4-tuple')
    extent = e as typeof extent
  }
  const spec: ImageArtist = {
    kind: 'image', ...artistBase(0), rows, cols, pixels: pix.pixels, values: pix.values, cmap: pix.rgb ? null : lut.name, vmin: 0, vmax: 1, auto: !given, rgb: pix.rgb, extent, alpha: alpha ?? 1,
    smooth: !!interp && !['nearest', 'none', 'antialiased', 'auto', 'data'].includes(interp),
  }
  const o = new ImageObj(spec, lut, norm, pix.rgb ? null : pix.values, [rows, cols])
  o.source = pix.a.copy()
  if (!isNone(extV)) o.extentRaw = py.list(h.iterate(extV!))
  o.origin = origin
  o.alphaSet = alpha
  if (!pix.rgb) {
    if (isNone(cmapV) && rcs(h, 'image.cmap') === 'viridis') note(h, 'info', 'default_cmap')
    norm.autoscaleNone(pix.values)
    o.recolor()
    if (!given) note(h, 'info', 'auto_range', { min: +(norm.vmin ?? 0).toPrecision(4), max: +(norm.vmax ?? 0).toPrecision(4) })
  } else if (origin === 'lower') spec.pixels = flipRows(spec.pixels, rows, cols)
  o.applyKw(kw, h)
  if (aspectV !== undefined || spec.tx === 'data') setAspect(h, ax, isNone(aspectV) ? rcv(h, 'image.aspect') : aspectV!)
  ax.add(o)
  ax.imageExtent(o, extent)
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

const setAspect = (h: Host, ax: Axes, v: Value) => {
  if (v.k === 'str') {
    if (v.v === 'auto') ax.aspect = 'auto'
    else if (v.v === 'equal') ax.aspect = 1
    else throw h.err('ValueError', `could not convert string to float: ${pyRepr(v)}`)
  } else {
    const a = h.num(v)
    if (!(a > 0) || !Number.isFinite(a)) throw h.err('ValueError', 'aspect must be finite and positive ')
    ax.aspect = a
  }
}

const matshow: AxMethod = (ax, args, kw, h) => {
  const Z = args[0] ?? pop(kw, 'Z')
  const o = imshow(ax, [Z!], { origin: py.str('upper'), interpolation: py.str('nearest'), aspect: py.str('equal'), ...kw }, h)
  ax.titleY = 1.05
  ax.x.style.tick1 = true
  ax.x.style.tick2 = true
  ax.x.style.label1 = false
  ax.x.style.label2 = true
  const m: MaxN = { nbins: 9, steps: validateSteps([1, 2, 5, 10]), integer: true, minN: 2, prune: null, symmetric: false }
  ax.x.ticker.loc = { k: 'maxn', m }
  ax.y.ticker.loc = { k: 'maxn', m }
  return o
}

const spy: AxMethod = (ax, args, kw, h) => {
  const Z = toArray(args[0] ?? pop(kw, 'Z')!, 'float64')
  const precision = optNum(h, pop(kw, 'precision')) ?? 0
  const marker = pop(kw, 'marker')
  const ms = pop(kw, 'markersize')
  if (Z.ndim !== 2) throw h.err('TypeError', 'spy() needs a 2-D array')
  const [r, c] = Z.shape
  const vals = Z.values()
  let out: Value
  if (isNone(marker) && isNone(ms)) {
    const mask = vals.map((v) => (Math.abs(v) > precision ? 1 : 0))
    const aspect = pop(kw, 'aspect') ?? py.str('equal')
    const origin = pop(kw, 'origin') ?? py.str('upper')
    out = imshow(ax, [py.arr(NDArray.create(mask, [r, c], 'float64'))], { cmap: py.str('Greys'), vmin: py.int(0), vmax: py.int(1), interpolation: py.str('nearest'), aspect, origin, ...kw }, h)
  } else {
    const xs: number[] = []
    const ys: number[] = []
    vals.forEach((v, i) => {
      if (Math.abs(v) > precision) {
        xs.push(i % c)
        ys.push(Math.floor(i / c))
      }
    })
    const lines = plotLines(ax, [floats(xs), floats(ys)], { linestyle: py.str('None'), marker: marker ?? py.str('s'), markersize: ms ?? py.int(6), ...kw }, h)
    setAspect(h, ax, py.str('equal'))
    ax.x.setLim(h, -0.5, c - 0.5, false)
    ax.y.setLim(h, r - 0.5, -0.5, false)
    out = py.obj(lines[0])
    added(h, ax, ax.children.length - 1)
  }
  ax.titleY = 1.05
  ax.x.style.tick1 = true
  ax.x.style.tick2 = true
  ax.x.style.label1 = false
  ax.x.style.label2 = true
  const m: MaxN = { nbins: 9, steps: validateSteps([1, 2, 5, 10]), integer: true, minN: 2, prune: null, symmetric: false }
  ax.x.ticker.loc = { k: 'maxn', m }
  ax.y.ticker.loc = { k: 'maxn', m }
  return out
}

// ---------------------------------------------------------------- filled regions

const contiguous = (mask: boolean[]): [number, number][] => {
  const out: [number, number][] = []
  let s = -1
  mask.forEach((m, i) => {
    if (m && s < 0) s = i
    if (!m && s >= 0) {
      out.push([s, i])
      s = -1
    }
  })
  if (s >= 0) out.push([s, mask.length])
  return out
}

const fillBetween = (vertical: boolean): AxMethod => (ax, args, kw, h) => {
  const [indName, d1, d2] = vertical ? ['x', 'y1', 'y2'] : ['y', 'x1', 'x2']
  const indV = args[0] ?? pop(kw, indName)
  const d1V = args[1] ?? pop(kw, d1)
  if (!indV) throw h.err('TypeError', `fill_between${vertical ? '' : 'x'}() missing 2 required positional arguments: '${indName}' and '${d1}'`)
  if (!d1V) throw h.err('TypeError', `fill_between${vertical ? '' : 'x'}() missing 1 required positional argument: '${d1}'`)
  const d2V = args[2] ?? pop(kw, d2)
  const whereV = args[3] ?? pop(kw, 'where')
  const interp = flag(h, args[4] ?? pop(kw, 'interpolate'), false)
  const stepV = optStr(h, args[5] ?? pop(kw, 'step'))
  const ind = nums(h, indV, indName)
  const check = (v: Value, name: string) => {
    const a = nums(h, v, name)
    if (a.length !== 1 && a.length !== ind.length) throw h.err('ValueError', `'${indName}' has size ${ind.length}, but '${name}' has an unequal size of ${a.length}`)
    return broadcast(a, ind.length)
  }
  const dep1 = check(d1V, d1)
  const dep2 = isNone(d2V) ? ind.map(() => 0) : check(d2V!, d2)
  const where = isNone(whereV) ? ind.map(() => true) : toArray(whereV!).values().map((v) => v !== 0)
  if (where.length !== ind.length) throw h.err('ValueError', `where size (${where.length}) does not match ${indName} size (${ind.length})`)
  const mask = where.map((w, i) => w && Number.isFinite(ind[i]) && Number.isFinite(dep1[i]) && Number.isFinite(dep2[i]))
  const colorV = pop(kw, 'color')
  const fcV = pop(kw, 'facecolor', 'facecolors', 'fc')
  const ecV = pop(kw, 'edgecolor', 'edgecolors', 'ec')
  const face = colorV && !isNone(colorV) ? toColors(h, colorV) : fcV && !isNone(fcV) ? toColors(h, fcV, 'facecolor') : [ax.nextPatchColor()]
  const edge = ecV && !isNone(ecV) ? toColors(h, ecV, 'edgecolor') : colorV && !isNone(colorV) ? face : ['none']
  const paths: Path[] = []
  const interpAt = (idx: number): [number, number] => {
    const im1 = Math.max(idx - 1, 0)
    const j = Math.min(idx, ind.length - 1)
    const d0 = dep1[im1] - dep2[im1]
    const dj = dep1[j] - dep2[j]
    if (d0 === dj) return [ind[j], dep1[j]]
    const t = (0 - d0) / (dj - d0)
    const root = ind[im1] + Math.max(0, Math.min(1, t)) * (ind[j] - ind[im1])
    const span = ind[j] - ind[im1]
    const v = span ? dep1[im1] + ((root - ind[im1]) / span) * (dep1[j] - dep1[im1]) : dep1[j]
    return [root, v]
  }
  const ptsAll: [number[], number[]] = [[], []]
  for (const [i0, i1] of contiguous(mask)) {
    let I = ind.slice(i0, i1)
    let A = dep1.slice(i0, i1)
    let B = dep2.slice(i0, i1)
    if (stepV) {
      if (!['pre', 'post', 'mid'].includes(stepV)) throw h.err('ValueError', `'${stepV}' is not a valid value for step; supported values are 'pre', 'post', 'mid'`)
      const m = stepXY(I, stepV as 'pre')
      A = stepVals(A, stepV as 'pre')
      B = stepVals(B, stepV as 'pre')
      I = m
    }
    if (!I.length) continue
    const start = interp ? interpAt(i0) : [I[0], B[0]]
    const end = interp ? interpAt(i1) : [I[I.length - 1], B[B.length - 1]]
    const uu = [start[0], ...I, end[0], ...[...I].reverse()]
    const vv = [start[1], ...A, end[1], ...[...B].reverse()]
    paths.push(vertical ? uu.flatMap((u, k) => [u, vv[k]]) : uu.flatMap((u, k) => [vv[k], u]))
    ptsAll[0].push(...I, ...I)
    ptsAll[1].push(...A, ...B)
  }
  const spec = newPoly(face[0], edge[0], rcn(h, 'patch.linewidth'))
  spec.fcs = face
  spec.ecs = edge
  spec.paths = paths
  const o = new PolyObj('FillBetweenPolyCollection', spec)
  o.applyKw(kw, h)
  ax.add(o)
  const mainAxis = vertical ? ax.x : ax.y
  const depAxis = vertical ? ax.y : ax.x
  if ((vertical ? spec.tx : spec.ty) === 'data') mainAxis.update(ptsAll[0])
  if ((vertical ? spec.ty : spec.tx) === 'data') depAxis.update(ptsAll[1])
  ax.request()
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

const stepXY = (x: number[], where: 'pre' | 'post' | 'mid'): number[] => {
  if (!x.length) return []
  if (where === 'mid') {
    const out: number[] = [x[0]]
    for (let i = 0; i < x.length - 1; i++) { const m = (x[i] + x[i + 1]) / 2; out.push(m, m) }
    out.push(x[x.length - 1])
    return out
  }
  const out: number[] = [x[0]]
  for (let i = 1; i < x.length; i++) out.push(where === 'pre' ? x[i - 1] : x[i], x[i])
  return out
}
const stepVals = (y: number[], where: 'pre' | 'post' | 'mid'): number[] => {
  if (!y.length) return []
  if (where === 'mid') return y.flatMap((v) => [v, v])
  const out: number[] = [y[0]]
  for (let i = 1; i < y.length; i++) out.push(where === 'pre' ? y[i] : y[i - 1], y[i])
  return out
}

const fill: AxMethod = (ax, args0, kw, h) => {
  const args = [...args0]
  const out: Value[] = []
  const first = ax.children.length
  const closed = flag(h, pop(kw, 'closed'), true)
  while (args.length) {
    const xv = args.shift()!
    const yv = args.shift()
    if (!yv) throw h.err('TypeError', 'fill() needs x and y in pairs')
    const c = args.length && args[0].k === 'str' ? args.shift() : undefined
    const xs = nums(h, xv, 'fill x')
    const ys = nums(h, yv, 'fill y')
    if (xs.length !== ys.length) throw h.err('ValueError', `x and y must have same first dimension, but have shapes (${xs.length},) and (${ys.length},)`)
    const o = new PatchObj('Polygon', { t: 'polygon', pts: xs.flatMap((x, i) => [x, ys[i]]), closed }, h)
    const given = c ?? kw.color ?? kw.facecolor ?? kw.fc
    const k: Kw = { ...kw }
    if (!given) k.facecolor = py.str(ax.nextPatchColor())
    else if (c) k.facecolor = c
    patchKw(o, k, h)
    addPatch(ax, o)
    out.push(py.obj(o))
  }
  ax.request()
  added(h, ax, first)
  return py.list(out)
}

const stackplot: AxMethod = (ax, args, kw, h) => {
  const xv = args[0] ?? pop(kw, 'x')
  if (!xv) throw h.err('TypeError', "stackplot() missing 1 required positional argument: 'x'")
  const x = nums(h, xv, 'x')
  const rows = args.slice(1).flatMap((v) => {
    const a = toArray(v, 'float64')
    if (a.ndim === 2) {
      const [r, c] = a.shape
      const d = a.values()
      return Array.from({ length: r }, (_, i) => d.slice(i * c, (i + 1) * c))
    }
    return [a.values()]
  })
  const labelsV = pop(kw, 'labels')
  const labels = isNone(labelsV) ? [] : h.iterate(labelsV!).map((v) => h.str(v))
  const colorsV = pop(kw, 'colors')
  const colors = isNone(colorsV) ? null : h.iterate(colorsV!)
  const baseline = oneOf(h, optStr(h, pop(kw, 'baseline')) ?? 'zero', 'baseline', ['zero', 'sym', 'wiggle', 'weighted_wiggle'])
  if (baseline === 'wiggle' || baseline === 'weighted_wiggle') throw h.err('NotImplementedError', `stackplot(baseline='${baseline}') is not available in this sandbox`)
  let stack = rows.map((r) => [...r])
  for (let i = 1; i < stack.length; i++) stack[i] = stack[i].map((v, j) => v + stack[i - 1][j])
  let firstLine = x.map(() => 0)
  if (baseline === 'sym') {
    firstLine = x.map((_, j) => -rows.reduce((s, r) => s + r[j], 0) * 0.5)
    stack = stack.map((r) => r.map((v, j) => v + firstLine[j]))
  }
  const firstIdx = ax.children.length
  const out: Value[] = []
  const color = (i: number) => (colors ? colors[i % colors.length] : py.str(ax.nextLineColor()))
  stack.forEach((top, i) => {
    const lower = i === 0 ? firstLine : stack[i - 1]
    const k: Kw = { ...kw, facecolor: color(i) }
    if (labels[i] !== undefined) k.label = py.str(labels[i])
    const coll = fillBetween(true)(ax, [floats(x), floats(lower), floats(top)], k, h)
    if (i === 0 && coll.k === 'obj' && coll.o instanceof ArtistObj) coll.o.sticky.y = [0]
    out.push(coll)
  })
  added(h, ax, firstIdx)
  return py.list(out)
}

const stairs: AxMethod = (ax, args, kw, h) => {
  const vals = nums(h, args[0] ?? pop(kw, 'values')!, 'values')
  const edgesV = args[1] ?? pop(kw, 'edges')
  const edges = isNone(edgesV) ? Array.from({ length: vals.length + 1 }, (_, i) => i) : nums(h, edgesV!, 'edges')
  const orientation = oneOf(h, optStr(h, pop(kw, 'orientation')) ?? 'vertical', 'orientation', ['vertical', 'horizontal'])
  const baseV = pop(kw, 'baseline')
  const baseline = baseV === undefined ? 0 : isNone(baseV) ? null : h.num(baseV)
  const fillOn = flag(h, pop(kw, 'fill'), false)
  if (edges.length !== vals.length + 1) throw h.err('ValueError', `Size mismatch between "values" and "edges". Expected \`len(values) + 1 == len(edges)\`, but \`len(values) = ${vals.length}\` and \`len(edges) = ${edges.length}\`.`)
  const cv = pop(kw, 'color')
  const color = cv && !isNone(cv) ? cv : py.str(ax.nextLineColor())
  const xs: number[] = []
  const ys: number[] = []
  if (baseline !== null) { xs.push(edges[0]); ys.push(baseline) } else { xs.push(edges[0]); ys.push(vals[0]) }
  vals.forEach((v, i) => { xs.push(edges[i], edges[i + 1]); ys.push(v, v) })
  xs.push(edges[edges.length - 1])
  ys.push(baseline ?? vals[vals.length - 1])
  const vertical = orientation === 'vertical'
  const pts = vertical ? xs.flatMap((x, i) => [x, ys[i]]) : xs.flatMap((x, i) => [ys[i], x])
  const o = new PatchObj('StepPatch', { t: 'polygon', pts, closed: fillOn }, h)
  o.fill = fillOn
  if (fillOn) patchKw(o, { facecolor: color, linewidth: py.int(0), ...kw, fill: py.bool(true) }, h)
  else {
    patchKw(o, { edgecolor: color, ...kw, fill: py.bool(false) }, h)
    o.spec.fcs = ['none']
  }
  addPatch(ax, o)
  const b = baseline ?? 0
  if (vertical) {
    o.sticky.y = [b]
    ax.x.update([edges[0]])
    ax.y.update([b])
  } else {
    o.sticky.x = [b]
    ax.x.update([b])
    ax.y.update([edges[0]])
  }
  ax.request()
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

// ---------------------------------------------------------------- line collections

const linesOf = (vertical: boolean): AxMethod => (ax, args, kw, h) => {
  const fn = vertical ? 'vlines' : 'hlines'
  const [posName, loName, hiName] = vertical ? ['x', 'ymin', 'ymax'] : ['y', 'xmin', 'xmax']
  const posV = args[0] ?? pop(kw, posName)
  const loV = args[1] ?? pop(kw, loName)
  const hiV = args[2] ?? pop(kw, hiName)
  const missing = [[posV, posName], [loV, loName], [hiV, hiName]].filter(([v]) => !v).map(([, n]) => `'${n as string}'`)
  if (missing.length) throw h.err('TypeError', `${fn}() missing ${missing.length} required positional argument${missing.length > 1 ? 's' : ''}: ${missing.join(' and ')}`)
  const pos = axisNums(h, vertical ? ax.x : ax.y, posV!, posName)
  const lo = nums(h, loV!, loName)
  const hi = nums(h, hiV!, hiName)
  const n = Math.max(pos.length, lo.length, hi.length)
  const P = broadcast(pos, n)
  const L = broadcast(lo, n)
  const H = broadcast(hi, n)
  const colorsV = args[3] ?? pop(kw, 'colors', 'color')
  const lsV = args[4] ?? pop(kw, 'linestyles', 'linestyle', 'ls')
  const labelV = args[5] ?? pop(kw, 'label')
  const spec = newSegs(colorsV && !isNone(colorsV) ? toColors(h, colorsV) : [rcc(h, 'lines.color')], rcn(h, 'lines.linewidth'))
  spec.paths = P.map((p, i) => (vertical ? [p, L[i], p, H[i]] : [L[i], p, H[i], p]))
  if (lsV) spec.dash = lineStyle(h, lsV).dash
  spec.z = 2
  const o = new SegsObj('LineCollection', spec)
  o.applyKw(kw, h)
  spec.label = isNone(labelV) || (labelV!.k === 'str' && labelV!.v === '') ? null : h.str(labelV!)
  ax.add(o)
  const valid = (i: number) => Number.isFinite(P[i]) && Number.isFinite(L[i]) && Number.isFinite(H[i])
  const goodP = P.filter((_, i) => valid(i))
  const goodV = [...L.filter((_, i) => valid(i)), ...H.filter((_, i) => valid(i))]
  if ((vertical ? spec.tx : spec.tx) === 'data') ax.x.update(vertical ? goodP : goodV)
  if (spec.ty === 'data') ax.y.update(vertical ? goodV : goodP)
  ax.request()
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

const errSpec = (h: Host, v: Value | undefined, n: number, what: string): [number[], number[]] | null => {
  if (isNone(v)) return null
  const a = toArray(v!, 'float64')
  const vals = a.values()
  if (vals.some((e) => e < 0)) throw h.err('ValueError', `'${what}' must not contain negative values`)
  if (a.ndim === 0 || vals.length === 1) return [new Array<number>(n).fill(vals[0]), new Array<number>(n).fill(vals[0])]
  if (a.ndim === 1 && vals.length === n) return [vals, vals]
  if (a.ndim === 2 && a.shape[0] === 2 && a.shape[1] === n) return [vals.slice(0, n), vals.slice(n)]
  throw h.err('ValueError', `'${what}' (shape: (${a.shape.join(', ')}${a.ndim === 1 ? ',' : ''})) must be a scalar or a 1D or (2, n) array-like whose shape matches '${what === 'yerr' ? 'y' : 'x'}' (shape: (${n},))`)
}

const errorbar: AxMethod = (ax, args, kw, h) => {
  const xv = args[0] ?? pop(kw, 'x')
  const yv = args[1] ?? pop(kw, 'y')
  if (!xv || !yv) throw h.err('TypeError', `errorbar() missing required positional arguments`)
  const x = axisNums(h, ax.x, xv, 'errorbar x')
  const y = axisNums(h, ax.y, yv, 'errorbar y')
  if (x.length !== y.length) throw h.err('ValueError', `'x' and 'y' must have the same size`)
  const yerrV = args[2] ?? pop(kw, 'yerr')
  const xerrV = args[3] ?? pop(kw, 'xerr')
  const fmt = optStr(h, args[4] ?? pop(kw, 'fmt')) ?? ''
  const ecolorV = pop(kw, 'ecolor')
  const elw = optNum(h, pop(kw, 'elinewidth'))
  const capsize = optNum(h, pop(kw, 'capsize')) ?? rcn(h, 'errorbar.capsize')
  const barsabove = flag(h, pop(kw, 'barsabove'), false)
  const capthick = optNum(h, pop(kw, 'capthick'))
  const every = pop(kw, 'errorevery')
  for (const k of ['lolims', 'uplims', 'xlolims', 'xuplims']) if (kw[k] && h.truthy(kw[k])) throw h.err('NotImplementedError', `errorbar(${k}=...) is not drawn in this sandbox`)
  for (const k of ['lolims', 'uplims', 'xlolims', 'xuplims']) pop(kw, k)
  const label = pop(kw, 'label')
  const zV = pop(kw, 'zorder')
  const first = ax.children.length
  const lineKw: Kw = { ...kw }
  if (zV) lineKw.zorder = zV
  const lines = plotLines(ax, fmt && fmt.toLowerCase() !== 'none' ? [py.arr(NDArray.create(x, [x.length], 'float64')), py.arr(NDArray.create(y, [y.length], 'float64')), py.str(fmt)] : [py.arr(NDArray.create(x, [x.length], 'float64')), py.arr(NDArray.create(y, [y.length], 'float64'))], lineKw, h)
  const dataLine = lines[0]
  dataLine.spec.z += barsabove ? -0.1 : 0.1
  const color = dataLine.spec.color
  const hasLine = fmt.toLowerCase() !== 'none'
  if (!hasLine) {
    ax.children.splice(ax.children.indexOf(dataLine), 1)
  }
  const ecolor = ecolorV && !isNone(ecolorV) ? toColor(h, ecolorV) : color
  const width = elw ?? (kw.linewidth || kw.lw ? h.num((kw.linewidth ?? kw.lw)!) : rcn(h, 'lines.linewidth'))
  const mew = capthick ?? (kw.markeredgewidth || kw.mew ? h.num((kw.markeredgewidth ?? kw.mew)!) : rcn(h, 'lines.markeredgewidth'))
  const alpha = optNum(h, kw.alpha)
  const n = x.length
  const stepEv = isNone(every) ? 1 : h.toInt(every!)
  const pick = (i: number) => i % stepEv === 0
  const caps: Value[] = []
  const bars: Value[] = []
  const make = (vertical: boolean, err: [number[], number[]]) => {
    const spec = newSegs([ecolor], width)
    spec.z = dataLine.spec.z
    if (zV) spec.z = h.num(zV)
    const segs: Path[] = []
    const low: number[] = []
    const high: number[] = []
    const at: number[] = []
    for (let i = 0; i < n; i++) {
      if (!pick(i)) continue
      const base = vertical ? y[i] : x[i]
      const lo = base - err[0][i]
      const hi = base + err[1][i]
      low.push(lo)
      high.push(hi)
      at.push(vertical ? x[i] : y[i])
      segs.push(vertical ? [x[i], lo, x[i], hi] : [lo, y[i], hi, y[i]])
    }
    spec.paths = segs
    spec.label = '_nolegend_'
    if (alpha !== null) spec.alpha = alpha
    const o = new SegsObj('LineCollection', spec)
    ax.add(o)
    bars.push(py.obj(o))
    ax.x.update(vertical ? at : [...low, ...high])
    ax.y.update(vertical ? [...low, ...high] : at)
    if (capsize > 0) {
      for (const ends of [low, high]) {
        const capLine = new LineObj(newLine(h, vertical ? at : ends, vertical ? ends : at, ecolor))
        capLine.spec.line = false
        capLine.spec.marker = vertical ? '_' : '|'
        capLine.spec.ms = 2 * capsize
        capLine.spec.mew = mew
        capLine.spec.mec = ecolor
        capLine.spec.z = dataLine.spec.z
        capLine.spec.label = '_nolegend_'
        if (alpha !== null) capLine.spec.alpha = alpha
        ax.add(capLine)
        caps.push(py.obj(capLine))
      }
    }
  }
  const ex = errSpec(h, xerrV, n, 'xerr')
  const ey = errSpec(h, yerrV, n, 'yerr')
  if (ex) make(false, ex)
  if (ey) make(true, ey)
  ax.request()
  const lineValue = hasLine ? py.obj(dataLine) : py.NONE
  const c = new ContainerObj('ErrorbarContainer', [lineValue, py.tuple(caps), py.tuple(bars)], { lines: py.tuple([lineValue, py.tuple(caps), py.tuple(bars)]), has_xerr: py.bool(!!ex), has_yerr: py.bool(!!ey) }, isNone(label) ? null : h.str(label!))
  c.ax = ax
  c.entry = { ...dataLine.legendEntry(), line: hasLine && dataLine.spec.line, marker: hasLine ? dataLine.spec.marker : null, bar: ecolor }
  ax.containers.push(c)
  added(h, ax, first)
  return py.obj(c)
}

const stem: AxMethod = (ax, args0, kw, h) => {
  const args = [...args0]
  if (args.length < 1 || args.length > 3) throw h.err('TypeError', `stem() takes 1-3 positional arguments but ${args.length} were given`)
  const linefmtKw = pop(kw, 'linefmt')
  const markerfmtKw = pop(kw, 'markerfmt')
  const basefmtKw = pop(kw, 'basefmt')
  const bottom = optNum(h, pop(kw, 'bottom')) ?? 0
  const label = pop(kw, 'label')
  const orientation = oneOf(h, optStr(h, pop(kw, 'orientation')) ?? 'vertical', 'orientation', ['horizontal', 'vertical'])
  const keys = Object.keys(kw)
  if (keys.length) throw h.err('TypeError', `Axes.stem() got an unexpected keyword argument '${keys[0]}'`)
  let heads: number[]
  let locs: number[]
  let rest: Value[]
  if (args.length === 1) {
    heads = nums(h, args[0], 'heads')
    locs = heads.map((_, i) => i)
    rest = []
  } else if (args[1].k === 'str') {
    heads = nums(h, args[0], 'heads')
    locs = heads.map((_, i) => i)
    rest = args.slice(1)
  } else {
    locs = nums(h, args[0], 'locs')
    heads = nums(h, args[1], 'heads')
    rest = args.slice(2)
  }
  const first = ax.children.length
  const lf = parseFmt(h, linefmtKw ? h.str(linefmtKw) : rest[0] ? h.str(rest[0]) : 'C0-')
  const mfS = markerfmtKw ? h.str(markerfmtKw) : 'o'
  const mf = parseFmt(h, mfS === '' ? ' ' : mfS)
  const bf = parseFmt(h, basefmtKw ? h.str(basefmtKw) : 'C3-')
  const lc = lf.color ?? py.str(CYCLE[0])
  const vertical = orientation === 'vertical'
  const stemlines = linesOf(vertical)(ax, [floats(locs), floats([bottom]), floats(heads)], { colors: lc, linestyles: py.str(lf.ls ?? '-'), label: py.str('_nolegend_') }, h)
  const [mx, my] = vertical ? [locs, heads] : [heads, locs]
  const markerLine = plotLines(ax, [floats(mx), floats(my)], { color: mf.color ?? lc, linestyle: py.str(mf.ls ?? 'None'), marker: py.str(mf.marker && mf.marker !== 'None' ? mf.marker : 'o'), label: py.str('_nolegend_') }, h)[0]
  const span = [minOf(locs), maxOf(locs)]
  const [bx, by] = vertical ? [span, [bottom, bottom]] : [[bottom, bottom], span]
  const baseLine = plotLines(ax, [floats(bx), floats(by)], { color: bf.color ?? py.str(CYCLE[3]), linestyle: py.str(bf.ls ?? '-'), marker: py.str(bf.marker ?? 'None'), label: py.str('_nolegend_') }, h)[0]
  const c = new ContainerObj('StemContainer', [py.obj(markerLine), stemlines, py.obj(baseLine)], { markerline: py.obj(markerLine), stemlines, baseline: py.obj(baseLine) }, isNone(label) ? null : h.str(label!))
  c.ax = ax
  const face = markerLine.spec.color
  c.entry = { ...baseLine.legendEntry(), marker: markerLine.spec.marker, ms: markerLine.spec.ms, mfc: face, mec: face, mew: markerLine.spec.mew, bar: toColor(h, lc) }
  ax.containers.push(c)
  added(h, ax, first)
  return py.obj(c)
}

const brokenBarh: AxMethod = (ax, args, kw, h) => {
  const xr = args[0] ?? pop(kw, 'xranges')
  const yr = args[1] ?? pop(kw, 'yrange')
  if (!xr || !yr) throw h.err('TypeError', "broken_barh() missing required positional arguments: 'xranges' and 'yrange'")
  const ranges = h.iterate(xr).map((r) => nums(h, r, 'xranges'))
  const [y0, yh] = nums(h, yr, 'yrange')
  const spec = newPoly(rcc(h, 'patch.facecolor'), 'none', rcn(h, 'patch.linewidth'))
  spec.paths = ranges.map(([x0, w]) => [x0, y0, x0 + w, y0, x0 + w, y0 + yh, x0, y0 + yh])
  const o = new PolyObj('PolyCollection', spec)
  o.applyKw(kw, h)
  ax.add(o)
  ax.x.update(ranges.flatMap(([x0, w]) => [x0, x0 + w]))
  ax.y.update([y0, y0 + yh])
  ax.request()
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

const eventplot: AxMethod = (ax, args, kw, h) => {
  const pv = args[0] ?? pop(kw, 'positions')
  const orientation = oneOf(h, optStr(h, args[1] ?? pop(kw, 'orientation')) ?? 'horizontal', 'orientation', ['horizontal', 'vertical'])
  if (!pv) throw h.err('TypeError', "eventplot() missing 1 required positional argument: 'positions'")
  const raw = pv.k === 'list' || pv.k === 'tuple' ? pv.items : [pv]
  const groups = raw.length && isNum(raw[0]) ? [nums(h, pv, 'positions')] : raw.map((g) => nums(h, g, 'positions'))
  const off = broadcast(isNone(kw.lineoffsets) ? groups.map((_, i) => i) : nums(h, kw.lineoffsets!, 'lineoffsets'), groups.length)
  if (isNone(kw.lineoffsets) && groups.length === 1) off[0] = 1
  const len = broadcast(isNone(kw.linelengths) ? [1] : nums(h, kw.linelengths!, 'linelengths'), groups.length)
  const lw = broadcast(isNone(kw.linewidths) ? [rcn(h, 'lines.linewidth')] : nums(h, kw.linewidths!, 'linewidths'), groups.length)
  const colorsV = kw.colors ?? kw.color
  const colors = colorsV && !isNone(colorsV) ? toColors(h, colorsV) : [rcc(h, 'lines.color')]
  const lsV = kw.linestyles
  for (const k of ['lineoffsets', 'linelengths', 'linewidths', 'colors', 'color', 'linestyles']) delete kw[k]
  const horizontal = orientation === 'horizontal'
  const firstIdx = ax.children.length
  const out: Value[] = []
  groups.forEach((g, i) => {
    const spec = newSegs([colors[i % colors.length]], lw[i])
    spec.paths = g.map((p) => (horizontal ? [p, off[i] - len[i] / 2, p, off[i] + len[i] / 2] : [off[i] - len[i] / 2, p, off[i] + len[i] / 2, p]))
    if (lsV) spec.dash = lineStyle(h, lsV).dash
    const o = new SegsObj('EventCollection', spec)
    o.applyKw(kw, h)
    ax.add(o)
    out.push(py.obj(o))
  })
  const all = groups.flat()
  if (all.length) {
    const lines = off.flatMap((v, i) => [v - len[i], v + len[i]])
    ax.x.update(horizontal ? [minOf(all), maxOf(all)] : [minOf(lines), maxOf(lines)])
    ax.y.update(horizontal ? [minOf(lines), maxOf(lines)] : [minOf(all), maxOf(all)])
  }
  ax.request()
  added(h, ax, firstIdx)
  return py.list(out)
}

const ecdf: AxMethod = (ax, args, kw, h) => {
  const v = args[0] ?? pop(kw, 'x')
  if (!v) throw h.err('TypeError', "ecdf() missing 1 required positional argument: 'x'")
  if (kw.drawstyle || kw.ds) throw h.err('TypeError', "Cannot pass 'drawstyle' or 'ds' to ecdf()")
  const x = nums(h, v, 'x')
  if (x.some(Number.isNaN)) throw h.err('ValueError', 'ecdf() does not support NaNs')
  if (!isNone(args[1] ?? pop(kw, 'weights'))) throw h.err('NotImplementedError', 'ecdf(weights=...) is not available in this sandbox')
  const comp = flag(h, pop(kw, 'complementary'), false)
  const orientation = oneOf(h, optStr(h, pop(kw, 'orientation')) ?? 'vertical', 'orientation', ['horizontal', 'vertical'])
  if (flag(h, pop(kw, 'compress'), false)) throw h.err('NotImplementedError', 'ecdf(compress=True) is not available in this sandbox')
  const s = [...x].sort((a, b) => a - b)
  const n = s.length
  const cum = s.map((_, i) => (i + 1) / n)
  const first = ax.children.length
  const X = comp ? [...s, s[n - 1]] : [s[0], ...s]
  const Y = comp ? [1, ...cum.map((c) => 1 - c)] : [0, ...cum]
  const vert = orientation === 'vertical'
  const ds = comp ? (vert ? 'steps-pre' : 'steps-post') : vert ? 'steps-post' : 'steps-pre'
  const line = plotLines(ax, vert ? [floats(X), floats(Y)] : [floats(Y), floats(X)], { drawstyle: py.str(ds), ...kw }, h)[0]
  if (vert) line.sticky.y = [0, 1]
  else line.sticky.x = [0, 1]
  added(h, ax, first)
  return py.obj(line)
}

// ---------------------------------------------------------------- pie

const pie: AxMethod = (ax, args, kw, h) => {
  const xv = args[0] ?? pop(kw, 'x')
  if (!xv) throw h.err('TypeError', "pie() missing 1 required positional argument: 'x'")
  const a = toArray(xv, 'float64')
  if (a.ndim > 1) throw h.err('ValueError', 'x must be 1D')
  const x = a.values()
  const explodeV = args[1] ?? pop(kw, 'explode')
  const labelsV = args[2] ?? pop(kw, 'labels')
  const colorsV = args[3] ?? pop(kw, 'colors')
  const autopct = args[4] ?? pop(kw, 'autopct')
  const pctdistance = optNum(h, args[5] ?? pop(kw, 'pctdistance')) ?? 0.6
  const shadow = args[6] ?? pop(kw, 'shadow')
  const labeldistV = args[7] ?? pop(kw, 'labeldistance')
  const labeldistance = labeldistV === undefined ? 1.1 : optNum(h, labeldistV)
  const startangle = optNum(h, args[8] ?? pop(kw, 'startangle')) ?? 0
  const radius = optNum(h, args[9] ?? pop(kw, 'radius')) ?? 1
  const counterclock = flag(h, args[10] ?? pop(kw, 'counterclock'), true)
  const wedgeprops = kwOfDict(h, args[11] ?? pop(kw, 'wedgeprops'))
  const textprops = kwOfDict(h, args[12] ?? pop(kw, 'textprops'))
  const center = isNone(args[13] ?? kw.center) ? [0, 0] : nums(h, (args[13] ?? kw.center)!, 'center')
  pop(kw, 'center')
  const frame = flag(h, args[14] ?? pop(kw, 'frame'), false)
  const rotatelabels = flag(h, args[15] ?? pop(kw, 'rotatelabels'), false)
  const normalize = flag(h, pop(kw, 'normalize'), true)
  pop(kw, 'hatch')
  const keys = Object.keys(kw)
  if (keys.length) throw h.err('TypeError', `Axes.pie() got an unexpected keyword argument '${keys[0]}'`)
  if (shadow && h.truthy(shadow)) throw h.err('NotImplementedError', 'pie(shadow=True) is not drawn in this sandbox')
  if (x.some((v) => v < 0)) throw h.err('ValueError', "Wedge sizes 'x' must be non negative values")
  if (radius <= 0) throw h.err('ValueError', `'radius' must be a positive number, not ${radius}`)
  // x goes through np.float32, and the angles with it (they end up in the label positions)
  const f32 = Math.fround
  const total = x.reduce((s, v) => f32(s + f32(v)), 0)
  if (!normalize && total > 1) throw h.err('ValueError', 'Cannot plot an unnormalized pie with sum(x) > 1')
  const fracs = normalize ? x.map((v) => f32(f32(v) / total)) : x.map(f32)
  const labels = isNone(labelsV) ? x.map(() => '') : h.iterate(labelsV!).map((v) => h.str(v))
  const explode = isNone(explodeV) ? x.map(() => 0) : nums(h, explodeV!, 'explode')
  if (labels.length !== x.length) throw h.err('ValueError', `'labels' must be of length 'x', not ${labels.length}`)
  if (explode.length !== x.length) throw h.err('ValueError', `'explode' must be of length 'x', not ${explode.length}`)
  const colorList = isNone(colorsV) ? null : toColors(h, colorsV!)
  setAspect(h, ax, py.str('equal'))
  const first = ax.children.length
  const slices: Value[] = []
  const texts: Value[] = []
  const autotexts: Value[] = []
  let theta1 = startangle / 360
  fracs.forEach((frac, i) => {
    const theta2 = f32(counterclock ? theta1 + frac : theta1 - frac)
    const thetam = f32(2 * Math.PI * 0.5 * f32(theta1 + theta2))
    const cx = center[0] + explode[i] * Math.cos(thetam)
    const cy = center[1] + explode[i] * Math.sin(thetam)
    const w = new PatchObj('Wedge', { t: 'wedge', x: cx, y: cy, r: radius, t1: 360 * Math.min(theta1, theta2), t2: 360 * Math.max(theta1, theta2), width: null }, h)
    patchKw(w, { facecolor: py.str(colorList ? colorList[i % colorList.length] : ax.nextPatchColor()), clip_on: py.bool(false), label: py.str(labels[i]) }, h)
    w.applyKw(wedgeprops, h)
    addPatch(ax, w)
    slices.push(py.obj(w))
    if (labeldistance !== null) {
      const xt = cx + labeldistance * radius * Math.cos(thetam)
      const yt = cy + labeldistance * radius * Math.sin(thetam)
      const t = newText(h, xt, yt, labels[i])
      t.ha = xt > 0 ? 'left' : 'right'
      t.va = rotatelabels ? (yt > 0 ? 'bottom' : 'top') : 'center'
      if (rotatelabels) t.rotation = (thetam * 180) / Math.PI + (xt > 0 ? 0 : 180)
      t.size = rcFont(h, 'xtick.labelsize')
      const o = new TextObj(t, pyRepr(py.float(xt)), pyRepr(py.float(yt)))
      o.applyKw(textprops, h)
      ax.add(o)
      texts.push(py.obj(o))
    }
    if (!isNone(autopct)) {
      const xt = cx + pctdistance * radius * Math.cos(thetam)
      const yt = cy + pctdistance * radius * Math.sin(thetam)
      const s = autopct!.k === 'str' ? printf(autopct!.v, f32(100 * frac)) : h.str(h.call(autopct!, [py.float(f32(100 * frac))]))
      const t = newText(h, xt, yt, s)
      t.ha = 'center'
      t.va = 'center'
      const o = new TextObj(t, pyRepr(py.float(xt)), pyRepr(py.float(yt)))
      o.applyKw(textprops, h)
      ax.add(o)
      autotexts.push(py.obj(o))
    }
    theta1 = theta2
  })
  if (frame) ax.request()
  else {
    ax.frame = false
    ax.x.ticker.loc = { k: 'fixed', at: [] }
    ax.y.ticker.loc = { k: 'fixed', at: [] }
    ax.x.setLim(h, -1.25 + center[0], 1.25 + center[0], false)
    ax.y.setLim(h, -1.25 + center[1], 1.25 + center[1], false)
  }
  added(h, ax, first)
  return py.tuple(isNone(autopct) ? [py.list(slices), py.list(texts)] : [py.list(slices), py.list(texts), py.list(autotexts)])
}

// ---------------------------------------------------------------- contours

/** matplotlib's MaxNLocator(nbins) with its default steps, as contour uses it to pick levels */
export const maxNLocator = (vmin: number, vmax: number, nbins: number): number[] => {
  if (vmin === vmax) return [vmin]
  return maxNTicks(vmin, vmax, { nbins, steps: MAXN_STEPS, integer: false, minN: 1, prune: null, symmetric: false })
}

/** ContourSet._autolev: the levels span the data, with the locator's excess trimmed */
export const autoLevels = (zmin: number, zmax: number, n: number, extend = 'neither'): number[] => {
  const lev = maxNLocator(zmin, zmax, n + 1)
  let i0 = 0
  for (let i = 0; i < lev.length; i++) if (lev[i] < zmin) i0 = i
  const over = lev.findIndex((l) => l > zmax)
  let i1 = over >= 0 ? over + 1 : lev.length
  if (extend === 'min' || extend === 'both') i0 += 1
  if (extend === 'max' || extend === 'both') i1 -= 1
  if (i1 - i0 < 3) {
    i0 = 0
    i1 = lev.length
  }
  return lev.slice(i0, i1)
}

/** contourf's levels: n bands picked by MaxNLocator(n + 1) (default 7), or the given list */
export const contourLevels = (z: number[], levels: number | number[] | null): number[] => {
  if (Array.isArray(levels)) return levels
  const f = finiteOf(z)
  return autoLevels(minOf(f), maxOf(f), levels ?? 7)
}

class ContourObj extends ArtistObj implements Mapped {
  cls = 'QuadContourSet'
  spec: PolyArtist | SegsArtist
  lut: Lut
  norm: NormObj
  values: number[] | null
  shape: number[]
  levels: number[]
  layers: number[]
  filled: boolean
  fixedColors: string[] | null
  byLevel: Path[][]
  /** level of each drawn path */
  pathLayer: number[] = []
  ext: string
  constructor(spec: PolyArtist | SegsArtist, lut: Lut, norm: NormObj, levels: number[], layers: number[], filled: boolean, fixedColors: string[] | null, byLevel: Path[][], ext: string) {
    super()
    this.spec = spec
    this.lut = lut
    this.norm = norm
    this.levels = levels
    this.layers = layers
    this.values = layers
    this.shape = [layers.length]
    this.filled = filled
    this.fixedColors = fixedColors
    this.byLevel = byLevel
    this.ext = ext
    this.pathLayer = byLevel.flatMap((ps, i) => ps.map(() => i))
    this.spec.paths = byLevel.flat()
    this.recolor()
  }
  layerColor(i: number): string {
    if (this.fixedColors) return this.fixedColors[i % this.fixedColors.length]
    return this.norm.color(this.lut, this.layers[i])
  }
  recolor() {
    const table = this.layers.map((_, i) => this.layerColor(i))
    const cs = this.pathLayer.map((k) => table[k])
    if (this.spec.kind === 'poly') {
      this.spec.fcs = cs.length ? cs : ['none']
      this.spec.ecs = ['none']
    } else this.spec.colors = cs.length ? cs : ['#000000']
  }
  props() { return CONTOUR_PROPS as unknown as Props<ArtistObj> }
  repr() { return `<matplotlib.contour.${this.cls} object at ${addr(this)}>` }
  isa(n: string) { return n === 'Collection' || n === 'ContourSet' }
  legendEntry(): LegendEntry | null { return null }
  getAttr(name: string, h: Host): Value | undefined {
    switch (name) {
      case 'levels': return floats(this.levels)
      case 'layers': return floats(this.layers)
      case 'cvalues': return floats(this.layers)
      case 'filled': return py.bool(this.filled)
      case 'extend': return py.str(this.ext)
      case 'cmap': return py.obj(new ColormapObj(this.lut))
      case 'norm': return py.obj(this.norm)
      case 'allsegs': return py.list(this.byLevel.map((ps) => py.list(ps.map((p) => floats(p, [p.length / 2, 2])))))
      case 'get_paths': return fnv(name, () => py.list(this.byLevel.map((ps) => py.list(ps.map((p) => floats(p, [p.length / 2, 2]))))))
      case 'clabel': return fnv(name, (args, kw) => clabel(this.ax!, [py.obj(this), ...args], { ...kw }, h))
    }
    return super.getAttr(name, h)
  }
}
const CONTOUR_PROPS: Props<ContourObj> = mappedProps<ContourObj>({
  linewidth: { s: (o, [v], h) => { if (o.spec.kind === 'segs') o.spec.widths = nums(h, v, 'linewidths') } },
  linestyle: { s: (o, [v], h) => { if (o.spec.kind === 'segs') o.spec.dash = lineStyle(h, v).dash } },
  linewidths: { s: (o, [v], h) => { if (o.spec.kind === 'segs') o.spec.widths = nums(h, v, 'linewidths') } },
  linestyles: { s: (o, [v], h) => { if (o.spec.kind === 'segs') o.spec.dash = lineStyle(h, v).dash } },
})

/** levels, colours and the artist for contour, contourf, tricontour and tricontourf */
const contourSet = (ax: Axes, h: Host, cls: string, filled: boolean, z: number[], kw: Kw, levelArg: Value | undefined, geo: { lines: (l: number) => Path[]; bands: (ls: number[]) => Path[][] }, bounds: { x: [number, number]; y: [number, number] }): ContourObj => {
  const f = finiteOf(z)
  const zmin = f.length ? minOf(f) : 0
  const zmax = f.length ? maxOf(f) : 0
  const colorsV = pop(kw, 'colors')
  const cmapV = pop(kw, 'cmap')
  if (!isNone(colorsV) && !isNone(cmapV)) throw h.err('ValueError', 'Either colors or cmap must be None')
  const ext = oneOf(h, optStr(h, pop(kw, 'extend')) ?? 'neither', 'extend', ['neither', 'both', 'min', 'max'])
  const locator = pop(kw, 'locator')
  const norm = normArg(h, kw)
  const log = norm.cls === 'LogNorm'
  const lv = pop(kw, 'levels') ?? levelArg
  let levels: number[]
  if (isNone(lv)) levels = locator ? locatorTicks((locator as { k: 'obj'; o: TickerObj }).o.loc!, zmin, zmax, 9, Infinity) : autoLevels(zmin, zmax, 7, ext)
  else if (isNum(lv!)) levels = autoLevels(zmin, zmax, h.toInt(lv!), ext)
  else levels = nums(h, lv!, 'levels')
  if (log && isNone(lv)) {
    const lo = Math.floor(Math.log10(f.filter((v) => v > 0).reduce((a, b) => Math.min(a, b), Infinity)))
    const hi = Math.ceil(Math.log10(zmax))
    levels = Array.from({ length: hi - lo + 1 }, (_, i) => 10 ** (lo + i))
  }
  if (filled && levels.length < 2) throw h.err('ValueError', 'Filled contours require at least 2 levels.')
  if (levels.length > 1 && levels.some((l, i) => i > 0 && l <= levels[i - 1])) throw h.err('ValueError', 'Contour levels must be increasing')
  const allLevels = [...(ext === 'both' || ext === 'min' ? [log ? 1e-250 : -1e250] : []), ...levels, ...(ext === 'both' || ext === 'max' ? [log ? 1e250 : 1e250] : [])]
  const layers = filled ? allLevels.slice(0, -1).map((l, i) => (log ? Math.sqrt(l) * Math.sqrt(allLevels[i + 1]) : 0.5 * (l + allLevels[i + 1]))) : levels
  norm.autoscaleNone(levels)
  if (!filled && norm.vmin === norm.vmax && levels.length === 1) norm.vmin = norm.vmax = levels[0]
  const lut = cmapArg(h, cmapV)
  const fixed = isNone(colorsV) ? null : toColors(h, colorsV!)
  const byLevel: Path[][] = filled ? geo.bands(allLevels) : levels.map((l) => geo.lines(l))
  let spec: PolyArtist | SegsArtist
  const alpha = optNum(h, pop(kw, 'alpha'))
  if (filled) {
    spec = newPoly('none', 'none', 0)
    spec.seamless = true
    spec.closed = true
  } else {
    const lwV = pop(kw, 'linewidths', 'linewidth')
    spec = newSegs([], isNone(lwV) ? rcn(h, 'lines.linewidth') : nums(h, lwV!, 'linewidths')[0])
    const lsV = pop(kw, 'linestyles', 'linestyle')
    if (lsV) spec.dash = lineStyle(h, lsV).dash
    spec.z = 2
  }
  spec.alpha = alpha ?? 1
  for (const k of ['antialiased', 'nchunk', 'corner_mask', 'algorithm', 'hatches', 'negative_linestyles', 'transform_', 'colorizer']) pop(kw, k)
  const o = new ContourObj(spec, lut, norm, levels, layers, filled, fixed, byLevel, ext)
  o.cls = cls
  o.alphaSet = alpha
  // single-coloured line contours draw negative levels dashed
  if (!filled && spec.kind === 'segs' && fixed && fixed.length === 1 && !spec.dash) {
    const dashed = lineStyle(h, rcv(h, 'contour.negative_linestyle')).dash
    const neg = byLevel.map((ps, i) => (levels[i] < 0 ? ps : []))
    const pos = byLevel.map((ps, i) => (levels[i] < 0 ? [] : ps))
    if (neg.some((p) => p.length)) {
      const negSpec = newSegs([fixed[0]], spec.widths[0])
      negSpec.dash = dashed
      negSpec.paths = neg.flat()
      negSpec.z = spec.z
      negSpec.alpha = spec.alpha
      spec.paths = pos.flat()
      o.pathLayer = pos.flatMap((ps, i) => ps.map(() => i))
      o.recolor()
      const negObj = new SegsObj('LineCollection', negSpec)
      negObj.spec.label = '_nolegend_'
      ax.add(negObj)
    }
  }
  o.applyKw(kw, h)
  spec.label = '_nolegend_'
  ax.add(o)
  ax.x.update(bounds.x)
  ax.y.update(bounds.y)
  o.sticky = { x: bounds.x, y: bounds.y }
  ax.autoscaleView(h, true)
  added(h, ax, ax.children.length - 1)
  return o
}

/** Triangulation for the tri* plots */
class TriObj extends PyObj {
  readonly cls = 'Triangulation'
  x: number[]
  y: number[]
  tris: number[][]
  constructor(x: number[], y: number[], tris: number[][]) {
    super()
    this.x = x
    this.y = y
    this.tris = tris
  }
  repr() { return `<matplotlib.tri._triangulation.Triangulation object at ${addr(this)}>` }
  getAttr(name: string): Value | undefined {
    switch (name) {
      case 'x': return floats(this.x)
      case 'y': return floats(this.y)
      case 'triangles': return py.arr(NDArray.create(this.tris.flat(), [this.tris.length, 3], 'int64'))
      case 'edges': {
        const seen = new Map<string, [number, number]>()
        for (const t of this.tris) for (let e = 0; e < 3; e++) { const [p, q] = [t[e], t[(e + 1) % 3]]; seen.set(p < q ? `${p}_${q}` : `${q}_${p}`, p < q ? [p, q] : [q, p]) }
        return py.arr(NDArray.create([...seen.values()].flat(), [seen.size, 2], 'int64'))
      }
    }
    return undefined
  }
}

const triangulation = (h: Host, xv: Value, yv: Value, trisV?: Value): TriObj => {
  const x = nums(h, xv, 'x')
  const y = nums(h, yv, 'y')
  if (x.length !== y.length) throw h.err('ValueError', 'x and y must be equal-length 1D arrays')
  if (x.length < 3) throw h.err('ValueError', 'x and y must be equal-length 1D arrays')
  if (x.length > 5000) throw h.err('ValueError', 'Triangulation: at most 5000 points in this sandbox')
  if (!isNone(trisV)) {
    const a = toArray(trisV!, 'int64')
    const v = a.values()
    return new TriObj(x, y, Array.from({ length: a.shape[0] }, (_, i) => v.slice(i * 3, i * 3 + 3)))
  }
  return new TriObj(x, y, delaunay(x, y))
}

/** the (triangulation, z, rest) a tri* call starts with */
const triArgs = (h: Host, args: Value[], kw: Kw, wantZ = true): { t: TriObj; z: number[]; rest: Value[] } => {
  const a = [...args]
  let t: TriObj
  const first = a[0]
  if (first && first.k === 'obj' && first.o instanceof TriObj) {
    t = first.o
    a.shift()
  } else {
    if (a.length < 2) throw h.err('TypeError', 'tri plots take a Triangulation or x, y')
    const tv = pop(kw, 'triangles')
    t = triangulation(h, a[0], a[1], tv)
    a.splice(0, 2)
    if (a[0] && a[0].k === 'array' && a[0].a.ndim === 2 && a[0].a.shape[1] === 3 && a[0].a.dtype === 'int64' && a[1]) t = triangulation(h, args[0], args[1], a.shift())
  }
  if (!wantZ) return { t, z: [], rest: a }
  const zv = a.shift() ?? pop(kw, 'z')
  if (!zv) return { t, z: [], rest: a }
  const z = nums(h, zv, 'z')
  if (z.length !== t.x.length) throw h.err('ValueError', 'z array must have same length as triangulation x and y arrays')
  return { t, z, rest: a }
}

const tricontour = (filled: boolean): AxMethod => (ax, args, kw, h) => {
  const { t, z, rest } = triArgs(h, args, kw)
  if (!z.length) throw h.err('TypeError', 'tricontour() needs z')
  const c = contourSet(ax, h, 'TriContourSet', filled, z, kw, rest[0], {
    lines: (l) => triLines(t.x, t.y, z, t.tris, l),
    bands: (ls) => triBands(t.x, t.y, z, t.tris, ls),
  }, { x: [minOf(t.x), maxOf(t.x)], y: [minOf(t.y), maxOf(t.y)] })
  return py.obj(c)
}

const triplot: AxMethod = (ax, args, kw, h) => {
  const { t, rest } = triArgs(h, args, kw, false)
  const f = rest[0]?.k === 'str' ? parseFmt(h, rest[0].v) : { ls: null, marker: null, color: null }
  const k: Kw = { ...kw }
  // keyword arguments win over the format string; lines and markers are separate plots, so each takes its own colour
  const color = pop(k, 'color', 'c') ?? f.color
  const lsV = pop(k, 'linestyle', 'ls') ?? (f.ls ? py.str(f.ls) : rcv(h, 'lines.linestyle'))
  const markerV = pop(k, 'marker') ?? (f.marker ? py.str(f.marker) : rcv(h, 'lines.marker'))
  const common: Kw = color && !isNone(color) ? { ...k, color } : { ...k }
  const edges = new Map<string, [number, number]>()
  for (const tri of t.tris) for (let e = 0; e < 3; e++) { const [p, q] = [tri[e], tri[(e + 1) % 3]]; edges.set(p < q ? `${p}_${q}` : `${q}_${p}`, p < q ? [p, q] : [q, p]) }
  const xs: number[] = []
  const ys: number[] = []
  const hasLine = lsV.k !== 'none' && !(lsV.k === 'str' && ['None', '', ' '].includes(lsV.v))
  if (hasLine) for (const [p, q] of edges.values()) {
    xs.push(t.x[p], t.x[q], NaN)
    ys.push(t.y[p], t.y[q], NaN)
  }
  const first = ax.children.length
  const lines = plotLines(ax, [floats(xs), floats(ys)], { zorder: py.int(1), ...common, linestyle: lsV, marker: py.str('None') }, h)
  const hasMarker = markerV.k !== 'none' && !(markerV.k === 'str' && ['None', '', ' '].includes(markerV.v))
  const markerKw: Kw = { ...common, linestyle: py.str('None'), marker: markerV }
  delete markerKw.label
  const pts = plotLines(ax, hasMarker ? [floats(t.x), floats(t.y)] : [floats([]), floats([])], markerKw, h)
  added(h, ax, first)
  return py.list([py.obj(lines[0]), py.obj(pts[0])])
}

const tripcolor: AxMethod = (ax, args, kw, h) => {
  const { t, z } = triArgs(h, args, kw)
  const facecolors = pop(kw, 'facecolors')
  const shading = optStr(h, pop(kw, 'shading')) ?? 'flat'
  if (shading !== 'flat') throw h.err('NotImplementedError', `tripcolor(shading='${shading}') is not drawn in this sandbox`)
  const faceVals = !isNone(facecolors) ? nums(h, facecolors!, 'facecolors') : t.tris.map((tri) => (z[tri[0]] + z[tri[1]] + z[tri[2]]) / 3)
  const cmapV = pop(kw, 'cmap')
  const norm = normArg(h, kw)
  norm.autoscaleNone(faceVals)
  const lut = cmapArg(h, cmapV)
  const spec = newPoly('none', 'none', 0)
  spec.seamless = true
  spec.paths = t.tris.map((tri) => tri.flatMap((i) => [t.x[i], t.y[i]]))
  const o = new PolyObj('TriMesh', spec, lut, norm)
  o.values = faceVals
  o.shape = [faceVals.length]
  o.recolor()
  o.applyKw(kw, h)
  ax.add(o)
  ax.x.update(t.x)
  ax.y.update(t.y)
  ax.autoscaleView(h, null)
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

/** contour's X, Y, Z in any accepted form, flattened row-major */
const contourArgs = (args: Value[], kw: Kw, h: Host): { X: number[]; Y: number[]; Z: number[]; nx: number; ny: number; level?: Value } => {
  const n = args.length
  const zv = n >= 3 ? args[2] : args[0] ?? kw.Z
  if (!zv) throw h.err('TypeError', 'contour() takes contour(Z), contour(X, Y, Z) with optional levels')
  const Za = toArray(zv, 'float64')
  if (Za.ndim !== 2) throw h.err('TypeError', `Input z must be 2D, not ${Za.ndim}D`)
  const [ny, nx] = Za.shape
  if (ny < 2 || nx < 2) throw h.err('TypeError', `Input z must be at least a (2, 2) shaped array, but has shape (${ny}, ${nx})`)
  const Z = Za.values()
  let X: number[]
  let Y: number[]
  if (n >= 3) {
    const xa = toArray(args[0], 'float64')
    const ya = toArray(args[1], 'float64')
    if (xa.ndim !== ya.ndim) throw h.err('TypeError', `Number of dimensions of x (${xa.ndim}) and y (${ya.ndim}) do not match`)
    if (xa.ndim === 1) {
      if (xa.size !== nx) throw h.err('TypeError', `Length of x (${xa.size}) must match number of columns in z (${nx})`)
      if (ya.size !== ny) throw h.err('TypeError', `Length of y (${ya.size}) must match number of rows in z (${ny})`)
      const xs = xa.values()
      const ys = ya.values()
      X = Array.from({ length: nx * ny }, (_, i) => xs[i % nx])
      Y = Array.from({ length: nx * ny }, (_, i) => ys[Math.floor(i / nx)])
    } else {
      if (xa.shape[0] !== ny || xa.shape[1] !== nx) throw h.err('TypeError', `Shapes of x (${xa.shape.join(', ')}) and z (${ny}, ${nx}) do not match`)
      if (ya.shape[0] !== ny || ya.shape[1] !== nx) throw h.err('TypeError', `Shapes of y (${ya.shape.join(', ')}) and z (${ny}, ${nx}) do not match`)
      X = xa.values()
      Y = ya.values()
    }
  } else {
    const origin = optStr(h, pop(kw, 'origin'))
    const extV = pop(kw, 'extent')
    let xs: number[]
    let ys: number[]
    if (origin === null && isNone(extV)) {
      xs = Array.from({ length: nx }, (_, i) => i)
      ys = Array.from({ length: ny }, (_, i) => i)
    } else {
      const e = isNone(extV) ? [0, nx, 0, ny] : nums(h, extV!, 'extent')
      const dx = (e[1] - e[0]) / nx
      const dy = (e[3] - e[2]) / ny
      xs = Array.from({ length: nx }, (_, i) => e[0] + (i + 0.5) * dx)
      ys = Array.from({ length: ny }, (_, i) => e[2] + (i + 0.5) * dy)
      if (origin === 'upper') ys.reverse()
    }
    X = Array.from({ length: nx * ny }, (_, i) => xs[i % nx])
    Y = Array.from({ length: nx * ny }, (_, i) => ys[Math.floor(i / nx)])
  }
  return { X, Y, Z, nx, ny, level: n === 2 || n === 4 ? args[n - 1] : undefined }
}

const contourOf = (filled: boolean): AxMethod => (ax, args, kw, h) => {
  const { X, Y, Z, nx, ny, level } = contourArgs(args, kw, h)
  const fx = finiteOf(X)
  const fy = finiteOf(Y)
  const c = contourSet(ax, h, 'QuadContourSet', filled, Z, kw, level, {
    lines: (l) => gridLines(X, Y, Z, nx, ny, l),
    bands: (ls) => gridBands(X, Y, Z, nx, ny, ls),
  }, { x: [minOf(fx), maxOf(fx)], y: [minOf(fy), maxOf(fy)] })
  return py.obj(c)
}

const clabel: AxMethod = (ax, args, kw, h) => {
  const cv = args[0] ?? pop(kw, 'CS')
  if (!cv || cv.k !== 'obj' || !(cv.o instanceof ContourObj)) throw h.err('TypeError', 'clabel() needs the contour set that contour() returned')
  const cs = cv.o
  const lv = args[1] ?? pop(kw, 'levels')
  const want = isNone(lv) ? cs.levels : nums(h, lv!, 'levels')
  const fontV = pop(kw, 'fontsize')
  const fmtV = pop(kw, 'fmt')
  const colorsV = pop(kw, 'colors')
  const inline = flag(h, pop(kw, 'inline'), true)
  for (const k of ['inline_spacing', 'rightside_up', 'use_clabeltext', 'manual', 'zorder']) pop(kw, k)
  const first = ax.children.length
  const out: Value[] = []
  const fallback = scalarLabels(cs.levels, [minOf(cs.levels), maxOf(cs.levels)]).labels
  cs.byLevel.forEach((paths, i) => {
    const lev = cs.levels[i]
    if (!want.includes(lev)) return
    for (const p of paths.length ? [paths.reduce((a, b) => (b.length > a.length ? b : a))] : []) {
      if (p.length < 8) continue
      const m = Math.floor(p.length / 4) * 2
      const s = fmtV ? (fmtV.k === 'str' ? printf(fmtV.v, lev) : fmtV.k === 'dict' ? h.str(fmtV.d.get(py.float(lev)) ?? py.str(fallback[i])) : h.str(h.call(fmtV, [py.float(lev)]))) : fallback[i]
      const t = newText(h, p[m], p[m + 1], s)
      t.ha = 'center'
      t.va = 'center'
      t.size = fontV ? fontPt(h, fontV) : rcFont(h, 'font.size')
      t.color = colorsV && !isNone(colorsV) ? toColor(h, colorsV) : cs.layerColor(i)
      if (inline) t.bbox = { fc: ax.facecolor, ec: 'none', lw: 0, pad: 1, round: false, alpha: 1 }
      const o = new TextObj(t, String(t.x), String(t.y))
      ax.add(o)
      out.push(py.obj(o))
    }
  })
  added(h, ax, first)
  return py.obj(new SilentList('Text', out))
}

// ---------------------------------------------------------------- meshes

const meshArgs = (h: Host, args: Value[], kw: Kw, fn: string): { X: number[]; Y: number[]; C: number[]; nx: number; ny: number } => {
  const n = args.length
  if (n !== 1 && n !== 3) throw h.err('TypeError', `${fn}() takes 1 or 3 positional arguments but ${n} were given`)
  const cv = n === 3 ? args[2] : args[0] ?? pop(kw, 'C')
  if (!cv) throw h.err('TypeError', `${fn}() missing C`)
  const Ca = toArray(cv, 'float64')
  if (Ca.ndim !== 2) {
    if (n === 1) throw h.err('ValueError', `not enough values to unpack (expected 2, got ${Ca.ndim})`)
    throw h.err('TypeError', `Dimensions of C (${Ca.shape.join(', ')}) are incompatible with X and Y`)
  }
  const [rows, cols] = Ca.shape
  const C = Ca.values()
  const shading = optStr(h, pop(kw, 'shading')) ?? rcs(h, 'pcolor.shading')
  oneOf(h, shading, 'shading', ['gouraud', 'nearest', 'flat', 'auto'])
  if (shading === 'gouraud') throw h.err('NotImplementedError', "pcolormesh(shading='gouraud') is not drawn in this sandbox: use 'flat' or 'nearest'")
  let xs: number[]
  let ys: number[]
  if (n === 1) {
    xs = Array.from({ length: (rows + 1) * (cols + 1) }, (_, i) => i % (cols + 1))
    ys = Array.from({ length: (rows + 1) * (cols + 1) }, (_, i) => Math.floor(i / (cols + 1)))
    return { X: xs, Y: ys, C, nx: cols, ny: rows }
  }
  const xa = toArray(args[0], 'float64')
  const ya = toArray(args[1], 'float64')
  const [xc, yr] = xa.ndim === 1 ? [xa.size, ya.size] : [xa.shape[1], xa.shape[0]]
  const mid = (shading === 'nearest' || shading === 'auto') && xc === cols && yr === rows
  if ((shading === 'flat' || (shading === 'auto' && !mid)) && (xc !== cols + 1 || yr !== rows + 1)) throw h.err('TypeError', `Dimensions of C (${rows}, ${cols}) should be one smaller than X(${xc}) and Y(${yr}) while using shading='flat' see help(pcolormesh)`)
  if (shading === 'nearest' && !mid) throw h.err('TypeError', `Dimensions of C (${rows}, ${cols}) are incompatible with X (${xc}) and/or Y (${yr}); see help(pcolormesh)`)
  const toGrid = (a: NDArray, vertical: boolean): number[] => {
    if (a.ndim === 2) return a.values()
    const v = a.values()
    return vertical ? Array.from({ length: yr * xc }, (_, i) => v[Math.floor(i / xc)]) : Array.from({ length: yr * xc }, (_, i) => v[i % xc])
  }
  xs = toGrid(xa, false)
  ys = toGrid(ya, true)
  if (mid) {
    // cell edges halfway between the given centres, extrapolated at the ends
    const edge = (src: number[], rowsIn: number, colsIn: number): number[] => {
      const at = (r: number, c: number) => src[r * colsIn + c]
      const expandC = Array.from({ length: rowsIn }, (_, r) => {
        const row = Array.from({ length: colsIn }, (_, c) => at(r, c))
        const d = row.slice(1).map((v, i) => (v - row[i]) / 2)
        if (colsIn === 1) return [row[0] - 0.5, row[0] + 0.5]
        return [row[0] - d[0], ...row.slice(0, -1).map((v, i) => v + d[i]), row[colsIn - 1] + d[d.length - 1]]
      })
      const newC = colsIn + 1
      const out: number[] = []
      for (let c = 0; c < newC; c++) {
        const col = expandC.map((r) => r[c])
        const d = col.slice(1).map((v, i) => (v - col[i]) / 2)
        const full = rowsIn === 1 ? [col[0] - 0.5, col[0] + 0.5] : [col[0] - d[0], ...col.slice(0, -1).map((v, i) => v + d[i]), col[rowsIn - 1] + d[d.length - 1]]
        full.forEach((v, r) => { out[r * newC + c] = v })
      }
      return out
    }
    xs = edge(xs, yr, xc)
    ys = edge(ys, yr, xc)
  }
  return { X: xs, Y: ys, C, nx: cols, ny: rows }
}

const uniformGrid = (X: number[], Y: number[], nx: number, ny: number): boolean => {
  const sx = nx + 1
  const eq = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol
  const dx = X[1] - X[0]
  const dy = Y[sx] - Y[0]
  if (!dx || !dy) return false
  const tx = Math.abs(dx) * 1e-6
  const ty = Math.abs(dy) * 1e-6
  for (let r = 0; r <= ny; r++) for (let c = 0; c <= nx; c++) {
    if (!eq(X[r * sx + c], X[0] + c * dx, tx) || !eq(Y[r * sx + c], Y[0] + r * dy, ty)) return false
  }
  return true
}

const meshOf = (ax: Axes, h: Host, X: number[], Y: number[], C: number[], nx: number, ny: number, kw: Kw, shape: number[]): MeshObj => {
  const cmapV = pop(kw, 'cmap')
  const norm = normArg(h, kw)
  const lut = cmapArg(h, cmapV)
  norm.autoscaleNone(C)
  const edgeV = pop(kw, 'edgecolors', 'edgecolor', 'ec')
  const lwV = pop(kw, 'linewidth', 'linewidths', 'lw')
  const alpha = optNum(h, pop(kw, 'alpha'))
  for (const k of ['antialiased', 'antialiaseds', 'snap', 'rasterized', 'colorizer']) pop(kw, k)
  const spec: MeshArtist = {
    kind: 'mesh', ...artistBase(1), nx, ny, X, Y, colors: [], alpha: alpha ?? 1, uniform: uniformGrid(X, Y, nx, ny), edge: isNone(edgeV) || (edgeV!.k === 'str' && edgeV!.v === 'none') ? null : edgeV!.k === 'str' && edgeV!.v === 'face' ? null : toColor(h, edgeV!, 'edgecolor'), lw: optNum(h, lwV) ?? 1,
  }
  const o = new MeshObj(spec, C, shape, lut, norm)
  o.alphaSet = alpha
  o.recolor()
  o.applyKw(kw, h)
  ax.add(o)
  const fx = finiteOf(X)
  const fy = finiteOf(Y)
  const bx: [number, number] = [minOf(fx), maxOf(fx)]
  const by: [number, number] = [minOf(fy), maxOf(fy)]
  o.sticky = { x: bx, y: by }
  ax.x.update(bx)
  ax.y.update(by)
  ax.request()
  return o
}

const pcolormesh = (name: string): AxMethod => (ax, args, kw, h) => {
  const { X, Y, C, nx, ny } = meshArgs(h, args, kw, name)
  const o = meshOf(ax, h, X, Y, C, nx, ny, kw, [ny, nx])
  if (name === 'pcolor') o.cls = 'PolyQuadMesh'
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

const hist2d: AxMethod = (ax, args, kw, h) => {
  const xv = args[0] ?? pop(kw, 'x')
  const yv = args[1] ?? pop(kw, 'y')
  if (!xv || !yv) throw h.err('TypeError', "hist2d() missing required positional arguments: 'x' and 'y'")
  const x = nums(h, xv, 'x')
  const y = nums(h, yv, 'y')
  if (x.length !== y.length) throw h.err('ValueError', 'x and y must have the same length.')
  const bv = args[2] ?? pop(kw, 'bins') ?? py.int(10)
  const rv = args[3] ?? pop(kw, 'range')
  const density = flag(h, args[4] ?? pop(kw, 'density'), false)
  const wv = args[5] ?? pop(kw, 'weights')
  const cmin = optNum(h, args[6] ?? pop(kw, 'cmin'))
  const cmax = optNum(h, args[7] ?? pop(kw, 'cmax'))
  const w = isNone(wv) ? null : nums(h, wv!, 'weights')
  const bx = (v: Value, data: number[], r?: number[]): number[] => {
    if (isNum(v)) {
      const n = h.toInt(v)
      if (n < 1) throw h.err('ValueError', '`bins` must be positive, when an integer')
      return histogram(data, n, r ? [r[0], r[1]] : undefined).edges
    }
    return nums(h, v, 'bins')
  }
  let [binsX, binsY] = [bv, bv]
  if ((bv.k === 'list' || bv.k === 'tuple') && bv.items.length === 2) [binsX, binsY] = bv.items
  const rng = isNone(rv) ? [undefined, undefined] : h.iterate(rv!).map((p) => nums(h, p, 'range'))
  const ex = bx(binsX, x, rng[0])
  const ey = bx(binsY, y, rng[1])
  const nx = ex.length - 1
  const ny = ey.length - 1
  const counts = new Array<number>(nx * ny).fill(0)
  const find = (edges: number[], v: number) => {
    if (!(v >= edges[0] && v <= edges[edges.length - 1])) return -1
    let lo = 0
    let hi = edges.length - 1
    while (hi - lo > 1) {
      const m = (lo + hi) >> 1
      if (v >= edges[m]) lo = m
      else hi = m
    }
    return lo
  }
  x.forEach((vx, i) => {
    const a = find(ex, vx)
    const b = find(ey, y[i])
    if (a >= 0 && b >= 0) counts[a * ny + b] += w ? w[i] : 1
  })
  let H = counts
  if (density) {
    const total = counts.reduce((p, q) => p + q, 0)
    H = counts.map((c, k) => c / total / ((ex[Math.floor(k / ny) + 1] - ex[Math.floor(k / ny)]) * (ey[(k % ny) + 1] - ey[k % ny])))
  }
  const masked = H.map((c) => ((cmin !== null && c < cmin) || (cmax !== null && c > cmax) ? NaN : c))
  // pcolormesh(xedges, yedges, h.T): columns are x
  const C = Array.from({ length: nx * ny }, (_, k) => masked[(k % nx) * ny + Math.floor(k / nx)])
  const X = Array.from({ length: (nx + 1) * (ny + 1) }, (_, k) => ex[k % (nx + 1)])
  const Y = Array.from({ length: (nx + 1) * (ny + 1) }, (_, k) => ey[Math.floor(k / (nx + 1))])
  const o = meshOf(ax, h, X, Y, C, nx, ny, kw, [ny, nx])
  ax.x.setLim(h, ex[0], ex[nx], false)
  ax.y.setLim(h, ey[0], ey[ny], false)
  added(h, ax, ax.children.length - 1)
  return py.tuple([floats(masked, [nx, ny]), floats(ex), floats(ey), py.obj(o)])
}

const hexbin: AxMethod = (ax, args, kw, h) => {
  const xv = args[0] ?? pop(kw, 'x')
  const yv = args[1] ?? pop(kw, 'y')
  if (!xv || !yv) throw h.err('TypeError', "hexbin() missing required positional arguments: 'x' and 'y'")
  const x = nums(h, xv, 'x')
  const y = nums(h, yv, 'y')
  if (x.length !== y.length) throw h.err('ValueError', `operands could not be broadcast together with shapes (${x.length},) (${y.length},) `)
  const Cv = args[2] ?? pop(kw, 'C')
  if (!isNone(Cv)) throw h.err('NotImplementedError', 'hexbin(C=...) is not available in this sandbox: it bins counts only')
  const gridV = pop(kw, 'gridsize') ?? py.int(100)
  const binsV = pop(kw, 'bins')
  const xscale = optStr(h, pop(kw, 'xscale')) ?? 'linear'
  const yscale = optStr(h, pop(kw, 'yscale')) ?? 'linear'
  if (xscale !== 'linear' || yscale !== 'linear') throw h.err('NotImplementedError', 'hexbin with log scales is not available in this sandbox')
  const mincnt = optNum(h, pop(kw, 'mincnt'))
  const extentV = pop(kw, 'extent')
  let nxG: number
  let nyG: number
  if (isNum(gridV)) {
    nxG = h.toInt(gridV)
    nyG = Math.trunc(nxG / Math.sqrt(3))
  } else [nxG, nyG] = h.iterate(gridV).map((v) => h.toInt(v))
  let [xmin, xmax] = x.length ? [minOf(x), maxOf(x)] : [0, 1]
  let [ymin, ymax] = y.length ? [minOf(y), maxOf(y)] : [0, 1]
  if (!isNone(extentV)) [xmin, xmax, ymin, ymax] = nums(h, extentV!, 'extent')
  ;[xmin, xmax] = nonsingular(xmin, xmax, 0.1)
  ;[ymin, ymax] = nonsingular(ymin, ymax, 0.1)
  const nx1 = nxG + 1
  const ny1 = nyG + 1
  const padding = 1e-9 * (xmax - xmin)
  xmin -= padding
  xmax += padding
  const sx = (xmax - xmin) / nxG
  const sy = (ymax - ymin) / nyG
  const c1 = new Array<number>(nx1 * ny1).fill(0)
  const c2 = new Array<number>(nxG * nyG).fill(0)
  x.forEach((vx, i) => {
    const ix = (vx - xmin) / sx
    const iy = (y[i] - ymin) / sy
    const ix1 = pyRound(ix)
    const iy1 = pyRound(iy)
    const ix2 = Math.floor(ix)
    const iy2 = Math.floor(iy)
    const d1 = (ix - ix1) ** 2 + 3 * (iy - iy1) ** 2
    const d2 = (ix - ix2 - 0.5) ** 2 + 3 * (iy - iy2 - 0.5) ** 2
    if (d1 < d2) {
      if (ix1 >= 0 && ix1 < nx1 && iy1 >= 0 && iy1 < ny1) c1[ix1 * ny1 + iy1]++
    } else if (ix2 >= 0 && ix2 < nxG && iy2 >= 0 && iy2 < nyG) c2[ix2 * nyG + iy2]++
  })
  const hex = [[0.5, -0.5], [0.5, 0.5], [0, 1], [-0.5, 0.5], [-0.5, -0.5], [0, -1]].map(([u, v]) => [u * sx, (v * sy) / 3])
  const paths: Path[] = []
  const accum: number[] = []
  const put = (cnt: number, ox: number, oy: number) => {
    if (mincnt !== null && cnt < mincnt) return
    paths.push(hex.flatMap(([u, v]) => [ox * sx + xmin + u, oy * sy + ymin + v]))
    accum.push(cnt)
  }
  for (let a = 0; a < nx1; a++) for (let b = 0; b < ny1; b++) put(c1[a * ny1 + b], a, b)
  for (let a = 0; a < nxG; a++) for (let b = 0; b < nyG; b++) put(c2[a * nyG + b], a + 0.5, b + 0.5)
  const vals = binsV && binsV.k === 'str' && binsV.v === 'log' ? accum.map((v) => v) : accum
  const cmapV = pop(kw, 'cmap')
  if (binsV && binsV.k === 'str' && binsV.v === 'log') kw.norm = py.obj(new NormObj('LogNorm', null, null))
  else if (!isNone(binsV)) throw h.err('NotImplementedError', "hexbin(bins=...): only None and 'log' work in this sandbox")
  const norm = normArg(h, kw)
  norm.autoscaleNone(vals)
  const lut = cmapArg(h, cmapV)
  const spec = newPoly('none', 'none', 0)
  spec.seamless = true
  spec.paths = paths
  const o = new PolyObj('PolyCollection', spec, lut, norm)
  const edgeV = pop(kw, 'edgecolors', 'edgecolor', 'ec')
  const lwV = pop(kw, 'linewidths', 'linewidth', 'lw')
  o.values = vals
  o.shape = [vals.length]
  o.recolor()
  if (!isNone(edgeV) && !(edgeV!.k === 'str' && edgeV!.v === 'face')) {
    spec.ecs = toColors(h, edgeV!, 'edgecolor')
    spec.seamless = false
    spec.lw = optNum(h, lwV) ?? 1
  }
  o.applyKw(kw, h)
  ax.add(o)
  ax.x.update([xmin, xmax])
  ax.y.update([ymin, ymax])
  ax.tight = true
  ax.request()
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

// ---------------------------------------------------------------- statistics plots

/** boxplot statistics as matplotlib computes them (NumPy's linear percentiles, whis = 1.5 by default) */
export const boxStats = (data: number[], pos: number, whis: number | [number, number] = 1.5) => {
  const d = data.filter(Number.isFinite)
  const s = [...d].sort((a, b) => a - b)
  const q = (p: number) => {
    const i = (s.length - 1) * p
    const lo = Math.floor(i)
    return lo + 1 < s.length ? s[lo] + (s[lo + 1] - s[lo]) * (i - lo) : s[lo]
  }
  const q1 = q(0.25)
  const q3 = q(0.75)
  const iqr = q3 - q1
  const loval = Array.isArray(whis) ? q(whis[0] / 100) : q1 - whis * iqr
  const hival = Array.isArray(whis) ? q(whis[1] / 100) : q3 + whis * iqr
  const hiIn = d.filter((v) => v <= hival)
  const loIn = d.filter((v) => v >= loval)
  const hi = !hiIn.length || maxOf(hiIn) < q3 ? q3 : maxOf(hiIn)
  const lo = !loIn.length || minOf(loIn) > q1 ? q1 : minOf(loIn)
  return { pos, q1, med: q(0.5), q3, lo, hi, fliers: [...d.filter((v) => v < lo), ...d.filter((v) => v > hi)] }
}

const boxplot: AxMethod = (ax, args, kw, h) => {
  const d = args[0] ?? pop(kw, 'x')
  if (!d) throw h.err('TypeError', "boxplot() missing 1 required positional argument: 'x'")
  const groups = datasets(h, d, 'x')
  const notch = flag(h, args[1] ?? pop(kw, 'notch'), rcb(h, 'boxplot.notch'))
  const sym = optStr(h, args[2] ?? pop(kw, 'sym'))
  const vertV = args[3] ?? pop(kw, 'vert')
  const orientation = oneOf(h, optStr(h, pop(kw, 'orientation')) ?? 'vertical', 'orientation', ['horizontal', 'vertical'])
  const vert = vertV ? h.truthy(vertV) && orientation === 'vertical' : orientation === 'vertical'
  const whisV = args[4] ?? pop(kw, 'whis')
  const whis: number | [number, number] = isNone(whisV) ? rcn(h, 'boxplot.whiskers') : isNum(whisV!) ? h.num(whisV!) : (nums(h, whisV!, 'whis') as [number, number])
  const posV = args[5] ?? pop(kw, 'positions')
  const widthsV = args[6] ?? pop(kw, 'widths')
  const patchArtist = flag(h, args[7] ?? pop(kw, 'patch_artist'), rcb(h, 'boxplot.patchartist'))
  if (!isNone(args[8] ?? pop(kw, 'bootstrap'))) throw h.err('NotImplementedError', 'boxplot(bootstrap=...) is not available in this sandbox')
  if (!isNone(pop(kw, 'usermedians')) || !isNone(pop(kw, 'conf_intervals'))) throw h.err('NotImplementedError', 'boxplot(usermedians / conf_intervals) is not available in this sandbox')
  const meanline = flag(h, pop(kw, 'meanline'), false)
  const showmeans = flag(h, pop(kw, 'showmeans'), false)
  const showcaps = flag(h, pop(kw, 'showcaps'), true)
  const showbox = flag(h, pop(kw, 'showbox'), true)
  let showfliers = flag(h, pop(kw, 'showfliers'), true)
  const boxprops = kwOfDict(h, pop(kw, 'boxprops'))
  const labelsV = pop(kw, 'tick_labels', 'labels')
  const flierprops = kwOfDict(h, pop(kw, 'flierprops'))
  const medianprops = kwOfDict(h, pop(kw, 'medianprops'))
  const meanprops = kwOfDict(h, pop(kw, 'meanprops'))
  const capprops = kwOfDict(h, pop(kw, 'capprops'))
  const whiskerprops = kwOfDict(h, pop(kw, 'whiskerprops'))
  const manage = flag(h, pop(kw, 'manage_ticks'), true)
  const autorange = flag(h, pop(kw, 'autorange'), false)
  const zorder = optNum(h, pop(kw, 'zorder')) ?? 2
  const capwV = pop(kw, 'capwidths')
  const labelV = pop(kw, 'label')
  const keys = Object.keys(kw)
  if (keys.length) throw h.err('TypeError', `Axes.boxplot() got an unexpected keyword argument '${keys[0]}'`)
  const N = groups.length
  const pos = isNone(posV) ? groups.map((_, i) => i + 1) : nums(h, posV!, 'positions')
  const posInt = isNone(posV) || toArray(posV!).dtype === 'int64'
  const msg = (what: string) => h.err('ValueError', `List of boxplot statistics and \`${what}\` values must have same the length`)
  if (pos.length !== N) throw msg('positions')
  const labels = isNone(labelsV) ? null : h.iterate(labelsV!).map((v) => h.str(v))
  if (labels && labels.length !== N) throw h.err('ValueError', 'Dimensions of labels and X must be compatible')
  const span = N ? maxOf(pos) - minOf(pos) : 0
  const dflt = Math.max(0.15, Math.min(0.5, 0.15 * span))
  let widths: number[] = new Array<number>(N).fill(dflt)
  if (!isNone(widthsV)) {
    widths = isNum(widthsV!) ? new Array<number>(N).fill(h.num(widthsV!)) : nums(h, widthsV!, 'widths')
    if (widths.length !== N) throw msg('widths')
  }
  let capw = widths.map((w) => 0.5 * w)
  if (!isNone(capwV)) {
    capw = isNum(capwV!) ? new Array<number>(N).fill(h.num(capwV!)) : nums(h, capwV!, 'capwidths')
    if (capw.length !== N) throw msg('capwidths')
  }
  const rcKw = (sub: string, marker: boolean): Kw => {
    const out: Kw = { color: rcv(h, `boxplot.${sub}props.color`), linestyle: rcv(h, `boxplot.${sub}props.linestyle`), linewidth: rcv(h, `boxplot.${sub}props.linewidth`) }
    if (marker) {
      out.marker = rcv(h, `boxplot.${sub}props.marker`)
      out.markerfacecolor = rcv(h, `boxplot.${sub}props.markerfacecolor`)
      out.markeredgecolor = rcv(h, `boxplot.${sub}props.markeredgecolor`)
      out.markersize = rcv(h, `boxplot.${sub}props.markersize`)
      if (sub === 'flier') out.markeredgewidth = rcv(h, 'boxplot.flierprops.markeredgewidth')
    } else out.marker = py.str('')
    return out
  }
  let flierExtra = flierprops
  if (sym !== null) {
    if (sym === '') showfliers = false
    else {
      const f = parseFmt(h, sym)
      flierExtra = { ...flierprops }
      if (f.marker !== null && f.marker !== 'None') flierExtra.marker = py.str(f.marker)
      if (f.color) {
        flierExtra.color = f.color
        flierExtra.markerfacecolor = f.color
        flierExtra.markeredgecolor = f.color
      }
    }
  }
  const whiskerKw: Kw = { ...rcKw('whisker', false), zorder: py.float(zorder), label: py.str('_nolegend_'), ...whiskerprops }
  const capKw: Kw = { ...rcKw('cap', false), zorder: py.float(zorder), label: py.str('_nolegend_'), ...capprops }
  const flierKw: Kw = { ...rcKw('flier', true), zorder: py.float(zorder), label: py.str('_nolegend_'), ...flierExtra }
  const medianKw: Kw = { ...rcKw('median', false), zorder: py.float(zorder + 0.1), label: py.str('_nolegend_'), ...medianprops }
  const meanKw: Kw = { ...rcKw('mean', true), zorder: py.float(zorder + 0.1), ...meanprops }
  if (!('linestyle' in meanprops) && !meanline) meanKw.linestyle = py.str('')
  if (!('marker' in meanprops) && meanline) meanKw.marker = py.str('')
  const boxLineKw: Kw = { ...rcKw('box', false), zorder: py.float(zorder), ...boxprops }
  const swap = (xs: number[], ys: number[]): [Value, Value] => (vert ? [floats(xs), floats(ys)] : [floats(ys), floats(xs)])
  const plotIt = (xs: number[], ys: number[], k: Kw): LineObj => plotLines(ax, swap(xs, ys), k, h)[0]
  const first = ax.children.length
  const whiskers: Value[] = []
  const caps: Value[] = []
  const boxes: Value[] = []
  const medians: Value[] = []
  const means: Value[] = []
  const fliers: Value[] = []
  groups.forEach((g, i) => {
    const p = pos[i]
    const st = boxStats(g, p, autorange && boxStats(g, p, whis).q1 === boxStats(g, p, whis).q3 ? [0, 100] : whis)
    const f = g.filter(Number.isFinite)
    const iqr = st.q3 - st.q1
    const w = widths[i]
    const [L, R] = [p - w * 0.5, p + w * 0.5]
    const cilo = st.med - (1.57 * iqr) / Math.sqrt(f.length)
    const cihi = st.med + (1.57 * iqr) / Math.sqrt(f.length)
    let bx: number[]
    let by: number[]
    let medX: number[]
    if (notch) {
      const nl = p - w * 0.25
      const nr = p + w * 0.25
      bx = [L, R, R, nr, R, R, L, L, nl, L, L]
      by = [st.q1, st.q1, cilo, st.med, cihi, st.q3, st.q3, cihi, st.med, cilo, st.q1]
      medX = [nl, nr]
    } else {
      bx = [L, R, R, L, L]
      by = [st.q1, st.q1, st.q3, st.q3, st.q1]
      medX = [L, R]
    }
    if (showbox) {
      if (patchArtist) {
        const pts = vert ? bx.flatMap((v, k) => [v, by[k]]) : bx.flatMap((v, k) => [by[k], v])
        const o = new PatchObj('PathPatch', { t: 'polygon', pts, closed: true }, h)
        const pk: Kw = { linestyle: rcv(h, 'boxplot.boxprops.linestyle'), linewidth: rcv(h, 'boxplot.boxprops.linewidth'), edgecolor: rcv(h, 'boxplot.boxprops.color'), facecolor: rcv(h, 'patch.facecolor'), zorder: py.float(zorder), ...boxprops }
        if ('color' in boxprops) { delete pk.edgecolor; delete pk.facecolor; if ('facecolor' in boxprops) pk.facecolor = boxprops.facecolor }
        const colorArg = pop(pk, 'color')
        const faceArg = pop(pk, 'facecolor', 'fc')
        patchKw(o, { ...pk, ...(colorArg ? { edgecolor: colorArg, ...(faceArg ? {} : { facecolor: colorArg }) } : {}), ...(faceArg ? { facecolor: faceArg } : {}) }, h)
        o.spec.label = '_nolegend_'
        ax.add(o)
        boxes.push(py.obj(o))
      } else boxes.push(py.obj(plotIt(bx, by, boxLineKw)))
    }
    whiskers.push(py.obj(plotIt([p, p], [st.q1, st.lo], whiskerKw)), py.obj(plotIt([p, p], [st.q3, st.hi], whiskerKw)))
    if (showcaps) {
      const cw = capw[i] * 0.5
      caps.push(py.obj(plotIt([p - cw, p + cw], [st.lo, st.lo], capKw)), py.obj(plotIt([p - cw, p + cw], [st.hi, st.hi], capKw)))
    }
    const med = plotIt(medX, [st.med, st.med], medianKw)
    if (vert) med.sticky.x = [p - 0.5, p + 0.5]
    else med.sticky.y = [p - 0.5, p + 0.5]
    medians.push(py.obj(med))
    if (showmeans) {
      const mean = f.reduce((a, b) => a + b, 0) / f.length
      means.push(py.obj(meanline ? plotIt([L, R], [mean, mean], meanKw) : plotIt([p], [mean], meanKw)))
    }
    if (showfliers) fliers.push(py.obj(plotIt(st.fliers.map(() => p), st.fliers, flierKw)))
  })
  if (patchArtist) ax.request()
  if (!isNone(labelV)) {
    const target = showbox && patchArtist ? boxes : medians
    const s = labelV!.k === 'list' || labelV!.k === 'tuple' ? labelV!.items.map((v) => h.str(v)) : [h.str(labelV!)]
    s.forEach((l, i) => { const v = target[i]; if (v && v.k === 'obj' && v.o instanceof ArtistObj) v.o.setLabel(l) })
  }
  if (manage && N) {
    const axis = vert ? ax.x : ax.y
    axis.update([minOf(pos) - 0.5, maxOf(pos) + 0.5])
    const prevAt = axis.ticker.loc.k === 'fixed' ? axis.ticker.loc.at : []
    const prevLabels = axis.ticker.fmt.k === 'fixed' ? axis.ticker.fmt.labels : []
    axis.ticker.loc = { k: 'fixed', at: [...prevAt, ...pos], dtype: posInt ? 'int64' : 'float64' }
    axis.ticker.fmt = { k: 'fixed', labels: [...prevLabels, ...pos.map((p, i) => labels?.[i] ?? (posInt ? String(p) : pyRepr(py.float(p))))] }
    ax.request()
  }
  added(h, ax, first)
  return dictOf([['whiskers', py.list(whiskers)], ['caps', py.list(caps)], ['boxes', py.list(boxes)], ['medians', py.list(medians)], ['fliers', py.list(fliers)], ['means', py.list(means)]])
}

const violinplot: AxMethod = (ax, args, kw, h) => {
  const d = args[0] ?? pop(kw, 'dataset')
  if (!d) throw h.err('TypeError', "violinplot() missing 1 required positional argument: 'dataset'")
  const groups = datasets(h, d, 'dataset')
  if (groups.some((g) => !g.length)) throw h.err('ValueError', 'zero-size array to reduction operation minimum which has no identity')
  const posV = args[1] ?? pop(kw, 'positions')
  const vertV = args[2] ?? pop(kw, 'vert')
  const orientation = oneOf(h, optStr(h, pop(kw, 'orientation')) ?? 'vertical', 'orientation', ['vertical', 'horizontal'])
  const vert = vertV ? h.truthy(vertV) && orientation === 'vertical' : orientation === 'vertical'
  const widthsV = args[3] ?? pop(kw, 'widths')
  const showmeans = flag(h, args[4] ?? pop(kw, 'showmeans'), false)
  const showextrema = flag(h, args[5] ?? pop(kw, 'showextrema'), true)
  const showmedians = flag(h, args[6] ?? pop(kw, 'showmedians'), false)
  const quantV = pop(kw, 'quantiles')
  if (!isNone(quantV)) throw h.err('NotImplementedError', 'violinplot(quantiles=...) is not available in this sandbox')
  const points = h.toInt(pop(kw, 'points') ?? py.int(100))
  const bwV = pop(kw, 'bw_method')
  if (!isNone(bwV) && !(bwV!.k === 'str' && bwV!.v === 'scott')) throw h.err('NotImplementedError', "violinplot(bw_method=...): only 'scott' works in this sandbox")
  const keys = Object.keys(kw)
  if (keys.length) throw h.err('TypeError', `Axes.violinplot() got an unexpected keyword argument '${keys[0]}'`)
  const N = groups.length
  const pos = isNone(posV) ? groups.map((_, i) => i + 1) : nums(h, posV!, 'positions')
  if (pos.length !== N) throw h.err('ValueError', 'List of violinplot statistics and `positions` values must have the same length')
  const widths = isNone(widthsV) ? new Array<number>(N).fill(0.5) : isNum(widthsV!) ? new Array<number>(N).fill(h.num(widthsV!)) : nums(h, widthsV!, 'widths')
  if (widths.length !== N) throw h.err('ValueError', 'List of violinplot statistics and `widths` values must have the same length')
  const color = py.str(ax.nextLineColor())
  const first = ax.children.length
  const bodies: Value[] = []
  const stats = groups.map((g) => {
    const lo = minOf(g)
    const hi = maxOf(g)
    const coords = Array.from({ length: points }, (_, i) => (i === points - 1 ? hi : lo + (i * (hi - lo)) / (points - 1)))
    const singular = lo === hi
    const vals = singular ? coords.map((c) => (g.includes(c) ? 1 : 0)) : gaussianKde(g, coords)
    const s = [...g].sort((a, b) => a - b)
    const n = s.length
    const median = n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2
    return { lo, hi, coords, vals, mean: g.reduce((a, b) => a + b, 0) / n, median }
  })
  stats.forEach((s, i) => {
    const top = maxOf(s.vals)
    const half = s.vals.map((v) => (0.5 * widths[i] * v) / top)
    const low = floats(half.map((v) => -v + pos[i]))
    const high = floats(half.map((v) => v + pos[i]))
    bodies.push(fillBetween(!vert)(ax, [floats(s.coords), low, high], { facecolor: color, alpha: py.float(0.3) }, h))
  })
  const ends = (i: number): [number, number] => [pos[i] - 0.25 * widths[i], pos[i] + 0.25 * widths[i]]
  const perp = (vals: number[]) => linesOf(!vert)(ax, [floats(vals), floats(vals.map((_, i) => ends(i)[0])), floats(vals.map((_, i) => ends(i)[1]))], { colors: color }, h)
  const out: [string, Value][] = [['bodies', py.list(bodies)]]
  if (showmeans) out.push(['cmeans', perp(stats.map((s) => s.mean))])
  if (showextrema) {
    out.push(['cmaxes', perp(stats.map((s) => s.hi))], ['cmins', perp(stats.map((s) => s.lo))])
    out.push(['cbars', linesOf(vert)(ax, [floats(pos), floats(stats.map((s) => s.lo)), floats(stats.map((s) => s.hi))], { colors: color }, h)])
  }
  if (showmedians) out.push(['cmedians', perp(stats.map((s) => s.median))])
  added(h, ax, first)
  return dictOf(out)
}

// ---------------------------------------------------------------- quiver

const quiver: AxMethod = (ax, args, kw, h) => {
  const n = args.length
  if (n < 2 || n > 5) throw h.err('TypeError', `quiver() takes from 2 to 5 positional arguments but ${n} were given`)
  const Ua = toArray(n >= 4 ? args[2] : args[0], 'float64')
  const Va = toArray(n >= 4 ? args[3] : args[1], 'float64')
  const Cv = n === 3 ? args[2] : n === 5 ? args[4] : undefined
  const U = Ua.values()
  const V = Va.values()
  if (U.length !== V.length) throw h.err('ValueError', `operands could not be broadcast together with shapes (${Ua.shape.join(',')},) (${Va.shape.join(',')},) `)
  const [rows, cols] = Ua.ndim === 2 ? Ua.shape : [1, U.length]
  let X: number[]
  let Y: number[]
  if (n >= 4) {
    const xa = toArray(args[0], 'float64')
    const ya = toArray(args[1], 'float64')
    if (xa.ndim === 1 && ya.ndim === 1 && Ua.ndim === 2 && xa.size === cols && ya.size === rows) {
      const xs = xa.values()
      const ys = ya.values()
      X = Array.from({ length: rows * cols }, (_, i) => xs[i % cols])
      Y = Array.from({ length: rows * cols }, (_, i) => ys[Math.floor(i / cols)])
    } else {
      X = xa.values()
      Y = ya.values()
    }
  } else {
    X = Array.from({ length: rows * cols }, (_, i) => i % cols)
    Y = Array.from({ length: rows * cols }, (_, i) => Math.floor(i / cols))
  }
  if (X.length === 1 && U.length > 1) X = U.map(() => X[0])
  if (Y.length === 1 && U.length > 1) Y = U.map(() => Y[0])
  if (X.length !== U.length || Y.length !== U.length) throw h.err('ValueError', `Argument U has a size ${U.length} which does not match ${X.length}, the number of arrow positions`)
  const unitsS = optStr(h, pop(kw, 'units')) ?? 'width'
  const anglesV = pop(kw, 'angles') ?? py.str('uv')
  const scaleArg = optNum(h, pop(kw, 'scale'))
  const scaleUnits = optStr(h, pop(kw, 'scale_units'))
  const widthArg = optNum(h, pop(kw, 'width'))
  const headwidth = optNum(h, pop(kw, 'headwidth')) ?? 3
  const headlength = optNum(h, pop(kw, 'headlength')) ?? 5
  const headaxis = optNum(h, pop(kw, 'headaxislength')) ?? 4.5
  const minshaft = optNum(h, pop(kw, 'minshaft')) ?? 1
  const minlength = optNum(h, pop(kw, 'minlength')) ?? 1
  const pivot0 = optStr(h, pop(kw, 'pivot')) ?? 'tail'
  const pivot = pivot0 === 'mid' ? 'middle' : pivot0
  oneOf(h, pivot, 'pivot', ['tail', 'middle', 'tip'])
  const colorV = pop(kw, 'color', 'facecolor', 'fc')
  const cmapV = pop(kw, 'cmap')
  const norm = normArg(h, kw)
  const lut = cmapArg(h, cmapV)
  const edgeV = pop(kw, 'edgecolor', 'ec')
  const lwV = pop(kw, 'linewidth', 'lw')
  const spec = newPoly(colorV && !isNone(colorV) ? toColor(h, colorV) : '#000000', edgeV && !isNone(edgeV) ? toColor(h, edgeV) : 'none', optNum(h, lwV) ?? 0)
  const o = new PolyObj('Quiver', spec, lut, norm)
  o.module = 'quiver'
  if (Cv) {
    const vals = nums(h, Cv, 'C')
    norm.autoscaleNone(vals)
    o.values = vals
    o.shape = [vals.length]
    o.recolor()
  }
  o.applyKw(kw, h)
  ax.add(o)
  ax.x.update(X)
  ax.y.update(Y)
  ax.request()
  ax.dynamic.set(o, () => {
    const g = ax.drawn!
    const [W, H] = ax.fig.spec.size
    const bw = g.rect[2] * W * 100
    const bh = g.rect[3] * H * 100
    const tf = (a: Axis, v: number) => (a.isLog() ? Math.log10(v) : v)
    const itf = (a: Axis, v: number) => (a.isLog() ? 10 ** v : v)
    const toPx = (a: Axis, lim: [number, number], len: number, v: number) => ((tf(a, v) - tf(a, lim[0])) / (tf(a, lim[1]) - tf(a, lim[0]))) * len
    const fromPx = (a: Axis, lim: [number, number], len: number, p: number) => itf(a, tf(a, lim[0]) + (p / len) * (tf(a, lim[1]) - tf(a, lim[0])))
    const vw = Math.abs(g.xlim[1] - g.xlim[0])
    const vh = Math.abs(g.ylim[1] - g.ylim[0])
    const dots = (u: string): number => {
      switch (u) {
        case 'x': return bw / vw
        case 'y': return bh / vh
        case 'xy': return Math.hypot(bw, bh) / Math.hypot(vw, vh)
        case 'width': return bw
        case 'height': return bh
        case 'dots': return 1
        case 'inches': return 100
        default: throw h.err('ValueError', `'${u}' is not a valid value for units; supported values are 'x', 'y', 'xy', 'width', 'height', 'dots', 'inches'`)
      }
    }
    const transScale = dots(unitsS)
    const span = bw / transScale
    const N = U.length
    const width = widthArg ?? (0.06 * span) / Math.max(8, Math.min(25, Math.sqrt(N)))
    const anglesStr = anglesV.k === 'str' ? anglesV.v : ''
    const lengthsOf = (eps: number) => X.map((x0, i) => {
      const px0 = toPx(ax.x, g.xlim, bw, x0)
      const py0 = toPx(ax.y, g.ylim, bh, Y[i])
      const px1 = toPx(ax.x, g.xlim, bw, x0 + eps * U[i])
      const py1 = toPx(ax.y, g.ylim, bh, Y[i] + eps * V[i])
      return { ang: Math.atan2(py1 - py0, px1 - px0), len: Math.hypot(px1 - px0, py1 - py0) / eps }
    })
    let derived: { ang: number; len: number }[] | null = null
    if (anglesStr === 'xy' && scaleUnits === 'xy') derived = lengthsOf(1)
    else if (anglesStr === 'xy' || scaleUnits === 'xy') derived = lengthsOf(Math.max(Math.abs(ax.x.lo), Math.abs(ax.x.hi), Math.abs(ax.y.lo), Math.abs(ax.y.hi)) * 0.001)
    const a = anglesStr && scaleUnits === 'xy' ? derived!.map((d) => d.len) : U.map((u, i) => Math.hypot(u, V[i]))
    const sn = Math.max(10, Math.sqrt(N))
    const amean = a.reduce((s, v) => s + v, 0) / N
    const auto = (1.8 * amean * sn) / span
    let scale = scaleArg
    let widthuPerLenu = 1
    if (scaleUnits === null) scale ??= auto
    else {
      const dx = scaleUnits === 'xy' ? 1 : dots(scaleUnits)
      widthuPerLenu = dx / transScale
      scale ??= auto * widthuPerLenu
    }
    const minsh = minshaft * headlength
    const angleList = anglesV.k === 'str' ? null : broadcast(nums(h, anglesV, 'angles'), N)
    spec.paths = X.map((x0, i) => {
      const L = Math.min(2 ** 16, Math.max(0, a[i] * (widthuPerLenu / (scale! * width))))
      let hx: number[]
      let hy: number[]
      const ii = [0, 1, 2, 3, 2, 1, 0]
      const sgn = [1, 1, 1, 1, -1, -1, -1]
      if (L < minlength) {
        hx = Array.from({ length: 7 }, (_, t) => Math.cos((t * Math.PI) / 3) * minlength * 0.5)
        hy = Array.from({ length: 7 }, (_, t) => Math.sin((t * Math.PI) / 3) * minlength * 0.5)
      } else if (L < minsh) {
        const shrink = minsh ? L / minsh : 0
        const bx = [0, minsh - headaxis, minsh - headlength, minsh]
        const by = [0.5, 0.5, 0.5 * headwidth, 0]
        hx = ii.map((k) => shrink * bx[k])
        hy = ii.map((k, t) => shrink * by[k] * sgn[t])
      } else {
        const bx = [0, L - headaxis, L - headlength, L]
        const by = [0.5, 0.5, 0.5 * headwidth, 0]
        hx = ii.map((k) => bx[k])
        hy = ii.map((k, t) => by[k] * sgn[t])
      }
      if (L >= minlength) {
        if (pivot === 'middle') { const m = 0.5 * hx[3]; hx = hx.map((v) => v - m) } else if (pivot === 'tip') { const m = hx[3]; hx = hx.map((v) => v - m) }
      }
      const theta = anglesStr === 'xy' ? derived![i].ang : anglesStr === 'uv' ? Math.atan2(V[i], U[i]) : ((angleList?.[i] ?? 0) * Math.PI) / 180
      const c = Math.cos(theta)
      const s = Math.sin(theta)
      const bxp = toPx(ax.x, g.xlim, bw, x0)
      const byp = toPx(ax.y, g.ylim, bh, Y[i])
      return hx.flatMap((u, t) => {
        const ex = (u * c - hy[t] * s) * width * transScale
        const ey = (u * s + hy[t] * c) * width * transScale
        return [fromPx(ax.x, g.xlim, bw, bxp + ex), fromPx(ax.y, g.ylim, bh, byp + ey)]
      })
    })
  })
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

// ---------------------------------------------------------------- text

const COORDS: Record<string, Coord> = { data: 'data', 'axes fraction': 'axes', 'figure fraction': 'figure', 'subfigure fraction': 'figure' }

const text: AxMethod = (ax, args, kw, h) => {
  if (args.length < 3 && !kw.s) {
    const miss = ['x', 'y', 's'].slice(args.length).filter((k) => !(k in kw))
    if (miss.length) throw h.err('TypeError', `text() missing ${miss.length} required positional argument${miss.length > 1 ? 's' : ''}: ${miss.map((m) => `'${m}'`).join(' and ')}`)
  }
  const xv = args[0] ?? pop(kw, 'x')!
  const yv = args[1] ?? pop(kw, 'y')!
  const s = args[2] ?? pop(kw, 's')!
  const fd = args[3] ?? pop(kw, 'fontdict')
  const t = newText(h, h.num(xv), h.num(yv), h.str(s))
  const o = new TextObj(t, h.str(xv), h.str(yv))
  if (fd) o.applyKw(kwOfDict(h, fd), h, true)
  o.applyKw(kw, h, true)
  ax.add(o)
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

const annotate: AxMethod = (ax, args, kw, h) => {
  const textV = args[0] ?? pop(kw, 'text')
  const xyV = args[1] ?? pop(kw, 'xy')
  if (!textV) throw h.err('TypeError', "annotate() missing 2 required positional arguments: 'text' and 'xy'")
  if (!xyV) throw h.err('TypeError', "annotate() missing 1 required positional argument: 'xy'")
  const xy = nums(h, xyV, 'xy')
  const xtV = args[2] ?? pop(kw, 'xytext')
  const xycoords = optStr(h, args[3] ?? pop(kw, 'xycoords')) ?? 'data'
  const textcoords = optStr(h, args[4] ?? pop(kw, 'textcoords')) ?? xycoords
  const apV = args[5] ?? pop(kw, 'arrowprops')
  pop(kw, 'annotation_clip')
  const targetC = COORDS[xycoords]
  if (!targetC) throw h.err('NotImplementedError', `annotate(xycoords='${xycoords}') is not available in this sandbox`)
  const hasText = !isNone(xtV)
  const tp = hasText ? nums(h, xtV!, 'xytext') : xy
  const t = newText(h, tp[0], tp[1], h.str(textV))
  t.clip = false
  if (hasText) {
    const off = /^offset (points|pixels|fontsize)$/.exec(textcoords)
    if (off) {
      const scale = off[1] === 'points' ? 1 : off[1] === 'pixels' ? 0.72 : rcn(h, 'font.size')
      t.x = xy[0]
      t.y = xy[1]
      t.tx = targetC
      t.ty = targetC
      t.dx = tp[0] * scale
      t.dy = tp[1] * scale
    } else {
      const c = COORDS[textcoords]
      if (!c) throw h.err('NotImplementedError', `annotate(textcoords='${textcoords}') is not available in this sandbox`)
      t.tx = c
      t.ty = c
    }
  } else {
    t.tx = targetC
    t.ty = targetC
  }
  const posV = hasText ? h.iterate(xtV!) : h.iterate(xyV)
  const o = new TextObj(t, h.str(posV[0]), h.str(posV[1]))
  o.cls = 'Annotation'
  o.xy = [xy[0], xy[1]]
  o.xyValue = xyV
  o.applyKw(kw, h, true)
  if (!isNone(apV)) {
    const ap = kwOfDict(h, apV)
    const style = optStr(h, pop(ap, 'arrowstyle'))
    const colorV = pop(ap, 'color')
    const fcV = pop(ap, 'facecolor', 'fc')
    const ecV = pop(ap, 'edgecolor', 'ec')
    const lw = optNum(h, pop(ap, 'linewidth', 'lw'))
    const shrink = optNum(h, pop(ap, 'shrink')) ?? 0
    const widthA = optNum(h, pop(ap, 'width')) ?? 4
    const head = optNum(h, pop(ap, 'headwidth')) ?? 12
    const arrow: TextArrow = style
      ? { x: xy[0], y: xy[1], tx: targetC, ty: targetC, style, color: colorV ? toColor(h, colorV) : ecV ? toColor(h, ecV) : '#000000', lw: lw ?? rcn(h, 'patch.linewidth'), head: 0, shrink: 0, edge: 'none' }
      : { x: xy[0], y: xy[1], tx: targetC, ty: targetC, style: 'simple', color: colorV ? toColor(h, colorV) : fcV ? toColor(h, fcV) : rcc(h, 'patch.facecolor'), lw: widthA, head, shrink, edge: colorV ? toColor(h, colorV) : ecV ? toColor(h, ecV) : '#000000' }
    t.arrow = arrow
  }
  ax.add(o)
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

const setTitle: AxMethod = (ax, args, kw, h) => {
  const label = args[0] ?? pop(kw, 'label')
  if (!label) throw h.err('TypeError', "set_title() missing 1 required positional argument: 'label'")
  const fd = args[1] ?? pop(kw, 'fontdict')
  const loc = oneOf(h, optStr(h, args[2] ?? pop(kw, 'loc')) ?? rcs(h, 'axes.titlelocation'), 'loc', ['left', 'center', 'right']) as 'left' | 'center' | 'right'
  const pad = optNum(h, args[3] ?? pop(kw, 'pad'))
  const y = optNum(h, pop(kw, 'y'))
  const o = ax.titles[loc]
  o.spec.text = h.str(label)
  o.spec.size = rcFont(h, 'axes.titlesize')
  o.spec.weight = rcs(h, 'axes.titleweight')
  if (pad !== null) ax.titlePad = pad
  if (y !== null) ax.titleY = y
  if (fd) o.applyKw(kwOfDict(h, fd), h)
  o.applyKw(kw, h)
  emitFig(h, ax.fig, ax)
  return py.obj(o)
}

const setAxisLabel = (name: AxisName): AxMethod => (ax, args, kw, h) => {
  const a = ax.axisOf(name)
  const fn = `set_${name}label`
  const label = args[0] ?? pop(kw, `${name}label`)
  if (!label) throw h.err('TypeError', `${fn}() missing 1 required positional argument: '${name}label'`)
  const fd = args[1] ?? pop(kw, 'fontdict')
  const pad = optNum(h, args[2] ?? pop(kw, 'labelpad'))
  const locV = optStr(h, pop(kw, 'loc'))
  const o = a.labelObj
  o.spec.text = h.str(label)
  if (pad !== null) a.style.labelPad = pad
  if (locV !== null) {
    const allowed = name === 'y' ? ['bottom', 'center', 'top'] : ['left', 'center', 'right']
    oneOf(h, locV, 'loc', allowed)
    a.labelLoc = locV === 'bottom' || locV === 'left' ? 'left' : locV === 'top' || locV === 'right' ? 'right' : 'center'
    if (name === 'y') {
      o.ys = locV === 'bottom' ? '0' : locV === 'top' ? '1' : '0.5'
    } else o.xs = locV === 'left' ? '0' : locV === 'right' ? '1' : '0.5'
  }
  if (fd) o.applyKw(kwOfDict(h, fd), h)
  o.applyKw(kw, h)
  emitFig(h, ax.fig, ax)
  return py.obj(o)
}

// ---------------------------------------------------------------- colorbar

class ColorbarObj extends PyObj {
  readonly cls = 'Colorbar'
  ax: Axes
  mappable: Value
  long: Axis
  constructor(ax: Axes, mappable: Value, long: Axis) {
    super()
    this.ax = ax
    this.mappable = mappable
    this.long = long
  }
  repr() { return `<matplotlib.colorbar.Colorbar object at ${addr(this)}>` }
  getAttr(name: string, h: Host): Value | undefined {
    const redraw = () => { emitFig(h, this.ax.fig, this.ax); return py.NONE }
    switch (name) {
      case 'ax': return py.obj(this.ax)
      case 'mappable': return this.mappable
      case 'vmin': return py.float(Math.min(...this.long.view()))
      case 'vmax': return py.float(Math.max(...this.long.view()))
      case 'orientation': return py.str(this.long.name === 'y' ? 'vertical' : 'horizontal')
      case 'set_label': return fnv(name, (args, kw0) => {
        const kw = { ...kw0 }
        this.long.labelObj.spec.text = h.str(args[0] ?? pop(kw, 'label') ?? py.str(''))
        const loc = optStr(h, pop(kw, 'loc'))
        if (loc) this.long.labelLoc = loc === 'bottom' || loc === 'left' ? 'left' : loc === 'top' || loc === 'right' ? 'right' : 'center'
        this.long.labelObj.applyKw(kw, h)
        return redraw()
      })
      case 'set_ticks': return fnv(name, (args, kw0) => {
        const kw = { ...kw0 }
        const labels = args[1] ?? pop(kw, 'labels')
        const minor = flag(h, pop(kw, 'minor'), false)
        const v = args[0] ?? pop(kw, 'ticks')!
        if (v.k === 'obj' && v.o instanceof TickerObj && v.o.loc) this.long.ticker.loc = v.o.loc
        else {
          const lim = this.long.view()
          this.long.setTicks(h, v, labels, minor, kw)
          this.long.lim = lim
        }
        return redraw()
      })
      case 'get_ticks': return fnv(name, () => floats(this.long.current(h).locs))
      case 'set_ticklabels': return fnv(name, (args, kw) => { this.long.setTickLabels(h, args[0] ?? kw.ticklabels, false); return redraw() })
      case 'minorticks_on': return fnv(name, () => { this.long.minorOn(); return redraw() })
      case 'minorticks_off': return fnv(name, () => { this.long.ticker.minorLoc = { k: 'null' }; return redraw() })
      case 'update_normal': case 'update_ticks': return fnv(name, redraw)
      case 'set_alpha': case 'draw_all': return fnv(name, () => py.NONE)
      case 'remove': return fnv(name, () => {
        this.ax.fig.axes = this.ax.fig.axes.filter((a) => a !== this.ax)
        return redraw()
      })
      case 'outline': case 'solids': case 'dividers': return py.NONE
    }
    return undefined
  }
}

const COLORBAR_LOC: Record<string, { anchor: [number, number]; panchor: [number, number]; pad: number; orientation: 'vertical' | 'horizontal' }> = {
  right: { anchor: [0, 0.5], panchor: [1, 0.5], pad: 0.05, orientation: 'vertical' },
  left: { anchor: [1, 0.5], panchor: [0, 0.5], pad: 0.05, orientation: 'vertical' },
  bottom: { anchor: [0.5, 1], panchor: [0.5, 0], pad: 0.15, orientation: 'horizontal' },
  top: { anchor: [0.5, 0], panchor: [0.5, 1], pad: 0.05, orientation: 'horizontal' },
}

/** the mappable's colours as a mesh along the long axis, rebuilt at each drawing so set_clim and set_cmap show */
const fillColorbar = (cax: Axes, h: Host, m: ArtistObj & Mapped, long: Axis, vertical: boolean, keepLim: boolean) => {
  const set = (lo: number, hi: number) => { if (!keepLim) long.setLim(h, lo, hi, false) }
  let body: ArtistObj
  if (m instanceof ContourObj && !m.filled) {
    const spec = newSegs([], 1.5)
    const levs = m.levels
    spec.paths = levs.map((l) => (vertical ? [0, l, 1, l] : [l, 0, l, 1]))
    spec.colors = levs.map((_, i) => m.layerColor(i))
    spec.widths = [m.spec.kind === 'segs' ? m.spec.widths[0] : 1.5]
    spec.label = '_nolegend_'
    body = new SegsObj('LineCollection', spec)
    set(levs[0], levs[levs.length - 1])
    if (levs.length === 1) long.setLim(h, levs[0] - 0.05 * Math.abs(levs[0] || 1), levs[0] + 0.05 * Math.abs(levs[0] || 1), false)
    long.ticker.loc = { k: 'fixed', at: levs, nbins: 10 }
    cax.add(body)
    cax.before.push(() => { (body as SegsObj).spec.colors = m.levels.map((_, i) => m.layerColor(i)) })
    return
  }
  const spec: MeshArtist = { kind: 'mesh', ...artistBase(1), nx: 1, ny: 1, X: [], Y: [], colors: [], alpha: 1, uniform: true, edge: null, lw: 0 }
  const refresh = () => {
    spec.alpha = m.alphaSet ?? 1
    let edges: number[]
    let colors: string[]
    if (m instanceof ContourObj) {
      edges = m.levels
      colors = edges.slice(1).map((_, i) => m.layerColor(i + (m.ext === 'both' || m.ext === 'min' ? 1 : 0)))
    } else if (m.norm.cls === 'BoundaryNorm') {
      edges = m.norm.bounds
      colors = edges.slice(1).map((b, i) => m.norm.color(m.lut, (edges[i] + b) / 2))
    } else {
      const lo = m.norm.vmin ?? 0
      const hi = m.norm.vmax ?? 1
      const n = m.lut.n
      const log = m.norm.cls === 'LogNorm'
      edges = Array.from({ length: n + 1 }, (_, i) => (log ? lo * (hi / lo) ** (i / n) : lo + ((hi - lo) * i) / n))
      colors = lutHex(m.lut)
    }
    const n = colors.length
    spec.colors = colors
    if (vertical) {
      spec.nx = 1
      spec.ny = n
      spec.X = edges.flatMap(() => [0, 1])
      spec.Y = edges.flatMap((e) => [e, e])
    } else {
      spec.nx = n
      spec.ny = 1
      spec.X = [...edges, ...edges]
      spec.Y = [...edges.map(() => 0), ...edges.map(() => 1)]
    }
    spec.uniform = uniformGrid(spec.X, spec.Y, spec.nx, spec.ny)
    const a = edges[0]
    const b = edges[edges.length - 1]
    if (a !== b && !keepLim && (long.lim[0] !== a || long.lim[1] !== b)) long.lim = [a, b]
  }
  refresh()
  const lo = spec.colors.length ? (vertical ? spec.Y[0] : spec.X[0]) : 0
  const hi = spec.colors.length ? (vertical ? spec.Y[spec.Y.length - 1] : spec.X[spec.nx]) : 1
  if (lo === hi) set(lo - 0.05 * Math.abs(lo || 1), hi + 0.05 * Math.abs(hi || 1))
  else set(lo, hi)
  if (m.norm.cls === 'LogNorm') long.setScale(h, 'log', {})
  body = new MeshObj(spec, [], [], m.lut, m.norm)
  ;(body as MeshObj).values = null
  cax.add(body)
  cax.before.push(refresh)
  if (m instanceof ContourObj) long.ticker.loc = { k: 'fixed', at: m.levels, nbins: 10 }
}

const makeColorbar = (h: Host, fig: Figure, m: Value, caxV: Value | undefined, axV: Value | undefined, kw: Kw): Value => {
  if (m.k !== 'obj' || !(m.o instanceof ArtistObj) || !('norm' in m.o)) throw h.err('AttributeError', `'${m.k === 'obj' ? m.o.cls : m.k}' object has no attribute 'get_array'`)
  const mobj = m.o as ArtistObj & Mapped
  if (mobj.values === null && !(mobj instanceof ContourObj) && !(mobj instanceof ImageObj && !mobj.spec.rgb)) throw h.err('TypeError', 'You must first set_array for mappable')
  if (mobj instanceof ImageObj && mobj.spec.rgb) throw h.err('TypeError', 'You must first set_array for mappable')
  const locS = optStr(h, pop(kw, 'location'))
  const oriS = optStr(h, pop(kw, 'orientation'))
  if (locS && oriS && COLORBAR_LOC[locS].orientation !== oriS) throw h.err('TypeError', 'location and orientation are mutually exclusive')
  const location = locS ?? (oriS === 'horizontal' ? 'bottom' : 'right')
  if (!COLORBAR_LOC[location]) throw h.err('ValueError', `'${location}' is not a valid value for location; supported values are 'left', 'right', 'top', 'bottom'`)
  const info = COLORBAR_LOC[location]
  const vertical = info.orientation === 'vertical'
  const fraction = optNum(h, pop(kw, 'fraction')) ?? 0.15
  const shrink = optNum(h, pop(kw, 'shrink')) ?? 1
  const aspect0 = optNum(h, pop(kw, 'aspect')) ?? 20
  const pad = optNum(h, pop(kw, 'pad')) ?? info.pad
  const anchorV = pop(kw, 'anchor')
  const anchor = isNone(anchorV) ? info.anchor : (nums(h, anchorV!, 'anchor') as [number, number])
  const panchorV = pop(kw, 'panchor')
  const panchor = panchorV === undefined ? info.panchor : panchorV.k === 'bool' && !panchorV.v ? null : (nums(h, panchorV, 'panchor') as [number, number])
  pop(kw, 'use_gridspec')
  const label = pop(kw, 'label')
  const ticks = pop(kw, 'ticks')
  const format = pop(kw, 'format')
  const extend = optStr(h, pop(kw, 'extend'))
  if (extend && extend !== 'neither') h.print(`(sandbox) colorbar(extend='${extend}'): the end triangles are not drawn\n`)
  for (const k of ['spacing', 'drawedges', 'extendfrac', 'extendrect', 'boundaries', 'values', 'alpha', 'cmap', 'norm', 'ticklocation']) pop(kw, k)
  const keys = Object.keys(kw)
  if (keys.length) throw h.err('TypeError', `Colorbar.__init__() got an unexpected keyword argument '${keys[0]}'`)
  let cax: Axes
  if (caxV && !isNone(caxV)) {
    if (caxV.k !== 'obj' || !(caxV.o instanceof Axes)) throw h.err('TypeError', 'cax must be an Axes')
    cax = caxV.o
  } else {
    let parents: Axes[]
    if (axV && !isNone(axV)) parents = axV.k === 'obj' && axV.o instanceof Axes ? [axV.o] : h.iterate(axV).map((v) => (v as unknown as { o: Axes }).o)
    else parents = mobj.ax ? [mobj.ax] : [fig.gca()]
    if (parents.length === 1 && parents[0].ss) {
      const parent = parents[0]
      const ws = (2 * pad) / (1 - pad)
      const g = vertical
        ? new GridSpec(fig, 3, 2, location === 'left' ? [fraction, 1 - fraction - pad] : [1 - fraction - pad, fraction], [(1 - anchor[1]) * (1 - shrink), shrink, anchor[1] * (1 - shrink)])
        : new GridSpec(fig, 2, 3, [anchor[0] * (1 - shrink), shrink, (1 - anchor[0]) * (1 - shrink)], location === 'top' ? [fraction, 1 - fraction - pad] : [1 - fraction - pad, fraction])
      g.parent = parent.ss
      g.own = vertical ? { wspace: ws, hspace: 0 } : { hspace: ws, wspace: 0 }
      let main: SubplotSpec
      let cb: SubplotSpec
      if (vertical) {
        main = location === 'left' ? new SubplotSpec(g, 0, 3, 1, 2) : new SubplotSpec(g, 0, 3, 0, 1)
        cb = location === 'left' ? new SubplotSpec(g, 1, 2, 0, 1) : new SubplotSpec(g, 1, 2, 1, 2)
      } else {
        main = location === 'top' ? new SubplotSpec(g, 1, 2, 0, 3) : new SubplotSpec(g, 0, 1, 0, 3)
        cb = location === 'top' ? new SubplotSpec(g, 0, 1, 1, 2) : new SubplotSpec(g, 1, 2, 1, 2)
      }
      parent.ss = main
      if (panchor) parent.anchor = panchor
      cax = new Axes(fig, cb, null)
      cax.boxAspect = vertical ? aspect0 : 1 / aspect0
    } else {
      // make_axes: taken from the bounding box of all the parents
      const rects = parents.map((p) => p.position())
      const x0 = minOf(rects.map((r) => r[0]))
      const y0 = minOf(rects.map((r) => r[1]))
      const x1 = maxOf(rects.map((r) => r[0] + r[2]))
      const y1 = maxOf(rects.map((r) => r[1] + r[3]))
      const [W0, H0] = [x1 - x0, y1 - y0]
      let main: Rect
      let cbox: Rect
      if (location === 'right') {
        main = [x0, y0, W0 * (1 - fraction - pad), H0]
        cbox = [x0 + W0 * (1 - fraction), y0, W0 * fraction, H0]
      } else if (location === 'left') {
        main = [x0 + W0 * (fraction + pad), y0, W0 * (1 - fraction - pad), H0]
        cbox = [x0, y0, W0 * fraction, H0]
      } else if (location === 'top') {
        main = [x0, y0, W0, H0 * (1 - fraction - pad)]
        cbox = [x0, y0 + H0 * (1 - fraction), W0, H0 * fraction]
      } else {
        main = [x0, y0 + H0 * (fraction + pad), W0, H0 * (1 - fraction - pad)]
        cbox = [x0, y0, W0, H0 * fraction]
      }
      if (vertical) {
        const hh = cbox[3] * shrink
        cbox = [cbox[0], cbox[1] + (cbox[3] - hh) * anchor[1], cbox[2], hh]
      } else {
        const ww = cbox[2] * shrink
        cbox = [cbox[0] + (cbox[2] - ww) * anchor[0], cbox[1], ww, cbox[3]]
      }
      for (const p of parents) {
        const r = p.position()
        p.fixed = [main[0] + ((r[0] - x0) / W0) * main[2], main[1] + ((r[1] - y0) / H0) * main[3], (r[2] / W0) * main[2], (r[3] / H0) * main[3]]
        p.ss = null
        if (panchor) p.anchor = panchor
      }
      cax = new Axes(fig, null, cbox)
      cax.boxAspect = vertical ? aspect0 : 1 / aspect0
    }
    cax.anchor = anchor
    cax.label = '<colorbar>'
    fig.axes.push(cax)
  }
  const long = vertical ? cax.y : cax.x
  const short = vertical ? cax.x : cax.y
  mobj.norm.autoscaleNone(mobj.values ?? [])
  short.ticker.loc = { k: 'fixed', at: [] }
  short.setLim(h, 0, 1, false)
  if (vertical) {
    if (location === 'left') {
      long.style.tick1 = true
      long.style.label1 = true
      long.style.tick2 = false
      long.style.label2 = false
    } else {
      long.style.tick1 = false
      long.style.label1 = false
      long.style.tick2 = true
      long.style.label2 = true
      long.style.labelSide = 'high'
    }
  } else if (location === 'top') {
    long.style.tick1 = false
    long.style.label1 = false
    long.style.tick2 = true
    long.style.label2 = true
    long.style.labelSide = 'high'
  }
  fillColorbar(cax, h, mobj, long, vertical, false)
  if (!isNone(label)) long.labelObj.spec.text = h.str(label!)
  if (ticks && !isNone(ticks)) {
    if (ticks.k === 'obj' && ticks.o instanceof TickerObj && ticks.o.loc) long.ticker.loc = ticks.o.loc
    else long.ticker.loc = { k: 'fixed', at: nums(h, ticks, 'ticks') }
  }
  if (format && !isNone(format)) {
    if (format.k === 'str') long.ticker.fmt = /%/.test(format.v) ? { k: 'fmt', fmt: format.v } : { k: 'str', fmt: format.v }
    else if (format.k === 'obj' && format.o instanceof TickerObj && format.o.fmt) long.ticker.fmt = format.o.fmt
    else if (format.k === 'func' || format.k === 'fn') long.ticker.fmt = { k: 'func', fn: format }
  }
  emitFig(h, fig, cax)
  return py.obj(new ColorbarObj(cax, m, long))
}

const seabornHeatmap = (h: Host, args: Value[], kw0: Kw): Value => {
  const kw = { ...kw0 }
  const a = toArray(args[0] ?? pop(kw, 'data')!, 'float64')
  if (a.ndim !== 2) throw h.err('ValueError', 'heatmap needs a 2-D array (rows × columns)')
  const axV = pop(kw, 'ax')
  const ax = axV && axV.k === 'obj' && axV.o instanceof Axes ? axV.o : gca(h)
  const [rows, cols] = a.shape
  const vals = a.values()
  const cmapV = pop(kw, 'cmap') ?? py.str('rocket')
  const lut = cmapArg(h, cmapV)
  const center = optNum(h, pop(kw, 'center'))
  let vmin = optNum(h, pop(kw, 'vmin')) ?? minOf(finiteOf(vals))
  let vmax = optNum(h, pop(kw, 'vmax')) ?? maxOf(finiteOf(vals))
  if (center !== null) {
    const r = Math.max(vmax - center, center - vmin)
    vmin = center - r
    vmax = center + r
  }
  const norm = new NormObj('Normalize', vmin, vmax)
  const X = Array.from({ length: (rows + 1) * (cols + 1) }, (_, k) => k % (cols + 1))
  const Y = Array.from({ length: (rows + 1) * (cols + 1) }, (_, k) => Math.floor(k / (cols + 1)))
  const lwV = pop(kw, 'linewidths', 'linewidth')
  const lcV = pop(kw, 'linecolor')
  const mk: Kw = { cmap: py.obj(new ColormapObj(lut)), norm: py.obj(norm) }
  const lw = optNum(h, lwV) ?? 0
  if (lw > 0) {
    mk.edgecolors = lcV ?? py.str('white')
    mk.linewidth = py.float(lw)
  }
  const first = ax.children.length
  const m = meshOf(ax, h, X, Y, vals, cols, rows, mk, [rows, cols])
  ax.x.setLim(h, 0, cols, false)
  ax.y.setLim(h, rows, 0, false)
  ax.spines.left.spec.visible = false
  ax.spines.right.spec.visible = false
  ax.spines.top.spec.visible = false
  ax.spines.bottom.spec.visible = false
  const ticksFor = (v: Value | undefined, n: number): string[] | null => {
    if (v === undefined || (v.k === 'bool' && v.v) || (v.k === 'str' && v.v === 'auto')) return Array.from({ length: n }, (_, i) => String(i))
    if (v.k === 'bool') return null
    return h.iterate(v).map((x) => h.str(x))
  }
  const xl = ticksFor(pop(kw, 'xticklabels'), cols)
  const yl = ticksFor(pop(kw, 'yticklabels'), rows)
  ax.x.ticker.loc = { k: 'fixed', at: xl ? xl.map((_, i) => i + 0.5) : [] }
  ax.x.ticker.fmt = { k: 'fixed', labels: xl ?? [] }
  ax.y.ticker.loc = { k: 'fixed', at: yl ? yl.map((_, i) => i + 0.5) : [] }
  ax.y.ticker.fmt = { k: 'fixed', labels: yl ?? [] }
  ax.y.style.rotation = 90
  ax.y.style.va = 'center'
  const annot = pop(kw, 'annot')
  const fmtV = optStr(h, pop(kw, 'fmt')) ?? '.2g'
  const akw = kwOfDict(h, pop(kw, 'annot_kws'))
  if (annot && h.truthy(annot)) {
    const custom = annot.k === 'bool' ? null : toArray(annot).values()
    vals.forEach((v, i) => {
      if (!Number.isFinite(v)) return
      const [r, g, b] = rgbaOf(norm.color(lut, v))
      const lin = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)
      const lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
      const val = custom ? custom[i] : v
      const s = formatSpec(fmtV === 'd' ? py.int(Math.trunc(val)) : py.float(val), fmtV)
      const t = newText(h, (i % cols) + 0.5, Math.floor(i / cols) + 0.5, s)
      t.ha = 'center'
      t.va = 'center'
      t.color = lum > 0.408 ? '#262626' : '#ffffff'
      const o = new TextObj(t)
      o.applyKw(akw, h)
      ax.add(o)
    })
  }
  const square = pop(kw, 'square')
  if (square && h.truthy(square)) setAspect(h, ax, py.str('equal'))
  const cbarV = pop(kw, 'cbar')
  const cbarKw = kwOfDict(h, pop(kw, 'cbar_kws'))
  const caxV = pop(kw, 'cbar_ax')
  pop(kw, 'robust', 'mask', 'xticklabels', 'yticklabels')
  const keys = Object.keys(kw)
  if (keys.length && keys.some((k) => !COSMETIC.has(k))) throw h.err('NotImplementedError', `seaborn.heatmap(${keys[0]}=...) is not available in this sandbox`)
  if (!cbarV || h.truthy(cbarV)) makeColorbar(h, ax.fig, py.obj(m), caxV, py.obj(ax), { ...cbarKw })
  ax.fig.cur = ax
  added(h, ax, first)
  return py.obj(ax)
}

// ---------------------------------------------------------------- limits, ticks and small Axes methods

const limSetter = (n: AxisName): AxMethod => (ax, args, kw, h) => {
  const axis = ax.axisOf(n)
  if (args.length > 2) throw h.err('TypeError', `_AxesBase.set_${n}lim() takes from 1 to 3 positional arguments but ${args.length + 1} were given`)
  const [loName, hiName] = n === 'x' ? ['left', 'right'] : n === 'y' ? ['bottom', 'top'] : ['bottom', 'top']
  let lo = args[0] ?? pop(kw, loName)
  let hi = args[1] ?? pop(kw, hiName)
  const mn = pop(kw, `${n}min`)
  const mx = pop(kw, `${n}max`)
  if (mn !== undefined) {
    if (!isNone(lo)) throw h.err('TypeError', `Cannot pass both '${loName}' and '${n}min'`)
    lo = mn
  }
  if (mx !== undefined) {
    if (!isNone(hi)) throw h.err('TypeError', `Cannot pass both '${hiName}' and '${n}max'`)
    hi = mx
  }
  pop(kw, 'emit')
  const autoV = pop(kw, 'auto')
  const auto = autoV === undefined ? false : autoV.k === 'none' ? null : h.truthy(autoV)
  const keys = Object.keys(kw)
  if (keys.length) throw h.err('TypeError', `set_${n}lim() got an unexpected keyword argument '${keys[0]}'`)
  if (isNone(hi) && lo && !isNone(lo) && !isNum(lo)) [lo, hi] = h.iterate(lo)
  const lim = axis.setLim(h, isNone(lo) ? null : h.num(lo!), isNone(hi) ? null : h.num(hi!), auto)
  emitFig(h, ax.fig, ax)
  return py.tuple(lim.map(py.float))
}

const tickSetter = (n: AxisName): AxMethod => (ax, args, kw, h) => {
  const axis = ax.axisOf(n)
  const tv = args[0] ?? pop(kw, 'ticks')
  if (!tv) throw h.err('TypeError', `set_${n}ticks() missing 1 required positional argument: 'ticks'`)
  const labels = args[1] ?? pop(kw, 'labels')
  const minor = flag(h, pop(kw, 'minor'), false)
  if (isNone(labels) && Object.keys(kw).length) throw h.err('ValueError', 'labels argument cannot be None when kwargs are passed')
  const at = axis.setTicks(h, tv, labels, minor, kw)
  emitFig(h, ax.fig, ax)
  return py.list(at.map((_, i) => py.obj(new TickObj(axis, i))))
}

const tickLabelSetter = (n: AxisName): AxMethod => (ax, args, kw, h) => {
  const axis = ax.axisOf(n)
  const lv = args[0] ?? pop(kw, 'labels')
  if (!lv) throw h.err('TypeError', `set_${n}ticklabels() missing 1 required positional argument: 'labels'`)
  const minor = flag(h, pop(kw, 'minor'), false)
  const fd = pop(kw, 'fontdict')
  axis.setTickLabels(h, lv, minor, { ...kwOfDict(h, fd), ...kw })
  emitFig(h, ax.fig, ax)
  return py.list(axis.tickLabels(h, minor))
}

const getters = (n: AxisName): Record<string, AxMethod> => ({
  [`get_${n}lim`]: (ax) => py.tuple(ax.axisOf(n).view().map(py.float)),
  [`get_${n}bound`]: (ax) => {
    const [a, b] = ax.axisOf(n).view()
    return py.tuple([Math.min(a, b), Math.max(a, b)].map(py.float))
  },
  [`set_${n}bound`]: (ax, args, kw, h) => {
    let lo = args[0] ?? kw.lower
    let hi = args[1] ?? kw.upper
    if (isNone(hi) && lo && !isNum(lo)) [lo, hi] = h.iterate(lo)
    const axis = ax.axisOf(n)
    const [a, b] = axis.view()
    axis.setBound(h, isNone(lo) ? Math.min(a, b) : h.num(lo!), isNone(hi) ? Math.max(a, b) : h.num(hi!))
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  [`get_${n}ticks`]: (ax, _a, kw, h) => {
    const axis = ax.axisOf(n)
    if (kw.minor && h.truthy(kw.minor)) {
      const v = axis.view()
      const sp = axis.tickSpace(ax.position())
      return floats(axis.minorLocs(v, sp, axis.majorLocs(v, sp)))
    }
    return axis.tickValue(h)
  },
  [`get_${n}ticklabels`]: (ax, _a, kw, h) => py.list(ax.axisOf(n).tickLabels(h, flag(h, kw.minor, false))),
  [`get_${n}majorticklabels`]: (ax, _a, _k, h) => py.list(ax.axisOf(n).tickLabels(h)),
  [`get_${n}minorticklabels`]: (ax, _a, _k, h) => py.list(ax.axisOf(n).tickLabels(h, true)),
  [`get_${n}label`]: (ax) => py.str(ax.axisOf(n).labelObj.spec.text),
  [`get_${n}scale`]: (ax) => py.str(ax.axisOf(n).scale),
  [`set_${n}scale`]: (ax, args, kw, h) => {
    const v = args[0] ?? pop(kw, 'value')
    if (!v) throw h.err('TypeError', `set_${n}scale() missing 1 required positional argument: 'value'`)
    ax.axisOf(n).setScale(h, h.str(v), kw)
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  [`get_${n}margin`]: (ax) => ax.axisOf(n).marginValue ?? py.float(ax.axisOf(n).margin),
  [`set_${n}margin`]: (ax, args, _k, h) => {
    const m = h.num(args[0])
    if (m <= -0.5) throw h.err('ValueError', 'margin must be greater than -0.5')
    ax.axisOf(n).margin = m
    ax.axisOf(n).marginValue = args[0]
    ax.requestAxis(n)
    return py.NONE
  },
  [`get_autoscale${n}_on`]: (ax) => py.bool(ax.axisOf(n).auto),
  [`set_autoscale${n}_on`]: (ax, args, _k, h) => { ax.axisOf(n).auto = h.truthy(args[0]); return py.NONE },
  [`invert_${n}axis`]: (ax, _a, _k, h) => {
    const axis = ax.axisOf(n)
    const [a, b] = axis.view()
    axis.setLim(h, b, a, null)
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  [`${n}axis_inverted`]: (ax) => py.bool(ax.axisOf(n).inverted()),
  [`get_${n}axis`]: (ax) => py.obj(ax.axisOf(n)),
  [`set_${n}lim`]: limSetter(n),
  [`set_${n}ticks`]: tickSetter(n),
  [`set_${n}ticklabels`]: tickLabelSetter(n),
  [`set_${n}label`]: setAxisLabel(n),
})

const setLegend: AxMethod = (ax, args, kw, h) => {
  const l = buildLegend(h, ax.legendHandles(), args, kw, { ax, fig: null })
  ax.legendObj = l
  emitFig(h, ax.fig, ax)
  return py.obj(l)
}

const addArtistTo = (ax: Axes, v: Value, h: Host): Value => {
  if (v.k === 'obj' && v.o instanceof LegendObj) {
    v.o.ax = ax
    if (!ax.extraLegends.includes(v.o)) ax.extraLegends.push(v.o)
    if (ax.legendObj === v.o) ax.legendObj = null
  } else if (v.k === 'obj' && v.o instanceof ArtistObj) {
    if (v.o.ax && v.o.ax !== ax) throw h.err('RuntimeError', 'Can not reset the Axes. You are probably trying to re-use an artist in more than one Axes which is not supported')
    ax.add(v.o)
  } else throw h.err('AttributeError', `'${v.k === 'obj' ? v.o.cls : v.k}' object has no attribute 'set_figure'`)
  added(h, ax, ax.children.length - 1)
  return v
}

const twin = (axisName: 'x' | 'y'): AxMethod => (ax, _a, _k, h) => {
  if (ax instanceof Axes3D) throw h.err('NotImplementedError', 'twin axes on a 3D plot are not available')
  const f = ax.fig
  const t = new Axes(f, ax.ss, ax.fixed)
  t.anchor = ax.anchor
  t.twinned = true
  ax.twinned = true
  f.axes.push(t)
  f.cur = t
  t.patchVisible = false
  if (axisName === 'x') {
    shareAxis(t.x, ax.x)
    t.y.style.tick1 = false
    t.y.style.label1 = false
    t.y.style.tick2 = true
    t.y.style.label2 = true
    t.y.style.labelSide = 'high'
    ax.y.style.tick1 = true
    ax.y.style.tick2 = false
    ax.y.style.label1 = true
    ax.y.style.label2 = false
    t.x.visible = false
  } else {
    shareAxis(t.y, ax.y)
    t.x.style.tick1 = false
    t.x.style.label1 = false
    t.x.style.tick2 = true
    t.x.style.label2 = true
    t.x.style.labelSide = 'high'
    ax.x.style.tick1 = true
    ax.x.style.tick2 = false
    ax.x.style.label1 = true
    ax.x.style.label2 = false
    t.y.visible = false
  }
  emitFig(h, f, t)
  return py.obj(t)
}

const axisCall: AxMethod = (ax, args, kw, h) => {
  const a = args[0] ?? pop(kw, 'option')
  pop(kw, 'emit')
  if (a && (a.k === 'str' || a.k === 'bool')) {
    const s = a.k === 'bool' ? (a.v ? 'on' : 'off') : a.v.toLowerCase()
    if (s === 'on') ax.axisOff = false
    else if (s === 'off') ax.axisOff = true
    else if (['equal', 'tight', 'scaled', 'auto', 'image', 'square'].includes(s)) {
      ax.setAuto(true)
      ax.aspect = 'auto'
      ax.autoscaleView(h, false)
      if (s === 'equal') {
        ax.aspect = 1
        ax.adjustable = 'datalim'
      } else if (s === 'scaled') {
        ax.aspect = 1
        ax.adjustable = 'box'
        ax.anchor = ANCHORS.C
        ax.setAuto(false)
      } else if (s === 'tight') {
        ax.autoscaleView(h, true)
        ax.setAuto(false)
      } else if (s === 'image') {
        ax.autoscaleView(h, true)
        ax.setAuto(false)
        ax.aspect = 1
        ax.adjustable = 'box'
        ax.anchor = ANCHORS.C
      } else if (s === 'square') {
        ax.aspect = 1
        ax.adjustable = 'box'
        ax.anchor = ANCHORS.C
        ax.setAuto(false)
        const xl = ax.x.view()
        const yl = ax.y.view()
        const edge = Math.max(xl[1] - xl[0], yl[1] - yl[0])
        ax.x.setLim(h, xl[0], xl[0] + edge, false)
        ax.y.setLim(h, yl[0], yl[0] + edge, false)
      }
    } else throw h.err('ValueError', `Unrecognized string ${pyRepr(a)} to axis; try 'on' or 'off'`)
  } else {
    let [x0, x1, y0, y1]: (Value | undefined)[] = [pop(kw, 'xmin'), pop(kw, 'xmax'), pop(kw, 'ymin'), pop(kw, 'ymax')]
    if (a && !isNone(a)) {
      const items = h.iterate(a)
      if (items.length !== 4) throw h.err('TypeError', 'The first argument to axis() must be an iterable of the form [xmin, xmax, ymin, ymax]')
      ;[x0, x1, y0, y1] = items
    }
    if (!isNone(x0) || !isNone(x1)) ax.x.setLim(h, isNone(x0) ? null : h.num(x0!), isNone(x1) ? null : h.num(x1!), false)
    if (!isNone(y0) || !isNone(y1)) ax.y.setLim(h, isNone(y0) ? null : h.num(y0!), isNone(y1) ? null : h.num(y1!), false)
  }
  const keys = Object.keys(kw)
  if (keys.length) throw h.err('TypeError', `axis() got an unexpected keyword argument '${keys[0]}'`)
  emitFig(h, ax.fig, ax)
  return py.tuple([...ax.x.view(), ...ax.y.view()].map(py.float))
}

const AXES_METHODS: Record<string, AxMethod> = {
  plot, scatter, bar, barh, hist, imshow, pie, errorbar, boxplot, annotate, step, text, stem, stairs, stackplot, fill, arrow, quiver, eventplot, ecdf, broken_barh: brokenBarh, matshow, spy, hexbin, hist2d, violinplot, axline,
  bar_label: barLabel, contour: contourOf(false), contourf: contourOf(true), clabel, tricontour: tricontour(false), tricontourf: tricontour(true), triplot, tripcolor,
  pcolormesh: pcolormesh('pcolormesh'), pcolor: pcolormesh('pcolor'),
  semilogx: scaleThen(true, false), semilogy: scaleThen(false, true), loglog: scaleThen(true, true),
  fill_between: fillBetween(true), fill_betweenx: fillBetween(false),
  axhline: refLine(false), axvline: refLine(true), axhspan: spanOf(false), axvspan: spanOf(true),
  hlines: linesOf(false), vlines: linesOf(true),
  twinx: twin('x'), twiny: twin('y'),
  streamplot: (_a, _b, _c, h) => { throw h.err('NotImplementedError', 'streamplot is not drawn in this sandbox: use quiver') },
  barbs: (_a, _b, _c, h) => { throw h.err('NotImplementedError', 'barbs is not drawn in this sandbox: use quiver') },
  table: (_a, _b, _c, h) => { throw h.err('NotImplementedError', 'ax.table is not drawn in this sandbox') },
  ...getters('x'), ...getters('y'),
  set_title: setTitle,
  get_title: (ax, args, kw, h) => {
    const loc = oneOf(h, optStr(h, args[0] ?? kw.loc) ?? 'center', 'loc', ['center', 'left', 'right']) as 'center'
    return py.str(ax.titles[loc].spec.text)
  },
  legend: setLegend,
  get_legend: (ax) => (ax.legendObj ? py.obj(ax.legendObj) : py.NONE),
  get_legend_handles_labels: (ax) => {
    const hs = ax.legendHandles().filter((v) => { const l = labelOfHandle(v); return l && !l.startsWith('_') })
    return py.tuple([py.list(hs), py.list(hs.map((v) => py.str(labelOfHandle(v))))])
  },
  get_lines: (ax) => py.obj(new SilentList('Line2D', ax.children.filter((c) => c instanceof LineObj).map((c) => py.obj(c)))),
  get_images: (ax) => py.list(ax.children.filter((c) => c instanceof ImageObj).map((c) => py.obj(c))),
  get_children: (ax) => py.list(ax.children.map((c) => py.obj(c))),
  has_data: (ax) => py.bool(ax.children.some((c) => !(c instanceof TextObj))),
  grid: (ax, args, kw, h) => { ax.setGrid(h, args, kw); return py.NONE },
  axis: axisCall,
  set_aspect: (ax, args, kw, h) => {
    setAspect(h, ax, args[0] ?? pop(kw, 'aspect')!)
    const adj = optStr(h, args[1] ?? pop(kw, 'adjustable'))
    if (adj !== null) ax.adjustable = oneOf(h, adj, 'adjustable', ['box', 'datalim']) as 'box'
    const anchorV = args[2] ?? pop(kw, 'anchor')
    if (!isNone(anchorV)) ax.anchor = anchorV!.k === 'str' ? ANCHORS[anchorV!.v] ?? ax.anchor : (nums(h, anchorV!, 'anchor') as [number, number])
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  get_aspect: (ax) => (ax.aspect === 'auto' ? py.str('auto') : py.float(ax.aspect)),
  set_adjustable: (ax, args, _k, h) => { ax.adjustable = oneOf(h, h.str(args[0]), 'adjustable', ['box', 'datalim']) as 'box'; return py.NONE },
  get_adjustable: (ax) => py.str(ax.adjustable),
  set_anchor: (ax, args, _k, h) => { ax.anchor = args[0].k === 'str' ? ANCHORS[args[0].v] ?? ax.anchor : (nums(h, args[0], 'anchor') as [number, number]); return py.NONE },
  set_box_aspect: (ax, args, _k, h) => { ax.boxAspect = optNum(h, args[0]); emitFig(h, ax.fig, ax); return py.NONE },
  get_box_aspect: (ax) => (ax.boxAspect === null ? py.NONE : py.float(ax.boxAspect)),
  apply_aspect: () => py.NONE,
  get_data_ratio: (ax) => {
    const size = (a: Axis) => { const [p, q] = a.view(); const t = (v: number) => (a.isLog() ? Math.log10(v) : v); return Math.max(Math.abs(t(q) - t(p)), 1e-30) }
    return py.float(size(ax.y) / size(ax.x))
  },
  set_facecolor: (ax, args, _k, h) => { ax.facecolor = toColor(h, args[0], 'facecolor'); emitFig(h, ax.fig, ax); return py.NONE },
  get_facecolor: (ax) => rgbaValue(ax.facecolor),
  set_fc: (ax, args, _k, h) => { ax.facecolor = toColor(h, args[0], 'facecolor'); return py.NONE },
  get_fc: (ax) => rgbaValue(ax.facecolor),
  set_axis_off: (ax, _a, _k, h) => { ax.axisOff = true; emitFig(h, ax.fig, ax); return py.NONE },
  set_axis_on: (ax, _a, _k, h) => { ax.axisOff = false; emitFig(h, ax.fig, ax); return py.NONE },
  set_frame_on: (ax, args, _k, h) => {
    if (!args.length) throw h.err('TypeError', "_AxesBase.set_frame_on() missing 1 required positional argument: 'b'")
    ax.frame = h.truthy(args[0])
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  get_frame_on: (ax) => py.bool(ax.frame),
  set_axisbelow: (ax, args, _k, h) => { ax.grid.below = args[0].k === 'str' ? args[0].v === 'line' : h.truthy(args[0]); return py.NONE },
  get_axisbelow: (ax) => py.bool(ax.grid.below),
  set_visible: (ax, args, _k, h) => { ax.visible = h.truthy(args[0]); emitFig(h, ax.fig, ax); return py.NONE },
  get_visible: (ax) => py.bool(ax.visible),
  set_label: (ax, args, _k, h) => { ax.label = h.str(args[0]); return py.NONE },
  get_label: (ax) => py.str(ax.label),
  set_zorder: (ax, args, _k, h) => { ax.zorder = h.num(args[0]); return py.NONE },
  get_zorder: (ax) => py.float(ax.zorder),
  set_alpha: () => py.NONE,
  get_figure: (ax) => py.obj(ax.fig),
  remove: (ax, _a, _k, h) => {
    ax.fig.axes = ax.fig.axes.filter((a) => a !== ax)
    if (ax.fig.cur === ax) ax.fig.cur = ax.fig.axes[ax.fig.axes.length - 1] ?? null
    emitFig(h, ax.fig, null)
    return py.NONE
  },
  set_prop_cycle: (ax, args, kw, h) => {
    if (args.length === 1 && isNone(args[0])) { ax.cycle = defaultCycle(h); ax.cycleRaw = null; ax.lineCycle = 0; ax.patchCycle = 0; return py.NONE }
    const c = args.length === 1 && args[0].k === 'obj' && args[0].o instanceof CyclerObj ? args[0].o : cyclerOf(h, args, kw)
    ax.cycle = c.colors
    ax.cycleRaw = c.raw
    ax.lineCycle = 0
    ax.patchCycle = 0
    return py.NONE
  },
  margins: (ax, args, kw, h) => {
    const xk = pop(kw, 'x')
    const yk = pop(kw, 'y')
    const tight = kw.tight === undefined ? true : isNone(kw.tight) ? null : h.truthy(kw.tight)
    if (args.length && (!isNone(xk) || !isNone(yk))) throw h.err('TypeError', 'Cannot pass both positional and keyword arguments for x and/or y.')
    if (args.length > 2) throw h.err('TypeError', 'Must pass a single positional argument for all margins, or one for each margin (x, y).')
    const mx = args.length === 1 ? args[0] : args.length === 2 ? args[0] : xk
    const my = args.length === 1 ? args[0] : args.length === 2 ? args[1] : yk
    if (isNone(mx) && isNone(my)) return py.tuple([ax.x.marginValue ?? py.float(ax.x.margin), ax.y.marginValue ?? py.float(ax.y.margin)])
    if (tight !== null) ax.tight = tight
    for (const [m, a, n] of [[mx, ax.x, 'x'], [my, ax.y, 'y']] as [Value | undefined, Axis, 'x' | 'y'][]) {
      if (isNone(m)) continue
      const v = h.num(m!)
      if (v <= -0.5) throw h.err('ValueError', 'margin must be greater than -0.5')
      a.margin = v
      a.marginValue = m!
      ax.requestAxis(n)
    }
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  autoscale: (ax, args, kw, h) => {
    const en = args[0] ?? kw.enable
    const which = oneOf(h, optStr(h, args[1] ?? kw.axis) ?? 'both', 'axis', ['x', 'y', 'both'])
    const tightV = args[2] ?? kw.tight
    const tight = isNone(tightV) ? null : h.truthy(tightV!)
    let sx = true
    let sy = true
    if (en === undefined || !isNone(en)) {
      const on = en === undefined ? true : h.truthy(en)
      sx = false
      sy = false
      if (which !== 'y') { ax.x.auto = on; sx = on }
      if (which !== 'x') { ax.y.auto = on; sy = on }
    }
    if (tight && sx) ax.x.margin = 0
    if (tight && sy) ax.y.margin = 0
    if (tight !== null) ax.tight = tight
    ax.request(sx, sy)
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  autoscale_view: (ax, args, kw, h) => {
    const t = args[0] ?? kw.tight
    ax.autoscaleView(h, isNone(t) ? null : h.truthy(t!), flag(h, args[1] ?? kw.scalex, true), flag(h, args[2] ?? kw.scaley, true))
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  get_autoscale_on: (ax) => py.bool(ax.x.auto && ax.y.auto),
  set_autoscale_on: (ax, args, _k, h) => { ax.setAuto(h.truthy(args[0])); return py.NONE },
  relim: (ax) => { ax.relim(); return py.NONE },
  update_datalim: (ax, args, kw, h) => {
    const a = toArray(args[0], 'float64')
    const v = a.values()
    const updatex = flag(h, args[1] ?? kw.updatex, true)
    const updatey = flag(h, args[2] ?? kw.updatey, true)
    if (updatex) ax.x.update(v.filter((_, i) => i % 2 === 0))
    if (updatey) ax.y.update(v.filter((_, i) => i % 2 === 1))
    return py.NONE
  },
  clear: (ax, _a, _k, h) => { ax.clear(h); emitFig(h, ax.fig, ax); return py.NONE },
  cla: (ax, _a, _k, h) => { ax.clear(h); emitFig(h, ax.fig, ax); return py.NONE },
  add_patch: (ax, args, _k, h) => {
    const v = args[0]
    if (v.k !== 'obj' || !(v.o instanceof PatchObj)) throw h.err('TypeError', 'add_patch() needs a Patch')
    if (v.o.ax && v.o.ax !== ax) throw h.err('RuntimeError', 'Can not reset the Axes. You are probably trying to re-use an artist in more than one Axes which is not supported')
    addPatch(ax, v.o)
    added(h, ax, ax.children.length - 1)
    return v
  },
  add_artist: (ax, args, _k, h) => addArtistTo(ax, args[0], h),
  add_line: (ax, args, _k, h) => {
    const v = args[0]
    if (v.k !== 'obj' || !(v.o instanceof LineObj)) throw h.err('TypeError', 'add_line() needs a Line2D')
    ax.add(v.o)
    if (v.o.spec.tx === 'data') ax.x.update(v.o.spec.x)
    if (v.o.spec.ty === 'data') ax.y.update(v.o.spec.y)
    added(h, ax, ax.children.length - 1)
    return v
  },
  add_collection: (ax, args, kw, h) => {
    const v = args[0]
    if (v.k !== 'obj' || !(v.o instanceof ArtistObj)) throw h.err('TypeError', 'add_collection() needs a Collection')
    ax.add(v.o)
    if (flag(h, args[1] ?? kw.autolim, true)) {
      const s = v.o.spec
      if (s.kind === 'segs' || s.kind === 'poly') {
        const all = s.paths.flat()
        ax.x.update(all.filter((_, i) => i % 2 === 0))
        ax.y.update(all.filter((_, i) => i % 2 === 1))
      }
    }
    added(h, ax, ax.children.length - 1)
    return v
  },
  add_image: (ax, args, _k, h) => addArtistTo(ax, args[0], h),
  add_container: (ax, args) => {
    const v = args[0]
    if (v.k === 'obj' && v.o instanceof ContainerObj) ax.containers.push(v.o)
    return v
  },
  tick_params: (ax, args, kw, h) => {
    const axis = oneOf(h, optStr(h, args[0] ?? pop(kw, 'axis')) ?? 'both', 'axis', ['x', 'y', 'both'])
    const which = oneOf(h, optStr(h, pop(kw, 'which')) ?? 'major', 'which', ['major', 'minor', 'both'])
    pop(kw, 'reset')
    if (axis !== 'y') ax.x.tickParams(h, which, kw)
    if (axis !== 'x') ax.y.tickParams(h, which, kw)
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  ticklabel_format: (ax, _a, kw, h) => {
    const axis = oneOf(h, optStr(h, kw.axis) ?? 'both', 'axis', ['x', 'y', 'both'])
    const style = optStr(h, kw.style) ?? ''
    const lim = isNone(kw.scilimits) ? null : (nums(h, kw.scilimits!, 'scilimits') as [number, number])
    const useOff = kw.useOffset
    for (const a of axis === 'both' ? [ax.x, ax.y] : [axis === 'x' ? ax.x : ax.y]) {
      if (a.ticker.fmt.k !== 'scalar') throw h.err('AttributeError', 'This method only works with the ScalarFormatter')
      if (style === 'sci' || style === 'scientific') a.ticker.fmt.sci = true
      if (style === 'plain') a.ticker.fmt.sci = false
      if (lim) a.ticker.fmt.limits = lim
      if (useOff !== undefined && !isNone(useOff)) a.ticker.fmt.useOffset = h.truthy(useOff)
    }
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  locator_params: (ax, _a, kw, h) => {
    const axis = oneOf(h, optStr(h, pop(kw, 'axis')) ?? 'both', 'axis', ['both', 'x', 'y'])
    pop(kw, 'tight')
    const nbins = pop(kw, 'nbins')
    const integer = pop(kw, 'integer')
    for (const a of axis === 'both' ? [ax.x, ax.y] : [axis === 'x' ? ax.x : ax.y]) {
      const base: MaxN = a.ticker.loc.k === 'maxn' ? a.ticker.loc.m : { ...AUTO }
      a.ticker.loc = { k: 'maxn', m: { ...base, nbins: nbins ? (nbins.k === 'str' ? 'auto' : h.toInt(nbins)) : base.nbins, integer: integer ? h.truthy(integer) : base.integer } }
    }
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  minorticks_on: (ax, _a, _k, h) => { ax.x.minorOn(); ax.y.minorOn(); emitFig(h, ax.fig, ax); return py.NONE },
  minorticks_off: (ax, _a, _k, h) => { ax.x.ticker.minorLoc = { k: 'null' }; ax.y.ticker.minorLoc = { k: 'null' }; emitFig(h, ax.fig, ax); return py.NONE },
  get_position: (ax, args, kw, h) => {
    const r = flag(h, args[0] ?? kw.original, false) ? ax.position() : untouched(ax.fig, () => ax.geometry().rect)
    return py.obj(new Bbox(r[0], r[1], r[0] + r[2], r[1] + r[3]))
  },
  set_position: (ax, args, _k, h) => {
    const v = args[0]
    const r = v.k === 'obj' && v.o instanceof Bbox ? [v.o.x0, v.o.y0, v.o.x1 - v.o.x0, v.o.y1 - v.o.y0] : nums(h, v, 'pos')
    ax.fixed = r as Rect
    ax.ss = null
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  get_subplotspec: (ax) => (ax.ss ? py.obj(ax.ss) : py.NONE),
  set_subplotspec: (ax, args, _k, h) => {
    const v = args[0]
    if (v.k !== 'obj' || !(v.o instanceof SubplotSpec)) throw h.err('TypeError', 'set_subplotspec() needs a SubplotSpec')
    ax.ss = v.o
    return py.NONE
  },
  get_gridspec: (ax) => (ax.ss ? py.obj(ax.ss.gs) : py.NONE),
  label_outer: (ax) => {
    const ss = ax.ss
    if (!ss) return py.NONE
    if (ss.r1 !== ss.gs.nrows) {
      ax.x.style.label1 = false
      ax.x.labelObj.spec.text = ''
      ax.x.offsetVisible = false
    }
    if (ss.r0 !== 0) ax.x.style.label2 = false
    if (ss.c0 !== 0) {
      ax.y.style.label1 = false
      ax.y.labelObj.spec.text = ''
      ax.y.offsetVisible = false
    }
    if (ss.c1 !== ss.gs.ncols) ax.y.style.label2 = false
    return py.NONE
  },
  sharex: (ax, args) => { shareAxis(ax.x, (args[0] as unknown as { o: Axes }).o.x); return py.NONE },
  sharey: (ax, args) => { shareAxis(ax.y, (args[0] as unknown as { o: Axes }).o.y); return py.NONE },
  get_xaxis_transform: () => py.obj(new TransformObj('data', 'axes')),
  get_yaxis_transform: () => py.obj(new TransformObj('axes', 'data')),
  get_shared_x_axes: (ax) => py.obj(sharedView(ax.x)),
  get_shared_y_axes: (ax) => py.obj(sharedView(ax.y)),
  inset_axes: (ax, args, kw0, h) => {
    const kw = { ...kw0 }
    const b = nums(h, args[0] ?? pop(kw, 'bounds')!, 'bounds')
    if (b.length !== 4) throw h.err('ValueError', 'bounds must be [x0, y0, width, height]')
    const r = ax.position()
    const t = new Axes(ax.fig, null, [r[0] + b[0] * r[2], r[1] + b[1] * r[3], b[2] * r[2], b[3] * r[3]])
    setKw(t, kw, h)
    ax.fig.axes.push(t)
    emitFig(h, ax.fig, t)
    return py.obj(t)
  },
  set: (ax, _args, kw, h) => {
    setKw(ax, kw, h)
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
}

const sharedView = (a: Axis) => new (class extends PyObj {
  readonly cls = 'GrouperView'
  repr() { return '<matplotlib.cbook.GrouperView object>' }
  getAttr(n: string): Value | undefined {
    if (n === 'joined') return fnv(n, (args) => py.bool(a.siblings.some((s) => s.ax === (args[1] as unknown as { o: Axes }).o)))
    if (n === 'get_siblings') return fnv(n, () => py.list(a.siblings.map((s) => py.obj(s.ax))))
    return undefined
  }
})()

// ---------------------------------------------------------------- 3D

const THREE_D_2D = new Set(['set_title', 'get_title', 'legend', 'get_legend', 'get_legend_handles_labels', 'get_lines', 'get_children', 'has_data', 'grid', 'set_facecolor', 'get_facecolor', 'set_axis_off', 'set_axis_on', 'set_frame_on', 'get_frame_on', 'set_visible', 'get_visible', 'set_label', 'get_label', 'get_figure', 'set_prop_cycle', 'margins', 'clear', 'cla', 'tick_params', 'get_position', 'set_position', 'get_subplotspec', 'set', 'set_zorder', 'get_zorder', 'autoscale_view', 'autoscale', 'relim', 'add_artist', 'minorticks_on', 'minorticks_off', 'set_alpha', 'get_autoscale_on', 'set_autoscale_on', 'apply_aspect'])

const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const norm3 = (a: number[]) => Math.hypot(a[0], a[1], a[2])
const mat = (a: number[][], b: number[][]) => a.map((r) => b[0].map((_, j) => r.reduce((s, v, k) => s + v * b[k][j], 0)))

const PLANES = [[0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 5, 4], [3, 2, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7]]
const AXINFO = [{ tickdir: 1, juggled: [1, 0, 2] }, { tickdir: 0, juggled: [0, 1, 2] }, { tickdir: 0, juggled: [0, 2, 1] }]
const LIGHT = (() => {
  const az = ((90 - 225) * Math.PI) / 180
  const alt = (19.4712 * Math.PI) / 180
  return [Math.cos(az) * Math.cos(alt), Math.sin(az) * Math.cos(alt), Math.sin(alt)]
})()
const shadeColor = (c: string, normal: number[]): string => {
  const [r, g, b, a] = rgbaOf(c)
  const n = norm3(normal)
  const s = n ? (normal[0] * LIGHT[0] + normal[1] * LIGHT[1] + normal[2] * LIGHT[2]) / n : 0
  const f = n ? 0.3 + ((s + 1) / 2) * 0.7 : 1
  return rgbHex(r * f, g * f, b * f, a)
}
const faceNormal = (p: number[][]): number[] => {
  const n = p.length
  const [i1, i2, i3] = [0, Math.floor(n / 3), Math.floor((2 * n) / 3)]
  const v1 = [p[i1][0] - p[i3][0], p[i1][1] - p[i3][1], p[i1][2] - p[i3][2]]
  const v2 = [p[i2][0] - p[i3][0], p[i2][1] - p[i3][1], p[i2][2] - p[i3][2]]
  return cross(v1, v2)
}

class Axes3D extends Axes {
  cls = 'Axes3D'
  z: Axis
  elev = 30
  azim = -60
  roll = 0
  box: number[]
  focal = 1
  dist = 10
  had = false
  drawGrid = true
  depths = new Map<ArtistObj, number>()
  constructor(fig: Figure, ss: SubplotSpec | null, fixed: Rect | null) {
    super(fig, ss, fixed)
    const h = fig.h
    this.z = new Axis('z', this, h)
    this.frame = false
    for (const n of SPINE_NAMES) this.spines[n].spec.visible = false
    this.box = this.boxAspectOf([4, 4, 3], 1)
    const m = (0.05 * 10) / 11
    for (const a of [this.x, this.y]) {
      a.lo = m
      a.hi = 1 - m
    }
    this.z.lo = 0
    this.z.hi = 1
    this.autoscaleView(h, null)
  }
  boxAspectOf(v: number[], zoom: number) {
    const f = (1.8294640721620434 * (25 / 24) * zoom) / norm3(v)
    return v.map((x) => x * f)
  }
  axisList() { return [this.x, this.y, this.z] }
  axisOf(n: AxisName): Axis { return n === 'z' ? this.z : super.axisOf(n) }
  requestAxis(n: AxisName) { if (n === 'z') this.z.stale = true; else super.requestAxis(n) }
  setAuto(on: boolean) { super.setAuto(on); this.z.auto = on }
  method(name: string): AxMethod | undefined {
    const m = AXES3D_METHODS[name]
    if (m) return m
    if (THREE_D_2D.has(name) || /^(get|set)_[xy](lim|label|ticks|ticklabels|margin|bound|scale|axis)$/.test(name) || /^invert_[xy]axis$/.test(name) || /^[xy]axis_inverted$/.test(name)) return AXES_METHODS[name]
    if (name in AXES_METHODS) return () => { throw this.h.err('NotImplementedError', `${name} is not available on a 3D Axes in this sandbox`) }
    return undefined
  }
  unstale() {
    if (!this.x.stale && !this.y.stale && !this.z.stale) return
    const [sx, sy, sz] = [this.x.stale, this.y.stale, this.z.stale]
    this.x.stale = this.y.stale = this.z.stale = false
    this.scale3d(this.h, sx, sy, sz)
  }
  /** Axes3D.autoscale_view */
  scale3d(h: Host, sx = true, sy = true, sz = true) {
    const one = (a: Axis) => {
      if (!a.auto) return
      let [x0, x1] = nonsingular(a.lo, a.hi, 0.05)
      if (a.margin > 0) {
        const d = (x1 - x0) * a.margin
        x0 -= d
        x1 += d
      }
      if (!this.tight) [x0, x1] = nonsingular(x0, x1, 1e-12, 1e-13)
      const d = (x1 - x0) * (1 / 48)
      const inv = a.lim[0] > a.lim[1]
      const lo = x0 - d
      const hi = x1 + d
      a.setLim(h, inv ? hi : lo, inv ? lo : hi, null)
    }
    if (sx) one(this.x)
    if (sy) one(this.y)
    if (sz) one(this.z)
  }
  autoscaleView(h: Host, tight: boolean | null, scalex = true, scaley = true) {
    if (tight !== null) this.tight = tight
    this.scale3d(h, scalex, scaley, scalex && scaley)
  }
  /** auto_scale_xyz: the first artist replaces the placeholder limits */
  fit(xs: number[], ys: number[], zs: number[]) {
    if (!this.had) {
      for (const a of [this.x, this.y, this.z]) {
        a.lo = Infinity
        a.hi = -Infinity
      }
      this.had = true
    }
    this.x.update(xs)
    this.y.update(ys)
    this.z.update(zs)
    this.scale3d(this.h)
  }
  proj(): number[][] {
    const [x0, x1] = this.x.view()
    const [y0, y1] = this.y.view()
    const [z0, z1] = this.z.view()
    const [ax, ay, az] = this.box
    const dx = (x1 - x0) / ax
    const dy = (y1 - y0) / ay
    const dz = (z1 - z0) / az
    const world = [[1 / dx, 0, 0, -x0 / dx], [0, 1 / dy, 0, -y0 / dy], [0, 0, 1 / dz, -z0 / dz], [0, 0, 0, 1]]
    const R = this.box.map((b) => 0.5 * b)
    const e = (this.elev * Math.PI) / 180
    const a = (this.azim * Math.PI) / 180
    const ps = [Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e)]
    const eye = R.map((r, i) => r + this.dist * ps[i])
    const wv = eye.map((v, i) => v - R[i])
    const wn = norm3(wv)
    const w = wv.map((v) => v / wn)
    const uv = cross([0, 0, 1], w)
    const un = norm3(uv)
    let u = uv.map((v) => v / un)
    let v = cross(w, u)
    if (this.roll) {
      const t = (-this.roll * Math.PI) / 180
      const [c, s] = [Math.cos(t), Math.sin(t)]
      const rot = (p: number[]) => {
        const d = w[0] * p[0] + w[1] * p[1] + w[2] * p[2]
        const cr = cross(w, p)
        return p.map((pi, i) => pi * c + cr[i] * s + w[i] * d * (1 - c))
      }
      u = rot(u)
      v = rot(v)
    }
    const focal = this.focal
    if (focal === Infinity) {
      const view = mat([[...u, 0], [...v, 0], [...w, 0], [0, 0, 0, 1]], [[1, 0, 0, -eye[0]], [0, 1, 0, -eye[1]], [0, 0, 1, -eye[2]], [0, 0, 0, 1]])
      const [zf, zb] = [-this.dist, this.dist]
      return mat([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, -2, 0], [0, 0, -(zf + zb), -(zf - zb)]], mat(view, world))
    }
    const eyeF = R.map((r, i) => r + this.dist * ps[i] * focal)
    const view = mat([[...u, 0], [...v, 0], [...w, 0], [0, 0, 0, 1]], [[1, 0, 0, -eyeF[0]], [0, 1, 0, -eyeF[1]], [0, 0, 1, -eyeF[2]], [0, 0, 0, 1]])
    const [zf, zb] = [-this.dist, this.dist]
    const b = (zf + zb) / (zf - zb)
    const c = (-2 * (zf * zb)) / (zf - zb)
    return mat([[focal, 0, 0, 0], [0, focal, 0, 0], [0, 0, b, c], [0, 0, -1, 0]], mat(view, world))
  }
  M: number[][] = []
  project(p: number[]): [number, number, number] {
    const M = this.M
    const x = M[0][0] * p[0] + M[0][1] * p[1] + M[0][2] * p[2] + M[0][3]
    const y = M[1][0] * p[0] + M[1][1] * p[1] + M[1][2] * p[2] + M[1][3]
    const z = M[2][0] * p[0] + M[2][1] * p[1] + M[2][2] * p[2] + M[2][3]
    const w = M[3][0] * p[0] + M[3][1] * p[1] + M[3][2] * p[2] + M[3][3]
    return [x / w, y / w, z / w]
  }
  decor(h: Host): Artist[] {
    const out: Artist[] = []
    if (this.axisOff) return out
    const views = [this.x.view(), this.y.view(), this.z.view()]
    const mins = views.map((v) => Math.min(...v))
    const maxs = views.map((v) => Math.max(...v))
    const cube = [[mins[0], mins[1], mins[2]], [maxs[0], mins[1], mins[2]], [maxs[0], maxs[1], mins[2]], [mins[0], maxs[1], mins[2]], [mins[0], mins[1], maxs[2]], [maxs[0], mins[1], maxs[2]], [maxs[0], maxs[1], maxs[2]], [mins[0], maxs[1], maxs[2]]].map((p) => this.project(p))
    const highs = [0, 1, 2].map((i) => {
      const mean = (k: number) => PLANES[k].reduce((s, c) => s + cube[c][2], 0) / 4
      return mean(2 * i) < mean(2 * i + 1)
    })
    const panes = ['axes3d.xaxis.panecolor', 'axes3d.yaxis.panecolor', 'axes3d.zaxis.panecolor']
    panes.forEach((key, i) => {
      const pane = newPoly(toColor(h, rcv(h, key)), 'none', 0, -13)
      pane.paths = [PLANES[2 * i + (highs[i] ? 1 : 0)].flatMap((c) => [cube[c][0], cube[c][1]])]
      pane.clip = false
      out.push(pane)
    })
    const minmax = highs.map((hi, i) => (hi ? maxs[i] : mins[i]))
    const maxmin = highs.map((hi, i) => (!hi ? maxs[i] : mins[i]))
    const centers = mins.map((m, i) => 0.5 * (m + maxs[i]))
    const deltas = mins.map((m, i) => (maxs[i] - m) * 0.08)
    const rect = this.drawn!.rect
    const [W, H] = this.fig.spec.size
    const axPts = 72 * (rect[2] * W + rect[3] * H)
    const perPoint = 48 / axPts
    const axes = [this.x, this.y, this.z]
    const space = this.x.tickSpace(rect)
    const away = (p: number[], ds: number[], skip: number) => p.map((v, i) => (i === skip ? v : v + (v - centers[i] < 0 || Object.is(v - centers[i], -0) ? -1 : 1) * ds[i]))
    axes.forEach((axis, i) => {
      const { juggled, tickdir } = AXINFO[i]
      const view = axis.lim
      const locs0 = axis.majorLocs(view, space)
      const fmt = axis.formatTicks(h, axis.ticker.fmt, locs0, view).labels
      const lo = Math.min(...view)
      const hi = Math.max(...view)
      const tol = (hi - lo) * 1e-10
      const keep = locs0.map((l, k) => [l, k] as const).filter(([l]) => l >= lo - tol && l <= hi + tol)
      // grid
      if (this.drawGrid) {
        const segs = newSegs([rcc(h, 'grid.color')], rcn(h, 'grid.linewidth'))
        segs.z = -12
        segs.clip = false
        segs.paths = keep.map(([l]) => {
          const p0 = [...minmax]
          p0[i] = l
          const a = [...p0]
          a[(i + 1) % 3] = maxmin[(i + 1) % 3]
          const c = [...p0]
          c[(i + 2) % 3] = maxmin[(i + 2) % 3]
          return [a, p0, c].flatMap((p) => { const q = this.project(p); return [q[0], q[1]] })
        })
        out.push(segs)
      }
      // axis line
      const e1 = [...minmax]
      e1[juggled[0]] = maxmin[juggled[0]]
      const e2 = [...e1]
      e2[juggled[1]] = maxmin[juggled[1]]
      const p1 = this.project(e1)
      const p2 = this.project(e2)
      const line = newLine(h, [p1[0], p2[0]], [p1[1], p2[1]], rcc(h, 'axes.edgecolor'))
      line.width = rcn(h, 'axes.linewidth')
      line.z = -11
      line.clip = false
      out.push(line)
      // ticks and labels
      const tickDelta = highs[tickdir] ? deltas[tickdir] : -deltas[tickdir]
      const tickOut = 0.1 * tickDelta
      const tickIn = 0.2 * tickDelta
      const ticks = newSegs([axis.style.color], rcn(h, 'xtick.major.width'))
      ticks.z = -11
      ticks.clip = false
      const paths: Path[] = []
      const labelD = deltas.map((d) => (3.5 + 8) * perPoint * d)
      keep.forEach(([l, k]) => {
        const pos = [...e1]
        pos[i] = l
        pos[tickdir] = e1[tickdir] + tickOut
        const a = this.project(pos)
        pos[tickdir] = e1[tickdir] - tickIn
        const b = this.project(pos)
        paths.push([a[0], a[1], b[0], b[1]])
        pos[tickdir] = e1[tickdir]
        const lab = this.project(away(pos, labelD, i))
        const t = newText(h, lab[0], lab[1], fmt[k])
        t.size = axis.style.labelSize
        t.color = axis.style.labelColor
        t.ha = 'center'
        t.va = 'top'
        t.z = -11
        out.push(t)
      })
      ticks.paths = paths
      out.push(ticks)
      // axis label
      const text = axis.labelObj.spec
      if (text.text) {
        const mid = e1.map((v, k) => 0.5 * (v + e2[k]))
        const ld = deltas.map((d) => (axis.style.labelPad + 21) * perPoint * d)
        const lab = this.project(away(mid, ld, i))
        const t = newText(h, lab[0], lab[1], text.text)
        t.size = text.size
        t.color = text.color
        t.weight = text.weight
        t.ha = 'center'
        t.va = 'center'
        t.z = -11
        if (stripMath(text.text).length > 4) {
          let ang = (Math.atan2(p2[1] - p1[1], p2[0] - p1[0]) * 180) / Math.PI
          ang = ((ang + 180) % 180 + 180) % 180
          if (ang > 90) ang -= 180
          t.rotation = ang
        }
        out.push(t)
      }
    })
    return out
  }
  /** apply_aspect: a 3D Axes only keeps its box square */
  geometry(): { rect: Rect; xlim: [number, number]; ylim: [number, number] } {
    const [W, H] = this.fig.spec.size
    const lim2d: [number, number] = [-0.95 / this.dist, 0.9 / this.dist]
    return { rect: fitAspect(this.position(), 1, H / W, this.anchor), xlim: lim2d, ylim: lim2d }
  }
  resolve(h: Host): AxesSpec {
    const { rect, xlim: lim2d } = this.geometry()
    this.drawn = { rect, xlim: lim2d, ylim: lim2d }
    this.M = this.proj()
    this.depths.clear()
    for (const fn of this.dynamic.values()) fn()
    const ordered = [...this.depths.entries()].sort((a, b) => b[1] - a[1])
    ordered.forEach(([o], i) => { o.spec.z = 3.5 + i })
    const off = (a: Axis, view: [number, number]) => ({ ...a.resolve(h, view, 9), at: [], labels: [], minor: [], minorLabels: [], offset: '', label: null, visible: false })
    const titles = (['center', 'left', 'right'] as const).filter((k) => this.titles[k].spec.text).map((k) => {
      const t = this.titles[k].spec
      return { text: t.text, size: t.size, color: t.color, weight: t.weight, italic: t.italic, loc: k, rotation: 0, pad: this.titlePad, y: this.titleY }
    })
    const sp = (n: string) => this.spines[n].spec
    return {
      rect, title: this.titles.center.spec.text || null, titles, xlim: lim2d, ylim: lim2d, xscale: 'linear', yscale: 'linear', xticks: off(this.x, this.x.view()), yticks: off(this.y, this.y.view()),
      axisOff: false, frame: false, facecolor: this.facecolor, patchVisible: this.patchVisible, visible: this.visible,
      grid: { x: false, y: false, xminor: false, yminor: false, ...this.grid },
      spines: { left: { ...sp('left') }, right: { ...sp('right') }, top: { ...sp('top') }, bottom: { ...sp('bottom') } },
      legends: [...this.extraLegends, ...(this.legendObj ? [this.legendObj] : [])].map((l) => l.spec),
      artists: [...this.children.map((c) => ({ ...c.spec })), ...this.decor(h)],
    }
  }
  clear(h: Host) {
    super.clear(h)
    this.had = false
    const m = (0.05 * 10) / 11
    for (const a of [this.x, this.y, this.z]) {
      a.lo = m
      a.hi = 1 - m
      a.lim = [0, 1]
    }
    this.scale3d(h)
  }
  getAttr(name: string, h: Host): Value | undefined {
    if (name === 'zaxis') return py.obj(this.z)
    if (name === 'elev') return py.float(this.elev)
    if (name === 'azim') return py.float(this.azim)
    if (name === 'roll') return py.float(this.roll)
    if (name === 'name') return py.str('3d')
    return super.getAttr(name, h)
  }
  isa(n: string) { return n === 'Axes' }
}

const broadcast3 = (h: Host, xv: Value, yv: Value, zv: Value | undefined): [number[], number[], number[]] => {
  const a = nums(h, xv, 'x')
  const b = nums(h, yv, 'y')
  const c = isNone(zv) ? [0] : nums(h, zv!, 'z')
  const n = Math.max(a.length, b.length, c.length)
  const bad = [a, b, c].find((v) => v.length !== 1 && v.length !== n)
  if (bad) throw h.err('ValueError', `shape mismatch: objects cannot be broadcast to a single shape.`)
  return [broadcast(a, n), broadcast(b, n), broadcast(c, n)]
}

/** the zs a flat scatter or line gets from zdir */
const juggle = (h: Host, kw: Kw, a: number[], b: number[], c: number[]): [number[], number[], number[]] => {
  const zdir = optStr(h, pop(kw, 'zdir')) ?? 'z'
  if (zdir === 'z') return [a, b, c]
  if (zdir === 'y') return [a, c, b]
  if (zdir === 'x') return [c, a, b]
  throw h.err('NotImplementedError', `zdir='${zdir}' is not available in this sandbox`)
}

const d3 = (ax: Axes) => ax as Axes3D

const plot3d: AxMethod = (ax0, args0, kw, h) => {
  const ax = d3(ax0)
  const args = [...args0]
  const xv = args.shift()
  const yv = args.shift()
  if (!xv || !yv) throw h.err('TypeError', 'plot() needs xs and ys on a 3D Axes')
  let zv: Value | undefined
  if (args.length && args[0].k !== 'str') zv = args.shift()
  else zv = pop(kw, 'zs')
  const [xs0, ys0, zs0] = broadcast3(h, xv, yv, zv)
  const [xs, ys, zs] = juggle(h, kw, xs0, ys0, zs0)
  const lines = plotLines(ax, [floats(xs), floats(ys), ...args], kw, h)
  ax.x.stale = false
  ax.y.stale = false
  for (const l of lines) {
    ax.dynamic.set(l, () => {
      const p = xs.map((x, i) => ax.project([x, ys[i], zs[i]]))
      l.spec.x = p.map((q) => q[0])
      l.spec.y = p.map((q) => q[1])
    })
  }
  ax.fit(xs, ys, zs)
  added(h, ax, ax.children.length - lines.length)
  return py.list(lines.map((l) => py.obj(l)))
}

const scatter3d: AxMethod = (ax0, args0, kw, h) => {
  const ax = d3(ax0)
  const xv = args0[0] ?? pop(kw, 'xs')
  const yv = args0[1] ?? pop(kw, 'ys')
  if (!xv || !yv) throw h.err('TypeError', 'scatter() needs xs and ys on a 3D Axes')
  const zv = args0[2] ?? pop(kw, 'zs')
  const [xs0, ys0, zs0] = broadcast3(h, xv, yv, zv)
  if (args0[3] && args0[3].k === 'str') kw.zdir = args0[3]
  const [xs, ys, zs] = juggle(h, kw, xs0, ys0, zs0)
  const depth = flag(h, pop(kw, 'depthshade'), true)
  const sV = args0[4] ?? pop(kw, 's') ?? py.int(20)
  const cV = args0[5] ?? pop(kw, 'c')
  const k: Kw = { ...kw, s: sV }
  if (cV) k.c = cV
  const r = scatter(ax, [floats(xs), floats(ys)], k, h)
  const o = (r as unknown as { o: ScatterObj }).o
  ax.x.stale = false
  ax.y.stale = false
  const base = () => o.spec.colors
  const faces = base()
  const sizes = o.spec.sizes
  ax.dynamic.set(o, () => {
    const p = xs.map((x, i) => ax.project([x, ys[i], zs[i]]))
    const order = p.map((_, i) => i).sort((a, b) => p[b][2] - p[a][2])
    const vz = p.map((q) => q[2])
    const lo = minOf(vz)
    const hi = maxOf(vz)
    o.spec.x = order.map((i) => p[i][0])
    o.spec.y = order.map((i) => p[i][1])
    const colorSrc = o.values ? o.values.map((v) => o.norm.color(o.lut, v)) : faces
    o.spec.colors = order.map((i) => {
      const c = colorSrc[i % colorSrc.length]
      if (!depth || c === 'none') return c
      const [r0, g, b, a] = rgbaOf(c)
      const t = hi > lo ? (vz[i] - lo) / (hi - lo) : 0
      return rgbHex(r0, g, b, a * (1 - t * 0.7))
    })
    o.spec.sizes = sizes.length === 1 ? sizes : order.map((i) => sizes[i])
    ax.depths.set(o, vz.reduce((s, v) => s + v, 0) / vz.length)
  })
  if (ax.z.margin < 0.05 && xs.length) ax.z.margin = 0.05
  ax.fit(xs, ys, zs)
  added(h, ax, ax.children.length - 1)
  return r
}

/** a collection of 3D polygons, drawn back to front */
const polys3d = (ax: Axes3D, h: Host, polys: number[][][], cls: string, faces: string[], edges: string[], lw: number, kw: Kw, values: number[] | null, lut: Lut, norm: NormObj, zsort = 'average'): PolyObj => {
  const spec = newPoly('none', 'none', lw)
  spec.seamless = edges[0] === 'none'
  spec.ecs = edges
  const o = new PolyObj(cls, spec, lut, norm)
  o.values = null
  o.applyKw(kw, h)
  ax.add(o)
  const metric = (zs: number[]) => (zsort === 'min' ? minOf(zs) : zsort === 'max' ? maxOf(zs) : zs.reduce((a, b) => a + b, 0) / zs.length)
  ax.dynamic.set(o, () => {
    const proj = polys.map((p) => p.map((q) => ax.project(q)))
    const depth = proj.map((p) => metric(p.map((q) => q[2])))
    const order = polys.map((_, i) => i).sort((a, b) => depth[b] - depth[a])
    spec.paths = order.map((i) => proj[i].flatMap((q) => [q[0], q[1]]))
    const cs = values ? values.map((v) => o.norm.color(o.lut, v)) : faces
    spec.fcs = order.map((i) => cs[i % cs.length])
    if (edges.length > 1) spec.ecs = order.map((i) => edges[i % edges.length])
    ax.depths.set(o, depth.reduce((a, b) => a + b, 0) / (depth.length || 1))
  })
  return o
}

const surface3d: AxMethod = (ax0, args, kw, h) => {
  const ax = d3(ax0)
  const [X, Y, Z] = [args[0] ?? pop(kw, 'X'), args[1] ?? pop(kw, 'Y'), args[2] ?? pop(kw, 'Z')]
  if (!X || !Y || !Z) throw h.err('TypeError', "plot_surface() missing required positional arguments: 'X', 'Y' and 'Z'")
  const Za = toArray(Z, 'float64')
  if (Za.ndim !== 2) throw h.err('ValueError', 'Argument Z must be 2-dimensional.')
  const [rows, cols] = Za.shape
  const fill2 = (v: Value) => {
    const a = toArray(v, 'float64')
    if (a.ndim === 2 && a.shape[0] === rows && a.shape[1] === cols) return a.values()
    throw h.err('ValueError', `shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (${a.shape.join(', ')}) and arg 2 with shape (${rows}, ${cols}).`)
  }
  const xs = fill2(X)
  const ys = fill2(Y)
  const zs = Za.values()
  const hasStride = 'rstride' in kw || 'cstride' in kw
  const hasCount = 'rcount' in kw || 'ccount' in kw
  if (hasStride && hasCount) throw h.err('ValueError', 'Cannot specify both stride and count arguments')
  const rcount = h.toInt(pop(kw, 'rcount') ?? py.int(50))
  const ccount = h.toInt(pop(kw, 'ccount') ?? py.int(50))
  const rs0 = pop(kw, 'rstride')
  const cs0 = pop(kw, 'cstride')
  const rstride = hasStride ? h.toInt(rs0 ?? py.int(10)) : Math.max(Math.ceil(rows / rcount), 1)
  const cstride = hasStride ? h.toInt(cs0 ?? py.int(10)) : Math.max(Math.ceil(cols / ccount), 1)
  const rowI = [...Array.from({ length: Math.ceil((rows - 1) / rstride) }, (_, i) => i * rstride), rows - 1]
  const colI = [...Array.from({ length: Math.ceil((cols - 1) / cstride) }, (_, i) => i * cstride), cols - 1]
  const colorV = pop(kw, 'color')
  const facecolorsV = pop(kw, 'facecolors')
  if (!isNone(facecolorsV)) throw h.err('NotImplementedError', 'plot_surface(facecolors=...) is not available in this sandbox: use cmap')
  const cmapV = pop(kw, 'cmap')
  const shade = flag(h, pop(kw, 'shade'), isNone(cmapV))
  pop(kw, 'lightsource', 'antialiased', 'axlim_clip')
  const norm = normArg(h, kw)
  const lut = cmapArg(h, cmapV)
  const edgeV = pop(kw, 'edgecolor', 'edgecolors', 'ec')
  const lwV = pop(kw, 'linewidth', 'linewidths', 'lw')
  const at = (r: number, c: number) => [xs[r * cols + c], ys[r * cols + c], zs[r * cols + c]]
  const polys: number[][][] = []
  for (let a = 0; a < rowI.length - 1; a++) {
    for (let b = 0; b < colI.length - 1; b++) {
      const [r0, r1, c0, c1] = [rowI[a], rowI[a + 1], colI[b], colI[b + 1]]
      const p: number[][] = []
      for (let c = c0; c < c1; c++) p.push(at(r0, c))
      for (let r = r0; r < r1; r++) p.push(at(r, c1))
      for (let c = c1; c > c0; c--) p.push(at(r1, c))
      for (let r = r1; r > r0; r--) p.push(at(r, c0))
      if (p.some((q) => q.some((v) => !Number.isFinite(v)))) continue
      polys.push(p)
    }
  }
  const color = isNone(colorV) ? ax.nextLineColor() : toColor(h, colorV!)
  let values: number[] | null = null
  let faces: string[] = [color]
  if (!isNone(cmapV)) {
    values = polys.map((p) => p.reduce((s, q) => s + q[2], 0) / p.length)
    norm.autoscaleNone(values)
  } else if (shade) faces = polys.map((p) => shadeColor(color, faceNormal(p)))
  const edges = isNone(edgeV) ? ['none'] : toColors(h, edgeV!, 'edgecolor')
  const o = polys3d(ax, h, polys, 'Poly3DCollection', faces, edges, optNum(h, lwV) ?? 1, kw, values, lut, norm)
  if (values) o.values = values
  ax.fit(xs, ys, zs)
  added(h, ax, ax.children.length - 1)
  // a colorbar reads the object's values; the dynamic step recolours from them
  o.recolor = () => {}
  return py.obj(o)
}

const wireframe3d: AxMethod = (ax0, args, kw, h) => {
  const ax = d3(ax0)
  const [X, Y, Z] = [args[0] ?? pop(kw, 'X'), args[1] ?? pop(kw, 'Y'), args[2] ?? pop(kw, 'Z')]
  if (!X || !Y || !Z) throw h.err('TypeError', "plot_wireframe() missing required positional arguments: 'X', 'Y' and 'Z'")
  const Za = toArray(Z, 'float64')
  if (Za.ndim !== 2) throw h.err('ValueError', 'Argument Z must be 2-dimensional.')
  const [rows, cols] = Za.shape
  const xs = toArray(X, 'float64').values()
  const ys = toArray(Y, 'float64').values()
  const zs = Za.values()
  const hasStride = 'rstride' in kw || 'cstride' in kw
  const hasCount = 'rcount' in kw || 'ccount' in kw
  if (hasStride && hasCount) throw h.err('ValueError', 'Cannot specify both stride and count arguments')
  const rcount = h.toInt(pop(kw, 'rcount') ?? py.int(50))
  const ccount = h.toInt(pop(kw, 'ccount') ?? py.int(50))
  const rs0 = pop(kw, 'rstride')
  const cs0 = pop(kw, 'cstride')
  const rstride = hasStride ? h.toInt(rs0 ?? py.int(1)) : rcount ? Math.max(Math.ceil(rows / rcount), 1) : 0
  const cstride = hasStride ? h.toInt(cs0 ?? py.int(1)) : ccount ? Math.max(Math.ceil(cols / ccount), 1) : 0
  const indices = (n: number, s: number) => {
    if (!s) return []
    const out: number[] = []
    for (let i = 0; i < n; i += s) out.push(i)
    if (out[out.length - 1] !== n - 1) out.push(n - 1)
    return out
  }
  const lines: number[][][] = []
  for (const r of indices(rows, rstride)) lines.push(Array.from({ length: cols }, (_, c) => [xs[r * cols + c], ys[r * cols + c], zs[r * cols + c]]))
  for (const c of indices(cols, cstride)) lines.push(Array.from({ length: rows }, (_, r) => [xs[r * cols + c], ys[r * cols + c], zs[r * cols + c]]))
  const colorsV = pop(kw, 'color', 'colors')
  const spec = newSegs(colorsV ? toColors(h, colorsV) : [rcc(h, 'lines.color')], rcn(h, 'lines.linewidth'))
  const o = new SegsObj('Line3DCollection', spec)
  o.module = 'art3d'
  o.applyKw(kw, h)
  ax.add(o)
  ax.dynamic.set(o, () => {
    const proj = lines.map((l) => l.map((p) => ax.project(p)))
    spec.paths = proj.map((l) => l.flatMap((q) => [q[0], q[1]]))
    const all = proj.flat().map((q) => q[2])
    ax.depths.set(o, all.reduce((a, b) => a + b, 0) / (all.length || 1))
  })
  ax.fit(xs, ys, zs)
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

const trisurf3d: AxMethod = (ax0, args, kw, h) => {
  const ax = d3(ax0)
  const { t, z, rest } = triArgs(h, args, kw)
  const zv = rest[0]
  const zs = z.length ? z : zv ? nums(h, zv, 'z') : []
  if (!zs.length) throw h.err('TypeError', 'plot_trisurf() needs z')
  const colorV = pop(kw, 'color')
  const cmapV = pop(kw, 'cmap')
  const shade = flag(h, pop(kw, 'shade'), isNone(cmapV))
  pop(kw, 'lightsource', 'antialiased', 'axlim_clip')
  const norm = normArg(h, kw)
  const lut = cmapArg(h, cmapV)
  const edgeV = pop(kw, 'edgecolor', 'edgecolors', 'ec')
  const lwV = pop(kw, 'linewidth', 'linewidths', 'lw')
  const polys = t.tris.map((tri) => tri.map((i) => [t.x[i], t.y[i], zs[i]]))
  const color = isNone(colorV) ? ax.nextLineColor() : toColor(h, colorV!)
  let values: number[] | null = null
  let faces: string[] = [color]
  if (!isNone(cmapV)) {
    values = polys.map((p) => p.reduce((s, q) => s + q[2], 0) / 3)
    norm.autoscaleNone(values)
  } else if (shade) faces = polys.map((p) => shadeColor(color, faceNormal(p)))
  const edges = isNone(edgeV) ? ['none'] : toColors(h, edgeV!, 'edgecolor')
  const o = polys3d(ax, h, polys, 'Poly3DCollection', faces, edges, optNum(h, lwV) ?? 1, kw, values, lut, norm)
  if (values) o.values = values
  o.recolor = () => {}
  ax.fit(t.x, t.y, zs)
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

const bar3d: AxMethod = (ax0, args, kw, h) => {
  const ax = d3(ax0)
  if (args.length < 6) throw h.err('TypeError', 'bar3d() takes x, y, z, dx, dy, dz')
  const arrs = args.slice(0, 6).map((v) => nums(h, v, 'bar3d'))
  const n = Math.max(...arrs.map((a) => a.length))
  const [x, y, z, dx, dy, dz] = arrs.map((a) => broadcast(a, n))
  const colorV = args[6] ?? pop(kw, 'color')
  const zsort = optStr(h, args[7] ?? pop(kw, 'zsort')) ?? 'average'
  const shade = flag(h, args[8] ?? pop(kw, 'shade'), true)
  pop(kw, 'lightsource', 'axlim_clip')
  const base = isNone(colorV) ? [ax.nextPatchColor()] : toColors(h, colorV!)
  const polys: number[][][] = []
  const faces: string[] = []
  const cube = [[[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]], [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]]
  for (let i = 0; i < n; i++) {
    cube.forEach((face, f) => {
      const p = face.map(([u, v, w]) => [x[i] + u * dx[i], y[i] + v * dy[i], z[i] + w * dz[i]])
      polys.push(p)
      const c = base.length === n ? base[i] : base.length === 6 * n ? base[i * 6 + f] : base[(i * 6 + f) % base.length]
      faces.push(shade ? shadeColor(c, faceNormal(p)) : c)
    })
  }
  const edgeV = pop(kw, 'edgecolor', 'edgecolors', 'ec')
  const lwV = pop(kw, 'linewidth', 'linewidths', 'lw')
  const edges = isNone(edgeV) ? ['none'] : toColors(h, edgeV!, 'edgecolor')
  const o = polys3d(ax, h, polys, 'Poly3DCollection', faces, edges, optNum(h, lwV) ?? 1, kw, null, getLut('viridis')!, new NormObj('Normalize', null, null), zsort)
  ax.fit([minOf(x), maxOf(x.map((v, i) => v + dx[i]))], [minOf(y), maxOf(y.map((v, i) => v + dy[i]))], [minOf(z), maxOf(z.map((v, i) => v + dz[i]))])
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

const contour3d: AxMethod = (ax0, args, kw, h) => {
  const ax = d3(ax0)
  const zdir = optStr(h, pop(kw, 'zdir')) ?? 'z'
  if (zdir !== 'z') throw h.err('NotImplementedError', `3D contour(zdir='${zdir}') is not available in this sandbox`)
  const offset = optNum(h, pop(kw, 'offset'))
  pop(kw, 'extend3d', 'stride', 'axlim_clip')
  const { X, Y, Z, nx, ny, level } = contourArgs(args, kw, h)
  const f = finiteOf(Z)
  const lv = pop(kw, 'levels') ?? level
  const levels = isNone(lv) ? autoLevels(minOf(f), maxOf(f), 7) : isNum(lv!) ? autoLevels(minOf(f), maxOf(f), h.toInt(lv!)) : nums(h, lv!, 'levels')
  const cmapV = pop(kw, 'cmap')
  const colorsV = pop(kw, 'colors')
  const norm = normArg(h, kw)
  norm.autoscaleNone(levels)
  const lut = cmapArg(h, cmapV)
  const fixed = isNone(colorsV) ? null : toColors(h, colorsV!)
  const lwV = pop(kw, 'linewidths', 'linewidth')
  const spec = newSegs([], optNum(h, lwV) ?? rcn(h, 'lines.linewidth'))
  const paths3: number[][][] = []
  const pathColors: string[] = []
  levels.forEach((l, i) => {
    for (const p of gridLines(X, Y, Z, nx, ny, l)) {
      paths3.push(Array.from({ length: p.length / 2 }, (_, k) => [p[k * 2], p[k * 2 + 1], offset ?? l]))
      pathColors.push(fixed ? fixed[i % fixed.length] : norm.color(lut, l))
    }
  })
  spec.colors = pathColors.length ? pathColors : ['#000000']
  const o = new SegsObj('Line3DCollection', spec)
  o.module = 'art3d'
  o.applyKw(kw, h)
  ax.add(o)
  ax.dynamic.set(o, () => {
    spec.paths = paths3.map((l) => l.flatMap((p) => { const q = ax.project(p); return [q[0], q[1]] }))
    const all = paths3.flat().map((p) => ax.project(p)[2])
    ax.depths.set(o, all.reduce((a, b) => a + b, 0) / (all.length || 1))
  })
  ax.fit(X, Y, offset === null ? Z : [...Z, offset])
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

const quiver3d: AxMethod = (ax0, args, kw, h) => {
  const ax = d3(ax0)
  if (args.length < 6) throw h.err('ValueError', `Wrong number of arguments. Expected 6, got ${args.length}`)
  const arrs = args.slice(0, 6).map((v) => nums(h, v, 'quiver'))
  const n = Math.max(...arrs.map((a) => a.length))
  const [X, Y, Z, U0, V0, W0] = arrs.map((a) => broadcast(a, n))
  const length = optNum(h, pop(kw, 'length')) ?? 1
  const ratio = optNum(h, pop(kw, 'arrow_length_ratio')) ?? 0.3
  const pivot = oneOf(h, optStr(h, pop(kw, 'pivot')) ?? 'tail', 'pivot', ['tail', 'middle', 'tip'])
  const normalize = flag(h, pop(kw, 'normalize'), false)
  pop(kw, 'axlim_clip')
  const scaleBy = U0.map((u, i) => (normalize ? 1 / (Math.hypot(u, V0[i], W0[i]) || 1) : 1) * length)
  const colorsV = pop(kw, 'color', 'colors')
  const spec = newSegs(colorsV ? toColors(h, colorsV) : [rcc(h, 'lines.color')], rcn(h, 'lines.linewidth'))
  const lines: number[][][] = []
  const [c, s] = [Math.cos((15 * Math.PI) / 180), Math.sin((15 * Math.PI) / 180)]
  // the limits follow the arrows' positions, not their tips
  const all: number[][] = []
  for (let i = 0; i < n; i++) {
    const uvw = [U0[i] * scaleBy[i], V0[i] * scaleBy[i], W0[i] * scaleBy[i]]
    if (!uvw[0] && !uvw[1] && !uvw[2]) continue
    all.push([X[i], Y[i], Z[i]])
    const shift = pivot === 'tail' ? 0 : pivot === 'tip' ? 1 : 0.5
    const start = [X[i] - shift * uvw[0], Y[i] - shift * uvw[1], Z[i] - shift * uvw[2]]
    const tip = start.map((v, k) => v + uvw[k])
    const nn = Math.hypot(uvw[0], uvw[1])
    const xp = nn ? uvw[1] / nn : 0
    const yp = nn ? -uvw[0] / nn : 1
    const heads = [1, -1].map((sg) => {
      const R = [[c + xp * xp * (1 - c), xp * yp * (1 - c), sg * yp * s], [xp * yp * (1 - c), c + yp * yp * (1 - c), -sg * xp * s], [-sg * yp * s, sg * xp * s, c]]
      return R.map((r) => r[0] * uvw[0] + r[1] * uvw[1] + r[2] * uvw[2])
    })
    lines.push([start, tip])
    for (const hv of heads) lines.push([tip, tip.map((v, k) => v - ratio * hv[k])])
  }
  const o = new SegsObj('Line3DCollection', spec)
  o.module = 'art3d'
  o.applyKw(kw, h)
  ax.add(o)
  ax.dynamic.set(o, () => {
    spec.paths = lines.map((l) => l.flatMap((p) => { const q = ax.project(p); return [q[0], q[1]] }))
  })
  ax.fit(all.map((p) => p[0]), all.map((p) => p[1]), all.map((p) => p[2]))
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

const text3d: AxMethod = (ax0, args, kw, h) => {
  const ax = d3(ax0)
  if (args.length < 4) throw h.err('TypeError', 'text() on a 3D Axes takes x, y, z, s')
  const [x, y, z] = args.slice(0, 3).map((v) => h.num(v))
  pop(kw, 'zdir', 'axlim_clip')
  const t = newText(h, 0, 0, h.str(args[3]))
  const o = new TextObj(t, h.str(args[0]), h.str(args[1]))
  o.applyKw(kw, h)
  ax.add(o)
  ax.dynamic.set(o, () => {
    const q = ax.project([x, y, z])
    t.x = q[0]
    t.y = q[1]
  })
  added(h, ax, ax.children.length - 1)
  return py.obj(o)
}

const AXES3D_METHODS: Record<string, AxMethod> = {
  plot: plot3d, plot3D: plot3d, scatter: scatter3d, scatter3D: scatter3d, plot_surface: surface3d, plot_wireframe: wireframe3d, plot_trisurf: trisurf3d,
  bar3d, contour: contour3d, contour3D: contour3d, quiver: quiver3d, quiver3D: quiver3d, text: text3d, text3D: text3d,
  ...getters('z'),
  set_zlim3d: limSetter('z'),
  set_xlim3d: limSetter('x'),
  set_ylim3d: limSetter('y'),
  view_init: (ax0, args, kw, h) => {
    const ax = d3(ax0)
    const e = args[0] ?? kw.elev
    const a = args[1] ?? kw.azim
    const r = args[2] ?? kw.roll
    const vert = optStr(h, args[3] ?? kw.vertical_axis) ?? 'z'
    if (vert !== 'z') throw h.err('NotImplementedError', `view_init(vertical_axis='${vert}') is not available in this sandbox`)
    ax.elev = isNone(e) ? 30 : h.num(e!)
    ax.azim = isNone(a) ? -60 : h.num(a!)
    ax.roll = isNone(r) ? 0 : h.num(r!)
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  set_box_aspect: (ax0, args, kw, h) => {
    const ax = d3(ax0)
    const zoom = optNum(h, args[1] ?? kw.zoom) ?? 1
    if (zoom <= 0) throw h.err('ValueError', `Argument zoom = ${zoom} must be > 0`)
    const v = isNone(args[0]) ? [4, 4, 3] : nums(h, args[0], 'aspect')
    if (v.length !== 3) throw h.err('ValueError', `'aspect' must be 1D with shape (3,), but your input has shape (${v.length},)`)
    ax.box = ax.boxAspectOf(v, zoom)
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  set_aspect: (ax0, args, kw, h) => {
    const ax = d3(ax0)
    const a = oneOf(h, h.str(args[0] ?? kw.aspect), 'aspect', ['auto', 'equal', 'equalxy', 'equalyz', 'equalxz'])
    if (a === 'equal') {
      const r = [ax.x, ax.y, ax.z].map((s) => Math.abs(s.view()[1] - s.view()[0]))
      const b = norm3(ax.box)
      const f = b / norm3(r)
      ax.box = r.map((v) => v * f)
    } else if (a !== 'auto') throw h.err('NotImplementedError', `3D set_aspect('${a}') is not available in this sandbox`)
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  get_box_aspect: () => py.NONE,
  grid: (ax0, args, kw, h) => {
    const ax = d3(ax0)
    const v = args[0] ?? kw.visible
    ax.drawGrid = isNone(v) ? !ax.drawGrid : h.truthy(v!)
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  set_proj_type: (ax0, args, kw, h) => {
    const ax = d3(ax0)
    const t = oneOf(h, h.str(args[0] ?? kw.proj_type), 'proj_type', ['persp', 'ortho'])
    const f = optNum(h, args[1] ?? kw.focal_length)
    ax.focal = t === 'ortho' ? Infinity : f ?? 1
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  invert_zaxis: (ax0, _a, _k, h) => {
    const ax = d3(ax0)
    const [a, b] = ax.z.view()
    ax.z.setLim(h, b, a, null)
    return py.NONE
  },
}

const makeAxes = (h: Host, fig: Figure, ss: SubplotSpec | null, rect: Rect | null, projection: Value | undefined, polar: boolean): Axes => {
  const p = isNone(projection) ? (polar ? 'polar' : 'rectilinear') : h.str(projection!)
  if (polar && p !== 'polar') throw h.err('ValueError', `polar=True, yet projection='${p}'. Only one of these arguments should be supplied.`)
  let ax: Axes
  if (p === 'rectilinear') ax = new Axes(fig, ss, rect)
  else if (p === '3d') ax = new Axes3D(fig, ss, rect)
  else if (p === 'polar') throw h.err('NotImplementedError', 'polar axes are not drawn in this sandbox')
  else if (['aitoff', 'hammer', 'lambert', 'mollweide'].includes(p)) throw h.err('NotImplementedError', `projection='${p}' is not drawn in this sandbox`)
  else throw h.err('ValueError', `Unknown projection ${pyRepr(py.str(p))}`)
  fig.axes.push(ax)
  return ax
}

/** Figure.subplot_mosaic: 'AB;CC' or nested lists of labels; each label must fill a rectangle */
const mosaicOn = (h: Host, f: Figure, layout: Value, kw: Kw): Value => {
  const empty = optStr(h, pop(kw, 'empty_sentinel')) ?? '.'
  const rows: string[][] = layout.k === 'str'
    ? layout.v.split(/[;\n]/).map((r) => r.trim()).filter((r) => r).map((r) => [...r])
    : h.iterate(layout).map((r) => h.iterate(r).map((c) => (isNone(c) ? empty : h.str(c))))
  const nrows = rows.length
  const ncols = rows[0]?.length ?? 0
  if (!nrows || !ncols || rows.some((r) => r.length !== ncols)) throw h.err('ValueError', `All of the rows must be the same length, however the first row (${pyRepr(py.str((rows[0] ?? []).join('')))}) has length ${ncols} and a later row does not.`)
  const subKw = kwOfDict(h, pop(kw, 'subplot_kw'))
  const gsKw = kwOfDict(h, pop(kw, 'gridspec_kw'))
  const wr = pop(kw, 'width_ratios')
  const hr = pop(kw, 'height_ratios')
  if (wr) gsKw.width_ratios = wr
  if (hr) gsKw.height_ratios = hr
  const shareX = flag(h, pop(kw, 'sharex'), false)
  const shareY = flag(h, pop(kw, 'sharey'), false)
  const g = gridSpecOf(h, f, [py.int(nrows), py.int(ncols)], gsKw)
  const seen = new Map<string, { r0: number; r1: number; c0: number; c1: number }>()
  rows.forEach((row, r) => row.forEach((name, c) => {
    if (name === empty) return
    const b = seen.get(name)
    if (!b) seen.set(name, { r0: r, r1: r + 1, c0: c, c1: c + 1 })
    else {
      b.r1 = Math.max(b.r1, r + 1)
      b.c0 = Math.min(b.c0, c)
      b.c1 = Math.max(b.c1, c + 1)
    }
  }))
  const out: [string, Value][] = []
  let first: Axes | null = null
  for (const [name, b] of seen) {
    for (let r = b.r0; r < b.r1; r++) for (let c = b.c0; c < b.c1; c++) {
      if (rows[r][c] !== name) throw h.err('ValueError', `While trying to layout\narray([${rows.map((x) => `[${x.map((c) => `'${c}'`).join(', ')}]`).join(',\n       ')}], dtype=object)\nwe found that the label ${pyRepr(py.str(name))} specifies a non-rectangular or non-contiguous area.`)
    }
    const sk = { ...subKw }
    const ax = makeAxes(h, f, new SubplotSpec(g, b.r0, b.r1, b.c0, b.c1), null, pop(sk, 'projection'), flag(h, pop(sk, 'polar'), false))
    setKw(ax, sk, h)
    ax.label = name
    if (first) {
      if (shareX) shareAxis(ax.x, first.x)
      if (shareY) shareAxis(ax.y, first.y)
    }
    first ??= ax
    out.push([name, py.obj(ax)])
  }
  f.cur = first
  emitFig(h, f, null)
  return dictOf(out)
}

const subplotsOn = (h: Host, f: Figure, args: Value[], kw: Kw, emit: boolean): Value => {
  const nrows = h.toInt(args[0] ?? pop(kw, 'nrows') ?? py.int(1))
  const ncols = h.toInt(args[1] ?? pop(kw, 'ncols') ?? py.int(1))
  if (nrows < 1) throw h.err('ValueError', `Number of rows must be a positive integer, not ${nrows}`)
  if (ncols < 1) throw h.err('ValueError', `Number of columns must be a positive integer, not ${ncols}`)
  if (nrows * ncols > 256) throw h.err('ValueError', 'subplots: at most 256 Axes in this sandbox')
  const sh = (v: Value | undefined, name: string): string => {
    if (v === undefined) return 'none'
    if (v.k === 'bool') return v.v ? 'all' : 'none'
    if (v.k === 'str') return oneOf(h, v.v, name, ['all', 'row', 'col', 'none'])
    throw h.err('ValueError', `sharex must be a bool or one of 'all', 'row', 'col', 'none'`)
  }
  const shareX = (() => {
    const v = args[2] ?? pop(kw, 'sharex')
    if (v?.k === 'str' && !['all', 'row', 'col', 'none'].includes(v.v)) throw h.err('ValueError', `'${v.v}' is not a valid value for sharex; supported values are 'all', 'row', 'col', 'none', False, True`)
    return sh(v, 'sharex')
  })()
  const shareY = (() => {
    const v = args[3] ?? pop(kw, 'sharey')
    if (v?.k === 'str' && !['all', 'row', 'col', 'none'].includes(v.v)) throw h.err('ValueError', `'${v.v}' is not a valid value for sharey; supported values are 'all', 'row', 'col', 'none', False, True`)
    return sh(v, 'sharey')
  })()
  const squeeze = flag(h, args[4] ?? pop(kw, 'squeeze'), true)
  const wr = pop(kw, 'width_ratios')
  const hr = pop(kw, 'height_ratios')
  const subKw = kwOfDict(h, pop(kw, 'subplot_kw'))
  const gsKw = kwOfDict(h, pop(kw, 'gridspec_kw'))
  if (wr) gsKw.width_ratios = wr
  if (hr) gsKw.height_ratios = hr
  const g = gridSpecOf(h, f, [py.int(nrows), py.int(ncols)], gsKw)
  const axes: Axes[] = []
  for (let r = 0; r < nrows; r++) {
    for (let c = 0; c < ncols; c++) {
      const ss = new SubplotSpec(g, r, r + 1, c, c + 1)
      const sk = { ...subKw }
      const a = makeAxes(h, f, ss, null, pop(sk, 'projection'), flag(h, pop(sk, 'polar'), false))
      setKw(a, sk, h)
      if (shareX !== 'none' && axes.length) {
        const leader = shareX === 'all' ? axes[0] : shareX === 'row' ? axes[r * ncols] : axes[c]
        if (leader && leader !== a && (shareX === 'all' || (shareX === 'row' ? c > 0 : r > 0))) shareAxis(a.x, leader.x)
      }
      if (shareY !== 'none' && axes.length) {
        const leader = shareY === 'all' ? axes[0] : shareY === 'row' ? axes[r * ncols] : axes[c]
        if (leader && leader !== a && (shareY === 'all' || (shareY === 'row' ? c > 0 : r > 0))) shareAxis(a.y, leader.y)
      }
      axes.push(a)
    }
  }
  if (shareX === 'col' || shareX === 'all') axes.forEach((a, i) => { if (Math.floor(i / ncols) !== nrows - 1) { a.x.style.label1 = false; a.x.offsetVisible = false } })
  if (shareY === 'row' || shareY === 'all') axes.forEach((a, i) => { if (i % ncols !== 0) { a.y.style.label1 = false; a.y.offsetVisible = false } })
  f.cur = axes[0]
  if (emit) emitFig(h, f, null)
  const out = squeeze && axes.length === 1 ? py.obj(axes[0]) : py.obj(new AxesGrid(axes, squeeze && (nrows === 1 || ncols === 1) ? [axes.length] : [nrows, ncols]))
  return py.tuple([py.obj(f), out])
}

// ---------------------------------------------------------------- the module

const FIG_KEYS = ['figsize', 'dpi', 'facecolor', 'edgecolor', 'frameon', 'layout', 'tight_layout', 'constrained_layout', 'clear', 'num', 'FigureClass', 'linewidth']
const figureFrom = (h: Host, numV: Value | undefined, kw: Kw): Figure => {
  const s = stateOf(h)
  const fs = pop(kw, 'figsize')
  let size: [number, number] | undefined
  if (fs && !isNone(fs)) {
    if (isNum(fs)) throw h.err('TypeError', `Value after * must be an iterable, not ${fs.k}`)
    const p = h.iterate(fs).map((v) => h.num(v))
    if (p.length !== 2 || p.some((v) => !(v > 0) || !Number.isFinite(v))) throw h.err('ValueError', `figure size must be positive finite not ${pyRepr(py.tuple(p.map(py.float)))}`)
    size = [p[0], p[1]]
  }
  const fc = pop(kw, 'facecolor')
  pop(kw, 'dpi', 'edgecolor', 'frameon', 'FigureClass', 'linewidth')
  const clear = flag(h, pop(kw, 'clear'), false)
  const layout = pop(kw, 'layout')
  const tl = pop(kw, 'tight_layout')
  const cl = pop(kw, 'constrained_layout')
  let f: Figure | undefined
  let label = ''
  if (numV && numV.k === 'obj' && numV.o instanceof Figure) f = numV.o
  else if (numV && numV.k === 'str') {
    label = numV.v
    f = s.figs.find((x) => !x.closed && x.label === label)
  } else if (numV && !isNone(numV)) {
    const n = h.toInt(numV)
    f = s.figs.find((x) => !x.closed && !x.shown && x.spec.num === n)
    if (!f) f = newFigure(h, size, n, fc ? toColor(h, fc) : undefined)
  }
  if (f) {
    s.cur = f
    if (clear) f.clear()
    return f
  }
  const made = newFigure(h, size, undefined, fc ? toColor(h, fc) : undefined)
  made.label = label
  if (layout) made.engine = layoutEngine(h, layout)
  if (tl && h.truthy(tl)) made.engine = 'tight'
  if (cl && h.truthy(cl)) made.engine = 'constrained'
  return made
}

const figaspect = (h: Host, v: Value): [number, number] => {
  const a = isNum(v) ? null : toArray(v)
  const ratio = a ? a.shape[0] / a.shape[1] : h.num(v)
  const height = nums(h, rcv(h, 'figure.figsize'), 'figsize')[1]
  let w = height / ratio
  let t = height
  const d1 = Math.min(1, w / 4, t / 2)
  w /= d1
  t /= d1
  const d2 = Math.max(1, w / 16, t / 16)
  w /= d2
  t /= d2
  return [Math.max(4, Math.min(16, w)), Math.max(2, Math.min(16, t))]
}

const RC_GROUP_ALIAS: Record<string, string> = { lw: 'linewidth', ls: 'linestyle', c: 'color', fc: 'facecolor', ec: 'edgecolor', mew: 'markeredgewidth', aa: 'antialiased' }
const RC_OTHER = /^(agg|animation|backend|date|docstring|font|hatch|interactive|keymap|macosx|mathtext|path|pdf|pgf|polaraxes|ps|svg|timezone|tk|toolbar|webagg)(\.|$)/

class RcParams extends PyObj {
  readonly cls = 'RcParams'
  h: Host
  frozen: Map<string, Value> | null
  constructor(h: Host, frozen: Map<string, Value> | null = null) {
    super()
    this.h = h
    this.frozen = frozen
  }
  store() { return stateOf(this.h).rc }
  has(k: string) { return k in RC_DEFAULTS || k === 'axes.prop_cycle' || this.store().has(k) || RC_OTHER.test(k) }
  repr() { return 'RcParams({...})' }
  len() { return Object.keys(RC_DEFAULTS).length + 1 }
  read(k: string): Value {
    if (k === 'axes.prop_cycle') return this.store().get(k) ?? py.obj(new CyclerObj(CYCLE.map(py.str), CYCLE))
    return this.store().get(k) ?? toValue(RC_DEFAULTS[k])
  }
  getItem(idx: Value, h: Host): Value {
    const k = h.str(idx)
    if (!(k in RC_DEFAULTS) && k !== 'axes.prop_cycle' && !this.store().has(k)) throw h.err('KeyError', pyRepr(py.str(k)))
    return this.read(k)
  }
  setItem(idx: Value, v: Value, h: Host) {
    const k = h.str(idx)
    if (!this.has(k)) throw h.err('KeyError', `'${k} is not a valid rc parameter (see rcParams.keys() for a list of valid parameters)'`)
    this.store().set(k, v)
  }
  contains(v: Value) { return v.k === 'str' && this.has(v.v) }
  iter(): Value[] { return [...Object.keys(RC_DEFAULTS), 'axes.prop_cycle'].map(py.str) }
  getAttr(name: string, h: Host): Value | undefined {
    switch (name) {
      case 'keys': return fnv(name, () => py.list(this.iter()))
      case 'values': return fnv(name, () => py.list(this.iter().map((k) => this.read(h.str(k)))))
      case 'items': return fnv(name, () => py.list(this.iter().map((k) => py.tuple([k, this.read(h.str(k))]))))
      case 'get': return fnv(name, (a) => { const k = h.str(a[0]); return this.has(k) ? this.read(k) : a[1] ?? py.NONE })
      case 'update': return fnv(name, (a, kw) => {
        const d = a[0] ? kwOfDict(h, a[0]) : kw
        for (const [k, v] of Object.entries(d)) this.setItem(py.str(k), v, h)
        return py.NONE
      })
    }
    return undefined
  }
}

const rcContext = (h: Host, args: Value[], kw: Kw): Value => {
  const rc = stateOf(h).rc
  const saved = new Map(rc)
  const apply = () => {
    const d = kwOfDict(h, args[0] ?? kw.rc)
    for (const [k, v] of Object.entries(d)) rc.set(k, v)
  }
  apply()
  return py.obj(new (class extends PyObj {
    readonly cls = 'rc_context'
    repr() { return '<matplotlib.rc_context object>' }
    enter() { return py.NONE }
    exit() {
      rc.clear()
      for (const [k, v] of saved) rc.set(k, v)
    }
  })())
}

const rcCall = (h: Host, args: Value[], kw: Kw): Value => {
  const groups = args[0].k === 'str' ? [args[0].v] : h.iterate(args[0]).map((v) => h.str(v))
  for (const g of groups) {
    for (const [k, v] of Object.entries(kw)) {
      const name = RC_GROUP_ALIAS[k] ?? k
      const key = `${g}.${name}`
      if (!(key in RC_DEFAULTS) && !RC_OTHER.test(key)) throw h.err('KeyError', `'Unrecognized key "${key}" for group "${g}" and name "${name}"'`)
      stateOf(h).rc.set(key, v)
    }
  }
  return py.NONE
}

/** Lazily built attributes: every colormap would otherwise be built at import */
const lazyCmaps = (): Record<string, Value> => {
  const attrs: Record<string, Value> = {}
  for (const n of [...ALL_CMAPS, ...SEABORN]) {
    let made: Value | null = null
    Object.defineProperty(attrs, n, { enumerable: true, get: () => (made ??= py.obj(new ColormapObj({ ...getLut(n)! }))) })
  }
  return attrs
}

class ScalarMappable extends ArtistObj implements Mapped {
  cls = 'ScalarMappable'
  spec: PolyArtist
  lut: Lut
  norm: NormObj
  values: number[] | null = []
  shape: number[] = [0]
  constructor(lut: Lut, norm: NormObj) {
    super()
    this.spec = newPoly('none', 'none', 0)
    this.lut = lut
    this.norm = norm
  }
  recolor() {}
  props() { return MAPPED_PROPS as unknown as Props<ArtistObj> }
  repr() { return `<matplotlib.cm.ScalarMappable object at ${addr(this)}>` }
  getAttr(name: string, h: Host): Value | undefined {
    if (name === 'to_rgba') return fnv(name, (args) => {
      const v = args[0]
      const cm = new ColormapObj(this.lut)
      if (isNum(v)) return cm.call([py.float(this.norm.t(h.num(v)))], {}, h)
      const a = toArray(v, 'float64')
      return cm.call([floats(a.values().map((x) => this.norm.t(x)), a.shape)], {}, h)
    })
    return super.getAttr(name, h)
  }
}

const imread = (h: Host) => fnv('imread', (args) => {
  const f = args[0] ? h.str(args[0]) : ''
  throw h.err('FileNotFoundError', `[Errno 2] No such file or directory: ${pyRepr(py.str(f))} (there are no files in this sandbox: build the image with NumPy instead)`)
})
const imsave = (h: Host) => fnv('imsave', (args, kw) => {
  const name = args[0] ?? kw.fname
  const arr = args[1] ?? kw.arr
  if (!name || !arr) throw h.err('TypeError', "imsave() missing required positional arguments: 'fname' and 'arr'")
  const a = toArray(arr)
  if (a.ndim < 2) throw h.err('ValueError', 'buffer is not large enough')
  h.print(`(sandbox) image would be saved as ${h.str(name)}\n`)
  return py.NONE
})

const colorsModule = (h: Host): Record<string, Value> => {
  const norm = (cls: string) => typ(cls, (args, kw) => {
    const o = new NormObj(cls, optNum(h, args[0] ?? kw.vmin), optNum(h, args[1] ?? kw.vmax), flag(h, args[2] ?? kw.clip, false))
    if (cls === 'LogNorm' && o.vmin !== null && o.vmax !== null && o.vmin <= 0) throw h.err('ValueError', 'Invalid vmin or vmax')
    return py.obj(o)
  })
  const toHexName = (v: Value) => toColor(h, v)
  const listed = (args: Value[], kw: Kw): Value => {
    const cv = args[0] ?? kw.colors
    if (!cv) throw h.err('TypeError', "ListedColormap.__init__() missing 1 required positional argument: 'colors'")
    let cs = toColors(h, cv)
    const name = optStr(h, args[1] ?? kw.name) ?? 'from_list'
    const nV = optNum(h, args[2] ?? kw.N)
    if (!cs.length) throw h.err('ValueError', 'A ListedColormap must have at least one color')
    if (nV !== null) cs = Array.from({ length: nV }, (_, i) => cs[i % cs.length])
    const rgb = cs.flatMap((c) => rgbaOf(c).slice(0, 3))
    return py.obj(new ColormapObj({ name, n: cs.length, rgb, listed: true, under: null, over: null, bad: null, raw: cv }))
  }
  const fromList = fnv('from_list', (args, kw) => {
    const name = h.str(args[0] ?? kw.name)
    const cv = args[1] ?? kw.colors
    const N = optNum(h, args[2] ?? kw.N) ?? 256
    const items = h.iterate(cv)
    const pairs = items.length && (items[0].k === 'tuple' || items[0].k === 'list') && (items[0] as { items: Value[] }).items.length === 2
    const vals = pairs ? items.map((p) => h.num((p as { items: Value[] }).items[0])) : items.map((_, i) => (items.length > 1 ? i / (items.length - 1) : 0))
    const cols = (pairs ? items.map((p) => (p as { items: Value[] }).items[1]) : items).map((c) => rgbaOf(toColor(h, c)))
    const seg = (c: number) => cols.flatMap((p, i) => [vals[i], p[c], p[c]])
    const ch = [0, 1, 2].map((c) => channelLut(N, seg(c)))
    return py.obj(new ColormapObj({ name, n: N, rgb: Array.from({ length: N * 3 }, (_, i) => ch[i % 3][Math.floor(i / 3)]), listed: false, under: null, over: null, bad: null }))
  })
  const rgb2hsv = (r: number, g: number, b: number) => {
    const mx = Math.max(r, g, b)
    const mn = Math.min(r, g, b)
    const d = mx - mn
    let hh = 0
    if (d) {
      if (mx === r) hh = ((g - b) / d) % 6
      else if (mx === g) hh = (b - r) / d + 2
      else hh = (r - g) / d + 4
      hh = (hh / 6 + 1) % 1
    }
    return [hh, mx ? d / mx : 0, mx]
  }
  const hsv2rgb = (hh: number, s: number, v: number) => {
    const i = Math.floor(hh * 6)
    const f = hh * 6 - i
    const p = v * (1 - s)
    const q = v * (1 - s * f)
    const t = v * (1 - s * (1 - f))
    return [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6]
  }
  const on3 = (f: (a: number, b: number, c: number) => number[]) => (args: Value[]) => {
    const a = toArray(args[0], 'float64')
    const v = a.values()
    const out: number[] = []
    for (let i = 0; i < v.length; i += 3) out.push(...f(v[i], v[i + 1], v[i + 2]))
    return floats(out, a.shape)
  }
  const dict = (obj: Record<string, string>): Value => ({ k: 'dict', d: PyDict.from(Object.entries(obj).map(([a, b]) => [py.str(a), py.str(b)])) })
  return {
    Normalize: norm('Normalize'), LogNorm: norm('LogNorm'), NoNorm: norm('NoNorm'),
    TwoSlopeNorm: typ('TwoSlopeNorm', (args, kw) => {
      const c = args[0] ?? kw.vcenter
      if (!c) throw h.err('TypeError', "TwoSlopeNorm.__init__() missing 1 required positional argument: 'vcenter'")
      const o = new NormObj('TwoSlopeNorm', optNum(h, args[1] ?? kw.vmin), optNum(h, args[2] ?? kw.vmax))
      o.center = h.num(c)
      if (o.vmin !== null && o.vmax !== null && !(o.vmin < o.center && o.center < o.vmax)) throw h.err('ValueError', 'vmin, vcenter, and vmax must be in ascending order')
      return py.obj(o)
    }),
    CenteredNorm: typ('CenteredNorm', (args, kw) => {
      const c = optNum(h, args[0] ?? kw.vcenter) ?? 0
      const r = optNum(h, args[1] ?? kw.halfrange)
      return py.obj(new NormObj('Normalize', r === null ? null : c - r, r === null ? null : c + r))
    }),
    BoundaryNorm: typ('BoundaryNorm', (args, kw) => {
      const b = nums(h, args[0] ?? kw.boundaries, 'boundaries')
      const n = h.toInt(args[1] ?? kw.ncolors)
      if (b.length < 2) throw h.err('ValueError', 'You must provide at least 2 boundaries (1 region)')
      if (n < b.length - 1) throw h.err('ValueError', `There are ${b.length - 1} color bins including extensions, but ncolors = ${n}; ncolors must equal or exceed the number of bins`)
      const o = new NormObj('BoundaryNorm', b[0], b[b.length - 1])
      o.bounds = b
      o.ncolors = n
      return py.obj(o)
    }),
    ListedColormap: typ('ListedColormap', listed),
    Colormap: typ('Colormap', () => { throw h.err('NotImplementedError', 'Colormap: use ListedColormap or LinearSegmentedColormap.from_list') }),
    LinearSegmentedColormap: { k: 'module', name: 'LinearSegmentedColormap', attrs: { from_list: fromList } },
    to_rgb: fnv('to_rgb', (args) => py.tuple(rgbaOf(toHexName(args[0])).slice(0, 3).map(py.float))),
    to_rgba: fnv('to_rgba', (args, kw) => {
      const c = rgbaOf(toHexName(args[0]))
      const a = optNum(h, args[1] ?? kw.alpha)
      return py.tuple([c[0], c[1], c[2], a ?? c[3]].map(py.float))
    }),
    to_hex: fnv('to_hex', (args, kw) => {
      const c = toHexName(args[0])
      const keep = flag(h, kw.keep_alpha, false)
      const [r, g, b, a] = rgbaOf(c)
      return py.str(`#${hex2(r)}${hex2(g)}${hex2(b)}${keep ? hex2(a) : ''}`)
    }),
    rgb2hex: fnv('rgb2hex', (args) => py.str(toHexName(args[0]).slice(0, 7))),
    is_color_like: fnv('is_color_like', (args) => {
      try {
        toColor(h, args[0])
        return py.bool(true)
      } catch {
        return py.bool(false)
      }
    }),
    rgb_to_hsv: fnv('rgb_to_hsv', on3(rgb2hsv)),
    hsv_to_rgb: fnv('hsv_to_rgb', on3(hsv2rgb)),
    CSS4_COLORS: dict(Object.fromEntries(CSS4.split(' ').map((e) => { const [n, c] = e.split(':'); return [n, `#${c.toUpperCase()}`] }))),
    TABLEAU_COLORS: dict(Object.fromEntries(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'].map((n, i) => [`tab:${n}`, CYCLE[i]]))),
    BASE_COLORS: { k: 'dict', d: PyDict.from(Object.entries({ b: [0, 0, 1], g: [0, 0.5, 0], r: [1, 0, 0], c: [0, 0.75, 0.75], m: [0.75, 0, 0.75], y: [0.75, 0.75, 0], k: [0, 0, 0], w: [1, 1, 1] }).map(([a, b]) => [py.str(a), py.tuple(b.map(py.float))])) },
  }
}

const tickerModule = (h: Host): Record<string, Value> => {
  const loc = (cls: string, make: (a: Value[], k: Kw) => Locator): Value => typ(cls, (a, k) => py.obj(new TickerObj(cls, make(a, k), null)))
  const fmt = (cls: string, make: (a: Value[], k: Kw) => Formatter): Value => typ(cls, (a, k) => py.obj(new TickerObj(cls, null, make(a, k))))
  return {
    AutoLocator: loc('AutoLocator', () => ({ k: 'auto' })),
    MaxNLocator: loc('MaxNLocator', (a, k) => {
      const nb = a[0] ?? k.nbins
      const steps = k.steps && !isNone(k.steps) ? validateSteps(nums(h, k.steps, 'steps')) : MAXN_STEPS
      const prune = optStr(h, k.prune) as MaxN['prune']
      return { k: 'maxn', m: { nbins: isNone(nb) ? 10 : nb!.k === 'str' ? 'auto' : h.toInt(nb!), steps, integer: flag(h, k.integer, false), minN: Math.max(1, optNum(h, k.min_n_ticks) ?? 2), prune, symmetric: flag(h, k.symmetric, false) } }
    }),
    MultipleLocator: loc('MultipleLocator', (a, k) => ({ k: 'multiple', base: optNum(h, a[0] ?? k.base) ?? 1, offset: optNum(h, a[1] ?? k.offset) ?? 0 })),
    FixedLocator: loc('FixedLocator', (a, k) => ({ k: 'fixed', at: nums(h, a[0] ?? k.locs, 'locs'), nbins: optNum(h, a[1] ?? k.nbins) ?? undefined })),
    NullLocator: loc('NullLocator', () => ({ k: 'null' })),
    LinearLocator: loc('LinearLocator', (a, k) => ({ k: 'linear', n: optNum(h, a[0] ?? k.numticks) ?? 11 })),
    LogLocator: loc('LogLocator', (a, k) => ({ k: 'log', base: optNum(h, a[0] ?? k.base) ?? 10, subs: isNone(a[1] ?? k.subs) ? null : (a[1] ?? k.subs)!.k === 'str' ? 'auto' : nums(h, (a[1] ?? k.subs)!, 'subs') })),
    AutoMinorLocator: loc('AutoMinorLocator', (a, k) => ({ k: 'minor', ndivs: optNum(h, a[0] ?? k.n) })),
    ScalarFormatter: fmt('ScalarFormatter', (a, k) => ({ k: 'scalar', useOffset: flag(h, a[0] ?? k.useOffset, true), limits: null, sci: true })),
    NullFormatter: fmt('NullFormatter', () => ({ k: 'null' })),
    FuncFormatter: fmt('FuncFormatter', (a, k) => ({ k: 'func', fn: (a[0] ?? k.func)! })),
    FormatStrFormatter: fmt('FormatStrFormatter', (a, k) => ({ k: 'fmt', fmt: h.str((a[0] ?? k.fmt)!) })),
    StrMethodFormatter: fmt('StrMethodFormatter', (a, k) => ({ k: 'str', fmt: h.str((a[0] ?? k.fmt)!) })),
    FixedFormatter: fmt('FixedFormatter', (a, k) => ({ k: 'fixed', labels: h.iterate((a[0] ?? k.seq)!).map((v) => h.str(v)) })),
    PercentFormatter: fmt('PercentFormatter', (a, k) => ({ k: 'percent', xmax: optNum(h, a[0] ?? k.xmax) ?? 100, decimals: optNum(h, a[1] ?? k.decimals), symbol: optStr(h, a[2] ?? k.symbol) ?? '%' })),
    LogFormatter: fmt('LogFormatter', (a, k) => ({ k: 'log', base: optNum(h, a[0] ?? k.base) ?? 10 })),
    LogFormatterSciNotation: fmt('LogFormatterSciNotation', (a, k) => ({ k: 'log', base: optNum(h, a[0] ?? k.base) ?? 10 })),
    LogFormatterMathtext: fmt('LogFormatterMathtext', (a, k) => ({ k: 'log', base: optNum(h, a[0] ?? k.base) ?? 10 })),
  }
}

const COLOR_SETTERS = ['autumn', 'bone', 'cool', 'copper', 'flag', 'gray', 'hot', 'hsv', 'jet', 'pink', 'prism', 'spring', 'summer', 'winter', 'magma', 'inferno', 'plasma', 'viridis', 'nipy_spectral']

const pyplot = (h: Host, patches: Record<string, Value>, cm: Value, colors: Record<string, Value>, ticker: Record<string, Value>, rcParams: RcParams): Value => {
  const attrs: Record<string, Value> = {}
  // every Axes method has a pyplot twin that acts on the current Axes; these also make the current image
  const sciTwins = new Set(['imshow', 'pcolor', 'pcolormesh', 'contour', 'contourf', 'scatter', 'hexbin', 'tricontour', 'tricontourf', 'tripcolor', 'spy'])
  const twins: Record<string, string> = {
    plot: 'plot', scatter: 'scatter', bar: 'bar', barh: 'barh', hist: 'hist', imshow: 'imshow', axhline: 'axhline', axvline: 'axvline', axhspan: 'axhspan', axvspan: 'axvspan', axline: 'axline',
    text: 'text', fill_between: 'fill_between', fill_betweenx: 'fill_betweenx', fill: 'fill', legend: 'legend', grid: 'grid', axis: 'axis', arrow: 'arrow', annotate: 'annotate', bar_label: 'bar_label',
    contour: 'contour', contourf: 'contourf', clabel: 'clabel', tricontour: 'tricontour', tricontourf: 'tricontourf', triplot: 'triplot', tripcolor: 'tripcolor', pie: 'pie', errorbar: 'errorbar', boxplot: 'boxplot', violinplot: 'violinplot', step: 'step', stem: 'stem', stairs: 'stairs', stackplot: 'stackplot',
    pcolor: 'pcolor', pcolormesh: 'pcolormesh', hist2d: 'hist2d', hexbin: 'hexbin', quiver: 'quiver', hlines: 'hlines', vlines: 'vlines', eventplot: 'eventplot', ecdf: 'ecdf', broken_barh: 'broken_barh', spy: 'spy', streamplot: 'streamplot', barbs: 'barbs', table: 'table',
    semilogx: 'semilogx', semilogy: 'semilogy', loglog: 'loglog', margins: 'margins', autoscale: 'autoscale', tick_params: 'tick_params', ticklabel_format: 'ticklabel_format', locator_params: 'locator_params', minorticks_on: 'minorticks_on', minorticks_off: 'minorticks_off',
    twinx: 'twinx', twiny: 'twiny', cla: 'cla', xscale: 'set_xscale', yscale: 'set_yscale',
  }
  for (const [name, method] of Object.entries(twins)) {
    attrs[name] = fnv(name, (args, kw) => {
      const ax = gca(h)
      const m = ax.method(method)
      if (!m) throw h.err('AttributeError', `'${ax.cls}' object has no attribute '${method}'`)
      const out = m(ax, args, { ...kw }, h)
      if (sciTwins.has(name) && out.k === 'obj' && out.o instanceof ArtistObj && 'norm' in out.o) ax.current = out.o as ArtistObj & Mapped
      if (name === 'hist2d' && out.k === 'tuple') ax.current = (out.items[3] as unknown as { o: MeshObj }).o
      return out
    })
  }
  const label = (name: string, method: string, arg: string) => {
    attrs[name] = fnv(name, (args, kw) => {
      if (!args.length && !(arg in kw) && !(method === 'set_title' && 'label' in kw)) throw h.err('TypeError', `${name}() missing 1 required positional argument: '${arg}'`)
      const ax = gca(h)
      return ax.method(method)!(ax, args, { ...kw }, h)
    })
  }
  label('title', 'set_title', 'label')
  label('xlabel', 'set_xlabel', 'xlabel')
  label('ylabel', 'set_ylabel', 'ylabel')
  for (const n of ['x', 'y'] as const) {
    attrs[`${n}lim`] = fnv(`${n}lim`, (args, kw) => {
      const ax = gca(h)
      if (!args.length && !Object.keys(kw).length) return py.tuple(ax.axisOf(n).view().map(py.float))
      return limSetter(n)(ax, args, { ...kw }, h)
    })
    attrs[`${n}ticks`] = fnv(`${n}ticks`, (args, kw0) => {
      const kw = { ...kw0 }
      const ax = gca(h)
      const axis = ax.axisOf(n)
      const tv = args[0] ?? pop(kw, 'ticks')
      const lv = args[1] ?? pop(kw, 'labels')
      const minor = flag(h, pop(kw, 'minor'), false)
      let locs: Value
      if (isNone(tv)) {
        if (!isNone(lv)) throw h.err('TypeError', `${n}ticks(): Parameter 'labels' can't be set without setting 'ticks'`)
        locs = floats(axis.current(h).locs)
      } else {
        const at = axis.setTicks(h, tv!, undefined, minor)
        locs = py.list(at.map((_, i) => py.obj(new TickObj(axis, i))))
      }
      if (isNone(lv)) axis.applyLabelKw(h, kw)
      else axis.setTickLabels(h, lv!, minor, kw)
      emitFig(h, ax.fig, ax)
      return py.tuple([locs, py.obj(new SilentList('Text major ticklabel', axis.tickLabels(h, minor)))])
    })
  }
  attrs.figure = fnv('figure', (args, kw0) => {
    const kw = { ...kw0 }
    const num = args[0] ?? pop(kw, 'num')
    if (args[1]) kw.figsize = args[1]
    const k = Object.keys(kw).find((x) => !FIG_KEYS.includes(x))
    if (k) throw h.err('TypeError', `Figure.__init__() got an unexpected keyword argument '${k}'`)
    return py.obj(figureFrom(h, num, kw))
  })
  attrs.subplot = fnv('subplot', (args, kw) => py.obj(gcf(h).gridAxes(h, args, { ...kw }, true)))
  attrs.subplots = fnv('subplots', (args, kw0) => {
    const kw = { ...kw0 }
    const figKw: Kw = {}
    for (const k of FIG_KEYS) if (k in kw) { figKw[k] = kw[k]; delete kw[k] }
    const f = figureFrom(h, pop(figKw, 'num'), figKw)
    return subplotsOn(h, f, args, kw, true)
  })
  attrs.subplot_mosaic = fnv('subplot_mosaic', (args, kw0) => {
    const kw = { ...kw0 }
    const figKw: Kw = {}
    for (const k of FIG_KEYS) if (k in kw) { figKw[k] = kw[k]; delete kw[k] }
    const f = figureFrom(h, pop(figKw, 'num'), figKw)
    return py.tuple([py.obj(f), mosaicOn(h, f, args[0] ?? pop(kw, 'mosaic')!, kw)])
  })
  attrs.axes = fnv('axes', (args, kw0) => {
    const kw = { ...kw0 }
    const a = args[0] ?? pop(kw, 'arg')
    const f = gcf(h)
    if (isNone(a)) return py.obj(f.gridAxes(h, [], kw, false))
    if (a!.k === 'obj' && a!.o instanceof Axes) {
      f.cur = a!.o
      return a!
    }
    const add = f.getAttr('add_axes', h) as { call: (x: Value[], y: Kw) => Value }
    return add.call([a!], kw)
  })
  attrs.subplot2grid = fnv('subplot2grid', (args, kw0) => {
    const kw = { ...kw0 }
    const [r, c] = h.iterate(args[0] ?? pop(kw, 'shape')!).map((v) => h.toInt(v))
    const [lr, lc] = h.iterate(args[1] ?? pop(kw, 'loc')!).map((v) => h.toInt(v))
    const rs = h.toInt(args[2] ?? pop(kw, 'rowspan') ?? py.int(1))
    const cs = h.toInt(args[3] ?? pop(kw, 'colspan') ?? py.int(1))
    const figV = pop(kw, 'fig')
    const f = figV && figV.k === 'obj' && figV.o instanceof Figure ? figV.o : gcf(h)
    if (lr >= r || lc >= c || lr < 0 || lc < 0) throw h.err('IndexError', 'GridSpec slice would result in no space allocated for subplot')
    const ss = new SubplotSpec(new GridSpec(f, r, c), lr, Math.min(lr + rs, r), lc, Math.min(lc + cs, c))
    return py.obj(f.gridAxes(h, [py.obj(ss)], kw, false))
  })
  attrs.GridSpec = typ('GridSpec', (args, kw0) => {
    const kw = { ...kw0 }
    const figV = pop(kw, 'figure')
    return py.obj(gridSpecOf(h, figV && figV.k === 'obj' && figV.o instanceof Figure ? figV.o : null, args, kw))
  })
  attrs.gcf = fnv('gcf', () => py.obj(gcf(h)))
  attrs.gca = fnv('gca', () => py.obj(gca(h)))
  attrs.sca = fnv('sca', (args) => {
    const a = args[0]
    if (a.k !== 'obj' || !(a.o instanceof Axes)) throw h.err('AttributeError', `'${a.k}' object has no attribute 'get_figure'`)
    stateOf(h).cur = a.o.fig
    a.o.fig.cur = a.o
    return py.NONE
  })
  attrs.suptitle = fnv('suptitle', (args, kw) => gcf(h).supText(h, 'title', args, kw))
  attrs.supxlabel = fnv('supxlabel', (args, kw) => gcf(h).supText(h, 'xlabel', args, kw))
  attrs.supylabel = fnv('supylabel', (args, kw) => gcf(h).supText(h, 'ylabel', args, kw))
  const figFn = (name: string, target = name) => {
    attrs[name] = fnv(name, (args, kw) => (gcf(h).getAttr(target, h) as { call: (a: Value[], k: Kw) => Value }).call(args, kw))
  }
  figFn('figtext', 'text')
  figFn('figlegend', 'legend')
  figFn('tight_layout')
  figFn('subplots_adjust')
  attrs.savefig = fnv('savefig', (args, kw) => (gcf(h).getAttr('savefig', h) as { call: (a: Value[], k: Kw) => Value }).call(args, kw))
  attrs.clf = fnv('clf', () => {
    const f = gcf(h)
    f.clear()
    emitFig(h, f, null)
    return py.NONE
  })
  attrs.delaxes = fnv('delaxes', (args) => {
    const f = gcf(h)
    return (f.getAttr('delaxes', h) as { call: (a: Value[], k: Kw) => Value }).call(args.length ? args : [py.obj(f.gca())], {})
  })
  attrs.colorbar = fnv('colorbar', (args, kw0) => {
    const kw = { ...kw0 }
    let m = args[0] ?? pop(kw, 'mappable')
    const cax = args[1] ?? pop(kw, 'cax')
    const axv = args[2] ?? pop(kw, 'ax')
    const f = gcf(h)
    if (isNone(m)) {
      const cur = f.cur?.current ?? [...f.axes].reverse().map((a) => a.current).find((c) => c)
      if (!cur) throw h.err('RuntimeError', 'No mappable was found to use for colorbar creation. First define a mappable such as an image (with imshow) or a contour set (with contourf).')
      m = py.obj(cur)
    }
    return makeColorbar(h, f, m!, cax, axv, kw)
  })
  const curImage = (): (ArtistObj & Mapped) | null => {
    const f = stateOf(h).cur
    if (!f || f.closed) return null
    return f.cur?.current ?? [...f.axes].reverse().map((a) => a.current).find((c) => c) ?? null
  }
  attrs.gci = fnv('gci', () => { const c = curImage(); return c ? py.obj(c) : py.NONE })
  attrs.sci = fnv('sci', (args) => {
    const a = args[0]
    if (a.k !== 'obj' || !(a.o instanceof ArtistObj) || !('norm' in a.o)) throw h.err('TypeError', 'sci() needs a colour-mapped artist')
    if (a.o.ax) a.o.ax.current = a.o as ArtistObj & Mapped
    return py.NONE
  })
  attrs.clim = fnv('clim', (args, kw) => {
    const im = curImage()
    if (!im) throw h.err('RuntimeError', 'You must first define an image, e.g., with imshow')
    MAPPED_PROPS.clim.s!(im, args, h, kw)
    im.touch(h)
    return py.NONE
  })
  const getCmap = (args: Value[], kw: Kw): Value => {
    const nameV = args[0] ?? kw.name
    const lutN = optNum(h, args[1] ?? kw.lut)
    let lut = isNone(nameV) ? cmapArg(h, undefined) : cmapArg(h, nameV, 'name')
    if (lutN !== null) lut = lut.listed ? { ...lut, n: lutN, rgb: resampleListed(lut, lutN), hex: undefined } : makeLut(lut.name, lutN) ?? { ...lut, n: lutN, rgb: resampleListed(lut, lutN), hex: undefined }
    return py.obj(new ColormapObj({ ...lut }))
  }
  attrs.get_cmap = fnv('get_cmap', getCmap)
  const setCmap = (v: Value) => {
    const lut = cmapArg(h, v)
    stateOf(h).rc.set('image.cmap', py.str(lut.name))
    const im = curImage()
    if (im) {
      im.lut = lut
      im.recolor()
      im.touch(h)
    }
  }
  attrs.set_cmap = fnv('set_cmap', (args, kw) => { setCmap(args[0] ?? kw.cmap); return py.NONE })
  for (const name of COLOR_SETTERS) attrs[name] = fnv(name, () => { setCmap(py.str(name)); return py.NONE })
  attrs.show = fnv('show', () => {
    const s = stateOf(h)
    for (const f of s.figs) if (!f.closed) f.shown = true
    return py.NONE
  })
  attrs.close = fnv('close', (args, kw) => {
    const s = stateOf(h)
    const which = args[0] ?? kw.fig
    const finish = (f: Figure) => {
      f.closed = true
      if (s.cur === f) s.cur = [...s.figs].reverse().find((x) => !x.closed && !x.shown) ?? null
    }
    if (isNone(which)) { if (s.cur) finish(s.cur) } else if (which!.k === 'str') {
      if (which!.v === 'all') s.figs.forEach(finish)
      else s.figs.filter((f) => f.label === (which as { v: string }).v).forEach(finish)
    } else if (which!.k === 'obj' && which!.o instanceof Figure) finish(which!.o)
    else if (isNum(which!)) s.figs.filter((f) => f.spec.num === h.num(which!)).forEach(finish)
    else throw h.err('TypeError', 'close() argument must be a Figure, an int, a string, or None')
    return py.NONE
  })
  attrs.pause = fnv('pause', () => py.NONE)
  attrs.draw = fnv('draw', () => py.NONE)
  attrs.draw_all = fnv('draw_all', () => py.NONE)
  attrs.draw_if_interactive = fnv('draw_if_interactive', () => py.NONE)
  attrs.ion = fnv('ion', () => { stateOf(h).interactive = true; return py.NONE })
  attrs.ioff = fnv('ioff', () => { stateOf(h).interactive = false; return py.NONE })
  attrs.isinteractive = fnv('isinteractive', () => py.bool(stateOf(h).interactive))
  attrs.interactive = fnv('interactive', (args) => { stateOf(h).interactive = h.truthy(args[0]); return py.NONE })
  attrs.get_backend = fnv('get_backend', () => py.str('agg'))
  attrs.switch_backend = fnv('switch_backend', () => py.NONE)
  attrs.fignum_exists = fnv('fignum_exists', (args) => py.bool(stateOf(h).figs.some((f) => !f.closed && f.spec.num === h.num(args[0]))))
  attrs.get_fignums = fnv('get_fignums', () => py.list(stateOf(h).figs.filter((f) => !f.closed).map((f) => py.int(f.spec.num))))
  attrs.get_figlabels = fnv('get_figlabels', () => py.list(stateOf(h).figs.filter((f) => !f.closed).map((f) => py.str(f.label))))
  attrs.figaspect = fnv('figaspect', (args) => floats(figaspect(h, args[0])))
  attrs.matshow = fnv('matshow', (args, kw0) => {
    const kw = { ...kw0 }
    const A = args[0] ?? pop(kw, 'A')
    const fignum = args[1] ?? pop(kw, 'fignum')
    let ax: Axes
    if (fignum && fignum.k === 'int' && fignum.v === 0) ax = gca(h)
    else {
      const f = figureFrom(h, fignum, { figsize: floats(figaspect(h, A!)) })
      ax = makeAxes(h, f, null, [0.15, 0.09, 0.775, 0.775], undefined, false)
      f.cur = ax
    }
    const im = matshow(ax, [A!], kw, h)
    ax.current = (im as unknown as { o: ImageObj }).o
    return im
  })
  attrs.polar = fnv('polar', () => { throw h.err('NotImplementedError', 'polar axes are not drawn in this sandbox') })
  attrs.box = fnv('box', (args, kw) => {
    const ax = gca(h)
    const on = args[0] ?? kw.on
    ax.frame = isNone(on) ? !ax.frame : h.truthy(on!)
    emitFig(h, ax.fig, ax)
    return py.NONE
  })
  attrs.setp = fnv('setp', (args0, kw) => {
    const obj = args0[0]
    if (!obj) throw h.err('TypeError', "setp() missing 1 required positional argument: 'obj'")
    const list = obj.k === 'list' || obj.k === 'tuple' ? obj.items : obj.k === 'obj' && obj.o.iter && !(obj.o instanceof ArtistObj) ? obj.o.iter(h) : obj.k === 'obj' && obj.o instanceof BarObj ? obj.o.iter() : [obj]
    const k: Kw = { ...kw }
    const rest = args0.slice(1)
    if (rest.length % 2) throw h.err('ValueError', 'The set args must be string, value pairs')
    for (let i = 0; i < rest.length; i += 2) k[h.str(rest[i])] = rest[i + 1]
    const out: Value[] = []
    for (const o of list) {
      for (const [name, v] of Object.entries(k)) {
        if (o.k !== 'obj') throw h.err('AttributeError', `'${o.k}' object has no attribute 'set'`)
        const f = o.o.getAttr?.(`set_${name}`, h)
        if (!f || f.k !== 'fn') throw h.err('AttributeError', `${o.o.cls}.set() got an unexpected keyword argument '${name}'`)
        f.call([v], {})
        out.push(py.NONE)
      }
    }
    return py.list(out)
  })
  const getp = fnv('getp', (args) => {
    const [o, prop] = args
    if (o.k !== 'obj') throw h.err('AttributeError', `'${o.k}' object has no attribute 'get'`)
    const f = o.o.getAttr?.(`get_${h.str(prop)}`, h)
    if (!f || f.k !== 'fn') throw h.err('AttributeError', `'${o.o.cls}' object has no attribute 'get_${h.str(prop)}'`)
    return f.call([], {})
  })
  attrs.getp = getp
  attrs.get = getp
  attrs.rc = fnv('rc', (args, kw) => rcCall(h, args, kw))
  attrs.rc_context = fnv('rc_context', (args, kw) => rcContext(h, args, kw))
  attrs.rcdefaults = fnv('rcdefaults', () => { stateOf(h).rc.clear(); return py.NONE })
  attrs.rcParams = py.obj(rcParams)
  attrs.rcParamsDefault = py.obj(new RcParams(h, new Map()))
  attrs.cycler = fnv('cycler', (args, kw) => py.obj(cyclerOf(h, args, kw)))
  attrs.imsave = imsave(h)
  attrs.imread = imread(h)
  attrs.cm = cm
  attrs.colormaps = py.obj(new ColormapRegistry(h))
  attrs.style = styleModule(h)
  attrs.Normalize = colors.Normalize
  attrs.Line2D = typ('Line2D', (args, kw) => line2dCtor(h, args, kw))
  attrs.Text = typ('Text', (args, kw) => textCtor(h, args, kw))
  attrs.Annotation = typ('Annotation', () => { throw h.err('NotImplementedError', 'Annotation: use ax.annotate(...)') })
  attrs.Figure = typ('Figure', (_a, kw) => {
    const f = new Figure(h, 0, kw.figsize ? (nums(h, kw.figsize, 'figsize') as [number, number]) : (nums(h, rcv(h, 'figure.figsize'), 'figsize') as [number, number]), rcc(h, 'figure.facecolor'))
    return py.obj(f)
  })
  for (const n of ['Circle', 'Rectangle', 'Polygon', 'Arrow']) attrs[n] = patches[n]
  for (const n of ['AutoLocator', 'MaxNLocator', 'MultipleLocator', 'FixedLocator', 'NullLocator', 'LinearLocator', 'LogLocator', 'FuncFormatter', 'FormatStrFormatter', 'FixedFormatter', 'NullFormatter', 'ScalarFormatter', 'LogFormatter', 'LogFormatterMathtext', 'LogFormatterExponent']) if (ticker[n]) attrs[n] = ticker[n]
  return { k: 'module', name: 'matplotlib.pyplot', attrs }
}

const line2dCtor = (h: Host, args: Value[], kw0: Kw): Value => {
  const kw = { ...kw0 }
  const xv = args[0] ?? pop(kw, 'xdata')
  const yv = args[1] ?? pop(kw, 'ydata')
  if (!xv || !yv) throw h.err('TypeError', "Line2D.__init__() missing required positional arguments: 'xdata' and 'ydata'")
  const xa = toArray(xv)
  const ya = toArray(yv)
  const o = new LineObj(newLine(h, xa.values(), ya.values(), rcc(h, 'lines.color')))
  o.setCoord(0, xv)
  o.setCoord(1, yv)
  o.raw = rcv(h, 'lines.color')
  o.applyKw(kw, h)
  return py.obj(o)
}

const textCtor = (h: Host, args: Value[], kw0: Kw): Value => {
  const kw = { ...kw0 }
  const x = optNum(h, args[0] ?? pop(kw, 'x')) ?? 0
  const y = optNum(h, args[1] ?? pop(kw, 'y')) ?? 0
  const s = optStr(h, args[2] ?? pop(kw, 'text')) ?? ''
  const o = new TextObj(newText(h, x, y, s), String(x), String(y))
  o.applyKw(kw, h)
  return py.obj(o)
}

class ColormapRegistry extends PyObj {
  readonly cls = 'ColormapRegistry'
  h: Host
  constructor(h: Host) {
    super()
    this.h = h
  }
  repr() { return `ColormapRegistry; available colormaps:\n${ALL_CMAPS.map((c) => `'${c}'`).join(', ')}` }
  len() { return ALL_CMAPS.length }
  iter(): Value[] { return ALL_CMAPS.map(py.str) }
  contains(v: Value) { return v.k === 'str' && isCmap(v.v) }
  getItem(idx: Value, h: Host): Value {
    const k = h.str(idx)
    const l = getLut(k)
    if (!l) throw h.err('KeyError', `'${k} is not a known colormap name'`)
    return py.obj(new ColormapObj({ ...l }))
  }
  getAttr(name: string): Value | undefined {
    if (name === 'get_cmap') return fnv(name, (args) => py.obj(new ColormapObj({ ...cmapArg(this.h, args[0]) })))
    if (name === 'keys') return fnv(name, () => py.list(this.iter()))
    return undefined
  }
}

const styleModule = (h: Host): Value => ({
  k: 'module',
  name: 'matplotlib.style',
  attrs: {
    available: py.list(['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'petroff10', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10'].map(py.str)),
    use: fnv('use', (args) => {
      const s = h.str(args[0])
      h.print(`(sandbox) style '${s}' only changes the look in real matplotlib; the sandbox keeps the default style\n`)
      return py.NONE
    }),
    context: fnv('context', () => rcContext(h, [{ k: 'dict', d: new PyDict() }], {})),
  },
})

export const MATPLOTLIB: PyLib = {
  modules: ['matplotlib', 'matplotlib.pyplot', 'matplotlib.cm', 'matplotlib.colors', 'matplotlib.patches', 'matplotlib.image', 'matplotlib.lines', 'matplotlib.text', 'matplotlib.ticker', 'matplotlib.gridspec', 'matplotlib.tri', 'matplotlib.collections', 'matplotlib.figure', 'matplotlib.axes', 'matplotlib.style', 'mpl_toolkits', 'mpl_toolkits.mplot3d', 'mpl_toolkits.mplot3d.axes3d', 'seaborn'],
  load(h) {
    const patchesAttrs = patchTypes(h)
    const colors = colorsModule(h)
    const tickerAttrs = tickerModule(h)
    const rcParams = new RcParams(h)
    const cmAttrs = lazyCmaps()
    cmAttrs.ScalarMappable = typ('ScalarMappable', (args, kw) => {
      const nv = args[0] ?? kw.norm
      const lut = cmapArg(h, args[1] ?? kw.cmap)
      const norm = nv && !isNone(nv) && nv.k === 'obj' && nv.o instanceof NormObj ? nv.o : new NormObj('Normalize', null, null)
      return py.obj(new ScalarMappable(lut, norm))
    })
    const cm: Value = { k: 'module', name: 'matplotlib.cm', attrs: cmAttrs }
    const mod = (name: string, attrs: Record<string, Value>): Value => ({ k: 'module', name, attrs })
    const colorsMod = mod('matplotlib.colors', colors)
    const patches = mod('matplotlib.patches', patchesAttrs)
    const image = mod('matplotlib.image', { imread: imread(h), imsave: imsave(h) })
    const lines = mod('matplotlib.lines', { Line2D: typ('Line2D', (a, k) => line2dCtor(h, a, k)) })
    const textMod = mod('matplotlib.text', { Text: typ('Text', (a, k) => textCtor(h, a, k)) })
    const ticker = mod('matplotlib.ticker', tickerAttrs)
    const gridspec = mod('matplotlib.gridspec', {
      GridSpec: typ('GridSpec', (args, kw0) => {
        const kw = { ...kw0 }
        const figV = pop(kw, 'figure')
        return py.obj(gridSpecOf(h, figV && figV.k === 'obj' && figV.o instanceof Figure ? figV.o : null, args, kw))
      }),
      GridSpecFromSubplotSpec: typ('GridSpecFromSubplotSpec', (args, kw) => {
        const parent = kw.subplot_spec
        if (!parent || parent.k !== 'obj' || !(parent.o instanceof SubplotSpec)) throw h.err('TypeError', 'GridSpecFromSubplotSpec needs subplot_spec=')
        const g = new GridSpec(parent.o.gs.fig, h.toInt(args[0] ?? kw.nrows), h.toInt(args[1] ?? kw.ncols), kw.width_ratios ? nums(h, kw.width_ratios, 'width_ratios') : undefined, kw.height_ratios ? nums(h, kw.height_ratios, 'height_ratios') : undefined)
        g.parent = parent.o
        if (kw.wspace) g.own.wspace = h.num(kw.wspace)
        if (kw.hspace) g.own.hspace = h.num(kw.hspace)
        return py.obj(g)
      }),
    })
    const tri = mod('matplotlib.tri', {
      Triangulation: typ('Triangulation', (args, kw) => py.obj(triangulation(h, args[0] ?? kw.x, args[1] ?? kw.y, args[2] ?? kw.triangles))),
    })
    const collections = mod('matplotlib.collections', {
      LineCollection: typ('LineCollection', (args, kw0) => {
        const kw = { ...kw0 }
        const segs = args[0] ?? pop(kw, 'segments')!
        const c = pop(kw, 'colors', 'color')
        const spec = newSegs(c ? toColors(h, c) : [rcc(h, 'lines.color')], rcn(h, 'lines.linewidth'))
        spec.paths = h.iterate(segs).map((s) => nums(h, s, 'segments'))
        const o = new SegsObj('LineCollection', spec)
        o.applyKw(kw, h)
        return py.obj(o)
      }),
      PolyCollection: typ('PolyCollection', (args, kw0) => {
        const kw = { ...kw0 }
        const verts = args[0] ?? pop(kw, 'verts')!
        const spec = newPoly(rcc(h, 'patch.facecolor'), 'none', rcn(h, 'patch.linewidth'))
        spec.paths = h.iterate(verts).map((s) => nums(h, s, 'verts'))
        const o = new PolyObj('PolyCollection', spec)
        o.applyKw(kw, h)
        return py.obj(o)
      }),
    })
    const figureMod = mod('matplotlib.figure', {
      Figure: typ('Figure', (_a, kw) => py.obj(new Figure(h, 0, kw.figsize ? (nums(h, kw.figsize, 'figsize') as [number, number]) : (nums(h, rcv(h, 'figure.figsize'), 'figsize') as [number, number]), rcc(h, 'figure.facecolor')))),
    })
    const axesMod = mod('matplotlib.axes', { Axes: typ('Axes', () => { throw h.err('NotImplementedError', 'matplotlib.axes.Axes: use plt.subplots() or fig.add_subplot()') }) })
    const style = styleModule(h)
    const plt = pyplot(h, patchesAttrs, cm, colors, tickerAttrs, rcParams)
    const axes3dMod = mod('mpl_toolkits.mplot3d.axes3d', {
      Axes3D: typ('Axes3D', () => { throw h.err('NotImplementedError', "Axes3D(fig) directly: use fig.add_subplot(projection='3d')") }),
      get_test_data: fnv('get_test_data', (args, kw) => {
        const delta = optNum(h, args[0] ?? kw.delta) ?? 0.05
        const n = Math.ceil(6 / delta)
        const c = Array.from({ length: n }, (_, i) => -3 + i * delta)
        const X = Array.from({ length: n * n }, (_, i) => c[i % n])
        const Y = Array.from({ length: n * n }, (_, i) => c[Math.floor(i / n)])
        const Z = X.map((x, i) => {
          const y = Y[i]
          const Z1 = Math.exp(-(x ** 2 + y ** 2) / 2) / (2 * Math.PI)
          const Z2 = Math.exp(-(((x - 1) / 1.5) ** 2 + ((y - 1) / 0.5) ** 2) / 2) / (2 * Math.PI * 0.5 * 1.5)
          return (Z2 - Z1) * 500
        })
        return py.tuple([floats(X.map((v) => v * 10), [n, n]), floats(Y.map((v) => v * 10), [n, n]), floats(Z, [n, n])])
      }),
    })
    const mplot3d = mod('mpl_toolkits.mplot3d', { Axes3D: (axes3dMod as { attrs: Record<string, Value> }).attrs.Axes3D, axes3d: axes3dMod })
    const toolkits = mod('mpl_toolkits', { mplot3d })
    const root = mod('matplotlib', {
      pyplot: plt, cm, colors: colorsMod, patches, image, lines, text: textMod, ticker, gridspec, tri, collections, figure: figureMod, axes: axesMod, style,
      rcParams: py.obj(rcParams), rcParamsDefault: (plt as { attrs: Record<string, Value> }).attrs.rcParamsDefault,
      rc: (plt as { attrs: Record<string, Value> }).attrs.rc, rc_context: (plt as { attrs: Record<string, Value> }).attrs.rc_context, rcdefaults: (plt as { attrs: Record<string, Value> }).attrs.rcdefaults,
      use: fnv('use', () => py.NONE), get_backend: fnv('get_backend', () => py.str('agg')), colormaps: (plt as { attrs: Record<string, Value> }).attrs.colormaps,
      cycler: (plt as { attrs: Record<string, Value> }).attrs.cycler, __version__: py.str('3.10.8'),
    })
    return {
      matplotlib: root, 'matplotlib.pyplot': plt, 'matplotlib.cm': cm, 'matplotlib.colors': colorsMod, 'matplotlib.patches': patches, 'matplotlib.image': image, 'matplotlib.lines': lines, 'matplotlib.text': textMod,
      'matplotlib.ticker': ticker, 'matplotlib.gridspec': gridspec, 'matplotlib.tri': tri, 'matplotlib.collections': collections, 'matplotlib.figure': figureMod, 'matplotlib.axes': axesMod, 'matplotlib.style': style,
      mpl_toolkits: toolkits, 'mpl_toolkits.mplot3d': mplot3d, 'mpl_toolkits.mplot3d.axes3d': axes3dMod,
      seaborn: mod('seaborn', { heatmap: fnv('heatmap', (args, kw) => seabornHeatmap(h, args, kw)) }),
    }
  },
  finish(h) {
    const s = stateOf(h)
    return s.figs.filter((f) => !f.closed && f.axes.length > 0).map((f) => ({ type: 'figure' as const, fig: snapshot(f) }))
  },
}
