/**
 * matplotlib.pyplot for the sandbox: figures are plain data (FigureSpec) that the playground draws as SVG.
 * Covers what the course uses — plot / scatter / bar / hist / imshow / subplots, titles, labels, ticks, legends —
 * with matplotlib's defaults where they matter for intuition: the tab10 colour cycle, viridis for 2-D data,
 * imshow stretching the data range to the colormap unless vmin / vmax are given, y pointing down on images.
 */
import { NDArray } from './ndarray'
import { PyObj, formatSpec, py, toArray } from './minipy'
import type { Host, Kw, PyLib, Value } from './minipy'
import type { Artist, AxesSpec, Display, FigureSpec, Marker } from './pyEvents'

// ---------------------------------------------------------------- colours

/** matplotlib's default property cycle (tab10) */
export const CYCLE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

const NAMED: Record<string, string> = {
  b: '#0000ff', g: '#008000', r: '#ff0000', c: '#00bfbf', m: '#bf00bf', y: '#bfbf00', k: '#000000', w: '#ffffff',
  blue: '#0000ff', green: '#008000', red: '#ff0000', cyan: '#00ffff', magenta: '#ff00ff', yellow: '#ffff00', black: '#000000', white: '#ffffff',
  orange: '#ffa500', purple: '#800080', gray: '#808080', grey: '#808080', pink: '#ffc0cb', brown: '#a52a2a', navy: '#000080',
  lightgray: '#d3d3d3', lightgrey: '#d3d3d3', darkgray: '#a9a9a9', darkgrey: '#a9a9a9', lightblue: '#add8e6', darkblue: '#00008b',
  darkgreen: '#006400', lightgreen: '#90ee90', darkred: '#8b0000', gold: '#ffd700', teal: '#008080', olive: '#808000', crimson: '#dc143c',
  skyblue: '#87ceeb', salmon: '#fa8072', violet: '#ee82ee', indigo: '#4b0082', lime: '#00ff00', tomato: '#ff6347', steelblue: '#4682b4',
  'tab:blue': CYCLE[0], 'tab:orange': CYCLE[1], 'tab:green': CYCLE[2], 'tab:red': CYCLE[3], 'tab:purple': CYCLE[4],
  'tab:brown': CYCLE[5], 'tab:pink': CYCLE[6], 'tab:gray': CYCLE[7], 'tab:olive': CYCLE[8], 'tab:cyan': CYCLE[9],
}

const hex2 = (x: number) => Math.round(Math.max(0, Math.min(1, x)) * 255).toString(16).padStart(2, '0')
const rgbHex = (r: number, g: number, b: number) => `#${hex2(r)}${hex2(g)}${hex2(b)}`

/** colormap control points (evenly spaced), sampled from matplotlib */
const CMAPS: Record<string, string[]> = {
  viridis: ['#440154', '#482475', '#414487', '#355f8d', '#2a788e', '#21918c', '#22a884', '#44bf70', '#7ad151', '#bddf26', '#fde725'],
  plasma: ['#0d0887', '#5302a3', '#8b0aa5', '#b83289', '#db5c68', '#f48849', '#febd2a', '#f0f921'],
  inferno: ['#000004', '#320a5e', '#781c6d', '#bc3754', '#ed6925', '#fbb61a', '#fcffa4'],
  magma: ['#000004', '#2c115f', '#721f81', '#b73779', '#f1605d', '#feb078', '#fcfdbf'],
  gray: ['#000000', '#ffffff'],
  Greys: ['#ffffff', '#000000'],
  binary: ['#ffffff', '#000000'],
  hot: ['#0b0000', '#ff0000', '#ffff00', '#ffffff'],
  jet: ['#00007f', '#0000ff', '#007fff', '#00ffff', '#7fff7f', '#ffff00', '#ff7f00', '#ff0000', '#7f0000'],
  coolwarm: ['#3b4cc0', '#7396f5', '#b0cbfc', '#dddddd', '#f6bfa6', '#ea7b60', '#b40426'],
  RdBu: ['#67001f', '#d6604d', '#fddbc7', '#f7f7f7', '#d1e5f0', '#4393c3', '#053061'],
  bwr: ['#0000ff', '#ffffff', '#ff0000'],
  Blues: ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'],
  Reds: ['#fff5f0', '#fcbba1', '#fb6a4a', '#cb181d', '#67000d'],
  Greens: ['#f7fcf5', '#c7e9c0', '#74c476', '#238b45', '#00441b'],
  Oranges: ['#fff5eb', '#fdd0a2', '#fd8d3c', '#d94801', '#7f2704'],
  rocket: ['#03051a', '#4c1d4b', '#a11a5b', '#e83f3f', '#f69c73', '#faebdd'],
  tab10: CYCLE,
}

export const CMAP_NAMES = Object.keys(CMAPS)

const parseHex = (h: string): [number, number, number] => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255) as [number, number, number]

/** colour of t ∈ [0, 1] under a colormap ('gray_r' reverses) */
export const cmapColor = (name: string, t: number): string => {
  const rev = name.endsWith('_r')
  const stops = CMAPS[rev ? name.slice(0, -2) : name] ?? CMAPS.viridis
  let x = Number.isFinite(t) ? Math.max(0, Math.min(1, t)) : 0
  if (rev) x = 1 - x
  if (stops === CYCLE) return CYCLE[Math.min(9, Math.floor(x * 10))]
  const f = x * (stops.length - 1)
  const i = Math.min(stops.length - 2, Math.floor(f))
  const u = f - i
  const a = parseHex(stops[i])
  const b = parseHex(stops[i + 1])
  return rgbHex(a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u)
}

export const isCmap = (name: string) => (name.endsWith('_r') ? name.slice(0, -2) : name) in CMAPS

// ---------------------------------------------------------------- figure objects

/** a returned artist handle (Line2D, PathCollection …): enough for printing it and calling set_label & co. */
class Handle extends PyObj {
  readonly cls: string
  text: string
  constructor(cls: string, text: string) {
    super()
    this.cls = cls
    this.text = text
  }
  repr() { return this.text }
  getAttr(name: string): Value | undefined {
    if (name.startsWith('set_') || name === 'remove') return { k: 'fn', name, call: () => py.NONE }
    return undefined
  }
}

const emptyAxes = (rows: number, cols: number, index: number): AxesSpec => ({
  pos: { rows, cols, index }, title: null, xlabel: null, ylabel: null, xlim: null, ylim: null, xticks: null, yticks: null,
  axisOff: false, grid: false, legend: null, equal: false, yInverted: false, artists: [], colorbar: null,
})

class Figure extends PyObj {
  readonly cls = 'Figure'
  spec: FigureSpec
  axes: Axes[] = []
  cur: Axes | null = null
  closed = false
  shown = false
  constructor(num: number, size: [number, number]) {
    super()
    this.spec = { num, size, suptitle: null, axes: [] }
  }
  repr() { return `<Figure size ${Math.round(this.spec.size[0] * 100)}x${Math.round(this.spec.size[1] * 100)} with ${this.axes.length} Axes>` }
  str() { return `Figure(${Math.round(this.spec.size[0] * 100)}x${Math.round(this.spec.size[1] * 100)})` }
  display(): Display { return { type: 'figure', fig: snapshot(this) } }

  /** the axes at a grid slot, created on first use */
  slot(rows: number, cols: number, index: number): Axes {
    const found = this.axes.find((a) => a.spec.pos.rows === rows && a.spec.pos.cols === cols && a.spec.pos.index === index)
    if (found) return found
    const ax = new Axes(this, emptyAxes(rows, cols, index))
    this.axes.push(ax)
    this.spec.axes.push(ax.spec)
    return ax
  }

  getAttr(name: string, h: Host): Value | undefined {
    const fn = (call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })
    switch (name) {
      case 'add_subplot': return fn((args) => {
        const [r, c, i] = gridArgs(args, h)
        const ax = this.slot(r, c, i)
        this.cur = ax
        return py.obj(ax)
      })
      case 'suptitle': return fn((args) => {
        this.spec.suptitle = h.str(args[0])
        emitFig(h, this, null)
        return py.obj(new Handle('Text', `Text(0.5, 0.98, '${this.spec.suptitle}')`))
      })
      case 'axes': return py.list(this.axes.map((a) => py.obj(a)))
      case 'tight_layout': case 'subplots_adjust': case 'set_size_inches': case 'show': return fn(() => py.NONE)
      case 'savefig': return fn((args) => {
        h.print(`(sandbox) figure would be saved as ${h.str(args[0])}\n`)
        return py.NONE
      })
      case 'colorbar': return fn((args) => colorbar(h, args[0], this.cur))
      case 'gca': return fn(() => py.obj(this.cur ?? this.slot(1, 1, 1)))
    }
    return undefined
  }
}

class Axes extends PyObj {
  readonly cls = 'Axes'
  fig: Figure
  spec: AxesSpec
  cycle = 0
  /** the last colour-mapped artist (for colorbar) */
  state: { cmap: string; vmin: number; vmax: number } | undefined
  constructor(fig: Figure, spec: AxesSpec) {
    super()
    this.fig = fig
    this.spec = spec
  }
  repr() {
    return this.spec.title ? `<Axes: title={'center': '${this.spec.title}'}>` : '<Axes: >'
  }
  nextColor() { return CYCLE[this.cycle++ % CYCLE.length] }
  getAttr(name: string, h: Host): Value | undefined {
    const m = AXES_METHODS[name]
    if (m) return { k: 'fn', name, call: (args, kw) => m(this, args, kw, h) }
    if (name === 'figure') return py.obj(this.fig)
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
      return py.obj(this.items[at(ints[0], this.shape[0])])
    }
    if (ints.length === 1) return this.iter()[at(ints[0], this.shape[0])]
    return py.obj(this.items[at(ints[0], this.shape[0]) * this.shape[1] + at(ints[1], this.shape[1])])
  }
  getAttr(name: string): Value | undefined {
    if (name === 'shape') return py.tuple(this.shape.map(py.int))
    if (name === 'flat') return py.list(this.items.map((a) => py.obj(a)))
    if (name === 'flatten' || name === 'ravel') return { k: 'fn', name, call: () => py.obj(new AxesGrid(this.items, [this.items.length])) }
    if (name === 'size') return py.int(this.items.length)
    return undefined
  }
}

// ---------------------------------------------------------------- pyplot state

interface PlotState {
  figs: Figure[]
  cur: Figure | null
}

const stateOf = (h: Host): PlotState => {
  let s = h.state.get('pyplot') as PlotState | undefined
  if (!s) {
    s = { figs: [], cur: null }
    h.state.set('pyplot', s)
  }
  return s
}

const newFigure = (h: Host, size: [number, number] = [6.4, 4.8]): Figure => {
  const s = stateOf(h)
  const f = new Figure(s.figs.length + 1, size)
  s.figs.push(f)
  s.cur = f
  return f
}

const gcf = (h: Host): Figure => {
  const s = stateOf(h)
  return s.cur && !s.cur.closed && !s.cur.shown ? s.cur : newFigure(h)
}

const gca = (h: Host): Axes => {
  const f = gcf(h)
  if (!f.cur) f.cur = f.slot(1, 1, 1)
  return f.cur
}

const snapshot = (f: Figure): FigureSpec => JSON.parse(JSON.stringify(f.spec)) as FigureSpec

/** record the figure after a call; focus = the axes and the artist the call added */
const emitFig = (h: Host, f: Figure, ax: Axes | null, artist: number | null = null) => {
  if (!h.tracing) return
  h.emit({ type: 'figure', fig: snapshot(f), focus: ax ? { ax: f.axes.indexOf(ax), artist } : null })
}

const note = (h: Host, tone: 'info' | 'warn', key: string, params: Record<string, string | number> = {}) => {
  if (h.tracing) h.emit({ type: 'note', tone, key, params })
}

// ---------------------------------------------------------------- argument helpers

const nums = (h: Host, v: Value, what: string): number[] => {
  try {
    return toArray(v, 'float64').values()
  } catch {
    throw h.err('TypeError', `${what}: expected numbers, got '${v.k === 'obj' ? v.o.cls : v.k}'`)
  }
}

const isStrList = (v: Value) => (v.k === 'list' || v.k === 'tuple') && v.items.length > 0 && v.items.every((x) => x.k === 'str')

const optStr = (h: Host, v: Value | undefined): string | null => (v && v.k !== 'none' ? h.str(v) : null)
const optNum = (h: Host, v: Value | undefined): number | null => (v && v.k !== 'none' ? h.num(v) : null)

/** a matplotlib colour spec → CSS colour */
export const toColor = (h: Host, v: Value): string => {
  if (v.k === 'str') {
    const s = v.v.trim()
    if (s.startsWith('#')) return s.length === 4 ? `#${s[1]}${s[1]}${s[2]}${s[2]}${s[3]}${s[3]}` : s.slice(0, 7)
    if (/^C\d$/.test(s)) return CYCLE[Number(s[1])]
    if (/^(0(\.\d+)?|1(\.0+)?)$/.test(s)) {
      const g = Number(s)
      return rgbHex(g, g, g)
    }
    const c = NAMED[s.toLowerCase()]
    if (!c) throw h.err('ValueError', `'${s}' is not a valid color value.`)
    return c
  }
  if (v.k === 'tuple' || v.k === 'list') {
    const xs = v.items.map((x) => h.num(x))
    if (xs.length < 3 || xs.some((x) => x < 0 || x > 1)) throw h.err('ValueError', 'RGB values must be in the 0-1 range')
    return rgbHex(xs[0], xs[1], xs[2])
  }
  throw h.err('ValueError', `invalid color: ${v.k}`)
}

/** 'ro--' → colour, marker, line style */
const parseFmt = (h: Host, fmt: string): { color: string | null; marker: Marker; dash: string | null; line: boolean } => {
  let s = fmt
  let color: string | null = null
  let marker: Marker = null
  let dash: string | null = null
  let line = true
  const m = /^C\d/.exec(s)
  if (m) {
    color = CYCLE[Number(m[0][1])]
    s = s.slice(2)
  }
  const styles: [string, string | null][] = [['--', '6 4'], ['-.', '6 3 1.5 3'], [':', '1.5 3'], ['-', null]]
  let sawStyle = false
  while (s.length) {
    const st = styles.find(([k]) => s.startsWith(k))
    if (st) {
      dash = st[1]
      sawStyle = true
      s = s.slice(st[0].length)
      continue
    }
    const c = s[0]
    if ('bgrcmykw'.includes(c) && !color) color = NAMED[c]
    else if ('o.s^v*x+Dd'.includes(c)) marker = (c === 'd' ? 'D' : c) as Marker
    else throw h.err('ValueError', `'${fmt}' is not a valid format string (unrecognized character '${c}')`)
    s = s.slice(1)
  }
  if (marker && !sawStyle) line = false
  return { color, marker, dash, line }
}

const LS: Record<string, string | null> = { '-': null, solid: null, '--': '6 4', dashed: '6 4', ':': '1.5 3', dotted: '1.5 3', '-.': '6 3 1.5 3', dashdot: '6 3 1.5 3' }

const gridArgs = (args: Value[], h: Host): [number, number, number] => {
  if (args.length === 1) {
    const n = h.toInt(args[0])
    if (n < 111 || n > 999) throw h.err('ValueError', `Single argument to subplot must be a three-digit integer, not ${n}`)
    return [Math.floor(n / 100), Math.floor(n / 10) % 10, n % 10]
  }
  const [r, c, i] = args.map((a) => h.toInt(a))
  if (r < 1 || c < 1) throw h.err('ValueError', 'Number of rows and columns must be positive integers')
  if (i < 1 || i > r * c) throw h.err('ValueError', `num must be an integer with 1 <= num <= ${r * c}, not ${i}`)
  return [r, c, i]
}

// ---------------------------------------------------------------- plotting calls

type AxMethod = (ax: Axes, args: Value[], kw: Kw, h: Host) => Value

const addArtist = (h: Host, ax: Axes, a: Artist): number => {
  ax.spec.artists.push(a)
  const i = ax.spec.artists.length - 1
  emitFig(h, ax.fig, ax, i)
  return i
}

const labelOf = (h: Host, kw: Kw) => optStr(h, kw.label)

const plot: AxMethod = (ax, args, kw, h) => {
  // plot(y), plot(x, y), plot(x, y, fmt), plot(y, fmt)
  const rest = [...args]
  let fmt: string | null = null
  if (rest.length && rest[rest.length - 1].k === 'str') fmt = (rest.pop() as { v: string }).v
  if (!rest.length || rest.length > 2) throw h.err('TypeError', 'plot() takes plot(y), plot(x, y) or plot(x, y, fmt) in this sandbox')
  const yv = rest[rest.length - 1]
  const yArr = toArray(yv, 'float64')
  const cols = yArr.ndim === 2 ? yArr.shape[1] : 1
  const rows = yArr.ndim === 2 ? yArr.shape[0] : yArr.size
  const ys = yArr.values()
  const x = rest.length === 2 ? nums(h, rest[0], 'plot x') : Array.from({ length: rows }, (_, i) => i)
  if (x.length !== rows) throw h.err('ValueError', `x and y must have same first dimension, but have shapes (${x.length},) and ${yArr.ndim === 2 ? `(${rows}, ${cols})` : `(${rows},)`}`)
  const f = fmt ? parseFmt(h, fmt) : { color: null, marker: null as Marker, dash: null, line: true }
  const handles: Value[] = []
  for (let c = 0; c < cols; c++) {
    const y = yArr.ndim === 2 ? Array.from({ length: rows }, (_, r) => ys[r * cols + c]) : ys
    const color = kw.color ?? kw.c ? toColor(h, (kw.color ?? kw.c)!) : f.color ?? ax.nextColor()
    const ls = kw.linestyle ?? kw.ls
    const dash = ls ? (ls.k === 'str' && ls.v in LS ? LS[ls.v] : f.dash) : f.dash
    const marker = (optStr(h, kw.marker) as Marker) ?? f.marker
    const noLine = !f.line && !ls
    addArtist(h, ax, {
      kind: 'line', x, y, color, width: noLine ? 0 : optNum(h, kw.linewidth ?? kw.lw) ?? 1.5, dash, marker,
      label: labelOf(h, kw), alpha: optNum(h, kw.alpha) ?? 1,
    })
    handles.push(py.obj(new Handle('Line2D', `<matplotlib.lines.Line2D object>`)))
  }
  return py.list(handles)
}

const scatter: AxMethod = (ax, args, kw, h) => {
  if (args.length < 2) throw h.err('TypeError', "scatter() missing 1 required positional argument: 'y'")
  const x = nums(h, args[0], 'scatter x')
  const y = nums(h, args[1], 'scatter y')
  if (x.length !== y.length) throw h.err('ValueError', 'x and y must be the same size')
  const cv = kw.c ?? kw.color ?? args[3]
  let colors: string[]
  const cmap = optStr(h, kw.cmap) ?? 'viridis'
  if (!cv || cv.k === 'none') colors = [ax.nextColor()]
  else if (cv.k === 'str' || (cv.k === 'tuple' && cv.items.length >= 3 && cv.items.every((i) => i.k === 'float' || i.k === 'int') && x.length !== cv.items.length)) colors = [toColor(h, cv)]
  else if (isStrList(cv)) colors = (cv as { items: Value[] }).items.map((c) => toColor(h, c))
  else {
    const vals = nums(h, cv, 'scatter c')
    if (vals.length !== x.length) throw h.err('ValueError', `'c' argument has ${vals.length} elements, which is inconsistent with 'x' and 'y' with size ${x.length}.`)
    if (!isCmap(cmap)) throw h.err('ValueError', `'${cmap}' is not a valid value for cmap`)
    const lo = optNum(h, kw.vmin) ?? Math.min(...vals)
    const hi = optNum(h, kw.vmax) ?? Math.max(...vals)
    colors = vals.map((v) => cmapColor(cmap, hi === lo ? 0 : (v - lo) / (hi - lo)))
    ax.state = { cmap, vmin: lo, vmax: hi }
  }
  const sv = kw.s ?? args[2]
  const sizes = sv && sv.k !== 'none' ? nums(h, sv, 'scatter s') : [36]
  addArtist(h, ax, {
    kind: 'scatter', x, y, colors, sizes, marker: (optStr(h, kw.marker) as Marker) ?? 'o', label: labelOf(h, kw),
    alpha: optNum(h, kw.alpha) ?? 1, edge: kw.edgecolors ?? kw.edgecolor ? toColor(h, (kw.edgecolors ?? kw.edgecolor)!) : null,
  })
  return py.obj(new Handle('PathCollection', '<matplotlib.collections.PathCollection object>'))
}

const bar = (horizontal: boolean): AxMethod => (ax, args, kw, h) => {
  const xv = args[0] ?? kw.x
  const hv = args[1] ?? kw.height ?? kw.width
  if (!xv || !hv) throw h.err('TypeError', `${horizontal ? 'barh' : 'bar'}() needs positions and ${horizontal ? 'widths' : 'heights'}`)
  let tickLabels: string[] | null = null
  let x: number[]
  if (isStrList(xv)) {
    tickLabels = (xv as { items: Value[] }).items.map((s) => h.str(s))
    x = tickLabels.map((_, i) => i)
  } else x = nums(h, xv, 'bar x')
  let heights = nums(h, hv, 'bar height')
  if (heights.length === 1 && x.length > 1) heights = x.map(() => heights[0])
  if (heights.length !== x.length) throw h.err('ValueError', `shape mismatch: objects cannot be broadcast to a single shape. Mismatch is between arg 0 with shape (${x.length},) and arg 1 with shape (${heights.length},).`)
  const bv = kw.bottom ?? kw.left
  const bottoms = bv ? (isStrList(bv) ? [] : nums(h, bv, 'bottom')) : []
  const cv = kw.color
  const colors = !cv ? [ax.nextColor()] : isStrList(cv) ? (cv as { items: Value[] }).items.map((c) => toColor(h, c)) : [toColor(h, cv)]
  addArtist(h, ax, {
    kind: 'bar', x, heights, bottoms: bottoms.length === 1 ? x.map(() => bottoms[0]) : bottoms.length ? bottoms : x.map(() => 0),
    width: optNum(h, horizontal ? kw.height : kw.width) ?? 0.8, colors, label: labelOf(h, kw), horizontal, alpha: optNum(h, kw.alpha) ?? 1, tickLabels,
  })
  return py.obj(new Handle('BarContainer', `<BarContainer object of ${x.length} artists>`))
}

/** numpy.histogram: equal-width bins over [min, max] (or range=), the last bin closed */
export const histogram = (data: number[], bins: number | number[], range?: [number, number]): { counts: number[]; edges: number[] } => {
  let edges: number[]
  if (Array.isArray(bins)) edges = bins
  else {
    let [lo, hi] = range ?? [Math.min(...data), Math.max(...data)]
    if (!data.length && !range) [lo, hi] = [0, 1]
    if (lo === hi) [lo, hi] = [lo - 0.5, hi + 0.5]
    edges = Array.from({ length: bins + 1 }, (_, i) => lo + ((hi - lo) * i) / bins)
  }
  const counts = new Array<number>(edges.length - 1).fill(0)
  const last = edges.length - 1
  for (const v of data) {
    if (!(v >= edges[0] && v <= edges[last])) continue
    let i = edges.findIndex((e, k) => k < last && v >= e && v < edges[k + 1])
    if (i < 0) i = last - 1
    counts[i]++
  }
  return { counts, edges }
}

const hist: AxMethod = (ax, args, kw, h) => {
  const data = nums(h, args[0] ?? kw.x, 'hist x').filter((v) => !Number.isNaN(v))
  const bv = kw.bins ?? args[1]
  const bins = !bv || bv.k === 'none' ? 10 : bv.k === 'int' ? bv.v : nums(h, bv, 'bins')
  if (typeof bins === 'number' && (bins < 1 || bins > 512)) throw h.err('ValueError', '`bins` must be between 1 and 512 in this sandbox')
  const rv = kw.range
  const range = rv && rv.k !== 'none' ? (nums(h, rv, 'range') as [number, number]) : undefined
  const { counts, edges } = histogram(data, bins, range)
  const density = kw.density && h.truthy(kw.density)
  const width = edges.length > 1 ? edges[1] - edges[0] : 1
  const shown = density ? counts.map((c) => c / (data.length * width)) : counts
  addArtist(h, ax, { kind: 'hist', edges, counts: shown, color: kw.color ? toColor(h, kw.color) : ax.nextColor(), label: labelOf(h, kw), alpha: optNum(h, kw.alpha) ?? 1 })
  return py.tuple([
    py.arr(NDArray.create(shown, [shown.length], 'float64')),
    py.arr(NDArray.create(edges, [edges.length], 'float64')),
    py.obj(new Handle('BarContainer', `<BarContainer object of ${counts.length} artists>`)),
  ])
}

const imshow: AxMethod = (ax, args, kw, h) => {
  const a = toArray(args[0] ?? kw.X)
  const vals = a.values()
  if (a.ndim !== 2 && !(a.ndim === 3 && (a.shape[2] === 3 || a.shape[2] === 4))) {
    throw h.err('TypeError', `Invalid shape ${a.ndim === 1 ? `(${a.shape[0]},)` : `(${a.shape.join(', ')})`} for image data`)
  }
  const rows = a.shape[0]
  const cols = a.shape[1]
  const rgb = a.ndim === 3
  const cmapV = kw.cmap ?? args[1]
  let cmap = cmapV && cmapV.k !== 'none' ? (cmapV.k === 'obj' ? cmapV.o.repr() : h.str(cmapV)) : null
  if (cmap && !isCmap(cmap)) throw h.err('ValueError', `'${cmap}' is not a valid value for cmap; supported values include ${CMAP_NAMES.slice(0, 8).map((c) => `'${c}'`).join(', ')} …`)
  const vminV = optNum(h, kw.vmin)
  const vmaxV = optNum(h, kw.vmax)
  const finite = vals.filter(Number.isFinite)
  const vmin = vminV ?? (finite.length ? Math.min(...finite) : 0)
  const vmax = vmaxV ?? (finite.length ? Math.max(...finite) : 1)
  let pixels: string[]
  if (rgb) {
    const isFloat = a.dtype === 'float64'
    const top = isFloat ? 1 : 255
    let clipped = false
    const ch = a.shape[2]
    pixels = Array.from({ length: rows * cols }, (_, p) => {
      const c = [0, 1, 2].map((k) => {
        const v = vals[p * ch + k]
        if (v < 0 || v > top) clipped = true
        return Math.max(0, Math.min(top, v)) / top
      })
      return rgbHex(c[0], c[1], c[2])
    })
    if (clipped) h.print('WARNING: Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).\n')
    cmap = null
  } else {
    if (!cmap) note(h, 'info', 'default_cmap')
    if (vminV === null && vmaxV === null) note(h, 'info', 'auto_range', { min: +vmin.toPrecision(4), max: +vmax.toPrecision(4) })
    const name = cmap ?? 'viridis'
    pixels = vals.map((v) => cmapColor(name, vmax === vmin ? 0 : (v - vmin) / (vmax - vmin)))
  }
  ax.spec.yInverted = true
  ax.spec.equal = true
  addArtist(h, ax, {
    kind: 'image', rows, cols, pixels, values: rgb ? [] : vals, cmap: rgb ? null : cmap ?? 'viridis', vmin, vmax,
    auto: vminV === null && vmaxV === null, rgb, extent: [-0.5, cols - 0.5, rows - 0.5, -0.5],
  })
  if (!rgb) ax.state = { cmap: cmap ?? 'viridis', vmin, vmax }
  return py.obj(new Handle('AxesImage', 'AxesImage(size=(' + cols + ', ' + rows + '))'))
}

const refLine = (kind: 'hline' | 'vline'): AxMethod => (ax, args, kw, h) => {
  const at = h.num(args[0] ?? (kind === 'hline' ? kw.y : kw.x) ?? py.int(0))
  const ls = kw.linestyle ?? kw.ls
  addArtist(h, ax, {
    kind, at, color: kw.color ?? kw.c ? toColor(h, (kw.color ?? kw.c)!) : ax.nextColor(),
    dash: ls && ls.k === 'str' ? LS[ls.v] ?? null : null, width: optNum(h, kw.linewidth ?? kw.lw) ?? 1.5, label: labelOf(h, kw),
  })
  return py.obj(new Handle('Line2D', '<matplotlib.lines.Line2D object>'))
}

const setText = (field: 'title' | 'xlabel' | 'ylabel'): AxMethod => (ax, args, _kw, h) => {
  ax.spec[field] = h.str(args[0] ?? py.str(''))
  emitFig(h, ax.fig, ax)
  return py.obj(new Handle('Text', `Text(0.5, 1.0, '${ax.spec[field]}')`))
}

const setLim = (field: 'xlim' | 'ylim'): AxMethod => (ax, args, kw, h) => {
  if (!args.length && !Object.keys(kw).length) {
    const cur = ax.spec[field] ?? [0, 1]
    return py.tuple(cur.map(py.float))
  }
  const pair = args.length === 1 ? nums(h, args[0], field) : args.map((v) => h.num(v))
  const lo = kw.left ?? kw.bottom
  const hi = kw.right ?? kw.top
  const next: [number, number] = [lo ? h.num(lo) : pair[0], hi ? h.num(hi) : pair[1]]
  ax.spec[field] = next
  emitFig(h, ax.fig, ax)
  return py.tuple(next.map(py.float))
}

const setTicks = (field: 'xticks' | 'yticks'): AxMethod => (ax, args, kw, h) => {
  const at = args[0] ? nums(h, args[0], field) : []
  const lv = args[1] ?? kw.labels
  ax.spec[field] = { at, labels: lv && lv.k !== 'none' ? h.iterate(lv).map((v) => h.str(v)) : null }
  emitFig(h, ax.fig, ax)
  return py.NONE
}

const colorbar = (h: Host, _mappable: Value | undefined, ax: Axes | null): Value => {
  const target = ax ?? gca(h)
  const st = target.state
  if (!st) throw h.err('RuntimeError', 'No mappable was found to use for colorbar creation. First define a mappable such as an image (with imshow) or a contour set (with contourf).')
  target.spec.colorbar = st
  emitFig(h, target.fig, target)
  return py.obj(new Handle('Colorbar', '<matplotlib.colorbar.Colorbar object>'))
}

const AXES_METHODS: Record<string, AxMethod> = {
  plot, scatter, bar: bar(false), barh: bar(true), hist, imshow,
  axhline: refLine('hline'), axvline: refLine('vline'),
  set_title: setText('title'), set_xlabel: setText('xlabel'), set_ylabel: setText('ylabel'),
  set_xlim: setLim('xlim'), set_ylim: setLim('ylim'), set_xticks: setTicks('xticks'), set_yticks: setTicks('yticks'),
  set_xticklabels: (ax, args, _kw, h) => {
    ax.spec.xticks = { at: ax.spec.xticks?.at ?? h.iterate(args[0]).map((_, i) => i), labels: h.iterate(args[0]).map((v) => h.str(v)) }
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  text: (ax, args, kw, h) => {
    if (args.length < 3) throw h.err('TypeError', "text() missing required arguments: 'x', 'y' and 's'")
    const ha = optStr(h, kw.ha ?? kw.horizontalalignment)
    const va = optStr(h, kw.va ?? kw.verticalalignment)
    addArtist(h, ax, {
      kind: 'text', x: h.num(args[0]), y: h.num(args[1]), text: h.str(args[2]), color: kw.color ? toColor(h, kw.color) : '#000000',
      size: optNum(h, kw.fontsize ?? kw.size) ?? 10, ha: (ha === 'center' || ha === 'right' ? ha : 'left'), va: (va === 'center' || va === 'top' ? va : 'bottom'),
    })
    return py.obj(new Handle('Text', `Text(${h.str(args[0])}, ${h.str(args[1])}, '${h.str(args[2])}')`))
  },
  fill_between: (ax, args, kw, h) => {
    const x = nums(h, args[0], 'fill_between x')
    const y1 = nums(h, args[1], 'fill_between y1')
    const y2v = args[2] ?? kw.y2
    const y2 = y2v ? nums(h, y2v, 'fill_between y2') : [0]
    const full = (ys: number[]) => (ys.length === 1 ? x.map(() => ys[0]) : ys)
    addArtist(h, ax, { kind: 'fill', x, y1: full(y1), y2: full(y2), color: kw.color ? toColor(h, kw.color) : ax.nextColor(), alpha: optNum(h, kw.alpha) ?? 0.3, label: labelOf(h, kw) })
    return py.obj(new Handle('PolyCollection', '<matplotlib.collections.PolyCollection object>'))
  },
  legend: (ax, _args, kw, h) => {
    if (!ax.spec.artists.some((a) => 'label' in a && a.label && !a.label.startsWith('_'))) {
      h.print('No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.\n')
    }
    ax.spec.legend = { loc: optStr(h, kw.loc) ?? 'best' }
    emitFig(h, ax.fig, ax)
    return py.obj(new Handle('Legend', '<matplotlib.legend.Legend object>'))
  },
  grid: (ax, args, kw, h) => {
    ax.spec.grid = args[0] ? h.truthy(args[0]) : kw.visible ? h.truthy(kw.visible) : !ax.spec.grid || args.length === 0
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  axis: (ax, args, _kw, h) => {
    const a = args[0]
    if (a?.k === 'str') {
      if (a.v === 'off') ax.spec.axisOff = true
      else if (a.v === 'on') ax.spec.axisOff = false
      else if (a.v === 'equal' || a.v === 'scaled' || a.v === 'square') ax.spec.equal = true
    } else if (a) {
      const [x0, x1, y0, y1] = nums(h, a, 'axis')
      ax.spec.xlim = [x0, x1]
      ax.spec.ylim = [y0, y1]
    }
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  set_aspect: (ax, args, _kw, h) => {
    ax.spec.equal = args[0]?.k === 'str' ? args[0].v === 'equal' : h.num(args[0]) === 1
    return py.NONE
  },
  invert_yaxis: (ax, _a, _k, h) => {
    ax.spec.yInverted = !ax.spec.yInverted
    emitFig(h, ax.fig, ax)
    return py.NONE
  },
  set: (ax, _args, kw, h) => {
    for (const [k, v] of Object.entries(kw)) {
      const m = AXES_METHODS[`set_${k}`]
      if (m) m(ax, [v], {}, h)
    }
    return py.NONE
  },
}

// ---------------------------------------------------------------- the module

const pyplot = (h: Host): Value => {
  const attrs: Record<string, Value> = {}
  const fn = (name: string, call: (args: Value[], kw: Kw) => Value): Value => ({ k: 'fn', name, call })
  // every Axes method has a pyplot twin that acts on the current Axes
  const twins: Record<string, string> = {
    plot: 'plot', scatter: 'scatter', bar: 'bar', barh: 'barh', hist: 'hist', imshow: 'imshow', axhline: 'axhline', axvline: 'axvline',
    title: 'set_title', xlabel: 'set_xlabel', ylabel: 'set_ylabel', xlim: 'xlim', ylim: 'ylim', xticks: 'xticks', yticks: 'yticks',
    text: 'text', fill_between: 'fill_between', legend: 'legend', grid: 'grid', axis: 'axis',
  }
  for (const [name, method] of Object.entries(twins)) {
    attrs[name] = fn(name, (args, kw) => {
      const ax = gca(h)
      const m = method === 'xlim' ? setLim('xlim') : method === 'ylim' ? setLim('ylim') : method === 'xticks' ? setTicks('xticks') : method === 'yticks' ? setTicks('yticks') : AXES_METHODS[method]
      return m(ax, args, kw, h)
    })
  }
  attrs.figure = fn('figure', (_args, kw) => {
    const fs = kw.figsize
    const size = fs ? (h.iterate(fs).map((v) => h.num(v)) as [number, number]) : undefined
    return py.obj(newFigure(h, size))
  })
  attrs.subplot = fn('subplot', (args) => {
    const [r, c, i] = gridArgs(args, h)
    const f = gcf(h)
    f.cur = f.slot(r, c, i)
    return py.obj(f.cur)
  })
  attrs.subplots = fn('subplots', (args, kw) => {
    const nrows = kw.nrows ? h.toInt(kw.nrows) : args[0] ? h.toInt(args[0]) : 1
    const ncols = kw.ncols ? h.toInt(kw.ncols) : args[1] ? h.toInt(args[1]) : 1
    if (nrows < 1 || ncols < 1 || nrows * ncols > 64) throw h.err('ValueError', 'subplots: between 1 and 64 axes in this sandbox')
    const fs = kw.figsize
    const f = newFigure(h, fs ? (h.iterate(fs).map((v) => h.num(v)) as [number, number]) : undefined)
    const axes = Array.from({ length: nrows * ncols }, (_, k) => f.slot(nrows, ncols, k + 1))
    f.cur = axes[0]
    emitFig(h, f, null)
    const squeeze = !kw.squeeze || h.truthy(kw.squeeze)
    const axv = squeeze && axes.length === 1 ? py.obj(axes[0]) : py.obj(new AxesGrid(axes, squeeze && (nrows === 1 || ncols === 1) ? [axes.length] : [nrows, ncols]))
    return py.tuple([py.obj(f), axv])
  })
  attrs.gcf = fn('gcf', () => py.obj(gcf(h)))
  attrs.gca = fn('gca', () => py.obj(gca(h)))
  attrs.suptitle = fn('suptitle', (args) => {
    const f = gcf(h)
    f.spec.suptitle = h.str(args[0])
    emitFig(h, f, null)
    return py.NONE
  })
  attrs.colorbar = fn('colorbar', (args) => colorbar(h, args[0], gcf(h).cur))
  attrs.show = fn('show', () => {
    const s = stateOf(h)
    for (const f of s.figs) if (!f.closed) f.shown = true
    return py.NONE
  })
  attrs.close = fn('close', (args) => {
    const s = stateOf(h)
    const which = args[0]
    if (which?.k === 'str' && which.v === 'all') s.figs.forEach((f) => (f.closed = true))
    else if (s.cur) s.cur.closed = true
    return py.NONE
  })
  attrs.tight_layout = fn('tight_layout', () => py.NONE)
  attrs.subplots_adjust = fn('subplots_adjust', () => py.NONE)
  attrs.savefig = fn('savefig', (args) => {
    h.print(`(sandbox) figure would be saved as ${h.str(args[0])}\n`)
    return py.NONE
  })
  attrs.imsave = fn('imsave', (args) => {
    h.print(`(sandbox) image would be saved as ${h.str(args[0])}\n`)
    return py.NONE
  })
  attrs.imread = fn('imread', () => {
    throw h.err('FileNotFoundError', 'there are no files in this sandbox — build the image with NumPy instead')
  })
  // plt.cm.gray, plt.cm.viridis …
  const cmapObj = (name: string): Value => py.obj(new Handle('Colormap', name))
  attrs.cm = { k: 'module', name: 'matplotlib.cm', attrs: Object.fromEntries(CMAP_NAMES.flatMap((n) => [[n, cmapObj(n)], [`${n}_r`, cmapObj(`${n}_r`)]])) }
  return { k: 'module', name: 'matplotlib.pyplot', attrs }
}

/** seaborn.heatmap as lecture 6 uses it for the confusion matrix: an image of the matrix with each value written in */
const heatmap = (h: Host, args: Value[], kw: Kw): Value => {
  const a = toArray(args[0] ?? kw.data, 'float64')
  if (a.ndim !== 2) throw h.err('ValueError', 'heatmap needs a 2-D array (rows × columns)')
  const ax = kw.ax && kw.ax.k === 'obj' && kw.ax.o instanceof Axes ? kw.ax.o : gca(h)
  const [rows, cols] = a.shape
  const vals = a.values()
  const cmap = optStr(h, kw.cmap) ?? 'rocket'
  if (!isCmap(cmap)) throw h.err('ValueError', `'${cmap}' is not a valid value for cmap`)
  const vmin = optNum(h, kw.vmin) ?? Math.min(...vals)
  const vmax = optNum(h, kw.vmax) ?? Math.max(...vals)
  const tOf = (v: number) => (vmax === vmin ? 0 : (v - vmin) / (vmax - vmin))
  const pixels = vals.map((v) => cmapColor(cmap, tOf(v)))
  ax.spec.yInverted = true
  ax.spec.equal = !!kw.square && h.truthy(kw.square)
  ax.spec.artists.push({ kind: 'image', rows, cols, pixels, values: vals, cmap, vmin, vmax, auto: false, rgb: false, extent: [-0.5, cols - 0.5, rows - 0.5, -0.5] })
  if (kw.annot && h.truthy(kw.annot)) {
    const fmt = optStr(h, kw.fmt) ?? '.2g'
    vals.forEach((v, i) => {
      const [r, g, b] = parseHex(pixels[i])
      const intLike = a.dtype !== 'float64' || Number.isInteger(v)
      ax.spec.artists.push({
        kind: 'text', x: i % cols, y: Math.floor(i / cols), text: formatSpec(intLike && fmt === 'd' ? py.int(v) : py.float(v), fmt),
        color: r * 0.3 + g * 0.59 + b * 0.11 > 0.55 ? '#000000' : '#ffffff', size: 10, ha: 'center', va: 'center',
      })
    })
  }
  ax.spec.xticks = { at: Array.from({ length: cols }, (_, i) => i), labels: null }
  ax.spec.yticks = { at: Array.from({ length: rows }, (_, i) => i), labels: null }
  ax.state = { cmap, vmin, vmax }
  if (!kw.cbar || h.truthy(kw.cbar)) ax.spec.colorbar = ax.state
  emitFig(h, ax.fig, ax, ax.spec.artists.length - 1)
  return py.obj(ax)
}

export const MATPLOTLIB: PyLib = {
  modules: ['matplotlib', 'matplotlib.pyplot', 'matplotlib.cm', 'seaborn'],
  load(h) {
    const plt = pyplot(h)
    const cm = (plt as { attrs: Record<string, Value> }).attrs.cm
    return {
      matplotlib: { k: 'module', name: 'matplotlib', attrs: { pyplot: plt, cm, __version__: py.str('3.10 (sandbox)') } },
      'matplotlib.pyplot': plt,
      'matplotlib.cm': cm,
      seaborn: { k: 'module', name: 'seaborn', attrs: { heatmap: { k: 'fn', name: 'heatmap', call: (args, kw) => heatmap(h, args, kw) } } },
    }
  },
  finish(h) {
    const s = stateOf(h)
    return s.figs.filter((f) => !f.closed && f.axes.length > 0).map((f) => ({ type: 'figure' as const, fig: snapshot(f) }))
  },
}
