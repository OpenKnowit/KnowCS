import { memo, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { PT, textHeightPx, textWidthPx, tickLabelDepth } from '../../lib/pyPlot'
import type { Artist, ArtistBase, AxesSpec, AxisSpec, Coord, Dash, FigureSpec, ImageArtist, LegendEntry, LegendSpec, MeshArtist, TextArrow, TextArtist } from '../../lib/pyEvents'

// --- 把沙盒里的 matplotlib 图（FigureSpec）画成 SVG。位置、刻度都已在 pyPlot 里按 matplotlib 算好，这里只负责落笔：1 pt = 100/72 px ---

const PX = 100 // pixels per inch, matplotlib's default dpi
const FONT = "'DejaVu Sans', Verdana, Geneva, sans-serif"

interface Box { x: number; y: number; w: number; h: number }

export interface PixelInfo { row: number; col: number; value: number | null; color: string; vmin: number; vmax: number; cmap: string | null; auto: boolean }

const fmt = (v: number) => {
  const a = Math.abs(v)
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1)
  return String(+v.toPrecision(6))
}

const n2 = (v: number) => (Number.isFinite(v) ? +v.toFixed(2) : 0)

const dashArray = (dash: Dash, width: number): string | undefined => (dash ? dash.map((d) => n2(d * Math.max(width, 0.1))).join(' ') : undefined)

/** '#rrggbb' or '#rrggbbaa' → canvas bytes */
const putPixel = (data: Uint8ClampedArray, k: number, c: string) => {
  if (!c.startsWith('#')) return
  data[k] = parseInt(c.slice(1, 3), 16)
  data[k + 1] = parseInt(c.slice(3, 5), 16)
  data[k + 2] = parseInt(c.slice(5, 7), 16)
  data[k + 3] = c.length > 7 ? parseInt(c.slice(7, 9), 16) : 255
}

const canvasUrl = (cols: number, rows: number, colors: string[], flipX = false, flipY = false): string | null => {
  if (typeof document === 'undefined') return null
  const c = document.createElement('canvas')
  c.width = cols
  c.height = rows
  const ctx = c.getContext('2d')
  if (!ctx) return null
  const img = ctx.createImageData(cols, rows)
  for (let i = 0; i < colors.length; i++) {
    const r = Math.floor(i / cols)
    const cc = i % cols
    putPixel(img.data, ((flipY ? rows - 1 - r : r) * cols + (flipX ? cols - 1 - cc : cc)) * 4, colors[i])
  }
  ctx.putImageData(img, 0, 0)
  return c.toDataURL()
}

// ---------------------------------------------------------------- mathtext and text

const GREEK: Record<string, string> = {
  alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ε', varepsilon: 'ε', zeta: 'ζ', eta: 'η', theta: 'θ', vartheta: 'ϑ', iota: 'ι', kappa: 'κ', lambda: 'λ', mu: 'μ', nu: 'ν', xi: 'ξ', pi: 'π', rho: 'ρ', sigma: 'σ', tau: 'τ', upsilon: 'υ', phi: 'φ', varphi: 'φ', chi: 'χ', psi: 'ψ', omega: 'ω',
  Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ', Xi: 'Ξ', Pi: 'Π', Sigma: 'Σ', Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω',
  times: '×', cdot: '·', pm: '±', mp: '∓', leq: '≤', le: '≤', geq: '≥', ge: '≥', neq: '≠', ne: '≠', approx: '≈', sim: '∼', equiv: '≡', infty: '∞', partial: '∂', nabla: '∇', sum: '∑', prod: '∏', int: '∫', to: '→', rightarrow: '→', leftarrow: '←', Rightarrow: '⇒', in: '∈', notin: '∉', subset: '⊂', cup: '∪', cap: '∩', forall: '∀', exists: '∃', ldots: '…', cdots: '⋯', circ: '∘', degree: '°', prime: '′', hbar: 'ħ', ell: 'ℓ', sqrt: '√', div: '÷', ast: '∗', star: '⋆', langle: '⟨', rangle: '⟩', quad: '  ', qquad: '    ',
  sin: 'sin', cos: 'cos', tan: 'tan', log: 'log', ln: 'ln', exp: 'exp', min: 'min', max: 'max', lim: 'lim', arg: 'arg', argmax: 'argmax', argmin: 'argmin',
}
const UPRIGHT = new Set(['sin', 'cos', 'tan', 'log', 'ln', 'exp', 'min', 'max', 'lim', 'arg', 'argmax', 'argmin'])

interface Span { t: string; shift: -1 | 0 | 1; italic: boolean }

/** a readable approximation of mathtext: symbols, sub- and superscripts, \frac and \sqrt written out */
const mathSpans = (src: string): Span[] => {
  const out: Span[] = []
  const push = (t: string, shift: Span['shift'], italic: boolean) => {
    const last = out[out.length - 1]
    if (last && last.shift === shift && last.italic === italic) last.t += t
    else if (t) out.push({ t, shift, italic })
  }
  const braced = (s: string, i: number): [string, number] => {
    if (s[i] !== '{') return s[i] === '\\' ? (() => { const m = /^\\[a-zA-Z]+/.exec(s.slice(i)); return m ? [m[0], i + m[0].length] as [string, number] : [s.slice(i, i + 2), i + 2] as [string, number] })() : [s[i] ?? '', i + 1]
    let depth = 0
    for (let j = i; j < s.length; j++) {
      if (s[j] === '{') depth++
      else if (s[j] === '}' && --depth === 0) return [s.slice(i + 1, j), j + 1]
    }
    return [s.slice(i + 1), s.length]
  }
  const walk = (s: string, shift: Span['shift'], italic: boolean) => {
    let i = 0
    while (i < s.length) {
      const c = s[i]
      if (c === '\\') {
        const m = /^\\([a-zA-Z]+|.)/.exec(s.slice(i))
        const name = m ? m[1] : ''
        i += m ? m[0].length : 1
        if (['mathdefault', 'mathrm', 'mathbf', 'mathit', 'text', 'textbf', 'textit', 'operatorname', 'mathsf', 'mathtt', 'mathcal', 'mathbb'].includes(name)) {
          const [b, j] = braced(s, i)
          walk(b, shift, name === 'mathit' || name === 'mathcal')
          i = j
        } else if (name === 'frac' || name === 'dfrac') {
          const [a, j] = braced(s, i)
          const [b, k] = braced(s, j)
          const wrap = (x: string) => (/^[\w.\\]+$/.test(x) ? x : `(${x})`)
          walk(`${wrap(a)}/${wrap(b)}`, shift, italic)
          i = k
        } else if (name === 'sqrt') {
          const [a, j] = braced(s, i)
          push('√', shift, false)
          walk(/^\w$/.test(a) ? a : `(${a})`, shift, italic)
          i = j
        } else if (name === 'hat' || name === 'bar' || name === 'vec' || name === 'tilde' || name === 'overline' || name === 'dot') {
          const [a, j] = braced(s, i)
          const mark = { hat: '̂', bar: '̄', vec: '⃗', tilde: '̃', overline: '̄', dot: '̇' }[name]
          walk(`${a}${mark}`, shift, italic)
          i = j
        } else if (name === 'left' || name === 'right' || name === 'big' || name === 'Big' || name === ',' || name === ';' || name === '!' || name === ' ') {
          if (name === ',' || name === ';' || name === ' ') push(' ', shift, false)
        } else if (name in GREEK) push(GREEK[name], shift, !UPRIGHT.has(name) && /^[a-zA-Zα-ω]/.test(GREEK[name]) && italic)
        else push(name, shift, false)
      } else if (c === '^' || c === '_') {
        const [b, j] = braced(s, i + 1)
        walk(b, c === '^' ? 1 : -1, italic)
        i = j
      } else if (c === '{' || c === '}') i++
      else if (c === '-' ) {
        push('−', shift, false)
        i++
      } else if (/[a-zA-Z]/.test(c)) {
        push(c, shift, italic)
        i++
      } else {
        push(c, shift, false)
        i++
      }
    }
  }
  const parts = src.split(/(?<!\\)\$/)
  if (parts.length < 3) return [{ t: src.replace(/\\\$/g, '$'), shift: 0, italic: false }]
  parts.forEach((p, i) => {
    if (i % 2) walk(p, 0, true)
    else if (p) push(p.replace(/\\\$/g, '$'), 0, false)
  })
  return out
}

/** one or more lines of text, with mathtext; x, y is where the anchor goes; rotation in degrees */
const TextNode = ({ x, y, s, size, color, anchor = 'start', va = 'baseline', rotate = 0, weight, italic, opacity }: { x: number; y: number; s: string; size: number; color: string; anchor?: 'start' | 'middle' | 'end'; va?: 'top' | 'center' | 'bottom' | 'baseline'; rotate?: number; weight?: string; italic?: boolean; opacity?: number }) => {
  const lines = s.split('\n')
  const lh = size * 1.2
  const block = (lines.length - 1) * lh
  const first = va === 'top' ? size * 0.8 : va === 'center' ? size * 0.36 - block / 2 : va === 'bottom' ? -size * 0.22 - block : -block
  return (
    <text x={n2(x)} y={n2(y)} fontSize={n2(size)} fill={color} textAnchor={anchor} fontWeight={weight === 'normal' ? undefined : weight} fontStyle={italic ? 'italic' : undefined} opacity={opacity} transform={rotate ? `rotate(${n2(-rotate)} ${n2(x)} ${n2(y)})` : undefined} style={{ whiteSpace: 'pre' }}>
      {lines.map((line, k) => {
        const spans = mathSpans(line)
        let cur = 0
        return (
          <tspan key={k} x={n2(x)} dy={n2(k === 0 ? first : lh)}>
            {spans.map((sp, j) => {
              const shift = sp.shift === 1 ? -size * 0.38 : sp.shift === -1 ? size * 0.2 : 0
              const dy = shift - cur
              cur = shift
              return <tspan key={j} dy={dy ? n2(dy) : undefined} fontSize={sp.shift ? n2(size * 0.7) : undefined} fontStyle={sp.italic ? 'italic' : undefined}>{sp.t}</tspan>
            })}
          </tspan>
        )
      })}
    </text>
  )
}

// ---------------------------------------------------------------- markers

const circlePath = (x: number, y: number, r: number) => `M${n2(x - r)},${n2(y)}a${n2(r)},${n2(r)} 0 1,0 ${n2(2 * r)},0a${n2(r)},${n2(r)} 0 1,0 ${n2(-2 * r)},0Z`

const poly = (pts: number[][], x: number, y: number, s: number) => `M${pts.map(([u, v]) => `${n2(x + u * s)},${n2(y - v * s)}`).join('L')}Z`
const regular = (n: number, rot: number, inner?: number) => Array.from({ length: inner ? 2 * n : n }, (_, i) => {
  const r = inner && i % 2 ? inner : 1
  const t = (2 * Math.PI * i) / (inner ? 2 * n : n) + Math.PI / 2 + rot
  return [r * Math.cos(t), r * Math.sin(t)]
})
const STAR = regular(5, 0, 0.381966)
const PENTAGON = regular(5, 0)
const HEX1 = regular(6, 0)
const HEX2 = regular(6, Math.PI / 6)
const OCT = regular(8, Math.PI / 8)
const PLUS_FILLED = [[-1, -3], [1, -3], [1, -1], [3, -1], [3, 1], [1, 1], [1, 3], [-1, 3], [-1, 1], [-3, 1], [-3, -1], [-1, -1]].map(([a, b]) => [a / 3, b / 3])
const X_FILLED = [[-1, -2], [0, -1], [1, -2], [2, -1], [1, 0], [2, 1], [1, 2], [0, 1], [-1, 2], [-2, 1], [-1, 0], [-2, -1]].map(([a, b]) => [a / 2, b / 2])

/** one marker's outline; size is its nominal diameter in pixels. Stroke-only markers come back with stroke = true */
const markerPath = (m: string | null, x: number, y: number, size: number): { d: string; stroke: boolean } => {
  const r = size / 2
  const seg = (a: number, b: number, c: number, d: number) => `M${n2(x + a * r)},${n2(y - b * r)}L${n2(x + c * r)},${n2(y - d * r)}`
  switch (m) {
    case 's': return { d: poly([[-1, -1], [1, -1], [1, 1], [-1, 1]], x, y, r), stroke: false }
    case '^': return { d: poly([[0, 1], [-1, -1], [1, -1]], x, y, r), stroke: false }
    case 'v': return { d: poly([[0, -1], [1, 1], [-1, 1]], x, y, r), stroke: false }
    case '<': return { d: poly([[-1, 0], [1, 1], [1, -1]], x, y, r), stroke: false }
    case '>': return { d: poly([[1, 0], [-1, -1], [-1, 1]], x, y, r), stroke: false }
    case 'D': return { d: poly([[0, Math.SQRT2], [Math.SQRT2, 0], [0, -Math.SQRT2], [-Math.SQRT2, 0]], x, y, r), stroke: false }
    case 'd': return { d: poly([[0, Math.SQRT2], [0.6 * Math.SQRT2, 0], [0, -Math.SQRT2], [-0.6 * Math.SQRT2, 0]], x, y, r), stroke: false }
    case '*': return { d: poly(STAR, x, y, r), stroke: false }
    case 'p': return { d: poly(PENTAGON, x, y, r), stroke: false }
    case 'h': return { d: poly(HEX1, x, y, r), stroke: false }
    case 'H': return { d: poly(HEX2, x, y, r), stroke: false }
    case '8': return { d: poly(OCT, x, y, r), stroke: false }
    case 'P': return { d: poly(PLUS_FILLED, x, y, r), stroke: false }
    case 'X': return { d: poly(X_FILLED, x, y, r), stroke: false }
    case '.': return { d: circlePath(x, y, r / 2), stroke: false }
    case ',': return { d: `M${n2(x - 0.5)},${n2(y - 0.5)}h1v1h-1Z`, stroke: false }
    case 'x': return { d: seg(-1, -1, 1, 1) + seg(-1, 1, 1, -1), stroke: true }
    case '+': return { d: seg(-1, 0, 1, 0) + seg(0, -1, 0, 1), stroke: true }
    case '|': return { d: seg(0, -1, 0, 1), stroke: true }
    case '_': return { d: seg(-1, 0, 1, 0), stroke: true }
    case '1': return { d: seg(0, 0, 0, -1) + seg(0, 0, -0.866, 0.5) + seg(0, 0, 0.866, 0.5), stroke: true }
    case '2': return { d: seg(0, 0, 0, 1) + seg(0, 0, -0.866, -0.5) + seg(0, 0, 0.866, -0.5), stroke: true }
    case '3': return { d: seg(0, 0, -1, 0) + seg(0, 0, 0.5, 0.866) + seg(0, 0, 0.5, -0.866), stroke: true }
    case '4': return { d: seg(0, 0, 1, 0) + seg(0, 0, -0.5, 0.866) + seg(0, 0, -0.5, -0.866), stroke: true }
    default: return { d: circlePath(x, y, r), stroke: false }
  }
}

// ---------------------------------------------------------------- pictures

/** an image's pixels as a data URL (one canvas pixel per array element; the SVG scales it) */
const ImageArtistView = ({ a, X, Y, onPixel, labels, opacity }: { a: ImageArtist; X: (x: number) => number; Y: (y: number) => number; onPixel: (p: PixelInfo | null) => void; labels: boolean; opacity: number }) => {
  const url = useMemo(() => canvasUrl(a.cols, a.rows, a.pixels), [a.cols, a.rows, a.pixels])
  const [e0, e1, e2, e3] = a.extent
  // row 0 of the picture sits at extent[3], the last row at extent[2]
  const x0 = X(e0)
  const x1 = X(e1)
  const yTop = Y(e3)
  const yBottom = Y(e2)
  const left = Math.min(x0, x1)
  const top = Math.min(yTop, yBottom)
  const w = Math.abs(x1 - x0)
  const h = Math.abs(yBottom - yTop)
  const flipX = x1 < x0
  const flipY = yBottom < yTop
  // pixel values are written into the cells only when every one of them fits its cell: short forms (0.63, 41) at a
  // readable size, otherwise none (hovering still shows each value)
  const cellW = w / a.cols
  const cellH = h / a.rows
  const short = (v: number) => (Number.isInteger(v) ? String(v) : Math.abs(v) < 0.005 ? '0' : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2).replace(/\.?0+$/, ''))
  const texts = a.rgb ? [] : a.values.map(short)
  const fontSize = Math.min(11, cellW * 0.32, cellH * 0.45)
  const fits = texts.length <= 64 && texts.every((x) => x.length * fontSize * 0.62 <= cellW - 3)
  const small = labels && a.rows * a.cols <= 64 && fontSize >= 7 && fits
  const transform = flipX || flipY ? `translate(${flipX ? 2 * left + w : 0} ${flipY ? 2 * top + h : 0}) scale(${flipX ? -1 : 1} ${flipY ? -1 : 1})` : undefined
  return (
    <g
      opacity={opacity * a.alpha}
      onMouseMove={(e) => {
        const r = (e.currentTarget as SVGGElement).getBoundingClientRect()
        const col = Math.floor(((e.clientX - r.left) / r.width) * a.cols)
        const row = Math.floor(((e.clientY - r.top) / r.height) * a.rows)
        if (row < 0 || col < 0 || row >= a.rows || col >= a.cols) return onPixel(null)
        const i = row * a.cols + col
        onPixel({ row, col, value: a.rgb ? null : a.values[i], color: a.pixels[i], vmin: a.vmin, vmax: a.vmax, cmap: a.cmap, auto: a.auto })
      }}
      onMouseLeave={() => onPixel(null)}
    >
      {url && <image href={url} x={left} y={top} width={w} height={h} preserveAspectRatio="none" transform={transform} style={{ imageRendering: a.smooth ? 'auto' : 'pixelated' }} />}
      {small && !a.rgb && a.values.map((_, i) => {
        const r = Math.floor(i / a.cols)
        const c = i % a.cols
        const bright = parseInt(a.pixels[i].slice(1, 3), 16) * 0.3 + parseInt(a.pixels[i].slice(3, 5), 16) * 0.59 + parseInt(a.pixels[i].slice(5, 7), 16) * 0.11
        return (
          <text key={i} x={left + ((c + 0.5) * w) / a.cols} y={top + ((r + 0.5) * h) / a.rows + 3.5} textAnchor="middle" fontSize={fontSize} fill={bright > 140 ? '#0f172a' : '#f8fafc'} pointerEvents="none">
            {texts[i]}
          </text>
        )
      })}
    </g>
  )
}

const MeshView = ({ a, X, Y, opacity }: { a: MeshArtist; X: (x: number) => number; Y: (y: number) => number; opacity: number }) => {
  const sx = a.nx + 1
  const xA = X(a.X[0])
  const xB = X(a.X[a.nx])
  const yA = Y(a.Y[0])
  const yB = Y(a.Y[a.ny * sx])
  const url = useMemo(() => (a.uniform && !a.edge ? canvasUrl(a.nx, a.ny, a.colors, xB < xA, yB < yA) : null), [a.uniform, a.edge, a.nx, a.ny, a.colors, xA, xB, yA, yB])
  if (url) return <image href={url} x={Math.min(xA, xB)} y={Math.min(yA, yB)} width={Math.abs(xB - xA)} height={Math.abs(yB - yA)} preserveAspectRatio="none" opacity={opacity * a.alpha} style={{ imageRendering: 'pixelated' }} />
  const byColor = new Map<string, string>()
  for (let r = 0; r < a.ny; r++) {
    for (let c = 0; c < a.nx; c++) {
      const col = a.colors[r * a.nx + c]
      if (!col) continue
      const i = r * sx + c
      const d = `M${n2(X(a.X[i]))},${n2(Y(a.Y[i]))}L${n2(X(a.X[i + 1]))},${n2(Y(a.Y[i + 1]))}L${n2(X(a.X[i + sx + 1]))},${n2(Y(a.Y[i + sx + 1]))}L${n2(X(a.X[i + sx]))},${n2(Y(a.Y[i + sx]))}Z`
      byColor.set(col, (byColor.get(col) ?? '') + d)
    }
  }
  return (
    <g opacity={opacity * a.alpha}>
      {[...byColor.entries()].map(([col, d]) => <path key={col} d={d} fill={col} stroke={a.edge ?? col} strokeWidth={a.edge ? a.lw * PT : 0.4} />)}
    </g>
  )
}

// ---------------------------------------------------------------- legend

const LOC_CODES: Record<string, [number, number]> = { 'upper right': [1, 1], 'upper left': [0, 1], 'lower left': [0, 0], 'lower right': [1, 0], right: [1, 0.5], 'center left': [0, 0.5], 'center right': [1, 0.5], 'lower center': [0.5, 0], 'upper center': [0.5, 1], center: [0.5, 0.5] }
const BEST_ORDER = ['upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center']

interface Occupied { points: [number, number][]; boxes: [number, number, number, number][]; paths: [number, number][][] }

const legendSize = (l: LegendSpec) => {
  const F = l.size * PT
  const ncol = Math.max(1, l.ncol)
  const perCol = Math.ceil(l.entries.length / ncol) || 1
  const rowH = F * 0.97
  const columns = Array.from({ length: ncol }, (_, c) => l.entries.slice(c * perCol, (c + 1) * perCol))
  const colW = columns.map((es) => (es.length ? Math.max(...es.map((e) => textWidthPx(e.label, l.size))) : 0) + (l.handlelength + l.handletextpad) * F)
  const titleH = l.title ? textHeightPx(l.title, l.titleSize) + 0.2 * F : 0
  const titleW = l.title ? textWidthPx(l.title, l.titleSize) : 0
  const bodyW = colW.reduce((a, b) => a + b, 0) + (ncol - 1) * l.columnspacing * F
  const w = Math.max(bodyW, titleW) + 2 * l.borderpad * F
  const h = perCol * rowH + (perCol - 1) * l.labelspacing * F + titleH + 2 * l.borderpad * F
  return { w, h, F, perCol, rowH, columns, colW, titleH }
}

/** Legend._find_best_position */
const bestSpot = (l: LegendSpec, parent: Box, w: number, h: number, pad: number, occ: Occupied): { x: number; y: number } => {
  let best: { x: number; y: number } | null = null
  let bestScore = Infinity
  for (const name of BEST_ORDER) {
    const p = anchored(name, parent, w, h, pad)
    const inside = (x: number, y: number) => x > p.x && x < p.x + w && y > p.y && y < p.y + h
    let score = 0
    for (const [x, y] of occ.points) if (inside(x, y)) score++
    for (const b of occ.boxes) if (b[0] < p.x + w && b[2] > p.x && b[1] < p.y + h && b[3] > p.y) score++
    for (const line of occ.paths) {
      let hit = false
      for (let i = 0; i < line.length; i++) {
        const [x, y] = line[i]
        if (inside(x, y)) score++
        if (hit || i === 0) continue
        const [px, py] = line[i - 1]
        if (segBox(px, py, x, y, p.x, p.y, p.x + w, p.y + h)) hit = true
      }
      if (hit) score++
    }
    if (score < bestScore) {
      bestScore = score
      best = p
    }
    if (score === 0) break
  }
  void l
  return best ?? { x: parent.x, y: parent.y }
}

const segBox = (x0: number, y0: number, x1: number, y1: number, l: number, t: number, r: number, b: number) => {
  let t0 = 0
  let t1 = 1
  const dx = x1 - x0
  const dy = y1 - y0
  for (const [p, q] of [[-dx, x0 - l], [dx, r - x0], [-dy, y0 - t], [dy, b - y0]]) {
    if (p === 0) {
      if (q < 0) return false
      continue
    }
    const u = q / p
    if (p < 0) t0 = Math.max(t0, u)
    else t1 = Math.min(t1, u)
    if (t0 > t1) return false
  }
  return true
}

const anchored = (name: string, parent: Box, w: number, h: number, pad: number) => {
  const [cx, cy] = LOC_CODES[name]
  const cw = parent.w - 2 * pad
  const ch = parent.h - 2 * pad
  // cy measures upward, the page's y downward
  return { x: parent.x + pad + cx * (cw - w), y: parent.y + pad + (1 - cy) * (ch - h) }
}

const LegendHandle = ({ e, x, y, w, h, scale }: { e: LegendEntry; x: number; y: number; w: number; h: number; scale: number }) => {
  const cy = y + h / 2
  if (e.kind === 'patch') {
    return <rect x={x} y={y} width={w} height={h} fill={e.fc} stroke={e.ec === 'none' ? undefined : e.ec} strokeWidth={e.lw * PT} strokeDasharray={dashArray(e.dash, e.lw * PT)} opacity={e.alpha} />
  }
  if (e.kind === 'scatter') {
    const size = Math.sqrt(e.size) * PT * scale
    const m = markerPath(e.marker, x + w / 2, cy, size)
    return <path d={m.d} fill={m.stroke ? 'none' : e.fc} stroke={m.stroke ? e.fc : e.ec === 'none' ? e.fc : e.ec} strokeWidth={e.lw * PT} opacity={e.alpha} />
  }
  const lineW = e.width * PT
  const m = e.marker ? markerPath(e.marker, x + w / 2, cy, e.ms * PT * scale) : null
  const face = e.mfc === 'auto' ? e.color : e.mfc
  const edge = e.mec === 'auto' ? e.color : e.mec
  return (
    <g opacity={e.alpha}>
      {e.line && <line x1={x} x2={x + w} y1={cy} y2={cy} stroke={e.color} strokeWidth={lineW} strokeDasharray={dashArray(e.dash, lineW)} />}
      {m && <path d={m.d} fill={m.stroke ? 'none' : face} stroke={m.stroke ? edge : edge} strokeWidth={e.mew * PT} />}
    </g>
  )
}

const LegendBox = ({ l, parent, fig, occ }: { l: LegendSpec; parent: Box; fig: Box; occ: Occupied }) => {
  const { w, h, F, rowH, columns, colW, titleH } = legendSize(l)
  const pad = l.borderaxespad * F
  let spot: { x: number; y: number }
  const base = l.anchorCoord === 'figure' ? fig : parent
  const toPage = (fx: number, fy: number) => ({ x: base.x + fx * base.w, y: base.y + base.h - fy * base.h })
  if (Array.isArray(l.loc)) {
    const ref = l.anchor ? (l.anchor.length === 4 ? { x: toPage(l.anchor[0], 0).x, y: toPage(0, l.anchor[1] + l.anchor[3]).y, w: l.anchor[2] * base.w, h: l.anchor[3] * base.h } : { ...toPage(l.anchor[0], l.anchor[1]), w: 0, h: 0 }) : parent
    spot = { x: ref.x + l.loc[0] * ref.w, y: ref.y + ref.h - l.loc[1] * ref.h - h }
  } else {
    let ref = parent
    if (l.anchor) {
      const a = l.anchor
      ref = a.length === 4 ? { x: toPage(a[0], 0).x, y: toPage(0, a[1] + a[3]).y, w: a[2] * base.w, h: a[3] * base.h } : { ...toPage(a[0], a[1]), w: 0, h: 0 }
    }
    spot = l.loc === 'best' ? bestSpot(l, ref, w, h, pad, occ) : anchored(l.loc, ref, w, h, pad)
  }
  if (!l.entries.length && !l.title) return null
  const handleH = 0.7 * F
  return (
    <g>
      {l.frame && <rect x={spot.x} y={spot.y} width={w} height={h} rx={0.2 * F} fill={l.fc} fillOpacity={l.alpha} stroke={l.ec} strokeOpacity={l.alpha} strokeWidth={PT} />}
      {l.title && <TextNode x={spot.x + w / 2} y={spot.y + l.borderpad * F} s={l.title} size={l.titleSize * PT} color="#000000" anchor="middle" va="top" />}
      {columns.map((es, c) => {
        const left = spot.x + l.borderpad * F + colW.slice(0, c).reduce((a, b) => a + b + l.columnspacing * F, 0)
        return es.map((e, i) => {
          const top = spot.y + l.borderpad * F + titleH + i * (rowH + l.labelspacing * F)
          return (
            <g key={`${c}.${i}`}>
              <LegendHandle e={e} x={left} y={top + (rowH - handleH) / 2} w={l.handlelength * F} h={handleH} scale={l.markerscale} />
              <TextNode x={left + (l.handlelength + l.handletextpad) * F} y={top + rowH / 2} s={e.label} size={F} color={l.labelcolor ?? '#000000'} va="center" />
            </g>
          )
        })
      })}
    </g>
  )
}

// ---------------------------------------------------------------- text artists and annotation arrows

const textBox = (a: TextArtist, x: number, y: number): { left: number; top: number; w: number; h: number } => {
  const size = a.size
  const w = textWidthPx(a.text, size)
  const h = textHeightPx(a.text, size)
  const left = a.ha === 'center' ? x - w / 2 : a.ha === 'right' ? x - w : x
  const top = a.va === 'top' ? y : a.va === 'center' ? y - h / 2 : a.va === 'bottom' ? y - h : y - h * 0.8
  return { left, top, w, h }
}

const ArrowView = ({ arrow, from, to, box }: { arrow: TextArrow; from: [number, number]; to: [number, number]; box: { left: number; top: number; w: number; h: number } }) => {
  // start from the text box's edge along the line to the target
  const [cx, cy] = from
  const [tx, ty] = to
  const dx = tx - cx
  const dy = ty - cy
  const len = Math.hypot(dx, dy) || 1
  const ux = dx / len
  const uy = dy / len
  const halfW = box.w / 2 + 4 * PT
  const halfH = box.h / 2 + 4 * PT
  const out = Math.min(Math.abs(ux) > 1e-9 ? halfW / Math.abs(ux) : Infinity, Math.abs(uy) > 1e-9 ? halfH / Math.abs(uy) : Infinity)
  if (arrow.style === 'simple') {
    const total = Math.max(0, len - out)
    const s = arrow.shrink * total
    const x0 = cx + ux * (out + s)
    const y0 = cy + uy * (out + s)
    const x1 = tx - ux * s
    const y1 = ty - uy * s
    const width = (arrow.lw * PT) / 2
    const headW = (arrow.head * PT) / 2
    const headL = Math.min(12 * PT, Math.hypot(x1 - x0, y1 - y0))
    const bx = x1 - ux * headL
    const by = y1 - uy * headL
    const nx = -uy
    const ny = ux
    const pts = [[x0 + nx * width, y0 + ny * width], [bx + nx * width, by + ny * width], [bx + nx * headW, by + ny * headW], [x1, y1], [bx - nx * headW, by - ny * headW], [bx - nx * width, by - ny * width], [x0 - nx * width, y0 - ny * width]]
    return <polygon points={pts.map((p) => `${n2(p[0])},${n2(p[1])}`).join(' ')} fill={arrow.color} stroke={arrow.edge} strokeWidth={PT} />
  }
  const shrink = 2 * PT
  const x0 = cx + ux * (out + shrink)
  const y0 = cy + uy * (out + shrink)
  const x1 = tx - ux * shrink
  const y1 = ty - uy * shrink
  const lw = arrow.lw * PT
  const size = 0.4 * 10 * PT * Math.max(1, arrow.lw)
  const head = (x: number, y: number, ax: number, ay: number, filled: boolean) => {
    const ang = Math.PI / 6
    const rot = (r: number) => [ax * Math.cos(r) - ay * Math.sin(r), ax * Math.sin(r) + ay * Math.cos(r)]
    const [a1, b1] = rot(ang)
    const [a2, b2] = rot(-ang)
    const p1 = [x - a1 * size, y - b1 * size]
    const p2 = [x - a2 * size, y - b2 * size]
    return filled
      ? <polygon points={`${n2(x)},${n2(y)} ${n2(p1[0])},${n2(p1[1])} ${n2(p2[0])},${n2(p2[1])}`} fill={arrow.color} stroke={arrow.color} strokeWidth={lw} />
      : <path d={`M${n2(p1[0])},${n2(p1[1])}L${n2(x)},${n2(y)}L${n2(p2[0])},${n2(p2[1])}`} fill="none" stroke={arrow.color} strokeWidth={lw} />
  }
  const st = arrow.style.replace(/,.*$/, '')
  const filled = st.includes('|')
  const atEnd = st.endsWith('>') || st === 'fancy' || st === 'wedge'
  const atStart = st.startsWith('<')
  return (
    <g>
      <line x1={n2(x0)} y1={n2(y0)} x2={n2(x1)} y2={n2(y1)} stroke={arrow.color} strokeWidth={lw} />
      {atEnd && head(x1, y1, ux, uy, filled || st === 'fancy' || st === 'wedge')}
      {atStart && head(x0, y0, -ux, -uy, filled)}
    </g>
  )
}

// ---------------------------------------------------------------- axes

const scaleFn = (lim: [number, number], log: boolean, from: number, len: number) => {
  const t = log ? (v: number) => Math.log10(v <= 0 ? 1e-300 : v) : (v: number) => v
  const a = t(lim[0])
  const b = t(lim[1])
  const k = len / (b - a || 1)
  return (v: number) => from + (t(v) - a) * k
}

const AxisTicks = ({ axis, horizontal, plot, X, gridSide, title }: { axis: AxisSpec; horizontal: boolean; plot: Box; X: (v: number) => number; gridSide: { low: number; high: number }; title?: boolean }) => {
  const tick = (v: number, side: 'low' | 'high', len: number, width: number, k: string) => {
    const base = side === 'low' ? gridSide.low : gridSide.high
    const out = (side === 'low' ? 1 : -1) * (horizontal ? 1 : -1)
    const inward = axis.dir === 'in' ? -len : axis.dir === 'inout' ? -len / 2 : 0
    const outward = axis.dir === 'in' ? 0 : axis.dir === 'inout' ? len / 2 : len
    const p = X(v)
    const a = base + out * inward * PT
    const b = base + out * outward * PT
    return horizontal ? <line key={k} x1={n2(p)} x2={n2(p)} y1={n2(a)} y2={n2(b)} stroke={axis.color} strokeWidth={width * PT} /> : <line key={k} x1={n2(a)} x2={n2(b)} y1={n2(p)} y2={n2(p)} stroke={axis.color} strokeWidth={width * PT} />
  }
  const label = (v: number, s: string, side: 'low' | 'high', k: string, style?: AxisSpec['styles'][number]) => {
    if (!s || style?.visible === false) return null
    const base = side === 'low' ? gridSide.low : gridSide.high
    const out = (side === 'low' ? 1 : -1) * (horizontal ? 1 : -1)
    const tickOut = axis.dir === 'out' ? axis.length : axis.dir === 'inout' ? axis.length / 2 : 0
    const ticks = side === 'low' ? axis.tick1 : axis.tick2
    const dist = ((ticks ? tickOut : 0) + axis.pad) * PT
    const p = X(v)
    const size = (style?.size ?? axis.labelSize) * PT
    const rotation = style?.rotation ?? axis.rotation
    const color = style?.color ?? axis.labelColor
    if (horizontal) {
      const y = base + out * dist
      const haDefault = axis.ha ?? 'center'
      const anchor = haDefault === 'right' ? 'end' : haDefault === 'left' ? 'start' : 'middle'
      if (rotation) return <RotatedText key={k} x={p} y={y} s={s} size={size} color={color} rotation={rotation} ha={haDefault} va={axis.va ?? (side === 'low' ? 'top' : 'bottom')} weight={style?.weight} />
      return <TextNode key={k} x={p} y={y} s={s} size={size} color={color} anchor={anchor} va={side === 'low' ? 'top' : 'bottom'} weight={style?.weight} />
    }
    const x = base + out * dist
    const anchor = side === 'low' ? 'end' : 'start'
    if (rotation) return <RotatedText key={k} x={x} y={p} s={s} size={size} color={color} rotation={rotation} ha={axis.ha ?? (side === 'low' ? 'right' : 'left')} va={axis.va ?? 'center'} weight={style?.weight} />
    return <TextNode key={k} x={x} y={p} s={s} size={size} color={color} anchor={anchor} va={axis.va === 'top' ? 'top' : axis.va === 'bottom' ? 'bottom' : 'center'} weight={style?.weight} />
  }
  void title
  return (
    <g>
      {axis.tick1 && axis.minor.map((v, i) => tick(v, 'low', axis.minorLength, axis.minorWidth, `m1${i}`))}
      {axis.tick2 && axis.minor.map((v, i) => tick(v, 'high', axis.minorLength, axis.minorWidth, `m2${i}`))}
      {axis.tick1 && axis.at.map((v, i) => (axis.styles[i]?.visible === false ? null : tick(v, 'low', axis.length, axis.width, `t1${i}`)))}
      {axis.tick2 && axis.at.map((v, i) => (axis.styles[i]?.visible === false ? null : tick(v, 'high', axis.length, axis.width, `t2${i}`)))}
      {axis.label1 && axis.at.map((v, i) => label(v, axis.labels[i], 'low', `l1${i}`, axis.styles[i]))}
      {axis.label2 && axis.at.map((v, i) => label(v, axis.labels[i], 'high', `l2${i}`, axis.styles[i]))}
      {axis.label1 && axis.minor.map((v, i) => label(v, axis.minorLabels[i], 'low', `ml1${i}`))}
      {axis.label2 && axis.minor.map((v, i) => label(v, axis.minorLabels[i], 'high', `ml2${i}`))}
    </g>
  )
}

/** rotated text aligned by its rotated box, as matplotlib's default rotation mode does */
const RotatedText = ({ x, y, s, size, color, rotation, ha, va, weight }: { x: number; y: number; s: string; size: number; color: string; rotation: number; ha: string; va: string; weight?: string }) => {
  const r = (rotation * Math.PI) / 180
  const w = (textWidthPx(s, 1) / PT) * size
  const h = size * 1.1 * s.split('\n').length
  const hw = (Math.abs(w * Math.cos(r)) + Math.abs(h * Math.sin(r))) / 2
  const hh = (Math.abs(w * Math.sin(r)) + Math.abs(h * Math.cos(r))) / 2
  const cx = ha === 'left' ? x + hw : ha === 'right' ? x - hw : x
  const cy = va === 'top' ? y + hh : va === 'bottom' ? y - hh : y
  return <TextNode x={cx} y={cy} s={s} size={size} color={color} anchor="middle" va="center" rotate={rotation} weight={weight} />
}

const isTextVisible = (a: ArtistBase) => a.visible

interface AxesProps {
  ax: AxesSpec
  fig: Box
  /** the artists to keep bright (others fade); undefined = no focus */
  focus: { from: number; to: number } | null | undefined
  onPixel: (p: PixelInfo | null) => void
  clip: string
}

const AxesView = ({ ax, fig, focus, onPixel, clip }: AxesProps) => {
  if (!ax.visible) return null
  const [l, b, w, h] = ax.rect
  const plot: Box = { x: l * fig.w, y: fig.h * (1 - b - h), w: w * fig.w, h: h * fig.h }
  const X = scaleFn(ax.xlim, ax.xscale === 'log', plot.x, plot.w)
  const Y = scaleFn(ax.ylim, ax.yscale === 'log', plot.y + plot.h, -plot.h)
  const coordX = (c: Coord, v: number) => (c === 'data' ? X(v) : c === 'axes' ? plot.x + v * plot.w : v * fig.w)
  const coordY = (c: Coord, v: number) => (c === 'data' ? Y(v) : c === 'axes' ? plot.y + plot.h - v * plot.h : fig.h * (1 - v))
  const op = (i: number) => (focus === undefined || focus === null || (i >= focus.from && i <= focus.to) ? 1 : 0.28)
  const clipUrl = `url(#${clip})`
  const items: { z: number; node: ReactNode }[] = []
  const occ: Occupied = { points: [], boxes: [], paths: [] }

  ax.artists.forEach((a, i) => {
    if (!a.visible) return
    const o = op(i)
    const gx = (v: number) => coordX(a.tx, v)
    const gy = (v: number) => coordY(a.ty, v)
    const clipPath = a.clip ? clipUrl : undefined
    const nodes = artistNode(a, i, o, gx, gy, X, Y, coordX, coordY, plot, ax, onPixel, occ)
    if (nodes) items.push({ z: a.z, node: <g key={i} clipPath={clipPath}>{nodes}</g> })
  })

  const showAxis = !ax.axisOff
  const sp = ax.spines
  const spineLow = (s: typeof sp.bottom, dflt: number, map: (v: number) => number, fromEnd: boolean) => {
    if (!s.at) return dflt
    return s.at[0] === 'data' ? map(s.at[1]) : fromEnd ? plot.y + plot.h - s.at[1] * plot.h : plot.x + s.at[1] * plot.w
  }
  const xLow = spineLow(sp.bottom, plot.y + plot.h, Y, true)
  const xHigh = spineLow(sp.top, plot.y, Y, true)
  const yLow = spineLow(sp.left, plot.x, X, false)
  const yHigh = spineLow(sp.right, plot.x + plot.w, X, false)

  const g = ax.grid
  const gridNodes: ReactNode[] = []
  if (showAxis) {
    const stroke = { stroke: g.color, strokeWidth: g.width * PT, strokeDasharray: dashArray(g.dash, g.width * PT), strokeOpacity: g.alpha }
    if (g.x) ax.xticks.at.forEach((v, k) => gridNodes.push(<line key={`gx${k}`} x1={n2(X(v))} x2={n2(X(v))} y1={plot.y} y2={plot.y + plot.h} {...stroke} />))
    if (g.y) ax.yticks.at.forEach((v, k) => gridNodes.push(<line key={`gy${k}`} y1={n2(Y(v))} y2={n2(Y(v))} x1={plot.x} x2={plot.x + plot.w} {...stroke} />))
    if (g.xminor) ax.xticks.minor.forEach((v, k) => gridNodes.push(<line key={`gmx${k}`} x1={n2(X(v))} x2={n2(X(v))} y1={plot.y} y2={plot.y + plot.h} {...stroke} />))
    if (g.yminor) ax.yticks.minor.forEach((v, k) => gridNodes.push(<line key={`gmy${k}`} y1={n2(Y(v))} y2={n2(Y(v))} x1={plot.x} x2={plot.x + plot.w} {...stroke} />))
    if (gridNodes.length) items.push({ z: g.below ? 1.5 : 2.5, node: <g key="grid">{gridNodes}</g> })
  }
  if (showAxis) {
    const label = (axis: AxisSpec, horizontal: boolean): ReactNode => {
      if (!axis.label || !axis.visible) return null
      const high = axis.labelSide === 'high'
      const depth = tickLabelDepth(axis, horizontal, high)
      const hasTicks = (high ? axis.label2 : axis.label1) && [...axis.labels, ...axis.minorLabels].some((t) => t)
      const away = (hasTicks ? depth : (high ? axis.tick2 : axis.tick1) && axis.dir !== 'in' ? (axis.length) * PT : 0) + axis.labelPad * PT
      const size = axis.label.size * PT
      const lab = axis.label
      if (horizontal) {
        const x = lab.loc === 'left' ? plot.x : lab.loc === 'right' ? plot.x + plot.w : plot.x + plot.w / 2
        const anchor = lab.loc === 'left' ? 'start' : lab.loc === 'right' ? 'end' : 'middle'
        const y = high ? xHigh - away : xLow + away
        return <TextNode key="xl" x={x} y={y} s={lab.text} size={size} color={lab.color} anchor={anchor} va={high ? 'bottom' : 'top'} weight={lab.weight} italic={lab.italic} />
      }
      const y = lab.loc === 'left' ? plot.y + plot.h : lab.loc === 'right' ? plot.y : plot.y + plot.h / 2
      const anchor = lab.loc === 'left' ? 'start' : lab.loc === 'right' ? 'end' : 'middle'
      const x = high ? yHigh + away : yLow - away
      return <TextNode key="yl" x={x} y={y} s={lab.text} size={size} color={lab.color} anchor={anchor} va={high ? 'top' : 'bottom'} rotate={90} weight={lab.weight} italic={lab.italic} />
    }
    const offset = (axis: AxisSpec, horizontal: boolean): ReactNode => {
      if (!axis.offset || !axis.visible) return null
      const size = axis.labelSize * PT
      if (horizontal) {
        const depth = axis.label1 ? tickLabelDepth(axis, true, false) : 0
        return <TextNode key="xo" x={plot.x + plot.w} y={xLow + depth + 3 * PT} s={axis.offset} size={size} color={axis.labelColor} anchor="end" va="top" />
      }
      return axis.label2 && !axis.label1
        ? <TextNode key="yo" x={plot.x + plot.w} y={plot.y - 3 * PT} s={axis.offset} size={size} color={axis.labelColor} anchor="end" va="bottom" />
        : <TextNode key="yo" x={plot.x} y={plot.y - 3 * PT} s={axis.offset} size={size} color={axis.labelColor} va="bottom" />
    }
    items.push({
      z: 2.5,
      node: (
        <g key="axes">
          {ax.frame && (['left', 'right', 'bottom', 'top'] as const).map((n) => {
            const s = sp[n]
            if (!s.visible) return null
            const vertical = n === 'left' || n === 'right'
            const at = n === 'left' ? yLow : n === 'right' ? yHigh : n === 'bottom' ? xLow : xHigh
            return vertical
              ? <line key={n} x1={n2(at)} x2={n2(at)} y1={plot.y} y2={plot.y + plot.h} stroke={s.color} strokeWidth={s.width * PT} strokeDasharray={dashArray(s.dash, s.width * PT)} strokeLinecap="square" />
              : <line key={n} y1={n2(at)} y2={n2(at)} x1={plot.x} x2={plot.x + plot.w} stroke={s.color} strokeWidth={s.width * PT} strokeDasharray={dashArray(s.dash, s.width * PT)} strokeLinecap="square" />
          })}
          {ax.xticks.visible && <AxisTicks axis={ax.xticks} horizontal plot={plot} X={X} gridSide={{ low: xLow, high: xHigh }} />}
          {ax.yticks.visible && <AxisTicks axis={ax.yticks} horizontal={false} plot={plot} X={Y} gridSide={{ low: yLow, high: yHigh }} />}
          {label(ax.xticks, true)}
          {label(ax.yticks, false)}
          {offset(ax.xticks, true)}
          {offset(ax.yticks, false)}
        </g>
      ),
    })
  }
  const topExtra = showAxis && ax.xticks.visible && ax.xticks.label2 ? tickLabelDepth(ax.xticks, true, true) : 0
  ax.titles.forEach((t) => {
    const x = t.loc === 'left' ? plot.x : t.loc === 'right' ? plot.x + plot.w : plot.x + plot.w / 2
    const anchor = t.loc === 'left' ? 'start' : t.loc === 'right' ? 'end' : 'middle'
    const labelAbove = showAxis && ax.xticks.label && ax.xticks.labelSide === 'high' ? ax.xticks.labelPad * PT + textHeightPx(ax.xticks.label.text, ax.xticks.label.size) : 0
    const y = t.y !== null ? plot.y + plot.h - t.y * plot.h - t.pad * PT : plot.y - t.pad * PT - Math.max(topExtra, 0) - labelAbove
    items.push({ z: 3, node: <TextNode key={`title${t.loc}`} x={x} y={y} s={t.text} size={t.size * PT} color={t.color} anchor={anchor} va="bottom" weight={t.weight} italic={t.italic} /> })
  })
  ax.legends.forEach((lg, k) => items.push({ z: lg.z, node: <LegendBox key={`legend${k}`} l={lg} parent={plot} fig={fig} occ={occ} /> }))
  items.sort((p, q) => p.z - q.z)

  return (
    <g>
      <defs>
        <clipPath id={clip}><rect x={plot.x} y={plot.y} width={plot.w} height={plot.h} /></clipPath>
      </defs>
      {showAxis && ax.frame && ax.patchVisible && <rect x={plot.x} y={plot.y} width={plot.w} height={plot.h} fill={ax.facecolor} />}
      {items.map((it) => it.node)}
    </g>
  )
}

/** one artist's drawing; also notes what it covers for legend(loc='best') */
const artistNode = (a: Artist, i: number, o: number, gx: (v: number) => number, gy: (v: number) => number, X: (v: number) => number, Y: (v: number) => number, coordX: (c: Coord, v: number) => number, coordY: (c: Coord, v: number) => number, plot: Box, ax: AxesSpec, onPixel: (p: PixelInfo | null) => void, occ: Occupied): ReactNode => {
  switch (a.kind) {
    case 'image':
      occ.boxes.push([Math.min(X(a.extent[0]), X(a.extent[1])), Math.min(Y(a.extent[2]), Y(a.extent[3])), Math.max(X(a.extent[0]), X(a.extent[1])), Math.max(Y(a.extent[2]), Y(a.extent[3]))])
      return <ImageArtistView a={a} X={X} Y={Y} onPixel={onPixel} labels={!ax.artists.some((x) => x.kind === 'text')} opacity={o} />
    case 'mesh':
      occ.boxes.push([Math.min(X(a.X[0]), X(a.X[a.nx])), Math.min(Y(a.Y[0]), Y(a.Y[a.ny * (a.nx + 1)])), Math.max(X(a.X[0]), X(a.X[a.nx])), Math.max(Y(a.Y[0]), Y(a.Y[a.ny * (a.nx + 1)]))])
      return <MeshView a={a} X={X} Y={Y} opacity={o} />
    case 'line': {
      const lw = a.width * PT
      const n = a.x.length
      let d = ''
      let pen = false
      const pts: [number, number][] = []
      const at = (k: number) => [gx(a.x[k]), gy(a.y[k])] as const
      const good = (k: number) => Number.isFinite(a.x[k]) && Number.isFinite(a.y[k])
      for (let k = 0; k < n; k++) {
        if (!good(k)) {
          pen = false
          continue
        }
        const [px, py] = at(k)
        pts.push([px, py])
        if (!pen || !a.steps) {
          d += `${pen ? 'L' : 'M'}${n2(px)},${n2(py)}`
        } else {
          const [qx, qy] = at(k - 1)
          if (a.steps === 'pre') d += `L${n2(qx)},${n2(py)}L${n2(px)},${n2(py)}`
          else if (a.steps === 'post') d += `L${n2(px)},${n2(qy)}L${n2(px)},${n2(py)}`
          else {
            const m = (qx + px) / 2
            d += `L${n2(m)},${n2(qy)}L${n2(m)},${n2(py)}L${n2(px)},${n2(py)}`
          }
        }
        pen = true
      }
      if (pts.length) occ.paths.push(pts)
      const every = a.markevery ?? 1
      const size = a.ms * PT
      const edge = a.mec === 'auto' ? a.color : a.mec
      const face = a.mfc === 'auto' ? a.color : a.mfc
      let md = ''
      let stroked = false
      if (a.marker && n <= 20000) {
        for (let k = 0; k < n; k += every) {
          if (!good(k)) continue
          const [px, py] = at(k)
          const m = markerPath(a.marker, px, py, size)
          md += m.d
          stroked = m.stroke
        }
      }
      return (
        <g opacity={o * a.alpha}>
          {a.line && lw > 0 && d && <path d={d} fill="none" stroke={a.color} strokeWidth={n2(lw)} strokeDasharray={dashArray(a.dash, lw)} strokeLinejoin="round" />}
          {md && <path d={md} fill={stroked ? 'none' : face} stroke={edge} strokeWidth={n2(a.mew * PT)} />}
        </g>
      )
    }
    case 'scatter': {
      const groups: ReactNode[] = []
      let run = ''
      let key = ''
      let fill = ''
      let stroke = ''
      let width = 0
      const flush = () => {
        if (!run) return
        groups.push(<path key={groups.length} d={run} fill={fill} stroke={stroke} strokeWidth={n2(width)} />)
        run = ''
      }
      a.x.forEach((vx, k) => {
        const vy = a.y[k]
        if (!Number.isFinite(vx) || !Number.isFinite(vy)) return
        const px = gx(vx)
        const py = gy(vy)
        occ.points.push([px, py])
        const size = Math.sqrt(a.sizes[k % a.sizes.length]) * PT
        const c = a.colors[k % a.colors.length]
        const e = a.edges ? a.edges[k % a.edges.length] : c
        const lw = a.lw[k % a.lw.length] * PT
        const m = markerPath(a.marker, px, py, size)
        const f = m.stroke ? 'none' : c
        const s = m.stroke ? c : e
        const id = `${f}|${s}|${lw}`
        if (id !== key) {
          flush()
          key = id
          fill = f
          stroke = s
          width = lw
        }
        run += m.d
      })
      flush()
      return <g opacity={o * a.alpha}>{groups}</g>
    }
    case 'bar':
      return (
        <g opacity={o * a.alpha}>
          {a.x.map((p, k) => {
            const lo = a.bottoms[k]
            const hi = lo + a.heights[k]
            const w = a.widths[k]
            const fill = a.colors[k % a.colors.length]
            const edge = a.edges[k % a.edges.length]
            const [x0, x1, y0, y1] = a.horizontal ? [gx(lo), gx(hi), gy(p), gy(p + w)] : [gx(p), gx(p + w), gy(lo), gy(hi)]
            const box: [number, number, number, number] = [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)]
            occ.boxes.push(box)
            return <rect key={k} x={n2(box[0])} y={n2(box[1])} width={n2(box[2] - box[0])} height={n2(box[3] - box[1])} fill={fill} stroke={edge === 'none' ? undefined : edge} strokeWidth={edge === 'none' ? undefined : n2(a.lw * PT)} strokeDasharray={dashArray(a.dash, a.lw * PT)} />
          })}
        </g>
      )
    case 'poly': {
      const groups: ReactNode[] = []
      let run = ''
      let fc = ''
      let ec = ''
      const nfc = a.fcs.length
      const nec = a.ecs.length
      const flush = () => {
        if (!run) return
        const seam = a.seamless && a.alpha === 1 && fc !== 'none'
        groups.push(<path key={groups.length} d={run} fill={fc} stroke={ec !== 'none' ? ec : seam ? fc : undefined} strokeWidth={ec !== 'none' ? n2(a.lw * PT) : seam ? 0.5 : undefined} strokeDasharray={ec !== 'none' ? dashArray(a.dash, a.lw * PT) : undefined} strokeLinejoin={a.seamless ? 'round' : undefined} />)
        run = ''
      }
      a.paths.forEach((p, k) => {
        if (!p.length) return
        const f = a.fcs[k % nfc]
        const e = a.ecs[k % nec]
        if (f !== fc || e !== ec) {
          flush()
          fc = f
          ec = e
        }
        let d = ''
        let pen = false
        let x0 = Infinity
        let x1 = -Infinity
        let y0 = Infinity
        let y1 = -Infinity
        for (let j = 0; j + 1 < p.length; j += 2) {
          if (!Number.isFinite(p[j]) || !Number.isFinite(p[j + 1])) {
            pen = false
            continue
          }
          const px = gx(p[j])
          const py = gy(p[j + 1])
          if (px < x0) x0 = px
          if (px > x1) x1 = px
          if (py < y0) y0 = py
          if (py > y1) y1 = py
          d += `${pen ? 'L' : 'M'}${n2(px)},${n2(py)}`
          pen = true
        }
        if (a.closed && d) d += 'Z'
        run += d
        if (a.paths.length < 500 && x0 <= x1) occ.boxes.push([x0, y0, x1, y1])
      })
      flush()
      return <g opacity={o * a.alpha}>{groups}</g>
    }
    case 'segs': {
      const groups: ReactNode[] = []
      let run = ''
      let color = ''
      let width = 0
      const flush = () => {
        if (!run) return
        groups.push(<path key={groups.length} d={run} fill="none" stroke={color} strokeWidth={n2(width * PT)} strokeDasharray={dashArray(a.dash, width * PT)} />)
        run = ''
      }
      a.paths.forEach((p, k) => {
        const c = a.colors[k % a.colors.length]
        const w = a.widths[k % a.widths.length]
        if (c !== color || w !== width) {
          flush()
          color = c
          width = w
        }
        let pen = false
        const pts: [number, number][] = []
        for (let j = 0; j + 1 < p.length; j += 2) {
          if (!Number.isFinite(p[j]) || !Number.isFinite(p[j + 1])) {
            pen = false
            continue
          }
          const px = gx(p[j])
          const py = gy(p[j + 1])
          run += `${pen ? 'L' : 'M'}${n2(px)},${n2(py)}`
          pen = true
          pts.push([px, py])
        }
        if (a.paths.length < 300 && pts.length) occ.paths.push(pts)
      })
      flush()
      return <g opacity={o * a.alpha}>{groups}</g>
    }
    case 'text': {
      if (!a.text && !a.arrow) return null
      const x = gx(a.x) + a.dx * PT
      const y = gy(a.y) - a.dy * PT
      const size = a.size * PT
      const box = textBox(a, x, y)
      return (
        <g opacity={o * a.alpha}>
          {a.arrow && <ArrowView arrow={a.arrow} from={[box.left + box.w / 2, box.top + box.h / 2]} to={[coordX(a.arrow.tx, a.arrow.x), coordY(a.arrow.ty, a.arrow.y)]} box={box} />}
          {a.bbox && a.text && (
            <rect x={n2(box.left - a.bbox.pad * PT)} y={n2(box.top - a.bbox.pad * PT)} width={n2(box.w + 2 * a.bbox.pad * PT)} height={n2(box.h + 2 * a.bbox.pad * PT)} rx={a.bbox.round ? a.bbox.pad * PT : 0} fill={a.bbox.fc} fillOpacity={a.bbox.alpha} stroke={a.bbox.ec === 'none' ? undefined : a.bbox.ec} strokeOpacity={a.bbox.alpha} strokeWidth={a.bbox.lw * PT} transform={a.rotation ? `rotate(${-a.rotation} ${x} ${y})` : undefined} />
          )}
          {a.text && !a.rotation && <TextNode x={x} y={y} s={a.text} size={size} color={a.color} anchor={a.ha === 'center' ? 'middle' : a.ha === 'right' ? 'end' : 'start'} va={a.va} weight={a.weight} italic={a.italic} />}
          {a.text && !!a.rotation && <RotatedText x={x} y={y} s={a.text} size={size} color={a.color} rotation={a.rotation} ha={a.ha} va={a.va} weight={a.weight} />}
        </g>
      )
    }
  }
  void plot
  return null
}

const FigureText = ({ a, fig }: { a: TextArtist; fig: Box }) => {
  if (!a.visible || !a.text) return null
  const x = a.x * fig.w + a.dx * PT
  const y = fig.h * (1 - a.y) - a.dy * PT
  const size = a.size * PT
  const box = textBox(a, x, y)
  return (
    <g opacity={a.alpha}>
      {a.bbox && <rect x={n2(box.left - a.bbox.pad * PT)} y={n2(box.top - a.bbox.pad * PT)} width={n2(box.w + 2 * a.bbox.pad * PT)} height={n2(box.h + 2 * a.bbox.pad * PT)} rx={a.bbox.round ? a.bbox.pad * PT : 0} fill={a.bbox.fc} fillOpacity={a.bbox.alpha} stroke={a.bbox.ec === 'none' ? undefined : a.bbox.ec} strokeWidth={a.bbox.lw * PT} />}
      {a.rotation
        ? <RotatedText x={x} y={y} s={a.text} size={size} color={a.color} rotation={a.rotation} ha={a.ha} va={a.va} weight={a.weight} />
        : <TextNode x={x} y={y} s={a.text} size={size} color={a.color} anchor={a.ha === 'center' ? 'middle' : a.ha === 'right' ? 'end' : 'start'} va={a.va} weight={a.weight} italic={a.italic} />}
    </g>
  )
}

let figId = 0

const FigureBody = memo(({ fig, focus, onPixel, clip }: { fig: FigureSpec; focus?: { ax: number; artist: number | null; last?: number } | null; onPixel: (p: PixelInfo | null) => void; clip: string }) => {
  const W = fig.size[0] * PX
  const H = fig.size[1] * PX
  const box: Box = { x: 0, y: 0, w: W, h: H }
  return (
    <>
      <rect x={0} y={0} width={W} height={H} fill={fig.facecolor} />
      {fig.axes.map((ax, k) => {
        const f = focus && focus.ax === k ? (focus.artist === null ? null : { from: focus.artist, to: focus.last ?? focus.artist }) : focus ? { from: -1, to: -1 } : undefined
        return <AxesView key={k} ax={ax} fig={box} focus={f} onPixel={onPixel} clip={`${clip}-${k}`} />
      })}
      {fig.legends.map((l, k) => <LegendBox key={k} l={l} parent={box} fig={box} occ={{ points: [], boxes: [], paths: [] }} />)}
      {fig.texts.filter(isTextVisible).map((a, k) => <FigureText key={k} a={a} fig={box} />)}
    </>
  )
})

/** One figure; focus = { ax, artist } dims the other artists of that Axes */
export const FigureView = ({ fig, focus }: { fig: FigureSpec; focus?: { ax: number; artist: number | null; last?: number } | null }) => {
  const { t } = useTranslation()
  const [pixel, setPixel] = useState<PixelInfo | null>(null)
  const [clip] = useState(() => `fig${++figId}`)
  useEffect(() => setPixel(null), [fig])
  const W = fig.size[0] * PX
  const H = fig.size[1] * PX
  return (
    <figure className="space-y-1">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg border border-slate-200 bg-white" role="img" aria-label={t('playground.view.figure_label', { n: fig.axes.length })} style={{ fontFamily: FONT }}>
        <FigureBody fig={fig} focus={focus} onPixel={setPixel} clip={clip} />
      </svg>
      <figcaption className="min-h-5 font-mono text-[11px] text-slate-500" aria-live="polite">
        {pixel
          ? pixel.value === null
            ? t('playground.view.pixel_rgb', { row: pixel.row, col: pixel.col, color: pixel.color })
            : t('playground.view.pixel', { row: pixel.row, col: pixel.col, value: fmt(pixel.value), vmin: fmt(pixel.vmin), vmax: fmt(pixel.vmax), cmap: pixel.cmap ?? '', t: fmt(Math.round(((pixel.value - pixel.vmin) / (pixel.vmax - pixel.vmin || 1)) * 1000) / 1000), color: pixel.color })
          : fig.axes.some((a) => a.artists.some((x) => x.kind === 'image')) ? t('playground.view.pixel_hint') : ''}
      </figcaption>
    </figure>
  )
}
