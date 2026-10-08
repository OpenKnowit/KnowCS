import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { niceTicks } from '../../lib/chart'
import { cmapColor } from '../../lib/pyPlot'
import type { Artist, AxesSpec, FigureSpec } from '../../lib/pyEvents'

// --- 把沙盒里的 matplotlib 图（FigureSpec）画成 SVG：子图网格、刻度、图例、色条；选中一步时其余图元变淡 ---

const PX = 100 // pixels per inch, matplotlib's default dpi
const M = { l: 46, r: 14, t: 26, b: 36 }

type Range = [number, number]

const fmt = (v: number) => {
  const a = Math.abs(v)
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1)
  return String(+v.toPrecision(6))
}

/** data limits of an Axes: user limits, else the artists' extent plus matplotlib's 5 % margins */
function limits(ax: AxesSpec): { x: Range; y: Range } {
  const xs: number[] = []
  const ys: number[] = []
  let stickyY0 = false
  let image: Artist | null = null
  for (const a of ax.artists) {
    switch (a.kind) {
      case 'line': case 'scatter': xs.push(...a.x); ys.push(...a.y); break
      case 'fill': xs.push(...a.x); ys.push(...a.y1, ...a.y2); break
      case 'bar':
        a.x.forEach((x, i) => {
          const lo = a.bottoms[i] ?? 0
          if (a.horizontal) { ys.push(x - a.width / 2, x + a.width / 2); xs.push(lo, lo + a.heights[i]) } else { xs.push(x - a.width / 2, x + a.width / 2); ys.push(lo, lo + a.heights[i]) }
        })
        stickyY0 = !a.horizontal
        break
      case 'hist': xs.push(a.edges[0], a.edges[a.edges.length - 1]); ys.push(0, ...a.counts); stickyY0 = true; break
      case 'image': image = a; break
      case 'hline': ys.push(a.at); break
      case 'vline': xs.push(a.at); break
      case 'text': xs.push(a.x); ys.push(a.y); break
    }
  }
  const pad = (v: number[], sticky0: boolean): Range => {
    const f = v.filter(Number.isFinite)
    if (!f.length) return [0, 1]
    let lo = Math.min(...f)
    let hi = Math.max(...f)
    if (lo === hi) { lo -= 0.5; hi += 0.5 }
    const m = (hi - lo) * 0.05
    return [sticky0 && lo >= 0 ? Math.min(lo, 0) : lo - m, hi + m]
  }
  let x: Range = pad(xs, false)
  let y: Range = pad(ys, stickyY0)
  if (image && image.kind === 'image') {
    const [x0, x1, y0, y1] = image.extent
    x = xs.length ? [Math.min(x0, x[0]), Math.max(x1, x[1])] : [x0, x1]
    y = ys.length ? [Math.min(y1, y[0]), Math.max(y0, y[1])] : [y1, y0]
  }
  return { x: ax.xlim ?? x, y: ax.ylim ?? y }
}

/** an image's pixels as a data URL (one canvas pixel per array element; the SVG scales it without smoothing) */
function useImageUrl(a: Artist & { kind: 'image' }): string | null {
  return useMemo(() => {
    if (typeof document === 'undefined') return null
    const c = document.createElement('canvas')
    c.width = a.cols
    c.height = a.rows
    const ctx = c.getContext('2d')
    if (!ctx) return null
    const img = ctx.createImageData(a.cols, a.rows)
    a.pixels.forEach((hex, i) => {
      img.data[i * 4] = parseInt(hex.slice(1, 3), 16)
      img.data[i * 4 + 1] = parseInt(hex.slice(3, 5), 16)
      img.data[i * 4 + 2] = parseInt(hex.slice(5, 7), 16)
      img.data[i * 4 + 3] = 255
    })
    ctx.putImageData(img, 0, 0)
    return c.toDataURL()
  }, [a])
}

interface Box { x: number; y: number; w: number; h: number }

interface AxesProps {
  ax: AxesSpec
  box: Box
  /** index of the artist to keep bright (others fade); undefined = no focus */
  focus: number | null | undefined
  onPixel: (p: PixelInfo | null) => void
  clip: string
}

export interface PixelInfo { row: number; col: number; value: number | null; color: string; vmin: number; vmax: number; cmap: string | null; auto: boolean }

const MarkerShape = ({ m, x, y, r, fill, stroke }: { m: string | null; x: number; y: number; r: number; fill: string; stroke?: string | null }) => {
  switch (m) {
    case 's': return <rect x={x - r} y={y - r} width={r * 2} height={r * 2} fill={fill} stroke={stroke ?? undefined} />
    case '^': return <polygon points={`${x},${y - r * 1.2} ${x - r},${y + r * 0.8} ${x + r},${y + r * 0.8}`} fill={fill} stroke={stroke ?? undefined} />
    case 'v': return <polygon points={`${x},${y + r * 1.2} ${x - r},${y - r * 0.8} ${x + r},${y - r * 0.8}`} fill={fill} stroke={stroke ?? undefined} />
    case 'D': return <polygon points={`${x},${y - r * 1.3} ${x + r},${y} ${x},${y + r * 1.3} ${x - r},${y}`} fill={fill} stroke={stroke ?? undefined} />
    case 'x': return <path d={`M${x - r},${y - r}L${x + r},${y + r}M${x - r},${y + r}L${x + r},${y - r}`} stroke={fill} strokeWidth={1.6} />
    case '+': return <path d={`M${x - r},${y}L${x + r},${y}M${x},${y - r}L${x},${y + r}`} stroke={fill} strokeWidth={1.6} />
    case '*': return <polygon points={Array.from({ length: 10 }, (_, k) => { const rr = k % 2 ? r * 0.45 : r * 1.25; const t = (Math.PI * k) / 5 - Math.PI / 2; return `${x + rr * Math.cos(t)},${y + rr * Math.sin(t)}` }).join(' ')} fill={fill} stroke={stroke ?? undefined} />
    case '.': return <circle cx={x} cy={y} r={Math.max(1.5, r * 0.45)} fill={fill} />
    default: return <circle cx={x} cy={y} r={r} fill={fill} stroke={stroke ?? undefined} />
  }
}

const ImageArtist = ({ a, X, Y, onPixel, labels }: { a: Artist & { kind: 'image' }; X: (x: number) => number; Y: (y: number) => number; onPixel: (p: PixelInfo | null) => void; labels: boolean }) => {
  const url = useImageUrl(a)
  const [x0, x1, y0, y1] = a.extent
  const left = Math.min(X(x0), X(x1))
  const top = Math.min(Y(y0), Y(y1))
  const w = Math.abs(X(x1) - X(x0))
  const h = Math.abs(Y(y1) - Y(y0))
  const small = labels && a.rows * a.cols <= 64
  return (
    <g
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
      {url && <image href={url} x={left} y={top} width={w} height={h} preserveAspectRatio="none" style={{ imageRendering: 'pixelated' }} />}
      {small && !a.rgb && a.values.map((v, i) => {
        const r = Math.floor(i / a.cols)
        const c = i % a.cols
        const bright = parseInt(a.pixels[i].slice(1, 3), 16) * 0.3 + parseInt(a.pixels[i].slice(3, 5), 16) * 0.59 + parseInt(a.pixels[i].slice(5, 7), 16) * 0.11
        return (
          <text key={i} x={left + ((c + 0.5) * w) / a.cols} y={top + ((r + 0.5) * h) / a.rows + 3.5} textAnchor="middle" fontSize={Math.min(11, (w / a.cols) * 0.32)} fill={bright > 140 ? '#0f172a' : '#f8fafc'} pointerEvents="none">
            {fmt(v)}
          </text>
        )
      })}
    </g>
  )
}

const AxesView = ({ ax, box, focus, onPixel, clip }: AxesProps) => {
  const { x: xr, y: yr0 } = limits(ax)
  const yr: Range = ax.yInverted ? [yr0[1], yr0[0]] : yr0
  let plot: Box = { x: box.x + M.l, y: box.y + M.t, w: Math.max(10, box.w - M.l - M.r - (ax.colorbar ? 46 : 0)), h: Math.max(10, box.h - M.t - M.b) }
  if (ax.axisOff) plot = { x: box.x + 6, y: box.y + (ax.title ? 20 : 6), w: box.w - 12 - (ax.colorbar ? 46 : 0), h: box.h - (ax.title ? 26 : 12) }
  if (ax.equal) {
    const dx = Math.abs(xr[1] - xr[0])
    const dy = Math.abs(yr[1] - yr[0])
    const s = Math.min(plot.w / dx, plot.h / dy)
    const w = dx * s
    const h = dy * s
    plot = { x: plot.x + (plot.w - w) / 2, y: plot.y + (plot.h - h) / 2, w, h }
  }
  const X = (v: number) => plot.x + ((v - xr[0]) / (xr[1] - xr[0])) * plot.w
  const Y = (v: number) => plot.y + plot.h - ((v - yr[0]) / (yr[1] - yr[0])) * plot.h
  // image axes: ticks on whole pixels (row / column numbers)
  const pixelAxes = ax.artists.length > 0 && ax.artists.every((a) => a.kind === 'image' || a.kind === 'text')
  const ticksFor = (r: Range) => {
    const lo = Math.min(...r)
    const hi = Math.max(...r)
    const raw = niceTicks(lo, hi, 7)
    let at = raw
    if (pixelAxes) {
      const step = Math.max(1, Math.round(raw.length > 1 ? raw[1] - raw[0] : 1))
      at = []
      for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) at.push(v)
    }
    return { at: at.filter((v) => v >= lo - 1e-9 && v <= hi + 1e-9), labels: null }
  }
  const xt = ax.xticks ?? ticksFor(xr)
  const yt = ax.yticks ?? ticksFor(yr)
  // bars with string categories label their own ticks
  const catBar = ax.artists.find((a) => a.kind === 'bar' && a.tickLabels)
  const xticks = !ax.xticks && catBar && catBar.kind === 'bar' && !catBar.horizontal ? { at: catBar.x, labels: catBar.tickLabels } : xt
  const yticks = !ax.yticks && catBar && catBar.kind === 'bar' && catBar.horizontal ? { at: catBar.x, labels: catBar.tickLabels } : yt
  const op = (i: number) => (focus === undefined || focus === null || focus === i ? 1 : 0.28)
  const legendItems = ax.legend ? ax.artists.flatMap((a) => ('label' in a && a.label && !a.label.startsWith('_') ? [a] : [])) : []
  const id = `${clip}-${ax.pos.index}`

  return (
    <g>
      <defs>
        <clipPath id={id}><rect x={plot.x} y={plot.y} width={plot.w} height={plot.h} /></clipPath>
      </defs>
      {!ax.axisOff && <rect x={plot.x} y={plot.y} width={plot.w} height={plot.h} fill="var(--fig-bg, #fff)" />}
      {ax.grid && !ax.axisOff && (
        <g stroke="#e2e8f0" strokeWidth={1}>
          {xticks.at.map((v) => <line key={`gx${v}`} x1={X(v)} x2={X(v)} y1={plot.y} y2={plot.y + plot.h} />)}
          {yticks.at.map((v) => <line key={`gy${v}`} y1={Y(v)} y2={Y(v)} x1={plot.x} x2={plot.x + plot.w} />)}
        </g>
      )}
      <g clipPath={`url(#${id})`}>
        {ax.artists.map((a, i) => {
          const o = op(i)
          switch (a.kind) {
            case 'image': return <g key={i} opacity={o}><ImageArtist a={a} X={X} Y={Y} onPixel={onPixel} labels={!ax.artists.some((x) => x.kind === 'text')} /></g>
            case 'line':
              return (
                <g key={i} opacity={o * a.alpha}>
                  {a.width > 0 && <polyline fill="none" stroke={a.color} strokeWidth={a.width * 1.2} strokeDasharray={a.dash ?? undefined} strokeLinejoin="round" points={a.x.map((x, k) => `${X(x).toFixed(1)},${Y(a.y[k]).toFixed(1)}`).join(' ')} />}
                  {a.marker && a.x.length <= 400 && a.x.map((x, k) => <MarkerShape key={k} m={a.marker} x={X(x)} y={Y(a.y[k])} r={3.2} fill={a.color} />)}
                </g>
              )
            case 'scatter':
              return (
                <g key={i} opacity={o * a.alpha}>
                  {a.x.map((x, k) => <MarkerShape key={k} m={a.marker} x={X(x)} y={Y(a.y[k])} r={Math.sqrt(a.sizes[k % a.sizes.length]) / 1.6} fill={a.colors[k % a.colors.length]} stroke={a.edge} />)}
                </g>
              )
            case 'bar':
              return (
                <g key={i} opacity={o * a.alpha}>
                  {a.x.map((x, k) => {
                    const lo = a.bottoms[k] ?? 0
                    const hi = lo + a.heights[k]
                    const fill = a.colors[k % a.colors.length]
                    if (a.horizontal) {
                      const y0 = Y(x + a.width / 2)
                      const y1 = Y(x - a.width / 2)
                      return <rect key={k} x={Math.min(X(lo), X(hi))} y={Math.min(y0, y1)} width={Math.abs(X(hi) - X(lo))} height={Math.abs(y1 - y0)} fill={fill} />
                    }
                    return <rect key={k} x={X(x - a.width / 2)} y={Math.min(Y(lo), Y(hi))} width={X(x + a.width / 2) - X(x - a.width / 2)} height={Math.abs(Y(hi) - Y(lo))} fill={fill} />
                  })}
                </g>
              )
            case 'hist':
              return (
                <g key={i} opacity={o * a.alpha}>
                  {a.counts.map((c, k) => <rect key={k} x={X(a.edges[k])} y={Math.min(Y(0), Y(c))} width={Math.max(0, X(a.edges[k + 1]) - X(a.edges[k]))} height={Math.abs(Y(c) - Y(0))} fill={a.color} stroke="#fff" strokeWidth={0.5} />)}
                </g>
              )
            case 'fill':
              return <polygon key={i} opacity={o * a.alpha} fill={a.color} points={[...a.x.map((x, k) => `${X(x)},${Y(a.y1[k])}`), ...a.x.map((x, k) => `${X(x)},${Y(a.y2[k])}`).reverse()].join(' ')} />
            case 'hline': return <line key={i} opacity={o} x1={plot.x} x2={plot.x + plot.w} y1={Y(a.at)} y2={Y(a.at)} stroke={a.color} strokeWidth={a.width * 1.2} strokeDasharray={a.dash ?? undefined} />
            case 'vline': return <line key={i} opacity={o} y1={plot.y} y2={plot.y + plot.h} x1={X(a.at)} x2={X(a.at)} stroke={a.color} strokeWidth={a.width * 1.2} strokeDasharray={a.dash ?? undefined} />
            case 'text':
              return <text key={i} opacity={o} x={X(a.x)} y={Y(a.y)} fill={a.color} fontSize={a.size * 1.1} textAnchor={a.ha === 'center' ? 'middle' : a.ha === 'right' ? 'end' : 'start'} dominantBaseline={a.va === 'center' ? 'central' : a.va === 'top' ? 'hanging' : 'auto'}>{a.text}</text>
          }
          return null
        })}
      </g>
      {!ax.axisOff && (
        <g fontSize={10} fill="#334155">
          <rect x={plot.x} y={plot.y} width={plot.w} height={plot.h} fill="none" stroke="#334155" strokeWidth={1} />
          {xticks.at.map((v, k) => (
            <g key={`x${k}`}>
              <line x1={X(v)} x2={X(v)} y1={plot.y + plot.h} y2={plot.y + plot.h + 4} stroke="#334155" />
              <text x={X(v)} y={plot.y + plot.h + 15} textAnchor="middle">{xticks.labels?.[k] ?? fmt(v)}</text>
            </g>
          ))}
          {yticks.at.map((v, k) => (
            <g key={`y${k}`}>
              <line x1={plot.x - 4} x2={plot.x} y1={Y(v)} y2={Y(v)} stroke="#334155" />
              <text x={plot.x - 6} y={Y(v) + 3.5} textAnchor="end">{yticks.labels?.[k] ?? fmt(v)}</text>
            </g>
          ))}
          {ax.xlabel && <text x={plot.x + plot.w / 2} y={plot.y + plot.h + 30} textAnchor="middle" fontSize={11}>{ax.xlabel}</text>}
          {ax.ylabel && <text transform={`translate(${box.x + 11} ${plot.y + plot.h / 2}) rotate(-90)`} textAnchor="middle" fontSize={11}>{ax.ylabel}</text>}
        </g>
      )}
      {ax.title && <text x={plot.x + plot.w / 2} y={plot.y - 8} textAnchor="middle" fontSize={12} fontWeight={600} fill="#0f172a">{ax.title}</text>}
      {legendItems.length > 0 && (() => {
        const w = Math.min(plot.w - 8, 18 + Math.max(...legendItems.map((a) => ('label' in a && a.label ? a.label.length : 0))) * 6.2 + 10)
        const h = legendItems.length * 15 + 6
        const left = /left/.test(ax.legend!.loc) ? plot.x + 6 : plot.x + plot.w - w - 6
        const top = /lower/.test(ax.legend!.loc) ? plot.y + plot.h - h - 6 : plot.y + 6
        return (
          <g fontSize={10}>
            <rect x={left} y={top} width={w} height={h} rx={3} fill="#ffffff" fillOpacity={0.9} stroke="#cbd5e1" />
            {legendItems.map((a, k) => {
              const color = a.kind === 'scatter' ? a.colors[0] : a.kind === 'bar' ? a.colors[0] : 'color' in a ? a.color : '#000'
              const y = top + 11 + k * 15
              return (
                <g key={k}>
                  {a.kind === 'line' || a.kind === 'hline' || a.kind === 'vline' ? <line x1={left + 5} x2={left + 19} y1={y - 3} y2={y - 3} stroke={color} strokeWidth={2} strokeDasharray={a.dash ?? undefined} /> : a.kind === 'scatter' ? <circle cx={left + 12} cy={y - 3} r={4} fill={color} /> : <rect x={left + 6} y={y - 8} width={12} height={9} fill={color} />}
                  <text x={left + 23} y={y} fill="#1e293b">{'label' in a ? a.label : ''}</text>
                </g>
              )
            })}
          </g>
        )
      })()}
      {ax.colorbar && (() => {
        const cb = ax.colorbar
        const x = plot.x + plot.w + 12
        const steps = 32
        return (
          <g fontSize={9} fill="#334155">
            {Array.from({ length: steps }, (_, k) => <rect key={k} x={x} y={plot.y + (plot.h * (steps - 1 - k)) / steps} width={10} height={plot.h / steps + 0.5} fill={cmapColor(cb.cmap, k / (steps - 1))} />)}
            <rect x={x} y={plot.y} width={10} height={plot.h} fill="none" stroke="#334155" />
            {niceTicks(cb.vmin, cb.vmax, 4).filter((v) => v >= cb.vmin && v <= cb.vmax).map((v) => (
              <text key={v} x={x + 14} y={plot.y + plot.h - ((v - cb.vmin) / (cb.vmax - cb.vmin || 1)) * plot.h + 3}>{fmt(v)}</text>
            ))}
          </g>
        )
      })()}
    </g>
  )
}

let figId = 0

/** One figure; focus = { ax, artist } dims the other artists of that Axes */
export const FigureView = ({ fig, focus }: { fig: FigureSpec; focus?: { ax: number; artist: number | null } | null }) => {
  const { t } = useTranslation()
  const [pixel, setPixel] = useState<PixelInfo | null>(null)
  const [clip] = useState(() => `fig${++figId}`)
  useEffect(() => setPixel(null), [fig])
  const W = fig.size[0] * PX
  const H = fig.size[1] * PX
  const top = fig.suptitle ? 24 : 0
  return (
    <figure className="space-y-1">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg border border-slate-200 bg-white" role="img" aria-label={t('playground.view.figure_label', { n: fig.axes.length })}>
        {fig.suptitle && <text x={W / 2} y={17} textAnchor="middle" fontSize={14} fontWeight={700} fill="#0f172a">{fig.suptitle}</text>}
        {fig.axes.map((ax, k) => {
          const { rows, cols, index } = ax.pos
          const r = Math.floor((index - 1) / cols)
          const c = (index - 1) % cols
          const box = { x: (c * W) / cols, y: top + (r * (H - top)) / rows, w: W / cols, h: (H - top) / rows }
          const f = focus && focus.ax === k ? focus.artist : focus ? -1 : undefined
          return <AxesView key={k} ax={ax} box={box} focus={f} onPixel={setPixel} clip={clip} />
        })}
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
