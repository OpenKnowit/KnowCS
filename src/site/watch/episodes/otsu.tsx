/* eslint-disable react-refresh/only-export-components -- episode data plus its scene components */
import { useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { eseg, lerp, seg } from '../../../lib/explainer'
import { contrastStretch, histogram, otsuIterations, threshold } from '../../../lib/otsu'
import { mulberry32 } from '../../../lib/crossval'
import type { Episode } from '../Player'
import { At, C, Grid, Svg, Tex, TitleCard } from '../stage'

// "Histograms and Otsu's threshold" — lecture 7; point operations and the iterative threshold are in Final 2022 and 2024.

const W = 48
const H = 36
/** Coins on a table: two clear peaks, reproducible. */
const PIX: number[] = (() => {
  const r = mulberry32(5)
  const discs = [[12, 12, 7], [31, 15, 8], [20, 27, 6], [40, 29, 5]]
  const out: number[] = []
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const on = discs.some(([cx, cy, rr]) => (x - cx) ** 2 + (y - cy) ** 2 < rr * rr)
      out.push(Math.round(Math.min(255, Math.max(0, (on ? 180 : 75) + (r() - 0.5) * 60 + (on ? 0 : x / 3)))))
    }
  return out
})()
const BINS = 32
const HIST = histogram(PIX, BINS)
const MAXC = Math.max(...HIST)
const ITERS = otsuIterations(PIX)
const T_FINAL = ITERS[ITERS.length - 1].next
const DIM = PIX.map((v) => Math.round(90 + (v / 255) * 70)) // squeezed into 90…160 for the stretch scene
const STRETCHED = contrastStretch(DIM)

// stage geometry
const IMG = { x: 90, y: 230, cell: 10 }
const HG = { x: 760, base: 760, w: 760, h: 480 }
const binOf = (v: number) => Math.min(BINS - 1, Math.floor((v / 256) * BINS))
const binX = (b: number) => HG.x + (b + 0.5) * (HG.w / BINS)
const vX = (v: number) => HG.x + (v / 256) * HG.w
const ROLE: { bin: number; k: number }[] = (() => {
  const seen = new Array(BINS).fill(0)
  return PIX.map((v) => {
    const bin = binOf(v)
    return { bin, k: seen[bin]++ }
  })
})()

/** Pixels on a full-stage canvas: `fly` moves them from the image into histogram stacks; `values` recolours them. */
function PixelCanvas({ fly, values, o = 1 }: { fly: number; values?: number[]; o?: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    c.width = 1600
    c.height = 900
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, 1600, 900)
    const vals = values ?? PIX
    const hpix = HG.h / MAXC
    PIX.forEach((_, i) => {
      const x0 = IMG.x + (i % W) * IMG.cell
      const y0 = IMG.y + Math.floor(i / W) * IMG.cell
      const { bin, k } = ROLE[i]
      // stagger: each pixel starts a little later than the previous row
      const t = Math.min(1, Math.max(0, (fly - (Math.floor(i / W) / H) * 0.4) / 0.6))
      const e = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2
      const x1 = binX(bin) - 6 + (k % 3) * 4
      const y1 = HG.base - (k + 0.5) * hpix
      const x = lerp(x0, x1, e)
      const y = lerp(y0, y1, e)
      const g = vals[i]
      ctx.fillStyle = `rgb(${g},${g},${g})`
      const s = lerp(IMG.cell - 1, Math.max(2, hpix), e)
      ctx.fillRect(x, y - s / 2, lerp(IMG.cell - 1, 4, e), s)
    })
  })
  return <canvas ref={ref} aria-hidden className="absolute inset-0 h-full w-full" style={{ opacity: o }} />
}

function Bars({ h, color = C.blue, o = 1, split }: { h: number[]; color?: string; o?: number; split?: number }) {
  const max = Math.max(...h, 1)
  const bw = HG.w / h.length
  return (
    <g opacity={o}>
      {h.map((n, b) => (
        <rect key={b} x={HG.x + b * bw + 1} y={HG.base - (n / max) * HG.h} width={bw - 2} height={(n / max) * HG.h} fill={split !== undefined ? ((b + 0.5) * (256 / h.length) > split ? C.yellow : C.blue) : color} opacity={0.8} />
      ))}
    </g>
  )
}

function Axis() {
  return (
    <g>
      <line x1={HG.x} y1={HG.base} x2={HG.x + HG.w} y2={HG.base} stroke={C.axis} strokeWidth={2} />
      {[0, 64, 128, 192, 255].map((v) => <text key={v} x={vX(v)} y={HG.base + 30} textAnchor="middle" fill={C.muted} fontSize={22}>{v}</text>)}
    </g>
  )
}

function Title({ p }: { p: number }) {
  const { t } = useTranslation()
  return <TitleCard p={p} kicker={t('watch.otsu.kicker')} title={t('watch.otsu.title')} sub={t('watch.otsu.sub')} />
}

function HistScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        <Axis />
      </Svg>
      <PixelCanvas fly={seg(p, 0.25, 0.9)} />
      <At x={330} y={180} size={24} color={C.muted}>{t('watch.otsu.label.image', { n: W * H })}</At>
      <At x={HG.x + HG.w / 2} y={200} size={28} w={700} o={eseg(p, 0.2, 0.3)} style={{ textAlign: 'center' }}>{t('watch.otsu.label.sort_by_grey')}</At>
      <At x={HG.x + HG.w / 2} y={HG.base + 70} size={22} color={C.muted}>{t('watch.otsu.label.grey_level')}</At>
    </>
  )
}

function StretchScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const s = eseg(p, 0.3, 0.7)
  const vals = DIM.map((v, i) => Math.round(lerp(v, STRETCHED[i], s)))
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        <Axis />
        <Bars h={histogram(vals, BINS)} color={C.teal} />
      </Svg>
      <PixelCanvas fly={0} values={vals} />
      <Tex f="I_{\text{new}} = \dfrac{I - I_{\min}}{I_{\max} - I_{\min}} \times 255" x={HG.x + HG.w / 2} y={170} size={36} o={eseg(p, 0.05, 0.15)} />
      <At x={330} y={650} size={24} w={520} color={C.muted} o={eseg(p, 0.7, 0.8)} style={{ textAlign: 'center' }}>{t('watch.otsu.label.same_shape')}</At>
    </>
  )
}

function ThresholdScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const T = lerp(60, 200, eseg(p, 0.1, 0.7))
  const out = threshold(PIX, T)
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        <Axis />
        <Bars h={HIST} split={T} />
        <line x1={vX(T)} y1={HG.base} x2={vX(T)} y2={HG.base - HG.h - 20} stroke={C.red} strokeWidth={3} />
      </Svg>
      <PixelCanvas fly={0} values={out} />
      <At x={vX(T)} y={HG.base - HG.h - 40} size={26} className="font-mono" color={C.red}>T = {T.toFixed(0)}</At>
      <At x={330} y={650} size={24} w={520} o={eseg(p, 0.2, 0.3)} style={{ textAlign: 'center' }}>{t('watch.otsu.label.binary')}</At>
      <At x={330} y={740} size={24} w={520} color={C.yellow} o={eseg(p, 0.75, 0.85)} style={{ textAlign: 'center' }}>{t('watch.otsu.label.which_T')}</At>
    </>
  )
}

function OtsuScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const n = ITERS.length
  const k = Math.min(n - 1, Math.floor(seg(p, 0.05, 0.8) * n))
  const it = ITERS[k]
  const f = seg(p, 0.05, 0.8) * n - k // progress within this iteration
  const Tshow = lerp(it.T, it.next, seg(f, 0.55, 0.95))
  const done = p > 0.82
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        <Axis />
        <Bars h={HIST} split={Tshow} o={0.75} />
        <line x1={vX(Tshow)} y1={HG.base} x2={vX(Tshow)} y2={HG.base - HG.h - 20} stroke={C.red} strokeWidth={3} />
        {f > 0.2 && <line x1={vX(it.mu1)} y1={HG.base} x2={vX(it.mu1)} y2={HG.base - HG.h} stroke={C.blue} strokeWidth={2} strokeDasharray="6 5" />}
        {f > 0.2 && <line x1={vX(it.mu2)} y1={HG.base} x2={vX(it.mu2)} y2={HG.base - HG.h} stroke={C.yellow} strokeWidth={2} strokeDasharray="6 5" />}
      </Svg>
      <PixelCanvas fly={0} values={done ? threshold(PIX, T_FINAL) : PIX} />
      <At x={vX(Tshow)} y={HG.base - HG.h - 40} size={24} className="font-mono" color={C.red}>T = {Tshow.toFixed(1)}</At>
      {f > 0.2 && <At x={vX(it.mu1)} y={HG.base - HG.h + 20} size={22} className="font-mono" color={C.blue} anchor="r">μ1 {it.mu1.toFixed(1)} </At>}
      {f > 0.2 && <At x={vX(it.mu2)} y={HG.base - HG.h + 20} size={22} className="font-mono" color={C.yellow} anchor="l"> μ2 {it.mu2.toFixed(1)}</At>}
      <At x={330} y={650} size={26} className="font-mono">{t('watch.otsu.label.iteration', { k: k + 1, n })}</At>
      <Tex f="T \leftarrow \dfrac{\mu_1 + \mu_2}{2}" x={330} y={740} size={36} color={C.red} />
    </>
  )
}

// Final 2024 Q5(b): this 3×3 image, T₀ = 100 (official answer: μ1 = 21, μ2 = 128, T = 74.5)
const SMALL = [2, 4, 8, 16, 32, 64, 128, 128, 128]
const SMALL_IT = otsuIterations(SMALL, 100)
function TableScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const rows = Math.floor(seg(p, 0.15, 0.75) * (SMALL_IT.length + 0.99))
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        {SMALL.map((v, i) => (
          <g key={i}>
            <rect x={160 + (i % 3) * 90} y={260 + Math.floor(i / 3) * 90} width={84} height={84} rx={8} fill={`rgb(${v},${v},${v})`} stroke={C.axis} />
            <text x={202 + (i % 3) * 90} y={310 + Math.floor(i / 3) * 90} textAnchor="middle" fill={v > 100 ? C.bg : C.text} fontSize={26} fontFamily="ui-monospace, monospace">{v}</text>
          </g>
        ))}
      </Svg>
      <At x={295} y={200} size={24} color={C.muted}>T₀ = 100</At>
      <At x={1050} y={220} size={24} className="font-mono" color={C.muted} style={{ whiteSpace: 'pre' }}>{'#    T       μ1       μ2    (μ1+μ2)/2'}</At>
      {SMALL_IT.slice(0, rows).map((r, i) => (
        <At key={i} x={1050} y={290 + i * 60} size={26} className="font-mono" style={{ whiteSpace: 'pre' }}>
          {`${i + 1}  ${r.T.toFixed(2).padStart(6)}  ${r.mu1.toFixed(2).padStart(7)}  ${r.mu2.toFixed(2).padStart(7)}  `}
          <b style={{ color: C.yellow }}>{r.next.toFixed(2).padStart(8)}</b>
        </At>
      ))}
      <At x={1050} y={290 + SMALL_IT.length * 60 + 30} size={26} color={C.teal} o={eseg(p, 0.8, 0.9)}>{t('watch.otsu.label.stable', { T: SMALL_IT[SMALL_IT.length - 1].next.toFixed(2) })}</At>
    </>
  )
}

function FlipScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const flipped = PIX.map((_, i) => PIX[Math.floor(i / W) * W + (W - 1 - (i % W))])
  const showFlip = p > 0.3
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        <Axis />
        <Bars h={histogram(showFlip ? flipped : PIX, BINS)} />
      </Svg>
      <PixelCanvas fly={0} values={showFlip ? flipped : PIX} />
      <At x={330} y={650} size={26} w={520} o={eseg(p, 0.35, 0.45)} style={{ textAlign: 'center' }}>{t('watch.otsu.label.flip_same')}</At>
      <At x={HG.x + HG.w / 2} y={180} size={26} w={700} color={C.yellow} o={eseg(p, 0.55, 0.65)} style={{ textAlign: 'center' }}>{t('watch.otsu.label.point_vs')}</At>
    </>
  )
}

function Recap({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.3} />
      </Svg>
      <At x={800} y={150} size={30} color={C.blue} className="font-bold uppercase tracking-[0.15em]" o={eseg(p, 0, 0.08)}>{t('watch.ui.recap')}</At>
      {[1, 2, 3].map((k) => (
        <At key={k} x={220} y={240 + k * 120} size={36} anchor="l" w={1180} o={eseg(p, 0.08 + k * 0.12, 0.16 + k * 0.12)}>
          <span style={{ color: C.yellow, fontWeight: 900 }}>{k}. </span>
          {t(`watch.otsu.recap.${k}`)}
        </At>
      ))}
      <At x={800} y={800} size={28} color={C.muted} o={eseg(p, 0.75, 0.85)}>{t('watch.ui.try_it')}</At>
    </>
  )
}

export const otsu: Episode = {
  id: 'otsu',
  scenes: [
    { id: 'title', dur: 5000, cues: [0], render: (p) => <Title p={p} /> },
    { id: 'hist', dur: 15000, cues: [0, 0.25, 0.7], render: (p) => <HistScene p={p} /> },
    { id: 'stretch', dur: 13000, cues: [0, 0.3, 0.7], render: (p) => <StretchScene p={p} /> },
    { id: 'threshold', dur: 13000, cues: [0, 0.2, 0.75], ponder: true, render: (p) => <ThresholdScene p={p} /> },
    { id: 'otsu', dur: 22000, cues: [0, 0.15, 0.5, 0.82], render: (p) => <OtsuScene p={p} /> },
    { id: 'table', dur: 16000, cues: [0, 0.15, 0.8], render: (p) => <TableScene p={p} /> },
    { id: 'flip', dur: 12000, cues: [0, 0.3, 0.55], render: (p) => <FlipScene p={p} /> },
    { id: 'recap', dur: 12000, cues: [0, 0.6], render: (p) => <Recap p={p} /> },
  ],
}
