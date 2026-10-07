/* eslint-disable react-refresh/only-export-components -- explainer kit: components plus data and helpers */
import { useTranslation } from 'react-i18next'
import { eseg, lerp, lerp2, seg, window01 } from '../../../lib/explainer'
import { XOR_POINTS, classify, hidden, linear, orNandAnd, sigmoid, sweepLine } from '../../../lib/xorWarp'
import type { Vec2 } from '../../../lib/xorWarp'
import { clipLine, halfPlane } from '../../../lib/geom2d'
import type { Episode } from '../Player'
import { At, C, DrawLine, Grid, Svg, Tex, TitleCard } from '../stage'

// "Why XOR needs a hidden layer" — lecture 6; asked in 2022S, 2023S, 2025S midterms and the 2023/2024 finals.

type Pt = [number, number]
const D = (o: Pt, s: number) => (v: Vec2): Pt => [o[0] + s * v[0], o[1] - s * v[1]]
const IN = D([410, 690], 280) // input space [-0.5, 1.5]²
const LIN = D([470, 360], 40) // z = 3(Wx + c)
const HID = D([300, 710], 520) // hidden space [0, 1]²

/** Where input point x is drawn at warp state (a: linear 0→1, then b: squash 0→1). */
const warp = (x: Vec2, a: number, b: number): Pt => {
  const pin = IN(x)
  const plin = LIN(linear(x))
  if (b <= 0) return lerp2(pin, plin, a)
  return lerp2(plin, HID(hidden(x)), b)
}

const colorOf = (t: 0 | 1) => (t ? C.blue : C.red)

function Dot({ at, t, r = 15, ring }: { at: Pt; t: 0 | 1; r?: number; ring?: string }) {
  return (
    <g>
      {ring && <circle cx={at[0]} cy={at[1]} r={r + 9} fill="none" stroke={ring} strokeWidth={4} />}
      <circle cx={at[0]} cy={at[1]} r={r} fill={colorOf(t)} stroke={C.bg} strokeWidth={3} />
    </g>
  )
}

/** Axes of the input plane with the unit square marked. */
function InputAxes({ o = 1 }: { o?: number }) {
  const [x0, y0] = IN([-0.5, 0])
  const [x1] = IN([1.5, 0])
  const [ax, ay0] = IN([0, -0.5])
  const [, ay1] = IN([0, 1.5])
  return (
    <g opacity={o}>
      <line x1={x0} y1={y0} x2={x1} y2={y0} stroke={C.axis} strokeWidth={2} />
      <line x1={ax} y1={ay0} x2={ax} y2={ay1} stroke={C.axis} strokeWidth={2} />
      {[1].map((v) => (
        <g key={v}>
          <line x1={IN([v, 0])[0]} y1={y0 - 8} x2={IN([v, 0])[0]} y2={y0 + 8} stroke={C.axis} strokeWidth={2} />
          <line x1={ax - 8} y1={IN([0, v])[1]} x2={ax + 8} y2={IN([0, v])[1]} stroke={C.axis} strokeWidth={2} />
        </g>
      ))}
    </g>
  )
}

const axisLabels = (o: number) => (
  <>
    <Tex f="x_1" x={IN([1.5, 0])[0] + 26} y={IN([0, 0])[1]} size={34} color={C.muted} o={o} />
    <Tex f="x_2" x={IN([0, 0])[0]} y={IN([0, 1.5])[1] - 30} size={34} color={C.muted} o={o} />
  </>
)

/** Warpable grid: lines x = c and y = c over the input window, sampled densely. */
const GRID_LINES: Vec2[][] = (() => {
  const vals = Array.from({ length: 9 }, (_, i) => -0.5 + i * 0.25)
  const n = 48
  const span = (k: number) => -0.5 + (2 * k) / n
  return [
    ...vals.map((c) => Array.from({ length: n + 1 }, (_, k): Vec2 => [c, span(k)])),
    ...vals.map((c) => Array.from({ length: n + 1 }, (_, k): Vec2 => [span(k), c])),
  ]
})()

function WarpGrid({ a, b, o = 1 }: { a: number; b: number; o?: number }) {
  return (
    <g opacity={o} fill="none">
      {GRID_LINES.map((line, i) => {
        const unit = [0, 1].includes(line[0][0]) || [0, 1].includes(line[0][1])
        return <polyline key={i} points={line.map((x) => warp(x, a, b).join(',')).join(' ')} stroke={unit ? '#4b5b78' : C.grid} strokeWidth={unit ? 2.5 : 1.5} />
      })}
    </g>
  )
}

/** The 2-2-1 network, drawn small. */
function Network({ x, y, s = 1, o = 1, hl }: { x: number; y: number; s?: number; o?: number; hl?: 'hidden' | 'out' }) {
  const n = { x1: [0, 0], x2: [0, 160], h1: [190, 0], h2: [190, 160], y: [380, 80] } as const
  const P = (k: keyof typeof n): Pt => [x + n[k][0] * s, y + n[k][1] * s]
  const edges: [keyof typeof n, keyof typeof n][] = [['x1', 'h1'], ['x2', 'h1'], ['x1', 'h2'], ['x2', 'h2'], ['h1', 'y'], ['h2', 'y']]
  return (
    <g opacity={o}>
      {edges.map(([a, b]) => <line key={a + b} x1={P(a)[0]} y1={P(a)[1]} x2={P(b)[0]} y2={P(b)[1]} stroke={C.axis} strokeWidth={2.5} />)}
      {(Object.keys(n) as (keyof typeof n)[]).map((k) => {
        const lit = (hl === 'hidden' && k.startsWith('h')) || (hl === 'out' && k === 'y')
        return <circle key={k} cx={P(k)[0]} cy={P(k)[1]} r={30 * s} fill={C.bg} stroke={lit ? C.yellow : k.startsWith('x') ? C.muted : C.blue} strokeWidth={lit ? 5 : 3} />
      })}
    </g>
  )
}

const netLabels = (x: number, y: number, s: number, o: number) => {
  const L: [string, number, number][] = [['x_1', 0, 0], ['x_2', 0, 160], ['h_1', 190, 0], ['h_2', 190, 160], ['y', 380, 80]]
  return L.map(([f, dx, dy]) => <Tex key={f} f={f} x={x + dx * s} y={y + dy * s} size={30 * s} o={o} />)
}

function Points({ a = 0, b = 0, o = 1, rings }: { a?: number; b?: number; o?: number; rings?: (string | undefined)[] }) {
  return (
    <g opacity={o}>
      {XOR_POINTS.map((p, i) => <Dot key={i} at={warp(p.x, a, b)} t={p.t} ring={rings?.[i]} />)}
    </g>
  )
}

const clipIn = (w1: number, w2: number, b: number): [Pt, Pt] | null => {
  const seg2 = clipLine(w1, w2, b, -0.5, 1.5)
  return seg2 ? [IN(seg2[0]), IN(seg2[1])] : null
}
const shadeIn = (w1: number, w2: number, b: number): string => halfPlane(w1, w2, b, -0.5, 1.5).map((v) => IN(v).join(',')).join(' ')

// ------------------------------------------------------------------ scenes

function Title({ p }: { p: number }) {
  const { t } = useTranslation()
  return <TitleCard p={p} kicker={t('watch.xor.kicker')} title={t('watch.xor.title')} sub={t('watch.xor.sub')} />
}

function PointsScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const rows: [string, string, string, 0 | 1][] = [['0', '0', '0', 0], ['0', '1', '1', 1], ['1', '0', '1', 1], ['1', '1', '0', 0]]
  return (
    <>
      <Svg>
        <Grid o={0.35} />
        <InputAxes o={eseg(p, 0, 0.12)} />
        <Points o={eseg(p, 0.08, 0.2)} />
      </Svg>
      {axisLabels(eseg(p, 0, 0.12))}
      {XOR_POINTS.map((pt, i) => (
        <At key={i} x={IN(pt.x)[0] + (pt.x[0] ? 30 : -30)} y={IN(pt.x)[1] + 40} size={26} color={C.muted} o={eseg(p, 0.15, 0.25)} anchor={pt.x[0] ? 'l' : 'r'}>
          ({pt.x[0]}, {pt.x[1]})
        </At>
      ))}
      <At x={1180} y={260} size={30} color={C.muted} o={eseg(p, 0.25, 0.35)} className="font-bold">
        XOR
      </At>
      {rows.map(([a, b, y, tt], i) => (
        <At key={i} x={1180} y={330 + i * 70} size={38} o={eseg(p, 0.28 + i * 0.05, 0.36 + i * 0.05)} className="font-mono">
          <span style={{ color: C.muted }}>{a}  {b}  →  </span>
          <span style={{ color: colorOf(tt), fontWeight: 800 }}>{y}</span>
        </At>
      ))}
      <At x={1180} y={680} size={28} color={C.yellow} o={eseg(p, 0.6, 0.7)} w={520} style={{ textAlign: 'center' }}>
        {t('watch.xor.label.one_line')}
      </At>
    </>
  )
}

function LinesScene({ p }: { p: number }) {
  const { t } = useTranslation()
  // sweep: angle turns 1.5 revolutions, offset breathes
  const a = lerp(0.3, 0.3 + 3 * Math.PI, eseg(p, 0.05, 0.92))
  const d = 0.42 * Math.sin(a * 1.7)
  const { w, score } = sweepLine(a, d)
  let best = 0
  for (let k = 0; k <= 60; k++) {
    const q = lerp(0.05, p, k / 60)
    const aa = lerp(0.3, 0.3 + 3 * Math.PI, eseg(q, 0.05, 0.92))
    best = Math.max(best, sweepLine(aa, 0.42 * Math.sin(aa * 1.7)).score)
  }
  const ln = clipIn(...w)
  const rings = XOR_POINTS.map((pt) => ((w[0] * pt.x[0] + w[1] * pt.x[1] + w[2] > 0 ? 1 : 0) === pt.t ? undefined : C.yellow))
  return (
    <>
      <Svg>
        <Grid o={0.35} />
        <polygon points={shadeIn(...w)} fill={C.blue} opacity={0.12} />
        <InputAxes />
        {ln && <line x1={ln[0][0]} y1={ln[0][1]} x2={ln[1][0]} y2={ln[1][1]} stroke={C.text} strokeWidth={3.5} />}
        <Points rings={rings} />
      </Svg>
      {axisLabels(1)}
      <At x={1180} y={330} size={30} color={C.muted}>{t('watch.xor.label.this_line')}</At>
      <At x={1180} y={400} size={80} className="font-black font-mono" color={score === 3 ? C.yellow : C.text}>
        {score} / 4
      </At>
      <At x={1180} y={520} size={30} color={C.muted}>{t('watch.xor.label.best')}</At>
      <At x={1180} y={590} size={64} className="font-black font-mono" color={C.yellow}>
        {best} / 4
      </At>
      <At x={1180} y={720} size={26} color={C.muted} o={eseg(p, 0.2, 0.3)} w={560} style={{ textAlign: 'center' }}>
        {t('watch.xor.label.blue_side')}
      </At>
    </>
  )
}

function ProofScene({ p }: { p: number }) {
  const lines: [string, number, string][] = [
    ['y = \\mathrm{step}(w_1x_1 + w_2x_2 + b)', 0.02, C.text],
    ['(0,0)\\to 0:\\quad b \\le 0', 0.14, C.red],
    ['(1,0)\\to 1:\\quad w_1 + b > 0', 0.24, C.blue],
    ['(0,1)\\to 1:\\quad w_2 + b > 0', 0.32, C.blue],
    ['\\text{add:}\\quad w_1 + w_2 + b > -b \\ge 0', 0.46, C.yellow],
    ['(1,1)\\to 0 \\text{ needs }\\; w_1 + w_2 + b \\le 0', 0.62, C.red],
  ]
  return (
    <>
      <Svg>
        <Grid o={0.25} />
        <rect x={360} y={700} width={880} height={110} rx={18} fill={C.red} opacity={0.14 * eseg(p, 0.76, 0.84)} />
      </Svg>
      {lines.map(([f, at, col], i) => (
        <Tex key={i} f={f} x={800} y={120 + i * 92} size={42} color={col} o={eseg(p, at, at + 0.08)} />
      ))}
      <Tex f="\Rightarrow\ \text{contradiction: no } w_1, w_2, b \text{ exist}" x={800} y={755} size={44} color={C.red} o={eseg(p, 0.76, 0.84)} scale={lerp(0.9, 1, eseg(p, 0.76, 0.86))} />
    </>
  )
}

function IdeaScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const o = eseg(p, 0.25, 0.4)
  return (
    <>
      <Svg>
        <Grid o={0.3} />
        <Network x={560} y={340} s={1.15} o={o} hl={p > 0.55 ? 'hidden' : undefined} />
      </Svg>
      {netLabels(560, 340, 1.15, o)}
      <At x={800} y={170} size={46} className="font-black" o={eseg(p, 0, 0.15)} w={1200} style={{ textAlign: 'center' }}>
        {t('watch.xor.label.dont_bend')}
      </At>
      <At x={800} y={740} size={30} color={C.yellow} o={eseg(p, 0.55, 0.7)} w={1100} style={{ textAlign: 'center' }}>
        {t('watch.xor.label.hidden_moves')}
      </At>
    </>
  )
}

function StretchScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const a = eseg(p, 0.25, 0.8)
  return (
    <>
      <Svg>
        <WarpGrid a={a} b={0} />
        <Points a={a} />
      </Svg>
      {axisLabels(1 - eseg(p, 0.2, 0.3))}
      <At x={1300} y={150} size={28} color={C.muted} className="font-bold uppercase tracking-[0.15em]">{t('watch.xor.label.step1')}</At>
      <Tex f="z = 3\,(W x + c)" x={1300} y={230} size={44} />
      <Tex f="W = \begin{bmatrix} 1.5 & 1 \\ 1 & 1.5 \end{bmatrix},\; c = \begin{bmatrix} -0.5 \\ -2 \end{bmatrix}" x={1300} y={350} size={30} o={eseg(p, 0.08, 0.18)} />
      <At x={1300} y={500} size={28} color={C.muted} w={500} o={eseg(p, 0.5, 0.6)} style={{ textAlign: 'center' }}>
        {t('watch.xor.label.lines_stay')}
      </At>
      <At x={1300} y={640} size={28} color={C.yellow} w={500} o={eseg(p, 0.82, 0.92)} style={{ textAlign: 'center' }}>
        {t('watch.xor.label.still_stuck')}
      </At>
    </>
  )
}

function SquashScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const b = eseg(p, 0.22, 0.75)
  // sigmoid inset
  const sx = (z: number) => 1170 + z * 30
  const sy = (v: number) => 640 - v * 200
  const curve = Array.from({ length: 121 }, (_, k) => {
    const z = -6 + k * 0.1
    return `${sx(z)},${sy(sigmoid(z))}`
  }).join(' ')
  const [hx0, hy0] = HID([0, 0])
  const [hx1, hy1] = HID([1, 1])
  return (
    <>
      <Svg>
        <rect x={hx0} y={hy1} width={hx1 - hx0} height={hy0 - hy1} fill="none" stroke={C.yellow} strokeWidth={3} strokeDasharray="10 8" opacity={eseg(p, 0.6, 0.75)} />
        <WarpGrid a={1} b={b} />
        <Points a={1} b={b} />
        <g opacity={eseg(p, 0.05, 0.15)}>
          <line x1={sx(-6)} y1={sy(0)} x2={sx(6)} y2={sy(0)} stroke={C.axis} strokeWidth={2} />
          <line x1={sx(0)} y1={sy(0)} x2={sx(0)} y2={sy(1.1)} stroke={C.axis} strokeWidth={2} />
          <line x1={sx(-6)} y1={sy(1)} x2={sx(6)} y2={sy(1)} stroke={C.axis} strokeWidth={1.5} strokeDasharray="6 6" />
          <polyline points={curve} fill="none" stroke={C.purple} strokeWidth={4} />
        </g>
      </Svg>
      <At x={1300} y={150} size={28} color={C.muted} className="font-bold uppercase tracking-[0.15em]">{t('watch.xor.label.step2')}</At>
      <Tex f="h = \sigma(z) = \dfrac{1}{1 + e^{-z}}" x={1300} y={250} size={40} />
      <Tex f="\sigma" x={sx(-5.6)} y={sy(1) - 26} size={30} color={C.purple} o={eseg(p, 0.05, 0.15)} />
      <At x={1300} y={740} size={28} color={C.yellow} w={520} o={eseg(p, 0.62, 0.72)} style={{ textAlign: 'center' }}>
        {t('watch.xor.label.into_square')}
      </At>
      <Tex f="h_1" x={hx1 + 30} y={hy0 + 6} size={32} color={C.muted} o={eseg(p, 0.7, 0.8)} />
      <Tex f="h_2" x={hx0 - 4} y={hy1 - 30} size={32} color={C.muted} o={eseg(p, 0.7, 0.8)} />
    </>
  )
}

function SeparateScene({ p }: { p: number }) {
  const { t } = useTranslation()
  // line h1 − h2 = 0.4 inside [0,1]²: from (0.4, 0) to (1, 0.6)
  const A = HID([0.4, 0])
  const B = HID([1, 0.6])
  const draw = eseg(p, 0.12, 0.32)
  const shade = eseg(p, 0.3, 0.42)
  const [hx0, hy0] = HID([0, 0])
  const [hx1, hy1] = HID([1, 1])
  const tri = [HID([0.4, 0]), HID([1, 0]), HID([1, 0.6])].map((q) => q.join(',')).join(' ')
  return (
    <>
      <Svg>
        <WarpGrid a={1} b={1} o={0.7} />
        <rect x={hx0} y={hy1} width={hx1 - hx0} height={hy0 - hy1} fill="none" stroke={C.axis} strokeWidth={2} />
        <polygon points={tri} fill={C.blue} opacity={0.18 * shade} />
        <DrawLine x1={A[0]} y1={A[1]} x2={B[0]} y2={B[1]} t={draw} color={C.yellow} width={5} />
        <Points a={1} b={1} />
      </Svg>
      {XOR_POINTS.map((pt, i) => {
        const h = hidden(pt.x)
        const at = HID(h)
        return (
          <At key={i} x={i === 0 ? at[0] : at[0] + 28} y={i === 0 ? at[1] + 40 : at[1]} size={24} color={colorOf(pt.t)} o={eseg(p, 0.02, 0.1)} anchor={i === 0 ? 't' : 'l'}>
            ({pt.x[0]},{pt.x[1]}) → ({h[0].toFixed(2)}, {h[1].toFixed(2)})
          </At>
        )
      })}
      <At x={1300} y={150} size={28} color={C.muted} className="font-bold uppercase tracking-[0.15em]">{t('watch.xor.label.step3')}</At>
      <Tex f="y = \mathrm{step}(h_1 - h_2 - 0.4)" x={1300} y={240} size={38} color={C.yellow} o={eseg(p, 0.12, 0.22)} />
      <At x={1300} y={380} size={30} w={520} o={eseg(p, 0.45, 0.55)} style={{ textAlign: 'center' }}>
        {t('watch.xor.label.one_line_now')}
      </At>
      <Tex f="h_1 \approx \mathrm{OR}(x_1, x_2)" x={1300} y={540} size={34} color={C.blue} o={eseg(p, 0.62, 0.7)} />
      <Tex f="h_2 \approx \mathrm{AND}(x_1, x_2)" x={1300} y={610} size={34} color={C.red} o={eseg(p, 0.7, 0.78)} />
      <Tex f="y = h_1 \wedge \neg h_2" x={1300} y={690} size={36} color={C.yellow} o={eseg(p, 0.78, 0.86)} />
    </>
  )
}

const CELLS = 40
function PullbackScene({ p }: { p: number }) {
  const { t } = useTranslation()
  // undo the squash, then undo the stretch
  const b = 1 - eseg(p, 0.05, 0.35)
  const a = b > 0 ? 1 : 1 - eseg(p, 0.35, 0.55)
  const reveal = eseg(p, 0.55, 0.85)
  const size = 2 / CELLS
  const cells = []
  if (reveal > 0) {
    for (let i = 0; i < CELLS; i++)
      for (let j = 0; j < CELLS; j++) {
        const x: Vec2 = [-0.5 + (i + 0.5) * size, -0.5 + (j + 0.5) * size]
        if ((i + (CELLS - j)) / (2 * CELLS) > reveal) continue
        const [px, py] = IN([x[0] - size / 2, x[1] + size / 2])
        cells.push(<rect key={i * CELLS + j} x={px} y={py} width={size * 280 + 0.5} height={size * 280 + 0.5} fill={classify(x) ? C.blue : C.red} opacity={0.2} />)
      }
  }
  return (
    <>
      <Svg>
        {cells}
        <WarpGrid a={a} b={b} o={0.8} />
        <InputAxes o={eseg(p, 0.5, 0.6)} />
        <Points a={a} b={b} />
      </Svg>
      {axisLabels(eseg(p, 0.5, 0.6))}
      <At x={1220} y={330} size={34} className="font-black" w={560} o={eseg(p, 0.55, 0.65)} style={{ textAlign: 'center' }}>
        {t('watch.xor.label.pulled_back')}
      </At>
      <At x={1220} y={480} size={28} color={C.muted} w={560} o={eseg(p, 0.7, 0.8)} style={{ textAlign: 'center' }}>
        {t('watch.xor.label.band')}
      </At>
    </>
  )
}

function ExamScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const rows = XOR_POINTS.map((pt) => ({ x: pt.x, ...orNandAnd(pt.x) }))
  return (
    <>
      <Svg>
        <Grid o={0.25} />
      </Svg>
      <At x={800} y={110} size={30} color={C.yellow} className="font-bold uppercase tracking-[0.15em]" o={eseg(p, 0, 0.08)}>
        {t('watch.xor.label.exam')}
      </At>
      <Tex f="h_1 = \mathrm{step}(x_1 + x_2 - 0.5)\quad \text{OR}" x={520} y={250} size={34} color={C.blue} o={eseg(p, 0.08, 0.16)} />
      <Tex f="h_2 = \mathrm{step}(-x_1 - x_2 + 1.5)\quad \text{NAND}" x={520} y={330} size={34} color={C.red} o={eseg(p, 0.16, 0.24)} />
      <Tex f="y = \mathrm{step}(h_1 + h_2 - 1.5)\quad \text{AND}" x={520} y={410} size={34} color={C.yellow} o={eseg(p, 0.24, 0.32)} />
      <At x={1260} y={210} size={28} color={C.muted} className="font-mono" o={eseg(p, 0.32, 0.4)}>
        x₁ x₂ │ h₁ h₂ │ y
      </At>
      {rows.map((r, i) => (
        <At key={i} x={1260} y={270 + i * 56} size={30} className="font-mono" o={eseg(p, 0.36 + i * 0.04, 0.42 + i * 0.04)}>
          <span style={{ color: C.muted }}>{r.x[0]}  {r.x[1]} │ </span>
          {r.h1}  {r.h2} <span style={{ color: C.muted }}>│</span> <b style={{ color: colorOf(r.y) }}>{r.y}</b>
        </At>
      ))}
      <At x={800} y={620} size={30} w={1300} o={eseg(p, 0.6, 0.7)} style={{ textAlign: 'center' }}>
        {t('watch.xor.label.trap_linear')}
      </At>
      <Tex f="W_2(W_1x + c_1) + c_2 = (W_2W_1)\,x + (W_2c_1 + c_2)" x={800} y={720} size={34} color={C.red} o={eseg(p, 0.7, 0.8)} />
    </>
  )
}

function RecapScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.3} />
      </Svg>
      <At x={800} y={150} size={30} color={C.blue} className="font-bold uppercase tracking-[0.15em]" o={eseg(p, 0, 0.08)}>
        {t('watch.ui.recap')}
      </At>
      {[1, 2, 3].map((k) => (
        <At key={k} x={240} y={240 + k * 120} size={38} anchor="l" w={1150} o={window01(p, 0.08 + k * 0.12, 1.2)}>
          <span style={{ color: C.yellow, fontWeight: 900 }}>{k}. </span>
          {t(`watch.xor.recap.${k}`)}
        </At>
      ))}
      <At x={800} y={800} size={28} color={C.muted} o={seg(p, 0.75, 0.85)}>
        {t('watch.ui.try_it')}
      </At>
    </>
  )
}

export const xor: Episode = {
  id: 'xor',
  scenes: [
    { id: 'title', dur: 5000, cues: [0], render: (p) => <Title p={p} /> },
    { id: 'points', dur: 13000, cues: [0, 0.3, 0.6], render: (p) => <PointsScene p={p} /> },
    { id: 'lines', dur: 17000, cues: [0, 0.35, 0.7], ponder: true, render: (p) => <LinesScene p={p} /> },
    { id: 'proof', dur: 17000, cues: [0, 0.14, 0.46, 0.76], render: (p) => <ProofScene p={p} /> },
    { id: 'idea', dur: 11000, cues: [0, 0.5], render: (p) => <IdeaScene p={p} /> },
    { id: 'stretch', dur: 15000, cues: [0, 0.25, 0.55, 0.82], render: (p) => <StretchScene p={p} /> },
    { id: 'squash', dur: 15000, cues: [0, 0.22, 0.62], render: (p) => <SquashScene p={p} /> },
    { id: 'separate', dur: 17000, cues: [0, 0.12, 0.45, 0.62], ponder: true, render: (p) => <SeparateScene p={p} /> },
    { id: 'pullback', dur: 15000, cues: [0, 0.35, 0.6], render: (p) => <PullbackScene p={p} /> },
    { id: 'exam', dur: 19000, cues: [0, 0.32, 0.6], render: (p) => <ExamScene p={p} /> },
    { id: 'recap', dur: 13000, cues: [0, 0.45], render: (p) => <RecapScene p={p} /> },
  ],
}
