/* eslint-disable react-refresh/only-export-components -- episode data plus its scene components */
import type { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { eseg, lerp, seg } from '../../../lib/explainer'
import { reflectX, translate, warpAffine } from '../../../lib/affine'
import type { Episode } from '../Player'
import { At, C, DrawLine, Grid, Svg, Tex, TitleCard } from '../stage'

// "One matrix moves every pixel" — lecture 7's affine transforms, what they keep, and the finals' questions on them.

const ROWS = 8
const COLS = 10
const X0 = COLS / 2
const Y0 = ROWS / 2
const IMG: number[][] = Array.from({ length: ROWS }, (_, y) =>
  Array.from({ length: COLS }, (_, x) => {
    if (x === 2 && y >= 1 && y <= 6) return 230
    if (y === 1 && x >= 3 && x <= 6) return 180
    if (y === 3 && x >= 3 && x <= 5) return 120
    if (x === 8 && y === 6) return 90
    return 40
  }),
)
const grey = (v: number) => `rgb(${v},${v},${v})`

interface Board {
  ox: number
  oy: number
  cs: number
}

/** The frame of the image (where warpAffine writes) plus pixel-index ticks. */
function Frame({ b, o = 1, ticks = true }: { b: Board; o?: number; ticks?: boolean }) {
  return (
    <g opacity={o}>
      <rect x={b.ox - 4} y={b.oy - 4} width={COLS * b.cs + 8} height={ROWS * b.cs + 8} fill="none" stroke={C.yellow} strokeWidth={2.5} strokeDasharray="10 7" rx={6} />
      {ticks && Array.from({ length: COLS }, (_, x) => <text key={`x${x}`} x={b.ox + (x + 0.5) * b.cs} y={b.oy - 14} textAnchor="middle" fill={C.muted} fontSize={17} fontFamily="ui-monospace, monospace">{x}</text>)}
      {ticks && Array.from({ length: ROWS }, (_, y) => <text key={`y${y}`} x={b.ox - 16} y={b.oy + (y + 0.5) * b.cs + 6} textAnchor="end" fill={C.muted} fontSize={17} fontFamily="ui-monospace, monospace">{y}</text>)}
    </g>
  )
}

/**
 * Every pixel drawn at map(x, y) (pixel units, continuous) and turned by `turn` degrees on screen.
 * Pixels whose final place is outside the frame fade by `cut`; `holes` (frame cells nothing lands on) appear as 0.
 */
function Pixels({ b, map, turn = 0, cut = 0, holes, mark }: { b: Board; map: (x: number, y: number) => [number, number]; turn?: number; cut?: number; holes?: boolean[][]; mark?: [number, number] }) {
  const out: ReactNode[] = []
  if (holes && cut > 0)
    holes.forEach((row, y) => row.forEach((h, x) => {
      if (h) out.push(<rect key={`h${x}-${y}`} x={b.ox + x * b.cs + 1} y={b.oy + y * b.cs + 1} width={b.cs - 2} height={b.cs - 2} fill="#000" stroke={C.axis} strokeDasharray="4 4" opacity={cut} />)
    }))
  IMG.forEach((row, y) => row.forEach((v, x) => {
    const [u, w] = map(x, y)
    const cx = b.ox + (u + 0.5) * b.cs
    const cy = b.oy + (w + 0.5) * b.cs
    const inside = u > -0.5 && u < COLS - 0.5 && w > -0.5 && w < ROWS - 0.5
    const on = mark && mark[0] === x && mark[1] === y
    out.push(
      <rect
        key={`${x}-${y}`}
        x={cx - b.cs / 2 + 1}
        y={cy - b.cs / 2 + 1}
        width={b.cs - 2}
        height={b.cs - 2}
        rx={3}
        fill={grey(v)}
        stroke={on ? C.red : 'none'}
        strokeWidth={on ? 4 : 0}
        opacity={inside ? 1 : 1 - 0.85 * cut}
        transform={turn ? `rotate(${turn} ${cx} ${cy})` : undefined}
      />,
    )
  }))
  return <g>{out}</g>
}

const holesOf = (M: Parameters<typeof warpAffine>[1]) => warpAffine(IMG, M).from.map((row) => row.map((s) => !s))

function Title({ p }: { p: number }) {
  const { t } = useTranslation()
  return <TitleCard p={p} kicker={t('watch.affine.kicker')} title={t('watch.affine.title')} sub={t('watch.affine.sub')} />
}

function CoordsScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const b: Board = { ox: 140, oy: 250, cs: 52 }
  const axes = eseg(p, 0.05, 0.25)
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        <Pixels b={b} map={(x, y) => [x, y]} mark={p > 0.15 ? [7, 1] : undefined} />
        <Frame b={b} />
        <DrawLine x1={b.ox} y1={b.oy - 50} x2={b.ox + COLS * b.cs + 40} y2={b.oy - 50} t={axes} color={C.blue} width={4} />
        <DrawLine x1={b.ox - 50} y1={b.oy} x2={b.ox - 50} y2={b.oy + ROWS * b.cs + 40} t={axes} color={C.green} width={4} />
      </Svg>
      <At x={b.ox + COLS * b.cs + 70} y={b.oy - 50} size={30} color={C.blue} className="font-mono font-black" o={axes}>x</At>
      <At x={b.ox - 50} y={b.oy + ROWS * b.cs + 70} size={30} color={C.green} className="font-mono font-black" o={axes}>y</At>
      <At x={1180} y={250} size={34} className="font-mono" color={C.red} o={eseg(p, 0.15, 0.25)}>P = (x, y) = (7, 1)</At>
      <At x={1180} y={330} size={26} color={C.muted} w={640} o={eseg(p, 0.3, 0.4)} style={{ textAlign: 'center' }}>{t('watch.affine.label.y_down')}</At>
      <Tex f="\begin{bmatrix} x' \\ y' \end{bmatrix} = \underbrace{\begin{bmatrix} a_{00} & a_{01} & b_{00} \\ a_{10} & a_{11} & b_{10} \end{bmatrix}}_{M} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}" x={1180} y={560} size={34} o={eseg(p, 0.62, 0.75)} />
      <At x={1180} y={760} size={24} color={C.yellow} w={640} o={eseg(p, 0.78, 0.88)} style={{ textAlign: 'center' }}>{t('watch.affine.label.the_one')}</At>
    </>
  )
}

function TranslateScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const b: Board = { ox: 140, oy: 250, cs: 52 }
  const a = eseg(p, 0.12, 0.4)
  const cut = eseg(p, 0.7, 0.82)
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        <Pixels b={b} map={(x, y) => [x + 2 * a, y + a]} cut={cut} holes={holesOf(translate(2, 1))} />
        <Frame b={b} />
      </Svg>
      <Tex f="x' = x + t_x \qquad y' = y + t_y" x={1180} y={240} size={38} o={eseg(p, 0, 0.1)} />
      <Tex f="M = \begin{bmatrix} 1 & 0 & \color{#ffd866}{2} \\ 0 & 1 & \color{#ffd866}{1} \end{bmatrix}" x={1180} y={430} size={44} o={eseg(p, 0.4, 0.5)} />
      <At x={1180} y={590} size={24} color={C.muted} w={600} o={eseg(p, 0.45, 0.55)} style={{ textAlign: 'center' }}>{t('watch.affine.label.last_column')}</At>
      <At x={1180} y={720} size={26} className="font-mono" color={C.text} o={eseg(p, 0.7, 0.8)}>borderValue = 0</At>
    </>
  )
}

function ReflectScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const b: Board = { ox: 160, oy: 440, cs: 50 }
  const a = eseg(p, 0.06, 0.24) // y → −y
  const s = eseg(p, 0.3, 0.48) // + (rows − 1)
  const map = (x: number, y: number): [number, number] => [x, lerp(y, -y, a) + (ROWS - 1) * s]
  const cut = eseg(p, 0.5, 0.58)
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        <line x1={b.ox - 60} x2={b.ox + COLS * b.cs + 60} y1={b.oy + b.cs / 2} y2={b.oy + b.cs / 2} stroke={C.blue} strokeWidth={3} opacity={1 - s} />
        <Pixels b={b} map={map} cut={cut} holes={holesOf(reflectX(ROWS))} mark={[2, 6]} />
        <Frame b={b} ticks={false} />
      </Svg>
      <At x={b.ox + COLS * b.cs + 70} y={b.oy + b.cs / 2} size={22} anchor="l" color={C.blue} o={1 - s}>{t('watch.affine.label.x_axis')}</At>
      <Tex f="y' = -y" x={1200} y={170} size={40} color={C.blue} o={eseg(p, 0.02, 0.1)} />
      <Tex f="y' = -y + (\text{rows} - 1)" x={1200} y={270} size={40} color={C.yellow} o={eseg(p, 0.28, 0.36)} />
      <Tex f="M = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & \text{rows} - 1 \end{bmatrix}" x={1200} y={450} size={42} o={eseg(p, 0.55, 0.65)} />
      <At x={1200} y={610} size={26} className="font-mono" color={C.red} o={eseg(p, 0.5, 0.6)}>(2, 6) → (2, 1)</At>
      <At x={1200} y={700} size={26} color={C.yellow} w={620} o={eseg(p, 0.6, 0.7)} style={{ textAlign: 'center' }}>{t('watch.affine.label.upside_down')}</At>
    </>
  )
}

function RotateScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const b: Board = { ox: 340, oy: 330, cs: 46 }
  const answer = 1 - eseg(p, 0.16, 0.2)
  const s1 = eseg(p, 0.22, 0.36)
  const th = 45 * eseg(p, 0.42, 0.62)
  const s3 = eseg(p, 0.68, 0.8)
  const r = (th * Math.PI) / 180
  const [c, sn] = [Math.cos(r), Math.sin(r)]
  const map = (x: number, y: number): [number, number] => {
    const u = x - X0 + 0.5 // pixel centres about the image centre (cols / 2, rows / 2) in continuous coordinates
    const v = y - Y0 + 0.5
    const ru = u * c + v * sn
    const rv = -u * sn + v * c
    // start at the image, move the centre to the origin (s1), turn (th), move back (s3)
    return [ru - 0.5 + X0 * (1 - s1) + X0 * s3, rv - 0.5 + Y0 * (1 - s1) + Y0 * s3]
  }
  const ox = b.ox
  const oy = b.oy
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        <Frame b={b} ticks={false} o={0.6} />
        <Pixels b={b} map={map} turn={-th} />
        <circle cx={ox} cy={oy} r={9} fill={C.red} opacity={eseg(p, 0.2, 0.26)} />
        <circle cx={b.ox + X0 * b.cs} cy={b.oy + Y0 * b.cs} r={7} fill="none" stroke={C.yellow} strokeWidth={3} opacity={eseg(p, 0.2, 0.26)} />
      </Svg>
      <At x={1200} y={300} size={32} w={680} o={answer} style={{ textAlign: 'center' }}>{t('watch.affine.label.answer')}</At>
      <Tex f="M = \begin{bmatrix} -1 & 0 & \text{cols} - 1 \\ 0 & 1 & 0 \end{bmatrix}" x={1200} y={460} size={42} color={C.yellow} o={answer} />
      <At x={ox} y={oy - 30} size={20} color={C.red} o={eseg(p, 0.2, 0.26) * (1 - s3)}>(0, 0)</At>
      <At x={1220} y={150} size={28} className="font-mono" color={s1 > 0 && th === 0 ? C.yellow : C.muted} o={eseg(p, 0.2, 0.28)}>① {t('watch.affine.label.step1')}</At>
      <At x={1220} y={220} size={28} className="font-mono" color={th > 0 && s3 === 0 ? C.yellow : C.muted} o={eseg(p, 0.4, 0.46)}>② {t('watch.affine.label.step2', { deg: Math.round(th) })}</At>
      <At x={1220} y={290} size={28} className="font-mono" color={s3 > 0 ? C.yellow : C.muted} o={eseg(p, 0.66, 0.72)}>③ {t('watch.affine.label.step3')}</At>
      <Tex f="\begin{aligned} x' &= (x - x_0)\cos\theta + (y - y_0)\sin\theta + x_0 \\ y' &= -(x - x_0)\sin\theta + (y - y_0)\cos\theta + y_0 \end{aligned}" x={1200} y={500} size={30} o={eseg(p, 0.82, 0.9)} />
      <At x={1200} y={680} size={24} color={C.muted} w={660} o={eseg(p, 0.48, 0.56)} style={{ textAlign: 'center' }}>{t('watch.affine.label.ccw')}</At>
      <At x={1200} y={770} size={22} className="font-mono" color={C.teal} o={eseg(p, 0.88, 0.95)}>cv2.getRotationMatrix2D((x0, y0), θ, 1.0)</At>
    </>
  )
}

type Lin = [number, number, number, number] // a, b, d, e of [[a, b], [d, e]]
const ROT: Lin = [Math.cos(Math.PI / 6), Math.sin(Math.PI / 6), -Math.sin(Math.PI / 6), Math.cos(Math.PI / 6)]
const SHEAR: Lin = [1, 0.7, 0, 1]
const SCALE: Lin = [1.6, 0, 0, 1.6]
const ID: Lin = [1, 0, 0, 1]
const mix = (A: Lin, B: Lin, t: number): Lin => [lerp(A[0], B[0], t), lerp(A[1], B[1], t), lerp(A[2], B[2], t), lerp(A[3], B[3], t)]

function SurviveScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const stage = p < 0.3 ? 0 : p < 0.6 ? 1 : 2
  const A = p < 0.3 ? mix(ID, ROT, eseg(p, 0.06, 0.2)) : p < 0.6 ? mix(ROT, SHEAR, eseg(p, 0.32, 0.46)) : mix(SHEAR, SCALE, eseg(p, 0.62, 0.74))
  const S = 54, cx = 470, cy = 470
  const L = (x: number, y: number): [number, number] => [cx + (A[0] * x + A[1] * y) * S, cy + (A[2] * x + A[3] * y) * S]
  const lines: ReactNode[] = []
  for (let k = -9; k <= 9; k++) {
    const [a1, a2, b1, b2] = [L(k, -12), L(k, 12), L(-12, k), L(12, k)]
    lines.push(<line key={`v${k}`} x1={a1[0]} y1={a1[1]} x2={a2[0]} y2={a2[1]} stroke={C.blue} strokeOpacity={k ? 0.45 : 0.9} strokeWidth={k ? 1.5 : 3} />)
    lines.push(<line key={`h${k}`} x1={b1[0]} y1={b1[1]} x2={b2[0]} y2={b2[1]} stroke={C.blue} strokeOpacity={k ? 0.45 : 0.9} strokeWidth={k ? 1.5 : 3} />)
  }
  const [pa, pb, pc] = [L(0, 0), L(3, 0), L(0, -2)]
  const len = (u: [number, number], v: [number, number]) => Math.hypot(u[0] - v[0], u[1] - v[1]) / S
  const dot = (pb[0] - pa[0]) * (pc[0] - pa[0]) + (pb[1] - pa[1]) * (pc[1] - pa[1])
  const ang = (Math.acos(Math.max(-1, Math.min(1, dot / (len(pa, pb) * len(pa, pc) * S * S)))) * 180) / Math.PI
  const rows: [string, boolean[]][] = [
    ['parallel', [true, true, true]],
    ['lengths', [true, false, false]],
    ['angles', [true, false, true]],
  ]
  return (
    <>
      <Svg>
        <defs>
          <clipPath id="aff-clip"><rect x={70} y={70} width={800} height={800} rx={20} /></clipPath>
        </defs>
        <rect x={70} y={70} width={800} height={800} rx={20} fill="#0b0e14" stroke={C.grid} />
        <g clipPath="url(#aff-clip)">
          {lines}
          <polygon points={[pa, pb, pc].map((q) => q.join(',')).join(' ')} fill={C.yellow} fillOpacity={0.25} stroke={C.yellow} strokeWidth={4} />
        </g>
      </Svg>
      <At x={1240} y={150} size={34} className="font-black" color={C.yellow}>{t(`watch.affine.label.stage${stage}`)}</At>
      <At x={1240} y={220} size={24} className="font-mono" color={C.muted}>|AB| = {len(pa, pb).toFixed(2)} · |AC| = {len(pa, pc).toFixed(2)} · ∠A = {ang.toFixed(0)}°</At>
      {rows.map(([k, v], i) => (
        <At key={k} x={960} y={330 + i * 80} size={28} anchor="l" w={600}>
          <span style={{ color: v[stage] ? C.green : C.red, fontWeight: 900 }}>{v[stage] ? '✓' : '✗'} </span>
          {t(`watch.affine.label.${k}`)}
        </At>
      ))}
      <At x={1240} y={680} size={26} color={C.teal} w={620} o={eseg(p, 0.82, 0.9)} style={{ textAlign: 'center' }}>{t('watch.affine.label.may')}</At>
    </>
  )
}

function GlobalScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const n = 12
  const cs = 92
  const x0 = (1600 - n * cs) / 2
  const row = Array.from({ length: n }, (_, i) => Math.round(30 + (i * 200) / (n - 1)))
  const arrows = eseg(p, 0.08, 0.3)
  const win = eseg(p, 0.38, 0.46)
  const cell = (y: number, i: number, v: number, stroke?: string) => (
    <g key={`${y}-${i}`}>
      <rect x={x0 + i * cs + 3} y={y} width={cs - 6} height={cs - 6} rx={8} fill={grey(v)} stroke={stroke ?? 'none'} strokeWidth={4} />
      <text x={x0 + i * cs + cs / 2} y={y + cs / 2 + 4} textAnchor="middle" fontSize={22} fontFamily="ui-monospace, monospace" fontWeight={700} fill={v > 120 ? '#0e1117' : C.text}>{v}</text>
    </g>
  )
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        {row.map((v, i) => cell(170, i, v, i === 0 && win > 0 ? C.red : undefined))}
        {row.map((_, i) => cell(560, i, row[n - 1 - i], i === 0 && win > 0 ? C.red : undefined))}
        {row.map((_, i) => {
          const from = x0 + (n - 1 - i) * cs + cs / 2
          const to = x0 + i * cs + cs / 2
          const k = seg(arrows, i / n / 2, i / n / 2 + 0.5)
          return k > 0 ? <path key={i} d={`M ${from} 262 C ${from} 410, ${to} 410, ${to} 556`} fill="none" stroke={i === 0 ? C.red : C.blue} strokeOpacity={i === 0 ? 1 : 0.45} strokeWidth={i === 0 ? 4 : 2} strokeDasharray={`${k * 600} 600`} /> : null
        })}
        <rect x={x0 - 4 - cs} y={556} width={cs * 3 + 8} height={cs + 2} rx={10} fill="none" stroke={C.yellow} strokeWidth={4} strokeDasharray="10 6" opacity={win} />
        <rect x={x0 - 4 - cs} y={166} width={cs * 3 + 8} height={cs + 2} rx={10} fill="none" stroke={C.yellow} strokeWidth={4} strokeDasharray="10 6" opacity={win} />
      </Svg>
      <At x={x0} y={140} size={22} anchor="l" color={C.muted}>{t('watch.affine.label.input_row')}</At>
      <At x={x0} y={700} size={22} anchor="l" color={C.muted}>{t('watch.affine.label.flipped_row')}</At>
      <At x={800} y={760} size={28} color={C.yellow} w={1300} o={win} style={{ textAlign: 'center' }}>{t('watch.affine.label.kernel_reach', { n: n - 1 })}</At>
      <At x={800} y={830} size={28} color={C.teal} w={1300} o={eseg(p, 0.72, 0.8)} style={{ textAlign: 'center' }}>{t('watch.affine.label.sixty_three')}</At>
    </>
  )
}

const CatShape = () => (
  <g fill="#cbd5e1">
    <ellipse cx="58" cy="68" rx="24" ry="19" />
    <circle cx="34" cy="42" r="14" />
    <polygon points="22,34 24,16 33,29" />
    <polygon points="36,28 45,15 47,34" />
    <rect x="34" y="70" width="7" height="20" rx="3" />
    <rect x="48" y="74" width="7" height="16" rx="3" />
    <path d="M80 76 C 97 70, 98 48, 86 40" fill="none" stroke="#cbd5e1" strokeWidth="6" strokeLinecap="round" />
    <circle cx="28" cy="40" r="2.4" fill={C.yellow} />
  </g>
)

function AugmentScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const flip = eseg(p, 0.1, 0.28)
  const turn = 180 * eseg(p, 0.66, 0.84)
  const sx = 1 - 2 * flip
  const box = (x: number, child: ReactNode, tr: string) => (
    <g transform={`translate(${x} 250)`}>
      <rect width={320} height={320} rx={24} fill="#141a24" stroke={C.grid} strokeWidth={2} />
      <g transform={`translate(160 160) ${tr} translate(-160 -160)`}>{child}</g>
    </g>
  )
  const digit = (d: string) => <text x={160} y={250} textAnchor="middle" fontSize={260} fontWeight={800} fill={C.text} fontFamily="'Comic Sans MS','Chalkboard SE','Segoe Print',cursive">{d}</text>
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        {box(130, <g transform="translate(10 10) scale(3)"><CatShape /></g>, `scale(${sx} 1)`)}
        {box(640, digit('2'), `scale(${sx} 1)`)}
        {box(1150, digit('6'), `rotate(${turn})`)}
      </Svg>
      <At x={290} y={640} size={30} className="font-black" color={C.green} o={eseg(p, 0.28, 0.34)}>✓ {t('watch.affine.label.still_cat')}</At>
      <At x={800} y={640} size={30} className="font-black" color={C.red} o={eseg(p, 0.34, 0.4)}>✗ {t('watch.affine.label.not_digit')}</At>
      <At x={1310} y={640} size={30} className="font-black" color={C.red} o={eseg(p, 0.84, 0.9)}>✗ 6 → 9</At>
      <At x={800} y={160} size={26} color={C.muted} w={1300} o={eseg(p, 0, 0.08)} style={{ textAlign: 'center' }}>{t('watch.affine.label.aug_q')}</At>
      <At x={800} y={760} size={26} color={C.yellow} w={1300} o={eseg(p, 0.44, 0.52)} style={{ textAlign: 'center' }}>{t('watch.affine.label.aug_answer')}</At>
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
        <At key={k} x={220} y={240 + k * 120} size={34} anchor="l" w={1180} o={eseg(p, 0.08 + k * 0.12, 0.16 + k * 0.12)}>
          <span style={{ color: C.yellow, fontWeight: 900 }}>{k}. </span>
          {t(`watch.affine.recap.${k}`)}
        </At>
      ))}
      <At x={800} y={800} size={28} color={C.muted} o={eseg(p, 0.75, 0.85)}>{t('watch.ui.try_it')}</At>
    </>
  )
}

export const affine: Episode = {
  id: 'affine',
  scenes: [
    { id: 'title', dur: 5000, cues: [0], render: (p) => <Title p={p} /> },
    { id: 'coords', dur: 15000, cues: [0, 0.3, 0.62], render: (p) => <CoordsScene p={p} /> },
    { id: 'translate', dur: 12000, cues: [0, 0.4, 0.7], render: (p) => <TranslateScene p={p} /> },
    { id: 'reflect', dur: 18000, cues: [0, 0.28, 0.55, 0.8], ponder: true, render: (p) => <ReflectScene p={p} /> },
    { id: 'rotate', dur: 22000, cues: [0, 0.2, 0.4, 0.66], render: (p) => <RotateScene p={p} /> },
    { id: 'survive', dur: 18000, cues: [0, 0.3, 0.6, 0.8], render: (p) => <SurviveScene p={p} /> },
    { id: 'global', dur: 16000, cues: [0, 0.36, 0.7], render: (p) => <GlobalScene p={p} /> },
    { id: 'augment', dur: 16000, cues: [0, 0.3, 0.62], render: (p) => <AugmentScene p={p} /> },
    { id: 'recap', dur: 12000, cues: [0, 0.6], render: (p) => <Recap p={p} /> },
  ],
}
