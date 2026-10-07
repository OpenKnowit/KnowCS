/* eslint-disable react-refresh/only-export-components -- episode data plus its scene components */
import { Fragment } from 'react'
import { useTranslation } from 'react-i18next'
import { eseg, lerp, seg } from '../../../lib/explainer'
import { clipLine, halfPlane } from '../../../lib/geom2d'
import { fire, perceptronTable } from '../../../lib/perceptronTable'
import type { PerceptronSample, StepActivation } from '../../../lib/perceptronTable'
import type { Episode } from '../Player'
import { At, C, Grid, Svg, Tex, TitleCard } from '../stage'

// "The perceptron learning rule" — lecture 5; the epoch table is in the 2022 Fall/Spring and 2023 Fall/Spring midterms.

type Pt = [number, number]
const F: StepActivation = { high: 1, low: 0, rule: 'ge' }
const AND: PerceptronSample[] = [
  { x: [0, 0], t: 0 },
  { x: [0, 1], t: 0 },
  { x: [1, 0], t: 0 },
  { x: [1, 1], t: 1 },
]
const XOR: PerceptronSample[] = [
  { x: [0, 0], t: 0 },
  { x: [0, 1], t: 1 },
  { x: [1, 0], t: 1 },
  { x: [1, 1], t: 0 },
]
const W0 = [0.1, 0.5]
const T0 = -0.8
const ETA = 0.2
const RUN = perceptronTable(AND, W0, T0, ETA, F, 12)
const XRUN = perceptronTable(XOR, [0, 0], 0, 1, F, 6)

const LO = -0.5
const HI = 1.5
const IN = (v: number[]): Pt => [140 + (v[0] - LO) * 300, 800 - (v[1] - LO) * 300]
const fmt = (v: number) => (Math.abs(v) < 1e-9 ? '0' : String(Number(v.toFixed(4))))

function Plane({ w, theta, data, hl, o = 1, ghost }: { w: number[]; theta: number; data: PerceptronSample[]; hl?: number; o?: number; ghost?: { w: number[]; theta: number } }) {
  const shade = halfPlane(w[0], w[1], theta, LO, HI, true).map((v) => IN(v).join(',')).join(' ')
  const ln = clipLine(w[0], w[1], theta, LO, HI)
  const gl = ghost ? clipLine(ghost.w[0], ghost.w[1], ghost.theta, LO, HI) : null
  return (
    <g opacity={o}>
      <rect x={IN([LO, HI])[0]} y={IN([LO, HI])[1]} width={600} height={600} fill="none" stroke={C.axis} strokeWidth={2} />
      {shade && <polygon points={shade} fill={C.blue} opacity={0.13} />}
      <line x1={IN([0, LO])[0]} y1={IN([0, LO])[1]} x2={IN([0, HI])[0]} y2={IN([0, HI])[1]} stroke={C.grid} strokeWidth={2} />
      <line x1={IN([LO, 0])[0]} y1={IN([LO, 0])[1]} x2={IN([HI, 0])[0]} y2={IN([HI, 0])[1]} stroke={C.grid} strokeWidth={2} />
      {gl && <line x1={IN(gl[0])[0]} y1={IN(gl[0])[1]} x2={IN(gl[1])[0]} y2={IN(gl[1])[1]} stroke={C.muted} strokeWidth={2} strokeDasharray="7 6" />}
      {ln && <line x1={IN(ln[0])[0]} y1={IN(ln[0])[1]} x2={IN(ln[1])[0]} y2={IN(ln[1])[1]} stroke={C.text} strokeWidth={4} />}
      {data.map((s, i) => {
        const [x, y] = IN(s.x)
        const wrong = fire(w[0] * s.x[0] + w[1] * s.x[1] + theta, F) !== s.t
        return (
          <g key={i}>
            {hl === i && <circle cx={x} cy={y} r={30} fill="none" stroke={C.yellow} strokeWidth={4} />}
            {wrong && <circle cx={x} cy={y} r={21} fill="none" stroke={C.red} strokeWidth={3} strokeDasharray="5 4" />}
            {s.t ? <circle cx={x} cy={y} r={14} fill={C.blue} stroke={C.bg} strokeWidth={3} /> : <rect x={x - 12} y={y - 12} width={24} height={24} fill={C.red} stroke={C.bg} strokeWidth={3} />}
          </g>
        )
      })}
    </g>
  )
}

function Title({ p }: { p: number }) {
  const { t } = useTranslation()
  return <TitleCard p={p} kicker={t('watch.perceptron.kicker')} title={t('watch.perceptron.title')} sub={t('watch.perceptron.sub')} />
}

function NeuronScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const nodes: [string, Pt][] = [['x_1', [260, 300]], ['x_2', [260, 600]]]
  const o = eseg(p, 0.05, 0.2)
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <g opacity={o}>
          {nodes.map(([, at], i) => <line key={i} x1={at[0]} y1={at[1]} x2={700} y2={450} stroke={C.blue} strokeWidth={4} />)}
          {nodes.map(([, at], i) => <circle key={'n' + i} cx={at[0]} cy={at[1]} r={50} fill={C.bg} stroke={C.muted} strokeWidth={4} />)}
          <circle cx={700} cy={450} r={80} fill={C.bg} stroke={C.teal} strokeWidth={5} />
          <line x1={780} y1={450} x2={1000} y2={450} stroke={C.teal} strokeWidth={4} />
          <polygon points="1000,436 1024,450 1000,464" fill={C.teal} />
        </g>
      </Svg>
      {nodes.map(([f, at]) => <Tex key={f} f={f} x={at[0]} y={at[1]} size={40} o={o} />)}
      <Tex f="w_1" x={470} y={345} size={32} color={C.blue} o={o} />
      <Tex f="w_2" x={470} y={555} size={32} color={C.blue} o={o} />
      <Tex f="\Sigma,\ \theta" x={700} y={450} size={36} o={o} />
      <Tex f="O" x={1012} y={400} size={40} color={C.teal} o={o} />
      <Tex f="z = w_1x_1 + w_2x_2 + \theta" x={1320} y={300} size={34} o={eseg(p, 0.3, 0.4)} />
      <Tex f="O = f(z) = \begin{cases} 1 & z \ge 0 \\ 0 & z < 0 \end{cases}" x={1320} y={470} size={34} o={eseg(p, 0.45, 0.55)} />
      <At x={1320} y={640} size={26} color={C.muted} w={480} o={eseg(p, 0.65, 0.75)} style={{ textAlign: 'center' }}>{t('watch.perceptron.label.step')}</At>
    </>
  )
}

function GeometryScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const a = lerp(Math.PI / 4, Math.PI / 4 + 2 * Math.PI, eseg(p, 0.25, 0.85))
  const w = [Math.cos(a), Math.sin(a)]
  const theta = -(w[0] * 0.5 + w[1] * 0.5) // the line always passes through (0.5, 0.5)
  const base = IN([0.5, 0.5])
  const tip: Pt = [base[0] + w[0] * 160, base[1] - w[1] * 160]
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Plane w={w} theta={theta} data={[]} />
        <line x1={base[0]} y1={base[1]} x2={tip[0]} y2={tip[1]} stroke={C.yellow} strokeWidth={5} />
        <circle cx={tip[0]} cy={tip[1]} r={9} fill={C.yellow} />
      </Svg>
      <Tex f="\vec w" x={tip[0] + 26} y={tip[1] - 20} size={34} color={C.yellow} />
      <Tex f="w_1x_1 + w_2x_2 + \theta = 0" x={1200} y={250} size={36} o={eseg(p, 0, 0.1)} />
      <At x={1200} y={380} size={28} w={620} o={eseg(p, 0.1, 0.2)} style={{ textAlign: 'center' }}>{t('watch.perceptron.label.line')}</At>
      <At x={1200} y={520} size={28} w={620} color={C.yellow} o={eseg(p, 0.25, 0.35)} style={{ textAlign: 'center' }}>{t('watch.perceptron.label.normal')}</At>
      <At x={1200} y={660} size={26} w={620} color={C.muted} o={eseg(p, 0.6, 0.7)} style={{ textAlign: 'center' }}>{t('watch.perceptron.label.theta_shift')}</At>
    </>
  )
}

function RuleScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <Tex f="\Delta w_i = \eta\,(T - O)\,x_i \qquad \Delta\theta = \eta\,(T - O)" x={800} y={170} size={46} o={eseg(p, 0, 0.1)} />
      {[
        ['T = O', 'watch.perceptron.label.case_ok', C.green, 0.15],
        ['T = 1,\\ O = 0', 'watch.perceptron.label.case_up', C.blue, 0.35],
        ['T = 0,\\ O = 1', 'watch.perceptron.label.case_down', C.red, 0.55],
      ].map(([f, key, col, at], i) => (
        <Fragment key={i}>
          <Tex f={String(f)} x={360} y={340 + i * 150} size={38} color={String(col)} o={eseg(p, Number(at), Number(at) + 0.08)} />
          <At x={620} y={340 + i * 150} size={30} anchor="l" w={880} o={eseg(p, Number(at) + 0.03, Number(at) + 0.11)}>{t(String(key))}</At>
        </Fragment>
      ))}
      <At x={800} y={800} size={28} color={C.yellow} w={1300} o={eseg(p, 0.78, 0.88)} style={{ textAlign: 'center' }}>{t('watch.perceptron.label.only_mistakes')}</At>
    </>
  )
}

function RunScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const n = RUN.rows.length
  const k = Math.min(n - 1, Math.floor(seg(p, 0.04, 0.94) * n))
  const row = RUN.rows[k]
  const prev = k > 0 ? RUN.rows[k - 1] : { w: W0, theta: T0 }
  const first = Math.max(0, k - 5)
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Plane w={row.w} theta={row.theta} data={AND} hl={row.index} ghost={row.updated ? prev : undefined} />
      </Svg>
      <At x={1180} y={140} size={26} color={C.muted}>{t('watch.perceptron.label.and_setup')}</At>
      <At x={1180} y={210} size={24} className="font-mono" color={C.muted}>{t('watch.perceptron.label.cols')}</At>
      {RUN.rows.slice(first, k + 1).map((r, i) => (
        <At key={first + i} x={1180} y={260 + i * 48} size={24} className="font-mono" color={first + i === k ? C.text : C.muted}>
          {r.epoch} │ ({r.x.join(',')}) {r.t} │ {fmt(r.z)} → {r.o} │ <span style={{ color: r.updated ? C.yellow : C.muted }}>({fmt(r.w[0])}, {fmt(r.w[1])}) {fmt(r.theta)}</span>
        </At>
      ))}
      <At x={1180} y={600} size={28} w={620} style={{ textAlign: 'center' }} color={row.updated ? C.yellow : C.green}>
        {row.updated ? t('watch.perceptron.label.updated', { d: row.t - row.o }) : t('watch.perceptron.label.correct')}
      </At>
      <At x={1180} y={720} size={30} className="font-black" color={C.teal} o={k === n - 1 ? 1 : 0}>{t('watch.perceptron.label.converged', { n: RUN.epochs })}</At>
    </>
  )
}

function XorScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const n = XRUN.rows.length
  const k = Math.min(n - 1, Math.floor(seg(p, 0.05, 0.8) * n))
  const row = XRUN.rows[k]
  const errs = (e: number) => XRUN.rows.filter((r) => r.epoch === e && r.updated).length
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Plane w={row.w} theta={row.theta} data={XOR} hl={row.index} />
      </Svg>
      <At x={1180} y={200} size={30} className="font-black">{t('watch.perceptron.label.xor')}</At>
      {Array.from({ length: row.epoch }, (_, e) => (
        <At key={e} x={1180} y={290 + e * 52} size={26} className="font-mono" color={C.muted}>
          {t('watch.perceptron.label.epoch_errors', { e: e + 1, n: errs(e + 1) })}
        </At>
      ))}
      <At x={1180} y={700} size={28} w={620} color={C.red} o={eseg(p, 0.82, 0.9)} style={{ textAlign: 'center' }}>{t('watch.perceptron.label.never')}</At>
    </>
  )
}

function TrapScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <At x={800} y={120} size={30} color={C.yellow} className="font-bold uppercase tracking-[0.15em]" o={eseg(p, 0, 0.08)}>{t('watch.perceptron.label.traps')}</At>
      {[1, 2, 3].map((k) => (
        <At key={k} x={200} y={160 + k * 150} size={32} anchor="l" w={1200} o={eseg(p, 0.05 + k * 0.18, 0.13 + k * 0.18)}>
          <span style={{ color: C.yellow, fontWeight: 900 }}>{k}. </span>
          {t(`watch.perceptron.trap.${k}`)}
        </At>
      ))}
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
          {t(`watch.perceptron.recap.${k}`)}
        </At>
      ))}
      <At x={800} y={800} size={28} color={C.muted} o={eseg(p, 0.75, 0.85)}>{t('watch.ui.try_it')}</At>
    </>
  )
}

export const perceptron: Episode = {
  id: 'perceptron',
  scenes: [
    { id: 'title', dur: 5000, cues: [0], render: (p) => <Title p={p} /> },
    { id: 'neuron', dur: 13000, cues: [0, 0.3, 0.65], render: (p) => <NeuronScene p={p} /> },
    { id: 'geometry', dur: 15000, cues: [0, 0.25, 0.6], render: (p) => <GeometryScene p={p} /> },
    { id: 'rule', dur: 16000, cues: [0, 0.15, 0.35, 0.55, 0.78], ponder: true, render: (p) => <RuleScene p={p} /> },
    { id: 'run', dur: 30000, cues: [0, 0.2, 0.55, 0.9], render: (p) => <RunScene p={p} /> },
    { id: 'xor', dur: 14000, cues: [0, 0.45, 0.82], render: (p) => <XorScene p={p} /> },
    { id: 'traps', dur: 15000, cues: [0, 0.23, 0.41, 0.59], render: (p) => <TrapScene p={p} /> },
    { id: 'recap', dur: 12000, cues: [0, 0.6], render: (p) => <Recap p={p} /> },
  ],
}
