/* eslint-disable react-refresh/only-export-components -- episode data plus its scene components */
import { useTranslation } from 'react-i18next'
import { eseg, lerp2, seg } from '../../../lib/explainer'
import { LECTURE_NET, XOR, backprop, forward, loss, trainEpochs } from '../../../lib/mlp'
import type { Net } from '../../../lib/mlp'
import { bowl, gdPath } from '../../../lib/gradient'
import type { Episode } from '../Player'
import { At, C, Grid, Svg, Tex, TitleCard } from '../stage'

// "Backpropagation by the numbers" — lecture 6's XOR network, round 1, step 1; asked in every final (2022–2024).

const ETA = 0.5
const S = XOR[0] // (0, 0) → 0
const STEP = backprop(LECTURE_NET, S.x, S.t, ETA)
const F = STEP.fwd
const E0 = 0.5 * (S.t - F.o) ** 2

// Loss over training, sampled every 100 epochs up to 10,000.
const CURVE: number[] = (() => {
  const out = [loss(LECTURE_NET)]
  let net: Net = LECTURE_NET
  for (let k = 1; k <= 100; k++) {
    net = trainEpochs(net, 100, ETA)
    out.push(loss(net))
  }
  return out
})()
const TRAINED = (() => trainEpochs(LECTURE_NET, 10000, ETA))()

type Pt = [number, number]
const NODE: Record<'x1' | 'x2' | 'h1' | 'h2' | 'o', Pt> = { x1: [150, 290], x2: [150, 640], h1: [470, 290], h2: [470, 640], o: [790, 465] }
type NodeId = keyof typeof NODE
const EDGES: { a: NodeId; b: NodeId; w: number; name: string }[] = [
  { a: 'x1', b: 'h1', w: 0, name: 'w_1' },
  { a: 'x2', b: 'h1', w: 1, name: 'w_2' },
  { a: 'x1', b: 'h2', w: 2, name: 'w_3' },
  { a: 'x2', b: 'h2', w: 3, name: 'w_4' },
  { a: 'h1', b: 'o', w: 4, name: 'w_5' },
  { a: 'h2', b: 'o', w: 5, name: 'w_6' },
]
const f4 = (v: number) => (Math.abs(v) < 5e-7 ? '0' : v.toFixed(4).replace(/0+$/, '').replace(/\.$/, ''))
const f6 = (v: number) => v.toFixed(6)

/** The network with its numbers. fwd: 0→1 forward pulses; back: 0→1 backward pulses; upd lights changed weights. */
function NetDiagram({ net, fwd = 0, back = 0, upd, hlEdges }: { net: Net; fwd?: number; back?: number; upd?: number[]; hlEdges?: number[] }) {
  const pulse = (e: (typeof EDGES)[number], t: number, color: string, reverse = false) => {
    if (t <= 0 || t >= 1) return null
    const [a, b] = reverse ? [NODE[e.b], NODE[e.a]] : [NODE[e.a], NODE[e.b]]
    const at = lerp2(a, b, t)
    return <circle cx={at[0]} cy={at[1]} r={11} fill={color} />
  }
  const layer1 = seg(fwd, 0, 0.45)
  const layer2 = seg(fwd, 0.55, 1)
  const back2 = seg(back, 0, 0.45)
  const back1 = seg(back, 0.55, 1)
  return (
    <g>
      {EDGES.map((e) => {
        const [a, b] = [NODE[e.a], NODE[e.b]]
        const v = net.w[e.w]
        const lit = hlEdges?.includes(e.w)
        return <line key={e.name} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke={lit ? C.yellow : v >= 0 ? C.blue : C.red} strokeOpacity={lit ? 1 : 0.55} strokeWidth={2 + Math.min(7, Math.abs(v) * 3)} />
      })}
      {EDGES.slice(0, 4).map((e) => <g key={'f' + e.name}>{pulse(e, layer1, C.yellow)}</g>)}
      {EDGES.slice(4).map((e) => <g key={'f' + e.name}>{pulse(e, layer2, C.yellow)}</g>)}
      {EDGES.slice(4).map((e) => <g key={'b' + e.name}>{pulse(e, back2, C.red, true)}</g>)}
      {EDGES.slice(0, 4).map((e) => <g key={'b' + e.name}>{pulse(e, back1, C.red, true)}</g>)}
      {(Object.keys(NODE) as NodeId[]).map((k) => (
        <circle key={k} cx={NODE[k][0]} cy={NODE[k][1]} r={54} fill={C.bg} stroke={k.startsWith('x') ? C.muted : k === 'o' ? C.teal : C.blue} strokeWidth={4} />
      ))}
      {upd?.map((i) => {
        const e = EDGES[i]
        const m = lerp2(NODE[e.a], NODE[e.b], 0.42)
        return <rect key={'u' + i} x={m[0] - 62} y={m[1] - 22} width={124} height={44} rx={10} fill="none" stroke={C.yellow} strokeWidth={3} />
      })}
    </g>
  )
}

/** HTML labels for the diagram (KaTeX names, values, deltas). */
function NetLabels({ net, show, oldNet }: { net: Net; show: { h?: number; o?: number; dk?: number; dj?: number; th?: number }; oldNet?: Net }) {
  const val = (k: NodeId) => (k === 'x1' ? S.x[0] : k === 'x2' ? S.x[1] : k === 'h1' ? F.h[0] : k === 'h2' ? F.h[1] : F.o)
  const name: Record<NodeId, string> = { x1: 'x_1', x2: 'x_2', h1: 'h_1', h2: 'h_2', o: 'O' }
  const o = (k: NodeId) => (k.startsWith('x') ? 1 : k === 'o' ? (show.o ?? 0) : (show.h ?? 0))
  return (
    <>
      {(Object.keys(NODE) as NodeId[]).map((k) => (
        <div key={k}>
          <Tex f={name[k]} x={NODE[k][0]} y={NODE[k][1] - 16} size={26} color={C.muted} />
          <At x={NODE[k][0]} y={NODE[k][1] + 16} size={26} className="font-mono font-bold" o={o(k)}>
            {k.startsWith('x') ? val(k) : val(k).toFixed(4)}
          </At>
        </div>
      ))}
      {EDGES.map((e) => {
        const m = lerp2(NODE[e.a], NODE[e.b], 0.42)
        const changed = oldNet && Math.abs(oldNet.w[e.w] - net.w[e.w]) > 1e-12
        return (
          <At key={e.name} x={m[0]} y={m[1]} size={21} className="font-mono" color={changed ? C.yellow : C.text} style={{ background: C.bg, padding: '0.1em 0.35em', borderRadius: '0.4em' }}>
            {e.name.replace('w_', 'w')} {f4(net.w[e.w])}
          </At>
        )
      })}
      <At x={NODE.o[0]} y={NODE.o[1] - 90} size={26} color={C.red} className="font-mono font-bold" o={show.dk ?? 0}>δk = {f6(STEP.deltaK)}</At>
      <At x={NODE.h1[0]} y={NODE.h1[1] - 90} size={24} color={C.red} className="font-mono font-bold" o={show.dj ?? 0}>δj1 = {f6(STEP.deltaJ[0])}</At>
      <At x={NODE.h2[0]} y={NODE.h2[1] + 92} size={24} color={C.red} className="font-mono font-bold" o={show.dj ?? 0}>δj2 = {f6(STEP.deltaJ[1])}</At>
      {(['h1', 'h2', 'o'] as const).map((k, i) => (
        <At key={k} x={NODE[k][0] + 70} y={NODE[k][1] + (k === 'h1' ? -60 : 60)} size={20} anchor="l" color={C.muted} className="font-mono" o={show.th ?? 0}>
          θ{i + 1} = {f4(net.theta[i])}
        </At>
      ))}
    </>
  )
}

function Title({ p }: { p: number }) {
  const { t } = useTranslation()
  return <TitleCard p={p} kicker={t('watch.backprop.kicker')} title={t('watch.backprop.title')} sub={t('watch.backprop.sub')} />
}

function SetupScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.25} />
        <NetDiagram net={LECTURE_NET} />
      </Svg>
      <NetLabels net={LECTURE_NET} show={{ th: eseg(p, 0.4, 0.5) }} />
      <At x={1240} y={200} size={28} color={C.muted} className="font-bold uppercase tracking-[0.15em]" o={eseg(p, 0.1, 0.2)}>{t('watch.backprop.label.lecture_net')}</At>
      <Tex f="\eta = 0.5,\quad \text{sigmoid } f" x={1240} y={280} size={36} o={eseg(p, 0.2, 0.3)} />
      <Tex f="(x_1, x_2) = (0, 0),\quad T = 0" x={1240} y={360} size={36} color={C.yellow} o={eseg(p, 0.55, 0.65)} />
      <At x={1240} y={500} size={28} color={C.muted} w={520} o={eseg(p, 0.7, 0.8)} style={{ textAlign: 'center' }}>{t('watch.backprop.label.edge_key')}</At>
    </>
  )
}

function ForwardScene({ p }: { p: number }) {
  const fwd = seg(p, 0.08, 0.6)
  return (
    <>
      <Svg>
        <Grid o={0.25} />
        <NetDiagram net={LECTURE_NET} fwd={fwd} />
      </Svg>
      <NetLabels net={LECTURE_NET} show={{ h: eseg(p, 0.3, 0.36), o: eseg(p, 0.58, 0.64) }} />
      <Tex f="h = f\big(\textstyle\sum w\,x + \theta\big)" x={1240} y={200} size={32} o={eseg(p, 0.1, 0.2)} />
      <Tex f="h_1 = f(0) = 0.5,\quad h_2 = f(0) = 0.5" x={1240} y={290} size={30} o={eseg(p, 0.3, 0.38)} />
      <Tex f="O = f(0.86 \cdot 0.5 - 1.38 \cdot 0.5)" x={1240} y={400} size={30} o={eseg(p, 0.6, 0.68)} />
      <Tex f={`= f(${(F.sums[2] + LECTURE_NET.theta[2]).toFixed(2)}) = ${F.o.toFixed(6)}`} x={1240} y={470} size={34} color={C.teal} o={eseg(p, 0.66, 0.74)} />
      <Tex f={`E = \\tfrac12 (T - O)^2 = ${E0.toFixed(4)}`} x={1240} y={610} size={34} color={C.red} o={eseg(p, 0.8, 0.88)} />
    </>
  )
}

function ChainScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const dEdO = F.o - S.t
  const dOdz = F.o * (1 - F.o)
  const terms: [string, string, string, number][] = [
    ['\\dfrac{\\partial E}{\\partial O}', `O - T = ${dEdO.toFixed(4)}`, t('watch.backprop.label.how_wrong'), 0.2],
    ['\\dfrac{\\partial O}{\\partial z}', `O(1 - O) = ${dOdz.toFixed(4)}`, t('watch.backprop.label.how_steep'), 0.38],
    ['\\dfrac{\\partial z}{\\partial w_5}', `h_1 = ${F.h[0]}`, t('watch.backprop.label.how_much_input'), 0.56],
  ]
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <Tex f="\dfrac{\partial E}{\partial w_5} = \dfrac{\partial E}{\partial O}\cdot\dfrac{\partial O}{\partial z}\cdot\dfrac{\partial z}{\partial w_5}" x={800} y={130} size={46} o={eseg(p, 0, 0.12)} />
      {terms.map(([lhs, rhs, why, at], i) => (
        <div key={i}>
          <Tex f={lhs} x={260} y={290 + i * 150} size={40} color={C.yellow} o={eseg(p, at, at + 0.06)} />
          <Tex f={`= ${rhs}`} x={600} y={290 + i * 150} size={38} o={eseg(p, at + 0.03, at + 0.09)} />
          <At x={1000} y={290 + i * 150} size={26} anchor="l" color={C.muted} w={540} o={eseg(p, at + 0.05, at + 0.11)}>{why}</At>
        </div>
      ))}
      <Tex f={`\\delta_k = (O - T)\\,O(1 - O) = ${STEP.deltaK.toFixed(6)}`} x={800} y={745} size={42} color={C.red} o={eseg(p, 0.76, 0.86)} />
      <At x={800} y={825} size={26} color={C.muted} o={eseg(p, 0.86, 0.94)}>{t('watch.backprop.label.delta_k_is')}</At>
    </>
  )
}

function UpdateOutScene({ p }: { p: number }) {
  const mid: Net = { w: [...LECTURE_NET.w.slice(0, 4), STEP.after.w[4], STEP.after.w[5]] as Net['w'], theta: [0, 0, STEP.after.theta[2]] }
  const done = p > 0.6
  return (
    <>
      <Svg>
        <Grid o={0.25} />
        <NetDiagram net={done ? mid : LECTURE_NET} back={seg(p, 0.05, 0.25) * 0.45} upd={done ? [4, 5] : undefined} />
      </Svg>
      <NetLabels net={done ? mid : LECTURE_NET} oldNet={LECTURE_NET} show={{ h: 1, o: 1, dk: eseg(p, 0.15, 0.25), th: done ? 1 : 0 }} />
      <Tex f="w \leftarrow w - \eta\,\delta\,(\text{input of } w)" x={1240} y={200} size={32} o={eseg(p, 0.2, 0.3)} />
      <Tex f={`w_5 = 0.86 - 0.5 \\cdot ${STEP.deltaK.toFixed(6)} \\cdot 0.5`} x={1240} y={320} size={28} o={eseg(p, 0.32, 0.4)} />
      <Tex f={`= ${f6(STEP.after.w[4])}`} x={1240} y={385} size={36} color={C.yellow} o={eseg(p, 0.38, 0.46)} />
      <Tex f={`w_6 = -1.38 - 0.5 \\cdot ${STEP.deltaK.toFixed(4)} \\cdot 0.5`} x={1240} y={490} size={26} o={eseg(p, 0.5, 0.58)} />
      <Tex f={`= ${f6(STEP.after.w[5])}`} x={1240} y={545} size={30} color={C.yellow} o={eseg(p, 0.54, 0.6)} />
      <Tex f={`\\theta_3 = 0 - 0.5 \\cdot ${STEP.deltaK.toFixed(4)} \\cdot 1 = ${f6(STEP.after.theta[2])}`} x={1240} y={650} size={26} o={eseg(p, 0.6, 0.68)} />
    </>
  )
}

function HiddenScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.25} />
        <NetDiagram net={LECTURE_NET} back={0.5 + seg(p, 0.1, 0.4) * 0.5} hlEdges={p > 0.15 && p < 0.7 ? [4, 5] : undefined} />
      </Svg>
      <NetLabels net={LECTURE_NET} show={{ h: 1, o: 1, dk: 1, dj: eseg(p, 0.5, 0.6) }} />
      <Tex f="\delta_j = O_j(1 - O_j)\,\delta_k\,w_{jk}" x={1240} y={200} size={38} o={eseg(p, 0, 0.1)} />
      <At x={1240} y={290} size={26} color={C.muted} w={540} o={eseg(p, 0.15, 0.25)} style={{ textAlign: 'center' }}>{t('watch.backprop.label.blame')}</At>
      <Tex f={`\\delta_{j1} = 0.25 \\cdot ${STEP.deltaK.toFixed(4)} \\cdot 0.86 = ${f6(STEP.deltaJ[0])}`} x={1240} y={420} size={28} color={C.red} o={eseg(p, 0.45, 0.55)} />
      <Tex f={`\\delta_{j2} = 0.25 \\cdot ${STEP.deltaK.toFixed(4)} \\cdot (-1.38) = ${f6(STEP.deltaJ[1])}`} x={1240} y={490} size={28} color={C.red} o={eseg(p, 0.55, 0.65)} />
      <At x={1240} y={620} size={26} color={C.yellow} w={540} o={eseg(p, 0.7, 0.8)} style={{ textAlign: 'center' }}>{t('watch.backprop.label.old_w5')}</At>
    </>
  )
}

function UpdateHiddenScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.25} />
        <NetDiagram net={STEP.after} upd={p > 0.25 ? [0, 1, 2, 3] : undefined} />
      </Svg>
      <NetLabels net={STEP.after} oldNet={LECTURE_NET} show={{ h: 1, o: 1, dk: 1, dj: 1, th: 1 }} />
      <Tex f="w_1 \leftarrow w_1 - \eta\,\delta_{j1}\,x_1" x={1240} y={200} size={34} o={eseg(p, 0.05, 0.15)} />
      <Tex f={`= -0.65 - 0.5 \\cdot ${STEP.deltaJ[0].toFixed(4)} \\cdot 0 = -0.65`} x={1240} y={280} size={30} o={eseg(p, 0.15, 0.25)} />
      <At x={1240} y={400} size={30} color={C.yellow} w={540} o={eseg(p, 0.3, 0.4)} style={{ textAlign: 'center' }}>{t('watch.backprop.label.zero_input')}</At>
      <Tex f={`\\theta_1 = 0 - 0.5 \\cdot ${STEP.deltaJ[0].toFixed(6)} = ${f6(STEP.after.theta[0])}`} x={1240} y={530} size={28} o={eseg(p, 0.5, 0.58)} />
      <Tex f={`\\theta_2 = 0 - 0.5 \\cdot (${STEP.deltaJ[1].toFixed(6)}) = ${f6(STEP.after.theta[1])}`} x={1240} y={600} size={28} o={eseg(p, 0.58, 0.66)} />
    </>
  )
}

function TrainScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const X = (k: number) => 160 + k * 8
  const max = CURVE[0] * 1.05
  const Y = (v: number) => 720 - (v / max) * 480
  const n = Math.round(seg(p, 0.1, 0.7) * 100)
  const pts = CURVE.slice(0, n + 1).map((v, k) => `${X(k)},${Y(v)}`).join(' ')
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <line x1={160} y1={720} x2={980} y2={720} stroke={C.axis} strokeWidth={2} />
        <line x1={160} y1={720} x2={160} y2={220} stroke={C.axis} strokeWidth={2} />
        <polyline points={pts} fill="none" stroke={C.yellow} strokeWidth={4} />
      </Svg>
      <At x={170} y={200} size={22} color={C.muted} anchor="l">{t('watch.backprop.label.loss')}</At>
      <At x={980} y={750} size={22} color={C.muted} anchor="r">{t('watch.backprop.label.epochs', { n: (n * 100).toLocaleString() })}</At>
      <At x={1240} y={200} size={28} color={C.muted} className="font-bold uppercase tracking-[0.15em]" o={eseg(p, 0.7, 0.78)}>{t('watch.backprop.label.after_10000')}</At>
      {XOR.map((s, i) => {
        const o = forward(TRAINED, s.x).o
        return (
          <At key={i} x={1240} y={290 + i * 80} size={34} className="font-mono" o={eseg(p, 0.72 + i * 0.04, 0.78 + i * 0.04)}>
            <span style={{ color: C.muted }}>({s.x.join(', ')}) → </span>
            <b style={{ color: s.t ? C.blue : C.red }}>{o.toFixed(3)}</b>
            <span style={{ color: C.muted }}>  T = {s.t}</span>
          </At>
        )
      })}
    </>
  )
}

const RATES: { eta: number; key: string; color: string }[] = [
  { eta: 0.1, key: 'small', color: C.blue },
  { eta: 0.45, key: 'good', color: C.teal },
  { eta: 1.05, key: 'large', color: C.red },
]
function RateScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const { f, df } = bowl(2)
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        {RATES.map((r, i) => {
          const ox = 100 + i * 480
          const X = (w: number) => ox + 40 + ((w + 1.5) / 7) * 340
          const Y = (v: number) => 640 - Math.min(v, 14) * 26
          const curve = Array.from({ length: 81 }, (_, k) => {
            const w = -1.5 + (7 * k) / 80
            return `${X(w)},${Y(f(w))}`
          }).join(' ')
          const path = gdPath(-1, r.eta, 8, df)
          const shown = Math.floor(seg(p, 0.15 + i * 0.1, 0.55 + i * 0.1) * 8)
          return (
            <g key={r.key}>
              <polyline points={curve} fill="none" stroke={C.axis} strokeWidth={3} />
              {path.slice(0, shown + 1).map((w, k) =>
                k ? <line key={'l' + k} x1={X(path[k - 1])} y1={Y(f(path[k - 1]))} x2={X(w)} y2={Y(f(w))} stroke={r.color} strokeWidth={2.5} strokeDasharray="6 5" /> : null,
              )}
              {path.slice(0, shown + 1).map((w, k) => (
                <circle key={k} cx={X(w)} cy={Y(Math.min(f(w), 14))} r={k === shown ? 11 : 7} fill={r.color} opacity={k === shown ? 1 : 0.6} />
              ))}
            </g>
          )
        })}
      </Svg>
      {RATES.map((r, i) => (
        <div key={r.key}>
          <Tex f={`\\eta = ${r.eta}`} x={310 + i * 480} y={720} size={34} color={r.color} />
          <At x={310 + i * 480} y={790} size={24} color={C.muted} w={420} o={eseg(p, 0.6 + i * 0.08, 0.68 + i * 0.08)} style={{ textAlign: 'center' }}>{t(`watch.backprop.label.rate_${r.key}`)}</At>
        </div>
      ))}
      <At x={800} y={110} size={36} className="font-black" o={eseg(p, 0, 0.1)}>{t('watch.backprop.label.rate_title')}</At>
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
          {t(`watch.backprop.recap.${k}`)}
        </At>
      ))}
      <At x={800} y={800} size={28} color={C.muted} o={eseg(p, 0.75, 0.85)}>{t('watch.ui.try_it')}</At>
    </>
  )
}

export const backpropEp: Episode = {
  id: 'backprop',
  scenes: [
    { id: 'title', dur: 5000, cues: [0], render: (p) => <Title p={p} /> },
    { id: 'setup', dur: 13000, cues: [0, 0.4, 0.7], render: (p) => <SetupScene p={p} /> },
    { id: 'forward', dur: 16000, cues: [0, 0.3, 0.6, 0.8], render: (p) => <ForwardScene p={p} /> },
    { id: 'chain', dur: 20000, cues: [0, 0.2, 0.38, 0.56, 0.76], render: (p) => <ChainScene p={p} /> },
    { id: 'update_out', dur: 15000, cues: [0, 0.2, 0.5], render: (p) => <UpdateOutScene p={p} /> },
    { id: 'hidden', dur: 17000, cues: [0, 0.15, 0.45, 0.7], render: (p) => <HiddenScene p={p} /> },
    { id: 'update_hidden', dur: 14000, cues: [0, 0.3, 0.5], ponder: true, render: (p) => <UpdateHiddenScene p={p} /> },
    { id: 'train', dur: 14000, cues: [0, 0.4, 0.72], render: (p) => <TrainScene p={p} /> },
    { id: 'rate', dur: 16000, cues: [0, 0.25, 0.55, 0.78], render: (p) => <RateScene p={p} /> },
    { id: 'recap', dur: 12000, cues: [0, 0.6], render: (p) => <Recap p={p} /> },
  ],
}
