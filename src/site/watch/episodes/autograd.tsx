/* eslint-disable react-refresh/only-export-components -- episode data plus its scene components */
import { Fragment } from 'react'
import { useTranslation } from 'react-i18next'
import { eseg, lerp2, seg } from '../../../lib/explainer'
import { lectureExample, sigmoid } from '../../../lib/autogradDemo'
import type { Episode } from '../Player'
import { At, C, Grid, Svg, Tex, TitleCard } from '../stage'

// "PyTorch: tensors and autograd" — lecture 9's computational graph, loss.backward(), and torch.no_grad().

const EX = lectureExample()
const f2 = (v: number) => v.toFixed(2)
const f4 = (v: number) => v.toFixed(4)

type Pt = [number, number]
const NODES: Record<'x' | 'w' | 'b' | 'mm' | 'add' | 'z' | 'loss' | 'y', Pt> = {
  x: [170, 250],
  w: [170, 450],
  b: [170, 650],
  mm: [480, 350],
  add: [760, 500],
  z: [1000, 500],
  y: [1000, 260],
  loss: [1280, 380],
}
type N = keyof typeof NODES
const EDGES: [N, N][] = [['x', 'mm'], ['w', 'mm'], ['mm', 'add'], ['b', 'add'], ['add', 'z'], ['z', 'loss'], ['y', 'loss']]

function Graph({ fwd = 1, back = 0, lit = [] as N[] }: { fwd?: number; back?: number; lit?: N[] }) {
  const leaf = (n: N) => n === 'x' || n === 'w' || n === 'b' || n === 'y'
  const param = (n: N) => n === 'w' || n === 'b'
  return (
    <g>
      {EDGES.map(([a, b], i) => {
        const k = seg(fwd, i / EDGES.length, (i + 1) / EDGES.length)
        const e = lerp2(NODES[a], NODES[b], k)
        const bk = seg(back, 1 - (i + 1) / EDGES.length, 1 - i / EDGES.length)
        return (
          <g key={a + b}>
            <line x1={NODES[a][0]} y1={NODES[a][1]} x2={e[0]} y2={e[1]} stroke={C.axis} strokeWidth={3} />
            {bk > 0 && bk < 1 && a !== 'y' ? <circle cx={lerp2(NODES[b], NODES[a], bk)[0]} cy={lerp2(NODES[b], NODES[a], bk)[1]} r={10} fill={C.red} /> : null}
          </g>
        )
      })}
      {(Object.keys(NODES) as N[]).map((n) => {
        const [x, y] = NODES[n]
        const on = lit.includes(n)
        return leaf(n) ? (
          <rect key={n} x={x - 60} y={y - 34} width={120} height={68} rx={12} fill={C.bg} stroke={on ? C.yellow : param(n) ? C.orange : C.muted} strokeWidth={on ? 5 : 3} />
        ) : (
          <circle key={n} cx={x} cy={y} r={n === 'loss' || n === 'z' ? 48 : 40} fill={C.bg} stroke={on ? C.yellow : n === 'loss' ? C.red : C.blue} strokeWidth={on ? 5 : 3} />
        )
      })}
    </g>
  )
}

function GraphLabels({ o = 1, grads = false }: { o?: number; grads?: boolean }) {
  const { t } = useTranslation()
  const L: [N, string][] = [['x', 'x'], ['w', 'w'], ['b', 'b'], ['mm', '@'], ['add', '+'], ['z', 'z'], ['y', 'y'], ['loss', '\\text{loss}']]
  return (
    <>
      {L.map(([n, f]) => <Tex key={n} f={f} x={NODES[n][0]} y={NODES[n][1]} size={30} o={o} />)}
      <At x={NODES.w[0]} y={NODES.w[1] + 56} size={20} color={C.orange} o={o}>requires_grad</At>
      <At x={NODES.b[0]} y={NODES.b[1] + 56} size={20} color={C.orange} o={o}>requires_grad</At>
      {grads && (
        <>
          <At x={NODES.w[0] + 80} y={NODES.w[1] - 20} size={20} anchor="l" color={C.red} className="font-mono">w.grad</At>
          <At x={NODES.b[0] + 80} y={NODES.b[1] - 20} size={20} anchor="l" color={C.red} className="font-mono">b.grad</At>
        </>
      )}
      <At x={NODES.x[0]} y={NODES.x[1] - 60} size={18} color={C.muted} o={o}>{t('watch.autograd.label.input')}</At>
    </>
  )
}

function Title({ p }: { p: number }) {
  const { t } = useTranslation()
  return <TitleCard p={p} kicker={t('watch.autograd.kicker')} title={t('watch.autograd.title')} sub={t('watch.autograd.sub')} />
}

function TensorScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <At x={800} y={170} size={32} className="font-black" w={1300} style={{ textAlign: 'center' }} o={eseg(p, 0, 0.1)}>{t('watch.autograd.label.tensor')}</At>
      {[
        ['t.shape', 'torch.Size([5, 3])', 0.15],
        ['t.dtype', 'torch.float32', 0.25],
        ['t.device', "cpu  /  cuda:0", 0.35],
        ['t.requires_grad', 'True / False', 0.45],
      ].map(([a, b, at], i) => (
        <Fragment key={i}>
          <At x={560} y={310 + i * 90} size={30} anchor="r" className="font-mono" color={C.blue} o={eseg(p, Number(at), Number(at) + 0.08)}>{String(a)}</At>
          <At x={620} y={310 + i * 90} size={30} anchor="l" className="font-mono" o={eseg(p, Number(at), Number(at) + 0.08)}>{String(b)}</At>
        </Fragment>
      ))}
      <At x={800} y={720} size={26} color={C.yellow} w={1300} o={eseg(p, 0.65, 0.75)} style={{ textAlign: 'center' }}>{t('watch.autograd.label.cpu_ok')}</At>
      <At x={800} y={800} size={24} color={C.muted} className="font-mono" o={eseg(p, 0.8, 0.88)}>torch.from_numpy(a)   ·   t.numpy()</At>
    </>
  )
}

function GraphScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const lines = [
    'x = torch.ones(5)',
    'y = torch.zeros(3)',
    'w = torch.randn(5, 3, requires_grad=True)',
    'b = torch.randn(3, requires_grad=True)',
    'z = torch.matmul(x, w) + b',
    'loss = F.binary_cross_entropy_with_logits(z, y)',
  ]
  const k = Math.min(lines.length, Math.floor(seg(p, 0.05, 0.75) * (lines.length + 0.99)))
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        <g opacity={eseg(p, 0.3, 0.45)}>
          <Graph fwd={seg(p, 0.45, 0.85)} />
        </g>
      </Svg>
      <GraphLabels o={eseg(p, 0.3, 0.45)} />
      <At x={800} y={800} size={24} className="font-mono" w={1500} style={{ textAlign: 'center' }} color={C.text}>
        {lines[Math.max(0, k - 1)]}
      </At>
      <At x={800} y={850} size={20} color={C.muted} o={eseg(p, 0.85, 0.92)}>{t('watch.autograd.label.recorded')}</At>
    </>
  )
}

function ForwardScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        <Graph lit={p > 0.5 ? ['z', 'loss'] : ['x', 'w', 'b']} />
      </Svg>
      <GraphLabels />
      <At x={NODES.z[0]} y={NODES.z[1] + 80} size={22} className="font-mono" color={C.blue} o={eseg(p, 0.3, 0.4)}>
        z = [{EX.z.map(f2).join(', ')}]
      </At>
      <At x={NODES.loss[0]} y={NODES.loss[1] + 80} size={24} className="font-mono" color={C.red} o={eseg(p, 0.55, 0.65)}>
        loss = {f4(EX.loss)}
      </At>
      <Tex f="\text{loss} = \tfrac13\sum_j \Big[-y_j\log\sigma(z_j) - (1 - y_j)\log\big(1 - \sigma(z_j)\big)\Big]" x={800} y={820} size={28} o={eseg(p, 0.7, 0.8)} />
      <At x={800} y={120} size={26} color={C.muted} w={1300} style={{ textAlign: 'center' }}>{t('watch.autograd.label.forward')}</At>
    </>
  )
}

function BackwardScene({ p }: { p: number }) {
  const showW = p > 0.55
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        <Graph back={seg(p, 0.05, 0.45)} lit={showW ? ['w', 'b'] : []} />
      </Svg>
      <GraphLabels grads={showW} />
      <At x={800} y={110} size={30} className="font-mono" color={C.red} o={eseg(p, 0, 0.08)}>loss.backward()</At>
      <Tex f="\dfrac{\partial\,\text{loss}}{\partial z_j} = \dfrac{\sigma(z_j) - y_j}{3}" x={1240} y={640} size={30} o={eseg(p, 0.25, 0.35)} />
      <At x={1240} y={730} size={22} className="font-mono" color={C.red} o={eseg(p, 0.35, 0.45)}>
        [{EX.dz.map(f4).join(', ')}]
      </At>
      <At x={560} y={660} size={21} className="font-mono" color={C.red} anchor="l" o={showW ? eseg(p, 0.55, 0.65) : 0} style={{ whiteSpace: 'pre', lineHeight: 1.35 }}>
        {`w.grad =\n${EX.wGrad.map((r) => `[${r.map(f4).join(', ')}]`).join('\n')}`}
      </At>
    </>
  )
}

function WhyRowsScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.15} />
      </Svg>
      <Tex f="\dfrac{\partial\,\text{loss}}{\partial w_{ij}} = \dfrac{\partial\,\text{loss}}{\partial z_j}\cdot\dfrac{\partial z_j}{\partial w_{ij}} = \dfrac{\sigma(z_j) - y_j}{3}\cdot x_i" x={800} y={230} size={40} o={eseg(p, 0.05, 0.18)} />
      <At x={800} y={400} size={30} w={1300} o={eseg(p, 0.25, 0.35)} style={{ textAlign: 'center' }}>{t('watch.autograd.label.rows_same')}</At>
      <Tex f={`x_i = 1 \\;\\Rightarrow\\; \\text{every row of } w.\\text{grad} = b.\\text{grad} = [${EX.bGrad.map(f4).join(',\\ ')}]`} x={800} y={540} size={30} color={C.yellow} o={eseg(p, 0.45, 0.55)} />
      <At x={800} y={700} size={26} color={C.muted} w={1300} o={eseg(p, 0.65, 0.75)} style={{ textAlign: 'center' }}>{t('watch.autograd.label.same_as_backprop')}</At>
      <At x={800} y={800} size={22} color={C.muted} className="font-mono" o={eseg(p, 0.8, 0.88)}>σ(z) = [{EX.z.map((z) => f4(sigmoid(z))).join(', ')}]</At>
    </>
  )
}

function NoGradScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.15} />
      </Svg>
      <At x={800} y={170} size={30} className="font-black" w={1300} style={{ textAlign: 'center' }} o={eseg(p, 0, 0.1)}>{t('watch.autograd.label.inference')}</At>
      <At x={560} y={340} size={28} className="font-mono" anchor="l" o={eseg(p, 0.1, 0.2)} style={{ whiteSpace: 'pre', lineHeight: 1.5 }}>
        {'z = torch.matmul(x, w) + b\nprint(z.requires_grad)   # True'}
      </At>
      <At x={560} y={520} size={28} className="font-mono" anchor="l" color={C.teal} o={eseg(p, 0.35, 0.45)} style={{ whiteSpace: 'pre', lineHeight: 1.5 }}>
        {'with torch.no_grad():\n    z = torch.matmul(x, w) + b\nprint(z.requires_grad)   # False'}
      </At>
      <At x={800} y={760} size={26} color={C.yellow} w={1300} o={eseg(p, 0.6, 0.7)} style={{ textAlign: 'center' }}>{t('watch.autograd.label.why_no_grad')}</At>
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
          {t(`watch.autograd.recap.${k}`)}
        </At>
      ))}
    </>
  )
}

export const autograd: Episode = {
  id: 'autograd',
  scenes: [
    { id: 'title', dur: 5000, cues: [0], render: (p) => <Title p={p} /> },
    { id: 'tensor', dur: 14000, cues: [0, 0.15, 0.65], render: (p) => <TensorScene p={p} /> },
    { id: 'graph', dur: 16000, cues: [0, 0.3, 0.85], render: (p) => <GraphScene p={p} /> },
    { id: 'forward', dur: 13000, cues: [0, 0.3, 0.7], render: (p) => <ForwardScene p={p} /> },
    { id: 'backward', dur: 16000, cues: [0, 0.25, 0.55], ponder: true, render: (p) => <BackwardScene p={p} /> },
    { id: 'rows', dur: 14000, cues: [0, 0.25, 0.65], render: (p) => <WhyRowsScene p={p} /> },
    { id: 'nograd', dur: 13000, cues: [0, 0.35, 0.6], render: (p) => <NoGradScene p={p} /> },
    { id: 'recap', dur: 11000, cues: [0], render: (p) => <Recap p={p} /> },
  ],
}
