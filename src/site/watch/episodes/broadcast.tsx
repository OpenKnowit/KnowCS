/* eslint-disable react-refresh/only-export-components -- episode data plus its scene components */
import { Fragment } from 'react'
import { useTranslation } from 'react-i18next'
import { eseg, lerp } from '../../../lib/explainer'
import { planBroadcast, shapeLabel } from '../../../lib/broadcast'
import { EXAM_TEST, EXAM_TRAIN, pairwiseDiff, pairwiseDist, pairwiseSqDist } from '../../../lib/pairwise'
import type { Episode } from '../Player'
import { At, C, Grid, Svg, Tex, TitleCard } from '../stage'

// "Broadcasting, and pairwise distances in one line" — lecture 1; a broadcasting question is in all six midterms.

const CS = 74 // cell size

/** A 2-D array drawn as cells. ghost = how far stretched copies have slid out (0…1). */
function Cells({ x, y, v, color, o = 1, cs = CS, dim = false, label }: { x: number; y: number; v: (number | string)[][]; color: string; o?: number; cs?: number; dim?: boolean; label?: string }) {
  return (
    <g opacity={o}>
      {v.flatMap((row, i) =>
        row.map((val, j) => (
          <g key={`${i}-${j}`}>
            <rect x={x + j * cs} y={y + i * cs} width={cs - 6} height={cs - 6} rx={10} fill={color} fillOpacity={dim ? 0.08 : 0.2} stroke={color} strokeOpacity={dim ? 0.4 : 1} strokeWidth={2.5} strokeDasharray={dim ? '6 5' : undefined} />
            <text x={x + j * cs + (cs - 6) / 2} y={y + i * cs + cs / 2 + 4} textAnchor="middle" fill={dim ? C.muted : C.text} fontSize={cs * 0.36} fontFamily="ui-monospace, monospace" fontWeight={700}>
              {val}
            </text>
          </g>
        )),
      )}
      {label && (
        <text x={x} y={y - 18} fill={C.muted} fontSize={22} fontFamily="ui-monospace, monospace">
          {label}
        </text>
      )}
    </g>
  )
}

const A = [[1, 2, 3], [4, 5, 6]]
const B = [[10, 20, 30], [40, 50, 60]]
const COL = [[1], [2], [3]]
const ROW = [[10, 20, 30, 40]]
const add = (a: number[][], b: number[][]) => a.map((r, i) => r.map((v, j) => v + b[i][j]))

function Title({ p }: { p: number }) {
  const { t } = useTranslation()
  return <TitleCard p={p} kicker={t('watch.broadcast.kicker')} title={t('watch.broadcast.title')} sub={t('watch.broadcast.sub')} />
}

function ShapeScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Cells x={300} y={300} v={A} color={C.blue} o={eseg(p, 0, 0.15)} />
        <line x1={260} y1={300} x2={260} y2={440} stroke={C.yellow} strokeWidth={4} opacity={eseg(p, 0.3, 0.4)} />
        <line x1={300} y1={470} x2={510} y2={470} stroke={C.teal} strokeWidth={4} opacity={eseg(p, 0.45, 0.55)} />
      </Svg>
      <At x={400} y={220} size={28} className="font-mono" o={eseg(p, 0, 0.15)}>a.shape == (2, 3)</At>
      <At x={240} y={370} size={24} anchor="r" color={C.yellow} o={eseg(p, 0.3, 0.4)}>axis 0</At>
      <At x={405} y={505} size={24} color={C.teal} o={eseg(p, 0.45, 0.55)}>axis 1</At>
      <At x={1100} y={330} size={30} w={700} o={eseg(p, 0.2, 0.3)} style={{ textAlign: 'center' }}>{t('watch.broadcast.label.shape')}</At>
      <At x={1100} y={480} size={28} color={C.muted} w={700} o={eseg(p, 0.6, 0.7)} style={{ textAlign: 'center' }}>{t('watch.broadcast.label.elementwise')}</At>
    </>
  )
}

function SameScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Cells x={140} y={330} v={A} color={C.blue} />
        <Cells x={560} y={330} v={B} color={C.orange} o={eseg(p, 0.1, 0.2)} />
        <Cells x={1020} y={330} v={add(A, B)} color={C.teal} o={eseg(p, 0.4, 0.55)} />
      </Svg>
      <At x={480} y={400} size={50} o={eseg(p, 0.1, 0.2)}>+</At>
      <At x={930} y={400} size={50} o={eseg(p, 0.4, 0.55)}>=</At>
      <At x={800} y={200} size={30} className="font-mono" o={eseg(p, 0, 0.1)}>(2, 3) + (2, 3) → (2, 3)</At>
      <At x={800} y={650} size={28} color={C.muted} w={1300} o={eseg(p, 0.6, 0.7)} style={{ textAlign: 'center' }}>{t('watch.broadcast.label.pair_up')}</At>
    </>
  )
}

function StretchScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const plan = planBroadcast([3, 1], [1, 4])
  const s1 = eseg(p, 0.35, 0.55) // column stretches right
  const s2 = eseg(p, 0.5, 0.7) // row stretches down
  const colX = 160
  const rowX = 640
  const y0 = 300
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        {/* the column, and its copies sliding right */}
        {[1, 2, 3].map((k) => <Cells key={'c' + k} x={colX + lerp(0, k * CS, s1)} y={y0} v={COL} color={C.blue} dim o={Math.min(1, s1 * 4)} />)}
        <Cells x={colX} y={y0} v={COL} color={C.blue} />
        {/* the row, and its copies sliding down */}
        {[1, 2].map((k) => <Cells key={'r' + k} x={rowX} y={y0 + lerp(0, k * CS, s2)} v={ROW} color={C.orange} dim o={Math.min(1, s2 * 4)} />)}
        <Cells x={rowX} y={y0} v={ROW} color={C.orange} />
        <Cells x={1100} y={y0} v={COL.map(([c]) => ROW[0].map((r) => c + r))} color={C.teal} o={eseg(p, 0.75, 0.85)} />
      </Svg>
      <At x={colX + 34} y={y0 - 40} size={24} className="font-mono" color={C.blue}>(3, 1)</At>
      <At x={rowX + 150} y={y0 - 40} size={24} className="font-mono" color={C.orange}>(1, 4)</At>
      <At x={560} y={y0 + 110} size={44}>+</At>
      <At x={1040} y={y0 + 110} size={44} o={eseg(p, 0.75, 0.85)}>=</At>
      <At x={1250} y={y0 - 40} size={24} className="font-mono" color={C.teal} o={eseg(p, 0.75, 0.85)}>(3, 4)</At>
      <At x={800} y={130} size={30} className="font-black" w={1300} style={{ textAlign: 'center' }} o={eseg(p, 0, 0.1)}>{t('watch.broadcast.label.rule')}</At>
      {plan.steps.map((s, i) => (
        <At key={i} x={800} y={680 + i * 50} size={28} className="font-mono" o={eseg(p, 0.12 + i * 0.1, 0.2 + i * 0.1)}>
          axis {s.axis}: {s.a} vs {s.b} → <b style={{ color: C.yellow }}>{s.out}</b>
        </At>
      ))}
      <At x={800} y={820} size={26} color={C.muted} w={1300} o={eseg(p, 0.3, 0.4)} style={{ textAlign: 'center' }}>{t('watch.broadcast.label.right_align')}</At>
    </>
  )
}

function MismatchScene({ p, fixed }: { p: number; fixed: boolean }) {
  const { t } = useTranslation()
  const bad = planBroadcast([2, 3], [2])
  const good = planBroadcast([2, 3], [2, 1])
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Cells x={200} y={330} v={A} color={C.blue} />
        {fixed ? <Cells x={600} y={330} v={[[100], [200]]} color={C.orange} /> : <Cells x={600} y={330} v={[[100, 200]]} color={C.orange} />}
        {fixed && <Cells x={1000} y={330} v={A.map((r, i) => r.map((v) => v + (i + 1) * 100))} color={C.teal} o={eseg(p, 0.3, 0.45)} />}
      </Svg>
      <At x={800} y={170} size={30} className="font-mono" color={fixed ? C.teal : C.red}>
        {fixed ? `a + b[:, None]   ${shapeLabel([2, 3])} + ${shapeLabel([2, 1])} → ${shapeLabel(good.outShape ?? [])}` : `a + b   ${shapeLabel([2, 3])} + ${shapeLabel([2])}`}
      </At>
      {!fixed &&
        bad.steps.map((s, i) => (
          <At key={i} x={800} y={590 + i * 54} size={28} className="font-mono" color={s.verdict === 'mismatch' ? C.red : C.muted} o={eseg(p, 0.1 + i * 0.1, 0.18 + i * 0.1)}>
            axis {s.axis}: {s.a} vs {s.b}{s.padB ? ' (padded)' : ''} → {s.verdict === 'mismatch' ? 'ValueError' : s.out}
          </At>
        ))}
      <At x={800} y={760} size={26} color={fixed ? C.teal : C.red} w={1300} o={fixed ? eseg(p, 0.45, 0.6) : eseg(p, 0.3, 0.4)} style={{ textAlign: 'center' }}>
        {t(fixed ? 'watch.broadcast.label.fixed' : 'watch.broadcast.label.mismatch')}
      </At>
    </>
  )
}

const DIFF = pairwiseDiff(EXAM_TEST, EXAM_TRAIN)
const SQ = pairwiseSqDist(EXAM_TEST, EXAM_TRAIN)
const DIST = pairwiseDist(EXAM_TEST, EXAM_TRAIN)
function PairwiseScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const phase = p < 0.3 ? 0 : p < 0.6 ? 1 : 2
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Cells x={110} y={290} v={EXAM_TEST.map((r) => [...r])} color={C.blue} cs={58} label="X_test (2, 2)" />
        <Cells x={110} y={480} v={EXAM_TRAIN.map((r) => [...r])} color={C.orange} cs={58} label="X_train (4, 2)" />
        {phase >= 1 &&
          DIFF.map((layer, i) => (
            <Cells key={i} x={500 + i * 190} y={290} v={layer} color={i ? C.purple : C.teal} cs={58} o={eseg(p, 0.3 + i * 0.06, 0.38 + i * 0.06)} label={`diff[${i}]`} />
          ))}
        {phase >= 2 && <Cells x={1000} y={250} v={SQ} color={C.yellow} cs={74} o={eseg(p, 0.6, 0.68)} label="(diff ** 2).sum(axis=-1)  (2, 4)" />}
        {phase >= 2 && <Cells x={1000} y={520} v={DIST.map((r) => r.map((v) => v.toFixed(2)))} color={C.teal} cs={92} o={eseg(p, 0.74, 0.82)} label="np.sqrt(…)  (2, 4)" />}
      </Svg>
      <At x={800} y={110} size={26} className="font-mono" w={1500} style={{ textAlign: 'center' }}>
        X_test[:, None, :] - X_train[None, :, :]
      </At>
      <At x={620} y={230} size={22} color={C.muted} className="font-mono" o={eseg(p, 0.3, 0.4)}>
        (2, 1, 2) − (1, 4, 2) → (2, 4, 2)
      </At>
      <At x={800} y={840} size={24} color={C.muted} w={1400} o={eseg(p, 0.85, 0.92)} style={{ textAlign: 'center' }}>{t('watch.broadcast.label.no_loops')}</At>
    </>
  )
}

function ExpandScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <Tex f="\lVert x - y \rVert^2 = \lVert x \rVert^2 + \lVert y \rVert^2 - 2\,x\cdot y" x={800} y={200} size={50} o={eseg(p, 0, 0.12)} />
      {[
        ['(X_test ** 2).sum(axis=1)[:, None]', '(2, 1)', C.blue, 0.2],
        ['(X_train ** 2).sum(axis=1)[None, :]', '(1, 4)', C.orange, 0.35],
        ['- 2 * X_test @ X_train.T', '(2, 4)', C.purple, 0.5],
      ].map(([code, shape, col, at], i) => (
        <Fragment key={i}>
          <At x={240} y={360 + i * 90} size={30} anchor="l" className="font-mono" color={String(col)} o={eseg(p, Number(at), Number(at) + 0.08)}>{String(code)}</At>
          <At x={1360} y={360 + i * 90} size={30} anchor="r" className="font-mono" color={C.muted} o={eseg(p, Number(at), Number(at) + 0.08)}>{String(shape)}</At>
        </Fragment>
      ))}
      <At x={800} y={680} size={30} color={C.yellow} w={1300} o={eseg(p, 0.66, 0.76)} style={{ textAlign: 'center' }}>{t('watch.broadcast.label.memory')}</At>
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
          {t(`watch.broadcast.recap.${k}`)}
        </At>
      ))}
      <At x={800} y={800} size={28} color={C.muted} o={eseg(p, 0.75, 0.85)}>{t('watch.ui.try_it')}</At>
    </>
  )
}

export const broadcast: Episode = {
  id: 'broadcast',
  scenes: [
    { id: 'title', dur: 5000, cues: [0], render: (p) => <Title p={p} /> },
    { id: 'shape', dur: 12000, cues: [0, 0.3, 0.6], render: (p) => <ShapeScene p={p} /> },
    { id: 'same', dur: 10000, cues: [0, 0.4], render: (p) => <SameScene p={p} /> },
    { id: 'stretch', dur: 20000, cues: [0, 0.12, 0.35, 0.75], render: (p) => <StretchScene p={p} /> },
    { id: 'mismatch', dur: 11000, cues: [0, 0.3], ponder: true, render: (p) => <MismatchScene p={p} fixed={false} /> },
    { id: 'fix', dur: 10000, cues: [0, 0.45], render: (p) => <MismatchScene p={p} fixed /> },
    { id: 'pairwise', dur: 24000, cues: [0, 0.3, 0.6, 0.85], render: (p) => <PairwiseScene p={p} /> },
    { id: 'expand', dur: 15000, cues: [0, 0.2, 0.66], render: (p) => <ExpandScene p={p} /> },
    { id: 'recap', dur: 12000, cues: [0, 0.6], render: (p) => <Recap p={p} /> },
  ],
}
