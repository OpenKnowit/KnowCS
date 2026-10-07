/* eslint-disable react-refresh/only-export-components -- episode data plus its scene components */
import { useTranslation } from 'react-i18next'
import { eseg, lerp2 } from '../../../lib/explainer'
import { classificationMetrics } from '../../../lib/metrics'
import { crossValidate, sortedLabels } from '../../../lib/crossval'
import type { Ordering } from '../../../lib/crossval'
import type { Episode } from '../Player'
import { At, C, Grid, Svg, Tex, TitleCard } from '../stage'

// "Is the model actually good?" — lecture 3 evaluation: confusion matrix, precision / recall / F1, and the
// unshuffled D-fold trap (2023 Fall Q7, 2024 Spring Q7, Final 2024 Q3b).

type Pt = [number, number]
const LAZY = classificationMetrics([[0, 5], [0, 95]])
const M = [[2, 18], [6, 100]] // rows actual yes/no, cols predicted yes/no
const MET = classificationMetrics(M)
const POS = MET.perClass[0]
const pct = (v: number, d = 0) => `${(v * 100).toFixed(d)}%`

// 126 patients: index → [actual, predicted] (1 = yes)
const PEOPLE: [number, number][] = [
  ...Array.from({ length: 2 }, (): [number, number] => [1, 1]),
  ...Array.from({ length: 18 }, (): [number, number] => [1, 0]),
  ...Array.from({ length: 6 }, (): [number, number] => [0, 1]),
  ...Array.from({ length: 100 }, (): [number, number] => [0, 0]),
]
// matrix cells (stage px): top-left of each cell, cell = 300 × 250
const CELL = { x: 260, y: 220, w: 300, h: 250 }
const cellOrigin = (a: number, p: number): Pt => [CELL.x + (p ? 0 : CELL.w), CELL.y + (a ? 0 : CELL.h)]
const slot = (a: number, p: number, k: number): Pt => {
  const [x, y] = cellOrigin(a, p)
  return [x + 30 + (k % 12) * 21, y + 40 + Math.floor(k / 12) * 21]
}
const PLACE: { from: Pt; to: Pt }[] = (() => {
  const count: Record<string, number> = {}
  return PEOPLE.map(([a, p], i) => {
    const key = `${a}${p}`
    const k = (count[key] = (count[key] ?? -1) + 1)
    return { from: [180 + (i % 21) * 26, 700 + Math.floor(i / 21) * 26], to: slot(a, p, k) }
  })
})()

function Matrix({ o = 1, hlRow, hlCol }: { o?: number; hlRow?: number; hlCol?: number }) {
  const { t } = useTranslation()
  const lbl = ['TP', 'FN', 'FP', 'TN']
  return (
    <>
      <svg viewBox="0 0 1600 900" className="absolute inset-0 h-full w-full" aria-hidden style={{ opacity: o }}>
        {[1, 0].map((a) =>
          [1, 0].map((p) => {
            const [x, y] = cellOrigin(a, p)
            const lit = hlRow === a || hlCol === p
            return <rect key={`${a}${p}`} x={x} y={y} width={CELL.w - 8} height={CELL.h - 8} rx={14} fill={a === p ? '#17322d' : '#33191d'} fillOpacity={lit ? 1 : 0.55} stroke={lit ? C.yellow : C.axis} strokeWidth={lit ? 4 : 2} />
          }),
        )}
      </svg>
      {[1, 0].map((a, ai) =>
        [1, 0].map((p, pi) => {
          const [x, y] = cellOrigin(a, p)
          return (
            <At key={`${a}${p}`} x={x + CELL.w - 24} y={y + 22} size={24} anchor="r" className="font-mono font-black" color={a === p ? C.teal : C.red} o={o}>
              {lbl[ai * 2 + pi]} {M[ai][pi]}
            </At>
          )
        }),
      )}
      <At x={CELL.x + CELL.w} y={CELL.y - 60} size={24} color={C.muted} o={o}>{t('watch.evaluate.label.predicted')}</At>
      <At x={CELL.x + CELL.w / 2} y={CELL.y - 24} size={22} color={C.muted} o={o}>{t('watch.evaluate.label.yes')}</At>
      <At x={CELL.x + CELL.w * 1.5} y={CELL.y - 24} size={22} color={C.muted} o={o}>{t('watch.evaluate.label.no')}</At>
      <At x={CELL.x - 30} y={CELL.y + CELL.h / 2} size={22} anchor="r" color={C.muted} o={o}>{t('watch.evaluate.label.actual_yes')}</At>
      <At x={CELL.x - 30} y={CELL.y + CELL.h * 1.5} size={22} anchor="r" color={C.muted} o={o}>{t('watch.evaluate.label.actual_no')}</At>
    </>
  )
}

function Title({ p }: { p: number }) {
  const { t } = useTranslation()
  return <TitleCard p={p} kicker={t('watch.evaluate.kicker')} title={t('watch.evaluate.title')} sub={t('watch.evaluate.sub')} />
}

function LazyScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        {Array.from({ length: 100 }, (_, i) => (
          <circle key={i} cx={200 + (i % 10) * 52} cy={220 + Math.floor(i / 10) * 52} r={18} fill={i < 5 ? C.red : C.axis} opacity={eseg(p, 0.02 + i * 0.002, 0.1 + i * 0.002)} />
        ))}
      </Svg>
      <At x={1150} y={220} size={30} w={700} o={eseg(p, 0.15, 0.25)} style={{ textAlign: 'center' }}>{t('watch.evaluate.label.lazy_model')}</At>
      <At x={1150} y={360} size={30} color={C.muted} o={eseg(p, 0.35, 0.45)}>{t('watch.evaluate.label.accuracy')}</At>
      <At x={1150} y={440} size={96} className="font-black font-mono" color={C.yellow} o={eseg(p, 0.35, 0.45)}>{pct(LAZY.accuracy)}</At>
      <At x={1150} y={580} size={30} color={C.muted} o={eseg(p, 0.6, 0.7)}>{t('watch.evaluate.label.sick_found')}</At>
      <At x={1150} y={650} size={64} className="font-black font-mono" color={C.red} o={eseg(p, 0.6, 0.7)}>0 / 5</At>
    </>
  )
}

function MatrixScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const fly = eseg(p, 0.2, 0.6)
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <Matrix o={eseg(p, 0.05, 0.15)} />
      <Svg>
        {PEOPLE.map(([a], i) => {
          const at = lerp2(PLACE[i].from, PLACE[i].to, fly)
          return <circle key={i} cx={at[0]} cy={at[1]} r={8} fill={a ? C.red : C.muted} opacity={eseg(p, 0, 0.1)} />
        })}
      </Svg>
      <At x={1250} y={250} size={28} w={560} o={eseg(p, 0.05, 0.15)} style={{ textAlign: 'center' }}>{t('watch.evaluate.label.patients', { n: MET.total })}</At>
      <At x={1250} y={400} size={26} color={C.muted} w={560} o={eseg(p, 0.6, 0.7)} style={{ textAlign: 'center' }}>{t('watch.evaluate.label.diag')}</At>
      <At x={1250} y={560} size={34} className="font-mono" o={eseg(p, 0.72, 0.82)}>
        {t('watch.evaluate.label.accuracy')} = {pct(MET.accuracy)}
      </At>
    </>
  )
}

function PrScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const col = p > 0.12 && p < 0.5 ? 1 : undefined
  const row = p >= 0.5 ? 1 : undefined
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <Matrix hlCol={col} hlRow={row} />
      <Tex f={`\\text{precision} = \\dfrac{TP}{TP + FP} = \\dfrac{2}{2 + 6} = ${pct(POS.precision).replace('%', '\\%')}`} x={1220} y={260} size={32} color={C.blue} o={eseg(p, 0.12, 0.22)} />
      <At x={1220} y={360} size={24} color={C.muted} w={600} o={eseg(p, 0.2, 0.3)} style={{ textAlign: 'center' }}>{t('watch.evaluate.label.precision_q')}</At>
      <Tex f={`\\text{recall} = \\dfrac{TP}{TP + FN} = \\dfrac{2}{2 + 18} = ${pct(POS.recall).replace('%', '\\%')}`} x={1220} y={520} size={32} color={C.orange} o={eseg(p, 0.5, 0.6)} />
      <At x={1220} y={620} size={24} color={C.muted} w={600} o={eseg(p, 0.58, 0.68)} style={{ textAlign: 'center' }}>{t('watch.evaluate.label.recall_q')}</At>
    </>
  )
}

function F1Scene({ p }: { p: number }) {
  const { t } = useTranslation()
  const pairs: [number, number][] = [[POS.precision, POS.recall], [1, 0.02], [0.8, 0.7]]
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <Tex f="F_1 = \dfrac{2\,P\,R}{P + R}" x={800} y={150} size={50} o={eseg(p, 0, 0.1)} />
      <At x={800} y={250} size={26} color={C.muted} o={eseg(p, 0.08, 0.18)}>{t('watch.evaluate.label.harmonic')}</At>
      <At x={800} y={350} size={24} className="font-mono" color={C.muted} o={eseg(p, 0.2, 0.3)} style={{ whiteSpace: 'pre' }}>
        {'   P       R     (P+R)/2    F1'}
      </At>
      {pairs.map(([P, R], i) => (
        <At key={i} x={800} y={420 + i * 70} size={30} className="font-mono" o={eseg(p, 0.25 + i * 0.15, 0.33 + i * 0.15)} style={{ whiteSpace: 'pre' }}>
          {`${P.toFixed(2)}    ${R.toFixed(2)}     `}<span style={{ color: C.muted }}>{((P + R) / 2).toFixed(2)}</span>{'     '}<b style={{ color: C.yellow }}>{((2 * P * R) / (P + R)).toFixed(2)}</b>
        </At>
      ))}
      <At x={800} y={760} size={28} color={C.yellow} w={1300} o={eseg(p, 0.75, 0.85)} style={{ textAlign: 'center' }}>{t('watch.evaluate.label.f1_punish')}</At>
    </>
  )
}

const LABELS = sortedLabels(4, 20)
const CLS = [C.blue, C.yellow, C.red, C.teal]
function Folds({ order, y, label, o = 1, testFold }: { order: Ordering; y: number; label: string; o?: number; testFold?: number }) {
  const res = crossValidate(LABELS, 2, order)
  return (
    <>
      <Svg>
        <g opacity={o}>
          {res.order.map((idx, k) => {
            const fold = k < 40 ? 0 : 1
            return <rect key={k} x={180 + k * 14 + fold * 30} y={y} width={12} height={46} rx={3} fill={CLS[LABELS[idx]]} opacity={testFold === undefined || testFold === fold ? 1 : 0.35} />
          })}
          <line x1={180 + 40 * 14 + 12} y1={y - 12} x2={180 + 40 * 14 + 12} y2={y + 58} stroke={C.text} strokeWidth={3} strokeDasharray="6 5" />
        </g>
      </Svg>
      <At x={170} y={y - 30} size={24} anchor="l" color={C.muted} o={o}>{label}</At>
      <At x={1460} y={y + 23} size={34} anchor="r" className="font-mono font-black" color={res.mean === 0 ? C.red : C.teal} o={o}>{pct(res.mean)}</At>
    </>
  )
}

function CvScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const test = p < 0.3 ? undefined : p < 0.45 ? 0 : p < 0.6 ? 1 : undefined
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <At x={800} y={110} size={30} className="font-black" w={1300} style={{ textAlign: 'center' }} o={eseg(p, 0, 0.08)}>{t('watch.evaluate.label.cv_setup')}</At>
      <Folds order="sorted" y={260} label={t('watch.evaluate.label.sorted')} o={eseg(p, 0.05, 0.15)} testFold={test} />
      <At x={800} y={380} size={26} color={C.red} w={1300} o={eseg(p, 0.32, 0.42)} style={{ textAlign: 'center' }}>{t('watch.evaluate.label.unseen')}</At>
      <Folds order="shuffled" y={520} label={t('watch.evaluate.label.shuffled')} o={eseg(p, 0.62, 0.72)} />
      <Folds order="stratified" y={690} label={t('watch.evaluate.label.stratified')} o={eseg(p, 0.75, 0.85)} />
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
          {t(`watch.evaluate.recap.${k}`)}
        </At>
      ))}
      <At x={800} y={800} size={28} color={C.muted} o={eseg(p, 0.75, 0.85)}>{t('watch.ui.try_it')}</At>
    </>
  )
}

export const evaluate: Episode = {
  id: 'evaluate',
  scenes: [
    { id: 'title', dur: 5000, cues: [0], render: (p) => <Title p={p} /> },
    { id: 'lazy', dur: 14000, cues: [0, 0.35, 0.6], ponder: true, render: (p) => <LazyScene p={p} /> },
    { id: 'matrix', dur: 16000, cues: [0, 0.2, 0.6, 0.72], render: (p) => <MatrixScene p={p} /> },
    { id: 'pr', dur: 17000, cues: [0, 0.12, 0.5, 0.75], render: (p) => <PrScene p={p} /> },
    { id: 'f1', dur: 14000, cues: [0, 0.25, 0.75], render: (p) => <F1Scene p={p} /> },
    { id: 'cv', dur: 22000, cues: [0, 0.3, 0.45, 0.62, 0.8], render: (p) => <CvScene p={p} /> },
    { id: 'recap', dur: 12000, cues: [0, 0.6], render: (p) => <Recap p={p} /> },
  ],
}
