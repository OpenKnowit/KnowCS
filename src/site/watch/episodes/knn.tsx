/* eslint-disable react-refresh/only-export-components -- episode data plus its scene components */
import type { ReactElement } from 'react'
import { useTranslation } from 'react-i18next'
import { eseg, lerp2, seg } from '../../../lib/explainer'
import { computeKnn } from '../../../lib/knn'
import { KNN_RAW_DATA, KNN_STATS } from '../../../data/constants'
import type { KnnPoint, TestPoint } from '../../../types'
import type { Episode } from '../Player'
import { At, C, Grid, Svg, Tex, TitleCard } from '../stage'

// "K nearest neighbours, and why units matter" — lecture 3's T-shirt example; KNN is in 7 of the 9 papers.

const Q: TestPoint = { h: 161, w: 61 }
const S = KNN_STATS
const RAW5 = computeKnn(KNN_RAW_DATA, Q, 5, false, S)
const STD5 = computeKnn(KNN_RAW_DATA, Q, 5, true, S)
const col = (s: KnnPoint['s']) => (s === 'M' ? C.blue : C.orange)

type Pt = [number, number]
// raw view: height 155–175 cm across, weight 55–70 kg up
const RAW = (h: number, w: number): Pt => [140 + (h - 155) * 30, 830 - (w - 55) * 40]
// standardized view: z-scores in the same box
const ZH = (h: number) => (h - S.meanH) / S.stdH
const ZW = (w: number) => (w - S.meanW) / S.stdW
const STD = (h: number, w: number): Pt => [440 + ZH(h) * 120, 530 - ZW(w) * 120]
const at = (h: number, w: number, t: number): Pt => lerp2(RAW(h, w), STD(h, w), t)

function Axes({ t }: { t: number }) {
  return (
    <g opacity={1 - t}>
      <line x1={140} y1={830} x2={740} y2={830} stroke={C.axis} strokeWidth={2} />
      <line x1={140} y1={830} x2={140} y2={230} stroke={C.axis} strokeWidth={2} />
      {[155, 160, 165, 170, 175].map((h) => <text key={h} x={RAW(h, 55)[0]} y={862} textAnchor="middle" fill={C.muted} fontSize={22}>{h}</text>)}
      {[55, 60, 65, 70].map((w) => <text key={w} x={124} y={RAW(155, w)[1] + 7} textAnchor="end" fill={C.muted} fontSize={22}>{w}</text>)}
    </g>
  )
}

function Points({ t = 0, hl, o = 1 }: { t?: number; hl?: Set<number>; o?: number }) {
  return (
    <g opacity={o}>
      {KNN_RAW_DATA.map((d, i) => {
        const p = at(d.h, d.w, t)
        return (
          <g key={i}>
            {hl?.has(i) && <circle cx={p[0]} cy={p[1]} r={20} fill="none" stroke={C.yellow} strokeWidth={3} />}
            <circle cx={p[0]} cy={p[1]} r={11} fill={col(d.s)} stroke={C.bg} strokeWidth={2.5} />
          </g>
        )
      })}
    </g>
  )
}

function Query({ t = 0 }: { t?: number }) {
  const p = at(Q.h, Q.w, t)
  return (
    <g>
      <circle cx={p[0]} cy={p[1]} r={16} fill={C.bg} stroke={C.yellow} strokeWidth={4} />
      <text x={p[0]} y={p[1] + 7} textAnchor="middle" fill={C.yellow} fontSize={20} fontWeight={900}>?</text>
    </g>
  )
}

const Legend = ({ o = 1 }: { o?: number }) => {
  const { t } = useTranslation()
  return (
    <At x={440} y={190} size={24} o={o}>
      <span style={{ color: C.blue }}>● {t('watch.knn.label.medium')}</span>
      <span style={{ color: C.muted }}>{'   '}</span>
      <span style={{ color: C.orange }}>● {t('watch.knn.label.large')}</span>
    </At>
  )
}

function Title({ p }: { p: number }) {
  const { t } = useTranslation()
  return <TitleCard p={p} kicker={t('watch.knn.kicker')} title={t('watch.knn.title')} sub={t('watch.knn.sub')} />
}

function DataScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Axes t={0} />
        <Points o={eseg(p, 0.05, 0.25)} />
        {p > 0.55 && <Query />}
      </Svg>
      <Legend o={eseg(p, 0.1, 0.2)} />
      <At x={440} y={890} size={22} color={C.muted}>{t('watch.knn.label.height')}</At>
      <At x={60} y={530} size={22} color={C.muted} style={{ transform: 'translate(-50%,-50%) rotate(-90deg)' }}>{t('watch.knn.label.weight')}</At>
      <At x={1200} y={330} size={34} className="font-black" w={640} o={eseg(p, 0.55, 0.65)} style={{ textAlign: 'center' }}>{t('watch.knn.label.new_customer')}</At>
      <At x={1200} y={450} size={30} color={C.muted} w={640} o={eseg(p, 0.7, 0.8)} style={{ textAlign: 'center' }}>{t('watch.knn.label.no_training')}</At>
    </>
  )
}

function DistanceScene({ p }: { p: number }) {
  const sorted = [...RAW5.data].sort((a, b) => a.dist - b.dist)
  const lines = seg(p, 0.05, 0.35)
  const rows = Math.floor(seg(p, 0.4, 0.8) * 5.99)
  const q = RAW(Q.h, Q.w)
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Axes t={0} />
        {KNN_RAW_DATA.map((d, i) => {
          const e = lerp2(q, RAW(d.h, d.w), lines)
          return <line key={i} x1={q[0]} y1={q[1]} x2={e[0]} y2={e[1]} stroke={col(d.s)} strokeOpacity={0.35} strokeWidth={2} />
        })}
        <Points />
        <Query />
      </Svg>
      <Tex f="d = \sqrt{(h_1 - h_2)^2 + (w_1 - w_2)^2}" x={1200} y={200} size={34} o={eseg(p, 0.05, 0.15)} />
      {sorted.slice(0, rows).map((d, i) => (
        <At key={d.idx} x={1200} y={300 + i * 64} size={28} className="font-mono">
          <span style={{ color: C.muted }}>{i + 1}. ({d.h}, {d.w}) </span>
          <b style={{ color: col(d.s) }}>{d.s}</b>
          <span style={{ color: C.muted }}>  d = </span>
          {d.dist.toFixed(2)}
        </At>
      ))}
    </>
  )
}

function VoteScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const ks = [1, 3, 5, 7]
  const ki = Math.min(3, Math.floor(seg(p, 0.05, 0.9) * 4))
  const k = ks[ki]
  const res = computeKnn(KNN_RAW_DATA, Q, k, false, S)
  const q = RAW(Q.h, Q.w)
  const r = res.radiusDist * 30 // radius in raw units drawn on the cm axis (1 unit ≈ 30–40 px)
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Axes t={0} />
        <ellipse cx={q[0]} cy={q[1]} rx={r} ry={res.radiusDist * 40} fill={C.yellow} fillOpacity={0.06} stroke={C.yellow} strokeWidth={2.5} strokeDasharray="8 6" />
        <Points hl={new Set(res.topK.map((n) => n.idx))} />
        <Query />
      </Svg>
      <At x={1200} y={230} size={56} className="font-black font-mono" color={C.yellow}>k = {k}</At>
      <At x={1200} y={340} size={34} className="font-mono">
        <b style={{ color: C.blue }}>{res.mCount} M</b>
        <span style={{ color: C.muted }}> : </span>
        <b style={{ color: C.orange }}>{res.lCount} L</b>
      </At>
      <At x={1200} y={430} size={32}>
        {t('watch.knn.label.predict')} <b style={{ color: col(res.prediction) }}>{res.prediction}</b>
      </At>
      {ks.slice(0, ki + 1).map((kk, i) => {
        const r2 = computeKnn(KNN_RAW_DATA, Q, kk, false, S)
        return (
          <At key={kk} x={1200} y={560 + i * 52} size={24} className="font-mono" color={C.muted}>
            k = {kk}: {r2.mCount} M / {r2.lCount} L → <b style={{ color: col(r2.prediction) }}>{r2.prediction}</b>
          </At>
        )
      })}
    </>
  )
}

function ScaleScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const tt = eseg(p, 0.3, 0.6)
  const show = p < 0.3 ? RAW5 : p > 0.6 ? STD5 : null
  const q = at(Q.h, Q.w, tt)
  // the 5-NN radius as an ellipse: a circle in whichever space the distance is measured in
  const rad = show ? (show === RAW5 ? { rx: RAW5.radiusDist * 30, ry: RAW5.radiusDist * 40 } : { rx: STD5.radiusDist * 120, ry: STD5.radiusDist * 120 }) : null
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Axes t={tt} />
        {rad && <ellipse cx={q[0]} cy={q[1]} rx={rad.rx} ry={rad.ry} fill={C.yellow} fillOpacity={0.06} stroke={C.yellow} strokeWidth={2.5} strokeDasharray="8 6" />}
        <Points t={tt} hl={show ? new Set(show.topK.map((n) => n.idx)) : undefined} />
        <Query t={tt} />
      </Svg>
      <At x={1200} y={200} size={30} className="font-black" w={640} style={{ textAlign: 'center' }}>{t(p < 0.45 ? 'watch.knn.label.raw_units' : 'watch.knn.label.z_units')}</At>
      <Tex f="z = \dfrac{x - \text{mean}}{\text{standard deviation}}" x={1200} y={320} size={36} o={eseg(p, 0.2, 0.3)} />
      <At x={1200} y={430} size={24} color={C.muted} className="font-mono" o={eseg(p, 0.25, 0.35)}>h: 164 ± 4.33   w: 62.33 ± 2.63</At>
      <At x={1200} y={560} size={28} w={640} o={eseg(p, 0.65, 0.75)} style={{ textAlign: 'center' }}>{t('watch.knn.label.fifth_changed')}</At>
      <At x={1200} y={680} size={26} color={C.yellow} w={640} o={eseg(p, 0.8, 0.9)} style={{ textAlign: 'center' }}>{t('watch.knn.label.still_m')}</At>
    </>
  )
}

const CELLS_X = 30
const CELLS_Y = 22
function regions(k: number): ReactElement[] {
  const out: ReactElement[] = []
  for (let i = 0; i < CELLS_X; i++)
    for (let j = 0; j < CELLS_Y; j++) {
      const h = 155 + ((i + 0.5) / CELLS_X) * 20
      const w = 55 + ((j + 0.5) / CELLS_Y) * 15
      const r = computeKnn(KNN_RAW_DATA, { h, w }, k, true, S)
      const [x, y] = RAW(155 + (i / CELLS_X) * 20, 55 + ((j + 1) / CELLS_Y) * 15)
      out.push(<rect key={i * CELLS_Y + j} x={x} y={y} width={600 / CELLS_X + 0.5} height={600 / CELLS_Y + 0.5} fill={col(r.prediction)} opacity={0.16} />)
    }
  return out
}
const REGIONS = { 1: regions(1), 15: regions(15) }

function BoundaryScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const k = p < 0.5 ? 1 : 15
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <g opacity={eseg(p, 0.05, 0.15)}>{REGIONS[k as 1 | 15]}</g>
        <Axes t={0} />
        <Points />
      </Svg>
      <At x={1200} y={230} size={56} className="font-black font-mono" color={C.yellow}>k = {k}</At>
      <At x={1200} y={360} size={28} w={640} style={{ textAlign: 'center' }}>{t(k === 1 ? 'watch.knn.label.k1' : 'watch.knn.label.k15')}</At>
      <At x={1200} y={560} size={26} color={C.muted} w={640} o={eseg(p, 0.75, 0.85)} style={{ textAlign: 'center' }}>{t('watch.knn.label.choose_k')}</At>
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
          {t(`watch.knn.recap.${k}`)}
        </At>
      ))}
      <At x={800} y={800} size={28} color={C.muted} o={eseg(p, 0.75, 0.85)}>{t('watch.ui.try_it')}</At>
    </>
  )
}

export const knn: Episode = {
  id: 'knn',
  scenes: [
    { id: 'title', dur: 5000, cues: [0], render: (p) => <Title p={p} /> },
    { id: 'data', dur: 12000, cues: [0, 0.55, 0.7], render: (p) => <DataScene p={p} /> },
    { id: 'distance', dur: 15000, cues: [0, 0.4], render: (p) => <DistanceScene p={p} /> },
    { id: 'vote', dur: 17000, cues: [0, 0.3, 0.55, 0.8], ponder: true, render: (p) => <VoteScene p={p} /> },
    { id: 'scale', dur: 19000, cues: [0, 0.2, 0.32, 0.65, 0.82], render: (p) => <ScaleScene p={p} /> },
    { id: 'boundary', dur: 15000, cues: [0, 0.5, 0.75], render: (p) => <BoundaryScene p={p} /> },
    { id: 'recap', dur: 12000, cues: [0, 0.6], render: (p) => <Recap p={p} /> },
  ],
}
