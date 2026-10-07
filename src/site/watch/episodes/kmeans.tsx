/* eslint-disable react-refresh/only-export-components -- episode data plus its scene components */
import { Fragment } from 'react'
import type { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { eseg, lerp, lerp2, seg } from '../../../lib/explainer'
import { kmeansTable } from '../../../lib/kmeansTable'
import { bestSse, makeBlobs } from '../../../lib/kmeansDemo'
import type { Episode } from '../Player'
import { At, C, Grid, Svg, Tex, TitleCard } from '../stage'

// "K-Means: two steps, repeated" — lecture 4; the round-by-round table is in 6 of the 9 papers.

const PTS = makeBlobs()
const GOOD = kmeansTable(PTS, [[1, 1], [2, 2], [9, 9.5]], 'euclidean', 50)
const BAD = kmeansTable(PTS, [[6, 1.5], [5, 3.5], [5, 8]], 'euclidean', 50)
const ELBOW = [1, 2, 3, 4, 5, 6].map((k) => bestSse(PTS, k))
const COLORS = [C.yellow, C.blue, C.red]

type Pt = [number, number]
const X = (v: number) => 120 + v * 62
const Y = (v: number) => 830 - v * 62
const P = (v: number[]): Pt => [X(v[0]), Y(v[1])]
const nearest = (q: number[], cs: number[][]) => cs.reduce((b, c, k) => ((q[0] - c[0]) ** 2 + (q[1] - c[1]) ** 2 < (q[0] - cs[b][0]) ** 2 + (q[1] - cs[b][1]) ** 2 ? k : b), 0)

function Plane({ children }: { children?: ReactNode }) {
  return (
    <>
      <rect x={X(0)} y={Y(10)} width={X(10) - X(0)} height={Y(0) - Y(10)} fill="none" stroke={C.axis} strokeWidth={2} />
      {children}
    </>
  )
}

/** Voronoi shading: each cell coloured by its nearest centroid. */
function Regions({ cs, o }: { cs: number[][]; o: number }) {
  if (o <= 0) return null
  const n = 28
  const s = 10 / n
  const cells = []
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const q = [(i + 0.5) * s, (j + 0.5) * s]
      cells.push(<rect key={i * n + j} x={X(i * s)} y={Y((j + 1) * s)} width={s * 62 + 0.6} height={s * 62 + 0.6} fill={COLORS[nearest(q, cs)]} opacity={0.1 * o} />)
    }
  return <g>{cells}</g>
}

function Star({ at, color, r = 22 }: { at: Pt; color: string; r?: number }) {
  const pts = Array.from({ length: 10 }, (_, i) => {
    const a = -Math.PI / 2 + (i * Math.PI) / 5
    const rr = i % 2 ? r * 0.45 : r
    return `${at[0] + rr * Math.cos(a)},${at[1] + rr * Math.sin(a)}`
  }).join(' ')
  return <polygon points={pts} fill={color} stroke={C.bg} strokeWidth={3} />
}

function Dots({ assign, o = 1, links }: { assign?: number[] | null; o?: number; links?: { cs: number[][]; t: number } }) {
  return (
    <g opacity={o}>
      {links && links.t > 0 && assign && PTS.map((q, i) => {
        const a = P(q)
        const b = lerp2(a, P(links.cs[assign[i]]), links.t)
        return <line key={'l' + i} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke={COLORS[assign[i]]} strokeOpacity={0.45} strokeWidth={2} />
      })}
      {PTS.map((q, i) => <circle key={i} cx={P(q)[0]} cy={P(q)[1]} r={9} fill={assign ? COLORS[assign[i]] : C.text} stroke={C.bg} strokeWidth={2} />)}
    </g>
  )
}

/** State of the good run at "round progress" q ∈ [0, rounds]: each unit = assign (first half), move (second half). */
function runState(run: typeof GOOD, q: number) {
  const r = Math.min(run.rounds.length - 1, Math.floor(q))
  const f = q - r
  const R = run.rounds[r]
  const move = seg(f, 0.5, 0.95)
  const cs = R.centroids.map((c, k) => [lerp(c[0], R.next[k][0], move), lerp(c[1], R.next[k][1], move)])
  return { r, R, cs, assigned: f > 0.1 || r > 0, linkT: seg(f, 0.08, 0.3) * (1 - seg(f, 0.4, 0.5)), sse: f > 0.95 ? R.sse : r > 0 ? run.rounds[r - 1].sse : null }
}

function Title({ p }: { p: number }) {
  const { t } = useTranslation()
  return <TitleCard p={p} kicker={t('watch.kmeans.kicker')} title={t('watch.kmeans.title')} sub={t('watch.kmeans.sub')} />
}

function DataScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const seeds = GOOD.rounds[0].centroids
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Plane />
        <Dots o={eseg(p, 0, 0.2)} />
        {seeds.map((c, k) => (p > 0.55 + k * 0.08 ? <Star key={k} at={P(c)} color={COLORS[k]} r={22 * eseg(p, 0.55 + k * 0.08, 0.62 + k * 0.08)} /> : null))}
      </Svg>
      <At x={1180} y={260} size={34} className="font-black" w={640} o={eseg(p, 0.15, 0.25)} style={{ textAlign: 'center' }}>{t('watch.kmeans.label.no_labels')}</At>
      <Tex f="k = 3" x={1180} y={420} size={48} color={C.yellow} o={eseg(p, 0.4, 0.5)} />
      <At x={1180} y={540} size={28} color={C.muted} w={600} o={eseg(p, 0.55, 0.65)} style={{ textAlign: 'center' }}>{t('watch.kmeans.label.seeds')}</At>
    </>
  )
}

function StepsScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const st = runState(GOOD, seg(p, 0.05, 0.95))
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Plane>
          <Regions cs={st.cs} o={eseg(p, 0.25, 0.35)} />
        </Plane>
        <Dots assign={st.assigned ? st.R.assign : null} links={{ cs: st.R.centroids, t: st.linkT }} />
        {st.R.centroids.map((c, k) => (p > 0.55 ? <circle key={'g' + k} cx={P(c)[0]} cy={P(c)[1]} r={16} fill="none" stroke={COLORS[k]} strokeWidth={2} strokeDasharray="4 4" /> : null))}
        {st.cs.map((c, k) => <Star key={k} at={P(c)} color={COLORS[k]} />)}
      </Svg>
      <At x={1180} y={200} size={28} color={C.muted} className="font-bold uppercase tracking-[0.15em]">{t('watch.kmeans.label.step_a')}</At>
      <At x={1180} y={260} size={30} w={640} style={{ textAlign: 'center' }} o={eseg(p, 0.05, 0.15)}>{t('watch.kmeans.label.assign')}</At>
      <At x={1180} y={440} size={28} color={C.muted} className="font-bold uppercase tracking-[0.15em]" o={eseg(p, 0.5, 0.58)}>{t('watch.kmeans.label.step_b')}</At>
      <At x={1180} y={500} size={30} w={640} style={{ textAlign: 'center' }} o={eseg(p, 0.5, 0.58)}>{t('watch.kmeans.label.update')}</At>
      <Tex f="c_k = \dfrac{1}{|S_k|}\sum_{x \in S_k} x" x={1180} y={640} size={40} color={C.yellow} o={eseg(p, 0.62, 0.72)} />
    </>
  )
}

function IterateScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const n = GOOD.rounds.length
  const st = runState(GOOD, 1 + seg(p, 0.02, 0.85) * (n - 1))
  const done = p > 0.86
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Plane>
          <Regions cs={st.cs} o={1} />
        </Plane>
        <Dots assign={st.R.assign} links={{ cs: st.R.centroids, t: st.linkT }} />
        {st.cs.map((c, k) => <Star key={k} at={P(c)} color={COLORS[k]} />)}
        {GOOD.rounds.map((R, i) => {
          const v = R.sse
          const x = 960 + i * 110
          const h = (v / GOOD.rounds[0].sse) * 280
          return i <= st.r ? <rect key={i} x={x} y={720 - h} width={70} height={h} rx={6} fill={i === st.r ? C.yellow : C.axis} /> : null
        })}
      </Svg>
      <At x={1180} y={190} size={30} className="font-mono">{t('watch.kmeans.label.round', { r: st.r + 1 })}</At>
      <At x={1180} y={250} size={26} color={C.muted}>{t('watch.kmeans.label.moved', { n: st.R.moved.length })}</At>
      <At x={1180} y={380} size={24} color={C.muted}>SSE</At>
      {GOOD.rounds.map((R, i) => (i <= st.r ? <At key={i} x={995 + i * 110} y={745} size={20} className="font-mono" color={C.muted}>{R.sse.toFixed(1)}</At> : null))}
      <At x={1180} y={820} size={30} color={C.teal} className="font-black" o={done ? eseg(p, 0.86, 0.92) : 0}>{t('watch.kmeans.label.converged')}</At>
    </>
  )
}

function WhyScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <Tex f="\text{SSE} = \sum_k \sum_{x \in S_k} \lVert x - c_k \rVert^2" x={800} y={150} size={46} o={eseg(p, 0, 0.1)} />
      <At x={800} y={300} size={32} w={1300} o={eseg(p, 0.15, 0.25)} style={{ textAlign: 'center' }}>
        <b style={{ color: C.yellow }}>{t('watch.kmeans.label.step_a')} · </b>
        {t('watch.kmeans.label.why_a')}
      </At>
      <At x={800} y={440} size={32} w={1300} o={eseg(p, 0.35, 0.45)} style={{ textAlign: 'center' }}>
        <b style={{ color: C.yellow }}>{t('watch.kmeans.label.step_b')} · </b>
        {t('watch.kmeans.label.why_b')}
      </At>
      <At x={800} y={600} size={32} w={1300} color={C.teal} o={eseg(p, 0.58, 0.68)} style={{ textAlign: 'center' }}>{t('watch.kmeans.label.why_stop')}</At>
    </>
  )
}

function LocalScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const st = runState(BAD, seg(p, 0.05, 0.6) * BAD.rounds.length)
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Plane>
          <Regions cs={st.cs} o={1} />
        </Plane>
        <Dots assign={st.assigned ? st.R.assign : null} links={{ cs: st.R.centroids, t: st.linkT }} />
        {st.cs.map((c, k) => <Star key={k} at={P(c)} color={COLORS[k]} />)}
      </Svg>
      <At x={1180} y={220} size={30} className="font-black" w={640} style={{ textAlign: 'center' }}>{t('watch.kmeans.label.other_seeds')}</At>
      <At x={1180} y={400} size={30} className="font-mono" o={eseg(p, 0.6, 0.7)}>
        SSE <b style={{ color: C.red }}>{BAD.rounds[BAD.rounds.length - 1].sse.toFixed(1)}</b>
        <span style={{ color: C.muted }}> {t('watch.kmeans.label.vs')} </span>
        <b style={{ color: C.teal }}>{GOOD.rounds[GOOD.rounds.length - 1].sse.toFixed(1)}</b>
      </At>
      <At x={1180} y={540} size={28} color={C.muted} w={640} o={eseg(p, 0.7, 0.8)} style={{ textAlign: 'center' }}>{t('watch.kmeans.label.stuck')}</At>
      <At x={1180} y={700} size={28} color={C.yellow} w={640} o={eseg(p, 0.82, 0.9)} style={{ textAlign: 'center' }}>{t('watch.kmeans.label.restarts')}</At>
    </>
  )
}

// Lecture 4's medicine example, as the exam table.
const MED = { names: ['A', 'B', 'C', 'D'], pts: [[1, 1], [2, 1], [4, 3], [5, 4]] }
const MED_RUN = kmeansTable(MED.pts, [[1, 1], [2, 1]], 'euclidean')
const fmt2 = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2))
function TableScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const r = p < 0.5 ? 0 : 1
  const R = MED_RUN.rounds[r]
  const rows = Math.floor(seg(p, r === 0 ? 0.08 : 0.55, r === 0 ? 0.35 : 0.8) * 4.99)
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <At x={800} y={100} size={30} color={C.muted}>{t('watch.kmeans.label.medicine')}</At>
      <At x={800} y={170} size={34} className="font-black">{t('watch.kmeans.label.round', { r: r + 1 })}</At>
      <At x={800} y={430} size={30} className="font-mono" w={1100}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'center' }}>
          <thead>
            <tr style={{ color: C.muted }}>
              <td>{t('watch.kmeans.label.point')}</td>
              {R.centroids.map((c, k) => <td key={k} style={{ color: COLORS[k] }}>d(·, C{k + 1}) <span style={{ fontSize: '0.75em' }}>({c.map(fmt2).join(', ')})</span></td>)}
              <td>{t('watch.kmeans.label.cluster')}</td>
            </tr>
          </thead>
          <tbody>
            {MED.pts.map((q, i) => (
              <tr key={i} style={{ height: '1.9em' }}>
                <td>{MED.names[i]} ({q.join(', ')})</td>
                {R.dist[i].map((d, k) => <td key={k}>{i <= rows ? Number(d.toFixed(2)) : ''}</td>)}
                <td style={{ color: COLORS[R.assign[i]], fontWeight: 800 }}>{i <= rows ? `C${R.assign[i] + 1}` : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </At>
      <At x={800} y={720} size={30} className="font-mono" o={eseg(p, r === 0 ? 0.36 : 0.82, r === 0 ? 0.44 : 0.9)}>
        {R.next.map((c, k) => (
          <Fragment key={k}>
            {k ? '   ' : ''}
            <span style={{ color: COLORS[k] }}>C{k + 1}′ = ({c.map(fmt2).join(', ')})</span>
          </Fragment>
        ))}
      </At>
      <At x={800} y={800} size={26} color={C.teal} o={r === 1 ? eseg(p, 0.9, 0.96) : 0}>{t('watch.kmeans.label.table_done')}</At>
    </>
  )
}

function ElbowScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const ex = (k: number) => 220 + (k - 1) * 220
  const ey = (v: number) => 760 - (v / ELBOW[0]) * 520
  const n = Math.min(6, 1 + Math.floor(seg(p, 0.08, 0.6) * 6))
  const pts = ELBOW.slice(0, n).map((v, i) => `${ex(i + 1)},${ey(v)}`).join(' ')
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <line x1={160} y1={760} x2={1400} y2={760} stroke={C.axis} strokeWidth={2} />
        <line x1={160} y1={760} x2={160} y2={200} stroke={C.axis} strokeWidth={2} />
        <polyline points={pts} fill="none" stroke={C.blue} strokeWidth={4} />
        {ELBOW.slice(0, n).map((v, i) => <circle key={i} cx={ex(i + 1)} cy={ey(v)} r={i === 2 && p > 0.65 ? 18 : 10} fill={i === 2 && p > 0.65 ? C.yellow : C.blue} />)}
      </Svg>
      {ELBOW.slice(0, n).map((v, i) => (
        <Fragment key={i}>
          <At x={ex(i + 1)} y={800} size={26} className="font-mono" color={C.muted}>k = {i + 1}</At>
          <At x={ex(i + 1) + 16} y={ey(v) - 34} size={22} className="font-mono" anchor="l">{v.toFixed(0)}</At>
        </Fragment>
      ))}
      <At x={180} y={185} size={24} color={C.muted} anchor="l">SSE</At>
      <At x={1000} y={300} size={30} color={C.yellow} w={700} o={eseg(p, 0.65, 0.75)} style={{ textAlign: 'center' }}>{t('watch.kmeans.label.elbow')}</At>
      <At x={1000} y={420} size={26} color={C.muted} w={700} o={eseg(p, 0.78, 0.88)} style={{ textAlign: 'center' }}>{t('watch.kmeans.label.not_min')}</At>
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
          {t(`watch.kmeans.recap.${k}`)}
        </At>
      ))}
      <At x={800} y={800} size={28} color={C.muted} o={eseg(p, 0.75, 0.85)}>{t('watch.ui.try_it')}</At>
    </>
  )
}

export const kmeans: Episode = {
  id: 'kmeans',
  scenes: [
    { id: 'title', dur: 5000, cues: [0], render: (p) => <Title p={p} /> },
    { id: 'data', dur: 12000, cues: [0, 0.4, 0.6], render: (p) => <DataScene p={p} /> },
    { id: 'steps', dur: 16000, cues: [0, 0.25, 0.5, 0.7], render: (p) => <StepsScene p={p} /> },
    { id: 'iterate', dur: 20000, cues: [0, 0.4, 0.86], ponder: true, render: (p) => <IterateScene p={p} /> },
    { id: 'why', dur: 15000, cues: [0, 0.15, 0.35, 0.58], render: (p) => <WhyScene p={p} /> },
    { id: 'local', dur: 15000, cues: [0, 0.6, 0.82], render: (p) => <LocalScene p={p} /> },
    { id: 'table', dur: 20000, cues: [0, 0.36, 0.55, 0.9], render: (p) => <TableScene p={p} /> },
    { id: 'elbow', dur: 14000, cues: [0, 0.65, 0.8], render: (p) => <ElbowScene p={p} /> },
    { id: 'recap', dur: 12000, cues: [0, 0.6], render: (p) => <Recap p={p} /> },
  ],
}
