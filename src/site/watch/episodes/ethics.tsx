/* eslint-disable react-refresh/only-export-components -- episode data plus its scene components */
import { Fragment } from 'react'
import { useTranslation } from 'react-i18next'
import { eseg, lerp, seg } from '../../../lib/explainer'
import type { Episode } from '../Player'
import { At, C, Grid, Svg, TitleCard } from '../stage'

// "AI ethics for the final" — lecture 11; a 3–4 point ethics question closes every final.

const EU = ['A', 'B', 'C', 'D', 'E', 'F', 'G'] as const
const GROUNDS = [1, 2, 3, 4, 5] as const
/** Lecture 11 practice problem, official answer. */
const MAP: Record<(typeof EU)[number], number> = { A: 1, B: 3, C: 1, D: 5, E: 4, F: 2, G: 1 }
const EU_Y = (i: number) => 190 + i * 88
const G_Y = (i: number) => 230 + i * 120
const EU_X = 610
const G_X = 1010
const G_COL = [C.blue, C.teal, C.green, C.yellow, C.purple]

function Title({ p }: { p: number }) {
  const { t } = useTranslation()
  return <TitleCard p={p} kicker={t('watch.ethics.kicker')} title={t('watch.ethics.title')} sub={t('watch.ethics.sub')} />
}

function AreasScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const cols = [C.blue, C.yellow, C.teal]
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        {[0, 1, 2].map((i) => (
          <rect key={i} x={120 + i * 470} y={300} width={420} height={360} rx={24} fill={cols[i]} fillOpacity={0.08} stroke={cols[i]} strokeWidth={3} opacity={eseg(p, 0.1 + i * 0.2, 0.2 + i * 0.2)} />
        ))}
      </Svg>
      <At x={800} y={170} size={32} className="font-black" w={1300} style={{ textAlign: 'center' }} o={eseg(p, 0, 0.08)}>{t('watch.ethics.label.areas_q')}</At>
      {[1, 2, 3].map((k, i) => (
        <Fragment key={k}>
          <At x={330 + i * 470} y={360} size={34} className="font-black" color={cols[i]} w={380} style={{ textAlign: 'center' }} o={eseg(p, 0.1 + i * 0.2, 0.2 + i * 0.2)}>{t(`watch.ethics.area.${k}.name`)}</At>
          <At x={330 + i * 470} y={500} size={24} w={360} color={C.text} style={{ textAlign: 'center' }} o={eseg(p, 0.15 + i * 0.2, 0.25 + i * 0.2)}>{t(`watch.ethics.area.${k}.what`)}</At>
        </Fragment>
      ))}
      <At x={800} y={760} size={26} color={C.yellow} o={eseg(p, 0.8, 0.9)}>{t('watch.ethics.label.areas_mark')}</At>
    </>
  )
}

function Columns({ p, lines }: { p: number; lines: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        {EU.map((e, i) => {
          const g = MAP[e] - 1
          const k = seg(lines, i / 7, (i + 1) / 7)
          if (k <= 0) return null
          const x2 = lerp(EU_X + 14, G_X - 14, k)
          const y2 = lerp(EU_Y(i), G_Y(g), k)
          return <line key={e} x1={EU_X + 14} y1={EU_Y(i)} x2={x2} y2={y2} stroke={G_COL[g]} strokeWidth={4} />
        })}
        {EU.map((e, i) => <circle key={e} cx={EU_X} cy={EU_Y(i)} r={12} fill={C.bg} stroke={C.muted} strokeWidth={3} opacity={eseg(p, 0.02 + i * 0.03, 0.08 + i * 0.03)} />)}
        {GROUNDS.map((g, i) => <circle key={g} cx={G_X} cy={G_Y(i)} r={14} fill={G_COL[i]} opacity={eseg(p, 0.25 + i * 0.03, 0.31 + i * 0.03)} />)}
      </Svg>
      {EU.map((e, i) => (
        <At key={e} x={EU_X - 30} y={EU_Y(i)} size={24} anchor="r" o={eseg(p, 0.02 + i * 0.03, 0.08 + i * 0.03)}>
          <b style={{ color: C.muted }}>({e}) </b>
          {t(`watch.ethics.eu.${e}`)}
        </At>
      ))}
      {GROUNDS.map((g, i) => (
        <At key={g} x={G_X + 34} y={G_Y(i)} size={28} anchor="l" color={G_COL[i]} className="font-black" o={eseg(p, 0.25 + i * 0.03, 0.31 + i * 0.03)}>
          ({g}) {t(`watch.ethics.ground.${g}`)}
        </At>
      ))}
    </>
  )
}

function PrinciplesScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Columns p={p} lines={0} />
      <At x={800} y={100} size={28} className="font-black" w={1400} style={{ textAlign: 'center' }} o={eseg(p, 0.4, 0.5)}>{t('watch.ethics.label.match_q')}</At>
    </>
  )
}

function MatchScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Columns p={1} lines={seg(p, 0.05, 0.75)} />
      <At x={800} y={100} size={26} className="font-mono" w={1400} style={{ textAlign: 'center' }} o={eseg(p, 0.78, 0.86)}>
        A1 · B3 · C1 · D5 · E4 · F2 · G1
      </At>
      <At x={1280} y={840} size={22} color={C.muted} w={560} o={eseg(p, 0.82, 0.9)} style={{ textAlign: 'center' }}>{t('watch.ethics.label.surprise')}</At>
    </>
  )
}

const CAUSES = ['human', 'loop', 'sample', 'unreliable', 'proxy', 'culture'] as const
function CausesScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        {CAUSES.map((c, i) => (
          <rect key={c} x={110 + (i % 3) * 470} y={190 + Math.floor(i / 3) * 330} width={440} height={300} rx={20} fill={C.red} fillOpacity={0.06} stroke={C.red} strokeOpacity={0.6} strokeWidth={2} opacity={eseg(p, 0.05 + i * 0.13, 0.12 + i * 0.13)} />
        ))}
      </Svg>
      <At x={800} y={120} size={30} className="font-black" o={eseg(p, 0, 0.06)}>{t('watch.ethics.label.causes')}</At>
      {CAUSES.map((c, i) => (
        <Fragment key={c}>
          <At x={330 + (i % 3) * 470} y={240 + Math.floor(i / 3) * 330} size={28} className="font-black" color={C.red} o={eseg(p, 0.05 + i * 0.13, 0.12 + i * 0.13)}>{t(`watch.ethics.cause.${c}.name`)}</At>
          <At x={330 + (i % 3) * 470} y={360 + Math.floor(i / 3) * 330} size={21} w={390} style={{ textAlign: 'center' }} o={eseg(p, 0.07 + i * 0.13, 0.14 + i * 0.13)}>{t(`watch.ethics.cause.${c}.eg`)}</At>
        </Fragment>
      ))}
    </>
  )
}

function LoopScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const nodes: [number, number][] = [[800, 230], [1180, 470], [800, 710], [420, 470]]
  const a = eseg(p, 0.05, 0.85) * 2 * Math.PI * 1.5
  const dot: [number, number] = [800 + 380 * Math.sin(a), 470 - 240 * Math.cos(a)]
  return (
    <>
      <Svg>
        <Grid o={0.15} />
        <ellipse cx={800} cy={470} rx={380} ry={240} fill="none" stroke={C.red} strokeWidth={3} strokeDasharray="10 8" />
        <circle cx={dot[0]} cy={dot[1]} r={14} fill={C.yellow} />
      </Svg>
      {nodes.map(([x, y], i) => (
        <At key={i} x={x} y={y} size={26} w={330} className="font-bold" style={{ textAlign: 'center', background: C.bg, padding: '0.4em', borderRadius: '0.6em', border: `2px solid ${C.axis}` }} o={eseg(p, 0.05 + i * 0.1, 0.12 + i * 0.1)}>
          {t(`watch.ethics.loop.${i + 1}`)}
        </At>
      ))}
      <At x={800} y={850} size={24} color={C.yellow} w={1300} o={eseg(p, 0.75, 0.85)} style={{ textAlign: 'center' }}>{t('watch.ethics.label.loop_note')}</At>
    </>
  )
}

function MoralScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.15} />
      </Svg>
      <At x={800} y={140} size={32} className="font-black" w={1300} style={{ textAlign: 'center' }} o={eseg(p, 0, 0.08)}>{t('watch.ethics.label.moral_q')}</At>
      {[1, 2, 3, 4].map((k) => (
        <At key={k} x={300} y={240 + k * 95} size={30} anchor="l" w={1000} o={eseg(p, 0.1 + k * 0.1, 0.17 + k * 0.1)}>
          <span style={{ color: C.teal, fontWeight: 900 }}>✓ </span>
          {t(`watch.ethics.moral.${k}`)}
        </At>
      ))}
      <At x={800} y={780} size={24} color={C.muted} w={1300} o={eseg(p, 0.65, 0.75)} style={{ textAlign: 'center' }}>{t('watch.ethics.label.trolley')}</At>
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
          {t(`watch.ethics.recap.${k}`)}
        </At>
      ))}
    </>
  )
}

export const ethics: Episode = {
  id: 'ethics',
  scenes: [
    { id: 'title', dur: 5000, cues: [0], render: (p) => <Title p={p} /> },
    { id: 'areas', dur: 16000, cues: [0, 0.1, 0.3, 0.5, 0.8], render: (p) => <AreasScene p={p} /> },
    { id: 'principles', dur: 12000, cues: [0, 0.25, 0.45], ponder: true, render: (p) => <PrinciplesScene p={p} /> },
    { id: 'match', dur: 16000, cues: [0, 0.3, 0.78], render: (p) => <MatchScene p={p} /> },
    { id: 'causes', dur: 24000, cues: [0, 0.05, 0.31, 0.57, 0.83], render: (p) => <CausesScene p={p} /> },
    { id: 'loop', dur: 14000, cues: [0, 0.4, 0.75], render: (p) => <LoopScene p={p} /> },
    { id: 'moral', dur: 13000, cues: [0, 0.2, 0.65], render: (p) => <MoralScene p={p} /> },
    { id: 'recap', dur: 11000, cues: [0], render: (p) => <Recap p={p} /> },
  ],
}
