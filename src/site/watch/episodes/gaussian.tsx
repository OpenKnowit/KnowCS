/* eslint-disable react-refresh/only-export-components -- episode data plus its scene components */
import { useTranslation } from 'react-i18next'
import { eseg, lerp } from '../../../lib/explainer'
import { gaussianPdf, meanStd } from '../../../lib/gaussianNb'
import type { Episode } from '../Player'
import { At, C, Grid, Svg, Tex, TitleCard } from '../stage'

// "Naive Bayes for continuous features" — lecture 2's Gaussian likelihoods (disease Z with BP = 66, fever = 97).

const BP = { yes: { mu: 73, s: 6.2 }, no: { mu: 75, s: 7.9 }, x: 66 }
const FE = { yes: { mu: 79, s: 10.2 }, no: { mu: 86, s: 9.7 }, x: 97 }
const L = {
  bpY: gaussianPdf(BP.x, BP.yes.mu, BP.yes.s),
  bpN: gaussianPdf(BP.x, BP.no.mu, BP.no.s),
  feY: gaussianPdf(FE.x, FE.yes.mu, FE.yes.s),
  feN: gaussianPdf(FE.x, FE.no.mu, FE.no.s),
}
const YES = (9 / 14) * L.bpY * L.feY * (3 / 9) * (3 / 9)
const NO = (5 / 14) * L.bpN * L.feN * (4 / 5) * (3 / 5)
const TEMPS = [39.5, 39.0, 38.2]
const SAMPLE = meanStd(TEMPS, 1)
const POP = meanStd(TEMPS, 0)
const sci = (v: number) => v.toExponential(2).replace(/e([+-])(\d+)/, (_m, s, d) => ` \\times 10^{${s === '-' ? '-' : ''}${d}}`)

/** Two class curves over [lo, hi] with a probe line at x. */
function Curves({ f, lo, hi, x, probe, label }: { f: typeof BP; lo: number; hi: number; x: number; probe: number; label: string }) {
  const X = (v: number) => 140 + ((v - lo) / (hi - lo)) * 880
  const peak = Math.max(gaussianPdf(f.yes.mu, f.yes.mu, f.yes.s), gaussianPdf(f.no.mu, f.no.mu, f.no.s))
  const Y = (v: number) => 720 - (v / peak) * 440
  const path = (c: { mu: number; s: number }) =>
    Array.from({ length: 161 }, (_, k) => {
      const v = lo + ((hi - lo) * k) / 160
      return `${X(v)},${Y(gaussianPdf(v, c.mu, c.s))}`
    }).join(' ')
  const hy = gaussianPdf(x, f.yes.mu, f.yes.s)
  const hn = gaussianPdf(x, f.no.mu, f.no.s)
  return (
    <g>
      <line x1={140} y1={720} x2={1020} y2={720} stroke={C.axis} strokeWidth={2} />
      {Array.from({ length: 6 }, (_, i) => lo + ((hi - lo) * i) / 5).map((v) => (
        <text key={v} x={X(v)} y={752} textAnchor="middle" fill={C.muted} fontSize={22}>{Math.round(v)}</text>
      ))}
      <text x={1020} y={790} textAnchor="end" fill={C.muted} fontSize={22}>{label}</text>
      <polyline points={path(f.yes)} fill="none" stroke={C.teal} strokeWidth={4} />
      <polyline points={path(f.no)} fill="none" stroke={C.purple} strokeWidth={4} />
      <line x1={X(f.yes.mu)} y1={720} x2={X(f.yes.mu)} y2={Y(gaussianPdf(f.yes.mu, f.yes.mu, f.yes.s))} stroke={C.teal} strokeDasharray="5 5" strokeWidth={1.5} />
      <line x1={X(f.no.mu)} y1={720} x2={X(f.no.mu)} y2={Y(gaussianPdf(f.no.mu, f.no.mu, f.no.s))} stroke={C.purple} strokeDasharray="5 5" strokeWidth={1.5} />
      {probe > 0 && (
        <g opacity={probe}>
          <line x1={X(x)} y1={720} x2={X(x)} y2={260} stroke={C.yellow} strokeWidth={3} />
          <circle cx={X(x)} cy={Y(hy)} r={10} fill={C.teal} />
          <circle cx={X(x)} cy={Y(hn)} r={10} fill={C.purple} />
        </g>
      )}
    </g>
  )
}

function Title({ p }: { p: number }) {
  const { t } = useTranslation()
  return <TitleCard p={p} kicker={t('watch.gaussian.kicker')} title={t('watch.gaussian.title')} sub={t('watch.gaussian.sub')} />
}

function ProblemScene({ p }: { p: number }) {
  const { t } = useTranslation()
  // an illustration with μ = 73 and sample σ ≈ 6.18 — the lecture gives only μ and σ
  const vals = [63, 67, 69, 71, 73, 75, 77, 80, 82]
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <line x1={160} y1={500} x2={1440} y2={500} stroke={C.axis} strokeWidth={3} />
        {vals.map((v, i) => <circle key={i} cx={160 + (v - 55) * 40} cy={500} r={14} fill={C.teal} opacity={eseg(p, 0.05 + i * 0.03, 0.12 + i * 0.03)} />)}
        <g opacity={eseg(p, 0.45, 0.55)}>
          <line x1={160 + 11 * 40} y1={400} x2={160 + 11 * 40} y2={560} stroke={C.yellow} strokeWidth={4} />
        </g>
      </Svg>
      {[55, 60, 65, 70, 75, 80, 85].map((v) => <At key={v} x={160 + (v - 55) * 40} y={545} size={22} color={C.muted}>{v}</At>)}
      <At x={800} y={200} size={32} className="font-black" w={1300} style={{ textAlign: 'center' }} o={eseg(p, 0, 0.1)}>{t('watch.gaussian.label.continuous')}</At>
      <At x={160 + 11 * 40} y={370} size={28} color={C.yellow} o={eseg(p, 0.45, 0.55)}>BP = 66 ?</At>
      <At x={800} y={680} size={30} color={C.red} w={1300} o={eseg(p, 0.6, 0.7)} style={{ textAlign: 'center' }}>{t('watch.gaussian.label.count_zero')}</At>
    </>
  )
}

function BellScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const g = eseg(p, 0.2, 0.5)
  const X = (v: number) => 160 + (v - 50) * 18
  const curve = Array.from({ length: 121 }, (_, k) => {
    const v = 50 + k * 0.5
    return `${X(v)},${720 - gaussianPdf(v, 73, 6.2) * 440 / gaussianPdf(73, 73, 6.2) * g}`
  }).join(' ')
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <line x1={160} y1={720} x2={1240} y2={720} stroke={C.axis} strokeWidth={2} />
        <polyline points={curve} fill={C.teal} fillOpacity={0.1} stroke={C.teal} strokeWidth={4} />
        <line x1={X(73)} y1={720} x2={X(73)} y2={280} stroke={C.yellow} strokeWidth={2} strokeDasharray="6 6" opacity={eseg(p, 0.5, 0.6)} />
        <line x1={X(73 - 6.2)} y1={600} x2={X(73 + 6.2)} y2={600} stroke={C.orange} strokeWidth={3} opacity={eseg(p, 0.6, 0.7)} />
      </Svg>
      <Tex f="f(x) = \dfrac{1}{\sqrt{2\pi}\,\sigma}\exp\!\left(-\dfrac{(x - \mu)^2}{2\sigma^2}\right)" x={1300} y={200} size={34} o={eseg(p, 0, 0.12)} />
      <Tex f="\mu = 73" x={X(73)} y={250} size={32} color={C.yellow} o={eseg(p, 0.5, 0.6)} />
      <Tex f="\sigma = 6.2" x={X(73 + 6.2) + 70} y={600} size={28} color={C.orange} o={eseg(p, 0.6, 0.7)} />
      <At x={1300} y={400} size={26} w={500} color={C.muted} o={eseg(p, 0.7, 0.8)} style={{ textAlign: 'center' }}>{t('watch.gaussian.label.per_class')}</At>
    </>
  )
}

function BpScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Curves f={BP} lo={45} hi={105} x={BP.x} probe={eseg(p, 0.3, 0.4)} label={t('watch.gaussian.label.bp')} />
      </Svg>
      <At x={1300} y={180} size={26} className="font-mono" color={C.teal}>Yes: N(73, 6.2)</At>
      <At x={1300} y={225} size={26} className="font-mono" color={C.purple}>No: N(75, 7.9)</At>
      <Tex f={`f(66 \\mid \\text{Yes}) = ${L.bpY.toFixed(4)}`} x={1300} y={380} size={32} color={C.teal} o={eseg(p, 0.42, 0.52)} />
      <Tex f={`f(66 \\mid \\text{No}) = ${L.bpN.toFixed(4)}`} x={1300} y={460} size={32} color={C.purple} o={eseg(p, 0.5, 0.6)} />
      <At x={1300} y={600} size={26} w={500} color={C.muted} o={eseg(p, 0.7, 0.8)} style={{ textAlign: 'center' }}>{t('watch.gaussian.label.read_height')}</At>
    </>
  )
}

function FeverScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <Curves f={FE} lo={50} hi={120} x={FE.x} probe={eseg(p, 0.2, 0.3)} label={t('watch.gaussian.label.fever')} />
      </Svg>
      <At x={1300} y={180} size={26} className="font-mono" color={C.teal}>Yes: N(79, 10.2)</At>
      <At x={1300} y={225} size={26} className="font-mono" color={C.purple}>No: N(86, 9.7)</At>
      <Tex f={`f(97 \\mid \\text{Yes}) = ${L.feY.toFixed(5)}`} x={1300} y={380} size={32} color={C.teal} o={eseg(p, 0.32, 0.42)} />
      <Tex f={`f(97 \\mid \\text{No}) = ${L.feN.toFixed(5)}`} x={1300} y={460} size={32} color={C.purple} o={eseg(p, 0.4, 0.5)} />
      <At x={1300} y={600} size={26} w={500} color={C.muted} o={eseg(p, 0.6, 0.7)} style={{ textAlign: 'center' }}>{t('watch.gaussian.label.fever_far')}</At>
    </>
  )
}

function CombineScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <At x={800} y={130} size={28} color={C.muted} w={1300} style={{ textAlign: 'center' }} o={eseg(p, 0, 0.1)}>{t('watch.gaussian.label.mix')}</At>
      <Tex f={`\\text{Yes}: \\tfrac{9}{14} \\times ${L.bpY.toFixed(4)} \\times ${L.feY.toFixed(5)} \\times \\tfrac{3}{9} \\times \\tfrac{3}{9} = ${sci(YES)}`} x={800} y={300} size={34} color={C.teal} o={eseg(p, 0.1, 0.25)} />
      <Tex f={`\\text{No}: \\tfrac{5}{14} \\times ${L.bpN.toFixed(4)} \\times ${L.feN.toFixed(5)} \\times \\tfrac{4}{5} \\times \\tfrac{3}{5} = ${sci(NO)}`} x={800} y={440} size={34} color={C.purple} o={eseg(p, 0.3, 0.45)} />
      <At x={800} y={580} size={34} o={eseg(p, 0.55, 0.65)}>
        {t('watch.gaussian.label.normalised')} <b style={{ color: C.teal }}>{((YES / (YES + NO)) * 100).toFixed(0)}%</b> / <b style={{ color: C.purple }}>{((NO / (YES + NO)) * 100).toFixed(0)}%</b>
      </At>
      <At x={800} y={700} size={28} color={C.yellow} w={1300} o={eseg(p, 0.7, 0.8)} style={{ textAlign: 'center' }}>{t('watch.gaussian.label.no_wins')}</At>
    </>
  )
}

function DensityScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const s = lerp(1.5, 0.1, eseg(p, 0.15, 0.6))
  const X = (v: number) => 800 + v * 260
  const Y = (v: number) => 700 - v * 110
  const curve = Array.from({ length: 161 }, (_, k) => {
    const v = -2 + k * 0.025
    return `${X(v)},${Math.max(140, Y(gaussianPdf(v, 0, s)))}`
  }).join(' ')
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        <line x1={260} y1={700} x2={1340} y2={700} stroke={C.axis} strokeWidth={2} />
        <line x1={260} y1={Y(1)} x2={1340} y2={Y(1)} stroke={C.red} strokeWidth={2} strokeDasharray="8 6" />
        <polyline points={curve} fill={C.teal} fillOpacity={0.12} stroke={C.teal} strokeWidth={4} />
      </Svg>
      <At x={250} y={Y(1)} size={24} anchor="r" color={C.red}>1</At>
      <Tex f={`\\sigma = ${s.toFixed(2)}, \\quad f(\\mu) = ${gaussianPdf(0, 0, s).toFixed(2)}`} x={800} y={110} size={36} />
      <At x={800} y={800} size={28} w={1300} color={C.yellow} o={eseg(p, 0.65, 0.75)} style={{ textAlign: 'center' }}>{t('watch.gaussian.label.density')}</At>
    </>
  )
}

function SigmaScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <At x={800} y={140} size={30} className="font-black" o={eseg(p, 0, 0.1)}>{t('watch.gaussian.label.sigma_trap')}</At>
      <At x={800} y={240} size={28} className="font-mono" o={eseg(p, 0.08, 0.18)}>{t('watch.gaussian.label.temps')}: {TEMPS.join(', ')}</At>
      <Tex f={`\\mu = \\tfrac{39.5 + 39.0 + 38.2}{3} = ${SAMPLE.mu.toFixed(2)}`} x={800} y={340} size={34} o={eseg(p, 0.18, 0.28)} />
      <Tex f={`s = \\sqrt{\\tfrac{\\sum (x - \\mu)^2}{n - 1}} = ${SAMPLE.sigma.toFixed(4)}`} x={800} y={470} size={34} color={C.teal} o={eseg(p, 0.35, 0.45)} />
      <Tex f={`\\sigma_{\\div n} = \\sqrt{\\tfrac{\\sum (x - \\mu)^2}{n}} = ${POP.sigma.toFixed(4)}`} x={800} y={600} size={30} color={C.muted} o={eseg(p, 0.5, 0.6)} />
      <At x={800} y={740} size={28} color={C.yellow} w={1300} o={eseg(p, 0.65, 0.75)} style={{ textAlign: 'center' }}>{t('watch.gaussian.label.which_sigma')}</At>
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
          {t(`watch.gaussian.recap.${k}`)}
        </At>
      ))}
      <At x={800} y={800} size={28} color={C.muted} o={eseg(p, 0.75, 0.85)}>{t('watch.ui.try_it')}</At>
    </>
  )
}

export const gaussian: Episode = {
  id: 'gaussian',
  scenes: [
    { id: 'title', dur: 5000, cues: [0], render: (p) => <Title p={p} /> },
    { id: 'problem', dur: 12000, cues: [0, 0.45, 0.6], render: (p) => <ProblemScene p={p} /> },
    { id: 'bell', dur: 14000, cues: [0, 0.2, 0.5, 0.7], render: (p) => <BellScene p={p} /> },
    { id: 'bp', dur: 14000, cues: [0, 0.3, 0.7], render: (p) => <BpScene p={p} /> },
    { id: 'fever', dur: 12000, cues: [0, 0.3, 0.6], render: (p) => <FeverScene p={p} /> },
    { id: 'combine', dur: 16000, cues: [0, 0.1, 0.55, 0.7], render: (p) => <CombineScene p={p} /> },
    { id: 'density', dur: 12000, cues: [0, 0.3, 0.65], ponder: true, render: (p) => <DensityScene p={p} /> },
    { id: 'sigma', dur: 14000, cues: [0, 0.35, 0.65], render: (p) => <SigmaScene p={p} /> },
    { id: 'recap', dur: 12000, cues: [0, 0.6], render: (p) => <Recap p={p} /> },
  ],
}
