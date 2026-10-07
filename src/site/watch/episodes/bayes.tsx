/* eslint-disable react-refresh/only-export-components -- episode data plus its scene components */
import { Fragment, useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { eseg, lerp, seg } from '../../../lib/explainer'
import { bayesPosterior, naturalCounts } from '../../../lib/bayesRule'
import { computeNaiveBayes } from '../../../lib/bayes'
import { mulberry32 } from '../../../lib/crossval'
import { BAYES_DATA } from '../../../data/constants'
import type { BayesFeature } from '../../../types'
import type { Episode } from '../Player'
import { At, C, Grid, Svg, Tex, TitleCard } from '../stage'

// "A 99% accurate test, and you are probably still healthy" — lecture 2; Bayes / naive Bayes is in 8 of the 9 papers.

const PRIOR = 0.001
const HIT = 0.99
const SPEC = 0.95
const N = 10000
const COUNTS = naturalCounts(PRIOR, HIT, 1 - SPEC, N) // tp 10, fn 0, fp 500, tn 9490
const POS = COUNTS.tp + COUNTS.fp
const ONE = bayesPosterior({ prior: PRIOR, hit: HIT, falseAlarm: 1 - SPEC, observed: 'E' })
const TWO = bayesPosterior({ prior: ONE.posterior, hit: HIT, falseAlarm: 1 - SPEC, observed: 'E' })
const THREE = bayesPosterior({ prior: TWO.posterior, hit: HIT, falseAlarm: 1 - SPEC, observed: 'E' })

// Who is who in the 100 × 100 crowd: fixed by a seeded shuffle.
const ROLE: Uint8Array = (() => {
  const r = mulberry32(2211)
  const idx = Array.from({ length: N }, (_, i) => i)
  for (let i = N - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1))
    ;[idx[i], idx[j]] = [idx[j], idx[i]]
  }
  const role = new Uint8Array(N) // 0 healthy, 1 infected & positive, 2 infected & negative, 3 healthy & positive
  idx.slice(0, COUNTS.tp).forEach((i) => (role[i] = 1))
  idx.slice(COUNTS.tp, COUNTS.tp + COUNTS.fn).forEach((i) => (role[i] = 2))
  idx.slice(COUNTS.tp + COUNTS.fn, COUNTS.tp + COUNTS.fn + COUNTS.fp).forEach((i) => (role[i] = 3))
  return role
})()
// Order in the "positives" block: infected first, so they sit together at the top-left.
const POS_ORDER: Map<number, number> = (() => {
  const ids = [...ROLE.keys()].filter((i) => ROLE[i] === 1)
  const fps = [...ROLE.keys()].filter((i) => ROLE[i] === 3)
  return new Map([...ids, ...fps].map((id, k) => [id, k]))
})()

const CROWD = { x: 110, y: 130, size: 640 } // stage px
const BLOCK_COLS = 30

/** The 10,000-person crowd on a canvas (redrawn every frame from the scene state). */
function Crowd({ infected, tested, gather, dimNeg }: { infected: number; tested: number; gather: number; dimNeg: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    const S = 1000
    c.width = S
    c.height = S
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, S, S)
    const cell = S / 100
    const r = cell * 0.36
    for (let i = 0; i < N; i++) {
      const role = ROLE[i]
      const isInf = role === 1 || role === 2
      const isPos = role === 1 || role === 3
      const gx = (i % 100) * cell + cell / 2
      const gy = Math.floor(i / 100) * cell + cell / 2
      let x = gx
      let y = gy
      let col = '#3b4556'
      let alpha = 1
      if (isInf && infected > 0) col = '#fc6255'
      if (isPos && tested > (i % 997) / 997) col = isInf ? '#fc6255' : '#ffd866'
      if (!isPos && dimNeg > 0) alpha = 1 - 0.8 * dimNeg
      if (isPos && gather > 0) {
        const k = POS_ORDER.get(i)!
        const bx = 520 + (k % BLOCK_COLS) * 15
        const by = 300 + Math.floor(k / BLOCK_COLS) * 15
        x = lerp(gx, bx, gather)
        y = lerp(gy, by, gather)
      }
      const big = isInf && infected > 0 ? 1.5 + 1.4 * Math.max(0, 1 - Math.abs(infected - 0.5) * 2) : 1
      ctx.globalAlpha = alpha
      ctx.fillStyle = col
      ctx.beginPath()
      ctx.arc(x, y, r * big, 0, Math.PI * 2)
      ctx.fill()
      if (isInf && infected > 0) {
        // a ring so ten people stay visible among ten thousand
        ctx.strokeStyle = '#fc6255'
        ctx.lineWidth = 3
        ctx.globalAlpha = Math.min(1, infected * 2) * (1 - 0.6 * gather)
        ctx.beginPath()
        ctx.arc(x, y, cell * 1.6, 0, Math.PI * 2)
        ctx.stroke()
      }
    }
    ctx.globalAlpha = 1
  })
  return (
    <canvas
      ref={ref}
      aria-hidden
      className="absolute"
      style={{ left: `${(CROWD.x / 1600) * 100}%`, top: `${(CROWD.y / 900) * 100}%`, width: `${(CROWD.size / 1600) * 100}%`, height: `${(CROWD.size / 900) * 100}%` }}
    />
  )
}

const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`

function Bar({ x, y, w, h, frac, color, o = 1 }: { x: number; y: number; w: number; h: number; frac: number; color: string; o?: number }) {
  return (
    <g opacity={o}>
      <rect x={x} y={y} width={w} height={h} rx={h / 2} fill={C.grid} />
      <rect x={x} y={y} width={Math.max(h, w * frac)} height={h} rx={h / 2} fill={color} />
    </g>
  )
}

// ------------------------------------------------------------------ scenes

function Title({ p }: { p: number }) {
  const { t } = useTranslation()
  return <TitleCard p={p} kicker={t('watch.bayes.kicker')} title={t('watch.bayes.title')} sub={t('watch.bayes.sub')} />
}

function Question({ p }: { p: number }) {
  const { t } = useTranslation()
  const facts: [string, string, string, number][] = [
    ['0.1%', t('watch.bayes.label.have_it'), C.red, 0.08],
    ['99%', t('watch.bayes.label.hit'), C.blue, 0.24],
    ['95%', t('watch.bayes.label.spec'), C.teal, 0.4],
  ]
  return (
    <>
      <Svg>
        <Grid o={0.3} />
      </Svg>
      {facts.map(([big, small, col, at], i) => (
        <Fragment key={i}>
          <At x={300 + i * 500} y={330} size={96} color={col} className="font-black font-mono" o={eseg(p, at, at + 0.1)}>{big}</At>
          <At x={300 + i * 500} y={440} size={28} color={C.muted} w={420} o={eseg(p, at, at + 0.1)} style={{ textAlign: 'center' }}>{small}</At>
        </Fragment>
      ))}
      <At x={800} y={650} size={46} className="font-black" o={eseg(p, 0.6, 0.7)}>
        {t('watch.bayes.label.positive_q')}
      </At>
    </>
  )
}

function CrowdScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const appear = eseg(p, 0, 0.25)
  const inf = seg(p, 0.45, 0.75)
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <div className="absolute inset-0" style={{ opacity: appear }}>
        <Crowd infected={inf} tested={0} gather={0} dimNeg={0} />
      </div>
      <At x={1180} y={300} size={110} className="font-black font-mono" o={appear}>10,000</At>
      <At x={1180} y={390} size={30} color={C.muted} o={appear}>{t('watch.bayes.label.people')}</At>
      <At x={1180} y={540} size={88} className="font-black font-mono" color={C.red} o={eseg(p, 0.45, 0.55)}>{COUNTS.tp + COUNTS.fn}</At>
      <At x={1180} y={620} size={30} color={C.muted} o={eseg(p, 0.45, 0.55)}>{t('watch.bayes.label.infected', { pct: '0.1%' })}</At>
    </>
  )
}

function TestScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const tested = seg(p, 0.15, 0.65)
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <Crowd infected={1} tested={tested} gather={0} dimNeg={0} />
      <At x={1180} y={220} size={30} color={C.muted}>{t('watch.bayes.label.everyone_tested')}</At>
      <At x={1180} y={330} size={30} color={C.red} o={eseg(p, 0.2, 0.3)}>{t('watch.bayes.label.inf_pos', { n: COUNTS.tp, of: COUNTS.tp + COUNTS.fn })}</At>
      <At x={1180} y={400} size={22} color={C.muted} o={eseg(p, 0.2, 0.3)}>10 × 99% = 9.9 ≈ 10</At>
      <At x={1180} y={500} size={30} color={C.yellow} o={eseg(p, 0.45, 0.55)}>{t('watch.bayes.label.healthy_pos', { n: COUNTS.fp })}</At>
      <At x={1180} y={570} size={22} color={C.muted} o={eseg(p, 0.45, 0.55)}>9,990 × 5% = 499.5 ≈ 500</At>
      <At x={1180} y={700} size={30} w={560} o={eseg(p, 0.72, 0.82)} style={{ textAlign: 'center' }}>{t('watch.bayes.label.false_alarms')}</At>
    </>
  )
}

function GatherScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const gather = eseg(p, 0.05, 0.4)
  const box = { x: CROWD.x + (520 / 1000) * CROWD.size, y: CROWD.y + (300 / 1000) * CROWD.size * (900 / 900) }
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <Crowd infected={1} tested={1} gather={gather} dimNeg={eseg(p, 0, 0.3)} />
      <At x={1220} y={240} size={30} color={C.muted} o={eseg(p, 0.35, 0.45)}>{t('watch.bayes.label.all_pos')}</At>
      <At x={1220} y={330} size={96} className="font-black font-mono" o={eseg(p, 0.35, 0.45)}>{POS}</At>
      <At x={1220} y={450} size={40} className="font-mono" o={eseg(p, 0.5, 0.6)}>
        <span style={{ color: C.red, fontWeight: 900 }}>{COUNTS.tp}</span>
        <span style={{ color: C.muted }}> {t('watch.bayes.label.infected_short')} + </span>
        <span style={{ color: C.yellow, fontWeight: 900 }}>{COUNTS.fp}</span>
        <span style={{ color: C.muted }}> {t('watch.bayes.label.healthy_short')}</span>
      </At>
      <Tex f={`\\frac{${COUNTS.tp}}{${POS}} \\approx ${pct(COUNTS.tp / POS).replace('%', '\\%')}`} x={1220} y={600} size={64} color={C.yellow} o={eseg(p, 0.66, 0.78)} />
      <At x={1220} y={740} size={28} color={C.muted} w={600} o={eseg(p, 0.8, 0.9)} style={{ textAlign: 'center' }}>{t('watch.bayes.label.not_99')}</At>
      <At x={box.x} y={box.y - 34} size={22} color={C.red} anchor="l" o={eseg(p, 0.42, 0.5)}>↓ {t('watch.bayes.label.these_10')}</At>
    </>
  )
}

function FormulaScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const num = p > 0.3 ? `\\color{${C.red}}` : ''
  const den = p > 0.45 ? `\\color{${C.yellow}}` : ''
  return (
    <>
      <Svg>
        <Grid o={0.25} />
      </Svg>
      <Tex f={`P(V \\mid +) = \\dfrac{${num}P(+ \\mid V)\\,P(V)}{${den}P(+ \\mid V)\\,P(V) + P(+ \\mid H)\\,P(H)}`} x={800} y={250} size={46} o={eseg(p, 0, 0.12)} />
      <At x={800} y={120} size={26} color={C.red} o={eseg(p, 0.3, 0.4)}>{t('watch.bayes.label.num_is', { n: COUNTS.tp })}</At>
      <At x={800} y={380} size={26} color={C.yellow} o={eseg(p, 0.45, 0.55)}>{t('watch.bayes.label.den_is', { n: POS })}</At>
      <Tex f={`= \\dfrac{${HIT} \\times ${PRIOR}}{${HIT} \\times ${PRIOR} + ${(1 - SPEC).toFixed(2)} \\times ${1 - PRIOR}}`} x={800} y={540} size={44} o={eseg(p, 0.6, 0.7)} />
      <Tex f={`= \\dfrac{${(ONE.likeB * ONE.pB).toFixed(5)}}{${ONE.evidence.toFixed(5)}} \\approx ${ONE.posterior.toFixed(4)}`} x={800} y={700} size={48} color={C.yellow} o={eseg(p, 0.76, 0.86)} />
    </>
  )
}

function AgainScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const steps = [
    { label: t('watch.bayes.label.before'), v: PRIOR, at: 0.05 },
    { label: t('watch.bayes.label.after_n', { n: 1 }), v: ONE.posterior, at: 0.22 },
    { label: t('watch.bayes.label.after_n', { n: 2 }), v: TWO.posterior, at: 0.45 },
    { label: t('watch.bayes.label.after_n', { n: 3 }), v: THREE.posterior, at: 0.68 },
  ]
  return (
    <>
      <Svg>
        <Grid o={0.2} />
        {steps.map((s, i) => (
          <Bar key={i} x={560} y={226 + i * 140} w={760} h={28} frac={s.v * eseg(p, s.at, s.at + 0.12)} color={i === 0 ? C.muted : C.red} o={eseg(p, s.at, s.at + 0.06)} />
        ))}
      </Svg>
      <At x={800} y={110} size={40} className="font-black" o={eseg(p, 0, 0.08)}>{t('watch.bayes.label.test_again')}</At>
      {steps.map((s, i) => (
        <Fragment key={i}>
          <At x={520} y={240 + i * 140} size={30} anchor="r" color={C.muted} o={eseg(p, s.at, s.at + 0.06)}>{s.label}</At>
          <At x={1350} y={240 + i * 140} size={40} anchor="l" className="font-mono font-black" color={i === 0 ? C.text : C.red} o={eseg(p, s.at + 0.06, s.at + 0.12)}>
            {pct(s.v)}
          </At>
        </Fragment>
      ))}
      <At x={800} y={800} size={28} color={C.yellow} w={1200} o={eseg(p, 0.84, 0.92)} style={{ textAlign: 'center' }}>{t('watch.bayes.label.posterior_prior')}</At>
    </>
  )
}

type Patient = Record<BayesFeature, string>
const P1: Patient = { BP: 'High', Fever: 'No', Diabetes: 'Yes', Vomit: 'Yes' }
const P2: Patient = { BP: 'Low', Fever: 'High', Diabetes: 'Yes', Vomit: 'Yes' }
const FEATS: BayesFeature[] = ['BP', 'Fever', 'Diabetes', 'Vomit']

/** One class's product, written out factor by factor. */
function Product({ y, cls, patient, alpha, show, hlZero }: { y: number; cls: 'yes' | 'no'; patient: Patient; alpha: number; show: number; hlZero?: boolean }) {
  const res = computeNaiveBayes(BAYES_DATA, patient, alpha, false)[cls]
  const col = cls === 'yes' ? C.teal : C.purple
  const cells = res.steps.map((s, i) => {
    const frac = s.formula ? s.formula.replace(/\\frac\{(\d+) \+ (\d+(?:\.\d+)?)\}\{(\d+) \+ (\d+) \\times (\d+(?:\.\d+)?)\}/, (_m, c, a, n, m, a2) => (Number(a) ? `\\tfrac{${c}+${a}}{${n}+${m}\\cdot${a2}}` : `\\tfrac{${c}}{${n}}`)) : `\\tfrac{${cls === 'yes' ? 9 : 5}}{14}`
    const zero = s.val === 0
    return { frac, zero, i }
  })
  return (
    <>
      <At x={120} y={y} size={30} anchor="l" color={col} className="font-black" o={eseg(show, 0, 0.2)}>Z = {cls === 'yes' ? 'Yes' : 'No'}</At>
      {cells.map((c) => (
        <Tex key={c.i} f={(c.i ? '\\times\\,' : '') + c.frac} x={330 + c.i * 170} y={y} size={36} color={c.zero && hlZero ? C.red : C.text} o={eseg(show, 0.1 + c.i * 0.12, 0.25 + c.i * 0.12)} scale={c.zero && hlZero ? 1.25 : 1} />
      ))}
      <Tex f={`= ${res.raw === 0 ? '0' : res.raw.toFixed(5)}`} x={1300} y={y} size={40} color={res.raw === 0 ? C.red : col} o={eseg(show, 0.8, 0.95)} />
    </>
  )
}

function PatientRow({ patient, o, changed }: { patient: Patient; o: number; changed?: BayesFeature }) {
  const { t } = useTranslation()
  return (
    <At x={800} y={170} size={30} o={o}>
      <span style={{ color: C.muted }}>{t('watch.bayes.label.patient')} </span>
      {FEATS.map((f, i) => (
        <span key={f}>
          {i ? ', ' : ''}
          <span style={{ color: C.muted }}>{t(`watch.bayes.feat.${f}`)} = </span>
          <b style={{ color: f === changed ? C.red : C.text }}>{t(`watch.bayes.val.${f}.${patient[f]}`)}</b>
        </span>
      ))}
    </At>
  )
}

function NaiveScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const r = computeNaiveBayes(BAYES_DATA, P1, 0, false)
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <Tex f="P(Z \mid e_1,\dots,e_4) \;\propto\; P(Z)\,\prod_j P(e_j \mid Z)" x={800} y={80} size={40} o={eseg(p, 0, 0.1)} />
      <PatientRow patient={P1} o={eseg(p, 0.08, 0.16)} />
      <Product y={340} cls="yes" patient={P1} alpha={0} show={seg(p, 0.18, 0.5)} />
      <Product y={500} cls="no" patient={P1} alpha={0} show={seg(p, 0.4, 0.72)} />
      <At x={800} y={680} size={36} o={eseg(p, 0.75, 0.85)}>
        <span style={{ color: C.muted }}>{t('watch.bayes.label.normalise')} </span>
        <b style={{ color: C.teal }}>{pct(r.yes.prob)}</b>
        <span style={{ color: C.muted }}> {t('watch.bayes.label.vs')} </span>
        <b style={{ color: C.purple }}>{pct(r.no.prob)}</b>
      </At>
      <At x={800} y={770} size={28} color={C.muted} o={eseg(p, 0.85, 0.93)} w={1200} style={{ textAlign: 'center' }}>{t('watch.bayes.label.argmax')}</At>
    </>
  )
}

function ZeroScene({ p }: { p: number }) {
  const { t } = useTranslation()
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <PatientRow patient={P2} o={1} changed="BP" />
      <Product y={340} cls="yes" patient={P2} alpha={0} show={seg(p, 0.05, 0.3)} />
      <Product y={500} cls="no" patient={P2} alpha={0} show={seg(p, 0.25, 0.5)} hlZero={p > 0.55} />
      <At x={800} y={680} size={34} color={C.red} w={1300} o={eseg(p, 0.6, 0.7)} style={{ textAlign: 'center' }}>{t('watch.bayes.label.zero_kills')}</At>
      <At x={800} y={770} size={26} color={C.muted} w={1300} o={eseg(p, 0.75, 0.85)} style={{ textAlign: 'center' }}>{t('watch.bayes.label.no_low')}</At>
    </>
  )
}

function LaplaceScene({ p }: { p: number }) {
  const { t } = useTranslation()
  const r = computeNaiveBayes(BAYES_DATA, P2, 1, false)
  return (
    <>
      <Svg>
        <Grid o={0.2} />
      </Svg>
      <Tex f="P(e_j = v \mid B) = \dfrac{\text{count}(e_j = v, B) + \alpha}{\text{count}(B) + m\,\alpha}" x={800} y={110} size={40} o={eseg(p, 0, 0.12)} />
      <At x={800} y={210} size={26} color={C.muted} o={eseg(p, 0.08, 0.16)}>{t('watch.bayes.label.alpha_note')}</At>
      <Product y={360} cls="yes" patient={P2} alpha={1} show={seg(p, 0.18, 0.45)} />
      <Product y={510} cls="no" patient={P2} alpha={1} show={seg(p, 0.4, 0.67)} />
      <At x={800} y={680} size={36} o={eseg(p, 0.72, 0.82)}>
        <span style={{ color: C.muted }}>{t('watch.bayes.label.normalise')} </span>
        <b style={{ color: C.teal }}>{pct(r.yes.prob)}</b>
        <span style={{ color: C.muted }}> {t('watch.bayes.label.vs')} </span>
        <b style={{ color: C.purple }}>{pct(r.no.prob)}</b>
      </At>
      <At x={800} y={770} size={28} color={C.yellow} o={eseg(p, 0.82, 0.9)} w={1200} style={{ textAlign: 'center' }}>{t('watch.bayes.label.all_evidence')}</At>
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
        <At key={k} x={240} y={240 + k * 120} size={38} anchor="l" w={1150} o={eseg(p, 0.08 + k * 0.12, 0.16 + k * 0.12)}>
          <span style={{ color: C.yellow, fontWeight: 900 }}>{k}. </span>
          {t(`watch.bayes.recap.${k}`)}
        </At>
      ))}
      <At x={800} y={800} size={28} color={C.muted} o={eseg(p, 0.75, 0.85)}>{t('watch.ui.try_it')}</At>
    </>
  )
}

export const bayes: Episode = {
  id: 'bayes',
  scenes: [
    { id: 'title', dur: 5000, cues: [0], render: (p) => <Title p={p} /> },
    { id: 'question', dur: 14000, cues: [0, 0.24, 0.6], ponder: true, render: (p) => <Question p={p} /> },
    { id: 'crowd', dur: 12000, cues: [0, 0.45], render: (p) => <CrowdScene p={p} /> },
    { id: 'test', dur: 16000, cues: [0, 0.2, 0.45, 0.72], render: (p) => <TestScene p={p} /> },
    { id: 'gather', dur: 16000, cues: [0, 0.35, 0.66], render: (p) => <GatherScene p={p} /> },
    { id: 'formula', dur: 16000, cues: [0, 0.3, 0.6], render: (p) => <FormulaScene p={p} /> },
    { id: 'again', dur: 15000, cues: [0, 0.22, 0.68], render: (p) => <AgainScene p={p} /> },
    { id: 'naive', dur: 18000, cues: [0, 0.18, 0.4, 0.75], render: (p) => <NaiveScene p={p} /> },
    { id: 'zero', dur: 14000, cues: [0, 0.25, 0.6], ponder: true, render: (p) => <ZeroScene p={p} /> },
    { id: 'laplace', dur: 16000, cues: [0, 0.18, 0.72], render: (p) => <LaplaceScene p={p} /> },
    { id: 'recap', dur: 12000, cues: [0, 0.6], render: (p) => <Recap p={p} /> },
  ],
}
