import { useEffect, useRef, useState } from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { bayesPosterior, naturalCounts, type NaturalCounts } from '../../lib/bayesRule'
import { fmt, int } from '../format'
import { Ans, Btn, Card, LabPage, Note, Presets, Seg, Slider, TableWrap, Workspace } from '../ui'

const N = 10000
const COLORS = { tp: '#e11d48', fn: '#fda4af', fp: '#f59e0b', tn: '#cbd5e1' }

interface Scenario {
  id: string
  prior: number
  hit: number
  falseAlarm: number
  observed?: 'E' | 'notE'
}

const PRESETS: Scenario[] = [
  { id: 'virus', prior: 0.001, hit: 0.99, falseAlarm: 0.05 },
  { id: 'grade', prior: 0.1, hit: 0.9, falseAlarm: 0.06 },
  { id: 'course', prior: 0.2, hit: 0.6, falseAlarm: 0.35, observed: 'notE' },
  { id: 'spam', prior: 0.3, hit: 0.92, falseAlarm: 0.04 },
]

// The prior slider is logarithmic (0.01% … 50%) so rare conditions get room.
const LOG_MIN = Math.log10(0.0001)
const LOG_MAX = Math.log10(0.5)
const toSlider = (p: number) => Math.round(((Math.log10(Math.min(0.5, Math.max(0.0001, p))) - LOG_MIN) / (LOG_MAX - LOG_MIN)) * 1000)
const fromSlider = (v: number) => {
  const p = 10 ** (LOG_MIN + (v / 1000) * (LOG_MAX - LOG_MIN))
  const mag = 10 ** Math.floor(Math.log10(p))
  return (Math.round((p / mag) * 10) / 10) * mag // two significant figures
}

function PeopleGrid({ counts, focus }: { counts: NaturalCounts; focus: Set<keyof NaturalCounts> | null }) {
  const { t } = useTranslation()
  const ref = useRef<HTMLCanvasElement>(null)
  const [width, setWidth] = useState(800)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const cols = width < 500 ? 100 : 200
    const cell = width / cols
    const h = Math.ceil(cell * (N / cols))
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.round(width * dpr)
    canvas.height = Math.round(h * dpr)
    canvas.style.height = `${h}px`
    const ctx = canvas.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, h)
    const gap = cell > 5 ? 1 : 0.5
    let k = 0
    for (const g of ['tp', 'fn', 'fp', 'tn'] as const) {
      ctx.fillStyle = COLORS[g]
      ctx.globalAlpha = focus && !focus.has(g) ? 0.08 : 1
      for (let i = 0; i < counts[g]; i++, k++) ctx.fillRect((k % cols) * cell + gap / 2, Math.floor(k / cols) * cell + gap / 2, cell - gap, cell - gap)
    }
  }, [counts, focus, width])
  return <canvas ref={ref} className="w-full rounded-xl bg-slate-50" role="img" aria-label={t('lab.bayes.grid_aria')} />
}

function Tree({ s, c, observedE, B, E }: { s: Scenario; c: NaturalCounts; observedE: boolean; B: string; E: string }) {
  const { t } = useTranslation()
  const not = (x: string) => t('lab.bayes.not', { x })
  const leaf = (x: number, y: number, n: number, label: string, hl = false) => (
    <g>
      <rect x={x - 60} y={y - 22} width={120} height={44} rx={10} fill={hl ? '#fff1f2' : '#f8fafc'} stroke={hl ? '#fda4af' : '#e2e8f0'} />
      <text x={x} y={y + 1} textAnchor="middle" className="fill-slate-900 font-mono text-[15px] font-extrabold">{int(n)}</text>
      <text x={x} y={y + 16} textAnchor="middle" className="fill-slate-500 text-[12px]">{label}</text>
    </g>
  )
  const edge = (x1: number, y1: number, x2: number, y2: number, p: string) => (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#cbd5e1" strokeWidth={2} />
      <text x={(x1 + x2) / 2 + (x2 > x1 ? 6 : -6)} y={(y1 + y2) / 2} textAnchor={x2 > x1 ? 'start' : 'end'} className="fill-slate-500 text-[12px] font-bold">{p}</text>
    </g>
  )
  const p = (v: number) => `${fmt(v * 100, 2)}%`
  return (
    <svg viewBox="0 0 520 250" className="w-full" role="img" aria-label={t('lab.bayes.tree_title')}>
      {edge(260, 40, 130, 110, p(s.prior))}
      {edge(260, 40, 390, 110, p(1 - s.prior))}
      {edge(130, 130, 66, 200, p(s.hit))}
      {edge(130, 130, 194, 200, p(1 - s.hit))}
      {edge(390, 130, 326, 200, p(s.falseAlarm))}
      {edge(390, 130, 454, 200, p(1 - s.falseAlarm))}
      {leaf(260, 26, N, t('lab.bayes.people'))}
      {leaf(130, 120, c.tp + c.fn, B)}
      {leaf(390, 120, c.fp + c.tn, not(B))}
      {leaf(66, 214, c.tp, E, observedE)}
      {leaf(194, 214, c.fn, not(E), !observedE)}
      {leaf(326, 214, c.fp, E, observedE)}
      {leaf(454, 214, c.tn, not(E), !observedE)}
    </svg>
  )
}

export default function BayesVirus() {
  const { t } = useTranslation()
  const [s, setS] = useState<Scenario>(PRESETS[0])
  const [preset, setPreset] = useState<string | null>('virus')
  // event names come from the last chosen scenario (sliders keep them)
  const [names, setNames] = useState('virus')
  const B = t(`lab.bayes.presets.${names}.B`)
  const E = t(`lab.bayes.presets.${names}.E`)
  const not = (x: string) => t('lab.bayes.not', { x })
  const [observed, setObserved] = useState<'E' | 'notE'>('E')
  const [focusOn, setFocusOn] = useState(false)

  const r = bayesPosterior({ prior: s.prior, hit: s.hit, falseAlarm: s.falseAlarm, observed })
  const c = naturalCounts(s.prior, s.hit, s.falseAlarm, N)
  const obsE = observed === 'E'
  const evid = obsE ? E : not(E)
  const hitN = obsE ? c.tp : c.fn
  const allN = obsE ? c.tp + c.fp : c.fn + c.tn
  const focus = focusOn ? new Set<keyof NaturalCounts>(obsE ? ['tp', 'fp'] : ['fn', 'tn']) : null
  const edit = (patch: Partial<Scenario>) => {
    setS((cur) => ({ ...cur, ...patch }))
    setPreset(null)
  }

  return (
    <LabPage id="bayes-virus" quiz lead={t('lab.bayes.lead')}>
      <Workspace
        controls={
          <>
            <Card title={t('lab.bayes.scenario')}>
              <Presets
                items={PRESETS.map((p) => ({ id: p.id, title: t(`lab.bayes.presets.${p.id}.title`), note: t(`lab.bayes.presets.${p.id}.note`) }))}
                value={preset}
                onPick={(id) => {
                  const p = PRESETS.find((x) => x.id === id)!
                  setS(p)
                  setPreset(id)
                  setNames(id)
                  setObserved(p.observed ?? 'E')
                }}
              />
            </Card>
            <Card title={t('lab.bayes.probabilities')}>
              <div className="grid gap-3">
                <Slider label={t('lab.bayes.prior_label', { B })} value={toSlider(s.prior)} min={0} max={1000} step={1} display={`${fmt(s.prior * 100, 2)}%`} onChange={(v) => edit({ prior: fromSlider(v) })} />
                <Slider label={t('lab.bayes.hit_label', { B, E })} value={s.hit} min={0} max={1} step={0.005} display={`${fmt(s.hit * 100, 1)}%`} onChange={(v) => edit({ hit: v })} />
                <Slider label={t('lab.bayes.fa_label', { B, E })} value={s.falseAlarm} min={0} max={0.5} step={0.001} display={`${fmt(s.falseAlarm * 100, 1)}%`} onChange={(v) => edit({ falseAlarm: v })} />
                <Seg label={t('lab.bayes.observed')} value={observed} onChange={setObserved} options={[{ v: 'E', label: E }, { v: 'notE', label: not(E) }]} />
              </div>
            </Card>
            <Card flat>
              <p className="text-[13px] text-slate-600">
                <Trans i18nKey="lab.bayes.again_text" components={{ 1: <b /> }} />
              </p>
              <div className="mt-2.5">
                <Btn primary onClick={() => edit({ prior: Math.min(0.999, Math.max(0.0001, r.posterior)) })}>
                  {t('lab.bayes.again_btn')}
                </Btn>
              </div>
            </Card>
          </>
        }
      >
        <Card
          step={1}
          title={t('lab.bayes.grid_title')}
          right={
            <label className="flex cursor-pointer items-center gap-2 text-[13px] font-semibold text-slate-600">
              <input type="checkbox" checked={focusOn} onChange={(e) => setFocusOn(e.target.checked)} className="h-4 w-4" />
              {t('lab.bayes.focus', { evid })}
            </label>
          }
        >
          <PeopleGrid counts={c} focus={focus} />
          <div className="mt-2.5 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600">
            {(
              [
                ['tp', `${B} & ${E}`],
                ['fn', `${B} & ${not(E)}`],
                ['fp', `${not(B)} & ${E}`],
                ['tn', `${not(B)} & ${not(E)}`],
              ] as const
            ).map(([g, label]) => (
              <span key={g} className="inline-flex items-center gap-1.5">
                <i className="inline-block h-3 w-3 rounded-[3px]" style={{ background: COLORS[g] }} />
                {label} <b className="font-mono">{int(c[g])}</b>
              </span>
            ))}
          </div>
        </Card>

        <div className="grid gap-5 md:grid-cols-2">
          <Card step={2} title={t('lab.bayes.tree_title')}>
            <Tree s={s} c={c} observedE={obsE} B={B} E={E} />
          </Card>
          <Card step={3} title={t('lab.bayes.answer_title')}>
            <p className="text-[13px] text-slate-500">{t('lab.bayes.answer_q', { evid, B })}</p>
            <div className="my-2.5 flex items-baseline gap-3">
              <span className={`font-mono text-[44px] font-black leading-none ${r.posterior >= 0.5 ? 'text-emerald-600' : 'text-rose-600'}`}>{fmt(r.posterior * 100, 2)}%</span>
              <span className="font-mono text-slate-500">
                ≈ {int(hitN)} / {int(allN)}
              </span>
            </div>
            <p className="text-[13px] text-slate-600">
              <Trans i18nKey="lab.bayes.prior_was" values={{ p: `${fmt(s.prior * 100, 2)}%`, lr: `${fmt(r.likelihoodRatio, 2)}×`, evid, B, notB: not(B) }} components={{ 1: <b />, 3: <b className="font-mono" /> }} />
            </p>
            {r.posterior < 0.5 && s.prior < 0.05 && obsE && (
              <div className="mt-3">
                <Note tone="bad" title={t('lab.bayes.base_rate_title')}>
                  {t('lab.bayes.base_rate_text', { fp: int(c.fp), tp: int(c.tp) })}
                </Note>
              </div>
            )}
          </Card>
        </div>

        <Card step={4} title={t('lab.bayes.working_title')} sub={t('lab.common.quiz_sub')}>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="formula">
              <span className="text-slate-500">{t('lab.bayes.total_prob')}</span>
              {`\nP(${evid}) = P(${evid}|${B})·P(${B}) + P(${evid}|${not(B)})·P(${not(B)})\n       = ${fmt(r.likeB, 4)} × ${fmt(r.pB, 4)} + ${fmt(r.likeNotB, 4)} × ${fmt(r.pNotB, 4)}\n       = `}
              <b className="text-blue-700">{fmt(r.evidence, 5)}</b>
              {'\n\n'}
              <span className="text-slate-500">{t('lab.bayes.bayes_rule')}</span>
              {`\nP(${B}|${evid}) = P(${B})·P(${evid}|${B}) / P(${evid})\n         = ${fmt(r.pB, 4)} × ${fmt(r.likeB, 4)} / ${fmt(r.evidence, 5)}\n         = `}
              <b className="text-rose-600">{fmt(r.posterior, 4)}</b>
            </div>
            <TableWrap>
              <tr>
                <th className="left">{t('lab.common.quantity')}</th>
                <th>{t('lab.common.value')}</th>
              </tr>
              <tr><td className="left">P({B})</td><td>{fmt(r.pB, 4)}</td></tr>
              <tr><td className="left">P({not(B)})</td><Ans k="pNotB" v={fmt(r.pNotB, 4)} /></tr>
              <tr><td className="left">P({evid} | {B})</td><td>{fmt(r.likeB, 4)}</td></tr>
              <tr><td className="left">P({evid} | {not(B)})</td><td>{fmt(r.likeNotB, 4)}</td></tr>
              <tr><td className="left">P({evid})</td><Ans k="evidence" v={fmt(r.evidence, 4)} /></tr>
              <tr><td className="left">P({B} | {evid})</td><Ans k="posterior" v={fmt(r.posterior, 4)} /></tr>
            </TableWrap>
          </div>
        </Card>

        <Note tone="warn" title={t('lab.common.exam_pattern')}>
          <Trans i18nKey="lab.bayes.exam_text" components={{ 1: <i /> }} />
        </Note>
      </Workspace>
    </LabPage>
  )
}
