import { useEffect, useRef, useState } from 'react'
import { bayesPosterior, naturalCounts, type NaturalCounts } from '../../lib/bayesRule'
import { fmt, int } from '../format'
import { Ans, Btn, Card, LabPage, Note, Presets, Seg, Slider, TableWrap, Workspace } from '../ui'

const N = 10000
const COLORS = { tp: '#e11d48', fn: '#fda4af', fp: '#f59e0b', tn: '#cbd5e1' }

interface Scenario {
  id: string
  title: string
  note: string
  B: string
  E: string
  prior: number
  hit: number
  falseAlarm: number
  observed?: 'E' | 'notE'
}

const PRESETS: Scenario[] = [
  { id: 'virus', title: 'Virus test', note: 'Lecture 2 · 0.1% infected, 99% / 95% accurate', B: 'Virus', E: 'Positive', prior: 0.001, hit: 0.99, falseAlarm: 0.05 },
  { id: 'grade', title: 'Exam-style: top grade', note: 'Pattern of 2022 Fall Q3 (numbers changed)', B: 'A+', E: 'Midterm > 90', prior: 0.1, hit: 0.9, falseAlarm: 0.06 },
  { id: 'course', title: 'Exam-style: “not good” course', note: 'Pattern of 2022 Spring Q3 — asks P(B | not E)', B: 'Prof teaches', E: 'Good course', prior: 0.2, hit: 0.6, falseAlarm: 0.35, observed: 'notE' },
  { id: 'spam', title: 'Spam filter', note: 'Common prior, decent filter', B: 'Spam', E: 'Flagged', prior: 0.3, hit: 0.92, falseAlarm: 0.04 },
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
  return <canvas ref={ref} className="w-full rounded-xl bg-slate-50" role="img" aria-label="Grid of 10,000 people coloured by group" />
}

function Tree({ s, c, observedE }: { s: Scenario; c: NaturalCounts; observedE: boolean }) {
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
    <svg viewBox="0 0 520 250" className="w-full" role="img" aria-label="Natural frequency tree">
      {edge(260, 40, 130, 110, p(s.prior))}
      {edge(260, 40, 390, 110, p(1 - s.prior))}
      {edge(130, 130, 66, 200, p(s.hit))}
      {edge(130, 130, 194, 200, p(1 - s.hit))}
      {edge(390, 130, 326, 200, p(s.falseAlarm))}
      {edge(390, 130, 454, 200, p(1 - s.falseAlarm))}
      {leaf(260, 26, N, 'people')}
      {leaf(130, 120, c.tp + c.fn, s.B)}
      {leaf(390, 120, c.fp + c.tn, `not ${s.B}`)}
      {leaf(66, 214, c.tp, s.E, observedE)}
      {leaf(194, 214, c.fn, `not ${s.E}`, !observedE)}
      {leaf(326, 214, c.fp, s.E, observedE)}
      {leaf(454, 214, c.tn, `not ${s.E}`, !observedE)}
    </svg>
  )
}

export default function BayesVirus() {
  const [s, setS] = useState<Scenario>(PRESETS[0])
  const [preset, setPreset] = useState<string | null>('virus')
  const [observed, setObserved] = useState<'E' | 'notE'>('E')
  const [focusOn, setFocusOn] = useState(false)

  const r = bayesPosterior({ prior: s.prior, hit: s.hit, falseAlarm: s.falseAlarm, observed })
  const c = naturalCounts(s.prior, s.hit, s.falseAlarm, N)
  const obsE = observed === 'E'
  const evid = obsE ? s.E : `not ${s.E}`
  const hitN = obsE ? c.tp : c.fn
  const allN = obsE ? c.tp + c.fp : c.fn + c.tn
  const focus = focusOn ? new Set<keyof NaturalCounts>(obsE ? ['tp', 'fp'] : ['fn', 'tn']) : null
  const edit = (patch: Partial<Scenario>) => {
    setS((cur) => ({ ...cur, ...patch }))
    setPreset(null)
  }

  return (
    <LabPage id="bayes-virus" quiz lead="A test that is 99% sensitive can still be wrong most of the time it says “positive”. The reason is the base rate: how rare the condition is in the first place.">
      <Workspace
        controls={
          <>
            <Card title="Scenario">
              <Presets
                items={PRESETS}
                value={preset}
                onPick={(id) => {
                  const p = PRESETS.find((x) => x.id === id)!
                  setS(p)
                  setPreset(id)
                  setObserved(p.observed ?? 'E')
                }}
              />
            </Card>
            <Card title="Probabilities">
              <div className="grid gap-3">
                <Slider label={`P(${s.B}) — base rate`} value={toSlider(s.prior)} min={0} max={1000} step={1} display={`${fmt(s.prior * 100, 2)}%`} onChange={(v) => edit({ prior: fromSlider(v) })} />
                <Slider label={`P(${s.E} | ${s.B}) — hit rate`} value={s.hit} min={0} max={1} step={0.005} display={`${fmt(s.hit * 100, 1)}%`} onChange={(v) => edit({ hit: v })} />
                <Slider label={`P(${s.E} | not ${s.B}) — false alarms`} value={s.falseAlarm} min={0} max={0.5} step={0.001} display={`${fmt(s.falseAlarm * 100, 1)}%`} onChange={(v) => edit({ falseAlarm: v })} />
                <Seg label="We observed" value={observed} onChange={setObserved} options={[{ v: 'E', label: s.E }, { v: 'notE', label: `not ${s.E}` }]} />
              </div>
            </Card>
            <Card flat>
              <p className="text-[13px] text-slate-600">
                <b>Test again.</b> Today’s posterior becomes tomorrow’s prior. Press it twice and watch a rare condition become likely.
              </p>
              <div className="mt-2.5">
                <Btn primary onClick={() => edit({ prior: Math.min(0.999, Math.max(0.0001, r.posterior)) })}>
                  Test again → posterior becomes prior
                </Btn>
              </div>
            </Card>
          </>
        }
      >
        <Card
          step={1}
          title="10,000 people, one square each"
          right={
            <label className="flex cursor-pointer items-center gap-2 text-[13px] font-semibold text-slate-600">
              <input type="checkbox" checked={focusOn} onChange={(e) => setFocusOn(e.target.checked)} className="h-4 w-4" />
              Show only people with “{evid}”
            </label>
          }
        >
          <PeopleGrid counts={c} focus={focus} />
          <div className="mt-2.5 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600">
            {(
              [
                ['tp', `${s.B} & ${s.E}`],
                ['fn', `${s.B} & not ${s.E}`],
                ['fp', `not ${s.B} & ${s.E}`],
                ['tn', `not ${s.B} & not ${s.E}`],
              ] as const
            ).map(([g, t]) => (
              <span key={g} className="inline-flex items-center gap-1.5">
                <i className="inline-block h-3 w-3 rounded-[3px]" style={{ background: COLORS[g] }} />
                {t} <b className="font-mono">{int(c[g])}</b>
              </span>
            ))}
          </div>
        </Card>

        <div className="grid gap-5 md:grid-cols-2">
          <Card step={2} title="Natural-frequency tree">
            <Tree s={s} c={c} observedE={obsE} />
          </Card>
          <Card step={3} title="The answer">
            <p className="text-[13px] text-slate-400">
              Among everyone with “{evid}”, how many really have “{s.B}”?
            </p>
            <div className="my-2.5 flex items-baseline gap-3">
              <span className={`font-mono text-[44px] font-black leading-none ${r.posterior >= 0.5 ? 'text-emerald-600' : 'text-rose-600'}`}>{fmt(r.posterior * 100, 2)}%</span>
              <span className="font-mono text-slate-500">
                ≈ {int(hitN)} / {int(allN)}
              </span>
            </div>
            <p className="text-[13px] text-slate-600">
              Prior was <b>{fmt(s.prior * 100, 2)}%</b>. The evidence multiplied the odds by <b className="font-mono">{fmt(r.likelihoodRatio, 2)}×</b> (likelihood ratio P({evid}|{s.B}) ÷ P({evid}|not {s.B})).
            </p>
            {r.posterior < 0.5 && s.prior < 0.05 && obsE && (
              <div className="mt-3">
                <Note tone="bad" title="Base-rate effect.">
                  {int(c.fp)} false alarms from the large healthy group swamp the {int(c.tp)} true hits.
                </Note>
              </div>
            )}
          </Card>
        </div>

        <Card step={4} title="The exam working" sub="Turn on “Quiz me” to fill in the blanks yourself">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="formula">
              <span className="text-slate-400">Total probability</span>
              {`\nP(${evid}) = P(${evid}|${s.B})·P(${s.B}) + P(${evid}|not ${s.B})·P(not ${s.B})\n       = ${fmt(r.likeB, 4)} × ${fmt(r.pB, 4)} + ${fmt(r.likeNotB, 4)} × ${fmt(r.pNotB, 4)}\n       = `}
              <b className="text-blue-700">{fmt(r.evidence, 5)}</b>
              {'\n\n'}
              <span className="text-slate-400">Bayes’ rule</span>
              {`\nP(${s.B}|${evid}) = P(${s.B})·P(${evid}|${s.B}) / P(${evid})\n         = ${fmt(r.pB, 4)} × ${fmt(r.likeB, 4)} / ${fmt(r.evidence, 5)}\n         = `}
              <b className="text-rose-600">{fmt(r.posterior, 4)}</b>
            </div>
            <TableWrap>
              <tr>
                <th className="left">Quantity</th>
                <th>Value</th>
              </tr>
              <tr><td className="left">P({s.B})</td><td>{fmt(r.pB, 4)}</td></tr>
              <tr><td className="left">P(not {s.B})</td><Ans k="pNotB" v={fmt(r.pNotB, 4)} /></tr>
              <tr><td className="left">P({evid} | {s.B})</td><td>{fmt(r.likeB, 4)}</td></tr>
              <tr><td className="left">P({evid} | not {s.B})</td><td>{fmt(r.likeNotB, 4)}</td></tr>
              <tr><td className="left">P({evid})</td><Ans k="evidence" v={fmt(r.evidence, 4)} /></tr>
              <tr><td className="left">P({s.B} | {evid})</td><Ans k="posterior" v={fmt(r.posterior, 4)} /></tr>
            </TableWrap>
          </div>
        </Card>

        <Note tone="warn" title="Exam pattern.">
          Midterms give three numbers — a prior, P(E|B) and P(E|not B) — and ask for P(E) with the total-probability rule, then P(B|E) with Bayes’ rule. Watch for the twist: some papers ask for P(B | <i>not</i> E), which uses 1 − P(E|B) and 1 − P(E|not B). Switch “We observed” to try it.
        </Note>
      </Workspace>
    </LabPage>
  )
}
