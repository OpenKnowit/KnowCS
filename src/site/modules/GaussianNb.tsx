import { useMemo, useState } from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { fitGaussianNb, gaussianPdf, parseLabeledCsv, scoreGaussianNb, type GaussianClass } from '../../lib/gaussianNb'
import { niceTicks } from '../../lib/chart'
import { errorText, fmt, PALETTE } from '../format'
import { Ans, Card, LabPage, Note, Presets, Seg, Slider, Swatch, TableWrap, Workspace } from '../ui'

interface Preset {
  id: string
  csv?: string
  given?: { features: string[]; classes: GaussianClass[] }
  test: number[]
}

const PRESETS: Preset[] = [
  {
    id: 'dengue',
    csv: `diagnosis, temperature, pulse
Dengue, 39.5, 62
Dengue, 39.0, 55
Dengue, 38.2, 70
No dengue, 36.4, 88
No dengue, 36.8, 76
No dengue, 37.1, 101`,
    test: [36.6, 84],
  },
  {
    id: 'lecture',
    given: {
      features: ['blood pressure', 'fever'],
      classes: [
        { name: 'Yes', prior: 9 / 14, priorLabel: '9/14', stats: [{ mu: 73, sigma: 6.2, values: null }, { mu: 79, sigma: 10.2, values: null }] },
        { name: 'No', prior: 5 / 14, priorLabel: '5/14', stats: [{ mu: 75, sigma: 7.9, values: null }, { mu: 86, sigma: 9.7, values: null }] },
      ],
    },
    test: [66, 97],
  },
  {
    id: 'bands',
    csv: `band, assignment, midterm
High, 88, 91
High, 92, 85
High, 85, 94
High, 90, 89
Mid, 75, 72
Mid, 80, 70
Mid, 70, 78
Low, 60, 52
Low, 72, 48
Low, 55, 61`,
    test: [78, 74],
  },
]

function Curves({ classes, j, x, likes, onPick }: { classes: GaussianClass[]; j: number; x: number; likes: number[]; onPick: (v: number) => void }) {
  const { t } = useTranslation()
  const W = 560, H = 230, L = 36, R = 14, T = 18, B = 46
  let lo = Infinity
  let hi = -Infinity
  classes.forEach((c) => {
    const { mu, sigma } = c.stats[j]
    lo = Math.min(lo, mu - 3.2 * sigma)
    hi = Math.max(hi, mu + 3.2 * sigma)
  })
  lo = Math.min(lo, x)
  hi = Math.max(hi, x)
  const pad = (hi - lo) * 0.04
  lo -= pad
  hi += pad
  const peak = Math.max(...classes.map((c) => gaussianPdf(c.stats[j].mu, c.stats[j].mu, c.stats[j].sigma)))
  const X = (v: number) => L + ((v - lo) / (hi - lo)) * (W - L - R)
  const Y = (v: number) => T + (1 - v / (peak * 1.12)) * (H - T - B)
  const right = X(x) > W * 0.7
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full cursor-crosshair"
      role="img"
      aria-label={t('lab.gnb.curve_aria')}
      onClick={(e) => {
        const r = e.currentTarget.getBoundingClientRect()
        const vx = ((e.clientX - r.left) / r.width) * W
        onPick(Number((lo + ((vx - L) / (W - L - R)) * (hi - lo)).toFixed(2)))
      }}
    >
      <line x1={L} y1={H - B} x2={W - R} y2={H - B} stroke="#cbd5e1" />
      {niceTicks(lo, hi, 6).map((v) => (
        <g key={v}>
          <line x1={X(v)} y1={H - B} x2={X(v)} y2={H - B + 4} stroke="#cbd5e1" />
          <text x={X(v)} y={H - B + 16} textAnchor="middle" className="fill-slate-400 text-[11px]">{fmt(v, 2)}</text>
        </g>
      ))}
      {classes.map((c, k) => {
        const { mu, sigma, values } = c.stats[j]
        const col = PALETTE[k % PALETTE.length]
        let d = ''
        for (let i = 0; i <= 160; i++) {
          const v = lo + (i / 160) * (hi - lo)
          d += `${i ? 'L' : 'M'}${X(v).toFixed(1)},${Y(gaussianPdf(v, mu, sigma)).toFixed(1)}`
        }
        return (
          <g key={c.name}>
            <path d={`${d}L${X(hi)},${Y(0)}L${X(lo)},${Y(0)}Z`} fill={col} opacity={0.07} />
            <path d={d} fill="none" stroke={col} strokeWidth={2.5} />
            <line x1={X(mu)} y1={Y(gaussianPdf(mu, mu, sigma))} x2={X(mu)} y2={H - B} stroke={col} strokeDasharray="3 3" opacity={0.6} />
            {values?.map((v, i) => <line key={i} x1={X(v)} y1={H - B + 22 + k * 6} x2={X(v)} y2={H - B + 27 + k * 6} stroke={col} strokeWidth={2.5} />)}
            <circle cx={X(x)} cy={Y(likes[k])} r={5} fill={col} stroke="#fff" strokeWidth={2} />
            <text x={X(x) + (right ? -9 : 9)} y={Y(likes[k]) - 6} textAnchor={right ? 'end' : 'start'} fill={col} className="font-mono text-[12px] font-bold">{fmt(likes[k], 4)}</text>
          </g>
        )
      })}
      <line x1={X(x)} y1={T} x2={X(x)} y2={H - B} stroke="#0f172a" strokeWidth={1.5} />
      <text x={X(x)} y={T - 4} textAnchor="middle" className="fill-slate-900 text-[12px] font-extrabold">x = {fmt(x, 2)}</text>
    </svg>
  )
}

export default function GaussianNb() {
  const { t } = useTranslation()
  // dataset words (feature / class names) are translated when we know them, shown as typed otherwise
  const nm = (s: string) => t(`lab.gnb.names.${s}`, { defaultValue: s })
  const [preset, setPreset] = useState<string | null>('dengue')
  const [csv, setCsv] = useState(PRESETS[0].csv!)
  const [given, setGiven] = useState<Preset['given'] | null>(null)
  const [ddof, setDdof] = useState<0 | 1>(1)
  const [test, setTest] = useState<number[]>(PRESETS[0].test)

  const model = useMemo(() => {
    if (given) return { features: given.features, classes: given.classes, error: null }
    try {
      const { features, rows } = parseLabeledCsv(csv)
      return { features, classes: fitGaussianNb(rows, features.length, ddof), error: null }
    } catch (e) {
      return { features: [], classes: [], error: e }
    }
  }, [csv, ddof, given])

  const x = model.features.map((_, j) => test[j] ?? model.classes[0]?.stats[j].mu ?? 0)
  const scored = model.error ? [] : scoreGaussianNb(model.classes, x)
  const best = scored.length ? scored.reduce((a, b) => (b.score > a.score ? b : a)) : null
  const ratio = scored.length === 2 ? Math.max(scored[0].score, scored[1].score) / Math.min(scored[0].score, scored[1].score) : null
  const setX = (j: number, v: number) => setTest(x.map((old, i) => (i === j ? v : old)))

  return (
    <LabPage id="gaussian-nb" quiz lead={t('lab.gnb.lead')}>
      <Workspace
        controls={
          <>
            <Card title={t('lab.common.dataset')}>
              <Presets
                items={PRESETS.map((p) => ({ id: p.id, title: t(`lab.gnb.presets.${p.id}.title`), note: t(`lab.gnb.presets.${p.id}.note`) }))}
                value={preset}
                onPick={(id) => {
                  const p = PRESETS.find((q) => q.id === id)!
                  setPreset(id)
                  setGiven(p.given ?? null)
                  if (p.csv) setCsv(p.csv)
                  setTest(p.test)
                }}
              />
            </Card>
            {!given && (
              <Card title={t('lab.gnb.training')} sub={t('lab.gnb.training_sub')}>
                <textarea
                  value={csv}
                  spellCheck={false}
                  aria-label={t('lab.gnb.training_aria')}
                  onChange={(e) => {
                    setCsv(e.target.value)
                    setPreset(null)
                  }}
                  className="min-h-[170px] w-full resize-y rounded-[10px] border border-slate-300 p-2.5 font-mono text-[13px] outline-none focus:ring-2 focus:ring-blue-300"
                />
                <p className={`mt-1.5 text-xs ${model.error ? 'text-rose-600' : 'text-slate-400'}`}>{model.error ? errorText(model.error, t) : t('lab.gnb.summary', { c: model.classes.length, f: model.features.length })}</p>
                <div className="mt-3">
                  <Seg label={t('lab.gnb.std')} value={ddof} onChange={setDdof} options={[{ v: 1, label: t('lab.gnb.std_sample') }, { v: 0, label: t('lab.gnb.std_pop') }]} />
                </div>
              </Card>
            )}
            {!model.error && (
              <Card title={t('lab.gnb.test')} sub={t('lab.gnb.test_sub')}>
                <div className="grid gap-3">
                  {model.features.map((f, j) => {
                    const lo = Math.min(...model.classes.map((c) => c.stats[j].mu - 3 * c.stats[j].sigma))
                    const hi = Math.max(...model.classes.map((c) => c.stats[j].mu + 3 * c.stats[j].sigma))
                    return <Slider key={f} label={nm(f)} value={x[j]} min={Number(lo.toFixed(2))} max={Number(hi.toFixed(2))} step={Number(((hi - lo) / 300).toPrecision(2))} display={fmt(x[j], 2)} onChange={(v) => setX(j, v)} />
                  })}
                </div>
              </Card>
            )}
          </>
        }
      >
        {model.error ? (
          <Note tone="bad" title={t('lab.common.cant_read')}>{errorText(model.error, t)}</Note>
        ) : (
          <>
            <Card step={1} title={t('lab.gnb.fit_title')} sub={given ? t('lab.gnb.given') : ddof === 1 ? 'σ = √( Σ(x − μ)² / (n − 1) )' : 'σ = √( Σ(x − μ)² / n )'}>
              <TableWrap>
                <tr>
                  <th className="left">{t('lab.common.class')}</th>
                  <th>{t('lab.common.prior')}</th>
                  {model.features.map((f) => [<th key={f + 'm'}>μ ({nm(f)})</th>, <th key={f + 's'}>σ ({nm(f)})</th>])}
                </tr>
                {model.classes.map((c, k) => (
                  <tr key={c.name}>
                    <td className="left"><Swatch color={PALETTE[k % PALETTE.length]} />{nm(c.name)}</td>
                    <Ans k={`prior${k}`} v={c.priorLabel} />
                    {c.stats.map((s, j) =>
                      given ? [<td key={j + 'm'}>{fmt(s.mu, 4)}</td>, <td key={j + 's'}>{fmt(s.sigma, 4)}</td>] : [<Ans key={j + 'm'} k={`mu${k}_${j}`} v={fmt(s.mu, 4)} />, <Ans key={j + 's'} k={`sd${k}_${j}`} v={fmt(s.sigma, 4)} />],
                    )}
                  </tr>
                ))}
              </TableWrap>
            </Card>

            <div className="grid gap-5 xl:grid-cols-2">
              {model.features.map((f, j) => (
                <Card
                  key={f}
                  step={2}
                  title={`f(${nm(f)} | ${t('lab.gnb.class_word')})`}
                  right={
                    <span className="flex gap-3 text-xs text-slate-600">
                      {model.classes.map((c, k) => (
                        <span key={c.name}><Swatch color={PALETTE[k % PALETTE.length]} />{nm(c.name)}</span>
                      ))}
                    </span>
                  }
                >
                  <Curves classes={model.classes} j={j} x={x[j]} likes={scored.map((c) => c.likelihoods[j])} onPick={(v) => setX(j, v)} />
                  <p className="mt-1.5 text-[13px] text-slate-500">
                    {scored.map((c, k) => (
                      <span key={c.name} className="mr-3">
                        f({fmt(x[j], 2)} | {nm(c.name)}) = <b className="font-mono" style={{ color: PALETTE[k % PALETTE.length] }}>{fmt(c.likelihoods[j], 5)}</b>
                      </span>
                    ))}
                  </p>
                </Card>
              ))}
            </div>

            <Card step={3} title={t('lab.gnb.multiply')} sub={t('lab.gnb.multiply_sub')}>
              <TableWrap>
                <tr>
                  <th className="left">{t('lab.common.class')}</th>
                  <th>P({t('lab.gnb.class_word')})</th>
                  {model.features.map((f) => <th key={f}>f({nm(f)})</th>)}
                  <th>{t('lab.common.product')}</th>
                  <th>{t('lab.common.posterior')}</th>
                </tr>
                {scored.map((c, k) => (
                  <tr key={c.name}>
                    <td className="left"><Swatch color={PALETTE[k % PALETTE.length]} />{nm(c.name)}</td>
                    <td>{c.priorLabel}</td>
                    {c.likelihoods.map((v, j) => <Ans key={j} k={`f${k}_${j}`} v={fmt(v, 4)} />)}
                    <Ans k={`score${k}`} v={c.score.toExponential(3)} className={c === best ? 'win' : ''} />
                    <td className={c === best ? 'win' : ''}>{c.posterior > 0 && c.posterior < 0.0001 ? '< 0.01' : fmt(c.posterior * 100, 2)}%</td>
                  </tr>
                ))}
              </TableWrap>
              {best && (
                <div className="mt-3">
                  <Note tone="good">
                    {ratio && Number.isFinite(ratio) ? (
                      <Trans i18nKey="lab.gnb.predict_ratio" values={{ name: nm(best.name), ratio: `${fmt(ratio, 1)}×` }} components={{ 1: <b />, 3: <b className="font-mono" /> }} />
                    ) : (
                      <Trans i18nKey="lab.gnb.predict" values={{ name: nm(best.name) }} components={{ 1: <b /> }} />
                    )}
                  </Note>
                </div>
              )}
            </Card>

            <div className="grid gap-5 md:grid-cols-2">
              <Note tone="warn" title={t('lab.common.exam_traps')}>
                <Trans i18nKey="lab.gnb.traps" components={{ 1: <i /> }} />
              </Note>
              <Note title={t('lab.gnb.naive_title')}>{t('lab.gnb.naive_text')}</Note>
            </div>
          </>
        )}
      </Workspace>
    </LabPage>
  )
}
