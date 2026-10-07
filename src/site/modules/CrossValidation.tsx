import { useState } from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { crossValidate, sortedLabels, type Ordering } from '../../lib/crossval'
import { fmt, PALETTE } from '../format'
import { Ans, Btn, Card, LabPage, Note, Presets, Seg, Slider, Stat, TableWrap, Workspace } from '../ui'

interface Preset {
  id: string
  classes: number
  per: number
  d: number
}

const PRESETS: Preset[] = [
  { id: 'four', classes: 4, per: 20, d: 2 },
  { id: 'five', classes: 5, per: 6, d: 5 },
  { id: 'six', classes: 2, per: 3, d: 3 },
]

const CLASS_NAMES = 'ABCDEF'

export default function CrossValidation() {
  const { t } = useTranslation()
  const [preset, setPreset] = useState<string | null>('four')
  const [classes, setClasses] = useState(4)
  const [per, setPer] = useState(20)
  const [d, setD] = useState(2)
  const [ordering, setOrdering] = useState<Ordering>('sorted')
  const [seed, setSeed] = useState(7)

  const labels = sortedLabels(classes, per)
  const n = labels.length
  const D = Math.min(d, n)
  const res = crossValidate(labels, D, ordering, seed)
  const curve = Array.from({ length: Math.min(n, 12) - 1 }, (_, i) => i + 2).map((k) => ({ k, acc: crossValidate(labels, k, ordering, seed).mean }))
  const cell = n > 60 ? 9 : n > 30 ? 13 : 22
  const missing = res.folds.filter((f) => f.testClasses.some((c) => !f.trainClasses.includes(c)))
  const custom = () => setPreset(null)

  return (
    <LabPage id="cross-validation" quiz lead={t('lab.cv.lead')}>
      <Workspace
        controls={
          <>
            <Card title={t('lab.common.dataset')}>
              <Presets
                items={PRESETS.map((p) => ({ id: p.id, title: t(`lab.cv.presets.${p.id}.title`), note: t(`lab.cv.presets.${p.id}.note`) }))}
                value={preset}
                onPick={(id) => {
                  const p = PRESETS.find((q) => q.id === id)!
                  setPreset(id)
                  setClasses(p.classes)
                  setPer(p.per)
                  setD(p.d)
                  setOrdering('sorted')
                }}
              />
            </Card>
            <Card title={t('lab.common.settings')}>
              <div className="grid gap-3">
                <Slider label={t('lab.cv.classes')} value={classes} min={2} max={6} step={1} onChange={(v) => { setClasses(v); custom() }} />
                <Slider label={t('lab.cv.per_class')} value={per} min={2} max={30} step={1} onChange={(v) => { setPer(v); custom() }} />
                <Slider label={t('lab.cv.folds')} value={D} min={2} max={Math.min(n, 20)} step={1} onChange={(v) => { setD(v); custom() }} />
                <Seg label={t('lab.cv.before')} value={ordering} onChange={setOrdering} options={[{ v: 'sorted', label: t('lab.cv.keep_sorted') }, { v: 'shuffled', label: t('lab.cv.shuffle') }, { v: 'stratified', label: t('lab.cv.stratify') }]} />
                {ordering === 'shuffled' && <Btn onClick={() => setSeed((s) => s + 1)}>{t('lab.cv.reshuffle')}</Btn>}
              </div>
            </Card>
            <Note title={t('lab.cv.assumption_title')}>{t('lab.cv.assumption')}</Note>
          </>
        }
      >
        <Card step={1} title={t('lab.cv.cut', { n, d: D })} sub={t(`lab.cv.order_${ordering}`)}>
          <div className="flex flex-wrap gap-2">
            {res.folds.map((f) => (
              <div key={f.fold} className="rounded-xl border border-slate-200 bg-slate-50 p-2">
                <div className="mb-1 text-[11px] font-extrabold uppercase tracking-wider text-slate-400">{t('lab.cv.fold', { k: f.fold + 1 })}</div>
                <div className="flex flex-wrap gap-[3px]" style={{ maxWidth: Math.max(4, Math.ceil(Math.sqrt(f.members.length * 3))) * (cell + 3) }}>
                  {f.members.map((i) => (
                    <span key={i} title={`x${i} · ${t('lab.cv.class_x', { c: CLASS_NAMES[labels[i]] })}`} className="grid place-items-center rounded-[4px] text-[10px] font-bold text-white" style={{ width: cell, height: cell, background: PALETTE[labels[i]] }}>
                      {cell >= 18 ? CLASS_NAMES[labels[i]] : ''}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="mt-3 flex flex-wrap gap-3 text-xs text-slate-600">
            {Array.from({ length: classes }, (_, c) => (
              <span key={c} className="inline-flex items-center gap-1.5">
                <i className="inline-block h-3 w-3 rounded-[3px]" style={{ background: PALETTE[c] }} />{t('lab.cv.class_x', { c: CLASS_NAMES[c] })}
              </span>
            ))}
          </div>
        </Card>

        <Card step={2} title={t('lab.cv.rounds')}>
          <div className="grid gap-2">
            {res.folds.map((f) => (
              <div key={f.fold} className="grid items-center gap-3 md:grid-cols-[110px_minmax(0,1fr)_150px]">
                <div className="text-[13px] font-bold text-slate-600">{t('lab.cv.test_fold', { k: f.fold + 1 })}</div>
                <div className="flex h-6 overflow-hidden rounded-md border border-slate-200">
                  {res.order.map((i, pos) => {
                    const inTest = f.members.includes(i)
                    const known = f.trainClasses.includes(labels[i])
                    return (
                      <span
                        key={pos}
                        className="h-full flex-1"
                        style={{ background: PALETTE[labels[i]], opacity: inTest ? 1 : 0.18, boxShadow: inTest && !known ? 'inset 0 -4px 0 #0f172a' : undefined }}
                        title={inTest ? (known ? t('lab.cv.tip_seen') : t('lab.cv.tip_unseen')) : t('lab.cv.tip_train')}
                      />
                    )
                  })}
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-200">
                    <i className="block h-full rounded-full" style={{ width: `${f.accuracy * 100}%`, background: f.accuracy === 1 ? '#059669' : f.accuracy === 0 ? '#e11d48' : '#d97706' }} />
                  </div>
                  <span className="w-12 text-right font-mono text-[13px] font-bold">{fmt(f.accuracy * 100, 0)}%</span>
                </div>
              </div>
            ))}
          </div>
          <p className="mt-2 text-xs text-slate-400">{t('lab.cv.legend')}</p>
        </Card>

        <div className="grid gap-5 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
          <Card step={3} title={t('lab.cv.exam_table')} sub={t('lab.common.quiz_sub')}>
            <TableWrap>
              <tr>
                <th>{t('lab.cv.th_fold')}</th>
                <th className="left">{t('lab.cv.th_train')}</th>
                <th className="left">{t('lab.cv.th_test')}</th>
                <th>{t('lab.cv.th_correct')}</th>
                <th>{t('lab.cv.th_acc')}</th>
              </tr>
              {res.folds.map((f) => (
                <tr key={f.fold}>
                  <td>{f.fold + 1}</td>
                  <td className="left">{f.trainClasses.map((c) => CLASS_NAMES[c]).join(', ')}</td>
                  <td className="left">
                    {f.testClasses.map((c) => (
                      <span key={c} className={f.trainClasses.includes(c) ? '' : 'font-extrabold text-rose-600'}>{CLASS_NAMES[c]} </span>
                    ))}
                  </td>
                  <td>{f.correct} / {f.members.length}</td>
                  <Ans k={`acc${f.fold}`} v={fmt(f.accuracy * 100, 1)} />
                </tr>
              ))}
            </TableWrap>
            <div className="mt-3 grid grid-cols-2 gap-2.5">
              <Stat k={t('lab.cv.mean')} v={`${fmt(res.mean * 100, 1)}%`} tone={res.mean === 1 ? 'good' : res.mean < 0.5 ? 'bad' : 'warn'} />
              <Stat k={t('lab.cv.unseen_rounds')} v={`${missing.length} / ${D}`} tone={missing.length ? 'bad' : 'good'} />
            </div>
          </Card>

          <Card step={4} title={t('lab.cv.every_d')} sub={t(`lab.cv.order_short_${ordering}`)}>
            <svg viewBox="0 0 360 200" className="w-full" role="img" aria-label={t('lab.cv.every_d')}>
              {[0, 0.5, 1].map((y) => (
                <g key={y}>
                  <line x1={34} x2={350} y1={170 - y * 150} y2={170 - y * 150} stroke="#e2e8f0" />
                  <text x={28} y={174 - y * 150} textAnchor="end" className="fill-slate-400 text-[11px]">{y * 100}%</text>
                </g>
              ))}
              {curve.map((p, i) => {
                const bw = 316 / curve.length
                const x = 34 + i * bw
                return (
                  <g key={p.k} onClick={() => { setD(p.k); custom() }} className="cursor-pointer">
                    <rect x={x + 3} y={170 - p.acc * 150} width={bw - 6} height={Math.max(1, p.acc * 150)} rx={3} fill={p.k === D ? '#2563eb' : '#93c5fd'} />
                    <text x={x + bw / 2} y={186} textAnchor="middle" className={`text-[11px] ${p.k === D ? 'fill-blue-700 font-bold' : 'fill-slate-400'}`}>{p.k}</text>
                  </g>
                )
              })}
              <text x={192} y={199} textAnchor="middle" className="fill-slate-400 text-[10px]">{t('lab.cv.click_bar')}</text>
            </svg>
          </Card>
        </div>

        <div className="grid gap-5 md:grid-cols-2">
          <Note tone="warn" title={t('lab.cv.how_title')}>
            <Trans i18nKey="lab.cv.how" components={{ 1: <b />, 3: <b /> }} />
          </Note>
          <Note title={t('lab.cv.loo_title')}>{t('lab.cv.loo')}</Note>
        </div>
      </Workspace>
    </LabPage>
  )
}
