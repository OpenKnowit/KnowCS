import { useMemo, useState } from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { LabError } from '../../lib/labError'
import { kmeansTable, type KMetric } from '../../lib/kmeansTable'
import { niceTicks } from '../../lib/chart'
import { errorText, fmt, PALETTE } from '../format'
import { Ans, Btn, Card, LabPage, Note, Presets, Seg, Stat, TableWrap, Workspace } from '../ui'
import { KMeansPractice } from './KMeansDrag'
import { plotFrame } from './kmeansFrame'
import type { DragState } from './KMeansDrag'

interface Preset {
  id: string
  points: string
  seeds: string
  metric: KMetric
}

const PRESETS: Preset[] = [
  { id: 'medicine', points: 'A 1 1\nB 2 1\nC 4 3\nD 5 4', seeds: '1 1\n2 1', metric: 'euclidean' },
  { id: 'line', points: 'P1 1\nP2 4\nP3 6\nP4 10\nP5 26\nP6 31', seeds: '4\n5', metric: 'euclidean' },
  { id: 'plane', points: 'A 1 3\nB 2 4\nC 5 6\nD 6 7\nE 8 2', seeds: '1 3\n6 7', metric: 'euclidean' },
  { id: 'squared', points: 'A 1 1\nB 2 3\nC 7 2\nD 8 3\nE 3 2\nF 9 1', seeds: '0 0\n8 0', metric: 'squared' },
  { id: 'empty', points: 'A 1 1\nB 1.5 2\nC 3 4\nD 5 7\nE 3.5 5', seeds: '1 1\n10 10\n4 5', metric: 'euclidean' },
]

function parsePoints(text: string): { names: string[]; pts: number[][] } {
  const names: string[] = []
  const pts: number[][] = []
  text.split('\n').map((l) => l.trim()).filter(Boolean).forEach((line, i) => {
    const parts = line.split(/[\s,]+/)
    const hasName = Number.isNaN(Number(parts[0]))
    const nums = (hasName ? parts.slice(1) : parts).map(Number)
    if (!nums.length || nums.some((v) => !Number.isFinite(v))) throw new LabError('point_line', { line: i + 1, text: line }, `Line ${i + 1}: “${line}” is not a point.`)
    names.push(hasName ? parts[0] : `P${i + 1}`)
    pts.push(nums)
  })
  return { names, pts }
}


function Plot({ pts, names, centroids, next, assign }: { pts: number[][]; names: string[]; centroids: number[][]; next: number[][]; assign: number[] }) {
  const { t } = useTranslation()
  const { oneD, W, H, P, x0, x1, y0, y1, X, Y } = plotFrame(pts, centroids, next)
  const yOf = (p: number[]) => (oneD ? 0 : p[1])
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={t('lab.km.plot_aria')}>
      {niceTicks(x0, x1, 7).map((v) => (
        <g key={'x' + v}>
          {!oneD && <line x1={X(v)} x2={X(v)} y1={P} y2={H - P} stroke="#f1f5f9" />}
          <text x={X(v)} y={H - 10} textAnchor="middle" className="fill-slate-400 text-[11px]">{fmt(v, 2)}</text>
        </g>
      ))}
      {!oneD &&
        niceTicks(y0, y1, 6).map((v) => (
          <g key={'y' + v}>
            <line x1={P} x2={W - P} y1={Y(v)} y2={Y(v)} stroke="#f1f5f9" />
            <text x={P - 6} y={Y(v) + 4} textAnchor="end" className="fill-slate-400 text-[11px]">{fmt(v, 2)}</text>
          </g>
        ))}
      {oneD && <line x1={P} x2={W - P} y1={H / 2} y2={H / 2} stroke="#cbd5e1" />}
      {pts.map((p, i) => (
        <line key={'l' + i} x1={X(p[0])} y1={Y(yOf(p))} x2={X(centroids[assign[i]][0])} y2={Y(yOf(centroids[assign[i]]))} stroke={PALETTE[assign[i]]} strokeOpacity={0.25} strokeWidth={1.5} />
      ))}
      {centroids.map((c, k) => (
        <g key={'c' + k}>
          <line x1={X(c[0])} y1={Y(yOf(c))} x2={X(next[k][0])} y2={Y(yOf(next[k]))} stroke={PALETTE[k]} strokeWidth={2} strokeDasharray="4 3" markerEnd="url(#arr)" />
          <circle cx={X(c[0])} cy={Y(yOf(c))} r={9} fill="#fff" stroke={PALETTE[k]} strokeWidth={2.5} />
          <text x={X(c[0])} y={Y(yOf(c)) - 13} textAnchor="middle" className="text-[11px] font-extrabold" fill={PALETTE[k]}>C{k + 1}</text>
          <rect x={X(next[k][0]) - 7} y={Y(yOf(next[k])) - 7} width={14} height={14} transform={`rotate(45 ${X(next[k][0])} ${Y(yOf(next[k]))})`} fill={PALETTE[k]} />
        </g>
      ))}
      {pts.map((p, i) => (
        <g key={'p' + i}>
          <circle cx={X(p[0])} cy={Y(yOf(p))} r={6.5} fill={PALETTE[assign[i]]} stroke="#fff" strokeWidth={1.5} />
          <text x={X(p[0]) + 9} y={Y(yOf(p)) + (oneD ? 20 : 4)} className="fill-slate-600 text-[12px] font-bold">{names[i]}</text>
        </g>
      ))}
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#64748b" />
        </marker>
      </defs>
    </svg>
  )
}

export default function KMeansTable() {
  const { t } = useTranslation()
  const [preset, setPreset] = useState<string | null>('medicine')
  const [pointsText, setPointsText] = useState(PRESETS[0].points)
  const [seedsText, setSeedsText] = useState(PRESETS[0].seeds)
  const [metric, setMetric] = useState<KMetric>('euclidean')
  const [round, setRound] = useState(0)
  // practice: drag each point to its centroid; watch: the round as the table works it out
  const [mode, setMode] = useState<'practice' | 'watch'>('practice')
  const [prac, setPrac] = useState<{ key: string; state: DragState; revealed: boolean; last: number | null }>({ key: '', state: { guess: [] }, revealed: false, last: null })

  const parsed = useMemo(() => {
    try {
      const { names, pts } = parsePoints(pointsText)
      const { pts: seeds } = parsePoints(seedsText)
      const dim = pts[0]?.length
      if (pts.length < 2) throw new LabError('points_min', {}, 'Need at least two points.')
      if (pts.some((p) => p.length !== dim) || seeds.some((p) => p.length !== dim)) throw new LabError('dims', {}, 'Every point and seed needs the same number of coordinates.')
      if (dim > 2) throw new LabError('dims_max', {}, 'This page draws 1-D or 2-D data.')
      if (seeds.length < 2) throw new LabError('seeds_min', {}, 'Need at least two seeds (one per line).')
      return { ok: true as const, names, pts, seeds, result: kmeansTable(pts, seeds, metric) }
    } catch (e) {
      return { ok: false as const, error: e }
    }
  }, [pointsText, seedsText, metric])

  const controls = (
      <>
        <Presets title={t('lab.common.example')}
            items={PRESETS.map((p) => ({ id: p.id, title: t(`lab.km.presets.${p.id}.title`), note: t(`lab.km.presets.${p.id}.note`) }))}
            value={preset}
            onPick={(id) => {
              const p = PRESETS.find((q) => q.id === id)!
              setPreset(id)
              setPointsText(p.points)
              setSeedsText(p.seeds)
              setMetric(p.metric)
              setRound(0)
            }}
          />
        <Card title={t('lab.km.data')} sub={t('lab.km.data_sub')}>
          <div className="grid grid-cols-[1.4fr_1fr] gap-2">
            <label className="grid gap-1">
              <span className="text-xs font-bold text-slate-500">{t('lab.km.points')}</span>
              <textarea value={pointsText} spellCheck={false} autoCapitalize="off" autoCorrect="off" onChange={(e) => { setPointsText(e.target.value); setPreset(null); setRound(0) }} className="min-h-[140px] rounded-[10px] border border-slate-300 p-2 font-mono text-[13px] outline-none focus:ring-2 focus:ring-blue-300" />
            </label>
            <label className="grid gap-1">
              <span className="text-xs font-bold text-slate-500">{t('lab.km.seeds')}</span>
              <textarea value={seedsText} spellCheck={false} autoCapitalize="off" autoCorrect="off" onChange={(e) => { setSeedsText(e.target.value); setPreset(null); setRound(0) }} className="min-h-[140px] rounded-[10px] border border-slate-300 p-2 font-mono text-[13px] outline-none focus:ring-2 focus:ring-blue-300" />
            </label>
          </div>
          <div className="mt-3">
            <Seg label={t('lab.km.distance')} value={metric} onChange={(v) => { setMetric(v); setRound(0) }} options={[{ v: 'euclidean', label: t('lab.km.m_euclidean') }, { v: 'squared', label: t('lab.km.m_squared') }, { v: 'manhattan', label: t('lab.km.m_manhattan') }]} />
          </div>
        </Card>
      </>
  )

  if (!parsed.ok) {
    return (
      <LabPage id="kmeans-table" quiz>
        <Workspace controls={controls}>
          <Note tone="bad" title={t('lab.common.cant_read')}>{errorText(parsed.error, t)}</Note>
        </Workspace>
      </LabPage>
    )
  }

  const { names, pts, result } = parsed
  const r = Math.min(round, result.rounds.length - 1)
  const R = result.rounds[r]
  const k = R.centroids.length
  const isLast = r === result.rounds.length - 1
  const point = (c: number[]) => `(${c.map((v) => fmt(v, 2)).join(', ')})`

  // a fresh round (or new data) starts with every point unassigned
  const pracKey = `${r}|${pointsText}|${seedsText}|${metric}`
  if (prac.key !== pracKey) setPrac({ key: pracKey, state: { guess: pts.map(() => null) }, revealed: false, last: null })
  const guess = prac.key === pracKey ? prac.state.guess : pts.map(() => null)
  const right = guess.filter((g, i) => g === R.assign[i]).length
  const allRight = right === pts.length
  const assign = (next: DragState) => {
    const changed = next.guess.findIndex((g, i) => g !== guess[i])
    setPrac({ key: pracKey, state: next, revealed: false, last: changed >= 0 ? changed : prac.last })
  }
  const pick = (i: number, g: number) => assign({ guess: guess.map((v, j) => (j === i ? g : v)) })
  const dists = (i: number) => R.dist[i].map((dv, j) => `d(${names[i]}, C${j + 1}) = ${fmt(dv, 2)}`).join(', ')
  const lastI = prac.last
  const lastG = lastI === null ? null : guess[lastI]
  const tie = lastI !== null && R.dist[lastI].filter((dv) => Math.abs(dv - R.dist[lastI][R.assign[lastI]]) < 1e-12).length > 1
  const feedback = lastI === null || lastG === null ? null : lastG === R.assign[lastI]
    ? { tone: 'good' as const, text: t('lab.km.drop_right', { p: names[lastI], c: `C${lastG + 1}`, dists: dists(lastI) }) }
    : { tone: 'bad' as const, text: t('lab.km.drop_wrong', { p: names[lastI], c: `C${lastG + 1}`, best: `C${R.assign[lastI] + 1}`, dists: dists(lastI) }) + (tie ? ` ${t('lab.km.drop_tie')}` : '') }

  return (
    <LabPage id="kmeans-table" quiz>
      <Workspace
        wide
        controls={
          <>
            {controls}
            <Card title={t('lab.km.rounds')}>
              <div className="flex flex-wrap items-center gap-2">
                <Btn onClick={() => setRound(Math.max(0, r - 1))} disabled={r === 0}>{t('lab.common.prev')}</Btn>
                <span className="font-mono text-sm font-bold">{t('lab.km.round_of', { r: r + 1, n: result.rounds.length })}</span>
                <Btn primary onClick={() => setRound(r + 1)} disabled={isLast}>{t('lab.common.next')}</Btn>
                <Btn onClick={() => setRound(result.rounds.length - 1)} disabled={isLast}>{t('lab.km.run_end')}</Btn>
              </div>
              <div className="mt-3 grid grid-cols-2 gap-2">
                <Stat k={t('lab.km.moved')} v={r === 0 ? '—' : R.moved.length} tone={r > 0 && R.moved.length === 0 ? 'good' : 'none'} />
                <Stat k={t('lab.km.sse')} v={fmt(R.sse, 3)} />
              </div>
            </Card>
          </>
        }
      >
        <Card step={1} title={t('lab.km.round_title', { r: r + 1 })} sub={mode === 'practice' ? t('lab.km.drag_hint', { m: t(`lab.km.m_${metric}`) }) : t('lab.km.round_sub')}>
          <div className="mb-3">
            <Seg value={mode} onChange={setMode} options={[{ v: 'practice', label: t('lab.km.mode_practice') }, { v: 'watch', label: t('lab.km.mode_watch') }]} />
          </div>
          {mode === 'watch' ? (
            <Plot pts={pts} names={names} centroids={R.centroids} next={R.next} assign={R.assign} />
          ) : (
            <>
              <KMeansPractice pts={pts} names={names} centroids={R.centroids} next={R.next} answer={R.assign} state={prac.key === pracKey ? prac.state : { guess }} revealed={prac.revealed} onDrop={assign} />
              <div className="mt-3 grid gap-2">
                {feedback && <p className={`rounded-lg px-3 py-2 text-[13px] ${feedback.tone === 'good' ? 'bg-emerald-50 text-emerald-800' : 'bg-rose-50 text-rose-800'}`} aria-live="polite">{feedback.text}</p>}
                <div className="flex flex-wrap items-center gap-2">
                  <Stat k={t('lab.km.drag_progress')} v={`${right} / ${pts.length}`} tone={allRight ? 'good' : 'none'} />
                  {allRight && !prac.revealed && <Btn primary onClick={() => setPrac({ ...prac, revealed: true })}>{t('lab.km.drag_reveal')}</Btn>}
                  {prac.revealed && !isLast && <Btn primary onClick={() => setRound(r + 1)}>{t('lab.km.drag_next')}</Btn>}
                  <Btn onClick={() => setPrac({ key: pracKey, state: { guess: pts.map(() => null) }, revealed: false, last: null })} disabled={guess.every((g) => g === null)}>{t('lab.km.drag_again')}</Btn>
                </div>
                {allRight && <p className="text-[13px] text-slate-600">{prac.revealed ? R.next.map((c, j) => `C${j + 1} → ${point(c)}`).join('  ·  ') : t('lab.km.drag_all_right')}</p>}
                <details className="text-[12.5px] text-slate-600">
                  <summary className="cursor-pointer select-none font-semibold">{t('lab.km.drag_tap')}</summary>
                  <div className="mt-2 grid gap-1.5">
                    {pts.map((_, i) => (
                      <div key={i} className="flex flex-wrap items-center gap-1.5">
                        <span className="w-14 font-mono font-bold">{names[i]}</span>
                        {R.centroids.map((_, j) => (
                          <button key={j} type="button" aria-pressed={guess[i] === j} onClick={() => pick(i, j)} className={`rounded-md border px-2 py-0.5 font-mono text-xs font-bold ${guess[i] === j ? 'text-white' : 'bg-white'}`} style={guess[i] === j ? { background: PALETTE[j], borderColor: PALETTE[j] } : { color: PALETTE[j], borderColor: '#cbd5e1' }}>C{j + 1}</button>
                        ))}
                      </div>
                    ))}
                  </div>
                </details>
              </div>
            </>
          )}
        </Card>

        <Card step={2} title={t('lab.km.dist_table', { m: t(`lab.km.m_${metric}`) })} sub={t('lab.common.quiz_sub')}>
          <TableWrap>
            <tr>
              <th className="left">{t('lab.km.point')}</th>
              {R.centroids.map((c, j) => (
                <th key={j} style={{ color: PALETTE[j] }}>{t('lab.km.to', { c: `C${j + 1}` })} {point(c)}</th>
              ))}
              <th>{t('lab.km.closest')}</th>
            </tr>
            {pts.map((p, i) => (
              <tr key={i} className={r > 0 && R.moved.includes(i) ? 'cur' : ''}>
                <td className="left">{names[i]} {point(p)}</td>
                {R.dist[i].map((dv, j) => <Ans key={j} k={`d${r}_${i}_${j}`} v={fmt(dv, 2)} className={j === R.assign[i] ? 'win' : ''} />)}
                <Ans k={`a${r}_${i}`} v={`C${R.assign[i] + 1}`} />
              </tr>
            ))}
          </TableWrap>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            {R.next.map((c, j) => {
              const members = pts.map((_, i) => i).filter((i) => R.assign[i] === j)
              return (
                <div key={j} className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-[13px]">
                  <b style={{ color: PALETTE[j] }}>{t('lab.km.new_c', { c: `C${j + 1}` })}</b>{' '}
                  {members.length ? (
                    <span className="font-mono">
                      = {t('lab.km.mean_of', { names: members.map((i) => names[i]).join(', ') })} = <b>{point(c)}</b>
                    </span>
                  ) : (
                    <span className="text-rose-600">{t('lab.km.empty_keeps', { c: point(c) })}</span>
                  )}
                </div>
              )
            })}
          </div>
          <p className="mt-2 text-xs text-slate-500">{t('lab.km.table_note')}</p>
        </Card>

        <Note tone={isLast && result.converged ? 'good' : 'info'} title={isLast && result.converged ? t('lab.km.converged_title') : t('lab.km.not_done_title')}>
          {isLast && result.converged
            ? t('lab.km.converged', { r: r + 1 })
            : t(result.converged ? 'lab.km.not_done' : 'lab.km.not_done_never', { n: result.rounds.length })}
          {result.emptyClusters && ` ${t('lab.km.empty_note')}`}
        </Note>

        <div className="grid gap-5 md:grid-cols-2">
          <Note tone="warn" title={t('lab.common.exam_traps')}>
            <Trans i18nKey="lab.km.traps" components={{ 1: <i /> }} />
          </Note>
          <Note title={`k = ${k}`}>{t('lab.km.choose_k')}</Note>
        </div>
      </Workspace>
    </LabPage>
  )
}
