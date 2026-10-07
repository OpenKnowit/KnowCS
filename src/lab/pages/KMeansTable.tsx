import { useMemo, useState } from 'react'
import { kmeansTable, type KMetric } from '../../lib/kmeansTable'
import { niceTicks } from '../../lib/chart'
import { fmt, PALETTE } from '../format'
import { Ans, Btn, Card, LabPage, Note, Presets, Seg, Stat, TableWrap, Workspace } from '../ui'

interface Preset {
  id: string
  title: string
  note: string
  points: string
  seeds: string
  metric: KMetric
}

const PRESETS: Preset[] = [
  { id: 'medicine', title: 'Lecture 4: medicines', note: 'Seeds A(1,1), B(2,1) — practice problem', points: 'A 1 1\nB 2 1\nC 4 3\nD 5 4', seeds: '1 1\n2 1', metric: 'euclidean' },
  { id: 'line', title: 'Exam-style: 1-D data', note: 'Pattern of 2022 Spring Q5 (numbers changed)', points: 'P1 1\nP2 4\nP3 6\nP4 10\nP5 26\nP6 31', seeds: '4\n5', metric: 'euclidean' },
  { id: 'plane', title: 'Exam-style: 2-D, run to convergence', note: 'Pattern of 2023 Spring Q5 — ties go to C1', points: 'A 1 3\nB 2 4\nC 5 6\nD 6 7\nE 8 2', seeds: '1 3\n6 7', metric: 'euclidean' },
  { id: 'squared', title: 'Exam-style: squared distance + SSE', note: 'Pattern of 2023 Fall Q5', points: 'A 1 1\nB 2 3\nC 7 2\nD 8 3\nE 3 2\nF 9 1', seeds: '0 0\n8 0', metric: 'squared' },
  { id: 'empty', title: 'An empty cluster', note: 'A bad seed can end up owning nothing', points: 'A 1 1\nB 1.5 2\nC 3 4\nD 5 7\nE 3.5 5', seeds: '1 1\n10 10\n4 5', metric: 'euclidean' },
]

function parsePoints(text: string): { names: string[]; pts: number[][] } {
  const names: string[] = []
  const pts: number[][] = []
  text.split('\n').map((l) => l.trim()).filter(Boolean).forEach((line, i) => {
    const parts = line.split(/[\s,]+/)
    const hasName = Number.isNaN(Number(parts[0]))
    const nums = (hasName ? parts.slice(1) : parts).map(Number)
    if (!nums.length || nums.some((v) => !Number.isFinite(v))) throw new Error(`Line ${i + 1}: “${line}” is not a point.`)
    names.push(hasName ? parts[0] : `P${i + 1}`)
    pts.push(nums)
  })
  return { names, pts }
}

const metricLabel: Record<KMetric, string> = { euclidean: 'Euclidean', squared: 'squared Euclidean', manhattan: 'Manhattan' }

function Plot({ pts, names, centroids, next, assign }: { pts: number[][]; names: string[]; centroids: number[][]; next: number[][]; assign: number[] }) {
  const oneD = pts[0].length === 1
  const W = 560, H = oneD ? 160 : 360, P = 34
  const all = [...pts, ...centroids, ...next]
  const xs = all.map((p) => p[0])
  const ys = oneD ? [0] : all.map((p) => p[1])
  const pad = (a: number[]) => {
    const lo = Math.min(...a), hi = Math.max(...a)
    const m = (hi - lo || 2) * 0.12
    return [lo - m, hi + m]
  }
  const [x0, x1] = pad(xs)
  const [y0, y1] = oneD ? [-1, 1] : pad(ys)
  const X = (v: number) => P + ((v - x0) / (x1 - x0)) * (W - 2 * P)
  const Y = (v: number) => (oneD ? H / 2 : H - P - ((v - y0) / (y1 - y0)) * (H - 2 * P))
  const yOf = (p: number[]) => (oneD ? 0 : p[1])
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Points, centroids and their movement">
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
  const [preset, setPreset] = useState<string | null>('medicine')
  const [pointsText, setPointsText] = useState(PRESETS[0].points)
  const [seedsText, setSeedsText] = useState(PRESETS[0].seeds)
  const [metric, setMetric] = useState<KMetric>('euclidean')
  const [round, setRound] = useState(0)

  const parsed = useMemo(() => {
    try {
      const { names, pts } = parsePoints(pointsText)
      const { pts: seeds } = parsePoints(seedsText)
      const dim = pts[0]?.length
      if (pts.length < 2) throw new Error('Need at least two points.')
      if (pts.some((p) => p.length !== dim) || seeds.some((p) => p.length !== dim)) throw new Error('Every point and seed needs the same number of coordinates.')
      if (dim > 2) throw new Error('This page draws 1-D or 2-D data.')
      if (seeds.length < 2) throw new Error('Need at least two seeds (one per line).')
      return { names, pts, seeds, result: kmeansTable(pts, seeds, metric), error: null }
    } catch (e) {
      return { error: (e as Error).message } as const
    }
  }, [pointsText, seedsText, metric])

  const controls = (
      <>
        <Card title="Example">
          <Presets
            items={PRESETS}
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
        </Card>
        <Card title="Data" sub="name x [y], one per line">
          <div className="grid grid-cols-[1.4fr_1fr] gap-2">
            <label className="grid gap-1">
              <span className="text-xs font-bold text-slate-500">Points</span>
              <textarea value={pointsText} spellCheck={false} onChange={(e) => { setPointsText(e.target.value); setPreset(null); setRound(0) }} className="min-h-[140px] rounded-[10px] border border-slate-300 p-2 font-mono text-[13px] outline-none focus:ring-2 focus:ring-blue-300" />
            </label>
            <label className="grid gap-1">
              <span className="text-xs font-bold text-slate-500">Initial centroids</span>
              <textarea value={seedsText} spellCheck={false} onChange={(e) => { setSeedsText(e.target.value); setPreset(null); setRound(0) }} className="min-h-[140px] rounded-[10px] border border-slate-300 p-2 font-mono text-[13px] outline-none focus:ring-2 focus:ring-blue-300" />
            </label>
          </div>
          <div className="mt-3">
            <Seg label="Distance" value={metric} onChange={(v) => { setMetric(v); setRound(0) }} options={[{ v: 'euclidean', label: 'Euclidean' }, { v: 'squared', label: 'Squared' }, { v: 'manhattan', label: 'Manhattan' }]} />
          </div>
        </Card>
      </>
  )

  if (parsed.error !== null) {
    return (
      <LabPage id="kmeans-table" quiz>
        <Workspace controls={controls}>
          <Note tone="bad" title="Can’t read the input.">{parsed.error}</Note>
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

  return (
    <LabPage id="kmeans-table" quiz lead="Exactly the table the midterm asks for: distance from every point to every centroid, pick the nearest, average each cluster, repeat until no point changes cluster.">
      <Workspace
        wide
        controls={
          <>
            {controls}
            <Card title="Rounds">
              <div className="flex flex-wrap items-center gap-2">
                <Btn onClick={() => setRound(Math.max(0, r - 1))} disabled={r === 0}>← Prev</Btn>
                <span className="font-mono text-sm font-bold">Round {r + 1} / {result.rounds.length}</span>
                <Btn primary onClick={() => setRound(r + 1)} disabled={isLast}>Next →</Btn>
                <Btn onClick={() => setRound(result.rounds.length - 1)} disabled={isLast}>Run to the end</Btn>
              </div>
              <div className="mt-3 grid grid-cols-2 gap-2">
                <Stat k="Moved this round" v={r === 0 ? '—' : R.moved.length} tone={r > 0 && R.moved.length === 0 ? 'good' : 'none'} />
                <Stat k="SSE after update" v={fmt(R.sse, 3)} />
              </div>
            </Card>
          </>
        }
      >
        <Card step={1} title={`Round ${r + 1}: measure, assign, re-average`} sub="○ centroid used this round  ◆ new mean">
          <Plot pts={pts} names={names} centroids={R.centroids} next={R.next} assign={R.assign} />
        </Card>

        <Card step={2} title={`Distance table (${metricLabel[metric]})`} sub="Turn on “Quiz me” to fill it in">
          <TableWrap>
            <tr>
              <th className="left">Point</th>
              {R.centroids.map((c, j) => (
                <th key={j} style={{ color: PALETTE[j] }}>to C{j + 1} {point(c)}</th>
              ))}
              <th>Closest</th>
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
                  <b style={{ color: PALETTE[j] }}>New C{j + 1}</b>{' '}
                  {members.length ? (
                    <span className="font-mono">
                      = mean of {members.map((i) => names[i]).join(', ')} = <b>{point(c)}</b>
                    </span>
                  ) : (
                    <span className="text-rose-600">empty cluster — keeps {point(c)}</span>
                  )}
                </div>
              )
            })}
          </div>
          <p className="mt-2 text-xs text-slate-400">Highlighted rows changed cluster since the previous round. Ties go to the lower-numbered centroid.</p>
        </Card>

        <Note tone={isLast && result.converged ? 'good' : 'info'} title={isLast && result.converged ? 'Converged.' : 'Not finished yet.'}>
          {isLast && result.converged
            ? `In round ${r + 1} every point kept its cluster, so the means cannot move again. That is the sentence the marking scheme wants: “the cluster assignment remains the same”.`
            : `Press Next. K-Means stops only after a full round in which no point changes cluster (${result.rounds.length} rounds here${result.converged ? '' : ', without converging'}).`}
          {result.emptyClusters && ' One cluster ended up empty — a true/false favourite (2024 Spring: “possible” is True).'}
        </Note>

        <div className="grid gap-5 md:grid-cols-2">
          <Note tone="warn" title="Exam traps.">
            Use the distance the question names — 2023 Fall wanted <i>squared</i> distances. Round only when the paper says so (usually 2 decimals). Ties go to centroid 1 unless told otherwise. K-Means always converges, but not necessarily to the best clustering: a different seed can give a different answer.
          </Note>
          <Note title={`k = ${k}`}>
            Choosing k is not part of the algorithm. Exams ask for the elbow method: plot SSE against k and pick where the curve stops dropping sharply — not the k with the smallest SSE (that is always k = n).
          </Note>
        </div>
      </Workspace>
    </LabPage>
  )
}
