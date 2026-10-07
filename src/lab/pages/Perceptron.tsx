import { useEffect, useMemo, useState } from 'react'
import { fire, perceptronTable, type PerceptronSample, type StepActivation } from '../../lib/perceptronTable'
import { niceTicks } from '../../lib/chart'
import { fmt } from '../format'
import { Ans, Btn, Card, LabPage, Note, NumberField, Presets, Seg, TableWrap, Workspace } from '../ui'

interface Preset {
  id: string
  title: string
  note: string
  data: string
  w: [number, number]
  theta: number
  eta: number
  f: StepActivation
}

const PRESETS: Preset[] = [
  { id: 'and', title: 'Lecture 5: logical AND', note: 'w = (0.1, 0.5), θ = −0.8, η = 0.2', data: '0 0 0\n0 1 0\n1 0 0\n1 1 1', w: [0.1, 0.5], theta: -0.8, eta: 0.2, f: { high: 1, low: 0, rule: 'ge' } },
  { id: 'or', title: 'Lecture 5 practice: logical OR', note: 'Same start as AND', data: '0 0 0\n0 1 1\n1 0 1\n1 1 1', w: [0.1, 0.5], theta: -0.8, eta: 0.2, f: { high: 1, low: 0, rule: 'ge' } },
  { id: 'course', title: 'Exam-style: course ratings', note: 'Pattern of 2022 Fall Q6 — f = 1 only if z > 0', data: '1 2 0\n3 3 1\n5 4 0\n3 5 1\n2 3 1', w: [0, 0], theta: -1, eta: 1, f: { high: 1, low: 0, rule: 'gt' } },
  { id: 'pm', title: 'Exam-style: ±1 outputs', note: 'Pattern of 2022 Spring Q6 — f = 1 if z ≥ 0 else −1', data: '9 10 1\n0 0 -1\n7 4 1\n3 2 -1\n4 7 1\n1 1 -1\n5 3 1\n2 6 1', w: [1, 1], theta: 0, eta: 1, f: { high: 1, low: -1, rule: 'ge' } },
  { id: 'xor', title: 'XOR — never settles', note: 'Not linearly separable (lecture 5 “Problem”)', data: '0 0 0\n0 1 1\n1 0 1\n1 1 0', w: [0, 0], theta: 0, eta: 1, f: { high: 1, low: 0, rule: 'ge' } },
]

function parseData(text: string): PerceptronSample[] {
  const rows = text.split('\n').map((l) => l.trim()).filter(Boolean).map((l, i) => {
    const v = l.split(/[\s,]+/).map(Number)
    if (v.length !== 3 || v.some((x) => !Number.isFinite(x))) throw new Error(`Line ${i + 1}: write “x1 x2 T”.`)
    return { x: [v[0], v[1]], t: v[2] }
  })
  if (rows.length < 2) throw new Error('Need at least two samples.')
  return rows
}

function Plane({ data, w, theta, f, highlight, ghost }: { data: PerceptronSample[]; w: number[]; theta: number; f: StepActivation; highlight: number | null; ghost: { w: number[]; theta: number } | null }) {
  const W = 420, H = 380, P = 34
  const xs = data.map((d) => d.x[0])
  const ys = data.map((d) => d.x[1])
  const span = (a: number[]) => {
    const lo = Math.min(...a), hi = Math.max(...a)
    const m = Math.max(0.6, (hi - lo) * 0.18)
    return [lo - m, hi + m]
  }
  const [x0, x1] = span(xs)
  const [y0, y1] = span(ys)
  const X = (v: number) => P + ((v - x0) / (x1 - x0)) * (W - 2 * P)
  const Y = (v: number) => H - P - ((v - y0) / (y1 - y0)) * (H - 2 * P)
  // the half-plane where the neuron fires, clipped to the plot box (Sutherland–Hodgman, one edge)
  const box: [number, number][] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
  const z = ([x, y]: [number, number]) => w[0] * x + w[1] * y + theta
  const firing = (p: [number, number]) => fire(z(p), f) === f.high
  const region: [number, number][] = []
  box.forEach((a, i) => {
    const b = box[(i + 1) % box.length]
    if (firing(a)) region.push(a)
    if (firing(a) !== firing(b)) {
      const t = z(a) / (z(a) - z(b))
      region.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])
    }
  })
  const toPts = (poly: [number, number][]) => poly.map(([x, y]) => `${X(x)},${Y(y)}`).join(' ')
  const line = (ww: number[], th: number) => {
    // w1 x + w2 y + θ = 0, clipped to the box
    const pts: [number, number][] = []
    if (Math.abs(ww[1]) > 1e-9) for (const x of [x0, x1]) pts.push([x, -(ww[0] * x + th) / ww[1]])
    if (Math.abs(ww[0]) > 1e-9) for (const y of [y0, y1]) pts.push([-(ww[1] * y + th) / ww[0], y])
    const inside = pts.filter(([x, y]) => x >= x0 - 1e-9 && x <= x1 + 1e-9 && y >= y0 - 1e-9 && y <= y1 + 1e-9)
    return inside.length >= 2 ? inside.slice(0, 2) : null
  }
  const cur = line(w, theta)
  const old = ghost ? line(ghost.w, ghost.theta) : null
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto w-full max-w-[520px]" role="img" aria-label="Training points and the current decision line">
      <rect x={P} y={P} width={W - 2 * P} height={H - 2 * P} fill="#e11d48" opacity={0.06} />
      {region.length >= 3 && <polygon points={toPts(region)} fill="#2563eb" opacity={0.1} />}
      <rect x={P} y={P} width={W - 2 * P} height={H - 2 * P} fill="none" stroke="#e2e8f0" />
      {niceTicks(x0, x1, 6).map((v) => <text key={'x' + v} x={X(v)} y={H - 12} textAnchor="middle" className="fill-slate-400 text-[11px]">{fmt(v, 2)}</text>)}
      {niceTicks(y0, y1, 6).map((v) => <text key={'y' + v} x={P - 6} y={Y(v) + 4} textAnchor="end" className="fill-slate-400 text-[11px]">{fmt(v, 2)}</text>)}
      <text x={W - P} y={H - 12} textAnchor="end" className="fill-slate-500 text-[11px] font-bold">x₁</text>
      <text x={P + 4} y={P - 8} className="fill-slate-500 text-[11px] font-bold">x₂</text>
      {old && <line x1={X(old[0][0])} y1={Y(old[0][1])} x2={X(old[1][0])} y2={Y(old[1][1])} stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="5 4" />}
      {cur && <line x1={X(cur[0][0])} y1={Y(cur[0][1])} x2={X(cur[1][0])} y2={Y(cur[1][1])} stroke="#0f172a" strokeWidth={2.5} />}
      {data.map((d, i) => {
        const hi = d.t === f.high
        const cx = X(d.x[0]), cy = Y(d.x[1])
        return (
          <g key={i}>
            {highlight === i && <circle cx={cx} cy={cy} r={15} fill="none" stroke="#f59e0b" strokeWidth={3} />}
            {hi ? <circle cx={cx} cy={cy} r={7} fill="#2563eb" stroke="#fff" strokeWidth={2} /> : <rect x={cx - 6.5} y={cy - 6.5} width={13} height={13} fill="#e11d48" stroke="#fff" strokeWidth={2} />}
          </g>
        )
      })}
    </svg>
  )
}

export default function Perceptron() {
  const [preset, setPreset] = useState<string | null>('and')
  const [dataText, setDataText] = useState(PRESETS[0].data)
  const [w0, setW0] = useState<[number, number]>(PRESETS[0].w)
  const [theta0, setTheta0] = useState(PRESETS[0].theta)
  const [eta, setEta] = useState(PRESETS[0].eta)
  const [f, setF] = useState<StepActivation>(PRESETS[0].f)
  const [step, setStep] = useState(0) // rows shown; 0 = only the initial row
  const [playing, setPlaying] = useState(false)

  const parsed = useMemo(() => {
    try {
      const data = parseData(dataText)
      return { data, run: perceptronTable(data, w0, theta0, eta, f, 12), error: null }
    } catch (e) {
      return { data: [], run: null, error: (e as Error).message }
    }
  }, [dataText, w0, theta0, eta, f])

  const total = parsed.run?.rows.length ?? 0
  const shown = Math.min(step, total)
  const running = playing && shown < total
  useEffect(() => {
    if (!running) return
    const t = setTimeout(() => setStep((s) => s + 1), 650)
    return () => clearTimeout(t)
  }, [running, shown])

  const reset = () => {
    setStep(0)
    setPlaying(false)
  }
  const custom = () => {
    setPreset(null)
    reset()
  }

  const rows = parsed.run?.rows.slice(0, shown) ?? []
  const last = rows[rows.length - 1]
  const w = last ? last.w : w0
  const theta = last ? last.theta : theta0
  const prev = rows.length >= 2 ? rows[rows.length - 2] : rows.length === 1 ? { w: w0, theta: theta0 } : null
  const ruleText = `f(z) = ${f.high} if z ${f.rule === 'ge' ? '≥' : '>'} 0, otherwise ${f.low}`
  const epochOf = (i: number) => parsed.run?.rows[i]?.epoch ?? 1

  return (
    <LabPage id="perceptron" quiz lead="Feed one row at a time: compute O, and only if it is wrong move w and θ by η(T − O)x. Watch the decision line jump until every point is on its side — or never, if no straight line exists.">
      <Workspace
        wide
        controls={
          <>
            <Card title="Example">
              <Presets
                items={PRESETS}
                value={preset}
                onPick={(id) => {
                  const p = PRESETS.find((q) => q.id === id)!
                  setPreset(id)
                  setDataText(p.data)
                  setW0(p.w)
                  setTheta0(p.theta)
                  setEta(p.eta)
                  setF(p.f)
                  reset()
                }}
              />
            </Card>
            <Card title="Setup">
              <label className="grid gap-1">
                <span className="text-xs font-bold text-slate-500">Training rows: x1 x2 T</span>
                <textarea value={dataText} spellCheck={false} onChange={(e) => { setDataText(e.target.value); custom() }} className="min-h-[120px] rounded-[10px] border border-slate-300 p-2 font-mono text-[13px] outline-none focus:ring-2 focus:ring-blue-300" />
              </label>
              <div className="mt-3 grid grid-cols-4 gap-2">
                <NumberField label="w₁" value={w0[0]} step={0.1} onChange={(v) => { setW0([v, w0[1]]); custom() }} />
                <NumberField label="w₂" value={w0[1]} step={0.1} onChange={(v) => { setW0([w0[0], v]); custom() }} />
                <NumberField label="θ" value={theta0} step={0.1} onChange={(v) => { setTheta0(v); custom() }} />
                <NumberField label="η" value={eta} step={0.1} min={0} onChange={(v) => { setEta(v); custom() }} />
              </div>
              <div className="mt-3 grid gap-3">
                <Seg label="Outputs" value={f.low} onChange={(v) => { setF({ ...f, low: v }); custom() }} options={[{ v: 0, label: '1 / 0' }, { v: -1, label: '1 / −1' }]} />
                <Seg label="At z = 0 the output is" value={f.rule} onChange={(v) => { setF({ ...f, rule: v }); custom() }} options={[{ v: 'ge', label: `${f.high} (z ≥ 0)` }, { v: 'gt', label: `${f.low} (z > 0)` }]} />
              </div>
              <p className="mt-2 font-mono text-xs text-slate-500">{ruleText}</p>
            </Card>
          </>
        }
      >
        {parsed.error || !parsed.run ? (
          <Note tone="bad" title="Can’t read the rows.">{parsed.error}</Note>
        ) : (
          <>
            <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
              <Card step={1} title="The decision line" sub="blue side → output high · dashed = before this row">
                <Plane data={parsed.data} w={w} theta={theta} f={f} highlight={last ? last.index : null} ghost={prev} />
              </Card>
              <Card step={2} title={last ? `Row ${shown}: epoch ${last.epoch}, sample ${last.index + 1}` : 'Start'}>
                <div className="flex flex-wrap gap-2">
                  <Btn onClick={() => { setStep(Math.max(0, shown - 1)); setPlaying(false) }} disabled={shown === 0}>← Back</Btn>
                  <Btn primary onClick={() => setStep(shown + 1)} disabled={shown >= total}>Next row →</Btn>
                  <Btn onClick={() => setPlaying(!running)} disabled={shown >= total}>{running ? 'Pause' : 'Play'}</Btn>
                  <Btn onClick={() => { const e = epochOf(shown); const end = parsed.run!.rows.findIndex((r) => r.epoch > e); setStep(end === -1 ? total : end) }} disabled={shown >= total}>Finish epoch</Btn>
                  <Btn onClick={reset}>Reset</Btn>
                </div>
                {last ? (
                  <div className="formula mt-3">
                    {`z = w₁x₁ + w₂x₂ + θ = ${fmt(prev!.w[0], 4)}·${fmt(last.x[0], 3)} + ${fmt(prev!.w[1], 4)}·${fmt(last.x[1], 3)} + (${fmt(prev!.theta, 4)}) = `}<b className="text-blue-700">{fmt(last.z, 4)}</b>
                    {`\nO = f(${fmt(last.z, 4)}) = `}<b>{last.o}</b>{`   T = ${last.t}   T − O = `}<b className={last.updated ? 'text-rose-600' : 'text-emerald-600'}>{last.t - last.o}</b>
                    {last.updated
                      ? `\nΔw₁ = η(T−O)x₁ = ${fmt(last.dw[0], 4)}   Δw₂ = ${fmt(last.dw[1], 4)}   Δθ = ${fmt(last.dTheta, 4)}\nnew w = (${fmt(last.w[0], 4)}, ${fmt(last.w[1], 4)}), θ = ${fmt(last.theta, 4)}`
                      : '\nCorrect — nothing changes.'}
                  </div>
                ) : (
                  <p className="mt-3 text-[13px] text-slate-500">Start: w = ({fmt(w0[0], 3)}, {fmt(w0[1], 3)}), θ = {fmt(theta0, 3)}, η = {fmt(eta, 3)}. Press <b>Next row</b>.</p>
                )}
                <div className="mt-3">
                  {shown >= total ? (
                    parsed.run.converged ? (
                      <Note tone="good" title="Converged.">Epoch {parsed.run.epochs} made no changes, so every row is classified correctly. The data is linearly separable.</Note>
                    ) : (
                      <Note tone="bad" title="No convergence after 12 epochs.">The weights keep changing. If no straight line can separate the classes (XOR), the perceptron rule never stops.</Note>
                    )
                  ) : (
                    <Note>An epoch has converged only when a <b>whole pass</b> makes no update — an exam answer of “not converged in 1 epoch” needs exactly that reason.</Note>
                  )}
                </div>
              </Card>
            </div>

            <Card step={3} title="The exam table" sub="Turn on “Quiz me” to fill it in row by row">
              <TableWrap>
                <tr>
                  <th>Ep.</th><th>x₁</th><th>x₂</th><th>T</th><th>O</th><th>Δw₁</th><th>w₁</th><th>Δw₂</th><th>w₂</th><th>Δθ</th><th>θ</th>
                </tr>
                <tr className="dim">
                  <td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>{fmt(w0[0], 4)}</td><td>—</td><td>{fmt(w0[1], 4)}</td><td>—</td><td>{fmt(theta0, 4)}</td>
                </tr>
                {rows.map((r, i) => (
                  <tr key={i} className={i === rows.length - 1 ? 'cur' : ''}>
                    <td>{r.epoch}</td>
                    <td>{fmt(r.x[0], 3)}</td>
                    <td>{fmt(r.x[1], 3)}</td>
                    <td>{r.t}</td>
                    <Ans k={`o${i}`} v={r.o} />
                    <Ans k={`dw1${i}`} v={fmt(r.dw[0], 4)} />
                    <Ans k={`w1${i}`} v={fmt(r.w[0], 4)} className={r.updated ? 'changed' : ''} />
                    <Ans k={`dw2${i}`} v={fmt(r.dw[1], 4)} />
                    <Ans k={`w2${i}`} v={fmt(r.w[1], 4)} className={r.updated ? 'changed' : ''} />
                    <Ans k={`dt${i}`} v={fmt(r.dTheta, 4)} />
                    <Ans k={`t${i}`} v={fmt(r.theta, 4)} className={r.updated ? 'changed' : ''} />
                  </tr>
                ))}
              </TableWrap>
            </Card>

            <div className="grid gap-5 md:grid-cols-2">
              <Note tone="warn" title="Read the activation first.">
                Papers disagree: lecture 5 and 2022 Spring output high when z ≥ 0, 2022 Fall only when z &gt; 0, and outputs may be 0/1 or −1/+1. With −1/+1 outputs a mistake gives T − O = ±2, so the step doubles. Rows that land exactly on z = 0 are where students lose marks.
              </Note>
              <Note title="Linearly separable?">
                The perceptron convergence theorem: if a separating line exists, this rule finds one in finitely many updates. If none exists (XOR, or “success iff both ratings are 2 or 3” in 2022 Fall), you need a hidden layer — see the next page.
              </Note>
            </div>
          </>
        )}
      </Workspace>
    </LabPage>
  )
}
