import { useEffect, useMemo, useRef, useState } from 'react'
import { contrastStretch, histogram, otsuIterations, threshold } from '../../lib/otsu'
import { mulberry32 } from '../../lib/crossval'
import { fmt } from '../format'
import { Ans, Btn, Card, LabPage, Note, Presets, Seg, Slider, TableWrap, Workspace } from '../ui'

interface Img {
  w: number
  h: number
  px: number[]
}

// --- procedural test images (no binary assets) ---
function lowContrast(): Img {
  const w = 96, h = 72
  const rnd = mulberry32(11)
  const px: number[] = []
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let v = 0.75 - y / h / 2 // sky gradient
      const hill = h * 0.62 + Math.sin(x / 11) * 6
      if (y > hill) v = 0.28 + 0.1 * Math.sin(x / 5 + y / 7)
      if ((x - 70) ** 2 + (y - 18) ** 2 < 90) v = 0.95 // sun
      if (x > 18 && x < 34 && y > hill - 14 && y < hill + 4) v = 0.12 // house
      v += (rnd() - 0.5) * 0.06
      px.push(Math.round(95 + Math.min(1, Math.max(0, v)) * 70)) // squeezed into 95…165
    }
  return { w, h, px }
}
function coins(): Img {
  const w = 96, h = 72
  const rnd = mulberry32(5)
  const discs = [[24, 24, 13], [62, 30, 16], [40, 54, 11], [80, 58, 9]]
  const px: number[] = []
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const on = discs.some(([cx, cy, r]) => (x - cx) ** 2 + (y - cy) ** 2 < r * r)
      const base = on ? 185 : 70
      px.push(Math.round(Math.min(255, Math.max(0, base + (rnd() - 0.5) * 70 + (on ? 0 : x / 4)))))
    }
  return { w, h, px }
}
const grid = (rows: number[][]): Img => ({ w: rows[0].length, h: rows.length, px: rows.flat() })

const SOURCES = [
  { id: 'photo', title: 'Low-contrast scene', note: 'All pixels squeezed into 95…165', make: lowContrast, T0: undefined as number | undefined },
  { id: 'coins', title: 'Coins on a table', note: 'Two clear peaks — Otsu’s favourite', make: coins, T0: undefined },
  { id: 'grid4', title: 'Exam-style: 4×4 grid', note: 'Pattern of Final 2022 A Q4(b)(iii) — start at the mean', make: () => grid([[0, 3, 5, 5], [4, 10, 13, 11], [9, 16, 19, 15], [9, 15, 17, 13]]), T0: undefined },
  { id: 'grid3', title: 'Exam-style: 3×3, T₀ = 100', note: 'Pattern of Final 2024 Q5(b)', make: () => grid([[3, 6, 10], [20, 30, 60], [120, 140, 130]]), T0: 100 },
]

type Op = 'original' | 'stretch' | 'flip' | 'threshold' | 'otsu'

function apply(img: Img, op: Op, T: number): Img {
  if (op === 'stretch') return { ...img, px: contrastStretch(img.px) }
  if (op === 'threshold' || op === 'otsu') return { ...img, px: threshold(img.px, T) }
  if (op === 'flip') {
    const px: number[] = []
    for (let y = 0; y < img.h; y++) for (let x = 0; x < img.w; x++) px.push(img.px[y * img.w + (img.w - 1 - x)])
    return { ...img, px }
  }
  return img
}

function Picture({ img, label, small = false }: { img: Img; label: string; small?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    c.width = img.w
    c.height = img.h
    const ctx = c.getContext('2d')!
    const data = ctx.createImageData(img.w, img.h)
    img.px.forEach((v, i) => {
      data.data.set([v, v, v, 255], i * 4)
    })
    ctx.putImageData(data, 0, 0)
  }, [img])
  if (img.w <= 6 && !small)
    return (
      <div className="inline-grid gap-1 font-mono text-[13px]" style={{ gridTemplateColumns: `repeat(${img.w}, 46px)` }} aria-label={label}>
        {img.px.map((v, i) => (
          <span key={i} className="grid h-[42px] place-items-center rounded-md border border-slate-200 font-bold" style={{ background: `rgb(${v},${v},${v})`, color: v > 120 ? '#0f172a' : '#fff' }}>{v}</span>
        ))}
      </div>
    )
  return <canvas ref={ref} aria-label={label} className="w-full rounded-lg border border-slate-200 [image-rendering:pixelated]" style={{ aspectRatio: `${img.w} / ${img.h}` }} />
}

function Histogram({ px, T, mu, height = 150, bins = 64 }: { px: number[]; T?: number; mu?: [number, number]; height?: number; bins?: number }) {
  const h = histogram(px, bins)
  const max = Math.max(...h, 1)
  const W = 520, P = 22
  const X = (v: number) => P + (v / 255) * (W - 2 * P)
  const bw = (W - 2 * P) / bins
  return (
    <svg viewBox={`0 0 ${W} ${height + 26}`} className="w-full" role="img" aria-label="Grey-level histogram">
      {h.map((n, i) => (
        <rect key={i} x={P + i * bw + 0.5} y={height - (n / max) * (height - 8)} width={Math.max(1, bw - 1)} height={(n / max) * (height - 8)} fill={T !== undefined && (i + 0.5) * (256 / bins) > T ? '#93c5fd' : '#64748b'} />
      ))}
      <line x1={P} x2={W - P} y1={height} y2={height} stroke="#cbd5e1" />
      {[0, 64, 128, 192, 255].map((v) => <text key={v} x={X(v)} y={height + 16} textAnchor="middle" className="fill-slate-400 text-[11px]">{v}</text>)}
      {mu?.map((m, i) => Number.isFinite(m) && (
        <g key={i}>
          <line x1={X(m)} x2={X(m)} y1={6} y2={height} stroke={i ? '#2563eb' : '#334155'} strokeDasharray="3 3" />
          <text x={X(m)} y={14} textAnchor="middle" className="text-[11px] font-bold" fill={i ? '#2563eb' : '#334155'}>μ{i + 1}</text>
        </g>
      ))}
      {T !== undefined && (
        <g>
          <line x1={X(T)} x2={X(T)} y1={0} y2={height} stroke="#e11d48" strokeWidth={2} />
          <text x={X(T) + 4} y={height - 6} className="fill-rose-600 text-[11px] font-extrabold">T = {fmt(T, 2)}</text>
        </g>
      )}
    </svg>
  )
}

const MATCH_OPS: { op: Op; name: string; why: string }[] = [
  { op: 'stretch', name: 'Contrast stretching', why: 'same shape, spread over 0…255' },
  { op: 'threshold', name: 'Binary thresholding', why: 'only two bars: 0 and 255' },
  { op: 'flip', name: 'Horizontal flip', why: 'identical histogram — pixels just move' },
]

export default function Otsu() {
  const [src, setSrc] = useState('photo')
  const [op, setOp] = useState<Op>('otsu')
  const [manualT, setManualT] = useState(128)
  const [iter, setIter] = useState(0)
  const [guess, setGuess] = useState<Record<string, string>>({})
  const [checked, setChecked] = useState(false)

  const source = SOURCES.find((s) => s.id === src)!
  const img = useMemo(() => source.make(), [source])
  const iters = useMemo(() => otsuIterations(img.px, source.T0), [img, source])
  const it = iters[Math.min(iter, iters.length - 1)]
  const otsuT = iters[iters.length - 1].next
  const T = op === 'threshold' ? manualT : op === 'otsu' ? (iter >= iters.length ? otsuT : it.next) : undefined
  const out = apply(img, op, T ?? 128)
  const small = img.w <= 6
  // shuffled histogram order for the matching game, fixed per image
  const order = useMemo(() => [2, 0, 1], [])
  const bins = small ? 32 : 64

  return (
    <LabPage id="otsu" quiz lead="A point operation changes each pixel on its own, so you can read its effect straight off the histogram. Otsu’s method picks the threshold automatically: split at T, average both sides, move T to the midpoint, repeat.">
      <Workspace
        wide
        controls={
          <>
            <Card title="Image">
              <Presets items={SOURCES} value={src} onPick={(id) => { setSrc(id); setIter(0); setChecked(false); setGuess({}) }} />
            </Card>
            <Card title="Operation">
              <div className="grid gap-3">
                <Seg value={op} onChange={(v) => { setOp(v); setIter(0) }} options={[{ v: 'original', label: 'Original' }, { v: 'stretch', label: 'Stretch' }, { v: 'flip', label: 'Flip' }, { v: 'threshold', label: 'Threshold' }, { v: 'otsu', label: 'Otsu' }]} />
                {op === 'threshold' && <Slider label="Threshold T (pixel > T → 255)" value={manualT} min={0} max={255} step={1} onChange={setManualT} />}
                {op === 'stretch' && (
                  <div className="formula">I_new = (I − {Math.min(...img.px)}) / ({Math.max(...img.px)} − {Math.min(...img.px)}) × 255</div>
                )}
                {op === 'otsu' && (
                  <div className="flex flex-wrap items-center gap-2">
                    <Btn onClick={() => setIter(Math.max(0, iter - 1))} disabled={iter === 0}>← Back</Btn>
                    <span className="font-mono text-[13px] font-bold">iteration {Math.min(iter, iters.length - 1) + 1} / {iters.length}</span>
                    <Btn primary onClick={() => setIter(iter + 1)} disabled={iter >= iters.length - 1}>Next →</Btn>
                  </div>
                )}
              </div>
            </Card>
          </>
        }
      >
        <div className="grid gap-5 xl:grid-cols-2">
          <Card step={1} title="Before">
            <Picture img={img} label="Original image" />
            <div className="mt-3"><Histogram px={img.px} bins={bins} T={op === 'otsu' ? it.T : op === 'threshold' ? manualT : undefined} mu={op === 'otsu' ? [it.mu1, it.mu2] : undefined} /></div>
          </Card>
          <Card step={2} title={op === 'original' ? 'After (no change)' : `After: ${op === 'otsu' ? `Otsu threshold T = ${fmt(T!, 2)}` : op}`}>
            <Picture img={out} label="Transformed image" />
            <div className="mt-3"><Histogram px={out.px} bins={bins} /></div>
          </Card>
        </div>

        <Card step={3} title="Otsu iterations" sub="Turn on “Quiz me” to fill the table">
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
            <TableWrap>
              <tr>
                <th>#</th><th>T</th><th>R1 (≤ T)</th><th>μ1</th><th>R2 (&gt; T)</th><th>μ2</th><th>(μ1 + μ2) / 2</th>
              </tr>
              {iters.map((x, i) => (
                <tr key={i} className={i === Math.min(iter, iters.length - 1) && op === 'otsu' ? 'cur' : ''}>
                  <td>{i + 1}</td>
                  {i === 0 ? <Ans k="T0" v={fmt(x.T, 3)} /> : <td>{fmt(x.T, 3)}</td>}
                  <td>{x.count1} px</td>
                  <Ans k={`m1_${i}`} v={fmt(x.mu1, 3)} />
                  <td>{x.count2} px</td>
                  <Ans k={`m2_${i}`} v={fmt(x.mu2, 3)} />
                  <Ans k={`nt_${i}`} v={fmt(x.next, 3)} />
                </tr>
              ))}
            </TableWrap>
            <div className="grid content-start gap-3">
              <Note>
                {source.T0 !== undefined ? <>The question fixes T₀ = {source.T0}.</> : <>T₀ is the mean intensity, {fmt(iters[0].T, 3)}.</>} Stop when T stops changing — here after {iters.length} iteration{iters.length > 1 ? 's' : ''}, at <b>T = {fmt(otsuT, 3)}</b>.
              </Note>
              <Note tone="good" title="Why Otsu?">It chooses T automatically from the histogram, so two people thresholding the same image get the same result (Final 2024 Q5b).</Note>
            </div>
          </div>
        </Card>

        {!small && (
          <Card step={4} title="Match each image to its histogram" sub="Final 2024 Q5(a) pattern">
            <div className="grid gap-4 lg:grid-cols-3">
              {MATCH_OPS.map((m, i) => (
                <div key={m.op} className="grid gap-2">
                  <Picture img={apply(img, m.op, otsuT)} label={`Transformed image ${'abc'[i]}`} small />
                  <label className="flex items-center gap-2 text-[13px] font-bold">
                    ({'abc'[i]}) goes with
                    <select value={guess[m.op] ?? ''} onChange={(e) => { setGuess({ ...guess, [m.op]: e.target.value }); setChecked(false) }} className="rounded-lg border border-slate-300 px-2 py-1">
                      <option value="">—</option>
                      {order.map((k, j) => <option key={j} value={MATCH_OPS[k].op}>histogram {'def'[j]}</option>)}
                    </select>
                    {checked && <span className={guess[m.op] === m.op ? 'text-emerald-600' : 'text-rose-600'}>{guess[m.op] === m.op ? '✓' : '✗'}</span>}
                  </label>
                  {checked && <span className="text-xs text-slate-500">{m.name}: {m.why}</span>}
                </div>
              ))}
            </div>
            <div className="mt-4 grid gap-3 lg:grid-cols-3">
              {order.map((k, j) => (
                <div key={j} className="rounded-xl border border-slate-200 p-2">
                  <div className="text-xs font-bold text-slate-500">histogram {'def'[j]}</div>
                  <Histogram px={apply(img, MATCH_OPS[k].op, otsuT).px} height={90} bins={bins} />
                </div>
              ))}
            </div>
            <div className="mt-3"><Btn primary onClick={() => setChecked(true)}>Check</Btn></div>
          </Card>
        )}

        <div className="grid gap-5 md:grid-cols-2">
          <Card title="Point, local or global?" sub="Final 2022 A Q1(f)">
            <TableWrap>
              <tr><th className="left">Operation</th><th className="left">Type</th><th className="left">Why</th></tr>
              <tr><td className="left">Contrast stretching, thresholding</td><td className="left">point</td><td className="left font-sans !whitespace-normal">new value depends only on the old value</td></tr>
              <tr><td className="left">Convolution (blur, edges)</td><td className="left">local</td><td className="left font-sans !whitespace-normal">needs the neighbourhood</td></tr>
              <tr><td className="left">Flip, rotate, translate</td><td className="left">not point</td><td className="left font-sans !whitespace-normal">the value comes from a different position</td></tr>
              <tr><td className="left">Choosing T with Otsu</td><td className="left">global</td><td className="left font-sans !whitespace-normal">uses the whole histogram</td></tr>
            </TableWrap>
          </Card>
          <Note tone="warn" title="Exam traps.">
            Write which side of T a pixel equal to T goes to (here R1: ≤ T). Thresholding outputs only two values, so its histogram has two bars. A flip leaves the histogram untouched — that is how the 2024 final expected you to spot it.
          </Note>
        </div>
      </Workspace>
    </LabPage>
  )
}
