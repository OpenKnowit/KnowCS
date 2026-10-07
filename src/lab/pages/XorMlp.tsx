import { useEffect, useMemo, useRef, useState } from 'react'
import { LECTURE_NET, XOR, backprop, forward, loss, type Act, type Net } from '../../lib/mlp'
import { mulberry32 } from '../../lib/crossval'
import { fmt, int } from '../format'
import { Ans, Btn, Card, LabPage, Note, NumberField, Presets, Seg, Slider, TableWrap, Workspace } from '../ui'

type W6 = Net['w']
type T3 = Net['theta']

const PRESETS: { id: string; title: string; note: string; net: Net; act: Act }[] = [
  { id: 'lecture', title: 'Lecture 6 example', note: 'w = −0.65, 0.64, 1.11, 0.84, 0.86, −1.38 · θ = 0 · η = 0.5', net: LECTURE_NET, act: 'sigmoid' },
  { id: 'hand', title: 'Hand-picked weights (step)', note: 'Pattern of 2023 Spring Q7(c) / 2025 Spring Q7 — OR & NAND → AND', net: { w: [1, 1, -1, -1, 1, 1], theta: [-0.5, 1.5, -1] }, act: 'step' },
  { id: 'zeros', title: 'All weights zero', note: 'Final 2022 Q3(d): why zero init fails', net: { w: [0, 0, 0, 0, 0, 0], theta: [0, 0, 0] }, act: 'sigmoid' },
]

// input-plane window shown around the unit square
const lo = -0.5
const hi = 1.5
const C1 = '#2563eb'
const C0 = '#e11d48'

/** Background coloured by the network output; optional hidden-unit boundary lines. */
function InputSpace({ net, act }: { net: Net; act: Act }) {
  const S = 300, P = 30
  const X = (v: number) => P + ((v - lo) / (hi - lo)) * (S - 2 * P)
  const Y = (v: number) => S - P - ((v - lo) / (hi - lo)) * (S - 2 * P)
  const canvas = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = canvas.current
    if (!c) return
    const n = 60
    c.width = n
    c.height = n
    const ctx = c.getContext('2d')!
    const img = ctx.createImageData(n, n)
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const o = forward(net, [lo + ((i + 0.5) / n) * (hi - lo), hi - ((j + 0.5) / n) * (hi - lo)], act).o
        const strength = 0.1 + Math.abs(o - 0.5) * 0.5
        const [r, g, b] = o >= 0.5 ? [37, 99, 235] : [225, 29, 72]
        // blend the class colour onto white
        img.data.set([255 - (255 - r) * strength, 255 - (255 - g) * strength, 255 - (255 - b) * strength, 255], (j * n + i) * 4)
      }
    ctx.putImageData(img, 0, 0)
  }, [net, act])
  const boundary = (a: number, b: number, c: number) => {
    // a·x1 + b·x2 + c = 0 inside the square
    const pts: [number, number][] = []
    if (Math.abs(b) > 1e-9) for (const x of [lo, hi]) pts.push([x, -(a * x + c) / b])
    if (Math.abs(a) > 1e-9) for (const y of [lo, hi]) pts.push([-(b * y + c) / a, y])
    const ok = pts.filter(([x, y]) => x >= lo - 1e-9 && x <= hi + 1e-9 && y >= lo - 1e-9 && y <= hi + 1e-9)
    return ok.length >= 2 ? ok : null
  }
  const lines = [boundary(net.w[0], net.w[1], net.theta[0]), boundary(net.w[2], net.w[3], net.theta[1])]
  return (
    <div className="relative mx-auto w-full max-w-[360px]">
    <canvas ref={canvas} aria-hidden className="absolute [image-rendering:auto]" style={{ left: `${(P / S) * 100}%`, top: `${(P / S) * 100}%`, width: `${((S - 2 * P) / S) * 100}%`, height: `${((S - 2 * P) / S) * 100}%` }} />
    <svg viewBox={`0 0 ${S} ${S}`} className="relative w-full" role="img" aria-label="Network output over the input plane">
      <rect x={P} y={P} width={S - 2 * P} height={S - 2 * P} fill="none" stroke="#e2e8f0" />
      {lines.map((l, k) =>
        l ? (
          <g key={k}>
            <line x1={X(l[0][0])} y1={Y(l[0][1])} x2={X(l[1][0])} y2={Y(l[1][1])} stroke="#334155" strokeWidth={1.5} strokeDasharray="5 4" />
            <text x={X(l[1][0]) + (l[1][0] >= hi - 1e-6 ? -4 : 4)} y={Y(l[1][1]) + (l[1][1] <= lo + 1e-6 ? -4 : 12)} textAnchor={l[1][0] >= hi - 1e-6 ? 'end' : 'start'} className="fill-slate-600 text-[11px] font-bold">h{k + 1}</text>
          </g>
        ) : null,
      )}
      {[0, 1].map((v) => (
        <g key={v}>
          <text x={X(v)} y={S - 10} textAnchor="middle" className="fill-slate-400 text-[11px]">{v}</text>
          <text x={P - 8} y={Y(v) + 4} textAnchor="end" className="fill-slate-400 text-[11px]">{v}</text>
        </g>
      ))}
      <text x={S - P} y={S - 10} textAnchor="end" className="fill-slate-500 text-[11px] font-bold">x₁</text>
      <text x={P + 4} y={P - 10} className="fill-slate-500 text-[11px] font-bold">x₂</text>
      {XOR.map((s) => (
        <g key={s.x.join()}>
          <circle cx={X(s.x[0])} cy={Y(s.x[1])} r={9} fill={s.t ? C1 : C0} stroke="#fff" strokeWidth={2.5} />
          <text x={X(s.x[0])} y={Y(s.x[1]) + 4} textAnchor="middle" className="fill-white text-[10px] font-extrabold">{s.t}</text>
        </g>
      ))}
    </svg>
    </div>
  )
}

/** The four samples after the hidden layer, with the output neuron's line. */
function HiddenSpace({ net, act }: { net: Net; act: Act }) {
  const S = 300, P = 30
  const X = (v: number) => P + v * (S - 2 * P)
  const Y = (v: number) => S - P - v * (S - 2 * P)
  const mapped = XOR.map((s) => ({ ...s, h: forward(net, s.x, act).h }))
  const [a, b, c] = [net.w[4], net.w[5], net.theta[2]]
  const pts: [number, number][] = []
  if (Math.abs(b) > 1e-9) for (const x of [0, 1]) pts.push([x, -(a * x + c) / b])
  if (Math.abs(a) > 1e-9) for (const y of [0, 1]) pts.push([-(b * y + c) / a, y])
  const line = pts.filter(([x, y]) => x >= -1e-9 && x <= 1 + 1e-9 && y >= -1e-9 && y <= 1 + 1e-9)
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="mx-auto w-full max-w-[360px]" role="img" aria-label="Samples in hidden-unit space">
      <rect x={P} y={P} width={S - 2 * P} height={S - 2 * P} fill="#f8fafc" stroke="#e2e8f0" />
      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <text x={X(v)} y={S - 10} textAnchor="middle" className="fill-slate-400 text-[11px]">{v}</text>
          <text x={P - 8} y={Y(v) + 4} textAnchor="end" className="fill-slate-400 text-[11px]">{v}</text>
        </g>
      ))}
      <text x={S - P} y={S - 10} textAnchor="end" className="fill-slate-500 text-[11px] font-bold">h₁</text>
      <text x={P + 4} y={P - 10} className="fill-slate-500 text-[11px] font-bold">h₂</text>
      {line.length >= 2 && <line x1={X(line[0][0])} y1={Y(line[0][1])} x2={X(line[1][0])} y2={Y(line[1][1])} stroke="#0f172a" strokeWidth={2.5} />}
      {mapped.map((s, i) => (
        <g key={i}>
          <circle cx={X(s.h[0])} cy={Y(s.h[1])} r={9} fill={s.t ? C1 : C0} stroke="#fff" strokeWidth={2.5} opacity={0.9} />
          <text x={X(s.h[0]) + 12} y={Y(s.h[1]) - 8} className="fill-slate-600 text-[11px] font-bold">({s.x.join(',')})</text>
        </g>
      ))}
    </svg>
  )
}

function Diagram({ net, act, values, deltas }: { net: Net; act: Act; values: { x: [number, number]; h: [number, number]; o: number }; deltas: { k: number; j: [number, number] } | null }) {
  const pos = { x1: [70, 70], x2: [70, 210], h1: [260, 70], h2: [260, 210], o: [450, 140] } as const
  const edges: [keyof typeof pos, keyof typeof pos, number, string][] = [
    ['x1', 'h1', net.w[0], 'w1'],
    ['x2', 'h1', net.w[1], 'w2'],
    ['x1', 'h2', net.w[2], 'w3'],
    ['x2', 'h2', net.w[3], 'w4'],
    ['h1', 'o', net.w[4], 'w5'],
    ['h2', 'o', net.w[5], 'w6'],
  ]
  const node = (id: keyof typeof pos, label: string, val: number, bias?: number, delta?: number) => (
    <g key={id}>
      <circle cx={pos[id][0]} cy={pos[id][1]} r={30} fill="#fff" stroke={id.startsWith('x') ? '#94a3b8' : '#2563eb'} strokeWidth={2.5} />
      <text x={pos[id][0]} y={pos[id][1] - 4} textAnchor="middle" className="fill-slate-500 text-[11px] font-bold">{label}</text>
      <text x={pos[id][0]} y={pos[id][1] + 12} textAnchor="middle" className="fill-slate-900 font-mono text-[13px] font-extrabold">{fmt(val, 3)}</text>
      {bias !== undefined && <text x={pos[id][0]} y={pos[id][1] + 46} textAnchor="middle" className="fill-slate-500 font-mono text-[11px]">θ = {fmt(bias, 3)}</text>}
      {delta !== undefined && <text x={pos[id][0]} y={pos[id][1] - 38} textAnchor="middle" className="fill-rose-600 font-mono text-[11px] font-bold">δ = {fmt(delta, 4)}</text>}
    </g>
  )
  return (
    <svg viewBox="0 0 520 280" className="w-full" role="img" aria-label="2-2-1 network with current values">
      {edges.map(([a, b, wv, name]) => {
        const [x1, y1] = pos[a]
        const [x2, y2] = pos[b]
        const mx = x1 + (x2 - x1) * 0.42
        const my = y1 + (y2 - y1) * 0.42
        return (
          <g key={name}>
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={wv >= 0 ? C1 : C0} strokeWidth={1 + Math.min(6, Math.abs(wv) * 1.2)} opacity={0.55} />
            <rect x={mx - 30} y={my - 10} width={60} height={20} rx={6} fill="#fff" stroke="#e2e8f0" />
            <text x={mx} y={my + 4} textAnchor="middle" className="fill-slate-700 font-mono text-[11px] font-bold">{name} {fmt(wv, 2)}</text>
          </g>
        )
      })}
      {node('x1', 'x₁', values.x[0])}
      {node('x2', 'x₂', values.x[1])}
      {node('h1', act === 'step' ? 'h₁ step' : 'h₁ σ', values.h[0], net.theta[0], deltas?.j[0])}
      {node('h2', act === 'step' ? 'h₂ step' : 'h₂ σ', values.h[1], net.theta[1], deltas?.j[1])}
      {node('o', 'O', values.o, net.theta[2], deltas?.k)}
    </svg>
  )
}

function paramCount(sizes: number[]): { total: number; work: string } {
  let total = 0
  const parts: string[] = []
  for (let i = 1; i < sizes.length; i++) {
    const p = (sizes[i - 1] + 1) * sizes[i]
    total += p
    parts.push(`(${sizes[i - 1]}+1)·${sizes[i]}`)
  }
  return { total, work: `${parts.join(' + ')} = ${int(total)}` }
}

export default function XorMlp() {
  const [preset, setPreset] = useState<string | null>('lecture')
  const [net, setNet] = useState<Net>(LECTURE_NET)
  const [act, setAct] = useState<Act>('sigmoid')
  const [eta, setEta] = useState(0.5)
  const [sample, setSample] = useState(0)
  const [epoch, setEpoch] = useState(0)
  const [history, setHistory] = useState<{ e: number; l: number }[]>([{ e: 0, l: loss(LECTURE_NET) }])
  const [layers, setLayers] = useState('2, 2, 1')

  const s = XOR[sample]
  const step = useMemo(() => (act === 'sigmoid' ? backprop(net, s.x, s.t, eta) : null), [net, s, eta, act])
  const fwd = forward(net, s.x, act)
  const outs = XOR.map((d) => forward(net, d.x, act).o)
  const solved = outs.every((o, i) => (o >= 0.5 ? 1 : 0) === XOR[i].t)
  const symmetric = Math.abs(net.w[0] - net.w[2]) < 1e-9 && Math.abs(net.w[1] - net.w[3]) < 1e-9 && Math.abs(net.theta[0] - net.theta[1]) < 1e-9 && Math.abs(net.w[4] - net.w[5]) < 1e-9

  const load = (n: Net, a: Act) => {
    setNet(n)
    setAct(a)
    setSample(0)
    setEpoch(0)
    setHistory([{ e: 0, l: loss(n, XOR, a) }])
  }
  const applyOne = () => {
    if (!step) return
    setNet(step.after)
    if (sample === 3) {
      setEpoch(epoch + 1)
      setHistory((h) => [...h, { e: epoch + 1, l: loss(step.after) }])
    }
    setSample((sample + 1) % 4)
  }
  const train = (epochs: number) => {
    let cur = net
    const pts: { e: number; l: number }[] = []
    // finish the current epoch first so whole epochs stay aligned with the sample counter
    for (let i = sample; i < 4; i++) cur = backprop(cur, XOR[i].x, XOR[i].t, eta).after
    const every = Math.max(1, Math.floor(epochs / 60))
    for (let e = 1; e < epochs; e++) {
      for (const d of XOR) cur = backprop(cur, d.x, d.t, eta).after
      if (e % every === 0) pts.push({ e: epoch + 1 + e, l: loss(cur) })
    }
    setNet(cur)
    setEpoch(epoch + epochs)
    setSample(0)
    setHistory((h) => [...h, ...pts, { e: epoch + epochs, l: loss(cur) }])
  }
  const setW = (i: number, v: number) => {
    const w = [...net.w] as W6
    w[i] = v
    setNet({ ...net, w })
    setPreset(null)
  }
  const setT = (i: number, v: number) => {
    const theta = [...net.theta] as T3
    theta[i] = v
    setNet({ ...net, theta })
    setPreset(null)
  }
  const sizes = layers.split(/[\s,]+/).map(Number).filter((v) => Number.isInteger(v) && v > 0)
  const pc = sizes.length >= 2 ? paramCount(sizes) : null
  const maxL = Math.max(...history.map((h) => h.l), 0.01)
  const maxE = Math.max(1, history[history.length - 1].e)

  return (
    <LabPage id="xor-mlp" quiz lead="A single neuron draws one straight line, and XOR needs two. A hidden layer draws two lines, squashes the plane, and in the new (h₁, h₂) space one line is enough. Backpropagation finds those weights δ by δ.">
      <Workspace
        wide
        controls={
          <>
            <Card title="Start from">
              <Presets
                items={PRESETS}
                value={preset}
                onPick={(id) => {
                  const p = PRESETS.find((q) => q.id === id)!
                  setPreset(id)
                  load(p.net, p.act)
                  if (id === 'lecture') setEta(0.5)
                }}
              />
              <div className="mt-2">
                <Btn
                  onClick={() => {
                    const r = mulberry32(Date.now() % 100000)
                    const v = () => Number((r() * 2 - 1).toFixed(2))
                    setPreset(null)
                    load({ w: [v(), v(), v(), v(), v(), v()], theta: [0, 0, 0] }, 'sigmoid')
                  }}
                >
                  Random weights
                </Btn>
              </div>
            </Card>
            <Card title="Network">
              <Seg label="Activation" value={act} onChange={(a) => { setAct(a); setHistory([{ e: epoch, l: loss(net, XOR, a) }]) }} options={[{ v: 'sigmoid', label: 'sigmoid' }, { v: 'step', label: 'binary step' }]} />
              <div className="mt-3 grid grid-cols-3 gap-2">
                {net.w.map((v, i) => <NumberField key={i} label={`w${i + 1}`} value={Number(v.toFixed(4))} step={0.1} onChange={(x) => setW(i, x)} />)}
                {net.theta.map((v, i) => <NumberField key={'t' + i} label={`θ${i + 1}`} value={Number(v.toFixed(4))} step={0.1} onChange={(x) => setT(i, x)} />)}
              </div>
            </Card>
            <Card title="Train">
              <Slider label="Learning rate η" value={eta} min={0.05} max={3} step={0.05} onChange={setEta} />
              <div className="mt-3 flex flex-wrap gap-2">
                <Btn primary onClick={applyOne} disabled={act === 'step'}>Apply update for ({s.x.join(', ')})</Btn>
                <Btn onClick={() => train(1)} disabled={act === 'step'}>+1 epoch</Btn>
                <Btn onClick={() => train(100)} disabled={act === 'step'}>+100</Btn>
                <Btn onClick={() => train(1000)} disabled={act === 'step'}>+1000</Btn>
              </div>
              <p className="mt-2 text-xs text-slate-500">
                Epoch {int(epoch)} · next sample {sample + 1} of 4 · loss {fmt(loss(net, XOR, act), 5)}
              </p>
              {act === 'step' && (
                <div className="mt-2">
                  <Note tone="warn" title="No training with a step function.">Its gradient is 0 almost everywhere, so backprop has nothing to follow (Final 2024 Q4c). The fix: a steep sigmoid σ(cx) approximates the step and still has a gradient.</Note>
                </div>
              )}
            </Card>
          </>
        }
      >
        <div className="grid gap-5 2xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
          <Card step={1} title={`Network on input (${s.x.join(', ')}), target ${s.t}`} sub="edge width = |weight| · blue +, red −">
            <Diagram net={net} act={act} values={{ x: s.x, h: fwd.h, o: fwd.o }} deltas={step ? { k: step.deltaK, j: step.deltaJ } : null} />
            <div className="mt-2 flex flex-wrap gap-1.5">
              {XOR.map((d, i) => (
                <button key={i} type="button" onClick={() => setSample(i)} aria-pressed={i === sample} className={`rounded-lg border px-2.5 py-1 font-mono text-xs font-bold ${i === sample ? 'border-blue-600 bg-blue-600 text-white' : 'border-slate-200 bg-white text-slate-600'}`}>
                  ({d.x.join(',')}) → {fmt(outs[i], 3)}
                </button>
              ))}
            </div>
          </Card>
          <div className="grid gap-5 sm:grid-cols-2">
            <Card step={2} title="Input space" sub="dashed = hidden units">
              <InputSpace net={net} act={act} />
            </Card>
            <Card step={3} title="Hidden space" sub="one line splits it">
              <HiddenSpace net={net} act={act} />
            </Card>
          </div>
        </div>

        <Note tone={solved ? 'good' : 'info'} title={solved ? 'XOR solved.' : 'Not solved yet.'}>
          {solved
            ? 'All four points fall on the right side. Look at the hidden-space plot: the hidden layer moved (0,1) and (1,0) to one side of a single straight line — that is exactly what the 2023 Spring hint (“make the points linearly separable after the first layer”) asks you to construct.'
            : 'Some points are still misclassified. Train more, or adjust weights by hand until the hidden-space plot becomes separable by one line.'}
          {symmetric && act === 'sigmoid' && ' Both hidden units have identical weights, so they always compute the same value and receive the same δ — they can never become different. This is why all-zero initialisation fails.'}
        </Note>

        {step && (
          <Card step={4} title={`One backprop step on (${s.x.join(', ')}), T = ${s.t}`} sub="exam formulas · Turn on “Quiz me”">
            <div className="grid gap-4 xl:grid-cols-2">
              <div className="formula">
                <span className="text-slate-400">Forward</span>
                {`\nh₁ = σ(${fmt(net.w[0], 4)}·${s.x[0]} + ${fmt(net.w[1], 4)}·${s.x[1]} + ${fmt(net.theta[0], 4)}) = ${fmt(step.fwd.h[0], 6)}`}
                {`\nh₂ = σ(${fmt(net.w[2], 4)}·${s.x[0]} + ${fmt(net.w[3], 4)}·${s.x[1]} + ${fmt(net.theta[1], 4)}) = ${fmt(step.fwd.h[1], 6)}`}
                {`\nO  = σ(${fmt(net.w[4], 4)}·h₁ + ${fmt(net.w[5], 4)}·h₂ + ${fmt(net.theta[2], 4)}) = `}<b className="text-blue-700">{fmt(step.fwd.o, 6)}</b>
                {'\n\n'}<span className="text-slate-400">Backward</span>
                {`\nδk  = (O − T)·O(1 − O) = `}<b className="text-rose-600">{fmt(step.deltaK, 6)}</b>
                {`\nδj1 = h₁(1 − h₁)·δk·w5 = ${fmt(step.deltaJ[0], 6)}`}
                {`\nδj2 = h₂(1 − h₂)·δk·w6 = ${fmt(step.deltaJ[1], 6)}`}
                {`\nw ← w − η·δ·(input of that weight),  η = ${fmt(eta, 3)}`}
              </div>
              <TableWrap>
                <tr><th className="left">Quantity</th><th>Before</th><th>Gradient term</th><th>After</th></tr>
                <tr><td className="left">O</td><Ans k="o" v={fmt(step.fwd.o, 4)} /><td>—</td><td>—</td></tr>
                <tr><td className="left">δk</td><Ans k="dk" v={fmt(step.deltaK, 4)} /><td>—</td><td>—</td></tr>
                {(['w5', 'w6'] as const).map((name, i) => (
                  <tr key={name}><td className="left">{name}</td><td>{fmt(net.w[4 + i], 4)}</td><td>δk·h{i + 1} = {fmt(step.deltaK * step.fwd.h[i], 4)}</td><Ans k={name} v={fmt(step.after.w[4 + i], 4)} /></tr>
                ))}
                <tr><td className="left">θ3</td><td>{fmt(net.theta[2], 4)}</td><td>δk = {fmt(step.deltaK, 4)}</td><Ans k="t3" v={fmt(step.after.theta[2], 4)} /></tr>
                {[0, 1, 2, 3].map((i) => {
                  const j = i < 2 ? 0 : 1
                  const xi = s.x[i % 2]
                  return (
                    <tr key={i}><td className="left">w{i + 1}</td><td>{fmt(net.w[i], 4)}</td><td>δj{j + 1}·x{(i % 2) + 1} = {fmt(step.deltaJ[j] * xi, 4)}</td><Ans k={`w${i + 1}`} v={fmt(step.after.w[i], 4)} /></tr>
                  )
                })}
                {[0, 1].map((j) => (
                  <tr key={'t' + j}><td className="left">θ{j + 1}</td><td>{fmt(net.theta[j], 4)}</td><td>δj{j + 1} = {fmt(step.deltaJ[j], 4)}</td><Ans k={`t${j + 1}`} v={fmt(step.after.theta[j], 4)} /></tr>
                ))}
              </TableWrap>
            </div>
            <p className="mt-2 text-xs text-slate-400">Hidden δ uses w5 and w6 <i>before</i> this step’s update, as on the lecture slides. With the lecture start, round 1 gives δk = 0.107022 and new w5 = 0.833245.</p>
          </Card>
        )}

        <div className="grid gap-5 xl:grid-cols-2">
          <Card title="Loss while training" sub="½(T − O)² averaged over the 4 samples">
            <svg viewBox="0 0 360 170" className="w-full" role="img" aria-label="Loss curve">
              <line x1={36} x2={350} y1={140} y2={140} stroke="#cbd5e1" />
              <line x1={36} x2={36} y1={14} y2={140} stroke="#cbd5e1" />
              <text x={30} y={18} textAnchor="end" className="fill-slate-400 text-[10px]">{fmt(maxL, 3)}</text>
              <text x={30} y={143} textAnchor="end" className="fill-slate-400 text-[10px]">0</text>
              <text x={350} y={156} textAnchor="end" className="fill-slate-400 text-[10px]">epoch {int(maxE)}</text>
              <polyline fill="none" stroke="#2563eb" strokeWidth={2} points={history.map((h) => `${36 + (h.e / maxE) * 314},${140 - (h.l / maxL) * 126}`).join(' ')} />
            </svg>
            {epoch > 0 && act === 'sigmoid' && !solved && eta > 2 && <Note tone="warn">A very large η can overshoot and oscillate instead of settling.</Note>}
          </Card>
          <Card title="Count the parameters" sub="2022 Spring Q7(c), Final 2024 Q4(a)">
            <label className="grid gap-1">
              <span className="text-xs font-bold text-slate-500">Neurons per layer (input, hidden…, output)</span>
              <input value={layers} onChange={(e) => setLayers(e.target.value)} className="rounded-[10px] border border-slate-300 px-2.5 py-1.5 font-mono text-sm outline-none focus:ring-2 focus:ring-blue-300" />
            </label>
            {pc && (
              <div className="formula mt-3">
                {`weights + biases = ${pc.work}`}
                {'\n'}<span className="text-slate-400">every non-input neuron has one weight per input plus one bias</span>
              </div>
            )}
            <p className="mt-2 text-xs text-slate-400">Try 3, 4, 2 (answer 26) or 3, 2, 1 (8 weights + 3 biases = 11).</p>
          </Card>
        </div>

        <div className="grid gap-5 md:grid-cols-2">
          <Note tone="warn" title="Exam traps.">
            The course writes δk = (O − T)·O(1 − O) and <b>subtracts</b>: w ← w − ηδO. Some books write (T − O) and add — same result, but use the paper’s convention. With a <i>linear</i> activation, stacking layers adds nothing: the whole network collapses to one neuron (2022 Spring Q7b).
          </Note>
          <Note title="Output layer choice.">
            Binary: 1 sigmoid unit. Multi-class: softmax over n units. Multi-label: n sigmoid units (Final 2022). Never ReLU at the output of a classifier: its range is unbounded, so there is no natural cut-off.
          </Note>
        </div>
      </Workspace>
    </LabPage>
  )
}
