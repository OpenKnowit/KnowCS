import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { TENSORS, bridge, graph, lectureTensor, LOOP, LOOP_CODE, loopError, matmul, mul, replay, sizeStr, transpose } from '../../lib/torchSim'
import type { BridgeOp, GradEvent, LoopLine } from '../../lib/torchSim'
import { moduleUrl } from '../../lib/sitemap'
import { fmt } from '../format'
import { Ans, Btn, Card, LabPage, Note, Seg, Slider, TableWrap } from '../ui'

const Mat = ({ m, label, hot }: { m: number[][]; label: string; hot?: (i: number, j: number) => boolean }) => (
  <div className="grid gap-1">
    <span className="font-mono text-[12px] font-bold text-slate-600">{label}</span>
    <div className="inline-grid w-fit gap-1 font-mono text-[13px]" style={{ gridTemplateColumns: `repeat(${m[0].length}, 34px)` }}>
      {m.flatMap((row, i) => row.map((v, j) => (
        <span key={`${i}-${j}`} className={`grid h-8 place-items-center rounded-md border font-bold ${hot?.(i, j) ? 'border-amber-300 bg-amber-50 text-amber-900' : v ? 'border-blue-200 bg-blue-50 text-blue-900' : 'border-slate-200 bg-white text-slate-500'}`}>{v}</span>
      )))}
    </div>
  </div>
)

const Row = ({ v, label, tone }: { v: number[]; label: string; tone: string }) => (
  <div className="flex flex-wrap items-center gap-2">
    <span className="w-24 font-mono text-[12px] font-bold text-slate-600">{label}</span>
    {v.map((x, i) => <span key={i} className={`grid h-8 w-9 place-items-center rounded-md border font-mono text-[13px] font-bold ${tone}`}>{x}</span>)}
  </div>
)

/** z = w·x + b → σ → BCE, with data above each node and .grad below once backward has run. */
function GraphView({ w, x, b, y, shown }: { w: number; x: number; b: number; y: number; shown: number }) {
  const { t } = useTranslation()
  const g = graph(w, x, b, y)
  // step: how many backward steps until this node's .grad is filled (x has requires_grad=False, so it gets none)
  const nodes: { id: string; x: number; y: number; label: string; data: number; grad?: number; step?: number; leaf?: boolean }[] = [
    { id: 'w', x: 60, y: 50, label: 'w', data: w, grad: g.dw, step: 3, leaf: true },
    { id: 'x', x: 60, y: 130, label: 'x', data: x },
    { id: 'b', x: 60, y: 210, label: 'b', data: b, grad: g.db, step: 3, leaf: true },
    { id: 'z', x: 250, y: 130, label: 'z = w·x + b', data: g.z, grad: g.dz, step: 2 },
    { id: 'loss', x: 470, y: 130, label: 'loss', data: g.loss, grad: 1, step: 1 },
  ]
  const edges: [string, string][] = [['w', 'z'], ['x', 'z'], ['b', 'z'], ['z', 'loss']]
  const at = (id: string) => nodes.find((n) => n.id === id)!
  return (
    <svg viewBox="0 0 560 260" className="w-full max-w-[760px] rounded-xl bg-slate-900" role="img" aria-label={t('lab.pytorch.graph_aria')}>
      {edges.map(([a, c]) => {
        const p = at(a), q = at(c)
        const back = p.step !== undefined && shown >= p.step
        return <line key={a + c} x1={p.x + 44} y1={p.y} x2={q.x - 60} y2={q.y} stroke={back ? '#fb7185' : '#475569'} strokeWidth={back ? 3 : 2} />
      })}
      {nodes.map((n) => (
        <g key={n.id}>
          <rect x={n.x - (n.id === 'z' ? 60 : 44)} y={n.y - 22} width={n.id === 'z' ? 120 : 88} height={44} rx={10} fill={n.leaf ? '#1e3a8a' : '#1e293b'} stroke={n.leaf ? '#60a5fa' : '#334155'} strokeWidth={2} />
          <text x={n.x} y={n.y - 4} textAnchor="middle" fontSize={12} fontWeight={800} fill="#e2e8f0" fontFamily="ui-monospace, monospace">{n.label}</text>
          <text x={n.x} y={n.y + 12} textAnchor="middle" fontSize={12} fill="#7dd3fc" fontFamily="ui-monospace, monospace">{fmt(n.data, 3)}</text>
          {n.grad !== undefined && n.step !== undefined && shown >= n.step && (
            <text x={n.x} y={n.y + 40} textAnchor="middle" fontSize={12} fontWeight={700} fill="#fda4af" fontFamily="ui-monospace, monospace">grad {fmt(n.grad, 3)}</text>
          )}
        </g>
      ))}
      <text x={470} y={70} textAnchor="middle" fontSize={11} fill="#94a3b8" fontFamily="ui-monospace, monospace">y = {y}</text>
    </svg>
  )
}

const BRIDGE_OPS: { op: BridgeOp; code: string }[] = [
  { op: 'add_', code: 'my_tensor.add_(1)' },
  { op: 'np_add_out', code: 'np.add(np_array, 1, out=np_array)' },
  { op: 'plus', code: 'my_tensor = my_tensor + 1' },
  { op: 'clone', code: 'my_tensor = my_tensor.clone()' },
]

/** the five loop lines, shuffled */
const POOL: LoopLine[] = ['loss', 'step', 'zero_grad', 'backward', 'forward']

export default function PyTorchPage() {
  const { t } = useTranslation()
  const [ti, setTi] = useState(0)
  const [ops, setOps] = useState<BridgeOp[]>(['add_'])
  const [w, setW] = useState(0.5)
  const [x, setX] = useState(2)
  const [b, setB] = useState(-1)
  const [y, setY] = useState(1)
  const [shown, setShown] = useState(0)
  const [events, setEvents] = useState<GradEvent[]>([])
  const [noGrad, setNoGrad] = useState(false)
  const [order, setOrder] = useState<LoopLine[]>([])

  const tensor = TENSORS[ti]
  const lt = lectureTensor()
  const br = bridge(ops)
  const g = graph(w, x, b, y)
  const st = replay({ w, b }, x, y, 0.1, events)
  const sinceZero = events.slice(events.lastIndexOf('zero_grad') + 1).filter((e) => e === 'backward').length
  const err = order.length === LOOP.length ? loopError(order) : null
  const resetGraph = (f: () => void) => {
    f()
    setShown(0)
    setEvents([])
  }

  return (
    <LabPage id="pytorch" quiz lead={t('lab.pytorch.lead')}>
      <div className="grid gap-5">
        <Card step={1} title={t('lab.pytorch.attr_title')} sub={t('lab.common.quiz_sub')}>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
            <div className="grid content-start gap-1.5">
              {TENSORS.map((x, i) => (
                <button key={x.code} type="button" aria-pressed={i === ti} onClick={() => setTi(i)} className={`rounded-lg border px-3 py-1.5 text-left font-mono text-[12.5px] ${i === ti ? 'border-blue-500 bg-blue-50 text-blue-900' : 'border-slate-200 bg-white text-slate-700 hover:border-slate-300'}`}>
                  {x.code}
                </button>
              ))}
            </div>
            <div className="grid content-start gap-3">
              <TableWrap>
                <tr><th className="left">{t('lab.pytorch.attr')}</th><th className="left">print(…)</th></tr>
                <tr><td className="left">tensor.shape</td><Ans k={`shape${ti}`} v={sizeStr(tensor.shape)} className="left" /></tr>
                <tr><td className="left">tensor.dtype</td><Ans k={`dtype${ti}`} v={tensor.dtype} className="left" /></tr>
                <tr><td className="left">tensor.device</td><td className="left">cpu</td></tr>
              </TableWrap>
              <Note>{t('lab.pytorch.attr_note')}</Note>
            </div>
          </div>
        </Card>

        <Card step={2} title={t('lab.pytorch.mm_title')} sub={t('lab.common.quiz_sub')}>
          <pre className="mb-3 overflow-x-auto rounded-xl bg-slate-900 p-3 text-[12px] text-slate-100">{'tensor = torch.ones(4, 4)\ntensor[:, 1] = 0\ny1 = tensor @ tensor.T     # matrix product\nz1 = tensor * tensor       # element-wise'}</pre>
          <div className="flex flex-wrap gap-6">
            <Mat m={lt} label="tensor" hot={(_, j) => j === 1} />
            <Mat m={matmul(lt, transpose(lt))} label="tensor @ tensor.T" />
            <Mat m={mul(lt, lt)} label="tensor * tensor" hot={(_, j) => j === 1} />
          </div>
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <Note>{t('lab.pytorch.mm_note')}</Note>
            <TableWrap>
              <tr><th className="left">{t('lab.common.quantity')}</th><th>{t('lab.common.value')}</th></tr>
              <tr><td className="left">y1[0, 0]</td><Ans k="y00" v={3} /></tr>
              <tr><td className="left">z1[0, 1]</td><Ans k="z01" v={0} /></tr>
              <tr><td className="left">tensor.sum().item()</td><Ans k="sum" v={12} /></tr>
            </TableWrap>
          </div>
        </Card>

        <Card step={3} title={t('lab.pytorch.bridge_title')}>
          <pre className="mb-3 overflow-x-auto rounded-xl bg-slate-900 p-3 text-[12px] text-slate-100">{['my_tensor = torch.ones(5)', 'np_array = my_tensor.numpy()   # shares memory', ...ops.map((o) => BRIDGE_OPS.find((x) => x.op === o)!.code)].join('\n')}</pre>
          <div className="mb-3 flex flex-wrap gap-2">
            {BRIDGE_OPS.map((o) => <Btn key={o.op} onClick={() => setOps([...ops, o.op])}><span className="font-mono text-[12px]">{o.code}</span></Btn>)}
            <Btn onClick={() => setOps([])}>{t('lab.common.reset')}</Btn>
          </div>
          <div className="grid gap-2">
            <Row v={br.tensor} label="my_tensor" tone="border-orange-200 bg-orange-50 text-orange-900" />
            <Row v={br.array} label="np_array" tone="border-sky-200 bg-sky-50 text-sky-900" />
          </div>
          <div className="mt-3"><Note tone={br.shared ? 'good' : 'warn'}>{t(br.shared ? 'lab.pytorch.shared' : 'lab.pytorch.not_shared')}</Note></div>
        </Card>

        <Card step={4} title={t('lab.pytorch.ag_title')} sub={t('lab.common.quiz_sub')}>
          <div className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
            <div className="grid min-w-0 grid-cols-[minmax(0,1fr)] content-start gap-4">
              <div className="grid gap-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
                <Slider label="w" value={w} min={-2} max={2} step={0.25} onChange={(v) => resetGraph(() => setW(v))} />
                <Slider label="x" value={x} min={-3} max={3} step={0.5} onChange={(v) => resetGraph(() => setX(v))} />
                <Slider label="b" value={b} min={-2} max={2} step={0.25} onChange={(v) => resetGraph(() => setB(v))} />
                <Seg label={t('lab.pytorch.target')} value={y} onChange={(v) => resetGraph(() => setY(v))} options={[{ v: 0, label: 'y = 0' }, { v: 1, label: 'y = 1' }]} />
              </div>
              <div>
                <pre className="overflow-x-auto rounded-xl bg-slate-900 p-3 text-[12px] leading-relaxed text-slate-100" aria-label={t('lab.pytorch.code_title')}>
                  {[`w = torch.tensor(${w}, requires_grad=True)`, `b = torch.tensor(${b}, requires_grad=True)`, `x, y = torch.tensor(${x}), torch.tensor(${y}.)`, 'z = w * x + b', 'loss = F.binary_cross_entropy_with_logits(z, y)', 'loss.backward()', 'print(w.grad, b.grad)'].join('\n')}
                </pre>
                <p className="mt-2 text-xs text-slate-500">{t('lab.pytorch.code_note')}</p>
              </div>
            </div>
            <div className="min-w-0">
              <GraphView w={w} x={x} b={b} y={y} shown={noGrad ? 0 : shown} />
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <Btn onClick={() => setShown(Math.max(0, shown - 1))} disabled={shown === 0}>{t('lab.common.back')}</Btn>
                <Btn primary onClick={() => setShown(Math.min(3, shown + 1))} disabled={shown >= 3 || noGrad}>{t('lab.pytorch.back_step')}</Btn>
                <span className="text-[13px] text-slate-600">{t(`lab.pytorch.back_${shown}`)}</span>
              </div>
            </div>
          </div>
          <div className="mt-3 grid gap-3 lg:grid-cols-2">
            <TableWrap>
              <tr><th className="left">{t('lab.common.quantity')}</th><th className="left">{t('lab.pytorch.formula')}</th><th>{t('lab.common.value')}</th></tr>
              <tr><td className="left">z</td><td className="left">w·x + b</td><Ans k="z" v={fmt(g.z, 3)} /></tr>
              <tr><td className="left">σ(z)</td><td className="left">1 / (1 + e^−z)</td><Ans k="p" v={fmt(g.p, 3)} /></tr>
              <tr><td className="left">∂loss/∂z</td><td className="left">σ(z) − y</td><Ans k="dz" v={fmt(g.dz, 3)} /></tr>
              <tr><td className="left">w.grad</td><td className="left">(σ(z) − y)·x</td><Ans k="dw" v={fmt(g.dw, 3)} /></tr>
              <tr><td className="left">b.grad</td><td className="left">σ(z) − y</td><Ans k="db" v={fmt(g.db, 3)} /></tr>
            </TableWrap>
            <div className="grid content-start gap-3">
              <div className="flex flex-wrap gap-2">
                <Btn onClick={() => setEvents([...events, 'backward'])} disabled={noGrad}><span className="font-mono text-[12px]">loss.backward()</span></Btn>
                <Btn onClick={() => setEvents([...events, 'step'])} disabled={noGrad}><span className="font-mono text-[12px]">optimizer.step()</span></Btn>
                <Btn onClick={() => setEvents([...events, 'zero_grad'])}><span className="font-mono text-[12px]">optimizer.zero_grad()</span></Btn>
                <Btn onClick={() => setEvents([])}>{t('lab.common.reset')}</Btn>
              </div>
              <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 font-mono text-[13px] leading-relaxed text-slate-800">
                w = {fmt(st.w, 4)}, b = {fmt(st.b, 4)}
                <br />
                w.grad = {fmt(st.wGrad, 4)}, b.grad = {fmt(st.bGrad, 4)}
              </div>
              <Note tone={sinceZero > 1 ? 'warn' : 'info'}>{t(sinceZero > 1 ? 'lab.pytorch.accumulated' : 'lab.pytorch.accumulate', { n: sinceZero })}</Note>
              <label className="flex items-center gap-2 text-[13px] font-semibold text-slate-700">
                <input type="checkbox" checked={noGrad} onChange={(e) => setNoGrad(e.target.checked)} />
                <span className="font-mono">with torch.no_grad():</span>
              </label>
              {noGrad && <Note tone="warn">{t('lab.pytorch.no_grad')}</Note>}
            </div>
          </div>
        </Card>

        <Card step={5} title={t('lab.pytorch.loop_title')} sub={t('lab.pytorch.loop_sub')}>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="grid content-start gap-1.5">
              {POOL.filter((l) => !order.includes(l)).map((l) => (
                <button key={l} type="button" onClick={() => setOrder([...order, l])} className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-left font-mono text-[12.5px] text-slate-800 hover:border-blue-400">
                  {LOOP_CODE[l]}
                </button>
              ))}
              {order.length > 0 && <div className="mt-1"><Btn onClick={() => setOrder([])}>{t('lab.common.reset')}</Btn></div>}
            </div>
            <div className="grid content-start gap-2">
              <pre className="min-h-[150px] overflow-x-auto rounded-xl bg-slate-900 p-3 text-[12px] leading-relaxed text-slate-100">
                {['for inputs, labels in trainloader:', ...order.map((l) => `    ${LOOP_CODE[l]}`)].join('\n')}
              </pre>
              {order.length === LOOP.length && (
                <Note tone={err ? 'bad' : 'good'}>{t(err ? `lab.pytorch.loop_err.${err}` : 'lab.pytorch.loop_ok')}</Note>
              )}
            </div>
          </div>
        </Card>

        <div className="grid gap-5 md:grid-cols-2">
          <Note tone="warn" title={t('lab.common.exam_traps')}>{t('lab.pytorch.traps')}</Note>
          <Note title={t('lab.pytorch.more_title')}>
            {t('lab.pytorch.more')}{' '}
            <a className="font-bold text-blue-700 hover:underline" href={moduleUrl('cnn-shapes')}>{t('site.modules.cnn-shapes.title')} →</a>
          </Note>
        </div>
      </div>
    </LabPage>
  )
}
