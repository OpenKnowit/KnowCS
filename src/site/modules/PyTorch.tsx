import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { TENSORS, bridge, graph, LOOP, LOOP_CODE, loopError, mnistForward, replay, sizeStr } from '../../lib/torchSim'
import type { BridgeOp, GradEvent, LoopLine } from '../../lib/torchSim'
import { KERAS, STEPS, TORCH } from '../../lib/kerasTorch'
import type { CodeLine, Step } from '../../lib/kerasTorch'
import { moduleUrl } from '../../lib/sitemap'
import { fmt } from '../format'
import { Ans, Btn, Card, LabPage, Note, Seg, Slider, TableWrap } from '../ui'
import { PyBlock } from '../../components/PyCode'
import { RunInPlayground } from '../../components/RunInPlayground'
import { highlightPython } from '../playground/pyHighlight'

const Row = ({ v, label, tone }: { v: number[]; label: string; tone: string }) => (
  <div className="flex flex-wrap items-center gap-2">
    <span className="w-24 font-mono text-[12px] font-bold text-slate-600">{label}</span>
    {v.map((x, i) => <span key={i} className={`grid h-8 w-9 place-items-center rounded-md border font-mono text-[13px] font-bold ${tone}`}>{x}</span>)}
  </div>
)

/** One side of the Keras ↔ PyTorch map: every line is a button that picks its step; the step's lines light up. */
function CodePane({ head, lines, step, onPick }: { head: string; lines: CodeLine[]; step: Step; onPick: (s: Step) => void }) {
  const pane = useRef<HTMLDivElement>(null)
  // bring the lit lines into view inside the pane (the page itself does not scroll)
  useEffect(() => {
    const box = pane.current
    const first = box?.querySelector<HTMLElement>('[data-on="true"]')
    if (box && first) box.scrollTo({ top: Math.max(0, first.offsetTop - 48), behavior: 'smooth' })
  }, [step])
  return (
    <div className="min-w-0">
      <div className="mb-1.5 text-[12px] font-extrabold text-slate-600">{head}</div>
      <div ref={pane} className="relative max-h-[22rem] overflow-auto rounded-xl bg-slate-900 p-2 font-mono text-[12px] leading-[1.6] text-slate-100 lg:max-h-[30rem]">
        <div className="w-max min-w-full">
          {lines.map(([code, ...steps], i) => {
            if (!code) return <div key={i} className="h-[1.6em]" />
            const on = steps.includes(step)
            // a line in two steps: clicking it again moves to its other step
            const next = on && steps.length > 1 ? steps[(steps.indexOf(step) + 1) % steps.length] : steps[0]
            return (
              <button key={i} type="button" data-on={on} onClick={() => onPick(next)} className={`block w-full whitespace-pre rounded px-1.5 text-left ${on ? 'bg-amber-300/20 ring-1 ring-amber-300/70' : 'opacity-60 hover:bg-white/5 hover:opacity-100'}`}>
                {highlightPython(code, 'dark')}
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}

const shapeStr = (s: number[]) => `(${s.join(', ')})`

/** the five loop lines, shuffled */
const POOL: LoopLine[] = ['loss', 'step', 'zero_grad', 'backward', 'forward']
const LOOP_HEAD = ['model.train()', 'for epoch in range(NUM_EPOCHS):', '    for i, (inputs, labels) in enumerate(trainloader):', '        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)']

const BRIDGE_OPS: { op: BridgeOp; code: string }[] = [
  { op: 'add_', code: 'my_tensor.add_(1)' },
  { op: 'np_add_out', code: 'np.add(np_array, 1, out=np_array)' },
  { op: 'plus', code: 'my_tensor = my_tensor + 1' },
  { op: 'clone', code: 'my_tensor = my_tensor.clone()' },
]

/** a Python float literal: 2 → 2.0, -0.25 → -0.25 */
const float = (v: number) => (Number.isInteger(v) ? v.toFixed(1) : String(v))
/** the loop's setup for the playground: a tiny model and made-up data, so the student's loop body runs */
const LOOP_SETUP = `import torch
import torch.nn as nn

# a tiny model and made-up data, so the loop can run
torch.manual_seed(0)
X = torch.randn(32, 4)
Y = (X.sum(dim=1) > 0).long()
trainloader = [(X[i:i + 8], Y[i:i + 8]) for i in range(0, 32, 8)]
model = nn.Linear(4, 2)
lossfunc = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
`

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

export default function PyTorchPage() {
  const { t } = useTranslation()
  const [ti, setTi] = useState(0)
  const [ops, setOps] = useState<BridgeOp[]>(['add_'])
  const [step, setStep] = useState<Step>('model')
  const [padding, setPadding] = useState<0 | 1>(1)
  const [inFeatures, setInFeatures] = useState(12544)
  const [w, setW] = useState(0.5)
  const [x, setX] = useState(2)
  const [b, setB] = useState(-1)
  const [y, setY] = useState(1)
  const [shown, setShown] = useState(0)
  const [events, setEvents] = useState<GradEvent[]>([])
  const [noGrad, setNoGrad] = useState(false)
  const [order, setOrder] = useState<LoopLine[]>([])
  const bridgeCode = ['my_tensor = torch.ones(5)', 'np_array = my_tensor.numpy()   # shares memory', ...ops.map((o) => BRIDGE_OPS.find((x) => x.op === o)!.code)].join('\n')
  // floats, as PyTorch needs for requires_grad (torch.tensor(-1) would be an integer tensor)
  const gradCode = [`w = torch.tensor(${float(w)}, requires_grad=True)`, `b = torch.tensor(${float(b)}, requires_grad=True)`, `x, y = torch.tensor(${float(x)}), torch.tensor(${float(y)})`, 'z = w * x + b', 'loss = F.binary_cross_entropy_with_logits(z, y)', 'loss.backward()', 'print(w.grad, b.grad)'].join('\n')
  const loopBody = ['for inputs, labels in trainloader:', ...order.map((l) => `    ${LOOP_CODE[l]}`)].join('\n')
  const pad = padding === 1 ? 'padding=1' : "padding='valid'"
  const initCode = [`self.conv1 = nn.Conv2d(1, 32, 3, ${pad})`, `self.conv2 = nn.Conv2d(32, 64, 3, ${pad})`, 'self.pool = nn.MaxPool2d(2, 2)', `self.fc1 = nn.Linear(${inFeatures}, 128)`, 'self.fc2 = nn.Linear(128, 10)'].join('\n')

  const tensor = TENSORS[ti]
  const br = bridge(ops)
  const fw = mnistForward(padding, inFeatures)
  const fwError = fw.find((r) => r.error)
  const flat = fw[4].shape[1]
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
    <LabPage id="pytorch" quiz>
      <div className="grid gap-5">
        <Card step={1} title={t('lab.pytorch.tensor_title')} sub={t('lab.common.quiz_sub')}>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
            <div className="grid content-start gap-1.5">
              {TENSORS.map((x, i) => (
                <button key={x.code} type="button" aria-pressed={i === ti} onClick={() => setTi(i)} className={`rounded-lg border px-3 py-1.5 text-left font-mono text-[12.5px] ${i === ti ? 'border-blue-500 bg-blue-50 text-blue-900' : 'border-slate-200 bg-white text-slate-700 hover:border-slate-300'}`}>
                  {highlightPython(x.code, 'light')}
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
              <Note>{t('lab.pytorch.np_note')}</Note>
            </div>
          </div>
          <h3 className="mb-2 mt-5 text-[13.5px] font-extrabold text-slate-800">{t('lab.pytorch.bridge_head')}</h3>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
            <div className="min-w-0">
              <PyBlock className="overflow-x-auto rounded-xl bg-slate-900 p-3 text-[12px] text-slate-100" code={bridgeCode} />
              <div className="mt-1.5 flex justify-end"><RunInPlayground lib="pytorch" code={`import torch\nimport numpy as np\n${bridgeCode}\nprint(my_tensor)\nprint(np_array)`} /></div>
            </div>
            <div className="grid content-start gap-3">
              <div className="flex flex-wrap gap-2">
                {BRIDGE_OPS.map((o) => <Btn key={o.op} onClick={() => setOps([...ops, o.op])}><span className="font-mono text-[12px]">{highlightPython(o.code, 'light')}</span></Btn>)}
                <Btn onClick={() => setOps([])}>{t('lab.common.reset')}</Btn>
              </div>
              <div className="grid gap-2">
                <Row v={br.tensor} label="my_tensor" tone="border-orange-200 bg-orange-50 text-orange-900" />
                <Row v={br.array} label="np_array" tone="border-sky-200 bg-sky-50 text-sky-900" />
              </div>
              <Note tone={br.shared ? 'good' : 'warn'}>{t(br.shared ? 'lab.pytorch.shared' : 'lab.pytorch.not_shared')}</Note>
            </div>
          </div>
        </Card>

        <Card step={2} title={t('lab.pytorch.map_title')} sub={t('lab.pytorch.map_sub')}>
          <div className="mb-3 flex flex-wrap gap-1.5">
            {STEPS.map((s) => (
              <button key={s} type="button" aria-pressed={s === step} onClick={() => setStep(s)} className={`rounded-full border px-2.5 py-1 text-[12px] font-bold ${s === step ? 'border-amber-400 bg-amber-100 text-amber-900' : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300'}`}>
                {t(`lab.pytorch.map.${s}.name`)}
              </button>
            ))}
          </div>
          <div className="mb-3"><Note title={t(`lab.pytorch.map.${step}.name`)}>{t(`lab.pytorch.map.${step}.text`)}</Note></div>
          <div className="grid gap-4 lg:grid-cols-2">
            <CodePane head={t('lab.pytorch.keras_head')} lines={KERAS} step={step} onPick={setStep} />
            <CodePane head={t('lab.pytorch.torch_head')} lines={TORCH} step={step} onPick={setStep} />
          </div>
        </Card>

        <Card step={3} title={t('lab.pytorch.fw_title')} sub={t('lab.common.quiz_sub')}>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
            <div className="grid min-w-0 grid-cols-[minmax(0,1fr)] content-start gap-3">
              <div className="grid gap-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
                <Seg label={t('lab.pytorch.fw_padding')} value={padding} onChange={setPadding} options={[{ v: 1, label: t('lab.pytorch.fw_pad1') }, { v: 0, label: t('lab.pytorch.fw_pad0') }]} />
                <Seg label={t('lab.pytorch.fw_in')} value={inFeatures} onChange={setInFeatures} options={[{ v: 12544, label: '64 * 14 * 14' }, { v: 9216, label: '64 * 12 * 12' }]} />
              </div>
              <PyBlock className="overflow-x-auto rounded-xl bg-slate-900 p-3 text-[12px] leading-relaxed text-slate-100" code={initCode} />
              <Note>{t('lab.pytorch.fw_note')}</Note>
            </div>
            <div className="grid min-w-0 content-start gap-3">
              <TableWrap>
                <tr><th className="left">{t('lab.pytorch.fw_line')}</th><th className="left">{t('lab.pytorch.fw_shape')}</th><th>{t('lab.pytorch.fw_params')}</th></tr>
                <tr><td className="left font-mono">x</td><td className="left">(64, 1, 28, 28)</td><td>—</td></tr>
                {fw.map((r, i) => (
                  <tr key={r.code}>
                    <td className="left whitespace-nowrap font-mono">{r.code}</td>
                    {r.error ? <td className="left font-mono font-bold text-rose-600">RuntimeError</td> : <Ans k={`fw${i}`} v={shapeStr(r.shape)} className="left" />}
                    {r.params ? <Ans k={`fwp${i}`} v={r.params} /> : <td>0</td>}
                  </tr>
                ))}
                {!fwError && <tr><td className="left font-bold">{t('lab.pytorch.fw_total')}</td><td /><Ans k="fwtotal" v={fw.reduce((a, r) => a + r.params, 0)} /></tr>}
              </TableWrap>
              {fwError ? (
                <Note tone="bad">
                  {t('lab.pytorch.fw_err', { flat, inf: inFeatures })}
                  <code className="mt-1.5 block whitespace-pre-wrap break-all font-mono text-[12px]">RuntimeError: {fwError.error}</code>
                </Note>
              ) : (
                <Note tone="good">{t('lab.pytorch.fw_ok', { n: flat })}</Note>
              )}
              <Note>{t('lab.pytorch.fw_params_note')}</Note>
            </div>
          </div>
        </Card>

        <Card step={4} title={t('lab.pytorch.ag_title')} sub={t('lab.common.quiz_sub')}>
          <div className="mb-3"><Note>{t('lab.pytorch.ag_l6')}</Note></div>
          <div className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
            <div className="grid min-w-0 grid-cols-[minmax(0,1fr)] content-start gap-4">
              <div className="grid gap-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
                <Slider label="w" value={w} min={-2} max={2} step={0.25} onChange={(v) => resetGraph(() => setW(v))} />
                <Slider label="x" value={x} min={-3} max={3} step={0.5} onChange={(v) => resetGraph(() => setX(v))} />
                <Slider label="b" value={b} min={-2} max={2} step={0.25} onChange={(v) => resetGraph(() => setB(v))} />
                <Seg label={t('lab.pytorch.target')} value={y} onChange={(v) => resetGraph(() => setY(v))} options={[{ v: 0, label: 'y = 0' }, { v: 1, label: 'y = 1' }]} />
              </div>
              <div>
                <PyBlock className="overflow-x-auto rounded-xl bg-slate-900 p-3 text-[12px] leading-relaxed text-slate-100" aria-label={t('lab.pytorch.code_title')} code={gradCode} />
                <div className="mt-1.5 flex justify-end"><RunInPlayground lib="pytorch" code={`import torch\nimport torch.nn.functional as F\n${gradCode}`} /></div>
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
                <Btn onClick={() => setEvents([...events, 'backward'])} disabled={noGrad}><span className="font-mono text-[12px]">{highlightPython('loss.backward()', 'light')}</span></Btn>
                <Btn onClick={() => setEvents([...events, 'step'])} disabled={noGrad}><span className="font-mono text-[12px]">{highlightPython('optimizer.step()', 'light')}</span></Btn>
                <Btn onClick={() => setEvents([...events, 'zero_grad'])}><span className="font-mono text-[12px]">{highlightPython('optimizer.zero_grad()', 'light')}</span></Btn>
                <Btn onClick={() => setEvents([])}>{t('lab.common.reset')}</Btn>
              </div>
              <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 font-mono text-[13px] leading-relaxed text-slate-800">
                w = {fmt(st.w, 4)}, b = {fmt(st.b, 4)}
                <br />
                w.grad = {fmt(st.wGrad, 4)}, b.grad = {fmt(st.bGrad, 4)}
              </div>
              <Note tone={sinceZero > 1 ? 'warn' : 'info'}>{t(sinceZero > 1 ? 'lab.pytorch.accumulated' : 'lab.pytorch.accumulate', { n: sinceZero })}</Note>
              <p className="text-xs text-slate-500">{t('lab.pytorch.step_note')}</p>
              <label className="flex items-center gap-2 text-[13px] font-semibold text-slate-700">
                <input type="checkbox" checked={noGrad} onChange={(e) => setNoGrad(e.target.checked)} />
                <span className="font-mono">with torch.no_grad():</span>
              </label>
              {noGrad && <Note tone="warn">{t('lab.pytorch.no_grad')}</Note>}
            </div>
          </div>
        </Card>

        <Card step={5} title={t('lab.pytorch.loop_title')} sub={t('lab.pytorch.loop_sub')}>
          <div className="grid gap-4 md:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
            <div className="grid content-start gap-1.5">
              {POOL.filter((l) => !order.includes(l)).map((l) => (
                <button key={l} type="button" onClick={() => setOrder([...order, l])} className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-left font-mono text-[12.5px] text-slate-800 hover:border-blue-400">
                  {highlightPython(LOOP_CODE[l], 'light')}
                </button>
              ))}
              {order.length > 0 && <div className="mt-1"><Btn onClick={() => setOrder([])}>{t('lab.common.reset')}</Btn></div>}
            </div>
            <div className="grid min-w-0 content-start gap-2">
              <div className="min-h-[190px] overflow-x-auto rounded-xl bg-slate-900 p-3 font-mono text-[12px] leading-[1.7] text-slate-100">
                <div className="w-max min-w-full">
                  {LOOP_HEAD.map((l) => <div key={l} className="whitespace-pre px-1.5">{highlightPython(l, 'dark')}</div>)}
                  {order.map((l) => (
                    <button key={l} type="button" onClick={() => setOrder(order.filter((o) => o !== l))} className="block w-full whitespace-pre rounded px-1.5 text-left ring-amber-300/70 hover:bg-white/10 hover:ring-1">
                      {highlightPython(`        ${LOOP_CODE[l]}`, 'dark')}
                    </button>
                  ))}
                  {order.length < LOOP.length && <div className="whitespace-pre px-1.5 text-slate-500">{`        ${t('lab.pytorch.loop_slot')}`}</div>}
                </div>
              </div>
              {order.length === LOOP.length && (
                <>
                  <Note tone={err ? 'bad' : 'good'}>{t(err ? `lab.pytorch.loop_err.${err}` : 'lab.pytorch.loop_ok')}</Note>
                  {/* the five lines run as they are ordered, inside a few epochs on made-up data: a wrong order shows up in the loss */}
                  <div className="flex justify-end">
                    <RunInPlayground lib="pytorch" code={`${LOOP_SETUP}\nfor epoch in range(5):\n${loopBody.split('\n').map((l) => `    ${l}`).join('\n')}\n    print(epoch, round(loss.item(), 4))`} />
                  </div>
                </>
              )}
              <Note>{t('lab.pytorch.eval_note')}</Note>
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
