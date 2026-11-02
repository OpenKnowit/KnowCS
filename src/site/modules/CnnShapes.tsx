import { useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { ChevronDown, ChevronRight, ChevronsDownUp, ChevronsUpDown, CircleHelp, Code2, Play, Plus, Table2, Undo2, X } from 'lucide-react'
import { analyse, totalParams, type Layer, type LayerInfo, type Padding, type Shape } from '../../lib/cnnShapes'
import { kerasCode, kerasSummary, type Head } from '../../lib/keras'
import { torchCode } from '../../lib/torchCode'
import { parseKeras, parseTorch } from '../../lib/cnnParse'
import { outputSize } from '../../lib/conv2d'
import { int } from '../format'
import { CodeEditor } from '../playground/CodeEditor'
import { Ans, LabPage, NumberField, Presets, Seg, Slider, Stat, TableWrap } from '../ui'

const conv = (filters: number, k: number, stride = 1, pad: Padding = 0): Layer => ({ kind: 'conv', filters, k, stride, pad, bias: true })
const pool = (k: number, stride = k, pad: Padding = 0): Layer => ({ kind: 'pool', k, stride, pad, op: 'max' })
const dense = (units: number): Layer => ({ kind: 'dense', units, bias: true })
const flatten: Layer = { kind: 'flatten' }

interface Preset {
  id: string
  input: Shape
  layers: Layer[]
}

const PRESETS: Preset[] = [
  { id: 'aio', input: [32, 32, 3], layers: [conv(10, 5, 1, 2)] },
  { id: 'mnist', input: [28, 28, 1], layers: [conv(32, 3), conv(64, 3), pool(2), { kind: 'dropout', rate: 0.25 }, flatten, dense(128), { kind: 'dropout', rate: 0.5 }, dense(10)] },
  { id: 'stem', input: [224, 224, 3], layers: [conv(64, 7, 2, 3), pool(3, 2, 1), conv(128, 3, 2, 0), conv(256, 3, 2, 1), conv(512, 3, 2, 1), { kind: 'globalpool', op: 'avg' }, dense(1000)] },
  { id: 'f23', input: [128, 128, 3], layers: [conv(32, 5, 1, 2), pool(2), conv(64, 5, 1, 2), pool(2), flatten, dense(128), dense(10)] },
  { id: 'same', input: [32, 32, 3], layers: [conv(32, 5, 1, 'same'), conv(64, 3, 1, 'same'), pool(2), conv(64, 3, 2, 'same'), conv(64, 3, 1, 'same'), pool(2), flatten, dense(128), dense(10)] },
  { id: 'f22keras', input: [27, 27, 3], layers: [conv(32, 3, 3), conv(64, 3, 2), pool(2), flatten, dense(12)] },
  { id: 'mlp', input: [224, 224, 3], layers: [flatten, dense(1000)] },
]

const KINDS: Layer['kind'][] = ['conv', 'pool', 'globalpool', 'flatten', 'dense', 'dropout']

function newLayer(kind: Layer['kind']): Layer {
  switch (kind) {
    case 'conv': return conv(16, 3, 1, 1)
    case 'pool': return pool(2)
    case 'globalpool': return { kind: 'globalpool', op: 'avg' }
    case 'flatten': return flatten
    case 'dense': return dense(10)
    case 'dropout': return { kind: 'dropout', rate: 0.5 }
  }
}

function PadInput({ value, onChange }: { value: Padding; onChange: (p: Padding) => void }) {
  const { t } = useTranslation()
  return (
    <label className="grid gap-1">
      <span className="text-xs font-bold text-slate-500">{t('lab.cnn.f.padding')}</span>
      <select value={String(value)} onChange={(e) => onChange(e.target.value === 'same' || e.target.value === 'valid' ? e.target.value : Number(e.target.value))} className="rounded-[10px] border border-slate-300 bg-white px-2 py-1.5 font-mono text-sm">
        {[0, 1, 2, 3].map((p) => <option key={p} value={p}>{p}</option>)}
        <option value="same">same</option>
        <option value="valid">valid</option>
      </select>
    </label>
  )
}

function LayerEditor({ layer, onChange }: { layer: Layer; onChange: (l: Layer) => void }) {
  const { t } = useTranslation()
  if (layer.kind === 'conv')
    return (
      <div className="grid grid-cols-4 gap-1.5">
        <NumberField label={t('lab.cnn.f.filters')} value={layer.filters} min={1} onChange={(v) => onChange({ ...layer, filters: Math.max(1, Math.round(v)) })} />
        <NumberField label={t('lab.cnn.f.kernel')} value={layer.k} min={1} onChange={(v) => onChange({ ...layer, k: Math.max(1, Math.round(v)) })} />
        <NumberField label={t('lab.cnn.f.stride')} value={layer.stride} min={1} onChange={(v) => onChange({ ...layer, stride: Math.max(1, Math.round(v)) })} />
        <PadInput value={layer.pad} onChange={(pad) => onChange({ ...layer, pad })} />
      </div>
    )
  if (layer.kind === 'pool')
    return (
      <div className="grid grid-cols-3 gap-1.5">
        <NumberField label={t('lab.cnn.f.size')} value={layer.k} min={1} onChange={(v) => onChange({ ...layer, k: Math.max(1, Math.round(v)) })} />
        <NumberField label={t('lab.cnn.f.stride')} value={layer.stride} min={1} onChange={(v) => onChange({ ...layer, stride: Math.max(1, Math.round(v)) })} />
        <PadInput value={layer.pad} onChange={(pad) => onChange({ ...layer, pad })} />
      </div>
    )
  if (layer.kind === 'dense') return <NumberField label={t('lab.cnn.f.units')} value={layer.units} min={1} onChange={(v) => onChange({ ...layer, units: Math.max(1, Math.round(v)) })} />
  return null
}

function StrideStrip() {
  const { t } = useTranslation()
  const [n, setN] = useState(7)
  const [k, setK] = useState(3)
  const [p, setP] = useState(0)
  const [s, setS] = useState(3)
  const total = n + 2 * p
  const o = outputSize(n, k, p, s)
  const cell = 30
  return (
    <div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Slider label={t('lab.cnn.strip.n')} value={n} min={3} max={12} step={1} onChange={setN} />
        <Slider label={t('lab.cnn.strip.k')} value={k} min={1} max={5} step={1} onChange={(v) => setK(Math.min(v, n + 2 * p))} />
        <Slider label={t('lab.cnn.strip.p')} value={p} min={0} max={3} step={1} onChange={setP} />
        <Slider label={t('lab.cnn.strip.s')} value={s} min={1} max={4} step={1} onChange={setS} />
      </div>
      <div className="mt-3 overflow-x-auto">
        <svg viewBox={`0 0 ${total * cell + 20} ${40 + o.size * 14 + 20}`} style={{ width: total * cell + 20, maxWidth: 'none' }} role="img" aria-label={t('lab.cnn.strip_aria')}>
          {Array.from({ length: total }, (_, i) => {
            const isPad = i < p || i >= p + n
            const covered = Array.from({ length: o.size }, (_, j) => j * s).some((st) => i >= st && i < st + k)
            return <rect key={i} x={10 + i * cell} y={6} width={cell - 3} height={26} rx={5} fill={isPad ? '#f1f5f9' : covered ? '#dbeafe' : '#fee2e2'} stroke={isPad ? '#cbd5e1' : covered ? '#60a5fa' : '#f87171'} strokeDasharray={isPad ? '3 2' : undefined} />
          })}
          {Array.from({ length: o.size }, (_, j) => (
            <g key={j}>
              <rect x={10 + j * s * cell} y={40 + j * 14} width={k * cell - 3} height={10} rx={4} fill="#2563eb" opacity={0.75} />
            </g>
          ))}
        </svg>
      </div>
      <div className="formula mt-2">
        {`⌊(${n} − ${k} + 2·${p}) / ${s}⌋ + 1 = ⌊${n - k + 2 * p} / ${s}⌋ + 1 = `}<b className="text-blue-700">{o.size}</b>
        {o.exact ? '' : `   ← ${t('lab.cnn.not_div', { a: n - k + 2 * p, s })}`}
      </div>
    </div>
  )
}


// colours per layer kind: [front face, side face, outline]
const KIND_COL: Record<string, [string, string, string]> = {
  input: ['#f1f5f9', '#cbd5e1', '#64748b'],
  output: ['#f1f5f9', '#cbd5e1', '#64748b'],
  conv: ['#dbeafe', '#93c5fd', '#3b82f6'],
  pool: ['#ede9fe', '#c4b5fd', '#8b5cf6'],
  globalpool: ['#ccfbf1', '#5eead4', '#14b8a6'],
  flatten: ['#fef3c7', '#fcd34d', '#f59e0b'],
  dense: ['#e0e7ff', '#a5b4fc', '#6366f1'],
  dropout: ['#f8fafc', '#e2e8f0', '#94a3b8'],
}
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

/** The volume after each layer, scaled to fit its box: no scrolling, few layers are not blown up past a sensible size. */
function Volumes({ input, infos, active, onPick }: { input: Shape; infos: LayerInfo[]; active: number | null; onPick: (i: number) => void }) {
  const { t } = useTranslation()
  const base = 186
  let x = 0
  const items = [{ s: input, kind: 'input', err: false }, ...infos.map((i, j) => ({ s: i.output, kind: j === infos.length - 1 ? 'output' : (i.layer.kind as string), err: !!i.error }))].map((it, i) => {
    const flat = it.s.length === 1
    const h = flat ? clamp(26 + Math.log2(Math.max(1, it.s[0])) * 8, 26, 160) : clamp(22 + Math.log2(Math.max(1, it.s[0])) * 14, 22, 160)
    const d = flat ? 12 : clamp(6 + Math.log2(Math.max(1, it.s[2])) * 5, 6, 60)
    const skew = flat ? 0 : h * 0.35
    const label = it.s.join('×')
    const name = i === 0 ? t('lab.cnn.input_name') : it.kind === 'output' ? t('lab.cnn.output_name') : t(`lab.cnn.kinds.${it.kind}`)
    const slot = Math.max(d + skew, label.length * 7.8, name.length * 6.4) + 8
    const item = { ...it, flat, h, d, skew, label, name, x, slot }
    x += slot + 34
    return item
  })
  const total = x - 34
  const W = Math.max(total + 24, 460)
  const off = (W - total) / 2
  return (
    <svg viewBox={`0 0 ${W} 236`} preserveAspectRatio="xMidYMid meet" className="h-full w-full" role="img" aria-label={t('lab.cnn.volumes_aria')}>
      {items.map((it, i) => {
        const cx = off + it.x + it.slot / 2
        const x0 = cx - (it.d + it.skew) / 2
        const top = base - it.h
        const [front, side, stroke] = KIND_COL[it.kind]
        const on = active === i - 1
        const sw = on ? 2.5 : 1.2
        return (
          <g key={i} onClick={() => onPick(i - 1)} className="cursor-pointer">
            <rect x={off + it.x} y={0} width={it.slot} height={236} fill="transparent" />
            {it.flat ? (
              <rect x={x0} y={top} width={it.d} height={it.h} rx={3} fill={side} stroke={stroke} strokeWidth={sw} strokeDasharray={it.kind === 'dropout' ? '4 3' : undefined} />
            ) : (
              <g strokeWidth={sw} stroke={stroke} strokeLinejoin="round" strokeDasharray={it.kind === 'dropout' ? '4 3' : undefined}>
                <polygon points={`${x0},${top + it.skew} ${x0 + it.skew},${top} ${x0 + it.skew + it.d},${top} ${x0 + it.d},${top + it.skew}`} fill={front} />
                <polygon points={`${x0 + it.d},${top + it.skew} ${x0 + it.skew + it.d},${top} ${x0 + it.skew + it.d},${base - it.skew} ${x0 + it.d},${base}`} fill={side} />
                <rect x={x0} y={top + it.skew} width={it.d} height={it.h - it.skew} fill={front} />
              </g>
            )}
            <text x={cx} y={base + 22} textAnchor="middle" className={`font-mono text-[13px] font-bold ${it.err ? 'fill-rose-600' : on ? 'fill-blue-700' : 'fill-slate-700'}`}>
              {it.err ? '!' : it.label}
            </text>
            <text x={cx} y={base + 39} textAnchor="middle" className={`text-[11px] ${on ? 'fill-blue-600 font-bold' : 'fill-slate-500'}`}>
              {it.name}
            </text>
            {i < items.length - 1 && (
              <text x={off + it.x + it.slot + 17} y={base - 40} textAnchor="middle" className="fill-slate-300 text-[18px]">
                →
              </text>
            )}
          </g>
        )
      })}
    </svg>
  )
}

const layerNote = (l: Layer): string =>
  l.kind === 'conv' ? `${l.filters}@${l.k}×${l.k} s${l.stride} p${l.pad}` : l.kind === 'pool' ? `${l.op} ${l.k}×${l.k} s${l.stride} p${l.pad}` : l.kind === 'dense' ? `${l.units}` : l.kind === 'dropout' ? `${l.rate}` : ''

const Chevron = ({ open }: { open: boolean }) => <ChevronRight size={15} aria-hidden className={`shrink-0 text-slate-400 transition-transform ${open ? 'rotate-90' : ''}`} />

/** The Example button: a small menu of the presets. */
function ExampleMenu({ value, onPick }: { value: string | null; onPick: (id: string) => void }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  return (
    <div className="relative">
      <button type="button" aria-expanded={open} onClick={() => setOpen((o) => !o)} className="flex items-center gap-1 rounded-full border border-slate-300 bg-white px-3 py-1 text-[13px] font-bold text-slate-700 hover:border-blue-400 hover:text-blue-700">
        {t('lab.common.example')}
        <ChevronDown size={14} aria-hidden />
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-20" onClick={() => setOpen(false)} />
          <div className="absolute right-0 top-full z-30 mt-1.5 max-h-[70vh] w-[300px] overflow-y-auto rounded-2xl border border-slate-200 bg-white p-2 shadow-xl shadow-slate-900/10">
            <Presets
              items={PRESETS.map((p) => ({ id: p.id, title: t(`lab.cnn.presets.${p.id}.title`), note: t(`lab.cnn.presets.${p.id}.note`) }))}
              value={value}
              onPick={(id) => {
                onPick(id)
                setOpen(false)
              }}
            />
          </div>
        </>
      )}
    </div>
  )
}

type Fw = 'keras' | 'torch'
interface LabState {
  input: Shape
  layers: Layer[]
  head: Head
}
const parserOf = (fw: Fw) => (fw === 'keras' ? parseKeras : parseTorch)

/** The generated code in an editor: edits are read back into the layers; the picked layer's lines are marked. */
function CodePanel({ st, active, onParsed, onHead, canUndo, onUndo }: { st: LabState; active: number | null; onParsed: (s: LabState) => void; onHead: (h: Head) => void; canUndo: boolean; onUndo: () => void }) {
  const { t } = useTranslation()
  const { input, layers, head } = st
  const [fw, setFw] = useState<Fw>('keras')
  const [draft, setDraft] = useState<{ fw: Fw; text: string } | null>(null)
  const generated = (fw === 'keras' ? kerasCode : torchCode)(input, layers, head)?.join('\n') ?? null
  // the layers moved some other way (a field, a preset, undo): the code is rewritten from them.
  // Edits read back from the code set `seen` themselves, so typing never resets the text.
  const [seen, setSeen] = useState(st)
  if (seen !== st) {
    setSeen(st)
    setDraft(null)
  }
  const mine = draft?.fw === fw ? draft.text : null
  const text = mine ?? generated ?? ''
  const parsed = useMemo(() => parserOf(fw)(text), [fw, text])
  const bad = mine !== null && !parsed.ok
  const marks = active !== null && active >= 0 && parsed.ok ? parsed.lines[active] : undefined
  const change = (v: string) => {
    setDraft({ fw, text: v })
    const p = parserOf(fw)(v)
    if (p.ok) {
      const next = { input: p.input, layers: p.layers, head: p.head }
      setSeen(next)
      onParsed(next)
    }
  }
  // bring the picked layer's code into view
  const box = useRef<HTMLDivElement>(null)
  const first = marks?.[0]
  useEffect(() => {
    const el = box.current
    if (!el || first === undefined) return
    const y = 12 + (first - 1) * 24
    if (y < el.scrollTop + 8 || y > el.scrollTop + el.clientHeight - 56) el.scrollTo({ top: Math.max(0, y - 48), behavior: 'smooth' })
  }, [first, active])
  const run = (() => {
    if (fw === 'keras') return /\.summary\s*\(/.test(text) ? text : `${text}\nmodel.summary()`
    const [h, w, c] = input
    return /print\s*\(/.test(text) ? text : `${text}\n\nx = torch.randn(1, ${c}, ${h}, ${w})\nprint(model(x).shape)`
  })()
  const href = `/playground/?code=${encodeURIComponent(run)}&from=${encodeURIComponent(window.location.pathname)}#${fw === 'keras' ? 'keras' : 'pytorch'}`
  const tab = (f: Fw) => `px-3 py-2 text-[13px] font-bold ${fw === f ? 'border-b-2 border-sky-400 text-white' : 'border-b-2 border-transparent text-slate-400 hover:text-slate-200'}`
  return (
    <section className="flex min-h-[320px] min-w-0 flex-col overflow-hidden rounded-2xl bg-slate-900 lg:min-h-0">
      <div className="flex items-center gap-1 border-b border-slate-700/70 pl-3 pr-2">
        <div className="flex" role="tablist" aria-label={t('lab.cnn.framework')}>
          {(['keras', 'torch'] as const).map((f) => (
            <button key={f} type="button" role="tab" aria-selected={fw === f} onClick={() => setFw(f)} className={tab(f)}>
              {f === 'keras' ? 'Keras' : 'PyTorch'}
            </button>
          ))}
        </div>
        <div className="ml-auto flex overflow-hidden rounded-lg border border-slate-600" role="group" aria-label={t('lab.cnn.head')}>
          {(['classify', 'regress'] as const).map((h) => (
            <button key={h} type="button" aria-pressed={head === h} title={t(`lab.cnn.head_${h}_note`)} onClick={() => onHead(h)} className={`px-2.5 py-1 text-xs font-bold ${head === h ? 'bg-sky-500 text-white' : 'text-slate-300 hover:bg-slate-800'}`}>
              {t(`lab.cnn.head_${h}`)}
            </button>
          ))}
        </div>
      </div>
      <div ref={box} className="min-h-0 flex-1 overflow-auto [&>div]:rounded-none [&>div]:border-0">
        {generated === null && mine === null ? (
          <p className="p-4 text-sm text-amber-300">{t(fw === 'torch' && kerasCode(input, layers, head) ? 'lab.cnn.torch_even' : 'lab.cnn.keras_error')}</p>
        ) : (
          <CodeEditor value={text} onChange={change} onRun={() => window.location.assign(href)} errorLine={bad && !parsed.ok ? parsed.line : undefined} marks={marks} label={t(fw === 'keras' ? 'lab.cnn.keras_code' : 'lab.cnn.torch_code')} minRows={8} />
        )}
      </div>
      <div className="flex flex-wrap items-center gap-2 border-t border-slate-700/70 px-3 py-2">
        <span className={`min-w-0 flex-1 text-xs ${bad ? 'text-rose-300' : 'text-slate-400'}`}>
          {bad && !parsed.ok ? t('lab.cnn.code_bad', { line: parsed.line, msg: t(`lab.cnn.parse.${parsed.error}`) }) : t('lab.cnn.code_synced')}
        </span>
        <button
          type="button"
          onClick={() => (bad ? setDraft(null) : onUndo())}
          disabled={!bad && !canUndo}
          title={t('lab.cnn.undo_hint')}
          className="flex items-center gap-1.5 rounded-lg border border-slate-600 px-2.5 py-1.5 text-xs font-bold text-slate-200 hover:border-slate-400 disabled:opacity-40"
        >
          <Undo2 size={13} aria-hidden />
          {t('lab.cnn.undo')}
        </button>
        <a href={href} className="flex items-center gap-1.5 rounded-lg bg-blue-600 px-2.5 py-1.5 text-xs font-bold text-white hover:bg-blue-500">
          <Play size={13} aria-hidden />
          {t('lab.cnn.run_playground')}
        </a>
      </div>
    </section>
  )
}

/** The calculation table: every layer with its working, the totals and the MLP comparison; or Keras's model.summary(). */
function CalcTable({ st, infos }: { st: LabState; infos: LayerInfo[] }) {
  const { t } = useTranslation()
  const { input, layers, head } = st
  const [view, setView] = useState<'layers' | 'summary'>('layers')
  const total = totalParams(infos)
  const firstDense = infos.find((i) => i.layer.kind === 'dense')
  const mlp = input.length === 3 && firstDense?.layer.kind === 'dense' ? (input[0] * input[1] * input[2] + 1) * firstDense.layer.units : null
  const summary = view === 'summary' ? kerasSummary(input, layers, head) : null
  const sp = (s: Shape) => s.join(' × ')
  return (
    <div className="grid min-h-0 min-w-0 content-start gap-3 overflow-auto lg:h-full">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Seg options={[{ v: 'layers' as const, label: t('lab.cnn.table_title') }, { v: 'summary' as const, label: 'model.summary()' }]} value={view} onChange={setView} />
        <span className="text-xs text-slate-500">{t('lab.common.quiz_sub')}</span>
      </div>
      {view === 'layers' ? (
        <TableWrap>
          <tr>
            <th className="left">{t('lab.cnn.th.layer')}</th>
            <th>{t('lab.cnn.th.input')}</th>
            <th>{t('lab.cnn.th.output')}</th>
            <th>{t('lab.cnn.th.params')}</th>
            <th className="left">{t('lab.cnn.th.working')}</th>
          </tr>
          {infos.map((info, i) => (
            <tr key={i} className={info.error ? 'dim' : ''}>
              <td className="left">
                {i + 1}. {t(`lab.cnn.kinds.${info.layer.kind}`)} {layerNote(info.layer)}
              </td>
              <td>{sp(info.input)}</td>
              {info.error ? (
                <td className="changed" colSpan={3}>
                  {t(`lab.errors.cnn_${info.error}`)}
                </td>
              ) : (
                <>
                  <Ans k={`out${i}`} v={sp(info.output)} />
                  <Ans k={`p${i}`} v={info.params} />
                  <td className="left font-sans text-xs text-slate-500">
                    {[info.sizeWork, info.paramWork].filter(Boolean).join(' · ') || (info.layer.kind === 'dropout' ? t('lab.cnn.dropout_work') : '')}
                    {info.floored && <b className="ml-1 text-amber-700">{t('lab.cnn.floored')}</b>}
                  </td>
                </>
              )}
            </tr>
          ))}
        </TableWrap>
      ) : summary ? (
        <TableWrap>
          <tr>
            <th className="left">Layer (type)</th>
            <th className="left">Output Shape</th>
            <th>Param #</th>
          </tr>
          {summary.rows.map((r) => (
            <tr key={r.name}>
              <td className="left">
                {r.name} ({r.cls})
              </td>
              <td className="left">{r.shape}</td>
              <Ans k={`k_${r.name}`} v={r.params} />
            </tr>
          ))}
          <tr>
            <td className="left font-sans font-bold" colSpan={2}>
              Total params
            </td>
            <Ans k="k_total" v={summary.total} />
          </tr>
        </TableWrap>
      ) : (
        <p className="text-sm text-rose-700">{t('lab.cnn.keras_error')}</p>
      )}
      <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
        <Stat k={t('lab.cnn.total')} v={int(total)} tone="blue" />
        {mlp !== null && <Stat k={t('lab.cnn.mlp')} v={int(mlp)} d={`Flatten ${sp(input)} → Dense ${firstDense?.layer.kind === 'dense' ? firstDense.layer.units : ''}`} tone="bad" />}
        {mlp !== null && total > 0 && <Stat k={t('lab.cnn.ratio')} v={`${(mlp / total).toFixed(1)}×`} d={t('lab.cnn.ratio_d')} />}
      </div>
    </div>
  )
}

/** Beside the code: each layer's input, output and parameters, the total pinned below. Rows pick the layer too. */
function ParamList({ st, infos, active, setActive }: { st: LabState; infos: LayerInfo[]; active: number | null; setActive: (i: number | null) => void }) {
  const { t } = useTranslation()
  const total = totalParams(infos)
  const firstDense = infos.find((i) => i.layer.kind === 'dense')
  const { input } = st
  const mlp = input.length === 3 && firstDense?.layer.kind === 'dense' ? (input[0] * input[1] * input[2] + 1) * firstDense.layer.units : null
  return (
    <section className="flex min-h-[320px] min-w-0 flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white lg:min-h-0">
      <div className="min-h-0 flex-1 overflow-auto">
        <table className="lab-table">
          <tbody>
            <tr className="sticky top-0 z-10">
              <th className="left">{t('lab.cnn.th.layer')}</th>
              <th>{t('lab.cnn.th.input')}</th>
              <th>{t('lab.cnn.th.output')}</th>
              <th>{t('lab.cnn.th.params')}</th>
            </tr>
            {infos.map((info, i) => (
              <tr key={i} onClick={() => setActive(active === i ? null : i)} className={`cursor-pointer hover:bg-slate-50 ${active === i ? 'cur' : ''} ${info.error ? 'dim' : ''}`}>
                <td className="left">
                  <span className="inline-flex items-center gap-1.5 font-sans font-semibold text-slate-800">
                    <i className="h-2.5 w-2.5 rounded-[3px]" style={{ background: KIND_COL[info.layer.kind][2] }} />
                    {i + 1}. {t(`lab.cnn.kinds.${info.layer.kind}`)}
                  </span>
                  <span className="ml-1.5 text-[11px] text-slate-500">{layerNote(info.layer)}</span>
                </td>
                <td>{info.input.join('×')}</td>
                {info.error ? (
                  <td className="changed whitespace-normal font-sans" colSpan={2}>
                    {t(`lab.errors.cnn_${info.error}`)}
                  </td>
                ) : (
                  <>
                    <Ans k={`out${i}`} v={info.output.join('×')} />
                    <Ans k={`p${i}`} v={info.params} />
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 border-t border-slate-200 bg-slate-50 px-4 py-2.5">
        <span className="text-xs font-extrabold uppercase tracking-wider text-slate-500">{t('lab.cnn.total')}</span>
        <b className="font-mono text-lg text-blue-700">{int(total)}</b>
        {mlp !== null && total > 0 && (
          <span className="ml-auto text-xs text-slate-500">
            {t('lab.cnn.mlp')}: <b className="font-mono text-rose-600">{int(mlp)}</b> ({(mlp / total).toFixed(1)}×)
          </span>
        )}
      </div>
    </section>
  )
}

/** The ? in the add block: what may go at the chosen position, and which parameters are valid there. */
function LayerGuide({ at, shape }: { at: number; shape: Shape }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const flat = shape.length !== 3
  const n = shape.reduce((a, b) => a * b, 1)
  const side = flat ? 0 : Math.min(shape[0], shape[1])
  const rule = (k: Layer['kind']): { ok: boolean; text: string } => {
    if (flat && (k === 'conv' || k === 'pool' || k === 'globalpool' || k === 'flatten')) return { ok: false, text: t('lab.cnn.guide.need_3d') }
    if (!flat && k === 'dense') return { ok: false, text: t('lab.cnn.guide.need_flat') }
    if (k === 'conv') return { ok: true, text: t('lab.cnn.guide.conv', { max: side }) }
    if (k === 'pool') return { ok: true, text: t('lab.cnn.guide.pool', { max: side }) }
    if (k === 'globalpool') return { ok: true, text: t('lab.cnn.guide.globalpool', { c: shape[2] }) }
    if (k === 'flatten') return { ok: true, text: t('lab.cnn.guide.flatten', { n }) }
    if (k === 'dense') return { ok: true, text: t('lab.cnn.guide.dense', { n }) }
    return { ok: true, text: t('lab.cnn.guide.dropout') }
  }
  return (
    <div className="relative">
      <button type="button" aria-expanded={open} aria-label={t('lab.cnn.guide_title')} title={t('lab.cnn.guide_title')} onClick={() => setOpen((o) => !o)} className="grid h-[34px] w-[34px] place-items-center rounded-[10px] border border-slate-300 bg-white text-slate-600 hover:border-blue-400 hover:text-blue-700">
        <CircleHelp size={16} aria-hidden />
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-20" onClick={() => setOpen(false)} />
          <div role="dialog" aria-label={t('lab.cnn.guide_title')} className="absolute bottom-full right-0 z-30 mb-2 w-[320px] rounded-2xl border border-slate-200 bg-white p-3.5 text-[13px] leading-snug text-slate-700 shadow-xl shadow-slate-900/10">
            <div className="mb-2.5 flex items-center justify-between gap-2">
              <b className="text-slate-900">{t('lab.cnn.guide.at', { n: at + 1, shape: shape.join('×') })}</b>
              <button type="button" aria-label={t('site.ui.close')} onClick={() => setOpen(false)} className="text-slate-400 hover:text-slate-700">
                <X size={16} aria-hidden />
              </button>
            </div>
            <ul className="grid gap-1.5">
              {KINDS.map((k) => {
                const r = rule(k)
                return (
                  <li key={k} className={`flex gap-2 ${r.ok ? '' : 'opacity-55'}`}>
                    <span className={`w-3 shrink-0 font-bold ${r.ok ? 'text-emerald-600' : 'text-rose-500'}`}>{r.ok ? '✓' : '✗'}</span>
                    <span>
                      <b>{t(`lab.cnn.kinds.${k}`)}</b> <span className="font-mono text-xs text-slate-600">{r.text}</span>
                    </span>
                  </li>
                )
              })}
            </ul>
          </div>
        </>
      )}
    </div>
  )
}

/** The layer settings: input, every layer (open by default, ↑ ↓ ✕), output, then the add block. */
function LayersPanel({ st, infos, active, setActive, commit, example }: { st: LabState; infos: LayerInfo[]; active: number | null; setActive: (i: number | null) => void; commit: (s: LabState) => void; example: ReactNode }) {
  const { t } = useTranslation()
  const { input, layers } = st
  const OUT = -2 // the output card's key in `closed`
  const [closed, setClosed] = useState<Set<number>>(() => new Set())
  const [addKind, setAddKind] = useState<Layer['kind']>('conv')
  const [addAt, setAddAt] = useState(-1) // -1 = after the last layer
  const [hoverAdd, setHoverAdd] = useState(false)
  const at = addAt < 0 || addAt > layers.length ? layers.length : addAt
  const allOpen = closed.size === 0
  const setLayers = (next: Layer[]) => commit({ ...st, layers: next })
  const move = (i: number, j: number) => {
    const n = [...layers]
    ;[n[i], n[j]] = [n[j], n[i]]
    setLayers(n)
    setActive(j)
  }
  const toggle = (i: number) =>
    setClosed((c) => {
      const n = new Set(c)
      if (n.has(i)) n.delete(i)
      else n.add(i)
      return n
    })
  const last = layers[layers.length - 1]
  const out = infos.length ? infos[infos.length - 1].output : input
  const shapeAt = at === 0 ? input : infos[at - 1].output
  const iconBtn = 'grid h-6 w-6 place-items-center rounded text-slate-500 hover:bg-white disabled:opacity-30'
  // floats in the gap above card i, so showing it never moves the add block out from under the pointer
  const marker = (i: number) => hoverAdd && at === i && <div aria-hidden className="pointer-events-none absolute inset-x-0 -top-[6px] h-1 rounded-full bg-blue-500 shadow-[0_0_0_3px_rgba(59,130,246,0.2)]" />
  const card = (on: boolean, err = false) => `rounded-xl border p-2 ${err ? 'border-rose-300 bg-rose-50/70' : on ? 'border-blue-400 bg-blue-50/60' : 'border-slate-200 bg-slate-50'}`
  const head = (key: number, label: ReactNode, color: string, right: ReactNode, actions?: ReactNode) => (
    <div className="flex min-w-0 items-center gap-1.5">
      <button type="button" aria-expanded={!closed.has(key)} aria-label={t(closed.has(key) ? 'site.ui.expand_all' : 'site.ui.collapse_all')} onClick={() => toggle(key)} className="grid h-6 w-6 shrink-0 place-items-center rounded hover:bg-white">
        <Chevron open={!closed.has(key)} />
      </button>
      <button type="button" onClick={() => key > -2 && setActive(active === key ? null : key)} className="flex min-w-0 flex-1 items-center gap-2 text-left">
        <i className="h-2.5 w-2.5 shrink-0 rounded-[3px]" style={{ background: color }} />
        <b className="truncate text-[13px]">{label}</b>
        <span className="ml-auto shrink-0 font-mono text-xs text-slate-600">{right}</span>
      </button>
      {actions}
    </div>
  )
  return (
    <aside className="relative z-10 flex min-h-[320px] min-w-0 flex-col rounded-[20px] border border-slate-200 bg-white lg:min-h-0">
      <div className="flex items-center gap-1.5 border-b border-slate-100 px-3 py-2.5">
        <h2 className="mr-auto text-[15px] font-extrabold">{t('lab.cnn.layers')}</h2>
        <button
          type="button"
          onClick={() => setClosed(allOpen ? new Set([-1, OUT, ...layers.map((_, i) => i)]) : new Set())}
          aria-label={t(allOpen ? 'site.ui.collapse_all' : 'site.ui.expand_all')}
          title={t(allOpen ? 'site.ui.collapse_all' : 'site.ui.expand_all')}
          className="grid h-[30px] w-[30px] place-items-center rounded-lg border border-slate-200 text-slate-600 hover:border-slate-300 hover:text-slate-900"
        >
          {allOpen ? <ChevronsDownUp size={15} aria-hidden /> : <ChevronsUpDown size={15} aria-hidden />}
        </button>
        {example}
      </div>
      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)] content-start gap-2 overflow-y-auto p-3">
        <div className={card(active === -1)}>
          {head(-1, t('lab.cnn.input_name'), KIND_COL.input[2], input.join('×'))}
          {!closed.has(-1) && (
            <div className="mt-1.5 grid grid-cols-3 gap-1.5">
              {['height', 'width', 'channels'].map((name, i) => (
                <NumberField
                  key={name}
                  label={t(`lab.cnn.f.${name}`)}
                  value={input[i]}
                  min={1}
                  onChange={(v) => {
                    const s = [...input]
                    s[i] = Math.max(1, Math.round(v))
                    commit({ ...st, input: s })
                  }}
                />
              ))}
            </div>
          )}
        </div>
        {layers.map((l, i) => {
          const info = infos[i]
          const open = !closed.has(i)
          return (
            <div key={i} className="relative grid grid-cols-[minmax(0,1fr)]">
              {marker(i)}
              <div className={card(active === i, !!info.error)}>
                {head(
                  i,
                  `${i + 1}. ${t(`lab.cnn.kinds.${l.kind}`)}`,
                  KIND_COL[l.kind][2],
                  <span className={info.error ? 'text-rose-600' : ''}>{info.error ? '!' : info.output.join('×')}</span>,
                  <span className="flex shrink-0">
                    <button type="button" aria-label={t('lab.cnn.move_up')} disabled={i === 0} onClick={() => move(i, i - 1)} className={iconBtn}>↑</button>
                    <button type="button" aria-label={t('lab.cnn.move_down')} disabled={i === layers.length - 1} onClick={() => move(i, i + 1)} className={iconBtn}>↓</button>
                    <button type="button" aria-label={t('lab.cnn.remove')} onClick={() => { setLayers(layers.filter((_, j) => j !== i)); setActive(null) }} className={`${iconBtn} text-rose-600`}>✕</button>
                  </span>,
                )}
                {open && (l.kind === 'conv' || l.kind === 'pool' || l.kind === 'dense' || info.error) && (
                  <div className="mt-1.5" onFocus={() => setActive(i)}>
                    <LayerEditor layer={l} onChange={(nl) => setLayers(layers.map((x, j) => (j === i ? nl : x)))} />
                    {info.error && <p className="mt-1.5 text-xs font-semibold text-rose-700">{t(`lab.errors.cnn_${info.error}`)}</p>}
                  </div>
                )}
              </div>
            </div>
          )
        })}
        <div className={`relative ${card(false)}`}>
          {marker(layers.length)}
          {head(OUT, t('lab.cnn.output_name'), KIND_COL.output[2], out.join('×'))}
          {!closed.has(OUT) &&
            (last?.kind === 'dense' ? (
              <div className="mt-1.5 grid grid-cols-2 gap-1.5">
                {st.head === 'regress' ? (
                  <NumberField label={t('lab.cnn.output_units')} value={1} min={1} max={1} onChange={() => {}} />
                ) : (
                  <NumberField label={t('lab.cnn.output_units')} value={last.units} min={1} onChange={(v) => setLayers(layers.map((x, j) => (j === layers.length - 1 ? { ...last, units: Math.max(1, Math.round(v)) } : x)))} />
                )}
              </div>
            ) : (
              <p className="mt-1 px-1 text-xs text-slate-500">{t('lab.cnn.output_no_dense')}</p>
            ))}
        </div>
        <div
          className={`flex items-center gap-1.5 rounded-xl border border-dashed p-2 transition ${hoverAdd ? 'border-blue-400 bg-blue-50/50' : 'border-slate-300'}`}
          onMouseEnter={() => setHoverAdd(true)}
          onMouseLeave={() => setHoverAdd(false)}
          onFocus={() => setHoverAdd(true)}
          onBlur={() => setHoverAdd(false)}
        >
          <select value={addKind} onChange={(e) => setAddKind(e.target.value as Layer['kind'])} aria-label={t('lab.cnn.add')} className="min-w-0 flex-1 rounded-[10px] border border-slate-300 bg-white px-2 py-1.5 text-sm">
            {KINDS.map((k) => (
              <option key={k} value={k}>
                {t(`lab.cnn.kinds.${k}`)}
              </option>
            ))}
          </select>
          <label className="flex items-center gap-1 font-mono text-sm text-slate-500">
            #
            <select value={addAt < 0 ? -1 : at} onChange={(e) => setAddAt(Number(e.target.value))} aria-label={t('lab.cnn.add_at')} className="rounded-[10px] border border-slate-300 bg-white px-1.5 py-1.5 font-mono text-sm text-slate-800">
              {layers.map((_, i) => (
                <option key={i} value={i}>
                  {i + 1}
                </option>
              ))}
              <option value={-1}>-1</option>
            </select>
          </label>
          <button
            type="button"
            aria-label={t('lab.cnn.add')}
            title={t('lab.cnn.add')}
            onClick={() => {
              setLayers([...layers.slice(0, at), newLayer(addKind), ...layers.slice(at)])
              setActive(at)
              setClosed((c) => new Set([...c].map((j) => (j >= at ? j + 1 : j))))
            }}
            className="grid h-[34px] w-[34px] place-items-center rounded-[10px] bg-blue-600 text-white hover:bg-blue-700"
          >
            <Plus size={17} aria-hidden />
          </button>
          <LayerGuide at={at} shape={shapeAt} />
        </div>
      </div>
    </aside>
  )
}

const SAVE_KEY = 'knowcs-cnn-shapes'
const saved = (): { st: LabState; preset: string | null; mode: 'code' | 'table' } | null => {
  try {
    const v = JSON.parse(sessionStorage.getItem(SAVE_KEY) ?? 'null')
    return v && Array.isArray(v.st?.input) && Array.isArray(v.st?.layers) && (v.st.head === 'classify' || v.st.head === 'regress') ? v : null
  } catch {
    return null
  }
}

export default function CnnShapes() {
  const { t } = useTranslation()
  const [preset, setPreset] = useState<string | null>(() => saved()?.preset ?? 'stem')
  const [st, setSt] = useState<LabState>(() => saved()?.st ?? { input: PRESETS[2].input, layers: PRESETS[2].layers, head: 'classify' })
  const [past, setPast] = useState<LabState[]>([])
  const [active, setActive] = useState<number | null>(null)
  const [mode, setMode] = useState<'code' | 'table'>(() => saved()?.mode ?? 'code')
  // kept for this tab, so coming back from Run in Playground (or a reload) shows the network as it was left
  useEffect(() => {
    try {
      sessionStorage.setItem(SAVE_KEY, JSON.stringify({ st, preset, mode }))
    } catch {
      // storage off: the lab just starts fresh next time
    }
  }, [st, preset, mode])
  const infos = analyse(st.input, st.layers)

  const commit = (next: LabState) => {
    setPast((p) => [...p.slice(-79), st])
    setSt(next)
    setPreset(null)
  }
  // Regress means one sigmoid unit: the last Dense really becomes 1 unit (the drawing, table and output agree with the
  // code), and Classify gives back the class count it had before
  const [classUnits, setClassUnits] = useState<number | null>(null)
  const switchHead = (h: Head) => {
    if (h === st.head) return
    const li = st.layers.length - 1
    const last = st.layers[li]
    if (last?.kind !== 'dense') return commit({ ...st, head: h })
    if (h === 'regress') setClassUnits(last.units)
    const units = h === 'regress' ? 1 : last.units === 1 && classUnits ? classUnits : last.units
    commit({ ...st, head: h, layers: st.layers.map((x, j) => (j === li ? { ...last, units } : x)) })
  }
  const undo = () => {
    if (!past.length) return
    setSt(past[past.length - 1])
    setPast((p) => p.slice(0, -1))
  }
  // Ctrl/⌘+Z anywhere outside a text field undoes the last change to the network
  const undoRef = useRef(undo)
  useEffect(() => {
    undoRef.current = undo
  })
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement
      if ((e.metaKey || e.ctrlKey) && !e.shiftKey && e.key.toLowerCase() === 'z' && !el.closest('input, textarea, select, [contenteditable]')) {
        e.preventDefault()
        undoRef.current()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <LabPage id="cnn-shapes" quiz>
      <div className="grid grid-cols-[minmax(0,1fr)] gap-4 lg:h-[calc(100dvh-170px)] lg:min-h-[640px] lg:grid-cols-[360px_minmax(0,1fr)]">
        <LayersPanel
          st={st}
          infos={infos}
          active={active}
          setActive={setActive}
          commit={commit}
          example={
            <ExampleMenu
              value={preset}
              onPick={(id) => {
                const p = PRESETS.find((q) => q.id === id)!
                commit({ input: p.input, layers: p.layers, head: st.head })
                setPreset(id)
                setActive(null)
              }}
            />
          }
        />
        <div className="grid min-h-0 min-w-0 grid-cols-[minmax(0,1fr)] gap-4 lg:grid-rows-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
          <section className="flex min-h-0 min-w-0 flex-col rounded-[20px] border border-slate-200 bg-white px-4 pb-2 pt-3">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
              <h2 className="text-[15px] font-extrabold">{t('lab.cnn.volumes_title')}</h2>
              <ul className="ml-auto flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-600">
                {(['input', ...new Set(st.layers.slice(0, -1).map((l) => l.kind))] as string[]).map((k) => (
                  <li key={k} className="flex items-center gap-1.5">
                    <i className="h-2.5 w-2.5 rounded-[3px] border" style={{ background: KIND_COL[k][1], borderColor: KIND_COL[k][2] }} />
                    {k === 'input' ? `${t('lab.cnn.input_name').replace(/^./, (c) => c.toUpperCase())} / ${t('lab.cnn.output_name')}` : t(`lab.cnn.kinds.${k}`)}
                  </li>
                ))}
              </ul>
            </div>
            <div className="h-[220px] min-h-0 lg:h-auto lg:flex-1">
              <Volumes input={st.input} infos={infos} active={active} onPick={(i) => setActive(active === i ? null : i)} />
            </div>
          </section>

          <section className="flex min-h-0 min-w-0 flex-col gap-3 rounded-[20px] border border-slate-200 bg-white p-3">
            <div className="flex items-center gap-3">
              <div className="inline-flex rounded-[10px] bg-slate-100 p-[3px]" role="tablist" aria-label={t('lab.cnn.mode')}>
                {(['code', 'table'] as const).map((m) => (
                  <button key={m} type="button" role="tab" aria-selected={mode === m} onClick={() => setMode(m)} className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[13px] ${mode === m ? 'bg-white font-bold text-slate-900 shadow-sm' : 'font-medium text-slate-500 hover:text-slate-800'}`}>
                    {m === 'code' ? <Code2 size={14} aria-hidden /> : <Table2 size={14} aria-hidden />}
                    {t(`lab.cnn.mode_${m}`)}
                  </button>
                ))}
              </div>
            </div>
            {mode === 'code' ? (
              <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)] gap-3 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
                <CodePanel st={st} active={active} onParsed={(s) => commit(s)} onHead={switchHead} canUndo={past.length > 0} onUndo={undo} />
                <ParamList st={st} infos={infos} active={active} setActive={setActive} />
              </div>
            ) : (
              <div className="min-h-0 flex-1">
                <CalcTable st={st} infos={infos} />
              </div>
            )}
          </section>
        </div>
      </div>

      <details className="group mt-4 rounded-[20px] border border-slate-200 bg-white">
        <summary className="flex cursor-pointer list-none items-center gap-1.5 px-4 py-3 text-[14px] font-extrabold [&::-webkit-details-marker]:hidden">
          <ChevronRight size={15} aria-hidden className="text-slate-400 transition-transform group-open:rotate-90" />
          {t('lab.cnn.strip_title')}
          <span className="ml-2 text-xs font-normal text-slate-500">{t('lab.cnn.strip_sub')}</span>
        </summary>
        <div className="px-4 pb-4">
          <StrideStrip />
        </div>
      </details>
    </LabPage>
  )
}
