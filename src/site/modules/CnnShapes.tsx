import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ChevronDown, ChevronRight, Info, Play, RotateCcw } from 'lucide-react'
import { analyse, totalParams, type Layer, type LayerInfo, type Padding, type Shape } from '../../lib/cnnShapes'
import { kerasCode, kerasSummary, type Head } from '../../lib/keras'
import { torchCode } from '../../lib/torchCode'
import { parseKeras, parseTorch } from '../../lib/cnnParse'
import { outputSize } from '../../lib/conv2d'
import { int } from '../format'
import { CodeEditor } from '../playground/CodeEditor'
import { Ans, Btn, LabPage, NumberField, Presets, Seg, Slider } from '../ui'

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
const shapeStr = (s: Shape) => s.join('×')

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
  const items = [{ s: input, kind: 'input', err: false }, ...infos.map((i) => ({ s: i.output, kind: i.layer.kind as string, err: !!i.error }))].map((it, i) => {
    const flat = it.s.length === 1
    const h = flat ? clamp(26 + Math.log2(Math.max(1, it.s[0])) * 8, 26, 160) : clamp(22 + Math.log2(Math.max(1, it.s[0])) * 14, 22, 160)
    const d = flat ? 12 : clamp(6 + Math.log2(Math.max(1, it.s[2])) * 5, 6, 60)
    const skew = flat ? 0 : h * 0.35
    const label = it.s.join('×')
    const name = i === 0 ? t('lab.cnn.input_name') : t(`lab.cnn.kinds.${it.kind}`)
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
          <div className="absolute left-0 top-full z-30 mt-1.5 w-[300px] rounded-2xl border border-slate-200 bg-white p-2 shadow-xl shadow-slate-900/10">
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

/** The generated code in an editor: edits are read back into the layers; Reset rewrites it from the layers. */
function CodePanel({ input, layers, head, onParsed }: { input: Shape; layers: Layer[]; head: Head; onParsed: (input: Shape, layers: Layer[], head: Head) => void }) {
  const { t } = useTranslation()
  const [fw, setFw] = useState<Fw>('keras')
  const [draft, setDraft] = useState<{ fw: Fw; text: string } | null>(null)
  const generated = (fw === 'keras' ? kerasCode : torchCode)(input, layers, head)?.join('\n') ?? null
  // the layers moved some other way (a field, a preset, the head): the code is rewritten from them.
  // Edits read back from the code set `seen` themselves, so typing never resets the text.
  const [seen, setSeen] = useState({ input, layers, head })
  if (seen.input !== input || seen.layers !== layers || seen.head !== head) {
    setSeen({ input, layers, head })
    setDraft(null)
  }
  const mine = draft?.fw === fw ? draft.text : null
  const text = mine ?? generated ?? ''
  const parsed = useMemo(() => (mine === null ? null : (fw === 'keras' ? parseKeras : parseTorch)(mine)), [mine, fw])
  const change = (v: string) => {
    setDraft({ fw, text: v })
    const p = (fw === 'keras' ? parseKeras : parseTorch)(v)
    if (p.ok) {
      setSeen({ input: p.input, layers: p.layers, head: p.head })
      onParsed(p.input, p.layers, p.head)
    }
  }
  const run = (() => {
    if (fw === 'keras') return /\.summary\s*\(/.test(text) ? text : `${text}\nmodel.summary()`
    const [h, w, c] = input
    return /print\s*\(/.test(text) ? text : `${text}\n\nx = torch.randn(1, ${c}, ${h}, ${w})\nprint(model(x).shape)`
  })()
  const href = `/playground/?code=${encodeURIComponent(run)}#${fw === 'keras' ? 'keras' : 'pytorch'}`
  const tab = (f: Fw) => `px-3 py-2 text-[13px] font-bold ${fw === f ? 'border-b-2 border-sky-400 text-white' : 'border-b-2 border-transparent text-slate-400 hover:text-slate-200'}`
  return (
    <section className="flex min-h-[320px] min-w-0 flex-col overflow-hidden rounded-[20px] bg-slate-900 lg:min-h-0">
      <div className="flex items-center gap-1 border-b border-slate-700/70 px-3" role="tablist" aria-label={t('lab.cnn.framework')}>
        {(['keras', 'torch'] as const).map((f) => (
          <button key={f} type="button" role="tab" aria-selected={fw === f} onClick={() => setFw(f)} className={tab(f)}>
            {f === 'keras' ? 'Keras' : 'PyTorch'}
          </button>
        ))}
        <span className="ml-auto grid h-6 w-6 cursor-help place-items-center text-slate-400" title={t(fw === 'keras' ? 'lab.cnn.keras_note' : 'lab.cnn.torch_note')}>
          <Info size={15} aria-hidden />
          <span className="sr-only">{t(fw === 'keras' ? 'lab.cnn.keras_note' : 'lab.cnn.torch_note')}</span>
        </span>
      </div>
      <div className="min-h-0 flex-1 overflow-auto [&>div]:rounded-none [&>div]:border-0">
        {generated === null && mine === null ? (
          <p className="p-4 text-sm text-amber-300">{t(fw === 'torch' && kerasCode(input, layers, head) ? 'lab.cnn.torch_even' : 'lab.cnn.keras_error')}</p>
        ) : (
          <CodeEditor value={text} onChange={change} onRun={() => window.location.assign(href)} errorLine={parsed && !parsed.ok ? parsed.line : undefined} label={t(fw === 'keras' ? 'lab.cnn.keras_code' : 'lab.cnn.torch_code')} minRows={8} />
        )}
      </div>
      <div className="flex flex-wrap items-center gap-2 border-t border-slate-700/70 px-3 py-2">
        <span className={`min-w-0 flex-1 text-xs ${parsed && !parsed.ok ? 'text-rose-300' : 'text-slate-400'}`}>
          {parsed && !parsed.ok ? t('lab.cnn.code_bad', { line: parsed.line, msg: t(`lab.cnn.parse.${parsed.error}`) }) : t('lab.cnn.code_synced')}
        </span>
        <button type="button" onClick={() => setDraft(null)} disabled={mine === null} title={t('lab.cnn.reset_hint')} className="flex items-center gap-1.5 rounded-lg border border-slate-600 px-2.5 py-1.5 text-xs font-bold text-slate-200 hover:border-slate-400 disabled:opacity-40">
          <RotateCcw size={13} aria-hidden />
          {t('lab.cnn.reset_code')}
        </button>
        <a href={href} className="flex items-center gap-1.5 rounded-lg bg-blue-600 px-2.5 py-1.5 text-xs font-bold text-white hover:bg-blue-500">
          <Play size={13} aria-hidden />
          {t('lab.cnn.run_playground')}
        </a>
      </div>
    </section>
  )
}

/** Shape and parameter table (the lab's layers, or Keras's model.summary() rows), with the totals pinned below. */
function ParamTable({ input, layers, infos, head, open, onToggle }: { input: Shape; layers: Layer[]; infos: LayerInfo[]; head: Head; open: boolean; onToggle: () => void }) {
  const { t } = useTranslation()
  const [view, setView] = useState<'layers' | 'summary'>('layers')
  const total = totalParams(infos)
  const firstDense = infos.find((i) => i.layer.kind === 'dense')
  const mlp = input.length === 3 && firstDense?.layer.kind === 'dense' ? (input[0] * input[1] * input[2] + 1) * firstDense.layer.units : null
  const summary = view === 'summary' ? kerasSummary(input, layers, head) : null
  return (
    <section className={`flex min-w-0 flex-col overflow-hidden rounded-[20px] border border-slate-200 bg-white ${open ? 'min-h-[320px] lg:min-h-0' : ''}`}>
      <div className="flex items-center gap-2 px-4 py-2.5">
        <button type="button" aria-expanded={open} onClick={onToggle} className="flex min-w-0 items-center gap-1.5 text-left text-[14px] font-extrabold text-slate-900">
          <Chevron open={open} />
          {t('lab.cnn.table_title')}
        </button>
        {open && (
          <div className="ml-auto">
            <Seg options={[{ v: 'layers' as const, label: t('lab.cnn.layers') }, { v: 'summary' as const, label: 'model.summary()' }]} value={view} onChange={setView} />
          </div>
        )}
        {!open && <b className="ml-auto font-mono text-[13px] text-blue-700">{int(total)}</b>}
      </div>
      {open && (
        <>
          <div className="min-h-0 flex-1 overflow-auto border-t border-slate-100">
            <table className="lab-table">
              <tbody>
                {view === 'layers' ? (
                  <>
                    <tr className="sticky top-0 z-10">
                      <th className="left">{t('lab.cnn.th.layer')}</th>
                      <th>{t('lab.cnn.th.input')}</th>
                      <th>{t('lab.cnn.th.output')}</th>
                      <th>{t('lab.cnn.th.params')}</th>
                    </tr>
                    {infos.map((info, i) => (
                      <tr key={i} className={info.error ? 'dim' : ''}>
                        <td className="left min-w-[200px] whitespace-normal">
                          <span className="font-sans font-semibold text-slate-800">
                            {i + 1}. {t(`lab.cnn.kinds.${info.layer.kind}`)}
                          </span>
                          <span className="ml-1.5 text-[11px] text-slate-500">{layerNote(info.layer)}</span>
                          {!info.error && (info.sizeWork || info.paramWork) && (
                            <div className="text-[11px] text-slate-500">
                              {[info.sizeWork, info.paramWork].filter(Boolean).join(' · ')}
                              {info.floored && <b className="ml-1 text-amber-700">{t('lab.cnn.floored')}</b>}
                            </div>
                          )}
                        </td>
                        <td>{shapeStr(info.input)}</td>
                        {info.error ? (
                          <td className="changed whitespace-normal font-sans" colSpan={2}>
                            {t(`lab.errors.cnn_${info.error}`)}
                          </td>
                        ) : (
                          <>
                            <Ans k={`out${i}`} v={shapeStr(info.output)} />
                            <Ans k={`p${i}`} v={info.params} />
                          </>
                        )}
                      </tr>
                    ))}
                  </>
                ) : summary ? (
                  <>
                    <tr className="sticky top-0 z-10">
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
                  </>
                ) : (
                  <tr>
                    <td className="whitespace-normal font-sans text-rose-700">{t('lab.cnn.keras_error')}</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 border-t border-slate-200 bg-slate-50 px-4 py-2.5">
            <span className="text-xs font-extrabold uppercase tracking-wider text-slate-500">{t('lab.cnn.total')}</span>
            <b className="font-mono text-lg text-blue-700">{int(total)}</b>
            {mlp !== null && total > 0 && (
              <span className="ml-auto text-xs text-slate-500" title={`Flatten ${shapeStr(input)} → Dense`}>
                {t('lab.cnn.mlp')}: <b className="font-mono text-rose-600">{int(mlp)}</b> ({(mlp / total).toFixed(1)}×)
              </span>
            )}
          </div>
        </>
      )}
    </section>
  )
}

export default function CnnShapes() {
  const { t } = useTranslation()
  const [preset, setPreset] = useState<string | null>('stem')
  const [input, setInput] = useState<Shape>(PRESETS[2].input)
  const [layers, setLayers] = useState<Layer[]>(PRESETS[2].layers)
  const [addKind, setAddKind] = useState<Layer['kind']>('conv')
  const [head, setHead] = useState<Head>('classify')
  // which row of the layer list is open: -1 = the input, null = none
  const [open, setOpen] = useState<number | null>(null)
  const [tableOpen, setTableOpen] = useState(true)

  const infos = analyse(input, layers)
  const last = infos.length ? infos[infos.length - 1].output : input
  const edit = (next: Layer[]) => {
    setLayers(next)
    setPreset(null)
  }
  const move = (i: number, j: number) => {
    const n = [...layers]
    ;[n[i], n[j]] = [n[j], n[i]]
    edit(n)
    setOpen(j)
  }
  const rowHead = 'flex w-full min-w-0 items-center gap-2 px-3 py-2 text-left'
  const iconBtn = 'grid h-6 w-6 place-items-center rounded text-slate-500 hover:bg-slate-100 disabled:opacity-30'

  return (
    <LabPage id="cnn-shapes" quiz>
      <div className="grid grid-cols-[minmax(0,1fr)] gap-4 lg:h-[calc(100dvh-170px)] lg:min-h-[600px] lg:grid-cols-[320px_minmax(0,1fr)]">
        {/* layers: input first, the stack, the output last */}
        <aside className="flex min-h-0 flex-col rounded-[20px] border border-slate-200 bg-white">
          <div className="flex items-center justify-between gap-2 px-4 pb-2 pt-3.5">
            <h2 className="text-[15px] font-extrabold">{t('lab.cnn.layers')}</h2>
            <ExampleMenu
              value={preset}
              onPick={(id) => {
                const p = PRESETS.find((q) => q.id === id)!
                setPreset(id)
                setInput(p.input)
                setLayers(p.layers)
                setOpen(null)
              }}
            />
          </div>
          <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)] content-start gap-1.5 overflow-y-auto px-3 pb-3">
            <div className={`rounded-xl border ${open === -1 ? 'border-slate-300 bg-slate-50' : 'border-slate-200'}`}>
              <button type="button" aria-expanded={open === -1} onClick={() => setOpen(open === -1 ? null : -1)} className={rowHead}>
                <Chevron open={open === -1} />
                <i className="h-2.5 w-2.5 shrink-0 rounded-[3px]" style={{ background: KIND_COL.input[2] }} />
                <b className="text-[13px]">{t('lab.cnn.input_name')}</b>
                <span className="ml-auto font-mono text-xs text-slate-600">{input.join('×')}</span>
              </button>
              {open === -1 && (
                <div className="grid grid-cols-3 gap-2 px-3 pb-3">
                  {['height', 'width', 'channels'].map((name, i) => (
                    <NumberField
                      key={name}
                      label={t(`lab.cnn.f.${name}`)}
                      value={input[i]}
                      min={1}
                      onChange={(v) => {
                        const s = [...input]
                        s[i] = Math.max(1, Math.round(v))
                        setInput(s)
                        setPreset(null)
                      }}
                    />
                  ))}
                </div>
              )}
            </div>
            {layers.map((l, i) => {
              const info = infos[i]
              const on = open === i
              return (
                <div key={i} className={`rounded-xl border ${info.error ? 'border-rose-300 bg-rose-50/60' : on ? 'border-blue-300 bg-blue-50/50' : 'border-slate-200'}`}>
                  <div className="flex items-center pr-1.5">
                    <button type="button" aria-expanded={on} onClick={() => setOpen(on ? null : i)} className={rowHead}>
                      <Chevron open={on} />
                      <i className="h-2.5 w-2.5 shrink-0 rounded-[3px]" style={{ background: KIND_COL[l.kind][2] }} />
                      <span className="min-w-0 truncate text-[13px]">
                        <b>{t(`lab.cnn.kinds.${l.kind}`)}</b> <span className="text-xs text-slate-500">{layerNote(l)}</span>
                      </span>
                      <span className={`ml-auto shrink-0 font-mono text-xs ${info.error ? 'text-rose-600' : 'text-slate-600'}`}>{info.error ? '!' : info.output.join('×')}</span>
                    </button>
                    {on && (
                      <span className="flex shrink-0">
                        <button type="button" aria-label={t('lab.cnn.move_up')} disabled={i === 0} onClick={() => move(i, i - 1)} className={iconBtn}>↑</button>
                        <button type="button" aria-label={t('lab.cnn.move_down')} disabled={i === layers.length - 1} onClick={() => move(i, i + 1)} className={iconBtn}>↓</button>
                        <button type="button" aria-label={t('lab.cnn.remove')} onClick={() => { edit(layers.filter((_, j) => j !== i)); setOpen(null) }} className={`${iconBtn} text-rose-600`}>✕</button>
                      </span>
                    )}
                  </div>
                  {on && (
                    <div className="px-3 pb-3">
                      <LayerEditor layer={l} onChange={(nl) => edit(layers.map((x, j) => (j === i ? nl : x)))} />
                      {info.error && <p className="mt-1.5 text-xs font-semibold text-rose-700">{t(`lab.errors.cnn_${info.error}`)}</p>}
                    </div>
                  )}
                </div>
              )
            })}
            <div className="flex gap-2 py-1">
              <select value={addKind} onChange={(e) => setAddKind(e.target.value as Layer['kind'])} aria-label={t('lab.cnn.add')} className="min-w-0 flex-1 rounded-[10px] border border-slate-300 bg-white px-2 py-1.5 text-sm">
                {KINDS.map((k) => (
                  <option key={k} value={k}>
                    {t(`lab.cnn.kinds.${k}`)}
                  </option>
                ))}
              </select>
              <Btn
                onClick={() => {
                  edit([...layers, newLayer(addKind)])
                  setOpen(layers.length)
                }}
              >
                + {t('lab.cnn.add')}
              </Btn>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5">
              <div className="flex items-center gap-2">
                <b className="text-[13px]">{t('lab.cnn.output_name')}</b>
                <span className="ml-auto font-mono text-xs text-slate-700">{last.join('×')}</span>
              </div>
              <div className="mt-2 flex gap-1.5">
                {(['classify', 'regress'] as const).map((h) => (
                  <button key={h} type="button" aria-pressed={head === h} title={t(`lab.cnn.head_${h}_note`)} onClick={() => setHead(h)} className={`rounded-lg border px-2.5 py-1 text-xs font-bold ${head === h ? 'border-blue-600 bg-blue-600 text-white' : 'border-slate-300 bg-white text-slate-600 hover:border-slate-400'}`}>
                    {t(`lab.cnn.head_${h}`)}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </aside>

        <div className="grid min-h-0 min-w-0 grid-cols-[minmax(0,1fr)] gap-4 lg:grid-rows-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
          <section className="flex min-h-0 min-w-0 flex-col rounded-[20px] border border-slate-200 bg-white p-4">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
              <h2 className="text-[15px] font-extrabold">{t('lab.cnn.volumes_title')}</h2>
              <ul className="ml-auto flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-600">
                {(['input', ...new Set(layers.map((l) => l.kind))] as string[]).map((k) => (
                  <li key={k} className="flex items-center gap-1.5">
                    <i className="h-2.5 w-2.5 rounded-[3px] border" style={{ background: KIND_COL[k][1], borderColor: KIND_COL[k][2] }} />
                    {k === 'input' ? t('lab.cnn.input_name') : t(`lab.cnn.kinds.${k}`)}
                  </li>
                ))}
              </ul>
            </div>
            <div className="h-[240px] min-h-0 lg:h-auto lg:flex-1">
              <Volumes input={input} infos={infos} active={open} onPick={(i) => setOpen(open === i ? null : i)} />
            </div>
          </section>
          <div className={`grid min-h-0 min-w-0 grid-cols-[minmax(0,1fr)] gap-4 ${tableOpen ? 'xl:grid-cols-2' : 'xl:grid-cols-[minmax(0,1fr)_auto]'}`}>
            <CodePanel
              input={input}
              layers={layers}
              head={head}
              onParsed={(i, l, h) => {
                setInput(i)
                setLayers(l)
                setHead(h)
                setPreset(null)
              }}
            />
            <ParamTable input={input} layers={layers} infos={infos} head={head} open={tableOpen} onToggle={() => setTableOpen((o) => !o)} />
          </div>
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
