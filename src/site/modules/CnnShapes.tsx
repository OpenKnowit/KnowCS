import { useState } from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { analyse, totalParams, type Layer, type Padding, type Shape } from '../../lib/cnnShapes'
import { outputSize } from '../../lib/conv2d'
import { int } from '../format'
import { Ans, Btn, Card, LabPage, Note, NumberField, Presets, Slider, Stat, TableWrap, Workspace } from '../ui'

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
  { id: 'mlp', input: [224, 224, 3], layers: [flatten, dense(1000)] },
]

const KINDS: Layer['kind'][] = ['conv', 'pool', 'globalpool', 'flatten', 'dense', 'dropout']
const shapeStr = (s: Shape) => s.join(' × ')

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

function Volumes({ input, infos }: { input: Shape; infos: ReturnType<typeof analyse> }) {
  const { t } = useTranslation()
  const shapes = [input, ...infos.map((i) => i.output)]
  const names = [t('lab.cnn.input_name'), ...infos.map((i) => t(`lab.cnn.kinds.${i.layer.kind}`))]
  const H = 190
  const size = (n: number) => 14 + Math.log2(Math.max(1, n)) * 11
  const gap = 26
  let x = 10
  const items = shapes.map((s, i) => {
    const flat = s.length === 1
    const h = flat ? Math.min(150, 20 + Math.log2(Math.max(1, s[0])) * 9) : Math.min(150, size(s[0]))
    const d = flat ? 10 : Math.min(60, 6 + Math.log2(Math.max(1, s[2])) * 5)
    const item = { x, h, d, flat, s, name: names[i] }
    x += (flat ? 14 : d + h * 0.35) + gap + 30
    return item
  })
  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${Math.max(400, x)} ${H + 46}`} style={{ minWidth: Math.max(400, x) }} className="h-auto" role="img" aria-label={t('lab.cnn.volumes_aria')}>
        {items.map((it, i) => {
          const base = H - 10
          const top = base - it.h
          const skew = it.flat ? 0 : it.h * 0.35
          return (
            <g key={i}>
              {it.flat ? (
                <rect x={it.x} y={top} width={14} height={it.h} rx={3} fill="#c7d2fe" stroke="#6366f1" />
              ) : (
                <g>
                  <polygon points={`${it.x},${top + skew} ${it.x + skew},${top} ${it.x + skew + it.d},${top} ${it.x + it.d},${top + skew}`} fill="#bfdbfe" stroke="#3b82f6" />
                  <polygon points={`${it.x + it.d},${top + skew} ${it.x + skew + it.d},${top} ${it.x + skew + it.d},${base - skew} ${it.x + it.d},${base}`} fill="#93c5fd" stroke="#3b82f6" />
                  <rect x={it.x} y={top + skew} width={it.d} height={it.h - skew} fill="#dbeafe" stroke="#3b82f6" />
                </g>
              )}
              <text x={it.x + 4} y={H + 10} className="fill-slate-700 font-mono text-[11px] font-bold">{it.s.join('×')}</text>
              <text x={it.x + 4} y={H + 26} className="fill-slate-400 text-[10px]">{it.name}</text>
              {i < items.length - 1 && <text x={it.x + (it.flat ? 14 : it.d + skew) + 10} y={base - it.h / 2} className="fill-slate-300 text-[16px]">→</text>}
            </g>
          )
        })}
      </svg>
    </div>
  )
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

export default function CnnShapes() {
  const { t } = useTranslation()
  const [preset, setPreset] = useState<string | null>('stem')
  const [input, setInput] = useState<Shape>(PRESETS[2].input)
  const [layers, setLayers] = useState<Layer[]>(PRESETS[2].layers)
  const [addKind, setAddKind] = useState<Layer['kind']>('conv')

  const infos = analyse(input, layers)
  const total = totalParams(infos)
  const firstDense = infos.find((i) => i.layer.kind === 'dense')
  const mlp = input.length === 3 && firstDense ? (input[0] * input[1] * input[2] + 1) * (firstDense.layer.kind === 'dense' ? firstDense.layer.units : 1) : null
  const edit = (next: Layer[]) => {
    setLayers(next)
    setPreset(null)
  }

  return (
    <LabPage id="cnn-shapes" quiz lead={t('lab.cnn.lead')}>
      <Workspace
        wide
        controls={
          <>
            <Card title={t('lab.common.example')}>
              <Presets
                items={PRESETS.map((p) => ({ id: p.id, title: t(`lab.cnn.presets.${p.id}.title`), note: t(`lab.cnn.presets.${p.id}.note`) }))}
                value={preset}
                onPick={(id) => {
                  const p = PRESETS.find((q) => q.id === id)!
                  setPreset(id)
                  setInput(p.input)
                  setLayers(p.layers)
                }}
              />
            </Card>
            <Card title={t('lab.cnn.input_title')}>
              <div className="grid grid-cols-3 gap-2">
                {['height', 'width', 'channels'].map((name, i) => (
                  <NumberField key={name} label={t(`lab.cnn.f.${name}`)} value={input[i]} min={1} onChange={(v) => { const s = [...input]; s[i] = Math.max(1, Math.round(v)); setInput(s); setPreset(null) }} />
                ))}
              </div>
            </Card>
            <Card title={t('lab.cnn.layers')}>
              <div className="grid gap-2">
                {layers.map((l, i) => (
                  <div key={i} className="rounded-xl border border-slate-200 bg-slate-50 p-2">
                    <div className="mb-1.5 flex items-center justify-between gap-2">
                      <b className="text-[13px]">{i + 1}. {t(`lab.cnn.kinds.${l.kind}`)}</b>
                      <span className="flex gap-1">
                        <button type="button" aria-label={t('lab.cnn.move_up')} disabled={i === 0} onClick={() => { const n = [...layers]; [n[i - 1], n[i]] = [n[i], n[i - 1]]; edit(n) }} className="rounded px-1.5 text-slate-500 hover:bg-white disabled:opacity-30">↑</button>
                        <button type="button" aria-label={t('lab.cnn.move_down')} disabled={i === layers.length - 1} onClick={() => { const n = [...layers]; [n[i + 1], n[i]] = [n[i], n[i + 1]]; edit(n) }} className="rounded px-1.5 text-slate-500 hover:bg-white disabled:opacity-30">↓</button>
                        <button type="button" aria-label={t('lab.cnn.remove')} onClick={() => edit(layers.filter((_, j) => j !== i))} className="rounded px-1.5 text-rose-500 hover:bg-white">✕</button>
                      </span>
                    </div>
                    <LayerEditor layer={l} onChange={(nl) => edit(layers.map((x, j) => (j === i ? nl : x)))} />
                  </div>
                ))}
                <div className="flex gap-2">
                  <select value={addKind} onChange={(e) => setAddKind(e.target.value as Layer['kind'])} className="flex-1 rounded-[10px] border border-slate-300 bg-white px-2 py-1.5 text-sm">
                    {KINDS.map((k) => <option key={k} value={k}>{t(`lab.cnn.kinds.${k}`)}</option>)}
                  </select>
                  <Btn onClick={() => edit([...layers, newLayer(addKind)])}>{t('lab.cnn.add')}</Btn>
                </div>
              </div>
            </Card>
          </>
        }
      >
        <Card step={1} title={t('lab.cnn.volumes_title')} sub={t('lab.cnn.volumes_sub')}>
          <Volumes input={input} infos={infos} />
        </Card>

        <Card step={2} title={t('lab.cnn.table_title')} sub={t('lab.common.quiz_sub')}>
          <TableWrap>
            <tr>
              <th className="left">{t('lab.cnn.th.layer')}</th><th>{t('lab.cnn.th.input')}</th><th>{t('lab.cnn.th.output')}</th><th>{t('lab.cnn.th.params')}</th><th className="left">{t('lab.cnn.th.working')}</th>
            </tr>
            {infos.map((info, i) => (
              <tr key={i} className={info.error ? 'dim' : ''}>
                <td className="left">{i + 1}. {t(`lab.cnn.kinds.${info.layer.kind}`)}{info.layer.kind === 'conv' ? ` ${info.layer.filters}@${info.layer.k}×${info.layer.k}, s${info.layer.stride}, p${info.layer.pad}` : info.layer.kind === 'pool' ? ` ${info.layer.k}×${info.layer.k}, s${info.layer.stride}, p${info.layer.pad}` : info.layer.kind === 'dense' ? ` ${info.layer.units}` : ''}</td>
                <td>{shapeStr(info.input)}</td>
                {info.error ? <td className="changed" colSpan={3}>{t(`lab.errors.cnn_${info.error}`)}</td> : (
                  <>
                    <Ans k={`out${i}`} v={shapeStr(info.output)} />
                    <Ans k={`p${i}`} v={info.params} />
                    <td className="left font-sans text-xs text-slate-500">
                      {[info.sizeWork, info.paramWork].filter(Boolean).join(' · ') || (info.layer.kind === 'dropout' ? t('lab.cnn.dropout_work') : '')}
                      {info.floored && <b className="ml-1 text-amber-600">{t('lab.cnn.floored')}</b>}
                    </td>
                  </>
                )}
              </tr>
            ))}
          </TableWrap>
          <div className="mt-3 grid grid-cols-2 gap-2.5 md:grid-cols-3">
            <Stat k={t('lab.cnn.total')} v={int(total)} tone="blue" />
            {mlp !== null && <Stat k={t('lab.cnn.mlp')} v={int(mlp)} d={`Flatten ${shapeStr(input)} → Dense ${firstDense?.layer.kind === 'dense' ? firstDense.layer.units : ''}`} tone="bad" />}
            {mlp !== null && total > 0 && <Stat k={t('lab.cnn.ratio')} v={`${(mlp / total).toFixed(1)}×`} d={t('lab.cnn.ratio_d')} />}
          </div>
          {preset === 'stem' && (
            <div className="mt-3">
              <Note tone="warn" title={t('lab.cnn.slip_title')}>{t('lab.cnn.slip')}</Note>
            </div>
          )}
        </Card>

        <Card step={3} title={t('lab.cnn.strip_title')} sub={t('lab.cnn.strip_sub')}>
          <StrideStrip />
        </Card>

        <div className="grid gap-5 md:grid-cols-2">
          <Note tone="warn" title={t('lab.cnn.rules_title')}>
            <Trans i18nKey="lab.cnn.rules" components={{ 1: <sub /> }} />
          </Note>
          <Note title={t('lab.cnn.why_title')}>{t('lab.cnn.why')}</Note>
        </div>
      </Workspace>
    </LabPage>
  )
}
