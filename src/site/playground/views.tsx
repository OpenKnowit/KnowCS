import { useMemo, useState } from 'react'
import { ArrowRight } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import type { CallTrace } from '../../lib/minipy'
import type { FlowRow, GraphNode, PyEvent, TableSpec } from '../../lib/pyEvents'
import { niceTicks } from '../../lib/chart'
import { CallView } from '../../modules/NpCallView'
import { FigureView } from './FigureView'

// --- 实验台每一步的可视化：调用网格、图、DataFrame、模型形状流、自动求导计算图、训练曲线 ---

const shape = (s: (number | null)[]) => `(${s.map((d) => (d === null ? 'None' : d)).join(', ')}${s.length === 1 ? ',' : ''})`

// ---------------------------------------------------------------- tables

interface TableProps {
  table: TableSpec
  /** row / column positions to highlight */
  rows?: Set<number> | null
  cols?: Set<number> | null
  /** colour per row (groupby) */
  rowTint?: (r: number) => string | undefined
  title?: string
}

const MAX_ROWS = 30

export const TableView = ({ table, rows, cols, rowTint, title }: TableProps) => {
  const { t } = useTranslation()
  const cut = table.index.length > MAX_ROWS
  const shown = cut ? [...table.index.keys()].slice(0, MAX_ROWS) : [...table.index.keys()]
  const colNames = table.series ? [table.name ?? ''] : table.columns
  const hl = (r: number, c: number) => (rows && rows.has(r)) || (cols && cols.has(c))
  return (
    <div className="min-w-0">
      {title && <div className="mb-1 font-mono text-xs font-bold text-slate-600">{title}</div>}
      <div className="overflow-x-auto rounded-lg border border-slate-200">
        <table className="min-w-full border-collapse font-mono text-xs">
          <thead>
            <tr className="bg-slate-50">
              <th className="border-b border-slate-200 px-2 py-1 text-left font-normal italic text-slate-400">{table.indexName ?? ''}</th>
              {colNames.map((c, k) => (
                <th key={k} className={`border-b border-slate-200 px-2 py-1 text-right font-bold ${cols?.has(k) ? 'bg-amber-100 text-amber-900' : 'text-slate-700'}`}>
                  {c}
                  {table.dtypes[k] && <span className="block text-[9px] font-normal text-slate-400">{table.dtypes[k]}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r} style={{ background: rowTint?.(r) }} className={rows?.has(r) ? 'bg-amber-50' : ''}>
                <th className={`border-b border-slate-100 px-2 py-1 text-left font-bold ${rows?.has(r) ? 'text-amber-800' : 'text-slate-500'}`}>{table.index[r]}</th>
                {table.cells[r].map((v, c) => (
                  <td key={c} className={`border-b border-slate-100 px-2 py-1 text-right ${v === null ? 'italic text-rose-700' : hl(r, c) ? 'bg-amber-100 font-bold text-amber-900' : 'text-slate-700'}`}>
                    {v ?? 'NaN'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-0.5 text-[10px] text-slate-400">
        {table.series ? t('playground.view.series_shape', { n: table.index.length }) : t('playground.view.frame_shape', { r: table.index.length, c: table.columns.length })}
        {cut && ` · ${t('playground.view.rows_cut', { n: MAX_ROWS })}`}
      </div>
    </div>
  )
}

const GROUP_TINT = ['#e0f2fe', '#fce7f3', '#dcfce7', '#fef3c7', '#ede9fe', '#ffe4e6', '#ccfbf1', '#f1f5f9']

const FrameStep = ({ ev }: { ev: PyEvent & { type: 'frame' } }) => {
  const { t } = useTranslation()
  const rows = ev.picks?.rows ? new Set(ev.picks.rows) : null
  const cols = ev.picks?.cols ? new Set(ev.picks.cols) : null
  const tint = ev.groups ? (r: number) => GROUP_TINT[ev.groups!.ofRow[r] % GROUP_TINT.length] : undefined
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <code className="rounded-lg bg-slate-900 px-2.5 py-1 font-mono text-xs text-slate-100">{ev.code}</code>
        <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-[10px] font-black uppercase text-indigo-700">{ev.api}</span>
      </div>
      <p className="text-sm leading-relaxed text-slate-600">{t(`playground.frame.${ev.api}`, { defaultValue: t('playground.frame.generic') })}</p>
      {ev.groups && (
        <div className="flex flex-wrap gap-1.5 text-[11px]">
          {ev.groups.keys.map((k, i) => (
            <span key={k} className="rounded-md px-2 py-0.5 font-mono text-slate-700" style={{ background: GROUP_TINT[i % GROUP_TINT.length] }}>{k}</span>
          ))}
        </div>
      )}
      <div className="flex flex-col gap-3 xl:flex-row xl:items-start">
        {ev.inputs.map((inp, k) => <TableView key={k} table={inp.table} title={inp.label} rows={k === 0 ? rows : null} cols={k === 0 ? cols : null} rowTint={k === 0 ? tint : undefined} />)}
        {ev.inputs.length > 0 && <ArrowRight className="hidden shrink-0 self-center text-slate-300 xl:block" size={20} aria-hidden />}
        {ev.result ? <TableView table={ev.result} title={t('playground.view.result')} /> : <pre className="rounded-lg bg-slate-50 px-3 py-2 font-mono text-xs text-slate-700">{ev.resultText}</pre>}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- model shape flow

const FlowStep = ({ ev }: { ev: PyEvent & { type: 'flow' } }) => {
  const { t } = useTranslation()
  const sizes = ev.rows.map((r) => r.output.reduce<number>((a, b) => a * (b ?? 1), 1))
  const maxLog = Math.log10(Math.max(10, ...sizes))
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <code className="rounded-lg bg-slate-900 px-2.5 py-1 font-mono text-xs text-slate-100">{ev.code}</code>
        <span className="rounded-full bg-violet-100 px-2 py-0.5 text-[10px] font-black uppercase text-violet-700">{ev.title}</span>
      </div>
      <p className="text-sm leading-relaxed text-slate-600">{t(`playground.view.flow_${ev.framework}`)}</p>
      <div className="overflow-x-auto rounded-lg border border-slate-200">
        <table className="min-w-full border-collapse text-xs">
          <thead className="bg-slate-50 text-left text-[11px] text-slate-500">
            <tr>
              <th className="px-2 py-1.5">{t('playground.view.layer')}</th>
              <th className="px-2 py-1.5">{t('playground.view.in_out')}</th>
              <th className="px-2 py-1.5 text-right">{t('playground.view.params')}</th>
            </tr>
          </thead>
          <tbody>
            {ev.rows.map((r: FlowRow, k) => (
              <tr key={k} className="border-t border-slate-100 align-top">
                <td className="px-2 py-1.5">
                  <div className="font-mono font-bold text-slate-800">{r.name}</div>
                  <div className="font-mono text-[11px] text-slate-500">{r.layer}</div>
                </td>
                <td className="px-2 py-1.5">
                  <div className="whitespace-nowrap font-mono text-slate-700">{shape(r.input)} → <b className="text-violet-700">{shape(r.output)}</b></div>
                  <div className="mt-1 h-1.5 rounded-full bg-violet-100">
                    <div className="h-1.5 rounded-full bg-violet-500" style={{ width: `${Math.max(4, (Math.log10(Math.max(1, sizes[k])) / maxLog) * 100)}%` }} />
                  </div>
                  {r.note && <div className="mt-1 font-mono text-[11px] text-slate-500">{r.note}</div>}
                </td>
                <td className="px-2 py-1.5 text-right font-mono text-slate-700">{r.params.toLocaleString('en-US')}</td>
              </tr>
            ))}
          </tbody>
          <tfoot className="border-t border-slate-200 bg-slate-50 font-mono text-[11px] text-slate-600">
            <tr>
              <td className="px-2 py-1.5" colSpan={2}>{t('playground.view.total_params')}</td>
              <td className="px-2 py-1.5 text-right font-bold">{ev.total.toLocaleString('en-US')}</td>
            </tr>
            {ev.trainable !== ev.total && (
              <tr>
                <td className="px-2 py-1" colSpan={2}>{t('playground.view.trainable')}</td>
                <td className="px-2 py-1 text-right">{ev.trainable.toLocaleString('en-US')}</td>
              </tr>
            )}
          </tfoot>
        </table>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- autograd graph

const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s)

const GraphStep = ({ ev }: { ev: PyEvent & { type: 'graph' } }) => {
  const { t } = useTranslation()
  const [hover, setHover] = useState<number | null>(null)
  const layout = useMemo(() => {
    // column = longest distance from a leaf; the root (loss) is in the last column
    const byId = new Map(ev.nodes.map((n) => [n.id, n]))
    const col = new Map<number, number>()
    const depth = (id: number): number => {
      if (col.has(id)) return col.get(id)!
      const n = byId.get(id)!
      const d = n.inputs.length ? 1 + Math.max(...n.inputs.map(depth)) : 0
      col.set(id, d)
      return d
    }
    ev.nodes.forEach((n) => depth(n.id))
    const cols = Math.max(...col.values()) + 1
    const perCol: number[][] = Array.from({ length: cols }, () => [])
    ev.nodes.forEach((n) => perCol[col.get(n.id)!].push(n.id))
    const pos = new Map<number, { x: number; y: number }>()
    const W = 196
    const H = 66
    perCol.forEach((ids, c) => ids.forEach((id, r) => pos.set(id, { x: 10 + c * (W + 40), y: 10 + r * (H + 18) + ((Math.max(...perCol.map((p) => p.length)) - ids.length) * (H + 18)) / 2 })))
    return { pos, W, H, width: 20 + cols * (W + 40) - 40, height: 20 + Math.max(...perCol.map((p) => p.length)) * (H + 18) - 18, byId }
  }, [ev.nodes])
  const { pos, W, H } = layout
  const active = hover === null ? null : new Set([hover, ...(layout.byId.get(hover)?.inputs ?? [])])
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <code className="rounded-lg bg-slate-900 px-2.5 py-1 font-mono text-xs text-slate-100">{ev.code}</code>
        <span className="rounded-full bg-rose-100 px-2 py-0.5 text-[10px] font-black uppercase text-rose-700">{ev.title}</span>
      </div>
      <p className="text-sm leading-relaxed text-slate-600">{t('playground.view.graph')}</p>
      <div className="overflow-x-auto rounded-lg border border-slate-200 bg-slate-50/60 p-2">
        <svg viewBox={`0 0 ${layout.width} ${layout.height}`} className="block w-full" style={{ minWidth: Math.min(layout.width, 560) }} role="img" aria-label={t('playground.view.graph_label')}>
          <defs>
            <marker id="gArrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0L8,4L0,8z" fill="#94a3b8" /></marker>
          </defs>
          {ev.nodes.flatMap((n) => n.inputs.map((i) => {
            const a = pos.get(i)!
            const b = pos.get(n.id)!
            const on = active?.has(n.id) && active.has(i)
            return <path key={`${i}-${n.id}`} d={`M${a.x + W},${a.y + H / 2} C${a.x + W + 22},${a.y + H / 2} ${b.x - 22},${b.y + H / 2} ${b.x},${b.y + H / 2}`} fill="none" stroke={on ? '#e11d48' : '#94a3b8'} strokeWidth={on ? 2 : 1.2} markerEnd="url(#gArrow)" />
          }))}
          {ev.nodes.map((n: GraphNode) => {
            const p = pos.get(n.id)!
            const root = n.id === ev.root
            return (
              <g key={n.id} onMouseEnter={() => setHover(n.id)} onMouseLeave={() => setHover(null)} transform={`translate(${p.x} ${p.y})`}>
                <rect width={W} height={H} rx={10} fill={root ? '#fff1f2' : n.leaf ? (n.requiresGrad ? '#eff6ff' : '#f8fafc') : '#ffffff'} stroke={hover === n.id ? '#e11d48' : root ? '#fda4af' : n.leaf && n.requiresGrad ? '#93c5fd' : '#cbd5e1'} strokeWidth={hover === n.id ? 2 : 1} />
                <title>{[n.label, n.op, `${t('playground.view.value')} ${n.value}`, n.grad ? `grad ${n.grad}` : ''].filter(Boolean).join('\n')}</title>
                <text x={8} y={16} fontSize={11} fontWeight={700} fill="#0f172a" fontFamily="ui-monospace, monospace">{clip(n.label, 14)}</text>
                {n.op && <text x={W - 8} y={16} fontSize={9} textAnchor="end" fill="#64748b" fontFamily="ui-monospace, monospace">{clip(n.op.replace(/Backward\d*$/, ''), 16)}</text>}
                <text x={8} y={34} fontSize={10} fill="#334155" fontFamily="ui-monospace, monospace">{clip(`${t('playground.view.value')} ${n.value}`, 30)}</text>
                <text x={8} y={52} fontSize={10} fill={n.grad ? '#be123c' : '#94a3b8'} fontFamily="ui-monospace, monospace">{clip(n.grad ? `grad ${n.grad}` : n.requiresGrad ? (n.leaf ? 'grad None' : t('playground.view.no_grad_kept')) : 'requires_grad=False', 30)}</text>
              </g>
            )
          })}
        </svg>
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-500">
        <span className="inline-flex items-center gap-1"><i className="inline-block h-3 w-3 rounded border border-blue-300 bg-blue-50" />{t('playground.view.leaf')}</span>
        <span className="inline-flex items-center gap-1"><i className="inline-block h-3 w-3 rounded border border-rose-300 bg-rose-50" />{t('playground.view.root')}</span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- training history

const HIST_COLORS: Record<string, string> = { loss: '#2563eb', val_loss: '#ea580c', accuracy: '#059669', val_accuracy: '#d946ef' }
const EXTRA = ['#0ea5e9', '#a855f7', '#f43f5e']

/** one panel of curves sharing a y-axis; best = epoch index to mark */
const Curves = ({ series, best, label }: { series: [string, number[]][]; best: number; label: string }) => {
  const W = 520
  const H = 170
  const L = 40
  const R = 10
  const T = 10
  const B = 22
  const n = Math.max(...series.map(([, v]) => v.length))
  const all = series.flatMap(([, v]) => v)
  const y0 = Math.min(0, ...all)
  const y1 = Math.max(...all) * 1.05 || 1
  const X = (i: number) => L + (n <= 1 ? 0 : (i / (n - 1)) * (W - L - R))
  const Y = (v: number) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg border border-slate-200 bg-white" role="img" aria-label={label}>
      {niceTicks(y0, y1, 4).map((v) => (
        <g key={v}>
          <line x1={L} x2={W - R} y1={Y(v)} y2={Y(v)} stroke="#f1f5f9" />
          <text x={L - 5} y={Y(v) + 3.5} fontSize={10} textAnchor="end" fill="#94a3b8">{+v.toFixed(3)}</text>
        </g>
      ))}
      {niceTicks(1, n, 6).filter((e) => e >= 1 && e <= n && Number.isInteger(e)).map((e) => (
        <text key={e} x={X(e - 1)} y={H - 6} fontSize={10} textAnchor="middle" fill="#94a3b8">{e}</text>
      ))}
      {best >= 0 && n > 1 && <line x1={X(best)} x2={X(best)} y1={T} y2={H - B} stroke="#ea580c" strokeDasharray="4 3" />}
      {series.map(([k, v], i) => (
        <polyline key={k} fill="none" stroke={HIST_COLORS[k] ?? EXTRA[i % EXTRA.length]} strokeWidth={2} points={v.map((y, j) => `${X(j)},${Y(y)}`).join(' ')} />
      ))}
    </svg>
  )
}

const HistoryStep = ({ ev }: { ev: PyEvent & { type: 'history' } }) => {
  const { t } = useTranslation()
  const series = Object.entries(ev.metrics).filter(([, v]) => v.length > 0)
  const n = Math.max(...series.map(([, v]) => v.length))
  const val = ev.metrics.val_loss
  const best = val && val.length ? val.indexOf(Math.min(...val)) : -1
  // losses and scores on separate axes: they live on different scales
  const groups = [series.filter(([k]) => k.includes('loss')), series.filter(([k]) => !k.includes('loss'))].filter((g) => g.length)
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <code className="rounded-lg bg-slate-900 px-2.5 py-1 font-mono text-xs text-slate-100">{ev.code}</code>
        <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-black uppercase text-emerald-700">{ev.title}</span>
      </div>
      <p className="text-sm leading-relaxed text-slate-600">{t('playground.view.history', { n })}</p>
      {groups.map((g, k) => (
        <div key={k} className="space-y-1">
          <Curves series={g} best={k === 0 ? best : -1} label={t('playground.view.history_label')} />
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-600">
            {g.map(([name, v], i) => (
              <span key={name} className="inline-flex items-center gap-1.5 font-mono">
                <i className="inline-block h-0.5 w-4" style={{ background: HIST_COLORS[name] ?? EXTRA[i % EXTRA.length] }} />
                {name} {v.length ? (+v[v.length - 1].toFixed(4)).toString() : ''}
              </span>
            ))}
            {k === 0 && best >= 0 && n > 1 && <span className="text-orange-700">{t('playground.view.best_epoch', { n: best + 1 })}</span>}
          </div>
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------- dispatch

export type StepData = { kind: 'call'; call: CallTrace } | { kind: 'event'; ev: PyEvent }

export const StepView = ({ step }: { step: StepData }) => {
  const { t } = useTranslation()
  if (step.kind === 'call') return <CallView call={step.call} />
  const ev = step.ev
  switch (ev.type) {
    case 'figure':
      return (
        <div className="space-y-2">
          <code className="inline-block rounded-lg bg-slate-900 px-2.5 py-1 font-mono text-xs text-slate-100">{ev.code}</code>
          <p className="text-sm text-slate-600">{t('playground.view.figure_step')}</p>
          <FigureView fig={ev.fig} focus={ev.focus} />
        </div>
      )
    case 'frame': return <FrameStep ev={ev} />
    case 'flow': return <FlowStep ev={ev} />
    case 'graph': return <GraphStep ev={ev} />
    case 'history': return <HistoryStep ev={ev} />
    default: return null
  }
}
