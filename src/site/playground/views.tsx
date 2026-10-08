import { useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { ArrowRight } from 'lucide-react'
import { KIND } from './kinds'
import type { StepKind } from './kinds'
import { useTranslation } from 'react-i18next'
import type { CallTrace } from '../../lib/minipy'
import type { FlowRow, GraphNode, PyEvent, TableSpec } from '../../lib/pyEvents'
import { niceTicks } from '../../lib/chart'
import { CallView } from '../../modules/NpCallView'
import { FigureView } from './FigureView'

// --- 实验台每一步的可视化：调用网格、图、DataFrame、模型形状流、自动求导计算图、训练曲线 ---

/** the line of code, a coloured badge for the kind of step, and a sentence about what is shown */
const StepHead = ({ kind, code, badge, children }: { kind: StepKind; code: string; badge: string; children?: ReactNode }) => {
  const K = KIND[kind]
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <code className="max-w-full overflow-x-auto whitespace-pre rounded-lg bg-slate-900 px-2.5 py-1 font-mono text-xs text-slate-100">{code}</code>
        <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-black uppercase tracking-wide ${K.badge}`}>
          <K.icon size={11} aria-hidden /> {badge}
        </span>
      </div>
      {children && <p className="text-sm leading-relaxed text-slate-600">{children}</p>}
    </div>
  )
}

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
  // with both rows and columns picked (loc / iloc), the cells where they meet are the result; the rest of the row / column is context
  const both = !!rows && !!cols
  const cellLook = (r: number, c: number): string | null => {
    const inR = !!rows?.has(r)
    const inC = !!cols?.has(c)
    if (both) return inR && inC ? 'bg-amber-300 font-bold text-amber-950' : inR || inC ? 'bg-amber-50 text-amber-900' : null
    return inR || inC ? 'bg-amber-100 font-bold text-amber-900' : null
  }
  return (
    <div className="min-w-0">
      {title && <div className="mb-1 font-mono text-xs font-bold text-slate-600">{title}</div>}
      <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
        <table className="min-w-full border-collapse font-mono text-xs">
          <thead>
            <tr className="bg-slate-100/80">
              <th className="sticky left-0 border-b border-slate-200 bg-slate-100 px-2 py-1.5 text-left font-normal italic text-slate-500">{table.indexName ?? ''}</th>
              {colNames.map((c, k) => (
                <th key={k} className={`border-b border-slate-200 px-2 py-1.5 text-right font-bold ${cols?.has(k) ? 'bg-amber-100 text-amber-900' : 'text-slate-700'}`}>
                  {c}
                  {table.dtypes[k] && <span className="block text-[9px] font-normal text-slate-500">{table.dtypes[k]}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r} style={{ background: rowTint?.(r) }} className={`${rows?.has(r) ? 'bg-amber-50' : 'even:bg-slate-50/70'} hover:bg-blue-50/60`}>
                <th className={`sticky left-0 border-b border-slate-100 bg-inherit px-2 py-1 text-left font-bold ${rows?.has(r) ? 'text-amber-800' : 'text-slate-500'}`}>{table.index[r]}</th>
                {table.cells[r].map((v, c) => (
                  <td key={c} className={`border-b border-slate-100 px-2 py-1 text-right ${v === null ? 'italic text-rose-700' : cellLook(r, c) ?? 'text-slate-700'}`}>
                    {v ?? 'NaN'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-1 text-[10px] text-slate-500">
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
      <StepHead kind="frame" code={ev.code} badge={ev.api}>{t(`playground.frame.${ev.api}`, { defaultValue: t('playground.frame.generic') })}</StepHead>
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
        {ev.result ? (
          <TableView table={ev.result} title={t('playground.view.result')} />
        ) : (
          <div className="min-w-0">
            <div className="mb-1 font-mono text-xs font-bold text-slate-600">{t('playground.view.result')}</div>
            <pre className="overflow-x-auto rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2 font-mono text-sm font-bold text-emerald-900 shadow-sm">{ev.resultText}</pre>
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- model shape flow

type BlockKind = 'input' | 'conv' | 'pool' | 'dense' | 'flat' | 'other'
const BLOCK_COLOR: Record<BlockKind, { face: string; side: string; top: string; ink: string }> = {
  input: { face: '#e2e8f0', side: '#cbd5e1', top: '#f1f5f9', ink: '#475569' },
  conv: { face: '#c4b5fd', side: '#a78bfa', top: '#ddd6fe', ink: '#6d28d9' },
  pool: { face: '#7dd3fc', side: '#38bdf8', top: '#bae6fd', ink: '#0369a1' },
  dense: { face: '#fcd34d', side: '#f59e0b', top: '#fde68a', ink: '#b45309' },
  flat: { face: '#cbd5e1', side: '#94a3b8', top: '#e2e8f0', ink: '#475569' },
  other: { face: '#a7f3d0', side: '#34d399', top: '#d1fae5', ink: '#047857' },
}
const blockKind = (layer: string): BlockKind =>
  /conv/i.test(layer) ? 'conv' : /pool/i.test(layer) ? 'pool' : /linear|dense/i.test(layer) ? 'dense' : /flatten|view|reshape/i.test(layer) ? 'flat' : 'other'
const short = (n: number) => (n >= 1e6 ? `${+(n / 1e6).toFixed(2)}M` : n >= 1e4 ? `${+(n / 1e3).toFixed(1)}k` : n.toLocaleString('en-US'))
const sameShape = (a: (number | null)[], b: (number | null)[]) => a.length === b.length && a.every((d, i) => d === b[i])
/** activations and dropout: same shape in and out, nothing to learn; drawn as a tag on the arrow instead of a block */
const isTag = (r: FlowRow) => r.params === 0 && sameShape(r.input, r.output) && !/pool|flatten|view|reshape/i.test(r.layer)
const tagName = (r: FlowRow) => (r.layer.match(/^(?:F\.)?([A-Za-z]+)/)?.[1] ?? r.name).replace(/^relu$/, 'ReLU')

/** the network as a row of blocks: feature maps as boxes (height ∝ size, depth ∝ channels), vectors as bars */
const LayerStack = ({ ev, active, onActive }: { ev: PyEvent & { type: 'flow' }; active: number | null; onActive: (k: number | null) => void }) => {
  const { t } = useTranslation()
  const torch = ev.framework === 'torch'
  type B = { row: number; kind: BlockKind; shape: (number | null)[]; name: string; params: number; tags: string[] }
  const blocks: B[] = []
  if (ev.rows.length) blocks.push({ row: -1, kind: 'input', shape: ev.rows[0].input, name: t('playground.view.input'), params: 0, tags: [] })
  ev.rows.forEach((r, k) => {
    if (isTag(r) && blocks.length) blocks[blocks.length - 1].tags.push(tagName(r))
    else blocks.push({ row: k, kind: blockKind(r.layer), shape: r.output, name: r.name, params: r.params, tags: [] })
  })
  const geo = blocks.map((b) => {
    const d = b.shape.slice(1).map((x) => x ?? 1)
    if (d.length === 3) {
      const [h, w, c] = torch ? [d[1], d[2], d[0]] : [d[0], d[1], d[2]]
      const H = Math.max(26, Math.min(112, 14 + 15 * Math.log2(Math.max(h, 1))))
      return { kind: 'box' as const, H, D: Math.max(6, Math.min(54, 4 + 7 * Math.log2(c + 1))), label: torch ? `${c}@${h}×${w}` : `${h}×${w}×${c}`, wDepth: H * 0.38, wh: w }
    }
    const f = d.reduce((a, x) => a * x, 1)
    return { kind: 'bar' as const, H: Math.max(22, Math.min(124, 10 + 10 * Math.log2(Math.max(f, 1)))), D: 12, label: d.join('×'), wDepth: 6, wh: f }
  })
  const GAP = 46
  const TOP = 22
  const maxH = Math.max(...geo.map((g) => g.H + g.wDepth))
  const base = TOP + maxH
  let x = 12
  const xs = geo.map((g) => {
    const at = x
    x += g.D + g.wDepth + Math.max(GAP, g.label.length * 6.4 - g.D - g.wDepth + 14)
    return at
  })
  const W = x - GAP + 24
  const Hsvg = base + 46
  return (
    <div className="overflow-x-auto rounded-xl border border-slate-200 bg-gradient-to-b from-white to-slate-50 p-2">
      <svg width={W} height={Hsvg} viewBox={`0 0 ${W} ${Hsvg}`} role="img" aria-label={t('playground.view.stack_label')} className="block">
        {blocks.map((b, i) => {
          const g = geo[i]
          const c = BLOCK_COLOR[b.kind]
          const x0 = xs[i]
          const yTop = base - g.H
          const on = active !== null && b.row === active
          const dim = active !== null && !on
          const next = i + 1 < blocks.length ? xs[i + 1] : null
          return (
            <g key={i} opacity={dim ? 0.45 : 1} onMouseEnter={() => b.row >= 0 && onActive(b.row)} onMouseLeave={() => onActive(null)} style={{ transition: 'opacity .15s' }}>
              <title>{`${b.name}\n${shape(b.shape)}${b.params ? `\n${b.params.toLocaleString('en-US')} params` : ''}`}</title>
              {/* top and side faces, then the front */}
              <path d={`M${x0},${yTop} l${g.wDepth},${-g.wDepth * 0.6} h${g.D} l${-g.wDepth},${g.wDepth * 0.6} z`} fill={c.top} stroke={c.side} strokeWidth={1} />
              <path d={`M${x0 + g.D},${yTop} l${g.wDepth},${-g.wDepth * 0.6} v${g.H} l${-g.wDepth},${g.wDepth * 0.6} z`} fill={c.side} stroke={c.side} strokeWidth={1} />
              <rect x={x0} y={yTop} width={g.D} height={g.H} fill={c.face} stroke={on ? '#0f172a' : c.side} strokeWidth={on ? 2 : 1} />
              <text x={x0 + (g.D + g.wDepth) / 2} y={base + 16} textAnchor="middle" fontSize={11} fontWeight={700} fill="#0f172a" fontFamily="ui-monospace, monospace">{g.label}</text>
              <text x={x0 + (g.D + g.wDepth) / 2} y={base + 30} textAnchor="middle" fontSize={10} fill={c.ink} fontFamily="ui-monospace, monospace">{clip(b.name, 14)}</text>
              {b.params > 0 && <text x={x0 + (g.D + g.wDepth) / 2} y={base + 42} textAnchor="middle" fontSize={9.5} fill="#64748b" fontFamily="ui-monospace, monospace">{short(b.params)} p</text>}
              {next !== null && (
                <g>
                  <line x1={x0 + g.D + g.wDepth + 4} x2={next - 6} y1={base - 10} y2={base - 10} stroke="#94a3b8" strokeWidth={1.4} markerEnd="url(#flowArrow)" />
                  {b.tags.length > 0 && (
                    <text x={(x0 + g.D + g.wDepth + next) / 2} y={base - 16} textAnchor="middle" fontSize={9.5} fontWeight={700} fill="#047857" fontFamily="ui-monospace, monospace">{clip(b.tags.join('·'), 16)}</text>
                  )}
                </g>
              )}
            </g>
          )
        })}
        <defs>
          <marker id="flowArrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L8,4L0,8z" fill="#94a3b8" /></marker>
        </defs>
      </svg>
    </div>
  )
}

const FlowStep = ({ ev }: { ev: PyEvent & { type: 'flow' } }) => {
  const { t } = useTranslation()
  const [active, setActive] = useState<number | null>(null)
  const sizes = ev.rows.map((r) => r.output.reduce<number>((a, b) => a * (b ?? 1), 1))
  const maxLog = Math.log10(Math.max(10, ...sizes))
  const maxParams = Math.max(1, ...ev.rows.map((r) => r.params))
  return (
    <div className="space-y-3">
      <StepHead kind="flow" code={ev.code} badge={ev.title}>{t(`playground.view.flow_${ev.framework}`)}</StepHead>
      <LayerStack ev={ev} active={active} onActive={setActive} />
      <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
        <table className="min-w-full border-collapse text-xs">
          <thead className="bg-slate-100/80 text-left text-[11px] text-slate-600">
            <tr>
              <th className="px-2.5 py-1.5">{t('playground.view.layer')}</th>
              <th className="px-2.5 py-1.5">{t('playground.view.in_out')}</th>
              <th className="px-2.5 py-1.5 text-right">{t('playground.view.params')}</th>
            </tr>
          </thead>
          <tbody>
            {ev.rows.map((r: FlowRow, k) => (
              <tr key={k} onMouseEnter={() => setActive(k)} onMouseLeave={() => setActive(null)} className={`border-t border-slate-100 align-top transition-colors ${active === k ? 'bg-violet-50' : isTag(r) ? 'bg-slate-50/60' : ''}`}>
                <td className="px-2.5 py-1.5">
                  <div className="flex items-center gap-1.5 font-mono font-bold text-slate-800">
                    <i className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: isTag(r) ? '#a7f3d0' : BLOCK_COLOR[blockKind(r.layer)].face }} aria-hidden />
                    {r.name}
                  </div>
                  <div className="font-mono text-[11px] text-slate-500">{r.layer}</div>
                </td>
                <td className="px-2.5 py-1.5">
                  <div className="whitespace-nowrap font-mono text-slate-700">{shape(r.input)} → <b className="text-violet-700">{shape(r.output)}</b></div>
                  <div className="mt-1 h-1.5 rounded-full bg-violet-100">
                    <div className="h-1.5 rounded-full bg-gradient-to-r from-violet-400 to-violet-600" style={{ width: `${Math.max(4, (Math.log10(Math.max(1, sizes[k])) / maxLog) * 100)}%` }} />
                  </div>
                  {r.note && <div className="mt-1 font-mono text-[11px] text-slate-500">{r.note}</div>}
                </td>
                <td className="px-2.5 py-1.5 text-right font-mono text-slate-700">
                  {r.params.toLocaleString('en-US')}
                  {r.params > 0 && (
                    <div className="ml-auto mt-1 h-1 w-16 rounded-full bg-amber-100">
                      <div className="h-1 rounded-full bg-amber-500" style={{ width: `${Math.max(6, (r.params / maxParams) * 100)}%` }} />
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot className="border-t border-slate-200 bg-slate-50 font-mono text-[11px] text-slate-600">
            <tr>
              <td className="px-2.5 py-1.5" colSpan={2}>{t('playground.view.total_params')}</td>
              <td className="px-2.5 py-1.5 text-right font-bold">{ev.total.toLocaleString('en-US')}</td>
            </tr>
            {ev.trainable !== ev.total && (
              <tr>
                <td className="px-2.5 py-1" colSpan={2}>{t('playground.view.trainable')}</td>
                <td className="px-2.5 py-1 text-right">{ev.trainable.toLocaleString('en-US')}</td>
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

const nodeRole = (n: GraphNode, root: number) => (n.id === root ? 'root' : n.leaf ? (n.requiresGrad ? 'param' : 'data') : 'op')
const ROLE = {
  root: { band: '#e11d48', body: '#fff1f2', stroke: '#fda4af' },
  param: { band: '#2563eb', body: '#eff6ff', stroke: '#93c5fd' },
  data: { band: '#64748b', body: '#f8fafc', stroke: '#cbd5e1' },
  op: { band: '#334155', body: '#ffffff', stroke: '#cbd5e1' },
} as const

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
    // fewer crossings: order each column by the mean row of the nodes feeding it (leaves by the first node they feed)
    const rowOf = new Map<number, number>()
    for (let c = 1; c < cols; c++) {
      perCol[c - 1].forEach((id, r) => rowOf.set(id, r))
      const key = (id: number) => {
        const ins = byId.get(id)!.inputs.filter((i) => rowOf.has(i))
        return ins.length ? ins.reduce((a, i) => a + rowOf.get(i)!, 0) / ins.length : 0
      }
      perCol[c].sort((a, b) => key(a) - key(b))
    }
    // top to bottom: leaves in the first row, the loss at the bottom, so the graph fits a narrow panel
    const W = 168
    const H = 76
    const GX = 16
    const GY = 44
    const most = Math.max(...perCol.map((p) => p.length))
    const width = 24 + most * (W + GX) - GX
    const pos = new Map<number, { x: number; y: number }>()
    perCol.forEach((ids, c) => ids.forEach((id, r) => pos.set(id, { x: 12 + r * (W + GX) + ((most - ids.length) * (W + GX)) / 2, y: 12 + c * (H + GY) })))
    return { pos, W, H, width, height: 24 + cols * (H + GY) - GY, byId }
  }, [ev.nodes])
  const { pos, W, H } = layout
  const active = hover === null ? null : new Set([hover, ...(layout.byId.get(hover)?.inputs ?? [])])
  const backward = ev.nodes.some((n) => n.grad !== null)
  return (
    <div className="space-y-3">
      <StepHead kind="graph" code={ev.code} badge={ev.title}>{t('playground.view.graph')}</StepHead>
      <div className="overflow-x-auto rounded-xl border border-slate-200 bg-[radial-gradient(#e2e8f0_1px,transparent_1px)] bg-[length:16px_16px] p-1">
        <svg viewBox={`0 0 ${layout.width} ${layout.height}`} className="mx-auto block" style={{ width: '100%', maxWidth: layout.width, minWidth: layout.width * 0.8 }} role="img" aria-label={t('playground.view.graph_label')}>
          <defs>
            <marker id="gArrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0L8,4L0,8z" fill="#94a3b8" /></marker>
            <marker id="gArrowOn" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0L8,4L0,8z" fill="#e11d48" /></marker>
          </defs>
          {ev.nodes.flatMap((n) => n.inputs.map((i) => {
            const a = pos.get(i)!
            const b = pos.get(n.id)!
            const on = active?.has(n.id) && active.has(i)
            const d = `M${a.x + W / 2},${a.y + H} C${a.x + W / 2},${a.y + H + 26} ${b.x + W / 2},${b.y - 26} ${b.x + W / 2},${b.y - 1}`
            const gradEdge = backward && layout.byId.get(i)?.requiresGrad
            return (
              <g key={`${i}-${n.id}`}>
                <path d={d} fill="none" stroke={on ? '#e11d48' : '#94a3b8'} strokeWidth={on ? 2.2 : 1.4} markerEnd={on ? 'url(#gArrowOn)' : 'url(#gArrow)'} />
                {/* gradients flow back along the edges into tensors that require grad */}
                {gradEdge && <path d={d} fill="none" stroke="#fb7185" strokeWidth={2} strokeDasharray="3 9" className="grad-flow" opacity={0.85} />}
              </g>
            )
          }))}
          {ev.nodes.map((n: GraphNode) => {
            const p = pos.get(n.id)!
            const role = nodeRole(n, ev.root)
            const c = ROLE[role]
            const hot = hover === n.id
            return (
              <g key={n.id} onMouseEnter={() => setHover(n.id)} onMouseLeave={() => setHover(null)} transform={`translate(${p.x} ${p.y})`} style={{ cursor: 'default' }}>
                <title>{[n.label, n.op, `${t('playground.view.value')} ${n.value}`, n.grad ? `grad ${n.grad}` : ''].filter(Boolean).join('\n')}</title>
                <rect width={W} height={H} rx={12} fill={c.body} stroke={hot ? '#e11d48' : c.stroke} strokeWidth={hot ? 2 : 1} filter="drop-shadow(0 1px 1.5px rgb(15 23 42 / 0.08))" />
                <path d={`M0,12 a12,12 0 0 1 12,-12 h${W - 24} a12,12 0 0 1 12,12 v10 h${-W} z`} fill={c.band} />
                <text x={10} y={15.5} fontSize={12} fontWeight={700} fill="#ffffff" fontFamily="ui-monospace, monospace">{clip(n.label, 11)}</text>
                <text x={W - 10} y={15.5} fontSize={10} textAnchor="end" fill="#e2e8f0" fontFamily="ui-monospace, monospace">
                  {n.op ? clip(n.op.replace(/Backward\d*$/, ''), 9) : n.shape.length ? clip(`(${n.shape.join(', ')})`, 10) : t(`playground.view.role_${role}`)}
                </text>
                <text x={10} y={42} fontSize={11} fill="#334155" fontFamily="ui-monospace, monospace">{clip(n.value, 22)}</text>
                {n.grad ? (
                  <g>
                    <rect x={8} y={51} width={Math.min(W - 16, 16 + 6.6 * Math.min(21, n.grad.length + 2))} height={17} rx={8.5} fill="#ffe4e6" />
                    <text x={16} y={63.5} fontSize={10.5} fontWeight={700} fill="#be123c" fontFamily="ui-monospace, monospace">{clip(`∂ ${n.grad}`, 21)}</text>
                  </g>
                ) : (
                  <text x={10} y={63} fontSize={10} fill="#64748b" fontFamily="ui-monospace, monospace">{clip(n.requiresGrad ? (n.leaf ? 'grad None' : t('playground.view.no_grad_kept')) : 'requires_grad=False', 24)}</text>
                )}
              </g>
            )
          })}
        </svg>
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-600">
        <span className="inline-flex items-center gap-1.5"><i className="inline-block h-3 w-3 rounded" style={{ background: ROLE.param.band }} />{t('playground.view.leaf')}</span>
        <span className="inline-flex items-center gap-1.5"><i className="inline-block h-3 w-3 rounded" style={{ background: ROLE.root.band }} />{t('playground.view.root')}</span>
        {backward && <span className="inline-flex items-center gap-1.5"><i className="inline-block h-0.5 w-4 border-t-2 border-dashed border-rose-400" />{t('playground.view.grad_flow')}</span>}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- training history

const HIST_COLORS: Record<string, string> = { loss: '#2563eb', val_loss: '#ea580c', accuracy: '#059669', val_accuracy: '#d946ef' }
const EXTRA = ['#0ea5e9', '#a855f7', '#f43f5e']
const colorOf = (k: string, i: number) => HIST_COLORS[k] ?? EXTRA[i % EXTRA.length]
const fmt = (v: number) => (+v.toFixed(4)).toString()

/** one panel of curves sharing a y-axis; best = epoch index to mark; hover shows every value at that epoch */
const Curves = ({ series, best, label, id }: { series: [string, number[]][]; best: number; label: string; id: string }) => {
  const { t } = useTranslation()
  const [at, setAt] = useState<number | null>(null)
  const W = 520
  const H = 180
  const L = 40
  const R = 12
  const T = 12
  const B = 24
  const n = Math.max(...series.map(([, v]) => v.length))
  const all = series.flatMap(([, v]) => v)
  const y0 = Math.min(0, ...all)
  const y1 = Math.max(...all) * 1.05 || 1
  const X = (i: number) => L + (n <= 1 ? 0 : (i / (n - 1)) * (W - L - R))
  const Y = (v: number) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B)
  const move = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const sx = ((e.clientX - r.left) / r.width) * W
    setAt(n <= 1 ? 0 : Math.max(0, Math.min(n - 1, Math.round(((sx - L) / (W - L - R)) * (n - 1)))))
  }
  const tipX = at === null ? 0 : Math.min(X(at) + 8, W - R - 132)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-xl border border-slate-200 bg-white" role="img" aria-label={label} onMouseMove={move} onMouseLeave={() => setAt(null)}>
      <defs>
        {series.map(([k], i) => (
          <linearGradient key={k} id={`${id}-${k}`} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor={colorOf(k, i)} stopOpacity={0.22} />
            <stop offset="1" stopColor={colorOf(k, i)} stopOpacity={0} />
          </linearGradient>
        ))}
      </defs>
      {niceTicks(y0, y1, 4).filter((v) => v >= y0 - 1e-9 && v <= y1 + 1e-9).map((v) => (
        <g key={v}>
          <line x1={L} x2={W - R} y1={Y(v)} y2={Y(v)} stroke="#eef2f7" />
          <text x={L - 6} y={Y(v) + 3.5} fontSize={10} textAnchor="end" fill="#64748b">{+v.toFixed(3)}</text>
        </g>
      ))}
      {niceTicks(1, n, 6).filter((e) => e >= 1 && e <= n && Number.isInteger(e)).map((e) => (
        <text key={e} x={X(e - 1)} y={H - 7} fontSize={10} textAnchor="middle" fill="#64748b">{e}</text>
      ))}
      {best >= 0 && n > 1 && <line x1={X(best)} x2={X(best)} y1={T} y2={H - B} stroke="#ea580c" strokeDasharray="4 3" />}
      {series.map(([k, v]) => v.length > 1 && (
        <path key={`a${k}`} d={`M${X(0)},${Y(y0)} ${v.map((y, j) => `L${X(j)},${Y(y)}`).join(' ')} L${X(v.length - 1)},${Y(y0)} Z`} fill={`url(#${id}-${k})`} />
      ))}
      {series.map(([k, v], i) => (
        <g key={k}>
          <polyline fill="none" stroke={colorOf(k, i)} strokeWidth={2.2} strokeLinejoin="round" strokeLinecap="round" points={v.map((y, j) => `${X(j)},${Y(y)}`).join(' ')} />
          {v.length > 0 && <circle cx={X(v.length - 1)} cy={Y(v[v.length - 1])} r={3.5} fill="#fff" stroke={colorOf(k, i)} strokeWidth={2} />}
        </g>
      ))}
      {at !== null && (
        <g pointerEvents="none">
          <line x1={X(at)} x2={X(at)} y1={T} y2={H - B} stroke="#0f172a" strokeOpacity={0.25} />
          {series.map(([k, v], i) => v[at] !== undefined && <circle key={k} cx={X(at)} cy={Y(v[at])} r={4} fill={colorOf(k, i)} stroke="#fff" strokeWidth={1.5} />)}
          <rect x={tipX} y={T} width={124} height={18 + 14 * series.length} rx={8} fill="#0f172a" fillOpacity={0.92} />
          <text x={tipX + 10} y={T + 14} fontSize={10.5} fontWeight={700} fill="#fff">{t('playground.view.epoch', { n: at + 1 })}</text>
          {series.map(([k, v], i) => (
            <text key={k} x={tipX + 10} y={T + 28 + 14 * i} fontSize={10} fill={colorOf(k, i)} fontFamily="ui-monospace, monospace" style={{ filter: 'brightness(1.6)' }}>
              {`${clip(k, 12)} ${v[at] !== undefined ? fmt(v[at]) : '–'}`}
            </text>
          ))}
        </g>
      )}
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
      <StepHead kind="history" code={ev.code} badge={ev.title}>{t('playground.view.history', { n })}</StepHead>
      {groups.map((g, k) => (
        <div key={k} className="space-y-1.5">
          <Curves series={g} best={k === 0 ? best : -1} label={t('playground.view.history_label')} id={`h${ev.id}-${k}`} />
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-600">
            {g.map(([name, v], i) => (
              <span key={name} className="inline-flex items-center gap-1.5 font-mono">
                <i className="inline-block h-1 w-4 rounded-full" style={{ background: colorOf(name, i) }} />
                {name} <b className="text-slate-800">{v.length ? fmt(v[v.length - 1]) : ''}</b>
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
        <div className="space-y-3">
          <StepHead kind="figure" code={ev.code} badge="pyplot">{t('playground.view.figure_step')}</StepHead>
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
