import { useState } from 'react'
import type { ReactNode } from 'react'
import { X } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { broadcastStrides, isRealCell, shapeLabel, sourceIndex } from '../lib/broadcast'
import type { AxisStep } from '../lib/broadcast'
import { MAX_DIM, MAX_NDIM, TONES, apply, flatOf, opSign, showCell, size, valA, valB } from '../lib/broadcastDemo'
import type { Op } from '../lib/broadcastDemo'

// --- The broadcasting lab's pieces: the shape editor, the axis-by-axis rule table, the operands and the stretched
// grids. The page that puts them in order is src/site/modules/NumpyBroadcasting.tsx. ---

// ---------- 形状编辑器 ----------

interface ShapeEditorProps {
  name: 'A' | 'B'
  /** the variable's own name, when it is not simply A or B */
  label?: string
  shape: number[]
  onChange: (s: number[]) => void
}

export const ShapeEditor = ({ name, label, shape, onChange }: ShapeEditorProps) => {
  const { t } = useTranslation()
  const tone = TONES[name]
  return (
    <div className={`rounded-xl border p-3 ${tone.chip}`}>
      <div className="flex items-baseline justify-between mb-2">
        <span className={`text-sm font-bold ${tone.text}`}>{t('numpy_module.broadcast.shape_of', { name: label ?? name })}</span>
        <span className="font-mono text-xs text-gray-500">{shapeLabel(shape)}</span>
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="font-mono text-gray-400">(</span>
        {shape.length === 0 && <span className="text-xs text-gray-400 italic">{t('numpy_module.broadcast.scalar')}</span>}
        {shape.map((d, k) => (
          <span key={k} className="inline-flex items-center rounded-md bg-white border border-gray-200 shadow-sm">
            <select
              value={d}
              onChange={(e) => onChange(shape.map((x, i) => (i === k ? Number(e.target.value) : x)))}
              aria-label={t('numpy_module.broadcast.dim_label', { name, axis: k })}
              className="bg-transparent font-mono text-sm pl-2 pr-1 py-1 outline-none cursor-pointer"
            >
              {Array.from({ length: MAX_DIM }, (_, i) => i + 1).map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
            <button
              onClick={() => onChange(shape.filter((_, i) => i !== k))}
              aria-label={t('numpy_module.broadcast.remove_axis', { name, axis: k })}
              className="px-1 text-gray-300 hover:text-rose-500"
            >
              <X size={12} />
            </button>
          </span>
        ))}
        <span className="font-mono text-gray-400">)</span>
        <button
          onClick={() => onChange([2, ...shape])}
          disabled={shape.length >= MAX_NDIM}
          className="ml-1 px-2 py-1 rounded-md text-xs border border-dashed border-gray-300 text-gray-500 hover:border-gray-500 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {t('numpy_module.broadcast.add_axis')}
        </button>
      </div>
    </div>
  )
}

// ---------- 网格（≤3 维，3 维时按第 0 轴拆块） ----------

export interface CellInfo {
  cls: string
  text: string
  ring?: 'hover' | 'source'
}

interface GridProps {
  title: ReactNode
  shape: number[]
  cell: (idx: number[]) => CellInfo
  onHover?: (idx: number[] | null) => void
}

export const Grid = ({ title, shape, cell, onHover }: GridProps) => {
  const n = shape.length
  const rows = n >= 2 ? shape[n - 2] : 1
  const cols = n >= 1 ? shape[n - 1] : 1
  const blocks = n === 3 ? shape[0] : 1
  return (
    <div className="flex flex-col items-center gap-1.5">
      <div className="text-[11px] text-gray-500 text-center leading-tight">{title}</div>
      <div className="space-y-2" onMouseLeave={() => onHover?.(null)}>
        {Array.from({ length: blocks }, (_, b) => (
          <div key={b}>
            {n === 3 && <div className="text-[10px] font-mono text-gray-400 mb-0.5">[{b}, :, :]</div>}
            <div className="inline-grid gap-1" style={{ gridTemplateColumns: `repeat(${cols}, 2.25rem)` }}>
              {Array.from({ length: rows * cols }, (_, i) => {
                const r = Math.floor(i / cols)
                const c = i % cols
                const idx = n === 0 ? [] : n === 1 ? [c] : n === 2 ? [r, c] : [b, r, c]
                const info = cell(idx)
                const ring = info.ring === 'hover' ? 'ring-2 ring-rose-500 ring-offset-1 z-10' : info.ring === 'source' ? 'ring-2 ring-slate-800 ring-offset-1 z-10' : ''
                return (
                  <div
                    key={i}
                    onMouseEnter={() => onHover?.(idx)}
                    onClick={() => onHover?.(idx)}
                    className={`h-9 flex items-center justify-center rounded border font-mono text-[11px] transition-colors cursor-default ${info.cls} ${ring}`}
                  >
                    {info.text}
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export const OpSign = ({ children }: { children: ReactNode }) => (
  <div className="text-2xl font-bold text-gray-300 self-center px-1">{children}</div>
)

// ---------- 规则逐轴表 ----------

interface RuleTableProps {
  a: number[]
  b: number[]
  nameA?: string
  nameB?: string
  steps: AxisStep[]
  /** 0 = 右对齐补 1；k = 已检查到 axis -k；>ndim = 全部完成 */
  phase: number
}

const VERDICT_CLS: Record<AxisStep['verdict'], string> = {
  equal: 'bg-gray-100 text-gray-600',
  stretchA: 'bg-blue-100 text-blue-700',
  stretchB: 'bg-orange-100 text-orange-700',
  mismatch: 'bg-rose-100 text-rose-700',
}

export const RuleTable = ({ a, b, nameA = 'A', nameB = 'B', steps, phase }: RuleTableProps) => {
  const { t } = useTranslation()
  const cols = [...steps].reverse() // 左 → 右：axis -ndim … -1
  const failAt = steps.findIndex((s) => s.verdict === 'mismatch')
  const checked = (s: AxisStep) => -s.axis <= phase && (failAt < 0 || -s.axis <= failAt + 1)
  const current = (s: AxisStep) => -s.axis === phase
  const dimCell = (v: number, pad: boolean, tone: 'A' | 'B', s: AxisStep) => (
    <td key={s.axis} className={`px-2 py-1.5 text-center font-mono ${current(s) ? 'bg-yellow-50' : ''}`}>
      {pad ? (
        <span className="text-gray-400" title={t('numpy_module.broadcast.padded')}>1<sup className="text-[9px] text-gray-400">+</sup></span>
      ) : (
        <span className={`font-semibold ${TONES[tone].text}`}>{v}</span>
      )}
    </td>
  )
  return (
    <div className="overflow-x-auto">
      <table className="text-sm border-separate border-spacing-0 mx-auto">
        <thead>
          <tr className="text-[11px] text-gray-400">
            <th className="px-2 py-1 text-left font-normal" />
            {cols.map((s) => (
              <th key={s.axis} className={`px-2 py-1 font-mono font-normal ${current(s) ? 'bg-yellow-50 rounded-t-md text-yellow-700' : ''}`}>
                {t('numpy_module.broadcast.axis', { axis: s.axis })}
              </th>
            ))}
            <th className="px-2 py-1 font-normal text-left">{t('numpy_module.broadcast.shape')}</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className={`px-2 py-1.5 text-xs font-bold ${TONES.A.text}`}>{nameA}</td>
            {cols.map((s) => dimCell(s.a, s.padA, 'A', s))}
            <td className="px-2 font-mono text-xs text-gray-500">{shapeLabel(a)}</td>
          </tr>
          <tr>
            <td className={`px-2 py-1.5 text-xs font-bold ${TONES.B.text}`}>{nameB}</td>
            {cols.map((s) => dimCell(s.b, s.padB, 'B', s))}
            <td className="px-2 font-mono text-xs text-gray-500">{shapeLabel(b)}</td>
          </tr>
          <tr className="border-t">
            <td className="px-2 py-1.5 text-xs font-bold text-gray-500 border-t border-gray-200">{t('numpy_module.broadcast.row_check')}</td>
            {cols.map((s) => (
              <td key={s.axis} className={`px-1 py-1.5 text-center border-t border-gray-200 ${current(s) ? 'bg-yellow-50' : ''}`}>
                {checked(s) ? (
                  <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold whitespace-nowrap ${VERDICT_CLS[s.verdict]}`}>
                    {t(`numpy_module.broadcast.verdict.${s.verdict}`)}
                  </span>
                ) : <span className="text-gray-300">·</span>}
              </td>
            ))}
            <td className="border-t border-gray-200" />
          </tr>
          <tr>
            <td className="px-2 py-1.5 text-xs font-bold text-emerald-600">{nameA} {'∘'} {nameB}</td>
            {cols.map((s) => (
              <td key={s.axis} className={`px-2 py-1.5 text-center font-mono font-bold ${current(s) ? 'bg-yellow-50 rounded-b-md' : ''}`}>
                {checked(s) ? (s.out === null ? <span className="text-rose-500">✗</span> : <span className="text-emerald-600">{s.out}</span>) : <span className="text-gray-300">?</span>}
              </td>
            ))}
            <td className="px-2 font-mono text-xs text-gray-500">
              {phase > steps.length || (failAt >= 0 && phase > failAt) ? (failAt >= 0 ? '✗' : shapeLabel(steps.map((s) => s.out!).reverse())) : ''}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}

/** real data for the two sides (flat, row by row) and what the code calls them; without it, the demo's generated values */
export interface OperandData {
  a: number[]
  b: number[]
  nameA: string
  nameB: string
}
const reader = (data: OperandData | undefined) => ({
  a: (idx: number[], shape: number[]) => (data ? data.a[flatOf(idx, shape)] : valA(idx, shape)),
  b: (idx: number[], shape: number[]) => (data ? data.b[flatOf(idx, shape)] : valB(idx, shape)),
  nameA: data?.nameA ?? 'A',
  nameB: data?.nameB ?? 'B',
})

// ---------- the stretch: A and B drawn in the result's shape, ghost copies dashed, hover any cell ----------

/** withResult = false stops at the stretch: the two operands in the result's shape, not yet combined */
export const StretchView = ({ a, b, op, out, withResult = true, data }: { a: number[]; b: number[]; op: Op; out: number[]; withResult?: boolean; data?: OperandData }) => {
  const { t } = useTranslation()
  const read = reader(data)
  const [hover, setHover] = useState<number[] | null>(null)
  const [showGhost, setShowGhost] = useState(true)
  const sign = opSign(op)
  const idxStr = (idx: number[]) => (idx.length ? `[${idx.join(', ')}]` : '')
  // where the hovered cell really lives in each operand (in result coordinates)
  const realPos = (idx: number[], shape: number[]) => {
    const pad = idx.length - shape.length
    return idx.map((v, k) => (k < pad || shape[k - pad] === 1 ? 0 : v))
  }
  const same = (x: number[] | null, y: number[]) => !!x && x.length === y.length && x.every((v, i) => v === y[i])
  const operandCell = (name: 'A' | 'B', shape: number[]) => (idx: number[]): CellInfo => {
    const real = isRealCell(idx, shape)
    const v = showCell(name === 'A' ? read.a(sourceIndex(idx, shape), shape) : read.b(sourceIndex(idx, shape), shape))
    const tone = TONES[name]
    const ring = same(hover, idx) ? 'hover' : hover && same(realPos(hover, shape), idx) ? 'source' : undefined
    if (real) return { cls: tone.real, text: v, ring }
    return { cls: showGhost ? tone.ghost : 'bg-white border-dashed border-gray-200 text-transparent', text: showGhost ? v : '', ring }
  }
  const resultCell = (idx: number[]): CellInfo => {
    const v = apply(op, read.a(sourceIndex(idx, a), a), read.b(sourceIndex(idx, b), b))
    return { cls: TONES.R.real, text: showCell(v), ring: same(hover, idx) ? 'hover' : undefined }
  }
  const formula = hover
    ? (() => {
        const ia = sourceIndex(hover, a)
        const ib = sourceIndex(hover, b)
        const x = read.a(ia, a)
        const y = read.b(ib, b)
        return `${read.nameA}${idxStr(ia)} ${sign} ${read.nameB}${idxStr(ib)} = ${showCell(x)} ${sign} ${showCell(y)} = ${showCell(apply(op, x, y))}`
      })()
    : null
  const stridesA = broadcastStrides(a, out)
  const stridesB = broadcastStrides(b, out)
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-gray-500">
          <span className="flex items-center gap-1"><span className="inline-block h-3 w-3 rounded-sm bg-blue-500" /><span className="inline-block h-3 w-3 rounded-sm bg-orange-500" /> {t('numpy_module.broadcast.legend_real')}</span>
          <span className="flex items-center gap-1"><span className="inline-block h-3 w-3 rounded-sm border border-dashed border-blue-300 bg-blue-50" /><span className="inline-block h-3 w-3 rounded-sm border border-dashed border-orange-300 bg-orange-50" /> {t('numpy_module.broadcast.legend_ghost')}</span>
        </div>
        <label className="flex cursor-pointer select-none items-center gap-1.5 text-xs text-gray-600">
          <input type="checkbox" checked={showGhost} onChange={(e) => setShowGhost(e.target.checked)} className="accent-blue-600" />
          {t('numpy_module.broadcast.show_ghost')}
        </label>
      </div>
      <div className="overflow-x-auto">
        {/* a narrow panel moves the result to its own row rather than cutting it off */}
        <div className="flex flex-wrap items-start justify-center gap-3 py-1">
          <Grid title={<><b className={TONES.A.text}>{read.nameA}</b> {shapeLabel(a)} → {shapeLabel(out)}</>} shape={out} cell={operandCell('A', a)} onHover={setHover} />
          <OpSign>{sign}</OpSign>
          <Grid title={<><b className={TONES.B.text}>{read.nameB}</b> {shapeLabel(b)} → {shapeLabel(out)}</>} shape={out} cell={operandCell('B', b)} onHover={setHover} />
          {withResult && (
            <div className="flex items-start gap-3">
              <OpSign>=</OpSign>
              <Grid title={<><b className={TONES.R.text}>{read.nameA} {sign} {read.nameB}</b> {shapeLabel(out)}</>} shape={out} cell={resultCell} onHover={setHover} />
            </div>
          )}
        </div>
      </div>
      <div className="min-h-9 rounded-lg border border-gray-200 bg-white px-3 py-2 text-center font-mono text-sm" aria-live="polite">
        {formula ?? <span className="font-sans text-xs text-gray-500">{t('numpy_module.broadcast.hover_hint', { a: read.nameA, b: read.nameB })}</span>}
      </div>
      <details className="rounded-lg border border-gray-200 bg-white px-3 py-2 text-xs text-gray-600">
        <summary className="cursor-pointer font-semibold text-gray-700">{t('numpy_module.broadcast.memory_title')}</summary>
        <div className="mt-2 space-y-0.5 font-mono">
          <div><span className="text-gray-500">np.broadcast_to(</span><b className={TONES.A.text}>{read.nameA}</b><span className="text-gray-500">, {shapeLabel(out)}).strides →</span> ({stridesA.join(', ')})</div>
          <div><span className="text-gray-500">np.broadcast_to(</span><b className={TONES.B.text}>{read.nameB}</b><span className="text-gray-500">, {shapeLabel(out)}).strides →</span> ({stridesB.join(', ')})</div>
        </div>
        <p className="mt-1.5">{t('numpy_module.broadcast.memory_desc', { a: size(a), b: size(b), out: size(out) })}</p>
      </details>
    </div>
  )
}

// ---------- the two operands as they are ----------

export const InputView = ({ a, b, op, data }: { a: number[]; b: number[]; op: Op; data?: OperandData }) => {
  const read = reader(data)
  const plainCell = (name: 'A' | 'B', shape: number[]) => (idx: number[]): CellInfo => ({ cls: TONES[name].real, text: showCell(name === 'A' ? read.a(idx, shape) : read.b(idx, shape)) })
  return (
    <div className="flex flex-wrap items-start justify-center gap-6 py-2">
      <Grid title={<><b className={TONES.A.text}>{read.nameA}</b> {shapeLabel(a)}</>} shape={a} cell={plainCell('A', a)} />
      <OpSign>{opSign(op)}</OpSign>
      <Grid title={<><b className={TONES.B.text}>{read.nameB}</b> {shapeLabel(b)}</>} shape={b} cell={plainCell('B', b)} />
    </div>
  )
}
