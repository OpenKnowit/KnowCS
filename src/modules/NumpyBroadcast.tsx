import { useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { m } from 'framer-motion'
import { AlertTriangle, ChevronLeft, ChevronRight, Footprints, ListChecks, Wand2, X } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { broadcastStrides, isRealCell, planBroadcast, shapeLabel, sourceIndex, suggestFix } from '../lib/broadcast'
import type { AxisStep } from '../lib/broadcast'
import { runPython } from '../lib/minipy'

// --- 广播实验台：自选两个形状，逐轴走一遍规则，看虚拟拉伸与结果的逐格对应 ---

type Op = '+' | '-' | '*'

const PRESETS: { id: string; a: number[]; b: number[]; op: Op }[] = [
  { id: 'outer', a: [3, 1], b: [1, 4], op: '+' },
  { id: 'row', a: [3, 4], b: [4], op: '-' },
  { id: 'scalar', a: [3, 4], b: [], op: '*' },
  { id: 'mismatch', a: [3, 4], b: [3], op: '+' },
  { id: 'silent', a: [3], b: [3, 1], op: '+' },
  { id: 'cube', a: [2, 3, 4], b: [3, 1], op: '+' },
]

const MAX_NDIM = 3
const MAX_DIM = 5

const size = (shape: number[]) => shape.reduce((p, x) => p * x, 1)
const flatOf = (idx: number[], shape: number[]) => idx.reduce((f, v, k) => f * shape[k] + v, 0)
// A = arange，B = (arange + 1) * 10：结果里一眼能看出两边各贡献了什么
const valA = (idx: number[], shape: number[]) => flatOf(idx, shape)
const valB = (idx: number[], shape: number[]) => (flatOf(idx, shape) + 1) * 10
const apply = (op: Op, x: number, y: number) => (op === '+' ? x + y : op === '-' ? x - y : x * y)

const pyShape = (s: number[]) => (s.length === 1 ? `${s[0]}` : s.join(', '))
const makeCode = (a: number[], b: number[], op: Op) => {
  const decl = (name: string, s: number[], isB: boolean) => {
    if (s.length === 0) return `${name} = ${isB ? 10 : 0}`
    const range = isB ? `np.arange(1, ${size(s) + 1})` : `np.arange(${size(s)})`
    const base = s.length === 1 ? range : `${range}.reshape(${pyShape(s)})`
    return `${name} = ${base}${isB ? ' * 10' : ''}`
  }
  return `import numpy as np\n${decl('A', a, false)}\n${decl('B', b, true)}\nA ${op} B`
}

const TONES = {
  A: { real: 'bg-blue-500 border-blue-600 text-white', ghost: 'bg-blue-50 border-blue-300 text-blue-400 border-dashed', text: 'text-blue-600', chip: 'bg-blue-50 border-blue-200' },
  B: { real: 'bg-orange-500 border-orange-600 text-white', ghost: 'bg-orange-50 border-orange-300 text-orange-400 border-dashed', text: 'text-orange-600', chip: 'bg-orange-50 border-orange-200' },
  R: { real: 'bg-emerald-500 border-emerald-600 text-white', ghost: '', text: 'text-emerald-600', chip: 'bg-emerald-50 border-emerald-200' },
}

// ---------- 形状编辑器 ----------

interface ShapeEditorProps {
  name: 'A' | 'B'
  shape: number[]
  onChange: (s: number[]) => void
}

const ShapeEditor = ({ name, shape, onChange }: ShapeEditorProps) => {
  const { t } = useTranslation()
  const tone = TONES[name]
  return (
    <div className={`rounded-xl border p-3 ${tone.chip}`}>
      <div className="flex items-baseline justify-between mb-2">
        <span className={`text-sm font-bold ${tone.text}`}>{t('numpy_module.broadcast.shape_of', { name })}</span>
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

interface CellInfo {
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

const Grid = ({ title, shape, cell, onHover }: GridProps) => {
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

const OpSign = ({ children }: { children: ReactNode }) => (
  <div className="text-2xl font-bold text-gray-300 self-center px-1">{children}</div>
)

// ---------- 规则逐轴表 ----------

interface RuleTableProps {
  a: number[]
  b: number[]
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

const RuleTable = ({ a, b, steps, phase }: RuleTableProps) => {
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
            <td className={`px-2 py-1.5 text-xs font-bold ${TONES.A.text}`}>A</td>
            {cols.map((s) => dimCell(s.a, s.padA, 'A', s))}
            <td className="px-2 font-mono text-xs text-gray-500">{shapeLabel(a)}</td>
          </tr>
          <tr>
            <td className={`px-2 py-1.5 text-xs font-bold ${TONES.B.text}`}>B</td>
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
            <td className="px-2 py-1.5 text-xs font-bold text-emerald-600">A {'∘'} B</td>
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

const ML_USES = [
  { id: 'std', code: 'X - X.mean(axis=0)', shapes: '(n, d) − (d,)' },
  { id: 'bias', code: 'X @ W + b', shapes: '(n, k) + (k,)' },
  { id: 'knn', code: 'X[:, None, :] - Y[None, :, :]', shapes: '(n, 1, d) − (1, m, d) → (n, m, d)' },
  { id: 'gray', code: '(img * [0.299, 0.587, 0.114]).sum(axis=2)', shapes: '(H, W, 3) × (3,) → (H, W, 3)' },
]

const InfoCard = ({ title, children, warn }: { title: string; children: ReactNode; warn?: boolean }) => (
  <div className={`rounded-xl border p-4 text-sm leading-relaxed ${warn ? 'bg-amber-50 border-amber-200 text-amber-900' : 'bg-white border-gray-200 text-gray-600'}`}>
    <div className={`font-semibold mb-1.5 ${warn ? 'text-amber-800' : 'text-gray-800'}`}>{title}</div>
    {children}
  </div>
)

// ---------- 主组件 ----------

export const NumpyBroadcast = () => {
  const { t } = useTranslation()
  const [a, setA] = useState<number[]>(PRESETS[0].a)
  const [b, setB] = useState<number[]>(PRESETS[0].b)
  const [op, setOp] = useState<Op>(PRESETS[0].op)
  const [phase, setPhase] = useState<number | null>(null)
  const [hover, setHover] = useState<number[] | null>(null)
  const [showGhost, setShowGhost] = useState(true)

  const plan = useMemo(() => planBroadcast(a, b), [a, b])
  const fix = useMemo(() => suggestFix(a, b), [a, b])
  const code = makeCode(a, b, op)
  const run = useMemo(() => runPython(code), [code])

  const reset = (na: number[], nb: number[], nop: Op = op) => {
    setA(na)
    setB(nb)
    setOp(nop)
    setPhase(null)
    setHover(null)
  }
  const preset = PRESETS.find((p) => p.a.join() === a.join() && p.b.join() === b.join() && p.op === op)

  const nd = plan.ndim
  const failAt = plan.steps.findIndex((s) => s.verdict === 'mismatch')
  const lastPhase = failAt >= 0 ? failAt + 1 : nd + 1
  const ph = phase ?? lastPhase
  const done = ph >= lastPhase

  // 当前步骤的讲解：phase 为 null 时直接给出结论
  const summary = plan.ok
    ? t('numpy_module.broadcast.narr.done_ok', { shape: shapeLabel(plan.outShape!) })
    : t('numpy_module.broadcast.narr.done_fail', { axis: plan.steps[failAt].axis })
  const stepText = (s: AxisStep) => t(`numpy_module.broadcast.narr.${s.verdict}`, { axis: s.axis, a: s.a, b: s.b, out: s.out })
  const narration = phase === null ? summary : ph === 0 ? t('numpy_module.broadcast.narr.align', { a: shapeLabel(a), b: shapeLabel(b) }) : ph <= nd ? stepText(plan.steps[ph - 1]) : summary

  const out = plan.outShape
  const opSign = op === '*' ? '×' : op === '-' ? '−' : '+'
  const idxStr = (idx: number[]) => (idx.length ? `[${idx.join(', ')}]` : '')

  // 悬停位置在各操作数中的「真身」坐标（以结果坐标表示）
  const realPos = (idx: number[], shape: number[]) => {
    const pad = idx.length - shape.length
    return idx.map((v, k) => (k < pad || shape[k - pad] === 1 ? 0 : v))
  }
  const same = (x: number[] | null, y: number[]) => !!x && x.length === y.length && x.every((v, i) => v === y[i])

  const operandCell = (name: 'A' | 'B', shape: number[]) => (idx: number[]): CellInfo => {
    const real = isRealCell(idx, shape)
    const v = name === 'A' ? valA(sourceIndex(idx, shape), shape) : valB(sourceIndex(idx, shape), shape)
    const tone = TONES[name]
    const ring = same(hover, idx) ? 'hover' : hover && same(realPos(hover, shape), idx) ? 'source' : undefined
    if (real) return { cls: tone.real, text: String(v), ring }
    return { cls: showGhost ? tone.ghost : 'bg-white border-dashed border-gray-200 text-transparent', text: showGhost ? String(v) : '', ring }
  }
  const resultCell = (idx: number[]): CellInfo => {
    const v = apply(op, valA(sourceIndex(idx, a), a), valB(sourceIndex(idx, b), b))
    return { cls: TONES.R.real, text: String(v), ring: same(hover, idx) ? 'hover' : undefined }
  }

  const formula = hover && out
    ? (() => {
        const ia = sourceIndex(hover, a)
        const ib = sourceIndex(hover, b)
        const x = valA(ia, a)
        const y = valB(ib, b)
        return `A${idxStr(ia)} ${opSign} B${idxStr(ib)} = ${x} ${opSign} ${y} = ${apply(op, x, y)}`
      })()
    : null

  const stridesA = out ? broadcastStrides(a, out) : []
  const stridesB = out ? broadcastStrides(b, out) : []

  const plainCell = (name: 'A' | 'B', shape: number[]) => (idx: number[]): CellInfo => ({
    cls: TONES[name].real,
    text: String(name === 'A' ? valA(idx, shape) : valB(idx, shape)),
  })

  return (
    <div className="space-y-6">
      <p className="text-sm text-gray-600 leading-relaxed">{t('numpy_module.broadcast.intro')}</p>

      {/* 预设 */}
      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-gray-400 mr-1">{t('numpy_module.broadcast.presets')}</span>
          {PRESETS.map((p) => (
            <button
              key={p.id}
              onClick={() => reset(p.a, p.b, p.op)}
              aria-pressed={preset?.id === p.id}
              className={`px-3 py-1.5 rounded-lg text-xs transition shadow-sm ${preset?.id === p.id ? 'bg-slate-800 text-white' : 'bg-white border border-gray-200 text-gray-700 hover:border-gray-400'}`}
            >
              {t(`numpy_module.broadcast.preset.${p.id}`)}
            </button>
          ))}
        </div>
        {preset && <p className="text-xs text-gray-500 italic">{t(`numpy_module.broadcast.story.${preset.id}`)}</p>}
      </div>

      {/* 形状与运算符 */}
      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-3 items-center">
        <ShapeEditor name="A" shape={a} onChange={(s) => reset(s, b)} />
        <div className="flex md:flex-col justify-center gap-1" role="group" aria-label={t('numpy_module.broadcast.op')}>
          {(['+', '-', '*'] as Op[]).map((o) => (
            <button
              key={o}
              onClick={() => setOp(o)}
              aria-pressed={op === o}
              className={`w-9 h-9 rounded-lg font-mono font-bold text-lg ${op === o ? 'bg-slate-800 text-white' : 'bg-white border border-gray-200 text-gray-500'}`}
            >
              {o === '*' ? '×' : o === '-' ? '−' : '+'}
            </button>
          ))}
        </div>
        <ShapeEditor name="B" shape={b} onChange={(s) => reset(a, s)} />
      </div>

      {/* 第一步：规则逐轴检查 */}
      <div className="rounded-xl border border-gray-200 bg-white p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 text-sm font-semibold text-gray-700">
            <ListChecks size={16} className="text-blue-500" /> {t('numpy_module.broadcast.rule_title')}
          </div>
          <div className="flex items-center gap-1.5">
            {phase === null ? (
              <button onClick={() => setPhase(0)} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-600 text-white text-xs font-semibold hover:bg-blue-500">
                <Footprints size={14} /> {t('numpy_module.broadcast.walk')}
              </button>
            ) : (
              <>
                <button onClick={() => setPhase(Math.max(0, ph - 1))} disabled={ph === 0} aria-label={t('numpy_module.broadcast.back')} className="p-1.5 rounded-lg border border-gray-200 disabled:opacity-40">
                  <ChevronLeft size={16} />
                </button>
                <span className="text-xs font-mono text-gray-500 w-12 text-center">{ph}/{lastPhase}</span>
                <button onClick={() => setPhase(Math.min(lastPhase, ph + 1))} disabled={done} aria-label={t('numpy_module.broadcast.next')} className="p-1.5 rounded-lg border border-gray-200 disabled:opacity-40">
                  <ChevronRight size={16} />
                </button>
                <button onClick={() => setPhase(null)} className="px-2 py-1.5 rounded-lg text-xs text-gray-500 hover:text-gray-800">
                  {t('numpy_module.broadcast.show_all')}
                </button>
              </>
            )}
          </div>
        </div>
        <RuleTable a={a} b={b} steps={plan.steps} phase={ph} />
        <m.p
          key={`${ph}-${a.join()}-${b.join()}`}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          aria-live="polite"
          className={`text-sm rounded-lg px-3 py-2 ${done && !plan.ok ? 'bg-rose-50 text-rose-700' : done ? 'bg-emerald-50 text-emerald-800' : 'bg-yellow-50 text-yellow-900'}`}
        >
          {narration}
        </m.p>
      </div>

      {/* 第二步：可视化 */}
      <div className="rounded-xl border border-gray-200 bg-gray-50 p-4 space-y-4">
        {plan.ok && out ? (
          <>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="text-sm font-semibold text-gray-700">{t('numpy_module.broadcast.visual_title', { shape: shapeLabel(out) })}</div>
              <label className="flex items-center gap-1.5 text-xs text-gray-600 cursor-pointer select-none">
                <input type="checkbox" checked={showGhost} onChange={(e) => setShowGhost(e.target.checked)} className="accent-blue-600" />
                {t('numpy_module.broadcast.show_ghost')}
              </label>
            </div>
            <div className="overflow-x-auto">
              <div className="flex items-start justify-center gap-3 w-max mx-auto py-1">
                <Grid title={<><b className={TONES.A.text}>A</b> {shapeLabel(a)} → {shapeLabel(out)}</>} shape={out} cell={operandCell('A', a)} onHover={setHover} />
                <OpSign>{opSign}</OpSign>
                <Grid title={<><b className={TONES.B.text}>B</b> {shapeLabel(b)} → {shapeLabel(out)}</>} shape={out} cell={operandCell('B', b)} onHover={setHover} />
                <OpSign>=</OpSign>
                <Grid title={<><b className={TONES.R.text}>A {opSign} B</b> {shapeLabel(out)}</>} shape={out} cell={resultCell} onHover={setHover} />
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-gray-500">
              <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm bg-blue-500 inline-block" /><span className="w-3 h-3 rounded-sm bg-orange-500 inline-block" /> {t('numpy_module.broadcast.legend_real')}</span>
              <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm border border-dashed border-blue-300 bg-blue-50 inline-block" /><span className="w-3 h-3 rounded-sm border border-dashed border-orange-300 bg-orange-50 inline-block" /> {t('numpy_module.broadcast.legend_ghost')}</span>
            </div>
            <div className="min-h-9 font-mono text-sm text-center rounded-lg bg-white border border-gray-200 px-3 py-2" aria-live="polite">
              {formula ?? <span className="font-sans text-xs text-gray-400">{t('numpy_module.broadcast.hover_hint')}</span>}
            </div>

            {/* 内存：步长为 0 的虚拟拉伸 */}
            <div className="text-xs text-gray-600 space-y-1.5 border-t border-gray-200 pt-3">
              <div className="font-semibold text-gray-700">{t('numpy_module.broadcast.memory_title')}</div>
              <div className="font-mono space-y-0.5">
                <div><span className="text-gray-400">np.broadcast_to(</span><b className={TONES.A.text}>A</b><span className="text-gray-400">, {shapeLabel(out)}).strides →</span> ({stridesA.join(', ')})</div>
                <div><span className="text-gray-400">np.broadcast_to(</span><b className={TONES.B.text}>B</b><span className="text-gray-400">, {shapeLabel(out)}).strides →</span> ({stridesB.join(', ')})</div>
              </div>
              <p>{t('numpy_module.broadcast.memory_desc', { a: size(a), b: size(b), out: size(out) })}</p>
            </div>
          </>
        ) : (
          <div className="space-y-4">
            <div className="flex items-start gap-2 p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-sm">
              <AlertTriangle size={18} className="shrink-0 mt-0.5" />
              <div className="space-y-1">
                <div className="font-bold">{t('numpy_module.broadcast.error_title')}</div>
                <div>{t('numpy_module.broadcast.error_axis', { axis: plan.steps[failAt].axis, a: plan.steps[failAt].a, b: plan.steps[failAt].b })}</div>
              </div>
            </div>
            <div className="flex flex-wrap items-start justify-center gap-6">
              <Grid title={<><b className={TONES.A.text}>A</b> {shapeLabel(a)}</>} shape={a} cell={plainCell('A', a)} />
              <OpSign>{opSign}</OpSign>
              <Grid title={<><b className={TONES.B.text}>B</b> {shapeLabel(b)}</>} shape={b} cell={plainCell('B', b)} />
            </div>
            {fix && (
              <div className="flex flex-wrap items-center gap-3 p-3 rounded-lg bg-white border border-emerald-200">
                <Wand2 size={16} className="text-emerald-600" />
                <span className="text-sm text-gray-700">
                  {t('numpy_module.broadcast.fix', { from: shapeLabel(fix.side === 'A' ? a : b), to: shapeLabel(fix.shape) })}{' '}
                  <code className="px-1.5 py-0.5 rounded bg-slate-100 font-mono text-xs">{fix.code}</code>
                </span>
                <button
                  onClick={() => (fix.side === 'A' ? reset(fix.shape, b) : reset(a, fix.shape))}
                  className="px-3 py-1.5 rounded-lg bg-emerald-600 text-white text-xs font-semibold hover:bg-emerald-500"
                >
                  {t('numpy_module.broadcast.fix_btn')}
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* 等价代码（由站内解释器真实运行） */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="rounded-xl bg-slate-900 p-3">
          <div className="text-[11px] text-slate-400 mb-1">{t('numpy_module.broadcast.code_title')}</div>
          <pre className="font-mono text-[12.5px] leading-5 text-slate-100 whitespace-pre overflow-x-auto">{code}</pre>
        </div>
        <div className="rounded-xl border border-gray-200 bg-white p-3 min-w-0">
          <div className="text-[11px] text-gray-400 mb-1">{t('numpy_module.broadcast.output_title')}</div>
          {run.error ? (
            <p className="font-mono text-[12.5px] leading-5 text-rose-600 whitespace-pre-wrap break-words"><b>{run.error.type}</b>: {run.error.message}</p>
          ) : (
            <pre className="font-mono text-[12.5px] leading-5 text-sky-700 whitespace-pre overflow-x-auto">{run.out}</pre>
          )}
        </div>
      </div>

      {/* 知识卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <InfoCard title={t('numpy_module.broadcast.cards.rule_t')}>
          <ol className="list-decimal pl-4 space-y-1">
            <li>{t('numpy_module.broadcast.cards.rule_1')}</li>
            <li>{t('numpy_module.broadcast.cards.rule_2')}</li>
            <li>{t('numpy_module.broadcast.cards.rule_3')}</li>
          </ol>
        </InfoCard>
        <InfoCard title={t('numpy_module.broadcast.cards.mem_t')}>{t('numpy_module.broadcast.cards.mem')}</InfoCard>
        <InfoCard title={t('numpy_module.broadcast.cards.ml_t')}>
          <ul className="space-y-1.5">
            {ML_USES.map((u) => (
              <li key={u.id}>
                <code className="font-mono text-[11.5px] px-1 py-0.5 rounded bg-slate-100 text-slate-800">{u.code}</code>
                <span className="block text-[11px] text-gray-500 mt-0.5"><span className="font-mono">{u.shapes}</span> · {t(`numpy_module.broadcast.cards.ml_${u.id}`)}</span>
              </li>
            ))}
          </ul>
        </InfoCard>
        <InfoCard title={t('numpy_module.broadcast.cards.pit_t')} warn>
          <p>{t('numpy_module.broadcast.cards.pit')}</p>
          <button onClick={() => reset([3], [3, 1], '+')} className="mt-2 text-xs font-semibold text-amber-700 underline underline-offset-2 hover:text-amber-900">
            {t('numpy_module.broadcast.cards.pit_try')}
          </button>
        </InfoCard>
      </div>
    </div>
  )
}
