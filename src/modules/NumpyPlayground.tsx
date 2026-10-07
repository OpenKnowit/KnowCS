import { useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { m } from 'framer-motion'
import { AlertTriangle, ArrowRight, Play, TerminalSquare } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { prod, shapeStr, unravel } from '../lib/ndarray'
import type { DType } from '../lib/ndarray'
import { runPython } from '../lib/minipy'
import type { GridSnapshot, IndexTrace, RunResult } from '../lib/minipy'

// --- NumPy 索引实验台：浏览器内运行 Python/NumPy 子集（lib/minipy），可视化每一次下标读写 ---

const PRESETS: { id: string; code: string }[] = [
  { id: 'basic', code: 'import numpy as np\na = np.arange(16).reshape(4, 4)\na[1:3, ::2]' },
  { id: 'fancy', code: 'import numpy as np\na = np.arange(16).reshape(4, 4)\nrows = a[[0, 2]]          # whole rows 0 and 2\na[[0, 2, 3], [1, 3, 0]]   # points (0,1) (2,3) (3,0)' },
  { id: 'mask', code: 'import numpy as np\na = np.arange(16).reshape(4, 4)\nmask = (a % 3 == 0) | (a > 12)\na[mask]' },
  { id: 'pitfall', code: 'import numpy as np\na = np.arange(16).reshape(4, 4)\nb = a[1:3, 1:3]   # view\nb[0, 0] = 99      # ...so a changes too!\nc = a[[1, 2]]     # copy\nc[0, 0] = -1      # a is safe\na' },
  { id: 'axes', code: 'import numpy as np\na = np.arange(24).reshape(2, 3, 4)\na[:, [0, 2], 1:3]\na[0, :, [1, 3]]   # shape (2, 3), not (3, 2)!' },
  { id: 'newaxis', code: 'import numpy as np\nx = np.arange(4)\ncol = x[:, None]\nrow = x[None, :]\ncol + row' },
  { id: 'relu', code: 'import numpy as np\nz = np.array([[-2, 3, -1], [4, -5, 6]])\nz[z < 0] = 0      # ReLU, in place\nz' },
  { id: 'precedence', code: 'import numpy as np\na = np.arange(16).reshape(4, 4)\na[a > 3 & a < 10]   # fix: (a > 3) & (a < 10)' },
]

type Tone = 'view' | 'copy' | 'mask' | 'scalar' | 'write'

const TONE: Record<Tone, { cell: string; chip: string; badge: string }> = {
  view: { cell: 'bg-green-100 border-green-500 text-green-900', chip: 'border-green-500 text-green-700', badge: 'bg-green-600' },
  copy: { cell: 'bg-blue-100 border-blue-500 text-blue-900', chip: 'border-blue-500 text-blue-700', badge: 'bg-blue-600' },
  mask: { cell: 'bg-purple-100 border-purple-500 text-purple-900', chip: 'border-purple-500 text-purple-700', badge: 'bg-purple-600' },
  scalar: { cell: 'bg-slate-200 border-slate-500 text-slate-900', chip: 'border-slate-500 text-slate-700', badge: 'bg-slate-600' },
  write: { cell: 'bg-amber-100 border-amber-500 text-amber-900', chip: 'border-amber-500 text-amber-700', badge: 'bg-amber-600' },
}

const toneOf = (tr: IndexTrace): Tone => (tr.result === 'copy' && tr.hasMask ? 'mask' : tr.result)

const MAX_DRAW = 400

const cellText = (v: number, dtype: DType): string => {
  if (dtype === 'bool') return v ? 'T' : 'F'
  if (dtype === 'int64') return String(v)
  if (Number.isNaN(v)) return 'nan'
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf'
  return Number.isInteger(v) ? `${v}.` : String(+v.toFixed(3))
}

interface CellLook {
  tone?: Tone
  dim?: boolean
  badge?: string
  sub?: string
  active?: boolean
}

interface ArrayGridProps {
  title: ReactNode
  snap: GridSnapshot
  look: (flat: number) => CellLook
  onHover?: (flat: number | null) => void
}

/** 任意维数组的网格：≥3 维时按前导下标拆成多个二维块 */
const ArrayGrid = ({ title, snap, look, onHover }: ArrayGridProps) => {
  const { t } = useTranslation()
  const { shape } = snap
  const n = shape.length
  const rows = n >= 2 ? shape[n - 2] : 1
  const cols = n >= 1 ? shape[n - 1] : 1
  const lead = shape.slice(0, Math.max(0, n - 2))
  const per = rows * cols
  const blocks = n <= 2 ? [{ label: null as string | null, base: 0 }] : Array.from({ length: prod(lead) }, (_, b) => ({ label: `[${unravel(b, lead).join(', ')}, :, :]`, base: b * per }))

  return (
    <div className="min-w-0">
      <div className="text-xs font-semibold text-gray-600 mb-1.5 flex flex-wrap items-baseline gap-x-2">
        <span>{title}</span>
        <span className="font-mono text-gray-400 font-normal">{shapeStr(shape)} {snap.dtype}</span>
      </div>
      {snap.values.length > MAX_DRAW ? (
        <p className="text-xs text-gray-400 italic">{t('numpy_module.playground.too_large', { shape: shapeStr(shape) })}</p>
      ) : snap.values.length === 0 ? (
        <div className="text-xs font-mono text-gray-400 px-3 py-2 border border-dashed border-gray-300 rounded">[ ]</div>
      ) : (
        <div className="overflow-x-auto pb-1 space-y-2" onMouseLeave={() => onHover?.(null)}>
          {blocks.map(({ label, base }) => (
            <div key={base}>
              {label && <div className="text-[10px] font-mono text-gray-400 mb-0.5">{label}</div>}
              <div className="inline-grid gap-1" style={{ gridTemplateColumns: `repeat(${cols}, minmax(2.25rem, auto))` }}>
                {Array.from({ length: per }, (_, i) => {
                  const flat = base + i
                  const l = look(flat)
                  const tone = l.tone ? TONE[l.tone].cell : l.dim ? 'bg-white border-gray-200 text-gray-400' : 'bg-white border-gray-300 text-gray-700'
                  return (
                    <div
                      key={flat}
                      onMouseEnter={() => onHover?.(flat)}
                      onClick={() => onHover?.(flat)}
                      className={`relative h-10 px-1 flex flex-col items-center justify-center rounded border font-mono text-xs transition-shadow cursor-default
                        ${tone} ${l.active ? 'ring-2 ring-rose-500 ring-offset-1 z-10' : ''}`}
                    >
                      <span className={l.tone ? 'font-semibold' : ''}>{cellText(snap.values[flat], snap.dtype)}</span>
                      {l.sub && <span className="text-[8px] leading-none opacity-70">{l.sub}</span>}
                      {l.badge && <span className="absolute -top-1.5 -right-1.5 min-w-4 h-4 px-0.5 rounded-full bg-slate-800 text-white text-[9px] leading-4 text-center">{l.badge}</span>}
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

type Hover = { side: 'src' | 'out'; flat: number } | null

const TraceView = ({ tr }: { tr: IndexTrace }) => {
  const { t } = useTranslation()
  const [hover, setHover] = useState<Hover>(null)
  const tone = toneOf(tr)
  const isWrite = tr.mode === 'write'

  // 源格子 → 结果中的顺序号（可能重复出现）
  const outsOf = useMemo(() => {
    const map = new Map<number, number[]>()
    tr.srcFlat.forEach((s, k) => map.set(s, [...(map.get(s) ?? []), k]))
    return map
  }, [tr])

  const activeSrc = new Set<number>()
  const activeOut = new Set<number>()
  if (hover?.side === 'out') {
    activeOut.add(hover.flat)
    if (tr.srcFlat[hover.flat] !== undefined) activeSrc.add(tr.srcFlat[hover.flat])
  } else if (hover?.side === 'src') {
    activeSrc.add(hover.flat)
    for (const k of outsOf.get(hover.flat) ?? []) activeOut.add(k)
  }

  const showOrder = tr.srcFlat.length <= 48 && tr.srcFlat.length > 1
  const srcLook = (flat: number): CellLook => {
    const ks = outsOf.get(flat)
    return {
      tone: ks ? tone : undefined,
      dim: !ks,
      badge: ks && showOrder && !isWrite ? ks.map((k) => k + 1).join(',') : undefined,
      active: activeSrc.has(flat),
    }
  }
  const coord = (flat: number) => (tr.source.shape.length ? unravel(flat, tr.source.shape).join(',') : '')
  const outLook = (k: number): CellLook => ({
    tone,
    badge: showOrder ? String(k + 1) : undefined,
    sub: tr.out.values.length <= 64 ? `[${coord(tr.srcFlat[k])}]` : undefined,
    active: activeOut.has(k),
  })
  const afterLook = (flat: number): CellLook => ({
    tone: outsOf.has(flat) ? 'write' : undefined,
    dim: !outsOf.has(flat),
    active: activeSrc.has(flat),
  })

  const why = tr.result === 'view' ? 'why_view' : tr.result === 'scalar' ? 'why_scalar' : tr.result === 'write' ? 'why_write' : tr.hasMask ? 'why_mask' : 'why_fancy'
  const mem = tr.memory
  const srcAddrSet = new Set(mem.sourceAddrs)
  const pickedSet = new Set(mem.picked)
  const showMemory = mem.buffer.length <= 64

  return (
    <m.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="space-y-5">
      <div className="flex flex-wrap items-start gap-3">
        <code className="px-2 py-1 rounded bg-slate-900 text-slate-100 text-sm font-mono break-all">{tr.code}</code>
        <span className={`px-2 py-1 rounded text-white text-[11px] font-bold uppercase tracking-wider ${TONE[tone].badge}`}>
          {t(`numpy_module.playground.badge.${tr.result}`)}
        </span>
      </div>
      <p className="text-sm text-gray-600 leading-relaxed -mt-2">
        {t(`numpy_module.playground.${why}`, { target: tr.target })}
        {tr.advFront && <span className="block mt-1 text-amber-700">{t('numpy_module.playground.adv_front')}</span>}
      </p>

      {/* 每个索引项作用在哪个轴上 */}
      {tr.notes.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {tr.notes.map((n, i) => (
            <div key={i} className={`text-xs border rounded-lg px-2 py-1 bg-white ${n.implicit ? 'border-dashed border-gray-300 text-gray-400' : 'border-gray-200 text-gray-600'}`}>
              {n.axes.length > 0 && <span className="text-gray-400 mr-1">{n.axes.map((a) => t('numpy_module.playground.axis', { axis: a })).join(' + ')} ←</span>}
              <code className="font-mono font-semibold text-gray-800">{n.label}</code>
              <span className="mx-1">·</span>
              {t(`numpy_module.playground.note.${n.kind}`, { count: n.picks.length })}
              {n.implicit && <span className="ml-1">({t('numpy_module.playground.implicit')})</span>}
              {(n.kind === 'slice' || n.kind === 'fancy') && n.picks.length > 0 && n.picks.length <= 12 && (
                <span className="ml-1 font-mono text-gray-500">→ [{n.picks.join(', ')}]</span>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="flex flex-col lg:flex-row lg:items-start gap-4 lg:gap-6">
        <ArrayGrid
          title={isWrite ? t('numpy_module.playground.before', { name: tr.target }) : t('numpy_module.playground.source', { name: tr.target })}
          snap={tr.source}
          look={srcLook}
          onHover={(f) => setHover(f === null ? null : { side: 'src', flat: f })}
        />
        {tr.mask && (
          <ArrayGrid
            title={t('numpy_module.playground.mask_grid')}
            snap={tr.mask.snapshot}
            look={(f) => ({ tone: tr.mask!.snapshot.values[f] ? 'mask' : undefined, dim: !tr.mask!.snapshot.values[f] })}
          />
        )}
        <ArrowRight className="hidden lg:block text-gray-300 shrink-0 mt-10" />
        {isWrite && tr.after ? (
          <ArrayGrid
            title={t('numpy_module.playground.after', { name: tr.target })}
            snap={tr.after}
            look={afterLook}
            onHover={(f) => setHover(f === null ? null : { side: 'src', flat: f })}
          />
        ) : (
          <ArrayGrid
            title={t('numpy_module.playground.result')}
            snap={tr.out}
            look={outLook}
            onHover={(f) => setHover(f === null ? null : { side: 'out', flat: f })}
          />
        )}
      </div>

      {/* 写穿视图：被连带修改的变量 */}
      {tr.aliases.map((al) => (
        <div key={al.name} className="p-4 rounded-xl border-2 border-rose-200 bg-rose-50 space-y-3">
          <div className="flex items-start gap-2 text-rose-800 text-sm">
            <AlertTriangle size={18} className="shrink-0 mt-0.5" />
            <div>
              <div className="font-bold">{t('numpy_module.playground.alias_title', { name: al.name })}</div>
              <div>{t('numpy_module.playground.alias_desc', { name: al.name, target: tr.target, count: al.changed.length })}</div>
            </div>
          </div>
          <ArrayGrid
            title={t('numpy_module.playground.after', { name: al.name })}
            snap={al.after}
            look={(f) => ({ tone: al.changed.includes(f) ? 'write' : undefined, dim: !al.changed.includes(f) })}
          />
        </div>
      ))}

      {/* 底层一维缓冲区：视图复用、副本新建 */}
      {showMemory && (
        <div className="space-y-2">
          <div className="text-xs font-semibold text-gray-600">{t('numpy_module.playground.memory')}</div>
          <div className="overflow-x-auto pb-1">
            <div className="flex gap-0.5 w-max" onMouseLeave={() => setHover(null)}>
              {mem.buffer.map((v, addr) => {
                const inSrc = srcAddrSet.has(addr)
                const picked = pickedSet.has(addr)
                const srcIdx = mem.sourceAddrs.indexOf(addr)
                return (
                  <div key={addr} className="flex flex-col items-center" onMouseEnter={() => (inSrc ? setHover({ side: 'src', flat: srcIdx }) : setHover(null))}>
                    <div className={`w-8 h-8 flex items-center justify-center rounded-sm border font-mono text-[11px]
                      ${picked ? TONE[tone].cell : inSrc ? 'bg-white border-gray-300 text-gray-600' : 'bg-gray-50 border-dashed border-gray-300 text-gray-300'}
                      ${inSrc && activeSrc.has(srcIdx) ? 'ring-2 ring-rose-500 ring-offset-1' : ''}`}>
                      {cellText(v, mem.dtype)}
                    </div>
                    <span className="text-[9px] text-gray-400 font-mono">{addr}</span>
                  </div>
                )
              })}
            </div>
          </div>
          {mem.sourceAddrs.length < mem.buffer.length && (
            <p className="text-[11px] text-gray-400">{t('numpy_module.playground.memory_legend', { name: tr.target })}</p>
          )}
          {tr.result === 'view' && tr.view && (
            <p className="text-xs text-green-700">
              {t('numpy_module.playground.memory_view', { offset: tr.view.offset, strides: `(${tr.view.strides.join(', ')})` })}
            </p>
          )}
          {tr.result === 'copy' && tr.out.values.length <= 64 && (
            <div className="space-y-1">
              <p className={`text-xs ${tone === 'mask' ? 'text-purple-700' : 'text-blue-700'}`}>{t('numpy_module.playground.memory_copy')}</p>
              <div className="overflow-x-auto pb-1">
                <div className="flex gap-0.5 w-max">
                  {tr.out.values.map((v, k) => (
                    <div key={k} className="flex flex-col items-center" onMouseEnter={() => setHover({ side: 'out', flat: k })} onMouseLeave={() => setHover(null)}>
                      <div className={`w-8 h-8 flex items-center justify-center rounded-sm border font-mono text-[11px] ${TONE[tone].cell} ${activeOut.has(k) ? 'ring-2 ring-rose-500 ring-offset-1' : ''}`}>
                        {cellText(v, tr.out.dtype)}
                      </div>
                      <span className="text-[9px] text-gray-400 font-mono">{k}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      <p className="text-[11px] text-gray-400">{t('numpy_module.playground.hover_hint')}</p>
    </m.div>
  )
}

export const NumpyPlayground = () => {
  const { t } = useTranslation()
  const [code, setCode] = useState(PRESETS[0].code)
  const [result, setResult] = useState<RunResult>(() => runPython(PRESETS[0].code))
  const [selected, setSelected] = useState<number | null>(null)

  // 输入停顿后自动运行（解释器是纯 TS，毫秒级）
  useEffect(() => {
    const h = window.setTimeout(() => setResult(runPython(code)), 300)
    return () => window.clearTimeout(h)
  }, [code])

  const edit = (next: string) => {
    setCode(next)
    setSelected(null)
  }
  const runNow = () => setResult(runPython(code))

  const lines = code.split('\n')
  const errLine = result.error?.line
  const trace = result.traces.find((tr) => tr.id === selected) ?? result.traces.at(-1)
  const activePreset = PRESETS.find((p) => p.code === code)?.id

  return (
    <div className="space-y-5">
      <p className="text-sm text-gray-600 leading-relaxed">{t('numpy_module.playground.intro')}</p>

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-gray-400 mr-1">{t('numpy_module.playground.presets')}</span>
        {PRESETS.map((p) => (
          <button
            key={p.id}
            onClick={() => edit(p.code)}
            aria-pressed={activePreset === p.id}
            className={`px-3 py-1.5 rounded-lg text-xs transition shadow-sm ${activePreset === p.id ? 'bg-slate-800 text-white' : 'bg-white border border-gray-200 text-gray-700 hover:border-gray-400'}`}
          >
            {t(`numpy_module.playground.preset.${p.id}`)}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* 编辑器 */}
        <div className="rounded-xl bg-slate-900 overflow-hidden shadow-sm flex flex-col">
          <div className="flex font-mono text-[13px] leading-6 flex-1">
            <div aria-hidden className="select-none py-3 pl-3 pr-2 text-right text-slate-500 border-r border-slate-700/60">
              {lines.map((_, i) => (
                <div key={i} className={errLine === i + 1 ? 'text-rose-400 font-bold' : ''}>{i + 1}</div>
              ))}
            </div>
            <textarea
              value={code}
              onChange={(e) => edit(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault()
                  runNow()
                }
              }}
              rows={Math.max(lines.length, 7)}
              wrap="off"
              spellCheck={false}
              autoCapitalize="off"
              autoCorrect="off"
              aria-label={t('numpy_module.playground.editor_label')}
              className="flex-1 min-w-0 bg-transparent text-slate-100 py-3 px-3 outline-none resize-none overflow-x-auto whitespace-pre caret-sky-400"
            />
          </div>
          <div className="flex flex-wrap items-center gap-3 px-3 py-2 border-t border-slate-700/60 bg-slate-950/40">
            <button onClick={runNow} className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-sky-700 hover:bg-sky-600 text-white text-xs font-semibold">
              <Play size={12} /> {t('numpy_module.playground.run')}
            </button>
            <span className="text-[11px] text-slate-400">{t('numpy_module.playground.shortcut')}</span>
          </div>
        </div>

        {/* 输出 + 变量 */}
        <div className="flex flex-col gap-3 min-w-0">
          <div className="rounded-xl border border-gray-200 bg-white p-3 flex-1 min-h-32" aria-live="polite">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-gray-500 mb-2">
              <TerminalSquare size={14} /> {t('numpy_module.playground.output')}
            </div>
            <pre className="text-[12.5px] leading-5 font-mono whitespace-pre overflow-x-auto text-gray-800">
              {result.stdout}
              {result.out !== null && <span className="text-sky-700">{result.out}</span>}
              {!result.stdout && result.out === null && !result.error && <span className="text-gray-400 font-sans italic">{t('numpy_module.playground.no_output')}</span>}
            </pre>
            {result.error && (
              <p className={`text-[12.5px] leading-5 font-mono whitespace-pre-wrap break-words text-rose-600 ${result.stdout || result.out ? 'mt-1' : ''}`}>
                {result.error.line ? `${t('numpy_module.playground.error_line', { line: result.error.line })} · ` : ''}
                <strong>{result.error.type}</strong>: {result.error.message}
              </p>
            )}
          </div>
          {result.vars.length > 0 && (
            <div className="rounded-xl border border-gray-200 bg-white p-3">
              <div className="text-xs font-semibold text-gray-500 mb-2">{t('numpy_module.playground.variables')}</div>
              <ul className="space-y-1 text-xs font-mono">
                {result.vars.map((v) => (
                  <li key={v.name} className="flex flex-wrap items-baseline gap-x-2">
                    <span className="font-semibold text-gray-800">{v.name}</span>
                    {v.shape ? (
                      <>
                        <span className="text-gray-500">{shapeStr(v.shape)} {v.dtype}</span>
                        <span className={`font-sans ${v.isView && v.sharesWith.length ? 'text-green-700' : 'text-gray-400'}`}>
                          {v.sharesWith.length
                            ? t('numpy_module.playground.shares', { names: v.sharesWith.join(', ') })
                            : v.isView ? t('numpy_module.playground.view_only') : t('numpy_module.playground.owns')}
                        </span>
                      </>
                    ) : (
                      <span className="text-gray-500 break-all">{v.preview}</span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
      <p className="text-[11px] text-gray-400 -mt-2">{t('numpy_module.playground.subset')}</p>

      {/* 索引轨迹 */}
      <div className="rounded-xl border border-gray-200 bg-gray-50 p-4 space-y-4">
        <div className="text-xs font-semibold text-gray-500">{t('numpy_module.playground.ops', { count: result.traces.length })}</div>
        {result.traces.length === 0 ? (
          <p className="text-sm text-gray-400 italic">{t('numpy_module.playground.no_ops')}</p>
        ) : (
          <>
            <div className="flex flex-wrap gap-2">
              {result.traces.map((tr) => {
                const on = tr.id === trace?.id
                return (
                  <button
                    key={tr.id}
                    onClick={() => setSelected(tr.id)}
                    aria-pressed={on}
                    className={`px-2 py-1 rounded-md border text-xs font-mono transition ${on ? `bg-white shadow-sm ${TONE[toneOf(tr)].chip}` : 'border-gray-200 bg-white/60 text-gray-500 hover:border-gray-400'}`}
                  >
                    <span className="text-gray-400 mr-1.5">L{tr.line}</span>
                    {tr.code}
                  </button>
                )
              })}
            </div>
            {trace && <TraceView key={`${trace.id}:${trace.code}:${trace.line}`} tr={trace} />}
          </>
        )}
      </div>
    </div>
  )
}
