import { useEffect, useMemo, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { AlertTriangle, ArrowRight, Search, Sparkles } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { API_CATS, NUMPY_APIS } from '../data/numpyApis'
import type { ApiCat, ApiEntry } from '../data/numpyApis'
import { prod, shapeStr, unravel } from '../lib/ndarray'
import type { DType } from '../lib/ndarray'
import { runPython } from '../lib/minipy'
import type { CallTrace, GridSnapshot } from '../lib/minipy'
import { dependents, explainCall } from '../lib/npTrace'
import type { Cell, Explain } from '../lib/npTrace'

// --- NumPy API 可视化面板：在浏览器里运行示例（lib/minipy），并标出结果每个元素来自哪些输入元素 ---

const MAX_DRAW = 240
const OPERAND_TINT = ['bg-sky-50', 'bg-rose-50', 'bg-emerald-50', 'bg-amber-50']

const cellText = (v: number, dtype: DType): string => {
  if (dtype === 'bool') return v ? 'T' : 'F'
  if (dtype === 'int64') return String(v)
  if (Number.isNaN(v)) return 'nan'
  if (!Number.isFinite(v)) return v > 0 ? 'inf' : '-inf'
  if (Number.isInteger(v)) return `${v}.`
  const a = Math.abs(v)
  return a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(1) : String(+v.toFixed(3))
}

type Look = 'none' | 'hover' | 'source' | 'pick' | 'dependent'

const LOOK: Record<Look, string> = {
  none: 'bg-white border-slate-200 text-slate-700',
  hover: 'bg-blue-600 border-blue-600 text-white',
  source: 'bg-amber-100 border-amber-400 text-amber-900',
  pick: 'bg-emerald-200 border-emerald-500 text-emerald-900 ring-2 ring-emerald-500 ring-offset-1',
  dependent: 'bg-blue-100 border-blue-400 text-blue-900',
}

interface GridProps {
  title: string
  snap: GridSnapshot
  look: (flat: number) => Look
  tint?: (flat: number) => string | undefined
  onHover: (flat: number | null) => void
}

/** 任意维数组：≥3 维时按前导下标拆成多个二维块 */
const NdGrid = ({ title, snap, look, tint, onHover }: GridProps) => {
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
      <div className="mb-1 flex flex-wrap items-baseline gap-x-2 text-xs font-bold text-slate-600">
        <span className="font-mono">{title}</span>
        <span className="font-mono font-normal text-slate-400">{n === 0 ? t('numpy_api.scalar') : `${shapeStr(shape)} ${snap.dtype}`}</span>
      </div>
      {snap.values.length > MAX_DRAW ? (
        <p className="text-xs italic text-slate-400">{t('numpy_api.too_large', { shape: shapeStr(shape) })}</p>
      ) : snap.values.length === 0 ? (
        <div className="rounded border border-dashed border-slate-300 px-3 py-2 font-mono text-xs text-slate-400">[ ]</div>
      ) : (
        <div className="space-y-1.5 overflow-x-auto pb-1" onMouseLeave={() => onHover(null)}>
          {blocks.map(({ label, base }) => (
            <div key={base}>
              {label && <div className="mb-0.5 font-mono text-[10px] text-slate-400">{label}</div>}
              <div className="inline-grid gap-1" style={{ gridTemplateColumns: `repeat(${cols}, minmax(2.1rem, auto))` }}>
                {Array.from({ length: per }, (_, i) => {
                  const flat = base + i
                  const l = look(flat)
                  return (
                    <button
                      type="button"
                      key={flat}
                      onMouseEnter={() => onHover(flat)}
                      onFocus={() => onHover(flat)}
                      onClick={() => onHover(flat)}
                      className={`h-9 rounded border px-1 font-mono text-xs transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ${LOOK[l]} ${l === 'none' ? tint?.(flat) ?? '' : ''}`}
                    >
                      {cellText(snap.values[flat], snap.dtype)}
                    </button>
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

type Hover = { side: 'out'; flat: number } | { side: 'in'; k: number; flat: number } | null

/** 针对一种调用，给出一句「结果怎么来的」规则说明 */
const useRule = (call: CallTrace, ex: Explain): string => {
  const { t } = useTranslation()
  const op = call.api.split('.').pop() ?? call.api
  const input = call.operands[0]?.snap
  const out = call.result
  switch (call.kind) {
    case 'create': return t('numpy_api.rule.create')
    case 'random': return t('numpy_api.rule.random')
    case 'info': return t('numpy_api.rule.info')
    case 'move': return call.operands.length > 1 ? t('numpy_api.rule.move_many', { n: call.operands.length }) : t('numpy_api.rule.move')
    case 'elementwise':
      return t('numpy_api.rule.elementwise', {
        shapes: call.operands.map((o) => (o.snap.shape.length ? shapeStr(o.snap.shape) : t('numpy_api.scalar'))).join(' , '),
        out: out ? shapeStr(out.shape) : '—',
      })
    case 'reduce':
      if (ex.axis === null || !input) return t('numpy_api.rule.reduce_all', { n: input ? input.values.length : 0, op })
      return t('numpy_api.rule.reduce_axis', { axis: ex.axis, len: input.shape[ex.axis], from: shapeStr(input.shape), to: out ? shapeStr(out.shape) : '()', op })
    case 'scan': return ex.axis === null ? t('numpy_api.rule.scan_flat') : t('numpy_api.rule.scan_axis', { axis: ex.axis })
    case 'matmul':
      if (op === 'outer') return t('numpy_api.rule.outer')
      return t('numpy_api.rule.matmul', { k: input ? input.shape[input.shape.length - 1] : '?' })
    case 'sort': return op === 'argsort' ? t('numpy_api.rule.argsort') : ex.axis === null ? t('numpy_api.rule.sort_flat') : t('numpy_api.rule.sort', { axis: ex.axis })
    case 'unique': return t('numpy_api.rule.unique')
    case 'linalg': return op === 'trace' ? t('numpy_api.rule.trace') : t('numpy_api.rule.linalg')
  }
}

const CallView = ({ call }: { call: CallTrace }) => {
  const { t } = useTranslation()
  const ex = useMemo(() => explainCall(call), [call])
  const deps = useMemo(() => dependents(ex, call.operands.length), [ex, call.operands.length])
  const [hover, setHover] = useState<Hover>(null)
  const rule = useRule(call, ex)
  const traced = ex.sources.some((s) => s.length)

  const sourceSet = new Set<string>()
  const pickSet = new Set<string>()
  const depSet = new Set<number>()
  if (hover?.side === 'out') {
    ex.sources[hover.flat]?.forEach(([k, f]) => sourceSet.add(`${k}:${f}`))
    const p = ex.pick?.[hover.flat]
    if (p) pickSet.add(`${p[0]}:${p[1]}`)
  } else if (hover?.side === 'in') {
    deps[hover.k]?.[hover.flat]?.forEach((o) => depSet.add(o))
  }
  const firstSource = (o: number): Cell | undefined => ex.sources[o]?.[0]
  const tintByOperand = call.kind === 'move' && call.operands.length > 1

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <code className="rounded-lg bg-slate-900 px-2.5 py-1 font-mono text-xs text-slate-100">{call.code}</code>
        <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-[10px] font-black uppercase text-indigo-700">{t(`numpy_api.kind.${call.kind}`)}</span>
        {call.result && call.operands.length > 0 && (
          <span className="font-mono text-[11px] text-slate-400">
            {call.operands.map((o) => (o.snap.shape.length ? shapeStr(o.snap.shape) : '·')).join(' , ')} → {call.result.shape.length ? shapeStr(call.result.shape) : t('numpy_api.scalar')}
          </span>
        )}
      </div>
      <p className="text-sm leading-relaxed text-slate-600">{rule}</p>

      <div className="flex flex-wrap items-start gap-x-4 gap-y-3 rounded-2xl border border-slate-200 bg-slate-50/60 p-3">
        {call.operands.map((op, k) => (
          <NdGrid
            key={k}
            title={op.label}
            snap={op.snap}
            tint={tintByOperand ? () => OPERAND_TINT[k % OPERAND_TINT.length] : undefined}
            onHover={(f) => setHover(f === null ? null : { side: 'in', k, flat: f })}
            look={(f) =>
              hover?.side === 'in' && hover.k === k && hover.flat === f ? 'hover' : pickSet.has(`${k}:${f}`) ? 'pick' : sourceSet.has(`${k}:${f}`) ? 'source' : 'none'
            }
          />
        ))}
        {call.operands.length > 0 && call.result && <ArrowRight className="mt-7 shrink-0 text-slate-300" size={20} aria-hidden />}
        {call.result ? (
          <NdGrid
            title={t('numpy_api.result')}
            snap={call.result}
            tint={tintByOperand ? (o) => { const s = firstSource(o); return s ? OPERAND_TINT[s[0] % OPERAND_TINT.length] : undefined } : undefined}
            onHover={(f) => setHover(f === null ? null : { side: 'out', flat: f })}
            look={(f) => (hover?.side === 'out' && hover.flat === f ? 'hover' : depSet.has(f) ? 'dependent' : 'none')}
          />
        ) : (
          <pre className="whitespace-pre-wrap font-mono text-xs text-slate-700">{call.resultText}</pre>
        )}
      </div>

      {traced && (
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-500">
          <span>{t('numpy_api.hover_hint')}</span>
          <span className="inline-flex items-center gap-1"><i className="inline-block h-3 w-3 rounded border border-amber-400 bg-amber-100" />{t('numpy_api.legend_source')}</span>
          {ex.pick && <span className="inline-flex items-center gap-1"><i className="inline-block h-3 w-3 rounded border border-emerald-500 bg-emerald-200" />{t('numpy_api.legend_pick')}</span>}
          <span className="inline-flex items-center gap-1"><i className="inline-block h-3 w-3 rounded border border-blue-400 bg-blue-100" />{t('numpy_api.legend_dependent')}</span>
        </div>
      )}
    </div>
  )
}

const shortApi = (api: string): string => (api === 'index' ? 'a[…]' : api.startsWith('op:') ? api.slice(3) : api.replace(/^ndarray\./, '.').replace(/^np\./, 'np.'))

interface PanelProps {
  /** Group to open first (from a link in the note) */
  initialCat?: ApiCat | null
  /** Shown when the note linked to a namespace outside the sandbox */
  missing?: string | null
}

export const NumpyApiPanel = ({ initialCat, missing }: PanelProps) => {
  const { t } = useTranslation()
  const [cat, setCat] = useState<ApiCat>(initialCat ?? 'reduce')
  const [query, setQuery] = useState('')
  const firstOf = (c: ApiCat) => NUMPY_APIS.find((e) => e.cat === c)!
  const [entry, setEntry] = useState<ApiEntry>(() => firstOf(initialCat ?? 'reduce'))
  const [code, setCode] = useState(entry.code)
  const [ran, setRan] = useState(entry.code)
  const [picked, setPicked] = useState<{ code: string; id: number } | null>(null)

  // run 300 ms after the last keystroke
  useEffect(() => {
    if (code === ran) return
    const id = setTimeout(() => setRan(code), 300)
    return () => clearTimeout(id)
  }, [code, ran])

  const result = useMemo(() => runPython(ran), [ran])
  const calls = result.calls
  const autoId = (calls.find((c) => entry.match.includes(c.api)) ?? calls[calls.length - 1])?.id
  const selectedId = picked && picked.code === ran ? picked.id : autoId
  const selected = calls.find((c) => c.id === selectedId)

  const q = query.trim().toLowerCase()
  const list = q
    ? NUMPY_APIS.filter((e) => e.label.toLowerCase().includes(q) || e.id.includes(q) || t(`numpy_api.api.${e.id}`).toLowerCase().includes(q))
    : NUMPY_APIS.filter((e) => e.cat === cat)

  const open = (e: ApiEntry) => {
    setEntry(e)
    setCode(e.code)
    setRan(e.code)
    setPicked(null)
  }
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault()
      setRan(code)
    }
  }

  return (
    <aside aria-label={t('numpy_api.title')} className="space-y-4 rounded-[1.5rem] border border-slate-200 bg-white p-4 shadow-xl shadow-slate-200/40 sm:p-5">
      <div>
        <div className="flex items-center gap-2">
          <span className="rounded-lg bg-blue-600 p-1.5 text-white" aria-hidden><Sparkles size={16} /></span>
          <h3 className="text-base font-black text-slate-900">{t('numpy_api.title')}</h3>
        </div>
        <p className="mt-1 text-xs text-slate-500">{t('numpy_api.subtitle', { n: NUMPY_APIS.length })}</p>
      </div>

      {missing && (
        <div className="flex gap-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" aria-hidden />
          <span>{t('numpy_api.missing', { name: missing })}</span>
        </div>
      )}

      <div className="space-y-2">
        <label className="relative block">
          <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" aria-hidden />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t('numpy_api.search')}
            aria-label={t('numpy_api.search')}
            className="w-full rounded-xl border border-slate-200 bg-slate-50 py-2 pl-8 pr-3 text-sm outline-none focus:border-blue-400 focus:bg-white"
          />
        </label>
        {!q && (
          <div className="flex flex-wrap gap-1" role="tablist" aria-label={t('numpy_api.groups')}>
            {API_CATS.map((c) => (
              <button
                key={c}
                role="tab"
                aria-selected={c === cat}
                onClick={() => {
                  setCat(c)
                  open(firstOf(c))
                }}
                className={`rounded-full px-2.5 py-1 text-xs font-bold transition ${c === cat ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
              >
                {t(`numpy_api.cat.${c}`)}
              </button>
            ))}
          </div>
        )}
        <div className="flex flex-wrap gap-1.5">
          {list.map((e) => (
            <button
              key={e.id}
              onClick={() => open(e)}
              aria-pressed={e.id === entry.id}
              className={`rounded-lg border px-2 py-1 font-mono text-xs transition ${e.id === entry.id ? 'border-blue-500 bg-blue-50 text-blue-700' : 'border-slate-200 text-slate-600 hover:border-blue-300'}`}
            >
              {e.label}
            </button>
          ))}
          {!list.length && <span className="text-xs text-slate-400">{t('numpy_api.no_match')}</span>}
        </div>
      </div>

      <p className="rounded-xl bg-blue-50 px-3 py-2 text-sm text-blue-950">
        <b className="font-mono">{entry.label}</b> — {t(`numpy_api.api.${entry.id}`)}
      </p>

      <div>
        <div className="mb-1 flex items-center justify-between text-[11px] font-bold text-slate-400">
          <span>{t('numpy_api.editor')}</span>
          <span>{t('numpy_api.shortcut')}</span>
        </div>
        <textarea
          value={code}
          onChange={(e) => setCode(e.target.value)}
          onKeyDown={onKey}
          spellCheck={false}
          aria-label={t('numpy_api.editor')}
          rows={Math.min(8, Math.max(3, code.split('\n').length))}
          className="w-full resize-y rounded-xl border border-slate-800 bg-slate-900 p-3 font-mono text-[13px] leading-relaxed text-slate-100 outline-none focus:ring-2 focus:ring-blue-400"
        />
      </div>

      {calls.length > 1 && (
        <div className="flex flex-wrap items-center gap-1" aria-label={t('numpy_api.steps')}>
          <span className="mr-1 text-[11px] font-bold text-slate-400">{t('numpy_api.steps')}</span>
          {calls.map((c) => (
            <button
              key={c.id}
              title={c.code}
              aria-pressed={c.id === selectedId}
              onClick={() => setPicked({ code: ran, id: c.id })}
              className={`rounded-md border px-1.5 py-0.5 font-mono text-[11px] ${c.id === selectedId ? 'border-slate-900 bg-slate-900 text-white' : 'border-slate-200 text-slate-500 hover:border-slate-400'}`}
            >
              L{c.line} {shortApi(c.api)}
            </button>
          ))}
        </div>
      )}

      {selected ? <CallView key={`${ran}#${selected.id}`} call={selected} /> : !result.error && <p className="text-sm text-slate-400">{t('numpy_api.nothing')}</p>}

      {(result.error || result.out || result.stdout) && (
        <div aria-live="polite" className="rounded-xl border border-slate-200 bg-slate-50 p-3">
          <div className="mb-1 text-[11px] font-bold text-slate-400">{t('numpy_api.output')}</div>
          {result.stdout && <pre className="whitespace-pre-wrap font-mono text-xs text-slate-700">{result.stdout}</pre>}
          {result.out && <pre className="overflow-x-auto font-mono text-xs text-slate-800">{result.out}</pre>}
          {result.error && (
            <p className="whitespace-pre-wrap font-mono text-xs text-rose-600">
              {result.error.type}: {result.error.message}
              {result.error.line ? ` (${t('numpy_api.line', { n: result.error.line })})` : ''}
            </p>
          )}
        </div>
      )}
      <p className="text-[11px] leading-relaxed text-slate-400">{t('numpy_api.sandbox_note')}</p>
    </aside>
  )
}
