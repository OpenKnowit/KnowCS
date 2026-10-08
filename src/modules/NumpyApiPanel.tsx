import { useEffect, useMemo, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { createPortal } from 'react-dom'
import { AlertTriangle, Maximize2, Minimize2, Search, Sparkles } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { API_CATS, NUMPY_APIS } from '../data/numpyApis'
import type { ApiCat, ApiEntry } from '../data/numpyApis'
import { runPython } from '../lib/minipy'
import { CallView } from './NpCallView'
import { useFullScreen } from '../site/playground/useFullScreen'

// --- NumPy API 可视化面板：在浏览器里运行示例（lib/minipy），并标出结果每个元素来自哪些输入元素 ---

const shortApi = (api: string): string => (api === 'index' ? 'a[…]' : api.startsWith('op:') ? api.slice(3) : api.replace(/^ndarray\./, '.').replace(/^np\./, 'np.'))

interface PanelProps {
  /** Group to open first (from a link in the note) */
  initialCat?: ApiCat | null
  /** Example to open first (from a code block in the note) */
  initialEntry?: string | null
  /** Shown when the note linked to a namespace outside the sandbox */
  missing?: string | null
}

export const NumpyApiPanel = ({ initialCat, initialEntry, missing }: PanelProps) => {
  const { t } = useTranslation()
  const start = NUMPY_APIS.find((e) => e.id === initialEntry)
  const [cat, setCat] = useState<ApiCat>(start?.cat ?? initialCat ?? 'reduce')
  const [query, setQuery] = useState('')
  const firstOf = (c: ApiCat) => NUMPY_APIS.find((e) => e.cat === c)!
  const [entry, setEntry] = useState<ApiEntry>(() => start ?? firstOf(initialCat ?? 'reduce'))
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
    if (e.key === 'Escape') {
      e.stopPropagation()
      e.currentTarget.blur()
    }
  }
  const { full, enter, leave, ref } = useFullScreen()

  const header = (
    <div className="flex items-start justify-between gap-3">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="rounded-lg bg-blue-600 p-1.5 text-white" aria-hidden><Sparkles size={16} /></span>
          <h3 className="text-base font-black text-slate-900">{t('numpy_api.title')}</h3>
        </div>
        <p className="mt-1 text-xs text-slate-500">{t('numpy_api.subtitle', { n: NUMPY_APIS.length })}</p>
      </div>
      <button
        type="button"
        onClick={full ? leave : enter}
        aria-pressed={full}
        className="flex shrink-0 items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-xs font-bold text-slate-700 shadow-sm transition hover:border-blue-300 hover:text-blue-700"
      >
        {full ? <Minimize2 size={14} aria-hidden /> : <Maximize2 size={14} aria-hidden />}
        {full ? t('playground.ui.exit_full') : t('playground.ui.full')}
      </button>
    </div>
  )

  const left = (
    <>

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

      <div className={full ? 'flex min-h-0 flex-1 flex-col' : ''}>
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
          rows={full ? Math.max(8, code.split('\n').length) : Math.min(8, Math.max(3, code.split('\n').length))}
          className={`w-full ${full ? 'min-h-48 flex-1' : ''} resize-y rounded-xl border border-slate-800 bg-slate-900 p-3 font-mono text-[13px] leading-relaxed text-slate-100 outline-none focus:ring-2 focus:ring-blue-400`}
        />
      </div>

    </>
  )

  const right = (
    <>
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
    </>
  )

  const output = (
    <>

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
    </>
  )

  if (full) {
    return createPortal(
      <div ref={ref} role="dialog" aria-modal="true" aria-label={t('numpy_api.title')} className="fixed inset-0 z-[70] flex flex-col bg-slate-50">
        <div className="border-b border-slate-200 bg-white px-4 py-3 sm:px-6">{header}</div>
        <div className="grid min-h-0 flex-1 gap-4 overflow-y-auto p-4 sm:p-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:overflow-hidden">
          <div className="flex min-h-0 flex-col gap-4 lg:overflow-y-auto lg:pr-1">
            {left}
            {output}
          </div>
          <div className="min-h-0 space-y-4 rounded-2xl border border-slate-200 bg-white p-4 lg:overflow-y-auto">{right}</div>
        </div>
      </div>,
      document.body,
    )
  }

  return (
    <aside ref={ref} aria-label={t('numpy_api.title')} className="space-y-4 rounded-[1.5rem] border border-slate-200 bg-white p-4 shadow-xl shadow-slate-200/40 sm:p-5">
      {header}
      {left}
      {right}
      {output}
      <p className="text-[11px] leading-relaxed text-slate-400">{t('numpy_api.sandbox_note')}</p>
    </aside>
  )
}
