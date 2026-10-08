import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, Info, Maximize2, Minimize2, Play, RotateCcw, Search, Sparkles, TerminalSquare } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { createPortal } from 'react-dom'
import type { CallTrace } from '../../lib/minipy'
import { runIn } from '../../data/playgrounds/run'
import type { Display, PyEvent } from '../../lib/pyEvents'
import type { PlayConfig, PlayEntry } from '../../data/playgrounds/types'
import { CodeEditor } from './CodeEditor'
import { StepView, TableView } from './views'
import { FigureView } from './FigureView'
import { useFullScreen } from './useFullScreen'

// --- 各库实验台：示例目录 + 可编辑代码 + 逐步可视化（调用 / 图 / 模型形状 / 计算图 …）+ 全屏 ---

type Step = { kind: 'call'; seq: number; call: CallTrace } | { kind: 'event'; seq: number; ev: PyEvent }

const stepLine = (s: Step) => (s.kind === 'call' ? s.call.line : s.ev.line)
const stepApi = (s: Step) => (s.kind === 'call' ? s.call.api : s.ev.type)

const shorten = (code: string, n = 26) => {
  const one = code.replace(/\s+/g, ' ').trim()
  return one.length > n ? `${one.slice(0, n - 1)}…` : one
}

const stepLabel = (s: Step): string => {
  if (s.kind === 'call') return s.call.api.startsWith('op:') ? s.call.code : s.call.api.replace(/^ndarray\./, '.')
  return shorten(s.ev.code || s.ev.type)
}

interface Props {
  config: PlayConfig
  /** entry to open first (a link from the note) */
  initialEntry?: string | null
  /** code to open first (a code block from the note) */
  initialCode?: string | null
}

export default function Playground({ config, initialEntry, initialCode }: Props) {
  const { t } = useTranslation()
  const ns = `playground.${config.id}`
  const start = config.entries.find((e) => e.id === initialEntry) ?? config.entries[0]
  const [cat, setCat] = useState(start.cat)
  // null = the code came from a block in the note, not from the catalogue
  const [entry, setEntry] = useState<PlayEntry | null>(initialCode ? null : start)
  const original = entry ? entry.code : initialCode ?? start.code
  const [query, setQuery] = useState('')
  const [code, setCode] = useState(initialCode ?? start.code)
  const [ran, setRan] = useState(code)
  const [picked, setPicked] = useState<{ code: string; seq: number } | null>(null)

  // re-run after a pause in typing
  useEffect(() => {
    if (code === ran) return
    const id = setTimeout(() => setRan(code), config.debounce)
    return () => clearTimeout(id)
  }, [code, ran, config.debounce])

  const result = useMemo(() => runIn(config, ran), [config, ran])

  const steps = useMemo<Step[]>(() => {
    const out: Step[] = [
      ...result.calls.filter((c) => config.callApis.some((p) => c.api.startsWith(p))).map((call): Step => ({ kind: 'call', seq: call.seq, call })),
      ...result.events.filter((ev) => ev.type !== 'note').map((ev): Step => ({ kind: 'event', seq: ev.seq, ev })),
    ]
    return out.sort((a, b) => a.seq - b.seq)
  }, [result, config.callApis])
  const notes = result.events.filter((e): e is PyEvent & { type: 'note' } => e.type === 'note')

  const autoSeq = useMemo(() => {
    if (!steps.length) return null
    const want = entry?.focus
    const hit = want && [...steps].reverse().find((s) => want.includes(stepApi(s)))
    return (hit ?? steps[steps.length - 1]).seq
  }, [steps, entry?.focus])
  const selectedSeq = picked && picked.code === ran ? picked.seq : autoSeq
  const selected = steps.find((s) => s.seq === selectedSeq) ?? null

  const { full, enter, leave, ref: rootRef } = useFullScreen()

  const q = query.trim().toLowerCase()
  const list = q
    ? config.entries.filter((e) => e.label.toLowerCase().includes(q) || e.code.toLowerCase().includes(q) || t(`${ns}.e.${e.id}`).toLowerCase().includes(q))
    : config.entries.filter((e) => e.cat === cat)

  const open = (e: PlayEntry) => {
    setEntry(e)
    setCode(e.code)
    setRan(e.code)
    setPicked(null)
  }

  const header = (
    <div className="flex items-start justify-between gap-3">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="rounded-lg bg-blue-600 p-1.5 text-white" aria-hidden><Sparkles size={16} /></span>
          <h3 className="text-base font-black text-slate-900">{t(`${ns}.title`)}</h3>
        </div>
        <p className="mt-1 text-xs text-slate-500">{t(`${ns}.subtitle`, { n: config.entries.length })}</p>
      </div>
      <button
        type="button"
        onClick={full ? leave : enter}
        className="flex shrink-0 items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-xs font-bold text-slate-700 shadow-sm transition hover:border-blue-300 hover:text-blue-700"
        aria-pressed={full}
      >
        {full ? <Minimize2 size={14} aria-hidden /> : <Maximize2 size={14} aria-hidden />}
        {full ? t('playground.ui.exit_full') : t('playground.ui.full')}
      </button>
    </div>
  )

  const catalog = (
    <div className="space-y-2">
      <label className="relative block">
        <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" aria-hidden />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={t('playground.ui.search')}
          aria-label={t('playground.ui.search')}
          className="w-full rounded-xl border border-slate-200 bg-slate-50 py-2 pl-8 pr-3 text-sm outline-none focus:border-blue-400 focus:bg-white"
        />
      </label>
      {!q && (
        <div className="flex flex-wrap gap-1" role="tablist" aria-label={t('playground.ui.groups')}>
          {config.cats.map((c) => (
            <button
              key={c}
              role="tab"
              aria-selected={c === cat}
              onClick={() => {
                setCat(c)
                open(config.entries.find((e) => e.cat === c)!)
              }}
              className={`rounded-full px-2.5 py-1 text-xs font-bold transition ${c === cat ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
            >
              {t(`${ns}.cat.${c}`)}
            </button>
          ))}
        </div>
      )}
      <div className="flex flex-wrap gap-1.5">
        {list.map((e) => (
          <button
            key={e.id}
            onClick={() => open(e)}
            aria-pressed={e.id === entry?.id}
            className={`rounded-lg border px-2 py-1 font-mono text-xs transition ${e.id === entry?.id ? 'border-blue-500 bg-blue-50 text-blue-700' : 'border-slate-200 text-slate-600 hover:border-blue-300'}`}
          >
            {e.label}
          </button>
        ))}
        {!list.length && <span className="text-xs text-slate-400">{t('playground.ui.no_match')}</span>}
      </div>
      <p className="rounded-xl bg-blue-50 px-3 py-2 text-sm leading-relaxed text-blue-950">
        {entry ? <><b className="font-mono">{entry.label}</b> — {t(`${ns}.e.${entry.id}`)}</> : t('playground.ui.from_note')}
      </p>
    </div>
  )

  const editor = (
    <div className={full ? 'flex min-h-0 flex-1 flex-col' : ''}>
      <div className="mb-1 flex items-center justify-between gap-2 text-[11px] font-bold text-slate-400">
        <span>{t('playground.ui.editor')}</span>
        <span className="flex items-center gap-2">
          <span className="hidden sm:inline">{t('playground.ui.shortcut')}</span>
          {code !== original && (
            <button type="button" onClick={() => setCode(original)} className="flex items-center gap-1 rounded-md px-1.5 py-0.5 text-slate-500 hover:bg-slate-100 hover:text-slate-800">
              <RotateCcw size={11} aria-hidden /> {t('playground.ui.reset')}
            </button>
          )}
          <button type="button" onClick={() => setRan(code)} className="flex items-center gap-1 rounded-md bg-sky-700 px-2 py-0.5 text-white hover:bg-sky-600">
            <Play size={11} aria-hidden /> {t('playground.ui.run')}
          </button>
        </span>
      </div>
      <CodeEditor value={code} onChange={setCode} onRun={() => setRan(code)} errorLine={result.error?.line} label={t('playground.ui.editor')} tall={full} />
    </div>
  )

  const output = (
    <div aria-live="polite" className="rounded-xl border border-slate-200 bg-slate-50 p-3">
      <div className="mb-1 flex items-center gap-1.5 text-[11px] font-bold text-slate-400">
        <TerminalSquare size={13} aria-hidden /> {t('playground.ui.output')}
      </div>
      {result.stdout && <pre className="max-h-72 overflow-auto whitespace-pre font-mono text-xs leading-5 text-slate-700">{result.stdout}</pre>}
      {result.outDisplay?.type === 'table' ? (
        <TableView table={result.outDisplay.table} />
      ) : (
        result.out && <pre className="max-h-72 overflow-auto font-mono text-xs leading-5 text-sky-800">{result.out}</pre>
      )}
      {result.error && (
        <p className="whitespace-pre-wrap font-mono text-xs text-rose-600">
          {result.error.line ? `${t('playground.ui.line', { n: result.error.line })} · ` : ''}
          <strong>{result.error.type}</strong>: {result.error.message}
        </p>
      )}
      {!result.stdout && !result.out && !result.error && <p className="text-xs italic text-slate-400">{t('playground.ui.no_output')}</p>}
    </div>
  )

  // the final figures repeat the selected figure step, so they start folded when such a step is shown
  const figureStep = selected?.kind === 'event' && selected.ev.type === 'figure'
  const displays = result.displays.length > 0 && (
    <details key={`${ran}:${figureStep}`} open={!figureStep || full} className="group space-y-3">
      <summary className="cursor-pointer text-[11px] font-bold text-slate-400 hover:text-slate-600">{t('playground.ui.figures', { count: result.displays.length })}</summary>
      <div className="mt-3 space-y-3">
        {result.displays.map((d: Display, k) => (d.type === 'figure' ? <FigureView key={`${ran}#${k}`} fig={d.fig} /> : <TableView key={k} table={d.table} />))}
      </div>
    </details>
  )

  const stepsView = (
    <div className="space-y-3">
      {steps.length > 0 && (
        <div className="flex flex-wrap items-center gap-1" aria-label={t('playground.ui.steps')}>
          <span className="mr-1 text-[11px] font-bold text-slate-400">{t('playground.ui.steps')}</span>
          {steps.map((s) => (
            <button
              key={s.seq}
              title={s.kind === 'call' ? s.call.code : s.ev.code}
              aria-pressed={s.seq === selectedSeq}
              onClick={() => setPicked({ code: ran, seq: s.seq })}
              className={`max-w-[16rem] truncate rounded-md border px-1.5 py-0.5 font-mono text-[11px] ${s.seq === selectedSeq ? 'border-slate-900 bg-slate-900 text-white' : 'border-slate-200 text-slate-500 hover:border-slate-400'}`}
            >
              L{stepLine(s)} {stepLabel(s)}
            </button>
          ))}
        </div>
      )}
      {notes.map((n) => {
        const here = selected !== null && n.line === stepLine(selected)
        return (
          <p key={n.id} className={`flex gap-2 rounded-xl px-3 py-2 text-xs leading-relaxed ${n.tone === 'warn' ? 'bg-amber-50 text-amber-900' : 'bg-sky-50 text-sky-900'} ${here ? 'ring-2 ring-sky-300' : ''}`}>
            {n.tone === 'warn' ? <AlertTriangle size={14} className="mt-0.5 shrink-0" aria-hidden /> : <Info size={14} className="mt-0.5 shrink-0" aria-hidden />}
            <span><b className="font-mono">L{n.line}</b> {t(`playground.note.${n.key}`, n.params)}</span>
          </p>
        )
      })}
      {selected ? (
        <StepView key={`${ran}#${selected.seq}`} step={selected.kind === 'call' ? { kind: 'call', call: selected.call } : { kind: 'event', ev: selected.ev }} />
      ) : (
        !result.error && <p className="text-sm text-slate-400">{t('playground.ui.nothing')}</p>
      )}
    </div>
  )

  if (full) {
    return createPortal(
      <div ref={rootRef} role="dialog" aria-modal="true" aria-label={t(`${ns}.title`)} className="fixed inset-0 z-[70] flex flex-col bg-slate-50">
        <div className="border-b border-slate-200 bg-white px-4 py-3 sm:px-6">{header}</div>
        <div className="grid min-h-0 flex-1 gap-4 overflow-y-auto p-4 sm:p-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:overflow-hidden">
          <div className="flex min-h-0 flex-col gap-4 lg:overflow-y-auto lg:pr-1">
            {catalog}
            {editor}
            {output}
          </div>
          <div className="min-h-0 space-y-4 rounded-2xl border border-slate-200 bg-white p-4 lg:overflow-y-auto">
            {stepsView}
            {displays}
          </div>
        </div>
      </div>,
      document.body,
    )
  }

  return (
    <aside ref={rootRef} aria-label={t(`${ns}.title`)} className="space-y-4 rounded-[1.5rem] border border-slate-200 bg-white p-4 shadow-xl shadow-slate-200/40 sm:p-5">
      {header}
      {catalog}
      {editor}
      {stepsView}
      {output}
      {displays}
      <p className="text-[11px] leading-relaxed text-slate-400">{t(`${ns}.sandbox_note`)}</p>
    </aside>
  )
}
