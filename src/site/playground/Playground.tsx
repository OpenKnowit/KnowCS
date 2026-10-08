import { useEffect, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { AlertTriangle, ChevronLeft, ChevronRight, Info, Maximize2, Minimize2, Play, RotateCcw, Search, Sparkles, Square, TerminalSquare } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { createPortal } from 'react-dom'
import type { CallTrace } from '../../lib/minipy'
import { runIn } from '../../data/playgrounds/run'
import type { Display, PyEvent } from '../../lib/pyEvents'
import type { PlayConfig, PlayEntry } from '../../data/playgrounds/types'
import { CodeEditor } from './CodeEditor'
import { StepView, TableView } from './views'
import { KIND } from './kinds'
import type { StepKind } from './kinds'
import { FigureView } from './FigureView'
import { useFullScreen } from './useFullScreen'
import { REAL_PYTHON_LIBS } from '../../lib/pyodide/config'
import { EngineSwitch } from './EngineSwitch'
import { RealConsole, RealFigures } from './RealOutput'
import { realErrorLine, useEngineChoice, useRealRun, useWarmEngine } from './useEngine'

// --- 各库实验台：示例目录 + 可编辑代码 + 逐步可视化（调用 / 图 / 模型形状 / 计算图 …）+ 全屏 ---

type Step = { kind: 'call'; seq: number; call: CallTrace } | { kind: 'event'; seq: number; ev: PyEvent }

const stepLine = (s: Step) => (s.kind === 'call' ? s.call.line : s.ev.line)
const stepApi = (s: Step) => (s.kind === 'call' ? s.call.api : s.ev.type)

const shorten = (code: string, n = 26) => {
  const one = code.replace(/\s+/g, ' ').trim()
  return one.length > n ? `${one.slice(0, n - 1)}…` : one
}

const stepKind = (s: Step): StepKind => (s.kind === 'call' ? 'call' : s.ev.type === 'note' ? 'call' : s.ev.type)

const stepLabel = (s: Step): string => {
  if (s.kind === 'call') return s.call.api.startsWith('op:') ? s.call.code : s.call.api.replace(/^ndarray\./, '.')
  return shorten(s.ev.code || s.ev.type)
}

const LIB_NAME: Record<string, string> = { matplotlib: 'matplotlib', pandas: 'pandas', pytorch: 'PyTorch', keras: 'Keras', tensorflow: 'TensorFlow' }

interface Props {
  config: PlayConfig
  /** entry to open first (a link from the note) */
  initialEntry?: string | null
  /** code to open first (a code block from the note) */
  initialCode?: string | null
  /** two columns on wide screens (the Playground page) instead of the narrow panel beside a note */
  wide?: boolean
}

export default function Playground({ config, initialEntry, initialCode, wide }: Props) {
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

  // the engine: minipy traces every step; real Python (Pyodide, only where the library has a browser build) runs it as is
  const realOk = REAL_PYTHON_LIBS.has(config.id)
  const [engine, setEngine] = useEngineChoice()
  const realOn = realOk && engine === 'real'
  useWarmEngine(realOk)
  const real = useRealRun(ran, realOn)
  const result = useMemo(() => runIn(config, realOn ? '' : ran), [config, ran, realOn])
  const runNow = () => (code !== ran ? setRan(code) : realOn && real.rerun())

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
    const hit = want && (entry?.focusFirst ? steps : [...steps].reverse()).find((s) => want.includes(stepApi(s)))
    return (hit ?? steps[steps.length - 1]).seq
  }, [steps, entry?.focus, entry?.focusFirst])
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
    <div className={full ? 'flex min-h-0 flex-1 flex-col gap-3' : 'space-y-3'}>
      <EngineSwitch engine={engine} onChange={setEngine} available={realOk} lib={LIB_NAME[config.id] ?? config.id} />
      <div className={full ? 'flex min-h-0 flex-1 flex-col' : ''}>
      <div className="mb-1 flex items-center justify-between gap-2 text-[11px] font-bold text-slate-400">
        <span>{t('playground.ui.editor')}</span>
        <span className="flex items-center gap-2">
          {code !== original && (
            <button type="button" onClick={() => setCode(original)} className="flex items-center gap-1 rounded-md px-1.5 py-0.5 text-slate-500 hover:bg-slate-100 hover:text-slate-800">
              <RotateCcw size={11} aria-hidden /> {t('playground.ui.reset')}
            </button>
          )}
          {real.busy ? (
            <button type="button" onClick={real.stop} className="flex items-center gap-1 rounded-md bg-rose-700 px-2 py-0.5 text-white hover:bg-rose-600">
              <Square size={10} aria-hidden /> {t('playground.ui.real.stop')}
            </button>
          ) : (
            <button type="button" onClick={runNow} className={`flex items-center gap-1 rounded-md px-2 py-0.5 text-white ${realOn ? 'bg-amber-700 hover:bg-amber-800' : 'bg-sky-700 hover:bg-sky-600'}`}>
              <Play size={11} aria-hidden /> {t('playground.ui.run')}
            </button>
          )}
        </span>
      </div>
      <CodeEditor value={code} onChange={setCode} onRun={runNow} errorLine={realOn ? realErrorLine(real.run) : result.error?.line} label={t('playground.ui.editor')} tall={full} />
      <p className="mt-1 hidden text-[11px] text-slate-500 sm:block">{t('playground.ui.shortcut')}</p>
      </div>
    </div>
  )

  const output = realOn ? (
    <RealConsole run={real.run} />
  ) : (
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
  const displays = realOn ? (
    <RealFigures run={real.run} />
  ) : result.displays.length > 0 && (
    <details key={`${ran}:${figureStep}`} open={!figureStep || full} className="group space-y-3">
      <summary className="cursor-pointer text-[11px] font-bold text-slate-400 hover:text-slate-600">{t('playground.ui.figures', { count: result.displays.length })}</summary>
      <div className="mt-3 space-y-3">
        {result.displays.map((d: Display, k) => (d.type === 'figure' ? <FigureView key={`${ran}#${k}`} fig={d.fig} /> : <TableView key={k} table={d.table} />))}
      </div>
    </details>
  )

  const stripRef = useRef<HTMLDivElement>(null)
  const selIndex = steps.findIndex((s) => s.seq === selectedSeq)
  const go = (k: number) => {
    const s = steps[Math.max(0, Math.min(steps.length - 1, k))]
    if (!s) return
    setPicked({ code: ran, seq: s.seq })
    // keep the chosen chip in view without scrolling the page
    requestAnimationFrame(() => {
      const strip = stripRef.current
      const chip = strip?.querySelector<HTMLElement>(`[data-seq="${s.seq}"]`)
      if (strip && chip) strip.scrollTo({ left: chip.offsetLeft - strip.clientWidth / 2 + chip.clientWidth / 2, behavior: 'smooth' })
    })
  }
  const onStripKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const k = e.key === 'ArrowRight' ? selIndex + 1 : e.key === 'ArrowLeft' ? selIndex - 1 : e.key === 'Home' ? 0 : e.key === 'End' ? steps.length - 1 : null
    if (k === null) return
    e.preventDefault()
    go(k)
    requestAnimationFrame(() => stripRef.current?.querySelector<HTMLElement>('[aria-pressed="true"]')?.focus())
  }

  const stepsView = realOn ? null : (
    <div className="space-y-3">
      {steps.length > 0 && (
        <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-2">
          <div className="mb-1.5 flex items-center justify-between gap-2 px-1">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
              {t('playground.ui.steps')} <span className="font-mono normal-case tracking-normal text-slate-600">{selIndex + 1} / {steps.length}</span>
            </span>
            <span className="flex gap-1">
              <button type="button" onClick={() => go(selIndex - 1)} disabled={selIndex <= 0} aria-label={t('playground.ui.prev')} className="grid h-7 w-7 place-items-center rounded-lg border border-slate-200 bg-white text-slate-600 hover:border-slate-400 disabled:opacity-40">
                <ChevronLeft size={15} aria-hidden />
              </button>
              <button type="button" onClick={() => go(selIndex + 1)} disabled={selIndex >= steps.length - 1} aria-label={t('playground.ui.next')} className="grid h-7 w-7 place-items-center rounded-lg border border-slate-200 bg-white text-slate-600 hover:border-slate-400 disabled:opacity-40">
                <ChevronRight size={15} aria-hidden />
              </button>
            </span>
          </div>
          <div ref={stripRef} className="relative flex gap-1.5 overflow-x-auto pb-1" role="group" aria-label={t('playground.ui.steps')} onKeyDown={onStripKey}>
            {steps.map((s, k) => {
              const K = KIND[stepKind(s)]
              const on = s.seq === selectedSeq
              return (
                <button
                  key={s.seq}
                  data-seq={s.seq}
                  type="button"
                  title={s.kind === 'call' ? s.call.code : s.ev.code}
                  aria-pressed={on}
                  tabIndex={on || (selIndex < 0 && k === 0) ? 0 : -1}
                  onClick={() => go(k)}
                  className={`group flex max-w-[15rem] shrink-0 items-center gap-1.5 rounded-xl border px-2 py-1.5 text-left font-mono text-[11px] transition ${on ? `${K.on} text-white shadow-md` : 'border-slate-200 bg-white text-slate-600 hover:border-slate-400'}`}
                >
                  <K.icon size={13} className={on ? 'text-white' : K.icon_} aria-hidden />
                  <span className={`rounded px-1 text-[10px] ${on ? 'bg-white/20' : 'bg-slate-100 text-slate-500'}`}>L{stepLine(s)}</span>
                  <span className="truncate">{stepLabel(s)}</span>
                </button>
              )
            })}
          </div>
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
            {!realOn && output}
          </div>
          <div className="min-h-0 space-y-4 rounded-2xl border border-slate-200 bg-white p-4 lg:overflow-y-auto">
            {stepsView}
            {realOn && output}
            {displays}
          </div>
        </div>
      </div>,
      document.body,
    )
  }

  if (wide) {
    return (
      <section ref={rootRef} aria-label={t(`${ns}.title`)} className="rounded-[1.5rem] border border-slate-200 bg-white p-4 shadow-xl shadow-slate-200/40 sm:p-6">
        {header}
        <div className="mt-4 grid grid-cols-[minmax(0,1fr)] gap-5 lg:grid-cols-2">
          <div className="min-w-0 space-y-4">
            {catalog}
            {editor}
            {!realOn && output}
          </div>
          {/* real Python has no steps: its output sits beside the editor, above its figures */}
          <div className="min-w-0 space-y-4 lg:border-l lg:border-slate-100 lg:pl-5">
            {stepsView}
            {realOn && output}
            {displays}
          </div>
        </div>
        {!realOn && <p className="mt-4 text-[11px] leading-relaxed text-slate-500">{t(`${ns}.sandbox_note`)}</p>}
      </section>
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
      {!realOn && <p className="text-[11px] leading-relaxed text-slate-400">{t(`${ns}.sandbox_note`)}</p>}
    </aside>
  )
}
