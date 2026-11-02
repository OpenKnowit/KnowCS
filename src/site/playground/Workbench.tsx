import { useEffect, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { AlertTriangle, ChartArea, ChevronLeft, ChevronRight, FileCode, Info, NotebookText, PanelLeftClose, PanelLeftOpen, Play, RotateCcw, Search, Square } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import type { CallTrace } from '../../lib/minipy'
import type { Display, PyEvent } from '../../lib/pyEvents'
import { CodeEditor } from './CodeEditor'
import { StepView, TableView } from './views'
import { FigureView } from './FigureView'
import { KIND } from './kinds'
import type { StepKind } from './kinds'
import { RealConsole, RealFigures } from './RealOutput'
import { realErrorLine, useEngineChoice, useEngineState, useRealRun, useWarmEngine } from './useEngine'
import { LIBS } from './libs'
import type { BenchEntry, BenchProps } from './libs'

// /playground/ as a workbench: examples on the left (folds away), code in the middle,
// the traced step or figure on the top right, output / explanation / variables below it.

type Step = { kind: 'call'; seq: number; call: CallTrace } | { kind: 'event'; seq: number; ev: PyEvent }
type Tab = 'output' | 'explain' | 'vars'

const stepLine = (s: Step) => (s.kind === 'call' ? s.call.line : s.ev.line)
const stepApi = (s: Step) => (s.kind === 'call' ? s.call.api : s.ev.type)
const stepKind = (s: Step): StepKind => (s.kind === 'call' ? 'call' : s.ev.type === 'note' ? 'call' : s.ev.type)
const shorten = (code: string, n = 24) => {
  const one = code.replace(/\s+/g, ' ').trim()
  return one.length > n ? `${one.slice(0, n - 1)}…` : one
}
const stepLabel = (s: Step): string => {
  if (s.kind === 'call') return s.call.api.startsWith('op:') ? s.call.code : s.call.api.replace(/^ndarray\./, '.')
  return shorten(s.ev.code || s.ev.type)
}

export default function Workbench({ lib, spec, side, onSide, onPickLib }: BenchProps) {
  const { t } = useTranslation()
  const current = LIBS.find((l) => l.id === lib)!
  const start = spec.entries.find((e) => e.id === spec.start) ?? spec.entries[0]
  const [entry, setEntry] = useState<BenchEntry>(start)
  const [code, setCode] = useState(start.code)
  const [ran, setRan] = useState(start.code)
  const [picked, setPicked] = useState<{ code: string; seq: number } | null>(null)
  const [openCats, setOpenCats] = useState<Set<string>>(() => new Set([start.cat]))
  const [query, setQuery] = useState('')
  const [tab, setTab] = useState<Tab>('output')

  // re-run after a pause in typing
  useEffect(() => {
    if (code === ran) return
    const id = setTimeout(() => setRan(code), spec.debounce)
    return () => clearTimeout(id)
  }, [code, ran, spec.debounce])

  // minipy traces every step; real Python (Pyodide) runs the code as is where the library has a browser build
  const [engine, setEngine] = useEngineChoice()
  const realOn = spec.realOk && engine === 'real'
  const engineState = useEngineState()
  useWarmEngine(spec.realOk)
  const real = useRealRun(ran, realOn)
  const result = useMemo(() => spec.run(realOn ? '' : ran), [spec, ran, realOn])
  const runNow = () => (code !== ran ? setRan(code) : realOn && real.rerun())

  const steps = useMemo<Step[]>(() => {
    const calls = spec.callApis ? result.calls.filter((c) => spec.callApis!.some((p) => c.api.startsWith(p))) : result.calls
    return [
      ...calls.map((call): Step => ({ kind: 'call', seq: call.seq, call })),
      ...result.events.filter((ev) => ev.type !== 'note').map((ev): Step => ({ kind: 'event', seq: ev.seq, ev })),
    ].sort((a, b) => a.seq - b.seq)
  }, [result, spec.callApis])
  const notes = result.events.filter((e): e is PyEvent & { type: 'note' } => e.type === 'note')
  const autoSeq = useMemo(() => {
    if (!steps.length) return null
    const hit = entry.focus && (entry.focusFirst ? steps : [...steps].reverse()).find((s) => entry.focus!.includes(stepApi(s)))
    return (hit || steps[steps.length - 1]).seq
  }, [steps, entry])
  const selectedSeq = picked && picked.code === ran ? picked.seq : autoSeq
  const selIndex = steps.findIndex((s) => s.seq === selectedSeq)
  const selected = steps[selIndex] ?? null

  const open = (e: BenchEntry) => {
    setEntry(e)
    setCode(e.code)
    setRan(e.code)
    setPicked(null)
  }

  const stripRef = useRef<HTMLDivElement>(null)
  const go = (k: number) => {
    const s = steps[Math.max(0, Math.min(steps.length - 1, k))]
    if (!s) return
    setPicked({ code: ran, seq: s.seq })
    requestAnimationFrame(() => stripRef.current?.querySelector<HTMLElement>(`[data-seq="${s.seq}"]`)?.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' }))
  }
  const onStripKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const k = e.key === 'ArrowRight' ? selIndex + 1 : e.key === 'ArrowLeft' ? selIndex - 1 : e.key === 'Home' ? 0 : e.key === 'End' ? steps.length - 1 : null
    if (k === null) return
    e.preventDefault()
    go(k)
    requestAnimationFrame(() => stripRef.current?.querySelector<HTMLElement>('[aria-pressed="true"]')?.focus())
  }

  const q = query.trim().toLowerCase()
  const hit = (e: BenchEntry) => !q || e.label.toLowerCase().includes(q) || t(spec.explainKey(e.id)).toLowerCase().includes(q)
  const toggleCat = (c: string) =>
    setOpenCats((s) => {
      const n = new Set(s)
      if (n.has(c)) n.delete(c)
      else n.add(c)
      return n
    })

  // ---------------------------------------------------------------- left: libraries → groups → examples
  const sidebar = (
    <aside className="hidden w-[272px] shrink-0 flex-col border-r border-slate-200 bg-white lg:flex" aria-label={t('site.playground.examples')}>
      <div className="flex items-center justify-between py-3 pl-4 pr-3">
        <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-slate-500">{t('site.playground.examples')}</span>
        <button type="button" onClick={() => onSide(false)} className="grid h-8 w-8 place-items-center rounded-lg text-slate-500 hover:bg-slate-100 hover:text-slate-900" aria-label={t('site.playground.hide_libs')} title={t('site.playground.hide_libs')}>
          <PanelLeftClose size={16} aria-hidden />
        </button>
      </div>
      <label className="mx-3 mb-2 flex h-9 items-center gap-2 rounded-lg border border-slate-200 px-2.5 focus-within:border-blue-400">
        <Search size={14} className="shrink-0 text-slate-500" aria-hidden />
        <input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t('site.playground.filter')} aria-label={t('site.playground.filter')} className="min-w-0 flex-1 bg-transparent text-[13px] outline-none placeholder:text-slate-500" />
      </label>
      <nav className="min-h-0 flex-1 overflow-y-auto px-2 pb-4">
        {LIBS.map((l) => {
          const on = l.id === lib
          return (
            <div key={l.id}>
              <button
                type="button"
                onClick={() => !on && onPickLib(l.id)}
                aria-current={on ? 'true' : undefined}
                className={`flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left ${on ? '' : 'hover:bg-slate-50'}`}
              >
                <ChevronRight size={14} className={`shrink-0 text-slate-500 transition-transform ${on ? 'rotate-90' : ''}`} aria-hidden />
                <i className="h-2 w-2 shrink-0 rounded-full" style={{ background: l.color }} aria-hidden />
                <span className={`flex-1 text-[13px] ${on ? 'font-bold text-slate-900' : 'font-semibold text-slate-600'}`}>{l.name}</span>
                {on && <span className="font-mono text-[11px] text-slate-500">{spec.entries.length}</span>}
              </button>
              {on && (
                <div className="mb-1">
                  {spec.cats.map((c) => {
                    const list = spec.entries.filter((e) => e.cat === c && hit(e))
                    if (!list.length) return null
                    const expanded = q !== '' || openCats.has(c)
                    return (
                      <div key={c}>
                        <button type="button" onClick={() => toggleCat(c)} aria-expanded={expanded} className="flex w-full items-center gap-1.5 rounded-md py-1.5 pl-7 pr-2 text-left text-[11px] font-bold uppercase tracking-wide text-slate-500 hover:text-slate-800">
                          <ChevronRight size={12} className={`shrink-0 transition-transform ${expanded ? 'rotate-90' : ''}`} aria-hidden />
                          <span className="flex-1 truncate">{t(spec.catKey(c))}</span>
                          <span className="font-mono font-normal">{list.length}</span>
                        </button>
                        {expanded &&
                          list.map((e) => (
                            <button
                              key={e.id}
                              type="button"
                              onClick={() => open(e)}
                              aria-pressed={e.id === entry.id}
                              className={`block w-full truncate rounded-md py-1.5 pl-12 pr-2 text-left font-mono text-[12px] ${e.id === entry.id ? 'bg-blue-50 font-semibold text-blue-700' : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'}`}
                            >
                              {e.label}
                            </button>
                          ))}
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )
        })}
      </nav>
    </aside>
  )

  // ---------------------------------------------------------------- toolbar
  const engineStatus = realOn
    ? { text: t(`playground.ui.engine.real_${engineState}`), dot: engineState === 'ready' ? 'bg-emerald-500' : engineState === 'failed' ? 'bg-rose-500' : 'bg-amber-500', tone: engineState === 'ready' ? 'bg-emerald-50 text-emerald-800' : engineState === 'failed' ? 'bg-rose-50 text-rose-800' : 'bg-amber-50 text-amber-800' }
    : { text: t('playground.ui.engine.mini_sub'), dot: 'bg-sky-500', tone: 'bg-sky-50 text-sky-800' }
  const toolbar = (
    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-white px-4 py-2.5">
      <div className="flex min-w-0 items-center gap-2.5">
        {!side && (
          <button type="button" onClick={() => onSide(true)} className="hidden h-9 w-9 shrink-0 place-items-center rounded-[9px] border border-slate-200 text-slate-600 hover:text-slate-900 lg:grid" aria-label={t('site.playground.show_libs')} title={t('site.playground.show_libs')}>
            <PanelLeftOpen size={17} aria-hidden />
          </button>
        )}
        <div className="flex min-w-0 items-center gap-2 rounded-[9px] border border-slate-200 px-3 py-2">
          <NotebookText size={16} className="shrink-0 text-blue-600" aria-hidden />
          <span className="text-sm font-medium text-slate-500">{current.name} /</span>
          <span className="truncate font-mono text-sm font-bold text-slate-900">{entry.label}</span>
        </div>
        {/* below lg the sidebar is gone: pick an example here instead */}
        <select
          value={entry.id}
          onChange={(e) => open(spec.entries.find((x) => x.id === e.target.value)!)}
          aria-label={t('site.playground.examples')}
          className="h-9 max-w-[10rem] rounded-[9px] border border-slate-200 bg-white px-2 text-sm lg:hidden"
        >
          {spec.cats.map((c) => (
            <optgroup key={c} label={t(spec.catKey(c))}>
              {spec.entries.filter((e) => e.cat === c).map((e) => (
                <option key={e.id} value={e.id}>{e.label}</option>
              ))}
            </optgroup>
          ))}
        </select>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <div className="inline-flex rounded-[9px] bg-slate-100 p-[3px]" role="radiogroup" aria-label={t('playground.ui.engine.label')}>
          {(['mini', 'real'] as const).map((e) => {
            const disabled = e === 'real' && !spec.realOk
            const on = (e === 'real') === realOn
            return (
              <button
                key={e}
                type="button"
                role="radio"
                aria-checked={on}
                disabled={disabled}
                title={disabled ? t('playground.ui.engine.unavailable', { name: current.name }) : undefined}
                onClick={() => setEngine(e)}
                className={`rounded-md px-2.5 py-1 text-[12px] transition disabled:cursor-not-allowed disabled:opacity-40 ${on ? 'bg-white font-semibold text-slate-900 shadow-sm' : 'font-medium text-slate-500 hover:text-slate-800'}`}
              >
                {t(`playground.ui.engine.${e}`)}
              </button>
            )
          })}
        </div>
        <span className={`hidden items-center gap-2 rounded-full px-3 py-1.5 text-xs font-semibold xl:inline-flex ${engineStatus.tone}`}>
          <i className={`h-2 w-2 rounded-full ${engineStatus.dot}`} aria-hidden />
          {engineStatus.text}
        </span>
        <button type="button" onClick={() => setCode(entry.code)} disabled={code === entry.code} className="flex items-center gap-1.5 rounded-[9px] border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-600 hover:text-slate-900 disabled:opacity-40">
          <RotateCcw size={15} aria-hidden /> {t('playground.ui.reset')}
        </button>
        {real.busy ? (
          <button type="button" onClick={real.stop} className="flex items-center gap-2 rounded-[9px] bg-rose-600 px-4 py-2 text-sm font-bold text-white hover:bg-rose-500">
            <Square size={14} aria-hidden /> {t('playground.ui.real.stop')}
          </button>
        ) : (
          <button type="button" onClick={runNow} className="flex items-center gap-2 rounded-[9px] bg-blue-600 px-4 py-2 text-sm font-bold text-white hover:bg-blue-700">
            <Play size={15} aria-hidden /> {t('playground.ui.run')} <kbd className="hidden font-sans text-xs font-medium opacity-80 sm:inline">⌘↵</kbd>
          </button>
        )}
      </div>
    </div>
  )

  // ---------------------------------------------------------------- right: the traced step (or real figures)
  const figureStep = selected?.kind === 'event' && selected.ev.type === 'figure'
  const visual = (
    <section className="flex min-h-[22rem] flex-col lg:min-h-0 lg:flex-1" aria-label={t('site.playground.visual')}>
      <div className="flex items-center justify-between gap-3 border-b border-slate-200 px-4 py-2.5">
        <span className="flex items-center gap-2 text-[13px] font-bold text-slate-900">
          <ChartArea size={15} className="text-blue-600" aria-hidden /> {t('site.playground.visual')}
        </span>
        {!realOn && steps.length > 0 && (
          <span className="flex items-center gap-1.5">
            <span className="font-mono text-[11px] text-slate-500">
              {t('playground.ui.steps')} {selIndex + 1} / {steps.length}
            </span>
            <button type="button" onClick={() => go(selIndex - 1)} disabled={selIndex <= 0} aria-label={t('playground.ui.prev')} className="grid h-7 w-7 place-items-center rounded-md border border-slate-200 text-slate-600 hover:border-slate-400 disabled:opacity-40">
              <ChevronLeft size={15} aria-hidden />
            </button>
            <button type="button" onClick={() => go(selIndex + 1)} disabled={selIndex >= steps.length - 1} aria-label={t('playground.ui.next')} className="grid h-7 w-7 place-items-center rounded-md border border-slate-200 text-slate-600 hover:border-slate-400 disabled:opacity-40">
              <ChevronRight size={15} aria-hidden />
            </button>
          </span>
        )}
      </div>
      {!realOn && steps.length > 1 && (
        <div ref={stripRef} className="flex gap-1.5 overflow-x-auto border-b border-slate-100 px-4 py-2" role="group" aria-label={t('playground.ui.steps')} onKeyDown={onStripKey}>
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
                className={`flex max-w-[13rem] shrink-0 items-center gap-1.5 rounded-lg border px-2 py-1 font-mono text-[11px] transition ${on ? `${K.on} text-white` : 'border-slate-200 bg-white text-slate-600 hover:border-slate-400'}`}
              >
                <K.icon size={12} className={on ? 'text-white' : K.icon_} aria-hidden />
                <span className={`rounded px-1 text-[10px] ${on ? 'bg-white/20' : 'bg-slate-100 text-slate-500'}`}>L{stepLine(s)}</span>
                <span className="truncate">{stepLabel(s)}</span>
              </button>
            )
          })}
        </div>
      )}
      <div className="min-h-0 flex-1 space-y-4 overflow-auto p-4">
        {realOn ? (
          <RealFigures run={real.run} />
        ) : (
          <>
            {selected ? (
              <StepView key={`${ran}#${selected.seq}`} step={selected.kind === 'call' ? { kind: 'call', call: selected.call } : { kind: 'event', ev: selected.ev }} />
            ) : (
              !result.error && <p className="text-sm text-slate-500">{t('playground.ui.nothing')}</p>
            )}
            {!figureStep && result.displays.map((d: Display, k) => (d.type === 'figure' ? <FigureView key={`${ran}#${k}`} fig={d.fig} /> : <TableView key={k} table={d.table} />))}
          </>
        )}
      </div>
    </section>
  )

  // ---------------------------------------------------------------- right, below: output / explain / variables
  const tabs: Tab[] = ['output', 'explain', 'vars']
  const console_ = realOn ? (
    <RealConsole run={real.run} />
  ) : (
    <div aria-live="polite" className="space-y-1.5 font-mono text-[13px] leading-6">
      {result.stdout && <pre className="whitespace-pre text-slate-800">{result.stdout}</pre>}
      {result.outDisplay?.type === 'table' ? <TableView table={result.outDisplay.table} /> : result.out && <pre className="whitespace-pre text-sky-800">{result.out}</pre>}
      {result.error && (
        <p className="whitespace-pre-wrap text-rose-600">
          {result.error.line ? `${t('playground.ui.line', { n: result.error.line })} · ` : ''}
          <strong>{result.error.type}</strong>: {result.error.message}
        </p>
      )}
      {!result.stdout && !result.out && !result.error && <p className="font-sans text-sm italic text-slate-500">{t('playground.ui.no_output')}</p>}
    </div>
  )
  const explain = (
    <div className="space-y-2.5 text-sm leading-relaxed text-slate-700">
      <p>
        <b className="font-mono text-slate-900">{entry.label}</b> — {t(spec.explainKey(entry.id))}
      </p>
      {!realOn &&
        notes.map((n) => (
          <p key={n.id} className={`flex gap-2 rounded-lg px-3 py-2 text-[13px] ${n.tone === 'warn' ? 'bg-amber-50 text-amber-900' : 'bg-sky-50 text-sky-900'}`}>
            {n.tone === 'warn' ? <AlertTriangle size={14} className="mt-0.5 shrink-0" aria-hidden /> : <Info size={14} className="mt-0.5 shrink-0" aria-hidden />}
            <span>
              <b className="font-mono">L{n.line}</b> {t(`playground.note.${n.key}`, n.params)}
            </span>
          </p>
        ))}
      <p className="text-xs text-slate-500">{t(spec.noteKey)}</p>
    </div>
  )
  const vars = realOn ? (
    <p className="text-sm text-slate-500">{t('playground.ui.engine.steps_off')}</p>
  ) : result.vars.length ? (
    <table className="w-full text-left text-[12.5px]">
      <thead>
        <tr className="border-b border-slate-200 text-[11px] uppercase tracking-wide text-slate-500">
          {['name', 'type', 'shape', 'value'].map((h) => (
            <th key={h} className="py-1.5 pr-3 font-bold">{t(`site.playground.var_${h}`)}</th>
          ))}
        </tr>
      </thead>
      <tbody className="font-mono">
        {result.vars.map((v) => (
          <tr key={v.name} className="border-b border-slate-100 align-top">
            <td className="py-1.5 pr-3 font-semibold text-slate-900">{v.name}</td>
            <td className="py-1.5 pr-3 text-slate-600">{v.kind}{v.dtype ? ` · ${v.dtype}` : ''}</td>
            <td className="py-1.5 pr-3 text-slate-600">{v.shape ? `(${v.shape.join(', ')}${v.shape.length === 1 ? ',' : ''})` : '—'}</td>
            <td className="max-w-[16rem] truncate py-1.5 text-slate-600" title={v.preview}>{v.preview}</td>
          </tr>
        ))}
      </tbody>
    </table>
  ) : (
    <p className="text-sm text-slate-500">{t('site.playground.no_vars')}</p>
  )
  const output = (
    <section className="flex h-[260px] shrink-0 flex-col border-t border-slate-200">
      <div className="flex gap-1 border-b border-slate-200 px-3" role="tablist" aria-label={t('playground.ui.output')}>
        {tabs.map((k) => (
          <button
            key={k}
            type="button"
            role="tab"
            aria-selected={tab === k}
            onClick={() => setTab(k)}
            className={`-mb-px border-b-2 px-2.5 py-2.5 text-[13px] ${tab === k ? 'border-blue-600 font-bold text-slate-900' : 'border-transparent font-medium text-slate-500 hover:text-slate-800'}`}
          >
            {t(`site.playground.tab_${k}`)}
          </button>
        ))}
      </div>
      <div role="tabpanel" className="min-h-0 flex-1 overflow-auto px-4 py-3">
        {tab === 'output' ? console_ : tab === 'explain' ? explain : vars}
      </div>
    </section>
  )

  return (
    <div className="flex border-t border-slate-200 bg-slate-50 lg:h-[calc(100vh-67px)] lg:min-h-[620px]">
      {side && sidebar}
      <div className="flex min-w-0 flex-1 flex-col">
        {toolbar}
        <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
          <div className="flex min-h-[20rem] min-w-0 flex-1 flex-col bg-slate-900">
            <div className="flex h-10 items-end bg-slate-800 px-2">
              <span className="flex items-center gap-2 rounded-t-lg bg-slate-900 px-3.5 py-2 font-mono text-xs font-semibold text-slate-200">
                <FileCode size={14} className="text-sky-300" aria-hidden /> main.py
              </span>
              <span className="ml-auto hidden pb-2 pr-2 text-[11px] text-slate-400 sm:inline">{t('playground.ui.shortcut')}</span>
            </div>
            <div className="flex min-h-0 flex-1 flex-col overflow-auto p-3 [&>div]:rounded-lg [&>div]:border-0">
              <CodeEditor value={code} onChange={setCode} onRun={runNow} errorLine={realOn ? realErrorLine(real.run) : result.error?.line} label={t('playground.ui.editor')} tall minRows={14} />
            </div>
          </div>
          <div className="flex min-w-0 flex-col border-slate-200 bg-white lg:w-[46%] lg:max-w-[640px] lg:border-l">
            {visual}
            {output}
          </div>
        </div>
      </div>
    </div>
  )
}
