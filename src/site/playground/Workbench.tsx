import { Component, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent, PointerEvent as ReactPointerEvent, ReactNode } from 'react'
import { AlertTriangle, ChartArea, ChevronLeft, ChevronRight, FileCode, Info, LayoutGrid, NotebookText, PanelLeftClose, PanelLeftOpen, Play, RotateCcw, Search, Square, X } from 'lucide-react'
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
import type { BenchEntry, BenchProps, LibId, LibSpec } from './libs'

// /playground/ as a workbench: examples on the left (folds away), code in the middle,
// the traced step on the right with output / explanation / variables below it, and a status bar.
// With real Python (Pyodide) there are no steps, so the whole right side is output.

type Step = { kind: 'call'; seq: number; call: CallTrace } | { kind: 'event'; seq: number; ev: PyEvent }
type Tab = 'output' | 'explain' | 'vars'
export type Specs = Partial<Record<LibId, LibSpec>>

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
const entryOf = (spec: LibSpec, id: string | null) => spec.entries.find((e) => e.id === id) ?? spec.entries.find((e) => e.id === spec.start) ?? spec.entries[0]

// one picture per example, taken from its own visual (scripts/playground-thumbs.mjs); examples without one show their code
const THUMBS = import.meta.glob<string>('../../assets/playground/*/*.webp', { eager: true, import: 'default' })
const thumbOf = (lib: LibId, id: string) => THUMBS[`../../assets/playground/${lib}/${id}.webp`]

/** one broken visual (a trace the view cannot draw) must not take the whole workbench down */
class VisualBoundary extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children
  }
}

const SPLIT_KEY = 'knowcs-playground-split'
const readSplit = () => {
  try {
    const v = Number(localStorage.getItem(SPLIT_KEY))
    return v >= 25 && v <= 70 ? v : 46
  } catch {
    return 46
  }
}

// ---------------------------------------------------------------- sidebar: libraries → groups → examples

interface SidebarProps {
  lib: LibId
  entryId: string | null
  specs: Specs
  onPick: (lib: LibId, entry: string | null) => void
  onCollapse: () => void
}

/** Every library, each one folding open on its own (several can be open at once). */
export function ExamplesSidebar({ lib, entryId, specs, onPick, onCollapse }: SidebarProps) {
  const { t } = useTranslation()
  const [query, setQuery] = useState('')
  const [openLibs, setOpenLibs] = useState<Set<LibId>>(() => new Set([lib]))
  const [openCats, setOpenCats] = useState<Set<string>>(() => new Set())
  const spec = specs[lib]
  const curCat = spec ? `${lib}:${entryOf(spec, entryId).cat}` : null
  // the example on screen is always visible in the tree (also when it was picked in the gallery)
  const want = `${lib}|${curCat}`
  const [revealed, setRevealed] = useState<string | null>(null)
  if (revealed !== want) {
    setRevealed(want)
    setOpenLibs((s) => (s.has(lib) ? s : new Set(s).add(lib)))
    if (curCat) setOpenCats((s) => (s.has(curCat) ? s : new Set(s).add(curCat)))
  }
  const flip = <T,>(set: (f: (s: Set<T>) => Set<T>) => void, k: T) =>
    set((s) => {
      const n = new Set(s)
      if (n.has(k)) n.delete(k)
      else n.add(k)
      return n
    })
  const q = query.trim().toLowerCase()
  const current = spec ? entryOf(spec, entryId).id : null
  return (
    <aside className="hidden w-[272px] shrink-0 flex-col border-r border-slate-200 bg-white lg:flex" aria-label={t('site.playground.examples')}>
      <div className="flex items-center justify-between py-3 pl-4 pr-3">
        <span className="text-[11px] font-bold uppercase tracking-[0.08em] text-slate-500">{t('site.playground.examples')}</span>
        <button type="button" onClick={onCollapse} className="grid h-8 w-8 place-items-center rounded-lg text-slate-500 hover:bg-slate-100 hover:text-slate-900" aria-label={t('site.playground.hide_libs')} title={t('site.playground.hide_libs')}>
          <PanelLeftClose size={16} aria-hidden />
        </button>
      </div>
      <label className="mx-3 mb-2 flex h-9 items-center gap-2 rounded-lg border border-slate-200 px-2.5 focus-within:border-blue-400">
        <Search size={14} className="shrink-0 text-slate-500" aria-hidden />
        <input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t('site.playground.filter')} aria-label={t('site.playground.filter')} className="min-w-0 flex-1 bg-transparent text-[13px] outline-none placeholder:text-slate-500" />
      </label>
      <nav className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-2 pb-4">
        {LIBS.map((l) => {
          const s = specs[l.id]
          const hit = (e: BenchEntry) => !q || e.label.toLowerCase().includes(q) || (s ? t(s.explainKey(e.id)).toLowerCase().includes(q) : false)
          const matches = s ? s.entries.filter(hit) : []
          if (q && !matches.length) return null
          const open = q !== '' || openLibs.has(l.id)
          return (
            <div key={l.id}>
              <button type="button" onClick={() => flip(setOpenLibs, l.id)} aria-expanded={open} className="flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left hover:bg-slate-50">
                <ChevronRight size={14} className={`shrink-0 text-slate-500 transition-transform duration-150 ${open ? 'rotate-90' : ''}`} aria-hidden />
                <i className="h-2 w-2 shrink-0 rounded-full" style={{ background: l.color }} aria-hidden />
                <span className={`flex-1 text-[13px] ${l.id === lib ? 'font-bold text-slate-900' : 'font-semibold text-slate-600'}`}>{l.name}</span>
                {s && <span className="font-mono text-[11px] text-slate-500">{q ? matches.length : s.entries.length}</span>}
              </button>
              {open &&
                (s ? (
                  <div className="mb-1">
                    {s.cats.map((c) => {
                      const list = matches.filter((e) => e.cat === c)
                      if (!list.length) return null
                      const key = `${l.id}:${c}`
                      const expanded = q !== '' || openCats.has(key)
                      return (
                        <div key={c}>
                          <button type="button" onClick={() => flip(setOpenCats, key)} aria-expanded={expanded} className="flex w-full items-center gap-1.5 rounded-md py-1.5 pl-7 pr-2 text-left text-[11px] font-bold uppercase tracking-wide text-slate-500 hover:text-slate-800">
                            <ChevronRight size={12} className={`shrink-0 transition-transform duration-150 ${expanded ? 'rotate-90' : ''}`} aria-hidden />
                            <span className="flex-1 truncate">{t(s.catKey(c))}</span>
                            <span className="font-mono font-normal">{list.length}</span>
                          </button>
                          {expanded &&
                            list.map((e) => {
                              const on = l.id === lib && e.id === current
                              return (
                                <button
                                  key={e.id}
                                  type="button"
                                  data-entry={`${l.id}:${e.id}`}
                                  onClick={() => onPick(l.id, e.id)}
                                  aria-pressed={on}
                                  className={`block w-full truncate rounded-md py-1.5 pl-12 pr-2 text-left font-mono text-[12px] transition-colors ${on ? 'bg-blue-50 font-semibold text-blue-700' : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'}`}
                                >
                                  {e.label}
                                </button>
                              )
                            })}
                        </div>
                      )
                    })}
                  </div>
                ) : (
                  <p className="py-1.5 pl-12 text-xs text-slate-500">{t('playground.ui.loading', { defaultValue: '…' })}</p>
                ))}
            </div>
          )
        })}
      </nav>
    </aside>
  )
}

// ---------------------------------------------------------------- gallery: every example as a picture

interface GalleryProps {
  lib: LibId
  specs: Specs
  onPick: (lib: LibId, entry: string) => void
  onClose: () => void
}

export function Gallery({ lib, specs, onPick, onClose }: GalleryProps) {
  const { t } = useTranslation()
  const closeRef = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    closeRef.current?.focus()
    const onKey = (e: globalThis.KeyboardEvent) => e.key === 'Escape' && onClose()
    document.addEventListener('keydown', onKey)
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = overflow
    }
  }, [onClose])
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-8" role="dialog" aria-modal="true" aria-label={t('site.playground.gallery')}>
      <div className="absolute inset-0 bg-slate-900/40" onClick={onClose} aria-hidden />
      <div className="lang-menu relative flex max-h-full w-full max-w-[1280px] flex-col overflow-hidden rounded-2xl bg-white shadow-2xl shadow-slate-900/30">
        <div className="flex items-center justify-between border-b border-slate-200 px-6 py-4">
          <h2 className="flex items-center gap-2 text-lg font-extrabold text-slate-900">
            <LayoutGrid size={18} className="text-blue-600" aria-hidden /> {t('site.playground.gallery')}
          </h2>
          <button ref={closeRef} type="button" onClick={onClose} className="grid h-9 w-9 place-items-center rounded-[9px] border border-slate-200 text-slate-600 hover:text-slate-900" aria-label={t('site.ui.close')}>
            <X size={18} aria-hidden />
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-6 py-4">
          {LIBS.map((l) => {
            const s = specs[l.id]
            return (
              <details key={l.id} open={l.id === lib} className="group border-b border-slate-100 py-2 last:border-0">
                <summary className="flex cursor-pointer list-none items-center gap-2.5 py-2 text-[17px] font-bold text-slate-800 [&::-webkit-details-marker]:hidden">
                  <ChevronRight size={18} className="text-slate-500 transition-transform group-open:rotate-90" aria-hidden />
                  <i className="h-2.5 w-2.5 rounded-full" style={{ background: l.color }} aria-hidden />
                  {l.name}
                  {s && <span className="font-mono text-xs font-medium text-slate-500">{s.entries.length}</span>}
                </summary>
                {s ? (
                  s.cats.map((c) => (
                    <section key={c} className="mb-4 mt-2">
                      <h3 className="mb-2 text-[11px] font-bold uppercase tracking-wide text-slate-500">{t(s.catKey(c))}</h3>
                      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
                        {s.entries
                          .filter((e) => e.cat === c)
                          .map((e) => {
                            const img = thumbOf(l.id, e.id)
                            return (
                              <button
                                key={e.id}
                                type="button"
                                onClick={() => onPick(l.id, e.id)}
                                className="group/card flex flex-col overflow-hidden rounded-xl border border-slate-200 bg-slate-50 text-left transition hover:-translate-y-0.5 hover:border-blue-300 hover:shadow-lg hover:shadow-blue-100"
                              >
                                <span className="px-3 pb-1.5 pt-2 font-mono text-[12px] font-semibold text-slate-800 group-hover/card:text-blue-700">{e.label}</span>
                                <span className="mx-2 mb-2 block aspect-[8/5] overflow-hidden rounded-lg border border-slate-200 bg-white">
                                  {img ? (
                                    <img src={img} alt="" loading="lazy" decoding="async" className="h-full w-full object-cover object-left-top" />
                                  ) : (
                                    <pre className="h-full overflow-hidden bg-slate-900 p-2.5 font-mono text-[9.5px] leading-[1.45] text-slate-300">{e.code}</pre>
                                  )}
                                </span>
                              </button>
                            )
                          })}
                      </div>
                    </section>
                  ))
                ) : (
                  <p className="pb-3 pl-7 text-sm text-slate-500">{t('playground.ui.loading', { defaultValue: '…' })}</p>
                )}
              </details>
            )
          })}
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- the workbench

export default function Workbench({ lib, spec, entryId, side, onSide, onGallery }: BenchProps) {
  const { t } = useTranslation()
  const current = LIBS.find((l) => l.id === lib)!
  const entry = entryOf(spec, entryId)
  const [code, setCode] = useState(entry.code)
  const [ran, setRan] = useState(entry.code)
  const [picked, setPicked] = useState<{ code: string; seq: number } | null>(null)
  const [tab, setTab] = useState<Tab>('output')
  const [split, setSplit] = useState(readSplit)
  // another example picked (sidebar or gallery): start from its code
  const [shownEntry, setShownEntry] = useState(entry.id)
  if (shownEntry !== entry.id) {
    setShownEntry(entry.id)
    setCode(entry.code)
    setRan(entry.code)
    setPicked(null)
  }

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
  // the click (highlight, new code) paints first; the run follows in a deferred render
  const deferredRan = useDeferredValue(ran)
  const stale = deferredRan !== ran
  const result = useMemo(() => spec.run(realOn ? '' : deferredRan), [spec, deferredRan, realOn])
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
  const selectedSeq = picked && picked.code === deferredRan ? picked.seq : autoSeq
  const selIndex = steps.findIndex((s) => s.seq === selectedSeq)
  const selected = steps[selIndex] ?? null

  const stripRef = useRef<HTMLDivElement>(null)
  const go = (k: number) => {
    const s = steps[Math.max(0, Math.min(steps.length - 1, k))]
    if (!s) return
    setPicked({ code: deferredRan, seq: s.seq })
    requestAnimationFrame(() => stripRef.current?.querySelector<HTMLElement>(`[data-seq="${s.seq}"]`)?.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' }))
  }
  const onStripKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const k = e.key === 'ArrowRight' ? selIndex + 1 : e.key === 'ArrowLeft' ? selIndex - 1 : e.key === 'Home' ? 0 : e.key === 'End' ? steps.length - 1 : null
    if (k === null) return
    e.preventDefault()
    go(k)
    requestAnimationFrame(() => stripRef.current?.querySelector<HTMLElement>('[aria-pressed="true"]')?.focus())
  }

  // ---------------------------------------------------------------- the editor | results divider
  const splitRef = useRef<HTMLDivElement>(null)
  const saveSplit = (v: number) => {
    const c = Math.round(Math.max(25, Math.min(70, v)))
    setSplit(c)
    try {
      localStorage.setItem(SPLIT_KEY, String(c))
    } catch {
      /* private mode: the width lasts for this visit */
    }
  }
  const onDrag = (e: ReactPointerEvent<HTMLDivElement>) => {
    const box = splitRef.current?.getBoundingClientRect()
    if (!box || !(e.buttons & 1)) return
    saveSplit(((box.right - e.clientX) / box.width) * 100)
  }
  const onDividerKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const d = e.key === 'ArrowLeft' ? 2 : e.key === 'ArrowRight' ? -2 : 0
    if (!d) return
    e.preventDefault()
    saveSplit(split + d)
  }

  // ---------------------------------------------------------------- toolbar
  const toolbar = (
    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-white px-4 py-2.5">
      <div className="flex min-w-0 items-center gap-2">
        {!side && (
          <button type="button" onClick={() => onSide(true)} className="hidden h-9 w-9 shrink-0 place-items-center rounded-[9px] border border-slate-200 text-slate-600 hover:text-slate-900 lg:grid" aria-label={t('site.playground.show_libs')} title={t('site.playground.show_libs')}>
            <PanelLeftOpen size={17} aria-hidden />
          </button>
        )}
        <button type="button" onClick={onGallery} className="grid h-9 w-9 shrink-0 place-items-center rounded-[9px] border border-slate-200 text-slate-600 hover:border-blue-300 hover:text-blue-700" aria-label={t('site.playground.gallery_open')} title={t('site.playground.gallery_open')}>
          <LayoutGrid size={17} aria-hidden />
        </button>
        <div className="flex min-w-0 items-center gap-2 rounded-[9px] border border-slate-200 px-3 py-2">
          <NotebookText size={16} className="shrink-0 text-blue-600" aria-hidden />
          <span className="text-sm font-medium text-slate-500">{current.name} /</span>
          <span className="truncate font-mono text-sm font-bold text-slate-900">{entry.label}</span>
        </div>
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

  // ---------------------------------------------------------------- right, top: the traced step
  const figureStep = selected?.kind === 'event' && selected.ev.type === 'figure'
  const visual = (
    <section className="flex min-h-[22rem] flex-col lg:min-h-0 lg:flex-1" aria-label={t('site.playground.visual')}>
      <div className="flex items-center justify-between gap-3 border-b border-slate-200 px-4 py-2.5">
        <span className="flex items-center gap-2 text-[13px] font-bold text-slate-900">
          <ChartArea size={15} className="text-blue-600" aria-hidden /> {t('site.playground.visual')}
        </span>
        {steps.length > 0 && (
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
      {steps.length > 1 && (
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
      <div data-visual-body className={`min-h-0 flex-1 space-y-4 overflow-auto p-4 transition-opacity ${stale ? 'opacity-60' : ''}`}>
        {selected ? (
          <VisualBoundary key={`${deferredRan}#${selected.seq}`} fallback={<p className="text-sm text-slate-500">{t('site.playground.visual_failed')}</p>}>
            <StepView step={selected.kind === 'call' ? { kind: 'call', call: selected.call } : { kind: 'event', ev: selected.ev }} />
          </VisualBoundary>
        ) : (
          !result.error && <p className="text-sm text-slate-500">{t('playground.ui.nothing')}</p>
        )}
        {!figureStep && result.displays.map((d: Display, k) => (d.type === 'figure' ? <FigureView key={`${deferredRan}#${k}`} fig={d.fig} /> : <TableView key={k} table={d.table} />))}
      </div>
    </section>
  )

  // ---------------------------------------------------------------- output / explain / variables
  const console_ = realOn ? (
    <div className="space-y-4">
      <RealConsole run={real.run} />
      <RealFigures run={real.run} />
    </div>
  ) : (
    <div aria-live="polite" className={`space-y-1.5 font-mono text-[13px] leading-6 transition-opacity ${stale ? 'opacity-60' : ''}`}>
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
  const vars = result.vars.length ? (
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
  // real Python has no steps and no variable trace: the whole column is output
  const tabs: Tab[] = realOn ? ['output', 'explain'] : ['output', 'explain', 'vars']
  const shownTab = tabs.includes(tab) ? tab : 'output'
  const outputPanel = (
    <section className={`flex flex-col ${realOn ? 'min-h-[22rem] lg:min-h-0 lg:flex-1' : 'h-[260px] shrink-0 border-t border-slate-200'}`}>
      <div className="flex gap-1 border-b border-slate-200 px-3" role="tablist" aria-label={t('playground.ui.output')}>
        {tabs.map((k) => (
          <button
            key={k}
            type="button"
            role="tab"
            aria-selected={shownTab === k}
            onClick={() => setTab(k)}
            className={`-mb-px border-b-2 px-2.5 py-2.5 text-[13px] ${shownTab === k ? 'border-blue-600 font-bold text-slate-900' : 'border-transparent font-medium text-slate-500 hover:text-slate-800'}`}
          >
            {t(`site.playground.tab_${k}`)}
          </button>
        ))}
      </div>
      <div role="tabpanel" className="min-h-0 flex-1 overflow-auto px-4 py-3">
        {shownTab === 'output' ? console_ : shownTab === 'explain' ? explain : vars}
      </div>
    </section>
  )

  // ---------------------------------------------------------------- status bar: which kernel ran this
  const lastReal = real.run.s === 'done' ? real.run.r : real.run.s === 'busy' ? real.run.prev : null
  const versions = lastReal ? ['python', 'numpy', 'pandas', 'matplotlib'].filter((k) => lastReal.versions[k]).map((k) => `${k === 'python' ? 'Python' : k} ${lastReal.versions[k]}`) : []
  const kernelDot = realOn ? (engineState === 'ready' ? 'bg-emerald-500' : engineState === 'failed' ? 'bg-rose-500' : 'bg-amber-500') : 'bg-sky-500'
  const statusBar = (
    <div className="flex h-7 shrink-0 items-center gap-4 border-t border-slate-200 bg-white px-4 text-[11.5px] text-slate-500">
      <span className="flex items-center gap-1.5 font-semibold text-slate-700">
        <i className={`h-2 w-2 rounded-full ${kernelDot}`} aria-hidden />
        {t('site.playground.kernel')}: {realOn ? t('site.playground.kernel_real') : t('site.playground.kernel_mini')}
      </span>
      <span>{realOn ? t(`playground.ui.engine.real_${engineState}`) : t('playground.ui.engine.mini_sub')}</span>
      {realOn && versions.length > 0 && <span className="hidden font-mono md:inline">{versions.join(' · ')}</span>}
      <span className="ml-auto hidden font-mono sm:inline">{t('site.playground.lines', { count: code.split('\n').length })}</span>
      <span className="font-mono">{realOn ? (lastReal ? t('playground.ui.real.took', { ms: lastReal.ms }) : '') : t('site.playground.traced', { count: steps.length })}</span>
    </div>
  )

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      {toolbar}
      <div ref={splitRef} className="flex min-h-0 flex-1 flex-col lg:flex-row" style={{ ['--split' as string]: `${split}%` }}>
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
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label={t('site.playground.resize')}
          aria-valuemin={25}
          aria-valuemax={70}
          aria-valuenow={split}
          tabIndex={0}
          onPointerDown={(e) => e.currentTarget.setPointerCapture(e.pointerId)}
          onPointerMove={onDrag}
          onKeyDown={onDividerKey}
          className="hidden w-1.5 shrink-0 cursor-col-resize touch-none bg-slate-200 transition-colors hover:bg-blue-400 focus-visible:bg-blue-500 focus-visible:outline-none active:bg-blue-500 lg:block"
        />
        <div className="flex min-w-0 flex-col bg-white lg:w-[var(--split)] lg:shrink-0">
          {!realOn && visual}
          {outputPanel}
        </div>
      </div>
      {statusBar}
    </div>
  )
}
