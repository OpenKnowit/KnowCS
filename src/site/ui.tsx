/* eslint-disable react-refresh/only-export-components -- shared lab UI kit: components + one hook */
import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { ArrowLeft, ArrowRight, ChevronRight, ChevronsDownUp, ChevronsUpDown, Menu, Search, Star, X } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { answerMatches } from './format'
import { LECTURES, MODULES, PAPERS, STARRED, moduleUrl, neighbours } from '../lib/sitemap'
import type { ModuleId, ModuleInfo } from '../lib/sitemap'

// ---------------------------------------------------------------- quiz mode
interface Quiz {
  on: boolean
  get: (key: string) => string
  set: (key: string, value: string) => void
  toggle: () => void
}
const QuizCtx = createContext<Quiz>({ on: false, get: () => '', set: () => {}, toggle: () => {} })

/**
 * A table cell holding an answer. With "Quiz me" on it becomes an input that is graded on blur.
 * Typed answers are remembered per (key, expected value), so changing the data clears them.
 */
export function Ans({ v, k, className = '' }: { v: string | number; k: string; className?: string }) {
  const quiz = useContext(QuizCtx)
  const expected = String(v)
  if (!quiz.on) return <td className={className}>{expected}</td>
  return (
    <td className={className}>
      <QuizInput id={`${k}=${expected}`} expected={expected} quiz={quiz} label={k} />
    </td>
  )
}

function QuizInput({ id, expected, quiz, label }: { id: string; expected: string; quiz: Quiz; label: string }) {
  const { t } = useTranslation()
  const [graded, setGraded] = useState(() => quiz.get(id) !== '')
  const value = quiz.get(id)
  const ok = value.trim() !== '' && answerMatches(value, expected)
  const tone = !graded || value.trim() === '' ? 'border-dashed border-indigo-400 bg-indigo-50' : ok ? 'border-emerald-500 bg-emerald-50' : 'border-rose-500 bg-rose-50'
  return (
    <input
      aria-label={t('site.ui.answer_for', { label })}
      className={`w-20 rounded-md border px-1.5 py-0.5 text-center font-mono text-[13px] outline-none focus:ring-2 focus:ring-indigo-300 ${tone}`}
      value={value}
      onChange={(e) => {
        quiz.set(id, e.target.value)
        setGraded(false)
      }}
      onBlur={() => setGraded(true)}
      onKeyDown={(e) => e.key === 'Enter' && setGraded(true)}
      title={graded && !ok && value ? t('site.ui.not_quite') : undefined}
    />
  )
}

// ---------------------------------------------------------------- progress (per-browser convenience only)
const VISITED_KEY = 'knowcs-lab:visited'
export function readVisited(): Record<string, number> {
  try {
    return JSON.parse(localStorage.getItem(VISITED_KEY) ?? '{}') as Record<string, number>
  } catch {
    return {}
  }
}
function recordVisit(id: string) {
  try {
    localStorage.setItem(VISITED_KEY, JSON.stringify({ ...readVisited(), [id]: Date.now() }))
  } catch {
    /* storage blocked: progress is optional */
  }
}

// ---------------------------------------------------------------- page shell
/** Papers as short badges: "2022 Fall mid", "Final 2024" (translated). */
export function usePaperLabel(): (id: string) => string {
  const { t } = useTranslation()
  return (id) => {
    const p = PAPERS.find((x) => x.id === id)
    if (!p) return id
    return p.kind === 'M' ? t('site.paper_mid', { year: p.year, term: t(`site.term_${p.term}`) }) : t('site.paper_final', { year: p.year })
  }
}

const ALL_OPEN_KEY = 'knowcs-drawer-all-open'
const readAllOpen = (): boolean => {
  try {
    return localStorage.getItem(ALL_OPEN_KEY) === '1'
  } catch {
    return false
  }
}

/** The list of labs that slides in from the left; the current lecture starts open (or every lecture, once asked for). */
function LabsDrawer({ current, onClose }: { current: ModuleId; onClose: () => void }) {
  const { t } = useTranslation()
  const cur = MODULES.find((m) => m.id === current)!
  const [allOpen, setAllOpen] = useState(readAllOpen)
  const [openLecs, setOpenLecs] = useState<Set<number>>(() => new Set(allOpen ? LECTURES : [cur.lec]))
  const toggleAll = () => {
    const next = !allOpen
    setAllOpen(next)
    setOpenLecs(new Set(next ? LECTURES : [cur.lec]))
    try {
      localStorage.setItem(ALL_OPEN_KEY, next ? '1' : '0')
    } catch {
      /* private mode: the choice lasts until the drawer closes */
    }
  }
  const toggleLec = (n: number) =>
    setOpenLecs((s) => {
      const next = new Set(s)
      if (next.has(n)) next.delete(n)
      else next.add(n)
      return next
    })
  const [q, setQ] = useState('')
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
  const query = q.trim().toLowerCase()
  const hit = (id: ModuleId) => !query || t(`site.modules.${id}.title`).toLowerCase().includes(query)
  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-label={t('site.ui.all_labs')}>
      <div className="absolute inset-0 bg-slate-900/35" onClick={onClose} aria-hidden />
      <nav id="labs-drawer" className="drawer-in absolute inset-y-0 left-0 flex w-[340px] max-w-[88vw] flex-col border-r border-slate-200 bg-white shadow-2xl shadow-slate-900/20">
        <div className="flex items-center py-3 pl-5 pr-4">
          <b className="text-base font-extrabold text-slate-900">{t('site.ui.all_labs')}</b>
          <button type="button" onClick={toggleAll} aria-pressed={allOpen} className="ml-auto mr-2 flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[13px] font-semibold text-slate-600 hover:bg-slate-100 hover:text-slate-900">
            {allOpen ? <ChevronsDownUp size={15} aria-hidden /> : <ChevronsUpDown size={15} aria-hidden />}
            {t(allOpen ? 'site.ui.collapse_all' : 'site.ui.expand_all')}
          </button>
          <button ref={closeRef} type="button" onClick={onClose} className={ICON_BTN} aria-label={t('site.ui.close')}>
            <X size={18} aria-hidden />
          </button>
        </div>
        <label className="mx-4 mb-3 flex h-[38px] items-center gap-2.5 rounded-[10px] border border-slate-200 px-3 focus-within:border-blue-400">
          <Search size={16} className="shrink-0 text-slate-500" aria-hidden />
          <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder={t('site.ui.filter_labs')} aria-label={t('site.ui.filter_labs')} className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-slate-500" />
        </label>
        <ul className="flex-1 overflow-y-auto px-2 pb-4">
          {LECTURES.map((n) => {
            const labs = MODULES.filter((m) => m.lec === n && hit(m.id))
            if (!labs.length) return null
            const open = query !== '' || openLecs.has(n)
            return (
              <li key={n}>
                <button type="button" onClick={() => !query && toggleLec(n)} aria-expanded={open} className="flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left hover:bg-slate-50">
                  <ChevronRight size={14} className={`shrink-0 text-slate-500 transition-transform ${open ? 'rotate-90' : ''}`} aria-hidden />
                  <span className="w-7 font-mono text-[11px] font-bold text-slate-500">{t('site.lecture_short', { n })}</span>
                  <span className={`flex-1 text-[13px] ${n === cur.lec ? 'font-bold text-slate-900' : 'font-semibold text-slate-600'}`}>{t(`site.lectures.${n}`)}</span>
                  <span className="font-mono text-[11px] text-slate-500">{labs.length}</span>
                </button>
                {open && (
                  <ul>
                    {labs.map((m) => (
                      <li key={m.id}>
                        <a
                          href={moduleUrl(m.id)}
                          aria-current={m.id === current ? 'page' : undefined}
                          className={`flex items-center gap-2 rounded-lg py-2 pl-12 pr-3 text-[13px] ${m.id === current ? 'bg-blue-50 font-bold text-blue-700' : 'font-medium text-slate-600 hover:bg-slate-50 hover:text-slate-900'}`}
                        >
                          <span className="flex-1">{t(`site.modules.${m.id}.title`)}</span>
                          {STARRED.has(m.id) && <Star size={12} className="fill-amber-500 text-amber-500" aria-label={t('site.home.starred')} />}
                          {m.id === current && <i className="h-1.5 w-1.5 rounded-full bg-blue-600" aria-hidden />}
                        </a>
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            )
          })}
        </ul>
      </nav>
    </div>
  )
}

const ICON_BTN = 'grid h-9 w-9 shrink-0 place-items-center rounded-[9px] border border-slate-200 bg-white text-slate-600 transition hover:border-slate-300 hover:text-slate-900'

/** Sticky bar under the site header: ☰ (all labs), lecture + title, extra buttons, prev / next. */
/** bare: just the row, for a page that places it inside its own panel */
export function PageBar({ id, extra, bare = false }: { id: ModuleId; extra?: ReactNode; bare?: boolean }) {
  const { t } = useTranslation()
  const [drawer, setDrawer] = useState(false)
  const idx = MODULES.findIndex((m) => m.id === id)
  const mod = MODULES[idx]
  const { prev, next } = neighbours(id)
  useEffect(() => {
    document.title = `${t(`site.modules.${id}.title`)} · KnowCS`
  }, [id, t])
  const pager = (m: ModuleInfo | null, dir: 'prev' | 'next') =>
    m ? (
      <a className={ICON_BTN} href={moduleUrl(m.id)} rel={dir} title={`${t(`site.ui.${dir}`)}: ${t(`site.modules.${m.id}.title`)}`} aria-label={`${t(`site.ui.${dir}`)}: ${t(`site.modules.${m.id}.title`)}`}>
        {dir === 'prev' ? <ArrowLeft size={18} aria-hidden /> : <ArrowRight size={18} aria-hidden />}
      </a>
    ) : (
      <span className={`${ICON_BTN} opacity-40`} aria-hidden>
        {dir === 'prev' ? <ArrowLeft size={18} /> : <ArrowRight size={18} />}
      </span>
    )
  return (
    <>
      <div className={bare ? '' : 'sticky top-0 z-40 border-b border-slate-200 bg-white/95 backdrop-blur'}>
        <div className={bare ? 'flex items-center gap-2' : 'flex items-center gap-3.5 px-4 py-2.5 sm:px-6'}>
          <button type="button" onClick={() => setDrawer(true)} className={ICON_BTN} aria-label={t('site.ui.open_labs')} aria-expanded={drawer} aria-controls="labs-drawer">
            <Menu size={18} aria-hidden />
          </button>
          <div className="min-w-0 flex-1">
            <a href={`/#lecture-${mod.lec}`} className="block truncate text-[11px] font-bold uppercase tracking-[0.06em] text-blue-600 hover:underline">
              {t('site.lecture_short', { n: mod.lec })} · {t(`site.lectures.${mod.lec}`)}
            </a>
            <h1 className={`truncate font-extrabold tracking-tight text-slate-900 ${bare ? 'text-[17px]' : 'text-[19px]'}`}>{t(`site.modules.${id}.title`)}</h1>
          </div>
          {extra}
          <div className={`flex items-center ${bare ? 'gap-1.5' : 'gap-2.5'}`}>
            {pager(prev, 'prev')}
            {!bare && (
              <span className="hidden font-mono text-xs font-semibold text-slate-500 sm:inline" aria-label={t('site.ui.position', { n: idx + 1, total: MODULES.length })}>
                {idx + 1} / {MODULES.length}
              </span>
            )}
            {pager(next, 'next')}
          </div>
        </div>
      </div>
      {drawer && <LabsDrawer current={id} onClose={() => setDrawer(false)} />}
    </>
  )
}

/** the "Quiz me" switch for the answer cells inside the page */
export function QuizToggle() {
  const { t } = useTranslation()
  const quiz = useContext(QuizCtx)
  return (
    <button
      type="button"
      onClick={quiz.toggle}
      aria-pressed={quiz.on}
      title={t('site.ui.quiz_hint')}
      className={`whitespace-nowrap rounded-full border px-3 py-1.5 text-[13px] font-semibold ${quiz.on ? 'border-indigo-600 bg-indigo-600 text-white' : 'border-slate-200 bg-white text-slate-600'}`}
    >
      ✎<span className="hidden sm:inline"> {t('site.ui.quiz')}</span>
    </button>
  )
}

/** dashboard: the page lays out its own screen, title included (PageBar bare) */
export function LabPage({ id, quiz = false, dashboard = false, children }: { id: ModuleId; quiz?: boolean; dashboard?: boolean; children: ReactNode }) {
  const [quizOn, setQuizOn] = useState(false)
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const ctx = useMemo<Quiz>(() => ({ on: quizOn, get: (k) => answers[k] ?? '', set: (k, v) => setAnswers((a) => ({ ...a, [k]: v })), toggle: () => setQuizOn((q) => !q) }), [quizOn, answers])
  useEffect(() => recordVisit(id), [id])
  return (
    <QuizCtx.Provider value={ctx}>
      {dashboard ? (
        children
      ) : (
        <>
          <PageBar id={id} extra={quiz ? <QuizToggle /> : undefined} />
          <main className="px-4 pb-10 pt-5 sm:px-6">{children}</main>
        </>
      )}
    </QuizCtx.Provider>
  )
}

// ---------------------------------------------------------------- layout
export function Workspace({ controls, children, wide = false }: { controls: ReactNode; children: ReactNode; wide?: boolean }) {
  return (
    <div className={`grid items-start gap-5 ${wide ? 'lg:grid-cols-[400px_minmax(0,1fr)]' : 'lg:grid-cols-[340px_minmax(0,1fr)]'}`}>
      <aside className="grid gap-4 lg:sticky lg:top-[68px]">{controls}</aside>
      <div className="grid min-w-0 gap-5">{children}</div>
    </div>
  )
}

export function Card({ title, step, sub, right, children, className = '', flat = false }: { title?: ReactNode; step?: number; sub?: ReactNode; right?: ReactNode; children?: ReactNode; className?: string; flat?: boolean }) {
  return (
    <section className={`min-w-0 rounded-[20px] border border-slate-200 p-4 sm:p-5 ${flat ? 'bg-slate-50' : 'bg-white'} ${className}`}>
      {(title || sub || right) && (
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          {title && (
            <h2 className="flex items-center text-[15px] font-extrabold">
              {step !== undefined && <span className="mr-2 grid h-[22px] w-[22px] place-items-center rounded-full bg-blue-600 text-xs font-extrabold text-white">{step}</span>}
              {title}
            </h2>
          )}
          {sub && <span className="text-xs text-slate-500">{sub}</span>}
          {right}
        </div>
      )}
      {children}
    </section>
  )
}

// ---------------------------------------------------------------- controls
export interface PresetItem<T extends string = string> {
  id: T
  title: string
  note?: string
}

const PRESETS_OPEN_KEY = 'knowcs-presets-open'
const readPresetsOpen = () => {
  try {
    return localStorage.getItem(PRESETS_OPEN_KEY) !== '0'
  } catch {
    return true
  }
}

/**
 * A lab's list of examples. With a title it is its own card that folds away (the choice is remembered across labs);
 * folded, the header still names the example that is loaded.
 */
export function Presets<T extends string>({ items, value, onPick, title, children }: { items: PresetItem<T>[]; value: T | null; onPick: (id: T) => void; title?: ReactNode; children?: ReactNode }) {
  const [open, setOpen] = useState(readPresetsOpen)
  const list = (
    <div className="grid gap-1.5">
      {items.map((p) => (
        <button
          key={p.id}
          type="button"
          onClick={() => onPick(p.id)}
          aria-pressed={p.id === value}
          className={`rounded-xl border px-3 py-2 text-left text-[13px] font-semibold transition ${p.id === value ? 'border-blue-500 bg-blue-50' : 'border-slate-200 bg-white hover:border-slate-300'}`}
        >
          {p.title}
          {p.note && <small className="mt-0.5 block text-xs font-medium text-slate-500">{p.note}</small>}
        </button>
      ))}
    </div>
  )
  if (title === undefined) return list
  const current = items.find((p) => p.id === value)
  const toggle = () => {
    setOpen((o) => {
      try {
        localStorage.setItem(PRESETS_OPEN_KEY, o ? '0' : '1')
      } catch {
        // private mode: just not remembered
      }
      return !o
    })
  }
  return (
    <section className="min-w-0 rounded-[20px] border border-slate-200 bg-white p-4 sm:p-5">
      <button type="button" aria-expanded={open} onClick={toggle} className="flex w-full min-w-0 items-center gap-2 text-left">
        <h2 className="shrink-0 text-[15px] font-extrabold">{title}</h2>
        {!open && current && <span className="min-w-0 truncate text-[13px] font-semibold text-blue-700">{current.title}</span>}
        <ChevronRight size={16} aria-hidden className={`ml-auto shrink-0 text-slate-400 transition-transform ${open ? 'rotate-90' : ''}`} />
      </button>
      {open && (
        <div className="mt-3">
          {list}
          {children}
        </div>
      )}
    </section>
  )
}

export function Seg<T extends string | number>({ options, value, onChange, label }: { options: { v: T; label: ReactNode }[]; value: T; onChange: (v: T) => void; label?: string }) {
  return (
    <div className="grid gap-1.5">
      {label && <span className="text-xs font-bold text-slate-500">{label}</span>}
      <div className="inline-flex w-fit flex-wrap overflow-hidden rounded-[10px] border border-slate-300" role="group" aria-label={label}>
        {options.map((o, i) => (
          <button
            key={String(o.v)}
            type="button"
            aria-pressed={o.v === value}
            onClick={() => onChange(o.v)}
            className={`px-3 py-1.5 text-xs font-bold ${i ? 'border-l border-slate-200' : ''} ${o.v === value ? 'bg-blue-600 text-white' : 'bg-white text-slate-600 hover:bg-slate-50'}`}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  )
}

export function Slider({ label, value, min, max, step, onChange, display }: { label: ReactNode; value: number; min: number; max: number; step: number; onChange: (v: number) => void; display?: string }) {
  return (
    <label className="grid gap-1">
      <span className="flex justify-between gap-2 text-xs font-bold text-slate-500">
        <span>{label}</span>
        <output className="font-mono text-blue-700">{display ?? value}</output>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="w-full" />
    </label>
  )
}

export function NumberField({ label, value, onChange, step = 1, min, max, className = '' }: { label: ReactNode; value: number; onChange: (v: number) => void; step?: number; min?: number; max?: number; className?: string }) {
  // what the box shows while it is being edited: it may be empty or out of range for a moment ("" while
  // replacing the last digit); the last valid number is what the page uses, and leaving the box shows it again
  const [text, setText] = useState<string | null>(null)
  return (
    <label className={`grid gap-1 ${className}`}>
      <span className="text-xs font-bold text-slate-500">{label}</span>
      <input
        type="number"
        value={text ?? (Number.isFinite(value) ? value : '')}
        step={step}
        min={min}
        max={max}
        onChange={(e) => {
          setText(e.target.value)
          const v = Number(e.target.value)
          if (e.target.value.trim() !== '' && Number.isFinite(v)) onChange(v)
        }}
        onBlur={() => setText(null)}
        className="w-full rounded-[10px] border border-slate-300 bg-white px-2.5 py-1.5 font-mono text-sm outline-none focus:ring-2 focus:ring-blue-300"
      />
    </label>
  )
}

export function Btn({ children, onClick, primary = false, disabled = false, title }: { children: ReactNode; onClick: () => void; primary?: boolean; disabled?: boolean; title?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`rounded-[10px] border px-3.5 py-2 text-[13px] font-bold transition disabled:cursor-not-allowed disabled:opacity-40 ${primary ? 'border-blue-600 bg-blue-600 text-white hover:bg-blue-700' : 'border-slate-300 bg-white text-slate-800 hover:border-slate-400'}`}
    >
      {children}
    </button>
  )
}

// ---------------------------------------------------------------- read-outs
type Tone = 'info' | 'warn' | 'bad' | 'good'
const NOTE_TONES: Record<Tone, string> = {
  info: 'border-blue-200 bg-blue-50 text-blue-950',
  warn: 'border-amber-200 bg-amber-50 text-amber-950',
  bad: 'border-rose-200 bg-rose-50 text-rose-950',
  good: 'border-emerald-200 bg-emerald-50 text-emerald-950',
}
export function Note({ tone = 'info', title, children }: { tone?: Tone; title?: string; children: ReactNode }) {
  return (
    <div className={`rounded-xl border px-3.5 py-2.5 text-[13.5px] leading-relaxed ${NOTE_TONES[tone]}`}>
      {title && <b className="font-extrabold">{title} </b>}
      {children}
    </div>
  )
}

const STAT_TONES = { none: 'text-slate-900', good: 'text-emerald-600', bad: 'text-rose-600', blue: 'text-blue-600', warn: 'text-amber-600' }
export function Stat({ k, v, d, tone = 'none' }: { k: string; v: ReactNode; d?: ReactNode; tone?: keyof typeof STAT_TONES }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5">
      <div className="text-[11px] font-extrabold uppercase tracking-wider text-slate-500">{k}</div>
      <div className={`mt-0.5 font-mono text-[22px] font-extrabold ${STAT_TONES[tone]}`}>{v}</div>
      {d && <div className="text-xs text-slate-500">{d}</div>}
    </div>
  )
}

export function TableWrap({ children }: { children: ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-slate-200">
      <table className="lab-table">
        <tbody>{children}</tbody>
      </table>
    </div>
  )
}

export function Swatch({ color }: { color: string }) {
  return <i className="mr-1.5 inline-block h-2.5 w-2.5 rounded-[3px] align-[-1px]" style={{ background: color }} />
}
