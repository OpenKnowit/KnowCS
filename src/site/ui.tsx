/* eslint-disable react-refresh/only-export-components -- shared lab UI kit: components + one hook */
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { answerMatches } from './format'
import { EPISODES, MODULES, PAPERS, moduleUrl, neighbours, watchUrl } from '../lib/sitemap'
import type { ModuleId } from '../lib/sitemap'

// ---------------------------------------------------------------- quiz mode
interface Quiz {
  on: boolean
  get: (key: string) => string
  set: (key: string, value: string) => void
}
const QuizCtx = createContext<Quiz>({ on: false, get: () => '', set: () => {} })

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

/** Sticky bar under the site header: where you are, prev / next, optional extra buttons. */
export function PageBar({ id, extra }: { id: ModuleId; extra?: ReactNode }) {
  const { t } = useTranslation()
  const mod = MODULES.find((m) => m.id === id)!
  const { prev, next } = neighbours(id)
  const pill = 'whitespace-nowrap rounded-full border border-slate-200 bg-white px-3 py-1.5 text-[13px] font-semibold text-slate-600 hover:border-slate-300 hover:text-slate-900'
  return (
    <div className="sticky top-0 z-40 border-b border-slate-200 bg-[#f1f5f9]/90 backdrop-blur">
      <div className="mx-auto flex max-w-[1480px] items-center gap-3 px-4 py-2 sm:px-6">
        <nav className="min-w-0 truncate text-[13px] text-slate-500" aria-label={t('site.ui.breadcrumb')}>
          <a href="/" className="font-semibold text-slate-500 hover:text-blue-600">{t('app.nav.course')}</a>
          {' / '}
          <a href={`/#lecture-${mod.lec}`} className="font-bold text-slate-600 hover:text-blue-600">
            {t('site.lecture_short', { n: mod.lec })} · {t(`site.lectures.${mod.lec}`)}
          </a>
          <span className="hidden md:inline"> / {t(`site.modules.${id}.title`)}</span>
        </nav>
        <span className="flex-1" />
        <div className="flex gap-1.5">
          {extra}
          {prev && (
            <a className={pill} href={moduleUrl(prev.id)} title={t(`site.modules.${prev.id}.title`)} rel="prev">
              ←<span className="hidden sm:inline"> {t('site.ui.prev')}</span>
            </a>
          )}
          {next && (
            <a className={pill} href={moduleUrl(next.id)} title={t(`site.modules.${next.id}.title`)} rel="next">
              <span className="hidden sm:inline">{t('site.ui.next')} </span>→
            </a>
          )}
        </div>
      </div>
    </div>
  )
}

/** Title block of a module page: lecture, title, one-line lead, exam badges. */
export function ModuleHero({ id, lead }: { id: ModuleId; lead?: ReactNode }) {
  const { t } = useTranslation()
  const paper = usePaperLabel()
  const mod = MODULES.find((m) => m.id === id)!
  useEffect(() => {
    document.title = `${t(`site.modules.${id}.title`)} · KnowCS`
  }, [id, t])
  return (
    <section className="mb-5 flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-3xl">
        <div className="text-xs font-extrabold uppercase tracking-[0.12em] text-blue-600">
          {t('site.lecture_long', { n: mod.lec })} · {t(`site.lectures.${mod.lec}`)}
        </div>
        <h1 className="mt-1 text-[clamp(26px,3.2vw,36px)] font-black tracking-tight">{t(`site.modules.${id}.title`)}</h1>
        <p className="mt-1.5 text-slate-600">{lead ?? t(`site.modules.${id}.blurb`)}</p>
        {EPISODES.filter((e) => e.modules.includes(id)).map((e) => (
          <a key={e.id} href={watchUrl(e.id)} className="mt-3 inline-flex items-center gap-2 rounded-full bg-slate-900 py-1.5 pl-1.5 pr-4 text-[13px] font-bold text-white shadow-md shadow-slate-300 hover:bg-blue-700">
            <span className="grid h-6 w-6 place-items-center rounded-full bg-white text-[10px] text-slate-900" aria-hidden>▶</span>
            {t('watch.ui.watch_first', { title: t(`watch.${e.id}.title`) })}
          </a>
        ))}
      </div>
      {mod.exams.length > 0 && (
        <div className="flex max-w-xl flex-wrap justify-start gap-1.5 sm:justify-end" aria-label={t('site.ui.asked_in')}>
          {mod.exams.map((p) => (
            <span key={p} className="whitespace-nowrap rounded-full border border-amber-200 bg-amber-50 px-2.5 py-0.5 text-xs font-bold text-amber-800">
              {paper(p)}
            </span>
          ))}
        </div>
      )}
    </section>
  )
}

export function LabPage({ id, lead, quiz = false, children }: { id: ModuleId; lead?: ReactNode; quiz?: boolean; children: ReactNode }) {
  const { t } = useTranslation()
  const [quizOn, setQuizOn] = useState(false)
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const ctx = useMemo<Quiz>(() => ({ on: quizOn, get: (k) => answers[k] ?? '', set: (k, v) => setAnswers((a) => ({ ...a, [k]: v })) }), [quizOn, answers])
  useEffect(() => recordVisit(id), [id])
  const toggle = quiz ? (
    <button
      type="button"
      onClick={() => setQuizOn((q) => !q)}
      aria-pressed={quizOn}
      title={t('site.ui.quiz_hint')}
      className={`whitespace-nowrap rounded-full border px-3 py-1.5 text-[13px] font-semibold ${quizOn ? 'border-indigo-600 bg-indigo-600 text-white' : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:text-slate-900'}`}
    >
      ✎<span className="hidden sm:inline"> {t('site.ui.quiz')}</span>
    </button>
  ) : undefined
  return (
    <QuizCtx.Provider value={ctx}>
      <PageBar id={id} extra={toggle} />
      <main className="mx-auto max-w-[1480px] px-4 pb-10 pt-6 sm:px-6">
        <ModuleHero id={id} lead={lead} />
        {children}
      </main>
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

export function Presets<T extends string>({ items, value, onPick }: { items: PresetItem<T>[]; value: T | null; onPick: (id: T) => void }) {
  return (
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
  return (
    <label className={`grid gap-1 ${className}`}>
      <span className="text-xs font-bold text-slate-500">{label}</span>
      <input
        type="number"
        value={Number.isFinite(value) ? value : ''}
        step={step}
        min={min}
        max={max}
        onChange={(e) => e.target.value !== '' && onChange(Number(e.target.value))}
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
