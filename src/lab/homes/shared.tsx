import type { ReactNode } from 'react'
import { PAPERS, hrefOf, lecture, type Module } from '../registry'

/** Floating switcher so the three mockups can be compared side by side. */
export function MockupSwitcher({ current }: { current: 'a' | 'b' | 'c' }) {
  const items = [
    { id: 'a', label: 'A · Course map' },
    { id: 'b', label: 'B · Exam radar' },
    { id: 'c', label: 'C · Study path' },
  ] as const
  return (
    <div className="fixed bottom-4 left-1/2 z-50 flex max-w-[calc(100vw-24px)] -translate-x-1/2 items-center gap-1 overflow-x-auto rounded-full border border-slate-200 bg-white/95 p-1 shadow-xl shadow-slate-300/40 backdrop-blur">
      <a href="index.html" className="whitespace-nowrap rounded-full px-3 py-1.5 text-xs font-bold text-slate-500 hover:text-slate-900">Mockup</a>
      {items.map((it) => (
        <a key={it.id} href={`home-${it.id}.html`} aria-current={it.id === current ? 'page' : undefined} className={`whitespace-nowrap rounded-full px-3 py-1.5 text-xs font-bold ${it.id === current ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-100'}`}>
          {it.label}
        </a>
      ))}
    </div>
  )
}

/** The main-site header as it would look on every page (static in the mockups). */
export function SiteHeader({ dark = false, children }: { dark?: boolean; children?: ReactNode }) {
  const ink = dark ? 'text-white' : 'text-slate-900'
  return (
    <header className={`flex flex-wrap items-center gap-3 px-4 py-4 sm:px-8 ${dark ? '' : 'border-b border-slate-200 bg-white'}`}>
      <a href="home-a.html" className={`flex items-center gap-2.5 font-black tracking-tight ${ink}`}>
        <span className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 text-sm font-black text-white shadow-md shadow-blue-500/30">K</span>
        <span className="text-lg">KnowCS</span>
        <span className={`hidden rounded-full px-2 py-0.5 text-[10px] font-black uppercase sm:inline ${dark ? 'bg-white/15 text-white' : 'bg-indigo-100 text-indigo-600'}`}>COMP2211</span>
      </a>
      <span className="flex-1" />
      {children}
      <nav className={`flex items-center gap-1 rounded-full p-1 text-sm font-bold ${dark ? 'bg-white/10' : 'border border-slate-200 bg-white'}`}>
        {['Course', 'Notes', 'Extend'].map((n, i) => (
          <span key={n} className={`rounded-full px-3 py-1 ${i === 0 ? (dark ? 'bg-white text-slate-900' : 'bg-blue-600 text-white') : dark ? 'text-white/70' : 'text-slate-500'}`}>{n}</span>
        ))}
      </nav>
      <span className={`rounded-full px-3 py-1.5 text-sm font-bold ${dark ? 'bg-white/10 text-white' : 'border border-slate-200 bg-white text-slate-600'}`}>EN / 简 / 繁</span>
    </header>
  )
}

export function ExamDots({ papers, size = 8 }: { papers: string[]; size?: number }) {
  return (
    <span className="inline-flex items-center gap-[3px]" title={`${papers.length} of ${PAPERS.length} papers`}>
      {PAPERS.map((p) => (
        <i key={p.id} className={`inline-block rounded-full ${papers.includes(p.id) ? (p.kind === 'M' ? 'bg-amber-500' : 'bg-rose-500') : 'bg-slate-200'}`} style={{ width: size, height: size }} />
      ))}
    </span>
  )
}

export function ModuleCard({ m, compact = false }: { m: Module; compact?: boolean }) {
  return (
    <a href={hrefOf(m)} className="group flex flex-col gap-1.5 rounded-2xl border border-slate-200 bg-white p-4 transition hover:-translate-y-0.5 hover:border-blue-300 hover:shadow-lg hover:shadow-blue-100">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-extrabold uppercase tracking-wider text-slate-400">L{m.lec} · {lecture(m.lec).title}</span>
        <span className={`rounded-full px-2 py-0.5 text-[10px] font-black uppercase ${m.status === 'lab' ? 'bg-violet-100 text-violet-700' : 'bg-emerald-100 text-emerald-700'}`}>{m.status === 'lab' ? 'new' : 'live'}</span>
      </div>
      <b className="text-[15px] font-extrabold text-slate-900 group-hover:text-blue-700">{m.title}</b>
      {!compact && <p className="text-[13px] leading-snug text-slate-500">{m.blurb}</p>}
      {m.exams.length > 0 && (
        <div className="mt-auto flex items-center gap-2 pt-1 text-[11px] font-bold text-slate-500">
          <ExamDots papers={m.exams} /> in {m.exams.length}/{PAPERS.length} papers
        </div>
      )}
    </a>
  )
}
