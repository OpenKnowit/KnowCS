import { useState } from 'react'
import { ALL_MODULES, LECTURES, PAPERS } from '../registry'
import { MockupSwitcher, ModuleCard, SiteHeader } from './shared'

type Scope = 'all' | 'mid' | 'final'

export default function HomeCourseMap() {
  const [scope, setScope] = useState<Scope>('all')
  const [q, setQ] = useState('')
  const query = q.trim().toLowerCase()
  const lectures = LECTURES.filter((l) => scope === 'all' || (scope === 'mid' ? l.scope === 'mid' : true))
  const matches = (title: string, blurb: string) => !query || title.toLowerCase().includes(query) || blurb.toLowerCase().includes(query)

  return (
    <div className="min-h-screen pb-24">
      <SiteHeader />
      <section className="mx-auto max-w-[1280px] px-4 pb-6 pt-10 sm:px-8">
        <div className="grid items-end gap-6 lg:grid-cols-[minmax(0,1fr)_420px]">
          <div>
            <p className="text-xs font-extrabold uppercase tracking-[0.14em] text-blue-600">HKUST COMP2211 · Exploring AI</p>
            <h1 className="mt-2 text-[clamp(32px,4.4vw,52px)] font-black leading-[1.05] tracking-tight text-slate-900">
              Every lecture, <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">one page you can touch.</span>
            </h1>
            <p className="mt-3 max-w-2xl text-[17px] text-slate-600">Pick a lecture, open its page, drag the numbers. Each page shows the exact table the exam asks for — and lets you fill it in yourself.</p>
            <div className="mt-5 flex flex-wrap gap-2 text-sm font-bold">
              <span className="rounded-full bg-white px-3 py-1.5 text-slate-700 shadow-sm">{ALL_MODULES.length} interactive pages</span>
              <span className="rounded-full bg-white px-3 py-1.5 text-slate-700 shadow-sm">{PAPERS.length} past papers mapped</span>
              <span className="rounded-full bg-white px-3 py-1.5 text-slate-700 shadow-sm">English · 简体 · 繁體</span>
            </div>
          </div>
          <div className="rounded-3xl border border-slate-200 bg-white p-4 shadow-sm">
            <label className="text-xs font-bold text-slate-500" htmlFor="q">Find a topic</label>
            <input id="q" value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. padding, F1, backprop…" className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2.5 text-[15px] outline-none focus:ring-2 focus:ring-blue-300" />
            <div className="mt-3 grid grid-cols-3 gap-1 rounded-xl bg-slate-100 p-1 text-sm font-bold" role="group" aria-label="Exam scope">
              {([['all', 'Everything'], ['mid', 'Midterm'], ['final', 'Final']] as const).map(([v, label]) => (
                <button key={v} type="button" onClick={() => setScope(v)} aria-pressed={scope === v} className={`rounded-lg py-1.5 ${scope === v ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500'}`}>{label}</button>
              ))}
            </div>
            <p className="mt-2 text-xs text-slate-400">{scope === 'mid' ? 'Midterms cover lectures 1–6.' : scope === 'final' ? 'Finals cover everything, weighted towards lectures 6–11.' : 'All eleven lectures.'}</p>
          </div>
        </div>
      </section>

      <main className="mx-auto grid max-w-[1280px] gap-4 px-4 sm:px-8">
        {lectures.map((l) => {
          const mods = ALL_MODULES.filter((m) => m.lec === l.n && matches(m.title, m.blurb))
          if (query && !mods.length) return null
          const finalFocus = scope === 'final' && l.n >= 6
          return (
            <section key={l.n} className={`grid gap-4 rounded-3xl border bg-white/60 p-4 sm:p-5 md:grid-cols-[220px_minmax(0,1fr)] ${finalFocus ? 'border-rose-200 bg-rose-50/40' : 'border-slate-200'}`}>
              <div className="flex items-start gap-3 md:block">
                <div className="font-mono text-4xl font-black leading-none text-slate-200 md:text-5xl">{String(l.n).padStart(2, '0')}</div>
                <div>
                  <h2 className="mt-1 text-lg font-black text-slate-900 md:mt-2">{l.title}</h2>
                  <span className={`mt-1 inline-block rounded-full px-2 py-0.5 text-[10px] font-black uppercase ${l.scope === 'mid' ? 'bg-amber-100 text-amber-700' : 'bg-rose-100 text-rose-700'}`}>{l.scope === 'mid' ? 'midterm + final' : 'final only'}</span>
                </div>
              </div>
              {mods.length ? (
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                  {mods.map((m) => <ModuleCard key={m.id} m={m} />)}
                </div>
              ) : (
                <div className="grid place-items-center rounded-2xl border border-dashed border-slate-300 p-6 text-center text-sm text-slate-500">
                  {l.n === 9 ? 'No past paper tested PyTorch — read the PyTorch notes instead.' : l.n === 11 ? 'Ethics is multiple choice: flashcards are planned.' : 'Coming soon.'}
                </div>
              )}
            </section>
          )
        })}
      </main>
      <MockupSwitcher current="a" />
    </div>
  )
}
