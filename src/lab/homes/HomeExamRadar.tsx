import { useState } from 'react'
import { PAPERS, TOPICS, hrefOf, lecture, moduleById } from '../registry'
import { MockupSwitcher, ModuleCard, SiteHeader } from './shared'

type Which = 'all' | 'M' | 'F'

export default function HomeExamRadar() {
  const [which, setWhich] = useState<Which>('all')
  const papers = PAPERS.filter((p) => which === 'all' || p.kind === which)
  const count = (t: (typeof TOPICS)[number]) => t.papers.filter((id) => papers.some((p) => p.id === id)).length
  const rows = [...TOPICS].map((t) => ({ t, n: count(t) })).filter((r) => r.n > 0).sort((a, b) => b.n - a.n || a.t.lec - b.t.lec)
  const top = rows.flatMap((r) => r.t.links).filter((id, i, a) => a.indexOf(id) === i).map(moduleById).filter((m) => m !== undefined).slice(0, 6)

  return (
    <div className="min-h-screen bg-slate-100 pb-24">
      <div className="bg-[radial-gradient(1200px_500px_at_20%_-10%,#4f46e5_0%,transparent_60%),linear-gradient(135deg,#0f172a,#1e293b)] pb-16">
        <SiteHeader dark />
        <section className="mx-auto grid max-w-[1280px] gap-8 px-4 pt-10 sm:px-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,0.9fr)]">
          <div>
            <p className="text-xs font-extrabold uppercase tracking-[0.14em] text-indigo-300">Built from {PAPERS.length} past papers with solutions</p>
            <h1 className="mt-2 text-[clamp(32px,4.4vw,52px)] font-black leading-[1.05] tracking-tight text-white">Practise what the exam <span className="text-amber-300">actually asks.</span></h1>
            <p className="mt-3 max-w-xl text-[17px] text-slate-300">Sixteen question patterns come back year after year. Each row below shows when it appeared — and opens the page where you can drill it with the real table format.</p>
          </div>
          <div className="grid grid-cols-2 gap-3 self-end">
            {[
              ['6 / 6', 'midterms had a one-line NumPy distance question'],
              ['7 / 9', 'papers made you run K-Means or KNN by hand'],
              ['3 / 3', 'finals asked CNN shapes, minimax and image processing'],
              ['4', 'different perceptron activation conventions'],
            ].map(([n, t]) => (
              <div key={t} className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur">
                <div className="font-mono text-3xl font-black text-white">{n}</div>
                <div className="mt-1 text-[13px] leading-snug text-slate-300">{t}</div>
              </div>
            ))}
          </div>
        </section>
      </div>

      <main className="mx-auto -mt-10 grid max-w-[1280px] grid-cols-1 gap-6 px-4 sm:px-8 [&>*]:min-w-0">
        <section className="rounded-3xl border border-slate-200 bg-white p-4 shadow-xl shadow-slate-300/30 sm:p-6">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-xl font-black">Exam radar</h2>
            <div className="flex items-center gap-1 rounded-full bg-slate-100 p-1 text-sm font-bold" role="group" aria-label="Which papers">
              {([['all', 'All papers'], ['M', 'Midterms'], ['F', 'Finals']] as const).map(([v, label]) => (
                <button key={v} type="button" onClick={() => setWhich(v)} aria-pressed={which === v} className={`rounded-full px-3 py-1 ${which === v ? 'bg-slate-900 text-white' : 'text-slate-500'}`}>{label}</button>
              ))}
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full border-separate border-spacing-y-1 text-sm">
              <thead>
                <tr className="text-xs text-slate-400">
                  <th className="pb-1 text-left font-bold">Question pattern</th>
                  {papers.map((p) => (
                    <th key={p.id} className="px-0.5 pb-1 font-bold">
                      <span className="block whitespace-nowrap">{p.kind === 'M' ? `'${p.id.slice(0, 2)} ${p.id[2]}` : `F '${p.id.slice(1)}`}</span>
                    </th>
                  ))}
                  <th className="pb-1 text-right font-bold">Practise</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(({ t, n }) => (
                  <tr key={t.title} className="group">
                    <td className="rounded-l-xl bg-slate-50 py-2 pl-3 pr-2 group-hover:bg-blue-50">
                      <div className="font-bold text-slate-800">{t.title}</div>
                      <div className="text-[11px] text-slate-400">L{t.lec} · {lecture(t.lec).title} · {n}/{papers.length}</div>
                    </td>
                    {papers.map((p) => {
                      const hit = t.papers.includes(p.id)
                      return (
                        <td key={p.id} className="bg-slate-50 px-0.5 text-center group-hover:bg-blue-50">
                          <span className={`mx-auto block h-6 w-6 rounded-md sm:h-7 sm:w-7 ${hit ? (p.kind === 'M' ? 'bg-amber-400' : 'bg-rose-500') : 'bg-white ring-1 ring-inset ring-slate-200'}`} title={`${t.title} · ${p.label}${hit ? '' : ' (not asked)'}`} />
                        </td>
                      )
                    })}
                    <td className="rounded-r-xl bg-slate-50 py-2 pl-2 pr-3 text-right group-hover:bg-blue-50">
                      <span className="inline-flex flex-wrap justify-end gap-1">
                        {t.links.length ? t.links.map((id) => {
                          const m = moduleById(id)!
                          return <a key={id} href={hrefOf(m)} className={`whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-bold ${m.status === 'lab' ? 'bg-violet-100 text-violet-700 hover:bg-violet-200' : 'bg-emerald-100 text-emerald-700 hover:bg-emerald-200'}`}>{m.title} →</a>
                        }) : <span className="text-xs text-slate-400">flashcards soon</span>}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-slate-400"><span className="mr-1 inline-block h-3 w-3 rounded bg-amber-400 align-middle" /> midterm · <span className="mx-1 inline-block h-3 w-3 rounded bg-rose-500 align-middle" /> final. Sorted by how often the pattern appeared.</p>
        </section>

        <section>
          <h2 className="mb-3 text-xl font-black">Start with these</h2>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {top.map((m) => <ModuleCard key={m.id} m={m} />)}
          </div>
        </section>
      </main>
      <MockupSwitcher current="b" />
    </div>
  )
}
