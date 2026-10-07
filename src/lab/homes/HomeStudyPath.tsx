import { useState } from 'react'
import { ALL_MODULES, LECTURES, hrefOf, moduleById } from '../registry'
import { readVisited } from '../ui'
import { MockupSwitcher, SiteHeader } from './shared'

const CHECKS: Record<number, string[]> = {
  1: ['Predict slicing output, including “Error” and “Empty Array”', 'Write a pairwise-distance one-liner with broadcasting'],
  2: ['Get P(B|E) from P(B), P(E|B), P(E|not B)', 'Run Naive Bayes with Laplace smoothing', 'Use a Gaussian likelihood with the sample σ'],
  3: ['Fill a distance table (Euclidean, Manhattan, Hamming, cosine)', 'Explain why unshuffled D-fold scores 0%', 'Go from a confusion matrix to macro-F1'],
  4: ['Run K-Means rounds until nothing moves', 'Explain the elbow method and outliers'],
  5: ['Fill a perceptron update table', 'Decide whether data is linearly separable'],
  6: ['Pick XOR weights for a 2-2-1 network', 'Do one backprop step with δk = (O−T)O(1−O)', 'Count weights and biases'],
  7: ['Pad an image four ways', 'Run one Otsu iteration', 'Name a kernel from its numbers'],
  8: ['Output shape and parameters for each layer', 'Explain why a CNN needs fewer parameters than an MLP'],
  9: ['Not examined so far — know tensors, autograd and the training loop'],
  10: ['Minimax value of every node', 'Which edges α-β prunes, with α/β at each node'],
  11: ['Three areas of AI ethics', 'Six sources of unfair models'],
}

const CHECK_KEY = 'knowcs-lab:checks'
function readChecks(): Record<string, boolean> {
  try {
    return JSON.parse(localStorage.getItem(CHECK_KEY) ?? '{}') as Record<string, boolean>
  } catch {
    return {}
  }
}

export default function HomeStudyPath() {
  const [visited] = useState(readVisited)
  const [checks, setChecks] = useState(readChecks)
  const toggle = (key: string) => {
    const next = { ...checks, [key]: !checks[key] }
    setChecks(next)
    try {
      localStorage.setItem(CHECK_KEY, JSON.stringify(next))
    } catch {
      /* storage blocked: ticks just won't persist */
    }
  }
  const labIds = ALL_MODULES.filter((m) => m.status === 'lab').map((m) => m.id)
  const seen = labIds.filter((id) => visited[id]).length
  const totalChecks = Object.values(CHECKS).flat().length
  const done = Object.values(checks).filter(Boolean).length
  const lastId = Object.entries(visited).sort((a, b) => b[1] - a[1])[0]?.[0]
  const last = lastId ? moduleById(lastId) : undefined
  const pct = Math.round((done / totalChecks) * 100)

  return (
    <div className="min-h-screen bg-[#f6f7fb] pb-24">
      <SiteHeader />
      <section className="mx-auto grid max-w-[1180px] gap-6 px-4 pt-10 sm:px-8 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div>
          <p className="text-xs font-extrabold uppercase tracking-[0.14em] text-emerald-600">Your COMP2211 path</p>
          <h1 className="mt-2 text-[clamp(30px,4vw,46px)] font-black leading-[1.08] tracking-tight text-slate-900">Eleven lectures. Tick off what you can do in the exam.</h1>
          <p className="mt-3 max-w-2xl text-[17px] text-slate-600">Each step lists the skills past papers test. Open the page, practise, and tick the skill when you can do it without help. Progress stays in this browser.</p>
        </div>
        <div className="flex items-center gap-5 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
          <svg viewBox="0 0 100 100" className="h-24 w-24 shrink-0 -rotate-90" role="img" aria-label={`${pct}% of skills ticked`}>
            <circle cx={50} cy={50} r={42} fill="none" stroke="#e2e8f0" strokeWidth={10} />
            <circle cx={50} cy={50} r={42} fill="none" stroke="#10b981" strokeWidth={10} strokeLinecap="round" strokeDasharray={`${(pct / 100) * 264} 264`} />
          </svg>
          <div>
            <div className="font-mono text-3xl font-black">{pct}%</div>
            <div className="text-sm text-slate-500">{done} of {totalChecks} skills · {seen} of {labIds.length} new pages opened</div>
            {last && <a href={hrefOf(last)} className="mt-2 inline-block rounded-full bg-slate-900 px-3 py-1.5 text-xs font-bold text-white">Continue: {last.title} →</a>}
          </div>
        </div>
      </section>

      <main className="relative mx-auto mt-10 max-w-[1180px] px-4 sm:px-8">
        <div className="absolute bottom-6 left-[38px] top-6 w-[3px] rounded-full bg-gradient-to-b from-amber-300 via-amber-300 to-rose-300 sm:left-[54px]" aria-hidden />
        <ol className="grid gap-5">
          {LECTURES.map((l) => {
            const mods = ALL_MODULES.filter((m) => m.lec === l.n)
            const keys = CHECKS[l.n].map((_, i) => `${l.n}-${i}`)
            const all = keys.every((k) => checks[k])
            const some = keys.some((k) => checks[k])
            const firstFinal = l.n === 7
            return (
              <li key={l.n} className="relative">
                {firstFinal && <div className="mb-3 ml-[60px] text-xs font-extrabold uppercase tracking-[0.14em] text-rose-500 sm:ml-[84px]">Final exam only from here</div>}
                <div className="grid grid-cols-[48px_minmax(0,1fr)] gap-3 sm:grid-cols-[72px_minmax(0,1fr)]">
                  <div className={`relative z-10 mx-auto grid h-12 w-12 place-items-center rounded-full border-4 font-mono text-lg font-black ${all ? 'border-emerald-500 bg-emerald-500 text-white' : some ? 'border-emerald-400 bg-white text-emerald-600' : 'border-white bg-white text-slate-400 shadow'}`}>
                    {all ? '✓' : l.n}
                  </div>
                  <div className="rounded-3xl border border-slate-200 bg-white p-4 sm:p-5">
                    <div className="flex flex-wrap items-center gap-2">
                      <h2 className="text-lg font-black">{l.title}</h2>
                      <span className={`rounded-full px-2 py-0.5 text-[10px] font-black uppercase ${l.scope === 'mid' ? 'bg-amber-100 text-amber-700' : 'bg-rose-100 text-rose-700'}`}>{l.scope === 'mid' ? 'midterm + final' : 'final'}</span>
                    </div>
                    <div className="mt-3 grid gap-4 md:grid-cols-2">
                      <ul className="grid content-start gap-1.5">
                        {CHECKS[l.n].map((c, i) => {
                          const key = `${l.n}-${i}`
                          return (
                            <li key={key}>
                              <label className="flex cursor-pointer items-start gap-2 text-[14px] text-slate-700">
                                <input type="checkbox" checked={!!checks[key]} onChange={() => toggle(key)} className="mt-0.5 h-4 w-4" />
                                <span className={checks[key] ? 'text-slate-400 line-through' : ''}>{c}</span>
                              </label>
                            </li>
                          )
                        })}
                      </ul>
                      <div className="flex flex-wrap content-start gap-2">
                        {mods.length ? mods.map((m) => (
                          <a key={m.id} href={hrefOf(m)} className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[13px] font-bold transition hover:-translate-y-0.5 ${visited[m.id] ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-slate-200 bg-white text-slate-700 hover:border-blue-300'}`}>
                            {visited[m.id] ? '✓' : m.status === 'lab' ? '✦' : '●'} {m.title}
                          </a>
                        )) : <span className="text-[13px] text-slate-400">Notes only</span>}
                      </div>
                    </div>
                  </div>
                </div>
              </li>
            )
          })}
        </ol>
      </main>
      <MockupSwitcher current="c" />
    </div>
  )
}
