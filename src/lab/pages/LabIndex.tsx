import { LAB, PAPERS, lecture, paperLabel } from '../registry'
import { TopBar } from '../ui'

const HOMES = [
  { id: 'a', title: 'A · Course map', text: 'Lectures in order, each with its pages. Search box and a midterm / final scope switch. Calm and familiar.', tone: 'from-blue-500 to-indigo-600' },
  { id: 'b', title: 'B · Exam radar', text: 'Opens with the 16 recurring question patterns × 9 papers heat map, then the pages that drill the most frequent ones.', tone: 'from-slate-800 to-indigo-700' },
  { id: 'c', title: 'C · Study path', text: 'A timeline of the 11 lectures with “can you do this in the exam?” checklists and progress saved in the browser.', tone: 'from-emerald-500 to-teal-600' },
]

export default function LabIndex() {
  return (
    <>
      <TopBar />
      <main className="mx-auto max-w-[1180px] px-4 pb-16 pt-8 sm:px-6">
        <p className="text-xs font-extrabold uppercase tracking-[0.12em] text-blue-600">KnowCS Lab · prototypes</p>
        <h1 className="mt-1 text-[clamp(28px,3.4vw,40px)] font-black tracking-tight">New pages for COMP2211, one HTML page each</h1>
        <p className="mt-2 max-w-3xl text-slate-600">
          Ten new module pages built from the lectures and {PAPERS.length} past papers, plus three ideas for the home page. Every page is a separate URL with its own back / next navigation, full-width layout, and a “Quiz me” switch that hides the answers in the exam tables.
        </p>

        <h2 className="mb-3 mt-8 text-lg font-black">1 · Pick a home page</h2>
        <div className="grid gap-4 md:grid-cols-3">
          {HOMES.map((h) => (
            <a key={h.id} href={`home-${h.id}.html`} className="group overflow-hidden rounded-3xl border border-slate-200 bg-white transition hover:-translate-y-0.5 hover:shadow-xl">
              <div className={`h-24 bg-gradient-to-br ${h.tone} p-4 font-black text-white`}>
                <span className="text-3xl">{h.id.toUpperCase()}</span>
              </div>
              <div className="p-4">
                <b className="text-[15px] group-hover:text-blue-700">{h.title}</b>
                <p className="mt-1 text-[13px] text-slate-500">{h.text}</p>
              </div>
            </a>
          ))}
        </div>

        <h2 className="mb-3 mt-10 text-lg font-black">2 · The ten new module pages</h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {LAB.map((m, i) => (
            <a key={m.id} href={`${m.id}.html`} className="group rounded-2xl border border-slate-200 bg-white p-4 transition hover:border-blue-300 hover:shadow-lg">
              <div className="text-[11px] font-extrabold uppercase tracking-wider text-slate-400">{i + 1} · L{m.lec} {lecture(m.lec).title}</div>
              <b className="mt-1 block text-[15px] group-hover:text-blue-700">{m.title}</b>
              <p className="mt-1 text-[13px] text-slate-500">{m.blurb}</p>
              <p className="mt-2 text-[11px] font-bold text-amber-700">{m.exams.map(paperLabel).join(' · ')}</p>
            </a>
          ))}
        </div>

        <p className="mt-10 text-sm text-slate-500">
          The current site stays at <a className="font-bold text-blue-600" href="/">knowcs.online</a>. These prototypes are English-only for now; the chosen version will get the usual three languages when it moves into the main site.
        </p>
      </main>
    </>
  )
}
