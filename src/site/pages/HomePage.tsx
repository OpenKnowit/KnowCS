import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { EPISODES, LECTURES, MODULES, PAPERS, lectureScope, moduleUrl, noteUrl, watchUrl } from '../../lib/sitemap'
import type { ModuleInfo } from '../../lib/sitemap'
import { readVisited } from '../ui'
import { EpisodeCard } from '../watch/EpisodeCard'

type Scope = 'all' | 'mid' | 'final'

function ExamDots({ papers }: { papers: string[] }) {
  return (
    <span className="inline-flex items-center gap-[3px]" aria-hidden>
      {PAPERS.map((p) => (
        <i key={p.id} className={`inline-block h-2 w-2 rounded-full ${papers.includes(p.id) ? (p.kind === 'M' ? 'bg-amber-500' : 'bg-rose-500') : 'bg-slate-200'}`} />
      ))}
    </span>
  )
}

function ModuleCard({ m, seen }: { m: ModuleInfo; seen: boolean }) {
  const { t } = useTranslation()
  return (
    <a href={moduleUrl(m.id)} className="group flex flex-col gap-1.5 rounded-2xl border border-slate-200 bg-white p-4 transition hover:-translate-y-0.5 hover:border-blue-300 hover:shadow-lg hover:shadow-blue-100">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-extrabold uppercase tracking-wider text-slate-400">{t('site.lecture_short', { n: m.lec })} · {t(`site.lectures.${m.lec}`)}</span>
        {seen && <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-black uppercase text-emerald-700">{t('site.home.seen')}</span>}
      </div>
      <b className="text-[15px] font-extrabold text-slate-900 group-hover:text-blue-700">{t(`site.modules.${m.id}.title`)}</b>
      <p className="text-[13px] leading-snug text-slate-500">{t(`site.modules.${m.id}.blurb`)}</p>
      {m.exams.length > 0 && (
        <div className="mt-auto flex items-center gap-2 pt-1 text-[11px] font-bold text-slate-500">
          <ExamDots papers={m.exams} /> {t('site.home.in_papers', { n: m.exams.length, total: PAPERS.length })}
        </div>
      )}
    </a>
  )
}

export default function HomePage() {
  const { t } = useTranslation()
  const [scope, setScope] = useState<Scope>('all')
  const [q, setQ] = useState('')
  const [visited] = useState(readVisited)
  const query = q.trim().toLowerCase()
  useEffect(() => {
    document.title = `KnowCS · ${t('site.home.tab_title')}`
  }, [t])
  const matches = (m: ModuleInfo) => !query || `${t(`site.modules.${m.id}.title`)} ${t(`site.modules.${m.id}.blurb`)}`.toLowerCase().includes(query)

  return (
    <main className="pb-10">
      <section className="mx-auto max-w-[1280px] px-4 pb-6 pt-8 sm:px-8">
        <div className="grid items-end gap-6 lg:grid-cols-[minmax(0,1fr)_420px]">
          <div>
            <p className="text-xs font-extrabold uppercase tracking-[0.14em] text-blue-600">{t('site.home.eyebrow')}</p>
            <h1 className="mt-2 text-[clamp(32px,4.4vw,52px)] font-black leading-[1.05] tracking-tight text-slate-900">
              {t('site.home.title_a')}<span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">{t('site.home.title_b')}</span>
            </h1>
            <p className="mt-3 max-w-2xl text-[17px] text-slate-600">{t('site.home.lead')}</p>
            <div className="mt-5 flex flex-wrap gap-2 text-sm font-bold">
              <span className="rounded-full bg-white px-3 py-1.5 text-slate-700 shadow-sm">{t('site.home.stat_pages', { n: MODULES.length })}</span>
              <span className="rounded-full bg-white px-3 py-1.5 text-slate-700 shadow-sm">{t('site.home.stat_papers', { n: PAPERS.length })}</span>
              <a href={noteUrl()} className="rounded-full bg-white px-3 py-1.5 text-blue-700 shadow-sm hover:bg-blue-50">{t('site.home.stat_notes')} →</a>
            </div>
          </div>
          <div className="rounded-3xl border border-slate-200 bg-white p-4 shadow-sm">
            <label className="text-xs font-bold text-slate-500" htmlFor="topic-search">{t('site.home.find')}</label>
            <input id="topic-search" value={q} onChange={(e) => setQ(e.target.value)} placeholder={t('site.home.find_placeholder')} className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2.5 text-[15px] outline-none focus:ring-2 focus:ring-blue-300" />
            <div className="mt-3 grid grid-cols-3 gap-1 rounded-xl bg-slate-100 p-1 text-sm font-bold" role="group" aria-label={t('site.home.scope')}>
              {(['all', 'mid', 'final'] as const).map((v) => (
                <button key={v} type="button" onClick={() => setScope(v)} aria-pressed={scope === v} className={`rounded-lg py-1.5 ${scope === v ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500'}`}>
                  {t(`site.home.scope_${v}`)}
                </button>
              ))}
            </div>
            <p className="mt-2 text-xs text-slate-400">{t(`site.home.scope_${scope}_hint`)}</p>
          </div>
        </div>
      </section>

      {!query && (
        <section className="mx-auto mb-6 max-w-[1280px] px-4 sm:px-8" aria-labelledby="home-watch">
          <div className="rounded-3xl bg-slate-900 p-4 text-white sm:p-6">
            <div className="flex flex-wrap items-end justify-between gap-2">
              <div>
                <h2 id="home-watch" className="text-xl font-black">▶ {t('site.home.watch_title')}</h2>
                <p className="mt-1 text-sm text-slate-300">{t('site.home.watch_lead')}</p>
              </div>
              <a href={watchUrl()} className="text-sm font-bold text-blue-300 hover:text-blue-200">{t('site.home.watch_all')} →</a>
            </div>
            <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {EPISODES.slice(0, 3).map((e) => <EpisodeCard key={e.id} info={e} />)}
            </div>
          </div>
        </section>
      )}

      <div className="mx-auto grid max-w-[1280px] gap-4 px-4 sm:px-8">
        {LECTURES.filter((n) => scope !== 'mid' || lectureScope(n) === 'mid').map((n) => {
          const mods = MODULES.filter((m) => m.lec === n && matches(m))
          if (query && !mods.length) return null
          const focus = scope === 'final' && n >= 6
          return (
            <section key={n} id={`lecture-${n}`} className={`grid scroll-mt-4 gap-4 rounded-3xl border p-4 sm:p-5 md:grid-cols-[220px_minmax(0,1fr)] ${focus ? 'border-rose-200 bg-rose-50/40' : 'border-slate-200 bg-white/60'}`}>
              <div className="flex items-start gap-3 md:block">
                <div className="font-mono text-4xl font-black leading-none text-slate-200 md:text-5xl" aria-hidden>{String(n).padStart(2, '0')}</div>
                <div>
                  <h2 className="mt-1 text-lg font-black text-slate-900 md:mt-2">
                    <span className="sr-only">{t('site.lecture_long', { n })}: </span>
                    {t(`site.lectures.${n}`)}
                  </h2>
                  <span className={`mt-1 inline-block rounded-full px-2 py-0.5 text-[10px] font-black uppercase ${lectureScope(n) === 'mid' ? 'bg-amber-100 text-amber-700' : 'bg-rose-100 text-rose-700'}`}>
                    {t(lectureScope(n) === 'mid' ? 'site.home.badge_mid' : 'site.home.badge_final')}
                  </span>
                </div>
              </div>
              {mods.length ? (
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                  {mods.map((m) => <ModuleCard key={m.id} m={m} seen={!!visited[m.id]} />)}
                </div>
              ) : (
                <div className="grid place-items-center rounded-2xl border border-dashed border-slate-300 p-6 text-center text-sm text-slate-500">
                  {n === 9 ? (
                    <a href={noteUrl('pytorch')} className="text-blue-700 hover:underline">{t('site.home.empty_9')}</a>
                  ) : n === 11 ? t('site.home.empty_11') : t('site.home.empty')}
                </div>
              )}
            </section>
          )
        })}
      </div>
    </main>
  )
}
