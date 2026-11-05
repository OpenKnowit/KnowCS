import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { LayoutGrid, List, Search, Star } from 'lucide-react'
import { LECTURES, MODULES, STARRED, moduleUrl } from '../../lib/sitemap'
import type { ModuleInfo } from '../../lib/sitemap'
import LabArt from '../LabArt'

// Visualize: every lab as a card whose picture is a small drawing of what the lab is about.

type View = 'grid' | 'chapters'
const VIEW_KEY = 'knowcs-home-view'
const readView = (): View => {
  // links like /#lecture-3 (the module bar) want the chapter they name
  if (window.location.hash.startsWith('#lecture-')) return 'chapters'
  try {
    return localStorage.getItem(VIEW_KEY) === 'chapters' ? 'chapters' : 'grid'
  } catch {
    return 'grid'
  }
}

function LabCard({ m, compact = false }: { m: ModuleInfo; compact?: boolean }) {
  const { t } = useTranslation()
  const starred = STARRED.has(m.id)
  return (
    <a href={moduleUrl(m.id)} className="group flex min-w-0 flex-col overflow-hidden rounded-[14px] border border-slate-200 bg-white transition hover:-translate-y-0.5 hover:border-blue-300 hover:shadow-lg hover:shadow-blue-100">
      <div className="relative aspect-[8/5] border-b border-slate-200 bg-slate-100">
        <LabArt id={m.id} />
        {starred && (
          <span className="absolute left-2.5 top-2.5 grid h-[26px] w-[26px] place-items-center rounded-full bg-white shadow-md shadow-slate-900/20" title={t('site.home.starred')}>
            <Star size={15} className="fill-amber-500 text-amber-500" aria-hidden />
            <span className="sr-only">{t('site.home.starred')}</span>
          </span>
        )}
      </div>
      <div className={`flex flex-col gap-1.5 ${compact ? 'px-3 pb-3 pt-2.5' : 'px-4 pb-4 pt-3.5'}`}>
        {!compact && (
          <span className="text-[11px] font-bold uppercase tracking-[0.06em] text-blue-600">
            {t('site.lecture_short', { n: m.lec })} · {t(`site.lectures.${m.lec}`)}
          </span>
        )}
        <b className={`font-bold text-slate-900 group-hover:text-blue-700 ${compact ? 'text-sm' : 'text-base'}`}>{t(`site.modules.${m.id}.title`)}</b>
        {!compact && <p className="text-[13px] leading-normal text-slate-500">{t(`site.modules.${m.id}.blurb`)}</p>}
      </div>
    </a>
  )
}

function ViewSwitch({ view, onChange }: { view: View; onChange: (v: View) => void }) {
  const { t } = useTranslation()
  return (
    <div className="inline-flex gap-0.5 rounded-[10px] bg-slate-200/70 p-[3px]" role="group" aria-label={t('site.home.view_label')}>
      {(['grid', 'chapters'] as const).map((v) => (
        <button
          key={v}
          type="button"
          onClick={() => onChange(v)}
          aria-pressed={view === v}
          className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[13px] transition ${view === v ? 'bg-white font-semibold text-slate-900 shadow-sm' : 'font-medium text-slate-500 hover:text-slate-800'}`}
        >
          {v === 'grid' ? <LayoutGrid size={15} aria-hidden /> : <List size={15} aria-hidden />}
          {t(`site.home.view_${v}`)}
        </button>
      ))}
    </div>
  )
}

export default function HomePage() {
  const { t } = useTranslation()
  const [q, setQ] = useState('')
  const [view, setViewState] = useState<View>(readView)
  const setView = (v: View) => {
    setViewState(v)
    try {
      localStorage.setItem(VIEW_KEY, v)
    } catch {
      // private mode: the choice just isn't remembered
    }
  }
  useEffect(() => {
    document.title = `KnowCS · ${t('site.home.tab_title')}`
  }, [t])
  // a /#lecture-n link scrolls to its chapter once the chapters are on the page
  useEffect(() => {
    if (view === 'chapters' && window.location.hash.startsWith('#lecture-')) document.getElementById(window.location.hash.slice(1))?.scrollIntoView()
  }, [view])

  const query = q.trim().toLowerCase()
  const matches = (m: ModuleInfo) =>
    !query || `${t(`site.modules.${m.id}.title`)} ${t(`site.modules.${m.id}.blurb`)} ${t(`site.lectures.${m.lec}`)}`.toLowerCase().includes(query)
  const shown = MODULES.filter(matches)

  return (
    <main className="px-[clamp(1rem,9vw,13rem)] pb-16 pt-6 lg:pt-8">
      <h1 className="text-[clamp(24px,2.6vw,30px)] font-extrabold tracking-tight text-slate-900">{t('site.home.title_full')}</h1>
      <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
        <label className="flex h-[42px] w-full max-w-[420px] items-center gap-2.5 rounded-[10px] border border-slate-200 bg-white px-3.5 focus-within:border-blue-400 focus-within:ring-2 focus-within:ring-blue-100">
          <Search size={17} className="shrink-0 text-slate-500" aria-hidden />
          <span className="sr-only">{t('site.home.find')}</span>
          <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder={t('site.home.find_placeholder')} className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-slate-500" />
        </label>
        <ViewSwitch view={view} onChange={setView} />
      </div>

      {shown.length === 0 && <p className="mt-10 text-center text-slate-500">{t('site.home.no_match', { q: q.trim() })}</p>}

      {/* both views stay mounted, so switching never reloads a thumbnail */}
      <div className={`${view === 'grid' ? 'grid' : 'hidden'} mt-6 gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4`}>
          {shown.map((m) => (
            <LabCard key={m.id} m={m} />
          ))}
      </div>
      <div className={`${view === 'chapters' ? 'grid' : 'hidden'} mt-6 gap-4`}>
          {LECTURES.map((n) => {
            const labs = shown.filter((m) => m.lec === n)
            if (!labs.length) return null
            return (
              <section key={n} id={`lecture-${n}`} className="flex scroll-mt-4 flex-col gap-5 rounded-2xl border border-slate-200 bg-white p-5 md:flex-row">
                <div className="md:w-[200px] md:shrink-0">
                  <div className="font-mono text-[28px] font-extrabold leading-none text-slate-300">{String(n).padStart(2, '0')}</div>
                  <h2 className="mt-2 text-base font-extrabold text-slate-900">{t(`site.lectures.${n}`)}</h2>
                </div>
                <div className="grid flex-1 grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
                  {labs.map((m) => (
                    <LabCard key={m.id} m={m} compact />
                  ))}
                </div>
              </section>
            )
          })}
      </div>
    </main>
  )
}
