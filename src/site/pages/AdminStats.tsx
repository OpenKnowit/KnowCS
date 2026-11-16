import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { summarise } from '../../lib/visits'
import type { VisitData } from '../../lib/visits'

// The visit counts on /admin/ (server/admin.mjs, "Visits"): today, a range, a bar per day, and where the views went and came from.

const RANGES = [7, 30, 90] as const
const TOP = 12

/** rows of a name and its count, each with a bar as long as its share of the largest */
function Ranked({ title, rows, empty }: { title: string; rows: [string, number][]; empty: string }) {
  const max = Math.max(1, ...rows.map((r) => r[1]))
  return (
    <div className="min-w-0">
      <h3 className="text-xs font-bold text-slate-500">{title}</h3>
      {rows.length === 0 ? (
        <p className="mt-2 text-xs text-slate-400">{empty}</p>
      ) : (
        <ol className="mt-2 grid gap-1">
          {rows.map(([name, n]) => (
            <li key={name} className="relative flex items-center justify-between gap-3 overflow-hidden rounded-md px-2 py-1 text-xs">
              <span className="absolute inset-y-0 left-0 rounded-md bg-blue-50" style={{ width: `${(n / max) * 100}%` }} aria-hidden />
              <span className="relative min-w-0 truncate font-mono text-slate-700" title={name}>
                {name}
              </span>
              <b className="relative shrink-0 tabular-nums text-slate-900">{n.toLocaleString()}</b>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

const Tile = ({ label, views, visitors, of }: { label: string; views: number; visitors: number; of: { views: string; visitors: string } }) => (
  <div className="rounded-xl border border-slate-200 px-3.5 py-3">
    <div className="text-xs font-bold text-slate-500">{label}</div>
    <div className="mt-1 flex flex-wrap items-baseline gap-x-4 gap-y-1">
      <span>
        <b className="text-2xl font-extrabold tabular-nums text-slate-900">{views.toLocaleString()}</b> <span className="text-xs text-slate-500">{of.views}</span>
      </span>
      <span>
        <b className="text-2xl font-extrabold tabular-nums text-slate-900">{visitors.toLocaleString()}</b> <span className="text-xs text-slate-500">{of.visitors}</span>
      </span>
    </div>
  </div>
)

export function AdminStats() {
  const { t } = useTranslation()
  const [data, setData] = useState<VisitData | null>(null)
  const [failed, setFailed] = useState(false)
  const [range, setRange] = useState<(typeof RANGES)[number]>(30)
  useEffect(() => {
    let live = true
    fetch(`/api/stats?days=${RANGES[RANGES.length - 1]}`)
      .then((r) => (r.ok ? (r.json() as Promise<VisitData>) : Promise.reject(new Error(String(r.status)))))
      .then((d) => live && setData(d))
      .catch(() => live && setFailed(true))
    return () => {
      live = false
    }
  }, [])
  const s = useMemo(() => (data ? summarise(data, range) : null), [data, range])
  const of = { views: t('site.admin.stats.views'), visitors: t('site.admin.stats.visitors') }
  const peak = Math.max(1, ...(s?.series.map((d) => d.views) ?? []))

  return (
    <section aria-labelledby="admin-stats" className="mt-5 rounded-[14px] border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="admin-stats" className="text-[15px] font-extrabold text-slate-900">
          {t('site.admin.stats.title')}
          {data && <span className="ml-2 font-mono text-xs font-normal text-slate-500">{data.host}</span>}
        </h2>
        <div className="flex rounded-lg bg-slate-100 p-0.5" role="group" aria-label={t('site.admin.stats.range_label')}>
          {RANGES.map((n) => (
            <button key={n} type="button" aria-pressed={range === n} onClick={() => setRange(n)} className={`rounded-md px-2.5 py-1 text-xs font-bold ${range === n ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}>
              {t('site.admin.stats.range', { n })}
            </button>
          ))}
        </div>
      </div>
      {failed && <p className="mt-3 text-sm text-rose-700">{t('site.admin.stats.failed')}</p>}
      {!failed && !s && <p className="mt-3 text-sm text-slate-500">…</p>}
      {data && s && (
        <>
          {!data.counted && <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-900">{t('site.admin.stats.uncounted', { host: data.host })}</p>}
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            <Tile label={t('site.admin.stats.today')} views={s.today.views} visitors={s.today.visitors} of={of} />
            <Tile label={t('site.admin.stats.total', { n: range })} views={s.views} visitors={s.visitors} of={of} />
          </div>
          {s.views === 0 ? (
            <p className="mt-4 text-sm text-slate-500">{t('site.admin.stats.empty')}</p>
          ) : (
            <>
              {/* page views per day, oldest on the left; hover or focus a bar for its numbers */}
              <div className="mt-4">
                <div className="flex items-center justify-between text-[11px] text-slate-500">
                  <span>{t('site.admin.stats.per_day')}</span>
                  <span className="tabular-nums">{t('site.admin.stats.peak', { n: peak.toLocaleString() })}</span>
                </div>
                <ol className="mt-1 flex h-24 items-end gap-px border-b border-slate-300" aria-label={t('site.admin.stats.per_day')}>
                  {s.series.map((d) => {
                    const label = t('site.admin.stats.bar', { day: d.day, views: d.views, visitors: d.visitors })
                    return (
                      <li key={d.day} title={label} aria-label={label} className="group flex h-full min-w-0 flex-1 items-end">
                        <span className="w-full rounded-t-[2px] bg-blue-500 group-hover:bg-blue-700" style={{ height: d.views ? `max(2px, ${(d.views / peak) * 100}%)` : 0 }} />
                      </li>
                    )
                  })}
                </ol>
                <div className="mt-1 flex justify-between font-mono text-[11px] tabular-nums text-slate-500">
                  <span>{s.series[0].day}</span>
                  <span>{s.series[s.series.length - 1].day}</span>
                </div>
              </div>
              <div className="mt-4 grid gap-5 md:grid-cols-3">
                <Ranked title={t('site.admin.stats.pages')} rows={s.pages.slice(0, TOP)} empty="–" />
                <Ranked title={t('site.admin.stats.refs')} rows={[...s.refs.slice(0, TOP - 1), [t('site.admin.stats.direct'), s.direct] as [string, number]].filter((r) => r[1] > 0)} empty="–" />
                <Ranked title={t('site.admin.stats.langs')} rows={s.langs} empty="–" />
              </div>
            </>
          )}
          <p className="mt-4 text-[11px] leading-relaxed text-slate-500">{t('site.admin.stats.how')}</p>
        </>
      )}
    </section>
  )
}
