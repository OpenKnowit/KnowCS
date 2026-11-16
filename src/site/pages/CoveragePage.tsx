import { useEffect, useMemo, useState } from 'react'
import { ChevronRight, Search } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { LIBS, apiEntryId } from '../playground/libs'
import { useRaise } from '../raise'

// /playground/coverage/: how much of each real library the sandbox has. The list of names comes from the real libraries
// (scripts/api-real-names.py), and each name's status from src/data/apiCoverage.json's generator.

type Status = 'c' | 'e' | 'd' | 'x' | 'y' | 'r' | 'n'
interface Group {
  m: string
  lib: string
  names: string[]
  /** one status letter per name */
  code: string
}
interface Coverage {
  versions: Record<string, string>
  groups: Group[]
}

/** best to worst: the order of the bars and the legend; the darker the blue, the more was checked */
const ORDER: Status[] = ['c', 'e', 'y', 'd', 'x', 'r', 'n']
const BAR: Record<Status, string> = { c: 'bg-blue-700', e: 'bg-blue-400', y: 'bg-blue-200', d: 'bg-amber-400', x: 'bg-amber-400', r: 'bg-slate-400', n: 'bg-slate-200' }
const CHIP: Record<Status, string> = {
  c: 'border-blue-700 bg-blue-700 text-white hover:bg-blue-800',
  e: 'border-blue-300 bg-blue-100 text-blue-900 hover:bg-blue-200',
  y: 'border-blue-100 bg-blue-50 text-blue-900',
  d: 'border-amber-400 bg-amber-100 text-amber-900 hover:bg-amber-200',
  x: 'border-amber-300 bg-amber-50 text-amber-900',
  r: 'border-dashed border-slate-400 bg-slate-50 text-slate-500',
  n: 'border-slate-200 bg-white text-slate-400',
}
const FILTERS = ['all', 'has', 'missing', 'differs'] as const
type Filter = (typeof FILTERS)[number]
/** a name that exists but refuses an ordinary call counts as missing */
const missing = (s: Status) => s === 'n' || s === 'r'
const passes = (s: Status, f: Filter) => f === 'all' || (f === 'has' ? !missing(s) : f === 'missing' ? missing(s) : s === 'd' || s === 'x')

const tally = (groups: Group[]) => {
  const n: Record<Status, number> = { c: 0, e: 0, d: 0, x: 0, y: 0, r: 0, n: 0 }
  for (const g of groups) for (const ch of g.code) n[ch as Status]++
  const total = groups.reduce((a, g) => a + g.names.length, 0)
  return { n, total, has: total - n.n - n.r }
}

function Meter({ n, total, label }: { n: Record<Status, number>; total: number; label: string }) {
  return (
    <div role="img" aria-label={label} title={label} className="flex h-2.5 w-full min-w-[120px] gap-[2px] overflow-hidden rounded-[4px]">
      {ORDER.map((s) => (n[s] ? <i key={s} className={BAR[s]} style={{ width: `${(n[s] / (total || 1)) * 100}%` }} /> : null))}
    </div>
  )
}

const hrefOf = (g: Group, name: string) => `/playground/#${g.lib}/${encodeURIComponent(apiEntryId(g.m, name))}`

function NameChip({ g, name, status, label }: { g: Group; name: string; status: Status; label: string }) {
  const cls = `inline-block rounded-md border px-1.5 py-0.5 font-mono text-[11.5px] ${CHIP[status]}`
  if (status === 'c' || status === 'e' || status === 'd') return <a href={hrefOf(g, name)} title={label} className={cls}>{name}</a>
  return <span title={label} className={cls}>{name}</span>
}

export default function CoveragePage() {
  const { t } = useTranslation()
  const raise = useRaise()
  const [data, setData] = useState<Coverage | null>(null)
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<Filter>('all')
  const [open, setOpen] = useState<Set<string>>(() => new Set())
  useEffect(() => {
    let live = true
    void import('../../data/apiCoverage.json').then((m) => live && setData(m.default as Coverage), (e: unknown) => live && raise(e))
    document.title = `${t('site.coverage.title')} · KnowCS`
    return () => {
      live = false
    }
  }, [t, raise])
  const statusText = (s: Status) => t(`site.coverage.status.${s}`)
  const sum = useMemo(() => (data ? tally(data.groups) : null), [data])
  const q = query.trim().toLowerCase()
  const found = useMemo(() => {
    if (!data || !q) return []
    const out: { g: Group; name: string; status: Status }[] = []
    for (const g of data.groups) {
      g.names.forEach((name, i) => {
        if (`${g.m}.${name}`.toLowerCase().includes(q) && passes(g.code[i] as Status, filter)) out.push({ g, name, status: g.code[i] as Status })
      })
    }
    return out
  }, [data, q, filter])
  const flip = (k: string) => setOpen((o) => {
    const next = new Set(o)
    if (next.has(k)) next.delete(k)
    else next.add(k)
    return next
  })
  const pct = (a: number, b: number) => `${Math.round((a / (b || 1)) * 100)}%`
  const meterLabel = (x: ReturnType<typeof tally>) => ORDER.filter((s) => x.n[s]).map((s) => `${statusText(s)}: ${x.n[s]}`).join(', ')

  return (
    <main className="mx-auto max-w-[1180px] px-4 pb-16 pt-6 sm:px-6">
      <h1 className="text-2xl font-black text-slate-900">{t('site.coverage.title')}</h1>
      <p className="mt-2 max-w-[820px] text-[14px] leading-relaxed text-slate-600">{t('site.coverage.lead')}</p>
      {!data || !sum ? (
        <p className="mt-8 text-sm text-slate-500">{t('site.coverage.loading')}</p>
      ) : (
        <>
          <div className="mt-5 grid grid-cols-2 gap-3 md:grid-cols-4">
            {[
              { k: t('site.coverage.stat_total'), v: sum.total },
              { k: t('site.coverage.stat_has'), v: sum.has, d: sum.n.r ? `${pct(sum.has, sum.total)} · ${t('site.coverage.stat_refuses', { n: sum.n.r })}` : pct(sum.has, sum.total) },
              { k: t('site.coverage.stat_checked'), v: sum.n.c, d: pct(sum.n.c, sum.has) + ' ' + t('site.coverage.of_has') },
              { k: t('site.coverage.stat_differs'), v: sum.n.d + sum.n.x, d: t('site.coverage.stat_differs_d') },
            ].map((s) => (
              <div key={s.k} className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5">
                <div className="text-[11px] font-extrabold uppercase tracking-wider text-slate-500">{s.k}</div>
                <div className="mt-0.5 font-mono text-[22px] font-extrabold text-slate-900">{s.v.toLocaleString('en-US')}</div>
                {s.d && <div className="text-xs text-slate-500">{s.d}</div>}
              </div>
            ))}
          </div>

          <ul className="mt-4 flex flex-wrap gap-x-4 gap-y-1.5 text-[12px] text-slate-600" aria-label={t('site.coverage.legend')}>
            {ORDER.filter((s) => s !== 'x').map((s) => (
              <li key={s} className="flex items-center gap-1.5"><i className={`h-2.5 w-2.5 rounded-[3px] ${BAR[s]}`} aria-hidden />{statusText(s)}</li>
            ))}
          </ul>

          <div className="mt-4 flex flex-wrap items-center gap-2">
            <label className="relative min-w-[220px] flex-1 sm:max-w-[360px]">
              <Search size={15} className="absolute left-2.5 top-2.5 text-slate-400" aria-hidden />
              <span className="sr-only">{t('site.coverage.search')}</span>
              <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t('site.coverage.search')} className="w-full rounded-lg border border-slate-300 py-2 pl-8 pr-3 text-[13.5px]" />
            </label>
            <div className="inline-flex overflow-hidden rounded-[10px] border border-slate-300" role="group" aria-label={t('site.coverage.filter')}>
              {FILTERS.map((f) => (
                <button key={f} type="button" aria-pressed={filter === f} onClick={() => setFilter(f)} className={`px-3 py-1.5 text-[12.5px] font-semibold ${filter === f ? 'bg-blue-600 text-white' : 'bg-white text-slate-600 hover:bg-slate-50'}`}>{t(`site.coverage.f_${f}`)}</button>
              ))}
            </div>
          </div>

          {q ? (
            <section className="mt-5" aria-live="polite">
              <p className="mb-2 text-[13px] text-slate-600">{t('site.coverage.found', { count: found.length, n: found.length })}</p>
              <ul className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-3">
                {found.slice(0, 300).map(({ g, name, status }) => (
                  <li key={`${g.m}.${name}`} className="flex items-center gap-2 overflow-hidden text-[12px] text-slate-500">
                    <NameChip g={g} name={name} status={status} label={statusText(status)} />
                    <span className="truncate">{g.m} · {statusText(status)}</span>
                  </li>
                ))}
              </ul>
            </section>
          ) : (
            <div className="mt-5 grid gap-3">
              {LIBS.map((l) => {
                const groups = data.groups.filter((g) => g.lib === l.id).sort((a, b) => b.names.length - a.names.length)
                const x = tally(groups)
                const on = open.has(l.id)
                return (
                  <section key={l.id} className="rounded-2xl border border-slate-200 bg-white">
                    <button type="button" onClick={() => flip(l.id)} aria-expanded={on} className="grid w-full grid-cols-[18px_minmax(0,1fr)] items-center gap-x-2 gap-y-1.5 p-3.5 text-left sm:grid-cols-[18px_170px_minmax(0,1fr)_170px]">
                      <ChevronRight size={15} className={`text-slate-400 transition-transform ${on ? 'rotate-90' : ''}`} aria-hidden />
                      <span className="font-bold text-slate-900">
                        {l.name} <span className="ml-1 font-mono text-[11px] font-normal text-slate-500">{data.versions[l.id]}</span>
                      </span>
                      <span className="col-start-2 sm:col-start-3"><Meter n={x.n} total={x.total} label={meterLabel(x)} /></span>
                      <span className="col-start-2 font-mono text-[12px] text-slate-600 sm:col-start-4 sm:text-right">{x.has} / {x.total} · {pct(x.has, x.total)}</span>
                    </button>
                    {on && (
                      <div className="grid gap-1 border-t border-slate-100 p-2">
                        {groups.map((g) => {
                          const y = tally([g])
                          const k = `${l.id}/${g.m}`
                          const expanded = open.has(k)
                          const shown = g.names.map((name, i) => ({ name, status: g.code[i] as Status })).filter((a) => passes(a.status, filter)).sort((a, b) => ORDER.indexOf(a.status) - ORDER.indexOf(b.status))
                          return (
                            <div key={g.m}>
                              <button type="button" onClick={() => flip(k)} aria-expanded={expanded} className="grid w-full grid-cols-[14px_minmax(0,1fr)] items-center gap-x-2 gap-y-1 rounded-lg px-2 py-1.5 text-left hover:bg-slate-50 sm:grid-cols-[14px_170px_minmax(0,1fr)_170px]">
                                <ChevronRight size={13} className={`text-slate-400 transition-transform ${expanded ? 'rotate-90' : ''}`} aria-hidden />
                                <span className="truncate font-mono text-[12.5px] text-slate-800">{g.m}</span>
                                <span className="col-start-2 sm:col-start-3"><Meter n={y.n} total={y.total} label={meterLabel(y)} /></span>
                                <span className="col-start-2 font-mono text-[11.5px] text-slate-500 sm:col-start-4 sm:text-right">{y.has} / {y.total} · {pct(y.has, y.total)}</span>
                              </button>
                              {expanded && (
                                <div className="flex flex-wrap gap-1 px-2 pb-3 pt-1 sm:pl-6">
                                  {shown.length ? shown.map((a) => <NameChip key={a.name} g={g} name={a.name} status={a.status} label={statusText(a.status)} />) : <span className="text-[12px] text-slate-500">{t('site.coverage.none')}</span>}
                                </div>
                              )}
                            </div>
                          )
                        })}
                      </div>
                    )}
                  </section>
                )
              })}
            </div>
          )}
          <p className="mt-5 max-w-[820px] text-[12px] leading-relaxed text-slate-500">{t('site.coverage.method')}</p>
        </>
      )}
    </main>
  )
}
