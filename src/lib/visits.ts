// The visit counts the admin page shows (server/admin.mjs keeps them per day; nothing here identifies a visitor).

export interface DayVisits {
  views: number
  /** people told apart within that day only: the same person on two days counts twice */
  visitors: number
  pages: Record<string, number>
  /** the other sites visitors followed a link from */
  refs: Record<string, number>
  langs: Record<string, number>
}
export interface VisitData {
  host: string
  /** today's date where the server counts (Hong Kong), YYYY-MM-DD */
  today: string
  /** false when this site's visits are not counted (a local copy, say) */
  counted: boolean
  days: Record<string, DayVisits>
}
export interface VisitSummary {
  /** one entry per day of the range, oldest first, days without a visit included */
  series: { day: string; views: number; visitors: number }[]
  views: number
  visitors: number
  today: { views: number; visitors: number }
  /** most opened first */
  pages: [string, number][]
  refs: [string, number][]
  langs: [string, number][]
  /** views that did not come from another site (typed in, a bookmark, a link inside the site, or unknown) */
  direct: number
}

/** the date `back` days before a YYYY-MM-DD date */
export function dayBefore(day: string, back: number): string {
  const [y, m, d] = day.split('-').map(Number)
  return new Date(Date.UTC(y, m - 1, d - back)).toISOString().slice(0, 10)
}

const ranked = (counts: Map<string, number>): [string, number][] => [...counts].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))

/** the last `n` days up to today, added up */
export function summarise(data: Pick<VisitData, 'today' | 'days'>, n: number): VisitSummary {
  const pages = new Map<string, number>()
  const refs = new Map<string, number>()
  const langs = new Map<string, number>()
  const add = (into: Map<string, number>, from: Record<string, number> | undefined) => {
    for (const [k, v] of Object.entries(from ?? {})) into.set(k, (into.get(k) ?? 0) + v)
  }
  const series = Array.from({ length: n }, (_, i) => {
    const day = dayBefore(data.today, n - 1 - i)
    const d = data.days[day]
    add(pages, d?.pages)
    add(refs, d?.refs)
    add(langs, d?.langs)
    return { day, views: d?.views ?? 0, visitors: d?.visitors ?? 0 }
  })
  const views = series.reduce((s, d) => s + d.views, 0)
  const fromSites = [...refs.values()].reduce((s, v) => s + v, 0)
  const last = series[series.length - 1]
  return {
    series,
    views,
    visitors: series.reduce((s, d) => s + d.visitors, 0),
    today: { views: last.views, visitors: last.visitors },
    pages: ranked(pages),
    refs: ranked(refs),
    langs: ranked(langs),
    direct: Math.max(0, views - fromSites),
  }
}
