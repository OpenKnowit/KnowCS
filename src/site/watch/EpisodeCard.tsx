/* eslint-disable react-refresh/only-export-components -- card plus its duration helper */
import { PlayCircle } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { fmtClock, timeline } from '../../lib/explainer'
import { watchUrl } from '../../lib/sitemap'
import type { EpisodeInfo } from '../../lib/sitemap'
import { Frame } from './Player'
import type { Episode } from './Player'
import { useEpisode } from './useEpisode'
export { readWatched } from './watched'

export const lengthOf = (e: Episode) => timeline(e.scenes).total


export function EpisodeCard({ info, big = false, watched = false }: { info: EpisodeInfo; big?: boolean; watched?: boolean }) {
  const { t } = useTranslation()
  const ep = useEpisode(info.id)
  return (
    <a href={watchUrl(info.id)} className="group block overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm transition hover:-translate-y-0.5 hover:border-blue-300 hover:shadow-xl hover:shadow-blue-100">
      <div className="relative">
        {ep ? <Frame episode={ep} t={info.thumb} /> : <div className="aspect-video w-full animate-pulse bg-[#0e1117]" />}
        <span className="absolute inset-0 grid place-items-center bg-slate-900/0 transition group-hover:bg-slate-900/25">
          <PlayCircle className="h-14 w-14 text-white opacity-0 drop-shadow-lg transition group-hover:opacity-100" />
        </span>
        <span className="absolute bottom-2 right-2 rounded-md bg-slate-900 px-1.5 py-0.5 font-mono text-[11px] font-bold text-white">{ep ? fmtClock(lengthOf(ep)) : '…'}</span>
        {watched && <span className="absolute left-2 top-2 rounded-full bg-emerald-700 px-2 py-0.5 text-[10px] font-black uppercase text-white shadow">{t('watch.ui.watched')}</span>}
      </div>
      <div className="p-4">
        <div className="text-[11px] font-extrabold uppercase tracking-wider text-slate-500">
          {t('site.lecture_short', { n: info.lec })} · {t(`site.lectures.${info.lec}`)}
        </div>
        <b className={`mt-1 block font-extrabold text-slate-900 group-hover:text-blue-700 ${big ? 'text-xl' : 'text-[15px]'}`}>{t(`watch.${info.id}.title`)}</b>
        <p className="mt-1 text-[13px] leading-snug text-slate-500">{t(`watch.${info.id}.sub`)}</p>
      </div>
    </a>
  )
}
