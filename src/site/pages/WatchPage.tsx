import { useEffect, useMemo, useRef, useState } from 'react'
import { Clock } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { EPISODES, LECTURES, lectureScope, moduleUrl, watchUrl } from '../../lib/sitemap'
import type { EpisodeInfo } from '../../lib/sitemap'
import { fmtClock, locate, timeline } from '../../lib/explainer'
import { Player, capKey } from '../watch/Player'
import { EpisodeCard, lengthOf, readWatched } from '../watch/EpisodeCard'
import type { Episode } from '../watch/Player'
import { useEpisode } from '../watch/useEpisode'
import { usePaperLabel } from '../ui'

function Gallery() {
  const { t } = useTranslation()
  const [watched] = useState(readWatched)
  useEffect(() => {
    document.title = `${t('watch.ui.gallery_title')} · KnowCS`
  }, [t])
  return (
    <main className="mx-auto max-w-[1280px] px-4 pb-10 pt-6 sm:px-8">
      <p className="text-xs font-extrabold uppercase tracking-[0.14em] text-blue-600">{t('watch.ui.gallery_eyebrow')}</p>
      <h1 className="mt-2 text-[clamp(30px,4vw,46px)] font-black leading-tight tracking-tight">{t('watch.ui.gallery_title')}</h1>
      <p className="mt-2 max-w-2xl text-[17px] text-slate-600">{t('watch.ui.gallery_lead')}</p>
      <p className="mt-3 text-sm font-bold text-slate-500">{t('watch.ui.gallery_count', { n: EPISODES.length, done: EPISODES.filter((e) => watched[e.id]).length })}</p>
      {LECTURES.filter((n) => EPISODES.some((e) => e.lec === n)).map((n) => (
        <section key={n} className="mt-8" aria-labelledby={`watch-lec-${n}`}>
          <h2 id={`watch-lec-${n}`} className="flex items-center gap-2 text-sm font-black uppercase tracking-wider text-slate-500">
            {t('site.lecture_long', { n })} · {t(`site.lectures.${n}`)}
            <span className={`rounded-full px-2 py-0.5 text-[10px] ${lectureScope(n) === 'mid' ? 'bg-amber-100 text-amber-700' : 'bg-rose-100 text-rose-700'}`}>{t(lectureScope(n) === 'mid' ? 'site.home.badge_mid' : 'site.home.badge_final')}</span>
          </h2>
          <div className="mt-3 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {EPISODES.filter((e) => e.lec === n).map((e) => <EpisodeCard key={e.id} info={e} watched={!!watched[e.id]} />)}
          </div>
        </section>
      ))}
    </main>
  )
}

function Transcript({ episode, now, onSeek }: { episode: Episode; now: number; onSeek: (t: number) => void }) {
  const { t } = useTranslation()
  const tl = useMemo(() => timeline(episode.scenes), [episode])
  const cur = locate(tl, episode.scenes, now).i
  return (
    <ol className="grid gap-1">
      {episode.scenes.map((s, i) => (
        <li key={s.id}>
          <button type="button" onClick={() => onSeek(tl.starts[i])} className={`w-full rounded-xl px-3 py-2.5 text-left transition ${i === cur ? 'bg-blue-50 ring-1 ring-blue-200' : 'hover:bg-slate-50'}`}>
            <span className="flex items-baseline gap-2">
              <span className="font-mono text-xs font-bold text-blue-600">{fmtClock(tl.starts[i])}</span>
              <b className="text-[14px] text-slate-900">{t(`watch.${episode.id}.scenes.${s.id}.title`)}</b>
            </span>
            <span className="mt-1 block text-[13px] leading-relaxed text-slate-600">{s.cues.map((_, k) => t(capKey(episode.id, s.id, k))).join(' ')}</span>
          </button>
        </li>
      ))}
    </ol>
  )
}

function EpisodePage({ info }: { info: EpisodeInfo }) {
  const { t } = useTranslation()
  const paper = usePaperLabel()
  const ep = useEpisode(info.id)
  const [now, setNow] = useState(0)
  const [shown, setShown] = useState(false)
  const seek = useRef<((t: number) => void) | null>(null)
  const idx = EPISODES.findIndex((e) => e.id === info.id)
  const next = EPISODES[idx + 1] ?? null
  const points = Object.keys(t(`watch.${info.id}.exam.points`, { returnObjects: true }) as Record<string, string>)
  useEffect(() => {
    document.title = `${t(`watch.${info.id}.title`)} · KnowCS`
  }, [info.id, t])

  return (
    <main className="mx-auto max-w-[1280px] px-4 pb-10 pt-4 sm:px-8">
      <nav className="text-[13px] text-slate-500" aria-label={t('site.ui.breadcrumb')}>
        <a href={watchUrl()} className="font-semibold text-slate-500 hover:text-blue-600">{t('app.nav.watch')}</a>
        {' / '}
        <span>{t('site.lecture_short', { n: info.lec })} · {t(`site.lectures.${info.lec}`)}</span>
      </nav>
      <h1 className="mt-2 text-[clamp(26px,3.4vw,40px)] font-black leading-tight tracking-tight">{t(`watch.${info.id}.title`)}</h1>
      <p className="mt-1 max-w-3xl text-[16px] text-slate-600">{t(`watch.${info.id}.sub`)}</p>
      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        <span className="inline-flex items-center gap-1 rounded-full bg-white px-2.5 py-0.5 text-xs font-bold text-slate-600 shadow-sm">
          <Clock className="h-3 w-3" /> {ep ? fmtClock(lengthOf(ep)) : '…'}
        </span>
        {info.exams.map((p) => (
          <span key={p} className="whitespace-nowrap rounded-full border border-amber-200 bg-amber-50 px-2.5 py-0.5 text-xs font-bold text-amber-800">{paper(p)}</span>
        ))}
      </div>

      <div className="mt-5">
        {ep ? <Player episode={ep} onTime={setNow} seekRef={seek} /> : <div className="aspect-video w-full animate-pulse rounded-2xl bg-[#0e1117]" aria-busy="true" />}
        <p className="mt-2 hidden text-center text-xs text-slate-500 sm:block">{t('watch.ui.keys')}</p>
        <p className="mt-2 text-center text-xs font-semibold text-slate-500 sm:hidden">{t('watch.ui.rotate')}</p>
      </div>

      <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,1fr)_400px]">
        <section className="rounded-3xl border border-slate-200 bg-white p-4 sm:p-6">
          <h2 className="mb-3 text-lg font-black">{t('watch.ui.transcript')}</h2>
          {ep && (
            <Transcript
              episode={ep}
              now={now}
              onSeek={(v) => {
                seek.current?.(v)
                // shareable link to this chapter
                history.replaceState(null, '', `#t=${Math.round(v / 1000)}`)
              }}
            />
          )}
        </section>
        <aside className="grid content-start gap-5">
          <section className="rounded-3xl border border-amber-200 bg-amber-50/60 p-5">
            <h2 className="text-lg font-black text-amber-900">{t('watch.ui.exam_corner')}</h2>
            <ul className="mt-3 grid gap-2.5 text-[14px] leading-relaxed text-slate-700">
              {points.map((k) => (
                <li key={k} className="flex gap-2">
                  <span className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-amber-500" />
                  <span>{t(`watch.${info.id}.exam.points.${k}`)}</span>
                </li>
              ))}
            </ul>
            <div className="mt-4 rounded-2xl bg-white p-4 shadow-sm">
              <div className="text-[11px] font-black uppercase tracking-wider text-amber-700">{t(`watch.${info.id}.exam.source`)}</div>
              <p className="mt-1 text-[14px] font-semibold leading-relaxed text-slate-800">{t(`watch.${info.id}.exam.q`)}</p>
              {shown ? (
                <p className="mt-2 rounded-xl bg-emerald-50 p-3 text-[14px] leading-relaxed text-emerald-900">{t(`watch.${info.id}.exam.a`)}</p>
              ) : (
                <button type="button" onClick={() => setShown(true)} className="mt-3 rounded-full bg-slate-900 px-4 py-1.5 text-sm font-bold text-white hover:bg-slate-700">
                  {t('watch.ui.show_answer')}
                </button>
              )}
            </div>
          </section>
          {info.modules.length > 0 && (
          <section className="rounded-3xl border border-slate-200 bg-white p-5">
            <h2 className="text-lg font-black">{t('watch.ui.practise')}</h2>
            <div className="mt-3 grid gap-2">
              {info.modules.map((m) => (
                <a key={m} href={moduleUrl(m)} className="rounded-xl border border-slate-200 px-3 py-2.5 hover:border-blue-300 hover:bg-blue-50/40">
                  <b className="block text-[14px] text-slate-900">{t(`site.modules.${m}.title`)} →</b>
                  <span className="text-[12px] text-slate-500">{t(`site.modules.${m}.blurb`)}</span>
                </a>
              ))}
            </div>
          </section>
          )}
          {next && (
            <section>
              <h2 className="mb-2 text-sm font-black uppercase tracking-wider text-slate-500">{t('watch.ui.up_next')}</h2>
              <EpisodeCard info={next} />
            </section>
          )}
        </aside>
      </div>
    </main>
  )
}

/** /watch/ (gallery) and /watch/<id>/ (one explainer). */
export default function WatchPage({ id }: { id: string | null }) {
  const info = id ? EPISODES.find((e) => e.id === id) : null
  return info ? <EpisodePage info={info} /> : <Gallery />
}
