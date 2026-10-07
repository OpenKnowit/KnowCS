import { useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { EPISODES, LECTURES, MODULES, PAPERS, moduleUrl, watchUrl } from '../../lib/sitemap'
import { usePaperLabel } from '../ui'

/** /papers/: which topic each past paper asked, as a grid — the inverse view of the exam badges. */
export default function PapersPage() {
  const { t } = useTranslation()
  const paper = usePaperLabel()
  useEffect(() => {
    document.title = `${t('papers.title')} · KnowCS`
  }, [t])
  const rows = MODULES.filter((m) => m.exams.length > 0)
  const perPaper = (id: string) => rows.filter((m) => m.exams.includes(id)).length
  return (
    <main className="mx-auto max-w-[1280px] px-4 pb-12 pt-6 sm:px-8">
      <p className="text-xs font-extrabold uppercase tracking-[0.14em] text-blue-600">{t('papers.eyebrow')}</p>
      <h1 className="mt-2 text-[clamp(30px,4vw,44px)] font-black leading-tight tracking-tight">{t('papers.title')}</h1>
      <p className="mt-2 max-w-3xl text-[16px] text-slate-600">{t('papers.lead')}</p>

      <div className="mt-6 overflow-x-auto rounded-3xl border border-slate-200 bg-white shadow-sm">
        <table className="w-full min-w-[860px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left">
              <th className="sticky left-0 z-10 bg-white px-4 py-3 text-xs font-black uppercase tracking-wider text-slate-500">{t('papers.topic')}</th>
              {PAPERS.map((p) => (
                <th key={p.id} className="px-1 py-3 text-center align-bottom">
                  <span className={`inline-block rounded-full px-2 py-0.5 text-[11px] font-black ${p.kind === 'M' ? 'bg-amber-100 text-amber-800' : 'bg-rose-100 text-rose-700'}`}>{paper(p.id)}</span>
                </th>
              ))}
              <th className="px-3 py-3 text-center text-xs font-black uppercase tracking-wider text-slate-500">{t('papers.count')}</th>
            </tr>
          </thead>
          {LECTURES.filter((n) => rows.some((m) => m.lec === n)).map((n) => (
            <tbody key={n}>
              <tr>
                <td colSpan={PAPERS.length + 2} className="sticky left-0 bg-slate-50 px-4 py-1.5 text-[11px] font-black uppercase tracking-wider text-slate-500">
                  {t('site.lecture_long', { n })} · {t(`site.lectures.${n}`)}
                </td>
              </tr>
              {rows
                .filter((m) => m.lec === n)
                .map((m) => {
                  const ep = EPISODES.find((e) => e.modules.includes(m.id))
                  return (
                    <tr key={m.id} className="border-t border-slate-100 hover:bg-blue-50/40">
                      <td className="sticky left-0 z-10 bg-white px-4 py-2.5">
                        <a href={moduleUrl(m.id)} className="font-bold text-slate-900 hover:text-blue-700">{t(`site.modules.${m.id}.title`)}</a>
                        {ep && (
                          <a href={watchUrl(ep.id)} className="ml-2 whitespace-nowrap text-xs font-bold text-blue-600 hover:underline" title={t(`watch.${ep.id}.title`)}>
                            ▶ {t('papers.watch')}
                          </a>
                        )}
                      </td>
                      {PAPERS.map((p) => (
                        <td key={p.id} className="px-1 py-2.5 text-center">
                          {m.exams.includes(p.id) ? (
                            <span className={`inline-block h-4 w-4 rounded-full ${p.kind === 'M' ? 'bg-amber-500' : 'bg-rose-500'}`} role="img" aria-label={t('papers.asked', { paper: paper(p.id) })} />
                          ) : (
                            <span className="inline-block h-1.5 w-1.5 rounded-full bg-slate-200" aria-hidden />
                          )}
                        </td>
                      ))}
                      <td className="px-3 py-2.5 text-center font-mono font-black text-slate-700">{m.exams.length}</td>
                    </tr>
                  )
                })}
            </tbody>
          ))}
          <tfoot>
            <tr className="border-t-2 border-slate-200">
              <td className="sticky left-0 z-10 bg-white px-4 py-3 text-xs font-black uppercase tracking-wider text-slate-500">{t('papers.per_paper')}</td>
              {PAPERS.map((p) => (
                <td key={p.id} className="px-1 py-3 text-center font-mono font-black text-slate-700">{perPaper(p.id)}</td>
              ))}
              <td />
            </tr>
          </tfoot>
        </table>
      </div>
      <p className="mt-3 text-xs text-slate-500">{t('papers.note')}</p>
    </main>
  )
}
