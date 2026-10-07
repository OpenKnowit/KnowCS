import { useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { drillUrl, formulasUrl, watchUrl } from '../../lib/sitemap'

/** /404.html: shown by the server for unknown paths (once Nginx's error_page points here). */
export default function NotFoundPage() {
  const { t } = useTranslation()
  useEffect(() => {
    document.title = `${t('notfound.title')} · KnowCS`
  }, [t])
  const link = 'rounded-full bg-white px-4 py-2 text-sm font-bold text-blue-700 shadow-sm hover:bg-blue-50'
  return (
    <main className="mx-auto grid min-h-[60vh] max-w-2xl place-content-center px-4 py-16 text-center">
      <p className="font-mono text-6xl font-black text-slate-300" aria-hidden>404</p>
      <h1 className="mt-4 text-3xl font-black tracking-tight">{t('notfound.title')}</h1>
      <p className="mt-2 text-slate-600">{t('notfound.lead')}</p>
      <div className="mt-6 flex flex-wrap justify-center gap-2">
        <a href="/" className={link}>{t('app.nav.course')}</a>
        <a href={watchUrl()} className={link}>{t('app.nav.watch')}</a>
        <a href={drillUrl()} className={link}>{t('app.nav.drill')}</a>
        <a href={formulasUrl()} className={link}>{t('formulas.title')}</a>
      </div>
    </main>
  )
}
