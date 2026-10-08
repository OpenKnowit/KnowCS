import { useEffect, useState } from 'react'
import { Printer } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { Latex } from '../../components/Latex'
import { FORMULAS } from '../../data/formulas'
import { LECTURES, moduleUrl, watchUrl } from '../../lib/sitemap'

/** /formulas/: every examined formula on one printable page. */
export default function FormulasPage() {
  const { t } = useTranslation()
  useEffect(() => {
    document.title = `${t('formulas.title')} · KnowCS`
  }, [t])
  // KaTeX's fonts change every formula's size as they arrive, and a font only loads once text needs it: render the
  // sheet invisibly so every face starts loading, then reveal it when they are in (3 s at most) — no visible jumps
  const [fonts, setFonts] = useState(() => typeof document === 'undefined' || !document.fonts)
  useEffect(() => {
    if (fonts) return
    let live = true
    const done = () => live && setFonts(true)
    const t = setTimeout(done, 3000)
    void document.fonts.ready.then(done, done)
    return () => {
      live = false
      clearTimeout(t)
    }
  }, [fonts])
  return (
    <main className="mx-auto max-w-[1100px] px-4 pb-12 pt-6 sm:px-8 print:max-w-none print:p-0">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs font-extrabold uppercase tracking-[0.14em] text-blue-600 print:hidden">{t('formulas.eyebrow')}</p>
          <h1 className="mt-2 text-[clamp(30px,4vw,44px)] font-black leading-tight tracking-tight print:mt-0 print:text-2xl">{t('formulas.title')} · COMP2211</h1>
          <p className="mt-2 max-w-2xl text-[16px] text-slate-600 print:hidden">{t('formulas.lead')}</p>
        </div>
        <button type="button" onClick={() => window.print()} className="inline-flex items-center gap-2 rounded-full bg-slate-900 px-4 py-2 text-sm font-bold text-white hover:bg-blue-700 print:hidden">
          <Printer className="h-4 w-4" /> {t('formulas.print')}
        </button>
      </div>
      <div className={fonts ? undefined : 'invisible'} aria-busy={!fonts}>
      {LECTURES.filter((n) => FORMULAS.some((f) => f.lec === n)).map((n) => (
        <section key={n} className="mt-7 break-inside-avoid print:mt-3" aria-labelledby={`fm-${n}`}>
          <h2 id={`fm-${n}`} className="mb-2 text-sm font-black uppercase tracking-wider text-slate-500 print:mb-1 print:text-[11px]">
            {t('site.lecture_long', { n })} · {t(`site.lectures.${n}`)}
          </h2>
          <div className="grid gap-3 md:grid-cols-2 print:grid-cols-2 print:gap-1.5">
            {FORMULAS.filter((f) => f.lec === n).map((f) => (
              <article key={f.id} className="break-inside-avoid rounded-2xl border border-slate-200 bg-white p-4 print:rounded-md print:p-2 print:shadow-none">
                <div className="flex items-baseline justify-between gap-2">
                  <h3 className="text-[14px] font-black text-slate-900 print:text-[11px]">{t(`formulas.items.${f.id}.name`)}</h3>
                  {f.link && (
                    <a href={f.link.kind === 'watch' ? watchUrl(f.link.id) : moduleUrl(f.link.id)} className="shrink-0 text-xs font-bold text-blue-700 hover:underline print:hidden">
                      {f.link.kind === 'watch' ? '▶' : '✎'} {f.link.kind === 'watch' ? t(`watch.${f.link.id}.title`) : t(`site.modules.${f.link.id}.title`)}
                    </a>
                  )}
                </div>
                <div className="my-2 overflow-x-auto py-1 text-[17px] print:my-1 print:text-[12px]">
                  <Latex formula={f.tex} displayMode />
                </div>
                <p className="text-[13px] leading-snug text-slate-600 print:text-[10px]">{t(`formulas.items.${f.id}.note`)}</p>
              </article>
            ))}
          </div>
        </section>
      ))}
      </div>
    </main>
  )
}
