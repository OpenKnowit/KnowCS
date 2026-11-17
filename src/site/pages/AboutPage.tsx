import { useEffect } from 'react'
import type { ReactNode } from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { ArrowUpRight, Mail, Star } from 'lucide-react'
import authorPhoto from '../../assets/about/zhangjiahao.webp'
import supervisorPhoto from '../../assets/about/desmond.webp'
import inspirationPhoto from '../../assets/about/moyunxiang.webp'

// /about/: who made the site, who helped, where the code is and what to do about a mistake. Strings are in about.*.

const REPO = 'https://github.com/OpenKnowit/KnowCS'
const CONTACT = 'jzhangkb@connect.ust.hk'

// Names are written here, not in the locale files: the zh-HK file is generated and would rewrite them.
// The photos are shown whole, at their own proportions (w × h is the file's size, so the page does not jump).
const AUTHOR = { name: 'Zhang Jiahao (Charles)', zh: '张嘉豪', photo: authorPhoto, w: 652, h: 878 }
const HELPERS = [
  { key: 'supervisor', name: 'Dr. Desmond Tsoi', zh: null, photo: supervisorPhoto, w: 640, h: 853 },
  { key: 'inspiration', name: 'Mo Yunxiang (Kevin)', zh: '莫云翔', photo: inspirationPhoto, w: 600, h: 600, url: 'https://github.com/moyunxiang' },
] as const

const BUILT = [
  { key: 'convnet', name: 'ConvNetJS', url: 'https://cs.stanford.edu/people/karpathy/convnetjs/' },
  { key: 'dragology', name: 'Dragology', url: 'https://declarative-dragging.github.io' },
  { key: 'pyodide', name: 'Pyodide', url: 'https://pyodide.org/' },
] as const

const link = 'font-bold text-blue-700 underline decoration-blue-200 underline-offset-2 hover:decoration-blue-600'
const out = { target: '_blank', rel: 'noopener noreferrer' }

const Block = ({ title, children }: { title: string; children: ReactNode }) => (
  <section className="rounded-3xl border border-slate-200 bg-white px-6 py-5 shadow-sm">
    <h2 className="text-xs font-black uppercase tracking-[0.14em] text-slate-500">{title}</h2>
    <div className="mt-2 text-[15px] leading-relaxed text-slate-700">{children}</div>
  </section>
)

export default function AboutPage() {
  const { t } = useTranslation()
  useEffect(() => {
    document.title = `${t('about.title')} · KnowCS`
  }, [t])
  return (
    <main className="mx-auto max-w-[1000px] px-4 pb-12 pt-6 sm:px-8">
      <p className="text-xs font-extrabold uppercase tracking-[0.14em] text-blue-600">{t('about.eyebrow')}</p>
      <h1 className="mt-2 text-[clamp(30px,4vw,44px)] font-black leading-tight tracking-tight">{t('about.title')}</h1>
      <p className="mt-2 max-w-3xl text-[16px] text-slate-600">{t('about.lead')}</p>

      {/* the author, large; the people who helped come after, smaller */}
      <section className="mt-6 overflow-hidden rounded-3xl border border-slate-200 bg-gradient-to-br from-blue-50 via-white to-white shadow-sm">
        <div className="flex flex-col gap-5 p-5 sm:flex-row sm:items-center sm:gap-8 sm:p-7">
          <img src={AUTHOR.photo} alt="" width={AUTHOR.w} height={AUTHOR.h} className="h-auto w-40 shrink-0 rounded-2xl shadow-lg shadow-slate-300/60 sm:w-56" />
          <div className="min-w-0">
            <p className="text-xs font-black uppercase tracking-[0.14em] text-blue-600">{t('about.roles.author')}</p>
            <h2 className="mt-2 text-[clamp(26px,3.4vw,36px)] font-black leading-tight tracking-tight text-slate-900">{AUTHOR.name}</h2>
            <p lang="zh-Hans" className="mt-1 text-lg font-bold text-slate-500">{AUTHOR.zh}</p>
            <p className="mt-4 max-w-md text-[15px] leading-relaxed text-slate-700">{t('about.author')}</p>
            <a href={`mailto:${CONTACT}`} className="mt-4 inline-flex max-w-full items-center gap-2 rounded-full border border-slate-200 bg-white px-3.5 py-1.5 text-sm font-bold text-slate-700 transition hover:border-slate-300 hover:text-blue-700">
              <Mail size={15} className="shrink-0 text-blue-600" aria-hidden />
              <span className="truncate">{CONTACT}</span>
            </a>
          </div>
        </div>
      </section>

      <div className="mt-4 grid gap-4">
        <Block title={t('about.thanks_title')}>
          <ul className="mt-4 grid gap-6 sm:grid-cols-2">
            {HELPERS.map((p) => (
              <li key={p.key} className="flex items-start gap-4">
                <img src={p.photo} alt="" width={p.w} height={p.h} loading="lazy" className="h-auto w-24 shrink-0 rounded-xl sm:w-28" />
                <div className="min-w-0">
                  <p className="text-[11px] font-black uppercase tracking-wider text-slate-500">{t(`about.roles.${p.key}`)}</p>
                  <h3 className="mt-0.5 text-[16px] font-black leading-snug tracking-tight text-slate-900">
                    {'url' in p ? (
                      <a href={p.url} {...out} className="hover:text-blue-700">
                        {p.name}
                        <ArrowUpRight size={14} className="ml-0.5 inline align-[-1px] text-slate-400" aria-hidden />
                      </a>
                    ) : (
                      p.name
                    )}
                  </h3>
                  {p.zh && <p lang="zh-Hans" className="text-[13px] font-bold text-slate-500">{p.zh}</p>}
                  <p className="mt-1.5 text-sm leading-relaxed text-slate-600">
                    {t(`about.${p.key}`)}
                  </p>
                </div>
              </li>
            ))}
          </ul>
          <p className="mt-5 border-t border-slate-100 pt-4">
            <Trans i18nKey="about.thanks" components={{ b: <b className="font-extrabold text-slate-900" /> }} />
          </p>
        </Block>

        <Block title={t('about.open_title')}>
          <Trans i18nKey="about.open" components={{ a: <a className={link} href={REPO} {...out} /> }} />
          <p className="mt-3">
            <a href={REPO} {...out} className="inline-flex items-center gap-2 rounded-full bg-slate-900 px-4 py-2 text-sm font-bold text-white transition hover:bg-slate-700">
              <Star size={15} aria-hidden /> {t('about.star')}
            </a>
          </p>
        </Block>

        <Block title={t('about.accuracy_title')}>
          <Trans i18nKey="about.accuracy" components={{ a: <a className={link} href={`mailto:${CONTACT}`} /> }} />
        </Block>

        <Block title={t('about.built_title')}>
          <p>{t('about.built_lead')}</p>
          <ul className="mt-3 grid gap-3 sm:grid-cols-3">
            {BUILT.map((b) => (
              <li key={b.key}>
                <a href={b.url} {...out} className="group block h-full rounded-2xl border border-slate-200 px-4 py-3 transition hover:border-slate-300 hover:bg-slate-50">
                  <span className="flex items-center justify-between font-black text-slate-900">
                    {b.name}
                    <ArrowUpRight size={15} className="text-slate-400 transition group-hover:text-blue-600" aria-hidden />
                  </span>
                  <span className="mt-1 block text-[13px] leading-relaxed text-slate-600">{t(`about.built.${b.key}`)}</span>
                </a>
              </li>
            ))}
          </ul>
        </Block>
      </div>
    </main>
  )
}
