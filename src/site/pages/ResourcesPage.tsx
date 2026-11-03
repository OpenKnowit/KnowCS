import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ArrowUpRight, BookOpen, Globe, History, Lock, Pencil, Search } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { fetchResources, inLang } from '../../lib/resources'
import type { ResourceCat, ResourceItem, ResourceKind } from '../../lib/resources'
import { previewSrc } from '../resourcePreviews'

// /resources/: outside sites worth keeping open next to the course. Titles and blurbs are in site.resources.items.<id>.

// The list itself is edited on /admin/ and served by /api/resources; RESOURCES below is the built-in copy shown until
// (or if) the API answers.
type Cat = ResourceCat
type Kind = ResourceKind
interface Resource {
  id: string
  url: string
  cat: Cat
  kind: Kind
  /** a screenshot in src/assets/resources/<id>.webp; sites behind a login have none */
  preview?: boolean
}

const RESOURCES: Resource[] = [
  { id: 'comp2211', url: 'https://course.cse.ust.hk/comp2211/', cat: 'course', kind: 'login' },
  { id: 'comp1023', url: 'https://course.cse.ust.hk/comp2211/comp1023-videos.html', cat: 'course', kind: 'login' },
  { id: 'jupyterlite', url: 'https://jupyter.org/try-jupyter/lab/', cat: 'code', kind: 'web' },
  { id: 'colab', url: 'https://colab.research.google.com/', cat: 'code', kind: 'web', preview: true },
  { id: 'visualgo', url: 'https://visualgo.net/en', cat: 'visualize', kind: 'web', preview: true },
  { id: 'programiz', url: 'https://www.programiz.com/python-programming/online-compiler/', cat: 'code', kind: 'web', preview: true },
  { id: 'tf-playground', url: 'https://playground.tensorflow.org/', cat: 'visualize', kind: 'web', preview: true },
  { id: 'knowcs-old', url: 'https://knowcs.pinit.eth.link/', cat: 'archive', kind: 'archive', preview: true },
  { id: 'list-copy', url: 'https://www.likeimingbai.dev/python-list-copy-demo/index.html', cat: 'visualize', kind: 'web', preview: true },
  { id: 'comp1945', url: 'https://course.cse.ust.hk/comp1945/demos/', cat: 'course', kind: 'login' },
  { id: 'csdiy', url: 'https://csdiy.wiki/', cat: 'study', kind: 'web', preview: true },
  { id: 'cs231n', url: 'https://cs231n.stanford.edu/', cat: 'study', kind: 'web', preview: true },
  { id: 'comp2211-notes', url: 'https://github.com/moyunxiang/COMP2211', cat: 'notes', kind: 'notes', preview: true },
]

const CAT_COLOR: Record<Cat, string> = { course: 'bg-indigo-600', visualize: 'bg-blue-600', code: 'bg-emerald-600', study: 'bg-rose-600', notes: 'bg-amber-600', archive: 'bg-slate-500' }
const KIND_ICON: Record<Kind, LucideIcon> = { web: Globe, login: Lock, notes: BookOpen, archive: History }
const FILTERS: ('all' | Cat)[] = ['all', 'course', 'visualize', 'code', 'study', 'notes']

const host = (url: string) => {
  const u = new URL(url)
  const path = u.pathname.replace(/\/(index\.html)?$/, '')
  const full = u.hostname.replace(/^www\./, '') + path
  return full.length > 32 ? `${full.slice(0, 31)}…` : full
}

/** the built-in list in the shape the API serves */
const builtIn = (t: (k: string) => string): ResourceItem[] =>
  RESOURCES.map((r) => {
    const title = t(`site.resources.items.${r.id}.title`)
    const desc = t(`site.resources.items.${r.id}.desc`)
    return { id: r.id, url: r.url, cat: r.cat, kind: r.kind, title: { en: title, zh: title, 'zh-HK': title }, desc: { en: desc, zh: desc, 'zh-HK': desc }, preview: r.preview ? 'builtin' : null }
  })

function ResourceCard({ r }: { r: ResourceItem }) {
  const { t, i18n } = useTranslation()
  const Icon = KIND_ICON[r.kind] ?? Globe
  const preview = previewSrc(r)
  return (
    <a href={r.url} target="_blank" rel="noopener noreferrer" className="group mb-6 block break-inside-avoid overflow-hidden rounded-[14px] border border-slate-200 bg-white transition hover:-translate-y-0.5 hover:border-slate-300 hover:shadow-lg hover:shadow-slate-200">
      <div className={`h-1 ${CAT_COLOR[r.cat]}`} aria-hidden />
      <div className="flex flex-col gap-3.5 px-[22px] pb-5 pt-[18px]">
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span className="font-semibold">{t(`site.resources.cats.${r.cat}`)}</span>
          <span className="flex items-center gap-1.5 font-medium">
            <Icon size={13} aria-hidden /> {t(`site.resources.kinds.${r.kind}`)}
          </span>
        </div>
        {preview && <img src={preview} alt="" loading="lazy" decoding="async" className="aspect-[8/5] w-full rounded-lg border border-slate-200 bg-slate-100 object-cover object-top" />}
        <div>
          <h2 className="text-[19px] font-bold text-slate-900 group-hover:text-blue-700">{inLang(r.title, i18n.language)}</h2>
          {inLang(r.desc, i18n.language) && <p className="mt-1.5 text-sm leading-relaxed text-slate-600">{inLang(r.desc, i18n.language)}</p>}
        </div>
        <div className="flex items-center justify-between gap-3 border-t border-slate-200 pt-3.5">
          <span className="min-w-0 truncate font-mono text-xs text-slate-500">{host(r.url)}</span>
          <span className="flex shrink-0 items-center gap-1 text-sm font-semibold text-blue-600">
            {t('site.resources.visit')} <ArrowUpRight size={15} aria-hidden />
          </span>
        </div>
      </div>
    </a>
  )
}

export default function ResourcesPage() {
  const { t } = useTranslation()
  const [q, setQ] = useState('')
  const [cat, setCat] = useState<'all' | Cat>('all')
  const { i18n } = useTranslation()
  // the edited list from the admin API; null until it answers (or when there is none)
  const [live, setLive] = useState<ResourceItem[] | null>(null)
  useEffect(() => {
    document.title = `${t('site.resources.tab_title')} · KnowCS`
  }, [t])
  useEffect(() => {
    const ctl = new AbortController()
    void fetchResources(ctl.signal).then((d) => d && setLive(d.resources))
    return () => ctl.abort()
  }, [])
  const items = live ?? builtIn(t)
  const query = q.trim().toLowerCase()
  const lang = i18n.language
  const shown = items.filter((r) => (cat === 'all' || r.cat === cat) && (!query || `${inLang(r.title, lang)} ${inLang(r.desc, lang)} ${t(`site.resources.cats.${r.cat}`)} ${r.url}`.toLowerCase().includes(query)))
  return (
    <main className="px-[clamp(1rem,9vw,13rem)] pb-16 pt-6 lg:pt-8">
      <h1 className="sr-only">{t('site.resources.title')}</h1>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <label className="flex h-[42px] w-full max-w-[420px] items-center gap-2.5 rounded-[10px] border border-slate-200 bg-white px-3.5 focus-within:border-blue-400 focus-within:ring-2 focus-within:ring-blue-100">
          <Search size={17} className="shrink-0 text-slate-500" aria-hidden />
          <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder={t('site.resources.search')} aria-label={t('site.resources.search')} className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-slate-500" />
        </label>
        <div className="flex flex-wrap gap-2" role="group" aria-label={t('site.resources.categories')}>
          {FILTERS.map((c) => (
            <button
              key={c}
              type="button"
              onClick={() => setCat(c)}
              aria-pressed={cat === c}
              className={`rounded-full border px-3 py-1.5 text-[13px] font-semibold transition ${cat === c ? 'border-slate-900 bg-slate-900 text-white' : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300'}`}
            >
              {t(`site.resources.cats.${c}`)}
            </button>
          ))}
        </div>
      </div>
      {shown.length ? (
        <div className="mt-6 gap-6 sm:columns-2 lg:columns-3">
          {shown.map((r) => (
            <ResourceCard key={r.id} r={r} />
          ))}
        </div>
      ) : (
        <p className="mt-12 text-center text-slate-500">{t('site.resources.empty')}</p>
      )}
      {/* a quiet way in for whoever maintains the list */}
      <div className="mt-10 flex justify-end">
        <a href="/admin/" title={t('site.resources.edit')} aria-label={t('site.resources.edit')} className="grid h-7 w-7 place-items-center rounded-md text-slate-300 transition hover:bg-slate-200/60 hover:text-slate-500">
          <Pencil size={13} aria-hidden />
        </a>
      </div>
    </main>
  )
}
