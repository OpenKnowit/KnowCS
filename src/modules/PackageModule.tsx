import { useEffect } from 'react'
import { m } from 'framer-motion'
import { ArrowLeft, FileText, Package } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { NOTES } from '../data/notes'
import { normalizeLang } from '../lib/lang'

interface PackageModuleProps {
  openId: string | null
  onOpen: (id: string | null) => void
}

// Package 资料包：3×2 笔记卡片网格 + 阅读视图。
// 笔记在构建期已渲染为 HTML（scripts/vite-plugins.mjs），样式见 index.css 的 .note-prose。
export const PackageModule = ({ openId, onOpen }: PackageModuleProps) => {
  const { t, i18n } = useTranslation()
  const lang = normalizeLang(i18n.resolvedLanguage)
  const openNote = NOTES.find((n) => n.id === openId)

  // 打开笔记时回到页面顶部，避免停留在卡片网格的滚动位置
  useEffect(() => {
    if (openId) window.scrollTo({ top: 0 })
  }, [openId])

  if (openNote) {
    return (
      <m.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
        <div className="flex items-center justify-between gap-4">
          <button
            onClick={() => onOpen(null)}
            className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 rounded-xl text-sm font-bold text-slate-600 transition"
          >
            <ArrowLeft size={16} aria-hidden /> {t('package.back')}
          </button>
          <span className={`px-2.5 py-1 rounded-full text-[10px] font-black uppercase ${openNote.tagClass}`}>{openNote.tag}</span>
        </div>
        <article key={lang} lang={lang} className="note-prose max-w-3xl" dangerouslySetInnerHTML={{ __html: openNote.body[lang].html }} />
      </m.div>
    )
  }

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {NOTES.map((note, i) => (
          <m.button
            key={note.id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            onClick={() => onOpen(note.id)}
            className="group text-left bg-white border border-slate-200 rounded-[1.5rem] p-6 shadow-sm hover:shadow-xl hover:-translate-y-1 hover:border-blue-200 transition-all duration-300 flex flex-col gap-4 focus:outline-none focus-visible:ring-4 focus-visible:ring-blue-200"
          >
            <div className="flex items-start justify-between">
              <div className="p-3 bg-slate-100 group-hover:bg-blue-600 group-hover:text-white text-slate-500 rounded-xl transition-colors" aria-hidden>
                <FileText size={20} />
              </div>
              <span className={`px-2.5 py-1 rounded-full text-[10px] font-black uppercase ${note.tagClass}`}>{note.tag}</span>
            </div>
            <div>
              <h4 className="font-black text-slate-800 text-sm leading-snug group-hover:text-blue-600 transition-colors">{t(note.titleKey)}</h4>
              <p className="text-[11px] text-slate-400 mt-2 font-medium">
                {Math.round(note.body[lang].chars / 100) / 10}k {t('package.chars')} · Markdown
              </p>
            </div>
          </m.button>
        ))}
      </div>
      <p className="text-xs text-slate-400 flex items-center gap-2">
        <Package size={14} aria-hidden /> {t('package.hint')}
      </p>
    </div>
  )
}
