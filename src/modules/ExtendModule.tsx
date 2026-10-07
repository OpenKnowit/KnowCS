import { useEffect, useState } from 'react'
import { m } from 'framer-motion'
import { ArrowLeft, Sparkles, Zap } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { EXTENSIONS } from '../data/extensions'
import { normalizeLang } from '../lib/lang'

interface ExtendModuleProps {
  openId: string | null
  onOpen: (id: string | null) => void
}

// Extend 拓展：卡片画廊 + iframe 阅读视图（打开状态由 URL hash 驱动，可分享 / 后退）
export const ExtendModule = ({ openId, onOpen }: ExtendModuleProps) => {
  const { t, i18n } = useTranslation()
  const lang = normalizeLang(i18n.resolvedLanguage)
  const openEntry = EXTENSIONS.find((e) => e.id === openId)
  // the page's HTML loads on demand, per language
  const [doc, setDoc] = useState<{ key: string; html: string } | null>(null)
  const docKey = openEntry ? `${openEntry.id}:${lang}` : ''
  useEffect(() => {
    if (!openEntry) return
    let live = true
    void openEntry.load[lang]().then((mod) => live && setDoc({ key: `${openEntry.id}:${lang}`, html: mod.default }))
    return () => {
      live = false
    }
  }, [openEntry, lang])
  const html = doc && doc.key === docKey ? doc.html : null

  if (openEntry) {
    return (
      <m.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-4 h-full flex flex-col">
        <div className="flex items-center justify-between gap-4">
          <button
            onClick={() => onOpen(null)}
            className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 rounded-xl text-sm font-bold text-slate-600 transition"
          >
            <ArrowLeft size={16} aria-hidden /> {t('extend.back')}
          </button>
          <span className={`px-2.5 py-1 rounded-full text-[10px] font-black uppercase ${openEntry.tagClass}`}>{openEntry.tag}</span>
        </div>
        {html === null ? (
          <div className="w-full flex-1 min-h-[75vh] animate-pulse rounded-2xl bg-[#0d1117]" aria-busy="true" />
        ) : (
        <iframe
          key={lang}
          srcDoc={html}
          title={t(openEntry.titleKey)}
          sandbox="allow-scripts"
          loading="lazy"
          className="w-full flex-1 min-h-[75vh] rounded-2xl border border-slate-200 shadow-inner bg-[#0d1117]"
        />
        )}
      </m.div>
    )
  }

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {EXTENSIONS.map((entry, i) => (
          <m.button
            key={entry.id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            onClick={() => onOpen(entry.id)}
            className="group text-left bg-gradient-to-br from-slate-900 to-slate-800 rounded-[1.5rem] p-6 shadow-lg hover:shadow-2xl hover:-translate-y-1 transition-all duration-300 flex flex-col gap-4 text-white focus:outline-none focus-visible:ring-4 focus-visible:ring-violet-300"
          >
            <div className="flex items-start justify-between">
              <div className="p-3 bg-white/10 group-hover:bg-violet-500 rounded-xl transition-colors" aria-hidden>
                <Zap size={20} />
              </div>
              <span className={`px-2.5 py-1 rounded-full text-[10px] font-black uppercase ${entry.tagClass}`}>{entry.tag}</span>
            </div>
            <div>
              <h4 className="font-black text-sm leading-snug group-hover:text-violet-300 transition-colors">{t(entry.titleKey)}</h4>
              <p className="text-[11px] text-slate-400 mt-2 leading-relaxed">{t(entry.descKey)}</p>
            </div>
          </m.button>
        ))}
      </div>
      <p className="text-xs text-slate-400 flex items-center gap-2">
        <Sparkles size={14} aria-hidden /> {t('extend.hint')}
      </p>
    </div>
  )
}
