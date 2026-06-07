import { useState } from 'react'
import { motion } from 'framer-motion'
import { ArrowLeft, Sparkles, Zap } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import attentionHtml from '../content/attention.html?raw'

interface ExtendEntry {
  id: string
  titleKey: string // i18n 键（extend.items.<id>.title / .desc）
  descKey: string
  tag: string
  tagClass: string
  html: string
}

const EXTENSIONS: ExtendEntry[] = [
  {
    id: 'attention',
    titleKey: 'extend.items.attention.title',
    descKey: 'extend.items.attention.desc',
    tag: 'Transformer',
    tagClass: 'bg-violet-100 text-violet-700',
    html: attentionHtml,
  },
]

// Extend 拓展：课外延伸内容（自包含 HTML 经 iframe 隔离渲染）
export const ExtendModule = () => {
  const [openId, setOpenId] = useState<string | null>(null)
  const { t } = useTranslation()
  const openEntry = EXTENSIONS.find((e) => e.id === openId)

  if (openEntry) {
    return (
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-4 h-full flex flex-col">
        <div className="flex items-center justify-between gap-4">
          <button
            onClick={() => setOpenId(null)}
            className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 rounded-xl text-sm font-bold text-slate-600 transition"
          >
            <ArrowLeft size={16} /> {t('extend.back')}
          </button>
          <span className={`px-2.5 py-1 rounded-full text-[10px] font-black uppercase ${openEntry.tagClass}`}>{openEntry.tag}</span>
        </div>
        <iframe
          srcDoc={openEntry.html}
          title={t(openEntry.titleKey)}
          sandbox="allow-scripts"
          className="w-full flex-1 min-h-[75vh] rounded-2xl border border-slate-200 shadow-inner bg-[#0d1117]"
        />
      </motion.div>
    )
  }

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {EXTENSIONS.map((entry, i) => (
          <motion.button
            key={entry.id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            onClick={() => setOpenId(entry.id)}
            className="group text-left bg-gradient-to-br from-slate-900 to-slate-800 rounded-[1.5rem] p-6 shadow-lg hover:shadow-2xl hover:-translate-y-1 transition-all duration-300 flex flex-col gap-4 text-white"
          >
            <div className="flex items-start justify-between">
              <div className="p-3 bg-white/10 group-hover:bg-violet-500 rounded-xl transition-colors">
                <Zap size={20} />
              </div>
              <span className={`px-2.5 py-1 rounded-full text-[10px] font-black uppercase ${entry.tagClass}`}>{entry.tag}</span>
            </div>
            <div>
              <h4 className="font-black text-sm leading-snug group-hover:text-violet-300 transition-colors">{t(entry.titleKey)}</h4>
              <p className="text-[11px] text-slate-400 mt-2 leading-relaxed">{t(entry.descKey)}</p>
            </div>
          </motion.button>
        ))}
      </div>
      <p className="text-xs text-slate-400 flex items-center gap-2">
        <Sparkles size={14} /> {t('extend.hint')}
      </p>
    </div>
  )
}
