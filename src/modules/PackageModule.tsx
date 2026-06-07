import { useState } from 'react'
import { motion } from 'framer-motion'
import { ArrowLeft, FileText, Package } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Components } from 'react-markdown'
import { NOTE_IMAGES, NOTES } from '../data/notes'

// Markdown 渲染样式映射（Tailwind 工具类）
const mdComponents: Components = {
  h1: (props) => <h1 className="text-2xl font-black text-slate-900 mt-8 mb-4 first:mt-0" {...props} />,
  h2: (props) => <h2 className="text-xl font-black text-slate-800 mt-8 mb-3 pb-2 border-b border-slate-200" {...props} />,
  h3: (props) => <h3 className="text-base font-bold text-slate-800 mt-6 mb-2" {...props} />,
  h4: (props) => <h4 className="text-sm font-bold text-slate-700 mt-4 mb-2" {...props} />,
  p: (props) => <p className="text-sm text-slate-600 leading-relaxed my-3" {...props} />,
  ul: (props) => <ul className="list-disc pl-6 my-3 space-y-1.5 text-sm text-slate-600" {...props} />,
  ol: (props) => <ol className="list-decimal pl-6 my-3 space-y-1.5 text-sm text-slate-600" {...props} />,
  li: (props) => <li className="leading-relaxed" {...props} />,
  blockquote: (props) => (
    <blockquote className="border-l-4 border-blue-200 bg-blue-50/50 pl-4 py-2 my-4 text-sm text-slate-500 rounded-r-lg" {...props} />
  ),
  table: (props) => (
    <div className="overflow-x-auto my-4 rounded-xl border border-slate-200">
      <table className="w-full text-xs" {...props} />
    </div>
  ),
  thead: (props) => <thead className="bg-slate-50" {...props} />,
  th: (props) => <th className="px-3 py-2 text-left font-bold text-slate-700 border-b border-slate-200 whitespace-nowrap" {...props} />,
  td: (props) => <td className="px-3 py-2 text-slate-600 border-b border-slate-100 align-top" {...props} />,
  code: ({ className, children, ...props }) => {
    const isBlock = /language-/.test(className ?? '')
    return isBlock ? (
      <code className={`${className} block`} {...props}>{children}</code>
    ) : (
      <code className="px-1.5 py-0.5 bg-slate-100 text-rose-600 rounded text-[0.85em] font-mono" {...props}>{children}</code>
    )
  },
  pre: (props) => (
    <pre className="bg-slate-900 text-slate-100 rounded-xl p-4 my-4 overflow-x-auto text-xs leading-relaxed font-mono" {...props} />
  ),
  a: (props) => <a className="text-blue-600 font-medium hover:underline" target="_blank" rel="noreferrer" {...props} />,
  hr: () => <hr className="my-6 border-slate-200" />,
  strong: (props) => <strong className="font-bold text-slate-800" {...props} />,
  img: ({ src, alt }) => (
    <img
      src={typeof src === 'string' ? NOTE_IMAGES[src] ?? src : undefined}
      alt={alt ?? ''}
      className="max-w-full rounded-xl border border-slate-200 shadow-sm my-4"
      loading="lazy"
    />
  ),
}

// Package 资料包：3×2 笔记卡片网格 + Markdown 阅读视图
export const PackageModule = () => {
  const [openId, setOpenId] = useState<string | null>(null)
  const { t } = useTranslation()
  const openNote = NOTES.find((n) => n.id === openId)

  if (openNote) {
    return (
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
        <div className="flex items-center justify-between gap-4">
          <button
            onClick={() => setOpenId(null)}
            className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 rounded-xl text-sm font-bold text-slate-600 transition"
          >
            <ArrowLeft size={16} /> {t('package.back')}
          </button>
          <span className={`px-2.5 py-1 rounded-full text-[10px] font-black uppercase ${openNote.tagClass}`}>{openNote.tag}</span>
        </div>
        <article className="max-w-3xl">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
            {openNote.content}
          </ReactMarkdown>
        </article>
      </motion.div>
    )
  }

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {NOTES.map((note, i) => (
          <motion.button
            key={note.id}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            onClick={() => setOpenId(note.id)}
            className="group text-left bg-white border border-slate-200 rounded-[1.5rem] p-6 shadow-sm hover:shadow-xl hover:-translate-y-1 hover:border-blue-200 transition-all duration-300 flex flex-col gap-4"
          >
            <div className="flex items-start justify-between">
              <div className="p-3 bg-slate-100 group-hover:bg-blue-600 group-hover:text-white text-slate-500 rounded-xl transition-colors">
                <FileText size={20} />
              </div>
              <span className={`px-2.5 py-1 rounded-full text-[10px] font-black uppercase ${note.tagClass}`}>{note.tag}</span>
            </div>
            <div>
              <h4 className="font-black text-slate-800 text-sm leading-snug group-hover:text-blue-600 transition-colors">{t(note.titleKey)}</h4>
              <p className="text-[11px] text-slate-400 mt-2 font-medium">
                {Math.round(note.content.length / 100) / 10}k {t('package.chars')} · Markdown
              </p>
            </div>
          </motion.button>
        ))}
      </div>
      <p className="text-xs text-slate-400 flex items-center gap-2">
        <Package size={14} /> {t('package.hint')}
      </p>
    </div>
  )
}
