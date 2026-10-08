import { useEffect, useMemo, useRef, useState } from 'react'
import type { ComponentType, KeyboardEvent, MouseEvent } from 'react'
import { m } from 'framer-motion'
import { ArrowLeft, FileText, MousePointerClick, Package } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { NOTES } from '../data/notes'
import { NOTE_LINKS, blockEntry } from '../data/numpyApis'
import type { ApiCat } from '../data/numpyApis'
import { normalizeLang } from '../lib/lang'
import { NumpyApiPanel } from './NumpyApiPanel'
import { PLAYGROUND_LOADERS, loadPlaygroundView } from '../site/playground/loaders'
import { decorateBlocks, pythonBlocks, runnableBlocks, withPrelude } from '../site/playground/noteBlocks'
import type { PlayConfig } from '../data/playgrounds/types'

type PlaygroundComp = ComponentType<{ config: PlayConfig; initialCode?: string | null }>

/** Placeholder while a note's HTML chunk downloads. */
const NoteSkeleton = () => (
  <div className="max-w-3xl space-y-3" aria-busy="true">
    {[70, 95, 88, 92, 60, 85].map((w, i) => (
      <div key={i} className="h-4 animate-pulse rounded bg-slate-100" style={{ width: `${w}%` }} />
    ))}
  </div>
)

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
  const withPanel = openNote?.id === 'numpy'
  const playLoader = openNote ? PLAYGROUND_LOADERS[openNote.id] : undefined
  const panelRef = useRef<HTMLDivElement>(null)
  // 其他库的笔记：右侧是该库的实验台（配置与组件懒加载）
  const [play, setPlay] = useState<{ id: string; config: PlayConfig; View: PlaygroundComp } | null>(null)
  useEffect(() => {
    if (!openNote || !playLoader) return
    let live = true
    void Promise.all([loadPlaygroundView(), playLoader()]).then(([m, config]) => {
      if (live) setPlay({ id: openNote.id, config, View: m.default as PlaygroundComp })
    })
    return () => {
      live = false
    }
  }, [openNote, playLoader])
  const playReady = play && play.id === openNote?.id ? play : null
  // 笔记代码块中能在实验台运行的那些；点「试一试」把代码送进实验台（n 递增让实验台按新代码重置）
  const [runnable, setRunnable] = useState<{ key: string; ok: boolean[] } | null>(null)
  const [tryCode, setTryCode] = useState<{ code: string | null; n: number }>({ code: null, n: 0 })
  // 笔记里点了哪个名字：决定 NumPy 面板打开哪一组（key 递增让面板按新入口重置）
  const [focus, setFocus] = useState<{ cat: ApiCat | null; entry: string | null; missing: string | null; n: number }>({ cat: null, entry: null, missing: null, n: 0 })

  // 正文懒加载：打开的笔记 × 当前语言
  const [body, setBody] = useState<{ key: string; html: string } | null>(null)
  const bodyKey = openNote ? `${openNote.id}:${lang}` : ''
  useEffect(() => {
    if (!openNote) return
    let live = true
    void openNote.load[lang]().then((m) => live && setBody({ key: `${openNote.id}:${lang}`, html: m.default.html }))
    return () => {
      live = false
    }
  }, [openNote, lang])
  const bodyHtml = body && body.key === bodyKey ? body.html : null
  const blocks = useMemo(() => (playLoader && bodyHtml ? pythonBlocks(bodyHtml) : []), [playLoader, bodyHtml])
  useEffect(() => {
    if (!playReady || !blocks.length) return
    let live = true
    void runnableBlocks(playReady.config, blocks, () => live).then((ok) => live && setRunnable({ key: bodyKey, ok }))
    return () => {
      live = false
    }
  }, [playReady, blocks, bodyKey])

  // 打开笔记时回到页面顶部，避免停留在卡片网格的滚动位置
  useEffect(() => {
    if (openId) window.scrollTo({ top: 0 })
  }, [openId])

  // NumPy 笔记：把能在面板里演示的行内 `code` 名字标成可点击（直接改 HTML 字符串，重渲染 / 切换语言都不会丢）
  const noteHtml = useMemo(() => {
    if (!openNote || bodyHtml === null) return ''
    const html = bodyHtml
    if (playLoader) return runnable && runnable.key === bodyKey ? decorateBlocks(html, runnable.ok, t('playground.ui.try_block'), t('playground.ui.try_hint')) : html
    if (!withPanel) return html
    const title = t('numpy_api.link_hint').replace(/"/g, '&quot;')
    const run = t('numpy_api.run_block')
    // 行内 code 没有属性；代码块里的是 <code class="language-…">，不会被匹配
    return html
      .replace(/<code>([^<]+)<\/code>/g, (whole, name: string) =>
        name in NOTE_LINKS ? `<code class="np-link" role="button" tabindex="0" title="${title}">${name}</code>` : whole,
      )
      .replace(/<pre>[\s\S]*?<\/pre>/g, (block) => {
        const id = blockEntry(block)
        return id ? `<div class="np-block">${block}<button type="button" class="np-run" data-entry="${id}">▶ ${run}</button></div>` : block
      })
  }, [openNote, bodyHtml, withPanel, playLoader, runnable, bodyKey, t])

  const followLink = (target: EventTarget) => {
    const runBtn = (target as HTMLElement).closest?.('button.np-run') as HTMLElement | null
    const el = (target as HTMLElement).closest?.('code.np-link')
    if (runBtn && runBtn.dataset.block !== undefined && playReady) {
      const code = blocks[Number(runBtn.dataset.block)]
      setTryCode((f) => ({ code: withPrelude(playReady.config, code), n: f.n + 1 }))
    } else if (runBtn) {
      setFocus((f) => ({ cat: null, entry: runBtn.dataset.entry ?? null, missing: null, n: f.n + 1 }))
    } else if (el) {
      const name = el.textContent?.trim() ?? ''
      const cat = NOTE_LINKS[name] ?? null
      setFocus((f) => ({ cat, entry: null, missing: cat ? null : name, n: f.n + 1 }))
    } else return
    // 窄屏时面板在正文下方：滚过去
    if (window.matchMedia('(max-width: 1279px)').matches) panelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
  const onArticleClick = (e: MouseEvent) => followLink(e.target)
  const onArticleKey = (e: KeyboardEvent) => {
    // only the marked <code> names need keyboard handling; the run buttons are real <button>s
    if ((e.key === 'Enter' || e.key === ' ') && (e.target as HTMLElement).closest?.('code.np-link')) {
      e.preventDefault()
      followLink(e.target)
    }
  }

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
        {playLoader ? (
          <div className="grid grid-cols-[minmax(0,1fr)] items-start gap-8 xl:grid-cols-[minmax(0,1fr)_minmax(0,600px)]">
            <div className="min-w-0">
              {noteHtml ? (
                <article key={lang} lang={lang} className="note-prose" onClick={onArticleClick} dangerouslySetInnerHTML={{ __html: noteHtml }} />
              ) : (
                <NoteSkeleton />
              )}
            </div>
            <div ref={panelRef} className="scroll-mt-6 xl:sticky xl:top-6 xl:max-h-[calc(100vh-3rem)] xl:overflow-y-auto xl:pb-2">
              {playReady ? (
                <playReady.View key={tryCode.n} config={playReady.config} initialCode={tryCode.code} />
              ) : (
                <div className="rounded-[1.5rem] border border-slate-200 bg-white p-5 text-sm text-slate-400" aria-busy="true">{t('playground.ui.loading', { defaultValue: '…' })}</div>
              )}
            </div>
          </div>
        ) : withPanel ? (
          <div className="grid grid-cols-[minmax(0,1fr)] items-start gap-8 xl:grid-cols-[minmax(0,1fr)_minmax(0,560px)]">
            <div className="min-w-0 space-y-4">
              <p className="flex items-center gap-2 rounded-xl bg-blue-50 px-3 py-2 text-xs font-medium text-blue-900">
                <MousePointerClick size={14} aria-hidden /> {t('numpy_api.link_hint')}
              </p>
              {noteHtml ? (
                <article key={lang} lang={lang} className="note-prose" onClick={onArticleClick} onKeyDown={onArticleKey} dangerouslySetInnerHTML={{ __html: noteHtml }} />
              ) : (
                <NoteSkeleton />
              )}
            </div>
            <div ref={panelRef} className="scroll-mt-6 xl:sticky xl:top-6 xl:max-h-[calc(100vh-3rem)] xl:overflow-y-auto xl:pb-2">
              <NumpyApiPanel key={focus.n} initialCat={focus.cat} initialEntry={focus.entry} missing={focus.missing} />
            </div>
          </div>
        ) : (
          noteHtml ? <article key={lang} lang={lang} className="note-prose max-w-3xl" dangerouslySetInnerHTML={{ __html: noteHtml }} /> : <NoteSkeleton />
        )}
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
                {Math.round(note.chars[lang] / 100) / 10}k {t('package.chars')} · Markdown
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
