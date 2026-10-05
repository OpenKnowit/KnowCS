import { useEffect, useRef, useState } from 'react'
import type { ComponentType, KeyboardEvent, ReactNode } from 'react'
import { AnimatePresence, LazyMotion, MotionConfig, domAnimation, m } from 'framer-motion'
import {
  BarChart3,
  BookOpen,
  Boxes,
  Calculator,
  Check,
  ChevronDown,
  Cpu,
  GitBranch,
  Grid3X3,
  Info,
  Languages,
  Layers,
  MousePointer2,
  Sparkles,
  Sigma,
  WifiOff,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from './components/Latex'
import { SectionTitle } from './components/SectionTitle'
import { NOTES } from './data/notes'
import { useHashRoute } from './hooks/useHashRoute'
import { normalizeLang, SUPPORTED_LANGS } from './lib/lang'
import type { LangCode } from './lib/lang'
import type { AppMode, RouteTable } from './lib/route'
import { AlphaBetaModule } from './modules/AlphaBetaModule'
import { BackpropModule } from './modules/BackpropModule'
import { BayesBasicsModule } from './modules/BayesBasicsModule'
import { EXTENSIONS } from './data/extensions'
import { ExtendModule } from './modules/ExtendModule'
import { KernelModule } from './modules/KernelModule'
import { KMeansModule } from './modules/KMeansModule'
import { KnnModule } from './modules/KnnModule'
import { NaiveBayesModule } from './modules/NaiveBayesModule'
import { NumpyModule } from './modules/NumpyModule'
import { PackageModule } from './modules/PackageModule'
import type { TabId } from './types'

const LANG_LABELS: Record<LangCode, string> = {
  en: 'English',
  zh: '简体中文',
  'zh-HK': '繁體中文',
}

// --- 课程模块注册表：新增模块只需在此追加一项（+ i18n 文案） ---
interface CourseTab {
  id: TabId
  icon: LucideIcon
  Component: ComponentType
  tip: ReactNode // 侧栏 Exam Tip（i18n key + 内嵌公式）
}

const COURSE_TABS: CourseTab[] = [
  {
    id: 'numpy',
    icon: Cpu,
    Component: NumpyModule,
    tip: (
      <Trans
        i18nKey="app.sidebar.exam_tip.content_numpy"
        components={{
          1: <Latex formula="\text{Broadcasting}" />,
          3: <Latex formula="\mathbf{A} \cdot \mathbf{B}" />,
          5: <Latex formula="\mathbf{A} \odot \mathbf{B}" />,
        }}
      />
    ),
  },
  {
    id: 'backprop',
    icon: Layers,
    Component: BackpropModule,
    tip: (
      <Trans
        i18nKey="app.sidebar.exam_tip.content_backprop"
        components={{
          1: <Latex formula="\delta_k" />,
          3: <Latex formula="\Delta w = \eta \cdot \delta \cdot O" />,
        }}
      />
    ),
  },
  {
    id: 'kernel',
    icon: Grid3X3,
    Component: KernelModule,
    tip: <Trans i18nKey="app.sidebar.exam_tip.content_kernel" components={{ 1: <Latex formula="N - K + 1" /> }} />,
  },
  {
    id: 'bayesBasics',
    icon: Calculator,
    Component: BayesBasicsModule,
    tip: (
      <Trans
        i18nKey="app.sidebar.exam_tip.content_bayesBasics"
        components={{ 1: <Latex formula="P(B|E) \propto P(B) \cdot P(E|B)" /> }}
      />
    ),
  },
  {
    id: 'naiveBayes',
    icon: BarChart3,
    Component: NaiveBayesModule,
    tip: (
      <Trans
        i18nKey="app.sidebar.exam_tip.content_naiveBayes"
        components={{ 1: <Latex formula="\alpha" />, 3: <Latex formula="\log" /> }}
      />
    ),
  },
  {
    id: 'knn',
    icon: MousePointer2,
    Component: KnnModule,
    tip: <Trans i18nKey="app.sidebar.exam_tip.content_knn" components={{ 1: <Latex formula="K" /> }} />,
  },
  {
    id: 'alphabeta',
    icon: GitBranch,
    Component: AlphaBetaModule,
    tip: (
      <Trans
        i18nKey="app.sidebar.exam_tip.content_alphabeta"
        components={{ 1: <Latex formula="\beta \leq \alpha" />, 3: <strong className="font-bold" /> }}
      />
    ),
  },
  {
    id: 'kmeans',
    icon: Boxes,
    Component: KMeansModule,
    tip: (
      <Trans
        i18nKey="app.sidebar.exam_tip.content_kmeans"
        components={{ 1: <Latex formula="\text{WCSS}" />, 3: <Latex formula="K" /> }}
      />
    ),
  },
]

const MODES: AppMode[] = ['course', 'package', 'extend']

const ROUTES: RouteTable = {
  course: COURSE_TABS.map((tab) => tab.id),
  package: NOTES.map((n) => n.id),
  extend: EXTENSIONS.map((e) => e.id),
}

const LanguageMenu = () => {
  const { t, i18n } = useTranslation()
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const currentLang = normalizeLang(i18n.resolvedLanguage ?? i18n.language)

  useEffect(() => {
    if (!open) return
    const handleClickOutside = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [open])

  const select = (code: LangCode) => {
    i18n.changeLanguage(code)
    setOpen(false)
    buttonRef.current?.focus()
  }

  const onMenuKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Escape') {
      setOpen(false)
      buttonRef.current?.focus()
      return
    }
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return
    e.preventDefault()
    const items = [...(e.currentTarget.querySelectorAll<HTMLButtonElement>('[role="menuitemradio"]'))]
    const idx = items.indexOf(document.activeElement as HTMLButtonElement)
    const next = (idx + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length
    items[next]?.focus()
  }

  return (
    <div className="relative" ref={rootRef}>
      <button
        ref={buttonRef}
        onClick={() => setOpen((o) => !o)}
        className="bg-white rounded-full shadow-sm hover:bg-gray-50 transition flex items-center gap-2 px-4 py-2 font-bold text-slate-600"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={t('app.a11y.language')}
      >
        <Languages size={20} className="text-blue-600" aria-hidden />
        <span>{LANG_LABELS[currentLang]}</span>
        <ChevronDown size={14} className={`text-slate-400 transition-transform ${open ? 'rotate-180' : ''}`} aria-hidden />
      </button>
      <AnimatePresence>
        {open && (
          <m.div
            role="menu"
            aria-label={t('app.a11y.language')}
            onKeyDown={onMenuKeyDown}
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 mt-2 w-40 bg-white border border-slate-200 rounded-2xl shadow-xl shadow-slate-200/50 overflow-hidden z-50"
          >
            {SUPPORTED_LANGS.map((code) => (
              <button
                key={code}
                role="menuitemradio"
                aria-checked={currentLang === code}
                lang={code}
                autoFocus={currentLang === code}
                onClick={() => select(code)}
                className={`w-full px-4 py-2.5 text-left text-sm font-bold transition flex items-center justify-between focus:outline-none focus-visible:bg-slate-100
                  ${currentLang === code ? 'bg-blue-50 text-blue-600' : 'text-slate-600 hover:bg-slate-50'}`}
              >
                {LANG_LABELS[code]}
                {currentLang === code && <Check size={14} aria-hidden />}
              </button>
            ))}
          </m.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function App() {
  const { t } = useTranslation()
  const [route, navigate] = useHashRoute(ROUTES)
  // 记住最近一次打开的课程模块：从 Package / Extend 切回 Course 时恢复
  const [lastTab, setLastTab] = useState<TabId>(COURSE_TABS[0].id)
  const activeTab = route.mode === 'course' ? (route.tab as TabId) : lastTab
  if (route.mode === 'course' && route.tab !== lastTab) setLastTab(route.tab as TabId)

  const active = COURSE_TABS.find((tab) => tab.id === activeTab) ?? COURSE_TABS[0]
  const ActiveModule = active.Component

  // 移动端模块条是横向滚动的：把当前模块滚到可视区中央（只改 scrollLeft，不影响页面纵向滚动）
  const tabStripRef = useRef<HTMLUListElement>(null)
  useEffect(() => {
    const strip = tabStripRef.current
    if (!strip || strip.scrollWidth <= strip.clientWidth) return
    const item = strip.querySelector<HTMLElement>('[aria-current="page"]')?.parentElement
    if (item) strip.scrollTo({ left: item.offsetLeft - (strip.clientWidth - item.offsetWidth) / 2, behavior: 'smooth' })
  }, [activeTab, route.mode])

  const goMode = (mode: AppMode) =>
    navigate(mode === 'course' ? { mode, tab: lastTab } : { mode, item: null })

  const examTip = (
    <div className="p-6 bg-gradient-to-br from-blue-600 to-indigo-700 rounded-[1.5rem] text-white shadow-lg shadow-blue-200 relative overflow-hidden group">
      <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:scale-125 transition-transform duration-500" aria-hidden>
        <Cpu size={80} />
      </div>
      <div className="relative z-10">
        <div className="flex items-center gap-2 mb-3">
          <Info size={16} aria-hidden />
          <span className="text-xs font-bold uppercase tracking-wider">{t('app.sidebar.exam_tip.title')}</span>
        </div>
        <p className="text-[11px] leading-relaxed opacity-90 font-medium">{active.tip}</p>
      </div>
    </div>
  )

  return (
    <LazyMotion features={domAnimation} strict>
      <MotionConfig reducedMotion="user">
        <div className="min-h-screen bg-[#f1f5f9] text-slate-900 font-sans p-4 md:p-8">
          <a
            href="#main-content"
            onClick={(e) => {
              e.preventDefault()
              document.getElementById('main-content')?.focus()
            }}
            className="sr-only focus:not-sr-only focus:fixed focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-blue-600 focus:text-white focus:rounded-lg focus:font-bold"
          >
            {t('app.a11y.skip')}
          </a>

          {/* Header */}
          <header className="max-w-6xl mx-auto mb-8 flex flex-col md:flex-row md:items-end justify-between gap-4">
            <div>
              <h1 className="text-3xl md:text-4xl font-black text-slate-900 tracking-tight">{t('app.title')}</h1>
              <div className="flex items-center gap-3 mt-2">
                <p className="text-slate-500 font-medium">{t('app.subtitle')}</p>
                <span className="px-2 py-0.5 bg-indigo-100 text-indigo-600 text-[10px] rounded-full uppercase font-black">v{__APP_VERSION__}</span>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-3 md:gap-4">
              <nav aria-label={t('app.a11y.mode_nav')} className="flex items-center bg-white border border-slate-200 rounded-full shadow-sm p-1">
                {MODES.map((mode) => (
                  <button
                    key={mode}
                    onClick={() => goMode(mode)}
                    aria-current={route.mode === mode ? 'page' : undefined}
                    className={`px-4 py-1.5 rounded-full text-sm font-bold transition-all flex items-center gap-2
                      ${route.mode === mode ? 'bg-blue-600 text-white shadow-md shadow-blue-200' : 'text-slate-500 hover:text-slate-800'}`}
                  >
                    {route.mode === mode && <span className="h-1.5 w-1.5 rounded-full bg-blue-200" aria-hidden />}
                    {t(`app.nav.${mode}`)}
                  </button>
                ))}
              </nav>
              <LanguageMenu />
            </div>
          </header>

          {/* Package / Extend 视图（全宽，无课程侧栏） */}
          {route.mode !== 'course' && (
            <main
              id="main-content"
              tabIndex={-1}
              className="max-w-6xl mx-auto bg-white rounded-[2rem] shadow-2xl shadow-slate-200/50 border border-slate-200 min-h-[700px] p-5 sm:p-8 md:p-12 focus:outline-none"
            >
              <AnimatePresence mode="wait">
                <m.div
                  key={route.mode}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="h-full"
                >
                  <SectionTitle
                    icon={route.mode === 'package' ? BookOpen : Sparkles}
                    title={t(`app.section.${route.mode}.title`)}
                    subtitle={t(`app.section.${route.mode}.subtitle`)}
                  />
                  {route.mode === 'package' ? (
                    <PackageModule openId={route.item} onOpen={(item) => navigate({ mode: 'package', item })} />
                  ) : (
                    <ExtendModule openId={route.item} onOpen={(item) => navigate({ mode: 'extend', item })} />
                  )}
                </m.div>
              </AnimatePresence>
            </main>
          )}

          {/* Main Dashboard */}
          {route.mode === 'course' && (
            <div className="max-w-6xl mx-auto bg-white rounded-[2rem] shadow-2xl shadow-slate-200/50 border border-slate-200 overflow-hidden flex flex-col md:flex-row min-h-[700px]">
              {/* Sidebar Nav：移动端为横向滚动条，桌面端为竖排侧栏 */}
              <nav
                aria-label={t('app.a11y.module_nav')}
                className="w-full md:w-72 shrink-0 bg-slate-50 border-b md:border-b-0 md:border-r border-slate-200 p-3 md:p-8"
              >
                <p className="hidden md:block text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-6 px-2">{t('app.sidebar.knowledge_core')}</p>
                <ul ref={tabStripRef} className="relative flex md:flex-col gap-2 md:gap-3 overflow-x-auto md:overflow-visible -mx-1 px-1 pb-1 md:pb-0 snap-x">
                  {COURSE_TABS.map((tab) => {
                    const isActive = activeTab === tab.id
                    return (
                      <li key={tab.id} className="shrink-0 snap-start">
                        <a
                          href={`#/course/${tab.id}`}
                          aria-current={isActive ? 'page' : undefined}
                          className={`w-full flex items-center gap-3 md:gap-4 px-3 py-2 md:px-5 md:py-4 rounded-2xl transition-all duration-300 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400
                            ${isActive
                              ? 'bg-white shadow-xl shadow-blue-100/50 text-blue-600 border border-blue-50 md:-translate-y-0.5'
                              : 'text-slate-500 hover:bg-slate-200/50 hover:text-slate-800'}`}
                        >
                          <span className={`p-2 rounded-lg ${isActive ? 'bg-blue-600 text-white' : 'bg-slate-200 text-slate-400'}`} aria-hidden>
                            <tab.icon size={18} />
                          </span>
                          <span className="flex flex-col">
                            <span className="text-sm font-black whitespace-nowrap">{t(`app.tabs.${tab.id}`)}</span>
                            <span className="hidden md:block text-[10px] opacity-60 font-medium truncate w-36">{t(`app.tabs_sub.${tab.id}`)}</span>
                          </span>
                        </a>
                      </li>
                    )
                  })}
                </ul>
                <div className="hidden md:block mt-16">{examTip}</div>
              </nav>

              {/* Content Area */}
              <main id="main-content" tabIndex={-1} className="flex-1 min-w-0 p-5 sm:p-8 md:p-12 focus:outline-none">
                <AnimatePresence mode="wait">
                  <m.div
                    key={activeTab}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.3, ease: 'easeOut' }}
                    className="h-full"
                  >
                    <SectionTitle
                      icon={active.icon}
                      title={t(`app.section.${active.id}.title`)}
                      subtitle={t(`app.section.${active.id}.subtitle`)}
                    />
                    <ActiveModule />
                    <div className="md:hidden mt-8">{examTip}</div>
                  </m.div>
                </AnimatePresence>
              </main>
            </div>
          )}

          {/* Footer */}
          <footer className="max-w-6xl mx-auto mt-12 pt-8 border-t border-slate-200 flex flex-col md:flex-row items-center justify-between gap-6 text-slate-400 text-xs font-medium uppercase tracking-widest text-center md:text-left">
            <p>{t('app.footer.copyright')}</p>
            <div className="flex flex-wrap justify-center gap-6 md:gap-8">
              <span className="flex items-center gap-2"><WifiOff size={14} aria-hidden /> {t('app.footer.local')}</span>
              <span className="flex items-center gap-2"><Sigma size={14} aria-hidden /> {t('app.footer.math')}</span>
            </div>
          </footer>
        </div>
      </MotionConfig>
    </LazyMotion>
  )
}
