import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  BarChart3,
  Calculator,
  ChevronDown,
  Cpu,
  Grid3X3,
  Info,
  Languages,
  Layers,
  MousePointer2,
  RefreshCw,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { Trans, useTranslation } from 'react-i18next'
import { Latex } from './components/Latex'
import { SectionTitle } from './components/SectionTitle'
import { BackpropModule } from './modules/BackpropModule'
import { BayesBasicsModule } from './modules/BayesBasicsModule'
import { KernelModule } from './modules/KernelModule'
import { KnnModule } from './modules/KnnModule'
import { NaiveBayesModule } from './modules/NaiveBayesModule'
import { NumpyModule } from './modules/NumpyModule'
import type { TabId } from './types'

interface TabConfig {
  id: TabId
  label: string
  icon: LucideIcon
  subtitle: string
}

const LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'zh', label: '简体中文' },
  { code: 'zh-HK', label: '繁體中文' },
] as const

type LangCode = (typeof LANGUAGES)[number]['code']

const normalizeLang = (lang: string): LangCode => {
  if (lang === 'zh-HK' || lang === 'zh-TW' || lang === 'zh-Hant') return 'zh-HK'
  if (lang.startsWith('zh')) return 'zh'
  return 'en'
}

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('numpy')
  const { t, i18n } = useTranslation()

  const tabs: TabConfig[] = [
    { id: 'numpy', label: t('app.tabs.numpy'), icon: Cpu, subtitle: 'View vs Copy & Broadcasting' },
    { id: 'backprop', label: t('app.tabs.backprop'), icon: Layers, subtitle: 'The Chain Rule Visualized' },
    { id: 'kernel', label: t('app.tabs.kernel'), icon: Grid3X3, subtitle: 'Edge Detection Simulation' },
    { id: 'bayesBasics', label: t('app.tabs.bayesBasics'), icon: Calculator, subtitle: 'Formula & Fire Alarm Case' },
    { id: 'naiveBayes', label: t('app.tabs.naiveBayes'), icon: BarChart3, subtitle: 'Inference & Smoothing' },
    { id: 'knn', label: t('app.tabs.knn'), icon: MousePointer2, subtitle: 'Distance Metric & Decision Boundary' },
  ]

  const [langMenuOpen, setLangMenuOpen] = useState(false)
  const langMenuRef = useRef<HTMLDivElement>(null)
  const currentLang = normalizeLang(i18n.language)

  useEffect(() => {
    if (!langMenuOpen) return
    const handleClickOutside = (e: MouseEvent) => {
      if (langMenuRef.current && !langMenuRef.current.contains(e.target as Node)) {
        setLangMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [langMenuOpen])

  const selectLanguage = (code: LangCode) => {
    i18n.changeLanguage(code)
    setLangMenuOpen(false)
  }

  return (
    <div className="min-h-screen bg-[#f1f5f9] text-slate-900 font-sans p-4 md:p-8">
      {/* Header */}
      <header className="max-w-6xl mx-auto mb-8 flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-4xl font-black text-slate-900 tracking-tight">
            {t('app.title')}
          </h1>
          <div className="flex items-center gap-3 mt-2">
            <p className="text-slate-500 font-medium">{t('app.subtitle')}</p>
            <span className="px-2 py-0.5 bg-indigo-100 text-indigo-600 text-[10px] rounded-full uppercase font-black">{t('app.version')}</span>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="px-4 py-2 bg-white border border-slate-200 rounded-full shadow-sm flex items-center gap-3 text-xs font-mono text-slate-500 hidden sm:flex">
            <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></div>
            Environment: Ready / KaTeX Engine Active
          </div>
          <div className="relative" ref={langMenuRef}>
            <button
              onClick={() => setLangMenuOpen((open) => !open)}
              className="p-2 bg-white rounded-full shadow-sm hover:bg-gray-50 transition flex items-center gap-2 px-4 font-bold text-slate-600"
              title="Switch Language"
            >
              <Languages size={20} className="text-blue-600" />
              {LANGUAGES.find((lang) => lang.code === currentLang)?.label}
              <ChevronDown size={14} className={`text-slate-400 transition-transform ${langMenuOpen ? 'rotate-180' : ''}`} />
            </button>
            <AnimatePresence>
              {langMenuOpen && (
                <motion.div
                  initial={{ opacity: 0, y: -6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  transition={{ duration: 0.15 }}
                  className="absolute right-0 mt-2 w-40 bg-white border border-slate-200 rounded-2xl shadow-xl shadow-slate-200/50 overflow-hidden z-50"
                >
                  {LANGUAGES.map((lang) => (
                    <button
                      key={lang.code}
                      onClick={() => selectLanguage(lang.code)}
                      className={`w-full px-4 py-2.5 text-left text-sm font-bold transition
                        ${currentLang === lang.code
                          ? 'bg-blue-50 text-blue-600'
                          : 'text-slate-600 hover:bg-slate-50'}`}
                    >
                      {lang.label}
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </header>

      {/* Main Dashboard */}
      <main className="max-w-6xl mx-auto bg-white rounded-[2rem] shadow-2xl shadow-slate-200/50 border border-slate-200 overflow-hidden flex flex-col md:flex-row min-h-[700px]">
        {/* Sidebar Nav */}
        <nav className="w-full md:w-72 bg-slate-50 border-r border-slate-200 p-8 space-y-3">
          <p className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-6 px-2">{t('app.sidebar.knowledge_core')}</p>
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`w-full flex items-center gap-4 px-5 py-4 rounded-2xl transition-all duration-300 text-left
                ${activeTab === tab.id
                  ? 'bg-white shadow-xl shadow-blue-100/50 text-blue-600 border border-blue-50 -translate-y-0.5'
                  : 'text-slate-500 hover:bg-slate-200/50 hover:text-slate-800'}`}
            >
              <div className={`p-2 rounded-lg ${activeTab === tab.id ? 'bg-blue-600 text-white' : 'bg-slate-200 text-slate-400'}`}>
                <tab.icon size={18} />
              </div>
              <div className="flex flex-col">
                <span className="text-sm font-black">{tab.label}</span>
                <span className="text-[10px] opacity-60 font-medium truncate w-32">{tab.subtitle}</span>
              </div>
            </button>
          ))}

          <div className="mt-16 p-6 bg-gradient-to-br from-blue-600 to-indigo-700 rounded-[1.5rem] text-white shadow-lg shadow-blue-200 relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:scale-125 transition-transform duration-500">
               <Cpu size={80} />
            </div>
            <div className="relative z-10">
              <div className="flex items-center gap-2 mb-3">
                <Info size={16} />
                <span className="text-xs font-bold uppercase tracking-wider">{t('app.sidebar.exam_tip.title')}</span>
              </div>
              <p className="text-[11px] leading-relaxed opacity-90 font-medium">
                <Trans i18nKey="app.sidebar.exam_tip.content">
                  考试注意：<Latex formula="\text{Broadcasting}" /> 只能扩展 Size 为 1 的维度。
                  <Latex formula="\mathbf{A} \cdot \mathbf{B}" /> (np.dot) 和 <Latex formula="\mathbf{A} \odot \mathbf{B}" /> (Element-wise) 物理含义完全不同！
                </Trans>
              </p>
            </div>
          </div>
        </nav>

        {/* Content Area */}
        <section className="flex-1 p-8 md:p-12 overflow-y-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.4, ease: 'easeOut' }}
              className="h-full"
            >
              {activeTab === 'numpy' && (
                <>
                  <SectionTitle
                    icon={Cpu}
                    title={t('app.section.numpy.title')}
                    subtitle={t('app.section.numpy.subtitle')}
                  />
                  <NumpyModule />
                </>
              )}
              {activeTab === 'backprop' && (
                <>
                  <SectionTitle
                    icon={Layers}
                    title={t('app.section.backprop.title')}
                    subtitle={t('app.section.backprop.subtitle')}
                  />
                  <BackpropModule />
                </>
              )}
              {activeTab === 'kernel' && (
                <>
                  <SectionTitle
                    icon={Grid3X3}
                    title={t('app.section.kernel.title')}
                    subtitle={t('app.section.kernel.subtitle')}
                  />
                  <KernelModule />
                </>
              )}
              {activeTab === 'bayesBasics' && (
                <>
                  <SectionTitle
                    icon={Calculator}
                    title={t('app.section.bayesBasics.title')}
                    subtitle={t('app.section.bayesBasics.subtitle')}
                  />
                  <BayesBasicsModule />
                </>
              )}
              {activeTab === 'naiveBayes' && (
                <>
                  <SectionTitle
                    icon={BarChart3}
                    title={t('app.section.naiveBayes.title')}
                    subtitle={t('app.section.naiveBayes.subtitle')}
                  />
                  <NaiveBayesModule />
                </>
              )}
              {activeTab === 'knn' && (
                <>
                  <SectionTitle
                    icon={MousePointer2}
                    title={t('app.section.knn.title')}
                    subtitle={t('app.section.knn.subtitle')}
                  />
                  <KnnModule />
                </>
              )}
            </motion.div>
          </AnimatePresence>
        </section>
      </main>

      {/* Footer Branding */}
      <footer className="max-w-6xl mx-auto mt-12 pt-8 border-t border-slate-200 flex flex-col md:flex-row items-center justify-between gap-6 text-slate-400 text-xs font-medium uppercase tracking-widest">
        <p>{t('app.footer.copyright')}</p>
        <div className="flex gap-8">
          <span className="flex items-center gap-2 hover:text-blue-500 cursor-default transition"><RefreshCw size={14} /> {t('app.footer.latency')}</span>
          <span className="flex items-center gap-2 hover:text-blue-500 cursor-default transition"><Cpu size={14} /> {t('app.footer.render')}</span>
        </div>
      </footer>
    </div>
  )
}
