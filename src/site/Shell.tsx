import { useEffect, useRef, useState } from 'react'
import type { KeyboardEvent, ReactNode } from 'react'
import { AnimatePresence, m } from 'framer-motion'
import { Check, ChevronDown, Languages, Sigma, WifiOff } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { normalizeLang, SUPPORTED_LANGS } from '../lib/lang'
import type { LangCode } from '../lib/lang'

const LANG_LABELS: Record<LangCode, string> = {
  en: 'English',
  zh: '简体中文',
  'zh-HK': '繁體中文',
}

export type Section = 'course' | 'watch' | 'drill' | 'notes' | 'extend'

const LanguageMenu = () => {
  const { t, i18n } = useTranslation()
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const currentLang = normalizeLang(i18n.resolvedLanguage ?? i18n.language)

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
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
    const items = [...e.currentTarget.querySelectorAll<HTMLButtonElement>('[role="menuitemradio"]')]
    const idx = items.indexOf(document.activeElement as HTMLButtonElement)
    items[(idx + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length]?.focus()
  }

  return (
    <div className="relative" ref={rootRef}>
      <button
        ref={buttonRef}
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 rounded-full bg-white px-3.5 py-2 text-sm font-bold text-slate-600 shadow-sm transition hover:bg-gray-50"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={t('app.a11y.language')}
      >
        <Languages size={18} className="text-blue-600" aria-hidden />
        <span className="hidden sm:inline">{LANG_LABELS[currentLang]}</span>
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
            className="absolute right-0 z-50 mt-2 w-40 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-xl shadow-slate-200/50"
          >
            {SUPPORTED_LANGS.map((code) => (
              <button
                key={code}
                role="menuitemradio"
                aria-checked={currentLang === code}
                lang={code}
                autoFocus={currentLang === code}
                onClick={() => select(code)}
                className={`flex w-full items-center justify-between px-4 py-2.5 text-left text-sm font-bold transition focus:outline-none focus-visible:bg-slate-100 ${currentLang === code ? 'bg-blue-50 text-blue-600' : 'text-slate-600 hover:bg-slate-50'}`}
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

const NAV: { section: Section; href: string; key: string }[] = [
  { section: 'course', href: '/', key: 'app.nav.course' },
  { section: 'watch', href: '/watch/', key: 'app.nav.watch' },
  { section: 'drill', href: '/drill/', key: 'app.nav.drill' },
  { section: 'notes', href: '/notes/', key: 'app.nav.package' },
  { section: 'extend', href: '/extend/', key: 'app.nav.extend' },
]

/** Site header + footer around every page. */
export function Shell({ section, children }: { section: Section; children: ReactNode }) {
  const { t } = useTranslation()
  return (
    <div className="min-h-screen bg-[#f1f5f9] text-slate-900">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-blue-600 focus:px-4 focus:py-2 focus:font-bold focus:text-white"
      >
        {t('app.a11y.skip')}
      </a>
      <header className="mx-auto flex max-w-[1480px] flex-wrap items-center gap-3 px-4 pb-2 pt-4 sm:px-6">
        <a href="/" className="flex min-w-0 items-center gap-2.5 font-black tracking-tight text-slate-900">
          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 text-sm font-black text-white shadow-md shadow-blue-500/30" aria-hidden>K</span>
          <span className="truncate text-lg">{t('app.title')}</span>
          <span className="hidden rounded-full bg-indigo-100 px-2 py-0.5 text-[10px] font-black uppercase text-indigo-600 md:inline">v{__APP_VERSION__}</span>
        </a>
        <span className="flex-1" />
        <nav aria-label={t('app.a11y.mode_nav')} className="flex items-center rounded-full border border-slate-200 bg-white p-1 shadow-sm">
          {NAV.map((n) => (
            <a
              key={n.section}
              href={n.href}
              aria-current={section === n.section ? 'page' : undefined}
              className={`rounded-full px-2.5 py-1.5 text-sm font-bold transition sm:px-3.5 ${section === n.section ? 'bg-blue-600 text-white shadow-md shadow-blue-200' : 'text-slate-500 hover:text-slate-800'}`}
            >
              {t(n.key)}
            </a>
          ))}
        </nav>
        <LanguageMenu />
      </header>
      <div id="main-content" tabIndex={-1} className="focus:outline-none">
        {children}
      </div>
      <footer className="mx-auto mt-8 flex max-w-[1480px] flex-col items-center justify-between gap-4 border-t border-slate-200 px-4 py-8 text-center text-xs font-medium uppercase tracking-widest text-slate-400 sm:px-6 md:flex-row md:text-left">
        <p>{t('app.footer.copyright')}</p>
        <div className="flex flex-wrap justify-center gap-6">
          <span className="flex items-center gap-2"><WifiOff size={14} aria-hidden /> {t('app.footer.local')}</span>
          <span className="flex items-center gap-2"><Sigma size={14} aria-hidden /> {t('app.footer.math')}</span>
        </div>
      </footer>
    </div>
  )
}
