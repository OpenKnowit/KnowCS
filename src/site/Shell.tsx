import { useEffect, useRef, useState } from 'react'
import type { KeyboardEvent, ReactNode } from 'react'
import { Check, ChevronDown, Languages } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { normalizeLang, SUPPORTED_LANGS } from '../lib/lang'
import type { LangCode } from '../lib/lang'

const LANG_LABELS: Record<LangCode, string> = {
  en: 'English',
  zh: '简体中文',
  'zh-HK': '繁體中文',
}

export type Section = 'course' | 'playground' | 'watch' | 'drill' | 'notes' | 'extend'

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
        <span className="hidden lg:inline">{LANG_LABELS[currentLang]}</span>
        <ChevronDown size={14} className={`text-slate-500 transition-transform ${open ? 'rotate-180' : ''}`} aria-hidden />
      </button>
      {open && (
          // a CSS fade keeps the animation library out of the entry chunk every page loads
          <div
            role="menu"
            aria-label={t('app.a11y.language')}
            onKeyDown={onMenuKeyDown}
            className="lang-menu absolute right-0 z-50 mt-2 w-40 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-xl shadow-slate-200/50"
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
          </div>
        )}
    </div>
  )
}

/** The site icon (same drawing as the favicon and app icons in vite.config.js). */
const SiteLogo = () => (
  <svg viewBox="0 0 64 64" className="h-9 w-9 shrink-0 drop-shadow-md" aria-hidden>
    <defs>
      <linearGradient id="knowcs-logo" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stopColor="#3b82f6" />
        <stop offset="1" stopColor="#4f46e5" />
      </linearGradient>
    </defs>
    <rect width="64" height="64" rx="14" fill="url(#knowcs-logo)" />
    <g stroke="#fff" strokeWidth="2.8" opacity="0.8">
      {[[16, 20, 32, 14], [16, 20, 32, 32], [16, 20, 32, 50], [16, 44, 32, 14], [16, 44, 32, 32], [16, 44, 32, 50], [32, 14, 48, 32], [32, 32, 48, 32], [32, 50, 48, 32]].map(([x1, y1, x2, y2], i) => (
        <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} />
      ))}
    </g>
    <g fill="#fff">
      {[[16, 20], [16, 44], [32, 14], [32, 32], [32, 50], [48, 32]].map(([cx, cy], i) => (
        <circle key={i} cx={cx} cy={cy} r="6" />
      ))}
    </g>
  </svg>
)

const NAV: { section: Section; href: string; key: string }[] = [
  { section: 'course', href: '/', key: 'app.nav.course' },
  { section: 'playground', href: '/playground/', key: 'app.nav.playground' },
]
/** The other sections, grouped under "More" in this order. */
const MORE: { section: Section; href: string; key: string }[] = [
  { section: 'watch', href: '/watch/', key: 'app.nav.watch' },
  { section: 'notes', href: '/notes/', key: 'app.nav.package' },
  { section: 'drill', href: '/drill/', key: 'app.nav.drill' },
  { section: 'extend', href: '/extend/', key: 'app.nav.extend' },
]

const pill = (on: boolean) => `rounded-full px-2.5 py-1.5 text-sm font-bold transition sm:px-3.5 ${on ? 'bg-blue-600 text-white shadow-md shadow-blue-200' : 'text-slate-500 hover:text-slate-800'}`

/** "More ▾": Watch, Package, Check, Extend. Shows the current one's name when you are in it. */
function MoreMenu({ section }: { section: Section }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const current = MORE.find((m) => m.section === section)
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Escape') {
      setOpen(false)
      buttonRef.current?.focus()
      return
    }
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return
    e.preventDefault()
    const items = [...e.currentTarget.querySelectorAll<HTMLAnchorElement>('[role="menuitem"]')]
    const idx = items.indexOf(document.activeElement as HTMLAnchorElement)
    items[(idx + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length]?.focus()
  }
  return (
    <div className="relative" ref={rootRef}>
      <button ref={buttonRef} type="button" onClick={() => setOpen((o) => !o)} aria-haspopup="menu" aria-expanded={open} className={`flex items-center gap-1 ${pill(!!current)}`}>
        {current ? t(current.key) : t('app.nav.more')}
        <ChevronDown size={14} className={`transition-transform ${open ? 'rotate-180' : ''}`} aria-hidden />
      </button>
      {open && (
        <div role="menu" aria-label={t('app.nav.more')} onKeyDown={onKey} className="lang-menu absolute right-0 z-50 mt-2 w-44 overflow-hidden rounded-2xl border border-slate-200 bg-white py-1 shadow-xl shadow-slate-200/50">
          {MORE.map((m, i) => (
            <a
              key={m.section}
              role="menuitem"
              href={m.href}
              autoFocus={i === 0}
              aria-current={section === m.section ? 'page' : undefined}
              className={`flex items-center justify-between px-4 py-2.5 text-sm font-bold focus:outline-none focus-visible:bg-slate-100 ${section === m.section ? 'bg-blue-50 text-blue-600' : 'text-slate-600 hover:bg-slate-50'}`}
            >
              {t(m.key)}
              {section === m.section && <Check size={14} aria-hidden />}
            </a>
          ))}
        </div>
      )}
    </div>
  )
}

/** Site header around every page. */
export function Shell({ section, children }: { section: Section; children: ReactNode }) {
  const { t } = useTranslation()
  return (
    <div className="min-h-screen bg-[#f1f5f9] text-slate-900 print:bg-white">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-blue-600 focus:px-4 focus:py-2 focus:font-bold focus:text-white"
      >
        {t('app.a11y.skip')}
      </a>
      <header className="mx-auto flex max-w-[1480px] flex-wrap items-center gap-3 px-4 pb-2 pt-4 sm:px-6 print:hidden">
        <a href="/" className="flex min-w-0 flex-1 items-center gap-2.5 font-black tracking-tight text-slate-900 lg:flex-none">
          <SiteLogo />
          <span className="truncate text-lg">{t('app.title')}</span>
          <span className="hidden rounded-full bg-indigo-100 px-2 py-0.5 text-[10px] font-black uppercase text-indigo-600 md:inline">Beta</span>
        </a>
        <span className="hidden flex-1 lg:block" />
        <nav aria-label={t('app.a11y.mode_nav')} className="order-last flex w-full items-center justify-between rounded-full border border-slate-200 bg-white p-1 shadow-sm sm:order-none sm:w-auto sm:justify-start">
          {NAV.map((n) => (
            <a key={n.section} href={n.href} aria-current={section === n.section ? 'page' : undefined} className={pill(section === n.section)}>
              {t(n.key)}
            </a>
          ))}
          <MoreMenu section={section} />
        </nav>
        <LanguageMenu />
      </header>
      <div id="main-content" tabIndex={-1} className="focus:outline-none">
        {children}
      </div>
      <div className="h-10 print:hidden" aria-hidden />
    </div>
  )
}
