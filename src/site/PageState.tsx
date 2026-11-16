import { Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import { LoaderCircle, RotateCw } from 'lucide-react'
import { isLoadFailure } from '../lib/failure'
import { detectLang } from '../lib/lang'
import type { LangCode } from '../lib/lang'
import { markReloaded, wouldReloadFor } from './raise'

// What a visitor sees instead of a blank screen: a mark while a slow page loads, and a message with a way out when a
// page cannot be shown. The message's words are written here and not in the locale files: the strings are one of
// the things that may not have arrived.

const TEXT: Record<LangCode, { load: [string, string]; crash: [string, string]; reload: string; home: string }> = {
  en: {
    load: ['This page did not load', 'Part of it could not be downloaded. Check your connection, then reload.'],
    crash: ['Something went wrong on this page', 'Reloading usually fixes it. If it keeps happening, the address at the bottom of the home page reaches us.'],
    reload: 'Reload',
    home: 'Home',
  },
  zh: {
    load: ['页面没有加载出来', '页面的一部分没有下载成功。请检查网络，然后重新加载。'],
    crash: ['这个页面出了点问题', '重新加载通常就能解决。如果反复出现，可以用首页底部的邮箱告诉我们。'],
    reload: '重新加载',
    home: '回到首页',
  },
  'zh-HK': {
    load: ['頁面沒有載入', '頁面的一部分沒有下載成功。請檢查網絡，然後重新載入。'],
    crash: ['這個頁面出了點問題', '重新載入通常就能解決。如果反覆出現，可以用首頁底部的電郵告訴我們。'],
    reload: '重新載入',
    home: '回到首頁',
  },
}

/** Shown while a page's code is on its way. Nothing for the first 0.6 s (most pages are up by then), then a quiet spinner. */
export function PageLoading() {
  return (
    <div className="grid min-h-[70vh] place-items-center" aria-busy="true">
      <LoaderCircle size={22} className="late-in text-slate-400" aria-hidden />
    </div>
  )
}

/**
 * The page could not be shown: why, a Reload button and a link home.
 * `bare` is for when the site's header is not there either (nothing could be rendered around it).
 */
export function PageFailure({ kind, bare = false }: { kind: 'load' | 'crash'; bare?: boolean }) {
  const text = TEXT[detectLang()]
  const [title, body] = text[kind]
  return (
    <main role="alert" className={`mx-auto grid max-w-xl place-content-center px-4 py-16 text-center ${bare ? 'min-h-screen' : 'min-h-[60vh]'}`}>
      <h1 className="text-2xl font-black tracking-tight text-slate-900">{title}</h1>
      <p className="mt-3 leading-relaxed text-slate-600">{body}</p>
      <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
        <button type="button" onClick={() => window.location.reload()} className="flex items-center gap-1.5 rounded-[10px] bg-blue-600 px-4 py-2 text-sm font-bold text-white hover:bg-blue-700">
          <RotateCw size={15} aria-hidden />
          {text.reload}
        </button>
        <a href="/" className="rounded-[10px] border border-slate-300 bg-white px-4 py-2 text-sm font-bold text-slate-700 hover:border-slate-400">
          {text.home}
        </a>
      </div>
    </main>
  )
}

interface State {
  /** null: fine · reload: a download failed and the page is reloading itself, once · load / crash: tell the visitor */
  failed: null | 'reload' | 'load' | 'crash'
}

/**
 * Catches whatever a page throws while rendering, a lazily loaded page's code that did not arrive included, so that
 * one broken page is a message under the site's header and not a blank screen.
 */
export class PageBoundary extends Component<{ bare?: boolean; children: ReactNode }, State> {
  state: State = { failed: null }
  // React may ask this more than once for one error, so it only reads; the reload itself happens in componentDidCatch
  static getDerivedStateFromError(error: unknown): State {
    return { failed: wouldReloadFor(error) ? 'reload' : isLoadFailure(error) ? 'load' : 'crash' }
  }
  componentDidCatch(error: unknown, info: ErrorInfo) {
    console.error(error, info.componentStack)
    if (this.state.failed === 'reload') {
      markReloaded()
      window.location.reload()
    }
  }
  render() {
    const { failed } = this.state
    if (!failed) return this.props.children
    if (failed === 'reload') return <PageLoading />
    return <PageFailure kind={failed} bare={this.props.bare} />
  }
}
