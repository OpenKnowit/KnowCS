// --- 语言代码归一化：浏览器 / URL / localStorage 给出的任意 BCP-47 标签 → 站点支持的三种语言 ---

export const SUPPORTED_LANGS = ['en', 'zh', 'zh-HK'] as const
export type LangCode = (typeof SUPPORTED_LANGS)[number]

/**
 * 繁体地区（HK/TW/MO）与显式 Hant 脚本 → zh-HK；其余 zh-* → 简体 zh；其他一律 en。
 * 大小写不敏感，兼容 `zh_TW` 这类下划线写法。
 */
export const normalizeLang = (lang: string | undefined | null): LangCode => {
  const tag = (lang ?? '').toLowerCase().replace(/_/g, '-')
  if (!tag.startsWith('zh')) return 'en'
  if (/(^|-)(hant|hk|tw|mo)(-|$)/.test(tag)) return 'zh-HK'
  return 'zh'
}

/**
 * The language i18next's detector will pick (order: ?lang=, saved choice, browser), available before init has run.
 * Keep in step with the detection options in src/i18n.ts and the inline preload script in vite.config.js.
 */
export function detectLang(): LangCode {
  try {
    const q = new URLSearchParams(location.search).get('lang')
    const saved = localStorage.getItem('knowcs-lang')
    return normalizeLang(q || saved || navigator.languages?.[0] || navigator.language)
  } catch {
    return 'en'
  }
}
