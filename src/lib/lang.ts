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
