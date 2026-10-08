import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import LanguageDetector from 'i18next-browser-languagedetector'
import { normalizeLang, SUPPORTED_LANGS } from './lib/lang'
import type { BackendModule, ReadCallback } from 'i18next'
import { fetchEpisodeStrings, mergeEpisodeStrings, usedEpisodes } from './site/watch/episodeStrings'

// 每种语言一个独立 chunk：页面只下载当前语言。讲解视频的字幕等按集另行加载（site/watch/useEpisode.ts），
// 所以这里只取 ?core（见 scripts/vite-plugins.mjs 的 localeSplit）。
// en / zh 键严格对齐、zh-HK 由 zh 全量生成，所以不预载回退语言。
const loaders: Record<string, () => Promise<{ default: Record<string, unknown> }>> = {
  en: () => import('./locales/en.json?core'),
  zh: () => import('./locales/zh.json?core'),
  'zh-HK': () => import('./locales/zh-HK.json?core'),
}
const lazyLocales: BackendModule = {
  type: 'backend',
  init() {},
  read(lng: string, _ns: string, cb: ReadCallback) {
    const load = loaders[lng] ?? loaders.en
    // on a language switch, bring along the captions of episodes already on the page so a playing video keeps going
    Promise.all([load(), ...[...usedEpisodes].map((id) => fetchEpisodeStrings(id, lng))]).then(
      ([m, ...eps]) => cb(null, mergeEpisodeStrings(m.default, eps)),
      (e: Error) => cb(e, false),
    )
  },
}

/** Resolves once the current language's strings are loaded; render after this. */
export const i18nReady = i18n
  .use(lazyLocales)
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    supportedLngs: [...SUPPORTED_LANGS],
    fallbackLng: false,
    load: 'currentOnly',
    react: { useSuspense: false },
    // 检测顺序：?lang= 链接参数 > 用户上次选择 > 浏览器语言；选择结果写回 localStorage
    detection: {
      order: ['querystring', 'localStorage', 'navigator'],
      lookupQuerystring: 'lang',
      lookupLocalStorage: 'knowcs-lang',
      caches: ['localStorage'],
      convertDetectedLanguage: normalizeLang,
    },
    debug: import.meta.env.DEV,
    interpolation: {
      escapeValue: false, // not needed for react as it escapes by default
    },
  })

// 同步 <html lang>，让屏幕阅读器与浏览器字体回退使用正确语言
const syncHtmlLang = (lng: string) => {
  document.documentElement.lang = normalizeLang(lng)
}
syncHtmlLang(i18n.language)
i18n.on('languageChanged', syncHtmlLang)

export default i18n
