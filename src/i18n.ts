import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import LanguageDetector from 'i18next-browser-languagedetector'
import { normalizeLang, SUPPORTED_LANGS } from './lib/lang'
import translationEN from './locales/en.json'
import translationZH from './locales/zh.json'
import translationZHHK from './locales/zh-HK.json'

const resources = {
  en: { translation: translationEN },
  zh: { translation: translationZH },
  'zh-HK': { translation: translationZHHK },
}

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    supportedLngs: [...SUPPORTED_LANGS],
    // zh-HK 漏译时先回退简体，再回退英文
    fallbackLng: { 'zh-HK': ['zh', 'en'], default: ['en'] },
    load: 'currentOnly',
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
