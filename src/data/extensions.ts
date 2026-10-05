// --- Extend 拓展：课外延伸内容（自包含 HTML，经 iframe 隔离渲染） ---
// 维护 en / zh 两份源文件；zh-HK 在构建期由 zh 转换（?raw-hk，见 scripts/vite-plugins.mjs）。
import type { LangCode } from '../lib/lang'
import attentionEn from '../content/attention.en.html?raw'
import attentionZh from '../content/attention.zh.html?raw'
import attentionHk from '../content/attention.zh.html?raw-hk'

export interface ExtendEntry {
  id: string
  titleKey: string // i18n 键（extend.items.<id>.title / .desc）
  descKey: string
  tag: string
  tagClass: string
  html: Record<LangCode, string>
}

export const EXTENSIONS: ExtendEntry[] = [
  {
    id: 'attention',
    titleKey: 'extend.items.attention.title',
    descKey: 'extend.items.attention.desc',
    tag: 'Transformer',
    tagClass: 'bg-violet-100 text-violet-700',
    html: { en: attentionEn, zh: attentionZh, 'zh-HK': attentionHk },
  },
]
