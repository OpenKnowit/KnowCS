// --- Extend 拓展：课外延伸内容（自包含 HTML，经 iframe 隔离渲染） ---
// 维护 en / zh 两份源文件；zh-HK 在构建期由 zh 转换（?raw-hk，见 scripts/vite-plugins.mjs）。
// 每种语言一个 chunk，打开时才下载。
import type { LangCode } from '../lib/lang'

export interface ExtendEntry {
  id: string
  titleKey: string // i18n 键（extend.items.<id>.title / .desc）
  descKey: string
  tag: string
  tagClass: string
  load: Record<LangCode, () => Promise<{ default: string }>>
}

export const EXTENSIONS: ExtendEntry[] = [
  {
    id: 'attention',
    titleKey: 'extend.items.attention.title',
    descKey: 'extend.items.attention.desc',
    tag: 'Transformer',
    tagClass: 'bg-violet-100 text-violet-700',
    load: {
      en: () => import('../content/attention.en.html?raw'),
      zh: () => import('../content/attention.zh.html?raw'),
      'zh-HK': () => import('../content/attention.zh.html?raw-hk'),
    },
  },
]
