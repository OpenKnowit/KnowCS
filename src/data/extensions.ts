// --- Extend 拓展：课外延伸内容（自包含 HTML，经 iframe 隔离渲染） ---
import attentionHtml from '../content/attention.html?raw'

export interface ExtendEntry {
  id: string
  titleKey: string // i18n 键（extend.items.<id>.title / .desc）
  descKey: string
  tag: string
  tagClass: string
  html: string
}

export const EXTENSIONS: ExtendEntry[] = [
  {
    id: 'attention',
    titleKey: 'extend.items.attention.title',
    descKey: 'extend.items.attention.desc',
    tag: 'Transformer',
    tagClass: 'bg-violet-100 text-violet-700',
    html: attentionHtml,
  },
]
