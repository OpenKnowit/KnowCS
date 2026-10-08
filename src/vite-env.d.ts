/// <reference types="vite/client" />

/** 构建期预渲染的 Markdown 笔记（见 scripts/vite-plugins.mjs 的 markdownHtml） */
declare module '*.md?html' {
  const note: { html: string; chars: number }
  export default note
}

/** 同上，但先把简体源文件转为香港繁体再渲染 */
declare module '*.md?html-hk' {
  const note: { html: string; chars: number }
  export default note
}

/** 只要笔记的字符数（列表页用，不加载正文） */
declare module '*.md?chars' {
  const chars: number
  export default chars
}
declare module '*.md?chars-hk' {
  const chars: number
  export default chars
}

/** 拓展页面的香港繁体版本（见 scripts/vite-plugins.mjs 的 rawHk） */
declare module '*.html?raw-hk' {
  const html: string
  export default html
}

/** 主文案：不含各页文案包（见 scripts/vite-plugins.mjs 的 localeSplit） */
declare module '*.json?core' {
  const strings: Record<string, unknown>
  export default strings
}

/** 只在某一页用到的文案包，按包 × 语言懒加载 */
declare module 'virtual:locale-packs' {
  const loaders: Record<string, Record<string, () => Promise<{ default: Record<string, unknown> }>>>
  export default loaders
}
