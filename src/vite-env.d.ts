/// <reference types="vite/client" />

/** package.json 的 version，由 vite.config.js 的 define 注入 */
declare const __APP_VERSION__: string

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

/** 拓展页面的香港繁体版本（见 scripts/vite-plugins.mjs 的 rawHk） */
declare module '*.html?raw-hk' {
  const html: string
  export default html
}
