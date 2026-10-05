/// <reference types="vite/client" />

/** package.json 的 version，由 vite.config.js 的 define 注入 */
declare const __APP_VERSION__: string

/** 构建期预渲染的 Markdown 笔记（见 scripts/vite-plugins.mjs 的 markdownHtml） */
declare module '*.md?html' {
  const note: { html: string; chars: number }
  export default note
}
