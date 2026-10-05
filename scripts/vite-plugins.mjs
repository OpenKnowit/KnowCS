// 构建期 Vite 插件：在单文件打包的前提下压缩产物体积。
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkGfm from 'remark-gfm'
import remarkRehype from 'remark-rehype'
import rehypeStringify from 'rehype-stringify'

// ---------------------------------------------------------------------------
// 1) KaTeX 字体瘦身
//    katex.min.css 为每个字体声明 woff2 / woff / ttf 三份 src，单文件打包会把三份都
//    base64 内联（≈1.4MB）。现代浏览器只用 woff2，因此只保留 woff2；同时删掉站内公式
//    用不到的字族（花体 / 哥特 / 手写 / 无衬线 / 等宽）。若以后公式用到 \mathcal、
//    \mathfrak、\mathscr、\mathsf、\texttt，把对应字族加回 KATEX_FONT_KEEP 即可。
// ---------------------------------------------------------------------------
const KATEX_FONT_KEEP = /KaTeX_(Main|Math|AMS|Size[1-4])\b/

export function katexFontSlim() {
  return {
    name: 'knowcs:katex-font-slim',
    enforce: 'pre',
    transform(code, id) {
      if (!/[\\/]katex[\\/]dist[\\/]katex(\.min)?\.css$/.test(id.split('?')[0])) return null
      const out = code
        .replace(/@font-face\s*\{[^}]*\}/g, (rule) => (KATEX_FONT_KEEP.test(rule) ? rule : ''))
        .replace(/,\s*url\([^)]*\.(?:woff|ttf)\)\s*format\(["'](?:woff|truetype)["']\)/g, '')
      return { code: out, map: null }
    },
  }
}

// ---------------------------------------------------------------------------
// 2) Markdown 构建期预渲染：`import note from './x.md?html'` → { html, chars }
//    用 remark/rehype 在构建期把笔记转成 HTML 字符串，运行时无需打包 markdown 解析器。
//    - 相对路径图片改为 ES import（单文件打包下即 data URI），并加 loading="lazy"
//    - 外链新窗口打开；表格包一层可横向滚动的容器
//    - Markdown 中的原始 HTML 一律按文本显示（与 react-markdown 默认行为一致）
// ---------------------------------------------------------------------------
const IMG_TOKEN = (i) => `__KNOWCS_IMG_${i}__`

/** mdast：把原始 HTML 节点降级为纯文本，避免被当作标签解析或丢弃 */
const remarkHtmlAsText = () => (tree) => {
  const walk = (node) => {
    if (node.type === 'html') node.type = 'text'
    node.children?.forEach(walk)
  }
  walk(tree)
}

/** hast：图片 / 链接 / 表格后处理，收集需要 import 的图片路径 */
const rehypeNoteTweaks = (imports, baseDir) => () => (tree) => {
  const walk = (node) => {
    if (!node.children) return
    node.children = node.children.map((child) => {
      if (child.type !== 'element') return child
      walk(child)
      const props = child.properties ?? (child.properties = {})
      if (child.tagName === 'img') {
        const src = typeof props.src === 'string' ? props.src : ''
        if (src && !/^(?:[a-z]+:|\/)/i.test(src)) {
          props.src = IMG_TOKEN(imports.length)
          imports.push(resolve(baseDir, src))
        }
        props.loading = 'lazy'
        props.decoding = 'async'
      } else if (child.tagName === 'a') {
        props.target = '_blank'
        props.rel = 'noreferrer'
      } else if (child.tagName === 'table') {
        return { type: 'element', tagName: 'div', properties: { className: ['note-table'] }, children: [child] }
      }
      return child
    })
  }
  walk(tree)
}

export function markdownHtml() {
  return {
    name: 'knowcs:markdown-html',
    enforce: 'pre',
    async load(id) {
      const [file, query] = id.split('?')
      if (!file.endsWith('.md') || query !== 'html') return null
      this.addWatchFile(file)
      const md = readFileSync(file, 'utf8')
      const imports = []
      const html = String(
        await unified()
          .use(remarkParse)
          .use(remarkGfm)
          .use(remarkHtmlAsText)
          .use(remarkRehype)
          .use(rehypeNoteTweaks(imports, dirname(file)))
          .use(rehypeStringify)
          .process(md)
      )
      // 占位符切分：偶数段为 HTML 文本，奇数段为图片序号
      const parts = html.split(/__KNOWCS_IMG_(\d+)__/)
      const expr = parts.map((p, i) => (i % 2 ? `__img${p}` : JSON.stringify(p))).join(' + ')
      return [
        ...imports.map((p, i) => `import __img${i} from ${JSON.stringify(p)};`),
        `export default { html: ${expr}, chars: ${md.length} };`,
      ].join('\n')
    },
  }
}
