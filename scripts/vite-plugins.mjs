// 构建期 Vite 插件：在单文件打包的前提下压缩产物体积。
import { readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkGfm from 'remark-gfm'
import remarkRehype from 'remark-rehype'
import rehypeStringify from 'rehype-stringify'
import { toHK } from './zh-hk.mjs'

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
// 2) Markdown 构建期预渲染：`import note from './x.md?html'` → { html, chars }；`?chars` 只给字符数
//    `?html-hk`：先把简体源文件转成香港繁体（与 zh-HK.json 同一套规则）再渲染
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
      if (!file.endsWith('.md') || !['html', 'html-hk', 'chars', 'chars-hk'].includes(query)) return null
      this.addWatchFile(file)
      const src = readFileSync(file, 'utf8')
      const md = query.endsWith('-hk') ? toHK(src) : src
      // `?chars` / `?chars-hk`: just the length, so a note list need not load every note's HTML
      if (query.startsWith('chars')) return `export default ${md.length};`
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

// ---------------------------------------------------------------------------
// 3) 拓展页面的香港繁体版本：`import page from './x.zh.html?raw-hk'` → 转换后的字符串
//    与 ?raw 相同但先做简→港繁转换，并把 <html lang> 改为 zh-HK；繁体不单独维护源文件。
// ---------------------------------------------------------------------------
export function rawHk() {
  return {
    name: 'knowcs:raw-hk',
    enforce: 'pre',
    load(id) {
      const [file, query] = id.split('?')
      if (query !== 'raw-hk') return null
      this.addWatchFile(file)
      const text = toHK(readFileSync(file, 'utf8')).replace(/<html([^>]*)\blang="[^"]*"/, '<html$1lang="zh-HK"')
      return `export default ${JSON.stringify(text)};`
    },
  }
}

// ---------------------------------------------------------------------------
// 4) 文案拆包：只在某一页用到的文案（讲解视频字幕、各实验页、NumPy、自测、公式表）不放进每页都加载的主文案。
//    `locales/<lng>.json?core`        → 主文案：去掉所有包；watch.<集> 只保留 kicker / title / sub
//    `locales/<lng>.json?pack=<路径>`  → 只含该路径的对象，如 { lab: { otsu: … } }
//    `virtual:locale-packs`           → { <路径>: { <lng>: () => import(...) } }，每包每语言一个 chunk
//    包清单由 en.json 推出（packNames）；load 返回 JSON 文本，交给 Vite 自带的 JSON 插件转为模块。
// ---------------------------------------------------------------------------
const EP_CORE = ['kicker', 'title', 'sub']
const LAB_SHARED = ['common', 'errors']
// one page each (the classic modules, NumPy, drill, formula sheet); perceptron / pytorch belong to retired modules
const FIXED_PACKS = ['numpy_module', 'numpy_api', 'drill', 'formulas.items', 'bayes', 'knn', 'alphabeta', 'backprop_module', 'kmeans', 'kernel_module', 'perceptron', 'pytorch']

export function packNames(strings) {
  return [
    ...Object.keys(strings.watch ?? {}).filter((k) => k !== 'ui').map((k) => `watch.${k}`),
    ...Object.keys(strings.lab ?? {}).filter((k) => !LAB_SHARED.includes(k)).map((k) => `lab.${k}`),
    // the library playgrounds on the note pages: shared UI strings and one pack per library
    ...Object.keys(strings.playground ?? {}).map((k) => `playground.${k}`),
    ...FIXED_PACKS,
  ]
}

const getPath = (o, path) => (path ? path.split('.').reduce((v, k) => (v == null ? v : v[k]), o) : o)
function setPath(o, path, value) {
  const keys = path.split('.')
  let cur = o
  keys.slice(0, -1).forEach((k) => (cur = cur[k] ??= {}))
  cur[keys.at(-1)] = value
  return o
}
const pick = (o, fields, keep) => Object.fromEntries(Object.entries(o ?? {}).filter(([k]) => fields.includes(k) === keep))

export function corePart(strings) {
  const out = structuredClone(strings)
  for (const name of packNames(strings)) {
    const keys = name.split('.')
    const parent = getPath(out, keys.slice(0, -1).join('.'))
    if (!parent) continue
    if (keys[0] === 'watch') parent[keys[1]] = pick(parent[keys[1]], EP_CORE, true)
    else delete parent[keys.at(-1)]
  }
  return out
}

export function packPart(strings, name) {
  const value = getPath(strings, name)
  return setPath({}, name, name.startsWith('watch.') ? pick(value, EP_CORE, false) : value)
}

export function localeSplit() {
  const VIRTUAL = 'virtual:locale-packs'
  const RESOLVED = '\0' + VIRTUAL
  // the Vite root is site/, so locate the locales from this file instead
  const localesDir = join(dirname(fileURLToPath(import.meta.url)), '../src/locales')
  return {
    name: 'knowcs:locale-split',
    enforce: 'pre',
    resolveId(id) {
      return id === VIRTUAL ? RESOLVED : null
    },
    load(id) {
      if (id === RESOLVED) {
        const file = join(localesDir, 'en.json')
        this.addWatchFile(file)
        const lngs = ['en', 'zh', 'zh-HK']
        const body = packNames(JSON.parse(readFileSync(file, 'utf8')))
          .map((p) => `  ${JSON.stringify(p)}: { ${lngs.map((l) => `${JSON.stringify(l)}: () => import(${JSON.stringify(`${join(localesDir, l)}.json?pack=${p}`)})`).join(', ')} },`)
          .join('\n')
        return `export default {\n${body}\n}\n`
      }
      const [file, query] = id.split('?')
      if (!file.endsWith('.json') || !query || !(query === 'core' || query.startsWith('pack='))) return null
      this.addWatchFile(file)
      const all = JSON.parse(readFileSync(file, 'utf8'))
      return JSON.stringify(query === 'core' ? corePart(all) : packPart(all, query.slice(5)))
    },
  }
}
