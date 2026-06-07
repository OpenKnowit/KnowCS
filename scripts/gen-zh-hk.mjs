// 从 src/locales/zh.json 生成香港繁体 src/locales/zh-HK.json（OpenCC s2hk）
// 用法：npm run gen:zh-hk —— 繁体文件由脚本生成，请勿手改。
import { Converter } from 'opencc-js'
import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const srcPath = join(root, 'src/locales/zh.json')
const outPath = join(root, 'src/locales/zh-HK.json')

const convert = Converter({ from: 'cn', to: 'hk' })

// OpenCC 只做字级转换，大陆 IT 用语需二次映射为香港惯用语（在转换后的繁体上替换）
const HK_TERMS = [
  ['內存', '記憶體'],
  ['算法', '演算法'],
  ['交互', '互動'], // 交互式 → 互動式 一并覆盖
  ['創建', '建立'],
  ['噪聲', '雜訊'],
  ['過濾器', '濾波器'],
  ['學長寄語', '師兄寄語'], // 港校用「師兄/師姐」
]

const toHK = (text) => {
  let out = convert(text)
  for (const [from, to] of HK_TERMS) out = out.replaceAll(from, to)
  return out
}

const convertDeep = (node) => {
  if (typeof node === 'string') return toHK(node)
  if (Array.isArray(node)) return node.map(convertDeep)
  if (node && typeof node === 'object') {
    return Object.fromEntries(Object.entries(node).map(([k, v]) => [k, convertDeep(v)]))
  }
  return node
}

const zh = JSON.parse(readFileSync(srcPath, 'utf8'))
writeFileSync(outPath, `${JSON.stringify(convertDeep(zh), null, 2)}\n`)
console.log(`Generated ${outPath}`)
