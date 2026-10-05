// 从 src/locales/zh.json 生成香港繁体 src/locales/zh-HK.json（OpenCC s2hk）
// 用法：npm run gen:zh-hk —— 繁体文件由脚本生成，请勿手改。转换规则见 zh-hk.mjs。
import { toHK } from './zh-hk.mjs'
import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const srcPath = join(root, 'src/locales/zh.json')
const outPath = join(root, 'src/locales/zh-HK.json')

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
