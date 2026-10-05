// 简体 → 香港繁体（OpenCC cn→hk + 香港术语映射）。
// 供 gen-zh-hk.mjs（生成 zh-HK.json）与 vite-plugins.mjs（构建期转换笔记 / 拓展页面）共用。
import { Converter } from 'opencc-js'

const convert = Converter({ from: 'cn', to: 'hk' })

// OpenCC 只做字级转换，大陆 IT 用语需二次映射为香港惯用语（在转换后的繁体上替换）
export const HK_TERMS = [
  ['內存', '記憶體'],
  ['算法', '演算法'],
  ['交互', '互動'], // 交互式 → 互動式 一并覆盖
  ['創建', '建立'],
  ['噪聲', '雜訊'],
  ['過濾器', '濾波器'],
  ['學長寄語', '師兄寄語'], // 港校用「師兄/師姐」
]

export const toHK = (text) => {
  let out = convert(text)
  for (const [from, to] of HK_TERMS) out = out.replaceAll(from, to)
  return out
}
