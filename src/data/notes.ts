// --- Package 资料包：构建期预渲染为 HTML 的 Markdown 笔记（?html，见 scripts/vite-plugins.mjs） ---
// 每篇笔记维护 en / zh 两份源文件；zh-HK 由 zh 在构建期经 OpenCC + 香港术语映射生成（?html-hk），不单独维护。
// 图片以 ES import 引入，三种语言共用同一份 data URI，不会重复打包。
import type { LangCode } from '../lib/lang'
import kerasEn from '../content/notes/keras.en.md?html'
import kerasZh from '../content/notes/keras.zh.md?html'
import kerasHk from '../content/notes/keras.zh.md?html-hk'
import kevinEn from '../content/notes/kevin.en.md?html'
import kevinZh from '../content/notes/kevin.zh.md?html'
import kevinHk from '../content/notes/kevin.zh.md?html-hk'
import numpyEn from '../content/notes/numpy.en.md?html'
import numpyZh from '../content/notes/numpy.zh.md?html'
import numpyHk from '../content/notes/numpy.zh.md?html-hk'
import pandasEn from '../content/notes/pandas.en.md?html'
import pandasZh from '../content/notes/pandas.zh.md?html'
import pandasHk from '../content/notes/pandas.zh.md?html-hk'
import pytorchEn from '../content/notes/pytorch.en.md?html'
import pytorchZh from '../content/notes/pytorch.zh.md?html'
import pytorchHk from '../content/notes/pytorch.zh.md?html-hk'
import tensorflowEn from '../content/notes/tensorflow.en.md?html'
import tensorflowZh from '../content/notes/tensorflow.zh.md?html'
import tensorflowHk from '../content/notes/tensorflow.zh.md?html-hk'

export interface NoteBody {
  html: string // 预渲染 HTML（图片已内联）
  chars: number // 原始 Markdown 字符数
}

export interface NoteEntry {
  id: string
  titleKey: string // i18n 键（package.notes.<id>）
  tag: string
  tagClass: string // 卡片标签配色
  body: Record<LangCode, NoteBody>
}

export const NOTES: NoteEntry[] = [
  { id: 'numpy', titleKey: 'package.notes.numpy', tag: 'NumPy', tagClass: 'bg-sky-100 text-sky-700', body: { en: numpyEn, zh: numpyZh, 'zh-HK': numpyHk } },
  { id: 'pandas', titleKey: 'package.notes.pandas', tag: 'pandas', tagClass: 'bg-indigo-100 text-indigo-700', body: { en: pandasEn, zh: pandasZh, 'zh-HK': pandasHk } },
  { id: 'pytorch', titleKey: 'package.notes.pytorch', tag: 'PyTorch', tagClass: 'bg-orange-100 text-orange-700', body: { en: pytorchEn, zh: pytorchZh, 'zh-HK': pytorchHk } },
  { id: 'tensorflow', titleKey: 'package.notes.tensorflow', tag: 'TensorFlow', tagClass: 'bg-amber-100 text-amber-700', body: { en: tensorflowEn, zh: tensorflowZh, 'zh-HK': tensorflowHk } },
  { id: 'keras', titleKey: 'package.notes.keras', tag: 'Keras', tagClass: 'bg-rose-100 text-rose-700', body: { en: kerasEn, zh: kerasZh, 'zh-HK': kerasHk } },
  { id: 'kevin', titleKey: 'package.notes.kevin', tag: 'COMP2211', tagClass: 'bg-emerald-100 text-emerald-700', body: { en: kevinEn, zh: kevinZh, 'zh-HK': kevinHk } },
]
