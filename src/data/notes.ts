// --- Package 资料包：构建期预渲染为 HTML 的 Markdown 笔记（?html，见 scripts/vite-plugins.mjs） ---
// 每篇笔记维护 en / zh 两份源文件；zh-HK 由 zh 在构建期经 OpenCC + 香港术语映射生成（?html-hk），不单独维护。
// 正文按「笔记 × 语言」分 chunk 懒加载（打开哪篇、用哪种语言才下载哪份）；列表只静态引入字符数（?chars）。
import type { LangCode } from '../lib/lang'
import numpyEnChars from '../content/notes/numpy.en.md?chars'
import numpyZhChars from '../content/notes/numpy.zh.md?chars'
import numpyHkChars from '../content/notes/numpy.zh.md?chars-hk'
import pandasEnChars from '../content/notes/pandas.en.md?chars'
import pandasZhChars from '../content/notes/pandas.zh.md?chars'
import pandasHkChars from '../content/notes/pandas.zh.md?chars-hk'
import pytorchEnChars from '../content/notes/pytorch.en.md?chars'
import pytorchZhChars from '../content/notes/pytorch.zh.md?chars'
import pytorchHkChars from '../content/notes/pytorch.zh.md?chars-hk'
import tensorflowEnChars from '../content/notes/tensorflow.en.md?chars'
import tensorflowZhChars from '../content/notes/tensorflow.zh.md?chars'
import tensorflowHkChars from '../content/notes/tensorflow.zh.md?chars-hk'
import kerasEnChars from '../content/notes/keras.en.md?chars'
import kerasZhChars from '../content/notes/keras.zh.md?chars'
import kerasHkChars from '../content/notes/keras.zh.md?chars-hk'
import kevinEnChars from '../content/notes/kevin.en.md?chars'
import kevinZhChars from '../content/notes/kevin.zh.md?chars'
import kevinHkChars from '../content/notes/kevin.zh.md?chars-hk'

export interface NoteBody {
  html: string // 预渲染 HTML（图片已内联）
  chars: number // 原始 Markdown 字符数
}

type Loader = () => Promise<{ default: NoteBody }>

export interface NoteEntry {
  id: string
  titleKey: string // i18n 键（package.notes.<id>）
  tag: string
  tagClass: string // 卡片标签配色
  chars: Record<LangCode, number>
  load: Record<LangCode, Loader>
}

export const NOTES: NoteEntry[] = [
  {
    id: 'numpy',
    titleKey: 'package.notes.numpy',
    tag: 'NumPy',
    tagClass: 'bg-sky-100 text-sky-700',
    chars: { en: numpyEnChars, zh: numpyZhChars, 'zh-HK': numpyHkChars },
    load: {
      en: () => import('../content/notes/numpy.en.md?html'),
      zh: () => import('../content/notes/numpy.zh.md?html'),
      'zh-HK': () => import('../content/notes/numpy.zh.md?html-hk'),
    },
  },
  {
    id: 'pandas',
    titleKey: 'package.notes.pandas',
    tag: 'pandas',
    tagClass: 'bg-indigo-100 text-indigo-700',
    chars: { en: pandasEnChars, zh: pandasZhChars, 'zh-HK': pandasHkChars },
    load: {
      en: () => import('../content/notes/pandas.en.md?html'),
      zh: () => import('../content/notes/pandas.zh.md?html'),
      'zh-HK': () => import('../content/notes/pandas.zh.md?html-hk'),
    },
  },
  {
    id: 'pytorch',
    titleKey: 'package.notes.pytorch',
    tag: 'PyTorch',
    tagClass: 'bg-orange-100 text-orange-700',
    chars: { en: pytorchEnChars, zh: pytorchZhChars, 'zh-HK': pytorchHkChars },
    load: {
      en: () => import('../content/notes/pytorch.en.md?html'),
      zh: () => import('../content/notes/pytorch.zh.md?html'),
      'zh-HK': () => import('../content/notes/pytorch.zh.md?html-hk'),
    },
  },
  {
    id: 'tensorflow',
    titleKey: 'package.notes.tensorflow',
    tag: 'TensorFlow',
    tagClass: 'bg-amber-100 text-amber-700',
    chars: { en: tensorflowEnChars, zh: tensorflowZhChars, 'zh-HK': tensorflowHkChars },
    load: {
      en: () => import('../content/notes/tensorflow.en.md?html'),
      zh: () => import('../content/notes/tensorflow.zh.md?html'),
      'zh-HK': () => import('../content/notes/tensorflow.zh.md?html-hk'),
    },
  },
  {
    id: 'keras',
    titleKey: 'package.notes.keras',
    tag: 'Keras',
    tagClass: 'bg-rose-100 text-rose-700',
    chars: { en: kerasEnChars, zh: kerasZhChars, 'zh-HK': kerasHkChars },
    load: {
      en: () => import('../content/notes/keras.en.md?html'),
      zh: () => import('../content/notes/keras.zh.md?html'),
      'zh-HK': () => import('../content/notes/keras.zh.md?html-hk'),
    },
  },
  {
    id: 'kevin',
    titleKey: 'package.notes.kevin',
    tag: 'COMP2211',
    tagClass: 'bg-emerald-100 text-emerald-700',
    chars: { en: kevinEnChars, zh: kevinZhChars, 'zh-HK': kevinHkChars },
    load: {
      en: () => import('../content/notes/kevin.en.md?html'),
      zh: () => import('../content/notes/kevin.zh.md?html'),
      'zh-HK': () => import('../content/notes/kevin.zh.md?html-hk'),
    },
  },
]
