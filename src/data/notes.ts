// --- Package 资料包：构建期预渲染为 HTML 的 Markdown 笔记（?html，见 scripts/vite-plugins.mjs） ---
import kerasNote from '../content/notes/keres.md?html'
import kevinNote from '../content/notes/KevinelwNote.md?html'
import numpyNote from '../content/notes/numpy.md?html'
import pandasNote from '../content/notes/pandas.md?html'
import pytorchNote from '../content/notes/pytorch.md?html'
import tensorflowNote from '../content/notes/tensorflow.md?html'

export interface NoteEntry {
  id: string
  titleKey: string // i18n 键（package.notes.<id>）
  tag: string
  tagClass: string // 卡片标签配色
  html: string // 预渲染 HTML（图片已内联）
  chars: number // 原始 Markdown 字符数
}

export const NOTES: NoteEntry[] = [
  { id: 'numpy', titleKey: 'package.notes.numpy', tag: 'NumPy', tagClass: 'bg-sky-100 text-sky-700', ...numpyNote },
  { id: 'pandas', titleKey: 'package.notes.pandas', tag: 'pandas', tagClass: 'bg-indigo-100 text-indigo-700', ...pandasNote },
  { id: 'pytorch', titleKey: 'package.notes.pytorch', tag: 'PyTorch', tagClass: 'bg-orange-100 text-orange-700', ...pytorchNote },
  { id: 'tensorflow', titleKey: 'package.notes.tensorflow', tag: 'TensorFlow', tagClass: 'bg-amber-100 text-amber-700', ...tensorflowNote },
  { id: 'keras', titleKey: 'package.notes.keras', tag: 'Keras', tagClass: 'bg-rose-100 text-rose-700', ...kerasNote },
  { id: 'kevin', titleKey: 'package.notes.kevin', tag: 'COMP2211', tagClass: 'bg-emerald-100 text-emerald-700', ...kevinNote },
]
