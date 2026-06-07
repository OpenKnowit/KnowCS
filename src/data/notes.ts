// --- Package 资料包：打包进构建的 Markdown 笔记（?raw 构建期内联，兼容单文件打包） ---
import kerasMd from '../content/notes/keres.md?raw'
import kevinMd from '../content/notes/KevinelwNote.md?raw'
import numpyMd from '../content/notes/numpy.md?raw'
import pandasMd from '../content/notes/pandas.md?raw'
import pytorchMd from '../content/notes/pytorch.md?raw'
import tensorflowMd from '../content/notes/tensorflow.md?raw'

export interface NoteEntry {
  id: string
  titleKey: string // i18n 键（package.notes.<id>）
  tag: string
  tagClass: string // 卡片标签配色
  content: string
}

export const NOTES: NoteEntry[] = [
  { id: 'numpy', titleKey: 'package.notes.numpy', tag: 'NumPy', tagClass: 'bg-sky-100 text-sky-700', content: numpyMd },
  { id: 'pandas', titleKey: 'package.notes.pandas', tag: 'pandas', tagClass: 'bg-indigo-100 text-indigo-700', content: pandasMd },
  { id: 'pytorch', titleKey: 'package.notes.pytorch', tag: 'PyTorch', tagClass: 'bg-orange-100 text-orange-700', content: pytorchMd },
  { id: 'tensorflow', titleKey: 'package.notes.tensorflow', tag: 'TensorFlow', tagClass: 'bg-amber-100 text-amber-700', content: tensorflowMd },
  { id: 'keras', titleKey: 'package.notes.keras', tag: 'Keras', tagClass: 'bg-rose-100 text-rose-700', content: kerasMd },
  { id: 'kevin', titleKey: 'package.notes.kevin', tag: 'COMP2211', tagClass: 'bg-emerald-100 text-emerald-700', content: kevinMd },
]

// 笔记内 images/xxx.png 相对引用 → 构建产物 URL（单文件打包时为 data URI）
const imageModules = import.meta.glob('../content/notes/images/*', {
  eager: true,
  query: '?url',
  import: 'default',
}) as Record<string, string>

export const NOTE_IMAGES: Record<string, string> = Object.fromEntries(
  Object.entries(imageModules).map(([path, url]) => [path.replace('../content/notes/', ''), url])
)
