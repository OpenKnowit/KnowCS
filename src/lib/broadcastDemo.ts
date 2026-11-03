// The broadcasting lab's demo data: example pairs, the values shown in the grids, the equivalent code, the quiz.
import { planBroadcast, shapeLabel } from './broadcast'

export type Op = '+' | '-' | '*'

export interface BroadcastPreset {
  id: string
  a: number[]
  b: number[]
  op: Op
}

export const PRESETS: BroadcastPreset[] = [
  { id: 'scalar', a: [3, 4], b: [], op: '*' },
  { id: 'row', a: [3, 4], b: [4], op: '-' },
  { id: 'outer', a: [3, 1], b: [1, 4], op: '+' },
  { id: 'cube', a: [2, 3, 4], b: [3, 1], op: '+' },
  { id: 'mismatch', a: [3, 4], b: [3], op: '+' },
  { id: 'silent', a: [3], b: [3, 1], op: '+' },
]

export const MAX_NDIM = 3
export const MAX_DIM = 5

export const size = (shape: number[]) => shape.reduce((p, x) => p * x, 1)
const flatOf = (idx: number[], shape: number[]) => idx.reduce((f, v, k) => f * shape[k] + v, 0)
// A = arange, B = (arange + 1) * 10: each result cell shows at a glance what both sides gave
export const valA = (idx: number[], shape: number[]) => flatOf(idx, shape)
export const valB = (idx: number[], shape: number[]) => (flatOf(idx, shape) + 1) * 10
export const apply = (op: Op, x: number, y: number) => (op === '+' ? x + y : op === '-' ? x - y : x * y)
export const opSign = (op: Op) => (op === '*' ? '×' : op === '-' ? '−' : '+')

const pyShape = (s: number[]) => (s.length === 1 ? `${s[0]}` : s.join(', '))
export const makeCode = (a: number[], b: number[], op: Op) => {
  const decl = (name: string, s: number[], isB: boolean) => {
    if (s.length === 0) return `${name} = ${isB ? 10 : 0}`
    const range = isB ? `np.arange(1, ${size(s) + 1})` : `np.arange(${size(s)})`
    const base = s.length === 1 ? range : `${range}.reshape(${pyShape(s)})`
    return `${name} = ${base}${isB ? ' * 10' : ''}`
  }
  return `import numpy as np\n${decl('A', a, false)}\n${decl('B', b, true)}\nA ${op} B`
}

export const TONES = {
  A: { real: 'bg-blue-500 border-blue-600 text-white', ghost: 'bg-blue-50 border-blue-300 text-blue-400 border-dashed', text: 'text-blue-600', chip: 'bg-blue-50 border-blue-200' },
  B: { real: 'bg-orange-500 border-orange-600 text-white', ghost: 'bg-orange-50 border-orange-300 text-orange-400 border-dashed', text: 'text-orange-600', chip: 'bg-orange-50 border-orange-200' },
  R: { real: 'bg-emerald-500 border-emerald-600 text-white', ghost: '', text: 'text-emerald-600', chip: 'bg-emerald-50 border-emerald-200' },
}

/** where broadcasting shows up in the course, each loadable into the lab with its real shapes (small sizes) */
export const ML_USES: { id: string; code: string; shapes: string; a: number[]; b: number[]; op: Op }[] = [
  { id: 'std', code: 'X - X.mean(axis=0)', shapes: '(n, d) − (d,)', a: [4, 3], b: [3], op: '-' },
  { id: 'bias', code: 'X @ W + b', shapes: '(n, k) + (k,)', a: [4, 2], b: [2], op: '+' },
  { id: 'knn', code: 'X[:, None, :] - Y[None, :, :]', shapes: '(n, 1, d) − (1, m, d) → (n, m, d)', a: [2, 1, 2], b: [1, 3, 2], op: '-' },
  { id: 'gray', code: '(img * [0.299, 0.587, 0.114]).sum(axis=2)', shapes: '(H, W, 3) × (3,) → (H, W, 3)', a: [2, 2, 3], b: [3], op: '*' },
]

/** the exam question: two shapes in, the result shape (or an error) out */
export const QUIZ: [number[], number[]][] = [
  [[3, 4], [4]],
  [[3, 4], [3]],
  [[5, 1], [1, 6]],
  [[2, 3, 4], [3, 1]],
  [[3], [3, 1]],
  [[4, 1, 2], [3, 2]],
]
export const quizAnswer = (a: number[], b: number[]): string => {
  const p = planBroadcast(a, b)
  return p.ok ? shapeLabel(p.outShape!) : 'error'
}
