// The broadcasting lab's demo data: example pairs, the values shown in the grids, the generated program, the quiz.
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

// ---- the program: the shapes on the left written as code, with each length-1 axis made in the chosen spelling

/** the four ways to add a length-1 axis */
export type Syntax = 'none' | 'newaxis' | 'expand' | 'reshape'
export const SYNTAXES: { id: Syntax; label: string }[] = [
  { id: 'none', label: 'None' },
  { id: 'newaxis', label: 'np.newaxis' },
  { id: 'expand', label: 'expand_dims' },
  { id: 'reshape', label: 'reshape' },
]

/** the axes of `to` that have to be inserted: the 1s, except one kept as the data when every axis is 1 */
const insertedAxes = (to: number[]): number[] => {
  const ones = to.flatMap((d, i) => (d === 1 ? [i] : []))
  return ones.length === to.length ? ones.slice(0, -1) : ones
}
export const withoutOnes = (to: number[]): number[] => {
  const gone = new Set(insertedAxes(to))
  return to.filter((_, i) => !gone.has(i))
}

/** the expression that turns `v` (shaped withoutOnes(to)) into shape `to`; null when no axis is needed */
export const constructCode = (v: string, to: number[], syntax: Syntax): string | null => {
  const axes = insertedAxes(to)
  if (!axes.length) return null
  if (syntax === 'reshape') return `${v}.reshape(${to.join(', ')})`
  if (syntax === 'expand') return axes.reduce((acc, axis) => `np.expand_dims(${acc}, axis=${axis})`, v)
  const token = syntax === 'none' ? 'None' : 'np.newaxis'
  return `${v}[${to.map((_, i) => (axes.includes(i) ? token : ':')).join(', ')}]`
}

const declare = (s: number[], isB: boolean) => {
  if (s.length === 0) return isB ? '10' : '0'
  const range = isB ? `np.arange(1, ${size(s) + 1})` : `np.arange(${size(s)})`
  return `${s.length === 1 ? range : `${range}.reshape(${s.join(', ')})`}${isB ? ' * 10' : ''}`
}

/** a line of the program; steps = the visual steps it belongs to (see BROADCAST_STEPS) */
export interface ProgramLine {
  text: string
  steps: number[]
}
export interface Construct {
  name: string
  from: number[]
  to: number[]
  code: string
}
export interface Program {
  lines: ProgramLine[]
  code: string
  constructs: Construct[]
}
export const BROADCAST_STEPS = ['input', 'target', 'expand', 'result'] as const

export const makeProgram = (a: number[], b: number[], op: Op, syntax: Syntax): Program => {
  const lines: ProgramLine[] = [{ text: 'import numpy as np', steps: [] }]
  const constructs: Construct[] = []
  const operand = (name: 'A' | 'B', low: string, shape: number[]) => {
    const made = constructCode(low, shape, syntax)
    if (!made) {
      lines.push({ text: `${name} = ${declare(shape, name === 'B')}`, steps: [0] })
      return
    }
    const from = withoutOnes(shape)
    lines.push({ text: `${low} = ${declare(from, name === 'B')}`, steps: [0] })
    lines.push({ text: `${name} = ${made}  # ${shapeLabel(from)} → ${shapeLabel(shape)}`, steps: [1] })
    constructs.push({ name, from, to: shape, code: made })
  }
  operand('A', 'a', a)
  operand('B', 'b', b)
  const plan = planBroadcast(a, b)
  const note = plan.ok ? `${shapeLabel(a)} ${op} ${shapeLabel(b)} → ${shapeLabel(plan.outShape!)}` : `${shapeLabel(a)} ${op} ${shapeLabel(b)} → ValueError`
  lines.push({ text: `C = A ${op} B  # ${note}`, steps: !plan.ok || constructs.length === 0 ? [1, 2, 3] : [2, 3] })
  lines.push({ text: 'print(np.shape(C))', steps: [3] })
  lines.push({ text: 'print(C)', steps: [3] })
  return { lines, code: lines.map((l) => l.text).join('\n'), constructs }
}

/** the course's classic: every test point against every training point. Line 4 onwards is traced step by step */
export const makePairwise = (syntax: Syntax): Program => {
  const test = constructCode('X_test', [2, 1, 2], syntax)!
  const train = constructCode('X_train', [1, 4, 2], syntax)!
  const text = [
    'import numpy as np',
    'X_train = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])',
    'X_test = np.array([[5, 6], [7, 8]])',
    `A = ${test}`,
    `B = ${train}`,
    'diff = A - B',
    'sq = diff ** 2',
    'd2 = sq.sum(axis=-1)',
    'D = np.sqrt(d2)',
    'nearest = np.argsort(D, axis=1)[:, :2]',
    'print(D)',
    'print(nearest)',
  ]
  return {
    lines: text.map((t) => ({ text: t, steps: [] })),
    code: text.join('\n'),
    constructs: [
      { name: 'X_test', from: [2, 2], to: [2, 1, 2], code: test },
      { name: 'X_train', from: [4, 2], to: [1, 4, 2], code: train },
    ],
  }
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

/** past-paper questions, loadable with their own spelling. The papers' / has the same shape rule as × here */
export const EXAM_EXAMPLES: { id: string; a: number[]; b: number[]; op: Op; syntax?: Syntax; program?: boolean }[] = [
  { id: 'pairwise', a: [2, 1, 2], b: [1, 4, 2], op: '-', syntax: 'none', program: true },
  { id: 'e23s_i', a: [4, 2], b: [2], op: '*' },
  { id: 'e23s_ii', a: [4, 3], b: [4], op: '+' },
  { id: 'e23s_iii', a: [4, 1], b: [3], op: '+', syntax: 'newaxis' },
  { id: 'e24s_x', a: [3, 2, 4], b: [3, 2], op: '*' },
  { id: 'e24s_xi', a: [3, 2, 4], b: [3, 2, 1], op: '*', syntax: 'newaxis' },
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
