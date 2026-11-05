// The broadcasting lab's demo data: example pairs, the values shown in the grids, the generated program, the quiz.
import { planBroadcast, shapeLabel } from './broadcast'

export type Op = '+' | '-' | '*' | '/'

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
export const flatOf = (idx: number[], shape: number[]) => idx.reduce((f, v, k) => f * shape[k] + v, 0)
// A = arange, B = (arange + 1) * 10: each result cell shows at a glance what both sides gave
export const valA = (idx: number[], shape: number[]) => flatOf(idx, shape)
export const valB = (idx: number[], shape: number[]) => (flatOf(idx, shape) + 1) * 10
export const apply = (op: Op, x: number, y: number) => (op === '+' ? x + y : op === '-' ? x - y : op === '*' ? x * y : x / y)
export const opSign = (op: Op) => (op === '*' ? '×' : op === '-' ? '−' : op === '/' ? '÷' : '+')
/** a number short enough for a grid cell */
export const showCell = (v: number) => (Number.isInteger(v) ? String(v) : String(Number(v.toFixed(2))))

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

/** one side of a quoted expression: what to call it, the values the source gives it, and any axis the source adds */
export interface QuotedOperand {
  name: string
  /** the variable the axis goes on and what multiplies it, when this side is more than a variable */
  variable?: string
  prefix?: string
  shape: number[]
  values: number[]
  to?: number[]
  /** the source's own spelling of that added axis */
  paper?: string
}
/** code quoted word for word from a paper, the slides or a lab: the code above the expression, then its line */
export interface QuotedCode {
  id: string
  setup: string
  line: string
  /** the line again with {} for the expression, when a spelling or a fix rewrites it; print({}) if left out */
  rewrite?: string
  /** code after the expression's line */
  after?: string
  op: Op
  l: QuotedOperand
  r: QuotedOperand
  syntax?: Syntax
}
const range = (n: number, from = 0) => Array.from({ length: n }, (_, i) => from + i)
const SETUP_24S = `import numpy as np
# np.arange(start, stop)
# - return an array of evenly spaced values within the half-open interval
#   [start,stop), the default step size is 1.
# np.ones(shape):
# - return a new array of given shape, filled with ones.
A = np.arange(5,15)
B = np.ones((3,2))
C = np.array([[[0,1,2,3],
               [4,5,6,7]],
              [[8,9,10,11],
               [12,13,14,15]],
              [[16,17,18,19],
               [20,21,22,23]]])`
const C_24S: QuotedOperand = { name: 'C', shape: [3, 2, 4], values: range(24) }
const B_24S: QuotedOperand = { name: 'B', shape: [3, 2], values: new Array(6).fill(1) }

export const EXAM_QUESTIONS: QuotedCode[] = [
  {
    id: 'e23s_i',
    setup: `import numpy as np
A = np.array([1,2])
B = np.array([[1,2],
              [2,4],
              [3,6],
              [4,8]])`,
    line: 'print(B/A)',
    op: '/',
    l: { name: 'B', shape: [4, 2], values: [1, 2, 2, 4, 3, 6, 4, 8] },
    r: { name: 'A', shape: [2], values: [1, 2] },
  },
  {
    id: 'e23s_ii',
    setup: `import numpy as np
A = np.array([[0, 0, 0],
              [1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
B = np.array([10, 11, 12, 13])`,
    line: 'print(A + B)',
    op: '+',
    l: { name: 'A', shape: [4, 3], values: [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] },
    r: { name: 'B', shape: [4], values: [10, 11, 12, 13] },
  },
  {
    id: 'e23s_iii',
    setup: `import numpy as np
A = np.array([0, 10, 20, 30])
B = np.array([1, 2, 3])
# np.newaxis
# increase the dimension of an array by adding new axis`,
    line: 'print(A[:, np.newaxis] + B)',
    op: '+',
    l: { name: 'A', shape: [4], values: [0, 10, 20, 30], to: [4, 1], paper: 'A[:, np.newaxis]' },
    r: { name: 'B', shape: [3], values: [1, 2, 3] },
    syntax: 'newaxis',
  },
  { id: 'e24s_x', setup: SETUP_24S, line: 'print( C / B )', op: '/', l: C_24S, r: B_24S },
  {
    id: 'e24s_xi',
    setup: `${SETUP_24S}
# np.newaxis
# - increase the dimension of an array by adding new axis.`,
    line: 'print(C / B[..., np.newaxis])',
    op: '/',
    l: C_24S,
    r: { ...B_24S, to: [3, 2, 1], paper: 'B[..., np.newaxis]' },
    syntax: 'newaxis',
  },
]

/** the slides and labs of the course. Page numbers: 2-advanced-python.pdf, COMP2211 Fall 2025 */
const IMG = [200, 100, 0, 0, 100, 200, 50, 50, 50, 255, 255, 255]
const LUMINANCE = [0.299, 0.587, 0.114]
const GREY = [0, 3, 6, 9].map((i) => IMG[i] * LUMINANCE[0] + IMG[i + 1] * LUMINANCE[1] + IMG[i + 2] * LUMINANCE[2])

export const COURSE_CODE: QuotedCode[] = [
  {
    id: 'c_xv',
    setup: `import numpy as np
# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])`,
    line: 'y = x + v  # Add v to each row of x using broadcasting',
    rewrite: 'y = {}',
    after: 'print(y)',
    op: '+',
    l: { name: 'x', shape: [4, 3], values: range(12, 1) },
    r: { name: 'v', shape: [3], values: [1, 0, 1] },
  },
  {
    id: 'c_rank',
    setup: `import numpy as np
A = np.array([1, 2, 3])
print(A.ndim)  # Print 1
print(A.shape) # Print (3,)
B = np.array([[4, 4, 4],
              [3, 3, 3]])
print(B.ndim)  # Print 2
print(B.shape) # Print (2,3)`,
    line: 'print(A*B)',
    op: '*',
    l: { name: 'A', shape: [3], values: [1, 2, 3] },
    r: { name: 'B', shape: [2, 3], values: [4, 4, 4, 3, 3, 3] },
  },
  {
    id: 'c_scores',
    setup: `import numpy as np
scores = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
score_mean = scores.mean(0)  # 0 here means the first dimension
print(score_mean)`,
    line: 'scores_centered = scores - score_mean',
    rewrite: 'scores_centered = {}',
    after: 'print(scores_centered)',
    op: '-',
    l: { name: 'scores', shape: [4, 3], values: range(12, 1) },
    r: { name: 'score_mean', shape: [3], values: [5.5, 6.5, 7.5] },
  },
  {
    id: 'c_lab1_open',
    setup: `import numpy as np
a = np.arange(4) * 10
b = np.arange(1, 4)`,
    line: 'a[:, None] + b',
    rewrite: '{}',
    op: '+',
    l: { name: 'a', shape: [4], values: [0, 10, 20, 30], to: [4, 1], paper: 'a[:, None]' },
    r: { name: 'b', shape: [3], values: [1, 2, 3] },
    syntax: 'none',
  },
  {
    id: 'c_lab1_ex',
    setup: `import numpy as np
a = np.arange(3)     # [0, 1, 2], shape (3,)
b = np.array([1, 1, 1, 1, 1])   # shape (5,)`,
    line: 'a.reshape(1, 3) * b.reshape(5, 1)',
    rewrite: '{}',
    op: '*',
    l: { name: 'a', shape: [3], values: [0, 1, 2], to: [1, 3], paper: 'a.reshape(1, 3)' },
    r: { name: 'b', shape: [5], values: [1, 1, 1, 1, 1], to: [5, 1], paper: 'b.reshape(5, 1)' },
    syntax: 'reshape',
  },
  {
    id: 'c_lab1_contrast',
    setup: `import numpy as np
# The lab reads a photo into img; here a 2×2 picture, so every value fits on screen
img = np.array([[[200, 100, 0], [0, 100, 200]],
                [[50, 50, 50], [255, 255, 255]]])
luminance = np.array([0.299, 0.587, 0.114]) # Luminance coefficients
grey_img = img @ luminance
alpha = 0.5`,
    line: 'new_img = alpha * grey_img[..., np.newaxis] + (1 - alpha) * img',
    rewrite: 'new_img = {}',
    after: 'print(new_img)',
    op: '+',
    l: { name: 'alpha * grey_img', variable: 'grey_img', prefix: 'alpha * ', shape: [2, 2], values: GREY.map((g) => 0.5 * g), to: [2, 2, 1], paper: 'grey_img[..., np.newaxis]' },
    r: { name: '(1 - alpha) * img', variable: 'img', prefix: '(1 - alpha) * ', shape: [2, 2, 3], values: IMG.map((v) => 0.5 * v) },
    syntax: 'newaxis',
  },
]

/** a fix the learner applied to code that cannot broadcast: this side gets this shape */
export type QuoteFix = { side: 'l' | 'r'; shape: number[] } | null
export const quotedShape = (o: QuotedOperand, side: 'l' | 'r', fix: QuoteFix) => (fix?.side === side ? fix.shape : (o.to ?? o.shape))

/** the source's code as printed; a spelling, an operator or a fix the learner changes rewrites only the expression's line */
export const makeQuoted = (q: QuotedCode, syntax: Syntax, op: Op, fix: QuoteFix): Program & { expr: string } => {
  const a = quotedShape(q.l, 'l', fix)
  const b = quotedShape(q.r, 'r', fix)
  const asPrinted = !fix && op === q.op && (!q.syntax || syntax === q.syntax)
  const axis = (o: QuotedOperand, to: number[]) => {
    const made = constructCode(o.variable ?? o.name, to, syntax)
    return made && o.paper && syntax === q.syntax && !fix ? o.paper : made
  }
  const side = (o: QuotedOperand, to: number[]) => {
    const e = `${o.prefix ?? ''}${axis(o, to) ?? o.variable ?? o.name}`
    return o.prefix && (op === '*' || op === '/') ? `(${e})` : e
  }
  const expr = `${side(q.l, a)} ${op} ${side(q.r, b)}`
  const plan = planBroadcast(a, b)
  const lines: ProgramLine[] = q.setup.split('\n').map((text) => ({ text, steps: text.startsWith('#') || text.startsWith('import') ? [] : [0] }))
  lines.push({ text: asPrinted ? q.line : (q.rewrite ?? 'print({})').replace('{}', expr), steps: plan.ok ? [1, 2, 3] : [1] })
  if (q.after) lines.push({ text: q.after, steps: plan.ok ? [3] : [1] })
  const constructs = ([[q.l, a], [q.r, b]] as const).flatMap(([o, to]) => {
    const code = axis(o, to)
    return code ? [{ name: o.variable ?? o.name, from: o.shape, to, code }] : []
  })
  return { lines, code: lines.map((l) => l.text).join('\n'), constructs, expr }
}

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
