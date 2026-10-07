/**
 * Small, exact models of the lecture 9 PyTorch behaviours the practice page demonstrates:
 * tensor attributes, @ versus *, memory shared with NumPy, scalar autograd with accumulation, and the training loop order.
 */
import { sigmoid } from './autogradDemo'

// ---------------------------------------------------------------- tensor attributes
export interface TensorInfo {
  code: string
  shape: number[]
  dtype: 'torch.int64' | 'torch.float32' | 'torch.float64' | 'torch.bool'
}

/** The constructors the lecture uses, with what print(t.shape / t.dtype) shows. */
export const TENSORS: TensorInfo[] = [
  { code: 'torch.tensor([1, 3, 5])', shape: [3], dtype: 'torch.int64' },
  { code: 'torch.tensor([[1., 2.], [3., 4.]])', shape: [2, 2], dtype: 'torch.float32' },
  { code: 'torch.ones(4, 4)', shape: [4, 4], dtype: 'torch.float32' },
  { code: 'torch.rand(2, 3)', shape: [2, 3], dtype: 'torch.float32' },
  { code: 'torch.from_numpy(np.ones(5))', shape: [5], dtype: 'torch.float64' },
  { code: 'torch.ones(4, 4).sum()', shape: [], dtype: 'torch.float32' },
  { code: 'torch.tensor([1, 3, 5]) > 2', shape: [3], dtype: 'torch.bool' },
]

export const sizeStr = (shape: number[]) => `torch.Size([${shape.join(', ')}])`

// ---------------------------------------------------------------- @ versus *
export const matmul = (a: number[][], b: number[][]): number[][] => a.map((row) => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)))
export const mul = (a: number[][], b: number[][]): number[][] => a.map((row, i) => row.map((v, j) => v * b[i][j]))
export const transpose = (a: number[][]): number[][] => a[0].map((_, j) => a.map((row) => row[j]))

/** tensor = torch.ones(n, n); tensor[:, col] = 0 — the lecture's example. */
export const lectureTensor = (n = 4, col = 1): number[][] => Array.from({ length: n }, () => Array.from({ length: n }, (_, j) => (j === col ? 0 : 1)))

// ---------------------------------------------------------------- NumPy bridge
export type BridgeOp = 'add_' | 'np_add_out' | 'plus' | 'clone'

/**
 * my_tensor = torch.ones(5); np_array = my_tensor.numpy(), then a sequence of operations.
 * In-place ops (add_, np.add(..., out=)) write to the shared buffer; x = x + 1 and .clone() make new memory.
 */
export function bridge(ops: BridgeOp[]): { tensor: number[]; array: number[]; shared: boolean } {
  let tensor = Array(5).fill(1) as number[]
  let array = tensor
  let shared = true
  for (const op of ops) {
    if (op === 'add_') tensor = shared ? (array = tensor.map((v) => v + 1)) : tensor.map((v) => v + 1)
    else if (op === 'np_add_out') array = shared ? (tensor = array.map((v) => v + 1)) : array.map((v) => v + 1)
    else if (op === 'plus') {
      tensor = tensor.map((v) => v + 1)
      shared = false
    } else {
      tensor = [...tensor]
      shared = false
    }
  }
  return { tensor, array, shared }
}

// ---------------------------------------------------------------- scalar autograd: z = w·x + b, loss = BCE(σ(z), y)
export interface Graph {
  z: number
  p: number
  loss: number
  /** ∂loss/∂z = σ(z) − y */
  dz: number
  dw: number
  db: number
}

export function graph(w: number, x: number, b: number, y: number): Graph {
  const z = w * x + b
  const p = sigmoid(z)
  const loss = Math.max(z, 0) - z * y + Math.log(1 + Math.exp(-Math.abs(z)))
  const dz = p - y
  return { z, p, loss, dz, dw: dz * x, db: dz }
}

export type GradEvent = 'backward' | 'zero_grad' | 'step'

/**
 * What w.grad / b.grad hold after a sequence of calls on the same graph: backward() adds, zero_grad() clears,
 * step() moves w and b by −lr·grad (and the graph is rebuilt from the new values).
 */
export function replay(start: { w: number; b: number }, x: number, y: number, lr: number, events: GradEvent[]): { w: number; b: number; wGrad: number; bGrad: number } {
  let { w, b } = start
  let wGrad = 0
  let bGrad = 0
  for (const e of events) {
    if (e === 'backward') {
      const g = graph(w, x, b, y)
      wGrad += g.dw
      bGrad += g.db
    } else if (e === 'zero_grad') {
      wGrad = 0
      bGrad = 0
    } else {
      w -= lr * wGrad
      b -= lr * bGrad
    }
  }
  return { w, b, wGrad, bGrad }
}

// ---------------------------------------------------------------- the training loop
export const LOOP = ['zero_grad', 'forward', 'loss', 'backward', 'step'] as const
export type LoopLine = (typeof LOOP)[number]

export const LOOP_CODE: Record<LoopLine, string> = {
  zero_grad: 'optimizer.zero_grad()',
  forward: 'outputs = model(inputs)',
  loss: 'loss = lossfunc(outputs, labels)',
  backward: 'loss.backward()',
  step: 'optimizer.step()',
}

/** Why an order is wrong, or null when it is the lecture's order. Translated under lab.pytorch.loop_err.<code>. */
export function loopError(order: LoopLine[]): 'loss_first' | 'backward_first' | 'step_early' | 'zero_late' | null {
  const at = (l: LoopLine) => order.indexOf(l)
  if (at('loss') < at('forward')) return 'loss_first'
  if (at('backward') < at('loss')) return 'backward_first'
  if (at('step') < at('backward')) return 'step_early'
  if (at('zero_grad') > at('backward') && at('zero_grad') < at('step')) return 'zero_late'
  return null
}
