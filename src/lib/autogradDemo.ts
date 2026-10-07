/**
 * Lecture 9's autograd example, z = x·w + b with binary cross-entropy (with logits) against y, computed
 * exactly so the explainer can show forward values and the gradients loss.backward() would produce.
 */
import { mulberry32 } from './crossval'

export const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z))

/** Mean BCE-with-logits over the outputs. */
export const bceWithLogits = (z: number[], y: number[]): number =>
  z.reduce((s, zj, j) => s + (Math.max(zj, 0) - zj * y[j] + Math.log(1 + Math.exp(-Math.abs(zj)))), 0) / z.length

export interface AutogradExample {
  x: number[]
  y: number[]
  w: number[][] // shape (5, 3)
  b: number[]
  z: number[]
  loss: number
  /** ∂loss/∂z_j = (σ(z_j) − y_j) / n_out */
  dz: number[]
  wGrad: number[][]
  bGrad: number[]
}

/** w and b drawn from a seeded normal (like torch.randn), so the frames are reproducible. */
export function lectureExample(seed = 2211): AutogradExample {
  const r = mulberry32(seed)
  const randn = () => Math.sqrt(-2 * Math.log(Math.max(1e-9, r()))) * Math.cos(2 * Math.PI * r())
  const round = (v: number) => Math.round(v * 100) / 100
  const x = [1, 1, 1, 1, 1]
  const y = [0, 0, 0]
  const w = Array.from({ length: 5 }, () => Array.from({ length: 3 }, () => round(randn())))
  const b = Array.from({ length: 3 }, () => round(randn()))
  return forwardBackward(x, y, w, b)
}

export function forwardBackward(x: number[], y: number[], w: number[][], b: number[]): AutogradExample {
  const z = b.map((bj, j) => x.reduce((s, xi, i) => s + xi * w[i][j], bj))
  const loss = bceWithLogits(z, y)
  const dz = z.map((zj, j) => (sigmoid(zj) - y[j]) / z.length)
  const wGrad = x.map((xi) => dz.map((d) => xi * d))
  return { x, y, w, b, z, loss, dz, wGrad, bGrad: [...dz] }
}
