/**
 * The 2-2-1 network of the lecture 6 XOR example, with the course's notation:
 *   h1 = f(w1·x1 + w2·x2 + θ1), h2 = f(w3·x1 + w4·x2 + θ2), O = f(w5·h1 + w6·h2 + θ3)
 *   δk = (O − T)·O(1 − O),  δj = Oj(1 − Oj)·δk·wjk,  w ← w − η·δ·(input of that weight)
 * Online learning: one sample at a time, hidden deltas use the weights before this sample's update.
 */

export const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z))
export const step = (z: number): number => (z > 0 ? 1 : 0)

export interface Net {
  w: [number, number, number, number, number, number] // w1 … w6
  theta: [number, number, number] // θ1, θ2 (hidden), θ3 (output)
}

export type Act = 'sigmoid' | 'step'

export interface Forward {
  x: [number, number]
  sums: [number, number, number] // Σ1, Σ2, Σ3 without θ
  h: [number, number]
  o: number
}

export function forward(net: Net, x: [number, number], act: Act = 'sigmoid'): Forward {
  const f = act === 'sigmoid' ? sigmoid : step
  const { w, theta } = net
  const s1 = x[0] * w[0] + x[1] * w[1]
  const s2 = x[0] * w[2] + x[1] * w[3]
  const h: [number, number] = [f(s1 + theta[0]), f(s2 + theta[1])]
  const s3 = h[0] * w[4] + h[1] * w[5]
  return { x, sums: [s1, s2, s3], h, o: f(s3 + theta[2]) }
}

export interface BackStep {
  fwd: Forward
  t: number
  deltaK: number
  deltaJ: [number, number]
  before: Net
  after: Net
}

export function backprop(net: Net, x: [number, number], t: number, eta: number): BackStep {
  const fwd = forward(net, x)
  const { h, o } = fwd
  const { w, theta } = net
  const deltaK = (o - t) * o * (1 - o)
  const deltaJ: [number, number] = [h[0] * (1 - h[0]) * deltaK * w[4], h[1] * (1 - h[1]) * deltaK * w[5]]
  const after: Net = {
    w: [
      w[0] - eta * deltaJ[0] * x[0],
      w[1] - eta * deltaJ[0] * x[1],
      w[2] - eta * deltaJ[1] * x[0],
      w[3] - eta * deltaJ[1] * x[1],
      w[4] - eta * deltaK * h[0],
      w[5] - eta * deltaK * h[1],
    ],
    theta: [theta[0] - eta * deltaJ[0], theta[1] - eta * deltaJ[1], theta[2] - eta * deltaK],
  }
  return { fwd, t, deltaK, deltaJ, before: net, after }
}

export const XOR: { x: [number, number]; t: number }[] = [
  { x: [0, 0], t: 0 },
  { x: [0, 1], t: 1 },
  { x: [1, 0], t: 1 },
  { x: [1, 1], t: 0 },
]

/** Train for whole epochs over `data` in order. */
export function trainEpochs(net: Net, epochs: number, eta: number, data = XOR): Net {
  let cur = net
  for (let e = 0; e < epochs; e++) for (const s of data) cur = backprop(cur, s.x, s.t, eta).after
  return cur
}

/** Mean squared error ½(T − O)² averaged over the data. */
export const loss = (net: Net, data = XOR, act: Act = 'sigmoid'): number =>
  data.reduce((a, s) => a + 0.5 * (s.t - forward(net, s.x, act).o) ** 2, 0) / data.length

export const LECTURE_NET: Net = { w: [-0.65, 0.64, 1.11, 0.84, 0.86, -1.38], theta: [0, 0, 0] }
