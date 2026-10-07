// The hand-picked 2-2-1 network used by the XOR explainer (/watch/xor/).
//   h1 = σ(3(1.5 x1 + 1.0 x2 − 0.5))  ≈ OR
//   h2 = σ(3(1.0 x1 + 1.5 x2 − 2.0))  ≈ AND
//   y  = step(h1 − h2 − 0.4)           = OR and not AND = XOR
// W is non-singular, so the linear step stretches the plane instead of flattening it.

export type Vec2 = readonly [number, number]

export const XOR_POINTS: { x: Vec2; t: 0 | 1 }[] = [
  { x: [0, 0], t: 0 },
  { x: [1, 0], t: 1 },
  { x: [0, 1], t: 1 },
  { x: [1, 1], t: 0 },
]

export const GAIN = 3
export const W: readonly [Vec2, Vec2] = [
  [1.5, 1],
  [1, 1.5],
]
export const C: Vec2 = [-0.5, -2]
export const OUT = { w: [1, -1] as Vec2, theta: -0.4 }

export const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z))

/** z = GAIN · (W x + c) */
export const linear = (x: Vec2): [number, number] => [GAIN * (W[0][0] * x[0] + W[0][1] * x[1] + C[0]), GAIN * (W[1][0] * x[0] + W[1][1] * x[1] + C[1])]

/** h = σ(z) */
export const hidden = (x: Vec2): [number, number] => {
  const z = linear(x)
  return [sigmoid(z[0]), sigmoid(z[1])]
}

/** Output neuron on the hidden values: 1 when h1 − h2 − 0.4 > 0. */
export const outputOf = (h: Vec2): 0 | 1 => (OUT.w[0] * h[0] + OUT.w[1] * h[1] + OUT.theta > 0 ? 1 : 0)

export const classify = (x: Vec2): 0 | 1 => outputOf(hidden(x))

/** How many XOR points a single step neuron step(w1 x1 + w2 x2 + b) gets right (output 1 when > 0). */
export const lineScore = (w1: number, w2: number, b: number): number =>
  XOR_POINTS.filter((p) => (w1 * p.x[0] + w2 * p.x[1] + b > 0 ? 1 : 0) === p.t).length

/** The sweeping line of the explainer: normal at angle a, passing d away from the square's centre (0.5, 0.5). Returns its score and (w1, w2, b). */
export const sweepLine = (a: number, d: number): { w: [number, number, number]; score: number } => {
  const n: [number, number] = [Math.cos(a), Math.sin(a)]
  const w: [number, number, number] = [n[0], n[1], -(0.5 * n[0] + 0.5 * n[1] + d)]
  return { w, score: lineScore(...w) }
}

/** Hand-made step network from past papers: h1 = OR, h2 = NAND, y = AND. */
export const step = (z: number): 0 | 1 => (z >= 0 ? 1 : 0)
export const orNandAnd = (x: Vec2): { h1: 0 | 1; h2: 0 | 1; y: 0 | 1 } => {
  const h1 = step(x[0] + x[1] - 0.5)
  const h2 = step(-x[0] - x[1] + 1.5)
  return { h1, h2, y: step(h1 + h2 - 1.5) }
}
