/**
 * Perceptron learning rule as the exam table runs it (lecture 5):
 *   O = f(Σ wᵢxᵢ + θ),  Δwᵢ = η(T − O)xᵢ,  Δθ = η(T − O), update after every row.
 * Exams disagree on f, so both the output pair and the comparison at 0 are configurable.
 */

export interface StepActivation {
  high: number // output when the comparison holds
  low: number
  /** 'ge': z ≥ 0 gives high (lecture 5, 2022 Spring) · 'gt': z > 0 gives high (2022 Fall) */
  rule: 'ge' | 'gt'
}

export const fire = (z: number, f: StepActivation): number => ((f.rule === 'ge' ? z >= 0 : z > 0) ? f.high : f.low)

export interface PerceptronSample {
  x: number[]
  t: number
}

export interface PerceptronRow {
  epoch: number
  index: number
  x: number[]
  t: number
  z: number
  o: number
  dw: number[]
  w: number[]
  dTheta: number
  theta: number
  updated: boolean
}

// Keep table values exact-looking: 0.1 + 0.2 should print as 0.3.
const tidy = (v: number) => Math.round(v * 1e10) / 1e10

export function perceptronTable(
  data: PerceptronSample[],
  w0: number[],
  theta0: number,
  eta: number,
  f: StepActivation,
  maxEpochs = 10,
): { rows: PerceptronRow[]; converged: boolean; epochs: number } {
  const rows: PerceptronRow[] = []
  let w = [...w0]
  let theta = theta0
  for (let epoch = 1; epoch <= maxEpochs; epoch++) {
    let changed = false
    data.forEach((s, index) => {
      const z = tidy(s.x.reduce((a, xi, i) => a + xi * w[i], theta))
      const o = fire(z, f)
      const err = s.t - o
      const dw = s.x.map((xi) => tidy(eta * err * xi))
      const dTheta = tidy(eta * err)
      w = w.map((wi, i) => tidy(wi + dw[i]))
      theta = tidy(theta + dTheta)
      if (err !== 0) changed = true
      rows.push({ epoch, index, x: s.x, t: s.t, z, o, dw, w: [...w], dTheta, theta, updated: err !== 0 })
    })
    if (!changed) return { rows, converged: true, epochs: epoch }
  }
  return { rows, converged: false, epochs: maxEpochs }
}
