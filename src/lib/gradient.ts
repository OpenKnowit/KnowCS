/** Gradient descent on a 1-D loss, for the learning-rate part of the backprop explainer. */

/** Loss (w − target)² · a and its slope. */
export const bowl = (target: number, a = 1) => ({
  f: (w: number) => a * (w - target) ** 2,
  df: (w: number) => 2 * a * (w - target),
})

/** w₀, w₁, … with w ← w − η·f′(w). */
export function gdPath(w0: number, eta: number, steps: number, df: (w: number) => number): number[] {
  const out = [w0]
  for (let i = 0; i < steps; i++) out.push(out[i] - eta * df(out[i]))
  return out
}
