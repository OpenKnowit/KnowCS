/** Line and half-plane clipping for decision boundaries drawn in a square window [lo, hi]². */

export type P2 = [number, number]

/** The segment of w1·x + w2·y + b = 0 inside the window, or null if it misses it. */
export function clipLine(w1: number, w2: number, b: number, lo: number, hi: number): [P2, P2] | null {
  const pts: P2[] = []
  if (Math.abs(w2) > 1e-9) for (const x of [lo, hi]) pts.push([x, -(w1 * x + b) / w2])
  if (Math.abs(w1) > 1e-9) for (const y of [lo, hi]) pts.push([-(w2 * y + b) / w1, y])
  const ok = pts.filter(([x, y]) => x >= lo - 1e-9 && x <= hi + 1e-9 && y >= lo - 1e-9 && y <= hi + 1e-9)
  const z = (v: P2): P2 => [v[0] + 0, v[1] + 0] // no -0
  return ok.length >= 2 ? [z(ok[0]), z(ok[1])] : null
}

/** Corners of the part of the window where w1·x + w2·y + b > 0 (or ≥ 0 with inclusive), in order. */
export function halfPlane(w1: number, w2: number, b: number, lo: number, hi: number, inclusive = false): P2[] {
  const box: P2[] = [[lo, lo], [hi, lo], [hi, hi], [lo, hi]]
  const f = (v: P2) => w1 * v[0] + w2 * v[1] + b
  const inside = (v: P2) => (inclusive ? f(v) >= 0 : f(v) > 0)
  const out: P2[] = []
  box.forEach((a, i) => {
    const c = box[(i + 1) % 4]
    if (inside(a)) out.push(a)
    if (inside(a) !== inside(c)) {
      const k = f(a) / (f(a) - f(c))
      out.push([a[0] + k * (c[0] - a[0]), a[1] + k * (c[1] - a[1])])
    }
  })
  return out
}
