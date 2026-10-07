/**
 * Affine image transforms as lecture 7 writes them: a 2 × 3 matrix M with
 *   [x', y']ᵀ = M · [x, y, 1]ᵀ
 * in image coordinates (x = column, rightwards; y = row, downwards), applied with cv2.warpAffine semantics.
 */

export type Affine = [[number, number, number], [number, number, number]]

export const translate = (tx: number, ty: number): Affine => [[1, 0, tx], [0, 1, ty]]

/** "Reflect along the x-axis": y' = −y + (rows − 1) — the picture turns upside down. */
export const reflectX = (rows: number): Affine => [[1, 0, 0], [0, -1, rows - 1]]

/** "Reflect along the y-axis": x' = −x + (cols − 1) — a left–right mirror. */
export const reflectY = (cols: number): Affine => [[-1, 0, cols - 1], [0, 1, 0]]

/**
 * Rotation by θ degrees about (x0, y0), exactly the lecture's matrix (and cv2.getRotationMatrix2D with scale 1).
 * With y pointing down, a positive angle turns the picture counter-clockwise on screen.
 */
export function rotate(deg: number, x0: number, y0: number): Affine {
  const a = (deg * Math.PI) / 180
  const c = Math.cos(a)
  const s = Math.sin(a)
  return [
    [c, s, -x0 * c - y0 * s + x0],
    [-s, c, x0 * s - y0 * c + y0],
  ]
}

/** Uniform or non-uniform scaling about (x0, y0). */
export const scale = (sx: number, sy: number, x0: number, y0: number): Affine => [[sx, 0, (1 - sx) * x0], [0, sy, (1 - sy) * y0]]

/** Horizontal shear: x' = x + k·y. */
export const shear = (k: number): Affine => [[1, k, 0], [0, 1, 0]]

export function applyPoint(M: Affine, x: number, y: number): [number, number] {
  return [M[0][0] * x + M[0][1] * y + M[0][2], M[1][0] * x + M[1][1] * y + M[1][2]]
}

/** The inverse map, or null when the 2 × 2 part is singular. */
export function invert(M: Affine): Affine | null {
  const [[a, b, c], [d, e, f]] = M
  const det = a * e - b * d
  if (Math.abs(det) < 1e-12) return null
  const ia = e / det, ib = -b / det, id = -d / det, ie = a / det
  return [
    [ia, ib, -(ia * c + ib * f)],
    [id, ie, -(id * c + ie * f)],
  ]
}

/** Nearest-pixel rounding as cv2.INTER_NEAREST does it inside warpAffine (halves round up). */
const near = (v: number) => Math.floor(v + 0.5 + 1e-9)

/**
 * cv2.warpAffine(src, M, (cols, rows), flags=cv2.INTER_NEAREST) with the default constant border of 0.
 * Like OpenCV it walks the *output*: dst(x, y) = src(M⁻¹ · (x, y)). `from` records the source pixel of each output pixel.
 */
export function warpAffine(img: number[][], M: Affine, cols = img[0].length, rows = img.length): { out: number[][]; from: ([number, number] | null)[][] } {
  const inv = invert(M)
  const out: number[][] = []
  const from: ([number, number] | null)[][] = []
  for (let y = 0; y < rows; y++) {
    out.push([])
    from.push([])
    for (let x = 0; x < cols; x++) {
      const src = inv ? applyPoint(inv, x, y) : null
      const sx = src ? near(src[0]) : -1
      const sy = src ? near(src[1]) : -1
      const inside = sy >= 0 && sy < img.length && sx >= 0 && sx < img[0].length
      out[y].push(inside ? img[sy][sx] : 0)
      from[y].push(inside ? [sx, sy] : null)
    }
  }
  return { out, from }
}

export interface AffineProps {
  /** lengths unchanged: the 2 × 2 part is orthogonal (AᵀA = I) */
  distances: boolean
  /** angles unchanged: AᵀA = s²·I for some s > 0 */
  angles: boolean
  /** collinearity, parallel lines and ratios along a line — true for every affine map */
  parallel: true
  /** the determinant is negative: the picture is mirrored */
  mirrored: boolean
  det: number
}

export function properties(M: Affine): AffineProps {
  const [[a, b], [d, e]] = [M[0], M[1]]
  const p = a * a + d * d // |first column|²
  const q = b * b + e * e // |second column|²
  const r = a * b + d * e // columns' dot product
  const eps = 1e-9
  const angles = Math.abs(r) < eps && Math.abs(p - q) < eps && p > eps
  const det = a * e - b * d
  return { distances: angles && Math.abs(p - 1) < eps, angles, parallel: true, mirrored: det < -eps, det }
}

/**
 * How far (in pixels, Chebyshev distance) the farthest output pixel reaches for its source value —
 * the exam's argument that a flip is a global operation (Final 2024 Q5(e)).
 */
export function reach(from: ([number, number] | null)[][]): number {
  let r = 0
  from.forEach((row, y) => row.forEach((s, x) => {
    if (s) r = Math.max(r, Math.abs(s[0] - x), Math.abs(s[1] - y))
  }))
  return r
}

/**
 * Could a single 3 × 3 convolution do this? Only if every output pixel takes its value from the same
 * neighbour offset (the 2 × 2 part is the identity) and that offset is at most one pixel away.
 */
export function doableBy3x3(M: Affine): boolean {
  const [[a, b, c], [d, e, f]] = M
  const id = Math.abs(a - 1) < 1e-9 && Math.abs(b) < 1e-9 && Math.abs(d) < 1e-9 && Math.abs(e - 1) < 1e-9
  return id && Math.abs(c) <= 1 && Math.abs(f) <= 1 && Number.isInteger(c) && Number.isInteger(f)
}
