/** Final 2024 Q2: loop-free image masking and blurring, mirrored in plain TypeScript. */

/** dist[i][j] = distance of pixel (row i, column j) from (cx, cy); mask = dist ≤ r (boundary included). */
export function circleMask(h: number, w: number, cx: number, cy: number, r: number): { dist: number[][]; mask: number[][] } {
  const dist = Array.from({ length: h }, (_, i) => Array.from({ length: w }, (_, j) => Math.sqrt((j - cx) ** 2 + (i - cy) ** 2)))
  return { dist, mask: dist.map((row) => row.map((d) => (d <= r ? 1 : 0))) }
}

/** np.convolve(a, v, 'valid') */
export function convolveValid(a: number[], v: number[]): number[] {
  if (a.length < v.length) [a, v] = [v, a]
  const out: number[] = []
  for (let i = 0; i + v.length <= a.length; i++) {
    let s = 0
    for (let k = 0; k < v.length; k++) s += a[i + k] * v[v.length - 1 - k]
    out.push(s)
  }
  return out
}

/**
 * The paper's img_flatten_conv_1d for a length-3 filter: pad each row with one zero on both sides, flatten,
 * convolve once, put one 0 back at each end, reshape and drop the padding columns.
 * Returns the intermediate arrays too, for the explainer.
 */
export function flattenConv1d(img: number[][], v: number[]): { padded: number[][]; flat: number[]; conv: number[]; out: number[][] } {
  const h = img.length
  const w = img[0].length
  const padded = img.map((row) => [0, ...row, 0])
  const flat = padded.flat()
  const conv = [0, ...convolveValid(flat, v), 0]
  const out = Array.from({ length: h }, (_, i) => conv.slice(i * (w + 2) + 1, i * (w + 2) + 1 + w))
  return { padded, flat, conv, out }
}

export const transpose = (m: number[][]): number[][] => m[0].map((_, j) => m.map((row) => row[j]))

/** Horizontal pass, then the same on the transpose: a 3×3 box blur with zero padding. */
export function separableBlur(img: number[][], v: number[]): number[][] {
  const once = flattenConv1d(img, v).out
  return transpose(flattenConv1d(transpose(once), v).out)
}
