/**
 * 2-D convolution with the boundary options of lecture 7 and the stride / padding of lecture 8.
 *
 * Padding modes, named as in the lecture:
 *  - zero:      0 0 | a b c | 0 0
 *  - replicate: a a | a b c | c c
 *  - reflect:   b a | a b c | c b   (edge pixel repeated — the Final 2022 / 2024 answers)
 *  - mirror:    c b | a b c | b a   (edge pixel not repeated)
 */

export type PadMode = 'zero' | 'replicate' | 'reflect' | 'mirror'

function sourceIndex(i: number, n: number, mode: PadMode): number | null {
  if (i >= 0 && i < n) return i
  if (mode === 'zero') return null
  if (mode === 'replicate') return i < 0 ? 0 : n - 1
  // fold back and forth until inside [0, n)
  const period = mode === 'reflect' ? 2 * n : 2 * n - 2
  if (period <= 0) return 0
  const k = ((i % period) + period) % period
  if (mode === 'reflect') return k < n ? k : period - 1 - k
  return k < n ? k : period - k
}

/** Pad by `p` on every side. Cells that came from outside the image are flagged. */
export function pad(img: number[][], p: number, mode: PadMode): { value: number; inside: boolean; from: [number, number] | null }[][] {
  const h = img.length
  const w = img[0]?.length ?? 0
  return Array.from({ length: h + 2 * p }, (_, r) =>
    Array.from({ length: w + 2 * p }, (_, c) => {
      const sr = sourceIndex(r - p, h, mode)
      const sc = sourceIndex(c - p, w, mode)
      const inside = r - p >= 0 && r - p < h && c - p >= 0 && c - p < w
      if (sr === null || sc === null) return { value: 0, inside, from: null }
      return { value: img[sr][sc], inside, from: [sr, sc] as [number, number] }
    }),
  )
}

export const flipKernel = (k: number[][]): number[][] => k.map((row) => [...row].reverse()).reverse()

/** ⌊(n − k + 2p) / s⌋ + 1, plus whether the last window lands exactly on the edge. */
export function outputSize(n: number, k: number, p: number, s: number): { size: number; exact: boolean } {
  const span = n - k + 2 * p
  if (span < 0) return { size: 0, exact: false }
  return { size: Math.floor(span / s) + 1, exact: span % s === 0 }
}

export interface ConvOptions {
  pad: number
  mode: PadMode
  stride: number
  /** true = mathematical convolution (kernel rotated 180°), false = cross-correlation */
  flip: boolean
}

export interface ConvResult {
  padded: ReturnType<typeof pad>
  kernel: number[][] // the kernel actually slid over the image (after flipping)
  out: number[][]
  /** top-left corner (in padded coordinates) of the window for out[r][c] */
  origin: (r: number, c: number) => [number, number]
}

export function convolve(img: number[][], kernel: number[][], o: ConvOptions): ConvResult {
  const padded = pad(img, o.pad, o.mode)
  const k = o.flip ? flipKernel(kernel) : kernel
  const kh = k.length
  const kw = k[0].length
  const oh = outputSize(img.length, kh, o.pad, o.stride).size
  const ow = outputSize(img[0].length, kw, o.pad, o.stride).size
  const origin = (r: number, c: number): [number, number] => [r * o.stride, c * o.stride]
  const out = Array.from({ length: oh }, (_, r) =>
    Array.from({ length: ow }, (_, c) => {
      const [r0, c0] = origin(r, c)
      let s = 0
      for (let i = 0; i < kh; i++) for (let j = 0; j < kw; j++) s += k[i][j] * padded[r0 + i][c0 + j].value
      return Math.round(s * 1e9) / 1e9
    }),
  )
  return { padded, kernel: k, out, origin }
}
