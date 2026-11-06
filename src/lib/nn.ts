/**
 * What the small training labs need around ConvNetJS: a network from a list of widths, and their data.
 * The update is the library's SGDTrainer; nn.test.ts checks the labs' settings do learn.
 */
import convnetjs from './vendor/convnet.js'
import type { Net, SGDTrainer } from './vendor/convnet.js'

export type NnAct = 'tanh' | 'relu'

export interface Rng {
  next(): number
}

/** a small generator, so a page or a test can start from the same weights twice */
export const mulberry = (seed: number): Rng => {
  let a = seed >>> 0
  return {
    next() {
      a = (a + 0x6d2b79f5) >>> 0
      let t = a
      t = Math.imul(t ^ (t >>> 15), t | 1)
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296
    },
  }
}

/** input → fully connected layers → softmax over `outputs` classes, or regression onto `outputs` numbers */
export const makeNet = (hidden: readonly number[], act: NnAct, top: 'softmax' | 'regression', outputs: number, inputs = 2): Net => {
  const net = new convnetjs.Net()
  net.makeLayers([
    { type: 'input', out_sx: 1, out_sy: 1, out_depth: inputs },
    ...hidden.map((n) => ({ type: 'fc', num_neurons: n, activation: act })),
    top === 'softmax' ? { type: 'softmax', num_classes: outputs } : { type: 'regression', num_neurons: outputs },
  ])
  return net
}

export const makeTrainer = (net: Net, lr: number, momentum: number, batch: number, l2 = 0): SGDTrainer =>
  new convnetjs.SGDTrainer(net, { learning_rate: lr, momentum, batch_size: batch, l2_decay: l2 })

export const vol = (x: ArrayLike<number>) => new convnetjs.Vol(x)

/** the output for one input */
export const outputOf = (net: Net, x: ArrayLike<number>): Float64Array => net.forward(vol(x), false).w

/** one input as the last hidden layer holds it (ConvNetJS keeps each layer's output after a forward pass) */
export const lastHidden = (net: Net, x: ArrayLike<number>): Float64Array => {
  net.forward(vol(x), false)
  return net.layers[net.layers.length - 3].out_act.w
}

/** the straight line a x + b y + c = 0 where the output layer changes its mind between two classes */
export const outputLine = (net: Net): [number, number, number] => {
  const fc = net.layers[net.layers.length - 2]
  const [w0, w1] = fc.filters!
  return [w1.w[0] - w0.w[0], w1.w[1] - w0.w[1], fc.biases!.w[1] - fc.biases!.w[0]]
}

// ---------------------------------------------------------------- toy data on [-1, 1]²

export type ToyPoint = { x: [number, number]; y: 0 | 1 }
export type ToyKind = 'blobs' | 'xor' | 'circle' | 'spiral'

export const toyData = (kind: ToyKind, rng: Rng): ToyPoint[] => {
  const pts: ToyPoint[] = []
  const jitter = (s: number) => (rng.next() - 0.5) * s
  if (kind === 'blobs') {
    for (let i = 0; i < 40; i++) pts.push({ x: [-0.45 + jitter(0.6), -0.4 + jitter(0.6)], y: 0 }, { x: [0.45 + jitter(0.6), 0.4 + jitter(0.6)], y: 1 })
  } else if (kind === 'xor') {
    for (let i = 0; i < 25; i++) {
      for (const [sx, sy] of [[-1, -1], [1, 1], [-1, 1], [1, -1]] as const) pts.push({ x: [sx * 0.5 + jitter(0.5), sy * 0.5 + jitter(0.5)], y: sx === sy ? 0 : 1 })
    }
  } else if (kind === 'circle') {
    for (let i = 0; i < 50; i++) {
      const a = rng.next() * 2 * Math.PI
      const r0 = 0.3 * rng.next()
      const r1 = 0.65 + 0.2 * rng.next()
      pts.push({ x: [r0 * Math.cos(a), r0 * Math.sin(a)], y: 0 }, { x: [r1 * Math.cos(a + 1), r1 * Math.sin(a + 1)], y: 1 })
    }
  } else {
    for (let i = 0; i < 60; i++) {
      const t = i / 60
      const r = 0.1 + 0.8 * t
      const a = 3.2 * Math.PI * t
      pts.push({ x: [r * Math.cos(a) + jitter(0.04), r * Math.sin(a) + jitter(0.04)], y: 0 }, { x: [-r * Math.cos(a) + jitter(0.04), -r * Math.sin(a) + jitter(0.04)], y: 1 })
    }
  }
  return pts
}

// ---------------------------------------------------------------- pictures to paint

export const PAINT_SIZE = 64
export type PaintKind = 'sunset' | 'stripes' | 'face'
const clamp01 = (v: number) => Math.min(1, Math.max(0, v))
const mix = (a: number[], b: number[], t: number) => a.map((v, i) => v + (b[i] - v) * clamp01(t))

/** the colour at (u, v) in [0, 1]² of a picture drawn from formulas, so no image file is needed */
const paintAt = (kind: PaintKind, u: number, v: number): number[] => {
  if (kind === 'stripes') {
    const band = Math.floor(((u + v) * 5) % 4)
    return [[0.93, 0.26, 0.21], [1, 0.76, 0.03], [0.13, 0.59, 0.95], [0.3, 0.69, 0.31]][band]
  }
  if (kind === 'face') {
    const d = Math.hypot(u - 0.5, v - 0.5)
    const eye = Math.hypot(Math.abs(u - 0.5) - 0.16, v - 0.38)
    const smile = Math.abs(Math.hypot(u - 0.5, v - 0.42) - 0.24) < 0.035 && v > 0.52
    if (d > 0.42) return [0.15, 0.45, 0.75]
    return eye < 0.05 || smile ? [0.1, 0.1, 0.15] : [0.99, 0.82, 0.25]
  }
  const ridge = 0.72 + 0.09 * Math.sin(u * 9) + 0.05 * Math.sin(u * 23 + 1)
  if (v > ridge) return mix([0.2, 0.12, 0.32], [0.07, 0.05, 0.15], (v - ridge) / (1 - ridge))
  if (Math.hypot(u - 0.68, v - 0.42) < 0.13) return [1, 0.9, 0.55]
  return mix([0.25, 0.3, 0.65], [0.98, 0.55, 0.3], v / ridge)
}

/** RGB in [0, 1], row by row */
export const paintPicture = (kind: PaintKind, size = PAINT_SIZE): Float32Array => {
  const out = new Float32Array(size * size * 3)
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const c = paintAt(kind, (x + 0.5) / size, (y + 0.5) / size)
      out.set(c, (y * size + x) * 3)
    }
  }
  return out
}

/** RGBA pixels into RGB in [0, 1] */
export const rgbFromRgba = (rgba: ArrayLike<number>, n: number): Float32Array => {
  const out = new Float32Array(n * 3)
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) out[i * 3 + c] = rgba[i * 4 + c] / 255
  return out
}
