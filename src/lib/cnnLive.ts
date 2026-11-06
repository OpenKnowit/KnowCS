/**
 * The small CNN of the /cnn-live/ lab, on ConvNetJS: (conv relu conv relu pool) × 3 and one dense layer, ten 3×3 filters
 * per conv, the layout of CS231n's front-page demo. The weights come from scripts/train-cnn-live.py, in PyTorch's order;
 * cnnLive.test.ts checks every layer against what PyTorch computed.
 */
import convnetjs from './vendor/convnet.js'
import type { Net } from './vendor/convnet.js'

export const INPUT_SIZE = 32

/** the same network as the course writes it; its parameter count is checked in cnnLive.test.ts */
export const KERAS_CODE = `from keras import Sequential, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

block = lambda: [Conv2D(10, 3, padding='same', activation='relu'),
                 Conv2D(10, 3, padding='same', activation='relu'),
                 MaxPooling2D(2)]
model = Sequential([Input((32, 32, 3)), *block(), *block(), *block(),
                    Flatten(), Dense(10, activation='softmax')])
model.summary()`

export type LayerKind = 'input' | 'conv' | 'relu' | 'pool' | 'fc'

/** channels × height × width, channel first */
export interface LiveLayer {
  kind: LayerKind
  depth: number
  size: number
  data: Float32Array
}

export interface LiveRun {
  layers: LiveLayer[]
  probs: number[]
}

const LAYERS = [
  { type: 'input', out_sx: 32, out_sy: 32, out_depth: 3 },
  ...Array.from({ length: 3 }, () => [
    { type: 'conv', sx: 3, filters: 10, stride: 1, pad: 1, activation: 'relu' },
    { type: 'conv', sx: 3, filters: 10, stride: 1, pad: 1, activation: 'relu' },
    { type: 'pool', sx: 2, stride: 2 },
  ]).flat(),
  { type: 'softmax', num_classes: 10 },
]

/** the network with PyTorch's weights moved into ConvNetJS's order: filters are height, width, channel where PyTorch has channel, height, width */
export function loadNet(weights: Float32Array): Net {
  const net = new convnetjs.Net()
  net.makeLayers(LAYERS)
  let at = 0
  for (const layer of net.layers) {
    if (layer.layer_type === 'conv') {
      const inD = layer.filters![0].depth
      for (const f of layer.filters!) {
        for (let c = 0; c < inD; c++) for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) f.w[(3 * y + x) * inD + c] = weights[at++]
      }
    } else if (layer.layer_type === 'fc') {
      for (const f of layer.filters!) {
        for (let c = 0; c < 10; c++) for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) f.w[(4 * y + x) * 10 + c] = weights[at++]
      }
    } else continue
    layer.biases!.w.set(weights.subarray(at, at + layer.filters!.length))
    at += layer.filters!.length
  }
  if (at !== weights.length) throw new Error(`the network takes ${at} weights, the file has ${weights.length}`)
  return net
}

/** `input`: 3 × 32 × 32, channel first, each value pixel / 255 − 0.5. Every layer comes back channel first too. */
export function runCnn(net: Net, input: Float32Array): LiveRun {
  const n = INPUT_SIZE
  const v = new convnetjs.Vol(n, n, 3, 0)
  for (let c = 0; c < 3; c++) for (let i = 0; i < n * n; i++) v.w[i * 3 + c] = input[c * n * n + i]
  const out = net.forward(v, false)
  const layers: LiveLayer[] = []
  for (const l of net.layers) {
    if (l.layer_type === 'softmax') continue
    const a = l.out_act
    const data = new Float32Array(a.depth * a.sx * a.sy)
    const area = a.sx * a.sy
    for (let c = 0; c < a.depth; c++) for (let i = 0; i < area; i++) data[c * area + i] = a.w[i * a.depth + c]
    layers.push({ kind: l.layer_type as LayerKind, depth: a.depth, size: a.sx, data })
  }
  return { layers, probs: Array.from(out.w) }
}

/** the input the network expects, from RGBA pixels of a 32 × 32 picture */
export const toInput = (rgba: ArrayLike<number>): Float32Array => {
  const n = INPUT_SIZE * INPUT_SIZE
  const out = new Float32Array(3 * n)
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) out[c * n + i] = rgba[i * 4 + c] / 255 - 0.5
  }
  return out
}

/** one channel of a layer as grey RGBA. A layer shares its range, so a channel that hardly fires looks dark next to one that does. */
export const channelPixels = (layer: LiveLayer, channel: number): Uint8ClampedArray => {
  const { min, max } = layerRange(layer)
  const span = max - min || 1
  const n = layer.size * layer.size
  const px = new Uint8ClampedArray(n * 4)
  const from = channel * n
  for (let i = 0; i < n; i++) {
    const g = ((layer.data[from + i] - min) / span) * 255
    px[i * 4] = px[i * 4 + 1] = px[i * 4 + 2] = g
    px[i * 4 + 3] = 255
  }
  return px
}

export const layerRange = (layer: LiveLayer): { min: number; max: number } => {
  let min = Infinity
  let max = -Infinity
  for (const v of layer.data) {
    if (v < min) min = v
    if (v > max) max = v
  }
  return { min, max }
}

/** the classes from most to least likely */
export const ranked = (probs: readonly number[]): { k: number; p: number }[] =>
  probs.map((p, k) => ({ k, p })).sort((a, b) => b.p - a.p)
