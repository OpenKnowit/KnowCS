/**
 * The /mnist/ page's network: ConvNetJS's MNIST demo (Karpathy), trained in the browser. ConvNetJS runs the forward and
 * backward passes; the start and the updates are PyTorch's, so the page's PyTorch code describes exactly what trains:
 * - weights and biases start as nn.Conv2d / nn.Linear start them, uniform in ±1/√fan_in;
 * - the gradient is the batch mean of −log p(y) (nn.CrossEntropyLoss), weight decay added to it as torch.optim does;
 * - SGD (with momentum), Adam and Adadelta step as torch.optim's do (ConvNetJS's own Adam has its bias correction
 *   upside down, so its trainer is not used).
 * mnistNet.test.ts checks one step of each optimizer against numbers torch 2.14 printed.
 */
import convnetjs from './vendor/convnet.js'
import type { Net, Vol } from './vendor/convnet.js'

export const IMAGE = 28
export const INPUT = 24
export const CLASSES = 10

/** the demo's layers: conv 5×5×8 → max-pool 2 → conv 5×5×16 → max-pool 3 → 10 scores */
export const LAYERS = [
  { type: 'input', out_sx: INPUT, out_sy: INPUT, out_depth: 1 },
  { type: 'conv', sx: 5, filters: 8, stride: 1, pad: 2, activation: 'relu' },
  { type: 'pool', sx: 2, stride: 2 },
  { type: 'conv', sx: 5, filters: 16, stride: 1, pad: 2, activation: 'relu' },
  { type: 'pool', sx: 3, stride: 3 },
  { type: 'softmax', num_classes: CLASSES },
]

/** what each ConvNetJS layer's output is called on the page, and its shape (channels, side) */
export const SHOWN = [
  { at: 0, key: 'input', depth: 1, side: INPUT },
  { at: 2, key: 'conv1', depth: 8, side: INPUT },
  { at: 3, key: 'pool1', depth: 8, side: 12 },
  { at: 5, key: 'conv2', depth: 16, side: 12 },
  { at: 6, key: 'pool2', depth: 16, side: 4 },
] as const

export type Rand = () => number

/** a new network, started the way PyTorch starts nn.Conv2d and nn.Linear */
export function makeNet(rand: Rand = Math.random): Net {
  const net = new convnetjs.Net()
  net.makeLayers(LAYERS)
  for (const layer of net.layers) {
    if (!layer.filters || !layer.biases) continue
    const fanIn = layer.layer_type === 'conv' ? layer.sx! * layer.sx! * layer.in_depth! : layer.num_inputs!
    const bound = 1 / Math.sqrt(fanIn)
    for (const f of layer.filters) for (let i = 0; i < f.w.length; i++) f.w[i] = (2 * rand() - 1) * bound
    for (let i = 0; i < layer.biases.w.length; i++) layer.biases.w[i] = (2 * rand() - 1) * bound
  }
  return net
}

export const paramCount = (net: Net) => net.getParamsAndGrads().reduce((s, pg) => s + pg.params.length, 0)

// ---------------------------------------------------------------- data

/** the 24×24 window at (dx, dy) of image i, scaled as ToTensor() then Normalize((0.5,), (1.0,)): x / 255 − 0.5 */
export function crop(images: Uint8Array, i: number, dx: number, dy: number): Vol {
  const v = new convnetjs.Vol(INPUT, INPUT, 1, 0)
  const base = i * IMAGE * IMAGE
  for (let y = 0; y < INPUT; y++) {
    for (let x = 0; x < INPUT; x++) v.w[y * INPUT + x] = images[base + (y + dy) * IMAGE + x + dx] / 255 - 0.5
  }
  return v
}
/** transforms.RandomCrop(24): any of the 5 × 5 positions */
export const randomCrop = (images: Uint8Array, i: number, rand: Rand = Math.random) => crop(images, i, Math.floor(rand() * 5), Math.floor(rand() * 5))
/** transforms.CenterCrop(24) */
export const centerCrop = (images: Uint8Array, i: number) => crop(images, i, 2, 2)

/** one image out of a sprite of 60 per row (RGBA from a canvas), as 784 grey bytes */
export function unpackSprite(rgba: Uint8ClampedArray, width: number, count: number): Uint8Array {
  const out = new Uint8Array(count * IMAGE * IMAGE)
  const perRow = width / IMAGE
  for (let i = 0; i < count; i++) {
    const r0 = Math.floor(i / perRow) * IMAGE
    const c0 = (i % perRow) * IMAGE
    for (let y = 0; y < IMAGE; y++) {
      for (let x = 0; x < IMAGE; x++) out[i * IMAGE * IMAGE + y * IMAGE + x] = rgba[((r0 + y) * width + c0 + x) * 4]
    }
  }
  return out
}

// ---------------------------------------------------------------- the optimizers, as torch.optim writes them

export type OptName = 'sgd' | 'adam' | 'adadelta'
export interface TrainConfig {
  optimizer: OptName
  lr: number
  momentum: number
  weightDecay: number
  batchSize: number
}
export const DEFAULTS: Record<OptName, Pick<TrainConfig, 'lr' | 'momentum'>> = {
  // the demo trains with Adadelta; lr=1.0 is PyTorch's default, rho=0.95 the demo's
  adadelta: { lr: 1, momentum: 0 },
  sgd: { lr: 0.01, momentum: 0.9 },
  adam: { lr: 0.001, momentum: 0 },
}
export const RHO = 0.95
export const ADADELTA_EPS = 1e-6
export const BETAS = [0.9, 0.999] as const
export const ADAM_EPS = 1e-8

export class Optimizer {
  private a: Float64Array[] = []
  private b: Float64Array[] = []
  private t = 0
  /** a new learning rate or batch size applies from the next step */
  cfg: TrainConfig
  constructor(cfg: TrainConfig) {
    this.cfg = cfg
  }
  /** optimizer.step() for gradients summed over `batch` examples, then optimizer.zero_grad() */
  step(net: Net, batch: number) {
    const { optimizer, lr, momentum, weightDecay: wd } = this.cfg
    const pgs = net.getParamsAndGrads()
    if (!this.a.length) {
      this.a = pgs.map((pg) => new Float64Array(pg.params.length))
      this.b = pgs.map((pg) => new Float64Array(pg.params.length))
    }
    this.t++
    const t = this.t
    const [b1, b2] = BETAS
    pgs.forEach(({ params: p, grads: g }, k) => {
      const a = this.a[k]
      const b = this.b[k]
      for (let j = 0; j < p.length; j++) {
        const gj = g[j] / batch + wd * p[j]
        if (optimizer === 'sgd') {
          if (momentum > 0) {
            a[j] = t === 1 ? gj : momentum * a[j] + gj
            p[j] -= lr * a[j]
          } else p[j] -= lr * gj
        } else if (optimizer === 'adam') {
          a[j] = b1 * a[j] + (1 - b1) * gj
          b[j] = b2 * b[j] + (1 - b2) * gj * gj
          const denom = Math.sqrt(b[j]) / Math.sqrt(1 - b2 ** t) + ADAM_EPS
          p[j] -= (lr / (1 - b1 ** t)) * (a[j] / denom)
        } else {
          a[j] = RHO * a[j] + (1 - RHO) * gj * gj
          const delta = (Math.sqrt(b[j] + ADADELTA_EPS) / Math.sqrt(a[j] + ADADELTA_EPS)) * gj
          b[j] = RHO * b[j] + (1 - RHO) * delta * delta
          p[j] -= lr * delta
        }
        g[j] = 0
      }
    })
  }
}

/** forward + backward of one example; the gradients add up until the optimizer steps. Returns the loss and whether it was right. */
export function learn(net: Net, x: Vol, y: number): { loss: number; right: boolean; fwd: number; bwd: number } {
  const t0 = performance.now()
  net.forward(x, true)
  const t1 = performance.now()
  const right = net.getPrediction() === y
  const loss = net.backward(y)
  return { loss, right, fwd: t1 - t0, bwd: performance.now() - t1 }
}

export const probs = (net: Net, x: Vol): number[] => Array.from(net.forward(x, false).w)

/** each shown layer of the last forward pass, channel by channel (ConvNetJS keeps x, y, channel order) */
export function activations(net: Net): Float32Array[][] {
  return SHOWN.map(({ at, depth, side }) => {
    const w = net.layers[at].out_act.w
    return Array.from({ length: depth }, (_, d) => {
      const out = new Float32Array(side * side)
      for (let i = 0; i < side * side; i++) out[i] = w[i * depth + d]
      return out
    })
  })
}

/** conv1's 8 kernels, 5 × 5 each */
export const firstKernels = (net: Net): Float32Array[] => net.layers[1].filters!.map((f) => Float32Array.from(f.w))

// ---------------------------------------------------------------- the same thing in PyTorch

const num = (v: number) => (v >= 0.001 || v === 0 ? String(v) : v.toExponential())

export function pytorchCode(cfg: TrainConfig): string {
  const wd = cfg.weightDecay ? `, weight_decay=${num(cfg.weightDecay)}` : ''
  const opt = cfg.optimizer === 'sgd'
    ? `torch.optim.SGD(model.parameters(), lr=${num(cfg.lr)}${cfg.momentum ? `, momentum=${cfg.momentum}` : ''}${wd})`
    : cfg.optimizer === 'adam'
      ? `torch.optim.Adam(model.parameters(), lr=${num(cfg.lr)}${wd})`
      : `torch.optim.Adadelta(model.parameters(), lr=${num(cfg.lr)}, rho=${RHO}${wd})`
  return `import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# 28×28 → a random 24×24 window, then [0, 255] → [-0.5, 0.5]
train_tf = transforms.Compose([transforms.RandomCrop(24), transforms.ToTensor(),
                               transforms.Normalize((0.5,), (1.0,))])
test_tf = transforms.Compose([transforms.CenterCrop(24), transforms.ToTensor(),
                              transforms.Normalize((0.5,), (1.0,))])
data = datasets.MNIST('data', train=True, download=True, transform=train_tf)
train_set, val_set = random_split(data, [0.9, 0.1])     # 1 in 10 kept for validation
test_set = datasets.MNIST('data', train=False, transform=test_tf)
train_loader = DataLoader(train_set, batch_size=${cfg.batchSize}, shuffle=True)

model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=5, padding=2), nn.ReLU(),   # (8, 24, 24)
    nn.MaxPool2d(2),                                        # (8, 12, 12)
    nn.Conv2d(8, 16, kernel_size=5, padding=2), nn.ReLU(),  # (16, 12, 12)
    nn.MaxPool2d(3),                                        # (16, 4, 4)
    nn.Flatten(),                                           # 256
    nn.Linear(16 * 4 * 4, 10),                              # 10 scores
)
loss_fn = nn.CrossEntropyLoss()        # softmax, then −log p(true class)
optimizer = ${opt}

for epoch in range(3):
    model.train()
    for x, y in train_loader:          # x: (${cfg.batchSize}, 1, 24, 24), y: (${cfg.batchSize},)
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    x, y = next(iter(DataLoader(test_set, batch_size=len(test_set))))
    print('test accuracy', (model(x).argmax(1) == y).float().mean().item())`
}
