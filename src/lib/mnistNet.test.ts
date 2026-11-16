import { describe, expect, it } from 'vitest'
import type { Net } from './vendor/convnet.js'
import { CLASSES, INPUT, Optimizer, activations, centerCrop, crop, firstKernels, learn, makeNet, paramCount, probs, pytorchCode, randomCrop, unpackSprite } from './mnistNet'
import type { TrainConfig } from './mnistNet'

// what torch 2.14 printed for three steps on p = [0.5, -0.3, 0.2] with these gradients, summed over a batch of 2
const GRADS = [[0.4, -0.2, 0.1], [-0.1, 0.3, 0.05], [0.2, 0.2, -0.4]]
const TORCH: [Partial<TrainConfig>, number[]][] = [
  [{ optimizer: 'sgd', lr: 0.1, momentum: 0.9 }, [0.4425683795, -0.3097414297, 0.2006004548]],
  [{ optimizer: 'sgd', lr: 0.1, momentum: 0 }, [0.47353647949999994, -0.31410588970000003, 0.2119130948]],
  [{ optimizer: 'adam', lr: 0.01 }, [0.47878033596572217, -0.29682701488057117, 0.1847051631530245]],
  [{ optimizer: 'adadelta', lr: 1 }, [0.4939427272890751, -0.30471994135449343, 0.19922647411249567]],
]

describe('the optimizers step as torch.optim does', () => {
  for (const [cfg, want] of TORCH) {
    it(`${cfg.optimizer} momentum=${cfg.momentum ?? 0}`, () => {
      const params = Float64Array.from([0.5, -0.3, 0.2])
      const grads = new Float64Array(3)
      const net = { getParamsAndGrads: () => [{ params, grads }] } as unknown as Net
      const opt = new Optimizer({ optimizer: 'sgd', lr: 0.1, momentum: 0, weightDecay: 0.01, batchSize: 2, ...cfg })
      for (const g of GRADS) {
        grads.set(g)
        opt.step(net, 2)
        expect([...grads]).toEqual([0, 0, 0])
      }
      params.forEach((p, i) => expect(p).toBeCloseTo(want[i], 12))
    })
  }
})

describe('the network', () => {
  it('has the demo layers and as many parameters as the PyTorch model', () => {
    const net = makeNet()
    // conv 1·8·25 + 8, conv 8·16·25 + 16, linear 256·10 + 10
    expect(paramCount(net)).toBe(208 + 3216 + 2570)
    net.forward(centerCrop(new Uint8Array(28 * 28), 0))
    expect(activations(net).map((l) => [l.length, Math.sqrt(l[0].length)])).toEqual([[1, 24], [8, 24], [8, 12], [16, 12], [16, 4]])
    expect(firstKernels(net).map((k) => k.length)).toEqual(Array(8).fill(25))
  })

  it('starts as nn.Conv2d and nn.Linear do: uniform in ±1/√fan_in, biases too', () => {
    const net = makeNet()
    const bounds = [1 / 5, 1 / Math.sqrt(200), 1 / 16]
    net.layers.filter((l) => l.filters).forEach((l, k) => {
      const all = [...l.filters!.flatMap((f) => [...f.w]), ...l.biases!.w]
      expect(Math.max(...all.map(Math.abs))).toBeLessThanOrEqual(bounds[k])
      expect(Math.max(...all.map(Math.abs))).toBeGreaterThan(bounds[k] * 0.8)
    })
  })

  it('crops 24 × 24 and scales to [−0.5, 0.5]', () => {
    const img = Uint8Array.from({ length: 2 * 784 }, (_, i) => i % 256)
    const v = crop(img, 1, 3, 4)
    expect(v.w[0]).toBeCloseTo(((784 + 4 * 28 + 3) % 256) / 255 - 0.5)
    expect(centerCrop(img, 0).w[INPUT * INPUT - 1]).toBeCloseTo(((25 * 28 + 25) % 256) / 255 - 0.5)
    const r = randomCrop(img, 0, () => 0.99)
    expect(r.w[0]).toBeCloseTo(((4 * 28 + 4) % 256) / 255 - 0.5)
  })

  it('unpacks a sprite of 60 images per row', () => {
    const width = 60 * 28
    const rgba = new Uint8ClampedArray(width * 56 * 4)
    // image 61 is row 1, column 1: put 200 at its (x=3, y=5)
    rgba[((28 + 5) * width + 28 + 3) * 4] = 200
    const out = unpackSprite(rgba, width, 62)
    expect(out[61 * 784 + 5 * 28 + 3]).toBe(200)
    expect(out.reduce((s, v) => s + v, 0)).toBe(200)
  })

  it('learns: the loss on a made-up task falls', () => {
    // class k: a bright bar across row 2 + 2k
    let seed = 1
    const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
    const n = 200
    const images = new Uint8Array(n * 784)
    const labels = Array.from({ length: n }, (_, i) => i % CLASSES)
    labels.forEach((y, i) => { for (let x = 0; x < 28; x++) images[i * 784 + (2 + 2 * y) * 28 + x] = 255 })
    const net = makeNet(rand)
    const opt = new Optimizer({ optimizer: 'adadelta', lr: 1, momentum: 0, weightDecay: 0, batchSize: 10 })
    const lossAt = () => labels.reduce((s, y, i) => s - Math.log(probs(net, centerCrop(images, i))[y]), 0) / n
    const before = lossAt()
    for (let step = 0; step < 600; step++) {
      const i = Math.floor(rand() * n)
      learn(net, centerCrop(images, i), labels[i])
      if ((step + 1) % 10 === 0) opt.step(net, 10)
    }
    expect(lossAt()).toBeLessThan(before / 3)
  })

  it('writes the PyTorch code for the chosen optimizer', () => {
    const code = (c: Partial<TrainConfig>) => pytorchCode({ optimizer: 'adadelta', lr: 1, momentum: 0, weightDecay: 0, batchSize: 20, ...c })
    expect(code({})).toContain('torch.optim.Adadelta(model.parameters(), lr=1, rho=0.95)')
    expect(code({ optimizer: 'sgd', lr: 0.01, momentum: 0.9, weightDecay: 1e-4 })).toContain('torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)')
    expect(code({ optimizer: 'adam', lr: 0.001, batchSize: 50 })).toContain('batch_size=50')
  })
})
