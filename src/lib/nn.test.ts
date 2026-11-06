import { afterEach, describe, expect, it, vi } from 'vitest'
import { PAINT_SIZE, lastHidden, makeNet, makeTrainer, mulberry, outputLine, outputOf, paintPicture, toyData, vol } from './nn'
import type { NnAct, ToyKind } from './nn'

/** ConvNetJS draws its starting weights with Math.random: fix that, so a run is the same every time */
const seeded = (seed: number) => {
  const r = mulberry(seed)
  vi.spyOn(Math, 'random').mockImplementation(() => r.next())
}
afterEach(() => vi.restoreAllMocks())

const train = (kind: ToyKind, hidden: number[], act: NnAct, lr: number, batches: number) => {
  seeded(1)
  const pts = toyData(kind, mulberry(1))
  const net = makeNet(hidden, act, 'softmax', 2)
  const opt = makeTrainer(net, lr, 0.9, 10, 1e-3)
  const r = mulberry(2)
  for (let i = 0; i < batches * 10; i++) {
    const p = pts[Math.floor(r.next() * pts.length)]
    opt.train(vol(p.x), p.y)
  }
  const good = pts.filter((p) => {
    const q = outputOf(net, p.x)
    return (q[1] > q[0] ? 1 : 0) === p.y
  }).length
  return { acc: good / pts.length, net, pts }
}

describe('the 2-D classification lab', () => {
  it.each([
    ['blobs', [6, 2], 'tanh', 300],
    ['xor', [6, 2], 'tanh', 3000],
    ['circle', [6, 2], 'tanh', 4000],
    ['spiral', [16, 16, 2], 'tanh', 25000],
    ['circle', [6, 2], 'relu', 4000],
  ] as const)('%s with %j %s learns with the default settings', (kind, hidden, act, batches) => {
    expect(train(kind, [...hidden], act, 0.03, batches).acc).toBeGreaterThan(0.95)
  })
  it('what the page says about the circle: no hidden layer fails, 2 units stop short, 6-2 gets there', () => {
    expect(train('circle', [], 'tanh', 0.03, 4000).acc).toBeLessThan(0.75)
    const two = train('circle', [2], 'tanh', 0.03, 8000).acc
    expect(two).toBeGreaterThan(0.8)
    expect(two).toBeLessThan(0.97)
  })
  it('what the page says about the spiral: a learning rate of 0.1 collapses', () => {
    expect(train('spiral', [16, 16, 2], 'tanh', 0.1, 15000).acc).toBeLessThan(0.8)
  })
  it('what the page draws on the right: the line in hidden space separates what the network separates', () => {
    const { net, pts } = train('circle', [6, 2], 'tanh', 0.03, 4000)
    const [a, b, c] = outputLine(net)
    for (const p of pts) {
      const h = lastHidden(net, p.x)
      const q = outputOf(net, p.x)
      expect(h.length).toBe(2)
      expect(a * h[0] + b * h[1] + c > 0).toBe(q[1] > q[0])
    }
  })
})

describe('the painting lab', () => {
  it('draws its pictures in range', () => {
    for (const k of ['sunset', 'face', 'stripes'] as const) {
      const p = paintPicture(k)
      expect(p.length).toBe(PAINT_SIZE * PAINT_SIZE * 3)
      expect(Math.min(...p)).toBeGreaterThanOrEqual(0)
      expect(Math.max(...p)).toBeLessThanOrEqual(1)
      expect(new Set(p).size).toBeGreaterThan(3)
    }
  })
  it('the default network gets close to the picture', () => {
    seeded(3)
    const pic = paintPicture('sunset')
    const net = makeNet([20, 20, 20, 20, 20], 'relu', 'regression', 3)
    const opt = makeTrainer(net, 0.01, 0.9, 5)
    const r = mulberry(4)
    const coord = (i: number) => (i + 0.5) / PAINT_SIZE - 0.5
    const error = () => {
      let s = 0
      for (let y = 0; y < PAINT_SIZE; y += 2) {
        for (let x = 0; x < PAINT_SIZE; x += 2) {
          const q = outputOf(net, [coord(x), coord(y)])
          for (let c = 0; c < 3; c++) s += (q[c] - pic[(y * PAINT_SIZE + x) * 3 + c]) ** 2
        }
      }
      return s / (3 * (PAINT_SIZE / 2) ** 2)
    }
    const before = error()
    for (let i = 0; i < 60000; i++) {
      const x = Math.floor(r.next() * PAINT_SIZE)
      const y = Math.floor(r.next() * PAINT_SIZE)
      const k = (y * PAINT_SIZE + x) * 3
      opt.train(vol([coord(x), coord(y)]), [pic[k], pic[k + 1], pic[k + 2]])
    }
    expect(error()).toBeLessThan(before / 4)
    expect(error()).toBeLessThan(0.02)
  })
})
