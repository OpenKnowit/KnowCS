// The /mnist/ page's training loop, off the main thread: ConvNetJS forward and backward, PyTorch's updates (mnistNet.ts).
// Like the ConvNetJS demo it keeps 1 training image in 10 for validation and reports averages over the last 100.
import { CLASSES, Optimizer, activations, centerCrop, firstKernels, learn, makeNet, probs, randomCrop } from '../../lib/mnistNet'
import type { TrainConfig } from '../../lib/mnistNet'
import type { Vol } from '../../lib/vendor/convnet.js'
import type { FromWorker, Point, ToWorker } from './mnistProtocol'

const PER_BATCH = 3000
const WINDOW = 100
const post = (m: FromWorker) => (self as unknown as Worker).postMessage(m)

/** the mean of the last WINDOW values */
class Window {
  private v: number[] = []
  add(x: number) {
    this.v.push(x)
    if (this.v.length > WINDOW) this.v.shift()
  }
  get mean(): number | null {
    return this.v.length ? this.v.reduce((s, x) => s + x, 0) / this.v.length : null
  }
}

let cfg: TrainConfig = { optimizer: 'adadelta', lr: 1, momentum: 0, weightDecay: 0, batchSize: 20 }
let net = makeNet()
let opt = new Optimizer(cfg)
const batches: { images: Uint8Array; labels: Uint8Array }[] = []
const trainIdx: number[] = []
const valIdx: number[] = []
let order: number[] = []
let pos = 0
let valPos = 0
let test: { images: Uint8Array; labels: Uint8Array; shown: number[] } | null = null

let running = false
let seen = 0
let inBatch = 0
let loss = new Window()
let trainAcc = new Window()
let valAcc = new Window()
let fwd = new Window()
let bwd = new Window()
let points: Point[] = []
let last: { x: Vol; y: number } | null = null
let rate = { at: 0, seen: 0, perSec: null as number | null }
const sent = { stats: 0, vis: 0, preds: 0 }

function shuffle(a: number[]) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}
const image = (n: number) => ({ images: batches[Math.floor(n / PER_BATCH)].images, k: n % PER_BATCH, y: batches[Math.floor(n / PER_BATCH)].labels[n % PER_BATCH] })

/** the next training image: one pass over the loaded images in a random order, then another (shuffle=True) */
function next(): number {
  if (pos >= order.length) {
    order = shuffle([...trainIdx])
    pos = 0
  }
  return order[pos++]
}

function sendStats() {
  const now = performance.now()
  if (running && rate.at) rate.perSec = ((seen - rate.seen) * 1000) / Math.max(1, now - rate.at)
  rate = { at: now, seen, perSec: running ? rate.perSec : null }
  post({ type: 'stats', running, seen, trainSize: trainIdx.length, loss: loss.mean, trainAcc: trainAcc.mean, valAcc: valAcc.mean, fwdMs: fwd.mean, bwdMs: bwd.mean, perSec: rate.perSec, points })
}

function sendVis() {
  if (!last) return
  const p = probs(net, last.x)
  post({ type: 'vis', label: last.y, layers: activations(net), kernels: firstKernels(net), probs: p })
}

function sendPreds() {
  if (!test) return
  post({
    type: 'preds',
    top: test.shown.map((i) => {
      const p = probs(net, centerCrop(test!.images, i))
      const k = p.map((_, c) => c).sort((a, b) => p[b] - p[a]).slice(0, 3)
      return { k, p: k.map((c) => p[c]) }
    }),
  })
}

function tick() {
  if (!running) return
  if (!trainIdx.length) {
    setTimeout(tick, 100)
    return
  }
  const start = performance.now()
  while (performance.now() - start < 30) {
    const { images, k, y } = image(next())
    const x = randomCrop(images, k)
    const r = learn(net, x, y)
    seen++
    inBatch++
    loss.add(r.loss)
    trainAcc.add(r.right ? 1 : 0)
    fwd.add(r.fwd)
    bwd.add(r.bwd)
    last = { x, y }
    if (inBatch >= cfg.batchSize) {
      opt.step(net, inBatch)
      inBatch = 0
    }
    // one validation image every 10 training images, never trained on
    if (seen % 10 === 0 && valIdx.length) {
      const v = image(valIdx[valPos++ % valIdx.length])
      valAcc.add(probs(net, centerCrop(v.images, v.k)).reduce((best, p, c, all) => (p > all[best] ? c : best), 0) === v.y ? 1 : 0)
    }
    if (seen % 200 === 0 && loss.mean !== null) points.push({ seen, loss: loss.mean, train: trainAcc.mean ?? 0, val: valAcc.mean ?? 0 })
  }
  const now = performance.now()
  if (now - sent.stats > 250) { sendStats(); sent.stats = now }
  if (now - sent.vis > 1000) { sendVis(); sent.vis = now }
  if (now - sent.preds > 1500) { sendPreds(); sent.preds = now }
  setTimeout(tick, 0)
}

function reset() {
  net = makeNet()
  opt = new Optimizer(cfg)
  seen = 0
  inBatch = 0
  loss = new Window()
  trainAcc = new Window()
  valAcc = new Window()
  fwd = new Window()
  bwd = new Window()
  points = []
  last = null
  pos = order.length
  rate = { at: 0, seen: 0, perSec: null }
}

function evaluate() {
  if (!test) return
  const confusion = Array.from({ length: CLASSES }, () => new Array<number>(CLASSES).fill(0))
  const mistakes: { i: number; k: number; p: number }[] = []
  let right = 0
  for (let i = 0; i < test.labels.length; i++) {
    const p = probs(net, centerCrop(test.images, i))
    const k = p.reduce((best, v, c) => (v > p[best] ? c : best), 0)
    confusion[test.labels[i]][k]++
    if (k === test.labels[i]) right++
    else if (mistakes.length < 32) mistakes.push({ i, k, p: p[k] })
  }
  post({ type: 'eval', seen, accuracy: right / test.labels.length, confusion, mistakes })
}

self.onmessage = (e: MessageEvent<ToWorker>) => {
  const m = e.data
  if (m.type === 'train-batch') {
    batches[m.batch] = { images: m.images, labels: m.labels }
    for (let k = 0; k < m.labels.length; k++) {
      const n = m.batch * PER_BATCH + k
      ;(n % 10 === 0 ? valIdx : trainIdx).push(n)
    }
    // a new batch joins at the next pass
    sendStats()
  } else if (m.type === 'test') {
    test = { images: m.images, labels: m.labels, shown: m.shown }
    sendPreds()
  } else if (m.type === 'run') {
    running = m.on
    rate = { at: 0, seen, perSec: null }
    sendStats()
    if (running) tick()
  } else if (m.type === 'config') {
    const same = m.cfg.optimizer === cfg.optimizer
    cfg = m.cfg
    // a new learning rate or batch size keeps what the optimizer has learnt; another optimizer starts afresh
    if (same) opt.cfg = cfg
    else opt = new Optimizer(cfg)
    if (inBatch >= cfg.batchSize) {
      opt.step(net, inBatch)
      inBatch = 0
    }
  } else if (m.type === 'reset') {
    reset()
    sendStats()
    sendPreds()
    post({ type: 'vis', label: -1, layers: [], kernels: firstKernels(net), probs: [] })
  } else if (m.type === 'evaluate') {
    evaluate()
  }
}
