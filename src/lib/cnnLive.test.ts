// @ts-expect-error node's types are not installed, and this test is the one place that reads binary files
import { readFileSync } from 'node:fs'
// @ts-expect-error as above
import { inflateSync } from 'node:zlib'
import { describe, expect, it } from 'vitest'
import { INPUT_SIZE, KERAS_CODE, channelPixels, layerRange, loadNet, ranked, runCnn, toInput } from './cnnLive'
import type { LiveRun } from './cnnLive'
import { runPython } from './minipy'
import { KERAS } from './pyKeras'
import frozen from './cnnLive.cases.json'
import meta from '../data/cnnLive/meta.json'

const file = readFileSync('src/data/cnnLive/weights.bin')
const weights = new Float32Array(file.buffer.slice(file.byteOffset, file.byteOffset + file.length))

/** the sprite as RGB: the training script writes a PNG of one IDAT chunk with filter 0 on every row */
const sprite = (() => {
  const png = readFileSync('src/data/cnnLive/samples.png')
  const width = png.readUInt32BE(16)
  const idat = png.indexOf('IDAT')
  const raw = inflateSync(png.subarray(idat + 4, idat + 4 + png.readUInt32BE(idat - 4)))
  return { width, rgb: (x: number, y: number, c: number) => raw[y * (width * 3 + 1) + 1 + x * 3 + c] }
})()

const rgbaOf = (index: number) => {
  const out = new Uint8ClampedArray(INPUT_SIZE * INPUT_SIZE * 4)
  const left = (index % 10) * INPUT_SIZE
  const top = Math.floor(index / 10) * INPUT_SIZE
  for (let y = 0; y < INPUT_SIZE; y++) {
    for (let x = 0; x < INPUT_SIZE; x++) {
      for (let c = 0; c < 3; c++) out[(y * INPUT_SIZE + x) * 4 + c] = sprite.rgb(left + x, top + y, c)
      out[(y * INPUT_SIZE + x) * 4 + 3] = 255
    }
  }
  return out
}

const net = loadNet(weights)
const runs: LiveRun[] = frozen.cases.map((_, i) => runCnn(net, toInput(rgbaOf(i))))

describe(`the network on the page against PyTorch ${frozen.torch}`, () => {
  it('has the layers of the lecture’s layout and takes every weight', () => {
    expect(runs[0].layers.map((l) => l.kind).join(' ')).toBe('input conv relu conv relu pool conv relu conv relu pool conv relu conv relu pool fc')
    expect(weights.length).toBe(meta.params)
    expect(() => loadNet(weights.subarray(1))).toThrow()
  })
  it.each(frozen.cases.map((c, i) => [i, c] as const))('sample %i: every layer and the class probabilities', (i, c) => {
    const run = runs[i]
    run.probs.forEach((p, k) => expect(Math.abs(p - c.probs[k])).toBeLessThan(2e-4))
    c.layers.forEach((want, n) => {
      const l = run.layers[n + 1]
      expect(l.kind === 'fc' ? [10] : [l.depth, l.size, l.size]).toEqual(want.shape)
      const { min, max } = layerRange(l)
      const mean = l.data.reduce((a, b) => a + b, 0) / l.data.length
      const near = (a: number, b: number) => expect(Math.abs(a - b)).toBeLessThan(1e-3 * Math.max(1, Math.abs(b)))
      near(mean, want.mean)
      near(min, want.min)
      near(max, want.max)
      near(l.data[0], want.first)
    })
  })
  it('the accuracy on the page is what these 60 suggest, not a number from elsewhere', () => {
    const right = runs.filter((r, i) => ranked(r.probs)[0].k === meta.labels[i]).length / runs.length
    expect(Math.abs(right - meta.testAccuracy)).toBeLessThan(0.2)
  })
})

describe('drawing', () => {
  it('a layer shares one brightness scale', () => {
    const relu = runs[0].layers[2]
    const { max } = layerRange(relu)
    const all = Array.from({ length: relu.depth }, (_, c) => channelPixels(relu, c))
    expect(Math.max(...all.map((px) => Math.max(...px.filter((_, i) => i % 4 === 0))))).toBe(255)
    expect(max).toBeGreaterThan(0)
    expect(all[0].length).toBe(32 * 32 * 4)
  })
})

describe('the Keras code on the page', () => {
  it('builds the same network: same parameter count, same last shapes', () => {
    const r = runPython(`${KERAS_CODE}\nprint(model.count_params(), model.output_shape)`, { libs: [KERAS] })
    expect(r.error).toBeNull()
    expect(r.stdout.trim().endsWith(`${meta.params} (None, 10)`)).toBe(true)
  })
})
