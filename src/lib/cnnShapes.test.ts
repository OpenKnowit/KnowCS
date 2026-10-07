import { describe, expect, it } from 'vitest'
import { analyse, totalParams, type Layer } from './cnnShapes'

const conv = (filters: number, k: number, stride = 1, pad: number | 'same' | 'valid' = 0, bias = true): Layer => ({ kind: 'conv', filters, k, stride, pad, bias })
const pool = (k: number, stride = k, pad: number | 'same' | 'valid' = 0): Layer => ({ kind: 'pool', k, stride, pad, op: 'max' })

describe('analyse', () => {
  it('reproduces the lecture 8 all-in-one example (32×32×3, ten 5×5 kernels, pad 2 → 760 params)', () => {
    const [l] = analyse([32, 32, 3], [conv(10, 5, 1, 2)])
    expect(l.output).toEqual([32, 32, 10])
    expect(l.params).toBe(760)
  })

  it('reproduces the lecture 8 MNIST model.summary()', () => {
    const infos = analyse([28, 28, 1], [conv(32, 3), conv(64, 3), pool(2), { kind: 'dropout', rate: 0.25 }, { kind: 'flatten' }, { kind: 'dense', units: 128, bias: true }, { kind: 'dropout', rate: 0.5 }, { kind: 'dense', units: 10, bias: true }])
    expect(infos.map((i) => i.output)).toEqual([[26, 26, 32], [24, 24, 64], [12, 12, 64], [12, 12, 64], [9216], [128], [128], [10]])
    expect(infos.map((i) => i.params)).toEqual([320, 18496, 0, 0, 0, 1179776, 0, 1290])
  })

  it('reproduces Final 2022 Part B Q1(c)(e)(f)', () => {
    const body: Layer[] = [conv(64, 7, 2, 3), pool(3, 2, 1), conv(128, 3, 2, 0), conv(256, 3, 2, 1), conv(512, 3, 2, 1)]
    const infos = analyse([224, 224, 3], body)
    // The marking scheme prints 13 × 13 for layer 4, but ⌊(27 − 3 + 2·1) / 2⌋ + 1 = 14 (13 would mean padding was forgotten).
    // Layer 5 is 7 × 7 either way and parameter counts do not depend on it.
    expect(infos.map((i) => i.output)).toEqual([[112, 112, 64], [56, 56, 64], [27, 27, 128], [14, 14, 256], [7, 7, 512]])
    expect(infos[2].floored).toBe(true)
    expect(totalParams(infos)).toBe(1558656)
    const full = analyse([224, 224, 3], [...body, { kind: 'globalpool', op: 'avg' }, { kind: 'dense', units: 1000, bias: true }])
    expect(totalParams(full)).toBe(2071656)
    expect(totalParams(analyse([224, 224, 3], [{ kind: 'flatten' }, { kind: 'dense', units: 1000, bias: true }]))).toBe(150529000)
  })

  it('reproduces Final 2023 Q6(c)', () => {
    const infos = analyse([256, 256, 3], [conv(32, 5, 1, 2), pool(2), conv(64, 5, 1, 2), pool(2), { kind: 'flatten' }, { kind: 'dense', units: 128, bias: true }, { kind: 'dense', units: 10, bias: true }])
    expect(infos.map((i) => i.params)).toEqual([2432, 0, 51264, 0, 0, 33554560, 1290])
  })

  it("handles Keras 'same' padding with stride 2", () => {
    expect(analyse([32, 32, 3], [conv(64, 3, 2, 'same')])[0].output).toEqual([16, 16, 64])
  })

  it('flags Dense before Flatten', () => {
    expect(analyse([8, 8, 3], [{ kind: 'dense', units: 4, bias: true }])[0].error).toBe('needs_flatten')
  })
})
