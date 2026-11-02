import { describe, expect, it } from 'vitest'
import { runPython } from './minipy'
import { TORCH } from './pyTorch'
import { torchCode } from './torchCode'
import type { Layer } from './cnnShapes'

// what the CNN lab's "Run in Playground" hands to the PyTorch sandbox must run there
describe('CNN lab code in the PyTorch sandbox', () => {
  it('runs the deep-stem network and prints its output shape', () => {
    const conv = (filters: number, k: number, stride: number, pad: number): Layer => ({ kind: 'conv', filters, k, stride, pad, bias: true })
    const layers: Layer[] = [conv(8, 7, 2, 3), { kind: 'pool', k: 3, stride: 2, pad: 1, op: 'max' }, conv(16, 3, 2, 1), { kind: 'globalpool', op: 'avg' }, { kind: 'dense', units: 10, bias: true }]
    const code = `${torchCode([32, 32, 3], layers)!.join('\n')}\n\nx = torch.randn(1, 3, 32, 32)\nprint(model(x).shape)`
    const r = runPython(code, { libs: [TORCH], maxSize: 2_000_000, timeBudget: 20_000 })
    expect(r.error).toBeNull()
    expect(r.stdout).toBe('torch.Size([1, 10])\n')
  })
})
