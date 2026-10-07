import { describe, expect, it } from 'vitest'
import type { Layer } from './cnnShapes'
import { torchCode } from './torchCode'

const conv = (filters: number, k: number, stride = 1, pad: number | 'same' = 0): Layer => ({ kind: 'conv', filters, k, stride, pad, bias: true })
const pool = (k: number): Layer => ({ kind: 'pool', k, stride: k, pad: 0, op: 'max' })
const dense = (units: number): Layer => ({ kind: 'dense', units, bias: true })

// Lecture 9's MNIST_CNN: two padded 3×3 convs, 2×2 pool, dropout, flatten, 128, dropout, 10.
const LEC9: Layer[] = [conv(32, 3, 1, 1), conv(64, 3, 1, 1), pool(2), { kind: 'dropout', rate: 0.25 }, { kind: 'flatten' }, dense(128), { kind: 'dropout', rate: 0.5 }, dense(10)]

describe('torchCode', () => {
  it('reproduces the layers of lecture 9’s MNIST_CNN', () => {
    const code = torchCode([28, 28, 1], LEC9)!
    expect(code).toContain('        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)')
    expect(code).toContain('        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)')
    expect(code).toContain('        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)')
    expect(code).toContain('        self.fc1 = nn.Linear(64 * 14 * 14, 128)')
    expect(code).toContain('        self.fc2 = nn.Linear(128, 10)')
    expect(code).toContain('        x = x.view(x.size(0), -1)  # (batch, 12544)')
    expect(code).toContain('        x = self.fc2(x)  # (batch, 10) raw logits')
  })
  it('writes channels-first shape comments', () => {
    expect(torchCode([28, 28, 1], LEC9)).toContain('        x = F.relu(self.conv2(x))  # (batch, 64, 28, 28)')
  })
  it('maps Keras same padding to k // 2, even with stride 2', () => {
    expect(torchCode([16, 16, 64], [conv(64, 3, 2, 'same')])).toContain('        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)')
    expect(torchCode([16, 16, 3], [conv(8, 4, 1, 'same')])).toBeNull()
  })
  it('uses one sigmoid unit and MSELoss for a 0…1 regression', () => {
    const code = torchCode([28, 28, 1], LEC9, 'regress')!
    expect(code).toContain('        self.fc2 = nn.Linear(128, 1)')
    expect(code.at(-2)).toBe('lossfunc = nn.MSELoss()')
  })
  it('needs a flatten before Linear', () => {
    expect(torchCode([8, 8, 1], [dense(4)])).toBeNull()
  })
})
