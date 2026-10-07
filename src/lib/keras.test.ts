import { describe, expect, it } from 'vitest'
import type { Layer } from './cnnShapes'
import { kerasCode, kerasSummary } from './keras'

const conv = (filters: number, k: number, stride = 1, pad: number | 'same' = 0): Layer => ({ kind: 'conv', filters, k, stride, pad, bias: true })
const pool = (k: number, stride = k, pad: number = 0): Layer => ({ kind: 'pool', k, stride, pad, op: 'max' })
const dense = (units: number): Layer => ({ kind: 'dense', units, bias: true })
const flatten: Layer = { kind: 'flatten' }

// Final 2022 Part B Q2: 27×27×3 faces → 12 emotions, no padding, 3×3 kernels (strides read off the figure).
const F22 = [conv(32, 3, 3), conv(64, 3, 2), pool(2), flatten, dense(12)]

describe('kerasCode', () => {
  it('writes the Final 2022 model and compile line', () => {
    expect(kerasCode([27, 27, 3], F22)).toEqual([
      'from keras.models import Sequential',
      'from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense',
      '',
      'model = Sequential()',
      "model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(3, 3), activation='relu', input_shape=(27, 27, 3)))",
      "model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu'))",
      'model.add(MaxPooling2D(pool_size=(2, 2)))',
      'model.add(Flatten())',
      "model.add(Dense(units=12, activation='softmax'))",
      "model.compile(optimizer='adam', loss='categorical_crossentropy')",
    ])
  })
  it('switches the head for a 0…1 regression (Final 2022 B Q2(b))', () => {
    const code = kerasCode([27, 27, 3], F22, 'regress')!
    expect(code).toContain("model.add(Dense(units=1, activation='sigmoid'))")
    expect(code.at(-1)).toBe("model.compile(optimizer='adam', loss='mean_squared_error')")
  })
  it('turns an explicit border into ZeroPadding2D, keeps same padding as a keyword', () => {
    const code = kerasCode([32, 32, 3], [conv(10, 5, 1, 2), conv(8, 3, 1, 'same')])!
    expect(code[4]).toBe('model.add(ZeroPadding2D(padding=(2, 2), input_shape=(32, 32, 3)))')
    expect(code[6]).toBe("model.add(Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu'))")
  })
  it('refuses a stack with a shape error', () => {
    expect(kerasCode([4, 4, 1], [conv(1, 5)])).toBeNull()
  })
})

describe('kerasSummary', () => {
  it('matches the Final 2022 figure and counts parameters', () => {
    const s = kerasSummary([27, 27, 3], F22)!
    expect(s.rows.map((r) => [r.name, r.shape, r.params])).toEqual([
      ['conv2d', '(None, 9, 9, 32)', 896],
      ['conv2d_1', '(None, 4, 4, 64)', 18496],
      ['max_pooling2d', '(None, 2, 2, 64)', 0],
      ['flatten', '(None, 256)', 0],
      ['dense', '(None, 12)', 3084],
    ])
    expect(s.total).toBe(22476)
  })
  // The marking scheme's blanks: (16, 16, 64), (8, 8, 64), 36928 and 1024.
  it('fills in the Final 2024 Q7(a) summary (same padding, one stride-2 conv)', () => {
    const s = kerasSummary([32, 32, 3], [conv(32, 5, 1, 'same'), conv(64, 3, 1, 'same'), pool(2), conv(64, 3, 2, 'same'), conv(64, 3, 1, 'same'), pool(2), flatten, dense(128), dense(10)])!
    expect(s.rows.map((r) => r.shape)).toEqual(['(None, 32, 32, 32)', '(None, 32, 32, 64)', '(None, 16, 16, 64)', '(None, 8, 8, 64)', '(None, 8, 8, 64)', '(None, 4, 4, 64)', '(None, 1024)', '(None, 128)', '(None, 10)'])
    expect(s.rows.map((r) => r.params)).toEqual([2432, 18496, 0, 36928, 36928, 0, 0, 131200, 1290])
    expect(s.rows.map((r) => r.name)).toEqual(['conv2d', 'conv2d_1', 'max_pooling2d', 'conv2d_2', 'conv2d_3', 'max_pooling2d_1', 'flatten', 'dense', 'dense_1'])
  })
})
