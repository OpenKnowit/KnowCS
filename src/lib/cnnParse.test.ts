import { describe, expect, it } from 'vitest'
import { parseKeras, parseTorch } from './cnnParse'
import { kerasCode } from './keras'
import { torchCode } from './torchCode'
import type { Layer, Shape } from './cnnShapes'

const conv = (filters: number, k: number, stride = 1, pad: number | 'same' = 0): Layer => ({ kind: 'conv', filters, k, stride, pad, bias: true })
const pool = (k: number, stride = k, pad: number | 'same' = 0): Layer => ({ kind: 'pool', k, stride, pad, op: 'max' })
const dense = (units: number): Layer => ({ kind: 'dense', units, bias: true })

const STACKS: [Shape, Layer[]][] = [
  [[32, 32, 3], [conv(10, 5, 1, 2)]],
  [[28, 28, 1], [conv(32, 3), conv(64, 3), pool(2), { kind: 'dropout', rate: 0.25 }, { kind: 'flatten' }, dense(128), { kind: 'dropout', rate: 0.5 }, dense(10)]],
  [[224, 224, 3], [conv(64, 7, 2, 3), pool(3, 2, 1), conv(128, 3, 2, 0), conv(256, 3, 2, 1), conv(512, 3, 2, 1), { kind: 'globalpool', op: 'avg' }, dense(1000)]],
  [[32, 32, 3], [conv(32, 5, 1, 'same'), pool(2), conv(64, 3, 2, 'same'), { kind: 'flatten' }, dense(10)]],
]

describe('parseKeras', () => {
  it.each(STACKS)('reads back what kerasCode writes (%j)', (input, layers) => {
    const r = parseKeras(kerasCode(input, layers)!.join('\n'))
    expect(r).toMatchObject({ ok: true, input, layers, head: 'classify' })
  })
  it('reads the hand-written exam style, positional arguments and a sigmoid head', () => {
    const r = parseKeras(`from keras.models import Sequential
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
model = Sequential()
model.add(ZeroPadding2D(padding=(3, 3), input_shape=(224, 224, 3)))
model.add(Conv2D(64, (7, 7), strides=(2, 2), activation='relu'))
model.add(MaxPooling2D((3, 3), strides=2))
model.add(GlobalAveragePooling2D())
model.add(Dense(units=1, activation='sigmoid'))`)
    expect(r).toEqual({ ok: true, input: [224, 224, 3], layers: [conv(64, 7, 2, 3), pool(3, 2), { kind: 'globalpool', op: 'avg' }, dense(1)], head: 'regress', lines: [[4, 5], [6], [7], [8]] })
  })
  it('names the line it cannot use', () => {
    expect(parseKeras('model.add(Conv2D(filters=8, kernel_size=3, input_shape=(8, 8, 1)))\nmodel.add(ZeroPadding2D(1))\nmodel.add(Flatten())')).toEqual({ ok: false, error: 'pad_alone', line: 2 })
    expect(parseKeras('model.add(Conv2D(filters=8, kernel_size=3))')).toMatchObject({ ok: false, error: 'no_input' })
    expect(parseKeras('model.add(Dense(units=0, input_shape=(4,)))')).toMatchObject({ ok: false, error: 'bad_args', line: 1 })
  })
})

describe('parseTorch', () => {
  it('maps each layer to its __init__ line and its forward line', () => {
    const r = parseTorch(torchCode([8, 8, 1], [conv(4, 3, 1, 1), pool(2), { kind: 'flatten' }, dense(3)])!.join('\n'))
    expect(r.ok && r.lines).toEqual([[8, 13], [9, 14], [15], [10, 16]])
  })
  it.each(STACKS.filter(([, l]) => !l.some((x) => x.kind === 'conv' && x.pad === 'same')))('reads back what torchCode writes (%j)', (input, layers) => {
    const r = parseTorch(torchCode(input, layers)!.join('\n'))
    expect(r).toMatchObject({ ok: true, input, layers, head: 'classify' })
  })
})
