/**
 * The Keras side of the CNN shapes lab: the Sequential code that builds a layer stack, and the rows of its
 * model.summary(). Finals ask for both (Final 2022 B Q2(a) writes the model, Final 2024 Q7(a) fills the summary).
 *
 * Keras only knows padding='valid' or 'same', so an explicit border of p pixels becomes a ZeroPadding2D layer.
 */
import { analyse } from './cnnShapes'
import type { Layer, Shape } from './cnnShapes'

export type Head = 'classify' | 'regress'

/** Classification ends in softmax + categorical cross-entropy; a 0…1 regression in one sigmoid unit + MSE. */
export const COMPILE: Record<Head, { activation: string; loss: string }> = {
  classify: { activation: 'softmax', loss: 'categorical_crossentropy' },
  regress: { activation: 'sigmoid', loss: 'mean_squared_error' },
}

/** One Keras layer of the expanded stack (ZeroPadding2D inserted where needed). */
interface KLayer {
  cls: string
  args: string[]
  /** the lab layer it came from; null for an inserted ZeroPadding2D */
  from: number | null
  /** shape after this layer, without the batch axis */
  out: Shape
  params: number
}

const pair = (n: number) => `(${n}, ${n})`
const tuple = (s: Shape) => (s.length === 1 ? `(${s[0]},)` : `(${s.join(', ')})`)

function expand(input: Shape, layers: Layer[], head: Head): KLayer[] | null {
  const infos = analyse(input, layers)
  if (infos.some((i) => i.error)) return null
  const lastDense = layers.map((l) => l.kind).lastIndexOf('dense')
  const out: KLayer[] = []
  layers.forEach((l, i) => {
    const info = infos[i]
    const pad = (l.kind === 'conv' || l.kind === 'pool') && typeof l.pad === 'number' && l.pad > 0 ? l.pad : 0
    if (pad) out.push({ cls: 'ZeroPadding2D', args: [`padding=${pair(pad)}`], from: null, out: [info.input[0] + 2 * pad, info.input[1] + 2 * pad, info.input[2]], params: 0 })
    const mode = (l.kind === 'conv' || l.kind === 'pool') && l.pad === 'same' ? 'same' : null
    switch (l.kind) {
      case 'conv':
        out.push({
          cls: 'Conv2D',
          args: [`filters=${l.filters}`, `kernel_size=${pair(l.k)}`, ...(l.stride !== 1 ? [`strides=${pair(l.stride)}`] : []), ...(mode ? [`padding='same'`] : []), `activation='relu'`, ...(l.bias ? [] : ['use_bias=False'])],
          from: i, out: info.output, params: info.params,
        })
        break
      case 'pool':
        out.push({ cls: l.op === 'max' ? 'MaxPooling2D' : 'AveragePooling2D', args: [`pool_size=${pair(l.k)}`, ...(l.stride !== l.k ? [`strides=${pair(l.stride)}`] : []), ...(mode ? [`padding='same'`] : [])], from: i, out: info.output, params: 0 })
        break
      case 'globalpool':
        out.push({ cls: l.op === 'max' ? 'GlobalMaxPooling2D' : 'GlobalAveragePooling2D', args: [], from: i, out: info.output, params: 0 })
        break
      case 'flatten':
        out.push({ cls: 'Flatten', args: [], from: i, out: info.output, params: 0 })
        break
      case 'dense': {
        const last = i === lastDense
        const units = last && head === 'regress' ? 1 : l.units
        const inN = info.input.reduce((a, b) => a * b, 1)
        out.push({ cls: 'Dense', args: [`units=${units}`, `activation='${last ? COMPILE[head].activation : 'relu'}'`, ...(l.bias ? [] : ['use_bias=False'])], from: i, out: [units], params: (inN + (l.bias ? 1 : 0)) * units })
        break
      }
      case 'dropout':
        out.push({ cls: 'Dropout', args: [`rate=${l.rate}`], from: i, out: info.output, params: 0 })
        break
    }
  })
  return out
}

/** The Sequential code, one string per line. null when the stack has a shape error. */
export function kerasCode(input: Shape, layers: Layer[], head: Head = 'classify'): string[] | null {
  const ks = expand(input, layers, head)
  if (!ks) return null
  const used = [...new Set(ks.map((k) => k.cls))]
  const lines = ['from keras.models import Sequential', `from keras.layers import ${used.join(', ')}`, '', 'model = Sequential()']
  ks.forEach((k, j) => {
    const args = j === 0 ? [...k.args, `input_shape=${tuple(input)}`] : k.args
    lines.push(`model.add(${k.cls}(${args.join(', ')}))`)
  })
  lines.push(`model.compile(optimizer='adam', loss='${COMPILE[head].loss}')`)
  return lines
}

export interface SummaryRow {
  name: string
  cls: string
  shape: string
  params: number
}

const SNAKE: Record<string, string> = {
  Conv2D: 'conv2d', MaxPooling2D: 'max_pooling2d', AveragePooling2D: 'average_pooling2d', GlobalAveragePooling2D: 'global_average_pooling2d',
  GlobalMaxPooling2D: 'global_max_pooling2d', Flatten: 'flatten', Dense: 'dense', Dropout: 'dropout', ZeroPadding2D: 'zero_padding2d',
}

/** model.summary() rows with Keras's automatic names (conv2d, conv2d_1, …) and "(None, …)" shapes. */
export function kerasSummary(input: Shape, layers: Layer[], head: Head = 'classify'): { rows: SummaryRow[]; total: number } | null {
  const ks = expand(input, layers, head)
  if (!ks) return null
  const seen: Record<string, number> = {}
  const rows = ks.map((k) => {
    const base = SNAKE[k.cls]
    const n = seen[base] ?? 0
    seen[base] = n + 1
    return { name: n ? `${base}_${n}` : base, cls: k.cls, shape: `(None, ${k.out.join(', ')})`, params: k.params }
  })
  return { rows, total: rows.reduce((s, r) => s + r.params, 0) }
}
