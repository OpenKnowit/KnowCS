/**
 * Reads the CNN lab's own code back into a layer stack, so editing the Keras or PyTorch code moves the layers.
 * It understands what kerasCode / torchCode write and the usual hand-written variants (positional arguments,
 * tuples or plain ints, keras.layers.X / layers.X / nn.X prefixes); anything else is reported with its line.
 */
import type { Layer, Padding, Shape } from './cnnShapes'
import type { Head } from './keras'

export type ParseError = 'empty' | 'no_input' | 'unknown' | 'pad_alone' | 'bad_args'
/** lines: for each layer, the 1-based code lines that make it (a ZeroPadding2D line belongs to the layer it pads) */
export type Parsed = { ok: true; input: Shape; layers: Layer[]; head: Head; lines: number[][] } | { ok: false; error: ParseError; line: number }

interface Call {
  name: string
  args: string[]
  kw: Record<string, string>
  line: number
}

/** split "a, (1, 2), b=[3, 4]" at top-level commas */
function splitTop(s: string): string[] {
  const out: string[] = []
  let depth = 0
  let cur = ''
  let quote = ''
  for (const ch of s) {
    if (quote) {
      if (ch === quote) quote = ''
    } else if (ch === '"' || ch === "'") quote = ch
    else if ('([{'.includes(ch)) depth++
    else if (')]}'.includes(ch)) depth--
    if (ch === ',' && depth === 0 && !quote) {
      out.push(cur.trim())
      cur = ''
    } else cur += ch
  }
  if (cur.trim()) out.push(cur.trim())
  return out
}

/** every Name( … ) call in the code whose name is in `names`, in reading order, with its arguments */
function findCalls(code: string, names: string[]): Call[] {
  const src = code.replace(/#[^\n]*/g, (m) => ' '.repeat(m.length))
  const re = new RegExp(`(?:[A-Za-z_][\\w]*\\.)*\\b(${names.join('|')})\\s*\\(`, 'g')
  const calls: Call[] = []
  for (const m of src.matchAll(re)) {
    let depth = 1
    let i = m.index + m[0].length
    const start = i
    while (i < src.length && depth) {
      if (src[i] === '(') depth++
      else if (src[i] === ')') depth--
      i++
    }
    const parts = splitTop(src.slice(start, i - 1))
    const args: string[] = []
    const kw: Record<string, string> = {}
    for (const p of parts) {
      const k = p.match(/^([A-Za-z_]\w*)\s*=(?!=)\s*([\s\S]+)$/)
      if (k) kw[k[1]] = k[2].trim()
      else args.push(p)
    }
    calls.push({ name: m[1], args, kw, line: src.slice(0, m.index).split('\n').length })
  }
  return calls
}

/** the first integer of "3", "(3, 3)" or "((1, 1), (1, 1))"; NaN when there is none */
const firstInt = (v: string | undefined): number => {
  const m = v?.match(/-?\d+/)
  return m ? Number(m[0]) : NaN
}
const ints = (v: string | undefined): number[] => (v?.match(/\d+/g) ?? []).map(Number)
const str = (v: string | undefined): string | null => v?.match(/^['"](.*)['"]$/)?.[1] ?? null
const num = (v: string | undefined): number => (v === undefined ? NaN : Number(v))
const pos = (n: number) => Number.isFinite(n) && n >= 1

const KERAS = ['Input', 'InputLayer', 'ZeroPadding2D', 'Conv2D', 'MaxPooling2D', 'MaxPool2D', 'AveragePooling2D', 'AvgPool2D', 'GlobalAveragePooling2D', 'GlobalMaxPooling2D', 'Flatten', 'Dense', 'Dropout']

export function parseKeras(code: string): Parsed {
  const calls = findCalls(code, KERAS)
  let input: Shape | null = null
  let pending: { p: number; line: number } | null = null
  const layers: Layer[] = []
  const lines: number[][] = []
  let head: Head = 'classify'
  const fail = (error: ParseError, line: number): Parsed => ({ ok: false, error, line })
  for (const c of calls) {
    const at = layers.length
    const padLine = pending?.line
    const shape = c.kw.input_shape ?? (c.name === 'Input' || c.name === 'InputLayer' ? (c.kw.shape ?? c.args[0]) : undefined)
    if (shape && !input) {
      const s = ints(shape)
      if (!(s.length === 3 || s.length === 1) || !s.every(pos)) return fail('bad_args', c.line)
      input = s
    }
    if (c.name === 'Input' || c.name === 'InputLayer') continue
    if (pending && !['Conv2D', 'MaxPooling2D', 'MaxPool2D', 'AveragePooling2D', 'AvgPool2D'].includes(c.name)) return fail('pad_alone', pending.line)
    const pad = (): Padding => {
      if (pending) {
        const p = pending.p
        pending = null
        return p
      }
      return str(c.kw.padding) === 'same' ? 'same' : 0
    }
    switch (c.name) {
      case 'ZeroPadding2D': {
        const p = firstInt(c.kw.padding ?? c.args[0] ?? '1')
        if (!(p >= 0)) return fail('bad_args', c.line)
        pending = { p, line: c.line }
        continue
      }
      case 'Conv2D': {
        const filters = firstInt(c.kw.filters ?? c.args[0])
        const k = firstInt(c.kw.kernel_size ?? c.args[1])
        const stride = firstInt(c.kw.strides ?? c.args[2] ?? '1')
        if (![filters, k, stride].every(pos)) return fail('bad_args', c.line)
        layers.push({ kind: 'conv', filters, k, stride, pad: pad(), bias: c.kw.use_bias !== 'False' })
        break
      }
      case 'MaxPooling2D':
      case 'MaxPool2D':
      case 'AveragePooling2D':
      case 'AvgPool2D': {
        const k = firstInt(c.kw.pool_size ?? c.args[0] ?? '2')
        const stride = firstInt(c.kw.strides ?? c.args[1] ?? String(k))
        if (![k, stride].every(pos)) return fail('bad_args', c.line)
        layers.push({ kind: 'pool', k, stride, pad: pad(), op: c.name.startsWith('Max') ? 'max' : 'avg' })
        break
      }
      case 'GlobalAveragePooling2D':
      case 'GlobalMaxPooling2D':
        layers.push({ kind: 'globalpool', op: c.name.includes('Max') ? 'max' : 'avg' })
        break
      case 'Flatten':
        layers.push({ kind: 'flatten' })
        break
      case 'Dense': {
        const units = firstInt(c.kw.units ?? c.args[0])
        if (!pos(units)) return fail('bad_args', c.line)
        layers.push({ kind: 'dense', units, bias: c.kw.use_bias !== 'False' })
        head = str(c.kw.activation ?? c.args[1]) === 'sigmoid' ? 'regress' : 'classify'
        break
      }
      case 'Dropout': {
        const rate = num(c.kw.rate ?? c.args[0])
        if (!(rate >= 0 && rate < 1)) return fail('bad_args', c.line)
        layers.push({ kind: 'dropout', rate })
        break
      }
    }
    if (layers.length > at) lines.push(padLine !== undefined ? [padLine, c.line] : [c.line])
  }
  if (pending) return fail('pad_alone', (pending as { line: number }).line)
  if (!layers.length) return fail('empty', 1)
  if (!input) return fail('no_input', calls[0]?.line ?? 1)
  return { ok: true, input, layers, head, lines }
}

const TORCH = ['Conv2d', 'MaxPool2d', 'AvgPool2d', 'AdaptiveAvgPool2d', 'AdaptiveMaxPool2d', 'Linear', 'Dropout']

export function parseTorch(code: string): Parsed {
  const fail = (error: ParseError, line: number): Parsed => ({ ok: false, error, line })
  // self.name = nn.Layer(...) in __init__
  const defs = new Map<string, Call>()
  for (const c of findCalls(code, TORCH)) {
    const lineText = code.split('\n')[c.line - 1] ?? ''
    const m = lineText.match(/self\.(\w+)\s*=/)
    if (m) defs.set(m[1], c)
  }
  // the input: "def forward(self, x):  # (batch, C, H, W)" or a torch.randn(1, C, H, W)
  const lines = code.split('\n')
  const fwdAt = lines.findIndex((l) => /def\s+forward\s*\(/.test(l))
  const shapeSrc = (fwdAt >= 0 ? lines[fwdAt].match(/#\s*\(([^)]*)\)/)?.[1] : undefined) ?? code.match(/torch\.(?:randn|rand|zeros|ones)\s*\(([^)]*)\)/)?.[1]
  const dims = ints(shapeSrc)
  const input: Shape | null = dims.length >= 3 ? [dims[dims.length - 2], dims[dims.length - 1], dims[dims.length - 3]] : null
  if (fwdAt < 0) return fail('empty', 1)
  const layers: Layer[] = []
  const spans: number[][] = []
  let head: Head = 'classify'
  for (let i = fwdAt + 1; i < lines.length; i++) {
    const raw = lines[i]
    if (raw.trim() && !/^\s/.test(raw)) break // dedented: the class is over
    const l = raw.replace(/#.*/, '')
    if (!l.trim() || /^\s*return\b/.test(l)) continue
    const used = [...l.matchAll(/self\.(\w+)\s*\(/g)].map((m) => m[1])
    let flatDone = false
    for (const name of used) {
      const c = defs.get(name)
      if (!c) return fail('unknown', i + 1)
      spans.push([c.line, i + 1])
      switch (c.name) {
        case 'Conv2d': {
          const filters = firstInt(c.kw.out_channels ?? c.args[1])
          const k = firstInt(c.kw.kernel_size ?? c.args[2])
          const stride = firstInt(c.kw.stride ?? c.args[3] ?? '1')
          const p = firstInt(c.kw.padding ?? c.args[4] ?? '0')
          if (![filters, k, stride].every(pos) || !(p >= 0)) return fail('bad_args', c.line)
          layers.push({ kind: 'conv', filters, k, stride, pad: p, bias: c.kw.bias !== 'False' })
          break
        }
        case 'MaxPool2d':
        case 'AvgPool2d': {
          const k = firstInt(c.kw.kernel_size ?? c.args[0])
          const stride = firstInt(c.kw.stride ?? c.args[1] ?? String(k))
          const p = firstInt(c.kw.padding ?? c.args[2] ?? '0')
          if (![k, stride].every(pos) || !(p >= 0)) return fail('bad_args', c.line)
          layers.push({ kind: 'pool', k, stride, pad: p, op: c.name === 'MaxPool2d' ? 'max' : 'avg' })
          break
        }
        case 'AdaptiveAvgPool2d':
        case 'AdaptiveMaxPool2d':
          layers.push({ kind: 'globalpool', op: c.name.includes('Max') ? 'max' : 'avg' })
          flatDone = true
          break
        case 'Linear': {
          const units = firstInt(c.kw.out_features ?? c.args[1])
          if (!pos(units)) return fail('bad_args', c.line)
          layers.push({ kind: 'dense', units, bias: c.kw.bias !== 'False' })
          head = /torch\.sigmoid|F\.sigmoid/.test(l) ? 'regress' : 'classify'
          break
        }
        case 'Dropout': {
          const rate = num(c.kw.p ?? c.args[0] ?? '0.5')
          if (!(rate >= 0 && rate < 1)) return fail('bad_args', c.line)
          layers.push({ kind: 'dropout', rate })
          break
        }
      }
    }
    if (!flatDone && /\.view\s*\(|\.reshape\s*\(|torch\.flatten\s*\(|\.flatten\s*\(/.test(l)) {
      layers.push({ kind: 'flatten' })
      spans.push([i + 1])
    }
  }
  if (!layers.length) return fail('empty', fwdAt + 1)
  if (!input) return fail('no_input', fwdAt + 1)
  return { ok: true, input, layers, head, lines: spans }
}
