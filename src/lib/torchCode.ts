/**
 * The same layer stack as a PyTorch nn.Module, the way lecture 9 writes its MNIST CNN: layers in __init__,
 * F.relu in forward, x.view(x.size(0), -1) to flatten, raw logits out (CrossEntropyLoss applies softmax).
 * PyTorch is channels-first, so the shape comments read (batch, C, H, W).
 */
import { analyse } from './cnnShapes'
import type { Layer, Padding, Shape } from './cnnShapes'
import type { Head } from './keras'

const shapeComment = (s: Shape) => (s.length === 3 ? `(batch, ${s[2]}, ${s[0]}, ${s[1]})` : `(batch, ${s.join(', ')})`)

/**
 * Padding as PyTorch needs it. 'same' becomes k // 2, which matches Keras's ⌈n / s⌉ for odd kernels at any stride
 * (PyTorch's own padding='same' refuses strides above 1). null when no symmetric padding can reproduce it.
 */
function torchPad(pad: Padding, k: number): number | null {
  if (pad === 'valid') return 0
  if (pad === 'same') return k % 2 === 1 ? (k - 1) / 2 : null
  return pad
}

export function torchCode(input: Shape, layers: Layer[], head: Head = 'classify', name = 'Net'): string[] | null {
  const infos = analyse(input, layers)
  if (infos.some((i) => i.error) || input.length !== 3) return null
  const init: string[] = []
  const fwd: string[] = []
  const lastDense = layers.map((l) => l.kind).lastIndexOf('dense')
  const count: Record<string, number> = {}
  const nameOf = (base: string) => {
    count[base] = (count[base] ?? 0) + 1
    return `${base}${count[base]}`
  }
  let flat = false
  let preFlat: Shape | null = null // the C × H × W that the first Linear after a flatten receives
  for (let i = 0; i < layers.length; i++) {
    const l = layers[i]
    const info = infos[i]
    const out = shapeComment(info.output)
    if (l.kind === 'conv') {
      const p = torchPad(l.pad, l.k)
      if (p === null) return null
      const n = nameOf('conv')
      init.push(`self.${n} = nn.Conv2d(in_channels=${info.input[2]}, out_channels=${l.filters}, kernel_size=${l.k}, stride=${l.stride}, padding=${p}${l.bias ? '' : ', bias=False'})`)
      fwd.push(`x = F.relu(self.${n}(x))  # ${out}`)
    } else if (l.kind === 'pool') {
      const p = torchPad(l.pad, l.k)
      if (p === null) return null
      const n = nameOf('pool')
      init.push(`self.${n} = nn.${l.op === 'max' ? 'MaxPool2d' : 'AvgPool2d'}(kernel_size=${l.k}, stride=${l.stride}${p ? `, padding=${p}` : ''})`)
      fwd.push(`x = self.${n}(x)  # ${out}`)
    } else if (l.kind === 'globalpool') {
      const n = nameOf('gap')
      init.push(`self.${n} = nn.${l.op === 'max' ? 'AdaptiveMaxPool2d' : 'AdaptiveAvgPool2d'}(1)`)
      fwd.push(`x = self.${n}(x).view(x.size(0), -1)  # ${out}`)
      flat = true
    } else if (l.kind === 'flatten') {
      fwd.push(`x = x.view(x.size(0), -1)  # ${out}`)
      flat = true
      preFlat = info.input
    } else if (l.kind === 'dense') {
      if (!flat) return null
      const last = i === lastDense
      const units = last && head === 'regress' ? 1 : l.units
      const inN = info.input.reduce((a, b) => a * b, 1)
      const inExpr = preFlat ? `${preFlat[2]} * ${preFlat[0]} * ${preFlat[1]}` : String(inN)
      preFlat = null
      const n = nameOf('fc')
      init.push(`self.${n} = nn.Linear(${inExpr}, ${units}${l.bias ? '' : ', bias=False'})`)
      if (!last) fwd.push(`x = F.relu(self.${n}(x))  # ${shapeComment([units])}`)
      else if (head === 'regress') fwd.push(`x = torch.sigmoid(self.${n}(x))  # ${shapeComment([units])}`)
      else fwd.push(`x = self.${n}(x)  # ${shapeComment([units])} raw logits`)
    } else {
      const n = nameOf('dropout')
      init.push(`self.${n} = nn.Dropout(${l.rate})`)
      fwd.push(`x = self.${n}(x)  # ${out}`)
    }
  }
  return [
    'import torch',
    'import torch.nn as nn',
    'import torch.nn.functional as F',
    '',
    `class ${name}(nn.Module):`,
    '    def __init__(self):',
    '        super().__init__()',
    ...init.map((s) => `        ${s}`),
    '',
    '    def forward(self, x):  # ' + shapeComment(input),
    ...fwd.map((s) => `        ${s}`),
    '        return x',
    '',
    `model = ${name}()`,
    head === 'regress' ? 'lossfunc = nn.MSELoss()' : 'lossfunc = nn.CrossEntropyLoss()  # applies softmax itself',
    'optimizer = torch.optim.Adam(model.parameters(), lr=0.001)',
  ]
}
