/**
 * Output shapes and parameter counts for a stack of CNN layers (lecture 8, Keras conventions, channels last).
 *   conv / pool output = ⌊(n − k + 2p) / s⌋ + 1 ;  'same' padding → ⌈n / s⌉
 *   conv params = (k·k·C_in + 1)·filters (no +1 without bias) ; dense params = (in + 1)·units
 */

export type Padding = number | 'same' | 'valid'

export type Layer =
  | { kind: 'conv'; filters: number; k: number; stride: number; pad: Padding; bias: boolean }
  | { kind: 'pool'; k: number; stride: number; pad: Padding; op: 'max' | 'avg' }
  | { kind: 'globalpool'; op: 'max' | 'avg' }
  | { kind: 'flatten' }
  | { kind: 'dense'; units: number; bias: boolean }
  | { kind: 'dropout'; rate: number }

export type Shape = number[] // [h, w, c] or [n]

export interface LayerInfo {
  layer: Layer
  input: Shape
  output: Shape
  params: number
  /** how the spatial size was computed, e.g. "⌊(224 − 7 + 2·3) / 2⌋ + 1 = 112" */
  sizeWork: string | null
  paramWork: string | null
  error: string | null
  /** true when (n − k + 2p) is not divisible by the stride, so the last pixels are skipped */
  floored: boolean
}

function spatial(n: number, k: number, s: number, pad: Padding): { size: number; work: string; floored: boolean } {
  if (pad === 'same') {
    const size = Math.ceil(n / s)
    return { size, work: `'same': ⌈${n} / ${s}⌉ = ${size}`, floored: false }
  }
  const p = pad === 'valid' ? 0 : pad
  const span = n - k + 2 * p
  const size = Math.floor(span / s) + 1
  return { size, work: `⌊(${n} − ${k} + 2·${p}) / ${s}⌋ + 1 = ${size}`, floored: span % s !== 0 }
}

export function analyse(input: Shape, layers: Layer[]): LayerInfo[] {
  const out: LayerInfo[] = []
  let cur = input
  for (const layer of layers) {
    const base = { layer, input: cur, sizeWork: null, paramWork: null, error: null, floored: false }
    let info: LayerInfo
    if (layer.kind === 'conv' || layer.kind === 'pool') {
      if (cur.length !== 3) {
        info = { ...base, output: cur, params: 0, error: 'Needs an h × w × c input (place it before Flatten).' }
      } else {
        const [h, w, c] = cur
        const sh = spatial(h, layer.k, layer.stride, layer.pad)
        const sw = spatial(w, layer.k, layer.stride, layer.pad)
        const ch = layer.kind === 'conv' ? layer.filters : c
        if (sh.size < 1 || sw.size < 1) {
          info = { ...base, output: [Math.max(0, sh.size), Math.max(0, sw.size), ch], params: 0, error: 'Kernel is larger than the (padded) input.' }
        } else if (layer.kind === 'conv') {
          const per = layer.k * layer.k * c + (layer.bias ? 1 : 0)
          info = {
            ...base,
            output: [sh.size, sw.size, ch],
            params: per * layer.filters,
            sizeWork: h === w ? sh.work : `${sh.work} × ${sw.work}`,
            paramWork: `(${layer.k}·${layer.k}·${c}${layer.bias ? ' + 1' : ''}) · ${layer.filters} = ${(per * layer.filters).toLocaleString('en-US')}`,
            floored: sh.floored || sw.floored,
          }
        } else {
          info = { ...base, output: [sh.size, sw.size, ch], params: 0, sizeWork: h === w ? sh.work : `${sh.work} × ${sw.work}`, floored: sh.floored || sw.floored }
        }
      }
    } else if (layer.kind === 'globalpool') {
      info = cur.length === 3 ? { ...base, output: [cur[2]], params: 0, sizeWork: `${cur[0]}×${cur[1]} → 1 per channel` } : { ...base, output: cur, params: 0, error: 'Needs an h × w × c input.' }
    } else if (layer.kind === 'flatten') {
      const n = cur.reduce((a, b) => a * b, 1)
      info = { ...base, output: [n], params: 0, sizeWork: `${cur.join(' × ')} = ${n.toLocaleString('en-US')}` }
    } else if (layer.kind === 'dense') {
      if (cur.length !== 1) {
        info = { ...base, output: cur, params: 0, error: 'Dense expects a flat vector — add Flatten first.' }
      } else {
        const params = (cur[0] + (layer.bias ? 1 : 0)) * layer.units
        info = { ...base, output: [layer.units], params, paramWork: `(${cur[0].toLocaleString('en-US')}${layer.bias ? ' + 1' : ''}) · ${layer.units} = ${params.toLocaleString('en-US')}` }
      }
    } else {
      info = { ...base, output: cur, params: 0 }
    }
    out.push(info)
    cur = info.output
  }
  return out
}

export const totalParams = (infos: LayerInfo[]): number => infos.reduce((a, l) => a + l.params, 0)
