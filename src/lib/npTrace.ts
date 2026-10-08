// --- API 可视化：对每次调用算出「结果的每个元素由哪些输入元素决定」 ---
import { NDArray, broadcastTo, cStrides, normAxis, prod, unravel } from './ndarray'
import { sortAlong } from './ndops'
import type { CallTrace, GridSnapshot } from './minipy'

/** [第几个操作数, 该操作数里的扁平下标] */
export type Cell = [number, number]

export interface Explain {
  /** sources[o]：结果第 o 个元素依赖的输入元素 */
  sources: Cell[][]
  /** 规约取最值 / 排序时，真正被选中的那个输入元素 */
  pick?: (Cell | null)[]
  /** 规约 / 排序 / 累加实际使用的轴（已归一化；null = 展平后整体处理） */
  axis: number | null
}

const flatOf = (idx: number[], shape: readonly number[]): number => {
  const st = cStrides(shape)
  return idx.reduce((s, v, k) => s + v * st[k], 0)
}

const toNd = (g: GridSnapshot): NDArray => NDArray.create(g.values, g.shape, g.dtype)
const outSize = (c: CallTrace): number => (c.result ? prod(c.result.shape) : 0)
const method = (api: string): string => api.split('.').pop() ?? api

function elementwise(c: CallTrace): Explain {
  const shape = c.result?.shape ?? []
  const n = outSize(c)
  const sources: Cell[][] = Array.from({ length: n }, () => [])
  c.operands.forEach((op, k) => {
    try {
      const ids = NDArray.create(op.snap.values.map((_, f) => f), op.snap.shape, 'int64')
      broadcastTo(ids, shape).values().forEach((f, o) => sources[o].push([k, f]))
    } catch {
      /* operand does not broadcast to the result (e.g. astype dtype argument) — skip it */
    }
  })
  return { sources, axis: null }
}

function reduceLike(c: CallTrace): Explain {
  const input = c.operands[0]
  if (!input || !c.result) return { sources: [], axis: null }
  const vals = input.snap.values
  const n = Math.max(1, outSize(c))
  const sources: Cell[][] = Array.from({ length: n }, () => [])
  let axis: number | null = null
  if (c.axis === null || input.snap.shape.length === 0) {
    vals.forEach((_, f) => sources[0].push([0, f]))
  } else {
    axis = normAxis(c.axis, input.snap.shape.length)
    const outShape = input.snap.shape.filter((_, k) => k !== axis)
    vals.forEach((_, f) => {
      const idx = unravel(f, input.snap.shape)
      idx.splice(axis!, 1)
      sources[flatOf(idx, outShape)].push([0, f])
    })
  }
  const op = method(c.api)
  let pick: (Cell | null)[] | undefined
  if (['max', 'min', 'argmax', 'argmin'].includes(op)) {
    const better = op.endsWith('max') ? (x: number, y: number) => x > y : (x: number, y: number) => x < y
    pick = sources.map((group) => group.reduce<Cell | null>((best, cell) => (best === null || better(vals[cell[1]], vals[best[1]]) ? cell : best), null))
  }
  return { sources, pick, axis }
}

function scan(c: CallTrace): Explain {
  const input = c.operands[0]
  const n = outSize(c)
  if (!input) return { sources: [], axis: null }
  if (c.axis === null || input.snap.shape.length <= 1) {
    return { sources: Array.from({ length: n }, (_, o) => Array.from({ length: o + 1 }, (_, f) => [0, f] as Cell)), axis: null }
  }
  const shape = input.snap.shape
  const axis = normAxis(c.axis, shape.length)
  const st = cStrides(shape)
  return {
    sources: Array.from({ length: n }, (_, o) => {
      const i = unravel(o, shape)[axis]
      return Array.from({ length: i + 1 }, (_, t) => [0, o - (i - t) * st[axis]] as Cell)
    }),
    axis,
  }
}

function matmul(c: CallTrace): Explain {
  const [A, B] = c.operands
  const n = Math.max(1, outSize(c))
  if (!A || !B) return { sources: [], axis: null }
  const as = A.snap.shape
  const bs = B.snap.shape
  if (method(c.api) === 'outer') {
    const m = prod(bs)
    return { sources: Array.from({ length: n }, (_, o) => [[0, Math.floor(o / m)], [1, o % m]] as Cell[]), axis: null }
  }
  // row i of A (or all of a 1-D A) and column j of B (or all of a 1-D B)
  const row = (i: number): Cell[] => (as.length === 2 ? Array.from({ length: as[1] }, (_, t) => [0, i * as[1] + t] as Cell) : Array.from({ length: as[0] }, (_, t) => [0, t] as Cell))
  const col = (j: number): Cell[] => (bs.length === 2 ? Array.from({ length: bs[0] }, (_, t) => [1, t * bs[1] + j] as Cell) : Array.from({ length: bs[0] }, (_, t) => [1, t] as Cell))
  if (as.length === 2 && bs.length === 2) return { sources: Array.from({ length: n }, (_, o) => [...row(Math.floor(o / bs[1])), ...col(o % bs[1])]), axis: null }
  if (as.length === 2 && bs.length === 1) return { sources: Array.from({ length: n }, (_, i) => [...row(i), ...col(0)]), axis: null }
  if (as.length === 1 && bs.length === 2) return { sources: Array.from({ length: n }, (_, j) => [...row(0), ...col(j)]), axis: null }
  if (as.length === 1 && bs.length === 1) return { sources: [[...row(0), ...col(0)]], axis: null }
  return elementwise(c) // a scalar operand: plain multiplication
}

function sorted(c: CallTrace): Explain {
  const input = c.operands[0]
  if (!input) return { sources: [], axis: null }
  const axis = c.axis === null ? null : normAxis(c.axis, Math.max(1, input.snap.shape.length))
  const { source } = sortAlong(toNd(input.snap), axis)
  const cells = source.map((f) => [0, f] as Cell)
  return { sources: cells.map((cell) => [cell]), pick: cells, axis }
}

function uniqueLike(c: CallTrace): Explain {
  const input = c.operands[0]
  if (!input || !c.result) return { sources: [], axis: null }
  return {
    sources: c.result.values.map((v) => input.snap.values.flatMap((x, f) => (Object.is(x, v) || x === v ? [[0, f] as Cell] : []))),
    axis: null,
  }
}

function linalg(c: CallTrace): Explain {
  const input = c.operands[0]
  if (!input) return { sources: [], axis: null }
  const n = Math.max(1, outSize(c))
  if (method(c.api) === 'trace') {
    const [r, k] = input.snap.shape
    return { sources: [Array.from({ length: Math.min(r, k) }, (_, i) => [0, i * k + i] as Cell)], axis: null }
  }
  const all = input.snap.values.map((_, f) => [0, f] as Cell)
  return { sources: Array.from({ length: n }, () => all), axis: null }
}

export function explainCall(c: CallTrace): Explain {
  // the function worked out its own provenance (bincount, histogram, pad …)
  if (c.groups) return { sources: c.groups, axis: null }
  switch (c.kind) {
    case 'move':
      return { sources: (c.source ?? []).map((cell) => [cell]), axis: c.axis }
    case 'elementwise':
      return elementwise(c)
    case 'reduce':
      return reduceLike(c)
    case 'scan':
      return scan(c)
    case 'matmul':
      return matmul(c)
    case 'sort':
      return sorted(c)
    case 'unique':
      return uniqueLike(c)
    case 'linalg':
      return linalg(c)
    default:
      return { sources: [], axis: null }
  }
}

/** 反向索引：每个输入元素影响了结果的哪些元素 */
export function dependents(e: Explain, operandCount: number): number[][][] {
  const out: number[][][] = Array.from({ length: operandCount }, () => [])
  e.sources.forEach((cells, o) =>
    cells.forEach(([k, f]) => {
      ;(out[k][f] ??= []).push(o)
    }),
  )
  return out
}
