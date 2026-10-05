// --- 广播规则讲解：逐轴比对（从右往左）、结果下标 → 操作数下标的映射、修复建议 ---

export type AxisVerdict = 'equal' | 'stretchA' | 'stretchB' | 'mismatch'

export interface AxisStep {
  /** 负数轴号：-1 为最后一维 */
  axis: number
  /** 对齐后的维度长度；左侧补出来的维度为 1 */
  a: number
  b: number
  /** 该维是否为左侧补齐的 1 */
  padA: boolean
  padB: boolean
  verdict: AxisVerdict
  out: number | null
}

export interface BroadcastPlan {
  ndim: number
  /** 按检查顺序排列：从最后一维（axis -1）开始 */
  steps: AxisStep[]
  ok: boolean
  outShape: number[] | null
}

export const planBroadcast = (a: readonly number[], b: readonly number[]): BroadcastPlan => {
  const ndim = Math.max(a.length, b.length)
  const steps: AxisStep[] = []
  for (let k = 1; k <= ndim; k++) {
    const padA = k > a.length
    const padB = k > b.length
    const da = padA ? 1 : a[a.length - k]
    const db = padB ? 1 : b[b.length - k]
    const verdict: AxisVerdict = da === db ? 'equal' : da === 1 ? 'stretchA' : db === 1 ? 'stretchB' : 'mismatch'
    steps.push({ axis: -k, a: da, b: db, padA, padB, verdict, out: verdict === 'mismatch' ? null : Math.max(da, db) })
  }
  const ok = steps.every((s) => s.verdict !== 'mismatch')
  return { ndim, steps, ok, outShape: ok ? steps.map((s) => s.out as number).reverse() : null }
}

/** 结果中的下标 → 操作数（形状 shape，右对齐）中真正被读取的下标 */
export const sourceIndex = (outIdx: readonly number[], shape: readonly number[]): number[] => {
  const pad = outIdx.length - shape.length
  return shape.map((d, k) => (d === 1 ? 0 : outIdx[pad + k]))
}

/** 结果中的该位置是否就是操作数自己的元素（而非广播出来的虚拟副本） */
export const isRealCell = (outIdx: readonly number[], shape: readonly number[]): boolean => {
  const pad = outIdx.length - shape.length
  return outIdx.every((v, k) => (k < pad || shape[k - pad] === 1 ? v === 0 : true))
}

/** np.broadcast_to(x, outShape) 的步长（以元素计）：被拉伸的轴步长为 0，不复制数据 */
export const broadcastStrides = (shape: readonly number[], outShape: readonly number[]): number[] => {
  const own: number[] = new Array(shape.length)
  let acc = 1
  for (let i = shape.length - 1; i >= 0; i--) {
    own[i] = acc
    acc *= shape[i]
  }
  const pad = outShape.length - shape.length
  return outShape.map((d, k) => (k < pad || (shape[k - pad] === 1 && d !== 1) ? 0 : own[k - pad]))
}

export interface BroadcastFix {
  side: 'A' | 'B'
  shape: number[]
  /** 修复写法，如 B[:, None] */
  code: string
}

/** 形状不兼容时，尝试给某个操作数末尾加一维（列向量化），且结果形状等于较大操作数的形状 */
export const suggestFix = (a: readonly number[], b: readonly number[], maxNdim = 3): BroadcastFix | null => {
  if (planBroadcast(a, b).ok) return null
  const candidates: BroadcastFix[] = []
  const tryAdd = (side: 'A' | 'B', own: readonly number[], other: readonly number[]) => {
    if (own.length === 0 || own.length >= maxNdim) return
    const shape = [...own, 1]
    const plan = planBroadcast(side === 'A' ? shape : a, side === 'A' ? b : shape)
    const bigger = own.length >= other.length ? shape : other
    if (plan.ok && plan.outShape!.join() === bigger.join()) {
      candidates.push({ side, shape, code: own.length === 1 ? `${side}[:, None]` : `${side}[..., None]` })
    }
  }
  tryAdd('B', b, a)
  tryAdd('A', a, b)
  return candidates[0] ?? null
}

export const shapeLabel = (shape: readonly number[]): string =>
  shape.length === 0 ? '()' : shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(', ')})`
