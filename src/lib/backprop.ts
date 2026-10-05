// --- 反向传播（单输出神经元）纯计算：i, j → k，sigmoid 激活，误差 E = ½(T − O)² ---
// 记号与讲义一致：net_k = Σ w_xk·O_x，O_k = f(net_k)，δ_k = (T_k − O_k)·f'(net_k)，Δw_xk = η·δ_k·O_x

export interface NeuronParams {
  oi: number // 输入神经元 i 的输出 O_i
  oj: number // 输入神经元 j 的输出 O_j
  wik: number // 权重 w_ik
  wjk: number // 权重 w_jk
  target: number // 目标 T_k
  eta: number // 学习率 η
}

export interface BackpropTrace {
  net: number // net_k
  out: number // O_k = σ(net_k)
  diff: number // T_k − O_k
  error: number // E = ½(T_k − O_k)²
  fprime: number // f'(net_k) = O_k(1 − O_k)
  delta: number // δ_k = (T_k − O_k)·f'(net_k)
  dwik: number // Δw_ik = η·δ_k·O_i
  dwjk: number // Δw_jk = η·δ_k·O_j
}

export const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x))

/** 前向 + 反向一次，返回推导链上的全部中间量 */
export const traceNeuron = (p: NeuronParams): BackpropTrace => {
  const net = p.wik * p.oi + p.wjk * p.oj
  const out = sigmoid(net)
  const diff = p.target - out
  const fprime = out * (1 - out)
  const delta = diff * fprime
  return {
    net,
    out,
    diff,
    error: 0.5 * diff * diff,
    fprime,
    delta,
    dwik: p.eta * delta * p.oi,
    dwjk: p.eta * delta * p.oj,
  }
}

/** 执行一次梯度下降更新：w ← w + Δw */
export const applyUpdate = (p: NeuronParams): NeuronParams => {
  const t = traceNeuron(p)
  return { ...p, wik: p.wik + t.dwik, wjk: p.wjk + t.dwjk }
}
