import type { PerceptronPoint, PerceptronStep } from '../types'

/** 阶跃激活：sign(w·x + b)，约定 0 归为 +1。 */
export function predict(w: [number, number], b: number, p: PerceptronPoint): number {
  const net = w[0] * p.x + w[1] * p.y + b
  return net >= 0 ? 1 : -1
}

/**
 * 单神经元 / 感知机的错误驱动学习（Rosenblatt 规则）。
 * 逐点遍历多个 epoch，仅在分错时更新：w += η·y·x, b += η·y。
 * 返回每一次「看一个样本」的步骤，便于动画展示。
 */
export function trainPerceptron(
  data: PerceptronPoint[],
  lr = 0.1,
  epochs = 10,
  w0: [number, number] = [0, 0],
  b0 = 0
): PerceptronStep[] {
  const steps: PerceptronStep[] = []
  let w: [number, number] = [...w0]
  let b = b0

  for (let e = 0; e < epochs; e++) {
    let errors = 0
    for (let i = 0; i < data.length; i++) {
      const p = data[i]
      const yhat = predict(w, b, p)
      const correct = yhat === p.label
      if (!correct) {
        errors++
        w = [w[0] + lr * p.label * p.x, w[1] + lr * p.label * p.y]
        b = b + lr * p.label
      }
      steps.push({
        w: [...w] as [number, number],
        b,
        pointIndex: i,
        predicted: yhat,
        target: p.label,
        correct,
        errors,
      })
    }
    // 一整个 epoch 无错分 → 已线性分开，提前停止
    if (errors === 0) break
  }
  return steps
}

/** 当前权重在整份数据上的错分数量。 */
export function countErrors(data: PerceptronPoint[], w: [number, number], b: number): number {
  return data.reduce((s, p) => s + (predict(w, b, p) === p.label ? 0 : 1), 0)
}
