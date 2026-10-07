/** Classification metrics from a confusion matrix m[actual][predicted] (lecture 3 "Model Evaluation"). */

export interface ConfusionMatrix {
  classes: string[]
  m: number[][]
}

const splitLabels = (s: string) => s.split(/[\s,]+/).map((t) => t.trim()).filter(Boolean)

export function confusionFromLabels(actual: string, predicted: string): ConfusionMatrix {
  const a = splitLabels(actual)
  const p = splitLabels(predicted)
  if (a.length !== p.length) throw new Error(`Actual has ${a.length} labels, predicted has ${p.length}.`)
  if (!a.length) throw new Error('No labels yet.')
  const numeric = [...a, ...p].every((t) => !Number.isNaN(Number(t)))
  const classes = [...new Set([...a, ...p])].sort((x, y) => (numeric ? Number(x) - Number(y) : x.localeCompare(y)))
  const m = classes.map(() => classes.map(() => 0))
  a.forEach((t, i) => m[classes.indexOf(t)][classes.indexOf(p[i])]++)
  return { classes, m }
}

export interface ClassScores {
  TP: number
  FN: number
  FP: number
  TN: number
  precision: number
  recall: number
  f1: number
  support: number
}

export interface Metrics {
  perClass: ClassScores[]
  total: number
  accuracy: number
  macroF1: number
  weightedF1: number
  /** Only for two classes, treating class 0 as positive. */
  mcc: number | null
}

export function classificationMetrics(m: number[][]): Metrics {
  const total = m.flat().reduce((a, b) => a + b, 0)
  const perClass = m.map((row, i) => {
    const TP = row[i]
    const FN = row.reduce((a, b) => a + b, 0) - TP
    const FP = m.reduce((a, r) => a + r[i], 0) - TP
    const TN = total - TP - FN - FP
    return {
      TP,
      FN,
      FP,
      TN,
      precision: TP + FP ? TP / (TP + FP) : NaN,
      recall: TP + FN ? TP / (TP + FN) : NaN,
      f1: 2 * TP + FP + FN ? (2 * TP) / (2 * TP + FP + FN) : NaN,
      support: TP + FN,
    }
  })
  const correct = perClass.reduce((a, c) => a + c.TP, 0)
  const f1s = perClass.map((c) => (Number.isFinite(c.f1) ? c.f1 : 0))
  let mcc: number | null = null
  if (m.length === 2) {
    const { TP, FN, FP, TN } = perClass[0]
    const den = Math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = den ? (TP * TN - FP * FN) / den : 0
  }
  return {
    perClass,
    total,
    accuracy: total ? correct / total : NaN,
    macroF1: f1s.reduce((a, b) => a + b, 0) / m.length,
    weightedF1: total ? perClass.reduce((a, c, i) => a + f1s[i] * c.support, 0) / total : NaN,
    mcc,
  }
}
