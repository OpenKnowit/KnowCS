import type { BayesClass, BayesData, BayesFeature, NaiveBayesResults } from '../types'

// --- 贝叶斯纯计算逻辑（BayesBasicsModule / NaiveBayesModule） ---

const CLASSES: BayesClass[] = ['yes', 'no']

/** 贝叶斯后验：P(B|E) = P(B) × P(E|B) / P(E) */
export const bayesPosterior = (pB: number, pEGivenB: number, pE: number): number =>
  (pB * pEGivenB) / pE

/**
 * 证据概率的下界：由全概率公式 P(E) = P(E|B)P(B) + P(E|¬B)P(¬B) ≥ P(E|B)P(B)。
 * P(E) 低于该值时三个输入自相矛盾（后验会超过 100%），可视化需据此限制滑块。
 */
export const minEvidence = (pB: number, pEGivenB: number): number => pB * pEGivenB

/** 每个类别的训练样本数：任取一个特征，把该类在各取值上的计数相加。 */
export const classTotals = (data: BayesData): Record<BayesClass, number> => {
  const anyFeature = Object.keys(data.counts)[0] as BayesFeature
  const totals: Record<BayesClass, number> = { yes: 0, no: 0 }
  for (const counts of Object.values(data.counts[anyFeature])) {
    for (const cls of CLASSES) totals[cls] += counts[cls]
  }
  return totals
}

/**
 * 朴素贝叶斯推断：先验 × 各特征似然（含 α 平滑 / m-估计），支持连乘与对数求和两种模式。
 * 对数模式下 log(0) = -Infinity 如实保留——这正是零频率问题，α > 0 才能消除。
 */
export const computeNaiveBayes = (
  data: BayesData,
  inputs: Record<BayesFeature, string>,
  alpha: number,
  useLog: boolean
): NaiveBayesResults => {
  const a = Number.isFinite(alpha) ? Math.max(0, alpha) : 0
  const totals = classTotals(data)
  const results: NaiveBayesResults = {
    yes: { score: 0, raw: 1, steps: [], prob: 0 },
    no: { score: 0, raw: 1, steps: [], prob: 0 },
  }
  CLASSES.forEach((cls) => {
    const prior = data.priors[cls]
    results[cls].steps.push({ name: 'Prior', val: prior, label: cls === 'yes' ? 'P(Z=Yes)' : 'P(Z=No)' })
    let likelihoodProduct = 1
    let logSum = Math.log(prior)
    ;(Object.entries(inputs) as [BayesFeature, string][]).forEach(([feat, val]) => {
      const count = data.counts[feat][val][cls]
      const totalCls = totals[cls]
      const m = data.m_values[feat]
      const prob = (count + a) / (totalCls + m * a)
      results[cls].steps.push({
        name: feat, val: prob, label: `P(${feat}|${cls})`,
        formula: `\\frac{${count} + ${a}}{${totalCls} + ${m} \\times ${a}}`,
      })
      likelihoodProduct *= prob
      logSum += Math.log(prob)
    })
    results[cls].raw = prior * likelihoodProduct
    results[cls].score = useLog ? logSum : results[cls].raw
  })
  const totalRaw = results.yes.raw + results.no.raw
  results.yes.prob = totalRaw > 0 ? results.yes.raw / totalRaw : 0
  results.no.prob = totalRaw > 0 ? results.no.raw / totalRaw : 0
  return results
}
