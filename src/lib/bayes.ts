import type { BayesClass, BayesData, BayesFeature, NaiveBayesResults } from '../types'

// --- 贝叶斯纯计算逻辑（从 BayesBasicsModule / NaiveBayesModule 提取，行为保持不变） ---

const CLASSES: BayesClass[] = ['yes', 'no']

/** 贝叶斯后验：P(B|E) = P(B) × P(E|B) / P(E) */
export const bayesPosterior = (pB: number, pEGivenB: number, pE: number): number =>
  (pB * pEGivenB) / pE

/** 朴素贝叶斯推断：先验 × 各特征似然（含 α 平滑 / m-估计），支持连乘与对数求和两种模式 */
export const computeNaiveBayes = (
  data: BayesData,
  inputs: Record<BayesFeature, string>,
  alpha: number,
  useLog: boolean
): NaiveBayesResults => {
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
      const totalCls = cls === 'yes' ? 9 : 5
      const m = data.m_values[feat]
      const prob = (count + alpha) / (totalCls + m * alpha)
      results[cls].steps.push({
        name: feat, val: prob, label: `P(${feat}|${cls})`,
        formula: `\\frac{${count} + ${alpha}}{${totalCls} + ${m} \\times ${alpha}}`,
      })
      likelihoodProduct *= prob
      logSum += Math.log(prob || 1e-10)
    })
    results[cls].raw = prior * likelihoodProduct
    results[cls].score = useLog ? logSum : results[cls].raw
  })
  const totalRaw = results.yes.raw + results.no.raw
  results.yes.prob = totalRaw > 0 ? results.yes.raw / totalRaw : 0
  results.no.prob = totalRaw > 0 ? results.no.raw / totalRaw : 0
  return results
}
