import { describe, expect, it } from 'vitest'
import { BAYES_DATA } from '../data/constants'
import type { BayesFeature } from '../types'
import { bayesPosterior, computeNaiveBayes } from './bayes'

const DEFAULT_INPUTS: Record<BayesFeature, string> = {
  BP: 'High',
  Fever: 'No',
  Diabetes: 'Yes',
  Vomit: 'Yes',
}

describe('bayesPosterior', () => {
  it('黄金值：火警案例 P(Fire|Smoke) = 0.01 × 0.9 / 0.1 = 9.00%', () => {
    expect(bayesPosterior(0.01, 0.9, 0.1)).toBeCloseTo(0.09, 10)
  })

  it('似然为 1 且证据概率等于先验时后验为 1', () => {
    expect(bayesPosterior(0.2, 1, 0.2)).toBeCloseTo(1, 10)
  })
})

describe('computeNaiveBayes', () => {
  it('黄金值：默认输入 α=0 → YES 20.46% / NO 79.54%（讲义 P.29）', () => {
    const r = computeNaiveBayes(BAYES_DATA, DEFAULT_INPUTS, 0, false)
    expect(r.yes.prob * 100).toBeCloseTo(20.46, 2)
    expect(r.no.prob * 100).toBeCloseTo(79.54, 2)
    expect(r.yes.prob + r.no.prob).toBeCloseTo(1, 10)
  })

  it('似然链：每类 5 个步骤（先验 + 4 特征），P(BP=High|yes) = 2/9', () => {
    const r = computeNaiveBayes(BAYES_DATA, DEFAULT_INPUTS, 0, false)
    expect(r.yes.steps).toHaveLength(5)
    expect(r.no.steps).toHaveLength(5)
    expect(r.yes.steps[0].val).toBeCloseTo(9 / 14, 10) // Prior
    expect(r.yes.steps[1].val).toBeCloseTo(2 / 9, 10) // P(BP=High|yes)
    expect(r.no.steps[1].val).toBeCloseTo(3 / 5, 10) // P(BP=High|no)
  })

  it('零频率问题：BP=Low 时 P(Low|no)=0，α=0 → NO 概率塌缩为 0', () => {
    const inputs = { ...DEFAULT_INPUTS, BP: 'Low' }
    const r = computeNaiveBayes(BAYES_DATA, inputs, 0, false)
    expect(r.no.raw).toBe(0)
    expect(r.no.prob).toBe(0)
    expect(r.yes.prob).toBeCloseTo(1, 10)
  })

  it('α=1 平滑消除零频率：P(Low|no) = (0+1)/(5+3×1) = 0.125，NO 概率恢复非零', () => {
    const inputs = { ...DEFAULT_INPUTS, BP: 'Low' }
    const r = computeNaiveBayes(BAYES_DATA, inputs, 1, false)
    expect(r.no.steps[1].val).toBeCloseTo(0.125, 10)
    expect(r.no.prob).toBeGreaterThan(0)
    expect(r.yes.prob + r.no.prob).toBeCloseTo(1, 10)
  })

  it('α 平滑公式：(count + α) / (totalCls + m × α)，m 取各特征取值数', () => {
    const r = computeNaiveBayes(BAYES_DATA, DEFAULT_INPUTS, 0.5, false)
    // P(BP=High|yes) = (2 + 0.5) / (9 + 3 × 0.5)
    expect(r.yes.steps[1].val).toBeCloseTo(2.5 / 10.5, 10)
    // P(Diabetes=Yes|no) = (4 + 0.5) / (5 + 2 × 0.5)
    expect(r.no.steps[3].val).toBeCloseTo(4.5 / 6, 10)
  })

  it('对数模式与连乘模式一致：exp(logScore) ≈ raw（无零概率时）', () => {
    const log = computeNaiveBayes(BAYES_DATA, DEFAULT_INPUTS, 0.5, true)
    const prod = computeNaiveBayes(BAYES_DATA, DEFAULT_INPUTS, 0.5, false)
    expect(Math.exp(log.yes.score)).toBeCloseTo(prod.yes.raw, 10)
    expect(Math.exp(log.no.score)).toBeCloseTo(prod.no.raw, 10)
    // 归一化概率不受模式影响
    expect(log.yes.prob).toBeCloseTo(prod.yes.prob, 10)
  })

  it('对数模式下分数为负（概率连乘 < 1）且 YES 分数低于 NO（默认输入）', () => {
    const r = computeNaiveBayes(BAYES_DATA, DEFAULT_INPUTS, 0, true)
    expect(r.yes.score).toBeLessThan(0)
    expect(r.no.score).toBeLessThan(0)
    expect(r.yes.score).toBeLessThan(r.no.score)
  })
})
