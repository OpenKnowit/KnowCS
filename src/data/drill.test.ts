import { describe, expect, it } from 'vitest'
import en from '../locales/en.json'
import { DRILL } from './drill'
import { EPISODES, MODULES } from '../lib/sitemap'
import { planBroadcast } from '../lib/broadcast'
import { analyse, totalParams } from '../lib/cnnShapes'
import { outputSize, dilatedConv } from '../lib/conv2d'
import { gaussianPdf } from '../lib/gaussianNb'
import { classificationMetrics } from '../lib/metrics'
import { crossValidate, sortedLabels } from '../lib/crossval'
import { generateAbSteps, leavesEvaluated, minimaxValue, treeWithLeaves } from '../lib/alphabeta'

const by = (id: string) => DRILL.find((d) => d.id === id)!.answer

describe('concept-check drill', () => {
  it('every item has text and a link that exists', () => {
    const items = (en as { drill: { items: Record<string, { q: string; why: string }> } }).drill.items
    for (const d of DRILL) {
      expect(items[d.id]?.q, d.id).toBeTruthy()
      expect(items[d.id]?.why, d.id).toBeTruthy()
      if (d.link.kind === 'watch') expect(EPISODES.some((e) => e.id === d.link.id), d.id).toBe(true)
      else expect(MODULES.some((m) => m.id === d.link.id), d.id).toBe(true)
    }
    expect(new Set(DRILL.map((d) => d.id)).size).toBe(DRILL.length)
  })

  it('answers that can be computed agree with the libraries', () => {
    expect(by('np2')).toBe(planBroadcast([4, 1], [3]).ok && planBroadcast([4, 1], [3]).outShape?.join() === '4,3')
    expect(by('np3')).toBe(planBroadcast([2, 3], [2]).ok)
    const net = analyse([3], [{ kind: 'dense', units: 4, bias: true }, { kind: 'dense', units: 2, bias: true }])
    expect(by('ml3')).toBe(totalParams(net) === 26)
    expect(by('cn2')).toBe(outputSize(32, 5, 2, 1).size === 32)
    const conv = analyse([32, 32, 3], [{ kind: 'conv', filters: 10, k: 5, stride: 1, pad: 2, bias: true }])
    const conv64 = analyse([64, 64, 3], [{ kind: 'conv', filters: 10, k: 5, stride: 1, pad: 2, bias: true }])
    expect(by('cn1')).toBe(conv[0].params !== conv64[0].params)
    // Final 2024 Q1(g): 36×36×8 outputs, each a 3×3×3 dot product
    const c8 = analyse([32, 32, 3], [{ kind: 'conv', filters: 8, k: 3, stride: 1, pad: 3, bias: true }])[0]
    expect(c8.output).toEqual([36, 36, 8])
    expect(by('cn6')).toBe(8 * 3 * 3 * 36 * 36 === c8.output[0] * c8.output[1] * c8.output[2] * 3 * 3 * 3)
    expect(by('cn4')).toBe(dilatedConv([[0]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 2).effective === 5)
    expect(by('by5')).toBe(gaussianPdf(0, 0, 0.1) > 1)
    expect(by('ev1')).toBe(classificationMetrics([[0, 5], [0, 95]]).accuracy === 0.95 && classificationMetrics([[0, 5], [0, 95]]).perClass[0].recall === 0)
    expect(by('ev3')).toBe(crossValidate(sortedLabels(4, 20), 2, 'sorted').mean === 0)
    const worst = treeWithLeaves([0, 4, 1, 2, 8, 9, 6, 7])
    expect(by('ab2')).toBe(leavesEvaluated(generateAbSteps(worst)) < 8)
    expect(by('ab1')).toBe(minimaxValue(worst) !== generateAbSteps(worst).filter((s) => s.type === 'exit' && s.nodeId === 'A').at(-1)?.returnValue)
  })
})
