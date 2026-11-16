import { describe, expect, it } from 'vitest'
import en from '../locales/en.json'
import zh from '../locales/zh.json'
import { KERAS, STEPS, TORCH } from './kerasTorch'

describe('Keras ↔ PyTorch map', () => {
  it('every step has a line on both sides', () => {
    for (const s of STEPS) {
      expect(KERAS.some(([, ...t]) => t.includes(s)), `Keras ${s}`).toBe(true)
      expect(TORCH.some(([, ...t]) => t.includes(s)), `PyTorch ${s}`).toBe(true)
    }
  })
  it('only empty lines are untagged', () => {
    for (const [code, ...t] of [...KERAS, ...TORCH]) expect(t.length > 0).toBe(code !== '')
  })
  it('every step is explained in English and Chinese', () => {
    const maps = [en.lab.pytorch.map, zh.lab.pytorch.map] as Record<string, { name: string; text: string }>[]
    for (const m of maps) for (const s of STEPS) expect(m[s]?.name && m[s]?.text, s).toBeTruthy()
  })
})
