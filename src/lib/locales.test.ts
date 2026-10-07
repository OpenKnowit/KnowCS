import { describe, expect, it } from 'vitest'
import en from '../locales/en.json'
import zh from '../locales/zh.json'
import zhHK from '../locales/zh-HK.json'

const keys = (o: unknown, p = ''): string[] =>
  o && typeof o === 'object' && !Array.isArray(o)
    ? Object.entries(o as Record<string, unknown>).flatMap(([k, v]) => (v && typeof v === 'object' && !Array.isArray(v) ? keys(v, `${p}${k}.`) : [`${p}${k}`]))
    : []

describe('locale files', () => {
  const E = new Set(keys(en))
  it('en and zh have exactly the same keys', () => {
    const Z = new Set(keys(zh))
    expect([...E].filter((k) => !Z.has(k))).toEqual([])
    expect([...Z].filter((k) => !E.has(k))).toEqual([])
  })
  it('zh-HK (generated) has every key', () => {
    const H = new Set(keys(zhHK))
    expect([...E].filter((k) => !H.has(k))).toEqual([])
  })
  it('every explainer caption key exists in English', () => {
    // watch.<ep>.scenes.<scene>.c<k> must be dense: c1, c2, … with no gaps
    for (const k of E) {
      const m = k.match(/^watch\.(\w+)\.scenes\.(\w+)\.c(\d+)$/)
      if (m && Number(m[3]) > 1) expect(E.has(`watch.${m[1]}.scenes.${m[2]}.c${Number(m[3]) - 1}`)).toBe(true)
    }
  })
})
