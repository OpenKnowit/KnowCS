import { describe, expect, it } from 'vitest'
import en from './locales/en.json'
import zh from './locales/zh.json'
import core from './locales/en.json?core'
import PACKS from 'virtual:locale-packs'
import { mergePacks } from './i18nPacks'
// @ts-expect-error -- plain JS build helper
import { packNames } from '../scripts/vite-plugins.mjs'

const flat = (o: unknown, p = ''): string[] =>
  o && typeof o === 'object' ? Object.entries(o).flatMap(([k, v]) => flat(v, p ? `${p}.${k}` : k)) : [p]

describe('locale packs', () => {
  it('core keeps titles of each episode, shared lab strings and formula sheet title', () => {
    const c = core as { watch: Record<string, Record<string, unknown>>; lab: Record<string, unknown>; formulas: Record<string, unknown> }
    expect(Object.keys(c.watch.affine).sort()).toEqual(['kicker', 'sub', 'title'])
    expect(Object.keys(c.lab).sort()).toEqual(['common', 'errors'])
    expect(c.formulas.title).toBe(en.formulas.title)
    expect(JSON.stringify(core).length).toBeLessThan(JSON.stringify(en).length * 0.15)
    for (const k of ['numpy_module', 'drill', 'bayes']) expect(core).not.toHaveProperty(k)
  })
  it('core plus every pack gives back the whole file', async () => {
    const packs = await Promise.all(Object.values(PACKS).map((l) => l.en().then((m) => m.default)))
    expect(flat(mergePacks(core, packs)).sort()).toEqual(flat(en).sort())
  })
  it('has every pack in all three languages, and zh has the same packs as en', () => {
    expect(Object.keys(PACKS).sort()).toEqual([...packNames(en)].sort())
    expect([...packNames(zh)].sort()).toEqual([...packNames(en)].sort())
    for (const name of Object.keys(PACKS)) expect(Object.keys(PACKS[name]).sort()).toEqual(['en', 'zh', 'zh-HK'])
  })
})
