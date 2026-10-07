import { describe, expect, it } from 'vitest'
import { LAB_REDIRECTS, MODULES, PAGES, legacyTarget, neighbours } from './sitemap'

describe('legacyTarget', () => {
  it('sends old course hashes to the module pages', () => {
    expect(legacyTarget('#/course/knn')).toBe('/knn/')
    expect(legacyTarget('#/course/bayesBasics')).toBe('/bayes-basics/')
    expect(legacyTarget('#/course/naiveBayes')).toBe('/naive-bayes/')
  })

  it('sends package and extend hashes to notes and extend pages', () => {
    expect(legacyTarget('#/package/numpy')).toBe('/notes/numpy/')
    expect(legacyTarget('#/package')).toBe('/notes/')
    expect(legacyTarget('#/package/nope')).toBe('/notes/')
    expect(legacyTarget('#/extend/attention')).toBe('/extend/attention/')
  })

  it('stays on the home page for empty, unknown or broken hashes', () => {
    expect(legacyTarget('')).toBeNull()
    expect(legacyTarget('#/course/perceptron-old')).toBeNull()
    expect(legacyTarget('#/course/%')).toBeNull()
    expect(legacyTarget('#section-2')).toBeNull()
  })
})

describe('PAGES', () => {
  it('has one page per module, unique output paths', () => {
    const paths = PAGES.map((p) => p.path)
    expect(new Set(paths).size).toBe(paths.length)
    for (const m of MODULES) expect(paths).toContain(`${m.id}/index.html`)
  })

  it('redirects every old lab prototype URL', () => {
    expect(LAB_REDIRECTS['xor-mlp']).toBe('/xor-mlp/')
    expect(LAB_REDIRECTS['home-a']).toBe('/')
    expect(PAGES.filter((p) => p.kind === 'redirect')).toHaveLength(14)
  })
})

describe('neighbours', () => {
  it('links modules in reading order and stops at the ends', () => {
    expect(neighbours('numpy').prev).toBeNull()
    expect(neighbours('numpy').next?.id).toBe('bayes-basics')
    expect(neighbours('alphabeta').next).toBeNull()
  })

  it('keeps lectures in non-decreasing order', () => {
    MODULES.forEach((m, i) => i && expect(m.lec).toBeGreaterThanOrEqual(MODULES[i - 1].lec))
  })
})
