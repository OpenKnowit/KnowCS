import { describe, expect, it } from 'vitest'
import katex from 'katex'
import en from '../locales/en.json'
import { FORMULAS } from './formulas'
import { EPISODES, MODULES } from '../lib/sitemap'

describe('formula sheet', () => {
  const items = (en as { formulas: { items: Record<string, { name: string; note: string }> } }).formulas.items
  for (const f of FORMULAS)
    it(`${f.id} typesets and is described`, () => {
      expect(() => katex.renderToString(f.tex, { throwOnError: true, displayMode: true })).not.toThrow()
      expect(items[f.id]?.name).toBeTruthy()
      expect(items[f.id]?.note).toBeTruthy()
      if (f.link?.kind === 'watch') expect(EPISODES.some((e) => e.id === f.link!.id)).toBe(true)
      if (f.link?.kind === 'module') expect(MODULES.some((m) => m.id === f.link!.id)).toBe(true)
    })
})
