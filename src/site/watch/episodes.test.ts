import { describe, expect, it } from 'vitest'
import en from '../../locales/en.json'
import { EPISODES } from '../../lib/sitemap'
import { EPISODE_CODE } from './episodes'

type Tree = Record<string, unknown>
const get = (o: unknown, path: string): unknown => path.split('.').reduce<unknown>((a, k) => (a && typeof a === 'object' ? (a as Tree)[k] : undefined), o)

describe('explainer episodes', () => {
  it('every listed episode has code and every coded episode is listed', () => {
    expect(EPISODES.map((e) => e.id).sort()).toEqual(Object.keys(EPISODE_CODE).sort())
  })

  for (const ep of Object.values(EPISODE_CODE))
    it(`${ep.id}: captions match cues, titles and ponder questions exist`, () => {
      expect(typeof get(en, `watch.${ep.id}.title`)).toBe('string')
      expect(typeof get(en, `watch.${ep.id}.exam.q`)).toBe('string')
      for (const s of ep.scenes) {
        const node = get(en, `watch.${ep.id}.scenes.${s.id}`) as Tree | undefined
        expect(node, `${ep.id}.${s.id}`).toBeTruthy()
        expect(typeof node!.title, `${ep.id}.${s.id}.title`).toBe('string')
        const caps = Object.keys(node!).filter((k) => /^c\d+$/.test(k))
        expect(caps.length, `${ep.id}.${s.id} captions`).toBe(s.cues.length)
        expect(s.cues[0]).toBe(0)
        for (let i = 1; i < s.cues.length; i++) expect(s.cues[i]).toBeGreaterThan(s.cues[i - 1])
        if (s.ponder) expect(typeof node!.ponder, `${ep.id}.${s.id}.ponder`).toBe('string')
      }
    })
})
