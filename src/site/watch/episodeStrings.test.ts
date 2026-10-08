import { describe, expect, it } from 'vitest'
import en from '../../locales/en.json'
import core from '../../locales/en.json?core'
import EP_STRINGS from 'virtual:episode-strings'
import { mergeEpisodeStrings } from './episodeStrings'

const flat = (o: unknown, p = ''): string[] =>
  o && typeof o === 'object' ? Object.entries(o).flatMap(([k, v]) => flat(v, p ? `${p}.${k}` : k)) : [p]

describe('locale split', () => {
  it('core keeps only the titles of each episode', () => {
    const w = (core as { watch: Record<string, Record<string, unknown>> }).watch
    expect(Object.keys(w.affine).sort()).toEqual(['kicker', 'sub', 'title'])
    expect(w.ui).toEqual(en.watch.ui)
    expect(JSON.stringify(core).length).toBeLessThan(JSON.stringify(en).length * 0.6)
  })
  it('core plus every episode chunk gives back the whole file', async () => {
    const eps = await Promise.all(Object.values(EP_STRINGS).map((l) => l.en().then((m) => m.default as { watch: Record<string, Record<string, unknown>> })))
    expect(flat(mergeEpisodeStrings(core, eps)).sort()).toEqual(flat(en).sort())
  })
  it('has a loader for every episode in all three languages', () => {
    const ids = Object.keys(en.watch).filter((k) => k !== 'ui')
    expect(Object.keys(EP_STRINGS).sort()).toEqual(ids.sort())
    for (const id of ids) expect(Object.keys(EP_STRINGS[id]).sort()).toEqual(['en', 'zh', 'zh-HK'])
  })
})
