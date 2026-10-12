import { describe, expect, it } from 'vitest'
import en from '../../locales/en.json'
import zh from '../../locales/zh.json'
import { locate, narratedTiming, piecewise, timeline } from '../../lib/explainer'
import { EPISODE_CODE } from './episodes'
import { narrate, narrationLang, remapTime } from './narration'
import type { NarrationManifest } from './narration'
import type { Episode } from './Player'

type Tree = Record<string, unknown>
const get = (o: unknown, path: string): unknown => path.split('.').reduce<unknown>((a, k) => (a && typeof a === 'object' ? (a as Tree)[k] : undefined), o)
type Line = string | { text: string; say?: string }
const scripts = import.meta.glob<Record<string, Record<string, Line[]>>>('../../content/narration/*.json', { eager: true, import: 'default' })

const toy: Episode = {
  id: 'toy',
  scenes: [
    { id: 'a', dur: 4000, cues: [0, 0.5], render: (p) => p },
    { id: 'b', dur: 2000, cues: [0], ponder: true, render: (p) => p },
  ],
}
const manifest: NarrationManifest = {
  v: 1,
  episode: 'toy',
  lang: 'en',
  audio: 'en.abc.m4a',
  scenes: [
    { id: 'a', seg: [1000, 3000], lines: ['one', 'two'] },
    { id: 'b', seg: [5000], lines: ['three'] },
  ],
}

describe('narration timing', () => {
  it('piecewise maps through its knots and clamps', () => {
    expect(piecewise([0, 0.5, 1], [0, 0.2, 1], 0.25)).toBeCloseTo(0.1)
    expect(piecewise([0, 0.5, 1], [0, 0.2, 1], 0.75)).toBeCloseTo(0.6)
    expect(piecewise([0, 1], [0, 1], -1)).toBe(0)
    expect(piecewise([0, 1], [0, 1], 2)).toBe(1)
  })

  it('stretches each caption to its spoken length', () => {
    const r = narratedTiming([0, 0.5], [1000, 3000])
    expect(r.dur).toBe(4000)
    expect(r.cues).toEqual([0, 0.25])
    expect(r.warp).toEqual({ q: [0, 0.25, 1], p: [0, 0.5, 1] })
  })

  it('narrates an episode: new lengths, spoken captions, warped render, the track URL', () => {
    const n = narrate(toy, manifest)!
    expect(n.audio).toBe('/audio/toy/en.abc.m4a')
    expect(timeline(n.scenes).total).toBe(9000)
    expect(n.scenes[0].lines).toEqual(['one', 'two'])
    expect(n.scenes[1].ponder).toBe(true)
    // a quarter of the way in, line two starts: the animation is at the original second cue
    expect(n.scenes[0].render(0.25)).toBeCloseTo(0.5)
    expect(n.scenes[0].render(0.125)).toBeCloseTo(0.25)
  })

  it('ignores a manifest that does not fit the episode', () => {
    expect(narrate(toy, { ...manifest, scenes: manifest.scenes.slice(0, 1) })).toBeNull()
    expect(narrate(toy, { ...manifest, scenes: [{ ...manifest.scenes[0], seg: [1000] }, manifest.scenes[1]] })).toBeNull()
    expect(narrate(toy, { ...manifest, scenes: [{ ...manifest.scenes[0], id: 'x' }, manifest.scenes[1]] })).toBeNull()
    expect(narrate(toy, { ...manifest, episode: 'other' })).toBeNull()
  })

  it('keeps the same moment when the voice is switched on or off', () => {
    const n = narrate(toy, manifest)!
    // original: scene a at p = 0.5 (2 s) → narrated: where line two starts (1 s)
    expect(remapTime(toy.scenes, n.scenes, 2000)).toBeCloseTo(1000)
    expect(remapTime(n.scenes, toy.scenes, 1000)).toBeCloseTo(2000)
    // scene b halfway
    expect(remapTime(toy.scenes, n.scenes, 5000)).toBeCloseTo(6500)
    for (const t of [0, 700, 2600, 3999, 4500, 6000]) expect(remapTime(n.scenes, toy.scenes, remapTime(toy.scenes, n.scenes, t))).toBeCloseTo(t)
    expect(locate(timeline(n.scenes), n.scenes, remapTime(toy.scenes, n.scenes, 4500)).i).toBe(1)
  })

  it('narrates Hong Kong pages in Cantonese', () => {
    expect(narrationLang('zh-HK')).toBe('yue')
    expect(narrationLang('zh')).toBe('zh')
    expect(narrationLang('en')).toBe('en')
  })
})

describe('narration scripts', () => {
  it('has at least one script', () => {
    expect(Object.keys(scripts).length).toBeGreaterThan(0)
  })
  for (const [path, script] of Object.entries(scripts)) {
    const id = path.replace(/^.*\//, '').replace('.json', '')
    it(`${id}: one line per caption in every language`, () => {
      const ep = EPISODE_CODE[id]
      expect(ep, `${id} is an episode`).toBeTruthy()
      for (const [lang, scenes] of Object.entries(script)) {
        expect(['en', 'zh', 'yue']).toContain(lang)
        expect(Object.keys(scenes), `${id}/${lang} scenes`).toEqual(ep.scenes.map((s) => s.id))
        for (const s of ep.scenes) {
          const lines = scenes[s.id]
          expect(lines.length, `${id}/${lang}/${s.id}`).toBe(s.cues.length)
          for (const [k, l] of lines.entries()) {
            const text = typeof l === 'string' ? l : l.text
            expect(text.trim().length, `${id}/${lang}/${s.id}`).toBeGreaterThan(0)
            // English and Mandarin show the site's own captions (only what is read may differ); Cantonese shows the spoken script
            const site = { en, zh }[lang as 'en' | 'zh']
            if (site) expect(text, `${id}/${lang}/${s.id}.c${k + 1} must equal the caption`).toBe(get(site, `watch.${id}.scenes.${s.id}.c${k + 1}`))
            // symbols a voice reads badly belong in "say", not in the read text
            const say = typeof l === 'string' ? l : (l.say ?? l.text)
            expect(say, `${id}/${lang}/${s.id}: ${say}`).not.toMatch(/[=<>←→≈≤≥∑√σθηδ∂∇×·²³⁻ⁱ₀-₉−°‖[\]]|\$/)
          }
        }
      }
    })
  }
})
