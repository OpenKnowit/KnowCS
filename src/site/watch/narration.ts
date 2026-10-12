import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { locate, narratedTiming, piecewise, timeline } from '../../lib/explainer'
import { normalizeLang } from '../../lib/lang'
import type { LangCode } from '../../lib/lang'
import type { Episode, Scene } from './Player'

// Spoken narration for the explainers. scripts/tts/narrate.py builds, per episode and language, one audio track and a
// manifest at /audio/<episode>/<lang>.json: for every scene, how long each caption is spoken (seg, ms) and the line
// read (shown as the caption while the voice is on). The audio files are not part of the build; an episode without a
// manifest simply plays silent with its original timing.

export interface NarrationManifest {
  v: 1
  episode: string
  lang: string
  /** file name of the track, next to the manifest */
  audio: string
  scenes: { id: string; seg: number[]; lines: string[] }[]
}

/** Site language → narration language (Hong Kong pages are narrated in Cantonese). */
export const narrationLang = (lang: LangCode): 'en' | 'zh' | 'yue' => (lang === 'zh-HK' ? 'yue' : lang)

const audioBase = (id: string) => `/audio/${id}/`

/**
 * The episode re-timed to its narration: scene lengths follow the spoken lines, captions show them, and render()
 * is warped so each caption's animation plays while its line is read. null when the manifest does not fit the
 * episode's scenes (an old track for changed code is ignored rather than played out of step).
 */
export function narrate(episode: Episode, m: NarrationManifest): Episode | null {
  if (m.v !== 1 || m.episode !== episode.id || m.scenes.length !== episode.scenes.length) return null
  const scenes: Scene[] = []
  for (const [i, s] of episode.scenes.entries()) {
    const n = m.scenes[i]
    if (n.id !== s.id || n.seg.length !== s.cues.length || n.lines.length !== s.cues.length || n.seg.some((x) => !(x > 0))) return null
    const { dur, cues, warp } = narratedTiming(s.cues, n.seg)
    scenes.push({ ...s, dur, cues, warp, lines: n.lines, render: (q) => s.render(piecewise(warp.q, warp.p, q)) })
  }
  return { ...episode, scenes, audio: audioBase(episode.id) + m.audio }
}

/** The same moment (scene and point in its animation) on another timing of the episode: toggling the voice keeps your place. */
export function remapTime(from: readonly Scene[], to: readonly Scene[], t: number): number {
  const a = locate(timeline(from), from, t)
  const s = from[a.i]
  const p = s.warp ? piecewise(s.warp.q, s.warp.p, a.p) : a.p
  const d = to[a.i]
  if (!d) return 0
  const q = d.warp ? piecewise(d.warp.p, d.warp.q, p) : p
  return timeline(to).starts[a.i] + q * d.dur
}

const VOICE_KEY = 'knowcs-voice'

const readVoice = (): boolean => {
  try {
    return localStorage.getItem(VOICE_KEY) !== 'off'
  } catch {
    return true
  }
}

const manifests = new Map<string, Promise<NarrationManifest | null>>()

const loadManifest = (url: string): Promise<NarrationManifest | null> => {
  let p = manifests.get(url)
  if (!p) {
    p = fetch(url, { cache: 'no-cache' })
      .then((r) => (r.ok && (r.headers.get('content-type') ?? '').includes('json') ? (r.json() as Promise<NarrationManifest>) : null))
      .catch(() => null)
    manifests.set(url, p)
  }
  return p
}

/**
 * The episode as it should play: narrated when a track exists for the page language and the voice is on (the default,
 * remembered per browser), otherwise unchanged. `voice` is null when there is no narration to switch.
 */
export function useNarration(episode: Episode | null): { episode: Episode | null; voice: { on: boolean; toggle: () => void } | null } {
  const { i18n } = useTranslation()
  const lang = narrationLang(normalizeLang(i18n.resolvedLanguage ?? i18n.language))
  const url = episode ? `${audioBase(episode.id)}${lang}.json` : null
  const [loaded, setLoaded] = useState<{ url: string; m: NarrationManifest | null } | null>(null)
  const [on, setOn] = useState(readVoice)
  useEffect(() => {
    if (!url) return
    let live = true
    void loadManifest(url).then((m) => live && setLoaded({ url, m }))
    return () => {
      live = false
    }
  }, [url])
  const narrated = useMemo(() => (episode && loaded?.m && loaded.url === url ? narrate(episode, loaded.m) : null), [episode, loaded, url])
  const toggle = () => {
    setOn(!on)
    try {
      localStorage.setItem(VOICE_KEY, on ? 'off' : 'on')
    } catch {
      /* private mode: the choice lasts for this page only */
    }
  }
  return { episode: narrated && on ? narrated : episode, voice: narrated ? { on, toggle } : null }
}
