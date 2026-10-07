// Timeline math for the in-browser explainer videos (/watch/…). Everything is a pure function of time,
// so any frame can be drawn directly: scrubbing, seeking and thumbnails all reuse the same render.

export const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v)

/** Progress of p through the window [a, b], clamped to 0…1. */
export const seg = (p: number, a: number, b: number): number => (b <= a ? (p >= b ? 1 : 0) : clamp01((p - a) / (b - a)))

/** Smooth start and stop (cubic), the default motion curve. */
export const ease = (t: number): number => {
  const x = clamp01(t)
  return x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2
}

/** Eased progress through [a, b]. */
export const eseg = (p: number, a: number, b: number): number => ease(seg(p, a, b))

export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t

export const lerp2 = (a: readonly [number, number], b: readonly [number, number], t: number): [number, number] => [lerp(a[0], b[0], t), lerp(a[1], b[1], t)]

/** Rises over [a, a + d] and falls over [b − d, b]: an element that appears, stays, then leaves. */
export const window01 = (p: number, a: number, b: number, d = 0.06): number => Math.min(eseg(p, a, a + d), 1 - eseg(p, b - d, b))

const hex = (c: string): [number, number, number] => {
  const h = c.replace('#', '')
  const full = h.length === 3 ? h.split('').map((x) => x + x).join('') : h
  return [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16)) as [number, number, number]
}

/** Blend two #rrggbb colours. */
export const lerpColor = (a: string, b: string, t: number): string => {
  const [x, y] = [hex(a), hex(b)]
  return `#${x.map((v, i) => Math.round(lerp(v, y[i], clamp01(t))).toString(16).padStart(2, '0')).join('')}`
}

export interface TimedScene {
  /** duration in ms at 1× speed */
  dur: number
  /** caption switch points as fractions of the scene; caption k starts at cues[k] */
  cues: readonly number[]
  /** pause here at the end of the scene and ask the viewer to think first */
  ponder?: boolean
}

export interface Timeline {
  starts: number[]
  total: number
  /** absolute times (ms) where playback stops for a "pause and ponder" */
  stops: number[]
}

export const timeline = (scenes: readonly TimedScene[]): Timeline => {
  const starts: number[] = []
  let t = 0
  for (const s of scenes) {
    starts.push(t)
    t += s.dur
  }
  const stops = scenes.flatMap((s, i) => (s.ponder ? [starts[i] + s.dur] : []))
  return { starts, total: t, stops }
}

/** Which scene contains absolute time t, and the progress p ∈ [0, 1] inside it. The very end maps to the last scene at p = 1. */
export const locate = (tl: Timeline, scenes: readonly TimedScene[], t: number): { i: number; p: number } => {
  const x = Math.max(0, Math.min(tl.total, t))
  let i = 0
  while (i < scenes.length - 1 && x >= tl.starts[i + 1]) i++
  return { i, p: scenes[i].dur ? clamp01((x - tl.starts[i]) / scenes[i].dur) : 1 }
}

/** Index of the caption showing at scene progress p. */
export const cueAt = (cues: readonly number[], p: number): number => {
  let k = 0
  for (let j = 0; j < cues.length; j++) if (cues[j] <= p) k = j
  return k
}

/** The first ponder stop in (from, to]; playback that crosses it should halt there. */
export const stopBetween = (stops: readonly number[], from: number, to: number): number | null => {
  for (const s of stops) if (s > from && s <= to) return s
  return null
}

export const fmtClock = (ms: number): string => {
  const s = Math.floor(Math.max(0, ms) / 1000)
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}
