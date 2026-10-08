import { describe, expect, it } from 'vitest'
import { MATPLOTLIB_PLAYGROUND } from './matplotlib'
import { runIn } from './run'
import type { PlayConfig } from './types'

const CONFIGS: PlayConfig[] = [MATPLOTLIB_PLAYGROUND]

describe.each(CONFIGS.map((c) => [c.id, c] as const))('%s playground', (_id, config) => {
  it('every category has an example and every example has a known category', () => {
    for (const c of config.cats) expect(config.entries.some((e) => e.cat === c)).toBe(true)
    for (const e of config.entries) expect(config.cats).toContain(e.cat)
    expect(new Set(config.entries.map((e) => e.id)).size).toBe(config.entries.length)
  })

  it.each(config.entries.map((e) => [e.id, e] as const))('%s runs without errors', (_e, entry) => {
    const r = runIn(config, entry.code, 10_000)
    expect(r.error).toBeNull()
    expect(r.calls.length + r.events.length).toBeGreaterThan(0)
  })
})
