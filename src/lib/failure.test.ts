import { describe, expect, it } from 'vitest'
import { RELOAD_GAP_MS, isLoadFailure, mayReload } from './failure'

describe('isLoadFailure', () => {
  it('knows the download errors of each browser', () => {
    const messages = [
      'Failed to fetch dynamically imported module: https://knowcs.online/assets/KnnModule-BKh4PHRr.js',
      'error loading dynamically imported module: https://knowcs.online/assets/KnnModule-BKh4PHRr.js',
      'Importing a module script failed.',
      'Unable to preload CSS for /assets/Latex-DC5v-VjN.css',
      'Failed to fetch',
      'Load failed',
      'NetworkError when attempting to fetch resource.',
      'Loading chunk 12 failed.',
    ]
    for (const m of messages) expect([m, isLoadFailure(new TypeError(m))]).toEqual([m, true])
    expect(isLoadFailure(Object.assign(new Error('x'), { name: 'ChunkLoadError' }))).toBe(true)
    expect(isLoadFailure('Failed to fetch')).toBe(true)
  })
  it('does not take a bug in the page for one', () => {
    for (const e of [new TypeError("Cannot read properties of undefined (reading 'map')"), new RangeError('Maximum call stack size exceeded'), new Error('Minified React error #310'), new Error(''), null, undefined, 42, {}]) {
      expect(isLoadFailure(e)).toBe(false)
    }
  })
})

describe('mayReload', () => {
  const now = 1_800_000_000_000
  it('reloads once, then not again for a while', () => {
    expect(mayReload(null, now, true)).toBe(true)
    expect(mayReload(String(now - 1000), now, true)).toBe(false)
    expect(mayReload(String(now - RELOAD_GAP_MS), now, true)).toBe(false)
    expect(mayReload(String(now - RELOAD_GAP_MS - 1), now, true)).toBe(true)
  })
  it('never while offline, where a reload cannot help', () => {
    expect(mayReload(null, now, false)).toBe(false)
    expect(mayReload(String(now - 10 * RELOAD_GAP_MS), now, false)).toBe(false)
  })
  it('is not stopped for good by a value it cannot read, or one from the future', () => {
    expect(mayReload('yes', now, true)).toBe(true)
    expect(mayReload('', now, true)).toBe(true)
    expect(mayReload(String(now + 3_600_000), now, true)).toBe(true)
  })
})
