import { describe, expect, it } from 'vitest'
import { dayBefore, summarise } from './visits'

const day = (views: number, visitors: number, pages = {}, refs = {}, langs = {}) => ({ views, visitors, pages, refs, langs })

describe('dayBefore', () => {
  it('steps back across months and years', () => {
    expect(dayBefore('2026-11-16', 0)).toBe('2026-11-16')
    expect(dayBefore('2026-11-01', 1)).toBe('2026-10-31')
    expect(dayBefore('2027-01-01', 1)).toBe('2026-12-31')
    expect(dayBefore('2028-03-01', 1)).toBe('2028-02-29')
    expect(dayBefore('2026-11-16', 29)).toBe('2026-10-18')
  })
})

describe('summarise', () => {
  const data = {
    today: '2026-11-16',
    days: {
      '2026-11-16': day(5, 2, { '/': 3, '/knn/': 2 }, { 'google.com': 1 }, { zh: 4, en: 1 }),
      '2026-11-14': day(3, 3, { '/knn/': 3 }, { 'github.com': 2, 'google.com': 1 }, { en: 3 }),
      '2026-11-01': day(100, 50, { '/old/': 100 }, {}, { zh: 100 }),
    },
  }
  it('gives every day of the range, the empty ones too, oldest first', () => {
    const s = summarise(data, 7)
    expect(s.series.map((d) => d.day)).toEqual(['2026-11-10', '2026-11-11', '2026-11-12', '2026-11-13', '2026-11-14', '2026-11-15', '2026-11-16'])
    expect(s.series.map((d) => d.views)).toEqual([0, 0, 0, 0, 3, 0, 5])
    expect(s.today).toEqual({ views: 5, visitors: 2 })
  })
  it('adds up only the days in the range', () => {
    const s = summarise(data, 7)
    expect([s.views, s.visitors]).toEqual([8, 5])
    expect(s.pages).toEqual([['/knn/', 5], ['/', 3]])
    expect(s.refs).toEqual([['github.com', 2], ['google.com', 2]])
    expect(s.langs).toEqual([['zh', 4], ['en', 4]].sort((a, b) => (a[0] < b[0] ? -1 : 1)))
    expect(s.direct).toBe(4)
    expect(summarise(data, 30).views).toBe(108)
  })
  it('is all zeros with nothing counted', () => {
    const s = summarise({ today: '2026-11-16', days: {} }, 3)
    expect(s).toMatchObject({ views: 0, visitors: 0, today: { views: 0, visitors: 0 }, pages: [], refs: [], langs: [], direct: 0 })
    expect(s.series.length).toBe(3)
  })
})
