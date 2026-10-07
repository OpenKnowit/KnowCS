import { LabError } from '../lib/labError'

/** Display helpers for exam-style numbers (unicode minus, trailing zeros dropped). */

export const fmt = (x: number, d = 2): string => {
  if (x === Infinity) return '∞'
  if (x === -Infinity) return '−∞'
  if (!Number.isFinite(x)) return '—'
  const r = Number(x.toFixed(d))
  return (Object.is(r, -0) ? 0 : r).toString().replace('-', '−')
}

export const fixed = (x: number, d = 2): string => (Number.isFinite(x) ? x.toFixed(d).replace('-', '−') : '—')

export const pct = (x: number, d = 1): string => `${fmt(x * 100, d)}%`

export const int = (x: number): string => x.toLocaleString('en-US')

export const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x))

/** Class / cluster colours shared by every lab page. */
export const PALETTE = ['#2563eb', '#e11d48', '#059669', '#d97706', '#7c3aed', '#0891b2']

/** Parse a typed answer: "5/68", "0.25", "−3", "c1". */
export function parseAnswer(s: string): number | string {
  const t = s.trim().replace(/−/g, '-')
  const frac = t.match(/^(-?\d+(?:\.\d+)?)\s*\/\s*(\d+(?:\.\d+)?)$/)
  if (frac) return Number(frac[1]) / Number(frac[2])
  if (t !== '' && !Number.isNaN(Number(t))) return Number(t)
  return t.toLowerCase()
}

export function answerMatches(given: string, expected: string): boolean {
  const g = parseAnswer(given)
  const e = parseAnswer(expected)
  if (typeof g === 'number' && typeof e === 'number') {
    const a = Math.abs(e)
    // small values (densities, products) are graded relatively; others allow 2-decimal rounding
    return a < 0.05 ? Math.abs(g - e) <= a * 0.05 + 1e-12 : Math.abs(g - e) <= Math.max(0.011, a * 0.006)
  }
  return g === e
}

/** Translate an input error from a parser (LabError) or fall back to its message. */
export const errorText = (e: unknown, t: (key: string, params?: Record<string, string | number>) => string): string =>
  e instanceof LabError ? t(`lab.errors.${e.code}`, e.params) : e instanceof Error ? e.message : String(e)
