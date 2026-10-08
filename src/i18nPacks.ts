import type { i18n as I18n } from 'i18next'
import PACKS from 'virtual:locale-packs'
import { normalizeLang } from './lib/lang'

/**
 * Strings that only one page needs (an explainer's captions, a lab page, the NumPy module, the drill, the formula
 * sheet) live in packs outside the core bundle — see localeSplit in scripts/vite-plugins.mjs.
 */
export const usedPacks = new Set<string>()
const loaded = new Set<string>()

export const hasPack = (name: string, lang: string) => loaded.has(`${name}:${lang}`)

export async function fetchPack(name: string, lang: string): Promise<Record<string, unknown> | null> {
  const load = PACKS[name]?.[lang]
  if (!load) return null
  const m = await load()
  loaded.add(`${name}:${lang}`)
  return m.default
}

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v)

/** Deep merge of packs into a language's resources. */
export function mergePacks(res: Record<string, unknown>, extra: (Record<string, unknown> | null)[]): Record<string, unknown> {
  const merge = (a: Record<string, unknown>, b: Record<string, unknown>): Record<string, unknown> => {
    const out = { ...a }
    for (const [k, v] of Object.entries(b)) out[k] = isObj(v) && isObj(out[k]) ? merge(out[k] as Record<string, unknown>, v) : v
    return out
  }
  return extra.reduce<Record<string, unknown>>((acc, e) => (e ? merge(acc, e) : acc), res)
}

/** Load packs for the current language into i18n (and remember them for language switches). */
export async function ensurePacks(i18n: I18n, names: string[]): Promise<void> {
  const lang = normalizeLang(i18n.resolvedLanguage ?? i18n.language)
  await Promise.all(
    names.map(async (n) => {
      usedPacks.add(n)
      if (hasPack(n, lang)) return
      const p = await fetchPack(n, lang)
      if (p) i18n.addResourceBundle(lang, 'translation', p, true, true)
    }),
  )
}

/** A lazy page or component whose strings arrive in parallel with its code. */
export const withPacks = <T,>(i18n: I18n, names: string[], load: () => Promise<T>): (() => Promise<T>) => () =>
  Promise.all([load(), ensurePacks(i18n, names)]).then(([m]) => m)
