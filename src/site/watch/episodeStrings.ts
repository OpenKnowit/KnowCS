import EP_STRINGS from 'virtual:episode-strings'

type Strings = { watch: Record<string, Record<string, unknown>> }

/** Episodes whose captions this page has asked for; a language switch reloads their strings too (see i18n.ts). */
export const usedEpisodes = new Set<string>()
const loaded = new Set<string>()

export const hasEpisodeStrings = (id: string, lang: string) => loaded.has(`${id}:${lang}`)

/** Captions, labels, recap and exam corner of one episode in one language — one small chunk each. */
export async function fetchEpisodeStrings(id: string, lang: string): Promise<Strings | null> {
  const load = EP_STRINGS[id]?.[lang]
  if (!load) return null
  const m = await load()
  loaded.add(`${id}:${lang}`)
  return m.default as Strings
}

/** Deep-merges episode strings into a language's resources (only the watch.<episode> level differs). */
export function mergeEpisodeStrings(res: Record<string, unknown>, extra: (Strings | null)[]): Record<string, unknown> {
  const watch = { ...(res.watch as Record<string, Record<string, unknown>>) }
  for (const e of extra) if (e) for (const [k, v] of Object.entries(e.watch)) watch[k] = { ...watch[k], ...v }
  return { ...res, watch }
}
