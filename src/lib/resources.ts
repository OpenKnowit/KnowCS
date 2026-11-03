/**
 * The Resources list as the admin API serves it (server/admin.mjs, GET /api/resources). The page ships a built-in
 * copy and switches to the API's once it answers, so a static build (or an API that is down) still shows the list.
 */
export const RESOURCE_CATS = ['course', 'visualize', 'code', 'study', 'notes', 'archive'] as const
export const RESOURCE_KINDS = ['web', 'login', 'notes', 'archive'] as const
export type ResourceCat = (typeof RESOURCE_CATS)[number]
export type ResourceKind = (typeof RESOURCE_KINDS)[number]
export type Langs = { en: string; zh: string; 'zh-HK': string }

export interface ResourceItem {
  id: string
  url: string
  cat: ResourceCat
  kind: ResourceKind
  title: Langs
  desc: Langs
  /** 'builtin' = the screenshot bundled with the site (src/assets/resources/<id>.webp); a path = an uploaded image */
  preview: string | null
}
export interface ResourceData {
  updatedAt: string
  resources: ResourceItem[]
}

/** the text in the page's language: zh-HK falls back to zh, both fall back to English (and the other way round) */
export const inLang = (o: Langs, lang: string): string => (lang === 'en' ? o.en || o.zh : lang === 'zh-HK' ? o['zh-HK'] || o.zh || o.en : o.zh || o.en)

const isData = (v: unknown): v is ResourceData => !!v && typeof v === 'object' && Array.isArray((v as ResourceData).resources)

/** the live list, or null when there is no API (static hosting, offline) */
export async function fetchResources(signal?: AbortSignal): Promise<ResourceData | null> {
  try {
    const res = await fetch('/api/resources', { cache: 'no-store', signal })
    if (!res.ok) return null
    const data: unknown = await res.json()
    return isData(data) ? data : null
  } catch {
    return null
  }
}
