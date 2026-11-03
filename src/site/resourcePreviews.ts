import type { ResourceItem } from '../lib/resources'

// the screenshots bundled with the site, one per built-in resource (src/assets/resources/<id>.webp)
const PREVIEWS = import.meta.glob<string>('../assets/resources/*.webp', { eager: true, import: 'default' })

/** where a resource's picture comes from: the bundled screenshot, an uploaded image, or none */
export const previewSrc = (r: Pick<ResourceItem, 'id' | 'preview'>): string | undefined =>
  r.preview === 'builtin' ? PREVIEWS[`../assets/resources/${r.id}.webp`] : r.preview || undefined
