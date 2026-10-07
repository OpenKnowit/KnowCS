/** Episodes this browser has watched to the end — a per-viewer convenience, safe when storage is blocked. */
const WATCHED_KEY = 'knowcs-watched'
export function readWatched(): Record<string, number> {
  try {
    return JSON.parse(localStorage.getItem(WATCHED_KEY) ?? '{}') as Record<string, number>
  } catch {
    return {}
  }
}
export function markWatched(id: string) {
  try {
    localStorage.setItem(WATCHED_KEY, JSON.stringify({ ...readWatched(), [id]: Date.now() }))
  } catch {
    /* storage blocked: the mark is optional */
  }
}
