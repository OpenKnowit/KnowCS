// When a page cannot be shown: was it a download that did not arrive (worth one automatic reload: after a new release
// the page a browser kept may ask for files that are gone), or the page's own code going wrong?

/**
 * True for the errors browsers raise when a script, stylesheet or data file could not be fetched:
 * Chrome "Failed to fetch dynamically imported module", Firefox "error loading dynamically imported module",
 * Safari "Importing a module script failed", Vite "Unable to preload CSS", and fetch's own
 * "Failed to fetch" / "Load failed" / "NetworkError when attempting to fetch resource".
 */
export function isLoadFailure(error: unknown): boolean {
  const e = error as { name?: unknown; message?: unknown } | null
  const text = `${typeof e?.name === 'string' ? e.name : ''} ${typeof e?.message === 'string' ? e.message : typeof error === 'string' ? error : ''}`
  return /dynamically imported module|importing a module script failed|unable to preload css|failed to fetch|load failed|networkerror|chunkloaderror|loading (css )?chunk/i.test(text)
}

/** how long after an automatic reload another one is not tried: if the files are still missing, the visitor is told instead */
export const RELOAD_GAP_MS = 120_000

/** whether to reload by itself now, given when it last did (a timestamp as text, or nothing) and whether the browser is online */
export function mayReload(last: string | null, now: number, online: boolean): boolean {
  if (!online) return false
  const at = Number(last)
  return !last || !Number.isFinite(at) || now - at > RELOAD_GAP_MS || at > now
}
