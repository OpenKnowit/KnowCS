import { useCallback, useState } from 'react'
import { isLoadFailure, mayReload } from '../lib/failure'

// Helpers for the page boundary (PageState.tsx): bringing a failure that happened outside rendering to it,
// and the one automatic reload after a download that did not arrive.

const RELOAD_KEY = 'knowcs-reloaded-at'

/**
 * A download failed: is one automatic reload worth it? Yes if the browser is online and this tab has not just tried
 * (a reload fetches the current release's page). Only reads, so it is safe to ask while rendering; whoever then
 * reloads calls markReloaded() first, and if the files are still missing the visitor is told instead.
 */
export function wouldReloadFor(error: unknown): boolean {
  if (!isLoadFailure(error)) return false
  try {
    return mayReload(sessionStorage.getItem(RELOAD_KEY), Date.now(), navigator.onLine !== false)
  } catch {
    return false // no session storage: without a memory of having tried, do not risk reloading for ever
  }
}
export function markReloaded(): void {
  try {
    sessionStorage.setItem(RELOAD_KEY, String(Date.now()))
  } catch {
    /* wouldReloadFor said no in this case */
  }
}

/**
 * For work a page starts itself (a download in an effect or a click): pass a failure to the returned function and
 * the page boundary shows it, as it would an error thrown while rendering.
 */
export function useRaise(): (error: unknown) => void {
  const [failure, setFailure] = useState<{ error: unknown } | null>(null)
  if (failure) throw failure.error instanceof Error ? failure.error : new Error(String(failure.error))
  return useCallback((error: unknown) => setFailure({ error }), [])
}
