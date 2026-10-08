import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Full-screen mode for a panel: a fixed overlay (render it in a portal) plus the browser's full-screen mode where
 * it exists. Esc, the browser's own exit, or leave() closes it; #playground in the URL opens it on load.
 */
export function useFullScreen() {
  const [full, setFull] = useState(() => typeof window !== 'undefined' && window.location.hash === '#playground')
  const ref = useRef<HTMLDivElement>(null)

  const leave = useCallback(() => {
    setFull(false)
    if (document.fullscreenElement) void document.exitFullscreen().catch(() => undefined)
  }, [])

  const enter = useCallback(() => {
    setFull(true)
    // the overlay alone is enough where the API is missing (iPhone Safari)
    requestAnimationFrame(() => void ref.current?.requestFullscreen?.().catch(() => undefined))
  }, [])

  useEffect(() => {
    if (!full) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') leave()
    }
    const onFs = () => {
      if (!document.fullscreenElement) setFull(false)
    }
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    window.addEventListener('keydown', onKey)
    document.addEventListener('fullscreenchange', onFs)
    // the page's own hash (the Playground page keeps the library there) comes back on exit
    const before = window.location.hash === '#playground' ? '' : window.location.hash
    history.replaceState(null, '', '#playground')
    return () => {
      document.body.style.overflow = prev
      window.removeEventListener('keydown', onKey)
      document.removeEventListener('fullscreenchange', onFs)
      history.replaceState(null, '', window.location.pathname + window.location.search + before)
    }
  }, [full, leave])

  return { full, enter, leave, ref }
}
