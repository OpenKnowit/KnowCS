// One count per page opened, sent to the site's own server (server/admin.mjs, "Visits"): the path, the language and
// the site that linked here. No cookie, nothing kept in the browser, and nothing sent where the browser asks not to be tracked.

/** pages that are not counted: the admin page and the 404 page */
const UNCOUNTED = new Set(['admin', 'notfound'])

export function countVisit(page: string, lang: string) {
  if (!import.meta.env.PROD || UNCOUNTED.has(page)) return
  const nav = navigator as Navigator & { globalPrivacyControl?: boolean }
  if (nav.doNotTrack === '1' || nav.globalPrivacyControl === true) return
  const send = () => {
    let from = ''
    try {
      const host = document.referrer ? new URL(document.referrer).host : ''
      if (host !== window.location.host) from = host
    } catch {
      /* a referrer that is not a URL: leave it out */
    }
    try {
      navigator.sendBeacon('/api/hit', new Blob([JSON.stringify({ p: window.location.pathname, r: from, l: lang })], { type: 'application/json' }))
    } catch {
      /* counting must never get in the page's way */
    }
  }
  // a page prerendered on hover has not been opened yet: count it when it is shown
  if ((document as Document & { prerendering?: boolean }).prerendering) document.addEventListener('prerenderingchange', send, { once: true })
  else send()
}
