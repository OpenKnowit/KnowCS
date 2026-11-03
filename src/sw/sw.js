/* KnowCS service worker.
 * /assets/* file names are content-hashed, so they never change: serve them cache-first (instant repeat visits even
 * though the server sends no-cache). Pages are stale-while-revalidate: a page seen before opens at once from the
 * cache while a fresh copy is fetched for next time (the server is far away; a round trip costs ~0.5 s), so a new
 * deploy shows from the second visit of a page on. That needs the old pages' chunks to stay on the server: each
 * release keeps the previous release's recent /assets (deploy/README.md). Pages never seen fall back to the network.
 * Copied to the site root by vite.config.js; registered in src/site/main.tsx. */
const ASSETS = 'knowcs-assets-v1'
const PAGES = 'knowcs-pages-v2'
const MAX_ASSETS = 200

self.addEventListener('install', () => self.skipWaiting())
self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches
      .keys()
      // knowcs-pyodide:* belongs to the real-Python worker (src/lib/pyodide/worker.ts), which manages its own versions
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('knowcs-') && !k.startsWith('knowcs-pyodide:') && k !== ASSETS && k !== PAGES).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  )
})

async function assetFirst(request) {
  const hit = await caches.match(request)
  if (hit) return hit
  const cache = await caches.open(ASSETS)
  const res = await fetch(request)
  if (res.ok) {
    await cache.put(request, res.clone())
    // keep the cache bounded: old deploys' chunks are dropped first (keys come back in insertion order)
    const keys = await cache.keys()
    for (const k of keys.slice(0, Math.max(0, keys.length - MAX_ASSETS))) await cache.delete(k)
  }
  return res
}

async function pageStaleWhileRevalidate(event) {
  const request = event.request
  // one copy per page: ?lang=…, ?code=… and #hashes all get the same static HTML
  const url = new URL(request.url)
  const key = url.origin + url.pathname
  const cache = await caches.open(PAGES)
  const hit = await cache.match(key)
  const fresh = fetch(request).then(async (res) => {
    if (res.ok && !res.redirected) await cache.put(key, res.clone())
    return res
  })
  if (hit) {
    event.waitUntil(fresh.catch(() => {}))
    return hit
  }
  try {
    return await fresh
  } catch (err) {
    return (await caches.match(url.origin + '/')) || Promise.reject(err)
  }
}

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url)
  if (e.request.method !== 'GET' || url.origin !== self.location.origin) return
  if (url.pathname.startsWith('/assets/')) e.respondWith(assetFirst(e.request))
  else if (e.request.mode === 'navigate') e.respondWith(pageStaleWhileRevalidate(e))
})

