/* KnowCS service worker.
 * /assets/* file names are content-hashed, so they never change: serve them cache-first (instant repeat visits even
 * though the server sends no-cache). Pages are network-first so a new deploy is seen at once, with the last copy
 * as an offline fallback. Copied to the site root by vite.config.js; registered in src/site/main.tsx. */
const ASSETS = 'knowcs-assets-v1'
const PAGES = 'knowcs-pages-v1'
const MAX_ASSETS = 200

self.addEventListener('install', () => self.skipWaiting())
self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('knowcs-') && k !== ASSETS && k !== PAGES).map((k) => caches.delete(k))))
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

async function pageNetworkFirst(request) {
  const cache = await caches.open(PAGES)
  try {
    const res = await fetch(request)
    if (res.ok) await cache.put(request, res.clone())
    return res
  } catch (err) {
    return (await caches.match(request)) || (await caches.match('/')) || Promise.reject(err)
  }
}

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url)
  if (e.request.method !== 'GET' || url.origin !== self.location.origin) return
  if (url.pathname.startsWith('/assets/')) e.respondWith(assetFirst(e.request))
  else if (e.request.mode === 'navigate') e.respondWith(pageNetworkFirst(e.request))
})

