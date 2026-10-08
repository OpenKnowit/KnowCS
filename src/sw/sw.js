/* KnowCS service worker.
 * /assets/* file names are content-hashed, so they never change: serve them cache-first (instant repeat visits even
 * though the server sends no-cache). Pages are network-first so a new deploy is seen at once, with the last copy
 * as an offline fallback. "Save for offline" (a message from the page) copies every page and asset listed in
 * /offline.json into a versioned cache that is never trimmed. Copied to the site root by vite.config.js;
 * registered in src/site/main.tsx. */
const ASSETS = 'knowcs-assets-v1'
const PAGES = 'knowcs-pages-v1'
const OFFLINE = 'knowcs-offline-'
const MAX_ASSETS = 200

self.addEventListener('install', () => self.skipWaiting())
self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('knowcs-') && k !== ASSETS && k !== PAGES && !k.startsWith(OFFLINE)).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  )
})

async function assetFirst(request) {
  // any cache will do: the runtime one or a saved offline copy
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

/** Copy every page and asset of the current build into knowcs-offline-<version>, reporting progress. */
async function saveOffline(client) {
  const post = (msg) => client && client.postMessage(msg)
  try {
    const manifest = await (await fetch('/offline.json', { cache: 'no-store' })).json()
    const name = OFFLINE + manifest.version
    const cache = await caches.open(name)
    const queue = [...manifest.pages, ...manifest.assets]
    const total = queue.length
    let done = 0
    let failed = 0
    const worker = async () => {
      while (queue.length) {
        const url = queue.shift()
        try {
          if (!(await cache.match(url))) {
            const res = await fetch(url, { cache: 'no-cache' })
            if (res.ok) await cache.put(url, res)
            else failed++
          }
        } catch {
          failed++
        }
        done++
        post({ type: 'offline-progress', done, total })
      }
    }
    await Promise.all([worker(), worker(), worker(), worker()])
    for (const k of await caches.keys()) if (k.startsWith(OFFLINE) && k !== name) await caches.delete(k)
    post({ type: 'offline-done', version: manifest.version, failed })
  } catch {
    post({ type: 'offline-done', version: null, failed: 1 })
  }
}

self.addEventListener('message', (e) => {
  if (e.data && e.data.type === 'save-offline') e.waitUntil(saveOffline(e.source))
})
