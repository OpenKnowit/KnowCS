// KnowCS admin API: the Resources list, editable behind a passphrase. One small Node process (no framework),
// modelled on the pmpweb community server. Data is a JSON file outside the web root; uploaded previews sit next to it.
//
//   GET    /api/resources                 public list (the Resources page reads it; falls back to its built-in copy)
//   GET    /api/media/<file>              uploaded preview images
//   GET    /api/session                   { authenticated }
//   POST   /api/session  {passphrase}     sign in (5 tries / minute / IP) → HttpOnly signed cookie, 12 h
//   DELETE /api/session                   sign out
//   POST   /api/resources                 add      ┐
//   PATCH  /api/resources/<id>            edit     │ signed in, and only from this site's own pages
//   DELETE /api/resources/<id>            remove   │
//   PATCH  /api/resources/order {ids, baseIds}     │ reorder
//   PUT    /api/resources/<id>/preview    image body (PNG / JPEG / WebP ≤ 3 MB); DELETE removes it
//   POST   /api/hit  {p, r, l}            one page view: path, the site that linked here, language (see "Visits")
//   GET    /api/stats?days=30             signed in: the counts per day for the site asked from
//
// Env: ADMIN_PASSPHRASE (required), DATA_DIR, PORT (3021), HOST (127.0.0.1), COOKIE_SECURE=true,
// TRUST_PROXY=true (behind a proxy that sets X-Real-IP and X-Forwarded-Host),
// STATS_HOSTS (the sites whose visits are counted; knowcs.online,dev.knowcs.online).
import http from 'node:http'
import { promises as fs, createReadStream } from 'node:fs'
import path from 'node:path'
import { createHash, createHmac, randomBytes, randomUUID, timingSafeEqual } from 'node:crypto'
import { isIP } from 'node:net'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const password = process.env.ADMIN_PASSPHRASE
if (!password) throw new Error('Set ADMIN_PASSPHRASE before starting the admin server.')
const dataDir = path.resolve(process.env.DATA_DIR || path.join(here, 'data'))
const mediaDir = path.join(dataDir, 'media')
const dbPath = path.join(dataDir, 'resources.json')
const secure = process.env.COOKIE_SECURE === 'true'
await fs.mkdir(mediaDir, { recursive: true })
try {
  await fs.access(dbPath)
} catch {
  await fs.copyFile(path.join(here, 'resources-seed.json'), dbPath)
}
// the signing key survives restarts, so a restart does not sign everybody out
const secretPath = path.join(dataDir, 'session.key')
let secret
try {
  secret = await fs.readFile(secretPath)
} catch {
  secret = randomBytes(32)
  await fs.writeFile(secretPath, secret, { mode: 0o600 })
}

// Traditional Chinese (Hong Kong) is derived from the Simplified text, as the site's own strings are
let toHK = (s) => s
try {
  const { Converter } = await import('opencc-js')
  const convert = Converter({ from: 'cn', to: 'hk' })
  toHK = (s) => convert(s)
} catch {
  console.warn('opencc-js not installed: zh-HK falls back to the Simplified text')
}

let content = JSON.parse(await fs.readFile(dbPath, 'utf8'))
let writing = Promise.resolve()

const CATS = ['course', 'visualize', 'code', 'study', 'notes', 'archive']
const KINDS = ['web', 'login', 'notes', 'archive']
const IMAGE = { png: [0x89, 0x50, 0x4e, 0x47], jpg: [0xff, 0xd8, 0xff], webp: [0x52, 0x49, 0x46, 0x46] }
const TYPES = { png: 'image/png', jpg: 'image/jpeg', webp: 'image/webp' }

const sign = (s) => createHmac('sha256', secret).update(s).digest('hex')
const equal = (a, b) => {
  const x = Buffer.from(a)
  const y = Buffer.from(b)
  return x.length === y.length && timingSafeEqual(x, y)
}
const authenticated = (req) => {
  const token = /(?:^|;\s*)knowcs_admin=([^;]+)/.exec(req.headers.cookie || '')?.[1] || ''
  const [expires, signature = ''] = token.split('.')
  return Number(expires) > Date.now() && equal(sign(expires || ''), signature)
}
const json = (res, status, body) => {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' })
  res.end(JSON.stringify(body))
}
const fail = (status, message) => Object.assign(new Error(message), { status })

async function readBody(req, limit) {
  const parts = []
  let length = 0
  for await (const chunk of req) {
    length += chunk.length
    if (length > limit) throw fail(413, 'too_large')
    parts.push(chunk)
  }
  return Buffer.concat(parts)
}
/** a JSON object from the body (anything else, null and arrays included, is a bad request) */
const readJSON = async (req, limit = 32 * 1024) => {
  let value
  try {
    value = JSON.parse((await readBody(req, limit)).toString() || '{}')
  } catch (e) {
    if (e.status) throw e
    throw fail(400, 'bad_json')
  }
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw fail(400, 'bad_json')
  return value
}

const ipOf = (req) => {
  const fwd = req.headers['x-real-ip']
  return process.env.TRUST_PROXY === 'true' && typeof fwd === 'string' && isIP(fwd) ? fwd : req.socket.remoteAddress
}
/** the site the request was made to (the proxy passes it on) */
const hostOf = (req) => {
  const fwd = req.headers['x-forwarded-host']
  return process.env.TRUST_PROXY === 'true' && typeof fwd === 'string' ? fwd.split(',')[0].trim() : req.headers.host
}
const hostOfUrl = (url) => {
  try {
    return new URL(url).host
  } catch {
    return null
  }
}

const attempts = new Map()
function limitSignIn(req, res) {
  const now = Date.now()
  for (const [k, v] of attempts) if (v.until <= now) attempts.delete(k)
  const ip = ipOf(req)
  const entry = attempts.get(ip) || { count: 0, until: now + 60_000 }
  if (entry.count >= 5) {
    res.setHeader('Retry-After', String(Math.max(1, Math.ceil((entry.until - now) / 1000))))
    throw fail(429, 'too_many_tries')
  }
  entry.count++
  attempts.set(ip, entry)
}

/** a resource from the form: validated, trimmed, zh-HK filled in from zh */
function fields(input, old = {}) {
  const text = (v, max) => String(v ?? '').trim().slice(0, max)
  const lang = (o, max) => {
    const en = text(o?.en, max)
    const zh = text(o?.zh, max)
    return { en, zh, 'zh-HK': zh ? toHK(zh) : '' }
  }
  // a link too long to keep is refused, not cut short into one that leads nowhere
  const url = String(input.url ?? '').trim()
  if (url.length > 2048 || !/^https?:\/\/[^\s]+$/i.test(url)) throw fail(400, 'bad_url')
  const cat = CATS.includes(input.cat) ? input.cat : null
  const kind = KINDS.includes(input.kind) ? input.kind : null
  if (!cat || !kind) throw fail(400, 'bad_category')
  const title = lang(input.title, 120)
  if (!title.en && !title.zh) throw fail(400, 'no_title')
  return { url, cat, kind, title, desc: lang(input.desc, 600), preview: old.preview ?? null }
}

async function save(change) {
  const op = writing.then(async () => {
    const draft = structuredClone(content)
    const result = change(draft)
    draft.updatedAt = new Date().toISOString()
    await fs.copyFile(dbPath, `${dbPath}.backup`)
    await fs.writeFile(`${dbPath}.tmp`, JSON.stringify(draft, null, 2))
    await fs.rename(`${dbPath}.tmp`, dbPath)
    content = draft
    return result
  })
  writing = op.catch(() => {})
  return op
}
// --- Visits: how many pages were opened, per day and site. Counts only: no cookie, and no address or browser is kept.
// A visitor is told apart for one day by a hash of address + browser under a key that lives in memory and changes
// every day; the hashes are never written anywhere, so a restart counts that day's returning visitors again.
const statsPath = path.join(dataDir, 'stats.json')
const STATS_HOSTS = (process.env.STATS_HOSTS || 'knowcs.online,dev.knowcs.online').split(',').map((h) => h.trim()).filter(Boolean)
const STATS_LANGS = ['en', 'zh', 'zh-HK']
const KEEP_DAYS = 400
const MAX_KEYS = 300 // different pages or referring sites per day; the rest are counted together
const OTHER = '(other)'
const BOT = /bot|crawl|spider|slurp|headless|lighthouse|preview|monitor|curl|wget|python|scrapy/i
/** { host: { 'YYYY-MM-DD': { views, visitors, pages: {path: n}, refs: {host: n}, langs: {lang: n} } } } */
let stats = {}
try {
  stats = JSON.parse(await fs.readFile(statsPath, 'utf8'))
} catch {
  /* none yet */
}
/** the date in Hong Kong, where the course is */
const dayOf = (t = Date.now()) => new Date(t + 8 * 3600_000).toISOString().slice(0, 10)
let today = dayOf()
let salt = randomBytes(16)
const seen = new Set()
const hitRate = new Map()
let statsDirty = false

const bump = (counts, key) => {
  const k = Object.hasOwn(counts, key) || Object.keys(counts).length < MAX_KEYS ? key : OTHER
  counts[k] = (counts[k] || 0) + 1
}
function countHit(req, host, body) {
  if (!STATS_HOSTS.includes(host)) return
  const ua = String(req.headers['user-agent'] || '')
  if (!ua || BOT.test(ua)) return
  const page = typeof body.p === 'string' && body.p.length <= 80 && /^\/(?:[a-z0-9-]+\/)*$/.test(body.p) ? body.p : null
  if (!page) return
  const now = Date.now()
  const ip = ipOf(req)
  // at most 60 a minute from one address: a person cannot open pages faster, a loop can
  for (const [k, v] of hitRate) if (v.until <= now) hitRate.delete(k)
  const rate = hitRate.get(ip) || { count: 0, until: now + 60_000 }
  if (rate.count >= 60) return
  rate.count++
  hitRate.set(ip, rate)

  const day = dayOf(now)
  if (day !== today) {
    today = day
    salt = randomBytes(16)
    seen.clear()
    const oldest = dayOf(now - KEEP_DAYS * 86400_000)
    for (const days of Object.values(stats)) for (const d of Object.keys(days)) if (d < oldest) delete days[d]
  }
  const site = (stats[host] ??= {})
  const entry = (site[day] ??= { views: 0, visitors: 0, pages: {}, refs: {}, langs: {} })
  entry.views++
  const who = createHash('sha256').update(salt).update(`${host}\n${ip}\n${ua}`).digest('base64')
  if (!seen.has(who)) {
    seen.add(who)
    entry.visitors++
  }
  bump(entry.pages, page)
  const ref = typeof body.r === 'string' ? body.r.toLowerCase() : ''
  if (ref && ref.length <= 100 && /^[a-z0-9.-]+(?::\d+)?$/.test(ref) && ref !== host) bump(entry.refs, ref)
  if (STATS_LANGS.includes(body.l)) bump(entry.langs, body.l)
  statsDirty = true
}
async function flushStats() {
  if (!statsDirty) return
  statsDirty = false
  try {
    await fs.writeFile(`${statsPath}.tmp`, JSON.stringify(stats))
    await fs.rename(`${statsPath}.tmp`, statsPath)
  } catch (e) {
    statsDirty = true
    console.error(e)
  }
}
setInterval(flushStats, 30_000).unref()
for (const signal of ['SIGTERM', 'SIGINT']) process.once(signal, () => void flushStats().finally(() => process.exit(0)))

const removeMedia = async (preview) => {
  const m = /^\/api\/media\/([a-f0-9-]+\.(png|jpg|webp))$/.exec(preview || '')
  if (m) await fs.rm(path.join(mediaDir, m[1]), { force: true })
}

const server = http.createServer(async (req, res) => {
  res.setHeader('X-Content-Type-Options', 'nosniff')
  res.setHeader('Referrer-Policy', 'same-origin')
  try {
    const { pathname, searchParams } = new URL(req.url, 'http://localhost')
    const host = hostOf(req)
    if (pathname === '/api/resources' && req.method === 'GET') return json(res, 200, content)
    const media = /^\/api\/media\/([a-f0-9-]+)\.(png|jpg|webp)$/.exec(pathname)
    if (media && req.method === 'GET') {
      const file = path.join(mediaDir, `${media[1]}.${media[2]}`)
      const stat = await fs.stat(file).catch(() => null)
      if (!stat) throw fail(404, 'not_found')
      res.writeHead(200, { 'Content-Type': TYPES[media[2]], 'Content-Length': stat.size, 'Cache-Control': 'public, max-age=31536000, immutable' })
      return createReadStream(file).pipe(res)
    }
    if (pathname === '/api/session' && req.method === 'GET') return json(res, 200, { authenticated: authenticated(req) })
    if (pathname === '/api/stats' && req.method === 'GET') {
      if (!authenticated(req)) throw fail(401, 'signed_out')
      const n = Math.min(KEEP_DAYS, Math.max(1, Math.floor(Number(searchParams.get('days')) || 30)))
      const from = dayOf(Date.now() - (n - 1) * 86400_000)
      const days = Object.fromEntries(Object.entries(stats[host] || {}).filter(([d]) => d >= from))
      return json(res, 200, { host, today: dayOf(), counted: STATS_HOSTS.includes(host), days })
    }

    // everything below changes data: only from this site's own pages
    const origin = req.headers.origin
    if (!origin || hostOfUrl(origin) !== host || req.headers['sec-fetch-site'] === 'cross-site') throw fail(403, 'wrong_origin')
    if (pathname === '/api/hit' && req.method === 'POST') {
      countHit(req, host, await readJSON(req, 1024))
      res.writeHead(204, { 'Cache-Control': 'no-store' })
      return res.end()
    }
    if (pathname === '/api/session' && req.method === 'POST') {
      limitSignIn(req, res)
      const { passphrase } = await readJSON(req, 4096)
      if (typeof passphrase !== 'string' || !equal(passphrase, password)) {
        console.warn(`sign-in refused for ${ipOf(req)}`)
        throw fail(401, 'wrong_passphrase')
      }
      console.log(`signed in from ${ipOf(req)}`)
      const expires = String(Date.now() + 12 * 3600_000)
      res.setHeader('Set-Cookie', `knowcs_admin=${expires}.${sign(expires)}; HttpOnly; SameSite=Strict; Path=/api; Max-Age=43200${secure ? '; Secure' : ''}`)
      return json(res, 200, { authenticated: true })
    }
    if (pathname === '/api/session' && req.method === 'DELETE') {
      res.setHeader('Set-Cookie', 'knowcs_admin=; HttpOnly; SameSite=Strict; Path=/api; Max-Age=0')
      return json(res, 200, { ok: true })
    }
    if (!authenticated(req)) throw fail(401, 'signed_out')

    if (pathname === '/api/resources/order' && req.method === 'PATCH') {
      const { ids, baseIds } = await readJSON(req)
      await save((d) => {
        const current = d.resources.map((r) => r.id)
        if (JSON.stringify(baseIds) !== JSON.stringify(current)) throw fail(409, 'changed_elsewhere')
        if (!Array.isArray(ids) || ids.length !== current.length || new Set(ids).size !== ids.length || ids.some((id) => !current.includes(id))) throw fail(400, 'bad_order')
        const byId = new Map(d.resources.map((r) => [r.id, r]))
        d.resources = ids.map((id) => byId.get(id))
      })
      return json(res, 200, content)
    }
    const preview = /^\/api\/resources\/([a-z0-9-]+)\/preview$/.exec(pathname)
    if (preview && (req.method === 'PUT' || req.method === 'DELETE')) {
      const id = preview[1]
      const item = content.resources.find((r) => r.id === id)
      if (!item) throw fail(404, 'not_found')
      let url = null
      if (req.method === 'PUT') {
        const bytes = await readBody(req, 3 * 1024 * 1024)
        const ext = Object.entries(IMAGE).find(([, sig]) => sig.every((b, i) => bytes[i] === b))?.[0]
        if (!ext || (ext === 'webp' && bytes.subarray(8, 12).toString() !== 'WEBP')) throw fail(400, 'bad_image')
        const file = `${randomUUID()}.${ext}`
        await fs.writeFile(path.join(mediaDir, file), bytes)
        url = `/api/media/${file}`
      }
      const old = item.preview
      await save((d) => {
        d.resources.find((r) => r.id === id).preview = url
      })
      await removeMedia(old)
      return json(res, 200, content)
    }
    const one = /^\/api\/resources(?:\/([a-z0-9-]+))?$/.exec(pathname)
    if (!one) throw fail(404, 'not_found')
    const id = one[1]
    if (req.method === 'POST' && !id) {
      const input = await readJSON(req)
      const item = { id: randomUUID(), ...fields(input) }
      await save((d) => {
        d.resources.unshift(item)
      })
      return json(res, 201, content)
    }
    if (req.method === 'PATCH' && id) {
      const input = await readJSON(req)
      await save((d) => {
        const i = d.resources.findIndex((r) => r.id === id)
        if (i < 0) throw fail(404, 'not_found')
        d.resources[i] = { id, ...fields(input, d.resources[i]) }
      })
      return json(res, 200, content)
    }
    if (req.method === 'DELETE' && id) {
      const old = content.resources.find((r) => r.id === id)
      if (!old) throw fail(404, 'not_found')
      await save((d) => {
        d.resources = d.resources.filter((r) => r.id !== id)
      })
      await removeMedia(old.preview)
      return json(res, 200, content)
    }
    throw fail(405, 'not_allowed')
  } catch (error) {
    if (!error.status) console.error(error)
    if (!res.headersSent) json(res, error.status || 500, { error: error.status ? error.message : 'server_error' })
    else res.end()
  }
})
server.requestTimeout = 60_000
const port = Number(process.env.PORT || 3021)
server.listen(port, process.env.HOST || '127.0.0.1', () => console.log(`KnowCS admin API on ${server.address().port}`))
