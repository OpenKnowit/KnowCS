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
//
// Env: ADMIN_PASSPHRASE (required), DATA_DIR, PORT (3021), HOST (127.0.0.1), COOKIE_SECURE=true,
// TRUST_PROXY=true (behind a proxy that sets X-Real-IP and X-Forwarded-Host).
import http from 'node:http'
import { promises as fs, createReadStream } from 'node:fs'
import path from 'node:path'
import { createHmac, randomBytes, randomUUID, timingSafeEqual } from 'node:crypto'
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
const readJSON = async (req, limit = 32 * 1024) => {
  try {
    return JSON.parse((await readBody(req, limit)).toString() || '{}')
  } catch (e) {
    if (e.status) throw e
    throw fail(400, 'bad_json')
  }
}

const attempts = new Map()
function limitSignIn(req, res) {
  const now = Date.now()
  for (const [k, v] of attempts) if (v.until <= now) attempts.delete(k)
  const fwd = req.headers['x-real-ip']
  const ip = process.env.TRUST_PROXY === 'true' && typeof fwd === 'string' && isIP(fwd) ? fwd : req.socket.remoteAddress
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
  const url = text(input.url, 2048)
  if (!/^https?:\/\/[^\s]+$/i.test(url)) throw fail(400, 'bad_url')
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
const removeMedia = async (preview) => {
  const m = /^\/api\/media\/([a-f0-9-]+\.(png|jpg|webp))$/.exec(preview || '')
  if (m) await fs.rm(path.join(mediaDir, m[1]), { force: true })
}

const server = http.createServer(async (req, res) => {
  res.setHeader('X-Content-Type-Options', 'nosniff')
  res.setHeader('Referrer-Policy', 'same-origin')
  try {
    const { pathname } = new URL(req.url, 'http://localhost')
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

    // everything below changes data: only from this site's own pages
    const origin = req.headers.origin
    const fwdHost = req.headers['x-forwarded-host']
    const host = process.env.TRUST_PROXY === 'true' && typeof fwdHost === 'string' ? fwdHost.split(',')[0].trim() : req.headers.host
    if (!origin || new URL(origin).host !== host || req.headers['sec-fetch-site'] === 'cross-site') throw fail(403, 'wrong_origin')
    if (pathname === '/api/session' && req.method === 'POST') {
      limitSignIn(req, res)
      const { passphrase } = await readJSON(req, 4096)
      if (!equal(String(passphrase || ''), password)) throw fail(401, 'wrong_passphrase')
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
server.listen(port, process.env.HOST || '127.0.0.1', () => console.log(`KnowCS admin API on ${port}`))
