// The admin API from the outside: the real server (server/admin.mjs) on a spare port with an empty data folder,
// spoken to over HTTP the way the proxy does (X-Forwarded-Host, X-Real-IP). Run with `npm run test:server`.
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { spawn } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const PASS = 'correct horse battery staple'
const SITE = 'knowcs.online'
const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15'
const PNG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0])

let dataDir, child, base
let nextIp = 1
/** a new address each time, so one test's sign-ins and visits do not use up another's allowance */
const freshIp = () => `10.0.${Math.floor(nextIp / 250)}.${(nextIp++ % 250) + 1}`

function start() {
  return new Promise((resolve, reject) => {
    child = spawn(process.execPath, [path.join(here, 'admin.mjs')], {
      env: { ...process.env, ADMIN_PASSPHRASE: PASS, DATA_DIR: dataDir, PORT: '0', HOST: '127.0.0.1', TRUST_PROXY: 'true', STATS_HOSTS: SITE, COOKIE_SECURE: 'true' },
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let out = ''
    child.stdout.on('data', (d) => {
      out += d
      const m = /admin API on (\d+)/.exec(out)
      if (m) resolve(`http://127.0.0.1:${m[1]}`)
    })
    child.stderr.on('data', (d) => (out += d))
    child.once('exit', (code) => reject(new Error(`server stopped (${code}): ${out}`)))
  })
}
const stop = () =>
  new Promise((resolve) => {
    if (!child || child.exitCode !== null) return resolve()
    child.once('exit', resolve)
    child.kill('SIGTERM')
  })

/** a request as the site's own page would send it through the proxy; `from: null` leaves Origin out */
async function call(method, url, { body, cookie, ip = '10.9.9.9', from = `https://${SITE}`, host = SITE, ua = UA, headers = {} } = {}) {
  const raw = body instanceof Buffer || typeof body === 'string'
  const res = await fetch(base + url, {
    method,
    headers: {
      'X-Forwarded-Host': host,
      'X-Real-IP': ip,
      'User-Agent': ua,
      ...(from ? { Origin: from } : {}),
      ...(cookie ? { Cookie: cookie } : {}),
      ...(body !== undefined && !raw ? { 'Content-Type': 'application/json' } : {}),
      ...headers,
    },
    body: body === undefined ? undefined : raw ? body : JSON.stringify(body),
  })
  const text = await res.text()
  let data = null
  try {
    data = JSON.parse(text)
  } catch {
    /* not JSON (an image, or nothing) */
  }
  return { status: res.status, data, text, headers: res.headers }
}
async function signIn(ip = freshIp()) {
  const r = await call('POST', '/api/session', { body: { passphrase: PASS }, ip })
  expect(r.status).toBe(200)
  return r.headers.get('set-cookie').split(';')[0]
}
const item = (over = {}) => ({ url: 'https://example.org/a', cat: 'study', kind: 'web', title: { en: 'A page', zh: '学习资源' }, desc: { en: 'About it', zh: '说明' }, ...over })

beforeAll(async () => {
  dataDir = mkdtempSync(path.join(tmpdir(), 'knowcs-admin-test-'))
  base = await start()
})
afterAll(async () => {
  await stop()
  rmSync(dataDir, { recursive: true, force: true })
})

describe('reading', () => {
  it('gives the list to anybody, starting from the seed', async () => {
    const r = await call('GET', '/api/resources', { from: null })
    expect(r.status).toBe(200)
    expect(r.data.resources.length).toBeGreaterThan(0)
    expect(r.headers.get('x-content-type-options')).toBe('nosniff')
    expect(r.headers.get('cache-control')).toBe('no-store')
  })
  it('says nobody is signed in without a cookie, or with one that was made up or has run out', async () => {
    expect((await call('GET', '/api/session')).data).toEqual({ authenticated: false })
    const cookie = await signIn()
    expect((await call('GET', '/api/session', { cookie })).data).toEqual({ authenticated: true })
    const [expires, signature] = cookie.split('=')[1].split('.')
    expect((await call('GET', '/api/session', { cookie: `knowcs_admin=${Number(expires) + 1}.${signature}` })).data.authenticated).toBe(false)
    expect((await call('GET', '/api/session', { cookie: `knowcs_admin=${Date.now() + 1e6}.${'0'.repeat(64)}` })).data.authenticated).toBe(false)
    expect((await call('GET', '/api/session', { cookie: 'knowcs_admin=1.' })).data.authenticated).toBe(false)
  })
  it('has no file outside its media folder', async () => {
    const key = readFileSync(path.join(dataDir, 'session.key'))
    for (const url of ['/api/media/../session.key', '/api/media/%2e%2e%2fsession.key', '/api/media/..%2fsession.key', '/api/media/session.key', '/api/media/../resources.json', '/api/session.key']) {
      const res = await fetch(base + url)
      expect([url, res.status === 404 || res.status === 403]).toEqual([url, true])
      expect(Buffer.from(await res.arrayBuffer()).includes(key)).toBe(false)
    }
    expect((await call('GET', '/api/media/00000000-0000-0000-0000-000000000000.png')).status).toBe(404)
  })
})

describe('signing in', () => {
  it('refuses a wrong passphrase, and a sixth try within the minute', async () => {
    const ip = freshIp()
    for (let i = 0; i < 5; i++) {
      const r = await call('POST', '/api/session', { body: { passphrase: `guess ${i}` }, ip })
      expect(r.status).toBe(401)
      expect(r.data.error).toBe('wrong_passphrase')
      expect(r.headers.get('set-cookie')).toBeNull()
    }
    const sixth = await call('POST', '/api/session', { body: { passphrase: PASS }, ip })
    expect(sixth.status).toBe(429)
    expect(Number(sixth.headers.get('retry-after'))).toBeGreaterThan(0)
    // another address is not held up by it
    expect((await call('POST', '/api/session', { body: { passphrase: PASS }, ip: freshIp() })).status).toBe(200)
  })
  it('hands out a cookie scripts cannot read and other sites cannot send', async () => {
    const r = await call('POST', '/api/session', { body: { passphrase: PASS }, ip: freshIp() })
    const cookie = r.headers.get('set-cookie')
    expect(cookie).toMatch(/^knowcs_admin=\d+\.[0-9a-f]{64};/)
    for (const flag of ['HttpOnly', 'SameSite=Strict', 'Path=/api', 'Max-Age=43200', 'Secure']) expect(cookie).toContain(flag)
    expect(r.text).not.toContain(PASS)
  })
  it('does not take a passphrase that only starts right, is empty, or is not text', async () => {
    for (const passphrase of [PASS.slice(0, -1), `${PASS} `, '', null, 0, [PASS], { a: 1 }]) {
      expect((await call('POST', '/api/session', { body: { passphrase }, ip: freshIp() })).status).toBe(401)
    }
  })
  it('signs out by clearing the cookie', async () => {
    const r = await call('DELETE', '/api/session')
    expect(r.status).toBe(200)
    expect(r.headers.get('set-cookie')).toContain('Max-Age=0')
  })
})

describe('who may change things', () => {
  it('only a page of the same site', async () => {
    const cookie = await signIn()
    const tries = [
      { from: null },
      { from: 'https://evil.example' },
      { from: 'null' },
      { from: 'not a url' },
      { from: `https://${SITE}.evil.example` },
      { headers: { 'Sec-Fetch-Site': 'cross-site' } },
    ]
    for (const t of tries) {
      const r = await call('POST', '/api/resources', { body: item(), cookie, ...t })
      expect([t, r.status, r.data.error]).toEqual([t, 403, 'wrong_origin'])
    }
    expect((await call('POST', '/api/session', { body: { passphrase: PASS }, from: 'https://evil.example', ip: freshIp() })).status).toBe(403)
  })
  it('only somebody signed in', async () => {
    const id = (await call('GET', '/api/resources')).data.resources[0].id
    const before = (await call('GET', '/api/resources')).text
    const tries = [
      ['POST', '/api/resources', item()],
      ['PATCH', `/api/resources/${id}`, item()],
      ['DELETE', `/api/resources/${id}`],
      ['PATCH', '/api/resources/order', { ids: [], baseIds: [] }],
      ['PUT', `/api/resources/${id}/preview`, PNG],
      ['DELETE', `/api/resources/${id}/preview`],
    ]
    for (const [method, url, body] of tries) {
      const r = await call(method, url, { body })
      expect([method, url, r.status, r.data.error]).toEqual([method, url, 401, 'signed_out'])
    }
    expect((await call('GET', '/api/stats')).status).toBe(401)
    expect((await call('GET', '/api/resources')).text).toBe(before)
  })
})

describe('editing the list', () => {
  let cookie
  beforeAll(async () => {
    cookie = await signIn()
  })

  it('adds at the top, with the Traditional Chinese text made from the Simplified', async () => {
    const r = await call('POST', '/api/resources', { body: item({ title: { en: '  A page  ', zh: '学习资源' } }), cookie })
    expect(r.status).toBe(201)
    const added = r.data.resources[0]
    expect(added).toMatchObject({ url: 'https://example.org/a', cat: 'study', kind: 'web', preview: null })
    expect(added.title).toEqual({ en: 'A page', zh: '学习资源', 'zh-HK': '學習資源' })
    expect(added.id).toMatch(/^[0-9a-f-]{36}$/)
    // on disk, with the version before kept beside it
    const disk = JSON.parse(readFileSync(path.join(dataDir, 'resources.json'), 'utf8'))
    expect(disk.resources[0].id).toBe(added.id)
    expect(existsSync(path.join(dataDir, 'resources.json.backup'))).toBe(true)
  })
  it('keeps only the fields it knows, cut to their lengths', async () => {
    const r = await call('POST', '/api/resources', { body: { ...item({ title: { en: 'x'.repeat(500), zh: '' }, desc: { en: 'y'.repeat(5000), zh: '' } }), id: 'mine', preview: '/api/media/x.png', extra: '<script>' }, cookie })
    const added = r.data.resources[0]
    expect(Object.keys(added).sort()).toEqual(['cat', 'desc', 'id', 'kind', 'preview', 'title', 'url'])
    expect([added.id === 'mine', added.preview, added.title.en.length, added.desc.en.length, added.title['zh-HK']]).toEqual([false, null, 120, 600, ''])
  })
  it('refuses what is not a resource', async () => {
    const bad = [
      [item({ url: 'javascript:alert(1)' }), 'bad_url'],
      [item({ url: 'https://example.org/a b' }), 'bad_url'],
      [item({ url: '' }), 'bad_url'],
      [item({ url: `https://example.org/${'a'.repeat(3000)}` }), 'bad_url'],
      [item({ cat: 'nope' }), 'bad_category'],
      [item({ kind: '__proto__' }), 'bad_category'],
      [item({ title: { en: ' ', zh: '' } }), 'no_title'],
      [item({ title: null }), 'no_title'],
      [{}, 'bad_url'],
    ]
    const before = (await call('GET', '/api/resources')).text
    for (const [body, error] of bad) {
      const r = await call('POST', '/api/resources', { body, cookie })
      expect([r.status, r.data.error]).toEqual([400, error])
    }
    expect((await call('GET', '/api/resources')).text).toBe(before)
  })
  it('answers 400 to a body that is not a JSON object, and 413 to one too large', async () => {
    for (const body of ['{"url":', 'null', '[1]', '"text"', '7']) {
      const r = await call('POST', '/api/resources', { body, cookie, headers: { 'Content-Type': 'application/json' } })
      expect([body, r.status, r.data.error]).toEqual([body, 400, 'bad_json'])
    }
    const big = await call('POST', '/api/resources', { body: item({ desc: { en: 'z'.repeat(40_000), zh: '' } }), cookie })
    expect([big.status, big.data.error]).toEqual([413, 'too_large'])
  })
  it('edits one, keeping its place and its picture', async () => {
    const list = (await call('GET', '/api/resources')).data.resources
    const target = list[1]
    const r = await call('PATCH', `/api/resources/${target.id}`, { body: item({ url: 'https://example.org/edited', cat: 'code' }), cookie })
    expect(r.status).toBe(200)
    expect(r.data.resources[1]).toMatchObject({ id: target.id, url: 'https://example.org/edited', cat: 'code', preview: target.preview })
    expect((await call('PATCH', '/api/resources/00000000-0000-0000-0000-000000000000', { body: item(), cookie })).status).toBe(404)
  })
  it('reorders only from the order the editor saw, and only into a rearrangement of it', async () => {
    const ids = (await call('GET', '/api/resources')).data.resources.map((r) => r.id)
    const swapped = [ids[1], ids[0], ...ids.slice(2)]
    const ok = await call('PATCH', '/api/resources/order', { body: { ids: swapped, baseIds: ids }, cookie })
    expect(ok.data.resources.map((r) => r.id)).toEqual(swapped)
    const stale = await call('PATCH', '/api/resources/order', { body: { ids, baseIds: ids }, cookie })
    expect([stale.status, stale.data.error]).toEqual([409, 'changed_elsewhere'])
    for (const next of [swapped.slice(1), [...swapped.slice(1), swapped[1]], [...swapped.slice(1), 'someone-else'], 'abc', null]) {
      const r = await call('PATCH', '/api/resources/order', { body: { ids: next, baseIds: swapped }, cookie })
      expect([r.status, r.data.error]).toEqual([400, 'bad_order'])
    }
    expect((await call('GET', '/api/resources')).data.resources.map((r) => r.id)).toEqual(swapped)
  })
  it('takes a picture only if it is one, serves it, and removes the file with it', async () => {
    const id = (await call('GET', '/api/resources')).data.resources[0].id
    const refused = [
      [Buffer.from('<script>alert(1)</script>'), 400, 'bad_image'],
      [Buffer.from('<svg xmlns="http://www.w3.org/2000/svg"/>'), 400, 'bad_image'],
      [Buffer.from('RIFF\0\0\0\0AVI LIST'), 400, 'bad_image'],
      [Buffer.alloc(0), 400, 'bad_image'],
      [Buffer.concat([PNG, Buffer.alloc(3 * 1024 * 1024)]), 413, 'too_large'],
    ]
    for (const [body, status, error] of refused) {
      const r = await call('PUT', `/api/resources/${id}/preview`, { body, cookie })
      expect([r.status, r.data?.error]).toEqual([status, error])
    }
    expect(readdirSync(path.join(dataDir, 'media'))).toEqual([])

    const put = await call('PUT', `/api/resources/${id}/preview`, { body: PNG, cookie })
    const url = put.data.resources[0].preview
    expect(url).toMatch(/^\/api\/media\/[0-9a-f-]{36}\.png$/)
    const got = await fetch(base + url)
    expect([got.status, got.headers.get('content-type'), got.headers.get('x-content-type-options')]).toEqual([200, 'image/png', 'nosniff'])
    expect(Buffer.from(await got.arrayBuffer()).equals(PNG)).toBe(true)

    // a new picture replaces the old file; removing it leaves none
    const again = await call('PUT', `/api/resources/${id}/preview`, { body: Buffer.concat([Buffer.from('RIFF\0\0\0\0WEBP'), Buffer.alloc(8)]), cookie })
    expect(again.data.resources[0].preview).toMatch(/\.webp$/)
    expect(readdirSync(path.join(dataDir, 'media')).length).toBe(1)
    expect((await fetch(base + url)).status).toBe(404)
    const gone = await call('DELETE', `/api/resources/${id}/preview`, { cookie })
    expect(gone.data.resources[0].preview).toBeNull()
    expect(readdirSync(path.join(dataDir, 'media'))).toEqual([])
    expect((await call('PUT', '/api/resources/00000000-0000-0000-0000-000000000000/preview', { body: PNG, cookie })).status).toBe(404)
  })
  it('deletes one, with its picture', async () => {
    const id = (await call('GET', '/api/resources')).data.resources[0].id
    await call('PUT', `/api/resources/${id}/preview`, { body: PNG, cookie })
    const r = await call('DELETE', `/api/resources/${id}`, { cookie })
    expect(r.status).toBe(200)
    expect(r.data.resources.some((x) => x.id === id)).toBe(false)
    expect(readdirSync(path.join(dataDir, 'media'))).toEqual([])
    expect((await call('DELETE', `/api/resources/${id}`, { cookie })).status).toBe(404)
  })
  it('answers 404 and 405 to what it does not have', async () => {
    expect((await call('POST', '/api/nothing', { cookie })).status).toBe(404)
    expect((await call('PUT', '/api/resources', { body: item(), cookie })).status).toBe(405)
    expect((await call('POST', '/api/resources/00000000-0000-0000-0000-000000000000', { body: item(), cookie })).status).toBe(405)
  })
  it('keeps every change when many arrive together', async () => {
    const before = (await call('GET', '/api/resources')).data.resources.length
    const results = await Promise.all(Array.from({ length: 12 }, (_, i) => call('POST', '/api/resources', { body: item({ url: `https://example.org/many/${i}` }), cookie })))
    expect(results.every((r) => r.status === 201)).toBe(true)
    const after = (await call('GET', '/api/resources')).data.resources
    expect(after.length).toBe(before + 12)
    expect(new Set(after.map((r) => r.id)).size).toBe(after.length)
    expect(JSON.parse(readFileSync(path.join(dataDir, 'resources.json'), 'utf8')).resources.length).toBe(before + 12)
  })
})

describe('visits', () => {
  let cookie
  const stats = async (host = SITE) => (await call('GET', '/api/stats', { cookie, host, from: null })).data
  const day = async () => {
    const s = await stats()
    return s.days[s.today] ?? { views: 0, visitors: 0, pages: {}, refs: {}, langs: {} }
  }
  const hit = (body, opts = {}) => call('POST', '/api/hit', { body, ...opts })
  beforeAll(async () => {
    cookie = await signIn()
  })

  it('counts a page view, and a visitor once a day', async () => {
    const ip = freshIp()
    const before = await day()
    expect((await hit({ p: '/knn/', r: 'www.google.com', l: 'zh' }, { ip })).status).toBe(204)
    await hit({ p: '/knn/', r: '', l: 'zh' }, { ip })
    await hit({ p: '/', r: '', l: 'en' }, { ip })
    await hit({ p: '/', r: 'github.com', l: 'zh-HK' }, { ip: freshIp() })
    const after = await day()
    expect(after.views - before.views).toBe(4)
    expect(after.visitors - before.visitors).toBe(2)
    expect(after.pages['/knn/']).toBe(2)
    expect(after.pages['/']).toBe(2)
    expect(after.refs).toMatchObject({ 'www.google.com': 1, 'github.com': 1 })
    expect(after.langs).toMatchObject({ zh: 2, en: 1, 'zh-HK': 1 })
    expect((await stats()).counted).toBe(true)
  })
  it('counts the same address with another browser as another visitor', async () => {
    const ip = freshIp()
    const before = await day()
    await hit({ p: '/', l: 'en' }, { ip })
    await hit({ p: '/', l: 'en' }, { ip, ua: `${UA} Edg/130` })
    expect((await day()).visitors - before.visitors).toBe(2)
  })
  it('leaves out what is not a visit to a page of the site', async () => {
    const before = JSON.stringify(await day())
    const ignored = [
      [{ p: '/knn/', l: 'zh' }, { ua: 'Googlebot/2.1 (+http://www.google.com/bot.html)' }],
      [{ p: '/knn/', l: 'zh' }, { ua: 'Mozilla/5.0 HeadlessChrome/130.0' }],
      [{ p: '/knn/', l: 'zh' }, { ua: 'curl/8.7.1' }],
      [{ p: '/knn/', l: 'zh' }, { ua: '' }],
      [{ p: 'knn', l: 'zh' }, {}],
      [{ p: '/knn', l: 'zh' }, {}],
      [{ p: '/KNN/', l: 'zh' }, {}],
      [{ p: '/../etc/passwd/', l: 'zh' }, {}],
      [{ p: '/<script>/', l: 'zh' }, {}],
      [{ p: `/${'a'.repeat(100)}/`, l: 'zh' }, {}],
      [{ p: 5, l: 'zh' }, {}],
      [{}, {}],
    ]
    for (const [body, opts] of ignored) expect((await hit(body, { ip: freshIp(), ...opts })).status).toBe(204)
    expect(JSON.stringify(await day())).toBe(before)
    // a view with a referrer or language it does not accept still counts, without them
    await hit({ p: '/mnist/', r: 'a site with spaces', l: 'xx' }, { ip: freshIp() })
    await hit({ p: '/mnist/', r: SITE, l: ['zh'] }, { ip: freshIp() })
    await hit({ p: '/mnist/', r: '__proto__', l: '__proto__' }, { ip: freshIp() })
    const after = await day()
    expect(after.pages['/mnist/']).toBe(3)
    expect([after.refs, after.langs].map((o) => JSON.stringify(o))).toEqual([JSON.parse(before).refs, JSON.parse(before).langs].map((o) => JSON.stringify(o)))
  })
  it('takes them only from the site itself, as a small JSON object', async () => {
    const before = JSON.stringify(await day())
    expect((await hit({ p: '/' }, { from: null })).status).toBe(403)
    expect((await hit({ p: '/' }, { from: 'https://evil.example' })).status).toBe(403)
    expect((await hit('null', { headers: { 'Content-Type': 'application/json' } })).status).toBe(400)
    expect((await hit({ p: '/', pad: 'x'.repeat(2000) })).status).toBe(413)
    expect(JSON.stringify(await day())).toBe(before)
  })
  it('keeps each site apart, and counts only the sites it was told to', async () => {
    const other = 'somewhere-else.example'
    expect((await hit({ p: '/', l: 'en' }, { host: other, from: `https://${other}`, ip: freshIp() })).status).toBe(204)
    const s = await stats(other)
    expect([s.host, s.counted, s.days]).toEqual([other, false, {}])
  })
  it('stops counting one address after 60 in a minute', async () => {
    const ip = freshIp()
    const before = await day()
    for (let i = 0; i < 70; i++) await hit({ p: '/otsu/', l: 'en' }, { ip })
    expect((await day()).views - before.views).toBe(60)
  })
  it('counts at most 300 different pages a day, the rest together', async () => {
    for (let i = 0; i < 320; i++) await hit({ p: `/made-up-${i}/`, l: 'en' }, { ip: freshIp() })
    const pages = (await day()).pages
    expect(Object.keys(pages).length).toBeLessThanOrEqual(301)
    expect(pages['(other)']).toBeGreaterThan(0)
  })
  it('gives only the days asked for', async () => {
    const s = await call('GET', '/api/stats?days=1', { cookie, from: null })
    expect(Object.keys(s.data.days)).toEqual([s.data.today])
    expect(s.data.today).toMatch(/^\d{4}-\d\d-\d\d$/)
    for (const days of ['0', '-5', 'abc', '99999', '1.5']) expect((await call('GET', `/api/stats?days=${days}`, { cookie, from: null })).status).toBe(200)
  })
})

describe('restarting', () => {
  it('writes the counts on the way out, with no address or browser in them, and reads everything back', async () => {
    const cookie = await signIn()
    const list = (await call('GET', '/api/resources')).text
    const before = (await call('GET', '/api/stats', { cookie, from: null })).data
    await stop()
    const file = readFileSync(path.join(dataDir, 'stats.json'), 'utf8')
    expect(JSON.parse(file)[SITE][before.today].views).toBe(before.days[before.today].views)
    expect(file).not.toMatch(/10\.\d+\.\d+\.\d+|Mozilla|Safari/)
    expect(readdirSync(dataDir).sort()).toEqual(['media', 'resources.json', 'resources.json.backup', 'session.key', 'stats.json'])

    base = await start()
    expect((await call('GET', '/api/resources')).text).toBe(list)
    // the key that signs cookies is kept, so the editor stays signed in
    const after = (await call('GET', '/api/stats', { cookie, from: null })).data
    expect(after.days).toEqual(before.days)
  })
})
